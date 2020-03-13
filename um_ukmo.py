#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
um_ukmo.py
Extract data from UK Met Office UM pp format files.

Package requirements:
    # conda install -c conda-forge iris
    # conda install -c conda-forge/label/gcc7 mo_pack
    # conda install -c conda-forge gdal
Created on Tue Aug 20 11:30:15 2019

command line call:
    python um_ukmo.py input_file output_file
        input_file: a string file name or a string with '*' 
@author: Xiaodong Ming
"""
import sys
import pickle
import iris
import warnings
import gzip
import copy
import imageio
import os
import glob
import gc
import numpy as np
import pandas as pd
import grid_show as gs
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import iris.plot as iplt
from myclass import Raster

#%% grid data for HiPIMS input format
class UM_ukmo(object):
    """
    read UM pp file and save selected data as an object
    Properties:
        ppFileName: the name of a UM pp file        
    methods(private): 
    """    
    def __init__(self, file_name, var_name='stratiform_rainfall_flux'):
        """Read UKMO UM pp files
        
        file_name: one pp file name or a list of pp file names
        var_name: variable to be read
        Return:
            time: a datetime series
            data: 3d array, each layer corresponds to a datetime
            coordinates: of grid cells    
            attributes dict: 'time', 'grid_latitude', 'grid_longitude', 
            'forecast_reference_time', 'forecast_period'
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cube_list = iris.cube.CubeList()
        if type(file_name) is not list:
            if '*' in file_name:
                file_names = glob.glob(file_name)
            else:
                file_names = [file_name]
        else:
            file_names = file_name
        print(file_names)
        for one_file in file_names:
            cubes = iris.load(one_file, var_name)
            cube = cubes[0]
            cube_list.append(cube)
        cube = cube_list.concatenate()[0]
        # data
        self.cube = cube
        coord_names = []
        for coord in cube.coords():
            coord_names.append(coord.standard_name)
        # general information
        self.coord_names = coord_names
        summary = {}
        summary['file_name'] = file_names # pp file name
        summary['var_name'] = cube.name()
        summary['data_units'] = cube.units.symbol
        time_units_str = cube.coord('time').units.name.split()
        summary['time_units'] = time_units_str[0]
        self.summary = summary
        # forecast_reference_time
        forecast_reference_time = cube.coord('forecast_reference_time')
        ref_time_num = forecast_reference_time.points[0]
        time_units = forecast_reference_time.units
        self.forecast_ref_datetime = time_units.num2date(ref_time_num)
        # forecast datetime
        time_units = cube.coord('time').units
        time_num = cube.coord('time').points
        self.date_time = pd.to_datetime(time_units.num2date(time_num))
        self.time_from_ref = time_num-ref_time_num # use time_units(hours)
        self.shape = self.cube.shape

    def save_object(self, filename=None):
        """
        Save the objec as a gz file
        """
        if filename is None:
            filename = self.ppFileName[:-3]+'.gz'
        with gzip.open(filename, 'wb') as output:  # Overwrites existing files
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)
        print(filename+' saved')
    
    def save_cube(self, filename=None, append=False):
        """
        Save the cube to a pp file
        append: if True, appending to the contents of an existing the file
        """
        if filename is None:
            filename = self.ppFileName[:-3]+'.pp'
        iris.save(self.cube, filename, append=append)
        print(filename+' saved')    
    
    def get_rain_mask_source(self, dem_file=None, cellsize=1500):
        """get rainfall mask and source file based on the dem_file
        dem_file: (str) the DEM file name or a Raster object
        mask_values: an array with the same size of DEM. if it is not given
                an int sequency value starting from 0 will be given to each
                mask cell.
        cellsize: 1500 m for UKV, 2200 m for MOGREPS
        Return:
            mask_obj: a Raster object provide the mask value array and its header
            rain_source: (numpy array) rainfall source array
                The 1st column is usually time series in seconds, from the 2nd 
                column towards end columns are rainfall rate in m/s
        """
        # cut data grid according to the DEM extent 
        if not hasattr(self, 'header'):
            new_obj = self.regrid2osgb(dem_file, cellsize)
        else:
            new_obj = self
        data = new_obj.cube.data
        x_meter = new_obj.cube.coord('longitude').points
        y_meter = new_obj.cube.coord('latitude').points
        x_meter, y_meter = np.meshgrid(x_meter, y_meter)
        source_array = data.reshape(data.shape[0], -1, order='F')
        mask_values = np.arange(0, x_meter.size)
        mask_values = mask_values.reshape(x_meter.shape, order='F')
        # read a raster from a file
        if type(dem_file) is str:
            dem_obj = Raster(dem_file)
        elif dem_file is None:
            shape = (new_obj.header['nrows'], new_obj.header['ncols'])
            dem_obj = Raster(array=np.zeros(shape), header=new_obj.header)
        else:
            dem_obj = dem_file
        points = np.c_[x_meter.flatten(), y_meter.flatten()]
        values = mask_values.flatten()
        mask_obj = dem_obj.Interpolate_to(points, values)
        time_series = new_obj.time_from_ref*60*60 # hour to seconds
        source_array = source_array/1000 # m-2.kg.s-1 to m/s
        rain_source = np.c_[time_series, source_array]
        return rain_source, mask_obj
    
    def regrid2osgb(self, new_grid=None, cellsize=1500):
        """ Regird the object to a new grid
        new_grid: an Raster object with array and header or a raster file name
            if new_grid is None, then a Raster object will be created based on
            the original data grid
        cellsize: cellsize in the new object
        Return:
            new_obj: a new UM_ukmo object
        """
        if new_grid is None:
            x_meter, y_meter = self.get_osgb_coordiantes()
            x_left = np.nanmin(x_meter)
            x_right = np.nanmax(x_meter)
            y_bottom = np.nanmin(y_meter)
            y_top = np.nanmax(y_meter)
            cellsize0 = (x_right-x_left)/x_meter.shape[0]
            cellsize1 = (y_top-y_bottom)/x_meter.shape[1]
            cellsize = round((cellsize0+cellsize1)/2)
            extent = (x_left, x_right, y_bottom, y_top)
        else:
            if type(new_grid) is str:
                grid_obj = Raster(new_grid)
            else:
                grid_obj = new_grid
            extent = grid_obj.extent
        header = {}
        header['cellsize'] = cellsize
        header['NODATA_value'] = -9999
        ncols = np.ceil((extent[1]-extent[0])/cellsize).astype('int64')
        nrows = np.ceil((extent[3]-extent[2])/cellsize).astype('int64')
        header['xllcorner'] = extent[0]
        header['yllcorner'] = extent[2]
        header['nrows'] = nrows
        header['ncols'] = ncols
        x_coords = np.arange(extent[0], extent[0]+ncols*cellsize, cellsize)
        y_coords = np.arange(extent[2], extent[2]+nrows*cellsize, cellsize)
        cube_new = _regrid_cube(self.cube, x_coords, y_coords)
        new_obj = copy.deepcopy(self)
        new_obj.cube = cube_new
        new_obj.header = header
        return new_obj
    
    def get_osgb_coordiantes(self):
        """Convert roatated pole coordiantes to OSGB coordinates
        Return:
            x_meter: gridded X coordinates in OSGB CRS
            y_meter: gridded Y coordinates in OSGB CRS
            the grid has the same shape with the rainfall data array 
        """
        crs_osgb = iris.coord_systems.OSGB
        crs_cube = self.cube.coord(standard_name='grid_longitude').coord_system
        crtp_osgb = crs_osgb.as_cartopy_crs(crs_osgb)
        crtp_cube = crs_cube.as_cartopy_crs()
        rotate_lon = self.cube.coord(standard_name='grid_longitude').points
        rotate_lat = self.cube.coord(standard_name='grid_latitude').points
        grid_lon, grid_lat = np.meshgrid(rotate_lon,rotate_lat)
        x_meter = grid_lon*0
        y_meter = grid_lat*0
        for ind2d, ind1d in np.ndenumerate(grid_lon):
            x0, y0 = crtp_osgb.transform_point(grid_lon[ind2d], 
                                               grid_lat[ind2d], crtp_cube)
            x_meter[ind2d] = x0
            y_meter[ind2d] = y0
        return x_meter, y_meter
#============================Visulization======================================    
    def mapshow(self, layer_index, title_str=None, shape_file=None,
                figsize=None, cmap='YlGnBu', **kwarg):
        """Plot the grided value of rainfall
        layer_index: (int) the number reprenting time layers
        projected: True to show km grid, False to show lan/lon grid
        """
        if not hasattr(self, 'header'):
            # use iris to plot map
            data = copy.deepcopy(self.cube[layer_index])
            data.data = data.data*3600 # to mm/h
            data.data[data.data == 0] = np.nan
            fig = plt.figure(figsize=figsize)
            map_prj = ccrs.PlateCarree()
            ax = plt.axes(projection=map_prj)            
            ax.coastlines(linewidth=0.5, color='black', resolution='50m')
            im = iplt.pcolormesh(data, cmap=cmap, **kwarg)
            cax = fig.add_axes()
            fig.colorbar(im, cax=cax, orientation='vertical')
#            plt.axis('on')
            ax.gridlines(crs=map_prj, linestyle='--')
            xticks = ax.get_xticks()
            ax.set_xticks(xticks)
            yticks = ax.get_yticks()
            ax.set_yticks(yticks)
            ax.set_xlabel('longitude (degree)')
            ax.set_ylabel('latitude (degree)')           
        else:
            # Already projected to OSGB, use grid_show to plot map
            data = self.cube[layer_index].data
            data = data*3600
            data[data == 0] = np.nan
            header = self.header
            obj_ras = Raster(header=header, array=data)
            extent = obj_ras.extent
            if extent[1]-extent[0] > 10000:
                scale_ratio = 1000
            else:
                scale_ratio = 1
            fig, ax = gs.mapshow(obj_ras, cmap=cmap, scale_ratio=scale_ratio,
                                 **kwarg)
            obj_ras = None
            if shape_file is not None:
                gs.plot_shape_file(shape_file, ax=ax)
        if title_str is None:
            dt = self.date_time[layer_index]
            title_str = dt.strftime('%Y-%m-%d %H:%M')
            ax.set_title(title_str)
        return fig, ax
    
    def gif_generator(self, output_file, layers=None, duration=0.5, **kwarg):
        """Generate a gif file for the gridded rainfall rate values
        layers: numpy array provide the time layer numbers to be included in 
            the gif animation
        duration: duration for each frame (seconds)
        """
        if layers is None:
            layers = np.arange(self.cube.shape[0])
        fig_name_list = []
        for time_ind in layers:
            fig, ax = self.mapshow(time_ind, **kwarg)
            temp_figname = 'temp_{:04d}'.format(int(time_ind))+'.png'
            fig.savefig(temp_figname)
            plt.close(fig)
            fig_name_list.append(temp_figname)
        # create animation with the images
        images = []
        for fig_name in fig_name_list:
            images.append(imageio.imread(fig_name))
            os.remove(fig_name)
        # save animation and delete images
        if not output_file.endswith('.gif'):
            output_file = output_file+'.gif'
        imageio.mimsave(output_file, images, duration=duration)
    
    def video_generator(self, output_file, layers=None, fps=10, **kwarg):
        """Make a mp4 video to show rainfall rates
        """
        if layers is None:
            layers = np.arange(self.cube.shape[0])
#        fig_name_list = []
        i_seq = 0
#        import time
        for time_ind in layers: 
            temp_figname = 'temp_{:04d}'.format(int(time_ind))+'.png'
            fig, _ = self.mapshow(time_ind, **kwarg)
            fig.savefig(temp_figname)
            plt.close(fig)
#            fig_name_list.append(temp_figname)
            print(temp_figname)
            i_seq = i_seq+1
            if i_seq%10 == 0:
                gc.collect()
                gc.garbage
#                time.sleep(2)
                
        if not output_file.endswith('.mp4'):
            output_file = output_file+'.mp4'
        print('creating '+output_file+'...')
        fig_name_list = glob.glob('temp_*.png')
        fig_name_list.sort()
        writer = imageio.get_writer(output_file, 'MP4', fps=fps)
        for fig_name in fig_name_list:
            writer.append_data(imageio.imread(fig_name))
            os.remove(fig_name)
        writer.close()
        print('Done!')

def load_object(filename):
    """
    Read gz file
    """
    if filename.endswith('.gz'):
        with gzip.open(filename, 'rb') as f:
            output_obj = pickle.load(f)
    else:
        with open(filename, 'rb') as f:
            output_obj = pickle.load(f)                
    return output_obj

def _regrid_cube(cube_origin, x_coords, y_coords):
    """ Regrid cube to a new grid
    cube_origin: a cube read from pp file
    x_coords: numpy array of x coordinates
    y_coords: numpy array of y coordinates
    return:
        a new cube based on the coordinates provided
    """
    import iris.coord_systems as icoord_systems
    from iris.coords import DimCoord
    from iris.cube import Cube
    bng_crs = icoord_systems.OSGB()
    x_coords = np.sort(x_coords) # sort ascend
    y_coords = np.sort(y_coords)[::-1] # sort descend
    x_lons = DimCoord(x_coords, standard_name='longitude',
                      coord_system=bng_crs, units='metres')
    y_lats = DimCoord(y_coords, standard_name='latitude',
                      coord_system=bng_crs, units='metres') 
    time_coords = cube_origin.coord('time')
    shape = (time_coords.shape[0], y_lats.shape[0], x_lons.shape[0])
    array = np.zeros(shape)
    new_cube = Cube(array, dim_coords_and_dims=[(time_coords, 0), 
                                                (y_lats, 1), (x_lons, 2)])
    new_cube = cube_origin.regrid(new_cube, iris.analysis.Nearest())
    return new_cube

def _generate_raster(x_meter, y_meter):
    """
    generate a raster object from x and y gridded coordinates 
    """
    x_left = np.nanmin(x_meter)
    x_right = np.nanmax(x_meter)
    y_bottom = np.nanmin(y_meter)
    y_top = np.nanmax(y_meter)
    cellsize0 = (x_right-x_left)/x_meter.shape[0]
    cellsize1 = (y_top-y_bottom)/x_meter.shape[1]
    cellsize = round((cellsize0+cellsize1)/2)
    extent = (x_left, x_right, y_bottom, y_top)
    header = {}
    header['cellsize'] = cellsize
    header['NODATA_value'] = -9999
    ncols = np.ceil((extent[1]-extent[0])/cellsize).astype('int64')
    nrows = np.ceil((extent[3]-extent[2])/cellsize).astype('int64')
    header['xllcorner'] = extent[0]
    header['yllcorner'] = extent[2]
    header['nrows'] = nrows
    header['ncols'] = ncols
    obj_raster = Raster(array=np.zeros((nrows, ncols)), header=header)
    return obj_raster

def main():
    args = sys.argv
    if len(args)==2:
        obj_um = UM_ukmo(args[1])
    elif len(args)>2:
        obj_um = UM_ukmo(args[1:])
    else:
        obj_um = None
        raise IOError('At least one argument is required!')
    return obj_um

if __name__=='__main__':
    main()
"""
Order of grid points
Data is normally in the same order as model grid points. The values are stored row-wise i.e. data
for a complete row is followed by data from the next row. A row is normally along a line of constant
latitude. The order is indicated by the sign of header variables BDX and BDY (so a negative value
of BDY and a positive BDX indicate rows ordered north to south and points within a row ordered
west to east). Before data is plotted it is re-oriented as appropriate (so that, for a lat-long chart the
northern-most row is at the top).
"""

"""
#%%
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import iris.quickplot as qplt
osgb_xy = ccrs.OSGB()
plt.figure()
crs_latlon = ccrs.PlateCarree()
ax = plt.axes(projection=osgb_xy)
#ax.set_extent(extent, crs=osgb_xy)
#ax.set_extent((-15.0, 10.0, 45.0, 65.0), crs=crs_latlon)
ax.coastlines(linewidth=0.75, color='red')
ax.gridlines(crs=crs_latlon, linestyle='-')
iplt.pcolormesh(cube[10], cmap='RdBu_r', vmin=new_cube[10].data.min(),
                vmax=new_cube[10].data.max())

#qplt.contourf(cube, 15)
#    plt.gca().coastlines()

#ax = plt.axes(projection=projection_crs)
#iplt.pcolormesh(main_data, cmap='RdBu_r')
"""
