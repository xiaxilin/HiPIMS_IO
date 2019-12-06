#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 13:03:25 2019

@author: Xiaodong Ming
"""
import os
import sys
import copy
import gzip
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.colors import LightSource
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import interpolate
#%% *******************************To deal with raster data********************
#   ***************************************************************************    
class Raster(object):
    """
    Created on Tue Apr 7 2019
    @author: Xiaodong Ming 
    
    To deal with raster data with a ESRI ASCII or GTiff format  
    
    Properties:
        source_file: file name to read grid data
        output_file: file name to write a raster object
        array: a numpy array storing grid cell values
        header: a dict storing reference information of the grid
        extent: a tuple storing outline limits of the raster (left, right, bottom, top)
        extent_dict: a dictionary storing outline limits of the raster
        projection: (string) the Well-Known_Text (wkt) projection information
    
    Methods(public):
        Write_asc: write grid data into an asc file with or without 
            compression(.gz)
        To_osgeo_raster: convert this object to an osgeo raster object
        RectClip: clip raster according to a rectangle extent
        Clip: clip raster according to a polygon
        Rasterize: rasterize a shapefile on the Raster object and return a 
            bool array with 'Ture' in and on the polygon/polyline
        Resample: resample the raster to a new cellsize
        GetXYcoordinate: Get X and Y coordinates of all raster cells
        GetGeoTransform: Get GeoTransform tuple for osgeo raster dataset
        Mapshow: draw a map of the raster dataset
        VelocityShow: draw velocity vectors as arrows with values on two Raster
            datasets (u, v)
        
    
    Methods(private):
        __header2extent: convert header to extent
        __read_asc: read an asc file ends with .asc or .gz
        __map2sub: convert map coordinates of points to subscripts of a matrix
            with a reference header
        __sub2map: convert subscripts of a matrix to map coordinates
        __read_tif: read tiff file
        
    """

#%%======================== initialization function ===========================   
    def __init__(self, source_file=None, array=None, header=None, epsg=None, projection=None):
        """
        source_file: name of a asc/tif file if a file read is needed
        array: values in each raster cell [a numpy array]
        header: georeference of the raster [a dictionary containing 6 keys]:
            nrows, nclos [int]
            cellsize, xllcorner, yllcorner
            NODATA_value
        epsg: epsg code [int]
        projection: WktProjection [string]
        """
        self.source_file = source_file
        self.projection = projection
        if epsg is not None:
            self.projection = self.__SetWktProjection(epsg)
        else:
            self.projection = None
        if source_file is None:
            self.array = array
            self.header = header
            self.source_file = 'source_file.asc'
        elif type(source_file) is str:
            if os.path.exists(source_file):
                if source_file.endswith('.tif'):
                    self.__read_tif() # only read the first band
                else:
                    self.__read_asc()
            else:
                raise IOError 
                sys.exit(1)
        else:  #try a binary file-like object
            self.__read_bytes()
#        else:
#            raise ValueError('source file format is supported')

        if isinstance(self.header, dict)==0:
            raise ValueError('header is not a dictionary')
        else:
            # create self.extent and self.extent_dict 
            self.__header2extent()
            
#%%============================= Spatial analyst ==============================   
    def RectClip(self, clipExtent):
        """
        clipExtent: left, right, bottom, top
        clip raster according to a rectangle extent
        return:
           a new raster object
        """
        new_obj = copy.deepcopy(self)
        X = clipExtent[0:2]
        Y = clipExtent[2:4]
        rows, cols = self.__map2sub(X, Y)
        Xcentre, Ycentre = self.__sub2map(rows, cols)
        xllcorner = min(Xcentre)-0.5*self.header['cellsize']
        yllcorner = min(Ycentre)-0.5*self.header['cellsize']
        # new array
        new_obj.array = self.array[min(rows):max(rows), min(cols):max(cols)]
        # new header
        new_obj.header['nrows'] = new_obj.array.shape[0]
        new_obj.header['ncols'] = new_obj.array.shape[1]
        new_obj.header['xllcorner'] = xllcorner
        new_obj.header['yllcorner'] = yllcorner
        # new extent
        new_obj.__header2extent()
        new_obj.source_file = None       
        return new_obj
    
    def Clip(self, mask=None):
        """
        clip raster according to a mask
        mask: 
            1. string name of a shapefile
            2. numpy vector giving X and Y coords of the mask points
        
        return:
            a new raster object
        """
        from osgeo import ogr
        if isinstance(mask, str):
            shpName =  mask
        # Open shapefile datasets        
        shpDriver = ogr.GetDriverByName('ESRI Shapefile')
        shpDataset = shpDriver.Open(shpName, 0) # 0=Read-only, 1=Read-Write
        layer = shpDataset.GetLayer()
        shpExtent = np.array(layer.GetExtent()) #(minX, maxY, maxX, minY)           
        # 1. rectangle clip raster
        new_obj = self.RectClip(shpExtent)
        new_raster = copy.deepcopy(new_obj)                
        indexArray = new_raster.Rasterize(shpDataset)
        arrayClip = new_raster.array
        arrayClip[indexArray==0]=new_raster.header['NODATA_value']
        new_raster.array = arrayClip        
        shpDataset.Destroy()
        return new_raster
    
    def Rasterize(self, shpDSName, rasterDS=None):
        """
        rasterize the shapefile to the raster object and return a bool array
            with Ture value in and on the polygon/polyline
        shpDSName: string for shapefilename, dataset for ogr('ESRI Shapefile') object
        
        return numpy array
        """
        from osgeo import gdal, ogr
        if isinstance(shpDSName, str):
            shpDataset = ogr.Open(shpDSName)
        else:
            shpDataset = shpDSName
        layer = shpDataset.GetLayer()
        if rasterDS is None:
            target_ds = self.To_osgeo_raster()
        else:
            target_ds = rasterDS
        gdal.RasterizeLayer(target_ds, [1], layer, burn_values=[-1])
        rasterized_array = target_ds.ReadAsArray()
        indexArray = np.full(rasterized_array.shape, False)
        indexArray[rasterized_array==1] = True
        target_ds=None
        return indexArray
    
    def Resample(self,newCellsize,method='bilinear'):
        """
        resample the raster to a new cellsize
        newCellsize: cellsize of the new raster
        method: Resampling method to use. Available methods are:
            near: nearest neighbour resampling (default, fastest algorithm, worst interpolation quality).        
            bilinear: bilinear resampling.        
            cubic: cubic resampling.        
            cubicspline: cubic spline resampling.        
            lanczos: Lanczos windowed sinc resampling.        
            average: average resampling, computes the average of all 
                    non-NODATA contributing pixels.        
            mode: mode resampling, selects the value which appears most often 
                    of all the sampled points.        
            max: maximum resampling, selects the maximum value from all 
                    non-NODATA contributing pixels.        
            min: minimum resampling, selects the minimum value from all 
                    non-NODATA contributing pixels.        
            med: median resampling, selects the median value of all 
                    non-NODATA contributing pixels.        
            q1: first quartile resampling, selects the first quartile 
                value of all non-NODATA contributing pixels.        
            q3: third quartile resampling, selects the third quartile 
                value of all non-NODATA contributing pixels
        """
        cellSize = self.header['cellsize']
        rasterXSize = self.header['ncols']
        newRasterXSize = int(rasterXSize*cellSize/newCellsize)
        rasterYSize = self.header['nrows']
        newRasterYSize = int(rasterYSize*cellSize/newCellsize)
        
        from osgeo import gdal
        g = self.To_osgeo_raster() # get original gdal dataset
        total_obs = g.RasterCount
        drv = gdal.GetDriverByName( "MEM" )
        dst_ds = drv.Create('', g.RasterXSize, g.RasterYSize,1,eType=gdal.GDT_Float32)
        dst_ds.SetGeoTransform( g.GetGeoTransform())
        dst_ds.SetProjection ( g.GetProjectionRef() )
        hires_data = self.array
        dst_ds.GetRasterBand(1).WriteArray ( hires_data )
        
        geoT = g.GetGeoTransform()
        drv = gdal.GetDriverByName( "MEM" )
        resampled_ds = drv.Create('',newRasterXSize, newRasterYSize, 1, eType=gdal.GDT_Float32)

        newGeoT = (geoT[0], newCellsize, geoT[2],
                   geoT[3], geoT[3], -newCellsize)
        resampled_ds.SetGeoTransform(newGeoT )
        resampled_ds.SetProjection (g.GetProjectionRef() )
        resampled_ds.SetMetadata ({"TotalNObs":"%d" % total_obs})

        gdal.RegenerateOverviews(dst_ds.GetRasterBand(1),[resampled_ds.GetRasterBand(1)], method)
    
        resampled_ds.GetRasterBand(1).SetNoDataValue(self.header['NODATA_value'])
        
        new_obj = self.__osgeoDS2raster(resampled_ds)
        resampled_ds = None

        return new_obj
    
    def Interpolate_to(self, points, values, method='nearest'):
        """ Interpolate values of 2D points to all cells on the Raster object
        2D interpolate
        points: ndarray of floats, shape (n, 2)
            Data point coordinates. Can either be an array of shape (n, 2), 
            or a tuple of ndim arrays.
        values: ndarray of float or complex, shape (n, )
            Data values.
        method: {‘linear’, ‘nearest’, ‘cubic’}, optional
            Method of interpolation.
        """
        grid_x, grid_y = self.GetXYcoordinate()
        array_interp = interpolate.griddata(points, values, (grid_x, grid_y), method=method)
        new_obj = copy.deepcopy(self)
        new_obj.array = array_interp
        new_obj.source_file = 'mask_'+new_obj.source_file
        return new_obj
    
    def GridResample(self,newsize):
        """
        resample a grid to a new grid resolution via nearest interpolation
        """
        zMat = self.array
        header = self.header
        if isinstance(newsize, dict):
            head_new = newsize.copy()
        else:            
            head_new = header.copy()
            head_new['cellsize'] = newsize
            ncols = math.floor(header['cellsize']*header['ncols']/newsize)
            nrows = math.floor(header['cellsize']*header['nrows']/newsize)
            head_new['ncols']=ncols
            head_new['nrows']=nrows
        #centre of the first cell in zMat
        x11 = head_new['xllcorner']+0.5*head_new['cellsize']
        y11 = head_new['yllcorner']+(head_new['nrows']-0.5)*head_new['cellsize']
        xAll = np.linspace(x11,x11+(head_new['ncols']-1)*head_new['cellsize'],head_new['ncols'])
        yAll = np.linspace(y11,y11-(head_new['nrows']-1)*head_new['cellsize'],head_new['nrows'])
        rowAll,colAll = self.__map2sub(xAll,yAll)
        rows_Z,cols_Z = np.meshgrid(rowAll,colAll) # nrows*ncols array
        zNew = zMat[rows_Z,cols_Z]
        zNew = zNew.transpose()
        zNew = zNew.astype(zMat.dtype)
#        extent_new = demHead2Extent(head_new)
        new_obj = Raster(array=zNew, header=head_new)
        return new_obj 
         
#%%=============================Visualization==================================
    #%% draw inundation map with domain outline
    def Mapshow(self, figname=None, figsize=None, dpi=300, vmin=None, vmax=None, 
                cax=True, dem_array=None, relocate=False, scale_ratio=1):
        """
        Display raster data without projection
        figname: the file name to export map, if figname is empty, then
            the figure will not be saved
        figsize: the size of map
        dpi: The resolution in dots per inch
        vmin and vmax define the data range that the colormap covers
        """
        np.warnings.filterwarnings('ignore')    
        fig, ax = plt.subplots(1, figsize=figsize)
        # draw grid data
        if dem_array is  None:
            dem_array = self.array+0
        dem_array[dem_array==self.header['NODATA_value']]=np.nan
        # adjust tick label and axis label
        map_extent = self.extent
        map_extent = _adjust_map_extent(map_extent, relocate, scale_ratio)
        img=plt.imshow(dem_array, extent=map_extent, vmin=vmin, vmax=vmax)
        # colorbar
    	# create an axes on the right side of ax. The width of cax will be 5%
        # of ax and the padding between cax and ax will be fixed at 0.05 inch.
        if cax==True:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(img, cax=cax)
        ax.axes.grid(linestyle='-.', linewidth=0.2)
        
        # save figure
        if figname is not None:
            fig.savefig(figname, dpi=dpi)
            
        return fig, ax
    
    def rank_show(self,figname=None, figsize=None, dpi=300, 
                breaks=[0.2, 0.3, 0.5, 1, 2], 
                show_colorbar=True, show_colorlegend=False,
                dem_array=None, relocate=False, scale_ratio=1):
        """ Display water depth map in a range defined by (d_min, d_max)
        """
        np.warnings.filterwarnings('ignore')    
        if breaks[0] > np.nanmin(self.array):
            breaks.insert(0, np.nanmin(self.array))
        if breaks[-1] < np.nanmax(self.array):
            breaks.append(np.nanmax(self.array))        
        norm = colors.BoundaryNorm(breaks, len(breaks))
        blues = cm.get_cmap('Blues', norm.N)
        newcolors = blues(np.linspace(0, 1, norm.N))
        white = np.array([255/256, 255/256, 255/256, 1])
        newcolors[0, :] = white
        newcmp = ListedColormap(newcolors)
        map_extent = self.extent
        map_extent = _adjust_map_extent(map_extent, relocate, scale_ratio)
#        cellsize = self.header['cellsize']
        if dem_array is not None:
            array = dem_array+0
            array[np.isnan(array)] = np.nanmin(self.array)
            ls = LightSource(azdeg=315, altdeg=45)
            cmap = plt.cm.gist_gray
            fig, ax = plt.subplots(figsize=figsize)
            rgb = ls.shade(array, cmap=cmap,
                           blend_mode='overlay',vert_exag=5)
            ax.imshow(rgb, extent=map_extent)
#            ax.set_axis_off()
        else:
            fig, ax = plt.subplots(figsize=figsize)

        chm_plot = ax.imshow(self.array, extent=map_extent, 
                             cmap=newcmp, norm=norm, alpha=0.7)
        # create colorbar
        if show_colorbar is True:
            _set_colorbar(ax, chm_plot, norm)
        if show_colorlegend is True: # legend
            _set_color_legend(ax, norm, newcmp)
#        plt.show()
        # save figure
        if figname is not None:
            fig.savefig(figname, dpi=dpi)
        return fig, ax
    
    def hillshade_show(self,figsize=None,azdeg=315, altdeg=45, vert_exag=1):
        """ Draw a hillshade map
        """
        array = self.array+0
        array[np.isnan(array)]=0
        ls = LightSource(azdeg=azdeg, altdeg=altdeg)
        cmap = plt.cm.gist_earth
        fig, ax = plt.subplots(figsize=figsize)
        rgb = ls.shade(array, cmap=cmap, 
                       blend_mode='overlay',vert_exag=vert_exag)
        ax.imshow(rgb)
        ax.set_axis_off()
#        plt.show()
        return fig, ax

    #%% draw velocity (vector) map
    def VelocityShow(self, other, figname=None, figsize=None, dpi=300, **kw):
        """
        plot velocity map of U and V, whose values stored in two raster
        objects seperately
        """
        X, Y = self.GetXYcoordinate()        
        U = self.array
        V = other.array
        if U.shape!=V.shape:
            raise TypeError('bad argument: header')
        if 'figsize' in kw:
            figsize = kw['figsize']
        else:
            figsize = None
        fig, ax = plt.subplots(1, figsize=figsize)
        plt.quiver(X, Y, U, V)
        ax.set_aspect('equal', 'box')
        ax.tick_params(axis='y', labelrotation=90)
        if figname is not None:
            fig.savefig(figname, dpi=dpi)
        return fig, ax
#%%===============================output=======================================
    
    def GetXYcoordinate(self):
        """ Get X and Y coordinates of all raster cells
        return xv, yv numpy array with the same size of the raster object
        """
        ny, nx = self.array.shape
        cellsize = self.header['cellsize']
        # coordinate of the centre on the top-left pixel
        x00centre = self.extent_dict['left'] + cellsize/2
        y00centre = self.extent_dict['top'] - cellsize/2
        x = np.arange(x00centre, x00centre+cellsize*nx, cellsize)
        y = np.arange(y00centre, y00centre-cellsize*ny, -cellsize)
        xv, yv = np.meshgrid(x, y)
        return xv, yv
    
    def GetGeoTransform(self):
        """
        get GeoTransform tuple for osgeo raster dataset
        """
        GeoTransform = (self.extent_dict['left'], self.header['cellsize'], 0.0, 
                        self.extent_dict['top'], 0.0, -self.header['cellsize'])
        return GeoTransform
    
    def Write_asc(self, output_file, EPSG=None, compression=False):
        
        """
        write raster as asc format file 
        output_file: output file name
        EPSG: epsg code, if it is given, a .prj file will be written
        compression: logic, whether compress write the asc file as gz
        """
        if compression:
            if not output_file.endswith('.gz'):
                output_file=output_file+'.gz'        
        self.output_file = output_file
        array = self.array+0
        header = self.header
        array[np.isnan(array)]= header['NODATA_value']
        if not isinstance(header, dict):
            raise TypeError('bad argument: header')
                     
        if output_file.endswith('.gz'):
            f = gzip.open(output_file, 'wb') # write compressed file
        else:
            f = open(output_file, 'wb')
        f.write(b"ncols    %d\n" % header['ncols'])
        f.write(b"nrows    %d\n" % header['nrows'])
        f.write(b"xllcorner    %g\n" % header['xllcorner'])
        f.write(b"yllcorner    %g\n" % header['yllcorner'])
        f.write(b"cellsize    %g\n" % header['cellsize'])
        f.write(b"NODATA_value    %g\n" % header['NODATA_value'])
        np.savetxt(f, array, fmt='%g', delimiter=' ')
        f.close()
        if EPSG is not None:
            self.__SetWktProjection(EPSG)
        # if projection is defined, write .prj file for asc file

        if output_file.endswith('.asc'):
            if self.projection is not None:
                prj_file=output_file[0:-4]+'.prj'
                wkt = self.projection
                with open(prj_file, "w") as prj:        
                    prj.write(wkt)
        return None
    
    # convert this object to an osgeo raster object
    def To_osgeo_raster(self, filename=None, fileformat = 'GTiff', destEPSG=27700):        
        """
        convert this object to an osgeo raster object, write a tif file if 
            necessary
        filename: the output file name, if it is given, a tif file will be produced
        fileformat: GTiff or AAIGrid
        destEPSG: the EPSG projection code default: British National Grid
        
        return:
            an osgeo raster dataset
            or a tif filename if it is written
        """
        from osgeo import gdal, osr
        if filename is None:
            dst_filename = ''
            driverName = 'MEM'
        else:
            dst_filename = filename
            driverName = fileformat
        if not dst_filename.endswith('.tif'):
            dst_filename = dst_filename+'.tif'
    
        # You need to get those values like you did.
        PIXEL_SIZE = self.header['cellsize']  # size of the pixel...        
        x_min = self.extent[0] # left  
        y_max = self.extent[3] # top
        dest_crs = osr.SpatialReference()
        dest_crs.ImportFromEPSG(destEPSG)
        # create dataset with driver
        driver = gdal.GetDriverByName(driverName)
        ncols = self.header['ncols']
        nrows = self.header['nrows']
#        print('ncols:', type(ncols), ' - nrows:'+type(nrows))
        dataset = driver.Create(dst_filename, 
            xsize=ncols, 
            ysize=nrows, 
            bands=1, 
            eType=gdal.GDT_Float32)
    
        dataset.SetGeoTransform((
            x_min,    # 0
            PIXEL_SIZE,  # 1
            0,  # 2
            y_max,    # 3
            0,                      # 4
            -PIXEL_SIZE))  
    
        dataset.SetProjection(dest_crs.ExportToWkt())
        array = self.array
#        array[array==self.header['NODATA_value']]=np.nan
        dataset.GetRasterBand(1).WriteArray(array)
        dataset.GetRasterBand(1).SetNoDataValue(self.header['NODATA_value'])
        if filename is not None:
            dataset.FlushCache()  # Write to disk.
            dataset = None
            return dst_filename
        else:
            return dataset

#%%=========================== private functions ==============================
    def __osgeoDS2raster(self, ds):
        """
        convert an osgeo dataset to a raster object
        """
        array = ds.ReadAsArray()
        geoT = ds.GetGeoTransform()
        projection = ds.GetProjection()
        left = geoT[0]
        top = geoT[3]
        cellsize = geoT[1]
        nrows = ds.RasterYSize
        ncols = ds.RasterXSize
        xllcorner = left
        yllcorner = top - cellsize*nrows
        NODATA_value = ds.GetRasterBand(1).GetNoDataValue()
        if NODATA_value is None:
            NODATA_value = -9999
        header = {'ncols':ncols, 'nrows':nrows,
                  'xllcorner':xllcorner, 'yllcorner':yllcorner,                  
                  'cellsize':cellsize, 'NODATA_value':NODATA_value}
        newObj = Raster(array=array, header=header, projection=projection)
        return newObj
        
    def __header2extent(self):
        """
        To convert header (dict) to a spatial extent of the DEM
        extent: (left, right, bottom, top)
        """
        R = self.header
        left = R['xllcorner']
        right = R['xllcorner']+R['ncols']*R['cellsize']
        bottom = R['yllcorner']
        top = R['yllcorner']+R['nrows']*R['cellsize']
        self.extent = (left, right, bottom, top)
        self.extent_dict = {'left':left, 'right':right, 'bottom':bottom, 'top':top}

    def __map2sub(self, X, Y):
        """
        convert map points to subscripts of a matrix with geo reference header
        X, Y: coordinates in map units
        return
            rows, cols: (int) subscripts of the data matrix
        """
        #x and y coordinate of the centre of the first cell in the matrix
        X = np.array(X)
        Y = np.array(Y)
        header = self.header
        
        x0 = header['xllcorner']+0.5*header['cellsize']
        y0 = header['yllcorner']+(header['nrows']-0.5)*header['cellsize']
        rows = (y0-Y)/header['cellsize'] # row and col number starts from 0
        cols = (X-x0)/header['cellsize']
        if isinstance(rows, np.ndarray):
            rows = rows.astype('int64')
            cols = cols.astype('int64') #.astype('int64')
        else:
            rows = int(rows)
            cols = int(cols)
        return rows, cols

    def __sub2map(self, rows, cols):
        """
        convert subscripts of a matrix to map coordinates 
        rows, cols: subscripts of the data matrix, starting from 0
        return
            X, Y: coordinates in map units
        """
        #x and y coordinate of the centre of the first cell in the matrix
        if not isinstance(rows, np.ndarray):
            rows = np.array(rows)
            cols = np.array(cols)        
        
        header = self.header
        left = self.extent[0] #(left, right, bottom, top)
        top = self.extent[3]
        X = left + (cols+0.5)*header['cellsize']
        Y = top  - (rows+0.5)*header['cellsize']
         
        return X, Y

# read ascii file        
    def __read_asc(self):
        """
        read asc file and return array, header
        if self.source_file ends with '.gz', then read the compressed file
        """
        fileName = self.source_file
        try:
            fh = open(fileName, 'r')
            fh.close()
        # Store configuration file values
        except FileNotFoundError:
            # Keep preset values
            print('Error: '+fileName+' does not appear to exist')
            return
        # read header
        header = {} # store header information including ncols, nrows, ...
        numheaderrows = 6
        n=1
        if fileName.endswith('.gz'):
            # read header
            with gzip.open(fileName, 'rt') as f:                
                for line in f:
                    if n<=numheaderrows:
                        line = line.split(" ", 1)
                        header[line[0]] = float(line[1])
                    else:
                        break
                    n = n+1
        else:
            # read header
            with open(fileName, 'rt') as f:            
                for line in f:
                    if n<=numheaderrows:
                        line = line.split(" ", 1)
                        header[line[0]] = float(line[1])
                    else:
                        break
                    n = n+1
    # read value array
        array  = np.loadtxt(fileName, skiprows=numheaderrows, dtype='float64')
        array[array == header['NODATA_value']] = float('nan')
        header['ncols']=int(header['ncols'])
        header['nrows']=int(header['nrows'])
        self.array = array
        self.header = header
        prjFile = self.source_file[:-4]+'.prj'
        if os.path.isfile(prjFile):
            with open(prjFile, 'r') as file:
                projection = file.read()
            self.projection = projection
        return None

# read GTiff file
    def __read_tif(self):
        """
        read tif file and return array, header
        only read the first band
        """
        from osgeo import gdal
        tifName = self.source_file
        ds = gdal.Open(tifName)        
        ncols = ds.RasterXSize
        nrows = ds.RasterYSize
        geoTransform = ds.GetGeoTransform()
        x_min = geoTransform[0]
        cellsize = geoTransform[1]
        y_max = geoTransform[3]
        xllcorner = x_min
        yllcorner = y_max - nrows*cellsize
        rasterBand = ds.GetRasterBand(1)
        NODATA_value = rasterBand.GetNoDataValue()
        array = rasterBand.ReadAsArray()
        header = {'ncols':ncols, 'nrows':nrows, 
                  'xllcorner':xllcorner, 'yllcorner':yllcorner, 
                  'cellsize':cellsize, 'NODATA_value':NODATA_value}        
        if not np.isscalar(header['NODATA_value']):
            header['NODATA_value'] = -9999
        array[array == header['NODATA_value']] = float('nan')
        self.header = header
        self.array = array
        self.projection = ds.GetProjection()
        rasterBand = None
        ds = None
        return None

    def __read_bytes(self):
        """ Read file from a bytes object
        """
        f = self.source_file
        # read header
        header = {} # store header information including ncols, nrows, ...
        numheaderrows = 6
        for _ in range(numheaderrows):
            line = f.readline()
            line = line.strip().decode("utf-8").split(" ", 1)
            header[line[0]] = float(line[1])
            # read value array
        array  = np.loadtxt(f, skiprows=numheaderrows, dtype='float64')
        array[array == header['NODATA_value']] = float('nan')
        header['ncols'] = int(header['ncols'])
        header['nrows'] = int(header['nrows'])
        self.array = array
        self.header = header

    def __SetWktProjection(self, epsg_code):
        """
        get coordinate reference system (crs) as Well Known Text (WKT) 
            from https://epsg.io
        epsg_code: the epsg code of a crs, e.g. BNG:27700, WGS84:4326
        return wkt text
        """
        import requests
        # access projection information
        wkt = requests.get('https://epsg.io/{0}.prettywkt/'.format(epsg_code))
        # remove spaces between charachters
        remove_spaces = wkt.text.replace(" ", "")
        # place all the text on one line
        output = remove_spaces.replace("\n", "")
        self.projection = output
        return output

#%% shapePoints= makeDiagonalShape(extent)
def makeDiagonalShape(extent):
    #extent = (left, right, bottom, top)
    shapePoints = np.array([[extent[0], extent[2]], 
                           [extent[1], extent[2]], 
                           [extent[1], extent[3]], 
                           [extent[0], extent[3]]])
    return shapePoints

#%% convert header data to extent
def demHead2Extent(demHead):
    # convert dem header file (dict) to a spatial extent of the DEM
    R = demHead
    left = R['xllcorner']
    right = R['xllcorner']+R['ncols']*R['cellsize']
    bottom = R['yllcorner']
    top = R['yllcorner']+R['nrows']*R['cellsize']
    extent = (left, right, bottom, top)
    return extent         

def _set_colorbar(ax,img,norm):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(img, cax=cax)
    y_tick_values = cax.get_yticks()
    boundary_means = [np.mean((y_tick_values[ii],y_tick_values[ii-1])) 
                        for ii in range(1, len(y_tick_values))]
    category_names = [(str(norm.boundaries[ii-1])+'~'+
                       str(norm.boundaries[ii]))
                      for ii in range(1, len(norm.boundaries))]
    category_names[0] = '<='+str(norm.boundaries[1])
    category_names[-1] = '>'+str(norm.boundaries[-2])
    cax.yaxis.set_ticks(boundary_means)
    cax.yaxis.set_ticklabels(category_names,rotation=0)
    return cax

def _set_color_legend(ax, norm, cmp, 
                      loc='lower right', bbox_to_anchor=(1,0),
                      facecolor=None):
    category_names = [(str(norm.boundaries[ii-1])+'~'+
                       str(norm.boundaries[ii]))
                      for ii in range(1, len(norm.boundaries))]
    category_names[0] = '<='+str(norm.boundaries[1])
    category_names[-1] = '>'+str(norm.boundaries[-2])
    ii = 0
    legend_labels = {}
    for category_name in category_names:
        legend_labels[category_name] = cmp.colors[ii,]
        ii = ii+1
    patches = [Patch(color=color, label=label)
               for label, color in legend_labels.items()]
    ax.legend(handles=patches, loc=loc,
              bbox_to_anchor=bbox_to_anchor,
              facecolor=facecolor)
    return ax

def _adjust_map_extent(extent, relocate=True, scale_ratio=1):
    """
    Adjust the extent (left, right, bottom, top) to a new staring point 
        and new unit. extent values will be divided by the scale_ratio
    Example:
        if scale_ratio = 1000, and the original extent unit is meter,
        then the unit is converted to km, and the extent is divided by 1000
    """
    if relocate:
        left = 0 
        right = (extent[1]-extent[0])/scale_ratio
        bottom = 0
        top = (extent[3]-extent[2])/scale_ratio
    else:
        left = extent[0]/scale_ratio
        right = extent[1]/scale_ratio
        bottom = extent[2]/scale_ratio
        top = extent[3]/scale_ratio
    return (left, right, bottom, top)
    
