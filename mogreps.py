#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 11:30:15 2019

@author: Xiaodong Ming
"""
import pickle
import iris
import warnings
import gzip
import numpy as np
import pandas as pd
import datetime
import glob
from pyproj import Proj, transform
import sys
# position storing HiPIMS_IO.py and ArcGridDataProcessing.py
scriptsPath = '/home/cvxx/HiPIMS/scripts/HiPIMS_IO' 
sys.path.insert(0,scriptsPath)
scriptsPath = '/Users/b4042552/Dropbox/Python/HiPIMS' 
sys.path.insert(0,scriptsPath)
#%% grid data for HiPIMS input format
class MOGREPS_data(object):
    """
    read MOGREPS pp file and save selected data as an object
    
    Properties:
        ppFileName: the name of a MOGREPS pp file
        
    methods(private): 
    """

    
    def __init__(self,ppFileName,varName='stratiform_rainfall_flux'):
        if ppFileName.endswith('.pp'):
            self.ppFileName=ppFileName
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cubes = iris.load(ppFileName,varName)
            cube = cubes[0]
            self.name = cube.name()
            self.data = cube.data
            self.units = cube.units.symbol
            cube_time = cube.coord('time')
#            self.time = cube_time.core_points()
            self.time_units = cube_time.units.name
            gridCoord = cube.coord('grid_longitude')
#            self.grid_longitude = gridCoord.core_points()
            self.grid_units = gridCoord.units.name
            self.grid_coord_system = str(gridCoord.coord_system)
            all_coords= cube.coords()
            attributs = {}
            for coord in all_coords:
                attributs[coord.standard_name]=coord.points
            self.attributs = attributs
            
    def Save_object(self, filename=None):
        if filename is None:
            filename = self.ppFileName[:-3]+'.gz'
        with gzip.open(filename, 'wb') as output:  # Overwrites any existing file.
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)
        # write the object to a file
        
    def Read_object(filename):
        if filename.endswith('.gz'):
            with gzip.open(filename, 'rb') as f:
                output_obj = pickle.load(f)
        else:
            with open(filename, 'rb') as f:
                output_obj = pickle.load(f)
                
        return output_obj
    
    def Create_rain_mask(self,demFile=None,mask_values=None):
        """
        # create and return an rainfall mask object of raster
        # demFile: the DEM file name
        # mask_values: an array with the same size of DEM. if it is not given
                an int sequency value starting from 0 will be given to each
                mask cell. 
        """
        from myclass import raster
        x,y = self.Coords_transform()

           
        mask_resolution = np.max([x[0,1]-x[0,0],x[1,0]-x[0,0]]).round(1)
#        print(mask_resolution)
        mask_resolution = np.absolute(mask_resolution)
        if demFile is not None:
            # read a raster from a file
            demRaster = raster(demFile)
            
            mask_left = demRaster.extent_dict['left']-mask_resolution/2
            mask_right = demRaster.extent_dict['right']+mask_resolution/2
            mask_bottom = demRaster.extent_dict['bottom']-mask_resolution/2
            mask_top = demRaster.extent_dict['top']+mask_resolution/2
            # cut rainfall array according to mask extent
            ind1 = x>=mask_left
            ind2 = x<=mask_right
            ind3 = y>=mask_bottom
            ind4 = y<=mask_top
            #index of the rainfall data points inside the DEM
            indArray = ind1 & ind2 & ind3 & ind4 
            
            points = np.c_[x[indArray],y[indArray]]
    #        values = mask_array
            
        else:
            # create a raster object 
            # header
            demHeader = {}
            demHeader['ncols'] = x.shape[1]
            demHeader['nrows'] = x.shape[0]
            demHeader['xllcorner'] = x.min().round(2)-mask_resolution/2
            demHeader['yllcorner'] = y.min().round(2)-mask_resolution/2            
            demHeader['cellsize'] = mask_resolution
            demHeader['NODATA_value'] = -9999
            demArray = np.zeros((demHeader['nrows'],demHeader['ncols']))
#            demArray = demArray.astype('float')
            
            demRaster = raster(array=demArray,header=demHeader,epsg=27700)
            points = np.c_[x.flatten(),y.flatten()]
            values = np.arange(x.size)
            indArray = demArray==0
                    
        if mask_values is not None:
            values = mask_values.flatten()
        else:
            values = np.arange(points.shape[0])
            
        mask_array=demRaster.Interpolate_to(points,values)
        del demRaster
        
        return mask_array, indArray#points,values#
    
    def Export_rain_source_array(self,demFile=None):
        """
        Export rainfall source array and timeArray
        
        """
        #3D array: [layer,row,col]
        _,indArray = self.Create_rain_mask(demFile)
        
        #2D array: row--time, col--value
        rain_source_array = self.data[:,indArray]
        #original date
        t_origin = self.time_units
        t_origin = datetime.datetime.strptime(t_origin,'hour since %Y-%m-%d %H:%M:%S.%f0 UTC')
        ref_time = t_origin+datetime.timedelta(self.attributs['forecast_reference_time'][0]/24)
        
        dt_series = t_origin+pd.to_timedelta(self.attributs['time'],unit='hour').round('s')
        time_delta_s = dt_series - ref_time
        time_delta_s = np.array(time_delta_s.total_seconds()).round()

#        file_dt_str = self.ppFileName.split('_')
#        hour_ref = int(file_dt_str[5][0:3])
#        t0 = datetime.datetime.strptime(file_dt_str[3],'%Y%m%d')+datetime.timedelta(int(hour_ref)/24)
        return rain_source_array,time_delta_s,ref_time
    
    def Coords_transform(self, SP_coor=[177.5-180, -37.5], option=2):
        """
        convert rotated lon/lat degrees to regular lon/lat degrees
        and then transformed to projected coordinates X,Y
        for pp/nc file degree values
        based on the Matlab version from Simon Funder
        SP_coor: south pole longitude and latitude, it can be transformed from
        north pole lon/lat as well. SP_coor = [lon_NP-180,-lat_NP]
        
        """
        inProj = Proj(init='epsg:4326') # WGS 84
        outProj = Proj(init='epsg:27700') # British National Grid
        lat = self.attributs['grid_latitude']
        lon = self.attributs['grid_longitude']
        lon2,lat2 = np.meshgrid(lon,lat)
    
        
        pi = np.pi
        lon = (lon2.flatten()*pi)/180 # Convert degrees to radians
        lat = (lat2.flatten()*pi)/180
    
        SP_lon = SP_coor[0]
        SP_lat = SP_coor[1]
    
        theta = 90+SP_lat # Rotation around y-axis
        phi = SP_lon # Rotation around z-axis
    
        phi = (phi*pi)/180 # Convert degrees to radians
        theta = (theta*pi)/180
    
        x = np.cos(lon)*np.cos(lat) # Convert from spherical to cartesian coordinates
        y = np.sin(lon)*np.cos(lat)
        z = np.sin(lat)
    
        if option == 1: # Regular -> Rotated
    
            x_new = np.cos(theta)*np.cos(phi)*x + np.cos(theta)*np.sin(phi)*y + np.sin(theta)*z
            y_new = -np.sin(phi)*x + np.cos(phi)*y
            z_new = -np.sin(theta)*np.cos(phi)*x - np.sin(theta)*np.sin(phi)*y + np.cos(theta)*z
    
        elif option == 2: # Rotated -> Regular
            phi = -phi
            theta = -theta
    
            x_new = np.cos(theta)*np.cos(phi)*x + np.sin(phi)*y + np.sin(theta)*np.cos(phi)*z
            y_new = -np.cos(theta)*np.sin(phi)*x + np.cos(phi)*y - np.sin(theta)*np.sin(phi)*z
            z_new = -np.sin(theta)*x + np.cos(theta)*z
    
        lon_new = np.arctan2(y_new,x_new) # Convert cartesian back to spherical coordinates
        lat_new = np.arcsin(z_new)
        
        lon_new = (lon_new*180)/pi # Convert radians back to degrees
        lat_new = (lat_new*180)/pi
        
    #    grid_out = [lon_new, lat_new]
        x,y = transform(inProj,outProj,lon_new,lat_new)
        x = x.reshape(lon2.shape)
        y = y.reshape(lat2.shape)
        return x,y

def WriteRainSourceArray(datetimeStr,realization,demFile=None,filetail=None):
    """
    datetimeStr: yyyymmdd_HH [20190617_02]
    realization: 3-element string '006'
    """
    # read mogreps object
    namestr = datetimeStr+'_'+realization
    filenames = glob.glob('*'+namestr+'*.gz')
    filenames.sort()
    rain_source_array = []
    time_delta_s = []
    
    for gzfile in filenames:
        obj = MOGREPS_data.Read_object(gzfile)
        A,B,ref_time = obj.Export_rain_source_array(demFile)
        rain_source_array.append(A)
        time_delta_s.append(B.flatten())
#        print(gzfile)
    rain_source_array = np.vstack(rain_source_array)
    time_delta_s = np.hstack(time_delta_s)
    time_delta_s = time_delta_s.reshape((time_delta_s.size,1))
    outputArray = np.hstack([time_delta_s,rain_source_array])
    if filetail is not None:
        namestr = namestr+'_'+filetail
    np.savetxt(namestr+'.txt',outputArray,fmt='%g')
    return None



    