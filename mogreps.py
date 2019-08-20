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

        