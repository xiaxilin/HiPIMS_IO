#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
To demonstrate how to generate input data for HiPIMS for both single and multiple GPUs
Created on Sun Dec  2 21:40:54 2018

@author: Xiaodong MIng
"""
import os
import sys
import numpy as np
import time
scriptsPath = '/Users/ming/Dropbox/Python/HiPIMS' # position storing HiPIMS_IO.py and ArcGridDataProcessing.py
sys.path.insert(0,scriptsPath)

import hipims_case_class as hp

# define root path for the example case
rootPath='/Users/ming/Dropbox/Python/CaseP'

# read example DEM data
demFile = 'ExampleDEM.asc'
#demMat,demHead,demExtent = AP.arcgridread('ExampleDEM.asc') # stored in the same dir with HiPIMS_IO.py

HP_obj = hp.InputHipims(dem_file=demFile,num_of_sections=3)
#%% define boundary condition
bound1Points = np.array([[535, 206], [545, 206], [545, 210], [535, 210]])*1000
bound2Points = np.array([[520, 230], [530, 230], [530, 235], [520, 235]])*1000
#dBound0 = {'polyPoints': [],'type': 'open','h': [],'hU': []}
dBound1 = {'polyPoints': bound1Points,'type': 'open','h': [[0,10],[60,10]]}
dBound2 = {'polyPoints': bound2Points,'type': 'open','hU': [[0,50000],[60,30000]]}
boundList = [dBound1,dBound2]
del dBound1,dBound2,bound1Points,bound2Points
HP_obj.set_boundary_condition(boundList)
dem_array = HP_obj.Raster.array+0
dem_array[np.isnan(dem_array)]=500
h0 = dem_array*0
h0[dem_array<50]=1
HP_obj.set_parameter('h0',h0)
#HP_obj.Sections[1].Summary.display()
HP_obj.Summary.display()
#output_list = HP_obj.write_input_files('boundary_condition')

#%% show one section
#for obj_section in HP_obj.Sections:
#
#    grid_values = obj_section.Raster.array
#    
#    #% show boundary cells
#    subs = obj_section.bound.boundSubs
#    value = 1
#    grid_values = grid_values+np.nan
#    for onebound in subs:
#        grid_values[onebound]=value
#        value = value+1
#    obj_section.DEM.array=grid_values
#    obj_section.DEM.Mapshow(vmin=1,vmax=value-1)
##%% 
#grid_values = HP_obj.DEM.array
#subs = HP_obj.bound.boundSubs
#grid_values = grid_values*0+np.nan
#value=1
#for onebound in subs:
#    grid_values[onebound]=value
#    value = value+1
#HP_obj.DEM.array=grid_values
#HP_obj.DEM.Mapshow()
#%% define rainfall source, a same rainfall source for the whole model domain
# the rainfall mask is default defined as 0 for all the domain cells
rain_source = np.array([[0,100/1000/3600/24],
                        [86400,100/1000/3600/24],
                        [86401,0]])
# define monitor positions
gauges_pos = np.array([[534.5,231.3],
                       [510.2,224.5],
                       [542.5,225.0],
                       [538.2,212.5],
                       [530.3,219.4]])*1000
HP_obj.set_gauges_position(gauges_pos)
HP_obj.set_rainfall(rain_source=rain_source)

# generate input files for HiPIMS
start = time.perf_counter()
HP_obj.write_input_files() # write all input files
end = time.perf_counter()
print('total time elapse: '+str(end-start))
HP_obj.Summary.display()
HP_obj.save_object('hp_obj')