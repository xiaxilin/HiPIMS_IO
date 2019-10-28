#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example_class_setup
To demonstrate how to generate input data for HiPIMS using hipims_case_class 
Created on Sun Dec  2 21:40:54 2018

@author: Xiaodong MIng
-------------------------------------------------------------------------------
Assumptions:
- Input DEM is a regular Arc-grid file
- its map unit is meter
- its cellsize is the same in both x and y direction
- its reference position is on the lower left corner of the southwest cell
- All the other grid-based input files must be consistent with the DEM file
To do:
- create an object of InputHipims for single GPU and multiple GPUs
- set boundary conditions, rainfall data, model parameters
- generate input folders and files
- save and reload InputHipims objects
"""

import sys
import numpy as np
import hipims_case_class as hp
# position storing hipims_case_class.py and myclass.py
scriptsPath = '/Users/ming/Dropbox/Python/HiPIMS' 
sys.path.insert(0,scriptsPath)

#%%========================= For single GPU ===================================
# define root path for the example case
case_folder='/Users/ming/Dropbox/Python/CaseP'
dem_data = 'ExampleDEM.asc' # provide dem data
HP_obj = hp.InputHipims(dem_data=dem_data, 
                        num_of_sections=1,
                        case_folder=case_folder)
# define initial condition
h0 = HP_obj.Raster.array+0
h0[np.isnan(h0)] = 0
h0[h0 < 50] = 0
h0[h0 >= 50] = 1
# set initial water depth (h0) and velocity (hU0x, hU0y)
HP_obj.set_parameter('h0', h0)
HP_obj.set_parameter('hU0x', h0*0.0001)
HP_obj.set_parameter('hU0y', h0*0.0002)

# define boundary condition
bound1_points = np.array([[535, 206], [545, 206], [545, 210], [535, 210]])*1000
bound2_points = np.array([[520, 230], [530, 230], [530, 235], [520, 235]])*1000
bound1_dict = {'polyPoints': bound1_points,
               'type': 'open', 'h': [[0, 10], [60, 10]]}
bound2_dict = {'polyPoints': bound2_points,
               'type': 'open', 'hU': [[0, 50000], [60, 30000]]}
bound_list = [bound1_dict,bound2_dict]
del bound1_points, bound2_points, bound1_dict, bound2_dict
HP_obj.set_boundary_condition(bound_list)

# define and set rainfall mask and source (two rainfall sources)
rain_source = np.array([[0, 100/1000/3600, 0],
                        [86400, 100/1000/3600, 0],
                        [86401, 0, 0]])
rain_mask = HP_obj.Raster.array+0
rain_mask[np.isnan(rain_mask)] = 0
rain_mask[rain_mask < 50] = 0
rain_mask[rain_mask >= 50] = 1
#HP_obj.set_parameter('precipitation_mask', rain_mask)
HP_obj.set_rainfall(rain_mask=rain_mask, rain_source=rain_source)

# define and set monitor positions
gauges_pos = np.array([[534.5, 231.3], [510.2, 224.5], [542.5, 225.0],
                       [538.2, 212.5], [530.3, 219.4]])*1000
HP_obj.set_gauges_position(gauges_pos)
# add a user-defined parameter
HP_obj.add_user_defined_parameter('new_param',0)
# write all input files
HP_obj.write_input_files() # write all input files
# show model summary
HP_obj.Summary.display()
#  save the model object as a pickle file
HP_obj.save_object(case_folder+'/hp_obj')

#%%========================= For multiple GPUs ================================
# load a model object and show its summary
HP_obj_m = hp.load_object(case_folder+'/hp_obj.pickle')

"""
Multi-GPU model object can also be created directly with the following codes:
HP_obj_m = hp.InputHipims(dem_data=dem_data,
                        num_of_sections=3,
                        case_folder=case_folder)
"""
# divide the model domain into several parts and get a list of child objects
# in HP_obj.Sections. Inherite all the single-gpu data
HP_obj_m.decomposite_domain(num_of_sections=3)
HP_obj_m.Sections
# change the case foder
HP_obj_m.set_case_folder(new_folder='/Users/ming/Dropbox/Python/CaseP/MG')
# change the rainfall data
rain_source = np.array([[0, 0, 50/1000/3600],
                        [86400, 0, 30/1000/3600],
                        [86401, 0, 0]])
HP_obj_m.set_rainfall(rain_source=rain_source)
# generate all input files
HP_obj_m.write_input_files()
# change manning parameters
HP_obj_m.set_parameter('manning',0.055)
# write manning.dat for multiple gpu
HP_obj_m.write_input_files('manning')
# show summary
HP_obj_m.Summary.display()
# save the object as a file
HP_obj_m.save_object(case_folder+'/hp_mg_obj')

#%%==================show boundary of condition cells==========================
#%show boundary of each section
for obj_section in HP_obj_m.Sections:

    grid_values = obj_section.Raster.array
    
    #% show boundary cells
    subs = obj_section.Boundary.cell_subs
    value = 1
    grid_values = grid_values+np.nan
    for onebound in subs:
        grid_values[onebound]=value
        value = value+1
    obj_section.Raster.Mapshow(vmin=1,vmax=value-1,dem_array=grid_values)
#%show boundary of all domain
grid_values = HP_obj_m.Raster.array
subs = HP_obj_m.Boundary.cell_subs
grid_values = grid_values*0+np.nan
value = 1
for onebound in subs:
    grid_values[onebound]=value
    value = value+1
HP_obj_m.Raster.Mapshow(dem_array=grid_values)


