#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------
hipims_post_process
Manage output files of a hipims flood model
-------------------------------------------------------------------------------
Created on Tue Nov 26 00:07:48 2019

@author: Xiaodong Ming
"""

import numpy as np
#import pandas as pd
#import shutil
#import glob
#import gzip
import os
from myclass import Raster
#%% Combine Grid files from Multiple GPU outputs:
def combine_grid_file(root_path, num_section, file_tag, delete=False):
    """
    obj_raster = combine_grid_file(root_path, num_section, file_tag)
    Input:
    root_path: directory of case input files containing domain ID 0,1,...folder
    num_section: number of subsection domains
    file_tag: h_TTTT, h_max_TTTT_max
    Return:
    """
    overlay_rows = 4
    if 'DEM' in file_tag:
        file_tail = '/input/mesh/DEM.txt'
    else:
        if file_tag.endswith('.asc'):
            file_tail = '/output/' + file_tag
        else:
            file_tail = '/output/' + file_tag + '.asc'
    if not root_path.endswith('/'):
        root_path = root_path+'/'
    # read the bottom sub grid file
    file_name = root_path+str(0)+file_tail
    obj_bottom = Raster(file_name)
    header_global = obj_bottom.header
    for i in range(num_section): 
        file_name = root_path+str(i)+file_tail
        if i==0:
            array_global = obj_bottom.array
        else:
            
            obj_section = Raster(file_name)
            array_local = obj_section.array
            array_global = np.r_[array_local[0:-overlay_rows, :], array_global]
        if delete:   
            os.remove(file_name)
    row,col = array_global.shape
    header_global['ncols'] = col
    header_global['nrows'] = row
    raster_global = Raster(array=array_global, header=header_global)
    return raster_global