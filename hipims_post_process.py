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

#import pandas as pd
#import shutil
#import glob
#import gzip
import os
import datetime
import pickle
import gzip
import numpy as np
import pandas as pd
import spatial_analysis as sp
#import hipims_case_class
class OutputHipims:
    """To read and analyze otuput files from a HiPIMS flood model
    Properties (public):
        case_folder: (str) the absolute path of the case folder
        output_folder: (str|list of strings) the absolute path of the 
            output folder(s)
        number_of_sections: (int) the number of subdomains of the model
        header: (dict or list of dict) provide the header information
    Methods (public):
        read_gauges_file:
        read_grid_file:
        set_headers_from_output: Read header information of each model 
            domain/subdomain from asc files in the output folder
    Methods (private):
    """  
    def __init__(self, input_obj=None, case_folder=None, num_of_sections=None):
        """Initialize the object with a InputHiPIMS object or a case folder and
        the number of sections
        """
        # pass argument values
        if input_obj is None:
            self.case_folder = case_folder
            self.num_of_sections = num_of_sections
        elif hasattr(input_obj, 'case_folder'):
            case_folder = input_obj.case_folder
            self.case_folder = case_folder
            num_of_sections = input_obj.num_of_sections
            self.num_of_sections = num_of_sections
            self.header = input_obj.Raster.header
            self.Summary = input_obj.Summary
        else:
            raise IOError('The first argument (input_obj) must be',
                          'a InputHipims object')
        if self.num_of_sections == 1:
            output_folder = case_folder+'/output'
            input_folder = case_folder+'/input'
            dem_name = input_folder+'/mesh/DEM.txt'
            if os.path.exists(dem_name):
                self.header = sp.arc_header_read(dem_name)
                self.header_global = self.header
        else:
            output_folder = []
            input_folder = []
            headers = []
            for i in range(self.num_of_sections):
                output_folder.append(case_folder+'/'+str(i)+'/output')  
                input_folder.append(case_folder+'/'+str(i)+'/input')
                dem_name = case_folder+'/'+str(i)+'/input/mesh/DEM.txt'
                if os.path.exists(dem_name):
                    header = sp.arc_header_read(dem_name)
                else:
                    header = None
                headers.append(header)
            self.header = headers
            if type(header) is list:
                self.header_global = _header_local2global(self.header)
        self.output_folder = output_folder
        self.input_folder = input_folder
    
    def read_gauges_file(self, file_tag='h', compressed=False):
        """ Read gauges files for time seires of values at rhe monitored gauges
        file_tag: h, hU, eta, corresponding to h_gauges.dat, hU_gauges.dat,
            and eta_gauges.dat, respectively
        Return:
            gauges_pos: the coordinates of gauges within the model domain
            time_series: time in seconds
            values: gauge values corresponding to the gauges position
        """
        if self.num_of_sections==1:
            output_folder = self.output_folder+'/'
            gauge_output_file = output_folder+file_tag+'_gauges.dat'
            gauge_pos_file = self.input_folder+'/field/gauges_pos.dat'
            if compressed:
                gauge_output_file = gauge_output_file+'.gz'
                gauge_pos_file = gauge_output_file+'.gz'
            times, values = _read_one_gauge_file(gauge_output_file)
            gauges = np.loadtxt(gauge_pos_file, dtype='float64', ndmin=2)
        else: # multi-GPU
            if not hasattr(self, 'header'):
                output_asc = input('Type an asc file name',
                                   'in output folder to read header:\n')
                self.set_headers_from_output(output_asc)
            header_list = self.header
            gauges, times, values = \
                _combine_multi_gpu_gauges_data(header_list, 
                                                self.case_folder, file_tag)
        self.times_simu = pd.DataFrame({'times':times})
        if hasattr(self, 'ref_datetime'):
            date_times = np.datetime64(self.ref_datetime)+times.astype('timedelta64[s]')
            self.times_simu['date_times'] = date_times
        return gauges, times, values
    
    def read_grid_file(self, file_tag='h_0', compressed=False):
        """Read asc grid files from output
        Return
            grid_array: a numpy array provides the cell values in grid
            header: a dict provide the header information of the grid
        """
        if not file_tag.endswith('.asc'):
            file_tag = file_tag+'.asc'
        if compressed:
            file_tag = file_tag+'.gz'
        if self.num_of_sections==1:
            file_name = self.output_folder+'/'+file_tag
            grid_array, header, _ = sp.arcgridread(file_name)
        else:
            # multi-GPU
            grid_array, header = self._combine_multi_gpu_grid_data(file_tag)        
        return grid_array, header
    
    def add_gauge_results(self, gauge_name, gauge_ind, var_name, 
                          compressed=False):
        """ add simulated value to the object gauge by gauge 
        """
        if not hasattr(self, 'gauge_values'):
            self.gauge_values = {}
        _, _, values = self.read_gauges_file(var_name, compressed)
        values_pd = self.times_simu.copy()
        if var_name=='h': # calculation method is min
            values = values[:, gauge_ind]
            values = values.max(axis=1)
            values_pd['values'] = values
        elif var_name=='hU':
            values = values[:, :, gauge_ind]
            values = values.sum(axis=2)*self.header_global['cellsize']
            values_pd['values_x'] = values[0]
            values_pd['values_y'] = values[1]
        else:
            values = values[:, gauge_ind]
            values_pd['values'] = values        
        if gauge_name in self.gauge_values.keys():
            gauge_dict = self.gauge_values[gauge_name]
            gauge_dict[var_name] = values_pd
        else:
            gauge_dict = {var_name:values_pd}
        self.gauge_values[gauge_name] = gauge_dict
    
    def add_grid_results(self, result_names, compressed=False):
        """Read and add grid results to the object
        result_names: string or list of string, gives the name of grid file
        """
        if not hasattr(self, 'grid_results'):
            self.grid_results = {}
        if type(result_names) is list: # for a list of files
            for file_tag in result_names:
                grid_array, header = self.read_grid_file(file_tag, compressed)
                self.grid_results['file_tag'] = grid_array
        else: # for one file
            file_tag = result_names
            grid_array, header = self.read_grid_file(file_tag, compressed)
            self.grid_results[file_tag] = grid_array
        
    def set_headers_from_output(self, output_asc='h_0.asc'):
        """ Read header information of each model domain/subdomain
        output_asc is the asc file providing header information if mesh files
        are not available
        """
        if self.num_of_sections==1:
            file_name = self.output_folder+'/'+output_asc
            header = sp.arc_header_read(file_name)
        else:
            header = []
            for output_folder in self.output_folder:
                file_name = output_folder+'/'+output_asc
                header.append(sp.arc_header_read(file_name))
            self.header_global = _header_local2global(header)
        self.header = header
    
    def set_ref_datetime(self, date_time,
                             str_format='%Y-%m-%d %H:%M:%S'):
        """Set the refernce datetime of the simulation
        """
        if type(date_time) is str:
            self.ref_datetime = datetime.datetime.strptime(date_time,
                                                               str_format)
        elif type(date_time) is datetime.datetime:
            self.ref_datetime = date_time
        else:
            raise IOError('date_time must be a datetime object or a string')
    
    def _combine_multi_gpu_grid_data(self, asc_file_name):
        """Combine multi-gpu grid files into a single file
        asc_file_name: string endswith '.asc'
        """
        if not hasattr(self, 'header'):
            self.set_headers_from_output(asc_file_name)
        header_global = _header_local2global(self.header)
        grid_shape = (header_global['nrows'], header_global['ncols'])
        array_global = np.zeros(grid_shape)
        for header, output_folder in zip(self.header, self.output_folder):
            ind_top, ind_bottom = _header2row_numbers(header, header_global)
            file_name = output_folder+'/'+asc_file_name
            array_local, _, _ = sp.arcgridread(file_name)
            array_global[ind_top:ind_bottom+1,:] = array_local
        return array_global, header_global
    
    def save_object(self, file_name):
        """Save the object to a pickle file
        """
        save_object(self, file_name, compression=True)
    
def load_object(file_name):
    """ Read a pickle file as an InputHipims object
    """
    #read an object file
    try:
        with gzip.open(file_name, 'rb') as input_file:
            obj = pickle.load(input_file)
    except:
        with open(file_name, 'rb') as input_file:
            obj = pickle.load(input_file)
    print(file_name+' loaded')
    return obj

def save_object(obj, file_name, compression=True):
    """ Save the object
    """
    # Overwrites any existing file.
    if not file_name.endswith('.pickle'):
        file_name = file_name+'.pickle'
    if compression:
        with gzip.open(file_name, 'wb') as output_file:
            pickle.dump(obj, output_file, pickle.HIGHEST_PROTOCOL)
    else:
        with open(file_name, 'wb') as output_file:
            pickle.dump(obj, output_file, pickle.HIGHEST_PROTOCOL)
    print(file_name+' has been saved')
#%% =======================Supporting functions===============================
def _combine_multi_gpu_gauges_data(header_list, case_folder, file_tag):
    """ Combine gauges outputs from multi-gpu models according to gauges
    position data.
    gauges_pos.dat for each domain must be available
    """
    gauges_array = []
    value_array = []
    for i in range(len(header_list)):
        domain_header = header_list[i]
        gauge_pos_file = case_folder+'/'+str(i)+'/input/field/gauges_pos.dat'
        gauge_xy = np.loadtxt(gauge_pos_file, dtype='float64', ndmin=2)
        gauge_ind = _find_gauges_inside_domain(domain_header, gauge_xy)
        gauges_array.append(gauge_xy[gauge_ind,:])
        file_name = case_folder+'/'+str(i)+'/output/'+file_tag+'_gauges.dat'
        times, values = _read_one_gauge_file(file_name, gauge_ind)
        value_array.append(values)
    gauges_array = np.concatenate(gauges_array, axis=0)
    gauges_array, ind = np.unique(gauges_array, axis=0, return_index=True)
    if values.ndim == 2:
        value_array = np.concatenate(value_array, axis=1)
        value_array = value_array[:, ind]
    else: # values.ndim == 3
        value_array = np.concatenate(value_array, axis=2)
        value_array = value_array[:, :, ind]
    return gauges_array, times, value_array

def _find_gauges_inside_domain(domain_header, gauge_xy):
    """ Find the gauges inside a domain
    domain_header: (dict) header of the domain DEM
    gauge_xy: xy coordinate of the gauges
    Return: (numpy array) index of gauges inside the model domain
    """
    extent = sp.header2extent(domain_header)
    ind_x = np.logical_and(gauge_xy[:, 0] > extent[0],
                           gauge_xy[:, 0] < extent[1])
    ind_y = np.logical_and(gauge_xy[:, 1] > extent[2],
                           gauge_xy[:, 1] < extent[3])
    gauge_ind = np.where(np.logical_and(ind_x, ind_y))
    gauge_ind = gauge_ind[0]
    return gauge_ind
        
def _read_one_gauge_file(file_name, gauge_ind=None):
    """ Read a gauge file and return time series and values with outside gauges
    removed
    Supporting function to read_gauges_file
    """
    t_value = np.loadtxt(file_name, dtype='float64')
    times = t_value[:, 0]
    values = t_value[:, 1:]
    if 'hU_gauges.dat' in file_name:
        values = np.array([values[:, 0::2], values[:, 1::2]])
    if gauge_ind is not None:
        if values.ndim==2:
            values = values[:, gauge_ind]
        else: #ndim=3
            values = values[:, :, gauge_ind]
    return times, values

def _header2row_numbers(local_header, global_header):
    """Calculate local grid starting and ending rows in global grid
    Return:
        ind_top: the index of the top row
        ind_bottom: the index of the bottom row
    """
    y_bottom_gap = local_header['yllcorner']-global_header['yllcorner']
    row_gap = round(y_bottom_gap/local_header['cellsize'])
    ind_bottom = global_header['nrows']-1-row_gap
    ind_top = ind_bottom-local_header['nrows']+1
    ind_top = int(ind_top)
    ind_bottom = int(ind_bottom)
    return ind_top, ind_bottom
    
def _header_local2global(header_list):
    """Convert local headers to a global header
    """
    extent_array = []
    for header in header_list:
        extent = sp.header2extent(header)
        extent_array.append(extent)
    extent_array = np.asarray(extent_array)
    header_global = header_list[0].copy()
    y_bottom = extent_array[:,2].min()
    header_global['yllcorner'] = y_bottom
    y_top = extent_array[:,3].max()
    nrows = (y_top-y_bottom)/header_global['cellsize']
    header_global['nrows'] = int(round(nrows))
    return header_global