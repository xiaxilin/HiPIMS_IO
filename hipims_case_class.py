#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------
hipims_case_class
Generate input files for a hipims flood model case
-------------------------------------------------------------------------------
@author: Xiaodong Ming
Created on Sun Sep  1 10:01:43 2019
-------------------------------------------------------------------------------
Assumptions:
- Input DEM is a regular DEM file
- its map unit is meter
- its cellsize is the same in both x and y direction
- its reference position is on the lower left corner of the southwest cell
- All the other grid-based input files must be consistent with the DEM file
To do:
- generate input (including sub-folder mesh and field) and output folders
- generate mesh file (DEM.txt) and field files
- divide model domain into small sections if multiple GPU is used
"""
__author__ = "Xiaodong Ming"
import os
import sys
import warnings
import shutil
import pickle
import gzip
import scipy.signal  # to call scipy.signal.convolve2d to get outline boundary
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mplP
import myclass  # to call class Raster
from datetime import datetime
#%% grid data for HiPIMS input format
class InputHipims:
    """To define input files for a HiPIMS flood model case
    Read data, process data, write input files, and save data of a model case.
    Properties (public):
        case_folder: (str) the absolute path of the case folder
        data_folders: (dict) paths for data folders (input, output, mesh, field)
        num_of_sections: (scalar) number of GPUs to run the model
        Boundary: a boundary object of class Boundary to provide boundary cells
            location, ID, type, and source data.
        Raster: a raster object to provide DEM data
        Summary: a ModelSummary object to record model information
        Sections: a list of objects of child-class InputHipimsSub
        attributes_default: (dict) default model attribute names and values
        attributes: (dict) model attribute names and values
    Properties (Private):
        _valid_cell_subs: (tuple, int numpy) two numpy array indicating rows
            and cols of valid cells on the DEM grid
        _outline_cell_subs: (tuple, int numpy) two numpy array indicating rows
            and cols of outline cells on the DEM grid
        _global_header: (dict) header of the DEM for the whole model domain
    Methods (public):
        set_parameter: set grid-based parameters
        set_boundary_condition: set boundary condition with a boundary list
        set_rainfall: set rainfall mask and sources
        set_gauges_position: set X-Y coordinates of monitoring  gauges
        write_input_files: write all input files or a specific file
        write_grid_files: write grid-based data files
        write_boundary_conditions: write boundary sources, if flow time series
            is given, it will be converted to velocities in x and y directions
        write_gauges_position: write coordinates of monitoring gauges
        write_halo_file: write overlayed cell ID for multiple GPU cases
    Methods(protected):
        _get_cell_subs:
        __divide_grid: split model domain:
        _get_boundary_id_code_array:
        _get_vector_value:
        _write_grid_files:
        _initialize_summary_obj:
        (Private only for base class)
        __write_boundary_conditions:
        __write_gauge_pos:
        __copy_to_all_sections:
    Classes:
        InputHipimsSub: child class of InputHiPIMS, provide information
            of each sub-domain
        Boundary: provide information of boundary conditions
        ModelSummary: record basic information of an object of InputHiPIMS
    """
    # default parameters
    attributes_default = {'h0':0, 'hU0x':0, 'hU0y':0,
                          'precipitation':0,
                          'precipitation_mask':0,
                          'precipitation_source':np.array([[0, 0], [1, 0]]),
                          'manning':0.035,
                          'sewer_sink':0,
                          'cumulative_depth':0, 'hydraulic_conductivity':0,
                          'capillary_head':0, 'water_content_diff':0,
                          'gauges_pos':np.array([[0, 0]])}
    grid_files = ['z', 'h', 'hU', 'precipitation_mask',
                  'manning', 'sewer_sink', 'precipitation',
                  'cumulative_depth', 'hydraulic_conductivity',
                  'capillary_head', 'water_content_diff']
    def __init__(self, dem_data, num_of_sections=1, case_folder=None):
        """
        dem_data: (Raster object) or (str) provides file name of the DEM data
        """
        if type(dem_data) is str:
            self.Raster = myclass.Raster(dem_data) # create Raster object
        elif type(dem_data) is myclass.Raster:
            self.Raster = dem_data
        if case_folder is None:
            case_folder = os.getcwd()
        self.case_folder = case_folder
        self.num_of_sections = num_of_sections
        self.attributes = InputHipims.attributes_default.copy()
        self.times = None
        # get row and col index of all cells on DEM grid
        self._get_cell_subs()  # add _valid_cell_subs and _outline_cell_subs
        # divide model domain to several sections if it is not a sub section
        # each section contains a "HiPIMS_IO_class.InputHipimsSub" object
        if isinstance(self, InputHipimsSub):
#            print('child class object '+str(self.section_id))
            pass
        else:
            self.__divide_grid()
            self._global_header = self.Raster.header
        self._initialize_summary_obj()# get a Model Summary object
        self.set_case_folder() # set data_folders
        self.set_boundary_condition(outline_boundary='open')
        self.birthday = datetime.now()
    def __str__(self):
        """
        To show object summary information when it is called in console
        """
        self.Summary.display()
        time_str = self.birthday.strftime('%Y-%m-%d %H:%M:%S')
        return  self.__class__.__name__+' object created on '+ time_str
#    __repr__ = __str__
#******************************************************************************
#************************Setup the object**************************************
    def set_boundary_condition(self, boundary_list=None,
                               outline_boundary='open'):
        """To create a Boundary object for boundary conditions
        The object contains
              outline_boundary, a dataframe of boundary type, extent, source
              data, code, ect..., and a boundary subscrpits
              tuple (cell_subs).For multi-GPU case, a boundary subscrpits tuple
              (cell_subs_l) based on sub-grids will be created for each section
        outline_boundary: (str) 'open'|'rigid', default outline boundary is
            open and both h and hU are set as zero
        boundary_list: (list of dicts), each dict contain keys (polyPoints,
            type, h, hU) to define a IO boundary's position, type, and
            Input-Output (IO) sources timeseries. Keys including:
            1.polyPoints is a numpy array giving X(1st col) and Y(2nd col)
                coordinates of points to define the position of a boundary.
                An empty polyPoints means outline boundary.
            2.type: 'open'|'rigid'
            3.h: a two-col numpy array. The 1st col is time(s). The 2nd col is
                water depth(m)
            4.hU: a two-col numpy array. The 1st col is time(s). The 2nd col is
                discharge(m3/s) or a three-col numpy array, the 2nd col and the
                3rd col are velocities(m/s) in x and y direction, respectively.
        cell_subs: (list of tuple) subsripts of cells for each boundary
        cell_id: (list of vector) valid id of cells for each boundary
        """
        # initialize a Boundary object
        if boundary_list is None and hasattr(self, 'Boundary'):
            boundary_list = self.Boundary.boundary_list
        bound_obj = Boundary(boundary_list, outline_boundary)
        valid_subs = self._valid_cell_subs
        outline_subs = self._outline_cell_subs
        if not isinstance(self, InputHipimsSub):
            dem_header = self._global_header
        # add the subsripts and id of boundary cells on the domain grid
            bound_obj._fetch_boundary_cells(valid_subs,
                                            outline_subs, dem_header)
        self.Boundary = bound_obj
        if hasattr(self, 'Sections'):
            bound_obj._divide_domain(self)
        summary_str = bound_obj.get_summary()
        self.Summary.add_items('Boundary conditions', summary_str)
#        self.Boundary.print_summary()

    def set_parameter(self, parameter_name, parameter_value):
        """ Set grid-based parameters
        parameter_name: (str) including: h0, hU0x, hU0y, manning,
            precipitation_mask, sewer_sink, cumulative_depth,
            hydraulic_conductivity, capillary_head, water_content_diff
        parameter_value: (scalar)|(numpy array) with the same size of DEM. All
            parameter values are given to the global grid and can be divided
            to local grids in writing process if multiple sections are defined.
        """
        if parameter_name not in InputHipims.attributes_default.keys():
            raise ValueError('Parameter is not recognized: '+parameter_name)
        if type(parameter_value) is np.ndarray:
            if parameter_value.shape != self.Raster.array.shape:
                raise ValueError('The array of the parameter '
                                 'value should have the same '
                                 'shape with the DEM array')
        elif np.isscalar(parameter_value) is False:
            raise ValueError('The parameter value must be either '
                             'a scalar or an numpy array')
        self.attributes[parameter_name] = parameter_value
        # renew summary information
        self.Summary.add_param_infor(parameter_name, parameter_value)

    def set_rainfall(self, rain_mask=None, rain_source=None):
        """ Set rainfall mask and rainfall source
        rainfall_mask: numpy int array withe the same size of DEM, each mask
                 value indicates one rainfall source
        rainfall_source: numpy array the 1st column is time in seconds, 2nd to
             the end columns are rainfall rates in m/s. The number of columns
             in rainfall_source should be equal to the number of mask values
             plus one (the time column)
        """
        if rain_mask is None:
            rain_mask = self.attributes['precipitation_mask']
        elif rain_mask.shape != self.Raster.array.shape:
            raise ValueError('The shape of rainfall_mask array '
                             'is not consistent with DEM')
        else:
            self.attributes['precipitation_mask'] = rain_mask
        rain_mask = rain_mask.astype('int32')
        num_of_masks = rain_mask.max()+1 # starting from 0
        if rain_source.shape[1]-1 != num_of_masks:
            warning_str= 'The column of rain source '+\
                         'is not consistent with the number of rain masks'
            warnings.warn(warning_str)
        _check_rainfall_rate_values(rain_source, times_in_1st_col=True)
        self.Summary.add_param_infor('precipitation_mask', rain_mask)
        if rain_source is not None:
            self.attributes['precipitation_source'] = rain_source
            rain_mask_unique = np.unique(rain_mask).flatten()
            rain_mask_unique = rain_mask_unique.astype('int32')
            rain_source_valid = np.c_[rain_source[:,0],
                                      rain_source[:,rain_mask_unique+1]]
            self.Summary.add_param_infor('precipitation_source', 
                                         rain_source_valid)
            if self.times is None:
                time_seires = rain_source[:,0]
                self.times = [time_seires.min(), time_seires.max(),
                              np.ptp(time_seires), np.ptp(time_seires)] 
        # renew summary information

    def set_gauges_position(self, gauges_pos=None):
        """Set coordinates of monitoring gauges
        """
        if gauges_pos is None:
            gauges_pos = self.attributes['gauges_pos']
        if type(gauges_pos) is list:
            gauges_pos = np.array(gauges_pos)
        if gauges_pos.shape[1] != 2:
            raise ValueError('The gauges_pos arraymust have two columns')
        self.attributes['gauges_pos'] = gauges_pos
        self.Summary.add_param_infor('gauges_pos', gauges_pos)
        # for multi_GPU, divide gauges based on the extent of each section
        if self.num_of_sections > 1:
            pos_X = gauges_pos[:,0]
            pos_Y = gauges_pos[:,1]
            for obj_section in self.Sections:
                extent = obj_section.Raster.extent
                ind_x = np.logical_and(pos_X >= extent[0], pos_X <= extent[1])
                ind_y = np.logical_and(pos_Y >= extent[2], pos_Y <= extent[3])
                ind = np.where(np.logical_and(ind_x, ind_y))
                ind = ind[0]
                obj_section.attributes['gauges_pos'] = gauges_pos[ind,:]
                obj_section.attributes['gauges_ind'] = ind

    def set_case_folder(self, new_folder=None, make_dir=False):
        """ Initialize, renew, or create case and data folders
        new_folder: (str) renew case and data folder if it is given
        make_dir: True|False create folders if it is True
        """
        # to change case_folder
        if new_folder is not None:
            if new_folder.endswith('/'):
                new_folder = new_folder[:-1]
            self.case_folder = new_folder
            # for multiple GPUs
            if hasattr(self, 'Sections'):
                for obj in self.Sections:
                    sub_case_folder = new_folder+'/'+str(obj.section_id)
                    obj.set_case_folder(sub_case_folder)
        # for single gpu
        self.data_folders = _create_io_folders(self.case_folder,
                                               make_dir)
        if hasattr(self, 'Summary'):
            self.Summary.set_param('Case folder', self.case_folder)

    def add_user_defined_parameter(self, param_name, param_value):
        """ Add a grid-based user-defined parameter to the model
        param_name: (str) name the parameter and the input file name as well
        param_value: (scalar) or (numpy arary) with the same size of DEM array
        """
        if param_name not in InputHipims.grid_files:
            InputHipims.grid_files.append(param_name)
        self.attributes[param_name] = param_value
        print(param_name+ 'is added to the InputHipims object')
        self.Summary.add_param_infor(param_name, param_value)

    def decomposite_domain(self, num_of_sections):
        if isinstance(self, InputHipims):
            self.num_of_sections = num_of_sections
            self._global_header = self.Raster.header
            self.__divide_grid()
            self.set_case_folder() # set data_folders
            outline_boundary = self.Boundary.outline_boundary
            self.set_boundary_condition(outline_boundary=outline_boundary)
            self.set_gauges_position()
            self.Boundary._divide_domain(self)
            self.birthday = datetime.now()
        else:
            raise ValueError('The object cannot be decomposited!')

    def write_input_files(self, file_tag=None):
        """ Write input files
        To classify the input files and call functions needed to write each
            input files
        file_tag: 'all'|'z', 'h', 'hU', 'manning', 'sewer_sink',
                        'cumulative_depth', 'hydraulic_conductivity',
                        'capillary_head', 'water_content_diff'
                        'precipitation_mask', 'precipitation_source',
                        'boundary_condition', 'gauges_pos'
        """
        self._make_data_dirs()
        grid_files = InputHipims.grid_files
        if file_tag is None or file_tag == 'all':
            for grid_file in grid_files: # grid-based files
                self.write_grid_files(grid_file)
            self.write_boundary_conditions()
            self.write_rainfall_source()
            self.write_gauges_position()
            if self.num_of_sections > 1:
                self.write_halo_file()
            self.write_mesh_file(self)
            write_times_setup(self.case_folder, 
                              self.num_of_sections, self.times)
            write_device_setup(self.case_folder, self.num_of_sections)
        elif file_tag == 'boundary_condition':
            self.write_boundary_conditions()
        elif file_tag == 'gauges_pos':
            self.write_gauges_position()
        elif file_tag == 'halo':
            self.write_halo_file()
        else:
            if file_tag in grid_files:
                self.write_grid_files(file_tag)
            else:
                raise ValueError('file_tag is not recognized')

    def write_grid_files(self, file_tag, is_single_gpu=False):
        """Write grid-based files
        Public version for both single and multiple GPUs
        file_tag: the pure name of a grid-based file
        """
        self._make_data_dirs()
        grid_files = InputHipims.grid_files
        if file_tag not in grid_files:
            raise ValueError(file_tag+' is not a grid-based file')
        if is_single_gpu or self.num_of_sections == 1:
            # write as single GPU even the num of sections is more than one
            self._write_grid_files(file_tag, is_multi_gpu=False)
        else:
            self._write_grid_files(file_tag, is_multi_gpu=True)
        self.Summary.write_readme(self.case_folder+'/readme.txt')
        print(file_tag+' created')

    def write_boundary_conditions(self):
        """ Write boundary condtion files
        if there are multiple domains, write in the first folder
            and copy to others
        """
        self._make_data_dirs()
        if self.num_of_sections > 1:  # multiple-GPU
            field_dir = self.Sections[0].data_folders['field']
            file_names_list = self.__write_boundary_conditions(field_dir)
            self.__copy_to_all_sections(file_names_list)
        else:  # single-GPU
            field_dir = self.data_folders['field']
            self.__write_boundary_conditions(field_dir)
        self.Summary.write_readme(self.case_folder+'/readme.txt')
        print('boundary condition files created')

    def write_rainfall_source(self):
        """Write rainfall source data
        rainfall mask can be written by function write_grid_files
        """
        self._make_data_dirs()
        rain_source = self.attributes['precipitation_source']
        case_folder = self.case_folder
        num_of_sections = self.num_of_sections
        write_rain_source(rain_source, case_folder, num_of_sections)
        self.Summary.write_readme(self.case_folder+'/readme.txt')

    def write_gauges_position(self, gauges_pos=None):
        """ Write the gauges position file
        Public version for both single and multiple GPUs
        """
        self._make_data_dirs()
        if gauges_pos is not None:
            self.set_gauges_position(np.array(gauges_pos))
        if self.num_of_sections > 1:  # multiple-GPU
            for obj_section in self.Sections:
                field_dir = obj_section.data_folders['field']
                obj_section.__write_gauge_ind(field_dir)
                obj_section.__write_gauge_pos(field_dir)
        else:  # single-GPU
            field_dir = self.data_folders['field']
            self.__write_gauge_pos(field_dir)
        self.Summary.write_readme(self.case_folder+'/readme.txt')
        print('gauges_pos.dat created')

    def write_halo_file(self):
        """ Write overlayed cell IDs
        """
        num_section = self.num_of_sections
        case_folder = self.case_folder
        if not case_folder.endswith('/'):
            case_folder = case_folder+'/'
        file_name = case_folder+'halo.dat'
        with open(file_name, 'w') as file2write:
            file2write.write("No. of Domains\n")
            file2write.write("%d\n" % num_section)
            for obj_section in self.Sections:
                file2write.write("#%d\n" % obj_section.section_id)
                overlayed_id = obj_section.overlayed_id
                for key in ['bottom_low', 'bottom_high',
                            'top_high', 'top_low']:
                    if key in overlayed_id.keys():
                        line_ids = overlayed_id[key]
                        line_ids = np.reshape(line_ids, (1, line_ids.size))
                        np.savetxt(file2write,
                                   line_ids, fmt='%d', delimiter=' ')
                    else:
                        file2write.write(' \n')
        print('halo.dat created')

    def write_mesh_file(self, is_single_gpu=False):
        """ Write mesh file DEM.txt, compatoble for both single and multiple
        GPU model
        """
        self._make_data_dirs()
        if is_single_gpu is True or self.num_of_sections == 1:
            file_name = self.data_folders['mesh']+'DEM.txt'
            self.Raster.Write_asc(file_name)
        else:
            for obj_section in self.Sections:
                file_name = obj_section.data_folders['mesh']+'DEM.txt'
                obj_section.Raster.Write_asc(file_name)
        self.Summary.write_readme(self.case_folder+'/readme.txt')
        print('DEM.txt created')

    def save_object(self, file_name):
        """ Save object as a pickle file
        """
        if not file_name.endswith('.pickle'):
            file_name = file_name+'.pickle'
        with open(file_name, 'wb') as output:  # Overwrites any existing file.
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)
#------------------------------------------------------------------------------
#******************************* Visualization ********************************
#------------------------------------------------------------------------------
    def plot_rainfall_source(self,start_date=None, method='mean'):
        """ Plot time series of average rainfall rate inside the model domain
        start_date: a datetime object to give the initial date and time of rain
        method: 'mean'|'max','min','mean'method to calculate gridded rainfall 
        over the model domain
            
        """        
        rain_source = self.attributes['precipitation_source']
        rain_mask = self.attributes['precipitation_mask']
        if type(rain_mask) is np.ndarray:
            rain_mask = rain_mask[~np.isnan(self.Raster.array)]
        rain_mask = rain_mask.astype('int32')
        rain_mask_unique = np.unique(rain_mask).flatten()
        rain_mask_unique = rain_mask_unique.astype('int32')
        
        rain_source_valid = rain_source[:,rain_mask_unique+1]
        time_series = rain_source[:,0]
        if type(start_date) is datetime:
            from datetime import timedelta
            time_delta = np.array([timedelta(seconds=i) for i in time_series])
            time_x = start_date+time_delta
        else:
            time_x = time_series
        if method == 'mean':
            value_y = np.mean(rain_source_valid,axis=1)
        elif method== 'max':
            value_y = np.max(rain_source_valid,axis=1)
        elif method== 'min':
            value_y = np.min(rain_source_valid,axis=1)
        elif method== 'median':
            value_y = np.median(rain_source_valid,axis=1)
        else:
            raise ValueError('Cannot recognise the calculation method')
        value_y =  value_y*3600*1000
        plot_data = np.c_[time_x,value_y]
        fig, ax = plt.subplots()
        ax.plot(time_x,value_y)
        ax.set_ylabel('Rainfall rate (mm/h)')
        ax.grid(True)
        plt.show()
        return plot_data
        
        
#------------------------------------------------------------------------------
#*************************** Protected methods ********************************
#------------------------------------------------------------------------------
    def _get_cell_subs(self, dem_array=None):
        """ To get valid_cell_subs and outline_cell_subs for the object
        To get the subscripts of each valid cell on grid
        Input arguments are for sub Hipims objects
        _valid_cell_subs
        _outline_cell_subs
        """
        if dem_array is None:
            dem_array = self.Raster.array
        valid_id, outline_id = _get_cell_id_array(dem_array)
        subs = np.where(~np.isnan(valid_id))
        id_vector = valid_id[subs]
        # sort the subscripts according to cell id values
        sorted_vectors = np.c_[id_vector, subs[0], subs[1]]
        sorted_vectors = sorted_vectors[sorted_vectors[:, 0].argsort()]
        self._valid_cell_subs = (sorted_vectors[:, 1].astype('int32'),
                                 sorted_vectors[:, 2].astype('int32'))
        subs = np.where(outline_id == 0) # outline boundary cell
        outline_id_vect = outline_id[subs]
        sorted_array = np.c_[outline_id_vect, subs[0], subs[1]]
        self._outline_cell_subs = (sorted_array[:, 1].astype('int32'),
                                   sorted_array[:, 2].astype('int32'))
    def __divide_grid(self):
        """
        Divide DEM grid to sub grids
        Create objects based on sub-class InputHipimsSub
        """
        if isinstance(self, InputHipimsSub):
            return 0  # do not divide InputHipimsSub objects, return a number
        else:
            if self.num_of_sections == 1:
                return 1 # do not divide if num_of_sections is 1
        num_of_sections = self.num_of_sections
        dem_header = self.Raster.header
        self._global_header = dem_header
        # subscripts of the split row [0, 1,...] from bottom to top
        split_rows = _get_split_rows(self.Raster.array, num_of_sections)
        array_local, header_local = \
            _split_array_by_rows(self.Raster.array, dem_header, split_rows)
        # to receive InputHipimsSub objects for sections
        Sections = []
        section_sequence = np.arange(num_of_sections)
        header_global = self._global_header
        for i in section_sequence:  # from bottom to top
            case_folder = self.case_folder+'/'+str(i)
            # create a sub object of InputHipims
            sub_hipims = InputHipimsSub(array_local[i], header_local[i],
                                        case_folder, num_of_sections)
            # get valid_cell_subs on the global grid
            valid_cell_subs = sub_hipims._valid_cell_subs
            valid_subs_global = \
                 _cell_subs_convertor(valid_cell_subs, header_global,
                                      header_local[i], to_global=True)
            sub_hipims.valid_subs_global = valid_subs_global
            # record section sequence number
#            sub_hipims.section_id = i
            #get overlayed_id (top two rows and bottom two rows)
            top_h = np.where(valid_cell_subs[0] == 0)
            top_l = np.where(valid_cell_subs[0] == 1)
            bottom_h = np.where(
                valid_cell_subs[0] == valid_cell_subs[0].max()-1)
            bottom_l = np.where(valid_cell_subs[0] == valid_cell_subs[0].max())
            if i == 0: # the bottom section
                overlayed_id = {'top_high':top_h[0], 'top_low':top_l[0]}
            elif i == self.num_of_sections-1: # the top section
                overlayed_id = {'bottom_low':bottom_l[0],
                                'bottom_high':bottom_h[0]}
            else:
                overlayed_id = {'top_high':top_h[0], 'top_low':top_l[0],
                                'bottom_high':bottom_h[0],
                                'bottom_low':bottom_l[0]}
            sub_hipims.overlayed_id = overlayed_id
            Sections.append(sub_hipims)
        # reset global var section_id of InputHipimsSub
        InputHipimsSub.section_id = 0
        self.Sections = Sections
        self._initialize_summary_obj()# get a Model Summary object

    # only for global object
    def _get_vector_value(self, attribute_name, is_multi_gpu=True,
                          add_initial_water=True):
        """ Generate a single vector for values in each grid cell sorted based
        on cell IDs
        attribute_name: attribute names based on a grid
        Return:
            output_vector: a vector of values in global valid grid cells
                            or a list of vectors for each sub domain
        """
        # get grid value
        dem_shape = self.Raster.array.shape
        grid_values = np.zeros(dem_shape)
        if add_initial_water:
            add_value = 0.0001
        else:
            add_value = 0

        def add_water_on_io_cells(bound_obj, grid_values, source_key,
                                  add_value):
            """ add a small water depth/velocity to IO boundary cells
            """
            for ind_num in np.arange(bound_obj.num_of_bound):
                bound_source = bound_obj.data_table[source_key][ind_num]
                if bound_source is not None:
                    source_value = np.unique(bound_source[:, 1:])
                    # zero boundary conditions
                    if not (source_value.size == 1 and source_value[0] == 0):
                        cell_subs = bound_obj.cell_subs[ind_num]
                        grid_values[cell_subs] = add_value
            return grid_values
        # set grid value for the entire domain
        if attribute_name == 'z':
            grid_values = self.Raster.array
        elif attribute_name == 'h':
            grid_values = grid_values+self.attributes['h0']
            # traversal each boundary to add initial water
            grid_values = add_water_on_io_cells(self.Boundary, grid_values,
                                                'hSources', add_value)
        elif attribute_name == 'hU':
            grid_values0 = grid_values+self.attributes['hU0x']
            grid_values1 = grid_values+self.attributes['hU0y']
            grid_values1 = add_water_on_io_cells(self.Boundary, grid_values1,
                                                 'hUSources', add_value)
            grid_values = [grid_values0, grid_values1]
        else:
            grid_values = grid_values+self.attributes[attribute_name]

        def grid_to_vect(grid_values, cell_subs):
            """ Convert grid values to 1 or 2 col vector values
            """
            if type(grid_values) is list:
                vector_value0 = grid_values[0][cell_subs]
                vector_value1 = grid_values[1][cell_subs]
                vector_value = np.c_[vector_value0, vector_value1]
            else:
                vector_value = grid_values[cell_subs]
            return vector_value
        #
        if is_multi_gpu: # generate vector value for multiple GPU
            output_vector = []
            for obj_section in self.Sections:
                cell_subs = obj_section.valid_subs_global
                vector_value = grid_to_vect(grid_values, cell_subs)
                output_vector.append(vector_value)
        else:
            output_vector = grid_to_vect(grid_values, self._valid_cell_subs)
        return output_vector

    def _get_boundary_id_code_array(self, file_tag='z'):
        """
        To generate a 4-col array of boundary cell id (0) and code (1~3)
        """
        bound_obj = self.Boundary
        output_array_list = []
        for ind_num in np.arange(bound_obj.num_of_bound):
            if file_tag == 'h':
                bound_code = bound_obj.data_table.h_code[ind_num]
            elif file_tag == 'hU':
                bound_code = bound_obj.data_table.hU_code[ind_num]
            else:
                bound_code = np.array([[2, 0, 0]]) # shape (1, 3)
            if bound_code.ndim < 2:
                bound_code = np.reshape(bound_code, (1, bound_code.size))
            cell_id = bound_obj.cell_id[ind_num]
            if cell_id.size > 0:
                bound_code_array = np.repeat(bound_code, cell_id.size, axis=0)
                id_code_array = np.c_[cell_id, bound_code_array]
                output_array_list.append(id_code_array)
        # add overlayed cells with [4, 0, 0]
        # if it is a sub section object, there should be attributes:
        # overlayed_id, and section_id
        if hasattr(self, 'overlayed_id'):
            cell_id = list(self.overlayed_id.values())
            cell_id = np.concatenate(cell_id, axis=0)
            bound_code = np.array([[4, 0, 0]]) # shape (1, 3)
            bound_code_array = np.repeat(bound_code, cell_id.size, axis=0)
            id_code_array = np.c_[cell_id, bound_code_array]
            output_array_list.append(id_code_array)
        output_array = np.concatenate(output_array_list, axis=0)
        # when unique the output array according to cell id
        # keep the last occurrence rather than the default first occurrence
        output_array = np.flipud(output_array) # make the IO boundaries first
        _, ind = np.unique(output_array[:, 0], return_index=True)
        output_array = output_array[ind]
        return output_array

    def _initialize_summary_obj(self):
        """ Initialize the model summary object
        """
        summary_obj = ModelSummary(self)
        self.Summary = summary_obj

    def _write_grid_files(self, file_tag, is_multi_gpu=True):
        """ Write input files consistent with the DEM grid
        Private function called by public function write_grid_files
        file_name: includes ['h','hU','precipitation_mask',
                             'manning','sewer_sink',
                             'cumulative_depth', 'hydraulic_conductivity',
                             'capillary_head', 'water_content_diff']
        """
        if is_multi_gpu is True:  # write for multi-GPU, use child object
            vector_value_list = self._get_vector_value(file_tag, is_multi_gpu)
            for obj_section in self.Sections:
                vector_value = vector_value_list[obj_section.section_id]
                cell_id = np.arange(vector_value.shape[0])
                cells_vect = np.c_[cell_id, vector_value]
                file_name = obj_section.data_folders['field']+file_tag+'.dat'
                if file_tag == 'precipitation_mask':
                    bounds_vect = None
                else:
                    bounds_vect = \
                        obj_section._get_boundary_id_code_array(file_tag)
                _write_two_arrays(file_name, cells_vect, bounds_vect)
        else:  # single GPU, use global object
            file_name = self.data_folders['field']+file_tag+'.dat'
            vector_value = self._get_vector_value(file_tag, is_multi_gpu=False)
            cell_id = np.arange(vector_value.shape[0])
            cells_vect = np.c_[cell_id, vector_value]
            if file_tag == 'precipitation_mask':
                bounds_vect = None
            else:
                bounds_vect = self._get_boundary_id_code_array(file_tag)
            _write_two_arrays(file_name, cells_vect, bounds_vect)
        return None

    def _make_data_dirs(self):
        """ Create folders in current device
        """
        if hasattr(self, 'Sections'):
            for obj_section in self.Sections:
                _create_io_folders(obj_section.case_folder, make_dir=True)
        else:
            _create_io_folders(self.case_folder, make_dir=True)
#------------------------------------------------------------------------------
#*************** Private methods only for the parent class ********************
#------------------------------------------------------------------------------

    def __write_boundary_conditions(self, field_dir, file_tag='both'):
        """ Write boundary condition source files,if hU is given as flow
        timeseries, convert flow to hUx and hUy.
        Private function to call by public function write_boundary_conditions
        file_tag: 'h', 'hU', 'both'
        h_BC_[N].dat, hU_BC_[N].dat
        if hU is given as flow timeseries, convert flow to hUx and hUy
        """
        obj_boundary = self.Boundary
        file_names_list = []
        fmt_h = ['%g', '%g']
        fmt_hu = ['%g', '%g', '%g']
        # write h_BC_[N].dat
        if file_tag in ['both', 'h']:
            h_sources = obj_boundary.data_table['hSources']
            ind_num = 0
            for i in np.arange(obj_boundary.num_of_bound):
                h_source = h_sources[i]
                if h_source is not None:
                    file_name = field_dir+'h_BC_'+str(ind_num)+'.dat'
                    np.savetxt(file_name, h_source, fmt=fmt_h, delimiter=' ')
                    ind_num = ind_num+1
                    file_names_list.append(file_name)
        # write hU_BC_[N].dat
        if file_tag in ['both', 'hU']:
            hU_sources = obj_boundary.data_table['hUSources']
            ind_num = 0
            for i in np.arange(obj_boundary.num_of_bound):
                hU_source = hU_sources[i]
                cell_subs = obj_boundary.cell_subs[i]
                if hU_source is not None:
                    file_name = field_dir+'hU_BC_'+str(ind_num)+'.dat'
                    if hU_source.shape[1] == 2:
                        # flow is given rather than speed
                        boundary_slope = np.polyfit(cell_subs[0],
                                                    cell_subs[1], 1)
                        theta = np.arctan(boundary_slope[0])
                        boundary_length = cell_subs[0].size* \
                                          self.Raster.header['cellsize']
                        hUx = hU_source[:, 1]*np.cos(theta)/boundary_length
                        hUy = hU_source[:, 1]*np.sin(theta)/boundary_length
                        hU_source = np.c_[hU_source[:, 0], hUx, hUy]
                        print('Flow series on boundary '+str(i)+
                              ' is converted to velocities')
                        print('Theta = '+'{:.3f}'.format(theta/np.pi)+'*pi')
                    np.savetxt(file_name, hU_source, fmt=fmt_hu, delimiter=' ')
                    ind_num = ind_num+1
                    file_names_list.append(file_name)
        return file_names_list

    def __write_gauge_pos(self, file_folder):
        """write monitoring gauges
        Private version of write_gauge_position
        gauges_pos.dat
        file_folder: folder to write file
        gauges_pos: 2-col numpy array of X and Y coordinates
        """
        gauges_pos = self.attributes['gauges_pos']
        file_name = file_folder+'gauges_pos.dat'
        fmt = ['%g %g']
        fmt = '\n'.join(fmt*gauges_pos.shape[0])
        gauges_pos_str = fmt % tuple(gauges_pos.ravel())
        with open(file_name, 'w') as file2write:
            file2write.write(gauges_pos_str)
        return file_name

    def __write_gauge_ind(self, file_folder):
        """write monitoring gauges index for mult-GPU sections
        Private function of write_gauge_position
        gauges_ind.dat
        file_folder: folder to write file
        gauges_ind: 1-col numpy array of index values
        """
        gauges_ind = self.attributes['gauges_ind']
        file_name = file_folder+'gauges_ind.dat'
        fmt = ['%g']
        fmt = '\n'.join(fmt*gauges_ind.shape[0])
        gauges_ind_str = fmt % tuple(gauges_ind.ravel())
        with open(file_name, 'w') as file2write:
            file2write.write(gauges_ind_str)
        return file_name

    def __copy_to_all_sections(self, file_names):
        """ Copy files that are the same in each sections
        file_names: (str) files written in the first seciton [0]
        boundary source files: h_BC_[N].dat, hU_BC_[N].dat
        rainfall source files: precipitation_source_all.dat
        gauges position file: gauges_pos.dat
        """
        if type(file_names) is not list:
            file_names = [file_names]
        for i in np.arange(1, self.num_of_sections):
            field_dir = self.Sections[i].data_folders['field']
            for file in file_names:
                shutil.copy2(file, field_dir)
#%% sub-class definition
class InputHipimsSub(InputHipims):
    """object for each section, child class of InputHipims
    Attributes:
        sectionNO: the serial number of each section
        _valid_cell_subs: (tuple, int) two numpy array indicating rows and cols of valid
                    cells on the local grid
        valid_cell_subsOnGlobal: (tuple, int) two numpy array indicating rows and cols of valid
                    cells on the global grid
        shared_cells_id: 2-row shared Cells id on a local grid
        case_folder: input folder of each section
        _outline_cell_subs: (tuple, int) two numpy array indicating rows and cols of valid
                    cells on a local grid
    """
    section_id = 0
    def __init__(self, dem_array, header, case_folder, num_of_sections):
        self.section_id = InputHipimsSub.section_id
        InputHipimsSub.section_id = self.section_id+1
        dem_data = myclass.Raster(array=dem_array, header=header)
        super().__init__(dem_data, num_of_sections, case_folder)
#%% boundary class definition
class Boundary(object):
    """Class for boundary conditions
    Public Properties:
        num_of_bound: number of boundaries
        data_table (data_frame) including attributes:
            type: a list of string 'open','rigid',
                    input-output boundary is open boundary with given water
                    depth and/or velocities
            extent: (2-col numpy array) poly points to define the extent of a
                    IO boundary. If extent is not given, then the boundary is
                    the domain outline
            hSources: a two-col numpy array. The 1st col is time(s). The 2nd
                    col is water depth(m)
            hUSources: a two-col numpy array. The 1st col is time(s). The 2nd
                    col is discharge(m3/s) or a three-col numpy array, the 2nd
                    col and the 3rd col are velocities(m/s) in x and y
                    direction, respectively.
            h_code: 3-element int to define the type of depth boundary
            hU_code: 3-element int to define th type of velocity boundary
            description: (str) description of a boundary
    Private Properties:
        code: 3-element row vector for each boundary cell
    Methods
        print_summary: print the summary information of a boundary object
        Gen3Code: Generate 3-element boundary codes
        CellLocate: fine boundary cells with given extent
    """
    def __init__(self, boundary_list=None, outline_boundary='open'):
        """default outline boundary: IO, h and Q are given as constant 0 values
        1. setup data_table including attributtes:
            type, extent, hSources, hUSources
        2. add boundary code 'h_code', 'hU_code' and 'description' to
            the data_table
        """
        data_table = _setup_boundary_data_table(boundary_list, outline_boundary)
        data_table = _get_boundary_code(data_table)
        num_of_bound = data_table.shape[0]
        self.data_table = data_table
        self.num_of_bound = num_of_bound
        self.h_sources = data_table['hSources']
        self.hU_sources = data_table['hUSources']
        self.boundary_list = boundary_list
        self.outline_boundary = outline_boundary
        self.cell_subs = None
        self.cell_id = None

    def print_summary(self):
        print('Number of boundaries: '+str(self.num_of_bound))
        for n in range(self.num_of_bound):
            if self.cell_subs is not None:
                num_cells = self.cell_subs[n][0].size
                description = self.data_table.description[n] \
                                 + ', number of cells: '+str(num_cells)
                print(str(n)+'. '+description)

    def get_summary(self):
        """ Get summary information strings
        """
        summary_str = []
        summary_str.append('Number of boundaries: '+str(self.num_of_bound))
        for n in range(self.num_of_bound):
            if self.cell_subs is not None:
                num_cells = self.cell_subs[n][0].size
                description = self.data_table.description[n] \
                                 + ', number of cells: '+str(num_cells)
                summary_str.append(str(n)+'. '+description)
        return summary_str

    def _fetch_boundary_cells(self, valid_subs, outline_subs, dem_header):
        """ To get the subsripts and id of boundary cells on the domain grid
        valid_subs, outline_subs, dem_header are from hipims object
            _valid_cell_subs, _outline_cell_subs, _global_header
        cell_subs: (tuple)subscripts of outline boundary cells
        cell_id: (numpy vector)valid id of outline boundary cells
        """
        # to get outline cell id based on _outline_cell_subs
        vector_id = np.arange(valid_subs[0].size)
        nrows = dem_header['nrows']
        ncols = dem_header['ncols']
        cellsize = dem_header['cellsize']
        xllcorner = dem_header['xllcorner']
        yllcorner = dem_header['yllcorner']
        grid_cell_id = np.zeros((nrows, ncols))
        grid_cell_id[valid_subs] = vector_id
        outline_id = grid_cell_id[outline_subs]
        outline_id = outline_id.astype('int64')
        # to get boundary cells based on the spatial extent of each boundary
        bound_cell_x = xllcorner+(outline_subs[1]+0.5)*cellsize
        bound_cell_y = yllcorner+(nrows-outline_subs[0]-0.5) *cellsize
        n = 1 # sequence number of boundaries
        data_table = self.data_table
        cell_subs = []
        cell_id = []
        for n in range(data_table.shape[0]):
            if data_table.extent[n] is None: #outline boundary
                dem_extent = myclass.demHead2Extent(dem_header)
                polyPoints = myclass.makeDiagonalShape(dem_extent)
            elif len(data_table.extent[n]) == 2:
                xyv = data_table.extent[n]
                polyPoints = myclass.makeDiagonalShape([np.min(xyv[:, 0]),
                                                        np.max(xyv[:, 0]),
                                                        np.min(xyv[:, 1]),
                                                        np.max(xyv[:, 1])])
            else:
                polyPoints = data_table.extent[n]
            poly = mplP.Polygon(polyPoints, closed=True)
            bound_cell_xy = np.array([bound_cell_x, bound_cell_y])
            bound_cell_xy = np.transpose(bound_cell_xy)
            ind1 = poly.contains_points(bound_cell_xy)
            row = outline_subs[0][ind1]
            col = outline_subs[1][ind1]
            cell_id.append(outline_id[ind1])
            cell_subs.append((row, col))
        self.cell_subs = cell_subs
        self.cell_id = cell_id

    def _divide_domain(self, hipims_obj):
        """ Create Boundary objects for each sub-domain
        IF hipims_obj has sub sections
        """
        boundary_list = hipims_obj.Boundary.boundary_list
        outline_boundary = hipims_obj.Boundary.outline_boundary
        header_global = hipims_obj._global_header
        outline_subs = hipims_obj._outline_cell_subs
        for i in range(hipims_obj.num_of_sections):
            obj_section = hipims_obj.Sections[i]
            header_local = obj_section.Raster.header
            # convert global subscripts to local
            outline_subs_local = _cell_subs_convertor(
                outline_subs, header_global, header_local, to_global=False)
            valid_subs_local = obj_section._valid_cell_subs
            bound_obj = Boundary(boundary_list, outline_boundary)
            bound_obj._fetch_boundary_cells(
                valid_subs_local, outline_subs_local, header_local)
            obj_section.Boundary = bound_obj
            summary_str = bound_obj.get_summary()
            obj_section.Summary.add_items('Boundary conditions', summary_str)


#===================================Static method==============================
def _cell_subs_convertor(input_cell_subs, header_global,
                         header_local, to_global=True):
    """
    Convert global cell subs to divided local cell subs or the otherwise
    and return output_cell_subs, only rows need to be changed
    input_cell_subs : (tuple) input rows and cols of a grid
    header_global : head information of the global grid
    header_local : head information of the local grid
    to_global : logical values, True (local to global) or
                                False(global to local)
    Return:
        output_cell_subs: (tuple) output rows and cols of a grid
    """
    # X and Y coordinates of the centre of the first cell
    y00_centre_global = header_global['yllcorner']+\
                         (header_global['nrows']+0.5)*header_global['cellsize']
    y00_centre_local = header_local['yllcorner']+\
                        (header_local['nrows']+0.5)*header_local['cellsize']
    row_gap = (y00_centre_global-y00_centre_local)/header_local['cellsize']
    row_gap = round(row_gap)
    rows = input_cell_subs[0]
    cols = input_cell_subs[1]
    if to_global:
        rows = rows+row_gap
        # remove subs out of range of the global DEM
        ind = np.logical_and(rows >= 0, rows < header_global['nrows'])
    else:
        rows = rows-row_gap
        # remove subs out of range of the global DEM
        ind = np.logical_and(rows >= 0, rows < header_local['nrows'])
    rows = rows.astype(cols.dtype)
    rows = rows[ind]
    cols = cols[ind]
    output_cell_subs = (rows, cols)
    return output_cell_subs

def _write_two_arrays(file_name, id_values, bound_id_code=None):
    """Write two arrays: cell_id-value pairs and bound_id-bound_code pairs
    Inputs:
        file_name :  the full file name including path
        id_values: valid cell ID - value pair
        bound_id_code: boundary cell ID - codes pair. If bound_id_code is not
            given, then the second part of the file won't be written (only
            the case for precipitatin_mask.dat)
    """
    if not file_name.endswith('.dat'):
        file_name = file_name+'.dat'
    if id_values.shape[1] == 3:
        fmt = ['%d %g %g']
    elif id_values.shape[1] == 2:
        fmt = ['%d %g']
    else:
        raise ValueError('Please check the shape of the 1st array: id_values')
    fmt = '\n'.join(fmt*id_values.shape[0])
    id_values_str = fmt % tuple(id_values.ravel())
    if bound_id_code is not None:
        fmt = ['%-12d %2d %2d %2d']
        fmt = '\n'.join(fmt*bound_id_code.shape[0])
        bound_id_code_str = fmt % tuple(bound_id_code.ravel())
    with open(file_name, 'w') as file2write:
        file2write.write("$Element Number\n")
        file2write.write("%d\n" % id_values.shape[0])
        file2write.write("$Element_id  Value\n")
        file2write.write(id_values_str)
        if bound_id_code is not None:
            file2write.write("\n$Boundary Numbers\n")
            file2write.write("%d\n" % bound_id_code.shape[0])
            file2write.write("$Element_id  Value\n")
            file2write.write(bound_id_code_str)

def _get_cell_id_array(dem_array):
    """ to generate two arrays with the same size of dem_array:
    1. valid_id: to store valid cell id values (sequence number )
        starting from 0, from bottom, left to right, top
    2. outline_id: to store valid cell id on the boundary cells
    valid_id, outline_id = __get_cell_id_array(dem_array)
    """
    # convert DEM to a two-value array: NaNs and Ones
    # and flip up and down
    dem_array_flip = np.flipud(dem_array*0+1)
    # Return the cumulative sum of array elements over a given axis
    # treating NaNs) as zero.
    nancumsum_vector = np.nancumsum(dem_array_flip)
    # sequence number of valid cells: 0 to number of cells-1
    valid_id = nancumsum_vector-1
    # reshape as an array with the same size of DEM
    valid_id = np.reshape(valid_id, np.shape(dem_array_flip))
    # set NaN cells as NaNs
    valid_id[np.isnan(dem_array_flip)] = np.nan
    valid_id = np.flipud(valid_id)
    # find the outline boundary cells
    array_for_outline = dem_array*0
    array_for_outline[np.isnan(dem_array)] = -1
    h_hv = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    # Convolve two array_for_outline arrays
    ind_array = scipy.signal.convolve2d(array_for_outline, h_hv, mode='same')
    ind_array[ind_array < 0] = np.nan
    ind_array[0, :] = np.nan
    ind_array[-1, :] = np.nan
    ind_array[:, 0] = np.nan
    ind_array[:, -1] = np.nan
    # extract the outline cells by a combination
    ind_array = np.isnan(ind_array) & ~np.isnan(dem_array)
    # boundary cells with valid cell id are extracted
    outline_id = dem_array*0-2 # default inner cells value:-2
    outline_id[ind_array] = 0 # outline cells:0
    return valid_id, outline_id

def _get_split_rows(input_array, num_of_sections):
    """ Split array by the number of valid cells (not NaNs) on each rows
    input_array : an array with some NaNs
    num_of_sections : (int) number of sections that the array to be splited
    return split_rows : a list of row subscripts to split the array
    Split from bottom to top
    """
    valid_cells = ~np.isnan(input_array)
    # valid_cells_count by rows
    valid_cells_count = np.sum(valid_cells, axis=1)
    valid_cells_count = np.cumsum(valid_cells_count)
    split_rows = []  # subscripts of the split row [0, 1,...]
    total_valid_cells = valid_cells_count[-1]
    for i in np.arange(num_of_sections-1): 
        num_section_cells = total_valid_cells*(i+1)/num_of_sections
        split_row = np.sum(valid_cells_count<=num_section_cells)-1
        split_rows.append(split_row)
    # sort from bottom to top
    split_rows.sort(reverse=True)
    return split_rows

def _split_array_by_rows(input_array, header, split_rows, overlayed_rows=2):
    """ Clip an array into small ones according to the seperating rows
    input_array : the DEM array
    header : the DEM header
    split_rows : a list of row subscripts to split the array
    Split from bottom to top
    Return array_local, header_local: lists to store local DEM array and header
    """
    header_global = header
    end_row = header_global['nrows']-1
    overlayed_rows = 2
    array_local = []
    header_local = []
    section_sequence = np.arange(len(split_rows)+1)
    for i in section_sequence:  # from bottom to top
        section_id = i
        if section_id == section_sequence.max(): # the top section
            start_row = 0
        else:
            start_row = split_rows[i]-overlayed_rows
        if section_id == 0: # the bottom section
            end_row = header_global['nrows']-1
        else:
            end_row = split_rows[i-1]+overlayed_rows-1
        sub_array = input_array[start_row:end_row+1, :]
        array_local.append(sub_array)
        sub_header = header_global.copy()
        sub_header['nrows'] = sub_array.shape[0]
        sub_yllcorner = (header_global['yllcorner']+
                         (header_global['nrows']-1-end_row)*
                         header_global['cellsize'])
        sub_header['yllcorner'] = sub_yllcorner
        header_local.append(sub_header)
    return array_local, header_local

# private function called by Class Boundary
def _setup_boundary_data_table(boundary_list, outline_boundary='open'):
    """ Initialize boundary data table based on boundary_list
    Add attributes type, extent, hSources, hUSources
    boundary_list: (list) of dict with keys polyPoints, h, hU
    outline_boundary: (str) 'open' or 'rigid'
    """
    data_table = pd.DataFrame(columns=['type', 'extent',
                                       'hSources', 'hUSources',
                                       'h_code', 'hU_code'])
    # set default outline boundary [0]
    if outline_boundary == 'open':
        hSources = np.array([[0, 0], [1, 0]])
        hUSources = np.array([[0, 0, 0], [1, 0, 0]])
        data_table = data_table.append({'type':'open', 'extent':None,
                                        'hSources':hSources,
                                        'hUSources':hUSources},
                                       ignore_index=True)
    elif outline_boundary == 'rigid':
        data_table = data_table.append({'type':'rigid', 'extent':None,
                                        'hSources':None, 'hUSources':None},
                                       ignore_index=True)
    else:
        raise ValueError("outline_boundary must be either open or rigid!")
    # convert boundary_list to a dataframe
    bound_ind = 1  # bound index
    if boundary_list is None:
        boundary_list = []
    for one_bound in boundary_list:
        # Only a bound with polyPoints can be regarded as a boundary
        if ('polyPoints' in one_bound.keys()) and \
               (type(one_bound['polyPoints']) is np.ndarray):
            data_table = data_table.append(
                {'extent':one_bound['polyPoints'], }, ignore_index=True)
            data_table.type[bound_ind] = one_bound['type']
            if 'h' in one_bound.keys():
                data_table.hSources[bound_ind] = np.array(one_bound['h'])
            else:
                data_table.hSources[bound_ind] = None
            if 'hU' in one_bound.keys():
                data_table.hUSources[bound_ind] = np.array(one_bound['hU'])
            else:
                data_table.hUSources[bound_ind] = None
            bound_ind = bound_ind+1
        else:
            warning_str = ('The boundary without polyPoints is ignored: '+
                           str(bound_ind-1))
            warnings.warn(warning_str)
    return data_table

# private function called by Class Boundary
def _get_boundary_code(boudnary_data_table):
    """ Get the 3-element boundary code for h and hU
    boudnary_data_table: the boundary data table with
        columns ['type', 'hSources', 'hUSources']
    Return a new data_table added with h_code, hU_code, and description
    """
#        Get the three column boundary code
        #default outline boundary
    data_table = boudnary_data_table
    num_of_bound = data_table.shape[0]
    description = []
    n_seq = 0  # sequence of boundary
    m_h = 0  # sequence of boundary with IO source files
    m_hu = 0
    for n_seq in range(num_of_bound):
        description1 = data_table.type[n_seq]
        data_table.h_code[n_seq] = np.array([[2, 0, 0]])
        if data_table.type[n_seq] == 'rigid':
            data_table.hU_code[n_seq] = np.array([[2, 2, 0]])
        else:
            data_table.hU_code[n_seq] = np.array([[2, 1, 0]])
            h_sources = data_table.hSources[n_seq]
            hU_sources = data_table.hUSources[n_seq]
            if h_sources is not None:
                h_sources = np.unique(h_sources[:, 1:])
                data_table.h_code[n_seq] = np.array([[3, 0, m_h]]) #[3 0 m]
                if h_sources.size == 1 and h_sources[0] == 0:
                    description1 = description1+', h given as zero'
                else:
                    description1 = description1+', h given'
                m_h = m_h+1
            if hU_sources is not None:
                hU_sources = np.unique(hU_sources[:, 1:])
                data_table.hU_code[n_seq] = np.array([[3, 0, m_hu]]) #[3 0 m]
                if hU_sources.size == 1 and hU_sources[0] == 0:
                    description1 = description1+', hU given as zero'
                else:
                    description1 = description1+', hU given'
                m_hu = m_hu+1
        description.append(description1)
    description[0] = '(outline) '+ description[0] # indicate outline boundary
    data_table['description'] = description
    return data_table

#%create IO Folders for each case
def _create_io_folders(case_folder, make_dir=False):
    """ create Input-Output path for a Hipims case
        (compatible for single/multi-GPU)
    Return:
      dir_input, dir_output, dir_mesh, dir_field
    """
    folder_name = case_folder
    if not folder_name.endswith('/'):
        folder_name = folder_name+'/'
    dir_input = folder_name+'input/'
    dir_output = folder_name+'output/'
    if not os.path.exists(dir_output) and make_dir:
        os.makedirs(dir_output)
    if not os.path.exists(dir_input) and make_dir:
        os.makedirs(dir_input)
    dir_mesh = dir_input+'mesh/'
    if not os.path.exists(dir_mesh) and make_dir:
        os.makedirs(dir_mesh)
    dir_field = dir_input+'field/'
    if not os.path.exists(dir_field) and make_dir:
        os.makedirs(dir_field)
    data_folders = {'input':dir_input, 'output':dir_output,
                    'mesh':dir_mesh, 'field':dir_field}
    return data_folders

def _check_rainfall_rate_values(rain_source, times_in_1st_col=True):
    """ Check the rainfall rate values in rain source array
    rain_source: (numpy array) rainfall source array
          The 1st column is usually time series in seconds, from the 2nd column
          towards end columns are rainfall rate in m/s
    times_in_1st_col: indicate whether the first column is times
    Return:
        values_max: maximum rainfall rate in mm/h
        values_mean: average rainfall rate in mm/h
    """
    # get the pure rainfall rate values
    if times_in_1st_col:
        rain_values = rain_source[:, 1:]
        time_series = rain_source[:, 0]
    else:
        rain_values = rain_source
        time_series = np.arange(rain_values.shape[0])
    # convert the unit of rain rate values from m/s to mm/h
    rain_values_mmh = rain_values*3600*1000
    values_max = rain_values_mmh.max()
    values_mean = rain_values.mean(axis=1)
    rain_total_amount = np.trapz(y=values_mean, x=time_series) # in meter
    duration = np.ptp(time_series)
    rain_rate_mean = rain_total_amount*1000/(duration/3600) #mm/h
    if values_max > 100 or rain_rate_mean > 50:
        warnings.warn('Very large rainfall rates, better check your data!')
        print('Max rain: {:.2f} mm/h, Average rain: {:.2f} mm/h'.\
              format(values_max, rain_rate_mean))
    return values_max, rain_rate_mean

#%% ***************************************************************************
# *************************Public functions************************************
def write_times_setup(case_folder=None, num_of_sections=1, time_values=None):
    """
    Generate a times_setup.dat file. The file contains numbers representing
    the start time, end time, output interval, and backup interval in seconds
    time_values: array or list of int/float, representing time in seconds,
        default values are [0, 3600, 1800, 3600]
    """
    if case_folder is None:
        case_folder = os.getcwd()
    if not case_folder.endswith('/'):
        case_folder = case_folder+'/'
    if time_values is None:
        time_values = np.array([0, 3600, 1800, 3600])
    time_values = np.array(time_values)
    time_values = time_values.reshape((1, time_values.size))
    if num_of_sections == 1:
        np.savetxt(case_folder+'/input/times_setup.dat', time_values, fmt='%g')
    else:
        np.savetxt(case_folder+'/times_setup.dat', time_values, fmt='%g')
    print('times_setup.dat created')

def write_device_setup(case_folder=None,
                       num_of_sections=1, device_values=None):
    """
    Generate a device_setup.dat file. The file contains numbers representing
    the GPU number for each section
    case_folder: string, the path of model
    num_of_sections: int, the number of GPUs to use
    device_values: array or list of int, representing the GPU number
    """
    if case_folder is None:
        case_folder = os.getcwd()
    if device_values is None:
        device_values = np.array(range(num_of_sections))
    device_values = np.array(device_values)
    device_values = device_values.reshape((1, device_values.size))
    if num_of_sections == 1:
        np.savetxt(case_folder+'/input/device_setup.dat',
                   device_values, fmt='%g')
    else:
        np.savetxt(case_folder+'/device_setup.dat', device_values, fmt='%g')
    print('device_setup.dat created')

def write_rain_source(rain_source, case_folder=None, num_of_sections=1):
    """ Write rainfall sources [Independent function from hipims class]
    rain_source: numpy array, The 1st column is time in seconds, the 2nd
        towards the end columns are rainfall rate in m/s for each source ID in
        rainfall mask array
    if for multiple GPU, then copy the rain source file to all domain folders
    case_folder: string, the path of model
    """
    rain_source = np.array(rain_source)
    # check rainfall source value to avoid very large raifall rates
    _check_rainfall_rate_values(rain_source)
    if case_folder is None:
        case_folder = os.getcwd()
    if not case_folder.endswith('/'):
        case_folder = case_folder+'/'
    fmt1 = '%g'  # for the first col: times in seconds
    fmt2 = '%.8e'  # for the rest array for rainfall rate m/s
    num_mask_cells = rain_source.shape[1]-1
    format_spec = [fmt2]*num_mask_cells
    format_spec.insert(0, fmt1)
    if num_of_sections == 1: # single GPU
        file_name = case_folder+'input/field/precipitation_source_all.dat'
    else: # multi-GPU
        file_name = case_folder+'0/input/field/precipitation_source_all.dat'
    with open(file_name, 'w') as file2write:
        file2write.write("%d\n" % num_mask_cells)
        np.savetxt(file2write, rain_source, fmt=format_spec, delimiter=' ')
    if num_of_sections > 1:
        for i in np.arange(1, num_of_sections):
            field_dir = case_folder+str(i)+'/input/field/'
            shutil.copy2(file_name, field_dir)
    print('precipitation_source_all.dat created')

#%% model information summary
class ModelSummary:
    """ Store and disply all model information including:
    case_folder
    Domain area
    Grid size
    Number of Section
    Initial condition
    Boundary condition
    Rainfall data
    Parameters
    """
    #%======================== initialization function ========================
    def __init__(self, hipims_obj):
        num_valid_cells = hipims_obj._valid_cell_subs[0].size
        self.__case_folder = hipims_obj.case_folder
        num_of_sections = hipims_obj.num_of_sections
        dem_header = hipims_obj.Raster.header
        self.domain_area = num_valid_cells*(dem_header['cellsize']**2)
        self.information_dict = {
            '*******************':'Model summary***************',
            'Case folder':self.__case_folder,
            'Number of Sections':str(num_of_sections),
            'Grid size':'{:d} rows * {:d} cols, {:.2f}m resolution'.format(
                dem_header['nrows'], dem_header['ncols'],
                dem_header['cellsize']),
            'Domain area':'{1:,} m^2 with {0:,} valid cells'.format(
                num_valid_cells, self.domain_area)}
        if hasattr(hipims_obj, 'section_id'): # sub-domain object
            self.information_dict['*******************'] = \
                'Section summary**************'
            self.add_items('Domain ID', '{:d}'.format(hipims_obj.section_id))
        else:
            # add all param information if hipims_obj is not a child object
            for key, value in hipims_obj.attributes.items():
                self.add_param_infor(key, value)

    def display(self):
        """
        Display the model summary information
        """
        for key in self.information_dict.keys():
            if key.endswith('-') or key.endswith('*'):
                print(key+self.information_dict[key])
            else:
                print(key+': '+self.information_dict[key])
        print('***********************************************')

    def write_readme(self, filename=None):
        """
        Write readme file for the summary information
        """
        if filename is None:
            filename = self.__case_folder+'/readme.txt'
        with open(filename, 'w') as file2write:
            for key, value in self.information_dict.items():
                file2write.write(key+': '+value+'\n')

    def set_param(self, param_name, param_value):
        """ Set values in the information_dict
        """
        if param_name in self.information_dict.keys():
            self.information_dict[param_name] = param_value
        else:
            raise ValueError(param_name+ ' is not defined in the object')

    def add_items(self, item_name, item_value):
        """ Add a new item to the information_dict
        """
        if type(item_value) is not str:
            item_value = str(item_value)
        self.information_dict[item_name] = item_value

    def add_param_infor(self, param_name, param_value):
        """ Add information of hipims model parameters to the information_dict
        """
        if '--------------' not in self.information_dict.keys():
            self.add_items('--------------', '-----Parameters----------------')
        param_value = np.array(param_value)
        item_name = param_name
        if param_value.size == 1:
            item_value = ' {:} for all cells'.format(param_value)
        else:
            if param_name in ['h0', 'hU0x', 'hU0y']:
                num_wet_cells = np.sum(param_value > 0)
                num_wet_cells_rate = num_wet_cells/param_value.size
                item_value = ' Wet cells ratio: {:.2f}%'.format(
                    num_wet_cells_rate*100)
            elif param_name == 'precipitation_mask':
                if param_value.dtype != 'int32':
                    param_value = param_value.astype('int32')
                item_value = '{:d} rainfall sources'.format(
                    param_value.max()+1)
            elif param_name == 'precipitation_source':
                rain_max, rain_mean = _check_rainfall_rate_values(param_value)
                display_str = 'max rain: {:.2f} mm/h, '+\
                                'average rain: {:.2f} mm/h'
                item_value = display_str.format(rain_max, rain_mean)
            elif param_name == 'gauges_pos':
                item_value = '{:d} gauges'.format(param_value.shape[0])
            else:
                item_numbers, item_number_counts = \
                    np.unique(param_value, return_counts=True)
                ratio = item_number_counts*100/param_value.size
                formatter = {'float_kind':lambda ratio: "%.3f" % ratio}
                ratio_str = np.array2string(ratio, formatter=formatter)+'%'
                Values_str = ' Values{:}, Ratios'.format(item_numbers)
                item_value = Values_str+ratio_str
        self.information_dict[item_name] = item_value

    def save_object(self, file_name, compression=True):
        """ Save Summary object
        """
        # Overwrites any existing file.
        save_object(self, file_name, compression)

def load_object(file_name):
    """ Read a pickle file as an InputHipims object
    """
    #read an InputHipims object file
    with open(file_name, 'rb') as input_file:
        obj = pickle.load(input_file)
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

#%% Displays or updates a console progress bar
def progress_display(total, progress, file_tag, time_left):
    """
    Displays or updates a console progress bar.
    """
    if total == progress:
        file_tag = 'finished'
    else:
        file_tag = file_tag+'...'
    bar_length, status = 50, ""
    progress = float(progress) / float(total)
    if progress >= 1.:
        progress, status = 1, "\r\n"
    block = int(round(bar_length * progress))
    text = "\r|{}| {:.0f}% {:<16} time left: {:.0f}s {}".format(
        chr(9608) * block + "-" * (bar_length - block),
        round(progress * 100, 0),
        file_tag, time_left, status)
    sys.stdout.write(text)
    sys.stdout.flush()
