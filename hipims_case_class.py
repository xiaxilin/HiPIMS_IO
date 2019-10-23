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
import os
import sys
import warnings
import shutil                
import scipy.signal  # to call scipy.signal.convolve2d to get outline boundary 
import pickle
import myclass  # to call class Raster
import numpy as np
import pandas as pd
import matplotlib.patches as mplP
#%% grid data for HiPIMS input format
class InputHipims(object):
    """To define input files for a HiPIMS flood model case
    To read data, process data, write input files and save data of a model case.
    Properties (public):
        case_folder: (str) the absolute path of the case folder
        data_folders: (dict) paths for data folders (input,output,mesh,field)
        num_of_sections: (scalar) number of GPUs to run the model
        Boundary: a boundary object of class _Boundary
        Raster: a raster object to provide DEM data
        Summary: an Model_Summary object to record model information 
        Sections: a list of objects of sub-class InputHipims_MG
        attributes_default: (dict) default model attribute names and values
        attributes: (dict) model attribute names and values        
    Properties (Private):
        __valid_cell_subs: (tuple,int numpy) two numpy array indicating rows 
            and cols of valid cells on the DEM grid
        __outline_cell_subs: (tuple,int numpy) two numpy array indicating rows
            and cols of outline cells on the DEM grid
        __global_header: (dict) header of the DEM for the whole model domain
    Methods (public):
        set_parameter: set grid-based parameters
        set_boundary_condition: set boundary condition with a boundary list
        set_rainfall: set rainfall mask and sources
        write_input_files:
        write_grid_files: write grid-based data files
        write_boundary_conditions: write boundary sources
        write_gauges_position: write coordinates of monitoring gauges
        write_halo_file: write overlayed cell ID for multiple GPU cases
    Methods(private):
        __get_cell_subs:
        __divide_grid: split model domain:
        __get_boundary_id_code_array:
        __get_vector_value:
        __write_grid_files:
        __write_boundary_conditions:
        __write_gauge_pos:
        __copy_to_all_sections:
        __initialize_summary_obj:
    Classes:
        InputHipims_MG: child class of InputHiPIMS, provide information
            of each sub-domain 
        _Boundary: provide information of boundary conditions
        Model_Summary: record basic information of an object of InputHiPIMS       
    """
    # default parameters
    attributes_default = {'h0':0,'hU0x':0,'hU0y':0,
                       'precipitation_mask':0,
                       'precipitation_source':np.array([[0,0],[1,0]]),
                       'manning':0.035,
                       'sewer_sink':0,
                       'cumulative_depth':0,'hydraulic_conductivity':0,
                       'capillary_head':0,'water_content_diff':0,
                       'gauges_pos':np.array([[0,0]])}
    def __init__(self,dem_array=None,header=None,num_of_sections=1,
                 dem_file=None,case_folder=None):
        """
        dem_array: (numpy float array) dem array
        header:  (dict) header of the dem file
        dem_file: (string) provide filename of the DEM data if dem_array
                 and header are not given        
        """
        if case_folder is None: 
            case_folder=os.getcwd()
        self.case_folder = case_folder
        if num_of_sections>1:
            make_dir=False
        else:
            make_dir=True
        self.data_folders = _create_IO_folders(case_folder,make_dir)
        self.num_of_sections = num_of_sections
        self.attributes = InputHipims.attributes_default.copy()
        if dem_file is not None:
            self.Raster = myclass.raster(dem_file)
        else:
            self.Raster = myclass.raster(array=dem_array,header=header)
        # get row and col index of all cells on DEM grid
        self.__get_cell_subs()  # add __valid_cell_subs and __outline_cell_subs          
        # divide model domain to several sections if it is not a sub section        
        # each section contains a "HiPIMS_IO_class.InputHipims_MG" object 
        self.__divide_grid()
        # get a Model Summary object
        self.__initialize_summary_obj()
        self.__global_header = self.Raster.header

#******************************************************************************
#************************Setup the object***************************************            
    def set_boundary_condition(self,boundary_list=None,
                               outline_boundary='open'):
        """
        create a boundary object for boundary conditions, containing 
              outline_boundary, a dataframe of boundary type, extent, 
              source data, code, ect...,
        and a boundary subscrpits tuple (cell_subs)
        If the number of section is larger than 1, then a boundary subscrpits 
        tuple (cell_subs_l) based on sub-grids will be created for each section
        outline_boundary: 'open' or 'rigid'
        boundary_list: a list of dict, each dict contain keys (polyPoints,type,h,hU)
            to define a IO boundary's position, type, and Input-Output (IO) sources 
            1.polyPoints is a numpy array giving X(1st col) and Y(2nd col) 
                coordinates of points to define the position of a boundary. 
                An empty polyPoints means outline boundary.
            2.type: 'open'|'rigid'
            3.h: a two-col numpy array. The 1st col is time(s). The 2nd col is 
                water depth(m)
            4.hU: a two-col numpy array. The 1st col is time(s). The 2nd col is 
                discharge(m3/s) or a three-col numpy array, the 2nd col and the
                3rd col are velocities(m/s) in x and y direction, respectively.
        """
        bound_object = _Boundary(boundary_list,
                                outline_boundary=outline_boundary)
        vector_id = np.arange(self.__valid_cell_subs[0].size)
        grid_cell_id = self.Raster.array*0
        grid_cell_id[self.__valid_cell_subs]=vector_id
        outline_id = grid_cell_id[self.__outline_cell_subs]
        outline_id = outline_id.astype('int64')
        bound_object.fetch_boundary_cells(self.__outline_cell_subs,outline_id,
                                          self.__global_header)
        self.Boundary = bound_object
        self.Boundary.print_summary()
        if self.num_of_sections>1:
            header_global = self.__global_header
            outline_cell_subs = self.__outline_cell_subs
            for i in range(self.num_of_sections):
                obj_section = self.Sections[i]
                header_local = obj_section.Raster.header
                # convert global subscripts to local
                outline_subs_local = \
                          _cell_subs_convertor(outline_cell_subs,header_global,
                                               header_local,to_global=False)
                bound_object = _Boundary(boundary_list,
                                         outline_boundary=outline_boundary)
                vector_id = np.arange(obj_section.__valid_cell_subs[0].size)
                grid_cell_id = obj_section.Raster.array*0
                grid_cell_id[obj_section.__valid_cell_subs]=vector_id
                outline_id_local = grid_cell_id[outline_subs_local]
                outline_id_local=outline_id_local.astype('int64')
                bound_object.fetch_boundary_cells(outline_subs_local,
                                                  outline_id_local,
                                                  header_local)
                obj_section.Boundary = bound_object
        summary_str = self.Boundary.get_summary()
        self.Summary.add_items('Boundary conditions',summary_str)
        return None
    
    def set_parameter(self,parameter_name,parameter_value):
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
            if parameter_value.shape!=self.Raster.array.shape:
                raise ValueError('The array of the parameter '
                                 'value should have the same '
                                 'shape with the DEM array')
        elif np.isscalar(parameter_value) is False:
            raise ValueError('The parameter value must be either', 
                             'a scalar or an numpy array')
        self.attributes[parameter_name] = parameter_value
        # renew summary information
        self.Summary.add_param_infor(parameter_name,parameter_value)
        return None
    
    def set_rainfall(self,rain_mask=None,rain_source=None):
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
        num_of_masks = np.unique(rain_mask).size
        if rain_source.shape[1]-1 != num_of_masks:
            raise ValueError('The column of rain source', 
                             'is not consistent with the number of rain masks')
        _check_rainfall_rate_values(rain_source,times_in_1st_col=True)        
        self.Summary.add_param_infor('precipitation_mask',rain_mask)
        if rain_source is not None:
            self.attributes['precipitation_source'] = rain_source
            self.Summary.add_param_infor('precipitation_source',rain_source)
        # renew summary information 
        return None
    def set_gauges_position(self,gauges_pos):
        if type(gauges_pos) is list:
            gauges_pos = np.array(gauges_pos)
        if gauges_pos.shape[1]!=2:
            raise ValueError('The gauges_pos arraymust have two columns')
        self.attributes['gauges_pos'] = gauges_pos
        self.Summary.add_param_infor('gauges_pos',gauges_pos)
            
    def write_input_files(self,file_tag=None):
        """ Write input files
        To classify the input files and call functions needed to write each
            input files
        file_tag: 'all'|'z','h','hU','manning','sewer_sink',
                        'cumulative_depth','hydraulic_conductivity',
                        'capillary_head','water_content_diff'
                        'precipitation_mask','precipitation_source',
                        'boundary_condition','gauges_pos'        
        """
        grid_files = ['z','h','hU','precipitation_mask',
                             'manning','sewer_sink',
                             'cumulative_depth', 'hydraulic_conductivity',
                             'capillary_head', 'water_content_diff']
        if file_tag is None or file_tag=='all':
            for grid_file in grid_files: # grid-based files
                self.write_grid_files(grid_file)
            self.write_boundary_conditions()
            self.write_rainfall_source()
            self.write_gauges_position()
            if self.num_of_sections>1:
                self.write_halo_file()
            self.write_mesh_file(self)
            write_times_setup(self.case_folder,self.num_of_sections)
            write_device_setup(self.case_folder,self.num_of_sections)
        elif file_tag=='boundary_condition':
            self.write_boundary_conditions()
        elif file_tag=='gauges_pos':
            self.write_gauges_position()
        elif file_tag=='halo':
            self.write_halo_file()
        else:
            raise ValueError('file_tag is not recognized')
        return None
    
    def write_grid_files(self,file_tag,singleGPU=False):
        """Write grid-based files
        Public version for both single and multiple GPUs
        file_tag: the pure name of a grid-based file
        """
        if file_tag not in ['z','h','hU','manning','sewer_sink',
                        'cumulative_depth', 'precipitation_mask',
                        'hydraulic_conductivity',
                        'capillary_head', 'water_content_diff']:
            raise ValueError(file_tag+' is not a grid-based file')
        if singleGPU or self.num_of_sections==1: 
            # write as single GPU even the num of sections is more than one
            self.__write_grid_files(file_tag,multi_GPU=False)
        else:
            self.__write_grid_files(file_tag,multi_GPU=True)
            
    def write_boundary_conditions(self):
        """ Write boundary condtion files
        if there are multiple domains, write in the first folder 
            and copy to others
        """
        if self.num_of_sections>1:  # multiple-GPU
            field_dir = self.Sections[0].data_folders['field']
            file_names_list = self.__write_boundary_conditions(field_dir)
            self.__copy_to_all_sections(file_names_list)
        else:  # single-GPU
            field_dir = self.data_folders['field']
            _ = self.__write_boundary_conditions(field_dir)
    
    def write_rainfall_source(self):
        """Write rainfall source data
        rainfall mask can be written by function write_grid_files
        """
        rain_source = self.attributes['precipitation_source']
        case_folder = self.case_folder
        num_of_sections = self.num_of_sections
        write_rain_source(rain_source,case_folder,num_of_sections)
        return None
    
    def write_gauges_position(self,gauges_pos=None):
        """ Write the gauges position file
        Public version for both single and multiple GPUs
        """
        if gauges_pos is not None:
            self.gauges_pos = np.array(gauges_pos)
        if self.num_of_sections>1:  # multiple-GPU
            field_dir = self.Sections[0].data_folders['field']
            file_name = self.__write_gauge_pos(field_dir)
            self.__copy_to_all_sections(file_name)
        else:  # single-GPU
            field_dir = self.data_folders['field']
            _ = self.__write_gauge_pos(field_dir)        

    def write_halo_file(self):
        """ Write overlayed cell IDs
        """
        num_section = self.num_of_sections
        case_folder = self.case_folder
        if not case_folder.endswith('/'):
            case_folder = case_folder+'/'
        file_name = case_folder+'halo.dat'
        with open(file_name,'w') as file2write:
            file2write.write("No. of Domains\n")
            file2write.write("%d\n" % num_section)
            for obj_section in self.Sections:
                file2write.write("#%d\n" % obj_section.section_NO )
                overlayed_id = obj_section.overlayed_id
                for key in ['bottom_low','bottom_high','top_high','top_low']:                    
                    if key in overlayed_id.keys():
                        line_ids = overlayed_id[key]
                        line_ids = np.reshape(line_ids,(1,line_ids.size))
                        np.savetxt(file2write,line_ids,fmt='%d', delimiter=' ')
                    else:
                        file2write.write(' \n')                        
        return None
    
    def write_mesh_file(self,singleGPU=False):
        """ Write mesh file DEM.txt, compatoble for both single and multiple
        GPU model
        """
        if singleGPU is True or self.num_of_sections==1:
            file_name = self.data_folders['mesh']+'DEM.txt'
            self.Raster.Write_asc(file_name)
        else:
            for obj_section in self.Sections:
                file_name = obj_section.data_folders['mesh']+'DEM.txt'
                obj_section.Raster.Write_asc(file_name)
    
    def save_object(self, file_name):
        """ Save object as a pickle file 
        """
        if not file_name.endswith('.pickle'):
            file_name = file_name+'.pickle'
        with open(file_name, 'wb') as output:  # Overwrites any existing file.
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)
                    
#%========================private method==================================    
    def __get_cell_subs(self,dem_array=None):
        """ To get valid_cell_subs and outline_cell_subs for the object
        To get the subscripts of each valid cell on grid
        Input arguments are for sub Hipims objects
        __valid_cell_subs
        __outline_cell_subs
        """
        if dem_array is None:
            dem_array = self.Raster.array
        valid_id,outline_id = _get_cell_id_array(dem_array)
        subs = np.where(~np.isnan(valid_id))
        id_vector = valid_id[subs]
        # sort the subscripts according to cell id values
        sorted_vectors = np.c_[id_vector,subs[0],subs[1]]
        sorted_vectors = sorted_vectors[sorted_vectors[:,0].argsort()]
        self.__valid_cell_subs = (sorted_vectors[:,1].astype('int32'),
                                sorted_vectors[:,2].astype('int32'))
        subs = np.where(outline_id==0) # outline boundary cell
        outline_id_vect = outline_id[subs]
        sorted_array = np.c_[outline_id_vect,subs[0],subs[1]]
        self.__outline_cell_subs = (sorted_array[:,1].astype('int32'),
                                  sorted_array[:,2].astype('int32'))
        return None

    def __divide_grid(self):
        """
        Divide DEM grid to sub grids
        Create objects based on sub-class InputHipims_MG
        """
        num_of_sections = self.num_of_sections
        dem_header = self.Raster.header
        if num_of_sections>1:
            if hasattr(self, 'section_NO'):
                return 1
            else: # continue to do 
                self.__global_header = dem_header
        else:
            return 1
        # subscripts of the split row [0,1,...] from bottom to top
        split_rows = _get_split_rows(self.Raster.array,num_of_sections)
        array_local,header_local = _split_array_by_rows(self.Raster.array,
                                                        dem_header,split_rows)
        # to receive InputHipims_MG objects for sections
        Sections = []  
        section_sequence = np.arange(num_of_sections)
        header_global = self.__global_header

        for i in section_sequence:  # from bottom to top
            case_folder = self.case_folder+'/'+str(i)
            # create a sub object of InputHipims
            sub_hipims = InputHipims_MG(array_local[i],header_local[i],
                                              case_folder,num_of_sections)            
            # get valid_cell_subs on the global grid
            valid_cell_subs = sub_hipims.__valid_cell_subs
            valid_subs_global = _cell_subs_convertor(valid_cell_subs,
                                     header_global, header_local[i],
                                     to_global=True)
            sub_hipims.valid_subs_global = valid_subs_global
            # record section sequence number
#            sub_hipims.section_NO = i
            #get overlayed_id (top two rows and bottom two rows)
            top_h = np.where(valid_cell_subs[0]==0)
            top_l = np.where(valid_cell_subs[0]==1)
            bottom_h = np.where(valid_cell_subs[0]==valid_cell_subs[0].max()-1)
            bottom_l = np.where(valid_cell_subs[0]==valid_cell_subs[0].max())
            if i==0: # the bottom section
                overlayed_id = {'top_high':top_h[0], 'top_low':top_l[0]}
            elif i==self.num_of_sections-1: # the top section
                overlayed_id = {'bottom_low':bottom_l[0], 
                                'bottom_high':bottom_h[0]}
            else:
                overlayed_id = {'top_high':top_h[0], 'top_low':top_l[0],
                               'bottom_high':bottom_h[0], 
                               'bottom_low':bottom_l[0]}             
            sub_hipims.overlayed_id=overlayed_id
            Sections.append(sub_hipims)
        # reset global var section_NO of InputHipims_MG
        InputHipims_MG.section_NO = 0
        self.Sections = Sections

    # only for global object  
    def __get_vector_value(self,attribute_name,multi_GPU=True,
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
            water_on_boundary = 0.0001
        else:
            water_on_boundary = 0
        # set grid value for the entire domain
        if attribute_name=='z':
            grid_values = self.Raster.array
        elif attribute_name=='h':
            grid_values = grid_values+self.attributes['h0']
            # traversal each boundary to add initial water
            for ind_num in np.arange(self.Boundary.num_of_bound):
                h_source = self.Boundary.data_table['hSources'][ind_num]
                if h_source is not None:
                    source_value = np.unique(h_source[:,1:])
                    # zero boundary conditions 
                    if not (source_value.size==1 and source_value[0]==0):
                        cell_subs = self.Boundary.cell_subs[ind_num]
                        grid_values[cell_subs] = water_on_boundary
        elif attribute_name=='hU':
            grid_values1 = grid_values+self.attributes['hU0y']
            for ind_num in np.arange(self.Boundary.num_of_bound):
                hU_source = self.Boundary.data_table['hUSources'][ind_num]
                if hU_source is not None:
                    source_value = np.unique(hU_source[:,1:])
                    # zero boundary conditions 
                    if not (source_value.size==1 and source_value[0]==0):
                        cell_subs = self.Boundary.cell_subs[ind_num]
                        grid_values1[cell_subs] = water_on_boundary
            grid_values0 = grid_values+self.attributes['hU0x']
            grid_values = [grid_values0,grid_values1]
        else:
            grid_values = grid_values+self.attributes[attribute_name]
        
        # define a function to convert grid value to vector value
        def grid_to_vect(grid_values,cell_subs):
            if type(grid_values) is list:
                vector_value0 = grid_values[0][cell_subs]
                vector_value1 = grid_values[1][cell_subs]
                vector_value = np.c_[vector_value0,vector_value1]
            else:
                vector_value = grid_values[cell_subs]
            return vector_value
        #
        if multi_GPU: # generate vector value for multiple GPU
            output_vector = []
            for obj_section in self.Sections:
                cell_subs = obj_section.valid_subs_global
                vector_value = grid_to_vect(grid_values,cell_subs)
                output_vector.append(vector_value)
        else:
            output_vector = grid_to_vect(grid_values, self.__valid_cell_subs)
        return output_vector
    
    def __get_boundary_id_code_array(self,file_tag = 'z'):
        """
        To generate a 4-col array of boundary cell id (0) and code (1~3)
        """
        bound_obj = self.Boundary
        output_array_list = []
        for ind_num in np.arange(bound_obj.num_of_bound):
            if file_tag=='h':
                bound_code = bound_obj.data_table.h_code[ind_num]
            elif file_tag=='hU':
                bound_code = bound_obj.data_table.hU_code[ind_num]
            else:
                bound_code = np.array([[2,0,0]]) # shape (1,3)
            if bound_code.ndim<2:
                bound_code = np.reshape(bound_code,(1,bound_code.size))
            cell_id = bound_obj.cell_id[ind_num]
            if cell_id.size>0:
               bound_code_array = np.repeat(bound_code,cell_id.size,axis=0)
               id_code_array = np.c_[cell_id,bound_code_array]
               output_array_list.append(id_code_array)
        # add overlayed cells with [4,0,0]       
        # if it is a sub section object, there should be attributes:
        # overlayed_id,and section_NO
        if hasattr(self, 'overlayed_id'):
            cell_id = list(self.overlayed_id.values())    
            cell_id = np.concatenate(cell_id, axis=0)
            bound_code = np.array([[4,0,0]]) # shape (1,3)
            bound_code_array = np.repeat(bound_code,cell_id.size,axis=0)
            id_code_array = np.c_[cell_id,bound_code_array]
            output_array_list.append(id_code_array)
        output_array = np.concatenate(output_array_list, axis=0)
        # when unique the output array according to cell id
        # keep the last occurrence rather than the default first occurrence
        output_array = np.flipud(output_array) # make the IO boundaries first
        _,ind = np.unique(output_array[:,0],return_index=True)
        output_array = output_array[ind]
        return output_array
    
    def __initialize_summary_obj(self):
        """ Initialize the model summary object
        """
        num_valid_cells = self.__valid_cell_subs[0].size
        case_folder = self.case_folder
        summary_obj = Model_Summary(case_folder=case_folder, 
                                    num_of_sections=self.num_of_sections,
                                    dem_header=self.Raster.header,
                                    num_valid_cells=num_valid_cells)
        if hasattr(self, 'section_NO'): # sub-domain object
            summary_obj.add_items('Domain ID', '{:d}'.format(self.section_NO))
        else: # only show parameter information for the global domain                   
            for key,value in self.attributes.items():
                summary_obj.add_param_infor(key,value)
        summary_obj.add_items('----------------------',
                              '-------------------------')
        self.Summary = summary_obj

#******************************************************************************                            
#*********************Private functions to write input files*******************
#******************************************************************************        
    def __write_grid_files(self,file_tag,multi_GPU=True):
        """ Write input files consistent with the DEM grid
        Private function called by public function write_grid_files
        file_name: includes ['h','hU','precipitation_mask',
                             'manning','sewer_sink',
                             'cumulative_depth', 'hydraulic_conductivity',
                             'capillary_head', 'water_content_diff']
        """
        if multi_GPU is True:  # write for multi-GPU, use child object
            vector_value_list = self.__get_vector_value(file_tag,multi_GPU)
            for obj_section in self.Sections:
                vector_value = vector_value_list[obj_section.section_NO]
                cell_ID = np.arange(vector_value.shape[0])
                cells_vect = np.c_[cell_ID,vector_value]
                file_name = obj_section.data_folders['field']+file_tag+'.dat'
                if file_tag=='precipitation_mask':
                    bounds_vect = None
                else:
                    bounds_vect = \
                        obj_section.__get_boundary_id_code_array(file_tag)
                _write_two_arrays(file_name,cells_vect,bounds_vect)
        else:  # single GPU, use global object
            file_name = self.data_folders['field']+file_tag+'.dat'
            vector_value = self.__get_vector_value(file_tag,multi_GPU=False)
            cell_ID = np.arange(vector_value.shape[0])
            cells_vect = np.c_[cell_ID,vector_value]
            if file_tag=='precipitation_mask':
                bounds_vect = None
            else:
                bounds_vect = self.__get_boundary_id_code_array(file_tag)
                _write_two_arrays(file_name,cells_vect,bounds_vect)
        return None
      
    def __write_boundary_conditions(self,field_dir,file_tag='both'):
        """ Write boundary condition source files
        Private function to call by public function write_boundary_conditions
        file_tag: 'h', 'hU', 'both'
        h_BC_[N].dat, hU_BC_[N].dat
        if hU is given as flow timeseries, convert flow to hUx and hUy
        """
        obj_boundary = self.Boundary
        file_names_list = []
        fmt_h  = ['%g','%g']
        fmt_hU = ['%g','%.8e','%.8e']
        # write h_BC_[N].dat 
        if file_tag=='both' or file_tag=='h':
            h_sources = obj_boundary.data_table['hSources']
            ind_num = 0
            for i in np.arange(obj_boundary.num_of_bound):
                h_source = h_sources[i]
                if h_source is not None:
                    file_name = field_dir+'h_BC_'+str(ind_num)+'.dat'
                    np.savetxt(file_name,h_source,fmt=fmt_h,delimiter=' ')
                    ind_num = ind_num+1
                    file_names_list.append(file_name)
        # write hU_BC_[N].dat            
        if file_tag=='both' or file_tag=='hU':
            hU_sources = obj_boundary.data_table['hUSources']
            ind_num = 0
            for i in np.arange(obj_boundary.num_of_bound):
                hU_source = hU_sources[i]
                cell_subs = obj_boundary.cell_subs[i]
                if hU_source is not None:
                    file_name = field_dir+'hU_BC_'+str(ind_num)+'.dat'
                    if hU_source.shape[1]==2: # flow is given rather than speed
                        boundary_slope = np.polyfit(cell_subs[0],
                                                    cell_subs[1],1)
                        theta = np.arctan(boundary_slope[0])
                        boundary_length = cell_subs[0].size* \
                                          self.Raster.header['cellsize']
                        hUx = hU_source[:,1]*np.cos(theta)/boundary_length
                        hUy = hU_source[:,1]*np.sin(theta)/boundary_length
                        hU_source = np.c_[hU_source[:,0],hUx,hUy]
                        print('Flow series on boundary '+str(i)+
                              ' is converted to velocities')
                        print('Theta = '+'{:.3f}'.format(theta/np.pi)+'pi')                    
                    np.savetxt(file_name,hU_source,fmt=fmt_hU,delimiter=' ')
                    ind_num = ind_num+1
                    file_names_list.append(file_name)            
        return file_names_list   
       
    def __write_gauge_pos(self,file_folder,):
        """write monitoring gauges
        Private version of write_gauge_position
        gauges_pos.dat
        file_folder: folder to write file
        gauges_pos: 2-col numpy array of X and Y coordinates
        """
        gauges_pos = self.attributes['gauges_pos']
        if type(gauges_pos)==list:
            gauges_pos = np.array(gauges_pos,ndmin=2)
        file_name = file_folder+'gauges_pos.dat'
        fmt=['%g %g']
        fmt = '\n'.join(fmt*gauges_pos.shape[0])
        gauges_pos_str = fmt % tuple(gauges_pos.ravel()) 
        with open(file_name,'w') as file2write:
            file2write.write(gauges_pos_str)
        return file_name
    
    def __copy_to_all_sections(self,file_names):
        """ Copy files that are the same in each sections
        file_names: (str) files written in the first seciton [0]
        boundary source files: h_BC_[N].dat,hU_BC_[N].dat
        rainfall source files: precipitation_source_all.dat
        gauges position file: gauges_pos.dat
        """
        if type(file_names) is not list:
            file_names = [file_names]
        for i in np.arange(1,self.num_of_sections):
            field_dir = self.Sections[i].data_folders['field']
            for file in file_names:
                    shutil.copy2(file,field_dir)
        return None
        

#%% sub-class definition
class InputHipims_MG(InputHipims):
    """object for each section, child class of InputHipims
    Attributes:
        sectionNO: the serial number of each section
        __valid_cell_subs: (tuple,int) two numpy array indicating rows and cols of valid
                    cells on the local grid        
        valid_cell_subsOnGlobal: (tuple,int) two numpy array indicating rows and cols of valid
                    cells on the global grid
        shared_cells_id: 2-row shared Cells id on a local grid
        case_folder: input folder of each section
        __outline_cell_subs: (tuple,int) two numpy array indicating rows and cols of valid
                    cells on a local grid
    """
    section_NO = 0
    def __init__(self,dem_array,header,case_folder,num_of_sections):
        self.case_folder = case_folder
        self.data_folders = _create_IO_folders(case_folder,make_dir=True)
        self.num_of_sections = num_of_sections
        self.Raster = myclass.raster(array=dem_array,header=header)
        self.section_NO = InputHipims_MG.section_NO
        InputHipims_MG.section_NO = self.section_NO+1
        # get row and col index of all cells on DEM grid
        self._InputHipims__get_cell_subs()  # add __valid_cell_subs and __outline_cell_subs          
        # get a Model Summary object
        self._InputHipims__initialize_summary_obj()
#        InputHipims.__init__(self,dem_array,header,case_folder=case_folder)
#%% boundary class definition
class _Boundary(object):
    """Private class for boundary conditions
    default outline boundary: IO, h and Q are given as constant 0 values 
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
            h_code:
            hU_code:
            description:
        
    Private Properties:
        code: 3-element row vector for each boundary cell
        
    Methods    
        print_summary: print the summary information of a boundary object
        Gen3Code: Generate 3-element boundary codes
        CellLocate: fine boundary cells with given extent
    """
    def __init__(self,boundary_list=None,outline_boundary = 'open'):
        # setup data_table including attributtes 
        # type, extent, hSources, hUSources
        data_table = _setup_boundary_data_table(boundary_list,outline_boundary)
        # add boundary code 'h_code', 'hU_code' and 'description' to data_table
        data_table = _get_boundary_code(data_table)
        num_of_bound = data_table.shape[0]
        self.data_table = data_table
        self.num_of_bound = num_of_bound
        self.h_sources = data_table['hSources']
        self.hU_sources = data_table['hUSources']
        
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

    def fetch_boundary_cells(self,outline_cell_subs,outline_cell_id,dem_header):
        """
        To get the subs of boundary cells after
        # BoundaryClassify(bnMat_outline,dem_header,boundary_list)
        # get rows and columns of outline bound cells
        # bnMat: nan: invalida cell; -2: non-bound cell; 0: outline cell;
        #               1,~: user-defined IO bound cell on the outline
        """
        R = dem_header
        Extent = myclass.demHead2Extent(dem_header)
        Bound_Cell_X = R['xllcorner']+(outline_cell_subs[1]+0.5)*R['cellsize']
        Bound_Cell_Y = R['yllcorner']+(R['nrows']-outline_cell_subs[0]-0.5)*R['cellsize']    
        outline_cell_subs = np.array([outline_cell_subs[0],outline_cell_subs[1]]) # convert to numpy array
        outline_cell_subs = np.transpose(outline_cell_subs)
        n=1 # sequence number of boundaries
        data_table = self.data_table
        cell_subs = []
        cell_id = []
        for n in range(data_table.shape[0]):        
            if data_table.extent[n] is None: #outline boundary
                polyPoints = myclass.makeDiagonalShape(Extent)
            elif len(data_table.extent[n])==2:
                xyv = data_table.extent[n]
                polyPoints = myclass.makeDiagonalShape([np.min(xyv[:,0]),
                                                np.max(xyv[:,0]),
                                                np.min(xyv[:,1]),
                                                np.max(xyv[:,1])])
            else:
                polyPoints = data_table.extent[n]            
            poly = mplP.Polygon(polyPoints, closed=True)
            Bound_Cell_XY = np.array([Bound_Cell_X,Bound_Cell_Y])
            Bound_Cell_XY = np.transpose(Bound_Cell_XY)
            ind1 = poly.contains_points(Bound_Cell_XY)
            row = outline_cell_subs[ind1,0]
            col = outline_cell_subs[ind1,1]
            cell_id.append(outline_cell_id[ind1])
            cell_subs.append((row,col))
        self.cell_subs = cell_subs
        self.cell_id = cell_id
        return None
#===================================Static method==============================

def _cell_subs_convertor(input_cell_subs,header_global,
                            header_local,to_global=True):
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
        ind = np.logical_and(rows>=0,rows<header_global['nrows'])
    else:
        rows = rows-row_gap
        # remove subs out of range of the global DEM
        ind = np.logical_and(rows>=0,rows<header_local['nrows'])        
    rows = rows.astype(cols.dtype)
    rows = rows[ind]
    cols = cols[ind]
    output_cell_subs = (rows,cols)
    return output_cell_subs

def _write_two_arrays(file_name,id_values,bound_id_code=None):
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
    if id_values.shape[1]==3:
        fmt = ['%d %g %g']
    elif id_values.shape[1]==2:
        fmt = ['%d %g']
    else:
        raise ValueError('Please check the shape of the 1st array: id_values')
    fmt = '\n'.join(fmt*id_values.shape[0])
    id_values_str = fmt % tuple(id_values.ravel())
    if bound_id_code is not None:
        fmt=['%-12d %2d %2d %2d']
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
    return None

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
    valid_id = np.reshape(valid_id,np.shape(dem_array_flip))
    # set NaN cells as NaNs
    valid_id[np.isnan(dem_array_flip)]=np.nan
    valid_id = np.flipud(valid_id)
    
    # find the outline boundary cells
    array_for_outline = dem_array*0
    array_for_outline[np.isnan(dem_array)]=-1
    h_hv   = np.array([[0,1,0], [1,0,1], [0,1,0]])
    # Convolve two array_for_outline arrays
    ind_array = scipy.signal.convolve2d(array_for_outline, h_hv,mode='same')
    ind_array[ind_array<0]=np.nan    
    ind_array[0,:] = np.nan
    ind_array[-1,:] = np.nan
    ind_array[:,0] = np.nan
    ind_array[:,-1] = np.nan
    # extract the outline cells by a combination
    ind_array = np.isnan(ind_array)&~np.isnan(dem_array)
    # boundary cells with valid cell id are extracted
    outline_id = dem_array*0-2 # default inner cells value:-2
    outline_id[ind_array] = 0 # outline cells:0 
    return valid_id,outline_id

def _get_split_rows(input_array,num_of_sections):
    """ Split array by the number of valid cells (not NaNs) on each rows
    input_array : an array with some NaNs
    num_of_sections : (int) number of sections that the array to be splited
    return split_rows : a list of row subscripts to split the array
    Split from bottom to top
    """
    valid_cells = ~np.isnan(input_array)
    # valid_cells_count by rows
    valid_cells_count = np.sum(valid_cells,axis=1)
    valid_cells_count = np.cumsum(valid_cells_count)
    split_rows = []  # subscripts of the split row [0,1,...]
    for i in np.arange(num_of_sections): # from bottom to top
        num_of_sectionsCells =\
            valid_cells_count[-1]*(i+1)/num_of_sections
        splitRow = np.sum(valid_cells_count-num_of_sectionsCells>0)
        split_rows.append(splitRow)
    return split_rows

def _split_array_by_rows(input_array,header,split_rows,overlayed_rows=2):
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
    section_sequence = np.arange(len(split_rows))
    for i in section_sequence:  # from bottom to top
        section_NO = i          
        if section_NO == section_sequence.max(): # the top section
            start_row = 0
        else:
            start_row = split_rows[i]-overlayed_rows    
        if section_NO == 0: # the bottom section
            end_row = header_global['nrows']-1
        else:
            end_row = split_rows[i-1]+overlayed_rows-1        
        sub_array = input_array[start_row:end_row+1,:]
        array_local.append(sub_array)
        sub_header = header_global.copy()
        sub_header['nrows'] = sub_array.shape[0]
        sub_yllcorner = (header_global['yllcorner']+
                         (header_global['nrows']-1-end_row)*
                         header_global['cellsize'])
        sub_header['yllcorner'] = sub_yllcorner
        header_local.append(sub_header)
    return array_local,header_local

# private function called by Class _Boundary

def _setup_boundary_data_table(boundary_list,outline_boundary='open'):
    """ Initialize boundary data table based on boundary_list
    Add attributes type, extent, hSources, hUSources
    boundary_list: (list) of dict with keys polyPoints, h, hU
    outline_boundary: (str) 'open' or 'rigid'     
    """
    
    data_table = pd.DataFrame(columns=['type','extent',
                               'hSources','hUSources',
                               'h_code','hU_code'])
    # set default outline boundary [0]
    if outline_boundary == 'open':
        hSources = np.array([[0,0],[1,0]])
        hUSources = np.array([[0,0,0],[1,0,0]])
        data_table = data_table.append({'type':'open', 'extent':None,
                               'hSources':hSources, 'hUSources':hUSources},
                               ignore_index=True)
    elif outline_boundary == 'rigid':
        data_table = data_table.append({'type':'rigid','extent':None,
                               'hSources':None,'hUSources':None},
                               ignore_index=True)
    else:
        raise ValueError("outline_boundary must be either open or rigid!")
                
    # convert boundary_list to a dataframe
    bound_ind = 1  # bound index
    if boundary_list is None:
        boundary_list = []
    for one_bound in boundary_list:
        # Only a bound with polyPoints can be regarded as a boundary 
        if ('polyPoints' in one_bound.keys() ) and \
               (type(one_bound['polyPoints']) is np.ndarray):                
            data_table = data_table.append(
                    {'extent':one_bound['polyPoints'],},ignore_index=True)
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
             
# private function called by Class _Boundary
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
    n = 0  # sequence of boundary
    m_h = 0  # sequence of boundary with IO source files  
    m_hU = 0
    for n in range(num_of_bound):
        description1 = data_table.type[n]
        data_table.h_code[n] = np.array([[2,0,0]])        
        if data_table.type[n]=='rigid':
            data_table.hU_code[n] = np.array([[2,2,0]])
        else:
            data_table.hU_code[n] = np.array([[2,1,0]])
            h_sources = data_table.hSources[n]
            hU_sources = data_table.hUSources[n]
            if h_sources is not None:
                h_sources = np.unique(h_sources[:,1:])
                data_table.h_code[n] = np.array([[3,0,m_h]]) #[3 0 m]
                if h_sources.size == 1 and h_sources[0] ==0:
                    description1 = description1+', h given as zero'
                else:
                    description1 = description1+', h given'                                
                m_h = m_h+1
            if hU_sources is not None:
                hU_sources = np.unique(hU_sources[:,1:])
                data_table.hU_code[n] = np.array([[3,0,m_hU]]) #[3 0 m]
                if hU_sources.size == 1 and hU_sources[0] ==0:
                    description1 = description1+', hU given as zero'
                else:
                    description1 = description1+', hU given'                    
                m_hU = m_hU+1
        description.append(description1)
    description[0] = '(outline) '+ description[0] # indicate outline boundary
    data_table['description'] = description
    return data_table

#%create IO Folders for each case
def _create_IO_folders(case_folder,make_dir=False):
    """ create Input-Output path for a Hipims case 
        (compatible for single/multi-GPU)
    Return:
      dir_input,dir_output,dir_mesh,dir_field  
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

def _check_rainfall_rate_values(rain_source,times_in_1st_col=True):
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
        rain_values = rain_source[:,1:]
    else:
        rain_values = rain_source
    # convert the unit of rain rate values from m/s to mm/h
    rain_values = rain_values*3600*1000
    values_max  = rain_values.max()
    values_mean = rain_values.mean()
    if values_max>100 or values_mean>50:
        warnings.warn('Very large rainfall rates, better check your data!')
        print('Max rain: {:.2f} mm/h, Average rain: {:.2f} mm/h'.\
              format(values_max,values_mean))
    return values_max,values_mean

#%% ***************************************************************************
# *************************Public functions************************************
def write_times_setup(case_folder=None,num_of_sections=1,time_values=None):
    """
    Generate a times_setup.dat file. The file contains numbers representing
    the start time, end time, output interval, and backup interval in seconds
    time_values: array or list of int/float, representing time in seconds, default
        values are [0,3600,1800,3600]
    """
    if case_folder is None:
        case_folder = os.getcwd()
    if not case_folder.endswith('/'):
        case_folder = case_folder+'/'
    if time_values is None:
        time_values=np.array([0,3600,1800,3600])
    time_values=np.array(time_values)
    time_values = time_values.reshape((1,time_values.size))

    if num_of_sections==1:
        np.savetxt(case_folder+'/input/times_setup.dat',time_values,fmt='%g')
    else:
        np.savetxt(case_folder+'/times_setup.dat',time_values,fmt='%g')
    return None 

def write_device_setup(case_folder=None,num_of_sections=1,device_values=None):
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
        device_values=np.array(range(num_of_sections))
    device_values=np.array(device_values)
    device_values = device_values.reshape((1,device_values.size))
    if num_of_sections==1:
        np.savetxt(case_folder+'/input/device_setup.dat',device_values,fmt='%g')
    else:
        np.savetxt(case_folder+'/device_setup.dat',device_values,fmt='%g')
    return None

def write_rain_source(rain_source,case_folder=None,num_of_sections=1):
    """ Write rainfall sources
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
    format_spec.insert(0,fmt1)
    if num_of_sections==1: # single GPU
        file_name = case_folder+'input/field/precipitation_source_all.dat'        
    else: # multi-GPU
        file_name = case_folder+'0/input/field/precipitation_source_all.dat'
    with open(file_name,'w') as file2write:
        file2write.write("%d\n" % num_mask_cells)
        np.savetxt(file2write,rain_source,fmt=format_spec,delimiter=' ')  
    if num_of_sections>1:
        for i in np.arange(1,num_of_sections):
            field_dir = case_folder+str(i)+'/input/field/'
            shutil.copy2(file_name,field_dir) 
    return None

#%% model information summary
class Model_Summary(object):
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
    #%======================== initialization function ===========================   
    def __init__(self, case_folder, num_of_sections, dem_header, 
                 num_valid_cells,
                 initial_condition=None, boundary_condition=None):
        self.__case_folder = case_folder
        self.__num_valid_cells = num_valid_cells
        self.__domain_area = num_valid_cells*(dem_header['cellsize']**2)
        self.__summaryInfor = {
                'Case folder':case_folder,
                'Number of Sections':str(num_of_sections),
                'Grid size':'{:d} rows * {:d} cols, {:.2f}m resolution'.format(
                            dem_header['nrows'],dem_header['ncols'],
                            dem_header['cellsize']),
                'Domain area':'{1:,} m^2 with {0:,} valid cells'.format(
                                    num_valid_cells,self.__domain_area)
                              }
        self.add_items('--------------','Additional parameters------------')
    
    def display(self):
        """
        Display the model summary information
        
        """
        print('*******************Model summary***************')
        for key in self.__summaryInfor.keys():
            if key.endswith('-'):
                print(key+self.__summaryInfor[key])
            else:
                print(key+': '+self.__summaryInfor[key])
        print('***********************************************')
    
    def write_readme(self,filename=None):
        """
        Write readme file for the summary information
        """
        if filename is None:
            filename = self.__case_folder+'/readme.txt'
        with open(filename,'w') as f:
            for key,value in self.__summaryInfor.items():
                f.write(key+': '+value+'\n')
    
    def add_items(self,itemName,itemValue):
        if not isinstance(itemValue,str):
            itemValue = str(itemValue)
        self.__summaryInfor[itemName]=itemValue
    
    def add_param_infor(self,paramName,paramValue):
        paramValue = np.array(paramValue)
        itemName = paramName
        if paramValue.size==1:
            itemValue = ' {:} for all cells'.format(paramValue)
        else:
            if paramName in ['h0','hU0']:
                numWetCells = np.sum(paramValue>0)
                numWetCellsRate = numWetCells/paramValue.size
                itemValue = ' Wet cells ratio: {:.2f}%'.format(
                        numWetCellsRate*100)
            elif paramName == 'precipitation_mask':
                itemValue = '{:d} rainfall sources'.format(paramValue.max()+1)
            elif paramName == 'precipitation_source':
                rain_max,rain_mean = _check_rainfall_rate_values(paramValue)
                display_str = 'max rain: {:.2f} mm/h, '+\
                                'average rain: {:.2f} mm/h'
                itemValue = display_str.format(rain_max,rain_mean)
            elif paramName == 'gauges_pos':
                itemValue = '{:d} gauges'.format(paramValue.shape[0])
            else:
                itemNumbers,itemNumberCounts = np.unique(paramValue,
                                                         return_counts=True)
                itemValue = ' Values{:}, ratio{:}'.format(itemNumbers,
                                    itemNumberCounts/paramValue.size)
        self.__summaryInfor[itemName]=itemValue
    def save_object(self,file_name):
        # Overwrites any existing file.
        with open(file_name, 'wb') as output_file:  
            pickle.dump(self, output_file, pickle.HIGHEST_PROTOCOL)        

def load_object(file_name):
    """ Read a pickle file as an InputHipims object 
    """
    #read an InputHipims object file
    with open(file_name, 'rb') as input_file:
        obj = pickle.load(input_file)
    return obj

def save_object(obj,file_name):
    # Overwrites any existing file.
    if not file_name.endswith('.pickle'):
        file_name = file_name+'.pickle'
    with open(file_name, 'wb') as output_file:  
        pickle.dump(obj, output_file, pickle.HIGHEST_PROTOCOL)     
        
#%% Displays or updates a console progress bar
def progress_display(total, progress, fileTag, timeLeft):
    """
    Displays or updates a console progress bar.
    """
    if total==progress:
        fileTag = 'finished'
    else:
        fileTag = fileTag+'...'    
    barLength, status = 50, ""
    progress = float(progress) / float(total)
    if progress >= 1.:
        progress, status = 1, "\r\n"
    block = int(round(barLength * progress))
    text = "\r|{}| {:.0f}% {:<16} time left: {:.0f}s {}".format(
        chr(9608) * block + "-" * (barLength - block), round(progress * 100, 0),
        fileTag,timeLeft,status)
    sys.stdout.write(text)
    sys.stdout.flush()        