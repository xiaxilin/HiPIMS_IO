#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
example_process_MO_rainfall_dataset
show examples to process three rainfall datasets from UK Met Office including:
    NIMROD radar rainfall, UKV rainfall prediction, MOGREPS rainfall prediction
To do:
    dowload UK Met Office NIMROD radar rainfall data
    read UM rainfall prediction files
    produce rainfall input files (rainfall source and mask) for HiPIMS    

Created on Tue Feb 10 11:56:46 2020

@author: XIaodong Ming
"""
import sys
# position storing HiPIMS_IO.py and ArcGridDataProcessing.py
scriptsPath = '/Users/ming/Dropbox/Python/HiPIMS' 
sys.path.insert(0,scriptsPath)
import glob
import os
import time
import mogreps # for both UKV and MOGREPS
import nimrodFuncs as NF 
from myclass import Raster
#%% ===================Process NIMROD radar file==================================
"""
download compressed tar file (one file per day)
read tar file and convert it to grid files (one file for each timestamp)
cut gridded rainfall data to an user-defined extent
produce rainfall input files (rainfall source and mask) for HiPIMS
"""
#% # download file from CEDA ftp 
#need your username and password on CEDA website
date_str_list = ['20190929', '20190930', '20191001']
# username and password on CEDA website are required in the popup messages
NF.downloadNimrodTar(dateStr=date_str_list)

#% get gridded rainfall data from a nimrod tar file
tar_file = 'metoffice-c-band-rain-radar_uk_20190929_1km-composite.dat.gz.tar'
datetime_strs = NF.get_datetime_str_from_tar(tar_file)
date_time_str = datetime_strs[10] # yyyy MM dd HH mm
# return a list of Raster objects
raster_obj_list, _ = NF.read_gridded_data_from_tar(tar_file, date_time_str)
raster_obj = raster_obj_list[0]
# show the gridded data
#raster_obj.array[raster_obj.array == 0] = raster_obj.header['NODATA_value']
raster_obj.mapshow()
#%% produce rainfall source and mask files 
# a DEM file mask be provided to generate rainfall mask file for HiPIMS
# rain_source is a numpy array with the 1st column representing time
#   and from 2nd to the end column representing rainfall rates (m/s) in each 
#   rainfall grid cell (column no. - cell value)
# rain_mask is an array with the same size and georeference information of the
#   DEM file array and provides values indicating rainfall sources
# from a list of nimrod tar files
tar_file_list = glob.glob('*.tar')
dem_obj = Raster('ExampleDEM.asc')
datetime_start = '201909291030' # a reference datetime for rainfall source time
rainMask, rainSource = NF.getRainMaskSource(tarList=tar_file_list,
                                            demHead=dem_obj.header,
                                            datetimeStart=datetime_start)
#%% ===================Process UKV data==================================
# read UKV pp file

# read one file and return a data object
pp_file = 'prods_op_ukv_20190610_03_000.pp'
data_obj = mogreps.MOGREPS_data(pp_file, 'stratiform_rainfall_flux') #only read rainfall data
#mask_obj, _ = data_obj.Create_rain_mask() #creat a mask object on the full extent
# create a mask object covering the extent of a given DEM file
mask_obj, indArray = data_obj.Create_rain_mask('ExampleDEM.asc')
# mask_obj can be exported as an asc file
mask_obj.Write_asc('rain_mask.asc')
# data_obj can be saved as a much smaller file compared by the pp file
data_obj.Save_object('prods_op_ukv_20190610_03_000.gz')
# export rainfall source array for one forecast
# need all gz/pp files with date_hour_string '20190610_03'
date_hour_Str = '20190610_03'
gzfileList = glob.glob('*'+date_hour_Str+'*.gz')
mogreps.WriteRainSourceArray(gzfileList=gzfileList, demFile='rain_mask.asc')
#%% ===================Process MOGREPS data==================================
"""
Mogres data has the same formtat with the data from UKV but with much more files.
Because of the large amount of data with a lot of information we do not need,
we will read and convert pp files to mogreps objects and save them as gz files.
"""
datestr = input('Enter a date [yyyymmdd]:')
files = glob.glob(datestr+'/*.pp')
# transfer pp file to a mogreps object and compressed in a gz file
start = time.perf_counter()
for ppFile in files:
#ppFile = 'prods_op_mogreps-uk_20190617_02_34_009.pp'
    obj = mogreps.MOGREPS_data(ppFile)
    obj.Save_object()
    print(ppFile+' to gz file')
    os.remove(ppFile)
end = time.perf_counter()
print('number of files: '+ str(len(files)))
print('Time elapse: '+ str(end-start))
# after all pp files are converted to gz files any file can be read by class MOGREPS