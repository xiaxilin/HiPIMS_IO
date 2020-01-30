#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rainfall_processing
Created on Wed Dec  4 14:26:04 2019

@author: Xiaodong Ming
"""
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from datetime import timedelta
#%%
def plot_time_series(rain_source, rain_mask=None, start_date=None, method='mean'):
    """ Plot time series of average rainfall rate inside the model domain
    rain_mask: (numpy array)
    start_date: a datetime object to give the initial date and time of rain
    method: 'mean'|'max','min','mean'method to calculate gridded rainfall 
    over the model domain
    """
    if rain_mask is not None:
        rain_mask = rain_mask[~np.isnan(rain_mask)]
        rain_mask = rain_mask.astype('int32')
        rain_mask_unique = np.unique(rain_mask).flatten()
        rain_mask_unique = rain_mask_unique.astype('int32') 
        rain_source_valid = rain_source[:, rain_mask_unique+1]
    else:
        rain_source_valid = rain_source[:, 1:]
    time_series = rain_source[:,0]
    if type(start_date) is datetime:
        time_delta = np.array([timedelta(seconds=i) for i in time_series])
        time_x = start_date+time_delta
    else:
        time_x = time_series
    if method == 'mean':
        value_y = np.mean(rain_source_valid, axis=1)
    elif method== 'max':
        value_y = np.max(rain_source_valid, axis=1)
    elif method== 'min':
        value_y = np.min(rain_source_valid, axis=1)
    elif method== 'median':
        value_y = np.median(rain_source_valid, axis=1)
    else:
        raise ValueError('Cannot recognise the calculation method')
    value_y =  value_y*3600*1000
    plot_data = np.c_[time_x,value_y]
    fig, ax = plt.subplots()
    ax.plot(time_x,value_y)
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.set_ylabel('Rainfall rate (mm/h)')
    ax.grid(True)
    title_str = method+' precipitation in the model domain'
    ax.set_title(title_str)
    plt.show()
    return plot_data
    
    
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