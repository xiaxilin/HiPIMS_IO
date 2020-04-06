#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 18:14:37 2019

@author: ming
"""

#%% Example for output class
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import hipims_post_process as hpp
#%% read observations
import os
import pandas as pd
import datetime
data_folder = '/Users/ming/OneDrive - Newcastle University/Infiltration/Alyn/'
os.chdir(data_folder)
output_obj = hpp.load_object(data_folder+'Outputs/alyn_output_0.pickle')
file_name = (data_folder+'AlynGaugeObservations.xlsx')
gauge_obs_pont = pd.read_excel(file_name, sheet_name='Pont')
gauge_obs_rhyd = pd.read_excel(file_name, sheet_name='Rhydymwyn')
date_times = pd.to_datetime(gauge_obs_pont['Time']-1, unit='D',
                           origin=pd.Timestamp('2019-06-01'))
gauge_obs_pont['date_times'] = date_times
date_times = pd.to_datetime(gauge_obs_rhyd['Time']-1, unit='D',
                           origin=pd.Timestamp('2019-06-01'))
gauge_obs_rhyd['date_times'] = date_times
gauge_sim_pont = output_obj.gauge_values['Pont_y_Capel']
gauge_sim_pont['Waterdepth'] = gauge_sim_pont['h']['values']
gauge_sim_pont['Discharge'] = gauge_sim_pont['hU']['values_x']
gauge_sim_rhyd = output_obj.gauge_values['Rhydymwyn']
gauge_sim_rhyd['Waterdepth'] = gauge_sim_rhyd['h']['values']
gauge_sim_rhyd['Discharge'] = -gauge_sim_rhyd['hU']['values_y']
def plot_time_series(axs,data_sim, data_obs, gauge_name, var_name='Waterdepth'):
    x_sim = data_sim['h']['date_times']
    y_sim = data_sim[var_name]
    x_obs = data_obs['date_times']
    y_obs = data_obs[var_name]
    date_left = datetime.datetime(2019,6,7,0)
    date_right = datetime.datetime(2019,6,11,0)
    axs.plot(x_sim, y_sim)
    axs.plot(x_obs, y_obs)
    axs.set_xlim(date_left, date_right)
    axs.tick_params(axis='y', labelcolor='tab:red')
    if var_name == 'Waterdepth':
        y_label = var_name+' (m)'
    else:
        y_label = var_name+' (m^3/s)'
    axs.set_ylabel(y_label)
    axs.set_title(gauge_name)
    axs.xaxis.set_major_locator(mdates.HourLocator(interval=24))
    axs.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    axs.grid(True)
    
fig, axs = plt.subplots(2, 1)
# Pont_y_Capel
var_name = 'Discharge' #'Waterdepth' #
gauge_name = 'Pont_y_Capel'
plot_time_series(axs[0], gauge_sim_pont, gauge_obs_pont, gauge_name, var_name)
gauge_name = 'Rhydymwyn'
plot_time_series(axs[1], gauge_sim_rhyd, gauge_obs_rhyd, gauge_name, var_name)
plt.tight_layout()