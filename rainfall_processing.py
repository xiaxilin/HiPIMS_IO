#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rainfall_processing
Created on Wed Dec  4 14:26:04 2019

@author: Xiaodong Ming
"""
import os
import warnings
import imageio
import shapefile # Requires the pyshp package
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from datetime import timedelta
from myclass import Raster
"""
Explaination of general parameters 
rain_source: (numpy array) rainfall source array
          The 1st column is usually time series in seconds, from the 2nd column
          towards end columns are rainfall rate in m/s
rain_mask: (numpy array) provide sequnce number of each gridded rainfall source
start_date: a datetime object to give the initial date and time of rain
    
"""
#%%
def get_time_series(rain_source, rain_mask=None, 
                     start_date=None, method='mean'):
    """ Plot time series of average rainfall rate inside the model domain   
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
    return plot_data

def plot_time_series(rain_source, method='mean', **kwargs):
    """ Plot time series of average rainfall rate inside the model domain   
    method: 'mean'|'max','min','mean'method to calculate gridded rainfall 
    over the model domain
    """
    plot_data = get_time_series(rain_source, method=method, **kwargs)
    time_x = plot_data[:,0]
    value_y = plot_data[:,1]
    fig, ax = plt.subplots()
    ax.plot(time_x,value_y)
    if 'start_date' in kwargs.keys():
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.set_ylabel('Rainfall rate (mm/h)')
    ax.grid(True)
    title_str = method+' precipitation in the model domain'
    ax.set_title(title_str)
    plt.show()
    return fig, ax

def create_animation(output_file, rain_source, mask_file,
                     duration=0.5, **kwargs):
    """ Create animation of gridded rainfall rate    
    mask_header: (dict) header file provide georeference of rainfall mask
    start_date: a datetime object to give the initial date and time of rain
    duration: duration for each frame (seconds)
    cellsize: sclar (meter) the size of rainfall grid cells
    """
    fig_names = create_pictures(rain_source, mask_file, **kwargs)
    # create animation with the images
    images = []
    for fig_name in fig_names:
        images.append(imageio.imread(fig_name))
        os.remove(fig_name)
    # save animation and delete images
    if not output_file.endswith('.gif'):
        output_file = output_file+'.gif'
    imageio.mimsave(output_file, images, duration=duration)

def create_mp4(output_file, rain_source, mask_file, fps=10, **kwargs):
    fig_names = create_pictures(rain_source, mask_file, **kwargs)    
    if not output_file.endswith('.mp4'):
        output_file = output_file+'.mp4'
    print(output_file)
    writer = imageio.get_writer(output_file, 'MP4', fps=fps)
    for fig_name in fig_names:
        writer.append_data(imageio.imread(fig_name))
        os.remove(fig_name)
    writer.close()
    
def create_pictures(rain_source, mask_file, cellsize=1000, 
                    start_date=None, shp_file=None, **kwargs):
    """ create rainfall rate images
    rain_source
    mask_file: a arc grid or a Raster object
    """
    if type(mask_file) is str:
        mask_file = Raster(mask_file)
    mask_obj = mask_file.resample(cellsize, 'near')
    mask_header = mask_obj.header
    time_series = rain_source[:,0]
    rain_values = rain_source[:,1:]*1000*3600 # m/s to mm/h
    vmin = 0#rain_values.min()
#    vmax = 15#rain_values.max()
    # read shapefile if provided
    if shp_file is not None:
        sf = shapefile.Reader(shp_file)
    # create images
    fig_names = []
    for i in np.arange(0, time_series.size):
        print(str(i)+'/'+str(time_series.size))
        rain_array = mask_obj.array*0.0
        mask_values = np.unique(mask_obj.array).astype('int')
        for value in mask_values:
            rain_array[mask_obj.array == value] = rain_values[i, value]
        rain_array[rain_array == 0] = np.nan
        rain_obj = Raster(array=rain_array, header=mask_header)
        fig_name = 'temp'+str(i)+'.png'
        fig_names.append(fig_name)
        fig, ax = rain_obj.mapshow(vmin=vmin, **kwargs)#
        if start_date is None:
            title_str = '{:.0f}'.format(time_series[i])+'s'
        else:
            title_str = start_date+timedelta(seconds=time_series[i])
            title_str = title_str.strftime("%m/%d/%Y %H:%M:%S")
        ax.set_title(title_str)
        xbound = ax.get_xbound()
        ybound = ax.get_ybound()
        # draw shape file on the rainfall map
        if shp_file is not None:
            for shape in sf.shapeRecords():
                x = [i[0] for i in shape.shape.points[:]]
                y = [i[1] for i in shape.shape.points[:]]
                ax.plot(x, y, color='r',linewidth=1)
        ax.set_xbound(xbound)
        ax.set_ybound(ybound)
        fig.savefig(fig_name, dpi=100)
        plt.close(fig)
    return fig_names

#%% ============================Visualization==================================
#%% 
def initialMap(zMat,zExtent,poly_df=[],mapExtent=[],figsize=(6,8),vmin=0,vmax=10):
    """ plot the initial map, then renew raster files only 
    functions to be editted
    """      
    # create figure
    fig,ax = plt.subplots(1, figsize=figsize)
    # plot shapefile outline
    if len(poly_df)!=0:
        poly_df.plot(ax=ax,facecolor='none',edgecolor='r',linewidth=0.5, animated=True)
    else:
        mapExtent=zExtent 
    # create raster map    
    img = ax.imshow(zMat,extent=zExtent,vmin=vmin,vmax=vmax)    
    plt.axis('equal')
    if len(mapExtent)==0:
        mapExtent = (min(poly_df.bounds.minx),max(poly_df.bounds.maxx),
                     min(poly_df.bounds.miny),max(poly_df.bounds.maxy))
    ax.set_xlim(mapExtent[0], mapExtent[1])
    ax.set_ylim(mapExtent[2], mapExtent[3])
    
    # deal with x and y tick labels
    if mapExtent[1]-mapExtent[0]>10000:
        labels = [str(int(value/1000)) for value in ax.get_xticks()]
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels(labels)
        labels = [str(int(value/1000)) for value in ax.get_yticks()]
        ax.set_yticks(ax.get_yticks())
        ax.set_yticklabels(labels,rotation=90)
        ax.set_xlabel('km towards east')
        ax.set_ylabel('km towards north')
    else:
        ax.set_xlabel('metre towards east')
        ax.set_ylabel('metre towards north')
        plt.yticks(rotation=90)
    
    ax.axes.grid(linestyle='--',linewidth=0.5)
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    axins = inset_axes(ax,
               width="5%",  # width = 5% of parent_bbox width
               height="100%",  # height : 50%
               loc='lower right',
               bbox_to_anchor=(0.06, 0., 1, 1),
               bbox_transform=ax.transAxes,
               borderpad=0,
               )
    cbar=plt.colorbar(img,pad=0.05,cax=axins)
#        labels = cbar.ax.get_yticklabels()
#        cbar.ax.set_yticklabels(labels, rotation='vertical')
    cbar.ax.set_title('mm/h',loc='left')        
    return fig,ax,img
    
def _check_rainfall_rate_values(rain_source, times_in_1st_col=True):
    """ Check the rainfall rate values in rain source array
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