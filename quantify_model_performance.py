#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------
quantify_model_performance
Quantify the model performance for timeseries data and spatial data
Quantification methods refer to 
https://www.cawcr.gov.au/projects/verification/
-------------------------------------------------------------------------------
@author: Xiaodong Ming
Created on Thu Dec 12 11:07:20 2019
-------------------------------------------------------------------------------
Assumptions:
- Observations are complete and correct
- Date time series is in pandas datetime format
- Time steps of quantification is based on the time step of observations, 
        simulated values will be linearly interpolated to the time step of 
        observations
- Time ranage of quantification is based on the joint time range of 
        observations and simulations

To do:
- calculate the model performance for timeseries data, including:
    mean error
    bias
    mean absolute error
    root mean_square error
    mean square error
    Nash-Sutcliffe efficiency coefficient
- calculate the model performance of spatial data
"""
__author__ = "Xiaodong Ming"
#import datetime
import pandas as pd
import numpy as np

def get_time_series_scores(df_obs, df_sim, method='NSE'):
    """ Caculate model performance with the observed and simulated timeseries
    df_obs: dataframe with variable 'Time'[1st col] and 'Value'[2nd col]
    df_sim: dataframe with variable 'Time'[1st col] and 'Value'[2nd col]
    method: ['mean_error', 'bias', 'MAE', 'RMSE', 'MSE']
    """
    matched_data = match_records(df_obs, df_sim)
    scores = {}
    # define the type of scores to calculate
    if type(method) is not list:
        if method == 'all' or method == 'All':
            methods = ['NSE', 'mean_error', 'bias', 'MAE', 'RMSE', 'MSE']
        else:
            methods = [method]
    else:
        methods = method
    
    # define function
    def cal_score(matched_data, method):
        """ To calll the function to calculate scores
        """
        if method == 'NSE':
            score = nash_sutcliffe_efficiency_coefficient(matched_data)
        elif method == 'mean_error':
            score = mean_error(matched_data)
        elif method == 'bias':
            score = bias(matched_data)
        elif method == 'MAE':
            score = mean_absolute_error(matched_data)
        elif method == 'RMSE':
            score = root_mean_square_error(matched_data)
        elif method == 'MSE':
            score = mean_square_error(matched_data)
        else:
            raise ValueError(method+ ' is not recognized')
        return score
    
    for method_name in methods:
        score = cal_score(matched_data, method_name)
        scores[method_name] = score
    return scores, matched_data
    
def mean_error(matched_data):
    """ Calculate mean error score for the matched data
    matched_data: a dataframe with two variables 'Sim' and 'Obs'
    Answers the question: What is the average forecast error?
    Range: -∞ to ∞. Perfect score: 0.
    """
#    sample_size = matched_data.shape[0]
    score = np.mean(matched_data['Sim'] - matched_data['Obs'])
    return score

def bias(matched_data):
    """ Calculate bias score for the matched data
    Answers the question: How does the average forecast magnitude compare to 
        the average observed magnitude?
    Range: -∞ to ∞. Perfect score: 1.
    """
    score = np.mean(matched_data['Sim'])/np.mean(matched_data['Obs'])
    return score

def mean_absolute_error(matched_data):
    """ Calculate mean absolute error score for the matched data
    Answers the question: What is the average magnitude of the forecast errors?
    Range: 0 to ∞.  Perfect score: 0.
    """
    score = np.abs(matched_data['Sim'] - matched_data['Obs']).mean()
    return score

def root_mean_square_error(matched_data):
    """ Calculate the root mean square error score for the matched data
    Answers the question: What is the average magnitude of the forecast errors?
    Range: 0 to ∞.  Perfect score: 0.
    """
    score = np.mean((matched_data['Sim'] - matched_data['Obs'])**2)
    score = np.sqrt(score)
    return score

def mean_square_error(matched_data):
    """ Calculate the mean square error score for the matched data
    Measures the mean squared difference between the forecasts and observations
    Range: 0 to ∞.  Perfect score: 0.
    """
    score = np.mean((matched_data['Sim'] - matched_data['Obs'])**2)
    return score

def nash_sutcliffe_efficiency_coefficient(matched_data):
    """Calculate Nash-Sutcliffe efficiency coefficient for the matched data
    Answers the question: How well does the forecast predict the observed time series?
    Range: -∞ to 1.  Perfect score: 1.
    Frequently used to quantify the accuracy of hydrological predictions. 
        If E=0 then the model forecast is no more accurate than the mean of the
        observations; if E<0 then the mean observed value is a more accurate
        predictor than the model.
    """
    score_up = np.sum((matched_data['Sim'] - matched_data['Obs'])**2)
    score_down =  np.sum((matched_data['Obs'] - matched_data['Obs'].mean())**2)
    score = 1-score_up/score_down
    return score

def match_records(df_obs, df_sim, interp_to_obs=True):
    """ Match the observed and simulated observations
    df_obs: dataframe with variable 'Time'[1st col] and 'Value'[2nd col]
    df_sim: dataframe with variable 'Time'[1st col] and 'Value'[2nd col]
    """
    time_min, time_max =  get_time_range(df_sim.iloc[:,0], df_obs.iloc[:,0])
    df_obs = df_obs.loc[(df_obs.iloc[:,0] >= time_min) & 
                        (df_obs.iloc[:,0] <= time_max)]
    df_obs = df_obs.reset_index(drop=True)
    df_sim = df_sim.loc[(df_sim.iloc[:,0] >= time_min) & 
                        (df_sim.iloc[:,0] <= time_max)]
    df_sim = df_sim.reset_index(drop=True)
    time_sim = df_sim.iloc[:,0]
    value_sim = df_sim.iloc[:,1]
    time_obs = df_obs.iloc[:,0]
    value_obs = df_obs.iloc[:,1]
    if type(time_sim[0]) is pd.Timestamp:
        time_sim = datetime_to_seconds(time_sim, time_min)
        time_obs = datetime_to_seconds(time_obs, time_min)
    value_sim_interp = interpolate_timeseries(time_obs, time_sim, value_sim)
    value_obs_interp = interpolate_timeseries(time_sim, time_obs, value_obs)
    if interp_to_obs:
        matched_data = pd.DataFrame({'Sim':value_sim_interp, 'Obs':value_obs})
        time_output = time_obs
    else:
        matched_data = pd.DataFrame({'Sim':value_sim, 'Obs':value_obs_interp})
        time_output = time_sim
    matched_data = matched_data.set_index(time_output)
    return matched_data

def interpolate_timeseries(x, xp, fp):
    """Interpolate simulated values of time series to the observed time steps
    time_sim: pure values in seconds
    time_obs: pure values in seconds
    """
    value_interp = np.interp(x, xp, fp)
    return value_interp

def get_time_range(datetime_series_1, datetime_series_2):
    """ Find the overlayed time window between two time series
    """
    time_min_1 = datetime_series_1.min()
    time_min_2 = datetime_series_2.min()
    if time_min_1 > time_min_2:
        time_min = time_min_1
    else:
        time_min = time_min_2
    time_max_1 = datetime_series_1.max()
    time_max_2 = datetime_series_2.max()
    if time_max_1 < time_max_2:
        time_max = time_max_1
    else:
        time_max = time_max_2
    return time_min, time_max

def datetime_to_seconds(datetime_series, origin):
    """ Convert date time series to pure values refer to an origin time
    datetime_series: a pandas dataframe series of datetime
    origin: datetime like string. Define the reference date
            example: '2019-1-1 3:00', '2019-1-1T03'
            or a pandas timestamp object
    Return:
        time_values: time in seconds from the reference time (origin)
    """
    origin = pd.Timestamp(origin)
    time_values = datetime_series-origin
    time_values = pd.to_timedelta(time_values).dt.total_seconds()
    return time_values
