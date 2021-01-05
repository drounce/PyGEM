"""Run bias adjustments a given climate dataset"""

# Built-in libraries
import os
import argparse
import multiprocessing
import inspect
import time
from time import strftime
# External libraries
import pandas as pd
import numpy as np
import xarray as xr
from scipy.optimize import minimize
from scipy.ndimage import uniform_filter
import pickle
import matplotlib.pyplot as plt
# Local libraries
import pygem.pygem_input as pygem_prms
import pygem.pygem_modelsetup as modelsetup
import class_climate

#%% ===== SCRIPT SPECIFIC INPUT DATA =====
option_plot_adj = True
figure_fp = '/Users/davidrounce/Documents/Dave_Rounce/HiMAT/Output/figures/cmip5/'


#%% FUNCTIONS
def getparser():
    """
    Use argparse to add arguments from the command line
    
    Parameters
    ----------
    gcm_file : str
        full filepath to text file that has list of gcm names to be processed
    num_simultaneous_processes (optional) : int
        number of cores to use in parallels
    option_parallels (optional) : int
        switch to use parallels or not
    rgi_glac_number_fn : str
        filename of .pkl file containing a list of glacier numbers that used to run batches on the supercomputer
    progress_bar : int
        Switch for turning the progress bar on or off (default = 0 (off))
    debug : int
        Switch for turning debug printing on or off (default = 0 (off))
        
    Returns
    -------
    Object containing arguments and their respective values.
    """
    parser = argparse.ArgumentParser(description="run gcm bias corrections from gcm list in parallel")
    # add arguments
#    parser.add_argument('-gcm_file', action='store', type=str, default=None,
#                        help='text file full of gcm names')
    parser.add_argument('-gcm_list_fn', action='store', type=str, default=pygem_prms.ref_gcm_name,
                        help='text file full of commands to run')
    parser.add_argument('-gcm_name', action='store', type=str, default=None,
                        help='GCM name used for model run')
    parser.add_argument('-rcp', action='store', type=str, default=None,
                        help='rcp scenario used for model run (ex. rcp26)')
    parser.add_argument('-num_simultaneous_processes', action='store', type=int, default=2,
                        help='number of simultaneous processes (cores) to use')
    parser.add_argument('-option_parallels', action='store', type=int, default=1,
                        help='Switch to use or not use parallels (1 - use parallels, 0 - do not)')
    parser.add_argument('-rgi_glac_number_fn', action='store', type=str, default=None,
                        help='Filename containing list of rgi_glac_number, helpful for running batches on spc')
    parser.add_argument('-debug', action='store', type=int, default=0,
                        help='Boolean for debugging to turn it on or off (default 0 is off')
    return parser

def annual_avg_2darray(x):
    """
    Annual average of dataset, where columns are a monthly timeseries (temperature)
    """
    return x.reshape(-1,12).mean(1).reshape(x.shape[0],int(x.shape[1]/12))


def annual_sum_2darray(x):
    """
    Annual sum of dataset, where columns are a monthly timeseries (precipitation)
    """
    return x.reshape(-1,12).sum(1).reshape(x.shape[0],int(x.shape[1]/12))


def monthly_avg_2darray(x):
    """
    Monthly average for a given 2d dataset where columns are monthly timeseries
    """
    return x.reshape(-1,12).transpose().reshape(-1,int(x.shape[1]/12)).mean(1).reshape(12,-1).transpose()


def monthly_std_2darray(x):
    """
    Monthly standard deviation for a given 2d dataset where columns are monthly timeseries
    """
    return x.reshape(-1,12).transpose().reshape(-1,int(x.shape[1]/12)).std(1).reshape(12,-1).transpose()

    
def temp_biasadj_HH2015(ref_temp, ref_elev, gcm_temp, dates_table_ref, dates_table):
    """
    Huss and Hock (2015) temperature bias correction based on mean and interannual variability
    
    Note: the mean over the reference period will only equal the mean of the gcm for the same time period when the GCM
    time series is run for the same period, i.e., due to the 25-year moving average, the mean gcm temps from 2000-2019
    will differ if using a reference period of 2000-2020 to bias adjust gcm temps from 2000-2100.
    
    Parameters
    ----------
    ref_temp : np.array
        time series of reference temperature
    gcm_temp : np.array
        time series of GCM temperature
    dates_table_ref : pd.DataFrame
        dates table for reference time period
    dates_table : pd.DataFrame
        dates_table for GCM time period
    
    Returns
    -------
    gcm_temp_biasadj : np.array
        GCM temperature bias corrected to the reference climate dataset according to Huss and Hock (2015)
    gcm_elev_biasadj : float
        new gcm elevation is the elevation of the reference climate dataset
    """
    # GCM subset to agree with reference time period to calculate bias corrections
    gcm_subset_idx_start = np.where(dates_table.date.values == dates_table_ref.date.values[0])[0][0]
    gcm_subset_idx_end = np.where(dates_table.date.values == dates_table_ref.date.values[-1])[0][0]
    gcm_temp_subset = gcm_temp[:,gcm_subset_idx_start:gcm_subset_idx_end+1]

    # Remove spinup years, so adjustment performed over calibration period
    ref_temp_nospinup = ref_temp[:,pygem_prms.ref_spinupyears*12:]
    gcm_temp_nospinup = gcm_temp_subset[:,pygem_prms.gcm_spinupyears*12:]
    
    # Roll months so they are aligned with simulation months
    roll_amt = -1*(12 - gcm_subset_idx_start%12)
    if roll_amt == -12:
        roll_amt = 0 
    
    # Mean monthly temperature
    ref_temp_monthly_avg = np.roll(monthly_avg_2darray(ref_temp_nospinup), roll_amt, axis=1)
    gcm_temp_monthly_avg = np.roll(monthly_avg_2darray(gcm_temp_nospinup), roll_amt, axis=1)
    # Standard deviation monthly temperature
    ref_temp_monthly_std = np.roll(monthly_std_2darray(ref_temp_nospinup), roll_amt, axis=1)
    gcm_temp_monthly_std = np.roll(monthly_std_2darray(gcm_temp_nospinup), roll_amt, axis=1)

    # Monthly bias adjustment (additive)
    gcm_temp_monthly_adj = ref_temp_monthly_avg - gcm_temp_monthly_avg

    # Monthly variability
    variability_monthly_std = ref_temp_monthly_std / gcm_temp_monthly_std
    
    # Monthly temperature bias adjusted according to monthly average
    t_mt = gcm_temp + np.tile(gcm_temp_monthly_adj, int(gcm_temp.shape[1]/12))
    
    # Mean monthly temperature bias adjusted according to monthly average
    #  t_m25avg is the avg monthly temp in a 25-year period around the given year
    N = 25
    t_m_Navg = np.zeros(t_mt.shape)
    for month in range(0,12):
        t_m_subset = t_mt[:,month::12]
        # Uniform filter computes running average and uses 'reflects' values at borders
        t_m_Navg_subset = uniform_filter(t_m_subset,size=(1,N))
        t_m_Navg[:,month::12] = t_m_Navg_subset

    gcm_temp_biasadj = t_m_Navg + (t_mt - t_m_Navg) * np.tile(variability_monthly_std, int(gcm_temp.shape[1]/12))
    
    # Update elevation
    gcm_elev_biasadj = ref_elev
    
    # Assert that mean temperatures for all the glaciers must be more-or-less equal
    gcm_temp_biasadj_subset = (
            gcm_temp_biasadj[:,gcm_subset_idx_start:gcm_subset_idx_end+1][:,pygem_prms.ref_spinupyears*12:])
    assert np.max(np.abs(np.mean(gcm_temp_biasadj_subset, axis=1) - 
                         np.mean(ref_temp[:,pygem_prms.ref_spinupyears*12:], axis=1))) < 1, (
            'Error with gcm temperature bias adjustment: mean ref and gcm temps differ by more than 1 degree')
    
    return gcm_temp_biasadj, gcm_elev_biasadj


def prec_biasadj_HH2015(ref_prec, ref_elev, gcm_prec, dates_table_ref, dates_table):
    """
    Huss and Hock (2015) precipitation bias correction based on mean (multiplicative)
    
    Parameters
    ----------
    ref_prec : np.array
        time series of reference precipitation
    gcm_prec : np.array
        time series of GCM precipitation
    dates_table_ref : pd.DataFrame
        dates table for reference time period
    dates_table : pd.DataFrame
        dates_table for GCM time period
    gcm_elev_biasadj : float
        new gcm elevation is the elevation of the reference climate dataset
    
    Returns
    -------
    gcm_prec_biasadj : np.array
        GCM precipitation bias corrected to the reference climate dataset according to Huss and Hock (2015)
    """
    # GCM subset to agree with reference time period to calculate bias corrections
    gcm_subset_idx_start = np.where(dates_table.date.values == dates_table_ref.date.values[0])[0][0]
    gcm_subset_idx_end = np.where(dates_table.date.values == dates_table_ref.date.values[-1])[0][0]
    gcm_prec_subset = gcm_prec[:,gcm_subset_idx_start:gcm_subset_idx_end+1]
    
    # Remove spinup years, so adjustment performed over calibration period
    ref_prec_nospinup = ref_prec[:,pygem_prms.ref_spinupyears*12:]
    gcm_prec_nospinup = gcm_prec_subset[:,pygem_prms.gcm_spinupyears*12:]
    
    # Roll months so they are aligned with simulation months
    roll_amt = -1*(12 - gcm_subset_idx_start%12)
    
    # PRECIPITATION BIAS CORRECTIONS
    # Monthly mean precipitation
    ref_prec_monthly_avg = np.roll(monthly_avg_2darray(ref_prec_nospinup), roll_amt, axis=1)
    gcm_prec_monthly_avg = np.roll(monthly_avg_2darray(gcm_prec_nospinup), roll_amt, axis=1)
    bias_adj_prec_monthly = ref_prec_monthly_avg / gcm_prec_monthly_avg
    # Bias adjusted precipitation accounting for differences in monthly mean
    gcm_prec_biasadj = gcm_prec * np.tile(bias_adj_prec_monthly, int(gcm_prec.shape[1]/12))
    
    # Update elevation
    gcm_elev_biasadj = ref_elev
    
    # Assertion that bias adjustment does not drastically modify the precipitation and are reasonable
    gcm_prec_biasadj_subset = (
            gcm_prec_biasadj[:,gcm_subset_idx_start:gcm_subset_idx_end+1][:,pygem_prms.gcm_spinupyears*12:])
    gcm_prec_biasadj_frac = gcm_prec_biasadj_subset.sum(axis=1) / ref_prec_nospinup.sum(axis=1)
    assert np.min(gcm_prec_biasadj_frac) > 0.5 and np.max(gcm_prec_biasadj_frac) < 2, (
            'Error with gcm precipitation bias adjustment: total ref and gcm prec differ by more than factor of 2')
    assert gcm_prec_biasadj.max() <= 10, 'gcm_prec_adj (precipitation bias adjustment) too high, needs to be modified'
    assert gcm_prec_biasadj.min() >= 0, 'gcm_prec_adj is producing a negative precipitation value'   
    
    return gcm_prec_biasadj, gcm_elev_biasadj


def prec_biasadj_opt1(ref_prec, ref_elev, gcm_prec, dates_table_ref, dates_table):
    """
    Precipitation bias correction based on mean with limited maximum
    
    Parameters
    ----------
    ref_prec : np.array
        time series of reference precipitation
    gcm_prec : np.array
        time series of GCM precipitation
    dates_table_ref : pd.DataFrame
        dates table for reference time period
    dates_table : pd.DataFrame
        dates_table for GCM time period
    
    Returns
    -------
    gcm_prec_biasadj : np.array
        GCM precipitation bias corrected to the reference climate dataset according to Huss and Hock (2015)
    gcm_elev_biasadj : float
        new gcm elevation is the elevation of the reference climate dataset
    """
    # GCM subset to agree with reference time period to calculate bias corrections
    gcm_subset_idx_start = np.where(dates_table.date.values == dates_table_ref.date.values[0])[0][0]
    gcm_subset_idx_end = np.where(dates_table.date.values == dates_table_ref.date.values[-1])[0][0]
    gcm_prec_subset = gcm_prec[:,gcm_subset_idx_start:gcm_subset_idx_end+1]
    
    # Remove spinup years, so adjustment performed over calibration period
    ref_prec_nospinup = ref_prec[:,pygem_prms.ref_spinupyears*12:]
    gcm_prec_nospinup = gcm_prec_subset[:,pygem_prms.gcm_spinupyears*12:]
    
    # Roll months so they are aligned with simulation months
    roll_amt = -1*(12 - gcm_subset_idx_start%12)
    
    # PRECIPITATION BIAS CORRECTIONS
    # Monthly mean precipitation
    ref_prec_monthly_avg = np.roll(monthly_avg_2darray(ref_prec_nospinup), roll_amt, axis=1)
    gcm_prec_monthly_avg = np.roll(monthly_avg_2darray(gcm_prec_nospinup), roll_amt, axis=1)
    bias_adj_prec_monthly = ref_prec_monthly_avg / gcm_prec_monthly_avg
    # Bias adjusted precipitation accounting for differences in monthly mean
    gcm_prec_biasadj_raw = gcm_prec * np.tile(bias_adj_prec_monthly, int(gcm_prec.shape[1]/12))
    
    # Adjust variance based on zscore and reference standard deviation
    ref_prec_monthly_std = np.roll(monthly_std_2darray(ref_prec_nospinup), roll_amt, axis=1)
    gcm_prec_biasadj_raw_monthly_avg = monthly_avg_2darray(gcm_prec_biasadj_raw[:,0:ref_prec.shape[1]])
    gcm_prec_biasadj_raw_monthly_std = monthly_std_2darray(gcm_prec_biasadj_raw[:,0:ref_prec.shape[1]])
    # Calculate value compared to mean and standard deviation
    gcm_prec_biasadj_zscore = (
            (gcm_prec_biasadj_raw - np.tile(gcm_prec_biasadj_raw_monthly_avg, int(gcm_prec.shape[1]/12))) / 
             np.tile(gcm_prec_biasadj_raw_monthly_std, int(gcm_prec.shape[1]/12)))
    gcm_prec_biasadj = (
            np.tile(gcm_prec_biasadj_raw_monthly_avg, int(gcm_prec.shape[1]/12)) +
            gcm_prec_biasadj_zscore * np.tile(ref_prec_monthly_std, int(gcm_prec.shape[1]/12)))
    gcm_prec_biasadj[gcm_prec_biasadj < 0] = 0
    
    # Identify outliers using reference's monthly maximum adjusted for future increases
    ref_prec_monthly_max = np.roll((ref_prec_nospinup.reshape(-1,12).transpose()
                                    .reshape(-1,int(ref_prec_nospinup.shape[1]/12)).max(1).reshape(12,-1).transpose()), 
                                   roll_amt, axis=1)
    gcm_prec_max_check = np.tile(ref_prec_monthly_max, int(gcm_prec_biasadj.shape[1]/12))        
    # For wetter years in future, adjust monthly max by the annual increase in precipitation
    gcm_prec_annual = annual_sum_2darray(gcm_prec)
    gcm_prec_annual_norm = gcm_prec_annual / gcm_prec_annual.mean(1)[:,np.newaxis]
    gcm_prec_annual_norm_repeated = np.repeat(gcm_prec_annual_norm, 12).reshape(gcm_prec_biasadj.shape)
    gcm_prec_max_check_adj = gcm_prec_max_check * gcm_prec_annual_norm_repeated
    gcm_prec_max_check_adj[gcm_prec_max_check_adj < gcm_prec_max_check] = (
            gcm_prec_max_check[gcm_prec_max_check_adj < gcm_prec_max_check])
    
    # Replace outliers with monthly mean adjusted for the normalized annual variation
    outlier_replacement = (gcm_prec_annual_norm_repeated * 
                           np.tile(ref_prec_monthly_avg, int(gcm_prec_biasadj.shape[1]/12)))
    gcm_prec_biasadj[gcm_prec_biasadj > gcm_prec_max_check_adj] = (
            outlier_replacement[gcm_prec_biasadj > gcm_prec_max_check_adj])
    
    # Update elevation
    gcm_elev_biasadj = ref_elev
    
    # Assertion that bias adjustment does not drastically modify the precipitation and are reasonable
    gcm_prec_biasadj_subset = (
            gcm_prec_biasadj[:,gcm_subset_idx_start:gcm_subset_idx_end+1][:,pygem_prms.gcm_spinupyears*12:])
    gcm_prec_biasadj_frac = gcm_prec_biasadj_subset.sum(axis=1) / ref_prec_nospinup.sum(axis=1)
    assert np.min(gcm_prec_biasadj_frac) > 0.5 and np.max(gcm_prec_biasadj_frac) < 2, (
            'Error with gcm precipitation bias adjustment: total ref and gcm prec differ by more than factor of 2')
    assert gcm_prec_biasadj.max() <= 10, 'gcm_prec_adj (precipitation bias adjustment) too high, needs to be modified'
    assert gcm_prec_biasadj.min() >= 0, 'gcm_prec_adj is producing a negative precipitation value'   
    
    return gcm_prec_biasadj, gcm_elev_biasadj

    
def monthly_avg_array_rolled(ref_array, dates_table_ref, dates_table):
    """ Monthly average array from reference data rolled to ensure proper months 
    
    Parameters
    ----------
    ref_array : np.array
        time series of reference lapse rates
    dates_table_ref : pd.DataFrame
        dates table for reference time period
    dates_table : pd.DataFrame
        dates_table for GCM time period
    
    Returns
    -------
    gcm_array : np.array
        gcm climate data based on monthly average of reference data
    """
    # GCM subset to agree with reference time period to calculate bias corrections
    gcm_subset_idx_start = np.where(dates_table.date.values == dates_table_ref.date.values[0])[0][0]
    
    # Roll months so they are aligned with simulation months
    roll_amt = -1*(12 - gcm_subset_idx_start%12)
    ref_array_monthly_avg = np.roll(monthly_avg_2darray(ref_array), roll_amt, axis=1)
    gcm_array = np.tile(ref_array_monthly_avg, int(dates_table.shape[0]/12))
    return gcm_array
        
    
def plot_biasadj(ref_temp, gcm_temp_biasadj, ref_prec, gcm_prec, gcm_prec_biasadj, dates_table_ref, dates_table):
    """
    Plot bias adjusted climate data to check values are realistic and adjustment performing as expected
    """
    #%%
    fig_h = 16
    fig_w = 12
    fig, ax = plt.subplots(5, 1, squeeze=False, figsize=(fig_w,fig_h), gridspec_kw = {'wspace':0, 'hspace':0.2})
    #  Subplot #1: Monthly temperatures with annual on top
    gcm_temp_biasadj_annual_avg = annual_avg_2darray(gcm_temp_biasadj)
    ax[0,0].plot(dates_table_ref.date.values, ref_temp[0,:], color='b', linewidth=1, label='era')
    ax[0,0].plot(dates_table.date.values, gcm_temp_biasadj[0,:], color='r', linewidth=1, label='gcm_adj')
    ax[0,0].plot(dates_table.date.values[::12], gcm_temp_biasadj_annual_avg[0,:], color='k', linewidth=2, 
                 label='gcm_adj_yr')
    ax[0,0].set_ylabel('Temp [degC]')
    ax[0,0].legend()    
    
    # GCM subset to agree with reference time period to calculate bias corrections
    gcm_subset_idx_start = np.where(dates_table.date.values == dates_table_ref.date.values[0])[0][0]
    gcm_subset_idx_end = np.where(dates_table.date.values == dates_table_ref.date.values[-1])[0][0]
    gcm_prec_subset = gcm_prec[:,gcm_subset_idx_start:gcm_subset_idx_end+1]
    
    # Remove spinup years, so adjustment performed over calibration period
    ref_prec_nospinup = ref_prec[:,pygem_prms.ref_spinupyears*12:]
    gcm_prec_nospinup = gcm_prec_subset[:,pygem_prms.gcm_spinupyears*12:]
    
    # PRECIPITATION BIAS CORRECTIONS
    # Monthly mean precipitation
    ref_prec_monthly_avg = (ref_prec_nospinup.reshape(-1,12).transpose()
                            .reshape(-1,int(ref_prec_nospinup.shape[1]/12)).mean(1).reshape(12,-1).transpose())
    gcm_prec_monthly_avg = (gcm_prec_nospinup.reshape(-1,12).transpose()
                            .reshape(-1,int(gcm_prec_nospinup.shape[1]/12)).mean(1).reshape(12,-1).transpose())
    
    # Subplot #2: Monthly Precipitation bias adjusted
    ax[1,0].plot(dates_table_ref.date.values, ref_prec[0,:], color='b', linewidth=1, label='era')
    ax[1,0].plot(dates_table.date.values, gcm_prec_biasadj[0,:], color='k', linewidth=1, label='gcm_adj')
    ax[1,0].set_ylabel('Prec [m]')
    ax[1,0].legend()
    
    # Subplot #3: Monthly Precipitation (reference, gcm, and gcm corrected)
    ax[2,0].plot(dates_table_ref.date.values, ref_prec[0,:], color='b', linewidth=1, label='era')
    ax[2,0].plot(dates_table_ref.date.values, gcm_prec_subset[0,:], color='r', linewidth=1, label='gcm')
    ax[2,0].plot(dates_table_ref.date.values, gcm_prec_biasadj[0,0:ref_prec.shape[1]], color='k', linewidth=1, 
                 label='gcm_adj')
    ax[2,0].set_ylabel('Prec [m]')
    ax[2,0].legend()
    
    
    # Subplot #4: Monthly Mean Precipitation (reference, gcm, and gcm corrected)
    month_labels = [0]
    month_labels += dates_table.month.values[0:12].tolist()
    
    ref_prec_monthly_std = monthly_std_2darray(ref_prec_nospinup)
    ref_prec_monthly_low = ref_prec_monthly_avg[0,:] - ref_prec_monthly_std[0,:]
    ref_prec_monthly_high = ref_prec_monthly_avg[0,:] + ref_prec_monthly_std[0,:]
    ax[3,0].plot(np.arange(0,12), ref_prec_monthly_avg[0,:], color='b', linewidth=1, label='era')
    ax[3,0].fill_between(np.arange(0,12), ref_prec_monthly_low, ref_prec_monthly_high, facecolor='b', alpha=0.2)  
    
    gcm_prec_monthly_std = monthly_std_2darray(gcm_prec_nospinup)
    gcm_prec_monthly_low = gcm_prec_monthly_avg[0,:] - gcm_prec_monthly_std[0,:]
    gcm_prec_monthly_high = gcm_prec_monthly_avg[0,:] + gcm_prec_monthly_std[0,:]
    ax[3,0].plot(np.arange(0,12), gcm_prec_monthly_avg[0,:], color='r', linewidth=1, label='gcm')
    ax[3,0].fill_between(np.arange(0,12), gcm_prec_monthly_low, gcm_prec_monthly_high, facecolor='r', alpha=0.2)
    ax[3,0].set_ylabel('Prec [m]')
    ax[3,0].xaxis.set_major_locator(plt.MultipleLocator(1))
    ax[3,0].set_xticklabels(month_labels)
    ax[3,0].legend()
    
    # Subplot #5: Monthly Mean Precipitation (reference and gcm corrected)
    ax[4,0].plot(np.arange(0,12), ref_prec_monthly_avg[0,:], color='b', linewidth=1, label='era')
    ax[4,0].fill_between(np.arange(0,12), ref_prec_monthly_low, ref_prec_monthly_high, facecolor='b', alpha=0.2)  
    
    gcm_prec_biasadj_monthly_avg = monthly_avg_2darray(gcm_prec_biasadj[:,0:ref_prec.shape[1]])
    gcm_prec_biasadj_monthly_std = monthly_std_2darray(gcm_prec_biasadj[:,0:ref_prec.shape[1]])
    gcm_prec_biasadj_monthly_low = gcm_prec_biasadj_monthly_avg[0,:] - gcm_prec_biasadj_monthly_std[0,:]
    gcm_prec_biasadj_monthly_high = gcm_prec_biasadj_monthly_avg[0,:] + gcm_prec_biasadj_monthly_std[0,:]
    ax[4,0].plot(np.arange(0,12), gcm_prec_biasadj_monthly_avg[0,:], color='k', linewidth=1, label='gcm_adj')
    ax[4,0].fill_between(np.arange(0,12), gcm_prec_biasadj_monthly_low, gcm_prec_biasadj_monthly_high, 
                         facecolor='k', alpha=0.2) 
    ax[4,0].set_ylabel('Prec [m]')
    ax[4,0].xaxis.set_major_locator(plt.MultipleLocator(1))
    ax[4,0].set_xticklabels(month_labels)
    ax[4,0].legend()
    #%%
    # Save figure
    fig.set_size_inches(fig_w, fig_h)
    figure_fn = ('biasadjplots_' + gcm_name + '_' + rcp_scenario + '_biasadjopt' + str(pygem_prms.option_bias_adjustment) +
                 '_' + str(pygem_prms.rgi_regionsO1[0]) + '-' + str(rgi_glac_number[0]) + '.png')
    fig.savefig(figure_fp + figure_fn, bbox_inches='tight', dpi=300)


def main(list_packed_vars):
    """
    Climate data bias adjustment
    
    Parameters
    ----------
    list_packed_vars : list
        list of packed variables that enable the use of parallels
        
    Returns
    -------
    csv files of bias adjustment output
        The bias adjustment parameters are output instead of the actual temperature and precipitation to reduce file
        sizes.  Additionally, using the bias adjustment will cause the GCM climate data to use the reference elevation
        since the adjustments were made from the GCM climate data to be consistent with the reference dataset.
    """
    # Unpack variables    
    count = list_packed_vars[0]
    chunk = list_packed_vars[1]
    main_glac_rgi_all = list_packed_vars[2]
    chunk_size = list_packed_vars[3]
    gcm_name = list_packed_vars[4]

    time_start = time.time()
    parser = getparser()
    args = parser.parse_args()
    
    if (gcm_name != pygem_prms.ref_gcm_name) and (args.rcp is None):
        rcp_scenario = os.path.basename(args.gcm_list_fn).split('_')[1]
    elif args.rcp is not None:
        rcp_scenario = args.rcp
#    rcp_scenario = os.path.basename(args.gcm_file).split('_')[1]

    # ===== LOAD OTHER GLACIER DATA =====
    main_glac_rgi = main_glac_rgi_all.iloc[chunk:chunk + chunk_size, :]
    # Glacier hypsometry [km**2], total area
    main_glac_hyps = modelsetup.import_Husstable(main_glac_rgi, pygem_prms.hyps_filepath,
                                                 pygem_prms.hyps_filedict, pygem_prms.hyps_colsdrop)
    # Ice thickness [m], average
    main_glac_icethickness = modelsetup.import_Husstable(main_glac_rgi, pygem_prms.thickness_filepath,
                                                         pygem_prms.thickness_filedict, pygem_prms.thickness_colsdrop)
    main_glac_hyps[main_glac_icethickness == 0] = 0
    # Width [km], average
    main_glac_width = modelsetup.import_Husstable(main_glac_rgi, pygem_prms.width_filepath,
                                                  pygem_prms.width_filedict, pygem_prms.width_colsdrop)
    elev_bins = main_glac_hyps.columns.values.astype(int)

    # Select dates including future projections
    # If reference climate data starts or ends before or after the GCM data, then adjust reference climate data such
    # that the reference and GCM span the same period of time.
    if pygem_prms.startyear >= pygem_prms.gcm_startyear:
        ref_startyear = pygem_prms.startyear
    else:
        ref_startyear = pygem_prms.gcm_startyear
    if pygem_prms.endyear <= pygem_prms.gcm_endyear:
        ref_endyear = pygem_prms.endyear
    else:
        ref_endyear = pygem_prms.gcm_endyear
    dates_table_ref = modelsetup.datesmodelrun(startyear=ref_startyear, endyear=ref_endyear, 
                                               spinupyears=pygem_prms.ref_spinupyears, 
                                               option_wateryear=pygem_prms.ref_wateryear)
    dates_table = modelsetup.datesmodelrun(startyear=pygem_prms.gcm_startyear, endyear=pygem_prms.gcm_endyear, 
                                           spinupyears=pygem_prms.gcm_spinupyears, 
                                           option_wateryear=pygem_prms.gcm_wateryear)

    # ===== LOAD CLIMATE DATA =====
    # Reference climate data
    ref_gcm = class_climate.GCM(name=pygem_prms.ref_gcm_name)
    # Air temperature [degC], Precipitation [m], Elevation [masl], Lapse rate [K m-1]
    ref_temp, ref_dates = ref_gcm.importGCMvarnearestneighbor_xarray(ref_gcm.temp_fn, ref_gcm.temp_vn, main_glac_rgi, 
                                                                     dates_table_ref)
    ref_prec, ref_dates = ref_gcm.importGCMvarnearestneighbor_xarray(ref_gcm.prec_fn, ref_gcm.prec_vn, main_glac_rgi, 
                                                                     dates_table_ref)
    ref_elev = ref_gcm.importGCMfxnearestneighbor_xarray(ref_gcm.elev_fn, ref_gcm.elev_vn, main_glac_rgi)
    ref_lr, ref_dates = ref_gcm.importGCMvarnearestneighbor_xarray(ref_gcm.lr_fn, ref_gcm.lr_vn, main_glac_rgi, 
                                                                   dates_table_ref)
    ref_lr_monthly_avg = (ref_lr.reshape(-1,12).transpose().reshape(-1,int(ref_temp.shape[1]/12)).mean(1)
                          .reshape(12,-1).transpose())
    
    # GCM climate data
    if gcm_name == 'ERA-Interim' or gcm_name == 'COAWST':
        gcm = class_climate.GCM(name=gcm_name)
    else:
        gcm = class_climate.GCM(name=gcm_name, rcp_scenario=rcp_scenario)
    # Air temperature [degC], Precipitation [m], Elevation [masl], Lapse rate [K m-1]
    gcm_temp, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(gcm.temp_fn, gcm.temp_vn, main_glac_rgi, dates_table)
    gcm_prec, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(gcm.prec_fn, gcm.prec_vn, main_glac_rgi, dates_table)
    gcm_elev = gcm.importGCMfxnearestneighbor_xarray(gcm.elev_fn, gcm.elev_vn, main_glac_rgi)
    if gcm_name == 'ERA-Interim':
        gcm_lr, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(gcm.lr_fn, gcm.lr_vn, main_glac_rgi, dates_table)
    else:
        gcm_lr = monthly_avg_array_rolled(ref_lr, dates_table_ref, dates_table)

    # COAWST data has two domains, so need to merge the two domains
    if gcm_name == 'COAWST':
        gcm_temp_d01, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(gcm.temp_fn_d01, gcm.temp_vn, main_glac_rgi, 
                                                                         dates_table)
        gcm_prec_d01, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(gcm.prec_fn_d01, gcm.prec_vn, main_glac_rgi, 
                                                                         dates_table)
        gcm_elev_d01 = gcm.importGCMfxnearestneighbor_xarray(gcm.elev_fn_d01, gcm.elev_vn, main_glac_rgi)
        # Check if glacier outside of high-res (d02) domain
        for glac in range(main_glac_rgi.shape[0]):
            glac_lat = main_glac_rgi.loc[glac,pygem_prms.rgi_lat_colname]
            glac_lon = main_glac_rgi.loc[glac,pygem_prms.rgi_lon_colname]
            if (~(pygem_prms.coawst_d02_lat_min <= glac_lat <= pygem_prms.coawst_d02_lat_max) or 
                ~(pygem_prms.coawst_d02_lon_min <= glac_lon <= pygem_prms.coawst_d02_lon_max)):
                gcm_prec[glac,:] = gcm_prec_d01[glac,:]
                gcm_temp[glac,:] = gcm_temp_d01[glac,:]
                gcm_elev[glac] = gcm_elev_d01[glac]

    #%% ===== BIAS CORRECTIONS =====     
    # OPTION 1: Adjust temp and prec similar to Huss and Hock (2015) but limit maximum precipitation
    #  - temperature accounts for means and interannual variability
    #  - precipitation corrects
    if pygem_prms.option_bias_adjustment == 1:
        # Temperature bias correction
        gcm_temp_biasadj, gcm_elev_biasadj = temp_biasadj_HH2015(ref_temp, ref_elev, gcm_temp, dates_table_ref, 
                                                                 dates_table)
        # Precipitation bias correction
        gcm_prec_biasadj, gcm_elev_biasadj = prec_biasadj_opt1(ref_prec, ref_elev, gcm_prec, dates_table_ref, 
                                                               dates_table)
    
    # OPTION 2: Adjust temp and prec according to Huss and Hock (2015) accounts for means and interannual variability
    elif pygem_prms.option_bias_adjustment == 2:
        # Temperature bias correction
        gcm_temp_biasadj, gcm_elev_biasadj = temp_biasadj_HH2015(ref_temp, ref_elev, gcm_temp, dates_table_ref, 
                                                                 dates_table)
        # Precipitation bias correction
        gcm_prec_biasadj, gcm_elev_biasadj = prec_biasadj_HH2015(ref_prec, ref_elev, gcm_prec, dates_table_ref, 
                                                                 dates_table)
 
    if gcm_prec_biasadj.max() > 10:
        print('precipitation bias too high, needs to be modified')
        print(np.where(gcm_prec_biasadj > 10))
    elif gcm_prec_biasadj.min() < 0:
        print('Negative precipitation value')
        print(np.where(gcm_prec_biasadj < 0))
        
    #%% PLOT BIAS ADJUSTED DATA
    if option_plot_adj:
        print('plotting')
        plot_biasadj(ref_temp, gcm_temp_biasadj, ref_prec, gcm_prec, gcm_prec_biasadj, dates_table_ref, dates_table)

    #%% Export variables as global to view in variable explorer
    if args.option_parallels == 0:
        global main_vars
        main_vars = inspect.currentframe().f_locals

    print('\nProcessing time of', gcm_name, 'for', count,':',time.time()-time_start, 's')

#%% PARALLEL PROCESSING
if __name__ == '__main__':
    time_start = time.time()
    parser = getparser()
    args = parser.parse_args()
    
    if args.debug == 1:
        debug = True
    else:
        debug = False
    
    # Reference GCM name
    print('Reference climate data is:', pygem_prms.ref_gcm_name)
    
    # RGI glacier number
    if args.rgi_glac_number_fn is not None:
        with open(args.rgi_glac_number_fn, 'rb') as f:
            rgi_glac_number = pickle.load(f)
    else:
        rgi_glac_number = pygem_prms.rgi_glac_number   

    # Select glaciers and define chunks
    main_glac_rgi_all = modelsetup.selectglaciersrgitable(rgi_regionsO1=pygem_prms.rgi_regionsO1, rgi_regionsO2 = 'all',
                                                          rgi_glac_number=pygem_prms.rgi_glac_number)
    # Define chunk size for parallel processing
    if args.option_parallels != 0:
        num_cores = int(np.min([main_glac_rgi_all.shape[0], args.num_simultaneous_processes]))
        chunk_size = int(np.ceil(main_glac_rgi_all.shape[0] / num_cores))
    else:
        # if not running in parallel, chunk size is all glaciers
        chunk_size = main_glac_rgi_all.shape[0]

    # Read GCM names from argument parser
    gcm_name = args.gcm_list_fn
    if args.gcm_name is not None:
        gcm_list = [args.gcm_name]
        rcp_scenario = args.rcp
    elif args.gcm_list_fn == pygem_prms.ref_gcm_name:
        gcm_list = [pygem_prms.ref_gcm_name]
    else:
        with open(args.gcm_list_fn, 'r') as gcm_fn:
            gcm_list = gcm_fn.read().splitlines()
            rcp_scenario = os.path.basename(args.gcm_list_fn).split('_')[1]
            print('Found %d gcms to process'%(len(gcm_list)))

    # Loop through all GCMs
    for gcm_name in gcm_list:
        if args.rcp is None:
            print('Processing:', gcm_name)
        else:
            print('Processing:', gcm_name, rcp_scenario)
        # Pack variables for multiprocessing
        list_packed_vars = []
        n = 0
        for chunk in range(0, main_glac_rgi_all.shape[0], chunk_size):
            n = n + 1
            list_packed_vars.append([n, chunk, main_glac_rgi_all, chunk_size, gcm_name])
            
        # Parallel processing
        if args.option_parallels != 0:
            print('Processing in parallel with ' + str(num_cores) + ' cores...')
            with multiprocessing.Pool(args.num_simultaneous_processes) as p:
                p.map(main,list_packed_vars)
        # If not in parallel, then only should be one loop
        else:
            # Loop through the chunks and export bias adjustments
            for n in range(len(list_packed_vars)):
                main(list_packed_vars[n])

#    # Read GCM names from command file
#    with open(args.gcm_file, 'r') as gcm_fn:
#        gcm_list = gcm_fn.read().splitlines()
#        rcp_scenario = os.path.basename(args.gcm_file).split('_')[1]
#        print('Found %d gcm(s) to process'%(len(gcm_list)))
#
#    # Loop through all GCMs
#    for gcm_name in gcm_list:
#        # Pack variables for multiprocessing
#        list_packed_vars = []
#        n = 0
#        for chunk in range(0, main_glac_rgi_all.shape[0], chunk_size):
#            n += 1
#            list_packed_vars.append([n, chunk, main_glac_rgi_all, chunk_size, gcm_name])
#
#        # Parallel processing
#        if args.option_parallels != 0:
#            print('Processing', gcm_name, 'in parallel')
#            with multiprocessing.Pool(args.num_simultaneous_processes) as p:
#                p.map(main,list_packed_vars)
#        # No parallel processing
#        else:
#            print('Processing', gcm_name, 'without parallel')
#            # Loop through the chunks and export bias adjustments
#            for n in range(len(list_packed_vars)):
#                main(list_packed_vars[n])

    print('Total processing time:', time.time()-time_start, 's')

#%% ===== PLOTTING AND PROCESSING FOR MODEL DEVELOPMENT =====
    # Place local variables in variable explorer
    if args.option_parallels == 0:
        main_vars_list = list(main_vars.keys())
        gcm_name = main_vars['gcm_name']
        rcp_scenario = main_vars['rcp_scenario']
        main_glac_rgi = main_vars['main_glac_rgi']
        main_glac_hyps = main_vars['main_glac_hyps']
        main_glac_icethickness = main_vars['main_glac_icethickness']
        main_glac_width = main_vars['main_glac_width']
        elev_bins = main_vars['elev_bins']
        dates_table = main_vars['dates_table']
        dates_table_ref = main_vars['dates_table_ref']
        
        ref_temp = main_vars['ref_temp']
        ref_prec = main_vars['ref_prec']
        ref_elev = main_vars['ref_elev']
        ref_lr = main_vars['ref_lr']
        ref_lr_monthly_avg = main_vars['ref_lr_monthly_avg']
        gcm_temp = main_vars['gcm_temp']
        gcm_prec = main_vars['gcm_prec']
        gcm_elev = main_vars['gcm_elev']
        gcm_lr = main_vars['gcm_lr']
        gcm_temp_biasadj = main_vars['gcm_temp_biasadj']        
        gcm_prec_biasadj = main_vars['gcm_prec_biasadj']
