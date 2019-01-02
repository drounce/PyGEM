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
import pygem_input as input
import pygemfxns_modelsetup as modelsetup
import pygemfxns_massbalance as massbalance
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
    parser.add_argument('gcm_file', action='store', type=str, default=None,
                        help='text file full of gcm names')
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


def temp_biasadj_HH2015(ref_temp, gcm_temp, dates_table_ref, dates_table):
    """
    Huss and Hock (2015) temperature bias correction based on mean and interannual variability
    
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
    gcm_temp_bias_adj : np.array
        GCM temperature bias corrected to the reference climate dataset according to Huss and Hock (2015)
    """
    # GCM subset to agree with reference time period to calculate bias corrections
    gcm_subset_idx_start = np.where(dates_table.date.values == dates_table_ref.date.values[0])[0][0]
    gcm_subset_idx_end = np.where(dates_table.date.values == dates_table_ref.date.values[-1])[0][0]
    gcm_temp_subset = gcm_temp[:,gcm_subset_idx_start:gcm_subset_idx_end+1]
    
    # Remove spinup years, so adjustment performed over calibration period
    ref_temp_nospinup = ref_temp[:,input.spinupyears*12:]
    gcm_temp_nospinup = gcm_temp_subset[:,input.spinupyears*12:]        
    
    # Mean monthly temperature
    ref_temp_monthly_avg = (ref_temp_nospinup.reshape(-1,12).transpose()
                            .reshape(-1,int(ref_temp_nospinup.shape[1]/12)).mean(1).reshape(12,-1).transpose())
    gcm_temp_monthly_avg = (gcm_temp_nospinup.reshape(-1,12).transpose()
                            .reshape(-1,int(gcm_temp_nospinup.shape[1]/12)).mean(1).reshape(12,-1).transpose())
    # Monthly bias adjustment (additive)
    gcm_temp_monthly_adj = ref_temp_monthly_avg - gcm_temp_monthly_avg
    # Monthly temperature bias adjusted according to monthly average
    t_mt = gcm_temp + np.tile(gcm_temp_monthly_adj, int(gcm_temp.shape[1]/12))
    
    # Calculate monthly standard deviation of temperature
    ref_temp_monthly_std = (ref_temp_nospinup.reshape(-1,12).transpose()
                            .reshape(-1,int(ref_temp_nospinup.shape[1]/12)).std(1).reshape(12,-1).transpose())
    gcm_temp_monthly_std = (gcm_temp_nospinup.reshape(-1,12).transpose()
                            .reshape(-1,int(gcm_temp_nospinup.shape[1]/12)).std(1).reshape(12,-1).transpose())
    variability_monthly_std = ref_temp_monthly_std / gcm_temp_monthly_std
    # Mean monthly temperature bias adjusted according to monthly average
    #  t_m25avg is the avg monthly temp in a 25-year period around the given year
    N = 25
    t_m_Navg = np.zeros(t_mt.shape)
    for month in range(0,12):
        t_m_subset = t_mt[:,month::12]
        # Uniform filter computes running average and uses 'reflects' values at borders
        t_m_Navg_subset = uniform_filter(t_m_subset,size=(1,N))
        t_m_Navg[:,month::12] = t_m_Navg_subset
    
    gcm_temp_bias_adj = t_m_Navg + (t_mt - t_m_Navg) * np.tile(variability_monthly_std, int(gcm_temp.shape[1]/12))
    
    return gcm_temp_bias_adj


def prec_biasadj_HH2015(ref_prec, gcm_prec, dates_table_ref, dates_table):
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
    
    Returns
    -------
    gcm_prec_bias_adj : np.array
        GCM precipitation bias corrected to the reference climate dataset according to Huss and Hock (2015)
    """
    # GCM subset to agree with reference time period to calculate bias corrections
    gcm_subset_idx_start = np.where(dates_table.date.values == dates_table_ref.date.values[0])[0][0]
    gcm_subset_idx_end = np.where(dates_table.date.values == dates_table_ref.date.values[-1])[0][0]
    gcm_prec_subset = gcm_prec[:,gcm_subset_idx_start:gcm_subset_idx_end+1]
    
    # Remove spinup years, so adjustment performed over calibration period
    ref_prec_nospinup = ref_prec[:,input.spinupyears*12:]
    gcm_prec_nospinup = gcm_prec_subset[:,input.spinupyears*12:]
    
    # PRECIPITATION BIAS CORRECTIONS
    # Calculate monthly mean precipitation
    ref_prec_monthly_avg = (ref_prec_nospinup.reshape(-1,12).transpose()
                            .reshape(-1,int(ref_prec_nospinup.shape[1]/12)).mean(1).reshape(12,-1).transpose())
    gcm_prec_monthly_avg = (gcm_prec_nospinup.reshape(-1,12).transpose()
                            .reshape(-1,int(gcm_prec_nospinup.shape[1]/12)).mean(1).reshape(12,-1).transpose())
    bias_adj_prec_monthly = ref_prec_monthly_avg / gcm_prec_monthly_avg
    # Bias adjusted precipitation accounting for differences in monthly mean
    gcm_prec_bias_adj = gcm_prec * np.tile(bias_adj_prec_monthly, int(gcm_temp.shape[1]/12))
    
    return gcm_prec_bias_adj


def prec_biasadj_opt1(ref_prec, gcm_prec, dates_table_ref, dates_table):
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
    gcm_prec_bias_adj : np.array
        GCM precipitation bias corrected to the reference climate dataset according to Huss and Hock (2015)
    """
    # GCM subset to agree with reference time period to calculate bias corrections
    gcm_subset_idx_start = np.where(dates_table.date.values == dates_table_ref.date.values[0])[0][0]
    gcm_subset_idx_end = np.where(dates_table.date.values == dates_table_ref.date.values[-1])[0][0]
    gcm_prec_subset = gcm_prec[:,gcm_subset_idx_start:gcm_subset_idx_end+1]
    
    # Remove spinup years, so adjustment performed over calibration period
    ref_prec_nospinup = ref_prec[:,input.spinupyears*12:]
    gcm_prec_nospinup = gcm_prec_subset[:,input.spinupyears*12:]
    
    # PRECIPITATION BIAS CORRECTIONS
    # Monthly mean precipitation
    ref_prec_monthly_avg = (ref_prec_nospinup.reshape(-1,12).transpose()
                            .reshape(-1,int(ref_prec_nospinup.shape[1]/12)).mean(1).reshape(12,-1).transpose())
    gcm_prec_monthly_avg = (gcm_prec_nospinup.reshape(-1,12).transpose()
                            .reshape(-1,int(gcm_prec_nospinup.shape[1]/12)).mean(1).reshape(12,-1).transpose())
    bias_adj_prec_monthly = ref_prec_monthly_avg / gcm_prec_monthly_avg
    # Bias adjusted precipitation accounting for differences in monthly mean
    gcm_prec_bias_adj = gcm_prec * np.tile(bias_adj_prec_monthly, int(gcm_prec.shape[1]/12))
    
    # Monthly maximum used for identifying outliers
    ref_prec_monthly_max = (ref_prec_nospinup.reshape(-1,12).transpose()
                            .reshape(-1,int(ref_prec_nospinup.shape[1]/12)).max(1).reshape(12,-1).transpose())
    gcm_prec_max_check = np.tile(ref_prec_monthly_max, int(gcm_prec_bias_adj.shape[1]/12))        
    # For wet years in the future, enable monthly maximum to be increased by the annual increase in precipitation
    # for that given wet year.
    gcm_prec_annual = annual_sum_2darray(gcm_prec)
    gcm_prec_annual_norm = gcm_prec_annual / gcm_prec_annual.mean(1)[:,np.newaxis]
    gcm_prec_annual_norm_repeated = np.repeat(gcm_prec_annual_norm, 12).reshape(gcm_prec_bias_adj.shape)
    gcm_prec_max_check_adj = gcm_prec_max_check * gcm_prec_annual_norm_repeated
    gcm_prec_max_check_adj[gcm_prec_max_check_adj < gcm_prec_max_check] = (
            gcm_prec_max_check[gcm_prec_max_check_adj < gcm_prec_max_check])
    
    # Identify outliers and replace with monthly mean adjusted for the normalized annual variation
    outlier_replacement = (gcm_prec_annual_norm_repeated * 
                           np.tile(ref_prec_monthly_avg, int(gcm_prec_bias_adj.shape[1]/12)))
    gcm_prec_bias_adj[gcm_prec_bias_adj > gcm_prec_max_check_adj] = (
            outlier_replacement[gcm_prec_bias_adj > gcm_prec_max_check_adj])
    return gcm_prec_bias_adj
    
    
def plot_biasadj(ref_temp, gcm_temp_bias_adj, ref_prec, gcm_prec, gcm_prec_bias_adj, dates_table_ref, dates_table):
    """
    Plot bias adjusted climate data to check values are realistic and adjustment performing as expected
    """
    fig_h = 16
    fig_w = 12
    fig, ax = plt.subplots(5, 1, squeeze=False, figsize=(fig_w,fig_h), gridspec_kw = {'wspace':0, 'hspace':0.2})
    #  Subplot #1: Monthly temperatures with annual on top
    gcm_temp_bias_adj_annual_avg = annual_avg_2darray(gcm_temp_bias_adj)
    ax[0,0].plot(dates_table_ref.date.values, ref_temp[0,:], color='b', linewidth=1, label='era')
    ax[0,0].plot(dates_table.date.values, gcm_temp_bias_adj[0,:], color='r', linewidth=1, label='gcm_adj')
    ax[0,0].plot(dates_table.date.values[::12], gcm_temp_bias_adj_annual_avg[0,:], color='k', linewidth=2, 
                 label='gcm_adj_yr')
    ax[0,0].set_ylabel('Temp [degC]')
    ax[0,0].legend()    
    
    # GCM subset to agree with reference time period to calculate bias corrections
    gcm_subset_idx_start = np.where(dates_table.date.values == dates_table_ref.date.values[0])[0][0]
    gcm_subset_idx_end = np.where(dates_table.date.values == dates_table_ref.date.values[-1])[0][0]
    gcm_prec_subset = gcm_prec[:,gcm_subset_idx_start:gcm_subset_idx_end+1]
    
    # Remove spinup years, so adjustment performed over calibration period
    ref_prec_nospinup = ref_prec[:,input.spinupyears*12:]
    gcm_prec_nospinup = gcm_prec_subset[:,input.spinupyears*12:]
    
    # PRECIPITATION BIAS CORRECTIONS
    # Monthly mean precipitation
    ref_prec_monthly_avg = (ref_prec_nospinup.reshape(-1,12).transpose()
                            .reshape(-1,int(ref_prec_nospinup.shape[1]/12)).mean(1).reshape(12,-1).transpose())
    gcm_prec_monthly_avg = (gcm_prec_nospinup.reshape(-1,12).transpose()
                            .reshape(-1,int(gcm_prec_nospinup.shape[1]/12)).mean(1).reshape(12,-1).transpose())
    
    # Subplot #2: Monthly Precipitation bias adjusted
    ax[1,0].plot(dates_table_ref.date.values, ref_prec[0,:], color='b', linewidth=1, label='era')
    ax[1,0].plot(dates_table.date.values, gcm_prec_bias_adj[0,:], color='k', linewidth=1, label='gcm_adj')
    ax[1,0].set_ylabel('Prec [m]')
    ax[1,0].legend()
    
    # Subplot #3: Monthly Precipitation (reference, gcm, and gcm corrected)
    ax[2,0].plot(dates_table_ref.date.values, ref_prec[0,:], color='b', linewidth=1, label='era')
    ax[2,0].plot(dates_table_ref.date.values, gcm_prec_subset[0,:], color='r', linewidth=1, label='gcm')
    ax[2,0].plot(dates_table_ref.date.values, gcm_prec_bias_adj[0,0:ref_prec.shape[1]], color='k', linewidth=1, 
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
    
    gcm_prec_biasadj_monthly_avg = monthly_avg_2darray(gcm_prec_bias_adj[:,0:ref_prec.shape[1]])
    gcm_prec_biasadj_monthly_std = monthly_std_2darray(gcm_prec_bias_adj[:,0:ref_prec.shape[1]])
    gcm_prec_biasadj_monthly_low = gcm_prec_biasadj_monthly_avg[0,:] - gcm_prec_biasadj_monthly_std[0,:]
    gcm_prec_biasadj_monthly_high = gcm_prec_biasadj_monthly_avg[0,:] + gcm_prec_biasadj_monthly_std[0,:]
    ax[4,0].plot(np.arange(0,12), gcm_prec_biasadj_monthly_avg[0,:], color='k', linewidth=1, label='gcm_adj')
    ax[4,0].fill_between(np.arange(0,12), gcm_prec_biasadj_monthly_low, gcm_prec_biasadj_monthly_high, 
                         facecolor='k', alpha=0.2) 
    ax[4,0].set_ylabel('Prec [m]')
    ax[4,0].xaxis.set_major_locator(plt.MultipleLocator(1))
    ax[4,0].set_xticklabels(month_labels)
    ax[4,0].legend()
    
    # Save figure
    fig.set_size_inches(fig_w, fig_h)
    figure_fn = ('biasadjplots_' + gcm_name + '_' + rcp_scenario + '_biasadjopt' + str(input.option_bias_adjustment) +
                 '_' + str(input.rgi_regionsO1[0]) + '-' + str(rgi_glac_number[0]) + '.png')
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
    rcp_scenario = os.path.basename(args.gcm_file).split('_')[1]

    # ===== LOAD OTHER GLACIER DATA =====
    main_glac_rgi = main_glac_rgi_all.iloc[chunk:chunk + chunk_size, :]
    # Glacier hypsometry [km**2], total area
    main_glac_hyps = modelsetup.import_Husstable(main_glac_rgi, input.rgi_regionsO1, input.hyps_filepath,
                                                 input.hyps_filedict, input.hyps_colsdrop)
    # Ice thickness [m], average
    main_glac_icethickness = modelsetup.import_Husstable(main_glac_rgi, input.rgi_regionsO1, input.thickness_filepath,
                                                         input.thickness_filedict, input.thickness_colsdrop)
    main_glac_hyps[main_glac_icethickness == 0] = 0
    # Width [km], average
    main_glac_width = modelsetup.import_Husstable(main_glac_rgi, input.rgi_regionsO1, input.width_filepath,
                                                  input.width_filedict, input.width_colsdrop)
    elev_bins = main_glac_hyps.columns.values.astype(int)

    # Select dates including future projections
    # If reference climate data starts or ends before or after the GCM data, then adjust reference climate data such
    # that the reference and GCM span the same period of time.
    if input.startyear >= input.gcm_startyear:
        ref_startyear = input.startyear
    else:
        ref_startyear = input.gcm_startyear
    if input.endyear <= input.gcm_endyear:
        ref_endyear = input.endyear
    else:
        ref_endyear = input.gcm_endyear
    dates_table_ref = modelsetup.datesmodelrun(startyear=ref_startyear, endyear=ref_endyear, 
                                               spinupyears=input.spinupyears)
    dates_table = modelsetup.datesmodelrun(startyear=input.gcm_startyear, endyear=input.gcm_endyear,
                                           spinupyears=input.spinupyears)

    # ===== LOAD CLIMATE DATA =====
    # Reference climate data
    ref_gcm = class_climate.GCM(name=input.ref_gcm_name)
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
        gcm_lr = np.tile(ref_lr_monthly_avg, int(gcm_temp.shape[1]/12))
    # COAWST data has two domains, so need to merge the two domains
    if gcm_name == 'COAWST':
        gcm_temp_d01, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(gcm.temp_fn_d01, gcm.temp_vn, main_glac_rgi, 
                                                                         dates_table)
        gcm_prec_d01, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(gcm.prec_fn_d01, gcm.prec_vn, main_glac_rgi, 
                                                                         dates_table)
        gcm_elev_d01 = gcm.importGCMfxnearestneighbor_xarray(gcm.elev_fn_d01, gcm.elev_vn, main_glac_rgi)
        # Check if glacier outside of high-res (d02) domain
        for glac in range(main_glac_rgi.shape[0]):
            glac_lat = main_glac_rgi.loc[glac,input.rgi_lat_colname]
            glac_lon = main_glac_rgi.loc[glac,input.rgi_lon_colname]
            if (~(input.coawst_d02_lat_min <= glac_lat <= input.coawst_d02_lat_max) or 
                ~(input.coawst_d02_lon_min <= glac_lon <= input.coawst_d02_lon_max)):
                gcm_prec[glac,:] = gcm_prec_d01[glac,:]
                gcm_temp[glac,:] = gcm_temp_d01[glac,:]
                gcm_elev[glac] = gcm_elev_d01[glac]

    #%% ===== BIAS CORRECTIONS =====     
    # OPTION 2: Adjust temp and prec according to Huss and Hock (2015) accounts for means and interannual variability
    if input.option_bias_adjustment == 2:
        # Temperature bias correction
        gcm_temp_bias_adj = temp_biasadj_HH2015(ref_temp, gcm_temp, dates_table_ref, dates_table)
        # Precipitation bias correction
        gcm_prec_bias_adj = prec_biasadj_HH2015(ref_prec, gcm_prec, dates_table_ref, dates_table)

    # OPTION 1: Adjust temp and prec similar to Huss and Hock (2015) but limit maximum precipitation
    #  - temperature accounts for means and interannual variability
    #  - precipitation corrects
    elif input.option_bias_adjustment == 1:
        # Temperature bias correction
        gcm_temp_bias_adj = temp_biasadj_HH2015(ref_temp, gcm_temp, dates_table_ref, dates_table)
        # Precipitation bias correction
        gcm_prec_bias_adj = prec_biasadj_opt1(ref_prec, gcm_prec, dates_table_ref, dates_table)
            
    #%% CHECK FOR UNREALISTIC VALUES
    if gcm_prec_bias_adj.max() > 10:
        print('precipitation bias too high, needs to be modified')
        print(np.where(gcm_prec_bias_adj > 10))
    elif gcm_prec_bias_adj.min() < 0:
        print('Negative precipitation value')
        print(np.where(gcm_prec_bias_adj < 0))
        
    #%% PLOT BIAS ADJUSTED DATA
    if option_plot_adj:
        plot_biasadj(ref_temp, gcm_temp_bias_adj, ref_prec, gcm_prec, gcm_prec_bias_adj, dates_table_ref, dates_table)

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
    print('Reference climate data is:', input.ref_gcm_name)
    
    # RGI glacier number
    if args.rgi_glac_number_fn is not None:
        with open(args.rgi_glac_number_fn, 'rb') as f:
            rgi_glac_number = pickle.load(f)
    else:
        rgi_glac_number = input.rgi_glac_number   

    # Select glaciers and define chunks
    main_glac_rgi_all = modelsetup.selectglaciersrgitable(rgi_regionsO1=input.rgi_regionsO1, rgi_regionsO2 = 'all',
                                                          rgi_glac_number=input.rgi_glac_number)
    # Define chunk size for parallel processing
    if args.option_parallels != 0:
        num_cores = int(np.min([main_glac_rgi_all.shape[0], args.num_simultaneous_processes]))
        chunk_size = int(np.ceil(main_glac_rgi_all.shape[0] / num_cores))
    else:
        # if not running in parallel, chunk size is all glaciers
        chunk_size = main_glac_rgi_all.shape[0]

    # Read GCM names from command file
    with open(args.gcm_file, 'r') as gcm_fn:
        gcm_list = gcm_fn.read().splitlines()
        rcp_scenario = os.path.basename(args.gcm_file).split('_')[1]
        print('Found %d gcm(s) to process'%(len(gcm_list)))

    # Loop through all GCMs
    for gcm_name in gcm_list:
        # Pack variables for multiprocessing
        list_packed_vars = []
        n = 0
        for chunk in range(0, main_glac_rgi_all.shape[0], chunk_size):
            n += 1
            list_packed_vars.append([n, chunk, main_glac_rgi_all, chunk_size, gcm_name])

        # Parallel processing
        if args.option_parallels != 0:
            print('Processing', gcm_name, 'in parallel')
            with multiprocessing.Pool(args.num_simultaneous_processes) as p:
                p.map(main,list_packed_vars)
        # No parallel processing
        else:
            print('Processing', gcm_name, 'without parallel')
            # Loop through the chunks and export bias adjustments
            for n in range(len(list_packed_vars)):
                main(list_packed_vars[n])

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
        gcm_temp_bias_adj = main_vars['gcm_temp_bias_adj']        
        gcm_prec_bias_adj = main_vars['gcm_prec_bias_adj']