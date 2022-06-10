"""Run the model calibration"""
# Spyder cannot run parallels, so always set -option_parallels=0 when testing in Spyder.

# Built-in libraries
import os
import argparse
import multiprocessing
import resource
import time
import inspect
# External libraries
from datetime import datetime
import pandas as pd
import numpy as np
import xarray as xr
import pymc
from pymc import deterministic
from scipy.optimize import minimize
import pickle
from scipy import stats
# Local libraries
import pygem.pygem_input as pygem_prms
import pygemfxns_modelsetup as modelsetup
import pygemfxns_massbalance as massbalance
import class_climate
#import class_mbdata

#%% FUNCTIONS
def getparser():
    """
    Use argparse to add arguments from the command line

    Parameters
    ----------
    ref_gcm_name (optional) : str
        reference gcm name
    num_simultaneous_processes (optional) : int
        number of cores to use in parallels
    option_parallels (optional) : int
        switch to use parallels or not
    rgi_glac_number_fn : str
        filename of .pkl file containing a list of glacier numbers which is used to run batches on the supercomputer
    rgi_glac_number : str
        rgi glacier number to run for supercomputer
    progress_bar : int
        Switch for turning the progress bar on or off (default = 0 (off))
    debug : int
        Switch for turning debug printing on or off (default = 0 (off))

    Returns
    -------
    Object containing arguments and their respective values.
    """
    parser = argparse.ArgumentParser(description="run calibration in parallel")
    # add arguments
    parser.add_argument('-ref_gcm_name', action='store', type=str, default=pygem_prms.ref_gcm_name,
                        help='reference gcm name')
    parser.add_argument('-num_simultaneous_processes', action='store', type=int, default=4,
                        help='number of simultaneous processes (cores) to use')
    parser.add_argument('-option_parallels', action='store', type=int, default=1,
                        help='Switch to use or not use parallels (1 - use parallels, 0 - do not)')
    parser.add_argument('-rgi_glac_number_fn', action='store', type=str, default=None,
                        help='Filename containing list of rgi_glac_number, helpful for running batches on spc')
    parser.add_argument('-progress_bar', action='store', type=int, default=0,
                        help='Boolean for the progress bar to turn it on or off (default 0 is off)')
    parser.add_argument('-debug', action='store', type=int, default=0,
                        help='Boolean for debugging to turn it on or off (default 0 is off)')
    parser.add_argument('-rgi_glac_number', action='store', type=str, default=None,
                        help='rgi glacier number for supercomputer')
    return parser


def weighted_percentile(sorted_list, weights, percentile):
    """
    Calculate weighted percentile of a sorted list
    """
    weights_cumsum_norm_high = np.cumsum(weights) / np.sum(weights)
#     print(weights_cumsum_norm_high)
    weights_norm = weights / np.sum(weights)
    weights_cumsum_norm_low = weights_cumsum_norm_high - weights_norm
#     print(weights_cumsum_norm_low)
    
    percentile_idx_high = np.where(weights_cumsum_norm_high >= percentile)[0][0]
#     print(percentile_idx_high)
    percentile_idx_low = np.where(weights_cumsum_norm_low <= percentile)[0][-1]
#     print(percentile_idx_low)
    
    if percentile_idx_low == percentile_idx_high:
        value_percentile = sorted_list[percentile_idx_low]
    else:
        value_percentile = np.mean([sorted_list[percentile_idx_low], sorted_list[percentile_idx_high]])

    return value_percentile


def mb_mwea_calc(modelparameters, glacier_rgi_table, glacier_area_initial, icethickness_initial, width_initial, 
     elev_bins, glacier_gcm_temp, glacier_gcm_tempstd, glacier_gcm_prec, glacier_gcm_elev, glacier_gcm_lrgcm, 
     glacier_gcm_lrglac, dates_table, t1_idx, t2_idx, option_areaconstant=1, glacier_debrismf=None):
    """
    Run the mass balance and calculate the mass balance [mwea]
    """
    # Mass balance calculations
    (glac_bin_temp, glac_bin_prec, glac_bin_acc, glac_bin_refreeze, glac_bin_snowpack, glac_bin_melt,
     glac_bin_frontalablation, glac_bin_massbalclim, glac_bin_massbalclim_annual, glac_bin_area_annual,
     glac_bin_icethickness_annual, glac_bin_width_annual, glac_bin_surfacetype_annual,
     glac_wide_massbaltotal, glac_wide_runoff, glac_wide_snowline, glac_wide_snowpack,
     glac_wide_area_annual, glac_wide_volume_annual, glac_wide_ELA_annual, offglac_wide_prec,
     offglac_wide_refreeze, offglac_wide_melt, offglac_wide_snowpack, offglac_wide_runoff) = (
        massbalance.runmassbalance(modelparameters, glacier_rgi_table, glacier_area_initial, icethickness_initial,
                                   width_initial, elev_bins, glacier_gcm_temp, glacier_gcm_tempstd, glacier_gcm_prec,
                                   glacier_gcm_elev, glacier_gcm_lrgcm, glacier_gcm_lrglac, dates_table,
                                   option_areaconstant=option_areaconstant, glacier_debrismf=glacier_debrismf,
                                   debug=False)) 
    
    # Compute glacier volume change for every time step and use this to compute mass balance
    glac_wide_area = glac_wide_area_annual[:-1].repeat(12)
    # Mass change [km3 mwe]
    #  mb [mwea] * (1 km / 1000 m) * area [km2]
    glac_wide_masschange = glac_wide_massbaltotal[t1_idx:t2_idx+1] / 1000 * glac_wide_area[t1_idx:t2_idx+1]
    # Mean annual mass balance [mwea]
    mb_mwea = (glac_wide_masschange.sum() / glac_wide_area[0] * 1000 / (glac_wide_masschange.shape[0] / 12))

    return mb_mwea


def main(list_packed_vars):
    """
    Model calibration

    Parameters
    ----------
    list_packed_vars : list
        list of packed variables that enable the use of parallels

    Returns
    -------
    netcdf files of the calibration output
        Depending on the calibration scheme additional output may be exported as well
    """
    #%%
    # Unpack variables
    main_glac_rgi = list_packed_vars[0]
    gcm_name = list_packed_vars[1]
    
    time_start = time.time()
    parser = getparser()
    args = parser.parse_args()

    if args.debug == 1:
        debug = True
    else:
        debug = False
        
    # ===== MASS BALANCE DATA AND DATES TABLE =====
    data_source = 'regional'
#    data_source = 'individual_glaciers'

    if data_source in ['individual_glaciers']:
        dates_table = modelsetup.datesmodelrun(startyear=2000, endyear=2018, spinupyears=0)
        mb_shean_fullfn = pygem_prms.shean_fp + pygem_prms.shean_fn
        mb_shean_df = pd.read_csv(mb_shean_fullfn)
        mb_shean_df['RGIId'] = ['RGI60-' + str(int(x)) + '.' + str(int(np.round((x - int(x)) * 1e5,0))).zfill(5) 
                                for x in mb_shean_df.RGIId.values]
    elif data_source in ['regional']:
        dates_table = modelsetup.datesmodelrun(startyear=2006, endyear=2015, spinupyears=0)
        roi_mbobs_dict = {'01': [-0.70, 0.18],
                          '02': [-0.50, 0.91],
                          '03': [-0.38, 0.80],
                          '04': [-0.80, 0.22],
                          '05': [-0.57, 0.20],
                          '06': [-0.69, 0.26],
                          '07': [-0.27, 0.17],
                          '08': [-0.66, 0.27],
                          '09': [-0.30, 0.27],
                          '10': [-0.40, 0.31],
                          '11': [-0.91, 0.70],
                          '12': [-0.88, 0.57],
                          '13': [-0.19, 0.15],
                          '14': [-0.11, 0.15],
                          '15': [-0.44, 0.15],
                          '16': [-0.59, 0.58],
                          '17': [-0.86, 0.17],
                          '18': [-0.59, 1.14]}

    # ===== LOAD CLIMATE DATA =====
    gcm = class_climate.GCM(name=gcm_name)
    # Air temperature [degC], Air temperature Std [K], Precipitation [m], Elevation [masl], Lapse rate [K m-1]
    gcm_temp, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(gcm.temp_fn, gcm.temp_vn, main_glac_rgi, dates_table)
    gcm_prec, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(gcm.prec_fn, gcm.prec_vn, main_glac_rgi, dates_table)
    gcm_elev = gcm.importGCMfxnearestneighbor_xarray(gcm.elev_fn, gcm.elev_vn, main_glac_rgi)
    # Air temperature standard deviation [K]
    if pygem_prms.option_ablation != 2 or gcm_name not in ['ERA5']:
        gcm_tempstd = np.zeros(gcm_temp.shape)
    elif gcm_name in ['ERA5']:
        gcm_tempstd, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(gcm.tempstd_fn, gcm.tempstd_vn, 
                                                                        main_glac_rgi, dates_table)
    # Lapse rate [K m-1]
    if gcm_name in ['ERA-Interim', 'ERA5']:
        gcm_lr, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(gcm.lr_fn, gcm.lr_vn, main_glac_rgi, dates_table)
    else:
        # Mean monthly lapse rate
        ref_lr_monthly_avg = np.genfromtxt(gcm.lr_fp + gcm.lr_fn, delimiter=',')
        gcm_lr = np.tile(ref_lr_monthly_avg, int(gcm_temp.shape[1]/12))
        
    # Huss and Hock (2015) parameters and bounds
    tempchange_init = 0
    tempchange_bndlow = -10
    tempchange_bndhigh = 10
    precfactor_init = 1.5
    precfactor_bndlow = 0.8
    precfactor_bndhigh = 2
    ddfsnow_init = 0.003
    ddfsnow_bndlow = 0.00175
    ddfsnow_bndhigh = 0.0045
    ddfsnow_iceratio = 0.5
    
    # Area change rate dictionary to account for clean ice retreat scenario
    area_chg_rate_dict = {'01':[-0.42, 0.23],
                          '02':[-0.54, 0.24],
                          '03':[-0.07, 0.03],
                          '04':[-0.08, 0.05],
                          '05':[-0.18, 0.23],
                          '06':[-0.58, 0.23],
                          '07':[-0.26, 0.23],
                          '08':[-0.18, 0.11],
                          '09':[-0.18, 0.23],
                          '10':[-0.52, 0.67],
                          '11':[-0.93, 0.47],
                          '12':[-0.18, 0.23],
                          '13':[-0.18, 0.17],
                          '14':[-0.36, 0.08],
                          '15':[-0.47, 0.13],
                          '16':[-1.19, 0.54],
                          '17':[-0.20, 0.08],
                          '18':[-0.69, 0.23]}
    
    for nglac, rgiid in enumerate(main_glac_rgi.rgino_str):
        if debug:
            print(nglac, rgiid)
        
        # ===== LOAD GLACIER DATA =====
        binnedcsv = pd.read_csv(main_glac_rgi.loc[nglac,'binned_fullfn'])
        
        if prescribe_retreat:
            roi = rgiid.split('.')[0].zfill(2)
            dc_perc_min = area_chg_rate_dict[roi][0] * (main_glac_rgi.loc[nglac,'RefYear'] - 2015)
            binnedcsv['area_cumsum_%'] = (np.cumsum(binnedcsv.z1_bin_area_valid_km2) / 
                                          binnedcsv.z1_bin_area_valid_km2.sum() * 100)
            binnedcsv = binnedcsv[binnedcsv['area_cumsum_%'] > dc_perc_min].copy()
            binnedcsv.reset_index(inplace=True, drop=True)
        
        # Elevation bin statistics
        elev_bins = binnedcsv['bin_center_elev_m'].values
        bins_area = binnedcsv['z1_bin_area_valid_km2'].values
        
        main_glac_rgi['Zmed'] = weighted_percentile(elev_bins, bins_area, 0.5)
        main_glac_rgi['Zmin'] = elev_bins.min()
        main_glac_rgi['Zmax'] = elev_bins.max()
        
        # Set model parameters
        modelparameters = [pygem_prms.lrgcm, pygem_prms.lrglac, precfactor_init, pygem_prms.precgrad, ddfsnow_init, 
                           ddfsnow_init/ddfsnow_iceratio, pygem_prms.tempsnow, tempchange_init]

        # Select subsets of data
        glacier_rgi_table = main_glac_rgi.loc[main_glac_rgi.index.values[nglac], :]
        glacier_gcm_elev = gcm_elev[nglac]
        glacier_gcm_prec = gcm_prec[nglac,:]
        glacier_gcm_temp = gcm_temp[nglac,:]
        glacier_gcm_tempstd = gcm_tempstd[nglac,:]
        glacier_gcm_lrgcm = gcm_lr[nglac,:]
        glacier_gcm_lrglac = glacier_gcm_lrgcm.copy()
        glacier_area_initial = binnedcsv['z1_bin_area_valid_km2'].values
        icethickness_initial = binnedcsv['H_mean']
        glacier_debrismf = binnedcsv[mf_cn].values

        # Glacier widths (from OGGM)
        oggm_widths_fp = (pygem_prms.main_directory + '/../oggm_widths/' + 
                          main_glac_rgi.loc[nglac,'RGIId'].split('.')[0] + '/')
        widths_fn = main_glac_rgi.loc[nglac,'RGIId'] + '_widths_m.csv'
        try:
            # Add width (km) to each elevation bin
            widths_df = pd.read_csv(oggm_widths_fp + widths_fn)
            elev_nearidx = (np.abs(np.array(elev_bins)[:,np.newaxis] - widths_df['elev'].values).argmin(axis=1))
            width_initial = widths_df.loc[elev_nearidx,'width_m'].values / 1000
        except:
            width_initial = np.zeros(glacier_area_initial.shape[0])
            

        # Mass balance data
        if data_source in ['individual_glaciers']:
            assert glacier_rgi_table.O1Region in [13,14,15], 'Individual mb data not available'
            mb_shean_idx = np.where(glacier_rgi_table.RGIId == mb_shean_df.RGIId.values)[0][0]
            observed_massbal = mb_shean_df.loc[mb_shean_idx,'mb_mwea']
            observed_massbal_std = mb_shean_df.loc[mb_shean_idx,'mb_mwea_sigma']
        elif data_source in ['regional']:
            observed_massbal = roi_mbobs_dict[str(glacier_rgi_table.O1Region).zfill(2)][0]
            observed_massbal_std = roi_mbobs_dict[str(glacier_rgi_table.O1Region).zfill(2)][1]
        
        t1_idx = 0
        t2_idx = dates_table.shape[0]
            
        if debug:
            print('obs_mwea:', np.round(observed_massbal,2), '+/-', np.round(observed_massbal_std,2))            
            
#        #%%
##        dates_table = dates_table.loc[0:11,:]
##        glacier_gcm_temp = glacier_gcm_temp[0:12]
##        glacier_gcm_tempstd = glacier_gcm_tempstd[0:12]
##        glacier_gcm_prec = glacier_gcm_prec[0:12]
##        glacier_gcm_lrgcm = glacier_gcm_lrgcm[0:12]
##        glacier_gcm_lrglac = glacier_gcm_lrglac[0:12]
#        option_areaconstant = 1
#        (glac_bin_temp, glac_bin_prec, glac_bin_acc, glac_bin_refreeze, glac_bin_snowpack, glac_bin_melt,
#         glac_bin_frontalablation, glac_bin_massbalclim, glac_bin_massbalclim_annual, glac_bin_area_annual,
#         glac_bin_icethickness_annual, glac_bin_width_annual, glac_bin_surfacetype_annual,
#         glac_wide_massbaltotal, glac_wide_runoff, glac_wide_snowline, glac_wide_snowpack,
#         glac_wide_area_annual, glac_wide_volume_annual, glac_wide_ELA_annual, offglac_wide_prec,
#         offglac_wide_refreeze, offglac_wide_melt, offglac_wide_snowpack, offglac_wide_runoff) = (
#            massbalance.runmassbalance(modelparameters, glacier_rgi_table, glacier_area_initial, icethickness_initial,
#                                       width_initial, elev_bins, glacier_gcm_temp, glacier_gcm_tempstd, glacier_gcm_prec,
#                                       glacier_gcm_elev, glacier_gcm_lrgcm, glacier_gcm_lrglac, dates_table,
#                                       option_areaconstant=option_areaconstant, glacier_debrismf=glacier_debrismf,
#                                       debug=True)) 
#        
#        # Compute glacier volume change for every time step and use this to compute mass balance
#        glac_wide_area = glac_wide_area_annual[:-1].repeat(12)
#        # Mass change [km3 mwe]
#        #  mb [mwea] * (1 km / 1000 m) * area [km2]
#        glac_wide_masschange = glac_wide_massbaltotal[t1_idx:t2_idx+1] / 1000 * glac_wide_area[t1_idx:t2_idx+1]
#        # Mean annual mass balance [mwea]
#        mb_mwea = (glac_wide_masschange.sum() / glac_wide_area[0] * 1000 / (glac_wide_masschange.shape[0] / 12))
#        
##        mb_mwea = mb_mwea_calc(modelparameters, glacier_rgi_table, glacier_area_initial, icethickness_initial, 
##                               width_initial, elev_bins, glacier_gcm_temp, glacier_gcm_tempstd, glacier_gcm_prec, 
##                               glacier_gcm_elev, glacier_gcm_lrgcm, glacier_gcm_lrglac, dates_table, t1_idx, t2_idx, 
##                               option_areaconstant=1, glacier_debrismf=glacier_debrismf)
#        print(np.round(mb_mwea,2))
        
        #%%
        # Huss and Hock (2015) model calibration steps
        if pygem_prms.option_calibration == 3:

            def objective(modelparameters_subset):
                """ Objective function for mass balance data.
                
                Parameters
                ----------
                modelparameters_subset : list of model parameters to calibrate
                    [precipitation factor, precipitation gradient, degree-day factor of snow, temperature bias]
                    
                Returns
                -------
                mb_dif_mwea : difference in modeled vs observed mass balance [mwea]
                """
                # Use a subset of model parameters to reduce number of constraints required
                modelparameters[2] = modelparameters_subset[0]
                modelparameters[3] = modelparameters_subset[1]
                modelparameters[4] = modelparameters_subset[2]
                modelparameters[5] = modelparameters[4] / ddfsnow_iceratio
                modelparameters[7] = modelparameters_subset[3]
                mb_mwea = mb_mwea_calc(modelparameters, glacier_rgi_table, glacier_area_initial, icethickness_initial, 
                                       width_initial, elev_bins, glacier_gcm_temp, glacier_gcm_tempstd, 
                                       glacier_gcm_prec, glacier_gcm_elev, glacier_gcm_lrgcm, glacier_gcm_lrglac, 
                                       dates_table, t1_idx, t2_idx, option_areaconstant=1, 
                                       glacier_debrismf=glacier_debrismf)
                # Differnece [mwea] = Observed mass balance [mwea] - mb_mwea
                mb_dif_mwea_abs = abs(observed_massbal - mb_mwea)    
                return mb_dif_mwea_abs


        def run_objective(modelparameters_init, observed_massbal, precfactor_bnds=(0.33,3), tempchange_bnds=(-10,10),
                          ddfsnow_bnds=(0.0026,0.0056), precgrad_bnds=(0.0001,0.0001), run_opt=True, 
                          ftol_opt=pygem_prms.ftol_opt):
            """ Run the optimization for the single glacier objective function.
            
            Parameters
            ----------
            modelparams_init : list
                List of model parameters to calibrate
                [precipitation factor, precipitation gradient, degree day factor of snow, temperature change]
            precfactor_bnds, tempchange_bnds, ddfsnow_bnds, precgrad_bnds : tuples
                Lower and upper bounds for various model parameters
            run_opt : boolean
                Boolean statement allowing one to bypass the optimization and run through with initial parameters
                (default is True - run the optimization)
                
            Returns
            -------
            modelparams : list of model parameters
            mb_mwea : optimized modeled mass balance (mwea)
            """
            # Bounds
            modelparameters_bnds = (precfactor_bnds, precgrad_bnds, ddfsnow_bnds, tempchange_bnds)
            # Run the optimization
            #  'L-BFGS-B' - much slower
            #  'SLSQP' did not work for some geodetic measurements using the sum_abs_zscore.  One work around was to
            #    divide the sum_abs_zscore by 1000, which made it work in all cases.  However, methods were switched
            #    to 'L-BFGS-B', which may be slower, but is still effective.
            # note: switch enables running through with given parameters
            if run_opt:
                modelparameters_opt = minimize(objective, modelparameters_init, method=pygem_prms.method_opt,
                                               bounds=modelparameters_bnds, options={'ftol':ftol_opt})
                # Record the optimized parameters
                modelparameters_subset = modelparameters_opt.x
            else:
                modelparameters_subset = modelparameters_init.copy()
            modelparams = (
                    [modelparameters[0], modelparameters[1], modelparameters_subset[0], modelparameters_subset[1],
                     modelparameters_subset[2], modelparameters_subset[2] / ddfsnow_iceratio, modelparameters[6],
                     modelparameters_subset[3]])
            # Re-run the optimized parameters in order to see the mass balance
            mb_mwea = mb_mwea_calc(modelparameters, glacier_rgi_table, glacier_area_initial, icethickness_initial, 
                                       width_initial, elev_bins, glacier_gcm_temp, glacier_gcm_tempstd, 
                                       glacier_gcm_prec, glacier_gcm_elev, glacier_gcm_lrgcm, glacier_gcm_lrglac, 
                                       dates_table, t1_idx, t2_idx, option_areaconstant=1, 
                                       glacier_debrismf=glacier_debrismf)
            return modelparams, mb_mwea


        continue_param_search = True
        # ===== ROUND 1: PRECIPITATION FACTOR ======
        if debug:
            print('Round 1:')
        # Lower bound
        modelparameters[2] = precfactor_bndlow
        mb_mwea_kp_low = mb_mwea_calc(modelparameters, glacier_rgi_table, glacier_area_initial, icethickness_initial, 
                                      width_initial, elev_bins, glacier_gcm_temp, glacier_gcm_tempstd, 
                                      glacier_gcm_prec, glacier_gcm_elev, glacier_gcm_lrgcm, glacier_gcm_lrglac, 
                                      dates_table, t1_idx, t2_idx, option_areaconstant=1, 
                                      glacier_debrismf=glacier_debrismf)
        # Upper bound
        modelparameters[2] = precfactor_bndhigh
        mb_mwea_kp_high = mb_mwea_calc(modelparameters, glacier_rgi_table, glacier_area_initial, icethickness_initial, 
                                       width_initial, elev_bins, glacier_gcm_temp, glacier_gcm_tempstd, 
                                       glacier_gcm_prec, glacier_gcm_elev, glacier_gcm_lrgcm, glacier_gcm_lrglac, 
                                       dates_table, t1_idx, t2_idx, option_areaconstant=1, 
                                       glacier_debrismf=glacier_debrismf)
        # Optimimum precipitation factor
        if observed_massbal < mb_mwea_kp_low:
            precfactor_opt = precfactor_bndlow
            mb_mwea = mb_mwea_kp_low
        elif observed_massbal > mb_mwea_kp_high:
            precfactor_opt = precfactor_bndhigh
            mb_mwea = mb_mwea_kp_high
        else:
            modelparameters[2] = precfactor_init
            modelparameters_subset = [modelparameters[2], modelparameters[3], modelparameters[4], modelparameters[7]]
            precfactor_bnds = (precfactor_bndlow, precfactor_bndhigh)
            ddfsnow_bnds = (ddfsnow_init, ddfsnow_init)
            tempchange_bnds = (tempchange_init, tempchange_init)
            modelparams, mb_mwea = run_objective(modelparameters_subset, observed_massbal,
                                                 precfactor_bnds=precfactor_bnds, tempchange_bnds=tempchange_bnds,
                                                 ddfsnow_bnds=ddfsnow_bnds, ftol_opt=1e-3)
            precfactor_opt = modelparams[2]
            continue_param_search = False
        
        # Update parameter values
        modelparameters[2] = precfactor_opt
        
        if debug:
            print('  kp:', np.round(precfactor_opt,2), 'mb_mwea:', np.round(mb_mwea,2))


        # ===== ROUND 2: DEGREE-DAY FACTOR OF SNOW ======
        if continue_param_search:
            if debug:
                print('Round 2:')
            # Lower bound
            modelparameters[4] = ddfsnow_bndlow
            modelparameters[5] = modelparameters[4] / ddfsnow_iceratio
            mb_mwea_ddflow = mb_mwea_calc(modelparameters, glacier_rgi_table, glacier_area_initial, icethickness_initial, 
                                          width_initial, elev_bins, glacier_gcm_temp, glacier_gcm_tempstd, 
                                          glacier_gcm_prec, glacier_gcm_elev, glacier_gcm_lrgcm, glacier_gcm_lrglac, 
                                          dates_table, t1_idx, t2_idx, option_areaconstant=1, 
                                          glacier_debrismf=glacier_debrismf)
            # Upper bound
            modelparameters[4] = ddfsnow_bndhigh
            modelparameters[5] = modelparameters[4] / ddfsnow_iceratio
            mb_mwea_ddfhigh = mb_mwea_calc(modelparameters, glacier_rgi_table, glacier_area_initial, icethickness_initial, 
                                           width_initial, elev_bins, glacier_gcm_temp, glacier_gcm_tempstd, 
                                           glacier_gcm_prec, glacier_gcm_elev, glacier_gcm_lrgcm, glacier_gcm_lrglac, 
                                           dates_table, t1_idx, t2_idx, option_areaconstant=1, 
                                           glacier_debrismf=glacier_debrismf)
            # Optimimum degree-day factor of snow
            if observed_massbal < mb_mwea_ddfhigh:
                ddfsnow_opt = ddfsnow_bndhigh
                mb_mwea = mb_mwea_ddfhigh
            elif observed_massbal > mb_mwea_ddflow:
                ddfsnow_opt = ddfsnow_bndlow
                mb_mwea = mb_mwea_ddflow
            else:
                modelparameters_subset = [precfactor_opt, modelparameters[3], modelparameters[4], modelparameters[7]]
                precfactor_bnds = (precfactor_opt, precfactor_opt)
                ddfsnow_bnds = (ddfsnow_bndlow, ddfsnow_bndhigh)
                tempchange_bnds = (tempchange_init, tempchange_init)
                modelparams, mb_mwea = run_objective(modelparameters_subset, observed_massbal,
                                                     precfactor_bnds=precfactor_bnds, tempchange_bnds=tempchange_bnds,
                                                     ddfsnow_bnds=ddfsnow_bnds, ftol_opt=1e-5)
                ddfsnow_opt = modelparams[4]
                continue_param_search = False
            
            # Update parameter values
            modelparameters[4] = ddfsnow_opt
            modelparameters[5] = modelparameters[4] / ddfsnow_iceratio
                
            if debug:
                print('  ddfsnow:', np.round(ddfsnow_opt,4), 'mb_mwea:', np.round(mb_mwea,2))
        else:
            ddfsnow_opt = modelparams[4]
               

        # ===== ROUND 3: TEMPERATURE BIAS ======
        if continue_param_search:
            if debug:
                print('Round 3:')
            
            tc_step = 0.5
            
            # ----- TEMPBIAS: max accumulation -----
            # Lower temperature bound based on no positive temperatures
            # Temperature at the lowest bin
            #  T_bin = T_gcm + lr_gcm * (z_ref - z_gcm) + lr_glac * (z_bin - z_ref) + tempchange
            lowest_bin = np.where(glacier_area_initial > 0)[0][0]
            tempchange_max_acc = (-1 * (glacier_gcm_temp + glacier_gcm_lrgcm *
                                        (elev_bins[lowest_bin] - glacier_gcm_elev)).max())
            tempchange_bndlow = tempchange_max_acc
            
            modelparameters[7] = tempchange_bndlow
            mb_mwea = mb_mwea_calc(modelparameters, glacier_rgi_table, glacier_area_initial, icethickness_initial, 
                                          width_initial, elev_bins, glacier_gcm_temp, glacier_gcm_tempstd, 
                                          glacier_gcm_prec, glacier_gcm_elev, glacier_gcm_lrgcm, glacier_gcm_lrglac, 
                                          dates_table, t1_idx, t2_idx, option_areaconstant=1, 
                                          glacier_debrismf=glacier_debrismf)
            
            if debug:
                print('  tc_bndlow:', np.round(tempchange_bndlow,2), 'mb_mwea:', np.round(mb_mwea,2))
    
            while mb_mwea > observed_massbal and modelparameters[7] < 20:
                
                modelparameters[7] = modelparameters[7] + tc_step
                mb_mwea = mb_mwea_calc(modelparameters, glacier_rgi_table, glacier_area_initial, icethickness_initial, 
                                          width_initial, elev_bins, glacier_gcm_temp, glacier_gcm_tempstd, 
                                          glacier_gcm_prec, glacier_gcm_elev, glacier_gcm_lrgcm, glacier_gcm_lrglac, 
                                          dates_table, t1_idx, t2_idx, option_areaconstant=1, 
                                          glacier_debrismf=glacier_debrismf)
                if debug:
                    print('  tc:', np.round(modelparameters[7],2), 'mb_mwea:', np.round(mb_mwea,2))
                
                tempchange_bndhigh = modelparameters[7]
                
            modelparameters_subset = [precfactor_opt, modelparameters[3], ddfsnow_opt, modelparameters[7] - tc_step/2]
            precfactor_bnds = (precfactor_opt, precfactor_opt)
            ddfsnow_bnds = (ddfsnow_opt, ddfsnow_opt)
            tempchange_bnds = (tempchange_bndlow, tempchange_bndhigh)
            
            modelparams, mb_mwea = run_objective(modelparameters_subset, observed_massbal,
                                                 precfactor_bnds=precfactor_bnds, tempchange_bnds=tempchange_bnds,
                                                 ddfsnow_bnds=ddfsnow_bnds, ftol_opt=1e-3)

            # Update parameter values
            tc_opt = modelparams[7]
            modelparameters[7] = tc_opt
            
            if debug:
                print('  tc:', np.round(tc_opt,3), 'mb_mwea:', np.round(mb_mwea,2))
            
        else:
            tc_opt = modelparams[7]

        
        #%%
        
        
        mb_mwea = mb_mwea_calc(modelparameters, glacier_rgi_table, glacier_area_initial, icethickness_initial, 
                                      width_initial, elev_bins, glacier_gcm_temp, glacier_gcm_tempstd, 
                                      glacier_gcm_prec, glacier_gcm_elev, glacier_gcm_lrgcm, glacier_gcm_lrglac, 
                                      dates_table, t1_idx, t2_idx, option_areaconstant=1, 
                                      glacier_debrismf=glacier_debrismf)
        mb_mwea_nodebris = mb_mwea_calc(modelparameters, glacier_rgi_table, glacier_area_initial, icethickness_initial, 
                                      width_initial, elev_bins, glacier_gcm_temp, glacier_gcm_tempstd, 
                                      glacier_gcm_prec, glacier_gcm_elev, glacier_gcm_lrgcm, glacier_gcm_lrglac, 
                                      dates_table, t1_idx, t2_idx, option_areaconstant=1, 
                                      glacier_debrismf=None)
        
        if debug:
            print('mod_mwea:', np.round(mb_mwea,2), 'no debris:', np.round(mb_mwea_nodebris,2))

        if abs(mb_mwea - observed_massbal) > observed_massbal_std:
            print(rgiid, ' check as observed mass balance not close')
            # Skip and replace with observation for both to not skew results
            #  ex. 3.02860 - frontal ablation appears too much causing far too negative mass balance
            troubleshoot_fp = pygem_prms.output_fp_cal + '_4debrispaper/' + data_source + '/errors/'
            if not os.path.exists(troubleshoot_fp):
                os.makedirs(troubleshoot_fp)
            txt_fn = rgiid + "-mb_agreement_notfound.txt"
            with open(troubleshoot_fp + txt_fn, "w") as text_file:
                text_file.write(rgiid + ' mass balance not close to observation (' + str(np.round(mb_mwea,2)) 
                                + ' vs. ' + str(np.round(observed_massbal,2)) + '), so replaced with observed mb')
            mb_mwea = observed_massbal
            mb_mwea_nodebris = observed_massbal
        
        # ===== EXPORT RESULTS =====
        output_cns = ['RGIId', 'area_km2', 'kp', 'ddfsnow', 'ddfice', 'tc', 'tsnow', 'obs_mwea', 'obs_mwea_std', 
                      'mod_mwea', 'mod_mwea_nodebris']
        output_df = pd.DataFrame(np.zeros((1,len(output_cns))), columns=output_cns)
        output_df['RGIId'] = rgiid
        output_df['area_km2'] = binnedcsv.z1_bin_area_valid_km2.sum()
        print(binnedcsv.z1_bin_area_valid_km2.sum(), bins_area.sum(), main_glac_rgi.loc[nglac,'Area'])
        output_df['kp'] = precfactor_opt
        output_df['ddfsnow'] = ddfsnow_opt
        output_df['ddfice'] = ddfsnow_opt / ddfsnow_iceratio
        output_df['tc'] = tc_opt
        output_df['tsnow'] = pygem_prms.tempsnow
        output_df['obs_mwea'] = observed_massbal
        output_df['obs_mwea_std'] = observed_massbal_std
        output_df['mod_mwea'] = mb_mwea
        output_df['mod_mwea_nodebris'] = mb_mwea_nodebris
        
        # EXPORT TO NETCDF
        if mf_cn == 'mf_ts_mean_bndlow':
            mf_str = 'bndlow_'
        elif mf_cn == 'mf_ts_mean_bndhigh':
            mf_str = 'bndhigh_'
        else:
            mf_str = ''
        output_fp = (pygem_prms.output_fp_cal + '_4debrispaper/' + mf_str + data_source + '/' + 
                     str(glacier_rgi_table.O1Region).zfill(2) + '/')
        if not os.path.exists(output_fp):
            os.makedirs(output_fp)
        output_fn = rgiid + '_HH2015_wdebris.csv'
        output_df.to_csv(output_fp + output_fn, index=False)
        
        
        #%% ===== CALCULATE MASS BALANCE WITH AND WITHOUT DEBRIS =====  
#        modelparameters = [-0.0065, -0.0065, 0.8, 0.0001, 0.0045, 0.009, 1.0, 1.067748599396429]
#        print(modelparameters)
        (glac_bin_temp, glac_bin_prec, glac_bin_acc, glac_bin_refreeze, glac_bin_snowpack, glac_bin_melt,
         glac_bin_frontalablation, glac_bin_massbalclim, glac_bin_massbalclim_annual, glac_bin_area_annual,
         glac_bin_icethickness_annual, glac_bin_width_annual, glac_bin_surfacetype_annual,
         glac_wide_massbaltotal, glac_wide_runoff, glac_wide_snowline, glac_wide_snowpack,
         glac_wide_area_annual, glac_wide_volume_annual, glac_wide_ELA_annual, offglac_wide_prec,
         offglac_wide_refreeze, offglac_wide_melt, offglac_wide_snowpack, offglac_wide_runoff) = (
            massbalance.runmassbalance(modelparameters, glacier_rgi_table, glacier_area_initial, icethickness_initial,
                                       width_initial, elev_bins, glacier_gcm_temp, glacier_gcm_tempstd, glacier_gcm_prec,
                                       glacier_gcm_elev, glacier_gcm_lrgcm, glacier_gcm_lrglac, dates_table,
                                       option_areaconstant=1, glacier_debrismf=glacier_debrismf,
                                       debug=False)) 
        glac_bin_massbalclim_annual_mean_wdebris = glac_bin_massbalclim_annual.mean(axis=1)
        
        (glac_bin_temp, glac_bin_prec, glac_bin_acc, glac_bin_refreeze, glac_bin_snowpack, glac_bin_melt,
         glac_bin_frontalablation, glac_bin_massbalclim, glac_bin_massbalclim_annual, glac_bin_area_annual,
         glac_bin_icethickness_annual, glac_bin_width_annual, glac_bin_surfacetype_annual,
         glac_wide_massbaltotal, glac_wide_runoff, glac_wide_snowline, glac_wide_snowpack,
         glac_wide_area_annual, glac_wide_volume_annual, glac_wide_ELA_annual, offglac_wide_prec,
         offglac_wide_refreeze, offglac_wide_melt, offglac_wide_snowpack, offglac_wide_runoff) = (
            massbalance.runmassbalance(modelparameters, glacier_rgi_table, glacier_area_initial, icethickness_initial,
                                       width_initial, elev_bins, glacier_gcm_temp, glacier_gcm_tempstd, glacier_gcm_prec,
                                       glacier_gcm_elev, glacier_gcm_lrgcm, glacier_gcm_lrglac, dates_table,
                                       option_areaconstant=1, glacier_debrismf=None,
                                       debug=False)) 
        glac_bin_massbalclim_annual_mean_nodebris = glac_bin_massbalclim_annual.mean(axis=1)
        
        mbclim_output_cns = ['elev', 'area', 'mf', 'mbclim_mwea_wdebris', 'mbclim_mwea_nodebris', 'frontalablation']
        mbclim_output_df = pd.DataFrame(np.zeros((len(elev_bins), len(mbclim_output_cns))), columns=mbclim_output_cns)
        mbclim_output_df['elev'] = elev_bins
        mbclim_output_df['area'] = bins_area
        mbclim_output_df['mf'] = glacier_debrismf
        mbclim_output_df['mbclim_mwea_wdebris'] = glac_bin_massbalclim_annual_mean_wdebris
        mbclim_output_df['mbclim_mwea_nodebris'] = glac_bin_massbalclim_annual_mean_nodebris
        mbclim_output_df['frontalablation'] = glac_bin_frontalablation.mean(1)
        mbclim_output_fn = rgiid + '_mbclim_data.csv'
        mbclim_output_fp = (pygem_prms.output_fp_cal + '_4debrispaper/' + mf_str + data_source + '-mbclim' + '/' + 
                            str(glacier_rgi_table.O1Region).zfill(2) + '/')
        if not os.path.exists(mbclim_output_fp):
            os.makedirs(mbclim_output_fp)
        mbclim_output_df.to_csv(mbclim_output_fp + mbclim_output_fn, index=False)
        

    if debug:
        return main_glac_rgi

#    # Export variables as global to view in variable explorers
#    if args.option_parallels == 0:
#        global main_vars
#        main_vars = inspect.currentframe().f_locals

    print('\nProcessing time of', gcm_name, ':',time.time()-time_start, 's')

#%% PARALLEL PROCESSING
if __name__ == '__main__':
    parser = getparser()
    args = parser.parse_args()
    
    if args.debug == 1:
        debug = True
    else:
        debug = False
        
    if 18 in pygem_prms.rgi_regionsO1:
        prescribe_retreat = True
    else:
        prescribe_retreat = False
#    mf_cn = 'mf_ts_mean'
    mf_cn = 'mf_ts_mean_bndlow'
#    mf_cn = 'mf_ts_mean_bndhigh'

#    # Select glaciers with debris data
#    if os.getcwd().startswith('/Users/'):
#        binned_fp = '/Users/davidrounce/Documents/Dave_Rounce/DebrisGlaciers_WG/Melt_Intercomparison/output/mb_bins/csv/'
#    else:
#        binned_fp = pygem_prms.main_directory + '/../mb_bins/csv/'
#    binned_fp_whd = binned_fp + '_wdebris_hdts/'
#    binned_fp_whd_extrap = binned_fp + '_wdebris_hdts_extrap/'
#    rgiids = []
#    binned_fullfns = []
#    for roi in pygem_prms.rgi_regionsO1:
#        for i in os.listdir(binned_fp_whd):
#            if i.endswith('.csv') and int(i.split('.')[0]) == roi:
#                rgiids.append(i.split('_')[0])
#                binned_fullfns.append(binned_fp_whd + i)
#        for i in os.listdir(binned_fp_whd_extrap):
#            if i.endswith('.csv') and int(i.split('.')[0]) == roi:
#                if roi < 10:    
#                    rgiids.append(i.split('_')[0][1:])
#                else:
#                    rgiids.append(i.split('_')[0])
#                binned_fullfns.append(binned_fp_whd_extrap + i)
                
    # Select glaciers with debris data
    if os.getcwd().startswith('/Users/'):
        binned_fp = '/Users/davidrounce/Documents/Dave_Rounce/DebrisGlaciers_WG/Melt_Intercomparison/output/mb_bins_wbnds/'
    else:
#        binned_fp = pygem_prms.main_directory + '/../mb_bins/csv/'
        binned_fp = pygem_prms.main_directory + '/../mb_bins_wbnds/'
    rgiids = []
    binned_fullfns = []
    for roi in pygem_prms.rgi_regionsO1:
        for i in os.listdir(binned_fp):
            if i.endswith('.csv') and int(i.split('.')[0]) == roi:
                rgiids.append(i.split('_')[0])
                binned_fullfns.append(binned_fp + i)
            
    # Sorted files        
    binned_fullfns = [x for _,x in sorted(zip(rgiids, binned_fullfns))]
    rgiids = sorted(rgiids)

#    print('\n\nDELETE ME TO RUN ALL!\n\n')
#    rgiids = rgiids[0:1]
#    binned_fullfns = binned_fullfns[0:1]


    # Reference GCM name
    gcm_name = args.ref_gcm_name
    print('Reference climate data is:', gcm_name)

    # Select all glaciers in a region
    main_glac_rgi_all = modelsetup.selectglaciersrgitable(glac_no=rgiids)
    main_glac_rgi_all['binned_fullfn'] = binned_fullfns
    main_glac_rgi_all['CenLon_360'] = main_glac_rgi_all['CenLon']
    main_glac_rgi_all.loc[main_glac_rgi_all['CenLon_360'] < 0, 'CenLon_360'] = (
        360 + main_glac_rgi_all.loc[main_glac_rgi_all['CenLon_360'] < 0, 'CenLon_360'])
    main_glac_rgi_all['RefYear'] = (main_glac_rgi_all['RefDate'] / 1e4).astype(int)
    
#    print(np.where(main_glac_rgi_all.rgino_str.values == '08.00005'))

    # Define chunk size for parallel processing
    if args.option_parallels != 0:
        num_cores = int(np.min([main_glac_rgi_all.shape[0], args.num_simultaneous_processes]))
        chunk_size = int(np.ceil(main_glac_rgi_all.shape[0] / num_cores))
    else:
        # if not running in parallel, chunk size is all glaciers
        chunk_size = main_glac_rgi_all.shape[0]
        
#    #%% Missing glaciers
#    output_fp = '/Users/davidrounce/Documents/Dave_Rounce/HiMAT/Output/cal_opt3/_4debrispaper/'
#    rois = [str(x).zfill(2) for x in pygem_prms.rgi_regionsO1]
#    data_source='regional'
#    for nroi, roi in enumerate(rois):
#
#        output_fp_roi = output_fp + data_source + '/' + roi + '/'
#        
#        roi_fns = []
#        for i in os.listdir(output_fp_roi):
#            if i.endswith('_HH2015_wdebris.csv'):
#                roi_fns.append(i)
#        roi_fns = sorted(roi_fns)
#        rgiids_wdebris = [x.split('_')[0].split('.')[0].zfill(2) + '.' + x.split('_')[0].split('.')[1] 
#                          for x in roi_fns]
#        
#        main_glac_rgi = modelsetup.selectglaciersrgitable(glac_no=rgiids_wdebris)
#        
#        rgiids_missing = set(main_glac_rgi_all.RGIId.values) - set(main_glac_rgi.RGIId.values)
#        rgiids_missing = sorted(rgiids_missing)

    #%%===================================================
    # Pack variables for multiprocessing
    list_packed_vars = []
    for chunk in range(0, main_glac_rgi_all.shape[0], chunk_size):
        main_glac_rgi_raw = main_glac_rgi_all.loc[chunk:chunk+chunk_size-1].copy()
        main_glac_rgi_raw.reset_index(inplace=True,drop=True)
        list_packed_vars.append([main_glac_rgi_raw, gcm_name])
        
    # Parallel processing
    if args.option_parallels != 0:
        print('Processing in parallel with ' + str(num_cores) + ' cores...')
        with multiprocessing.Pool(args.num_simultaneous_processes) as p:
            p.map(main,list_packed_vars)
    # If not in parallel, then only should be one loop
    else:
        for n in range(len(list_packed_vars)):
            if debug:
                main_glac_rgi = main(list_packed_vars[n])
            else:
                main(list_packed_vars[n])