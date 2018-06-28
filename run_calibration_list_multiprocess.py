r"""
run_calibration_list_multiprocess.py runs calibration for glaciers and stores results in csv files.  The script runs 
using the reference climate data.
    
    (Command line) python run_calibration_list_multiprocess.py 
      - Default is running ERA-Interim in parallel with five processors.

    (Spyder) %run run_calibration_list_multiprocess.py -option_parallels=0
      - Spyder cannot run parallels, so always set -option_parallels=0 when testing in Spyder.
      
"""

import pandas as pd
import numpy as np
import os
import argparse
import inspect
#import subprocess as sp
import multiprocessing
from scipy.optimize import minimize
import time
import matplotlib.pyplot as plt
from time import strftime
import xarray as xr
import netCDF4 as nc

import pygem_input as input
import pygemfxns_modelsetup as modelsetup
import pygemfxns_massbalance as massbalance
import pygemfxns_output as output
import class_climate
import class_mbdata

#%% ===== SCRIPT SPECIFIC INPUT DATA ===== 
# Glacier selection
rgi_regionsO1 = [1]
rgi_glac_number = 'all'
#rgi_glac_number = ['03733']
#rgi_glac_number = ['03473', '03733']
#rgi_glac_number = ['00038', '00046', '00049', '00068', '00118', '00119', '00164', '00204', '00211', '03473', '03733']
#rgi_glac_number = ['00001', '00038', '00046', '00049', '00068', '00118', '03507', '03473', '03591', '03733', '03734']
#rgi_glac_number = ['03507']
#rgi_glac_number = ['03591']
#rgi_glac_number = ['03734']

# Required input
gcm_startyear = 2000
gcm_endyear = 2015
gcm_spinupyears = 5
option_calibration = 1

# Calibration datasets
cal_datasets = ['shean', 'wgms_ee']
#cal_datasets = ['shean']
#cal_datasets = ['wgms_ee']

# Calibration methods
method_opt = 'SLSQP'
ftol_opt = 1e-2
#method_opt = 'L-BFGS-B'
#ftol_opt = 1e-1

# Export option
option_export = 1
output_filepath = input.main_directory + '/../Output/'

#%% FUNCTIONS
def getparser():
    parser = argparse.ArgumentParser(description="run calibration in parallel")
    # add arguments
    parser.add_argument('-ref_gcm_name', action='store', type=str, default=input.ref_gcm_name, 
                        help='text file full of commands to run')
    parser.add_argument('-num_simultaneous_processes', action='store', type=int, default=5, 
                        help='number of simultaneous processes (cores) to use')
    parser.add_argument('-option_parallels', action='store', type=int, default=1,
                        help='Switch to use or not use parallels (1 - use parallels, 0 - do not)')
    return parser


def main(list_packed_vars):    
    # Unpack variables
    count = list_packed_vars[0]
    chunk = list_packed_vars[1]
    chunk_size = list_packed_vars[2]
    main_glac_rgi_all = list_packed_vars[3]
    gcm_name = list_packed_vars[4]
    
    time_start = time.time()
    parser = getparser()
    args = parser.parse_args()

    # ===== LOAD GLACIER DATA ===== 
    #  'raw' refers to the glacier subset that includes glaciers with and without calibration data
    #  after the calibration data has been imported, then all glaciers without data will be dropped
    # Glacier RGI data
    main_glac_rgi_raw = main_glac_rgi_all.iloc[chunk:chunk + chunk_size, :].copy()
    # Glacier hypsometry [km**2], total area
    main_glac_hyps_raw = modelsetup.import_Husstable(main_glac_rgi_raw, rgi_regionsO1, input.hyps_filepath, 
                                                     input.hyps_filedict, input.hyps_colsdrop)
    # Ice thickness [m], average
    main_glac_icethickness_raw = modelsetup.import_Husstable(main_glac_rgi_raw, rgi_regionsO1, input.thickness_filepath, 
                                                             input.thickness_filedict, input.thickness_colsdrop)
    main_glac_hyps_raw[main_glac_icethickness_raw == 0] = 0
    # Width [km], average
    main_glac_width_raw = modelsetup.import_Husstable(main_glac_rgi_raw, rgi_regionsO1, input.width_filepath, 
                                                      input.width_filedict, input.width_colsdrop)
    elev_bins = main_glac_hyps_raw.columns.values.astype(int)
    # Volume [km**3] and mean elevation [m a.s.l.]
    main_glac_rgi_raw['Volume'], main_glac_rgi_raw['Zmean'] = (
            modelsetup.hypsometrystats(main_glac_hyps_raw, main_glac_icethickness_raw))
    # Select dates including future projections
    #  - nospinup dates_table needed to get the proper time indices
    dates_table_nospinup, start_date, end_date = modelsetup.datesmodelrun(startyear=gcm_startyear, endyear=gcm_endyear, 
                                                                          spinupyears=0)
    dates_table, start_date, end_date = modelsetup.datesmodelrun(startyear=gcm_startyear, endyear=gcm_endyear, 
                                                                 spinupyears=gcm_spinupyears)
    
    # ===== LOAD CALIBRATION DATA =====
    cal_data = pd.DataFrame()
    for dataset in cal_datasets:
        cal_subset = class_mbdata.MBData(name=dataset, rgi_regionO1=rgi_regionsO1[0])
        cal_subset_data = cal_subset.masschange_total(main_glac_rgi_raw, main_glac_hyps_raw, dates_table_nospinup)
        cal_data = cal_data.append(cal_subset_data, ignore_index=True)
    cal_data = cal_data.sort_values(['glacno', 't1_idx'])
    cal_data.reset_index(drop=True, inplace=True)
    
    # Drop glaciers that do not have any calibration data
    main_glac_rgi = ((main_glac_rgi_raw.iloc[np.where(
            main_glac_rgi_raw[input.rgi_O1Id_colname].isin(cal_data['glacno']) == True)[0],:]).copy())
    main_glac_hyps = main_glac_hyps_raw.iloc[main_glac_rgi.index.values]
    main_glac_icethickness = main_glac_icethickness_raw.iloc[main_glac_rgi.index.values]  
    main_glac_width = main_glac_width_raw.iloc[main_glac_rgi.index.values]
    # Reset index    
    main_glac_rgi.reset_index(drop=True, inplace=True)
    main_glac_hyps.reset_index(drop=True, inplace=True)
    main_glac_icethickness.reset_index(drop=True, inplace=True)
    main_glac_width.reset_index(drop=True, inplace=True)
    
    # ===== LOAD CLIMATE DATA =====
    gcm = class_climate.GCM(name=gcm_name)
    # Air temperature [degC]
    gcm_temp, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(gcm.temp_fn, gcm.temp_vn, main_glac_rgi, dates_table)
    # Precipitation [m]
    gcm_prec, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(gcm.prec_fn, gcm.prec_vn, main_glac_rgi, dates_table)
    # Elevation [m asl]
    gcm_elev = gcm.importGCMfxnearestneighbor_xarray(gcm.elev_fn, gcm.elev_vn, main_glac_rgi)  
    # Lapse rate
    if gcm_name == 'ERA-Interim':
        gcm_lr, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(gcm.lr_fn, gcm.lr_vn, main_glac_rgi, dates_table)
    else:
        # Mean monthly lapse rate
        ref_lr_monthly_avg = np.genfromtxt(gcm.lr_fp + gcm.lr_fn, delimiter=',')
        gcm_lr = np.tile(ref_lr_monthly_avg, int(gcm_temp.shape[1]/12))
        
    # ===== CALIBRATION =====
    # Option 1: mimize mass balance difference using three-step approach to expand solution space
    if option_calibration == 1:
    
        # Model parameter output
        output_cols = ['glacno', 'obs_type', 'obs_unit', 'obs', 'model', 'uncertainty', 'zscore', 'calround']
        main_glac_modelparamsopt = np.zeros((main_glac_rgi.shape[0], 8))
        main_glac_cal_compare = pd.DataFrame(np.zeros((cal_data.shape[0],len(output_cols))), 
                                             columns=output_cols)
        main_glac_cal_compare.index = cal_data.index.values
        
        for glac in range(main_glac_rgi.shape[0]):
#            if glac%200 == 0:
#                print(count,':', main_glac_rgi.loc[main_glac_rgi.index.values[glac],'RGIId'])  
#            print(count, main_glac_rgi.loc[main_glac_rgi.index.values[glac],'RGIId'])
            
            # Set model parameters
            modelparameters = [input.lrgcm, input.lrglac, input.precfactor, input.precgrad, input.ddfsnow, input.ddfice, 
                               input.tempsnow, input.tempchange]
            # Select subsets of data
            glacier_rgi_table = main_glac_rgi.loc[main_glac_rgi.index.values[glac], :]
            glacier_gcm_elev = gcm_elev[glac]
            glacier_gcm_prec = gcm_prec[glac,:]
            glacier_gcm_temp = gcm_temp[glac,:]
            glacier_gcm_lrgcm = gcm_lr[glac,:]
            glacier_gcm_lrglac = glacier_gcm_lrgcm.copy()
            glacier_area_t0 = main_glac_hyps.iloc[glac,:].values.astype(float)   
            icethickness_t0 = main_glac_icethickness.iloc[glac,:].values.astype(float)
            width_t0 = main_glac_width.iloc[glac,:].values.astype(float)
            glacier_cal_data = ((cal_data.iloc[np.where(
                    glacier_rgi_table[input.rgi_O1Id_colname] == cal_data['glacno'])[0],:]).copy())
            
            # Modeled vs. Measured comparison dataframe
            glacier_cal_compare = pd.DataFrame(np.zeros((glacier_cal_data.shape[0], len(output_cols))), 
                                               columns=output_cols)
            glacier_cal_compare.index = glacier_cal_data.index.values
            glacier_cal_compare[['glacno', 'obs_type']] = glacier_cal_data[['glacno', 'obs_type']]

            # Record the calibration round
            calround = 0
            
            # OPTIMIZATION FUNCTION: Define the function that you are trying to minimize
            #  - modelparameters are the parameters that will be optimized
            #  - return value is the value is the value used to run the optimization
            # One way to improve objective function to include other observations (snowlines, etc.) is to normalize the
            # measured and modeled difference by the estimated error - this would mean we are minimizing the cumulative
            # absolute z-score.
            def objective(modelparameters_subset):
                # Use a subset of model parameters to reduce number of constraints required
                modelparameters[2] = modelparameters_subset[0]
                modelparameters[3] = modelparameters_subset[1]
                modelparameters[4] = modelparameters_subset[2]
                modelparameters[5] = modelparameters[4] / input.ddfsnow_iceratio
                modelparameters[7] = modelparameters_subset[3]
                # Mass balance calculations
                (glac_bin_temp, glac_bin_prec, glac_bin_acc, glac_bin_refreeze, glac_bin_snowpack, glac_bin_melt, 
                 glac_bin_frontalablation, glac_bin_massbalclim, glac_bin_massbalclim_annual, glac_bin_area_annual, 
                 glac_bin_icethickness_annual, glac_bin_width_annual, glac_bin_surfacetype_annual, 
                 glac_wide_massbaltotal, glac_wide_runoff, glac_wide_snowline, glac_wide_snowpack, 
                 glac_wide_area_annual, glac_wide_volume_annual, glac_wide_ELA_annual) = (
                    massbalance.runmassbalance(modelparameters, glacier_rgi_table, glacier_area_t0, icethickness_t0, 
                                               width_t0, elev_bins, glacier_gcm_temp, glacier_gcm_prec, 
                                               glacier_gcm_elev, glacier_gcm_lrgcm, glacier_gcm_lrglac, dates_table, 
                                               option_areaconstant=1))  
                # Loop through all measurements
                for x in range(glacier_cal_data.shape[0]):
                    cal_idx = glacier_cal_data.index.values[x]
                    # Mass balance comparisons
                    if ((glacier_cal_data.loc[cal_idx, 'obs_type'] == 'mb_geo') or 
                        (glacier_cal_data.loc[cal_idx, 'obs_type'] == 'mb_glac')):
                        # Observed mass balance [mwe]
                        glacier_cal_compare.loc[cal_idx, 'obs'] = glacier_cal_data.loc[cal_idx, 'mb_mwe']
                        glacier_cal_compare.loc[cal_idx, 'obs_unit'] = 'mwe'
                        # Modeled mass balance [mwe]
                        #  Sum(mass balance x area) / total area
                        t1_idx = glacier_cal_data.loc[cal_idx, 't1_idx'].astype(int)
                        t2_idx = glacier_cal_data.loc[cal_idx, 't2_idx'].astype(int)
                        z1_idx = glacier_cal_data.loc[cal_idx, 'z1_idx'].astype(int)
                        z2_idx = glacier_cal_data.loc[cal_idx, 'z2_idx'].astype(int)
                        year_idx = int(t1_idx / 12)
                        bin_area_subset = glac_bin_area_annual[z1_idx:z2_idx, year_idx]
                        glacier_cal_compare.loc[cal_idx, 'model'] = (
                                glac_bin_massbalclim[z1_idx:z2_idx, t1_idx:t2_idx].sum())
                        glacier_cal_compare.loc[cal_idx, 'model'] = (
                                (glac_bin_massbalclim[z1_idx:z2_idx, t1_idx:t2_idx] * 
                                 bin_area_subset[:,np.newaxis]).sum() / bin_area_subset.sum())
                        # Z-score for modeled mass balance based on observed mass balance and uncertainty
                        #  z-score = (model - measured) / uncertainty
                        glacier_cal_compare.loc[cal_idx, 'uncertainty'] = (input.massbal_uncertainty_mwea * 
                                (glacier_cal_data.loc[cal_idx, 't2'] - glacier_cal_data.loc[cal_idx, 't1']))
                        glacier_cal_compare.loc[cal_idx, 'zscore'] = (
                                (glacier_cal_compare.loc[cal_idx, 'model'] - glacier_cal_compare.loc[cal_idx, 'obs']) /
                                glacier_cal_compare.loc[cal_idx, 'uncertainty'])
                # Minimize the sum of differences
                sum_abs_zscore = abs(glacier_cal_compare['zscore']).sum()
                return sum_abs_zscore
            
            # INITIAL GUESS
            modelparameters_init = [input.precfactor, input.precgrad, input.ddfsnow, input.tempchange]
            # PARAMETER BOUNDS (Braithwaite, 2008 for DDFsnow)
            precfactor_bnds = (0.9,1.2)
            precgrad_bnds = (0.0001,0.00025)
            ddfsnow_bnds = (0.0036, 0.0046) 
            tempchange_bnds = (-1,1)
            modelparameters_bnds = (precfactor_bnds, precgrad_bnds, ddfsnow_bnds, tempchange_bnds) 
            # OPTIMIZATION ROUND #1: optimize precfactor, DDFsnow, tempchange
            # Run the optimization
            #  'L-BFGS-B' - much slower
            #  'SLSQP' did not work for some geodetic measurements using the sum_abs_zscore.  One work around was to
            #    divide the sum_abs_zscore by 1000, which made it work in all cases.  However, methods were switched
            #    to 'L-BFGS-B', which may be slower, but is still effective.
            modelparameters_opt = minimize(objective, modelparameters_init, method=method_opt, 
                                           bounds=modelparameters_bnds, options={'ftol':ftol_opt})
            # Record the calibration round
            calround = calround + 1
            # Record the optimized parameters
            modelparameters_init = modelparameters_opt.x
            main_glac_modelparamsopt[glac] = [modelparameters[0], modelparameters[1], modelparameters_init[0], 
                     modelparameters_init[1], modelparameters_init[2], modelparameters_init[2] / input.ddfsnow_iceratio,
                     modelparameters[6], modelparameters_init[3]]
            modelparameters = main_glac_modelparamsopt[glac]
            # Re-run the optimized parameters in order to see the mass balance
            # Mass balance calculations
            (glac_bin_temp, glac_bin_prec, glac_bin_acc, glac_bin_refreeze, glac_bin_snowpack, glac_bin_melt, 
             glac_bin_frontalablation, glac_bin_massbalclim, glac_bin_massbalclim_annual, glac_bin_area_annual, 
             glac_bin_icethickness_annual, glac_bin_width_annual, glac_bin_surfacetype_annual, 
             glac_wide_massbaltotal, glac_wide_runoff, glac_wide_snowline, glac_wide_snowpack, 
             glac_wide_area_annual, glac_wide_volume_annual, glac_wide_ELA_annual) = (
                massbalance.runmassbalance(modelparameters, glacier_rgi_table, glacier_area_t0, icethickness_t0, 
                                           width_t0, elev_bins, glacier_gcm_temp, glacier_gcm_prec, 
                                           glacier_gcm_elev, glacier_gcm_lrgcm, glacier_gcm_lrglac, dates_table, 
                                           option_areaconstant=1))  
            # Loop through all measurements
            for x in range(glacier_cal_data.shape[0]):
                cal_idx = glacier_cal_data.index.values[x]
                # Mass balance comparisons
                if ((glacier_cal_data.loc[cal_idx, 'obs_type'] == 'mb_geo') or 
                    (glacier_cal_data.loc[cal_idx, 'obs_type'] == 'mb_glac')):
                    # Observed mass balance [mwe]
                    glacier_cal_compare.loc[cal_idx, 'obs'] = glacier_cal_data.loc[cal_idx, 'mb_mwe']
                    glacier_cal_compare.loc[cal_idx, 'obs_unit'] = 'mwe'
                    # Modeled mass balance [mwe]
                    #  Sum(mass balance x area) / total area
                    t1_idx = glacier_cal_data.loc[cal_idx, 't1_idx'].astype(int)
                    t2_idx = glacier_cal_data.loc[cal_idx, 't2_idx'].astype(int)
                    z1_idx = glacier_cal_data.loc[cal_idx, 'z1_idx'].astype(int)
                    z2_idx = glacier_cal_data.loc[cal_idx, 'z2_idx'].astype(int)
                    year_idx = int(t1_idx / 12)
                    bin_area_subset = glac_bin_area_annual[z1_idx:z2_idx, year_idx]
                    glacier_cal_compare.loc[cal_idx, 'model'] = (
                            glac_bin_massbalclim[z1_idx:z2_idx, t1_idx:t2_idx].sum())
                    glacier_cal_compare.loc[cal_idx, 'model'] = (
                            (glac_bin_massbalclim[z1_idx:z2_idx, t1_idx:t2_idx] * 
                             bin_area_subset[:,np.newaxis]).sum() / bin_area_subset.sum())
                    # Z-score for modeled mass balance based on observed mass balance and uncertainty
                    #  z-score = (model - measured) / uncertainty
                    glacier_cal_compare.loc[cal_idx, 'uncertainty'] = (input.massbal_uncertainty_mwea * 
                            (glacier_cal_data.loc[cal_idx, 't2'] - glacier_cal_data.loc[cal_idx, 't1']))
                    glacier_cal_compare.loc[cal_idx, 'zscore'] = (
                            (glacier_cal_compare.loc[cal_idx, 'model'] - glacier_cal_compare.loc[cal_idx, 'obs']) /
                            glacier_cal_compare.loc[cal_idx, 'uncertainty'])
        
            # OPTIMIZATION ROUND #2:
            # Set zscore to compare and the tolerance
            # if only one calibration point, then zscore should be small
            if glacier_cal_compare.shape[0] == 1:
                zscore_compare = glacier_cal_compare.loc[cal_idx, 'zscore']
                zscore_tolerance = input.zscore_tolerance_single
            # else if multiple calibration points and one is a geodetic MB, check that geodetic MB is within 1
            elif (glacier_cal_compare.obs_type.isin(['mb_geo']).any() == True) and (glacier_cal_compare.shape[0] > 1):
                zscore_compare = glacier_cal_compare.loc[glacier_cal_compare.index.values[np.where(
                        glacier_cal_compare['obs_type'] == 'mb_geo')[0][0]], 'zscore']
                zscore_tolerance = input.zscore_tolerance_all
            # otherwise, check mean zscore
            else:
                zscore_compare = abs(glacier_cal_compare['zscore']).sum() / glacier_cal_compare.shape[0]
                zscore_tolerance = input.zscore_tolerance_all
            # Check if need to expand the bounds
            if abs(zscore_compare) > zscore_tolerance:
                # Bounds
                precfactor_bnds = (0.75,1.5)
                precgrad_bnds = (0.0001,0.00025)
                ddfsnow_bnds = (0.0031, 0.0051)
                tempchange_bnds = (-2,2)
                modelparameters_bnds = (precfactor_bnds, precgrad_bnds, ddfsnow_bnds, tempchange_bnds) 
                # Run optimization
                modelparameters_opt = minimize(objective, modelparameters_init, method=method_opt, 
                                               bounds=modelparameters_bnds, options={'ftol':ftol_opt})
                # Record the calibration round
                calround = calround + 1
                # Record the optimized parameters
                modelparameters_init = modelparameters_opt.x
                main_glac_modelparamsopt[glac] = [modelparameters[0], modelparameters[1], modelparameters_init[0], 
                         modelparameters_init[1], modelparameters_init[2], 
                         modelparameters_init[2] / input.ddfsnow_iceratio, modelparameters[6], modelparameters_init[3]]
                modelparameters = main_glac_modelparamsopt[glac]
                # Re-run the optimized parameters in order to see the mass balance
                # Mass balance calculations
                (glac_bin_temp, glac_bin_prec, glac_bin_acc, glac_bin_refreeze, glac_bin_snowpack, glac_bin_melt, 
                 glac_bin_frontalablation, glac_bin_massbalclim, glac_bin_massbalclim_annual, glac_bin_area_annual, 
                 glac_bin_icethickness_annual, glac_bin_width_annual, glac_bin_surfacetype_annual, 
                 glac_wide_massbaltotal, glac_wide_runoff, glac_wide_snowline, glac_wide_snowpack, 
                 glac_wide_area_annual, glac_wide_volume_annual, glac_wide_ELA_annual) = (
                    massbalance.runmassbalance(modelparameters, glacier_rgi_table, glacier_area_t0, icethickness_t0, 
                                               width_t0, elev_bins, glacier_gcm_temp, glacier_gcm_prec, 
                                               glacier_gcm_elev, glacier_gcm_lrgcm, glacier_gcm_lrglac, dates_table, 
                                               option_areaconstant=1))  
                # Loop through all measurements
                for x in range(glacier_cal_data.shape[0]):
                    cal_idx = glacier_cal_data.index.values[x]
                    # Mass balance comparisons
                    if ((glacier_cal_data.loc[cal_idx, 'obs_type'] == 'mb_geo') or 
                        (glacier_cal_data.loc[cal_idx, 'obs_type'] == 'mb_glac')):
                        # Observed mass balance [mwe]
                        glacier_cal_compare.loc[cal_idx, 'obs'] = glacier_cal_data.loc[cal_idx, 'mb_mwe']
                        glacier_cal_compare.loc[cal_idx, 'obs_unit'] = 'mwe'
                        # Modeled mass balance [mwe]
                        #  Sum(mass balance x area) / total area
                        t1_idx = glacier_cal_data.loc[cal_idx, 't1_idx'].astype(int)
                        t2_idx = glacier_cal_data.loc[cal_idx, 't2_idx'].astype(int)
                        z1_idx = glacier_cal_data.loc[cal_idx, 'z1_idx'].astype(int)
                        z2_idx = glacier_cal_data.loc[cal_idx, 'z2_idx'].astype(int)
                        year_idx = int(t1_idx / 12)
                        bin_area_subset = glac_bin_area_annual[z1_idx:z2_idx, year_idx]
                        glacier_cal_compare.loc[cal_idx, 'model'] = (
                                glac_bin_massbalclim[z1_idx:z2_idx, t1_idx:t2_idx].sum())
                        glacier_cal_compare.loc[cal_idx, 'model'] = (
                                (glac_bin_massbalclim[z1_idx:z2_idx, t1_idx:t2_idx] * 
                                 bin_area_subset[:,np.newaxis]).sum() / bin_area_subset.sum())
                        # Z-score for modeled mass balance based on observed mass balance and uncertainty
                        #  z-score = (model - measured) / uncertainty
                        glacier_cal_compare.loc[cal_idx, 'uncertainty'] = (input.massbal_uncertainty_mwea * 
                                (glacier_cal_data.loc[cal_idx, 't2'] - glacier_cal_data.loc[cal_idx, 't1']))
                        glacier_cal_compare.loc[cal_idx, 'zscore'] = (
                                (glacier_cal_compare.loc[cal_idx, 'model'] - glacier_cal_compare.loc[cal_idx, 'obs']) /
                                glacier_cal_compare.loc[cal_idx, 'uncertainty'])

            # OPTIMIZATION ROUND #3: if tolerance not reached, increase bounds again
            # Set zscore to compare and the tolerance
            # if only one calibration point, then zscore should be small
            if glacier_cal_compare.shape[0] == 1:
                zscore_compare = glacier_cal_compare.loc[cal_idx, 'zscore']
                zscore_tolerance = input.zscore_tolerance_single
            # else if multiple calibration points and one is a geodetic MB, check that geodetic MB is within 1
            elif (glacier_cal_compare.obs_type.isin(['mb_geo']).any() == True) and (glacier_cal_compare.shape[0] > 1):
                zscore_compare = glacier_cal_compare.loc[glacier_cal_compare.index.values[np.where(
                        glacier_cal_compare['obs_type'] == 'mb_geo')[0][0]], 'zscore']
                zscore_tolerance = input.zscore_tolerance_all
            # otherwise, check mean zscore
            else:
                zscore_compare = abs(glacier_cal_compare['zscore']).sum() / glacier_cal_compare.shape[0]
                zscore_tolerance = input.zscore_tolerance_all
            # Check if need to expand the bounds
            if abs(zscore_compare) > zscore_tolerance:
                # Bounds
                precfactor_bnds = (0.5,2)
                precgrad_bnds = (0.0001,0.00025)
                ddfsnow_bnds = (0.0026, 0.0056)
                tempchange_bnds = (-5,5)
                modelparameters_bnds = (precfactor_bnds, precgrad_bnds, ddfsnow_bnds, tempchange_bnds) 
                # Run optimization
                modelparameters_opt = minimize(objective, modelparameters_init, method=method_opt, 
                                               bounds=modelparameters_bnds, options={'ftol':ftol_opt})
                # Record the calibration round
                calround = calround + 1
                # Record the optimized parameters
                modelparameters_init = modelparameters_opt.x
                main_glac_modelparamsopt[glac] = [modelparameters[0], modelparameters[1], modelparameters_init[0], 
                         modelparameters_init[1], modelparameters_init[2], 
                         modelparameters_init[2] / input.ddfsnow_iceratio, modelparameters[6], modelparameters_init[3]]
                modelparameters = main_glac_modelparamsopt[glac]
                # Re-run the optimized parameters in order to see the mass balance
                (glac_bin_temp, glac_bin_prec, glac_bin_acc, glac_bin_refreeze, glac_bin_snowpack, glac_bin_melt, 
                 glac_bin_frontalablation, glac_bin_massbalclim, glac_bin_massbalclim_annual, glac_bin_area_annual, 
                 glac_bin_icethickness_annual, glac_bin_width_annual, glac_bin_surfacetype_annual, 
                 glac_wide_massbaltotal, glac_wide_runoff, glac_wide_snowline, glac_wide_snowpack, 
                 glac_wide_area_annual, glac_wide_volume_annual, glac_wide_ELA_annual) = (
                    massbalance.runmassbalance(modelparameters, glacier_rgi_table, glacier_area_t0, icethickness_t0, 
                                               width_t0, elev_bins, glacier_gcm_temp, glacier_gcm_prec, 
                                               glacier_gcm_elev, glacier_gcm_lrgcm, glacier_gcm_lrglac, dates_table, 
                                               option_areaconstant=1))  
                # Loop through all measurements
                for x in range(glacier_cal_data.shape[0]):
                    cal_idx = glacier_cal_data.index.values[x]
                    # Mass balance comparisons
                    if ((glacier_cal_data.loc[cal_idx, 'obs_type'] == 'mb_geo') or 
                        (glacier_cal_data.loc[cal_idx, 'obs_type'] == 'mb_glac')):
                        # Observed mass balance [mwe]
                        glacier_cal_compare.loc[cal_idx, 'obs'] = glacier_cal_data.loc[cal_idx, 'mb_mwe']
                        glacier_cal_compare.loc[cal_idx, 'obs_unit'] = 'mwe'
                        # Modeled mass balance [mwe]
                        #  Sum(mass balance x area) / total area
                        t1_idx = glacier_cal_data.loc[cal_idx, 't1_idx'].astype(int)
                        t2_idx = glacier_cal_data.loc[cal_idx, 't2_idx'].astype(int)
                        z1_idx = glacier_cal_data.loc[cal_idx, 'z1_idx'].astype(int)
                        z2_idx = glacier_cal_data.loc[cal_idx, 'z2_idx'].astype(int)
                        year_idx = int(t1_idx / 12)
                        bin_area_subset = glac_bin_area_annual[z1_idx:z2_idx, year_idx]
                        glacier_cal_compare.loc[cal_idx, 'model'] = (
                                glac_bin_massbalclim[z1_idx:z2_idx, t1_idx:t2_idx].sum())
                        glacier_cal_compare.loc[cal_idx, 'model'] = (
                                (glac_bin_massbalclim[z1_idx:z2_idx, t1_idx:t2_idx] * 
                                 bin_area_subset[:,np.newaxis]).sum() / bin_area_subset.sum())
                        # Z-score for modeled mass balance based on observed mass balance and uncertainty
                        #  z-score = (model - measured) / uncertainty
                        glacier_cal_compare.loc[cal_idx, 'uncertainty'] = (input.massbal_uncertainty_mwea * 
                                (glacier_cal_data.loc[cal_idx, 't2'] - glacier_cal_data.loc[cal_idx, 't1']))
                        glacier_cal_compare.loc[cal_idx, 'zscore'] = (
                                (glacier_cal_compare.loc[cal_idx, 'model'] - glacier_cal_compare.loc[cal_idx, 'obs']) /
                                glacier_cal_compare.loc[cal_idx, 'uncertainty'])
                
            # OPTIMIZATION ROUND #4: Isolate geodetic MB if necessary
            #  if there are multiple measurements and geodetic measurement still has a zscore greater than 1, then 
            #  only calibrate the geodetic measurement since this provides longest snapshot of glacier
            if (glacier_cal_compare.obs_type.isin(['mb_geo']).any() == True) and (glacier_cal_compare.shape[0] > 1):
                zscore_compare = glacier_cal_compare.loc[glacier_cal_compare.index.values[np.where(
                        glacier_cal_compare['obs_type'] == 'mb_geo')[0][0]], 'zscore']
                zscore_tolerance = input.zscore_tolerance_all
                # Important to remain within this if loop as this is a special case
                if abs(zscore_compare) > zscore_tolerance:
                    # Select only geodetic for glacier calibration data
                    glacier_cal_data = pd.DataFrame(glacier_cal_data.loc[glacier_cal_data.index.values[np.where(
                            glacier_cal_data['obs_type'] == 'mb_geo')[0][0]]]).transpose()
                    # Run optimization
                    modelparameters_opt = minimize(objective, modelparameters_init, method=method_opt, 
                                                   bounds=modelparameters_bnds, options={'ftol':ftol_opt})
                    # Record the calibration round
                    calround = calround + 1
                    # Record the optimized parameters
                    modelparameters_init = modelparameters_opt.x
                    main_glac_modelparamsopt[glac] = [modelparameters[0], modelparameters[1], modelparameters_init[0], 
                             modelparameters_init[1], modelparameters_init[2], 
                             modelparameters_init[2] / input.ddfsnow_iceratio, modelparameters[6], modelparameters_init[3]]
                    modelparameters = main_glac_modelparamsopt[glac]
                    # Re-run the optimized parameters in order to see the mass balance
                    # Mass balance calculations
                    (glac_bin_temp, glac_bin_prec, glac_bin_acc, glac_bin_refreeze, glac_bin_snowpack, glac_bin_melt, 
                     glac_bin_frontalablation, glac_bin_massbalclim, glac_bin_massbalclim_annual, glac_bin_area_annual, 
                     glac_bin_icethickness_annual, glac_bin_width_annual, glac_bin_surfacetype_annual, 
                     glac_wide_massbaltotal, glac_wide_runoff, glac_wide_snowline, glac_wide_snowpack, 
                     glac_wide_area_annual, glac_wide_volume_annual, glac_wide_ELA_annual) = (
                        massbalance.runmassbalance(modelparameters, glacier_rgi_table, glacier_area_t0, icethickness_t0, 
                                                   width_t0, elev_bins, glacier_gcm_temp, glacier_gcm_prec, 
                                                   glacier_gcm_elev, glacier_gcm_lrgcm, glacier_gcm_lrglac, dates_table, 
                                                   option_areaconstant=1))  
                    # Reset calibration data to all values for comparison
                    glacier_cal_data = ((cal_data.iloc[np.where(
                            glacier_rgi_table[input.rgi_O1Id_colname] == cal_data['glacno'])[0],:]).copy())
                    # Loop through all measurements
                    for x in range(glacier_cal_data.shape[0]):
                        cal_idx = glacier_cal_data.index.values[x]
                        # Mass balance comparisons
                        if ((glacier_cal_data.loc[cal_idx, 'obs_type'] == 'mb_geo') or 
                            (glacier_cal_data.loc[cal_idx, 'obs_type'] == 'mb_glac')):
                            # Observed mass balance [mwe]
                            glacier_cal_compare.loc[cal_idx, 'obs'] = glacier_cal_data.loc[cal_idx, 'mb_mwe']
                            glacier_cal_compare.loc[cal_idx, 'obs_unit'] = 'mwe'
                            # Modeled mass balance [mwe]
                            #  Sum(mass balance x area) / total area
                            t1_idx = glacier_cal_data.loc[cal_idx, 't1_idx'].astype(int)
                            t2_idx = glacier_cal_data.loc[cal_idx, 't2_idx'].astype(int)
                            z1_idx = glacier_cal_data.loc[cal_idx, 'z1_idx'].astype(int)
                            z2_idx = glacier_cal_data.loc[cal_idx, 'z2_idx'].astype(int)
                            year_idx = int(t1_idx / 12)
                            bin_area_subset = glac_bin_area_annual[z1_idx:z2_idx, year_idx]
                            glacier_cal_compare.loc[cal_idx, 'model'] = (
                                    glac_bin_massbalclim[z1_idx:z2_idx, t1_idx:t2_idx].sum())
                            glacier_cal_compare.loc[cal_idx, 'model'] = (
                                    (glac_bin_massbalclim[z1_idx:z2_idx, t1_idx:t2_idx] * 
                                     bin_area_subset[:,np.newaxis]).sum() / bin_area_subset.sum())
                            # Z-score for modeled mass balance based on observed mass balance and uncertainty
                            #  z-score = (model - measured) / uncertainty
                            glacier_cal_compare.loc[cal_idx, 'uncertainty'] = (input.massbal_uncertainty_mwea * 
                                    (glacier_cal_data.loc[cal_idx, 't2'] - glacier_cal_data.loc[cal_idx, 't1']))
                            glacier_cal_compare.loc[cal_idx, 'zscore'] = (
                                    (glacier_cal_compare.loc[cal_idx, 'model'] - glacier_cal_compare.loc[cal_idx, 'obs']) /
                                    glacier_cal_compare.loc[cal_idx, 'uncertainty'])
                
            # Record output
            glacier_cal_compare['calround'] = calround
            main_glac_cal_compare.loc[glacier_cal_data.index.values] = glacier_cal_compare
            
            print(count, main_glac_rgi.loc[main_glac_rgi.index.values[glac],'RGIId'])
            print('precfactor:', modelparameters[2])
            print('precgrad:', modelparameters[3])
            print('ddfsnow:', modelparameters[4])
            print('ddfice:', modelparameters[5])
            print('tempchange:', modelparameters[7])
            print('calround:', calround)
            print('modeled mass balance [mwe]:', glacier_cal_compare.loc[glacier_cal_data.index.values, 'model'].values)
            print('measured mass balance [mwe]:', glacier_cal_compare.loc[glacier_cal_data.index.values, 'obs'].values)
            print('zscore:', glacier_cal_compare.loc[glacier_cal_data.index.values, 'zscore'].values)
            print(' ')
            
        # ===== EXPORT OUTPUT =====
        # Export (i) main_glac_rgi w optimized model parameters and (ii) comparison
        main_glac_output = main_glac_rgi.copy()
        main_glac_modelparamsopt_pd = pd.DataFrame(main_glac_modelparamsopt, columns=input.modelparams_colnames)
        main_glac_modelparamsopt_pd.index = main_glac_rgi.index.values
        main_glac_output = pd.concat([main_glac_output, main_glac_modelparamsopt_pd], axis=1)
        # Export output
        if (option_calibration == 1) and (option_export == 1):
            # main_glac_rgi w model parameters
            modelparams_fn = ('cal_modelparams_opt' + str(option_calibration) + '_R' + str(rgi_regionsO1[0]) + '_' + 
                              gcm_name + '_' + str(gcm_startyear - gcm_spinupyears) + '_' + str(gcm_endyear) + '_' + 
                              str(count) + '.csv')
            main_glac_output.to_csv(input.output_filepath + modelparams_fn)
            # calibration comparison
            calcompare_fn = ('cal_compare_opt' + str(option_calibration) + '_R' + str(rgi_regionsO1[0]) + '_' + 
                              gcm_name + '_' + str(gcm_startyear - gcm_spinupyears) + '_' + str(gcm_endyear) + '_' + 
                              str(count) + '.csv')
            main_glac_cal_compare.to_csv(input.output_filepath + calcompare_fn)

    # Export variables as global to view in variable explorer
    if (args.option_parallels == 0) or (main_glac_rgi_all.shape[0] < 2 * args.num_simultaneous_processes):
        global main_vars
        main_vars = inspect.currentframe().f_locals

    print('\nProcessing time of', gcm_name, 'for', count,':',time.time()-time_start, 's')

#%% PARALLEL PROCESSING
if __name__ == '__main__':
    time_start = time.time()
    parser = getparser()
    args = parser.parse_args()
    
    # Reference GCM name
    gcm_name = args.ref_gcm_name
    print('Reference climate data is:', gcm_name)
    
    # Select all glaciers in a region
    main_glac_rgi_all = modelsetup.selectglaciersrgitable(rgi_regionsO1=rgi_regionsO1, rgi_regionsO2 = 'all', 
                                                          rgi_glac_number=rgi_glac_number)
    # Define chunk size for parallel processing
    if (args.option_parallels != 0) and (main_glac_rgi_all.shape[0] >= 2 * args.num_simultaneous_processes):
        chunk_size = int(np.ceil(main_glac_rgi_all.shape[0] / args.num_simultaneous_processes))
    else:
        # if not running in parallel, chunk size is all glaciers
        chunk_size = main_glac_rgi_all.shape[0]

    # Pack variables for parallel processing
    list_packed_vars = [] 
    n = 0
    for chunk in range(0, main_glac_rgi_all.shape[0], chunk_size):
        n = n + 1
        list_packed_vars.append([n, chunk, chunk_size, main_glac_rgi_all, gcm_name])
        
    # Parallel processing
    if (args.option_parallels != 0) and (main_glac_rgi_all.shape[0] >= 2 * args.num_simultaneous_processes):
        print('Processing in parallel...')
        with multiprocessing.Pool(args.num_simultaneous_processes) as p:
            p.map(main,list_packed_vars)      
    # If not in parallel, then only should be one loop
    else:
        for n in range(len(list_packed_vars)):
            main(list_packed_vars[n])
    
    # Combine output into single csv
    if ((args.option_parallels != 0) and (main_glac_rgi_all.shape[0] >= 2 * args.num_simultaneous_processes) and
        (option_export == 1)):
        # Model parameters
        output_prefix = ('cal_modelparams_opt' + str(option_calibration) + '_R' + str(rgi_regionsO1[0]) + '_' + 
                         gcm_name + '_' + str(gcm_startyear - gcm_spinupyears) + '_' + str(gcm_endyear) + '_')
        output_list = []
        for i in os.listdir(output_filepath):
            # Append results
            if i.startswith(output_prefix) == True:
                output_list.append(i)
                if len(output_list) == 1:
                    output_all = pd.read_csv(output_filepath + i, index_col=0)
                else:
                    output_2join = pd.read_csv(output_filepath + i, index_col=0)
                    output_all = output_all.append(output_2join, ignore_index=True)
                # Remove file after its been merged
                os.remove(output_filepath + i)
        # Export joined files
        output_all_fn = ('cal_modelparams_opt' + str(option_calibration) + '_R' + str(rgi_regionsO1[0]) + '_' + 
                         gcm_name + '_' + str(gcm_startyear - gcm_spinupyears) + '_' + str(gcm_endyear) + '_' + 
                         str(strftime("%Y%m%d")) + '.csv')
        output_all.to_csv(output_filepath + output_all_fn)
        
        # Calibration comparison
        output_prefix2 = ('cal_compare_opt' + str(option_calibration) + '_R' + str(rgi_regionsO1[0]) + '_' + 
                          gcm_name + '_' + str(gcm_startyear - gcm_spinupyears) + '_' + str(gcm_endyear) + '_')
        output_list = []
        for i in os.listdir(output_filepath):
            # Append results
            if i.startswith(output_prefix2) == True:
                output_list.append(i)
                if len(output_list) == 1:
                    output_all = pd.read_csv(output_filepath + i, index_col=0)
                else:
                    output_2join = pd.read_csv(output_filepath + i, index_col=0)
                    output_all = output_all.append(output_2join, ignore_index=True)
                # Remove file after its been merged
                os.remove(output_filepath + i)
        # Export joined files
        output_all_fn = ('cal_compare_opt' + str(option_calibration) + '_R' + str(rgi_regionsO1[0]) + '_' + 
                         gcm_name + '_' + str(gcm_startyear - gcm_spinupyears) + '_' + str(gcm_endyear) + '_' +
                         str(strftime("%Y%m%d")) + '.csv')
        output_all.to_csv(output_filepath + output_all_fn)
        
    print('Total processing time:', time.time()-time_start, 's')
            
    #%% ===== PLOTTING AND PROCESSING FOR MODEL DEVELOPMENT =====          
    # Place local variables in variable explorer
    if (args.option_parallels == 0) or (main_glac_rgi_all.shape[0] < 2 * args.num_simultaneous_processes):
        main_vars_list = list(main_vars.keys())
        gcm_name = main_vars['gcm_name']
        main_glac_rgi = main_vars['main_glac_rgi']
        main_glac_hyps = main_vars['main_glac_hyps']
        main_glac_icethickness = main_vars['main_glac_icethickness']
        main_glac_width = main_vars['main_glac_width']
        elev_bins = main_vars['elev_bins']
        dates_table = main_vars['dates_table']
        dates_table_nospinup = main_vars['dates_table_nospinup']
        cal_data = main_vars['cal_data']
        gcm_temp = main_vars['gcm_temp']
        gcm_prec = main_vars['gcm_prec']
        gcm_elev = main_vars['gcm_elev']
        glac_bin_acc = main_vars['glac_bin_acc']
        glac_bin_temp = main_vars['glac_bin_temp']
        modelparameters = main_vars['modelparameters']
        glac_bin_area_annual = main_vars['glac_bin_area_annual']
        glacier_cal_compare = main_vars['glacier_cal_compare']
        main_glac_cal_compare = main_vars['main_glac_cal_compare']
        main_glac_modelparamsopt = main_vars['main_glac_modelparamsopt']
#        glac_wide_massbaltotal = main_vars['glac_wide_massbaltotal']
#        glac_wide_area_annual = main_vars['glac_wide_area_annual']
#        glac_wide_volume_annual = main_vars['glac_wide_volume_annual']
#        glacier_rgi_table = main_vars['glacier_rgi_table']
#        main_glac_modelparamsopt = main_vars['main_glac_modelparamsopt']
#        main_glac_massbal_compare = main_vars['main_glac_massbal_compare']
#        main_glac_output = main_vars['main_glac_output']
    
#%%