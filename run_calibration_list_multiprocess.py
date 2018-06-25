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
rgi_regionsO1 = [15]
#rgi_glac_number = 'all'
#rgi_glac_number = ['03473', '03733']
#rgi_glac_number = ['00038', '00046', '00049', '00068', '00118', '00119', '00164', '00204', '00211', '03473', '03733']
rgi_glac_number = ['00001', '00038', '00046', '00049', '00068', '00118', '03507', '03473', '03591', '03733', '03734']
#rgi_glac_number = ['03507']
#rgi_glac_number = ['03591']

# Required input
gcm_startyear = 2000
gcm_endyear = 2015
gcm_spinupyears = 5
option_calibration = 1

# Calibration datasets
cal_datasets = ['shean', 'wgms_ee']
#cal_datasets = ['shean']
#cal_datasets = ['wgms_ee']

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
                        help='Switch to use or not use paralles (1 - use parallels, 0 - do not)')
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
        cal_subset = class_mbdata.MBData(name=dataset)
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
        output_cols = ['glacno', 'obs_type', 'mb_gt', 'mb_perc', 'model_mb_gt', 'model_mb_perc', 'abs_dif_perc', 
                       'calround']
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
            glacier_cal_compare = pd.DataFrame(np.zeros((glacier_cal_data.shape[0], len(output_cols))), 
                                               columns=output_cols)
            glacier_cal_compare.index = glacier_cal_data.index.values

            # Record the calibration round
            calround = 0
            
            # OPTIMIZATION FUNCTION: Define the function that you are trying to minimize
            #  - modelparameters are the parameters that will be optimized
            #  - return value is the value is the value used to run the optimization
            # One way to improve objective function to include other observations (snowlines, etc.) is to normalize the
            # measured and modeled difference by the estimated error - this would mean we are minimizing the cumulative
            # absolute z-score.
            def objective(modelparameters):
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
                # Record glacier number and observation type
                glacier_cal_compare[['glacno', 'obs_type']] = glacier_cal_data[['glacno', 'obs_type']]
                # Monthly glacier area and ice thickness for each bin
                glac_bin_area = np.tile(glac_bin_area_annual[:,0:-1], 12)
                glac_bin_icethickness = np.tile(glac_bin_icethickness_annual[:,0:-1], 12)
                # Climatic mass balance [gt] for every bin for every time step
                glac_bin_mbclim_gt = glac_bin_massbalclim / 1000 * glac_bin_area * input.density_water / 1000
                # Ice mass [gt] in each bin for every time step (note: converted from ice thickness to water equivalent)
                glac_bin_mass_gt = (glac_bin_icethickness / 1000 * input.density_ice / input.density_water * glac_bin_area * 
                                    input.density_water / 1000)
                # Loop through all measurements
                for x in range(glacier_cal_data.shape[0]):
                    cal_idx = glacier_cal_data.index.values[x]
                    # Mass balance comparisons
                    if glacier_cal_data.loc[cal_idx, 'obs_type'] == 'mb':
                        # Bin and time indices for comparison
                        t1_idx = glacier_cal_data.loc[cal_idx, 't1_idx'].astype(int)
                        t2_idx = glacier_cal_data.loc[cal_idx, 't2_idx'].astype(int)
                        z1_idx = glacier_cal_data.loc[cal_idx, 'z1_idx'].astype(int)
                        z2_idx = glacier_cal_data.loc[cal_idx, 'z2_idx'].astype(int)
                        # Glacier mass [gt] for normalized comparison
                        glac_mass = glac_bin_mass_gt[z1_idx:z2_idx+1, t1_idx].sum()
                        # Observed mass balance [gt, %]
                        glacier_cal_compare.loc[cal_idx, 'mb_gt'] = glacier_cal_data.loc[cal_idx, 'mb_gt']
                        glacier_cal_compare.loc[cal_idx, 'mb_perc'] = (
                                glacier_cal_data.loc[cal_idx, 'mb_gt'] / glac_mass * 100)
                        # Modeled mass balance [gt, %]
                        glacier_cal_compare.loc[cal_idx, 'model_mb_gt'] = (
                                glac_bin_mbclim_gt[z1_idx:z2_idx, t1_idx:t2_idx].sum())
                        glacier_cal_compare.loc[cal_idx, 'model_mb_perc'] = (
                                glacier_cal_compare.loc[cal_idx, 'model_mb_gt'] / glac_mass * 100)
                        # Difference between observed and modeled mass balance
                        glacier_cal_compare.loc[cal_idx, 'abs_dif_perc'] = (
                                abs(glacier_cal_compare.loc[cal_idx, 'mb_perc'] - 
                                    glacier_cal_compare.loc[cal_idx, 'model_mb_perc']))
                # Minimize the sum of differences
                abs_dif_perc_cum = glacier_cal_compare['abs_dif_perc'].sum()
                return abs_dif_perc_cum
            
            # CONSTRAINTS
            #  everything goes on one side of the equation compared to zero
            #  ex. return x[0] - input.lr_gcm with an equality constraint means x[0] = input.lr_gcm (see below)
            def constraint_lrgcm(modelparameters):
                return modelparameters[0] - input.lrgcm
            def constraint_lrglac(modelparameters):
                return modelparameters[1] - input.lrglac
            def constraint_precfactor(modelparameters):
                return modelparameters[2] - input.precfactor
            def constraint_precgrad(modelparameters):
                return modelparameters[3] - input.precgrad
            def constraint_ddfsnow(modelparameters):
                return modelparameters[4] - input.ddfsnow
            def constraint_ddfice(modelparameters):
                return modelparameters[5] - input.ddfice
            def constraint_tempsnow(modelparameters):
                return modelparameters[6] - input.tempsnow
            def constraint_tempchange(modelparameters):
                return modelparameters[7] - input.tempchange
            def constraint_ddficefxsnow(modelparameters):
                return modelparameters[4] - input.ddfsnow_iceratio * modelparameters[5] 
            def constraint_ddficegtsnow(modelparameters):
                return modelparameters[5] - modelparameters[4]
            def constraint_lrsequal(modelparameters):
                return modelparameters[0] - modelparameters[1]
            # Define constraint type for each function
            con_lrgcm = {'type':'eq', 'fun':constraint_lrgcm}
            con_lrglac = {'type':'eq', 'fun':constraint_lrglac}
            con_precfactor = {'type':'eq', 'fun':constraint_precfactor}
            con_precgrad = {'type':'eq', 'fun':constraint_precgrad}
            con_ddfsnow = {'type':'eq', 'fun':constraint_ddfsnow}
            con_ddfice = {'type':'eq', 'fun':constraint_ddfice}
            con_tempsnow = {'type':'eq', 'fun':constraint_tempsnow}
            con_tempchange = {'type':'eq', 'fun':constraint_tempchange}
            con_ddficefxsnow = {'type':'eq', 'fun':constraint_ddficefxsnow}
            con_ddficegtsnow = {'type':'ineq', 'fun':constraint_ddficegtsnow}
            con_lrsequal = {'type':'eq', 'fun':constraint_lrsequal}
            # INITIAL GUESS
            modelparameters_init = modelparameters
            # PARAMETER BOUNDS
            lrgcm_bnds = (-0.008,-0.004)
            lrglac_bnds = (-0.008,-0.004)
            precfactor_bnds = (0.9,1.2)
            precgrad_bnds = (0.0001,0.00025)
            ddfsnow_bnds = (0.0036, 0.0046)
            #  Braithwaite (2008)
            ddfice_bnds = (ddfsnow_bnds[0]/input.ddfsnow_iceratio, ddfsnow_bnds[1]/input.ddfsnow_iceratio)
            tempsnow_bnds = (0,2) 
            tempchange_bnds = (-1,1)
            modelparameters_bnds = (lrgcm_bnds, lrglac_bnds, precfactor_bnds, precgrad_bnds, ddfsnow_bnds, ddfice_bnds,
                                    tempsnow_bnds, tempchange_bnds)            
            # OPTIMIZATION ROUND #1: optimize precfactor, DDFsnow, tempchange
            # Select constraints used to optimize precfactor
            cons = [con_lrgcm, con_lrglac, con_ddficefxsnow, con_tempsnow]
            # Run the optimization
            #  'L-BFGS-B' - much slower
            modelparameters_opt = minimize(objective, modelparameters_init, method='SLSQP', bounds=modelparameters_bnds,
                                           constraints=cons, tol=1e-3)
            # Record the calibration round
            calround = calround + 1
            # Record the optimized parameters
            main_glac_modelparamsopt[glac] = modelparameters_opt.x
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
            # Record glacier number and observation type
            glacier_cal_compare[['glacno', 'obs_type']] = glacier_cal_data[['glacno', 'obs_type']]
            # Monthly glacier area and ice thickness for each bin
            glac_bin_area = np.tile(glac_bin_area_annual[:,0:-1], 12)
            glac_bin_icethickness = np.tile(glac_bin_icethickness_annual[:,0:-1], 12)
            # Climatic mass balance [gt] for every bin for every time step
            glac_bin_mbclim_gt = glac_bin_massbalclim / 1000 * glac_bin_area * input.density_water / 1000
            # Ice mass [gt] in each bin for every time step (note: converted from ice thickness to water equivalent)
            glac_bin_mass_gt = (glac_bin_icethickness / 1000 * input.density_ice / input.density_water * glac_bin_area * 
                                input.density_water / 1000)
            # Loop through all measurements
            for x in range(glacier_cal_data.shape[0]):
                cal_idx = glacier_cal_data.index.values[x]
                # Mass balance comparisons
                if glacier_cal_data.loc[cal_idx, 'obs_type'] == 'mb':
                    # Bin and time indices for comparison
                    t1_idx = glacier_cal_data.loc[cal_idx, 't1_idx'].astype(int)
                    t2_idx = glacier_cal_data.loc[cal_idx, 't2_idx'].astype(int)
                    z1_idx = glacier_cal_data.loc[cal_idx, 'z1_idx'].astype(int)
                    z2_idx = glacier_cal_data.loc[cal_idx, 'z2_idx'].astype(int)
                    # Glacier mass [gt] for normalized comparison
                    glac_mass = glac_bin_mass_gt[z1_idx:z2_idx+1, t1_idx].sum()
                    # Observed mass balance [gt, %]
                    glacier_cal_compare.loc[cal_idx, 'mb_gt'] = glacier_cal_data.loc[cal_idx, 'mb_gt']
                    glacier_cal_compare.loc[cal_idx, 'mb_perc'] = (
                            glacier_cal_data.loc[cal_idx, 'mb_gt'] / glac_mass * 100)
                    # Modeled mass balance [gt, %]
                    glacier_cal_compare.loc[cal_idx, 'model_mb_gt'] = (
                            glac_bin_mbclim_gt[z1_idx:z2_idx, t1_idx:t2_idx].sum())
                    glacier_cal_compare.loc[cal_idx, 'model_mb_perc'] = (
                            glacier_cal_compare.loc[cal_idx, 'model_mb_gt'] / glac_mass * 100)
                    # Difference between observed and modeled mass balance
                    glacier_cal_compare.loc[cal_idx, 'abs_dif_perc'] = (
                            abs(glacier_cal_compare.loc[cal_idx, 'mb_perc'] - 
                                glacier_cal_compare.loc[cal_idx, 'model_mb_perc']))
            # Minimize the sum of differences
            abs_dif_perc_cum = glacier_cal_compare['abs_dif_perc'].sum()
            
            # OPTIMIZATION ROUND #2: if tolerance not reached, increase bounds
            if (abs_dif_perc_cum / glacier_cal_compare.shape[0]) > input.masschange_tolerance:
                # Constraints
                cons = [con_lrgcm, con_lrglac, con_precgrad, con_ddficefxsnow, con_tempsnow]
                # Bounds
                lrgcm_bnds = (-0.008,-0.004)
                lrglac_bnds = (-0.008,-0.004)
                precfactor_bnds = (0.75,1.5)
                precgrad_bnds = (0.0001,0.00025)
                ddfsnow_bnds = (0.0031, 0.0051)
                ddfice_bnds = (ddfsnow_bnds[0]/input.ddfsnow_iceratio, ddfsnow_bnds[1]/input.ddfsnow_iceratio)
                tempsnow_bnds = (0,2) 
                tempchange_bnds = (-2,2)
                modelparameters_bnds = (lrgcm_bnds, lrglac_bnds, precfactor_bnds, precgrad_bnds, ddfsnow_bnds, 
                                        ddfice_bnds, tempsnow_bnds, tempchange_bnds)  
                # Run optimization
                modelparameters_opt = minimize(objective, main_glac_modelparamsopt[glac], method='SLSQP', 
                                               bounds=modelparameters_bnds, constraints=cons, tol=1e-3)
                # Record the calibration round
                calround = calround + 1
                # Record the optimized parameters
                main_glac_modelparamsopt[glac] = modelparameters_opt.x
                
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
                # Record glacier number and observation type
                glacier_cal_compare[['glacno', 'obs_type']] = glacier_cal_data[['glacno', 'obs_type']]
                # Monthly glacier area and ice thickness for each bin
                glac_bin_area = np.tile(glac_bin_area_annual[:,0:-1], 12)
                glac_bin_icethickness = np.tile(glac_bin_icethickness_annual[:,0:-1], 12)
                # Climatic mass balance [gt] for every bin for every time step
                glac_bin_mbclim_gt = glac_bin_massbalclim / 1000 * glac_bin_area * input.density_water / 1000
                # Ice mass [gt] in each bin for every time step (note: converted from ice thickness to water equivalent)
                glac_bin_mass_gt = (glac_bin_icethickness / 1000 * input.density_ice / input.density_water * glac_bin_area * 
                                    input.density_water / 1000)
                # Loop through all measurements
                for x in range(glacier_cal_data.shape[0]):
                    cal_idx = glacier_cal_data.index.values[x]
                    # Mass balance comparisons
                    if glacier_cal_data.loc[cal_idx, 'obs_type'] == 'mb':
                        # Bin and time indices for comparison
                        t1_idx = glacier_cal_data.loc[cal_idx, 't1_idx'].astype(int)
                        t2_idx = glacier_cal_data.loc[cal_idx, 't2_idx'].astype(int)
                        z1_idx = glacier_cal_data.loc[cal_idx, 'z1_idx'].astype(int)
                        z2_idx = glacier_cal_data.loc[cal_idx, 'z2_idx'].astype(int)
                        # Glacier mass [gt] for normalized comparison
                        glac_mass = glac_bin_mass_gt[z1_idx:z2_idx+1, t1_idx].sum()
                        # Observed mass balance [gt, %]
                        glacier_cal_compare.loc[cal_idx, 'mb_gt'] = glacier_cal_data.loc[cal_idx, 'mb_gt']
                        glacier_cal_compare.loc[cal_idx, 'mb_perc'] = (
                                glacier_cal_data.loc[cal_idx, 'mb_gt'] / glac_mass * 100)
                        # Modeled mass balance [gt, %]
                        glacier_cal_compare.loc[cal_idx, 'model_mb_gt'] = (
                                glac_bin_mbclim_gt[z1_idx:z2_idx, t1_idx:t2_idx].sum())
                        glacier_cal_compare.loc[cal_idx, 'model_mb_perc'] = (
                                glacier_cal_compare.loc[cal_idx, 'model_mb_gt'] / glac_mass * 100)
                        # Difference between observed and modeled mass balance
                        glacier_cal_compare.loc[cal_idx, 'abs_dif_perc'] = (
                                abs(glacier_cal_compare.loc[cal_idx, 'mb_perc'] - 
                                    glacier_cal_compare.loc[cal_idx, 'model_mb_perc']))
                # Minimize the sum of differences
                abs_dif_perc_cum = glacier_cal_compare['abs_dif_perc'].sum()
            
            # OPTIMIZATION ROUND #3: if tolerance not reached, increase bounds again
            if (abs_dif_perc_cum / glacier_cal_compare.shape[0]) > input.masschange_tolerance:
                # Constraints
                cons = [con_lrgcm, con_lrglac, con_precgrad, con_ddficefxsnow, con_tempsnow]
                # Bounds
                lrgcm_bnds = (-0.008,-0.004)
                lrglac_bnds = (-0.008,-0.004)
                precfactor_bnds = (0.5,2)
                precgrad_bnds = (0.0001,0.00025)
                ddfsnow_bnds = (0.0026, 0.0056)
                ddfice_bnds = (ddfsnow_bnds[0]/input.ddfsnow_iceratio, ddfsnow_bnds[1]/input.ddfsnow_iceratio)
                tempsnow_bnds = (0,2) 
                tempchange_bnds = (-5,5)
                modelparameters_bnds = (lrgcm_bnds, lrglac_bnds, precfactor_bnds, precgrad_bnds, ddfsnow_bnds, 
                                        ddfice_bnds, tempsnow_bnds, tempchange_bnds)  
                # Run optimization
                modelparameters_opt = minimize(objective, main_glac_modelparamsopt[glac], method='SLSQP', 
                                               bounds=modelparameters_bnds, constraints=cons, tol=1e-3)
                # Record the calibration round
                calround = calround + 1
                # Record the optimized parameters
                main_glac_modelparamsopt[glac] = modelparameters_opt.x
                
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
                # Record glacier number and observation type
                glacier_cal_compare[['glacno', 'obs_type']] = glacier_cal_data[['glacno', 'obs_type']]
                # Monthly glacier area and ice thickness for each bin
                glac_bin_area = np.tile(glac_bin_area_annual[:,0:-1], 12)
                glac_bin_icethickness = np.tile(glac_bin_icethickness_annual[:,0:-1], 12)
                # Climatic mass balance [gt] for every bin for every time step
                glac_bin_mbclim_gt = glac_bin_massbalclim / 1000 * glac_bin_area * input.density_water / 1000
                # Ice mass [gt] in each bin for every time step (note: converted from ice thickness to water equivalent)
                glac_bin_mass_gt = (glac_bin_icethickness / 1000 * input.density_ice / input.density_water * glac_bin_area * 
                                    input.density_water / 1000)
                # Loop through all measurements
                for x in range(glacier_cal_data.shape[0]):
                    cal_idx = glacier_cal_data.index.values[x]
                    # Mass balance comparisons
                    if glacier_cal_data.loc[cal_idx, 'obs_type'] == 'mb':
                        # Bin and time indices for comparison
                        t1_idx = glacier_cal_data.loc[cal_idx, 't1_idx'].astype(int)
                        t2_idx = glacier_cal_data.loc[cal_idx, 't2_idx'].astype(int)
                        z1_idx = glacier_cal_data.loc[cal_idx, 'z1_idx'].astype(int)
                        z2_idx = glacier_cal_data.loc[cal_idx, 'z2_idx'].astype(int)
                        # Glacier mass [gt] for normalized comparison
                        glac_mass = glac_bin_mass_gt[z1_idx:z2_idx+1, t1_idx].sum()
                        # Observed mass balance [gt, %]
                        glacier_cal_compare.loc[cal_idx, 'mb_gt'] = glacier_cal_data.loc[cal_idx, 'mb_gt']
                        glacier_cal_compare.loc[cal_idx, 'mb_perc'] = (
                                glacier_cal_data.loc[cal_idx, 'mb_gt'] / glac_mass * 100)
                        # Modeled mass balance [gt, %]
                        glacier_cal_compare.loc[cal_idx, 'model_mb_gt'] = (
                                glac_bin_mbclim_gt[z1_idx:z2_idx, t1_idx:t2_idx].sum())
                        glacier_cal_compare.loc[cal_idx, 'model_mb_perc'] = (
                                glacier_cal_compare.loc[cal_idx, 'model_mb_gt'] / glac_mass * 100)
                        # Difference between observed and modeled mass balance
                        glacier_cal_compare.loc[cal_idx, 'abs_dif_perc'] = (
                                abs(glacier_cal_compare.loc[cal_idx, 'mb_perc'] - 
                                    glacier_cal_compare.loc[cal_idx, 'model_mb_perc']))
                # Minimize the sum of differences
                abs_dif_perc_cum = glacier_cal_compare['abs_dif_perc'].sum()
                
            # Record output
            glacier_cal_compare['calround'] = calround
            main_glac_cal_compare.loc[glacier_cal_data.index.values] = glacier_cal_compare
            
            output_cols = ['glacno', 'obs_type', 'mb_gt', 'mb_perc', 'model_mb_gt', 'model_mb_perc', 'abs_dif_perc', 
                           'calround']
            
            print(count, main_glac_rgi.loc[main_glac_rgi.index.values[glac],'RGIId'])
            print('precfactor:', modelparameters[2])
            print('precgrad:', modelparameters[3])
            print('ddfsnow:', modelparameters[4])
            print('tempchange:', modelparameters[7])
            print('calround:', calround)
            print('modeled mass change [%]:', glacier_cal_compare.loc[glacier_cal_data.index.values, 'mb_perc'].values)
            print('measured mass change [%]:', glacier_cal_compare.loc[glacier_cal_data.index.values, 'model_mb_perc'].values)
            print('measured mass change error [%]:', glacier_cal_compare.loc[glacier_cal_data.index.values, 'abs_dif_perc'].values)
            print(' ')
            
#        # ===== EXPORT OUTPUT =====
#        main_glac_output = main_glac_rgi.copy()
#        main_glac_modelparamsopt_pd = pd.DataFrame(main_glac_modelparamsopt, columns=input.modelparams_colnames)
#        main_glac_modelparamsopt_pd.index = main_glac_rgi.index.values
#        main_glac_output = pd.concat([main_glac_output, main_glac_massbal_compare, main_glac_modelparamsopt_pd], axis=1)
#        # Export output
#        if (option_calibration == 1) and (option_export == 1):
#            output_fn = ('cal_opt' + str(option_calibration) + '_R' + str(rgi_regionsO1[0]) + '_' + gcm_name + '_' + 
#                         str(gcm_startyear - gcm_spinupyears) + '_' + str(gcm_endyear) + '_' + str(count) + '.csv')
#            main_glac_output.to_csv(input.output_filepath + output_fn)

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
    
#    # Combine output into single csv
#    if ((args.option_parallels != 0) and (main_glac_rgi_all.shape[0] >= 2 * args.num_simultaneous_processes) and
#        (option_export == 1)):
#        # Single output file
#        output_prefix = ('cal_opt' + str(option_calibration) + '_R' + str(rgi_regionsO1[0]) + '_' + gcm_name + '_' + 
#                         str(gcm_startyear - gcm_spinupyears) + '_' + str(gcm_endyear) + '_')
#        output_list = []
#        for i in os.listdir(output_filepath):
#            # Append results
#            if i.startswith(output_prefix) == True:
#                output_list.append(i)
#                if len(output_list) == 1:
#                    output_all = pd.read_csv(output_filepath + i, index_col=0)
#                else:
#                    output_2join = pd.read_csv(output_filepath + i, index_col=0)
#                    output_all = output_all.append(output_2join, ignore_index=True)
#                # Remove file after its been merged
#                os.remove(output_filepath + i)
#        # Export joined files
#        output_all_fn = ('cal_opt' + str(option_calibration) + '_R' + str(rgi_regionsO1[0]) + '_' + gcm_name + '_' + 
#                         str(gcm_startyear - gcm_spinupyears) + '_' + str(gcm_endyear) + '_' + 
#                         str(strftime("%Y%m%d")) + '.csv')
#        output_all.to_csv(output_filepath + output_all_fn)
        
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
        cal_data = main_vars['cal_data']
        gcm_temp = main_vars['gcm_temp']
        gcm_prec = main_vars['gcm_prec']
        gcm_elev = main_vars['gcm_elev']
        modelparameters = main_vars['modelparameters']
        glacier_cal_compare = main_vars['glacier_cal_compare']
        main_glac_cal_compare = main_vars['main_glac_cal_compare']
#        glac_wide_massbaltotal = main_vars['glac_wide_massbaltotal']
#        glac_wide_area_annual = main_vars['glac_wide_area_annual']
#        glac_wide_volume_annual = main_vars['glac_wide_volume_annual']
#        glacier_rgi_table = main_vars['glacier_rgi_table']
#        main_glac_modelparamsopt = main_vars['main_glac_modelparamsopt']
#        main_glac_massbal_compare = main_vars['main_glac_massbal_compare']
#        main_glac_output = main_vars['main_glac_output']
    
#%%