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
import climate_class
import mbdata_class

#%% ===== SCRIPT SPECIFIC INPUT DATA ===== 
# Glacier selection
rgi_regionsO1 = [15]
#rgi_glac_number = 'all'
rgi_glac_number = ['03473', '03733']

# Required input
# Time period
gcm_startyear = 2000
gcm_endyear = 2015
gcm_spinupyears = 5

option_calibration = 1

## Output
#output_package = 2
#output_filepath = input.main_directory + '/../Output/'

#%% FUNCTIONS
def getparser():
    parser = argparse.ArgumentParser(description="run calibration in parallel")
    # add arguments
    parser.add_argument('-gcm_file', action='store', type=str, default=input.ref_gcm_name, 
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
    main_glac_rgi_all = list_packed_vars[2]
    chunk_size = list_packed_vars[3]
    gcm_name = list_packed_vars[4]
    
    time_start = time.time()
    parser = getparser()
    args = parser.parse_args()

    # ===== LOAD GLACIER DATA ===== 
    main_glac_rgi = main_glac_rgi_all.iloc[chunk:chunk + chunk_size, :].copy()
    
    # Load calibration data and add to main_glac_rgi
    mb1 = mbdata_class.MBData(name='shean')
    mb1.mbdata = mb1.masschange_total(main_glac_rgi, gcm_startyear, gcm_endyear)
    # Merge datasets
    main_glac_rgi = pd.concat([main_glac_rgi, mb1.mbdata], axis=1)
    mb1.mbdata = mb1.mbdata.dropna()
    main_glac_rgi = main_glac_rgi.dropna()
    
    # Glacier hypsometry [km**2], total area
    main_glac_hyps = modelsetup.import_Husstable(main_glac_rgi, rgi_regionsO1, input.hyps_filepath, 
                                                 input.hyps_filedict, input.hyps_colsdrop)
    # Ice thickness [m], average
    main_glac_icethickness = modelsetup.import_Husstable(main_glac_rgi, rgi_regionsO1, input.thickness_filepath, 
                                                         input.thickness_filedict, input.thickness_colsdrop)
    main_glac_hyps[main_glac_icethickness == 0] = 0
    # Width [km], average
    main_glac_width = modelsetup.import_Husstable(main_glac_rgi, rgi_regionsO1, input.width_filepath, 
                                                  input.width_filedict, input.width_colsdrop)
    elev_bins = main_glac_hyps.columns.values.astype(int)
    # Volume [km**3] and mean elevation [m a.s.l.]
    main_glac_rgi['Volume'], main_glac_rgi['Zmean'] = modelsetup.hypsometrystats(main_glac_hyps, main_glac_icethickness)
    # Select dates including future projections
    dates_table, start_date, end_date = modelsetup.datesmodelrun(startyear=gcm_startyear, endyear=gcm_endyear, 
                                                                 spinupyears=gcm_spinupyears)
    # ===== LOAD CLIMATE DATA =====
    gcm = climate_class.GCM(name=gcm_name)
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
        output_cols = ['mb_gt', 'mb_perc', 'model_mb_gt', 'model_mb_perc']
        main_glac_modelparamsopt = np.zeros((main_glac_rgi.shape[0], 8))
        main_glac_massbal_compare = pd.DataFrame(np.zeros((main_glac_rgi.shape[0],len(output_cols))), 
                                                 columns=output_cols)
        
        for glac in range(main_glac_rgi.shape[0]):
    #        if glac%200 == 0:
    #            print(gcm_name,':', main_glac_rgi.loc[main_glac_rgi.index.values[glac],'RGIId'])  
            print(main_glac_rgi.loc[glac,'RGIId'])
            
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
            
            # Record the calibration round
            calround = 0
            
            # OPTIMIZATION FUNCTION: Define the function that you are trying to minimize
            #  - modelparameters are the parameters that will be optimized
            #  - return value is the value is the value used to run the optimization
            def objective(modelparameters):
                # Mass balance calcs
                (glac_bin_temp, glac_bin_prec, glac_bin_acc, glac_bin_refreeze, glac_bin_snowpack, glac_bin_melt, 
                 glac_bin_frontalablation, glac_bin_massbalclim, glac_bin_massbalclim_annual, glac_bin_area_annual, 
                 glac_bin_icethickness_annual, glac_bin_width_annual, glac_bin_surfacetype_annual, 
                 glac_wide_massbaltotal, glac_wide_runoff, glac_wide_snowline, glac_wide_snowpack, 
                 glac_wide_area_annual, glac_wide_volume_annual, glac_wide_ELA_annual) = (
                    massbalance.runmassbalance(modelparameters, glacier_rgi_table, glacier_area_t0, icethickness_t0, 
                                               width_t0, elev_bins, glacier_gcm_temp, glacier_gcm_prec, 
                                               glacier_gcm_elev, glacier_gcm_lrgcm, glacier_gcm_lrglac, dates_table, 
                                               option_areaconstant=1))
                # Time indices associated with mass balance data
                t1_idx = mb1.mbdata.loc[glac,'t1_idx'].astype(int)
                t2_idx = mb1.mbdata.loc[glac,'t2_idx'].astype(int)
                # Total mass change [Gt, %] consistent with geodetic measurement
                glac_wide_area = np.repeat(glac_wide_area_annual[0:-1],16)
                masschange_total =  ((glac_wide_massbaltotal[t1_idx:t2_idx] * glac_wide_area[t1_idx:t2_idx]).sum() 
                                     / 1000 * input.density_water / 1000) 
                masschange_total_perc = (masschange_total / (glac_wide_volume_annual[0] * input.density_water / 1000) 
                                         * 100)
                # Geodetic total mass change [%]
                mb1_gt_perc = (mb1.mbdata.loc[glac,'mb_gt'] / (glac_wide_volume_annual[0] * input.density_water / 1000) 
                               * 100)
                # Difference between geodetic and modeled mass balance
                masschange_difference = abs(mb1_gt_perc - masschange_total_perc)
                return masschange_difference
            
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
            # Time indices associated with mass balance data
            t1_idx = mb1.mbdata.loc[glac,'t1_idx'].astype(int)
            t2_idx = mb1.mbdata.loc[glac,'t2_idx'].astype(int)
            # Total mass change [Gt, %] consistent with geodetic measurement
            glac_wide_area = np.repeat(glac_wide_area_annual[0:-1],16)
            masschange_total =  ((glac_wide_massbaltotal[t1_idx:t2_idx] * glac_wide_area[t1_idx:t2_idx]).sum() 
                                 / 1000 * input.density_water / 1000) 
            masschange_total_perc = (masschange_total / (glac_wide_volume_annual[0] * input.density_water / 1000) 
                                     * 100)
            # Geodetic total mass change [%]
            mb1_gt_perc = (mb1.mbdata.loc[glac,'mb_gt'] / (glac_wide_volume_annual[0] * input.density_water / 1000) 
                           * 100)
            mb1_gt_err_perc = (mb1.mbdata.loc[glac,'mb_gt_err'] / (glac_wide_volume_annual[0] * input.density_water 
                               / 1000) * 100)
            # Difference between geodetic and modeled mass balance
            masschange_difference = abs(mb1_gt_perc - masschange_total_perc)
            
            # OPTIMIZATION ROUND #2: if tolerance not reached, increase bounds
            if masschange_difference > input.masschange_tolerance:
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
                # Time indices associated with mass balance data
                t1_idx = mb1.mbdata.loc[glac,'t1_idx'].astype(int)
                t2_idx = mb1.mbdata.loc[glac,'t2_idx'].astype(int)
                # Total mass change [Gt, %] consistent with geodetic measurement
                glac_wide_area = np.repeat(glac_wide_area_annual[0:-1],16)
                masschange_total =  ((glac_wide_massbaltotal[t1_idx:t2_idx] * glac_wide_area[t1_idx:t2_idx]).sum() 
                                     / 1000 * input.density_water / 1000) 
                masschange_total_perc = (masschange_total / (glac_wide_volume_annual[0] * input.density_water / 1000) 
                                         * 100)
                # Geodetic total mass change [%]
                mb1_gt_perc = (mb1.mbdata.loc[glac,'mb_gt'] / (glac_wide_volume_annual[0] * input.density_water / 1000) 
                               * 100)
                mb1_gt_err_perc = (mb1.mbdata.loc[glac,'mb_gt_err'] / (glac_wide_volume_annual[0] * input.density_water 
                                   / 1000) * 100)
                # Difference between geodetic and modeled mass balance
                masschange_difference = abs(mb1_gt_perc - masschange_total_perc)
                
            # OPTIMIZATION ROUND #3: if tolerance not reached, increase bounds again
            if masschange_difference > input.masschange_tolerance:
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
                # Time indices associated with mass balance data
                t1_idx = mb1.mbdata.loc[glac,'t1_idx'].astype(int)
                t2_idx = mb1.mbdata.loc[glac,'t2_idx'].astype(int)
                # Total mass change [Gt, %] consistent with geodetic measurement
                glac_wide_area = np.repeat(glac_wide_area_annual[0:-1],16)
                masschange_total =  ((glac_wide_massbaltotal[t1_idx:t2_idx] * glac_wide_area[t1_idx:t2_idx]).sum() 
                                     / 1000 * input.density_water / 1000) 
                masschange_total_perc = (masschange_total / (glac_wide_volume_annual[0] * input.density_water / 1000) 
                                         * 100)
                # Geodetic total mass change [%]
                mb1_gt_perc = (mb1.mbdata.loc[glac,'mb_gt'] / (glac_wide_volume_annual[0] * input.density_water / 1000) 
                               * 100)
                mb1_gt_err_perc = (mb1.mbdata.loc[glac,'mb_gt_err'] / (glac_wide_volume_annual[0] * input.density_water 
                                   / 1000) * 100)
                # Difference between geodetic and modeled mass balance
                masschange_difference = abs(mb1_gt_perc - masschange_total_perc)
                
                
            main_glac_massbal_compare.iloc[glac,:] = [mb1.mbdata.loc[glac, 'mb_gt'], mb1_gt_perc, masschange_total, 
                                                      masschange_total_perc]

            print('precfactor:', modelparameters[2])
            print('precgrad:', modelparameters[3])
            print('ddfsnow:', modelparameters[4])
            print('tempchange:', modelparameters[7])
            print(masschange_total_perc, mb1_gt_perc, mb1_gt_err_perc)

#        if output_package != 0:
#            # Output calibration results to .csv file
#            #  pandas dataframe used instead of numpy arrays here, so column headings can be exported
#            main_glac_caloutput = main_glac_rgi.copy()
#            main_glac_caloutput['MB_model_mwea'] = main_glac_massbal_compare[:,0] 
#            main_glac_caloutput['MB_geodetic_mwea'] = main_glac_massbal_compare[:,1] 
#            main_glac_caloutput['MB_difference_mwea'] = main_glac_massbal_compare[:,2]
#            main_glac_caloutput['calround'] = main_glac_massbal_compare[:,3]
#            main_glac_caloutput['lrgcm'] = main_glac_modelparamsopt[:,0] 
#            main_glac_caloutput['lrglac'] = main_glac_modelparamsopt[:,1] 
#            main_glac_caloutput['precfactor'] = main_glac_modelparamsopt[:,2] 
#            main_glac_caloutput['precgrad'] = main_glac_modelparamsopt[:,3] 
#            main_glac_caloutput['ddfsnow'] = main_glac_modelparamsopt[:,4] 
#            main_glac_caloutput['ddfice'] = main_glac_modelparamsopt[:,5] 
#            main_glac_caloutput['tempsnow'] = main_glac_modelparamsopt[:,6]
#            main_glac_caloutput['tempchange'] = main_glac_modelparamsopt[:,7]
#            # export csv
#            cal_output_fullfile = (input.output_filepath + input.calibrationcsv_filenameprefix + 'R' + str(regionO1_number) 
#                                   + '_' + str(strftime("%Y%m%d")) + '.csv')
#            main_glac_caloutput.to_csv(cal_output_fullfile)

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
    
    # Select glaciers and define chunks
    main_glac_rgi_all = modelsetup.selectglaciersrgitable(rgi_regionsO1=rgi_regionsO1, rgi_regionsO2 = 'all', 
                                                          rgi_glac_number=rgi_glac_number)
    # Processing needed for netcdf files
    main_glac_rgi_all['RGIId_float'] = (np.array([np.str.split(main_glac_rgi_all['RGIId'][x],'-')[1] 
                                              for x in range(main_glac_rgi_all.shape[0])]).astype(float))
    main_glac_rgi_all_float = main_glac_rgi_all.copy()
    main_glac_rgi_all_float.drop(labels=['RGIId'], axis=1, inplace=True)
    main_glac_hyps = modelsetup.import_Husstable(main_glac_rgi_all, rgi_regionsO1, input.hyps_filepath, 
                                                 input.hyps_filedict, input.hyps_colsdrop)
    dates_table, start_date, end_date = modelsetup.datesmodelrun(startyear=gcm_startyear, endyear=gcm_endyear, 
                                                                 spinupyears=gcm_spinupyears)
    
    if (args.option_parallels != 0) and (main_glac_rgi_all.shape[0] >= 2 * args.num_simultaneous_processes):
        chunk_size = int(np.ceil(main_glac_rgi_all.shape[0] / args.num_simultaneous_processes))
    else:
        chunk_size = main_glac_rgi_all.shape[0]
    
    # Read GCM names from command file
    if args.gcm_file == input.ref_gcm_name:
        gcm_list = [input.ref_gcm_name]
    else:
        with open(args.gcm_file, 'r') as gcm_fn:
            gcm_list = gcm_fn.read().splitlines()
            rcp_scenario = os.path.basename(args.gcm_file).split('_')[1]
            print('Found %d gcms to process'%(len(gcm_list)))
        
    # Loop through all GCMs
    for gcm_name in gcm_list:
        print('Processing:', gcm_name)
        # Pack variables for multiprocessing
        list_packed_vars = [] 
        n = 0
        for chunk in range(0, main_glac_rgi_all.shape[0], chunk_size):
            n = n + 1
            list_packed_vars.append([n, chunk, main_glac_rgi_all, chunk_size, gcm_name])
        
        # Parallel processing
        if (args.option_parallels != 0) and (main_glac_rgi_all.shape[0] >= 2 * args.num_simultaneous_processes):
            with multiprocessing.Pool(args.num_simultaneous_processes) as p:
                p.map(main,list_packed_vars)
        
        # No parallel processing
        else:
            # Loop through the chunks and export bias adjustments
            for n in range(len(list_packed_vars)):
                main(list_packed_vars[n])
    
#         # Combine output into single package and export lapse rate if necessary
#        if (args.option_parallels != 0) and (main_glac_rgi_all.shape[0] >= 2 * args.num_simultaneous_processes):
#            # Netcdf outputs
#            output_prefix = ('PyGEM_R' + str(rgi_regionsO1[0]) + '_' + gcm_name + '_' + rcp_scenario + '_biasadj_opt' + 
#                             str(option_bias_adjustment) + '_' + str(gcm_startyear - gcm_spinupyears) + '_' + 
#                             str(gcm_endyear) + '_' + 'test')
#            output_all_fn = ('PyGEM_R' + str(rgi_regionsO1[0]) + '_' + gcm_name + '_' + rcp_scenario + '_biasadj_opt' + 
#                             str(option_bias_adjustment) + '_' + str(gcm_startyear - gcm_spinupyears) + '_' + 
#                             str(gcm_endyear) + '_all.nc')
#            
#            # Select netcdf files produced in parallel
#            output_list = []
#            for i in os.listdir(output_filepath):
#                # Append bias adjustment results
#                if i.startswith(output_prefix) == True:
#                    output_list.append(i)
#            
#            # Merge netcdfs together
#            if (len(output_list) > 1) and (output_package != 0):
#                # Create netcdf that will have them all together
#                output.netcdfcreate(output_all_fn, main_glac_rgi_all_float, main_glac_hyps, dates_table, 
#                                    output_filepath=input.output_filepath)
#                # Open file to write
#                netcdf_output = nc.Dataset(output_filepath + output_all_fn, 'r+')
#                
#                glac_count = -1
#                for n in range(len(output_list)):
#                    ds = nc.Dataset(output_filepath + output_list[n])
#                    for glac in range(ds['glac_idx'][:].shape[0]):
#                        glac_count = glac_count + 1
#                        if output_package == 2:
#                            netcdf_output.variables['temp_glac_monthly'][glac_count,:] = (
#                                    ds['temp_glac_monthly'][glac,:])
#                            netcdf_output.variables['prec_glac_monthly'][glac_count,:] = (
#                                    ds['prec_glac_monthly'][glac,:])
#                            netcdf_output.variables['acc_glac_monthly'][glac_count,:] = (
#                                    ds['acc_glac_monthly'][glac,:])
#                            netcdf_output.variables['refreeze_glac_monthly'][glac_count,:] = (
#                                    ds['refreeze_glac_monthly'][glac,:])
#                            netcdf_output.variables['melt_glac_monthly'][glac_count,:] = (
#                                    ds['melt_glac_monthly'][glac,:])
#                            netcdf_output.variables['frontalablation_glac_monthly'][glac_count,:] = (
#                                    ds['frontalablation_glac_monthly'][glac,:])
#                            netcdf_output.variables['massbaltotal_glac_monthly'][glac_count,:] = (
#                                    ds['massbaltotal_glac_monthly'][glac,:])
#                            netcdf_output.variables['runoff_glac_monthly'][glac_count,:] = (
#                                    ds['runoff_glac_monthly'][glac,:])
#                            netcdf_output.variables['snowline_glac_monthly'][glac_count,:] = (
#                                    ds['snowline_glac_monthly'][glac,:])
#                            netcdf_output.variables['area_glac_annual'][glac_count,:] = (
#                                    ds['area_glac_annual'][glac,:])
#                            netcdf_output.variables['volume_glac_annual'][glac_count,:] = (
#                                    ds['volume_glac_annual'][glac,:])
#                            netcdf_output.variables['ELA_glac_annual'][glac_count,:] = (
#                                    ds['ELA_glac_annual'][glac,:])
#                        else:
#                            print('Code merge for output package')  
#                    ds.close()
#                    # Remove file after its been merged
#                    os.remove(output_filepath + output_list[n])
#                # Close the netcdf file
#                netcdf_output.close()
        
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
#        main_glac_modelparams = main_vars['main_glac_modelparams']
        elev_bins = main_vars['elev_bins']
        dates_table = main_vars['dates_table']
        gcm_temp = main_vars['gcm_temp']
        gcm_prec = main_vars['gcm_prec']
        gcm_elev = main_vars['gcm_elev']
        modelparameters = main_vars['modelparameters']
        glac_wide_massbaltotal = main_vars['glac_wide_massbaltotal']
        glac_wide_area_annual = main_vars['glac_wide_area_annual']
        glac_wide_volume_annual = main_vars['glac_wide_volume_annual']
        glacier_rgi_table = main_vars['glacier_rgi_table']
        main_glac_modelparamsopt = main_vars['main_glac_modelparamsopt']
        main_glac_massbal_compare = main_vars['main_glac_massbal_compare']
#        glacier_gcm_temp = main_vars['glacier_gcm_temp'][gcm_spinupyears*12:]
#        glacier_gcm_prec = main_vars['glacier_gcm_prec'][gcm_spinupyears*12:]
#        glacier_gcm_elev = main_vars['glacier_gcm_elev']
#        glacier_gcm_lrgcm = main_vars['glacier_gcm_lrgcm'][gcm_spinupyears*12:]    
        
    #    # Adjust temperature and precipitation to 'Zmed' so variables can properly be compared
    #    glacier_elev_zmed = glacier_rgi_table.loc['Zmed']  
    #    glacier_gcm_temp_zmed = glacier_gcm_temp + glacier_gcm_lrgcm * (glacier_elev_zmed - glacier_gcm_elev)
    #    glacier_gcm_prec_zmed = glacier_gcm_prec * modelparameters['precfactor']
    #    
    #    glac_wide_massbaltotal_annual = glac_wide_massbaltotal.reshape(-1,12).sum(axis=1)
    #    # Plot reference vs. GCM temperature and precipitation
    #    # Monthly trends
    #    months = dates_table['date'][gcm_spinupyears*12:]
    #    years = np.unique(dates_table['wateryear'].values)[gcm_spinupyears:]
    #    
    #    # Temperature
    #    plt.plot(months, glacier_gcm_temp_zmed, label='gcm_temp')
    #    plt.ylabel('Monthly temperature [degC]')
    #    plt.legend()
    #    plt.show()
    #    # Precipitation
    #    plt.plot(months, glacier_gcm_prec_zmed, label='gcm_prec')
    #    plt.ylabel('Monthly precipitation [m]')
    #    plt.legend()
    #    plt.show()
    #    
    #    # Annual trends
    #    glacier_gcm_temp_zmed_annual = glacier_gcm_temp_zmed.reshape(-1,12).mean(axis=1)
    #    glacier_gcm_prec_zmed_annual = glacier_gcm_prec_zmed.reshape(-1,12).sum(axis=1)
    #    # Temperature
    #    plt.plot(years, glacier_gcm_temp_zmed_annual, label='gcm_temp')
    #    plt.ylabel('Mean annual temperature [degC]')
    #    plt.legend()
    #    plt.show()
    #    # Precipitation
    #    plt.plot(years, glacier_gcm_prec_zmed_annual, label='gcm_prec')
    #    plt.ylabel('Total annual precipitation [m]')
    #    plt.legend()
    #    plt.show()
    #    # Mass balance - bar plot
    #    bar_width = 0.35
    #    plt.bar(years+bar_width, glac_wide_massbaltotal_annual, bar_width, label='gcm_MB')
    #    plt.ylabel('Glacier-wide mass balance [mwea]')
    #    plt.legend()
    #    plt.show()
    #    # Cumulative mass balance - bar plot
    #    glac_wide_massbaltotal_annual_cumsum = np.cumsum(glac_wide_massbaltotal_annual)
    #    bar_width = 0.35
    #    plt.bar(years+bar_width, glac_wide_massbaltotal_annual_cumsum, bar_width, label='gcm_MB')
    #    plt.ylabel('Cumulative glacier-wide mass balance [mwe]')
    #    plt.legend()
    #    plt.show() 
        
    #    # Histogram of differences
    #    mb_dif = main_glac_bias_adj['ref_mb_mwea'] - main_glac_bias_adj['gcm_mb_mwea']
    #    plt.hist(mb_dif)
    #    plt.show()      