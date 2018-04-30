r"""
preprocessing_gcmbiascorrections.py outputs the adjustment parameters for temperature and precipitation as well as the
mean monthly lapse rates derived via comparisons with the calibration climate dataset.  These will be used to correct 
the GCM climate data for future simulations.

How to run file?
  - In command line:
      change directory to folder with script
      python run_gcmbiascorrections_list_multiprocess.py C:\Users\David\Dave_Rounce\HiMAT\Climate_data\cmip5\gcm_rcpXX_f
      ilenames.txt
  - In spyder:
      %run run_gcmbiascorrections_list_multiprocess.py C:\Users\David\Dave_Rounce\HiMAT\Climate_data\cmip5\gcm_rcpXX_fil
      enames.txt

Adjustment Options:
  Option 1 (default) - adjust the mean tempearture such that the cumulative positive degree days [degC*day] is equal
                       (cumulative positive degree days [degC*day] are exact, but mean temperature is different and does
                        a poor job handling all negative values)
  Option 2 - adjust mean monthly temperature and incorporate interannual variability and
             adjust mean monthly precipitation [Huss and Hock, 2015]
             (cumulative positive degree days [degC*day] is closer than Options 3 & 4, mean temp similar)
  Option 3 - adjust so the mean temperature is the same for both datasets
             (cumulative positive degree days [degC*day] can be significantly different, mean temp similar)
  Option 4 - adjust the mean monthly temperature to be the same for both datasets
             (cumulative positive degree days [degC*day] is closer than Option 1, mean temp similar)

Why use Option 1 instead of Huss and Hock [2015]?
      The model estimates mass balance, which is a function of melt, refreeze, and accumulation.  Another way of
      interpreting this is the cumulative positive degree days, mean temperature, and precipitation as a function of the
      temperature compared to the snow threshold temperature.  The default option applies a single temperature bias
      adjustment to ensure that the cumulative positive degree days is consistent, i.e., the melt over the calibration
      period should be consistent.  Similarly, adjusting the precipitation ensures the accumulation is consistent.  The 
      mean annual temperature may change, but the impact of this is considered to be negligible since refreeze
      contributes the least to the mass balance.  Huss and Hock [2015] on the other hand make the mean temperature
      fairly consistent while trying to capture the interannual variability, but the changes to melt and accumulation
      could still be significant.
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

import pygem_input as input
import pygemfxns_modelsetup as modelsetup
import pygemfxns_climate as climate

#%% INPUT 
# Glacier selection
rgi_regionsO1 = [15]
#rgi_glac_number = 'all'
rgi_glac_number = ['03473', '03733']
# Required input
option_bias_adjustment = 1
gcm_endyear = 2100
output_filepath = input.main_directory + '/../Climate_data/cmip5/bias_adjusted_1995_2100/'
gcm_filepath_var_prefix = input.main_directory + '/../Climate_data/cmip5/'
gcm_filepath_var_ending = '_r1i1p1_monNG/'
gcm_filepath_fx_prefix = input.main_directory + '/../Climate_data/cmip5/'
gcm_filepath_fx_ending = '_r0i0p0_fx/'
gcm_temp_fn_prefix = 'tas_mon_'
gcm_prec_fn_prefix = 'pr_mon_'
gcm_var_ending = '_r1i1p1_native.nc'
gcm_elev_fn_prefix  = 'orog_fx_'
gcm_fx_ending  = '_r0i0p0.nc'
gcm_startyear = 2000
gcm_spinupyears = 5
glac_elev_name4temp = 'Zmin'
glac_elev_name4prec = 'Zmed'
gcm_temp_varname = 'tas'
gcm_prec_varname = 'pr'
gcm_elev_varname = 'orog'
gcm_lat_varname = 'lat'
gcm_lon_varname = 'lon'
# Reference data
filepath_ref = input.main_directory + '/../Climate_data/ERA_Interim/' 
filename_ref_temp = input.gcmtemp_filedict[rgi_regionsO1[0]]
filename_ref_prec = input.gcmprec_filedict[rgi_regionsO1[0]]
filename_ref_elev = input.gcmelev_filedict[rgi_regionsO1[0]]
filename_ref_lr = input.gcmlapserate_filedict[rgi_regionsO1[0]]
# Calibrated model parameters
filepath_modelparams = input.main_directory + '/../Calibration_datasets/'
filename_modelparams = 'calparams_R15_20180403_nearest.csv'

#%% FUNCTIONS
def getparser():
    parser = argparse.ArgumentParser(description="run gcm bias corrections from gcm list in parallel")
    # add arguments
    parser.add_argument('gcm_file', action='store', type=str, default='gcm_rcpXX_filenames.txt', 
                        help='text file full of commands to run')
    parser.add_argument('-num_simultaneous_processes', action='store', type=int, default=5, 
                        help='number of simultaneous processes (cores) to use')
    parser.add_argument('-option_parallels', action='store', type=int, default=1,
                        help='Switch to use or not use paralles (1 - use parallels, 0 - do not)')
    return parser

def main(gcm_name):
    time_start = time.time()
    parser = getparser()
    args = parser.parse_args()
    # RCP scenario
    rcp_scenario = os.path.basename(args.gcm_file).split('_')[1]
    
    # LOAD REFERENCE DATA 
    # Select glaciers that adjustment is being performed on
    main_glac_rgi = modelsetup.selectglaciersrgitable(rgi_regionsO1=rgi_regionsO1, rgi_glac_number=rgi_glac_number)
    # Select glacier elevations used for the temperature and precipitation corrections
    #  default for temperature is 'Zmin' since this is assumed to have maximum melt (cumulative positive degree days)
    #  default for precipitation is 'Zmed' since this is assumed to be around the ELA and have snow accumulation
    glac_elev4temp = main_glac_rgi[glac_elev_name4temp].values
    glac_elev4prec = main_glac_rgi[glac_elev_name4prec].values
    # Select dates including future projections
    dates_table, start_date, end_date = modelsetup.datesmodelrun(endyear=gcm_endyear)
    # Load reference data
    # Import air temperature, precipitation, lapse rates, and elevation from pre-processed csv files for a given region
    #  This saves time as opposed to running the nearest neighbor for the reference data as well
    ref_temp_all = np.genfromtxt(filepath_ref + filename_ref_temp, delimiter=',')
    ref_prec_all = np.genfromtxt(filepath_ref + filename_ref_prec, delimiter=',')
    ref_elev_all = np.genfromtxt(filepath_ref + filename_ref_elev, delimiter=',')
    ref_lr_all = np.genfromtxt(filepath_ref + filename_ref_lr, delimiter=',')
    modelparameters_all = pd.read_csv(filepath_modelparams + filename_modelparams)
    modelparameters_all = modelparameters_all.values
    # Select the climate data for the glaciers included in the study
    if rgi_glac_number == 'all':
        ref_temp_raw = ref_temp_all
        ref_prec_raw = ref_prec_all
        ref_elev = ref_elev_all
        ref_lr = ref_lr_all
        modelparameters = modelparameters_all
    else:
        ref_temp_raw = np.zeros((main_glac_rgi.shape[0], ref_temp_all.shape[1]))
        ref_prec_raw = np.zeros((main_glac_rgi.shape[0], ref_temp_all.shape[1]))
        ref_elev = np.zeros((main_glac_rgi.shape[0]))
        ref_lr = np.zeros((main_glac_rgi.shape[0], ref_temp_all.shape[1]))
        modelparameters = np.zeros((main_glac_rgi.shape[0], modelparameters_all.shape[1]))
        # Select climate data for each glacier using O1Index
        for glac in range(main_glac_rgi.shape[0]):
            ref_temp_raw[glac,:] = ref_temp_all[main_glac_rgi.loc[glac,'O1Index'],:]
            ref_prec_raw[glac,:] = ref_prec_all[main_glac_rgi.loc[glac,'O1Index'],:]
            ref_elev[glac] = ref_elev_all[main_glac_rgi.loc[glac,'O1Index']]
            ref_lr[glac,:] = ref_lr_all[main_glac_rgi.loc[glac,'O1Index'],:]
            modelparameters[glac,:] = modelparameters_all[main_glac_rgi.loc[glac,'O1Index'],:]
    # Monthly lapse rate
    ref_lr_monthly_avg = (ref_lr.reshape(-1,12).transpose().reshape(-1,int(ref_temp_raw.shape[1]/12)).mean(1).reshape(12,-1)
                          .transpose())
    # Adjust temperature and precipitation to glacier elevation using the model parameters
    #  T = T_gcm + lr_gcm * (z_ref - z_gcm) + tempchange    
    ref_temp = (ref_temp_raw + ref_lr*(glac_elev4temp - ref_elev)[:,np.newaxis]) + modelparameters[:,7][:,np.newaxis]
    #  P = P_gcm * prec_factor * (1 + prec_grad * (z_glac - z_ref))
    ref_prec = (ref_prec_raw * (modelparameters[:,2] * (1 + modelparameters[:,3] * 
                                (glac_elev4prec - ref_elev)))[:,np.newaxis])
    # GCM filepaths
    gcm_filepath_var = gcm_filepath_var_prefix + rcp_scenario + gcm_filepath_var_ending
    gcm_filepath_fx = gcm_filepath_fx_prefix + rcp_scenario + gcm_filepath_fx_ending
    # GCM filenames
    gcm_temp_filename = gcm_temp_fn_prefix + gcm_name + '_' + rcp_scenario + gcm_var_ending
    gcm_prec_filename = gcm_prec_fn_prefix + gcm_name + '_' + rcp_scenario + gcm_var_ending
    gcm_elev_filename = gcm_elev_fn_prefix + gcm_name + '_' + rcp_scenario + gcm_fx_ending
    # GCM data
    gcm_temp_raw, gcm_dates = climate.importGCMvarnearestneighbor_xarray(
            gcm_temp_filename, gcm_temp_varname, main_glac_rgi, dates_table, start_date, end_date, 
            filepath=gcm_filepath_var, gcm_lon_varname=gcm_lon_varname, gcm_lat_varname=gcm_lat_varname)
    gcm_prec_raw, gcm_dates = climate.importGCMvarnearestneighbor_xarray(
            gcm_prec_filename, gcm_prec_varname, main_glac_rgi, dates_table, start_date, end_date, 
            filepath=gcm_filepath_var, gcm_lon_varname=gcm_lon_varname, gcm_lat_varname=gcm_lat_varname)
    gcm_elev = climate.importGCMfxnearestneighbor_xarray(
            gcm_elev_filename, gcm_elev_varname, main_glac_rgi, filepath=gcm_filepath_fx, 
            gcm_lon_varname=gcm_lon_varname, gcm_lat_varname=gcm_lat_varname)
    # Monthly lapse rate
    gcm_lr = np.tile(ref_lr_monthly_avg, int(gcm_temp_raw.shape[1]/12))
    # Adjust temperature and precipitation to glacier elevation using the model parameters
    #  T = T_gcm + lr_gcm * (z_ref - z_gcm) + tempchange 
    gcm_temp = (gcm_temp_raw + gcm_lr*(glac_elev4temp - gcm_elev)[:,np.newaxis]) + modelparameters[:,7][:,np.newaxis]
    #  P = P_gcm * prec_factor * (1 + prec_grad * (z_glac - z_ref))
    gcm_prec = (gcm_prec_raw * (modelparameters[:,2] * (1 + modelparameters[:,3] * 
                                (glac_elev4prec - gcm_elev)))[:,np.newaxis])
    # GCM subset to agree with reference time period to calculate bias corrections
    gcm_temp_subset = gcm_temp[:,0:ref_temp.shape[1]]
    gcm_prec_subset = gcm_prec[:,0:ref_temp.shape[1]]
    
    # TEMPERATURE BIAS CORRECTIONS
    if option_bias_adjustment == 1:
        # Remove negative values for positive degree day calculation
        ref_temp_pos = ref_temp.copy()
        ref_temp_pos[ref_temp_pos < 0] = 0
        # Select days per month
        daysinmonth = dates_table['daysinmonth'].values[0:ref_temp.shape[1]]
        # Cumulative positive degree days [degC*day] for reference period
        ref_PDD = (ref_temp_pos * daysinmonth).sum(1)
        # Optimize bias adjustment such that PDD are equal
        bias_adj_temp = np.zeros(ref_temp.shape[0])
        for glac in range(ref_temp.shape[0]):
            ref_PDD_glac = ref_PDD[glac]
            gcm_temp_glac = gcm_temp_subset[glac,:]
            def objective(bias_adj_glac):
                gcm_temp_glac_adj = gcm_temp_glac + bias_adj_glac
                gcm_temp_glac_adj[gcm_temp_glac_adj < 0] = 0
                gcm_PDD_glac = (gcm_temp_glac_adj * daysinmonth).sum()
                return abs(ref_PDD_glac - gcm_PDD_glac)
            # - initial guess
            bias_adj_init = 0      
            # - run optimization
            bias_adj_temp_opt = minimize(objective, bias_adj_init, method='SLSQP', tol=1e-5)
            bias_adj_temp[glac] = bias_adj_temp_opt.x
#        gcm_temp_bias_adj = gcm_temp + bias_adj_temp[:,np.newaxis]
    elif option_bias_adjustment == 2:
        # Calculate monthly mean temperature
        ref_temp_monthly_avg = (ref_temp.reshape(-1,12).transpose().reshape(-1,int(ref_temp.shape[1]/12)).mean(1)
                                .reshape(12,-1).transpose())
        gcm_temp_monthly_avg = (gcm_temp_subset.reshape(-1,12).transpose().reshape(-1,int(ref_temp.shape[1]/12)).mean(1)
                                .reshape(12,-1).transpose())
        gcm_temp_monthly_adj = ref_temp_monthly_avg - gcm_temp_monthly_avg
        # Monthly temperature bias adjusted according to monthly average
#        t_mt = gcm_temp + np.tile(gcm_temp_monthly_adj, int(gcm_temp.shape[1]/12))
        # Mean monthly temperature bias adjusted according to monthly average
#        t_m25avg = np.tile(gcm_temp_monthly_avg + gcm_temp_monthly_adj, int(gcm_temp.shape[1]/12))
        # Calculate monthly standard deviation of temperature
        ref_temp_monthly_std = (ref_temp.reshape(-1,12).transpose().reshape(-1,int(ref_temp.shape[1]/12)).std(1)
                                .reshape(12,-1).transpose())
        gcm_temp_monthly_std = (gcm_temp_subset.reshape(-1,12).transpose().reshape(-1,int(ref_temp.shape[1]/12)).std(1)
                                .reshape(12,-1).transpose())
        variability_monthly_std = ref_temp_monthly_std / gcm_temp_monthly_std
        # Bias adjusted temperature accounting for monthly mean and variability
#        gcm_temp_bias_adj = t_m25avg + (t_mt - t_m25avg) * np.tile(variability_monthly_std, int(gcm_temp.shape[1]/12))
#    elif option_bias_adjustment == 3:
#        # Reference - GCM difference
#        bias_adj_temp= (ref_temp - gcm_temp_subset).mean(axis=1)
#        # Bias adjusted temperature accounting for mean of entire time period
##        gcm_temp_bias_adj = gcm_temp + bias_adj_temp[:,np.newaxis]
#    elif option_bias_adjustment == 4:
#        # Calculate monthly mean temperature
#        ref_temp_monthly_avg = (ref_temp.reshape(-1,12).transpose().reshape(-1,int(ref_temp.shape[1]/12)).mean(1)
#                                .reshape(12,-1).transpose())
#        gcm_temp_monthly_avg = (gcm_temp_subset.reshape(-1,12).transpose().reshape(-1,int(ref_temp.shape[1]/12)).mean(1)
#                                .reshape(12,-1).transpose())
#        bias_adj_temp = ref_temp_monthly_avg - gcm_temp_monthly_avg
#        # Bias adjusted temperature accounting for monthly mean
##        gcm_temp_bias_adj = gcm_temp + np.tile(bias_adj_temp, int(gcm_temp.shape[1]/12))
    
    # PRECIPITATION BIAS CORRECTIONS
    if option_bias_adjustment == 1:
        # Temperature consistent with precipitation elevation
        #  T = T_gcm + lr_gcm * (z_ref - z_gcm) + tempchange + bias_adjustment
        ref_temp4prec = ((ref_temp_raw + ref_lr*(glac_elev4prec - ref_elev)[:,np.newaxis]) + (modelparameters[:,7] + 
                         bias_adj_temp)[:,np.newaxis])
        gcm_temp4prec = ((gcm_temp_raw + gcm_lr*(glac_elev4prec - gcm_elev)[:,np.newaxis]) + (modelparameters[:,7] + 
                         bias_adj_temp)[:,np.newaxis])[:,0:ref_temp.shape[1]]
        # Snow accumulation should be consistent for reference and gcm datasets
        if input.option_accumulation == 1:
            # Single snow temperature threshold
            ref_snow = np.zeros(ref_temp.shape)
            gcm_snow = np.zeros(ref_temp.shape)
            for glac in range(main_glac_rgi.shape[0]):
                ref_snow[glac, ref_temp4prec[glac,:] < modelparameters[glac,6]] = (
                        ref_prec[glac, ref_temp4prec[glac,:] < modelparameters[glac,6]])
                gcm_snow[glac, gcm_temp4prec[glac,:] < modelparameters[glac,6]] = (
                        gcm_prec_subset[glac, gcm_temp4prec[glac,:] < modelparameters[glac,6]])
        elif input.option_accumulation == 2:
            # Linear snow threshold +/- 1 degree
            # If temperature between min/max, then mix of snow/rain using linear relationship between min/max
            ref_snow = (1/2 + (ref_temp4prec - modelparameters[:,6][:,np.newaxis]) / 2) * ref_prec
            gcm_snow = (1/2 + (gcm_temp4prec - modelparameters[:,6][:,np.newaxis]) / 2) * gcm_prec_subset
            # If temperature above or below the max or min, then all rain or snow, respectively. 
            for glac in range(main_glac_rgi.shape[0]):
                ref_snow[glac, ref_temp4prec[glac,:] > modelparameters[glac,6] + 1] = 0 
                ref_snow[glac, ref_temp4prec[glac,:] < modelparameters[glac,6] - 1] = (
                        ref_prec[glac, ref_temp4prec[glac,:] < modelparameters[glac,6] - 1])
                gcm_snow[glac, gcm_temp4prec[glac,:] > modelparameters[glac,6] + 1] = 0
                gcm_snow[glac, gcm_temp4prec[glac,:] < modelparameters[glac,6] - 1] = (
                        gcm_prec_subset[glac, gcm_temp4prec[glac,:] < modelparameters[glac,6] - 1])
        # precipitation bias adjustment
        bias_adj_prec = ref_snow.sum(1) / gcm_snow.sum(1)
#        gcm_prec_bias_adj = gcm_prec * bias_adj_prec[:,np.newaxis]
    else:
        # Calculate monthly mean precipitation
        ref_prec_monthly_avg = (ref_prec.reshape(-1,12).transpose().reshape(-1,int(ref_temp.shape[1]/12)).mean(1)
                                .reshape(12,-1).transpose())
        gcm_prec_monthly_avg = (gcm_prec_subset.reshape(-1,12).transpose().reshape(-1,int(ref_temp.shape[1]/12)).mean(1)
                                .reshape(12,-1).transpose())
        bias_adj_prec = ref_prec_monthly_avg / gcm_prec_monthly_avg
        # Bias adjusted precipitation accounting for differences in monthly mean
#        gcm_prec_bias_adj = gcm_prec * np.tile(bias_adj_prec, int(gcm_temp.shape[1]/12))
    
#    # EXPORT THE ADJUSTMENT VARIABLES (greatly reduces space)
#    # Set up directory to store climate data
#    if os.path.exists(output_filepath) == False:
#        os.makedirs(output_filepath)
#    # Lapse rate parameters (same for all GCMs - only need to export once)
#    output_filename_lr = 'biasadj_mon_lravg_' + str(gcm_startyear - gcm_spinupyears) + '_' + str(gcm_endyear) + '.csv'
#    if os.path.exists(output_filepath + output_filename_lr) == False:
#        np.savetxt(output_filepath + output_filename_lr, ref_lr_monthly_avg, delimiter=",")
#    # Temperature and precipitation parameters
#    if (option_bias_adjustment == 1) or (option_bias_adjustment == 3) or (option_bias_adjustment == 4):
#        # Temperature parameters
#        output_tempadj = (gcm_name + '_' + rcp_scenario + '_temp_biasadjparam_opt1_' + 
#                          str(gcm_startyear - gcm_spinupyears) + '_' + str(gcm_endyear) + '.csv')
#        np.savetxt(output_filepath + output_tempadj, bias_adj_temp, delimiter=",")
#        # Precipitation parameters
#        output_precadj = (gcm_name + '_' + rcp_scenario + '_prec_biasadjparam_opt1_' + 
#                          str(gcm_startyear - gcm_spinupyears) + '_' + str(gcm_endyear) + '.csv')
#        np.savetxt(output_filepath + output_precadj, bias_adj_prec, delimiter=",")
#    elif option_bias_adjustment == 2:
#        # Temperature parameters
#        output_tempvar = (gcm_name + '_' + rcp_scenario + '_biasadjparam_HH2015_mon_tempvar_' + 
#                          str(gcm_startyear - gcm_spinupyears) + '_' + str(gcm_endyear) + '.csv')
#        output_tempavg = (gcm_name + '_' + rcp_scenario + '_biasadjparam_HH2015_mon_tempavg_' + 
#                          str(gcm_startyear - gcm_spinupyears) + '_' + str(gcm_endyear) + '.csv')
#        output_tempadj = (gcm_name + '_' + rcp_scenario + '_biasadjparam_HH2015_mon_tempadj_' + 
#                          str(gcm_startyear - gcm_spinupyears) + '_' + str(gcm_endyear) + '.csv')
#        np.savetxt(output_filepath + output_tempvar, variability_monthly_std, delimiter=",") 
#        np.savetxt(output_filepath + output_tempavg, gcm_temp_monthly_avg, delimiter=",") 
#        np.savetxt(output_filepath + output_tempadj, gcm_temp_monthly_adj, delimiter=",")
#        # Precipitation parameters
#        output_precadj = (gcm_name + '_' + rcp_scenario + '_biasadjparam_HH2015_mon_precadj_' + 
#                          str(gcm_startyear - gcm_spinupyears) + '_' + str(gcm_endyear) + '.csv')
#        np.savetxt(output_filepath + output_precadj, bias_adj_prec, delimiter=",")  
        
    # Export variables as global to view in variable explorer
    global main_vars
    main_vars = inspect.currentframe().f_locals
    
    print('\nProcessing time of',gcm_name,':',time.time()-time_start, 's')

#%% PARALLEL PROCESSING
if __name__ == '__main__':
    time_start = time.time()
    parser = getparser()
    args = parser.parse_args()
    # Read GCM names from command file
    with open(args.gcm_file, 'r') as gcm_fn:
        gcm_list = gcm_fn.read().splitlines()
    print('Found %d gcms to process'%(len(gcm_list)))
    
    if args.option_parallels != 0:
        with multiprocessing.Pool(args.num_simultaneous_processes) as p:
            p.map(main,gcm_list)
    else:
        # Loop through GCMs and export bias adjustments
        for n_gcm in range(len(gcm_list)):
            gcm_name = gcm_list[n_gcm]
            # Perform GCM Bias adjustments
            main(gcm_name)
            
            # Place local variables in variable explorer
            vars_list = list(main_vars.keys())
            gcm_name = main_vars['gcm_name']
            rcp_scenario = main_vars['rcp_scenario']
            main_glac_rgi = main_vars['main_glac_rgi']
#            main_glac_hyps = main_vars['main_glac_hyps']
#            main_glac_icethickness = main_vars['main_glac_icethickness']
#            main_glac_width = main_vars['main_glac_width']
#            elev_bins = main_vars['elev_bins']
#            main_glac_gcmtemp = main_vars['main_glac_gcmtemp']
#            main_glac_gcmprec = main_vars['main_glac_gcmprec']
#            main_glac_gcmelev = main_vars['main_glac_gcmelev']
#            main_glac_gcmlapserate = main_vars['main_glac_gcmlapserate']
#            main_glac_modelparams = main_vars['main_glac_modelparams']
#            end_date = main_vars['end_date']
#            start_date = main_vars['start_date']
#            dates_table = main_vars['dates_table']
#            glac_wide_volume_annual = main_vars['glac_wide_volume_annual']
#            glac_wide_area_annual = main_vars['glac_wide_area_annual']
#            glac_bin_area_annual = main_vars['glac_bin_area_annual']
#            glac_wide_massbaltotal = main_vars['glac_wide_massbaltotal']
#            glac_bin_massbalclim_annual = main_vars['glac_bin_massbalclim_annual']
#            biasadj_temp = main_vars['biasadj_temp']
#            biasadj_prec = main_vars['biasadj_prec']  
            
    print('Total processing time:', time.time()-time_start, 's')            