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
import pygemfxns_massbalance as massbalance

#%% INPUT 
# Glacier selection
rgi_regionsO1 = [15]
rgi_glac_number = 'all'
#rgi_glac_number = ['03473', '03733']
#rgi_glac_number = ['03473']  # Ngozumpa
# Required input
option_bias_adjustment = 1
gcm_endyear = 2100
output_filepath = input.main_directory + '/../Climate_data/cmip5/bias_adjusted_1995_2100/'
#output_filepath = input.main_directory + '/../Climate_data/cmip5/biasadj_comparison/'
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
massbal_idx_start = 0  # 2000
massbal_idx_end = 16   # 2015
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

# Delete when done modeling and restore main()
parser = getparser()
args = parser.parse_args()
with open(args.gcm_file, 'r') as gcm_fn:
    gcm_list = gcm_fn.read().splitlines()
    gcm_name = gcm_list[0]
print('Found %d gcms to process'%(len(gcm_list)))
for batman in [0]:
    
#def main(gcm_name):
    print(gcm_name)
    time_start = time.time()
    parser = getparser()
    args = parser.parse_args()
    # RCP scenario
    rcp_scenario = os.path.basename(args.gcm_file).split('_')[1]
    
    # LOAD GLACIER DATA 
    main_glac_rgi = modelsetup.selectglaciersrgitable(rgi_regionsO1=rgi_regionsO1, rgi_glac_number=rgi_glac_number)
    # Glacier hypsometry [km**2], total area
    main_glac_hyps = modelsetup.import_Husstable(main_glac_rgi, rgi_regionsO1, input.hyps_filepath, 
                                                 input.hyps_filedict, input.hyps_colsdrop)
    # Ice thickness [m], average
    main_glac_icethickness = modelsetup.import_Husstable(main_glac_rgi, input.rgi_regionsO1, input.thickness_filepath, 
                                                         input.thickness_filedict, input.thickness_colsdrop)
    # Width [km], average
    main_glac_width = modelsetup.import_Husstable(main_glac_rgi, input.rgi_regionsO1, input.width_filepath, 
                                                  input.width_filedict, input.width_colsdrop)
    elev_bins = main_glac_hyps.columns.values.astype(int)
    # Model parameters
    main_glac_modelparams_all = (pd.read_csv(filepath_modelparams + filename_modelparams)).values
    main_glac_modelparams = main_glac_modelparams_all[main_glac_rgi['O1Index'].values]
    # Select dates including future projections
    dates_table, start_date, end_date = modelsetup.datesmodelrun(endyear=gcm_endyear)
    
    # LOAD REFERENCE CLIMATE DATA
    # Import air temperature, precipitation, lapse rates, and elevation from pre-processed csv files for a given region
    #  This saves time as opposed to running the nearest neighbor for the reference data as well
    ref_temp_all = np.genfromtxt(filepath_ref + filename_ref_temp, delimiter=',')
    ref_prec_all = np.genfromtxt(filepath_ref + filename_ref_prec, delimiter=',')
    ref_elev_all = np.genfromtxt(filepath_ref + filename_ref_elev, delimiter=',')
    ref_lr_all = np.genfromtxt(filepath_ref + filename_ref_lr, delimiter=',')
    # Select the climate data for the glaciers included in the study
    ref_temp = ref_temp_all[main_glac_rgi['O1Index'].values]
    ref_prec = ref_prec_all[main_glac_rgi['O1Index'].values]
    ref_elev = ref_elev_all[main_glac_rgi['O1Index'].values]
    ref_lr = ref_lr_all[main_glac_rgi['O1Index'].values]
    # Monthly lapse rate
    ref_lr_monthly_avg = (ref_lr.reshape(-1,12).transpose().reshape(-1,int(ref_temp.shape[1]/12)).mean(1)
                          .reshape(12,-1).transpose())
    # Days per month
    daysinmonth = dates_table['daysinmonth'].values[0:ref_temp.shape[1]]
    dates_table_subset = dates_table.iloc[0:ref_temp.shape[1],:]
    
    # LOAD GCM DATA
    gcm_filepath_var = gcm_filepath_var_prefix + rcp_scenario + gcm_filepath_var_ending
    gcm_filepath_fx = gcm_filepath_fx_prefix + rcp_scenario + gcm_filepath_fx_ending
    gcm_temp_fn = gcm_temp_fn_prefix + gcm_name + '_' + rcp_scenario + gcm_var_ending
    gcm_prec_fn = gcm_prec_fn_prefix + gcm_name + '_' + rcp_scenario + gcm_var_ending
    gcm_elev_fn = gcm_elev_fn_prefix + gcm_name + '_' + rcp_scenario + gcm_fx_ending
    gcm_temp, gcm_dates = climate.importGCMvarnearestneighbor_xarray(
            gcm_temp_fn, gcm_temp_varname, main_glac_rgi, dates_table, start_date, end_date, 
            filepath=gcm_filepath_var, gcm_lon_varname=gcm_lon_varname, gcm_lat_varname=gcm_lat_varname)
    gcm_prec, gcm_dates = climate.importGCMvarnearestneighbor_xarray(
            gcm_prec_fn, gcm_prec_varname, main_glac_rgi, dates_table, start_date, end_date, 
            filepath=gcm_filepath_var, gcm_lon_varname=gcm_lon_varname, gcm_lat_varname=gcm_lat_varname)
    gcm_elev = climate.importGCMfxnearestneighbor_xarray(
            gcm_elev_fn, gcm_elev_varname, main_glac_rgi, filepath=gcm_filepath_fx, 
            gcm_lon_varname=gcm_lon_varname, gcm_lat_varname=gcm_lat_varname)    
    # Monthly lapse rate
    gcm_lr = np.tile(ref_lr_monthly_avg, int(gcm_temp.shape[1]/12))
    # GCM subset to agree with reference time period to calculate bias corrections
    gcm_temp_subset = gcm_temp[:,0:ref_temp.shape[1]]
    gcm_prec_subset = gcm_prec[:,0:ref_temp.shape[1]]
    gcm_lr_subset = gcm_lr[:,0:ref_temp.shape[1]]
    
    
    # BIAS CORRECTIONS: OPTION 1
    if option_bias_adjustment == 1:
        # Bias adjustment parameters
        main_glac_bias_adj_params = np.zeros((main_glac_rgi.shape[0],2))
        # BIAS ADJUSTMENT CALCULATIONS
        for glac in range(main_glac_rgi.shape[0]):    
            # Print every 100th glacier
            if glac%100 == 0:
                print(gcm_name,':', main_glac_rgi.loc[glac,'RGIId'])    
            # Glacier data
            glacier_rgi_table = main_glac_rgi.loc[glac, :]
            glacier_area_t0 = main_glac_hyps.iloc[glac,:].values.astype(float)
            icethickness_t0 = main_glac_icethickness.iloc[glac,:].values.astype(float)
            width_t0 = main_glac_width.iloc[glac,:].values.astype(float)
            modelparameters = main_glac_modelparams[glac,:]
            glac_idx_t0 = glacier_area_t0.nonzero()[0]
            surfacetype, firnline_idx = massbalance.surfacetypebinsinitial(glacier_area_t0, glacier_rgi_table, 
                                                                           elev_bins)
            surfacetype_ddf_dict = massbalance.surfacetypeDDFdict(modelparameters, option_DDF_firn=0)
            #  option_DDF_firn=0 uses DDF_snow in accumulation area because not account for snow vs. firn here
            surfacetype_ddf = np.zeros(glacier_area_t0.shape)
            for surfacetype_idx in surfacetype_ddf_dict: 
                surfacetype_ddf[surfacetype == surfacetype_idx] = surfacetype_ddf_dict[surfacetype_idx]
            # Reference data
            glacier_ref_temp = ref_temp[glac,:]
            glacier_ref_prec = ref_prec[glac,:]
            glacier_ref_elev = ref_elev[glac]
            glacier_ref_lrgcm = ref_lr[glac,:]
            glacier_ref_lrglac = ref_lr[glac,:]
            # GCM data
            glacier_gcm_temp = gcm_temp_subset[glac,:]
            glacier_gcm_prec = gcm_prec_subset[glac,:]
            glacier_gcm_elev = gcm_elev[glac]
            glacier_gcm_lrgcm = gcm_lr_subset[glac,:]
            glacier_gcm_lrglac = gcm_lr_subset[glac,:]
            
            # AIR TEMPERATURE: Downscale the gcm temperature [deg C] to each bin
            if input.option_temp2bins == 1:
                #  T_bin = T_gcm + lr_gcm * (z_ref - z_gcm) + lr_glac * (z_bin - z_ref) + tempchange
                glac_bin_temp_ref = (
                        glacier_ref_temp + glacier_ref_lrgcm * 
                        (glacier_rgi_table.loc[input.option_elev_ref_downscale] - glacier_ref_elev) + 
                        glacier_ref_lrglac * (elev_bins - 
                        glacier_rgi_table.loc[input.option_elev_ref_downscale])[:,np.newaxis] + modelparameters[7])
                glac_bin_temp_gcm = (
                        glacier_gcm_temp + glacier_gcm_lrgcm * 
                        (glacier_rgi_table.loc[input.option_elev_ref_downscale] - glacier_gcm_elev) + 
                        glacier_gcm_lrglac * (elev_bins - 
                        glacier_rgi_table.loc[input.option_elev_ref_downscale])[:,np.newaxis] + modelparameters[7])
            # remove off-glacier values
            glac_bin_temp_ref[glacier_area_t0==0,:] = 0
            glac_bin_temp_gcm[glacier_area_t0==0,:] = 0
        
            # TEMPERATURE BIAS CORRECTIONS
            # Energy available for melt [degC day]    
            melt_energy_available_ref = glac_bin_temp_ref * daysinmonth
            melt_energy_available_ref[melt_energy_available_ref < 0] = 0
            # Melt [mwe for each month]
            melt_ref = melt_energy_available_ref * surfacetype_ddf[:,np.newaxis]
            # Melt volume total [mwe * km2]
            melt_vol_ref = (melt_ref * glacier_area_t0[:,np.newaxis]).sum()
            # Optimize bias adjustment such that PDD are equal                
            def objective(bias_adj_glac):
                glac_bin_temp_gcm_adj = glac_bin_temp_gcm + bias_adj_glac
#                glac_bin_temp_gcm_adj[glac_bin_temp_gcm_adj < 0] = 0
                melt_energy_available_gcm = glac_bin_temp_gcm_adj * daysinmonth
                melt_energy_available_gcm[melt_energy_available_gcm < 0] = 0
                melt_gcm = melt_energy_available_gcm * surfacetype_ddf[:,np.newaxis]
                melt_vol_gcm = (melt_gcm * glacier_area_t0[:,np.newaxis]).sum()
#                glac_bin_temp_gcm_adj_warea = glac_bin_temp_gcm_adj * glacier_area_t0[:,np.newaxis]
#                glac_bin_temp_gcm_PDD_warea = (glac_bin_temp_gcm_adj_warea * daysinmonth).sum()
#                return abs(glac_bin_temp_ref_PDD_warea - glac_bin_temp_gcm_PDD_warea)
                return abs(melt_vol_ref - melt_vol_gcm)
            # - initial guess
            bias_adj_init = 0      
            # - run optimization
            bias_adj_temp_opt = minimize(objective, bias_adj_init, method='SLSQP', tol=1e-5)
            bias_adj_temp_init = bias_adj_temp_opt.x
            glac_bin_temp_gcm_adj = glac_bin_temp_gcm + bias_adj_temp_init
            # PRECIPITATION/ACCUMULATION: Downscale the precipitation (liquid and solid) to each bin
            glac_bin_acc_ref = np.zeros(glac_bin_temp_ref.shape)
            glac_bin_acc_gcm = np.zeros(glac_bin_temp_ref.shape)
            glac_bin_prec_ref = np.zeros(glac_bin_temp_ref.shape)
            glac_bin_prec_gcm = np.zeros(glac_bin_temp_ref.shape)
            if input.option_prec2bins == 1:
                # Precipitation using precipitation factor and precipitation gradient
                #  P_bin = P_gcm * prec_factor * (1 + prec_grad * (z_bin - z_ref))
                glac_bin_precsnow_ref = (glacier_ref_prec * modelparameters[2] * (1 + modelparameters[3] * (elev_bins - 
                                         glacier_rgi_table.loc[input.option_elev_ref_downscale]))[:,np.newaxis])
                glac_bin_precsnow_gcm = (glacier_gcm_prec * modelparameters[2] * (1 + modelparameters[3] * (elev_bins - 
                                         glacier_rgi_table.loc[input.option_elev_ref_downscale]))[:,np.newaxis])
            # Option to adjust prec of uppermost 25% of glacier for wind erosion and reduced moisture content
            if input.option_preclimit == 1:
                # If elevation range > 1000 m, apply corrections to uppermost 25% of glacier (Huss and Hock, 2015)
                if elev_bins[glac_idx_t0[-1]] - elev_bins[glac_idx_t0[0]] > 1000:
                    # Indices of upper 25%
                    glac_idx_upper25 = glac_idx_t0[(glac_idx_t0 - glac_idx_t0[0] + 1) / glac_idx_t0.shape[0] * 100 > 75]   
                    # Exponential decay according to elevation difference from the 75% elevation
                    #  prec_upper25 = prec * exp(-(elev_i - elev_75%)/(elev_max- - elev_75%))
                    glac_bin_precsnow_ref[glac_idx_upper25,:] = (
                            glac_bin_precsnow_ref[glac_idx_upper25[0],:] * np.exp(-1*(elev_bins[glac_idx_upper25] - 
                            elev_bins[glac_idx_upper25[0]]) / (elev_bins[glac_idx_upper25[-1]] - 
                            elev_bins[glac_idx_upper25[0]]))[:,np.newaxis])
                    glac_bin_precsnow_gcm[glac_idx_upper25,:] = (
                            glac_bin_precsnow_gcm[glac_idx_upper25[0],:] * np.exp(-1*(elev_bins[glac_idx_upper25] - 
                            elev_bins[glac_idx_upper25[0]]) / (elev_bins[glac_idx_upper25[-1]] - 
                            elev_bins[glac_idx_upper25[0]]))[:,np.newaxis])
                    # Precipitation cannot be less than 87.5% of the maximum accumulation elsewhere on the glacier
                    for month in range(glac_bin_precsnow_ref.shape[1]):
                        glac_bin_precsnow_ref[glac_idx_upper25[(glac_bin_precsnow_ref[glac_idx_upper25,month] < 0.875 * 
                        glac_bin_precsnow_ref[glac_idx_t0,month].max()) & 
                        (glac_bin_precsnow_ref[glac_idx_upper25,month] != 0)], month] = (
                                                            0.875 * glac_bin_precsnow_ref[glac_idx_t0,month].max())
                        glac_bin_precsnow_gcm[glac_idx_upper25[(glac_bin_precsnow_gcm[glac_idx_upper25,month] < 0.875 * 
                        glac_bin_precsnow_gcm[glac_idx_t0,month].max()) & 
                        (glac_bin_precsnow_gcm[glac_idx_upper25,month] != 0)], month] = (
                                                            0.875 * glac_bin_precsnow_gcm[glac_idx_t0,month].max())
            # Separate total precipitation into liquid (glac_bin_prec) and solid (glac_bin_acc)
            if input.option_accumulation == 1:
                # if temperature above threshold, then rain
                glac_bin_prec_ref[glac_bin_temp_ref > modelparameters[6]] = (
                    glac_bin_precsnow_ref[glac_bin_temp_ref > modelparameters[6]])
                glac_bin_prec_gcm[glac_bin_temp_gcm_adj > modelparameters[6]] = (
                    glac_bin_precsnow_gcm[glac_bin_temp_gcm_adj > modelparameters[6]])
                # if temperature below threshold, then snow
                glac_bin_acc_ref[glac_bin_temp_ref <= modelparameters[6]] = (
                    glac_bin_precsnow_ref[glac_bin_temp_ref <= modelparameters[6]])
                glac_bin_acc_gcm[glac_bin_temp_gcm_adj <= modelparameters[6]] = (
                    glac_bin_precsnow_gcm[glac_bin_temp_gcm_adj <= modelparameters[6]])
            elif input.option_accumulation == 2:
                # If temperature between min/max, then mix of snow/rain using linear relationship between min/max
                glac_bin_prec_ref = (1/2 + (glac_bin_temp_ref - modelparameters[6]) / 2) * glac_bin_precsnow_ref
                glac_bin_prec_gcm = (1/2 + (glac_bin_temp_gcm_adj - modelparameters[6]) / 2) * glac_bin_precsnow_gcm
                glac_bin_acc_ref = glac_bin_precsnow_ref - glac_bin_prec_ref
                glac_bin_acc_gcm = glac_bin_precsnow_gcm - glac_bin_prec_gcm
                # If temperature above maximum threshold, then all rain
                glac_bin_prec_ref[glac_bin_temp_ref > modelparameters[6] + 1] = (
                    glac_bin_precsnow_ref[glac_bin_temp_ref > modelparameters[6] + 1])
                glac_bin_prec_gcm[glac_bin_temp_gcm_adj > modelparameters[6] + 1] = (
                    glac_bin_precsnow_gcm[glac_bin_temp_gcm_adj > modelparameters[6] + 1])
                glac_bin_acc_ref[glac_bin_temp_ref > modelparameters[6] + 1] = 0
                glac_bin_acc_gcm[glac_bin_temp_gcm_adj > modelparameters[6] + 1] = 0
                # If temperature below minimum threshold, then all snow
                glac_bin_acc_ref[glac_bin_temp_ref <= modelparameters[6] - 1] = (
                        glac_bin_precsnow_ref[glac_bin_temp_ref <= modelparameters[6] - 1])
                glac_bin_acc_gcm[glac_bin_temp_gcm_adj <= modelparameters[6] - 1] = (
                        glac_bin_precsnow_gcm[glac_bin_temp_gcm_adj <= modelparameters[6] - 1])
                glac_bin_prec_ref[glac_bin_temp_ref <= modelparameters[6] - 1] = 0
                glac_bin_prec_gcm[glac_bin_temp_gcm_adj <= modelparameters[6] - 1] = 0
            # remove off-glacier values
            glac_bin_acc_ref[glacier_area_t0==0,:] = 0
            glac_bin_acc_gcm[glacier_area_t0==0,:] = 0
            glac_bin_prec_ref[glacier_area_t0==0,:] = 0
            glac_bin_prec_gcm[glacier_area_t0==0,:] = 0
            # account for hypsometry
            glac_bin_acc_ref_warea = glac_bin_acc_ref * glacier_area_t0[:,np.newaxis]
            glac_bin_acc_gcm_warea = glac_bin_acc_gcm * glacier_area_t0[:,np.newaxis]
            # precipitation bias adjustment
            bias_adj_prec_init = glac_bin_acc_ref_warea.sum() / glac_bin_acc_gcm_warea.sum()

            # BIAS ADJUSTMENT PARAMETER OPTIMIZATION such that mass balance between two datasets are equal 
            bias_adj_params = np.zeros((2))
            bias_adj_params[0] = bias_adj_temp_init
            bias_adj_params[1] = bias_adj_prec_init        
            def objective_2(bias_adj_params):
                # Mass balance for reference data
                (glac_bin_temp, glac_bin_prec, glac_bin_acc, glac_bin_refreeze, glac_bin_snowpack, glac_bin_melt, 
                 glac_bin_frontalablation, glac_bin_massbalclim, glac_bin_massbalclim_annual, glac_bin_area_annual, 
                 glac_bin_icethickness_annual, glac_bin_width_annual, glac_bin_surfacetype_annual, 
                 glac_wide_massbaltotal, glac_wide_runoff, glac_wide_snowline, glac_wide_snowpack, 
                 glac_wide_area_annual, glac_wide_volume_annual, glac_wide_ELA_annual) = (
                    massbalance.runmassbalance(modelparameters, glacier_rgi_table, glacier_area_t0, icethickness_t0, 
                                               width_t0, elev_bins, glacier_ref_temp, glacier_ref_prec, 
                                               glacier_ref_elev, glacier_ref_lrgcm, glacier_ref_lrglac, 
                                               dates_table_subset, biasadj_temp=0, biasadj_prec=1, 
                                               option_calibration=1))
                # Total volume loss
                glac_wide_massbaltotal_annual_ref = np.sum(glac_wide_massbaltotal.reshape(-1,12), axis=1)
                glac_wide_volume_loss_total_ref = (
                        np.cumsum(glac_wide_area_annual[glac_wide_massbaltotal_annual_ref.shape] * 
                                  glac_wide_massbaltotal_annual_ref / 1000)[-1])
                glac_wide_volume_loss_total_ref_perc = (
                        glac_wide_volume_loss_total_ref / glac_wide_volume_annual[0] * 100)
                
                # Mass balance for GCM data
                (glac_bin_temp, glac_bin_prec, glac_bin_acc, glac_bin_refreeze, glac_bin_snowpack, glac_bin_melt, 
                 glac_bin_frontalablation, glac_bin_massbalclim, glac_bin_massbalclim_annual, glac_bin_area_annual, 
                 glac_bin_icethickness_annual, glac_bin_width_annual, glac_bin_surfacetype_annual, 
                 glac_wide_massbaltotal, glac_wide_runoff, glac_wide_snowline, glac_wide_snowpack, 
                 glac_wide_area_annual, glac_wide_volume_annual, glac_wide_ELA_annual) = (
                    massbalance.runmassbalance(modelparameters, glacier_rgi_table, glacier_area_t0, icethickness_t0, 
                                               width_t0, elev_bins, glacier_gcm_temp, glacier_gcm_prec, 
                                               glacier_gcm_elev, glacier_gcm_lrgcm, glacier_gcm_lrglac, 
                                               dates_table_subset, biasadj_temp=bias_adj_params[0], 
                                               biasadj_prec=bias_adj_params[1], option_calibration=1))
                # Total volume loss
                glac_wide_massbaltotal_annual_gcm = np.sum(glac_wide_massbaltotal.reshape(-1,12), axis=1)
                glac_wide_volume_loss_total_gcm = (
                        np.cumsum(glac_wide_area_annual[glac_wide_massbaltotal_annual_gcm.shape] * 
                                  glac_wide_massbaltotal_annual_gcm / 1000)[-1])
                glac_wide_volume_loss_total_gcm_perc = (
                        glac_wide_volume_loss_total_gcm / glac_wide_volume_annual[0] * 100)
                return abs(glac_wide_volume_loss_total_ref_perc - glac_wide_volume_loss_total_gcm_perc)
            # INITIAL GUESS
            bias_adj_params_init = bias_adj_params          
            # Run the optimization
            bias_adj_params_opt = minimize(objective_2, bias_adj_params_init, method='SLSQP', tol=1e-3)
            # Record the optimized parameters
            main_glac_bias_adj_params[glac] = bias_adj_params_opt.x
            
            # EXPORT THE ADJUSTMENT VARIABLES (greatly reduces space)
            # Set up directory to store climate data
            if os.path.exists(output_filepath) == False:
                os.makedirs(output_filepath)
            # Temperature and precipitation parameters
            output_biasadjparams = (gcm_name + '_' + rcp_scenario + '_biasadjparams_opt1_' + 
                                    str(gcm_startyear - gcm_spinupyears) + '_' + str(gcm_endyear) + '.csv')
            main_glac_bias_adj_params_export = pd.DataFrame(main_glac_bias_adj_params, 
                                                            columns=['temp_biasadj', 'prec_biasadj'])
            main_glac_bias_adj_params_export.to_csv(output_filepath + output_biasadjparams)
            # Lapse rate parameters (same for all GCMs - only need to export once)
            output_filename_lr = ('biasadj_mon_lravg_' + str(gcm_startyear - gcm_spinupyears) + '_' + str(gcm_endyear) +
                                  '.csv')
            if os.path.exists(output_filepath + output_filename_lr) == False:
                np.savetxt(output_filepath + output_filename_lr, ref_lr_monthly_avg, delimiter=",")
        
            # Compute mass balances to have output data
            # Mass balance for reference data
            (glac_bin_temp, glac_bin_prec, glac_bin_acc, glac_bin_refreeze, glac_bin_snowpack, glac_bin_melt, 
             glac_bin_frontalablation, glac_bin_massbalclim, glac_bin_massbalclim_annual, glac_bin_area_annual, 
             glac_bin_icethickness_annual, glac_bin_width_annual, glac_bin_surfacetype_annual, 
             glac_wide_massbaltotal, glac_wide_runoff, glac_wide_snowline, glac_wide_snowpack, 
             glac_wide_area_annual, glac_wide_volume_annual, glac_wide_ELA_annual) = (
                massbalance.runmassbalance(modelparameters, glacier_rgi_table, glacier_area_t0, icethickness_t0, 
                                           width_t0, elev_bins, glacier_ref_temp, glacier_ref_prec, 
                                           glacier_ref_elev, glacier_ref_lrgcm, glacier_ref_lrglac, 
                                           dates_table_subset, biasadj_temp=0, biasadj_prec=1, 
                                           option_calibration=1))
            # Glacier volume loss
            glac_wide_massbaltotal_annual_ref = np.sum(glac_wide_massbaltotal.reshape(-1,12), axis=1)
            glac_wide_volume_loss_total_ref = (
                    np.cumsum(glac_wide_area_annual[glac_wide_massbaltotal_annual_ref.shape] * 
                              glac_wide_massbaltotal_annual_ref / 1000)[-1])
            glac_wide_volume_loss_total_ref_perc = glac_wide_volume_loss_total_ref / glac_wide_volume_annual[0] * 100
            # Mass balance for GCM data
            (glac_bin_temp, glac_bin_prec, glac_bin_acc, glac_bin_refreeze, glac_bin_snowpack, glac_bin_melt, 
             glac_bin_frontalablation, glac_bin_massbalclim, glac_bin_massbalclim_annual, glac_bin_area_annual, 
             glac_bin_icethickness_annual, glac_bin_width_annual, glac_bin_surfacetype_annual, 
             glac_wide_massbaltotal, glac_wide_runoff, glac_wide_snowline, glac_wide_snowpack, 
             glac_wide_area_annual, glac_wide_volume_annual, glac_wide_ELA_annual) = (
                massbalance.runmassbalance(modelparameters, glacier_rgi_table, glacier_area_t0, icethickness_t0, 
                                           width_t0, elev_bins, glacier_gcm_temp, glacier_gcm_prec, 
                                           glacier_gcm_elev, glacier_gcm_lrgcm, glacier_gcm_lrglac, 
                                           dates_table_subset, biasadj_temp=main_glac_bias_adj_params[glac,0], 
                                           biasadj_prec=main_glac_bias_adj_params[glac,1], option_calibration=1))
            # Glacier volume loss
            glac_wide_massbaltotal_annual_gcm = np.sum(glac_wide_massbaltotal.reshape(-1,12), axis=1)
            glac_wide_volume_loss_total_gcm = (
                    np.cumsum(glac_wide_area_annual[glac_wide_massbaltotal_annual_gcm.shape] * 
                              glac_wide_massbaltotal_annual_gcm / 1000)[-1])
            glac_wide_volume_loss_total_gcm_perc = glac_wide_volume_loss_total_gcm / glac_wide_volume_annual[0] * 100
            
    elif option_bias_adjustment == 2:
        # Huss and Hock (2015)
        # TEMPERATURE BIAS CORRECTIONS
        # Calculate monthly mean temperature
        ref_temp_monthly_avg = (ref_temp.reshape(-1,12).transpose().reshape(-1,int(ref_temp.shape[1]/12)).mean(1)
                                .reshape(12,-1).transpose())
        gcm_temp_monthly_avg = (gcm_temp_subset.reshape(-1,12).transpose().reshape(-1,int(ref_temp.shape[1]/12)).mean(1)
                                .reshape(12,-1).transpose())
        gcm_temp_monthly_adj = ref_temp_monthly_avg - gcm_temp_monthly_avg
        # Monthly temperature bias adjusted according to monthly average
        t_mt = gcm_temp + np.tile(gcm_temp_monthly_adj, int(gcm_temp.shape[1]/12))
        # Mean monthly temperature bias adjusted according to monthly average
        t_m25avg = np.tile(gcm_temp_monthly_avg + gcm_temp_monthly_adj, int(gcm_temp.shape[1]/12))
        # Calculate monthly standard deviation of temperature
        ref_temp_monthly_std = (ref_temp.reshape(-1,12).transpose().reshape(-1,int(ref_temp.shape[1]/12)).std(1)
                                .reshape(12,-1).transpose())
        gcm_temp_monthly_std = (gcm_temp_subset.reshape(-1,12).transpose().reshape(-1,int(ref_temp.shape[1]/12)).std(1)
                                .reshape(12,-1).transpose())
        variability_monthly_std = ref_temp_monthly_std / gcm_temp_monthly_std
        # Bias adjusted temperature accounting for monthly mean and variability
        gcm_temp_bias_adj = t_m25avg + (t_mt - t_m25avg) * np.tile(variability_monthly_std, int(gcm_temp.shape[1]/12))
        # PRECIPITATION BIAS CORRECTIONS
        # Calculate monthly mean precipitation
        ref_prec_monthly_avg = (ref_prec.reshape(-1,12).transpose().reshape(-1,int(ref_temp.shape[1]/12)).mean(1)
                                .reshape(12,-1).transpose())
        gcm_prec_monthly_avg = (gcm_prec_subset.reshape(-1,12).transpose().reshape(-1,int(ref_temp.shape[1]/12)).mean(1)
                                .reshape(12,-1).transpose())
        bias_adj_prec = ref_prec_monthly_avg / gcm_prec_monthly_avg
        # Bias adjusted precipitation accounting for differences in monthly mean
        gcm_prec_bias_adj = gcm_prec * np.tile(bias_adj_prec, int(gcm_temp.shape[1]/12))
        
        # MASS BALANCES FOR DATA COMPARISON
#        for glac in range(main_glac_rgi.shape[0]):
        for glac in [0]:
            # Glacier data
            modelparameters = main_glac_modelparams[glac,:]
            glacier_rgi_table = main_glac_rgi.loc[glac, :]
            glacier_gcm_elev = ref_elev[glac]
            glacier_gcm_prec = ref_prec[glac,:]
            glacier_gcm_temp = ref_temp[glac,:]
            glacier_gcm_lrgcm = ref_lr[glac,:]
            glacier_gcm_lrglac = glacier_gcm_lrgcm.copy()
            glacier_area_t0 = main_glac_hyps.iloc[glac,:].values.astype(float)   
            # Inclusion of ice thickness and width, i.e., loading values may be only required for Huss mass redistribution!
            icethickness_t0 = main_glac_icethickness.iloc[glac,:].values.astype(float)
            width_t0 = main_glac_width.iloc[glac,:].values.astype(float)
            
            # Mass balance for reference data
            (glac_bin_temp, glac_bin_prec, glac_bin_acc, glac_bin_refreeze, glac_bin_snowpack, glac_bin_melt, 
             glac_bin_frontalablation, glac_bin_massbalclim, glac_bin_massbalclim_annual, glac_bin_area_annual, 
             glac_bin_icethickness_annual, glac_bin_width_annual, glac_bin_surfacetype_annual, 
             glac_wide_massbaltotal, glac_wide_runoff, glac_wide_snowline, glac_wide_snowpack, 
             glac_wide_area_annual, glac_wide_volume_annual, glac_wide_ELA_annual) = (
                massbalance.runmassbalance(modelparameters, glacier_rgi_table, glacier_area_t0, icethickness_t0, 
                                           width_t0, elev_bins, glacier_gcm_temp, glacier_gcm_prec, 
                                           glacier_gcm_elev, glacier_gcm_lrgcm, glacier_gcm_lrglac, 
                                           dates_table_subset, biasadj_temp=0, biasadj_prec=1, 
                                           option_calibration=1))
            # Total volume loss
            glac_wide_massbaltotal_annual_ref = np.sum(glac_wide_massbaltotal.reshape(-1,12), axis=1)
            glac_wide_volume_loss_total_ref = (
                    np.cumsum(glac_wide_area_annual[glac_wide_massbaltotal_annual_ref.shape] * 
                              glac_wide_massbaltotal_annual_ref / 1000)[-1])
            
            # Mass balance for GCM data
            glacier_gcm_temp = gcm_temp_bias_adj[glac,0:ref_temp.shape[1]]
            glacier_gcm_prec = gcm_prec_bias_adj[glac,0:ref_temp.shape[1]]
            (glac_bin_temp, glac_bin_prec, glac_bin_acc, glac_bin_refreeze, glac_bin_snowpack, glac_bin_melt, 
             glac_bin_frontalablation, glac_bin_massbalclim, glac_bin_massbalclim_annual, glac_bin_area_annual, 
             glac_bin_icethickness_annual, glac_bin_width_annual, glac_bin_surfacetype_annual, 
             glac_wide_massbaltotal, glac_wide_runoff, glac_wide_snowline, glac_wide_snowpack, 
             glac_wide_area_annual, glac_wide_volume_annual, glac_wide_ELA_annual) = (
                massbalance.runmassbalance(modelparameters, glacier_rgi_table, glacier_area_t0, icethickness_t0, 
                                           width_t0, elev_bins, glacier_gcm_temp, glacier_gcm_prec, 
                                           glacier_gcm_elev, glacier_gcm_lrgcm, glacier_gcm_lrglac, 
                                           dates_table_subset, biasadj_temp=0, 
                                           biasadj_prec=1, option_calibration=1))
            # Total volume loss
            glac_wide_massbaltotal_annual_gcm = np.sum(glac_wide_massbaltotal.reshape(-1,12), axis=1)
            glac_wide_volume_loss_total_gcm = (
                    np.cumsum(glac_wide_area_annual[glac_wide_massbaltotal_annual_gcm.shape] * 
                              glac_wide_massbaltotal_annual_gcm / 1000)[-1])
            
        # PRINTING BIAS ADJUSTMENT OPTION 2
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
        

# OLD TEMP BIAS CORRECTIONS
##    elif option_bias_adjustment == 3:
##        # Reference - GCM difference
##        bias_adj_temp= (ref_temp - gcm_temp_subset).mean(axis=1)
##        # Bias adjusted temperature accounting for mean of entire time period
###        gcm_temp_bias_adj = gcm_temp + bias_adj_temp[:,np.newaxis]
##    elif option_bias_adjustment == 4:
##        # Calculate monthly mean temperature
##        ref_temp_monthly_avg = (ref_temp.reshape(-1,12).transpose().reshape(-1,int(ref_temp.shape[1]/12)).mean(1)
##                                .reshape(12,-1).transpose())
##        gcm_temp_monthly_avg = (gcm_temp_subset.reshape(-1,12).transpose().reshape(-1,int(ref_temp.shape[1]/12)).mean(1)
##                                .reshape(12,-1).transpose())
##        bias_adj_temp = ref_temp_monthly_avg - gcm_temp_monthly_avg
##        # Bias adjusted temperature accounting for monthly mean
###        gcm_temp_bias_adj = gcm_temp + np.tile(bias_adj_temp, int(gcm_temp.shape[1]/12))
    
        

#    # Export variables as global to view in variable explorer
#    global main_vars
#    main_vars = inspect.currentframe().f_locals
#    
#    print('\nProcessing time of',gcm_name,':',time.time()-time_start, 's')

#%% PARALLEL PROCESSING
#if __name__ == '__main__':
#    time_start = time.time()
#    parser = getparser()
#    args = parser.parse_args()
#    # Read GCM names from command file
#    with open(args.gcm_file, 'r') as gcm_fn:
#        gcm_list = gcm_fn.read().splitlines()
#    print('Found %d gcms to process'%(len(gcm_list)))
#    
#    if args.option_parallels != 0:
#        with multiprocessing.Pool(args.num_simultaneous_processes) as p:
#            p.map(main,gcm_list)
#    else:
#        # Loop through GCMs and export bias adjustments
#        for n_gcm in range(len(gcm_list)):
#            gcm_name = gcm_list[n_gcm]
#            # Perform GCM Bias adjustments
#            main(gcm_name)
#            
#            # Place local variables in variable explorer
#            vars_list = list(main_vars.keys())
#            gcm_name = main_vars['gcm_name']
#            rcp_scenario = main_vars['rcp_scenario']
#            main_glac_rgi = main_vars['main_glac_rgi']
#            main_glac_hyps = main_vars['main_glac_hyps']
#            main_glac_icethickness = main_vars['main_glac_icethickness']
#            main_glac_width = main_vars['main_glac_width']
#            elev_bins = main_vars['elev_bins']
#            main_glac_bias_adj_params = main_vars['main_glac_bias_adj_params']
#            main_glac_modelparams = main_vars['main_glac_modelparams']
#            end_date = main_vars['end_date']
#            start_date = main_vars['start_date']
#            dates_table = main_vars['dates_table']
#            
#    print('Total processing time:', time.time()-time_start, 's')            