r"""
run_gcmbiasadj_list_multiprocess.py outputs the adjustment parameters for temperature and precipitation as well as the
mean monthly lapse rates derived via comparisons with the calibration climate dataset.  These will be used to correct 
the GCM climate data for future simulations.  This may be done using parallels.

How to run file?
  - In command line:
      change directory to folder with script
      python run_gcmbiascorrections_list_multiprocess.py C:\Users\David\Dave_Rounce\HiMAT\Climate_data\cmip5\gcm_rcpXX_f
      ilenames.txt
  - In spyder:
      %run run_gcmbiascorrections_list_multiprocess.py C:\Users\David\Dave_Rounce\HiMAT\Climate_data\cmip5\gcm_rcpXX_fil
      enames.txt

Adjustment Options:
  Option 1 (default) - adjust the temperature such that the positive degree days [degC*day] is equal, then adjust the
             precipitation such that the accumulation is equal.  Use these adjustments as the initial condition, then 
             optimize them such that the mass balance between the reference and GCM is equal.  This ensures that mass 
             changes are due to the GCM itself and not the bias adjustments.
  Option 2 - Huss and Hock (2015): adjust mean monthly temperature and incorporate interannual variability, and
             adjust mean monthly precipitation
  Option 3 - adjust the mean monthly temperature and precipitation to be the same for both datasets 
             (Radic and Hock, 2011; Marzeion et al., 2012; etc.)

Why use Option 1 instead of Huss and Hock [2015]?
      The model estimates mass balance.  In an ideal setting, the MB for each GCM over the calibration period would be
      equal.  Option 1 ensures that this is the case, which ensures that any mass changes in the future are strictly due
      to the GCM and not a result of run-away effects due to the bias adjustments.
      Huss and Hock [2015] on the other hand make the mean temperature fairly consistent while trying to capture the 
      interannual variability, but on average this causes the mass balance to vary by a mean of -0.03 mwea for region
      15; hence, this automatically biases the future simulations.
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

import pygem_input as input
import pygemfxns_modelsetup as modelsetup
import pygemfxns_massbalance as massbalance
import class_climate

#%% ===== SCRIPT SPECIFIC INPUT DATA ===== 
# Glacier selection
rgi_regionsO1 = [15]
rgi_glac_number = 'all'
#rgi_glac_number = ['03473', '03733']
#rgi_glac_number = ['03473']
#rgi_glac_number = ['06881']
#rgi_glac_number = ['00001', '00002', '00003', '00004', '00005', '00006', '00007', '00008', '03473', '03733']

# Required input
option_bias_adjustment = 1
# Time period
gcm_startyear = 2000
gcm_endyear = 2015
gcm_spinupyears = 5
# Reference climate data
ref_name = 'ERA-Interim'
filepath_ref = input.main_directory + '/../Climate_data/ERA_Interim/' 
filename_ref_temp = input.gcmtemp_filedict[rgi_regionsO1[0]]
filename_ref_prec = input.gcmprec_filedict[rgi_regionsO1[0]]
filename_ref_elev = input.gcmelev_filedict[rgi_regionsO1[0]]
filename_ref_lr = input.gcmlapserate_filedict[rgi_regionsO1[0]]
# Calibrated model parameters
#  calibrated parameters are the same for all climate datasets (only bias adjustments differ for each climate dataset)
filepath_modelparams = input.main_directory + '/../Calibration_datasets/'
filename_modelparams = 'calibration_R15_20180403_Opt02solutionspaceexpanding_wnnbrs_20180523.csv'
modelparams_colnames = ['lrgcm', 'lrglac', 'precfactor', 'precgrad', 'ddfsnow', 'ddfice', 'tempsnow', 'tempchange']
# Output
output_filepath = input.main_directory + '/../Climate_data/cmip5/bias_adjusted_1995_2100/'
option_export = 1
option_run_mb = 1 # only for options 2 and 3


#%% FUNCTIONS
def getparser():
    parser = argparse.ArgumentParser(description="run gcm bias corrections from gcm list in parallel")
    # add arguments
    parser.add_argument('gcm_file', action='store', type=str, default='gcm_rcpXX_filenames.txt', 
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
    # Model parameters
    main_glac_modelparams_all = pd.read_csv(filepath_modelparams + filename_modelparams, index_col=0)
    main_glac_modelparams = main_glac_modelparams_all.loc[main_glac_rgi['O1Index'].values, :] 
    # Select dates including future projections
    dates_table, start_date, end_date = modelsetup.datesmodelrun(startyear=gcm_startyear, endyear=gcm_endyear, 
                                                                 spinupyears=gcm_spinupyears)
    
    # ===== LOAD CLIMATE DATA =====
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
    gcm = class_climate.GCM(name=gcm_name, rcp_scenario=rcp_scenario)
    gcm_temp, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(gcm.temp_fn, gcm.temp_vn, main_glac_rgi, dates_table)
    gcm_prec, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(gcm.prec_fn, gcm.prec_vn, main_glac_rgi, dates_table)
    gcm_elev = gcm.importGCMfxnearestneighbor_xarray(gcm.elev_fn, gcm.elev_vn, main_glac_rgi)    
    gcm_lr = np.tile(ref_lr_monthly_avg, int(gcm_temp.shape[1]/12))
    
    # GCM subset to agree with reference time period to calculate bias corrections
    gcm_temp_subset = gcm_temp[:,0:ref_temp.shape[1]]
    gcm_prec_subset = gcm_prec[:,0:ref_temp.shape[1]]
    gcm_lr_subset = gcm_lr[:,0:ref_temp.shape[1]]

    #%% ===== BIAS CORRECTIONS =====
    # OPTION 1: Adjust temp and prec such that ref and GCM mass balances over calibration period are equal
    if option_bias_adjustment == 1:
        # Bias adjustment parameters
        main_glac_bias_adj_colnames = ['RGIId', 'ref', 'GCM', 'rcp_scenario', 'temp_adj', 'prec_adj', 'ref_mb_mwea', 
                                       'ref_vol_change_perc', 'gcm_mb_mwea', 'gcm_vol_change_perc', 'lrgcm', 'lrglac', 
                                       'precfactor', 'precgrad', 'ddfsnow', 'ddfice', 'tempsnow', 'tempchange']
        main_glac_bias_adj = pd.DataFrame(np.zeros((main_glac_rgi.shape[0],len(main_glac_bias_adj_colnames))), 
                                          columns=main_glac_bias_adj_colnames)
        main_glac_bias_adj['RGIId'] = main_glac_rgi['RGIId'].values
        main_glac_bias_adj['ref'] = ref_name
        main_glac_bias_adj['GCM'] = gcm_name
        main_glac_bias_adj['rcp_scenario'] = rcp_scenario
        main_glac_bias_adj[modelparams_colnames] = main_glac_modelparams[modelparams_colnames].values

        # BIAS ADJUSTMENT CALCULATIONS
        for glac in range(main_glac_rgi.shape[0]): 
            if glac%200 == 0:
                print(gcm_name,':', main_glac_rgi.loc[main_glac_rgi.index.values[glac],'RGIId'])    
            glacier_rgi_table = main_glac_rgi.iloc[glac, :]
            glacier_area_t0 = main_glac_hyps.iloc[glac,:].values.astype(float)
            icethickness_t0 = main_glac_icethickness.iloc[glac,:].values.astype(float)
            width_t0 = main_glac_width.iloc[glac,:].values.astype(float)
            modelparameters = main_glac_modelparams.loc[main_glac_modelparams.index.values[glac],modelparams_colnames]
            glac_idx_t0 = glacier_area_t0.nonzero()[0]
            
            if icethickness_t0.max() > 0:  
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
                            glacier_rgi_table.loc[input.option_elev_ref_downscale])[:,np.newaxis] 
                            + modelparameters['tempchange'])
                    glac_bin_temp_gcm = (
                            glacier_gcm_temp + glacier_gcm_lrgcm * 
                            (glacier_rgi_table.loc[input.option_elev_ref_downscale] - glacier_gcm_elev) + 
                            glacier_gcm_lrglac * (elev_bins - 
                            glacier_rgi_table.loc[input.option_elev_ref_downscale])[:,np.newaxis] 
                            + modelparameters['tempchange'])
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
                    melt_energy_available_gcm = glac_bin_temp_gcm_adj * daysinmonth
                    melt_energy_available_gcm[melt_energy_available_gcm < 0] = 0
                    melt_gcm = melt_energy_available_gcm * surfacetype_ddf[:,np.newaxis]
                    melt_vol_gcm = (melt_gcm * glacier_area_t0[:,np.newaxis]).sum()
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
                    glac_bin_precsnow_ref = (glacier_ref_prec * modelparameters['precfactor'] * 
                                             (1 + modelparameters['precgrad'] * (elev_bins - 
                                             glacier_rgi_table.loc[input.option_elev_ref_downscale]))[:,np.newaxis])
                    glac_bin_precsnow_gcm = (glacier_gcm_prec * modelparameters['precfactor'] * 
                                             (1 + modelparameters['precgrad'] * (elev_bins - 
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
                    glac_bin_prec_ref[glac_bin_temp_ref > modelparameters['tempsnow']] = (
                        glac_bin_precsnow_ref[glac_bin_temp_ref > modelparameters['tempsnow']])
                    glac_bin_prec_gcm[glac_bin_temp_gcm_adj > modelparameters['tempsnow']] = (
                        glac_bin_precsnow_gcm[glac_bin_temp_gcm_adj > modelparameters['tempsnow']])
                    # if temperature below threshold, then snow
                    glac_bin_acc_ref[glac_bin_temp_ref <= modelparameters['tempsnow']] = (
                        glac_bin_precsnow_ref[glac_bin_temp_ref <= modelparameters['tempsnow']])
                    glac_bin_acc_gcm[glac_bin_temp_gcm_adj <= modelparameters['tempsnow']] = (
                        glac_bin_precsnow_gcm[glac_bin_temp_gcm_adj <= modelparameters['tempsnow']])
                elif input.option_accumulation == 2:
                    # If temperature between min/max, then mix of snow/rain using linear relationship between min/max
                    glac_bin_prec_ref = (
                            (1/2 + (glac_bin_temp_ref - modelparameters['tempsnow']) / 2) * glac_bin_precsnow_ref)
                    glac_bin_prec_gcm = (
                            (1/2 + (glac_bin_temp_gcm_adj - modelparameters['tempsnow']) / 2) * glac_bin_precsnow_gcm)
                    glac_bin_acc_ref = glac_bin_precsnow_ref - glac_bin_prec_ref
                    glac_bin_acc_gcm = glac_bin_precsnow_gcm - glac_bin_prec_gcm
                    # If temperature above maximum threshold, then all rain
                    glac_bin_prec_ref[glac_bin_temp_ref > modelparameters['tempsnow'] + 1] = (
                        glac_bin_precsnow_ref[glac_bin_temp_ref > modelparameters['tempsnow'] + 1])
                    glac_bin_prec_gcm[glac_bin_temp_gcm_adj > modelparameters['tempsnow'] + 1] = (
                        glac_bin_precsnow_gcm[glac_bin_temp_gcm_adj > modelparameters['tempsnow'] + 1])
                    glac_bin_acc_ref[glac_bin_temp_ref > modelparameters['tempsnow'] + 1] = 0
                    glac_bin_acc_gcm[glac_bin_temp_gcm_adj > modelparameters['tempsnow'] + 1] = 0
                    # If temperature below minimum threshold, then all snow
                    glac_bin_acc_ref[glac_bin_temp_ref <= modelparameters['tempsnow'] - 1] = (
                            glac_bin_precsnow_ref[glac_bin_temp_ref <= modelparameters['tempsnow'] - 1])
                    glac_bin_acc_gcm[glac_bin_temp_gcm_adj <= modelparameters['tempsnow'] - 1] = (
                            glac_bin_precsnow_gcm[glac_bin_temp_gcm_adj <= modelparameters['tempsnow'] - 1])
                    glac_bin_prec_ref[glac_bin_temp_ref <= modelparameters['tempsnow'] - 1] = 0
                    glac_bin_prec_gcm[glac_bin_temp_gcm_adj <= modelparameters['tempsnow'] - 1] = 0
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
                    # Reference data
                    # Mass balance
                    (glac_bin_temp, glac_bin_prec, glac_bin_acc, glac_bin_refreeze, glac_bin_snowpack, glac_bin_melt, 
                     glac_bin_frontalablation, glac_bin_massbalclim, glac_bin_massbalclim_annual, glac_bin_area_annual, 
                     glac_bin_icethickness_annual, glac_bin_width_annual, glac_bin_surfacetype_annual, 
                     glac_wide_massbaltotal, glac_wide_runoff, glac_wide_snowline, glac_wide_snowpack, 
                     glac_wide_area_annual, glac_wide_volume_annual, glac_wide_ELA_annual) = (
                        massbalance.runmassbalance(modelparameters, glacier_rgi_table, glacier_area_t0, icethickness_t0, 
                                                   width_t0, elev_bins, glacier_ref_temp, glacier_ref_prec, 
                                                   glacier_ref_elev, glacier_ref_lrgcm, glacier_ref_lrglac, 
                                                   dates_table_subset, option_areaconstant=1))
                    # Annual glacier-wide mass balance [m w.e.]
                    glac_wide_massbaltotal_annual_ref = np.sum(glac_wide_massbaltotal.reshape(-1,12), axis=1)
                    # Average annual glacier-wide mass balance [m w.e.a.]
                    mb_mwea_ref = glac_wide_massbaltotal_annual_ref.mean()
                    
                    # GCM data
                    # Bias corrections
                    glacier_gcm_temp_adj = glacier_gcm_temp + bias_adj_params[0]
                    glacier_gcm_prec_adj = glacier_gcm_prec * bias_adj_params[1]
                    
                    # Mass balance
                    (glac_bin_temp, glac_bin_prec, glac_bin_acc, glac_bin_refreeze, glac_bin_snowpack, glac_bin_melt, 
                     glac_bin_frontalablation, glac_bin_massbalclim, glac_bin_massbalclim_annual, glac_bin_area_annual, 
                     glac_bin_icethickness_annual, glac_bin_width_annual, glac_bin_surfacetype_annual, 
                     glac_wide_massbaltotal, glac_wide_runoff, glac_wide_snowline, glac_wide_snowpack, 
                     glac_wide_area_annual, glac_wide_volume_annual, glac_wide_ELA_annual) = (
                        massbalance.runmassbalance(modelparameters, glacier_rgi_table, glacier_area_t0, icethickness_t0, 
                                                   width_t0, elev_bins, glacier_gcm_temp_adj, glacier_gcm_prec_adj, 
                                                   glacier_gcm_elev, glacier_gcm_lrgcm, glacier_gcm_lrglac, 
                                                   dates_table_subset, option_areaconstant=1))
                    # Annual glacier-wide mass balance [m w.e.]
                    glac_wide_massbaltotal_annual_gcm = np.sum(glac_wide_massbaltotal.reshape(-1,12), axis=1)
                    # Average annual glacier-wide mass balance [m w.e.a.]
                    mb_mwea_gcm = glac_wide_massbaltotal_annual_gcm.mean()
                    return abs(mb_mwea_ref - mb_mwea_gcm)
                # CONSTRAINTS
                #  everything goes on one side of the equation compared to zero
                def constraint_temp_prec(bias_adj_params):
                    return -1 * (bias_adj_params[0] * (bias_adj_params[1] - 1))
                    #  To avoid increases/decreases in temp compensating for increases/decreases in prec, respectively,
                    #  ensure that if temp increases, then prec decreases, and vice versa.  This works because
                    #  (prec_adj - 1) is positive or negative for increases or decrease, respectively, so multiplying 
                    #  this by temp_adj gives a positive or negative value.  We want it to always be negative, but since 
                    #  inequality constraint is for >= 0, we multiply it by -1.
                # Define constraint type for each function
                con_temp_prec = {'type':'ineq', 'fun':constraint_temp_prec}
                #  inequalities are non-negative, i.e., >= 0
                # Select constraints used to optimize precfactor
                cons = [con_temp_prec]
                # INITIAL GUESS
                bias_adj_params_init = bias_adj_params          
                # Run the optimization
                bias_adj_params_opt_raw = minimize(objective_2, bias_adj_params_init, method='SLSQP', constraints=cons,
                                                   tol=1e-3)
                # Record the optimized parameters
                bias_adj_params_opt = bias_adj_params_opt_raw.x
                main_glac_bias_adj.loc[glac, ['temp_adj', 'prec_adj']] = bias_adj_params_opt 
            
                # Compute mass balances to have output data
                # Reference data
                (glac_bin_temp, glac_bin_prec, glac_bin_acc, glac_bin_refreeze, glac_bin_snowpack, glac_bin_melt, 
                 glac_bin_frontalablation, glac_bin_massbalclim, glac_bin_massbalclim_annual, glac_bin_area_annual, 
                 glac_bin_icethickness_annual, glac_bin_width_annual, glac_bin_surfacetype_annual, 
                 glac_wide_massbaltotal, glac_wide_runoff, glac_wide_snowline, glac_wide_snowpack, 
                 glac_wide_area_annual, glac_wide_volume_annual, glac_wide_ELA_annual) = (
                    massbalance.runmassbalance(modelparameters, glacier_rgi_table, glacier_area_t0, icethickness_t0, 
                                               width_t0, elev_bins, glacier_ref_temp, glacier_ref_prec, 
                                               glacier_ref_elev, glacier_ref_lrgcm, glacier_ref_lrglac, 
                                               dates_table_subset, option_areaconstant=1))
                # Annual glacier-wide mass balance [m w.e.]
                glac_wide_massbaltotal_annual_ref = np.sum(glac_wide_massbaltotal.reshape(-1,12), axis=1)
                # Average annual glacier-wide mass balance [m w.e.a.]
                mb_mwea_ref = glac_wide_massbaltotal_annual_ref.mean()
                #  units: m w.e. based on initial area
                # Volume change [%]
                if icethickness_t0.max() > 0:
                    glac_vol_change_perc_ref = (mb_mwea_ref / 1000 * glac_wide_area_annual[0] * 
                                                glac_wide_massbaltotal_annual_ref.shape[0] / glac_wide_volume_annual[0] 
                                                * 100)
                # Record reference results
                main_glac_bias_adj.loc[glac, ['ref_mb_mwea', 'ref_vol_change_perc']] = (
                        [mb_mwea_ref, glac_vol_change_perc_ref])
                
                # Climate data
                # Bias corrections
                glacier_gcm_temp_adj = glacier_gcm_temp + bias_adj_params_opt[0]
                glacier_gcm_prec_adj = glacier_gcm_prec * bias_adj_params_opt[1]
                (glac_bin_temp, glac_bin_prec, glac_bin_acc, glac_bin_refreeze, glac_bin_snowpack, glac_bin_melt, 
                 glac_bin_frontalablation, glac_bin_massbalclim, glac_bin_massbalclim_annual, glac_bin_area_annual, 
                 glac_bin_icethickness_annual, glac_bin_width_annual, glac_bin_surfacetype_annual, 
                 glac_wide_massbaltotal, glac_wide_runoff, glac_wide_snowline, glac_wide_snowpack, 
                 glac_wide_area_annual, glac_wide_volume_annual, glac_wide_ELA_annual) = (
                    massbalance.runmassbalance(modelparameters, glacier_rgi_table, glacier_area_t0, icethickness_t0, 
                                               width_t0, elev_bins, glacier_gcm_temp_adj, glacier_gcm_prec_adj, 
                                               glacier_gcm_elev, glacier_gcm_lrgcm, glacier_gcm_lrglac, 
                                               dates_table_subset, option_areaconstant=1))
                # Annual glacier-wide mass balance [m w.e.]
                glac_wide_massbaltotal_annual_gcm = np.sum(glac_wide_massbaltotal.reshape(-1,12), axis=1)
                # Average annual glacier-wide mass balance [m w.e.a.]
                mb_mwea_gcm = glac_wide_massbaltotal_annual_gcm.mean()
                #  units: m w.e. based on initial area
                # Volume change [%]
                if icethickness_t0.max() > 0:
                    glac_vol_change_perc_gcm = (mb_mwea_gcm / 1000 * glac_wide_area_annual[0] * 
                                                glac_wide_massbaltotal_annual_gcm.shape[0] / glac_wide_volume_annual[0] 
                                                * 100) 
                # Record GCM results
                main_glac_bias_adj.loc[glac, ['gcm_mb_mwea', 'gcm_vol_change_perc']] = (
                        [mb_mwea_gcm, glac_vol_change_perc_gcm])
                
      
    #%% OPTION 2: Adjust temp and prec according to Huss and Hock (2015) accounts for means and interannual variability
    elif option_bias_adjustment == 2:
        # Bias adjustment parameters
        main_glac_bias_adj_colnames = ['RGIId', 'ref', 'GCM', 'rcp_scenario', 'ref_mb_mwea', 'ref_vol_change_perc', 
                                       'gcm_mb_mwea', 'gcm_vol_change_perc', 'new_gcmelev', 'lrgcm', 'lrglac', 
                                       'precfactor', 'precgrad', 'ddfsnow', 'ddfice', 'tempsnow', 'tempchange']
        main_glac_bias_adj = pd.DataFrame(np.zeros((main_glac_rgi.shape[0],len(main_glac_bias_adj_colnames))), 
                                          columns=main_glac_bias_adj_colnames)
        main_glac_bias_adj['RGIId'] = main_glac_rgi['RGIId'].values
        main_glac_bias_adj['ref'] = ref_name
        main_glac_bias_adj['GCM'] = gcm_name
        main_glac_bias_adj['rcp_scenario'] = rcp_scenario
        main_glac_bias_adj['new_gcmelev'] = ref_elev
        main_glac_bias_adj[modelparams_colnames] = main_glac_modelparams[modelparams_colnames].values
        
        tempvar_cols = []
        tempavg_cols = []
        tempadj_cols = []
        precadj_cols = []
        # Monthly temperature variability
        for n in range(1,13):
            tempvar_colname = 'tempvar_' + str(n)
            main_glac_bias_adj[tempvar_colname] = np.nan
            tempvar_cols.append(tempvar_colname)
        # Monthly mean temperature
        for n in range(1,13):
            tempavg_colname = 'tempavg_' + str(n)
            main_glac_bias_adj[tempavg_colname] = np.nan
            tempavg_cols.append(tempavg_colname)
        # Monthly temperature adjustment
        for n in range(1,13):
            tempadj_colname = 'tempadj_' + str(n)
            main_glac_bias_adj[tempadj_colname] = np.nan
            tempadj_cols.append(tempadj_colname)
        # Monthly precipitation adjustment
        for n in range(1,13):
            precadj_colname = 'precadj_' + str(n)
            main_glac_bias_adj[precadj_colname] = np.nan
            precadj_cols.append(precadj_colname)
        
        # Remove spinup years, so adjustment performed over calibration period
        ref_temp_nospinup = ref_temp[:,gcm_spinupyears*12:]
        gcm_temp_nospinup = gcm_temp_subset[:,gcm_spinupyears*12:]
        ref_prec_nospinup = ref_prec[:,gcm_spinupyears*12:]
        gcm_prec_nospinup = gcm_prec_subset[:,gcm_spinupyears*12:]
        # TEMPERATURE BIAS CORRECTIONS
        # Mean monthly temperature
        ref_temp_monthly_avg = (ref_temp_nospinup.reshape(-1,12).transpose()
                                .reshape(-1,int(ref_temp_nospinup.shape[1]/12)).mean(1).reshape(12,-1).transpose())
        gcm_temp_monthly_avg = (gcm_temp_nospinup.reshape(-1,12).transpose()
                                .reshape(-1,int(gcm_temp_nospinup.shape[1]/12)).mean(1).reshape(12,-1).transpose())
        # Monthly bias adjustment
        gcm_temp_monthly_adj = ref_temp_monthly_avg - gcm_temp_monthly_avg
        # Monthly temperature bias adjusted according to monthly average
        t_mt = gcm_temp + np.tile(gcm_temp_monthly_adj, int(gcm_temp.shape[1]/12))
        # Mean monthly temperature bias adjusted according to monthly average
        t_m25avg = np.tile(gcm_temp_monthly_avg + gcm_temp_monthly_adj, int(gcm_temp.shape[1]/12))
        # Calculate monthly standard deviation of temperature
        ref_temp_monthly_std = (ref_temp_nospinup.reshape(-1,12).transpose()
                                .reshape(-1,int(ref_temp_nospinup.shape[1]/12)).std(1).reshape(12,-1).transpose())
        gcm_temp_monthly_std = (gcm_temp_nospinup.reshape(-1,12).transpose()
                                .reshape(-1,int(gcm_temp_nospinup.shape[1]/12)).std(1).reshape(12,-1).transpose())
        variability_monthly_std = ref_temp_monthly_std / gcm_temp_monthly_std
        # Bias adjusted temperature accounting for monthly mean and variability
        gcm_temp_bias_adj = t_m25avg + (t_mt - t_m25avg) * np.tile(variability_monthly_std, int(gcm_temp.shape[1]/12))
        # PRECIPITATION BIAS CORRECTIONS
        # Calculate monthly mean precipitation
        ref_prec_monthly_avg = (ref_prec_nospinup.reshape(-1,12).transpose()
                                .reshape(-1,int(ref_temp_nospinup.shape[1]/12)).mean(1).reshape(12,-1).transpose())
        gcm_prec_monthly_avg = (gcm_prec_nospinup.reshape(-1,12).transpose()
                                .reshape(-1,int(gcm_temp_nospinup.shape[1]/12)).mean(1).reshape(12,-1).transpose())
        bias_adj_prec = ref_prec_monthly_avg / gcm_prec_monthly_avg
        # Bias adjusted precipitation accounting for differences in monthly mean
        gcm_prec_bias_adj = gcm_prec * np.tile(bias_adj_prec, int(gcm_temp.shape[1]/12))
        
        # Record adjustment parameters
        main_glac_bias_adj[precadj_cols] = bias_adj_prec
        main_glac_bias_adj[tempvar_cols] = variability_monthly_std
        main_glac_bias_adj[tempavg_cols] = gcm_temp_monthly_avg
        main_glac_bias_adj[tempadj_cols] = gcm_temp_monthly_adj
        
    # OPTION 3: Adjust temp and prec such mean monthly temp and mean annual precipitation are equal
    elif option_bias_adjustment == 3:
        # Bias adjustment parameters
        main_glac_bias_adj_colnames = ['RGIId', 'ref', 'GCM', 'rcp_scenario', 'ref_mb_mwea', 'ref_vol_change_perc', 
                                       'gcm_mb_mwea', 'gcm_vol_change_perc', 'new_gcmelev', 'lrgcm', 'lrglac', 
                                       'precfactor', 'precgrad', 'ddfsnow', 'ddfice', 'tempsnow', 'tempchange']
        main_glac_bias_adj = pd.DataFrame(np.zeros((main_glac_rgi.shape[0],len(main_glac_bias_adj_colnames))), 
                                          columns=main_glac_bias_adj_colnames)
        main_glac_bias_adj['RGIId'] = main_glac_rgi['RGIId'].values
        main_glac_bias_adj['ref'] = ref_name
        main_glac_bias_adj['GCM'] = gcm_name
        main_glac_bias_adj['rcp_scenario'] = rcp_scenario
        main_glac_bias_adj['new_gcmelev'] = ref_elev
        main_glac_bias_adj[modelparams_colnames] = main_glac_modelparams[modelparams_colnames].values
        
        tempadj_cols = []
        precadj_cols = []
        # Monthly temperature adjustment
        for n in range(1,13):
            tempadj_colname = 'tempadj_' + str(n)
            main_glac_bias_adj[tempadj_colname] = np.nan
            tempadj_cols.append(tempadj_colname)
        # Monthly precipitation adjustment
        for n in range(1,13):
            precadj_colname = 'precadj_' + str(n)
            main_glac_bias_adj[precadj_colname] = np.nan
            precadj_cols.append(precadj_colname)
        
        # Remove spinup years, so adjustment performed over calibration period
        ref_temp_nospinup = ref_temp[:,gcm_spinupyears*12:]
        gcm_temp_nospinup = gcm_temp_subset[:,gcm_spinupyears*12:]
        ref_prec_nospinup = ref_prec[:,gcm_spinupyears*12:]
        gcm_prec_nospinup = gcm_prec_subset[:,gcm_spinupyears*12:]
        # TEMPERATURE BIAS CORRECTIONS
        # Mean monthly temperature
        ref_temp_monthly_avg = (ref_temp_nospinup.reshape(-1,12).transpose()
                                .reshape(-1,int(ref_temp_nospinup.shape[1]/12)).mean(1).reshape(12,-1).transpose())
        gcm_temp_monthly_avg = (gcm_temp_nospinup.reshape(-1,12).transpose()
                                .reshape(-1,int(gcm_temp_nospinup.shape[1]/12)).mean(1).reshape(12,-1).transpose())
        # Monthly bias adjustment
        bias_adj_temp = ref_temp_monthly_avg - gcm_temp_monthly_avg
        # Bias adjusted temperature accounting for monthly mean
        gcm_temp_bias_adj = gcm_temp + np.tile(bias_adj_temp, int(gcm_temp.shape[1]/12))
        # PRECIPITATION BIAS CORRECTIONS
        # Calculate monthly mean precipitation
        ref_prec_monthly_avg = (ref_prec_nospinup.reshape(-1,12).transpose()
                                .reshape(-1,int(ref_temp_nospinup.shape[1]/12)).mean(1).reshape(12,-1).transpose())
        gcm_prec_monthly_avg = (gcm_prec_nospinup.reshape(-1,12).transpose()
                                .reshape(-1,int(gcm_temp_nospinup.shape[1]/12)).mean(1).reshape(12,-1).transpose())
        bias_adj_prec = ref_prec_monthly_avg / gcm_prec_monthly_avg
        # Bias adjusted precipitation accounting for differences in monthly mean
        gcm_prec_bias_adj = gcm_prec * np.tile(bias_adj_prec, int(gcm_temp.shape[1]/12))
        
        # Record adjustment parameters
        main_glac_bias_adj[precadj_cols] = bias_adj_prec
        main_glac_bias_adj[tempadj_cols] = bias_adj_temp
        
    # MASS BALANCE: compute for model comparisons
    if ((option_bias_adjustment == 2) or (option_bias_adjustment == 3)) and (option_run_mb == 1):
        # Compute mass balances to have output data        
        for glac in range(main_glac_rgi.shape[0]): 
            if glac%500 == 0:
                print(gcm_name,':', main_glac_rgi.loc[main_glac_rgi.index.values[glac],'RGIId'])
            glacier_rgi_table = main_glac_rgi.iloc[glac, :]
            glacier_area_t0 = main_glac_hyps.iloc[glac,:].values.astype(float)
            icethickness_t0 = main_glac_icethickness.iloc[glac,:].values.astype(float)
            width_t0 = main_glac_width.iloc[glac,:].values.astype(float)
            modelparameters = main_glac_modelparams.loc[main_glac_modelparams.index.values[glac],modelparams_colnames]
            glac_idx_t0 = glacier_area_t0.nonzero()[0]
            
            # Reference data
            glacier_ref_temp = ref_temp[glac,:]
            glacier_ref_prec = ref_prec[glac,:]
            glacier_ref_elev = ref_elev[glac]
            glacier_ref_lrgcm = ref_lr[glac,:]
            glacier_ref_lrglac = ref_lr[glac,:]
            # GCM data
            glacier_gcm_temp_adj = gcm_temp_bias_adj[glac,:]
            glacier_gcm_prec_adj = gcm_prec_bias_adj[glac,:]
            glacier_gcm_elev = ref_elev[glac]
            #  using the REFERENCE elev here because the adjusted temp is corrected for the mean ref temp already
            glacier_gcm_lrgcm = gcm_lr_subset[glac,:]
            glacier_gcm_lrglac = gcm_lr_subset[glac,:]
            
            # Mass balance
            # Reference data
            (glac_bin_temp, glac_bin_prec, glac_bin_acc, glac_bin_refreeze, glac_bin_snowpack, glac_bin_melt, 
             glac_bin_frontalablation, glac_bin_massbalclim, glac_bin_massbalclim_annual, glac_bin_area_annual, 
             glac_bin_icethickness_annual, glac_bin_width_annual, glac_bin_surfacetype_annual, 
             glac_wide_massbaltotal, glac_wide_runoff, glac_wide_snowline, glac_wide_snowpack, 
             glac_wide_area_annual, glac_wide_volume_annual, glac_wide_ELA_annual) = (
                massbalance.runmassbalance(modelparameters, glacier_rgi_table, glacier_area_t0, icethickness_t0, 
                                           width_t0, elev_bins, glacier_ref_temp, glacier_ref_prec, 
                                           glacier_ref_elev, glacier_ref_lrgcm, glacier_ref_lrglac, 
                                           dates_table_subset, option_areaconstant=1))
            # Annual glacier-wide mass balance [m w.e.]
            glac_wide_massbaltotal_annual_ref = np.sum(glac_wide_massbaltotal.reshape(-1,12), axis=1)
            # Average annual glacier-wide mass balance [m w.e.a.]
            mb_mwea_ref = glac_wide_massbaltotal_annual_ref.mean()
            #  units: m w.e. based on initial area
            # Volume change [%]
            if icethickness_t0.max() > 0:
                glac_vol_change_perc_ref = (mb_mwea_ref / 1000 * glac_wide_area_annual[0] * 
                                            glac_wide_massbaltotal_annual_ref.shape[0] / glac_wide_volume_annual[0] 
                                            * 100)
            main_glac_bias_adj.loc[glac,'ref_mb_mwea'] = mb_mwea_ref
            main_glac_bias_adj.loc[glac,'ref_vol_change_perc'] = glac_vol_change_perc_ref
        
            # GCM data
            (glac_bin_temp, glac_bin_prec, glac_bin_acc, glac_bin_refreeze, glac_bin_snowpack, glac_bin_melt, 
             glac_bin_frontalablation, glac_bin_massbalclim, glac_bin_massbalclim_annual, glac_bin_area_annual, 
             glac_bin_icethickness_annual, glac_bin_width_annual, glac_bin_surfacetype_annual, 
             glac_wide_massbaltotal, glac_wide_runoff, glac_wide_snowline, glac_wide_snowpack, 
             glac_wide_area_annual, glac_wide_volume_annual, glac_wide_ELA_annual) = (
                massbalance.runmassbalance(modelparameters, glacier_rgi_table, glacier_area_t0, icethickness_t0, 
                                           width_t0, elev_bins, glacier_gcm_temp_adj, glacier_gcm_prec_adj, 
                                           glacier_gcm_elev, glacier_gcm_lrgcm, glacier_gcm_lrglac, 
                                           dates_table_subset, option_areaconstant=1))
            # Annual glacier-wide mass balance [m w.e.]
            glac_wide_massbaltotal_annual_gcm = np.sum(glac_wide_massbaltotal.reshape(-1,12), axis=1)
            # Average annual glacier-wide mass balance [m w.e.a.]
            mb_mwea_gcm = glac_wide_massbaltotal_annual_gcm.mean()
            #  units: m w.e. based on initial area
            # Volume change [%]
            if icethickness_t0.max() > 0:
                glac_vol_change_perc_gcm = (mb_mwea_gcm / 1000 * glac_wide_area_annual[0] * 
                                            glac_wide_massbaltotal_annual_ref.shape[0] / glac_wide_volume_annual[0] 
                                            * 100)
            
            main_glac_bias_adj.loc[glac,'gcm_mb_mwea'] = mb_mwea_gcm
            main_glac_bias_adj.loc[glac,'gcm_vol_change_perc'] = glac_vol_change_perc_gcm
        
            
    #%% EXPORT THE ADJUSTMENT VARIABLES (greatly reduces space)
    if option_export == 1:
        # Set up directory to store climate data
        if os.path.exists(output_filepath) == False:
            os.makedirs(output_filepath)
        # Temperature and precipitation parameters
        output_biasadjparams_fn = (gcm_name + '_' + rcp_scenario + '_biasadj_opt' + str(option_bias_adjustment) + '_' + 
                                   str(gcm_startyear - gcm_spinupyears) + '_' + str(gcm_endyear) + '_' +  'R' + 
                                   str(rgi_regionsO1[0]) + '_' + str(count) + '.csv')
        main_glac_bias_adj.to_csv(output_filepath + output_biasadjparams_fn)
        # Lapse rate parameters (same for all GCMs - only need to export once)
        output_filename_lr = ('biasadj_mon_lravg_' + str(gcm_startyear - gcm_spinupyears) + '_' + str(gcm_endyear) +
                              '_' + 'R' + str(rgi_regionsO1[0]) + '_' + str(count) + '.csv')
        if os.path.exists(output_filepath + output_filename_lr) == False:
            np.savetxt(output_filepath + output_filename_lr, ref_lr_monthly_avg, delimiter=",")


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
    if (args.option_parallels != 0) and (main_glac_rgi_all.shape[0] >= 2 * args.num_simultaneous_processes):
        chunk_size = int(np.ceil(main_glac_rgi_all.shape[0] / args.num_simultaneous_processes))
    else:
        chunk_size = main_glac_rgi_all.shape[0]
    
    # Read GCM names from command file
    with open(args.gcm_file, 'r') as gcm_fn:
        gcm_list = gcm_fn.read().splitlines()
        rcp_scenario = os.path.basename(args.gcm_file).split('_')[1]
        print('Found %d gcms to process'%(len(gcm_list)))
        
    # Loop through all GCMs
    for gcm_name in gcm_list:
        # Pack variables for multiprocessing
        list_packed_vars = [] 
        n = 0
        for chunk in range(0, main_glac_rgi_all.shape[0], chunk_size):
            n = n + 1
            list_packed_vars.append([n, chunk, main_glac_rgi_all, chunk_size, gcm_name])
        
        # Parallel processing
        if (args.option_parallels != 0) and (main_glac_rgi_all.shape[0] >= 2 * args.num_simultaneous_processes):
            print('Processing', gcm_name, 'in parallel')
            with multiprocessing.Pool(args.num_simultaneous_processes) as p:
                p.map(main,list_packed_vars)
        
        # No parallel processing
        else:
            print('Processing', gcm_name, 'without parallel')
            # Loop through the chunks and export bias adjustments
            for n in range(len(list_packed_vars)):
                main(list_packed_vars[n])
                                                
         
        # Combine output into single package and export lapse rate if necessary
        if ((args.option_parallels != 0) and (main_glac_rgi_all.shape[0] >= 2 * args.num_simultaneous_processes) and
            (option_export == 1)):
            # Bias adjustment parameters
            output_biasadj_prefix = (gcm_name + '_' + rcp_scenario + '_biasadj_opt' + str(option_bias_adjustment) + '_' 
                                     + str(gcm_startyear - gcm_spinupyears) + '_' + str(gcm_endyear) + '_' +  'R' + 
                                     str(rgi_regionsO1[0]))
            biasadj_list = []
            for i in os.listdir(output_filepath):
                # Append bias adjustment results
                if i.startswith(output_biasadj_prefix) == True:
                    biasadj_list.append(i)
                    if len(biasadj_list) == 1:
                        biasadj_all = pd.read_csv(output_filepath + i, index_col=0)
                    else:
                        biasadj_2join = pd.read_csv(output_filepath + i, index_col=0)
                        biasadj_all = biasadj_all.append(biasadj_2join, ignore_index=True)
                    # Remove file after its been merged
                    os.remove(output_filepath + i)
            # Export joined files
            biasadj_all_fn = (gcm_name + '_' + rcp_scenario + '_biasadj_opt' + str(option_bias_adjustment) + '_' 
                              + str(gcm_startyear - gcm_spinupyears) + '_' + str(gcm_endyear) + '_' +  'R' + 
                              str(rgi_regionsO1[0]) + '_' + str(strftime("%Y%m%d")) + '.csv')
            biasadj_all.to_csv(output_filepath + biasadj_all_fn)
            
            # Lapse rate parameters (same for all GCMs - only need to export once)
            lr_all_fn = ('biasadj_mon_lravg_' + str(gcm_startyear - gcm_spinupyears) + '_' + str(gcm_endyear) +
                         '_' + 'R' + str(rgi_regionsO1[0]) + '_' + str(strftime("%Y%m%d")) +'.csv')                
            output_lr_prefix = ('biasadj_mon_lravg_' + str(gcm_startyear - gcm_spinupyears) + '_' + str(gcm_endyear) +
                            '_' + 'R' + str(rgi_regionsO1[0]))
            # If it doesn't exist, then combine and export
            if os.path.exists(output_filepath + lr_all_fn) == False:
                # Append lapse rates
                for n_output in range(1,args.num_simultaneous_processes+1):
                    output_lr_solo = output_lr_prefix + '_' + str(n_output) + '.csv'
                    if n_output == 1:
                        lr_all = np.genfromtxt(output_filepath + output_lr_solo, delimiter=',')
                    else:
                        lr_2join = np.genfromtxt(output_filepath + output_lr_solo, delimiter=',')
                        lr_all = np.concatenate((lr_all, lr_2join), axis=0)
                    # Remove file after its been merged
                    os.remove(output_filepath + output_lr_solo)  
                np.savetxt(output_filepath + lr_all_fn, lr_all, delimiter=",")
            # Otherwise, simply remove files 
            else:
                for n_output in range(1,args.num_simultaneous_processes+1):
                    output_lr_solo = output_lr_prefix + '_' + str(n_output) + '.csv'
                    if os.path.exists(output_filepath + output_lr_solo) == True:
                        os.remove(output_filepath + output_lr_solo)

    print('Total processing time:', time.time()-time_start, 's')
            
#%% ===== PLOTTING AND PROCESSING FOR MODEL DEVELOPMENT =====     
    # Place local variables in variable explorer
    if (args.option_parallels == 0) or (main_glac_rgi_all.shape[0] < 2 * args.num_simultaneous_processes):     
        main_vars_list = list(main_vars.keys())
        gcm_name = main_vars['gcm_name']
        rcp_scenario = main_vars['rcp_scenario']
        main_glac_rgi = main_vars['main_glac_rgi']
        main_glac_hyps = main_vars['main_glac_hyps']
        main_glac_icethickness = main_vars['main_glac_icethickness']
        main_glac_width = main_vars['main_glac_width']
        elev_bins = main_vars['elev_bins']
        dates_table = main_vars['dates_table']
        glacier_rgi_table = main_vars['glacier_rgi_table']
        glacier_ref_temp = main_vars['glacier_ref_temp']
        glacier_ref_prec = main_vars['glacier_ref_prec']
        glacier_ref_elev = main_vars['glacier_ref_elev']
        glacier_ref_lrgcm = main_vars['glacier_ref_lrgcm']
        glacier_gcm_temp_adj = main_vars['glacier_gcm_temp_adj']
        glacier_gcm_prec_adj = main_vars['glacier_gcm_prec_adj']
        glacier_gcm_elev = main_vars['glacier_gcm_elev']
        glacier_gcm_lrgcm = main_vars['glacier_gcm_lrgcm']
        modelparameters = main_vars['modelparameters']
        glac_wide_massbaltotal_annual_gcm = main_vars['glac_wide_massbaltotal_annual_gcm']
        glac_wide_massbaltotal_annual_ref = main_vars['glac_wide_massbaltotal_annual_ref']
        main_glac_bias_adj = main_vars['main_glac_bias_adj']
        glacier_area_t0 = main_vars['glacier_area_t0']
        icethickness_t0 = main_vars['icethickness_t0']
        width_t0 = main_vars['width_t0']
        dates_table_subset = main_vars['dates_table_subset']
    
#        # Adjust temperature and precipitation to 'Zmed' so variables can properly be compared
#        glacier_elev_zmed = glacier_rgi_table.loc['Zmed']  
#        glacier_ref_temp_zmed = ((glacier_ref_temp + glacier_ref_lrgcm * (glacier_elev_zmed - glacier_ref_elev)
#                                  )[gcm_spinupyears*12:])
#        glacier_ref_prec_zmed = (glacier_ref_prec * modelparameters['precfactor'])[gcm_spinupyears*12:]
#        #  recall 'precfactor' is used to adjust for precipitation differences between gcm elev and zmed    
#        if option_bias_adjustment == 1:
#            glacier_gcm_temp_zmed = ((glacier_gcm_temp_adj + glacier_gcm_lrgcm * (glacier_elev_zmed - glacier_gcm_elev)
#                                      )[gcm_spinupyears*12:])
#            glacier_gcm_prec_zmed = (glacier_gcm_prec_adj * modelparameters['precfactor'])[gcm_spinupyears*12:]
#        elif (option_bias_adjustment == 2) or (option_bias_adjustment == 3):
#            glacier_gcm_temp_zmed = ((glacier_gcm_temp_adj + glacier_gcm_lrgcm * (glacier_elev_zmed - glacier_ref_elev)
#                                      )[gcm_spinupyears*12:])
#            glacier_gcm_prec_zmed = (glacier_gcm_prec_adj * modelparameters['precfactor'])[gcm_spinupyears*12:]
        
    #    # Plot reference vs. GCM temperature and precipitation
    #    # Monthly trends
    #    months = dates_table['date'][gcm_spinupyears*12:]
    #    years = np.unique(dates_table['wateryear'].values)[gcm_spinupyears:]
    #    # Temperature
    #    plt.plot(months, glacier_ref_temp_zmed, label='ref_temp')
    #    plt.plot(months, glacier_gcm_temp_zmed, label='gcm_temp')
    #    plt.ylabel('Monthly temperature [degC]')
    #    plt.legend()
    #    plt.show()
    #    # Precipitation
    #    plt.plot(months, glacier_ref_prec_zmed, label='ref_prec')
    #    plt.plot(months, glacier_gcm_prec_zmed, label='gcm_prec')
    #    plt.ylabel('Monthly precipitation [m]')
    #    plt.legend()
    #    plt.show()
    #    
    #    # Annual trends
    #    glacier_ref_temp_zmed_annual = glacier_ref_temp_zmed.reshape(-1,12).mean(axis=1)
    #    glacier_gcm_temp_zmed_annual = glacier_gcm_temp_zmed.reshape(-1,12).mean(axis=1)
    #    glacier_ref_prec_zmed_annual = glacier_ref_prec_zmed.reshape(-1,12).sum(axis=1)
    #    glacier_gcm_prec_zmed_annual = glacier_gcm_prec_zmed.reshape(-1,12).sum(axis=1)
    #    # Temperature
    #    plt.plot(years, glacier_ref_temp_zmed_annual, label='ref_temp')
    #    plt.plot(years, glacier_gcm_temp_zmed_annual, label='gcm_temp')
    #    plt.ylabel('Mean annual temperature [degC]')
    #    plt.legend()
    #    plt.show()
    #    # Precipitation
    #    plt.plot(years, glacier_ref_prec_zmed_annual, label='ref_prec')
    #    plt.plot(years, glacier_gcm_prec_zmed_annual, label='gcm_prec')
    #    plt.ylabel('Total annual precipitation [m]')
    #    plt.legend()
    #    plt.show()
    #    # Mass balance - bar plot
    #    bar_width = 0.35
    #    plt.bar(years, glac_wide_massbaltotal_annual_ref, bar_width, label='ref_MB')
    #    plt.bar(years+bar_width, glac_wide_massbaltotal_annual_gcm, bar_width, label='gcm_MB')
    #    plt.ylabel('Glacier-wide mass balance [mwea]')
    #    plt.legend()
    #    plt.show()
    #    # Cumulative mass balance - bar plot
    #    glac_wide_massbaltotal_annual_ref_cumsum = np.cumsum(glac_wide_massbaltotal_annual_ref)
    #    glac_wide_massbaltotal_annual_gcm_cumsum = np.cumsum(glac_wide_massbaltotal_annual_gcm)
    #    bar_width = 0.35
    #    plt.bar(years, glac_wide_massbaltotal_annual_ref_cumsum, bar_width, label='ref_MB')
    #    plt.bar(years+bar_width, glac_wide_massbaltotal_annual_gcm_cumsum, bar_width, label='gcm_MB')
    #    plt.ylabel('Cumulative glacier-wide mass balance [mwe]')
    #    plt.legend()
    #    plt.show() 
    #    # Histogram of differences
    #    mb_dif = main_glac_bias_adj['ref_mb_mwea'] - main_glac_bias_adj['gcm_mb_mwea']
    #    plt.hist(mb_dif)
    #    plt.show()      