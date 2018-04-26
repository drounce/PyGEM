"""
Main script for future simulations.  Separated from main script for calibrations to be cleaner for development, but
should consider merging back in future.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
import xarray as xr
import netCDF4 as nc
from time import strftime
import time
from scipy.optimize import minimize
from scipy.stats import linregress
import matplotlib.pyplot as plt
import cartopy

import pygem_input as input
import pygemfxns_modelsetup as modelsetup
import pygemfxns_climate as climate
import pygemfxns_massbalance as massbalance
import pygemfxns_output as output

time_start = time.time()

#%% ===== LOAD GLACIER DATA =====
# RGI glacier attributes
main_glac_rgi = modelsetup.selectglaciersrgitable()
# For calibration, filter data to those that have calibration data
if input.option_calibration == 1:
    # Select calibration data from geodetic mass balance from David Shean
    main_glac_calmassbal = modelsetup.selectcalibrationdata(main_glac_rgi)
    # Concatenate massbal data to the main glacier
    main_glac_rgi = pd.concat([main_glac_rgi, main_glac_calmassbal], axis=1)
#    # Drop those with nan values
#    main_glac_calmassbal = main_glac_calmassbal.dropna()
#    main_glac_rgi = main_glac_rgi.dropna()
# Glacier hypsometry [km**2], total area
main_glac_hyps = modelsetup.import_Husstable(main_glac_rgi, input.rgi_regionsO1, input.hyps_filepath, 
                                             input.hyps_filedict, input.indexname, input.hyps_colsdrop)
elev_bins = main_glac_hyps.columns.values.astype(int)
# Ice thickness [m], average
main_glac_icethickness = modelsetup.import_Husstable(main_glac_rgi, input.rgi_regionsO1, input.thickness_filepath, 
                                                 input.thickness_filedict, input.indexname, input.thickness_colsdrop)
# Width [km], average
main_glac_width = modelsetup.import_Husstable(main_glac_rgi, input.rgi_regionsO1, input.width_filepath, 
                                              input.width_filedict, input.indexname, input.width_colsdrop)
# Add volume [km**3] and mean elevation [m a.s.l.] to the main glaciers table
main_glac_rgi['Volume'], main_glac_rgi['Zmean'] = modelsetup.hypsometrystats(main_glac_hyps, main_glac_icethickness)
# Model time frame
dates_table, start_date, end_date = modelsetup.datesmodelrun(input.startyear, input.endyear, input.spinupyears)
# Quality control - if ice thickness = 0, glacier area = 0 (problem identified by glacier RGIV6-15.00016 on 03/06/2018)
main_glac_hyps[main_glac_icethickness == 0] = 0
# Model parameters
main_glac_modelparams = pd.read_csv(input.modelparams_filepath + input.modelparams_filename) 

#%% ===== LOAD BIAS ADJUSTMENTS =====
if input.option_bias_adjustment != 0:
    biasadj_temp_all = np.genfromtxt(input.biasadj_filepath + input.biasadj_fn_temp, delimiter=',')
    biasadj_prec_all = np.genfromtxt(input.biasadj_filepath + input.biasadj_fn_prec, delimiter=',')
    biasadj_lr_all   = np.genfromtxt(input.biasadj_filepath + input.biasadj_fn_lr, delimiter=',')
    # Select the lapse rates for the respective glaciers included in the simulation
    if input.rgi_glac_number == 'all':
        biasadj_lr = biasadj_lr_all
        biasadj_temp = biasadj_temp_all
        biasadj_prec = biasadj_prec_all
    elif input.option_bias_adjustment == 1:
        biasadj_temp = np.zeros(main_glac_rgi.shape[0])
        biasadj_prec = np.zeros(main_glac_rgi.shape[0])
        biasadj_lr   = np.zeros((main_glac_rgi.shape[0], biasadj_lr_all.shape[1]))
        for glac in range(main_glac_rgi.shape[0]):
            biasadj_temp[glac] = biasadj_temp_all[main_glac_rgi.loc[glac,'O1Index']]
            biasadj_prec[glac] = biasadj_prec_all[main_glac_rgi.loc[glac,'O1Index']]
            biasadj_lr[glac] = biasadj_lr_all[main_glac_rgi.loc[glac,'O1Index'],:]
    elif input.option_bias_adjustment == 2:
        print('Set up options for Huss and Hock [2015] import methods')

#%% ===== LOAD CLIMATE DATA =====
if input.option_gcm_downscale == 1:  
    # Air Temperature [degC] and GCM dates
    main_glac_gcmtemp_raw, main_glac_gcmdate = climate.importGCMvarnearestneighbor_xarray(
            input.gcm_temp_filename, input.gcm_temp_varname, main_glac_rgi, dates_table, start_date, end_date)
    # Precipitation [m] and GCM dates
    main_glac_gcmprec_raw, main_glac_gcmdate = climate.importGCMvarnearestneighbor_xarray(
            input.gcm_prec_filename, input.gcm_prec_varname, main_glac_rgi, dates_table, start_date, end_date)
    # Elevation [m a.s.l] associated with air temperature  and precipitation data
    main_glac_gcmelev = climate.importGCMfxnearestneighbor_xarray(
            input.gcm_elev_filename, input.gcm_elev_varname, main_glac_rgi)
    # Add GCM time series to the dates_table
    dates_table['date_gcm'] = main_glac_gcmdate
    # Lapse rate[degC m-1] 
    if input.option_bias_adjustment != 0:
        if input.timestep == 'monthly':
            main_glac_gcmlapserate = np.tile(biasadj_lr, int(dates_table.shape[0]/12))
    elif input.option_lapserate_fromgcm == 1:
        print('Likely need to update this function with new importGCMvar input sequence')
        main_glac_gcmlapserate, main_glac_gcmdate = climate.importGCMvarnearestneighbor_xarray(
                input.gcm_filepath_var, input.gcm_lapserate_filename, input.gcm_lapserate_varname, main_glac_rgi, 
                dates_table, start_date, end_date)
    
    # Temperature and Precipitation Bias Adjustments
    if input.option_bias_adjustment == 0:
        main_glac_gcmtemp = main_glac_gcmtemp_raw
        main_glac_gcmprec = main_glac_gcmprec_raw
    elif input.option_bias_adjustment == 1:
        main_glac_gcmtemp = main_glac_gcmtemp_raw + biasadj_temp[:,np.newaxis]
        main_glac_gcmprec = main_glac_gcmprec_raw * biasadj_prec[:,np.newaxis]
    elif input.option_bias_adjustment == 2:
        print('Set up options for Huss and Hock [2015] import methods')

#%%===== SIMULATION RUN =====
# Create output netcdf file
if input.output_package != 1:
    netcdf_fullfilename = output.netcdfcreate(input.rgi_regionsO1[0], main_glac_hyps, dates_table)
    
# ENTER GLACIER LOOP
for glac in range(main_glac_rgi.shape[0]):
#    for glac in [0]:
    print(main_glac_rgi.loc[glac,'RGIId'])
    # Select subset of variables to reduce the amount of data being passed to the function
    modelparameters = main_glac_modelparams.loc[glac,:].values
    glacier_rgi_table = main_glac_rgi.loc[glac, :]
    glacier_gcm_elev = main_glac_gcmelev[glac]
    glacier_gcm_prec = main_glac_gcmprec[glac,:]
    glacier_gcm_temp = main_glac_gcmtemp[glac,:]
    glacier_gcm_lrgcm = main_glac_gcmlapserate[glac]
    glacier_gcm_lrglac = glacier_gcm_lrgcm.copy()
    glacier_area_t0 = main_glac_hyps.iloc[glac,:].values.astype(float)   
    # Inclusion of ice thickness and width, i.e., loading values may be only required for Huss mass redistribution!
    icethickness_t0 = main_glac_icethickness.iloc[glac,:].values.astype(float)
    width_t0 = main_glac_width.iloc[glac,:].values.astype(float)
    # MASS BALANCE
    # Run the mass balance function (spinup years have been removed from output)
    (glac_bin_temp, glac_bin_prec, glac_bin_acc, glac_bin_refreeze, glac_bin_snowpack, glac_bin_melt, 
     glac_bin_frontalablation, glac_bin_massbalclim, glac_bin_massbalclim_annual, glac_bin_area_annual, 
     glac_bin_icethickness_annual, glac_bin_width_annual, glac_bin_surfacetype_annual, 
     glac_wide_massbaltotal, glac_wide_runoff, glac_wide_snowline, glac_wide_snowpack, glac_wide_area_annual, 
     glac_wide_volume_annual, glac_wide_ELA_annual) = (
        massbalance.runmassbalance(modelparameters, glacier_rgi_table, glacier_area_t0, icethickness_t0, width_t0, 
                                   elev_bins, glacier_gcm_temp, glacier_gcm_prec, glacier_gcm_elev, glacier_gcm_lrgcm, 
                                   glacier_gcm_lrglac, dates_table))
    # OUTPUT: Record variables according to output package
    #  must be done within glacier loop since the variables will be overwritten 
    if input.output_package != 0:
        output.netcdfwrite(netcdf_fullfilename, glac, modelparameters, glacier_rgi_table, elev_bins, glac_bin_temp, 
                           glac_bin_prec, glac_bin_acc, glac_bin_refreeze, glac_bin_snowpack, glac_bin_melt, 
                           glac_bin_frontalablation, glac_bin_massbalclim, glac_bin_massbalclim_annual, 
                           glac_bin_area_annual, glac_bin_icethickness_annual, glac_bin_width_annual,
                           glac_bin_surfacetype_annual)
            
#%%=== Model testing ===============================================================================
#netcdf_output = nc.Dataset(input.main_directory + '/' + netcdf_fullfilename, 'r+')
print('Processing time:', time.time()-time_start, 's')