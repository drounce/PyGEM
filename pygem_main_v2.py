#%% ###################################################################################################################
"""
Python Glacier Evolution Model "PyGEM" V1.0
Prepared by David Rounce with support from Regine Hock.
This work was funded under the NASA-ROSES program (grant no. NNX17AB27G).

pygem_main_v2.py changes the structure so the glacier loop is used for the entire setup

PyGEM is an open source glacier evolution model written in python.  The model expands upon previous models from
Radic et al. (2013), Bliss et al. (2014), and Huss and Hock (2015).
"""
#######################################################################################################################
# This is the main script that provides the architecture and framework for all of the model runs. All input data is
# included in a separate module called pygem_input.py. It is recommended to not make any changes to this file unless
# you are a PyGEM developer and making changes to the model architecture.
#
#%%========= IMPORT PACKAGES ==========================================================================================
# Various packages are used to provide the proper architecture and framework for the calculations used in this script.
# Some packages (e.g., datetime) are included in order to speed up calculations and simplify code.
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
#from datetime import datetime
#import os # os is used with re to find name matches
#import re # see os
#import xarray as xr
#import netCDF4 as nc
#from time import strftime
import timeit
#from scipy.optimize import minimize
#from scipy.stats import linregress
#import matplotlib.pyplot as plt
#import cartopy

#========== IMPORT INPUT AND FUNCTIONS FROM MODULES ===================================================================
import pygem_input as input
import pygemfxns_modelsetup as modelsetup
import pygemfxns_climate as climate
import pygemfxns_massbalance as massbalance
#import pygemfxns_output as output

#%%======== DEVELOPER'S TO-DO LIST ====================================================================================
# > Output log file, i.e., file that states input parameters, date of model run, model options selected,
#   and any errors that may have come up (e.g., precipitation corrected because negative value, etc.)

# ===== PRE-PROCESSING: GLACIERS, GLACIER CHARACTERISTICS, AND CLIMATE DATA ===========================================
# Select glaciers, glacier characteristics, and climate data
timestart_step1 = timeit.default_timer()
# RGI glacier attributes
main_glac_rgi = modelsetup.selectglaciersrgitable()
# For calibration, filter data to those that have calibration data
if input.option_calibration == 1:
    # Select calibration data from geodetic mass balance from David Shean
    main_glac_calmassbal = modelsetup.selectcalibrationdata(main_glac_rgi)
    # Concatenate massbal data to the main glacier
    main_glac_rgi = pd.concat([main_glac_rgi, main_glac_calmassbal], axis=1)
    # Drop those with nan values
    main_glac_calmassbal = main_glac_calmassbal.dropna()
    main_glac_rgi = main_glac_rgi.dropna()
# ===== GLACIER PROPERTIES =====
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
dates_table, start_date, end_date, monthly_columns, annual_columns, annual_divisor = modelsetup.datesmodelrun()
# Quality control - if ice thickness = 0, glacier area = 0 (problem identified by glacier RGIV6-15.00016 on 03/06/2018)
main_glac_hyps[main_glac_icethickness == 0] = 0
# ===== GLACIER CLIMATE DATA =====
# Air Temperature [degC] and GCM dates
main_glac_gcmtemp, main_glac_gcmdate = climate.importGCMvarnearestneighbor_xarray(
        input.gcm_temp_filename, input.gcm_temp_varname, main_glac_rgi, dates_table, start_date, end_date)
# Precipitation [m] and GCM dates
main_glac_gcmprec, main_glac_gcmdate = climate.importGCMvarnearestneighbor_xarray(
        input.gcm_prec_filename, input.gcm_prec_varname, main_glac_rgi, dates_table, start_date, end_date)
# Elevation [m a.s.l] associated with air temperature  and precipitation data
main_glac_gcmelev = climate.importGCMfxnearestneighbor_xarray(input.gcm_elev_filename, input.gcm_elev_varname, 
                                                              main_glac_rgi)
# Add GCM time series to the dates_table
dates_table['date_gcm'] = main_glac_gcmdate
# Lapse rates [degC m-1]  
if input.option_lapserate_fromgcm == 1:
    main_glac_gcmlapserate, main_glac_gcmdate = climate.importGCMvarnearestneighbor_xarray(
            input.gcm_lapserate_filename, input.gcm_lapserate_varname, main_glac_rgi, dates_table, start_date, 
            end_date)


# ===== MODEL RUN =====================================================================================================
#%% ENTER GLACIER LOOP
for glac in range(main_glac_rgi.shape[0]):
    # For calibraiton, index may not start at 0, so used the O1 index to get glacier numbers
    print(main_glac_rgi.iloc[glac,1])
    
    # SET MODEL PARAMETERS
    modelparameters = np.array([input.lrgcm, input.lrglac, input.precfactor, input.precgrad,
                                input.ddfsnow, input.ddfice, input.tempsnow, input.tempchange])
    
    # MASS BALANCE
    # Select subsets of data to send to the mass balance function
    glacier_rgi_table = main_glac_rgi.iloc[glac, :]
    glacier_gcm_temp = main_glac_gcmtemp[glac,:]
    glacier_gcm_prec = main_glac_gcmprec[glac,:]
    glacier_gcm_elev = main_glac_gcmelev[glac]
    if input.option_lapserate_fromgcm == 1:
        glacier_gcm_lrgcm = main_glac_gcmlapserate[glac]
        glacier_gcm_lrglac = glacier_gcm_lrgcm.copy()
    else:
        glacier_gcm_lrgcm = np.zeros(glacier_gcm_temp.shape) + modelparameters[0]
        glacier_gcm_lrglac = np.zeros(glacier_gcm_temp.shape) + modelparameters[1]
    glacier_area_t0 = main_glac_hyps.iloc[glac,:].values.astype(float)   
    icethickness_t0 = main_glac_icethickness.iloc[glac,:].values.astype(float)
    width_t0 = main_glac_width.iloc[glac,:].values.astype(float)
    # Run the mass balance function (spinup years have been removed from output)
    (glac_bin_temp, glac_bin_prec, glac_bin_acc, glac_bin_refreeze, glac_bin_snowpack, glac_bin_melt, 
     glac_bin_frontalablation, glac_bin_massbalclim, glac_bin_massbalclim_annual, glac_bin_area_annual, 
     glac_bin_icethickness_annual, glac_bin_width_annual, glac_bin_surfacetype_annual) = (
             massbalance.runmassbalance(glac, modelparameters, glacier_rgi_table, glacier_area_t0, 
                                        icethickness_t0, width_t0, elev_bins, glacier_gcm_temp, glacier_gcm_prec, 
                                        glacier_gcm_elev, glacier_gcm_lrgcm, glacier_gcm_lrglac, 
                                        dates_table, annual_columns, annual_divisor))
    if input.option_calibration == 1:
        # Compare calibration data
        # Column index for start and end year based on dates of geodetic mass balance observations
        massbal_idx_start = (glacier_rgi_table.loc[input.massbal_time1] - input.startyear).astype(int)
        massbal_idx_end = (massbal_idx_start + glacier_rgi_table.loc[input.massbal_time2] - 
                           glacier_rgi_table.loc[input.massbal_time1] + 1).astype(int)
        massbal_years = massbal_idx_end - massbal_idx_start
        # Average annual glacier-wide mass balance [m w.e.]
        glac_wide_massbalclim_mwea = ((glac_bin_massbalclim_annual[:, massbal_idx_start:massbal_idx_end] *
                                       glac_bin_area_annual[:, massbal_idx_start:massbal_idx_end]).sum() /
                                       glacier_area_t0.sum() / massbal_years)
        #  units: m w.e. based on initial area
        # Difference between geodetic and modeled mass balance
        massbal_difference = abs(glacier_rgi_table[input.massbal_colname] - glac_wide_massbalclim_mwea)
        # Output: Measured mass balance, Measured mass balance uncertainty, modeled mass balance, mass balance difference
        print(glacier_rgi_table.loc[input.massbal_colname], glacier_rgi_table.loc[input.massbal_uncertainty_colname], 
              glac_wide_massbalclim_mwea, massbal_difference)

timeelapsed_step1 = timeit.default_timer() - timestart_step1
print('\ntime:', timeelapsed_step1, "s\n")