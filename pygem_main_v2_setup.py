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
import netCDF4 as nc
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
import pygemfxns_output as output

#%%======== DEVELOPER'S TO-DO LIST ====================================================================================
# > Output log file, i.e., file that states input parameters, date of model run, model options selected,
#   and any errors that may have come up (e.g., precipitation corrected because negative value, etc.)

#%% ===== PRE-PROCESSING: GLACIERS, GLACIER CHARACTERISTICS, AND CLIMATE DATA ===========================================
# Select glaciers, glacier characteristics, and climate data
timestart_step1 = timeit.default_timer()
# RGI glacier attributes
main_glac_rgi = modelsetup.selectglaciersrgitable()
# Select calibration data from geodetic mass balance from David Shean
main_glac_calmassbal = modelsetup.selectcalibrationdata(main_glac_rgi)
# Concatenate massbal data to the main glacier
main_glac_rgi = pd.concat([main_glac_rgi, main_glac_calmassbal], axis=1)
# For calibration, filter data to those that have calibration data
if input.option_removeNaNcal == 1:
    # Drop glaciers who are missing calibration data
    main_glac_calmassbal = main_glac_calmassbal.dropna()
    main_glac_rgi = main_glac_rgi.dropna()
    main_glac_rgi.reset_index(drop=True, inplace=True)
    
## ===== GLACIER PROPERTIES =====
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
# Quality control - if ice thickness = 0, glacier area = 0 (problem identified by glacier RGIV6-15.00016 on 03/06/2018)
main_glac_hyps[main_glac_icethickness == 0] = 0
# Add volume [km**3] and mean elevation [m a.s.l.] to the main glaciers table
main_glac_rgi['Volume'], main_glac_rgi['Zmean'] = modelsetup.hypsometrystats(main_glac_hyps, main_glac_icethickness) 

# ===== MODEL TIME FRAME =====
dates_table, start_date, end_date, monthly_columns, annual_columns, annual_divisor = modelsetup.datesmodelrun()

# ===== GLACIER CLIMATE DATA =====
## SELECT VIA NEAREST NEIGHBOR AND COMPUTE LAPSE RATES
## Air Temperature [degC] and GCM dates
#main_glac_gcmtemp, main_glac_gcmdate = climate.importGCMvarnearestneighbor_xarray(
#        input.gcm_temp_filename, input.gcm_temp_varname, main_glac_rgi, dates_table, start_date, end_date)
## Precipitation [m] and GCM dates
#main_glac_gcmprec, main_glac_gcmdate = climate.importGCMvarnearestneighbor_xarray(
#        input.gcm_prec_filename, input.gcm_prec_varname, main_glac_rgi, dates_table, start_date, end_date)
# Elevation [m a.s.l] associated with air temperature  and precipitation data
main_glac_gcmelev = climate.importGCMfxnearestneighbor_xarray(input.gcm_elev_filename, input.gcm_elev_varname, main_glac_rgi)
## Add GCM time series to the dates_table
#dates_table['date_gcm'] = main_glac_gcmdate
## Lapse rates [degC m-1]  
#if input.option_lapserate_fromgcm == 1:
#    main_glac_gcmlapserate, main_glac_gcmdate = climate.importGCMvarnearestneighbor_xarray(
#            input.gcm_lapserate_filename, input.gcm_lapserate_varname, main_glac_rgi, dates_table, start_date, 
#            end_date)
# LOAD CLIMATE DATA ALREADY IN CSV FILES
#main_glac_gcmtemp = np.genfromtxt(input.gcm_filepath_var + input.gcmtemp_filedict[input.rgi_regionsO1[0]], 
#                                  delimiter=',')
#main_glac_gcmprec = np.genfromtxt(input.gcm_filepath_var + input.gcmprec_filedict[input.rgi_regionsO1[0]], 
#                                  delimiter=',')
#main_glac_gcmelev = np.genfromtxt(input.gcm_filepath_var + input.gcmelev_filedict[input.rgi_regionsO1[0]], 
#                                  delimiter=',')
#main_glac_gcmlapserate = np.genfromtxt(input.gcm_filepath_var + 
#                                       input.gcmlapserate_filedict[input.rgi_regionsO1[0]], delimiter=',')
    
#%% ===== WRITE CSV FILES OF INPUT DATA ===============================================================================
# Write list of glaciers    
if input.option_removeNaNcal == 0:
    glacier_list_fn = input.modelsetup_dir + input.glacier_list_name + '.csv'
else:
    glacier_list_fn = input.modelsetup_dir + input.glacier_list_name + 'removeNaNcal.csv'
glacier_list = main_glac_rgi['RGIId']
np.savetxt(glacier_list_fn, glacier_list, fmt='%s')
# Write dates table
dates_table_fn = input.modelsetup_dir + 'dates_table_1995_2015_monthly.csv'
dates_table.to_csv(dates_table_fn)
# Write datasets for all other glaciers
# ERROR ASSOCIATED WITH THE SELECTINO OF THE GLACIER NUMBER!
for glac in range(main_glac_rgi.shape[0]):
#    # Write glacier data
#    glacier_properties = np.zeros((4,main_glac_hyps.shape[1]))
#    glacier_properties[0,:] = elev_bins
#    glacier_properties[1,:] = main_glac_hyps.loc[glac,:].values
#    glacier_properties[2,:] = main_glac_icethickness.loc[glac,:].values
#    glacier_properties[3,:] = main_glac_width.loc[glac,:].values
#    glacier_properties_fn = input.modelsetup_dir + main_glac_rgi.loc[glac,'RGIId'] + '_glacierproperties.csv'
#    np.savetxt(glacier_properties_fn, glacier_properties)
#    
#    # Write climate data
#    glacier_climate = np.zeros((4,main_glac_gcmtemp.shape[1]))
#    glacier_climate[0,:] = main_glac_gcmtemp[glac,:]
#    glacier_climate[1,:] = main_glac_gcmprec[glac,:]
#    glacier_climate[2,:] = main_glac_gcmlapserate[glac,:]
#    glacier_climate[3,:] = main_glac_gcmelev[glac]
#    glacier_climate_fn = input.modelsetup_dir + main_glac_rgi.loc[glac,'RGIId'] + '_ERAInterim_tple_1995_2015.csv'
#    np.savetxt(glacier_climate_fn, glacier_climate)
    
    # Write rgi info
    glacier_rgi = main_glac_rgi.loc[glac,:]
    glacier_rgi_fn = input.modelsetup_dir + main_glac_rgi.loc[glac,'RGIId'] + '_RGIinfo.csv'
    glacier_rgi.to_csv(glacier_rgi_fn)
    

## ====== MODEL PARAMETERS ======
#if input.option_calibration == 0:
#     # Load model parameters
#    if input.option_loadparameters == 1:
#        main_glac_modelparams = pd.read_csv(input.modelparams_filepath + input.modelparams_filename) 
#    else:
#        main_glac_modelparams = pd.DataFrame(np.repeat([input.lrgcm, input.lrglac, input.precfactor, input.precgrad, 
#            input.ddfsnow, input.ddfice, input.tempsnow, input.tempchange], main_glac_rgi.shape[0]).reshape(-1, 
#            main_glac_rgi.shape[0]).transpose(), columns=input.modelparams_colnames)
#elif input.option_calibration == 1:
#    # Create grid of all parameter sets
#    grid_modelparameters = np.zeros((input.grid_precfactor.shape[0] * input.grid_tempbias.shape[0] * 
#                                     input.grid_ddfsnow.shape[0] * input.grid_precgrad.shape[0],8))
#    grid_count = 0
#    for n_precfactor in range(input.grid_precfactor.shape[0]):
#        for n_tempbias in range(input.grid_tempbias.shape[0]):
#            for n_ddfsnow in range(input.grid_ddfsnow.shape[0]):
#                for n_precgrad in range(input.grid_precgrad.shape[0]):
#                    # Set grid of model parameters
#                    grid_modelparameters[grid_count,:] = [input.lrgcm, input.lrglac, 
#                          input.grid_precfactor[n_precfactor], input.grid_precgrad[n_precgrad], 
#                          input.grid_ddfsnow[n_ddfsnow], input.grid_ddfsnow[n_ddfsnow] / 0.7, input.tempsnow, 
#                          input.grid_tempbias[n_tempbias]]
#                    grid_count = grid_count + 1