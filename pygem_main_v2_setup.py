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

#%% ==
timestart_step1 = timeit.default_timer()
# RGI glacier attributes
main_glac_rgi = modelsetup.selectglaciersrgitable()
# Select calibration data from geodetic mass balance from David Shean
main_glac_calmassbal = modelsetup.selectcalibrationdata(main_glac_rgi)
# Concatenate massbal data to the main glacier
main_glac_rgi = pd.concat([main_glac_rgi, main_glac_calmassbal], axis=1)
main_glac_gcmtemp = np.genfromtxt(input.gcm_filepath_var + input.gcmtemp_filedict[input.rgi_regionsO1[0]], 
                                      delimiter=',')
main_glac_gcmprec = np.genfromtxt(input.gcm_filepath_var + input.gcmprec_filedict[input.rgi_regionsO1[0]], 
                                  delimiter=',')
main_glac_gcmelev = np.genfromtxt(input.gcm_filepath_var + input.gcmelev_filedict[input.rgi_regionsO1[0]], 
                                  delimiter=',')
# Lapse rates [degC m-1]  
if input.option_lapserate_fromgcm == 1:
    main_glac_gcmlapserate = np.genfromtxt(input.gcm_filepath_var + 
                                           input.gcmlapserate_filedict[input.rgi_regionsO1[0]], delimiter=',')
    
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


timeelapsed_step1 = timeit.default_timer() - timestart_step1
print('\ntime:', timeelapsed_step1, "s\n")
#%% ===== PRE-PROCESSING: GLACIERS, GLACIER CHARACTERISTICS, AND CLIMATE DATA ===========================================
## Select glaciers, glacier characteristics, and climate data
#timestart_step1 = timeit.default_timer()
## RGI glacier attributes
#main_glac_rgi = modelsetup.selectglaciersrgitable()
## Select calibration data from geodetic mass balance from David Shean
#main_glac_calmassbal = modelsetup.selectcalibrationdata(main_glac_rgi)
## Concatenate massbal data to the main glacier
#main_glac_rgi = pd.concat([main_glac_rgi, main_glac_calmassbal], axis=1)
### For calibration, filter data to those that have calibration data
##if input.option_calibration == 1:
##    # Select calibration data from geodetic mass balance from David Shean
##    main_glac_calmassbal = modelsetup.selectcalibrationdata(main_glac_rgi)
##    # Concatenate massbal data to the main glacier
##    main_glac_rgi = pd.concat([main_glac_rgi, main_glac_calmassbal], axis=1)
##    # Drop those with nan values
##    main_glac_calmassbal = main_glac_calmassbal.dropna()
##    main_glac_rgi = main_glac_rgi.dropna()
## ===== GLACIER PROPERTIES =====
## Glacier hypsometry [km**2], total area
#main_glac_hyps = modelsetup.import_Husstable(main_glac_rgi, input.rgi_regionsO1, input.hyps_filepath, 
#                                             input.hyps_filedict, input.indexname, input.hyps_colsdrop)
#elev_bins = main_glac_hyps.columns.values.astype(int)
## Ice thickness [m], average
#main_glac_icethickness = modelsetup.import_Husstable(main_glac_rgi, input.rgi_regionsO1, input.thickness_filepath, 
#                                                 input.thickness_filedict, input.indexname, input.thickness_colsdrop)
## Width [km], average
#main_glac_width = modelsetup.import_Husstable(main_glac_rgi, input.rgi_regionsO1, input.width_filepath, 
#                                              input.width_filedict, input.indexname, input.width_colsdrop)
## Add volume [km**3] and mean elevation [m a.s.l.] to the main glaciers table
#main_glac_rgi['Volume'], main_glac_rgi['Zmean'] = modelsetup.hypsometrystats(main_glac_hyps, main_glac_icethickness)
## Model time frame
#dates_table, start_date, end_date, monthly_columns, annual_columns, annual_divisor = modelsetup.datesmodelrun()
## Quality control - if ice thickness = 0, glacier area = 0 (problem identified by glacier RGIV6-15.00016 on 03/06/2018)
#main_glac_hyps[main_glac_icethickness == 0] = 0
## ===== GLACIER CLIMATE DATA =====
## Air Temperature [degC] and GCM dates
#main_glac_gcmtemp, main_glac_gcmdate = climate.importGCMvarnearestneighbor_xarray(
#        input.gcm_temp_filename, input.gcm_temp_varname, main_glac_rgi, dates_table, start_date, end_date)
## Precipitation [m] and GCM dates
#main_glac_gcmprec, main_glac_gcmdate = climate.importGCMvarnearestneighbor_xarray(
#        input.gcm_prec_filename, input.gcm_prec_varname, main_glac_rgi, dates_table, start_date, end_date)
## Elevation [m a.s.l] associated with air temperature  and precipitation data
#main_glac_gcmelev = climate.importGCMfxnearestneighbor_xarray(input.gcm_elev_filename, input.gcm_elev_varname, 
#                                                              main_glac_rgi)
## Add GCM time series to the dates_table
#dates_table['date_gcm'] = main_glac_gcmdate
## Lapse rates [degC m-1]  
#if input.option_lapserate_fromgcm == 1:
#    main_glac_gcmlapserate, main_glac_gcmdate = climate.importGCMvarnearestneighbor_xarray(
#            input.gcm_lapserate_filename, input.gcm_lapserate_varname, main_glac_rgi, dates_table, start_date, 
#            end_date)
    
# WRITE CSV FILES OF INPUT DATA
modelsetup_dir = input.main_directory + '/../PyGEM_modelsetup/'
# Write list of glaciers    
glacier_list_fn = modelsetup_dir + 'glacier_list_R15_all.csv'
glacier_list = main_glac_rgi['RGIId']
np.savetxt(glacier_list_fn, glacier_list, fmt='%s')
# Write list of only glaciers with calibration data
glacier_list_dropna_fn = modelsetup_dir + 'glacier_list_R15_calmbonly.csv'
main_glac_rgi_dropna = main_glac_rgi.dropna()
glacier_list_dropna = main_glac_rgi_dropna['RGIId']
np.savetxt(glacier_list_dropna_fn, glacier_list_dropna, fmt='%s')
# Write dates table
dates_table_fn = modelsetup_dir + 'dates_table_1995_2015_monthly.csv'
dates_table.to_csv(dates_table_fn)
# Write datasets for all other glaciers
for glac in range(main_glac_rgi.shape[0]):
#for glac in [0]:
    # Write glacier data
    glacier_properties = np.zeros((4,main_glac_hyps.shape[1]))
    glacier_properties[0,:] = elev_bins
    glacier_properties[1,:] = main_glac_hyps.loc[glac,:].values
    glacier_properties[2,:] = main_glac_icethickness.loc[glac,:].values
    glacier_properties[3,:] = main_glac_width.loc[glac,:].values
    glacier_properties_fn = modelsetup_dir + main_glac_rgi.loc[glac,'RGIId'] + '_glacierproperties.csv'
    np.savetxt(glacier_properties_fn, glacier_properties)
    
    # Write climate data
    glacier_climate = np.zeros((4,main_glac_gcmtemp.shape[1]))
    glacier_climate[0,:] = main_glac_gcmtemp[glac,:]
    glacier_climate[1,:] = main_glac_gcmprec[glac,:]
    glacier_climate[2,:] = main_glac_gcmlapserate[glac,:]
    glacier_climate[3,:] = main_glac_gcmelev[glac]
    glacier_climate_fn = modelsetup_dir + main_glac_rgi.loc[glac,'RGIId'] + '_ERAInterim_tple_19952015.csv'
    np.savetxt(glacier_climate_fn, glacier_climate)
    
    # Write rgi info
    glacier_rgi = main_glac_rgi.loc[glac,:]
    glacier_rgi_fn = modelsetup_dir + main_glac_rgi.loc[glac,'RGIId'] + '_RGIinfo.csv'
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
## ===== OUTPUT FILE =====
## Create output netcdf file
#if input.option_calibration == 0 and input.output_package != 0:
#    output_fullfilename = output.netcdfcreate(input.rgi_regionsO1[0], main_glac_hyps, dates_table, annual_columns)
#elif input.option_calibration == 1:
#    output_fullfilename = output.netcdfcreate_calgridsearch(input.rgi_regionsO1[0], main_glac_hyps, dates_table, 
#                                                            annual_columns, grid_modelparameters)

## ===== MODEL RUN =====================================================================================================
##%% ENTER GLACIER LOOP
#for glac in range(main_glac_rgi.shape[0]):
##for glac in [0]:
#    print(main_glac_rgi.iloc[glac,1])
#
#    # MASS BALANCE
#    # Select subsets of data to send to the mass balance function
#    glacier_rgi_table = main_glac_rgi.iloc[glac, :]
#    glacier_gcm_temp = main_glac_gcmtemp[glac,:]
#    glacier_gcm_prec = main_glac_gcmprec[glac,:]
#    glacier_gcm_elev = main_glac_gcmelev[glac]
#    glacier_gcm_lrgcm = main_glac_gcmlapserate[glac]
#    glacier_gcm_lrglac = glacier_gcm_lrgcm.copy()
#    glacier_area_t0 = main_glac_hyps.iloc[glac,:].values.astype(float)   
#    icethickness_t0 = main_glac_icethickness.iloc[glac,:].values.astype(float)
#    width_t0 = main_glac_width.iloc[glac,:].values.astype(float)
#    # Single simulation - glacier area varies
#    if input.option_calibration == 0:
#        # Select model parameters
#        modelparameters = main_glac_modelparams.loc[glac,:].values
#        # Run the mass balance function (spinup years have been removed from output)
#        (glac_bin_temp, glac_bin_prec, glac_bin_acc, glac_bin_refreeze, glac_bin_snowpack, glac_bin_melt, 
#         glac_bin_frontalablation, glac_bin_massbalclim, glac_bin_massbalclim_annual, glac_bin_area_annual, 
#         glac_bin_icethickness_annual, glac_bin_width_annual, glac_bin_surfacetype_annual, glac_wide_massbaltotal, 
#         glac_wide_runoff, glac_wide_snowline, glac_wide_snowpack, glac_wide_area_annual, glac_wide_volume_annual, 
#         glac_wide_ELA_annual) = (
#                 massbalance.runmassbalance(glac, modelparameters, glacier_rgi_table, glacier_area_t0, 
#                                            icethickness_t0, width_t0, elev_bins, glacier_gcm_temp, glacier_gcm_prec, 
#                                            glacier_gcm_elev, glacier_gcm_lrgcm, glacier_gcm_lrglac, 
#                                            dates_table, annual_columns, annual_divisor))
##        # Output for Shane: Measured mass balance, Measured mass balance uncertainty, modeled mass balance, mass balance difference
##        print(glacier_rgi_table.loc[input.massbal_colname], glacier_rgi_table.loc[input.massbal_uncertainty_colname], 
##              glac_wide_massbalclim_mwea, massbal_difference)
#        # OUTPUT: Record variables according to output package
#        #  must be done within glacier loop since the variables will be overwritten 
#        if input.output_package != 0:
#            output.netcdfwrite(output_fullfilename, glac, modelparameters, glacier_rgi_table, elev_bins, glac_bin_temp, 
#                               glac_bin_prec, glac_bin_acc, glac_bin_refreeze, glac_bin_snowpack, glac_bin_melt, 
#                               glac_bin_frontalablation, glac_bin_massbalclim, glac_bin_massbalclim_annual, 
#                               glac_bin_area_annual, glac_bin_icethickness_annual, glac_bin_width_annual, 
#                               glac_bin_surfacetype_annual)
#    # Grid search calibration - glacier area constant
#    if input.option_calibration == 1:
#        # Set new output 
#        output_glac_wide_massbaltotal = np.zeros((grid_modelparameters.shape[0], glacier_gcm_temp.shape[0] - input.spinupyears*12))
#        output_glac_wide_runoff = np.zeros((grid_modelparameters.shape[0], glacier_gcm_temp.shape[0] - input.spinupyears*12))
#        output_glac_wide_snowline = np.zeros((grid_modelparameters.shape[0], glacier_gcm_temp.shape[0] - input.spinupyears*12))
#        output_glac_wide_snowpack = np.zeros((grid_modelparameters.shape[0], glacier_gcm_temp.shape[0] - input.spinupyears*12))
#        output_glac_wide_area_annual = np.zeros((grid_modelparameters.shape[0], annual_columns.shape[0] - input.spinupyears + 1))
#        output_glac_wide_volume_annual = np.zeros((grid_modelparameters.shape[0], annual_columns.shape[0] - input.spinupyears + 1))
#        output_glac_wide_ELA_annual = np.zeros((grid_modelparameters.shape[0], annual_columns.shape[0] - input.spinupyears))
#        # Loop through each set of parameters
#        for grid_round in range(grid_modelparameters.shape[0]):
#            # Set model parameters
#            modelparameters = grid_modelparameters[grid_round, :]
#            # Reset initial parameters
#            glacier_area_t0 = main_glac_hyps.iloc[glac,:].values.astype(float)   
#            icethickness_t0 = main_glac_icethickness.iloc[glac,:].values.astype(float)
#            width_t0 = main_glac_width.iloc[glac,:].values.astype(float)
#            # Run the mass balance function (spinup years have been removed from output)
#            (glac_wide_massbaltotal, glac_wide_runoff, glac_wide_snowline, glac_wide_snowpack, glac_wide_area_annual, 
#             glac_wide_volume_annual, glac_wide_ELA_annual) = (
#                     massbalance.runmassbalance(glac, modelparameters, glacier_rgi_table, glacier_area_t0, 
#                                                icethickness_t0, width_t0, elev_bins, glacier_gcm_temp, glacier_gcm_prec, 
#                                                glacier_gcm_elev, glacier_gcm_lrgcm, glacier_gcm_lrglac, 
#                                                dates_table, annual_columns, annual_divisor))
#            # Record desired output
#            output_glac_wide_massbaltotal[grid_round,:] = glac_wide_massbaltotal
#            output_glac_wide_runoff[grid_round,:] = glac_wide_runoff
#            output_glac_wide_snowline[grid_round,:] = glac_wide_snowline
#            output_glac_wide_snowpack[grid_round,:] = glac_wide_snowpack
#            output_glac_wide_area_annual[grid_round,:] = glac_wide_area_annual
#            output_glac_wide_volume_annual[grid_round,:] = glac_wide_volume_annual
#            output_glac_wide_ELA_annual[grid_round,:] = glac_wide_ELA_annual
#            # Write output to netcdf
#            output.netcdfwrite_calgridsearch(output_fullfilename, glac, glacier_rgi_table, 
#                     output_glac_wide_massbaltotal, output_glac_wide_runoff, output_glac_wide_snowline, 
#                     output_glac_wide_snowpack, output_glac_wide_area_annual, output_glac_wide_volume_annual, 
#                     output_glac_wide_ELA_annual)
##    # Single run
##    elif input.option_calibration == 2:
##        # Select model parameters
##        modelparameters = main_glac_modelparams.loc[glac,:].values
##        # Run the mass balance function (spinup years have been removed from output)
##        (glac_bin_temp, glac_bin_prec, glac_bin_acc, glac_bin_refreeze, glac_bin_snowpack, glac_bin_melt, 
##         glac_bin_frontalablation, glac_bin_massbalclim, glac_bin_massbalclim_annual, glac_bin_area_annual, 
##         glac_bin_icethickness_annual, glac_bin_width_annual, glac_bin_surfacetype_annual, glac_wide_massbaltotal, 
##         glac_wide_runoff, glac_wide_snowline, glac_wide_snowpack, glac_wide_area_annual, glac_wide_volume_annual, 
##         glac_wide_ELA_annual) = (
##                 massbalance.runmassbalance(glac, modelparameters, glacier_rgi_table, glacier_area_t0, 
##                                            icethickness_t0, width_t0, elev_bins, glacier_gcm_temp, glacier_gcm_prec, 
##                                            glacier_gcm_elev, glacier_gcm_lrgcm, glacier_gcm_lrglac, 
##                                            dates_table, annual_columns, annual_divisor))
##        # Output for Shane: Measured mass balance, Measured mass balance uncertainty, modeled mass balance, mass balance difference
##        print(glacier_rgi_table.loc[input.massbal_colname], glacier_rgi_table.loc[input.massbal_uncertainty_colname], 
##              glac_wide_massbalclim_mwea, massbal_difference)
##            # Compare calibration data
##            # Column index for start and end year based on dates of geodetic mass balance observations
##            massbal_idx_start = (glacier_rgi_table.loc[input.massbal_time1] - input.startyear).astype(int)
##            massbal_idx_end = (massbal_idx_start + glacier_rgi_table.loc[input.massbal_time2] - 
##                               glacier_rgi_table.loc[input.massbal_time1] + 1).astype(int)
##            massbal_years = massbal_idx_end - massbal_idx_start
##            # Average annual glacier-wide mass balance [m w.e.]
##            glac_wide_massbalclim_mwea = ((glac_bin_massbalclim_annual[:, massbal_idx_start:massbal_idx_end] *
##                                           glac_bin_area_annual[:, massbal_idx_start:massbal_idx_end]).sum() /
##                                           glacier_area_t0.sum() / massbal_years)
##            #  units: m w.e. based on initial area
##            # Difference between geodetic and modeled mass balance
##            massbal_difference = abs(glacier_rgi_table[input.massbal_colname] - glac_wide_massbalclim_mwea)
##            # Output: Measured mass balance, Measured mass balance uncertainty, modeled mass balance, mass balance difference
##            print(glacier_rgi_table.loc[input.massbal_colname], glacier_rgi_table.loc[input.massbal_uncertainty_colname], 
##                  glac_wide_massbalclim_mwea, massbal_difference)
#
#timeelapsed_step1 = timeit.default_timer() - timestart_step1
#print('\ntime:', timeelapsed_step1, "s\n")
#
###%%=== Model testing ===============================================================================
##import netCDF4 as nc
#output = nc.Dataset(input.main_directory + '/../Output/calibration_gridsearchcoarse_R15_20180324.nc', 'r+')
