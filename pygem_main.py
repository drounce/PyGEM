#%% ###################################################################################################################
"""
Python Glacier Evolution Model "PyGEM" V1.0
Prepared by David Rounce with support from Regine Hock.
This work was funded under the NASA-ROSES program (grant no. NNX17AB27G).

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
from datetime import datetime
import os # os is used with re to find name matches
#import re # see os
import xarray as xr
import netCDF4 as nc
from time import strftime
import timeit
from scipy.optimize import minimize
from scipy.stats import linregress
import matplotlib.pyplot as plt
import cartopy

#========== IMPORT INPUT AND FUNCTIONS FROM MODULES ===================================================================
import pygem_input as input
import pygemfxns_modelsetup as modelsetup
import pygemfxns_climate as climate
import pygemfxns_massbalance as massbalance
import pygemfxns_output as output

#%% ===== LOAD GLACIER DATA ===================================================================================================
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
dates_table, start_date, end_date = modelsetup.datesmodelrun()
# Quality control - if ice thickness = 0, glacier area = 0 (problem identified by glacier RGIV6-15.00016 on 03/06/2018)
main_glac_hyps[main_glac_icethickness == 0] = 0

#%%=== LOAD CLIMATE DATA ================================================================================
if input.option_gcm_downscale == 1:
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
elif input.option_gcm_downscale == 2:
    # Import air temperature, precipitation, and elevation from pre-processed csv files for a given region
    #  this simply saves time from re-running the fxns above
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

#%%=== STEP FOUR: CALIBRATION =========================================================================================
timestart_step4 = timeit.default_timer()
if input.option_calibration == 1:
    #----- ENTER CALIBRATION RUN --------------------------------------------------------------------------------------
    # [INSERT REGIONAL LOOP HERE] if want to do all regions at the same time.  Separate netcdf files will be generated
    #  for each loop to reduce file size and make files easier to read/share
    regionO1_number = input.rgi_regionsO1[0]
    # Create output netcdf file
#    if input.output_package != 0:
#        output.netcdfcreate(regionO1_number, main_glac_hyps, dates_table, annual_columns)
    
    # CREATE A SEPARATE OUTPUT FOR CALIBRATION with only data relevant to calibration
    #   - annual glacier-wide massbal, area, ice thickness, snowline
    # Model parameter output
    main_glac_modelparamsopt = np.zeros((main_glac_rgi.shape[0], 8))
    main_glac_massbal_compare = np.zeros((main_glac_rgi.shape[0],4))

    for glac in range(main_glac_rgi.shape[0]):
#    for glac in range(25):
#    for glac in [0]:
        print(main_glac_rgi.loc[glac,'RGIId'])

        # Set model parameters
        modelparameters = [input.lrgcm, input.lrglac, input.precfactor, input.precgrad, input.ddfsnow, input.ddfice, 
                           input.tempsnow, input.tempchange]
        # Select subset of variables to reduce the amount of data being passed to the function
        glacier_rgi_table = main_glac_rgi.loc[glac, :]
        glacier_gcm_elev = main_glac_gcmelev[main_glac_rgi.loc[glac,'O1Index']]
        glacier_gcm_prec = main_glac_gcmprec[main_glac_rgi.loc[glac,'O1Index'],:]
        glacier_gcm_temp = main_glac_gcmtemp[main_glac_rgi.loc[glac,'O1Index'],:]
        glacier_gcm_lrgcm = main_glac_gcmlapserate[main_glac_rgi.loc[glac,'O1Index']]
        glacier_gcm_lrglac = glacier_gcm_lrgcm.copy()
        glacier_area_t0 = main_glac_hyps.iloc[glac,:].values.astype(float)   
        # Inclusion of ice thickness and width, i.e., loading values may be only required for Huss mass redistribution!
        icethickness_t0 = main_glac_icethickness.iloc[glac,:].values.astype(float)
        width_t0 = main_glac_width.iloc[glac,:].values.astype(float)
        
        (glac_wide_massbaltotal, glac_wide_runoff, glac_wide_snowline, glac_wide_snowpack, glac_wide_area_annual, 
         glac_wide_volume_annual, glac_wide_ELA_annual) = (
            massbalance.runmassbalance(modelparameters, glacier_rgi_table, glacier_area_t0, icethickness_t0, width_t0, 
                                       elev_bins, glacier_gcm_temp, glacier_gcm_prec, glacier_gcm_elev, glacier_gcm_lrgcm, 
                                       glacier_gcm_lrglac, dates_table))
        # Compare calibration data
        # Column index for start and end year based on dates of geodetic mass balance observations
        massbal_idx_start = int(glacier_rgi_table.loc[input.massbal_time1] - input.startyear)
        massbal_idx_end = int(massbal_idx_start + glacier_rgi_table.loc[input.massbal_time2] - 
                              glacier_rgi_table.loc[input.massbal_time1] + 1)
        # Annual glacier-wide mass balance [m w.e.]
        glac_wide_massbaltotal_annual = np.sum(glac_wide_massbaltotal.reshape(-1,12), axis=1)
        # Average annual glacier-wide mass balance [m w.e.a.]
        glac_wide_massbaltotal_annual_avg = glac_wide_massbaltotal_annual[massbal_idx_start:massbal_idx_end].mean()
        #  units: m w.e. based on initial area
        # Difference between geodetic and modeled mass balance
        massbal_difference = abs(glacier_rgi_table[input.massbal_colname] - glac_wide_massbaltotal_annual_avg)
        
        # Record the calibration round
        calround = 0
        
        # Run calibration only for glaciers that have calibration data 
        if np.isnan(main_glac_rgi.loc[glac,input.massbal_colname]) == False:
            # OPTIMIZATION FUNCTION: Define the function that you are trying to minimize
            #  - modelparameters are the parameters that will be optimized
            #  - return value is the value is the value used to run the optimization
            def objective(modelparameters):
                (glac_wide_massbaltotal, glac_wide_runoff, glac_wide_snowline, glac_wide_snowpack, glac_wide_area_annual, 
                 glac_wide_volume_annual, glac_wide_ELA_annual) = (
                    massbalance.runmassbalance(modelparameters, glacier_rgi_table, glacier_area_t0, icethickness_t0, width_t0, 
                                               elev_bins, glacier_gcm_temp, glacier_gcm_prec, glacier_gcm_elev, glacier_gcm_lrgcm, 
                                               glacier_gcm_lrglac, dates_table))
                # Compare calibration data
                # Column index for start and end year based on dates of geodetic mass balance observations
                massbal_idx_start = int(glacier_rgi_table.loc[input.massbal_time1] - input.startyear)
                massbal_idx_end = int(massbal_idx_start + glacier_rgi_table.loc[input.massbal_time2] - 
                                      glacier_rgi_table.loc[input.massbal_time1] + 1)
                # Annual glacier-wide mass balance [m w.e.]
                glac_wide_massbaltotal_annual = np.sum(glac_wide_massbaltotal.reshape(-1,12), axis=1)
                # Average annual glacier-wide mass balance [m w.e.a.]
                glac_wide_massbaltotal_annual_avg = glac_wide_massbaltotal_annual[massbal_idx_start:massbal_idx_end].mean()
                #  units: m w.e. based on initial area
                # Difference between geodetic and modeled mass balance
                massbal_difference = abs(glacier_rgi_table[input.massbal_colname] - glac_wide_massbaltotal_annual_avg)
                return massbal_difference
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
#            lrgcm_bnds = (-0.008,-0.004)
#            lrglac_bnds = (-0.008,-0.004)
#            precfactor_bnds = (0.95,1.25)
#            precgrad_bnds = (0.0001,0.00025)
#            ddfsnow_bnds = (0.0036, 0.0046)
#            #  Braithwaite (2008)
#            ddfice_bnds = (ddfsnow_bnds[0]/input.ddfsnow_iceratio, ddfsnow_bnds[1]/input.ddfsnow_iceratio)
#            tempsnow_bnds = (0,2) 
#            tempchange_bnds = (-2,2)
            
            lrgcm_bnds = (-0.008,-0.004)
            lrglac_bnds = (-0.008,-0.004)
            precfactor_bnds = (0.9,1.25)
            precgrad_bnds = (0.0001,0.0005)
            ddfsnow_bnds = (0.0026, 0.0051)
            #  Braithwaite (2008)
            ddfice_bnds = (ddfsnow_bnds[0]/input.ddfsnow_iceratio, ddfsnow_bnds[1]/input.ddfsnow_iceratio)
            tempsnow_bnds = (0,2) 
            tempchange_bnds = (-2,2)
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
            # Re-run the optimized parameters in order to see the mass balance
            (glac_wide_massbaltotal, glac_wide_runoff, glac_wide_snowline, glac_wide_snowpack, glac_wide_area_annual, 
             glac_wide_volume_annual, glac_wide_ELA_annual) = (
                massbalance.runmassbalance(main_glac_modelparamsopt[glac], glacier_rgi_table, glacier_area_t0, icethickness_t0, width_t0, 
                                           elev_bins, glacier_gcm_temp, glacier_gcm_prec, glacier_gcm_elev, glacier_gcm_lrgcm, 
                                           glacier_gcm_lrglac, dates_table))
            # Compare calibration data
            # Column index for start and end year based on dates of geodetic mass balance observations
            massbal_idx_start = int(glacier_rgi_table.loc[input.massbal_time1] - input.startyear)
            massbal_idx_end = int(massbal_idx_start + glacier_rgi_table.loc[input.massbal_time2] - 
                                  glacier_rgi_table.loc[input.massbal_time1] + 1)
            # Annual glacier-wide mass balance [m w.e.]
            glac_wide_massbaltotal_annual = np.sum(glac_wide_massbaltotal.reshape(-1,12), axis=1)
            # Average annual glacier-wide mass balance [m w.e.a.]
            glac_wide_massbaltotal_annual_avg = glac_wide_massbaltotal_annual[massbal_idx_start:massbal_idx_end].mean()
            #  units: m w.e. based on initial area
            # Difference between geodetic and modeled mass balance
            massbal_difference = abs(glacier_rgi_table[input.massbal_colname] - glac_wide_massbaltotal_annual_avg)
            main_glac_massbal_compare[glac] = [glac_wide_massbaltotal_annual_avg, glacier_rgi_table.loc[input.massbal_colname], 
                                               massbal_difference, calround]
            print('precfactor:', main_glac_modelparamsopt[glac,2])
            print('precgrad:', main_glac_modelparamsopt[glac,3])
            print('ddfsnow:', main_glac_modelparamsopt[glac,4])
            print('tempchange:', main_glac_modelparamsopt[glac,7])
            print(main_glac_massbal_compare[glac], '\n')
 
#            # OPTIMIZATION ROUND #2: if tolerance not reached, increase bounds
#            if massbal_difference > input.massbal_tolerance:
#                # Constraints
#                cons = [con_lrgcm, con_lrglac, con_precgrad, con_ddficefxsnow, con_tempsnow]
#                # Bounds
#                lrgcm_bnds = (-0.008,-0.004)
#                lrglac_bnds = (-0.008,-0.004)
#                precfactor_bnds = (0.9,1.5)
#                precgrad_bnds = (0.0001,0.00025)
#                ddfsnow_bnds = (0.0031, 0.0051)
#                ddfice_bnds = (ddfsnow_bnds[0]/input.ddfsnow_iceratio, ddfsnow_bnds[1]/input.ddfsnow_iceratio)
#                tempsnow_bnds = (0,2) 
#                tempchange_bnds = (-5,5)
#                modelparameters_bnds = (lrgcm_bnds, lrglac_bnds, precfactor_bnds, precgrad_bnds, ddfsnow_bnds, 
#                                        ddfice_bnds, tempsnow_bnds, tempchange_bnds)  
#                # Run optimization
#                modelparameters_opt = minimize(objective, main_glac_modelparamsopt[glac], method='SLSQP', 
#                                               bounds=modelparameters_bnds, constraints=cons, tol=1e-3)
#                # Record the calibration round
#                calround = calround + 1
#                # Record the optimized parameters
#                main_glac_modelparamsopt[glac] = modelparameters_opt.x
#                # Re-run the optimized parameters in order to see the mass balance
#                (glac_bin_temp, glac_bin_prec, glac_bin_acc, glac_bin_refreeze, glac_bin_snowpack, glac_bin_melt, 
#                 glac_bin_frontalablation, glac_bin_massbalclim, glac_bin_massbalclim_annual, glac_bin_area_annual, 
#                 glac_bin_icethickness_annual, glac_bin_width_annual, glac_bin_surfacetype_annual) = (
#                         massbalance.runmassbalance(glac, main_glac_modelparamsopt[glac], regionO1_number, 
#                                                    glacier_rgi_table, glacier_area_t0, icethickness_t0, width_t0, 
#                                                    glacier_gcm_temp, glacier_gcm_prec, glacier_gcm_elev, 
#                                                    glacier_gcm_lrgcm, glacier_gcm_lrglac, elev_bins, dates_table, 
#                                                    annual_columns, annual_divisor))
#                # Column index for start and end year based on dates of geodetic mass balance observations
#                massbal_idx_start = (main_glac_calmassbal[glac,1] - input.startyear).astype(int)
#                massbal_idx_end = (massbal_idx_start + main_glac_calmassbal[glac,2] - 
#                                   main_glac_calmassbal[glac,1] + 1).astype(int)
#                massbal_years = massbal_idx_end - massbal_idx_start
#                # Average annual glacier-wide mass balance [m w.e.]
#                glac_wide_massbalclim_mwea = ((glac_bin_massbalclim_annual[:, massbal_idx_start:massbal_idx_end] * 
#                                               glac_bin_area_annual[:, massbal_idx_start:massbal_idx_end]).sum() / 
#                                               glacier_area_t0.sum() / massbal_years)
#                massbal_difference = abs(main_glac_calmassbal[glac,0] - glac_wide_massbalclim_mwea)
#                main_glac_massbal_compare[glac] = [glac_wide_massbalclim_mwea, main_glac_calmassbal[glac,0], 
#                                                   massbal_difference, calround]
#                print('precfactor:', main_glac_modelparamsopt[glac,2])
#                print('ddfsnow:', main_glac_modelparamsopt[glac,4])
#                print('tempchange:', main_glac_modelparamsopt[glac,7])
#                print(main_glac_massbal_compare[glac], '\n')
#                
#            # OPTIMIZATION ROUND #3: if tolerance not reached, increase bounds again
#            if massbal_difference > input.massbal_tolerance:
#                # Constraints
#                cons = [con_lrgcm, con_lrglac, con_precgrad, con_ddficefxsnow, con_tempsnow]
#                # Bounds
#                lrgcm_bnds = (-0.008,-0.004)
#                lrglac_bnds = (-0.008,-0.004)
#                precfactor_bnds = (0.8,2)
#                precgrad_bnds = (0.0001,0.00025)
#                ddfsnow_bnds = (0.0026, 0.0056)
#                ddfice_bnds = (ddfsnow_bnds[0]/input.ddfsnow_iceratio, ddfsnow_bnds[1]/input.ddfsnow_iceratio)
#                tempsnow_bnds = (0,2) 
#                tempchange_bnds = (-10,10)
#                modelparameters_bnds = (lrgcm_bnds, lrglac_bnds, precfactor_bnds, precgrad_bnds, ddfsnow_bnds, 
#                                        ddfice_bnds, tempsnow_bnds, tempchange_bnds)  
#                # Run optimization
#                modelparameters_opt = minimize(objective, main_glac_modelparamsopt[glac], method='SLSQP', 
#                                               bounds=modelparameters_bnds, constraints=cons, tol=1e-4)
#                #  requires higher tolerance due to the increase in parameters being optimized
#                # Record the calibration round
#                calround = calround + 1
#                # Record the optimized parameters
#                main_glac_modelparamsopt[glac] = modelparameters_opt.x
#                # Re-run the optimized parameters in order to see the mass balance
#                (glac_bin_temp, glac_bin_prec, glac_bin_acc, glac_bin_refreeze, glac_bin_snowpack, glac_bin_melt, 
#                 glac_bin_frontalablation, glac_bin_massbalclim, glac_bin_massbalclim_annual, glac_bin_area_annual, 
#                 glac_bin_icethickness_annual, glac_bin_width_annual, glac_bin_surfacetype_annual) = (
#                         massbalance.runmassbalance(glac, main_glac_modelparamsopt[glac], regionO1_number, 
#                                                    glacier_rgi_table, glacier_area_t0, icethickness_t0, width_t0, 
#                                                    glacier_gcm_temp, glacier_gcm_prec, glacier_gcm_elev, 
#                                                    glacier_gcm_lrgcm, glacier_gcm_lrglac, elev_bins, dates_table, 
#                                                    annual_columns, annual_divisor))
#                # Column index for start and end year based on dates of geodetic mass balance observations
#                massbal_idx_start = (main_glac_calmassbal[glac,1] - input.startyear).astype(int)
#                massbal_idx_end = (massbal_idx_start + main_glac_calmassbal[glac,2] - 
#                                   main_glac_calmassbal[glac,1] + 1).astype(int)
#                massbal_years = massbal_idx_end - massbal_idx_start
#                # Average annual glacier-wide mass balance [m w.e.]
#                glac_wide_massbalclim_mwea = ((glac_bin_massbalclim_annual[:, massbal_idx_start:massbal_idx_end] * 
#                                               glac_bin_area_annual[:, massbal_idx_start:massbal_idx_end]).sum() / 
#                                               glacier_area_t0.sum() / massbal_years)
#                massbal_difference = abs(main_glac_calmassbal[glac,0] - glac_wide_massbalclim_mwea)
#                main_glac_massbal_compare[glac] = [glac_wide_massbalclim_mwea, main_glac_calmassbal[glac,0], 
#                                                   massbal_difference, calround]
#                print('precfactor:', main_glac_modelparamsopt[glac,2])
#                print('ddfsnow:', main_glac_modelparamsopt[glac,4])
#                print('tempchange:', main_glac_modelparamsopt[glac,7])
#                print(main_glac_massbal_compare[glac], '\n')
#        else:
#        # if calibration data not available for a glacier, then insert NaN into calibration output
#            main_glac_modelparamsopt[glac] = float('NaN')
#            main_glac_massbal_compare[glac] = float('NaN')
            
        # Output calibration results to .csv file
        #  pandas dataframe used instead of numpy arrays here, so column headings can be exported
        main_glac_caloutput = main_glac_rgi.copy()
        main_glac_caloutput['MB_model_mwea'] = main_glac_massbal_compare[:,0] 
        main_glac_caloutput['MB_geodetic_mwea'] = main_glac_massbal_compare[:,1] 
        main_glac_caloutput['MB_difference_mwea'] = main_glac_massbal_compare[:,2]
        main_glac_caloutput['calround'] = main_glac_massbal_compare[:,3]
        main_glac_caloutput['lrgcm'] = main_glac_modelparamsopt[:,0] 
        main_glac_caloutput['lrglac'] = main_glac_modelparamsopt[:,1] 
        main_glac_caloutput['precfactor'] = main_glac_modelparamsopt[:,2] 
        main_glac_caloutput['precgrad'] = main_glac_modelparamsopt[:,3] 
        main_glac_caloutput['ddfsnow'] = main_glac_modelparamsopt[:,4] 
        main_glac_caloutput['ddfice'] = main_glac_modelparamsopt[:,5] 
        main_glac_caloutput['tempsnow'] = main_glac_modelparamsopt[:,6]
        main_glac_caloutput['tempchange'] = main_glac_modelparamsopt[:,7]
        # export csv
        cal_output_fullfile = (input.output_filepath + input.calibrationcsv_filenameprefix + 'R' + str(regionO1_number) 
                               + '_' + str(strftime("%Y%m%d")) + '.csv')
        main_glac_caloutput.to_csv(cal_output_fullfile)
        

#elif input.option_calibration == 2:
#    #----- IMPORT CALIBRATION DATASETS --------------------------------------------------------------------------------
#    # Import geodetic mass balance from David Shean
#    main_glac_calmassbal = modelsetup.selectcalibrationdata(main_glac_rgi)
#
#    #----- ENTER GRID SEARCH RUN --------------------------------------------------------------------------------------
#    # [INSERT REGIONAL LOOP HERE] if want to do all regions at the same time.  Separate netcdf files will be generated
#    #  for each loop to reduce file size and make files easier to read/share
#    regionO1_number = input.rgi_regionsO1[0]
#    # Create output netcdf file
##    if input.output_package != 0:
##        output.netcdfcreate(regionO1_number, main_glac_hyps, dates_table, annual_columns)
#    
#    # Create grid of all parameter sets
#    grid_precfactor = np.arange(0.75, 2, 0.25)
#    grid_tempbias = np.arange(-4, 6, 2)
#    grid_ddfsnow = np.arange(0.0031, 0.0056, 0.0005)
#    grid_precgrad = np.arange(0.001, 0.007, 0.002)
#    
#    grid_modelparameters = np.zeros((grid_precfactor.shape[0] * grid_tempbias.shape[0] * grid_ddfsnow.shape[0] * 
#                                     grid_precgrad.shape[0],8))
#    grid_count = 0
#    for n_precfactor in range(grid_precfactor.shape[0]):
#        for n_tempbias in range(grid_tempbias.shape[0]):
#            for n_ddfsnow in range(grid_ddfsnow.shape[0]):
#                for n_precgrad in range(grid_precgrad.shape[0]):
#                    # Set grid of model parameters
#                    grid_modelparameters[grid_count,:] = [input.lrgcm, input.lrglac, grid_precfactor[n_precfactor], 
#                          grid_precgrad[n_precgrad], grid_ddfsnow[n_ddfsnow], grid_ddfsnow[n_ddfsnow] / 0.7, 
#                          input.tempsnow, grid_tempbias[n_tempbias]]
#                    grid_count = grid_count + 1
#                    
##    # Create netcdf HERE
##    # Netcdf file path and name
##    filename = input.calibrationnetcdf_filenameprefix + str(regionO1_number) + '_' + str(strftime("%Y%m%d")) + '.nc'
##    fullfile = input.output_filepath + filename
##    # Create netcdf file ('w' will overwrite existing files, 'r+' will open existing file to write)
##    netcdf_output = nc.Dataset(fullfile, 'w', format='NETCDF4')
##    # Global attributes
##    netcdf_output.description = 'Results from glacier evolution model'
##    netcdf_output.history = 'Created ' + str(strftime("%Y-%m-%d %H:%M:%S"))
##    netcdf_output.source = 'Python Glacier Evolution Model'
##    # Dimensions
##    glacier = netcdf_output.createDimension('glacier', None)
##    binelev = netcdf_output.createDimension('binelev', main_glac_hyps.shape[1])
##    if input.timestep == 'monthly':
##        time = netcdf_output.createDimension('time', dates_table.shape[0] - input.spinupyears * 12)
##    year = netcdf_output.createDimension('year', annual_columns.shape[0] - input.spinupyears)
##    year_plus1 = netcdf_output.createDimension('year_plus1', annual_columns.shape[0] - input.spinupyears + 1)
##    gridround = netcdf_output.createDimension('gridround', grid_modelparameters.shape[0])
##    glacierinfo = netcdf_output.createDimension('glacierinfo', 3)
##    # Variables associated with dimensions 
##    glaciers = netcdf_output.createVariable('glacier', np.int32, ('glacier',))
##    glaciers.long_name = "glacier number associated with model run"
##    glaciers.standard_name = "GlacNo"
##    glaciers.comment = ("The glacier number is defined for each model run. The user should look at the main_glac_rgi"
##                           + " table to determine the RGIID or other information regarding this particular glacier.")
##    binelevs = netcdf_output.createVariable('binelev', np.int32, ('binelev',))
##    binelevs.standard_name = "center bin_elevation"
##    binelevs.units = "m a.s.l."
##    binelevs[:] = main_glac_hyps.columns.values
##    binelevs.comment = ("binelev are the bin elevations that were used for the model run.")
##    times = netcdf_output.createVariable('time', np.float64, ('time',))
##    times.standard_name = "date"
##    times.units = "days since 1900-01-01 00:00:00"
##    times.calendar = "gregorian"
##    if input.timestep == 'monthly':
##        times[:] = (nc.date2num(dates_table.loc[input.spinupyears*12:dates_table.shape[0]+1,'date'].astype(datetime), 
##                                units = times.units, calendar = times.calendar))
##    years = netcdf_output.createVariable('year', np.int32, ('year',))
##    years.standard_name = "year"
##    if input.option_wateryear == 1:
##        years.units = 'water year'
##    elif input.option_wateryear == 0:
##        years.units = 'calendar year'
##    years[:] = annual_columns[input.spinupyears:annual_columns.shape[0]]
##    # years_plus1 adds an additional year such that the change in glacier dimensions (area, etc.) is recorded
##    years_plus1 = netcdf_output.createVariable('year_plus1', np.int32, ('year_plus1',))
##    years_plus1.standard_name = "year with additional year to record glacier dimension changes"
##    if input.option_wateryear == 1:
##        years_plus1.units = 'water year'
##    elif input.option_wateryear == 0:
##        years_plus1.units = 'calendar year'
##    years_plus1 = np.concatenate((annual_columns[input.spinupyears:annual_columns.shape[0]], 
##                                  np.array([annual_columns[annual_columns.shape[0]-1]+1])))
##    gridrounds = netcdf_output.createVariable('gridround', np.int32, ('gridround',))
##    gridrounds.long_name = "number associated with the calibration grid search"
##    glacierinfoheader = netcdf_output.createVariable('glacierinfoheader', str, ('glacierinfo',))
##    glacierinfoheader.standard_name = "information about each glacier from main_glac_rgi"
##    glacierinfoheader[:] = np.array(['RGIID','lat','lon'])
##    glacierinfo = netcdf_output.createVariable('glacierinfo',str,('glacier','glacierinfo',))
##    
##    # Variables associated with the output
##    #  monthly glacier-wide variables (massbal_total, runoff, snowline, snowpack)
##    #  annual glacier-wide variables (area, volume, ELA)
##    massbaltotal_glac_monthly = netcdf_output.createVariable('massbaltotal_glac_monthly', np.float64, ('glacier', 'gridround', 'time'))
##    massbaltotal_glac_monthly.standard_name = "glacier-wide total mass balance"
##    massbaltotal_glac_monthly.units = "m w.e."
##    massbaltotal_glac_monthly.comment = ("total mass balance is the sum of the climatic mass balance and frontal "
##                                         + "ablation.") 
##    runoff_glac_monthly = netcdf_output.createVariable('runoff_glac_monthly', np.float64, ('glacier', 'gridround', 'time'))
##    runoff_glac_monthly.standard_name = "glacier runoff"
##    runoff_glac_monthly.units = "m**3"
##    runoff_glac_monthly.comment = "runoff from the glacier terminus, which moves over time"
##    snowline_glac_monthly = netcdf_output.createVariable('snowline_glac_monthly', np.float64, ('glacier', 'gridround', 'time'))
##    snowline_glac_monthly.standard_name = "transient snowline"
##    snowline_glac_monthly.units = "m a.s.l."
##    snowline_glac_monthly.comment = "transient snowline is the line separating the snow from ice/firn"
##    snowpack_glac_monthly = netcdf_output.createVariable('snowpack_glac_monthly', np.float64, ('glacier', 'gridround', 'time'))
##    snowpack_glac_monthly.standard_name = "snowpack volume"
##    snowpack_glac_monthly.units = "km**3 w.e."
##    snowpack_glac_monthly.comment = "m w.e. multiplied by the area converted to km**3"
##    area_glac_annual = netcdf_output.createVariable('area_glac_annual', np.float64, ('glacier', 'gridround', 'year_plus1'))
##    area_glac_annual.standard_name = "glacier area"
##    area_glac_annual.units = "km**2"
##    area_glac_annual.comment = "the area that was used for the duration of the year"
##    volume_glac_annual = netcdf_output.createVariable('volume_glac_annual', np.float64, ('glacier', 'gridround', 'year_plus1'))
##    volume_glac_annual.standard_name = "glacier volume"
##    volume_glac_annual.units = "km**3 ice"
##    volume_glac_annual.comment = "the volume based on area and ice thickness used for that year"
##    ELA_glac_annual = netcdf_output.createVariable('ELA_glac_annual', np.float64, ('glacier', 'gridround', 'year'))
##    ELA_glac_annual.standard_name = "annual equilibrium line altitude"
##    ELA_glac_annual.units = "m a.s.l."
##    ELA_glac_annual.comment = "equilibrium line altitude is the elevation where the climatic mass balance is zero"
##    netcdf_output.close()
#    
#     # Loop through the glaciers for each grid of parameter sets
#    for glac in range(main_glac_rgi.shape[0]):
##    for glac in range(25):
##    for glac in [1]:
#        print(glac)
#        # Run calibration only for glaciers that have calibration data 
#        if np.isnan(main_glac_calmassbal[glac,0]) == False:
#            # Select subset of variables to reduce the amount of data being passed to the function
#            glacier_rgi_table = main_glac_rgi.loc[glac, :]
#            glacier_gcm_elev = main_glac_gcmelev[glac]
#            glacier_gcm_prec = main_glac_gcmprec[glac,:]
#            glacier_gcm_temp = main_glac_gcmtemp[glac,:]
#            # Set new output 
#            output_glac_wide_massbaltotal = np.zeros((grid_modelparameters.shape[0], glacier_gcm_temp.shape[0] - input.spinupyears*12))
#            output_glac_wide_runoff = np.zeros((grid_modelparameters.shape[0], glacier_gcm_temp.shape[0] - input.spinupyears*12))
#            output_glac_wide_snowline = np.zeros((grid_modelparameters.shape[0], glacier_gcm_temp.shape[0] - input.spinupyears*12))
#            output_glac_wide_snowpack = np.zeros((grid_modelparameters.shape[0], glacier_gcm_temp.shape[0] - input.spinupyears*12))
#            output_glac_wide_area_annual = np.zeros((grid_modelparameters.shape[0], annual_columns.shape[0] - input.spinupyears + 1))
#            output_glac_wide_volume_annual = np.zeros((grid_modelparameters.shape[0], annual_columns.shape[0] - input.spinupyears + 1))
#            output_glac_wide_ELA_annual = np.zeros((grid_modelparameters.shape[0], annual_columns.shape[0] - input.spinupyears))
#            # Loop through each set of parameters
#            for grid_round in range(grid_modelparameters.shape[0]):
#                # Set model parameters
#                modelparameters = grid_modelparameters[grid_round, :]
#                # Set lapse rate based on options
#                if input.option_lapserate_fromgcm == 1:
#                    glacier_gcm_lrgcm = main_glac_gcmlapserate[glac]
#                    glacier_gcm_lrglac = glacier_gcm_lrgcm.copy()
#                else:
#                    glacier_gcm_lrgcm = np.zeros(glacier_gcm_temp.shape) + modelparameters[0]
#                    glacier_gcm_lrglac = np.zeros(glacier_gcm_temp.shape) + modelparameters[1]
#                # Reset initial parameters
#                glacier_area_t0 = main_glac_hyps.iloc[glac,:].values.astype(float)   
#                icethickness_t0 = main_glac_icethickness.iloc[glac,:].values.astype(float)
#                width_t0 = main_glac_width.iloc[glac,:].values.astype(float)
#                # MASS BALANCE
#                # Run the mass balance function (spinup years have been removed from output)
#                (glac_bin_temp, glac_bin_prec, glac_bin_acc, glac_bin_refreeze, glac_bin_snowpack, glac_bin_melt, 
#                 glac_bin_frontalablation, glac_bin_massbalclim, glac_bin_massbalclim_annual, glac_bin_area_annual, 
#                 glac_bin_icethickness_annual, glac_bin_width_annual, glac_bin_surfacetype_annual) = (
#                         massbalance.runmassbalance(glac, modelparameters, regionO1_number, glacier_rgi_table, glacier_area_t0, 
#                                                    icethickness_t0, width_t0, glacier_gcm_temp, glacier_gcm_prec, 
#                                                    glacier_gcm_elev, glacier_gcm_lrgcm, glacier_gcm_lrglac, elev_bins, 
#                                                    dates_table, annual_columns, annual_divisor))
#                glac_wide_prec = np.zeros(glac_bin_temp.shape[1])
#                glac_wide_acc = np.zeros(glac_bin_temp.shape[1])
#                glac_wide_refreeze = np.zeros(glac_bin_temp.shape[1])
#                glac_wide_melt = np.zeros(glac_bin_temp.shape[1])
#                glac_wide_frontalablation = np.zeros(glac_bin_temp.shape[1])
#                # Compute desired output
#                glac_bin_area = glac_bin_area_annual[:,0:glac_bin_area_annual.shape[1]-1].repeat(12,axis=1)
#                glac_wide_area = glac_bin_area.sum(axis=0)
#                glac_wide_prec_mkm2 = (glac_bin_prec * glac_bin_area).sum(axis=0)
#                glac_wide_prec[glac_wide_prec_mkm2 > 0] = (glac_wide_prec_mkm2[glac_wide_prec_mkm2 > 0] / 
#                                                           glac_wide_area[glac_wide_prec_mkm2 > 0])
#                glac_wide_acc_mkm2 = (glac_bin_acc * glac_bin_area).sum(axis=0)
#                glac_wide_acc[glac_wide_acc_mkm2 > 0] = (glac_wide_acc_mkm2[glac_wide_acc_mkm2 > 0] / 
#                                                         glac_wide_area[glac_wide_acc_mkm2 > 0])
#                glac_wide_refreeze_mkm2 = (glac_bin_refreeze * glac_bin_area).sum(axis=0)
#                glac_wide_refreeze[glac_wide_refreeze_mkm2 > 0] = (glac_wide_refreeze_mkm2[glac_wide_refreeze_mkm2 > 0] / 
#                                                                   glac_wide_area[glac_wide_refreeze_mkm2 > 0])
#                glac_wide_melt_mkm2 = (glac_bin_melt * glac_bin_area).sum(axis=0)
#                glac_wide_melt[glac_wide_melt_mkm2 > 0] = (glac_wide_melt_mkm2[glac_wide_melt_mkm2 > 0] / 
#                                                           glac_wide_area[glac_wide_melt_mkm2 > 0])
#                glac_wide_frontalablation_mkm2 = (glac_bin_frontalablation * glac_bin_area).sum(axis=0)
#                glac_wide_frontalablation[glac_wide_frontalablation_mkm2 > 0] = (
#                        glac_wide_frontalablation_mkm2[glac_wide_frontalablation_mkm2 > 0] / 
#                        glac_wide_area[glac_wide_frontalablation_mkm2 > 0])
#                glac_wide_massbalclim = glac_wide_acc + glac_wide_refreeze - glac_wide_melt
#                glac_wide_massbaltotal = glac_wide_massbalclim - glac_wide_frontalablation
#                glac_wide_runoff = (glac_wide_prec + glac_wide_melt - glac_wide_refreeze) * glac_wide_area * (1000)**2
#                #  units: (m + m w.e. - m w.e.) * km**2 * (1000 m / 1 km)**2 = m**3
#                glac_wide_snowline = (glac_bin_snowpack > 0).argmax(axis=0)
#                glac_wide_snowline[glac_wide_snowline > 0] = (elev_bins[glac_wide_snowline[glac_wide_snowline > 0]] - 
#                                                              input.binsize/2)
#                glac_wide_area_annual = glac_bin_area_annual.sum(axis=0)
#                glac_wide_volume_annual = (glac_bin_area_annual * glac_bin_icethickness_annual / 1000).sum(axis=0)
#                glac_wide_ELA_annual = (glac_bin_massbalclim_annual > 0).argmax(axis=0)
#                glac_wide_ELA_annual[glac_wide_ELA_annual > 0] = (elev_bins[glac_wide_ELA_annual[glac_wide_ELA_annual > 0]] - 
#                                                                  input.binsize/2)
#                # Record desired output
#                output_glac_wide_massbaltotal[grid_round,:] = glac_wide_massbaltotal
#                output_glac_wide_runoff[grid_round,:] = glac_wide_runoff
#                output_glac_wide_snowline[grid_round,:] = glac_wide_snowline
#                output_glac_wide_snowpack[grid_round,:] = glac_wide_acc_mkm2 / 1000
#                output_glac_wide_area_annual[grid_round,:] = glac_wide_area_annual
#                output_glac_wide_volume_annual[grid_round,:] = glac_wide_volume_annual
#                output_glac_wide_ELA_annual[grid_round,:] = glac_wide_ELA_annual
#            # Netcdf file path and name
#            filename = input.calibrationnetcdf_filenameprefix + str(regionO1_number) + '_' + str(strftime("%Y%m%d")) + '.nc'
#            fullfile = input.output_filepath + filename
#            # Open netcdf file to write to existing file ('r+')
#            netcdf_output = nc.Dataset(fullfile, 'r+')
#            # Write variables to netcdf
#            netcdf_output.variables['glacierinfo'][glac,:] = np.array([glacier_rgi_table.loc['RGIId'],
#                    glacier_rgi_table.loc[input.lat_colname], glacier_rgi_table.loc[input.lon_colname]])
#            netcdf_output.variables['massbaltotal_glac_monthly'][glac,:,:] = output_glac_wide_massbaltotal
#            netcdf_output.variables['runoff_glac_monthly'][glac,:,:] = output_glac_wide_runoff
#            netcdf_output.variables['snowline_glac_monthly'][glac,:,:] = output_glac_wide_snowline
#            netcdf_output.variables['snowpack_glac_monthly'][glac,:,:] = output_glac_wide_snowpack
#            netcdf_output.variables['area_glac_annual'][glac,:,:] = output_glac_wide_area_annual
#            netcdf_output.variables['volume_glac_annual'][glac,:,:] = output_glac_wide_volume_annual
#            netcdf_output.variables['ELA_glac_annual'][glac,:,:] = output_glac_wide_ELA_annual
#            netcdf_output.close()
#        
#timeelapsed_step4 = timeit.default_timer() - timestart_step4
#print('Step 4 time:', timeelapsed_step4, "s\n")
#
##%%=== STEP FIVE: SIMULATION RUN ======================================================================================
#timestart_step5 = timeit.default_timer()
#
#if input.option_calibration == 0:
#    # [INSERT REGIONAL LOOP HERE] if want to do all regions at the same time.  Separate netcdf files will be generated
#    #  for each loop to reduce file size and make files easier to read/share
#    
#    regionO1_number = input.rgi_regionsO1[0]
#    # Create output netcdf file
#    if input.output_package != 0:
#        output.netcdfcreate(regionO1_number, main_glac_hyps, dates_table, annual_columns)
#        
#    # Load model parameters
#    if input.option_loadparameters == 1:
#        main_glac_modelparams = pd.read_csv(input.modelparams_filepath + input.modelparams_filename) 
#    else:
#        main_glac_modelparams = pd.DataFrame(np.repeat([input.lrgcm, input.lrglac, input.precfactor, input.precgrad, 
#            input.ddfsnow, input.ddfice, input.tempsnow, input.tempchange], main_glac_rgi.shape[0]).reshape(-1, 
#            main_glac_rgi.shape[0]).transpose(), columns=input.modelparams_colnames)
#    # Test range
#    #glac = 0
#    #prec_factor_low = 0.8
#    #prec_factor_high = 2.0
#    #prec_factor_step = 0.005
#    #prec_factor_range = np.arange(prec_factor_low, prec_factor_high + prec_factor_step, prec_factor_step)
#    #glac_wide_massbal_record = np.zeros(prec_factor_range.shape)
#    #for n in range(len(prec_factor_range)):
#    #    prec_factor = prec_factor_range[n]
#        
#    # ENTER GLACIER LOOP
#    for glac in range(main_glac_rgi.shape[0]):
##    for glac in [0]:
#        print(glac)
#        # Select subset of variables to reduce the amount of data being passed to the function
#        modelparameters = main_glac_modelparams.loc[glac,:].values
#        glacier_rgi_table = main_glac_rgi.loc[glac, :]
#        glacier_gcm_elev = main_glac_gcmelev[glac]
#        glacier_gcm_prec = main_glac_gcmprec[glac,:]
#        glacier_gcm_temp = main_glac_gcmtemp[glac,:]
#        if input.option_lapserate_fromgcm == 1:
#            glacier_gcm_lrgcm = main_glac_gcmlapserate[glac]
#            glacier_gcm_lrglac = glacier_gcm_lrgcm.copy()
#        else:
#            glacier_gcm_lrgcm = np.zeros(glacier_gcm_temp.shape) + modelparameters[0]
#            glacier_gcm_lrglac = np.zeros(glacier_gcm_temp.shape) + modelparameters[1]
#        glacier_area_t0 = main_glac_hyps.iloc[glac,:].values.astype(float)   
#        # Inclusion of ice thickness and width, i.e., loading values may be only required for Huss mass redistribution!
#        icethickness_t0 = main_glac_icethickness.iloc[glac,:].values.astype(float)
#        width_t0 = main_glac_width.iloc[glac,:].values.astype(float)
#        # MASS BALANCE
#        # Run the mass balance function (spinup years have been removed from output)
#        (glac_bin_temp, glac_bin_prec, glac_bin_acc, glac_bin_refreeze, glac_bin_snowpack, glac_bin_melt, 
#         glac_bin_frontalablation, glac_bin_massbalclim, glac_bin_massbalclim_annual, glac_bin_area_annual, 
#         glac_bin_icethickness_annual, glac_bin_width_annual, glac_bin_surfacetype_annual) = (
#                 massbalance.runmassbalance(glac, modelparameters, regionO1_number, glacier_rgi_table, glacier_area_t0, 
#                                            icethickness_t0, width_t0, glacier_gcm_temp, glacier_gcm_prec, 
#                                            glacier_gcm_elev, glacier_gcm_lrgcm, glacier_gcm_lrglac, elev_bins, 
#                                            dates_table, annual_columns, annual_divisor))
#        # OUTPUT: Record variables according to output package
#        #  must be done within glacier loop since the variables will be overwritten 
#        if input.output_package != 0:
#            output.netcdfwrite(regionO1_number, glac, modelparameters, glacier_rgi_table, elev_bins, glac_bin_temp, 
#                               glac_bin_prec, glac_bin_acc, glac_bin_refreeze, glac_bin_snowpack, glac_bin_melt, 
#                               glac_bin_frontalablation, glac_bin_massbalclim, glac_bin_massbalclim_annual, 
#                               glac_bin_area_annual, glac_bin_icethickness_annual, glac_bin_width_annual, 
#                               glac_bin_surfacetype_annual)
#            # POTENTIAL IMPROVEMENT: OPEN THE FILE WITHIN THE MAIN SCRIPT AND THEN GO INTO THE FUNCTION TO WRITE TO THE
#            #                        NETCDF FILE - THIS WAY NO LONGER HAVE TO OPEN AND CLOSE EACH TIME
#            
#timeelapsed_step5 = timeit.default_timer() - timestart_step5
#print('Step 5 time:', timeelapsed_step5, "s\n")
#
##%%=== Model testing ===============================================================================
##netcdf_output = nc.Dataset(input.main_directory + '/../Output/calibration_gridsearchcoarse_R15_20180319.nc', 'r+')
