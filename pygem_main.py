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

#%%======== DEVELOPER'S TO-DO LIST ====================================================================================
# > Output log file, i.e., file that states input parameters, date of model run, model options selected, 
#   and any errors that may have come up (e.g., precipitation corrected because negative value, etc.)

# ===== STEP ONE: Select glaciers included in model run ===============================================================
timestart_step1 = timeit.default_timer()
if input.option_glacier_selection == 1:
    # RGI glacier attributes
    main_glac_rgi = modelsetup.selectglaciersrgitable()
elif input.option_glacier_selection == 2:
    print('\n\tMODEL ERROR (selectglaciersrgi): this option to use shapefiles to select glaciers has not been coded '
          '\n\tyet. Please choose an option that exists. Exiting model run.\n')
    exit()
else:
    # Should add options to make regions consistent with Brun et al. (2017), which used ASTER DEMs to get mass 
    # balance of 92% of the HMA glaciers.
    print('\n\tModel Error (selectglaciersrgi): please choose an option that exists for selecting glaciers.'
          '\n\tExiting model run.\n')
    exit()
timeelapsed_step1 = timeit.default_timer() - timestart_step1
print('Step 1 time:', timeelapsed_step1, "s\n")

#%%=== STEP TWO: HYPSOMETRY, ICE THICKNESS, MODEL TIME FRAME, SURFACE TYPE ============================================
timestart_step2 = timeit.default_timer()
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
# Print time elapsed
timeelapsed_step2 = timeit.default_timer() - timestart_step2
print('Step 2 time:', timeelapsed_step2, "s\n")

#%%=== STEP THREE: IMPORT CLIMATE DATA ================================================================================
timestart_step3 = timeit.default_timer()
#  Downscale option 1 takes 10 minutes to run for Region 15 ()
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
    main_glac_gcmtemp = np.genfromtxt(input.gcm_filepath_var + 'csv_ERAInterim_temp_19952015_15_SouthAsiaEast.csv', 
                                      delimiter=',')
    main_glac_gcmprec = np.genfromtxt(input.gcm_filepath_var + 'csv_ERAInterim_prec_19952015_15_SouthAsiaEast.csv', 
                                      delimiter=',')
    main_glac_gcmelev = np.genfromtxt(input.gcm_filepath_var + 'csv_ERAInterim_elev_15_SouthAsiaEast.csv', 
                                      delimiter=',')
    # Lapse rates [degC m-1]  
    if input.option_lapserate_fromgcm == 1:
        main_glac_gcmlapserate = np.genfromtxt(input.gcm_filepath_var + 
                                               'csv_ERAInterim_lapserate_19952015_15_SouthAsiaEast.csv', delimiter=',')
else:
    print('\n\tModel Error: please choose an option that exists for downscaling climate data. Exiting model run now.\n')
    exit()
# Print time elapsed
timeelapsed_step3 = timeit.default_timer() - timestart_step3
print('Step 3 time:', timeelapsed_step3, "s\n")

#%%=== STEP FOUR: CALIBRATION =========================================================================================
timestart_step4 = timeit.default_timer()
if input.option_calibration == 1:

    #----- IMPORT CALIBRATION DATASETS --------------------------------------------------------------------------------
    # Import geodetic mass balance from David Shean
    main_glac_calmassbal = modelsetup.selectcalibrationdata(main_glac_rgi)

    ## add start and end date
    #ds = pd.read_csv(input.cal_mb_filepath + input.cal_mb_filename)
    #main_glac_calmassbal = np.zeros((main_glac_rgi.shape[0],3))
    #ds[input.rgi_O1Id_colname] = ((ds[input.cal_rgi_colname] % 1) * 10**5).round(0).astype(int) 
    #ds_subset = ds[[input.rgi_O1Id_colname, input.massbal_colname, input.massbal_time1, input.massbal_time2]].values
    #rgi_O1Id = main_glac_rgi[input.rgi_O1Id_colname].values
    #glac = 1
    ## Grab the mass balance based on the RGIId Order 1 glacier number
    #main_glac_calmassbal[glac,:] = ds_subset[np.where(np.in1d(ds_subset[:,0],rgi_O1Id[glac])==True)[0][0],1:]

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
#    for glac in range(50):
#    for glac in [0]:
        print(glac)
        lrgcm = input.lrgcm
        lrglac = input.lrglac
        precfactor = input.precfactor
        precgrad = input.precgrad
        ddfsnow = input.ddfsnow
        ddfice = input.ddfice
        tempsnow = input.tempsnow
        tempchange = input.tempchange
        
        # Set model parameters
        modelparameters = [lrgcm, lrglac, precfactor, precgrad, ddfsnow, ddfice, tempsnow, tempchange]
        # Select subset of variables to reduce the amount of data being passed to the function
        glacier_rgi_table = main_glac_rgi.loc[glac, :]
        glacier_gcm_elev = main_glac_gcmelev[glac]
        glacier_gcm_prec = main_glac_gcmprec[glac,:]
        glacier_gcm_temp = main_glac_gcmtemp[glac,:]
        if input.option_lapserate_fromgcm == 1:
            glacier_gcm_lrgcm = main_glac_gcmlapserate[glac]
            glacier_gcm_lrglac = glacier_gcm_lrgcm.copy()
        else:
            glacier_gcm_lrgcm = np.zeros(glacier_gcm_temp.shape) + modelparameters[0]
            glacier_gcm_lrglac = np.zeros(glacier_gcm_temp.shape) + modelparameters[1]
        glacier_area_t0 = main_glac_hyps.iloc[glac,:].values.astype(float)   
        # Inclusion of ice thickness and width, i.e., loading values may be only required for Huss mass redistribution!
        icethickness_t0 = main_glac_icethickness.iloc[glac,:].values.astype(float)
        width_t0 = main_glac_width.iloc[glac,:].values.astype(float)
        
        # Record the calibration round
        calround = 0
        # Run calibration only for glaciers that have calibration data 
        if np.isnan(main_glac_calmassbal[glac,0]) == False:
            # OPTIMIZATION FUNCTION: Define the function that you are trying to minimize
            #  - modelparameters are the parameters that will be optimized
            #  - return value is the value is the value used to run the optimization
            def objective(modelparameters):
                (glac_bin_temp, glac_bin_prec, glac_bin_acc, glac_bin_refreeze, glac_bin_snowpack, glac_bin_melt, 
                 glac_bin_frontalablation, glac_bin_massbalclim, glac_bin_massbalclim_annual, glac_bin_area_annual, 
                 glac_bin_icethickness_annual, glac_bin_width_annual, glac_bin_surfacetype_annual) = (
                         massbalance.runmassbalance(glac, modelparameters, regionO1_number, glacier_rgi_table, 
                                                    glacier_area_t0, icethickness_t0, width_t0, glacier_gcm_temp, 
                                                    glacier_gcm_prec, glacier_gcm_elev, glacier_gcm_lrgcm, 
                                                    glacier_gcm_lrglac, elev_bins, dates_table, annual_columns, 
                                                    annual_divisor))
                # Column index for start and end year based on dates of geodetic mass balance observations
                massbal_idx_start = (main_glac_calmassbal[glac,1] - input.startyear).astype(int)
                massbal_idx_end = (massbal_idx_start + main_glac_calmassbal[glac,2] - 
                                   main_glac_calmassbal[glac,1] + 1).astype(int)
                massbal_years = massbal_idx_end - massbal_idx_start
                # Average annual glacier-wide mass balance [m w.e.]
                glac_wide_massbalclim_mwea = ((glac_bin_massbalclim_annual[:, massbal_idx_start:massbal_idx_end] * 
                                               glac_bin_area_annual[:, massbal_idx_start:massbal_idx_end]).sum() / 
                                               glacier_area_t0.sum() / massbal_years)
                #  units: m w.e. based on initial area
                # Difference between geodetic and modeled mass balance
                massbal_difference = abs(main_glac_calmassbal[glac,0] - glac_wide_massbalclim_mwea)
                return massbal_difference
            # CONSTRAINTS
            #  everything goes on one side of the equation compared to zero
            #  ex. return x[0] - input.lr_gcm with an equality constraint means x[0] = input.lr_gcm (see below)
            def constraint_lrgcm(modelparameters):
                return modelparameters[0] - lrgcm
            def constraint_lrglac(modelparameters):
                return modelparameters[1] - lrglac
            def constraint_precfactor(modelparameters):
                return modelparameters[2] - precfactor
            def constraint_precgrad(modelparameters):
                return modelparameters[3] - precgrad
            def constraint_ddfsnow(modelparameters):
                return modelparameters[4] - ddfsnow
            def constraint_ddfice(modelparameters):
                return modelparameters[5] - ddfice
            def constraint_tempsnow(modelparameters):
                return modelparameters[6] - tempsnow
            def constraint_tempchange(modelparameters):
                return modelparameters[7] - tempchange
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
            modelparameters_init = ([lrgcm, lrglac, precfactor, precgrad, ddfsnow, ddfice, tempsnow, tempchange])
            # PARAMETER BOUNDS
            lrgcm_bnds = (-0.008,-0.004)
            lrglac_bnds = (-0.008,-0.004)
            precfactor_bnds = (0.95,1.25)
            precgrad_bnds = (0.0001,0.00025)
            ddfsnow_bnds = (0.0036, 0.0046)
            #  Braithwaite (2008)
            ddfice_bnds = (ddfsnow_bnds[0]/input.ddfsnow_iceratio, ddfsnow_bnds[1]/input.ddfsnow_iceratio)
            tempsnow_bnds = (0,2) 
            tempchange_bnds = (-2,2)
            modelparameters_bnds = (lrgcm_bnds, lrglac_bnds, precfactor_bnds, precgrad_bnds, ddfsnow_bnds, ddfice_bnds,
                                    tempsnow_bnds, tempchange_bnds)            
            # OPTIMIZATION ROUND #1: optimize precfactor, DDFsnow, tempchange
            # Select constraints used to optimize precfactor
            cons = [con_lrgcm, con_lrglac, con_precgrad, con_ddficefxsnow, con_tempsnow]
            # Run the optimization
            #  'L-BFGS-B' - much slower
            modelparameters_opt = minimize(objective, modelparameters_init, method='SLSQP', bounds=modelparameters_bnds,
                                           constraints=cons, tol=1e-3)
            # Record the calibration round
            calround = calround + 1
            # Record the optimized parameters
            main_glac_modelparamsopt[glac] = modelparameters_opt.x
            # Re-run the optimized parameters in order to see the mass balance
            (glac_bin_temp, glac_bin_prec, glac_bin_acc, glac_bin_refreeze, glac_bin_snowpack, glac_bin_melt, 
             glac_bin_frontalablation, glac_bin_massbalclim, glac_bin_massbalclim_annual, glac_bin_area_annual, 
             glac_bin_icethickness_annual, glac_bin_width_annual, glac_bin_surfacetype_annual) = (
                     massbalance.runmassbalance(glac, main_glac_modelparamsopt[glac], regionO1_number, 
                                                glacier_rgi_table, glacier_area_t0, icethickness_t0, width_t0, 
                                                glacier_gcm_temp, glacier_gcm_prec, glacier_gcm_elev, glacier_gcm_lrgcm, 
                                                glacier_gcm_lrglac, elev_bins, dates_table, annual_columns, 
                                                annual_divisor))
            # Column index for start and end year based on dates of geodetic mass balance observations
            massbal_idx_start = (main_glac_calmassbal[glac,1] - input.startyear).astype(int)
            massbal_idx_end = (massbal_idx_start + main_glac_calmassbal[glac,2] - 
                               main_glac_calmassbal[glac,1] + 1).astype(int)
            massbal_years = massbal_idx_end - massbal_idx_start
            # Average annual glacier-wide mass balance [m w.e.]
            glac_wide_massbalclim_mwea = ((glac_bin_massbalclim_annual[:, massbal_idx_start:massbal_idx_end] * 
                                           glac_bin_area_annual[:, massbal_idx_start:massbal_idx_end]).sum() / 
                                           glacier_area_t0.sum() / massbal_years)
            massbal_difference = abs(main_glac_calmassbal[glac,0] - glac_wide_massbalclim_mwea)
            main_glac_massbal_compare[glac] = [glac_wide_massbalclim_mwea, main_glac_calmassbal[glac,0], 
                                               massbal_difference, calround]
            print('precfactor:', main_glac_modelparamsopt[glac,2])
            print('ddfsnow:', main_glac_modelparamsopt[glac,4])
            print('tempchange:', main_glac_modelparamsopt[glac,7])
            print(main_glac_massbal_compare[glac], '\n')
            
            # OPTIMIZATION ROUND #2: if tolerance not reached, increase bounds
            if massbal_difference > input.massbal_tolerance:
                # Constraints
                cons = [con_lrgcm, con_lrglac, con_precgrad, con_ddficefxsnow, con_tempsnow]
                # Bounds
                lrgcm_bnds = (-0.008,-0.004)
                lrglac_bnds = (-0.008,-0.004)
                precfactor_bnds = (0.9,1.5)
                precgrad_bnds = (0.0001,0.00025)
                ddfsnow_bnds = (0.0031, 0.0051)
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
                 glac_bin_icethickness_annual, glac_bin_width_annual, glac_bin_surfacetype_annual) = (
                         massbalance.runmassbalance(glac, main_glac_modelparamsopt[glac], regionO1_number, 
                                                    glacier_rgi_table, glacier_area_t0, icethickness_t0, width_t0, 
                                                    glacier_gcm_temp, glacier_gcm_prec, glacier_gcm_elev, 
                                                    glacier_gcm_lrgcm, glacier_gcm_lrglac, elev_bins, dates_table, 
                                                    annual_columns, annual_divisor))
                # Column index for start and end year based on dates of geodetic mass balance observations
                massbal_idx_start = (main_glac_calmassbal[glac,1] - input.startyear).astype(int)
                massbal_idx_end = (massbal_idx_start + main_glac_calmassbal[glac,2] - 
                                   main_glac_calmassbal[glac,1] + 1).astype(int)
                massbal_years = massbal_idx_end - massbal_idx_start
                # Average annual glacier-wide mass balance [m w.e.]
                glac_wide_massbalclim_mwea = ((glac_bin_massbalclim_annual[:, massbal_idx_start:massbal_idx_end] * 
                                               glac_bin_area_annual[:, massbal_idx_start:massbal_idx_end]).sum() / 
                                               glacier_area_t0.sum() / massbal_years)
                massbal_difference = abs(main_glac_calmassbal[glac,0] - glac_wide_massbalclim_mwea)
                main_glac_massbal_compare[glac] = [glac_wide_massbalclim_mwea, main_glac_calmassbal[glac,0], 
                                                   massbal_difference, calround]
                print('precfactor:', main_glac_modelparamsopt[glac,2])
                print('ddfsnow:', main_glac_modelparamsopt[glac,4])
                print('tempchange:', main_glac_modelparamsopt[glac,7])
                print(main_glac_massbal_compare[glac], '\n')
                
            # OPTIMIZATION ROUND #3: if tolerance not reached, increase bounds again
            if massbal_difference > input.massbal_tolerance:
                # Constraints
                cons = [con_lrgcm, con_lrglac, con_precgrad, con_ddficefxsnow, con_tempsnow]
                # Bounds
                lrgcm_bnds = (-0.008,-0.004)
                lrglac_bnds = (-0.008,-0.004)
                precfactor_bnds = (0.8,2)
                precgrad_bnds = (0.0001,0.00025)
                ddfsnow_bnds = (0.0026, 0.0056)
                ddfice_bnds = (ddfsnow_bnds[0]/input.ddfsnow_iceratio, ddfsnow_bnds[1]/input.ddfsnow_iceratio)
                tempsnow_bnds = (0,2) 
                tempchange_bnds = (-10,10)
                modelparameters_bnds = (lrgcm_bnds, lrglac_bnds, precfactor_bnds, precgrad_bnds, ddfsnow_bnds, 
                                        ddfice_bnds, tempsnow_bnds, tempchange_bnds)  
                # Run optimization
                modelparameters_opt = minimize(objective, main_glac_modelparamsopt[glac], method='SLSQP', 
                                               bounds=modelparameters_bnds, constraints=cons, tol=1e-4)
                #  requires higher tolerance due to the increase in parameters being optimized
                # Record the calibration round
                calround = calround + 1
                # Record the optimized parameters
                main_glac_modelparamsopt[glac] = modelparameters_opt.x
                # Re-run the optimized parameters in order to see the mass balance
                (glac_bin_temp, glac_bin_prec, glac_bin_acc, glac_bin_refreeze, glac_bin_snowpack, glac_bin_melt, 
                 glac_bin_frontalablation, glac_bin_massbalclim, glac_bin_massbalclim_annual, glac_bin_area_annual, 
                 glac_bin_icethickness_annual, glac_bin_width_annual, glac_bin_surfacetype_annual) = (
                         massbalance.runmassbalance(glac, main_glac_modelparamsopt[glac], regionO1_number, 
                                                    glacier_rgi_table, glacier_area_t0, icethickness_t0, width_t0, 
                                                    glacier_gcm_temp, glacier_gcm_prec, glacier_gcm_elev, 
                                                    glacier_gcm_lrgcm, glacier_gcm_lrglac, elev_bins, dates_table, 
                                                    annual_columns, annual_divisor))
                # Column index for start and end year based on dates of geodetic mass balance observations
                massbal_idx_start = (main_glac_calmassbal[glac,1] - input.startyear).astype(int)
                massbal_idx_end = (massbal_idx_start + main_glac_calmassbal[glac,2] - 
                                   main_glac_calmassbal[glac,1] + 1).astype(int)
                massbal_years = massbal_idx_end - massbal_idx_start
                # Average annual glacier-wide mass balance [m w.e.]
                glac_wide_massbalclim_mwea = ((glac_bin_massbalclim_annual[:, massbal_idx_start:massbal_idx_end] * 
                                               glac_bin_area_annual[:, massbal_idx_start:massbal_idx_end]).sum() / 
                                               glacier_area_t0.sum() / massbal_years)
                massbal_difference = abs(main_glac_calmassbal[glac,0] - glac_wide_massbalclim_mwea)
                main_glac_massbal_compare[glac] = [glac_wide_massbalclim_mwea, main_glac_calmassbal[glac,0], 
                                                   massbal_difference, calround]
                print('precfactor:', main_glac_modelparamsopt[glac,2])
                print('ddfsnow:', main_glac_modelparamsopt[glac,4])
                print('tempchange:', main_glac_modelparamsopt[glac,7])
                print(main_glac_massbal_compare[glac], '\n')
        else:
        # if calibration data not available for a glacier, then insert NaN into calibration output
            main_glac_modelparamsopt[glac] = float('NaN')
            main_glac_massbal_compare[glac] = float('NaN')
            
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
        
timeelapsed_step4 = timeit.default_timer() - timestart_step4
print('Step 4 time:', timeelapsed_step4, "s\n")

#%%=== STEP FIVE: SIMULATION RUN ======================================================================================
timestart_step5 = timeit.default_timer()

if input.option_calibration == 0:
    # [INSERT REGIONAL LOOP HERE] if want to do all regions at the same time.  Separate netcdf files will be generated
    #  for each loop to reduce file size and make files easier to read/share
    regionO1_number = input.rgi_regionsO1[0]
    # Create output netcdf file
    if input.output_package != 0:
        output.netcdfcreate(regionO1_number, main_glac_hyps, dates_table, annual_columns)
    
    # Test range
    #glac = 0
    #prec_factor_low = 0.8
    #prec_factor_high = 2.0
    #prec_factor_step = 0.005
    #prec_factor_range = np.arange(prec_factor_low, prec_factor_high + prec_factor_step, prec_factor_step)
    #glac_wide_massbal_record = np.zeros(prec_factor_range.shape)
    #for n in range(len(prec_factor_range)):
    #    prec_factor = prec_factor_range[n]
    
    # ENTER GLACIER LOOP
    #for glac in range(main_glac_rgi.shape[0]):
    for glac in [0]:
    
        lrgcm = input.lrgcm
        lrglac = input.lrglac
        precfactor = input.precfactor
        precgrad = input.precgrad
        ddfsnow = input.ddfsnow
        ddfice = input.ddfice
        tempsnow = input.tempsnow
        tempchange = input.tempchange
        
        # Set model parameters
        modelparameters = [lrgcm, lrglac, precfactor, precgrad, ddfsnow, ddfice, tempsnow, tempchange]
        # Select subset of variables to reduce the amount of data being passed to the function
        glacier_rgi_table = main_glac_rgi.loc[glac, :]
        glacier_gcm_elev = main_glac_gcmelev[glac]
        glacier_gcm_prec = main_glac_gcmprec[glac,:]
        glacier_gcm_temp = main_glac_gcmtemp[glac,:]
        if input.option_lapserate_fromgcm == 1:
            glacier_gcm_lrgcm = main_glac_gcmlapserate[glac]
            glacier_gcm_lrglac = glacier_gcm_lrgcm.copy()
        else:
            glacier_gcm_lrgcm = np.zeros(glacier_gcm_temp.shape) + modelparameters[0]
            glacier_gcm_lrglac = np.zeros(glacier_gcm_temp.shape) + modelparameters[1]
        glacier_area_t0 = main_glac_hyps.iloc[glac,:].values.astype(float)   
        # Inclusion of ice thickness and width, i.e., loading values may be only required for Huss mass redistribution!
        icethickness_t0 = main_glac_icethickness.iloc[glac,:].values.astype(float)
        width_t0 = main_glac_width.iloc[glac,:].values.astype(float)
        # MASS BALANCE:
        # Run the mass balance function (spinup years have been removed from output)
        (glac_bin_temp, glac_bin_prec, glac_bin_acc, glac_bin_refreeze, glac_bin_snowpack, glac_bin_melt, 
         glac_bin_frontalablation, glac_bin_massbalclim, glac_bin_massbalclim_annual, glac_bin_area_annual, 
         glac_bin_icethickness_annual, glac_bin_width_annual, glac_bin_surfacetype_annual) = (
                 massbalance.runmassbalance(glac, modelparameters, regionO1_number, glacier_rgi_table, glacier_area_t0, 
                                            icethickness_t0, width_t0, glacier_gcm_temp, glacier_gcm_prec, 
                                            glacier_gcm_elev, glacier_gcm_lrgcm, glacier_gcm_lrglac, elev_bins, 
                                            dates_table, annual_columns, annual_divisor))
        # OUTPUT: Record variables according to output package
        #  must be done within glacier loop since the variables will be overwritten 
        if input.output_package != 0:
            output.netcdfwrite(regionO1_number, glac, modelparameters, glacier_rgi_table, elev_bins, glac_bin_temp, 
                               glac_bin_prec, glac_bin_acc, glac_bin_refreeze, glac_bin_snowpack, glac_bin_melt, 
                               glac_bin_frontalablation, glac_bin_massbalclim, glac_bin_massbalclim_annual, 
                               glac_bin_area_annual, glac_bin_icethickness_annual, glac_bin_width_annual, 
                               glac_bin_surfacetype_annual)
        
## Plot the results
#difference = abs(glac_wide_massbal_clim_range - main_glac_calmassbal[glac])
#
#fig, ax1 = plt.subplots()
#ax2 = ax1.twinx()
#ax1.plot(prec_factor_range, glac_wide_massbal_clim_range, 'g-')
#ax2.plot(prec_factor_range, difference, 'b-')
#ax1.set_xlabel('prec_factor')
#ax1.set_ylabel('glac_wide_massbalclim', color='g')
#ax2.set_ylabel('Model - Measured', color='b')
#ax2.set_ylim(0,5)
#plt.show()

timeelapsed_step5 = timeit.default_timer() - timestart_step5
print('Step 5 time:', timeelapsed_step5, "s\n")

##%%=== Model testing ===============================================================================
###netcdf_output = nc.Dataset('../Output/PyGEM_output_rgiregion15_20180202.nc', 'r+')
###netcdf_output.close()

# Plot histograms and regional variations
data = pd.read_csv(input.output_filepath + 'calibration_R15_20180306.csv')
# drop NaN values and select subset of data
data = data.dropna()

def plot_latlonvar(lons, lats, variable, rangelow, rangehigh, title, xlabel, ylabel, east, west, south, north, xtick, 
                   ytick):
    """
    Plot a variable according to its latitude and longitude
    """
    # Create the projection
    ax = plt.axes(projection=cartopy.crs.PlateCarree())
    # Add country borders for reference
    ax.add_feature(cartopy.feature.BORDERS)
    # Set the extent
    ax.set_extent([east, west, south, north], cartopy.crs.PlateCarree())
    # Label title, x, and y axes
    plt.title(title)
    ax.set_xticks(np.arange(east,west+1,xtick), cartopy.crs.PlateCarree())
    ax.set_yticks(np.arange(south,north+1,ytick), cartopy.crs.PlateCarree())
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # Plot the data 
    plt.scatter(lons, lats, c=variable, cmap='jet')
    #  plotting x, y, size [s=__], color bar [c=__]
    plt.clim(rangelow,rangehigh)
    #  set the range of the color bar
    plt.colorbar(fraction=0.02, pad=0.04)
    #  fraction resizes the colorbar, pad is the space between the plot and colorbar
    plt.show()
    

# Set extent
east = int(round(data['CenLon'].min())) - 1
west = int(round(data['CenLon'].max())) + 1
south = int(round(data['CenLat'].min())) - 1
north = int(round(data['CenLat'].max())) + 1
xtick = 1
ytick = 1
# Select relevant data
lats = data['CenLat'][:]
lons = data['CenLon'][:]
precfactor = data['precfactor'][:]
tempchange = data['tempchange'][:]
ddfsnow = data['ddfsnow'][:]
calround = data['calround'][:]
massbal = data['MB_geodetic_mwea']
# Plot regional maps
plot_latlonvar(lons, lats, massbal, -1.5, 0.5, 'Geodetic mass balance [mwea]', 'longitude [deg]', 'latitude [deg]', 
               east, west, south, north, xtick, ytick)
# Plot regional maps
plot_latlonvar(lons, lats, precfactor, 0.8, 1.3, 'Preciptiation factor [-]', 'longitude [deg]', 'latitude [deg]', 
               east, west, south, north, xtick, ytick)
plot_latlonvar(lons, lats, tempchange, -4, 2, 'Temperature bias [degC]', 'longitude [deg]', 'latitude [deg]', 
               east, west, south, north, xtick, ytick)
plot_latlonvar(lons, lats, ddfsnow, 0.003, 0.005, 'DDF_snow [m w.e. d-1 degC-1]', 'longitude [deg]', 'latitude [deg]', 
               east, west, south, north, xtick, ytick)
plot_latlonvar(lons, lats, calround, 1, 3, 'Calibration round', 'longitude [deg]', 'latitude [deg]', 
               east, west, south, north, xtick, ytick)
# Plot histograms
data.hist(column='MB_difference_mwea', bins=50)
plt.title('Mass Balance Difference [mwea]')
data.hist(column='precfactor', bins=50)
plt.title('Precipitation factor [-]')
data.hist(column='tempchange', bins=50)
plt.title('Temperature bias [degC]')
data.hist(column='ddfsnow', bins=50)
plt.title('DDFsnow [mwe d-1 degC-1]')
plt.xticks(rotation=60)
data.hist(column='calround', bins = [0.5, 1.5, 2.5, 3.5])
plt.title('Calibration round')
plt.xticks([1, 2, 3])
    
## run plot function
#output.plot_caloutput(data)
