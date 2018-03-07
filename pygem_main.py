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

# Quality control - if ice thickness = 0, glacier area = 0 (problem identified by glacier RGIV6-15.00016 on 03/06/2018)
main_glac_hyps[main_glac_icethickness == 0] = 0

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
        
        # Set model parameters
        modelparameters = [input.lrgcm, input.lrglac, input.precfactor, input.precgrad, input.ddfsnow, input.ddfice, 
                           input.tempsnow, input.tempchange]
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
    
    # Load model parameters
    if input.option_loadparameters == 1:
        main_glac_modelparams = pd.read_csv(input.modelparams_filepath + input.modelparams_filename, 
                                            index_col=input.indexname) 
    else:
        main_glac_modelparams = pd.DataFrame(np.repeat([input.lrgcm, input.lrglac, input.precfactor, input.precgrad, 
            input.ddfsnow, input.ddfice, input.tempsnow, input.tempchange], main_glac_rgi.shape[0]).reshape(-1, 
            main_glac_rgi.shape[0]).transpose(), columns=input.modelparams_colnames)
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
    for glac in range(main_glac_rgi.shape[0]):
#    for glac in [0]:
        print(glac)
        # Select subset of variables to reduce the amount of data being passed to the function
        modelparameters = main_glac_modelparams.loc[glac,:].values
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
#        # MASS BALANCE:
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
            
    glac = 0
    # Variables to export
    glac_bin_temp = np.zeros((elev_bins.shape[0],glacier_gcm_temp.shape[0]))
    glac_bin_prec = np.zeros((elev_bins.shape[0],glacier_gcm_temp.shape[0]))
    glac_bin_acc = np.zeros((elev_bins.shape[0],glacier_gcm_temp.shape[0]))
    glac_bin_refreezepotential = np.zeros((elev_bins.shape[0],glacier_gcm_temp.shape[0]))
    glac_bin_refreeze = np.zeros((elev_bins.shape[0],glacier_gcm_temp.shape[0]))
    glac_bin_melt = np.zeros((elev_bins.shape[0],glacier_gcm_temp.shape[0]))
    glac_bin_meltsnow = np.zeros((elev_bins.shape[0],glacier_gcm_temp.shape[0]))
    glac_bin_meltrefreeze = np.zeros((elev_bins.shape[0],glacier_gcm_temp.shape[0]))
    glac_bin_meltglac = np.zeros((elev_bins.shape[0],glacier_gcm_temp.shape[0]))
    glac_bin_frontalablation = np.zeros((elev_bins.shape[0],glacier_gcm_temp.shape[0]))
    glac_bin_snowpack = np.zeros((elev_bins.shape[0],glacier_gcm_temp.shape[0]))
    glac_bin_massbalclim = np.zeros((elev_bins.shape[0],glacier_gcm_temp.shape[0]))
    glac_bin_massbalclim_annual = np.zeros((elev_bins.shape[0],annual_columns.shape[0]))
    glac_bin_surfacetype_annual = np.zeros((elev_bins.shape[0],annual_columns.shape[0]))
    glac_bin_icethickness_annual = np.zeros((elev_bins.shape[0], annual_columns.shape[0] + 1))
    glac_bin_area_annual = np.zeros((elev_bins.shape[0], annual_columns.shape[0] + 1))
    glac_bin_width_annual = np.zeros((elev_bins.shape[0], annual_columns.shape[0] + 1))
    # Local variables
    glac_bin_precsnow = np.zeros((elev_bins.shape[0],glacier_gcm_temp.shape[0]))
    refreeze_potential = np.zeros(elev_bins.shape[0])
    snowpack_remaining = np.zeros(elev_bins.shape[0])
    dayspermonth = dates_table['daysinmonth'].values
    surfacetype_ddf = np.zeros(elev_bins.shape[0])
    glac_idx_initial = glacier_area_t0.nonzero()[0]
    #  glac_idx_initial is used with advancing glaciers to ensure no bands are added in a discontinuous section
    if input.option_adjusttemp_surfelev == 1:
        # ice thickness initial is used to adjust temps to changes in surface elevation
        icethickness_adjusttemp = icethickness_t0.copy()
        icethickness_adjusttemp[0:icethickness_adjusttemp.nonzero()[0][0]] = (
                icethickness_adjusttemp[icethickness_adjusttemp.nonzero()[0][0]])
        #  bins that advance need to have an initial ice thickness; otherwise, the temp adjustment will be based on ice
        #  thickness - 0, which is wrong  Since advancing bins take the thickness of the previous bin, set the initial 
        #  ice thickness of all bins below the terminus to the ice thickness at the terminus.
    # Compute the initial surface type [0=off-glacier, 1=ice, 2=snow, 3=firn, 4=debris]
    surfacetype, firnline_idx = massbalance.surfacetypebinsinitial(glacier_area_t0, glacier_rgi_table, elev_bins)
    # Create surface type DDF dictionary (manipulate this function for calibration or for each glacier)
    surfacetype_ddf_dict = massbalance.surfacetypeDDFdict(modelparameters)
    
    # ANNUAL LOOP (daily or monthly timestep contained within loop)
#    for year in range(0, annual_columns.shape[0]):
    for year in range(0,11):
        print(year)        
        # Glacier indices
        glac_idx_t0 = glacier_area_t0.nonzero()[0]
        # Functions currently set up for monthly timestep
        #  only compute mass balance while glacier exists
        if (input.timestep == 'monthly') and (glac_idx_t0.shape[0] != 0):      
            
            # AIR TEMPERATURE: Downscale the gcm temperature [deg C] to each bin
            if input.option_temp2bins == 1:
                # Downscale using gcm and glacier lapse rates
                #  T_bin = T_gcm + lr_gcm * (z_ref - z_gcm) + lr_glac * (z_bin - z_ref)
                glac_bin_temp[:,12*year:12*(year+1)] = (glacier_gcm_temp[12*year:12*(year+1)] + 
                     glacier_gcm_lrgcm[12*year:12*(year+1)] * (glacier_rgi_table.loc[input.option_elev_ref_downscale] - 
                     glacier_gcm_elev) + glacier_gcm_lrglac[12*year:12*(year+1)] * (elev_bins - 
                     glacier_rgi_table.loc[input.option_elev_ref_downscale])[:,np.newaxis] + modelparameters[7])
            # Option to adjust air temperature based on changes in surface elevation
            if input.option_adjusttemp_surfelev == 1:
                # T_air = T_air + lr_glac * (icethickness_present - icethickness_initial)
                glac_bin_temp[:,12*year:12*(year+1)] = (glac_bin_temp[:,12*year:12*(year+1)] + (modelparameters[1] * 
                                                        (icethickness_t0 - icethickness_adjusttemp))[:,np.newaxis])
             # remove off-glacier values
            glac_bin_temp[surfacetype==0,12*year:12*(year+1)] = 0
            
            # PRECIPITATION/ACCUMULATION: Downscale the precipitation (liquid and solid) to each bin
            if input.option_prec2bins == 1:
                # Precipitation using precipitation factor and precipitation gradient
                #  P_bin = P_gcm * prec_factor * (1 + prec_grad * (z_bin - z_ref))
                glac_bin_precsnow[:,12*year:12*(year+1)] = (glacier_gcm_prec[12*year:12*(year+1)] * modelparameters[2] 
                        * (1 + modelparameters[3] * (elev_bins - 
                        glacier_rgi_table.loc[input.option_elev_ref_downscale]))[:,np.newaxis])
            # Option to adjust precipitation of uppermost 25% of glacier for wind erosion and reduced moisture content
            if input.option_preclimit:
                # If elevation range > 1000 m, then apply corrections to uppermost 25% of glacier (Huss and Hock, 2015)
                if elev_bins[glac_idx_t0[-1]] - elev_bins[glac_idx_t0[0]] > 1000:
                    # Indices of upper 25%
                    glac_idx_upper25 = glac_idx_t0[(glac_idx_t0 - glac_idx_t0[0] + 1) / glac_idx_t0.shape[0] * 100 > 75]   
                    # Exponential decay according to elevation difference from the 75% elevation
                    #  prec_upper25 = prec * exp(-(elev_i - elev_75%)/(elev_max- - elev_75%))
                    glac_bin_precsnow[glac_idx_upper25,12*year:12*(year+1)] = (
                            glac_bin_precsnow[glac_idx_upper25[0],12*year:12*(year+1)] * 
                            np.exp(-1*(elev_bins[glac_idx_upper25] - elev_bins[glac_idx_upper25[0]]) / 
                                   (elev_bins[glac_idx_upper25[-1]] - elev_bins[glac_idx_upper25[0]]))[:,np.newaxis])
                    # Precipitation cannot be less than 87.5% of the maximum accumulation elsewhere on the glacier
                    for month in range(0,12):
                        glac_bin_precsnow[glac_idx_upper25[(glac_bin_precsnow[glac_idx_upper25,month] < 0.875 * 
                            glac_bin_precsnow[glac_idx_t0,month].max()) & 
                            (glac_bin_precsnow[glac_idx_upper25,month] != 0)], month] = (
                                                            0.875 * glac_bin_precsnow[glac_idx_t0,month].max())
            # Separate total precipitation into liquid (glac_bin_prec) and solid (glac_bin_acc)
            if input.option_accumulation == 1:
                # if temperature above threshold, then rain
                glac_bin_prec[:,12*year:12*(year+1)][glac_bin_temp[:,12*year:12*(year+1)] > modelparameters[6]] = (
                    glac_bin_precsnow[:,12*year:12*(year+1)][glac_bin_temp[:,12*year:12*(year+1)] > modelparameters[6]])
                # if temperature below threshold, then snow
                glac_bin_acc[:,12*year:12*(year+1)][glac_bin_temp[:,12*year:12*(year+1)] <= modelparameters[6]] = (
                    glac_bin_precsnow[:,12*year:12*(year+1)][glac_bin_temp[:,12*year:12*(year+1)] 
                                                             <= modelparameters[6]])
            elif input.option_accumulation == 2:
                # If temperature between min/max, then mix of snow/rain using linear relationship between min/max
                glac_bin_prec[:,12*year:12*(year+1)] = ((1/2 + (glac_bin_temp[:,12*year:12*(year+1)] - 
                             modelparameters[6]) / 2) * glac_bin_precsnow[:,12*year:12*(year+1)])
                glac_bin_acc[:,12*year:12*(year+1)] = (glac_bin_precsnow[:,12*year:12*(year+1)] - 
                            glac_bin_prec[:,12*year:12*(year+1)])
                # If temperature above maximum threshold, then all rain
                glac_bin_prec[:,12*year:12*(year+1)][glac_bin_temp[:,12*year:12*(year+1)] > modelparameters[6] + 1] = (
                            glac_bin_precsnow[:,12*year:12*(year+1)][glac_bin_temp[:,12*year:12*(year+1)] > 
                                                                     modelparameters[6] + 1])
                glac_bin_acc[:,12*year:12*(year+1)][glac_bin_temp[:,12*year:12*(year+1)] > modelparameters[6] + 1] = 0
                # If temperature below minimum threshold, then all snow
                glac_bin_acc[:,12*year:12*(year+1)][glac_bin_temp[:,12*year:12*(year+1)] <= modelparameters[6] - 1] = (
                        glac_bin_precsnow[:,12*year:12*(year+1)][glac_bin_temp[:,12*year:12*(year+1)] <= 
                                                                 modelparameters[6] - 1])
                glac_bin_prec[:,12*year:12*(year+1)][glac_bin_temp[:,12*year:12*(year+1)] <= modelparameters[6] - 1] = 0
            # remove off-glacier values
            glac_bin_prec[surfacetype==0,12*year:12*(year+1)] = 0
            glac_bin_acc[surfacetype==0,12*year:12*(year+1)] = 0
            
            # POTENTIAL REFREEZE: compute potential refreeze [m w.e.] for each bin
            if input.option_refreezing == 1:
                # Heat conduction approach based on Huss and Hock (2015)
                print('Heat conduction approach has not been coded yet.  Please choose an option that exists.'
                      '\n\nExiting model run.\n\n')
                exit()
            elif input.option_refreezing == 2:
                # Refreeze based on air temperature based on Woodward et al. (1997)
                bin_temp_annual = massbalance.annualweightedmean_array(glac_bin_temp[:,12*year:12*(year+1)], 
                                                           dates_table.iloc[12*year:12*(year+1),:])
                bin_refreezepotential_annual = (-0.69 * bin_temp_annual + 0.0096) * 1/100
                #   R(m) = -0.69 * Tair + 0.0096 * (1 m / 100 cm)
                #   Note: conversion from cm to m is included
                # Remove negative refreezing values
                bin_refreezepotential_annual[bin_refreezepotential_annual < 0] = 0
                # Place annual refreezing in user-defined month for accounting and melt purposes
                placeholder = (12 - dates_table.loc[0,'month'] + input.refreeze_month) % 12
                glac_bin_refreezepotential[:,12*year + placeholder] = bin_refreezepotential_annual  
            # remove off-glacier values
            glac_bin_refreezepotential[surfacetype==0,12*year:12*(year+1)] = 0
            
            # ENTER MONTHLY LOOP (monthly loop required as )
            for month in range(0,12):
                # Step is the position as a function of year and month, which improves readability
                step = 12*year + month
                
                # SNOWPACK, REFREEZE, MELT, AND CLIMATIC MASS BALANCE
                # Snowpack [m w.e.] = snow remaining + new snow
                glac_bin_snowpack[:,step] = snowpack_remaining + glac_bin_acc[:,step]
                # Energy available for melt [degC day]    
                melt_energy_available = glac_bin_temp[:,step]*dayspermonth[step]
                melt_energy_available[melt_energy_available < 0] = 0
                # Snow melt [m w.e.]
                glac_bin_meltsnow[:,step] = surfacetype_ddf_dict[2] * melt_energy_available
                # snow melt cannot exceed the snow depth
                glac_bin_meltsnow[glac_bin_meltsnow[:,step] > glac_bin_snowpack[:,step], step] = (
                        glac_bin_snowpack[glac_bin_meltsnow[:,step] > glac_bin_snowpack[:,step], step])
                # Energy remaining after snow melt [degC day]
                melt_energy_available = melt_energy_available - glac_bin_meltsnow[:,step] / surfacetype_ddf_dict[2]
                # remove low values of energy available caused by rounding errors in the step above
                melt_energy_available[abs(melt_energy_available) < input.tolerance] = 0
                # Compute the refreeze, refreeze melt, and any changes to the snow depth
                # Refreeze potential [m w.e.]
                #  timing of refreeze potential will vary with the method (air temperature approach updates annual and 
                #  heat conduction approach updates monthly), so check if refreeze is being udpated
                if glac_bin_refreezepotential[:,step].max() > 0:
                    refreeze_potential = glac_bin_refreezepotential[:,step]
                # Refreeze [m w.e.]
                #  refreeze in ablation zone cannot exceed the amount of snow melt (accumulation zone modified below)
                glac_bin_refreeze[:,step] = glac_bin_meltsnow[:,step]
                # refreeze cannot exceed refreeze potential
                glac_bin_refreeze[glac_bin_refreeze[:,step] > refreeze_potential, step] = (
                        refreeze_potential[glac_bin_refreeze[:,step] > refreeze_potential])
                glac_bin_refreeze[abs(glac_bin_refreeze[:,step]) < input.tolerance, step] = 0
                # Refreeze melt [m w.e.]
                glac_bin_meltrefreeze[:,step] = surfacetype_ddf_dict[2] * melt_energy_available
                # refreeze melt cannot exceed the refreeze
                glac_bin_meltrefreeze[glac_bin_meltrefreeze[:,step] > glac_bin_refreeze[:,step], step] = (
                        glac_bin_refreeze[glac_bin_meltrefreeze[:,step] > glac_bin_refreeze[:,step], step])
                # Energy remaining after refreeze melt [degC day]
                melt_energy_available = melt_energy_available - glac_bin_meltrefreeze[:,step] / surfacetype_ddf_dict[2]
                melt_energy_available[abs(melt_energy_available) < input.tolerance] = 0
                # Snow remaining [m w.e.]
                snowpack_remaining = (glac_bin_snowpack[:,step] + glac_bin_refreeze[:,step] - glac_bin_meltsnow[:,step] 
                                      - glac_bin_meltrefreeze[:,step])
                snowpack_remaining[abs(snowpack_remaining) < input.tolerance] = 0
                # Compute melt from remaining energy, if any exits, and additional refreeze in the accumulation zone
                # DDF based on surface type [m w.e. degC-1 day-1]
                for surfacetype_idx in surfacetype_ddf_dict: 
                    surfacetype_ddf[surfacetype == surfacetype_idx] = surfacetype_ddf_dict[surfacetype_idx]
                # Glacier melt [m w.e.] based on remaining energy
                glac_bin_meltglac[:,step] = surfacetype_ddf * melt_energy_available
                # Energy remaining after glacier surface melt [degC day]
                #  must specify on-glacier values, otherwise this will divide by zero and cause an error
                melt_energy_available[surfacetype != 0] = (melt_energy_available[surfacetype != 0] - 
                                     glac_bin_meltglac[surfacetype != 0, step] / surfacetype_ddf[surfacetype != 0])
                melt_energy_available[abs(melt_energy_available) < input.tolerance] = 0
                # Additional refreeze in the accumulation area [m w.e.]
                #  refreeze in accumulation zone = refreeze of snow + refreeze of underlying snow/firn
                glac_bin_refreeze[elev_bins >= elev_bins[firnline_idx], step] = (
                        glac_bin_refreeze[elev_bins >= elev_bins[firnline_idx], step] +
                        glac_bin_melt[elev_bins >= elev_bins[firnline_idx], step])
                # refreeze cannot exceed refreeze potential
                glac_bin_refreeze[glac_bin_refreeze[:,step] > refreeze_potential, step] = (
                        refreeze_potential[glac_bin_refreeze[:,step] > refreeze_potential])
                # update refreeze potential
                refreeze_potential = refreeze_potential - glac_bin_refreeze[:,step]
                refreeze_potential[abs(refreeze_potential) < input.tolerance] = 0
                # TOTAL MELT (snow + refreeze + glacier)
                glac_bin_melt[:,step] = (glac_bin_meltglac[:,step] + glac_bin_meltrefreeze[:,step] + 
                                         glac_bin_meltsnow[:,step])
                # CLIMATIC MASS BALANCE [m w.e.]
                #  climatic mass balance = accumulation + refreeze - melt
                glac_bin_massbalclim[:,step] = glac_bin_acc[:,step] + glac_bin_refreeze[:,step] - glac_bin_melt[:,step]
                
                # FRONTAL ABLATION
                if glacier_rgi_table.loc['TermType'] != 0:
                    print('CODE FRONTAL ABLATION: includes changes to mass redistribution (uses climatic mass balance)')
                    # FRONTAL ABLATION IS CALCULATED ANNUALLY IN HUSS AND HOCK (2015)
                    # How should frontal ablation pair with geometry changes?
                    #  - track the length of the last bin and have the calving losses control the bin length after mass 
                    #    redistribution
                    #  - the ice thickness will be determined by the mass redistribution
                    # Note: output functions calculate total mass balance assuming frontal ablation is a positive value
                    #       that is then subtracted from the climatic mass balance.
            
            # RETURN TO ANNUAL LOOP
            # SURFACE TYPE
            # Annual surface type [-]
            glac_bin_surfacetype_annual[:,year] = surfacetype
            # Annual climatic mass balance [m w.e.], which is used to determine the surface type
            glac_bin_massbalclim_annual[:,year] = glac_bin_massbalclim[:,12*year:12*(year+1)].sum(1)
            # Compute the surface type for each bin
            surfacetype, firnline_idx = massbalance.surfacetypebinsannual(surfacetype, glac_bin_massbalclim_annual, year)
            
            # MASS REDISTRIBUTION
            # Mass redistribution ignored for calibration (glacier properties constant) 
            if input.option_calibration == 1:
                glacier_area_t1 = glacier_area_t0
                icethickness_t1 = icethickness_t0
                width_t1 = width_t0
            else:
#                # Mass redistribution according to Huss empirical curves
#                glacier_area_t1, icethickness_t1, width_t1 = massbalance.massredistributionHuss(glacier_area_t0, icethickness_t0, 
#                        width_t0, glac_bin_massbalclim_annual, year, glac_idx_initial)
                # Reset the annual glacier area and ice thickness
                glacier_area_t1 = np.zeros(glacier_area_t0.shape)
                icethickness_t1 = np.zeros(glacier_area_t0.shape)
                width_t1 = np.zeros(glacier_area_t0.shape)
                # Annual glacier-wide volume change [km**3]
                glacier_volumechange = ((glac_bin_massbalclim_annual[:, year] / 1000 * input.density_water / 
                                         input.density_ice * glacier_area_t0).sum())
                #  units: [m w.e.] * (1 km / 1000 m) * (1000 kg / (1 m water * m**2) * (1 m ice * m**2 / 900 kg) * [km**2] 
                #         = km**3 ice          
                # If volume loss is less than the glacier volume, then redistribute mass loss/gains across the glacier;
                #  otherwise, the glacier disappears (area and thickness were already set to zero above)
                if -1 * glacier_volumechange < (icethickness_t0 / 1000 * glacier_area_t0).sum():
                    # Determine where glacier exists
                    glac_idx_t0 = glacier_area_t0.nonzero()[0]
                    # Compute ice thickness [m ice], glacier area [km**2] and ice thickness change [m ice] after 
                    #  redistribution of gains/losses
                    if input.option_massredistribution == 1:
#                        # Option 1: apply mass redistribution using Huss' empirical geometry change equations
#                        icethickness_t1, glacier_area_t1, width_t1, icethickness_change = massbalance.massredistributioncurveHuss(
#                                icethickness_t0, glacier_area_t0, width_t0, glac_idx_t0, glacier_volumechange,
#                                glac_bin_massbalclim_annual[:, year])
                        
                        
                        # LOOK AT FUNCTION HERE TO SEE IF IT'S PROPERLY DISTRIBUTING MASS GAIN OVER THE GLACIER - SEEMS LIKE IT'S DOING LOSS!
                        # MASSREDISTRIBUTIONCURVEHUSS
                        # Apply Huss redistribution if there are at least 3 elevation bands; otherwise, use the mass balance
                        # reset variables
                        icethickness_t1 = np.zeros(glacier_area_t0.shape)
                        glacier_area_t1 = np.zeros(glacier_area_t0.shape)
                        width_t1 = np.zeros(glacier_area_t0.shape) 
                        if glac_idx_t0.shape[0] > 3:
                            #Select the factors for the normalized ice thickness change curve based on glacier area
                            if glacier_area_t0.sum() > 20:
                                [gamma, a, b, c] = [6, -0.02, 0.12, 0]
                            elif glacier_area_t0.sum() > 5:
                                [gamma, a, b, c] = [4, -0.05, 0.19, 0.01]
                            else:
                                [gamma, a, b, c] = [2, -0.30, 0.60, 0.09]
                            # reset variables
                            elevrange_norm = np.zeros(glacier_area_t0.shape)
                            icethicknesschange_norm = np.zeros(glacier_area_t0.shape)
                            # Normalized elevation range [-]
                            #  (max elevation - bin elevation) / (max_elevation - min_elevation)
                            elevrange_norm[glacier_area_t0 > 0] = (glac_idx_t0[-1] - glac_idx_t0) / (glac_idx_t0[-1] - glac_idx_t0[0])
                            #  using indices as opposed to elevations automatically skips bins on the glacier that have no area
                            #  such that the normalization is done only on bins where the glacier lies
                            # Normalized ice thickness change [-]
                            icethicknesschange_norm[glacier_area_t0 > 0] = ((elevrange_norm[glacier_area_t0 > 0] + a)**gamma + 
                                                                            b*(elevrange_norm[glacier_area_t0 > 0] + a) + c)
                            #  delta_h = (h_n + a)**gamma + b*(h_n + a) + c
                            #  indexing is faster here
                            # limit the icethicknesschange_norm to between 0 - 1 (ends of fxns not exactly 0 and 1)
                            icethicknesschange_norm[icethicknesschange_norm > 1] = 1
                            icethicknesschange_norm[icethicknesschange_norm < 0] = 0
                            # Huss' ice thickness scaling factor, fs_huss [m ice]         
                            fs_huss = glacier_volumechange / (glacier_area_t0 * icethicknesschange_norm).sum() * 1000
                            #  units: km**3 / (km**2 * [-]) * (1000 m / 1 km) = m ice
                            # Volume change [km**3 ice]
                            bin_volumechange = icethicknesschange_norm * fs_huss / 1000 * glacier_area_t0
#                        # Otherwise, compute volume change in each bin based on the climatic mass balance
#                        else:
#                            bin_volumechange = massbalclim_annual / 1000 * glacier_area_t0
#                        if input.option_glaciershape == 1:
#                            # Ice thickness at end of timestep for parabola [m ice]
#                            #  run in two steps to avoid errors with negative numbers and fractional exponents
#                            #  H_1 = (H_0**1.5 + delta_Vol * H_0**0.5 / A_0)**(2/3)
#                            icethickness_t1[glac_idx_t0] = ((icethickness_t0[glac_idx_t0] / 1000)**1.5 + 
#                                           (icethickness_t0[glac_idx_t0] / 1000)**0.5 * bin_volumechange[glac_idx_t0] / 
#                                           glacier_area_t0[glac_idx_t0])
#                            icethickness_t1[icethickness_t1 < 0] = 0
#                            icethickness_t1[glac_idx_t0] = icethickness_t1[glac_idx_t0]**(2/3) * 1000
#                            # Glacier area for parabola [km**2]
#                            #  A_1 = A_0 * (H_1 / H_0)**0.5
#                            glacier_area_t1[glac_idx_t0] = (glacier_area_t0[glac_idx_t0] * (icethickness_t1[glac_idx_t0] / 
#                                                            icethickness_t0[glac_idx_t0])**0.5)
#                            # Glacier width for parabola [km]
#                            #  w_1 = w_0 * (A_1 / A_0)
#                            width_t1[glac_idx_t0] = width_t0[glac_idx_t0] * glacier_area_t1[glac_idx_t0] / glacier_area_t0[glac_idx_t0]
#                        elif input.option_glaciershape == 2:
#                            # Ice thickness at end of timestep for rectangle [m ice]
#                            #  H_1 = H_0 + delta_Vol / A_0
#                            icethickness_t1[glac_idx_t0] = (((icethickness_t0[glac_idx_t0] / 1000) + 
#                                                             bin_volumechange[glac_idx_t0] / glacier_area_t0[glac_idx_t0]) * 1000)
#                            # Glacier area constant for rectangle [km**2]
#                            #  A_1 = A_0
#                            glacier_area_t1[glac_idx_t0] = glacier_area_t0[glac_idx_t0]
#                            # Glacier width constant for rectangle [km]
#                            #  w_1 = w_0
#                            width_t1[glac_idx_t0] = width_t0[glac_idx_t0]
#                        elif input.option_glaciershape == 3:
#                            # Ice thickness at end of timestep for triangle [m ice]
#                            #  run in two steps to avoid errors with negative numbers and fractional exponents
#                            icethickness_t1[glac_idx_t0] = ((icethickness_t0[glac_idx_t0] / 1000)**2 + 
#                                           bin_volumechange[glac_idx_t0] * (icethickness_t0[glac_idx_t0] / 1000) / 
#                                           glacier_area_t0[glac_idx_t0])                                   
#                            icethickness_t1[icethickness_t1 < 0] = 0
#                            icethickness_t1[glac_idx_t0] = icethickness_t1[glac_idx_t0]**(1/2) * 1000
#                            # Glacier area for triangle [km**2]
#                            #  A_1 = A_0 * H_1 / H_0
#                            glacier_area_t1[glac_idx_t0] = (glacier_area_t0[glac_idx_t0] * icethickness_t1[glac_idx_t0] / 
#                                                            icethickness_t0[glac_idx_t0])
#                            # Glacier width for triangle [km]
#                            #  w_1 = w_0 * (A_1 / A_0)
#                            width_t1[glac_idx_t0] = width_t0[glac_idx_t0] * glacier_area_t1[glac_idx_t0] / glacier_area_t0[glac_idx_t0]
#                        # Ice thickness change [m ice]
#                        icethickness_change = icethickness_t1 - icethickness_t0
#                        # return the ice thickness [m ice] and ice thickness change [m ice]                        
                        
                        
                    # Glacier retreat
                    #  if glacier retreats (ice thickness < 0), then redistribute mass loss across the rest of the glacier
                    glac_idx_t0_raw = glac_idx_t0.copy()
                    if (icethickness_t1[glac_idx_t0] <= 0).any() == True:
                        # Record glacier area and ice thickness before retreat corrections applied
                        glacier_area_t0_raw = glacier_area_t0.copy()
                        icethickness_t0_raw = icethickness_t0.copy()
                        width_t0_raw = width_t0.copy()
                        #  this is only used when there are less than 3 bins
                    while (icethickness_t1[glac_idx_t0_raw] <= 0).any() == True:
                        # Glacier volume change associated with retreat [km**3]
                        glacier_volumechange_retreat = (-1*(icethickness_t0[glac_idx_t0][icethickness_t1[glac_idx_t0] <= 0] 
                                / 1000 * glacier_area_t0[glac_idx_t0][icethickness_t1[glac_idx_t0] <= 0]).sum())
                        #  multiplying by -1 makes volume change negative
                        # Glacier volume change remaining [km**3]
                        glacier_volumechange = glacier_volumechange - glacier_volumechange_retreat
                        # update glacier area and ice thickness to account for retreat
                        glacier_area_t0_raw[icethickness_t1 <= 0] = 0
                        icethickness_t0_raw[icethickness_t1 <= 0] = 0
                        width_t0_raw[icethickness_t1 <= 0] = 0
                        glac_idx_t0_raw = glacier_area_t0_raw.nonzero()[0]
                        # Climatic mass balance for the case when there are less than 3 bins and the glacier is retreating, 
                        #  distribute the remaining glacier volume change over the entire glacier (remaining bins)
                        massbal_clim_retreat = np.zeros(glacier_area_t0_raw.shape)
                        massbal_clim_retreat[glac_idx_t0_raw] = glacier_volumechange/glacier_area_t0_raw.sum() * 1000
                        # Compute mass redistribution
                        if input.option_massredistribution == 1:
                            # Option 1: apply mass redistribution using Huss' empirical geometry change equations
                            icethickness_t1, glacier_area_t1, width_t1, icethickness_change = massbalance.massredistributioncurveHuss(
                                    icethickness_t0_raw, glacier_area_t0_raw, width_t0_raw, glac_idx_t0_raw, glacier_volumechange,
                                    massbal_clim_retreat)
                    # Glacier advances
                    #  if glacier advances (ice thickness change exceeds threshold), then redistribute mass gain in new bins
#                    while (icethickness_change > input.icethickness_advancethreshold).any() == True: 
                    if (icethickness_change > input.icethickness_advancethreshold).any() == True: 
                        print('glacier advance')
#                        # Record glacier area and ice thickness before advance corrections applied
#                        glacier_area_t1_raw = glacier_area_t1.copy()
#                        icethickness_t1_raw = icethickness_t1.copy()
#                        width_t1_raw = width_t1.copy()
#                        # Index bins that are surging
#                        icethickness_change[icethickness_change <= input.icethickness_advancethreshold] = 0
#                        glac_idx_advance = icethickness_change.nonzero()[0]
#                        # Update ice thickness based on maximum advance threshold [m ice]
#                        icethickness_t1[glac_idx_advance] = icethickness_t1[glac_idx_advance] + input.icethickness_advancethreshold
#                        # Update glacier area based on reduced ice thicknesses [km**2]
#                        if input.option_glaciershape == 1:
#                            # Glacier area for parabola [km**2] (A_1 = A_0 * (H_1 / H_0)**0.5)
#                            glacier_area_t1[glac_idx_advance] = (glacier_area_t1_raw[glac_idx_advance] * 
#                                           (icethickness_t1[glac_idx_advance] / icethickness_t1_raw[glac_idx_advance])**0.5)
#                            # Glacier width for parabola [km] (w_1 = w_0 * A_1 / A_0)
#                            width_t1[glac_idx_advance] = (width_t1_raw[glac_idx_advance] * glacier_area_t1[glac_idx_advance] 
#                                                          / glacier_area_t1_raw[glac_idx_advance])
#                        elif input.option_glaciershape == 2:
#                            # Glacier area constant for rectangle [km**2] (A_1 = A_0)
#                            glacier_area_t1[glac_idx_advance] = glacier_area_t1_raw[glac_idx_advance]
#                            # Glacier with constant for rectangle [km] (w_1 = w_0)
#                            width_t1[glac_idx_advance] = width_t1_raw[glac_idx_advance]
#                        elif input.option_glaciershape == 3:
#                            # Glacier area for triangle [km**2] (A_1 = A_0 * H_1 / H_0)
#                            glacier_area_t1[glac_idx_t0] = (glacier_area_t1_raw[glac_idx_t0] * 
#                                           icethickness_t1[glac_idx_t0] / icethickness_t1_raw[glac_idx_t0])
#                            # Glacier width for triangle [km] (w_1 = w_0 * A_1 / A_0)
#                            width_t1[glac_idx_advance] = (width_t1_raw[glac_idx_advance] * glacier_area_t1[glac_idx_advance] 
#                                                          / glacier_area_t1_raw[glac_idx_advance])
#                        # Advance volume [km**3]
#                        advance_volume = ((glacier_area_t1_raw[glac_idx_advance] * 
#                                          icethickness_t1_raw[glac_idx_advance] / 1000).sum() - 
#                                          (glacier_area_t1[glac_idx_advance] * icethickness_t1[glac_idx_advance] / 
#                                           1000).sum())
#                        # Advance characteristics
#                        # Indices that define the glacier terminus
#                        glac_idx_terminus = (glac_idx_t0[(glac_idx_t0 - glac_idx_t0[0] + 1) / 
#                                                         glac_idx_t0.shape[0] * 100 < input.terminus_percentage])
#                        # Average area of glacier terminus [km**2]
#                        terminus_area_avg = glacier_area_t0[glac_idx_terminus[1]:
#                                                            glac_idx_terminus[glac_idx_terminus.shape[0]-1]+1].mean()    
#                        #  exclude the bin at the terminus, since this bin may need to be filled first
#                        # Check if the last bin's area is below the terminus' average and fill it up if it is
#                        if glacier_area_t1[glac_idx_terminus[0]] < terminus_area_avg:
#                            print('glacier advance fill up terminus bin')
#                            # Volume required to fill the bin at the terminus
#                            advance_volume_fillbin = (icethickness_t1[glac_idx_terminus[0]] / 1000 * (terminus_area_avg - 
#                                                      glacier_area_t1[glac_idx_terminus[0]]))
#                            # If the advance volume is less than that required to fill the bin, then fill the bin as much as
#                            #  possible by adding area (thickness remains the same - glacier front is only thing advancing)
#                            if advance_volume < advance_volume_fillbin:
#                                # add advance volume to the bin (area increases, thickness and width constant)
#                                glacier_area_t1[glac_idx_terminus[0]] = (glacier_area_t1[glac_idx_terminus[0]] + 
#                                               advance_volume / (icethickness_t1[glac_idx_terminus[0]] / 1000))
#                                # set advance volume equal to zero
#                                advance_volume = 0
#                            else:
#                                # fill the bin (area increases, thickness and width constant)
#                                glacier_area_t1[glac_idx_terminus[0]] = (glacier_area_t1[glac_idx_terminus[0]] + 
#                                               advance_volume_fillbin / (icethickness_t1[glac_idx_terminus[0]] / 1000))
#                                advance_volume = advance_volume - advance_volume_fillbin
#                        # With remaining advance volume, add a bin
#                        if advance_volume > 0:
#                            print('glacier advance add a bin')
#                            # Index for additional bin below the terminus
#                            glac_idx_bin2add = np.array([glac_idx_terminus[0] - 1])
#                            # Check if bin2add is in a discontinuous section of the initial glacier
#                            while ((glac_idx_bin2add > glac_idx_initial.min()) & 
#                                   ((glac_idx_bin2add == glac_idx_initial).any() == False)):
#                                # Advance should not occur in a discontinuous section of the glacier (e.g., vertical drop),
#                                #  so change the bin2add to the next bin down valley
#                                glac_idx_bin2add = glac_idx_bin2add - 1
#                            # if the added bin would be below sea-level, then volume is distributed over the glacier without
#                            #  any adjustments
#                            if glac_idx_bin2add < 0:
#                                glacier_area_t1 = glacier_area_t1_raw
#                                icethickness_t1 = icethickness_t1_raw
#                                width_t1 = width_t1_raw
#                                advance_volume = 0
#                            # otherwise, add a bin with thickness and width equal to the previous bin and fill it up
#                            else:
#                                # ice thickness of new bin equals ice thickness of bin at the terminus
#                                icethickness_t1[glac_idx_bin2add] = icethickness_t1[glac_idx_terminus[0]]
#                                width_t1[glac_idx_bin2add] = width_t1[glac_idx_terminus[0]]
#                                # volume required to fill the bin at the terminus
#                                advance_volume_fillbin = icethickness_t1[glac_idx_bin2add] / 1000 * terminus_area_avg 
#                                # If the advance volume is unable to fill entire bin, then fill it as much as possible
#                                if advance_volume < advance_volume_fillbin:
#                                    # add advance volume to the bin (area increases, thickness and width constant)
#                                    glacier_area_t1[glac_idx_bin2add] = (advance_volume / (icethickness_t1[glac_idx_bin2add]
#                                                                         / 1000))
#                                    advance_volume = 0
#                                else:
#                                    # fill the bin (area increases, thickness and width constant)
#                                    glacier_area_t1[glac_idx_bin2add] = terminus_area_avg
#                                    advance_volume = advance_volume - advance_volume_fillbin
#                        # update the glacier indices
#                        glac_idx_t0 = glacier_area_t1.nonzero()[0]
#                        massbal_clim_advance = np.zeros(glacier_area_t1.shape)
#                        # Record glacier area and ice thickness before advance corrections applied
#                        glacier_area_t1_raw = glacier_area_t1.copy()
#                        icethickness_t1_raw = icethickness_t1.copy()
#                        width_t1_raw = width_t1.copy()
#                        # If a full bin has been added and volume still remains, then redistribute mass across the
#                        #  glacier, thereby enabling the bins to get thicker once again prior to adding a new bin.
#                        #  This is important for glaciers that have very thin ice at the terminus as this prevents the 
#                        #  glacier from having a thin layer of ice advance tremendously far down valley without thickening.
#                        if advance_volume > 0:
#                            if input.option_massredistribution == 1:
##                                # Option 1: apply mass redistribution using Huss' empirical geometry change equations
##                                icethickness_t1, glacier_area_t1, width_t1, icethickness_change = massbalance.massredistributioncurveHuss(
##                                        icethickness_t1, glacier_area_t1, width_t1, glac_idx_t0, advance_volume,
##                                        massbal_clim_advance)
#                                # Apply Huss redistribution if there are at least 3 elevation bands; otherwise, use the mass balance
#                                # reset variables
#                                icethickness_t1 = np.zeros(glacier_area_t0.shape)
#                                glacier_area_t1 = np.zeros(glacier_area_t0.shape)
#                                width_t1 = np.zeros(glacier_area_t0.shape) 
#                                if glac_idx_t0.shape[0] > 3:
#                                    #Select the factors for the normalized ice thickness change curve based on glacier area
#                                    if glacier_area_t0.sum() > 20:
#                                        [gamma, a, b, c] = [6, -0.02, 0.12, 0]
#                                    elif glacier_area_t0.sum() > 5:
#                                        [gamma, a, b, c] = [4, -0.05, 0.19, 0.01]
#                                    else:
#                                        [gamma, a, b, c] = [2, -0.30, 0.60, 0.09]
#                                    # reset variables
#                                    elevrange_norm = np.zeros(glacier_area_t0.shape)
#                                    icethicknesschange_norm = np.zeros(glacier_area_t0.shape)
#                                    # Normalized elevation range [-]
#                                    #  (max elevation - bin elevation) / (max_elevation - min_elevation)
#                                    elevrange_norm[glacier_area_t0 > 0] = (glac_idx_t0[-1] - glac_idx_t0) / (glac_idx_t0[-1] - glac_idx_t0[0])
#                                    #  using indices as opposed to elevations automatically skips bins on the glacier that have no area
#                                    #  such that the normalization is done only on bins where the glacier lies
#                                    # Normalized ice thickness change [-]
#                                    icethicknesschange_norm[glacier_area_t0 > 0] = ((elevrange_norm[glacier_area_t0 > 0] + a)**gamma + 
#                                                                                    b*(elevrange_norm[glacier_area_t0 > 0] + a) + c)
#                                    #  delta_h = (h_n + a)**gamma + b*(h_n + a) + c
#                                    #  indexing is faster here
#                                    # limit the icethicknesschange_norm to between 0 - 1 (ends of fxns not exactly 0 and 1)
#                                    icethicknesschange_norm[icethicknesschange_norm > 1] = 1
#                                    icethicknesschange_norm[icethicknesschange_norm < 0] = 0
#                                    # Huss' ice thickness scaling factor, fs_huss [m ice]         
#                                    fs_huss = glacier_volumechange / (glacier_area_t0 * icethicknesschange_norm).sum() * 1000
#                                    #  units: km**3 / (km**2 * [-]) * (1000 m / 1 km) = m ice
#                                    # Volume change [km**3 ice]
#                                    bin_volumechange = icethicknesschange_norm * fs_huss / 1000 * glacier_area_t0
#                                # Otherwise, compute volume change in each bin based on the climatic mass balance
#                                else:
#                                    bin_volumechange = massbalclim_annual / 1000 * glacier_area_t0
#                                if input.option_glaciershape == 1:
#                                    # Ice thickness at end of timestep for parabola [m ice]
#                                    #  run in two steps to avoid errors with negative numbers and fractional exponents
#                                    #  H_1 = (H_0**1.5 + delta_Vol * H_0**0.5 / A_0)**(2/3)
#                                    icethickness_t1[glac_idx_t0] = ((icethickness_t0[glac_idx_t0] / 1000)**1.5 + 
#                                                   (icethickness_t0[glac_idx_t0] / 1000)**0.5 * bin_volumechange[glac_idx_t0] / 
#                                                   glacier_area_t0[glac_idx_t0])
#                                    icethickness_t1[icethickness_t1 < 0] = 0
#                                    icethickness_t1[glac_idx_t0] = icethickness_t1[glac_idx_t0]**(2/3) * 1000
#                                    # Glacier area for parabola [km**2]
#                                    #  A_1 = A_0 * (H_1 / H_0)**0.5
#                                    glacier_area_t1[glac_idx_t0] = (glacier_area_t0[glac_idx_t0] * (icethickness_t1[glac_idx_t0] / 
#                                                                    icethickness_t0[glac_idx_t0])**0.5)
#                                    # Glacier width for parabola [km]
#                                    #  w_1 = w_0 * (A_1 / A_0)
#                                    width_t1[glac_idx_t0] = width_t0[glac_idx_t0] * glacier_area_t1[glac_idx_t0] / glacier_area_t0[glac_idx_t0]
#                                elif input.option_glaciershape == 2:
#                                    # Ice thickness at end of timestep for rectangle [m ice]
#                                    #  H_1 = H_0 + delta_Vol / A_0
#                                    icethickness_t1[glac_idx_t0] = (((icethickness_t0[glac_idx_t0] / 1000) + 
#                                                                     bin_volumechange[glac_idx_t0] / glacier_area_t0[glac_idx_t0]) * 1000)
#                                    # Glacier area constant for rectangle [km**2]
#                                    #  A_1 = A_0
#                                    glacier_area_t1[glac_idx_t0] = glacier_area_t0[glac_idx_t0]
#                                    # Glacier width constant for rectangle [km]
#                                    #  w_1 = w_0
#                                    width_t1[glac_idx_t0] = width_t0[glac_idx_t0]
#                                elif input.option_glaciershape == 3:
#                                    # Ice thickness at end of timestep for triangle [m ice]
#                                    #  run in two steps to avoid errors with negative numbers and fractional exponents
#                                    icethickness_t1[glac_idx_t0] = ((icethickness_t0[glac_idx_t0] / 1000)**2 + 
#                                                   bin_volumechange[glac_idx_t0] * (icethickness_t0[glac_idx_t0] / 1000) / 
#                                                   glacier_area_t0[glac_idx_t0])                                   
#                                    icethickness_t1[icethickness_t1 < 0] = 0
#                                    icethickness_t1[glac_idx_t0] = icethickness_t1[glac_idx_t0]**(1/2) * 1000
#                                    # Glacier area for triangle [km**2]
#                                    #  A_1 = A_0 * H_1 / H_0
#                                    glacier_area_t1[glac_idx_t0] = (glacier_area_t0[glac_idx_t0] * icethickness_t1[glac_idx_t0] / 
#                                                                    icethickness_t0[glac_idx_t0])
#                                    # Glacier width for triangle [km]
#                                    #  w_1 = w_0 * (A_1 / A_0)
#                                    width_t1[glac_idx_t0] = width_t0[glac_idx_t0] * glacier_area_t1[glac_idx_t0] / glacier_area_t0[glac_idx_t0]
#                                # Ice thickness change [m ice]
#                                icethickness_change = icethickness_t1 - icethickness_t0
#                                # return the ice thickness [m ice] and ice thickness change [m ice]
                                
                                
                                


                # ignore mass redistribution during spinup years
                if year < input.spinupyears:
                    glacier_area_t1 = glacier_area_t0
                    icethickness_t1 = icethickness_t0
                    width_t1 = width_t0
                # update surface type for bins that have retreated
                surfacetype[glacier_area_t1 == 0] = 0
                # update surface type for bins that have advanced 
                surfacetype[(surfacetype == 0) & (glacier_area_t1 != 0)] = surfacetype[glacier_area_t0.nonzero()[0][0]]
            # Record glacier properties (area [km**2], thickness [m], width [km])
            # if first year, record initial glacier properties (area [km**2], ice thickness [m ice], width [km])
            if year == 0:
                glac_bin_area_annual[:,year] = glacier_area_t0
                glac_bin_icethickness_annual[:,year] = icethickness_t0
                glac_bin_width_annual[:,year] = width_t0
            # record the next year's properties as well
            # 'year + 1' used so the glacier properties are consistent with those used in the mass balance computations
            glac_bin_icethickness_annual[:,year + 1] = icethickness_t1
            glac_bin_area_annual[:,year + 1] = glacier_area_t1
            glac_bin_width_annual[:,year + 1] = width_t1
            # Update glacier properties for the mass balance computations
            icethickness_t0 = icethickness_t1.copy()
            glacier_area_t0 = glacier_area_t1.copy()
            width_t0 = width_t1.copy()  
            
            
        
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

## Plot histograms and regional variations
#data = pd.read_csv(input.output_filepath + 'calibration_R15_20180306.csv')
## drop NaN values and select subset of data
#data = data.dropna()
#
#def plot_latlonvar(lons, lats, variable, rangelow, rangehigh, title, xlabel, ylabel, east, west, south, north, xtick, 
#                   ytick):
#    """
#    Plot a variable according to its latitude and longitude
#    """
#    # Create the projection
#    ax = plt.axes(projection=cartopy.crs.PlateCarree())
#    # Add country borders for reference
#    ax.add_feature(cartopy.feature.BORDERS)
#    # Set the extent
#    ax.set_extent([east, west, south, north], cartopy.crs.PlateCarree())
#    # Label title, x, and y axes
#    plt.title(title)
#    ax.set_xticks(np.arange(east,west+1,xtick), cartopy.crs.PlateCarree())
#    ax.set_yticks(np.arange(south,north+1,ytick), cartopy.crs.PlateCarree())
#    plt.xlabel(xlabel)
#    plt.ylabel(ylabel)
#    # Plot the data 
#    plt.scatter(lons, lats, c=variable, cmap='jet')
#    #  plotting x, y, size [s=__], color bar [c=__]
#    plt.clim(rangelow,rangehigh)
#    #  set the range of the color bar
#    plt.colorbar(fraction=0.02, pad=0.04)
#    #  fraction resizes the colorbar, pad is the space between the plot and colorbar
#    plt.show()
#    
#
## Set extent
#east = int(round(data['CenLon'].min())) - 1
#west = int(round(data['CenLon'].max())) + 1
#south = int(round(data['CenLat'].min())) - 1
#north = int(round(data['CenLat'].max())) + 1
#xtick = 1
#ytick = 1
## Select relevant data
#lats = data['CenLat'][:]
#lons = data['CenLon'][:]
#precfactor = data['precfactor'][:]
#tempchange = data['tempchange'][:]
#ddfsnow = data['ddfsnow'][:]
#calround = data['calround'][:]
#massbal = data['MB_geodetic_mwea']
## Plot regional maps
#plot_latlonvar(lons, lats, massbal, -1.5, 0.5, 'Geodetic mass balance [mwea]', 'longitude [deg]', 'latitude [deg]', 
#               east, west, south, north, xtick, ytick)
## Plot regional maps
#plot_latlonvar(lons, lats, precfactor, 0.8, 1.3, 'Preciptiation factor [-]', 'longitude [deg]', 'latitude [deg]', 
#               east, west, south, north, xtick, ytick)
#plot_latlonvar(lons, lats, tempchange, -4, 2, 'Temperature bias [degC]', 'longitude [deg]', 'latitude [deg]', 
#               east, west, south, north, xtick, ytick)
#plot_latlonvar(lons, lats, ddfsnow, 0.003, 0.005, 'DDF_snow [m w.e. d-1 degC-1]', 'longitude [deg]', 'latitude [deg]', 
#               east, west, south, north, xtick, ytick)
#plot_latlonvar(lons, lats, calround, 1, 3, 'Calibration round', 'longitude [deg]', 'latitude [deg]', 
#               east, west, south, north, xtick, ytick)
## Plot histograms
#data.hist(column='MB_difference_mwea', bins=50)
#plt.title('Mass Balance Difference [mwea]')
#data.hist(column='precfactor', bins=50)
#plt.title('Precipitation factor [-]')
#data.hist(column='tempchange', bins=50)
#plt.title('Temperature bias [degC]')
#data.hist(column='ddfsnow', bins=50)
#plt.title('DDFsnow [mwe d-1 degC-1]')
#plt.xticks(rotation=60)
#data.hist(column='calround', bins = [0.5, 1.5, 2.5, 3.5])
#plt.title('Calibration round')
#plt.xticks([1, 2, 3])
    
## run plot function
#output.plot_caloutput(data)
