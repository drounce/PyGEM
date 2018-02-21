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
#from datetime import datetime
import os # os is used with re to find name matches
#import re # see os
import xarray as xr
import netCDF4 as nc
#from time import strftime
import timeit
from scipy.optimize import minimize
from scipy.stats import linregress
import matplotlib.pyplot as plt

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
if input.option_gcm_downscale == 1:
    # Air Temperature [degC] and GCM dates
    main_glac_gcmtemp, main_glac_gcmdate = climate.importGCMvarnearestneighbor_xarray(
            input.gcm_temp_filename, input.gcm_temp_varname, main_glac_rgi, dates_table, start_date, end_date)
    # Precipitation [m] and GCM dates
    main_glac_gcmprec, main_glac_gcmdate = climate.importGCMvarnearestneighbor_xarray(
            input.gcm_prec_filename, input.gcm_prec_varname, main_glac_rgi, dates_table, start_date, end_date)
    # Elevation [m a.s.l] associated with air temperature data
    main_glac_gcmelev = climate.importGCMfxnearestneighbor_xarray(input.gcm_elev_filename, input.gcm_elev_varname, 
                                                                  main_glac_rgi)    
    # Add GCM time series to the dates_table
    dates_table['date_gcm'] = main_glac_gcmdate
elif input.option_gcm_downscale == 2:
    # Import air temperature, precipitation, and elevation from pre-processed csv files for a given region
    #  this simply saves time from re-running the fxns above
    main_glac_gcmtemp = np.genfromtxt(input.gcm_filepath_var + 'csv_ERAInterim_temp_19952015_15_SouthAsiaEast.csv', 
                                      delimiter=',')
    main_glac_gcmprec = np.genfromtxt(input.gcm_filepath_var + 'csv_ERAInterim_prec_19952015_15_SouthAsiaEast.csv', 
                                      delimiter=',')
    main_glac_gcmelev = np.genfromtxt(input.gcm_filepath_var + 'csv_ERAInterim_elev_15_SouthAsiaEast.csv', 
                                      delimiter=',')
else:
    print('\n\tModel Error: please choose an option that exists for downscaling climate data. Exiting model run now.\n')
    exit()
# Print time elapsed
timeelapsed_step3 = timeit.default_timer() - timestart_step3
print('Step 3 time:', timeelapsed_step3, "s\n")

#%%=== STEP FOUR: IMPORT CALIBRATION DATASETS (IF NECESSARY) ==========================================================
timestart_step4 = timeit.default_timer()

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

timeelapsed_step4 = timeit.default_timer() - timestart_step4
print('Step 4 time:', timeelapsed_step4, "s\n")

#%%=== STEP FIVE: MASS BALANCE CALCULATIONS ===========================================================================
timestart_step5 = timeit.default_timer()

# Insert regional loop here if want to do all regions at the same time.  Separate netcdf files will be generated for
#  each loop to reduce file size and make files easier to read/share
regionO1_number = input.rgi_regionsO1[0]
# Create output netcdf file
if input.output_package != 0:
    output.netcdfcreate(regionO1_number, main_glac_hyps, dates_table, annual_columns)

# CREATE A SEPARATE OUTPUT FOR CALIBRATION with only data relevant to calibration
#   - annual glacier-wide massbal, area, ice thickness, snowline

# Model parameter output
if input.option_calibration == 1:
    main_glac_modelparamsopt = np.zeros((main_glac_rgi.shape[0], 7))
    main_glac_massbal_compare = np.zeros((main_glac_rgi.shape[0],3))

# Test range
#glac = 0
#prec_factor_low = 0.8
#prec_factor_high = 2.0
#prec_factor_step = 0.005
#prec_factor_range = np.arange(prec_factor_low, prec_factor_high + prec_factor_step, prec_factor_step)
#glac_wide_massbal_record = np.zeros(prec_factor_range.shape)
#for n in range(len(prec_factor_range)):
#    prec_factor = prec_factor_range[n]

#for glac in range(main_glac_rgi.shape[0]):
#for glac in range(50):
for glac in [0]:

    lr_gcm = input.lr_gcm
    lr_glac = input.lr_glac
    prec_factor = input.prec_factor
    prec_grad = input.prec_grad
    ddf_snow = input.ddf_snow
    ddf_ice = input.ddf_ice
    temp_snow = input.temp_snow
    
    # Set model parameters
    modelparameters = [lr_gcm, lr_glac, prec_factor, prec_grad, ddf_snow, ddf_ice, temp_snow]
    # Select subset of variables to reduce the amount of data being passed to the function
    glacier_rgi_table = main_glac_rgi.loc[glac, :]
    glacier_gcm_elev = main_glac_gcmelev[glac]
    glacier_gcm_prec = main_glac_gcmprec[glac,:]
    glacier_gcm_temp = main_glac_gcmtemp[glac,:]
    glacier_area_t0 = main_glac_hyps.iloc[glac,:].values.astype(float)   
    # Inclusion of ice thickness and width, i.e., loading the values may be only required for Huss mass redistribution!
    icethickness_t0 = main_glac_icethickness.iloc[glac,:].values.astype(float)
    width_t0 = main_glac_width.iloc[glac,:].values.astype(float)
    
    if input.option_calibration == 0:  
        # Run the mass balance function (spinup years have been removed from output)
        (glac_bin_temp, glac_bin_prec, glac_bin_acc, glac_bin_refreeze, glac_bin_snowpack, glac_bin_melt, 
         glac_bin_frontalablation, glac_bin_massbalclim, glac_bin_massbalclim_annual, glac_bin_area_annual, 
         glac_bin_icethickness_annual, glac_bin_width_annual, glac_bin_surfacetype_annual) = (
                 massbalance.runmassbalance(glac, modelparameters, regionO1_number, glacier_rgi_table, glacier_area_t0, 
                                            icethickness_t0, width_t0, glacier_gcm_temp, glacier_gcm_prec, 
                                            glacier_gcm_elev, elev_bins, dates_table, annual_columns, annual_divisor))
        # Column index for start and end year based on dates of geodetic mass balance observations
        massbal_idx_start = (main_glac_calmassbal[glac,1] - input.startyear).astype(int)
        massbal_idx_end = (massbal_idx_start + main_glac_calmassbal[glac,2] - 
                           main_glac_calmassbal[glac,1] + 1).astype(int)
        massbal_years = massbal_idx_end - massbal_idx_start
        # Average annual glacier-wide mass balance [m w.e.]
        glac_wide_massbalclim_mwea = ((glac_bin_massbalclim_annual[:, massbal_idx_start:massbal_idx_end] * 
                                       glac_bin_area_annual[:, massbal_idx_start:massbal_idx_end]).sum() / 
                                       glacier_area_t0.sum() / massbal_years)
#        glac_wide_massbal_record[n] = glac_wide_massbalclim_mwea
        # Record variables from output package here - need to be in glacier loop since the variables will be overwritten 
        if input.output_package != 0:
            output.netcdfwrite(regionO1_number, glac, modelparameters, glacier_rgi_table, elev_bins, glac_bin_temp, 
                               glac_bin_prec, glac_bin_acc, glac_bin_refreeze, glac_bin_snowpack, glac_bin_melt, 
                               glac_bin_frontalablation, glac_bin_massbalclim, glac_bin_massbalclim_annual, 
                               glac_bin_area_annual, glac_bin_icethickness_annual, glac_bin_width_annual, 
                               glac_bin_surfacetype_annual)
    #  ADJUST NETCDF FILE FOR CALIBRATION
    elif input.option_calibration == 1 and np.isnan(main_glac_calmassbal[glac,0]) == False:
        # Optimized parameters
        # Define the function that you are trying to minimize
        #  modelparameters are the parameters that will be optimized
        #  the return value is the value is the value used to run the optimization
        def objective(modelparameters):
            (glac_bin_temp, glac_bin_prec, glac_bin_acc, glac_bin_refreeze, glac_bin_snowpack, glac_bin_melt, 
             glac_bin_frontalablation, glac_bin_massbalclim, glac_bin_massbalclim_annual, glac_bin_area_annual, 
             glac_bin_icethickness_annual, glac_bin_width_annual, glac_bin_surfacetype_annual) = (
                     massbalance.runmassbalance(glac, modelparameters, regionO1_number, glacier_rgi_table, 
                                                glacier_area_t0, icethickness_t0, width_t0, glacier_gcm_temp, 
                                                glacier_gcm_prec, glacier_gcm_elev, elev_bins, dates_table, 
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
            #  units: m w.e. based on initial area
            # Difference between geodetic and modeled mass balance
            massbal_difference = abs(main_glac_calmassbal[glac,0] - glac_wide_massbalclim_mwea)
#            print(modelparameters[2], massbal_difference)
            return massbal_difference
        # Define constraints
        #  everything goes on one side of the equation compared to zero
        #  ex. return x[0] - input.lr_gcm with an equality means x[0] = input.lr_gcm
        def constraint_lrgcm(modelparameters):
            return modelparameters[0] - input.lr_gcm
        def constraint_lrglac(modelparameters):
            return modelparameters[1] - input.lr_glac
        def constraint_precfactor(modelparameters):
            return modelparameters[2] - input.prec_factor
        def constraint_precgrad(modelparameters):
            return modelparameters[3] - input.prec_grad
        def constraint_ddfsnow(modelparameters):
            return modelparameters[4] - input.DDF_snow
        def constraint_ddfice(modelparameters):
            return modelparameters[5] - input.DDF_ice
        def constraint_tempsnow(modelparameters):
            return modelparameters[6] - input.T_snow
        def constraint_ddfice2xsnow(modelparameters):
            return modelparameters[4] - 0.5*modelparameters[5] 
        def constraint_ddficegtsnow(modelparameters):
            return modelparameters[5] - modelparameters[4]
        def constraint_lrsequal(modelparameters):
            return modelparameters[0] - modelparameters[1]
        # Define the initial guess
        modelparameters_init = ([input.lr_gcm, input.lr_glac, input.prec_factor, input.prec_grad, input.DDF_snow, 
                                 input.DDF_ice, input.T_snow])
        # Define bounds
        lrgcm_bnds = (-0.008,-0.004)
        lrglac_bnds = (-0.008,-0.004)
        precfactor_bnds = (0.8,2.0)
        precgrad_bnds = (0.0001,0.00025)
        ddfsnow_bnds = (0.00175, 0.0045)
        ddfice_bnds = (0.003, 0.009)
        tempsnow_bnds = (0,2) 
        modelparameters_bnds = (lrgcm_bnds,lrglac_bnds,precfactor_bnds,precgrad_bnds,ddfsnow_bnds,ddfice_bnds,
                               tempsnow_bnds)
        # Define constraints
        con_lrgcm = {'type':'eq', 'fun':constraint_lrgcm}
        con_lrglac = {'type':'eq', 'fun':constraint_lrglac}
        con_precfactor = {'type':'eq', 'fun':constraint_precfactor}
        con_precgrad = {'type':'eq', 'fun':constraint_precgrad}
        con_ddfsnow = {'type':'eq', 'fun':constraint_ddfsnow}
        con_ddfice = {'type':'eq', 'fun':constraint_ddfice}
        con_tempsnow = {'type':'eq', 'fun':constraint_tempsnow}
        con_ddfice2xsnow = {'type':'eq', 'fun':constraint_ddfice2xsnow}
        con_ddficegtsnow = {'type':'ineq', 'fun':constraint_ddficegtsnow}
        con_lrsequal = {'type':'eq', 'fun':constraint_lrsequal}
        # Select constraints used for calibration:
#        # Optimize all parameters
#        cons = []
        # Optimize precfactor
        cons = [con_lrgcm, con_lrglac, con_precgrad, con_ddfsnow, con_ddfice, con_tempsnow]
        # Optimization Round #1: vary precfactor
        modelparameters_opt = minimize(objective, modelparameters_init, method='SLSQP', bounds=modelparameters_bnds,
                                       constraints=cons, tol=1e-3)
        #  'L-BFGS-B' - much slower
        # Print the optimal parameter set
        print(modelparameters_opt.x)
        main_glac_modelparamsopt[glac] = modelparameters_opt.x
        # Re-run the optimized parameters in order to see the mass balance
        (glac_bin_temp, glac_bin_prec, glac_bin_acc, glac_bin_refreeze, glac_bin_snowpack, glac_bin_melt, 
         glac_bin_frontalablation, glac_bin_massbalclim, glac_bin_massbalclim_annual, glac_bin_area_annual, 
         glac_bin_icethickness_annual, glac_bin_width_annual, glac_bin_surfacetype_annual) = (
                 massbalance.runmassbalance(glac, main_glac_modelparamsopt[glac], regionO1_number, glacier_rgi_table, 
                                            glacier_area_t0, icethickness_t0, width_t0, glacier_gcm_temp, 
                                            glacier_gcm_prec, glacier_gcm_elev, elev_bins, dates_table, annual_columns, 
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
                                             massbal_difference]
        print(glac_wide_massbalclim_mwea, main_glac_calmassbal[glac,0], massbal_difference, '\n')
        
        # Optimization Round #2: if tolerance not reached, vary precfactor, DDFsnow/ice
        if massbal_difference > input.massbal_tolerance:
            # Optimize precfactor, DDFsnow, DDFice; DDFice = 2 x DDFsnow
            cons = [con_lrgcm, con_lrglac, con_precgrad, con_tempsnow, con_ddfice2xsnow]
            # Run optimization
            modelparameters_opt = minimize(objective, main_glac_modelparamsopt[glac], method='SLSQP', 
                                           bounds=modelparameters_bnds, constraints=cons, tol=1e-3)
            #  'L-BFGS-B' - much slower
            # Print the optimal parameter set
            print(modelparameters_opt.x)
            main_glac_modelparamsopt[glac] = modelparameters_opt.x
            # Re-run the optimized parameters in order to see the mass balance
            (glac_bin_temp, glac_bin_prec, glac_bin_acc, glac_bin_refreeze, glac_bin_snowpack, glac_bin_melt, 
             glac_bin_frontalablation, glac_bin_massbalclim, glac_bin_massbalclim_annual, glac_bin_area_annual, 
             glac_bin_icethickness_annual, glac_bin_width_annual, glac_bin_surfacetype_annual) = (
                     massbalance.runmassbalance(glac, main_glac_modelparamsopt[glac], regionO1_number, 
                                                glacier_rgi_table, glacier_area_t0, icethickness_t0, width_t0, 
                                                glacier_gcm_temp, glacier_gcm_prec, glacier_gcm_elev, elev_bins, 
                                                dates_table, annual_columns, annual_divisor))
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
                                               massbal_difference]
            print(glac_wide_massbalclim_mwea, main_glac_calmassbal[glac,0], massbal_difference, '\n')
            
        # Optimization Round #3: if tolerance not reached, vary precfactor, DDFsnow/ice, and lr_gcm
        if massbal_difference > input.massbal_tolerance:
            # Optimize precfactor, DDFsnow, DDFice, and lr_gcm; DDFice = 2 x DDFsnow
            cons = [con_precgrad, con_tempsnow, con_ddfice2xsnow, con_lrsequal]
            # Run optimization
            modelparameters_opt = minimize(objective, main_glac_modelparamsopt[glac], method='SLSQP', 
                                           bounds=modelparameters_bnds, constraints=cons, tol=1e-4)
            #  requires higher tolerance due to the increase in parameters being optimized
            #  'L-BFGS-B' - much slower
            # Print the optimal parameter set
            print(modelparameters_opt.x)
            main_glac_modelparamsopt[glac] = modelparameters_opt.x
            # Re-run the optimized parameters in order to see the mass balance
            (glac_bin_temp, glac_bin_prec, glac_bin_acc, glac_bin_refreeze, glac_bin_snowpack, glac_bin_melt, 
             glac_bin_frontalablation, glac_bin_massbalclim, glac_bin_massbalclim_annual, glac_bin_area_annual, 
             glac_bin_icethickness_annual, glac_bin_width_annual, glac_bin_surfacetype_annual) = (
                     massbalance.runmassbalance(glac, main_glac_modelparamsopt[glac], regionO1_number, 
                                                glacier_rgi_table, glacier_area_t0, icethickness_t0, width_t0, 
                                                glacier_gcm_temp, glacier_gcm_prec, glacier_gcm_elev, elev_bins, 
                                                dates_table, annual_columns, annual_divisor))
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
                                               massbal_difference]
            print(glac_wide_massbalclim_mwea, main_glac_calmassbal[glac,0], massbal_difference, '\n')
    else:
        main_glac_modelparamsopt[glac] = float('NaN')
        main_glac_massbal_compare[glac] = float('NaN')
        # create parameter matrix for each optimized glacier - fill the rest with NaN
        
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

#%%=== STEP SIX: DATA ANALYSIS / OUTPUT ===============================================================================
#netcdf_output = nc.Dataset('../Output/PyGEM_output_rgiregion15_20180202.nc', 'r+')
#netcdf_output.close()

# Create csv such that not importing the air temperature each time (takes 90 seconds for 13,119 glaciers)
#output_csvfullfilename = input.main_directory + '/../Output/ERAInterim_elev_15_SouthAsiaEast.csv'
#climate.createcsv_GCMvarnearestneighbor(input.gcm_prec_filename, input.gcm_prec_varname, dates_table, main_glac_rgi, 
#                                        output_csvfullfilename)
#np.savetxt(output_csvfullfilename, main_glac_gcmelev, delimiter=",") 

gcm_filepath_var = os.getcwd() + '/../Climate_data/ERA_Interim/'
#  _var refers to variable data; NG refers to New Generation of CMIP5 data, i.e., a homogenized dataset
# Temperature filename
gcm_temp_pressurelevels_filename = 'ERAInterim_2015_temp_pressurelevels.nc'
#  netcdf files downloaded from cmip5-archive at ethz or ERA-Interim reanalysis data (ECMWF)

## Temperature variable name given by GCM
#gcm_temp_varname = 't2m'
##  't2m' for ERA Interim, 'tas' for CMIP5
#gcm_lat_varname = 'latitude'
##  'latitude' for ERA Interim, 'lat' for CMIP5
## Longitude variable name given by GCM
#gcm_lon_varname = 'longitude'
##  'longitude' for ERA Interim, 'lon' for CMIP5
## Time variable name given by GCM
#gcm_time_varname = 'time'

filefull = gcm_filepath_var + gcm_temp_pressurelevels_filename
data = xr.open_dataset(filefull)
glac_variable_series = np.zeros((main_glac_rgi.shape[0],dates_table.shape[0]))
variablename = 't'

# Explore the dataset properties
#print('Explore the dataset:\n', data)
# Explore the variable of interest
#print('\nExplore the variable of interest:\n', data[variablename])
# Extract the variable's attributes (ex. units)
#print(data.variables[variablename].attrs['units'])
#print('\n\nExplore the data in more detail:')
#print(data[variablename].isel(time=0, latitude=0, longitude=0))

# Function question???
# Add a minimum number of pressure levels?
#A_elev_idx_range = abs(A_elev_idx_max - A_elev_idx_min) + 1


# Extract the pressure levels [Pa]
if data['level'].attrs['units'] == 'millibars':
    # Convert pressure levels from millibars to Pa
    A_levels = data['level'].values * 100
# Compute the elevation [m a.s.l] of the pressure levels using the barometric pressure formula (pressure in Pa)
A_elev = -input.R_gas*input.temp_std/(input.gravity*input.molarmass_air)*np.log(A_levels/input.pressure_std)
# Grab minimum and maximum elevation bands from 
A_elev_min = main_glac_rgi['Zmin'].min()
A_elev_max = main_glac_rgi['Zmax'].max()
# Pressure level indices that span min and max elevations
A_elev_idx_max = np.where(A_elev_max - A_elev <= 0)[0][-1]
# if minimum elevation below minimum pressure level, then use lowest pressure level
if A_elev_min < A_elev[-1]:
    A_elev_idx_min = A_elev.shape[0] - 1
else:
    A_elev_idx_min = np.where(A_elev - A_elev_min <= 0)[0][0]
# Latitude and longitude indices (nearest neighbor)
lat_nearidx = (np.abs(main_glac_rgi[input.lat_colname].values[:,np.newaxis] - 
                      data.variables[input.gcm_lat_varname][:].values).argmin(axis=1))
lon_nearidx = (np.abs(main_glac_rgi[input.lon_colname].values[:,np.newaxis] - 
                      data.variables[input.gcm_lon_varname][:].values).argmin(axis=1))
#  argmin() is finding the minimum distance between the glacier lat/lon and the GCM pixel; .values is used to 
#  extract the position's value as opposed to having an array

# Extract the dates [month-year]
A_dates = pd.Series(data.variables[input.gcm_time_varname]).apply(lambda x: x.strftime('%Y-%m'))
# For each year, for each glacier ...
#for year in range(...) # must account for spinup!
for year in [0]:
    #for glac in range(main_glac_rgi.shape[0]):
    for glac in [0]:
        # Add in the function to grab data from that particular year...
        A = data[variablename].isel(level=range(A_elev_idx_max,A_elev_idx_min+1),latitude=lat_nearidx[glac],longitude=lon_nearidx[glac]).values
        A_slope = ((A_elev[A_elev_idx_max:A_elev_idx_min+1]*A).mean(axis=1) - A_elev[A_elev_idx_max:A_elev_idx_min+1].mean()*A.mean(axis=1)) / ((A_elev[A_elev_idx_max:A_elev_idx_min+1]**2).mean() - (A_elev[A_elev_idx_max:A_elev_idx_min+1].mean())**2)
        # Extract the dates [month-year]
        A_dates = pd.Series(data.variables[input.gcm_time_varname]).apply(lambda x: x.strftime('%Y-%m'))
        glac_variable_series[glac,year*12:(year+1)*12] = A_slope

#for step in range(0,12):
#    print(step)
#    B = data[variablename].isel(time=step,latitude=0,longitude=0).values
#    plt.figure()
#    plt.plot(A_levels,B)
#    ## evenly sampled time at 200ms intervals
#    #t = np.arange(0., 5., 0.2)
#    
#    # red dashes, blue squares and green triangles
#    #plt.plot(t, t, 'r--', t, t**2, 'bs', t, t**3, 'g^')
#    plt.show()
