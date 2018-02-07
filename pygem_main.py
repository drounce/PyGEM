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
#import xarray as xr
import netCDF4 as nc
#from time import strftime
import timeit
from scipy.optimize import minimize

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
elev_bins = main_glac_hyps.columns.values
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
    main_glac_gcmprec, main_glac_gcmtimedate = climate.importGCMvarnearestneighbor_xarray(
            input.gcm_prec_filename, input.gcm_prec_varname, main_glac_rgi, dates_table, start_date, end_date)
    # Elevation [m a.s.l] associated with air temperature data
    main_glac_gcmelev = climate.importGCMfxnearestneighbor_xarray(
            input.gcm_elev_filename, input.gcm_elev_varname, main_glac_rgi)
else:
    print('\n\tModel Error: please choose an option that exists for downscaling climate data. Exiting model run now.\n')
    exit()
# Add GCM time series to the dates_table
dates_table['date_gcm'] = main_glac_gcmdate
# Print time elapsed
timeelapsed_step3 = timeit.default_timer() - timestart_step3
print('Step 3 time:', timeelapsed_step3, "s\n")

#%%=== STEP FOUR: IMPORT CALIBRATION DATASETS (IF NECESSARY) ==========================================================

## Import .csv file
#cal_filepath = os.path.dirname(__file__) + '/../DEMs/'
## Dictionary of hypsometry filenames
#cal_filename = 'hma_mb_20170717_1846.csv'
#ds = pd.read_csv(cal_filepath + cal_filename)
#main_glac_calmassbal = np.zeros((main_glac_rgi.shape[0],3))
main_glac_calmassbal = np.array([-0.39, -0.7])

#%%=== STEP FIVE: MASS BALANCE CALCULATIONS ===========================================================================
timestart_step4 = timeit.default_timer()

# Insert regional loop here if want to do all regions at the same time.  Separate netcdf files will be generated for
#  each loop to reduce file size and make files easier to read/share
regionO1_number = input.rgi_regionsO1[0]
# Create output netcdf file
if input.output_package != 0:
    output.netcdfcreate(regionO1_number, main_glac_hyps, dates_table, annual_columns)

# CREATE A SEPARATE OUTPUT FOR CALIBRATION with only data relevant to calibration
#   - annual glacier-wide massbal, area, ice thickness, snowline

#for glac in range(main_glac_rgi.shape[0]):
for glac in [0]:
    lr_gcm = input.lr_gcm
    lr_glac = input.lr_glac
    prec_factor = input.prec_factor
    prec_grad = input.prec_grad
    ddf_snow = input.DDF_snow
    ddf_ice = input.DDF_ice
    temp_snow = input.T_snow
    modelparameters = [lr_gcm, lr_glac, prec_factor, prec_grad, ddf_snow, ddf_ice, temp_snow]
    
    # Select subset of variables to reduce the amount of data being passed to the function
    glacier_rgi_table = main_glac_rgi.loc[glac, :]
    glacier_gcm_elev = main_glac_gcmelev.iloc[glac]
    glacier_gcm_prec = main_glac_gcmprec.iloc[glac,:].values
    glacier_gcm_temp = main_glac_gcmtemp.iloc[glac,:].values
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
                                            icethickness_t0, width_t0, glacier_gcm_temp, glacier_gcm_prec, glacier_gcm_elev, 
                                            elev_bins, dates_table, annual_columns, annual_divisor))
        glac_wide_massbalclim_allyrs = (glac_bin_area_annual[:,0:glac_bin_massbalclim_annual.shape[1]] * 
                                        glac_bin_massbalclim_annual).sum(axis=1).sum()/glacier_area_t0.sum()
        # Record variables from output package here - need to be in glacier loop since the variables will be overwritten 
        if input.output_package != 0:
            output.netcdfwrite(regionO1_number, glac, modelparameters, glacier_rgi_table, elev_bins, glac_bin_temp, 
                               glac_bin_prec, glac_bin_acc, glac_bin_refreeze, glac_bin_snowpack, glac_bin_melt, 
                               glac_bin_frontalablation, glac_bin_massbalclim, glac_bin_massbalclim_annual, 
                               glac_bin_area_annual, glac_bin_icethickness_annual, glac_bin_width_annual, 
                               glac_bin_surfacetype_annual)
        #  ADJUST NETCDF FILE FOR CALIBRATION
    elif input.option_calibration == 1:
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
            
            # Compute the glacier-wide climatic mass balance for all years
            glac_wide_massbalclim_allyrs = (glac_bin_area_annual[:,0:glac_bin_massbalclim_annual.shape[1]] * 
                                            glac_bin_massbalclim_annual).sum(axis=1).sum()/glacier_area_t0.sum()
            #  units: m w.e. based on initial area
            # Difference between geodetic and modeled mass balance
            massbal_difference = abs(main_glac_calmassbal[glac] - glac_wide_massbalclim_allyrs)
            return massbal_difference
        # Define constraints
        #  everything goes on one side of the equation compared to zero
        #  ex. return x[0] - input.lr_gcm with an equality means x[0] = input.lr_gcm
        def constraint1(modelparameters):
            return modelparameters[0] - input.lr_gcm
        def constraint2(modelparameters):
            return modelparameters[1] - input.lr_glac
        def constraint3(modelparameters):
            return modelparameters[2] - input.prec_factor
        def constraint4(modelparameters):
            return modelparameters[3] - input.prec_grad
        def constraint5(modelparameters):
            return modelparameters[4] - input.DDF_snow
        def constraint6(modelparameters):
            return modelparameters[5] - input.DDF_ice
        def constraint7(modelparameters):
            return modelparameters[6] - input.T_snow
        def ddfice2xsnow(modelparameters):
            return modelparameters[4] - 0.5*modelparameters[5] 
        def ddficegtsnow(modelparameters):
            return modelparameters[5] - modelparameters[4]
        # Define the initial guess
        modelparameters0 = ([input.lr_gcm, input.lr_glac, input.prec_factor, input.prec_grad, input.DDF_snow, 
                             input.DDF_ice, input.T_snow])
        # Define bounds
        lrgcm_bnds = (-0.007,-0.006)
        lrglac_bnds = (-0.007,-0.006)
        precfactor_bnds = (0.8,2.0)
        precgrad_bnds = (0.0001,0.00025)
        ddfsnow_bnds = (0.00175, 0.0045)
        ddfice_bnds = (0.003, 0.009)
        tempsnow_bnds = (0,2) 
        modelparameters_bnds = (lrgcm_bnds,lrglac_bnds,precfactor_bnds,precgrad_bnds,ddfsnow_bnds,ddfice_bnds,
                               tempsnow_bnds)
        # Define constraints
        con1 = {'type':'eq', 'fun':constraint1}
        con2 = {'type':'eq', 'fun':constraint2}
        con3 = {'type':'eq', 'fun':constraint3}
        con4 = {'type':'eq', 'fun':constraint4}
        con5 = {'type':'eq', 'fun':constraint5}
        con6 = {'type':'eq', 'fun':constraint6}
        con7 = {'type':'eq', 'fun':constraint7}
        con8 = {'type':'eq', 'fun':ddfice2xsnow}
        con9 = {'type':'ineq', 'fun':ddficegtsnow}
        # Select constraints used for calibration:
        if input.option_calibration_constraints == 1:
            # Option 1 - optimize all parameters
            cons = []
        elif input.option_calibration_constraints == 2: 
            # Option 2 - only optimize precfactor
            cons = [con1,con2,con4,con5,con6,con7]
        elif input.option_calibration_constraints == 3:
            # Option 3 - only optimize precfactor, DDFsnow, DDFice
            cons = [con1,con2,con4,con7]
        elif input.option_calibration_constraints == 4:
            # Option 4 - only optimize precfactor, DDFsnow, DDFice; DDFice = 2 x DDFsnow
            cons = [con1,con2,con4,con7,con8]
        elif input.option_calibration_constraints == 5:
            # Option 4 - only optimize precfactor, DDFsnow, DDFice; DDFice = 2 x DDFsnow
            cons = [con1,con2,con4,con7,con9]
        # Calibrate: all
        modelparameters_opt = minimize(objective,modelparameters0,method='SLSQP',bounds=modelparameters_bnds,constraints=cons)
        print(modelparameters_opt.x)
        
timeelapsed_step4 = timeit.default_timer() - timestart_step4
print('Step 4 time:', timeelapsed_step4, "s\n")

#%%=== STEP SIX: DATA ANALYSIS / OUTPUT ===============================================================================
#netcdf_output = nc.Dataset('../Output/PyGEM_output_rgiregion15_20180202.nc', 'r+')
#netcdf_output.close()

