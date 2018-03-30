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
#import pandas as pd
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
import argparse

#========== IMPORT INPUT AND FUNCTIONS FROM MODULES ===================================================================
import pygem_input as input
#import pygemfxns_modelsetup as modelsetup
#import pygemfxns_climate as climate
import pygemfxns_massbalance as massbalance
#import pygemfxns_output as output

#%% ===== ARGUMENT PARSER =============================================================================================
def getparser():
    parser = argparse.ArgumentParser(description="run glacier mass balance model for a given glacier")
    # Add arguments to the parser
    #  '--' before an argument indicates it's optional
    parser.add_argument('--precfactor', type=float, default=input.precfactor, help='Precipitation factor')
    parser.add_argument('--precgrad', type=float, default=input.precgrad, help='Precipitation gradient (% m-1)')
    parser.add_argument('--ddfsnow', type=float, default=input.ddfsnow, 
                        help='Degree day factor of snow (m w.e. degC-1 day-1')
    parser.add_argument('--tempchange', type=float, default=input.tempchange, 
                        help='Temperature change to correct for bias between GCM and glacier (degC)')
#    parser.add_argument('--lrgcm', type=float, default=input.lrgcm, help='Lapse rate from GCM to glacier')
#    parser.add_argument('--lrglac', type=float, default=input.lrglac, help='Lapse rate on the glacier')
    parser.add_argument('--dir_modelsetup', type=str, default=input.main_directory + '/../PyGEM_modelsetup/')
    # ADD RGIID, CLIMATE FILENAME, AND DATES TABLE FILENAME AS NON-OPTIONAL ARGUMENTS
    # ADD MODEL SETUP FILE DIRECTORY AS OPTIONAL ARGUMENT
    
    return parser

def main():
    parser = getparser()
    args = parser.parse_args()
    print(args.precfactor, args.precgrad, args.ddfsnow, args.tempchange)
    
    modelparameters = [input.lrgcm, input.lrglac, args.precfactor, args.precgrad, 
                       args.ddfsnow, input.ddfice, input.tempsnow, input.tempchange]

#    glacier_list = np.genfromtxt(input.main_directory + '/../PyGEM_modelsetup/glacier_list_R15_calmbonly.csv', dtype=str)
    RGIId = 'RGI60-15.03473'
    
    (glac_bin_temp, glac_bin_prec, glac_bin_acc, glac_bin_refreeze, glac_bin_snowpack, glac_bin_melt, 
     glac_bin_frontalablation, glac_bin_massbalclim, glac_bin_massbalclim_annual, glac_bin_area_annual, 
     glac_bin_icethickness_annual, glac_bin_width_annual, glac_bin_surfacetype_annual, 
     glac_wide_massbaltotal, glac_wide_runoff, glac_wide_snowline, glac_wide_snowpack, glac_wide_area_annual, 
     glac_wide_volume_annual, glac_wide_ELA_annual) = (
        massbalance.runmassbalance_v2(RGIId, modelparameters, args.dir_modelsetup, 
                                      '_ERAinterim_tple_1995_2015.csv', 'dates_table_1995_2015_monthly.csv'))

if __name__ == "__main__":
    main()

#%% ===== OLD SETUP (pre-03/30/2018) ==================================================================================
## ===== OUTPUT FILE =====
## Create output netcdf file
#if input.option_calibration == 0 and input.output_package != 0:
#    output_fullfilename = output.netcdfcreate(input.rgi_regionsO1[0], main_glac_hyps, dates_table, annual_columns)
#elif input.option_calibration == 1:
#    output_fullfilename = output.netcdfcreate_calgridsearch(input.rgi_regionsO1[0], main_glac_hyps, dates_table, 
#                                                            annual_columns, grid_modelparameters)

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
#    # Single run
#    elif input.option_calibration == 2:
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
#        # Output for Shane: Measured mass balance, Measured mass balance uncertainty, modeled mass balance, mass balance difference
#        print(glacier_rgi_table.loc[input.massbal_colname], glacier_rgi_table.loc[input.massbal_uncertainty_colname], 
#              glac_wide_massbalclim_mwea, massbal_difference)
#            # Compare calibration data
#            # Column index for start and end year based on dates of geodetic mass balance observations
#            massbal_idx_start = (glacier_rgi_table.loc[input.massbal_time1] - input.startyear).astype(int)
#            massbal_idx_end = (massbal_idx_start + glacier_rgi_table.loc[input.massbal_time2] - 
#                               glacier_rgi_table.loc[input.massbal_time1] + 1).astype(int)
#            massbal_years = massbal_idx_end - massbal_idx_start
#            # Average annual glacier-wide mass balance [m w.e.]
#            glac_wide_massbalclim_mwea = ((glac_bin_massbalclim_annual[:, massbal_idx_start:massbal_idx_end] *
#                                           glac_bin_area_annual[:, massbal_idx_start:massbal_idx_end]).sum() /
#                                           glacier_area_t0.sum() / massbal_years)
#            #  units: m w.e. based on initial area
#            # Difference between geodetic and modeled mass balance
#            massbal_difference = abs(glacier_rgi_table[input.massbal_colname] - glac_wide_massbalclim_mwea)
#            # Output: Measured mass balance, Measured mass balance uncertainty, modeled mass balance, mass balance difference
#            print(glacier_rgi_table.loc[input.massbal_colname], glacier_rgi_table.loc[input.massbal_uncertainty_colname], 
#                  glac_wide_massbalclim_mwea, massbal_difference)

#timeelapsed_step1 = timeit.default_timer() - timestart_step1
#print('\ntime:', timeelapsed_step1, "s\n")
#
###%%=== Model testing ===============================================================================
##import netCDF4 as nc
#output = nc.Dataset(input.main_directory + '/../Output/calibration_gridsearchcoarse_R15_20180324.nc', 'r+')
