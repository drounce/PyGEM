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
import os # os is used with re to find name matches
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
import csv
import itertools
import pickle

#========== IMPORT INPUT AND FUNCTIONS FROM MODULES ===================================================================
import pygem_input as input
import pygemfxns_modelsetup as modelsetup
#import pygemfxns_climate as climate
import pygemfxns_massbalance as massbalance
#import pygemfxns_output as output

#%% ===== ARGUMENT PARSER =============================================================================================
def getparser():
    parser = argparse.ArgumentParser(description="run glacier mass balance model for a given glacier")
    # Add arguments to the parser
    #  '--' before an argument indicates it's optional
    parser.add_argument('--precfactor', type=float, default=input.precfactor, help='Precipitation factor')
    parser.add_argument('--precgrad', type=float, default=input.precgrad, help='Precipitation gradient [% m-1]')
    parser.add_argument('--ddfsnow', type=float, default=input.ddfsnow, 
                        help='Degree day factor of snow [m w.e. d-1 degC-1]')
    parser.add_argument('--ddfice', type=float, default=input.ddfsnow / input.ddfsnow_iceratio, 
                        help='Degree day factor of ice [m w.e. d-1 degC-1]')
    parser.add_argument('--tempchange', type=float, default=input.tempchange, 
                        help='Temperature adjustment to correct for bias between GCM and glacier [degC]')
    parser.add_argument('--lrgcm', type=float, default=input.lrgcm, help='Lapse rate from gcm to glacier [K m-1]')
    parser.add_argument('--lrglac', type=float, default=input.lrglac, help='Lapse rate on the glacier for bins [K m-1]')
    parser.add_argument('--tempsnow', type=float, default=input.tempsnow, 
                        help='Temperature threshold to determine liquid or solid precipitation')
    parser.add_argument('RGIId', type=str, help='RGIId (ex. RGI60-15.03473)')
    return parser

def main():
    parser = getparser()
    args = parser.parse_args()
    
    modelparameters = [args.lrgcm, args.lrglac, args.precfactor, args.precgrad, args.ddfsnow, args.ddfsnow / 0.7, 
                       args.tempsnow, args.tempchange]
    print(modelparameters)
    # Set up directory for files
    if os.path.exists(input.modelsetup_dir) == False:
        os.makedirs(input.modelsetup_dir)    
    
    # LOAD DATA 
    # ===== RGI INFO =====
    if os.path.exists(input.modelsetup_dir + args.RGIId + '_rgi_table.pk') == True:
        glacier_rgi_table = pd.read_pickle(input.modelsetup_dir + args.RGIId + '_rgi_table.pk')
        icethickness_t0 = pd.read_pickle(input.modelsetup_dir + args.RGIId + '_icethickness.pk')
        glacier_area_t0 = pd.read_pickle(input.modelsetup_dir + args.RGIId + '_glacierarea.pk')
        width_t0 = pd.read_pickle(input.modelsetup_dir + args.RGIId + '_width.pk')
        elev_bins = np.load(input.modelsetup_dir + 'elevbins.pk')
        glacier_gcm_temp = np.load(input.modelsetup_dir + args.RGIId + '_gcmtemp.pk')
        glacier_gcm_prec = np.load(input.modelsetup_dir + args.RGIId + '_gcmprec.pk')
        glacier_gcm_elev = np.load(input.modelsetup_dir + args.RGIId + '_gcmelev.pk')
        glacier_gcm_lrgcm = np.load(input.modelsetup_dir + args.RGIId + '_gcmlr.pk')
        glacier_gcm_lrglac = glacier_gcm_lrgcm.copy()
        dates_table = pd.read_pickle(input.modelsetup_dir + 'dates_table.pk')
        print('Imported pickled data')
    else:
        with open(input.rgi_filepath + input.rgi_dict[input.rgi_regionsO1[0]], 'r') as file_rgi:
            rgi_reader = csv.reader(file_rgi, delimiter=',')
            header = next(itertools.islice(rgi_reader, 0, None))
            for row in rgi_reader:
                if args.RGIId == row[0]:
                    glacier_rgi_table_raw = pd.Series(row, index=header)
                    # convert values to floats
                    glacier_rgi_table = pd.to_numeric(glacier_rgi_table_raw, errors='coerce')
                    # overwrite RGIId (otherwise it would be NaN)
                    glacier_rgi_table['RGIId'] = glacier_rgi_table_raw['RGIId']
                    # record position number to index climate and glacier data
                    #  subtract 1 to account for line_num returning 1 for python's 0 row
                    glacier_rgi_table['O1Line'] = rgi_reader.line_num - 1                
        # Load calibration data and append to glacier table
        if input.option_calibration == 1:
            # Import .csv file
            ds = pd.read_csv(input.cal_mb_filepath + input.cal_mb_filedict[input.rgi_regionsO1[0]])
            ds[input.rgi_O1Id_colname] = ((ds[input.cal_rgi_colname] % 1) * 10**5).round(0).astype(int) 
            ds_subset = ds[[input.rgi_O1Id_colname, input.massbal_colname, input.massbal_uncertainty_colname, 
                            input.massbal_time1, input.massbal_time2]].values
            try:
                glacier_calmassbal = (ds_subset[np.where(np.in1d(ds_subset[:,0], 
                                                                 glacier_rgi_table.loc['O1Line'])==True)[0][0],1:])
            except:
                glacier_calmassbal = np.empty(4)
                glacier_calmassbal = np.nan
            glacier_rgi_table[input.massbal_colname] = glacier_calmassbal[0]
            glacier_rgi_table[input.massbal_uncertainty_colname] = glacier_calmassbal[1]
            glacier_rgi_table[input.massbal_time1] = glacier_calmassbal[2]
            glacier_rgi_table[input.massbal_time2] = glacier_calmassbal[3]
            
        # ===== GLACIER PROPERTIES ===== 
        # Ice thickness [m]
        with open(input.thickness_filepath + input.thickness_filedict[input.rgi_regionsO1[0]], 'r') as file_thickness:
            icethickness_t0 = (np.array(next(itertools.islice(csv.reader(file_thickness), glacier_rgi_table.loc['O1Line'], 
                                                              None))[2:]).astype(float))
            icethickness_t0[icethickness_t0==-99] = 0
        # Glacier area [km**2]
        with open(input.hyps_filepath + input.hyps_filedict[input.rgi_regionsO1[0]], 'r') as file_hyps:
            glacier_area_t0 = (np.array(next(itertools.islice(csv.reader(file_hyps), glacier_rgi_table.loc['O1Line'], 
                                                              None))[2:]).astype(float))
            glacier_area_t0[glacier_area_t0==-99] = 0
            # if ice thickness = 0, glacier area = 0 (problem identified by glacier RGIV6-15.00016 on 03/06/2018)
            glacier_area_t0[icethickness_t0==0] = 0
        # Elevation bins
        #  requires separate open or else the iterator is thrown off and it does not return the proper value
        with open(input.hyps_filepath + input.hyps_filedict[input.rgi_regionsO1[0]], 'r') as file_hyps2:
            elev_bins = np.array(next(itertools.islice(csv.reader(file_hyps2), 0, None))[2:]).astype(int)
        # Glacier width [km]
        with open(input.width_filepath + input.width_filedict[input.rgi_regionsO1[0]], 'r') as file_width:
            width_t0 = (np.array(next(itertools.islice(csv.reader(file_width), glacier_rgi_table.loc['O1Line'], 
                                                       None))[2:]).astype(float))
            width_t0[width_t0==-99] = 0.
        # Add volume [km**3] and mean elevation [m a.s.l.] to glacier table
        glacier_rgi_table['Volume'] = (glacier_area_t0 * icethickness_t0/1000).sum()
        glacier_rgi_table['Zmean'] = (glacier_area_t0 * elev_bins).sum() / glacier_area_t0.sum()
        
        # ===== CLIMATE DATA =====
        #  Subtract one from the 'O1Line' to account for the climate data not having a header
        # Temperature [degC]
        with open(input.gcm_filepath_var + input.gcmtemp_filedict[input.rgi_regionsO1[0]], 'r') as file_temp:
            glacier_gcm_temp = (np.array(next(itertools.islice(csv.reader(file_temp), glacier_rgi_table.loc['O1Line'] - 1, 
                                                               None))).astype(float))
        # Precipitation [m]
        with open(input.gcm_filepath_var + input.gcmprec_filedict[input.rgi_regionsO1[0]], 'r') as file_prec:
            glacier_gcm_prec = (np.array(next(itertools.islice(csv.reader(file_prec), glacier_rgi_table.loc['O1Line'] - 1, 
                                                               None))).astype(float))
        # Elevation [m a.s.l] associated with air temperature  and precipitation data
        with open(input.gcm_filepath_var + input.gcmelev_filedict[input.rgi_regionsO1[0]], 'r') as file_elev:
            glacier_gcm_elev = float(next(itertools.islice(csv.reader(file_elev), glacier_rgi_table.loc['O1Line'] - 1, 
                                                           None))[0])
        # Lapse rates [degC m-1] 
        with open(input.gcm_filepath_var + input.gcmlapserate_filedict[input.rgi_regionsO1[0]], 'r') as file_lr:
            glacier_gcm_lrgcm = (np.array(next(itertools.islice(csv.reader(file_lr), glacier_rgi_table.loc['O1Line'] - 1, 
                                                                None))).astype(float)) 
            glacier_gcm_lrglac = glacier_gcm_lrgcm.copy()
        # ===== MODEL TIME FRAME =====
        dates_table, start_date, end_date = modelsetup.datesmodelrun()
    
        # PICKLE DATA
        with open(input.modelsetup_dir + args.RGIId + '_rgi_table.pk', 'wb') as pickle_rgi:
            # dump your data into the file
            pickle.dump(glacier_rgi_table, pickle_rgi)
            print('Data pickled')
        with open(input.modelsetup_dir + args.RGIId + '_icethickness.pk', 'wb') as pickle_icethickness:
            # dump your data into the file
            pickle.dump(icethickness_t0, pickle_icethickness)
        with open(input.modelsetup_dir + args.RGIId + '_glacierarea.pk', 'wb') as pickle_glacierarea:
            # dump your data into the file
            pickle.dump(glacier_area_t0, pickle_glacierarea)    
        with open(input.modelsetup_dir + args.RGIId + '_width.pk', 'wb') as pickle_width:
            # dump your data into the file
            pickle.dump(width_t0, pickle_width)
        with open(input.modelsetup_dir + 'elevbins.pk', 'wb') as pickle_elevbins:
            # dump your data into the file
            pickle.dump(elev_bins, pickle_elevbins)
        with open(input.modelsetup_dir + args.RGIId + '_gcmtemp.pk', 'wb') as pickle_gcmtemp:
            # dump your data into the file
            pickle.dump(glacier_gcm_temp, pickle_gcmtemp)
        with open(input.modelsetup_dir + args.RGIId + '_gcmprec.pk', 'wb') as pickle_gcmprec:
            # dump your data into the file
            pickle.dump(glacier_gcm_prec, pickle_gcmprec)
        with open(input.modelsetup_dir + args.RGIId + '_gcmelev.pk', 'wb') as pickle_gcmelev:
            # dump your data into the file
            pickle.dump(glacier_gcm_elev, pickle_gcmelev)
        with open(input.modelsetup_dir + args.RGIId + '_gcmlr.pk', 'wb') as pickle_gcmlr:
            # dump your data into the file
            pickle.dump(glacier_gcm_lrgcm, pickle_gcmlr)
        with open(input.modelsetup_dir + 'dates_table.pk', 'wb') as pickle_datestable:
            # dump your data into the file
            pickle.dump(dates_table, pickle_datestable)


    # ===== RUN MASS BALANCE MODEL ===== 
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
    
    print(glacier_rgi_table.loc[input.massbal_colname], glacier_rgi_table.loc[input.massbal_uncertainty_colname], 
          glac_wide_massbaltotal_annual_avg, massbal_difference)
    
    # Return desired output
    return (glacier_rgi_table, icethickness_t0, glacier_area_t0, width_t0, elev_bins, glacier_gcm_temp, 
            glacier_gcm_prec, glacier_gcm_elev, glacier_gcm_lrgcm, glacier_gcm_lrglac, dates_table, 
            glac_wide_massbaltotal, glac_wide_runoff, glac_wide_snowline, glac_wide_snowpack, glac_wide_area_annual, 
            glac_wide_volume_annual, glac_wide_ELA_annual, modelparameters)
    
    
if __name__ == "__main__":
#    timestart_step1 = timeit.default_timer()
    
    (glacier_rgi_table, icethickness_t0, glacier_area_t0, width_t0, elev_bins, glacier_gcm_temp, 
    glacier_gcm_prec, glacier_gcm_elev, glacier_gcm_lrgcm, glacier_gcm_lrglac, dates_table, 
    glac_wide_massbaltotal, glac_wide_runoff, glac_wide_snowline, glac_wide_snowpack, glac_wide_area_annual, 
    glac_wide_volume_annual, glac_wide_ELA_annual, modelparameters) = main()
    
#    # Compare calibration data
#    # Column index for start and end year based on dates of geodetic mass balance observations
#    massbal_idx_start = int(glacier_rgi_table.loc[input.massbal_time1] - input.startyear)
#    massbal_idx_end = int(massbal_idx_start + glacier_rgi_table.loc[input.massbal_time2] - 
#                          glacier_rgi_table.loc[input.massbal_time1] + 1)
#    # Annual glacier-wide mass balance [m w.e.]
#    glac_wide_massbaltotal_annual = np.sum(glac_wide_massbaltotal.reshape(-1,12), axis=1)
#    # Average annual glacier-wide mass balance [m w.e.a.]
#    glac_wide_massbaltotal_annual_avg = glac_wide_massbaltotal_annual[massbal_idx_start:massbal_idx_end].mean()
#
#    #  units: m w.e. based on initial area
#    # Difference between geodetic and modeled mass balance
#    massbal_difference = abs(glacier_rgi_table[input.massbal_colname] - glac_wide_massbaltotal_annual_avg)
#    
#    print(glacier_rgi_table.loc[input.massbal_colname], glacier_rgi_table.loc[input.massbal_uncertainty_colname], 
#          glac_wide_massbaltotal_annual_avg, massbal_difference)
#
#
#    timeelapsed_step1 = timeit.default_timer() - timestart_step1
#    print('\ntime:', timeelapsed_step1, "s\n")
#    
#%% ===== OLD SETUP (keep for output) =================================================================================
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
