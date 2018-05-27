r"""
run_simulation_list_multiprocess.py runs simulations for gcms and stores results in netcdf files.
"""

import pandas as pd
import numpy as np
import os
import argparse
import inspect
#import subprocess as sp
import multiprocessing
from scipy.optimize import minimize
import time
import matplotlib.pyplot as plt
from time import strftime
import xarray as xr
import netCDF4 as nc

import pygem_input as input
import pygemfxns_modelsetup as modelsetup
import pygemfxns_climate as climate
import pygemfxns_massbalance as massbalance
import pygemfxns_output as output

#%% ===== SCRIPT SPECIFIC INPUT DATA ===== 
# Glacier selection
rgi_regionsO1 = [15]
rgi_glac_number = 'all'
#rgi_glac_number = ['03473', '03733']
#rgi_glac_number = ['03473']
#rgi_glac_number = ['06881']
#rgi_glac_number=['10694']
#rgi_glac_number = ['00001', '00002', '00003', '00004', '00005', '00006', '00007', '00008', '03473', '03733']

# Required input
# Time period
gcm_startyear = 2000
gcm_endyear = 2100
gcm_spinupyears = 5
# GCM data
gcm_filepath_var_prefix = input.main_directory + '/../Climate_data/cmip5/'
gcm_filepath_var_ending = '_r1i1p1_monNG/'
gcm_filepath_fx_prefix = input.main_directory + '/../Climate_data/cmip5/'
gcm_filepath_fx_ending = '_r0i0p0_fx/'
gcm_temp_fn_prefix = 'tas_mon_'
gcm_prec_fn_prefix = 'pr_mon_'
gcm_var_ending = '_r1i1p1_native.nc'
gcm_elev_fn_prefix  = 'orog_fx_'
gcm_fx_ending  = '_r0i0p0.nc'
gcm_temp_varname = 'tas'
gcm_prec_varname = 'pr'
gcm_elev_varname = 'orog'
gcm_lat_varname = 'lat'
gcm_lon_varname = 'lon'
gcm_lr_fn = 'biasadj_mon_lravg_1995_2015_R15.csv'
# Calibrated model parameters
option_bias_adjustment = 1
#  0 - means no bias adjustment
modelparams_fp = input.main_directory + '/../Climate_data/cmip5/bias_adjusted_1995_2100/2018_0524/'
#modelparams_fn_body1 = '_biasadj_opt'
#modelparams_fn_body2 = '_1995_2015_R'
modelparams_fn_ending = '_biasadj_opt' + str(option_bias_adjustment) + '_1995_2015_R' + str(rgi_regionsO1[0]) + '.csv'
modelparams_colnames = ['lrgcm', 'lrglac', 'precfactor', 'precgrad', 'ddfsnow', 'ddfice', 'tempsnow', 'tempchange']
# Output
output_package = 2
output_filepath = input.main_directory + '/../Output/'

#%% FUNCTIONS
def getparser():
    parser = argparse.ArgumentParser(description="run gcm bias corrections from gcm list in parallel")
    # add arguments
    parser.add_argument('gcm_file', action='store', type=str, default='gcm_rcpXX_filenames.txt', 
                        help='text file full of commands to run')
    parser.add_argument('-num_simultaneous_processes', action='store', type=int, default=5, 
                        help='number of simultaneous processes (cores) to use')
    parser.add_argument('-option_parallels', action='store', type=int, default=1,
                        help='Switch to use or not use paralles (1 - use parallels, 0 - do not)')
    return parser


def main(list_packed_vars):
    # Unpack variables
    count = list_packed_vars[0]
    chunk = list_packed_vars[1]
    main_glac_rgi_all = list_packed_vars[2]
    chunk_size = list_packed_vars[3]
    gcm_name = list_packed_vars[4]
    
    time_start = time.time()
    parser = getparser()
    args = parser.parse_args()
    rcp_scenario = os.path.basename(args.gcm_file).split('_')[1]

    # ===== LOAD OTHER GLACIER DATA ===== 
    main_glac_rgi = main_glac_rgi_all.iloc[chunk:chunk + chunk_size, :]
    # Glacier hypsometry [km**2], total area
    main_glac_hyps = modelsetup.import_Husstable(main_glac_rgi, rgi_regionsO1, input.hyps_filepath, 
                                                 input.hyps_filedict, input.hyps_colsdrop)
    # Ice thickness [m], average
    main_glac_icethickness = modelsetup.import_Husstable(main_glac_rgi, rgi_regionsO1, input.thickness_filepath, 
                                                         input.thickness_filedict, input.thickness_colsdrop)
    main_glac_hyps[main_glac_icethickness == 0] = 0
    # Width [km], average
    main_glac_width = modelsetup.import_Husstable(main_glac_rgi, rgi_regionsO1, input.width_filepath, 
                                                  input.width_filedict, input.width_colsdrop)
    elev_bins = main_glac_hyps.columns.values.astype(int)
    # Model parameters
    modelparams_fn = (gcm_name + '_' + rcp_scenario + modelparams_fn_ending)
    main_glac_modelparams_all = pd.read_csv(modelparams_fp + modelparams_fn, index_col=0)    
    main_glac_modelparams = main_glac_modelparams_all.loc[main_glac_rgi['O1Index'].values, :] 
    
    # Select dates including future projections
    dates_table, start_date, end_date = modelsetup.datesmodelrun(startyear=gcm_startyear, endyear=gcm_endyear, 
                                                                 spinupyears=gcm_spinupyears)
    
    # ===== LOAD CLIMATE DATA =====
    gcm_filepath_var = gcm_filepath_var_prefix + rcp_scenario + gcm_filepath_var_ending
    gcm_filepath_fx = gcm_filepath_fx_prefix + rcp_scenario + gcm_filepath_fx_ending
    gcm_temp_fn = gcm_temp_fn_prefix + gcm_name + '_' + rcp_scenario + gcm_var_ending
    gcm_prec_fn = gcm_prec_fn_prefix + gcm_name + '_' + rcp_scenario + gcm_var_ending
    gcm_elev_fn = gcm_elev_fn_prefix + gcm_name + '_' + rcp_scenario + gcm_fx_ending
    gcm_temp, gcm_dates = climate.importGCMvarnearestneighbor_xarray(
            gcm_temp_fn, gcm_temp_varname, main_glac_rgi, dates_table, start_date, end_date, 
            filepath=gcm_filepath_var, gcm_lon_varname=gcm_lon_varname, gcm_lat_varname=gcm_lat_varname)
    gcm_prec, gcm_dates = climate.importGCMvarnearestneighbor_xarray(
            gcm_prec_fn, gcm_prec_varname, main_glac_rgi, dates_table, start_date, end_date, 
            filepath=gcm_filepath_var, gcm_lon_varname=gcm_lon_varname, gcm_lat_varname=gcm_lat_varname)
    gcm_elev = climate.importGCMfxnearestneighbor_xarray(
            gcm_elev_fn, gcm_elev_varname, main_glac_rgi, filepath=gcm_filepath_fx, 
            gcm_lon_varname=gcm_lon_varname, gcm_lat_varname=gcm_lat_varname)    
    # Monthly lapse rate
    ref_lr_monthly_avg = np.genfromtxt(modelparams_fp + gcm_lr_fn, delimiter=',')
    gcm_lr = np.tile(ref_lr_monthly_avg, int(gcm_temp.shape[1]/12))

    # ===== BIAS CORRECTIONS =====
    # Option 1
    if option_bias_adjustment == 1:
        gcm_temp_adj = gcm_temp + main_glac_modelparams['temp_adj'].values[:,np.newaxis]
        gcm_prec_adj = gcm_prec * main_glac_modelparams['prec_adj'].values[:,np.newaxis]
        gcm_elev_adj = gcm_elev
    # Option 2
    elif option_bias_adjustment == 2:
        tempvar_cols = []
        tempavg_cols = []
        tempadj_cols = []
        precadj_cols = []
        # Monthly adjustment parameters
        for n in range(1,13):
            tempvar_colname = 'tempvar_' + str(n)
            tempvar_cols.append(tempvar_colname)
            tempavg_colname = 'tempavg_' + str(n)
            tempavg_cols.append(tempavg_colname)
            tempadj_colname = 'tempadj_' + str(n)
            tempadj_cols.append(tempadj_colname)
            precadj_colname = 'precadj_' + str(n)
            precadj_cols.append(precadj_colname)
        bias_adj_prec = main_glac_modelparams[precadj_cols].values
        variability_monthly_std = main_glac_modelparams[tempvar_cols].values
        gcm_temp_monthly_avg = main_glac_modelparams[tempavg_cols].values
        gcm_temp_monthly_adj = main_glac_modelparams[tempadj_cols].values
        # Monthly temperature bias adjusted according to monthly average
        t_mt = gcm_temp + np.tile(gcm_temp_monthly_adj, int(gcm_temp.shape[1]/12))
        # Mean monthly temperature bias adjusted according to monthly average
        t_m25avg = np.tile(gcm_temp_monthly_avg + gcm_temp_monthly_adj, int(gcm_temp.shape[1]/12))
        # Bias adjusted temperature accounting for monthly mean and variability
        gcm_temp_adj = t_m25avg + (t_mt - t_m25avg) * np.tile(variability_monthly_std, int(gcm_temp.shape[1]/12))
        # Bias adjusted precipitation
        gcm_prec_adj = gcm_prec * np.tile(bias_adj_prec, int(gcm_temp.shape[1]/12))
        # Updated elevation, since adjusted according to reference elevation
        gcm_elev_adj = main_glac_modelparams['new_gcmelev'].values
    # Option 3
    elif option_bias_adjustment == 3:
        tempadj_cols = []
        precadj_cols = []
        # Monthly adjustment parameters
        for n in range(1,13):
            tempadj_colname = 'tempadj_' + str(n)
            tempadj_cols.append(tempadj_colname)
            precadj_colname = 'precadj_' + str(n)
            precadj_cols.append(precadj_colname)
        bias_adj_prec = main_glac_modelparams[precadj_cols].values
        bias_adj_temp = main_glac_modelparams[tempadj_cols].values
        # Bias adjusted temperature
        gcm_temp_adj = gcm_temp + np.tile(bias_adj_temp, int(gcm_temp.shape[1]/12))
        # Bias adjusted precipitation
        gcm_prec_adj = gcm_prec * np.tile(bias_adj_prec, int(gcm_temp.shape[1]/12))
        # Updated elevation, since adjusted according to reference elevation
        gcm_elev_adj = main_glac_modelparams['new_gcmelev'].values
    # Otherwise, no bias adjustment
    else:
        gcm_temp_adj = gcm_temp
        gcm_prec_adj = gcm_prec
        gcm_elev_adj = gcm_elev
        
    # ===== CREATE OUTPUT FILE =====
    if output_package != 0:
        netcdf_fn = ('PyGEM_R' + str(rgi_regionsO1[0]) + '_' + gcm_name + '_' + rcp_scenario + '_biasadj_opt' + 
                     str(option_bias_adjustment) + '_' + str(gcm_startyear - gcm_spinupyears) + '_' + str(gcm_endyear) 
                     + '_' + str(count) + '.nc')
        main_glac_rgi_float = main_glac_rgi.copy()
        main_glac_rgi_float.drop(labels=['RGIId'], axis=1, inplace=True)
        output.netcdfcreate(netcdf_fn, main_glac_rgi_float, main_glac_hyps, dates_table)

    # ===== RUN MASS BALANCE =====
    for glac in range(main_glac_rgi.shape[0]):
        if glac%200 == 0:
            print(gcm_name,':', main_glac_rgi.loc[main_glac_rgi.index.values[glac],'RGIId'])  
        # Select subsets of data
        glacier_rgi_table = main_glac_rgi.loc[main_glac_rgi.index.values[glac], :]
        glacier_gcm_elev = gcm_elev_adj[glac]
        glacier_gcm_prec = gcm_prec_adj[glac,:]
        glacier_gcm_temp = gcm_temp_adj[glac,:]
        glacier_gcm_lrgcm = gcm_lr[glac,:]
        glacier_gcm_lrglac = glacier_gcm_lrgcm.copy()
        glacier_area_t0 = main_glac_hyps.iloc[glac,:].values.astype(float)   
        icethickness_t0 = main_glac_icethickness.iloc[glac,:].values.astype(float)
        width_t0 = main_glac_width.iloc[glac,:].values.astype(float)
        modelparameters = main_glac_modelparams.loc[main_glac_modelparams.index.values[glac],modelparams_colnames]
        
        # Mass balance calcs
        (glac_bin_temp, glac_bin_prec, glac_bin_acc, glac_bin_refreeze, glac_bin_snowpack, glac_bin_melt, 
         glac_bin_frontalablation, glac_bin_massbalclim, glac_bin_massbalclim_annual, glac_bin_area_annual, 
         glac_bin_icethickness_annual, glac_bin_width_annual, glac_bin_surfacetype_annual, 
         glac_wide_massbaltotal, glac_wide_runoff, glac_wide_snowline, glac_wide_snowpack, 
         glac_wide_area_annual, glac_wide_volume_annual, glac_wide_ELA_annual) = (
            massbalance.runmassbalance(modelparameters, glacier_rgi_table, glacier_area_t0, icethickness_t0, 
                                       width_t0, elev_bins, glacier_gcm_temp, glacier_gcm_prec, glacier_gcm_elev, 
                                       glacier_gcm_lrgcm, glacier_gcm_lrglac, dates_table, option_calibration=0))
        # Annual glacier-wide mass balance [m w.e.]
        glac_wide_massbaltotal_annual = np.sum(glac_wide_massbaltotal.reshape(-1,12), axis=1)
        # Average annual glacier-wide mass balance [m w.e.a.]
        mb_mwea = glac_wide_massbaltotal_annual.mean()
        #  units: m w.e. based on initial area
        # Volume change [%]
        if icethickness_t0.max() > 0:
            glac_vol_change_perc = ((glac_wide_volume_annual[-1] - glac_wide_volume_annual[0]) / 
                                    glac_wide_volume_annual[0] * 100)
            
#        print(mb_mwea, glac_vol_change_perc)

        if output_package != 0:
            output.netcdfwrite(netcdf_fn, glac, modelparameters, glacier_rgi_table, elev_bins, glac_bin_temp, 
                               glac_bin_prec, glac_bin_acc, glac_bin_refreeze, glac_bin_snowpack, glac_bin_melt, 
                               glac_bin_frontalablation, glac_bin_massbalclim, glac_bin_massbalclim_annual, 
                               glac_bin_area_annual, glac_bin_icethickness_annual, glac_bin_width_annual,
                               glac_bin_surfacetype_annual, output_filepath=output_filepath)

    # Export variables as global to view in variable explorer
    if (args.option_parallels == 0) or (main_glac_rgi_all.shape[0] < 2 * args.num_simultaneous_processes):
        global main_vars
        main_vars = inspect.currentframe().f_locals

    print('\nProcessing time of', gcm_name, 'for', count,':',time.time()-time_start, 's')

#%% PARALLEL PROCESSING
if __name__ == '__main__':
    time_start = time.time()
    parser = getparser()
    args = parser.parse_args()
    
    # Select glaciers and define chunks
    main_glac_rgi_all = modelsetup.selectglaciersrgitable(rgi_regionsO1=rgi_regionsO1, rgi_regionsO2 = 'all', 
                                                          rgi_glac_number=rgi_glac_number)
    # Processing needed for netcdf files
    main_glac_rgi_all['RGIId_float'] = (np.array([np.str.split(main_glac_rgi_all['RGIId'][x],'-')[1] 
                                              for x in range(main_glac_rgi_all.shape[0])]).astype(float))
    main_glac_rgi_all_float = main_glac_rgi_all.copy()
    main_glac_rgi_all_float.drop(labels=['RGIId'], axis=1, inplace=True)
    main_glac_hyps = modelsetup.import_Husstable(main_glac_rgi_all, rgi_regionsO1, input.hyps_filepath, 
                                                 input.hyps_filedict, input.hyps_colsdrop)
    dates_table, start_date, end_date = modelsetup.datesmodelrun(startyear=gcm_startyear, endyear=gcm_endyear, 
                                                                 spinupyears=gcm_spinupyears)
    
    if (args.option_parallels != 0) and (main_glac_rgi_all.shape[0] >= 2 * args.num_simultaneous_processes):
        chunk_size = int(np.ceil(main_glac_rgi_all.shape[0] / args.num_simultaneous_processes))
    else:
        chunk_size = main_glac_rgi_all.shape[0]
    
    # Read GCM names from command file
    with open(args.gcm_file, 'r') as gcm_fn:
        gcm_list = gcm_fn.read().splitlines()
        rcp_scenario = os.path.basename(args.gcm_file).split('_')[1]
        print('Found %d gcms to process'%(len(gcm_list)))
        
    # Loop through all GCMs
    for gcm_name in gcm_list:
        print('Processing:', gcm_name)
        # Pack variables for multiprocessing
        list_packed_vars = [] 
        n = 0
        for chunk in range(0, main_glac_rgi_all.shape[0], chunk_size):
            n = n + 1
            list_packed_vars.append([n, chunk, main_glac_rgi_all, chunk_size, gcm_name])
        
        # Parallel processing
        if (args.option_parallels != 0) and (main_glac_rgi_all.shape[0] >= 2 * args.num_simultaneous_processes):
            with multiprocessing.Pool(args.num_simultaneous_processes) as p:
                p.map(main,list_packed_vars)
        
        # No parallel processing
        else:
            # Loop through the chunks and export bias adjustments
            for n in range(len(list_packed_vars)):
                main(list_packed_vars[n])
    
         # Combine output into single package and export lapse rate if necessary
        if (args.option_parallels != 0) and (main_glac_rgi_all.shape[0] >= 2 * args.num_simultaneous_processes):
            # Netcdf outputs
            output_prefix = ('PyGEM_R' + str(rgi_regionsO1[0]) + '_' + gcm_name + '_' + rcp_scenario + '_biasadj_opt' + 
                             str(option_bias_adjustment) + '_' + str(gcm_startyear - gcm_spinupyears) + '_' + 
                             str(gcm_endyear) + '_' + 'test')
            output_all_fn = ('PyGEM_R' + str(rgi_regionsO1[0]) + '_' + gcm_name + '_' + rcp_scenario + '_biasadj_opt' + 
                             str(option_bias_adjustment) + '_' + str(gcm_startyear - gcm_spinupyears) + '_' + 
                             str(gcm_endyear) + '_all.nc')
            
            # Select netcdf files produced in parallel
            output_list = []
            for i in os.listdir(output_filepath):
                # Append bias adjustment results
                if i.startswith(output_prefix) == True:
                    output_list.append(i)
            
            # Merge netcdfs together
            if (len(output_list) > 1) and (output_package != 0):
                # Create netcdf that will have them all together
                output.netcdfcreate(output_all_fn, main_glac_rgi_all_float, main_glac_hyps, dates_table, 
                                    output_filepath=input.output_filepath)
                # Open file to write
                netcdf_output = nc.Dataset(output_filepath + output_all_fn, 'r+')
                
                glac_count = -1
                for n in range(len(output_list)):
                    ds = nc.Dataset(output_filepath + output_list[n])
                    for glac in range(ds['glac_idx'][:].shape[0]):
                        glac_count = glac_count + 1
                        if output_package == 2:
                            netcdf_output.variables['temp_glac_monthly'][glac_count,:] = (
                                    ds['temp_glac_monthly'][glac,:])
                            netcdf_output.variables['prec_glac_monthly'][glac_count,:] = (
                                    ds['prec_glac_monthly'][glac,:])
                            netcdf_output.variables['acc_glac_monthly'][glac_count,:] = (
                                    ds['acc_glac_monthly'][glac,:])
                            netcdf_output.variables['refreeze_glac_monthly'][glac_count,:] = (
                                    ds['refreeze_glac_monthly'][glac,:])
                            netcdf_output.variables['melt_glac_monthly'][glac_count,:] = (
                                    ds['melt_glac_monthly'][glac,:])
                            netcdf_output.variables['frontalablation_glac_monthly'][glac_count,:] = (
                                    ds['frontalablation_glac_monthly'][glac,:])
                            netcdf_output.variables['massbaltotal_glac_monthly'][glac_count,:] = (
                                    ds['massbaltotal_glac_monthly'][glac,:])
                            netcdf_output.variables['runoff_glac_monthly'][glac_count,:] = (
                                    ds['runoff_glac_monthly'][glac,:])
                            netcdf_output.variables['snowline_glac_monthly'][glac_count,:] = (
                                    ds['snowline_glac_monthly'][glac,:])
                            netcdf_output.variables['area_glac_annual'][glac_count,:] = (
                                    ds['area_glac_annual'][glac,:])
                            netcdf_output.variables['volume_glac_annual'][glac_count,:] = (
                                    ds['volume_glac_annual'][glac,:])
                            netcdf_output.variables['ELA_glac_annual'][glac_count,:] = (
                                    ds['ELA_glac_annual'][glac,:])
                        else:
                            print('Code merge for output package')  
                    ds.close()
                    # Remove file after its been merged
                    os.remove(output_filepath + output_list[n])
                # Close the netcdf file
                netcdf_output.close()
        
    print('Total processing time:', time.time()-time_start, 's')
            
#%% ===== PLOTTING AND PROCESSING FOR MODEL DEVELOPMENT =====          
    # Place local variables in variable explorer
    if (args.option_parallels == 0) or (main_glac_rgi_all.shape[0] < 2 * args.num_simultaneous_processes):
        main_vars_list = list(main_vars.keys())
        gcm_name = main_vars['gcm_name']
        rcp_scenario = main_vars['rcp_scenario']
        main_glac_rgi = main_vars['main_glac_rgi']
        main_glac_hyps = main_vars['main_glac_hyps']
        main_glac_icethickness = main_vars['main_glac_icethickness']
        main_glac_width = main_vars['main_glac_width']
        main_glac_modelparams = main_vars['main_glac_modelparams']
        elev_bins = main_vars['elev_bins']
        dates_table = main_vars['dates_table']
        gcm_temp = main_vars['gcm_temp']
        gcm_prec = main_vars['gcm_prec']
        gcm_elev = main_vars['gcm_elev']
        gcm_temp_adj = main_vars['gcm_temp_adj']
        gcm_prec_adj = main_vars['gcm_prec_adj']
        gcm_elev_adj = main_vars['gcm_elev_adj']
        modelparameters = main_vars['modelparameters']
        glac_wide_massbaltotal = main_vars['glac_wide_massbaltotal']
        glac_wide_area_annual = main_vars['glac_wide_area_annual']
        glac_wide_volume_annual = main_vars['glac_wide_volume_annual']
        glacier_rgi_table = main_vars['glacier_rgi_table']
        glacier_gcm_temp = main_vars['glacier_gcm_temp'][gcm_spinupyears*12:]
        glacier_gcm_prec = main_vars['glacier_gcm_prec'][gcm_spinupyears*12:]
        glacier_gcm_elev = main_vars['glacier_gcm_elev']
        glacier_gcm_lrgcm = main_vars['glacier_gcm_lrgcm'][gcm_spinupyears*12:]    
        
    #    # Adjust temperature and precipitation to 'Zmed' so variables can properly be compared
    #    glacier_elev_zmed = glacier_rgi_table.loc['Zmed']  
    #    glacier_gcm_temp_zmed = glacier_gcm_temp + glacier_gcm_lrgcm * (glacier_elev_zmed - glacier_gcm_elev)
    #    glacier_gcm_prec_zmed = glacier_gcm_prec * modelparameters['precfactor']
    #    
    #    glac_wide_massbaltotal_annual = glac_wide_massbaltotal.reshape(-1,12).sum(axis=1)
    #    # Plot reference vs. GCM temperature and precipitation
    #    # Monthly trends
    #    months = dates_table['date'][gcm_spinupyears*12:]
    #    years = np.unique(dates_table['wateryear'].values)[gcm_spinupyears:]
    #    
    #    # Temperature
    #    plt.plot(months, glacier_gcm_temp_zmed, label='gcm_temp')
    #    plt.ylabel('Monthly temperature [degC]')
    #    plt.legend()
    #    plt.show()
    #    # Precipitation
    #    plt.plot(months, glacier_gcm_prec_zmed, label='gcm_prec')
    #    plt.ylabel('Monthly precipitation [m]')
    #    plt.legend()
    #    plt.show()
    #    
    #    # Annual trends
    #    glacier_gcm_temp_zmed_annual = glacier_gcm_temp_zmed.reshape(-1,12).mean(axis=1)
    #    glacier_gcm_prec_zmed_annual = glacier_gcm_prec_zmed.reshape(-1,12).sum(axis=1)
    #    # Temperature
    #    plt.plot(years, glacier_gcm_temp_zmed_annual, label='gcm_temp')
    #    plt.ylabel('Mean annual temperature [degC]')
    #    plt.legend()
    #    plt.show()
    #    # Precipitation
    #    plt.plot(years, glacier_gcm_prec_zmed_annual, label='gcm_prec')
    #    plt.ylabel('Total annual precipitation [m]')
    #    plt.legend()
    #    plt.show()
    #    # Mass balance - bar plot
    #    bar_width = 0.35
    #    plt.bar(years+bar_width, glac_wide_massbaltotal_annual, bar_width, label='gcm_MB')
    #    plt.ylabel('Glacier-wide mass balance [mwea]')
    #    plt.legend()
    #    plt.show()
    #    # Cumulative mass balance - bar plot
    #    glac_wide_massbaltotal_annual_cumsum = np.cumsum(glac_wide_massbaltotal_annual)
    #    bar_width = 0.35
    #    plt.bar(years+bar_width, glac_wide_massbaltotal_annual_cumsum, bar_width, label='gcm_MB')
    #    plt.ylabel('Cumulative glacier-wide mass balance [mwe]')
    #    plt.legend()
    #    plt.show() 
        
    #    # Histogram of differences
    #    mb_dif = main_glac_bias_adj['ref_mb_mwea'] - main_glac_bias_adj['gcm_mb_mwea']
    #    plt.hist(mb_dif)
    #    plt.show()      