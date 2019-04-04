#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 10:30:23 2019

@author: davidrounce
"""

# Built-in libraries
import argparse
import inspect
import multiprocessing
import os
import pickle
# External libraries
import numpy as np
import pandas as pd
import xarray as xr
# Local libraries
import class_climate
import class_mbdata
import pygem_input as input
import pygemfxns_gcmbiasadj as gcmbiasadj
import pygemfxns_massbalance as massbalance
import pygemfxns_modelsetup as modelsetup
import run_calibration as calibration



def getparser():
    """
    Use argparse to add arguments from the command line
    
    Parameters
    ----------
    ref_gcm_name (optional) : str
        reference gcm name
    num_simultaneous_processes (optional) : int
        number of cores to use in parallels
    option_parallels (optional) : int
        switch to use parallels or not
    spc_region (optional) : str
        RGI region number for supercomputer 
    rgi_glac_number_fn : str
        filename of .pkl file containing a list of glacier numbers that used to run batches on the supercomputer
        
    Returns
    -------
    Object containing arguments and their respective values.
    """
    parser = argparse.ArgumentParser(description="run calibration in parallel")
    # add arguments
    parser.add_argument('-ref_gcm_name', action='store', type=str, default=input.ref_gcm_name,
                        help='reference gcm name')
    parser.add_argument('-num_simultaneous_processes', action='store', type=int, default=4,
                        help='number of simultaneous processes (cores) to use')
    parser.add_argument('-option_parallels', action='store', type=int, default=1,
                        help='Switch to use or not use parallels (1 - use parallels, 0 - do not)')
    parser.add_argument('-spc_region', action='store', type=int, default=None,
                        help='rgi region number for supercomputer')
    parser.add_argument('-rgi_glac_number_fn', action='store', type=str, default=None,
                        help='Filename containing list of rgi_glac_number, helpful for running batches on spc')
    return parser
    
    
    
def main(list_packed_vars):
    """
    Add priors to calibrated netcdf
    
    Parameters
    ----------
    list_packed_vars : list
        list of packed variables that enable the use of parallels
        
    Returns
    -------
    netcdf files with priors added
    """         
    # Unpack variables
    count = list_packed_vars[0]
    gcm_name = list_packed_vars[1] 
    main_glac_rgi = list_packed_vars[2]
    main_glac_hyps = list_packed_vars[3]
    main_glac_icethickness = list_packed_vars[4]
    main_glac_width = list_packed_vars[5]
    gcm_temp = list_packed_vars[6]
    gcm_prec = list_packed_vars[7]
    gcm_elev = list_packed_vars[8]
    gcm_lr = list_packed_vars[9]
    cal_data = list_packed_vars[10]
    
    parser = getparser()
    args = parser.parse_args()
    
        
    # RGI region
    if args.spc_region is not None:
        rgi_regionsO1 = [int(args.spc_region)]
    else:
        rgi_regionsO1 = input.rgi_regionsO1
        
    # ===== Begin MCMC process =====
    # loop through each glacier selected
    for glac in range(main_glac_rgi.shape[0]):

        if glac%500 == 0:
            print(count, main_glac_rgi.loc[main_glac_rgi.index.values[glac],'RGIId_float'])

        # Set model parameters
        modelparameters = [input.lrgcm, input.lrglac, input.precfactor, input.precgrad, input.ddfsnow, input.ddfice,
                           input.tempsnow, input.tempchange]

        # Select subsets of data
        glacier_rgi_table = main_glac_rgi.loc[main_glac_rgi.index.values[glac], :]
        glacier_gcm_elev = gcm_elev[glac]
        glacier_gcm_prec = gcm_prec[glac,:]
        glacier_gcm_temp = gcm_temp[glac,:]
        glacier_gcm_lrgcm = gcm_lr[glac,:]
        glacier_gcm_lrglac = glacier_gcm_lrgcm.copy()
        glacier_area_t0 = main_glac_hyps.iloc[glac,:].values.astype(float)
        icethickness_t0 = main_glac_icethickness.iloc[glac,:].values.astype(float)
        width_t0 = main_glac_width.iloc[glac,:].values.astype(float)
        glacier_cal_data = ((cal_data.iloc[np.where(
                glacier_rgi_table[input.rgi_O1Id_colname] == cal_data['glacno'])[0],:]).copy())
        glacier_str = '{0:0.5f}'.format(glacier_rgi_table['RGIId_float'])

        # Select observed mass balance, error, and time data
        cal_idx = glacier_cal_data.index.values[0]
        #  Note: index to main_glac_rgi may differ from cal_idx
        t1 = glacier_cal_data.loc[cal_idx, 't1']
        t2 = glacier_cal_data.loc[cal_idx, 't2']
        t1_idx = int(glacier_cal_data.loc[cal_idx,'t1_idx'])
        t2_idx = int(glacier_cal_data.loc[cal_idx,'t2_idx'])
        # Observed mass balance [mwea]
        observed_massbal = glacier_cal_data.loc[cal_idx,'mb_mwe'] / (t2 - t1)
        observed_error = glacier_cal_data.loc[cal_idx,'mb_mwe_err'] / (t2 - t1)
        mb_obs_max = observed_massbal + 3 * observed_error
        mb_obs_min = observed_massbal - 3 * observed_error


#        print('observed_massbal:',observed_massbal, 'observed_error:',observed_error)
            
        # Retrieve priors
        tempchange_mu = input.tempchange_mu
        tempchange_sigma = input.tempchange_sigma
        tempchange_boundlow = input.tempchange_boundlow
        tempchange_boundhigh = input.tempchange_boundhigh
        precfactor_boundlow = input.precfactor_boundlow
        precfactor_boundhigh = input.precfactor_boundhigh
        precfactor_mu = (precfactor_boundlow + precfactor_boundhigh) / 2
        ddfsnow_boundlow = input.ddfsnow_boundlow
        ddfsnow_boundhigh=input.ddfsnow_boundhigh
        ddfsnow_mu = input.ddfsnow_mu
        ddfsnow_sigma = input.ddfsnow_sigma
        if input.new_setup == 1 and icethickness_t0.max() > 0:             
            (precfactor_boundlow, precfactor_boundhigh, precfactor_mu, precfactor_start, tempchange_boundlow, 
             tempchange_boundhigh, tempchange_mu, tempchange_sigma, tempchange_start, tempchange_max_loss, 
             tempchange_max_acc, mb_max_loss, mb_max_acc, precfactor_opt_init, tempchange_opt_init) = (
                     calibration.retrieve_prior_parameters(
                             modelparameters, glacier_rgi_table, glacier_area_t0, icethickness_t0, 
                             width_t0, elev_bins, glacier_gcm_temp, glacier_gcm_prec, 
                             glacier_gcm_elev, glacier_gcm_lrgcm, glacier_gcm_lrglac, dates_table, 
                             t1_idx, t2_idx, t1, t2, observed_massbal, mb_obs_min, mb_obs_max)
                     )
    
#            print('\nParameters:\nPF_low:', np.round(precfactor_boundlow,2), 'PF_high:', 
#                  np.round(precfactor_boundhigh,2), '\nTC_low:', np.round(tempchange_boundlow,2), 
#                  'TC_high:', np.round(tempchange_boundhigh,2),
#                  '\nTC_mu:', np.round(tempchange_mu,2), 'TC_sigma:', np.round(tempchange_sigma,2))
            
        # Add priors
        prior_cns = ['pf_bndlow', 'pf_bndhigh', 'pf_mu', 'tc_bndlow', 'tc_bndhigh', 'tc_mu', 'tc_std', 
                     'ddfsnow_bndlow', 'ddfsnow_bndhigh', 'ddfsnow_mu', 'ddfsnow_std', 'mb_max_loss', 'mb_max_acc', 
                     'tc_max_loss', 'tc_max_acc','pf_opt_init', 'tc_opt_init']
        prior_values = [precfactor_boundlow, precfactor_boundhigh, precfactor_mu, tempchange_boundlow, 
                        tempchange_boundhigh, tempchange_mu, tempchange_sigma, ddfsnow_boundlow, ddfsnow_boundhigh, 
                        ddfsnow_mu, ddfsnow_sigma, mb_max_loss, mb_max_acc, tempchange_max_loss, tempchange_max_acc,
                        precfactor_opt_init, tempchange_opt_init]
        
        
        # Export dataset with priors
        ds = xr.open_dataset(input.output_fp_cal + glacier_str + '.nc')
        ds_priors = xr.Dataset({'priors': (('prior_cns'), prior_values)},
                               coords={'prior_cns': prior_cns})
        ds_wpriors = xr.merge([ds, ds_priors])
        
        fp_wpriors = input.output_fp_cal + 'wpriors/'
        if not os.path.exists(fp_wpriors):
            os.makedirs(fp_wpriors)
        ds_wpriors.to_netcdf(fp_wpriors + glacier_str + '.nc')
        ds.close()
        ds_priors.close()
        ds_wpriors.close()
        
        
        
    # Export variables as global to view in variable explorer
    if args.option_parallels == 0:
        global main_vars
        main_vars = inspect.currentframe().f_locals
        
    
#%% PARALLEL PROCESSING
if __name__ == '__main__':
    parser = getparser()
    args = parser.parse_args()

    # Reference GCM name
    gcm_name = args.ref_gcm_name
    print('Reference climate data is:', gcm_name)
        
    # RGI region
    if args.spc_region is not None:
        rgi_regionsO1 = [int(args.spc_region)]
    else:
        rgi_regionsO1 = input.rgi_regionsO1
    
    # RGI glacier number
    if args.rgi_glac_number_fn is not None:
        with open(args.rgi_glac_number_fn, 'rb') as f:
            rgi_glac_number = pickle.load(f)
    else:
        rgi_glac_number = input.rgi_glac_number    

    # Select all glaciers in a region
    main_glac_rgi_all = modelsetup.selectglaciersrgitable(rgi_regionsO1=rgi_regionsO1, rgi_regionsO2='all',
                                                          rgi_glac_number=rgi_glac_number)
    # Define chunk size for parallel processing
    if args.option_parallels != 0:
        num_cores = int(np.min([main_glac_rgi_all.shape[0], args.num_simultaneous_processes]))
        chunk_size = int(np.ceil(main_glac_rgi_all.shape[0] / num_cores))
    else:
        # if not running in parallel, chunk size is all glaciers
        chunk_size = main_glac_rgi_all.shape[0]
        
        
    # ===== LOAD GLACIER DATA =====
    # Glacier hypsometry [km**2], total area
    main_glac_hyps_all = modelsetup.import_Husstable(main_glac_rgi_all, rgi_regionsO1, input.hyps_filepath,
                                                     input.hyps_filedict, input.hyps_colsdrop)

    # Ice thickness [m], average
    main_glac_icethickness_all = modelsetup.import_Husstable(main_glac_rgi_all, rgi_regionsO1, 
                                                             input.thickness_filepath, input.thickness_filedict, 
                                                             input.thickness_colsdrop)
    main_glac_hyps_all[main_glac_icethickness_all == 0] = 0
    # Width [km], average
    main_glac_width_all = modelsetup.import_Husstable(main_glac_rgi_all, rgi_regionsO1, input.width_filepath,
                                                      input.width_filedict, input.width_colsdrop)
    elev_bins = main_glac_hyps_all.columns.values.astype(int)
    # Add volume [km**3] and mean elevation [m a.s.l.]
    main_glac_rgi_all['Volume'], main_glac_rgi_all['Zmean'] = (
            modelsetup.hypsometrystats(main_glac_hyps_all, main_glac_icethickness_all))
    # Select dates including future projections
    #  - nospinup dates_table needed to get the proper time indices
    dates_table_nospinup  = modelsetup.datesmodelrun(startyear=input.startyear, endyear=input.endyear, 
                                                     spinupyears=0)
    dates_table = modelsetup.datesmodelrun(startyear=input.startyear, endyear=input.endyear, 
                                           spinupyears=input.spinupyears)

    # ===== LOAD CALIBRATION DATA =====
    cal_data = pd.DataFrame()
    for dataset in input.cal_datasets:
        cal_subset = class_mbdata.MBData(name=dataset, rgi_regionO1=rgi_regionsO1[0])
        cal_subset_data = cal_subset.retrieve_mb(main_glac_rgi_all, main_glac_hyps_all, dates_table_nospinup)
        cal_data = cal_data.append(cal_subset_data, ignore_index=True)
    cal_data = cal_data.sort_values(['glacno', 't1_idx'])
    cal_data.reset_index(drop=True, inplace=True)
    
    # If group data is included, then add group dictionary and add group name to main_glac_rgi
    if set(['group']).issubset(input.cal_datasets) == True:
        # Group dictionary
        group_dict_raw = pd.read_csv(input.mb_group_fp + input.mb_group_dict_fn)
        # Remove groups that have no data
        group_names_wdata = np.unique(cal_data[np.isnan(cal_data.glacno)].group_name.values).tolist()
        group_dict_raw_wdata = group_dict_raw.loc[group_dict_raw.group_name.isin(group_names_wdata)]
        # Create dictionary to map group names to main_glac_rgi
        group_dict = dict(zip(group_dict_raw_wdata['RGIId'], group_dict_raw_wdata['group_name']))
        group_names_unique = list(set(group_dict.values()))
        group_dict_keyslist = [[] for x in group_names_unique]
        for n, group in enumerate(group_names_unique):
            group_dict_keyslist[n] = [group, [k for k, v in group_dict.items() if v == group]]
        # Add group name to main_glac_rgi
        main_glac_rgi_all['group_name'] = main_glac_rgi_all['RGIId'].map(group_dict)
    else:
        main_glac_rgi_all['group_name'] = np.nan

    # Drop glaciers that do not have any calibration data (individual or group)    
    main_glac_rgi = ((main_glac_rgi_all.iloc[np.unique(
            np.append(main_glac_rgi_all[main_glac_rgi_all['group_name'].notnull() == True].index.values, 
                      np.where(main_glac_rgi_all[input.rgi_O1Id_colname].isin(cal_data['glacno']) == True)[0])), :])
            .copy())
    # select glacier data
    main_glac_hyps = main_glac_hyps_all.iloc[main_glac_rgi.index.values]
    main_glac_icethickness = main_glac_icethickness_all.iloc[main_glac_rgi.index.values]
    main_glac_width = main_glac_width_all.iloc[main_glac_rgi.index.values]
    # Reset index
    main_glac_rgi.reset_index(drop=True, inplace=True)
    main_glac_hyps.reset_index(drop=True, inplace=True)
    main_glac_icethickness.reset_index(drop=True, inplace=True)
    main_glac_width.reset_index(drop=True, inplace=True)

    # ===== LOAD CLIMATE DATA =====
    gcm = class_climate.GCM(name=gcm_name)
    # Air temperature [degC], Precipitation [m], Elevation [masl], Lapse rate [K m-1]
    gcm_temp, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(gcm.temp_fn, gcm.temp_vn, main_glac_rgi, dates_table)
    gcm_prec, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(gcm.prec_fn, gcm.prec_vn, main_glac_rgi, dates_table)
    gcm_elev = gcm.importGCMfxnearestneighbor_xarray(gcm.elev_fn, gcm.elev_vn, main_glac_rgi)
    # Lapse rate [K m-1]
    if gcm_name == 'ERA-Interim':
        gcm_lr, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(gcm.lr_fn, gcm.lr_vn, main_glac_rgi, dates_table)
    else:
        # Mean monthly lapse rate
        ref_lr_monthly_avg = np.genfromtxt(gcm.lr_fp + gcm.lr_fn, delimiter=',')
        gcm_lr = np.tile(ref_lr_monthly_avg, int(gcm_temp.shape[1]/12))
    # COAWST data has two domains, so need to merge the two domains
    if gcm_name == 'COAWST':
        gcm_temp_d01, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(gcm.temp_fn_d01, gcm.temp_vn, main_glac_rgi, 
                                                                         dates_table)
        gcm_prec_d01, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(gcm.prec_fn_d01, gcm.prec_vn, main_glac_rgi, 
                                                                         dates_table)
        gcm_elev_d01 = gcm.importGCMfxnearestneighbor_xarray(gcm.elev_fn_d01, gcm.elev_vn, main_glac_rgi)
        # Check if glacier outside of high-res (d02) domain
        for glac in range(main_glac_rgi.shape[0]):
            glac_lat = main_glac_rgi.loc[glac,input.rgi_lat_colname]
            glac_lon = main_glac_rgi.loc[glac,input.rgi_lon_colname]
            if (~(input.coawst_d02_lat_min <= glac_lat <= input.coawst_d02_lat_max) or 
                ~(input.coawst_d02_lon_min <= glac_lon <= input.coawst_d02_lon_max)):
                gcm_prec[glac,:] = gcm_prec_d01[glac,:]
                gcm_temp[glac,:] = gcm_temp_d01[glac,:]
                gcm_elev[glac] = gcm_elev_d01[glac]
                
                
    # Pack variables for multiprocessing
    list_packed_vars = []
    n = 0
    for chunk in range(0, main_glac_rgi.shape[0], chunk_size):
        n += 1
        main_glac_rgi_chunk = main_glac_rgi.loc[chunk:chunk+chunk_size-1].copy()
        main_glac_hyps_chunk = main_glac_hyps.loc[chunk:chunk+chunk_size-1].copy()
        main_glac_icethickness_chunk = main_glac_icethickness.loc[chunk:chunk+chunk_size-1].copy()
        main_glac_width_chunk = main_glac_width.loc[chunk:chunk+chunk_size-1].copy()
        gcm_temp_chunk = gcm_temp[chunk:chunk+chunk_size]
        gcm_prec_chunk = gcm_prec[chunk:chunk+chunk_size]
        gcm_elev_chunk = gcm_elev[chunk:chunk+chunk_size]
        gcm_lr_chunk = gcm_lr[chunk:chunk+chunk_size]
        cal_data_chunk = cal_data.loc[chunk:chunk+chunk_size-1]
        
        list_packed_vars.append([n,
                                 gcm_name, 
                                 main_glac_rgi_chunk, 
                                 main_glac_hyps_chunk, 
                                 main_glac_icethickness_chunk, 
                                 main_glac_width_chunk,
                                 gcm_temp_chunk,
                                 gcm_prec_chunk,
                                 gcm_elev_chunk,
                                 gcm_lr_chunk,
                                 cal_data_chunk
                                 ])
    
    #%%===================================================
    # Parallel processing
    if args.option_parallels != 0:
        print('Processing in parallel with ' + str(num_cores) + ' cores...')
        with multiprocessing.Pool(args.num_simultaneous_processes) as p:
            p.map(main,list_packed_vars)
    # If not in parallel, then only should be one loop
    else:
        for n in range(len(list_packed_vars)):
            main(list_packed_vars[n])
            
    # Place local variables in variable explorer
    if (args.option_parallels == 0):
        main_vars_list = list(main_vars.keys())
        main_glac_rgi = main_vars['main_glac_rgi']
        precfactor_boundlow = main_vars['precfactor_boundlow']
        precfactor_boundhigh = main_vars['precfactor_boundhigh']
        precfactor_mu = main_vars['precfactor_mu']
        tempchange_boundlow = main_vars['tempchange_boundlow']
        tempchange_boundhigh = main_vars['tempchange_boundhigh']
        tempchange_mu = main_vars['tempchange_mu']
        tempchange_sigma = main_vars['tempchange_sigma']
        ddfsnow_boundlow = main_vars['ddfsnow_boundlow']
        ddfsnow_boundhigh = main_vars['ddfsnow_boundhigh']
        ddfsnow_mu = main_vars['ddfsnow_mu']
        ddfsnow_sigma = main_vars['ddfsnow_sigma']
        mb_max_loss = main_vars['mb_max_loss']
        mb_max_acc = main_vars['mb_max_acc']
        tempchange_max_loss = main_vars['tempchange_max_loss']
        tempchange_max_acc = main_vars['tempchange_max_acc']
        precfactor_opt_init = main_vars['precfactor_opt_init']
        tempchange_opt_init = main_vars['tempchange_opt_init']
        
        