#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 11:41:03 2018

@author: davidrounce
"""

# Built-in Libraries
import os
import collections
# External Libraries
import numpy as np
import pandas as pd 
#import netCDF4 as nc
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import matplotlib.patches as mpatches
import scipy
import cartopy
#import geopandas
import xarray as xr
from osgeo import ogr
import pickle
# Local Libraries
import pygem_input as input
import pygemfxns_modelsetup as modelsetup
import pygemfxns_massbalance as massbalance
import class_mbdata
import class_climate
import run_simulation

netcdf_fp_cmip5 = '/Users/davidrounce/Documents/Dave_Rounce/HiMAT/Output/simulations/'
gcm_names = ['CanESM2', 'CCSM4', 'CNRM-CM5', 'GFDL-CM3', 'GFDL-ESM2M', 'GISS-E2-R', 'IPSL-CM5A-LR', 'IPSL-CM5A-MR', 
             'MIROC5', 'MRI-CGCM3', 'NorESM1-M']
rcps = ['rcp26', 'rcp45', 'rcp85']
regions = [13, 14, 15]
vn = 'massbaltotal_glac_monthly'
startyear = 1970
endyear= 2017

#%%
def extract_model_comparison(gcm_names, rcps, vn, main_glac_rgi_all, ds_mb):
    """Partition multimodel data by each group for all GCMs for a given variable
    
    Parameters
    ----------
    gcm_names : list
        list of GCM names
    rcps : list
        list of rcp names
    vn : str
        variable name
    main_glac_rgi_all : pd.DataFrame
        glacier table
    ds_mb : pd.DataFrame
        mass balance observations
        
    Output
    ------
    ds_mb_sim : np.array
        array of each glaciers mass balance for every gcm/rcp combination (glac, gcm, rcp) compared to the MB 
        observations
    """
    ds_mb_sim = np.zeros((len(ds_mb), len(gcm_names), len(rcps)))
    ds_mb_sim_std = np.zeros((len(ds_mb), len(gcm_names), len(rcps)))
    ds_mb_sim_zscore = np.zeros((len(ds_mb), len(gcm_names), len(rcps)))
    for ngcm, gcm_name in enumerate(gcm_names):
        for nrcp, rcp in enumerate(rcps):                  
            # Load datasets
            netcdf_fp = netcdf_fp_cmip5 + '/' + gcm_name + '/'
            ds_fn = ('R' + str(region) + '_' + gcm_name + '_' + rcp + '_c2_ba2_100sets_1970_2017.nc')    
            
            # Bypass GCMs that are missing a rcp scenario
            try:
                ds = xr.open_dataset(netcdf_fp + ds_fn)
            except:
                continue
#            # Extract time variable
#            if 'annual' in vn:
#                try:
#                    time_values = ds[vn].coords['year_plus1'].values
#                except:
#                    time_values = ds[vn].coords['year'].values
#            else:
#                time_values = ds[vn].coords['time'].values

            for glac in range(len(main_glac_rgi_all)):
                # Model and uncertainty
                t1_idx = int(ds_mb.loc[glac, 't1_idx'])
                t2_idx = int(ds_mb.loc[glac, 't2_idx'])
                vn_glac = ds[vn].values[glac,t1_idx:t2_idx+1,0]
                vn_glac_mwea = vn_glac.mean()*12
                vn_glac_std = ds[vn].values[glac,t1_idx:t2_idx+1,1]
                vn_glac_var = vn_glac_std**2
                vn_glac_std_mwea = (vn_glac_var.sum())**0.5 / (len(vn_glac)/12)
                # Observation
                vn_glac_obs_mwea = ds_mb.loc[glac,'mb_mwe'] / (len(vn_glac)/12)
                
                # Record data
                ds_mb_sim[glac, ngcm, nrcp] = vn_glac_mwea
                ds_mb_sim_std[glac, ngcm, nrcp] = vn_glac_std_mwea
                
                # Can we standardize how far off these are?
                #  z_score = (model - obs) / model_uncertainty
                ds_mb_sim_zscore[glac, ngcm, nrcp] = (vn_glac_mwea - vn_glac_obs_mwea) / vn_glac_std_mwea
                
                # Regional mean, standard deviation, and variance
                #  mean: E(X+Y) = E(X) + E(Y)
                #  var: Var(X+Y) = Var(X) + Var(Y) + 2*Cov(X,Y)
                #    assuming X and Y are indepdent, then Cov(X,Y)=0, so Var(X+Y) = Var(X) + Var(Y)
                #  std: std(X+Y) = (Var(X+Y))**0.5
                
#                print('\nRGIId:', ds_mb.loc[glac, 'RGIId'], gcm_name, rcp)
#                print('Annual mb [mwea]:', np.round(vn_glac.mean()*12,2), '+/-', np.round(vn_glac_std_mwea,2))
#                print('Actual mb [mwea]:', np.round(ds_mb.loc[glac,'mb_mwe'] / (len(vn_glac)/12), 2), '+/-', 
#                      np.round(ds_mb.loc[glac,'mb_mwe_err'] / (len(vn_glac)/12), 2))
            
#            # Merge datasets
#            if region == rgi_regions[0]:
#                vn_glac_all = ds[vn].values[:,:,0]
#                vn_glac_std_all = ds[vn].values[:,:,1]
#            else:
#                vn_glac_all = np.concatenate((vn_glac_all, ds[vn].values[:,:,0]), axis=0)
#                vn_glac_std_all = np.concatenate((vn_glac_std_all, ds[vn].values[:,:,1]), axis=0)
#            
#        if ngcm == 0:
#            ds_glac = vn_glac_all[np.newaxis,:,:]
#            ds_glac_std = vn_glac_std_all[np.newaxis,:,:]
#        else:
#            ds_glac = np.concatenate((ds_glac, vn_glac_all[np.newaxis,:,:]), axis=0)
#            ds_glac_std = np.concatenate((ds_glac_std, vn_glac_std_all[np.newaxis,:,:]), axis=0)
                
    return ds_mb_sim, ds_mb_sim_std, ds_mb_sim_zscore

#%% VALIDATION - COMPARE MODEL WITH MEASURED GEODETIC MASS BALANCE
for nregion, region in enumerate(regions):
    mauer_pickle_fn = 'R' + str(region) + '_mauer_1970s_2000_rgi_glac_number.pkl'
    with open(mauer_pickle_fn, 'rb') as f:
        rgi_glac_number = pickle.load(f)

     # Select glaciers
    main_glac_rgi = modelsetup.selectglaciersrgitable(rgi_regionsO1=[region], rgi_regionsO2 = 'all', 
                                                      rgi_glac_number=rgi_glac_number)
    # Glacier hypsometry [km**2], total area
    main_glac_hyps = modelsetup.import_Husstable(main_glac_rgi, [region], input.hyps_filepath, 
                                                 input.hyps_filedict, input.hyps_colsdrop)
    # Determine dates_table_idx that coincides with data
    dates_table = modelsetup.datesmodelrun(startyear, endyear, spinupyears=0)

    
    # Select mass balance data
    mb1 = class_mbdata.MBData(name='mauer', rgi_regionO1=region)
    ds_mb = mb1.retrieve_mb(main_glac_rgi, main_glac_hyps, dates_table)
    
    # Select comparisons
    ds_sim_mb, ds_sim_mb_std, ds_sim_mb_zscore = extract_model_comparison(gcm_names, rcps, vn, main_glac_rgi, ds_mb)
    
    # Merge processed data
    if nregion == 0:
        main_glac_rgi_all = main_glac_rgi
        ds_mb_all = ds_mb
        ds_sim_mb_all = ds_sim_mb
        ds_sim_mb_std_all = ds_sim_mb_std
        ds_sim_mb_zscore_all = ds_sim_mb_zscore
    else:
        main_glac_rgi_all = pd.concat([main_glac_rgi_all, main_glac_rgi])
        ds_mb_all = pd.concat([ds_mb_all, ds_mb])
        ds_sim_mb_all = np.concatenate((ds_sim_mb_all, ds_sim_mb), axis=0)
        ds_sim_mb_std_all = np.concatenate((ds_sim_mb_std_all, ds_sim_mb_std), axis=0)
        ds_sim_mb_zscore_all = np.concatenate((ds_sim_mb_zscore_all, ds_sim_mb_zscore), axis=0)
        
#%%
# Compare GCMs
# Average zscores for all the glaciers
zscore_mean_glac = ds_sim_mb_zscore_all.mean(axis=0)
zscore_mean_glac_rcp = ds_sim_mb_zscore_all.mean(axis=0).mean(axis=1)
zscore_df = pd.DataFrame(zscore_mean_glac, index=gcm_names, columns=rcps)
gcm_loss_ranking = {'CanESM2': 6,
                    'CCSM4': 8,
                    'CNRM-CM5': 10,
                    'GFDL-CM3': 1,
                    'GFDL-ESM2M': 11,
                    'GISS-E2-R': 9,
                    'IPSL-CM5A-LR': 3,
                    'IPSL-CM5A-MR': 2,
                    'MIROC5': 4,
                    'MRI-CGCM3': 7,
                    'NorESM1-M': 5}
zscore_df['ranking'] = np.array([gcm_loss_ranking[x] for x in zscore_df.index.values.tolist()])
zscore_df = zscore_df.sort_values('ranking')


            
