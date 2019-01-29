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

netcdf_fp_cmip5 = '/Users/davidrounce/Documents/Dave_Rounce/HiMAT/Output/simulations/validation/'
figure_fp = '/Users/davidrounce/Documents/Dave_Rounce/HiMAT/Output/figures/cmip5/'
gcm_names = ['CanESM2', 'CCSM4', 'CSIRO-Mk3-6-0', 'CNRM-CM5', 'GFDL-CM3', 'GFDL-ESM2M', 'GISS-E2-R', 'IPSL-CM5A-LR', 
             'IPSL-CM5A-MR', 'MIROC5', 'MRI-CGCM3', 'NorESM1-M']
rcps = ['rcp26', 'rcp45', 'rcp85']
regions = [13, 14, 15]
vn = 'massbaltotal_glac_monthly'
startyear = 1970
endyear= 2017

# Colors list
#colors_rgb = [(0.00, 0.57, 0.57), (0.71, 0.43, 1.00), (0.86, 0.82, 0.00), (0.00, 0.29, 0.29), (0.00, 0.43, 0.86), 
#              (0.57, 0.29, 0.00), (1.00, 0.43, 0.71), (0.43, 0.71, 1.00), (0.14, 1.00, 0.14), (1.00, 0.71, 0.47), 
#              (0.29, 0.00, 0.57), (0.57, 0.00, 0.00), (0.71, 0.47, 1.00), (1.00, 1.00, 0.47)]
colors_rgb = ['firebrick', 'greenyellow', 'cyan', 'b', 'midnightblue', 'red', 'steelblue','bisque', 'yellow',
              'lightblue', 'lightslategrey', 'k']

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

ds_mb_all['mb_mwea'] = (ds_mb_all.mb_mwe / (ds_mb_all.t2 - ds_mb_all.t1)).values
ds_mb_all['mb_mwea_std'] = (ds_mb_all.mb_mwe_err / (ds_mb_all.t2 - ds_mb_all.t1)).values
        
#%% COMPARE ZSCORES IN PANDAS DATAFRAME
# Compare GCMs
# Average zscores for all the glaciers
zscore_mean_glac = ds_sim_mb_zscore_all.mean(axis=0)
zscore_mean_glac_rcp = ds_sim_mb_zscore_all.mean(axis=0).mean(axis=1)
zscore_df = pd.DataFrame(zscore_mean_glac, index=gcm_names, columns=rcps)
# GCM loss ranking based on All group for rcp85
gcm_loss_ranking = {'CanESM2': 7,
                    'CCSM4': 9,
                    'CSIRO-Mk3-6-0': 3,
                    'CNRM-CM5': 11,
                    'GFDL-CM3': 1,
                    'GFDL-ESM2M': 12,
                    'GISS-E2-R': 10,
                    'IPSL-CM5A-LR': 4,
                    'IPSL-CM5A-MR': 2,
                    'MIROC5': 5,
                    'MRI-CGCM3': 8,
                    'NorESM1-M': 6}
zscore_df['rcp_avg'] = zscore_df[rcps].mean(axis=1)
zscore_df['rcp_std'] = zscore_df[rcps].std(axis=1)
zscore_df['abs_rcp_avg'] = zscore_df.rcp_avg.abs()
zscore_ranking = dict(zip(zscore_df.sort_values('abs_rcp_avg').index.values.tolist(), np.arange(0,len(zscore_df))+1))
zscore_df['zscore_ranking'] = np.array([zscore_ranking[x] for x in zscore_df.index.values.tolist()])
zscore_df['loss_ranking'] = np.array([gcm_loss_ranking[x] for x in zscore_df.index.values.tolist()])
zscore_df = zscore_df.sort_values('zscore_ranking')

#%% CALCULATE THE SUM OF SQUARES
df_ss = pd.DataFrame(np.zeros((len(gcm_names), len(rcps))), index=gcm_names, columns=rcps)
obs_mwea = ds_mb_all.mb_mwea.values
for ngcm, gcm in enumerate(gcm_names):
    for nrcp, rcp in enumerate(rcps):
        df_ss.loc[gcm,rcp] = ((ds_sim_mb_all[:,ngcm,nrcp] - ds_mb_all.mb_mwea.values)**2).sum()
df_ss['rcp_avg'] = df_ss[rcps].mean(axis=1)
df_ss['rcp_std'] = df_ss[rcps].std(axis=1)
ss_ranking = dict(zip(df_ss.sort_values('rcp_avg').index.values.tolist(), np.arange(0,len(zscore_df))+1))
df_ss['ss_ranking'] = np.array([ss_ranking[x] for x in df_ss.index.values.tolist()])
df_ss = df_ss.sort_values('ss_ranking')
df_ss['loss_ranking'] = np.array([gcm_loss_ranking[x] for x in df_ss.index.values.tolist()])

#%% PLOT SIMULATION VS OBSERVATIONS (including uncertainty)
fig, ax = plt.subplots(1, 1, squeeze=False, figsize=(5,4), gridspec_kw = {'wspace':0, 'hspace':0})

ngcms2plot = 4
#rankby = 'zscore'
rankby = 'sum_squares'

if rankby == 'zscore':
    gcm_names2plot = zscore_df.index.values.tolist()
elif rankby == 'sum_squares':
    gcm_names2plot = df_ss.index.values.tolist()
    
gcm_colordict = dict(zip(gcm_names2plot, colors_rgb[0:len(gcm_names2plot)]))
for ngcm, gcm in enumerate(gcm_names2plot):
#for ngcm, gcm in enumerate(good_gcms):
    for nrcp, rcp in enumerate(rcps):
        
        if nrcp == 1:
            print(rcp)
            
            if ngcm < ngcms2plot:
                ax[0,0].errorbar(ds_mb_all.mb_mwea.values, ds_sim_mb_all[:,ngcm,nrcp], 
                                 xerr=ds_mb_all.mb_mwea_std, yerr=ds_sim_mb_std_all[:,ngcm,nrcp],
                                 color=gcm_colordict[gcm], fmt='o', linewidth=0.5, markersize=3,
                                 label=gcm,
                                 zorder=len(gcm_names) - ngcm)
            else:
                ax[0,0].errorbar(ds_mb_all.mb_mwea.values, ds_sim_mb_all[:,ngcm,nrcp], 
                                 xerr=ds_mb_all.mb_mwea_std, yerr=ds_sim_mb_std_all[:,ngcm,nrcp],
                                 color=gcm_colordict[gcm], fmt='o', linewidth=0, markersize=0,
                                 label=gcm,
                                 zorder=len(gcm_names) - ngcm)
                    
# Add 1:1 line
ax[0,0].plot([-5,5],[-5,5], color='k', label='1:1 line')

# X-axis
ax[0,0].set_xlabel('Observed Mass Balance [mwea]', size=14)
ax[0,0].set_xlim(-1,0.5)
ax[0,0].xaxis.set_tick_params(labelsize=14)
ax[0,0].xaxis.set_major_locator(plt.MultipleLocator(0.5))
ax[0,0].xaxis.set_minor_locator(plt.MultipleLocator(0.1))

# Y-axis
ax[0,0].set_ylabel('Modeled Mass Balance [mwea]', size=14)
ax[0,0].set_ylim(-1.5,1.0)
ax[0,0].yaxis.set_tick_params(labelsize=14)
ax[0,0].yaxis.set_major_locator(plt.MultipleLocator(0.5))
ax[0,0].yaxis.set_minor_locator(plt.MultipleLocator(0.1))

# Legend
ax[0,0].legend(loc='center left', bbox_to_anchor=(1,0.5), fontsize=12, labelspacing=0.25, handletextpad=0.5, borderpad=0, 
               frameon=False, handlelength=1)
plt.show()

