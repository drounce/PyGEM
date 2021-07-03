#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 10:17:33 2021

@author: drounce
"""


# Built-in libraries
import calendar
#from collections import OrderedDict
import datetime
#import glob
import os
#import pickle
# External libraries
#import cartopy
#import matplotlib as mpl
import matplotlib.pyplot as plt
#from matplotlib.pyplot import MaxNLocator
#from matplotlib.lines import Line2D
#import matplotlib.patches as mpatches
#from matplotlib.ticker import MultipleLocator
#from matplotlib.ticker import EngFormatter
#from matplotlib.ticker import StrMethodFormatter
#from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import numpy as np
import pandas as pd
from scipy.stats import linregress
#from scipy.ndimage import uniform_filter
#import scipy
import xarray as xr
# Local libraries
#import class_climate
#import class_mbdata
import pygem.pygem_input as pygem_prms
import pygem.pygem_modelsetup as modelsetup

wgms_annual_comparison = False
wgms_winter_comparison = False
wgms_summer_comparison = True


netcdf_fp_era5 = '/Users/drounce/Documents/HiMAT/spc_backup/simulations/'
netcdf_fn_ending_stats = '_ERA5_MCMC_ba1_50sets_1979_2019_all.nc'
#netcdf_fn_ending_binned = '_ERA5_MCMC_ba1_50sets_1979_2019_all.nc'

startyear = 1979
endyear = 2019

dates_table = modelsetup.datesmodelrun(startyear=startyear, endyear=endyear,
                                       option_wateryear='calendar')

wgms_fp = pygem_prms.main_directory +  '/../WGMS/DOI-WGMS-FoG-2020-08/'
wgms_eee_fn = 'WGMS-FoG-2020-08-EEE-MASS-BALANCE-POINT.csv'
wgms_ee_fn = 'WGMS-FoG-2020-08-EE-MASS-BALANCE.csv'
wgms_e_fn = 'WGMS-FoG-2020-08-E-MASS-BALANCE-OVERVIEW.csv'
wgms_id_fn = 'WGMS-FoG-2020-08-AA-GLACIER_ID_LUT.csv'

# Load data 
wgms_e_df = pd.read_csv(wgms_fp + wgms_e_fn, encoding='unicode_escape')
wgms_ee_df_raw = pd.read_csv(wgms_fp + wgms_ee_fn, encoding='unicode_escape')
wgms_eee_df_raw = pd.read_csv(wgms_fp + wgms_eee_fn, encoding='unicode_escape')
wgms_id_df = pd.read_csv(wgms_fp + wgms_id_fn, encoding='unicode_escape')
    
# Map dictionary
wgms_id_dict = dict(zip(wgms_id_df.WGMS_ID, wgms_id_df.RGI_ID))
wgms_e_df['RGIId_raw'] = wgms_e_df.WGMS_ID.map(wgms_id_dict)
wgms_e_df = wgms_e_df.dropna(subset=['RGIId_raw'])
wgms_ee_df_raw['RGIId_raw'] = wgms_ee_df_raw.WGMS_ID.map(wgms_id_dict)
wgms_ee_df_raw = wgms_ee_df_raw.dropna(subset=['RGIId_raw'])
wgms_eee_df_raw['RGIId_raw'] = wgms_eee_df_raw.WGMS_ID.map(wgms_id_dict)
wgms_eee_df_raw = wgms_eee_df_raw.dropna(subset=['RGIId_raw'])

# Link RGIv5.0 with RGIv6.0
rgi60_fp = pygem_prms.main_directory +  '/../RGI/rgi60/00_rgi60_attribs/'
rgi50_fp = pygem_prms.main_directory +  '/../RGI/00_rgi50_attribs/'

# Process each region
regions_str = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18']
rgi60_df = None
rgi50_df = None
for reg_str in regions_str:
    # RGI60 data
    for i in os.listdir(rgi60_fp):
        if i.startswith(reg_str) and i.endswith('.csv'):
            rgi60_df_reg = pd.read_csv(rgi60_fp + i, encoding='unicode_escape')
    # append datasets
    if rgi60_df is None:
        rgi60_df = rgi60_df_reg
    else:
        rgi60_df = pd.concat([rgi60_df, rgi60_df_reg], axis=0)

    # RGI50 data
    for i in os.listdir(rgi50_fp):
        if i.startswith(reg_str) and i.endswith('.csv'):
            rgi50_df_reg = pd.read_csv(rgi50_fp + i, encoding='unicode_escape')
    # append datasets
    if rgi50_df is None:
        rgi50_df = rgi50_df_reg
    else:
        rgi50_df = pd.concat([rgi50_df, rgi50_df_reg], axis=0)
    
# Merge based on GLIMSID
glims_rgi50_dict = dict(zip(rgi50_df.GLIMSId, rgi50_df.RGIId))
rgi60_df['RGIId_50'] = rgi60_df.GLIMSId.map(glims_rgi50_dict)
rgi60_df_4dict = rgi60_df.dropna(subset=['RGIId_50'])
rgi50_rgi60_dict = dict(zip(rgi60_df_4dict.RGIId_50, rgi60_df_4dict.RGIId))
rgi60_self_dict = dict(zip(rgi60_df.RGIId, rgi60_df.RGIId))
rgi50_rgi60_dict.update(rgi60_self_dict)

# Add RGIId for version 6 to WGMS
wgms_e_df['RGIId'] = wgms_e_df.RGIId_raw.map(rgi50_rgi60_dict)
wgms_ee_df_raw['RGIId'] = wgms_ee_df_raw.RGIId_raw.map(rgi50_rgi60_dict)
wgms_eee_df_raw['RGIId'] = wgms_eee_df_raw.RGIId_raw.map(rgi50_rgi60_dict)

# Drop points without data
wgms_ee_df = wgms_ee_df_raw.dropna(subset=['RGIId'])
wgms_eee_df = wgms_eee_df_raw.dropna(subset=['RGIId'])

# Select only those with entire glacier
wgms_ee_df = wgms_ee_df.loc[wgms_ee_df['LOWER_BOUND']==9999,:]
# Year limits
wgms_ee_df = wgms_ee_df.loc[wgms_ee_df['YEAR']>=startyear,:]
wgms_ee_df = wgms_ee_df.loc[wgms_ee_df['YEAR']<=endyear,:]

wgms_ee_df['O1Region'] = [int(x.split('-')[1].split('.')[0]) for x in wgms_ee_df.RGIId.values]

#%% ===== ANNUAL BALANCES ONLY =====
if wgms_annual_comparison:
    wgms_ee_df_annual = wgms_ee_df.dropna(subset=['ANNUAL_BALANCE'])
    wgms_ee_df_annual = wgms_ee_df_annual.sort_values('RGIId')
    wgms_ee_df_annual.reset_index(inplace=True, drop=True)
    wgms_ee_df_annual['period'] = 'annual'
    
    wgms_glacno_list = [x.split('-')[1] for x in sorted(list(np.unique(wgms_ee_df_annual['RGIId'])))]
    main_glac_rgi = modelsetup.selectglaciersrgitable(glac_no=wgms_glacno_list)
    
    # Add survey dates
    wgms_ee_df_annual['TIME_SYSTEM'] = np.nan
    wgms_ee_df_annual['BEGIN_PERIOD'] = np.nan 
    wgms_ee_df_annual['END_WINTER'] = np.nan
    wgms_ee_df_annual['END_PERIOD'] = np.nan 
    for x in range(wgms_ee_df_annual.shape[0]):
        wgms_ee_df_annual.loc[x,'TIME_SYSTEM'] = (
                wgms_e_df[(wgms_ee_df_annual.loc[x,'WGMS_ID'] == wgms_e_df['WGMS_ID']) & 
                          (wgms_ee_df_annual.loc[x,'YEAR'] == wgms_e_df['Year'])]['TIME_SYSTEM'].values[0]) 
        wgms_ee_df_annual.loc[x,'BEGIN_PERIOD'] = (
                wgms_e_df[(wgms_ee_df_annual.loc[x,'WGMS_ID'] == wgms_e_df['WGMS_ID']) & 
                          (wgms_ee_df_annual.loc[x,'YEAR'] == wgms_e_df['Year'])]['BEGIN_PERIOD'].values)
        wgms_ee_df_annual.loc[x,'END_WINTER'] = (
                wgms_e_df[(wgms_ee_df_annual.loc[x,'WGMS_ID'] == wgms_e_df['WGMS_ID']) & 
                          (wgms_ee_df_annual.loc[x,'YEAR'] == wgms_e_df['Year'])]['END_WINTER'].values)
        wgms_ee_df_annual.loc[x,'END_PERIOD'] = (
                wgms_e_df[(wgms_ee_df_annual.loc[x,'WGMS_ID'] == wgms_e_df['WGMS_ID']) & 
                          (wgms_ee_df_annual.loc[x,'YEAR'] == wgms_e_df['Year'])]['END_PERIOD'].values)
        
    # Add survey dates
    wgms_ee_df_annual['TIME_SYSTEM'] = np.nan
    wgms_ee_df_annual['BEGIN_PERIOD'] = np.nan 
    wgms_ee_df_annual['END_PERIOD'] = np.nan
    for x in range(wgms_ee_df_annual.shape[0]):
        wgms_ee_df_annual.loc[x,'TIME_SYSTEM'] = (
                wgms_e_df[(wgms_ee_df_annual.loc[x,'WGMS_ID'] == wgms_e_df['WGMS_ID']) & 
                          (wgms_ee_df_annual.loc[x,'YEAR'] == wgms_e_df['Year'])]['TIME_SYSTEM'].values[0]) 
        wgms_ee_df_annual.loc[x,'BEGIN_PERIOD'] = (
                wgms_e_df[(wgms_ee_df_annual.loc[x,'WGMS_ID'] == wgms_e_df['WGMS_ID']) & 
                          (wgms_ee_df_annual.loc[x,'YEAR'] == wgms_e_df['Year'])]['BEGIN_PERIOD'].values)
        wgms_ee_df_annual.loc[x,'END_PERIOD'] = (
                wgms_e_df[(wgms_ee_df_annual.loc[x,'WGMS_ID'] == wgms_e_df['WGMS_ID']) & 
                          (wgms_ee_df_annual.loc[x,'YEAR'] == wgms_e_df['Year'])]['END_PERIOD'].values)
    # Time indices
    #  winter and summer balances typically have the same data for 'BEGIN_PERIOD' and 'END_PERIOD' as the annual
    #  measurements, so need to set these dates manually
    # Remove glaciers without begin or end period
    wgms_ee_df_annual = wgms_ee_df_annual.drop(np.where(np.isnan(wgms_ee_df_annual['BEGIN_PERIOD'].values))[0].tolist(), axis=0)
    wgms_ee_df_annual.reset_index(drop=True, inplace=True)
    wgms_ee_df_annual = wgms_ee_df_annual.drop(np.where(np.isnan(wgms_ee_df_annual['END_PERIOD'].values))[0].tolist(), axis=0)
    wgms_ee_df_annual.reset_index(drop=True, inplace=True)
    wgms_ee_df_annual['t1_year'] = wgms_ee_df_annual['BEGIN_PERIOD'].astype(str).str.split('.').str[0].str[:4].astype(int)
    wgms_ee_df_annual['t1_month'] = wgms_ee_df_annual['BEGIN_PERIOD'].astype(str).str.split('.').str[0].str[4:6].astype(int)
    wgms_ee_df_annual['t1_day'] = wgms_ee_df_annual['BEGIN_PERIOD'].astype(str).str.split('.').str[0].str[6:].astype(int)
    wgms_ee_df_annual['t2_year'] = wgms_ee_df_annual['END_PERIOD'].astype(str).str.split('.').str[0].str[:4].astype(int)
    wgms_ee_df_annual['t2_month'] = wgms_ee_df_annual['END_PERIOD'].astype(str).str.split('.').str[0].str[4:6].astype(int)
    wgms_ee_df_annual['t2_day'] = wgms_ee_df_annual['END_PERIOD'].astype(str).str.split('.').str[0].str[6:].astype(int)      
    wgms_ee_df_annual.loc[wgms_ee_df_annual['t1_month'] > 13,'t1_month'] = np.nan
    wgms_ee_df_annual.loc[wgms_ee_df_annual['t1_month'] < 1, 't1_month'] = np.nan
    wgms_ee_df_annual.loc[wgms_ee_df_annual['t2_month'] > 13,'t2_month'] = np.nan
    wgms_ee_df_annual.loc[wgms_ee_df_annual['t2_month'] < 1, 't2_month'] = np.nan
    wgms_ee_df_annual.loc[wgms_ee_df_annual['t1_day'] < 1, 't1_day'] = np.nan
    wgms_ee_df_annual.loc[wgms_ee_df_annual['t2_day'] < 1, 't2_day'] = np.nan
    wgms_ee_df_annual.loc[wgms_ee_df_annual['t1_day'] > 31, 't1_day'] = np.nan
    wgms_ee_df_annual.loc[wgms_ee_df_annual['t2_day'] > 31,'t2_day'] = np.nan
    wgms_ee_df_annual.loc[wgms_ee_df_annual['t1_year'] < startyear,'t1_year'] = np.nan
    wgms_ee_df_annual.loc[wgms_ee_df_annual['t2_year'] > endyear,'t2_year'] = np.nan
    wgms_ee_df_annual = wgms_ee_df_annual.dropna(subset=['t1_year', 't2_year', 't1_month', 't2_month', 't1_day', 't2_day'])
    wgms_ee_df_annual.reset_index(inplace=True, drop=True)
    
    # Correct poor dates
    for nmonth in [4,6,9,11]:
        wgms_ee_df_annual.loc[(wgms_ee_df_annual['t1_month'] == nmonth) & (wgms_ee_df_annual['t1_day'] == 31), 't1_day'] = 30
        wgms_ee_df_annual.loc[(wgms_ee_df_annual['t2_month'] == nmonth) & (wgms_ee_df_annual['t2_day'] == 31), 't2_day'] = 30

    # Calculate decimal year and drop measurements outside of calibration period
    wgms_ee_df_annual['t1_datetime'] = pd.to_datetime(
            pd.DataFrame({'year':wgms_ee_df_annual.t1_year.values, 'month':wgms_ee_df_annual.t1_month.values, 'day':wgms_ee_df_annual.t1_day.values}))
    wgms_ee_df_annual['t2_datetime'] = pd.to_datetime(
            pd.DataFrame({'year':wgms_ee_df_annual.t2_year.values, 'month':wgms_ee_df_annual.t2_month.values, 'day':wgms_ee_df_annual.t2_day.values}))
    wgms_ee_df_annual['t1_doy'] = wgms_ee_df_annual.t1_datetime.dt.strftime("%j").astype(float)
    wgms_ee_df_annual['t2_doy'] = wgms_ee_df_annual.t2_datetime.dt.strftime("%j").astype(float)
    wgms_ee_df_annual['t1_daysinyear'] = (
            (pd.to_datetime(pd.DataFrame({'year':wgms_ee_df_annual.t1_year.values, 'month':12, 'day':31})) - 
             pd.to_datetime(pd.DataFrame({'year':wgms_ee_df_annual.t1_year.values, 'month':1, 'day':1}))).dt.days + 1)
    wgms_ee_df_annual['t2_daysinyear'] = (
            (pd.to_datetime(pd.DataFrame({'year':wgms_ee_df_annual.t2_year.values, 'month':12, 'day':31})) - 
             pd.to_datetime(pd.DataFrame({'year':wgms_ee_df_annual.t2_year.values, 'month':1, 'day':1}))).dt.days + 1)
    wgms_ee_df_annual['t1'] = wgms_ee_df_annual.t1_year + wgms_ee_df_annual.t1_doy / wgms_ee_df_annual.t1_daysinyear
    wgms_ee_df_annual['t2'] = wgms_ee_df_annual.t2_year + wgms_ee_df_annual.t2_doy / wgms_ee_df_annual.t2_daysinyear
    end_datestable = dates_table.loc[dates_table.shape[0]-1, 'date']
    end_datesyear = end_datestable.year
    end_datesmonth = end_datestable.month + 1
    end_datesday = end_datestable.day
    if end_datesmonth > 12:
        end_datesyear += 1
        end_datesmonth = 1
    end_datetime = datetime.datetime(end_datesyear, end_datesmonth, end_datesday)
    wgms_ee_df_annual = wgms_ee_df_annual[wgms_ee_df_annual['t1_datetime'] >= dates_table.loc[0, 'date']]
    wgms_ee_df_annual = wgms_ee_df_annual[wgms_ee_df_annual['t2_datetime'] < end_datetime]
    wgms_ee_df_annual.reset_index(drop=True, inplace=True)
    # Annual, summer, and winter time indices
    #  exclude spinup years, since massbal fxn discarwgms_ee_df_annual spinup years
    wgms_ee_df_annual['t1_idx'] = np.nan
    wgms_ee_df_annual['t2_idx'] = np.nan
    for x in range(wgms_ee_df_annual.shape[0]):
        wgms_ee_df_annual.loc[x,'t1_idx'] = (dates_table[(wgms_ee_df_annual.loc[x, 't1_year'] == dates_table['year']) & 
                                             (wgms_ee_df_annual.loc[x, 't1_month'] == dates_table['month'])].index.values[0])
        wgms_ee_df_annual.loc[x,'t2_idx'] = (dates_table[(wgms_ee_df_annual.loc[x, 't2_year'] == dates_table['year']) & 
                                             (wgms_ee_df_annual.loc[x, 't2_month'] == dates_table['month'])].index.values[0])
    
    # Remove values that are less than 8 months
    wgms_ee_df_annual['months'] = wgms_ee_df_annual['t2_idx'] - wgms_ee_df_annual['t1_idx']
    wgms_ee_df_annual = wgms_ee_df_annual.loc[wgms_ee_df_annual['months'] >= 8,:]
    wgms_ee_df_annual.reset_index(inplace=True, drop=True)
    
    # Process data
    wgms_ee_df_annual['mod_annual_mwe'] = np.nan
    wgms_ee_df_annual['mod_annual_mwe_mad_ind'] = np.nan
    wgms_ee_df_annual['mod_annual_mwe_mad_cor'] = np.nan
    wgms_ee_df_annual['annual_mwe'] = wgms_ee_df_annual['ANNUAL_BALANCE'].values / 1000
    for nrgiid, RGIId in enumerate(main_glac_rgi.RGIId):
        glac_str = str(int(RGIId.split('-')[1].split('.')[0])) + '.' + RGIId.split('-')[1].split('.')[1]
        wgms_ee_df_annual_glac = wgms_ee_df_annual.loc[wgms_ee_df_annual['RGIId'] == RGIId,:]
        
        glac_area = main_glac_rgi.loc[nrgiid,'Area'] * 1e6
        
        # Load dataset
        netcdf_fp_stats = netcdf_fp_era5 + glac_str.split('.')[0].zfill(2) + '/ERA5/stats/'
        netcdf_fn_stats = glac_str + '_ERA5_MCMC_ba1_50sets_1979_2019_all.nc'
        ds_stats = xr.open_dataset(netcdf_fp_stats + netcdf_fn_stats)
        
        mb_monthly = ds_stats.glac_massbaltotal_monthly.values[0,:]
        mb_monthly_mad = ds_stats.glac_massbaltotal_monthly_mad.values[0,:]
        for wgms_idx in wgms_ee_df_annual_glac.index.values:
            t1_idx = int(wgms_ee_df_annual_glac.loc[wgms_idx,'t1_idx'])
            t2_idx = int(wgms_ee_df_annual_glac.loc[wgms_idx,'t2_idx'])
            mb_monthly_subset = mb_monthly[t1_idx:t2_idx+1]
            mb_monthly_mad_subset = mb_monthly_mad[t1_idx:t2_idx+1]
            mb_mwea = mb_monthly_subset.sum() / glac_area
             # aggregate monthly data assuming monthly are independent and perfectly correlated
            mb_mwea_mad_ind = ((mb_monthly_mad_subset**2).sum())**0.5 / glac_area
            mb_mwea_mad_cor = mb_monthly_mad_subset.sum() / glac_area
            wgms_ee_df_annual.loc[wgms_idx,'mod_annual_mwe'] = mb_mwea
            wgms_ee_df_annual.loc[wgms_idx,'mod_annual_mwe_mad_ind'] = mb_mwea_mad_ind
            wgms_ee_df_annual.loc[wgms_idx,'mod_annual_mwe_mad_cor'] = mb_mwea_mad_cor

    # Difference statistics
    mb_dif_annual = wgms_ee_df_annual['mod_annual_mwe'].values - wgms_ee_df_annual['annual_mwe'].values
    print('  Difference stats: \n    Mean (+/-) std [mwe]:', 
          np.round(mb_dif_annual.mean(),2), '+/-', np.round(mb_dif_annual.std(),2), 
          'count:', len(mb_dif_annual),
          '\n    Median (+/-) std [mwe]:', 
          np.round(np.median(mb_dif_annual),2), '+/- XXX', 
    #          np.round(cal_data_subset['dif_mb_mwea'].std(),2),
    #      '\n    Mean standard deviation (correlated):',np.round(mb_dif_annual.mean(),2),
    #      '\n    Mean standard deviation (uncorrelated):',np.round(cal_data_subset['mb_mwea_era_std_rsos'].mean(),2)
          )
    
    # Validation Figure
    fig, ax = plt.subplots(1, 1, squeeze=False, sharex=False, sharey=True, 
                           gridspec_kw = {'wspace':0, 'hspace':0})
    
    ax[0,0].plot(wgms_ee_df_annual['annual_mwe'].values, 
                 wgms_ee_df_annual['mod_annual_mwe'].values,
                 linewidth=0, marker='o', mec='k', mew=1, mfc='none')
    
    ax[0,0].set_xlabel('B_obs (mwe)')
    ax[0,0].set_ylabel('B_mod (mwe)')
    ax[0,0].text(0.98, 1.06, 'Glaciological (annual)', size=10, horizontalalignment='right', 
                 verticalalignment='top', transform=ax[0,0].transAxes)
    ax[0,0].set_xlim(-6,5)
    ax[0,0].set_ylim(-6,5)
    ax[0,0].plot([-6,5],[-6,5], color='k',lw=0.5)
    #ax[0,0].legend(
    #        )        
    ax[0,0].tick_params(direction='inout', right=True)
    # Save figure
    fig_fn = 'validation_mb_mwea_annual_all.png'
    fig_fp = netcdf_fp_era5 + '../analysis/figures/validation/'
    if not os.path.exists(fig_fp):
        os.makedirs(fig_fp)
    fig.set_size_inches(3,3)
    fig.savefig(fig_fp + fig_fn, bbox_inches='tight', dpi=300)
    
    # ----- Regional statistics -----
    regions = np.unique(wgms_ee_df_annual.O1Region.values)
    for reg in regions:
        wgms_ee_df_annual_reg = wgms_ee_df_annual.loc[wgms_ee_df_annual['O1Region'] == reg, :]
        mb_dif_annual_reg = wgms_ee_df_annual_reg['mod_annual_mwe'].values - wgms_ee_df_annual_reg['annual_mwe'].values
        
        print('Region', reg, '  Difference stats: \n    Mean (+/-) std [mwe]:',  np.round(mb_dif_annual_reg.mean(),2), 
              '+/-', np.round(mb_dif_annual_reg.std(),2), 'count:', len(mb_dif_annual_reg),
              '\n    Median (+/-) std [mwe]:', np.round(np.median(mb_dif_annual_reg),2), '+/- XXX')
        fig, ax = plt.subplots(1, 1, squeeze=False, sharex=False, sharey=True, 
                               gridspec_kw = {'wspace':0, 'hspace':0})
        
        ax[0,0].plot(wgms_ee_df_annual_reg['annual_mwe'].values, 
                     wgms_ee_df_annual_reg['mod_annual_mwe'].values,
                     linewidth=0, marker='o', mec='k', mew=1, mfc='none')
        
        ax[0,0].set_xlabel('B_obs (mwe)')
        ax[0,0].set_ylabel('B_mod (mwe)')
        ax[0,0].text(0.98, 1.06, 'Glaciological (annual) - Region ' + str(reg), size=10, horizontalalignment='right', 
                     verticalalignment='top', transform=ax[0,0].transAxes)
        ax[0,0].set_xlim(-6,5)
        ax[0,0].set_ylim(-6,5)
        ax[0,0].plot([-6,5],[-6,5], color='k',lw=0.5)
        #ax[0,0].legend(
        #        )        
        ax[0,0].tick_params(direction='inout', right=True)
        # Save figure
        fig_fn = 'validation_mb_mwe_annual_' + str(reg).zfill(2) + '.png'
        fig_fp = netcdf_fp_era5 + '../analysis/figures/validation/'
        if not os.path.exists(fig_fp):
            os.makedirs(fig_fp)
        fig.set_size_inches(3,3)
        fig.savefig(fig_fp + fig_fn, bbox_inches='tight', dpi=300)
    
    
#%% ===== WINTER BALANCES ONLY =====
if wgms_winter_comparison:
    wgms_ee_df_winter = wgms_ee_df.dropna(subset=['WINTER_BALANCE'])
    wgms_ee_df_winter = wgms_ee_df_winter.sort_values('RGIId')
    wgms_ee_df_winter.reset_index(inplace=True, drop=True)
    wgms_ee_df_winter['period'] = 'winter'
    
    wgms_glacno_list_winter = [x.split('-')[1] for x in sorted(list(np.unique(wgms_ee_df_winter['RGIId'])))]
    main_glac_rgi = modelsetup.selectglaciersrgitable(glac_no=wgms_glacno_list_winter)
    
    # Add survey dates
    wgms_ee_df_winter['TIME_SYSTEM'] = np.nan
    wgms_ee_df_winter['BEGIN_PERIOD'] = np.nan 
    wgms_ee_df_winter['END_WINTER'] = np.nan
    wgms_ee_df_winter['END_PERIOD'] = np.nan 
    for x in range(wgms_ee_df_winter.shape[0]):
        wgms_ee_df_winter.loc[x,'TIME_SYSTEM'] = (
                wgms_e_df[(wgms_ee_df_winter.loc[x,'WGMS_ID'] == wgms_e_df['WGMS_ID']) & 
                          (wgms_ee_df_winter.loc[x,'YEAR'] == wgms_e_df['Year'])]['TIME_SYSTEM'].values[0]) 
        wgms_ee_df_winter.loc[x,'BEGIN_PERIOD'] = (
                wgms_e_df[(wgms_ee_df_winter.loc[x,'WGMS_ID'] == wgms_e_df['WGMS_ID']) & 
                          (wgms_ee_df_winter.loc[x,'YEAR'] == wgms_e_df['Year'])]['BEGIN_PERIOD'].values)
        wgms_ee_df_winter.loc[x,'END_WINTER'] = (
                wgms_e_df[(wgms_ee_df_winter.loc[x,'WGMS_ID'] == wgms_e_df['WGMS_ID']) & 
                          (wgms_ee_df_winter.loc[x,'YEAR'] == wgms_e_df['Year'])]['END_WINTER'].values)
        wgms_ee_df_winter.loc[x,'END_PERIOD'] = (
                wgms_e_df[(wgms_ee_df_winter.loc[x,'WGMS_ID'] == wgms_e_df['WGMS_ID']) & 
                          (wgms_ee_df_winter.loc[x,'YEAR'] == wgms_e_df['Year'])]['END_PERIOD'].values)
        
    # Add survey dates
    wgms_ee_df_winter['TIME_SYSTEM'] = np.nan
    wgms_ee_df_winter['BEGIN_PERIOD'] = np.nan 
    wgms_ee_df_winter['END_WINTER'] = np.nan
    for x in range(wgms_ee_df_winter.shape[0]):
        wgms_ee_df_winter.loc[x,'TIME_SYSTEM'] = (
                wgms_e_df[(wgms_ee_df_winter.loc[x,'WGMS_ID'] == wgms_e_df['WGMS_ID']) & 
                          (wgms_ee_df_winter.loc[x,'YEAR'] == wgms_e_df['Year'])]['TIME_SYSTEM'].values[0]) 
        wgms_ee_df_winter.loc[x,'BEGIN_PERIOD'] = (
                wgms_e_df[(wgms_ee_df_winter.loc[x,'WGMS_ID'] == wgms_e_df['WGMS_ID']) & 
                          (wgms_ee_df_winter.loc[x,'YEAR'] == wgms_e_df['Year'])]['BEGIN_PERIOD'].values)
        wgms_ee_df_winter.loc[x,'END_WINTER'] = (
                wgms_e_df[(wgms_ee_df_winter.loc[x,'WGMS_ID'] == wgms_e_df['WGMS_ID']) & 
                          (wgms_ee_df_winter.loc[x,'YEAR'] == wgms_e_df['Year'])]['END_WINTER'].values)
    # Time indices
    #  winter and summer balances typically have the same data for 'BEGIN_PERIOD' and 'END_PERIOD' as the annual
    #  measurements, so need to set these dates manually
    # Remove glaciers without begin or end period
    wgms_ee_df_winter = wgms_ee_df_winter.drop(np.where(np.isnan(wgms_ee_df_winter['BEGIN_PERIOD'].values))[0].tolist(), axis=0)
    wgms_ee_df_winter.reset_index(inplace=True, drop=True)
    wgms_ee_df_winter = wgms_ee_df_winter.drop(np.where(np.isnan(wgms_ee_df_winter['END_WINTER'].values))[0].tolist(), axis=0)
    wgms_ee_df_winter.reset_index(drop=True, inplace=True)
    wgms_ee_df_winter['t1_year'] = wgms_ee_df_winter['BEGIN_PERIOD'].astype(str).str.split('.').str[0].str[:4].astype(int)
    wgms_ee_df_winter['t1_month'] = wgms_ee_df_winter['BEGIN_PERIOD'].astype(str).str.split('.').str[0].str[4:6].astype(int)
    wgms_ee_df_winter['t1_day'] = wgms_ee_df_winter['BEGIN_PERIOD'].astype(str).str.split('.').str[0].str[6:].astype(int)
    wgms_ee_df_winter['t2_year'] = wgms_ee_df_winter['END_WINTER'].astype(str).str.split('.').str[0].str[:4].astype(int)
    wgms_ee_df_winter['t2_month'] = wgms_ee_df_winter['END_WINTER'].astype(str).str.split('.').str[0].str[4:6].astype(int)
    wgms_ee_df_winter['t2_day'] = wgms_ee_df_winter['END_WINTER'].astype(str).str.split('.').str[0].str[6:].astype(int)        
    wgms_ee_df_winter.loc[wgms_ee_df_winter['t1_month'] > 13,'t1_month'] = np.nan
    wgms_ee_df_winter.loc[wgms_ee_df_winter['t1_month'] < 1, 't1_month'] = np.nan
    wgms_ee_df_winter.loc[wgms_ee_df_winter['t2_month'] > 13,'t2_month'] = np.nan
    wgms_ee_df_winter.loc[wgms_ee_df_winter['t2_month'] < 1, 't2_month'] = np.nan
    wgms_ee_df_winter.loc[wgms_ee_df_winter['t1_day'] < 1, 't1_day'] = np.nan
    wgms_ee_df_winter.loc[wgms_ee_df_winter['t2_day'] < 1, 't2_day'] = np.nan
    wgms_ee_df_winter.loc[wgms_ee_df_winter['t1_day'] > 31, 't1_day'] = np.nan
    wgms_ee_df_winter.loc[wgms_ee_df_winter['t2_day'] > 31,'t2_day'] = np.nan
    wgms_ee_df_winter.loc[wgms_ee_df_winter['t1_year'] < startyear,'t1_year'] = np.nan
    wgms_ee_df_winter.loc[wgms_ee_df_winter['t2_year'] > endyear,'t2_year'] = np.nan
    wgms_ee_df_winter = wgms_ee_df_winter.dropna(subset=['t1_year', 't2_year', 't1_month', 't2_month', 't1_day', 't2_day'])
    wgms_ee_df_winter.reset_index(inplace=True, drop=True)
    
    # Correct poor dates
    for nmonth in [4,6,9,11]:
        wgms_ee_df_winter.loc[(wgms_ee_df_winter['t1_month'] == nmonth) & (wgms_ee_df_winter['t1_day'] == 31), 't1_day'] = 30
        wgms_ee_df_winter.loc[(wgms_ee_df_winter['t2_month'] == nmonth) & (wgms_ee_df_winter['t2_day'] == 31), 't2_day'] = 30

    # Calculate decimal year and drop measurements outside of calibration period
    wgms_ee_df_winter['t1_datetime'] = pd.to_datetime(
            pd.DataFrame({'year':wgms_ee_df_winter.t1_year.values, 'month':wgms_ee_df_winter.t1_month.values, 'day':wgms_ee_df_winter.t1_day.values}))
    wgms_ee_df_winter['t2_datetime'] = pd.to_datetime(
            pd.DataFrame({'year':wgms_ee_df_winter.t2_year.values, 'month':wgms_ee_df_winter.t2_month.values, 'day':wgms_ee_df_winter.t2_day.values}))
    wgms_ee_df_winter['t1_doy'] = wgms_ee_df_winter.t1_datetime.dt.strftime("%j").astype(float)
    wgms_ee_df_winter['t2_doy'] = wgms_ee_df_winter.t2_datetime.dt.strftime("%j").astype(float)
    wgms_ee_df_winter['t1_daysinyear'] = (
            (pd.to_datetime(pd.DataFrame({'year':wgms_ee_df_winter.t1_year.values, 'month':12, 'day':31})) - 
             pd.to_datetime(pd.DataFrame({'year':wgms_ee_df_winter.t1_year.values, 'month':1, 'day':1}))).dt.days + 1)
    wgms_ee_df_winter['t2_daysinyear'] = (
            (pd.to_datetime(pd.DataFrame({'year':wgms_ee_df_winter.t2_year.values, 'month':12, 'day':31})) - 
             pd.to_datetime(pd.DataFrame({'year':wgms_ee_df_winter.t2_year.values, 'month':1, 'day':1}))).dt.days + 1)
    wgms_ee_df_winter['t1'] = wgms_ee_df_winter.t1_year + wgms_ee_df_winter.t1_doy / wgms_ee_df_winter.t1_daysinyear
    wgms_ee_df_winter['t2'] = wgms_ee_df_winter.t2_year + wgms_ee_df_winter.t2_doy / wgms_ee_df_winter.t2_daysinyear
    end_datestable = dates_table.loc[dates_table.shape[0]-1, 'date']
    end_datesyear = end_datestable.year
    end_datesmonth = end_datestable.month + 1
    end_datesday = end_datestable.day
    if end_datesmonth > 12:
        end_datesyear += 1
        end_datesmonth = 1
    end_datetime = datetime.datetime(end_datesyear, end_datesmonth, end_datesday)
    wgms_ee_df_winter = wgms_ee_df_winter[wgms_ee_df_winter['t1_datetime'] >= dates_table.loc[0, 'date']]
    wgms_ee_df_winter = wgms_ee_df_winter[wgms_ee_df_winter['t2_datetime'] < end_datetime]
    wgms_ee_df_winter.reset_index(drop=True, inplace=True)
    # Annual, summer, and winter time indices
    #  exclude spinup years, since massbal fxn discarwgms_ee_df_winter spinup years
    wgms_ee_df_winter['t1_idx'] = np.nan
    wgms_ee_df_winter['t2_idx'] = np.nan
    for x in range(wgms_ee_df_winter.shape[0]):
        wgms_ee_df_winter.loc[x,'t1_idx'] = (dates_table[(wgms_ee_df_winter.loc[x, 't1_year'] == dates_table['year']) & 
                                             (wgms_ee_df_winter.loc[x, 't1_month'] == dates_table['month'])].index.values[0])
        wgms_ee_df_winter.loc[x,'t2_idx'] = (dates_table[(wgms_ee_df_winter.loc[x, 't2_year'] == dates_table['year']) & 
                                             (wgms_ee_df_winter.loc[x, 't2_month'] == dates_table['month'])].index.values[0])
    
    # Remove values that are less than 8 months
    wgms_ee_df_winter['months'] = wgms_ee_df_winter['t2_idx'] - wgms_ee_df_winter['t1_idx']
#    wgms_ee_df_winter = wgms_ee_df_winter.loc[wgms_ee_df_winter['months'] >= 8,:]
#    wgms_ee_df_winter.reset_index(inplace=True, drop=True)
    
    # Process data
    wgms_ee_df_winter['mod_winter_mwe'] = np.nan
    wgms_ee_df_winter['mod_winter_mwe_mad_ind'] = np.nan
    wgms_ee_df_winter['mod_winter_mwe_mad_cor'] = np.nan
    wgms_ee_df_winter['winter_mwe'] = wgms_ee_df_winter['WINTER_BALANCE'].values / 1000
    for nrgiid, RGIId in enumerate(main_glac_rgi.RGIId):
        glac_str = str(int(RGIId.split('-')[1].split('.')[0])) + '.' + RGIId.split('-')[1].split('.')[1]
        wgms_ee_df_winter_glac = wgms_ee_df_winter.loc[wgms_ee_df_winter['RGIId'] == RGIId,:]
        
        glac_area = main_glac_rgi.loc[nrgiid,'Area'] * 1e6
        
        # Load dataset
        netcdf_fp_stats = netcdf_fp_era5 + glac_str.split('.')[0].zfill(2) + '/ERA5/stats/'
        netcdf_fn_stats = glac_str + '_ERA5_MCMC_ba1_50sets_1979_2019_all.nc'
        ds_stats = xr.open_dataset(netcdf_fp_stats + netcdf_fn_stats)
        
        mb_monthly = ds_stats.glac_massbaltotal_monthly.values[0,:]
        mb_monthly_mad = ds_stats.glac_massbaltotal_monthly_mad.values[0,:]
        for wgms_idx in wgms_ee_df_winter_glac.index.values:
            t1_idx = int(wgms_ee_df_winter_glac.loc[wgms_idx,'t1_idx'])
            t2_idx = int(wgms_ee_df_winter_glac.loc[wgms_idx,'t2_idx'])
            mb_monthly_subset = mb_monthly[t1_idx:t2_idx+1]
            mb_monthly_mad_subset = mb_monthly_mad[t1_idx:t2_idx+1]
            mb_mwea = mb_monthly_subset.sum() / glac_area
             # aggregate monthly data assuming monthly are independent and perfectly correlated
            mb_mwea_mad_ind = ((mb_monthly_mad_subset**2).sum())**0.5 / glac_area
            mb_mwea_mad_cor = mb_monthly_mad_subset.sum() / glac_area
            wgms_ee_df_winter.loc[wgms_idx,'mod_winter_mwe'] = mb_mwea
            wgms_ee_df_winter.loc[wgms_idx,'mod_winter_mwe_mad_ind'] = mb_mwea_mad_ind
            wgms_ee_df_winter.loc[wgms_idx,'mod_winter_mwe_mad_cor'] = mb_mwea_mad_cor

    # Difference statistics
    mb_dif_winter = wgms_ee_df_winter['mod_winter_mwe'].values - wgms_ee_df_winter['winter_mwe'].values
    print('  Difference stats: \n    Mean (+/-) std [mwe]:', 
          np.round(mb_dif_winter.mean(),2), '+/-', np.round(mb_dif_winter.std(),2), 
          'count:', len(mb_dif_winter),
          '\n    Median (+/-) std [mwe]:', 
          np.round(np.median(mb_dif_winter),2), '+/- XXX', 
    #          np.round(cal_data_subset['dif_mb_mwea'].std(),2),
    #      '\n    Mean standard deviation (correlated):',np.round(mb_dif_winter.mean(),2),
    #      '\n    Mean standard deviation (uncorrelated):',np.round(cal_data_subset['mb_mwea_era_std_rsos'].mean(),2)
          )
    
    # Validation Figure
    fig, ax = plt.subplots(1, 1, squeeze=False, sharex=False, sharey=True, 
                           gridspec_kw = {'wspace':0, 'hspace':0})
    
    ax[0,0].plot(wgms_ee_df_winter['winter_mwe'].values, 
                 wgms_ee_df_winter['mod_winter_mwe'].values,
                 linewidth=0, marker='o', mec='k', mew=1, mfc='none')
    
    ax[0,0].set_xlabel('B_obs (mwe)')
    ax[0,0].set_ylabel('B_mod (mwe)')
    ax[0,0].text(0.98, 1.06, 'Glaciological (winter)', size=10, horizontalalignment='right', 
                 verticalalignment='top', transform=ax[0,0].transAxes)
    ax[0,0].set_xlim(-2,6.5)
    ax[0,0].set_ylim(-2,6.5)
    ax[0,0].plot([-2,6.5],[-2,6.5], color='k',lw=0.5)
    #ax[0,0].legend(
    #        )        
    ax[0,0].tick_params(direction='inout', right=True)
    # Save figure
    fig_fn = 'validation_mb_mwea_winter_all.png'
    fig_fp = netcdf_fp_era5 + '../analysis/figures/validation/'
    if not os.path.exists(fig_fp):
        os.makedirs(fig_fp)
    fig.set_size_inches(3,3)
    fig.savefig(fig_fp + fig_fn, bbox_inches='tight', dpi=300)

    # ----- Regional statistics -----
    regions = np.unique(wgms_ee_df_winter.O1Region.values)
    for reg in regions:
        wgms_ee_df_winter_reg = wgms_ee_df_winter.loc[wgms_ee_df_winter['O1Region'] == reg, :]
        mb_dif_winter_reg = wgms_ee_df_winter_reg['mod_winter_mwe'].values - wgms_ee_df_winter_reg['winter_mwe'].values
        
        print('Region', reg, '  Difference stats: \n    Mean (+/-) std [mwe]:',  np.round(mb_dif_winter_reg.mean(),2), 
              '+/-', np.round(mb_dif_winter_reg.std(),2), 'count:', len(mb_dif_winter_reg),
              '\n    Median (+/-) std [mwe]:', np.round(np.median(mb_dif_winter_reg),2), '+/- XXX')
        fig, ax = plt.subplots(1, 1, squeeze=False, sharex=False, sharey=True, 
                               gridspec_kw = {'wspace':0, 'hspace':0})
        
        ax[0,0].plot(wgms_ee_df_winter_reg['winter_mwe'].values, 
                     wgms_ee_df_winter_reg['mod_winter_mwe'].values,
                     linewidth=0, marker='o', mec='k', mew=1, mfc='none')
        
        ax[0,0].set_xlabel('B_obs (mwe)')
        ax[0,0].set_ylabel('B_mod (mwe)')
        ax[0,0].text(0.98, 1.06, 'Glaciological (winter) - Region ' + str(reg), size=10, horizontalalignment='right', 
                     verticalalignment='top', transform=ax[0,0].transAxes)
        ax[0,0].set_xlim(-2,6.5)
        ax[0,0].set_ylim(-2,6.5)
        ax[0,0].plot([-2,6.5],[-2,6.5], color='k',lw=0.5)
        #ax[0,0].legend(
        #        )        
        ax[0,0].tick_params(direction='inout', right=True)
        # Save figure
        fig_fn = 'validation_mb_mwe_winter_' + str(reg).zfill(2) + '.png'
        fig_fp = netcdf_fp_era5 + '../analysis/figures/validation/'
        if not os.path.exists(fig_fp):
            os.makedirs(fig_fp)
        fig.set_size_inches(3,3)
        fig.savefig(fig_fp + fig_fn, bbox_inches='tight', dpi=300)
      

#%% ===== SUMMER BALANCES ONLY =====
if wgms_summer_comparison:
    wgms_ee_df_summer = wgms_ee_df.dropna(subset=['SUMMER_BALANCE'])
    wgms_ee_df_summer = wgms_ee_df_summer.sort_values('RGIId')
    wgms_ee_df_summer.reset_index(inplace=True, drop=True)
    wgms_ee_df_summer['period'] = 'summer'
    
    wgms_glacno_list_summer = [x.split('-')[1] for x in sorted(list(np.unique(wgms_ee_df_summer['RGIId'])))]
    main_glac_rgi = modelsetup.selectglaciersrgitable(glac_no=wgms_glacno_list_summer)
    
    # Add survey dates
    wgms_ee_df_summer['TIME_SYSTEM'] = np.nan
    wgms_ee_df_summer['BEGIN_PERIOD'] = np.nan 
    wgms_ee_df_summer['END_WINTER'] = np.nan
    wgms_ee_df_summer['END_PERIOD'] = np.nan 
    for x in range(wgms_ee_df_summer.shape[0]):
        wgms_ee_df_summer.loc[x,'TIME_SYSTEM'] = (
                wgms_e_df[(wgms_ee_df_summer.loc[x,'WGMS_ID'] == wgms_e_df['WGMS_ID']) & 
                          (wgms_ee_df_summer.loc[x,'YEAR'] == wgms_e_df['Year'])]['TIME_SYSTEM'].values[0]) 
        wgms_ee_df_summer.loc[x,'BEGIN_PERIOD'] = (
                wgms_e_df[(wgms_ee_df_summer.loc[x,'WGMS_ID'] == wgms_e_df['WGMS_ID']) & 
                          (wgms_ee_df_summer.loc[x,'YEAR'] == wgms_e_df['Year'])]['BEGIN_PERIOD'].values)
        wgms_ee_df_summer.loc[x,'END_WINTER'] = (
                wgms_e_df[(wgms_ee_df_summer.loc[x,'WGMS_ID'] == wgms_e_df['WGMS_ID']) & 
                          (wgms_ee_df_summer.loc[x,'YEAR'] == wgms_e_df['Year'])]['END_WINTER'].values)
        wgms_ee_df_summer.loc[x,'END_PERIOD'] = (
                wgms_e_df[(wgms_ee_df_summer.loc[x,'WGMS_ID'] == wgms_e_df['WGMS_ID']) & 
                          (wgms_ee_df_summer.loc[x,'YEAR'] == wgms_e_df['Year'])]['END_PERIOD'].values)
        
    # Add survey dates
    wgms_ee_df_summer['TIME_SYSTEM'] = np.nan
    wgms_ee_df_summer['END_WINTER'] = np.nan 
    wgms_ee_df_summer['END_PERIOD'] = np.nan
    for x in range(wgms_ee_df_summer.shape[0]):
        wgms_ee_df_summer.loc[x,'TIME_SYSTEM'] = (
                wgms_e_df[(wgms_ee_df_summer.loc[x,'WGMS_ID'] == wgms_e_df['WGMS_ID']) & 
                          (wgms_ee_df_summer.loc[x,'YEAR'] == wgms_e_df['Year'])]['TIME_SYSTEM'].values[0]) 
        wgms_ee_df_summer.loc[x,'END_WINTER'] = (
                wgms_e_df[(wgms_ee_df_summer.loc[x,'WGMS_ID'] == wgms_e_df['WGMS_ID']) & 
                          (wgms_ee_df_summer.loc[x,'YEAR'] == wgms_e_df['Year'])]['END_WINTER'].values)
        wgms_ee_df_summer.loc[x,'END_PERIOD'] = (
                wgms_e_df[(wgms_ee_df_summer.loc[x,'WGMS_ID'] == wgms_e_df['WGMS_ID']) & 
                          (wgms_ee_df_summer.loc[x,'YEAR'] == wgms_e_df['Year'])]['END_PERIOD'].values)
    # Time indices
    #  winter and summer balances typically have the same data for 'BEGIN_PERIOD' and 'END_PERIOD' as the annual
    #  measurements, so need to set these dates manually
    # Remove glaciers without begin or end period
    wgms_ee_df_summer = wgms_ee_df_summer.drop(np.where(np.isnan(wgms_ee_df_summer['END_WINTER'].values))[0].tolist(), axis=0)
    wgms_ee_df_summer.reset_index(inplace=True, drop=True)
    wgms_ee_df_summer = wgms_ee_df_summer.drop(np.where(np.isnan(wgms_ee_df_summer['END_PERIOD'].values))[0].tolist(), axis=0)
    wgms_ee_df_summer.reset_index(drop=True, inplace=True)
    wgms_ee_df_summer['t1_year'] = wgms_ee_df_summer['END_WINTER'].astype(str).str.split('.').str[0].str[:4].astype(int)
    wgms_ee_df_summer['t1_month'] = wgms_ee_df_summer['END_WINTER'].astype(str).str.split('.').str[0].str[4:6].astype(int)
    wgms_ee_df_summer['t1_day'] = wgms_ee_df_summer['END_WINTER'].astype(str).str.split('.').str[0].str[6:].astype(int)
    wgms_ee_df_summer['t2_year'] = wgms_ee_df_summer['END_PERIOD'].astype(str).str.split('.').str[0].str[:4].astype(int)
    wgms_ee_df_summer['t2_month'] = wgms_ee_df_summer['END_PERIOD'].astype(str).str.split('.').str[0].str[4:6].astype(int)
    wgms_ee_df_summer['t2_day'] = wgms_ee_df_summer['END_PERIOD'].astype(str).str.split('.').str[0].str[6:].astype(int)      
    wgms_ee_df_summer.loc[wgms_ee_df_summer['t1_month'] > 13,'t1_month'] = np.nan
    wgms_ee_df_summer.loc[wgms_ee_df_summer['t1_month'] < 1, 't1_month'] = np.nan
    wgms_ee_df_summer.loc[wgms_ee_df_summer['t2_month'] > 13,'t2_month'] = np.nan
    wgms_ee_df_summer.loc[wgms_ee_df_summer['t2_month'] < 1, 't2_month'] = np.nan
    wgms_ee_df_summer.loc[wgms_ee_df_summer['t1_day'] < 1, 't1_day'] = np.nan
    wgms_ee_df_summer.loc[wgms_ee_df_summer['t2_day'] < 1, 't2_day'] = np.nan
    wgms_ee_df_summer.loc[wgms_ee_df_summer['t1_day'] > 31, 't1_day'] = np.nan
    wgms_ee_df_summer.loc[wgms_ee_df_summer['t2_day'] > 31,'t2_day'] = np.nan
    wgms_ee_df_summer.loc[wgms_ee_df_summer['t1_year'] < startyear,'t1_year'] = np.nan
    wgms_ee_df_summer.loc[wgms_ee_df_summer['t2_year'] > endyear,'t2_year'] = np.nan
    wgms_ee_df_summer = wgms_ee_df_summer.dropna(subset=['t1_year', 't2_year', 't1_month', 't2_month', 't1_day', 't2_day'])
    wgms_ee_df_summer.reset_index(inplace=True, drop=True)
    
    # Correct poor dates
    for nmonth in [4,6,9,11]:
        wgms_ee_df_summer.loc[(wgms_ee_df_summer['t1_month'] == nmonth) & (wgms_ee_df_summer['t1_day'] == 31), 't1_day'] = 30
        wgms_ee_df_summer.loc[(wgms_ee_df_summer['t2_month'] == nmonth) & (wgms_ee_df_summer['t2_day'] == 31), 't2_day'] = 30

    # Calculate decimal year and drop measurements outside of calibration period
    wgms_ee_df_summer['t1_datetime'] = pd.to_datetime(
            pd.DataFrame({'year':wgms_ee_df_summer.t1_year.values, 'month':wgms_ee_df_summer.t1_month.values, 'day':wgms_ee_df_summer.t1_day.values}))
    wgms_ee_df_summer['t2_datetime'] = pd.to_datetime(
            pd.DataFrame({'year':wgms_ee_df_summer.t2_year.values, 'month':wgms_ee_df_summer.t2_month.values, 'day':wgms_ee_df_summer.t2_day.values}))
    wgms_ee_df_summer['t1_doy'] = wgms_ee_df_summer.t1_datetime.dt.strftime("%j").astype(float)
    wgms_ee_df_summer['t2_doy'] = wgms_ee_df_summer.t2_datetime.dt.strftime("%j").astype(float)
    wgms_ee_df_summer['t1_daysinyear'] = (
            (pd.to_datetime(pd.DataFrame({'year':wgms_ee_df_summer.t1_year.values, 'month':12, 'day':31})) - 
             pd.to_datetime(pd.DataFrame({'year':wgms_ee_df_summer.t1_year.values, 'month':1, 'day':1}))).dt.days + 1)
    wgms_ee_df_summer['t2_daysinyear'] = (
            (pd.to_datetime(pd.DataFrame({'year':wgms_ee_df_summer.t2_year.values, 'month':12, 'day':31})) - 
             pd.to_datetime(pd.DataFrame({'year':wgms_ee_df_summer.t2_year.values, 'month':1, 'day':1}))).dt.days + 1)
    wgms_ee_df_summer['t1'] = wgms_ee_df_summer.t1_year + wgms_ee_df_summer.t1_doy / wgms_ee_df_summer.t1_daysinyear
    wgms_ee_df_summer['t2'] = wgms_ee_df_summer.t2_year + wgms_ee_df_summer.t2_doy / wgms_ee_df_summer.t2_daysinyear
    end_datestable = dates_table.loc[dates_table.shape[0]-1, 'date']
    end_datesyear = end_datestable.year
    end_datesmonth = end_datestable.month + 1
    end_datesday = end_datestable.day
    if end_datesmonth > 12:
        end_datesyear += 1
        end_datesmonth = 1
    end_datetime = datetime.datetime(end_datesyear, end_datesmonth, end_datesday)
    wgms_ee_df_summer = wgms_ee_df_summer[wgms_ee_df_summer['t1_datetime'] >= dates_table.loc[0, 'date']]
    wgms_ee_df_summer = wgms_ee_df_summer[wgms_ee_df_summer['t2_datetime'] < end_datetime]
    wgms_ee_df_summer.reset_index(drop=True, inplace=True)
    # Annual, summer, and winter time indices
    #  exclude spinup years, since massbal fxn discarwgms_ee_df_summer spinup years
    wgms_ee_df_summer['t1_idx'] = np.nan
    wgms_ee_df_summer['t2_idx'] = np.nan
    for x in range(wgms_ee_df_summer.shape[0]):
        wgms_ee_df_summer.loc[x,'t1_idx'] = (dates_table[(wgms_ee_df_summer.loc[x, 't1_year'] == dates_table['year']) & 
                                             (wgms_ee_df_summer.loc[x, 't1_month'] == dates_table['month'])].index.values[0])
        wgms_ee_df_summer.loc[x,'t2_idx'] = (dates_table[(wgms_ee_df_summer.loc[x, 't2_year'] == dates_table['year']) & 
                                             (wgms_ee_df_summer.loc[x, 't2_month'] == dates_table['month'])].index.values[0])
    
    # Remove values that are less than 8 months
    wgms_ee_df_summer['months'] = wgms_ee_df_summer['t2_idx'] - wgms_ee_df_summer['t1_idx']
#    wgms_ee_df_summer = wgms_ee_df_summer.loc[wgms_ee_df_summer['months'] >= 8,:]
#    wgms_ee_df_summer.reset_index(inplace=True, drop=True)
    
    # Process data
    wgms_ee_df_summer['mod_summer_mwe'] = np.nan
    wgms_ee_df_summer['mod_summer_mwe_mad_ind'] = np.nan
    wgms_ee_df_summer['mod_summer_mwe_mad_cor'] = np.nan
    wgms_ee_df_summer['summer_mwe'] = wgms_ee_df_summer['SUMMER_BALANCE'].values / 1000
    for nrgiid, RGIId in enumerate(main_glac_rgi.RGIId):
        glac_str = str(int(RGIId.split('-')[1].split('.')[0])) + '.' + RGIId.split('-')[1].split('.')[1]
        wgms_ee_df_summer_glac = wgms_ee_df_summer.loc[wgms_ee_df_summer['RGIId'] == RGIId,:]
        
        glac_area = main_glac_rgi.loc[nrgiid,'Area'] * 1e6
        
        # Load dataset
        netcdf_fp_stats = netcdf_fp_era5 + glac_str.split('.')[0].zfill(2) + '/ERA5/stats/'
        netcdf_fn_stats = glac_str + '_ERA5_MCMC_ba1_50sets_1979_2019_all.nc'
        ds_stats = xr.open_dataset(netcdf_fp_stats + netcdf_fn_stats)
        
        mb_monthly = ds_stats.glac_massbaltotal_monthly.values[0,:]
        mb_monthly_mad = ds_stats.glac_massbaltotal_monthly_mad.values[0,:]
        for wgms_idx in wgms_ee_df_summer_glac.index.values:
            t1_idx = int(wgms_ee_df_summer_glac.loc[wgms_idx,'t1_idx'])
            t2_idx = int(wgms_ee_df_summer_glac.loc[wgms_idx,'t2_idx'])
            mb_monthly_subset = mb_monthly[t1_idx:t2_idx+1]
            mb_monthly_mad_subset = mb_monthly_mad[t1_idx:t2_idx+1]
            mb_mwea = mb_monthly_subset.sum() / glac_area
             # aggregate monthly data assuming monthly are independent and perfectly correlated
            mb_mwea_mad_ind = ((mb_monthly_mad_subset**2).sum())**0.5 / glac_area
            mb_mwea_mad_cor = mb_monthly_mad_subset.sum() / glac_area
            wgms_ee_df_summer.loc[wgms_idx,'mod_summer_mwe'] = mb_mwea
            wgms_ee_df_summer.loc[wgms_idx,'mod_summer_mwe_mad_ind'] = mb_mwea_mad_ind
            wgms_ee_df_summer.loc[wgms_idx,'mod_summer_mwe_mad_cor'] = mb_mwea_mad_cor
#%%
    # Difference statistics
    mb_dif_summer = wgms_ee_df_summer['mod_summer_mwe'].values - wgms_ee_df_summer['summer_mwe'].values
    print('  Difference stats: \n    Mean (+/-) std [mwe]:', 
          np.round(mb_dif_summer.mean(),2), '+/-', np.round(mb_dif_summer.std(),2), 
          'count:', len(mb_dif_summer),
          '\n    Median (+/-) std [mwe]:', 
          np.round(np.median(mb_dif_summer),2), '+/- XXX', 
    #          np.round(cal_data_subset['dif_mb_mwea'].std(),2),
    #      '\n    Mean standard deviation (correlated):',np.round(mb_dif_summer.mean(),2),
    #      '\n    Mean standard deviation (uncorrelated):',np.round(cal_data_subset['mb_mwea_era_std_rsos'].mean(),2)
          )
    
    # Validation Figure
    fig, ax = plt.subplots(1, 1, squeeze=False, sharex=False, sharey=True, 
                           gridspec_kw = {'wspace':0, 'hspace':0})
    
    ax[0,0].plot(wgms_ee_df_summer['summer_mwe'].values, 
                 wgms_ee_df_summer['mod_summer_mwe'].values,
                 linewidth=0, marker='o', mec='k', mew=1, mfc='none')
    
    ax[0,0].set_xlabel('B_obs (mwe)')
    ax[0,0].set_ylabel('B_mod (mwe)')
    ax[0,0].text(0.98, 1.06, 'Glaciological (summer)', size=10, horizontalalignment='right', 
                 verticalalignment='top', transform=ax[0,0].transAxes)
    ax[0,0].set_xlim(-6,1)
    ax[0,0].set_ylim(-6,1)
    ax[0,0].plot([-6,1],[-6,1], color='k',lw=0.5)
    #ax[0,0].legend(
    #        )        
    ax[0,0].tick_params(direction='inout', right=True)
    # Save figure
    fig_fn = 'validation_mb_mwea_summer_all.png'
    fig_fp = netcdf_fp_era5 + '../analysis/figures/validation/'
    if not os.path.exists(fig_fp):
        os.makedirs(fig_fp)
    fig.set_size_inches(3,3)
    fig.savefig(fig_fp + fig_fn, bbox_inches='tight', dpi=300)

    # ----- Regional statistics -----
    regions = np.unique(wgms_ee_df_summer.O1Region.values)
    for reg in regions:
        wgms_ee_df_summer_reg = wgms_ee_df_summer.loc[wgms_ee_df_summer['O1Region'] == reg, :]
        mb_dif_summer_reg = wgms_ee_df_summer_reg['mod_summer_mwe'].values - wgms_ee_df_summer_reg['summer_mwe'].values
        
        print('Region', reg, '  Difference stats: \n    Mean (+/-) std [mwe]:',  np.round(mb_dif_summer_reg.mean(),2), 
              '+/-', np.round(mb_dif_summer_reg.std(),2), 'count:', len(mb_dif_summer_reg),
              '\n    Median (+/-) std [mwe]:', np.round(np.median(mb_dif_summer_reg),2), '+/- XXX')
        fig, ax = plt.subplots(1, 1, squeeze=False, sharex=False, sharey=True, 
                               gridspec_kw = {'wspace':0, 'hspace':0})
        
        ax[0,0].plot(wgms_ee_df_summer_reg['summer_mwe'].values, 
                     wgms_ee_df_summer_reg['mod_summer_mwe'].values,
                     linewidth=0, marker='o', mec='k', mew=1, mfc='none')
        
        ax[0,0].set_xlabel('B_obs (mwe)')
        ax[0,0].set_ylabel('B_mod (mwe)')
        ax[0,0].text(0.98, 1.06, 'Glaciological (summer) - Region ' + str(reg), size=10, horizontalalignment='right', 
                     verticalalignment='top', transform=ax[0,0].transAxes)
        ax[0,0].set_xlim(-6,1)
        ax[0,0].set_ylim(-6,1)
        ax[0,0].plot([-6,1],[-6,1], color='k',lw=0.5)
        #ax[0,0].legend(
        #        )        
        ax[0,0].tick_params(direction='inout', right=True)
        # Save figure
        fig_fn = 'validation_mb_mwe_summer_' + str(reg).zfill(2) + '.png'
        fig_fp = netcdf_fp_era5 + '../analysis/figures/validation/'
        if not os.path.exists(fig_fp):
            os.makedirs(fig_fp)
        fig.set_size_inches(3,3)
        fig.savefig(fig_fp + fig_fn, bbox_inches='tight', dpi=300)
    
    