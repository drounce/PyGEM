"""
pygemfxns_preprocessing.py is a list of the model functions that are used to preprocess the data into the proper format.

"""

# Built-in libraries
import os
import glob
import argparse
# External libraries
import pandas as pd
import numpy as np
import xarray as xr
import netCDF4 as nc
from time import strftime
from datetime import datetime
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
import matplotlib.pyplot as plt
# Local libraries
import pygem.pygem_input as pygem_prms
import pygemfxns_modelsetup as modelsetup
import pygemfxns_massbalance as massbalance
import class_climate
from analyze_mcmc import load_glacierdata_byglacno


#%% TO-DO LIST:
# - clean up create lapse rate input data (put it all in pygem_prms.py)

#%%
def getparser():
    """
    Use argparse to add arguments from the command line
    
    Parameters
    ----------
    option_createlapserates : int
        Switch for processing lapse rates (default = 0 (no))
    option_wgms : int
        Switch for processing wgms data (default = 0 (no))
        
    Returns
    -------
    Object containing arguments and their respective values.
    """
    parser = argparse.ArgumentParser(description="select pre-processing options")
    # add arguments
    parser.add_argument('-option_createlapserates', action='store', type=int, default=0,
                        help='option to create lapse rates or not (1=yes, 0=no)')
    parser.add_argument('-option_createtempstd', action='store', type=int, default=0,
                        help='option to create temperature std of daily data or not (1=yes, 0=no)')
    parser.add_argument('-option_wgms', action='store', type=int, default=0,
                        help='option to pre-process wgms data (1=yes, 0=no)')
    parser.add_argument('-option_coawstmerge', action='store', type=int, default=0,
                        help='option to merge COAWST climate data products (1=yes, 0=no)')
    parser.add_argument('-option_mbdata_fillwregional', action='store', type=int, default=0,
                        help='option to fill in missing mass balance data with regional mean and std (1=yes, 0=no)')
    parser.add_argument('-option_frontalablation_cal', action='store', type=int, default=0,
                        help='option to calibrate frontal ablation for a glacier')
    parser.add_argument('-option_farinotti2019_input', action='store', type=int, default=0,
                        help='option to produce Farinotti 2019 input products (1=yes, 0=no)')
    parser.add_argument('-option_mbdata_regional', action='store', type=int, default=0,
                        help='option to analzye mass balance data from various sources (1=yes, 0=no)')
    parser.add_argument('-option_unh_climatedata', action='store', type=int, default=0,
                        help='option to pre-process UNH climate data into standard form (1=yes, 0=no)')
    parser.add_argument('-option_regional_meltfactors', action='store', type=int, default=0,
                        help='option to produce regional meltfactors consistent with hypsometry data (1=yes, 0=no)')
    return parser

parser = getparser()
args = parser.parse_args()

#%%
#rgi_regionsO1 = [13,14,15]
#main_glac_rgi_all = pd.DataFrame()
#for region in rgi_regionsO1:
#    main_glac_rgi_region = modelsetup.selectglaciersrgitable(rgi_regionsO1=[region], rgi_regionsO2='all', 
#                                                             rgi_glac_number='all')
#    main_glac_rgi_all = main_glac_rgi_all.append(main_glac_rgi_region)


#%%
if args.option_mbdata_regional == 1:
    option_alaska = 0
    option_iceland = 0
    option_svalbard = 0
    option_russianarctic = 0
    option_andes = 0
    
    option_wgms = 1
    
    if option_wgms == 1:
        ds_wgms = pd.read_csv(pygem_prms.wgms_fp + pygem_prms.wgms_d_fn_preprocessed)
        ds_wgms = ds_wgms.sort_values('RGIId', ascending=True)
        ds_wgms.reset_index(drop=True, inplace=True)
        ds_wgms['RegO1'] = [int(x.split('-')[1].split('.')[0]) for x in ds_wgms.RGIId.values]
        ds_wgms['glacno'] = [x.split('-')[1] for x in ds_wgms.RGIId.values]
        region_list = sorted(list(ds_wgms.RegO1.unique()))
        
        for region in region_list:
            print(region)
            ds_region = ds_wgms.loc[np.where(ds_wgms.RegO1 == region)[0]]
            glacno_list_wgms = sorted(list(ds_region.glacno.unique()))
            main_glac_rgi_wgms = load_glacierdata_byglacno(glacno_list_wgms, option_loadhyps_climate=0, 
                                                           option_loadcal_data=0)
            
            print('Region ' + str(region) + ':',
                  '\n Count:', str(ds_region.shape[0]),
                  '\n Glacier Area [km2]:', str(main_glac_rgi_wgms.Area.sum()))
                  
    
    if option_alaska == 1:
        ds_fp = pygem_prms.main_directory + '/../DEMs/McNabb_data/wgms_dv/'
        ds_fn1 = 'Alaska_dV_17jun.csv'
        ds_fn2 = 'BrooksRange_dV_17jun.csv'
        
        ds1 = pd.read_csv(ds_fp + ds_fn1)
        ds2 = pd.read_csv(ds_fp + ds_fn2)
        ds = ds1.append(ds2)
        ds = ds.sort_values('RGIId', ascending=True)
        # remove nan values
        ds = (ds.drop(np.where(np.isnan(ds['smb'].values) == True)[0].tolist(), axis=0))   
        ds.reset_index(drop=True, inplace=True)
        ds['RegO1'] = [int(x.split('-')[1].split('.')[0]) for x in ds.RGIId.values]
        ds['glacno'] = [x.split('-')[1] for x in ds.RGIId.values]
        glacno_list1 = sorted(list(ds.glacno.unique()))
        
        # Add Larsen
        ds3 = pd.read_csv(pygem_prms.larsen_fp + pygem_prms.larsen_fn)
        ds3 = (ds3.drop(np.where(np.isnan(ds3['mb_mwea'].values) == True)[0].tolist(), axis=0))   
        ds3.reset_index(drop=True, inplace=True)
        ds3['RegO1'] = [int(x.split('-')[1].split('.')[0]) for x in ds3.RGIId.values]
        ds3['glacno'] = [x.split('-')[1] for x in ds3.RGIId.values]
        glacno_list2 = sorted(list(ds3.glacno.unique()))
        
        glacno_list = glacno_list1 + glacno_list2
        glacno_list = sorted(list(set(glacno_list)))
        
        main_glac_rgi = load_glacierdata_byglacno(glacno_list, option_loadhyps_climate=0, option_loadcal_data=0)
        
        print('\nRegion 1:')
        print('Count:', main_glac_rgi.shape[0], '(' + str(np.round(main_glac_rgi.shape[0] / 27108*100, 1)) + '%)')
        print('Glacier Area [km2]:', np.round(main_glac_rgi.Area.sum(),1), '(' + 
               str(np.round(main_glac_rgi.Area.sum() / 86725.053 * 100,1)) + '%)')
    
    if option_iceland == 1:
        ds_fp = pygem_prms.main_directory + '/../DEMs/McNabb_data/wgms_dv/'
        ds_fn = 'Iceland_dV_29jun.csv'
        
        ds = pd.read_csv(ds_fp + ds_fn)
        ds = ds.sort_values('RGIId', ascending=True)
        # remove nan values
        ds = (ds.drop(np.where(np.isnan(ds['smb'].values) == True)[0].tolist(), axis=0))   
        ds.reset_index(drop=True, inplace=True)
        ds['RegO1'] = [int(x.split('-')[1].split('.')[0]) for x in ds.RGIId.values]
        ds['glacno'] = [x.split('-')[1] for x in ds.RGIId.values]
        glacno_list = sorted(list(ds.glacno.unique()))
        
        main_glac_rgi = load_glacierdata_byglacno(glacno_list, option_loadhyps_climate=0, option_loadcal_data=0)
        
        print('\nRegion 6:')
        print('Count:', main_glac_rgi.shape[0], '(' + str(np.round(main_glac_rgi.shape[0] / 568*100, 1)) + '%)')
        print('Glacier Area [km2]:', np.round(main_glac_rgi.Area.sum(),1), '(' + 
               str(np.round(main_glac_rgi.Area.sum() / 11059.7 * 100,1)) + '%)')
        
    if option_svalbard == 1:
        ds_fp = pygem_prms.main_directory + '/../DEMs/McNabb_data/wgms_dv/'
        ds_fn1 = 'Svalbard_dV_29jun.csv'
        ds_fn2 = 'JanMayen_dV_29jun.csv'
        
        ds1 = pd.read_csv(ds_fp + ds_fn1)
        ds2 = pd.read_csv(ds_fp + ds_fn2)
        ds = ds1.append(ds2)
        ds = ds.sort_values('RGIId', ascending=True)
        # remove nan values
        ds = (ds.drop(np.where(np.isnan(ds['smb'].values) == True)[0].tolist(), axis=0))   
        ds.reset_index(drop=True, inplace=True)
        ds['RegO1'] = [int(x.split('-')[1].split('.')[0]) for x in ds.RGIId.values]
        ds['glacno'] = [x.split('-')[1] for x in ds.RGIId.values]
        glacno_list = sorted(list(ds.glacno.unique()))
        
        main_glac_rgi = load_glacierdata_byglacno(glacno_list, option_loadhyps_climate=0, option_loadcal_data=0)
        
        print('\nRegion 7:')
        print('Count:', main_glac_rgi.shape[0], '(' + str(np.round(main_glac_rgi.shape[0] / 1615*100, 1)) + '%)')
        print('Glacier Area [km2]:', np.round(main_glac_rgi.Area.sum(),1), '(' + 
               str(np.round(main_glac_rgi.Area.sum() / 33958.934 * 100,1)) + '%)')
        
    if option_russianarctic == 1:
        ds_fp = pygem_prms.main_directory + '/../DEMs/McNabb_data/wgms_dv/'
        ds_fn1 = 'FranzJosefLand_17jun.csv'
        ds_fn2 = 'NovayaZemlya_dV_17jun.csv'
        ds_fn3 = 'SevernayaZemlya_dV_17jun.csv'
        
        ds1 = pd.read_csv(ds_fp + ds_fn1)
        ds2 = pd.read_csv(ds_fp + ds_fn2)
        ds3 = pd.read_csv(ds_fp + ds_fn3)
        ds = ds1.append(ds2)
        ds = ds.append(ds3)
        ds = ds.sort_values('RGIId', ascending=True)
        # remove nan values
        ds = (ds.drop(np.where(np.isnan(ds['smb'].values) == True)[0].tolist(), axis=0))   
        ds.reset_index(drop=True, inplace=True)
        ds['RegO1'] = [int(x.split('-')[1].split('.')[0]) for x in ds.RGIId.values]
        ds['glacno'] = [x.split('-')[1] for x in ds.RGIId.values]
        glacno_list = sorted(list(ds.glacno.unique()))
        
        main_glac_rgi = load_glacierdata_byglacno(glacno_list, option_loadhyps_climate=0, option_loadcal_data=0)
        
        print('\nRegion 9:')
        print('Count:', main_glac_rgi.shape[0], '(' + str(np.round(main_glac_rgi.shape[0] / 1069*100, 1)) + '%)')
        print('Glacier Area [km2]:', np.round(main_glac_rgi.Area.sum(),1), '(' + 
               str(np.round(main_glac_rgi.Area.sum() / 51591.6 * 100,1)) + '%)')
    
    if option_andes == 1:
        ds_fp = pygem_prms.main_directory + '/../DEMs/Berthier/'
        ds_fn = 'MB_all_glaciers_Andes_rgi60_2000.0-2018.3.csv'
        
        ds = pd.read_csv(ds_fp + ds_fn)
        ds = ds.sort_values('RGIId', ascending=True)
        # remove nan values
        ds = (ds.drop(np.where(np.isnan(ds['MB [m w.e a-1]'].values) == True)[0].tolist(), axis=0))   
        ds.reset_index(drop=True, inplace=True)
        ds['RegO1'] = [int(x.split('-')[1].split('.')[0]) for x in ds.RGIId.values]
        ds['glacno'] = [x.split('-')[1] for x in ds.RGIId.values]
        
        main_glac_rgi = load_glacierdata_byglacno(ds.glacno.values, option_loadhyps_climate=0, option_loadcal_data=0)
        
        # Count how many in region
        ds_r16 = ds.loc[np.where(ds.RegO1 == 16)[0]]
        main_glac_rgi16 = main_glac_rgi.loc[np.where(main_glac_rgi.O1Region == 16)[0]]
        print('Region 16:')
        print('Count: 2891 glaciers in South America; others in Region 16 in Mexico, Africa, and Papau New Guinea')
        print('Count:', ds_r16.shape[0], '(' + str(np.round(ds_r16.shape[0] / 2891*100, 1)) + '%)')
        print('Glacier Area [km2]:', np.round(main_glac_rgi16.Area.sum(),1), '(' + 
               str(np.round(main_glac_rgi16.Area.sum() / 2341 * 100,1)) + '%)')
        
        ds_r17 = ds.loc[np.where(ds.RegO1 == 17)[0]]
        main_glac_rgi17 = main_glac_rgi.loc[np.where(main_glac_rgi.O1Region == 17)[0]]
        print('\nRegion 17:')
        print('Count:', ds_r17.shape[0], '(' + str(np.round(ds_r17.shape[0] / 15908*100, 1)) + '%)')
        print('Glacier Area [km2]:', np.round(main_glac_rgi17.Area.sum(),1), '(' + 
               str(np.round(main_glac_rgi17.Area.sum() / 29429 * 100,1)) + '%)')

    
    
#%% REMOVE POOR OBSERVATIONS AND FILL MISSING MB DATA WITH REGIONAL MEAN AND STD
if args.option_mbdata_fillwregional == 1:
    print('Filling in missing data with regional estimates...')
    # Input data
    ds_fp = pygem_prms.shean_fp
    ds_fn = 'hma_mb_20190215_0815_std+mean.csv'
    
#    dict_fn = '/Users/davidrounce/Documents/Dave_Rounce/HiMAT/qgis_himat/rgi60_HMA_dict_kaab.csv'
#    dict_cn = 'kaab_name'
#    dict_csv = pd.read_csv(dict_fn)
#    rgi_dict = dict(zip(dict_csv.RGIId, dict_csv[dict_cn]))
    dict_fn = '/Users/davidrounce/Documents/Dave_Rounce/HiMAT/qgis_himat/rgi60_HMA_dict_bolch.csv'
    dict_cn = 'bolch_name'
    dict_csv = pd.read_csv(dict_fn)
    rgi_dict = dict(zip(dict_csv.RGIId, dict_csv[dict_cn]))
    
    # Load mass balance measurements and identify unique rgi regions 
    ds = pd.read_csv(ds_fp + ds_fn)
    ds = ds.sort_values('RGIId', ascending=True)
    ds.reset_index(drop=True, inplace=True)
    ds['RGIId'] = round(ds['RGIId'], 5)
    ds['rgi_regO1'] = ds['RGIId'].astype(int)
    ds['rgi_str'] = ds['RGIId'].apply(lambda x: '%.5f' % x)
    rgi_regionsO1 = sorted(ds['rgi_regO1'].unique().tolist())

    main_glac_rgi = pd.DataFrame()
    for region in rgi_regionsO1:
        main_glac_rgi_region = modelsetup.selectglaciersrgitable(rgi_regionsO1=[region], rgi_regionsO2='all', 
                                                                 rgi_glac_number='all')
        main_glac_rgi = main_glac_rgi.append(main_glac_rgi_region)
    main_glac_rgi.reset_index(drop=True, inplace=True)
    
    # Add mass balance and uncertainty to main_glac_rgi
    # Select glaciers with data such that main_glac_rgi and ds indices are aligned correctly
    main_glac_rgi_wdata = (
            main_glac_rgi.iloc[np.where(main_glac_rgi['RGIId_float'].isin(ds['RGIId']) == True)[0],:]).copy()
    main_glac_rgi_wdata.reset_index(drop=True, inplace=True)
    dict_rgi_mb = dict(zip(main_glac_rgi_wdata.RGIId, ds.mb_mwea))
    dict_rgi_mb_sigma = dict(zip(main_glac_rgi_wdata.RGIId, ds.mb_mwea_sigma))
    main_glac_rgi['mb_mwea'] = main_glac_rgi.RGIId.map(dict_rgi_mb)
    main_glac_rgi['mb_mwea_sigma'] = main_glac_rgi.RGIId.map(dict_rgi_mb_sigma)

    # Too high of sigma causes large issues for model
    #  sigma theoretically should be independent of region
    all_sigma_mean = main_glac_rgi['mb_mwea_sigma'].mean()
    all_sigma_std = main_glac_rgi['mb_mwea_sigma'].std()
#    all_sigma_q1 = main_glac_rgi['mb_mwea_sigma'].quantile(0.25)
#    all_sigma_q3 = main_glac_rgi['mb_mwea_sigma'].quantile(0.75)
#    all_sigma_IQR = all_sigma_q3 - all_sigma_q1
    all_sigma_threshold = all_sigma_mean + 3 * all_sigma_std
    
    print('Sigma Threshold:\n# glaciers removed:', 
          main_glac_rgi.query('(mb_mwea_sigma > @all_sigma_threshold)').shape[0],
          '\n% Area removed:', 
          np.round(main_glac_rgi.query('(mb_mwea_sigma > @all_sigma_threshold)').Area.sum() / main_glac_rgi.Area.sum() 
          * 100,1))
    
    main_glac_rgi.loc[main_glac_rgi.query('(mb_mwea_sigma > @all_sigma_threshold)').index.values, 'mb_mwea'] = np.nan
    (main_glac_rgi.loc[main_glac_rgi.query('(mb_mwea_sigma > @all_sigma_threshold)').index.values, 
                       'mb_mwea_sigma']) = np.nan
    
    # Loop through groups
    main_glac_rgi['group'] = main_glac_rgi.RGIId.map(rgi_dict)
    # Regional mass balance mean and stdev
    groups = main_glac_rgi.group.unique().tolist()
    group_cn = 'group'
    groups = [x for x in groups if str(x) != 'nan']

    removal_glaciers = 0
    removal_area = 0
    total_area = 0
    for ngroup, group in enumerate(groups):
        # Select subset of data
        main_glac_rgi_group = main_glac_rgi.loc[main_glac_rgi[group_cn] == group].copy()
        group_stats = pd.Series()
        group_stats['mean'] = np.nanmean(main_glac_rgi_group['mb_mwea'])
        group_stats['std'] = main_glac_rgi_group['mb_mwea'].std()
#        group_stats['q1'] = main_glac_rgi_group['mb_mwea'].quantile(0.25)
#        group_stats['q3'] = main_glac_rgi_group['mb_mwea'].quantile(0.75)
#        group_stats['IQR'] = group_stats['q3'] - group_stats['q1']
#        group_stats['sigma_mean'] = main_glac_rgi_group['mb_mwea_sigma'].mean()
#        group_stats['sigma_std'] = main_glac_rgi_group['mb_mwea_sigma'].std()
#        group_stats['sigma_q1'] = main_glac_rgi_group['mb_mwea_sigma'].quantile(0.25)
#        group_stats['sigma_q3'] = main_glac_rgi_group['mb_mwea_sigma'].quantile(0.75)
#        group_stats['sigma_IQR'] = group_stats['sigma_q3'] - group_stats['sigma_q1']
    
        main_glac_rgi_group['zscore'] = (main_glac_rgi_group['mb_mwea'] - group_stats['mean']) / group_stats['std']
        main_glac_rgi.loc[main_glac_rgi.query('(group == @group)').index.values, 'zscore'] = main_glac_rgi_group.zscore
        
        group_stats['mean_weighted'] = (
                (main_glac_rgi_group.query('(-3 <= zscore <= 3)').mb_mwea * 
                 main_glac_rgi_group.query('(-3 <= zscore <= 3)').Area).sum() / 
                main_glac_rgi_group.query('(-3 <= zscore <= 3)').Area.sum())

        group_stats['std_weighted'] = (
                ((main_glac_rgi_group.query('(-3 <= zscore <= 3)').mb_mwea_sigma**2 *
                  main_glac_rgi_group.query('(-3 <= zscore <= 3)').Area).sum() / 
                  main_glac_rgi_group.query('(-3 <= zscore <= 3)').Area.sum())**0.5)
                
        print('\n',group, 'mean:', np.round(group_stats.mean_weighted, 2), 'std:', np.round(group_stats.std_weighted,2), 
              '\n# glaciers removed:', main_glac_rgi.query('(group == @group) & (abs(zscore) > 3)').shape[0],
              '\n% area removed:', np.round(main_glac_rgi.query('(group == @group) & (abs(zscore) > 3)').Area.sum() / 
              main_glac_rgi.query('(group == @group)').Area.sum() * 100,2))
        
        removal_glaciers += main_glac_rgi.query('(group == @group) & (abs(zscore) > 3)').shape[0]
        removal_area += main_glac_rgi.query('(group == @group) & (abs(zscore) > 3)').Area.sum()
        total_area += main_glac_rgi.query('(group == @group)').Area.sum()
        
        # Replace regional outliers with mean and std
        main_glac_rgi.loc[main_glac_rgi.query('(group == @group) & (abs(zscore) > 3)').index.values, 'mb_mwea'] = (
                group_stats['mean_weighted'])
        main_glac_rgi.loc[main_glac_rgi.query('(group == @group) & (abs(zscore) > 3)').index.values, 'mb_mwea_sigma'] = (
                group_stats['std_weighted'])
        
        # Replace missing values with mean and std
        main_glac_rgi.loc[(main_glac_rgi['group'] == group) & 
                          (main_glac_rgi.mb_mwea.isnull() == True), 'mb_mwea'] = group_stats['mean_weighted']
        main_glac_rgi.loc[(main_glac_rgi['group'] == group) & 
                          (main_glac_rgi.mb_mwea_sigma.isnull() == True), 'mb_mwea_sigma'] = group_stats['std_weighted']
    
    print('\nHMA:\n # glaciers removed:', removal_glaciers, 'area_removed[%]:', removal_area/total_area*100)
        
    # Glaciers without a region compare to all HMA
    all_mean_weighted = ((main_glac_rgi.query('(-3 <= zscore <= 3)').mb_mwea * 
                          main_glac_rgi.query('(-3 <= zscore <= 3)').Area).sum() / 
                         main_glac_rgi.query('(-3 <= zscore <= 3)').Area.sum())
    all_std_weighted = (((main_glac_rgi.query('(-3 <= zscore <= 3)').mb_mwea_sigma**2 * 
                          main_glac_rgi.query('(-3 <= zscore <= 3)').Area).sum() / 
                         main_glac_rgi.query('(-3 <= zscore <= 3)').Area.sum())**0.5)
    
    # Replace outliers with mean and std
    main_glac_rgi.loc[main_glac_rgi['group'].isnull() == True, 'zscore'] = (
            main_glac_rgi.loc[main_glac_rgi['group'].isnull() == True, 'mb_mwea'] - all_mean_weighted / all_std_weighted)
    main_glac_rgi.loc[(main_glac_rgi['group'].isnull() == True) & 
                      (abs(main_glac_rgi['zscore']) > 3), 'mb_mwea'] = all_mean_weighted
    main_glac_rgi.loc[(main_glac_rgi['group'].isnull() == True) & 
                      (abs(main_glac_rgi['zscore']) > 3), 'mb_mwea_sigma'] = all_std_weighted
                      
    # Replace missing values with mean and std
    main_glac_rgi.loc[(main_glac_rgi['group'].isnull() == True) & 
                      (main_glac_rgi['mb_mwea'].isnull() == True), 'mb_mwea'] = all_mean_weighted
    main_glac_rgi.loc[(main_glac_rgi['group'].isnull() == True) & 
                      (main_glac_rgi['mb_mwea_sigma'].isnull() == True), 'mb_mwea_sigma'] = all_std_weighted
    
    print('\nHMA mean:', np.round(all_mean_weighted,2), 'std:', np.round(all_std_weighted,2))
    
#    # Export filled dataset
#    ds_export = pd.DataFrame(columns=ds.columns)
#    ds_export['RGIId'] = main_glac_rgi['RGIId_float']
#    export_cns = ds.columns.tolist()
#    remove_cns = ['RGIId', 'rgi_regO1', 'rgi_str', 'mb_mwea', 'mb_mwea_sigma', 'mb_m3wea', 'mb_m3wea_sigma']
#    for cn in remove_cns:
#        export_cns.remove(cn)
#    for cn in export_cns:
#        export_dict = dict(zip(main_glac_rgi_wdata.RGIId, ds[cn]))
#        ds_export[cn] = main_glac_rgi.RGIId.map(export_dict)
#    
#    ds_export['mb_mwea'] = main_glac_rgi['mb_mwea']
#    ds_export['mb_mwea_sigma'] = main_glac_rgi['mb_mwea_sigma']
#    nodata_idx = np.where(ds_export['z_min'].isnull() == True)[0]
#    ds_export.loc[nodata_idx, 'area_m2'] = main_glac_rgi.loc[nodata_idx, 'Area'] * 10**6
#    ds_export['mb_m3wea'] = ds_export['mb_mwea'] * ds_export['area_m2']
#    ds_export['mb_m3wea_sigma'] = ds_export['mb_mwea_sigma'] * ds_export['area_m2']
#    ds_export.loc[nodata_idx, 't1'] = ds_export['t1'].min()
#    ds_export.loc[nodata_idx, 't2'] = ds_export['t2'].max()
#    ds_export.loc[nodata_idx, 'dt'] = ds_export['t2'] - ds_export['t1']
#    ds_export.loc[nodata_idx, 'z_med'] = main_glac_rgi.loc[nodata_idx, 'Zmed']
#    ds_export.loc[nodata_idx, 'z_min'] = main_glac_rgi.loc[nodata_idx, 'Zmin']
#    ds_export.loc[nodata_idx, 'z_max'] = main_glac_rgi.loc[nodata_idx, 'Zmax']
#    ds_export.loc[nodata_idx, 'z_slope'] = main_glac_rgi.loc[nodata_idx, 'Slope']
#    ds_export.loc[nodata_idx, 'z_aspect'] = main_glac_rgi.loc[nodata_idx, 'Aspect']
#    output_fn = ds_fn.replace('.csv', '_all_filled.csv')
#    ds_export.to_csv(ds_fp + output_fn, index=False)

#%% COAWST Climate Data
if args.option_coawstmerge == 1:
    print('Merging COAWST climate data...')

    def coawst_merge_netcdf(vn, coawst_fp, coawst_fn_prefix):
        """
        Merge COAWST products to form a timeseries

        Parameters
        ----------
        vn : str
            variable name
        coawst_fp : str
            filepath of COAWST climate data
        
        Returns
        -------
        exports netcdf of merged climate data
        """
        # Sorted list of files to merge
        ds_list = []
        for i in os.listdir(coawst_fp):
            if i.startswith(coawst_fn_prefix):
                ds_list.append(i)
        ds_list = sorted(ds_list)
        # Merge files
        count = 0
        for i in ds_list:
            count += 1
            ds = xr.open_dataset(coawst_fp + i)
            var = ds[vn].values
            lat = ds.LAT.values
            lon = ds.LON.values
            if vn == 'HGHT':
                var_all = var
            elif count == 1:
                var_all = var
                month_start_str = i.split('_')[3].split('.')[0].split('-')[0]
            elif count == len(ds_list):
                var_all = np.append(var_all, var, axis=0)
                month_end_str = i.split('_')[3].split('.')[0].split('-')[1]
            else:
                var_all = np.append(var_all, var, axis=0)
                
            print('Max TOTPRECIP:', ds.TOTPRECIP.values.max())
            print('Max TOTRAIN:', ds.TOTRAIN.values.max())
            print('Max TOTSNOW:', ds.TOTSNOW.values.max())
                
        # Merged dataset
        if vn == 'HGHT':
            ds_all_fn = coawst_fn_prefix + vn + '.nc'
            ds_all = xr.Dataset({vn: (['x', 'y'], var)},
                        coords={'LON': (['x', 'y'], lon),
                                'LAT': (['x', 'y'], lat)},
                        attrs=ds[vn].attrs)
            ds_all[vn].attrs = ds[vn].attrs
        else:
            # reference time in format for pd.date_range
            time_ref = month_start_str[0:4] + '-' + month_start_str[4:6] + '-' + month_start_str[6:8]
            ds_all_fn = coawst_fn_prefix + vn + '_' + month_start_str + '-' + month_end_str + '.nc'
            ds_all = xr.Dataset({vn: (['time', 'x', 'y'], var_all)},
                                coords={'LON': (['x', 'y'], lon),
                                        'LAT': (['x', 'y'], lat),
                                        'time': pd.date_range(time_ref, periods=len(ds_list), freq='MS'),
                                        'reference_time': pd.Timestamp(time_ref)})
            ds_all[vn].attrs = ds[vn].attrs
        # Export to netcdf
        ds_all.to_netcdf(coawst_fp + '../' + ds_all_fn)
        ds_all.close()
        
    # Load climate data
    gcm = class_climate.GCM(name='COAWST')
    # Process each variable
    for vn in pygem_prms.coawst_vns:
        coawst_merge_netcdf(vn, pygem_prms.coawst_fp_unmerged, pygem_prms.coawst_fn_prefix_d02)
#        coawst_merge_netcdf(vn, pygem_prms.coawst_fp_unmerged, pygem_prms.coawst_fn_prefix_d01)
        

#%% WGMS PRE-PROCESSING
if args.option_wgms == 1:
    print('Processing WGMS datasets...')
    # Connect the WGMS mass balance datasets with the RGIIds and relevant elevation bands
    # Note: WGMS reports the RGI in terms of V5 as opposed to V6.  Some of the glaciers have changed their RGIId between
    #       the two versions, so need to convert WGMS V5 Ids to V6 Ids using the GLIMSID.
    # PROBLEMS WITH DATASETS:
    #  - need to be careful with information describing dataset as some descriptions appear to be incorrect.
        
    # ===== Dictionaries (WGMS --> RGIID V6) =====
    # Load RGI version 5 & 6 and create dictionary linking the two
    #  -required to avoid errors associated with changes in RGIId between the two versions in some regions
    rgiv6_fn_all = glob.glob(pygem_prms.rgiv6_fn_prefix)
    rgiv5_fn_all = glob.glob(pygem_prms.rgiv5_fn_prefix)
    # Create dictionary of all regions
    #  - regions that didn't change between versions (ex. 13, 14, 15) will all the be same.  Others that have changed
    #    may vary greatly.
    for n in range(len(rgiv6_fn_all)):
        print('Region', n+1)
        rgiv6_fn = glob.glob(pygem_prms.rgiv6_fn_prefix)[n]
        rgiv6 = pd.read_csv(rgiv6_fn, encoding='latin1')
        rgiv5_fn = glob.glob(pygem_prms.rgiv5_fn_prefix)[n]
        rgiv5 = pd.read_csv(rgiv5_fn, encoding='latin1')
        # Dictionary to link versions 5 & 6
        rgi_version_compare = rgiv5[['RGIId', 'GLIMSId']].copy()
        rgi_version_compare['RGIIdv6'] = np.nan
        # Link versions 5 & 6 based on GLIMSID
        for r in range(rgiv5.shape[0]):
            try:
                # Use GLIMSID
                rgi_version_compare.iloc[r,2] = (
                        rgiv6.iloc[rgiv6['GLIMSId'].values == rgiv5.loc[r,'GLIMSId'],0].values[0])
        #        # Use Lat/Lon
        #        latlon_dif = abs(rgiv6[['CenLon', 'CenLat']].values - rgiv5[['CenLon', 'CenLat']].values[r,:])
        #        latlon_dif[abs(latlon_dif) < 1e-6] = 0
        #        rgi_version_compare.iloc[r,2] = rgiv6.iloc[np.where(latlon_dif[:,0] + latlon_dif[:,1] < 0.001)[0][0],0]
            except:
                rgi_version_compare.iloc[r,2] = np.nan
        rgiv56_dict_reg = dict(zip(rgi_version_compare['RGIId'], rgi_version_compare['RGIIdv6']))
        latdict_reg = dict(zip(rgiv6['RGIId'], rgiv6['CenLat']))
        londict_reg = dict(zip(rgiv6['RGIId'], rgiv6['CenLon']))
        rgiv56_dict = {}
        latdict = {}
        londict = {}
        rgiv56_dict.update(rgiv56_dict_reg)
        latdict.update(latdict_reg)
        londict.update(londict_reg)
    # RGI Lookup table
    rgilookup = pd.read_csv(pygem_prms.rgilookup_fullfn, skiprows=2)
    rgidict = dict(zip(rgilookup['FoGId'], rgilookup['RGIId']))
    # WGMS Lookup table
    wgmslookup = pd.read_csv(pygem_prms.wgms_fp + pygem_prms.wgms_lookup_fn, encoding='latin1')
    wgmsdict = dict(zip(wgmslookup['WGMS_ID'], wgmslookup['RGI_ID']))
    # Manual lookup table
    mandict = {10402: 'RGI60-13.10093',
               10401: 'RGI60-15.03734',
               6846: 'RGI60-15.12707'}
    #%%
    # ===== WGMS (D) Geodetic mass balance data =====
    if 'wgms_d' in pygem_prms.wgms_datasets:
        print('Processing geodetic thickness change data')
        wgms_mb_geo_all = pd.read_csv(pygem_prms.wgms_fp + pygem_prms.wgms_d_fn, encoding='latin1')
        wgms_mb_geo_all['RGIId_rgidict'] = wgms_mb_geo_all['WGMS_ID'].map(rgidict)
        wgms_mb_geo_all['RGIId_mandict'] = wgms_mb_geo_all['WGMS_ID'].map(mandict)
        wgms_mb_geo_all['RGIId_wgmsdict'] = wgms_mb_geo_all['WGMS_ID'].map(wgmsdict)
        wgms_mb_geo_all['RGIId_wgmsdictv6'] = wgms_mb_geo_all['RGIId_wgmsdict'].map(rgiv56_dict)
        # Use dictionaries to convert wgms data to RGIIds
        wgms_mb_geo_RGIIds_all_raw_wdicts = wgms_mb_geo_all[['RGIId_rgidict', 'RGIId_mandict','RGIId_wgmsdictv6']]
        wgms_mb_geo_RGIIds_all_raw = wgms_mb_geo_RGIIds_all_raw_wdicts.apply(lambda x: sorted(x, key=pd.isnull)[0], 1)
        # Determine regions and glacier numbers
        wgms_mb_geo_all['RGIId'] = wgms_mb_geo_RGIIds_all_raw.values
        wgms_mb_geo_all['version'], wgms_mb_geo_all['glacno'] = wgms_mb_geo_RGIIds_all_raw.str.split('-').dropna().str
        wgms_mb_geo_all['glacno'] = wgms_mb_geo_all['glacno'].apply(pd.to_numeric)
        wgms_mb_geo_all['region'] = wgms_mb_geo_all['glacno'].apply(np.floor)
        wgms_mb_geo = wgms_mb_geo_all[np.isfinite(wgms_mb_geo_all['glacno'])].sort_values('glacno')
        wgms_mb_geo.reset_index(drop=True, inplace=True)
        # Add latitude and longitude 
        wgms_mb_geo['CenLat'] = wgms_mb_geo['RGIId'].map(latdict)
        wgms_mb_geo['CenLon'] = wgms_mb_geo['RGIId'].map(londict)

        # Export relevant information
        wgms_mb_geo_export = pd.DataFrame()
        export_cols_geo = ['RGIId', 'glacno', 'WGMS_ID', 'CenLat', 'CenLon', 'REFERENCE_DATE', 'SURVEY_DATE', 
                           'LOWER_BOUND', 'UPPER_BOUND', 'AREA_SURVEY_YEAR', 'AREA_CHANGE', 'AREA_CHANGE_UNC', 
                           'THICKNESS_CHG', 'THICKNESS_CHG_UNC', 'VOLUME_CHANGE', 'VOLUME_CHANGE_UNC', 
                           'SD_PLATFORM_METHOD', 'RD_PLATFORM_METHOD', 'REFERENCE', 'REMARKS', 'INVESTIGATOR', 
                           'SPONS_AGENCY']
        wgms_mb_geo_export = wgms_mb_geo.loc[(np.isfinite(wgms_mb_geo['THICKNESS_CHG']) | 
                                             (np.isfinite(wgms_mb_geo['VOLUME_CHANGE']))), export_cols_geo]
        # Add observation type for comparison (massbalance, snowline, etc.)
        wgms_mb_geo_export[pygem_prms.wgms_obs_type_cn] = 'mb_geo'
        wgms_mb_geo_export.reset_index(drop=True, inplace=True)
        wgms_mb_geo_export_fn = pygem_prms.wgms_fp + pygem_prms.wgms_d_fn_preprocessed
        wgms_mb_geo_export.to_csv(wgms_mb_geo_export_fn)
    
    # ===== WGMS (EE) Glaciological mass balance data =====
    if 'wgms_ee' in pygem_prms.wgms_datasets:
        print('Processing glaciological mass balance data')
        wgms_mb_glac_all = pd.read_csv(pygem_prms.wgms_fp + pygem_prms.wgms_ee_fn, encoding='latin1')
        wgms_mb_glac_all['RGIId_rgidict'] = wgms_mb_glac_all['WGMS_ID'].map(rgidict)
        wgms_mb_glac_all['RGIId_mandict'] = wgms_mb_glac_all['WGMS_ID'].map(mandict)
        wgms_mb_glac_all['RGIId_wgmsdict'] = wgms_mb_glac_all['WGMS_ID'].map(wgmsdict)
        wgms_mb_glac_all['RGIId_wgmsdictv6'] = wgms_mb_glac_all['RGIId_wgmsdict'].map(rgiv56_dict)
        # Use dictionaries to convert wgms data to RGIIds
        wgms_mb_glac_RGIIds_all_raw_wdicts = wgms_mb_glac_all[['RGIId_rgidict', 'RGIId_mandict','RGIId_wgmsdictv6']]
        wgms_mb_glac_RGIIds_all_raw = wgms_mb_glac_RGIIds_all_raw_wdicts.apply(lambda x: sorted(x, key=pd.isnull)[0], 1)        
        # Determine regions and glacier numbers
        wgms_mb_glac_all['RGIId'] = wgms_mb_glac_RGIIds_all_raw.values
        wgms_mb_glac_all['version'], wgms_mb_glac_all['glacno'] = (
                wgms_mb_glac_RGIIds_all_raw.str.split('-').dropna().str)
        wgms_mb_glac_all['glacno'] = wgms_mb_glac_all['glacno'].apply(pd.to_numeric)
        wgms_mb_glac_all['region'] = wgms_mb_glac_all['glacno'].apply(np.floor)
        wgms_mb_glac = wgms_mb_glac_all[np.isfinite(wgms_mb_glac_all['glacno'])].sort_values('glacno')
        wgms_mb_glac.reset_index(drop=True, inplace=True)
        # Add latitude and longitude 
        wgms_mb_glac['CenLat'] = wgms_mb_glac['RGIId'].map(latdict)
        wgms_mb_glac['CenLon'] = wgms_mb_glac['RGIId'].map(londict)
        # Import MB overview data to extract survey dates
        wgms_mb_overview = pd.read_csv(pygem_prms.wgms_fp + pygem_prms.wgms_e_fn, encoding='latin1')
        wgms_mb_glac['BEGIN_PERIOD'] = np.nan 
        wgms_mb_glac['END_PERIOD'] = np.nan 
        wgms_mb_glac['TIME_SYSTEM'] = np.nan
        wgms_mb_glac['END_WINTER'] = np.nan
        for x in range(wgms_mb_glac.shape[0]):
            wgms_mb_glac.loc[x,'BEGIN_PERIOD'] = (
                    wgms_mb_overview[(wgms_mb_glac.loc[x,'WGMS_ID'] == wgms_mb_overview['WGMS_ID']) & 
                                     (wgms_mb_glac.loc[x,'YEAR'] == wgms_mb_overview['Year'])]['BEGIN_PERIOD'].values)
            wgms_mb_glac.loc[x,'END_WINTER'] = (
                    wgms_mb_overview[(wgms_mb_glac.loc[x,'WGMS_ID'] == wgms_mb_overview['WGMS_ID']) & 
                                     (wgms_mb_glac.loc[x,'YEAR'] == wgms_mb_overview['Year'])]['END_WINTER'].values)
            wgms_mb_glac.loc[x,'END_PERIOD'] = (
                    wgms_mb_overview[(wgms_mb_glac.loc[x,'WGMS_ID'] == wgms_mb_overview['WGMS_ID']) & 
                                     (wgms_mb_glac.loc[x,'YEAR'] == wgms_mb_overview['Year'])]['END_PERIOD'].values)
            wgms_mb_glac.loc[x,'TIME_SYSTEM'] = (
                    wgms_mb_overview[(wgms_mb_glac.loc[x,'WGMS_ID'] == wgms_mb_overview['WGMS_ID']) & 
                                     (wgms_mb_glac.loc[x,'YEAR'] == wgms_mb_overview['Year'])]['TIME_SYSTEM'].values[0])  
        # Split summer, winter, and annual into separate rows so each becomes a data point in the calibration
        #  if summer and winter exist, then discard annual to avoid double-counting the annual measurement
        export_cols_annual = ['RGIId', 'glacno', 'WGMS_ID', 'CenLat', 'CenLon', 'YEAR', 'TIME_SYSTEM', 'BEGIN_PERIOD', 
                              'END_WINTER', 'END_PERIOD', 'LOWER_BOUND', 'UPPER_BOUND', 'ANNUAL_BALANCE', 
                              'ANNUAL_BALANCE_UNC', 'REMARKS']
        export_cols_summer = ['RGIId', 'glacno', 'WGMS_ID', 'CenLat', 'CenLon', 'YEAR', 'TIME_SYSTEM', 'BEGIN_PERIOD', 
                              'END_WINTER', 'END_PERIOD', 'LOWER_BOUND', 'UPPER_BOUND', 'SUMMER_BALANCE', 
                              'SUMMER_BALANCE_UNC', 'REMARKS']
        export_cols_winter = ['RGIId', 'glacno', 'WGMS_ID', 'CenLat', 'CenLon', 'YEAR', 'TIME_SYSTEM', 'BEGIN_PERIOD', 
                              'END_WINTER', 'END_PERIOD', 'LOWER_BOUND', 'UPPER_BOUND', 'WINTER_BALANCE', 
                              'WINTER_BALANCE_UNC', 'REMARKS']
        wgms_mb_glac_annual = wgms_mb_glac.loc[((np.isnan(wgms_mb_glac['WINTER_BALANCE'])) & 
                                                (np.isnan(wgms_mb_glac['SUMMER_BALANCE']))), export_cols_annual]
        wgms_mb_glac_summer = wgms_mb_glac.loc[np.isfinite(wgms_mb_glac['SUMMER_BALANCE']), export_cols_summer]
        wgms_mb_glac_winter = wgms_mb_glac.loc[np.isfinite(wgms_mb_glac['WINTER_BALANCE']), export_cols_winter]
        # Assign a time period to each of the measurements, which will be used for comparison with model data 
        wgms_mb_glac_annual['period'] = 'annual'
        wgms_mb_glac_summer['period'] = 'summer'
        wgms_mb_glac_winter['period'] = 'winter'
        # Rename columns such that all rows are the same
        wgms_mb_glac_annual.rename(columns={'ANNUAL_BALANCE': 'BALANCE', 'ANNUAL_BALANCE_UNC': 'BALANCE_UNC'}, 
                                   inplace=True)
        wgms_mb_glac_summer.rename(columns={'SUMMER_BALANCE': 'BALANCE', 'SUMMER_BALANCE_UNC': 'BALANCE_UNC'}, 
                                   inplace=True)
        wgms_mb_glac_winter.rename(columns={'WINTER_BALANCE': 'BALANCE', 'WINTER_BALANCE_UNC': 'BALANCE_UNC'}, 
                                   inplace=True)
        # Export relevant information
        wgms_mb_glac_export = (pd.concat([wgms_mb_glac_annual, wgms_mb_glac_summer, wgms_mb_glac_winter])
                                         .sort_values(['glacno', 'YEAR']))
        # Add observation type for comparison (massbalance, snowline, etc.)
        wgms_mb_glac_export[pygem_prms.wgms_obs_type_cn] = 'mb_glac'
        wgms_mb_glac_export.reset_index(drop=True, inplace=True)
        wgms_mb_glac_export_fn = pygem_prms.wgms_fp + pygem_prms.wgms_ee_fn_preprocessed
        wgms_mb_glac_export.to_csv(wgms_mb_glac_export_fn)


#%% Create netcdf file of lapse rates from temperature pressure level data
if args.option_createlapserates == 1:
    # Input data
    gcm_fp = pygem_prms.era5_fp
    gcm_fn = pygem_prms.era5_pressureleveltemp_fn
        
    tempname = 't'
    levelname = 'level'
    elev_idx_max = 1
    elev_idx_min = 19
    output_fn= 'ERA5_lapserates.nc'
    
    # Open dataset
    ds = xr.open_dataset(gcm_fp + gcm_fn)    
    # extract the pressure levels [Pa]
    if ds[levelname].attrs['units'] == 'millibars':
        # convert pressure levels from millibars to Pa
        levels = ds[levelname].values * 100
    # Compute the elevation [m a.s.l] of the pressure levels using the barometric pressure formula (pressure in Pa)
    elev = -pygem_prms.R_gas*pygem_prms.temp_std/(pygem_prms.gravity*pygem_prms.molarmass_air)*np.log(levels/pygem_prms.pressure_std)

    # Calculate lapse rates by year
    lr = np.zeros((ds.time.shape[0], ds.latitude.shape[0], ds.longitude.shape[0]))
    for ntime, t in enumerate(ds.time.values):        
        print('time:', ntime, t)
        
        ds_subset = ds[tempname][ntime, elev_idx_max:elev_idx_min+1, :, :].values
        ds_subset_reshape = ds_subset.reshape(ds_subset.shape[0],-1)
        lr[ntime,:,:] = (np.polyfit(elev[elev_idx_max:elev_idx_min+1], ds_subset_reshape, deg=1)[0]
                         .reshape(ds_subset.shape[1:]))

    # Export lapse rates with attibutes
    output_ds = ds.copy()
    output_ds = output_ds.drop('t')
    levels_str = (str(ds['level'][elev_idx_max].values) + ' to ' + str(ds['level'][elev_idx_min].values))
    output_ds['lapserate'] = (('time', 'latitude', 'longitude'), lr, 
                              {'long_name': 'lapse rate', 
                               'units': 'degC m-1',
                               'levels': levels_str})
    encoding = {'lapserate':{'_FillValue': False}}
    
    output_ds.to_netcdf(gcm_fp + output_fn, encoding=encoding)
   
     
#%%
if args.option_createtempstd == 1:
    ds_fp = '/Volumes/LaCie/ERA5/'
#    ds_fn = 't2m_hourly_1979_1989.nc'
#    ds_fn = 't2m_hourly_1990_1999.nc'
#    ds_fn = 't2m_hourly_2000_2009.nc'
#    ds_fn = 't2m_hourly_2010_2019.nc'
    ds_all_fn = 'ERA5_tempstd_monthly.nc'
    option_merge_files = 1
    
    # Merge completed files together
    if option_merge_files == 1:
        
        #%%
        tempstd_fns = []
        for i in os.listdir(ds_fp):
            if i.startswith('ERA5_tempstd_monthly') and i.endswith('.nc'):
                tempstd_fns.append(i)
        tempstd_fns = sorted(tempstd_fns)

        # Open datasets and combine
        for nfile, tempstd_fn in enumerate(tempstd_fns):
            print(tempstd_fn)
            ds = xr.open_dataset(ds_fp + tempstd_fn)
            # Merge datasets of stats into one output
            if nfile == 0:
                ds_all = ds
            else:
                ds_all = xr.concat([ds_all, ds], dim='time')
            
        # Export to netcdf
        encoding = {'t2m_std':{'_FillValue': False}}
        ds_all.to_netcdf(ds_fp + ds_all_fn, encoding=encoding)
        
    else:
    
        output_fn= 'ERA5_tempstd_monthly_' + ds_fn.split('_')[2] + '_' + ds_fn.split('_')[3]
        
        ds = xr.open_dataset(ds_fp + ds_fn)
    
    #    ds_subset = ds.t2m[0:30*24,:,:].values
    #    t2m_daily = np.moveaxis(np.moveaxis(ds_subset, 0, -1).reshape(-1,24).mean(axis=1)
    #                            .reshape(ds_subset.shape[1],ds_subset.shape[2],int(ds_subset.shape[0]/24)), -1, 0)
        
        # Calculate daily mean temperature
        ndays = int(ds.time.shape[0] / 24)
        t2m_daily = np.zeros((ndays, ds.latitude.shape[0], ds.longitude.shape[0]))
        for nday in np.arange(ndays):
            if nday%50 == 0:
                print(str(nday) + ' out of ' + str(ndays))
            ds_subset = ds.t2m[nday*24:(nday+1)*24, :, :].values
            t2m_daily[nday,:,:] = (
                    np.moveaxis(np.moveaxis(ds_subset, 0, -1).reshape(-1,24).mean(axis=1)
                                .reshape(ds_subset.shape[1],ds_subset.shape[2],int(ds_subset.shape[0]/24)), -1, 0))
    
        # Calculate monthly temperature standard deviation
        date = ds.time[::24].values
        date_month = [pd.Timestamp(date[x]).month for x in np.arange(date.shape[0])]
        date_year = [pd.Timestamp(date[x]).year for x in np.arange(date.shape[0])]
        
        date_yyyymm = [str(date_year[x]) + '-' + str(date_month[x]).zfill(2) for x in np.arange(date.shape[0])]
        date_yyyymm_unique = sorted(list(set(date_yyyymm)))
        
        t2m_monthly_std = np.zeros((len(date_yyyymm_unique), ds.latitude.shape[0], ds.longitude.shape[0]))
        date_monthly = []
        for count, yyyymm in enumerate(date_yyyymm_unique):
            if count%12 == 0:
                print(yyyymm)
            date_idx = np.where(np.array(date_yyyymm) == yyyymm)[0]
            date_monthly.append(date[date_idx[0]])
            t2m_monthly_std[count,:,:] = t2m_daily[date_idx,:,:].std(axis=0)
    
        # Export lapse rates with attibutes
        output_ds = ds.copy()
        output_ds = output_ds.drop('t2m')
        output_ds = output_ds.drop('time')
        output_ds['time'] = date_monthly
        output_ds['t2m_std'] = (('time', 'latitude', 'longitude'), t2m_monthly_std, 
                                 {'long_name': 'monthly 2m temperature standard deviation', 
                                  'units': 'K'})
        encoding = {'t2m_std':{'_FillValue': False}}
        output_ds.to_netcdf(ds_fp + output_fn, encoding=encoding)
    
        # Close dataset
        ds.close()
    

#%%
if args.option_frontalablation_cal == 1:
    region = [1]
    calving_data = pd.read_csv(pygem_prms.mcnabb_fp + '../alaska_gate_widths_flux.csv')
    
    glac_no = [x.split('-')[1] for x in list(calving_data.RGIId.values)]
    
    region_all = [int(x.split('.')[0]) for x in glac_no]
    rgi_glac_number_all = [x.split('.')[1] for x in glac_no]
    
    rgi_glac_number = []
    for n, reg in enumerate(region_all):
        if reg == region[0]:
            rgi_glac_number.append(rgi_glac_number_all[n])

    # Glacier RGI data
    main_glac_rgi = modelsetup.selectglaciersrgitable(rgi_regionsO1=region, rgi_regionsO2 = 'all',
                                                      rgi_glac_number=rgi_glac_number)
    # Glacier hypsometry [km**2], total area
    main_glac_hyps = modelsetup.import_Husstable(main_glac_rgi, pygem_prms.hyps_filepath,
                                                 pygem_prms.hyps_filedict, pygem_prms.hyps_colsdrop)
    # Ice thickness [m], average
    main_glac_icethickness = modelsetup.import_Husstable(main_glac_rgi, pygem_prms.thickness_filepath, 
                                                         pygem_prms.thickness_filedict, pygem_prms.thickness_colsdrop)
    main_glac_hyps[main_glac_icethickness == 0] = 0
    # Width [km], average
    main_glac_width = modelsetup.import_Husstable(main_glac_rgi, pygem_prms.width_filepath,
                                                  pygem_prms.width_filedict, pygem_prms.width_colsdrop)
    # Elevation bins
    elev_bins = main_glac_hyps.columns.values.astype(int)   
    # Select dates including future projections
    dates_table = modelsetup.datesmodelrun(startyear=2000, endyear=2005, spinupyears=0)
    
    # ===== LOAD CLIMATE DATA =====
    gcm = class_climate.GCM(name=pygem_prms.ref_gcm_name)
    # Air temperature [degC], Precipitation [m], Elevation [masl], Lapse rate [K m-1]
    gcm_temp, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(gcm.temp_fn, gcm.temp_vn, main_glac_rgi, dates_table)
    gcm_prec, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(gcm.prec_fn, gcm.prec_vn, main_glac_rgi, dates_table)
    gcm_elev = gcm.importGCMfxnearestneighbor_xarray(gcm.elev_fn, gcm.elev_vn, main_glac_rgi)
    # Air temperature standard deviation
    if pygem_prms.option_ablation != 2:
        gcm_tempstd = np.zeros(gcm_temp.shape)
    elif pygem_prms.ref_gcm_name in ['ERA5']:
        gcm_tempstd, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(gcm.tempstd_fn, gcm.tempstd_vn, 
                                                                        main_glac_rgi, dates_table)
    # Lapse rate [K m-1]
    gcm_lr, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(gcm.lr_fn, gcm.lr_vn, main_glac_rgi, dates_table)
    #%%
    
    for n, glac_str_wRGI in enumerate([main_glac_rgi['RGIId'].values[0]]):
#    for n, glac_str_wRGI in enumerate(main_glac_rgi['RGIId'].values):
        # Glacier string
        glacier_str = glac_str_wRGI.split('-')[1]
        print(glacier_str)
        # Glacier number
        glacno = int(glacier_str.split('.')[1])
        # RGI information
        glacier_rgi_table = main_glac_rgi.loc[main_glac_rgi.index.values[n], :]

        # Select subsets of data
        glacier_gcm_elev = gcm_elev[n]
        glacier_gcm_temp = gcm_temp[n,:]
        glacier_gcm_tempstd = gcm_tempstd[n,:]
        glacier_gcm_lrgcm = gcm_lr[n,:]
        glacier_gcm_lrglac = glacier_gcm_lrgcm.copy()
        glacier_gcm_prec = gcm_prec[n,:]
        glacier_area_t0 = main_glac_hyps.iloc[n,:].values.astype(float)
        icethickness_t0 = main_glac_icethickness.iloc[n,:].values.astype(float)
        width_t0 = main_glac_width.iloc[n,:].values.astype(float)
        glac_idx_t0 = glacier_area_t0.nonzero()[0]
        # Set model parameters
        modelparameters = [pygem_prms.lrgcm, pygem_prms.lrglac, pygem_prms.precfactor, pygem_prms.precgrad, pygem_prms.ddfsnow, pygem_prms.ddfice,
                           pygem_prms.tempsnow, pygem_prms.tempchange]
        frontalablation_k0 = pygem_prms.frontalablation_k0dict[int(glacier_str.split('.')[0])]
        
        (glac_bin_temp, glac_bin_prec, glac_bin_acc, glac_bin_refreeze, glac_bin_snowpack, glac_bin_melt,
         glac_bin_frontalablation, glac_bin_massbalclim, glac_bin_massbalclim_annual, glac_bin_area_annual,
         glac_bin_icethickness_annual, glac_bin_width_annual, glac_bin_surfacetype_annual,
         glac_wide_massbaltotal, glac_wide_runoff, glac_wide_snowline, glac_wide_snowpack,
         glac_wide_area_annual, glac_wide_volume_annual, glac_wide_ELA_annual, offglac_wide_prec, 
         offglac_wide_refreeze, offglac_wide_melt, offglac_wide_snowpack, offglac_wide_runoff) = (
            massbalance.runmassbalance(modelparameters, glacier_rgi_table, glacier_area_t0, 
                                       icethickness_t0, width_t0, elev_bins, glacier_gcm_temp, glacier_gcm_tempstd, 
                                       glacier_gcm_prec, glacier_gcm_elev, glacier_gcm_lrgcm, glacier_gcm_lrglac, 
                                       dates_table, option_areaconstant=0, frontalablation_k=None,
                                       debug=True))
        print('Add objective function and code ')
    
#%%    
if args.option_farinotti2019_input == 1:
    print("\nProcess the ice thickness and surface elevation data from Farinotti (2019) to produce area," + 
          "ice thickness, width, and length for each elevation bin\n")
    
    
#%%
if args.option_unh_climatedata == 1:
    climate_fp_hist = '/Users/davidrounce/Documents/Dave_Rounce/HiMAT/Climate_data/UNH_cmip5/CCSM4_RCP_85/'
    climate_fp_fut = climate_fp_hist + 'r1i1p1/'
    climate_fp_export = '/Users/davidrounce/Documents/Dave_Rounce/HiMAT/Climate_data/UNH_cmip5/'
    
    # Historical climate
    hist_fn_pr = []
    for i in os.listdir(climate_fp_hist):
        if i.startswith('pr') and i.endswith('.nc'):
            hist_fn_pr.append(climate_fp_hist + i)
    hist_fn_pr = sorted(hist_fn_pr)
    
    hist_fn_tas = []
    for i in os.listdir(climate_fp_hist):
        if i.startswith('tas') and i.endswith('.nc'):
            hist_fn_tas.append(climate_fp_hist + i)
    hist_fn_tas = sorted(hist_fn_tas)
    
    # Future climate
    fut_fn_pr = []
    for i in os.listdir(climate_fp_fut + 'pr/'):
        if (i.startswith('pr') and i.endswith('.nc') and 
            not i.endswith('_y.nc') and not i.endswith('_mc.nc') and not i.endswith('_yc.nc')):
            fut_fn_pr.append(climate_fp_fut + 'pr/' + i)
    fut_fn_pr = sorted(fut_fn_pr)
    
    fut_fn_tas = []
    for i in os.listdir(climate_fp_fut + 'tas/'):
        if (i.startswith('tas') and i.endswith('.nc') and 
            not i.endswith('_y.nc') and not i.endswith('_mc.nc') and not i.endswith('_yc.nc')):
            fut_fn_tas.append(climate_fp_fut + 'tas/' + i)
    fut_fn_tas = sorted(fut_fn_tas)
    
    # Merge lists
    tas_fn = hist_fn_tas.copy()
    tas_fn.extend(fut_fn_tas)
    pr_fn = hist_fn_pr.copy()
    pr_fn.extend(fut_fn_pr)
    
    # Example dataset
    ds_example = xr.open_dataset('/Users/davidrounce/Documents/Dave_Rounce/HiMAT/Climate_data/cmip5/' + 
                                 'rcp85_r1i1p1_monNG/tas_mon_CanESM2_rcp85_r1i1p1_native.nc')
    ds_example_pr = xr.open_dataset('/Users/davidrounce/Documents/Dave_Rounce/HiMAT/Climate_data/cmip5/' + 
                                 'rcp85_r1i1p1_monNG/pr_mon_CanESM2_rcp85_r1i1p1_native.nc')
    
    #%%
    # Merge datasets together
    ds_tas_all_fullfn = climate_fp_export + 'tas_mon_CCSM4_rcp85_r1i1p1_native.nc'
    
    data_tas = None
    if os.path.exists(ds_tas_all_fullfn) == False:
        ds_tas_all = None
        for fn in tas_fn[:301]:
#        for fn in tas_fn[:5]:
            print(fn)
            ds_tas = xr.open_dataset(fn)

            if data_tas is None:
                data_tas = ds_tas['tas'].values
            else:
                data_tas = np.concatenate((data_tas, ds_tas['tas'].values), axis=0)

    #%%
#    import collections
#    
#    lat_values = ds_tas.lat.values
#    lon_values = ds_tas.lon.values
#    time_values = pd.date_range('1960-01-01', '2260-12-15',freq='MS')
#    
#    # Create new dataset with correct format
#    output_coords_dict = collections.OrderedDict()
#    output_coords_dict['tas'] = collections.OrderedDict([('time', time_values), ('lat', lat_values), 
#                                                         ('lon', lon_values)])
#    output_attrs_dict = {
#            'time': {'standard_name': 'time'},
#            'lat': {'long_name': 'latitude',
#                    'units': 'degrees N'}, 
#            'lon': {'long_name': 'longitude',
#                    'units': 'degrees E'},
#            'tas': {'long_name': 'air_temperature',
#                     'units': 'K'}}
#    
#    output_ds_all = None
#    encoding = {}
#    for vn in output_coords_dict.keys():
#        empty_holder = np.zeros([len(output_coords_dict[vn][i]) for i in list(output_coords_dict[vn].keys())])
#        output_ds = xr.Dataset({vn: (list(output_coords_dict[vn].keys()), empty_holder)},
#                               coords=output_coords_dict[vn])
#        # Merge datasets of stats into one output
#        if output_ds_all is None:
#            output_ds_all = output_ds
#        else:
#            output_ds_all = xr.merge((output_ds_all, output_ds))
#            
#    # Add attributes
#    for vn in output_attrs_dict.keys():
#        try:
#            output_ds_all[vn].attrs = output_attrs_dict[vn]
#        except:
#            pass
#        # Encoding (specify _FillValue, offsets, etc.)
#        encoding[vn] = {'_FillValue': False,
##                        'zlib':True,
##                        'complevel':9
#                        }
#    
#    output_ds_all['lon'].values = lon_values
#    output_ds_all['lat'].values = lat_values
#    output_ds_all['time'].values = time_values
#    output_ds_all['tas'].values = data_tas
#    
#    output_ds_all.attrs = {
#                        'source': 'University of New Hampshire - Alex Prusevich',
#                        'history': 'revised by David Rounce (drounce@alaska.edu) for PyGEM format'}
#    
#    # Export
#    output_ds_all.to_netcdf(ds_tas_all_fullfn)
    
    
    #%%
    # Merge datasets together
    ds_pr_all_fullfn = climate_fp_export + 'pr_mon_CCSM4_rcp85_r1i1p1_native.nc'
    
    data_pr = None
    if os.path.exists(ds_pr_all_fullfn) == False:
        ds_pr_all = None
        for fn in pr_fn[:301]:
            print(fn)
            ds_pr = xr.open_dataset(fn)

            if data_pr is None:
                data_pr = ds_pr['pr'].values
            else:
                data_pr = np.concatenate((data_pr, ds_pr['pr'].values), axis=0)
                
    #%%
    print(ds_pr['pr'].units)
         
    #%%
    import collections
    
    lat_values = ds_pr.lat.values
    lon_values = ds_pr.lon.values
    time_values = pd.date_range('1960-01-01', '2260-12-15',freq='MS')
    
    # Create new dataset with correct format
    output_coords_dict = collections.OrderedDict()
    output_coords_dict['pr'] = collections.OrderedDict([('time', time_values), ('lat', lat_values), 
                                                         ('lon', lon_values)])
    output_attrs_dict = {
            'time': {'standard_name': 'time'},
            'lat': {'long_name': 'latitude',
                    'units': 'degrees N'}, 
            'lon': {'long_name': 'longitude',
                    'units': 'degrees E'},
            'pr': {'long_name': 'precipitation',
                     'units': ds_pr['pr'].units}}
    
    output_ds_all = None
    encoding = {}
    for vn in output_coords_dict.keys():
        empty_holder = np.zeros([len(output_coords_dict[vn][i]) for i in list(output_coords_dict[vn].keys())])
        output_ds = xr.Dataset({vn: (list(output_coords_dict[vn].keys()), empty_holder)},
                               coords=output_coords_dict[vn])
        # Merge datasets of stats into one output
        if output_ds_all is None:
            output_ds_all = output_ds
        else:
            output_ds_all = xr.merge((output_ds_all, output_ds))
    
    # Add attributes
    for vn in output_attrs_dict.keys():
        try:
            output_ds_all[vn].attrs = output_attrs_dict[vn]
        except:
            pass
        # Encoding (specify _FillValue, offsets, etc.)
        encoding[vn] = {'_FillValue': False,
#                        'zlib':True,
#                        'complevel':9
                        }
    
    output_ds_all['lon'].values = lon_values
    output_ds_all['lat'].values = lat_values
    output_ds_all['time'].values = time_values
    output_ds_all['pr'].values = data_pr
    
    output_ds_all.attrs = {
                        'source': 'University of New Hampshire - Alex Prusevich',
                        'history': 'revised by David Rounce (drounce@alaska.edu) for PyGEM format'}
    
    
    # Export
    output_ds_all.to_netcdf(ds_pr_all_fullfn)
    
    
#%%
if args.option_regional_meltfactors == 1:
    hd_fp = ('/Users/davidrounce/Documents/Dave_Rounce/DebrisGlaciers_WG/Melt_Intercomparison/output/mb_bins/csv/' + 
             '_wdebris_hdts/')
    hd_extrap_fp = ('/Users/davidrounce/Documents/Dave_Rounce/DebrisGlaciers_WG/Melt_Intercomparison/output/mb_bins/' + 
                    'csv/_wdebris_hdts_extrap/')
    
    main_glac_rgi = modelsetup.selectglaciersrgitable(rgi_regionsO1=pygem_prms.rgi_regionsO1, 
                                                      rgi_regionsO2=pygem_prms.rgi_regionsO2,
                                                      rgi_glac_number=pygem_prms.rgi_glac_number)
    # Glacier hypsometry [km**2], total area
    main_glac_hyps = modelsetup.import_Husstable(main_glac_rgi, pygem_prms.hyps_filepath, pygem_prms.hyps_filedict,
                                                 pygem_prms.hyps_colsdrop)
    # Ice thickness [m], average
    main_glac_icethickness = modelsetup.import_Husstable(main_glac_rgi, pygem_prms.thickness_filepath,
                                                         pygem_prms.thickness_filedict, pygem_prms.thickness_colsdrop)
    main_glac_icethickness[main_glac_icethickness < 0] = 0
    main_glac_hyps[main_glac_icethickness == 0] = 0
    # Width [km], average
    main_glac_width = modelsetup.import_Husstable(main_glac_rgi, pygem_prms.width_filepath, pygem_prms.width_filedict,
                                                  pygem_prms.width_colsdrop)
    elev_bins = main_glac_hyps.columns.values.astype(int)
    
    # Load debris thickness filenames
    # Glaciers optimized
    glac_hd_fullfns = []
    for i in os.listdir(hd_fp):
        if i.endswith('hd_hdts.csv'):
            region = int(i.split('.')[0])
            if region in pygem_prms.rgi_regionsO1:           
                glac_hd_fullfns.append(hd_fp + i)
        
    # Glaciers extrapolated
    for i in os.listdir(hd_extrap_fp):
        if i.endswith('hdts_extrap.csv'):
            region = int(i.split('.')[0])
            if region in pygem_prms.rgi_regionsO1:          
                glac_hd_fullfns.append(hd_extrap_fp + i)
    glac_hd_fullfns = sorted(glac_hd_fullfns)
    
    print('for each glacier, check for any z_offse due to the datasets?')
    
    
    
    #%%
    

#            if ds_tas_all is None:
#                ds_tas_all = ds_tas
#            else:
#                ds_tas_all = xr.concat((ds_tas_all, ds_tas), dim='time', coords='all')
#        ds_tas_all.to_netcdf(ds_tas_all_fullfn)
#
#    ds_pr_all_fullfn = climate_fp_export + 'pr_mon_CCSM4_rcp85_r1i1p1_native.nc'
#    if os.path.exists(ds_pr_all_fullfn) == False:
#        ds_pr_all = None
#        for fn in pr_fn[:301]:
#            print(fn)
#            ds_pr = xr.open_dataset(fn)
#            
#            if ds_pr_all is None:
#                ds_pr_all = ds_pr
#            else:
#                ds_pr_all = xr.concat((ds_pr_all, ds_pr), dim='time', coords='all')
#        ds_pr_all.to_netcdf(ds_pr_all_fullfn)
#    
#    startyear = 1960
#    endyear = 2260
#    subtractyear = 100
#    time_values = pd.date_range(str(int(startyear-subtractyear)) + '-01-01',
#                                str(int(endyear-subtractyear)) + '-12-15',freq='MS')
##    print(time_values)
##    ds_tas_all = xr.open_dataset('/Users/davidrounce/Documents/Dave_Rounce/HiMAT/Climate_data/UNH_cmip5/' + 
##                                 'rcp85_r1i1p1_monNG/tas_mon_CCSM4_rcp85_r1i1p1_native.nc')
##    ds_tas_all.time.values = time_values
##    ds_tas_all.to_netcdf('/Users/davidrounce/Documents/Dave_Rounce/HiMAT/Climate_data/UNH_cmip5/' + 
##                                 'rcp85_r1i1p1_monNG/tas_mon_CCSM4_rcp85_r1i1p1_native-v3.nc')
#    print(ds_tas_all.time)
#    print(ds_tas_all['tas'][9:3608,254,82])
#    
##    ds_pr_all = xr.open_dataset('/Users/davidrounce/Documents/Dave_Rounce/HiMAT/Climate_data/UNH_cmip5/' + 
##                                 'rcp85_r1i1p1_monNG/pr_mon_CCSM4_rcp85_r1i1p1_native.nc')
##    ds_pr_all.time.values = time_values
##    ds_pr_all.to_netcdf('/Users/davidrounce/Documents/Dave_Rounce/HiMAT/Climate_data/UNH_cmip5/' + 
##                                 'rcp85_r1i1p1_monNG/pr_mon_CCSM4_rcp85_r1i1p1_native-v3.nc')
##    print(ds_pr_all.time)


#%% Write csv file from model results
# Create csv such that not importing the air temperature each time (takes 90 seconds for 13,119 glaciers)
#output_csvfullfilename = pygem_prms.main_directory + '/../Output/ERAInterim_elev_15_SouthAsiaEast.csv'
#climate.createcsv_GCMvarnearestneighbor(pygem_prms.gcm_prec_filename, pygem_prms.gcm_prec_varname, dates_table, main_glac_rgi, 
#                                        output_csvfullfilename)
#np.savetxt(output_csvfullfilename, main_glac_gcmelev, delimiter=",") 
    

#%% NEAREST NEIGHBOR CALIBRATION PARAMETERS
## Load csv
#ds = pd.read_csv(pygem_prms.main_directory + '/../Output/calibration_R15_20180403_Opt02solutionspaceexpanding.csv', 
#                 index_col='GlacNo')
## Select data of interest
#data = ds[['CenLon', 'CenLat', 'lrgcm', 'lrglac', 'precfactor', 'precgrad', 'ddfsnow', 'ddfice', 'tempsnow', 
#           'tempchange']].copy()
## Drop nan data to retain only glaciers with calibrated parameters
#data_cal = data.dropna()
#A = data_cal.mean(0)
## Select latitude and longitude of calibrated parameters for distance estimate
#data_cal_lonlat = data_cal.iloc[:,0:2].values
## Loop through each glacier and select the parameters based on the nearest neighbor
#for glac in range(data.shape[0]):
#    # Avoid applying this to any glaciers that already were optimized
#    if data.iloc[glac, :].isnull().values.any() == True:
#        # Select the latitude and longitude of the glacier's center
#        glac_lonlat = data.iloc[glac,0:2].values
#        # Set point to be compatible with cdist function (from scipy)
#        pt = [[glac_lonlat[0],glac_lonlat[1]]]
#        # scipy function to calculate distance
#        distances = cdist(pt, data_cal_lonlat)
#        # Find minimum index (could be more than one)
#        idx_min = np.where(distances == distances.min())[1]
#        # Set new parameters
#        data.iloc[glac,2:] = data_cal.iloc[idx_min,2:].values.mean(0)
#        #  use mean in case multiple points are equidistant from the glacier
## Remove latitude and longitude to create csv file
#parameters_export = data.iloc[:,2:]
## Export csv file
#parameters_export.to_csv(pygem_prms.main_directory + '/../Calibration_datasets/calparams_R15_20180403_nearest.csv', 
#                         index=False)    