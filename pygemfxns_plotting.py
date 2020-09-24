# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 09:04:46 2017

@author: David Rounce

pygemfxns_plotting.py produces figures of simulation results
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
from scipy import stats
from scipy.ndimage import uniform_filter
import cartopy
#import geopandas
import xarray as xr
from osgeo import gdal, ogr, osr
import pickle
# Local Libraries
import pygem.pygem_input as pygem_prms
import pygemfxns_modelsetup as modelsetup
import pygemfxns_massbalance as massbalance
import pygemfxns_gcmbiasadj as gcmbiasadj
import class_mbdata
import class_climate
#import run_simulation


# Script options
option_plot_cmip5_normalizedchange = 1
option_plot_cmip5_runoffcomponents = 0
option_plot_cmip5_map = 0
option_output_tables = 0
option_subset_GRACE = 0
option_plot_modelparam = 0
option_plot_era_normalizedchange = 1
option_compare_GCMwCal = 0
option_plot_mcmc_errors = 0
option_plot_maxloss_issues = 0

option_plot_individual_glaciers = 0
option_plot_degrees = 0
option_plot_pies = 0
option_plot_individual_gcms = 0


#%% ===== Input data =====
netcdf_fp_cmip5 = '/Users/davidrounce/Documents/Dave_Rounce/HiMAT/Output/simulations/spc/'
netcdf_fp_era = '/Users/davidrounce/Documents/Dave_Rounce/HiMAT/Output/simulations/ERA-Interim/ERA-Interim_1980_2017_nochg'
#mcmc_fp = '/Users/davidrounce/Documents/Dave_Rounce/HiMAT/Output/cal_opt2_allglac_1ch_tn_20190108/'
#mcmc_fp = '/Users/davidrounce/Documents/Dave_Rounce/HiMAT/Output/cal_opt2_spc_20190222_adjp10/'
mcmc_fp = '/Users/davidrounce/Documents/Dave_Rounce/HiMAT/Output/cal_opt2_spc_20190308_adjp12/cal_opt2/'
figure_fp = '/Users/davidrounce/Documents/Dave_Rounce/HiMAT/Output/figures/cmip5/'
csv_fp = '/Users/davidrounce/Documents/Dave_Rounce/HiMAT/Output/csv/cmip5/'
cal_fp = '/Users/davidrounce/Documents/Dave_Rounce/HiMAT/Output/cal_opt2_spc_20190308_adjp12/cal_opt2/'

# Regions
rgi_regions = [13, 14, 15]
#rgi_regions = [13]

# Shapefiles
rgiO1_shp_fn = '/Users/davidrounce/Documents/Dave_Rounce/HiMAT/RGI/rgi60/00_rgi60_regions/00_rgi60_O1Regions.shp'
watershed_shp_fn = '/Users/davidrounce/Documents/Dave_Rounce/HiMAT/qgis_himat/HMA_basins_20181018_4plot.shp'
kaab_shp_fn = '/Users/davidrounce/Documents/Dave_Rounce/HiMAT/qgis_himat/kaab2015_regions.shp'
srtm_fn = '/Users/davidrounce/Documents/Dave_Rounce/HiMAT/qgis_himat/SRTM_HMA.tif'
srtm_contour_fn = '/Users/davidrounce/Documents/Dave_Rounce/HiMAT/qgis_himat/SRTM_HMA_countours_2km_gt3000m_smooth.shp'
rgi_glac_shp_fn = '/Users/davidrounce/Documents/Dave_Rounce/HiMAT/qgis_himat/rgi60_HMA.shp'
#kaab_dict_fn = '/Users/davidrounce/Documents/Dave_Rounce/HiMAT/qgis_himat/rgi60_HMA_w_watersheds_kaab.csv'
#kaab_csv = pd.read_csv(kaab_dict_fn)
#kaab_dict = dict(zip(kaab_csv.RGIId, kaab_csv.kaab))
# GCMs and RCP scenarios
#gcm_names = ['CanESM2', 'CCSM4', 'CNRM-CM5', 'CSIRO-Mk3-6-0', 'GFDL-CM3', 'GFDL-ESM2M', 'GISS-E2-R', 'IPSL-CM5A-LR', 
#             'IPSL-CM5A-MR', 'MIROC5', 'MRI-CGCM3', 'NorESM1-M']
gcm_names = ['CanESM2']
#gcm_names = ['CanESM2', 'CCSM4', 'CNRM-CM5', 'CSIRO-Mk3-6-0',  'GFDL-CM3', 'GFDL-ESM2M', 'GISS-E2-R', 'IPSL-CM5A-LR', 
#             'MPI-ESM-LR', 'NorESM1-M']
rcps = ['rcp26', 'rcp45', 'rcp85']
#rcps = ['rcp26']

# Grouping
grouping = 'all'
#grouping = 'rgi_region'
#grouping = 'watershed'
#grouping = 'kaab'

# Variable name
vn = 'mass_change'
#vn = 'volume_norm'
#vn = 'peakwater'

# Group dictionaries
watershed_dict_fn = '/Users/davidrounce/Documents/Dave_Rounce/HiMAT/qgis_himat/rgi60_HMA_dict_watershed.csv'
watershed_csv = pd.read_csv(watershed_dict_fn)
watershed_dict = dict(zip(watershed_csv.RGIId, watershed_csv.watershed))
kaab_dict_fn = '/Users/davidrounce/Documents/Dave_Rounce/HiMAT/qgis_himat/rgi60_HMA_dict_kaab.csv'
kaab_csv = pd.read_csv(kaab_dict_fn)
kaab_dict = dict(zip(kaab_csv.RGIId, kaab_csv.kaab_name))

# GRACE mascons
mascon_fp = pygem_prms.main_directory + '/../GRACE/GSFC.glb.200301_201607_v02.4/'
mascon_fn = 'mascon.txt'
mascon_cns = ['CenLat', 'CenLon', 'LatWidth', 'LonWidth', 'Area_arcdeg', 'Area_km2', 'location', 'basin', 
              'elevation_flag']
mascon_df = pd.read_csv(mascon_fp + mascon_fn, header=None, names=mascon_cns, skiprows=14, 
                        delim_whitespace=True)
mascon_df = mascon_df.sort_values(by=['CenLat', 'CenLon'])
mascon_df.reset_index(drop=True, inplace=True)

degree_size = 0.25
peakwater_Nyears = 10

# Plot label dictionaries
title_dict = {'Amu_Darya': 'Amu Darya',
              'Brahmaputra': 'Brahmaputra',
              'Ganges': 'Ganges',
              'Ili': 'Ili',
              'Indus': 'Indus',
              'Inner_Tibetan_Plateau': 'Inner TP',
              'Inner_Tibetan_Plateau_extended': 'Inner TP ext',
              'Irrawaddy': 'Irrawaddy',
              'Mekong': 'Mekong',
              'Salween': 'Salween',
              'Syr_Darya': 'Syr Darya',
              'Tarim': 'Tarim',
              'Yangtze': 'Yangtze',
              'inner_TP': 'Inner TP',
              'Karakoram': 'Karakoram',
              'Yigong': 'Yigong',
              'Yellow': 'Yellow',
              'Bhutan': 'Bhutan',
              'Everest': 'Everest',
              'West Nepal': 'West Nepal',
              'Spiti Lahaul': 'Spiti Lahaul',
              'tien_shan': 'Tien Shan',
              'Pamir': 'Pamir',
              'pamir_alai': 'Pamir Alai',
              'Kunlun': 'Kunlun',
              'Hindu Kush': 'Hindu Kush',
              13: 'Central Asia',
              14: 'South Asia West',
              15: 'South Asia East',
              'all': 'HMA'
              }
title_location = {'Syr_Darya': [68, 46.1],
                  'Ili': [83.6, 45.5],
                  'Amu_Darya': [64.6, 36.9],
                  'Tarim': [83.0, 39.2],
                  'Inner_Tibetan_Plateau_extended': [100, 40],
                  'Indus': [70.7, 31.9],
                  'Inner_Tibetan_Plateau': [85, 32.4],
                  'Yangtze': [106.0, 29.8],
                  'Ganges': [81.3, 26.6],
                  'Brahmaputra': [92.0, 26],
                  'Irrawaddy': [96.2, 23.8],
                  'Salween': [98.5, 20.8],
                  'Mekong': [103.8, 17.5],
                  'Yellow': [106.0, 36],
                  13: [83,39],
                  14: [70.8, 30],
                  15: [81,26.8],
                  'inner_TP': [89, 33.5],
                  'Karakoram': [68.7, 33.5],
                  'Yigong': [97.5, 26.2],
                  'Bhutan': [92.1, 26],
                  'Everest': [85, 26.3],
                  'West Nepal': [76.5, 28],
                  'Spiti Lahaul': [72, 31.9],
                  'tien_shan': [80, 42],
                  'Pamir': [67.3, 36.5],
                  'pamir_alai': [65.2, 40.2],
                  'Kunlun': [79, 37.5],
                  'Hindu Kush': [65.3, 35]
                  }
vn_dict = {'volume_glac_annual': 'Normalized Volume [-]',
           'volume_norm': 'Normalized Volume Remaining [-]',
           'runoff_glac_annual': 'Normalized Runoff [-]',
           'peakwater': 'Peak Water [yr]',
           'temp_glac_annual': 'Temperature [$^\circ$C]',
           'prec_glac_annual': 'Precipitation [m]',
           'precfactor': 'Precipitation Factor [-]',
           'tempchange': 'Temperature bias [$^\circ$C]', 
           'ddfsnow': 'DDFsnow [mm w.e. d$^{-1}$ $^\circ$C$^{-1}$]'}
rcp_dict = {'rcp26': '2.6',
            'rcp45': '4.5',
            'rcp60': '6.0',
            'rcp85': '8.5'}

# Colors list
colors_rgb = [(0.00, 0.57, 0.57), (0.71, 0.43, 1.00), (0.86, 0.82, 0.00), (0.00, 0.29, 0.29), (0.00, 0.43, 0.86), 
              (0.57, 0.29, 0.00), (1.00, 0.43, 0.71), (0.43, 0.71, 1.00), (0.14, 1.00, 0.14), (1.00, 0.71, 0.47), 
              (0.29, 0.00, 0.57), (0.57, 0.00, 0.00), (0.71, 0.47, 1.00), (1.00, 1.00, 0.47)]
gcm_colordict = dict(zip(gcm_names, colors_rgb[0:len(gcm_names)]))
rcp_colordict = {'rcp26':'b', 'rcp45':'k', 'rcp60':'m', 'rcp85':'r'}
rcp_styledict = {'rcp26':':', 'rcp45':'--', 'rcp85':'-.'}

east = 60
west = 110
south = 15
north = 50
xtick = 5
ytick = 5
xlabel = 'Longitude [$^\circ$]'
ylabel = 'Latitude [$^\circ$]'


#%% FUNCTIONS
def select_groups(grouping, main_glac_rgi_all):
    """
    Select groups based on grouping
    """
    if grouping == 'rgi_region':
        groups = main_glac_rgi_all.O1Region.unique().tolist()
        group_cn = 'O1Region'
    elif grouping == 'watershed':
        groups = main_glac_rgi_all.watershed.unique().tolist()
        group_cn = 'watershed'
    elif grouping == 'kaab':
        groups = main_glac_rgi_all.kaab.unique().tolist()
        group_cn = 'kaab'
        groups = [x for x in groups if str(x) != 'nan']  
    elif grouping == 'degree':
        groups = main_glac_rgi_all.deg_id.unique().tolist()
        group_cn = 'deg_id'
    elif grouping == 'mascon':
        groups = main_glac_rgi_all.mascon_idx.unique().tolist()
        groups = [int(x) for x in groups]
        group_cn = 'mascon_idx'
    else:
        groups = ['all']
        group_cn = 'all_group'
    try:
        groups = sorted(groups, key=str.lower)
    except:
        groups = sorted(groups)
    return groups, group_cn

def partition_multimodel_groups(gcm_names, grouping, vn, main_glac_rgi_all, rcp=None):
    """Partition multimodel data by each group for all GCMs for a given variable
    
    Parameters
    ----------
    gcm_names : list
        list of GCM names
    grouping : str
        name of grouping to use
    vn : str
        variable name
    main_glac_rgi_all : pd.DataFrame
        glacier table
    rcp : str
        rcp name
        
    Output
    ------
    time_values : np.array
        time values that accompany the multimodel data
    ds_group : list of lists
        dataset containing the multimodel data for a given variable for all the GCMs
    ds_glac : np.array
        dataset containing the variable of interest for each gcm and glacier
    """
    # Groups
    groups, group_cn = select_groups(grouping, main_glac_rgi_all)
    
    # variable name
    if vn == 'volume_norm' or vn == 'mass_change':
        vn_adj = 'volume_glac_annual'
    elif vn == 'peakwater':
        vn_adj = 'runoff_glac_annual'
    else:
        vn_adj = vn
    
    ds_group = [[] for group in groups]
    for ngcm, gcm_name in enumerate(gcm_names):
        for region in rgi_regions:
                   
            # Load datasets
            if gcm_name == 'ERA-Interim':
                netcdf_fp = netcdf_fp_era
                ds_fn = 'R' + str(region) + '_ERA-Interim_c2_ba1_100sets_1980_2017.nc'
            else:
                netcdf_fp = netcdf_fp_cmip5 + vn_adj + '/'
                ds_fn = ('R' + str(region) + '_' + gcm_name + '_' + rcp + '_c2_ba' + str(pygem_prms.option_bias_adjustment) +
                         '_100sets_2000_2100--' + vn_adj + '.nc')
            
            # Bypass GCMs that are missing a rcp scenario
            try:
                ds = xr.open_dataset(netcdf_fp + ds_fn)
            except:
                continue
            # Extract time variable
            if 'annual' in vn_adj:
                try:
                    time_values = ds[vn_adj].coords['year_plus1'].values
                except:
                    time_values = ds[vn_adj].coords['year'].values
            elif 'monthly' in vn_adj:
                time_values = ds[vn_adj].coords['time'].values
                
            # Merge datasets
            if region == rgi_regions[0]:
                vn_glac_all = ds[vn_adj].values[:,:,0]
                vn_glac_std_all = ds[vn_adj].values[:,:,1]
            else:
                vn_glac_all = np.concatenate((vn_glac_all, ds[vn_adj].values[:,:,0]), axis=0)
                vn_glac_std_all = np.concatenate((vn_glac_std_all, ds[vn_adj].values[:,:,1]), axis=0)
            
            try:
                ds.close()
            except:
                continue
            
        if ngcm == 0:
            ds_glac = vn_glac_all[np.newaxis,:,:]
        else:
            ds_glac = np.concatenate((ds_glac, vn_glac_all[np.newaxis,:,:]), axis=0)
        # Cycle through groups
        for ngroup, group in enumerate(groups):
            # Select subset of data
            main_glac_rgi = main_glac_rgi_all.loc[main_glac_rgi_all[group_cn] == group]                        
            vn_glac = vn_glac_all[main_glac_rgi.index.values.tolist(),:]
#            vn_glac_std = vn_glac_std_all[main_glac_rgi.index.values.tolist(),:]
#            vn_glac_var = vn_glac_std **2                
            # Regional sum
            vn_reg = vn_glac.sum(axis=0)                
            # Record data for multi-model stats
            if ngcm == 0:
                ds_group[ngroup] = [group, vn_reg]
            else:
                ds_group[ngroup][1] = np.vstack((ds_group[ngroup][1], vn_reg))
                
    return groups, time_values, ds_group, ds_glac

def partition_era_groups(grouping, vn, main_glac_rgi_all):
    """Partition multimodel data by each group for all GCMs for a given variable
    
    Parameters
    ----------
    grouping : str
        name of grouping to use
    vn : str
        variable name
    main_glac_rgi_all : pd.DataFrame
        glacier table
        
    Output
    ------
    time_values : np.array
        time values that accompany the multimodel data
    ds_group : list of lists
        dataset containing the multimodel data for a given variable for all the GCMs
    ds_glac : np.array
        dataset containing the variable of interest for each gcm and glacier
    """
    # Groups
    groups, group_cn = select_groups(grouping, main_glac_rgi_all)
    
    # variable name
    if vn == 'volume_norm' or vn == 'mass_change':
        vn_adj = 'volume_glac_annual'
    elif vn == 'peakwater':
        vn_adj = 'runoff_glac_annual'
    else:
        vn_adj = vn
    
    ds_group = [[] for group in groups]
    for region in rgi_regions:
        # Load datasets
        ds_fn = 'R' + str(region) + '_ERA-Interim_c2_ba1_100sets_1980_2017.nc'
        ds = xr.open_dataset(netcdf_fp_era + ds_fn)
        # Extract time variable
        if 'annual' in vn_adj:
            try:
                time_values = ds[vn_adj].coords['year_plus1'].values
            except:
                time_values = ds[vn_adj].coords['year'].values
        elif 'monthly' in vn_adj:
            time_values = ds[vn_adj].coords['time'].values
            
        # Merge datasets
        if region == rgi_regions[0]:
            vn_glac_all = ds[vn_adj].values[:,:,0]
            vn_glac_std_all = ds[vn_adj].values[:,:,1]
        else:
            vn_glac_all = np.concatenate((vn_glac_all, ds[vn_adj].values[:,:,0]), axis=0)
            vn_glac_std_all = np.concatenate((vn_glac_std_all, ds[vn_adj].values[:,:,1]), axis=0)
        
        # Close dataset
        ds.close()

    ds_glac = [vn_glac_all, vn_glac_std_all]
    
    # Cycle through groups
    for ngroup, group in enumerate(groups):
        # Select subset of data
        main_glac_rgi = main_glac_rgi_all.loc[main_glac_rgi_all[group_cn] == group]                        
        vn_glac = vn_glac_all[main_glac_rgi.index.values.tolist(),:]
        vn_glac_std = vn_glac_std_all[main_glac_rgi.index.values.tolist(),:]
        vn_glac_var = vn_glac_std **2                
        
        # Regional mean, standard deviation, and variance
        #  mean: E(X+Y) = E(X) + E(Y)
        #  var: Var(X+Y) = Var(X) + Var(Y) + 2*Cov(X,Y)
        #    assuming X and Y are indepdent, then Cov(X,Y)=0, so Var(X+Y) = Var(X) + Var(Y)
        #  std: std(X+Y) = (Var(X+Y))**0.5
        # Regional sum
        vn_reg = vn_glac.sum(axis=0)
        vn_reg_var = vn_glac_var.sum(axis=0)
#        vn_reg_std = vn_glac_var**0.5
        
        # Record data for multi-model stats
        ds_group[ngroup] = [group, vn_reg, vn_reg_var]
                
    return groups, time_values, ds_group, ds_glac


def partition_modelparams_groups(grouping, vn, main_glac_rgi_all):
    """Partition model parameters by each group
    
    Parameters
    ----------
    grouping : str
        name of grouping to use
    vn : str
        variable name
    main_glac_rgi_all : pd.DataFrame
        glacier table
        
    Output
    ------
    groups : list
        list of group names
    ds_group : list of lists
        dataset containing the multimodel data for a given variable for all the GCMs
    """
    # Groups
    groups, group_cn = select_groups(grouping, main_glac_rgi_all)
    
    ds_group = [[] for group in groups]
    
    # Cycle through groups
    for ngroup, group in enumerate(groups):
        # Select subset of data
        main_glac_rgi = main_glac_rgi_all.loc[main_glac_rgi_all[group_cn] == group]                        
        vn_glac = main_glac_rgi_all[vn].values[main_glac_rgi.index.values.tolist()]             
        # Regional sum
        vn_reg = vn_glac.mean(axis=0)                
        
        # Record data for each group
        ds_group[ngroup] = [group, vn_reg]
                
    return groups, ds_group

 
def vn_multimodel_mean_processed(vn, ds, idx, time_values, every_glacier=0):
    """
    Calculate multi-model mean for a given variable of interest
    
    Parameters
    ----------
    vn : str
        variable/parameter name
    ds : list
        dataset containing groups
    group_idx : int
        group index
    time_values : np.array
        array of years
    every_glacier : int
        switch to work with groups or work with concatenated dataframe
    
    Output
    ------
    
    """
    # Multi-model mean
    if every_glacier == 0:
        vn_multimodel_mean = ds[idx][1].mean(axis=0)
    else:
        vn_multimodel_mean = ds[:,idx,:].mean(axis=0)
    
    # Normalized volume based on initial volume
    if vn == 'volume_norm':
        if vn_multimodel_mean[0] > 0:
            output_multimodel_mean = vn_multimodel_mean / vn_multimodel_mean[0]
        else:
            output_multimodel_mean = np.zeros(vn_multimodel_mean.shape)
    # Peak water based on 10-yr running average
    elif vn == 'peakwater':
        vn_runningmean = uniform_filter(vn_multimodel_mean, peakwater_Nyears)
        output_multimodel_mean = time_values[np.where(vn_runningmean == vn_runningmean.max())[-1][0]]
    return output_multimodel_mean


def peakwater(runoff, time_values, nyears):
    """Compute peak water based on the running mean of N years
    
    Parameters
    ----------
    runoff : np.array
        one-dimensional array of runoff for each timestep
    time_values : np.array
        time associated with each timestep
    nyears : int
        number of years to compute running mean used to smooth peakwater variations
        
    Output
    ------
    peakwater_yr : int
        peakwater year
    peakwater_chg : float
        percent change of peak water compared to first timestep (running means used)
    runoff_chg : float
        percent change in runoff at the last timestep compared to the first timestep (running means used)
    """
    runningmean = uniform_filter(runoff, size=(nyears))
    peakwater_idx = np.where(runningmean == runningmean.max())[-1][0]
    peakwater_yr = time_values[peakwater_idx]
    peakwater_chg = (runningmean[peakwater_idx] - runningmean[0]) / runningmean[0] * 100
    runoff_chg = (runningmean[-1] - runningmean[0]) / runningmean[0] * 100
    return peakwater_yr, peakwater_chg, runoff_chg


def size_thresholds(variable, cutoffs, sizes):
    """Loop through size thresholds for a given variable to plot
    
    Parameters
    ----------
    variable : np.array
        data associated with glacier characteristic
    cutoffs : list
        values used as minimums for thresholds 
        (ex. 100 would give you greater than 100)
    sizes : list
        size values for the plot
        
    Output
    ------
    output : np.array
        plot size for each glacier
    """
    output = np.zeros(variable.shape)
    for i, cutoff in enumerate(cutoffs):
        output[(variable>cutoff) & (output==0)] = sizes[i]
    output[output==0] = 2
    return output


def select_region_climatedata(gcm_name, rcp, main_glac_rgi):
    """
    Get the regional temperature and precipitation for a given dataset.
    
    Extracts all nearest neighbor temperature and precipitation data for a given set of glaciers.  The mean temperature
    and precipitation of the group of glaciers is returned.  If two glaciers have the same temp/prec data, that data
    is only used once in the mean calculations.  Additionally, one would not expect for different GCMs to be similar
    because they all have different resolutions, so this mean calculations will have different numbers of pixels.
    
    Parameters
    ----------
    gcm_name : str
        GCM name
    rcp : str
        rcp scenario (ex. rcp26)
    main_glac_rgi : pd.DataFrame
        glacier dataset used to select the nearest neighbor climate data
    """
    # Date tables    
    print('select_region_climatedata fxn dates supplied manually')
    dates_table_ref = modelsetup.datesmodelrun(startyear=2000, endyear=2100, spinupyears=0, 
                                               option_wateryear=1)
    dates_table = modelsetup.datesmodelrun(startyear=2000, endyear=2100, spinupyears=0,
                                           option_wateryear=1)
    # Load gcm lat/lons
    gcm = class_climate.GCM(name=gcm_name, rcp_scenario=rcp)
    # Select lat/lon from GCM
    ds_elev = xr.open_dataset(gcm.fx_fp + gcm.elev_fn)
    gcm_lat_values_all = ds_elev.lat.values
    gcm_lon_values_all = ds_elev.lon.values
    ds_elev.close()
    # Lat/lon dictionary to convert
    gcm_lat_dict = dict(zip(range(gcm_lat_values_all.shape[0]), list(gcm_lat_values_all)))
    gcm_lon_dict = dict(zip(range(gcm_lon_values_all.shape[0]), list(gcm_lon_values_all)))
    
    # Find nearest neighbors for glaciers that have pixles
    latlon_nearidx = pd.DataFrame(np.zeros((main_glac_rgi.shape[0],2)), columns=['CenLat','CenLon'])
    latlon_nearidx.iloc[:,0] = (np.abs(main_glac_rgi.CenLat.values[:,np.newaxis] - gcm_lat_values_all).argmin(axis=1))
    latlon_nearidx.iloc[:,1] = (np.abs(main_glac_rgi.CenLon.values[:,np.newaxis] - gcm_lon_values_all).argmin(axis=1))
    latlon_nearidx = latlon_nearidx.drop_duplicates().sort_values(['CenLat', 'CenLon'])
    latlon_nearidx.reset_index(drop=True, inplace=True)
    latlon_reg = latlon_nearidx.copy()
    latlon_reg.CenLat.replace(gcm_lat_dict, inplace=True)
    latlon_reg.CenLon.replace(gcm_lon_dict, inplace=True)
    # ===== LOAD CLIMATE DATA =====
    # Reference climate data
    ref_gcm = class_climate.GCM(name=pygem_prms.ref_gcm_name)
    # Air temperature [degC], Precipitation [m], Elevation [masl], Lapse rate [K m-1]
    ref_temp, ref_dates = ref_gcm.importGCMvarnearestneighbor_xarray(ref_gcm.temp_fn, ref_gcm.temp_vn, latlon_reg, 
                                                                     dates_table_ref)
    ref_prec, ref_dates = ref_gcm.importGCMvarnearestneighbor_xarray(ref_gcm.prec_fn, ref_gcm.prec_vn, latlon_reg, 
                                                                     dates_table_ref)
#    ref_elev = ref_gcm.importGCMfxnearestneighbor_xarray(ref_gcm.elev_fn, ref_gcm.elev_vn, latlon_reg)
    # GCM climate data
    gcm_temp_all, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(gcm.temp_fn, gcm.temp_vn, latlon_reg, dates_table)
    gcm_prec_all, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(gcm.prec_fn, gcm.prec_vn, latlon_reg, dates_table)
#    gcm_elev = gcm.importGCMfxnearestneighbor_xarray(gcm.elev_fn, gcm.elev_vn, latlon_reg)
    # GCM subset to agree with reference time period to calculate bias corrections
    gcm_subset_idx_start = np.where(dates_table.date.values == dates_table_ref.date.values[0])[0][0]
    gcm_subset_idx_end = np.where(dates_table.date.values == dates_table_ref.date.values[-1])[0][0]
    gcm_temp = gcm_temp_all[:,gcm_subset_idx_start:gcm_subset_idx_end+1]
    gcm_prec = gcm_prec_all[:,gcm_subset_idx_start:gcm_subset_idx_end+1]
    
    ## ===== BIAS ADJUSTMENTS =====
    # OPTION 2: Adjust temp and prec according to Huss and Hock (2015) accounts for means and interannual variability
    if pygem_prms.option_bias_adjustment == 2:        
        # TEMPERATURE BIAS CORRECTIONS
        # Mean monthly temperature
        ref_temp_monthly_avg = (ref_temp.reshape(-1,12).transpose()
                                .reshape(-1,int(ref_temp.shape[1]/12)).mean(1).reshape(12,-1).transpose())
        gcm_temp_monthly_avg = (gcm_temp.reshape(-1,12).transpose()
                                .reshape(-1,int(gcm_temp.shape[1]/12)).mean(1).reshape(12,-1).transpose())
        # Monthly bias adjustment
        gcm_temp_monthly_adj = ref_temp_monthly_avg - gcm_temp_monthly_avg
        # Monthly temperature bias adjusted according to monthly average
        t_mt = gcm_temp_all + np.tile(gcm_temp_monthly_adj, int(gcm_temp_all.shape[1]/12))
        # Mean monthly temperature bias adjusted according to monthly average
        t_m25avg = np.tile(gcm_temp_monthly_avg + gcm_temp_monthly_adj, int(gcm_temp_all.shape[1]/12))
        # Calculate monthly standard deviation of temperature
        ref_temp_monthly_std = (ref_temp.reshape(-1,12).transpose()
                                .reshape(-1,int(ref_temp.shape[1]/12)).std(1).reshape(12,-1).transpose())
        gcm_temp_monthly_std = (gcm_temp.reshape(-1,12).transpose()
                                .reshape(-1,int(gcm_temp.shape[1]/12)).std(1).reshape(12,-1).transpose())
        variability_monthly_std = ref_temp_monthly_std / gcm_temp_monthly_std
        # Bias adjusted temperature accounting for monthly mean and variability
        gcm_temp_bias_adj = t_m25avg + (t_mt - t_m25avg) * np.tile(variability_monthly_std, int(gcm_temp_all.shape[1]/12))
        
        # PRECIPITATION BIAS CORRECTIONS
        # Calculate monthly mean precipitation
        ref_prec_monthly_avg = (ref_prec.reshape(-1,12).transpose()
                                .reshape(-1,int(ref_temp.shape[1]/12)).mean(1).reshape(12,-1).transpose())
        gcm_prec_monthly_avg = (gcm_prec.reshape(-1,12).transpose()
                                .reshape(-1,int(gcm_temp.shape[1]/12)).mean(1).reshape(12,-1).transpose())
        bias_adj_prec = ref_prec_monthly_avg / gcm_prec_monthly_avg
        # Bias adjusted precipitation accounting for differences in monthly mean
        gcm_prec_bias_adj = gcm_prec_all * np.tile(bias_adj_prec, int(gcm_temp_all.shape[1]/12))
        
        # Regional means
        reg_mean_temp_biasadj = gcm_temp_bias_adj.mean(axis=0)
        reg_mean_prec_biasadj = gcm_prec_bias_adj.mean(axis=0)
        
        return reg_mean_temp_biasadj, reg_mean_prec_biasadj


#%% LOAD ALL GLACIERS
# Load all glaciers
for rgi_region in rgi_regions:
    # Data on all glaciers
    main_glac_rgi_region = modelsetup.selectglaciersrgitable(rgi_regionsO1=[rgi_region], rgi_regionsO2 = 'all', 
                                                             rgi_glac_number='all')
     # Glacier hypsometry [km**2]
    main_glac_hyps_region = modelsetup.import_Husstable(main_glac_rgi_region, pygem_prms.hyps_filepath,
                                                        pygem_prms.hyps_filedict, pygem_prms.hyps_colsdrop)
    # Ice thickness [m], average
    main_glac_icethickness_region= modelsetup.import_Husstable(main_glac_rgi_region, 
                                                             pygem_prms.thickness_filepath, pygem_prms.thickness_filedict, 
                                                             pygem_prms.thickness_colsdrop)
    
    if rgi_region == rgi_regions[0]:
        main_glac_rgi_all = main_glac_rgi_region
        main_glac_hyps_all = main_glac_hyps_region
        main_glac_icethickness_all = main_glac_icethickness_region
    else:
        main_glac_rgi_all = pd.concat([main_glac_rgi_all, main_glac_rgi_region], sort=False)
        main_glac_hyps_all = pd.concat([main_glac_hyps_all, main_glac_hyps_region], sort=False)
        main_glac_icethickness_all = pd.concat([main_glac_icethickness_all, main_glac_icethickness_region], sort=False)
        
# Add watersheds, regions, degrees, mascons, and all groups to main_glac_rgi_all
# Watersheds
main_glac_rgi_all['watershed'] = main_glac_rgi_all.RGIId.map(watershed_dict)
# Regions
main_glac_rgi_all['kaab'] = main_glac_rgi_all.RGIId.map(kaab_dict)
# Degrees
main_glac_rgi_all['CenLon_round'] = np.floor(main_glac_rgi_all.CenLon.values/degree_size) * degree_size
main_glac_rgi_all['CenLat_round'] = np.floor(main_glac_rgi_all.CenLat.values/degree_size) * degree_size
deg_groups = main_glac_rgi_all.groupby(['CenLon_round', 'CenLat_round']).size().index.values.tolist()
deg_dict = dict(zip(deg_groups, np.arange(0,len(deg_groups))))
main_glac_rgi_all.reset_index(drop=True, inplace=True)
cenlon_cenlat = [(main_glac_rgi_all.loc[x,'CenLon_round'], main_glac_rgi_all.loc[x,'CenLat_round']) 
                 for x in range(len(main_glac_rgi_all))]
main_glac_rgi_all['CenLon_CenLat'] = cenlon_cenlat
main_glac_rgi_all['deg_id'] = main_glac_rgi_all.CenLon_CenLat.map(deg_dict)
# Mascons
if grouping == 'mascon' or option_subset_GRACE == 1:
    main_glac_rgi_all['mascon_idx'] = np.nan
    for glac in range(main_glac_rgi_all.shape[0]):
        latlon_dist = (((mascon_df.CenLat.values - main_glac_rgi_all.CenLat.values[glac])**2 + 
                         (mascon_df.CenLon.values - main_glac_rgi_all.CenLon.values[glac])**2)**0.5)
        main_glac_rgi_all.loc[glac,'mascon_idx'] = [x[0] for x in np.where(latlon_dist == latlon_dist.min())][0]
    mascon_groups = main_glac_rgi_all.mascon_idx.unique().tolist()
    mascon_groups = [int(x) for x in mascon_groups]
    mascon_groups = sorted(mascon_groups)
    mascon_latlondict = dict(zip(mascon_groups, mascon_df[['CenLat', 'CenLon']].values[mascon_groups].tolist()))
    
# All
main_glac_rgi_all['all_group'] = 'all'



#%% TIME SERIES OF SUBPLOTS FOR EACH GROUP
if option_plot_cmip5_normalizedchange == 1:
#    vns = ['volume_glac_annual', 'runoff_glac_annual']
    vns = ['volume_glac_annual']
#    vns = ['runoff_glac_annual']
#    vns = ['temp_glac_annual']
#    vns = ['prec_glac_annual']
    # NOTE: Temperatures and precipitation will not line up exactly because each region is covered by a different 
    #       number of pixels, and hence the mean of those pixels is not going to be equal.

    multimodel_linewidth = 2
    alpha=0.2
    
    # Determine grouping
    if grouping == 'rgi_region':
        groups = rgi_regions
        group_cn = 'O1Region'
    elif grouping == 'watershed':
        groups = main_glac_rgi_all.watershed.unique().tolist()
        group_cn = 'watershed'
    elif grouping == 'kaab':
        groups = main_glac_rgi_all.kaab.unique().tolist()
        group_cn = 'kaab'
        groups = [x for x in groups if str(x) != 'nan']  
    elif grouping == 'degree':
        groups = main_glac_rgi_all.deg_id.unique().tolist()
        group_cn = 'deg_id'
    elif grouping == 'all':
        groups = ['all']
        group_cn = 'all_group'
    try:
        groups = sorted(groups, key=str.lower)
    except:
        groups = groups
    
    if grouping == 'watershed':
        groups.remove('Irrawaddy')
        groups.remove('Yellow')
#    # Adjust groups if desired
#    remove_groups = ['Amu_Darya', 'Brahmaputra', 'Ili', 'Inner_Tibetan_Plateau', 'Inner_Tibetan_Plateau_extended',
#                     'Mekong', 'Salween', 'Syr_Darya', 'Tarim']        
#    for x in remove_groups:
#        print('removed group ', x)
#        groups.remove(x)
    
    reg_legend = []
    num_cols_max = 4
    if len(groups) < num_cols_max:
        num_cols = len(groups)
    else:
        num_cols = num_cols_max
    num_rows = int(np.ceil(len(groups)/num_cols))
        
    for vn in vns:
        fig, ax = plt.subplots(num_rows, num_cols, squeeze=False, sharex=False, sharey=True,
                               figsize=(5*num_rows,4*num_cols), gridspec_kw = {'wspace':0, 'hspace':0})
        add_group_label = 1
        
        for rcp in rcps:
#        for rcp in ['rcp85']:
            ds_multimodels = [[] for group in groups]
            if vn == 'volume_glac_annual':
                masschg_multimodels = [[] for group in groups]
            
            for ngcm, gcm_name in enumerate(gcm_names):
#            for ngcm, gcm_name in enumerate(['CSIRO-Mk3-6-0']):
#                print(ngcm, gcm_name)
            
                # Merge all data, then select group data
                for region in rgi_regions:                        
                    # Load datasets
                    if gcm_name == 'ERA-Interim':
                        netcdf_fp = netcdf_fp_era
                        ds_fn = 'R' + str(region) + '--ERA-Interim_c2_ba0_200sets_2000_2017_stats.nc'
                    else:
                        netcdf_fp = netcdf_fp_cmip5 + gcm_name + '/'
                        ds_fn = ('R' + str(region) + '_' + gcm_name + '_' + rcp + '_c2_ba1_100sets_2000_2100.nc')    
                    # Bypass GCMs that are missing a rcp scenario
                    try:
                        ds = xr.open_dataset(netcdf_fp + ds_fn)
                    except:
                        continue
                    # Extract time variable
                    if 'annual' in vn:
                        try:
                            time_values = ds[vn].coords['year_plus1'].values
                        except:
                            time_values = ds[vn].coords['year'].values
                    # Merge datasets
                    if region == rgi_regions[0]:
                        vn_glac_all = ds[vn].values[:,:,0]
                        vn_glac_std_all = ds[vn].values[:,:,1]
                    else:
                        vn_glac_all = np.concatenate((vn_glac_all, ds[vn].values[:,:,0]), axis=0)
                        vn_glac_std_all = np.concatenate((vn_glac_std_all, ds[vn].values[:,:,1]), axis=0)
                    
                    try:
                        ds.close()
                    except:
                        continue
                    
                
                # Cycle through groups  
                row_idx = 0
                col_idx = 0
                for ngroup, group in enumerate(groups):
#                for ngroup, group in enumerate([groups[1]]):
                    # Set subplot position
                    if (ngroup % num_cols == 0) and (ngroup != 0):
                        row_idx += 1
                        col_idx = 0
                        
                    # Select subset of data
                    main_glac_rgi = main_glac_rgi_all.loc[main_glac_rgi_all[group_cn] == group]                        
                    vn_glac = vn_glac_all[main_glac_rgi.index.values.tolist(),:]
                    vn_glac_std = vn_glac_std_all[main_glac_rgi.index.values.tolist(),:]
                    vn_glac_var = vn_glac_std **2
                     
                    # Plot data
                    if vn == 'volume_glac_annual':
                        # Regional mean, standard deviation, and variance
                        #  mean: E(X+Y) = E(X) + E(Y)
                        #  var: Var(X+Y) = Var(X) + Var(Y) + 2*Cov(X,Y)
                        #    assuming X and Y are indepdent, then Cov(X,Y)=0, so Var(X+Y) = Var(X) + Var(Y)
                        #  std: std(X+Y) = (Var(X+Y))**0.5
                        vn_reg = vn_glac.sum(axis=0)
                        vn_reg_var = vn_glac_var.sum(axis=0)
                        vn_reg_std = vn_reg_var**0.5
                        vn_reg_stdhigh = vn_reg + vn_reg_std
                        vn_reg_stdlow = vn_reg - vn_reg_std
                        # Regional normalized volume           
                        vn_reg_norm = vn_reg / vn_reg[0]
                        vn_reg_norm_stdhigh = vn_reg_stdhigh / vn_reg[0]
                        vn_reg_norm_stdlow = vn_reg_stdlow / vn_reg[0]
                        vn_reg_plot = vn_reg_norm.copy()
                        vn_reg_plot_stdlow = vn_reg_norm_stdlow.copy()
                        vn_reg_plot_stdhigh = vn_reg_norm_stdhigh.copy()
                        
                        # Mass change for text on plot
                        #  Gt = km3 ice * density_ice / 1000
                        #  divide by 1000 because density of ice is 900 kg/m3 or 0.900 Gt/km3
                        vn_reg_masschange = (vn_reg[-1] - vn_reg[0]) * pygem_prms.density_ice / 1000
                        
                    elif ('prec' in vn) or ('temp' in vn):       
                        # Regional mean function (monthly data)
                        reg_mean_temp_biasadj, reg_mean_prec_biasadj = (
                                select_region_climatedata(gcm_name, rcp, main_glac_rgi))
                        # Annual region mean
                        if 'prec' in vn:
                            reg_var_mean_annual = reg_mean_prec_biasadj.reshape(-1,12).sum(axis=1)
                        elif 'temp' in vn:
                            reg_var_mean_annual = reg_mean_temp_biasadj.reshape(-1,12).mean(axis=1)
                        # Plot data
                        vn_reg_plot = reg_var_mean_annual.copy()
                    elif vn == 'runoff_glac_annual':
                        # Regional mean, standard deviation, and variance
                        #  mean: E(X+Y) = E(X) + E(Y)
                        #  var: Var(X+Y) = Var(X) + Var(Y) + 2*Cov(X,Y)
                        #    assuming X and Y are indepdent, then Cov(X,Y)=0, so Var(X+Y) = Var(X) + Var(Y)
                        #  std: std(X+Y) = (Var(X+Y))**0.5
                        vn_reg = vn_glac.sum(axis=0)
                        vn_reg_var = vn_glac_var.sum(axis=0)
                        vn_reg_std = vn_reg_var**0.5
                        vn_reg_stdhigh = vn_reg + vn_reg_std
                        vn_reg_stdlow = vn_reg - vn_reg_std
                        # Runoff from 2000 - 2017
                        t1_idx = np.where(time_values == 2000)[0][0]
                        t2_idx = np.where(time_values == 2017)[0][0]
                        vn_reg_2000_2017_mean = vn_reg[t1_idx:t2_idx+1].sum() / (t2_idx - t1_idx + 1)
                        # Regional normalized volume        
                        vn_reg_norm = vn_reg / vn_reg_2000_2017_mean
                        vn_reg_norm_stdhigh = vn_reg_stdhigh / vn_reg_2000_2017_mean
                        vn_reg_norm_stdlow = vn_reg_stdlow / vn_reg_2000_2017_mean
                        vn_reg_plot = vn_reg_norm.copy()
                        vn_reg_plot_stdlow = vn_reg_norm_stdlow.copy()
                        vn_reg_plot_stdhigh = vn_reg_norm_stdhigh.copy()

                    # ===== Plot =====                    
                    if option_plot_individual_gcms == 1:
                        ax[row_idx, col_idx].plot(time_values, vn_reg_plot, color=rcp_colordict[rcp], linewidth=1, 
                                                  alpha=alpha, label=None)
    #                    # Volume change uncertainty
    #                    if vn == 'volume_glac_annual':
    #                        ax[row_idx, col_idx].fill_between(
    #                                time_values, vn_reg_plot_stdlow, vn_reg_plot_stdhigh, 
    #                                facecolor=gcm_colordict[gcm_name], alpha=0.15, label=None)
                    
                    # Group labels
#                    ax[row_idx, col_idx].set_title(title_dict[group], size=14)
                    if add_group_label == 1:
                        ax[row_idx, col_idx].text(0.5, 0.99, title_dict[group], size=14, horizontalalignment='center', 
                                                  verticalalignment='top', transform=ax[row_idx, col_idx].transAxes)
                                
                    # Tick parameters
                    ax[row_idx, col_idx].tick_params(axis='both', which='major', labelsize=25, direction='inout')
                    ax[row_idx, col_idx].tick_params(axis='both', which='minor', labelsize=15, direction='inout')
                    # X-label
                    ax[row_idx, col_idx].set_xlim(time_values.min(), time_values.max())
                    ax[row_idx, col_idx].xaxis.set_tick_params(labelsize=14)
                    ax[row_idx, col_idx].xaxis.set_major_locator(plt.MultipleLocator(50))
                    ax[row_idx, col_idx].xaxis.set_minor_locator(plt.MultipleLocator(10))
                    if col_idx == 0 and row_idx == num_rows-1:
                        ax[row_idx, col_idx].set_xticklabels(['','2000','2050','2100'])
                    elif row_idx == num_rows-1:
                        ax[row_idx, col_idx].set_xticklabels(['','','2050','2100'])
                    else:
                        ax[row_idx, col_idx].set_xticklabels(['','','',''])
                    #  labels are the first one, 2000, 2050, 2100, and 2101
                    # Y-label
                    ax[row_idx, col_idx].yaxis.set_tick_params(labelsize=14)
#                    ax[row_idx, col_idx].yaxis.set_major_locator(MaxNLocator(prune='both'))
                    if vn == 'volume_glac_annual':
                        if option_plot_individual_gcms == 1:
                            ax[row_idx, col_idx].set_ylim(0,1.35)
                        ax[row_idx, col_idx].yaxis.set_major_locator(plt.MultipleLocator(0.2))
                        ax[row_idx, col_idx].yaxis.set_minor_locator(plt.MultipleLocator(0.1))
                    elif vn == 'runoff_glac_annual':
                        ax[row_idx, col_idx].set_ylim(0,2)
                        ax[row_idx, col_idx].yaxis.set_major_locator(plt.MultipleLocator(0.5))
                        ax[row_idx, col_idx].yaxis.set_minor_locator(plt.MultipleLocator(0.1))
                        ax[row_idx, col_idx].set_yticklabels(['','','0.5','1.0','1.5', ''])
                    elif vn == 'temp_glac_annual':
                        ax[row_idx, col_idx].yaxis.set_major_locator(plt.MultipleLocator())
                        ax[row_idx, col_idx].yaxis.set_minor_locator(plt.MultipleLocator())
                    elif vn == 'prec_glac_annual':
                        ax[row_idx, col_idx].yaxis.set_major_locator(plt.MultipleLocator())
                        ax[row_idx, col_idx].yaxis.set_minor_locator(plt.MultipleLocator())
                    
                    # Count column index to plot
                    col_idx += 1
                    
                    # Record data for multi-model stats
                    if ngcm == 0:
                        ds_multimodels[ngroup] = [group, vn_reg_plot]
                    else:
                        ds_multimodels[ngroup][1] = np.vstack((ds_multimodels[ngroup][1], vn_reg_plot))
                    
                    if vn == 'volume_glac_annual':
                        if ngcm == 0:
    #                        print(group, rcp, gcm_name, vn_reg_masschange)
                            masschg_multimodels[ngroup] = [group, vn_reg_masschange]
                        else:
    #                        print(group, rcp, gcm_name, vn_reg_masschange)
                            masschg_multimodels[ngroup][1] = np.vstack((masschg_multimodels[ngroup][1], 
                                                                       vn_reg_masschange))
                        
                        
                # Only add group label once
                add_group_label = 0
            
            if vn == 'temp_glac_annual' or vn == 'prec_glac_annual':
                skip_fill = 1
            else:
                skip_fill = 0
            
#            # Multi-model mean
#            row_idx = 0
#            col_idx = 0
#            for ngroup, group in enumerate(groups):
#                if (ngroup % num_cols == 0) and (ngroup != 0):
#                    row_idx += 1
#                    col_idx = 0
#                # Multi-model statistics
#                vn_multimodel_mean = ds_multimodels[ngroup][1].mean(axis=0)
#                vn_multimodel_std = ds_multimodels[ngroup][1].std(axis=0)
#                vn_multimodel_stdlow = vn_multimodel_mean - vn_multimodel_std
#                vn_multimodel_stdhigh = vn_multimodel_mean + vn_multimodel_std
#                ax[row_idx, col_idx].plot(time_values, vn_multimodel_mean, color=rcp_colordict[rcp], 
#                                          linewidth=multimodel_linewidth, label=rcp)
#                if skip_fill == 0:
#                    ax[row_idx, col_idx].fill_between(time_values, vn_multimodel_stdlow, vn_multimodel_stdhigh, 
#                                                      facecolor=rcp_colordict[rcp], alpha=0.2, label=None)  
#                   
#                # Add mass change to plot
#                if vn == 'volume_glac_annual':
#                    masschg_multimodel_mean = masschg_multimodels[ngroup][1].mean(axis=0)[0]
#                
#                    print(group, rcp, np.round(masschg_multimodel_mean,0),'Gt', 
#                          np.round((vn_multimodel_mean[-1] - 1)*100,0), '%')
#                
#                if vn == 'volume_glac_annual' and rcp == rcps[-1]:
#                    masschange_str = '(' + str(masschg_multimodel_mean).split('.')[0] + ' Gt)'
#                    if grouping == 'all':
#                        ax[row_idx, col_idx].text(0.5, 0.93, masschange_str, size=12, horizontalalignment='center', 
#                                                  verticalalignment='top', transform=ax[row_idx, col_idx].transAxes, 
#                                                  color=rcp_colordict[rcp])
#                    else:
#                        ax[row_idx, col_idx].text(0.5, 0.88, masschange_str, size=12, horizontalalignment='center', 
#                                                  verticalalignment='top', transform=ax[row_idx, col_idx].transAxes, 
#                                                  color=rcp_colordict[rcp])
#                # Adjust subplot column index
#                col_idx += 1

        # RCP Legend
        rcp_lines = []
        for rcp in rcps:
            line = Line2D([0,1],[0,1], color=rcp_colordict[rcp], linewidth=multimodel_linewidth)
            rcp_lines.append(line)
        rcp_labels = [rcp_dict[rcp] for rcp in rcps]
        if vn == 'temp_glac_annual' or vn == 'prec_glac_annual':
            legend_loc = 'upper left'
        else:
            legend_loc = 'lower left'
        ax[0,0].legend(rcp_lines, rcp_labels, loc=legend_loc, fontsize=12, labelspacing=0, handlelength=1, 
                       handletextpad=0.5, borderpad=0, frameon=False, title='RCP')
        
#        # GCM Legend
#        gcm_lines = []
#        for gcm_name in gcm_names:
#            line = Line2D([0,1],[0,1], linestyle='-', color=gcm_colordict[gcm_name])
#            gcm_lines.append(line)
#        gcm_legend = gcm_names.copy()
#        fig.legend(gcm_lines, gcm_legend, loc='center right', title='GCMs', bbox_to_anchor=(1.06,0.5), 
#                   handlelength=0, handletextpad=0, borderpad=0, frameon=False)
        
        # Y-Label
        if len(groups) == 1:
            fig.text(-0.01, 0.5, vn_dict[vn], va='center', rotation='vertical', size=14)
        else:
            fig.text(0.03, 0.5, vn_dict[vn], va='center', rotation='vertical', size=16)
#        fig.text(0.03, 0.5, 'Normalized\nVolume [-]', va='center', ha='center', rotation='vertical', size=16)
#        fig.text(0.03, 0.5, 'Normalized\nGlacier Runoff [-]', va='center', ha='center', rotation='vertical', size=16)
        
        # Save figure
        if len(groups) == 1:
            fig.set_size_inches(4, 4)
        else:
            fig.set_size_inches(7, num_rows*2)
        if option_plot_individual_gcms == 1:
            figure_fn = grouping + '_' + vn + '_wgcms_' + str(len(gcm_names)) + 'gcms_' + str(len(rcps)) +  'rcps.png'
        else:
            figure_fn = grouping + '_' + vn + '_' + str(len(gcm_names)) + 'gcms_' + str(len(rcps)) +  'rcps.png'
                
        
        fig.savefig(figure_fp + figure_fn, bbox_inches='tight', dpi=300)


#%% RUNOFF COMPONENTS FOR EACH GROUP
if option_plot_cmip5_runoffcomponents == 1:
    vns = ['prec_glac_annual', 'melt_glac_annual', 'melt_glac_summer', 'refreeze_glac_annual']
    multimodel_linewidth = 1
    
    # Load all glaciers
    for rgi_region in rgi_regions:
        # Data on all glaciers
        main_glac_rgi_region = modelsetup.selectglaciersrgitable(rgi_regionsO1=[rgi_region], rgi_regionsO2 = 'all', 
                                                                 rgi_glac_number='all')
        if rgi_region == rgi_regions[0]:
            main_glac_rgi_all = main_glac_rgi_region
        else:
            main_glac_rgi_all = pd.concat([main_glac_rgi_all, main_glac_rgi_region])
    
    # Add watersheds to main_glac_rgi_all
    main_glac_rgi_all['watershed'] = main_glac_rgi_all.RGIId.map(watershed_dict)
    
    # Determine grouping
    if grouping == 'rgi_region':
        groups = rgi_regions
        group_cn = 'O1Region'
    elif grouping == 'watershed':
        groups = main_glac_rgi_all.watershed.unique().tolist()
        groups.remove('Irrawaddy')
        group_cn = 'watershed'
    groups = sorted(groups, key=str.lower)
    
    #%%
    reg_legend = []
    num_cols_max = 4
    if len(groups) < num_cols_max:
        num_cols = len(groups)
    else:
        num_cols = num_cols_max
    num_rows = int(np.ceil(len(groups)/num_cols))
    
    fig, ax = plt.subplots(num_rows, num_cols, squeeze=False, sharex=False, sharey=True,
                           figsize=(4*num_rows,3*num_cols), gridspec_kw = {'wspace':0, 'hspace':0})
    add_group_label = 1

    for rcp in rcps:
#    for rcp in ['rcp85']:
        ds_prec = [[] for group in groups]
        ds_melt = [[] for group in groups]
        ds_melt_summer = [[] for group in groups]
        ds_refreeze = [[] for group in groups]
            
        for ngcm, gcm_name in enumerate(gcm_names):
            
            for vn in vns:
                
                # Merge all data, then select group data
                for region in rgi_regions:                        
                    # Load datasets
                    if gcm_name == 'ERA-Interim':
                        netcdf_fp = netcdf_fp_era
                        ds_fn = 'R' + str(region) + '--ERA-Interim_c2_ba0_200sets_2000_2017_stats.nc'
                    else:
                        netcdf_fp = netcdf_fp_cmip5 + vn + '/'
                        ds_fn = ('R' + str(region) + '_' + gcm_name + '_' + rcp + '_c2_ba2_100sets_2000_2100--' 
                                 + vn + '.nc')    
                    # Bypass GCMs that are missing a rcp scenario
                    try:
                        ds = xr.open_dataset(netcdf_fp + ds_fn)
                    except:
                        continue
                    # Extract time variable
                    if 'annual' in vn:
                        try:
                            time_values = ds[vn].coords['year_plus1'].values
                        except:
                            time_values = ds[vn].coords['year'].values
                    # Merge datasets
                    if region == rgi_regions[0]:
                        vn_glac_all = ds[vn].values[:,:,0]
                        vn_glac_std_all = ds[vn].values[:,:,1]
                    else:
                        vn_glac_all = np.concatenate((vn_glac_all, ds[vn].values[:,:,0]), axis=0)
                        vn_glac_std_all = np.concatenate((vn_glac_std_all, ds[vn].values[:,:,1]), axis=0)
                    
                    try:
                        ds.close()
                    except:
                        continue
                
                # Cycle through groups
                for ngroup, group in enumerate(groups):
                        
                    # Select subset of data
                    main_glac_rgi = main_glac_rgi_all.loc[main_glac_rgi_all[group_cn] == group]                        
                    vn_glac = vn_glac_all[main_glac_rgi.index.values.tolist(),:]
                    vn_glac_std = vn_glac_std_all[main_glac_rgi.index.values.tolist(),:]
                    vn_glac_var = vn_glac_std **2
                     
                    # Regional mean
                    vn_reg = vn_glac.sum(axis=0)
                    
                    # Record data for multi-model stats
                    if vn == 'prec_glac_annual':
                        if ngcm == 0:
                            ds_prec[ngroup] = [group, vn_reg]
                        else:
                            ds_prec[ngroup][1] = np.vstack((ds_prec[ngroup][1], vn_reg))
                    elif vn == 'melt_glac_annual':
                        if ngcm == 0:
                            ds_melt[ngroup] = [group, vn_reg]
                        else:
                            ds_melt[ngroup][1] = np.vstack((ds_melt[ngroup][1], vn_reg))
                    elif vn == 'melt_glac_summer':
                        if ngcm == 0:
                            ds_melt_summer[ngroup] = [group, vn_reg]
                        else:
                            ds_melt_summer[ngroup][1] = np.vstack((ds_melt_summer[ngroup][1], vn_reg))
                    elif vn == 'refreeze_glac_annual':
                        if ngcm == 0:
                            ds_refreeze[ngroup] = [group, vn_reg]
                        else:
                            ds_refreeze[ngroup][1] = np.vstack((ds_refreeze[ngroup][1], vn_reg))
                    
        # Multi-model mean
        row_idx = 0
        col_idx = 0
        for ngroup, group in enumerate(groups):
            if (ngroup % num_cols == 0) and (ngroup != 0):
                row_idx += 1
                col_idx = 0
            # Multi-model statistics
            prec_multimodel_mean = ds_prec[ngroup][1].mean(axis=0)
            melt_multimodel_mean = ds_melt[ngroup][1].mean(axis=0)
            melt_summer_multimodel_mean = ds_melt_summer[ngroup][1].mean(axis=0)
            refreeze_multimodel_mean = ds_refreeze[ngroup][1].mean(axis=0)
            # Runoff  and components (melt adjusted = melt - refreeze)
            runoff_multimodel_mean = prec_multimodel_mean + melt_multimodel_mean - refreeze_multimodel_mean
            meltadj_multimodel_mean = melt_multimodel_mean - refreeze_multimodel_mean
            meltadj_multimodel_mean_frac = meltadj_multimodel_mean / runoff_multimodel_mean
            prec_multimodel_mean_frac = prec_multimodel_mean / runoff_multimodel_mean
            melt_summer_mean_frac = (melt_summer_multimodel_mean - refreeze_multimodel_mean) / runoff_multimodel_mean
            

            # ===== Plot =====
            # Precipitation
            ax[row_idx, col_idx].plot(time_values, prec_multimodel_mean_frac, color=rcp_colordict[rcp], 
                                      linewidth=multimodel_linewidth, label=rcp)
            if rcp == 'rcp45':
                ax[row_idx, col_idx].fill_between(time_values, 0, prec_multimodel_mean_frac, 
                                                  facecolor='b', alpha=0.2, label=None)
            # Melt
            melt_summer_mean_frac2plot = prec_multimodel_mean_frac +  melt_summer_mean_frac
            ax[row_idx, col_idx].plot(time_values, melt_summer_mean_frac2plot, color=rcp_colordict[rcp], 
                                      linewidth=multimodel_linewidth, label=rcp)
            if rcp == 'rcp45':
                ax[row_idx, col_idx].fill_between(time_values, prec_multimodel_mean_frac, melt_summer_mean_frac2plot,
                                                  facecolor='y', alpha=0.2, label=None)
                        
            # Group labels
            if add_group_label == 1:
                ax[row_idx, col_idx].text(0.5, 0.99, title_dict[group], size=14, horizontalalignment='center', 
                                          verticalalignment='top', transform=ax[row_idx, col_idx].transAxes)
            # Tick parameters
            ax[row_idx, col_idx].tick_params(axis='both', which='major', labelsize=25, direction='inout')
            ax[row_idx, col_idx].tick_params(axis='both', which='minor', labelsize=15, direction='inout')
            # X-label
            ax[row_idx, col_idx].set_xlim(time_values.min(), time_values.max())
            ax[row_idx, col_idx].xaxis.set_tick_params(labelsize=14)
            ax[row_idx, col_idx].xaxis.set_major_locator(plt.MultipleLocator(50))
            ax[row_idx, col_idx].xaxis.set_minor_locator(plt.MultipleLocator(10))
            if col_idx == 0 and row_idx == num_rows-1:
                ax[row_idx, col_idx].set_xticklabels(['','2000','2050','2100'])
            elif row_idx == num_rows-1:
                ax[row_idx, col_idx].set_xticklabels(['','','2050','2100'])
            else:
                ax[row_idx, col_idx].set_xticklabels(['','','',''])
            #  labels are the first one, 2000, 2050, 2100, and 2101
            # Y-label
            ax[row_idx, col_idx].yaxis.set_tick_params(labelsize=14)
            ax[row_idx, col_idx].set_ylim(0,1)
            ax[row_idx, col_idx].yaxis.set_major_locator(plt.MultipleLocator(0.5))
            ax[row_idx, col_idx].yaxis.set_minor_locator(plt.MultipleLocator(0.1))
            ax[row_idx, col_idx].set_yticklabels(['','','0.5','1.0','1.5', ''])
            
            # Adjust subplot column index
            col_idx += 1
            
        # Only add group label once
        add_group_label = 0

        # RCP Legend
        rcp_lines = []
        for rcp in rcps:
            line = Line2D([0,1],[0,1], color=rcp_colordict[rcp], linewidth=multimodel_linewidth)
            rcp_lines.append(line)
        rcp_labels = [rcp_dict[rcp] for rcp in rcps]
        ax[0,0].legend(rcp_lines, rcp_labels, loc='center left', bbox_to_anchor=(0, 0.7), fontsize=12, 
                       labelspacing=0, handlelength=1, handletextpad=0.5, borderpad=0, frameon=False)
        
        # GCM Legend
        gcm_lines = []
        for gcm_name in gcm_names:
            line = Line2D([0,1],[0,1], linestyle='-', color=gcm_colordict[gcm_name])
            gcm_lines.append(line)
        gcm_legend = gcm_names.copy()
        fig.legend(gcm_lines, gcm_legend, loc='center right', title='GCMs', bbox_to_anchor=(1.06,0.5), 
                   handlelength=0, handletextpad=0, borderpad=0, frameon=False)
        
        # Y-Label
        fig.text(0.03, 0.5, 'Normalized Runoff Components [-]', va='center', rotation='vertical', size=16)
        
        # Save figure
        fig.set_size_inches(7, num_rows*2)
        figure_fn = grouping + '_runoffcomponents_' + str(len(gcm_names)) + 'gcms_' + str(len(rcps)) +  'rcps.png'
        fig.savefig(figure_fp + figure_fn, bbox_inches='tight', dpi=300)


#%% MAP OF VARIABLE OVERLAID BY DEGREES OR GLACIER DATA
if option_plot_cmip5_map == 1:    
    
    for rcp in rcps:
#    for rcp in ['rcp45']:
        # Merge all data and partition into groups
        groups, time_values, ds_vn, ds_glac = partition_multimodel_groups(gcm_names, grouping, vn, main_glac_rgi_all, 
                                                                          rcp=rcp)
#%%
        # Create the projection
        fig, ax = plt.subplots(1, 1, figsize=(10,5), subplot_kw={'projection':cartopy.crs.PlateCarree()})
        # Add country borders for reference
        if grouping == 'rgi_region':
            ax.add_feature(cartopy.feature.BORDERS, alpha=0.15)
        ax.add_feature(cartopy.feature.COASTLINE)
        # Set the extent
        ax.set_extent([east, west, south, north], cartopy.crs.PlateCarree())    
        # Label title, x, and y axes
        ax.set_xticks(np.arange(east,west+1,xtick), cartopy.crs.PlateCarree())
        ax.set_yticks(np.arange(south,north+1,ytick), cartopy.crs.PlateCarree())
        ax.set_xlabel(xlabel, size=12)
        ax.set_ylabel(ylabel, size=12)
            
        # Add group and attribute of interest
        if grouping == 'rgi_region':
            group_shp = cartopy.io.shapereader.Reader(rgiO1_shp_fn)
            group_shp_attr = 'RGI_CODE'
        elif grouping == 'watershed':
            group_shp = cartopy.io.shapereader.Reader(watershed_shp_fn)
            group_shp_attr = 'watershed'
        elif grouping == 'kaab':
            group_shp = cartopy.io.shapereader.Reader(kaab_shp_fn)
            group_shp_attr = 'Name'
            
        colorbar_dict = {'volume_norm':[0,1],
                         'peakwater':[2000,2100]}
        cmap = mpl.cm.RdYlBu
        norm = plt.Normalize(colorbar_dict[vn][0], colorbar_dict[vn][1])
        
        # Add attribute of interest to the shapefile
        for rec in group_shp.records():
            if rec.attributes[group_shp_attr] in groups:
                # Group index
                ds_idx = groups.index(rec.attributes[group_shp_attr])                
                vn_multimodel_mean = vn_multimodel_mean_processed(vn, ds_vn, ds_idx, time_values)
                
                # Value to plot
                if vn == 'volume_norm':
                    rec.attributes['value'] = vn_multimodel_mean[-1]
                elif vn == 'peakwater':
                    rec.attributes['value'] = vn_multimodel_mean
                    
                print(rec.attributes[group_shp_attr], rec.attributes['value'])

    
                # Add polygon to plot
                ax.add_geometries(rec.geometry, cartopy.crs.PlateCarree(), 
                                  facecolor=cmap(norm(rec.attributes['value'])),
                                  edgecolor='None', zorder=1)
                # plot polygon outlines on top of everything with their labels
                ax.add_geometries(rec.geometry, cartopy.crs.PlateCarree(), facecolor='None', 
                                  edgecolor='grey', linewidth=0.75, zorder=3)
                ax.text(title_location[rec.attributes[group_shp_attr]][0], 
                        title_location[rec.attributes[group_shp_attr]][1], 
                        title_dict[rec.attributes[group_shp_attr]], horizontalalignment='center', size=12, zorder=4)
                if rec.attributes[group_shp_attr] == 'Karakoram':
                    ax.plot([72.2, 76.2], [34.3, 35.8], color='black', linewidth=0.75)
                elif rec.attributes[group_shp_attr] == 'Pamir':
                    ax.plot([69.2, 73], [37.3, 38.3], color='black', linewidth=0.75)
                        
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm._A = []
        plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
        fig.text(0.95, 0.5, vn_dict[vn], va='center', rotation='vertical', size=14)
        
        if option_plot_individual_glaciers == 1:
            # Plot individual glaciers
            area_cutoffs = [100, 10, 1, 0.1]
            area_cutoffs_size = [1000, 100, 10, 2]
            area_sizes = size_thresholds(main_glac_rgi_all.Area.values, area_cutoffs, area_cutoffs_size)
            # Multi-model mean of all glaciers
            output_multimodel_mean_all_list = []
            for glac in range(len(main_glac_rgi_all)):
                output_multimodel_mean = vn_multimodel_mean_processed(vn, ds_glac, glac, time_values, every_glacier=1)
                output_multimodel_mean_all_list.append(output_multimodel_mean)
            output_multimodel_mean_all = np.array(output_multimodel_mean_all_list)
            
            # Value to plot
            if vn == 'volume_norm':
                output_multimodel_mean_all_plot = output_multimodel_mean_all[:,-1]
            elif vn == 'peakwater':
                output_multimodel_mean_all_plot = output_multimodel_mean_all
                
            glac_lons = main_glac_rgi_all.CenLon.values
            glac_lats = main_glac_rgi_all.CenLat.values
            sc = ax.scatter(glac_lons, glac_lats, c=output_multimodel_mean_all_plot, cmap=cmap, norm=norm, 
                            s=area_sizes,
                            edgecolor='grey', linewidth=0.25, transform=cartopy.crs.PlateCarree(), zorder=2)
            
            # Add legend for glacier sizes
            legend_glac = []
            legend_glac_labels = []
            legend_glac_markersize = [20, 10, 5, 2]
            for i_area, area_cutoff_size in enumerate(area_cutoffs_size):
                legend_glac.append(Line2D([0], [0], linestyle='None', marker='o', color='grey', 
                                          label=(str(area_cutoffs[i_area]) + 'km$^2$'), 
                                          markerfacecolor='grey', markersize=legend_glac_markersize[i_area]))
            plt.legend(handles=legend_glac, loc='lower left', fontsize=12)

        elif option_plot_degrees == 1:
            # Group by degree  
            groups_deg, time_values, ds_vn_deg, ds_glac = (
                    partition_multimodel_groups(gcm_names, 'degree', vn, main_glac_rgi_all, rcp=rcp))
            # Get values for each group
            for group_idx in range(len(ds_vn_deg)):
                vn_multimodel_mean_deg = vn_multimodel_mean_processed(vn, ds_vn_deg, group_idx, time_values)
                # Value to plot
                if vn == 'volume_norm':
                    ds_vn_deg[group_idx].append(vn_multimodel_mean_deg[-1])
                elif vn == 'peakwater':
                    ds_vn_deg[group_idx].append(vn_multimodel_mean_deg)
            z = [ds_vn_deg[ds_idx][2] for ds_idx in range(len(ds_vn_deg))]
            x = np.array([x[0] for x in deg_groups]) 
            y = np.array([x[1] for x in deg_groups])
            lons = np.arange(x.min(), x.max() + 2 * degree_size, degree_size)
            lats = np.arange(y.min(), y.max() + 2 * degree_size, degree_size)
            x_adj = np.arange(x.min(), x.max() + 1 * degree_size, degree_size) - x.min()
            y_adj = np.arange(y.min(), y.max() + 1 * degree_size, degree_size) - y.min()
            z_array = np.zeros((len(y_adj), len(x_adj)))
            z_array[z_array==0] = np.nan
            for i in range(len(z)):
                row_idx = int((y[i] - y.min()) / degree_size)
                col_idx = int((x[i] - x.min()) / degree_size)
                z_array[row_idx, col_idx] = z[i]
            ax.pcolormesh(lons, lats, z_array, cmap='RdYlBu', norm=norm, zorder=2)
            
#        elif option_plot_pies == 1:
#            
#            #%%
##            pie_volumes = []
#            pie_years = [2040, 2070, 2100]
#            pie_years_idx = [np.where(time_values == pie_year)[0][0] for pie_year in pie_years]
#            pie_volumes = [vn_multimodel_mean[idx] for idx in pie_years_idx]
#            pie_radii = [1.3,1,0.7]
#            
#            # Make data: I have 3 groups and 3 subgroups
##            group_names=['2040', '2070', '2100']
##            group_size=[12,11,30]
#            group_size = [1]
#            subgroup_names=['A.1', 'A.2', 'A.3', 'B.1', 'B.2', 'C.1', 'C.2', 'C.3', 'C.4', 'C.5']
#            subgroup_size=[4,3,5,6,5,10,5,5,4,6]
#             
#            # Create colors
#            a, b, c=[plt.cm.Blues, plt.cm.Reds, plt.cm.Greens]
#             
#            # First Ring (outside)
#            fig, ax = plt.subplots()
#            ax.axis('equal')
#            for i, pie_year in enumerate(pie_years):
#                i_volume = [pie_volumes[i], 1-pie_volumes[i]]
#                mypie, _ = ax.pie(i_volume, radius=pie_radii[i], labels=[str(pie_year), ''], colors=['grey','None'],
#                                  startangle=90, textprops={'fontsize': 14})
#                plt.setp( mypie, width=0.3, edgecolor='white')
#
#            # show it
#            plt.show()
#
#            #%%
            
        
        # Add time period and RCP
        if 'volume' in vn:
            additional_text = 'RCP ' + rcp_dict[rcp] + ': ' + str(time_values.max()-1)
        else:
            additional_text = 'RCP ' + rcp_dict[rcp] + ': ' + str(time_values.max())
        ax.text(0.98*west, 0.95*north, additional_text, horizontalalignment='right', fontsize=14)
    
        # Save figure
        fig.set_size_inches(10,6)
        if option_plot_individual_glaciers == 1:
            figure_fn = grouping + '_wglac_' + vn + '_' + str(len(gcm_names)) + 'gcms_' + rcp +  '.png'
        elif option_plot_degrees == 1:
            figure_fn = grouping + '_wdeg_' + vn + '_' + str(len(gcm_names)) + 'gcms_' + rcp +  '.png'
        else:
            figure_fn = grouping + '_' + vn + '_' + str(len(gcm_names)) + 'gcms_' + rcp +  '.png'
        fig.savefig(figure_fp + figure_fn, bbox_inches='tight', dpi=300)


#%% Output tables of mass change and peakwater
if option_output_tables == 1:
    
#    vns = ['mass_change', 'peakwater']
#    vns = ['peakwater']
    vns = ['mass_change']
    
    
#    groupings = ['all', 'rgi_region', 'watershed', 'kaab']
#    groupings = ['all']
#    groupings = ['rgi_region']
#    groupings = ['watershed']
    groupings = ['kaab']
    
     # Create filepath if it does not exist
    if os.path.exists(csv_fp) == False:
        os.makedirs(csv_fp)
    
    for grouping in groupings:
    
        # Select groups
        groups, group_cn = select_groups(grouping, main_glac_rgi_all)
        
        for vn in vns:
            if vn == 'mass_change':
                masschg_table_fn = ('MassChg_' + grouping + '_' + str(len(gcm_names)) + '_gcms_' + str(len(rcps)) + 
                                    '_rcps.csv')
                table_cns = []
                for rcp in rcps:
                    table_cns.append(rcp + '_MassChg_Gt')
                    table_cns.append(rcp + '_MassChg_std_Gt')
                    table_cns.append(rcp + '_VolChg_%')
                    table_cns.append(rcp + '_VolChg_std_%')
                output_table = pd.DataFrame(np.zeros((len(groups), len(table_cns))), index=groups, columns=table_cns)
                
                #%%
                # Load volume_glac_annual
                ds_vn_rcps = {}
                for rcp in rcps:
                    groups, time_values, ds_vn, ds_glac = (
                            partition_multimodel_groups(gcm_names, grouping, vn, main_glac_rgi_all, rcp=rcp))
                    ds_vn_rcps[rcp] = ds_vn
                #%%

                # Load area_glac_annual
                ds_area_rcps = {}
                area_vn = 'area_glac_annual'
                for rcp in rcps:
                    groups, time_values, ds_area, ds_area_glac = (
                            partition_multimodel_groups(gcm_names, grouping, area_vn, main_glac_rgi_all, rcp=rcp))
                    ds_area_rcps[rcp] = ds_area

                for rcp in rcps:
                    for ngroup, group in enumerate(groups):
                        print(rcp, group)
                        ds_vn_multimodel = ds_vn_rcps[rcp][ngroup][1].mean(axis=0)
                        ds_vn_multimodel_std = ds_vn_rcps[rcp][ngroup][1].std(axis=0)
                        
                        ds_area_multimodel = ds_area_rcps[rcp][ngroup][1].mean(axis=0)
                        
                        # Mass change [Gt]
                        #  Gt = km3 ice * density_ice / 1000
                        #  divide by 1000 because density of ice is 900 kg/m3 or 0.900 Gt/km3
                        vn_reg_masschange = (ds_vn_multimodel - ds_vn_multimodel[0]) * pygem_prms.density_ice / 1000
                        vn_reg_masschange_std = ds_vn_multimodel_std * pygem_prms.density_ice / 1000
                        output_table.loc[group, rcp + '_MassChg_Gt'] = np.round(vn_reg_masschange[-1],1)
                        output_table.loc[group, rcp + '_MassChg_std_Gt'] = np.round(vn_reg_masschange_std[-1],1)
                        
                        # Volume change [%]
                        vn_reg_volchg = (ds_vn_multimodel - ds_vn_multimodel[0]) / ds_vn_multimodel[0] * 100
                        vn_reg_volchg_std = ds_vn_multimodel_std / ds_vn_multimodel[0] * 100                        
                        output_table.loc[group, rcp + '_VolChg_%'] = np.round(vn_reg_volchg[-1],1)
                        output_table.loc[group, rcp + '_VolChg_std_%'] = np.round(vn_reg_volchg_std[-1],1)
                        
                        
                        
                        
                        
                        # Mass change rate [Gt/yr]
                        runningmean_years = 10
                        vn_reg_mass = ds_vn_multimodel * pygem_prms.density_ice / 1000
                        vn_reg_masschgrate = vn_reg_mass[1:] - vn_reg_mass[0:-1]
                        vn_reg_masschgrate_runningmean = uniform_filter(vn_reg_masschgrate, (runningmean_years))
#                        print('Mass change rate [Gt/yr] 2015:', np.round(vn_reg_masschgrate_runningmean[idx_2015],1), 
#                              '\nMass change rate [Gt/yr] 2100:', np.round(vn_reg_masschgrate_runningmean[-1],1))
                        vn_reg_masschgrate_mwe = (
                                vn_reg_masschgrate * 10**9 * 1000 / 1000 / 10**6 / ds_area_multimodel[0])
                        vn_reg_masschgrate_mwe_runningmean = uniform_filter(vn_reg_masschgrate_mwe, runningmean_years)
                        
                        idx_2015 = np.where(time_values == 2015)[0][0]
                        if group in ['Karakoram', 'Kunlun']:
                            print('Vol change [%] 2015:', np.round(vn_reg_volchg[idx_2015],1), 
                                  '\nVol change [%] 2100:', np.round(vn_reg_volchg[-1],1))
                            print('Mass balance [mwea] 2000-2015:', np.round(vn_reg_masschgrate_mwe[idx_2015],2), 
                                  '\nMass balance [mwea] 2000-2100:', np.round(vn_reg_masschgrate_mwe[-1],2))
                        
                # Export table
                output_table.to_csv(csv_fp + masschg_table_fn)
            
            
            if vn == 'peakwater':
                peakwater_table_fn = ('PeakWater_' + grouping + '_' + str(len(gcm_names)) + '_gcms_' + str(len(rcps)) + 
                                    '_rcps.csv')
                runoff_cns = []
                for rcp in rcps:
                    runoff_cns.append(rcp + '_PeakWater_Yr')
                    runoff_cns.append(rcp + '_PeakWater_std_Yr')
                    runoff_cns.append(rcp + '_PeakWaterChg_%')
                    runoff_cns.append(rcp + '_PeakWaterChg_std_%')
                    runoff_cns.append(rcp + '_RunoffChg_%')
                    runoff_cns.append(rcp + '_RunoffChg_std_%')
                runoff_table = pd.DataFrame(np.zeros((len(groups), len(runoff_cns))), index=groups, columns=runoff_cns)

                ds_vn_rcps = {}
                for rcp in rcps:
                    groups, time_values, ds_vn, ds_glac = (
                            partition_multimodel_groups(gcm_names, grouping, vn, main_glac_rgi_all, rcp=rcp))
                    ds_vn_rcps[rcp] = ds_vn

                for rcp in rcps:
                    for ngroup, group in enumerate(groups):
                        runoff = ds_vn_rcps[rcp][ngroup][1]

                        # Compute peak water of each one
                        nyears = 10
                        
                        peakwater_yr_gcms = np.zeros((runoff.shape[0]))
                        peakwater_chg_gcms = np.zeros((runoff.shape[0]))
                        runoff_chg_gcms = np.zeros((runoff.shape[0]))
                        for n in range(runoff.shape[0]):
                            peakwater_yr_gcms[n], peakwater_chg_gcms[n], runoff_chg_gcms[n] = (
                                    peakwater(runoff[n,:], time_values, nyears))
                        
                        # Peakwater Year
                        runoff_table.loc[group, rcp + '_PeakWater_Yr'] = np.round(np.mean(peakwater_yr_gcms),1)
                        runoff_table.loc[group, rcp + '_PeakWater_std_Yr'] = np.round(np.std(peakwater_yr_gcms),1)
                        
                        # Peakwater Change [%]
                        runoff_table.loc[group, rcp + '_PeakWaterChg_%'] = np.round(np.mean(peakwater_chg_gcms),1)
                        runoff_table.loc[group, rcp + '_PeakWaterChg_std_%'] = np.round(np.std(peakwater_chg_gcms),1)
                        
                        # Runoff Change [%] end of simulation
                        runoff_table.loc[group, rcp + '_RunoffChg_%'] = np.round(np.mean(runoff_chg_gcms),1)
                        runoff_table.loc[group, rcp + '_RunoffChg_std_%'] = np.round(np.std(runoff_chg_gcms),1)

                # Export table
                runoff_table.to_csv(csv_fp + peakwater_table_fn)
                
#%%
if option_subset_GRACE == 1:
    
    vns = ['mass_change']    
    grouping = 'mascon'
    gcm_names = ['ERA-Interim']
    timestep = 'monthly'
    
    output_fn = '../mascon_' + timestep + '_GlacierMassGt_ERA-Interim.txt'


    # Load volume, mass balance and area to extract monthly glacier mass [Gt]
    groups, time_values_annual, ds_vol, ds_vol_glac = (
            partition_multimodel_groups(gcm_names, grouping, 'volume_glac_annual', main_glac_rgi_all))
    groups, time_values_monthly, ds_mb, ds_mb_glac = (
            partition_multimodel_groups(gcm_names, grouping, 'massbaltotal_glac_monthly', main_glac_rgi_all))
    groups, time_values_annual, ds_area, ds_area_glac = (
            partition_multimodel_groups(gcm_names, grouping, 'area_glac_annual', main_glac_rgi_all))
    
    # Monthly glacier area
    ds_area_glac_monthly = np.repeat(ds_area_glac[:,:,:-1], 12, axis=2)
    # Monthly glacier mass change
    #  Area [km2] * mb [mwe] * (1 km / 1000 m) * density_water [kg/m3] * (1 Gt/km3  /  1000 kg/m3)
    ds_masschange_glac_monthly = ds_area_glac_monthly * ds_mb_glac / 1000 * pygem_prms.density_water / 1000
    # Monthly glacier mass
    ds_mass_glac_initial = ds_vol_glac[:,:,0][:,:,np.newaxis] * pygem_prms.density_ice / 1000
    ds_masschange_glac_monthly_cumsum = np.cumsum(ds_masschange_glac_monthly, axis=2)
    ds_mass_glac_monthly = np.repeat(ds_mass_glac_initial, len(time_values_monthly), axis=2)
    ds_mass_glac_monthly[:,:,1:] = ds_mass_glac_monthly[:,:,1:] + ds_masschange_glac_monthly_cumsum[:,:,:-1]
    
    # Check to see difference in deriving mass from mass balance or volume
    # If you run 100 simulations, mass change is slightly different if it's computed using the MB and area or using
    # the volume.  This is not a problem when only run a single simulation nor is it a proble with the calculation
    # of volume in the first year.  This suggests that it has something to do with differences in the means and how
    # outliers might propagate through.
    ds_mass_glac_annual_fromMB = ds_mass_glac_monthly[:,:,::12]
    ds_mass_glac_annual_fromVol = ds_vol_glac * pygem_prms.density_ice / 1000
        
    # Group monthly mass
    ds_mass_mascon_monthly = [[] for group in groups]
    ngcm = 0
    for ngroup, group in enumerate(groups):
        # Select subset of data
        main_glac_rgi = main_glac_rgi_all.loc[main_glac_rgi_all['mascon_idx'] == group]                        
        mascon_mass_glac = ds_mass_glac_monthly[0,main_glac_rgi.index.values.tolist(),:]            
        # Regional sum
        mascon_mass = mascon_mass_glac.sum(axis=0)                
        # Record data for multi-model stats
        if ngcm == 0:
            ds_mass_mascon_monthly[ngroup] = [group, mascon_mass]
        else:
            ds_mass_mascon_monthly[ngroup][1] = np.vstack((ds_mass_mascon_monthly[ngroup][1], mascon_mass))

    # Convert monthly timestep to Year-Month
    if timestep == 'monthly':
        time_values_df = pd.DatetimeIndex(time_values_monthly)
        time_values = [str(x.year) + '-' + str(x.month) for x in time_values_df]
    elif timestep == 'annual':
        time_values = time_values_annual[:-1]
    
    # Output column names
    mascon_output_cns = ['mascon_idx', 'CenLat', 'CenLon']
    for x in time_values:
        mascon_output_cns.append(x)
    # Mascon index
    mascon_output_array = np.reshape(np.array(groups),(-1,1))
    # Mascon center lat/lon
    mascon_latlon = np.array([mascon_latlondict[x] for x in groups])
    mascon_output_array = np.hstack([mascon_output_array, mascon_latlon])
    # Mascon glacier mass [Gt]
    mascon_values_list = [ds_mass_mascon_monthly[x][1] for x in range(len(groups))]
    mascon_values_monthly = np.vstack(mascon_values_list)
    if timestep == 'annual':
        mascon_values = mascon_values_monthly[:,::12]
    elif timestep == 'monthly':
        mascon_values = mascon_values_monthly
    mascon_output_array = np.hstack([mascon_output_array, mascon_values])
    
    # Output dataframe
    mascon_output_df = pd.DataFrame(mascon_output_array, columns=mascon_output_cns)
    mascon_output_df.to_csv(mascon_fp + output_fn, sep=' ', index=False)
    
    if timestep == 'annual':
        headerline=('Annual glacier mass [Gt] for each mascon where at least 1 HMA glacier exists\n' + 
                    'Glacier mass change modeled using the Python Glacier Evolution Model (PyGEM)\n' +
                    'forced by ERA-Interim climate data\n'
                    'Glaciers aggregated according to nearest center latitude and longitude\n' + 
                    'Years refer to water years (e.g., 2000 is October 1999 - September 2000)\n' +
                    'Glacier mass refers to mass at the start of the year (e.g., mass_change_2000 = mass_2001 - '
                    'mass_2000\n' +
                    'Mascons provided by Bryant Loomis (NASA GSFC mascons)\n' + 
                    'Contact: drounce@alaska.edu\n' + 
                    'Column 1: Mascon index used to aggregate glaciers\n' + 
                    'Column 2: Mascon latitude center [deg]\n' +
                    'Column 3: Mascon longitude center [deg]\n' +
                    'Columns 4+: Water year\n' + 
                    'END OF COMMENTS (first row is header of column names)\n')
    elif timestep == 'monthly':
        headerline=('Monthly glacier mass [Gt] for each mascon where at least 1 HMA glacier exists\n' + 
                    'Glacier mass change modeled using the Python Glacier Evolution Model (PyGEM)\n' +
                    'forced by ERA-Interim climate data\n'
                    'Glaciers aggregated according to nearest center latitude and longitude\n' + 
                    'Glacier mass refers to mass at the start of the month (e.g., mass_change_Sept = mass_Oct - '
                    'mass_Sept)\n' +
                    'Mascons provided by Bryant Loomis (NASA GSFC mascons)\n' + 
                    'Contact: drounce@alaska.edu\n' + 
                    'Column 1: Mascon index used to aggregate glaciers\n' + 
                    'Column 2: Mascon latitude center [deg]\n' +
                    'Column 3: Mascon longitude center [deg]\n' +
                    'Columns 4+: Year-Month\n' + 
                    'END OF COMMENTS (first row is header of column names)\n')
            
    with open(mascon_fp + output_fn, 'r+') as f:
        content=f.read()
        f.seek(0,0)
        f.write(headerline.rstrip('\r\n') + '\n' + content)
    
    
    

#%%
## COUNT HOW MANY BIAS ADJUSTMENTS HAVE RIDICULOUS VALUES
#biasadj_fp = '/Users/davidrounce/Documents/Dave_Rounce/HiMAT/Output/biasadj/'
#gcm_names = ['CanESM2', 'CCSM4', 'CNRM-CM5', 'CSIRO-Mk3-6-0', 'GFDL-CM3', 'GFDL-ESM2M', 'GISS-E2-R', 'IPSL-CM5A-LR', 
#             'IPSL-CM5A-MR', 'MIROC5', 'MPI-ESM-LR', 'MRI-CGCM3', 'NorESM1-M']
#maxvalue = 1000
#for gcm in gcm_names:
#    for rcp in rcps:
#        for region in [13, 14, 15]:
#            biasadj_fn = 'R' + str(region) + '_' + gcm + '_' + rcp + '_biasadj_opt2_2000_2017_wy1.csv'
#            
#            ds = pd.read_csv(biasadj_fp + biasadj_fn, index_col=0)
#            cns = list(ds.columns.values)
#            prec_cns = []
#            for cn in cns:
#                if 'precadj' in cn:
#                    prec_cns.append(cn)
#            ds_prec = ds[prec_cns]
#            ds_prec_maxcol = ds_prec.max(axis=1)
#            ds_prec_large = ds_prec_maxcol.loc[ds_prec_maxcol > maxvalue]
#            
#            if ds_prec_large.shape[0] > 0:
#                print(gcm, rcp, region, ds_prec_large.shape[0])

#%% PLOT CALIBRATION MODEL PARAMETERS
if option_plot_modelparam == 1:
    
    vns = ['ddfsnow', 'tempchange', 'precfactor']
#    vns = ['precfactor']
    option_addbackground_group = 0
    modelparams_fn = 'modelparams_all_mean_20181018.csv'
    
    east = 104
    west = 67
    south = 25
    north = 48
    
    labelsize = 13
    
    colorbar_dict = {'volume_norm':[0,1],
                     'peakwater':[2000,2100],
                     'precfactor':[0.8,1.2],
                     'tempchange':[-2,2],
                     'ddfsnow':[3.6,4.6]}
    
    # Load mean of all model parameters
    if os.path.isfile(cal_fp + modelparams_fn) == False:
        modelparams_all = pd.DataFrame(np.zeros((main_glac_rgi_all.shape[0], len(pygem_prms.modelparams_colnames))), 
                                           columns=pygem_prms.modelparams_colnames)
        for glac in range(main_glac_rgi_all.shape[0]):
            
            glac_rgiid_full = main_glac_rgi_all.loc[main_glac_rgi_all.index.values[glac],'RGIId']
            if glac%200 == 0:
                print(glac_rgiid_full)
            
            glacno = str(glac_rgiid_full.split('-')[1])
            regionO1 = int(glacno.split('.')[0])
            
            ds_mp = xr.open_dataset(cal_fp + 'reg' + str(regionO1) + '/' + glacno + '.nc')
            cn_subset = pygem_prms.modelparams_colnames
            modelparams_all.iloc[glac,:] = (pd.DataFrame(ds_mp['mp_value'].sel(chain=0).values,columns=ds_mp.mp.values)
                                            [pygem_prms.modelparams_colnames].mean().values)
            modelparams_all.to_csv(cal_fp + modelparams_fn, index=False)
            ds_mp.close()
    else:
        modelparams_all = pd.read_csv(cal_fp + modelparams_fn)
        
    # Convert ddfsnow from [m w.e. to mm w.e.]
    modelparams_all['ddfsnow'] = modelparams_all['ddfsnow'] * 1000
    modelparams_all['ddfice'] = modelparams_all['ddfice'] * 1000
        
    # Add model parameters to main_glac_rgi_all
    main_glac_rgi_all[pygem_prms.modelparams_colnames] = modelparams_all

    for vn in vns:
    
        # Group data
        groups, ds_group = partition_modelparams_groups(grouping, vn, main_glac_rgi_all)
    
        # Create the projection
        fig, ax = plt.subplots(1, 1, figsize=(10,5), subplot_kw={'projection':cartopy.crs.PlateCarree()})
        # Add country borders for reference
        ax.add_feature(cartopy.feature.BORDERS, alpha=0.15, zorder=10)
        ax.add_feature(cartopy.feature.COASTLINE)
        # Set the extent
        ax.set_extent([east, west, south, north], cartopy.crs.PlateCarree())    
        # Label title, x, and y axes
        ax.set_xticks(np.arange(east,west+1,xtick), cartopy.crs.PlateCarree())
        ax.set_yticks(np.arange(south,north+1,ytick), cartopy.crs.PlateCarree())
        ax.set_xlabel(xlabel, size=labelsize)
        ax.set_ylabel(ylabel, size=labelsize)
        
#        # Add background DEM
#        gtif = gdal.Open(srtm_fn)
#        arr = gtif.ReadAsArray()   #Values
#        trans = gtif.GetGeoTransform()  #Defining bounds
#        extent = (trans[0], trans[0] + gtif.RasterXSize*trans[1],
#                trans[3] + gtif.RasterYSize*trans[5], trans[3])
#        norm_srtm = plt.Normalize(3000,6000)
#        srtm = ax.imshow(arr[:,:],extent=extent,transform=cartopy.crs.PlateCarree(), norm=norm_srtm, origin='upper', 
#                         cmap='Greys')
#        
#        plt.colorbar(srtm, ax=ax, fraction=0.03, pad=0.02)
        
#        # Add contour lines
        srtm_contour_fn
        srtm_contour_shp = cartopy.io.shapereader.Reader(srtm_contour_fn)
        srtm_contour_feature = cartopy.feature.ShapelyFeature(srtm_contour_shp.geometries(), cartopy.crs.PlateCarree(),
                                                              edgecolor='black', facecolor='none', linewidth=0.15)
        ax.add_feature(srtm_contour_feature, zorder=9)
        
        
            
        # Add group and attribute of interest
        if grouping == 'rgi_region':
            group_shp = cartopy.io.shapereader.Reader(rgiO1_shp_fn)
            group_shp_attr = 'RGI_CODE'
        elif grouping == 'watershed':
            group_shp = cartopy.io.shapereader.Reader(watershed_shp_fn)
            group_shp_attr = 'watershed'
        elif grouping == 'kaab':
            group_shp = cartopy.io.shapereader.Reader(kaab_shp_fn)
            group_shp_attr = 'kaab_name'
            
        
#        cmap = mpl.cm.RdYlBu
        cmap = 'RdYlBu'
        if vn == 'tempchange':
            cmap = 'RdYlBu_r'
#            cmap = mpl.cm.RdYlBu_r
        norm = plt.Normalize(colorbar_dict[vn][0], colorbar_dict[vn][1])
        
        # Add attribute of interest to the shapefile
        if option_addbackground_group == 1:
            for rec in group_shp.records():
                if rec.attributes[group_shp_attr] in groups:
                    # Group index
                    ds_idx = groups.index(rec.attributes[group_shp_attr])          
                    # Value to plot
                    rec.attributes['value'] = ds_group[ds_idx][1]
        
                    # Add polygon to plot
                    ax.add_geometries(rec.geometry, cartopy.crs.PlateCarree(), 
                                      facecolor=cmap(norm(rec.attributes['value'])),
                                      edgecolor='None', zorder=1)
                    # plot polygon outlines on top of everything with their labels
                    ax.add_geometries(rec.geometry, cartopy.crs.PlateCarree(), facecolor='None', 
                                      edgecolor='grey', linewidth=0.75, zorder=3)
                    ax.text(title_location[rec.attributes[group_shp_attr]][0], 
                            title_location[rec.attributes[group_shp_attr]][1], 
                            title_dict[rec.attributes[group_shp_attr]], horizontalalignment='center', size=12, zorder=4)
                    if rec.attributes[group_shp_attr] == 'Karakoram':
                        ax.plot([72.2, 76.2], [34.3, 35.8], color='black', linewidth=0.75)
                    elif rec.attributes[group_shp_attr] == 'Pamir':
                        ax.plot([69.2, 73], [37.3, 38.3], color='black', linewidth=0.75)
                        
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm._A = []
        plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.01)
        fig.text(0.98, 0.5, vn_dict[vn], va='center', rotation='vertical', size=labelsize)
        
        if option_plot_individual_glaciers == 1:
            # Plot individual glaciers
            area_cutoffs = [100, 10, 1, 0.1]
            area_cutoffs_size = [1000, 100, 10, 2]
            area_sizes = size_thresholds(main_glac_rgi_all.Area.values, area_cutoffs, area_cutoffs_size)
                
            modelparam_value = main_glac_rgi_all[vn].values
            glac_lons = main_glac_rgi_all.CenLon.values
            glac_lats = main_glac_rgi_all.CenLat.values
            sc = ax.scatter(glac_lons, glac_lats, c=modelparam_value, cmap=cmap, norm=norm, s=area_sizes,
                            edgecolor='grey', linewidth=0.25, transform=cartopy.crs.PlateCarree(), zorder=2)
            
            # Add legend for glacier sizes
            legend_glac = []
            legend_glac_labels = []
            legend_glac_markersize = [20, 10, 5, 2]
            for i_area, area_cutoff_size in enumerate(area_cutoffs_size):
                legend_glac.append(Line2D([0], [0], linestyle='None', marker='o', color='grey', 
                                          label=(str(area_cutoffs[i_area]) + 'km$^2$'), 
                                          markerfacecolor='grey', markersize=legend_glac_markersize[i_area]))
            plt.legend(handles=legend_glac, loc='lower left', fontsize=12)
    
        elif option_plot_degrees == 1:
            # Group by degree  
            groups_deg, ds_vn_deg = partition_modelparams_groups('degree', vn, main_glac_rgi_all)
    
            z = [ds_vn_deg[ds_idx][1] for ds_idx in range(len(ds_vn_deg))]
            x = np.array([x[0] for x in deg_groups]) 
            y = np.array([x[1] for x in deg_groups])
            lons = np.arange(x.min(), x.max() + 2 * degree_size, degree_size)
            lats = np.arange(y.min(), y.max() + 2 * degree_size, degree_size)
            x_adj = np.arange(x.min(), x.max() + 1 * degree_size, degree_size) - x.min()
            y_adj = np.arange(y.min(), y.max() + 1 * degree_size, degree_size) - y.min()
            z_array = np.zeros((len(y_adj), len(x_adj)))
            z_array[z_array==0] = np.nan
            for i in range(len(z)):
                row_idx = int((y[i] - y.min()) / degree_size)
                col_idx = int((x[i] - x.min()) / degree_size)
                z_array[row_idx, col_idx] = z[i]
            if vn == 'tempchange':
                ax.pcolormesh(lons, lats, z_array, cmap='RdYlBu_r', norm=norm, zorder=2, alpha=0.8)
            else:
                ax.pcolormesh(lons, lats, z_array, cmap='RdYlBu', norm=norm, zorder=2, alpha=0.8)
            
        
        # Save figure
        fig.set_size_inches(6,4)
        if option_plot_individual_glaciers == 1:
            fig_fn = 'mp_' + vn + '_wglac.png'
        elif option_plot_degrees == 1:
            if degree_size < 1:
                degsize_name = 'pt' + str(int(degree_size * 100))
            else:
                degsize_name = str(degree_size)
            fig_fn = 'mp_' + vn + '_' + degsize_name + 'deg.png'
        else:
            fig_fn = 'mp_' + vn + '_' + grouping + '.png'
        fig.savefig(figure_fp + '../cal/' + fig_fn, bbox_inches='tight', dpi=300)
        
#%% ERA-INTERIM NORMALIZED CHANGE 2000-2018
if option_plot_era_normalizedchange == 1:
    
    vns = ['volume_glac_annual']
    grouping = 'all'
    glac_float = 13.26360
    labelsize = 13
    
    for vn in vns:
        groups, time_values, ds_group, ds_glac = partition_era_groups(grouping, vn, main_glac_rgi_all)
        
        if vn == 'volume_glac_annual':
            group_idx = 0
            vn_norm = ds_group[group_idx][1] / ds_group[group_idx][1][0]
            vn_norm_upper = (ds_group[group_idx][1] + ds_group[group_idx][2]) / ds_group[group_idx][1][0]
            vn_norm_lower = (ds_group[group_idx][1] - ds_group[group_idx][2]) / ds_group[group_idx][1][0]
            
            glac_idx =  np.where(main_glac_rgi_all['RGIId_float'].values == glac_float)[0][0]
            vn_glac_norm = ds_glac[0][glac_idx,:] / ds_glac[0][glac_idx,0]
            vn_glac_norm_upper = (ds_glac[0][glac_idx,:] + ds_glac[1][glac_idx,:]) / ds_glac[0][glac_idx,0] 
            vn_glac_norm_lower = (ds_glac[0][glac_idx,:] - ds_glac[1][glac_idx,:]) / ds_glac[0][glac_idx,0]
            
        fig, ax = plt.subplots(1, 1, squeeze=False, figsize=(10,8), gridspec_kw = {'wspace':0, 'hspace':0})
        # All glaciers
        ax[0,0].plot(time_values, vn_norm, color='k', linewidth=1, label='HMA')
        ax[0,0].fill_between(time_values, vn_norm_lower, vn_norm_upper, facecolor='k', alpha=0.15, label=r'$\pm$1 std')

        # Individual glacier        
        ax[0,0].plot(time_values, vn_glac_norm, color='b', linewidth=1, label=glac_float)
        ax[0,0].fill_between(time_values, vn_glac_norm_lower, vn_glac_norm_upper, facecolor='b', alpha=0.15, label=None)
        
        # Tick parameters
        ax[0,0].tick_params(axis='both', which='major', labelsize=labelsize, direction='inout')
        ax[0,0].tick_params(axis='both', which='minor', labelsize=labelsize, direction='inout')
        # X-label
        ax[0,0].set_xlim(time_values.min(), time_values.max())
        ax[0,0].xaxis.set_tick_params(labelsize=labelsize)
        ax[0,0].xaxis.set_major_locator(plt.MultipleLocator(5))
        ax[0,0].xaxis.set_minor_locator(plt.MultipleLocator(1))
        # Y-label
        ax[0,0].set_ylabel('Normalized volume [-]', fontsize=labelsize+1)
        ax[0,0].yaxis.set_tick_params(labelsize=labelsize)
        ax[0,0].yaxis.set_major_locator(plt.MultipleLocator(0.1))
        ax[0,0].yaxis.set_minor_locator(plt.MultipleLocator(0.02))
        
        # Legend
        ax[0,0].legend(loc='lower left', fontsize=labelsize-2)

        # Save figure
        fig.set_size_inches(5,3)
        glac_float_str = str(glac_float).replace('.','-')
        figure_fn = 'HMA_volchange_wglac' + glac_float_str + '.png'
        
        fig.savefig(cal_fp + '../' + figure_fn, bbox_inches='tight', dpi=300)
        
#%% COMPARE GCM MASS BALANCE 2000-2018 TO CALIBRATION DATA
if option_compare_GCMwCal == 1:

    change_fp = '/Users/davidrounce/Documents/Dave_Rounce/HiMAT/Output/simulations/spc_vars_20180109_changeArea/'
    constant_fp = '/Users/davidrounce/Documents/Dave_Rounce/HiMAT/Output/simulations/spc_vars_20180109_constantArea/'
    change_fp_era = '/Users/davidrounce/Documents/Dave_Rounce/HiMAT/Output/simulations/spc_vars_era_chgArea/'
    constant_fp_era = '/Users/davidrounce/Documents/Dave_Rounce/HiMAT/Output/simulations/spc_vars_era_conArea_100sims/'
    
    gcm_names = ['CanESM2', 'CCSM4', 'CNRM-CM5', 'CSIRO-Mk3-6-0',  'GFDL-CM3', 'GFDL-ESM2M', 'GISS-E2-R', 
                 'IPSL-CM5A-LR', 'NorESM1-M']
    rcps = ['rcp26']
    region = 13

    netcdf_fp = netcdf_fp_cmip5
    
    # Glacier hypsometry [km**2], total area
    main_glac_hyps_raw = modelsetup.import_Husstable(main_glac_rgi_all, pygem_prms.hyps_filepath,
                                                     pygem_prms.hyps_filedict, pygem_prms.hyps_colsdrop)
    dates_table_nospinup  = modelsetup.datesmodelrun(startyear=pygem_prms.startyear, endyear=pygem_prms.endyear, 
                                                     spinupyears=0)
    cal_data = pd.DataFrame()
    for dataset in pygem_prms.cal_datasets:
        cal_subset = class_mbdata.MBData(name=dataset)
        cal_subset_data = cal_subset.retrieve_mb(main_glac_rgi_all, main_glac_hyps_raw, dates_table_nospinup)
        cal_data = cal_data.append(cal_subset_data, ignore_index=True)
    cal_data = cal_data.sort_values(['glacno', 't1_idx'])
    cal_data.reset_index(drop=True, inplace=True)
#%%
    def retrieve_mb_mwea_all(netcdf_fp, gcm, rcp=None):
        vn_area = 'area_glac_annual'
        if gcm == 'ERA-Interim':
            try:
                ds_area_fn = 'R' + str(region) + '_' + gcm + '_c2_ba1_1sets_2000_2018--' + vn_area + '.nc'
                ds_area = xr.open_dataset(netcdf_fp + vn_area + '/' + ds_area_fn)
            except:
                ds_area_fn = 'R' + str(region) + '_' + gcm + '_c2_ba1_100sets_2000_2018--' + vn_area + '.nc'
                ds_area = xr.open_dataset(netcdf_fp + vn_area + '/' + ds_area_fn)
        else:
            ds_area_fn = 'R' + str(region) + '_' + gcm + '_' + rcp + '_c2_ba1_1sets_2000_2018--' + vn_area + '.nc'
            ds_area = xr.open_dataset(netcdf_fp + vn_area + '/' + ds_area_fn)
        
        vn_mb = 'massbaltotal_glac_monthly'
        if gcm == 'ERA-Interim':
            try:
                ds_mb_fn = 'R' + str(region) + '_' + gcm + '_c2_ba1_1sets_2000_2018--' + vn_mb + '.nc'
                ds_mb = xr.open_dataset(netcdf_fp + vn_mb + '/' + ds_mb_fn)
            except:
                ds_mb_fn = 'R' + str(region) + '_' + gcm + '_c2_ba1_100sets_2000_2018--' + vn_mb + '.nc'
                ds_mb = xr.open_dataset(netcdf_fp + vn_mb + '/' + ds_mb_fn)
        else:
            ds_mb_fn = 'R' + str(region) + '_' + gcm + '_' + rcp + '_c2_ba1_1sets_2000_2018--' + vn_mb + '.nc'
            ds_mb = xr.open_dataset(netcdf_fp + vn_mb + '/' + ds_mb_fn)
        
        print(ds_mb_fn)
    
        # Convert to mass balance
        mb_monthly_all = ds_mb[vn_mb].values[:,:,0]
        area_all = ds_area[vn_area].values[:,:,0]
#        time_values = ds_mb.year_plus1.values
#        time_values_mb = time_values[:-1]
        mb_annual_all = gcmbiasadj.annual_sum_2darray(mb_monthly_all)
        mb_2000_2018_mwea_all = (mb_annual_all * area_all[:,:-1]).sum(axis=1) / area_all[:,0] / 18
        
        # Close datasets
        ds_area.close()
        ds_mb.close()
        
        return mb_2000_2018_mwea_all

    #%%
    gcm_names = ['CanESM2']
    for gcm in gcm_names:
        for rcp in rcps:
            
            mb_2000_2018_mwea_all_constantArea = retrieve_mb_mwea_all(constant_fp, gcm, rcp)
            mb_2000_2018_mwea_all_changeArea = retrieve_mb_mwea_all(change_fp, gcm, rcp)
            
            mb_2000_2018_mwea_all_cal = cal_data.mb_mwe.values / 18
            
#            mb_compare = pd.DataFrame(np.stack((mb_2000_2018_mwea_all_cal, mb_2000_2018_mwea_all_constantArea,
#                                mb_2000_2018_mwea_all_changeArea), axis=1), 
#                                columns=['cal_mb_mwea', 'constA_mb_mwea', 'chgA_mb_mwea'])
#            mb_compare['Const-Chg'] = mb_compare.constA_mb_mwea - mb_compare.chgA_mb_mwea
#            mb_compare['cal-const'] = mb_compare.cal_mb_mwea - mb_compare.constA_mb_mwea
#            print(gcm, rcp, '\nCal-Const median:', np.round(np.nanmedian(mb_compare['cal-const']),2),
#                  '\nCal-Const mean:', np.round(np.nanmean(mb_compare['cal-const']),2),
#                  '\nCal-Const std:', np.round(np.nanstd(mb_compare['cal-const']),2),
#                  '\nCal-Const min:', np.round(np.nanmin(mb_compare['cal-const']),2),
#                  '\nCal-Const max:', np.round(np.nanmax(mb_compare['cal-const']),2),
#                  '\nDif due to Area Change (median, std):', np.round(np.nanmedian(mb_compare['Const-Chg']),2),
#                  np.round(np.nanstd(mb_compare['Const-Chg']),2))
            
    for gcm in ['ERA-Interim']:
        mb_2000_2018_mwea_all_constantArea_era = retrieve_mb_mwea_all(constant_fp_era, gcm)
        mb_2000_2018_mwea_all_changeArea_era = retrieve_mb_mwea_all(change_fp_era, gcm)
        
        mb_compare = pd.DataFrame(np.stack((mb_2000_2018_mwea_all_cal, mb_2000_2018_mwea_all_constantArea_era,
                            mb_2000_2018_mwea_all_changeArea_era, mb_2000_2018_mwea_all_constantArea,
                            mb_2000_2018_mwea_all_changeArea), axis=1), 
                            columns=['cal_mb_mwea', 'constA_mb_mwea_era', 'chgA_mb_mwea_era', 'constA_mb_mwea', 
                                     'chgA_mb_mwea'])
        
        mb_compare['era-gcm'] = mb_compare['constA_mb_mwea_era'] - mb_compare['constA_mb_mwea']
#        mb_compare['cal-const'] = mb_compare.cal_mb_mwea_era - mb_compare.constA_mb_mwea_era
        print(gcm, rcp, '\nERA - GCM median:', np.round(np.nanmedian(mb_compare['era-gcm']),2),
              '\nera-gcm mean:', np.round(np.nanmean(mb_compare['era-gcm']),2),
              '\nera-gcm std:', np.round(np.nanstd(mb_compare['era-gcm']),2),
              '\nera-gcm min:', np.round(np.nanmin(mb_compare['era-gcm']),2),
              '\nera-gcm max:', np.round(np.nanmax(mb_compare['era-gcm']),2))
        

#%%
if option_plot_mcmc_errors == 1:
    print('plot mcmc errors for all of HMA:\n1. Spatial distribution - do certain regions perform poorly' + 
          '\n2. How many chains are actually "good" and how many are bad?')
    print(mcmc_fp)
    

    # Load cal data
    ds_cal = pd.read_csv(pygem_prms.shean_fp + pygem_prms.shean_fn)
    # Glacier number and index for comparison
    ds_cal['O1region'] = ds_cal['RGIId'].astype(int)
    ds_cal['glacno'] = ((ds_cal['RGIId'] % 1) * 10**5).round(0).astype(int)
    ds_cal['RGIId_full'] = ('RGI60-' + ds_cal['O1region'].map(str) + '.' + 
                           (ds_cal['glacno'] / 10**5).apply(lambda x: '%.5f' % x).astype(str).str.split('.').str[1])
    
    # Add calibration data to main_glac_rgi_all
    rand_idx = np.random.randint(0,main_glac_rgi_all.shape[0])
    if (main_glac_rgi_all.shape[0] == ds_cal.shape[0] and 
        (ds_cal.loc[rand_idx,'RGIId_full'] == main_glac_rgi_all.loc[rand_idx, 'RGIId'])):
        main_glac_rgi_all['mb_cal_mwea'] = ds_cal['mb_mwea'] 
        main_glac_rgi_all['mb_cal_sigma'] = ds_cal['mb_mwea_sigma'] 
    else:
        main_glac_rgi_all['mb_cal_mwea'] = np.nan
        for glac in range(main_glac_rgi_all.shape[0]):
            cal_idx = np.where(ds_cal['RGIId_full'] == main_glac_rgi_all.loc[glac, 'RGIId'])[0][0]
            main_glac_rgi_all.loc[glac,'mb_cal_mwea'] = ds_cal.loc[cal_idx,'mb_mwea']
            main_glac_rgi_all.loc[glac,'mb_cal_sigma'] = ds_cal.loc[cal_idx,'mb_mwea_sigma'] 


    #%%
    # Load ERA-Interim modeled mass balance to data
#    if (main_glac_rgi_all.shape[0] == ds_cal.shape[0] and 
#        (ds_cal.loc[rand_idx,'RGIId_full'] == main_glac_rgi_all.loc[rand_idx, 'RGIId'])):
#        df_mean_all = pd.read_csv(shean_fp + 'mcmc_mean_all.csv', index_col=0)
#        df_median_all = pd.read_csv(shean_fp + 'mcmc_median_all.csv', index_col=0)
#    else:
    
    burn_no = 200
    thin_interval = 1 
    print('\nBURN NUMBER:', burn_no,'\n')
    #%%
    # Load data for each glacier
    mcmc_cns = ['RGIId', 'massbal', 'precfactor', 'tempchange', 'ddfsnow', 'ddfice', 'lrgcm', 'lrglac', 'precgrad', 
                'tempsnow']
    df_cns = ['massbal', 'precfactor', 'tempchange', 'ddfsnow', 'ddfice', 'lrgcm', 'lrglac', 'precgrad', 'tempsnow']
    df_mean_all = pd.DataFrame(np.zeros((main_glac_rgi_all.shape[0], len(mcmc_cns))), columns=mcmc_cns)
    df_median_all = pd.DataFrame(np.zeros((main_glac_rgi_all.shape[0], len(mcmc_cns))), columns=mcmc_cns)
    df_std_all = pd.DataFrame(np.zeros((main_glac_rgi_all.shape[0], len(mcmc_cns))), columns=mcmc_cns)
    df_min_all = pd.DataFrame(np.zeros((main_glac_rgi_all.shape[0], len(mcmc_cns))), columns=mcmc_cns)
    df_max_all = pd.DataFrame(np.zeros((main_glac_rgi_all.shape[0], len(mcmc_cns))), columns=mcmc_cns)
    for n, glac_str_wRGI in enumerate(main_glac_rgi_all['RGIId'].values):
        if n%500 == 0:
            print(n, glac_str_wRGI)
        # Glacier string
        glacier_str = glac_str_wRGI.split('-')[1]
        ds = xr.open_dataset(mcmc_fp + glacier_str + '.nc')
        df = pd.DataFrame(ds['mp_value'].values[burn_no::thin_interval,:,0], columns=ds.mp.values)
        df = df[df_cns]
        df_mean_all.loc[n,df_cns] = df.mean()
        df_mean_all.loc[n,'RGIId'] = glac_str_wRGI
        df_median_all.loc[n,:] = df.median()
        df_median_all.loc[n,'RGIId'] = glac_str_wRGI
        df_std_all.loc[n,:] = df.std()
        df_std_all.loc[n,'RGIId'] = glac_str_wRGI
        df_min_all.loc[n,:] = df.min()
        df_min_all.loc[n,'RGIId'] = glac_str_wRGI
        df_max_all.loc[n,:] = df.max()
        df_max_all.loc[n,'RGIId'] = glac_str_wRGI
        
        ds.close()

    
    #%%
    # Maximum loss if entire glacier melted between 2000 and 2018
    mb_max_loss = (-1 * (main_glac_hyps_all * main_glac_icethickness_all * pygem_prms.density_ice / 
                   pygem_prms.density_water).sum(axis=1).values / main_glac_hyps_all.sum(axis=1).values / (2018 - 2000))
    main_glac_rgi_all['mb_max_loss'] = mb_max_loss

#    # Truncated normal - updated means to remove portions of observations that are below max mass loss!
#    main_glac_rgi_all['mb_cal_dist_mean'] = np.nan
#    main_glac_rgi_all['mb_cal_dist_med'] = np.nan
#    main_glac_rgi_all['mb_cal_dist_std'] = np.nan
#    main_glac_rgi_all['mb_cal_dist_95low'] = np.nan
#    main_glac_rgi_all['mb_cal_dist_95high']  = np.nan
#    for glac in range(main_glac_rgi_all.shape[0]):
#        cal_mu = main_glac_rgi_all.loc[glac, 'mb_cal_mwea']
#        cal_sigma = main_glac_rgi_all.loc[glac, 'mb_cal_sigma']
#        cal_lowbnd = main_glac_rgi_all.loc[glac, 'mb_max_loss']
#        cal_rvs = stats.truncnorm.rvs((cal_lowbnd - cal_mu) / cal_sigma, np.inf, loc=cal_mu, scale=cal_sigma, 
#                                      size=int(1e5))
#        main_glac_rgi_all.loc[glac,'mb_cal_dist_mean'] = np.mean(cal_rvs)
#        main_glac_rgi_all.loc[glac,'mb_cal_dist_med'] = np.median(cal_rvs)
#        main_glac_rgi_all.loc[glac,'mb_cal_dist_std'] = np.std(cal_rvs)
#        main_glac_rgi_all.loc[glac,'mb_cal_dist_95low'] = np.percentile(cal_rvs, 2.5)
#        main_glac_rgi_all.loc[glac,'mb_cal_dist_95high']  = np.percentile(cal_rvs, 97.5)        

    # Add to main_glac_rgi
    if (main_glac_rgi_all.shape[0] == df_mean_all.shape[0] and 
        (df_mean_all.loc[rand_idx,'RGIId'] == main_glac_rgi_all.loc[rand_idx, 'RGIId'])):
        main_glac_rgi_all['mb_era_mean'] = df_mean_all['massbal']
        main_glac_rgi_all['mb_era_med'] = df_median_all['massbal']
        main_glac_rgi_all['mb_era_std'] = df_std_all['massbal']
        main_glac_rgi_all['mb_era_min'] = df_min_all['massbal']
        main_glac_rgi_all['mb_era_max'] = df_max_all['massbal']


#    main_glac_rgi_all['dif_mean_med'] = main_glac_rgi_all['mb_era_mean'] - main_glac_rgi_all['mb_era_med']
    main_glac_rgi_all['dif_cal_era_mean'] = main_glac_rgi_all['mb_cal_mwea'] - main_glac_rgi_all['mb_era_mean']
    main_glac_rgi_all['dif_cal_era_med'] = main_glac_rgi_all['mb_cal_mwea'] - main_glac_rgi_all['mb_era_med']

    # remove nan values
    main_glac_rgi_all = (
            main_glac_rgi_all.drop(np.where(np.isnan(main_glac_rgi_all['mb_era_mean'].values) == True)[0].tolist(), 
                                   axis=0))
    main_glac_rgi_all.reset_index(drop=True, inplace=True)
    

    def plot_hist(df, cn, bins, xlabel=None, ylabel=None, fig_fn='hist.png', fig_fp=figure_fp):
        """
        Plot histogram for any bin size
        """           
        data = df[cn].values
        hist, bin_edges = np.histogram(data,bins) # make the histogram
        fig,ax = plt.subplots()    
        # Plot the histogram heights against integers on the x axis
        ax.bar(range(len(hist)),hist,width=1, edgecolor='k') 
        # Set the ticks to the middle of the bars
        ax.set_xticks([0.5+i for i,j in enumerate(hist)])
        # Set the xticklabels to a string that tells us what the bin edges were
        ax.set_xticklabels(['{} - {}'.format(bins[i],bins[i+1]) for i,j in enumerate(hist)], rotation=45, ha='right')
        ax.set_xlabel(xlabel, fontsize=16)
        ax.set_ylabel(ylabel, fontsize=16)
        # Save figure
        fig.set_size_inches(6,4)
        fig.savefig(fig_fp + fig_fn, bbox_inches='tight', dpi=300)

    # Histogram: Mass balance [mwea], Observation - ERA
    hist_cn = 'dif_cal_era_mean'
    low_bin = np.floor(main_glac_rgi_all[hist_cn].min())
    high_bin = np.ceil(main_glac_rgi_all[hist_cn].max())
    bins = [low_bin, -0.2, -0.1, -0.05, -0.02, 0.02, 0.05, 0.1, 0.2, high_bin]
    plot_hist(main_glac_rgi_all, hist_cn, bins, xlabel='Mass balance [mwea]\n(Calibration - MCMC_mean)', 
              ylabel='# Glaciers', fig_fn='MB_cal_vs_mcmc_hist.png', fig_fp=figure_fp + '../cal/')
              
         
    # Histogram: Mass balance [mwea], MCMC mean - median
    hist_cn = 'dif_cal_era_med'
    low_bin = np.floor(main_glac_rgi_all[hist_cn].min())
    high_bin = np.ceil(main_glac_rgi_all[hist_cn].max() + 1)
    bins = [low_bin, -0.2, -0.1, -0.05, -0.02, 0.02, 0.05, 0.1, 0.2, high_bin]
    plot_hist(main_glac_rgi_all, hist_cn, bins, xlabel='MCMC Mass balance [mwea]\n(Calibration - MCMC_median)', 
              ylabel='# Glaciers', fig_fn='MB_mean_vs_med_mcmc_hist.png', fig_fp=figure_fp + '../cal/')
              
    # Histogram: Glacier Area [km2]
    hist_cn = 'Area'
    low_bin = np.floor(main_glac_rgi_all[hist_cn].min())
    high_bin = np.ceil(main_glac_rgi_all[hist_cn].max() + 1)
    bins = [0, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 25, 50, 100, high_bin]
    plot_hist(main_glac_rgi_all, hist_cn, bins, xlabel='Glacier Area [km2]', 
              ylabel='# Glaciers', fig_fn='Glacier_Area.png', fig_fp=figure_fp + '../cal/')
    
    # Map: Mass change, difference between calibration data and median data
    #  Area [km2] * mb [mwe] * (1 km / 1000 m) * density_water [kg/m3] * (1 Gt/km3  /  1000 kg/m3)
    main_glac_rgi_all['mb_cal_Gta'] = main_glac_rgi_all['mb_cal_mwea'] * main_glac_rgi_all['Area'] / 1000
    main_glac_rgi_all['mb_cal_Gta_var'] = (main_glac_rgi_all['mb_cal_sigma'] * main_glac_rgi_all['Area'] / 1000)**2
    main_glac_rgi_all['mb_era_Gta'] = main_glac_rgi_all['mb_era_mean'] * main_glac_rgi_all['Area'] / 1000
    main_glac_rgi_all['mb_era_Gta_var'] = (main_glac_rgi_all['mb_era_std'] * main_glac_rgi_all['Area'] / 1000)**2
    main_glac_rgi_all['mb_era_Gta_med'] = main_glac_rgi_all['mb_era_med'] * main_glac_rgi_all['Area'] / 1000
    print('All MB cal (mean +/- 1 std) [gt/yr]:', np.round(main_glac_rgi_all['mb_cal_Gta'].sum(),3), 
          '+/-', np.round(main_glac_rgi_all['mb_cal_Gta_var'].sum()**0.5,3),
          '\nAll MB ERA (mean +/- 1 std) [gt/yr]:', np.round(main_glac_rgi_all['mb_era_Gta'].sum(),3), 
          '+/-', np.round(main_glac_rgi_all['mb_era_Gta_var'].sum()**0.5,3),
          '\nAll MB ERA (med) [gt/yr]:', np.round(main_glac_rgi_all['mb_era_Gta_med'].sum(),3))

    def partition_sum_groups(grouping, vn, main_glac_rgi_all):
        """Partition model parameters by each group
        
        Parameters
        ----------
        grouping : str
            name of grouping to use
        vn : str
            variable name
        main_glac_rgi_all : pd.DataFrame
            glacier table
            
        Output
        ------
        groups : list
            list of group names
        ds_group : list of lists
            dataset containing the multimodel data for a given variable for all the GCMs
        """
        # Groups
        groups, group_cn = select_groups(grouping, main_glac_rgi_all)
        
        ds_group = [[] for group in groups]
        
        # Cycle through groups
        for ngroup, group in enumerate(groups):
            # Select subset of data
            main_glac_rgi = main_glac_rgi_all.loc[main_glac_rgi_all[group_cn] == group]                        
            vn_glac = main_glac_rgi_all[vn].values[main_glac_rgi.index.values.tolist()]             
            # Regional sum
            vn_reg = vn_glac.sum(axis=0)                
            
            # Record data for each group
            ds_group[ngroup] = [group, vn_reg]
                
        return groups, ds_group

    grouping='degree'
    
    groups, ds_group_cal = partition_sum_groups(grouping, 'mb_cal_Gta', main_glac_rgi_all)
    groups, ds_group_era = partition_sum_groups(grouping, 'mb_era_Gta', main_glac_rgi_all)
    groups, ds_group_area = partition_sum_groups(grouping, 'Area', main_glac_rgi_all)
    
#    ds_group_dif = [[] for x in ds_group_cal ]

    # Group difference [Gt/yr]
    dif_cal_era_Gta = (np.array([x[1] for x in ds_group_cal]) - np.array([x[1] for x in ds_group_era])).tolist()
    ds_group_dif_cal_era_Gta = [[x[0],dif_cal_era_Gta[n]] for n, x in enumerate(ds_group_cal)]
    # Group difference [mwea]
    area = [x[1] for x in ds_group_area]
    ds_group_dif_cal_era_mwea = [[x[0], dif_cal_era_Gta[n] / area[n] * 1000] for n, x in enumerate(ds_group_cal)]

    fig_fn = 'MB_cal_vs_mcmc_map.png'
    
    east = 104
    west = 67
    south = 25
    north = 48
    
    labelsize = 13
    
    norm = plt.Normalize(-0.1, 0.1)
    
    # Create the projection
    fig, ax = plt.subplots(1, 1, figsize=(10,5), subplot_kw={'projection':cartopy.crs.PlateCarree()})
    # Add country borders for reference
    ax.add_feature(cartopy.feature.BORDERS, alpha=0.15, zorder=10)
    ax.add_feature(cartopy.feature.COASTLINE)
    # Set the extent
    ax.set_extent([east, west, south, north], cartopy.crs.PlateCarree())    
    # Label title, x, and y axes
    ax.set_xticks(np.arange(east,west+1,xtick), cartopy.crs.PlateCarree())
    ax.set_yticks(np.arange(south,north+1,ytick), cartopy.crs.PlateCarree())
    ax.set_xlabel(xlabel, size=labelsize)
    ax.set_ylabel(ylabel, size=labelsize)            
    
    cmap = 'RdYlBu_r'

    # Add colorbar
#    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm._A = []
    plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.01)
    fig.text(1, 0.5, 'Mass balance [mwea]\n(Observation - MCMC)', va='center', rotation='vertical', size=labelsize)    

    # Group by degree  
    groups_deg = groups
    ds_vn_deg = ds_group_dif_cal_era_mwea

    z = [ds_vn_deg[ds_idx][1] for ds_idx in range(len(ds_vn_deg))]
    x = np.array([x[0] for x in deg_groups]) 
    y = np.array([x[1] for x in deg_groups])
    lons = np.arange(x.min(), x.max() + 2 * degree_size, degree_size)
    lats = np.arange(y.min(), y.max() + 2 * degree_size, degree_size)
    x_adj = np.arange(x.min(), x.max() + 1 * degree_size, degree_size) - x.min()
    y_adj = np.arange(y.min(), y.max() + 1 * degree_size, degree_size) - y.min()
    z_array = np.zeros((len(y_adj), len(x_adj)))
    z_array[z_array==0] = np.nan
    for i in range(len(z)):
        row_idx = int((y[i] - y.min()) / degree_size)
        col_idx = int((x[i] - x.min()) / degree_size)
        z_array[row_idx, col_idx] = z[i]    
    ax.pcolormesh(lons, lats, z_array, cmap='RdYlBu_r', norm=norm, zorder=2, alpha=0.8)
    
    # Save figure
    fig.set_size_inches(6,4)
    fig.savefig(figure_fp + '../cal/' + fig_fn, bbox_inches='tight', dpi=300)
    
    main_glac_rgi_all.to_csv(pygem_prms.output_filepath + 'main_glac_rgi_HMA_20190308_adjp12_100iters.csv')
    

#%%
if option_plot_maxloss_issues == 1:

#    # Load cal data
#    shean_fp = pygem_prms.main_directory + '/../DEMs/Shean_2018_1109/'
#    shean_fn = 'hma_mb_20181108_0454_all_filled.csv'
    #%%
#    shean_fp = pygem_prms.main_directory + '/../DEMs/Shean_2019_0213/'
    #shean_fn = 'hma_mb_20190215_0815_std+mean.csv'
#    shean_fn = 'hma_mb_20190215_0815_std+mean_all_filled.csv'

    ds_cal = pd.read_csv(pygem_prms.shean_fp + pygem_prms.shean_fn)
    # Glacier number and index for comparison
    ds_cal['O1region'] = ds_cal['RGIId'].astype(int)
    ds_cal['glacno'] = ((ds_cal['RGIId'] % 1) * 10**5).round(0).astype(int)
    ds_cal['RGIId_full'] = ('RGI60-' + ds_cal['O1region'].map(str) + '.' + 
                           (ds_cal['glacno'] / 10**5).apply(lambda x: '%.5f' % x).astype(str).str.split('.').str[1])
    #%%
    
    # Add calibration data to main_glac_rgi_all
    main_glac_rgi_all['mb_cal_mwea'] = np.nan
    for glac in range(main_glac_rgi_all.shape[0]):
#    for glac in [6718]:
#        print(main_glac_rgi_all.loc[glac,'RGIId'])
        cal_idx = np.where(ds_cal['RGIId_full'] == main_glac_rgi_all.loc[glac, 'RGIId'])[0][0]
        main_glac_rgi_all.loc[glac,'mb_cal_mwea'] = ds_cal.loc[cal_idx,'mb_mwea']
        main_glac_rgi_all.loc[glac,'mb_cal_sigma'] = ds_cal.loc[cal_idx,'mb_mwea_sigma'] 
#        print(main_glac_rgi_all.loc[glac,'mb_cal_mwea'])
    #%%


    # Maximum loss if entire glacier melted between 2000 and 2018
    mb_max_loss = (-1 * (main_glac_hyps_all * main_glac_icethickness_all * pygem_prms.density_ice / 
                   pygem_prms.density_water).sum(axis=1).values / main_glac_hyps_all.sum(axis=1).values / (2018 - 2000))
    main_glac_rgi_all['mb_max_loss'] = mb_max_loss

    # Z-score of where max loss falls on mb_cal
    main_glac_rgi_all['max_loss_Z'] = ((main_glac_rgi_all['mb_max_loss'] - main_glac_rgi_all['mb_cal_mwea']) /
                                       main_glac_rgi_all['mb_cal_sigma'])
    
    # Truncated normal - updated means to remove portions of observations that are below max mass loss!
    main_glac_rgi_all['mb_cal_dist_mean'] = np.nan
    main_glac_rgi_all['mb_cal_dist_med'] = np.nan
    main_glac_rgi_all['mb_cal_dist_std'] = np.nan
    for glac in range(main_glac_rgi_all.shape[0]):
        cal_mu = main_glac_rgi_all.loc[glac, 'mb_cal_mwea']
        cal_sigma = main_glac_rgi_all.loc[glac, 'mb_cal_sigma']
        cal_lowbnd = main_glac_rgi_all.loc[glac, 'mb_max_loss']
        cal_rvs = stats.truncnorm.rvs((cal_lowbnd - cal_mu) / cal_sigma, np.inf, loc=cal_mu, scale=cal_sigma, 
                                      size=int(1e5))
        main_glac_rgi_all.loc[glac,'mb_cal_dist_mean'] = np.mean(cal_rvs)
        main_glac_rgi_all.loc[glac,'mb_cal_dist_med'] = np.median(cal_rvs)
        main_glac_rgi_all.loc[glac,'mb_cal_dist_std'] = np.std(cal_rvs)    
    
    # remove nan values
    main_glac_rgi_all = (
            main_glac_rgi_all.drop(np.where(np.isnan(main_glac_rgi_all['max_loss_Z'].values) == True)[0].tolist(), 
                                   axis=0))
    main_glac_rgi_all.reset_index(drop=True, inplace=True)
    
    #%%

    def plot_hist(df, cn, bins, xlabel=None, ylabel=None, fig_fn='hist.png', fig_fp=figure_fp):
        """
        Plot histogram for any bin size
        """           
        data = df[cn].values
        hist, bin_edges = np.histogram(data,bins) # make the histogram
        fig,ax = plt.subplots()    
        # Plot the histogram heights against integers on the x axis
        ax.bar(range(len(hist)),hist,width=1, edgecolor='k') 
        # Set the ticks to the middle of the bars
        ax.set_xticks([0.5+i for i,j in enumerate(hist)])
        # Set the xticklabels to a string that tells us what the bin edges were
        ax.set_xticklabels(['{} - {}'.format(bins[i],bins[i+1]) for i,j in enumerate(hist)], rotation=45, ha='right')
        ax.set_xlabel(xlabel, fontsize=16)
        ax.set_ylabel(ylabel, fontsize=16)
        ax.set_ylim(0,10000)
        # Save figure
        fig.set_size_inches(6,4)
        fig.savefig(fig_fp + fig_fn, bbox_inches='tight', dpi=300)

    # Histogram: Mass balance [mwea], Observation - ERA
    hist_cn = 'max_loss_Z'
    low_bin = np.floor(main_glac_rgi_all[hist_cn].min())
    high_bin = np.ceil(main_glac_rgi_all[hist_cn].max())
    bins = [low_bin, -2, -1.5, -1, -0.5, 0, 0.5, 1, 2, high_bin]
    plot_hist(main_glac_rgi_all, hist_cn, bins, xlabel='Max Loss Z-score \n[(Max_Loss - Obs_MB) / Obs_sigma]', 
              ylabel='# Glaciers', fig_fn='maxloss_zscore_hist.png', fig_fp=figure_fp + '../cal/')
              
         

#%%    
    # Map: Mass change, difference between calibration data and median data
    def partition_sum_groups(grouping, vn, main_glac_rgi_all):
        """Partition model parameters by each group
        
        Parameters
        ----------
        grouping : str
            name of grouping to use
        vn : str
            variable name
        main_glac_rgi_all : pd.DataFrame
            glacier table
            
        Output
        ------
        groups : list
            list of group names
        ds_group : list of lists
            dataset containing the multimodel data for a given variable for all the GCMs
        """
        # Groups
        groups, group_cn = select_groups(grouping, main_glac_rgi_all)
        
        ds_group = [[] for group in groups]
        
        # Cycle through groups
        for ngroup, group in enumerate(groups):
            # Select subset of data
            main_glac_rgi = main_glac_rgi_all.loc[main_glac_rgi_all[group_cn] == group]                        
            vn_glac = main_glac_rgi_all[vn].values[main_glac_rgi.index.values.tolist()]             
            # Regional sum
            vn_reg = vn_glac.sum(axis=0)                
            
            # Record data for each group
            ds_group[ngroup] = [group, vn_reg]
                
        return groups, ds_group

    grouping='degree'
    
    main_glac_rgi_all['count'] = 1
    main_glac_rgi_all['count_gtNeg2'] = 0
    main_glac_rgi_all['count_gtNeg1'] = 0
    main_glac_rgi_all['count_gt0'] = 0
    main_glac_rgi_all.loc[main_glac_rgi_all['max_loss_Z'] > -2, 'count_gtNeg2'] = 1
    main_glac_rgi_all.loc[main_glac_rgi_all['max_loss_Z'] > -1, 'count_gtNeg1'] = 1
    main_glac_rgi_all.loc[main_glac_rgi_all['max_loss_Z'] > 0, 'count_gt0'] = 1
    groups, ds_group_zscore = partition_sum_groups(grouping, 'count_gt0', main_glac_rgi_all)
    groups, ds_group_count = partition_sum_groups(grouping, 'count', main_glac_rgi_all)
    
    ds_group_zscore_perc = [[x[0], ds_group_zscore[n][1] / ds_group_count[n][1] * 100] 
                            for n, x in enumerate(ds_group_zscore)]
    
    fig_fn = 'max_loss_zscore_map.png'
    
    east = 104
    west = 67
    south = 25
    north = 48
    
    labelsize = 13
    
    norm = plt.Normalize(0, 10)
#    norm = plt.Normalize(0, 100)
    
    # Create the projection
    fig, ax = plt.subplots(1, 1, figsize=(10,5), subplot_kw={'projection':cartopy.crs.PlateCarree()})
    # Add country borders for reference
    ax.add_feature(cartopy.feature.BORDERS, alpha=0.15, zorder=10)
    ax.add_feature(cartopy.feature.COASTLINE)
    # Set the extent
    ax.set_extent([east, west, south, north], cartopy.crs.PlateCarree())    
    # Label title, x, and y axes
    ax.set_xticks(np.arange(east,west+1,xtick), cartopy.crs.PlateCarree())
    ax.set_yticks(np.arange(south,north+1,ytick), cartopy.crs.PlateCarree())
    ax.set_xlabel(xlabel, size=labelsize)
    ax.set_ylabel(ylabel, size=labelsize)            
    
    cmap = 'RdYlBu_r'

    # Add colorbar
#    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm._A = []
    plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.01)
    fig.text(1, 0.5, 'Count', va='center', rotation='vertical', size=labelsize)    

    # Group by degree  
    groups_deg = groups
    ds_vn_deg = ds_group_zscore
#    ds_vn_deg = ds_group_zscore_perc

    z = [ds_vn_deg[ds_idx][1] for ds_idx in range(len(ds_vn_deg))]
    x = np.array([x[0] for x in deg_groups]) 
    y = np.array([x[1] for x in deg_groups])
    lons = np.arange(x.min(), x.max() + 2 * degree_size, degree_size)
    lats = np.arange(y.min(), y.max() + 2 * degree_size, degree_size)
    x_adj = np.arange(x.min(), x.max() + 1 * degree_size, degree_size) - x.min()
    y_adj = np.arange(y.min(), y.max() + 1 * degree_size, degree_size) - y.min()
    z_array = np.zeros((len(y_adj), len(x_adj)))
    z_array[z_array==0] = np.nan
    for i in range(len(z)):
        row_idx = int((y[i] - y.min()) / degree_size)
        col_idx = int((x[i] - x.min()) / degree_size)
        z_array[row_idx, col_idx] = z[i]    
    ax.pcolormesh(lons, lats, z_array, cmap='RdYlBu_r', norm=norm, zorder=2, alpha=0.8)
    
    # Save figure
    fig.set_size_inches(6,4)
    fig.savefig(figure_fp + '../cal/' + fig_fn, bbox_inches='tight', dpi=300)
    
#%%
#main_glac_rgi_all['cal_norm'] = (main_glac_rgi_all.mb_cal_mwea - main_glac_rgi_all.mb_max_loss) / main_glac_rgi_all.mb_cal_sigma
#A = main_glac_rgi_all.copy()
#A.plot.scatter('Area', 'dif_cal_era_mean')

#%%

# Compute mass change from 2000 - 2018
    
## variable name
#vn = 'volume_glac_annual'
#rgi_regions = [13, 14, 15]
#
#for region in rgi_regions:
#    # Load datasets
#    ds_fn = 'R' + str(region) + '_ERA-Interim_c2_ba1_100sets_1980_2017.nc'
#    ds = xr.open_dataset(netcdf_fp_era + ds_fn)
#    # Extract time variable
#    if 'annual' in vn:
#        try:
#            time_values = ds[vn].coords['year_plus1'].values
#        except:
#            time_values = ds[vn].coords['year'].values
#    elif 'monthly' in vn:
#        time_values = ds[vn].coords['time'].values
#        
#    # Merge datasets
#    if region == rgi_regions[0]:
#        vn_glac_all = ds[vn].values[:,:,0]
#        vn_glac_std_all = ds[vn].values[:,:,1]
#    else:
#        vn_glac_all = np.concatenate((vn_glac_all, ds[vn].values[:,:,0]), axis=0)
#        vn_glac_std_all = np.concatenate((vn_glac_std_all, ds[vn].values[:,:,1]), axis=0)
#    
#    # Close dataset
#    ds.close()
#    
## Mass change for text on plot
##  Gt = km3 ice * density_ice / 1000
##  divide by 1000 because density of ice is 900 kg/m3 or 0.900 Gt/km3
#vn_reg_masschange = (vn_reg[-1] - vn_reg[0]) * pygem_prms.density_ice / 1000
#
#A = (vn_glac_all[:,20].sum() - vn_glac_all[:,-1].sum()) * pygem_prms.density_ice / 1000 / 18
