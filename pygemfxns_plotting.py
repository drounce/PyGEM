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
import cartopy
#import geopandas
import xarray as xr
from osgeo import ogr
# Local Libraries
import pygem_input as input
import pygemfxns_modelsetup as modelsetup
import pygemfxns_massbalance as massbalance
import class_mbdata
import class_climate
import run_simulation


# Script options
option_plot_cmip5_normalizedchange = 0
option_plot_cmip5_runoffcomponents = 0
option_plot_cmip5_map = 1

#%% ===== Input data =====
netcdf_fp_cmip5 = '/Users/davidrounce/Documents/Dave_Rounce/HiMAT/Output/simulations/spc/20181108_vars/'
netcdf_fp_era = '/Users/davidrounce/Documents/Dave_Rounce/HiMAT/Output/simulations/ERA-Interim_2000_2017wy_nobiasadj/'
figure_fp = '/Users/davidrounce/Documents/Dave_Rounce/HiMAT/Output/figures/cmip5/'

# Watersheds
watershed_shp_fn = '/Users/davidrounce/Documents/Dave_Rounce/HiMAT/qgis_himat/HMA_watersheds_merged_clipped.shp'
watershed_pts_fn = '/Users/davidrounce/Documents/Dave_Rounce/HiMAT/qgis_himat/HMA_watersheds_merged_clipped_pts.shp'
watershed_dict_fn = '/Users/davidrounce/Documents/Dave_Rounce/HiMAT/qgis_himat/rgi60_HMA_w_watersheds.csv'
watershed_csv = pd.read_csv(watershed_dict_fn)
watershed_dict = dict(zip(watershed_csv.RGIId, watershed_csv.watershed))


# Regions
rgi_regions = [13, 14, 15]
# GCMs and RCP scenarios
gcm_names = ['CanESM2', 'CCSM4', 'CNRM-CM5', 'GFDL-CM3', 'GFDL-ESM2M', 'GISS-E2-R', 'IPSL-CM5A-LR', 
             'IPSL-CM5A-MR', 'MIROC5', 'MRI-CGCM3', 'Nor-ESM1-M']
rcps = ['rcp26', 'rcp45', 'rcp85']

# Groups
#grouping = 'rgi_region'
grouping = 'watershed'

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
              13: 'Central Asia',
              14: 'South Asia West',
              15: 'South Asia East'
              }
vn_dict = {'volume_glac_annual': 'Normalized Volume [-]',
           'runoff_glac_annual': 'Normalized Runoff [-]',
           'temp_glac_annual': 'Temperature [degC]',
           'prec_glac_annual': 'Precipitation [m]'}
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

#%% REGIONAL BIAS ADJUSTED TEMPERATURE AND PRECIPITATION FOR GCMS
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
    dates_table_ref = modelsetup.datesmodelrun(startyear=input.startyear, endyear=input.endyear, spinupyears=0, 
                                               option_wateryear=1)
    dates_table = modelsetup.datesmodelrun(startyear=input.gcm_startyear, endyear=input.gcm_endyear, spinupyears=0,
                                           option_wateryear=1)
    # Load gcm lat/lons
    gcm = class_climate.GCM(name=gcm_name, rcp_scenario=rcp)
    # Select lat/lon from GCM
    ds_elev = xr.open_dataset(gcm.fx_fp + gcm.elev_fn)
    gcm_lat_values_all = ds_elev.lat.values
    gcm_lon_values_all = ds_elev.lon.values
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
    ref_gcm = class_climate.GCM(name=input.ref_gcm_name)
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
    if input.option_bias_adjustment == 2:        
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


#%% PLOT RESULTS
if option_plot_cmip5_normalizedchange == 1:
    vns = ['volume_glac_annual', 'runoff_glac_annual']
#    vns = ['volume_glac_annual']
#    vns = ['volume_glac_annual', 'temp_glac_annual', 'prec_glac_annual']
    # NOTE: Temperatures and precipitation will not line up exactly because each region is covered by a different 
    #       number of pixels, and hence the mean of those pixels is not going to be equal.
    
    option_plots_individual_gcms = 0
    multimodel_linewidth = 2
    
    # Load all glaciers
    for rgi_region in rgi_regions:
        # Data on all glaciers
        main_glac_rgi_region = modelsetup.selectglaciersrgitable(rgi_regionsO1=[rgi_region], rgi_regionsO2 = 'all', 
                                                                 rgi_glac_number='all')
        if rgi_region == rgi_regions[0]:
            main_glac_rgi_all = main_glac_rgi_region
        else:
            main_glac_rgi_all = pd.concat([main_glac_rgi_all, main_glac_rgi_region])
    
    # Load watershed dictionary
    watershed_csv = pd.read_csv(watershed_dict_fn)
    watershed_dict = dict(zip(watershed_csv.RGIId, watershed_csv.watershed))
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
    groups = sorted(groups)
    
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
                
                # Cycle through groups  
                row_idx = 0
                col_idx = 0
                for ngroup, group in enumerate(groups):
#                for ngroup, group in enumerate([groups[1]]):
#                    print(gcm_name, rcp, group)
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
                    if option_plots_individual_gcms == 1:
                        ax[row_idx, col_idx].plot(time_values, vn_reg_plot, color=rcp_colordict[rcp], linewidth=1, 
                                                  alpha=0, label=None)
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
                        ax[row_idx, col_idx].yaxis.set_major_locator(plt.MultipleLocator(0.2))
                        ax[row_idx, col_idx].yaxis.set_minor_locator(plt.MultipleLocator(0.1))
                    elif vn == 'runoff_glac_annual':
                        ax[row_idx, col_idx].set_ylim(0,2)
                        ax[row_idx, col_idx].yaxis.set_major_locator(plt.MultipleLocator(0.5))
                        ax[row_idx, col_idx].yaxis.set_minor_locator(plt.MultipleLocator(0.1))
                        ax[row_idx, col_idx].set_yticklabels(['','','0.5','1.0','1.5', ''])
                    
                    # Count column index to plot
                    col_idx += 1
                    
                    # Record data for multi-model stats
                    if ngcm == 0:
                        ds_multimodels[ngroup] = [group, vn_reg_plot]
                    else:
                        ds_multimodels[ngroup][1] = np.vstack((ds_multimodels[ngroup][1], vn_reg_plot))
                        
                # Only add group label once
                add_group_label = 0
            
            # Multi-model mean
            row_idx = 0
            col_idx = 0
            for ngroup, group in enumerate(groups):
                if (ngroup % num_cols == 0) and (ngroup != 0):
                    row_idx += 1
                    col_idx = 0
                # Multi-model statistics
                vn_multimodel_mean = ds_multimodels[ngroup][1].mean(axis=0)
                vn_multimodel_std = ds_multimodels[ngroup][1].std(axis=0)
                vn_multimodel_stdlow = vn_multimodel_mean - vn_multimodel_std
                vn_multimodel_stdhigh = vn_multimodel_mean + vn_multimodel_std
                ax[row_idx, col_idx].plot(time_values, vn_multimodel_mean, color=rcp_colordict[rcp], 
                                          linewidth=multimodel_linewidth, label=rcp)
                ax[row_idx, col_idx].fill_between(time_values, vn_multimodel_stdlow, vn_multimodel_stdhigh, 
                                                  facecolor=rcp_colordict[rcp], alpha=0.2, label=None)   
                # Adjust subplot column index
                col_idx += 1

        # RCP Legend
        rcp_lines = []
        for rcp in rcps:
            line = Line2D([0,1],[0,1], color=rcp_colordict[rcp], linewidth=multimodel_linewidth)
            rcp_lines.append(line)
        rcp_labels = [rcp_dict[rcp] for rcp in rcps]
        ax[0,0].legend(rcp_lines, rcp_labels, loc='lower left', fontsize=12, labelspacing=0, handlelength=1, 
                       handletextpad=0.5, borderpad=0, frameon=False)
        
        # GCM Legend
        gcm_lines = []
        for gcm_name in gcm_names:
            line = Line2D([0,1],[0,1], linestyle='-', color=gcm_colordict[gcm_name])
            gcm_lines.append(line)
        gcm_legend = gcm_names.copy()
        fig.legend(gcm_lines, gcm_legend, loc='center right', title='GCMs', bbox_to_anchor=(1.06,0.5), 
                   handlelength=0, handletextpad=0, borderpad=0, frameon=False)
        
        # Y-Label
        fig.text(0.03, 0.5, vn_dict[vn], va='center', rotation='vertical', size=16)
        
        # Save figure
        fig.set_size_inches(7, num_rows*2)
        figure_fn = grouping + '_' + vn + '_' + str(len(gcm_names)) + 'gcms_' + str(len(rcps)) +  'rcps.png'
        fig.savefig(figure_fp + figure_fn, bbox_inches='tight', dpi=300)

#%%
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
    groups = sorted(groups)
    
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


#%% PLOT MAP
if option_plot_cmip5_map == 1:
    vn = 'volume_glac_annual'
    
    # Load all glaciers
    for rgi_region in rgi_regions:
        # Data on all glaciers
        main_glac_rgi_region = modelsetup.selectglaciersrgitable(rgi_regionsO1=[rgi_region], rgi_regionsO2 = 'all', 
                                                                 rgi_glac_number='all')
        if rgi_region == rgi_regions[0]:
            main_glac_rgi_all = main_glac_rgi_region
        else:
            main_glac_rgi_all = pd.concat([main_glac_rgi_all, main_glac_rgi_region])
    
    # Load watershed dictionary
    watershed_csv = pd.read_csv(watershed_dict_fn)
    watershed_dict = dict(zip(watershed_csv.RGIId, watershed_csv.watershed))
    # Add watersheds to main_glac_rgi_all
    main_glac_rgi_all['watershed'] = main_glac_rgi_all.RGIId.map(watershed_dict)
    
    # Determine grouping
    if grouping == 'rgi_region':
        groups = rgi_regions
        group_cn = 'O1Region'
    elif grouping == 'watershed':
        groups = main_glac_rgi_all.watershed.unique().tolist()
        group_cn = 'watershed'
    groups = sorted(groups)
    
    #%% Load data to accompany watersheds  
    # Merge all data, then select group data
#    for rcp in rcps:
    for rcp in ['rcp85']:
        ds_vn = [[] for group in groups]
            
        for ngcm, gcm_name in enumerate(gcm_names):
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
                if ngcm == 0:
                    ds_vn[ngroup] = [group, vn_reg]
                else:
                    ds_vn[ngroup][1] = np.vstack((ds_vn[ngroup][1], vn_reg))

#%%
#    east = int(main_glac_rgi_all.CenLon.min())
#    west = int(np.ceil(main_glac_rgi_all.CenLon.max()))
#    south = int(main_glac_rgi_all.CenLat.min())
#    north = int(np.ceil(main_glac_rgi_all.CenLat.max()))
    east = 60
    west = 110
    south = 15
    north = 50
    xtick = 5
    ytick = 5
    xlabel = 'Longitude [deg]'
    ylabel = 'Latitude [deg]'
    
    # Create the projection
    fig, ax = plt.subplots(1, 1, figsize=(10,5), subplot_kw={'projection':cartopy.crs.PlateCarree()})
    # Add country borders for reference
#    ax.add_feature(cartopy.feature.BORDERS, alpha=0.15)
    ax.add_feature(cartopy.feature.COASTLINE)
    # Set the extent
    ax.set_extent([east, west, south, north], cartopy.crs.PlateCarree())    
    # Label title, x, and y axes
#    plt.title(title)
    ax.set_xticks(np.arange(east,west+1,xtick), cartopy.crs.PlateCarree())
    ax.set_yticks(np.arange(south,north+1,ytick), cartopy.crs.PlateCarree())
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    # Add watersheds
#    watershed_shp = cartopy.io.shapereader.Reader(watershed_shp_fn)
#    watershed_feature = cartopy.feature.ShapelyFeature(cartopy.io.shapereader.Reader(watershed_shp_fn).geometries(),
#                                   cartopy.crs.PlateCarree(), edgecolor='black', facecolor='none')
#    ax.add_feature(watershed_feature)
    
    watershed_centroid = {'Syr_Darya': [68, 46.1],
                      'Ili': [83.6, 45.5],
                      'Amu_Darya': [64.6, 38],
                      'Tarim': [83.0, 39.2],
                      'Inner_Tibetan_Plateau_extended': [103, 36],
                      'Indus': [70.7, 31.9],
                      'Inner_Tibetan_Plateau': [85, 32.4],
                      'Yangtze': [106.0, 30.4],
                      'Ganges': [81.3, 26.6],
                      'Brahmaputra': [92.0, 26],
                      'Irrawaddy': [96.2, 23.8],
                      'Salween': [98.5, 20.8],
                      'Mekong': [104, 17.5]
                      }
    
    if vn == 'volume_glac_annual':
        vn_thresholds = [0.25, 0.5, 0.75]
        vn_colors = ['red', 'pink', 'lightyellow', 'lightblue']
    # Add attribute of interest to shapefile
    watershed_shp_recs = []
    for rec in watershed_shp.records():
        if rec.attributes['name'] in groups:
            ds_idx = groups.index(rec.attributes['name'])
            vn_multimodel_mean = ds_vn[ds_idx][1].mean(axis=0)
            if vn == 'volume_glac_annual':
                rec.attributes['value'] = vn_multimodel_mean[-1] / vn_multimodel_mean[0]
                # simple scheme to assign color to each watershed
                print(rec.attributes['name'], rec.attributes['value'])
                if rec.attributes['value'] < vn_thresholds[0]:
                    facecolor = vn_colors[0]
                elif rec.attributes['value'] < vn_thresholds[1]:
                    facecolor = vn_colors[1]
                elif rec.attributes['value'] < vn_thresholds[2]:
                    facecolor = vn_colors[2]
                else:
                    facecolor = vn_colors[3]  
            # Add polygon to plot
#            ax.add_geometries(rec.geometry, cartopy.crs.PlateCarree(), facecolor=facecolor, edgecolor='grey', zorder=1)
            cmap = mpl.cm.RdYlBu
            ax.add_geometries(rec.geometry, cartopy.crs.PlateCarree(), facecolor=cmap(rec.attributes['value'], 1), 
                              edgecolor='grey', zorder=1)            
            ax.text(watershed_centroid[rec.attributes['name']][0], watershed_centroid[rec.attributes['name']][1], 
                    title_dict[rec.attributes['name']], horizontalalignment='center', size=12, zorder=3)
            
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0,1))
    sm._A = []
    plt.colorbar(sm, ax=ax, fraction=0.02, pad=0.02)
    fig.text(0.95, 0.5, vn_dict[vn], va='center', rotation='vertical', size=14)
            
#    # Add regional legend
#    legend_reg = []
#    legend_reg_labels = []
#    if vn == 'volume_glac_annual':
#        for i_threshold, vn_threshold in enumerate(vn_thresholds):
#            legend_reg.append(mpatches.Rectangle((0,0), 1, 1, facecolor=vn_colors[i_threshold], edgecolor='grey'))
#            legend_reg_labels.append('< ' + str(vn_threshold))
#        legend_reg.append(mpatches.Rectangle((0,0), 1, 1, facecolor=vn_colors[-1], edgecolor='grey'))
#        legend_reg_labels.append('> ' + str(vn_thresholds[-1]))
#    leg = plt.legend(legend_reg, legend_reg_labels, loc='lower left', bbox_to_anchor=(0,0))
#    ax.add_artist(leg)
    
    # Plot individual glaciers
    def size_thresholds(variable, cutoffs, sizes):
        """Loop through size thresholds for a given variable to plot"""
        output = np.zeros(variable.shape)
        for i, cutoff in enumerate(cutoffs):
            output[(variable>cutoff) & (output==0)] = sizes[i]
        output[output==0] = 2
        return output
    area_cutoffs = [100, 10, 1, 0.1]
    area_cutoffs_size = [1000, 100, 10, 1]
    area_sizes = size_thresholds(main_glac_rgi_all.Area.values, area_cutoffs, area_cutoffs_size)
    
    # Volume change
    if vn == 'volume_glac_annual':
        vn_glac_all_norm = np.zeros(vn_glac_all.shape[0])
        vn_glac_all_norm[vn_glac_all[:,0] > 0] = (
                vn_glac_all[vn_glac_all[:,0] > 0,-1] / vn_glac_all[vn_glac_all[:,0] > 0,0])
    
    glac_lons = main_glac_rgi_all.CenLon.values
    glac_lats = main_glac_rgi_all.CenLat.values
    sc = ax.scatter(glac_lons, glac_lats, c=vn_glac_all_norm, vmin=0, vmax=1, cmap='RdYlBu',
                    s=area_sizes,
                    edgecolor='grey', linewidth=0.25,
                    transform=cartopy.crs.PlateCarree(), zorder=2)
    
    # Add legend for glacier sizes
    legend_glac = []
    legend_glac_labels = []
    legend_glac_markersize = [20, 10, 5, 2]
    for i_area, area_cutoff_size in enumerate(area_cutoffs_size):
        legend_glac.append(Line2D([0], [0], linestyle='None', marker='o', color='grey', 
                                  label=(str(area_cutoffs[i_area]) + 'km$^2$'), 
                                  markerfacecolor='grey', markersize=legend_glac_markersize[i_area]))
    plt.legend(handles=legend_glac, loc='lower left')

    # Save figure
#    fig.set_size_inches(10,6)
#    figure_fn = grouping + '_wglac_' + vn + '_' + str(len(gcm_names)) + 'gcms_' + str(len(rcps)) +  'rcps.png'
#    fig.savefig(figure_fp + figure_fn, bbox_inches='tight', dpi=300)
    plt.show()