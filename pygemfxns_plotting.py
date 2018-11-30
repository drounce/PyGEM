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
import netCDF4 as nc
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import scipy
import cartopy
import xarray as xr
# Local Libraries
import pygem_input as input
import pygemfxns_modelsetup as modelsetup
import pygemfxns_massbalance as massbalance
import class_mbdata
import class_climate
import run_simulation


# Script options
option_plot_cmip5_volchange = 1

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
if option_plot_cmip5_volchange == 1:
    netcdf_fp_cmip5 = '/Users/davidrounce/Documents/Dave_Rounce/HiMAT/Output/simulations/spc/20181108_vars/'
    netcdf_fp_era = '/Users/davidrounce/Documents/Dave_Rounce/HiMAT/Output/simulations/ERA-Interim_2000_2017wy_nobiasadj/'
    figure_fp = '/Users/davidrounce/Documents/Dave_Rounce/HiMAT/Output/figures/cmip5/'
    watershed_dict_fn = '/Users/davidrounce/Documents/Dave_Rounce/HiMAT/qgis_himat/rgi60_HMA_w_watersheds.csv'
    
#    grouping = 'rgi_region'
    grouping = 'watershed'
    
    # Plot label dictionaries
    title_dict = {'Amu_Darya': 'Amu Darya',
                  'Brahmaputra': 'Brahmaputra',
                  'Ganges': 'Ganges',
                  'Ili': 'Ili',
                  'Indus': 'Indus',
                  'Inner_Tibetan_Plateau': 'Inner_TP',
                  'Inner_Tibetan_Plateau_extended': 'Inner_TP_ext',
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
    
    
    rgi_regions = [13, 14, 15]
    #gcm_names = ['ERA-Interim']
#    gcm_names = ['CanESM2', 'CCSM4', 'CNRM-CM5', 'CSIRO-Mk3-6-0', 'GFDL-CM3', 'GISS-E2-R', 'IPSL-CM5A-LR', 
#                 'IPSL-CM5A-MR', 'MIROC5', 'MRI-CGCM3']
#    gcm_names = ['CanESM2', 'CCSM4', 'CNRM-CM5', 'GFDL-CM3', 'GISS-E2-R', 'IPSL-CM5A-LR', 
#                 'IPSL-CM5A-MR', 'MIROC5', 'MRI-CGCM3']
    gcm_names = ['CSIRO-Mk3-6-0']
#    rcps = ['rcp26', 'rcp45', 'rcp60', 'rcp85']
#    rcps = ['rcp26', 'rcp45', 'rcp85']
    rcps = ['rcp26']
    vns = ['runoff_glac_annual']
#    vns = ['volume_glac_annual']
#    vns = ['volume_glac_annual', 'temp_glac_annual', 'prec_glac_annual']
    # NOTE: Temperatures and precipitation will not line up exactly because each region is covered by a different 
    #       number of pixels, and hence the mean of those pixels is not going to be equal.
    
    # Colors list
    colors_rgb = [(0.00, 0.57, 0.57), (0.71, 0.43, 1.00), (0.86, 0.82, 0.00), (0.00, 0.29, 0.29), (0.00, 0.43, 0.86), 
                  (0.57, 0.29, 0.00), (1.00, 0.43, 0.71), (0.43, 0.71, 1.00), (0.14, 1.00, 0.14), (1.00, 0.71, 0.47), 
                  (0.29, 0.00, 0.57), (0.57, 0.00, 0.00), (0.71, 0.47, 1.00), (1.00, 1.00, 0.47)]
    gcm_colordict = dict(zip(gcm_names, colors_rgb[0:len(gcm_names)]))
    rcp_colordict = {'rcp26':'b', 'rcp45':'k', 'rcp60':'m', 'rcp85':'r'}
    rcp_styledict = {'rcp26':':', 'rcp45':'--', 'rcp85':'-.'}
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
    
    #%%
    reg_legend = []
    num_cols_max = 4
    if len(groups) < num_cols_max:
        num_cols = len(groups)
    else:
        num_cols = num_cols_max
    num_rows = int(np.ceil(len(groups)/num_cols))
        
    for vn in vns:
        fig, ax = plt.subplots(num_rows, num_cols, squeeze=False, sharex=True, sharey=True,
                               figsize=(int(5*len(groups)),int(4*len(vns))), 
                               gridspec_kw = {'wspace':0.05, 'hspace':0.25})
        
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
#                for ngroup, group in enumerate(groups):
                for ngroup, group in enumerate([groups[1]]):
                    print(gcm_name, rcp, group)
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
                        vn_reg_2000_2017_mean = vn_reg[t1_idx:t2_idx+1].mean()
                        # Regional normalized volume        
                        vn_reg_norm = vn_reg / vn_reg_2000_2017_mean
                        vn_reg_norm_stdhigh = vn_reg_stdhigh / vn_reg_2000_2017_mean
                        vn_reg_norm_stdlow = vn_reg_stdlow / vn_reg_2000_2017_mean
                        vn_reg_plot = vn_reg_norm.copy()
                        vn_reg_plot_stdlow = vn_reg_norm_stdlow.copy()
                        vn_reg_plot_stdhigh = vn_reg_norm_stdhigh.copy()
                
#                    # ===== Plot =====
#                    ax[row_idx, col_idx].plot(time_values, vn_reg_plot, color=rcp_colordict[rcp], linewidth=1, 
#                                              alpha=0, label=None)
##                    # Volume change uncertainty
##                    if vn == 'volume_glac_annual':
##                        ax[row_idx, col_idx].fill_between(
##                                time_values, vn_reg_plot_stdlow, vn_reg_plot_stdhigh, 
##                                facecolor=gcm_colordict[gcm_name], alpha=0.15, label=None)
#                    
#                    # Group labels
#                    ax[row_idx, col_idx].set_title(title_dict[group], size=14)
#                    ax[row_idx, col_idx].tick_params(axis='both', which='major', labelsize=20)
#                    # X-label
#                    ax[row_idx, col_idx].set_xlim(time_values.min(), time_values.max())
#                    ax[row_idx, col_idx].xaxis.set_tick_params(labelsize=14)
#                    ax[row_idx, col_idx].xaxis.set_major_locator(plt.MultipleLocator(50))
#                    ax[row_idx, col_idx].xaxis.set_minor_locator(plt.MultipleLocator(10))
#                    xlabels = ax[row_idx, col_idx].xaxis.get_major_ticks()
#                    ax[row_idx, col_idx].set_xticklabels(['','2000','2050',''])
#                    # Y-label
#                    ax[row_idx, col_idx].yaxis.set_tick_params(labelsize=14)
#                    if vn == 'volume_glac_annual':
#                        ax[row_idx, col_idx].yaxis.set_major_locator(plt.MultipleLocator(0.2))
#                        ax[row_idx, col_idx].yaxis.set_minor_locator(plt.MultipleLocator(0.1))
#                    
#                    # Count column index to plot
#                    col_idx += 1
#                    
#                    # Record data for multi-model stats
#                    if ngcm == 0:
#                        ds_multimodels[ngroup] = [group, vn_reg_plot]
#                    else:
#                        ds_multimodels[ngroup][1] = np.vstack((ds_multimodels[ngroup][1], vn_reg_plot))
#            
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
#                ax[row_idx, col_idx].fill_between(time_values, vn_multimodel_stdlow, vn_multimodel_stdhigh, 
#                                                  facecolor=rcp_colordict[rcp], alpha=0.2, label=None)
#                col_idx += 1
#
#        # RCP Legend
#        rcp_lines = []
#        for rcp in rcps:
#            line = Line2D([0,1],[0,1], color=rcp_colordict[rcp], linewidth=multimodel_linewidth)
#            rcp_lines.append(line)
#        rcp_labels = [rcp_dict[rcp] for rcp in rcps]
#        ax[0,0].legend(rcp_lines, rcp_labels, loc='lower left', fontsize=12, labelspacing=0, handlelength=1, 
#                       handletextpad=0.5, borderpad=0, frameon=False)
#    #    # GCM Legend
#    #    gcm_lines = []
#    #    for gcm_name in gcm_names:
#    #        line = Line2D([0,1],[0,1], linestyle='-', color=gcm_colordict[gcm_name])
#    #        gcm_lines.append(line)
#    #    gcm_legend = gcm_names.copy()
#    #    gcm_legend.append('Mean')
#    #    line = Line2D([0,1],[0,1], linestyle='-', color='k', linewidth=5)
#    #    gcm_lines.append(line)
#    #    ax[0,2].legend(gcm_lines, gcm_legend, loc='center left', title='GCM', bbox_to_anchor=(1,0,0.5,1))
#        fig.text(0.03, 0.5, vn_dict[vn], va='center', rotation='vertical', size=16)
#        # Save figure
#        fig.set_size_inches(7, num_rows*2)
#        figure_fn = grouping + '_' + vn + '_' + str(len(gcm_names)) + 'gcms_' + str(len(rcps)) +  'rcps.png'
#        fig.savefig(figure_fp + figure_fn, bbox_inches='tight', dpi=300)
