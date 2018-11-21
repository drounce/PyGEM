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
region=13
gcm_name = 'CSIRO-Mk3-6-0'
rcp = 'rcp85'


def select_region_climatedata(gcm_name, rcp, main_glac_rgi):
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
    figure_fp = '/Users/davidrounce/Documents/Dave_Rounce/HiMAT/Output/figures/'
    
    regions = [13, 14, 15]
    #regions = [15]
    #gcm_names = ['ERA-Interim']
    gcm_names = ['CanESM2', 'CCSM4', 'CNRM-CM5', 'CSIRO-Mk3-6-0']
    #gcm_names = ['CanESM2']
#    rcps = ['rcp26', 'rcp45', 'rcp60', 'rcp85']
    rcps = ['rcp26', 'rcp45', 'rcp85']
#    vns = ['volume_glac_annual']
    vns = ['volume_glac_annual', 'temp_glac_annual', 'prec_glac_annual']
    
    # Colors list
    colors_rgb = [(0.00, 0.57, 0.57), (0.71, 0.43, 1.00), (0.86, 0.82, 0.00), (0.00, 0.29, 0.29), (0.00, 0.43, 0.86), 
                  (0.57, 0.29, 0.00), (1.00, 0.43, 0.71), (0.43, 0.71, 1.00), (0.14, 1.00, 0.14), (1.00, 0.71, 0.47), 
                  (0.29, 0.00, 0.57), (0.57, 0.00, 0.00), (0.71, 0.47, 1.00), (1.00, 1.00, 0.47)]
    gcm_colordict = dict(zip(gcm_names, colors_rgb[0:len(gcm_names)]))
    

    rcp_colordict = {'rcp26':'b', 'rcp45':'k', 'rcp60':'m', 'rcp85':'r'}
    rcp_styledict = {'rcp26':':', 'rcp45':'--', 'rcp85':'-.'}
    reg_legend = []
    fig, ax = plt.subplots(len(vns), len(regions), squeeze=False, 
                           figsize=(int(2*len(regions)),int(4*len(vns))), 
                           gridspec_kw = {'wspace':0.05, 'hspace':0.05})
    
    count_vn = 0
    for vn in vns:
#    count_vn=2
#    for vn in [vns[2]]:
        print(vn)
        count_reg = 0
        for region in regions:
#        for region in [regions[0]]:
            vn_reg_plot_all_mean_dict = {}
            
            # Load climate data so not influenced by retreating glaciers
            main_glac_rgi = modelsetup.selectglaciersrgitable(rgi_regionsO1=[region], rgi_regionsO2 = 'all', 
                                                              rgi_glac_number='all')
            
            for rcp in rcps:
#            for rcp in ['rcp85']:
                vn_reg_plot_all = None
                for gcm_name in gcm_names:
#                for gcm_name in ['CSIRO-Mk3-6-0']:
                    # NetCDF filename
                    if gcm_name == 'ERA-Interim':
                        netcdf_fp = netcdf_fp_era
                        ds_fn = 'R' + str(region) + '--ERA-Interim_c2_ba0_200sets_2000_2017_stats.nc'
                    else:
                        netcdf_fp = netcdf_fp_cmip5 + vn + '/'
                        ds_fn = ('R' + str(region) + '_' + gcm_name + '_' + rcp + '_c2_ba2_100sets_2000_2100--' + vn + 
                                 '.nc')    
                    # Open dataset
                    # try to open it, continue to bypass GCMs that don't have all rcps
                    try:
                        ds = xr.open_dataset(netcdf_fp + ds_fn)
                    except:
                        continue
                    
                    # Time
                    if 'annual' in vn:
                        try:
                            time_values = ds[vn].coords['year_plus1'].values
                        except:
                            time_values = ds[vn].coords['year'].values
                    # Glacier volume mean, standard deviation, and variance
                    vn_glac = ds[vn].values[:,:,0]
                    vn_glac_std = ds[vn].values[:,:,1]
                    vn_glac_var = vn_glac_std **2
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
                        # Multi-model mean of regional GCMs
                        if vn_reg_plot_all is None:
                            vn_reg_plot_all = vn_reg_norm
                        else:
                            vn_reg_plot_all = np.vstack((vn_reg_plot_all, vn_reg_norm))
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
                        # Multi-model mean of regional GCMs
                        if vn_reg_plot_all is None:
                            vn_reg_plot_all = vn_reg_plot
                        else:
                            vn_reg_plot_all = np.vstack((vn_reg_plot_all, vn_reg_plot))
                        
                    # ===== Plot =====
                    ax[count_vn, count_reg].plot(time_values, vn_reg_plot, color=gcm_colordict[gcm_name], 
                                                 linestyle=rcp_styledict[rcp], linewidth=1, label=None)
                    # Volume change uncertainty
                    if vn == 'volume_glac_annual':
                        ax[count_vn, count_reg].fill_between(time_values, vn_reg_plot_stdlow, vn_reg_plot_stdhigh, 
                                                             facecolor=gcm_colordict[gcm_name], alpha=0.15, label=None)
                    
                    # Regional labels
                    if count_vn == 0:
                        ax[count_vn, count_reg].set_title(('Region' + str(region)), size=14)
                    # Y-label
                    if vn == 'volume_glac_annual':
                        ax[count_vn, count_reg].set_ylim(0,1)
                        ax[count_vn, count_reg].yaxis.set_major_locator(plt.MultipleLocator(0.2))
                        ax[count_vn, count_reg].yaxis.set_minor_locator(plt.MultipleLocator(0.1))
                        if count_reg == 0:
                            ax[count_vn, count_reg].set_ylabel('Normalized volume [-]', size=12)
                        # Turn off tick labels
                        else:
                            ax[count_vn, count_reg].set_yticklabels([])
                        ax[count_vn, count_reg].xaxis.set_tick_params(labelsize=10)
                        ax[count_vn, count_reg].xaxis.set_major_locator(plt.MultipleLocator(40))
                        ax[count_vn, count_reg].xaxis.set_minor_locator(plt.MultipleLocator(10))
                        ax[count_vn, count_reg].set_xticklabels([])
                    elif vn == 'temp_glac_annual':
                        if count_reg == 0:
                            ax[count_vn, count_reg].set_ylabel('Temperature [degC]', size=12)
                        # Turn off tick labels
                        else:
                            ax[count_vn, count_reg].set_yticklabels([])
                        ax[count_vn, count_reg].xaxis.set_tick_params(labelsize=10)
                        ax[count_vn, count_reg].xaxis.set_major_locator(plt.MultipleLocator(40))
                        ax[count_vn, count_reg].xaxis.set_minor_locator(plt.MultipleLocator(10))
                        ax[count_vn, count_reg].set_xticklabels([])
                    elif vn == 'prec_glac_annual':
                        if count_reg == 0:
                            ax[count_vn, count_reg].set_ylabel('Precipitation [m]', size=12)   
                        ax[count_vn, count_reg].xaxis.set_tick_params(labelsize=10)
                        ax[count_vn, count_reg].xaxis.set_major_locator(plt.MultipleLocator(40))
                        ax[count_vn, count_reg].xaxis.set_minor_locator(plt.MultipleLocator(10))
                        # Turn off tick labels
#                        else:
#                            ax[count_vn, count_reg].set_yticklabels([])
#        axi.xaxis.set_minor_locator(plt.MultipleLocator(10))
                    
                # Multi-model mean
                vn_reg_plot_all_mean_dict.update({rcp:vn_reg_plot_all.mean(axis=0)})
            # Plot multi-model means last, so they are on top
            for rcp in rcps:
                ax[count_vn, count_reg].plot(time_values, vn_reg_plot_all_mean_dict[rcp], color='k', 
                                             linestyle=rcp_styledict[rcp], linewidth=3, label=rcp)
                
#            if vn == 'volume_glac_annual': 
#                ax[count_vn, count_reg].set_ylim(0,1)
#                ax[count_vn, count_reg].yaxis.set_major_locator(plt.MultipleLocator(0.2))
#                ax[count_vn, count_reg].yaxis.set_minor_locator(plt.MultipleLocator(0.1))
#                if count_reg > 0:
#                    # Turn off tick labels
#                    ax[count_vn, count_reg].set_yticklabels([])
#                    ax[count_vn, count_reg].set_xticklabels([])
#            elif vn == 'temp_glac_annual':
#                ax[count_vn, count_reg].yaxis.set_major_locator(plt.MultipleLocator(3))
#                ax[count_vn, count_reg].yaxis.set_minor_locator(plt.MultipleLocator(1))
#            elif vn == 'prec_glac_annual':
#                ax[count_vn, count_reg].yaxis.set_major_locator(plt.MultipleLocator(1))
#                ax[count_vn, count_reg].yaxis.set_minor_locator(plt.MultipleLocator(1))
            # Regional count for subplots
            count_reg += 1
        # Variable count for subplots
        count_vn += 1
    # RCP Legend
    rcp_lines = []
    for rcp in rcps:
        line = Line2D([0,1],[0,1], linestyle=rcp_styledict[rcp], color='k', linewidth=2)
        rcp_lines.append(line)
    if len(vns) == 1:
        ax[0,0].legend(rcp_lines, rcps, loc='lower left')
    else:
        ax[1,2].legend(rcp_lines, rcps, loc='center left', title='RCP scenario', bbox_to_anchor=(1,0,0.5,1))
    # GCM Legend
    gcm_lines = []
    for gcm_name in gcm_names:
        line = Line2D([0,1],[0,1], linestyle='-', color=gcm_colordict[gcm_name])
        gcm_lines.append(line)
    gcm_legend = gcm_names.copy()
    gcm_legend.append('Mean')
    line = Line2D([0,1],[0,1], linestyle='-', color='k', linewidth=5)
    gcm_lines.append(line)
    ax[0,2].legend(gcm_lines, gcm_legend, loc='center left', title='GCM', bbox_to_anchor=(1,0,0.5,1))
#    # Set limits and tick labels
#    for axi in ax.flat:
#        axi.xaxis.set_major_locator(plt.MultipleLocator(40))
#        axi.xaxis.set_minor_locator(plt.MultipleLocator(10))
    # Save figure
    fig.set_size_inches(7, int(len(vns)*3))
    fig.savefig(figure_fp + 'Regional_VolumeChange_allgcmsrcps.png', bbox_inches='tight', dpi=300)
    #plt.show()
