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
import run_simulation



# Script options
option_plot_cmip5_volchange = 1

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
    vn = 'volume_glac_annual'
    vns = ['volume_glac_annual', 'temp_glac_annual', 'prec_glac_annual']
    
    # Colors list
    colors_rgb = [(0.00, 0.57, 0.57), (0.71, 0.43, 1.00), (0.86, 0.82, 0.00), (0.00, 0.29, 0.29), (0.00, 0.43, 0.86), 
                  (0.57, 0.29, 0.00), (1.00, 0.43, 0.71), (0.43, 0.71, 1.00), (0.14, 1.00, 0.14), (1.00, 0.71, 0.47), 
                  (0.29, 0.00, 0.57), (0.57, 0.00, 0.00), (0.71, 0.47, 1.00), (1.00, 1.00, 0.47)]
    gcm_colordict = dict(zip(gcm_names, colors_rgb[0:len(gcm_names)]))
    

    rcp_colordict = {'rcp26':'b', 'rcp45':'k', 'rcp60':'m', 'rcp85':'r'}
    rcp_styledict = {'rcp26':':', 'rcp45':'--', 'rcp85':'-.'}
    reg_legend = []
    fig, ax = plt.subplots(len(vns), len(regions), sharex=True, squeeze=False, 
                           figsize=(int(2*len(regions)),int(4*len(vns))), 
                           gridspec_kw = {'wspace':0, 'hspace':0})
    
#    count_vn = 0
#    for vn in vns:
    count_vn=2
    for vn in [vns[2]]:
        print(vn)
        count_reg = 0
#        for region in regions:
        for region in [regions[0]]:
            vn_reg_plot_all_mean_dict = {}
            for rcp in rcps:
                vn_reg_plot_all = None
                for gcm_name in gcm_names:
                    # NetCDF filename
                    if gcm_name == 'ERA-Interim':
                        netcdf_fp = netcdf_fp_era
                        ds_fn = 'R' + str(region) + '--ERA-Interim_c2_ba0_200sets_2000_2017_stats.nc'
                    else:
                        netcdf_fp = netcdf_fp_cmip5 + vn + '/'
                        ds_fn = ('R' + str(region) + '_' + gcm_name + '_' + rcp + '_c2_ba2_100sets_2000_2100--' + vn + 
                                 '.nc')    
                    # Open dataset
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
                    else:
                        # Remove values that don't last the entire time series to avoid biasing the result with zeros
                        # when the glacier is gone.
                        mask_var = vn_glac[:,-1] == 0
                        vn_glac[mask_var,:] = np.nan
                        vn_glac_std[mask_var,:] = np.nan
                        vn_glac_var[mask_var,:] = np.nan
                        # Regional mean, standard deviation, and variance
                        count_glac = (~np.isnan(vn_glac[:,-1])).sum()
                        vn_reg = np.nanmean(vn_glac, axis=0)
                        vn_reg_var = np.nansum(vn_glac_var, axis=0) / count_glac
                        vn_reg_std = vn_reg_var**0.5
                        vn_reg_stdhigh = vn_reg + vn_reg_std
                        vn_reg_stdlow = vn_reg - vn_reg_std
                        # Multi-model mean of regional GCMs
                        if vn_reg_plot_all is None:
                            vn_reg_plot_all = vn_reg
                        else:
                            vn_reg_plot_all = np.vstack((vn_reg_plot_all, vn_reg))
                        vn_reg_plot = vn_reg.copy()
                        vn_reg_plot_stdlow = vn_reg_stdlow.copy()
                        vn_reg_plot_stdhigh = vn_reg_stdhigh.copy()
                        
                    # ===== Plot =====
                    ax[count_vn, count_reg].plot(time_values, vn_reg_plot, color=gcm_colordict[gcm_name], 
                                                 linestyle=rcp_styledict[rcp], linewidth=1, label=None)
                    if vn == 'volume_glac_annual':
                        ax[count_vn, count_reg].fill_between(time_values, vn_reg_plot_stdlow, vn_reg_plot_stdhigh, 
                                                             facecolor=gcm_colordict[gcm_name], alpha=0.15, label=None)
                    if count_vn == 0:
                        ax[count_vn, count_reg].set_title(('Region' + str(region)), size=14)
                    # Y-label
                    if vn == 'volume_glac_annual':
                        if count_reg == 0:
                            ax[count_vn, count_reg].set_ylabel('Normalized volume [-]', size=14)
                    elif vn == 'temp_glac_annual':
                        if count_reg == 0:
                            ax[count_vn, count_reg].set_ylabel('Temperature[degC]', size=14)
                    ax[count_vn, count_reg].xaxis.set_tick_params(labelsize=10)
                # Multi-model mean
                vn_reg_plot_all_mean_dict.update({rcp:vn_reg_plot_all.mean(axis=0)})
            # Plot multi-model means last, so they are on top
            for rcp in rcps:
                ax[count_vn, count_reg].plot(time_values, vn_reg_plot_all_mean_dict[rcp], color='k', 
                                             linestyle=rcp_styledict[rcp], linewidth=3, label=rcp)
            if vn == 'volume_glac_annual': 
                print('count_vn:', count_vn, 'count_reg:', count_reg, 'set 0,1')
                ax[count_vn, count_reg].set_ylim(0,1)
                ax[count_vn, count_reg].yaxis.set_major_locator(plt.MultipleLocator(0.2))
                ax[count_vn, count_reg].yaxis.set_minor_locator(plt.MultipleLocator(0.1))
                if count_reg > 0:
                    # Turn off tick labels
                    ax[count_vn, count_reg].set_yticklabels([])
                    ax[count_vn, count_reg].set_xticklabels([])
            elif vn == 'temp_glac_annual':
                print('count_vn:', count_vn, 'count_reg:', count_reg, 'set 0,1')
                ax[count_vn, count_reg].set_ylim(-12,-1)
                ax[count_vn, count_reg].yaxis.set_major_locator(plt.MultipleLocator(3))
                ax[count_vn, count_reg].yaxis.set_minor_locator(plt.MultipleLocator(1))
                if count_reg > 0:
                    # Turn off tick labels
                    ax[count_vn, count_reg].set_yticklabels([])
                    ax[count_vn, count_reg].set_xticklabels([])
            # Regional count for subplots
            count_reg += 1
        # Variable count for subplots
        count_vn += 1
    # RCP Legend
    rcp_lines = []
    for rcp in rcps:
        line = Line2D([0,1],[0,1], linestyle=rcp_styledict[rcp], color='k', linewidth=2)
        rcp_lines.append(line)
    ax[0,0].legend(rcp_lines, rcps, loc='lower left')
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
    # Set limits and tick labels
    for axi in ax.flat:
        axi.xaxis.set_major_locator(plt.MultipleLocator(40))
        axi.xaxis.set_minor_locator(plt.MultipleLocator(10))
#        axi[0].yaxis.set_ylim(0,1)
#        axi[0:2].yaxis.set_ylim(0,1)

    #ax.xticks(label)
    # Save figure
    fig.set_size_inches(7, int(len(vns)*3))
    fig.savefig(figure_fp + 'Regional_VolumeChange_allgcmsrcps.png', bbox_inches='tight', dpi=300)
    #plt.show()
