""" Analyze simulation output - mass change, runoff, etc. """

# Built-in libraries
from collections import OrderedDict
import datetime
import glob
import os
import pickle
# External libraries
import cartopy
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.pyplot import MaxNLocator
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import EngFormatter
from matplotlib.ticker import StrMethodFormatter
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import numpy as np
import pandas as pd
from scipy.stats import median_abs_deviation
from scipy.stats import linregress
from scipy.ndimage import uniform_filter
import scipy
import xarray as xr

# Local libraries
import pygem.pygem_input as pygem_prms
import pygem.pygem_modelsetup as modelsetup
from pygem.oggm_compat import single_flowline_glacier_directory
from pygem.shop import debris 
from oggm import tasks

# Script options
option_plot_era5_volchange = False
option_get_missing_glacno = True

option_plot_cmip5_volchange = False
option_plot_era5_AAD = False
option_process_data = False


#%% ===== Input data =====
netcdf_fp_sims = pygem_prms.main_directory + '/../Output/simulations/'

option_plot_cross_section = False            # Plot cross section of model output
option_plot_diag = True                     # Plot area, volume, and runoff output

#%% ===== PLOT CROSS SECTION =====
if option_plot_cross_section:
    glac_nos = ['15.03733']
    gcms = ['CESM2']
    rcps = ['ssp245']
    
    fig_fp = netcdf_fp_sims + 'figures/'
    fig_fp_ind = fig_fp + 'ind_glaciers/'
    if not os.path.exists(fig_fp_ind):
        os.makedirs(fig_fp_ind)

    startyear = 2000
    endyear = 2100
    normyear = 2000
    cs_year = 2100
    
    for glac_no in glac_nos:
        
        print('\n\n', glac_no)

        gdir = single_flowline_glacier_directory(glac_no, logging_level='CRITICAL')
        
        tasks.init_present_time_glacier(gdir) # adds bins below
        debris.debris_binned(gdir, fl_str='model_flowlines')  # add debris enhancement factors to flowlines
        nfls = gdir.read_pickle('model_flowlines')
        
        x = np.arange(nfls[0].nx) * nfls[0].dx * nfls[0].map_dx
        
        glac_idx = np.nonzero(nfls[0].thick)[0]
        xmax = np.ceil(x[glac_idx].max()/1000+0.5)*1000
        
        for gcm_name in gcms:
            
            if gcm_name in ['ERA5']:
                rcps = ['']
            else:
                rcps = rcps
                                
            for rcp in rcps:
                
                ds_binned_fp = netcdf_fp_sims + glac_no.split('.')[0].zfill(2) + '/' + gcm_name + '/' + rcp + '/binned/'
                for i in os.listdir(ds_binned_fp):
                    if i.startswith(glac_no):
                        ds_binned_fn = i
                ds_stats_fp = netcdf_fp_sims + glac_no.split('.')[0].zfill(2) + '/' + gcm_name + '/' + rcp + '/stats/'
                for i in os.listdir(ds_stats_fp):
                    if i.startswith(glac_no):
                        ds_stats_fn = i
                
                if glac_no in ds_stats_fn and rcp in ds_stats_fn and gcm_name in ds_stats_fn:
                    ds_binned = xr.open_dataset(ds_binned_fp + ds_binned_fn)
                    ds_stats = xr.open_dataset(ds_stats_fp + ds_stats_fn)
                    
                    years = ds_stats.year.values
                    startyear_idx = np.where(years == startyear)[0][0]
                    normyear_idx = np.where(years == normyear)[0][0]
                    endyear_idx = np.where(years == endyear)[0][0]
                    
                    thick = ds_binned.bin_thick_annual[0,:,:].values
                    zsurf_init = ds_binned.bin_surface_h_initial[normyear_idx].values
                    cs_idx = np.where(years == cs_year)[0][0] 
                    zbed = zsurf_init - thick[:,cs_idx]
                    zsurf_all = zbed[:,np.newaxis] + thick
                    vol = ds_stats.glac_volume_annual[0,:].values
                    
                    #%% ----- FIGURE: VOLUME CHANGE MULTI-GCM -----
                    fig = plt.figure()
                    ax = fig.add_axes([0,0,1,0.65])
                    ax.patch.set_facecolor('none')
                    ax2 = fig.add_axes([0,0.67,1,0.35])
                    ax2.patch.set_facecolor('none')
                    ax3 = fig.add_axes([0.67,0.32,0.3,0.3])
                    ax3.patch.set_facecolor('none')
                    
                    
                    ymin, ymax, thick_max = None, None, None
                    vol_med_all = []
                    
                    ax.fill_between(x[1:]/1000, zbed[1:]-20, zbed[1:], color='white', zorder=5)
                    ax.plot(x/1000, zbed[np.arange(len(x))],
                            color='k', linestyle='-', linewidth=1, zorder=5, label='zbed')
                    ax.plot(x/1000, zsurf_all[np.arange(len(x)), normyear_idx], 
                            color='grey', linestyle=':', linewidth=0.5, zorder=3, label=str(years[normyear_idx]))
                    ax.plot(x/1000, zsurf_all[np.arange(len(x)),endyear_idx], 
                            color='k', linestyle='-', linewidth=0.5, zorder=4, label=str(endyear))
                        
                    ax2.plot(x/1000, thick[:,normyear_idx], 
                             color='grey', linestyle=':', linewidth=0.5, zorder=4, label=str(endyear))
                    ax2.plot(x/1000, thick[np.arange(len(x)),endyear_idx],
                             color='k', linestyle='-', linewidth=0.5, zorder=4, label=str(endyear))
            
                    
                    ax3.plot(years, vol / vol[normyear_idx], color='k', 
                             linewidth=0.5, zorder=4, label=None)
            
                    # ymin and ymax for bounds
                    if ymin is None:
                        ymin = np.floor(zbed[glac_idx].min()/100)*100
                        ymax = np.ceil(zsurf_all[:,endyear_idx].max()/100)*100
                    if np.floor(zbed.min()/100)*100 < ymin:
                        ymin = np.floor(zbed[glac_idx].min()/100)*100
                    if np.ceil(zsurf_all[glac_idx,endyear_idx].max()/100)*100 > ymax:
                        ymax = np.ceil(zsurf_all[glac_idx,endyear_idx].max()/100)*100
                    
                    if ymin < 0:
                        water_idx = np.where(zbed < 0)[0]
                        # Add water level
                        ax.plot(x[water_idx]/1000, np.zeros(x[water_idx].shape), color='aquamarine', linewidth=1)
                    
                    if xmax/1000 > 25:
                        x_major, x_minor = 10, 2
                    elif xmax/1000 > 15:
                        x_major, x_minor = 5, 1
                    else:
                        x_major, x_minor = 2, 0.5
                    
                    y_major, y_minor = 500,100
                        
                    # ----- GLACIER SPECIFIC PLOTS -----
                    plot_legend = True
                    add_glac_name = True
                    plot_sealevel = True
                    leg_label = None
                    
                    if thick.max() > 200:
                        thick_major, thick_minor = 100, 20
                    else:
                        thick_major, thick_minor = 50, 10
                    
                    ax.set_ylim(ymin, ymax)
                    ax.set_xlim(0,xmax/1000)
                    ax.xaxis.set_major_locator(MultipleLocator(x_major))
                    ax.xaxis.set_minor_locator(MultipleLocator(x_minor))
                    ax.yaxis.set_major_locator(MultipleLocator(y_major))
                    ax.yaxis.set_minor_locator(MultipleLocator(y_minor)) 
                    ax.set_ylabel('Elevation (m a.s.l.)')
                    ax.set_xlabel('Distance along flowline (km)')
                    
                    ax2.set_xlim(0,xmax/1000)
                    ax2.xaxis.set_major_locator(MultipleLocator(x_major))
                    ax2.xaxis.set_minor_locator(MultipleLocator(x_minor))
                    ax2.yaxis.set_major_locator(MultipleLocator(thick_major))
                    ax2.yaxis.set_minor_locator(MultipleLocator(thick_minor))
                    ax2.set_ylabel('Thickness (m)')
                    ax2.get_xaxis().set_visible(False)

                    ax.tick_params(axis='both', which='major', direction='inout', right=True)
                    ax.tick_params(axis='both', which='minor', direction='in', right=True)

                    if add_glac_name:
                        ax2.text(0.98, 1.02, glac_no, size=10, horizontalalignment='right', 
                                verticalalignment='bottom', transform=ax2.transAxes)

                    ax3.set_ylabel('Mass (-)')
                    ax3.set_xlim(normyear, endyear)
                    ax3.xaxis.set_major_locator(MultipleLocator(40))
                    ax3.xaxis.set_minor_locator(MultipleLocator(10))
                    ax3.set_ylim(0,1.1)
                    ax3.yaxis.set_major_locator(MultipleLocator(0.5))
                    ax3.yaxis.set_minor_locator(MultipleLocator(0.1))
                    ax3.tick_params(axis='both', which='major', direction='inout', right=True)
                    ax3.tick_params(axis='both', which='minor', direction='in', right=True)
                    vol_norm_gt = np.median(vol[0]) * pygem_prms.density_ice / 1e12
                    
                    if vol_norm_gt > 10:
                        vol_norm_gt_str = str(int(np.round(vol_norm_gt,0))) + ' Gt'
                    elif vol_norm_gt > 1:
                        vol_norm_gt_str = str(np.round(vol_norm_gt,1)) + ' Gt'
                    else:
                        vol_norm_gt_str = str(np.round(vol_norm_gt,2)) + ' Gt'
                    ax3.text(0.95, 0.95, vol_norm_gt_str, size=10, horizontalalignment='right', 
                            verticalalignment='top', transform=ax3.transAxes)
                    
                    # Legend
                    if plot_legend:
                        ax.legend(loc=(0.02,0.02), fontsize=8, labelspacing=0.25, handlelength=1, 
                                  handletextpad=0.25, borderpad=0, ncol=1, columnspacing=0.5, frameon=False)
                        
                    # Save figure
                    fig_fn = (glac_no + '_profile_' + gcm_name + '_' + rcp + '_' + str(normyear) + '-' + str(endyear) + '.png')
                    fig.set_size_inches(4,3)
                    fig.savefig(fig_fp_ind + fig_fn, bbox_inches='tight', dpi=300)


#%% ===== PLOT DIAGNOSTICS: VOLUME, AREA, RUNOFF ======
if option_plot_diag:
    
    glac_nos = ['15.03733']
    gcms = ['CESM2']
    rcps = ['ssp245']
    
    fig_fp = netcdf_fp_sims + 'figures/'
    fig_fp_ind = fig_fp + 'ind_glaciers/'
    if not os.path.exists(fig_fp_ind):
        os.makedirs(fig_fp_ind)
        
    for glac_no in glac_nos:
    

        for gcm_name in gcms:
            
            if gcm_name in ['ERA5']:
                rcps = ['']
            else:
                rcps = rcps
                                
            for rcp in rcps:
                
                ds_stats_fp = netcdf_fp_sims + glac_no.split('.')[0].zfill(2) + '/' + gcm_name + '/' + rcp + '/stats/'
                for i in os.listdir(ds_stats_fp):
                    if i.startswith(glac_no):
                        ds_stats_fn = i
                
                ds_stats = xr.open_dataset(ds_stats_fp + ds_stats_fn)
                    
                years = ds_stats.year.values
                vol = ds_stats.glac_volume_annual[0,:].values
                area = ds_stats.glac_area_annual[0,:].values
                runoff_monthly = ds_stats.glac_runoff_monthly[0,:].values + ds_stats.offglac_runoff_monthly[0,:].values
                runoff_annual = runoff_monthly.reshape(int(runoff_monthly.shape[0]/12),12).sum(1)
                
                
                # ----- FIGURE: VOLUME CHANGE -----
                fig, ax = plt.subplots(1, 1)
        
                ax.plot(years, vol/vol[0], color='k', linestyle='-', linewidth=1, label=glac_no)
                
                ax.set_ylabel('Mass (rel. to ' + str(years[0]) + ')', size=12)
                    
                ax.set_xlim(years[0], years[-1])
                ax.xaxis.set_major_locator(MultipleLocator(20))
                ax.xaxis.set_minor_locator(MultipleLocator(10))      
                ax.set_ylim(0,1.1)
                ax.yaxis.set_major_locator(MultipleLocator(0.2))
                ax.yaxis.set_minor_locator(MultipleLocator(0.1))  
                ax.tick_params(axis='both', which='major', direction='inout', right=True)
                ax.tick_params(axis='both', which='minor', direction='in', right=True)
                
                # Save figure
                fig_fn = (glac_no + '_volume-norm_' + gcm_name + '_' + rcp + '_' + str(years[0]) + '-' + str(years[-1]) + '.png')
                fig.set_size_inches(4,3)
                fig.savefig(fig_fp_ind + fig_fn, bbox_inches='tight', dpi=300)
                
                
                # ----- FIGURE: AREA CHANGE -----
                fig, ax = plt.subplots(1, 1)
        
                ax.plot(years, area/area[0], color='k', linestyle='-', linewidth=1, label=glac_no)
                
                ax.set_ylabel('Area (rel. to ' + str(years[0]) + ')', size=12)
                    
                ax.set_xlim(years[0], years[-1])
                ax.xaxis.set_major_locator(MultipleLocator(20))
                ax.xaxis.set_minor_locator(MultipleLocator(10))      
                ax.set_ylim(0,1.1)
                ax.yaxis.set_major_locator(MultipleLocator(0.2))
                ax.yaxis.set_minor_locator(MultipleLocator(0.1))  
                ax.tick_params(axis='both', which='major', direction='inout', right=True)
                ax.tick_params(axis='both', which='minor', direction='in', right=True)
                
                # Save figure
                fig_fn = (glac_no + '_area-norm_' + gcm_name + '_' + rcp + '_' + str(years[0]) + '-' + str(years[-1]) + '.png')
                fig.set_size_inches(4,3)
                fig.savefig(fig_fp_ind + fig_fn, bbox_inches='tight', dpi=300)
                
                
                # ----- FIGURE: RUNOFF CHANGE -----
                fig, ax = plt.subplots(1, 1)
        
                ax.plot(years[0:-1], runoff_annual/runoff_annual[0], color='k', linestyle='-', linewidth=1, label=glac_no)
                
                ax.set_ylabel('Runoff (rel. to ' + str(years[0]) + ')', size=12)
                    
                ax.set_xlim(years[0], years[-1])
                ax.xaxis.set_major_locator(MultipleLocator(20))
                ax.xaxis.set_minor_locator(MultipleLocator(10))      
                ax.set_ylim(0)
                ax.yaxis.set_major_locator(MultipleLocator(0.2))
                ax.yaxis.set_minor_locator(MultipleLocator(0.1))  
                ax.tick_params(axis='both', which='major', direction='inout', right=True)
                ax.tick_params(axis='both', which='minor', direction='in', right=True)
                
                # Save figure
                fig_fn = (glac_no + '_runoff-norm_' + gcm_name + '_' + rcp + '_' + str(years[0]) + '-' + str(years[-1]) + '.png')
                fig.set_size_inches(4,3)
                fig.savefig(fig_fp_ind + fig_fn, bbox_inches='tight', dpi=300)
