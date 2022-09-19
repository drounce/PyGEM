""" Analyze simulation output - mass change, runoff, etc. """

# Built-in libraries
import argparse
#from collections import OrderedDict
#import datetime
#import glob
import os
import pickle
import shutil
import time
#import zipfile
# External libraries
import cartopy
import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.path as mpath
#from matplotlib.pyplot import MaxNLocator
from matplotlib.lines import Line2D
#import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import AutoMinorLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
#from matplotlib.ticker import EngFormatter
#from matplotlib.ticker import StrMethodFormatter
#from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
#from mpl_toolkits.basemap import Basemap
import geopandas
import numpy as np
import pandas as pd
from scipy.stats import median_abs_deviation
#from scipy.stats import linregress
from scipy.ndimage import uniform_filter
#import scipy
import xarray as xr
# Local libraries
#import class_climate
#import class_mbdata
import pygem.pygem_input as pygem_prms
#import pygemfxns_gcmbiasadj as gcmbiasadj
import pygem.pygem_modelsetup as modelsetup

#from oggm import utils
from pygem.oggm_compat import single_flowline_glacier_directory
from pygem.shop import debris 
from oggm import tasks

time_start = time.time()

#%% ===== Input data =====
# Script options
option_calving_difference = True    # CALVING/DEBRIS COMPARISONS

regions = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
regions = [5,7,9,17]

warming_groups = [1.5,2,3,4,5]
warming_groups_bnds = [0.25, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
#warming_groups = [2]
#warming_groups_bnds = [0.5]

normyear = 2015

# Land-terminating assumed filepaths
netcdf_fp_cmip5_land = '/Users/drounce/Documents/HiMAT/spc_backup/simulations/'
csv_fp_land = netcdf_fp_cmip5_land + '_csv/'

netcdf_fp_cmip5_calving = '/Users/drounce/Documents/HiMAT/spc_backup/simulations_calving/'
csv_fp_calving = netcdf_fp_cmip5_calving + '_csv/'

fig_fp_calving = '/Users/drounce/Documents/HiMAT/spc_backup/analysis_calving/figures/calving_compare/'

rgiids_master_rcp = 'ssp245'
rgiids_master_gcm = 'BCC-CSM2-MR'

# GCMs and RCP scenarios
gcm_names_rcps = ['CanESM2', 'CCSM4', 'CNRM-CM5', 'CSIRO-Mk3-6-0', 'GFDL-CM3', 
                  'GFDL-ESM2M', 'GISS-E2-R', 'IPSL-CM5A-LR', 'MPI-ESM-LR', 'NorESM1-M']
gcm_names_ssps = ['BCC-CSM2-MR', 'CESM2', 'CESM2-WACCM', 'EC-Earth3', 'EC-Earth3-Veg', 'FGOALS-f3-L', 
                  'GFDL-ESM4', 'INM-CM4-8', 'INM-CM5-0', 'MPI-ESM1-2-HR', 'MRI-ESM2-0', 'NorESM2-MM']
gcm_names_ssp119 = ['EC-Earth3', 'EC-Earth3-Veg', 'GFDL-ESM4', 'MRI-ESM2-0']
#rcps = ['rcp26', 'rcp45', 'rcp85']
rcps = ['ssp119', 'ssp126', 'ssp245', 'ssp370', 'ssp585']
    
rcps_plot_mad = ['ssp126', 'ssp585']

rgiids_master_rcp = 'ssp245'
rgiids_master_gcm = 'BCC-CSM2-MR'
    
degree_size = 0.1

vn_title_dict = {'massbal':'Mass\nBalance',                                                                      
                 'precfactor':'Precipitation\nFactor',                                                              
                 'tempchange':'Temperature\nBias',                                                               
                 'ddfsnow':'Degree-Day \nFactor of Snow'}
vn_label_dict = {'massbal':'Mass Balance\n[mwea]',                                                                      
                 'precfactor':'Precipitation Factor\n[-]',                                                              
                 'tempchange':'Temperature Bias\n[$^\circ$C]',                                                               
                 'ddfsnow':'Degree Day Factor of Snow\n[mwe d$^{-1}$ $^\circ$C$^{-1}$]',
                 'dif_masschange':'Mass Balance [mwea]\n(Observation - Model)'}
vn_label_units_dict = {'massbal':'[mwea]',                                                                      
                       'precfactor':'[-]',                                                              
                       'tempchange':'[$^\circ$C]',                                                               
                       'ddfsnow':'[mwe d$^{-1}$ $^\circ$C$^{-1}$]'}
rgi_reg_dict = {'all':'Global',
                1:'Alaska',
                2:'W Canada/USA',
                3:'Arctic Canada North',
                4:'Arctic Canada South',
                5:'Greenland',
                6:'Iceland',
                7:'Svalbard',
                8:'Scandinavia',
                9:'Russian Arctic',
                10:'North Asia',
                11:'Central Europe',
                12:'Caucasus/Middle East',
                13:'Central Asia',
                14:'South Asia West',
                15:'South Asia East',
                16:'Low Latitudes',
                17:'Southern Andes',
                18:'New Zealand',
                19:'Antarctica/Subantarctic'
                }
# Colors list
rcp_colordict = {'rcp26':'#3D52A4', 'rcp45':'#76B8E5', 'rcp60':'#F47A20', 'rcp85':'#ED2024', 
                 'ssp119':'blue', 'ssp126':'#3D52A4', 'ssp245':'#76B8E5', 'ssp370':'#F47A20', 'ssp585':'#ED2024'}
rcp_styledict = {'rcp26':':', 'rcp45':':', 'rcp85':':',
                 'ssp119':'-', 'ssp126':'-', 'ssp245':'-', 'ssp370':'-', 'ssp585':'-'}
    
#%% ----- COMPUTE THE DIFFERENCES FOR TIDEWATER GLACIERS COMPARED TO ASSUMING CLEAN ICE -----
if option_calving_difference:    
    years = np.arange(2000,2102)

    # ----- PROCESS LAND GLACIERS -----
    reg_glac_vol_all_land = {}
    reg_glac_vol_all_calving = {}
    reg_rgiids = {}
    
    # Set up Global region
    reg_glac_vol_all_land['all'] = {}
    reg_glac_vol_all_calving['all'] = {}
    for rcp in rcps:
        reg_glac_vol_all_land['all'][rcp] = {}
        reg_glac_vol_all_calving['all'][rcp] = {}
        if 'rcp' in rcp:
            gcm_names = gcm_names_rcps
        elif 'ssp' in rcp:
            if rcp in ['ssp119']:
                gcm_names = gcm_names_ssp119
            else:
                gcm_names = gcm_names_ssps
        for gcm_name in gcm_names:
            reg_glac_vol_all_land['all'][rcp][gcm_name] = None
            reg_glac_vol_all_calving['all'][rcp][gcm_name] = None
            
    for nreg, reg in enumerate(regions):
        reg_glac_vol_all_land[reg] = {}
        reg_glac_vol_all_calving[reg] = {}
        
        # Load Master RGIId list that is used to correct Region 17
        # Load glacier volume data
        vol_annual_fn = str(reg).zfill(2) + '_' + rgiids_master_gcm + '_' + rgiids_master_rcp + '_glac_vol_annual.csv'
        reg_glac_vol_annual_df_land = pd.read_csv(csv_fp_land + vol_annual_fn)
        reg_glac_vol_annual_df_calving = pd.read_csv(csv_fp_calving + vol_annual_fn)
        rgiids_gcm_raw = list(reg_glac_vol_annual_df_land.values[:,0])
        rgiids_master = [str(reg) + '.' + str(int(np.round((x-reg)*1e5))).zfill(5) for x in rgiids_gcm_raw]
        reg_rgiids[reg] = rgiids_master.copy()
        reg_glac_vol_annual_gcm_land = reg_glac_vol_annual_df_land.values[:,1:]
        reg_glac_vol_annual_gcm_calving = reg_glac_vol_annual_df_calving.values[:,1:]
        reg_glac_vol_annual_master = reg_glac_vol_annual_gcm_land.copy()
        
        if nreg == 0:
            reg_rgiids['all'] = rgiids_master.copy()
        else:
            for rgiid in rgiids_master:
                reg_rgiids['all'].append(rgiid)
    
        for rcp in rcps:
            reg_glac_vol_all_land[reg][rcp] = {}
            reg_glac_vol_all_calving[reg][rcp] = {}

            if 'rcp' in rcp:
                gcm_names = gcm_names_rcps
            elif 'ssp' in rcp:
                if rcp in ['ssp119']:
                    gcm_names = gcm_names_ssp119
                else:
                    gcm_names = gcm_names_ssps
                
            for ngcm, gcm_name in enumerate(gcm_names):
                print(reg, rcp, gcm_name)
                
                # ------ Regional Glacier Volume Data -----
                # Load glacier volume data
                vol_annual_fn = str(reg).zfill(2) + '_' + gcm_name + '_' + rcp + '_glac_vol_annual.csv'
                
                reg_glac_vol_annual_df_land = pd.read_csv(csv_fp_land + vol_annual_fn)
                reg_glac_vol_annual_df_calving = pd.read_csv(csv_fp_calving + vol_annual_fn)
                rgiids_gcm_raw = list(reg_glac_vol_annual_df_land.values[:,0])
                rgiids_gcm = [str(reg) + '.' + str(int(np.round((x-reg)*1e5))).zfill(5) for x in rgiids_gcm_raw]
                reg_glac_vol_annual_gcm_land = reg_glac_vol_annual_df_land.values[:,1:]
                reg_glac_vol_annual_gcm_calving = reg_glac_vol_annual_df_calving.values[:,1:]
                
                # Check that size aligned so computing statistics on the same glaciers                            
                if not rgiids_gcm == rgiids_master:
                    rgiids_missing = list(set(rgiids_master) - set(rgiids_gcm))
                    
                    # Correct GCM by replacing with np.nan
                    reg_glac_vol_annual_gcm_raw_land = reg_glac_vol_annual_gcm_land.copy()
                    reg_glac_vol_annual_gcm_raw_calving = reg_glac_vol_annual_gcm_calving.copy()
                    reg_glac_vol_annual_gcm_land = np.zeros((reg_glac_vol_annual_master.shape[0], reg_glac_vol_annual_master.shape[1]))
                    reg_glac_vol_annual_gcm_calving = np.zeros((reg_glac_vol_annual_master.shape[0], reg_glac_vol_annual_master.shape[1]))
                    reg_glac_vol_annual_gcm_land[:,:] = np.nan
                    reg_glac_vol_annual_gcm_calving[:,:] = np.nan
                    for nrow, rgiid in enumerate(rgiids_gcm):
                        if rgiid in rgiids_master:
                            rgiid_idx = rgiids_master.index(rgiid)
                            reg_glac_vol_annual_gcm_land[rgiid_idx,:] = reg_glac_vol_annual_gcm_raw_land[nrow,:]
                            reg_glac_vol_annual_gcm_calving[rgiid_idx,:] = reg_glac_vol_annual_gcm_raw_calving[nrow,:]
                            
                # Record the data
                reg_glac_vol_all_land[reg][rcp][gcm_name] = reg_glac_vol_annual_gcm_land
                reg_glac_vol_all_calving[reg][rcp][gcm_name] = reg_glac_vol_annual_gcm_calving
                
                if reg_glac_vol_all_land['all'][rcp][gcm_name] is None:
                    reg_glac_vol_all_land['all'][rcp][gcm_name] = reg_glac_vol_annual_gcm_land 
                    reg_glac_vol_all_calving['all'][rcp][gcm_name] = reg_glac_vol_annual_gcm_calving
                else:
                    reg_glac_vol_all_land['all'][rcp][gcm_name] = np.concatenate((reg_glac_vol_all_land['all'][rcp][gcm_name], 
                                                                                  reg_glac_vol_annual_gcm_land), axis=0)
                    reg_glac_vol_all_calving['all'][rcp][gcm_name] = np.concatenate((reg_glac_vol_all_calving['all'][rcp][gcm_name], 
                                                                                     reg_glac_vol_annual_gcm_calving), axis=0)
                                       
    regions.append('all')


    #%% MULTI-GCM STATISTICS
    ds_multigcm_glac_vol_land = {}
    ds_multigcm_glac_vol_calving = {}
    for reg in regions:
        ds_multigcm_glac_vol_land[reg] = {}
        ds_multigcm_glac_vol_calving[reg] = {}
        for rcp in rcps: 
            
            if 'rcp' in rcp:
                gcm_names = gcm_names_rcps
            elif 'ssp' in rcp:
                if rcp in ['ssp119']:
                    gcm_names = gcm_names_ssp119
                else:
                    gcm_names = gcm_names_ssps
                
            for ngcm, gcm_name in enumerate(gcm_names):
                
                print(reg, rcp, gcm_name)
                
                # ----- Process Individual Glacier Volumes -----
                reg_glac_vol_gcm_land = reg_glac_vol_all_land[reg][rcp][gcm_name]
                reg_glac_vol_gcm_calving = reg_glac_vol_all_calving[reg][rcp][gcm_name]
                
                if ngcm == 0:
                    reg_glac_vol_gcm_all_land = reg_glac_vol_gcm_land[np.newaxis,:,:]
                    reg_glac_vol_gcm_all_calving = reg_glac_vol_gcm_calving[np.newaxis,:,:]
                else:
                    reg_glac_vol_gcm_all_land = np.concatenate((reg_glac_vol_gcm_all_land, 
                                                                reg_glac_vol_gcm_land[np.newaxis,:,:]), axis=0)
                    reg_glac_vol_gcm_all_calving = np.concatenate((reg_glac_vol_gcm_all_calving, 
                                                                   reg_glac_vol_gcm_calving[np.newaxis,:,:]), axis=0)
                
            # Record datasets
            ds_multigcm_glac_vol_land[reg][rcp] = reg_glac_vol_gcm_all_land
            ds_multigcm_glac_vol_calving[reg][rcp] = reg_glac_vol_gcm_all_calving
            
        
    #%%
    # Calving glacier comparison
    normyear_idx = np.where(years == normyear)[0][0]
    for reg in regions:
        main_glac_rgi = modelsetup.selectglaciersrgitable(glac_no=reg_rgiids[reg])
        main_glac_rgi_tidewater = main_glac_rgi.loc[(main_glac_rgi['TermType'] == 1) | 
                                                    (main_glac_rgi['TermType'] == 5), :]
        
        fig, ax = plt.subplots(1, 1, squeeze=False, sharex=True, sharey=False, 
                               gridspec_kw = {'wspace':0, 'hspace':0.15})
    
        idx_tidewater = []
        for nglac, glacno in enumerate(main_glac_rgi_tidewater['glacno']):
            idx_tidewater.append(reg_rgiids[reg].index(glacno))
           
        perc_dif_bins = np.arange(-50,51,5)
        bar_df = pd.DataFrame(np.zeros((len(perc_dif_bins)-1,len(rcps))), columns=rcps)
        ymax = 0
        for nrcp, rcp in enumerate(rcps):
            
#            assert True==False, 'bin based on frontal ablation rate or glacier size, and then look at the differences (see Loris figure)'
            
            # ----- Volume Change (TIDEWATER ONLY) -----
            reg_glac_vol_calving = ds_multigcm_glac_vol_calving[reg][rcp][:,idx_tidewater,:]
            reg_glac_vol_calving_med = np.nanmedian(reg_glac_vol_calving, axis=0)
            reg_glac_vol_calving_std = np.nanstd(reg_glac_vol_calving, axis=0)
            
            reg_glac_vol_calving_med_norm = reg_glac_vol_calving_med / reg_glac_vol_calving_med[:,normyear_idx][:,np.newaxis]
            reg_glac_vol_calving_std_norm = reg_glac_vol_calving_std / reg_glac_vol_calving_med[:,normyear_idx][:,np.newaxis]
            
            
            reg_glac_vol_land = ds_multigcm_glac_vol_land[reg][rcp][:,idx_tidewater,:]
            reg_glac_vol_land_med = np.nanmedian(reg_glac_vol_land, axis=0)
            reg_glac_vol_land_std = np.nanstd(reg_glac_vol_land, axis=0)
            
            reg_glac_vol_land_med_norm = reg_glac_vol_land_med / reg_glac_vol_land_med[:,normyear_idx][:,np.newaxis]
            reg_glac_vol_land_std_norm = reg_glac_vol_land_std / reg_glac_vol_land_med[:,normyear_idx][:,np.newaxis]
            
            # Differences
            reg_glac_vol_med_dif = reg_glac_vol_calving_med - reg_glac_vol_land_med
            reg_glac_vol_med_norm_dif = reg_glac_vol_calving_med_norm - reg_glac_vol_land_med_norm
            
            # Regional statistics
            reg_vol_calving_med_norm = reg_glac_vol_calving_med.sum(0) / reg_glac_vol_calving_med[:,normyear_idx].sum()
            reg_vol_land_med_norm = reg_glac_vol_land_med.sum(0) / reg_glac_vol_land_med[:,normyear_idx].sum()
            
            print(reg, rcp, 'regional dif (calving-land) (%):', np.round(100*(reg_vol_calving_med_norm[-1] - reg_vol_land_med_norm[-1])))
            
            # All glacier number statistics
            reg_glac_vol_med_dif_perc_2100 = 100 * reg_glac_vol_med_norm_dif[:,-1]
            
            perc_dif_bins = np.arange(-50,51,5)
            hist_all, bins = np.histogram(reg_glac_vol_med_dif_perc_2100, bins=perc_dif_bins)
            
            # Record in dataframe for plot
            ax[0,0].hist(reg_glac_vol_med_dif_perc_2100, bins=perc_dif_bins, alpha=0.4, color=rcp_colordict[rcp], label=rcp)
#            area_bins_str = ['< 0.10', '0.1-0.25', '0.25-0.5', '0.5-1', '1-10', '> 10']
#            hist_all, bins = np.histogram(reg_glac_vol_med_dif_perc_2100)
#            hist_all_frac = hist_all / hist_all.sum()
#            ax.bar(x=bins, height=hist_all_frac, width=1, 
#                   align='center', edgecolor='black', color='grey', zorder=0)
            
            if hist_all.max() > ymax:
                ymax = hist_all.max()
                
        
        ax[0,0].set_ylim(0,np.ceil(ymax/10)*10)
        
        # Legend
        ax[0,0].legend(loc=(0.02, 0.9),fontsize=10, ncol=1, columnspacing=0.5, labelspacing=-2.3, 
                       handlelength=1, handletextpad=0.25, borderpad=0, frameon=False)

        # Save figure
        fig_fn = ('calving_land_dif_2100_' + str(reg) + '_multigcm.png')
        fig.set_size_inches(3,3)
        fig.savefig(fig_fp_calving + fig_fn, bbox_inches='tight', dpi=300)

            
            
            
