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
import zipfile
# External libraries
import cartopy
import cartopy.crs as ccrs
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.path as mpath
#from matplotlib.pyplot import MaxNLocator
from matplotlib.lines import Line2D
import matplotlib.patches as patches
from matplotlib.ticker import AutoMinorLocator
from matplotlib.ticker import MultipleLocator
#from matplotlib.ticker import EngFormatter
#from matplotlib.ticker import StrMethodFormatter
#from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
#from mpl_toolkits.basemap import Basemap
import geopandas
import numpy as np
import pandas as pd
from scipy.stats import median_abs_deviation
#from scipy.stats import linregress
from scipy.ndimage import generic_filter
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
option_process_data = False             # Processes data for regional statistics
option_multigcm_plots_reg = True        # Multi-GCM plots of various parameters for regions

regions = [1]

#assert True==False,'Set threshold for including a run; otherwise, replace with land-terminating since not all calving modeled?'

normyear = 2015

# GCMs and RCP scenarios
gcm_names_rcps = ['CanESM2', 'CCSM4', 'CNRM-CM5', 'CSIRO-Mk3-6-0', 'GFDL-CM3', 
                  'GFDL-ESM2M', 'GISS-E2-R', 'IPSL-CM5A-LR', 'MPI-ESM-LR', 'NorESM1-M']
gcm_names_ssps = ['BCC-CSM2-MR', 'CESM2', 'CESM2-WACCM', 'EC-Earth3', 'EC-Earth3-Veg', 'FGOALS-f3-L', 
                  'GFDL-ESM4', 'INM-CM4-8', 'INM-CM5-0', 'MPI-ESM1-2-HR', 'MRI-ESM2-0', 'NorESM2-MM']
gcm_names_ssp119 = ['EC-Earth3', 'EC-Earth3-Veg', 'GFDL-ESM4', 'MRI-ESM2-0']
rcps = ['ssp126', 'ssp245', 'ssp370', 'ssp585']

netcdf_fp_cmip5 = '/Users/drounce/Documents/HiMAT/NPS_AK/simulations/'

nps_reg_dict_fn = '/Users/drounce/Documents/HiMAT/NPS_AK/rgiids_nps.csv'
nps_reg_col = 'NPS_unit'

fig_fp = netcdf_fp_cmip5 + '/../analysis/figures/'
csv_fp = netcdf_fp_cmip5 + '/../analysis/csv/'
pickle_fp = fig_fp + '../pickle/'

rgi_shp_fn = '/Users/drounce/Documents/Papers/pygem_oggm_global/qgis/rgi60_all_simplified2_robinson.shp'
rgi_regions_fn = '/Users/drounce/Documents/Papers/pygem_oggm_global/qgis/rgi60_regions_robinson-v2.shp'


rgi_reg_dict = {'all':'Global',
                'all_no519':'Global, excl. GRL and ANT',
                'global':'Global',
                1:'Alaska',
                2:'W Canada & US',
                3:'Arctic Canada North',
                4:'Arctic Canada South',
                5:'Greenland Periphery',
                6:'Iceland',
                7:'Svalbard',
                8:'Scandinavia',
                9:'Russian Arctic',
                10:'North Asia',
                11:'Central Europe',
                12:'Caucasus & Middle East',
                13:'Central Asia',
                14:'South Asia West',
                15:'South Asia East',
                16:'Low Latitudes',
                17:'Southern Andes',
                18:'New Zealand',
                19:'Antarctic & Subantarctic'
                }
rcp_label_dict = {'ssp119':'SSP1-1.9',
                  'ssp126':'SSP1-2.6',
                  'ssp245':'SSP2-4.5',
                  'ssp370':'SSP3-7.0',
                  'ssp585':'SSP5-8.5',
                  'rcp26':'RCP2.6',
                  'rcp45':'RCP4.5',
                  'rcp85':'RCP8.5'}
# Colors list
colors_rgb = [(0.00, 0.57, 0.57), (0.71, 0.43, 1.00), (0.86, 0.82, 0.00), (0.00, 0.29, 0.29), (0.00, 0.43, 0.86), 
              (0.57, 0.29, 0.00), (1.00, 0.43, 0.71), (0.43, 0.71, 1.00), (0.14, 1.00, 0.14), (1.00, 0.71, 0.47), 
              (0.29, 0.00, 0.57), (0.57, 0.00, 0.00), (0.71, 0.47, 1.00), (1.00, 1.00, 0.47)]
#gcm_colordict = dict(zip(gcm_names, colors_rgb[0:len(gcm_names)]))
rcp_colordict = {'rcp26':'#3D52A4', 'rcp45':'#76B8E5', 'rcp60':'#F47A20', 'rcp85':'#ED2024', 
                 'ssp119':'blue', 'ssp126':'#3D52A4', 'ssp245':'#76B8E5', 'ssp370':'#F47A20', 'ssp585':'#ED2024'}
rcp_styledict = {'rcp26':':', 'rcp45':':', 'rcp85':':',
                 'ssp119':'-', 'ssp126':'-', 'ssp245':'-', 'ssp370':'-', 'ssp585':'-'}

# Bounds (90% bounds --> 95% above/below given threshold)
low_percentile = 5
high_percentile = 95

colors = ['#387ea0', '#fcb200', '#d20048']
linestyles = ['-', '--', ':']


#%% ===== FUNCTIONS =====
def slr_mmSLEyr(reg_vol, reg_vol_bsl, option='oggm'):
    """ Calculate annual SLR accounting for the ice below sea level
    
    Options
    -------
    oggm : accounts for BSL and the differences in density (new)
    farinotti : accounts for BSL but not the differences in density (Farinotti et al. 2019)
    None : provides mass loss in units of mm SLE
    """
    # Farinotti et al. (2019)
#    reg_vol_asl = reg_vol - reg_vol_bsl
#    return (-1*(reg_vol_asl[:,1:] - reg_vol_asl[:,0:-1]) * 
#            pygem_prms.density_ice / pygem_prms.density_water / pygem_prms.area_ocean * 1000)
    if option == 'oggm':
        # OGGM new approach
        return (-1*(((reg_vol[:,1:] - reg_vol[:,0:-1]) * pygem_prms.density_ice / pygem_prms.density_water - 
                 (reg_vol_bsl[:,1:] - reg_vol_bsl[:,0:-1])) / pygem_prms.area_ocean * 1000))
    elif option == 'farinotti':
        reg_vol_asl = reg_vol - reg_vol_bsl
        return (-1*(reg_vol_asl[:,1:] - reg_vol_asl[:,0:-1]) * 
                pygem_prms.density_ice / pygem_prms.density_water / pygem_prms.area_ocean * 1000)
    elif option == 'None':
        # No correction
        return -1*(reg_vol[:,1:] - reg_vol[:,0:-1]) * pygem_prms.density_ice / pygem_prms.density_water / pygem_prms.area_ocean * 1000


#%%
if option_process_data:

    overwrite_pickle = True
    
    grouping = 'all'

    analysis_fp = netcdf_fp_cmip5.replace('simulations','analysis')
    fig_fp = analysis_fp + 'figures/'
    if not os.path.exists(fig_fp):
        os.makedirs(fig_fp, exist_ok=True)
    csv_fp = analysis_fp + 'csv/'
    if not os.path.exists(csv_fp):
        os.makedirs(csv_fp, exist_ok=True)
    pickle_fp = analysis_fp + 'pickle/'
    if not os.path.exists(pickle_fp):
        os.makedirs(pickle_fp, exist_ok=True)
        
#    def mwea_to_gta(mwea, area):
#        return mwea * pygem_prms.density_water * area / 1e12
    
    #%%
    for reg in regions:
        # Load glaciers
        glacno_list = []
        for rcp in rcps:
            
            if 'rcp' in rcp:
                gcm_names = gcm_names_rcps
            elif 'ssp' in rcp:
                if rcp in ['ssp119']:
                    gcm_names = gcm_names_ssp119
                else:
                    gcm_names = gcm_names_ssps
                    
            for gcm_name in gcm_names:
                
                # Filepath where glaciers are stored
                netcdf_fp = netcdf_fp_cmip5 + str(reg).zfill(2) + '/' + gcm_name + '/' + rcp + '/stats/'
                
                # Load the glaciers
                glacno_list_gcmrcp = []
                for i in os.listdir(netcdf_fp):
                    if i.endswith('.nc'):
                        glacno_list_gcmrcp.append(i.split('_')[0])
                glacno_list_gcmrcp = sorted(glacno_list_gcmrcp)
                
                print(gcm_name, rcp, 'simulated', len(glacno_list_gcmrcp), 'glaciers')
                
                # Only include the glaciers that were simulated by all GCM/RCP combinations
                if len(glacno_list) == 0:
                    glacno_list = glacno_list_gcmrcp
                else:
                    glacno_list = list(set(glacno_list).intersection(glacno_list_gcmrcp))
                glacno_list = sorted(glacno_list)
        
        # All glaciers for fraction
        main_glac_rgi_all = modelsetup.selectglaciersrgitable(rgi_regionsO1=[reg], 
                                                              rgi_regionsO2='all', rgi_glac_number='all', 
                                                              glac_no=None)
        # Glaciers with successful runs to process
        main_glac_rgi = modelsetup.selectglaciersrgitable(glac_no=glacno_list)
        
        # Missing glaciers
        glacno_list_missing = sorted(np.setdiff1d(list(main_glac_rgi_all.glacno.values), glacno_list).tolist())
        if len(glacno_list_missing) > 0:
            main_glac_rgi_missing = modelsetup.selectglaciersrgitable(glac_no=glacno_list_missing)
        
        print('\nGCM/RCPs successfully simulated:\n  -', main_glac_rgi.shape[0], 'of', main_glac_rgi_all.shape[0], 'glaciers',
              '(', np.round(main_glac_rgi.shape[0]/main_glac_rgi_all.shape[0]*100,1),'%)')
        print('  -', np.round(main_glac_rgi.Area.sum(),0), 'km2 of', np.round(main_glac_rgi_all.Area.sum(),0), 'km2',
              '(', np.round(main_glac_rgi.Area.sum()/main_glac_rgi_all.Area.sum()*100,1),'%)')

        # ===== EXPORT RESULTS =====
        success_fullfn = csv_fp + 'CMIP5_success.csv'
        success_cns = ['O1Region', 'count_success', 'count', 'count_%', 'reg_area_km2_success', 'reg_area_km2', 'reg_area_%']
        success_df_single = pd.DataFrame(np.zeros((1,len(success_cns))), columns=success_cns)
        success_df_single.loc[0,:] = [reg, main_glac_rgi.shape[0], main_glac_rgi_all.shape[0],
                                      np.round(main_glac_rgi.shape[0]/main_glac_rgi_all.shape[0]*100,2),
                                      np.round(main_glac_rgi.Area.sum(),2), np.round(main_glac_rgi_all.Area.sum(),2),
                                      np.round(main_glac_rgi.Area.sum()/main_glac_rgi_all.Area.sum()*100,2)]
        if os.path.exists(success_fullfn):
            success_df = pd.read_csv(success_fullfn)
            
            # Add or overwrite existing file
            success_idx = np.where((success_df.O1Region == reg))[0]
            if len(success_idx) > 0:
                success_df.loc[success_idx,:] = success_df_single.values
            else:
                success_df = pd.concat([success_df, success_df_single], axis=0)
                
        else:
            success_df = success_df_single
            
        success_df = success_df.sort_values('O1Region', ascending=True)
        success_df.reset_index(inplace=True, drop=True)
        success_df.to_csv(success_fullfn, index=False)                
        
        #%%
        # Unique Groups
        # O2 Regions
        unique_regO2s = np.unique(main_glac_rgi['O2Region'])

        # Elevation bins
        elev_bin_size = 10
        zmax = int(np.ceil(main_glac_rgi.Zmax.max() / elev_bin_size) * elev_bin_size) + 500
        elev_bins = np.arange(0,zmax,elev_bin_size)
        elev_bins = np.insert(elev_bins, 0, -1000)
        
        
        # Pickle datasets
        # Glacier list
        fn_reg_glacno_list = 'R' + str(reg) + '_glacno_list.pkl'
        if not os.path.exists(pickle_fp + str(reg).zfill(2) + '/'):
            os.makedirs(pickle_fp + str(reg).zfill(2) + '/')
        with open(pickle_fp + str(reg).zfill(2) + '/' + fn_reg_glacno_list, 'wb') as f:
            pickle.dump(glacno_list, f)
        
        # O2Region dict
        fn_unique_regO2s = 'R' + str(reg) + '_unique_regO2s.pkl'
        with open(pickle_fp + str(reg).zfill(2) + '/' + fn_unique_regO2s, 'wb') as f:
            pickle.dump(unique_regO2s, f)      


        years = None        
        for rcp in rcps:
                
            if 'rcp' in rcp:
                gcm_names = gcm_names_rcps
            elif 'ssp' in rcp:
                if rcp in ['ssp119']:
                    gcm_names = gcm_names_ssp119
                else:
                    gcm_names = gcm_names_ssps
            
            for gcm_name in gcm_names:

                # Filepath where glaciers are stored
                netcdf_fp_binned = netcdf_fp_cmip5 + str(reg).zfill(2) + '/' + gcm_name + '/' + rcp + '/binned/'
                netcdf_fp_stats = netcdf_fp_cmip5 + str(reg).zfill(2) + '/' + gcm_name + '/' + rcp + '/stats/'
                
                # ----- GCM/RCP PICKLE FILEPATHS AND FILENAMES -----
                pickle_fp_reg =  pickle_fp + str(reg).zfill(2) + '/O1Regions/' + gcm_name + '/' + rcp + '/'
                if not os.path.exists(pickle_fp_reg):
                    os.makedirs(pickle_fp_reg)
                pickle_fp_regO2 =  pickle_fp + str(reg).zfill(2) + '/O2Regions/' + gcm_name + '/' + rcp + '/'
                if not os.path.exists(pickle_fp_regO2):
                    os.makedirs(pickle_fp_regO2)

                # Region string prefix
                reg_rcp_gcm_str = 'R' + str(reg) + '_' + rcp + '_' + gcm_name
                regO2_rcp_gcm_str = 'R' + str(reg) + '_O2Regions_' + rcp + '_' + gcm_name
                
                # Volume
                fn_reg_vol_annual = reg_rcp_gcm_str + '_vol_annual.pkl'
                fn_regO2_vol_annual = regO2_rcp_gcm_str + '_vol_annual.pkl'
                # Volume below sea level 
                fn_reg_vol_annual_bwl = reg_rcp_gcm_str + '_vol_annual_bwl.pkl'
                fn_regO2_vol_annual_bwl = regO2_rcp_gcm_str + '_vol_annual_bwl.pkl'
                # Area 
                fn_reg_area_annual = reg_rcp_gcm_str + '_area_annual.pkl'
                fn_regO2_area_annual = regO2_rcp_gcm_str + '_area_annual.pkl'
                
                # Glacier volume
                reg_vol_annual_fn = reg_rcp_gcm_str + '_glac_vol_annual.csv'
                reg_area_annual_fn = reg_rcp_gcm_str + '_glac_area_annual.csv'
                    
                if not os.path.exists(pickle_fp_reg + fn_reg_vol_annual) or overwrite_pickle:

                    #%%
                    # Entire region
                    years = np.arange(2000,2102)
                    reg_vol_annual = None
                    reg_vol_annual_bwl = None
                    reg_area_annual = None
                    
                    # Subregion groups
                    regO2_vol_annual = None
                    regO2_vol_annual_bwl = None
                    regO2_area_annual = None
                    
                    
                    reg_vol_glac_annual = pd.DataFrame(np.zeros((len(glacno_list),years.shape[0])), index=glacno_list, columns=years)
                    reg_area_glac_annual = pd.DataFrame(np.zeros((len(glacno_list),years.shape[0])), index=glacno_list, columns=years)

                    for nglac, glacno in enumerate(glacno_list):
                        if nglac%100 == 0:
                            print(gcm_name, rcp, nglac, glacno)
                        
                        # Group indices
                        glac_idx = np.where(main_glac_rgi['glacno'] == glacno)[0][0]
                        regO2 = main_glac_rgi.loc[glac_idx, 'O2Region']
                        regO2_idx = np.where(regO2 == unique_regO2s)[0][0]
                        
                        # Filenames
                        nsim_strs = ['50', '1', '100', '150', '200', '250']
                        ds_stats = None
                        nset = -1
                        while ds_stats is None and nset <= len(nsim_strs):
                            nset += 1
                            nsim_str = nsim_strs[nset]
                            
                            try:
                                netcdf_fn_binned_ending = 'MCMC_ba1_' + nsim_str + 'sets_2000_2100_binned.nc'
                                netcdf_fn_binned = '_'.join([glacno, gcm_name, rcp, netcdf_fn_binned_ending])
        
                                netcdf_fn_stats_ending = 'MCMC_ba1_' + nsim_str + 'sets_2000_2100_all.nc'
                                netcdf_fn_stats = '_'.join([glacno, gcm_name, rcp, netcdf_fn_stats_ending])

                                # Open files
#                                ds_binned = xr.open_dataset(netcdf_fp_binned + '/' + netcdf_fn_binned)
                                ds_stats = xr.open_dataset(netcdf_fp_stats + '/' + netcdf_fn_stats)
                            except:
                                ds_stats = None
                            
                        # Years
                        if years is None:
                            years = ds_stats.year.values
                            
                        # ----- 0. Individual glaciers (volume, m3) vs. year -----
                        reg_vol_glac_annual.iloc[nglac,:] = ds_stats.glac_volume_annual.values
                        reg_area_glac_annual.iloc[nglac,:] = ds_stats.glac_area_annual.values


#                        # ----- 1. Volume (m3) vs. Year ----- 
#                        glac_vol_annual = ds_stats.glac_volume_annual.values[0,:]
#                        # All
#                        if reg_vol_annual is None:
#                            reg_vol_annual = glac_vol_annual
#                        else:
#                            reg_vol_annual += glac_vol_annual
#                        # O2Region
#                        if regO2_vol_annual is None:
#                            regO2_vol_annual = np.zeros((len(unique_regO2s),years.shape[0]))
#                            regO2_vol_annual[regO2_idx,:] = glac_vol_annual
#                        else:
#                            regO2_vol_annual[regO2_idx,:] += glac_vol_annual
#                            
#                            
#                        # ----- 2. Volume below-sea-level (m3) vs. Year ----- 
#                        #  - initial elevation is stored
#                        #  - bed elevation is constant in time
#                        #  - assume sea level is at 0 m a.s.l.
#                        z_sealevel = 0
#                        bin_z_init = ds_binned.bin_surface_h_initial.values[0,:]
#                        bin_thick_annual = ds_binned.bin_thick_annual.values[0,:,:]
#                        bin_z_bed = bin_z_init - bin_thick_annual[:,0]
#                        # Annual surface height
#                        bin_z_surf_annual = bin_z_bed[:,np.newaxis] + bin_thick_annual
#                        
#                        # Annual volume (m3)
#                        bin_vol_annual = ds_binned.bin_volume_annual.values[0,:,:]
#                        # Annual area (m2)
#                        bin_area_annual = np.zeros(bin_vol_annual.shape)
#                        bin_area_annual[bin_vol_annual > 0] = (
#                                bin_vol_annual[bin_vol_annual > 0] / bin_thick_annual[bin_vol_annual > 0])
#                        
#                        # Processed based on OGGM's _vol_below_level function
#                        bwl = (bin_z_bed[:,np.newaxis] < 0) & (bin_thick_annual > 0)
#                        if bwl.any():
#                            # Annual surface height (max of sea level for calcs)
#                            bin_z_surf_annual_bwl = bin_z_surf_annual.copy()
#                            bin_z_surf_annual_bwl[bin_z_surf_annual_bwl > z_sealevel] = z_sealevel
#                            # Annual thickness below sea level (m)
#                            bin_thick_annual_bwl = bin_thick_annual.copy()
#                            bin_thick_annual_bwl = bin_z_surf_annual_bwl - bin_z_bed[:,np.newaxis]
#                            bin_thick_annual_bwl[~bwl] = 0
#                            # Annual volume below sea level (m3)
#                            bin_vol_annual_bwl = np.zeros(bin_vol_annual.shape)
#                            bin_vol_annual_bwl[bwl] = bin_thick_annual_bwl[bwl] * bin_area_annual[bwl]
#                            glac_vol_annual_bwl = bin_vol_annual_bwl.sum(0)
#                            
#                            # All
#                            if reg_vol_annual_bwl is None:
#                                reg_vol_annual_bwl = glac_vol_annual_bwl
#                            else:
#                                reg_vol_annual_bwl += glac_vol_annual_bwl
#                            # O2Region
#                            if regO2_vol_annual_bwl is None:
#                                regO2_vol_annual_bwl = np.zeros((len(unique_regO2s),years.shape[0]))
#                                regO2_vol_annual_bwl[regO2_idx,:] = glac_vol_annual_bwl
#                            else:
#                                regO2_vol_annual_bwl[regO2_idx,:] += glac_vol_annual_bwl
#                        
#                        # ----- 4. Area vs. Time ----- 
#                        glac_area_annual = ds_stats.glac_area_annual.values[0,:]
#                        # All
#                        if reg_area_annual is None:
#                            reg_area_annual = glac_area_annual
#                        else:
#                            reg_area_annual += glac_area_annual
#                        # O2Region
#                        if regO2_area_annual is None:
#                            regO2_area_annual = np.zeros((len(unique_regO2s),years.shape[0]))
#                            regO2_area_annual[regO2_idx,:] = glac_area_annual
#                        else:
#                            regO2_area_annual[regO2_idx,:] += glac_area_annual
#                        
#                    # ===== PICKLE DATASETS =====
#                    # Volume
#                    with open(pickle_fp_reg + fn_reg_vol_annual, 'wb') as f:
#                        pickle.dump(reg_vol_annual, f)
#                    with open(pickle_fp_regO2 + fn_regO2_vol_annual, 'wb') as f:
#                        pickle.dump(regO2_vol_annual, f)
#                    # Volume below sea level 
#                    with open(pickle_fp_reg + fn_reg_vol_annual_bwl, 'wb') as f:
#                        pickle.dump(reg_vol_annual_bwl, f)
#                    with open(pickle_fp_regO2 + fn_regO2_vol_annual_bwl, 'wb') as f:
#                        pickle.dump(regO2_vol_annual_bwl, f)
#                    # Area 
#                    with open(pickle_fp_reg + fn_reg_area_annual, 'wb') as f:
#                        pickle.dump(reg_area_annual, f)
#                    with open(pickle_fp_regO2 + fn_regO2_area_annual, 'wb') as f:
#                        pickle.dump(regO2_area_annual, f)
                        
                    # ===== EXPORT CSVs =====
                    reg_vol_glac_annual.to_csv(csv_fp + reg_vol_annual_fn)
                    reg_area_glac_annual.to_csv(csv_fp + reg_area_annual_fn)

if option_multigcm_plots_reg:
    
    startyear = 2015
    endyear = 2100
    years = np.arange(2000,2101+1)
    reg = 1
    
    rcps_plot_mad = ['ssp126', 'ssp585']
    
    fig_fp_multigcm = fig_fp + 'multi_gcm/'
    if not os.path.exists(fig_fp_multigcm):
        os.makedirs(fig_fp_multigcm, exist_ok=True)
        
    # Load glaciers
    glacno_list = []
    for rcp in rcps[0:1]:
        if 'rcp' in rcp:
            gcm_names = gcm_names_rcps
        elif 'ssp' in rcp:
            if rcp in ['ssp119']:
                gcm_names = gcm_names_ssp119
            else:
                gcm_names = gcm_names_ssps
        for gcm_name in gcm_names[0:1]:
            
            # Filepath where glaciers are stored
            netcdf_fp = netcdf_fp_cmip5 + str(reg).zfill(2) + '/' + gcm_name + '/' + rcp + '/stats/'
            
            # Load the glaciers
            for i in os.listdir(netcdf_fp):
                if i.endswith('.nc'):
                    glacno_list.append(i.split('_')[0])
            glacno_list = sorted(glacno_list)
            
            print(gcm_name, rcp, 'simulated', len(glacno_list), 'glaciers')
    
    main_glac_rgi = modelsetup.selectglaciersrgitable(glac_no=glacno_list)

    
    # Load NPS dictionary
    nps_reg_csv = pd.read_csv(nps_reg_dict_fn)    
    nps_dict = dict(zip(nps_reg_csv.RGIId, nps_reg_csv[nps_reg_col]))
    main_glac_rgi['NPS_unit'] = main_glac_rgi.RGIId.map(nps_dict)
    if len(np.where(main_glac_rgi.NPS_unit.isnull())[0]) > 0:
        main_glac_rgi.loc[np.where(main_glac_rgi.NPS_unit.isnull())[0],'NPS_unit'] = 'nan'

    nps_groups = main_glac_rgi.NPS_unit.unique().tolist()
    nps_groups = sorted(nps_groups)
    regions = nps_groups

    #%%
    # Set up processing
    reg_vol_all = {}
    reg_vol_all_bwl = {}
    reg_area_all = {} 
            
    for reg in regions:
    
        reg_vol_all[reg] = {}
        reg_vol_all_bwl[reg] = {}
        reg_area_all[reg] = {}
        
        for rcp in rcps:
            reg_vol_all[reg][rcp] = {}
            reg_vol_all_bwl[reg][rcp] = {}
            reg_area_all[reg][rcp] = {}
            
            if 'rcp' in rcp:
                gcm_names = gcm_names_rcps
            elif 'ssp' in rcp:
                if rcp in ['ssp119']:
                    gcm_names = gcm_names_ssp119
                else:
                    gcm_names = gcm_names_ssps
                
            for gcm_name in gcm_names:
                print('processing:', reg, rcp, gcm_name)
                
                
                # Region string prefix
                reg_rcp_gcm_str = 'R1' + '_' + rcp + '_' + gcm_name

                # Glacier volume
                reg_vol_annual_fn = reg_rcp_gcm_str + '_glac_vol_annual.csv'
                reg_area_annual_fn = reg_rcp_gcm_str + '_glac_area_annual.csv'
                
                reg_vol_glac_annual = pd.read_csv(csv_fp + reg_vol_annual_fn, index_col=0)
                reg_area_glac_annual = pd.read_csv(csv_fp + reg_area_annual_fn, index_col=0)
                
                main_glac_rgi_subset = main_glac_rgi.loc[main_glac_rgi.NPS_unit == reg]
                
                assert main_glac_rgi.shape[0] == reg_vol_glac_annual.shape[0], '# of glaciers do not match'
                
                
                reg_vol_all[reg][rcp][gcm_name] = reg_vol_glac_annual.iloc[main_glac_rgi_subset.index.values,:].values.sum(0)
                reg_area_all[reg][rcp][gcm_name] = reg_area_glac_annual.iloc[main_glac_rgi_subset.index.values,:].values.sum(0)
                
                #%%
                
    # MULTI-GCM STATISTICS
    ds_multigcm_vol = {}
    ds_multigcm_area = {}
    for reg in regions:
        ds_multigcm_vol[reg] = {}
        ds_multigcm_area[reg] = {}
        for rcp in rcps: 
            
            if 'rcp' in rcp:
                gcm_names = gcm_names_rcps
            elif 'ssp' in rcp:
                if rcp in ['ssp119']:
                    gcm_names = gcm_names_ssp119
                else:
                    gcm_names = gcm_names_ssps
                
            for ngcm, gcm_name in enumerate(gcm_names):
                
                print('multi-gcm average:', rcp, gcm_name)
    
                reg_vol_gcm = reg_vol_all[reg][rcp][gcm_name]
                reg_area_gcm = reg_area_all[reg][rcp][gcm_name]
    
                if ngcm == 0:
                    reg_vol_gcm_all = reg_vol_gcm   
                    reg_area_gcm_all = reg_area_gcm
                else:
                    reg_vol_gcm_all = np.vstack((reg_vol_gcm_all, reg_vol_gcm))
                    reg_area_gcm_all = np.vstack((reg_area_gcm_all, reg_area_gcm))
            
            ds_multigcm_vol[reg][rcp] = reg_vol_gcm_all
            ds_multigcm_area[reg][rcp] = reg_area_gcm_all


    #%%
    normyear_idx = np.where(years == normyear)[0][0]
    # ----- FIGURE: ALL MULTI-GCM NORMALIZED VOLUME CHANGE -----
    fig = plt.figure()
    gs = fig.add_gridspec(nrows=3,ncols=3,wspace=0.3,hspace=0.2)
    ax1 = fig.add_subplot(gs[0,0])
    ax2 = fig.add_subplot(gs[0,1])
    ax3 = fig.add_subplot(gs[0,2])
    ax4 = fig.add_subplot(gs[1,0])
    ax5 = fig.add_subplot(gs[1,1])
    ax6 = fig.add_subplot(gs[1,2])
    ax7 = fig.add_subplot(gs[2,0])
    ax8 = fig.add_subplot(gs[2,1])
    ax9 = fig.add_subplot(gs[2,2])
    
    for nax, ax in enumerate([ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9]):
        
        reg = regions[nax]            
        
        for rcp in rcps:
            
            # Median and absolute median deviation
            reg_vol = ds_multigcm_vol[reg][rcp]
            reg_vol_med = np.median(reg_vol, axis=0)
            reg_vol_mad = median_abs_deviation(reg_vol, axis=0)
            
            reg_vol_med_norm = reg_vol_med / reg_vol_med[normyear_idx]
            reg_vol_mad_norm = reg_vol_mad / reg_vol_med[normyear_idx]
            
            # Record statistics
            ax.plot(years, reg_vol_med_norm, color=rcp_colordict[rcp], linestyle=rcp_styledict[rcp], 
                    linewidth=1, zorder=4, label=rcp)
            if rcp in rcps_plot_mad:
                ax.fill_between(years, 
                                reg_vol_med_norm + 1.96*reg_vol_mad_norm, 
                                reg_vol_med_norm - 1.96*reg_vol_mad_norm, 
                                alpha=0.2, facecolor=rcp_colordict[rcp], label=None)
        
        if ax in [ax1,ax4,ax7]:
            ax.set_ylabel('Mass (rel. to 2015)')
        ax.set_xlim(startyear, endyear)
        ax.xaxis.set_major_locator(MultipleLocator(50))
        ax.xaxis.set_minor_locator(MultipleLocator(10))
        ax.set_ylim(0,1.1)
        ax.yaxis.set_major_locator(MultipleLocator(0.2))
        ax.yaxis.set_minor_locator(MultipleLocator(0.1))
        ax.text(1, 1.06, reg, size=10, horizontalalignment='right', 
                verticalalignment='top', transform=ax.transAxes)
        ax.tick_params(axis='both', which='major', direction='inout', right=True)
        ax.tick_params(axis='both', which='minor', direction='in', right=True)
        
        if nax == 0:
            labels = ['SSP1-2.6', 'SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5']
#            labels = ['SSP1-2.6', 'SSP2-4.5']
            ax.legend(loc='upper right', labels=labels, fontsize=10, ncol=1, columnspacing=0.5, labelspacing=0.25, 
                      handlelength=1, handletextpad=0.25, borderpad=0, frameon=False
                      )
    # Save figure
    fig_fn = ('allregions_volchange_norm_' + str(startyear) + '-' + str(endyear) + '_multigcm-ssps.png')
    fig.set_size_inches(8.5,11)
    fig.savefig(fig_fp_multigcm + fig_fn, bbox_inches='tight', dpi=300)
    #%%
    normyear_idx = np.where(years == normyear)[0][0]
    # ----- FIGURE: ALL MULTI-GCM NORMALIZED Area CHANGE -----
    fig = plt.figure()
    gs = fig.add_gridspec(nrows=3,ncols=3,wspace=0.3,hspace=0.2)
    ax1 = fig.add_subplot(gs[0,0])
    ax2 = fig.add_subplot(gs[0,1])
    ax3 = fig.add_subplot(gs[0,2])
    ax4 = fig.add_subplot(gs[1,0])
    ax5 = fig.add_subplot(gs[1,1])
    ax6 = fig.add_subplot(gs[1,2])
    ax7 = fig.add_subplot(gs[2,0])
    ax8 = fig.add_subplot(gs[2,1])
    ax9 = fig.add_subplot(gs[2,2])
    
    for nax, ax in enumerate([ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9]):
        
        reg = regions[nax]            
        
        for rcp in rcps:
            
            # Median and absolute median deviation
            reg_area = ds_multigcm_area[reg][rcp]
            reg_area_med = np.median(reg_area, axis=0)
            reg_area_mad = median_abs_deviation(reg_area, axis=0)
            
            reg_area_med_norm = reg_area_med / reg_area_med[normyear_idx]
            reg_area_mad_norm = reg_area_mad / reg_area_med[normyear_idx]
            
            # Record statistics
            ax.plot(years, reg_area_med_norm, color=rcp_colordict[rcp], linestyle=rcp_styledict[rcp], 
                    linewidth=1, zorder=4, label=rcp)
            if rcp in rcps_plot_mad:
                ax.fill_between(years, 
                                reg_area_med_norm + 1.96*reg_area_mad_norm, 
                                reg_area_med_norm - 1.96*reg_area_mad_norm, 
                                alpha=0.2, facecolor=rcp_colordict[rcp], label=None)
        
        if ax in [ax1,ax4,ax7]:
            ax.set_ylabel('Area (rel. to 2015)')
        ax.set_xlim(startyear, endyear)
        ax.xaxis.set_major_locator(MultipleLocator(50))
        ax.xaxis.set_minor_locator(MultipleLocator(10))
        ax.set_ylim(0,1.1)
        ax.yaxis.set_major_locator(MultipleLocator(0.2))
        ax.yaxis.set_minor_locator(MultipleLocator(0.1))
        ax.text(1, 1.06, reg, size=10, horizontalalignment='right', 
                verticalalignment='top', transform=ax.transAxes)
        ax.tick_params(axis='both', which='major', direction='inout', right=True)
        ax.tick_params(axis='both', which='minor', direction='in', right=True)
        
        if nax == 0:
            labels = ['SSP1-2.6', 'SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5']
#            labels = ['SSP1-2.6', 'SSP2-4.5']
            ax.legend(loc='upper right', labels=labels, fontsize=10, ncol=1, columnspacing=0.5, labelspacing=0.25, 
                      handlelength=1, handletextpad=0.25, borderpad=0, frameon=False
                      )
    # Save figure
    fig_fn = ('allregions_areachange_norm_' + str(startyear) + '-' + str(endyear) + '_multigcm-ssps.png')
    fig.set_size_inches(8.5,11)
    fig.savefig(fig_fp_multigcm + fig_fn, bbox_inches='tight', dpi=300)
    
    #%%
