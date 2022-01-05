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
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.path as mpath
#from matplotlib.pyplot import MaxNLocator
from matplotlib.lines import Line2D
#import matplotlib.patches as mpatches
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


#%% ===== Input data =====
# Script options
option_process_data = False             # Processes data for volume change and mass balance components
option_glac_debris_area = True          # Get debris-covered area for each glacier
option_disappearance = False             # Figures associated with disappearance statistics

regions = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
regions = [1,2,3,4,5,6,7,8,9,10,12,13,14,15,16,17,18,19]
#regions = [11]
#regions = [18,19]


normyear = 2015

# GCMs and RCP scenarios
gcm_names_rcps = ['CanESM2', 'CCSM4', 'CNRM-CM5', 'CSIRO-Mk3-6-0', 'GFDL-CM3', 
                  'GFDL-ESM2M', 'GISS-E2-R', 'IPSL-CM5A-LR', 'MPI-ESM-LR', 'NorESM1-M']
gcm_names_ssps = ['BCC-CSM2-MR', 'CESM2', 'CESM2-WACCM', 'EC-Earth3', 'EC-Earth3-Veg', 'FGOALS-f3-L', 
                  'GFDL-ESM4', 'INM-CM4-8', 'INM-CM5-0', 'MPI-ESM1-2-HR', 'MRI-ESM2-0', 'NorESM2-MM']
gcm_names_ssp119 = ['EC-Earth3', 'EC-Earth3-Veg', 'GFDL-ESM4', 'MRI-ESM2-0']
rcps = ['ssp119', 'ssp126', 'ssp245', 'ssp370', 'ssp585']

zipped_fp = '/Volumes/LaCie/globalsims_backup/simulations-cmip6/_zipped/'
netcdf_fp_cmip5 = '/Users/drounce/Documents/HiMAT/spc_backup/simulations/'


fig_fp = netcdf_fp_cmip5 + '/../analysis/figures/'
csv_fp = netcdf_fp_cmip5 + '/../analysis/csv/'
pickle_fp = fig_fp + '../pickle/'

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
                3:'Arctic Canada (North)',
                4:'Arctic Canada (South)',
                5:'Greenland',
                6:'Iceland',
                7:'Svalbard',
                8:'Scandinavia',
                9:'Russian Arctic',
                10:'North Asia',
                11:'Central Europe',
                12:'Caucasus/Middle East',
                13:'Central Asia',
                14:'South Asia (West)',
                15:'South Asia (East)',
                16:'Low Latitudes',
                17:'Southern Andes',
                18:'New Zealand',
                19:'Antarctica/Subantarctic'
                }



if option_process_data:

    overwrite_pickle = False
    
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
                print(reg, rcp, gcm_name)
                
                # Filename
                zipped_fp_reg = zipped_fp + str(reg).zfill(2) + '/stats/'
                zipped_fn = gcm_name + '_' + rcp + '_stats.zip'
                
                # Copy file
                copy_fp = netcdf_fp_cmip5 + str(reg).zfill(2) + '/'
                shutil.copy(zipped_fp_reg + zipped_fn, copy_fp)
                
                # Unzip filepath
                unzip_stats_fp = copy_fp + gcm_name + '/' + rcp + '/stats/'
                if not os.path.exists(unzip_stats_fp):
                    os.makedirs(unzip_stats_fp)
                with zipfile.ZipFile(copy_fp + zipped_fn, 'r') as zip_ref:
                    zip_ref.extractall(unzip_stats_fp)
                
                # Remove zipped file
                os.remove(copy_fp + zipped_fn)
                
                # Process data
                glac_stat_fns = []
                rgiid_list = []
                for i in os.listdir(unzip_stats_fp):
                    if i.endswith('.nc'):
                        glac_stat_fns.append(i)
                        rgiid_list.append(i.split('_')[0])
                glac_stat_fns = sorted(glac_stat_fns)
                rgiid_list = sorted(rgiid_list)
                

                years = np.arange(2000,2102)
                reg_vol_annual = pd.DataFrame(np.zeros((len(rgiid_list),years.shape[0])), index=rgiid_list, columns=years)
                for nglac, glac_stat_fn in enumerate(glac_stat_fns):
                    if nglac%1000==0:
                        print(reg, rcp, gcm_name, glac_stat_fn.split('_')[0])
                    ds = xr.open_dataset(unzip_stats_fp + glac_stat_fn)
                    reg_vol_annual.iloc[nglac,:] = ds.glac_volume_annual.values
                        
                reg_vol_annual_fn = str(reg).zfill(2) + '_' + zipped_fn.replace('stats.zip', 'glac_vol_annual.csv')
                csv_fp = copy_fp + '../_csv/'
                if not os.path.exists(csv_fp):
                    os.makedirs(csv_fp)
                reg_vol_annual.to_csv(csv_fp + reg_vol_annual_fn)
                
                # Remove netcdf files
                for i in os.listdir(unzip_stats_fp):
                    os.remove(unzip_stats_fp + i)
                    
                    
#%% ----- DEBRIS-COVERED AREA BY GLACIER (PRE PROCESSING) -----
if option_glac_debris_area:
    csv_fp_debris = csv_fp + 'debris/'
    if not os.path.exists(csv_fp_debris):
        os.makedirs(csv_fp_debris)
    for reg in regions:
        main_glac_rgi = modelsetup.selectglaciersrgitable(rgi_regionsO1=[reg], rgi_regionsO2 ='all', rgi_glac_number='all')
        main_glac_rgi['debris_km2'] = 0
        for nglac, glacno in enumerate(main_glac_rgi.glacno.values):
            if nglac%100 == 0:
                print(nglac, glacno)
            try:
                gdir = single_flowline_glacier_directory(glacno, logging_level='CRITICAL')
                fls = gdir.read_pickle('inversion_flowlines')
                bin_area_km2 = fls[0].widths_m * fls[0].dx_meter / 1e6
                try:
                    bin_debris_hd = fls[0].debris_hd
                except:
                    bin_debris_hd = np.zeros(bin_area_km2.shape)
                bin_debris_mask = bin_debris_hd.copy()
                bin_debris_mask[bin_debris_hd > 0] = 1
                if bin_debris_hd.sum() > 0:
                    main_glac_rgi.loc[nglac,'debris_km2'] = (bin_debris_mask * bin_area_km2).sum()
            except:
                main_glac_rgi.loc[nglac,'debris_km2'] = 0
        
        main_glac_rgi.to_csv(csv_fp_debris + str(reg).zfill(2) + '_main_glac_rgi_wdebris.csv', index=False)
                 

#%% ----- DISAPPEARANCE METRICS -----
if option_disappearance:
    
    analysis_fp = netcdf_fp_cmip5.replace('simulations','analysis')
    fig_fp = analysis_fp + 'figures/'
    if not os.path.exists(fig_fp):
        os.makedirs(fig_fp, exist_ok=True)
        
    years = np.arange(2000,2102)
        
    main_glac_rgi = None
    lost_rgiid_gcmrcp = {}
    lost_rcp_number_perc = {}
    lost_rcp_area_perc = {}
    reg_vol_annual_gcmrcp = {}
    for reg in regions:
        
        csv_fp_debris = csv_fp + 'debris/'
        main_glac_rgi_debris = pd.read_csv(csv_fp_debris + str(reg).zfill(2) + '_main_glac_rgi_wdebris.csv')
        main_glac_rgi_debris['debris_%'] = main_glac_rgi_debris['debris_km2'] / main_glac_rgi_debris['Area'] * 100
        rgiids_debris = list(main_glac_rgi_debris.RGIId)
        
        lost_rgiid_gcmrcp[reg] = {}
        lost_rcp_number_perc[reg] = {}
        lost_rcp_area_perc[reg] = {}
        reg_vol_annual_gcmrcp[reg] = {}
        
        for rcp in rcps[0:1]:
            lost_rgiid_gcmrcp[reg][rcp] = {}
            lost_rcp_number_perc[reg][rcp] = []
            lost_rcp_area_perc[reg][rcp] = []
            reg_vol_annual_gcmrcp[reg][rcp] = {}
            
            if 'rcp' in rcp:
                gcm_names = gcm_names_rcps
            elif 'ssp' in rcp:
                if rcp in ['ssp119']:
                    gcm_names = gcm_names_ssp119
                else:
                    gcm_names = gcm_names_ssps
        
            for gcm_name in gcm_names[0:1]:         
                # Load data
                csv_fp = netcdf_fp_cmip5 + '/_csv/'
                reg_vol_annual_fn = str(reg).zfill(2) + '_' + gcm_name + '_' + rcp + '_glac_vol_annual.csv'
                reg_vol_annual_raw = pd.read_csv(csv_fp + reg_vol_annual_fn, index_col=0)
                
                rgiid = [str(format(np.round(x,5),'.5f')) for x in reg_vol_annual_raw.index.values]
                
                if main_glac_rgi is None or not len(rgiid) == main_glac_rgi.shape[0]:
                    main_glac_rgi = modelsetup.selectglaciersrgitable(glac_no=rgiid)
                    main_glac_rgi_debris_idx = [rgiids_debris.index(x) for x in main_glac_rgi.RGIId.values]
                    main_glac_rgi['debris_%'] = main_glac_rgi_debris.loc[main_glac_rgi_debris_idx,'debris_%']
                
                reg_vol_annual = reg_vol_annual_raw.loc[:,:].values
                
                # Index of those that have disappeared by end of century
                lost_idx = np.where(reg_vol_annual[:,-1] == 0)[0]
                
                try:
                    count_lost = len(lost_idx)
                except:
                    count_lost = 0
                    
                main_glac_rgi_lost = main_glac_rgi.loc[lost_idx,:]
                print(reg, rcp, gcm_name, 
                      '% lost (number):', np.round(main_glac_rgi_lost.shape[0] / main_glac_rgi.shape[0] * 100, 2),
                      '% lost (area):', np.round(main_glac_rgi_lost.Area.sum() / main_glac_rgi.Area.sum() *100,2))
                
                # Record RGIIds lost
                reg_vol_annual_gcmrcp[reg][rcp][gcm_name] = reg_vol_annual
                lost_rgiid_gcmrcp[reg][rcp][gcm_name] = main_glac_rgi.RGIId.values
                lost_rcp_number_perc[reg][rcp].append(main_glac_rgi_lost.shape[0] / main_glac_rgi.shape[0] * 100)
                lost_rcp_area_perc[reg][rcp].append(main_glac_rgi_lost.Area.sum() / main_glac_rgi.Area.sum() *100)
                
                #%%
                # Debris-covered vs. Clean ice volume changes
                dc_idx = main_glac_rgi.loc[main_glac_rgi['debris_%'] > 50,:].index.values
                
        
        
    # Cummulative # and area of glaciers lost
#    for reg 

print('need to go to 2D maps to get the debris-covered area because the debris is averaged in bins, so using a mask overestimates')
            
            
    
    
#%%
print('\n\n----- TODO LIST -----')
print('- aggregate by 20-year periods')
print('- see if different relative volume changes based on debris-covered or not')
                
                
                

                
                
                
                
                
                
                
                
                
                