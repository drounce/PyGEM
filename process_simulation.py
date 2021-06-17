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
#import cartopy
#import matplotlib as mpl
import matplotlib.pyplot as plt
#from matplotlib.pyplot import MaxNLocator
from matplotlib.lines import Line2D
#import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator
#from matplotlib.ticker import EngFormatter
#from matplotlib.ticker import StrMethodFormatter
#from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
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
option_find_missing = False             # Checks file transfers and finds missing glaciers
option_zip_sims = False                 # Zips binned and stats output for each region (gcm/scenario)
option_process_data = False             # Processes data for regional statistics
option_process_data_nodebris = False     # Processes data for regional statistics
option_calving_mbclim_era5 = False      # mbclim of two lowest elevation bins for Will 
option_multigcm_plots_reg = False        # Multi-GCM plots of various parameters for RGI regions
option_multigcm_plots_ws = True        # Multi-GCM plots of various parameters for watersheds
option_glacier_cs_plots = False         # Multi-GCM plots of various parameters
option_debris_comparison = False         # Mutli-GCM comparison of including debris or not

regions = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
regions = [16]

# GCMs and RCP scenarios
#gcm_names = ['CanESM2', 'CCSM4', 'CNRM-CM5', 'CSIRO-Mk3-6-0', 'GFDL-CM3', 
#             'GFDL-ESM2M', 'GISS-E2-R', 'IPSL-CM5A-LR', 'MPI-ESM-LR', 'NorESM1-M']
#rcps = ['rcp26', 'rcp45', 'rcp85']

gcm_names = ['BCC-CSM2-MR', 'CESM2', 'CESM2-WACCM', 'EC-Earth3', 'EC-Earth3-Veg', 'FGOALS-f3-L', 
             'GFDL-ESM4', 'INM-CM4-8', 'INM-CM5-0', 'MPI-ESM1-2-HR', 'MRI-ESM2-0', 'NorESM2-MM']
rcps = ['ssp126', 'ssp245', 'ssp370', 'ssp585']

#netcdf_fp_cmip5 = '/Users/drounce/Documents/HiMAT/spc_backup/simulations/'
netcdf_fp_cmip5 = '/Users/drounce/Documents/HiMAT/spc_backup/simulations-nodebris/'

fig_fp = netcdf_fp_cmip5 + '/../analysis/figures/'
csv_fp = netcdf_fp_cmip5 + '/../analysis/csv/'
pickle_fp = fig_fp + '../pickle/'


def getparser():
    """
    Use argparse to add arguments from the command line

    Parameters
    ----------
    option_plot (optional) : int
        switch to plot or not
    """
    parser = argparse.ArgumentParser(description="run simulations from gcm list in parallel")
    # add arguments
    parser.add_argument('-option_plot', action='store', type=int, default=1,
                        help='switch to keep lists ordered or not')
    return parser
parser = getparser()
args = parser.parse_args()
#if args.option_plot == 1:
#    option_plot = True
#    netcdf_fp_cmip5 = '/Users/drounce/Documents/HiMAT/spc_backup/simulations/'
#else:
#    option_plot = False
#    netcdf_fp_cmip5 = pygem_prms.output_sim_fp
    
    

# Grouping
#grouping = 'all'
grouping = 'rgi_region'
#grouping = 'watershed'
#grouping = 'degree'

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
# Colors list
colors_rgb = [(0.00, 0.57, 0.57), (0.71, 0.43, 1.00), (0.86, 0.82, 0.00), (0.00, 0.29, 0.29), (0.00, 0.43, 0.86), 
              (0.57, 0.29, 0.00), (1.00, 0.43, 0.71), (0.43, 0.71, 1.00), (0.14, 1.00, 0.14), (1.00, 0.71, 0.47), 
              (0.29, 0.00, 0.57), (0.57, 0.00, 0.00), (0.71, 0.47, 1.00), (1.00, 1.00, 0.47)]
gcm_colordict = dict(zip(gcm_names, colors_rgb[0:len(gcm_names)]))
rcp_colordict = {'rcp26':'#3D52A4', 'rcp45':'#76B8E5', 'rcp60':'#F47A20', 'rcp85':'#ED2024', 
                 'ssp126':'#3D52A4', 'ssp245':'#76B8E5', 'ssp370':'#F47A20', 'ssp585':'#ED2024'}
rcp_styledict = {'rcp26':':', 'rcp45':':', 'rcp85':':',
                 'ssp126':'-', 'ssp245':'-', 'ssp370':'-', 'ssp585':'-'}

# Bounds (90% bounds --> 95% above/below given threshold)
low_percentile = 5
high_percentile = 95

colors = ['#387ea0', '#fcb200', '#d20048']
linestyles = ['-', '--', ':']


#%% ===== FUNCTIONS =====
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


def excess_meltwater_m3(glac_vol, option_lastloss=1):
    """ Excess meltwater based on running minimum glacier volume 
    
    Note: when analyzing excess meltwater for a region, if there are glaciers that gain mass, the excess meltwater will
    be zero. Consequently, the total excess meltwater will actually be more than the total mass loss because these
    positive mass balances do not "remove" total excess meltwater.
    
    Parameters
    ----------
    glac_vol : np.array
        glacier volume [km3]
    option_lastloss : int
        1 - excess meltwater based on last time glacier volume is lost for good
        0 - excess meltwater based on first time glacier volume is lost (poorly accounts for gains)
    option_lastloss = 1 calculates excess meltwater from the last time the glacier volume is lost for good
    option_lastloss = 0 calculates excess meltwater from the first time the glacier volume is lost, but does
      not recognize when the glacier volume returns
    """
    glac_vol_m3 = glac_vol * pygem_prms.density_ice / pygem_prms.density_water * 1000**3
    if option_lastloss == 1:
        glac_vol_runningmin = np.maximum.accumulate(glac_vol_m3[:,::-1],axis=1)[:,::-1]
        # initial volume sets limit of loss (gaining and then losing ice does not contribute to excess melt)
        for ncol in range(0,glac_vol_m3.shape[1]):
            mask = glac_vol_runningmin[:,ncol] > glac_vol_m3[:,0]
            glac_vol_runningmin[mask,ncol] = glac_vol_m3[mask,0]
    else:
        # Running minimum volume up until that time period (so not beyond it!)
        glac_vol_runningmin = np.minimum.accumulate(glac_vol_m3, axis=1)
    glac_excess = glac_vol_runningmin[:,:-1] - glac_vol_runningmin[:,1:] 
    return glac_excess
        

def select_groups(grouping, main_glac_rgi_all):
    """
    Select groups based on grouping
    """
    if grouping == 'rgi_region':
        groups = main_glac_rgi_all.O1Region.unique().tolist()
        group_cn = 'O1Region'
    elif grouping == 'degree':
        groups = main_glac_rgi_all.deg_id.unique().tolist()
        group_cn = 'deg_id'
    else:
        groups = ['all']
        group_cn = 'all_group'
    try:
        groups = sorted(groups, key=str.lower)
    except:
        groups = sorted(groups)
    return groups, group_cn



#%%
time_start = time.time()
if option_find_missing:
    for reg in regions:
        
        # All glaciers for fraction and missing
        main_glac_rgi_all = modelsetup.selectglaciersrgitable(rgi_regionsO1=[reg], 
                                                              rgi_regionsO2='all', rgi_glac_number='all', 
                                                              glac_no=None)
        
        # Load glaciers
        glacno_list = []
        glacno_list_gcmrcp_missing = {}
        for rcp in rcps:
            for gcm_name in gcm_names:
                
                # Filepath where glaciers are stored
                netcdf_fp = netcdf_fp_cmip5 + str(reg).zfill(2) + '/' + gcm_name + '/' + rcp + '/stats/'
                netcdf_fp_binned = netcdf_fp_cmip5 + str(reg).zfill(2) + '/' + gcm_name + '/' + rcp + '/binned/'
                
                # Load the glaciers
                glacno_list_gcmrcp = []
                for i in os.listdir(netcdf_fp):
                    if i.endswith('.nc'):
                        glacno_list_gcmrcp.append(i.split('_')[0])
                glacno_list_gcmrcp = sorted(glacno_list_gcmrcp)
                
                print(gcm_name, rcp, 'simulated', len(glacno_list_gcmrcp), 'glaciers')
                
                # Check other file too
                glacno_binned_count = 0
                for i in os.listdir(netcdf_fp_binned):
                    if i.endswith('.nc'):
                        glacno_binned_count += 1
                print('  count of stats  files:', len(glacno_list_gcmrcp))
                print('  count of binned files:', glacno_binned_count)
                
                # Only include the glaciers that were simulated by all GCM/RCP combinations
                if len(glacno_list) == 0:
                    glacno_list = glacno_list_gcmrcp
                else:
                    glacno_list = list(set(glacno_list).intersection(glacno_list_gcmrcp))
                glacno_list = sorted(glacno_list)
                
                # Missing glaciers by gcm/rcp
                glacno_list_gcmrcp_missing[gcm_name + '-' + rcp] = (
                        sorted(np.setdiff1d(list(main_glac_rgi_all.glacno.values), glacno_list_gcmrcp).tolist()))
                
#                #%%
#                # Hack to find missing compared to other runs
##                glacno_list_best = glacno_list_gcmrcp_missing['BCC-CSM2-MR-ssp126']
#                fn_reg_glacno_list = 'R' + str(reg) + '_glacno_list.pkl'
#                with open(pickle_fp + str(reg).zfill(2) + '/' + fn_reg_glacno_list, 'rb') as f:
#                    glacno_list_best = pickle.load(f)
#                
#                glacno_list_find = glacno_list_gcmrcp
#                
#                A = sorted(np.setdiff1d(glacno_list_best, glacno_list_find).tolist())
#                if len(A) > 0:
#                    print('  missing:', A)
#                #%%
                
                
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
        
        #%%




#%%
if option_zip_sims:
    """ Zip simulations """
    for reg in regions:
        
        # All glaciers for fraction
        main_glac_rgi_all = modelsetup.selectglaciersrgitable(rgi_regionsO1=[reg], 
                                                              rgi_regionsO2='all', rgi_glac_number='all', 
                                                              glac_no=None)
        # Calving glaciers
        termtype_list = [1,5]
        main_glac_rgi_calving = main_glac_rgi_all.loc[main_glac_rgi_all['TermType'].isin(termtype_list)]
        main_glac_rgi_calving.reset_index(inplace=True, drop=True)
        glacno_list_calving = list(main_glac_rgi_calving.glacno.values)
        
        for gcm_name in gcm_names:
            for rcp in rcps:
                print('zipping', reg, gcm_name, rcp)

                # Filepath where glaciers are stored
                netcdf_fp_binned = netcdf_fp_cmip5 + str(reg).zfill(2) + '/' + gcm_name + '/' + rcp + '/binned/'
                netcdf_fp_stats = netcdf_fp_cmip5 + str(reg).zfill(2) + '/' + gcm_name + '/' + rcp + '/stats/'
                
                # ----- Zip directories -----
                zipped_fp_binned = netcdf_fp_cmip5 + '_zipped/' + str(reg).zfill(2) + '/binned/'
                zipped_fp_stats = netcdf_fp_cmip5 + '_zipped/' + str(reg).zfill(2) + '/stats/'
                zipped_fn_binned = gcm_name + '_' + rcp + '_binned'
                zipped_fn_stats = gcm_name + '_' + rcp + '_stats'
                
                if not os.path.exists(zipped_fp_binned):
                    os.makedirs(zipped_fp_binned, exist_ok=True)
                if not os.path.exists(zipped_fp_stats):
                    os.makedirs(zipped_fp_stats, exist_ok=True)
                    
                shutil.make_archive(zipped_fp_binned + zipped_fn_binned, 'zip', netcdf_fp_binned)
                shutil.make_archive(zipped_fp_stats + zipped_fn_stats, 'zip', netcdf_fp_stats)
                
                if not 'nodebris' in netcdf_fp_cmip5:
                    # ----- Copy calving glaciers for comparison -----
                    if len(glacno_list_calving) > 0:
                        calving_fp_binned = netcdf_fp_cmip5 + '_calving/' + str(reg).zfill(2) + '/' + gcm_name + '/' + rcp + '/binned/'
                        calving_fp_stats = netcdf_fp_cmip5 + '_calving/' + str(reg).zfill(2) + '/' + gcm_name + '/' + rcp + '/stats/'
                        
                        if not os.path.exists(calving_fp_binned):
                            os.makedirs(calving_fp_binned, exist_ok=True)
                        if not os.path.exists(calving_fp_stats):
                            os.makedirs(calving_fp_stats, exist_ok=True)
                        
                        # Copy calving glaciers for comparison
                        for glacno in glacno_list_calving:
                            binned_fn = glacno + '_' + gcm_name + '_' + rcp + '_MCMC_ba1_50sets_2000_2100_binned.nc'
                            if os.path.exists(netcdf_fp_binned + binned_fn):
                                shutil.copyfile(netcdf_fp_binned + binned_fn, calving_fp_binned + binned_fn)
                            stats_fn = glacno + '_' + gcm_name + '_' + rcp + '_MCMC_ba1_50sets_2000_2100_all.nc'
                            if os.path.exists(netcdf_fp_stats + stats_fn):
                                shutil.copyfile(netcdf_fp_stats + stats_fn, calving_fp_stats + stats_fn)
    
#                # ----- Missing glaciers -----
#                # Filepath where glaciers are stored
#                # Load the glaciers
#                glacno_list_stats = []
#                for i in os.listdir(netcdf_fp_stats):
#                    if i.endswith('.nc'):
#                        glacno_list_stats.append(i.split('_')[0])
#                glacno_list_stats = sorted(glacno_list_stats)
#                
#                glacno_list_binned = []
#                for i in os.listdir(netcdf_fp_binned):
#                    if i.endswith('.nc'):
#                        glacno_list_binned.append(i.split('_')[0])
#                glacno_list_binned = sorted(glacno_list_binned)
#                
#                glacno_list_all = list(main_glac_rgi_all.glacno.values)
#                
#                A = np.setdiff1d(glacno_list_stats, glacno_list_binned).tolist()
#                B = np.setdiff1d(glacno_list_all, glacno_list_stats).tolist()
#                
#                print(len(B), B)
#                
#                if rcp in ['rcp26']:
#                    C = glacno_list_stats.copy()
#                elif rcp in ['rcp45']:
#                    D = glacno_list_stats.copy()
##                C_dif = np.setdiff1d(D, C).tolist()
                    
                
                #%%

if option_process_data:

    overwrite_pickle = False
    
    grouping = 'all'

    fig_fp = netcdf_fp_cmip5 + '/../analysis/figures/'
    if not os.path.exists(fig_fp):
        os.makedirs(fig_fp, exist_ok=True)
    csv_fp = netcdf_fp_cmip5 + '/../analysis/csv/'
    if not os.path.exists(csv_fp):
        os.makedirs(csv_fp, exist_ok=True)
    pickle_fp = fig_fp + '../pickle/'
    if not os.path.exists(pickle_fp):
        os.makedirs(pickle_fp, exist_ok=True)
        
#    def mwea_to_gta(mwea, area):
#        return mwea * pygem_prms.density_water * area / 1e12
    
    #%%
    for reg in regions:
        # Load glaciers
        glacno_list = []
        for rcp in rcps:
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
        
        # ----- Add Groups -----
        # Degrees (based on degree_size)
        main_glac_rgi['CenLon_round'] = np.floor(main_glac_rgi.CenLon.values/degree_size) * degree_size
        main_glac_rgi['CenLat_round'] = np.floor(main_glac_rgi.CenLat.values/degree_size) * degree_size
        deg_groups = main_glac_rgi.groupby(['CenLon_round', 'CenLat_round']).size().index.values.tolist()
        deg_dict = dict(zip(deg_groups, np.arange(0,len(deg_groups))))
        main_glac_rgi.reset_index(drop=True, inplace=True)
        cenlon_cenlat = [(main_glac_rgi.loc[x,'CenLon_round'], main_glac_rgi.loc[x,'CenLat_round']) 
                         for x in range(len(main_glac_rgi))]
        main_glac_rgi['CenLon_CenLat'] = cenlon_cenlat
        main_glac_rgi['deg_id'] = main_glac_rgi.CenLon_CenLat.map(deg_dict)
        
        # River Basin
        watershed_dict_fn = pygem_prms.main_directory + '/../qgis_datasets/rgi60_watershed_dict.csv'
        watershed_csv = pd.read_csv(watershed_dict_fn)
        watershed_dict = dict(zip(watershed_csv.RGIId, watershed_csv.watershed))
        main_glac_rgi['watershed'] = main_glac_rgi.RGIId.map(watershed_dict)
        if len(np.where(main_glac_rgi.watershed.isnull())[0]) > 0:
            main_glac_rgi.loc[np.where(main_glac_rgi.watershed.isnull())[0],'watershed'] = 'nan'
        
        #%%
        # Unique Groups
        # O2 Regions
        unique_regO2s = np.unique(main_glac_rgi['O2Region'])
        
        # Degrees
        if main_glac_rgi['deg_id'].isnull().all():
            unique_degids = None
        else:
            unique_degids = np.unique(main_glac_rgi['deg_id'])
        
        # Watersheds
        if main_glac_rgi['watershed'].isnull().all():
            unique_watersheds = None
        else:
            unique_watersheds = np.unique(main_glac_rgi['watershed'])

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
        # Watershed dict
        fn_unique_watersheds = 'R' + str(reg) + '_unique_watersheds.pkl'
        with open(pickle_fp + str(reg).zfill(2) + '/' + fn_unique_watersheds, 'wb') as f:
            pickle.dump(unique_watersheds, f) 
        # Degree ID dict
        fn_unique_degids = 'R' + str(reg) + '_unique_degids.pkl'
        with open(pickle_fp + str(reg).zfill(2) + '/' + fn_unique_degids, 'wb') as f:
            pickle.dump(unique_degids, f)
        
        fn_elev_bins = 'R' + str(reg) + '_elev_bins.pkl'
        with open(pickle_fp + str(reg).zfill(2) + '/' + fn_elev_bins, 'wb') as f:
            pickle.dump(elev_bins, f)
        
        #%%
        years = None        
        for gcm_name in gcm_names:
            for rcp in rcps:

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
                pickle_fp_watershed =  pickle_fp + str(reg).zfill(2) + '/watersheds/' + gcm_name + '/' + rcp + '/'
                if not os.path.exists(pickle_fp_watershed):
                    os.makedirs(pickle_fp_watershed)
                pickle_fp_degid =  pickle_fp + str(reg).zfill(2) + '/degids/' + gcm_name + '/' + rcp + '/'
                if not os.path.exists(pickle_fp_degid):
                    os.makedirs(pickle_fp_degid)
                # Region string prefix
                reg_rcp_gcm_str = 'R' + str(reg) + '_' + rcp + '_' + gcm_name
                regO2_rcp_gcm_str = 'R' + str(reg) + '_O2Regions_' + rcp + '_' + gcm_name
                watershed_rcp_gcm_str = 'R' + str(reg) + '_watersheds_' + rcp + '_' + gcm_name
                degid_rcp_gcm_str = 'R' + str(reg) + '_degids_' + rcp + '_' + gcm_name
                
                # Volume
                fn_reg_vol_annual = reg_rcp_gcm_str + '_vol_annual.pkl'
                fn_regO2_vol_annual = regO2_rcp_gcm_str + '_vol_annual.pkl'
                fn_watershed_vol_annual = watershed_rcp_gcm_str + '_vol_annual.pkl'
                fn_degid_vol_annual = degid_rcp_gcm_str + '_vol_annual.pkl'
                # Volume below sea level 
                fn_reg_vol_annual_bwl = reg_rcp_gcm_str + '_vol_annual_bwl.pkl'
                fn_regO2_vol_annual_bwl = regO2_rcp_gcm_str + '_vol_annual_bwl.pkl'
                fn_watershed_vol_annual_bwl = watershed_rcp_gcm_str + '_vol_annual_bwl.pkl'
                fn_degid_vol_annual_bwl = degid_rcp_gcm_str + '_vol_annual_bwl.pkl'
                # Volume below debris
                fn_reg_vol_annual_bd = reg_rcp_gcm_str + '_vol_annual_bd.pkl'
                fn_regO2_vol_annual_bd = regO2_rcp_gcm_str + '_vol_annual_bd.pkl'
                fn_watershed_vol_annual_bd = watershed_rcp_gcm_str + '_vol_annual_bd.pkl'
                fn_degid_vol_annual_bd = degid_rcp_gcm_str + '_vol_annual_bd.pkl'
                # Area 
                fn_reg_area_annual = reg_rcp_gcm_str + '_area_annual.pkl'
                fn_regO2_area_annual = regO2_rcp_gcm_str + '_area_annual.pkl'
                fn_watershed_area_annual = watershed_rcp_gcm_str + '_area_annual.pkl'
                fn_degid_area_annual = degid_rcp_gcm_str + '_area_annual.pkl'
                # Area below debris
                fn_reg_area_annual_bd = reg_rcp_gcm_str + '_area_annual_bd.pkl'
                fn_regO2_area_annual_bd = regO2_rcp_gcm_str + '_area_annual_bd.pkl'
                fn_watershed_area_annual_bd = watershed_rcp_gcm_str + '_area_annual_bd.pkl'
                fn_degid_area_annual_bd = degid_rcp_gcm_str + '_area_annual_bd.pkl'
                # Binned Volume
                fn_reg_vol_annual_binned = reg_rcp_gcm_str + '_vol_annual_binned.pkl'
                fn_regO2_vol_annual_binned = regO2_rcp_gcm_str + '_vol_annual_binned.pkl'
                fn_watershed_vol_annual_binned = watershed_rcp_gcm_str + '_vol_annual_binned.pkl'
                # Binned Volume below debris
                fn_reg_vol_annual_binned_bd = reg_rcp_gcm_str + '_vol_annual_binned_bd.pkl'
                fn_regO2_vol_annual_binned_bd = regO2_rcp_gcm_str + '_vol_annual_binned_bd.pkl'
                fn_watershed_vol_annual_binned_bd = watershed_rcp_gcm_str + '_vol_annual_binned_bd.pkl'
                # Binned Area
                fn_reg_area_annual_binned = reg_rcp_gcm_str + '_area_annual_binned.pkl'
                fn_regO2_area_annual_binned = regO2_rcp_gcm_str + '_area_annual_binned.pkl'
                fn_watershed_area_annual_binned = watershed_rcp_gcm_str + '_area_annual_binned.pkl'
                # Binned Area below debris
                fn_reg_area_annual_binned_bd = reg_rcp_gcm_str + '_area_annual_binned_bd.pkl'
                fn_regO2_area_annual_binned_bd = regO2_rcp_gcm_str + '_area_annual_binned_bd.pkl'
                fn_watershed_area_annual_binned_bd = watershed_rcp_gcm_str + '_area_annual_binned_bd.pkl'
                # Mass balance: accumulation
                fn_reg_acc_monthly = reg_rcp_gcm_str + '_acc_monthly.pkl'
                fn_regO2_acc_monthly = regO2_rcp_gcm_str + '_acc_monthly.pkl'
                fn_watershed_acc_monthly = watershed_rcp_gcm_str + '_acc_monthly.pkl'  
                # Mass balance: refreeze
                fn_reg_refreeze_monthly = reg_rcp_gcm_str + '_refreeze_monthly.pkl'
                fn_regO2_refreeze_monthly = regO2_rcp_gcm_str + '_refreeze_monthly.pkl'
                fn_watershed_refreeze_monthly = watershed_rcp_gcm_str + '_refreeze_monthly.pkl'
                # Mass balance: melt
                fn_reg_melt_monthly = reg_rcp_gcm_str + '_melt_monthly.pkl'
                fn_regO2_melt_monthly = regO2_rcp_gcm_str + '_melt_monthly.pkl'
                fn_watershed_melt_monthly = watershed_rcp_gcm_str + '_melt_monthly.pkl'
                # Mass balance: frontal ablation
                fn_reg_frontalablation_monthly = reg_rcp_gcm_str + '_frontalablation_monthly.pkl'
                fn_regO2_frontalablation_monthly = regO2_rcp_gcm_str + '_frontalablation_monthly.pkl'
                fn_watershed_frontalablation_monthly = watershed_rcp_gcm_str + '_frontalablation_monthly.pkl'
                # Mass balance: total mass balance
                fn_reg_massbaltotal_monthly = reg_rcp_gcm_str + '_massbaltotal_monthly.pkl'
                fn_regO2_massbaltotal_monthly = regO2_rcp_gcm_str + '_massbaltotal_monthly.pkl'
                fn_watershed_massbaltotal_monthly = watershed_rcp_gcm_str + '_massbaltotal_monthly.pkl' 
                fn_degid_massbaltotal_monthly = degid_rcp_gcm_str + '_massbaltotal_monthly.pkl'
                # Binned Climatic Mass Balance
                fn_reg_mbclim_annual_binned = reg_rcp_gcm_str + '_mbclim_annual_binned.pkl'
                fn_regO2_mbclim_annual_binned = regO2_rcp_gcm_str + '_mbclim_annual_binned.pkl'
                fn_watershed_mbclim_annual_binned = watershed_rcp_gcm_str + '_mbclim_annual_binned.pkl'
                # Runoff: moving-gauged
                fn_reg_runoff_monthly_moving = reg_rcp_gcm_str + '_runoff_monthly_moving.pkl'
                fn_regO2_runoff_monthly_moving = regO2_rcp_gcm_str + '_runoff_monthly_moving.pkl'
                fn_watershed_runoff_monthly_moving = watershed_rcp_gcm_str + '_runoff_monthly_moving.pkl'
                fn_degid_runoff_monthly_moving = degid_rcp_gcm_str + '_runoff_monthly_moving.pkl'
                # Runoff: fixed-gauged
                fn_reg_runoff_monthly_fixed = reg_rcp_gcm_str + '_runoff_monthly_fixed.pkl'
                fn_regO2_runoff_monthly_fixed = regO2_rcp_gcm_str + '_runoff_monthly_fixed.pkl'
                fn_watershed_runoff_monthly_fixed = watershed_rcp_gcm_str + '_runoff_monthly_fixed.pkl'
                fn_degid_runoff_monthly_fixed = degid_rcp_gcm_str + '_runoff_monthly_fixed.pkl'
                # Runoff: precipitation
                fn_reg_prec_monthly = reg_rcp_gcm_str + '_prec_monthly.pkl'
                fn_regO2_prec_monthly = regO2_rcp_gcm_str + '_prec_monthly.pkl'
                fn_watershed_prec_monthly = watershed_rcp_gcm_str + '_prec_monthly.pkl' 
                # Runoff: off-glacier precipitation
                fn_reg_offglac_prec_monthly = reg_rcp_gcm_str + '_offglac_prec_monthly.pkl'  
                fn_regO2_offglac_prec_monthly = regO2_rcp_gcm_str + '_offglac_prec_monthly.pkl'
                fn_watershed_offglac_prec_monthly = watershed_rcp_gcm_str + '_offglac_prec_monthly.pkl'
                # Runoff: off-glacier melt
                fn_reg_offglac_melt_monthly = reg_rcp_gcm_str + '_offglac_melt_monthly.pkl'
                fn_regO2_offglac_melt_monthly = regO2_rcp_gcm_str + '_offglac_melt_monthly.pkl'
                fn_watershed_offglac_melt_monthly = watershed_rcp_gcm_str + '_offglac_melt_monthly.pkl'
                # Runoff: off-glacier refreeze
                fn_reg_offglac_refreeze_monthly = reg_rcp_gcm_str + '_offglac_refreeze_monthly.pkl'
                fn_regO2_offglac_refreeze_monthly = regO2_rcp_gcm_str + '_offglac_refreeze_monthly.pkl'
                fn_watershed_offglac_refreeze_monthly = watershed_rcp_gcm_str + '_offglac_refreeze_monthly.pkl'
                # ELA
                fn_reg_ela_annual = reg_rcp_gcm_str + '_ela_annual.pkl'
                fn_regO2_ela_annual = regO2_rcp_gcm_str + '_ela_annual.pkl'
                fn_watershed_ela_annual = watershed_rcp_gcm_str + '_ela_annual.pkl'
                # AAR
                fn_reg_aar_annual = reg_rcp_gcm_str + '_aar_annual.pkl'
                fn_regO2_aar_annual = regO2_rcp_gcm_str + '_aar_annual.pkl'
                fn_watershed_aar_annual = watershed_rcp_gcm_str + '_aar_annual.pkl'
                    
                if not os.path.exists(pickle_fp_reg + fn_reg_vol_annual) or overwrite_pickle:

                    # Entire region
                    years = None
                    reg_vol_annual = None
                    reg_vol_annual_bwl = None
                    reg_vol_annual_bd = None
                    reg_area_annual = None
                    reg_area_annual_bd = None
                    reg_vol_annual_binned = None
                    reg_vol_annual_binned_bd = None
                    reg_area_annual_binned = None
                    reg_area_annual_binned_bd = None
                    reg_mbclim_annual_binned = None
                    reg_acc_monthly = None
                    reg_refreeze_monthly = None
                    reg_melt_monthly = None
                    reg_frontalablation_monthly = None
                    reg_massbaltotal_monthly = None
                    reg_runoff_monthly_fixed = None
                    reg_runoff_monthly_moving = None
                    reg_prec_monthly = None
                    reg_offglac_prec_monthly = None
                    reg_offglac_melt_monthly = None
                    reg_offglac_refreeze_monthly = None
                    reg_ela_annual = None
                    reg_ela_annual_area = None # used for weighted area calculations
                    reg_area_annual_acc = None
                    reg_area_annual_frombins = None
                    
                    # Subregion groups
                    regO2_vol_annual = None
                    regO2_vol_annual_bwl = None
                    regO2_vol_annual_bd = None
                    regO2_area_annual = None
                    regO2_area_annual_bd = None
                    regO2_vol_annual_binned = None
                    regO2_vol_annual_binned_bd = None
                    regO2_area_annual_binned = None
                    regO2_area_annual_binned_bd = None
                    regO2_mbclim_annual_binned = None
                    regO2_acc_monthly = None
                    regO2_refreeze_monthly = None
                    regO2_melt_monthly = None
                    regO2_frontalablation_monthly = None
                    regO2_massbaltotal_monthly = None
                    regO2_runoff_monthly_fixed = None
                    regO2_runoff_monthly_moving = None
                    regO2_prec_monthly = None
                    regO2_offglac_prec_monthly = None
                    regO2_offglac_melt_monthly = None
                    regO2_offglac_refreeze_monthly = None
                    regO2_ela_annual = None
                    regO2_ela_annual_area = None # used for weighted area calculations
                    regO2_area_annual_acc = None
                    regO2_area_annual_frombins = None
                    
                    # Watershed groups
                    watershed_vol_annual = None
                    watershed_vol_annual_bwl = None
                    watershed_vol_annual_bd = None
                    watershed_area_annual = None
                    watershed_area_annual_bd = None
                    watershed_vol_annual_binned = None
                    watershed_vol_annual_binned_bd = None
                    watershed_area_annual_binned = None
                    watershed_area_annual_binned_bd = None
                    watershed_mbclim_annual_binned = None
                    watershed_acc_monthly = None
                    watershed_refreeze_monthly = None
                    watershed_melt_monthly = None
                    watershed_frontalablation_monthly = None
                    watershed_massbaltotal_monthly = None
                    watershed_runoff_monthly_fixed = None
                    watershed_runoff_monthly_moving = None
                    watershed_prec_monthly = None
                    watershed_offglac_prec_monthly = None
                    watershed_offglac_melt_monthly = None
                    watershed_offglac_refreeze_monthly = None
                    watershed_ela_annual = None
                    watershed_ela_annual_area = None # used for weighted area calculations
                    watershed_area_annual_acc = None
                    watershed_area_annual_frombins = None
    
                    # Degree groups                
                    degid_vol_annual = None
                    degid_vol_annual_bwl = None
                    degid_vol_annual_bd = None
                    degid_area_annual = None
                    degid_area_annual_bd = None
                    degid_massbaltotal_monthly = None
                    degid_runoff_monthly_fixed = None
                    degid_runoff_monthly_moving = None
    
    
                    for nglac, glacno in enumerate(glacno_list):
                        if nglac%10 == 0:
                            print(gcm_name, rcp, glacno)
                        
                        # Group indices
                        glac_idx = np.where(main_glac_rgi['glacno'] == glacno)[0][0]
                        regO2 = main_glac_rgi.loc[glac_idx, 'O2Region']
                        regO2_idx = np.where(regO2 == unique_regO2s)[0][0]
                        watershed = main_glac_rgi.loc[glac_idx,'watershed']
                        watershed_idx = np.where(watershed == unique_watersheds)
                        degid = main_glac_rgi.loc[glac_idx, 'deg_id']
                        degid_idx = np.where(degid == unique_degids)[0][0]
                        
                        # Filenames
                        nsim_strs = ['50', '1', '100', '150', '200', '250']
                        ds_binned = None
                        nset = -1
                        while ds_binned is None and nset <= len(nsim_strs):
                            nset += 1
                            nsim_str = nsim_strs[nset]
                            
                            try:
                                netcdf_fn_binned_ending = 'MCMC_ba1_' + nsim_str + 'sets_2000_2100_binned.nc'
                                netcdf_fn_binned = '_'.join([glacno, gcm_name, rcp, netcdf_fn_binned_ending])
        
                                netcdf_fn_stats_ending = 'MCMC_ba1_' + nsim_str + 'sets_2000_2100_all.nc'
                                netcdf_fn_stats = '_'.join([glacno, gcm_name, rcp, netcdf_fn_stats_ending])
                                
                                # Open files
                                ds_binned = xr.open_dataset(netcdf_fp_binned + '/' + netcdf_fn_binned)
                                ds_stats = xr.open_dataset(netcdf_fp_stats + '/' + netcdf_fn_stats)
                            except:
                                ds_binned = None
                            
                        # Years
                        if years is None:
                            years = ds_stats.year.values
                            
                                
                        # ----- 1. Volume (m3) vs. Year ----- 
                        glac_vol_annual = ds_stats.glac_volume_annual.values[0,:]
                        # All
                        if reg_vol_annual is None:
                            reg_vol_annual = glac_vol_annual
                        else:
                            reg_vol_annual += glac_vol_annual
                        # O2Region
                        if regO2_vol_annual is None:
                            regO2_vol_annual = np.zeros((len(unique_regO2s),years.shape[0]))
                            regO2_vol_annual[regO2_idx,:] = glac_vol_annual
                        else:
                            regO2_vol_annual[regO2_idx,:] += glac_vol_annual
                        # Watershed
                        if watershed_vol_annual is None:
                            watershed_vol_annual = np.zeros((len(unique_watersheds),years.shape[0]))
                            watershed_vol_annual[watershed_idx,:] = glac_vol_annual
                        else:
                            watershed_vol_annual[watershed_idx,:] += glac_vol_annual
                        # DegId
                        if degid_vol_annual is None:
                            degid_vol_annual = np.zeros((len(unique_degids), years.shape[0]))
                            degid_vol_annual[degid_idx,:] = glac_vol_annual
                        else:
                            degid_vol_annual[degid_idx,:] += glac_vol_annual
                            
                            
                        # ----- 2. Volume below-sea-level (m3) vs. Year ----- 
                        #  - initial elevation is stored
                        #  - bed elevation is constant in time
                        #  - assume sea level is at 0 m a.s.l.
                        z_sealevel = 0
                        bin_z_init = ds_binned.bin_surface_h_initial.values[0,:]
                        bin_thick_annual = ds_binned.bin_thick_annual.values[0,:,:]
                        bin_z_bed = bin_z_init - bin_thick_annual[:,0]
                        # Annual surface height
                        bin_z_surf_annual = bin_z_bed[:,np.newaxis] + bin_thick_annual
                        
                        # Annual volume (m3)
                        bin_vol_annual = ds_binned.bin_volume_annual.values[0,:,:]
                        # Annual area (m2)
                        bin_area_annual = np.zeros(bin_vol_annual.shape)
                        bin_area_annual[bin_vol_annual > 0] = (
                                bin_vol_annual[bin_vol_annual > 0] / bin_thick_annual[bin_vol_annual > 0])
                        
                        # Processed based on OGGM's _vol_below_level function
                        bwl = (bin_z_bed[:,np.newaxis] < 0) & (bin_thick_annual > 0)
                        if bwl.any():
                            # Annual surface height (max of sea level for calcs)
                            bin_z_surf_annual_bwl = bin_z_surf_annual.copy()
                            bin_z_surf_annual_bwl[bin_z_surf_annual_bwl > z_sealevel] = z_sealevel
                            # Annual thickness below sea level (m)
                            bin_thick_annual_bwl = bin_thick_annual.copy()
                            bin_thick_annual_bwl = bin_z_surf_annual_bwl - bin_z_bed[:,np.newaxis]
                            bin_thick_annual_bwl[~bwl] = 0
                            # Annual volume below sea level (m3)
                            bin_vol_annual_bwl = np.zeros(bin_vol_annual.shape)
                            bin_vol_annual_bwl[bwl] = bin_thick_annual_bwl[bwl] * bin_area_annual[bwl]
                            glac_vol_annual_bwl = bin_vol_annual_bwl.sum(0)
                            
                            # All
                            if reg_vol_annual_bwl is None:
                                reg_vol_annual_bwl = glac_vol_annual_bwl
                            else:
                                reg_vol_annual_bwl += glac_vol_annual_bwl
                            # O2Region
                            if regO2_vol_annual_bwl is None:
                                regO2_vol_annual_bwl = np.zeros((len(unique_regO2s),years.shape[0]))
                                regO2_vol_annual_bwl[regO2_idx,:] = glac_vol_annual_bwl
                            else:
                                regO2_vol_annual_bwl[regO2_idx,:] += glac_vol_annual_bwl
                            # Watershed
                            if watershed_vol_annual_bwl is None:
                                watershed_vol_annual_bwl = np.zeros((len(unique_watersheds),years.shape[0]))
                                watershed_vol_annual_bwl[watershed_idx,:] = glac_vol_annual_bwl
                            else:
                                watershed_vol_annual_bwl[watershed_idx,:] += glac_vol_annual_bwl
                            # DegId
                            if degid_vol_annual_bwl is None:
                                degid_vol_annual_bwl = np.zeros((len(unique_degids), years.shape[0]))
                                degid_vol_annual_bwl[degid_idx,:] = glac_vol_annual_bwl
                            else:
                                degid_vol_annual_bwl[degid_idx,:] += glac_vol_annual_bwl
                        
    
                        # ----- 3. Volume below-debris vs. Time ----- 
                        gdir = single_flowline_glacier_directory(glacno, logging_level='CRITICAL')
                        fls = gdir.read_pickle('inversion_flowlines')
                        bin_debris_hd = np.zeros(bin_z_init.shape)
                        bin_debris_ed = np.zeros(bin_z_init.shape) + 1
                        if 'debris_hd' in dir(fls[0]):
                            bin_debris_hd[0:fls[0].debris_hd.shape[0]] = fls[0].debris_hd
                            bin_debris_ed[0:fls[0].debris_hd.shape[0]] = fls[0].debris_ed
                        if bin_debris_hd.sum() > 0:
                            bin_vol_annual_bd = np.zeros(bin_vol_annual.shape)
                            bin_vol_annual_bd[bin_debris_hd > 0, :] = bin_vol_annual[bin_debris_hd > 0, :]
                            glac_vol_annual_bd = bin_vol_annual_bd.sum(0)
                            
                            # All
                            if reg_vol_annual_bd is None:
                                reg_vol_annual_bd = glac_vol_annual_bd
                            else:
                                reg_vol_annual_bd += glac_vol_annual_bd
                            # O2Region
                            if regO2_vol_annual_bd is None:
                                regO2_vol_annual_bd = np.zeros((len(unique_regO2s),years.shape[0]))
                                regO2_vol_annual_bd[regO2_idx,:] = glac_vol_annual_bd
                            else:
                                regO2_vol_annual_bd[regO2_idx,:] += glac_vol_annual_bd
                            # Watershed
                            if watershed_vol_annual_bd is None:
                                watershed_vol_annual_bd = np.zeros((len(unique_watersheds),years.shape[0]))
                                watershed_vol_annual_bd[watershed_idx,:] = glac_vol_annual_bd
                            else:
                                watershed_vol_annual_bd[watershed_idx,:] += glac_vol_annual_bd
                            # DegId
                            if degid_vol_annual_bd is None:
                                degid_vol_annual_bd = np.zeros((len(unique_degids), years.shape[0]))
                                degid_vol_annual_bd[degid_idx,:] = glac_vol_annual_bd
                            else:
                                degid_vol_annual_bd[degid_idx,:] += glac_vol_annual_bd
                        
                        
                        # ----- 4. Area vs. Time ----- 
                        glac_area_annual = ds_stats.glac_area_annual.values[0,:]
                        # All
                        if reg_area_annual is None:
                            reg_area_annual = glac_area_annual
                        else:
                            reg_area_annual += glac_area_annual
                        # O2Region
                        if regO2_area_annual is None:
                            regO2_area_annual = np.zeros((len(unique_regO2s),years.shape[0]))
                            regO2_area_annual[regO2_idx,:] = glac_area_annual
                        else:
                            regO2_area_annual[regO2_idx,:] += glac_area_annual
                        # Watershed
                        if watershed_area_annual is None:
                            watershed_area_annual = np.zeros((len(unique_watersheds),years.shape[0]))
                            watershed_area_annual[watershed_idx,:] = glac_area_annual
                        else:
                            watershed_area_annual[watershed_idx,:] += glac_area_annual
                        # DegId
                        if degid_area_annual is None:
                            degid_area_annual = np.zeros((len(unique_degids), years.shape[0]))
                            degid_area_annual[degid_idx,:] = glac_area_annual
                        else:
                            degid_area_annual[degid_idx,:] += glac_area_annual
                        
                        
                        # ----- 5. Area below-debris vs. Time ----- 
                        if bin_debris_hd.sum() > 0:
                            bin_area_annual_bd = np.zeros(bin_area_annual.shape)
                            bin_area_annual_bd[bin_debris_hd > 0, :] = bin_area_annual[bin_debris_hd > 0, :]
                            glac_area_annual_bd = bin_area_annual_bd.sum(0)
                            
                            # All
                            if reg_area_annual_bd is None:
                                reg_area_annual_bd = glac_area_annual_bd
                            else:
                                reg_area_annual_bd += glac_area_annual_bd
                            # O2Region
                            if regO2_area_annual_bd is None:
                                regO2_area_annual_bd = np.zeros((len(unique_regO2s),years.shape[0]))
                                regO2_area_annual_bd[regO2_idx,:] = glac_area_annual_bd
                            else:
                                regO2_area_annual_bd[regO2_idx,:] += glac_area_annual_bd
                            # Watershed
                            if watershed_area_annual_bd is None:
                                watershed_area_annual_bd = np.zeros((len(unique_watersheds),years.shape[0]))
                                watershed_area_annual_bd[watershed_idx,:] = glac_area_annual_bd
                            else:
                                watershed_area_annual_bd[watershed_idx,:] += glac_area_annual_bd
                            # DegId
                            if degid_area_annual_bd is None:
                                degid_area_annual_bd = np.zeros((len(unique_degids), years.shape[0]))
                                degid_area_annual_bd[degid_idx,:] = glac_area_annual_bd
                            else:
                                degid_area_annual_bd[degid_idx,:] += glac_area_annual_bd
                        
                        
                        # ----- 6. Binned glacier volume vs. Time ----- 
                        bin_vol_annual_10m = np.zeros((len(elev_bins)-1, len(years)))
                        for ncol, year in enumerate(years):
                            bin_counts, bin_edges = np.histogram(bin_z_surf_annual[:,ncol], bins=elev_bins, 
                                                                 weights=bin_vol_annual[:,ncol])
                            bin_vol_annual_10m[:,ncol] = bin_counts
                        
                        # All
                        if reg_vol_annual_binned is None:
                            reg_vol_annual_binned = bin_vol_annual_10m
                        else:
                            reg_vol_annual_binned += bin_vol_annual_10m
                        # O2Region
                        if regO2_vol_annual_binned is None:
                            regO2_vol_annual_binned = np.zeros((len(unique_regO2s), bin_vol_annual_10m.shape[0], years.shape[0]))
                            regO2_vol_annual_binned[regO2_idx,:,:] = bin_vol_annual_10m
                        else:
                            regO2_vol_annual_binned[regO2_idx,:,:] += bin_vol_annual_10m
                        # Watershed
                        if watershed_vol_annual_binned is None:
                            watershed_vol_annual_binned = np.zeros((len(unique_watersheds), bin_vol_annual_10m.shape[0], years.shape[0]))
                            watershed_vol_annual_binned[watershed_idx,:,:] = bin_vol_annual_10m
                        else:
                            watershed_vol_annual_binned[watershed_idx,:,:] += bin_vol_annual_10m
                        
    
                        # ----- 7. Binned glacier volume below debris vs. Time ----- 
                        if bin_debris_hd.sum() > 0:
                            # Bin debris mask for the given elevation bins
                            bin_debris_mask_10m = np.zeros((bin_vol_annual_10m.shape[0]))
                            bin_counts, bin_edges = np.histogram(bin_z_init, bins=elev_bins, weights=bin_debris_hd)
                            bin_debris_mask_10m[bin_counts > 0] = 1
                            bin_vol_annual_10m_bd = bin_vol_annual_10m * bin_debris_mask_10m[:,np.newaxis]
                            
                            # All
                            if reg_vol_annual_binned_bd is None:
                                reg_vol_annual_binned_bd = bin_vol_annual_10m_bd
                            else:
                                reg_vol_annual_binned_bd += bin_vol_annual_10m_bd
                            # O2Region
                            if regO2_vol_annual_binned_bd is None:
                                regO2_vol_annual_binned_bd = np.zeros((len(unique_regO2s), bin_vol_annual_10m.shape[0], years.shape[0]))
                                regO2_vol_annual_binned_bd[regO2_idx,:,:] = bin_vol_annual_10m_bd
                            else:
                                regO2_vol_annual_binned_bd[regO2_idx,:,:] += bin_vol_annual_10m_bd
                            # Watershed
                            if watershed_vol_annual_binned_bd is None:
                                watershed_vol_annual_binned_bd = np.zeros((len(unique_watersheds), bin_vol_annual_10m.shape[0], years.shape[0]))
                                watershed_vol_annual_binned_bd[watershed_idx,:,:] = bin_vol_annual_10m_bd
                            else:
                                watershed_vol_annual_binned_bd[watershed_idx,:,:] += bin_vol_annual_10m_bd
    
    
                        # ----- 8. Binned glacier area vs. Time ----- 
                        bin_area_annual_10m = np.zeros((len(elev_bins)-1, len(years)))
                        for ncol, year in enumerate(years):
                            bin_counts, bin_edges = np.histogram(bin_z_surf_annual[:,ncol], bins=elev_bins, 
                                                                 weights=bin_area_annual[:,ncol])
                            bin_area_annual_10m[:,ncol] = bin_counts
                        
                        # All
                        if reg_area_annual_binned is None:
                            reg_area_annual_binned = bin_area_annual_10m
                        else:
                            reg_area_annual_binned += bin_area_annual_10m
                        # O2Region
                        if regO2_area_annual_binned is None:
                            regO2_area_annual_binned = np.zeros((len(unique_regO2s), bin_area_annual_10m.shape[0], years.shape[0]))
                            regO2_area_annual_binned[regO2_idx,:,:] = bin_area_annual_10m
                        else:
                            regO2_area_annual_binned[regO2_idx,:,:] += bin_area_annual_10m
                        # Watershed
                        if watershed_area_annual_binned is None:
                            watershed_area_annual_binned = np.zeros((len(unique_watersheds), bin_area_annual_10m.shape[0], years.shape[0]))
                            watershed_area_annual_binned[watershed_idx,:,:] = bin_area_annual_10m
                        else:
                            watershed_area_annual_binned[watershed_idx,:,:] += bin_area_annual_10m
    
    
                        
                        # ----- 9. Binned glacier area below debris vs. Time ----- 
                        if bin_debris_hd.sum() > 0:
                            # Bin debris mask for the given elevation bins
                            bin_debris_mask_10m = np.zeros((bin_vol_annual_10m.shape[0]))
                            bin_counts, bin_edges = np.histogram(bin_z_init, bins=elev_bins, weights=bin_debris_hd)
                            bin_debris_mask_10m[bin_counts > 0] = 1
                            bin_area_annual_10m_bd = bin_area_annual_10m * bin_debris_mask_10m[:,np.newaxis]
                            
                            # All
                            if reg_area_annual_binned_bd is None:
                                reg_area_annual_binned_bd = bin_area_annual_10m_bd
                            else:
                                reg_area_annual_binned_bd += bin_area_annual_10m_bd
                            # O2Region
                            if regO2_area_annual_binned_bd is None:
                                regO2_area_annual_binned_bd = np.zeros((len(unique_regO2s), bin_area_annual_10m.shape[0], years.shape[0]))
                                regO2_area_annual_binned_bd[regO2_idx,:,:] = bin_area_annual_10m_bd
                            else:
                                regO2_area_annual_binned_bd[regO2_idx,:,:] += bin_area_annual_10m_bd
                            # Watershed
                            if watershed_area_annual_binned_bd is None:
                                watershed_area_annual_binned_bd = np.zeros((len(unique_watersheds), bin_area_annual_10m.shape[0], years.shape[0]))
                                watershed_area_annual_binned_bd[watershed_idx,:,:] = bin_area_annual_10m_bd
                            else:
                                watershed_area_annual_binned_bd[watershed_idx,:,:] += bin_area_annual_10m_bd
                                
    
                        # ----- 10. Mass Balance Components vs. Time -----
                        # - these are only meant for monthly and/or relative purposes 
                        #   mass balance from volume change should be used for annual changes
                        # Accumulation
                        glac_acc_monthly = ds_stats.glac_acc_monthly.values[0,:]
                        # All
                        if reg_acc_monthly is None:
                            reg_acc_monthly = glac_acc_monthly
                        else:
                            reg_acc_monthly += glac_acc_monthly
                        # O2Region
                        if regO2_acc_monthly is None:
                            regO2_acc_monthly = np.zeros((len(unique_regO2s),glac_acc_monthly.shape[0]))
                            regO2_acc_monthly[regO2_idx,:] = glac_acc_monthly
                        else:
                            regO2_acc_monthly[regO2_idx,:] += glac_acc_monthly
                        # Watershed
                        if watershed_acc_monthly is None:
                            watershed_acc_monthly = np.zeros((len(unique_watersheds),glac_acc_monthly.shape[0]))
                            watershed_acc_monthly[watershed_idx,:] = glac_acc_monthly
                        else:
                            watershed_acc_monthly[watershed_idx,:] += glac_acc_monthly
                        
                        
                        # Refreeze
                        glac_refreeze_monthly = ds_stats.glac_refreeze_monthly.values[0,:]
                        # All
                        if reg_refreeze_monthly is None:
                            reg_refreeze_monthly = glac_refreeze_monthly
                        else:
                            reg_refreeze_monthly += glac_refreeze_monthly
                        # O2Region
                        if regO2_refreeze_monthly is None:
                            regO2_refreeze_monthly = np.zeros((len(unique_regO2s),glac_acc_monthly.shape[0]))
                            regO2_refreeze_monthly[regO2_idx,:] = glac_refreeze_monthly
                        else:
                            regO2_refreeze_monthly[regO2_idx,:] += glac_refreeze_monthly
                        # Watershed
                        if watershed_refreeze_monthly is None:
                            watershed_refreeze_monthly = np.zeros((len(unique_watersheds),glac_acc_monthly.shape[0]))
                            watershed_refreeze_monthly[watershed_idx,:] = glac_refreeze_monthly
                        else:
                            watershed_refreeze_monthly[watershed_idx,:] += glac_refreeze_monthly
                            
                        # Melt
                        glac_melt_monthly = ds_stats.glac_melt_monthly.values[0,:]
                        # All
                        if reg_melt_monthly is None:
                            reg_melt_monthly = glac_melt_monthly
                        else:
                            reg_melt_monthly += glac_melt_monthly
                        # O2Region
                        if regO2_melt_monthly is None:
                            regO2_melt_monthly = np.zeros((len(unique_regO2s),glac_acc_monthly.shape[0]))
                            regO2_melt_monthly[regO2_idx,:] = glac_melt_monthly
                        else:
                            regO2_melt_monthly[regO2_idx,:] += glac_melt_monthly
                        # Watershed
                        if watershed_melt_monthly is None:
                            watershed_melt_monthly = np.zeros((len(unique_watersheds),glac_acc_monthly.shape[0]))
                            watershed_melt_monthly[watershed_idx,:] = glac_melt_monthly
                        else:
                            watershed_melt_monthly[watershed_idx,:] += glac_melt_monthly
                            
                        # Frontal Ablation
                        glac_frontalablation_monthly = ds_stats.glac_frontalablation_monthly.values[0,:]
                        # All
                        if reg_frontalablation_monthly is None:
                            reg_frontalablation_monthly = glac_frontalablation_monthly
                        else:
                            reg_frontalablation_monthly += glac_frontalablation_monthly
                        # O2Region
                        if regO2_frontalablation_monthly is None:
                            regO2_frontalablation_monthly = np.zeros((len(unique_regO2s),glac_acc_monthly.shape[0]))
                            regO2_frontalablation_monthly[regO2_idx,:] = glac_frontalablation_monthly
                        else:
                            regO2_frontalablation_monthly[regO2_idx,:] += glac_frontalablation_monthly
                        # Watershed
                        if watershed_frontalablation_monthly is None:
                            watershed_frontalablation_monthly = np.zeros((len(unique_watersheds),glac_acc_monthly.shape[0]))
                            watershed_frontalablation_monthly[watershed_idx,:] = glac_frontalablation_monthly
                        else:
                            watershed_frontalablation_monthly[watershed_idx,:] += glac_frontalablation_monthly
                            
                        # Total Mass Balance
                        glac_massbaltotal_monthly = ds_stats.glac_massbaltotal_monthly.values[0,:]
                        # All
                        if reg_massbaltotal_monthly is None:
                            reg_massbaltotal_monthly = glac_massbaltotal_monthly
                        else:
                            reg_massbaltotal_monthly += glac_massbaltotal_monthly
                        # O2Region
                        if regO2_massbaltotal_monthly is None:
                            regO2_massbaltotal_monthly = np.zeros((len(unique_regO2s),glac_acc_monthly.shape[0]))
                            regO2_massbaltotal_monthly[regO2_idx,:] = glac_massbaltotal_monthly
                        else:
                            regO2_massbaltotal_monthly[regO2_idx,:] += glac_massbaltotal_monthly
                        # Watershed
                        if watershed_massbaltotal_monthly is None:
                            watershed_massbaltotal_monthly = np.zeros((len(unique_watersheds),glac_acc_monthly.shape[0]))
                            watershed_massbaltotal_monthly[watershed_idx,:] = glac_massbaltotal_monthly
                        else:
                            watershed_massbaltotal_monthly[watershed_idx,:] += glac_massbaltotal_monthly
                        # DegId
                        if degid_massbaltotal_monthly is None:
                            degid_massbaltotal_monthly = np.zeros((len(unique_degids),glac_acc_monthly.shape[0]))
                            degid_massbaltotal_monthly[degid_idx,:] = glac_massbaltotal_monthly
                        else:
                            degid_massbaltotal_monthly[degid_idx,:] += glac_massbaltotal_monthly
                        
                        
                        
                        # ----- 11. Binned Climatic Mass Balance vs. Time -----
                        # - Various mass balance datasets may have slight mismatch due to averaging
                        #   ex. mbclim_annual was reported in mwe, so the area average will cause difference
                        #   ex. mbtotal_monthly was averaged on a monthly basis, so the temporal average will cause difference
                        bin_mbclim_annual = ds_binned.bin_massbalclim_annual.values[0,:,:]
                        bin_mbclim_annual_m3we = bin_mbclim_annual * bin_area_annual
    
    #                    glac_massbaltotal_annual_0 = bin_mbclim_annual_m3we.sum(0)
    #                    glac_massbaltotal_annual_1 = glac_massbaltotal_monthly.reshape(-1,12).sum(1)
    #                    glac_massbaltotal_annual_2 = ((glac_vol_annual[1:] - glac_vol_annual[0:-1]) * 
    #                                                  pygem_prms.density_ice / pygem_prms.density_water)
                        
                        bin_mbclim_annual_10m = np.zeros((len(elev_bins)-1, len(years)))
                        for ncol, year in enumerate(years):
                            bin_counts, bin_edges = np.histogram(bin_z_surf_annual[:,ncol], bins=elev_bins, 
                                                                 weights=bin_mbclim_annual_m3we[:,ncol])
                            bin_mbclim_annual_10m[:,ncol] = bin_counts
                        # All
                        if reg_mbclim_annual_binned is None:
                            reg_mbclim_annual_binned = bin_mbclim_annual_10m
                        else:
                            reg_mbclim_annual_binned += bin_mbclim_annual_10m
                        # O2Region
                        if regO2_mbclim_annual_binned is None:
                            regO2_mbclim_annual_binned = np.zeros((len(unique_regO2s), bin_mbclim_annual_10m.shape[0], years.shape[0]))
                            regO2_mbclim_annual_binned[regO2_idx,:,:] = bin_mbclim_annual_10m
                        else:
                            regO2_mbclim_annual_binned[regO2_idx,:,:] += bin_mbclim_annual_10m
                        # Watershed
                        if watershed_mbclim_annual_binned is None:
                            watershed_mbclim_annual_binned = np.zeros((len(unique_watersheds), bin_mbclim_annual_10m.shape[0], years.shape[0]))
                            watershed_mbclim_annual_binned[watershed_idx,:,:] = bin_mbclim_annual_10m
                        else:
                            watershed_mbclim_annual_binned[watershed_idx,:,:] += bin_mbclim_annual_10m
    
                        
                        # ----- 12. Runoff vs. Time -----
                        glac_runoff_monthly = ds_stats.glac_runoff_monthly.values[0,:]
                        # Moving-gauge Runoff vs. Time
                        # All
                        if reg_runoff_monthly_moving is None:
                            reg_runoff_monthly_moving = glac_runoff_monthly
                        else:
                            reg_runoff_monthly_moving += glac_runoff_monthly
                        # O2Region
                        if regO2_runoff_monthly_moving is None:
                            regO2_runoff_monthly_moving = np.zeros((len(unique_regO2s),glac_acc_monthly.shape[0]))
                            regO2_runoff_monthly_moving[regO2_idx,:] = glac_runoff_monthly
                        else:
                            regO2_runoff_monthly_moving[regO2_idx,:] += glac_runoff_monthly
                        # watershed
                        if watershed_runoff_monthly_moving is None:
                            watershed_runoff_monthly_moving = np.zeros((len(unique_watersheds),glac_acc_monthly.shape[0]))
                            watershed_runoff_monthly_moving[watershed_idx,:] = glac_runoff_monthly
                        else:
                            watershed_runoff_monthly_moving[watershed_idx,:] += glac_runoff_monthly
                        # DegId
                        if degid_runoff_monthly_moving is None:
                            degid_runoff_monthly_moving = np.zeros((len(unique_degids),glac_acc_monthly.shape[0]))
                            degid_runoff_monthly_moving[degid_idx,:] = glac_runoff_monthly
                        else:
                            degid_runoff_monthly_moving[degid_idx,:] += glac_runoff_monthly
                            
                        # Fixed-gauge Runoff vs. Time
                        offglac_runoff_monthly = ds_stats.offglac_runoff_monthly.values[0,:]
                        glac_runoff_monthly_fixed = glac_runoff_monthly + offglac_runoff_monthly
                        # All
                        if reg_runoff_monthly_fixed is None:
                            reg_runoff_monthly_fixed = glac_runoff_monthly_fixed
                        else:
                            reg_runoff_monthly_fixed += glac_runoff_monthly_fixed
                        # O2Region
                        if regO2_runoff_monthly_fixed is None:
                            regO2_runoff_monthly_fixed = np.zeros((len(unique_regO2s),glac_acc_monthly.shape[0]))
                            regO2_runoff_monthly_fixed[regO2_idx,:] = glac_runoff_monthly_fixed
                        else:
                            regO2_runoff_monthly_fixed[regO2_idx,:] += glac_runoff_monthly_fixed
                        # Watershed
                        if watershed_runoff_monthly_fixed is None:
                            watershed_runoff_monthly_fixed = np.zeros((len(unique_watersheds),glac_acc_monthly.shape[0]))
                            watershed_runoff_monthly_fixed[watershed_idx,:] = glac_runoff_monthly_fixed
                        else:
                            watershed_runoff_monthly_fixed[watershed_idx,:] += glac_runoff_monthly_fixed
                        # DegId
                        if degid_runoff_monthly_fixed is None:
                            degid_runoff_monthly_fixed = np.zeros((len(unique_degids),glac_acc_monthly.shape[0]))
                            degid_runoff_monthly_fixed[degid_idx,:] = glac_runoff_monthly_fixed
                        else:
                            degid_runoff_monthly_fixed[degid_idx,:] += glac_runoff_monthly_fixed
                        
                        
                        # Runoff Components
                        # Precipitation
                        glac_prec_monthly = ds_stats.glac_prec_monthly.values[0,:]
                        # All
                        if reg_prec_monthly is None:
                            reg_prec_monthly = glac_prec_monthly
                        else:
                            reg_prec_monthly += glac_prec_monthly
                        # O2Region
                        if regO2_prec_monthly is None:
                            regO2_prec_monthly = np.zeros((len(unique_regO2s),glac_prec_monthly.shape[0]))
                            regO2_prec_monthly[regO2_idx,:] = glac_prec_monthly
                        else:
                            regO2_prec_monthly[regO2_idx,:] += glac_prec_monthly
                        # Watershed
                        if watershed_prec_monthly is None:
                            watershed_prec_monthly = np.zeros((len(unique_watersheds),glac_prec_monthly.shape[0]))
                            watershed_prec_monthly[watershed_idx,:] = glac_prec_monthly
                        else:
                            watershed_prec_monthly[watershed_idx,:] += glac_prec_monthly
                            
                        # Off-glacier Precipitation
                        offglac_prec_monthly = ds_stats.offglac_prec_monthly.values[0,:]
                        # All
                        if reg_offglac_prec_monthly is None:
                            reg_offglac_prec_monthly = offglac_prec_monthly
                        else:
                            reg_offglac_prec_monthly += offglac_prec_monthly
                        # O2Region
                        if regO2_offglac_prec_monthly is None:
                            regO2_offglac_prec_monthly = np.zeros((len(unique_regO2s),glac_prec_monthly.shape[0]))
                            regO2_offglac_prec_monthly[regO2_idx,:] = offglac_prec_monthly
                        else:
                            regO2_offglac_prec_monthly[regO2_idx,:] += offglac_prec_monthly
                        # Watershed
                        if watershed_offglac_prec_monthly is None:
                            watershed_offglac_prec_monthly = np.zeros((len(unique_watersheds),glac_prec_monthly.shape[0]))
                            watershed_offglac_prec_monthly[watershed_idx,:] = offglac_prec_monthly
                        else:
                            watershed_offglac_prec_monthly[watershed_idx,:] += offglac_prec_monthly
                            
                        # Off-glacier Melt
                        offglac_melt_monthly = ds_stats.offglac_melt_monthly.values[0,:]
                        # All
                        if reg_offglac_melt_monthly is None:
                            reg_offglac_melt_monthly = offglac_melt_monthly
                        else:
                            reg_offglac_melt_monthly += offglac_melt_monthly
                        # O2Region
                        if regO2_offglac_melt_monthly is None:
                            regO2_offglac_melt_monthly = np.zeros((len(unique_regO2s),glac_melt_monthly.shape[0]))
                            regO2_offglac_melt_monthly[regO2_idx,:] = offglac_melt_monthly
                        else:
                            regO2_offglac_melt_monthly[regO2_idx,:] += offglac_melt_monthly
                        # Watershed
                        if watershed_offglac_melt_monthly is None:
                            watershed_offglac_melt_monthly = np.zeros((len(unique_watersheds),glac_melt_monthly.shape[0]))
                            watershed_offglac_melt_monthly[watershed_idx,:] = offglac_melt_monthly
                        else:
                            watershed_offglac_melt_monthly[watershed_idx,:] += offglac_melt_monthly
                            
                        # Off-glacier Refreeze
                        # All
                        offglac_refreeze_monthly = ds_stats.offglac_refreeze_monthly.values[0,:]
                        if reg_offglac_refreeze_monthly is None:
                            reg_offglac_refreeze_monthly = offglac_refreeze_monthly
                        else:
                            reg_offglac_refreeze_monthly += offglac_refreeze_monthly
                        # O2Region
                        if regO2_offglac_refreeze_monthly is None:
                            regO2_offglac_refreeze_monthly = np.zeros((len(unique_regO2s),glac_refreeze_monthly.shape[0]))
                            regO2_offglac_refreeze_monthly[regO2_idx,:] = offglac_refreeze_monthly
                        else:
                            regO2_offglac_refreeze_monthly[regO2_idx,:] += offglac_refreeze_monthly
                        # Watershed
                        if watershed_offglac_refreeze_monthly is None:
                            watershed_offglac_refreeze_monthly = np.zeros((len(unique_watersheds),glac_refreeze_monthly.shape[0]))
                            watershed_offglac_refreeze_monthly[watershed_idx,:] = offglac_refreeze_monthly
                        else:
                            watershed_offglac_refreeze_monthly[watershed_idx,:] += offglac_refreeze_monthly
    
                        # ----- 13. ELA vs. Time -----
                        glac_ela_annual = ds_stats.glac_ELA_annual.values[0,:]
                        if np.isnan(glac_ela_annual).any():
                            # Quality control nan values 
                            #  - replace with max elev because occur when entire glacier has neg mb
                            bin_z_surf_annual_glaconly = bin_z_surf_annual.copy()
                            bin_z_surf_annual_glaconly[bin_thick_annual == 0] = np.nan
                            zmax_annual = np.nanmax(bin_z_surf_annual_glaconly, axis=0)
                            glac_ela_annual[np.isnan(glac_ela_annual)] = zmax_annual[np.isnan(glac_ela_annual)]
    
                        # Area-weighted ELA
                        # All
                        if reg_ela_annual is None:
                            reg_ela_annual = glac_ela_annual
                            reg_ela_annual_area = glac_area_annual.copy()
                        else:
                            # Use index to avoid dividing by 0 when glacier completely melts                            
                            ela_idx = np.where(reg_ela_annual_area + glac_area_annual > 0)[0]
                            reg_ela_annual[ela_idx] = (
                                    (reg_ela_annual[ela_idx] * reg_ela_annual_area[ela_idx] + 
                                     glac_ela_annual[ela_idx] * glac_area_annual[ela_idx]) / 
                                    (reg_ela_annual_area[ela_idx] + glac_area_annual[ela_idx]))
                            reg_ela_annual_area += glac_area_annual
                        
                        # O2Region
                        if regO2_ela_annual is None:
                            regO2_ela_annual = np.zeros((len(unique_regO2s),years.shape[0]))
                            regO2_ela_annual[regO2_idx,:] = glac_ela_annual
                            regO2_ela_annual_area = np.zeros((len(unique_regO2s),years.shape[0]))
                            regO2_ela_annual_area[regO2_idx,:] = glac_area_annual.copy()
                        else:
                            ela_idx = np.where(regO2_ela_annual_area[regO2_idx,:] + glac_area_annual > 0)[0]
                            regO2_ela_annual[regO2_idx,ela_idx] = (
                                    (regO2_ela_annual[regO2_idx,ela_idx] * regO2_ela_annual_area[regO2_idx,ela_idx] + 
                                     glac_ela_annual[ela_idx] * glac_area_annual[ela_idx]) / 
                                     (regO2_ela_annual_area[regO2_idx,ela_idx] + glac_area_annual[ela_idx]))
                            regO2_ela_annual_area[regO2_idx,:] += glac_area_annual
                        
                        # Watershed
                        if watershed_ela_annual is None:
                            watershed_ela_annual = np.zeros((len(unique_watersheds),years.shape[0]))
                            watershed_ela_annual[watershed_idx,:] = glac_ela_annual
                            watershed_ela_annual_area = np.zeros((len(unique_watersheds),years.shape[0]))
                            watershed_ela_annual_area[watershed_idx,:] = glac_area_annual.copy()
                        else:
                            ela_idx = np.where(watershed_ela_annual_area[watershed_idx,:] + glac_area_annual > 0)[0]
                            watershed_ela_annual[watershed_idx,ela_idx] = (
                                    (watershed_ela_annual[watershed_idx,ela_idx] * watershed_ela_annual_area[watershed_idx,ela_idx] + 
                                     glac_ela_annual[ela_idx] * glac_area_annual[ela_idx]) / 
                                     (watershed_ela_annual_area[watershed_idx,ela_idx] + glac_area_annual[ela_idx]))
                            watershed_ela_annual_area[watershed_idx,:] += glac_area_annual
                        

                        # ----- 14. AAR vs. Time -----
                        #  - averaging issue with bin_area_annual.sum(0) != glac_area_annual
                        #  - hence only use these 
                        bin_area_annual_acc = bin_area_annual.copy()
                        bin_area_annual_acc[bin_mbclim_annual <= 0] = 0
                        glac_area_annual_acc = bin_area_annual_acc.sum(0)
                        glac_area_annual_frombins = bin_area_annual.sum(0)
                        
                        # All
                        if reg_area_annual_acc is None:
                            reg_area_annual_acc = glac_area_annual_acc.copy()
                            reg_area_annual_frombins = glac_area_annual_frombins.copy()
                            reg_aar_annual = np.zeros(reg_area_annual_acc.shape)
                            reg_aar_annual[reg_area_annual_frombins > 0] = (
                                    reg_area_annual_acc[reg_area_annual_frombins > 0] / 
                                    reg_area_annual_frombins[reg_area_annual_frombins > 0])
                        else:
                            reg_area_annual_acc += glac_area_annual_acc
                            reg_area_annual_frombins += glac_area_annual_frombins
                            reg_aar_annual = np.zeros(reg_area_annual_acc.shape)
                            reg_aar_annual[reg_area_annual_frombins > 0] = (
                                    reg_area_annual_acc[reg_area_annual_frombins > 0] / 
                                    reg_area_annual_frombins[reg_area_annual_frombins > 0])
                        # O2Regions
                        if regO2_area_annual_acc is None:
                            regO2_area_annual_acc = np.zeros((len(unique_regO2s),years.shape[0]))
                            regO2_area_annual_acc[regO2_idx,:] = glac_area_annual_acc.copy()
                            regO2_area_annual_frombins = np.zeros((len(unique_regO2s),years.shape[0]))
                            regO2_area_annual_frombins[regO2_idx,:] = glac_area_annual_frombins.copy()
                            regO2_aar_annual = np.zeros((len(unique_regO2s),years.shape[0]))
                            regO2_aar_annual[regO2_idx,:][regO2_area_annual_frombins[regO2_idx,:] > 0] = (
                                    regO2_area_annual_acc[regO2_idx,:][regO2_area_annual_frombins[regO2_idx,:] > 0] / 
                                    regO2_area_annual_frombins[regO2_idx,:][regO2_area_annual_frombins[regO2_idx,:] > 0])
                        else:
                            regO2_area_annual_acc[regO2_idx,:] += glac_area_annual_acc
                            regO2_area_annual_frombins[regO2_idx,:] += glac_area_annual_frombins
                            regO2_aar_annual[regO2_idx,:][regO2_area_annual_frombins[regO2_idx,:] > 0] = (
                                    regO2_area_annual_acc[regO2_idx,:][regO2_area_annual_frombins[regO2_idx,:] > 0] / 
                                    regO2_area_annual_frombins[regO2_idx,:][regO2_area_annual_frombins[regO2_idx,:] > 0])
                        # Watersheds
                        if watershed_area_annual_acc is None:
                            watershed_area_annual_acc = np.zeros((len(unique_watersheds),years.shape[0]))
                            watershed_area_annual_acc[watershed_idx,:] = glac_area_annual_acc.copy()
                            watershed_area_annual_frombins = np.zeros((len(unique_watersheds),years.shape[0]))
                            watershed_area_annual_frombins[watershed_idx,:] = glac_area_annual_frombins.copy()
                            watershed_aar_annual = np.zeros((len(unique_watersheds),years.shape[0]))
                            watershed_aar_annual[watershed_idx,:][watershed_area_annual_frombins[watershed_idx,:] > 0] = (
                                    watershed_area_annual_acc[watershed_idx,:][watershed_area_annual_frombins[watershed_idx,:] > 0] / 
                                    watershed_area_annual_frombins[watershed_idx,:][watershed_area_annual_frombins[watershed_idx,:] > 0])
                        else:
                            watershed_area_annual_acc[watershed_idx,:] += glac_area_annual_acc
                            watershed_area_annual_frombins[watershed_idx,:] += glac_area_annual_frombins
                            watershed_aar_annual[watershed_idx,:][watershed_area_annual_frombins[watershed_idx,:] > 0] = (
                                    watershed_area_annual_acc[watershed_idx,:][watershed_area_annual_frombins[watershed_idx,:] > 0] / 
                                    watershed_area_annual_frombins[watershed_idx,:][watershed_area_annual_frombins[watershed_idx,:] > 0])
                        
                    # ===== PICKLE DATASETS =====
                    # Volume
                    with open(pickle_fp_reg + fn_reg_vol_annual, 'wb') as f:
                        pickle.dump(reg_vol_annual, f)
                    with open(pickle_fp_regO2 + fn_regO2_vol_annual, 'wb') as f:
                        pickle.dump(regO2_vol_annual, f)
                    with open(pickle_fp_watershed + fn_watershed_vol_annual, 'wb') as f:
                        pickle.dump(watershed_vol_annual, f)
                    with open(pickle_fp_degid + fn_degid_vol_annual, 'wb') as f:
                        pickle.dump(degid_vol_annual, f)
                    # Volume below sea level 
                    with open(pickle_fp_reg + fn_reg_vol_annual_bwl, 'wb') as f:
                        pickle.dump(reg_vol_annual_bwl, f)
                    with open(pickle_fp_regO2 + fn_regO2_vol_annual_bwl, 'wb') as f:
                        pickle.dump(regO2_vol_annual_bwl, f)
                    with open(pickle_fp_watershed + fn_watershed_vol_annual_bwl, 'wb') as f:
                        pickle.dump(watershed_vol_annual_bwl, f)
                    with open(pickle_fp_degid + fn_degid_vol_annual_bwl, 'wb') as f:
                        pickle.dump(degid_vol_annual_bwl, f) 
                    # Volume below debris
                    with open(pickle_fp_reg + fn_reg_vol_annual_bd, 'wb') as f:
                        pickle.dump(reg_vol_annual_bd, f)
                    with open(pickle_fp_regO2 + fn_regO2_vol_annual_bd, 'wb') as f:
                        pickle.dump(regO2_vol_annual_bd, f)
                    with open(pickle_fp_watershed + fn_watershed_vol_annual_bd, 'wb') as f:
                        pickle.dump(watershed_vol_annual_bd, f)
                    with open(pickle_fp_degid + fn_degid_vol_annual_bd, 'wb') as f:
                        pickle.dump(degid_vol_annual_bd, f)
                    # Area 
                    with open(pickle_fp_reg + fn_reg_area_annual, 'wb') as f:
                        pickle.dump(reg_area_annual, f)
                    with open(pickle_fp_regO2 + fn_regO2_area_annual, 'wb') as f:
                        pickle.dump(regO2_area_annual, f)
                    with open(pickle_fp_watershed + fn_watershed_area_annual, 'wb') as f:
                        pickle.dump(watershed_area_annual, f)
                    with open(pickle_fp_degid + fn_degid_area_annual, 'wb') as f:
                        pickle.dump(degid_area_annual, f)
                    # Area below debris
                    with open(pickle_fp_reg + fn_reg_area_annual_bd, 'wb') as f:
                        pickle.dump(reg_area_annual_bd, f)
                    with open(pickle_fp_regO2 + fn_regO2_area_annual_bd, 'wb') as f:
                        pickle.dump(regO2_area_annual_bd, f)
                    with open(pickle_fp_watershed + fn_watershed_area_annual_bd, 'wb') as f:
                        pickle.dump(watershed_area_annual_bd, f)
                    with open(pickle_fp_degid + fn_degid_area_annual_bd, 'wb') as f:
                        pickle.dump(degid_area_annual_bd, f)
                    # Binned Volume
                    with open(pickle_fp_reg + fn_reg_vol_annual_binned, 'wb') as f:
                        pickle.dump(reg_vol_annual_binned, f)
                    with open(pickle_fp_regO2 + fn_regO2_vol_annual_binned, 'wb') as f:
                        pickle.dump(regO2_vol_annual_binned, f)
                    with open(pickle_fp_watershed + fn_watershed_vol_annual_binned, 'wb') as f:
                        pickle.dump(watershed_vol_annual_binned, f)
                    # Binned Volume below debris
                    with open(pickle_fp_reg + fn_reg_vol_annual_binned_bd, 'wb') as f:
                        pickle.dump(reg_vol_annual_binned_bd, f)
                    with open(pickle_fp_regO2 + fn_regO2_vol_annual_binned_bd, 'wb') as f:
                        pickle.dump(regO2_vol_annual_binned_bd, f)
                    with open(pickle_fp_watershed + fn_watershed_vol_annual_binned_bd, 'wb') as f:
                        pickle.dump(watershed_vol_annual_binned_bd, f)
                    # Binned Area
                    with open(pickle_fp_reg + fn_reg_area_annual_binned, 'wb') as f:
                        pickle.dump(reg_area_annual_binned, f)
                    with open(pickle_fp_regO2 + fn_regO2_area_annual_binned, 'wb') as f:
                        pickle.dump(regO2_area_annual_binned, f)
                    with open(pickle_fp_watershed + fn_watershed_area_annual_binned, 'wb') as f:
                        pickle.dump(watershed_area_annual_binned, f)
                    # Binned Area below debris
                    with open(pickle_fp_reg + fn_reg_area_annual_binned_bd, 'wb') as f:
                        pickle.dump(reg_area_annual_binned_bd, f)
                    with open(pickle_fp_regO2 + fn_regO2_area_annual_binned_bd, 'wb') as f:
                        pickle.dump(regO2_area_annual_binned_bd, f)
                    with open(pickle_fp_watershed + fn_watershed_area_annual_binned_bd, 'wb') as f:
                        pickle.dump(watershed_area_annual_binned_bd, f)
                    # Mass balance: accumulation
                    with open(pickle_fp_reg + fn_reg_acc_monthly, 'wb') as f:
                        pickle.dump(reg_acc_monthly, f)
                    with open(pickle_fp_regO2 + fn_regO2_acc_monthly, 'wb') as f:
                        pickle.dump(regO2_acc_monthly, f)
                    with open(pickle_fp_watershed + fn_watershed_acc_monthly, 'wb') as f:
                        pickle.dump(watershed_acc_monthly, f)
                    # Mass balance: refreeze
                    with open(pickle_fp_reg + fn_reg_refreeze_monthly, 'wb') as f:
                        pickle.dump(reg_refreeze_monthly, f)
                    with open(pickle_fp_regO2 + fn_regO2_refreeze_monthly, 'wb') as f:
                        pickle.dump(regO2_refreeze_monthly, f)
                    with open(pickle_fp_watershed + fn_watershed_refreeze_monthly, 'wb') as f:
                        pickle.dump(watershed_refreeze_monthly, f)
                    # Mass balance: melt
                    with open(pickle_fp_reg + fn_reg_melt_monthly, 'wb') as f:
                        pickle.dump(reg_melt_monthly, f)
                    with open(pickle_fp_regO2 + fn_regO2_melt_monthly, 'wb') as f:
                        pickle.dump(regO2_melt_monthly, f)
                    with open(pickle_fp_watershed + fn_watershed_melt_monthly, 'wb') as f:
                        pickle.dump(watershed_melt_monthly, f)
                    # Mass balance: frontal ablation
                    with open(pickle_fp_reg + fn_reg_frontalablation_monthly, 'wb') as f:
                        pickle.dump(reg_frontalablation_monthly, f)
                    with open(pickle_fp_regO2 + fn_regO2_frontalablation_monthly, 'wb') as f:
                        pickle.dump(regO2_frontalablation_monthly, f)
                    with open(pickle_fp_watershed + fn_watershed_frontalablation_monthly, 'wb') as f:
                        pickle.dump(watershed_frontalablation_monthly, f)
                    # Mass balance: total mass balance
                    with open(pickle_fp_reg + fn_reg_massbaltotal_monthly, 'wb') as f:
                        pickle.dump(reg_massbaltotal_monthly, f)
                    with open(pickle_fp_regO2 + fn_regO2_massbaltotal_monthly, 'wb') as f:
                        pickle.dump(regO2_massbaltotal_monthly, f)
                    with open(pickle_fp_watershed + fn_watershed_massbaltotal_monthly, 'wb') as f:
                        pickle.dump(watershed_massbaltotal_monthly, f)
                    with open(pickle_fp_degid + fn_degid_massbaltotal_monthly, 'wb') as f:
                        pickle.dump(degid_massbaltotal_monthly, f)  
                    # Binned Climatic Mass Balance
                    with open(pickle_fp_reg + fn_reg_mbclim_annual_binned, 'wb') as f:
                        pickle.dump(reg_mbclim_annual_binned, f)
                    with open(pickle_fp_regO2 + fn_regO2_mbclim_annual_binned, 'wb') as f:
                        pickle.dump(regO2_mbclim_annual_binned, f)
                    with open(pickle_fp_watershed + fn_watershed_mbclim_annual_binned, 'wb') as f:
                        pickle.dump(watershed_mbclim_annual_binned, f)
                    # Runoff: moving-gauged
                    with open(pickle_fp_reg + fn_reg_runoff_monthly_moving, 'wb') as f:
                        pickle.dump(reg_runoff_monthly_moving, f)
                    with open(pickle_fp_regO2 + fn_regO2_runoff_monthly_moving, 'wb') as f:
                        pickle.dump(regO2_runoff_monthly_moving, f)
                    with open(pickle_fp_watershed + fn_watershed_runoff_monthly_moving, 'wb') as f:
                        pickle.dump(watershed_runoff_monthly_moving, f)
                    with open(pickle_fp_degid + fn_degid_runoff_monthly_moving, 'wb') as f:
                        pickle.dump(degid_runoff_monthly_moving, f)
                    # Runoff: fixed-gauged
                    with open(pickle_fp_reg + fn_reg_runoff_monthly_fixed, 'wb') as f:
                        pickle.dump(reg_runoff_monthly_fixed, f)     
                    with open(pickle_fp_regO2 + fn_regO2_runoff_monthly_fixed, 'wb') as f:
                        pickle.dump(regO2_runoff_monthly_fixed, f)  
                    with open(pickle_fp_watershed + fn_watershed_runoff_monthly_fixed, 'wb') as f:
                        pickle.dump(watershed_runoff_monthly_fixed, f)  
                    with open(pickle_fp_degid + fn_degid_runoff_monthly_fixed, 'wb') as f:
                        pickle.dump(degid_runoff_monthly_fixed, f)  
                    # Runoff: precipitation
                    with open(pickle_fp_reg + fn_reg_prec_monthly, 'wb') as f:
                        pickle.dump(reg_prec_monthly, f)
                    with open(pickle_fp_regO2 + fn_regO2_prec_monthly, 'wb') as f:
                        pickle.dump(regO2_prec_monthly, f)
                    with open(pickle_fp_watershed + fn_watershed_prec_monthly, 'wb') as f:
                        pickle.dump(watershed_prec_monthly, f)
                    # Runoff: off-glacier precipitation
                    with open(pickle_fp_reg + fn_reg_offglac_prec_monthly, 'wb') as f:
                        pickle.dump(reg_offglac_prec_monthly, f)
                    with open(pickle_fp_regO2 + fn_regO2_offglac_prec_monthly, 'wb') as f:
                        pickle.dump(regO2_offglac_prec_monthly, f)
                    with open(pickle_fp_watershed + fn_watershed_offglac_prec_monthly, 'wb') as f:
                        pickle.dump(watershed_offglac_prec_monthly, f)
                    # Runoff: off-glacier melt
                    with open(pickle_fp_reg + fn_reg_offglac_melt_monthly, 'wb') as f:
                        pickle.dump(reg_offglac_melt_monthly, f)
                    with open(pickle_fp_regO2 + fn_regO2_offglac_melt_monthly, 'wb') as f:
                        pickle.dump(regO2_offglac_melt_monthly, f)
                    with open(pickle_fp_watershed + fn_watershed_offglac_melt_monthly, 'wb') as f:
                        pickle.dump(watershed_offglac_melt_monthly, f)
                    # Runoff: off-glacier refreeze
                    with open(pickle_fp_reg + fn_reg_offglac_refreeze_monthly, 'wb') as f:
                        pickle.dump(reg_offglac_refreeze_monthly, f)
                    with open(pickle_fp_regO2 + fn_regO2_offglac_refreeze_monthly, 'wb') as f:
                        pickle.dump(regO2_offglac_refreeze_monthly, f)
                    with open(pickle_fp_watershed + fn_watershed_offglac_refreeze_monthly, 'wb') as f:
                        pickle.dump(watershed_offglac_refreeze_monthly, f)
                    # ELA
                    with open(pickle_fp_reg + fn_reg_ela_annual, 'wb') as f:
                        pickle.dump(reg_ela_annual, f)
                    with open(pickle_fp_regO2 + fn_regO2_ela_annual, 'wb') as f:
                        pickle.dump(regO2_ela_annual, f)
                    with open(pickle_fp_watershed + fn_watershed_ela_annual, 'wb') as f:
                        pickle.dump(watershed_ela_annual, f)
                    # AAR
                    with open(pickle_fp_reg + fn_reg_aar_annual, 'wb') as f:
                        pickle.dump(reg_aar_annual, f)
                    with open(pickle_fp_regO2 + fn_regO2_aar_annual, 'wb') as f:
                        pickle.dump(regO2_aar_annual, f)
                    with open(pickle_fp_watershed + fn_watershed_aar_annual, 'wb') as f:
                        pickle.dump(watershed_aar_annual, f)
                        
                # ----- OTHERWISE LOAD THE PROCESSED DATASETS -----
                else:
                    # Volume
                    with open(pickle_fp_reg + fn_reg_vol_annual, 'rb') as f:
                        reg_vol_annual = pickle.load(f)
                    with open(pickle_fp_regO2 + fn_regO2_vol_annual, 'rb') as f:
                        regO2_vol_annual = pickle.load(f)
                    with open(pickle_fp_watershed + fn_watershed_vol_annual, 'rb') as f:
                        watershed_vol_annual = pickle.load(f)
                    with open(pickle_fp_degid + fn_degid_vol_annual, 'rb') as f:
                        degid_vol_annual = pickle.load(f)
                    # Volume below sea level
                    with open(pickle_fp_reg + fn_reg_vol_annual_bwl, 'rb') as f:
                        reg_vol_annual_bwl = pickle.load(f)
                    with open(pickle_fp_regO2 + fn_regO2_vol_annual_bwl, 'rb') as f:
                        regO2_vol_annual_bwl = pickle.load(f)
                    with open(pickle_fp_watershed + fn_watershed_vol_annual_bwl, 'rb') as f:
                        watershed_vol_annual_bwl = pickle.load(f)
                    with open(pickle_fp_degid + fn_degid_vol_annual_bwl, 'rb') as f:
                        degid_vol_annual_bwl = pickle.load(f)
                    # Volume below debris
                    with open(pickle_fp_reg + fn_reg_vol_annual_bd, 'rb') as f:
                        reg_vol_annual_bd = pickle.load(f)
                    with open(pickle_fp_regO2 + fn_regO2_vol_annual_bd, 'rb') as f:
                        regO2_vol_annual_bd = pickle.load(f)
                    with open(pickle_fp_watershed + fn_watershed_vol_annual_bd, 'rb') as f:
                        watershed_vol_annual_bd = pickle.load(f)
                    with open(pickle_fp_degid + fn_degid_vol_annual_bd, 'rb') as f:
                        degid_vol_annual_bd = pickle.load(f)
                    # Area 
                    with open(pickle_fp_reg + fn_reg_area_annual, 'rb') as f:
                        reg_area_annual = pickle.load(f)
                    with open(pickle_fp_regO2 + fn_regO2_area_annual, 'rb') as f:
                        regO2_area_annual = pickle.load(f)
                    with open(pickle_fp_watershed + fn_watershed_area_annual, 'rb') as f:
                        watershed_area_annual = pickle.load(f)
                    with open(pickle_fp_degid + fn_degid_area_annual, 'rb') as f:
                        degid_area_annual = pickle.load(f)
                    # Area below debris
                    with open(pickle_fp_reg + fn_reg_area_annual_bd, 'rb') as f:
                        reg_area_annual_bd = pickle.load(f)
                    with open(pickle_fp_regO2 + fn_regO2_area_annual_bd, 'rb') as f:
                        regO2_area_annual_bd = pickle.load(f)
                    with open(pickle_fp_watershed + fn_watershed_area_annual_bd, 'rb') as f:
                        watershed_area_annual_bd = pickle.load(f)
                    with open(pickle_fp_degid + fn_degid_area_annual_bd, 'rb') as f:
                        degid_area_annual_bd = pickle.load(f)
                    # Binned Volume
                    with open(pickle_fp_reg + fn_reg_vol_annual_binned, 'rb') as f:
                        reg_vol_annual_binned = pickle.load(f)
                    with open(pickle_fp_regO2 + fn_regO2_vol_annual_binned, 'rb') as f:
                        regO2_vol_annual_binned = pickle.load(f)
                    with open(pickle_fp_watershed + fn_watershed_vol_annual_binned, 'rb') as f:
                        watershed_vol_annual_binned = pickle.load(f)
                    # Binned Volume below debris
                    with open(pickle_fp_reg + fn_reg_vol_annual_binned_bd, 'rb') as f:
                        reg_vol_annual_binned_bd = pickle.load(f)
                    with open(pickle_fp_regO2 + fn_regO2_vol_annual_binned_bd, 'rb') as f:
                        regO2_vol_annual_binned_bd = pickle.load(f)
                    with open(pickle_fp_watershed + fn_watershed_vol_annual_binned_bd, 'rb') as f:
                        watershed_vol_annual_binned_bd = pickle.load(f)
                    # Binned Area
                    with open(pickle_fp_reg + fn_reg_area_annual_binned, 'rb') as f:
                        reg_area_annual_binned = pickle.load(f)
                    with open(pickle_fp_regO2 + fn_regO2_area_annual_binned, 'rb') as f:
                        regO2_area_annual_binned = pickle.load(f)
                    with open(pickle_fp_watershed + fn_watershed_area_annual_binned, 'rb') as f:
                        watershed_area_annual_binned = pickle.load(f)
                    # Binned Area below debris
                    with open(pickle_fp_reg + fn_reg_area_annual_binned_bd, 'rb') as f:
                        reg_area_annual_binned_bd = pickle.load(f)
                    with open(pickle_fp_regO2 + fn_regO2_area_annual_binned_bd, 'rb') as f:
                        regO2_area_annual_binned_bd = pickle.load(f)
                    with open(pickle_fp_watershed + fn_watershed_area_annual_binned_bd, 'rb') as f:
                        watershed_area_annual_binned_bd = pickle.load(f)
                    # Mass balance: accumulation
                    with open(pickle_fp_reg + fn_reg_acc_monthly, 'rb') as f:
                        reg_acc_monthly = pickle.load(f)
                    with open(pickle_fp_regO2 + fn_regO2_acc_monthly, 'rb') as f:
                        regO2_acc_monthly = pickle.load(f)
                    with open(pickle_fp_watershed + fn_watershed_acc_monthly, 'rb') as f:
                        watershed_acc_monthly = pickle.load(f)
                    # Mass balance: refreeze
                    with open(pickle_fp_reg + fn_reg_refreeze_monthly, 'rb') as f:
                        reg_refreeze_monthly = pickle.load(f)
                    with open(pickle_fp_regO2 + fn_regO2_refreeze_monthly, 'rb') as f:
                        regO2_refreeze_monthly = pickle.load(f)
                    with open(pickle_fp_watershed + fn_watershed_refreeze_monthly, 'rb') as f:
                        watershed_refreeze_monthly = pickle.load(f)
                    # Mass balance: melt
                    with open(pickle_fp_reg + fn_reg_melt_monthly, 'rb') as f:
                        reg_melt_monthly = pickle.load(f)
                    with open(pickle_fp_regO2 + fn_regO2_melt_monthly, 'rb') as f:
                        regO2_melt_monthly = pickle.load(f)
                    with open(pickle_fp_watershed + fn_watershed_melt_monthly, 'rb') as f:
                        watershed_melt_monthly = pickle.load(f)
                    # Mass balance: frontal ablation
                    with open(pickle_fp_reg + fn_reg_frontalablation_monthly, 'rb') as f:
                        reg_frontalablation_monthly = pickle.load(f)
                    with open(pickle_fp_regO2 + fn_regO2_frontalablation_monthly, 'rb') as f:
                        regO2_frontalablation_monthly = pickle.load(f)
                    with open(pickle_fp_watershed + fn_watershed_frontalablation_monthly, 'rb') as f:
                        watershed_frontalablation_monthly = pickle.load(f)
                    # Mass balance: total mass balance
                    with open(pickle_fp_reg + fn_reg_massbaltotal_monthly, 'rb') as f:
                        reg_massbaltotal_monthly = pickle.load(f)
                    with open(pickle_fp_regO2 + fn_regO2_massbaltotal_monthly, 'rb') as f:
                        regO2_massbaltotal_monthly = pickle.load(f)
                    with open(pickle_fp_watershed + fn_watershed_massbaltotal_monthly, 'rb') as f:
                        watershed_massbaltotal_monthly = pickle.load(f)
                    with open(pickle_fp_degid + fn_degid_massbaltotal_monthly, 'rb') as f:
                        degid_massbaltotal_monthly = pickle.load(f)
                    # Binned Climatic Mass Balance
                    with open(pickle_fp_reg + fn_reg_mbclim_annual_binned, 'rb') as f:
                        reg_mbclim_annual_binned = pickle.load(f)
                    with open(pickle_fp_regO2 + fn_regO2_mbclim_annual_binned, 'rb') as f:
                        regO2_mbclim_annual_binned = pickle.load(f)
                    with open(pickle_fp_watershed + fn_watershed_mbclim_annual_binned, 'rb') as f:
                        watershed_mbclim_annual_binned = pickle.load(f)
                    # Runoff: moving-gauged
                    with open(pickle_fp_reg + fn_reg_runoff_monthly_moving, 'rb') as f:
                        reg_runoff_monthly_moving = pickle.load(f)
                    with open(pickle_fp_regO2 + fn_regO2_runoff_monthly_moving, 'rb') as f:
                        regO2_runoff_monthly_moving = pickle.load(f)
                    with open(pickle_fp_watershed + fn_watershed_runoff_monthly_moving, 'rb') as f:
                        watershed_runoff_monthly_moving = pickle.load(f)
                    with open(pickle_fp_degid + fn_degid_runoff_monthly_moving, 'rb') as f:
                        degid_runoff_monthly_moving = pickle.load(f)
                    # Runoff: fixed-gauged
                    with open(pickle_fp_reg + fn_reg_runoff_monthly_fixed, 'rb') as f:
                        reg_runoff_monthly_fixed = pickle.load(f)
                    with open(pickle_fp_regO2 + fn_regO2_runoff_monthly_fixed, 'rb') as f:
                        regO2_runoff_monthly_fixed= pickle.load(f)
                    with open(pickle_fp_watershed + fn_watershed_runoff_monthly_fixed, 'rb') as f:
                        watershed_runoff_monthly_fixed = pickle.load(f)
                    with open(pickle_fp_degid + fn_degid_runoff_monthly_fixed, 'rb') as f:
                        degid_runoff_monthly_fixed = pickle.load(f)
                    # Runoff: precipitation
                    with open(pickle_fp_reg + fn_reg_prec_monthly, 'rb') as f:
                        reg_prec_monthly = pickle.load(f)
                    with open(pickle_fp_regO2 + fn_regO2_prec_monthly, 'rb') as f:
                        regO2_prec_monthly = pickle.load(f)
                    with open(pickle_fp_watershed + fn_watershed_prec_monthly, 'rb') as f:
                        watershed_prec_monthly= pickle.load(f)
                    # Runoff: off-glacier precipitation
                    with open(pickle_fp_reg + fn_reg_offglac_prec_monthly, 'rb') as f:
                        reg_offglac_prec_monthly = pickle.load(f)
                    with open(pickle_fp_regO2 + fn_regO2_offglac_prec_monthly, 'rb') as f:
                        regO2_offglac_prec_monthly = pickle.load(f)
                    with open(pickle_fp_watershed + fn_watershed_offglac_prec_monthly, 'rb') as f:
                        watershed_offglac_prec_monthly = pickle.load(f)
                    # Runoff: off-glacier melt
                    with open(pickle_fp_reg + fn_reg_offglac_melt_monthly, 'rb') as f:
                        reg_offglac_melt_monthly = pickle.load(f)
                    with open(pickle_fp_regO2 + fn_regO2_offglac_melt_monthly, 'rb') as f:
                        regO2_offglac_melt_monthly = pickle.load(f)
                    with open(pickle_fp_watershed + fn_watershed_offglac_melt_monthly, 'rb') as f:
                        watershed_offglac_melt_monthly = pickle.load(f)
                    # Runoff: off-glacier refreeze
                    with open(pickle_fp_reg + fn_reg_offglac_refreeze_monthly, 'rb') as f:
                        reg_offglac_refreeze_monthly = pickle.load(f)
                    with open(pickle_fp_regO2 + fn_regO2_offglac_refreeze_monthly, 'rb') as f:
                        regO2_offglac_refreeze_monthly = pickle.load(f)
                    with open(pickle_fp_watershed + fn_watershed_offglac_refreeze_monthly, 'rb') as f:
                        watershed_offglac_refreeze_monthly = pickle.load(f)
                    # ELA
                    with open(pickle_fp_reg + fn_reg_ela_annual, 'rb') as f:
                        reg_ela_annual = pickle.load(f)
                    with open(pickle_fp_regO2 + fn_regO2_ela_annual, 'rb') as f:
                        regO2_ela_annual = pickle.load(f)
                    with open(pickle_fp_watershed + fn_watershed_ela_annual, 'rb') as f:
                        watershed_ela_annual = pickle.load(f)
                    # AAR
                    with open(pickle_fp_reg + fn_reg_aar_annual, 'rb') as f:
                        reg_aar_annual = pickle.load(f)
                    with open(pickle_fp_regO2 + fn_regO2_aar_annual, 'rb') as f:
                        regO2_aar_annual = pickle.load(f)
                    with open(pickle_fp_watershed + fn_watershed_aar_annual, 'rb') as f:
                        watershed_aar_annual = pickle.load(f)
                        
                    # Years
                    if years is None:
                        for nglac, glacno in enumerate(glacno_list[0:1]):
                            # Filenames
                            netcdf_fn_stats_ending = 'MCMC_ba1_50sets_2000_2100_all.nc'
                            netcdf_fn_stats = '_'.join([glacno, gcm_name, rcp, netcdf_fn_stats_ending])
                            ds_stats = xr.open_dataset(netcdf_fp_stats + '/' + netcdf_fn_stats)
                
                            # Years
                            years = ds_stats.year.values
        
                    
                #%%
                if args.option_plot:
                    # ===== REGIONAL PLOTS =====
                    fig_fp_reg = fig_fp + str(reg).zfill(2) + '/'
                    if not os.path.exists(fig_fp_reg):
                        os.makedirs(fig_fp_reg)
                        
                    # ----- FIGURE: DIAGNOSTIC OF EVERYTHING ----- 
                    fig, ax = plt.subplots(3, 4, squeeze=False, sharex=False, sharey=False, 
                                           gridspec_kw = {'wspace':0.7, 'hspace':0.5})
                    label= gcm_name + ' ' + rcp
                    
                    # VOLUME CHANGE
                    ax[0,0].plot(years, reg_vol_annual/1e9, color=rcp_colordict[rcp], linewidth=1, zorder=4, label=None)
                    if not reg_vol_annual_bwl is None:
                        ax[0,0].plot(years, reg_vol_annual_bwl/1e9, color=rcp_colordict[rcp], linewidth=0.5, linestyle='--', zorder=4, label='bwl')
                    if not reg_vol_annual_bd is None:
                        ax[0,0].plot(years, reg_vol_annual_bd/1e9, color=rcp_colordict[rcp], linewidth=0.5, linestyle=':', zorder=4, label='bd')
                    ax[0,0].set_ylabel('Volume (km$^{3}$)')
                    ax[0,0].set_xlim(years.min(), years.max())
                    ax[0,0].xaxis.set_major_locator(MultipleLocator(50))
                    ax[0,0].xaxis.set_minor_locator(MultipleLocator(10))
                    ax[0,0].set_ylim(0,reg_vol_annual.max()*1.05/1e9)
                    ax[0,0].tick_params(direction='inout', right=True)
                    ax[0,0].legend(fontsize=10, labelspacing=0.25, handlelength=1, handletextpad=0.25, borderpad=0, frameon=False
    #                               loc=(0.05,0.05),
                                   )        
                    
    
                    # AREA CHANGE
                    ax[0,1].plot(years, reg_area_annual/1e6, color=rcp_colordict[rcp], linewidth=1, zorder=4, label=None)
                    if not reg_area_annual_bd is None:
                        ax[0,1].plot(years, reg_area_annual_bd/1e6, color=rcp_colordict[rcp], linewidth=0.5, linestyle=':', zorder=4, label='bd')
                    ax[0,1].set_ylabel('Area (km$^{2}$)')
                    ax[0,1].set_xlim(years.min(), years.max())
                    ax[0,1].xaxis.set_major_locator(MultipleLocator(50))
                    ax[0,1].xaxis.set_minor_locator(MultipleLocator(10))
                    ax[0,1].set_ylim(0,reg_area_annual.max()*1.05/1e6)
                    ax[0,1].tick_params(direction='inout', right=True)
                    ax[0,1].legend(fontsize=10, labelspacing=0.25, handlelength=1, handletextpad=0.25, borderpad=0, frameon=False
    #                               loc=(0.05,0.05),
                                   )    
                    
                    
                    # MASS BALANCE
                    reg_mbmwea_annual = ((reg_vol_annual[1:] - reg_vol_annual[:-1]) / reg_area_annual[:-1] * 
                                         pygem_prms.density_ice / pygem_prms.density_water)
                    ax[0,2].plot(years[0:-1], reg_mbmwea_annual, color=rcp_colordict[rcp], linewidth=1, zorder=4, label=None)
                    ax[0,2].set_ylabel('$B$ (m w.e. yr$^{-1}$)')
                    ax[0,2].set_xlim(years.min(), years[0:-1].max())
                    ax[0,2].xaxis.set_major_locator(MultipleLocator(50))
                    ax[0,2].xaxis.set_minor_locator(MultipleLocator(10))
                    ax[0,2].tick_params(direction='inout', right=True)
                    
                    
                    # RUNOFF CHANGE 
                    reg_runoff_annual_fixed = reg_runoff_monthly_fixed.reshape(-1,12).sum(axis=1)
                    reg_runoff_annual_moving = reg_runoff_monthly_moving.reshape(-1,12).sum(axis=1)
                    ax[0,3].plot(years[0:-1], reg_runoff_annual_fixed/1e9, color=rcp_colordict[rcp], linewidth=1, zorder=4, label='Fixed')
                    ax[0,3].plot(years[0:-1], reg_runoff_annual_moving/1e9, color='k', linestyle=':', linewidth=1, zorder=4, label='Moving')
                    ax[0,3].set_ylabel('Runoff (km$^{3}$)')
                    ax[0,3].set_xlim(years.min(), years[0:-1].max())
                    ax[0,3].xaxis.set_major_locator(MultipleLocator(50))
                    ax[0,3].xaxis.set_minor_locator(MultipleLocator(10))
                    ax[0,3].set_ylim(0,reg_runoff_annual_fixed.max()*1.05/1e9)
                    ax[0,3].tick_params(direction='inout', right=True)
                    
                    
    
                    
                    # BINNED VOLUME
                    elev_bin_major = 1000
                    elev_bin_minor = 250
                    ymin = np.floor(elev_bins[np.nonzero(reg_vol_annual_binned.sum(1))[0][0]] / elev_bin_major) * elev_bin_major
                    ymax = np.ceil(elev_bins[np.nonzero(reg_vol_annual_binned.sum(1))[0][-1]] / elev_bin_major) * elev_bin_major
                    ax[1,0].plot(reg_vol_annual_binned[:,0]/1e9, elev_bins[1:], color='k', linewidth=0.5, zorder=4, label=str(years.min()))
                    ax[1,0].plot(reg_vol_annual_binned[:,-1]/1e9, elev_bins[1:], color='b', linewidth=0.5, zorder=4, label=str(years[-1]))
                    if not reg_vol_annual_bd is None:
                        ax[1,0].plot(reg_vol_annual_binned_bd[:,0]/1e9, elev_bins[1:], color='k', linestyle=':', linewidth=0.5, zorder=4, label=None)
                        ax[1,0].plot(reg_vol_annual_binned_bd[:,-1]/1e9, elev_bins[1:], color='b', linestyle=':', linewidth=0.5, zorder=4, label=None)
                    ax[1,0].set_ylabel('Elevation (m)')
                    ax[1,0].set_xlabel('Volume (km$^{3}$)')
                    ax[1,0].set_xlim(0, reg_vol_annual_binned.max()/1e9)
                    ax[1,0].set_ylim(ymin, ymax)
                    ax[1,0].yaxis.set_major_locator(MultipleLocator(elev_bin_major))
                    ax[1,0].yaxis.set_minor_locator(MultipleLocator(elev_bin_minor))
                    ax[1,0].tick_params(direction='inout', right=True)
                    ax[1,0].legend(fontsize=10, labelspacing=0.25, handlelength=1, handletextpad=0.25, borderpad=0, frameon=False
    #                               loc=(0.05,0.05),
                                   ) 
    
                    # BINNED AREA
                    ax[1,1].plot(reg_area_annual_binned[:,0]/1e6, elev_bins[1:], color='k', linewidth=0.5, zorder=4, label=str(years.min()))
                    ax[1,1].plot(reg_area_annual_binned[:,-1]/1e6, elev_bins[1:], color='b', linewidth=0.5, zorder=4, label=str(years[-1]))
                    if not reg_area_annual_binned_bd is None:
                        ax[1,1].plot(reg_area_annual_binned_bd[:,0]/1e6, elev_bins[1:], color='k', linestyle=':', linewidth=0.5, zorder=4, label=None)
                        ax[1,1].plot(reg_area_annual_binned_bd[:,-1]/1e6, elev_bins[1:], color='b', linestyle=':', linewidth=0.5, zorder=4, label=None)
                    ax[1,1].set_ylabel('Elevation (m)')
                    ax[1,1].set_xlabel('Area (km$^{2}$)')
                    ax[1,1].set_xlim(0, reg_area_annual_binned.max()/1e6)
                    ax[1,1].set_ylim(ymin, ymax)
                    ax[1,1].yaxis.set_major_locator(MultipleLocator(elev_bin_major))
                    ax[1,1].yaxis.set_minor_locator(MultipleLocator(elev_bin_minor))
                    ax[1,1].tick_params(direction='inout', right=True)
                    ax[1,1].legend(fontsize=10, labelspacing=0.25, handlelength=1, handletextpad=0.25, borderpad=0, frameon=False
    #                               loc=(0.05,0.05),
                                   ) 
    
                    # CLIMATIC MASS BALANCE GRADIENT
                    reg_mbclim_annual_binned_mwea = reg_mbclim_annual_binned / reg_area_annual_binned
                    ax[1,2].plot(reg_mbclim_annual_binned_mwea[:,0], elev_bins[1:], color='k', linewidth=0.5, zorder=4, label=str(years.min()))
                    ax[1,2].plot(reg_mbclim_annual_binned_mwea[:,-2], elev_bins[1:], color='b', linewidth=0.5, zorder=4, label=str(years.min()))
                    ax[1,2].set_ylabel('Elevation (m)')
                    ax[1,2].set_xlabel('$b_{clim}$ (m w.e. yr$^{-1}$)')
                    ax[1,2].set_ylim(ymin, ymax)
                    ax[1,2].yaxis.set_major_locator(MultipleLocator(elev_bin_major))
                    ax[1,2].yaxis.set_minor_locator(MultipleLocator(elev_bin_minor))
                    ax[1,2].tick_params(direction='inout', right=True)           
                    ax[1,2].axvline(0, color='k', linewidth=0.25)
                    
                    
                    # RUNOFF COMPONENTS
    #                reg_offglac_melt_annual = reg_offglac_melt_monthly.reshape(-1,12).sum(axis=1)
    #                reg_runoff_annual_moving = reg_runoff_monthly_moving.reshape(-1,12).sum(axis=1)
    #                ax[0,3].plot(years[0:-1], reg_runoff_annual_fixed/1e9, color=rcp_colordict[rcp], linewidth=1, zorder=4, label='Fixed')
    #                ax[0,3].plot(years[0:-1], reg_runoff_annual_moving/1e9, color='k', linestyle=':', linewidth=1, zorder=4, label='Moving')
    #                ax[0,3].set_ylabel('Runoff (km$^{3}$)')
    #                ax[0,3].set_xlim(years.min(), years[0:-1].max())
    #                ax[0,3].xaxis.set_major_locator(MultipleLocator(50))
    #                ax[0,3].xaxis.set_minor_locator(MultipleLocator(10))
    #                ax[0,3].set_ylim(0,reg_runoff_annual_fixed.max()*1.05/1e9)
    #                ax[0,3].tick_params(direction='inout', right=True)
                    
                    
                    # ELA
                    ela_min = np.floor(np.min(reg_ela_annual[0:-1]) / 100) * 100
                    ela_max = np.ceil(np.max(reg_ela_annual[0:-1]) / 100) * 100
                    ax[2,0].plot(years[0:-1], reg_ela_annual[0:-1], color='k', linewidth=0.5, zorder=4, label=str(years.min()))
                    ax[2,0].set_ylabel('ELA (m)')
                    ax[2,0].set_xlim(years.min(), years[0:-1].max())
    #                ax[2,0].set_ylim(ela_min, ela_max)
                    ax[2,0].tick_params(direction='inout', right=True)
                    
                    
                    # AAR
                    ax[2,1].plot(years, reg_aar_annual, color='k', linewidth=0.5, zorder=4, label=str(years.min()))
                    ax[2,1].set_ylabel('AAR (-)')
                    ax[2,1].set_ylim(0,1)
                    ax[2,1].set_xlim(years.min(), years[0:-1].max())
                    ax[2,1].tick_params(direction='inout', right=True)
                    
                    
                    # MASS BALANCE COMPONENTS
                    # - these are only meant for monthly and/or relative purposes 
                    #   mass balance from volume change should be used for annual changes
                    reg_acc_annual = reg_acc_monthly.reshape(-1,12).sum(axis=1)
                    # Refreeze
                    reg_refreeze_annual = reg_refreeze_monthly.reshape(-1,12).sum(axis=1)
                    # Melt
                    reg_melt_annual = reg_melt_monthly.reshape(-1,12).sum(axis=1)
                    # Frontal Ablation
                    reg_frontalablation_annual = reg_frontalablation_monthly.reshape(-1,12).sum(axis=1)
                    # Periods
                    if reg_acc_annual.shape[0] == 101:
                        period_yrs = 20
                        periods = (np.arange(years.min(), years[0:100].max(), period_yrs) + period_yrs/2).astype(int)
                        reg_acc_periods = reg_acc_annual[0:100].reshape(-1,period_yrs).sum(1)
                        reg_refreeze_periods = reg_refreeze_annual[0:100].reshape(-1,period_yrs).sum(1)
                        reg_melt_periods = reg_melt_annual[0:100].reshape(-1,period_yrs).sum(1)
                        reg_frontalablation_periods = reg_frontalablation_annual[0:100].reshape(-1,period_yrs).sum(1)
                        reg_massbaltotal_periods = reg_acc_periods + reg_refreeze_periods - reg_melt_periods - reg_frontalablation_periods
                        
                        # Convert to mwea
                        reg_area_periods = reg_area_annual[0:100].reshape(-1,period_yrs).mean(1)
                        reg_acc_periods_mwea = reg_acc_periods / reg_area_periods / period_yrs
                        reg_refreeze_periods_mwea = reg_refreeze_periods / reg_area_periods / period_yrs
                        reg_melt_periods_mwea = reg_melt_periods / reg_area_periods / period_yrs
                        reg_frontalablation_periods_mwea = reg_frontalablation_periods / reg_area_periods / period_yrs
                        reg_massbaltotal_periods_mwea = reg_massbaltotal_periods / reg_area_periods / period_yrs
                    else:
                        assert True==False, 'Set up for different time periods'
    
                    # Plot
                    ax[2,2].bar(periods, reg_acc_periods_mwea + reg_refreeze_periods_mwea, color='#3553A5', width=period_yrs/2-1, label='refreeze', zorder=2)
                    ax[2,2].bar(periods, reg_acc_periods_mwea, color='#3478BD', width=period_yrs/2-1, label='acc', zorder=3)
                    if not reg_frontalablation_periods_mwea.sum() == 0:
                        ax[2,2].bar(periods, -reg_frontalablation_periods_mwea, color='#83439A', width=period_yrs/2-1, label='frontal ablation', zorder=3)
                    ax[2,2].bar(periods, -reg_melt_periods_mwea - reg_frontalablation_periods_mwea, color='#F47A20', width=period_yrs/2-1, label='melt', zorder=2)
                    ax[2,2].bar(periods, reg_massbaltotal_periods_mwea, color='#555654', width=period_yrs-2, label='total', zorder=1)
                    ax[2,2].set_ylabel('$B$ (m w.e. yr$^{-1}$)')
                    ax[2,2].set_xlim(years.min(), years[0:-1].max())
                    ax[2,2].xaxis.set_major_locator(MultipleLocator(100))
                    ax[2,2].xaxis.set_minor_locator(MultipleLocator(20))
                    ax[2,2].yaxis.set_major_locator(MultipleLocator(1))
                    ax[2,2].yaxis.set_minor_locator(MultipleLocator(0.25))
                    ax[2,2].tick_params(direction='inout', right=True)
                    ax[2,2].legend(fontsize=10, labelspacing=0.25, handlelength=1, handletextpad=0.25, borderpad=0, frameon=False,
                                   loc=(1.2,0.25)) 
                    
                    
                    # Remove plot in lower right
                    fig.delaxes(ax[2,3])
                    
                    
                    # Title
                    fig.text(0.5, 0.95, rgi_reg_dict[reg] + ' (' + gcm_name + ' ' + rcp + ')', size=12, ha='center', va='top',)
                    
                    # Save figure
                    fig_fn = str(reg) + '_allplots_' + str(years.min()) + '-' + str(years.max()) + '_' + gcm_name + '_' + rcp + '.png'
                    fig.set_size_inches(8,6)
                    fig.savefig(fig_fp_reg + fig_fn, bbox_inches='tight', dpi=300)


if option_process_data_nodebris:

    overwrite_pickle = False
    
    grouping = 'all'

    fig_fp = netcdf_fp_cmip5 + '/../analysis-nodebris/figures/'
    if not os.path.exists(fig_fp):
        os.makedirs(fig_fp, exist_ok=True)
    csv_fp = netcdf_fp_cmip5 + '/../analysis-nodebris/csv/'
    if not os.path.exists(csv_fp):
        os.makedirs(csv_fp, exist_ok=True)
    pickle_fp = fig_fp + '../pickle-nodebris/'
    if not os.path.exists(pickle_fp):
        os.makedirs(pickle_fp, exist_ok=True)
        
#    def mwea_to_gta(mwea, area):
#        return mwea * pygem_prms.density_water * area / 1e12
    
    #%%
    for reg in regions:
        # Load glaciers
        glacno_list = []
        for rcp in rcps:
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
        
        # ----- Add Groups -----
        # Degrees (based on degree_size)
        main_glac_rgi['CenLon_round'] = np.floor(main_glac_rgi.CenLon.values/degree_size) * degree_size
        main_glac_rgi['CenLat_round'] = np.floor(main_glac_rgi.CenLat.values/degree_size) * degree_size
        deg_groups = main_glac_rgi.groupby(['CenLon_round', 'CenLat_round']).size().index.values.tolist()
        deg_dict = dict(zip(deg_groups, np.arange(0,len(deg_groups))))
        main_glac_rgi.reset_index(drop=True, inplace=True)
        cenlon_cenlat = [(main_glac_rgi.loc[x,'CenLon_round'], main_glac_rgi.loc[x,'CenLat_round']) 
                         for x in range(len(main_glac_rgi))]
        main_glac_rgi['CenLon_CenLat'] = cenlon_cenlat
        main_glac_rgi['deg_id'] = main_glac_rgi.CenLon_CenLat.map(deg_dict)
        
        # River Basin
        watershed_dict_fn = pygem_prms.main_directory + '/../qgis_datasets/rgi60_watershed_dict.csv'
        watershed_csv = pd.read_csv(watershed_dict_fn)
        watershed_dict = dict(zip(watershed_csv.RGIId, watershed_csv.watershed))
        main_glac_rgi['watershed'] = main_glac_rgi.RGIId.map(watershed_dict)
        if len(np.where(main_glac_rgi.watershed.isnull())[0]) > 0:
            main_glac_rgi.loc[np.where(main_glac_rgi.watershed.isnull())[0],'watershed'] = 'nan'
        
        #%%
        # Unique Groups
        # O2 Regions
        unique_regO2s = np.unique(main_glac_rgi['O2Region'])
        
        # Degrees
        if main_glac_rgi['deg_id'].isnull().all():
            unique_degids = None
        else:
            unique_degids = np.unique(main_glac_rgi['deg_id'])
        
        # Watersheds
        if main_glac_rgi['watershed'].isnull().all():
            unique_watersheds = None
        else:
            unique_watersheds = np.unique(main_glac_rgi['watershed'])

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
        # Watershed dict
        fn_unique_watersheds = 'R' + str(reg) + '_unique_watersheds.pkl'
        with open(pickle_fp + str(reg).zfill(2) + '/' + fn_unique_watersheds, 'wb') as f:
            pickle.dump(unique_watersheds, f) 
        # Degree ID dict
        fn_unique_degids = 'R' + str(reg) + '_unique_degids.pkl'
        with open(pickle_fp + str(reg).zfill(2) + '/' + fn_unique_degids, 'wb') as f:
            pickle.dump(unique_degids, f)
        
        fn_elev_bins = 'R' + str(reg) + '_elev_bins.pkl'
        with open(pickle_fp + str(reg).zfill(2) + '/' + fn_elev_bins, 'wb') as f:
            pickle.dump(elev_bins, f)
        
        #%%
        years = None        
        for gcm_name in gcm_names:
            for rcp in rcps:

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
                pickle_fp_watershed =  pickle_fp + str(reg).zfill(2) + '/watersheds/' + gcm_name + '/' + rcp + '/'
                if not os.path.exists(pickle_fp_watershed):
                    os.makedirs(pickle_fp_watershed)
                pickle_fp_degid =  pickle_fp + str(reg).zfill(2) + '/degids/' + gcm_name + '/' + rcp + '/'
                if not os.path.exists(pickle_fp_degid):
                    os.makedirs(pickle_fp_degid)
                # Region string prefix
                reg_rcp_gcm_str = 'R' + str(reg) + '_' + rcp + '_' + gcm_name
                regO2_rcp_gcm_str = 'R' + str(reg) + '_O2Regions_' + rcp + '_' + gcm_name
                watershed_rcp_gcm_str = 'R' + str(reg) + '_watersheds_' + rcp + '_' + gcm_name
                degid_rcp_gcm_str = 'R' + str(reg) + '_degids_' + rcp + '_' + gcm_name
                
                # Volume
                fn_reg_vol_annual = reg_rcp_gcm_str + '_vol_annual.pkl'
                fn_regO2_vol_annual = regO2_rcp_gcm_str + '_vol_annual.pkl'
                fn_watershed_vol_annual = watershed_rcp_gcm_str + '_vol_annual.pkl'
                fn_degid_vol_annual = degid_rcp_gcm_str + '_vol_annual.pkl'
                # Volume below sea level 
                fn_reg_vol_annual_bwl = reg_rcp_gcm_str + '_vol_annual_bwl.pkl'
                fn_regO2_vol_annual_bwl = regO2_rcp_gcm_str + '_vol_annual_bwl.pkl'
                fn_watershed_vol_annual_bwl = watershed_rcp_gcm_str + '_vol_annual_bwl.pkl'
                fn_degid_vol_annual_bwl = degid_rcp_gcm_str + '_vol_annual_bwl.pkl'
                # Volume below debris
                fn_reg_vol_annual_bd = reg_rcp_gcm_str + '_vol_annual_bd.pkl'
                fn_regO2_vol_annual_bd = regO2_rcp_gcm_str + '_vol_annual_bd.pkl'
                fn_watershed_vol_annual_bd = watershed_rcp_gcm_str + '_vol_annual_bd.pkl'
                fn_degid_vol_annual_bd = degid_rcp_gcm_str + '_vol_annual_bd.pkl'
                # Area 
                fn_reg_area_annual = reg_rcp_gcm_str + '_area_annual.pkl'
                fn_regO2_area_annual = regO2_rcp_gcm_str + '_area_annual.pkl'
                fn_watershed_area_annual = watershed_rcp_gcm_str + '_area_annual.pkl'
                fn_degid_area_annual = degid_rcp_gcm_str + '_area_annual.pkl'
                # Area below debris
                fn_reg_area_annual_bd = reg_rcp_gcm_str + '_area_annual_bd.pkl'
                fn_regO2_area_annual_bd = regO2_rcp_gcm_str + '_area_annual_bd.pkl'
                fn_watershed_area_annual_bd = watershed_rcp_gcm_str + '_area_annual_bd.pkl'
                fn_degid_area_annual_bd = degid_rcp_gcm_str + '_area_annual_bd.pkl'
                # Binned Volume
                fn_reg_vol_annual_binned = reg_rcp_gcm_str + '_vol_annual_binned.pkl'
                fn_regO2_vol_annual_binned = regO2_rcp_gcm_str + '_vol_annual_binned.pkl'
                fn_watershed_vol_annual_binned = watershed_rcp_gcm_str + '_vol_annual_binned.pkl'
                # Binned Volume below debris
                fn_reg_vol_annual_binned_bd = reg_rcp_gcm_str + '_vol_annual_binned_bd.pkl'
                fn_regO2_vol_annual_binned_bd = regO2_rcp_gcm_str + '_vol_annual_binned_bd.pkl'
                fn_watershed_vol_annual_binned_bd = watershed_rcp_gcm_str + '_vol_annual_binned_bd.pkl'
                # Binned Area
                fn_reg_area_annual_binned = reg_rcp_gcm_str + '_area_annual_binned.pkl'
                fn_regO2_area_annual_binned = regO2_rcp_gcm_str + '_area_annual_binned.pkl'
                fn_watershed_area_annual_binned = watershed_rcp_gcm_str + '_area_annual_binned.pkl'
                # Binned Area below debris
                fn_reg_area_annual_binned_bd = reg_rcp_gcm_str + '_area_annual_binned_bd.pkl'
                fn_regO2_area_annual_binned_bd = regO2_rcp_gcm_str + '_area_annual_binned_bd.pkl'
                fn_watershed_area_annual_binned_bd = watershed_rcp_gcm_str + '_area_annual_binned_bd.pkl'
                # Binned Climatic Mass Balance
                fn_reg_mbclim_annual_binned = reg_rcp_gcm_str + '_mbclim_annual_binned.pkl'
                fn_regO2_mbclim_annual_binned = regO2_rcp_gcm_str + '_mbclim_annual_binned.pkl'
                fn_watershed_mbclim_annual_binned = watershed_rcp_gcm_str + '_mbclim_annual_binned.pkl'
                # Runoff: moving-gauged
                fn_reg_runoff_monthly_moving = reg_rcp_gcm_str + '_runoff_monthly_moving.pkl'
                fn_regO2_runoff_monthly_moving = regO2_rcp_gcm_str + '_runoff_monthly_moving.pkl'
                fn_watershed_runoff_monthly_moving = watershed_rcp_gcm_str + '_runoff_monthly_moving.pkl'
                fn_degid_runoff_monthly_moving = degid_rcp_gcm_str + '_runoff_monthly_moving.pkl'
                # Runoff: fixed-gauged
                fn_reg_runoff_monthly_fixed = reg_rcp_gcm_str + '_runoff_monthly_fixed.pkl'
                fn_regO2_runoff_monthly_fixed = regO2_rcp_gcm_str + '_runoff_monthly_fixed.pkl'
                fn_watershed_runoff_monthly_fixed = watershed_rcp_gcm_str + '_runoff_monthly_fixed.pkl'
                fn_degid_runoff_monthly_fixed = degid_rcp_gcm_str + '_runoff_monthly_fixed.pkl'
                # ELA
                fn_reg_ela_annual = reg_rcp_gcm_str + '_ela_annual.pkl'
                fn_regO2_ela_annual = regO2_rcp_gcm_str + '_ela_annual.pkl'
                fn_watershed_ela_annual = watershed_rcp_gcm_str + '_ela_annual.pkl'
                # AAR
                fn_reg_aar_annual = reg_rcp_gcm_str + '_aar_annual.pkl'
                fn_regO2_aar_annual = regO2_rcp_gcm_str + '_aar_annual.pkl'
                fn_watershed_aar_annual = watershed_rcp_gcm_str + '_aar_annual.pkl'
                    
                if not os.path.exists(pickle_fp_reg + fn_reg_vol_annual) or overwrite_pickle:

                    # Entire region
                    years = None
                    reg_vol_annual = None
                    reg_vol_annual_bwl = None
                    reg_vol_annual_bd = None
                    reg_area_annual = None
                    reg_area_annual_bd = None
                    reg_vol_annual_binned = None
                    reg_vol_annual_binned_bd = None
                    reg_area_annual_binned = None
                    reg_area_annual_binned_bd = None
                    reg_mbclim_annual_binned = None
                    reg_runoff_monthly_fixed = None
                    reg_runoff_monthly_moving = None
                    reg_ela_annual = None
                    reg_ela_annual_area = None # used for weighted area calculations
                    reg_area_annual_acc = None
                    reg_area_annual_frombins = None
                    
                    # Subregion groups
                    regO2_vol_annual = None
                    regO2_vol_annual_bwl = None
                    regO2_vol_annual_bd = None
                    regO2_area_annual = None
                    regO2_area_annual_bd = None
                    regO2_vol_annual_binned = None
                    regO2_vol_annual_binned_bd = None
                    regO2_area_annual_binned = None
                    regO2_area_annual_binned_bd = None
                    regO2_mbclim_annual_binned = None
                    regO2_runoff_monthly_fixed = None
                    regO2_runoff_monthly_moving = None
                    regO2_ela_annual = None
                    regO2_ela_annual_area = None # used for weighted area calculations
                    regO2_area_annual_acc = None
                    regO2_area_annual_frombins = None
                    
                    # Watershed groups
                    watershed_vol_annual = None
                    watershed_vol_annual_bwl = None
                    watershed_vol_annual_bd = None
                    watershed_area_annual = None
                    watershed_area_annual_bd = None
                    watershed_vol_annual_binned = None
                    watershed_vol_annual_binned_bd = None
                    watershed_area_annual_binned = None
                    watershed_area_annual_binned_bd = None
                    watershed_mbclim_annual_binned = None
                    watershed_runoff_monthly_fixed = None
                    watershed_runoff_monthly_moving = None
                    watershed_ela_annual = None
                    watershed_ela_annual_area = None # used for weighted area calculations
                    watershed_area_annual_acc = None
                    watershed_area_annual_frombins = None
    
                    # Degree groups                
                    degid_vol_annual = None
                    degid_vol_annual_bwl = None
                    degid_vol_annual_bd = None
                    degid_area_annual = None
                    degid_area_annual_bd = None
                    degid_runoff_monthly_fixed = None
                    degid_runoff_monthly_moving = None
    
    
                    for nglac, glacno in enumerate(glacno_list):
                        if nglac%10 == 0:
                            print(gcm_name, rcp, glacno)
                        
                        # Group indices
                        glac_idx = np.where(main_glac_rgi['glacno'] == glacno)[0][0]
                        regO2 = main_glac_rgi.loc[glac_idx, 'O2Region']
                        regO2_idx = np.where(regO2 == unique_regO2s)[0][0]
                        watershed = main_glac_rgi.loc[glac_idx,'watershed']
                        watershed_idx = np.where(watershed == unique_watersheds)
                        degid = main_glac_rgi.loc[glac_idx, 'deg_id']
                        degid_idx = np.where(degid == unique_degids)[0][0]
                        
                        # Filenames
                        nsim_strs = ['50', '1', '100', '150', '200', '250']
                        ds_binned = None
                        nset = -1
                        while ds_binned is None and nset <= len(nsim_strs):
                            nset += 1
                            nsim_str = nsim_strs[nset]
                            
                            try:
                                netcdf_fn_binned_ending = 'MCMC_ba1_' + nsim_str + 'sets_2000_2100_binned.nc'
                                netcdf_fn_binned = '_'.join([glacno, gcm_name, rcp, netcdf_fn_binned_ending])
        
                                netcdf_fn_stats_ending = 'MCMC_ba1_' + nsim_str + 'sets_2000_2100_all.nc'
                                netcdf_fn_stats = '_'.join([glacno, gcm_name, rcp, netcdf_fn_stats_ending])
                                
                                # Open files
                                ds_binned = xr.open_dataset(netcdf_fp_binned + '/' + netcdf_fn_binned)
                                ds_stats = xr.open_dataset(netcdf_fp_stats + '/' + netcdf_fn_stats)
                            except:
                                ds_binned = None
                            
                        # Years
                        if years is None:
                            years = ds_stats.year.values
                            
                                
                        # ----- 1. Volume (m3) vs. Year ----- 
                        glac_vol_annual = ds_stats.glac_volume_annual.values[0,:]
                        # All
                        if reg_vol_annual is None:
                            reg_vol_annual = glac_vol_annual
                        else:
                            reg_vol_annual += glac_vol_annual
                        # O2Region
                        if regO2_vol_annual is None:
                            regO2_vol_annual = np.zeros((len(unique_regO2s),years.shape[0]))
                            regO2_vol_annual[regO2_idx,:] = glac_vol_annual
                        else:
                            regO2_vol_annual[regO2_idx,:] += glac_vol_annual
                        # Watershed
                        if watershed_vol_annual is None:
                            watershed_vol_annual = np.zeros((len(unique_watersheds),years.shape[0]))
                            watershed_vol_annual[watershed_idx,:] = glac_vol_annual
                        else:
                            watershed_vol_annual[watershed_idx,:] += glac_vol_annual
                        # DegId
                        if degid_vol_annual is None:
                            degid_vol_annual = np.zeros((len(unique_degids), years.shape[0]))
                            degid_vol_annual[degid_idx,:] = glac_vol_annual
                        else:
                            degid_vol_annual[degid_idx,:] += glac_vol_annual
                            
                            
                        # ----- 2. Volume below-sea-level (m3) vs. Year ----- 
                        #  - initial elevation is stored
                        #  - bed elevation is constant in time
                        #  - assume sea level is at 0 m a.s.l.
                        z_sealevel = 0
                        bin_z_init = ds_binned.bin_surface_h_initial.values[0,:]
                        bin_thick_annual = ds_binned.bin_thick_annual.values[0,:,:]
                        bin_z_bed = bin_z_init - bin_thick_annual[:,0]
                        # Annual surface height
                        bin_z_surf_annual = bin_z_bed[:,np.newaxis] + bin_thick_annual
                        
                        # Annual volume (m3)
                        bin_vol_annual = ds_binned.bin_volume_annual.values[0,:,:]
                        # Annual area (m2)
                        bin_area_annual = np.zeros(bin_vol_annual.shape)
                        bin_area_annual[bin_vol_annual > 0] = (
                                bin_vol_annual[bin_vol_annual > 0] / bin_thick_annual[bin_vol_annual > 0])
                        
                        # Processed based on OGGM's _vol_below_level function
                        bwl = (bin_z_bed[:,np.newaxis] < 0) & (bin_thick_annual > 0)
                        if bwl.any():
                            # Annual surface height (max of sea level for calcs)
                            bin_z_surf_annual_bwl = bin_z_surf_annual.copy()
                            bin_z_surf_annual_bwl[bin_z_surf_annual_bwl > z_sealevel] = z_sealevel
                            # Annual thickness below sea level (m)
                            bin_thick_annual_bwl = bin_thick_annual.copy()
                            bin_thick_annual_bwl = bin_z_surf_annual_bwl - bin_z_bed[:,np.newaxis]
                            bin_thick_annual_bwl[~bwl] = 0
                            # Annual volume below sea level (m3)
                            bin_vol_annual_bwl = np.zeros(bin_vol_annual.shape)
                            bin_vol_annual_bwl[bwl] = bin_thick_annual_bwl[bwl] * bin_area_annual[bwl]
                            glac_vol_annual_bwl = bin_vol_annual_bwl.sum(0)
                            
                            # All
                            if reg_vol_annual_bwl is None:
                                reg_vol_annual_bwl = glac_vol_annual_bwl
                            else:
                                reg_vol_annual_bwl += glac_vol_annual_bwl
                            # O2Region
                            if regO2_vol_annual_bwl is None:
                                regO2_vol_annual_bwl = np.zeros((len(unique_regO2s),years.shape[0]))
                                regO2_vol_annual_bwl[regO2_idx,:] = glac_vol_annual_bwl
                            else:
                                regO2_vol_annual_bwl[regO2_idx,:] += glac_vol_annual_bwl
                            # Watershed
                            if watershed_vol_annual_bwl is None:
                                watershed_vol_annual_bwl = np.zeros((len(unique_watersheds),years.shape[0]))
                                watershed_vol_annual_bwl[watershed_idx,:] = glac_vol_annual_bwl
                            else:
                                watershed_vol_annual_bwl[watershed_idx,:] += glac_vol_annual_bwl
                            # DegId
                            if degid_vol_annual_bwl is None:
                                degid_vol_annual_bwl = np.zeros((len(unique_degids), years.shape[0]))
                                degid_vol_annual_bwl[degid_idx,:] = glac_vol_annual_bwl
                            else:
                                degid_vol_annual_bwl[degid_idx,:] += glac_vol_annual_bwl
                        
    
                        # ----- 3. Volume below-debris vs. Time ----- 
                        gdir = single_flowline_glacier_directory(glacno, logging_level='CRITICAL')
                        fls = gdir.read_pickle('inversion_flowlines')
                        bin_debris_hd = np.zeros(bin_z_init.shape)
                        bin_debris_ed = np.zeros(bin_z_init.shape) + 1
                        if 'debris_hd' in dir(fls[0]):
                            bin_debris_hd[0:fls[0].debris_hd.shape[0]] = fls[0].debris_hd
                            bin_debris_ed[0:fls[0].debris_hd.shape[0]] = fls[0].debris_ed
                        if bin_debris_hd.sum() > 0:
                            bin_vol_annual_bd = np.zeros(bin_vol_annual.shape)
                            bin_vol_annual_bd[bin_debris_hd > 0, :] = bin_vol_annual[bin_debris_hd > 0, :]
                            glac_vol_annual_bd = bin_vol_annual_bd.sum(0)
                            
                            # All
                            if reg_vol_annual_bd is None:
                                reg_vol_annual_bd = glac_vol_annual_bd
                            else:
                                reg_vol_annual_bd += glac_vol_annual_bd
                            # O2Region
                            if regO2_vol_annual_bd is None:
                                regO2_vol_annual_bd = np.zeros((len(unique_regO2s),years.shape[0]))
                                regO2_vol_annual_bd[regO2_idx,:] = glac_vol_annual_bd
                            else:
                                regO2_vol_annual_bd[regO2_idx,:] += glac_vol_annual_bd
                            # Watershed
                            if watershed_vol_annual_bd is None:
                                watershed_vol_annual_bd = np.zeros((len(unique_watersheds),years.shape[0]))
                                watershed_vol_annual_bd[watershed_idx,:] = glac_vol_annual_bd
                            else:
                                watershed_vol_annual_bd[watershed_idx,:] += glac_vol_annual_bd
                            # DegId
                            if degid_vol_annual_bd is None:
                                degid_vol_annual_bd = np.zeros((len(unique_degids), years.shape[0]))
                                degid_vol_annual_bd[degid_idx,:] = glac_vol_annual_bd
                            else:
                                degid_vol_annual_bd[degid_idx,:] += glac_vol_annual_bd
                        
                        
                        # ----- 4. Area vs. Time ----- 
                        glac_area_annual = ds_stats.glac_area_annual.values[0,:]
                        # All
                        if reg_area_annual is None:
                            reg_area_annual = glac_area_annual
                        else:
                            reg_area_annual += glac_area_annual
                        # O2Region
                        if regO2_area_annual is None:
                            regO2_area_annual = np.zeros((len(unique_regO2s),years.shape[0]))
                            regO2_area_annual[regO2_idx,:] = glac_area_annual
                        else:
                            regO2_area_annual[regO2_idx,:] += glac_area_annual
                        # Watershed
                        if watershed_area_annual is None:
                            watershed_area_annual = np.zeros((len(unique_watersheds),years.shape[0]))
                            watershed_area_annual[watershed_idx,:] = glac_area_annual
                        else:
                            watershed_area_annual[watershed_idx,:] += glac_area_annual
                        # DegId
                        if degid_area_annual is None:
                            degid_area_annual = np.zeros((len(unique_degids), years.shape[0]))
                            degid_area_annual[degid_idx,:] = glac_area_annual
                        else:
                            degid_area_annual[degid_idx,:] += glac_area_annual
                        
                        
                        # ----- 5. Area below-debris vs. Time ----- 
                        if bin_debris_hd.sum() > 0:
                            bin_area_annual_bd = np.zeros(bin_area_annual.shape)
                            bin_area_annual_bd[bin_debris_hd > 0, :] = bin_area_annual[bin_debris_hd > 0, :]
                            glac_area_annual_bd = bin_area_annual_bd.sum(0)
                            
                            # All
                            if reg_area_annual_bd is None:
                                reg_area_annual_bd = glac_area_annual_bd
                            else:
                                reg_area_annual_bd += glac_area_annual_bd
                            # O2Region
                            if regO2_area_annual_bd is None:
                                regO2_area_annual_bd = np.zeros((len(unique_regO2s),years.shape[0]))
                                regO2_area_annual_bd[regO2_idx,:] = glac_area_annual_bd
                            else:
                                regO2_area_annual_bd[regO2_idx,:] += glac_area_annual_bd
                            # Watershed
                            if watershed_area_annual_bd is None:
                                watershed_area_annual_bd = np.zeros((len(unique_watersheds),years.shape[0]))
                                watershed_area_annual_bd[watershed_idx,:] = glac_area_annual_bd
                            else:
                                watershed_area_annual_bd[watershed_idx,:] += glac_area_annual_bd
                            # DegId
                            if degid_area_annual_bd is None:
                                degid_area_annual_bd = np.zeros((len(unique_degids), years.shape[0]))
                                degid_area_annual_bd[degid_idx,:] = glac_area_annual_bd
                            else:
                                degid_area_annual_bd[degid_idx,:] += glac_area_annual_bd
                        
                        
                        # ----- 6. Binned glacier volume vs. Time ----- 
                        bin_vol_annual_10m = np.zeros((len(elev_bins)-1, len(years)))
                        for ncol, year in enumerate(years):
                            bin_counts, bin_edges = np.histogram(bin_z_surf_annual[:,ncol], bins=elev_bins, 
                                                                 weights=bin_vol_annual[:,ncol])
                            bin_vol_annual_10m[:,ncol] = bin_counts
                        
                        # All
                        if reg_vol_annual_binned is None:
                            reg_vol_annual_binned = bin_vol_annual_10m
                        else:
                            reg_vol_annual_binned += bin_vol_annual_10m
                        # O2Region
                        if regO2_vol_annual_binned is None:
                            regO2_vol_annual_binned = np.zeros((len(unique_regO2s), bin_vol_annual_10m.shape[0], years.shape[0]))
                            regO2_vol_annual_binned[regO2_idx,:,:] = bin_vol_annual_10m
                        else:
                            regO2_vol_annual_binned[regO2_idx,:,:] += bin_vol_annual_10m
                        # Watershed
                        if watershed_vol_annual_binned is None:
                            watershed_vol_annual_binned = np.zeros((len(unique_watersheds), bin_vol_annual_10m.shape[0], years.shape[0]))
                            watershed_vol_annual_binned[watershed_idx,:,:] = bin_vol_annual_10m
                        else:
                            watershed_vol_annual_binned[watershed_idx,:,:] += bin_vol_annual_10m
                        
    
                        # ----- 7. Binned glacier volume below debris vs. Time ----- 
                        if bin_debris_hd.sum() > 0:
                            # Bin debris mask for the given elevation bins
                            bin_debris_mask_10m = np.zeros((bin_vol_annual_10m.shape[0]))
                            bin_counts, bin_edges = np.histogram(bin_z_init, bins=elev_bins, weights=bin_debris_hd)
                            bin_debris_mask_10m[bin_counts > 0] = 1
                            bin_vol_annual_10m_bd = bin_vol_annual_10m * bin_debris_mask_10m[:,np.newaxis]
                            
                            # All
                            if reg_vol_annual_binned_bd is None:
                                reg_vol_annual_binned_bd = bin_vol_annual_10m_bd
                            else:
                                reg_vol_annual_binned_bd += bin_vol_annual_10m_bd
                            # O2Region
                            if regO2_vol_annual_binned_bd is None:
                                regO2_vol_annual_binned_bd = np.zeros((len(unique_regO2s), bin_vol_annual_10m.shape[0], years.shape[0]))
                                regO2_vol_annual_binned_bd[regO2_idx,:,:] = bin_vol_annual_10m_bd
                            else:
                                regO2_vol_annual_binned_bd[regO2_idx,:,:] += bin_vol_annual_10m_bd
                            # Watershed
                            if watershed_vol_annual_binned_bd is None:
                                watershed_vol_annual_binned_bd = np.zeros((len(unique_watersheds), bin_vol_annual_10m.shape[0], years.shape[0]))
                                watershed_vol_annual_binned_bd[watershed_idx,:,:] = bin_vol_annual_10m_bd
                            else:
                                watershed_vol_annual_binned_bd[watershed_idx,:,:] += bin_vol_annual_10m_bd
    
    
                        # ----- 8. Binned glacier area vs. Time ----- 
                        bin_area_annual_10m = np.zeros((len(elev_bins)-1, len(years)))
                        for ncol, year in enumerate(years):
                            bin_counts, bin_edges = np.histogram(bin_z_surf_annual[:,ncol], bins=elev_bins, 
                                                                 weights=bin_area_annual[:,ncol])
                            bin_area_annual_10m[:,ncol] = bin_counts
                        
                        # All
                        if reg_area_annual_binned is None:
                            reg_area_annual_binned = bin_area_annual_10m
                        else:
                            reg_area_annual_binned += bin_area_annual_10m
                        # O2Region
                        if regO2_area_annual_binned is None:
                            regO2_area_annual_binned = np.zeros((len(unique_regO2s), bin_area_annual_10m.shape[0], years.shape[0]))
                            regO2_area_annual_binned[regO2_idx,:,:] = bin_area_annual_10m
                        else:
                            regO2_area_annual_binned[regO2_idx,:,:] += bin_area_annual_10m
                        # Watershed
                        if watershed_area_annual_binned is None:
                            watershed_area_annual_binned = np.zeros((len(unique_watersheds), bin_area_annual_10m.shape[0], years.shape[0]))
                            watershed_area_annual_binned[watershed_idx,:,:] = bin_area_annual_10m
                        else:
                            watershed_area_annual_binned[watershed_idx,:,:] += bin_area_annual_10m
    
    
                        
                        # ----- 9. Binned glacier area below debris vs. Time ----- 
                        if bin_debris_hd.sum() > 0:
                            # Bin debris mask for the given elevation bins
                            bin_debris_mask_10m = np.zeros((bin_vol_annual_10m.shape[0]))
                            bin_counts, bin_edges = np.histogram(bin_z_init, bins=elev_bins, weights=bin_debris_hd)
                            bin_debris_mask_10m[bin_counts > 0] = 1
                            bin_area_annual_10m_bd = bin_area_annual_10m * bin_debris_mask_10m[:,np.newaxis]
                            
                            # All
                            if reg_area_annual_binned_bd is None:
                                reg_area_annual_binned_bd = bin_area_annual_10m_bd
                            else:
                                reg_area_annual_binned_bd += bin_area_annual_10m_bd
                            # O2Region
                            if regO2_area_annual_binned_bd is None:
                                regO2_area_annual_binned_bd = np.zeros((len(unique_regO2s), bin_area_annual_10m.shape[0], years.shape[0]))
                                regO2_area_annual_binned_bd[regO2_idx,:,:] = bin_area_annual_10m_bd
                            else:
                                regO2_area_annual_binned_bd[regO2_idx,:,:] += bin_area_annual_10m_bd
                            # Watershed
                            if watershed_area_annual_binned_bd is None:
                                watershed_area_annual_binned_bd = np.zeros((len(unique_watersheds), bin_area_annual_10m.shape[0], years.shape[0]))
                                watershed_area_annual_binned_bd[watershed_idx,:,:] = bin_area_annual_10m_bd
                            else:
                                watershed_area_annual_binned_bd[watershed_idx,:,:] += bin_area_annual_10m_bd


                        # ----- 11. Binned Climatic Mass Balance vs. Time -----
                        # - Various mass balance datasets may have slight mismatch due to averaging
                        #   ex. mbclim_annual was reported in mwe, so the area average will cause difference
                        #   ex. mbtotal_monthly was averaged on a monthly basis, so the temporal average will cause difference
                        bin_mbclim_annual = ds_binned.bin_massbalclim_annual.values[0,:,:]
                        bin_mbclim_annual_m3we = bin_mbclim_annual * bin_area_annual
    
    #                    glac_massbaltotal_annual_0 = bin_mbclim_annual_m3we.sum(0)
    #                    glac_massbaltotal_annual_1 = glac_massbaltotal_monthly.reshape(-1,12).sum(1)
    #                    glac_massbaltotal_annual_2 = ((glac_vol_annual[1:] - glac_vol_annual[0:-1]) * 
    #                                                  pygem_prms.density_ice / pygem_prms.density_water)
                        
                        bin_mbclim_annual_10m = np.zeros((len(elev_bins)-1, len(years)))
                        for ncol, year in enumerate(years):
                            bin_counts, bin_edges = np.histogram(bin_z_surf_annual[:,ncol], bins=elev_bins, 
                                                                 weights=bin_mbclim_annual_m3we[:,ncol])
                            bin_mbclim_annual_10m[:,ncol] = bin_counts
                        # All
                        if reg_mbclim_annual_binned is None:
                            reg_mbclim_annual_binned = bin_mbclim_annual_10m
                        else:
                            reg_mbclim_annual_binned += bin_mbclim_annual_10m
                        # O2Region
                        if regO2_mbclim_annual_binned is None:
                            regO2_mbclim_annual_binned = np.zeros((len(unique_regO2s), bin_mbclim_annual_10m.shape[0], years.shape[0]))
                            regO2_mbclim_annual_binned[regO2_idx,:,:] = bin_mbclim_annual_10m
                        else:
                            regO2_mbclim_annual_binned[regO2_idx,:,:] += bin_mbclim_annual_10m
                        # Watershed
                        if watershed_mbclim_annual_binned is None:
                            watershed_mbclim_annual_binned = np.zeros((len(unique_watersheds), bin_mbclim_annual_10m.shape[0], years.shape[0]))
                            watershed_mbclim_annual_binned[watershed_idx,:,:] = bin_mbclim_annual_10m
                        else:
                            watershed_mbclim_annual_binned[watershed_idx,:,:] += bin_mbclim_annual_10m
    
                        
                        # ----- 12. Runoff vs. Time -----
                        glac_runoff_monthly = ds_stats.glac_runoff_monthly.values[0,:]
                        # Moving-gauge Runoff vs. Time
                        # All
                        if reg_runoff_monthly_moving is None:
                            reg_runoff_monthly_moving = glac_runoff_monthly
                        else:
                            reg_runoff_monthly_moving += glac_runoff_monthly
                        # O2Region
                        if regO2_runoff_monthly_moving is None:
                            regO2_runoff_monthly_moving = np.zeros((len(unique_regO2s),len(ds_stats.time.values)))
                            regO2_runoff_monthly_moving[regO2_idx,:] = glac_runoff_monthly
                        else:
                            regO2_runoff_monthly_moving[regO2_idx,:] += glac_runoff_monthly
                        # watershed
                        if watershed_runoff_monthly_moving is None:
                            watershed_runoff_monthly_moving = np.zeros((len(unique_watersheds),len(ds_stats.time.values)))
                            watershed_runoff_monthly_moving[watershed_idx,:] = glac_runoff_monthly
                        else:
                            watershed_runoff_monthly_moving[watershed_idx,:] += glac_runoff_monthly
                        # DegId
                        if degid_runoff_monthly_moving is None:
                            degid_runoff_monthly_moving = np.zeros((len(unique_degids),len(ds_stats.time.values)))
                            degid_runoff_monthly_moving[degid_idx,:] = glac_runoff_monthly
                        else:
                            degid_runoff_monthly_moving[degid_idx,:] += glac_runoff_monthly
                            
                        # Fixed-gauge Runoff vs. Time
                        offglac_runoff_monthly = ds_stats.offglac_runoff_monthly.values[0,:]
                        glac_runoff_monthly_fixed = glac_runoff_monthly + offglac_runoff_monthly
                        # All
                        if reg_runoff_monthly_fixed is None:
                            reg_runoff_monthly_fixed = glac_runoff_monthly_fixed
                        else:
                            reg_runoff_monthly_fixed += glac_runoff_monthly_fixed
                        # O2Region
                        if regO2_runoff_monthly_fixed is None:
                            regO2_runoff_monthly_fixed = np.zeros((len(unique_regO2s),len(ds_stats.time.values)))
                            regO2_runoff_monthly_fixed[regO2_idx,:] = glac_runoff_monthly_fixed
                        else:
                            regO2_runoff_monthly_fixed[regO2_idx,:] += glac_runoff_monthly_fixed
                        # Watershed
                        if watershed_runoff_monthly_fixed is None:
                            watershed_runoff_monthly_fixed = np.zeros((len(unique_watersheds),len(ds_stats.time.values)))
                            watershed_runoff_monthly_fixed[watershed_idx,:] = glac_runoff_monthly_fixed
                        else:
                            watershed_runoff_monthly_fixed[watershed_idx,:] += glac_runoff_monthly_fixed
                        # DegId
                        if degid_runoff_monthly_fixed is None:
                            degid_runoff_monthly_fixed = np.zeros((len(unique_degids),len(ds_stats.time.values)))
                            degid_runoff_monthly_fixed[degid_idx,:] = glac_runoff_monthly_fixed
                        else:
                            degid_runoff_monthly_fixed[degid_idx,:] += glac_runoff_monthly_fixed

    
                        # ----- 13. ELA vs. Time -----
                        glac_ela_annual = ds_stats.glac_ELA_annual.values[0,:]
                        if np.isnan(glac_ela_annual).any():
                            # Quality control nan values 
                            #  - replace with max elev because occur when entire glacier has neg mb
                            bin_z_surf_annual_glaconly = bin_z_surf_annual.copy()
                            bin_z_surf_annual_glaconly[bin_thick_annual == 0] = np.nan
                            zmax_annual = np.nanmax(bin_z_surf_annual_glaconly, axis=0)
                            glac_ela_annual[np.isnan(glac_ela_annual)] = zmax_annual[np.isnan(glac_ela_annual)]
    
                        # Area-weighted ELA
                        # All
                        if reg_ela_annual is None:
                            reg_ela_annual = glac_ela_annual
                            reg_ela_annual_area = glac_area_annual.copy()
                        else:
                            # Use index to avoid dividing by 0 when glacier completely melts                            
                            ela_idx = np.where(reg_ela_annual_area + glac_area_annual > 0)[0]
                            reg_ela_annual[ela_idx] = (
                                    (reg_ela_annual[ela_idx] * reg_ela_annual_area[ela_idx] + 
                                     glac_ela_annual[ela_idx] * glac_area_annual[ela_idx]) / 
                                    (reg_ela_annual_area[ela_idx] + glac_area_annual[ela_idx]))
                            reg_ela_annual_area += glac_area_annual
                        
                        # O2Region
                        if regO2_ela_annual is None:
                            regO2_ela_annual = np.zeros((len(unique_regO2s),years.shape[0]))
                            regO2_ela_annual[regO2_idx,:] = glac_ela_annual
                            regO2_ela_annual_area = np.zeros((len(unique_regO2s),years.shape[0]))
                            regO2_ela_annual_area[regO2_idx,:] = glac_area_annual.copy()
                        else:
                            ela_idx = np.where(regO2_ela_annual_area[regO2_idx,:] + glac_area_annual > 0)[0]
                            regO2_ela_annual[regO2_idx,ela_idx] = (
                                    (regO2_ela_annual[regO2_idx,ela_idx] * regO2_ela_annual_area[regO2_idx,ela_idx] + 
                                     glac_ela_annual[ela_idx] * glac_area_annual[ela_idx]) / 
                                     (regO2_ela_annual_area[regO2_idx,ela_idx] + glac_area_annual[ela_idx]))
                            regO2_ela_annual_area[regO2_idx,:] += glac_area_annual
                        
                        # Watershed
                        if watershed_ela_annual is None:
                            watershed_ela_annual = np.zeros((len(unique_watersheds),years.shape[0]))
                            watershed_ela_annual[watershed_idx,:] = glac_ela_annual
                            watershed_ela_annual_area = np.zeros((len(unique_watersheds),years.shape[0]))
                            watershed_ela_annual_area[watershed_idx,:] = glac_area_annual.copy()
                        else:
                            ela_idx = np.where(watershed_ela_annual_area[watershed_idx,:] + glac_area_annual > 0)[0]
                            watershed_ela_annual[watershed_idx,ela_idx] = (
                                    (watershed_ela_annual[watershed_idx,ela_idx] * watershed_ela_annual_area[watershed_idx,ela_idx] + 
                                     glac_ela_annual[ela_idx] * glac_area_annual[ela_idx]) / 
                                     (watershed_ela_annual_area[watershed_idx,ela_idx] + glac_area_annual[ela_idx]))
                            watershed_ela_annual_area[watershed_idx,:] += glac_area_annual
                        

                        # ----- 14. AAR vs. Time -----
                        #  - averaging issue with bin_area_annual.sum(0) != glac_area_annual
                        #  - hence only use these 
                        bin_area_annual_acc = bin_area_annual.copy()
                        bin_area_annual_acc[bin_mbclim_annual <= 0] = 0
                        glac_area_annual_acc = bin_area_annual_acc.sum(0)
                        glac_area_annual_frombins = bin_area_annual.sum(0)
                        
                        # All
                        if reg_area_annual_acc is None:
                            reg_area_annual_acc = glac_area_annual_acc.copy()
                            reg_area_annual_frombins = glac_area_annual_frombins.copy()
                            reg_aar_annual = np.zeros(reg_area_annual_acc.shape)
                            reg_aar_annual[reg_area_annual_frombins > 0] = (
                                    reg_area_annual_acc[reg_area_annual_frombins > 0] / 
                                    reg_area_annual_frombins[reg_area_annual_frombins > 0])
                        else:
                            reg_area_annual_acc += glac_area_annual_acc
                            reg_area_annual_frombins += glac_area_annual_frombins
                            reg_aar_annual = np.zeros(reg_area_annual_acc.shape)
                            reg_aar_annual[reg_area_annual_frombins > 0] = (
                                    reg_area_annual_acc[reg_area_annual_frombins > 0] / 
                                    reg_area_annual_frombins[reg_area_annual_frombins > 0])
                        # O2Regions
                        if regO2_area_annual_acc is None:
                            regO2_area_annual_acc = np.zeros((len(unique_regO2s),years.shape[0]))
                            regO2_area_annual_acc[regO2_idx,:] = glac_area_annual_acc.copy()
                            regO2_area_annual_frombins = np.zeros((len(unique_regO2s),years.shape[0]))
                            regO2_area_annual_frombins[regO2_idx,:] = glac_area_annual_frombins.copy()
                            regO2_aar_annual = np.zeros((len(unique_regO2s),years.shape[0]))
                            regO2_aar_annual[regO2_idx,:][regO2_area_annual_frombins[regO2_idx,:] > 0] = (
                                    regO2_area_annual_acc[regO2_idx,:][regO2_area_annual_frombins[regO2_idx,:] > 0] / 
                                    regO2_area_annual_frombins[regO2_idx,:][regO2_area_annual_frombins[regO2_idx,:] > 0])
                        else:
                            regO2_area_annual_acc[regO2_idx,:] += glac_area_annual_acc
                            regO2_area_annual_frombins[regO2_idx,:] += glac_area_annual_frombins
                            regO2_aar_annual[regO2_idx,:][regO2_area_annual_frombins[regO2_idx,:] > 0] = (
                                    regO2_area_annual_acc[regO2_idx,:][regO2_area_annual_frombins[regO2_idx,:] > 0] / 
                                    regO2_area_annual_frombins[regO2_idx,:][regO2_area_annual_frombins[regO2_idx,:] > 0])
                        # Watersheds
                        if watershed_area_annual_acc is None:
                            watershed_area_annual_acc = np.zeros((len(unique_watersheds),years.shape[0]))
                            watershed_area_annual_acc[watershed_idx,:] = glac_area_annual_acc.copy()
                            watershed_area_annual_frombins = np.zeros((len(unique_watersheds),years.shape[0]))
                            watershed_area_annual_frombins[watershed_idx,:] = glac_area_annual_frombins.copy()
                            watershed_aar_annual = np.zeros((len(unique_watersheds),years.shape[0]))
                            watershed_aar_annual[watershed_idx,:][watershed_area_annual_frombins[watershed_idx,:] > 0] = (
                                    watershed_area_annual_acc[watershed_idx,:][watershed_area_annual_frombins[watershed_idx,:] > 0] / 
                                    watershed_area_annual_frombins[watershed_idx,:][watershed_area_annual_frombins[watershed_idx,:] > 0])
                        else:
                            watershed_area_annual_acc[watershed_idx,:] += glac_area_annual_acc
                            watershed_area_annual_frombins[watershed_idx,:] += glac_area_annual_frombins
                            watershed_aar_annual[watershed_idx,:][watershed_area_annual_frombins[watershed_idx,:] > 0] = (
                                    watershed_area_annual_acc[watershed_idx,:][watershed_area_annual_frombins[watershed_idx,:] > 0] / 
                                    watershed_area_annual_frombins[watershed_idx,:][watershed_area_annual_frombins[watershed_idx,:] > 0])
                        
                    # ===== PICKLE DATASETS =====
                    # Volume
                    with open(pickle_fp_reg + fn_reg_vol_annual, 'wb') as f:
                        pickle.dump(reg_vol_annual, f)
                    with open(pickle_fp_regO2 + fn_regO2_vol_annual, 'wb') as f:
                        pickle.dump(regO2_vol_annual, f)
                    with open(pickle_fp_watershed + fn_watershed_vol_annual, 'wb') as f:
                        pickle.dump(watershed_vol_annual, f)
                    with open(pickle_fp_degid + fn_degid_vol_annual, 'wb') as f:
                        pickle.dump(degid_vol_annual, f)
                    # Volume below sea level 
                    with open(pickle_fp_reg + fn_reg_vol_annual_bwl, 'wb') as f:
                        pickle.dump(reg_vol_annual_bwl, f)
                    with open(pickle_fp_regO2 + fn_regO2_vol_annual_bwl, 'wb') as f:
                        pickle.dump(regO2_vol_annual_bwl, f)
                    with open(pickle_fp_watershed + fn_watershed_vol_annual_bwl, 'wb') as f:
                        pickle.dump(watershed_vol_annual_bwl, f)
                    with open(pickle_fp_degid + fn_degid_vol_annual_bwl, 'wb') as f:
                        pickle.dump(degid_vol_annual_bwl, f) 
                    # Volume below debris
                    with open(pickle_fp_reg + fn_reg_vol_annual_bd, 'wb') as f:
                        pickle.dump(reg_vol_annual_bd, f)
                    with open(pickle_fp_regO2 + fn_regO2_vol_annual_bd, 'wb') as f:
                        pickle.dump(regO2_vol_annual_bd, f)
                    with open(pickle_fp_watershed + fn_watershed_vol_annual_bd, 'wb') as f:
                        pickle.dump(watershed_vol_annual_bd, f)
                    with open(pickle_fp_degid + fn_degid_vol_annual_bd, 'wb') as f:
                        pickle.dump(degid_vol_annual_bd, f)
                    # Area 
                    with open(pickle_fp_reg + fn_reg_area_annual, 'wb') as f:
                        pickle.dump(reg_area_annual, f)
                    with open(pickle_fp_regO2 + fn_regO2_area_annual, 'wb') as f:
                        pickle.dump(regO2_area_annual, f)
                    with open(pickle_fp_watershed + fn_watershed_area_annual, 'wb') as f:
                        pickle.dump(watershed_area_annual, f)
                    with open(pickle_fp_degid + fn_degid_area_annual, 'wb') as f:
                        pickle.dump(degid_area_annual, f)
                    # Area below debris
                    with open(pickle_fp_reg + fn_reg_area_annual_bd, 'wb') as f:
                        pickle.dump(reg_area_annual_bd, f)
                    with open(pickle_fp_regO2 + fn_regO2_area_annual_bd, 'wb') as f:
                        pickle.dump(regO2_area_annual_bd, f)
                    with open(pickle_fp_watershed + fn_watershed_area_annual_bd, 'wb') as f:
                        pickle.dump(watershed_area_annual_bd, f)
                    with open(pickle_fp_degid + fn_degid_area_annual_bd, 'wb') as f:
                        pickle.dump(degid_area_annual_bd, f)
                    # Binned Volume
                    with open(pickle_fp_reg + fn_reg_vol_annual_binned, 'wb') as f:
                        pickle.dump(reg_vol_annual_binned, f)
                    with open(pickle_fp_regO2 + fn_regO2_vol_annual_binned, 'wb') as f:
                        pickle.dump(regO2_vol_annual_binned, f)
                    with open(pickle_fp_watershed + fn_watershed_vol_annual_binned, 'wb') as f:
                        pickle.dump(watershed_vol_annual_binned, f)
                    # Binned Volume below debris
                    with open(pickle_fp_reg + fn_reg_vol_annual_binned_bd, 'wb') as f:
                        pickle.dump(reg_vol_annual_binned_bd, f)
                    with open(pickle_fp_regO2 + fn_regO2_vol_annual_binned_bd, 'wb') as f:
                        pickle.dump(regO2_vol_annual_binned_bd, f)
                    with open(pickle_fp_watershed + fn_watershed_vol_annual_binned_bd, 'wb') as f:
                        pickle.dump(watershed_vol_annual_binned_bd, f)
                    # Binned Area
                    with open(pickle_fp_reg + fn_reg_area_annual_binned, 'wb') as f:
                        pickle.dump(reg_area_annual_binned, f)
                    with open(pickle_fp_regO2 + fn_regO2_area_annual_binned, 'wb') as f:
                        pickle.dump(regO2_area_annual_binned, f)
                    with open(pickle_fp_watershed + fn_watershed_area_annual_binned, 'wb') as f:
                        pickle.dump(watershed_area_annual_binned, f)
                    # Binned Area below debris
                    with open(pickle_fp_reg + fn_reg_area_annual_binned_bd, 'wb') as f:
                        pickle.dump(reg_area_annual_binned_bd, f)
                    with open(pickle_fp_regO2 + fn_regO2_area_annual_binned_bd, 'wb') as f:
                        pickle.dump(regO2_area_annual_binned_bd, f)
                    with open(pickle_fp_watershed + fn_watershed_area_annual_binned_bd, 'wb') as f:
                        pickle.dump(watershed_area_annual_binned_bd, f)
                    # Binned Climatic Mass Balance
                    with open(pickle_fp_reg + fn_reg_mbclim_annual_binned, 'wb') as f:
                        pickle.dump(reg_mbclim_annual_binned, f)
                    with open(pickle_fp_regO2 + fn_regO2_mbclim_annual_binned, 'wb') as f:
                        pickle.dump(regO2_mbclim_annual_binned, f)
                    with open(pickle_fp_watershed + fn_watershed_mbclim_annual_binned, 'wb') as f:
                        pickle.dump(watershed_mbclim_annual_binned, f)
                    # Runoff: moving-gauged
                    with open(pickle_fp_reg + fn_reg_runoff_monthly_moving, 'wb') as f:
                        pickle.dump(reg_runoff_monthly_moving, f)
                    with open(pickle_fp_regO2 + fn_regO2_runoff_monthly_moving, 'wb') as f:
                        pickle.dump(regO2_runoff_monthly_moving, f)
                    with open(pickle_fp_watershed + fn_watershed_runoff_monthly_moving, 'wb') as f:
                        pickle.dump(watershed_runoff_monthly_moving, f)
                    with open(pickle_fp_degid + fn_degid_runoff_monthly_moving, 'wb') as f:
                        pickle.dump(degid_runoff_monthly_moving, f)
                    # Runoff: fixed-gauged
                    with open(pickle_fp_reg + fn_reg_runoff_monthly_fixed, 'wb') as f:
                        pickle.dump(reg_runoff_monthly_fixed, f)     
                    with open(pickle_fp_regO2 + fn_regO2_runoff_monthly_fixed, 'wb') as f:
                        pickle.dump(regO2_runoff_monthly_fixed, f)  
                    with open(pickle_fp_watershed + fn_watershed_runoff_monthly_fixed, 'wb') as f:
                        pickle.dump(watershed_runoff_monthly_fixed, f)  
                    with open(pickle_fp_degid + fn_degid_runoff_monthly_fixed, 'wb') as f:
                        pickle.dump(degid_runoff_monthly_fixed, f)  
                    # ELA
                    with open(pickle_fp_reg + fn_reg_ela_annual, 'wb') as f:
                        pickle.dump(reg_ela_annual, f)
                    with open(pickle_fp_regO2 + fn_regO2_ela_annual, 'wb') as f:
                        pickle.dump(regO2_ela_annual, f)
                    with open(pickle_fp_watershed + fn_watershed_ela_annual, 'wb') as f:
                        pickle.dump(watershed_ela_annual, f)
                    # AAR
                    with open(pickle_fp_reg + fn_reg_aar_annual, 'wb') as f:
                        pickle.dump(reg_aar_annual, f)
                    with open(pickle_fp_regO2 + fn_regO2_aar_annual, 'wb') as f:
                        pickle.dump(regO2_aar_annual, f)
                    with open(pickle_fp_watershed + fn_watershed_aar_annual, 'wb') as f:
                        pickle.dump(watershed_aar_annual, f)
                        
                # ----- OTHERWISE LOAD THE PROCESSED DATASETS -----
                else:
                    # Volume
                    with open(pickle_fp_reg + fn_reg_vol_annual, 'rb') as f:
                        reg_vol_annual = pickle.load(f)
                    with open(pickle_fp_regO2 + fn_regO2_vol_annual, 'rb') as f:
                        regO2_vol_annual = pickle.load(f)
                    with open(pickle_fp_watershed + fn_watershed_vol_annual, 'rb') as f:
                        watershed_vol_annual = pickle.load(f)
                    with open(pickle_fp_degid + fn_degid_vol_annual, 'rb') as f:
                        degid_vol_annual = pickle.load(f)
                    # Volume below sea level
                    with open(pickle_fp_reg + fn_reg_vol_annual_bwl, 'rb') as f:
                        reg_vol_annual_bwl = pickle.load(f)
                    with open(pickle_fp_regO2 + fn_regO2_vol_annual_bwl, 'rb') as f:
                        regO2_vol_annual_bwl = pickle.load(f)
                    with open(pickle_fp_watershed + fn_watershed_vol_annual_bwl, 'rb') as f:
                        watershed_vol_annual_bwl = pickle.load(f)
                    with open(pickle_fp_degid + fn_degid_vol_annual_bwl, 'rb') as f:
                        degid_vol_annual_bwl = pickle.load(f)
                    # Volume below debris
                    with open(pickle_fp_reg + fn_reg_vol_annual_bd, 'rb') as f:
                        reg_vol_annual_bd = pickle.load(f)
                    with open(pickle_fp_regO2 + fn_regO2_vol_annual_bd, 'rb') as f:
                        regO2_vol_annual_bd = pickle.load(f)
                    with open(pickle_fp_watershed + fn_watershed_vol_annual_bd, 'rb') as f:
                        watershed_vol_annual_bd = pickle.load(f)
                    with open(pickle_fp_degid + fn_degid_vol_annual_bd, 'rb') as f:
                        degid_vol_annual_bd = pickle.load(f)
                    # Area 
                    with open(pickle_fp_reg + fn_reg_area_annual, 'rb') as f:
                        reg_area_annual = pickle.load(f)
                    with open(pickle_fp_regO2 + fn_regO2_area_annual, 'rb') as f:
                        regO2_area_annual = pickle.load(f)
                    with open(pickle_fp_watershed + fn_watershed_area_annual, 'rb') as f:
                        watershed_area_annual = pickle.load(f)
                    with open(pickle_fp_degid + fn_degid_area_annual, 'rb') as f:
                        degid_area_annual = pickle.load(f)
                    # Area below debris
                    with open(pickle_fp_reg + fn_reg_area_annual_bd, 'rb') as f:
                        reg_area_annual_bd = pickle.load(f)
                    with open(pickle_fp_regO2 + fn_regO2_area_annual_bd, 'rb') as f:
                        regO2_area_annual_bd = pickle.load(f)
                    with open(pickle_fp_watershed + fn_watershed_area_annual_bd, 'rb') as f:
                        watershed_area_annual_bd = pickle.load(f)
                    with open(pickle_fp_degid + fn_degid_area_annual_bd, 'rb') as f:
                        degid_area_annual_bd = pickle.load(f)
                    # Binned Volume
                    with open(pickle_fp_reg + fn_reg_vol_annual_binned, 'rb') as f:
                        reg_vol_annual_binned = pickle.load(f)
                    with open(pickle_fp_regO2 + fn_regO2_vol_annual_binned, 'rb') as f:
                        regO2_vol_annual_binned = pickle.load(f)
                    with open(pickle_fp_watershed + fn_watershed_vol_annual_binned, 'rb') as f:
                        watershed_vol_annual_binned = pickle.load(f)
                    # Binned Volume below debris
                    with open(pickle_fp_reg + fn_reg_vol_annual_binned_bd, 'rb') as f:
                        reg_vol_annual_binned_bd = pickle.load(f)
                    with open(pickle_fp_regO2 + fn_regO2_vol_annual_binned_bd, 'rb') as f:
                        regO2_vol_annual_binned_bd = pickle.load(f)
                    with open(pickle_fp_watershed + fn_watershed_vol_annual_binned_bd, 'rb') as f:
                        watershed_vol_annual_binned_bd = pickle.load(f)
                    # Binned Area
                    with open(pickle_fp_reg + fn_reg_area_annual_binned, 'rb') as f:
                        reg_area_annual_binned = pickle.load(f)
                    with open(pickle_fp_regO2 + fn_regO2_area_annual_binned, 'rb') as f:
                        regO2_area_annual_binned = pickle.load(f)
                    with open(pickle_fp_watershed + fn_watershed_area_annual_binned, 'rb') as f:
                        watershed_area_annual_binned = pickle.load(f)
                    # Binned Area below debris
                    with open(pickle_fp_reg + fn_reg_area_annual_binned_bd, 'rb') as f:
                        reg_area_annual_binned_bd = pickle.load(f)
                    with open(pickle_fp_regO2 + fn_regO2_area_annual_binned_bd, 'rb') as f:
                        regO2_area_annual_binned_bd = pickle.load(f)
                    with open(pickle_fp_watershed + fn_watershed_area_annual_binned_bd, 'rb') as f:
                        watershed_area_annual_binned_bd = pickle.load(f)
                    # Binned Climatic Mass Balance
                    with open(pickle_fp_reg + fn_reg_mbclim_annual_binned, 'rb') as f:
                        reg_mbclim_annual_binned = pickle.load(f)
                    with open(pickle_fp_regO2 + fn_regO2_mbclim_annual_binned, 'rb') as f:
                        regO2_mbclim_annual_binned = pickle.load(f)
                    with open(pickle_fp_watershed + fn_watershed_mbclim_annual_binned, 'rb') as f:
                        watershed_mbclim_annual_binned = pickle.load(f)
                    # Runoff: moving-gauged
                    with open(pickle_fp_reg + fn_reg_runoff_monthly_moving, 'rb') as f:
                        reg_runoff_monthly_moving = pickle.load(f)
                    with open(pickle_fp_regO2 + fn_regO2_runoff_monthly_moving, 'rb') as f:
                        regO2_runoff_monthly_moving = pickle.load(f)
                    with open(pickle_fp_watershed + fn_watershed_runoff_monthly_moving, 'rb') as f:
                        watershed_runoff_monthly_moving = pickle.load(f)
                    with open(pickle_fp_degid + fn_degid_runoff_monthly_moving, 'rb') as f:
                        degid_runoff_monthly_moving = pickle.load(f)
                    # Runoff: fixed-gauged
                    with open(pickle_fp_reg + fn_reg_runoff_monthly_fixed, 'rb') as f:
                        reg_runoff_monthly_fixed = pickle.load(f)
                    with open(pickle_fp_regO2 + fn_regO2_runoff_monthly_fixed, 'rb') as f:
                        regO2_runoff_monthly_fixed= pickle.load(f)
                    with open(pickle_fp_watershed + fn_watershed_runoff_monthly_fixed, 'rb') as f:
                        watershed_runoff_monthly_fixed = pickle.load(f)
                    with open(pickle_fp_degid + fn_degid_runoff_monthly_fixed, 'rb') as f:
                        degid_runoff_monthly_fixed = pickle.load(f)
                    # ELA
                    with open(pickle_fp_reg + fn_reg_ela_annual, 'rb') as f:
                        reg_ela_annual = pickle.load(f)
                    with open(pickle_fp_regO2 + fn_regO2_ela_annual, 'rb') as f:
                        regO2_ela_annual = pickle.load(f)
                    with open(pickle_fp_watershed + fn_watershed_ela_annual, 'rb') as f:
                        watershed_ela_annual = pickle.load(f)
                    # AAR
                    with open(pickle_fp_reg + fn_reg_aar_annual, 'rb') as f:
                        reg_aar_annual = pickle.load(f)
                    with open(pickle_fp_regO2 + fn_regO2_aar_annual, 'rb') as f:
                        regO2_aar_annual = pickle.load(f)
                    with open(pickle_fp_watershed + fn_watershed_aar_annual, 'rb') as f:
                        watershed_aar_annual = pickle.load(f)
                        
                    # Years
                    if years is None:
                        for nglac, glacno in enumerate(glacno_list[0:1]):
                            # Filenames
                            netcdf_fn_stats_ending = 'MCMC_ba1_50sets_2000_2100_all.nc'
                            netcdf_fn_stats = '_'.join([glacno, gcm_name, rcp, netcdf_fn_stats_ending])
                            ds_stats = xr.open_dataset(netcdf_fp_stats + '/' + netcdf_fn_stats)
                
                            # Years
                            years = ds_stats.year.values
        
                    
                #%%
                if args.option_plot:
                    # ===== REGIONAL PLOTS =====
                    fig_fp_reg = fig_fp + str(reg).zfill(2) + '/'
                    if not os.path.exists(fig_fp_reg):
                        os.makedirs(fig_fp_reg)
                        
                    # ----- FIGURE: DIAGNOSTIC OF EVERYTHING ----- 
                    fig, ax = plt.subplots(3, 4, squeeze=False, sharex=False, sharey=False, 
                                           gridspec_kw = {'wspace':0.7, 'hspace':0.5})
                    label= gcm_name + ' ' + rcp
                    
                    # VOLUME CHANGE
                    ax[0,0].plot(years, reg_vol_annual/1e9, color=rcp_colordict[rcp], linewidth=1, zorder=4, label=None)
                    if not reg_vol_annual_bwl is None:
                        ax[0,0].plot(years, reg_vol_annual_bwl/1e9, color=rcp_colordict[rcp], linewidth=0.5, linestyle='--', zorder=4, label='bwl')
                    if not reg_vol_annual_bd is None:
                        ax[0,0].plot(years, reg_vol_annual_bd/1e9, color=rcp_colordict[rcp], linewidth=0.5, linestyle=':', zorder=4, label='bd')
                    ax[0,0].set_ylabel('Volume (km$^{3}$)')
                    ax[0,0].set_xlim(years.min(), years.max())
                    ax[0,0].xaxis.set_major_locator(MultipleLocator(50))
                    ax[0,0].xaxis.set_minor_locator(MultipleLocator(10))
                    ax[0,0].set_ylim(0,reg_vol_annual.max()*1.05/1e9)
                    ax[0,0].tick_params(direction='inout', right=True)
                    ax[0,0].legend(fontsize=10, labelspacing=0.25, handlelength=1, handletextpad=0.25, borderpad=0, frameon=False
    #                               loc=(0.05,0.05),
                                   )        
                    
    
                    # AREA CHANGE
                    ax[0,1].plot(years, reg_area_annual/1e6, color=rcp_colordict[rcp], linewidth=1, zorder=4, label=None)
                    if not reg_area_annual_bd is None:
                        ax[0,1].plot(years, reg_area_annual_bd/1e6, color=rcp_colordict[rcp], linewidth=0.5, linestyle=':', zorder=4, label='bd')
                    ax[0,1].set_ylabel('Area (km$^{2}$)')
                    ax[0,1].set_xlim(years.min(), years.max())
                    ax[0,1].xaxis.set_major_locator(MultipleLocator(50))
                    ax[0,1].xaxis.set_minor_locator(MultipleLocator(10))
                    ax[0,1].set_ylim(0,reg_area_annual.max()*1.05/1e6)
                    ax[0,1].tick_params(direction='inout', right=True)
                    ax[0,1].legend(fontsize=10, labelspacing=0.25, handlelength=1, handletextpad=0.25, borderpad=0, frameon=False
    #                               loc=(0.05,0.05),
                                   )    
                    
                    
                    # MASS BALANCE
                    reg_mbmwea_annual = ((reg_vol_annual[1:] - reg_vol_annual[:-1]) / reg_area_annual[:-1] * 
                                         pygem_prms.density_ice / pygem_prms.density_water)
                    ax[0,2].plot(years[0:-1], reg_mbmwea_annual, color=rcp_colordict[rcp], linewidth=1, zorder=4, label=None)
                    ax[0,2].set_ylabel('$B$ (m w.e. yr$^{-1}$)')
                    ax[0,2].set_xlim(years.min(), years[0:-1].max())
                    ax[0,2].xaxis.set_major_locator(MultipleLocator(50))
                    ax[0,2].xaxis.set_minor_locator(MultipleLocator(10))
                    ax[0,2].tick_params(direction='inout', right=True)
                    
                    
                    # RUNOFF CHANGE 
                    reg_runoff_annual_fixed = reg_runoff_monthly_fixed.reshape(-1,12).sum(axis=1)
                    reg_runoff_annual_moving = reg_runoff_monthly_moving.reshape(-1,12).sum(axis=1)
                    ax[0,3].plot(years[0:-1], reg_runoff_annual_fixed/1e9, color=rcp_colordict[rcp], linewidth=1, zorder=4, label='Fixed')
                    ax[0,3].plot(years[0:-1], reg_runoff_annual_moving/1e9, color='k', linestyle=':', linewidth=1, zorder=4, label='Moving')
                    ax[0,3].set_ylabel('Runoff (km$^{3}$)')
                    ax[0,3].set_xlim(years.min(), years[0:-1].max())
                    ax[0,3].xaxis.set_major_locator(MultipleLocator(50))
                    ax[0,3].xaxis.set_minor_locator(MultipleLocator(10))
                    ax[0,3].set_ylim(0,reg_runoff_annual_fixed.max()*1.05/1e9)
                    ax[0,3].tick_params(direction='inout', right=True)
                    
                    
                    # BINNED VOLUME
                    elev_bin_major = 1000
                    elev_bin_minor = 250
                    ymin = np.floor(elev_bins[np.nonzero(reg_vol_annual_binned.sum(1))[0][0]] / elev_bin_major) * elev_bin_major
                    ymax = np.ceil(elev_bins[np.nonzero(reg_vol_annual_binned.sum(1))[0][-1]] / elev_bin_major) * elev_bin_major
                    ax[1,0].plot(reg_vol_annual_binned[:,0]/1e9, elev_bins[1:], color='k', linewidth=0.5, zorder=4, label=str(years.min()))
                    ax[1,0].plot(reg_vol_annual_binned[:,-1]/1e9, elev_bins[1:], color='b', linewidth=0.5, zorder=4, label=str(years[-1]))
                    if not reg_vol_annual_bd is None:
                        ax[1,0].plot(reg_vol_annual_binned_bd[:,0]/1e9, elev_bins[1:], color='k', linestyle=':', linewidth=0.5, zorder=4, label=None)
                        ax[1,0].plot(reg_vol_annual_binned_bd[:,-1]/1e9, elev_bins[1:], color='b', linestyle=':', linewidth=0.5, zorder=4, label=None)
                    ax[1,0].set_ylabel('Elevation (m)')
                    ax[1,0].set_xlabel('Volume (km$^{3}$)')
                    ax[1,0].set_xlim(0, reg_vol_annual_binned.max()/1e9)
                    ax[1,0].set_ylim(ymin, ymax)
                    ax[1,0].yaxis.set_major_locator(MultipleLocator(elev_bin_major))
                    ax[1,0].yaxis.set_minor_locator(MultipleLocator(elev_bin_minor))
                    ax[1,0].tick_params(direction='inout', right=True)
                    ax[1,0].legend(fontsize=10, labelspacing=0.25, handlelength=1, handletextpad=0.25, borderpad=0, frameon=False
    #                               loc=(0.05,0.05),
                                   ) 
    
                    # BINNED AREA
                    ax[1,1].plot(reg_area_annual_binned[:,0]/1e6, elev_bins[1:], color='k', linewidth=0.5, zorder=4, label=str(years.min()))
                    ax[1,1].plot(reg_area_annual_binned[:,-1]/1e6, elev_bins[1:], color='b', linewidth=0.5, zorder=4, label=str(years[-1]))
                    if not reg_area_annual_binned_bd is None:
                        ax[1,1].plot(reg_area_annual_binned_bd[:,0]/1e6, elev_bins[1:], color='k', linestyle=':', linewidth=0.5, zorder=4, label=None)
                        ax[1,1].plot(reg_area_annual_binned_bd[:,-1]/1e6, elev_bins[1:], color='b', linestyle=':', linewidth=0.5, zorder=4, label=None)
                    ax[1,1].set_ylabel('Elevation (m)')
                    ax[1,1].set_xlabel('Area (km$^{2}$)')
                    ax[1,1].set_xlim(0, reg_area_annual_binned.max()/1e6)
                    ax[1,1].set_ylim(ymin, ymax)
                    ax[1,1].yaxis.set_major_locator(MultipleLocator(elev_bin_major))
                    ax[1,1].yaxis.set_minor_locator(MultipleLocator(elev_bin_minor))
                    ax[1,1].tick_params(direction='inout', right=True)
                    ax[1,1].legend(fontsize=10, labelspacing=0.25, handlelength=1, handletextpad=0.25, borderpad=0, frameon=False
    #                               loc=(0.05,0.05),
                                   ) 
    
                    # CLIMATIC MASS BALANCE GRADIENT
                    reg_mbclim_annual_binned_mwea = reg_mbclim_annual_binned / reg_area_annual_binned
                    ax[1,2].plot(reg_mbclim_annual_binned_mwea[:,0], elev_bins[1:], color='k', linewidth=0.5, zorder=4, label=str(years.min()))
                    ax[1,2].plot(reg_mbclim_annual_binned_mwea[:,-2], elev_bins[1:], color='b', linewidth=0.5, zorder=4, label=str(years.min()))
                    ax[1,2].set_ylabel('Elevation (m)')
                    ax[1,2].set_xlabel('$b_{clim}$ (m w.e. yr$^{-1}$)')
                    ax[1,2].set_ylim(ymin, ymax)
                    ax[1,2].yaxis.set_major_locator(MultipleLocator(elev_bin_major))
                    ax[1,2].yaxis.set_minor_locator(MultipleLocator(elev_bin_minor))
                    ax[1,2].tick_params(direction='inout', right=True)           
                    ax[1,2].axvline(0, color='k', linewidth=0.25)
                    
                    
                    # RUNOFF COMPONENTS
    #                reg_offglac_melt_annual = reg_offglac_melt_monthly.reshape(-1,12).sum(axis=1)
    #                reg_runoff_annual_moving = reg_runoff_monthly_moving.reshape(-1,12).sum(axis=1)
    #                ax[0,3].plot(years[0:-1], reg_runoff_annual_fixed/1e9, color=rcp_colordict[rcp], linewidth=1, zorder=4, label='Fixed')
    #                ax[0,3].plot(years[0:-1], reg_runoff_annual_moving/1e9, color='k', linestyle=':', linewidth=1, zorder=4, label='Moving')
    #                ax[0,3].set_ylabel('Runoff (km$^{3}$)')
    #                ax[0,3].set_xlim(years.min(), years[0:-1].max())
    #                ax[0,3].xaxis.set_major_locator(MultipleLocator(50))
    #                ax[0,3].xaxis.set_minor_locator(MultipleLocator(10))
    #                ax[0,3].set_ylim(0,reg_runoff_annual_fixed.max()*1.05/1e9)
    #                ax[0,3].tick_params(direction='inout', right=True)
                    
                    
                    # ELA
                    ela_min = np.floor(np.min(reg_ela_annual[0:-1]) / 100) * 100
                    ela_max = np.ceil(np.max(reg_ela_annual[0:-1]) / 100) * 100
                    ax[2,0].plot(years[0:-1], reg_ela_annual[0:-1], color='k', linewidth=0.5, zorder=4, label=str(years.min()))
                    ax[2,0].set_ylabel('ELA (m)')
                    ax[2,0].set_xlim(years.min(), years[0:-1].max())
    #                ax[2,0].set_ylim(ela_min, ela_max)
                    ax[2,0].tick_params(direction='inout', right=True)
                    
                    
                    # AAR
                    ax[2,1].plot(years, reg_aar_annual, color='k', linewidth=0.5, zorder=4, label=str(years.min()))
                    ax[2,1].set_ylabel('AAR (-)')
                    ax[2,1].set_ylim(0,1)
                    ax[2,1].set_xlim(years.min(), years[0:-1].max())
                    ax[2,1].tick_params(direction='inout', right=True)
                    
                    
#                    # MASS BALANCE COMPONENTS
#                    # - these are only meant for monthly and/or relative purposes 
#                    #   mass balance from volume change should be used for annual changes
#                    reg_acc_annual = reg_acc_monthly.reshape(-1,12).sum(axis=1)
#                    # Refreeze
#                    reg_refreeze_annual = reg_refreeze_monthly.reshape(-1,12).sum(axis=1)
#                    # Melt
#                    reg_melt_annual = reg_melt_monthly.reshape(-1,12).sum(axis=1)
#                    # Frontal Ablation
#                    reg_frontalablation_annual = reg_frontalablation_monthly.reshape(-1,12).sum(axis=1)
#                    # Periods
#                    if reg_acc_annual.shape[0] == 101:
#                        period_yrs = 20
#                        periods = (np.arange(years.min(), years[0:100].max(), period_yrs) + period_yrs/2).astype(int)
#                        reg_acc_periods = reg_acc_annual[0:100].reshape(-1,period_yrs).sum(1)
#                        reg_refreeze_periods = reg_refreeze_annual[0:100].reshape(-1,period_yrs).sum(1)
#                        reg_melt_periods = reg_melt_annual[0:100].reshape(-1,period_yrs).sum(1)
#                        reg_frontalablation_periods = reg_frontalablation_annual[0:100].reshape(-1,period_yrs).sum(1)
#                        reg_massbaltotal_periods = reg_acc_periods + reg_refreeze_periods - reg_melt_periods - reg_frontalablation_periods
#                        
#                        # Convert to mwea
#                        reg_area_periods = reg_area_annual[0:100].reshape(-1,period_yrs).mean(1)
#                        reg_acc_periods_mwea = reg_acc_periods / reg_area_periods / period_yrs
#                        reg_refreeze_periods_mwea = reg_refreeze_periods / reg_area_periods / period_yrs
#                        reg_melt_periods_mwea = reg_melt_periods / reg_area_periods / period_yrs
#                        reg_frontalablation_periods_mwea = reg_frontalablation_periods / reg_area_periods / period_yrs
#                        reg_massbaltotal_periods_mwea = reg_massbaltotal_periods / reg_area_periods / period_yrs
#                    else:
#                        assert True==False, 'Set up for different time periods'
#    
#                    # Plot
#                    ax[2,2].bar(periods, reg_acc_periods_mwea + reg_refreeze_periods_mwea, color='#3553A5', width=period_yrs/2-1, label='refreeze', zorder=2)
#                    ax[2,2].bar(periods, reg_acc_periods_mwea, color='#3478BD', width=period_yrs/2-1, label='acc', zorder=3)
#                    if not reg_frontalablation_periods_mwea.sum() == 0:
#                        ax[2,2].bar(periods, -reg_frontalablation_periods_mwea, color='#83439A', width=period_yrs/2-1, label='frontal ablation', zorder=3)
#                    ax[2,2].bar(periods, -reg_melt_periods_mwea - reg_frontalablation_periods_mwea, color='#F47A20', width=period_yrs/2-1, label='melt', zorder=2)
#                    ax[2,2].bar(periods, reg_massbaltotal_periods_mwea, color='#555654', width=period_yrs-2, label='total', zorder=1)
#                    ax[2,2].set_ylabel('$B$ (m w.e. yr$^{-1}$)')
#                    ax[2,2].set_xlim(years.min(), years[0:-1].max())
#                    ax[2,2].xaxis.set_major_locator(MultipleLocator(100))
#                    ax[2,2].xaxis.set_minor_locator(MultipleLocator(20))
#                    ax[2,2].yaxis.set_major_locator(MultipleLocator(1))
#                    ax[2,2].yaxis.set_minor_locator(MultipleLocator(0.25))
#                    ax[2,2].tick_params(direction='inout', right=True)
#                    ax[2,2].legend(fontsize=10, labelspacing=0.25, handlelength=1, handletextpad=0.25, borderpad=0, frameon=False,
#                                   loc=(1.2,0.25)) 
                    
                    
                    # Remove plot in lower right
                    fig.delaxes(ax[2,3])
                    
                    
                    # Title
                    fig.text(0.5, 0.95, rgi_reg_dict[reg] + ' (' + gcm_name + ' ' + rcp + ')', size=12, ha='center', va='top',)
                    
                    # Save figure
                    fig_fn = str(reg) + '_allplots_' + str(years.min()) + '-' + str(years.max()) + '_' + gcm_name + '_' + rcp + '.png'
                    fig.set_size_inches(8,6)
                    fig.savefig(fig_fp_reg + fig_fn, bbox_inches='tight', dpi=300)
            

#%%
if option_calving_mbclim_era5:
    
    calving_rgiid_fn = '/Users/drounce/Documents/HiMAT/calving_data/rgiids_for_will.csv'
    sim_fp = '/Users/drounce/Documents/HiMAT/spc_backup/simulations_Will-calving/'
    rgiid_df = pd.read_csv(calving_rgiid_fn)
    rgiids = list(rgiid_df.rgiid.values)
    glac_str = [x.split('-')[1] for x in rgiids]
    glac_no_list = sorted([str(int(x.split('.')[0])) + '.' + x.split('.')[1] for x in glac_str])

    # Load glaciers
    main_glac_rgi_all = modelsetup.selectglaciersrgitable(glac_no=glac_no_list)

    #%%
    mbclim_term_decadal_all = None
    glacno_list_all = []
    for nglac, glacno in enumerate(glac_no_list[0:1]):
        if nglac%10 == 0:
            print(nglac, glacno)
        
        reg = glacno.split('.')[0]
        
        # Filepath and filename
        era5_fp_binned = sim_fp + str(reg).zfill(2) + '/ERA5/binned/'
        glac_fn_binned = glacno + '_ERA5_MCMC_ba1_50sets_2000_2019_binned.nc'
        
        if os.path.exists(era5_fp_binned + glac_fn_binned):
            ds = xr.open_dataset(era5_fp_binned + glac_fn_binned)
    
            mbclim_binned = ds.bin_massbalclim_annual.values[0,:,:-1]
            mbclim_binned_mad = ds.bin_massbalclim_annual_mad.values[0,:,:-1]
            
            # Terminus average (lower 2 bins)
            mbclim_term = mbclim_binned[-2:,:].mean(0)
            mbclim_term_mad = mbclim_binned_mad[-2:,:].mean(0)
            # Decadal average at the terminus (mwea)
            mbclim_term_decadal = mbclim_term.reshape(-1,10).mean(1)
            mbclim_term_decadal_mad = mbclim_term_mad.reshape(-1,10).mean(1)
                
            mbclim_term_decadal_output = np.concatenate((mbclim_term_decadal[np.newaxis,:], 
                                                         mbclim_term_decadal_mad[np.newaxis,:]), axis=1)

            if mbclim_term_decadal_all is None:
                mbclim_term_decadal_all = mbclim_term_decadal_output   
            else:
                mbclim_term_decadal_all = np.concatenate((mbclim_term_decadal_all, 
                                                          mbclim_term_decadal_output), axis=0)
            glacno_list_all.append(glacno)
                    
    mbclim_term_df_cns = ['RGIId', 'mbclim_mwea_2000_2009', 'mbclim_mwea_2010_2019', 
                          'mbclim_mwea_2000_2009_mad', 'mbclim_mwea_2010_2019_mad']
    mbclim_term_df = pd.DataFrame(np.zeros((len(glacno_list_all), mbclim_term_decadal_all.shape[1] +1)), columns=mbclim_term_df_cns)
    mbclim_term_df.loc[:,'RGIId'] = glacno_list_all
    mbclim_term_df.loc[:,mbclim_term_df_cns[1:]] = mbclim_term_decadal_all
    
    csv_fp = netcdf_fp_cmip5 + '/../analysis/csv/'
    if not os.path.exists(csv_fp):
        os.makedirs(csv_fp, exist_ok=True)
    mbclim_term_df_fn = 'ERA5_2000_2019_calving_mbclim_term_mwea.csv'
    mbclim_term_df.to_csv(csv_fp + mbclim_term_df_fn, index=False)


#%%
if option_multigcm_plots_reg:
    
    fig_fp = netcdf_fp_cmip5 + '/../analysis/figures/'
    csv_fp = netcdf_fp_cmip5 + '/../analysis/csv/'
    pickle_fp = fig_fp + '../pickle/'
    
    startyear = 2015
    endyear = 2100
    normyear = 2015
    years = np.arange(2000,2101+1)
    
    gcm_names_rcps = ['CanESM2', 'CCSM4', 'CNRM-CM5', 'CSIRO-Mk3-6-0', 'GFDL-CM3', 
                      'GFDL-ESM2M', 'GISS-E2-R', 'IPSL-CM5A-LR', 'MPI-ESM-LR', 'NorESM1-M']
    gcm_names_ssps = ['BCC-CSM2-MR', 'CESM2', 'CESM2-WACCM', 'EC-Earth3', 'EC-Earth3-Veg', 'FGOALS-f3-L', 
                      'GFDL-ESM4', 'INM-CM4-8', 'INM-CM5-0', 'MPI-ESM1-2-HR', 'MRI-ESM2-0', 'NorESM2-MM']
    rcps = ['ssp126', 'ssp245', 'ssp370', 'ssp585', 'rcp26', 'rcp45', 'rcp85']
    
    regions = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
    
#    rcps_plot_mad = ['rcp26', 'rcp45', 'rcp85', 'ssp126', 'ssp245', 'ssp370', 'ssp585']
    rcps_plot_mad = ['ssp126', 'ssp585']
    
    fig_fp_multigcm = fig_fp + 'multi_gcm/'
    if not os.path.exists(fig_fp_multigcm):
        os.makedirs(fig_fp_multigcm, exist_ok=True)
    
    # Set up processing
    reg_vol_all = {}
    reg_vol_all_bwl = {}
    reg_area_all = {} 
    reg_runoff_all = {}
    
    # Set up Global region
    reg_vol_all['all'] = {}
    reg_vol_all_bwl['all'] = {}
    reg_area_all['all'] = {}
    reg_runoff_all['all'] = {}
    for rcp in rcps:
        reg_vol_all['all'][rcp] = {}
        reg_vol_all_bwl['all'][rcp] = {}
        reg_area_all['all'][rcp] = {}
        reg_runoff_all['all'][rcp] = {}
        if 'rcp' in rcp:
            gcm_names = gcm_names_rcps
        elif 'ssp' in rcp:
            gcm_names = gcm_names_ssps
        for gcm_name in gcm_names:
            reg_vol_all['all'][rcp][gcm_name] = None
            reg_vol_all_bwl['all'][rcp][gcm_name] = None
            reg_area_all['all'][rcp][gcm_name] = None
            reg_runoff_all['all'][rcp][gcm_name] = None
            
    for reg in regions:
    
        reg_vol_all[reg] = {}
        reg_vol_all_bwl[reg] = {}
        reg_area_all[reg] = {}
        reg_runoff_all[reg] = {}
        
        for rcp in rcps:
            reg_vol_all[reg][rcp] = {}
            reg_vol_all_bwl[reg][rcp] = {}
            reg_area_all[reg][rcp] = {}
            reg_runoff_all[reg][rcp] = {}
            
            if 'rcp' in rcp:
                gcm_names = gcm_names_rcps
            elif 'ssp' in rcp:
                gcm_names = gcm_names_ssps
                
            for gcm_name in gcm_names:
                
                # ----- GCM/RCP PICKLE FILEPATHS AND FILENAMES -----
                pickle_fp_reg =  pickle_fp + str(reg).zfill(2) + '/O1Regions/' + gcm_name + '/' + rcp + '/'
                pickle_fp_watershed =  pickle_fp + str(reg).zfill(2) + '/watersheds/' + gcm_name + '/' + rcp + '/'
                # Region string prefix
                reg_rcp_gcm_str = 'R' + str(reg) + '_' + rcp + '_' + gcm_name
                watershed_rcp_gcm_str = 'R' + str(reg) + '_watersheds_' + rcp + '_' + gcm_name
                
                # Filenames
                fn_reg_vol_annual = reg_rcp_gcm_str + '_vol_annual.pkl' 
                fn_reg_vol_annual_bwl = reg_rcp_gcm_str + '_vol_annual_bwl.pkl'
                fn_reg_vol_annual_bd = reg_rcp_gcm_str + '_vol_annual_bd.pkl'
                fn_reg_area_annual = reg_rcp_gcm_str + '_area_annual.pkl'
                fn_reg_area_annual_bd = reg_rcp_gcm_str + '_area_annual_bd.pkl'
                fn_reg_vol_annual_binned = reg_rcp_gcm_str + '_vol_annual_binned.pkl'
                fn_reg_vol_annual_binned_bd = reg_rcp_gcm_str + '_vol_annual_binned_bd.pkl'
                fn_reg_area_annual_binned = reg_rcp_gcm_str + '_area_annual_binned.pkl'
                fn_reg_area_annual_binned_bd = reg_rcp_gcm_str + '_area_annual_binned_bd.pkl'
                fn_reg_acc_monthly = reg_rcp_gcm_str + '_acc_monthly.pkl'
                fn_reg_refreeze_monthly = reg_rcp_gcm_str + '_refreeze_monthly.pkl'
                fn_reg_melt_monthly = reg_rcp_gcm_str + '_melt_monthly.pkl'
                fn_reg_frontalablation_monthly = reg_rcp_gcm_str + '_frontalablation_monthly.pkl'
                fn_reg_massbaltotal_monthly = reg_rcp_gcm_str + '_massbaltotal_monthly.pkl'
                fn_reg_mbclim_annual_binned = reg_rcp_gcm_str + '_mbclim_annual_binned.pkl'
                fn_watershed_mbclim_annual_binned = watershed_rcp_gcm_str + '_mbclim_annual_binned.pkl'
                fn_reg_runoff_monthly_moving = reg_rcp_gcm_str + '_runoff_monthly_moving.pkl'
                fn_watershed_runoff_monthly_moving = watershed_rcp_gcm_str + '_runoff_monthly_moving.pkl'
                fn_reg_runoff_monthly_fixed = reg_rcp_gcm_str + '_runoff_monthly_fixed.pkl'
                fn_watershed_runoff_monthly_fixed = watershed_rcp_gcm_str + '_runoff_monthly_fixed.pkl'
                fn_reg_prec_monthly = reg_rcp_gcm_str + '_prec_monthly.pkl'
                fn_watershed_prec_monthly = watershed_rcp_gcm_str + '_prec_monthly.pkl' 
                fn_reg_offglac_prec_monthly = reg_rcp_gcm_str + '_offglac_prec_monthly.pkl'  
                fn_watershed_offglac_prec_monthly = watershed_rcp_gcm_str + '_offglac_prec_monthly.pkl'
                fn_reg_offglac_melt_monthly = reg_rcp_gcm_str + '_offglac_melt_monthly.pkl'
                fn_watershed_offglac_melt_monthly = watershed_rcp_gcm_str + '_offglac_melt_monthly.pkl'
                fn_reg_offglac_refreeze_monthly = reg_rcp_gcm_str + '_offglac_refreeze_monthly.pkl'
                fn_watershed_offglac_refreeze_monthly = watershed_rcp_gcm_str + '_offglac_refreeze_monthly.pkl'
                fn_reg_ela_annual = reg_rcp_gcm_str + '_ela_annual.pkl'
                fn_reg_aar_annual = reg_rcp_gcm_str + '_aar_annual.pkl'
                
                 # Volume
                with open(pickle_fp_reg + fn_reg_vol_annual, 'rb') as f:
                    reg_vol_annual = pickle.load(f)
                # Volume below sea level
                with open(pickle_fp_reg + fn_reg_vol_annual_bwl, 'rb') as f:
                    reg_vol_annual_bwl = pickle.load(f)
                # Volume below debris
                with open(pickle_fp_reg + fn_reg_vol_annual_bd, 'rb') as f:
                    reg_vol_annual_bd = pickle.load(f)
                # Area 
                with open(pickle_fp_reg + fn_reg_area_annual, 'rb') as f:
                    reg_area_annual = pickle.load(f)
                # Area below debris
                with open(pickle_fp_reg + fn_reg_area_annual_bd, 'rb') as f:
                    reg_area_annual_bd = pickle.load(f)
                # Binned Volume
                with open(pickle_fp_reg + fn_reg_vol_annual_binned, 'rb') as f:
                    reg_vol_annual_binned = pickle.load(f)
                # Binned Volume below debris
                with open(pickle_fp_reg + fn_reg_vol_annual_binned_bd, 'rb') as f:
                    reg_vol_annual_binned_bd = pickle.load(f)
                # Binned Area
                with open(pickle_fp_reg + fn_reg_area_annual_binned, 'rb') as f:
                    reg_area_annual_binned = pickle.load(f)
                # Binned Area below debris
                with open(pickle_fp_reg + fn_reg_area_annual_binned_bd, 'rb') as f:
                    reg_area_annual_binned_bd = pickle.load(f)
                # Mass balance: accumulation
                with open(pickle_fp_reg + fn_reg_acc_monthly, 'rb') as f:
                    reg_acc_monthly = pickle.load(f)
                # Mass balance: refreeze
                with open(pickle_fp_reg + fn_reg_refreeze_monthly, 'rb') as f:
                    reg_refreeze_monthly = pickle.load(f)
                # Mass balance: melt
                with open(pickle_fp_reg + fn_reg_melt_monthly, 'rb') as f:
                    reg_melt_monthly = pickle.load(f)
                # Mass balance: frontal ablation
                with open(pickle_fp_reg + fn_reg_frontalablation_monthly, 'rb') as f:
                    reg_frontalablation_monthly = pickle.load(f)
                # Mass balance: total mass balance
                with open(pickle_fp_reg + fn_reg_massbaltotal_monthly, 'rb') as f:
                    reg_massbaltotal_monthly = pickle.load(f)
                # Binned Climatic Mass Balance
                with open(pickle_fp_reg + fn_reg_mbclim_annual_binned, 'rb') as f:
                    reg_mbclim_annual_binned = pickle.load(f)
                # Runoff: moving-gauged
                with open(pickle_fp_reg + fn_reg_runoff_monthly_moving, 'rb') as f:
                    reg_runoff_monthly_moving = pickle.load(f)
                with open(pickle_fp_watershed + fn_watershed_runoff_monthly_moving, 'rb') as f:
                    watershed_runoff_monthly_moving = pickle.load(f)
                # Runoff: fixed-gauged
                with open(pickle_fp_reg + fn_reg_runoff_monthly_fixed, 'rb') as f:
                    reg_runoff_monthly_fixed = pickle.load(f)
                with open(pickle_fp_watershed + fn_watershed_runoff_monthly_fixed, 'rb') as f:
                    watershed_runoff_monthly_fixed = pickle.load(f)
                # Runoff: precipitation
                with open(pickle_fp_reg + fn_reg_prec_monthly, 'rb') as f:
                    reg_prec_monthly = pickle.load(f)
                with open(pickle_fp_watershed + fn_watershed_prec_monthly, 'rb') as f:
                    watershed_prec_monthly= pickle.load(f)
                # Runoff: off-glacier precipitation
                with open(pickle_fp_reg + fn_reg_offglac_prec_monthly, 'rb') as f:
                    reg_offglac_prec_monthly = pickle.load(f)
                with open(pickle_fp_watershed + fn_watershed_offglac_prec_monthly, 'rb') as f:
                    watershed_offglac_prec_monthly = pickle.load(f)
                # Runoff: off-glacier melt
                with open(pickle_fp_reg + fn_reg_offglac_melt_monthly, 'rb') as f:
                    reg_offglac_melt_monthly = pickle.load(f)
                with open(pickle_fp_watershed + fn_watershed_offglac_melt_monthly, 'rb') as f:
                    watershed_offglac_melt_monthly = pickle.load(f)
                # Runoff: off-glacier refreeze
                with open(pickle_fp_reg + fn_reg_offglac_refreeze_monthly, 'rb') as f:
                    reg_offglac_refreeze_monthly = pickle.load(f)
                with open(pickle_fp_watershed + fn_watershed_offglac_refreeze_monthly, 'rb') as f:
                    watershed_offglac_refreeze_monthly = pickle.load(f)
                # ELA
                with open(pickle_fp_reg + fn_reg_ela_annual, 'rb') as f:
                    reg_ela_annual = pickle.load(f)
                # AAR
                with open(pickle_fp_reg + fn_reg_aar_annual, 'rb') as f:
                    reg_aar_annual = pickle.load(f)
                    
                    
                reg_vol_all[reg][rcp][gcm_name] = reg_vol_annual
                if reg_vol_annual_bwl is None:
                    reg_vol_all_bwl[reg][rcp][gcm_name] = np.zeros(reg_vol_annual.shape)
                else:
                    reg_vol_all_bwl[reg][rcp][gcm_name] = reg_vol_annual_bwl
                reg_area_all[reg][rcp][gcm_name] = reg_area_annual  
                reg_runoff_all[reg][rcp][gcm_name] = reg_runoff_monthly_fixed
                
                if reg_vol_all['all'][rcp][gcm_name] is None:
                    reg_vol_all['all'][rcp][gcm_name] = reg_vol_all[reg][rcp][gcm_name]
                    reg_vol_all_bwl['all'][rcp][gcm_name] = reg_vol_all_bwl[reg][rcp][gcm_name]
                    reg_area_all['all'][rcp][gcm_name] = reg_area_all[reg][rcp][gcm_name]
                    reg_runoff_all['all'][rcp][gcm_name] = reg_runoff_all[reg][rcp][gcm_name]
                else:
                    reg_vol_all['all'][rcp][gcm_name] = reg_vol_all['all'][rcp][gcm_name] + reg_vol_all[reg][rcp][gcm_name]
                    reg_vol_all_bwl['all'][rcp][gcm_name] = reg_vol_all_bwl['all'][rcp][gcm_name] + reg_vol_all_bwl[reg][rcp][gcm_name]
                    reg_area_all['all'][rcp][gcm_name] = reg_area_all['all'][rcp][gcm_name] + reg_area_all[reg][rcp][gcm_name]
                    reg_runoff_all['all'][rcp][gcm_name] = reg_runoff_all['all'][rcp][gcm_name] + reg_runoff_all[reg][rcp][gcm_name]
                
             
    #%%
    regions.append('all')
    # MULTI-GCM STATISTICS
    ds_multigcm_vol = {}
    ds_multigcm_vol_bsl = {}
    ds_multigcm_area = {}
    ds_multigcm_runoff = {}
    for reg in regions:
        ds_multigcm_vol[reg] = {}
        ds_multigcm_vol_bsl[reg] = {}
        ds_multigcm_area[reg] = {}
        ds_multigcm_runoff[reg] = {}
        for rcp in rcps: 
            
            if 'rcp' in rcp:
                gcm_names = gcm_names_rcps
            elif 'ssp' in rcp:
                gcm_names = gcm_names_ssps
                
            for ngcm, gcm_name in enumerate(gcm_names):
                
                print(rcp, gcm_name)
    
                reg_vol_gcm = reg_vol_all[reg][rcp][gcm_name]
                reg_vol_bsl_gcm = reg_vol_all_bwl[reg][rcp][gcm_name]
                reg_area_gcm = reg_area_all[reg][rcp][gcm_name]
                reg_runoff_gcm = reg_runoff_all[reg][rcp][gcm_name]
    
                if ngcm == 0:
                    reg_vol_gcm_all = reg_vol_gcm   
                    reg_vol_bsl_gcm_all = reg_vol_bsl_gcm   
                    reg_area_gcm_all = reg_area_gcm
                    reg_runoff_gcm_all = reg_runoff_gcm
                else:
                    reg_vol_gcm_all = np.vstack((reg_vol_gcm_all, reg_vol_gcm))
                    reg_vol_bsl_gcm_all = np.vstack((reg_vol_bsl_gcm_all, reg_vol_bsl_gcm))
                    reg_area_gcm_all = np.vstack((reg_area_gcm_all, reg_area_gcm))
                    reg_runoff_gcm_all = np.vstack((reg_runoff_gcm_all, reg_runoff_gcm))
            
            ds_multigcm_vol[reg][rcp] = reg_vol_gcm_all
            ds_multigcm_vol_bsl[reg][rcp] = reg_vol_bsl_gcm_all
            ds_multigcm_area[reg][rcp] = reg_area_gcm_all
            ds_multigcm_runoff[reg][rcp] = reg_runoff_gcm_all

#%%
    #        # ----- FIGURE: VOLUME CHANGE FOR EACH GCM ----- 
    #        fig, ax = plt.subplots(1, 1, squeeze=False, sharex=False, sharey=True, 
    #                               gridspec_kw = {'wspace':0, 'hspace':0})
    #        
    #        # Load data
    #        for rcp in rcps:
    #            
    #            for ngcm, gcm_name in enumerate(gcm_names):
    #                
    #                if ngcm == 0:
    #                    label=rcp
    #                else:
    #                    label=None
    #                
    #                # Median and absolute median deviation
    #                reg_vol = reg_vol_all[rcp][gcm_name]
    #                reg_vol_med = np.median(reg_vol, axis=1)
    #                reg_vol_mad = median_abs_deviation(reg_vol, axis=1)
    #            
    #                ax[0,0].plot(years, reg_vol_med, color=rcp_colordict[rcp], linewidth=1, zorder=4, label=label)
    #                ax[0,0].fill_between(years, 
    #                                     reg_vol_med + 1.96*reg_vol_mad, 
    #                                     reg_vol_med - 1.96*reg_vol_mad, 
    #                                     alpha=0.2, facecolor=rcp_colordict[rcp], label=None)
    #                
    #        ax[0,0].set_ylabel('Volume (m$^{3}$)')
    #        ax[0,0].set_xlim(startyear, endyear)
    #        ax[0,0].text(0.98, 1.06, rgi_reg_dict[reg], size=10, horizontalalignment='right', 
    #                     verticalalignment='top', transform=ax[0,0].transAxes)
    #        ax[0,0].legend(
    ##                rcp_lines, rcp_labels, loc=(0.05,0.05), fontsize=10, labelspacing=0.25, handlelength=1, 
    ##                handletextpad=0.25, borderpad=0, frameon=False
    #                )        
    #        ax[0,0].tick_params(direction='inout', right=True)
    #        # Save figure
    #        fig_fn = str(reg) + '_volchange_' + str(startyear) + '-' + str(endyear) + '_all_gcmrcps.png'
    #        fig.set_size_inches(4,3)
    #        fig.savefig(fig_fp + fig_fn, bbox_inches='tight', dpi=300)
    #                    
    #    
    #        #%%
        # ----- FIGURE: VOLUME CHANGE MULTI-GCM -----
        fig, ax = plt.subplots(1, 1, squeeze=False, sharex=False, sharey=True, 
                               gridspec_kw = {'wspace':0, 'hspace':0})
    
        for rcp in rcps:
            
            # Median and absolute median deviation
            reg_vol = ds_multigcm_vol[reg][rcp]
            reg_vol_med = np.median(reg_vol, axis=0)
            reg_vol_mad = median_abs_deviation(reg_vol, axis=0)
            
            ax[0,0].plot(years, reg_vol_med, color=rcp_colordict[rcp], linestyle=rcp_styledict[rcp], linewidth=1, zorder=4, label=rcp)
            if rcp in rcps_plot_mad:
                ax[0,0].fill_between(years, 
                                     reg_vol_med + 1.96*reg_vol_mad, 
                                     reg_vol_med - 1.96*reg_vol_mad, 
                                     alpha=0.2, facecolor=rcp_colordict[rcp], label=None)
               
        ax[0,0].set_ylabel('Volume (m$^{3}$)')
        ax[0,0].set_xlim(startyear, endyear)
        ax[0,0].text(0.98, 1.06, rgi_reg_dict[reg], size=10, horizontalalignment='right', 
                     verticalalignment='top', transform=ax[0,0].transAxes)
        ax[0,0].legend(
    #                rcp_lines, rcp_labels, loc=(0.05,0.05), fontsize=10, labelspacing=0.25, handlelength=1, 
    #                handletextpad=0.25, borderpad=0, frameon=False
                )
        ax[0,0].tick_params(direction='inout', right=True)
        # Save figure
        if 'rcp26' in rcps and 'ssp126' in rcps:
            scenario_str = 'rcps_ssps'
        elif 'rcp26' in rcps:
            scenario_str = 'rcps'
        elif 'ssp126' in rcp:
            scenario_str = 'ssps'
        fig_fn = (str(reg) + '_volchange_' + str(startyear) + '-' + str(endyear) + '_multigcm' + 
                  '-' + scenario_str + '.png')
        fig.set_size_inches(4,3)
        fig.savefig(fig_fp_multigcm + fig_fn, bbox_inches='tight', dpi=300)
        
        #%%
        # ----- FIGURE: NORMALIZED VOLUME CHANGE MULTI-GCM -----
        fig, ax = plt.subplots(1, 1, squeeze=False, sharex=False, sharey=True, 
                               gridspec_kw = {'wspace':0, 'hspace':0})
    
        normyear_idx = np.where(years == normyear)[0][0]
    
        for rcp in rcps:
            
            # Median and absolute median deviation
            reg_vol = ds_multigcm_vol[reg][rcp]
            reg_vol_med = np.median(reg_vol, axis=0)
            reg_vol_mad = median_abs_deviation(reg_vol, axis=0)
            
            reg_vol_med_norm = reg_vol_med / reg_vol_med[normyear_idx]
            reg_vol_mad_norm = reg_vol_mad / reg_vol_med[normyear_idx]
            
            ax[0,0].plot(years, reg_vol_med_norm, color=rcp_colordict[rcp], linestyle=rcp_styledict[rcp], linewidth=1, zorder=4, label=rcp)
            
            if rcp in rcps_plot_mad:
                ax[0,0].fill_between(years, 
                                     reg_vol_med_norm + 1.96*reg_vol_mad_norm, 
                                     reg_vol_med_norm - 1.96*reg_vol_mad_norm, 
                                     alpha=0.2, facecolor=rcp_colordict[rcp], label=None)
        
        ax[0,0].set_ylabel('Volume (-)')
        ax[0,0].set_xlim(startyear, endyear)
        ax[0,0].set_ylim(0,1)
        ax[0,0].xaxis.set_major_locator(MultipleLocator(20))
        ax[0,0].xaxis.set_minor_locator(MultipleLocator(10))
        ax[0,0].yaxis.set_major_locator(MultipleLocator(0.2))
        ax[0,0].yaxis.set_minor_locator(MultipleLocator(0.1))
        ax[0,0].tick_params(direction='inout', right=True)
        ax[0,0].text(0.98, 1.06, rgi_reg_dict[reg], size=10, horizontalalignment='right', 
                     verticalalignment='top', transform=ax[0,0].transAxes)
        ax[0,0].legend(
    #                rcp_lines, rcp_labels, loc=(0.05,0.05), fontsize=10, labelspacing=0.25, handlelength=1, 
    #                handletextpad=0.25, borderpad=0, frameon=False
                )
        ax[0,0].tick_params(direction='inout', right=True)
        # Save figure
        fig_fn = (str(reg) + '_volchangenorm_' + str(startyear) + '-' + str(endyear) + '_multigcm' + 
                  '-' + scenario_str + '.png')
        fig.set_size_inches(4,3)
        fig.savefig(fig_fp_multigcm + fig_fn, bbox_inches='tight', dpi=300)
            
            
            #%%
    #        # ----- FIGURE: AREA CHANGE MULTI-GCM -----
    #        fig, ax = plt.subplots(1, 1, squeeze=False, sharex=False, sharey=True, 
    #                               gridspec_kw = {'wspace':0, 'hspace':0})
    #
    #        for rcp in rcps:
    #            
    #            # Median and absolute median deviation
    #            reg_area = ds_multigcm_area[reg][rcp]
    #            reg_area_med = np.median(reg_area, axis=0)
    #            reg_area_mad = median_abs_deviation(reg_area, axis=0)
    #            
    #            ax[0,0].plot(years, reg_area_med, color=rcp_colordict[rcp], linewidth=1, zorder=4, label=rcp)
    #            if rcp in rcps_plot_mad:
    #                ax[0,0].fill_between(years, 
    #                                     reg_area_med + 1.96*reg_area_mad, 
    #                                     reg_area_med - 1.96*reg_area_mad, 
    #                                     alpha=0.2, facecolor=rcp_colordict[rcp], label=None)
    #               
    #        ax[0,0].set_ylabel('Area (m$^{2}$)')
    #        ax[0,0].set_xlim(startyear, endyear)
    #        ax[0,0].text(0.98, 1.06, rgi_reg_dict[reg], size=10, horizontalalignment='right', 
    #                     verticalalignment='top', transform=ax[0,0].transAxes)
    #        ax[0,0].legend(
    ##                rcp_lines, rcp_labels, loc=(0.05,0.05), fontsize=10, labelspacing=0.25, handlelength=1, 
    ##                handletextpad=0.25, borderpad=0, frameon=False
    #                )
    #        ax[0,0].tick_params(direction='inout', right=True)
    #        # Save figure
    #        fig_fn = str(reg) + '_areachange_' + str(startyear) + '-' + str(endyear) + '_multigcm.png'
    #        fig.set_size_inches(4,3)
    #        fig.savefig(fig_fp + fig_fn, bbox_inches='tight', dpi=300)
    #        
    #        #%%
    #        # ----- FIGURE: NORMALIZED AREA CHANGE MULTI-GCM -----
    #        fig, ax = plt.subplots(1, 1, squeeze=False, sharex=False, sharey=True, 
    #                               gridspec_kw = {'wspace':0, 'hspace':0})
    #
    #        normyear_idx = np.where(years == normyear)[0][0]
    #
    #        for rcp in rcps:
    #            
    #            # Median and absolute median deviation
    #            reg_area = ds_multigcm_area[reg][rcp]
    #            reg_area_med = np.median(reg_area, axis=0)
    #            reg_area_mad = median_abs_deviation(reg_area, axis=0)
    #            
    #            reg_area_med_norm = reg_area_med / reg_area_med[normyear_idx]
    #            reg_area_mad_norm = reg_area_mad / reg_area_med[normyear_idx]
    #            
    #            ax[0,0].plot(years, reg_area_med_norm, color=rcp_colordict[rcp], linewidth=1, zorder=4, label=rcp)
    #            
    #            if rcp in rcps_plot_mad:
    #                ax[0,0].fill_between(years, 
    #                                     reg_area_med_norm + 1.96*reg_area_mad_norm, 
    #                                     reg_area_med_norm - 1.96*reg_area_mad_norm, 
    #                                     alpha=0.2, facecolor=rcp_colordict[rcp], label=None)
    #        
    #        ax[0,0].set_ylabel('Area (-)')
    #        ax[0,0].set_xlim(startyear, endyear)
    #        ax[0,0].text(0.98, 1.06, rgi_reg_dict[reg], size=10, horizontalalignment='right', 
    #                     verticalalignment='top', transform=ax[0,0].transAxes)
    #        ax[0,0].legend(
    ##                rcp_lines, rcp_labels, loc=(0.05,0.05), fontsize=10, labelspacing=0.25, handlelength=1, 
    ##                handletextpad=0.25, borderpad=0, frameon=False
    #                )
    #        ax[0,0].tick_params(direction='inout', right=True)
    #        # Save figure
    #        fig_fn = str(reg) + '_areachangenorm_' + str(startyear) + '-' + str(endyear) + '_multigcm.png'
    #        fig.set_size_inches(4,3)
    #        fig.savefig(fig_fp + fig_fn, bbox_inches='tight', dpi=300)
    
    #%%
    # ----- FIGURE: ALL MULTI-GCM NORMALIZED VOLUME CHANGE -----
    fig = plt.figure()
    gs = fig.add_gridspec(nrows=6,ncols=4,wspace=0.3,hspace=0.4)
    ax1 = fig.add_subplot(gs[0:2,0:2])
    ax2 = fig.add_subplot(gs[0,3])
    ax3 = fig.add_subplot(gs[1,2])
    ax4 = fig.add_subplot(gs[1,3])
    ax5 = fig.add_subplot(gs[2,0])
    ax6 = fig.add_subplot(gs[2,1])
    ax7 = fig.add_subplot(gs[2,2])
    ax8 = fig.add_subplot(gs[2,3])
    ax9 = fig.add_subplot(gs[3,0])
    ax10 = fig.add_subplot(gs[3,1])
    ax11 = fig.add_subplot(gs[3,2])
    ax12 = fig.add_subplot(gs[3,3])
    ax13 = fig.add_subplot(gs[4,0])
    ax14 = fig.add_subplot(gs[4,1])
    ax15 = fig.add_subplot(gs[4,2])
    ax16 = fig.add_subplot(gs[4,3])
    ax17 = fig.add_subplot(gs[5,0])
    ax18 = fig.add_subplot(gs[5,1])
    ax19 = fig.add_subplot(gs[5,2])
    ax20 = fig.add_subplot(gs[5,3])
    
    regions_ordered = ['all',19,3,1,5,9,4,7,17,13,6,14,2,15,8,10,11,16,12,18]
    for nax, ax in enumerate([ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9,ax10,ax11,ax12,ax13,ax14,ax15,ax16,ax17,ax18,ax19,ax20]):
        
        reg = regions_ordered[nax]            
        
        for rcp in rcps:
            
            # Median and absolute median deviation
            reg_vol = ds_multigcm_vol[reg][rcp]
            reg_vol_med = np.median(reg_vol, axis=0)
            reg_vol_mad = median_abs_deviation(reg_vol, axis=0)
            
            reg_vol_med_norm = reg_vol_med / reg_vol_med[normyear_idx]
            reg_vol_mad_norm = reg_vol_mad / reg_vol_med[normyear_idx]
            
            ax.plot(years, reg_vol_med_norm, color=rcp_colordict[rcp], linestyle=rcp_styledict[rcp], 
                    linewidth=1, zorder=4, label=rcp)
            if rcp in rcps_plot_mad:
                ax.fill_between(years, 
                                reg_vol_med_norm + 1.96*reg_vol_mad_norm, 
                                reg_vol_med_norm - 1.96*reg_vol_mad_norm, 
                                alpha=0.2, facecolor=rcp_colordict[rcp], label=None)
        
        if ax in [ax1,ax5,ax9,ax13,ax17]:
            ax.set_ylabel('Mass (rel. to 2015)')
        ax.set_xlim(startyear, endyear)
        ax.xaxis.set_major_locator(MultipleLocator(50))
        ax.xaxis.set_minor_locator(MultipleLocator(10))
        ax.set_ylim(0,1.1)
        ax.yaxis.set_major_locator(MultipleLocator(0.2))
        ax.yaxis.set_minor_locator(MultipleLocator(0.1))
        if nax == 0:
            label_height=1.06
        else:
            label_height=1.14
        ax.text(1, label_height, rgi_reg_dict[reg], size=10, horizontalalignment='right', 
                verticalalignment='top', transform=ax.transAxes)
        ax.tick_params(axis='both', which='major', direction='inout', right=True)
        ax.tick_params(axis='both', which='minor', direction='in', right=True)
        
        if nax == 1:
            if 'rcp26' in rcps and 'ssp126' in rcps and len(rcps)==7:
                labels = ['SSP1-26', 'SSP2-45', 'SSP3-70', 'SSP5-85', 'RCP2.6', 'RCP4.5', 'RCP8.5']
            elif 'rcp26' in rcps and len(rcps) == 3:
                labels = ['RCP2.6', 'RCP4.5', 'RCP8.5']
            elif 'ssp126' in rcp and len(rcps) == 4:
                labels = ['SSP1-26', 'SSP2-45', 'SSP3-70', 'SSP5-85']
            ax.legend(loc=(-1.4,0.2), labels=labels, fontsize=10, ncol=2, columnspacing=0.5, labelspacing=0.25, 
                      handlelength=1, handletextpad=0.25, borderpad=0, frameon=False
                      )
    # Save figure
    if 'rcp26' in rcps and 'ssp126' in rcps:
        scenario_str = 'rcps_ssps'
    elif 'rcp26' in rcps:
        scenario_str = 'rcps'
    elif 'ssp126' in rcp:
        scenario_str = 'ssps'
    fig_fn = ('allregions_volchange_norm_' + str(startyear) + '-' + str(endyear) + '_multigcm' + 
              '-' + scenario_str + '.png')
    fig.set_size_inches(8.5,11)
    fig.savefig(fig_fp_multigcm + fig_fn, bbox_inches='tight', dpi=300)
    
    
    #%% ----- FIGURE: ALL MULTI-GCM NORMALIZED AREA CHANGE -----
    fig = plt.figure()
    gs = fig.add_gridspec(nrows=6,ncols=4,wspace=0.3,hspace=0.4)
    ax1 = fig.add_subplot(gs[0:2,0:2])
    ax2 = fig.add_subplot(gs[0,3])
    ax3 = fig.add_subplot(gs[1,2])
    ax4 = fig.add_subplot(gs[1,3])
    ax5 = fig.add_subplot(gs[2,0])
    ax6 = fig.add_subplot(gs[2,1])
    ax7 = fig.add_subplot(gs[2,2])
    ax8 = fig.add_subplot(gs[2,3])
    ax9 = fig.add_subplot(gs[3,0])
    ax10 = fig.add_subplot(gs[3,1])
    ax11 = fig.add_subplot(gs[3,2])
    ax12 = fig.add_subplot(gs[3,3])
    ax13 = fig.add_subplot(gs[4,0])
    ax14 = fig.add_subplot(gs[4,1])
    ax15 = fig.add_subplot(gs[4,2])
    ax16 = fig.add_subplot(gs[4,3])
    ax17 = fig.add_subplot(gs[5,0])
    ax18 = fig.add_subplot(gs[5,1])
    ax19 = fig.add_subplot(gs[5,2])
    ax20 = fig.add_subplot(gs[5,3])
    
    regions_ordered = ['all',19,3,1,5,9,4,7,17,13,6,14,2,15,8,10,11,16,12,18]
    for nax, ax in enumerate([ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9,ax10,ax11,ax12,ax13,ax14,ax15,ax16,ax17,ax18,ax19,ax20]):
        
        reg = regions_ordered[nax]
        
        for rcp in rcps:
            
            # Median and absolute median deviation
            reg_area = ds_multigcm_area[reg][rcp]
            reg_area_med = np.median(reg_area, axis=0)
            reg_area_mad = median_abs_deviation(reg_area, axis=0)
            
            reg_area_med_norm = reg_area_med / reg_area_med[normyear_idx]
            reg_area_mad_norm = reg_area_mad / reg_area_med[normyear_idx]
            
            ax.plot(years, reg_area_med_norm, color=rcp_colordict[rcp], linestyle=rcp_styledict[rcp], 
                    linewidth=1, zorder=4, label=rcp)
            if rcp in rcps_plot_mad:
                ax.fill_between(years, 
                                reg_area_med_norm + 1.96*reg_area_mad_norm, 
                                reg_area_med_norm - 1.96*reg_area_mad_norm, 
                                alpha=0.2, facecolor=rcp_colordict[rcp], label=None)
        
        if ax in [ax1,ax5,ax9,ax13,ax17]:
            ax.set_ylabel('Area (rel. to 2015)')
        ax.set_xlim(startyear, endyear)
        ax.xaxis.set_major_locator(MultipleLocator(50))
        ax.xaxis.set_minor_locator(MultipleLocator(10))
        ax.set_ylim(0,1.1)
        ax.yaxis.set_major_locator(MultipleLocator(0.2))
        ax.yaxis.set_minor_locator(MultipleLocator(0.1))
        if nax == 0:
            label_height=1.06
        else:
            label_height=1.14
        ax.text(1, label_height, rgi_reg_dict[reg], size=10, horizontalalignment='right', 
                verticalalignment='top', transform=ax.transAxes)
        ax.tick_params(axis='both', which='major', direction='inout', right=True)
        ax.tick_params(axis='both', which='minor', direction='in', right=True)
        
        if nax == 1:
            if 'rcp26' in rcps and 'ssp126' in rcps and len(rcps)==7:
                labels = ['SSP1-26', 'SSP2-45', 'SSP3-70', 'SSP5-85', 'RCP2.6', 'RCP4.5', 'RCP8.5']
            elif 'rcp26' in rcps and len(rcps) == 3:
                labels = ['RCP2.6', 'RCP4.5', 'RCP8.5']
            elif 'ssp126' in rcp and len(rcps) == 4:
                labels = ['SSP1-26', 'SSP2-45', 'SSP3-70', 'SSP5-85']
            ax.legend(loc=(-1.4,0.2), labels=labels, fontsize=10, ncol=2, columnspacing=0.5, labelspacing=0.25, 
                      handlelength=1, handletextpad=0.25, borderpad=0, frameon=False
                      )
    # Save figure
    if 'rcp26' in rcps and 'ssp126' in rcps:
        scenario_str = 'rcps_ssps'
    elif 'rcp26' in rcps:
        scenario_str = 'rcps'
    elif 'ssp126' in rcp:
        scenario_str = 'ssps'
    fig_fn = ('allregions_areachange_norm_' + str(startyear) + '-' + str(endyear) + '_multigcm' + 
              '-' + scenario_str + '.png')
    fig.set_size_inches(8.5,11)
    fig.savefig(fig_fp_multigcm + fig_fn, bbox_inches='tight', dpi=300)
    
    #%% ----- FIGURE: ALL SPECIFIC MASS LOSS RATES -----
    fig = plt.figure()
    gs = fig.add_gridspec(nrows=6,ncols=4,wspace=0.3,hspace=0.4)
    ax1 = fig.add_subplot(gs[0:2,0:2])
    ax2 = fig.add_subplot(gs[0,3])
    ax3 = fig.add_subplot(gs[1,2])
    ax4 = fig.add_subplot(gs[1,3])
    ax5 = fig.add_subplot(gs[2,0])
    ax6 = fig.add_subplot(gs[2,1])
    ax7 = fig.add_subplot(gs[2,2])
    ax8 = fig.add_subplot(gs[2,3])
    ax9 = fig.add_subplot(gs[3,0])
    ax10 = fig.add_subplot(gs[3,1])
    ax11 = fig.add_subplot(gs[3,2])
    ax12 = fig.add_subplot(gs[3,3])
    ax13 = fig.add_subplot(gs[4,0])
    ax14 = fig.add_subplot(gs[4,1])
    ax15 = fig.add_subplot(gs[4,2])
    ax16 = fig.add_subplot(gs[4,3])
    ax17 = fig.add_subplot(gs[5,0])
    ax18 = fig.add_subplot(gs[5,1])
    ax19 = fig.add_subplot(gs[5,2])
    ax20 = fig.add_subplot(gs[5,3])
    
    regions_ordered = ['all',19,3,1,5,9,4,7,17,13,6,14,2,15,8,10,11,16,12,18]
    for nax, ax in enumerate([ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9,ax10,ax11,ax12,ax13,ax14,ax15,ax16,ax17,ax18,ax19,ax20]):
        
        reg = regions_ordered[nax]

        for rcp in rcps:
            
            # Median and absolute median deviation
            reg_vol = ds_multigcm_vol[reg][rcp]
            reg_mass = reg_vol * pygem_prms.density_ice
            reg_area = ds_multigcm_area[reg][rcp]

            # Specific mass change rate
            reg_mb = (reg_mass[:,1:] - reg_mass[:,0:-1]) / reg_area[:,0:-1]
            reg_mb_med = np.median(reg_mb, axis=0)
            reg_mb_mad = median_abs_deviation(reg_mb, axis=0)
            
            ax.plot(years[0:-1], reg_mb_med / 1000, color=rcp_colordict[rcp], linestyle=rcp_styledict[rcp], 
                    linewidth=1, zorder=4, label=rcp)
            if rcp in rcps_plot_mad:
                ax.fill_between(years[0:-1], 
                                (reg_mb_med + 1.96*reg_mb_mad) / 1000, 
                                (reg_mb_med - 1.96*reg_mb_mad) / 1000, 
                                alpha=0.35, facecolor=rcp_colordict[rcp], label=None)
        
        if ax in [ax1,ax5,ax9,ax13,ax17]:
#            ax.set_ylabel('$\Delta$M/$\Delta$t (m w.e. yr$^{-1}$)')
            ax.set_ylabel('$\Delta$M/$\Delta$t\n(10$^{3}$ kg m$^{-2}$ yr$^{-1}$)')
        ax.set_xlim(startyear, endyear)
        ax.xaxis.set_major_locator(MultipleLocator(50))
        ax.xaxis.set_minor_locator(MultipleLocator(10))
        ax.plot(years,np.zeros(years.shape), color='k', linewidth=0.5)
        ax.set_ylim(-5.5,0.5)
        ax.yaxis.set_major_locator(MultipleLocator(1))
        ax.yaxis.set_minor_locator(MultipleLocator(0.5))
        if nax == 0:
            label_height=1.06
        else:
            label_height=1.14
        ax.text(1, label_height, rgi_reg_dict[reg], size=10, horizontalalignment='right', 
                verticalalignment='top', transform=ax.transAxes)
        ax.tick_params(axis='both', which='major', direction='inout', right=True)
        ax.tick_params(axis='both', which='minor', direction='in', right=True)
        
        if nax == 1:
            if 'rcp26' in rcps and 'ssp126' in rcps and len(rcps)==7:
                labels = ['SSP1-26', 'SSP2-45', 'SSP3-70', 'SSP5-85', 'RCP2.6', 'RCP4.5', 'RCP8.5']
            elif 'rcp26' in rcps and len(rcps) == 3:
                labels = ['RCP2.6', 'RCP4.5', 'RCP8.5']
            elif 'ssp126' in rcp and len(rcps) == 4:
                labels = ['SSP1-26', 'SSP2-45', 'SSP3-70', 'SSP5-85']
            ax.legend(loc=(-1.4,0.2), labels=labels, fontsize=10, ncol=2, columnspacing=0.5, labelspacing=0.25, 
                      handlelength=1, handletextpad=0.25, borderpad=0, frameon=False
                      )
    # Save figure
    if 'rcp26' in rcps and 'ssp126' in rcps:
        scenario_str = 'rcps_ssps'
    elif 'rcp26' in rcps:
        scenario_str = 'rcps'
    elif 'ssp126' in rcp:
        scenario_str = 'ssps'
    fig_fn = ('allregions_mb_' + str(startyear) + '-' + str(endyear) + '_multigcm' + 
              '-' + scenario_str + '.png')
    fig.set_size_inches(8.5,11)
    fig.savefig(fig_fp_multigcm + fig_fn, bbox_inches='tight', dpi=300)
    
    #%% ----- FIGURE: ANNUAL SEA-LEVEL RISE -----
    fig = plt.figure()
    gs = fig.add_gridspec(nrows=6,ncols=4,wspace=0.33,hspace=0.4)
    ax1 = fig.add_subplot(gs[0:2,0:2])
    ax2 = fig.add_subplot(gs[0,3])
    ax3 = fig.add_subplot(gs[1,2])
    ax4 = fig.add_subplot(gs[1,3])
    ax5 = fig.add_subplot(gs[2,0])
    ax6 = fig.add_subplot(gs[2,1])
    ax7 = fig.add_subplot(gs[2,2])
    ax8 = fig.add_subplot(gs[2,3])
    ax9 = fig.add_subplot(gs[3,0])
    ax10 = fig.add_subplot(gs[3,1])
    ax11 = fig.add_subplot(gs[3,2])
    ax12 = fig.add_subplot(gs[3,3])
    ax13 = fig.add_subplot(gs[4,0])
    ax14 = fig.add_subplot(gs[4,1])
    ax15 = fig.add_subplot(gs[4,2])
    ax16 = fig.add_subplot(gs[4,3])
    ax17 = fig.add_subplot(gs[5,0])
    ax18 = fig.add_subplot(gs[5,1])
    ax19 = fig.add_subplot(gs[5,2])
    ax20 = fig.add_subplot(gs[5,3])
    
    regions_ordered = ['all',19,3,1,5,9,4,7,17,13,6,14,2,15,8,10,11,16,12,18]
    for nax, ax in enumerate([ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9,ax10,ax11,ax12,ax13,ax14,ax15,ax16,ax17,ax18,ax19,ax20]):
        
        reg = regions_ordered[nax]
        
        for rcp in rcps:
            
            # Median and absolute median deviation
            reg_vol = ds_multigcm_vol[reg][rcp]
            reg_vol_bsl = ds_multigcm_vol_bsl[reg][rcp]
            reg_area = ds_multigcm_area[reg][rcp]

            # Sea-level change [mm SLE yr-1]
            #  - accounts for water from glaciers replacing the ice that is below sea level as well
            #    from Fabien's blog post: https://nbviewer.jupyter.org/gist/jmalles/ca70090812e6499b34a22a3a7a7a8f2a
            reg_slr = (-1*(((reg_vol[:,1:] - reg_vol[:,0:-1]) * pygem_prms.density_ice / pygem_prms.density_water - 
                       (reg_vol_bsl[:,1:] - reg_vol_bsl[:,0:-1])) / pygem_prms.area_ocean * 1000))
            #  - assume all mass loss contributes to sea level rise
#                reg_slr = -1*(reg_vol[:,1:] - reg_vol[:,0:-1]) * pygem_prms.density_ice / pygem_prms.density_water / pygem_prms.area_ocean * 1000
            
            reg_slr_med = np.median(reg_slr, axis=0)
            reg_slr_mad = median_abs_deviation(reg_slr, axis=0)
            
            ax.plot(years[0:-1], reg_slr_med, color=rcp_colordict[rcp], linestyle=rcp_styledict[rcp], 
                    linewidth=1, zorder=4, label=rcp)
            if rcp in rcps_plot_mad:
                ax.fill_between(years[0:-1], 
                                (reg_slr_med + 1.96*reg_slr_mad), 
                                (reg_slr_med - 1.96*reg_slr_mad), 
                                alpha=0.35, facecolor=rcp_colordict[rcp], label=None)
        
        if ax in [ax1,ax5,ax9,ax13,ax17]:
            ax.set_ylabel('$\Delta$M/$\Delta$t\n(mm SLE yr$^{-1}$)')
        ax.set_xlim(startyear, endyear)
        ax.xaxis.set_major_locator(MultipleLocator(50))
        ax.xaxis.set_minor_locator(MultipleLocator(10))
        ax.plot(years,np.zeros(years.shape), color='k', linewidth=0.5)
            
        if reg in [19, 3, 1]:
            ax.set_ylim(-0.01,0.9)
            ax.yaxis.set_major_locator(MultipleLocator(0.2))
            ax.yaxis.set_minor_locator(MultipleLocator(0.1))    
        elif reg in [5, 9, 4, 7]:
            ax.set_ylim(-0.01,0.45)
            ax.yaxis.set_major_locator(MultipleLocator(0.2))
            ax.yaxis.set_minor_locator(MultipleLocator(0.1)) 
        elif reg in [17, 13, 6, 14, 2, 13]:
            ax.set_ylim(-0.01,0.175)
            ax.yaxis.set_major_locator(MultipleLocator(0.1))
            ax.yaxis.set_minor_locator(MultipleLocator(0.05)) 
        elif reg in [8, 10, 11, 16, 12, 18]:
            ax.set_ylim(-0.005,0.025)
            ax.yaxis.set_major_locator(MultipleLocator(0.01))
            ax.yaxis.set_minor_locator(MultipleLocator(0.005)) 
            
        if nax == 0:
            label_height=1.06
        else:
            label_height=1.14
        ax.text(1, label_height, rgi_reg_dict[reg], size=10, horizontalalignment='right', 
                verticalalignment='top', transform=ax.transAxes)
        ax.tick_params(axis='both', which='major', direction='inout', right=True)
        ax.tick_params(axis='both', which='minor', direction='in', right=True)
        
        if nax == 1:
            if 'rcp26' in rcps and 'ssp126' in rcps and len(rcps)==7:
                labels = ['SSP1-26', 'SSP2-45', 'SSP3-70', 'SSP5-85', 'RCP2.6', 'RCP4.5', 'RCP8.5']
            elif 'rcp26' in rcps and len(rcps) == 3:
                labels = ['RCP2.6', 'RCP4.5', 'RCP8.5']
            elif 'ssp126' in rcp and len(rcps) == 4:
                labels = ['SSP1-26', 'SSP2-45', 'SSP3-70', 'SSP5-85']
            ax.legend(loc=(-1.4,0.2), labels=labels, fontsize=10, ncol=2, columnspacing=0.5, labelspacing=0.25, 
                      handlelength=1, handletextpad=0.25, borderpad=0, frameon=False
                      )
    # Save figure
    if 'rcp26' in rcps and 'ssp126' in rcps:
        scenario_str = 'rcps_ssps'
    elif 'rcp26' in rcps:
        scenario_str = 'rcps'
    elif 'ssp126' in rcp:
        scenario_str = 'ssps'
    fig_fn = ('allregions_SLR_' + str(startyear) + '-' + str(endyear) + '_multigcm' + 
              '-' + scenario_str + '.png')
    fig.set_size_inches(8.5,11)
    fig.savefig(fig_fp_multigcm + fig_fn, bbox_inches='tight', dpi=300)
    
    #%% ----- FIGURE: CUMULATIVE SEA-LEVEL RISE -----
    slr_cum_cns = ['Region', 'Scenario', 'med_mmSLE', 'mean_mmSLE', 'std_mmSLE', 'mad_mmSLE']
    slr_cum_df = pd.DataFrame(np.zeros((len(regions)*len(rcps),len(slr_cum_cns))), columns=slr_cum_cns)
    ncount = 0
    for nreg, reg in enumerate(regions):
        for rcp in rcps:
            # Median and absolute median deviation
            reg_vol = ds_multigcm_vol[reg][rcp]
            reg_vol_bsl = ds_multigcm_vol_bsl[reg][rcp]
            reg_area = ds_multigcm_area[reg][rcp]
            
            # Cumulative Sea-level change [mm SLE]
            #  - accounts for water from glaciers replacing the ice that is below sea level as well
            #    from Fabien's blog post: https://nbviewer.jupyter.org/gist/jmalles/ca70090812e6499b34a22a3a7a7a8f2a
            reg_slr = (-1*(((reg_vol[:,1:] - reg_vol[:,0:-1]) * pygem_prms.density_ice / pygem_prms.density_water - 
                       (reg_vol_bsl[:,1:] - reg_vol_bsl[:,0:-1])) / pygem_prms.area_ocean * 1000))
#            reg_slr = (-1*(((reg_vol[:,1:] - reg_vol[:,0:-1]) * pygem_prms.density_ice / pygem_prms.density_water) / pygem_prms.area_ocean * 1000))
            reg_slr_cum_raw = np.cumsum(reg_slr, axis=1)
            
            reg_slr_cum = reg_slr_cum_raw - reg_slr_cum_raw[:,normyear_idx][:,np.newaxis]

            reg_slr_cum_med = np.median(reg_slr_cum, axis=0)
            reg_slr_cum_mean = np.mean(reg_slr_cum, axis=0)
            reg_slr_cum_std = np.std(reg_slr_cum, axis=0)
            reg_slr_cum_mad = median_abs_deviation(reg_slr_cum, axis=0)
            
            slr_cum_df.loc[ncount,'Region'] = reg
            slr_cum_df.loc[ncount,'Scenario'] = rcp
            slr_cum_df.loc[ncount,'med_mmSLE'] = reg_slr_cum_med[-1]
            slr_cum_df.loc[ncount,'mean_mmSLE'] = reg_slr_cum_mean[-1]
            slr_cum_df.loc[ncount,'std_mmSLE'] = reg_slr_cum_std[-1]
            slr_cum_df.loc[ncount,'mad_mmSLE'] = reg_slr_cum_med[-1]
            
            ncount += 1
    slr_cum_df.to_csv(csv_fp + 'SLR_cum_2100_rel2015.csv')
            
#            print(reg, 'SLR (mm SLE):', rcp, np.round(reg_slr_cum_avg[-1],2), '+/-', np.round(reg_slr_cum_var[-1],2))
        
    fig = plt.figure()
    gs = fig.add_gridspec(nrows=6,ncols=4,wspace=0.33,hspace=0.4)
    ax1 = fig.add_subplot(gs[0:2,0:2])
    ax2 = fig.add_subplot(gs[0,3])
    ax3 = fig.add_subplot(gs[1,2])
    ax4 = fig.add_subplot(gs[1,3])
    ax5 = fig.add_subplot(gs[2,0])
    ax6 = fig.add_subplot(gs[2,1])
    ax7 = fig.add_subplot(gs[2,2])
    ax8 = fig.add_subplot(gs[2,3])
    ax9 = fig.add_subplot(gs[3,0])
    ax10 = fig.add_subplot(gs[3,1])
    ax11 = fig.add_subplot(gs[3,2])
    ax12 = fig.add_subplot(gs[3,3])
    ax13 = fig.add_subplot(gs[4,0])
    ax14 = fig.add_subplot(gs[4,1])
    ax15 = fig.add_subplot(gs[4,2])
    ax16 = fig.add_subplot(gs[4,3])
    ax17 = fig.add_subplot(gs[5,0])
    ax18 = fig.add_subplot(gs[5,1])
    ax19 = fig.add_subplot(gs[5,2])
    ax20 = fig.add_subplot(gs[5,3])
    
    regions_ordered = ['all',19,3,1,5,9,4,7,17,13,6,14,2,15,8,10,11,16,12,18]
    for nax, ax in enumerate([ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9,ax10,ax11,ax12,ax13,ax14,ax15,ax16,ax17,ax18,ax19,ax20]):
        
        reg = regions_ordered[nax]
        
        for rcp in rcps:
            
#            reg_vol_med[normyear_idx]
#            print('NORMALIZE ALL TO BE 0 AT NORMYEAR!')
            
            # Median and absolute median deviation
            reg_vol = ds_multigcm_vol[reg][rcp]
            reg_vol_bsl = ds_multigcm_vol_bsl[reg][rcp]
            reg_area = ds_multigcm_area[reg][rcp]

            # Cumulative Sea-level change [mm SLE]
            #  - accounts for water from glaciers replacing the ice that is below sea level as well
            #    from Fabien's blog post: https://nbviewer.jupyter.org/gist/jmalles/ca70090812e6499b34a22a3a7a7a8f2a
            reg_slr = (-1*(((reg_vol[:,1:] - reg_vol[:,0:-1]) * pygem_prms.density_ice / pygem_prms.density_water - 
                       (reg_vol_bsl[:,1:] - reg_vol_bsl[:,0:-1])) / pygem_prms.area_ocean * 1000))
#            reg_slr = (-1*(((reg_vol[:,1:] - reg_vol[:,0:-1]) * pygem_prms.density_ice / pygem_prms.density_water) / pygem_prms.area_ocean * 1000))
            reg_slr_cum_raw = np.cumsum(reg_slr, axis=1)
            
            reg_slr_cum = reg_slr_cum_raw - reg_slr_cum_raw[:,normyear_idx][:,np.newaxis]

            reg_slr_cum_avg = np.median(reg_slr_cum, axis=0)
#            reg_slr_cum_avg = np.mean(reg_slr_cum, axis=0)
            reg_slr_cum_var = np.std(reg_slr_cum, axis=0)
#            reg_slr_cum_var = median_abs_deviation(reg_slr_cum, axis=0)
            

            ax.plot(years[0:-1], reg_slr_cum_avg, color=rcp_colordict[rcp], linestyle=rcp_styledict[rcp], 
                    linewidth=1, zorder=4, label=rcp)
            if rcp in rcps_plot_mad:
#                ax.fill_between(years[0:-1], 
#                                (reg_slr_cum_avg + 1.96*reg_slr_cum_var), 
#                                (reg_slr_cum_avg - 1.96*reg_slr_cum_var), 
#                                alpha=0.35, facecolor=rcp_colordict[rcp], label=None)
                ax.fill_between(years[0:-1], 
                                (reg_slr_cum_avg + reg_slr_cum_var), 
                                (reg_slr_cum_avg - reg_slr_cum_var), 
                                alpha=0.35, facecolor=rcp_colordict[rcp], label=None)
        
            if reg in ['all']:
                print('global glacier SLR (mm SLE):', rcp, np.round(reg_slr_cum_avg[-1],2), '+/-', np.round(reg_slr_cum_var[-1],2))
        
        if ax in [ax1,ax5,ax9,ax13,ax17]:
#            ax.set_ylabel('$\Delta$M (mm SLE)')
            ax.set_ylabel('SLR (mm SLE)')
        ax.set_xlim(startyear, endyear)
        ax.xaxis.set_major_locator(MultipleLocator(50))
        ax.xaxis.set_minor_locator(MultipleLocator(10))
        ax.plot(years,np.zeros(years.shape), color='k', linewidth=0.5)
            
        if reg in ['all']:
            ax.set_ylim(0,250)
            ax.yaxis.set_major_locator(MultipleLocator(50))
            ax.yaxis.set_minor_locator(MultipleLocator(10))
        if reg in [19, 3, 1]:
            ax.set_ylim(0,42)
            ax.yaxis.set_major_locator(MultipleLocator(10))
            ax.yaxis.set_minor_locator(MultipleLocator(5))    
        elif reg in [5, 9, 4, 7]:
            ax.set_ylim(0,27)
            ax.yaxis.set_major_locator(MultipleLocator(10))
            ax.yaxis.set_minor_locator(MultipleLocator(5)) 
        elif reg in [17, 13, 6, 14]:
            ax.set_ylim(0,11)
            ax.yaxis.set_major_locator(MultipleLocator(5))
            ax.yaxis.set_minor_locator(MultipleLocator(1)) 
        elif reg in [15, 2]:
            ax.set_ylim(0,3)
            ax.yaxis.set_major_locator(MultipleLocator(1))
            ax.yaxis.set_minor_locator(MultipleLocator(0.2)) 
        elif reg in [8]:
            ax.set_ylim(0,0.8)
            ax.yaxis.set_major_locator(MultipleLocator(0.2))
            ax.yaxis.set_minor_locator(MultipleLocator(0.1)) 
        elif reg in [10, 11, 16, 12, 18]:
            ax.set_ylim(0,0.45)
            ax.yaxis.set_major_locator(MultipleLocator(0.2))
            ax.yaxis.set_minor_locator(MultipleLocator(0.1)) 
            
        if nax == 0:
            label_height=1.06
        else:
            label_height=1.14
        ax.text(1, label_height, rgi_reg_dict[reg], size=10, horizontalalignment='right', 
                verticalalignment='top', transform=ax.transAxes)
        ax.tick_params(axis='both', which='major', direction='inout', right=True)
        ax.tick_params(axis='both', which='minor', direction='in', right=True)
        
        if nax == 1:
            if 'rcp26' in rcps and 'ssp126' in rcps and len(rcps)==7:
                labels = ['SSP1-26', 'SSP2-45', 'SSP3-70', 'SSP5-85', 'RCP2.6', 'RCP4.5', 'RCP8.5']
            elif 'rcp26' in rcps and len(rcps) == 3:
                labels = ['RCP2.6', 'RCP4.5', 'RCP8.5']
            elif 'ssp126' in rcp and len(rcps) == 4:
                labels = ['SSP1-26', 'SSP2-45', 'SSP3-70', 'SSP5-85']
            ax.legend(loc=(-1.4,0.2), labels=labels, fontsize=10, ncol=2, columnspacing=0.5, labelspacing=0.25, 
                      handlelength=1, handletextpad=0.25, borderpad=0, frameon=False
                      )
    # Save figure
    if 'rcp26' in rcps and 'ssp126' in rcps:
        scenario_str = 'rcps_ssps'
    elif 'rcp26' in rcps:
        scenario_str = 'rcps'
    elif 'ssp126' in rcp:
        scenario_str = 'ssps'
    fig_fn = ('allregions_SLR-cum_' + str(startyear) + '-' + str(endyear) + '_multigcm' + 
              '-' + scenario_str + '-medstd.png')
    fig.set_size_inches(8.5,11)
    fig.savefig(fig_fp_multigcm + fig_fn, bbox_inches='tight', dpi=300)
    
    
    #%% ----- FIGURE: CUMULATIVE SEA-LEVEL RISE w BOX AND WHISKERS -----
    fig = plt.figure()
    gs = fig.add_gridspec(nrows=6,ncols=8,wspace=0.66,hspace=0.4)
    ax1 = fig.add_subplot(gs[0:2,0:3])
    ax1b = fig.add_subplot(gs[0:2,2:4])
    ax2 = fig.add_subplot(gs[0,6:])
    ax3 = fig.add_subplot(gs[1,4:6])
    ax4 = fig.add_subplot(gs[1,6:])
    ax5 = fig.add_subplot(gs[2,0:2])
    ax6 = fig.add_subplot(gs[2,2:4])
    ax7 = fig.add_subplot(gs[2,4:6])
    ax8 = fig.add_subplot(gs[2,6:])
    ax9 = fig.add_subplot(gs[3,0:2])
    ax10 = fig.add_subplot(gs[3,2:4])
    ax11 = fig.add_subplot(gs[3,4:6])
    ax12 = fig.add_subplot(gs[3,6:])
    ax13 = fig.add_subplot(gs[4,0:2])
    ax14 = fig.add_subplot(gs[4,2:4])
    ax15 = fig.add_subplot(gs[4,4:6])
    ax16 = fig.add_subplot(gs[4,6:])
    ax17 = fig.add_subplot(gs[5,0:2])
    ax18 = fig.add_subplot(gs[5,2:4])
    ax19 = fig.add_subplot(gs[5,4:6])
    ax20 = fig.add_subplot(gs[5,6:])
    
    data_boxplot = []
    regions_ordered = ['all',19,3,1,5,9,4,7,17,13,6,14,2,15,8,10,11,16,12,18]
    for nax, ax in enumerate([ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9,ax10,ax11,ax12,ax13,ax14,ax15,ax16,ax17,ax18,ax19,ax20]):
        
        reg = regions_ordered[nax]
        
        for rcp in rcps:
            
            # Median and absolute median deviation
            reg_vol = ds_multigcm_vol[reg][rcp]
            reg_vol_bsl = ds_multigcm_vol_bsl[reg][rcp]
            reg_area = ds_multigcm_area[reg][rcp]

            # Cumulative Sea-level change [mm SLE]
            #  - accounts for water from glaciers replacing the ice that is below sea level as well
            #    from Fabien's blog post: https://nbviewer.jupyter.org/gist/jmalles/ca70090812e6499b34a22a3a7a7a8f2a
            reg_slr = (-1*(((reg_vol[:,1:] - reg_vol[:,0:-1]) * pygem_prms.density_ice / pygem_prms.density_water - 
                       (reg_vol_bsl[:,1:] - reg_vol_bsl[:,0:-1])) / pygem_prms.area_ocean * 1000))
#            reg_slr = (-1*(((reg_vol[:,1:] - reg_vol[:,0:-1]) * pygem_prms.density_ice / pygem_prms.density_water) / pygem_prms.area_ocean * 1000))
            reg_slr_cum_raw = np.cumsum(reg_slr, axis=1)
            
            reg_slr_cum = reg_slr_cum_raw - reg_slr_cum_raw[:,normyear_idx][:,np.newaxis]

            reg_slr_cum_avg = np.median(reg_slr_cum, axis=0)
            reg_slr_cum_var = np.std(reg_slr_cum, axis=0)
            
            ax.plot(years[0:-1], reg_slr_cum_avg, color=rcp_colordict[rcp], linestyle=rcp_styledict[rcp], 
                    linewidth=1, zorder=4, label=rcp)
            if rcp in rcps_plot_mad:
#                ax.fill_between(years[0:-1], 
#                                (reg_slr_cum_avg + 1.96*reg_slr_cum_var), 
#                                (reg_slr_cum_avg - 1.96*reg_slr_cum_var), 
#                                alpha=0.35, facecolor=rcp_colordict[rcp], label=None)
                ax.fill_between(years[0:-1], 
                                (reg_slr_cum_avg + reg_slr_cum_var), 
                                (reg_slr_cum_avg - reg_slr_cum_var), 
                                alpha=0.35, facecolor=rcp_colordict[rcp], label=None)
            
            # Aggregate boxplot data
            if reg in ['all']:
                data_boxplot.append(reg_slr_cum[:,-1])
        
        if ax in [ax1,ax5,ax9,ax13,ax17]:
#            ax.set_ylabel('$\Delta$M (mm SLE)')
            ax.set_ylabel('SLR (mm SLE)')
        ax.set_xlim(startyear, endyear)
        ax.xaxis.set_major_locator(MultipleLocator(50))
        ax.xaxis.set_minor_locator(MultipleLocator(10))
        ax.plot(years,np.zeros(years.shape), color='k', linewidth=0.5)
            
        if reg in ['all']:
            ax.set_ylim(0,250)
            ax.yaxis.set_major_locator(MultipleLocator(50))
            ax.yaxis.set_minor_locator(MultipleLocator(10))
        if reg in [19, 3, 1]:
            ax.set_ylim(0,42)
            ax.yaxis.set_major_locator(MultipleLocator(10))
            ax.yaxis.set_minor_locator(MultipleLocator(5))    
        elif reg in [5, 9, 4, 7]:
            ax.set_ylim(0,27)
            ax.yaxis.set_major_locator(MultipleLocator(10))
            ax.yaxis.set_minor_locator(MultipleLocator(5)) 
        elif reg in [17, 13, 6, 14]:
            ax.set_ylim(0,11)
            ax.yaxis.set_major_locator(MultipleLocator(5))
            ax.yaxis.set_minor_locator(MultipleLocator(1)) 
        elif reg in [15, 2]:
            ax.set_ylim(0,3)
            ax.yaxis.set_major_locator(MultipleLocator(1))
            ax.yaxis.set_minor_locator(MultipleLocator(0.2)) 
        elif reg in [8]:
            ax.set_ylim(0,0.8)
            ax.yaxis.set_major_locator(MultipleLocator(0.2))
            ax.yaxis.set_minor_locator(MultipleLocator(0.1)) 
        elif reg in [10, 11, 16, 12, 18]:
            ax.set_ylim(0,0.45)
            ax.yaxis.set_major_locator(MultipleLocator(0.2))
            ax.yaxis.set_minor_locator(MultipleLocator(0.1)) 
            
        if nax == 0:
            label_height=1.06
        else:
            label_height=1.14
        ax.text(1, label_height, rgi_reg_dict[reg], size=10, horizontalalignment='right', 
                verticalalignment='top', transform=ax.transAxes)
        ax.tick_params(axis='both', which='major', direction='inout', right=True)
        ax.tick_params(axis='both', which='minor', direction='in', right=True)
        
        if nax == 1:
            if 'rcp26' in rcps and 'ssp126' in rcps and len(rcps)==7:
                labels = ['SSP1-26', 'SSP2-45', 'SSP3-70', 'SSP5-85', 'RCP2.6', 'RCP4.5', 'RCP8.5']
            elif 'rcp26' in rcps and len(rcps) == 3:
                labels = ['RCP2.6', 'RCP4.5', 'RCP8.5']
            elif 'ssp126' in rcp and len(rcps) == 4:
                labels = ['SSP1-26', 'SSP2-45', 'SSP3-70', 'SSP5-85']
            ax.legend(loc=(-1.32,0.2), labels=labels, fontsize=10, ncol=2, columnspacing=0.5, labelspacing=0.25, 
                      handlelength=1, handletextpad=0.25, borderpad=0, frameon=False
                      )
    
#        bp_lc = []
#        for rcp in rcps:
#            bp_lc.append(rcp_colordict[rcp])
    bp = ax1b.boxplot(data_boxplot)
    ax1b.set_ylim(0,250)
    for nbox, box in enumerate(bp['boxes']):
        rcp = rcps[nbox]
        # change outline color
        box.set(color=rcp_colordict[rcp], linestyle=rcp_styledict[rcp], linewidth=1)
    for nitem, item in enumerate(bp['medians']):
        rcp = rcps[nitem]
        # change outline color
        item.set(color=rcp_colordict[rcp], linewidth=1)
    for nitem, item in enumerate(bp['whiskers']):
        rcp = rcps[int(np.floor(nitem/2))]
        # change outline color
        item.set(color=rcp_colordict[rcp], linestyle=rcp_styledict[rcp], linewidth=1)
    for nitem, item in enumerate(bp['caps']):
        rcp = rcps[int(np.floor(nitem/2))]
        # change outline color
        item.set(color=rcp_colordict[rcp], linewidth=1)
    # turn off axes
    ax1b.get_yaxis().set_visible(False)
    ax1b.get_xaxis().set_visible(False)
    ax1b.axis('off')
    ax1b.set_xlim(-5,8)
            
    # Save figure
    if 'rcp26' in rcps and 'ssp126' in rcps:
        scenario_str = 'rcps_ssps'
    elif 'rcp26' in rcps:
        scenario_str = 'rcps'
    elif 'ssp126' in rcp:
        scenario_str = 'ssps'
    fig_fn = ('allregions_SLR-cum_' + str(startyear) + '-' + str(endyear) + '_multigcm' + 
              '-' + scenario_str + '-medstd-BoxWhisker.png')
    fig.set_size_inches(8.5,11)
    fig.savefig(fig_fp_multigcm + fig_fn, bbox_inches='tight', dpi=300)
    
    
#%% ----- FIGURE: ANNUAL RUNOFF -----        
    fig = plt.figure()
    gs = fig.add_gridspec(nrows=6,ncols=4,wspace=0.33,hspace=0.4)
    ax1 = fig.add_subplot(gs[0:2,0:2])
    ax2 = fig.add_subplot(gs[0,3])
    ax3 = fig.add_subplot(gs[1,2])
    ax4 = fig.add_subplot(gs[1,3])
    ax5 = fig.add_subplot(gs[2,0])
    ax6 = fig.add_subplot(gs[2,1])
    ax7 = fig.add_subplot(gs[2,2])
    ax8 = fig.add_subplot(gs[2,3])
    ax9 = fig.add_subplot(gs[3,0])
    ax10 = fig.add_subplot(gs[3,1])
    ax11 = fig.add_subplot(gs[3,2])
    ax12 = fig.add_subplot(gs[3,3])
    ax13 = fig.add_subplot(gs[4,0])
    ax14 = fig.add_subplot(gs[4,1])
    ax15 = fig.add_subplot(gs[4,2])
    ax16 = fig.add_subplot(gs[4,3])
    ax17 = fig.add_subplot(gs[5,0])
    ax18 = fig.add_subplot(gs[5,1])
    ax19 = fig.add_subplot(gs[5,2])
    ax20 = fig.add_subplot(gs[5,3])
    
    regions_ordered = ['all',19,3,1,5,9,4,7,17,13,6,14,2,15,8,10,11,16,12,18]
    for nax, ax in enumerate([ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9,ax10,ax11,ax12,ax13,ax14,ax15,ax16,ax17,ax18,ax19,ax20]):
        
        reg = regions_ordered[nax]
        
        reg_runoff_annual_present_rcps_list = []
        for rcp in rcps:

            # Median and standard deviation
            reg_runoff_monthly = ds_multigcm_runoff[reg][rcp]
            reg_runoff_annual = np.zeros((reg_runoff_monthly.shape[0],101))
            for nrow in np.arange(0,reg_runoff_monthly.shape[0]):
                reg_runoff_annual[nrow,:] = reg_runoff_monthly[nrow,:].reshape(-1,12).sum(1)

            reg_runoff_annual_avg = np.median(reg_runoff_annual, axis=0)
            reg_runoff_annual_std = np.std(reg_runoff_annual, axis=0)

            reg_runoff_annual_present = reg_runoff_annual[:,0:16].mean(1)
            reg_runoff_norm = reg_runoff_annual / reg_runoff_annual_present[:,np.newaxis]
            reg_runoff_norm_avg = np.median(reg_runoff_norm, axis=0)
            reg_runoff_norm_var = np.std(reg_runoff_norm, axis=0)
            
            reg_runoff_annual_present_rcps_list.append(reg_runoff_annual_present.mean())
            
            # Peakwater
            peakwater_yr, peakwater_chg, runoff_chg = peakwater(reg_runoff_annual_avg, years[0:-1], 11)
            
            ax.plot(years[0:-1], reg_runoff_norm_avg, color=rcp_colordict[rcp], linestyle=rcp_styledict[rcp], 
                    linewidth=1, zorder=4, label=rcp)
            
            if 'ssp' in rcp:
                peakwater_yr_idx = np.where(peakwater_yr == years[0:-1])[0][0]
                ax.vlines(peakwater_yr, 0, reg_runoff_norm_avg[peakwater_yr_idx], 
                          color=rcp_colordict[rcp], linestyle=rcp_styledict[rcp], 
                          linewidth=1, zorder=5)
                
            if rcp in rcps_plot_mad:
#                ax.fill_between(years[0:-1], 
#                                (reg_slr_cum_avg + 1.96*reg_slr_cum_var), 
#                                (reg_slr_cum_avg - 1.96*reg_slr_cum_var), 
#                                alpha=0.35, facecolor=rcp_colordict[rcp], label=None)
                ax.fill_between(years[0:-1], 
                                (reg_runoff_norm_avg + reg_runoff_norm_var), 
                                (reg_runoff_norm_avg - reg_runoff_norm_var), 
                                alpha=0.35, facecolor=rcp_colordict[rcp], label=None)
        
        runoff_gta_str = str(np.round(np.mean(reg_runoff_annual_present_rcps_list) * pygem_prms.density_water / 1e12,1))
        print(reg, runoff_gta_str, 'Gt/yr')
        ax.text(0.97, 0.97, runoff_gta_str, size=10, horizontalalignment='right', 
                verticalalignment='top', transform=ax.transAxes, zorder=6)

        if ax in [ax1,ax5,ax9,ax13,ax17]:
            ax.set_ylabel('Runoff (-)')
        ax.set_xlim(startyear, endyear)
        ax.xaxis.set_major_locator(MultipleLocator(40))
        ax.xaxis.set_minor_locator(MultipleLocator(10))

        if reg in [19, 3, 9, 7, 5]:
            ax.set_ylim(0,8)
            ax.yaxis.set_major_locator(MultipleLocator(5))
            ax.yaxis.set_minor_locator(MultipleLocator(0.5))    
        else:
            ax.set_ylim(0,2.2)
            ax.yaxis.set_major_locator(MultipleLocator(1))
            ax.yaxis.set_minor_locator(MultipleLocator(0.2))
            
        if nax == 0:
            label_height=1.06
        else:
            label_height=1.14
        ax.text(1, label_height, rgi_reg_dict[reg], size=10, horizontalalignment='right', 
                verticalalignment='top', transform=ax.transAxes)
        ax.tick_params(axis='both', which='major', direction='inout', right=True)
        ax.tick_params(axis='both', which='minor', direction='in', right=True)
        
        if nax == 1:
            if 'rcp26' in rcps and 'ssp126' in rcps and len(rcps)==7:
                labels = ['SSP1-26', 'SSP2-45', 'SSP3-70', 'SSP5-85', 'RCP2.6', 'RCP4.5', 'RCP8.5']
            elif 'rcp26' in rcps and len(rcps) == 3:
                labels = ['RCP2.6', 'RCP4.5', 'RCP8.5']
            elif 'ssp126' in rcp and len(rcps) == 4:
                labels = ['SSP1-26', 'SSP2-45', 'SSP3-70', 'SSP5-85']
            ax.legend(loc=(-1.4,0.2), labels=labels, fontsize=10, ncol=2, columnspacing=0.5, labelspacing=0.25, 
                      handlelength=1, handletextpad=0.25, borderpad=0, frameon=False
                      )
    # Save figure
    if 'rcp26' in rcps and 'ssp126' in rcps:
        scenario_str = 'rcps_ssps'
    elif 'rcp26' in rcps:
        scenario_str = 'rcps'
    elif 'ssp126' in rcp:
        scenario_str = 'ssps'
    fig_fn = ('allregions_runoff_' + str(startyear) + '-' + str(endyear) + '_multigcm' + 
              '-' + scenario_str + '-medstd.png')
    fig.set_size_inches(8.5,11)
    fig.savefig(fig_fp_multigcm + fig_fn, bbox_inches='tight', dpi=300)


#%% ===== MULTI-GCM PLOTS OF RUNOFF BY WATERSHED =====
if option_multigcm_plots_ws:
    
    fig_fp = netcdf_fp_cmip5 + '/../analysis/figures/'
    csv_fp = netcdf_fp_cmip5 + '/../analysis/csv/'
    pickle_fp = fig_fp + '../pickle/'
    
    startyear = 2015
    endyear = 2100
    normyear = 2015
    years = np.arange(2000,2101+1)
    
    gcm_names_rcps = ['CanESM2', 'CCSM4', 'CNRM-CM5', 'CSIRO-Mk3-6-0', 'GFDL-CM3', 
                      'GFDL-ESM2M', 'GISS-E2-R', 'IPSL-CM5A-LR', 'MPI-ESM-LR', 'NorESM1-M']
    gcm_names_ssps = ['BCC-CSM2-MR', 'CESM2', 'CESM2-WACCM', 'EC-Earth3', 'EC-Earth3-Veg', 'FGOALS-f3-L', 
                      'GFDL-ESM4', 'INM-CM4-8', 'INM-CM5-0', 'MPI-ESM1-2-HR', 'MRI-ESM2-0', 'NorESM2-MM']
    rcps = ['ssp126', 'ssp245', 'ssp370', 'ssp585', 'rcp26', 'rcp45', 'rcp85']
    
#    regions = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
    
    # Watersheds
    ws_reg = 'AK'
#    ws_reg = 'HMA'
#    ws_reg = 'Euro_Asia'
#    ws_reg = 'SA'
    # North America / Alaska watersheds
    if ws_reg == 'AK':
        watersheds = ['Yukon', 'Kuskowin', 'Nushagak', 'GHAASBasin325', 'Copper', 'Alsek', 'Mackenzie', 
                      'Taku', 'Stikine', 'Nass','Skeena', 'Fraser', 'Columbia', 'Nelson']
        regions = [1,2]
    # Europe and North Asia watersheds
    elif ws_reg == 'Euro_Asia':
        watersheds = ['Danube', 'Po', 'Rhone', 'Rhine', 'Glama', 'Drammenselva', 'Lulealven', 'Kalix', 'Ob', 'Thjorsa', 'Olfusa']
        regions = [6,8,10,11]
    # High Mountain Asia watersheds
    elif ws_reg == 'HMA':
        watersheds = ['Brahmaputra', 'Ganges', 'Indus', 'Amu_Darya', 'Syr_Darya', 'Ili', 'Tarim', 'Inner_Tibetan_Plateau', 
                      'Inner_Tibetan_Plateau_extended', 'Mekong', 'Salween',  'Yangtze', 'Yellow', 'Irrawaddy']
        regions = [13,14,15]
    # South America watersheds
    elif ws_reg == 'SA':
        watersheds = ['Amazon', 'Santa', 'Baker (Chile)', 'Colorado (Argentina)', 'Rapel', 'Bio Bio']
        regions = [16,17]
        print('\n\nNEED TO MANUALLY ADD MORE WATERSHEDS!\n\n')
    watersheds_dict_correct = {'GHAASBasin325':'Susitna',
                               'Amu_Darya':'Amu Darya', 
                               'Syr_Darya':'Syr Darya', 
                               'Inner_Tibetan_Plateau':'Inner TP', 
                               'Inner_Tibetan_Plateau_extended':'Inner TP ext'}
    
#    rcps_plot_mad = ['rcp26', 'rcp45', 'rcp85', 'ssp126', 'ssp245', 'ssp370', 'ssp585']
    rcps_plot_mad = ['ssp126', 'ssp585']
    
    fig_fp_multigcm = fig_fp + 'multi_gcm_ws/'
    if not os.path.exists(fig_fp_multigcm):
        os.makedirs(fig_fp_multigcm, exist_ok=True)
    
    # Set up processing
    ws_runoff_all = {}
    for ws in watersheds:
        ws_runoff_all[ws] = {}
        for rcp in rcps:
            if 'rcp' in rcp:
                gcm_names = gcm_names_rcps
            elif 'ssp' in rcp:
                gcm_names = gcm_names_ssps
            ws_runoff_all[ws][rcp] = {}
            for gcm_name in gcm_names:
                ws_runoff_all[ws][rcp][gcm_name] = None
            
    for reg in regions:
        
        fnfull_watersheds = pickle_fp + str(reg).zfill(2) + '/R' + str(reg) + '_unique_watersheds.pkl'
        with open(fnfull_watersheds, 'rb') as f:
            watersheds_list = list(pickle.load(f))
        
        for rcp in rcps:
            if 'rcp' in rcp:
                gcm_names = gcm_names_rcps
            elif 'ssp' in rcp:
                gcm_names = gcm_names_ssps
                
            for gcm_name in gcm_names:
        
                # ----- GCM/RCP PICKLE FILEPATHS AND FILENAMES -----
                pickle_fp_watershed =  pickle_fp + str(reg).zfill(2) + '/watersheds/' + gcm_name + '/' + rcp + '/'
                # Region string prefix
                watershed_rcp_gcm_str = 'R' + str(reg) + '_watersheds_' + rcp + '_' + gcm_name
                
                # Filenames
                fn_watershed_runoff_monthly_moving = watershed_rcp_gcm_str + '_runoff_monthly_moving.pkl'
                fn_watershed_runoff_monthly_fixed = watershed_rcp_gcm_str + '_runoff_monthly_fixed.pkl'       

                # Runoff: moving-gauged
                with open(pickle_fp_watershed + fn_watershed_runoff_monthly_moving, 'rb') as f:
                    ws_runoff_monthly_moving = pickle.load(f)
                # Runoff: fixed-gauged
                with open(pickle_fp_watershed + fn_watershed_runoff_monthly_fixed, 'rb') as f:
                    ws_runoff_monthly_fixed = pickle.load(f)
                    
                for ws in watersheds:
                    if ws in watersheds_list:
                        ws_idx = watersheds_list.index(ws)
                        if ws_runoff_all[ws][rcp][gcm_name] is None:
                            ws_runoff_all[ws][rcp][gcm_name] = ws_runoff_monthly_fixed[ws_idx,:]
                        else:
                            ws_runoff_all[ws][rcp][gcm_name] = ws_runoff_all[ws][rcp][gcm_name] + ws_runoff_monthly_fixed[ws_idx,:]

    #%% MULTI-GCM STATISTICS
    ds_multigcm_runoff = {}
    for ws in watersheds:
        ds_multigcm_runoff[ws] = {}
        for rcp in rcps: 
            
            if 'rcp' in rcp:
                gcm_names = gcm_names_rcps
            elif 'ssp' in rcp:
                gcm_names = gcm_names_ssps
                
            for ngcm, gcm_name in enumerate(gcm_names):
                
                ws_runoff_gcm = ws_runoff_all[ws][rcp][gcm_name]
    
                if ngcm == 0:
                    ws_runoff_gcm_all = ws_runoff_gcm
                else:
                    ws_runoff_gcm_all = np.vstack((ws_runoff_gcm_all, ws_runoff_gcm))

            ds_multigcm_runoff[ws][rcp] = ws_runoff_gcm_all

    #%% ----- FIGURE: ANNUAL RUNOFF -----
    # Peakwater
    pw_cns = ['Watershed', 'Scenario', 'runoff_Gtyr_ref', 'peakwater_yr', 'peakwater_chg_perc', '2100_chg_perc']
    pw_df = pd.DataFrame(np.zeros((len(regions)*len(rcps),len(pw_cns))), columns=pw_cns)
    ncount = 0
    
    for nws, ws in enumerate(watersheds):
        for rcp in rcps:

            # Median and standard deviation
            ws_runoff_monthly = ds_multigcm_runoff[ws][rcp]
            ws_runoff_annual = np.zeros((ws_runoff_monthly.shape[0],101))
            for nrow_ws in np.arange(0,ws_runoff_monthly.shape[0]):
                ws_runoff_annual[nrow_ws,:] = ws_runoff_monthly[nrow_ws,:].reshape(-1,12).sum(1)

            ws_runoff_annual_avg = np.median(ws_runoff_annual, axis=0)
            ws_runoff_annual_std = np.std(ws_runoff_annual, axis=0)
            ws_runoff_annual_present = ws_runoff_annual[:,0:16].mean(1)
            
            # Peakwater
            peakwater_yr, peakwater_chg, runoff_chg = peakwater(ws_runoff_annual_avg, years[0:-1], 11)
            
            pw_df.loc[ncount,'Watershed'] = ws
            pw_df.loc[ncount,'Scenario'] = rcp
            pw_df.loc[ncount,'runoff_Gtyr_ref'] = ws_runoff_annual_present.mean()
            pw_df.loc[ncount,'peakwater_yr'] = peakwater_yr
            pw_df.loc[ncount,'peakwater_chg_perc'] = peakwater_chg
            pw_df.loc[ncount,'2100_chg_perc'] = runoff_chg
            
            ncount += 1
    pw_df.to_csv(csv_fp + 'Runoff_' + ws_reg + '_watersheds_' + str(startyear) + '-' + str(endyear) + '_multigcm.csv', index=False)
            
    #%%
    ncols = 4
    nrows = int(np.ceil(len(watersheds)/ncols))
    fig, ax = plt.subplots(nrows,ncols,gridspec_kw = {'wspace':0.3, 'hspace':0.3})
    
    nrow, ncol = 0,0
    for nws, ws in enumerate(watersheds):
        
        print(ws)
        
        ws_runoff_annual_present_rcps_list = []
        for rcp in rcps:

            # Median and standard deviation
            ws_runoff_monthly = ds_multigcm_runoff[ws][rcp]
            ws_runoff_annual = np.zeros((ws_runoff_monthly.shape[0],101))
            for nrow_ws in np.arange(0,ws_runoff_monthly.shape[0]):
                ws_runoff_annual[nrow_ws,:] = ws_runoff_monthly[nrow_ws,:].reshape(-1,12).sum(1)

            ws_runoff_annual_avg = np.median(ws_runoff_annual, axis=0)
            ws_runoff_annual_std = np.std(ws_runoff_annual, axis=0)

            ws_runoff_annual_present = ws_runoff_annual[:,0:16].mean(1)
            ws_runoff_norm = ws_runoff_annual / ws_runoff_annual_present[:,np.newaxis]
            ws_runoff_norm_avg = np.median(ws_runoff_norm, axis=0)
            ws_runoff_norm_var = np.std(ws_runoff_norm, axis=0)
            
            ws_runoff_annual_present_rcps_list.append(ws_runoff_annual_present.mean())
            
            # Peakwater
            peakwater_yr, peakwater_chg, runoff_chg = peakwater(ws_runoff_annual_avg, years[0:-1], 11)
            
            ax[nrow,ncol].plot(years[0:-1], ws_runoff_norm_avg, color=rcp_colordict[rcp], linestyle=rcp_styledict[rcp], 
                                linewidth=1, zorder=4, label=rcp)
            
            if 'ssp' in rcp:
                peakwater_yr_idx = np.where(peakwater_yr == years[0:-1])[0][0]
                ax[nrow,ncol].vlines(peakwater_yr, 0, ws_runoff_norm_avg[peakwater_yr_idx], 
                                     color=rcp_colordict[rcp], linestyle='--', 
                                     linewidth=1, zorder=5)
                
            if rcp in rcps_plot_mad:
                ax[nrow,ncol].fill_between(years[0:-1], 
                                           (ws_runoff_norm_avg + ws_runoff_norm_var), 
                                           (ws_runoff_norm_avg - ws_runoff_norm_var), 
                                           alpha=0.35, facecolor=rcp_colordict[rcp], label=None)
        
        runoff_gta_str = str(np.round(np.mean(ws_runoff_annual_present_rcps_list) * pygem_prms.density_water / 1e12,1))
        print(ws, runoff_gta_str, 'Gt/yr')
        ax[nrow,ncol].text(0.97, 0.97, runoff_gta_str, size=10, horizontalalignment='right', 
                           verticalalignment='top', transform=ax[nrow,ncol].transAxes, zorder=6)

        if ncol == 0:
            ax[nrow,ncol].set_ylabel('Runoff (-)')
        ax[nrow,ncol].set_xlim(startyear, endyear)
        ax[nrow,ncol].xaxis.set_major_locator(MultipleLocator(40))
        ax[nrow,ncol].xaxis.set_minor_locator(MultipleLocator(10))


        ax[nrow,ncol].set_ylim(0,3)
        ax[nrow,ncol].yaxis.set_major_locator(MultipleLocator(0.5))
        ax[nrow,ncol].yaxis.set_minor_locator(MultipleLocator(0.1))
            
        if ws in watersheds_dict_correct:
            ws_label = watersheds_dict_correct[ws]
        else:
            ws_label = ws
        ax[nrow,ncol].text(1, 1.1, ws_label, size=10, horizontalalignment='right', 
                           verticalalignment='top', transform=ax[nrow,ncol].transAxes)
        ax[nrow,ncol].tick_params(axis='both', which='major', direction='inout', right=True)
        ax[nrow,ncol].tick_params(axis='both', which='minor', direction='in', right=True)
        
        if nrow == 0 and ncol == ncols-1:
            if 'rcp26' in rcps and 'ssp126' in rcps and len(rcps)==7:
                labels = ['SSP1-26', 'SSP2-45', 'SSP3-70', 'SSP5-85', 'RCP2.6', 'RCP4.5', 'RCP8.5']
            elif 'rcp26' in rcps and len(rcps) == 3:
                labels = ['RCP2.6', 'RCP4.5', 'RCP8.5']
            elif 'ssp126' in rcp and len(rcps) == 4:
                labels = ['SSP1-26', 'SSP2-45', 'SSP3-70', 'SSP5-85']
            ax[nrow,ncol].legend(loc=(1.1,0.3), labels=labels, fontsize=10, ncol=1, columnspacing=0.5, labelspacing=0.25, 
                                 handlelength=1, handletextpad=0.25, borderpad=0, frameon=False
                                 )

        # Adjust row and column
        ncol += 1
        if ncol == ncols:
            nrow += 1
            ncol = 0
    
    for batman in np.arange(nrows*ncols-len(watersheds)):
        ax[nrow,ncol].get_yaxis().set_visible(False)
        ax[nrow,ncol].get_xaxis().set_visible(False)
        ax[nrow,ncol].axis('off')
        ncol += 1
        
    
    # Save figure
    if 'rcp26' in rcps and 'ssp126' in rcps:
        scenario_str = 'rcps_ssps'
    elif 'rcp26' in rcps:
        scenario_str = 'rcps'
    elif 'ssp126' in rcp:
        scenario_str = 'ssps'
    fig_fn = (ws_reg + '_watersheds_runoff_' + str(startyear) + '-' + str(endyear) + '_multigcm' + 
              '-' + scenario_str + '-medstd.png')
    fig.set_size_inches(8.5,11)
    fig.savefig(fig_fp_multigcm + fig_fn, bbox_inches='tight', dpi=300)
    
    #%% ----- MONTHLY RUNOFF CHANGE FIGURE -----
    ncols = 4
    nrows = int(np.ceil(len(watersheds)/ncols))
    fig, ax = plt.subplots(nrows,ncols,gridspec_kw = {'wspace':0.3, 'hspace':0.3})
    
    nrow, ncol = 0,0
    for nws, ws in enumerate(watersheds):
        
        print(ws)
        
        ws_runoff_annual_present_rcps_list = []
        for rcp in rcps:

            # Median and standard deviation
            ws_runoff_monthly = ds_multigcm_runoff[ws][rcp]
            ws_runoff_annual = np.zeros((ws_runoff_monthly.shape[0],101))
            for nrow_ws in np.arange(0,ws_runoff_monthly.shape[0]):
                ws_runoff_annual[nrow_ws,:] = ws_runoff_monthly[nrow_ws,:].reshape(-1,12).sum(1)

            ws_runoff_annual_avg = np.median(ws_runoff_annual, axis=0)
            ws_runoff_annual_std = np.std(ws_runoff_annual, axis=0)

            ws_runoff_annual_present = ws_runoff_annual[:,0:16].mean(1)
            ws_runoff_norm = ws_runoff_annual / ws_runoff_annual_present[:,np.newaxis]
            ws_runoff_norm_avg = np.median(ws_runoff_norm, axis=0)
            ws_runoff_norm_var = np.std(ws_runoff_norm, axis=0)
            
            ws_runoff_annual_present_rcps_list.append(ws_runoff_annual_present.mean())
            
            # Peakwater
            peakwater_yr, peakwater_chg, runoff_chg = peakwater(ws_runoff_annual_avg, years[0:-1], 11)
            
            ax[nrow,ncol].plot(years[0:-1], ws_runoff_norm_avg, color=rcp_colordict[rcp], linestyle=rcp_styledict[rcp], 
                                linewidth=1, zorder=4, label=rcp)
            
            if 'ssp' in rcp:
                peakwater_yr_idx = np.where(peakwater_yr == years[0:-1])[0][0]
                ax[nrow,ncol].vlines(peakwater_yr, 0, ws_runoff_norm_avg[peakwater_yr_idx], 
                                     color=rcp_colordict[rcp], linestyle='--', 
                                     linewidth=1, zorder=5)
                
            if rcp in rcps_plot_mad:
                ax[nrow,ncol].fill_between(years[0:-1], 
                                           (ws_runoff_norm_avg + ws_runoff_norm_var), 
                                           (ws_runoff_norm_avg - ws_runoff_norm_var), 
                                           alpha=0.35, facecolor=rcp_colordict[rcp], label=None)
        
        runoff_gta_str = str(np.round(np.mean(ws_runoff_annual_present_rcps_list) * pygem_prms.density_water / 1e12,1))
        print(ws, runoff_gta_str, 'Gt/yr')
        ax[nrow,ncol].text(0.97, 0.97, runoff_gta_str, size=10, horizontalalignment='right', 
                           verticalalignment='top', transform=ax[nrow,ncol].transAxes, zorder=6)

        if ncol == 0:
            ax[nrow,ncol].set_ylabel('Runoff (-)')
        ax[nrow,ncol].set_xlim(startyear, endyear)
        ax[nrow,ncol].xaxis.set_major_locator(MultipleLocator(40))
        ax[nrow,ncol].xaxis.set_minor_locator(MultipleLocator(10))


        ax[nrow,ncol].set_ylim(0,3)
        ax[nrow,ncol].yaxis.set_major_locator(MultipleLocator(0.5))
        ax[nrow,ncol].yaxis.set_minor_locator(MultipleLocator(0.1))
            
        if ws in watersheds_dict_correct:
            ws_label = watersheds_dict_correct[ws]
        else:
            ws_label = ws
        ax[nrow,ncol].text(1, 1.1, ws_label, size=10, horizontalalignment='right', 
                           verticalalignment='top', transform=ax[nrow,ncol].transAxes)
        ax[nrow,ncol].tick_params(axis='both', which='major', direction='inout', right=True)
        ax[nrow,ncol].tick_params(axis='both', which='minor', direction='in', right=True)
        
        if nrow == 0 and ncol == ncols-1:
            if 'rcp26' in rcps and 'ssp126' in rcps and len(rcps)==7:
                labels = ['SSP1-26', 'SSP2-45', 'SSP3-70', 'SSP5-85', 'RCP2.6', 'RCP4.5', 'RCP8.5']
            elif 'rcp26' in rcps and len(rcps) == 3:
                labels = ['RCP2.6', 'RCP4.5', 'RCP8.5']
            elif 'ssp126' in rcp and len(rcps) == 4:
                labels = ['SSP1-26', 'SSP2-45', 'SSP3-70', 'SSP5-85']
            ax[nrow,ncol].legend(loc=(1.1,0.3), labels=labels, fontsize=10, ncol=1, columnspacing=0.5, labelspacing=0.25, 
                                 handlelength=1, handletextpad=0.25, borderpad=0, frameon=False
                                 )

        # Adjust row and column
        ncol += 1
        if ncol == ncols:
            nrow += 1
            ncol = 0
    
    for batman in np.arange(nrows*ncols-len(watersheds)):
        ax[nrow,ncol].get_yaxis().set_visible(False)
        ax[nrow,ncol].get_xaxis().set_visible(False)
        ax[nrow,ncol].axis('off')
        ncol += 1
        
    
    # Save figure
    if 'rcp26' in rcps and 'ssp126' in rcps:
        scenario_str = 'rcps_ssps'
    elif 'rcp26' in rcps:
        scenario_str = 'rcps'
    elif 'ssp126' in rcp:
        scenario_str = 'ssps'
    fig_fn = (ws_reg + '_watersheds_runoff_' + str(startyear) + '-' + str(endyear) + '_multigcm' + 
              '-' + scenario_str + '-medstd.png')
    fig.set_size_inches(8.5,11)
    fig.savefig(fig_fp_multigcm + fig_fn, bbox_inches='tight', dpi=300)

#%% ---------------------------------------------------------------------------------------------------------
if option_glacier_cs_plots:
    glac_nos = ['1.22193', '2.14297', '11.02739', '11.03005', '11.03643', '14.06794', '15.03733', '17.05076', 
                '17.14140', '18.02342']
    glac_nos = ['18.02342']
    
    startyear = 2015
    endyear = 2100
    normyear = 2015
    years = np.arange(2000,2101+1)
    
    startyear_idx = np.where(years == startyear)[0][0]

    gcm_names_rcps = ['CanESM2', 'CCSM4', 'CNRM-CM5', 'CSIRO-Mk3-6-0', 'GFDL-CM3', 
                      'GFDL-ESM2M', 'GISS-E2-R', 'IPSL-CM5A-LR', 'MPI-ESM-LR', 'NorESM1-M']
    gcm_names_ssps = ['BCC-CSM2-MR', 'CESM2', 'CESM2-WACCM', 'EC-Earth3', 'EC-Earth3-Veg', 'FGOALS-f3-L', 
                      'GFDL-ESM4', 'INM-CM4-8', 'INM-CM5-0', 'MPI-ESM1-2-HR', 'MRI-ESM2-0', 'NorESM2-MM']
#    rcps = ['ssp126', 'ssp245', 'ssp370', 'ssp585', 'rcp26', 'rcp45', 'rcp85']
    rcps = ['ssp126', 'ssp245', 'ssp370', 'ssp585']
    
    netcdf_fp_cmip5 = '/Users/drounce/Documents/HiMAT/spc_backup/simulations-ind/'
#    netcdf_fp_cmip5 = '/Users/drounce/Documents/HiMAT/spc_backup/simulations-nodebris/'
    
#    rcps_plot_mad = ['rcp26', 'rcp45', 'rcp85', 'ssp126', 'ssp245', 'ssp370', 'ssp585']
    rcps_plot_mad = ['ssp126', 'ssp585']
    
    fig_fp_multigcm = netcdf_fp_cmip5 + '/../analysis/figures/ind_glaciers-nodebris/'
    if not os.path.exists(fig_fp_multigcm):
        os.makedirs(fig_fp_multigcm, exist_ok=True)
        

    # Set up processing
    glac_zbed_all = {}
    glac_thick_all = {}
    glac_zsurf_all = {}
    glac_multigcm_zbed = {}
    glac_multigcm_thick = {}
    glac_multigcm_zsurf = {}
    for glac_no in glac_nos:

        gdir = single_flowline_glacier_directory(glac_no, logging_level='CRITICAL')
        
        tasks.init_present_time_glacier(gdir) # adds bins below
        debris.debris_binned(gdir, fl_str='model_flowlines')  # add debris enhancement factors to flowlines
        nfls = gdir.read_pickle('model_flowlines')
        
        x = np.arange(nfls[0].nx) * nfls[0].dx * nfls[0].map_dx         

#        vol_m3_init = ds_binned.bin_volume_annual[0,:,0].values
#        thick_init = ds_binned.bin_thick_annual[0,:,0].values
#        widths_m = nfls[0].widths_m
#        lengths_m = vol_m3_init / thick_init / widths_m
                                
                                
        glac_zbed_all[glac_no] = {}
        glac_thick_all[glac_no] = {}
        glac_zsurf_all[glac_no] = {}
        
        for rcp in rcps:
#        for rcp in rcps[0:1]:
            
            glac_zbed_all[glac_no][rcp] = {}
            glac_thick_all[glac_no][rcp] = {}
            glac_zsurf_all[glac_no][rcp] = {}
            
            if 'rcp' in rcp:
                gcm_names = gcm_names_rcps
            elif 'ssp' in rcp:
                gcm_names = gcm_names_ssps
                
            for gcm_name in gcm_names:
#            for gcm_name in gcm_names[0:1]:

                ds_binned_fp = netcdf_fp_cmip5 + glac_no.split('.')[0].zfill(2) + '/' + gcm_name + '/' + rcp + '/binned/'
                for i in os.listdir(ds_binned_fp):
                    if i.startswith(glac_no):
                        ds_binned_fn = i
                
                ds_binned = xr.open_dataset(ds_binned_fp + ds_binned_fn)

                thick = ds_binned.bin_thick_annual[0,:,:].values
                zsurf_init = ds_binned.bin_surface_h_initial[0].values
                zbed = zsurf_init - thick[:,startyear_idx]
                
                glac_thick_all[glac_no][rcp][gcm_name] = thick
                glac_zbed_all[glac_no][rcp][gcm_name] = zbed
                glac_zsurf_all[glac_no][rcp][gcm_name] = zbed[:,np.newaxis] + thick
                

        #%%
        # MULTI-GCM STATISTICS
        glac_multigcm_zbed[glac_no] = {}
        glac_multigcm_thick[glac_no] = {}
        glac_multigcm_zsurf[glac_no] = {}
        for rcp in rcps: 
            
            if 'rcp' in rcp:
                gcm_names = gcm_names_rcps
            elif 'ssp' in rcp:
                gcm_names = gcm_names_ssps
                
            for ngcm, gcm_name in enumerate(gcm_names):
                
                print(rcp, gcm_name)
    
                glac_zbed_gcm = glac_zbed_all[glac_no][rcp][gcm_name]
                glac_thick_gcm = glac_thick_all[glac_no][rcp][gcm_name]
                glac_zsurf_gcm = glac_zsurf_all[glac_no][rcp][gcm_name]
                
                if x.shape[0] > glac_zbed_gcm.shape[0]:
                    x = x[0:glac_zbed_gcm.shape[0]]
    
                if ngcm == 0:
                    glac_zbed_gcm_all = glac_zbed_gcm 
                    glac_thick_gcm_all = glac_thick_gcm[np.newaxis,:,:]
                    glac_zsurf_gcm_all = glac_zsurf_gcm[np.newaxis,:,:]
                else:
                    glac_zbed_gcm_all = np.vstack((glac_zbed_gcm_all, glac_zbed_gcm))
                    glac_thick_gcm_all = np.vstack((glac_thick_gcm_all, glac_thick_gcm[np.newaxis,:,:]))
                    glac_zsurf_gcm_all = np.vstack((glac_zsurf_gcm_all, glac_zsurf_gcm[np.newaxis,:,:]))
            
            glac_multigcm_zbed[glac_no][rcp] = glac_zbed_gcm_all
            glac_multigcm_thick[glac_no][rcp] = glac_thick_gcm_all
            glac_multigcm_zsurf[glac_no][rcp] = glac_zsurf_gcm_all
    
    
        #%% ----- FIGURE: VOLUME CHANGE MULTI-GCM -----
        fig, ax = plt.subplots(1, 1, squeeze=False, sharex=False, sharey=True, 
                               gridspec_kw = {'wspace':0, 'hspace':0})
        ax2 = ax[0,0].twinx()
        
        add_zbed = True
        for rcp in rcps:
            zbed_med = np.median(glac_multigcm_zbed[glac_no][rcp],axis=0)
            zbed_mad = median_abs_deviation(glac_multigcm_zbed[glac_no][rcp], axis=0)
            
            thick_med = np.median(glac_multigcm_thick[glac_no][rcp],axis=0)
            thick_mad = median_abs_deviation(glac_multigcm_thick[glac_no][rcp], axis=0)
            
            zsurf_med = np.median(glac_multigcm_zsurf[glac_no][rcp],axis=0)
            zsurf_mad = median_abs_deviation(glac_multigcm_zsurf[glac_no][rcp], axis=0)
            
            if add_zbed:
                ax[0,0].plot(x/1000, zbed_med,
                             color='k', linestyle='-', linewidth=1, zorder=5, label='zbed')
                ax[0,0].plot(x/1000, zsurf_med[:,startyear_idx], 
                             color='k', linestyle=':', linewidth=0.5, zorder=4, label='2015')
                ax2.plot(x/1000, thick_med[:,startyear_idx], 
                         color='k', linestyle=':', linewidth=0.5, zorder=4, label='2015')
                add_zbed = False
                
            ax[0,0].plot(x/1000, zsurf_med[:,-1], 
                         color=rcp_colordict[rcp], linestyle='-', linewidth=0.5, zorder=4, label='2100')
            
            ax2.plot(x/1000, thick_med[:,-1],
                     color=rcp_colordict[rcp], linestyle='-', linewidth=0.5, zorder=4, label='2100')
#            if rcp in rcps_plot_mad:
#                ax[0,0].fill_between(years, 
#                                     reg_vol_med + 1.96*reg_vol_mad, 
#                                     reg_vol_med - 1.96*reg_vol_mad, 
#                                     alpha=0.2, facecolor=rcp_colordict[rcp], label=None)
            
        if glac_no in ['1.22193']:
            ax[0,0].set_ylim(-250, 4500)
            ax[0,0].set_xlim(0,60)
            ax[0,0].xaxis.set_major_locator(MultipleLocator(10))
            ax[0,0].xaxis.set_minor_locator(MultipleLocator(5))
            ax[0,0].yaxis.set_major_locator(MultipleLocator(1000))
            ax[0,0].yaxis.set_minor_locator(MultipleLocator(500)) 
            ax2.set_ylim(1200,0)
            ax2.yaxis.set_major_locator(MultipleLocator(500))
            ax2.yaxis.set_minor_locator(MultipleLocator(100))
        elif glac_no in ['2.14297']:
            ax[0,0].set_ylim(1400, 4700)
            ax[0,0].set_xlim(0,8)
            ax[0,0].xaxis.set_major_locator(MultipleLocator(2))
            ax[0,0].xaxis.set_minor_locator(MultipleLocator(1))
            ax[0,0].yaxis.set_major_locator(MultipleLocator(500))
            ax[0,0].yaxis.set_minor_locator(MultipleLocator(100)) 
            ax2.set_ylim(300,0)
            ax2.yaxis.set_major_locator(MultipleLocator(100))
            ax2.yaxis.set_minor_locator(MultipleLocator(20))
        elif glac_no in ['11.02739']:
            ax[0,0].set_ylim(2050, 4300)
            ax[0,0].set_xlim(0,8)
            ax[0,0].xaxis.set_major_locator(MultipleLocator(2))
            ax[0,0].xaxis.set_minor_locator(MultipleLocator(1))
            ax[0,0].yaxis.set_major_locator(MultipleLocator(500))
            ax[0,0].yaxis.set_minor_locator(MultipleLocator(100)) 
            ax2.set_ylim(300,0)
            ax2.yaxis.set_major_locator(MultipleLocator(100))
            ax2.yaxis.set_minor_locator(MultipleLocator(50))
        elif glac_no in ['11.03005']:
            ax[0,0].set_ylim(1600, 5400)
            ax[0,0].set_xlim(0,9)
            ax[0,0].xaxis.set_major_locator(MultipleLocator(2))
            ax[0,0].xaxis.set_minor_locator(MultipleLocator(1))
            ax[0,0].yaxis.set_major_locator(MultipleLocator(500))
            ax[0,0].yaxis.set_minor_locator(MultipleLocator(100)) 
            ax2.set_ylim(300,0)
            ax2.yaxis.set_major_locator(MultipleLocator(100))
            ax2.yaxis.set_minor_locator(MultipleLocator(50))
        elif glac_no in ['11.03643']:
            ax[0,0].set_ylim(1400, 4300)
            ax[0,0].set_xlim(0,11)
            ax[0,0].xaxis.set_major_locator(MultipleLocator(2))
            ax[0,0].xaxis.set_minor_locator(MultipleLocator(1))
            ax[0,0].yaxis.set_major_locator(MultipleLocator(500))
            ax[0,0].yaxis.set_minor_locator(MultipleLocator(100)) 
            ax2.set_ylim(350,0)
            ax2.yaxis.set_major_locator(MultipleLocator(100))
            ax2.yaxis.set_minor_locator(MultipleLocator(20))
        elif glac_no in ['14.06794']:
            ax[0,0].set_ylim(3300, 6800)
            ax[0,0].set_xlim(0,45)
            ax[0,0].xaxis.set_major_locator(MultipleLocator(10))
            ax[0,0].xaxis.set_minor_locator(MultipleLocator(5))
            ax[0,0].yaxis.set_major_locator(MultipleLocator(500))
            ax[0,0].yaxis.set_minor_locator(MultipleLocator(100)) 
            ax2.set_ylim(700,0)
            ax2.yaxis.set_major_locator(MultipleLocator(200))
            ax2.yaxis.set_minor_locator(MultipleLocator(100))
        elif glac_no in ['15.03733']:
            ax[0,0].set_ylim(4650, 7800)
            ax[0,0].set_xlim(0,14)
            ax[0,0].xaxis.set_major_locator(MultipleLocator(5))
            ax[0,0].xaxis.set_minor_locator(MultipleLocator(1))
            ax[0,0].yaxis.set_major_locator(MultipleLocator(500))
            ax[0,0].yaxis.set_minor_locator(MultipleLocator(100)) 
            ax2.set_ylim(300,0)
            ax2.yaxis.set_major_locator(MultipleLocator(50))
            ax2.yaxis.set_minor_locator(MultipleLocator(10))
        elif glac_no in ['17.05076']:
            ax[0,0].set_ylim(300, 3800)
            ax[0,0].set_xlim(0,61)
            ax[0,0].xaxis.set_major_locator(MultipleLocator(10))
            ax[0,0].xaxis.set_minor_locator(MultipleLocator(5))
            ax[0,0].yaxis.set_major_locator(MultipleLocator(500))
            ax[0,0].yaxis.set_minor_locator(MultipleLocator(100)) 
            ax2.set_ylim(900,0)
            ax2.yaxis.set_major_locator(MultipleLocator(100))
            ax2.yaxis.set_minor_locator(MultipleLocator(50))
        elif glac_no in ['17.14140']:
            ax[0,0].set_ylim(3400, 6200)
            ax[0,0].set_xlim(0,11)
            ax[0,0].xaxis.set_major_locator(MultipleLocator(2))
            ax[0,0].xaxis.set_minor_locator(MultipleLocator(1))
            ax[0,0].yaxis.set_major_locator(MultipleLocator(500))
            ax[0,0].yaxis.set_minor_locator(MultipleLocator(100)) 
            ax2.set_ylim(300,0)
            ax2.yaxis.set_major_locator(MultipleLocator(100))
            ax2.yaxis.set_minor_locator(MultipleLocator(20))
        elif glac_no in ['18.02342']:
            ax[0,0].set_ylim(300, 3800)
            ax[0,0].set_xlim(0,21)
            ax[0,0].xaxis.set_major_locator(MultipleLocator(5))
            ax[0,0].xaxis.set_minor_locator(MultipleLocator(1))
            ax[0,0].yaxis.set_major_locator(MultipleLocator(1000))
            ax[0,0].yaxis.set_minor_locator(MultipleLocator(200)) 
            ax2.set_ylim(700,0)
            ax2.yaxis.set_major_locator(MultipleLocator(100))
            ax2.yaxis.set_minor_locator(MultipleLocator(50))
            
        ax[0,0].set_ylabel('Elevation (m a.s.l.)')
        ax[0,0].set_xlabel('Distance along flowline (km)')
        ax2.set_ylabel('Ice thickness (m)')
                
#                ax[0,0].text(0.98, 1.06, rgi_reg_dict[reg], size=10, horizontalalignment='right', 
#                             verticalalignment='top', transform=ax[0,0].transAxes)
#        ax[0,0].legend(rcp_lines, rcp_labels, loc=(0.05,0.05), fontsize=10, labelspacing=0.25, handlelength=1, 
#                       handletextpad=0.25, borderpad=0, frameon=False)
        
        # Legend
        rcp_dict = {'ssp126': 'SSP1-26',
                    'ssp245': 'SSP2-45',
                    'ssp370': 'SSP3-70',
                    'ssp585': 'SSP5-85'}
        rcp_lines = []
        for rcp in rcps:
            line = Line2D([0,1],[0,1], color=rcp_colordict[rcp], linewidth=1)
            rcp_lines.append(line)
        rcp_labels = [rcp_dict[rcp] for rcp in rcps]
        
        # add years
        line = Line2D([0,1],[0,1], color='k', linestyle=':', linewidth=0)
        rcp_lines.append(line)
        rcp_labels.append('')
        line = Line2D([0,1],[0,1], color='k', linestyle=':', linewidth=1)
        rcp_lines.append(line)
        rcp_labels.append('2015')
        line = Line2D([0,1],[0,1], color='grey', linestyle='-', linewidth=1)
        rcp_lines.append(line)
        rcp_labels.append('2100')
        line = Line2D([0,1],[0,1], color='k', linewidth=1)
        rcp_lines.append(line)
        rcp_labels.append('bed')
        
        ax[0,0].legend(rcp_lines, rcp_labels, loc=(0.02,0.02), fontsize=8, labelspacing=0.25, handlelength=1, 
                       handletextpad=0.25, borderpad=0, ncol=2, columnspacing=0.5, frameon=False)
        
#        loc=(-1.4,0.2), labels=labels, fontsize=10, ncol=2, columnspacing=0.5, labelspacing=0.25, 
#                      handlelength=1, handletextpad=0.25, borderpad=0, frameon=False
        
        ax[0,0].tick_params(direction='inout')
        ax2.tick_params(direction='inout')
        # Save figure
        if 'rcp26' in rcps and 'ssp126' in rcps:
            scenario_str = 'rcps_ssps'
        elif 'rcp26' in rcps:
            scenario_str = 'rcps'
        elif 'ssp126' in rcp:
            scenario_str = 'ssps'
        fig_fn = (glac_no + '_profile_2100.png')
        fig.set_size_inches(4,3)
        fig.savefig(fig_fp_multigcm + fig_fn, bbox_inches='tight', dpi=300)
    
    
#%%   
if option_debris_comparison:
    
    pickle_fp = '/Users/drounce/Documents/HiMAT/spc_backup/analysis/pickle/'
    pickle_fp_nodebris = '/Users/drounce/Documents/HiMAT/spc_backup/analysis-nodebris/pickle-nodebris/'
    
    #%%
    fig_fp = pickle_fp + '/../figures/'
    
    startyear = 2015
    endyear = 2100
    normyear = 2015
    years = np.arange(2000,2101+1)
    
    gcm_names = ['BCC-CSM2-MR', 'CESM2', 'CESM2-WACCM', 'EC-Earth3', 'EC-Earth3-Veg', 'FGOALS-f3-L', 
                      'GFDL-ESM4', 'INM-CM4-8', 'INM-CM5-0', 'MPI-ESM1-2-HR', 'MRI-ESM2-0', 'NorESM2-MM']
    rcps = ['ssp126', 'ssp245', 'ssp370', 'ssp585']
    
    regions = [1, 2, 11, 12, 15, 18]
#    regions = [1]
    
    rcps_plot_mad = ['ssp126', 'ssp585']
    
    fig_fp_debriscompare = fig_fp + 'debris_compare/'
    if not os.path.exists(fig_fp_debriscompare):
        os.makedirs(fig_fp_debriscompare, exist_ok=True)
    
    # Set up processing
    reg_vol_all = {}
    reg_vol_all_nodebris = {}
#    reg_vol_all_bwl = {}
#    reg_area_all = {} 
            
    for reg in regions:
    
        reg_vol_all[reg] = {}
        reg_vol_all_nodebris[reg] = {}
#        reg_vol_all_bwl[reg] = {}
#        reg_area_all[reg] = {}
        
        for rcp in rcps:
            reg_vol_all[reg][rcp] = {}
            reg_vol_all_nodebris[reg][rcp] = {}
#            reg_vol_all_bwl[reg][rcp] = {}
#            reg_area_all[reg][rcp] = {}
            
                
            for gcm_name in gcm_names:
                
                # ----- GCM/RCP PICKLE FILEPATHS AND FILENAMES -----
                pickle_fp_reg =  pickle_fp + str(reg).zfill(2) + '/O1Regions/' + gcm_name + '/' + rcp + '/'
                pickle_fp_watershed =  pickle_fp + str(reg).zfill(2) + '/watersheds/' + gcm_name + '/' + rcp + '/'
                # Region string prefix
                reg_rcp_gcm_str = 'R' + str(reg) + '_' + rcp + '_' + gcm_name
                watershed_rcp_gcm_str = 'R' + str(reg) + '_watersheds_' + rcp + '_' + gcm_name
                
                # Filenames
                fn_reg_vol_annual = reg_rcp_gcm_str + '_vol_annual.pkl' 
#                fn_reg_vol_annual_bwl = reg_rcp_gcm_str + '_vol_annual_bwl.pkl'
#                fn_reg_vol_annual_bd = reg_rcp_gcm_str + '_vol_annual_bd.pkl'
#                fn_reg_area_annual = reg_rcp_gcm_str + '_area_annual.pkl'
#                fn_reg_area_annual_bd = reg_rcp_gcm_str + '_area_annual_bd.pkl'
#                fn_reg_vol_annual_binned = reg_rcp_gcm_str + '_vol_annual_binned.pkl'
#                fn_reg_vol_annual_binned_bd = reg_rcp_gcm_str + '_vol_annual_binned_bd.pkl'
#                fn_reg_area_annual_binned = reg_rcp_gcm_str + '_area_annual_binned.pkl'
#                fn_reg_area_annual_binned_bd = reg_rcp_gcm_str + '_area_annual_binned_bd.pkl'
#                fn_reg_acc_monthly = reg_rcp_gcm_str + '_acc_monthly.pkl'
#                fn_reg_refreeze_monthly = reg_rcp_gcm_str + '_refreeze_monthly.pkl'
#                fn_reg_melt_monthly = reg_rcp_gcm_str + '_melt_monthly.pkl'
#                fn_reg_frontalablation_monthly = reg_rcp_gcm_str + '_frontalablation_monthly.pkl'
#                fn_reg_massbaltotal_monthly = reg_rcp_gcm_str + '_massbaltotal_monthly.pkl'
#                fn_reg_mbclim_annual_binned = reg_rcp_gcm_str + '_mbclim_annual_binned.pkl'
#                fn_watershed_mbclim_annual_binned = watershed_rcp_gcm_str + '_mbclim_annual_binned.pkl'
#                fn_reg_runoff_monthly_moving = reg_rcp_gcm_str + '_runoff_monthly_moving.pkl'
#                fn_watershed_runoff_monthly_moving = watershed_rcp_gcm_str + '_runoff_monthly_moving.pkl'
#                fn_reg_runoff_monthly_fixed = reg_rcp_gcm_str + '_runoff_monthly_fixed.pkl'
#                fn_watershed_runoff_monthly_fixed = watershed_rcp_gcm_str + '_runoff_monthly_fixed.pkl'
#                fn_reg_prec_monthly = reg_rcp_gcm_str + '_prec_monthly.pkl'
#                fn_watershed_prec_monthly = watershed_rcp_gcm_str + '_prec_monthly.pkl' 
#                fn_reg_offglac_prec_monthly = reg_rcp_gcm_str + '_offglac_prec_monthly.pkl'  
#                fn_watershed_offglac_prec_monthly = watershed_rcp_gcm_str + '_offglac_prec_monthly.pkl'
#                fn_reg_offglac_melt_monthly = reg_rcp_gcm_str + '_offglac_melt_monthly.pkl'
#                fn_watershed_offglac_melt_monthly = watershed_rcp_gcm_str + '_offglac_melt_monthly.pkl'
#                fn_reg_offglac_refreeze_monthly = reg_rcp_gcm_str + '_offglac_refreeze_monthly.pkl'
#                fn_watershed_offglac_refreeze_monthly = watershed_rcp_gcm_str + '_offglac_refreeze_monthly.pkl'
#                fn_reg_ela_annual = reg_rcp_gcm_str + '_ela_annual.pkl'
#                fn_reg_aar_annual = reg_rcp_gcm_str + '_aar_annual.pkl'
                
                 # Volume
                with open(pickle_fp_reg + fn_reg_vol_annual, 'rb') as f:
                    reg_vol_annual = pickle.load(f)
#                # Volume below sea level
#                with open(pickle_fp_reg + fn_reg_vol_annual_bwl, 'rb') as f:
#                    reg_vol_annual_bwl = pickle.load(f)
#                # Volume below debris
#                with open(pickle_fp_reg + fn_reg_vol_annual_bd, 'rb') as f:
#                    reg_vol_annual_bd = pickle.load(f)
#                # Area 
#                with open(pickle_fp_reg + fn_reg_area_annual, 'rb') as f:
#                    reg_area_annual = pickle.load(f)
#                # Area below debris
#                with open(pickle_fp_reg + fn_reg_area_annual_bd, 'rb') as f:
#                    reg_area_annual_bd = pickle.load(f)
#                # Binned Volume
#                with open(pickle_fp_reg + fn_reg_vol_annual_binned, 'rb') as f:
#                    reg_vol_annual_binned = pickle.load(f)
#                # Binned Volume below debris
#                with open(pickle_fp_reg + fn_reg_vol_annual_binned_bd, 'rb') as f:
#                    reg_vol_annual_binned_bd = pickle.load(f)
#                # Binned Area
#                with open(pickle_fp_reg + fn_reg_area_annual_binned, 'rb') as f:
#                    reg_area_annual_binned = pickle.load(f)
#                # Binned Area below debris
#                with open(pickle_fp_reg + fn_reg_area_annual_binned_bd, 'rb') as f:
#                    reg_area_annual_binned_bd = pickle.load(f)
#                # Mass balance: accumulation
#                with open(pickle_fp_reg + fn_reg_acc_monthly, 'rb') as f:
#                    reg_acc_monthly = pickle.load(f)
#                # Mass balance: refreeze
#                with open(pickle_fp_reg + fn_reg_refreeze_monthly, 'rb') as f:
#                    reg_refreeze_monthly = pickle.load(f)
#                # Mass balance: melt
#                with open(pickle_fp_reg + fn_reg_melt_monthly, 'rb') as f:
#                    reg_melt_monthly = pickle.load(f)
#                # Mass balance: frontal ablation
#                with open(pickle_fp_reg + fn_reg_frontalablation_monthly, 'rb') as f:
#                    reg_frontalablation_monthly = pickle.load(f)
#                # Mass balance: total mass balance
#                with open(pickle_fp_reg + fn_reg_massbaltotal_monthly, 'rb') as f:
#                    reg_massbaltotal_monthly = pickle.load(f)
#                # Binned Climatic Mass Balance
#                with open(pickle_fp_reg + fn_reg_mbclim_annual_binned, 'rb') as f:
#                    reg_mbclim_annual_binned = pickle.load(f)
#                # Runoff: moving-gauged
#                with open(pickle_fp_reg + fn_reg_runoff_monthly_moving, 'rb') as f:
#                    reg_runoff_monthly_moving = pickle.load(f)
#                with open(pickle_fp_watershed + fn_watershed_runoff_monthly_moving, 'rb') as f:
#                    watershed_runoff_monthly_moving = pickle.load(f)
#                # Runoff: fixed-gauged
#                with open(pickle_fp_reg + fn_reg_runoff_monthly_fixed, 'rb') as f:
#                    reg_runoff_monthly_fixed = pickle.load(f)
#                with open(pickle_fp_watershed + fn_watershed_runoff_monthly_fixed, 'rb') as f:
#                    watershed_runoff_monthly_fixed = pickle.load(f)
#                # Runoff: precipitation
#                with open(pickle_fp_reg + fn_reg_prec_monthly, 'rb') as f:
#                    reg_prec_monthly = pickle.load(f)
#                with open(pickle_fp_watershed + fn_watershed_prec_monthly, 'rb') as f:
#                    watershed_prec_monthly= pickle.load(f)
#                # Runoff: off-glacier precipitation
#                with open(pickle_fp_reg + fn_reg_offglac_prec_monthly, 'rb') as f:
#                    reg_offglac_prec_monthly = pickle.load(f)
#                with open(pickle_fp_watershed + fn_watershed_offglac_prec_monthly, 'rb') as f:
#                    watershed_offglac_prec_monthly = pickle.load(f)
#                # Runoff: off-glacier melt
#                with open(pickle_fp_reg + fn_reg_offglac_melt_monthly, 'rb') as f:
#                    reg_offglac_melt_monthly = pickle.load(f)
#                with open(pickle_fp_watershed + fn_watershed_offglac_melt_monthly, 'rb') as f:
#                    watershed_offglac_melt_monthly = pickle.load(f)
#                # Runoff: off-glacier refreeze
#                with open(pickle_fp_reg + fn_reg_offglac_refreeze_monthly, 'rb') as f:
#                    reg_offglac_refreeze_monthly = pickle.load(f)
#                with open(pickle_fp_watershed + fn_watershed_offglac_refreeze_monthly, 'rb') as f:
#                    watershed_offglac_refreeze_monthly = pickle.load(f)
#                # ELA
#                with open(pickle_fp_reg + fn_reg_ela_annual, 'rb') as f:
#                    reg_ela_annual = pickle.load(f)
#                # AAR
#                with open(pickle_fp_reg + fn_reg_aar_annual, 'rb') as f:
#                    reg_aar_annual = pickle.load(f) 
                    
                # ===== NO DEBRIS =====
                pickle_fp_reg_nodebris =  pickle_fp_nodebris + str(reg).zfill(2) + '/O1Regions/' + gcm_name + '/' + rcp + '/'
                
                # Filenames
                fn_reg_vol_annual = reg_rcp_gcm_str + '_vol_annual.pkl' 
#                fn_reg_vol_annual_bwl = reg_rcp_gcm_str + '_vol_annual_bwl.pkl'
#                fn_reg_vol_annual_bd = reg_rcp_gcm_str + '_vol_annual_bd.pkl'
#                fn_reg_area_annual = reg_rcp_gcm_str + '_area_annual.pkl'
#                fn_reg_area_annual_bd = reg_rcp_gcm_str + '_area_annual_bd.pkl'
#                fn_reg_vol_annual_binned = reg_rcp_gcm_str + '_vol_annual_binned.pkl'
#                fn_reg_vol_annual_binned_bd = reg_rcp_gcm_str + '_vol_annual_binned_bd.pkl'
#                fn_reg_area_annual_binned = reg_rcp_gcm_str + '_area_annual_binned.pkl'
#                fn_reg_area_annual_binned_bd = reg_rcp_gcm_str + '_area_annual_binned_bd.pkl'
#                fn_reg_acc_monthly = reg_rcp_gcm_str + '_acc_monthly.pkl'
#                fn_reg_refreeze_monthly = reg_rcp_gcm_str + '_refreeze_monthly.pkl'
#                fn_reg_melt_monthly = reg_rcp_gcm_str + '_melt_monthly.pkl'
#                fn_reg_frontalablation_monthly = reg_rcp_gcm_str + '_frontalablation_monthly.pkl'
#                fn_reg_massbaltotal_monthly = reg_rcp_gcm_str + '_massbaltotal_monthly.pkl'
#                fn_reg_mbclim_annual_binned = reg_rcp_gcm_str + '_mbclim_annual_binned.pkl'
#                fn_watershed_mbclim_annual_binned = watershed_rcp_gcm_str + '_mbclim_annual_binned.pkl'
#                fn_reg_runoff_monthly_moving = reg_rcp_gcm_str + '_runoff_monthly_moving.pkl'
#                fn_watershed_runoff_monthly_moving = watershed_rcp_gcm_str + '_runoff_monthly_moving.pkl'
#                fn_reg_runoff_monthly_fixed = reg_rcp_gcm_str + '_runoff_monthly_fixed.pkl'
#                fn_watershed_runoff_monthly_fixed = watershed_rcp_gcm_str + '_runoff_monthly_fixed.pkl'
#                fn_reg_prec_monthly = reg_rcp_gcm_str + '_prec_monthly.pkl'
#                fn_watershed_prec_monthly = watershed_rcp_gcm_str + '_prec_monthly.pkl' 
#                fn_reg_offglac_prec_monthly = reg_rcp_gcm_str + '_offglac_prec_monthly.pkl'  
#                fn_watershed_offglac_prec_monthly = watershed_rcp_gcm_str + '_offglac_prec_monthly.pkl'
#                fn_reg_offglac_melt_monthly = reg_rcp_gcm_str + '_offglac_melt_monthly.pkl'
#                fn_watershed_offglac_melt_monthly = watershed_rcp_gcm_str + '_offglac_melt_monthly.pkl'
#                fn_reg_offglac_refreeze_monthly = reg_rcp_gcm_str + '_offglac_refreeze_monthly.pkl'
#                fn_watershed_offglac_refreeze_monthly = watershed_rcp_gcm_str + '_offglac_refreeze_monthly.pkl'
#                fn_reg_ela_annual = reg_rcp_gcm_str + '_ela_annual.pkl'
#                fn_reg_aar_annual = reg_rcp_gcm_str + '_aar_annual.pkl'
                
                 # Volume
                with open(pickle_fp_reg_nodebris + fn_reg_vol_annual, 'rb') as f:
                    reg_vol_annual_nodebris = pickle.load(f)
                    
                
                print(reg, gcm_name, rcp, 'difference [%]', np.round((reg_vol_annual_nodebris[50] - reg_vol_annual[50])/ reg_vol_annual[0] * 100,2))
                
                reg_vol_all[reg][rcp][gcm_name] = reg_vol_annual
                reg_vol_all_nodebris[reg][rcp][gcm_name] = reg_vol_annual_nodebris
                
    
    # MULTI-GCM STATISTICS
    ds_multigcm_vol = {}
    ds_multigcm_vol_nodebris = {}
#    ds_multigcm_vol_bsl = {}
#    ds_multigcm_area = {}
    for reg in regions:
        ds_multigcm_vol[reg] = {}
        ds_multigcm_vol_nodebris[reg] = {}
#        ds_multigcm_vol_bsl[reg] = {}
#        ds_multigcm_area[reg] = {}
        for rcp in rcps: 
                
            for ngcm, gcm_name in enumerate(gcm_names):
                
                print(rcp, gcm_name)
    
                reg_vol_gcm = reg_vol_all[reg][rcp][gcm_name]
                reg_vol_gcm_nodebris = reg_vol_all_nodebris[reg][rcp][gcm_name]
#                reg_vol_bsl_gcm = reg_vol_all_bwl[reg][rcp][gcm_name]
#                reg_area_gcm = reg_area_all[reg][rcp][gcm_name]
    
                if ngcm == 0:
                    reg_vol_gcm_all = reg_vol_gcm   
                    reg_vol_gcm_all_nodebris = reg_vol_gcm_nodebris
#                    reg_vol_bsl_gcm_all = reg_vol_bsl_gcm   
#                    reg_area_gcm_all = reg_area_gcm    
                else:
                    reg_vol_gcm_all = np.vstack((reg_vol_gcm_all, reg_vol_gcm))
                    reg_vol_gcm_all_nodebris = np.vstack((reg_vol_gcm_all_nodebris, reg_vol_gcm_nodebris))
#                    reg_vol_bsl_gcm_all = np.vstack((reg_vol_bsl_gcm_all, reg_vol_bsl_gcm))
#                    reg_area_gcm_all = np.vstack((reg_area_gcm_all, reg_area_gcm))
            
            ds_multigcm_vol[reg][rcp] = reg_vol_gcm_all
            ds_multigcm_vol_nodebris[reg][rcp] = reg_vol_gcm_all_nodebris
#            ds_multigcm_vol_bsl[reg][rcp] = reg_vol_bsl_gcm_all
#            ds_multigcm_area[reg][rcp] = reg_area_gcm_all
                
                
    #%% ----- FIGURE: ALL MULTI-GCM NORMALIZED VOLUME CHANGE -----
    normyear_idx = np.where(years == normyear)[0][0]
    
    for reg in regions:
        fig = plt.figure()    
        gs = fig.add_gridspec(nrows=6,ncols=4,wspace=0.3,hspace=0)
        ax1 = fig.add_subplot(gs[0:2,0:2])
        ax2 = fig.add_subplot(gs[2:3,0:2])
        ax3 = fig.add_subplot(gs[3:4,0:2])

        for rcp in rcps:
            
            # Median and absolute median deviation
            reg_vol = ds_multigcm_vol[reg][rcp]
            reg_vol_mean = np.median(reg_vol, axis=0)
            reg_vol_std = median_abs_deviation(reg_vol, axis=0)

            reg_vol_nodebris = ds_multigcm_vol_nodebris[reg][rcp]
            reg_vol_mean_nodebris = np.median(reg_vol_nodebris, axis=0)
            reg_vol_std_nodebris = median_abs_deviation(reg_vol_nodebris, axis=0)
            
            # Normalized
            reg_vol_mean_norm = reg_vol_mean / reg_vol_mean[normyear_idx]
            reg_vol_std_norm = reg_vol_std / reg_vol_mean[normyear_idx]
            
            reg_vol_mean_norm_nodebris = reg_vol_mean_nodebris / reg_vol_mean_nodebris[normyear_idx]
            reg_vol_std_norm_nodebris = reg_vol_std_nodebris / reg_vol_mean_nodebris[normyear_idx]
            
            # Delay in timing
            reg_vol_delay = np.zeros(reg_vol_mean_norm_nodebris.shape)
            for nyear, year in enumerate(years):
#                print(nyear, year, reg_vol_mean_norm[nyear], reg_vol_mean_norm_nodebris[nyear])
                year_idx = np.where(reg_vol_mean_norm_nodebris[nyear] < reg_vol_mean_norm)[0]
                if len(year_idx) > 0:
                    reg_vol_delay[nyear] = years[year_idx[-1]] - year

            # Plot
            ax1.plot(years, reg_vol_mean_norm, color=rcp_colordict[rcp], linestyle=rcp_styledict[rcp], 
                    linewidth=1, zorder=4, label=rcp)
            ax1.plot(years, reg_vol_mean_norm_nodebris, color=rcp_colordict[rcp], linestyle=':', 
                    linewidth=1, zorder=3)
            ax2.plot(years, (reg_vol_mean_norm - reg_vol_mean_norm_nodebris)*100, color=rcp_colordict[rcp], 
                     linewidth=1, zorder=4)
            ax3.plot(years, reg_vol_delay, color=rcp_colordict[rcp], linewidth=1, zorder=4)
#            if rcp in rcps_plot_mad:
#                ax.fill_between(years, 
#                                reg_vol_med_norm + 1.96*reg_vol_mad_norm, 
#                                reg_vol_med_norm - 1.96*reg_vol_mad_norm, 
#                                alpha=0.2, facecolor=rcp_colordict[rcp], label=None)
        
        ax1.set_ylabel('Mass (rel. to 2015)')
        ax1.set_xlim(startyear, endyear)
        ax1.xaxis.set_major_locator(MultipleLocator(40))
        ax1.xaxis.set_minor_locator(MultipleLocator(10))
        ax1.set_ylim(0,1.1)
        ax1.yaxis.set_major_locator(MultipleLocator(0.2))
        ax1.yaxis.set_minor_locator(MultipleLocator(0.1))
        ax1.tick_params(axis='both', which='major', direction='inout', right=True)
        ax1.tick_params(axis='both', which='minor', direction='in', right=True)
        
        ax2.yaxis.set_label_position("right")
        ax2.set_ylabel('Difference (%)')
        ax2.set_xlim(startyear, endyear)
        ax2.xaxis.set_major_locator(MultipleLocator(40))
        ax2.xaxis.set_minor_locator(MultipleLocator(10))
        if reg in [1,2,11]:
            ax2.set_ylim(0,7)
        elif reg in [18]:
            ax2.set_ylim(0,14)
        ax2.yaxis.set_major_locator(MultipleLocator(5))
        ax2.yaxis.set_minor_locator(MultipleLocator(1))
        ax2.tick_params(axis='both', which='major', direction='inout', right=True)
        ax2.tick_params(axis='both', which='minor', direction='in', right=True)
        
        ax3.set_ylabel('Difference (yrs)')
        ax3.set_xlim(startyear, endyear)
        ax3.xaxis.set_major_locator(MultipleLocator(40))
        ax3.xaxis.set_minor_locator(MultipleLocator(10))
#        if reg in [1,2,11]:
#            ax2.set_ylim(0,7)
#        elif reg in [18]:
#            ax2.set_ylim(0,14)
#        ax2.yaxis.set_major_locator(MultipleLocator(5))
#        ax2.yaxis.set_minor_locator(MultipleLocator(1))
        ax2.tick_params(axis='both', which='major', direction='inout', right=True)
        ax2.tick_params(axis='both', which='minor', direction='in', right=True)
        
        ax1.text(1, 1.16, rgi_reg_dict[reg], size=10, horizontalalignment='right', 
                 verticalalignment='top', transform=ax1.transAxes)
        ax1.axes.xaxis.set_ticklabels([])
            
        # Save figure
        fig_fn = (str(reg).zfill(2) + '_volchange_norm_' + str(startyear) + '-' + str(endyear) + '_' + 'debriscompare.png')
        fig.set_size_inches(4,4)
        fig.savefig(fig_fp_debriscompare + fig_fn, bbox_inches='tight', dpi=300)
    
        print('COMPUTE HOW MANY YEARS DELAY IT IS IN VOLUME CHANGE')
        print('\nADD GLACIER RUNOFF CHANGES\n')
    
#%%

print('Total processing time:', time.time()-time_start, 's')
        
        #%%
                    
# ----- FIGURE: DIAGNOSTIC OF EVERYTHING ----- 
#fig, ax = plt.subplots(1, 1, squeeze=False, sharex=False, sharey=False, 
#                       gridspec_kw = {'wspace':0.7, 'hspace':0.5})
#                                
## RUNOFF CHANGE 
#years = np.arange(0,102)
#reg_runoff_annual_fixed = B_annual + A_annual
#reg_runoff_annual_moving = A_annual
#ax[0,0].plot(years[0:-1], reg_runoff_annual_fixed, color='b', linewidth=1, zorder=4, label='Fixed')
#ax[0,0].plot(years[0:-1], reg_runoff_annual_moving, color='k', linestyle=':', linewidth=1, zorder=4, label='Moving')
#ax[0,0].set_ylabel('Runoff (km$^{3}$)')
#plt.show
        
#%%
#import zipfile
#zip_fp = '/Users/drounce/Documents/HiMAT/climate_data/cmip6/'
#zip_fn = 'BCC-CSM2-MR.zip'
#with zipfile.ZipFile(zip_fp + zip_fn, 'r') as zip_ref:
#    zip_ref.extractall(zip_fp)
        
#%% ----- MISSING DIFFERENT RCPS/GCMS -----
# Need to run script twice and comment out the processing (cheap shortcut)
#missing_rcp26 = glacno_list_missing.copy()
#missing_rcp45 = glacno_list_missing.copy()
#A = np.setdiff1d(missing_rcp45, missing_rcp26).tolist()
#print(A)
        
#%% ----- MOVE FILES -----
#gcm_names = ['CanESM2', 'CCSM4', 'CNRM-CM5', 'CSIRO-Mk3-6-0', 'GFDL-CM3', 
#             'GFDL-ESM2M', 'GISS-E2-R', 'IPSL-CM5A-LR', 'MPI-ESM-LR', 'NorESM1-M']
#rcps = ['rcp26', 'rcp45', 'rcp85']
#for gcm_name in gcm_names:
#    for rcp in rcps:
#        print('moving', gcm_name, rcp)
#
#        # Filepath where glaciers are stored
#        netcdf_fp_binned = netcdf_fp_cmip5 + '13-batch1/' + gcm_name + '/' + rcp + '/binned/'
#        netcdf_fp_stats = netcdf_fp_cmip5 + '13-batch1/' + gcm_name + '/' + rcp + '/stats/'
#        
#        move_binned = netcdf_fp_cmip5 + '13/' + gcm_name + '/' + rcp + '/binned/'
#        move_stats = netcdf_fp_cmip5 + '13/' + gcm_name + '/' + rcp + '/stats/'
#        
#        binned_fns_2move = []
#        for i in os.listdir(netcdf_fp_binned):
#            if i.endswith('.nc'):
#                binned_fns_2move.append(i)
#        binned_fns_2move = sorted(binned_fns_2move)
#
#        if len(binned_fns_2move) > 0:
#            for i in binned_fns_2move:
#                shutil.move(netcdf_fp_binned + i, move_binned + i)
#        
#        stats_fns_2move = []
#        for i in os.listdir(netcdf_fp_stats):
#            if i.endswith('.nc'):
#                stats_fns_2move.append(i)
#        stats_fns_2move = sorted(stats_fns_2move)
#
#        if len(stats_fns_2move) > 0:
#            for i in stats_fns_2move:
#                shutil.move(netcdf_fp_stats + i, move_stats + i)
        