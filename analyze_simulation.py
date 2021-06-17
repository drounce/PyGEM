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
#import class_climate
#import class_mbdata
import pygem.pygem_input as pygem_prms
#import pygemfxns_gcmbiasadj as gcmbiasadj
import pygem.pygem_modelsetup as modelsetup

# Script options
option_plot_era5_volchange = False
option_get_missing_glacno = True

option_plot_cmip5_volchange = False
option_plot_era5_AAD = False
option_process_data = False


#%% ===== Input data =====
netcdf_fp_cmip5 = '/Users/drounce/Documents/HiMAT/spc_backup/simulations/'
#netcdf_fp_cmip5 = '/Users/drounce/Documents/HiMAT/spc_backup/simulations-growth/'

netcdf_fp_sims = '/Users/drounce/Documents/HiMAT/spc_backup/simulations/'


#%%
#regions = [1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19]
regions = [12]

# GCMs and RCP scenarios
gcm_names = ['CanESM2', 'CCSM4', 'CNRM-CM5', 'CSIRO-Mk3-6-0', 'GFDL-CM3', 
             'GFDL-ESM2M', 'GISS-E2-R', 'IPSL-CM5A-LR', 'MPI-ESM-LR', 'NorESM1-M']
#gcm_names = ['CanESM2']
rcps = ['rcp26', 'rcp45', 'rcp85']
#rcps = ['rcp26']

# Grouping
#grouping = 'all'
grouping = 'rgi_region'
#grouping = 'watershed'
#grouping = 'degree'

degree_size = 0.5

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
rgi_reg_dict = {1:'Alaska'}
#title_dict = {}
#title_location = {}
#rcp_dict = {'rcp26': '2.6',
#            'rcp45': '4.5',
#            'rcp60': '6.0',
#            'rcp85': '8.5'}
# Colors list
colors_rgb = [(0.00, 0.57, 0.57), (0.71, 0.43, 1.00), (0.86, 0.82, 0.00), (0.00, 0.29, 0.29), (0.00, 0.43, 0.86), 
              (0.57, 0.29, 0.00), (1.00, 0.43, 0.71), (0.43, 0.71, 1.00), (0.14, 1.00, 0.14), (1.00, 0.71, 0.47), 
              (0.29, 0.00, 0.57), (0.57, 0.00, 0.00), (0.71, 0.47, 1.00), (1.00, 1.00, 0.47)]
gcm_colordict = dict(zip(gcm_names, colors_rgb[0:len(gcm_names)]))
rcp_colordict = {'rcp26':'b', 'rcp45':'k', 'rcp60':'m', 'rcp85':'r'}
rcp_styledict = {'rcp26':':', 'rcp45':'--', 'rcp85':'-.'}

# Bounds (90% bounds --> 95% above/below given threshold)
low_percentile = 5
high_percentile = 95

colors = ['#387ea0', '#fcb200', '#d20048']
linestyles = ['-', '--', ':']

# Shapefiles
rgiO1_shp_fn = '/Users/davidrounce/Documents/Dave_Rounce/HiMAT/RGI/rgi60/00_rgi60_regions/00_rgi60_O1Regions.shp'
#rgi_glac_shp_fn = '/Users/davidrounce/Documents/Dave_Rounce/HiMAT/qgis_himat/rgi60_HMA.shp'


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
if option_process_data:
    print('lets do it!')
    
    overwrite_pickle = False
    
    grouping = 'all'

    netcdf_fn_ending = '_ERA5_MCMC_ba1_50sets_2000_2019_annual.nc'
    fig_fp = netcdf_fp_sims + '/../analysis/figures/'
    if not os.path.exists(fig_fp):
        os.makedirs(fig_fp, exist_ok=True)
    csv_fp = netcdf_fp_sims + '/../analysis/csv/'
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
    #%%
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
#        #
##        # ===== MASS BALANCE COMPARISON =====
##        mb_compare_fullfn = csv_fp + str(reg).zfill(2) + '-mb_compare_glac-ERA5.csv'
##        
##        # Load mass balance data
##        # Observed mass loss
##        if mb_dataset in ['Hugonnet2020']:
##            mbdata_fp = pygem_prms.hugonnet_fp
##            mbdata_fn = pygem_prms.hugonnet_fn
##            rgiid_cn = pygem_prms.hugonnet_rgi_glacno_cn
##            mb_cn = pygem_prms.hugonnet_mb_cn
##            mberr_cn = pygem_prms.hugonnet_mb_err_cn
##            t1_cn = pygem_prms.hugonnet_time1_cn
##            t2_cn = pygem_prms.hugonnet_time2_cn
##        
##        assert os.path.exists(mbdata_fp + mbdata_fn), "Error: mb dataset does not exist."
##
##        mb_df_all = pd.read_csv(mbdata_fp + mbdata_fn)
##        mb_df_all_rgiids = list(mb_df_all[rgiid_cn])
##
##        rgiids = list(main_glac_rgi.RGIId.values)
##        mb_df_idx = [mb_df_all_rgiids.index(x) for x in rgiids]
##        mb_df = mb_df_all.loc[mb_df_idx,:]
##        mb_df = mb_df.sort_values('RGIId', ascending=True)
##        mb_df.reset_index(inplace=True, drop=True)
##        # gt/yr = mb_mwea * area_km2 * (1e6 m2 / 1 km2) * (1000 kg / 1 m3) * (1 Gt / 1e12 kg)
##        mb_df['mb_gta'] = mb_df['mb_mwea'] * mb_df['area'] * 1e6 * pygem_prms.density_water / 1e12
##
##        # Load model data
##        reg_vol_fn = 'R' + str(reg) + '_ERA5_volume_annual.pkl'
##        reg_area_fn = 'R' + str(reg) + '_ERA5_area_annual.pkl'
##                
##        if not os.path.exists(pickle_fp + reg_vol_fn) or overwrite_pickle:
##            
##            # MB comparison
##            mb_compare_cns = ['RGIId', 'area_km2', 'mb_obs_mwea', 'mb_obs_mwea_err', 
##                              'mb_mwea_emulator', 'mb_mwea_mcmc', 'mb_mwea_oggm',
##                              'mb_obs_gta', 'mb_gta_emulator', 'mb_gta_mcmc', 'mb_gta_oggm']
##            mb_compare = pd.DataFrame(np.zeros((main_glac_rgi.shape[0],len(mb_compare_cns))), columns=mb_compare_cns)
##            mb_compare['RGIId'] = main_glac_rgi['RGIId']
##            mb_compare['area_km2'] = main_glac_rgi['Area']       
##            mb_compare['mb_obs_mwea'] = mb_df['mb_mwea']
##            mb_compare['mb_obs_mwea_err'] = mb_df['mb_mwea_err']
##            mb_compare['mb_obs_gta'] = mb_df['mb_gta']
##            
##            years = None
##            reg_vol = None
##            reg_area = None
##            for nglac, glacno in enumerate(main_glac_rgi.glacno.values):
###            for nglac, glacno in enumerate(main_glac_rgi.glacno.values[0:1]):
##                
##                if nglac%500==0:
##                    print(nglac, glacno)
##                
##                ds_fn = glacno + netcdf_fn_ending
##                        
##                ds = xr.open_dataset(netcdf_fp + ds_fn)
##
##                # Time values
##                if years is None:
##                    years = ds.year.values
##                    idx_cal_startyr = np.where(years == cal_startyr)[0][0]
##                    idx_cal_endyr = np.where(years == cal_endyr)[0][0]
##                    
##                # Volume data
##                glac_vol = ds.glac_volume_annual.values[0,:,:]
##                glac_area = ds.glac_area_annual.values[0,:,:]
##                    
##                # Volume
##                # Fill nan values, i.e., simulations that failed, with the max run
##                #  as this is due to a glacier exceeding the original bounds (i.e., positive gain)
##                #  note: this should have limited impact as this happens to very few runs
##                nan_col_idx = np.where(np.isnan(glac_vol[0,:]))[0]
##                if len(nan_col_idx) > 0:
##                    max_vol_idx = np.where(glac_vol[-1,:] == np.nanmax(glac_vol[-1,:]))[0][0]
##                    glac_vol_annual_max = glac_vol[:,max_vol_idx]
##                    glac_vol[:,nan_col_idx] = glac_vol_annual_max[:,np.newaxis]
###                    glac_vol_annual_med = np.nanmedian(glac_vol, axis=1)
###                    glac_vol[:,nan_col_idx] = glac_vol_annual_med[:,np.newaxis]
##                        
###                # Check for any unrealistic major gains that are due to errors in code
###                glac_vol_dif = glac_vol[1:,:] - glac_vol[0:-1,:]
###                glac_vol_start_med = np.nanmedian(glac_vol[0,:])
###                # If glacier gains 10% of initial volume, then likely an error
###                dif_likely_error = np.where(glac_vol_dif > glac_vol_start_med/2)[0]
###                if len(dif_likely_error) > 0:
###                    print(nglac, glacno + ' may have error in the simulations')
##                    
##                # Combine to get regional dataset
##                if reg_vol is None:
##                    reg_vol = glac_vol
##                else:
##                    reg_vol += glac_vol
##                    
##                # Area
##                nan_col_idx_area = np.where(np.isnan(glac_area[0,:]))[0]
##                if len(nan_col_idx_area) > 0:
##                    glac_area_annual_max = glac_area[:,max_vol_idx]
##                    glac_area[:,nan_col_idx] = glac_area_annual_max[:,np.newaxis]
###                    glac_area_annual_med = np.nanmedian(glac_area, axis=1)
###                    glac_area[:,nan_col_idx_area] = glac_area_annual_med[:,np.newaxis]
##                    
##                # Record data
##                glac_mass = glac_vol * pygem_prms.density_ice
##                glac_mass_mean = np.mean(glac_mass,axis=1)
##                mb_mod_gta_mean = ((glac_mass_mean[idx_cal_endyr] - glac_mass_mean[idx_cal_startyr]) / 
##                                   (cal_endyr - cal_startyr) / 1e12)
##                mb_mod_mwea_mean = mb_mod_gta_mean * 1e12 / pygem_prms.density_water / glac_area[0,0]
##                
##                # MCMC calibration
##                mcmc_fp = cal_fp + str(reg).zfill(2) + '/'
##                glacier_str = glacno
##                modelprms_fn = glacier_str + '-modelprms_dict.pkl'
##                modelprms_fp = (pygem_prms.output_filepath + 'calibration/' + glacier_str.split('.')[0].zfill(2) 
##                                + '/')
##                modelprms_fullfn = modelprms_fp + modelprms_fn
##
##                assert os.path.exists(modelprms_fullfn), 'Calibrated parameters do not exist.'
##                with open(modelprms_fullfn, 'rb') as f:
##                    modelprms_dict = pickle.load(f)
##
##
##
##                mb_compare.loc[nglac,'mb_mwea_emulator'] = modelprms_dict['emulator']['mb_mwea'][0]
##                mb_compare.loc[nglac,'mb_mwea_mcmc'] = np.mean(modelprms_dict['MCMC']['mb_mwea']['chain_0'])
##                mb_compare.loc[nglac,'mb_mwea_oggm'] = mb_mod_mwea_mean
##                mb_compare.loc[nglac,'mb_gta_emulator'] = (
##                        mwea_to_gta(modelprms_dict['emulator']['mb_mwea'][0], glac_area[0,0]))
##                mb_compare.loc[nglac,'mb_gta_mcmc'] = (
##                        mwea_to_gta(np.mean(modelprms_dict['MCMC']['mb_mwea']['chain_0']), glac_area[0,0]))
##                mb_compare.loc[nglac,'mb_gta_oggm'] = mwea_to_gta(mb_mod_mwea_mean,glac_area[0,0])
##
##                # Combine to get regional dataset
##                if reg_area is None:
##                    reg_area = glac_area
##                else:
##                    reg_area += glac_area
##                    
###            plt.plot(years, reg_vol)
###            plt.ylabel('Volume [m3]')
###            plt.show()
##                  
##            # Pickle the dataset
##            with open(pickle_fp + reg_vol_fn, 'wb') as f:
##                pickle.dump(reg_vol, f)
##            with open(pickle_fp + reg_area_fn, 'wb') as f:
##                pickle.dump(reg_area, f)
##                
##            # Export csv
##            mb_compare.to_csv(mb_compare_fullfn, index=False)
##                    
##        else:
##            
##            with open(pickle_fp + reg_vol_fn, 'rb') as f:
##                reg_vol = pickle.load(f)
##            with open(pickle_fp + reg_area_fn, 'rb') as f:
##                reg_area = pickle.load(f)
##                
##            mb_compare = pd.read_csv(mb_compare_fullfn)
##            
##            # Load years
##            if reg_vol.shape[0] == 21:
##                years = np.arange(2000,2021)
##           
##        print(mb_compare.mb_gta_emulator.sum())
##        print(mb_compare.mb_gta_mcmc.sum())
##        print(mb_compare.mb_gta_oggm.sum())
##        
##        print(list(main_glac_rgi_missing.glacno.values))




#%%
if option_get_missing_glacno:
    """ Get list of missing glaciers for each rcp scenario! """
    for reg in regions:
        
#        # Load glaciers
#        glacno_list = {}
#        for gcm in gcm_names:
#            glacno_list[gcm] = {}
#            for rcp in rcps:
#                # Filepath where glaciers are stored
#                fail_fp = netcdf_fp_cmip5 + 'failed/' + gcm + '/' + rcp + '/'
#                
#                glacno_list_gcmrcp = []
#                for i in os.listdir(fail_fp):
#                    if i.endswith('-sim_failed.txt'):
#                        glacno_list_gcmrcp.append(i.split('-')[0])
#                        
#                glacno_list[gcm][rcp] = sorted(glacno_list_gcmrcp)
        
        
        # All glaciers for fraction
        main_glac_rgi_all = modelsetup.selectglaciersrgitable(rgi_regionsO1=[reg], 
                                                              rgi_regionsO2='all', rgi_glac_number='all', 
                                                              glac_no=None)
           
        # ----- Missing glaciers -----
        # Filepath where glaciers are stored
        # Load the glaciers
        glacno_list_fp = '/Users/drounce/Documents/HiMAT/spc_backup/calibration/' + str(reg).zfill(2) + '/'
        glacno_list = []
        for i in os.listdir(glacno_list_fp):
            if i.endswith('.pkl'):
                glacno_list.append(i.split('-')[0])
        glacno_list = sorted(glacno_list)
        
        glacno_list_all = list(main_glac_rgi_all.glacno.values)
        
        glacno_missing = np.setdiff1d(glacno_list_all, glacno_list).tolist()
        
        main_glac_rgi_missing = modelsetup.selectglaciersrgitable(glac_no=glacno_missing)
        print(reg, main_glac_rgi_missing.Area.sum() / main_glac_rgi_all.Area.sum() * 100, '% missing by area')
                        

#%%
if option_plot_era5_volchange:
    
    overwrite_pickle = True
    
    # Input information for analysis    
    cal_startyr = 2000
    cal_endyr = 2020
    
    grouping = 'all'
    mb_dataset = 'Hugonnet2020'
    
    #%%
#    cal_fp = pygem_prms.output_filepath + 'calibration/'
    cal_fp = '/Users/drounce/Documents/HiMAT/spc_backup/calibration/'
    netcdf_fn_ending = '_ERA5_MCMC_ba1_50sets_2000_2019_all.nc'
#    netcdf_fn_ending = '_ERA5_emulator_ba1_1sets_2000_2019_all.nc'
    #%%
    fig_fp = netcdf_fp_sims + '/../analysis/figures/'
    if not os.path.exists(fig_fp):
        os.makedirs(fig_fp, exist_ok=True)
    csv_fp = netcdf_fp_sims + '/../analysis/csv/'
    if not os.path.exists(csv_fp):
        os.makedirs(csv_fp, exist_ok=True)
    pickle_fp = fig_fp + '../pickle/'
    if not os.path.exists(pickle_fp):
        os.makedirs(pickle_fp, exist_ok=True)
        
    def mwea_to_gta(mwea, area):
        return mwea * pygem_prms.density_water * area / 1e12

    for reg in regions:
        
        # Load glaciers
        glacno_list = []
        # Filepath where glaciers are stored
        netcdf_fp = netcdf_fp_sims + str(reg).zfill(2) + '/ERA5/binned/'
        netcdf_fp_stats = netcdf_fp_sims + str(reg).zfill(2) + '/ERA5/stats/'
        for i in os.listdir(netcdf_fp):
            if i.endswith('.nc'):
                glacno_list.append(i.split('_')[0])
        glacno_list = sorted(glacno_list)
        
        print('\n\nLimiting by calibration files too\n\n')
        glacno_list_cal = []
        for i in os.listdir(cal_fp + str(reg).zfill(2) + '/'):
            if i.endswith('.pkl'):
                glacno_list_cal.append(i.split('-')[0])
        glacno_list_cal = sorted(glacno_list_cal)
        
        glacno_list = list(set(glacno_list).intersection(glacno_list_cal))
        glacno_list = sorted(glacno_list)
        
#        print('\n\nDELETE ME!\n\n')
#        glacno_list = glacno_list_cal
        
        #%%
        glacno_missing_wcal = list(set(glacno_list_cal).intersection(glacno_list_missing))
        #%%
        
        
        print('simulated', len(glacno_list), 'glaciers')
        
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
        
        # Export data of count and mass balance
        print('\nERA5 successfully simulated:\n  -', main_glac_rgi.shape[0], 'of', main_glac_rgi_all.shape[0], 'glaciers',
              '(', np.round(main_glac_rgi.shape[0]/main_glac_rgi_all.shape[0]*100,1),'%)')
        print('  -', np.round(main_glac_rgi.Area.sum(),1), 'km2 of', np.round(main_glac_rgi_all.Area.sum(),1), 'km2',
              '(', np.round(main_glac_rgi.Area.sum()/main_glac_rgi_all.Area.sum()*100,1),'%)')
        
        # ===== EXPORT RESULTS =====
        success_fullfn = csv_fp + 'ERA5_success.csv'
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
        # ===== MASS BALANCE COMPARISON =====
        mb_compare_fullfn = csv_fp + str(reg).zfill(2) + '-mb_compare_glac-ERA5.csv'
        
        # Load mass balance data
        # Observed mass loss
        if mb_dataset in ['Hugonnet2020']:
            mbdata_fp = pygem_prms.hugonnet_fp
            mbdata_fn = pygem_prms.hugonnet_fn
            rgiid_cn = pygem_prms.hugonnet_rgi_glacno_cn
            mb_cn = pygem_prms.hugonnet_mb_cn
            mberr_cn = pygem_prms.hugonnet_mb_err_cn
            t1_cn = pygem_prms.hugonnet_time1_cn
            t2_cn = pygem_prms.hugonnet_time2_cn
        
        assert os.path.exists(mbdata_fp + mbdata_fn), "Error: mb dataset does not exist."

        mb_df_all = pd.read_csv(mbdata_fp + mbdata_fn)
        mb_df_all_rgiids = list(mb_df_all[rgiid_cn])

        rgiids = list(main_glac_rgi.RGIId.values)
        mb_df_idx = [mb_df_all_rgiids.index(x) for x in rgiids]
        mb_df = mb_df_all.loc[mb_df_idx,:]
        mb_df = mb_df.sort_values('RGIId', ascending=True)
        mb_df.reset_index(inplace=True, drop=True)
        # gt/yr = mb_mwea * area_km2 * (1e6 m2 / 1 km2) * (1000 kg / 1 m3) * (1 Gt / 1e12 kg)
        mb_df['mb_gta'] = mb_df['mb_mwea'] * mb_df['area'] * 1e6 * pygem_prms.density_water / 1e12

        #%% Load model data
        reg_vol_fn = 'R' + str(reg) + '_ERA5_volume_annual.pkl'
        reg_area_fn = 'R' + str(reg) + '_ERA5_area_annual.pkl'
                
        if not os.path.exists(pickle_fp + reg_vol_fn) or overwrite_pickle:
            
            # MB comparison
            mb_compare_cns = ['RGIId', 'area_km2', 'mb_obs_mwea', 'mb_obs_mwea_err', 
                              'mb_mwea_emulator', 'mb_mwea_mcmc', 'mb_mwea_oggm',
                              'mb_obs_gta', 'mb_gta_emulator', 'mb_gta_mcmc', 'mb_gta_oggm']
            mb_compare = pd.DataFrame(np.zeros((main_glac_rgi.shape[0],len(mb_compare_cns))), columns=mb_compare_cns)
            mb_compare['RGIId'] = main_glac_rgi['RGIId']
            mb_compare['area_km2'] = main_glac_rgi['Area']       
            mb_compare['mb_obs_mwea'] = mb_df['mb_mwea']
            mb_compare['mb_obs_mwea_err'] = mb_df['mb_mwea_err']
            mb_compare['mb_obs_gta'] = mb_df['mb_gta']
            
            years = None
            reg_vol = None
            reg_area = None
            for nglac, glacno in enumerate(main_glac_rgi.glacno.values):
#            for nglac, glacno in enumerate(main_glac_rgi.glacno.values[0:1]):
                
                if nglac%500==0:
                    print(nglac, glacno)
                
                ds_fn = glacno + netcdf_fn_ending
                        
                ds = xr.open_dataset(netcdf_fp_stats + ds_fn)

                # Time values
                if years is None:
                    years = ds.year.values
                    idx_cal_startyr = np.where(years == cal_startyr)[0][0]
                    idx_cal_endyr = np.where(years == cal_endyr)[0][0]

                # Volume data
                glac_vol = ds.glac_volume_annual.values[0,:]
                glac_area = ds.glac_area_annual.values[0,:]

                # Volume
                # Fill nan values, i.e., simulations that failed, with the max run
                #  as this is due to a glacier exceeding the original bounds (i.e., positive gain)
                #  note: this should have limited impact as this happens to very few runs
                nan_col_idx = np.where(np.isnan(glac_vol))[0]
                if len(nan_col_idx) > 0:
                    assert True==False, 'This is broken; needs to be fixed'
                    max_vol_idx = np.where(glac_vol[-1,:] == np.nanmax(glac_vol[-1,:]))[0][0]
                    glac_vol_annual_max = glac_vol[:,max_vol_idx]
                    glac_vol[:,nan_col_idx] = glac_vol_annual_max[:,np.newaxis]
#                    glac_vol_annual_med = np.nanmedian(glac_vol, axis=1)
#                    glac_vol[:,nan_col_idx] = glac_vol_annual_med[:,np.newaxis]

#                # Check for any unrealistic major gains that are due to errors in code
#                glac_vol_dif = glac_vol[1:,:] - glac_vol[0:-1,:]
#                glac_vol_start_med = np.nanmedian(glac_vol[0,:])
#                # If glacier gains 10% of initial volume, then likely an error
#                dif_likely_error = np.where(glac_vol_dif > glac_vol_start_med/2)[0]
#                if len(dif_likely_error) > 0:
#                    print(nglac, glacno + ' may have error in the simulations')
                    
                # Combine to get regional dataset
                if reg_vol is None:
                    reg_vol = glac_vol
                else:
                    reg_vol += glac_vol
                    
                # Area
                nan_col_idx_area = np.where(np.isnan(glac_area))[0]
                if len(nan_col_idx_area) > 0:
                    assert True==False, 'This is broken; needs to be fixed'
                    glac_area_annual_max = glac_area[:,max_vol_idx]
                    glac_area[:,nan_col_idx] = glac_area_annual_max[:,np.newaxis]
#                    glac_area_annual_med = np.nanmedian(glac_area, axis=1)
#                    glac_area[:,nan_col_idx_area] = glac_area_annual_med[:,np.newaxis]
                    
                # Record data
                glac_mass = glac_vol * pygem_prms.density_ice
                
                mb_mod_gta = ((glac_mass[idx_cal_endyr] - glac_mass[idx_cal_startyr]) / 
                                   (cal_endyr - cal_startyr) / 1e12)
                mb_mod_mwea_mean = mb_mod_gta * 1e12 / pygem_prms.density_water / glac_area[0]
                
                # MCMC calibration
                glacier_str = glacno
                modelprms_fn = glacier_str + '-modelprms_dict.pkl'
                modelprms_fp = cal_fp + glacier_str.split('.')[0].zfill(2)  + '/'
                modelprms_fullfn = modelprms_fp + modelprms_fn

                assert os.path.exists(modelprms_fullfn), 'Calibrated parameters do not exist.'
                with open(modelprms_fullfn, 'rb') as f:
                    modelprms_dict = pickle.load(f)
                
                mb_compare.loc[nglac,'mb_mwea_emulator'] = modelprms_dict['emulator']['mb_mwea'][0]
                mb_compare.loc[nglac,'mb_mwea_mcmc'] = np.mean(modelprms_dict['MCMC']['mb_mwea']['chain_0'])
#                mb_compare.loc[nglac,'mb_mwea_oggm'] = mb_mod_mwea_mean
                mb_compare.loc[nglac,'mb_gta_emulator'] = (
                        mwea_to_gta(mb_compare.loc[nglac,'mb_mwea_emulator'], main_glac_rgi.loc[nglac,'Area']*1e6))
                mb_compare.loc[nglac,'mb_gta_mcmc'] = (
                        mwea_to_gta(np.mean(modelprms_dict['MCMC']['mb_mwea']['chain_0']), main_glac_rgi.loc[nglac,'Area']*1e6))
                mb_compare.loc[nglac,'mb_gta_oggm'] = mwea_to_gta(mb_mod_mwea_mean,glac_area[0])

                # Combine to get regional dataset
                if reg_area is None:
                    reg_area = glac_area
                else:
                    reg_area += glac_area
                  
            # Pickle the dataset
            with open(pickle_fp + reg_vol_fn, 'wb') as f:
                pickle.dump(reg_vol, f)
            with open(pickle_fp + reg_area_fn, 'wb') as f:
                pickle.dump(reg_area, f)
                
            # Export csv
            #%%
            mb_compare['dif_gta_obs_oggm'] = mb_compare['mb_obs_gta'] - mb_compare['mb_gta_oggm']
            mb_compare['dif_gta_obs_em'] = mb_compare['mb_obs_gta'] - mb_compare['mb_gta_emulator']
            mb_compare['dif_gta_oggm_em'] = mb_compare['mb_gta_oggm'] - mb_compare['mb_gta_emulator']
            #%%
            mb_compare.to_csv(mb_compare_fullfn, index=False)
                    
        else:
            
            with open(pickle_fp + reg_vol_fn, 'rb') as f:
                reg_vol = pickle.load(f)
            with open(pickle_fp + reg_area_fn, 'rb') as f:
                reg_area = pickle.load(f)
                
            mb_compare = pd.read_csv(mb_compare_fullfn)
            
            # Load years
            if reg_vol.shape[0] == 21:
                years = np.arange(2000,2021)
           
        #%%
        print('obs  [gta]:', mb_compare.mb_obs_gta.sum())
        print('em   [gta]:', mb_compare.mb_gta_emulator.sum())
        print('mcmc [gta]:', mb_compare.mb_gta_mcmc.sum())
        print('oggm [gta]:', mb_compare.mb_gta_oggm.sum())
        #%%
        print(list(main_glac_rgi_missing.glacno.values))
              #%%
            
#        # ----- MASS LOSS COMPARISON -----
#        # Modeled mass loss
#        reg_mass = reg_vol * pygem_prms.density_ice
#        reg_mass_med = np.median(reg_mass, axis=0)
#        reg_mass_mean = np.mean(reg_mass, axis=0)
#        idx_cal_startyr = np.where(years == cal_startyr)[0][0]
#        idx_cal_endyr = np.where(years == cal_endyr)[0][0]
#        # mass loss Gt/yr
#        #  units: kg / yrs * (1 Gt / 1e12 kg)
#        reg_mb_gta_2000_2020_med = ((reg_mass_med[idx_cal_endyr] - reg_mass_med[idx_cal_startyr]) / (cal_endyr - cal_startyr) 
#                                    / 1e12)
#        reg_mb_gta_2000_2020_mean = ((reg_mass_mean[idx_cal_endyr] - reg_mass_mean[idx_cal_startyr]) / 
#                                     (cal_endyr - cal_startyr) / 1e12)
#        
#        print(reg, 'Mass change med [gt/yr]:', np.round(reg_mb_gta_2000_2020_med,1))
#        print(reg, 'Mass change mean [gt/yr]:', np.round(reg_mb_gta_2000_2020_mean,1))
#        
#        
#  
#        #%%
#
##        #%%
##        # ----- FIGURE: VOLUME CHANGE FOR EACH GCM ----- 
##        fig, ax = plt.subplots(1, 1, squeeze=False, sharex=False, sharey=True, 
##                               gridspec_kw = {'wspace':0, 'hspace':0})
##        
##        # Load data
##        for rcp in rcps:
##            
##            for ngcm, gcm_name in enumerate(gcm_names):
##                
##                if ngcm == 0:
##                    label=rcp
##                else:
##                    label=None
##                
##                # Median and absolute median deviation
##                reg_vol = reg_vol_all[rcp][gcm_name]
##                reg_vol_med = np.median(reg_vol, axis=1)
##                reg_vol_mad = median_abs_deviation(reg_vol, axis=1)
##            
##                ax[0,0].plot(years, reg_vol_med, color=rcp_colordict[rcp], linewidth=1, zorder=4, label=label)
##                ax[0,0].fill_between(years, 
##                                     reg_vol_med + 1.96*reg_vol_mad, 
##                                     reg_vol_med - 1.96*reg_vol_mad, 
##                                     alpha=0.2, facecolor=rcp_colordict[rcp], label=None)
##                
##        ax[0,0].set_ylabel('Volume (m$^{3}$)')
##        ax[0,0].set_xlim(startyear, endyear)
##        ax[0,0].text(0.98, 1.06, rgi_reg_dict[reg], size=10, horizontalalignment='right', 
##                     verticalalignment='top', transform=ax[0,0].transAxes)
##        ax[0,0].legend(
###                rcp_lines, rcp_labels, loc=(0.05,0.05), fontsize=10, labelspacing=0.25, handlelength=1, 
###                handletextpad=0.25, borderpad=0, frameon=False
##                )        
##        ax[0,0].tick_params(direction='inout', right=True)
##        # Save figure
##        fig_fn = str(reg) + '_volchange_' + str(startyear) + '-' + str(endyear) + '_all_gcmrcps.png'
##        fig.set_size_inches(4,3)
##        fig.savefig(fig_fp + fig_fn, bbox_inches='tight', dpi=300)
##        
##        #%%
##        # ----- FIGURE: NORMALIZED VOLUME CHANGE MULTI-GCM -----
##        fig, ax = plt.subplots(1, 1, squeeze=False, sharex=False, sharey=True, 
##                               gridspec_kw = {'wspace':0, 'hspace':0})
##
##        normyear_idx = np.where(years == normyear)[0][0]
##
##        for rcp in rcps:
##            
##            # Median and absolute median deviation
##            reg_vol = ds_multigcm[rcp]
##            reg_vol_med = np.median(reg_vol, axis=0)
##            reg_vol_mad = median_abs_deviation(reg_vol, axis=0)
##            
##            reg_vol_med_norm = reg_vol_med / reg_vol_med[normyear_idx]
##            reg_vol_mad_norm = reg_vol_mad / reg_vol_med[normyear_idx]
##            
##            ax[0,0].plot(years, reg_vol_med_norm, color=rcp_colordict[rcp], linewidth=1, zorder=4, label=rcp)
##            
##            if rcp in rcps_plot_mad:
##                ax[0,0].fill_between(years, 
##                                     reg_vol_med_norm + 1.96*reg_vol_mad_norm, 
##                                     reg_vol_med_norm - 1.96*reg_vol_mad_norm, 
##                                     alpha=0.2, facecolor=rcp_colordict[rcp], label=None)
##        
##        ax[0,0].set_ylabel('Volume (-)')
##        ax[0,0].set_xlim(startyear, endyear)
##        ax[0,0].text(0.98, 1.06, rgi_reg_dict[reg], size=10, horizontalalignment='right', 
##                     verticalalignment='top', transform=ax[0,0].transAxes)
##        ax[0,0].legend(
###                rcp_lines, rcp_labels, loc=(0.05,0.05), fontsize=10, labelspacing=0.25, handlelength=1, 
###                handletextpad=0.25, borderpad=0, frameon=False
##                )
##        ax[0,0].tick_params(direction='inout', right=True)
##        # Save figure
##        fig_fn = str(reg) + '_volchangenorm_' + str(startyear) + '-' + str(endyear) + '_multigcm.png'
##        fig.set_size_inches(4,3)
##        fig.savefig(fig_fp + fig_fn, bbox_inches='tight', dpi=300)
##        
##        
##        #%%
##        # ----- FIGURE: AREA CHANGE MULTI-GCM -----
##        fig, ax = plt.subplots(1, 1, squeeze=False, sharex=False, sharey=True, 
##                               gridspec_kw = {'wspace':0, 'hspace':0})
##
##        for rcp in rcps:
##            
##            # Median and absolute median deviation
##            reg_area = ds_multigcm_area[rcp]
##            reg_area_med = np.median(reg_area, axis=0)
##            reg_area_mad = median_abs_deviation(reg_area, axis=0)
##            
##            ax[0,0].plot(years, reg_area_med, color=rcp_colordict[rcp], linewidth=1, zorder=4, label=rcp)
##            if rcp in rcps_plot_mad:
##                ax[0,0].fill_between(years, 
##                                     reg_area_med + 1.96*reg_area_mad, 
##                                     reg_area_med - 1.96*reg_area_mad, 
##                                     alpha=0.2, facecolor=rcp_colordict[rcp], label=None)
##               
##        ax[0,0].set_ylabel('Area (m$^{2}$)')
##        ax[0,0].set_xlim(startyear, endyear)
##        ax[0,0].text(0.98, 1.06, rgi_reg_dict[reg], size=10, horizontalalignment='right', 
##                     verticalalignment='top', transform=ax[0,0].transAxes)
##        ax[0,0].legend(
###                rcp_lines, rcp_labels, loc=(0.05,0.05), fontsize=10, labelspacing=0.25, handlelength=1, 
###                handletextpad=0.25, borderpad=0, frameon=False
##                )
##        ax[0,0].tick_params(direction='inout', right=True)
##        # Save figure
##        fig_fn = str(reg) + '_areachange_' + str(startyear) + '-' + str(endyear) + '_multigcm.png'
##        fig.set_size_inches(4,3)
##        fig.savefig(fig_fp + fig_fn, bbox_inches='tight', dpi=300)










             
                
                
                
#%%
if option_plot_cmip5_volchange:
    
    overwrite_pickle = True
    
    # Input information for analysis
    startyear = 2000
    endyear = 2019
    normyear = 2015
    
    grouping = 'all'
    option_plot_individual_gcms = 0
    rcps_plot_mad = ['rcp26', 'rcp45', 'rcp85']
    
    netcdf_fn_ending = '_MCMC_ba1_50sets_2000_2100_annual.nc'

    fig_fp = pygem_prms.main_directory + '/../Output/analysis/figures/'
    if not os.path.exists(fig_fp):
        os.makedirs(fig_fp, exist_ok=True)
    pickle_fp = fig_fp + '../pickle/'
    if not os.path.exists(pickle_fp):
        os.makedirs(pickle_fp, exist_ok=True)

    for reg in regions:
        
        # Load glaciers
        glacno_list = []
        for rcp in rcps:
            for gcm_name in gcm_names:
                
                # Filepath where glaciers are stored
                netcdf_fp = netcdf_fp_cmip5 + gcm_name + '/' + rcp + '/stats/'
                
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


        #%%
        # Load data
        reg_vol_all = {}
        reg_area_all = {}
        for rcp in rcps:
            reg_vol_all[rcp] = {}
            reg_area_all[rcp] = {}
            
            for ngcm, gcm_name in enumerate(gcm_names):
                
                print(rcp, gcm_name)
                
                reg_vol_fn = 'R' + str(reg) + '_' + gcm_name + '_' + rcp + '_volume_annual.pkl'
                reg_area_fn = 'R' + str(reg) + '_' + gcm_name + '_' + rcp + '_area_annual.pkl'
                
                if not os.path.exists(pickle_fp + reg_vol_fn) or overwrite_pickle:
#                for batman in [0]:
            
                    # Load datasets
                    netcdf_fp = netcdf_fp_cmip5 + gcm_name + '/' + rcp + '/stats/'

                    years = None
                    reg_vol = None
                    reg_area = None
                    for nglac, glacno in enumerate(main_glac_rgi.glacno.values):
#                    for nglac, glacno in enumerate(main_glac_rgi.glacno.values[15640:15641]):
                        
                        if nglac%500==0:
                            print(nglac, glacno)
                        
                        ds_fn = glacno + '_' + gcm_name + '_' + rcp + netcdf_fn_ending
                        
                        ds = xr.open_dataset(netcdf_fp + ds_fn)
                        #%%
                        # Time values
                        if years is None:
                            years = ds.year.values
                        
                        # Volume data
                        glac_vol = ds.glac_volume_annual.values[0,:,:]
                        glac_area = ds.glac_area_annual.values[0,:,:]
                        
                        
#                        plt.plot(years, glac_vol)
#                        plt.ylabel('Volume [m3]')
#                        plt.show()
                        
#                        # Median and absolute median deviation
#                        glac_mass = glac_vol * pygem_prms.density_ice
#                        glac_mass_med = np.median(glac_mass, axis=1)
#            
#                        cal_startyr = 2000
#                        cal_endyr = 2020
#                        idx_cal_startyr = np.where(years == cal_startyr)[0][0]
#                        idx_cal_endyr = np.where(years == cal_endyr)[0][0]
#                        
#                        # mass loss Gt/yr
#                        #  units: kg / yrs * (1 Gt / 1e12 kg)
#                        reg_mb_gta_2000_2020 = ((glac_mass_med[idx_cal_endyr] - glac_mass_med[idx_cal_startyr]) / (cal_endyr - cal_startyr)
#                                                / 1e12)
#                        #  units: kg / yrs / 
#                        glac_idx = np.where(main_glac_rgi.glacno.values==glacno)[0][0]
#                        glac_area = main_glac_rgi.loc[glac_idx,'Area'] * 1e6
#                        reg_mb_mwea_2000_2020 = ((glac_mass_med[idx_cal_endyr] - glac_mass_med[idx_cal_startyr]) / (cal_endyr - cal_startyr)
#                                                / pygem_prms.density_water / glac_area)
#                        
#                        print(reg, rcp, 'Mass change [gt/yr]:', np.round(reg_mb_gta_2000_2020,1))
#                        print(reg, rcp, 'Mass balance [mwea]:', np.round(reg_mb_mwea_2000_2020,3))
                    
                    #%%
#                        A = -0.76 * 86725053000 * pygem_prms.density_water / 1e12
#                        print('  Cal Mass change [gt/yr]:', np.round(A,1))
                        
                    
                    #%%
                        # Volume
                        # Fill nan values, i.e., simulations that failed, with the median
                        #  note: this should have limited impact as this happens to very few runs
                        #  however, it will reduce the nmad
                        nan_col_idx = np.where(np.isnan(glac_vol[0,:]))[0]
                        if len(nan_col_idx) > 0:
                            glac_vol_annual_med = np.nanmedian(glac_vol, axis=1)
                            glac_vol[:,nan_col_idx] = glac_vol_annual_med[:,np.newaxis]
                                
                        # Check for any unrealistic major gains that are due to errors in code
                        glac_vol_dif = glac_vol[1:,:] - glac_vol[0:-1,:]
                        glac_vol_start_med = np.nanmedian(glac_vol[0,:])
                        # If glacier gains 10% of initial volume, then likely an error
                        dif_likely_error = np.where(glac_vol_dif > glac_vol_start_med/2)[0]
                        if len(dif_likely_error) > 0:
                            print(nglac, glacno + ' may have error in the simulations')
                            
                        # Combine to get regional dataset
                        if reg_vol is None:
                            reg_vol = glac_vol
                        else:
                            reg_vol += glac_vol
                            
                            
                        # Area
                        nan_col_idx_area = np.where(np.isnan(glac_area[0,:]))[0]
                        if len(nan_col_idx_area) > 0:
                            glac_area_annual_med = np.nanmedian(glac_area, axis=1)
                            glac_area[:,nan_col_idx_area] = glac_area_annual_med[:,np.newaxis]
                            
                        # Combine to get regional dataset
                        if reg_area is None:
                            reg_area = glac_area
                        else:
                            reg_area += glac_area
                          
                    # Pickle the dataset
                    with open(pickle_fp + reg_vol_fn, 'wb') as f:
                        pickle.dump(reg_vol, f)
                    with open(pickle_fp + reg_area_fn, 'wb') as f:
                        pickle.dump(reg_area, f)
                        
                else:
                    
                    with open(pickle_fp + reg_vol_fn, 'rb') as f:
                        reg_vol = pickle.load(f)
                    with open(pickle_fp + reg_area_fn, 'rb') as f:
                        reg_area = pickle.load(f)
                    
                    # Load years
                    if reg_vol.shape[0] == 102:
                        years = np.arange(2000,2102)
                        
                reg_vol_all[rcp][gcm_name] = reg_vol
                reg_area_all[rcp][gcm_name] = reg_area
                    
        
        # MULTI-GCM VOLUME CHANGE (medians of volume change)
        ds_multigcm = {}
        for rcp in rcps:            
            for ngcm, gcm_name in enumerate(gcm_names):
                
                print(rcp, gcm_name)

                reg_vol_gcm = reg_vol_all[rcp][gcm_name]
                reg_vol_gcm_med = np.median(reg_vol_gcm, axis=1)

                if ngcm == 0:
                    reg_vol_gcm_all = reg_vol_gcm_med               
                else:
                    reg_vol_gcm_all = np.vstack((reg_vol_gcm_all, reg_vol_gcm_med))
            
            ds_multigcm[rcp] = reg_vol_gcm_all
           
            
        # MULTI-GCM AREA CHANGE (medians of area change)
        ds_multigcm_area = {}
        for rcp in rcps:            
            for ngcm, gcm_name in enumerate(gcm_names):
                
                print(rcp, gcm_name)

                reg_area_gcm = reg_area_all[rcp][gcm_name]
                reg_area_gcm_med = np.median(reg_area_gcm, axis=1)

                if ngcm == 0:
                    reg_area_gcm_all = reg_area_gcm_med               
                else:
                    reg_area_gcm_all = np.vstack((reg_area_gcm_all, reg_area_gcm_med))
            
            ds_multigcm_area[rcp] = reg_area_gcm_all
            
        #%%
        # ----- Check regional mass balance -----
        for rcp in rcps:
            
            # Median and absolute median deviation
            reg_mass = ds_multigcm[rcp] * pygem_prms.density_ice
            reg_mass_med = np.median(reg_mass, axis=0)

            cal_startyr = 2000
            cal_endyr = 2020
            idx_cal_startyr = np.where(years == cal_startyr)[0][0]
            idx_cal_endyr = np.where(years == cal_endyr)[0][0]
            
            # mass loss Gt/yr
            #  units: kg / yrs * (1 Gt / 1e12 kg)
            reg_mb_gta_2000_2020 = ((reg_mass_med[idx_cal_endyr] - reg_mass_med[idx_cal_startyr]) / (cal_endyr - cal_startyr)
                                    / 1e12)
            
            print(reg, rcp, 'Mass change [gt/yr]:', np.round(reg_mb_gta_2000_2020,1))
        
        A = -0.76 * main_glac_rgi.Area.sum() * 1e6 * pygem_prms.density_water / 1e12
        print('  Cal Mass change [gt/yr]:', np.round(A,1))
        
        #%%
        # ----- FIGURE: VOLUME CHANGE FOR EACH GCM ----- 
        fig, ax = plt.subplots(1, 1, squeeze=False, sharex=False, sharey=True, 
                               gridspec_kw = {'wspace':0, 'hspace':0})
        
        # Load data
        for rcp in rcps:
            
            for ngcm, gcm_name in enumerate(gcm_names):
                
                if ngcm == 0:
                    label=rcp
                else:
                    label=None
                
                # Median and absolute median deviation
                reg_vol = reg_vol_all[rcp][gcm_name]
                reg_vol_med = np.median(reg_vol, axis=1)
                reg_vol_mad = median_abs_deviation(reg_vol, axis=1)
            
                ax[0,0].plot(years, reg_vol_med, color=rcp_colordict[rcp], linewidth=1, zorder=4, label=label)
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
        fig_fn = str(reg) + '_volchange_' + str(startyear) + '-' + str(endyear) + '_all_gcmrcps.png'
        fig.set_size_inches(4,3)
        fig.savefig(fig_fp + fig_fn, bbox_inches='tight', dpi=300)
                    
    
        #%%
        # ----- FIGURE: VOLUME CHANGE MULTI-GCM -----
        fig, ax = plt.subplots(1, 1, squeeze=False, sharex=False, sharey=True, 
                               gridspec_kw = {'wspace':0, 'hspace':0})

        for rcp in rcps:
            
            # Median and absolute median deviation
            reg_vol = ds_multigcm[rcp]
            reg_vol_med = np.median(reg_vol, axis=0)
            reg_vol_mad = median_abs_deviation(reg_vol, axis=0)
            
            ax[0,0].plot(years, reg_vol_med, color=rcp_colordict[rcp], linewidth=1, zorder=4, label=rcp)
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
        fig_fn = str(reg) + '_volchange_' + str(startyear) + '-' + str(endyear) + '_multigcm.png'
        fig.set_size_inches(4,3)
        fig.savefig(fig_fp + fig_fn, bbox_inches='tight', dpi=300)
        
        #%%
        # ----- FIGURE: NORMALIZED VOLUME CHANGE MULTI-GCM -----
        fig, ax = plt.subplots(1, 1, squeeze=False, sharex=False, sharey=True, 
                               gridspec_kw = {'wspace':0, 'hspace':0})

        normyear_idx = np.where(years == normyear)[0][0]

        for rcp in rcps:
            
            # Median and absolute median deviation
            reg_vol = ds_multigcm[rcp]
            reg_vol_med = np.median(reg_vol, axis=0)
            reg_vol_mad = median_abs_deviation(reg_vol, axis=0)
            
            reg_vol_med_norm = reg_vol_med / reg_vol_med[normyear_idx]
            reg_vol_mad_norm = reg_vol_mad / reg_vol_med[normyear_idx]
            
            ax[0,0].plot(years, reg_vol_med_norm, color=rcp_colordict[rcp], linewidth=1, zorder=4, label=rcp)
            
            if rcp in rcps_plot_mad:
                ax[0,0].fill_between(years, 
                                     reg_vol_med_norm + 1.96*reg_vol_mad_norm, 
                                     reg_vol_med_norm - 1.96*reg_vol_mad_norm, 
                                     alpha=0.2, facecolor=rcp_colordict[rcp], label=None)
        
        ax[0,0].set_ylabel('Volume (-)')
        ax[0,0].set_xlim(startyear, endyear)
        ax[0,0].text(0.98, 1.06, rgi_reg_dict[reg], size=10, horizontalalignment='right', 
                     verticalalignment='top', transform=ax[0,0].transAxes)
        ax[0,0].legend(
#                rcp_lines, rcp_labels, loc=(0.05,0.05), fontsize=10, labelspacing=0.25, handlelength=1, 
#                handletextpad=0.25, borderpad=0, frameon=False
                )
        ax[0,0].tick_params(direction='inout', right=True)
        # Save figure
        fig_fn = str(reg) + '_volchangenorm_' + str(startyear) + '-' + str(endyear) + '_multigcm.png'
        fig.set_size_inches(4,3)
        fig.savefig(fig_fp + fig_fn, bbox_inches='tight', dpi=300)
        
        
        #%%
        # ----- FIGURE: AREA CHANGE MULTI-GCM -----
        fig, ax = plt.subplots(1, 1, squeeze=False, sharex=False, sharey=True, 
                               gridspec_kw = {'wspace':0, 'hspace':0})

        for rcp in rcps:
            
            # Median and absolute median deviation
            reg_area = ds_multigcm_area[rcp]
            reg_area_med = np.median(reg_area, axis=0)
            reg_area_mad = median_abs_deviation(reg_area, axis=0)
            
            ax[0,0].plot(years, reg_area_med, color=rcp_colordict[rcp], linewidth=1, zorder=4, label=rcp)
            if rcp in rcps_plot_mad:
                ax[0,0].fill_between(years, 
                                     reg_area_med + 1.96*reg_area_mad, 
                                     reg_area_med - 1.96*reg_area_mad, 
                                     alpha=0.2, facecolor=rcp_colordict[rcp], label=None)
               
        ax[0,0].set_ylabel('Area (m$^{2}$)')
        ax[0,0].set_xlim(startyear, endyear)
        ax[0,0].text(0.98, 1.06, rgi_reg_dict[reg], size=10, horizontalalignment='right', 
                     verticalalignment='top', transform=ax[0,0].transAxes)
        ax[0,0].legend(
#                rcp_lines, rcp_labels, loc=(0.05,0.05), fontsize=10, labelspacing=0.25, handlelength=1, 
#                handletextpad=0.25, borderpad=0, frameon=False
                )
        ax[0,0].tick_params(direction='inout', right=True)
        # Save figure
        fig_fn = str(reg) + '_areachange_' + str(startyear) + '-' + str(endyear) + '_multigcm.png'
        fig.set_size_inches(4,3)
        fig.savefig(fig_fp + fig_fn, bbox_inches='tight', dpi=300)
        
        #%%
        # ----- FIGURE: NORMALIZED AREA CHANGE MULTI-GCM -----
        fig, ax = plt.subplots(1, 1, squeeze=False, sharex=False, sharey=True, 
                               gridspec_kw = {'wspace':0, 'hspace':0})

        normyear_idx = np.where(years == normyear)[0][0]

        for rcp in rcps:
            
            # Median and absolute median deviation
            reg_area = ds_multigcm_area[rcp]
            reg_area_med = np.median(reg_area, axis=0)
            reg_area_mad = median_abs_deviation(reg_area, axis=0)
            
            reg_area_med_norm = reg_area_med / reg_area_med[normyear_idx]
            reg_area_mad_norm = reg_area_mad / reg_area_med[normyear_idx]
            
            ax[0,0].plot(years, reg_area_med_norm, color=rcp_colordict[rcp], linewidth=1, zorder=4, label=rcp)
            
            if rcp in rcps_plot_mad:
                ax[0,0].fill_between(years, 
                                     reg_area_med_norm + 1.96*reg_area_mad_norm, 
                                     reg_area_med_norm - 1.96*reg_area_mad_norm, 
                                     alpha=0.2, facecolor=rcp_colordict[rcp], label=None)
        
        ax[0,0].set_ylabel('Area (-)')
        ax[0,0].set_xlim(startyear, endyear)
        ax[0,0].text(0.98, 1.06, rgi_reg_dict[reg], size=10, horizontalalignment='right', 
                     verticalalignment='top', transform=ax[0,0].transAxes)
        ax[0,0].legend(
#                rcp_lines, rcp_labels, loc=(0.05,0.05), fontsize=10, labelspacing=0.25, handlelength=1, 
#                handletextpad=0.25, borderpad=0, frameon=False
                )
        ax[0,0].tick_params(direction='inout', right=True)
        # Save figure
        fig_fn = str(reg) + '_areachangenorm_' + str(startyear) + '-' + str(endyear) + '_multigcm.png'
        fig.set_size_inches(4,3)
        fig.savefig(fig_fp + fig_fn, bbox_inches='tight', dpi=300)



#%%
if option_plot_era5_AAD:
    
    overwrite_pickle = True
    
    # Input information for analysis
    startyear = 2000
    endyear = 2100
    normyear = 2015
    
    grouping = 'all'
    
#    glacno = '1.15645'
#    glacno = '1.00570'
    glacno = '1.00037'
#    netcdf_binned_fp = '/Users/drounce/Documents/HiMAT/Output/simulations/ERA5/binned/'
    netcdf_binned_fp = '/Users/drounce/Documents/HiMAT/spc_backup/simulations/ERA5/binned/'
#    netcdf_binned_fp = '/Users/drounce/Documents/HiMAT/spc_backup/simulations/ERA5-cflp01/binned/'
    netcdf_binned_fn_ending = '_ERA5_MCMC_ba1_50sets_2000_2019_binned.nc'
    
#    netcdf_fp = '/Users/drounce/Documents/HiMAT/Output/simulations/ERA5/stats/'
    netcdf_fp = '/Users/drounce/Documents/HiMAT/spc_backup/simulations/ERA5/stats/'
#    netcdf_fp = '/Users/drounce/Documents/HiMAT/spc_backup/simulations/ERA5-cflp01/stats/'
    netcdf_fn_ending = '_ERA5_MCMC_ba1_50sets_2000_2019_all.nc'

    fig_fp = pygem_prms.main_directory + '/../Output/analysis/figures/'
    if not os.path.exists(fig_fp):
        os.makedirs(fig_fp, exist_ok=True)
    pickle_fp = fig_fp + '../pickle/'
    if not os.path.exists(pickle_fp):
        os.makedirs(pickle_fp, exist_ok=True)
    
    # Load dataset        
    ds_binned_fn = glacno + netcdf_binned_fn_ending
    ds_binned = xr.open_dataset(netcdf_binned_fp + ds_binned_fn)
    
    ds_fn = glacno + netcdf_fn_ending
    ds = xr.open_dataset(netcdf_fp + ds_fn)
    
    # Time values
    years = None
    if years is None:
        years = ds_binned.year.values
    nyears = years.shape[0] - 1

    # Volume data
    bin_vol = ds_binned.bin_volume_annual.values[0,:,:]
    bin_vol_t0 = bin_vol[:,0]
    bin_vol_end = bin_vol[:,-1]
    # Thickness data
    bin_thick = ds_binned.bin_thick_annual.values[0,:,:]
    bin_thick_t0 = bin_thick[:,0]
    bin_thick_end = bin_thick[:,-1]
    # Area data
    bin_area = np.zeros(bin_thick.shape)
    bin_area[bin_thick>0] = bin_vol[bin_thick>0] / bin_thick[bin_thick>0]
    bin_area_t0 = bin_area[:,0]
    bin_area_end = bin_area[:,-1]
    # Surface elevations
    surface_h = ds_binned.bin_surface_h_initial.values[0,:]

    # MB clim
    mb_clim = np.mean(ds_binned.bin_massbalclim_annual.values[0,:,:],axis=1)
    
    #%%
    # Isolate flux divergence (thickness change = mb_clim + flux _divergence)
    bin_flux = mb_clim - (bin_thick_end - bin_thick_t0) / nyears
    
    #%%
#    from pygem.oggm_compat import single_flowline_glacier_directory
#    # Glacier directory
#    glacier_str = glacno
#    gdir = single_flowline_glacier_directory(glacier_str, logging_level='CRITICAL')
#    # Flowlines
#    fls = gdir.read_pickle('inversion_flowlines')
    
#    plt.plot(years, np.mean(bin_area,axis=0))
#    plt.ylabel('Area [m2]')
#    plt.show()
    
    #%%
    fig, ax = plt.subplots(1, 4, squeeze=False, sharex=False, sharey=True, 
                           gridspec_kw = {'wspace':0, 'hspace':0})
    
    ymin, ymax = surface_h.min(), surface_h.max()
    
    
    # Volume altitude distribution
    ax[0,0].plot(bin_vol_t0, surface_h, linewidth=1, linestyle='-', label='t_start')   
    ax[0,0].plot(bin_vol_end, surface_h, linewidth=1, linestyle='--', label='t_end')   
    ax[0,0].set_ylabel('Elevation (m asl)', size=12)
    ax[0,0].set_xlabel('Volume\n(m$^{3}$)', size=12)
    ax[0,0].set_ylim(ymin, ymax)
    ax[0,0].yaxis.set_major_locator(MultipleLocator(500))
    ax[0,0].yaxis.set_minor_locator(MultipleLocator(100))
    ax[0,0].set_xlim(0,np.max([bin_vol_t0.max(), bin_vol_end.max()]))
    ax[0,0].legend() 
    
    # Area altitude distribution
    ax[0,1].plot(bin_area_t0, surface_h, linewidth=1, linestyle='-')
    ax[0,1].plot(bin_area_end, surface_h, linewidth=1, linestyle='--')
    ax[0,1].set_xlabel('Area\n(m$^{2}$)', size=12)
    ax[0,1].set_ylim(ymin, ymax)
    
    # Thickness altitude distribution
    ax[0,2].plot(bin_thick_t0, surface_h, linewidth=1, linestyle='-')
    ax[0,2].plot(bin_thick_end, surface_h, linewidth=1, linestyle='--')
    ax[0,2].set_xlabel('Ice thickness\n(m)', size=12)
    ax[0,2].set_ylim(ymin, ymax)
    
    # MB altitude distribution
    ax[0,3].plot(mb_clim, surface_h, linewidth=1, color='#c7e9b4', label='$b_{clim}$')
    ax[0,3].plot(bin_flux, surface_h, linewidth=1, color='#41b6c4', label=r'$\nabla$' + '$q$')
    ax[0,3].plot(mb_clim-bin_flux, surface_h, linewidth=1, color='#253494', label=r'$\Delta$' + '$h$')
    ax[0,3].plot(np.zeros(surface_h.shape), surface_h, color='k', linewidth=0.5)
    ax[0,3].set_ylim(ymin, ymax)       
    ax[0,3].set_xlabel('Mass Balance\n(m w.e. a$^{-1}$)', size=12)
    ax[0,3].xaxis.set_major_locator(MultipleLocator(2))
    ax[0,3].xaxis.set_minor_locator(MultipleLocator(0.5))
    ax[0,3].legend(handlelength=0.5, handletextpad=0.2)
    
    # Legend
    ax[0,1].text(0.5, 1.06, glacno + ' ERA5', size=10, horizontalalignment='center', 
                 verticalalignment='top', transform=ax[0,1].transAxes)       
    ax[0,0].tick_params(direction='inout', right=True)
    # Save figure
    fig_fn = glacno + '_ERA5_AAD.png'
    fig.set_size_inches(7,3)
    fig.savefig(fig_fp + fig_fn, bbox_inches='tight', dpi=300)

    #%%
    # Calculate glacier-wide mass balance
    glac_wide_mbtotal_monthly = ds.glac_massbaltotal_monthly.values[0,:]
    glac_wide_area_initial = ds.glac_area_annual.values[0,0]
    nyears = ds.year.shape[0]-1
    glac_wide_mbtotal_mwea = glac_wide_mbtotal_monthly.sum() / glac_wide_area_initial / nyears
    
    print('glac_wide_mbtotal [mwea]:', np.round(glac_wide_mbtotal_mwea,2))
    
    #%%
    



