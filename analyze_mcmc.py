""" Analyze MCMC output - chain length, etc. """

# Built-in libraries
import glob
import os
import pickle
# External libraries
import cartopy
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
#from matplotlib.colors import Normalize
import matplotlib.colors as colors
import numpy as np
import pandas as pd
import pymc
from scipy import stats
from scipy.stats.kde import gaussian_kde
from scipy.stats import norm
from scipy.stats import truncnorm
from scipy.stats import uniform
from scipy.stats import linregress
from scipy.stats import lognorm
#from scipy.optimize import minimize
import xarray as xr
# Local libraries
import class_climate
import class_mbdata
import pygem_input as input
import pygemfxns_gcmbiasadj as gcmbiasadj
import pygemfxns_massbalance as massbalance
import pygemfxns_modelsetup as modelsetup
import run_calibration as calibration

#%%
# Paper figures
option_observation_vs_calibration = 0
option_papermcmc_prior_vs_posterior = 0
option_papermcmc_modelparameter_map_and_postvprior = 0
option_metrics_histogram_all = 0
option_metrics_vs_chainlength = 1
option_correlation_scatter = 0
option_regional_priors = 0
option_glacier_mb_vs_params = 0

option_papermcmc_solutionspace = 0
option_papermcmc_hh2015_map = 0

# Others
option_glacier_mcmc_plots = 0
option_raw_plotchain = 0
option_convertcal2table = 0


option_plot_era_normalizedchange = 0


# Export option
mcmc_output_netcdf_fp_3chain = input.output_filepath + 'cal_opt2_spc_20190815_3chain/'
mcmc_output_netcdf_fp_all = input.output_filepath + 'cal_opt2_spc_20190806/'
hh2015_output_netcdf_fp_all = input.output_filepath + 'cal_opt3/cal_opt3/'
mcmc_output_figures_fp = input.output_filepath + 'figures/'

regions = [13,14,15]
#regions = [13]

cal_datasets = ['shean']

burn=1000

chainlength = 10000
# Bounds (80% bounds --> 90% above/below given threshold)
low_percentile = 10
high_percentile = 90

variables = ['massbal', 'precfactor', 'tempchange', 'ddfsnow']  
vn_title_dict = {'massbal':'Mass Balance',                                                                      
                 'precfactor':'$\mathregular{k_{p}}$',                                                              
                 'tempchange':'$\mathregular{T_{bias}}$',                                                              
                 'ddfsnow':'$\mathregular{f_{snow}}$'}
vn_abbreviations_wunits_dict = {
                'massbal':'B (m w.e. $\mathregular{a^{-1}}$)',                                                                      
                'precfactor':'$\mathregular{k_{p}}$ (-)',                                                              
                'tempchange':'$\mathregular{T_{bias}}$ ($\mathregular{^{\circ}C}$)',                                                              
                'ddfsnow':'$\mathregular{f_{snow}}$ (mm w.e. $\mathregular{d^{-1}}$ $\mathregular{^{\circ}C^{-1}}$)'}
vn_abbreviations_dict = {'massbal':'$\mathregular{B}$',                                                                      
                         'precfactor':'$\mathregular{k_{p}}$',                                                              
                         'tempchange':'$\mathregular{T_{bias}}$',                                                              
                         'ddfsnow':'$\mathregular{f_{snow}}$'}
vn_title_wunits_dict = {'massbal':'Mass Balance (m w.e. $\mathregular{a^{-1}}$)',
                 'dif_masschange':'$\mathregular{B_{obs} - B_{mod}}$\n(m w.e. $\mathregular{a^{-1}}$)',
                 'precfactor':'$\mathregular{k_{p}}$ (-)',                                                              
                 'tempchange':'$\mathregular{T_{bias}}$ ($\mathregular{^{\circ}C}$)',                                                              
                 'ddfsnow':'$\mathregular{f_{snow}}$ (mm w.e. $\mathregular{d^{-1}}$ $\mathregular{^{\circ}C^{-1}}$)'}
vn_title_noabbreviations_dict = {'massbal':'Mass Balance',                                                                      
                                 'precfactor':'Precipitation Factor',                                                              
                                 'tempchange':'Temperature Bias',                                                              
                                 'ddfsnow':'$\mathregular{f_{snow}}$'}
vn_label_dict = {'massbal':'Mass Balance (m w.e. $\mathregular{a^{-1}}$)',                                                                      
                 'precfactor':'Precipitation Factor (-)',                                                              
                 'tempchange':'Temperature Bias ($\mathregular{^{\circ}C}$)',                                                               
                 'ddfsnow':'f$_{snow}$ (mm w.e. $\mathregular{d^{-1}}$ $\mathregular{^{\circ}C^{-1}}$)',
                 'dif_masschange':'Mass Balance (Observation - Model, mwea)'}
vn_label_units_dict = {'massbal':'(m w.e. $\mathregular{a^{-1}}$)',                                                                      
                       'precfactor':'(-)',                                                              
                       'tempchange':'($\mathregular{^{\circ}}$C)',                                                               
                       'ddfsnow':'(mm w.e. d$^{-1}$ $^\circ$C$^{-1}$)'}
metric_title_dict = {'Gelman-Rubin':'Gelman-Rubin Statistic',
                     'MC Error': 'Monte Carlo Error',
                     'Effective N': 'Effective Sample Size'}
metrics = ['Gelman-Rubin', 'MC Error', 'Effective N']
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
              'inner_TP': 'Inner TP',
              'Karakoram': 'Karakoram',
              'Yigong': 'Yigong',
              'Yellow': 'Yellow',
              'Bhutan': 'Bhutan',
              'Everest': 'Everest',
              'West Nepal': 'West Nepal',
              'Spiti Lahaul': 'Spiti Lahaul',
              'tien_shan': 'Tien Shan',
              'Pamir': 'Pamir',
              'pamir_alai': 'Pamir Alai',
              'Kunlun': 'Kunlun',
              'Hindu Kush': 'Hindu Kush',
              13: 'Central Asia',
              14: 'South Asia West',
              15: 'South Asia East',
              'all': 'HMA',
              'Altun Shan':'Altun Shan',
              'Central Himalaya':'C Himalaya',
              'Central Tien Shan':'C Tien Shan',
              'Dzhungarsky Alatau':'Dzhungarsky Alatau',
              'Eastern Himalaya':'E Himalaya',
              'Eastern Hindu Kush':'E Hindu Kush',
              'Eastern Kunlun Shan':'E Kunlun Shan',
              'Eastern Pamir':'E Pamir',
              'Eastern Tibetan Mountains':'E Tibetan Mtns',
              'Eastern Tien Shan':'E Tien Shan',
              'Gangdise Mountains':'Gangdise Mtns',
              'Hengduan Shan':'Hengduan Shan',
              'Karakoram':'Karakoram',
              'Northern/Western Tien Shan':'N/W Tien Shan',
              'Nyainqentanglha':'Nyainqentanglha',
              'Pamir Alay':'Pamir Alay',
              'Qilian Shan':'Qilian Shan',
              'Tanggula Shan':'Tanggula Shan',
              'Tibetan Interior Mountains':'Tibetan Int Mtns',
              'Western Himalaya':'W Himalaya',
              'Western Kunlun Shan':'W Kunlun Shan',
              'Western Pamir':'W Pamir'
              }

#colors = ['#387ea0', '#fcb200', '#d20048']
linestyles = ['-', '--', ':']

# Group dictionaries
watershed_dict_fn = '/Users/davidrounce/Documents/Dave_Rounce/HiMAT/qgis_himat/rgi60_HMA_dict_watershed.csv'
watershed_csv = pd.read_csv(watershed_dict_fn)
watershed_dict = dict(zip(watershed_csv.RGIId, watershed_csv.watershed))
kaab_dict_fn = '/Users/davidrounce/Documents/Dave_Rounce/HiMAT/qgis_himat/rgi60_HMA_dict_kaab.csv'
kaab_csv = pd.read_csv(kaab_dict_fn)
kaab_dict = dict(zip(kaab_csv.RGIId, kaab_csv.kaab_name))
himap_dict_fn = '/Users/davidrounce/Documents/Dave_Rounce/HiMAT/qgis_himat/rgi60_HMA_dict_bolch.csv'
himap_csv = pd.read_csv(himap_dict_fn)
himap_dict = dict(zip(himap_csv.RGIId, himap_csv.bolch_name))

# Shapefiles
rgiO1_shp_fn = '/Users/davidrounce/Documents/Dave_Rounce/HiMAT/RGI/rgi60/00_rgi60_regions/00_rgi60_O1Regions.shp'
rgi_glac_shp_fn = '/Users/davidrounce/Documents/Dave_Rounce/HiMAT/qgis_himat/rgi60_HMA.shp'
watershed_shp_fn = '/Users/davidrounce/Documents/Dave_Rounce/HiMAT/qgis_himat/HMA_basins_20181018_4plot.shp'
kaab_shp_fn = '/Users/davidrounce/Documents/Dave_Rounce/HiMAT/qgis_himat/kaab2015_regions.shp'
bolch_shp_fn = '/Users/davidrounce/Documents/Dave_Rounce/HiMAT/qgis_himat/himap_regions/boundary_mountain_regions_hma_v3.shp'
srtm_fn = '/Users/davidrounce/Documents/Dave_Rounce/HiMAT/qgis_himat/SRTM_HMA.tif'
srtm_contour_fn = '/Users/davidrounce/Documents/Dave_Rounce/HiMAT/qgis_himat/SRTM_HMA_countours_2km_gt3000m_smooth.shp'


class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # Note that I'm ignoring clipping and other edge cases here.
        result, is_scalar = self.process_value(value)
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.array(np.interp(value, x, y), mask=result.mask, copy=False)


def load_glacierdata_byglacno(glac_no, option_loadhyps_climate=1, option_loadcal_data=1):
    """ Load glacier data, climate data, and calibration data for list of glaciers 
    
    Parameters
    ----------
    glac_no : list
        list of glacier numbers (ex. ['13.0001', 15.00001'])
    
    Returns
    -------
    main_glac_rgi, main_glac_hyps, main_glac_icethickness, main_glac_width, gcm_temp, gcm_prec, gcm_elev, gcm_lr, 
    cal_data, dates_table
    """
    glac_no_byregion = {}
    regions = [int(i.split('.')[0]) for i in glac_no]
    regions = list(set(regions))
    for region in regions:
        glac_no_byregion[region] = []
    for i in glac_no:
        region = i.split('.')[0]
        glac_no_only = i.split('.')[1]
        glac_no_byregion[int(region)].append(glac_no_only)
        
    for region in regions:
        glac_no_byregion[region] = sorted(glac_no_byregion[region])
        
        # EXCEPTION COULD BE ADDED HERE INSTEAD
        
    # Load data for glaciers
    dates_table_nospinup = modelsetup.datesmodelrun(startyear=input.startyear, endyear=input.endyear, spinupyears=0)
    dates_table = modelsetup.datesmodelrun(startyear=input.startyear, endyear=input.endyear, 
                                           spinupyears=input.spinupyears)
    
    count = 0
    for region in regions:
        count += 1
        # ====== GLACIER data =====
        if ((region == 13 and len(glac_no_byregion[region]) == 54429) or 
            (region == 14 and len(glac_no_byregion[region]) == 27988) or
            (region == 15 and len(glac_no_byregion[region]) == 13119) ):
            main_glac_rgi_region = modelsetup.selectglaciersrgitable(
                rgi_regionsO1=[region], rgi_regionsO2 = 'all', rgi_glac_number='all')
        else:
            main_glac_rgi_region = modelsetup.selectglaciersrgitable(
                rgi_regionsO1=[region], rgi_regionsO2 = 'all', rgi_glac_number=glac_no_byregion[region])
        # Glacier hypsometry
        main_glac_hyps_region = modelsetup.import_Husstable(
                main_glac_rgi_region, input.hyps_filepath,input.hyps_filedict, input.hyps_colsdrop)
        
        if option_loadcal_data == 1:
            # ===== CALIBRATION DATA =====
            cal_data_region = pd.DataFrame()
            for dataset in cal_datasets:
                cal_subset = class_mbdata.MBData(name=dataset, rgi_regionO1=region)
                cal_subset_data = cal_subset.retrieve_mb(main_glac_rgi_region, main_glac_hyps_region, 
                                                         dates_table_nospinup)
                cal_data_region = cal_data_region.append(cal_subset_data, ignore_index=True)
            cal_data_region = cal_data_region.sort_values(['glacno', 't1_idx'])
            cal_data_region.reset_index(drop=True, inplace=True)
        
        # ===== OTHER DATA =====
        if option_loadhyps_climate == 1:
            # Ice thickness [m], average
            main_glac_icethickness_region = modelsetup.import_Husstable(
                    main_glac_rgi_region, input.thickness_filepath, input.thickness_filedict, 
                    input.thickness_colsdrop)
            main_glac_hyps_region[main_glac_icethickness_region == 0] = 0
            # Width [km], average
            main_glac_width_region = modelsetup.import_Husstable(
                    main_glac_rgi_region, input.width_filepath, input.width_filedict, input.width_colsdrop)
            # ===== CLIMATE DATA =====
            gcm = class_climate.GCM(name=input.ref_gcm_name)
            # Air temperature [degC], Precipitation [m], Elevation [masl], Lapse rate [K m-1]
            gcm_temp_region, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(
                    gcm.temp_fn, gcm.temp_vn, main_glac_rgi_region, dates_table_nospinup)
            gcm_prec_region, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(
                    gcm.prec_fn, gcm.prec_vn, main_glac_rgi_region, dates_table_nospinup)
            gcm_elev_region = gcm.importGCMfxnearestneighbor_xarray(gcm.elev_fn, gcm.elev_vn, main_glac_rgi_region)
            # Lapse rate [K m-1]
            gcm_lr_region, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(
                    gcm.lr_fn, gcm.lr_vn, main_glac_rgi_region, dates_table_nospinup)
        
        # ===== APPEND DATASETS =====
        if count == 1:
            main_glac_rgi = main_glac_rgi_region
            
            if option_loadcal_data == 1:
                cal_data = cal_data_region
        
            if option_loadhyps_climate == 1:
                main_glac_hyps = main_glac_hyps_region
                main_glac_icethickness = main_glac_icethickness_region
                main_glac_width = main_glac_width_region
                gcm_temp = gcm_temp_region
                gcm_prec = gcm_prec_region
                gcm_elev = gcm_elev_region
                gcm_lr = gcm_lr_region
                
        else:
            main_glac_rgi = main_glac_rgi.append(main_glac_rgi_region)
            
            if option_loadcal_data == 1:
                cal_data = cal_data.append(cal_data_region)
            
            if option_loadhyps_climate == 1:
                # If more columns in region, then need to expand existing dataset
                if main_glac_hyps_region.shape[1] > main_glac_hyps.shape[1]:
                    all_col = list(main_glac_hyps.columns.values)
                    reg_col = list(main_glac_hyps_region.columns.values)
                    new_cols = [item for item in reg_col if item not in all_col]
                    for new_col in new_cols:
                        main_glac_hyps[new_col] = 0
                        main_glac_icethickness[new_col] = 0
                        main_glac_width[new_col] = 0
                elif main_glac_hyps_region.shape[1] < main_glac_hyps.shape[1]:
                    all_col = list(main_glac_hyps.columns.values)
                    reg_col = list(main_glac_hyps_region.columns.values)
                    new_cols = [item for item in all_col if item not in reg_col]
                    for new_col in new_cols:
                        main_glac_hyps_region[new_col] = 0
                        main_glac_icethickness_region[new_col] = 0
                        main_glac_width_region[new_col] = 0
                main_glac_hyps = main_glac_hyps.append(main_glac_hyps_region)
                main_glac_icethickness = main_glac_icethickness.append(main_glac_icethickness_region)
                main_glac_width = main_glac_width.append(main_glac_width_region)
            
                gcm_temp = np.vstack([gcm_temp, gcm_temp_region])
                gcm_prec = np.vstack([gcm_prec, gcm_prec_region])
                gcm_elev = np.concatenate([gcm_elev, gcm_elev_region])
                gcm_lr = np.vstack([gcm_lr, gcm_lr_region])
            
    # reset index
    main_glac_rgi.reset_index(inplace=True, drop=True)
    
    if option_loadcal_data == 1:
        cal_data.reset_index(inplace=True, drop=True)
    
    if option_loadhyps_climate == 1:
        main_glac_hyps.reset_index(inplace=True, drop=True)
        main_glac_icethickness.reset_index(inplace=True, drop=True)
        main_glac_width.reset_index(inplace=True, drop=True)
    
    if option_loadhyps_climate == 0 and option_loadcal_data == 0:
        return main_glac_rgi
    if option_loadhyps_climate == 0 and option_loadcal_data == 1:
        return main_glac_rgi, cal_data
    else:
        return (main_glac_rgi, main_glac_hyps, main_glac_icethickness, main_glac_width, 
                gcm_temp, gcm_prec, gcm_elev, gcm_lr, 
                cal_data, dates_table)
        
        
def select_groups(grouping, main_glac_rgi_all):
    """
    Select groups based on grouping
    """
    if grouping == 'rgi_region':
        groups = main_glac_rgi_all.O1Region.unique().tolist()
        group_cn = 'O1Region'
    elif grouping == 'watershed':
        groups = main_glac_rgi_all.watershed.unique().tolist()
        group_cn = 'watershed'
    elif grouping == 'kaab':
        groups = main_glac_rgi_all.kaab.unique().tolist()
        group_cn = 'kaab'
        groups = [x for x in groups if str(x) != 'nan']  
    elif grouping == 'degree':
        groups = main_glac_rgi_all.deg_id.unique().tolist()
        group_cn = 'deg_id'
    elif grouping == 'mascon':
        groups = main_glac_rgi_all.mascon_idx.unique().tolist()
        groups = [int(x) for x in groups]
        group_cn = 'mascon_idx'
    else:
        groups = ['all']
        group_cn = 'all_group'
    try:
        groups = sorted(groups, key=str.lower)
    except:
        groups = sorted(groups)
    return groups, group_cn


def partition_groups(grouping, vn, main_glac_rgi_all, regional_calc='mean'):
    """Partition variable by each group
    
    Parameters
    ----------
    grouping : str
        name of grouping to use
    vn : str
        variable name
    main_glac_rgi_all : pd.DataFrame
        glacier table
    regional_calc : str
        calculation used to compute region value (mean, sum, area_weighted_mean)
        
    Output
    ------
    groups : list
        list of group names
    ds_group : list of lists
        dataset containing the multimodel data for a given variable for all the GCMs
    """
    # Groups
    groups, group_cn = select_groups(grouping, main_glac_rgi_all)
    
    ds_group = [[] for group in groups]
    
    # Cycle through groups
    for ngroup, group in enumerate(groups):
        # Select subset of data
        main_glac_rgi = main_glac_rgi_all.loc[main_glac_rgi_all[group_cn] == group]                        
        vn_glac = main_glac_rgi_all[vn].values[main_glac_rgi.index.values.tolist()]
        if 'area_weighted' in regional_calc:
            vn_glac_area = main_glac_rgi_all['Area'].values[main_glac_rgi.index.values.tolist()]

        # Regional calc
        if regional_calc == 'mean':           
            vn_reg = vn_glac.mean(axis=0)
        elif regional_calc == 'sum':
            vn_reg = vn_glac.sum(axis=0)
        elif regional_calc == 'area_weighted_mean':
            vn_reg = (vn_glac * vn_glac_area).sum() / vn_glac_area.sum()
        
        # Record data for each group
        ds_group[ngroup] = [group, vn_reg]
                
    return groups, ds_group

    
def effective_n(ds, vn, iters, burn, chain=0):
    """
    Compute the effective sample size of a trace.

    Takes the trace and computes the effective sample size
    according to its detrended autocorrelation.

    Parameters
    ----------
    ds : xarray.Dataset
        dataset containing mcmc traces
    vn : str
        Parameter variable name
    iters : int
        number of mcmc iterations to test
    burn : int
        number of initial iterations to throw away

    Returns
    -------
    effective_n : int
        effective sample size
    """
    # Effective sample size
    x = ds['mp_value'].sel(chain=chain, mp=vn).values[burn:iters]
    # detrend trace using mean to be consistent with statistics
    # definition of autocorrelation
    x = (x - x.mean())
    # compute autocorrelation (note: only need second half since
    # they are symmetric)
    rho = np.correlate(x, x, mode='full')
    rho = rho[len(rho)//2:]
    # normalize the autocorrelation values
    #  note: rho[0] is the variance * n_samples, so this is consistent
    #  with the statistics definition of autocorrelation on wikipedia
    # (dividing by n_samples gives you the expected value).
    rho_norm = rho / rho[0]
    # Iterate until sum of consecutive estimates of autocorrelation is
    # negative to avoid issues with the sum being -0.5, which returns an
    # effective_n of infinity
    negative_autocorr = False
    t = 1
    n = len(x)
    while not negative_autocorr and (t < n):
        if not t % 2:
            negative_autocorr = sum(rho_norm[t-1:t+1]) < 0
        t += 1
    return int(n / (1 + 2*rho_norm[1:t].sum()))


def gelman_rubin(ds, vn, iters=1000, burn=0, debug=False):
    """
    Calculate Gelman-Rubin statistic.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing MCMC iterations for a single glacier with 3 chains
    vn : str
        Parameter variable name
    iters : int
        number of MCMC iterations to test for the gelman-rubin statistic
    burn : int
        number of MCMC iterations to ignore at start of chain before performing test

    Returns
    -------
    gelman_rubin_stat : float
        gelman_rubin statistic (R_hat)
    """
    if debug:
        if len(ds.chain) != 3:
            raise ValueError('Given dataset has an incorrect number of chains')
        if iters > len(ds.chain):
            raise ValueError('iters value too high')
        if (burn >= iters):
            raise ValueError('Given iters and burn in are incompatible')

    # unpack iterations from dataset
    for n_chain in ds.chain.values:
        if n_chain == 0:
            chain = ds['mp_value'].sel(chain=n_chain, mp=vn).values[burn:iters]
            chain = np.reshape(chain, (1,len(chain)))
        else:
            chain2add = ds['mp_value'].sel(chain=n_chain, mp=vn).values[burn:iters]
            chain2add = np.reshape(chain2add, (1,chain.shape[1]))
            chain = np.append(chain, chain2add, axis=0)

    #calculate statistics with pymc in-built function
    return pymc.gelman_rubin(chain)


def mc_error(ds, vn, iters=None, burn=0, chain=None, method='overlapping'):
    """ Calculates Monte Carlo standard error using the batch mean method for each chain

    For multiple chains, it outputs a list of the values

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing MCMC iterations for a single glacier with 3 chains
    vn : str
        Parameter variable name
    iters : int
        Number of iterations to use
    
    Returns
    -------
    chains_mcse : list of floats
        list of the Monte Carlo standard error for each chain
    chains_ci : list of floats
        list of the +/- confidence interval value for each chain
    """
    if iters is None:
        iters = len(ds.mp_value)

    trace = [ds['mp_value'].sel(chain=n_chain, mp=vn).values[burn:iters] for n_chain in ds.chain.values]
    
    mcse_output = [mcse_batchmeans(i, method=method) for i in trace]
    
    chains_mcse = [i[0] for i in mcse_output]
    chains_ci = [i[1] for i in mcse_output]
    
    return chains_mcse, chains_ci


def mcse_batchmeans(trace, t_quantile=0.95, method='overlapping'):
    """ Calculates Monte Carlo standard error for a given trace using batch means method from Flegal and Jones (2010) 
    
    Splitting uses all values in trace, so batches can have different lengths (maximum difference is 1)
    
    Parameters
    ----------
    trace: np.ndarray
        Array representing MCMC chain
    t_quantile : float
        student t-test quantile (default = 0.95)
    method : str
        method used to compute batch means (default = 'overlapping', other option is 'nonoverlapping')
    
        
    Returns
    -------
    trace_mcse : float
        Monte Carlo standard error for a given trace
    trace_ci : float
        +/- value for confidence interval
    """
    # Number of batches (n**0.5 based on Flegal and Jones (2010))
    batches = int(len(trace)**0.5)
    batch_size = int(len(trace)/batches)
    # Split into batches
    if method == 'overlapping':
        trace_batches = [trace[i:i+batch_size] for i in range(0,int(len(trace)-batches+1))]
    elif method == 'nonoverlapping':
        trace_batches = split_array(trace,batches)
    # Sample batch means
    trace_batches_means = [np.mean(i) for i in trace_batches]
    # Batch mean estimator
    trace_batches_mean = np.mean(trace_batches_means)
    # Sample variance
    if method == 'overlapping':
        trace_samplevariance = (
                (len(trace)/batches) / len(trace) * np.sum([(i - trace_batches_mean)**2 for i in trace_batches_means]))
    elif method == 'nonoverlapping':
        trace_samplevariance = (
                (len(trace)/batches) / (batches-1) * np.sum([(i - trace_batches_mean)**2 for i in trace_batches_means]))
    # Monte Carlo standard error
    trace_mcse = trace_samplevariance**0.5 / len(trace)**0.5
    # Confidence interval value (actual confidence interval is batch_mean_estimator +/- trace_ci)
    trace_ci = stats.t.ppf(t_quantile, (len(trace)**0.5)-1) * trace_mcse
    
    return trace_mcse, trace_ci


def split_array(arr, n=1):
    """
    Split array of glaciers into batches for batch means.
    
    Parameters
    ----------
    arr : np.array
        array that you want to split into separate batches
    n : int
        Number of batches to split glaciers into.
    
    Returns
    -------
    arr_batches : np.array
        list of n arrays that have sequential values in each list
    """
    # If batches is more than list, the one in each list
    if n > len(arr):
        n = len(arr)
    # number of values per list rounded down/up
    n_perlist_low = int(len(arr)/n)
    n_perlist_high = int(np.ceil(len(arr)/n))
    # number of lists with higher number per list (uses all values of array, but chains not necessarily equal length)
    n_lists_high = len(arr)%n
    # loop through and select values
    count = 0
    arr_batches = []
    for x in np.arange(n):
        count += 1
        if count <= n_lists_high:
            arr_subset = arr[0:n_perlist_high]
            arr_batches.append(arr_subset)
            arr = arr[n_perlist_high:]
        else:
            arr_subset = arr[0:n_perlist_low]
            arr_batches.append(arr_subset)
            arr = arr[n_perlist_low:]
    return arr_batches 
    
    
def pickle_data(fn, data):
    """Pickle data
    
    Parameters
    ----------
    fn : str
        filename including filepath
    data : list, etc.
        data to be pickled
    
    Returns
    -------
    .pkl file
        saves .pkl file of the data
    """
    with open(fn, 'wb') as f:
        pickle.dump(data, f)
        
        
def plot_hist(df, cn, bins, xlabel=None, ylabel=None, fig_fn='hist.png', fig_fp=input.output_filepath):
        """
        Plot histogram for any bin size
        """           
        data = df[cn].values
        hist, bin_edges = np.histogram(data,bins) # make the histogram
        fig,ax = plt.subplots()    
        # Plot the histogram heights against integers on the x axis
        ax.bar(range(len(hist)),hist,width=1, edgecolor='k') 
        # Set the ticks to the middle of the bars
        ax.set_xticks([0.5+i for i,j in enumerate(hist)])
        # Set the xticklabels to a string that tells us what the bin edges were
        ax.set_xticklabels(['{} - {}'.format(bins[i],bins[i+1]) for i,j in enumerate(hist)], rotation=45, ha='right')
        ax.set_xlabel(xlabel, fontsize=16)
        ax.set_ylabel(ylabel, fontsize=16)
        # Save figure
        fig.set_size_inches(6,4)
        fig.savefig(fig_fp + fig_fn, bbox_inches='tight', dpi=300)
    
    
def plot_mb_vs_parameters(tempchange_iters, precfactor_iters, ddfsnow_iters, modelparameters, glacier_rgi_table, 
                          glacier_area_t0, icethickness_t0, width_t0, elev_bins, glacier_gcm_temp, glacier_gcm_prec, 
                          glacier_gcm_elev, glacier_gcm_lrgcm, glacier_gcm_lrglac, dates_table, observed_massbal, 
                          observed_error, tempchange_boundhigh, tempchange_boundlow, 
                          tempchange_opt_init=None, mb_max_acc=None, mb_max_loss=None, option_areaconstant=0, 
                          option_plotsteps=1, fig_fp=input.output_filepath):
    """
    Plot the mass balance [mwea] versus all model parameters to see how parameters effect mass balance
    """
    #%%
    mb_vs_parameters = pd.DataFrame(np.zeros((len(ddfsnow_iters) * len(precfactor_iters) * len(tempchange_iters), 4)),
                                    columns=['precfactor', 'tempbias', 'ddfsnow', 'massbal'])
    count=0
    for n, precfactor in enumerate(precfactor_iters):
        modelparameters[2] = precfactor
        
        # run mass balance calculation
#        if modelparameters[2] == 1:
#            option_areaconstant = 0
#        else:
#            option_areaconstant = 1
        option_areaconstant = 0
        print('PF:', precfactor, 'option_areaconstant:', option_areaconstant)
        
        for n, tempchange in enumerate(tempchange_iters):
            modelparameters[7] = tempchange

            for c, ddfsnow in enumerate(ddfsnow_iters):
                
                modelparameters[4] = ddfsnow
                modelparameters[5] = modelparameters[4] / input.ddfsnow_iceratio
                
                (glac_bin_temp, glac_bin_prec, glac_bin_acc, glac_bin_refreeze, glac_bin_snowpack, glac_bin_melt,
                 glac_bin_frontalablation, glac_bin_massbalclim, glac_bin_massbalclim_annual, glac_bin_area_annual,
                 glac_bin_icethickness_annual, glac_bin_width_annual, glac_bin_surfacetype_annual,
                 glac_wide_massbaltotal, glac_wide_runoff, glac_wide_snowline, glac_wide_snowpack,
                 glac_wide_area_annual, glac_wide_volume_annual, glac_wide_ELA_annual, offglac_wide_prec, 
                 offglac_wide_refreeze, offglac_wide_melt, offglac_wide_snowpack, offglac_wide_runoff) = (
                    massbalance.runmassbalance(modelparameters[0:8], glacier_rgi_table, glacier_area_t0, icethickness_t0,
                                               width_t0, elev_bins, glacier_gcm_temp, glacier_gcm_prec, 
                                               glacier_gcm_elev, glacier_gcm_lrgcm, glacier_gcm_lrglac, dates_table, 
                                               option_areaconstant=option_areaconstant))
                
                # Compute glacier volume change for every time step and use this to compute mass balance
                #  this will work for any indexing
                glac_wide_area = glac_wide_area_annual[:-1].repeat(12)
                # Mass change [km3 mwe]
                #  mb [mwea] * (1 km / 1000 m) * area [km2]
                glac_wide_masschange = glac_wide_massbaltotal / 1000 * glac_wide_area
                # Mean annual mass balance [mwea]
                mb_mwea = (glac_wide_masschange.sum() / glac_wide_area[0] * 1000 / 
                           (glac_wide_masschange.shape[0] / 12))
                mb_vs_parameters.loc[count,:] = np.array([precfactor, tempchange, ddfsnow, mb_mwea])
                count += 1
#            print(modelparameters[2], modelparameters[7], modelparameters[4], np.round(mb_mwea,3))

    # Subset data for each precfactor
    linestyles = ['-', '--', ':', '-.']
    linecolors = ['b', 'k', 'r']
    prec_linedict = {precfactor : linestyles[n] for n, precfactor in enumerate(precfactor_iters)} 
    ddfsnow_colordict = {ddfsnow : linecolors[n] for n, ddfsnow in enumerate(ddfsnow_iters)} 
    
    #%%
    # Plot the mass balance versus model parameters
    fig, ax = plt.subplots(figsize=(6,4))
    
    for precfactor in precfactor_iters:
        modelparameters[2] = precfactor
        mb_vs_parameters_subset = mb_vs_parameters.loc[mb_vs_parameters.loc[:,'precfactor'] == precfactor]
        for ddfsnow in ddfsnow_iters:
            mb_vs_parameters_plot =  mb_vs_parameters_subset.loc[mb_vs_parameters_subset.loc[:,'ddfsnow'] == ddfsnow]
            ax.plot(mb_vs_parameters_plot.loc[:,'tempbias'], mb_vs_parameters_plot.loc[:,'massbal'], 
                    linestyle=prec_linedict[precfactor], color=ddfsnow_colordict[ddfsnow])    
    
    # Add horizontal line of mass balance observations
    ax.axhline(observed_massbal, color='gray', linewidth=2, zorder=2)    
    observed_mb_min = observed_massbal - 3*observed_error
    observed_mb_max = observed_massbal + 3*observed_error  
    fillcolor = 'lightgrey'
    ax.fill_between([np.min(tempchange_iters), np.max(tempchange_iters)], observed_mb_min, observed_mb_max, 
                    facecolor=fillcolor, label=None, zorder=1)
    
    if option_plotsteps == 1:
#        marker='*'
#        marker_size = 20
        marker='D'
        marker_size = 10
        markeredge_color = 'black'
        marker_color = 'black'
        
        txt_xadj = -0.1
        txt_yadj = -0.06
        
        
        xytxt_list = [(tempchange_boundhigh, mb_max_loss, '1'), 
                      (tempchange_boundlow, mb_max_loss + 0.9*(mb_max_acc - mb_max_loss), '3'),
                      (tempchange_opt_init, observed_massbal, '4'),
                      (tempchange_opt_init + 3*tempchange_sigma, observed_mb_min, '6'),
                      (tempchange_opt_init - 3*tempchange_sigma, observed_mb_max, '6'),
                      (tempchange_opt_init - tempchange_sigma, observed_mb_max, '7'),
                      (tempchange_opt_init + tempchange_sigma, observed_mb_min, '7'),
                      (tempchange_mu, observed_massbal, '9')]
        for xytxt in xytxt_list: 
            x,y,txt = xytxt[0], xytxt[1], xytxt[2]
            ax.plot([x], [y], marker=marker, markersize=marker_size, 
                    markeredgecolor=markeredge_color, color=marker_color, zorder=3)
            ax.text(x+txt_xadj, y+txt_yadj, txt, zorder=4, color='white', fontsize=10)
     
    ax.set_xlim(np.min(tempchange_iters), np.max(tempchange_iters))
    if observed_massbal - 3*observed_error < mb_max_loss:
        ylim_lower = observed_massbal - 3*observed_error
    else:
        ylim_lower = np.floor(mb_max_loss)
    ax.set_ylim(int(ylim_lower),np.ceil(mb_vs_parameters['massbal'].max()))
#    ax.set_ylim(-2,2)
    
    # Labels
#    ax.set_title('Mass balance versus Parameters ' + glacier_str)
    ax.set_xlabel('Temperature Bias ($\mathregular{^{\circ}}$C)', fontsize=12)
    ax.set_ylabel('Mass Balance (m w.e. $\mathregular{a^{-1}}$)', fontsize=12)
    
    # Add legend
    leg_lines = []
    leg_names = []
    x_min = mb_vs_parameters.loc[:,'tempbias'].min()
    y_min = mb_vs_parameters.loc[:,'massbal'].min()
    for precfactor in reversed(precfactor_iters):
        line = Line2D([x_min,y_min],[x_min,y_min], linestyle=prec_linedict[precfactor], color='gray')
        leg_lines.append(line)
        leg_names.append('$\mathregular{k_{p}}$ ' + str(precfactor))
        
    for ddfsnow in ddfsnow_iters:
        line = Line2D([x_min,y_min],[x_min,y_min], linestyle='-', color=ddfsnow_colordict[ddfsnow])
        leg_lines.append(line)
        leg_names.append('$\mathregular{f_{snow}}$ ' + str(np.round(ddfsnow*10**3,1)))
        
        
    ax.legend(leg_lines, leg_names, loc='upper right', frameon=False, labelspacing=0.25)
    fig.savefig(fig_fp + glacier_str + '_mb_vs_parameters_areachg.eps', 
                bbox_inches='tight', dpi=300)    
    #%%

# ===== PLOT OPTIONS ==================================================================================================
def grid_values(vn, grouping, modelparams_all, midpt_value=np.nan):
    """ XYZ of grid values """    
    # Group data
    if vn in ['precfactor', 'tempchange', 'ddfsnow']:
        groups, ds_vn_deg = partition_groups(grouping, vn, modelparams_all, regional_calc='area_weighted_mean')
        groups, ds_group_area = partition_groups(grouping, 'Area', modelparams_all, regional_calc='sum')
    elif vn == 'dif_masschange':
        # Group calculations
        groups, ds_group_cal = partition_groups(grouping, 'mb_cal_Gta', modelparams_all, regional_calc='sum')
        groups, ds_group_era = partition_groups(grouping, 'mb_era_Gta', modelparams_all, regional_calc='sum')
        groups, ds_group_area = partition_groups(grouping, 'Area', modelparams_all, regional_calc='sum')
    
        # Group difference [Gt/yr]
        dif_cal_era_Gta = (np.array([x[1] for x in ds_group_cal]) - np.array([x[1] for x in ds_group_era])).tolist()
        # Group difference [mwea]
        area = [x[1] for x in ds_group_area]
        ds_group_dif_cal_era_mwea = [[x[0], dif_cal_era_Gta[n] / area[n] * 1000] for n, x in enumerate(ds_group_cal)]
        ds_vn_deg = ds_group_dif_cal_era_mwea
        
    z = [ds_vn_deg[ds_idx][1] for ds_idx in range(len(ds_vn_deg))]
    x = np.array([x[0] for x in deg_groups]) 
    y = np.array([x[1] for x in deg_groups])
    lons = np.arange(x.min(), x.max() + 2 * degree_size, degree_size)
    lats = np.arange(y.min(), y.max() + 2 * degree_size, degree_size)
    x_adj = np.arange(x.min(), x.max() + 1 * degree_size, degree_size) - x.min()
    y_adj = np.arange(y.min(), y.max() + 1 * degree_size, degree_size) - y.min()
    z_array = np.zeros((len(y_adj), len(x_adj)))
    z_array[z_array==0] = np.nan
    
    for i in range(len(z)):
        row_idx = int((y[i] - y.min()) / degree_size)
        col_idx = int((x[i] - x.min()) / degree_size)
        z_array[row_idx, col_idx] = z[i]
    return lons, lats, z_array
    
    
def plot_spatialmap_mbdif(vns, grouping, modelparams_all, xlabel, ylabel, figure_fp, fig_fn_prefix='', 
                          option_contour_lines=0, option_rgi_outlines=0, option_group_regions=0):
    """Plot spatial map of model parameters"""
    #%%
    fig = plt.figure()
    
    # Custom subplots
    gs = mpl.gridspec.GridSpec(20, 1)
    ax1 = plt.subplot(gs[0:11,0], projection=cartopy.crs.PlateCarree())
    ax2 = plt.subplot(gs[12:20,0])
    
#    # Third subplot
#    gs = mpl.gridspec.GridSpec(20, 20)
#    ax1 = plt.subplot(gs[0:11,0:20], projection=cartopy.crs.PlateCarree())
#    ax2 = plt.subplot(gs[12:20,0:7])
#    ax2 = plt.subplot(gs[12:20,13:20])
    
    cmap = 'RdYlBu_r'
#    cmap = plt.cm.get_cmap(cmap, 5)
    norm = plt.Normalize(colorbar_dict['dif_masschange'][0], colorbar_dict['dif_masschange'][1])    
    
    vn = 'dif_masschange'
    lons, lats, z_array = grid_values(vn, grouping, modelparams_all)
    ax1.pcolormesh(lons, lats, z_array, cmap=cmap, norm=norm, zorder=2, alpha=0.8)  

    # Add country borders for reference
#    ax1.add_feature(cartopy.feature.BORDERS, facecolor='none', edgecolor='lightgrey', zorder=10)
#    ax1.add_feature(cartopy.feature.COASTLINE, facecolor='none', edgecolor='lightgrey', zorder=10)
    # Set the extent
    ax1.set_extent([east, west, south, north], cartopy.crs.PlateCarree())    
    # Label title, x, and y axes
    ax1.set_xticks(np.arange(east,west+1,xtick), cartopy.crs.PlateCarree())
    ax1.set_yticks(np.arange(south,north+1,ytick), cartopy.crs.PlateCarree())
    ax1.set_xlabel(xlabel, size=labelsize, labelpad=0)
    ax1.set_ylabel(ylabel, size=labelsize)  
    # Add colorbar
#    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
#    sm._A = []
#    cbar = plt.colorbar(sm, ax=ax1, fraction=0.04, pad=0.01)
#    cbar.set_ticks(list(np.arange(colorbar_dict[vn][0], colorbar_dict[vn][1] + 0.01, 0.1))) 
#    fig.text(1.01, 0.6, '$\mathregular{B_{mod} - B_{obs}}$ (m w.e. $\mathregular{a^{-1}}$)', va='center',
#                 rotation='vertical', size=12)   
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm._A = []
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.5, 0.02, 0.35])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_ticks(list(np.arange(colorbar_dict['dif_masschange'][0], colorbar_dict['dif_masschange'][1] + 0.01, 0.1)))
    fig.text(1.04, 0.67, '$\mathregular{B_{mod} - B_{obs}}$ (m w.e. $\mathregular{a^{-1}}$)', va='center',
             rotation='vertical', size=12)
    # Add contour lines and/or rgi outlines
    if option_contour_lines == 1:
        srtm_contour_shp = cartopy.io.shapereader.Reader(srtm_contour_fn)
        srtm_contour_feature = cartopy.feature.ShapelyFeature(srtm_contour_shp.geometries(), cartopy.crs.PlateCarree(),
                                                              edgecolor='lightgrey', facecolor='none', linewidth=0.05)
        ax1.add_feature(srtm_contour_feature, zorder=9)   
    if option_rgi_outlines == 1:
        rgi_shp = cartopy.io.shapereader.Reader(rgi_glac_shp_fn)
        rgi_feature = cartopy.feature.ShapelyFeature(rgi_shp.geometries(), cartopy.crs.PlateCarree(),
                                                     edgecolor='black', facecolor='none', linewidth=0.1)
        ax1.add_feature(rgi_feature, zorder=9)         
    if option_group_regions == 1:
        rgi_shp = cartopy.io.shapereader.Reader(bolch_shp_fn)
        rgi_feature = cartopy.feature.ShapelyFeature(rgi_shp.geometries(), cartopy.crs.PlateCarree(),
                                                     edgecolor='lightgrey', facecolor='none', linewidth=1)
        ax1.add_feature(rgi_feature, zorder=9)
        ax1.text(101., 28.0, 'Hengduan\nShan', zorder=10, size=8, va='center', ha='center')
        ax1.text(99.0, 26.5, 'Nyainqentanglha', zorder=10, size=8, va='center', ha='center')
        ax1.plot([98,96], [27,29.3], color='k', linewidth=0.25, zorder=10)
        ax1.text(93.0, 27.5, 'Eastern Himalaya', zorder=10, size=8, va='center', ha='center')
        ax1.text(80.0, 27.3, 'Central Himalaya', zorder=10, size=8, va='center', ha='center')
        ax1.text(72.0, 31.7, 'Western Himalaya', zorder=10, size=8, va='center', ha='center')
        ax1.text(70.5, 33.7, 'Eastern\nHindu Kush', zorder=10, size=8, va='center', ha='center')
        ax1.text(79.0, 39.7, 'Karakoram', zorder=10, size=8, va='center', ha='center')
        ax1.plot([76,78], [36,39], color='k', linewidth=0.25, zorder=10)
        ax1.text(80.7, 38.0, 'Western\nKunlun Shan', zorder=10, size=8, va='center', ha='center')
        ax1.text(86.0, 33.7, 'Tibetan Interior\nMountains', zorder=10, size=8, va='center', ha='center')
        ax1.text(73.0, 29.0, 'Gandise Mountains', zorder=10, size=8, va='center', ha='center')
        ax1.plot([77.5,81.5], [29,31.4], color='k', linewidth=0.25, zorder=10)
        

    # Scatter plot
#    # Scatterplot: Model vs. Observed Mass balance colored by Area
#    cmap = 'RdYlBu_r'  
#    norm = colors.LogNorm(vmin=0.1, vmax=10)    
#    a = ax2.scatter(modelparams_all['mb_mwea'], modelparams_all['mb_mean'], c=modelparams_all['Area'],
#                   cmap=cmap, norm=norm, s=20, linewidth=0.5)
#    a.set_facecolor('none')
#    ax2.plot([-2.5,2],[-2.5,2], color='k', linewidth=0.5)
#    ax2.set_xlim([-2.5,1.75])
#    ax2.set_ylim([-2.5,1.75])
#    ax2.set_ylabel('$\mathregular{B_{obs}}$ $\mathregular{(m w.e. a^{-1})}$', size=12)
#    ax2.set_xlabel('$\mathregular{B_{mod}}$ $\mathregular{(m w.e. a^{-1})}$', size=12)
##    # Add colorbar
##    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
##    sm._A = []
##    cbar = plt.colorbar(sm, ax=ax2, fraction=0.04, pad=0.01)
##    fig.text(1.01, 0.5, 'Area ($\mathregular{km^{2}}$)', va='center', rotation='vertical', size=12)
#    
#    # Add colorbar
#    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
#    sm._A = []
#    cbar_ax = fig.add_axes([0.92, 0.13, 0.02, 0.29])
#    cbar = fig.colorbar(sm, cax=cbar_ax)
##    cbar.set_ticks(list(np.arange(colorbar_dict['massbal'][0], colorbar_dict['massbal'][1] + 0.01, 0.5)))
#    fig.text(1.04, 0.28, 'Area ($\mathregular{km^{2}}$)', va='center', rotation='vertical', size=12)
    
    # Z-score
    ax2.axhline(y=0, xmin=0, xmax=200, color='black', linewidth=0.5, zorder=1)
#    ax2.scatter(modelparams_all['Area'], modelparams_all['mb_mwea'], c=modelparams_all['dif_cal_era_mean'], 
#               cmap=cmap, norm=norm, s=5)
#    ax2.set_xlim([0,200])
#    ax2.set_ylim([-2.9,1.25])
#    ax2.set_ylabel('$\mathregular{B_{obs}}$ $\mathregular{(m w.e. a^{-1})}$', size=12)
#    ax2.set_xlabel('Area ($\mathregular{km^{2}}$)', size=12)
#    
#    # Inset axis over main axis
#    ax_inset = plt.axes([.37, 0.16, .51, .14])
#    ax_inset.axhline(y=0, xmin=0, xmax=5, color='black', linewidth=0.5)
#    ax_inset.scatter(modelparams_all['Area'], modelparams_all['mb_mwea'], c=modelparams_all['dif_cal_era_mean'], 
#               cmap=cmap, norm=norm, s=3)
#    ax_inset.set_xlim([0,5])
#    
#    # Add colorbar
#    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
#    sm._A = []
#    fig.subplots_adjust(right=0.9)
#    cbar_ax = fig.add_axes([0.92, 0.16, 0.03, 0.67])
#    cbar = fig.colorbar(sm, cax=cbar_ax)
#    cbar.set_ticks(list(np.arange(colorbar_dict['dif_masschange'][0], colorbar_dict['dif_masschange'][1] + 0.01, 0.1)))
#    fig.text(1.04, 0.5, '$\mathregular{B_{mod} - B_{obs}}$ (m w.e. $\mathregular{a^{-1}}$)', va='center',
#             rotation='vertical', size=12)

    # Scatterplot
    cmap = 'RdYlBu'
#    cmap = plt.cm.get_cmap(cmap, 5)
#    norm = plt.Normalize(colorbar_dict['massbal'][0], colorbar_dict['massbal'][1])   
    norm = MidpointNormalize(midpoint=0, vmin=colorbar_dict['massbal'][0], vmax=colorbar_dict['massbal'][1])  
    a = ax2.scatter(modelparams_all['Area'], modelparams_all['zscore'], c=modelparams_all['mb_mwea'], 
                   cmap=cmap, norm=norm, s=20, linewidth=0.5, zorder=2)
    a.set_facecolor('none')
    ax2.set_xlim([0,200])
    ax2.set_ylim([-3.8,2.5])
    ax2.set_ylabel('z-score ($\\frac{B_{mod} - B_{obs}}{B_{std}}$)', size=12)
    ax2.set_xlabel('Area ($\mathregular{km^{2}}$)', size=12)
    # Inset axis over main axis
    ax_inset = plt.axes([.37, 0.16, .51, .12])
    b = ax_inset.scatter(modelparams_all['Area'], modelparams_all['zscore'], c=modelparams_all['mb_mwea'], 
                         cmap=cmap, norm=norm, s=10,linewidth=0.5)
    b.set_facecolor('none')
    ax_inset.set_xlim([0,5])
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm._A = []
    cbar_ax = fig.add_axes([0.92, 0.13, 0.02, 0.29])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_ticks(list(np.arange(colorbar_dict['massbal'][0], colorbar_dict['massbal'][1] + 0.01, 0.5)))
    fig.text(1.04, 0.28, '$\mathregular{B_{obs}}$ $\mathregular{(m w.e. a^{-1})}$', va='center',
             rotation='vertical', size=12)
    
#    cbar = plt.colorbar(sm, ax=ax2, fraction=0.04, pad=0.01)
#    cbar.set_ticks(list(np.arange(colorbar_dict['massbal'][0], colorbar_dict['massbal'][1] + 0.01, 0.5)))
#    fig.text(1.01, 0.3, '$\mathregular{B_{obs}}$ $\mathregular{(m w.e. a^{-1})}$', va='center',
#             rotation='vertical', size=12)
    
    # Add subplot labels
    fig.text(0.15, 0.83, 'A', zorder=4, color='black', fontsize=12, fontweight='bold')
    fig.text(0.15, 0.40, 'B', zorder=4, color='black', fontsize=12, fontweight='bold')
    
    # Save figure
    fig.set_size_inches(6,7)
    if degree_size < 1:
        degsize_name = 'pt' + str(int(degree_size * 100))
    else:
        degsize_name = str(degree_size)
    fig_fn = fig_fn_prefix + 'MB_dif_map_scatter_' + degsize_name + 'deg.png'
    fig.savefig(figure_fp + fig_fn, bbox_inches='tight', dpi=300)
    #%%
    

def plot_spatialmap_parameters(vns, grouping, modelparams_all, xlabel, ylabel, midpt_dict, cmap_dict, title_adj, 
                               figure_fp, fig_fn_prefix='', option_contour_lines=0, option_rgi_outlines=0,
                               option_group_regions=0):
    """Plot spatial map of model parameters"""
    
    fig, ax = plt.subplots(len(vns), 1, subplot_kw={'projection':cartopy.crs.PlateCarree()},
                           gridspec_kw = {'wspace':0.1, 'hspace':0.03})
    
    for nvar, vn in enumerate(vns):
        
        class MidpointNormalize(colors.Normalize):
            def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
                self.midpoint = midpoint
                colors.Normalize.__init__(self, vmin, vmax, clip)
        
            def __call__(self, value, clip=None):
                # Note that I'm ignoring clipping and other edge cases here.
                result, is_scalar = self.process_value(value)
                x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
                return np.ma.array(np.interp(value, x, y), mask=result.mask, copy=False)
         
        cmap = cmap_dict[vn]
        norm = MidpointNormalize(midpoint=midpt_dict[vn], vmin=colorbar_dict[vn][0], vmax=colorbar_dict[vn][1])  
            
        
        lons, lats, z_array = grid_values(vn, grouping, modelparams_all)
        if len(vns) > 1:
            ax[nvar].pcolormesh(lons, lats, z_array, cmap=cmap, norm=norm, zorder=2, alpha=0.8)
        else:
            ax.pcolormesh(lons, lats, z_array, cmap=cmap, norm=norm, zorder=2, alpha=0.8)
        
        # Set the extent
        ax[nvar].set_extent([east, west, south, north], cartopy.crs.PlateCarree())    
        # Label title, x, and y axes
        ax[nvar].set_xticks(np.arange(east,west+1,xtick), cartopy.crs.PlateCarree())
        ax[nvar].set_yticks(np.arange(south,north+1,ytick), cartopy.crs.PlateCarree())
        if nvar + 1 == len(vns):
            ax[nvar].set_xlabel(xlabel, size=labelsize, labelpad=0)
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm._A = []
        cbar = plt.colorbar(sm, ax=ax[nvar], fraction=0.03, pad=0.01)
        # Set tick marks manually
        if vn == 'dif_masschange':
            cbar.set_ticks(list(np.arange(colorbar_dict[vn][0], colorbar_dict[vn][1] + 0.01, 0.05)))
        elif vn == 'tempchange':
            cbar.set_ticks(list(np.arange(colorbar_dict[vn][0], colorbar_dict[vn][1] + 0.01, 0.5))[1:-1])
        ax[nvar].text(lons.max()+title_adj[vn], lats.mean(), vn_title_wunits_dict[vn], va='center', ha='center', 
                      rotation='vertical', size=labelsize)
        
        if option_group_regions == 1:
            rgi_shp = cartopy.io.shapereader.Reader(bolch_shp_fn)
            rgi_feature = cartopy.feature.ShapelyFeature(rgi_shp.geometries(), cartopy.crs.PlateCarree(),
                                                         edgecolor='lightgrey', facecolor='none', linewidth=1)
            ax[nvar].add_feature(rgi_feature, zorder=9)
            ax[nvar].text(101., 28.0, 'Hengduan\nShan', zorder=10, size=8, va='center', ha='center')
            ax[nvar].text(99.0, 26.5, 'Nyainqentanglha', zorder=10, size=8, va='center', ha='center')
            ax[nvar].plot([98,96], [27,29.3], color='k', linewidth=0.25, zorder=10)
            ax[nvar].text(93.0, 27.5, 'Eastern Himalaya', zorder=10, size=8, va='center', ha='center')
            ax[nvar].text(80.0, 27.3, 'Central Himalaya', zorder=10, size=8, va='center', ha='center')
            ax[nvar].text(72.0, 31.7, 'Western Himalaya', zorder=10, size=8, va='center', ha='center')
            ax[nvar].text(70.5, 33.7, 'Eastern\nHindu Kush', zorder=10, size=8, va='center', ha='center')
            ax[nvar].text(79.0, 39.7, 'Karakoram', zorder=10, size=8, va='center', ha='center')
            ax[nvar].plot([76,78], [36,39], color='k', linewidth=0.25, zorder=10)
            ax[nvar].text(80.7, 38.0, 'Western\nKunlun Shan', zorder=10, size=8, va='center', ha='center')
            ax[nvar].text(86.0, 33.7, 'Tibetan Interior\nMountains', zorder=10, size=8, va='center', ha='center')
            ax[nvar].text(73.0, 29.0, 'Gandise Mountains', zorder=10, size=8, va='center', ha='center')
            ax[nvar].plot([77.5,81.5], [29,31.4], color='k', linewidth=0.25, zorder=10)
            
        else:
            # Add country borders for reference
            ax[nvar].add_feature(cartopy.feature.BORDERS, facecolor='none', edgecolor='lightgrey', zorder=10)
            ax[nvar].add_feature(cartopy.feature.COASTLINE, facecolor='none', edgecolor='lightgrey', zorder=10)
            
        # Add contour lines and/or rgi outlines
        if option_contour_lines == 1:
            srtm_contour_shp = cartopy.io.shapereader.Reader(srtm_contour_fn)
            srtm_contour_feature = cartopy.feature.ShapelyFeature(srtm_contour_shp.geometries(), 
                                                                  cartopy.crs.PlateCarree(),
                                                                  edgecolor='lightgrey', facecolor='none', 
                                                                  linewidth=0.05)
            ax[nvar].add_feature(srtm_contour_feature, zorder=9)   
        if option_rgi_outlines == 1:
            rgi_shp = cartopy.io.shapereader.Reader(rgi_glac_shp_fn)
            rgi_feature = cartopy.feature.ShapelyFeature(rgi_shp.geometries(), cartopy.crs.PlateCarree(),
                                                         edgecolor='black', facecolor='none', linewidth=0.1)
            ax[nvar].add_feature(rgi_feature, zorder=9)    
    

    # Add subplot labels
    if len(vns) == 3:
        fig.text(0.21, 0.86, 'A', zorder=4, color='black', fontsize=12, fontweight='bold')
        fig.text(0.21, 0.605, 'B', zorder=4, color='black', fontsize=12, fontweight='bold')
        fig.text(0.21, 0.35, 'C', zorder=4, color='black', fontsize=12, fontweight='bold')
    elif len(vns) == 2:
        fig.text(0.21, 0.85, 'A', zorder=4, color='black', fontsize=12, fontweight='bold')
        fig.text(0.21, 0.46, 'B', zorder=4, color='black', fontsize=12, fontweight='bold')
    
    if len(vns) > 1:
        fig.text(0.1, 0.5, ylabel, va='center', rotation='vertical', size=12)
    
    # Save figure
    fig.set_size_inches(6,3*len(vns))
    if degree_size < 1:
        degsize_name = 'pt' + str(int(degree_size * 100))
    else:
        degsize_name = str(degree_size)
    fig_fn = fig_fn_prefix + 'mp_maps_' + degsize_name + 'deg_' + str(len(vns)) + 'params.png'
    fig.savefig(figure_fp + fig_fn, bbox_inches='tight', dpi=300)

    
#%%        
def observation_vs_calibration(regions, netcdf_fp, chainlength=chainlength, burn=0, chain_no=0, netcdf_fn=None):
    """
    Compare mass balance observations with model calibration
    
    Parameters
    ----------
    regions : list of strings
        list of regions
    chainlength : int
        chain length
    burn : int
        burn-in number

    Returns
    -------
    .png files
        saves histogram of differences between observations and calibration
    .csv file
        saves .csv file of comparison
    """
#%%
#for batman in [0]:
#    netcdf_fp = mcmc_output_netcdf_fp_all
#    chain_no = 0
    
    csv_fp = netcdf_fp + 'csv/'
    fig_fp = netcdf_fp + 'figures/'
    
    # Load mean of all model parameters
    if os.path.isfile(csv_fp + netcdf_fn) == False:
        
        filelist = []
        for region in regions:
            filelist.extend(glob.glob(netcdf_fp + str(region) + '*.nc'))
        
        glac_no = []
        reg_no = []
        for netcdf in filelist:
            glac_str = netcdf.split('/')[-1].split('.nc')[0]
            glac_no.append(glac_str)
            reg_no.append(glac_str.split('.')[0])
        glac_no = sorted(glac_no)
            
        (main_glac_rgi, main_glac_hyps, main_glac_icethickness, main_glac_width, 
         gcm_temp, gcm_prec, gcm_elev, gcm_lr, cal_data, dates_table) = load_glacierdata_byglacno(glac_no)
        
        posterior_cns = ['glacno', 'mb_mean', 'mb_std', 'pf_mean', 'pf_std', 'tc_mean', 'tc_std', 'ddfsnow_mean', 
                         'ddfsnow_std']
        
        posterior_all = pd.DataFrame(np.zeros((main_glac_rgi.shape[0], len(posterior_cns))), columns=posterior_cns)
        
        print('burn:', burn, 'chain length:', chainlength)
        
        for n, glac_str_wRGI in enumerate(main_glac_rgi['RGIId'].values):
            if n%500 == 0:
                print(n, glac_str_wRGI)
            # Glacier string
            glacier_str = glac_str_wRGI.split('-')[1]
            # MCMC Analysis
            ds = xr.open_dataset(netcdf_fp + glacier_str + '.nc')
            df = pd.DataFrame(ds['mp_value'].values[burn:chainlength,:,0], columns=ds.mp.values)  
            
            # Posteriors            
            posterior_row = [glacier_str, 
                             df.massbal.mean(), df.massbal.std(), 
                             df.precfactor.mean(), df.precfactor.std(), 
                             df.tempchange.mean(), df.tempchange.std(),
                             df.ddfsnow.mean(), df.ddfsnow.std()]
            posterior_all.loc[n,:] = posterior_row
            
            ds.close()
        modelparams_all = main_glac_rgi[['RGIId', 'CenLon', 'CenLat', 'O1Region', 'Area', 'RefDate', 'glacno', 
                                         'RGIId_float']]
        modelparams_all = pd.concat([modelparams_all, cal_data[['mb_mwe', 'mb_mwe_err', 't1', 't2', 'area_km2']]], 
                                    axis=1)
        modelparams_all['mb_mwea'] = cal_data['mb_mwe'] / (cal_data['t2'] - cal_data['t1'])
        modelparams_all['mb_mwea_err'] = cal_data['mb_mwe_err'] / (cal_data['t2'] - cal_data['t1'])
        modelparams_all = pd.concat([modelparams_all, posterior_all], axis=1)
        
        # Add region and priors
        modelparams_all['Region'] = modelparams_all.RGIId.map(input.reg_dict)
        # Priors
        # precipitation factor
        precfactor_alpha_dict = {region: input.precfactor_gamma_region_dict[region][0] 
                                 for region in list(input.precfactor_gamma_region_dict.keys())}
        precfactor_beta_dict = {region: input.precfactor_gamma_region_dict[region][1] 
                                 for region in list(input.precfactor_gamma_region_dict.keys())}
        modelparams_all['prior_pf_alpha'] = modelparams_all.Region.map(precfactor_alpha_dict) 
        modelparams_all['prior_pf_beta'] = modelparams_all.Region.map(precfactor_beta_dict)
        modelparams_all['prior_pf_mu'] = modelparams_all['prior_pf_alpha'] / modelparams_all['prior_pf_beta'] 
        modelparams_all['prior_pf_std'] = (modelparams_all['prior_pf_alpha'] / modelparams_all['prior_pf_beta']**2)**0.5
        # temperature change
        tempchange_mu_dict = {region: input.tempchange_norm_region_dict[region][0] 
                              for region in list(input.tempchange_norm_region_dict.keys())}
        tempchange_std_dict = {region: input.tempchange_norm_region_dict[region][1] 
                               for region in list(input.tempchange_norm_region_dict.keys())}
        modelparams_all['prior_tc_mu'] = modelparams_all.Region.map(tempchange_mu_dict) 
        modelparams_all['prior_tc_std'] = modelparams_all.Region.map(tempchange_std_dict)
        # degree-day factor of snow
        modelparams_all['prior_ddfsnow_mu'] = input.ddfsnow_mu * 1000
        modelparams_all['prior_ddfsnow_std'] = input.ddfsnow_sigma * 1000
        
        if os.path.exists(csv_fp) == False:
            os.makedirs(csv_fp)
        modelparams_all.to_csv(csv_fp + netcdf_fn, index=False)
        
    else:
        modelparams_all = pd.read_csv(csv_fp + netcdf_fn)

    
    #%%
    # Change column names to enable use of existing scripts
    modelparams_all['obs_mwea'] = modelparams_all['mb_mwea']
    modelparams_all['obs_mwea_std'] = modelparams_all['mb_mwea_err']
    modelparams_all['mod_mwea'] = modelparams_all['mb_mean']
    modelparams_all['mod_mwea_std'] = modelparams_all['mb_std']
    modelparams_all['Area_km2'] = modelparams_all['Area']

    mb_compare = modelparams_all.copy()
    
#    # Mass balance comparison: observations and model
#    mb_compare_cols = ['glacno', 'obs_mwea', 'obs_mwea_std', 'mod_mwea', 'mod_mwea_std', 'dif_mwea']
#    mb_compare = pd.DataFrame(np.zeros((len(glac_no), len(mb_compare_cols))), columns=mb_compare_cols)
#    mb_compare['glacno'] = glac_no
#    mb_compare['obs_mwea'] = cal_data['mb_mwe'] / (cal_data['t2'] - cal_data['t1'])
#    mb_compare['obs_mwea_std'] = cal_data['mb_mwe_err'] / (cal_data['t2'] - cal_data['t1'])
#    for nglac, glac in enumerate(glac_no):
#        # open dataset
#        if nglac%500 == 0:
#            print(nglac, glac)
#        ds = xr.open_dataset(netcdf_fp + glac + '.nc')
#        mb_all = ds['mp_value'].sel(chain=chain_no, mp='massbal').values[burn:chainlength]
#        mb_compare.loc[nglac, 'mod_mwea'] = np.mean(mb_all)
#        mb_compare.loc[nglac, 'mod_mwea_std'] = np.std(mb_all)
#        # close dataset
#        ds.close()
    
    #%%
    mb_compare['dif_mwea'] = mb_compare['mod_mwea'] - mb_compare['obs_mwea'] 
    mb_compare['dif_zscore'] = (mb_compare['mod_mwea'] - mb_compare['obs_mwea']) / mb_compare['obs_mwea_std']
#    mb_compare['Area_km2'] = main_glac_rgi['Area']
#    mb_compare['Zmin'] = main_glac_rgi['Zmin']
#    mb_compare['Zmax'] = main_glac_rgi['Zmax']
#    mb_compare['Zmed'] = main_glac_rgi['Zmed']
    mb_compare['obs_Gta'] = mb_compare['obs_mwea'] / 1000 * mb_compare['Area_km2']
    mb_compare['obs_Gta_std'] = mb_compare['obs_mwea_std'] / 1000 * mb_compare['Area_km2']
    mb_compare['mod_Gta'] = mb_compare['mod_mwea'] / 1000 * mb_compare['Area_km2']
    mb_compare['mod_Gta_std'] = mb_compare['mod_mwea_std'] / 1000 * mb_compare['Area_km2']
    
    print('Observed MB [Gt/yr]:', np.round(mb_compare.obs_Gta.sum(),2), 
          '(+/-', np.round(mb_compare.obs_Gta_std.sum(),2),')',
          '\nModeled MB [Gt/yr]:', np.round(mb_compare.mod_Gta.sum(),2),
          '(+/-', np.round(mb_compare.mod_Gta_std.sum(),2),')'
          )

    # ===== HISTOGRAM: mass balance difference ======
    dif_bins = [-1.5, -1, -0.5, -0.2, -0.1, -0.05,-0.02, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 1.5]
    bin_min = np.floor((mb_compare['dif_mwea'].min() * 100))/100
    bin_max = np.ceil((mb_compare['dif_mwea'].max() * 100))/100
    if bin_min < dif_bins[0]:
        dif_bins[0] = bin_min
    if bin_max > dif_bins[-1]:
        dif_bins[-1] = bin_max
    hist_fn = 'hist_' + str(int(chainlength/1000)) + 'kch_dif_mwea.png'
    plot_hist(mb_compare, 'dif_mwea', dif_bins, 
              xlabel='$\mathregular{B_{mod} - B_{obs}}$ (m w.e. $\mathregular{a^{-1}}$)', ylabel='Count',
              fig_fp=fig_fp, fig_fn=hist_fn)
    
    # ===== HISTOGRAM: z-score =====
    dif_bins = [-2,-1, -0.75, -0.5, -0.25, -0.1, 0.1, 0.25, 0.5, 0.75, 1, 2]
    bin_min = np.floor((mb_compare['dif_zscore'].min() * 100))/100
    bin_max = np.ceil((mb_compare['dif_zscore'].max() * 100))/100
    if bin_min < dif_bins[0]:
        dif_bins[0] = bin_min
    if bin_max > dif_bins[-1]:
        dif_bins[-1] = bin_max
    hist_fn = 'hist_' + str(int(chainlength/1000)) + 'kch_zscore.png'
    plot_hist(mb_compare, 'dif_zscore', dif_bins, 
              xlabel='z-score ($\\frac{B_{mod} - B_{obs}}{B_{std}}$)', ylabel='Count',
              fig_fp=fig_fp, fig_fn=hist_fn)
    
    
    # ===== Scatterplot: Glacier Area [km2] vs. Mass balance, color-coded by mass balance difference ===== 
    fig, ax = plt.subplots()
    cmap = 'RdYlBu'
#    cmap = plt.cm.get_cmap(cmap, 5)
    colorbar_dict = {'precfactor':[0,5],
                     'tempchange':[-5,5],
                     'ddfsnow':[2.6,5.6],
                     'massbal':[-1.5,0.5],
                     'dif_masschange':[-0.5,0.5],
                     'dif_zscore':[-1,1]}
    norm = plt.Normalize(colorbar_dict['dif_masschange'][0], colorbar_dict['dif_masschange'][1])    
    a = ax.scatter(mb_compare['Area_km2'], mb_compare['obs_mwea'], c=mb_compare['dif_mwea'], 
                   cmap=cmap, norm=norm, s=20, linewidth=0.5)
    a.set_facecolor('none')
    ax.set_xlim([0,200])
    ax.set_ylim([-2.5,1.25])
    ax.set_ylabel('$\mathregular{B_{obs}}$ $\mathregular{(m w.e. a^{-1})}$', size=12)
    ax.set_xlabel('Area ($\mathregular{km^{2}}$)', size=12)
    # Inset axis over main axis
    ax_inset = plt.axes([.35, .19, .48, .35])
    b = ax_inset.scatter(mb_compare['Area_km2'], mb_compare['obs_mwea'], c=mb_compare['dif_mwea'], 
                         cmap=cmap, norm=norm, s=20,linewidth=0.5)
    b.set_facecolor('none')
    ax_inset.set_xlim([0,5])
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm._A = []
    cbar = plt.colorbar(sm, ax=ax, fraction=0.04, pad=0.01)
    cbar.set_ticks(list(np.arange(colorbar_dict['dif_masschange'][0], colorbar_dict['dif_masschange'][1] + 0.01, 0.25)))
    fig.text(1.01, 0.5, '$\mathregular{B_{mod} - B_{obs}}$ (m w.e. $\mathregular{a^{-1}}$)', va='center',
             rotation='vertical', size=12)
    # Save figure
    fig.set_size_inches(6,4)
    fig_fn = 'MB_vs_area_wdif_scatterplot.png'
    fig.savefig(fig_fp + fig_fn, bbox_inches='tight', dpi=300)
    
    # ===== Scatterplot: Glacier Area [km2] vs. Mass balance, color-coded by Z-SCORE difference =====
    fig, ax = plt.subplots()
    cmap = 'RdYlBu'
#    cmap = plt.cm.get_cmap(cmap, 5)
    norm = plt.Normalize(colorbar_dict['dif_zscore'][0], colorbar_dict['dif_zscore'][1])    
    a = ax.scatter(mb_compare['Area_km2'], mb_compare['obs_mwea'], c=mb_compare['dif_zscore'], 
                   cmap=cmap, norm=norm, s=20, linewidth=0.5)
    a.set_facecolor('none')
    ax.set_xlim([0,200])
    ax.set_ylim([-2.5,1.25])
    ax.set_ylabel('$\mathregular{B_{obs}}$ $\mathregular{(m w.e. a^{-1})}$', size=12)
    ax.set_xlabel('Area ($\mathregular{km^{2}}$)', size=12)
    # Inset axis over main axis
    ax_inset = plt.axes([.35, .19, .48, .35])
    b = ax_inset.scatter(mb_compare['Area_km2'], mb_compare['obs_mwea'], facecolor='None', c=mb_compare['dif_zscore'], 
                     cmap=cmap, norm=norm, s=20,linewidth=0.5)
    b.set_facecolor('none')
    ax_inset.set_xlim([0,5])
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm._A = []
    cbar = plt.colorbar(sm, ax=ax, fraction=0.04, pad=0.01)
    cbar.set_ticks(list(np.arange(colorbar_dict['dif_zscore'][0], colorbar_dict['dif_zscore'][1] + 0.01, 0.25)))
    fig.text(1.01, 0.5, 'z-score ($\\frac{B_{mod} - B_{obs}}{B_{std}}$)', va='center',
             rotation='vertical', size=12)
    # Save figure
    fig.set_size_inches(6,4)
    fig_fn = 'MB_vs_area_wdif_scatterplot_zscore.png'
    fig.savefig(fig_fp + fig_fn, bbox_inches='tight', dpi=300)
    

    # ===== Scatterplot: Glacier Area [km2] vs. mass balance difference, color-coded by Mass balance =====
    class MidpointNormalize(colors.Normalize):
        def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
            self.midpoint = midpoint
            colors.Normalize.__init__(self, vmin, vmax, clip)
    
        def __call__(self, value, clip=None):
            # Note that I'm ignoring clipping and other edge cases here.
            result, is_scalar = self.process_value(value)
            x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
            return np.ma.array(np.interp(value, x, y), mask=result.mask, copy=False)
    
    fig, ax = plt.subplots()
    cmap = 'RdYlBu'
#    cmap = plt.cm.get_cmap(cmap, 5)
#    norm = plt.Normalize(colorbar_dict['massbal'][0], colorbar_dict['massbal'][1])   
    norm = MidpointNormalize(midpoint=0, vmin=colorbar_dict['massbal'][0], vmax=colorbar_dict['massbal'][1])  
    a = ax.scatter(mb_compare['Area_km2'], mb_compare['dif_mwea'], c=mb_compare['obs_mwea'], 
                   cmap=cmap, norm=norm, s=20, linewidth=0.5)
    a.set_facecolor('none')
    ax.set_xlim([0,200])
    ax.set_ylim([-2.49,1.75])
    ax.set_ylabel('$\mathregular{B_{mod} - B_{obs}}$ (m w.e. $\mathregular{a^{-1}}$)', size=12)
    ax.set_xlabel('Area ($\mathregular{km^{2}}$)', size=12)
    # Inset axis over main axis
    ax_inset = plt.axes([.35, .19, .48, .35])
    a = ax_inset.scatter(mb_compare['Area_km2'], mb_compare['dif_zscore'], c=mb_compare['obs_mwea'], 
                     cmap=cmap, norm=norm, s=20,linewidth=0.5)
    a.set_facecolor('none')
    ax_inset.set_xlim([0,5])
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm._A = []
    cbar = plt.colorbar(sm, ax=ax, fraction=0.04, pad=0.01)
    cbar.set_ticks(list(np.arange(colorbar_dict['massbal'][0], colorbar_dict['massbal'][1] + 0.01, 0.25)))
    fig.text(1.01, 0.5, '$\mathregular{B_{obs}}$ $\mathregular{(m w.e. a^{-1})}$', va='center',
             rotation='vertical', size=12)
    # Save figure
    fig.set_size_inches(6,4)
    fig_fn = 'dif_vs_area_wMB_scatterplot.png'
    fig.savefig(fig_fp + fig_fn, bbox_inches='tight', dpi=300)
    
    # ===== Scatterplot: Glacier Area [km2] vs. Z-SCORE DIFFERENCE, color-coded by Mass balance =====
    fig, ax = plt.subplots()
    cmap = 'RdYlBu'
#    cmap = plt.cm.get_cmap(cmap, 5)
#    norm = plt.Normalize(colorbar_dict['massbal'][0], colorbar_dict['massbal'][1])   
    norm = MidpointNormalize(midpoint=0, vmin=colorbar_dict['massbal'][0], vmax=colorbar_dict['massbal'][1])  
    a = ax.scatter(mb_compare['Area_km2'], mb_compare['dif_zscore'], c=mb_compare['obs_mwea'], 
                   cmap=cmap, norm=norm, s=20, linewidth=0.5)
    a.set_facecolor('none')
    ax.set_xlim([0,200])
    ax.set_ylim([-3.99,2.5])
    ax.set_ylabel('z-score ($\\frac{B_{mod} - B_{obs}}{B_{std}}$)', size=12)
    ax.set_xlabel('Area ($\mathregular{km^{2}}$)', size=12)
    # Inset axis over main axis
    ax_inset = plt.axes([.35, .19, .48, .35])
    a = ax_inset.scatter(mb_compare['Area_km2'], mb_compare['dif_zscore'], c=mb_compare['obs_mwea'], 
                     cmap=cmap, norm=norm, s=10,linewidth=0.5)
    a.set_facecolor('none')
    ax_inset.set_xlim([0,5])
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm._A = []
    cbar = plt.colorbar(sm, ax=ax, fraction=0.04, pad=0.01)
    cbar.set_ticks(list(np.arange(colorbar_dict['massbal'][0], colorbar_dict['massbal'][1] + 0.01, 0.25)))
    fig.text(1.01, 0.5, '$\mathregular{B_{obs}}$ $\mathregular{(m w.e. a^{-1})}$', va='center',
             rotation='vertical', size=12)
    # Save figure
    fig.set_size_inches(6,4)
    fig_fn = 'dif_vs_area_wMB_scatterplot_zscore.png'
    fig.savefig(fig_fp + fig_fn, bbox_inches='tight', dpi=300)
    

#%%
if __name__ == '__main__':

    #%%         
    if option_metrics_vs_chainlength == 1:
    #    metrics_vs_chainlength(mcmc_output_netcdf_fp_3chain, regions, iterations, burn=burn, nchain=3, 
    #                           option_subplot_labels=1)
        netcdf_fp = mcmc_output_netcdf_fp_3chain
        fig_fp = netcdf_fp + 'figures/'
        csv_fp = netcdf_fp + 'csv/'
        nchain = 3
    #    burn = 1000
    #    iterstep = 5000
    #    itermax = 25000
        burn = 0
        iterstep = 2000
        itermax = 25000
        iterations = np.arange(0, itermax, iterstep)
        if iterations[0] < 1000: 
            iterations[0] = burn + 1000
        else:
            iterations = iterations[1:]
        if iterations[-1] != itermax:
            iterations = np.append(iterations, itermax)
        iters = iterations
        option_mcerror_normalize = 1
        option_subplot_labels = 0
        metrics = ['Gelman-Rubin', 'MC Error', 'Effective N']
        if nchain == 1:
            metrics.remove('Gelman-Rubin')
        low_percentile = 10
        high_percentile = 90
        
        print('iterations:', iterations)
        
        # File names
        en_fn_pkl = 'effective_n_list.pkl'
        mc_fn_pkl = 'mc_error_list.pkl'
        gr_fn_pkl = 'gelman_rubin_list.pkl'
        postmean_fn_pkl = 'postmean_list.pkl'
        poststd_fn_pkl = 'poststd_list.pkl'
        glacno_fn_pkl = 'glacno_list.pkl'
        iter_ending = '_' + str(iterstep) + 'iterstep_' + str(burn) + 'burn.pkl'
        
        # Check if files exist
        if os.path.isfile(csv_fp + en_fn_pkl.replace('.pkl', iter_ending)):
            with open(csv_fp + en_fn_pkl.replace('.pkl', iter_ending), 'rb') as f:
                en_list = pickle.load(f)
            with open(csv_fp + mc_fn_pkl.replace('.pkl', iter_ending), 'rb') as f:
                mc_list = pickle.load(f)
            if nchain > 1:
                with open(csv_fp + gr_fn_pkl.replace('.pkl', iter_ending), 'rb') as f:
                    gr_list = pickle.load(f)
            with open(csv_fp + postmean_fn_pkl.replace('.pkl', iter_ending), 'rb') as f:
                postmean_list = pickle.load(f)
            with open(csv_fp + poststd_fn_pkl.replace('.pkl', iter_ending), 'rb') as f:
                poststd_list = pickle.load(f)
            with open(csv_fp + glacno_fn_pkl, 'rb') as f:
                glac_no = pickle.load(f)
                
        # Otherwise, process and pickle data
        else:
            # Lists to record metrics
            glac_no = []
            en_list = {}
            gr_list = {}
            mc_list = {}
            postmean_list = {}
            poststd_list = {}
            
            # Load netcdf filenames    
            filelist = []
            for region in regions:
                filelist.extend(glob.glob(netcdf_fp + str(region) + '*.nc'))  
            filelist = sorted(filelist)
            
            # iterate through each glacier
            count = 0
            for count, netcdf in enumerate(filelist):
    #        for count, netcdf in enumerate(filelist[0:100]):
                glac_str = netcdf.split('/')[-1].split('.nc')[0]
                glac_no.append(glac_str)
    #            if count%100 == 0:
                print(count, glac_str)
                
                en_list[glac_str] = {}
                gr_list[glac_str] = {}
                mc_list[glac_str] = {}
                postmean_list[glac_str] = {}
                poststd_list[glac_str] = {}
                
                # open dataset
                ds = xr.open_dataset(netcdf)
    
                # Metrics for each parameter
                for nvar, vn in enumerate(variables):
                    
                    # Effective sample size
                    if 'Effective N' in metrics:
                        en = [effective_n(ds, vn=vn, iters=i, burn=burn) for i in iters]                
                        en_list[glac_str][vn] = dict(zip(iters, en))
                    
                    if 'MC Error' in metrics:
                        # Monte Carlo error
                        # the first [0] extracts the MC error as opposed to the confidence interval
                        # the second [0] extracts the first chain
                        mc = [mc_error(ds, vn=vn, iters=i, burn=burn, method='overlapping')[0][0] for i in iters]
                        mc_list[glac_str][vn] = dict(zip(iters, mc))
    
                    # Gelman-Rubin Statistic                
                    if len(ds.chain) > 1 and 'Gelman-Rubin' in metrics:
                        gr = [gelman_rubin(ds, vn=vn, iters=i, burn=burn) for i in iters]
                        gr_list[glac_str][vn] = dict(zip(iters, gr))
                        
                # Posteriors
                for nvar, vn in enumerate(variables):
                    postmean_list[glac_str][vn] = {}
                    poststd_list[glac_str][vn] = {}
                
                for n_iters in iterations:
                    df = pd.DataFrame(ds['mp_value'].values[burn:n_iters,:,0], columns=ds.mp.values)
                    
                    postmean_list[glac_str]['massbal'][n_iters] = df.massbal.mean() 
                    postmean_list[glac_str]['precfactor'][n_iters] = df.precfactor.mean() 
                    postmean_list[glac_str]['tempchange'][n_iters] = df.tempchange.mean() 
                    postmean_list[glac_str]['ddfsnow'][n_iters] = df.ddfsnow.mean() 
                    poststd_list[glac_str]['massbal'][n_iters] = df.massbal.std() 
                    poststd_list[glac_str]['precfactor'][n_iters] = df.precfactor.std() 
                    poststd_list[glac_str]['tempchange'][n_iters] = df.tempchange.std() 
                    poststd_list[glac_str]['ddfsnow'][n_iters] = df.ddfsnow.std() 
        
                # close datase
                ds.close()
                
            # Pickle lists for next time
            if os.path.exists(csv_fp) == False:
                os.makedirs(csv_fp)
                    
            pickle_data(csv_fp + en_fn_pkl.replace('.pkl', iter_ending), en_list)
            pickle_data(csv_fp + mc_fn_pkl.replace('.pkl', iter_ending), mc_list)
            if len(ds.chain) > 1:
                pickle_data(csv_fp + gr_fn_pkl.replace('.pkl', iter_ending), gr_list)
            pickle_data(csv_fp + postmean_fn_pkl.replace('.pkl', iter_ending), postmean_list)
            pickle_data(csv_fp + poststd_fn_pkl.replace('.pkl', iter_ending), poststd_list)
            pickle_data(csv_fp + glacno_fn_pkl, glac_no)
        
            
    #def metrics_vs_chainlength(netcdf_fp, regions, iters, burn=0, nchain=3, option_subplot_labels=0):
    #    """
    #    Plot Gelman-Rubin, Monte Carlo error, and effective sample size for each parameter for various chain lengths
    #
    #    Parameters
    #    ----------
    #    regions : list of strings
    #        list of regions
    #    iters : list of ints
    #        list of the number of iterations to compute metrics for
    #    burn : int
    #        burn-in number
    #
    #    Returns
    #    -------
    #    .png file
    #        saves figure of how metrics change according to the number of mcmc iterations
    #    .pkl files
    #        saves .pkl files of the metrics for various iterations (if they don't already exist)
    #    """
    ###%%
        
        #%%
            
        # ===== PLOT METRICS =====
    #    colors = ['#387ea0', '#fcb200', '#d20048']
        colors = ['black', 'black', 'black']
        fillcolors = ['lightgrey', 'lightgrey', 'lightgrey']
        figwidth=6.5
        figheight=8
        fig, ax = plt.subplots(len(variables), len(metrics), squeeze=False, sharex=False, sharey=False,
                               figsize=(figwidth,figheight), gridspec_kw = {'wspace':0.4, 'hspace':0.25})        
        
        # Metric statistics
        df_cns = ['iters', 'mean', 'std', 'median', 'lowbnd', 'highbnd']
    
        for nmetric, metric in enumerate(metrics):
    
            if metric == 'Effective N':
                metric_list = en_list
    
            elif metric == 'MC Error':             
                if option_mcerror_normalize == 0:
                    metric_list = mc_list
                else:
                    metric_list = {}
                    for glac_str in glac_no:
                        metric_list[glac_str] = {}
                        for nvar, vn in enumerate(variables):
                            metric_list[glac_str][vn] = {}
                            for niter, iteration in enumerate(iterations):
                                metric_list[glac_str][vn][iteration] = (mc_list[glac_str][vn][iteration] / 
                                                                        poststd_list[glac_str][vn][iteration])
            elif metric == 'Gelman-Rubin':
                metric_list = gr_list
                
            for nvar, vn in enumerate(variables):
                metric_df = pd.DataFrame(np.zeros((len(iterations), len(df_cns))), columns=df_cns)
                metric_df['iters'] = iterations
                
                for niter, iteration in enumerate(iterations):
                    iter_list = [metric_list[i][vn][iteration] for i in glac_no]
                    
                    
                    metric_df.loc[niter,'mean'] = np.mean(iter_list)
                    metric_df.loc[niter,'median'] = np.median(iter_list)
                    metric_df.loc[niter,'std'] = np.std(iter_list)
                    metric_df.loc[niter,'lowbnd'] = np.percentile(iter_list,low_percentile)
                    metric_df.loc[niter,'highbnd'] = np.percentile(iter_list,high_percentile)
                
                if metric == 'MC Error':
                    metric_idx = np.where(metric_df.iters == 10000)[0][0]
                    print(metric, vn, '\n', metric_df.loc[metric_idx,'highbnd'])
                elif metric == 'Effective N':
                    metric_idx = np.where(metric_df.iters == 10000)[0][0]
                    print(metric, vn, '\n', metric_df.loc[metric_idx,'lowbnd'])
                    
            
                # ===== Plot =====
                if vn == 'ddfsnow' and metric == 'MC Error' and option_mcerror_normalize == 0:
                    ax[nvar,nmetric].plot(metric_df['iters']/10**3, metric_df['median']*10**3, color=colors[nmetric])
                    ax[nvar,nmetric].fill_between(metric_df['iters']/10**3, metric_df['lowbnd']*10**3, metric_df['highbnd']*10**3, 
                                                  color=fillcolors[nmetric], alpha=0.5)
                else:
                    ax[nvar,nmetric].plot(metric_df['iters']/10**3, metric_df['median'], color=colors[nmetric])
                    ax[nvar,nmetric].fill_between(metric_df['iters']/10**3, metric_df['lowbnd'], metric_df['highbnd'], 
                                                  color=fillcolors[nmetric], alpha=0.5)
                
                # niceties
                ax[nvar,nmetric].xaxis.set_major_locator(MultipleLocator(10))
                ax[nvar,nmetric].xaxis.set_minor_locator(MultipleLocator(2))
                if nvar == 0:
                    ax[nvar,nmetric].set_title(metric_title_dict[metric], fontsize=10)
                elif nvar == len(variables) - 1:
                    ax[nvar,nmetric].set_xlabel('Steps ($10^3$)', fontsize=12)
                
                    
                if metric == 'Gelman-Rubin':
                    ax[nvar,nmetric].set_ylabel(vn_title_dict[vn], fontsize=12, labelpad=10)
                    ax[nvar,nmetric].set_ylim(1,1.12)
                    ax[nvar,nmetric].axhline(y=1.1, color='k', linestyle='--', linewidth=2)
                    ax[nvar,nmetric].yaxis.set_major_locator(MultipleLocator(0.05))
                    ax[nvar,nmetric].yaxis.set_minor_locator(MultipleLocator(0.01))
                elif metric == 'MC Error':
                    if option_mcerror_normalize == 1:
                        ax[nvar,nmetric].axhline(y=0.1, color='k', linestyle='--', linewidth=2)
                        ax[nvar,nmetric].set_ylim(0,0.12)
                        ax[nvar,nmetric].yaxis.set_major_locator(MultipleLocator(0.05))
                        ax[nvar,nmetric].yaxis.set_minor_locator(MultipleLocator(0.01))
                    else:
                        if vn == 'massbal':
                            ax[nvar,nmetric].axhline(y=0.0026, color='k', linestyle='--', linewidth=2)
                            ax[nvar,nmetric].set_ylim(0,0.012)
                            ax[nvar,nmetric].yaxis.set_major_locator(MultipleLocator(0.005))
                            ax[nvar,nmetric].yaxis.set_minor_locator(MultipleLocator(0.001))
                        elif vn == 'precfactor':
                            ax[nvar,nmetric].axhline(y=0.026, color='k', linestyle='--', linewidth=2)
                            ax[nvar,nmetric].set_ylim(0,0.12)
                            ax[nvar,nmetric].yaxis.set_major_locator(MultipleLocator(0.05))
                            ax[nvar,nmetric].yaxis.set_minor_locator(MultipleLocator(0.01))
                        elif vn == 'tempchange':
                            ax[nvar,nmetric].axhline(y=0.026, color='k', linestyle='--', linewidth=2)
                            ax[nvar,nmetric].set_ylim(0,0.12)
                            ax[nvar,nmetric].yaxis.set_major_locator(MultipleLocator(0.05))
                            ax[nvar,nmetric].yaxis.set_minor_locator(MultipleLocator(0.01))
                        elif vn == 'ddfsnow':
                            ax[nvar,nmetric].axhline(y=0.026, color='k', linestyle='--', linewidth=2)
                            ax[nvar,nmetric].set_ylim(0,0.12)
                            ax[nvar,nmetric].yaxis.set_major_locator(MultipleLocator(0.05))
                            ax[nvar,nmetric].yaxis.set_minor_locator(MultipleLocator(0.01))
                elif metric == 'Effective N':
                    ax[nvar,nmetric].set_ylim(0,1200)
                    ax[nvar,nmetric].axhline(y=100, color='k', linestyle='--', linewidth=2)
                    ax[nvar,nmetric].yaxis.set_major_locator(MultipleLocator(500))
                    ax[nvar,nmetric].yaxis.set_minor_locator(MultipleLocator(100))
        
        if option_subplot_labels == 1:
            fig.text(0.130, 0.86, 'A', size=12)
            fig.text(0.415, 0.86, 'B', size=12)
            fig.text(0.700, 0.86, 'C', size=12)
            fig.text(0.130, 0.66, 'D', size=12)
            fig.text(0.415, 0.66, 'E', size=12)
            fig.text(0.700, 0.66, 'F', size=12)
            fig.text(0.130, 0.4625, 'G', size=12)
            fig.text(0.415, 0.4625, 'H', size=12)
            fig.text(0.700, 0.4625, 'I', size=12)
            fig.text(0.130, 0.265, 'J', size=12)
            fig.text(0.415, 0.265, 'K', size=12)
            fig.text(0.700, 0.265, 'L', size=12)
                    
        # Save figure
        fig.set_size_inches(figwidth,figheight)
        if os.path.exists(fig_fp) == False:
            os.makedirs(fig_fp)
        figure_fn = 'chainlength_vs_metrics' + iter_ending.replace('.pkl','') + '.png'
        fig.savefig(fig_fp + figure_fn, bbox_inches='tight', dpi=300)


    #%%
    if option_metrics_histogram_all == 1:
        iters = 10000
        burn = 0
        netcdf_fp = mcmc_output_netcdf_fp_all
        figure_fp = netcdf_fp + '/figures/'
        csv_fp = netcdf_fp + '/csv/'
        regions = [13, 14, 15]
        option_merge_regions = 0
        
        metrics = ['MC Error', 'Effective N']
    #    metrics = ['MC Error']
        
        en_fn_pkl = csv_fp + 'effective_n_list.pkl'
        mc_fn_pkl = csv_fp + 'mc_error_list.pkl'
        glacno_fn_pkl = csv_fp + 'glacno_list.pkl'
        iter_ending = '_' + str(iters) + 'iters_' + str(burn) + 'burn.pkl'
        
        modelparams_fn = 'main_glac_rgi_20190806_wcal_wposteriors_all_' + str(burn) + 'burn.csv'
        modelparams_all = pd.read_csv(csv_fp + modelparams_fn)
        
        if option_merge_regions == 1:
            # Manually merge, since better to run files through each region due to their size
            glac_no = []
            en_list = {}
            mc_list = {}        
            
            for region in regions:
                # Glacier number
                glacno_fn_pkl_region = csv_fp + 'R' + str(region) + '_glacno_list.pkl'
                with open(glacno_fn_pkl_region, 'rb') as f:
                    glac_no_region = pickle.load(f)
                glac_no += glac_no_region
                
                en_fn_pkl_region = csv_fp + 'R' + str(region) + '_effective_n_list.pkl'
                with open(en_fn_pkl_region.replace('.pkl', iter_ending), 'rb') as f:
                    en_list_region = pickle.load(f)
                en_list = {**en_list, **en_list_region}
                
                mc_fn_pkl_region = csv_fp + 'R' + str(region) + '_mc_error_list.pkl'
                with open(mc_fn_pkl_region.replace('.pkl', iter_ending), 'rb') as f:
                    mc_list_region = pickle.load(f)
                mc_list = {**mc_list, **mc_list_region}
                
            pickle_data(en_fn_pkl.replace('.pkl', iter_ending), en_list)
            pickle_data(mc_fn_pkl.replace('.pkl', iter_ending), mc_list)
            pickle_data(glacno_fn_pkl, glac_no)
    
        # Check if list already exists
        en_fn_pkl.replace('.pkl', iter_ending)
        
        if os.path.isfile(en_fn_pkl.replace('.pkl', iter_ending)):
            with open(en_fn_pkl.replace('.pkl', iter_ending), 'rb') as f:
                en_list = pickle.load(f)
            with open(mc_fn_pkl.replace('.pkl', iter_ending), 'rb') as f:
                mc_list = pickle.load(f)
            with open(glacno_fn_pkl, 'rb') as f:
                glac_no = pickle.load(f)
        else:
            # Load netcdf filenames    
            filelist = []
            for region in regions:
                filelist.extend(glob.glob(netcdf_fp + str(region) + '*.nc'))
            filelist = sorted(filelist)
            
            # ===== CALCULATE METRICS =====
            # Lists to record metrics
            glac_no = []
            en_list = {}
            gr_list = {}
            mc_list = {}
                
            # iterate through each glacier
            for n, netcdf in enumerate(filelist):
    
                glac_str = netcdf.split('/')[-1].split('.nc')[0]
                
    #            if glac_str.startswith('13.'):
                if n%100 == 0:
                    print(n, glac_str)
                glac_no.append(glac_str)
            
                en_list[glac_str] = {}
                mc_list[glac_str] = {}
                
                # open dataset
                ds = xr.open_dataset(netcdf)
        
                # Metrics for each parameter
                for nvar, vn in enumerate(variables):
                    # Effective sample size
                    try:
                        en = effective_n(ds, vn=vn, iters=iters, burn=burn) 
                    except:
                        en = 0
                    en_list[glac_str][vn] = en
                    # Monte Carlo error
                    # the first [0] extracts the MC error as opposed to the confidence interval
                    # the second [0] extracts the first chain
                    mc = mc_error(ds, vn=vn, iters=iters, burn=burn, method='overlapping')[0][0]
                    mc_list[glac_str][vn] = mc
        
                # close datase
                ds.close()
                
            # Pickle lists for next time
            pickle_data(en_fn_pkl.replace('.pkl', iter_ending), en_list)
            pickle_data(mc_fn_pkl.replace('.pkl', iter_ending), mc_list)
            pickle_data(glacno_fn_pkl, glac_no)
                    
            
        #%%
        # ===== PLOT METRICS =====
    #    colors = ['#fcb200', '#d20048']
        metric_colors = ['lightgrey', 'lightgrey']
        figwidth=6.5
        figheight=8
        
        # bins and ticks
        bdict = {}
        tdict = {}
        major = {}
        minor = {}
        
        bdict['MC Error massbal'] = [0, 0.025, 0.001]
        bdict['MC Error precfactor'] = [0, 0.11, 0.005]
        bdict['MC Error tempchange'] = [0, 0.11, 0.005]
        bdict['MC Error ddfsnow'] = [0.0, 0.11, 0.005]
        bdict['Effective N massbal'] = [0, 5000, 200]
        bdict['Effective N precfactor'] = [0, 2200, 100]
        bdict['Effective N tempchange'] = [0, 2200, 100]
        bdict['Effective N ddfsnow'] = [0, 2200, 100]
        tdict['MC Error'] = np.arange(0, 26, 5)
        tdict['Effective N'] = np.arange(0, 26, 5)
        
        fig, ax = plt.subplots(len(variables), len(metrics), squeeze=False, sharex=False, sharey=False,
                               figsize=(figwidth,figheight), gridspec_kw = {'wspace':0.1, 'hspace':0.5})      
    
        for nmetric, metric in enumerate(metrics):
            if metric == 'Effective N':
                metric_list = en_list
            elif metric == 'MC Error':
                metric_list = mc_list
                
            for nvar, vn in enumerate(variables):
    #        for nvar, vn in enumerate(['massbal']):
    
                metric_vn_list = [metric_list[i][vn] for i in glac_no]
                
    #            # Adjust ddfsnow units [*10^3, so mm w.e.]
    #            if metric == 'MC Error' and vn == 'ddfsnow':
    #                metric_vn_list = [i * 10**3 for i in metric_vn_list]
    #            vn_label_units_dict = {'massbal':'[mwea]',                                                                      
    #                                   'precfactor':'[-]',                                                              
    #                                   'tempchange':'[$^\circ$C]',                                                               
    #                                   'ddfsnow':'[mm w.e. d$^{-1}$ $^\circ$C$^{-1}$]'}
                vn_label_units_dict = {'massbal':'[mwea]',                                                                      
                                       'precfactor':'[-]',                                                              
                                       'tempchange':'[$^\circ$C]',                                                               
                                       'ddfsnow':'[mm w.e. d$^{-1}$ $^\circ$C$^{-1}$]'}
                
                
                if metric == 'MC Error':
                    metric_vn_array = np.array(metric_vn_list)
                    norm_cn_dict = {'massbal': 'mb_mwea_err',
                                    'precfactor': 'pf_std',
                                    'tempchange': 'tc_std',
                                    'ddfsnow': 'ddfsnow_std'}
                    metric_vn_array_norm = metric_vn_array / modelparams_all[norm_cn_dict[vn]].values
                    metric_vn_list = list(metric_vn_array_norm)
                
                # Remove nan values
                metric_vn_list_nonan = [x for x in metric_vn_list if str(x) != 'nan']
                
                
                # ===== Plot =====
                # compute histogram and change to percentage of glaciers
                
                metric_vn = metric + ' ' + vn
                metric_vn_bins = np.arange(bdict[metric_vn][0], bdict[metric_vn][1], bdict[metric_vn][2])
                hist, bins = np.histogram(metric_vn_list_nonan, bins=metric_vn_bins)
                hist = hist * 100.0 / hist.sum()
    
                # plot histogram
                ax[nvar,nmetric].bar(x=bins[:-1] + bdict[metric_vn][2] /2, height=hist, width=(bins[1]-bins[0]), 
                                     align='center', edgecolor='black', color=metric_colors[nmetric])
                
                
                # create uniform bins based on metric
                ax[nvar,nmetric].set_yticks(tdict[metric])
                
                # find cumulative percentage and plot it
                ax2 = ax[nvar,nmetric].twinx()
                cum_hist = [hist[0:i].sum() for i in range(len(hist))]
                if metric=='Effective N':
    #                percent, q = 5, 0.05
                    percent, q = 10, 0.1
                else:
    #                percent, q = 95, 0.95
                    percent, q = 90, 0.9
                index = 0
                quantile = np.percentile(metric_vn_list_nonan,percent)
                ax2.plot(bins[:-1], cum_hist, color='black', linewidth=1.25, label='Cumulative %')
                ax2.set_yticks(np.arange(0, 110, 20))
                ax2.axvline(quantile, color='black', linewidth=1.25, linestyle='--')
                ax2.set_ylim([0,100])
                
                print(metric, vn, quantile)
                    
                # axis labels
                if nmetric == 0:
                    ax2.yaxis.set_major_formatter(plt.NullFormatter())
                    ax2.set_yticks([])
                if nmetric == 1:
                    ax[nvar,nmetric].set_yticks([])
    #            if metric == 'MC Error':
    #                ax[nvar,nmetric].set_xlabel(vn_label_dict[vn], fontsize=10, labelpad=1)
    #            elif metric == 'Effective N':
    #                ax[nvar,nmetric].set_xlabel(vn_title_noabbreviations_dict[vn] + ' (-)', fontsize=10, labelpad=1)
    #                if vn is not 'massbal':
    #                    ax[nvar,nmetric].set_xticks(np.arange(0,2100,500))
                ax[nvar,nmetric].set_xlabel(vn_title_noabbreviations_dict[vn] + ' (-)', fontsize=10, labelpad=1)
                if vn is not 'massbal' and metric == 'Effective N':
                    ax[nvar,nmetric].set_xticks(np.arange(0,2100,500))
                
                # niceties
                if nvar == 0:
                    ax[nvar,nmetric].set_title(metric_title_dict[metric], fontsize=12)
        
        fig.text(0.04, 0.5, 'Count (%)', va='center', rotation='vertical', size=12)
        fig.text(0.96, 0.5, 'Cumulative Count (%)', va='center', rotation='vertical', size=12)
        
        fig.text(0.135, 0.86, 'A', size=12)
        fig.text(0.540, 0.86, 'B', size=12)
        fig.text(0.135, 0.655, 'C', size=12)
        fig.text(0.540, 0.655, 'D', size=12)
        fig.text(0.135, 0.445, 'E', size=12)
        fig.text(0.540, 0.445, 'F', size=12)
        fig.text(0.135, 0.24, 'G', size=12)
        fig.text(0.540, 0.24, 'H', size=12)
                    
        # Save figure
        fig.set_size_inches(6.5,8)
        figure_fn = 'histograms_all' + str(burn) + 'burn.png'
        fig.savefig(figure_fp + figure_fn, bbox_inches='tight', dpi=300)
        #%%
            
            
    if option_observation_vs_calibration == 1:
    #    netcdf_fp = mcmc_output_netcdf_fp_3chain
        netcdf_fp = mcmc_output_netcdf_fp_all
        burn = 1000
        mb_compare_fn = 'main_glac_rgi_20190806_wcal_wposteriors_all_' + str(burn) + 'burn.csv'
        observation_vs_calibration(regions, netcdf_fp, chainlength=chainlength, burn=burn, netcdf_fn=mb_compare_fn)
            
            
    #%%
    if option_papermcmc_prior_vs_posterior == 1:
        print('Prior vs posterior showing two example glaciers side-by-side!')
        glac_no = ['13.26360', '14.08487']
        netcdf_fp = mcmc_output_netcdf_fp_3chain
        netcdf_fp = input.output_filepath + 'cal_opt2_3chain/'
        burn = 1000
        iters=[2000,10000]
        figure_fn = 'prior_v_posteriors_2glac.eps'
        
    #    glac_no = ['15.10755', '15.12457']
    #    burn = 1000
    #    iters=[10000]
    #    netcdf_fp = mcmc_output_netcdf_fp_all
    #    figure_fn = 'prior_v_posteriors_2glac_poorglaciers.eps'
    #    # note: need to change position and lines of legend below 
        
        fig_fp = netcdf_fp + 'figures/'
        if os.path.exists(fig_fp) == False:
            os.makedirs(fig_fp)
        
        iter_colors = ['#387ea0', '#fcb200', '#d20048']
        
        main_glac_rgi, cal_data = load_glacierdata_byglacno(glac_no, option_loadhyps_climate=0)
        # Add regions
        main_glac_rgi['region'] = main_glac_rgi.RGIId.map(input.reg_dict)
    
        # PRIOR VS POSTERIOR PLOTS 
        fig, ax = plt.subplots(4, 2, squeeze=False, figsize=(6.5, 7), 
                               gridspec_kw={'wspace':0.2, 'hspace':0.47})
    
        for n, glac_str_wRGI in enumerate(main_glac_rgi['RGIId'].values):
            # Glacier string
            glacier_str = glac_str_wRGI.split('-')[1]
            print(glacier_str)
            # Glacier number
            glacno = int(glacier_str.split('.')[1])
            # RGI information
            glacier_rgi_table = main_glac_rgi.loc[main_glac_rgi.index.values[n], :]
            # Calibration data
            cal_idx = np.where(cal_data['glacno'] == glacno)[0]
            glacier_cal_data = (cal_data.iloc[cal_idx,:]).copy()
            # Select observed mass balance, error, and time data
            t1 = glacier_cal_data.loc[cal_idx, 't1'].values[0]
            t2 = glacier_cal_data.loc[cal_idx, 't2'].values[0]
            t1_idx = int(glacier_cal_data.loc[cal_idx,'t1_idx'])
            t2_idx = int(glacier_cal_data.loc[cal_idx,'t2_idx'])
            observed_massbal = (glacier_cal_data.loc[cal_idx,'mb_mwe'] / (t2 - t1)).values[0]
            observed_error = (glacier_cal_data.loc[cal_idx,'mb_mwe_err'] / (t2 - t1)).values[0]
            mb_obs_max = observed_massbal + 3 * observed_error
            mb_obs_min = observed_massbal - 3 * observed_error
            
            # MCMC Analysis
            ds = xr.open_dataset(netcdf_fp + glacier_str + '.nc')
            df = pd.DataFrame(ds['mp_value'].values[:,:,0], columns=ds.mp.values)  
            print('MB (mod - obs):', np.round(df.massbal.mean() - observed_massbal,3))
            
            # Set model parameters
            modelparameters = [input.lrgcm, input.lrglac, input.precfactor, input.precgrad, input.ddfsnow, input.ddfice,
                               input.tempsnow, input.tempchange]
            
            # Regional priors
            precfactor_gamma_alpha = input.precfactor_gamma_region_dict[glacier_rgi_table.loc['region']][0]
            precfactor_gamma_beta = input.precfactor_gamma_region_dict[glacier_rgi_table.loc['region']][1]                      
            tempchange_mu = input.tempchange_norm_region_dict[glacier_rgi_table.loc['region']][0]
            tempchange_sigma = input.tempchange_norm_region_dict[glacier_rgi_table.loc['region']][1]
            
            ddfsnow_mu = input.ddfsnow_mu * 1000
            ddfsnow_sigma = input.ddfsnow_sigma * 1000
            ddfsnow_boundlow = input.ddfsnow_boundlow * 1000
            ddfsnow_boundhigh = input.ddfsnow_boundhigh * 1000
           
            param_idx_dict = {'massbal':[0,n],
                              'precfactor':[1,n],
                              'tempchange':[2,n],
                              'ddfsnow':[3,n]}
        
            for nvar, vn in enumerate(variables):        
                nrow = param_idx_dict[vn][0]
                
                # ====== PRIOR DISTRIBUTIONS ======
                if vn == 'massbal':
                    z_score = np.linspace(norm.ppf(0.01), norm.ppf(0.99), 100)
                    x_values = observed_massbal + observed_error * z_score
                    y_values = norm.pdf(x_values, loc=observed_massbal, scale=observed_error)
                elif vn == 'precfactor': 
                    if input.precfactor_disttype == 'gamma':
                        x_values = np.linspace(
                                stats.gamma.ppf(0,precfactor_gamma_alpha, scale=1/precfactor_gamma_beta), 
                                stats.gamma.ppf(0.999,precfactor_gamma_alpha, scale=1/precfactor_gamma_beta), 
                                100)                                
                        y_values = stats.gamma.pdf(x_values, a=precfactor_gamma_alpha, scale=1/precfactor_gamma_beta)    
                elif vn == 'tempchange':
                    if input.tempchange_disttype == 'normal':
                        z_score = np.linspace(norm.ppf(0.01), norm.ppf(0.99), 100)
                        x_values = tempchange_mu + tempchange_sigma * z_score
                        y_values = norm.pdf(x_values, loc=tempchange_mu, scale=tempchange_sigma)
                elif vn == 'ddfsnow':            
                    if input.ddfsnow_disttype == 'truncnormal':
                        ddfsnow_a = (ddfsnow_boundlow - ddfsnow_mu) / ddfsnow_sigma
                        ddfsnow_b = (ddfsnow_boundhigh - ddfsnow_mu) / ddfsnow_sigma
                        z_score = np.linspace(truncnorm.ppf(0.001, ddfsnow_a, ddfsnow_b),
                                              truncnorm.ppf(0.999, ddfsnow_a, ddfsnow_b), 100)
                        x_values = ddfsnow_mu + ddfsnow_sigma * z_score
                        y_values = truncnorm.pdf(x_values, ddfsnow_a, ddfsnow_b, loc=ddfsnow_mu, scale=ddfsnow_sigma)
                # PLOT PRIOR
                nrow = param_idx_dict[vn][0]
                ncol = param_idx_dict[vn][1]
                ax[nrow,ncol].plot(x_values, y_values, color='k')
                
                # Labels
                ax[nrow,ncol].set_xlabel(vn_label_dict[vn], size=10, labelpad=1)
                if nvar == 0:
                    ax[nrow,ncol].set_title('Glacier RGI60-' + glacier_str, fontsize=12)
                
                # PLOT POSTERIOR               
                # Ensemble/Posterior distribution                
                for n_chain in range(len(ds.chain.values)):
                    for count_iter, n_iters in enumerate(iters):
                        chain = ds['mp_value'].sel(chain=n_chain, mp=vn).values[burn:n_iters]
                        chain = ds['mp_value'].sel(chain=n_chain, mp=vn).values[burn:n_iters]
                        
                        if vn == 'ddfsnow':
                            chain = chain * 10**3
                    
                        # gaussian distribution
                        kde = gaussian_kde(chain)
                        x_values_kde = x_values.copy()
                        y_values_kde = kde(x_values_kde)
                        
                        # Plot fitted distribution
                        ax[nrow,ncol].plot(x_values_kde, y_values_kde, color=iter_colors[count_iter], 
                                           linestyle=linestyles[n_chain])
            # Close dataset
            ds.close()
        
        # Legend for first subplot
    #    ax[0,1].legend(title='Steps', loc='upper right', handlelength=1, handletextpad=0.05, borderpad=0.2)
        leg_lines = []
        leg_labels = []
        chain_labels = ['Prior', '1000', '10000']
        chain_colors = ['black', '#387ea0', '#fcb200']
    #    chain_labels = ['Prior', '10000']
    #    chain_colors = ['black', '#387ea0']
        for n_chain in range(len(chain_labels)):
    #        line = Line2D([0,1],[0,1], color='white')
    #        leg_lines.append(line)
    #        leg_labels.append('')
            line = Line2D([0,1],[0,1], color=chain_colors[n_chain])
            leg_lines.append(line)
            leg_labels.append(chain_labels[n_chain])
        fig.legend(leg_lines, leg_labels, loc='upper right', 
                   bbox_to_anchor=(0.87,0.885), 
    #               bbox_to_anchor=(0.87,0.815),
                   handlelength=1.5, handletextpad=0.25, borderpad=0.2, frameon=True)
        
    #    # Legend (Note: hard code the spacing between the two legends) 
    #    leg_lines = []
    #    leg_labels = []
    #    chain_labels = ['Center', 'Lower Bound', 'Upper Bound']
    #    for n_chain in range(len(ds.chain.values)):
    ##        line = Line2D([0,1],[0,1], color='white')
    ##        leg_lines.append(line)
    ##        leg_labels.append('')
    #        line = Line2D([0,1],[0,1], color='gray', linestyle=linestyles[n_chain])
    #        leg_lines.append(line)
    #        leg_labels.append(chain_labels[n_chain])
    #    fig.legend(leg_lines, leg_labels, title='Overdispersed Starting Point', loc='lower center', 
    #               bbox_to_anchor=(0.47,-0.005), handlelength=1.5, handletextpad=0.25, borderpad=0.2, frameon=True, 
    #               ncol=3, columnspacing=0.75)
            
        # OLD LEGEND WITH ALL OF THEM ON THE BOTTOM
    #    # Legend (Note: hard code the spacing between the two legends)
    #    leg_lines = []
    #    leg_labels = []
    #    for count_iter, n_iters in enumerate(iters):
    ##        line = Line2D([0,1],[0,1], color='white')
    ##        leg_lines.append(line)
    ##        leg_labels.append('')
    #        line = Line2D([0,1],[0,1], color=colors[count_iter])
    #        leg_lines.append(line)
    #        leg_labels.append(str(int(n_iters)))    
    #    chain_labels = ['Center', 'Lower Bound', 'Upper Bound']
    #    for n_chain in range(len(ds.chain.values)):
    ##        line = Line2D([0,1],[0,1], color='white')
    ##        leg_lines.append(line)
    ##        leg_labels.append('')
    #        line = Line2D([0,1],[0,1], color='gray', linestyle=linestyles[n_chain])
    #        leg_lines.append(line)
    #        leg_labels.append(chain_labels[n_chain])
    #    
    #    fig.legend(leg_lines, leg_labels, loc='lower center', bbox_to_anchor=(0.47,0),
    #               handlelength=1.5, handletextpad=0.25, borderpad=0.2, frameon=True, ncol=5, columnspacing=0.75)
    
        fig.text(0.03, 0.5, 'Probability Density', va='center', rotation='vertical', size=12)
        fig.text(0.14, 0.855, 'A', size=12)
        fig.text(0.56, 0.855, 'B', size=12)
        fig.text(0.14, 0.65, 'C', size=12)
        fig.text(0.56, 0.65, 'D', size=12)
        fig.text(0.14, 0.445, 'E', size=12)
        fig.text(0.56, 0.445, 'F', size=12)
        fig.text(0.14, 0.24, 'G', size=12)
        fig.text(0.56, 0.24, 'H', size=12)
            
        # Save figure
        fig.savefig(fig_fp + figure_fn, bbox_inches='tight', pad_inches=0.02, dpi=300)
        
        
        #%%
    if option_papermcmc_solutionspace == 1:
        
        glac_no = ['13.05086']
        netcdf_fp = input.output_fp_cal
    #    netcdf_fp = mcmc_output_netcdf_fp_3chain
        
    #    filelist = []
    #    for region in regions:
    #        filelist.extend(glob.glob(netcdf_fp + str(region) + '*.nc'))
    #    
    #    glac_no = []
    #    reg_no = []
    #    for netcdf in filelist:
    #        glac_str = netcdf.split('/')[-1].split('.nc')[0]
    #        glac_no.append(glac_str)
    #        reg_no.append(glac_str.split('.')[0])
    #    glac_no = sorted(glac_no)
        
        (main_glac_rgi, main_glac_hyps, main_glac_icethickness, main_glac_width, 
         gcm_temp, gcm_prec, gcm_elev, gcm_lr, cal_data, dates_table) = load_glacierdata_byglacno(glac_no)
        
        # Elevation bins
        elev_bins = main_glac_hyps.columns.values.astype(int) 
        
        for n, glac_str_wRGI in enumerate(main_glac_rgi['RGIId'].values):
            # Glacier string
            glacier_str = glac_str_wRGI.split('-')[1]
            # Glacier number
            glacno = int(glacier_str.split('.')[1])
            # RGI information
            glacier_rgi_table = main_glac_rgi.loc[main_glac_rgi.index.values[n], :]
            # Calibration data
            cal_idx = np.where(cal_data['glacno'] == glacno)[0]
            glacier_cal_data = (cal_data.iloc[cal_idx,:]).copy()
            # Select observed mass balance, error, and time data
            t1 = glacier_cal_data.loc[cal_idx, 't1'].values[0]
            t2 = glacier_cal_data.loc[cal_idx, 't2'].values[0]
            t1_idx = int(glacier_cal_data.loc[cal_idx,'t1_idx'])
            t2_idx = int(glacier_cal_data.loc[cal_idx,'t2_idx'])
            observed_massbal = (glacier_cal_data.loc[cal_idx,'mb_mwe'] / (t2 - t1)).values[0]
            observed_error = (glacier_cal_data.loc[cal_idx,'mb_mwe_err'] / (t2 - t1)).values[0]
            mb_obs_max = observed_massbal + 3 * observed_error
            mb_obs_min = observed_massbal - 3 * observed_error
            
            # MCMC Analysis
            ds = xr.open_dataset(netcdf_fp + glacier_str + '.nc')
            df = pd.DataFrame(ds['mp_value'].values[:,:,0], columns=ds.mp.values)  
            print('MB (obs - mean_model):', np.round(observed_massbal - df.massbal.mean(),3))
            
            # Priors
            try:
                priors = pd.Series(ds.priors, index=ds['dim_0'])
            except:
                priors = pd.Series(ds.priors, index=ds.prior_cns)
            
            precfactor_boundlow = priors['pf_bndlow']
            precfactor_boundhigh = priors['pf_bndhigh']
            precfactor_boundmu = priors['pf_mu']
            tempchange_boundlow = priors['tc_bndlow']
            tempchange_boundhigh = priors['tc_bndhigh']
            tempchange_mu = priors['tc_mu']
            tempchange_sigma = priors['tc_std']
            ddfsnow_boundhigh = priors['ddfsnow_bndhigh']
            ddfsnow_boundlow = priors['ddfsnow_bndlow']
            ddfsnow_mu = priors['ddfsnow_mu']
            ddfsnow_sigma = priors['ddfsnow_std']
            mb_max_loss = priors['mb_max_loss']
            mb_max_acc = priors['mb_max_acc']
            try:
                tempchange_max_loss = priors['tc_max_loss']
            except: # typo in initial code - issue fixed 03/08/2019
                tempchange_max_loss = priors['tc_maxloss']
            tempchange_max_acc = priors['tc_max_acc']
            precfactor_opt_init = priors['pf_opt_init']
            tempchange_opt_init = priors['tc_opt_init']
            
            print('\nParameters:\nPF_low:', np.round(precfactor_boundlow,2), 'PF_high:', 
                  np.round(precfactor_boundhigh,2), '\nTC_low:', np.round(tempchange_boundlow,2), 
                  'TC_high:', np.round(tempchange_boundhigh,2),
                  '\nTC_mu:', np.round(tempchange_mu,2), 'TC_sigma:', np.round(tempchange_sigma,2))
            
            
            # Select subsets of data
            glacier_gcm_elev = gcm_elev[n]
            glacier_gcm_temp = gcm_temp[n,:]
            glacier_gcm_lrgcm = gcm_lr[n,:]
            glacier_gcm_lrglac = glacier_gcm_lrgcm.copy()
            glacier_gcm_prec = gcm_prec[n,:]
            glacier_area_t0 = main_glac_hyps.iloc[n,:].values.astype(float)
            icethickness_t0 = main_glac_icethickness.iloc[n,:].values.astype(float)
            width_t0 = main_glac_width.iloc[n,:].values.astype(float)
            glac_idx_t0 = glacier_area_t0.nonzero()[0]
            # Set model parameters
            modelparameters = [input.lrgcm, input.lrglac, input.precfactor, input.precgrad, input.ddfsnow, input.ddfice,
                               input.tempsnow, input.tempchange]
            
            tc_iter_step = 0.1
            tc_iter_high = np.max([tempchange_max_loss, tempchange_boundhigh])
            tempchange_iters = np.arange(int(tempchange_max_acc), np.ceil(tc_iter_high)+tc_iter_step, tc_iter_step).tolist()
            ddfsnow_iters = [0.0026, 0.0041, 0.0056]
            precfactor_iters = [int(precfactor_boundlow*10)/10, int((precfactor_boundlow + precfactor_boundhigh)/2*10)/10, 
                                int(precfactor_boundhigh*10)/10]
            if 1 not in precfactor_iters:
                precfactor_iters.append(int(1))
                precfactor_iters = sorted(precfactor_iters)
                
            fig_fp = netcdf_fp + 'figures/'
            if os.path.exists(fig_fp) == False:
                os.makedirs(fig_fp)
            
            plot_mb_vs_parameters(tempchange_iters, precfactor_iters, ddfsnow_iters, modelparameters, glacier_rgi_table, 
                                  glacier_area_t0, icethickness_t0, width_t0, elev_bins, glacier_gcm_temp, glacier_gcm_prec, 
                                  glacier_gcm_elev, glacier_gcm_lrgcm, glacier_gcm_lrglac, dates_table, observed_massbal, 
                                  observed_error, tempchange_boundhigh, tempchange_boundlow, tempchange_opt_init, 
                                  mb_max_acc, mb_max_loss, tempchange_max_acc, tempchange_max_loss, option_areaconstant=0,
                                  option_plotsteps=1, fig_fp=fig_fp)
    
    
    #%%
    if option_papermcmc_modelparameter_map_and_postvprior == 1:    
        netcdf_fp = mcmc_output_netcdf_fp_all
        figure_fp = netcdf_fp + 'figures/'
        csv_fp = netcdf_fp + 'csv/'
        grouping = 'degree'
        degree_size = 0.5
        
        vns = ['ddfsnow', 'tempchange', 'precfactor', 'dif_masschange']
        modelparams_fn = 'main_glac_rgi_20190806_wcal_wposteriors_all_1000burn.csv'
        
        east = 104
        west = 67
        south = 26
        north = 46
        xtick = 5
        ytick = 5
        xlabel = 'Longitude ($\mathregular{^{\circ}}$)'
        ylabel = 'Latitude ($\mathregular{^{\circ}}$)'
        
        labelsize = 12
        
        colorbar_dict = {'precfactor':[0,3],
                         'tempchange':[-1.5,2.5],
                         'ddfsnow':[2.6,5.6],
                         'dif_masschange':[-0.3,0.3],
                         'massbal':[-1.5,0.5]}
        
        if os.path.exists(figure_fp) == False:
            os.makedirs(figure_fp)
    
        # Load mean of all model parameters
        if os.path.isfile(csv_fp + modelparams_fn) == False:
            
            # Load all glaciers
            filelist = []
            for region in regions:
                filelist.extend(glob.glob(netcdf_fp + str(region) + '*.nc'))
            
            glac_no = []
            reg_no = []
            for netcdf in filelist:
                glac_str = netcdf.split('/')[-1].split('.nc')[0]
                glac_no.append(glac_str)
                reg_no.append(glac_str.split('.')[0])
            glac_no = sorted(glac_no)
            
            main_glac_rgi, cal_data = load_glacierdata_byglacno(glac_no, option_loadhyps_climate=0)
            
            posterior_cns = ['glacno', 'mb_mean', 'mb_std', 'pf_mean', 'pf_std', 'tc_mean', 'tc_std', 'ddfsnow_mean', 
                             'ddfsnow_std']
            
            posterior_all = pd.DataFrame(np.zeros((main_glac_rgi.shape[0], len(posterior_cns))), columns=posterior_cns)
            
            for n, glac_str_wRGI in enumerate(main_glac_rgi['RGIId'].values):
                if n%500 == 0:
                    print(n, glac_str_wRGI)
                # Glacier string
                glacier_str = glac_str_wRGI.split('-')[1]
                # Glacier number
                glacno = int(glacier_str.split('.')[1])
                # RGI information
                glac_idx = main_glac_rgi.index.values[n]
                glacier_rgi_table = main_glac_rgi.loc[glac_idx, :]
                # Calibration data
                glacier_cal_data = (cal_data.loc[glac_idx,:]).copy()        
                # Select observed mass balance, error, and time data
                t1 = glacier_cal_data['t1']
                t2 = glacier_cal_data['t2']
                t1_idx = int(glacier_cal_data['t1_idx'])
                t2_idx = int(glacier_cal_data['t2_idx'])
                observed_massbal = glacier_cal_data['mb_mwe'] / (t2 - t1)
                observed_error = glacier_cal_data['mb_mwe_err'] / (t2 - t1)
                mb_obs_max = observed_massbal + 3 * observed_error
                mb_obs_min = observed_massbal - 3 * observed_error
                
                # MCMC Analysis
                ds = xr.open_dataset(netcdf_fp + glacier_str + '.nc')
                df = pd.DataFrame(ds['mp_value'].values[:,:,0], columns=ds.mp.values)
                
                # Posteriors            
                posterior_row = [glacier_str, 
                                 df.massbal.mean(), df.massbal.std(), 
                                 df.precfactor.mean(), df.precfactor.std(), 
                                 df.tempchange.mean(), df.tempchange.std(),
                                 df.ddfsnow.mean(), df.ddfsnow.std()]
                posterior_all.loc[n,:] = posterior_row
                
                ds.close()
            modelparams_all = main_glac_rgi[['RGIId', 'CenLon', 'CenLat', 'O1Region', 'Area', 'RefDate', 'glacno', 
                                             'RGIId_float']]
            modelparams_all = pd.concat([modelparams_all, cal_data[['mb_mwe', 'mb_mwe_err', 't1', 't2', 'area_km2']]], 
                                        axis=1)
            modelparams_all['mb_mwea'] = cal_data['mb_mwe'] / (cal_data['t2'] - cal_data['t1'])
            modelparams_all['mb_mwea_err'] = cal_data['mb_mwe_err'] / (cal_data['t2'] - cal_data['t1'])
            modelparams_all = pd.concat([modelparams_all, posterior_all], axis=1)
            if os.path.exists(csv_fp) == False:
                os.makedirs(csv_fp)
            modelparams_all.to_csv(csv_fp + modelparams_fn)
            
        else:
            modelparams_all = pd.read_csv(csv_fp + modelparams_fn)
        
        # Add priors
        if 'region' not in modelparams_all.columns.tolist():
            # Add region and priors
            modelparams_all['Region'] = modelparams_all.RGIId.map(input.reg_dict)
            # Priors
            # precipitation factor
            precfactor_alpha_dict = {region: input.precfactor_gamma_region_dict[region][0] 
                                     for region in list(input.precfactor_gamma_region_dict.keys())}
            precfactor_beta_dict = {region: input.precfactor_gamma_region_dict[region][1] 
                                     for region in list(input.precfactor_gamma_region_dict.keys())}
            modelparams_all['prior_pf_alpha'] = modelparams_all.Region.map(precfactor_alpha_dict) 
            modelparams_all['prior_pf_beta'] = modelparams_all.Region.map(precfactor_beta_dict)
            modelparams_all['prior_pf_mu'] = modelparams_all['prior_pf_alpha'] / modelparams_all['prior_pf_beta'] 
            modelparams_all['prior_pf_std'] = (modelparams_all['prior_pf_alpha'] / modelparams_all['prior_pf_beta']**2)**0.5
            # temperature change
            tempchange_mu_dict = {region: input.tempchange_norm_region_dict[region][0] 
                                  for region in list(input.tempchange_norm_region_dict.keys())}
            tempchange_std_dict = {region: input.tempchange_norm_region_dict[region][1] 
                                   for region in list(input.tempchange_norm_region_dict.keys())}
            modelparams_all['prior_tc_mu'] = modelparams_all.Region.map(tempchange_mu_dict) 
            modelparams_all['prior_tc_std'] = modelparams_all.Region.map(tempchange_std_dict)
            # degree-day factor of snow
            modelparams_all['prior_ddfsnow_mu'] = input.ddfsnow_mu * 1000
            modelparams_all['prior_ddfsnow_std'] = input.ddfsnow_sigma * 1000
            
            if os.path.exists(csv_fp) == False:
                os.makedirs(csv_fp)
            modelparams_all.to_csv(csv_fp + modelparams_fn, index=False)
        
        # Add convergence statistics
    #    en_fn_pkl = csv_fp + '../effective_n_list_10000iters.pkl'
    #    mc_fn_pkl = csv_fp + '../mc_error_list_10000iters.pkl'
    #    glacno_fn_pkl = csv_fp + '../glacno_list.pkl'
    #    if 'eff_n_mb' not in modelparams_all.columns.tolist() and os.path.isfile(csv_fp + en_fn_pkl):
    #        # Add convergence statistics
    #        with open(en_fn_pkl, 'rb') as f:
    #            en_list = pickle.load(f)
    #        with open(mc_fn_pkl, 'rb') as f:
    #            mc_list = pickle.load(f)
    #        with open(glacno_fn_pkl, 'rb') as f:
    #            glac_no_pkl = pickle.load(f)
    #    
    #        mc_list_mb = [mc_list[x]['massbal'] for x in glac_no_pkl]
    #        mc_list_pf = [mc_list[x]['precfactor'] for x in glac_no_pkl]
    #        mc_list_tc = [mc_list[x]['tempchange'] for x in glac_no_pkl]
    #        mc_list_ddf = [mc_list[x]['ddfsnow'] for x in glac_no_pkl]
    #        en_list_mb = [en_list[x]['massbal'] for x in glac_no_pkl]
    #        en_list_pf = [en_list[x]['precfactor'] for x in glac_no_pkl]
    #        en_list_tc = [en_list[x]['tempchange'] for x in glac_no_pkl]
    #        en_list_ddf = [en_list[x]['ddfsnow'] for x in glac_no_pkl]
    #        
    #        modelparams_all['glac_no'] = glac_no
    #        modelparams_all['mc_mb'] = mc_list_mb
    #        modelparams_all['mc_pf'] = mc_list_pf
    #        modelparams_all['mc_tc'] = mc_list_tc
    #        modelparams_all['mc_ddf'] = mc_list_ddf
    #        modelparams_all['eff_n_mb'] = en_list_mb
    #        modelparams_all['eff_n_pf'] = en_list_pf
    #        modelparams_all['eff_n_tc'] = en_list_tc
    #        modelparams_all['eff_n_ddf'] = en_list_ddf
    #        modelparams_all['mc_ddf'] = modelparams_all['mc_ddf'] * 10**3
    #
    #        if os.path.exists(csv_fp) == False:
    #            os.makedirs(csv_fp)
    #        modelparams_all.to_csv(csv_fp + modelparams_fn, index=False)
    
        
        #%%    
        modelparams_all['dif_cal_era_mean'] = modelparams_all['mb_mean'] - modelparams_all['mb_mwea']
        modelparams_all['zscore'] = ((modelparams_all['mb_mean'] - modelparams_all['mb_mwea']) / 
                                      modelparams_all['mb_mwea_err'])
    
        # remove nan values
        modelparams_all = (
                modelparams_all.drop(np.where(np.isnan(modelparams_all['mb_mean'].values) == True)[0].tolist(), 
                                       axis=0))
        modelparams_all.reset_index(drop=True, inplace=True)
        
        # Mass change
        #  Area [km2] * mb [mwe] * (1 km / 1000 m) * density_water [kg/m3] * (1 Gt/km3  /  1000 kg/m3)
        modelparams_all['mb_cal_Gta'] = modelparams_all['mb_mwea'] * modelparams_all['Area'] / 1000
        modelparams_all['mb_cal_Gta_var'] = (modelparams_all['mb_mwea_err'] * modelparams_all['Area'] / 1000)**2
        modelparams_all['mb_era_Gta'] = modelparams_all['mb_mean'] * modelparams_all['Area'] / 1000
        modelparams_all['mb_era_Gta_var'] = (modelparams_all['mb_std'] * modelparams_all['Area'] / 1000)**2
        print('All MB cal (mean +/- 1 std) [gt/yr]:', np.round(modelparams_all['mb_cal_Gta'].sum(),3), 
              '+/-', np.round(modelparams_all['mb_cal_Gta_var'].sum()**0.5,3),
              '\nAll MB ERA (mean +/- 1 std) [gt/yr]:', np.round(modelparams_all['mb_era_Gta'].sum(),3), 
              '+/-', np.round(modelparams_all['mb_era_Gta_var'].sum()**0.5,3))
        
        #%%
        # Add watersheds, regions, degrees, mascons, and all groups to main_glac_rgi_all
        # Watersheds
        modelparams_all['watershed'] = modelparams_all.RGIId.map(watershed_dict)
        # Regions
        modelparams_all['kaab'] = modelparams_all.RGIId.map(kaab_dict)
        # Degrees
        modelparams_all['CenLon_round'] = np.floor(modelparams_all.CenLon.values/degree_size) * degree_size
        modelparams_all['CenLat_round'] = np.floor(modelparams_all.CenLat.values/degree_size) * degree_size
        deg_groups = modelparams_all.groupby(['CenLon_round', 'CenLat_round']).size().index.values.tolist()
        deg_dict = dict(zip(deg_groups, np.arange(0,len(deg_groups))))
        modelparams_all.reset_index(drop=True, inplace=True)
        cenlon_cenlat = [(modelparams_all.loc[x,'CenLon_round'], modelparams_all.loc[x,'CenLat_round']) 
                         for x in range(len(modelparams_all))]
        modelparams_all['CenLon_CenLat'] = cenlon_cenlat
        modelparams_all['deg_id'] = modelparams_all.CenLon_CenLat.map(deg_dict)
        # All
        modelparams_all['all_group'] = 'all'
    
        # Rename columns    
        modelparams_all = modelparams_all.rename(columns={'pf_mean':'precfactor', 'tc_mean':'tempchange', 
                                                          'ddfsnow_mean':'ddfsnow'})
        # Convert DDFsnow to mm w.e.d-1C-1
        modelparams_all[['ddfsnow', 'ddfsnow_std']] = modelparams_all[['ddfsnow', 'ddfsnow_std']] * 10**3    
        
        #%%
        # Scatterplot: Model vs. Observed Mass balance colored by Area
        fig, ax = plt.subplots()
        cmap = 'RdYlBu_r'
    #    cmap = plt.cm.get_cmap(cmap, 5)
    #    norm = plt.Normalize(0.1, 10)    
        norm = colors.LogNorm(vmin=0.1, vmax=10)    
        a = ax.scatter(modelparams_all['mb_mwea'], modelparams_all['mb_mean'], c=modelparams_all['Area'],
                       cmap=cmap, norm=norm, s=20, linewidth=0.5)
        a.set_facecolor('none')
        ax.plot([-2.5,2],[-2.5,2], color='k', linewidth=0.5)
        ax.set_xlim([-2.5,1.75])
        ax.set_ylim([-2.5,1.75])
        ax.set_ylabel('$\mathregular{B_{obs}}$ $\mathregular{(m w.e. a^{-1})}$', size=12)
        ax.set_xlabel('$\mathregular{B_{mod}}$ $\mathregular{(m w.e. a^{-1})}$', size=12)
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm._A = []
        cbar = plt.colorbar(sm, ax=ax, fraction=0.04, pad=0.01)
    #    cbar.set_ticks(list(np.arange(colorbar_dict['dif_masschange'][0], colorbar_dict['dif_masschange'][1] + 0.01, 0.04)))
        fig.text(1.01, 0.5, 'Area ($\mathregular{km^{2}}$)', va='center', rotation='vertical', size=12)
        # Save figure
        fig.set_size_inches(6,4)
        fig_fn = 'MB_vs_model_warea.png'
        fig.savefig(figure_fp + fig_fn, bbox_inches='tight', dpi=300)
        
        #%%
        # Map & Scatterplot of mass balance difference
        plot_spatialmap_mbdif(vns, grouping, modelparams_all, xlabel, ylabel, figure_fp=figure_fp, option_group_regions=1)
        
        #%%
        # Spatial distribution of parameters    
    #    vns = ['dif_masschange', 'precfactor', 'tempchange', 'ddfsnow']
        vns = ['precfactor', 'tempchange', 'ddfsnow']
    #    vns = ['ddfsnow']
        
        midpt_dict = {'dif_masschange':0,
                      'precfactor':1,
                      'tempchange':0,
                      'ddfsnow':4.1}
        cmap_dict = {'dif_masschange':'RdYlBu_r',
                     'precfactor':'RdYlBu',
                     'tempchange':'RdYlBu_r',
                     'ddfsnow':'RdYlBu_r'}
        title_adj = {'dif_masschange':15,
                     'precfactor':6,
                     'tempchange':6,
                     'ddfsnow':6}
        
        plot_spatialmap_parameters(vns, grouping, modelparams_all, xlabel, ylabel, midpt_dict, cmap_dict, title_adj, 
                                   figure_fp=figure_fp, option_group_regions=1)
        
        #%%
        # ===== PRIOR VS POSTERIOR FOR EACH GLACIER =====    
        # Bin spacing (note: offset them, so centered on 0)
        bdict = {}
        bdict['massbal-Mean'] = np.arange(-0.6, 0.675, 0.05) - 0.025
        bdict['precfactor-Mean'] = np.arange(-1.6, 1.9, 0.2) - 0.1
        bdict['tempchange-Mean'] = np.arange(-2, 2.3, 0.2) - 0.1
        bdict['ddfsnow-Mean'] = np.arange(-2, 2.3, 0.2) - 0.1
        bdict['massbal-Standard Deviation'] = np.arange(-0.68, 0.2, 0.04) - 0.02
        bdict['precfactor-Standard Deviation'] = np.arange(-1, 0.5, 0.1) - 0.05
        bdict['tempchange-Standard Deviation'] = np.arange(-1, 0.2, 0.05) - 0.025
        bdict['ddfsnow-Standard Deviation'] = np.arange(-0.6, 0.2, 0.05) - 0.025
        
        tdict = {}
        glac_ylim = 40
        tdict['Mean'] = np.arange(0, glac_ylim + 1, 10)
        tdict['Standard Deviation'] = np.arange(0, glac_ylim + 1, 10)
        
        estimators = ['Mean', 'Standard Deviation']
        
        fig, ax = plt.subplots(len(variables), len(estimators), squeeze=False, sharex=False, sharey=False, 
                               gridspec_kw = {'wspace':0.1, 'hspace':0.4})    
        
        for nvar, vn in enumerate(variables):
    #    for nvar, vn in enumerate(['tempchange']):
            print(nvar, vn)
    
            if vn == 'massbal':
                mean_prior = modelparams_all['mb_mwea'].values
                mean_post = modelparams_all['mb_mean'].values
                std_prior = modelparams_all['mb_mwea_err'].values
                std_post = modelparams_all['mb_std'].values
            elif vn == 'precfactor':
                mean_prior = modelparams_all['prior_pf_mu'].values
                mean_post = modelparams_all['precfactor'].values
                std_prior = modelparams_all['prior_pf_std'].values
                std_post = modelparams_all['pf_std'].values
            elif vn == 'tempchange':
                mean_prior = modelparams_all['prior_tc_mu'].values
                mean_post = modelparams_all['tempchange'].values
                std_prior = modelparams_all['prior_tc_std'].values
                std_post = modelparams_all['tc_std'].values
            elif vn == 'ddfsnow':
                mean_prior = modelparams_all['prior_ddfsnow_mu'].values
                mean_post = modelparams_all['ddfsnow'].values
                std_prior = modelparams_all['prior_ddfsnow_std'].values
                std_post = modelparams_all['ddfsnow_std'].values   
                vn_label_units_dict['ddfsnow'] = '[10$^{3}$ mwe d$^{-1}$ $^\circ$C$^{-1}$]'
            
            dif_mean = mean_post - mean_prior
            dif_std = std_post - std_prior
            
            print('  dif_mean (min/max):', np.round(dif_mean.min(),2), np.round(dif_mean.max(),2))
            print('  dif_std (min/max):', np.round(dif_std.min(),2), np.round(dif_std.max(),2))
    
            for nest, estimator in enumerate(estimators):
                if estimator == 'Mean':
                    dif = dif_mean
                    bcolor = 'lightgrey'
                elif estimator == 'Standard Deviation':
                    dif = dif_std
                    bcolor = 'lightgrey'
            
                # ===== Plot =====
                hist, bins = np.histogram(dif, bins=bdict[vn + '-' + estimator])
                hist = hist * 100.0 / hist.sum()
                bins_centered = bins[1:] + (bins[0] - bins[1]) / 2
                # plot histogram
                ax[nvar,nest].bar(x=bins_centered, height=hist, width=(bins[1]-bins[0]), align='center',
                                  edgecolor='black', color=bcolor, alpha=0.5)
                ax[nvar,nest].set_yticks(tdict[estimator])
                ax[nvar,nest].set_ylim(0,glac_ylim)                
                
                # axis labels
                ax[nvar,nest].set_xlabel(vn_label_dict[vn], fontsize=10, labelpad=1)
                if nvar == 0:
                    ax[nvar,nest].set_title('$\Delta$ ' + estimator, fontsize=12)
                if nest == 1:
                    ax[nvar,nest].set_yticks([])
                    
                print('  ', estimator, '% near 0:', np.round(hist[np.where(bins > 0)[0][0] - 1]))
        
        fig.text(0.04, 0.5, 'Count (%)', va='center', rotation='vertical', size=12)
        fig.text(0.135, 0.86, 'A', size=12)
        fig.text(0.540, 0.86, 'B', size=12)
        fig.text(0.135, 0.655, 'C', size=12)
        fig.text(0.540, 0.655, 'D', size=12)
        fig.text(0.135, 0.45, 'E', size=12)
        fig.text(0.540, 0.45, 'F', size=12)
        fig.text(0.135, 0.25, 'G', size=12)
        fig.text(0.540, 0.25, 'H', size=12)
                    
        # Save figure
        fig.set_size_inches(6.5,8)
        figure_fn = 'prior_vs_posterior_hist.png'
        fig.savefig(figure_fp + figure_fn, bbox_inches='tight', dpi=300)
        
    #    #%%
    #    # Scatterplots: MB_sigma vs. Effective N and MC error
    #    endings = ['mb', 'pf', 'tc', 'ddf']
    #    endings_dict = {'mb':'Mass Balance [mwea-1]', 'pf':'Precipitation Factor [-]', 'tc':'Temperature Bias [degC]', 
    #                    'ddf': 'Degree-day Factor of Snow [mmwe d-1 degC-1]'}
    #    prefixes = ['eff_n_', 'mc_']
    #    
    #    for prefix in prefixes:
    #        for ending in endings:
    ##        for ending in ['mb']:
    #            fig, ax = plt.subplots()
    #            ax.scatter(prior_compare['mb_std'], prior_compare[prefix + ending], s=0.001)
    #            ax.set_xlabel('Observed MB Sigma')
    #            if 'eff_n' in prefix:
    #                ax.set_ylabel('Effective N')
    #            elif 'mc' in prefix:
    #                ax.set_ylabel('Monte Carlo Error')
    #            ax.set_title(endings_dict[ending])
    #            # Save figure
    #            fig.set_size_inches(4,4)
    #            fig_fn = 'mb_sigma_vs_' + prefix + ending + '.png'
    #            fig.savefig(fig_fp + fig_fn, bbox_inches='tight', dpi=300)
        
    #%%   
    if option_papermcmc_hh2015_map == 1:    
        netcdf_fp = hh2015_output_netcdf_fp_all
        figure_fp = netcdf_fp + '../'
        grouping = 'degree'
        degree_size = 0.5
        
        vns = ['ddfsnow', 'tempchange', 'precfactor', 'dif_masschange']
        modelparams_fn = 'hh2015_HMA_parameters.csv'
        
        option_add_calopt2 = 0
        if option_add_calopt2 == 1:
            cal_opt2_fn = '../main_glac_rgi_20190308_wcal_wposteriors.csv'
            prior_compare_fn = '../prior_compare_all.csv'
        
        
        east = 104
        west = 67
        south = 26
        north = 46
        xtick = 5
        ytick = 5
        xlabel = 'Longitude ($\mathregular{^{\circ}}$)'
        ylabel = 'Latitude ($\mathregular{^{\circ}}$)'
        
        labelsize = 12
        
    #    colorbar_dict = {'precfactor':[0,5],
    #                     'tempchange':[-5,5],
    #                     'ddfsnow':[2.6,5.6],
    #                     'dif_masschange':[-0.1,0.1]}
        colorbar_dict = {'precfactor':[0,3],
                         'tempchange':[-1.5,2.5],
                         'ddfsnow':[2.6,5.6],
                         'dif_masschange':[-0.3,0.3],
                         'massbal':[-1.5,0.5]}
    
        # Load mean of all model parameters
        if os.path.isfile(figure_fp + modelparams_fn) == False:
            
            # Load all glaciers
            filelist = []
            for region in regions:
                filelist.extend(glob.glob(netcdf_fp + str(region) + '*.nc'))
            
            glac_no = []
            reg_no = []
            for netcdf in filelist:
                glac_str = netcdf.split('/')[-1].split('.nc')[0]
                glac_no.append(glac_str)
                reg_no.append(glac_str.split('.')[0])
            glac_no = sorted(glac_no)
            
            main_glac_rgi, cal_data = load_glacierdata_byglacno(glac_no, option_loadhyps_climate=0)
            
            posterior_cns = ['glacno', 'mb_mod_mwea', 'precfactor', 'tempbias', 'ddfsnow']
            
            posterior_all = pd.DataFrame(np.zeros((main_glac_rgi.shape[0], len(posterior_cns))), columns=posterior_cns)
            
            for n, glac_str_wRGI in enumerate(main_glac_rgi['RGIId'].values):
                if n%500 == 0:
                    print(n, glac_str_wRGI)
                # Glacier string
                glacier_str = glac_str_wRGI.split('-')[1]
                # Glacier number
                glacno = int(glacier_str.split('.')[1])
                # RGI information
                glac_idx = main_glac_rgi.index.values[n]
                glacier_rgi_table = main_glac_rgi.loc[glac_idx, :]
                # Calibration data
                glacier_cal_data = (cal_data.loc[glac_idx,:]).copy()        
                # Select observed mass balance, error, and time data
                t1 = glacier_cal_data['t1']
                t2 = glacier_cal_data['t2']
                t1_idx = int(glacier_cal_data['t1_idx'])
                t2_idx = int(glacier_cal_data['t2_idx'])
                observed_massbal = glacier_cal_data['mb_mwe'] / (t2 - t1)
                observed_error = glacier_cal_data['mb_mwe_err'] / (t2 - t1)
                mb_obs_max = observed_massbal + 3 * observed_error
                mb_obs_min = observed_massbal - 3 * observed_error
                
                # MCMC Analysis
                ds = xr.open_dataset(netcdf_fp + glacier_str + '.nc')
                df = pd.DataFrame(ds['mp_value'].values[:,:,0], columns=ds.mp.values)  
                
                # Posteriors            
                posterior_row = [glacier_str, df.mb_mwea.mean(), df.precfactor.mean(), df.tempchange.mean(), 
                                 df.ddfsnow.mean()]
                posterior_all.loc[n,:] = posterior_row
                
                ds.close()
            modelparams_all = main_glac_rgi[['RGIId', 'CenLon', 'CenLat', 'O1Region', 'Area', 'RefDate', 'glacno', 
                                             'RGIId_float']]
            modelparams_all = pd.concat([modelparams_all, cal_data[['mb_mwe', 'mb_mwe_err', 't1', 't2', 'area_km2']]], 
                                        axis=1)
            modelparams_all['mb_mwea'] = cal_data['mb_mwe'] / (cal_data['t2'] - cal_data['t1'])
            modelparams_all['mb_mwea_err'] = cal_data['mb_mwe_err'] / (cal_data['t2'] - cal_data['t1'])
            modelparams_all = pd.concat([modelparams_all, posterior_all], axis=1)
            modelparams_all.to_csv(figure_fp + modelparams_fn)
        else:
            modelparams_all = pd.read_csv(figure_fp + modelparams_fn, index_col=0)
            
    
        modelparams_all['dif_cal_era_mean'] = modelparams_all['mb_mwea'] - modelparams_all['mb_mod_mwea']
        modelparams_all['zscore'] = ((modelparams_all['mb_mod_mwea'] - modelparams_all['mb_mwea']) / 
                                      modelparams_all['mb_mwea_err'])
        
        if option_add_calopt2 == 1:
            modelparams_all_opt2 = pd.read_csv(mcmc_output_netcdf_fp_all + cal_opt2_fn, index_col=0)
            modelparams_all_opt2_prior = pd.read_csv(mcmc_output_netcdf_fp_all + prior_compare_fn, index_col=0)
            # Replace tempbias where reached positive max mass balance with tc_bndlow
            modelparams_all['tc_bndlow'] = modelparams_all_opt2_prior['prior_tc_bndlow']
            modelparams_all['tempbias'] = np.where(modelparams_all['dif_cal_era_mean'] > 0.1, 
                                                   modelparams_all['tc_bndlow'], modelparams_all['tempbias'])
    
        # remove nan values
        modelparams_all = (
                modelparams_all.drop(np.where(np.isnan(modelparams_all['mb_mod_mwea'].values) == True)[0].tolist(), 
                                       axis=0))
        modelparams_all.reset_index(drop=True, inplace=True)
        
        # Add watersheds, regions, degrees, mascons, and all groups to main_glac_rgi_all
        # Watersheds
        modelparams_all['watershed'] = modelparams_all.RGIId.map(watershed_dict)
        # Regions
        modelparams_all['kaab'] = modelparams_all.RGIId.map(kaab_dict)
        # Degrees
        modelparams_all['CenLon_round'] = np.floor(modelparams_all.CenLon.values/degree_size) * degree_size
        modelparams_all['CenLat_round'] = np.floor(modelparams_all.CenLat.values/degree_size) * degree_size
        deg_groups = modelparams_all.groupby(['CenLon_round', 'CenLat_round']).size().index.values.tolist()
        deg_dict = dict(zip(deg_groups, np.arange(0,len(deg_groups))))
        modelparams_all.reset_index(drop=True, inplace=True)
        cenlon_cenlat = [(modelparams_all.loc[x,'CenLon_round'], modelparams_all.loc[x,'CenLat_round']) 
                         for x in range(len(modelparams_all))]
        modelparams_all['CenLon_CenLat'] = cenlon_cenlat
        modelparams_all['deg_id'] = modelparams_all.CenLon_CenLat.map(deg_dict)
        # All
        modelparams_all['all_group'] = 'all'
    
        # Rename columns    
        modelparams_all = modelparams_all.rename(columns={'tempbias':'tempchange'})
        
        # Convert DDFsnow to mm w.e.d-1C-1
        modelparams_all['ddfsnow'] = modelparams_all['ddfsnow'] * 10**3    
        
        #%%
        # Histogram: Mass balance [mwea], Observation - ERA
        hist_cn = 'dif_cal_era_mean'
        low_bin = np.floor(modelparams_all[hist_cn].min())
        high_bin = np.ceil(modelparams_all[hist_cn].max())
        bins = [low_bin, -0.1, -0.05, -0.01, 0.01, 0.05, 0.1, high_bin]
        plot_hist(modelparams_all, hist_cn, bins, xlabel='Mass balance [mwea]\n(Calibration - MCMC_mean)', 
                  ylabel='# Glaciers', fig_fn='HH2015_MB_cal_vs_mcmc_hist.png', fig_fp=figure_fp)
    
        # Map: Mass change, difference between calibration data and median data
        #  Area [km2] * mb [mwe] * (1 km / 1000 m) * density_water [kg/m3] * (1 Gt/km3  /  1000 kg/m3)
        modelparams_all['mb_cal_Gta'] = modelparams_all['mb_mwea'] * modelparams_all['Area'] / 1000
        modelparams_all['mb_era_Gta'] = modelparams_all['mb_mod_mwea'] * modelparams_all['Area'] / 1000
        print('All MB cal (mean) [gt/yr]:', np.round(modelparams_all['mb_cal_Gta'].sum(),3),
              '\nAll MB ERA (mean) [gt/yr]:', np.round(modelparams_all['mb_era_Gta'].sum(),3))
    
        #%%    
        # Map & Scatterplot of mass balance difference
        plot_spatialmap_mbdif(vns, grouping, modelparams_all, xlabel, ylabel, figure_fp=figure_fp, 
                              fig_fn_prefix='HH2015_', option_group_regions=1)
        
        # Spatial distribution of parameters    
    #    vns = ['dif_masschange', 'precfactor', 'tempchange', 'ddfsnow']
        vns = ['precfactor', 'tempchange', 'ddfsnow']
        
        midpt_dict = {'dif_masschange':0,
                      'precfactor':1,
                      'tempchange':0,
                      'ddfsnow':4.1}
        cmap_dict = {'dif_masschange':'RdYlBu_r',
                     'precfactor':'RdYlBu',
                     'tempchange':'RdYlBu_r',
                     'ddfsnow':'RdYlBu_r'}
        title_adj = {'dif_masschange':15,
                     'precfactor':6,
                     'tempchange':6,
                     'ddfsnow':6}
        
        plot_spatialmap_parameters(vns, grouping, modelparams_all, xlabel, ylabel, midpt_dict, cmap_dict, title_adj, 
                                   figure_fp=figure_fp, fig_fn_prefix='HH2015_', option_group_regions=1)
    
            
    #%%
    if option_plot_era_normalizedchange == 1:
        
        vn = 'massbaltotal_glac_monthly'
        grouping = 'all'
        glac_float = 13.26360
        glac_str = '13.26360'
        labelsize = 13
        
        netcdf_fp_era = input.output_sim_fp + '/ERA-Interim/ERA-Interim_2000_2018_nochg/'
        figure_fp = netcdf_fp_era + 'figures/'
        
        rgi_regions = [13,14,15]
        
        # Load all glaciers
        for rgi_region in rgi_regions:
            # Data on all glaciers
            main_glac_rgi_region = modelsetup.selectglaciersrgitable(rgi_regionsO1=[rgi_region], rgi_regionsO2 = 'all', 
                                                                     rgi_glac_number='all')
             # Glacier hypsometry [km**2]
            main_glac_hyps_region = modelsetup.import_Husstable(main_glac_rgi_region, input.hyps_filepath,
                                                                input.hyps_filedict, input.hyps_colsdrop)
            # Ice thickness [m], average
            main_glac_icethickness_region= modelsetup.import_Husstable(main_glac_rgi_region, 
                                                                     input.thickness_filepath, input.thickness_filedict, 
                                                                     input.thickness_colsdrop)
            
            if rgi_region == rgi_regions[0]:
                main_glac_rgi_all = main_glac_rgi_region
                main_glac_hyps_all = main_glac_hyps_region
                main_glac_icethickness_all = main_glac_icethickness_region
            else:
                main_glac_rgi_all = pd.concat([main_glac_rgi_all, main_glac_rgi_region], sort=False)
                main_glac_hyps_all = pd.concat([main_glac_hyps_all, main_glac_hyps_region], sort=False)
                main_glac_icethickness_all = pd.concat([main_glac_icethickness_all, main_glac_icethickness_region], 
                                                       sort=False)
        main_glac_hyps_all = main_glac_hyps_all.fillna(0)
        main_glac_icethickness_all = main_glac_icethickness_all.fillna(0)
        # All
        main_glac_rgi_all['all_group'] = 'all'
        
        def partition_era_groups(grouping, vn, main_glac_rgi_all):
            """Partition multimodel data by each group for all GCMs for a given variable
            
            Parameters
            ----------
            grouping : str
                name of grouping to use
            vn : str
                variable name
            main_glac_rgi_all : pd.DataFrame
                glacier table
                
            Output
            ------
            time_values : np.array
                time values that accompany the multimodel data
            ds_group : list of lists
                dataset containing the multimodel data for a given variable for all the GCMs
            ds_glac : np.array
                dataset containing the variable of interest for each gcm and glacier
            """
            # Groups
            groups, group_cn = select_groups(grouping, main_glac_rgi_all)
            
            # variable name
            if vn == 'volume_norm' or vn == 'mass_change':
                vn_adj = 'volume_glac_annual'
            elif vn == 'peakwater':
                vn_adj = 'runoff_glac_annual'
            else:
                vn_adj = vn
            
            ds_group = [[] for group in groups]
            for region in rgi_regions:
                # Load datasets
                ds_fn = 'R' + str(region) + '_ERA-Interim_c2_ba1_100sets_2000_2018.nc'
                ds = xr.open_dataset(netcdf_fp_era + ds_fn)
                # Extract time variable
                if 'annual' in vn_adj:
                    try:
                        time_values = ds[vn_adj].coords['year_plus1'].values
                    except:
                        time_values = ds[vn_adj].coords['year'].values
                elif 'monthly' in vn_adj:
                    time_values = ds[vn_adj].coords['time'].values
                    
                # Merge datasets
                if region == rgi_regions[0]:
                    vn_glac_all = ds[vn_adj].values[:,:,0]
                    vn_glac_std_all = ds[vn_adj].values[:,:,1]
                else:
                    vn_glac_all = np.concatenate((vn_glac_all, ds[vn_adj].values[:,:,0]), axis=0)
                    vn_glac_std_all = np.concatenate((vn_glac_std_all, ds[vn_adj].values[:,:,1]), axis=0)
                
                # Close dataset
                ds.close()
        
            ds_glac = [vn_glac_all, vn_glac_std_all]
            
            # Cycle through groups
            for ngroup, group in enumerate(groups):
                # Select subset of data
                main_glac_rgi = main_glac_rgi_all.loc[main_glac_rgi_all[group_cn] == group]                        
                vn_glac = vn_glac_all[main_glac_rgi.index.values.tolist(),:]
                vn_glac_std = vn_glac_std_all[main_glac_rgi.index.values.tolist(),:]
                vn_glac_var = vn_glac_std **2                
                
                # Regional mean, standard deviation, and variance
                #  mean: E(X+Y) = E(X) + E(Y)
                #  var: Var(X+Y) = Var(X) + Var(Y) + 2*Cov(X,Y)
                #    assuming X and Y are indepdent, then Cov(X,Y)=0, so Var(X+Y) = Var(X) + Var(Y)
                #  std: std(X+Y) = (Var(X+Y))**0.5
                # Regional sum
                vn_reg = vn_glac.sum(axis=0)
                vn_reg_var = vn_glac_var.sum(axis=0)
        #        vn_reg_std = vn_glac_var**0.5
                
                # Record data for multi-model stats
                ds_group[ngroup] = [group, vn_reg, vn_reg_var]
                        
            return groups, time_values, ds_group, ds_glac
        
        
        groups, time_values, ds_group_mb, ds_glac_mb = partition_era_groups(grouping, 'massbaltotal_glac_monthly', 
                                                                            main_glac_rgi_all)
        groups, time_values_area, ds_group_area, ds_glac_area = partition_era_groups(grouping, 'area_glac_annual', 
                                                                                     main_glac_rgi_all)
        
        # Datasets
        mb_glac = ds_glac_mb[0]
        mb_glac_std = ds_glac_mb[1]
        area_glac = np.repeat(ds_glac_area[0][:,:-1], 12, axis=1)
        area_glac_annual = ds_glac_area[0][:,:-1]
        
        # Mass balance [mwea]
        mb_glac_annual = gcmbiasadj.annual_sum_2darray(mb_glac)
        mb_glac_annual_mwea = mb_glac_annual.sum(axis=1) / mb_glac_annual.shape[1]
        mb_glac_var = mb_glac_std**2
        mb_glac_var_annual = gcmbiasadj.annual_sum_2darray(mb_glac_var)
        mb_glac_std_annual = mb_glac_var_annual**0.5
        mb_glac_std_annual_mwea = (mb_glac_var_annual.sum(axis=1) / mb_glac_var_annual.shape[1])**0.5
        
        # Monthly glacier mass change
        #  Area [km2] * mb [mwe] * (1 km / 1000 m) * density_water [kg/m3] * (1 Gt/km3  /  1000 kg/m3)
        masschange_glac = area_glac * mb_glac / 1000 
        masschange_all = masschange_glac.sum(axis=0)
        masschange_all_Gta = masschange_all.sum() / (masschange_all.shape[0] / 12)
        
        masschange_glac_var = (area_glac * mb_glac_std / 1000)**2
        masschange_glac_var_annual = gcmbiasadj.annual_sum_2darray(masschange_glac_var)
        masschange_all_var = masschange_glac_var.sum(axis=0)
        masschange_all_var_annual = masschange_glac_var_annual.sum(axis=0)
        masschange_all_std_Gta = (masschange_all_var_annual.sum() / masschange_all_var_annual.shape[0])**0.5
        
        print('check calculations for how to estimate standard deviation of Gt/yr', 
              '\n currently underestimates, but unclear if this is due to simulation or the methods for computing std')
        
        # Initial mass
        #  Area [km2] * ice thickness [m ice] * density_ice [kg/m3] / density_water [kg/m3] * (1 km / 1000 m) 
        #  * (1 Gt/ 1km3 water)
        mass_glac_init_Gt = ((main_glac_hyps_all.values * main_glac_icethickness_all.values).sum(axis=1) * 
                     input.density_ice / input.density_water / 1000)
        mass_all_init_Gt = mass_glac_init_Gt.sum()
        # Normalized mass time series
        mass_all_timeseries_cumsum = np.cumsum(masschange_all)
        mass_all_timeseries = mass_all_init_Gt + mass_all_timeseries_cumsum
        norm_mass_all_timeseries = mass_all_timeseries / mass_all_init_Gt
        # Normalized mass time series +/- 95% confidence interval
        mass_all_timeseries_cumsum_std = np.cumsum((masschange_all_var))**0.5
        norm_mass_all_timeseries_std = mass_all_timeseries_cumsum_std / mass_all_init_Gt    
        norm_mass_all_timeseries_upper = norm_mass_all_timeseries + 1.96*norm_mass_all_timeseries_std
        norm_mass_all_timeseries_lower = norm_mass_all_timeseries - 1.96*norm_mass_all_timeseries_std
        vn_norm = norm_mass_all_timeseries
        vn_norm_upper = norm_mass_all_timeseries_upper
        vn_norm_lower = norm_mass_all_timeseries_lower
        
        # One glacier
        glac_idx =  np.where(main_glac_rgi_all['RGIId_float'].values == glac_float)[0][0]
        single_mass_glac_init_Gt = mass_glac_init_Gt[glac_idx]
        # Normalized mass time series
        single_mass_timeseries_cumsum = np.cumsum(masschange_glac[glac_idx])
        single_mass_timeseries = single_mass_glac_init_Gt + single_mass_timeseries_cumsum
        single_norm_mass_timeseries = single_mass_timeseries / single_mass_glac_init_Gt
        # Normalized mass time series +/- 95% confidence interval
        single_mass_timeseries_cumsum_std = np.cumsum((masschange_glac_var[glac_idx]))**0.5
        single_norm_mass_timeseries_std = single_mass_timeseries_cumsum_std / single_mass_glac_init_Gt    
        single_norm_mass_timeseries_upper = single_norm_mass_timeseries + 1.96*single_norm_mass_timeseries_std
        single_norm_mass_timeseries_lower = single_norm_mass_timeseries - 1.96*single_norm_mass_timeseries_std
    
        vn_glac_norm = single_norm_mass_timeseries
        vn_glac_norm_upper = single_norm_mass_timeseries_upper
        vn_glac_norm_lower = single_norm_mass_timeseries_lower
        
        # ===== PLOT ======
        fig, ax = plt.subplots(1, 1, squeeze=False, figsize=(10,8), gridspec_kw = {'wspace':0, 'hspace':0})
        # All glaciers
        ax[0,0].plot(time_values, vn_norm, color='k', linewidth=1, label='HMA')
        ax[0,0].fill_between(time_values, vn_norm_lower, vn_norm_upper, facecolor='k', alpha=0.15, label=r'$\pm$95%')
    
        # Individual glacier        
        ax[0,0].plot(time_values, vn_glac_norm, color='b', linewidth=1, label=glac_str)
        ax[0,0].fill_between(time_values, vn_glac_norm_lower, vn_glac_norm_upper, facecolor='b', alpha=0.15, label=None)
        
        # Tick parameters
    #    ax[0,0].tick_params(axis='both', which='major', labelsize=labelsize, direction='inout')
    #    ax[0,0].tick_params(axis='both', which='minor', labelsize=labelsize, direction='inout')
        # X-label
        ax[0,0].set_xlim(time_values.min(), time_values.max())
    #    ax[0,0].xaxis.set_tick_params(labelsize=labelsize)
    #    ax[0,0].xaxis.set_major_locator(plt.MultipleLocator(5))
    #    ax[0,0].xaxis.set_minor_locator(plt.MultipleLocator(1))
        # Y-label
        ax[0,0].set_ylabel('Normalized volume [-]', fontsize=labelsize+1)
    #    ax[0,0].yaxis.set_tick_params(labelsize=labelsize)
    #    ax[0,0].yaxis.set_major_locator(plt.MultipleLocator(0.1))
    #    ax[0,0].yaxis.set_minor_locator(plt.MultipleLocator(0.02))
        
        # Legend
        ax[0,0].legend(loc='lower left', fontsize=labelsize-2)
    
        # Save figure
        fig.set_size_inches(5,3)
        glac_float_str = str(glac_str).replace('.','-')
    #    figure_fn = 'HMA_normalizedchange_wglac' + glac_float_str + '.png'
        figure_fn = 'test.png'
        
        fig.savefig(figure_fp + figure_fn, bbox_inches='tight', dpi=300)    
        
    
    
    #%%
    ## open dataset
    #glac_str = '13.00964'
    #netcdf = filepath + glac_str + '.nc'
    #chainlength = 10000
    #vn = 'massbal'
    #burn = 0
    #t_quantile = 0.95
    #
    #ds = xr.open_dataset(netcdf)
    #
    ## Trying to use kstest to test if distribution is not uniform
    #chain = ds['mp_value'].sel(chain=0, mp='precfactor').values[burn:chainlength]
    #bins = np.arange(int(chain.min()*100)/100,int(chain.max()*100+1)/100, 0.1)
    #plt.hist(chain, bins=bins)
    #plt.show()
    #from scipy.stats import kstest
    #n = uniform(loc=int(chain.min()*100)/100, scale=int(chain.max()*100+1)/100)
    #kstest(chain, 'norm')
    #
    #ds.close()
        
    #%%
    if option_raw_plotchain == 1:
        print('plot chain')
        iter_length = 20
        output_filepath = input.output_filepath + 'cal_opt2/'
        ds = xr.open_dataset(output_filepath + '13.03473.nc')
        df = pd.DataFrame(ds['mp_value'].values[:,:,0], columns=ds.mp.values)  
        ds.close()
        fig, ax = plt.subplots(2, 1, squeeze=False, figsize=(10,8), gridspec_kw = {'wspace':0, 'hspace':0.25})
        # All glaciers
        ax[0,0].plot(df.index.values[0:iter_length], df.massbal.values[0:iter_length], color='k', label='massbal')
        ax[0,0].plot(df.index.values[0:iter_length], df.precfactor.values[0:iter_length], color='b', label='precfactor')
        ax[0,0].plot(df.index.values[0:iter_length], df.tempchange.values[0:iter_length], color='r', label='tempchange')
        ax2 = ax[0,0].twinx()
        ax2.plot(df.index.values[0:iter_length], df.ddfsnow.values[0:iter_length], color='g', label='ddfsnow')
        ax[0,0].legend(loc='upper left')
        ax2.legend(loc='upper right')
        ax[0,0].set_title('Chains')
        ax[0,0].set_xlabel('Step Number')
        ax[0,0].set_ylabel('Parameter value')
        
        # Change
        dif_mb = df.massbal.values[1:iter_length+1] - df.massbal.values[0:iter_length]
        dif_ddfsnow = df.ddfsnow.values[1:iter_length+1] - df.ddfsnow.values[0:iter_length]
        dif_precfactor = df.precfactor.values[1:iter_length+1] - df.precfactor.values[0:iter_length]
        dif_tempchange = df.tempchange.values[1:iter_length+1] - df.tempchange.values[0:iter_length]
    #    ax[1,0].plot(df.index.values[0:iter_length], dif_mb, color='k', label='massbal')
        ax[1,0].plot(df.index.values[0:iter_length], dif_precfactor, color='b', label='precfactor')
        ax[1,0].plot(df.index.values[0:iter_length], dif_tempchange, color='r', label='tempchange')
        ax[1,0].set_ylim(-2,2)
        ax3 = ax[1,0].twinx()
        ax3.plot(df.index.values[0:iter_length], dif_ddfsnow, color='g', label='ddfsnow')
        ax3.set_ylim(-0.0015,0.0015)
        ax[1,0].set_title('Difference Step_t - Step_t-1')
        ax[1,0].set_xlabel('Step Number')
        ax[1,0].set_ylabel('Parameter value Step_t - Step_t-1')
        plt.show()
        
    #%% Regional prior distributions
    if option_regional_priors == 1:
        grouping = 'himap'
        fig_fp = input.output_filepath + 'cal_opt4_20190803/figures/'
        
    #    ds = pd.read_csv(input.output_filepath + 'cal_opt2_spc_20190308_adjp12_wpriors/prior_compare_all.csv')
        ds = pd.read_csv(input.output_filepath + 'cal_opt4_20190803/csv/df_all_95536_glac.csv')
        ds['glacno'] = [x.split('-')[1] for x in ds['RGIId'].values]
        # add himap regions
        ds['himap'] = ds.RGIId.map(himap_dict)
        ds['kaab'] = ds.RGIId.map(kaab_dict)
        regions = list(ds[grouping].unique())
        regions = [x for x in regions if str(x) != 'nan']    
    
        # Set up your plot (and/or subplots)
        # load glacier data
        main_glac_rgi = load_glacierdata_byglacno(ds.glacno.values.tolist(), option_loadhyps_climate=0, 
                                                  option_loadcal_data=0)
    #    # Look for trends in temp_bias priors
    #    ds['Area'] = main_glac_rgi['Area'] 
    #    ds['tc_range'] = ds['prior_tc_bndhigh'] - ds['prior_tc_bndlow']
    #    fig, ax = plt.subplots(1, 1, squeeze=False, sharex=False, sharey=False, gridspec_kw = {'wspace':0.4, 'hspace':0.15})
    #    ax[0,0].scatter(ds['Area'].values, ds['tc_range'].values, color='k', linewidth=1, marker='o', s=0.00001)
    #    # Labels
    #    ax[0,0].set_xlabel('Area [km2]', size=12)
    #    ax[0,0].set_ylabel('TC range [degC]', size=12)       
    #    # Limits
    #    ax[0,0].set_xlim([0,20])
    #    plt.show()
    #    fig, ax = plt.subplots(1, 1, squeeze=False, sharex=False, sharey=False, gridspec_kw = {'wspace':0.4, 'hspace':0.15})
    #    ax[0,0].scatter(ds['prior_tc_std'].values, ds['tc_range'].values, color='k', linewidth=1, marker='o', s=0.2)
    #    # Labels
    #    ax[0,0].set_xlabel('TC std [degC]', size=12)
    #    ax[0,0].set_ylabel('TC range [degC]', size=12)       
    #    # Limits
    ##    ax[0,0].set_xlim([0,20])
    #    plt.show()
        
        
        #%%
        # Loop through regions and get priors for each
        prior_pf_reg_dict = {}
        prior_tc_reg_dict = {}
        pf_vn = 'precfactor'
        tc_vn = 'tempchange'
        for region in regions:
    #    for region in ['Karakoram']:
    #    for region in ['inner_TP']:
            ds_region = ds[ds[grouping] == region]
            print('\n\n', region, ds_region.shape)
            
            # Precipitation Factor
            print('Precipitation factor:')
            ds_region[pf_vn].plot.hist(bins=100)
            plt.show()
            print('mean:', np.round(ds_region[pf_vn].mean(),2), 'std:', np.round(ds_region[pf_vn].std(),2))
            
            beta = ds_region[pf_vn].mean() / ds_region[pf_vn].std()
            alpha = ds_region[pf_vn].mean() * beta
            
            print('alpha:', np.round(alpha,2), 'beta:', np.round(beta,2))
            
            prior_pf_reg_dict[region] = [alpha, beta]
            
            # Temperature Bias
            print('Temperature bias:')
            ax = ds_region[tc_vn].plot.hist(bins=100)
            ax.plot([0,0],[400,400])
            plt.show()
            
            mu = ds_region[tc_vn].mean()
            std = ds_region[tc_vn].std()
            print('mean:', np.round(mu,2), 'std:', np.round(std,2))
            
    #        beta = ds_region.prior_tc_mu.mean() / ds_region.prior_tc_mu.std()
    #        alpha = ds_region.prior_tc_mu.mean() * beta
    #        
    #        print('alpha:', np.round(alpha,2), 'beta:', np.round(beta,2))
            
            prior_tc_reg_dict[region] = [mu, std]
        
        #%%
        # Plot histogram and distributions
        nbins = 50    
        ncols = 4
        nrows = int(np.ceil(len(regions)/ncols))
        
        # ===== REGIONAL PRIOR: PRECIPITATION FACTOR ======
        fig, ax = plt.subplots(nrows, ncols, squeeze=False, gridspec_kw={'wspace':0.5, 'hspace':0.5})
        
        regions = sorted(regions)
        nrow = 0
        ncol = 0
        for nregion, region in enumerate(regions):
            ds_region = ds[ds[grouping] == region]
            nglaciers = ds_region.shape[0]
            
            # Plot histogram
            counts, bins, patches = ax[nrow,ncol].hist(ds_region[pf_vn].values, facecolor='grey', edgecolor='grey', 
                                                       linewidth=0.1, bins=50, density=True)
            
            # Plot gamma distribution
            beta = ds_region[pf_vn].mean() / ds_region[pf_vn].std()
            alpha = ds_region[pf_vn].mean() * beta
            rv = stats.gamma(alpha, scale=1/beta)
            ax[nrow,ncol].plot(bins, rv.pdf(bins), color='k')
            # add alpha and beta as text
            gammatext = r'$\alpha$=' + str(np.round(alpha,2)) + '\n' + r'$\beta$=' + str(np.round(beta,2))
            ax[nrow,ncol].text(0.98, 0.95, gammatext, size=10, horizontalalignment='right', 
                               verticalalignment='top', transform=ax[nrow,ncol].transAxes)
            
            # Title
            ax[nrow,ncol].text(0.5, 1.01, title_dict[region], size=10, horizontalalignment='center', 
                               verticalalignment='bottom', transform=ax[nrow,ncol].transAxes)
            
            # Adjust row and column
            ncol += 1
            if ncol == ncols:
                nrow += 1
                ncol = 0
    
        # Remove extra plots    
        n_extras = len(regions)%ncols 
        if n_extras > 0:
            for nextra in np.arange(0,n_extras):
                ax[nrow,ncol].axis('off')
                ncol += 1
                
        # Y-label
        fig.text(0.04, 0.5, 'Probability Density', va='center', ha='center', rotation='vertical', size=12)
            
        # Save figure
        if os.path.exists(fig_fp) == False:
            os.makedirs(fig_fp)    
        fig.set_size_inches(7, 9)
        fig.savefig(fig_fp + 'regional_priors_precfactor.png', bbox_inches='tight', dpi=300)
        
        #%%
        # ===== REGIONAL PRIOR: TEMPERATURE BIAS ======
        fig, ax = plt.subplots(nrows, ncols, squeeze=False, gridspec_kw={'wspace':0.5, 'hspace':0.5})    
        
        regions = sorted(regions)
        nrow = 0
        ncol = 0
        for nregion, region in enumerate(regions):
            ds_region = ds[ds[grouping] == region]
            nglaciers = ds_region.shape[0]
            
            # Plot histogram
            counts, bins, patches = ax[nrow,ncol].hist(ds_region[tc_vn].values, facecolor='grey', edgecolor='grey', 
                                                       linewidth=0.1, bins=50, density=True)
            
            # Plot gamma distribution
            mu = ds_region[tc_vn].mean()
            sigma = ds_region[tc_vn].std()
            rv = stats.norm(loc=mu, scale=sigma)
            ax[nrow,ncol].plot(bins, rv.pdf(bins), color='k')
            # add alpha and beta as text
            normtext = r'$\mu$=' + str(np.round(mu,2)) + '\n' + r'$\sigma$=' + str(np.round(sigma,2))
            ax[nrow,ncol].text(0.98, 0.95, normtext, size=10, horizontalalignment='right', 
                               verticalalignment='top', transform=ax[nrow,ncol].transAxes)
            
            # Title
            ax[nrow,ncol].text(0.5, 1.01, title_dict[region], size=10, horizontalalignment='center', 
                               verticalalignment='bottom', transform=ax[nrow,ncol].transAxes)
            
            # Adjust row and column
            ncol += 1
            if ncol == ncols:
                nrow += 1
                ncol = 0
    
        # Remove extra plots    
        n_extras = len(regions)%ncols 
        if n_extras > 0:
            for nextra in np.arange(0,n_extras):
                ax[nrow,ncol].axis('off')
                ncol += 1
                
        # Y-label
        fig.text(0.04, 0.5, 'Probability Density', va='center', ha='center', rotation='vertical', size=12)
            
        # Save figure
        if os.path.exists(fig_fp) == False:
            os.makedirs(fig_fp)    
        fig.set_size_inches(7, 9)
        fig.savefig(fig_fp + 'regional_priors_tempbias.png', bbox_inches='tight', dpi=300)
        
        #%%
    #    # ===== REGIONAL PRIOR: TEMPERATURE BIAS (GAMMA) ======
    #    fig, ax = plt.subplots(nrows, ncols, squeeze=False, gridspec_kw={'wspace':0.5, 'hspace':0.5})
    #    
    #    regions = sorted(regions)
    #    nrow = 0
    #    ncol = 0
    #    for nregion, region in enumerate(regions):
    #        ds_region = ds[ds[grouping] == region]
    #        nglaciers = ds_region.shape[0]
    #        
    #        # Plot histogram
    #        counts, bins, patches = ax[nrow,ncol].hist(ds_region.prior_tc_mu.values, facecolor='grey', edgecolor='grey', 
    #                                                   linewidth=0.1, bins=50, density=True)
    #        
    #        # Plot gamma distribution
    #        beta = ds_region.prior_tc_mu.mean() / ds_region.prior_pf_mu.std()
    #        alpha = ds_region.prior_tc_mu.mean() * beta
    #        rv = stats.gamma(alpha, scale=1/beta)
    #        ax[nrow,ncol].plot(bins, rv.pdf(bins), color='k')
    #        # add alpha and beta as text
    #        gammatext = r'$\alpha$=' + str(np.round(alpha,2)) + '\n' + r'$\beta$=' + str(np.round(beta,2))
    #        ax[nrow,ncol].text(0.98, 0.95, gammatext, size=10, horizontalalignment='right', 
    #                           verticalalignment='top', transform=ax[nrow,ncol].transAxes)
    #        ax[nrow,ncol].set_ylim([0,0.5])
    #        
    #        # Title
    #        ax[nrow,ncol].text(0.5, 1.01, title_dict[region], size=10, horizontalalignment='center', 
    #                           verticalalignment='bottom', transform=ax[nrow,ncol].transAxes)
    #        
    #        # Adjust row and column
    #        ncol += 1
    #        if ncol == ncols:
    #            nrow += 1
    #            ncol = 0
    #
    #    # Remove extra plots    
    #    n_extras = len(regions)%ncols 
    #    if n_extras > 0:
    #        for nextra in np.arange(0,n_extras):
    #            ax[nrow,ncol].axis('off')
    #            ncol += 1
    #            
    #    # Y-label
    #    fig.text(0.04, 0.5, 'Probability Density', va='center', ha='center', rotation='vertical', size=12)
    #        
    #    # Save figure
    #    fig_fp = input.output_fp_cal + 'figures/'
    #    if os.path.exists(fig_fp) == False:
    #        os.makedirs(fig_fp)    
    #    fig.set_size_inches(7, 9)
    #    fig.savefig(fig_fp + 'regional_priors_tempbias_gamma.png', bbox_inches='tight', dpi=300)
    
     
    #%% PLOT MCMC CHAINS
    if option_glacier_mcmc_plots == 1:
    #    glac_no = str(input.rgi_regionsO1[0]) + '.' + input.rgi_glac_number[0]
        glac_no = '13.45048'
        netcdf_fp = input.main_directory + '/../Output/cal_opt2_spc_20190806/'
    #    glac_no = '15.03473'
    #    netcdf_fp = input.output_fp_cal
        variables = ['massbal', 'tempchange', 'precfactor', 'ddfsnow']
        burn = 0
        iters = [10000]
        colors_iters = ['blue']
        option_pairwise_scatter = 1
        option_mb_vs_params = 1
        
        region = [int(glac_no.split('.')[0])]
        rgi_glac_number = [glac_no.split('.')[1]]
    
        # Glacier RGI data
        main_glac_rgi = modelsetup.selectglaciersrgitable(rgi_regionsO1=region, rgi_regionsO2 = 'all',
                                                          rgi_glac_number=rgi_glac_number)
        # Add regions
        main_glac_rgi['region'] = main_glac_rgi.RGIId.map(input.reg_dict)
        # Glacier hypsometry [km**2], total area
        main_glac_hyps = modelsetup.import_Husstable(main_glac_rgi, region, input.hyps_filepath,
                                                     input.hyps_filedict, input.hyps_colsdrop)
        # Ice thickness [m], average
        main_glac_icethickness = modelsetup.import_Husstable(main_glac_rgi, input.thickness_filepath, 
                                                             input.thickness_filedict, input.thickness_colsdrop)
        main_glac_hyps[main_glac_icethickness == 0] = 0
        # Width [km], average
        main_glac_width = modelsetup.import_Husstable(main_glac_rgi, input.width_filepath,
                                                      input.width_filedict, input.width_colsdrop)
        # Elevation bins
        elev_bins = main_glac_hyps.columns.values.astype(int)   
        # Select dates including future projections
        dates_table = modelsetup.datesmodelrun(startyear=input.startyear, endyear=input.endyear, spinupyears=0)
        
        # ===== LOAD CLIMATE DATA =====
        gcm = class_climate.GCM(name=input.ref_gcm_name)
        # Air temperature [degC], Precipitation [m], Elevation [masl], Lapse rate [K m-1]
        gcm_temp, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(gcm.temp_fn, gcm.temp_vn, main_glac_rgi, dates_table)
        gcm_prec, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(gcm.prec_fn, gcm.prec_vn, main_glac_rgi, dates_table)
        gcm_elev = gcm.importGCMfxnearestneighbor_xarray(gcm.elev_fn, gcm.elev_vn, main_glac_rgi)
        # Lapse rate [K m-1]
        gcm_lr, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(gcm.lr_fn, gcm.lr_vn, main_glac_rgi, dates_table)
    
        # Select dates including future projections
        #dates_table_nospinup = modelsetup.datesmodelrun(startyear=input.startyear, endyear=input.endyear, spinupyears=0)
        dates_table_nospinup = modelsetup.datesmodelrun(startyear=2000, endyear=2018, spinupyears=0)
        
        # Calibration data
        cal_data = pd.DataFrame()
        for dataset in cal_datasets:
            cal_subset = class_mbdata.MBData(name=dataset, rgi_regionO1=region)
            cal_subset_data = cal_subset.retrieve_mb(main_glac_rgi, main_glac_hyps, dates_table_nospinup)
            cal_data = cal_data.append(cal_subset_data, ignore_index=True)
        cal_data = cal_data.sort_values(['glacno', 't1_idx'])
        cal_data.reset_index(drop=True, inplace=True)
    
        for n, glac_str_wRGI in enumerate(main_glac_rgi['RGIId'].values):
            # Glacier string
            glacier_str = glac_str_wRGI.split('-')[1]
            print(glacier_str)
            # Glacier number
            glacno = int(glacier_str.split('.')[1])
            # RGI information
            glacier_rgi_table = main_glac_rgi.loc[main_glac_rgi.index.values[n], :]
            # Calibration data
            cal_idx = np.where(cal_data['glacno'] == glacno)[0]
            glacier_cal_data = (cal_data.iloc[cal_idx,:]).copy()
            # Select observed mass balance, error, and time data
            t1 = glacier_cal_data.loc[cal_idx, 't1'].values[0]
            t2 = glacier_cal_data.loc[cal_idx, 't2'].values[0]
            t1_idx = int(glacier_cal_data.loc[cal_idx,'t1_idx'])
            t2_idx = int(glacier_cal_data.loc[cal_idx,'t2_idx'])
            observed_massbal = (glacier_cal_data.loc[cal_idx,'mb_mwe'] / (t2 - t1)).values[0]
            observed_error = (glacier_cal_data.loc[cal_idx,'mb_mwe_err'] / (t2 - t1)).values[0]
            mb_obs_max = observed_massbal + 3 * observed_error
            mb_obs_min = observed_massbal - 3 * observed_error
            
            # MCMC Analysis
            ds = xr.open_dataset(netcdf_fp + glacier_str + '.nc')
            df = pd.DataFrame(ds['mp_value'].values[burn:,:,0], columns=ds.mp.values)  
            df['ddfsnow'] = df['ddfsnow'] * 1000
            print('\nddfsnow converted to mm w.e. d-1 C-1\n')
            print('MB (obs - mean_model):', np.round(observed_massbal - df.massbal.mean(),3))
            
            # Select subsets of data
            glacier_gcm_elev = gcm_elev[n]
            glacier_gcm_temp = gcm_temp[n,:]
            glacier_gcm_lrgcm = gcm_lr[n,:]
            glacier_gcm_lrglac = glacier_gcm_lrgcm.copy()
            glacier_gcm_prec = gcm_prec[n,:]
            glacier_area_t0 = main_glac_hyps.iloc[n,:].values.astype(float)
            icethickness_t0 = main_glac_icethickness.iloc[n,:].values.astype(float)
            width_t0 = main_glac_width.iloc[n,:].values.astype(float)
            glac_idx_t0 = glacier_area_t0.nonzero()[0]
            # Set model parameters
            modelparameters = [input.lrgcm, input.lrglac, input.precfactor, input.precgrad, input.ddfsnow, input.ddfice,
                               input.tempsnow, input.tempchange]
            
            tempchange_boundlow, tempchange_boundhigh, mb_max_loss = (
                    calibration.retrieve_priors(modelparameters, glacier_rgi_table, glacier_area_t0, icethickness_t0, 
                                                width_t0, elev_bins, glacier_gcm_temp, glacier_gcm_prec, 
                                                glacier_gcm_elev, glacier_gcm_lrgcm, glacier_gcm_lrglac, dates_table, 
                                                t1_idx, t2_idx, t1, t2))
            
            # Regional priors
            precfactor_gamma_alpha = input.precfactor_gamma_region_dict[glacier_rgi_table.loc['region']][0]
            precfactor_gamma_beta = input.precfactor_gamma_region_dict[glacier_rgi_table.loc['region']][1]                      
            tempchange_mu = input.tempchange_norm_region_dict[glacier_rgi_table.loc['region']][0]
            tempchange_sigma = input.tempchange_norm_region_dict[glacier_rgi_table.loc['region']][1]
            
            ddfsnow_mu = input.ddfsnow_mu * 1000
            ddfsnow_sigma = input.ddfsnow_sigma * 1000
            ddfsnow_boundlow = input.ddfsnow_boundlow * 1000
            ddfsnow_boundhigh = input.ddfsnow_boundhigh * 1000
           
        # PRIOR VS POSTERIOR PLOTS 
        fig, ax = plt.subplots(4, 4, squeeze=False, gridspec_kw={'wspace':0.45, 'hspace':0.5})
        
        param_idx_dict = {'massbal':[0,0],
                          'precfactor':[1,0],
                          'tempchange':[2,0],
                          'ddfsnow':[3,0]}
        
        for nvar, vn in enumerate(variables):        
            nrow = param_idx_dict[vn][0]
            
            # ====== PRIOR DISTRIBUTIONS ======
            if vn == 'massbal':
                z_score = np.linspace(norm.ppf(0.01), norm.ppf(0.99), 100)
                x_values = observed_massbal + observed_error * z_score
                y_values = norm.pdf(x_values, loc=observed_massbal, scale=observed_error)
            elif vn == 'precfactor': 
                if input.precfactor_disttype == 'gamma':
                    x_values = np.linspace(
                            stats.gamma.ppf(0,precfactor_gamma_alpha, scale=1/precfactor_gamma_beta), 
                            stats.gamma.ppf(0.999,precfactor_gamma_alpha, scale=1/precfactor_gamma_beta), 
                            100)                                
                    y_values = stats.gamma.pdf(x_values, a=precfactor_gamma_alpha, scale=1/precfactor_gamma_beta)    
                elif input.precfactor_disttype == 'uniform':
                    z_score = np.linspace(uniform.ppf(0.01), uniform.ppf(0.99), 100)
                    x_values = precfactor_boundlow + z_score * (precfactor_boundhigh - precfactor_boundlow)
                    y_values = uniform.pdf(x_values, loc=precfactor_boundlow, 
                                           scale=(precfactor_boundhigh - precfactor_boundlow))
                elif input.precfactor_disttype == 'lognormal':
                    precfactor_lognorm_sigma = (1/input.precfactor_lognorm_tau)**0.5
                    x_values = np.linspace(lognorm.ppf(1e-6, precfactor_lognorm_sigma), 
                                           lognorm.ppf(0.99, precfactor_lognorm_sigma), 100)
                    y_values = lognorm.pdf(x_values, precfactor_lognorm_sigma)
            elif vn == 'tempchange':
                if input.tempchange_disttype == 'uniform':
                    z_score = np.linspace(uniform.ppf(0.01), uniform.ppf(0.99), 100)
                    x_values = tempchange_boundlow + z_score * (tempchange_boundhigh - tempchange_boundlow)
                    y_values = uniform.pdf(x_values, loc=tempchange_boundlow,
                                           scale=(tempchange_boundhigh - tempchange_boundlow))
                elif input.tempchange_disttype == 'normal':
                    z_score = np.linspace(norm.ppf(0.01), norm.ppf(0.99), 100)
                    x_values = tempchange_mu + tempchange_sigma * z_score
                    y_values = norm.pdf(x_values, loc=tempchange_mu, scale=tempchange_sigma)
                elif input.tempchange_disttype == 'truncnormal':
                    tempchange_a = (tempchange_boundlow - tempchange_mu) / tempchange_sigma
                    tempchange_b = (tempchange_boundhigh - tempchange_mu) / tempchange_sigma
                    z_score = np.linspace(truncnorm.ppf(0.01, tempchange_a, tempchange_b),
                                          truncnorm.ppf(0.99, tempchange_a, tempchange_b), 100)
                    x_values = tempchange_mu + tempchange_sigma * z_score
                    y_values = truncnorm.pdf(x_values, tempchange_a, tempchange_b, loc=tempchange_mu, 
                                             scale=tempchange_sigma)
            elif vn == 'ddfsnow':            
                if input.ddfsnow_disttype == 'truncnormal':
                    ddfsnow_a = (ddfsnow_boundlow - ddfsnow_mu) / ddfsnow_sigma
                    ddfsnow_b = (ddfsnow_boundhigh - ddfsnow_mu) / ddfsnow_sigma
                    z_score = np.linspace(truncnorm.ppf(0.001, ddfsnow_a, ddfsnow_b),
                                          truncnorm.ppf(0.999, ddfsnow_a, ddfsnow_b), 100)
                    x_values = ddfsnow_mu + ddfsnow_sigma * z_score
                    y_values = truncnorm.pdf(x_values, ddfsnow_a, ddfsnow_b, loc=ddfsnow_mu, scale=ddfsnow_sigma)
                elif input.ddfsnow_disttype == 'uniform':
                    z_score = np.linspace(uniform.ppf(0.01), uniform.ppf(0.99), 100)
                    x_values = ddfsnow_boundlow + z_score * (ddfsnow_boundhigh - ddfsnow_boundlow)
                    y_values = uniform.pdf(x_values, loc=ddfsnow_boundlow,
                                           scale=(ddfsnow_boundhigh - ddfsnow_boundlow))
            # PLOT PRIOR
            ax[nrow,0].plot(x_values, y_values, color='k')
            # Labels
            ax[nrow,0].set_xlabel(vn_label_dict[vn], labelpad=0, size=10)
            ax[nrow,0].set_ylabel('Probability Density', size=10)
    #        fig.text(0.03, 0.5, 'Probability Density', va='center', ha='center', rotation='vertical', size=12)
            
            # PLOT POSTERIOR               
            for n_chain in range(len(ds.chain.values)):
                for count_iter, n_iters in enumerate(iters):
                    chain = ds['mp_value'].sel(chain=n_chain, mp=vn).values[burn:n_iters]
                
                    # gaussian distribution
                    kde = gaussian_kde(chain)
                    x_values_kde = x_values.copy()
                    y_values_kde = kde(x_values_kde)
                    
                    # Plot fitted distribution
                    ax[nrow,0].plot(x_values_kde, y_values_kde, color=colors_iters[count_iter], 
                                    linestyle=linestyles[n_chain])
                    
            # ===== PLOT CHAINS ======
            x_values = df.index.values
            y_values = df[vn].values
            ax[nrow,1].plot(x_values, y_values, color='black', linewidth=0.25)
            step_idx1 = len(x_values) - 300
            ax[nrow,2].plot(x_values[step_idx1:], y_values[step_idx1:], color='black', linewidth=0.25)
            # labels
            ax[nrow,1].set_ylabel(vn_title_dict[vn], size=10)
            ax[nrow,1].set_xlabel('Step', labelpad=0, size=10)
            ax[nrow,2].set_xlabel('Step', labelpad=0, size=10)
            
            # ===== PLOT AUTOCORRELATION =====
            acorr_maxlags = 500
            chain = df[vn].values
            chain_norm = chain - chain.mean()
            if chain.shape[0] <= acorr_maxlags:
                acorr_lags = chain.shape[0] - 1
            else:
                acorr_lags = acorr_maxlags
            ax[nrow,3].acorr(chain_norm, maxlags=acorr_lags, usevlines=False, marker='.', color='k', ms=1)
            ax[nrow,3].plot([0,acorr_lags], [0,0], color='k', linewidth=1)
            ax[nrow,3].set_xlim(0,acorr_lags)
            ax[nrow,3].set_xlabel('lag')
            ax[nrow,3].set_ylabel('autocorrelation')
            
        
        # Close dataset
        if option_mb_vs_params == 0:
            ds.close()
            
        # Save figure
        str_ending = ''
        if 'tempchange' in variables:    
            if input.tempchange_disttype == 'truncnormal': 
                str_ending += '_TCtn'
            elif input.tempchange_disttype == 'uniform':
                str_ending += '_TCu'
        if 'precfactor' in variables:                
            if input.precfactor_disttype == 'lognormal': 
                str_ending += '_PFln'
            elif input.precfactor_disttype == 'uniform':
                str_ending += '_PFu'
        if 'ddfsnow' in variables:     
            if input.ddfsnow_disttype == 'truncnormal': 
                str_ending += '_DDFtn'
            elif input.ddfsnow_disttype == 'uniform':
                str_ending += '_DDFu'        
            
        fig_fp = netcdf_fp + 'figures/'
        if os.path.exists(fig_fp) == False:
            os.makedirs(fig_fp)    
        fig.set_size_inches(10, 8)
        fig.savefig(fig_fp + 'prior_v_posteriors_' + glacier_str + str_ending + '.png', 
                    bbox_inches='tight', dpi=300)
        plt.show()
        fig.clf()
        #%%
        # ===== PAIRWISE SCATTER PLOTS ===========================================================
        if option_pairwise_scatter == 1:
            fig, ax = plt.subplots(4, 4, squeeze=False, gridspec_kw={'wspace':0.1, 'hspace':0.1})
            
            for h, vn1 in enumerate(variables):
                v1 = chain = df[vn1].values
                
                for j, vn2 in enumerate(variables):
                    v2 = chain = df[vn2].values
                    
                    # Plot histogram
                    if h == j:
                        ax[h,j].hist(v1)
                        ax[h,j].tick_params(axis='both', bottom=False, left=False, labelleft=False, labelbottom=False)
                    # Plot scatter plot
                    elif h > j:
                        ax[h,j].plot(v2, v1, '.', mfc='none', mec='black', ms=1)
                    # Add text of relationship
                    else:
                        slope, intercept, r_value, p_value, std_err = linregress(v2, v1)
                        text2plot = (vn_abbreviations_dict[vn2] + '/\n' + vn_abbreviations_dict[vn1] + '\n$R$=' +
                                     '{:.2f}'.format((r_value)))
                        ax[h,j].text(0.5, 0.5, text2plot, fontsize=12,
                                     verticalalignment='center', horizontalalignment='center')
                    
                    # Labels
                    # Bottom left
                    if (h+1 == len(variables) and (j == 0)):
                        ax[h,j].tick_params(axis='both', which='both', left=True, right=False, labelbottom=True,
                                            labelleft=True, labelright=False)
                        ax[h,j].set_xlabel(vn_abbreviations_dict[vn2])
                        ax[h,j].set_ylabel(vn_abbreviations_dict[vn1])
                    # Bottom only
                    elif h + 1 == len(variables):
                        ax[h,j].tick_params(axis='both', which='both', left=False, right=False, labelbottom=True,
                                            labelleft=False, labelright=False)
                        ax[h,j].set_xlabel(vn_abbreviations_dict[vn2])
                    # Left only (exclude histogram values)
                    elif (h !=0) and (j == 0):
                        ax[h,j].tick_params(axis='both', which='both', left=True, right=False, labelbottom=False,
                                            labelleft=True, labelright=False)
                        ax[h,j].set_ylabel(vn_abbreviations_dict[vn1])
                    else:
                        ax[h,j].tick_params(axis='both', left=False, right=False, labelbottom=False,
                                            labelleft=False, labelright=False)
                        
            fig_fp = netcdf_fp + 'figures/autocorrelation/'
            if os.path.exists(fig_fp) == False:
                os.makedirs(fig_fp)    
            fig.set_size_inches(6, 8)
            fig.savefig(fig_fp + 'pairwisescatter_' + glacier_str + str_ending + '.png', 
                        bbox_inches='tight', dpi=300)
            plt.show()
            fig.clf()
        
        #%%
        if option_mb_vs_params == 1:
            fig = plt.figure()
            
            # Custom subplots
            gs = mpl.gridspec.GridSpec(30, 20)
            ax1 = plt.subplot(gs[0:7,0:8])
            ax2 = plt.subplot(gs[0:7,12:])
            ax3 = plt.subplot(gs[10:17,0:8])
            ax4 = plt.subplot(gs[10:17,12:])
            ax5 = plt.subplot(gs[20:,:])
            
            for nvar, vn in enumerate(variables):        
                nrow = param_idx_dict[vn][0]
                
                # ====== PRIOR DISTRIBUTIONS ======
                labelsize = 12
                if vn == 'massbal':
                    z_score = np.linspace(norm.ppf(0.01), norm.ppf(0.99), 100)
                    x_values = observed_massbal + observed_error * z_score
                    y_values = norm.pdf(x_values, loc=observed_massbal, scale=observed_error)
                    # PLOT PRIOR
                    ax1.plot(x_values, y_values, color='k')
                    # Labels
                    ax1.set_xlabel(vn_label_dict[vn], labelpad=0, size=labelsize)
                    ax1.set_ylabel('Probability Density', size=labelsize)
                    # PLOT POSTERIOR               
                    chain = ds['mp_value'].sel(chain=n_chain, mp=vn).values[burn:iters[0]]
                    # gaussian distribution
                    kde = gaussian_kde(chain)
                    x_values_kde = x_values.copy()
                    y_values_kde = kde(x_values_kde)
                    # fitted distribution
                    ax1.plot(x_values_kde, y_values_kde, color=colors_iters[count_iter], linestyle=linestyles[n_chain])
                elif vn == 'precfactor': 
                    if input.precfactor_disttype == 'gamma':
                        x_values = np.linspace(
                                stats.gamma.ppf(0,precfactor_gamma_alpha, scale=1/precfactor_gamma_beta), 
                                stats.gamma.ppf(0.999,precfactor_gamma_alpha, scale=1/precfactor_gamma_beta), 
                                100)                                
                        y_values = stats.gamma.pdf(x_values, a=precfactor_gamma_alpha, scale=1/precfactor_gamma_beta)    
                    elif input.precfactor_disttype == 'uniform':
                        z_score = np.linspace(uniform.ppf(0.01), uniform.ppf(0.99), 100)
                        x_values = precfactor_boundlow + z_score * (precfactor_boundhigh - precfactor_boundlow)
                        y_values = uniform.pdf(x_values, loc=precfactor_boundlow, 
                                               scale=(precfactor_boundhigh - precfactor_boundlow))
                    elif input.precfactor_disttype == 'lognormal':
                        precfactor_lognorm_sigma = (1/input.precfactor_lognorm_tau)**0.5
                        x_values = np.linspace(lognorm.ppf(1e-6, precfactor_lognorm_sigma), 
                                               lognorm.ppf(0.99, precfactor_lognorm_sigma), 100)
                        y_values = lognorm.pdf(x_values, precfactor_lognorm_sigma)
                    # PLOT PRIOR
                    ax2.plot(x_values, y_values, color='k')
                    # Labels
                    ax2.set_xlabel(vn_label_dict[vn], labelpad=0, size=labelsize)
                    ax2.set_ylabel('Probability Density', size=labelsize)
                    # PLOT POSTERIOR               
                    chain = ds['mp_value'].sel(chain=n_chain, mp=vn).values[burn:iters[0]]
                    # gaussian distribution
                    kde = gaussian_kde(chain)
                    x_values_kde = x_values.copy()
                    y_values_kde = kde(x_values_kde)
                    # fitted distribution
                    ax2.plot(x_values_kde, y_values_kde, color=colors_iters[count_iter], linestyle=linestyles[n_chain])
                    
                    leg_lines = []
                    leg_labels = []
                    chain_labels = ['Prior', '10000']
                    chain_colors = ['black', 'blue']
                    for n_chain_label in range(len(chain_labels)):
                        line = Line2D([0,1],[0,1], color=chain_colors[n_chain_label])
                        leg_lines.append(line)
                        leg_labels.append(chain_labels[n_chain_label])
                    ax2.legend(leg_lines, leg_labels, loc='upper right', 
    #                           bbox_to_anchor=(0.87,0.885), 
                               handlelength=1.5, handletextpad=0.25, borderpad=0.2, frameon=True)
                elif vn == 'tempchange':
                    if input.tempchange_disttype == 'uniform':
                        z_score = np.linspace(uniform.ppf(0.01), uniform.ppf(0.99), 100)
                        x_values = tempchange_boundlow + z_score * (tempchange_boundhigh - tempchange_boundlow)
                        y_values = uniform.pdf(x_values, loc=tempchange_boundlow,
                                               scale=(tempchange_boundhigh - tempchange_boundlow))
                    elif input.tempchange_disttype == 'normal':
                        z_score = np.linspace(norm.ppf(0.01), norm.ppf(0.99), 100)
                        x_values = tempchange_mu + tempchange_sigma * z_score
                        y_values = norm.pdf(x_values, loc=tempchange_mu, scale=tempchange_sigma)
                    elif input.tempchange_disttype == 'truncnormal':
                        tempchange_a = (tempchange_boundlow - tempchange_mu) / tempchange_sigma
                        tempchange_b = (tempchange_boundhigh - tempchange_mu) / tempchange_sigma
                        z_score = np.linspace(truncnorm.ppf(0.01, tempchange_a, tempchange_b),
                                              truncnorm.ppf(0.99, tempchange_a, tempchange_b), 100)
                        x_values = tempchange_mu + tempchange_sigma * z_score
                        y_values = truncnorm.pdf(x_values, tempchange_a, tempchange_b, loc=tempchange_mu, 
                                                 scale=tempchange_sigma)
                    # PLOT PRIOR
                    ax3.plot(x_values, y_values, color='k')
                    # Labels
                    ax3.set_xlabel(vn_label_dict[vn], labelpad=0, size=labelsize)
                    ax3.set_ylabel('Probability Density', size=labelsize)
                    # PLOT POSTERIOR               
                    chain = ds['mp_value'].sel(chain=n_chain, mp=vn).values[burn:iters[0]]
                    # gaussian distribution
                    kde = gaussian_kde(chain)
                    x_values_kde = x_values.copy()
                    y_values_kde = kde(x_values_kde)
                    # fitted distribution
                    ax3.plot(x_values_kde, y_values_kde, color=colors_iters[count_iter], linestyle=linestyles[n_chain])
                elif vn == 'ddfsnow':            
                    if input.ddfsnow_disttype == 'truncnormal':
                        ddfsnow_a = (ddfsnow_boundlow - ddfsnow_mu) / ddfsnow_sigma
                        ddfsnow_b = (ddfsnow_boundhigh - ddfsnow_mu) / ddfsnow_sigma
                        z_score = np.linspace(truncnorm.ppf(0.001, ddfsnow_a, ddfsnow_b),
                                              truncnorm.ppf(0.999, ddfsnow_a, ddfsnow_b), 100)
                        x_values = ddfsnow_mu + ddfsnow_sigma * z_score
                        y_values = truncnorm.pdf(x_values, ddfsnow_a, ddfsnow_b, loc=ddfsnow_mu, scale=ddfsnow_sigma)
                    elif input.ddfsnow_disttype == 'uniform':
                        z_score = np.linspace(uniform.ppf(0.01), uniform.ppf(0.99), 100)
                        x_values = ddfsnow_boundlow + z_score * (ddfsnow_boundhigh - ddfsnow_boundlow)
                        y_values = uniform.pdf(x_values, loc=ddfsnow_boundlow,
                                               scale=(ddfsnow_boundhigh - ddfsnow_boundlow))
                    # PLOT PRIOR
                    ax4.plot(x_values, y_values, color='k')
                    # Labels
                    ax4.set_xlabel(vn_label_dict[vn], labelpad=0, size=labelsize)
                    ax4.set_ylabel('Probability Density', size=labelsize)
                    # PLOT POSTERIOR               
                    chain = ds['mp_value'].sel(chain=n_chain, mp=vn).values[burn:iters[0]]
                    # gaussian distribution
                    kde = gaussian_kde(chain)
                    x_values_kde = x_values.copy()
                    y_values_kde = kde(x_values_kde)
                    # fitted distribution
                    ax4.plot(x_values_kde, y_values_kde, color=colors_iters[count_iter], linestyle=linestyles[n_chain])
                        
            # Close dataset
            ds.close()
            
            tempchange_boundlow, tempchange_boundhigh, mb_max_loss = (
                    calibration.retrieve_priors(
                        modelparameters, glacier_rgi_table, glacier_area_t0, icethickness_t0, 
                        width_t0, elev_bins, glacier_gcm_temp, glacier_gcm_prec, glacier_gcm_elev, 
                        glacier_gcm_lrgcm, glacier_gcm_lrglac, dates_table, t1_idx, t2_idx, t1, t2, debug=False))
            
            # Iterations to plot
            if input.precfactor_disttype == 'gamma':
                precfactor_expected = stats.gamma.ppf(0.5,input.precfactor_gamma_alpha, scale=1/input.precfactor_gamma_beta)
                precfactor_bndhigh = stats.gamma.ppf(0.95,input.precfactor_gamma_alpha, 
                                                       scale=1/input.precfactor_gamma_beta)
                precfactor_iters = [1, int(precfactor_expected*10)/10, int(precfactor_bndhigh*10)/10]
            tc_iter_step = 0.5
            tempchange_iters = np.arange(int(tempchange_boundlow), np.ceil(tempchange_boundhigh)+tc_iter_step, 
                                         tc_iter_step).tolist()
            
            # Set manually
    #        print('\n\nTempchange iters set manually for figure generation\n\n')        
    #        tempchange_iters = np.arange(-6, 6+tc_iter_step, 1).tolist()
            
            ddfsnow_iters = [0.0026, 0.0041, 0.0056]
            
            # Max loss [mwea] for plot
            mb_max_loss = -1 * (glacier_area_t0 * icethickness_t0).sum() / glacier_area_t0.sum() / (t2-t1)
            
            mb_vs_parameters = pd.DataFrame(np.zeros((len(ddfsnow_iters) * len(precfactor_iters) * 
                                                      len(tempchange_iters), 4)),
                                            columns=['precfactor', 'tempbias', 'ddfsnow', 'massbal'])
            count=0
            for n, precfactor in enumerate(precfactor_iters):
                modelparameters[2] = precfactor
                
                # run mass balance calculation
                option_areaconstant = 0
                print('PF:', precfactor, 'option_areaconstant:', option_areaconstant)
                
                for n, tempchange in enumerate(tempchange_iters):
                    modelparameters[7] = tempchange
        
                    for c, ddfsnow in enumerate(ddfsnow_iters):
                        
                        modelparameters[4] = ddfsnow
                        modelparameters[5] = modelparameters[4] / input.ddfsnow_iceratio
                        
                        (glac_bin_temp, glac_bin_prec, glac_bin_acc, glac_bin_refreeze, glac_bin_snowpack, glac_bin_melt,
                         glac_bin_frontalablation, glac_bin_massbalclim, glac_bin_massbalclim_annual, glac_bin_area_annual,
                         glac_bin_icethickness_annual, glac_bin_width_annual, glac_bin_surfacetype_annual,
                         glac_wide_massbaltotal, glac_wide_runoff, glac_wide_snowline, glac_wide_snowpack,
                         glac_wide_area_annual, glac_wide_volume_annual, glac_wide_ELA_annual, offglac_wide_prec, 
                         offglac_wide_refreeze, offglac_wide_melt, offglac_wide_snowpack, offglac_wide_runoff) = (
                            massbalance.runmassbalance(modelparameters[0:8], glacier_rgi_table, glacier_area_t0, 
                                                       icethickness_t0, width_t0, elev_bins, glacier_gcm_temp, 
                                                       glacier_gcm_prec, glacier_gcm_elev, glacier_gcm_lrgcm, 
                                                       glacier_gcm_lrglac, dates_table, 
                                                       option_areaconstant=option_areaconstant))
                        
                        # Compute glacier volume change for every time step and use this to compute mass balance
                        #  this will work for any indexing
                        glac_wide_area = glac_wide_area_annual[:-1].repeat(12)
                        # Mass change [km3 mwe]
                        #  mb [mwea] * (1 km / 1000 m) * area [km2]
                        glac_wide_masschange = glac_wide_massbaltotal / 1000 * glac_wide_area
                        # Mean annual mass balance [mwea]
                        mb_mwea = (glac_wide_masschange.sum() / glac_wide_area[0] * 1000 / 
                                   (glac_wide_masschange.shape[0] / 12))
                        mb_vs_parameters.loc[count,:] = np.array([precfactor, tempchange, ddfsnow, mb_mwea])
                        count += 1
        
            # Subset data for each precfactor
            linestyles = ['-', '--', ':', '-.']
            linecolors = ['b', 'k', 'r']
            prec_linedict = {precfactor : linestyles[n] for n, precfactor in enumerate(precfactor_iters)} 
            ddfsnow_colordict = {ddfsnow : linecolors[n] for n, ddfsnow in enumerate(ddfsnow_iters)} 
            
            for precfactor in precfactor_iters:
                modelparameters[2] = precfactor
                mb_vs_parameters_subset = mb_vs_parameters.loc[mb_vs_parameters.loc[:,'precfactor'] == precfactor]
                for ddfsnow in ddfsnow_iters:
                    mb_vs_parameters_plot =  mb_vs_parameters_subset.loc[mb_vs_parameters_subset.loc[:,'ddfsnow'] == ddfsnow]
                    ax5.plot(mb_vs_parameters_plot.loc[:,'tempbias'], mb_vs_parameters_plot.loc[:,'massbal'], 
                             linestyle=prec_linedict[precfactor], color=ddfsnow_colordict[ddfsnow])    
            
            # Add horizontal line of mass balance observations
            ax5.axhline(observed_massbal, color='gray', linewidth=2, zorder=2)    
            observed_mb_min = observed_massbal - 3*observed_error
            observed_mb_max = observed_massbal + 3*observed_error  
            fillcolor = 'lightgrey'
            ax5.fill_between([np.min(tempchange_iters), np.max(tempchange_iters)], observed_mb_min, observed_mb_max, 
                             facecolor=fillcolor, label=None, zorder=1)
             
            ax5.set_xlim(np.min(tempchange_iters), np.max(tempchange_iters))
            if observed_massbal - 3*observed_error < mb_max_loss:
                ylim_lower = observed_massbal - 3*observed_error
            else:
                ylim_lower = np.floor(mb_max_loss)
            ax5.set_ylim(int(ylim_lower),np.ceil(mb_vs_parameters['massbal'].max()))
            ax5.set_ylim(-2,1.75)
            
            # Labels
            ax5.set_xlabel('Temperature Bias ($\mathregular{^{\circ}}$C)', fontsize=labelsize)
            ax5.set_ylabel('Mass Balance (m w.e. $\mathregular{a^{-1}}$)', fontsize=labelsize)
            
            # Add legend
            leg_lines = []
            leg_names = []
            x_min = mb_vs_parameters.loc[:,'tempbias'].min()
            y_min = mb_vs_parameters.loc[:,'massbal'].min()
            for precfactor in reversed(precfactor_iters):
                line = Line2D([x_min,y_min],[x_min,y_min], linestyle=prec_linedict[precfactor], color='gray')
                leg_lines.append(line)
                leg_names.append('$\mathregular{k_{p}}$ ' + str(precfactor))
                
            for ddfsnow in ddfsnow_iters:
                line = Line2D([x_min,y_min],[x_min,y_min], linestyle='-', color=ddfsnow_colordict[ddfsnow])
                leg_lines.append(line)
                leg_names.append('$\mathregular{f_{snow}}$ ' + str(np.round(ddfsnow*10**3,1)))
                  
            ax5.legend(leg_lines, leg_names, loc='upper right', frameon=False, labelspacing=0.25)
    
                
            fig_fp = netcdf_fp + 'figures/'
            if os.path.exists(fig_fp) == False:
                os.makedirs(fig_fp)    
            fig.set_size_inches(6.5, 8)
            fig.savefig(fig_fp + glacier_str + '_prior_v_post_wMBvparams.png', bbox_inches='tight', dpi=300)
        
    #%% PLOT MASS BALANCE VS MODEL PARAMETERS
    if option_glacier_mb_vs_params == 1:
        glac_no = ['15.10755']
    #    glac_no = [str(input.rgi_regionsO1[0]) + '.' + input.rgi_glac_number[0]]
    #    netcdf_fp = input.output_fp_cal
        netcdf_fp = mcmc_output_netcdf_fp_all
        fig_fp = netcdf_fp + 'figures/'
        
        (main_glac_rgi, main_glac_hyps, main_glac_icethickness, main_glac_width, 
         gcm_temp, gcm_prec, gcm_elev, gcm_lr, cal_data, dates_table) = load_glacierdata_byglacno(glac_no)
        
        main_glac_rgi['region'] = main_glac_rgi.RGIId.map(input.reg_dict)
        
        # Elevation bins
        elev_bins = main_glac_hyps.columns.values.astype(int) 
        
        for n, glac_str_wRGI in enumerate(main_glac_rgi['RGIId'].values):
            # Glacier string
            glacier_str = glac_str_wRGI.split('-')[1]
            print(glacier_str)
            # Glacier number
            glacno = int(glacier_str.split('.')[1])
            # RGI information
            glacier_rgi_table = main_glac_rgi.loc[main_glac_rgi.index.values[n], :]
            # Calibration data
            cal_idx = np.where(cal_data['glacno'] == glacno)[0]
            glacier_cal_data = (cal_data.iloc[cal_idx,:]).copy()
            # Select observed mass balance, error, and time data
            t1 = glacier_cal_data.loc[cal_idx, 't1'].values[0]
            t2 = glacier_cal_data.loc[cal_idx, 't2'].values[0]
            t1_idx = int(glacier_cal_data.loc[cal_idx,'t1_idx'])
            t2_idx = int(glacier_cal_data.loc[cal_idx,'t2_idx'])
            observed_massbal = (glacier_cal_data.loc[cal_idx,'mb_mwe'] / (t2 - t1)).values[0]
            observed_error = (glacier_cal_data.loc[cal_idx,'mb_mwe_err'] / (t2 - t1)).values[0]
            mb_obs_max = observed_massbal + 3 * observed_error
            mb_obs_min = observed_massbal - 3 * observed_error
            
            # MCMC Analysis
    #        ds = xr.open_dataset(netcdf_fp + glacier_str + '.nc')
    #        df = pd.DataFrame(ds['mp_value'].values[:,:,0], columns=ds.mp.values)  
    #        print('MB (obs - mean_model):', np.round(observed_massbal - df.massbal.mean(),3))
            
            # Select subsets of data
            glacier_gcm_elev = gcm_elev[n]
            glacier_gcm_temp = gcm_temp[n,:]
            glacier_gcm_lrgcm = gcm_lr[n,:]
            glacier_gcm_lrglac = glacier_gcm_lrgcm.copy()
            glacier_gcm_prec = gcm_prec[n,:]
            glacier_area_t0 = main_glac_hyps.iloc[n,:].values.astype(float)
            icethickness_t0 = main_glac_icethickness.iloc[n,:].values.astype(float)
            width_t0 = main_glac_width.iloc[n,:].values.astype(float)
            glac_idx_t0 = glacier_area_t0.nonzero()[0]
            # Set model parameters
            modelparameters = [input.lrgcm, input.lrglac, input.precfactor, input.precgrad, input.ddfsnow, input.ddfice,
                               input.tempsnow, input.tempchange]
        
            
            tempchange_boundlow, tempchange_boundhigh, mb_max_loss = (
                    calibration.retrieve_priors(
                        modelparameters, glacier_rgi_table, glacier_area_t0, icethickness_t0, 
                        width_t0, elev_bins, glacier_gcm_temp, glacier_gcm_prec, glacier_gcm_elev, 
                        glacier_gcm_lrgcm, glacier_gcm_lrglac, dates_table, t1_idx, t2_idx, t1, t2, debug=True))
            
            # Iterations to plot
            if input.precfactor_disttype == 'gamma':
                precfactor_expected = stats.gamma.ppf(0.5,input.precfactor_gamma_alpha, scale=1/input.precfactor_gamma_beta)
                precfactor_bndhigh = stats.gamma.ppf(0.95,input.precfactor_gamma_alpha, 
                                                       scale=1/input.precfactor_gamma_beta)
                precfactor_iters = [1, int(precfactor_expected*10)/10, int(precfactor_bndhigh*10)/10]
            tc_iter_step = 0.1
            tempchange_iters = np.arange(int(tempchange_boundlow), np.ceil(tempchange_boundhigh)+tc_iter_step, 
                                         tc_iter_step).tolist()
            
            # Set manually
    #        print('\n\nTempchange iters set manually for figure generation\n\n')        
    #        tempchange_iters = np.arange(-6, 6+tc_iter_step, tc_iter_step).tolist()
            
            ddfsnow_iters = [0.0026, 0.0041, 0.0056]
            
            # Max loss [mwea] for plot
            mb_max_loss = -1 * (glacier_area_t0 * icethickness_t0).sum() / glacier_area_t0.sum() / (t2-t1)
            
            # Plot
            plot_mb_vs_parameters(tempchange_iters, precfactor_iters, ddfsnow_iters, modelparameters, glacier_rgi_table, 
                                  glacier_area_t0, icethickness_t0, width_t0, elev_bins, glacier_gcm_temp, glacier_gcm_prec, 
                                  glacier_gcm_elev, glacier_gcm_lrgcm, glacier_gcm_lrglac, dates_table, observed_massbal, 
                                  observed_error, tempchange_boundhigh, tempchange_boundlow, 
                                  mb_max_loss=mb_max_loss, option_areaconstant=0, option_plotsteps=0, fig_fp=fig_fp)
    
    
    if option_convertcal2table == 1:
        netcdf_fp = input.output_filepath + 'cal_opt4_20190803/'
        chain_no = 0
        
        csv_fp = netcdf_fp + 'csv/'
        fig_fp = netcdf_fp + 'figures/'
        
        filelist = []
        for region in regions:
            filelist.extend(glob.glob(netcdf_fp + str(region) + '*.nc'))
        
        filelist = sorted(filelist)
    
        glac_no = []
        reg_no = []
        df_all = None
        for n, netcdf in enumerate(filelist):
            glac_str = netcdf.split('/')[-1].split('.nc')[0]
            glac_no.append(glac_str)
            reg_no.append(glac_str.split('.')[0])
            
            if n%500 == 0:
                print(glac_str)
            
            ds = xr.open_dataset(netcdf)
            df = pd.DataFrame(ds['mp_value'].values[:,:,0], columns=ds.mp.values)
            df['glac_no'] = glac_str
            
            if df_all is None:
                df_all = df
            else:
                df_all = df_all.append(df)
    
        glac_no = sorted(glac_no)
            
        (main_glac_rgi, main_glac_hyps, main_glac_icethickness, main_glac_width, 
         gcm_temp, gcm_prec, gcm_elev, gcm_lr, cal_data, dates_table) = load_glacierdata_byglacno(glac_no)
        
        df_export = main_glac_rgi[['RGIId', 'O1Region', 'glacno', 'Zmin', 'Zmax', 'Zmed']].copy()
        df_export['precfactor'] = df_all['precfactor'].values
        df_export['tempchange'] = df_all['tempchange'].values
        df_export['ddfsnow'] = df_all['ddfsnow'].values
        df_export['ddfice'] = df_all['ddfice'].values
        df_export['mb_mwea'] = df_all['mb_mwea'].values
        df_export['obs_mwea'] = df_all['obs_mwea'].values
        df_export['dif_mwea'] = df_all['dif_mwea'].values
        df_export['glacno_str'] = df_all['glac_no'].values
            
        if os.path.exists(csv_fp) == False:
            os.makedirs(csv_fp)
        df_export.to_csv(csv_fp + 'df_all_' + str(df_export.shape[0]) + '_glac.csv', index=False)
        
    
    #%%
    if option_correlation_scatter == 1:
        netcdf_fp = input.output_filepath + 'cal_opt2_spc_20190806/'
        csv_fp = netcdf_fp + 'csv/'
        fig_fp = netcdf_fp + 'figures/'
        chain_no = 0
        burn = 0
        chainlength = 10000
        
        variables = ['massbal', 'precfactor', 'tempchange', 'ddfsnow']
        
        csv_fp = netcdf_fp + 'csv/'
        fig_fp = netcdf_fp + 'figures/'
        
        correlation_fn = 'correlation_table_' + str(burn) + 'burn.csv'
        
        
        if os.path.isfile(csv_fp + correlation_fn):
            df_all = pd.read_csv(csv_fp + correlation_fn)
        else:
            filelist = []
            for region in regions:
                filelist.extend(glob.glob(netcdf_fp + str(region) + '*.nc'))
            
            filelist = sorted(filelist)
    
            glac_no = []
            reg_no = []
            df_all = None
            for n, netcdf in enumerate(filelist):
    #        for n, netcdf in enumerate(filelist[0:100]):
                glac_str = netcdf.split('/')[-1].split('.nc')[0]
                glac_no.append(glac_str)
                reg_no.append(glac_str.split('.')[0])
                
                if n%500 == 0:
                    print(glac_str)
                
                ds = xr.open_dataset(netcdf)
                df = pd.DataFrame(ds['mp_value'].values[burn:chainlength,:,0], columns=ds.mp.values)
                ds.close()
                
                df_cor = pd.DataFrame(df.mean()).T
                df_cor['glac_str'] = glac_str
                col_order = ['glac_str', 'massbal', 'precfactor', 'tempchange', 'ddfsnow', 'ddfice', 'lrgcm', 'lrglac', 
                             'precgrad']
                df_cor = df_cor[col_order]
                
                # ===== CORRELATION =====
                def calc_correlation(df, vn1, vn2):
                    """ Calculate correlation between two variables 
                    
                    df : pd.DataFrame
                        dataframe containing chains
                    vn1, vn2 : str
                        variable names based on df columns
                    """
                    v1 = df[vn1]
                    v2 = df[vn2]
                    slope, intercept, r_value, p_value, std_err = linregress(v2, v1)
                    return r_value
                
                df_cor['mb/pf'] = calc_correlation(df, 'massbal', 'precfactor')
                df_cor['mb/tc'] = calc_correlation(df, 'massbal', 'tempchange')
                df_cor['mb/ddf'] = calc_correlation(df, 'massbal', 'ddfsnow')
                df_cor['pf/tc'] = calc_correlation(df, 'precfactor', 'tempchange')
                df_cor['pf/ddf'] = calc_correlation(df, 'precfactor', 'ddfsnow')
                df_cor['tc/ddf'] = calc_correlation(df, 'tempchange', 'ddfsnow')
                
                if df_all is None:
                    df_all = df_cor
                else:
                    df_all = df_all.append(df_cor)  
        
            if os.path.exists(csv_fp) == False:
                os.makedirs(csv_fp)
            df_all.to_csv(csv_fp + correlation_fn, index=False)
        #%%
        # Plot combinations
        combinations = ['mb/pf', 'mb/tc', 'mb/ddf', 'pf/tc', 'pf/ddf', 'tc/ddf']
        combination_dict = {'mb/pf':'$\mathregular{B}$ / $\mathregular{k_{p}}$',
                            'mb/tc':'$\mathregular{B}$ / $\mathregular{T_{bias}}$',
                            'mb/ddf':'$\mathregular{B}$ / $\mathregular{f_{snow}}$',
                            'pf/tc':'$\mathregular{k_{p}}$ / $\mathregular{T_{bias}}$',
                            'pf/ddf':'$\mathregular{k_{p}}$ / $\mathregular{f_{snow}}$',
                            'tc/ddf':'$\mathregular{T_{bias}}$ / $\mathregular{f_{snow}}$'}
        
        bdict = {}
        bdict['mb/pf'] = np.arange(-0.2, 1.05, 0.05) - 0.025
        bdict['mb/tc'] = np.arange(-1, 0.2, 0.05) - 0.025
        bdict['mb/ddf'] = np.arange(-1, 0.2, 0.05) - 0.025
        bdict['pf/tc'] = np.arange(-0.2, 1.05, 0.05) - 0.025
        bdict['pf/ddf'] = np.arange(-0.2, 1.05, 0.05) - 0.025
        bdict['tc/ddf'] = np.arange(-1, 0.2, 0.05) - 0.025
        
        tdict = {}
        tdict['mb/pf'] = np.arange(0, 1.05, 0.5)
        tdict['mb/tc'] = np.arange(-1, 0.05, 0.5)
        tdict['mb/ddf'] = np.arange(-1, 0.05, 0.5)
        tdict['pf/tc'] = np.arange(0, 1.05, 0.5)
        tdict['pf/ddf'] = np.arange(0, 1.05, 0.5)
        tdict['tc/ddf'] = np.arange(-1, 0.05, 0.5)
           
        nrows = 2
        ncols = 3 
        fig, ax = plt.subplots(nrows, ncols, squeeze=False, gridspec_kw={'wspace':0.5, 'hspace':0.4})
        nrow = 0
        ncol = 0
        for nvar, combination in enumerate(combinations):
                    
            # Plot histogram
            hist, bins = np.histogram(df_all[combination], bins=bdict[combination])
            hist = hist * 100.0 / hist.sum()
            bins_centered = bins[1:] + (bins[0] - bins[1]) / 2
            ax[nrow,ncol].bar(x=bins_centered, height=hist, width=(bins[1]-bins[0]), align='center',
                              edgecolor='black', color='lightgrey', linewidth=0.2)    
            ax[nrow,ncol].set_ylim(0,18)  
            ax[nrow,ncol].set_xticks(tdict[combination])          
            ax[nrow,ncol].text(0.5, 1.01, combination_dict[combination], size=12, horizontalalignment='center', 
                               verticalalignment='bottom', transform=ax[nrow,ncol].transAxes)
            r_mean = df_all[combination].mean()
            ax[nrow,ncol].text(0.98, 0.98, np.round(r_mean,2), size=12, horizontalalignment='right', 
                               verticalalignment='top', transform=ax[nrow,ncol].transAxes)
            
            # Adjust row and column
            ncol += 1
            if ncol == ncols:
                nrow += 1
                ncol = 0
        
        fig.text(0.04, 0.5, 'Count (%)', va='center', rotation='vertical', size=12)
        fig.text(0.5,0, 'Correlation Coefficient (R)', size=12, horizontalalignment='center')
                    
        if os.path.exists(fig_fp) == False:
            os.makedirs(fig_fp)    
        fig.set_size_inches(6, 4)
        figure_fn = 'correlation_scatter_' + str(burn) + 'burn.png'
        fig.savefig(fig_fp + figure_fn, bbox_inches='tight', dpi=300)   
            
            
    #%%
    #cal_fp = input.main_directory + '/../Output/cal_opt2_spc_20190806/'
    #
    #x_list = []
    #for i in os.listdir(cal_fp):
    #    if i.endswith('.nc'):
    #        x_list.append(i)
    #        
    #x_list = np.array(sorted(x_list))
    #y = [int(x.split('.')[1]) for x in x_list]
    #z = y - np.roll(y,1)
    #A = np.where(z!=1)[0]
    #B = [x_list[x-1] for x in A]
    #print(B)
    #            

