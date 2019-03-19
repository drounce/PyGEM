""" Analyze MCMC output - chain length, etc. """

# Built-in libraries
import glob
import os
import pickle
# External libraries
import cartopy
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd
import pymc
from scipy import stats
from scipy.stats.kde import gaussian_kde
from scipy.stats import norm
from scipy.stats import truncnorm
from scipy.stats import uniform
#from scipy.stats import linregress
from scipy.stats import lognorm
from scipy.optimize import minimize
import xarray as xr
# Local libraries
import class_climate
import class_mbdata
import pygem_input as input
import pygemfxns_massbalance as massbalance
import pygemfxns_modelsetup as modelsetup
import run_calibration as calibration

#%%
option_metrics_vs_chainlength = 0
option_metrics_histogram_all = 0
option_observation_vs_calibration = 0
option_prior_vs_posterior_single = 0

# Paper figures
option_papermcmc_prior_vs_posterior = 0
option_papermcmc_solutionspace = 0
option_papermcmc_allglaciers_posteriorchanges = 0
option_papermcmc_spatialdistribution_parameter = 1


variables = ['massbal', 'precfactor', 'tempchange', 'ddfsnow']  
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
#vn_label_dict = {'massbal':'[mwea]',
#                 'precfactor':'[-]',
#                 'tempchange':'[degC]',
#                 'ddfsnow':'[mwe $degC^{-1} d^{-1}$]'}
metric_title_dict = {'Gelman-Rubin':'Gelman-Rubin Statistic',
                     'MC Error': 'Monte Carlo Error',
                     'Effective N': 'Effective Sample Size'}
metrics = ['Gelman-Rubin', 'MC Error', 'Effective N']


# Export option
mcmc_output_netcdf_fp_3chain = input.output_filepath + 'cal_opt2_spc_3000glac_3chain_adj12_wpriors/'
mcmc_output_netcdf_fp_all = input.output_filepath + 'cal_opt2_spc_20190308_adjp12/cal_opt2/'
mcmc_output_figures_fp = input.output_filepath + 'figures/'
#mcmc_output_csv_fp = mcmc_output_netcdf_fp + 'csv/'

regions = ['13', '14', '15']

cal_datasets = ['shean']

burn=0

chainlength = 10000
# Bounds (90% bounds --> 95% above/below given threshold)
low_percentile = 5
high_percentile = 95

colors = ['#387ea0', '#fcb200', '#d20048']
linestyles = ['-', '--', ':']

# Group dictionaries
watershed_dict_fn = '/Users/davidrounce/Documents/Dave_Rounce/HiMAT/qgis_himat/rgi60_HMA_dict_watershed.csv'
watershed_csv = pd.read_csv(watershed_dict_fn)
watershed_dict = dict(zip(watershed_csv.RGIId, watershed_csv.watershed))
kaab_dict_fn = '/Users/davidrounce/Documents/Dave_Rounce/HiMAT/qgis_himat/rgi60_HMA_dict_kaab.csv'
kaab_csv = pd.read_csv(kaab_dict_fn)
kaab_dict = dict(zip(kaab_csv.RGIId, kaab_csv.kaab_name))

# Shapefiles
rgiO1_shp_fn = '/Users/davidrounce/Documents/Dave_Rounce/HiMAT/RGI/rgi60/00_rgi60_regions/00_rgi60_O1Regions.shp'
watershed_shp_fn = '/Users/davidrounce/Documents/Dave_Rounce/HiMAT/qgis_himat/HMA_basins_20181018_4plot.shp'
kaab_shp_fn = '/Users/davidrounce/Documents/Dave_Rounce/HiMAT/qgis_himat/kaab2015_regions.shp'
srtm_fn = '/Users/davidrounce/Documents/Dave_Rounce/HiMAT/qgis_himat/SRTM_HMA.tif'
srtm_contour_fn = '/Users/davidrounce/Documents/Dave_Rounce/HiMAT/qgis_himat/SRTM_HMA_countours_2km_gt3000m_smooth.shp'


def load_glacierdata_byglacno(glac_no, option_loadhyps_climate=1):
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
                main_glac_rgi_region, [region], input.hyps_filepath,input.hyps_filedict, input.hyps_colsdrop)
        # ===== CALIBRATION DATA =====
        cal_data_region = pd.DataFrame()
        for dataset in cal_datasets:
            cal_subset = class_mbdata.MBData(name=dataset, rgi_regionO1=region)
            cal_subset_data = cal_subset.retrieve_mb(main_glac_rgi_region, main_glac_hyps_region, dates_table_nospinup)
            cal_data_region = cal_data_region.append(cal_subset_data, ignore_index=True)
        cal_data_region = cal_data_region.sort_values(['glacno', 't1_idx'])
        cal_data_region.reset_index(drop=True, inplace=True)
        
        # ===== OTHER DATA =====
        if option_loadhyps_climate == 1:
            # Ice thickness [m], average
            main_glac_icethickness_region = modelsetup.import_Husstable(
                    main_glac_rgi_region, [region], input.thickness_filepath, input.thickness_filedict, 
                    input.thickness_colsdrop)
            main_glac_hyps_region[main_glac_icethickness_region == 0] = 0
            # Width [km], average
            main_glac_width_region = modelsetup.import_Husstable(
                    main_glac_rgi_region, [region], input.width_filepath, input.width_filedict, input.width_colsdrop)
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
    cal_data.reset_index(inplace=True, drop=True)
    
    if option_loadhyps_climate == 1:
        main_glac_hyps.reset_index(inplace=True, drop=True)
        main_glac_icethickness.reset_index(inplace=True, drop=True)
        main_glac_width.reset_index(inplace=True, drop=True)
    
    if option_loadhyps_climate == 0:
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
        calculation used to compute region value (mean or sum)
        
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

        # Regional calc
        if regional_calc == 'mean':           
            vn_reg = vn_glac.mean(axis=0)
        elif regional_calc == 'sum':
            vn_reg = vn_glac.sum(axis=0)      
        
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
                          observed_error, tempchange_boundhigh, tempchange_boundlow, tempchange_opt_init, mb_max_acc, 
                          mb_max_loss, tempchange_max_acc, tempchange_max_loss, option_areaconstant=0, 
                          option_plotsteps=1, fig_fp=input.output_filepath):
    """
    Plot the mass balance [mwea] versus all model parameters to see how parameters effect mass balance
    """
    mb_vs_parameters = pd.DataFrame(np.zeros((len(ddfsnow_iters) * len(precfactor_iters) * len(tempchange_iters), 4)),
                                    columns=['precfactor', 'tempbias', 'ddfsnow', 'massbal'])
    count=0
    for n, precfactor in enumerate(precfactor_iters):
        modelparameters[2] = precfactor
        
        for n, tempchange in enumerate(tempchange_iters):
            modelparameters[7] = tempchange

            for c, ddfsnow in enumerate(ddfsnow_iters):
                
                modelparameters[4] = ddfsnow
                modelparameters[5] = modelparameters[4] / input.ddfsnow_iceratio
                
                # run mass balance calculation
#                if modelparameters[2] == 1:
#                    option_areaconstant = 0
#                else:
#                    option_areaconstant = 1
                option_areaconstant = 0
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
            print(modelparameters[2], modelparameters[7], modelparameters[4], np.round(mb_mwea,3))

    # Subset data for each precfactor
    linestyles = ['-', '--', ':', '-.']
    linecolors = ['b', 'k', 'r']
    prec_linedict = {precfactor : linestyles[n] for n, precfactor in enumerate(precfactor_iters)} 
    ddfsnow_colordict = {ddfsnow : linecolors[n] for n, ddfsnow in enumerate(ddfsnow_iters)} 
    
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
    ax.axhline(observed_massbal, color='gray', linewidth=2)    
    observed_mb_min = observed_massbal - 3*observed_error
    observed_mb_max = observed_massbal + 3*observed_error  
    ax.fill_between([np.min(tempchange_iters), np.max(tempchange_iters)], observed_mb_min, observed_mb_max, 
                    facecolor='gray', alpha=0.15, label=None)
    
    if option_plotsteps == 1:
        marker_size = 20
        markeredge_color = 'black'
        marker_color = 'black'
        ax.plot([tempchange_boundhigh], [mb_max_loss], marker='*', markersize=marker_size, 
                markeredgecolor=markeredge_color, color=marker_color)
        ax.plot([tempchange_boundlow], [mb_max_loss + 0.9*(mb_max_acc - mb_max_loss)], marker='*', markersize=marker_size, 
                markeredgecolor=markeredge_color, color=marker_color)
        ax.plot([tempchange_opt_init], [observed_massbal], marker='*', markersize=marker_size, 
                markeredgecolor=markeredge_color, color=marker_color)
        ax.plot([tempchange_opt_init + tempchange_sigma*3], [observed_mb_min], marker='*', markersize=marker_size, 
                markeredgecolor=markeredge_color, color=marker_color)
        ax.plot([tempchange_opt_init - tempchange_sigma*3], [observed_mb_max], marker='*', markersize=marker_size, 
                markeredgecolor=markeredge_color, color=marker_color)
        ax.plot([tempchange_opt_init - tempchange_sigma], [observed_mb_max], marker='*', markersize=marker_size, 
                markeredgecolor=markeredge_color, color=marker_color)
        ax.plot([tempchange_opt_init + tempchange_sigma], [observed_mb_min], marker='*', markersize=marker_size, 
                markeredgecolor=markeredge_color, color=marker_color)
        ax.plot([tempchange_mu], [observed_massbal], marker='*', markersize=marker_size, 
                markeredgecolor=markeredge_color, color=marker_color)
    
    #    ax.text(tempchange_boundhigh, mb_max_loss, '1', fontsize=20)
     
    ax.set_xlim(np.min(tempchange_iters), np.max(tempchange_iters))
    if observed_massbal - 3*observed_error < mb_max_loss:
        ylim_lower = observed_massbal - 3*observed_error
    else:
        ylim_lower = np.floor(mb_max_loss)
    ax.set_ylim(ylim_lower,np.ceil(mb_vs_parameters['massbal'].max()))
    ax.set_ylim(-2,2)
    
    # Labels
    ax.set_title('Mass balance versus Parameters ' + glacier_str)
    ax.set_xlabel('Temperature Bias [$^\circ$C]', fontsize=14)
    ax.set_ylabel('Mass Balance [mwea]', fontsize=14)
    
    # Add legend
    leg_lines = []
    leg_names = []
    x_min = mb_vs_parameters.loc[:,'tempbias'].min()
    y_min = mb_vs_parameters.loc[:,'massbal'].min()
    for precfactor in reversed(precfactor_iters):
        line = Line2D([x_min,y_min],[x_min,y_min], linestyle=prec_linedict[precfactor], color='gray')
        leg_lines.append(line)
        leg_names.append('PF ' + str(precfactor))
    for ddfsnow in ddfsnow_iters:
        line = Line2D([x_min,y_min],[x_min,y_min], linestyle='-', color=ddfsnow_colordict[ddfsnow])
        leg_lines.append(line)
        leg_names.append('DDF ' + str(np.round(ddfsnow*10**3,1)))
        
        
    ax.legend(leg_lines, leg_names, loc='upper right', frameon=False)
    fig.savefig(fig_fp + glacier_str + '_mb_vs_parameters_areachg.png', 
                bbox_inches='tight', dpi=300)    
    

# ===== PLOT OPTIONS ==================================================================================================
def metrics_vs_chainlength(netcdf_fp, regions, iters, burn=0, nchain=3):
    """
    Plot Gelman-Rubin, Monte Carlo error, and effective sample size for each parameter for various chain lengths

    Parameters
    ----------
    regions : list of strings
        list of regions
    iters : list of ints
        list of the number of iterations to compute metrics for
    burn : int
        burn-in number

    Returns
    -------
    .png file
        saves figure of how metrics change according to the number of mcmc iterations
    .pkl files
        saves .pkl files of the metrics for various iterations (if they don't already exist)
    """
##%%
#for batman in [0]:
#    netcdf_fp = mcmc_output_netcdf_fp_3chain
#    nchain = 3
#    iters = iterations
    
    # Load netcdf filenames    
    filelist = []
    for region in regions:
        filelist.extend(glob.glob(netcdf_fp + str(region) + '*.nc'))  
    
    # ===== LOAD OR CALCULATE METRICS =====
    fig_fp = netcdf_fp + 'figures/'
    csv_fp = netcdf_fp + 'csv/'
    en_fn_pkl = csv_fp + 'effective_n_list.pkl'
    mc_fn_pkl = csv_fp + 'mc_error_list.pkl'
    gr_fn_pkl = csv_fp + 'gelman_rubin_list.pkl'
    glacno_fn_pkl = csv_fp + 'glacno_list.pkl'
    
    # Check if list already exists
    iter_ending = '_' + str(iterstep) + 'iterstep.pkl'
    en_fn_pkl.replace('.pkl', iter_ending)
    
    if os.path.isfile(en_fn_pkl.replace('.pkl', iter_ending)):
        with open(en_fn_pkl.replace('.pkl', iter_ending), 'rb') as f:
            en_list = pickle.load(f)
        with open(mc_fn_pkl.replace('.pkl', iter_ending), 'rb') as f:
            mc_list = pickle.load(f)
        if nchain > 1:
            with open(gr_fn_pkl.replace('.pkl', iter_ending), 'rb') as f:
                gr_list = pickle.load(f)
        with open(glacno_fn_pkl, 'rb') as f:
            glac_no = pickle.load(f)
    else:
        # Lists to record metrics
        glac_no = []
        en_list = {}
        gr_list = {}
        mc_list = {}
        
        # iterate through each glacier
        count = 0
        for netcdf in filelist:
            glac_str = netcdf.split('/')[-1].split('.nc')[0]
            glac_no.append(glac_str)
            count += 1
            if count%250 == 0:
                print(count, glac_str)
    
            en_list[glac_str] = {}
            gr_list[glac_str] = {}
            mc_list[glac_str] = {}
            # open dataset
            ds = xr.open_dataset(netcdf)

            # Metrics for each parameter
            for nvar, vn in enumerate(variables):
                # Effective sample size
                en = [effective_n(ds, vn=vn, iters=i, burn=burn) for i in iters]                
                en_list[glac_str][vn] = dict(zip(iters, en))
                
                # Monte Carlo error
                # the first [0] extracts the MC error as opposed to the confidence interval
                # the second [0] extracts the first chain
                mc = [mc_error(ds, vn=vn, iters=i, burn=burn, method='overlapping')[0][0] for i in iters]
                mc_list[glac_str][vn] = dict(zip(iters, mc))

                # Gelman-Rubin Statistic                
                if len(ds.chain) > 1:
                    gr = [gelman_rubin(ds, vn=vn, iters=i, burn=burn) for i in iters]
                    gr_list[glac_str][vn] = dict(zip(iters, gr))
    
            # close datase
            ds.close()
        
            
        # Pickle lists for next time
        if os.path.exists(csv_fp) == False:
            os.makedirs(csv_fp)
                
        pickle_data(en_fn_pkl.replace('.pkl', iter_ending), en_list)
        pickle_data(mc_fn_pkl.replace('.pkl', iter_ending), mc_list)
        if len(ds.chain) > 1:
            pickle_data(gr_fn_pkl.replace('.pkl', iter_ending), gr_list)
        pickle_data(glacno_fn_pkl, glac_no)
        
    # ===== PLOT METRICS =====
    colors = ['#387ea0', '#fcb200', '#d20048']
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
            metric_list = mc_list
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
                
        
            # ===== Plot =====
            if vn == 'ddfsnow' and metric == 'MC Error':
                ax[nvar,nmetric].plot(metric_df['iters']/10**3, metric_df['median']*10**3, color=colors[nmetric])
                ax[nvar,nmetric].fill_between(metric_df['iters']/10**3, metric_df['lowbnd']*10**3, metric_df['highbnd']*10**3, 
                                              color=colors[nmetric], alpha=0.5)
            else:
                ax[nvar,nmetric].plot(metric_df['iters']/10**3, metric_df['median'], color=colors[nmetric])
                ax[nvar,nmetric].fill_between(metric_df['iters']/10**3, metric_df['lowbnd'], metric_df['highbnd'], 
                                              color=colors[nmetric], alpha=0.5)
            
            # niceties
            ax[nvar,nmetric].xaxis.set_major_locator(MultipleLocator(10))
            ax[nvar,nmetric].xaxis.set_minor_locator(MultipleLocator(2))
            if nvar == 0:
                ax[nvar,nmetric].set_title(metric_title_dict[metric], fontsize=12)
            elif nvar == len(variables) - 1:
                ax[nvar,nmetric].set_xlabel('Steps [$10^3$]', fontsize=12)
            
                
            if metric == 'Gelman-Rubin':
                ax[nvar,nmetric].set_ylabel(vn_title_dict[vn], fontsize=12, labelpad=10)
                ax[nvar,nmetric].set_ylim(1,1.12)
                ax[nvar,nmetric].axhline(y=1.1, color='k', linestyle='--', linewidth=2)
                ax[nvar,nmetric].yaxis.set_major_locator(MultipleLocator(0.05))
                ax[nvar,nmetric].yaxis.set_minor_locator(MultipleLocator(0.01))
            elif metric == 'MC Error':
                if vn == 'massbal':
                    ax[nvar,nmetric].axhline(y=0.005, color='k', linestyle='--', linewidth=2)
                    ax[nvar,nmetric].set_ylim(0,0.012)
                    ax[nvar,nmetric].yaxis.set_major_locator(MultipleLocator(0.005))
                    ax[nvar,nmetric].yaxis.set_minor_locator(MultipleLocator(0.001))
                elif vn == 'precfactor':
                    ax[nvar,nmetric].axhline(y=0.05, color='k', linestyle='--', linewidth=2)
                    ax[nvar,nmetric].set_ylim(0,0.12)
                    ax[nvar,nmetric].yaxis.set_major_locator(MultipleLocator(0.05))
                    ax[nvar,nmetric].yaxis.set_minor_locator(MultipleLocator(0.01))
                elif vn == 'tempchange':
                    ax[nvar,nmetric].axhline(y=0.05, color='k', linestyle='--', linewidth=2)
                    ax[nvar,nmetric].set_ylim(0,0.12)
                    ax[nvar,nmetric].yaxis.set_major_locator(MultipleLocator(0.05))
                    ax[nvar,nmetric].yaxis.set_minor_locator(MultipleLocator(0.01))
                elif vn == 'ddfsnow':
                    ax[nvar,nmetric].axhline(y=0.05, color='k', linestyle='--', linewidth=2)
                    ax[nvar,nmetric].set_ylim(0,0.12)
                    ax[nvar,nmetric].yaxis.set_major_locator(MultipleLocator(0.05))
                    ax[nvar,nmetric].yaxis.set_minor_locator(MultipleLocator(0.01))
            elif metric == 'Effective N':
                ax[nvar,nmetric].set_ylim(0,1200)
                ax[nvar,nmetric].axhline(y=100, color='k', linestyle='--', linewidth=2)
                ax[nvar,nmetric].yaxis.set_major_locator(MultipleLocator(500))
                ax[nvar,nmetric].yaxis.set_minor_locator(MultipleLocator(100))
                
    # Save figure
    fig.set_size_inches(figwidth,figheight)
    if os.path.exists(fig_fp) == False:
        os.makedirs(fig_fp)
    figure_fn = 'chainlength_vs_metrics.png'
    fig.savefig(fig_fp + figure_fn, bbox_inches='tight', dpi=300)

    
#%%        
def observation_vs_calibration(regions, netcdf_fp, chainlength=chainlength, burn=0, chain_no=0):
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
#    netcdf_fp = mcmc_output_netcdf_fp_3chain
#    chain_no = 0
    
    csv_fp = netcdf_fp + 'csv/'
    
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
    
    
    #%%
    # Mass balance comparison: observations and model
    mb_compare_cols = ['glacno', 'obs_mwea', 'obs_mwea_std', 'mod_mwea', 'mod_mwea_std', 'dif_mwea']
    mb_compare = pd.DataFrame(np.zeros((len(glac_no), len(mb_compare_cols))), columns=mb_compare_cols)
    mb_compare['glacno'] = glac_no
    mb_compare['obs_mwea'] = cal_data['mb_mwe'] / (cal_data['t2'] - cal_data['t1'])
    mb_compare['obs_mwea_std'] = cal_data['mb_mwe_err'] / (cal_data['t2'] - cal_data['t1'])
    for nglac, glac in enumerate(glac_no):
        # open dataset
        if nglac%500 == 0:
            print(nglac, glac)
        ds = xr.open_dataset(netcdf_fp + glac + '.nc')
        mb_all = ds['mp_value'].sel(chain=chain_no, mp='massbal').values[burn:chainlength]
        mb_compare.loc[nglac, 'mod_mwea'] = np.mean(mb_all)
        mb_compare.loc[nglac, 'mod_mwea_std'] = np.std(mb_all)
        # close dataset
        ds.close()

    # export csv
    if os.path.exists(csv_fp) == False:
        os.makedirs(csv_fp)   
    mb_compare['dif_mwea'] = mb_compare['obs_mwea'] - mb_compare['mod_mwea']
    mb_compare.to_csv(csv_fp + 'mb_compare_' + str(int(chainlength/1000)) + 'k.csv')

    # plot histogram
    dif_bins = [-1,-0.2, -0.1, -0.05,-0.02, 0.02, 0.05, 0.1, 0.2, 1]
    bin_min = np.floor((mb_compare['dif_mwea'].min() * 100))/100
    bin_max = np.ceil((mb_compare['dif_mwea'].max() * 100))/100
    if bin_min < dif_bins[0]:
        dif_bins[0] = bin_min
    if bin_max > dif_bins[-1]:
        dif_bins[-1] = bin_max
    hist_fn = 'hist_' + str(int(chainlength/1000)) + 'kch_dif_mwea.png'
    plot_hist(mb_compare, 'dif_mwea', dif_bins, fig_fn=hist_fn)
    #%%
    

def prior_vs_posterior_single(glac_no, netcdf_fp, iters=[1000,15000], precfactor_disttype=input.precfactor_disttype, 
                              tempchange_disttype = input.tempchange_disttype, 
                              ddfsnow_disttype = input.ddfsnow_disttype):
    """ Plot prior vs posterior of individual glacier for different chain lengths
    
    Parameters
    ----------
    iters : list of ints
        list of chain lengths for compare posteriors of
    glac_no : str
        glacier number including region (ex. '15.00001')

    Returns
    -------
    .png files
        saves figure showing how prior and posterior comparison
    """    
    region = [int(glac_no.split('.')[0])]
    rgi_glac_number = [glac_no.split('.')[1]]

    # Glacier RGI data
    main_glac_rgi = modelsetup.selectglaciersrgitable(rgi_regionsO1=region, rgi_regionsO2 = 'all',
                                                      rgi_glac_number=rgi_glac_number)
    # Glacier hypsometry [km**2], total area
    main_glac_hyps = modelsetup.import_Husstable(main_glac_rgi, region, input.hyps_filepath,
                                                 input.hyps_filedict, input.hyps_colsdrop)
    # Ice thickness [m], average
    main_glac_icethickness = modelsetup.import_Husstable(main_glac_rgi, region, input.thickness_filepath, 
                                                         input.thickness_filedict, input.thickness_colsdrop)
    main_glac_hyps[main_glac_icethickness == 0] = 0
    # Width [km], average
    main_glac_width = modelsetup.import_Husstable(main_glac_rgi, region, input.width_filepath,
                                                  input.width_filedict, input.width_colsdrop)
    # Elevation bins
    elev_bins = main_glac_hyps.columns.values.astype(int)   
    # Select dates including future projections
    dates_table = modelsetup.datesmodelrun(startyear=input.startyear, endyear=input.endyear, 
                                           spinupyears=input.spinupyears)
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
        df = pd.DataFrame(ds['mp_value'].values[:,:,0], columns=ds.mp.values)  
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
        
        (precfactor_boundlow, precfactor_boundhigh, precfactor_mu, precfactor_start, tempchange_boundlow, 
         tempchange_boundhigh, tempchange_mu, tempchange_sigma, tempchange_start, tempchange_max_loss, 
         tempchange_max_acc, mb_max_loss, mb_max_acc, precfactor_opt_init, tempchange_opt_init) = (
                 calibration.retrieve_prior_parameters(
                         modelparameters, glacier_rgi_table, glacier_area_t0, icethickness_t0, 
                                           width_t0, elev_bins, glacier_gcm_temp, glacier_gcm_prec, glacier_gcm_elev, 
                                           glacier_gcm_lrgcm, glacier_gcm_lrglac, dates_table, t1_idx, t2_idx, t1, t2, 
                                           observed_massbal, mb_obs_min, mb_obs_max))
        
        ddfsnow_mu = input.ddfsnow_mu
        ddfsnow_sigma = input.ddfsnow_sigma
        ddfsnow_boundlow = input.ddfsnow_boundlow
        ddfsnow_boundhigh = input.ddfsnow_boundhigh
    
        print('\nParameters:\nPF_low:', np.round(precfactor_boundlow,2), 'PF_high:', 
              np.round(precfactor_boundhigh,2), '\nTC_low:', np.round(tempchange_boundlow,2), 
              'TC_high:', np.round(tempchange_boundhigh,2),
              '\nTC_mu:', np.round(tempchange_mu,2), 'TC_sigma:', np.round(tempchange_sigma,2))
       
        #%%
    # PRIOR VS POSTERIOR PLOTS 
    fig, ax = plt.subplots(2, 2, squeeze=False, figsize=(6.5, 5), 
                           gridspec_kw={'wspace':0.3, 'hspace':0.5})
    
    param_idx_dict = {'massbal':[0,0],
                      'precfactor':[0,1],
                      'tempchange':[1,0],
                      'ddfsnow':[1,1]}
    
    z_score = np.linspace(norm.ppf(0.01), norm.ppf(0.99), 100)
    for nvar, vn in enumerate(variables):
        
        # PRIOR DISTRIBUTIONS
        if vn == 'massbal':
            x_values = observed_massbal + observed_error * z_score
            y_values = norm.pdf(x_values, loc=observed_massbal, scale=observed_error)
        elif vn == 'precfactor': 
            if precfactor_disttype == 'uniform':
                z_score = np.linspace(uniform.ppf(0.01), uniform.ppf(0.99), 100)
                x_values = precfactor_boundlow + z_score * (precfactor_boundhigh - precfactor_boundlow)
                y_values = uniform.pdf(x_values, loc=precfactor_boundlow, 
                                       scale=(precfactor_boundhigh - precfactor_boundlow))
            elif precfactor_disttype == 'lognormal':
                precfactor_lognorm_sigma = (1/input.precfactor_lognorm_tau)**0.5
                x_values = np.linspace(lognorm.ppf(1e-6, precfactor_lognorm_sigma), 
                                       lognorm.ppf(0.99, precfactor_lognorm_sigma), 100)
                y_values = lognorm.pdf(x_values, precfactor_lognorm_sigma)
        elif vn == 'tempchange':
            if tempchange_disttype == 'uniform':
                z_score = np.linspace(uniform.ppf(0.01), uniform.ppf(0.99), 100)
                x_values = tempchange_boundlow + z_score * (tempchange_boundhigh - tempchange_boundlow)
                y_values = uniform.pdf(x_values, loc=tempchange_boundlow,
                                       scale=(tempchange_boundhigh - tempchange_boundlow))
            elif tempchange_disttype == 'truncnormal':
                tempchange_a = (tempchange_boundlow - tempchange_mu) / tempchange_sigma
                tempchange_b = (tempchange_boundhigh - tempchange_mu) / tempchange_sigma
                z_score = np.linspace(truncnorm.ppf(0.01, tempchange_a, tempchange_b),
                                      truncnorm.ppf(0.99, tempchange_a, tempchange_b), 100)
                x_values = tempchange_mu + tempchange_sigma * z_score
                y_values = truncnorm.pdf(x_values, tempchange_a, tempchange_b, loc=tempchange_mu,
                                         scale=tempchange_sigma)
        elif vn == 'ddfsnow':            
            if ddfsnow_disttype == 'truncnormal':
                ddfsnow_a = (ddfsnow_boundlow - ddfsnow_mu) / ddfsnow_sigma
                ddfsnow_b = (ddfsnow_boundhigh - ddfsnow_mu) / ddfsnow_sigma
                z_score = np.linspace(truncnorm.ppf(0.01, ddfsnow_a, ddfsnow_b),
                                      truncnorm.ppf(0.99, ddfsnow_a, ddfsnow_b), 100)
                x_values = ddfsnow_mu + ddfsnow_sigma * z_score
                y_values = truncnorm.pdf(x_values, ddfsnow_a, ddfsnow_b, loc=ddfsnow_mu, scale=ddfsnow_sigma)
            elif ddfsnow_disttype == 'uniform':
                z_score = np.linspace(uniform.ppf(0.01), uniform.ppf(0.99), 100)
                x_values = ddfsnow_boundlow + z_score * (ddfsnow_boundhigh - ddfsnow_boundlow)
                y_values = uniform.pdf(x_values, loc=ddfsnow_boundlow,
                                       scale=(ddfsnow_boundhigh - ddfsnow_boundlow))
        
        nrow = param_idx_dict[vn][0]
        ncol = param_idx_dict[vn][1]
        ax[nrow,ncol].plot(x_values, y_values, color='k')
        
        # Labels
        ax[nrow,ncol].set_xlabel(vn_label_dict[vn], size=12)
        if ncol == 0:
            ax[nrow,ncol].set_ylabel('Probability Density', size=12)

        # Ensemble/Posterior distribution                
        for n_chain in range(len(ds.chain.values)):
            for count_iter, n_iters in enumerate(iters):
                chain = ds['mp_value'].sel(chain=n_chain, mp=vn).values[burn:n_iters]
            
                # gaussian distribution
                kde = gaussian_kde(chain)
                x_values_kde = x_values.copy()
                y_values_kde = kde(x_values_kde)
                
                # Plot fitted distribution
                ax[nrow,ncol].plot(x_values_kde, y_values_kde, color=colors[count_iter], linestyle=linestyles[n_chain])
    
    # Close dataset
    ds.close()
    
    # Legend (Note: hard code the spacing between the two legends)
    leg_lines = []
    line = Line2D([0,1],[0,1], color='white')
    leg_lines.append(line)
    leg_labels = ['Steps']
    
    for count_iter, n_iters in enumerate(iters):
        line = Line2D([0,1],[0,1], color=colors[count_iter])
        leg_lines.append(line)
    iter_labels = [str(int(i)) for i in iters]
    for label in iter_labels:
        leg_labels.append(label)
    
    for n in range(8):
        line = Line2D([0,1],[0,1], color='white')
        leg_lines.append(line)
        leg_labels.append(' ')    
    line = Line2D([0,1],[0,1], color='white')
    leg_lines.append(line)
    leg_labels.append('Starting Point')
    
    for n_chain in range(len(ds.chain.values)):
        line = Line2D([0,1],[0,1], color='gray', linestyle=linestyles[n_chain])
        leg_lines.append(line)
    chain_labels = ['Center', 'Lower Bound', 'Upper Bound  ']
    for n in chain_labels:
        leg_labels.append(n)
    
    fig.legend(leg_lines, leg_labels, loc='upper right', bbox_to_anchor=(1.1,0.85), 
               handlelength=1, handletextpad=0.5, borderpad=0.3, frameon=False)
        
    # Save figure
    str_ending = ''
    if 'tempchange' in variables:    
        if tempchange_disttype == 'truncnormal': 
            str_ending += '_TCtn'
        elif tempchange_disttype == 'uniform':
            str_ending += '_TCu'
    if 'precfactor' in variables:                
        if precfactor_disttype == 'lognormal': 
            str_ending += '_PFln'
        elif precfactor_disttype == 'uniform':
            str_ending += '_PFu'
    if 'ddfsnow' in variables:     
        if ddfsnow_disttype == 'truncnormal': 
            str_ending += '_DDFtn'
        elif ddfsnow_disttype == 'uniform':
            str_ending += '_DDFu'        
    if input.tempchange_edge_method == 'mb':
        str_ending += '_edgeMBpt' + str(int(input.tempchange_edge_mb*100)).zfill(2)
    elif input.tempchange_edge_method == 'mb_norm':
        str_ending += '_edgeMBNormpt' + str(int(input.tempchange_edge_mbnorm*100)).zfill(2)
    elif input.tempchange_edge_method == 'mb_norm_slope':
        str_ending += '_edgeSpt' + str(int(input.tempchange_edge_mbnormslope*100)).zfill(2)
    str_ending += '_TCadjp' + str(int(input.tempchange_mu_adj*100)).zfill(2)
        
    if os.path.exists(mcmc_output_figures_fp) == False:
        os.makedirs(mcmc_output_figures_fp)        
    fig.savefig(mcmc_output_figures_fp + 'prior_v_posteriors_' + glacier_str + str_ending + '.png', 
                bbox_inches='tight', dpi=300)
    fig.clf()
 

#%%         
if option_metrics_vs_chainlength == 1:
    # 3 chain metrics
    print('3 CHAIN METRICS')
    iterstep = 1000
    itermax = 25000
    iterations = np.arange(0, itermax, iterstep)
    if iterations[1] < 1000: 
        iterations[0] = 1000
    else:
        iterations = iterations[1:]
    if iterations[-1] != itermax:
        iterations = np.append(iterations, itermax)
    metrics_vs_chainlength(mcmc_output_netcdf_fp_3chain, regions, iterations, burn=burn, nchain=3) 
    

if option_metrics_histogram_all == 1:
#    metrics_vs_chainlength(mcmc_output_netcdf_fp_3chain, regions, iterations, burn=burn, nchain=3) 
    print('code this plot!')


if option_prior_vs_posterior_single == 1:
    glac_no = ['13.26360']
    iters=[1000,10000]
#    main_glac_rgi, cal_data, glac_no = load_glacier_and_cal_data(regions)
    for nglac, glac in enumerate(glac_no):
#        if main_glac_rgi.loc[nglac,'Area'] > 20:
#            print(main_glac_rgi.loc[nglac,'RGIId'], glac)
        prior_vs_posterior_single(glac, iters=iters)
        
        
        
if option_observation_vs_calibration == 1:
    observation_vs_calibration(regions, mcmc_output_netcdf_fp_3chain, chainlength=chainlength, burn=burn)
        
        
#%%
if option_papermcmc_prior_vs_posterior == 1:
    print('Prior vs posterior showing two example glaciers side-by-side!')
    glac_no = ['13.26360', '14.08487']
    netcdf_fp = mcmc_output_netcdf_fp_3chain
    fig_fp = netcdf_fp + 'figures/'
    if os.path.exists(fig_fp) == False:
        os.makedirs(fig_fp)
    iters=[1000,15000]
    
    main_glac_rgi, cal_data = load_glacierdata_byglacno(glac_no, option_loadhyps_climate=0)

    # PRIOR VS POSTERIOR PLOTS 
    fig, ax = plt.subplots(4, 2, squeeze=False, figsize=(6.5, 8), 
                           gridspec_kw={'wspace':0.2, 'hspace':0.6})

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
        
        param_idx_dict = {'massbal':[0,n],
                          'precfactor':[1,n],
                          'tempchange':[2,n],
                          'ddfsnow':[3,n]}
        
        z_score = np.linspace(norm.ppf(0.01), norm.ppf(0.99), 100)
        for nvar, vn in enumerate(variables):
        
            # PRIOR DISTRIBUTIONS
            if vn == 'massbal':
                x_values = observed_massbal + observed_error * z_score
                y_values = norm.pdf(x_values, loc=observed_massbal, scale=observed_error)
            elif vn == 'precfactor': 
                if input.precfactor_disttype == 'uniform':
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
                    z_score = np.linspace(truncnorm.ppf(0.01, ddfsnow_a, ddfsnow_b),
                                          truncnorm.ppf(0.99, ddfsnow_a, ddfsnow_b), 100)
                    x_values = ddfsnow_mu + ddfsnow_sigma * z_score
                    y_values = truncnorm.pdf(x_values, ddfsnow_a, ddfsnow_b, loc=ddfsnow_mu, scale=ddfsnow_sigma)
                elif input.ddfsnow_disttype == 'uniform':
                    z_score = np.linspace(uniform.ppf(0.01), uniform.ppf(0.99), 100)
                    x_values = ddfsnow_boundlow + z_score * (ddfsnow_boundhigh - ddfsnow_boundlow)
                    y_values = uniform.pdf(x_values, loc=ddfsnow_boundlow,
                                           scale=(ddfsnow_boundhigh - ddfsnow_boundlow))
            
            nrow = param_idx_dict[vn][0]
            ncol = param_idx_dict[vn][1]
            ax[nrow,ncol].plot(x_values, y_values, color='k')
            
            # Labels
            ax[nrow,ncol].set_xlabel(vn_label_dict[vn], size=10)
            if nvar == 0:
                ax[nrow,ncol].set_title(glacier_str, fontsize=12)
    
            # Ensemble/Posterior distribution                
            for n_chain in range(len(ds.chain.values)):
                for count_iter, n_iters in enumerate(iters):
                    chain = ds['mp_value'].sel(chain=n_chain, mp=vn).values[burn:n_iters]
                
                    # gaussian distribution
                    kde = gaussian_kde(chain)
                    x_values_kde = x_values.copy()
                    y_values_kde = kde(x_values_kde)
                    
                    # Plot fitted distribution
                    ax[nrow,ncol].plot(x_values_kde, y_values_kde, color=colors[count_iter], 
                                       linestyle=linestyles[n_chain])
        # Close dataset
        ds.close()        
        
    # Legend (Note: hard code the spacing between the two legends)
    leg_lines = []
    leg_labels = []
    for count_iter, n_iters in enumerate(iters):
#        line = Line2D([0,1],[0,1], color='white')
#        leg_lines.append(line)
#        leg_labels.append('')
        line = Line2D([0,1],[0,1], color=colors[count_iter])
        leg_lines.append(line)
        leg_labels.append(str(int(n_iters)))    
    chain_labels = ['Center', 'Lower Bound', 'Upper Bound']
    for n_chain in range(len(ds.chain.values)):
#        line = Line2D([0,1],[0,1], color='white')
#        leg_lines.append(line)
#        leg_labels.append('')
        line = Line2D([0,1],[0,1], color='gray', linestyle=linestyles[n_chain])
        leg_lines.append(line)
        leg_labels.append(chain_labels[n_chain])
    
    fig.legend(leg_lines, leg_labels, loc='lower center', bbox_to_anchor=(0.5,0.01),
               handlelength=2, handletextpad=0.5, borderpad=0.3, frameon=True, ncol=5)
    fig.subplots_adjust(bottom=0.15)
    
    fig.text(0.02, 0.5, 'Probability Density', va='center', rotation='vertical', size=12)
    
        
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
    if input.tempchange_edge_method == 'mb':
        str_ending += '_edgeMBpt' + str(int(input.tempchange_edge_mb*100)).zfill(2)
    elif input.tempchange_edge_method == 'mb_norm':
        str_ending += '_edgeMBNormpt' + str(int(input.tempchange_edge_mbnorm*100)).zfill(2)
    elif input.tempchange_edge_method == 'mb_norm_slope':
        str_ending += '_edgeSpt' + str(int(input.tempchange_edge_mbnormslope*100)).zfill(2)
    str_ending += '_TCadjp' + str(int(input.tempchange_mu_adj*100)).zfill(2)
        
    if os.path.exists(mcmc_output_figures_fp) == False:
        os.makedirs(mcmc_output_figures_fp)        
    fig.savefig(fig_fp + 'prior_v_posteriors_2glac.png', 
                bbox_inches='tight', dpi=300)
    fig.clf()
    
    
    #%%
if option_papermcmc_solutionspace == 1:
    
    glac_no = ['13.26360']
    netcdf_fp = mcmc_output_netcdf_fp_3chain
    
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
if option_papermcmc_allglaciers_posteriorchanges == 1:
    netcdf_fp = mcmc_output_netcdf_fp_3chain
    
    fig_fp = netcdf_fp + 'figures/'
    if os.path.exists(fig_fp) == False:
        os.makedirs(fig_fp)
    
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
    
    prior_compare_cns = ['post_mb_mu', 'post_mb_std', 'mb_obs', 'mb_std',
                         'post_pf_mu', 'post_pf_std', 'prior_pf_mu', 'prior_pf_std',
                         'post_tc_mu', 'post_tc_std', 'prior_tc_mu', 'prior_tc_std',
                         'post_ddfsnow_mu', 'post_ddfsnow_std', 'prior_ddfsnow_mu', 'prior_ddfsnow_std']
    prior_compare = pd.DataFrame(np.zeros((main_glac_rgi.shape[0], len(prior_compare_cns))), columns=prior_compare_cns)
    
    for n, glac_str_wRGI in enumerate(main_glac_rgi['RGIId'].values):
#    for n, glac_str_wRGI in enumerate([main_glac_rgi['RGIId'].values[0]]):
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
        
        # Priors
        try:
            priors = pd.Series(ds.priors, index=ds['dim_0'])
        except:
            priors = pd.Series(ds.priors, index=ds.prior_cns)
        
        precfactor_boundlow = priors['pf_bndlow']
        precfactor_boundhigh = priors['pf_bndhigh']
        precfactor_mu = priors['pf_mu']
        precfactor_std = ((precfactor_boundhigh - precfactor_boundlow)**2 / 12)**0.5
        #  std_uniform = ((b - a)^2 / 12) ^ 0.5
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
        
        prior_compare_row = [df.massbal.mean(), df.massbal.std(), observed_massbal, observed_error,
                             df.precfactor.mean(), df.precfactor.std(), precfactor_mu, precfactor_std,
                             df.tempchange.mean(), df.tempchange.std(), tempchange_mu, tempchange_sigma,
                             df.ddfsnow.mean(), df.ddfsnow.std(), ddfsnow_mu, ddfsnow_sigma]
        prior_compare.loc[n,:] = prior_compare_row
        
        ds.close()
        
    # ===== PLOT METRICS =====
    figwidth=6.5
    figheight=8
    
    # Bin spacing (note: offset them, so centered on 0)
    bdict = {}
    bdict['massbal-Mean'] = np.arange(-0.1, 0.11, 0.01) - 0.005
    bdict['precfactor-Mean'] = np.arange(-0.3, 0.32, 0.02) - 0.01
    bdict['tempchange-Mean'] = np.arange(-0.3, 0.32, 0.02) - 0.01
    bdict['ddfsnow-Mean'] = np.arange(-0.5, 0.5, 0.05) - 0.025
    bdict['massbal-Standard Deviation'] = np.arange(-0.2, 0.22, 0.02) - 0.002
    bdict['precfactor-Standard Deviation'] = np.arange(-0.4, 0.42, 0.02) - 0.01
    bdict['tempchange-Standard Deviation'] = np.arange(-0.4, 0.42, 0.02) - 0.01
    bdict['ddfsnow-Standard Deviation'] = np.arange(-1.5, 1.51, 0.1) - 0.025
    
    tdict = {}
    glac_ylim = 40
    tdict['Mean'] = np.arange(0, glac_ylim + 1, 10)
    tdict['Standard Deviation'] = np.arange(0, glac_ylim + 1, 10)
    
    estimators = ['Mean', 'Standard Deviation']
    
    fig, ax = plt.subplots(len(variables), len(estimators), squeeze=False, sharex=False, sharey=False,
                           figsize=(figwidth,figheight), gridspec_kw = {'wspace':0.1, 'hspace':0.5})    
    
    for nvar, vn in enumerate(variables):

        if vn == 'massbal':
            mean_prior = prior_compare['mb_obs'].values
            mean_post = prior_compare['post_mb_mu'].values
            std_prior = prior_compare['mb_std'].values
            std_post = prior_compare['post_mb_std'].values
        elif vn == 'precfactor':
            mean_prior = prior_compare['prior_pf_mu'].values
            mean_post = prior_compare['post_pf_mu'].values
            std_prior = prior_compare['prior_pf_std'].values
            std_post = prior_compare['post_pf_std'].values
        elif vn == 'tempchange':
            mean_prior = prior_compare['prior_tc_mu'].values
            mean_post = prior_compare['post_tc_mu'].values
            std_prior = prior_compare['prior_tc_std'].values
            std_post = prior_compare['post_tc_std'].values
        elif vn == 'ddfsnow':
            mean_prior = prior_compare['prior_ddfsnow_mu'].values * 1000
            mean_post = prior_compare['post_ddfsnow_mu'].values * 1000
            std_prior = prior_compare['prior_ddfsnow_std'].values * 1000
            std_post = prior_compare['post_ddfsnow_std'].values * 1000     
            vn_label_units_dict['ddfsnow'] = '[10$^{3}$ mwe d$^{-1}$ $^\circ$C$^{-1}$]'
        
        dif_mean = mean_post - mean_prior
        dif_std = std_post - std_prior

        for nest, estimator in enumerate(estimators):
            if estimator == 'Mean':
                dif = dif_mean
                bcolor = 'blue'
            elif estimator == 'Standard Deviation':
                dif = dif_std
                bcolor = 'red'
        
            # ===== Plot =====
            hist, bins = np.histogram(dif, bins=bdict[vn + '-' + estimator])
            hist = hist * 100.0 / hist.sum()
            bins_centered = bins[1:] + (bins[0] - bins[1]) / 2
    
            # plot histogram
            ax[nvar,nest].bar(x=bins_centered, height=hist, width=(bins[1]-bins[0]), align='center',
                              edgecolor='black', color=bcolor, alpha=0.5)
            ax[nvar,nest].set_yticks(tdict[estimator])
            ax[nvar,nest].set_ylim(0,glac_ylim)
            
#            # Cumulative percentage
#            cum_hist = [hist[0:i].sum() for i in range(len(hist))]
#            ax2 = ax[nvar,nest].twinx()    
#            ax2.plot(bins_centered, cum_hist, color='black',
#                                linewidth=1, label='Cumulative %')
#            ax2.set_yticks(np.arange(0, 110, 20))
#            ax2.set_ylim(0,100)

            # Scatter plot instead
#            if estimator == 'Mean':
#                ax[nvar,nest].scatter(mean_prior, mean_post, marker='o', facecolors="none", edgecolors='b')
#                mean_min = np.min([np.min(mean_prior), np.min(mean_post)])
#                mean_max = np.max([np.max(mean_prior), np.max(mean_post)])
#                ax[nvar,nest].plot([mean_min, mean_max], [mean_min, mean_max], linestyle='-', color='k')
#            elif estimator == 'Standard Deviation':
#                ax[nvar,nest].scatter(std_prior, std_post, marker='o', facecolors="none", edgecolors='b')
#                std_min = np.min([np.min(std_prior), np.min(std_post)])
#                std_max = np.max([np.max(std_prior), np.max(std_post)])
#                ax[nvar,nest].plot([std_min, std_max], [std_min, std_max], linestyle='-', color='k')      
                
            # niceties
            if nvar == 0:
                ax[nvar,nest].set_title(estimator, fontsize=12)
            if nest == 0:
                ax[nvar,nest].set_ylabel(vn_title_dict[vn] + '\n\n% of Glaciers', fontsize=12, labelpad=3)
            if nest == 1:
                ax[nvar,nest].yaxis.set_major_locator(plt.NullLocator())
            
            ax[nvar,nest].set_xlabel(vn_label_units_dict[vn], fontsize=10)
                
    # Save figure
    fig.set_size_inches(figwidth,figheight)
    if os.path.exists(fig_fp) == False:
        os.makedirs(fig_fp)
    figure_fn = 'posterior_vs_prior_difference_histograms.png'
    fig.savefig(fig_fp + figure_fn, bbox_inches='tight', dpi=300)

#%%
if option_papermcmc_spatialdistribution_parameter == 1:
    print('plot spatial distribution')
    
    netcdf_fp = mcmc_output_netcdf_fp_all
    figure_fp = netcdf_fp + '../'
    grouping = 'degree'
    degree_size = 0.25
    
    vns = ['ddfsnow', 'tempchange', 'precfactor', 'dif_masschange']
    modelparams_fn = '../main_glac_rgi_20190308_wcal_wposteriors.csv'
    
    east = 104
    west = 67
    south = 25
    north = 48
    xtick = 5
    ytick = 5
    xlabel = 'Longitude [$^\circ$]'
    ylabel = 'Latitude [$^\circ$]'
    
    labelsize = 13
    
    colorbar_dict = {'precfactor':[0,5],
                     'tempchange':[-5,5],
                     'ddfsnow':[0.0036,0.0046],
                     'dif_masschange':[-0.1,0.1]}
    
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
    
    #%%
    # Load mean of all model parameters
    if os.path.isfile(mcmc_output_netcdf_fp_all + modelparams_fn) == False:
        
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
        modelparams_all.to_csv(netcdf_fp + modelparams_fn)
    else:
        modelparams_all = pd.read_csv(netcdf_fp + modelparams_fn)
        
    #%%
    modelparams_all['dif_cal_era_mean'] = modelparams_all['mb_mwea'] - modelparams_all['mb_mean']

    # remove nan values
    modelparams_all = (
            modelparams_all.drop(np.where(np.isnan(modelparams_all['mb_mean'].values) == True)[0].tolist(), 
                                   axis=0))
    modelparams_all.reset_index(drop=True, inplace=True)
    
    # Histogram: Mass balance [mwea], Observation - ERA
    hist_cn = 'dif_cal_era_mean'
    low_bin = np.floor(modelparams_all[hist_cn].min())
    high_bin = np.ceil(modelparams_all[hist_cn].max())
    bins = [low_bin, -0.2, -0.1, -0.05, -0.02, 0.02, 0.05, 0.1, 0.2, high_bin]
    plot_hist(modelparams_all, hist_cn, bins, xlabel='Mass balance [mwea]\n(Calibration - MCMC_mean)', 
              ylabel='# Glaciers', fig_fn='MB_cal_vs_mcmc_hist.png', fig_fp=figure_fp)
              
              
    # Histogram: Glacier Area [km2]
    hist_cn = 'Area'
    low_bin = np.floor(modelparams_all[hist_cn].min())
    high_bin = np.ceil(modelparams_all[hist_cn].max() + 1)
    bins = [0, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 25, 50, 100, high_bin]
    plot_hist(modelparams_all, hist_cn, bins, xlabel='Glacier Area [km2]', 
              ylabel='# Glaciers', fig_fn='Glacier_Area.png', fig_fp=figure_fp)
              
    # Scatterplot: Glacier Area [km2] vs. Mass balance, color-coded by mass balance difference
    fig, ax = plt.subplots()
    cmap = 'RdYlBu_r'
    norm = plt.Normalize(colorbar_dict[vn][0], colorbar_dict[vn][1])    
    ax.scatter(modelparams_all['Area'], modelparams_all['mb_mwea'], c=modelparams_all['dif_cal_era_mean'], 
               cmap=cmap, norm=norm, s=5)
    ax.set_xlim([0,200])
    ax.set_ylim([-2.5,1.25])
    ax.set_ylabel('Mass Balance [mwea]', size=12)
    ax.set_xlabel('Area [km$^2$]', size=12)
    # Inset axis over main axis
    ax_inset = plt.axes([.35, .19, .48, .35])
    ax_inset.scatter(modelparams_all['Area'], modelparams_all['mb_mwea'], c=modelparams_all['dif_cal_era_mean'], 
               cmap=cmap, norm=norm, s=3)
    ax_inset.set_xlim([0,5])
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm._A = []
    plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.01)
    fig.text(1.01, 0.5, 'Mass Balance [mwea]\n(Observation - Model)', va='center', rotation='vertical', size=12)
    # Save figure
    fig.set_size_inches(6,4)
    fig_fn = 'MB_vs_area_wdif.png'
    fig.savefig(figure_fp + fig_fn, bbox_inches='tight', dpi=300)


    # Map: Mass change, difference between calibration data and median data
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

    for vn in vns:
    
        # Group data
        if vn in ['precfactor', 'tempchange', 'ddfsnow']:
            groups, ds_vn_deg = partition_groups(grouping, vn, modelparams_all, regional_calc='mean')
        elif vn == 'dif_masschange':
            # Group calculations
            groups, ds_group_cal = partition_groups(grouping, 'mb_cal_Gta', modelparams_all, regional_calc='sum')
            groups, ds_group_era = partition_groups(grouping, 'mb_era_Gta', modelparams_all, regional_calc='sum')
            groups, ds_group_area = partition_groups(grouping, 'Area', modelparams_all, regional_calc='sum')
        
            # Group difference [Gt/yr]
            dif_cal_era_Gta = (np.array([x[1] for x in ds_group_cal]) - np.array([x[1] for x in ds_group_era])).tolist()
            ds_group_dif_cal_era_Gta = [[x[0],dif_cal_era_Gta[n]] for n, x in enumerate(ds_group_cal)]
            # Group difference [mwea]
            area = [x[1] for x in ds_group_area]
            ds_group_dif_cal_era_mwea = [[x[0], dif_cal_era_Gta[n] / area[n] * 1000] for n, x in enumerate(ds_group_cal)]
            ds_vn_deg = ds_group_dif_cal_era_mwea
    
        # Create the projection
        fig, ax = plt.subplots(1, 1, figsize=(10,5), subplot_kw={'projection':cartopy.crs.PlateCarree()})
        # Add country borders for reference
        ax.add_feature(cartopy.feature.BORDERS, alpha=0.15, zorder=10)
        ax.add_feature(cartopy.feature.COASTLINE)
        # Set the extent
        ax.set_extent([east, west, south, north], cartopy.crs.PlateCarree())    
        # Label title, x, and y axes
        ax.set_xticks(np.arange(east,west+1,xtick), cartopy.crs.PlateCarree())
        ax.set_yticks(np.arange(south,north+1,ytick), cartopy.crs.PlateCarree())
        ax.set_xlabel(xlabel, size=labelsize)
        ax.set_ylabel(ylabel, size=labelsize)
        
        # Add contour lines
        srtm_contour_shp = cartopy.io.shapereader.Reader(srtm_contour_fn)
        srtm_contour_feature = cartopy.feature.ShapelyFeature(srtm_contour_shp.geometries(), cartopy.crs.PlateCarree(),
                                                              edgecolor='black', facecolor='none', linewidth=0.15)
        ax.add_feature(srtm_contour_feature, zorder=9)            
            
        cmap = 'RdYlBu'
        if vn in ['tempchange', 'dif_masschange']:
            cmap = 'RdYlBu_r'
        norm = plt.Normalize(colorbar_dict[vn][0], colorbar_dict[vn][1])                        
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm._A = []
        plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.01)
        if vn in ['precfactor', 'tempchange']:
            fig.text(1, 0.5, vn_label_dict[vn], va='center', ha='center', rotation='vertical', size=labelsize)
        else:
            fig.text(1.05, 0.5, vn_label_dict[vn], va='center', ha='center', rotation='vertical', size=labelsize)

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
        ax.pcolormesh(lons, lats, z_array, cmap=cmap, norm=norm, zorder=2, alpha=0.8)            
        
        # Save figure
        fig.set_size_inches(6,4)
        if degree_size < 1:
            degsize_name = 'pt' + str(int(degree_size * 100))
        else:
            degsize_name = str(degree_size)
        fig_fn = 'mp_' + vn + '_' + degsize_name + 'deg.png'
        fig.savefig(figure_fp + fig_fn, bbox_inches='tight', dpi=300)

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