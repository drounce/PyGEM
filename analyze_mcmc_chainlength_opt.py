""" Analyze MCMC output - chain length, etc. """

# Built-in libraries
import glob
import os
import pickle
# External libraries
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

#%%
option_metrics_vs_chainlength = 0
option_observation_vs_calibration = 0
option_prior_vs_posterior = 1


variables = ['massbal', 'precfactor', 'tempchange', 'ddfsnow']  
vn_title_dict = {'massbal':'Mass\nBalance',                                                                      
                 'precfactor':'Precipitation\nFactor',                                                              
                 'tempchange':'Temperature\nBias',                                                               
                 'ddfsnow':'Degree-Day \nFactor of Snow'}
vn_label_dict = {'massbal':'Mass Balance\n[mwea]',                                                                      
                 'precfactor':'Precipitation Factor\n[-]',                                                              
                 'tempchange':'Temperature Bias\n[$^\circ$C]',                                                               
                 'ddfsnow':'Degree Day Factor of Snow\n[mwe d$^{-1}$ $^\circ$C$^{-1}$]'}
#vn_label_dict = {'massbal':'[mwea]',
#                 'precfactor':'[-]',
#                 'tempchange':'[degC]',
#                 'ddfsnow':'[mwe $degC^{-1} d^{-1}$]'}
metric_title_dict = {'Gelman-Rubin':'Gelman-Rubin Statistic',
                     'MC Error': 'Monte Carlo Error',
                     'Effective N': 'Effective Sample Size'}
metrics = ['Gelman-Rubin', 'MC Error', 'Effective N']


# Export option
mcmc_output_netcdf_fp = input.output_filepath + 'cal_opt2_spc_3000glac_3chain_adjp12/'
mcmc_output_figures_fp = mcmc_output_netcdf_fp + 'figures/'
mcmc_output_csv_fp = mcmc_output_netcdf_fp + 'csv/'

en_fn_pkl = mcmc_output_csv_fp + 'effective_n_list.pkl'
mc_fn_pkl = mcmc_output_csv_fp + 'mc_error_list.pkl'
gr_fn_pkl = mcmc_output_csv_fp + 'gelman_rubin_list.pkl'
en_fn_csv = mcmc_output_csv_fp + 'effective_n_stats.csv'
mc_fn_csv = mcmc_output_csv_fp + 'mc_error_stats.csv'
gr_fn_csv = mcmc_output_csv_fp + 'gelman_rubin_stats.csv'

regions = ['13', '14', '15']

cal_datasets = ['shean']

burn=0
iterstep = 1000
itermax = 25000
chainlength = 10000
iterations = np.arange(0, 25000, iterstep)
if iterations[1] < 1000: 
    iterations[0] = 1000
else:
    iterations = iterations[1:]
if iterations[-1] != itermax:
    iterations = np.append(iterations, itermax)
# Bounds (90% bounds --> 95% above/below given threshold)
low_percentile = 5
high_percentile = 95

colors = ['#387ea0', '#fcb200', '#d20048']
linestyles = ['-', '--', ':']


def load_glacier_data(regions, filepath=mcmc_output_netcdf_fp):
    """ Load main_glac_rgi data and cal_data
    
    Parameters
    ----------
    regions : list of strings
        list of regions
    filepath : str
        filepath of folder to load glacier data from
        
    Returns
    -------
    main_glac_rgi : pd.DataFrame
        main glacier dataframe containing glacier properties
    cal_data : pd.DataFrame
        calibration dataframe containing information regarding the calibration data for each glacier
    glac_no : list of strings
        list of the glacier numbers
    """
    filelist = []
    for region in regions:
        filelist.extend(glob.glob(mcmc_output_netcdf_fp + str(region) + '*.nc'))
    
    glac_no = []
    reg_no = []
    for netcdf in filelist:
        glac_str = netcdf.split('/')[-1].split('.nc')[0]
        glac_no.append(glac_str)
        reg_no.append(glac_str.split('.')[0])
        
    # Load data for glaciers
    main_glac_rgi = pd.DataFrame()
    cal_data = pd.DataFrame()
    dates_table_nospinup = modelsetup.datesmodelrun(startyear=input.startyear, endyear=input.endyear, spinupyears=0)
    reg_no = sorted([i for i in set(reg_no)])
    glac_no = sorted(glac_no)
    for region in reg_no:
        reg_glac_list = []
        for glac in glac_no:
            if glac.split('.')[0] == region:
                reg_glac_list.append(glac.split('.')[1])
                
        # Glacier data
        main_glac_rgi_region = modelsetup.selectglaciersrgitable(rgi_regionsO1=[int(region)], rgi_regionsO2 = 'all',
                                                                 rgi_glac_number=reg_glac_list)
        # Glacier hypsometry
        main_glac_hyps_region = modelsetup.import_Husstable(main_glac_rgi_region, [int(region)], input.hyps_filepath,
                                                            input.hyps_filedict, input.hyps_colsdrop)
        # Calibration data
        cal_data_region = pd.DataFrame()
        for dataset in cal_datasets:
            cal_subset = class_mbdata.MBData(name=dataset, rgi_regionO1=int(region))
            cal_subset_data = cal_subset.retrieve_mb(main_glac_rgi_region, main_glac_hyps_region, dates_table_nospinup)
            cal_data_region = cal_data_region.append(cal_subset_data, ignore_index=True)
        cal_data_region = cal_data_region.sort_values(['glacno', 't1_idx'])
        cal_data_region.reset_index(drop=True, inplace=True)
        
        # Append datasets
        main_glac_rgi = main_glac_rgi.append(main_glac_rgi_region)
        cal_data = cal_data.append(cal_data_region)
        
    # reset index
    main_glac_rgi.reset_index(inplace=True, drop=True)
    cal_data.reset_index(inplace=True, drop=True)
    
    return main_glac_rgi, cal_data, glac_no

    
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
        
        
def plot_hist(df, cn, bins, xlabel=None, ylabel=None, fig_fn='hist.png', fig_fp=mcmc_output_figures_fp):
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
        

def retrieve_prior_parameters(modelparameters, glacier_rgi_table, glacier_area_t0, icethickness_t0, width_t0, elev_bins, 
                              glacier_gcm_temp, glacier_gcm_prec, glacier_gcm_elev, glacier_gcm_lrgcm, 
                              glacier_gcm_lrglac, dates_table, t1_idx, t2_idx, t1, t2, observed_massbal, mb_obs_min,
                              mb_obs_max):    
    def mb_mwea_calc(modelparameters, option_areaconstant=1):
        """
        Run the mass balance and calculate the mass balance [mwea]
        """
        # Mass balance calculations
        (glac_bin_temp, glac_bin_prec, glac_bin_acc, glac_bin_refreeze, glac_bin_snowpack, glac_bin_melt, 
         glac_bin_frontalablation, glac_bin_massbalclim, glac_bin_massbalclim_annual, glac_bin_area_annual, 
         glac_bin_icethickness_annual, glac_bin_width_annual, glac_bin_surfacetype_annual, 
         glac_wide_massbaltotal, glac_wide_runoff, glac_wide_snowline, glac_wide_snowpack, 
         glac_wide_area_annual, glac_wide_volume_annual, glac_wide_ELA_annual, offglac_wide_prec, 
         offglac_wide_refreeze, offglac_wide_melt, offglac_wide_snowpack, offglac_wide_runoff) = (
            massbalance.runmassbalance(modelparameters, glacier_rgi_table, glacier_area_t0, icethickness_t0, 
                                       width_t0, elev_bins, glacier_gcm_temp, glacier_gcm_prec, 
                                       glacier_gcm_elev, glacier_gcm_lrgcm, glacier_gcm_lrglac, dates_table, 
                                       option_areaconstant=option_areaconstant))
        # Compute glacier volume change for every time step and use this to compute mass balance
        glac_wide_area = glac_wide_area_annual[:-1].repeat(12)
        # Mass change [km3 mwe]
        #  mb [mwea] * (1 km / 1000 m) * area [km2]
        glac_wide_masschange = glac_wide_massbaltotal / 1000 * glac_wide_area
        # Mean annual mass balance [mwea]
        mb_mwea = glac_wide_masschange[t1_idx:t2_idx+1].sum() / glac_wide_area[0] * 1000 / (t2 - t1)
        return mb_mwea
    
    def find_tempchange_opt(tempchange_4opt, precfactor_4opt):
        """
        Find optimal temperature based on observed mass balance
        """
        # Use a subset of model parameters to reduce number of constraints required
        modelparameters[2] = precfactor_4opt
        modelparameters[7] = tempchange_4opt[0]
        # Mean annual mass balance [mwea]
        mb_mwea = mb_mwea_calc(modelparameters, option_areaconstant=1)
        return abs(mb_mwea - observed_massbal)
    
    def find_precfactor_opt(precfactor_4opt, tempchange_4opt):
        """
        Find optimal temperature based on observed mass balance
        """
        # Use a subset of model parameters to reduce number of constraints required
        modelparameters[2] = precfactor_4opt[0]
        modelparameters[7] = tempchange_4opt
        # Mean annual mass balance [mwea]
        mb_mwea = mb_mwea_calc(modelparameters, option_areaconstant=1)
        return abs(mb_mwea - observed_massbal)
    
    # ----- TEMPBIAS: max accumulation -----
    # Lower temperature bound based on max positive mass balance adjusted to avoid edge effects
    # Temperature at the lowest bin
    #  T_bin = T_gcm + lr_gcm * (z_ref - z_gcm) + lr_glac * (z_bin - z_ref) + tempchange
    lowest_bin = np.where(glacier_area_t0 > 0)[0][0]
    tempchange_max_acc = (-1 * (glacier_gcm_temp + glacier_gcm_lrgcm * 
                                (elev_bins[lowest_bin] - glacier_gcm_elev)).max())
    tempchange_boundlow = tempchange_max_acc
    # Compute max accumulation [mwea]
    modelparameters[2] = 1
    modelparameters[7] = -100
    mb_max_acc = mb_mwea_calc(modelparameters, option_areaconstant=1)
    
    
    # ----- TEMPBIAS: UPPER BOUND -----
    # MAXIMUM LOSS - AREA EVOLVING
    mb_max_loss = (-1 * (glacier_area_t0 * icethickness_t0).sum() / glacier_area_t0.sum() * 
                   input.density_ice / input.density_water / (t2 - t1))
    # Looping forward and backward to ensure optimization does not get stuck
    modelparameters[2] = 1
    modelparameters[7] = tempchange_boundlow
    mb_mwea_1 = mb_mwea_calc(modelparameters, option_areaconstant=0)
    # use absolute value because with area evolving the maximum value is a limit
    while mb_mwea_1 - mb_max_loss > 0:
        modelparameters[7] = modelparameters[7] + 1
        mb_mwea_1 = mb_mwea_calc(modelparameters, option_areaconstant=0)
    # Looping backward for tempchange at max loss 
    while abs(mb_mwea_1 - mb_max_loss) < 0.005:
        modelparameters[7] = modelparameters[7] - input.tempchange_step
        mb_mwea_1 = mb_mwea_calc(modelparameters, option_areaconstant=0)
    tempchange_max_loss = modelparameters[7] + input.tempchange_step
    
    # MB_OBS_MIN - AREA CONSTANT
    # Check if tempchange below min observation; if not, adjust upper TC bound to include mb_obs_min
    mb_tc_boundhigh = mb_mwea_calc(modelparameters, option_areaconstant=1)
    if mb_tc_boundhigh < mb_obs_min:
        tempchange_boundhigh = tempchange_max_loss
    else:
        modelparameters[2] = 1
        modelparameters[7] = tempchange_boundlow
        mb_mwea_1 = mb_mwea_calc(modelparameters, option_areaconstant=1)
        # Loop forward
        while mb_mwea_1 > mb_obs_min:
            modelparameters[7] = modelparameters[7] + 1
            mb_mwea_1 = mb_mwea_calc(modelparameters, option_areaconstant=1)
        # Loop back
        while mb_mwea_1 < mb_obs_min:
            modelparameters[7] = modelparameters[7] - input.tempchange_step
            mb_mwea_1 = mb_mwea_calc(modelparameters, option_areaconstant=1)
        tempchange_boundhigh = modelparameters[7] + input.tempchange_step
    
    # ----- TEMPBIAS: LOWER BOUND -----
    # AVOID EDGE EFFECTS (ONLY RELEVANT AT TC LOWER BOUND)
    def mb_norm_calc(mb):
        """ Normalized mass balance based on max accumulation and max loss """
        return (mb - mb_max_loss) / (mb_max_acc - mb_max_loss)
    def tc_norm_calc(tc):
        """ Normalized temperature change based on max accumulation and max loss """
        return (tc - tempchange_max_acc) / (tempchange_max_loss - tempchange_max_acc)
    
    modelparameters[2] = 1
    if input.tempchange_edge_method == 'mb':
        modelparameters[7] = tempchange_max_acc
        mb_mwea = mb_max_acc
        while mb_mwea > mb_max_acc - input.tempchange_edge_mb:
            modelparameters[7] = modelparameters[7] + input.tempchange_step
            mb_mwea = mb_mwea_calc(modelparameters, option_areaconstant=1)
        tempchange_boundlow = modelparameters[7]
    elif input.tempchange_edge_method == 'mb_norm':
        modelparameters[7] = tempchange_max_acc
        mb_norm = mb_norm_calc(mb_max_acc)
        while mb_norm > input.tempchange_edge_mbnorm:
            modelparameters[7] = modelparameters[7] + input.tempchange_step
            mb_norm = mb_norm_calc(mb_mwea_calc(modelparameters, option_areaconstant=1))
        tempchange_boundlow = modelparameters[7]
    elif input.tempchange_edge_method == 'mb_norm_slope':
        tempchange_boundlow = tempchange_max_acc
        mb_slope = 0
        while mb_slope > input.tempchange_edge_mbnormslope:
            tempchange_boundlow += input.tempchange_step
            modelparameters[7] = tempchange_boundlow + 0.5
            tc_norm_2 = tc_norm_calc(modelparameters[7])
            mb_norm_2 = mb_norm_calc(mb_mwea_calc(modelparameters, option_areaconstant=1))
            modelparameters[7] = tempchange_boundlow - 0.5
            tc_norm_1 = tc_norm_calc(modelparameters[7])
            mb_norm_1 = mb_norm_calc(mb_mwea_calc(modelparameters, option_areaconstant=1))
            mb_slope = (mb_norm_2 - mb_norm_1) / (tc_norm_2 - tc_norm_1)
        
    # ----- OTHER PARAMETERS -----
    # Assign TC_sigma
    tempchange_sigma = input.tempchange_sigma
    if (tempchange_boundhigh - tempchange_boundlow) / 6 < tempchange_sigma:
        tempchange_sigma = (tempchange_boundhigh - tempchange_boundlow) / 6
    
    if input.tempchange_mu < tempchange_boundlow:
        tempchange_init = tempchange_boundlow 
    elif input.tempchange_mu > tempchange_boundhigh:
        tempchange_init = tempchange_boundhigh
    else:
        tempchange_init = input.tempchange_mu
        
    # OPTIMAL PRECIPITATION FACTOR (TC = 0 or TC_boundlow)
    # Find optimized tempchange in agreement with observed mass balance
    tempchange_4opt = tempchange_init
    precfactor_opt_init = [1]
    precfactor_opt_bnds = (0, 10)
    precfactor_opt_all = minimize(find_precfactor_opt, precfactor_opt_init, args=(tempchange_4opt), 
                                  bounds=[precfactor_opt_bnds], method='L-BFGS-B')
    precfactor_opt = precfactor_opt_all.x[0]
    
    # Adjust precfactor so it's not < 0.5 or greater than 5
    precfactor_opt_low = 0.5
    precfactor_opt_high = 5
    if precfactor_opt < precfactor_opt_low:
        precfactor_opt = precfactor_opt_low
        tempchange_opt_init = [(tempchange_boundlow + tempchange_boundhigh) / 2]
        tempchange_opt_bnds = (tempchange_boundlow, tempchange_boundhigh)
        tempchange_opt_all = minimize(find_tempchange_opt, tempchange_opt_init, args=(precfactor_opt), 
                                      bounds=[tempchange_opt_bnds], method='L-BFGS-B')
        tempchange_opt = tempchange_opt_all.x[0]
    elif precfactor_opt > precfactor_opt_high:
        precfactor_opt = precfactor_opt_high
        tempchange_opt_init = [(tempchange_boundlow + tempchange_boundhigh) / 2]
        tempchange_opt_bnds = (tempchange_boundlow, tempchange_boundhigh)
        tempchange_opt_all = minimize(find_tempchange_opt, tempchange_opt_init, args=(precfactor_opt), 
                                      bounds=[tempchange_opt_bnds], method='L-BFGS-B')
        tempchange_opt = tempchange_opt_all.x[0]
    else:
        tempchange_opt = tempchange_4opt

    # TEMPCHANGE_SIGMA: derived from mb_obs_min and mb_obs_max
    # MB_obs_min
    #  note: tempchange_boundhigh does not require a limit because glacier is not evolving
    tempchange_adj = input.tempchange_step
    modelparameters[2] = precfactor_opt
    modelparameters[7] = tempchange_opt + tempchange_adj
    mb_mwea = mb_mwea_calc(modelparameters, option_areaconstant=1)
    while mb_mwea > mb_obs_min:    
        tempchange_adj += input.tempchange_step
        modelparameters[7] = tempchange_opt + tempchange_adj
        mb_mwea = mb_mwea_calc(modelparameters, option_areaconstant=1)
        
    # Expand upper bound if necessary
    if modelparameters[7] > tempchange_boundhigh:
        tempchange_boundhigh = modelparameters[7]
    # MB_obs_max
    modelparameters[2] = precfactor_opt
    modelparameters[7] = tempchange_opt - tempchange_adj
    mb_mwea = mb_mwea_calc(modelparameters, option_areaconstant=1)
    while mb_mwea < mb_obs_max and modelparameters[7] > tempchange_boundlow:
        tempchange_adj += input.tempchange_step
        modelparameters[7] = tempchange_opt - tempchange_adj
        if modelparameters[7] < tempchange_boundlow:
            modelparameters[7] = tempchange_boundlow
        mb_mwea = mb_mwea_calc(modelparameters, option_areaconstant=1)

    tempchange_sigma = tempchange_adj / 3
    
    # PRECIPITATION FACTOR: LOWER BOUND
    # Check PF_boundlow = 0
    modelparameters[2] = 0
    modelparameters[7] = tempchange_opt + tempchange_sigma
    mb_mwea = mb_mwea_calc(modelparameters, option_areaconstant=1)
    if mb_mwea > mb_obs_min:
        # Adjust TC_opt if PF=0 and TC = TC_opt + TC_sigma doesn't reach minimum observation
        precfactor_boundlow = 0
        while mb_mwea > mb_obs_min:
            tempchange_opt += input.tempchange_step
            modelparameters[7] = tempchange_opt + tempchange_sigma
            mb_mwea = mb_mwea_calc(modelparameters, option_areaconstant=1)
    else:
        # Determine lower bound
        precfactor_boundlow = precfactor_opt
        modelparameters[2] = precfactor_boundlow
        modelparameters[7] = tempchange_opt + tempchange_sigma
        mb_mwea = mb_mwea_calc(modelparameters, option_areaconstant=1)
        while mb_mwea > mb_obs_min and precfactor_boundlow > 0:
            precfactor_boundlow -= input.precfactor_step
            if precfactor_boundlow < 0:
                precfactor_boundlow = 0
            modelparameters[2] = precfactor_boundlow
            mb_mwea = mb_mwea_calc(modelparameters, option_areaconstant=1)

    # PRECIPITATION FACTOR: UPPER BOUND
    precfactor_boundhigh = precfactor_opt
    modelparameters[2] = precfactor_boundhigh
    modelparameters[7] = tempchange_opt - tempchange_sigma
    mb_mwea = mb_mwea_calc(modelparameters, option_areaconstant=1)
    while mb_mwea < mb_obs_max:
        precfactor_boundhigh += input.precfactor_step
        modelparameters[2] = precfactor_boundhigh
        mb_mwea = mb_mwea_calc(modelparameters, option_areaconstant=1)
    
    # TEMPERATURE BIAS: RE-CENTER
    precfactor_mu = (precfactor_boundlow + precfactor_boundhigh) / 2
    if abs(precfactor_opt - precfactor_mu) > 0.01:
        tempchange_opt_init = [tempchange_opt]
        tempchange_opt_bnds = (tempchange_boundlow, tempchange_boundhigh)
        tempchange_opt_all = minimize(find_tempchange_opt, tempchange_opt_init, args=(precfactor_mu), 
                                      bounds=[tempchange_opt_bnds], method='L-BFGS-B')
        tempchange_opt = tempchange_opt_all.x[0]
    
    
    precfactor_start = precfactor_mu
    if tempchange_boundlow < tempchange_opt + input.tempchange_mu_adj < tempchange_boundhigh:
        tempchange_mu = tempchange_opt + input.tempchange_mu_adj
    else:
        tempchange_mu = tempchange_opt
    tempchange_mu = tempchange_opt + input.tempchange_mu_adj
    # Remove tempchange from bounds
    if tempchange_mu >= tempchange_boundhigh - tempchange_sigma:
        tempchange_mu = tempchange_boundhigh - tempchange_sigma
    elif tempchange_mu <= tempchange_boundlow + tempchange_sigma:
        tempchange_mu = tempchange_boundlow + tempchange_sigma
    tempchange_start = tempchange_mu

    return (precfactor_boundlow, precfactor_boundhigh, precfactor_mu, precfactor_start, tempchange_boundlow, 
            tempchange_boundhigh, tempchange_mu, tempchange_sigma, tempchange_start)
    

# ===== PLOT OPTIONS ==================================================================================================
def metrics_vs_chainlength(regions, iters, burn=0):
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
#for batman in [0]:
#    iters = iterations
    
    filelist = []
    for region in regions:
        filelist.extend(glob.glob(mcmc_output_netcdf_fp + str(region) + '*.nc'))
        
    # Check if list already exists
    iter_ending = '_' + str(iterstep) + 'iterstep.pkl'
    en_fn_pkl.replace('.pkl', iter_ending)
    
    if os.path.isfile(en_fn_pkl.replace('.pkl', iter_ending)):
        with open(en_fn_pkl.replace('.pkl', iter_ending), 'rb') as f:
            en_list = pickle.load(f)
        with open(mc_fn_pkl.replace('.pkl', iter_ending), 'rb') as f:
            mc_list = pickle.load(f)
        with open(gr_fn_pkl.replace('.pkl', iter_ending), 'rb') as f:
            gr_list = pickle.load(f)
    else:
        #%%
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
        if os.path.exists(mcmc_output_csv_fp) == False:
            os.makedirs(mcmc_output_csv_fp)
                
        pickle_data(en_fn_pkl.replace('.pkl', iter_ending), en_list)
        pickle_data(mc_fn_pkl.replace('.pkl', iter_ending), mc_list)
        if len(ds.chain) > 1:
            pickle_data(gr_fn_pkl.replace('.pkl', iter_ending), gr_list)
    
    colors = ['#387ea0', '#fcb200', '#d20048']
    figwidth=6.5
    figheight=8
    fig, ax = plt.subplots(len(variables), len(metrics), squeeze=False, sharex=False, sharey=False,
                           figsize=(figwidth,figheight), gridspec_kw = {'wspace':0.4, 'hspace':0.25})        
    
    # Metric statistics
    df_cns = ['iters', 'mean', 'std', 'median', 'lowbnd', 'highbnd']

    for nmetric, metric in enumerate(metrics):
#    for nmetric, metric in enumerate(['MC Error']):
        
        if metric == 'Effective N':
            metric_list = en_list
        elif metric == 'MC Error':
            metric_list = mc_list
        elif metric == 'Gelman-Rubin':
            metric_list = gr_list
            
        for nvar, vn in enumerate(variables):
#        for nvar, vn in enumerate(['massbal']):
            
            metric_df = pd.DataFrame(np.zeros((len(iterations), len(df_cns))), columns=df_cns)
            metric_df['iters'] = iterations
            
            for niter, iteration in enumerate(iterations):
                iter_list = [i[niter] for i in metric_list[nvar][1]]
                metric_df.loc[niter,'mean'] = np.mean(iter_list)
                metric_df.loc[niter,'median'] = np.median(iter_list)
                metric_df.loc[niter,'std'] = np.std(iter_list)
                metric_df.loc[niter,'lowbnd'] = np.percentile(iter_list,low_percentile)
                metric_df.loc[niter,'highbnd'] = np.percentile(iter_list,high_percentile)
                
#                if iteration == 10000:
#                    A = iter_list.copy()
            
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
    if os.path.exists(mcmc_output_figures_fp) == False:
        os.makedirs(mcmc_output_figures_fp)
    figure_fn = 'chainlength_vs_metrics.png'
    fig.savefig(mcmc_output_figures_fp + figure_fn, bbox_inches='tight', dpi=300)
    
##    #%%
##    # Plot glacier area vs. mass balance
##    A = [i[9] for i in mc_list[0][1]]
##    #%%
##    fig, ax = plt.subplots()
##    ax.scatter(main_glac_rgi['Area'].values, A, s=5)
##    
##    ax.set(xlabel='Area [km2]', ylabel='Monte Carlo Error\nMass Balance [mwea]')
##    ax.set_ylim(0,0.04)
##    fig.savefig(mcmc_output_figures_fp + "MB_MCerror_scatter.png")
##    plt.show()   
    
#%%        
def observation_vs_calibration(regions, chainlength=chainlength, burn=0):
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

    main_glac_rgi, cal_data, glac_no = load_glacier_data(regions)
    
    # Mass balance comparison: observations and model
    mb_compare_cols = ['glacno', 'obs_mwea', 'obs_mwea_std', 'mod_mwea', 'mod_mwea_std', 'dif_mwea']
    mb_compare = pd.DataFrame(np.zeros((len(glac_no), len(mb_compare_cols))), columns=mb_compare_cols)
    mb_compare['glacno'] = glac_no
    mb_compare['obs_mwea'] = cal_data['mb_mwe'] / (cal_data['t2'] - cal_data['t1'])
    mb_compare['obs_mwea_std'] = cal_data['mb_mwe_err'] / (cal_data['t2'] - cal_data['t1'])
    for nglac, glac in enumerate(glac_no):
        # open dataset
        if nglac%500 == 0:
            print(glac)
        ds = xr.open_dataset(mcmc_output_netcdf_fp + glac + '.nc')
        mb_all = ds['mp_value'].sel(chain=0, mp='massbal').values[burn:chainlength]
        mb_compare.loc[nglac, 'mod_mwea'] = np.mean(mb_all)
        mb_compare.loc[nglac, 'mod_mwea_std'] = np.std(mb_all)
        # close dataset
        ds.close()

    # export csv
    mb_compare['dif_mwea'] = mb_compare['obs_mwea'] - mb_compare['mod_mwea']
    mb_compare.to_csv(mcmc_output_csv_fp + 'mb_compare_' + str(int(chainlength/1000)) + 'k.csv')

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
    

def prior_vs_posterior_single(glac_no, iters=[1000,15000], precfactor_disttype=input.precfactor_disttype, 
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
        ds = xr.open_dataset(mcmc_output_netcdf_fp + glacier_str + '.nc')
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
         tempchange_boundhigh, tempchange_mu, tempchange_sigma, tempchange_start) = (
                 retrieve_prior_parameters(modelparameters, glacier_rgi_table, glacier_area_t0, icethickness_t0, 
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
    metrics_vs_chainlength(regions, iterations, burn=burn)    

if option_observation_vs_calibration == 1:
    observation_vs_calibration(regions, chainlength=chainlength, burn=burn)

if option_prior_vs_posterior == 1:
    glac_no = ['13.26360']
    iters=[1000,10000]
#    main_glac_rgi, cal_data, glac_no = load_glacier_data(regions)
    for nglac, glac in enumerate(glac_no):
#        if main_glac_rgi.loc[nglac,'Area'] > 20:
#            print(main_glac_rgi.loc[nglac,'RGIId'], glac)
        prior_vs_posterior_single(glac, iters=iters)
#%%
#percentiles = np.arange(5,100,5)
#mcerror_percentiles = []
#for npercentile in percentiles:
#    print(npercentile, np.percentile(A,npercentile))
#    mcerror_percentiles.append(np.percentile(A,npercentile))
#    
#fig, ax = plt.subplots()
#ax.plot(percentiles, mcerror_percentiles)
#
#ax.set(xlabel='Percentile', ylabel='Mass Balance MC Error [mwea]')
#ax.grid()
#
#fig.savefig(mcmc_output_figures_fp + "massbal_mcerror_vs_percentiles.png")
#plt.show()

#%%
## open dataset
#glac_str = '13.00964'
#netcdf = mcmc_output_netcdf_fp + glac_str + '.nc'
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