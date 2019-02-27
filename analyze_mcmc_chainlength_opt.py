""" Determine optimal chain length """

import os
import numpy as np
import scipy as sp
import pandas as pd
import pickle
import xarray as xr
import pymc
from pymc import utils
from pymc.database import base

import glob

#plotting functions
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

import pygem_input as input

variables = ['massbal', 'precfactor', 'tempchange', 'ddfsnow']  
vn_title_dict = {'massbal':'Mass\nBalance',                                                                      
                 'precfactor':'Precipitation\nFactor',                                                              
                 'tempchange':'Temperature\nBias',                                                               
                 'ddfsnow':'Degree-Day \nFactor of Snow'}
vn_label_dict = {'massbal':'Mass balance\n[mwea]',                                                                      
                 'precfactor':'Precipitation factor\n[-]',                                                              
                 'tempchange':'Temperature bias\n[degC]',                                                               
                 'ddfsnow':'DDFsnow\n[mwe $degC^{-1} d^{-1}$]'}
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
mcmc_output_tables_fp = input.output_filepath + 'tables/'
mcmc_output_csv_fp = mcmc_output_netcdf_fp + 'csv/'
mcmc_output_hist_fp = input.output_filepath + 'hist/'

en_fn_pkl = mcmc_output_csv_fp + 'effective_n_list.pkl'
mc_fn_pkl = mcmc_output_csv_fp + 'mc_error_list.pkl'
gr_fn_pkl = mcmc_output_csv_fp + 'gelman_rubin_list.pkl'
en_fn_csv = mcmc_output_csv_fp + 'effective_n_stats.csv'
mc_fn_csv = mcmc_output_csv_fp + 'mc_error_stats.csv'
gr_fn_csv = mcmc_output_csv_fp + 'gelman_rubin_stats.csv'

iterstep = 3000
itermax = 25000
iterations = np.arange(0, 25000, iterstep)
iterations[0] = 1000
if iterations[-1] != itermax:
    iterations = np.append(iterations, itermax)
    
    
def effective_n(ds, vn, iters, burn):
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
    x = ds['mp_value'].sel(chain=0, mp=vn).values[burn:iters]
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


def MC_error(ds, vn, iters=None, burn=0, chain_no=0, batches=5):
    """
    Calculates MC Error using the batch simulation method.
    Also returns mean of trace

    Calculates the simulation standard error, accounting for non-independent
    samples. The trace is divided into batches, and the standard deviation of
    the batch means is calculated.

    With datasets of multiple chains, choses the highest MC error of all
    the chains and returns this value unless a chain number is specified

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing MCMC iterations for a single glacier with 3 chains
    vn : str
        Parameter variable name
    chain_no : int
        Number of chain to use (0, 1 or 2)
        If none, finds the highest MC error of the three chains
        and returns this value
    batches : int
        Number of batches to divide the trace in (default 5)

    """
    if iters is None:
        iters = len(ds.mp_value)

    # get iterations from ds
    trace = [ds['mp_value'].sel(chain=n_chain, mp=vn).values[burn:iters]
             for n_chain in ds.chain.values]

    result = batchsd(trace, batches)
    mean = np.mean(trace[chain_no])

    if len(ds.chain) <= chain_no or chain_no < 0:

        raise ValueError('Given chain_no is invalid')

    else:

        return (result[chain_no], mean)


def batchsd(trace, batches=5):
    """
    Calculates MC Error using the batch simulation method.

    Calculates the simulation standard error, accounting for non-independent
    samples. The trace is divided into batches, and the standard deviation of
    the batch means is calculated.
    With datasets of multiple chains, choses the highest MC error of all
    the chains and returns this value unless a chain number is specified

    Parameters
    ----------
    trace: np.ndarray
        Array representing MCMC chain
    batches : int
        Number of batches to divide the trace in (default 5)

    """
    # see if one trace or multiple
    if len(np.shape(trace)) > 1:

        return np.array([batchsd(t, batches) for t in trace])

    else:
        if batches == 1:
            return np.std(trace) / np.sqrt(len(trace))

        try:
            batched_traces = np.resize(trace, (batches, int(len(trace) / batches)))
        except ValueError:
            # If batches do not divide evenly, trim excess samples
            resid = len(trace) % batches
            batched_traces = np.resize(trace[:-resid],
                (batches, len(trace[:-resid]) / batches))

        means = np.mean(batched_traces, 1)

        return np.std(means) / np.sqrt(batches)
    

#def assessment_vs_chain_length(iters, region='all', burn=0, mean=False):
#    """
#    Plot gelman-rubin statistic, effective_n (autocorrelation with lag
#    100) and markov chain error plots.
#
#    Takes the output from the Markov Chain model and plots the results
#    for the mass balance, temperature change, precipitation factor,
#    and degree day factor of snow.  Also, outputs the plots associated
#    with the model.
#
#    Parameters
#    ----------
#    iters : int
#        Number of iterations associated with the Markov Chain
#    burn : list of ints
#        List of burn in values to plot for Gelman-Rubin stats
#
#    Returns
#    -------
#    .png files
#        saves figure showing how assessment values change with
#        number of mcmc iterations
#    """
#%%
for batman in [0]:
    iters = iterations
    region='all'
    burn=0
    mean=True
    
    # find all netcdf files (representing glaciers)
    if region == 'all':
        regions = ['13', '14', '15']
        filelist = []
        for reg in regions:
            filelist.extend(glob.glob(mcmc_output_netcdf_fp + str(reg) + '*.nc'))
    else:
        filelist = glob.glob(mcmc_output_netcdf_fp + str(region) + '*.nc')
        
        #%%
        
#def write_table2(iters, region='all', burn=0):
#    '''
#    Writes a csv table that lists mean MCMC assessment values for
#    each glacier (represented by a netcdf file) for all glaciers at
#    different chain lengths.
#
#    Writes out the values of effective_n (autocorrelation with
#    lag 100), Gelman-Rubin Statistic, MC_error.
#
#    Parameters
#    ----------
#    region : int
#        number of the glacier region (13, 14 or 15)
#    iters : int
#        Number of iterations associated with the Markov Chain
#    burn : list of ints
#        List of burn in values to plot for Gelman-Rubin stats
#
#    Returns
#    -------
#    .csv files
#        Saves tables to csv file.
#
#    '''
        #%%
    # Check if list already exists
    if os.path.isfile(en_fn_pkl):
        with open(en_fn_pkl, 'rb') as f:
            en_list = pickle.load(f)
        with open(mc_fn_pkl, 'rb') as f:
            mc_list = pickle.load(f)
        with open(gr_fn_pkl, 'rb') as f:
            gr_list = pickle.load(f)
    else:
        # Lists to record metrics
        glac_no = []
        en_list = [[vn, []] for vn in variables]
        gr_list = [[vn, []] for vn in variables]
        mc_list = [[vn, []] for vn in variables]
    
        # iterate through each glacier
        for netcdf in filelist:
            glac_str = netcdf.split('/')[-1].split('.nc')[0]
            glac_no.append(glac_str)
            print(glac_str)
    
            # open dataset
            ds = xr.open_dataset(netcdf)
            
            # calculate metrics for each variable
            for nvar, vn in enumerate(variables):
                # metrics
                en = [effective_n(ds, vn=vn, iters=i, burn=burn) for i in iters]
                mc = [MC_error(ds, vn=vn, iters=i, burn=burn)[0] for i in iters]
                if len(ds.chain) > 1:
                    gr = [gelman_rubin(ds, vn=vn, iters=i, burn=burn) for i in iters]
                # append to list
                en_list[nvar][1].append(en)
                mc_list[nvar][1].append(mc)
                # test if multiple chains exist
                if len(ds.chain) > 1:
                    gr_list[nvar][1].append(gr)
    
            # close datase
            ds.close()
            
        # Pickle lists for next time
        if os.path.exists(mcmc_output_csv_fp) == False:
            os.makedirs(mcmc_output_csv_fp)
            
        def pickle_list(var_fn, var_list):
            with open(var_fn, 'wb') as f:
                pickle.dump(var_list, f)
                
        pickle_list(en_fn_pkl, en_list)
        pickle_list(mc_fn_pkl, mc_list)
        if len(ds.chain) > 1:
            pickle_list(gr_fn_pkl, gr_list)
    
    #%%
    colors = ['#387ea0', '#fcb200', '#d20048']
    figwidth=6.5
    figheight=8
    fig, ax = plt.subplots(len(variables), len(metrics), squeeze=False, sharex=False, sharey=False,
                           figsize=(figwidth,figheight), gridspec_kw = {'wspace':0.4, 'hspace':0.25})        
    # Bounds - note: using 90% bounds means that 95% will be above/below a given threshold.
    low_percentile = 5
    high_percentile = 95
    # Metric statistics
    df_cns = ['iters', 'mean', 'std', 'median', 'lowbnd', 'highbnd']

    for nmetric, metric in enumerate(metrics):
#    for nmetric, metric in enumerate(['Gelman-Rubin']):
        
        if metric == 'Effective N':
            metric_list = en_list
            metric_fn = en_fn_csv
        elif metric == 'MC Error':
            metric_list = mc_list
            metric_fn = mc_fn_csv
        elif metric == 'Gelman-Rubin':
            metric_list = gr_list
            metric_fn = gr_fn_csv
            
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
            
               
            
#assessment_vs_chain_length(iters=iterations, region='all', burn=0)