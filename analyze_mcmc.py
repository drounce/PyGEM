"""Run the model calibration"""
# Spyder cannot run parallels, so always set -option_parallels=0 when testing in Spyder.

# Built-in libraries
import os
import glob

# External libraries
import pandas as pd
import numpy as np
import xarray as xr
import pymc
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pymc import utils
from pymc.database import base
            
from scipy.stats.kde import gaussian_kde
from scipy.stats import norm
from scipy.stats import truncnorm
from scipy.stats import uniform
from scipy.stats import linregress
from scipy.stats import lognorm
from scipy.optimize import minimize
# Local libraries
import pygem_input as input
import pygemfxns_modelsetup as modelsetup
import pygemfxns_massbalance as massbalance
import class_mbdata
import class_climate


#%% ===== SCRIPT SPECIFIC INPUT DATA =====
cal_datasets = ['shean']
#cal_datasets = ['shean', 'wgms_d']

# mcmc model parameters
parameters = ['tempchange', 'precfactor', 'ddfsnow']
#parameters = ['tempchange', 'precfactor']
#parameters = ['tempchange']
parameters_all = ['ddfsnow', 'precfactor', 'tempchange', 'ddfice', 'lrgcm', 'lrglac', 'precgrad', 'tempsnow']
# Autocorrelation lags
acorr_maxlags = 100

# Export option
#mcmc_output_netcdf_fp = input.output_fp_cal + 'netcdf/'
#mcmc_output_netcdf_fp = input.output_filepath + 'cal_opt2_allglac_1ch_tn_20181018/reg13/'
#mcmc_output_netcdf_fp = input.output_filepath + 'cal_opt2_allglac_1ch_tn_20190108/'
mcmc_output_netcdf_fp = input.output_filepath + 'cal_opt2/'
mcmc_output_figures_fp = mcmc_output_netcdf_fp + 'figures/'
mcmc_output_tables_fp = input.output_fp_cal + 'tables/'
mcmc_output_csv_fp = input.output_fp_cal + 'csv/'
mcmc_output_hist_fp = input.output_fp_cal + 'hist/'

debug = False


def prec_transformation(precfactor_raw, lowbnd=input.precfactor_boundlow):
    """
    Converts raw precipitation factors from normal distribution to correct values.

    Takes raw values from normal distribution and converts them to correct precipitation factors according to:
        if x >= 0:
            f(x) = x + 1
        else:
            f(x) = 1 - x / lowbnd * (1 - (1/(1-lowbnd)))
    i.e., normally distributed values from -2 to 2 and converts them to be 1/3 to 3.

    Parameters
    ----------
    precfactor_raw : float
        numpy array of untransformed precipitation factor values

    Returns
    -------
    x : float
        array of corrected precipitation factors
    """        
    x = precfactor_raw.copy()
    x[x >= 0] = x[x >= 0] + 1
    x[x < 0] = 1 - x[x < 0] / lowbnd * (1 - (1/(1-lowbnd)))        
    return x


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


def gelman_rubin(ds, vn, iters=1000, burn=0):
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


def summary(netcdf, glacier_cal_data, iters=[5000, 10000, 25000], alpha=0.05, start=0,
            batches=100, chain=None, roundto=3, filename='output.txt'):
        """
        Generate a pretty-printed summary of the mcmc chain for different
        chain lengths.

        Parameters
        ----------
        ds : xr.Dataset
            Dataset containing MCMC results
        alpha : float
            The alpha level for generating posterior intervals. Defaults to
            0.05.
        start : int
          The starting index from which to summarize (each) chain. Defaults
          to zero.
        batches : int
          Batch size for calculating standard deviation for non-independent
          samples. Defaults to 100.
        chain : int
          The index for which chain to summarize. Defaults to None (all
          chains).
        roundto : int
          The number of digits to round posterior statistics.
        filename : str
            Name of the text

        Returns
        -------
        .txt file
            Summary statistics printed out to a text file of given name
        """

        # open dataset
        ds = xr.open_dataset(netcdf)

        # Extract calibration information needed for priors
        # Variables to plot
        variables = ds.mp.values[:].tolist()
        for i in parameters_all:
            if i in variables:
                variables.remove(i)
        variables.extend(parameters)
        # Observations data
        obs_type_list = []
        for x in range(glacier_cal_data.shape[0]):
            cal_idx = glacier_cal_data.index.values[x]
            obs_type = glacier_cal_data.loc[cal_idx, 'obs_type']
            obs_type_list.append(obs_type)

        # open file to write to
        file = open(filename, 'w')

        for iteration in iters:

            print('\n%s:' % (str(iteration) + ' iterations'), file=file)

            for vn in variables:

                # get trace from database
                trace = ds['mp_value'].sel(chain=0, mp=vn).values[:iteration]

                # Calculate statistics for Node
                statdict = stats(
                    trace,
                    alpha=alpha,
                    start=start,
                    batches=batches,
                    chain=chain)

                size = np.size(statdict['mean'])

                print('\n%s:' % vn, file=file)
                print(' ', file=file)

                # Initialize buffer
                buffer = []

                # Index to interval label
                iindex = [key.split()[-1] for key in statdict.keys()].index('interval')
                interval = list(statdict.keys())[iindex]

                # Print basic stats
                buffer += [
                    'Mean             SD            MC Error(percent of Mean)       %s' %
                    interval]
                buffer += ['-' * len(buffer[-1])]

                indices = range(size)
                if len(indices) == 1:
                    indices = [None]

                _format_str = lambda x, i=None, roundto=2: str(np.round(x.ravel()[i].squeeze(), roundto))

                for index in indices:
                    # Extract statistics and convert to string
                    m = _format_str(statdict['mean'], index, roundto)
                    sd = _format_str(statdict['standard deviation'], index, roundto)
                    mce = _format_str(statdict['mc error'], index, roundto)
                    hpd = str(statdict[interval].reshape(
                            (2, size))[:,index].squeeze().round(roundto))

                    # Build up string buffer of values
                    valstr = m
                    valstr += ' ' * (17 - len(m)) + sd
                    valstr += ' ' * (17 - len(sd)) + mce
                    valstr += ' ' * (len(buffer[-1]) - len(valstr) - len(hpd)) + hpd

                    buffer += [valstr]

                buffer += [''] * 2

                # Print quantiles
                buffer += ['Posterior quantiles:', '']

                buffer += [
                    '2.5             25              50              75             97.5']
                buffer += [
                    ' |---------------|===============|===============|---------------|']

                for index in indices:
                    quantile_str = ''
                    for i, q in enumerate((2.5, 25, 50, 75, 97.5)):
                        qstr = _format_str(statdict['quantiles'][q], index, roundto)
                        quantile_str += qstr + ' ' * (17 - i - len(qstr))
                    buffer += [quantile_str.strip()]

                buffer += ['']

                print('\t' + '\n\t'.join(buffer), file=file)

        file.close()


def stats(trace, alpha=0.05, start=0, batches=100,
              chain=None, quantiles=(2.5, 25, 50, 75, 97.5)):
        """
        Generate posterior statistics for node.

        Parameters
        ----------
        trace : numpy.ndarray
            single dimension array containing mcmc iterations
        alpha : float
          The alpha level for generating posterior intervals. Defaults to
          0.05.
        start : int
          The starting index from which to summarize (each) chain. Defaults
          to zero.
        batches : int
          Batch size for calculating standard deviation for non-independent
          samples. Defaults to 100.
        chain : int
          The index for which chain to summarize. Defaults to None (all
          chains).
        quantiles : tuple or list
          The desired quantiles to be calculated. Defaults to (2.5, 25, 50, 75, 97.5).

        Returns
        -------
        statdict : dict
            dict containing the following statistics of the trace (with the same key names)

            'n': length of mcmc chain
            'standard deviation':
            'mean':
            '%s%s HPD interval' % (int(100 * (1 - alpha)), '%'): utils.hpd(trace, alpha),
            'mc error': mc error as percentage of the mean
            'quantiles':

        """

        n = len(trace)

        return {
            'n': n,
            'standard deviation': trace.std(0),
            'mean': trace.mean(0),
            '%s%s HPD interval' % (int(100 * (1 - alpha)), '%'): utils.hpd(trace, alpha),
            'mc error': base.batchsd(trace, min(n, batches)) / (abs(trace.mean(0)) / 100),
            'quantiles': utils.quantiles(trace, qlist=quantiles)
        }


def write_csv_results(models, variables, distribution_type='truncnormal'):
    """
    Write parameter statistics (mean, standard deviation, effective sample number, gelman_rubin, etc.) to csv.

    Parameters
    ----------
    models : list of pymc.MCMC.MCMC
        Models containing traces of parameters, summary statistics, etc.
    distribution_type : str
        Distribution type either 'truncnormal' or 'uniform' (default truncnormal)

    Returns
    -------
    exports .csv
    """
    model = models[0]
    # Write statistics to csv
    output_csv_fn = (input.mcmc_output_csv_fp + glacier_str + '_' + distribution_type + '_statistics_' +
                     str(len(models)) + 'chain_' + str(input.mcmc_sample_no) + 'iter_' +
                     str(input.mcmc_burn_no) + 'burn' + '.csv')
    model.write_csv(output_csv_fn, variables=['massbal', 'precfactor', 'tempchange', 'ddfsnow'])
    # Import and export csv
    csv_input = pd.read_csv(output_csv_fn)
    # Add effective sample size to csv
    massbal_neff = effective_n(model, 'massbal')
    precfactor_neff = effective_n(model, 'precfactor')
    tempchange_neff = effective_n(model, 'tempchange')
    ddfsnow_neff = effective_n(model, 'ddfsnow')
    effective_n_values = [massbal_neff, precfactor_neff, tempchange_neff, ddfsnow_neff]
    csv_input['n_eff'] = effective_n_values
    # If multiple chains, add Gelman-Rubin Statistic
    if len(models) > 1:
        gelman_rubin_values = []
        for vn in variables:
            gelman_rubin_values.append(gelman_rubin(models, vn))
        csv_input['gelman_rubin'] = gelman_rubin_values
    csv_input.to_csv(output_csv_fn, index=False)

#%%
def plot_mc_results(netcdf_fn, glacier_cal_data,
                    iters=50, burn=0, newsetup=0, mb_max_acc=0, mb_max_loss=0, 
                    precfactor_disttype=input.precfactor_disttype,
                    precfactor_lognorm_mu=input.precfactor_lognorm_mu, 
                    precfactor_lognorm_tau=input.precfactor_lognorm_tau,
                    precfactor_mu=input.precfactor_mu, precfactor_sigma=input.precfactor_sigma,
                    precfactor_boundlow=input.precfactor_boundlow,
                    precfactor_boundhigh=input.precfactor_boundhigh,
                    tempchange_disttype = input.tempchange_disttype,
                    tempchange_mu=input.tempchange_mu, tempchange_sigma=input.tempchange_sigma,
                    tempchange_boundlow=input.tempchange_boundlow,
                    tempchange_boundhigh=input.tempchange_boundhigh,
                    ddfsnow_disttype=input.ddfsnow_disttype,
                    ddfsnow_mu=input.ddfsnow_mu, ddfsnow_sigma=input.ddfsnow_sigma,
                    ddfsnow_boundlow=input.ddfsnow_boundlow, ddfsnow_boundhigh=input.ddfsnow_boundhigh):
    """
    Plot trace, prior/posterior distributions, autocorrelation, and pairwise scatter for each parameter.

    Takes the output from the Markov Chain model and plots the results for the mass balance, temperature change,
    precipitation factor, and degree day factor of snow.  Also, outputs the plots associated with the model.

    Parameters
    ----------
    netcdf_fn : str
        Netcdf of MCMC methods with chains of model parameters
    iters : int
        Number of iterations associated with the Markov Chain
    burn : int
        Number of iterations to burn in with the Markov Chain
    precfactor_disttype : str
        Distribution type of precipitation factor (either 'lognormal', 'uniform', or 'custom')
    precfactor_lognorm_mu : float
        Lognormal mean of precipitation factor (default assigned from input)
    precfactor_lognorm_tau : float
        Lognormal tau (1/variance) of precipitation factor (default assigned from input)
    precfactor_mu : float
        Mean of precipitation factor (default assigned from input)
    precfactor_sigma : float
        Standard deviation of precipitation factor (default assigned from input)
    precfactor_boundlow : float
        Lower boundary of precipitation factor (default assigned from input)
    precfactor_boundhigh : float
        Upper boundary of precipitation factor (default assigned from input)
    tempchange_disttype : str
        Distribution type of tempchange (either 'truncnormal' or 'uniform')
    tempchange_mu : float
        Mean of temperature change (default assigned from input)
    tempchange_sigma : float
        Standard deviation of temperature change (default assigned from input)
    tempchange_boundlow : float
        Lower boundary of temperature change (default assigned from input)
    tempchange_boundhigh: float
        Upper boundary of temperature change (default assigned from input)
    ddfsnow_disttype : str
        Distribution type of degree day factor of snow (either 'truncnormal' or 'uniform')
    ddfsnow_mu : float
        Mean of degree day factor of snow (default assigned from input)
    ddfsnow_sigma : float
        Standard deviation of degree day factor of snow (default assigned from input)
    ddfsnow_boundlow : float
        Lower boundary of degree day factor of snow (default assigned from input)
    ddfsnow_boundhigh : float
        Upper boundary of degree day factor of snow (default assigned from input)

    Returns
    -------
    .png files
        Saves two figures of (1) trace, histogram, and autocorrelation, and (2) pair-wise scatter plots.
    """
    #%%
    # Open dataset
    ds = xr.open_dataset(netcdf_fn)
    # Create list of model output to be used with functions
    dfs = []
    for n_chain in ds.chain.values:
        dfs.append(pd.DataFrame(ds['mp_value'].sel(chain=n_chain).values[burn:burn+iters], columns=ds.mp.values))

    # Extract calibration information needed for priors
    # Variables to plot
    variables = ds.mp.values[:].tolist()
    for i in parameters_all:
        if i in variables:
            variables.remove(i)
    variables.extend(parameters)
    # Observations data
    obs_list = []
    obs_err_list_raw = []
    obs_type_list = []
    for x in range(glacier_cal_data.shape[0]):
        cal_idx = glacier_cal_data.index.values[x]
        obs_type = glacier_cal_data.loc[cal_idx, 'obs_type']
        obs_type_list.append(obs_type)
        # Mass balance comparisons
        if glacier_cal_data.loc[cal_idx, 'obs_type'].startswith('mb'):
            # Mass balance [mwea]
            t1 = glacier_cal_data.loc[cal_idx, 't1'].astype(int)
            t2 = glacier_cal_data.loc[cal_idx, 't2'].astype(int)
            observed_massbal = glacier_cal_data.loc[cal_idx,'mb_mwe'] / (t2 - t1)
            observed_error = glacier_cal_data.loc[cal_idx,'mb_mwe_err'] / (t2 - t1)
            obs_list.append(observed_massbal)
            obs_err_list_raw.append(observed_error)
    obs_err_list = [x if ~np.isnan(x) else np.nanmean(obs_err_list_raw) for x in obs_err_list_raw]
#%%        
    # ===== CHAIN, HISTOGRAM, AND AUTOCORRELATION PLOTS ===========================
    fig, ax = plt.subplots(len(variables), 3, squeeze=False, figsize=(12, len(variables)*3), 
                           gridspec_kw={'wspace':0.3, 'hspace':0.2})
    fig.suptitle('mcmc_ensembles_' + glacier_str, y=0.94)

    # Bounds (SciPy convention)
    precfactor_a = (precfactor_boundlow - precfactor_mu) / precfactor_sigma
    precfactor_b = (precfactor_boundhigh - precfactor_mu) / precfactor_sigma
    tempchange_a = (tempchange_boundlow - tempchange_mu) / tempchange_sigma
    tempchange_b = (tempchange_boundhigh - tempchange_mu) / tempchange_sigma
    ddfsnow_a = (ddfsnow_boundlow - ddfsnow_mu) / ddfsnow_sigma
    ddfsnow_b = (ddfsnow_boundhigh - ddfsnow_mu) / ddfsnow_sigma

    # Labels for plots
    vn_label_dict = {'massbal':'Mass balance\n[mwea]',
                     'precfactor':'Precipitation factor\n[-]',
                     'tempchange':'Temperature bias\n[$^\circ$C]',
                     'ddfsnow':'DDFsnow\n[m w.e. d$^{-1}$ $^\circ$C$^{-1}$]'
                     }
    vn_label_nounits_dict = {'massbal':'MB',
                             'precfactor':'Precfactor',
                             'tempchange':'Tempbias',
                             'ddfsnow':'DDFsnow'
                             }
#    vn_label_units_dict = {'massbal':'[mwea]',
#                           'precfactor':'[-]',
#                           'tempchange':'[$^\circ$C]',
#                           'ddfsnow':'[m w.e. d$^{-1}$ $^\circ$C$^{-1}$]'
#                           }
    
    def calc_histogram(chain, nbins):
        """
        Calculate heights of histogram based on given bins
        
        Parameters
        ----------
        chain : np.array
            values to bin
        nbins : int
            number of bins
            
        Returns
        -------
        hist : np.array
            number of values in each bin
        bins : np.array
            bin edges, note there will be one more edge than the len(hist) since the hist is between all the edges
        bin_spacing : float
            bin spacing
        """
        # Histogram
        x_range = chain.max() - chain.min()
        bin_spacing = np.round(x_range,-int(np.floor(np.log10(abs(x_range))))) / nbins
        bin_start = np.floor(chain.min()*(1/bin_spacing)) * bin_spacing
        bin_end = np.ceil(chain.max() * (1/bin_spacing)) * bin_spacing
        bins = np.arange(bin_start, bin_end + bin_spacing, bin_spacing)
        hist, bins = np.histogram(chain, bins=bins)
        return hist, bins, bin_spacing
    

    for row_idx, vn in enumerate(variables):
        # ===== FIRST COLUMN: Chains =====
        col_idx=0
        chain_legend = []
        for n_df, df in enumerate(dfs):
            chain = df[vn].values
            runs = np.arange(0,chain.shape[0])
            
#            print('\nplot subset of trace\n')
#            chain=chain[chain.shape[0] - 400 : chain.shape[0]]
#            runs=runs[chain.shape[0] - 400 : chain.shape[0]]
            
            if n_df == 0:
                ax[row_idx,col_idx].plot(runs, chain, color='b', linewidth=0.2)
            elif n_df == 1:
                ax[row_idx,col_idx].plot(runs, chain, color='r', linewidth=0.2)
            else:
                ax[row_idx,col_idx].plot(runs, chain, color='y', linewidth=0.2)
            chain_legend.append('chain' + str(n_df + 1))
            
        if row_idx == len(variables):
            ax[row_idx,col_idx].set_xlabel('Step Number', size=14)
        elif row_idx == 0:
            ax[row_idx,col_idx].legend(chain_legend)
        ax[row_idx,col_idx].set_ylabel(vn_label_dict[vn], size=14)
        # Set extent
        ax[row_idx,col_idx].set_xlim(0, len(chain))
        
        # ===== SECOND COLUMN: Prior and posterior distributions =====
        col_idx=1
        # Prior distribution
        z_score = np.linspace(norm.ppf(0.01), norm.ppf(0.99), 100)
        if vn.startswith('obs'):
            observed_massbal = obs_list[int(vn.split('_')[1])]
            observed_error = obs_err_list[int(vn.split('_')[1])]
            x_values = observed_massbal + observed_error * z_score
            y_values = norm.pdf(x_values, loc=observed_massbal, scale=observed_error)
        elif vn == 'massbal':
            observed_massbal = obs_list[0]
            observed_error = obs_err_list[0]
            x_values = observed_massbal + observed_error * z_score
            y_values = norm.pdf(x_values, loc=observed_massbal, scale=observed_error)
            
#            mb_a = (mb_max_loss - observed_massbal) / observed_error
#            y_values = truncnorm.pdf(x_values, mb_a, np.inf, loc=observed_massbal, scale=observed_error)
            
            # Add vertical lines where mb_max_acc and mb_max_loss
            if newsetup == 1:
                ax[row_idx,col_idx].axvline(x=mb_max_acc, color='gray', linestyle='--', linewidth=1)
                ax[row_idx,col_idx].axvline(x=mb_max_loss, color='gray', linestyle='--', linewidth=1)            
        elif vn == 'precfactor': 
            if precfactor_disttype == 'lognormal':
                precfactor_lognorm_sigma = (1/input.precfactor_lognorm_tau)**0.5
                x_values = np.linspace(lognorm.ppf(1e-6, precfactor_lognorm_sigma), 
                                       lognorm.ppf(0.99, precfactor_lognorm_sigma), 100)
                y_values = lognorm.pdf(x_values, precfactor_lognorm_sigma)
            elif precfactor_disttype == 'custom':
                z_score = np.linspace(truncnorm.ppf(0.01, precfactor_a, precfactor_b),
                                      truncnorm.ppf(0.99, precfactor_a, precfactor_b), 100)
                x_values_raw = precfactor_mu + precfactor_sigma * z_score
                y_values = truncnorm.pdf(x_values_raw, precfactor_a, precfactor_b, loc=precfactor_mu,
                                         scale=precfactor_sigma)       
                x_values = prec_transformation(x_values_raw)
            elif precfactor_disttype == 'uniform':
                z_score = np.linspace(uniform.ppf(0.01), uniform.ppf(0.99), 100)
                x_values = precfactor_boundlow + z_score * (precfactor_boundhigh - precfactor_boundlow)
                y_values = uniform.pdf(x_values, loc=precfactor_boundlow, 
                                       scale=(precfactor_boundhigh - precfactor_boundlow))
        elif vn == 'tempchange':
            if tempchange_disttype == 'truncnormal':
                z_score = np.linspace(truncnorm.ppf(0.01, tempchange_a, tempchange_b),
                                      truncnorm.ppf(0.99, tempchange_a, tempchange_b), 100)
                x_values = tempchange_mu + tempchange_sigma * z_score
                y_values = truncnorm.pdf(x_values, tempchange_a, tempchange_b, loc=tempchange_mu,
                                         scale=tempchange_sigma)
            elif tempchange_disttype == 'uniform':
                z_score = np.linspace(uniform.ppf(0.01), uniform.ppf(0.99), 100)
                x_values = tempchange_boundlow + z_score * (tempchange_boundhigh - tempchange_boundlow)
                y_values = uniform.pdf(x_values, loc=tempchange_boundlow,
                                       scale=(tempchange_boundhigh - tempchange_boundlow))
        elif vn == 'ddfsnow':            
            if ddfsnow_disttype == 'truncnormal':
                z_score = np.linspace(truncnorm.ppf(0.01, ddfsnow_a, ddfsnow_b),
                                      truncnorm.ppf(0.99, ddfsnow_a, ddfsnow_b), 100)
                x_values = ddfsnow_mu + ddfsnow_sigma * z_score
                y_values = truncnorm.pdf(x_values, ddfsnow_a, ddfsnow_b, loc=ddfsnow_mu, scale=ddfsnow_sigma)
            elif ddfsnow_disttype == 'uniform':
                z_score = np.linspace(uniform.ppf(0.01), uniform.ppf(0.99), 100)
                x_values = ddfsnow_boundlow + z_score * (ddfsnow_boundhigh - ddfsnow_boundlow)
                y_values = uniform.pdf(x_values, loc=ddfsnow_boundlow,
                                       scale=(ddfsnow_boundhigh - ddfsnow_boundlow))
            
        ax[row_idx,col_idx].plot(x_values, y_values, color='k')
        #ax[row_idx,col_idx].set_xlabel(vn_label_units_dict[vn], size=14)
        if row_idx + 1 == len(variables):
            ax[row_idx,col_idx].set_xlabel('Parameter Value', size=14)
        ax[row_idx,col_idx].set_ylabel('PDF', size=14)
        
        
        
        # Ensemble/Posterior distribution                
        for n_chain, df in enumerate(dfs):
            chain = df[vn].values
            
            # gaussian distribution
            kde = gaussian_kde(chain)
            x_values_kde = x_values.copy()
            y_values_kde = kde(x_values_kde)
            chain_legend.append('posterior' + str(n_chain + 1))
            
            # Plot fitted distribution
            if n_chain == 0:
                ax[row_idx,col_idx].plot(x_values_kde, y_values_kde, color='b')
            elif n_chain == 1:
                ax[row_idx,col_idx].plot(x_values_kde, y_values_kde, color='r')
            else:
                ax[row_idx,col_idx].plot(x_values_kde, y_values_kde, color='y')
        
        # Histogram
        nbins = 50
        hist, bins, bin_spacing = calc_histogram(chain, nbins)  
        scale_hist = y_values_kde.max() / hist.max()
        hist = hist * scale_hist
        # plot histogram
        ax[row_idx,col_idx].bar(bins[1:], hist, width=bin_spacing, align='center', alpha=0.2, edgecolor='black', 
                                color='b')
        
        # Set extent
        x_extent_min = np.min([chain.min(), x_values.min()])
        x_extent_max = np.max([chain.max(), x_values.max()])
        if vn == 'precfactor':
            ax[row_idx,col_idx].set_xlim(precfactor_boundlow, precfactor_boundhigh)
        else:
            ax[row_idx,col_idx].set_xlim(x_extent_min, x_extent_max)

        # ===== COLUMN 3: Normalized autocorrelation ======
        col_idx=2
        chain_norm = chain - chain.mean()
        if chain.shape[0] <= acorr_maxlags:
            acorr_lags = chain.shape[0] - 1
        else:
            acorr_lags = acorr_maxlags
        ax[row_idx,col_idx].acorr(chain_norm, maxlags=acorr_lags)
        ax[row_idx,col_idx].set_xlim(0,acorr_lags)
        if row_idx + 1 == len(variables):
            ax[row_idx,col_idx].set_xlabel('Lag', size=14)
        ax[row_idx,col_idx].set_ylabel('autocorrelation', size=14)
        chain_neff = effective_n(ds, vn, iters, burn)
        ax[row_idx,col_idx].text(int(0.6*acorr_lags), 0.85, 'n_eff=' + str(chain_neff))
        
    # Save figure
    str_ending = ''
#    if input.new_setup == 1:
#        str_ending += '_newsetup'
#    else:
#        str_ending += '_oldsetup'

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
        elif precfactor_disttype == 'custom':
            str_ending += '_PFc'
    
    if 'ddfsnow' in variables:     
        if ddfsnow_disttype == 'truncnormal': 
            str_ending += '_DDFtn'
        elif ddfsnow_disttype == 'uniform':
            str_ending += '_DDFu'
            
    str_ending += '_TCsig' + str(input.tempchange_sigma_adj)
      
    if input.tempchange_edge_method == 'mb':
        str_ending += '_edgeMBpt' + str(int(input.tempchange_edge_mb*100)).zfill(2)
    elif input.tempchange_edge_method == 'mb_norm':
        str_ending += '_edgeMBNormpt' + str(int(input.tempchange_edge_mbnorm*100)).zfill(2)
    elif input.tempchange_edge_method == 'mb_norm_slope':
        str_ending += '_edgeSpt' + str(int(input.tempchange_edge_mbnormslope*100)).zfill(2)
    
#    str_ending += '_PF+' + str(input.precfactor_boundhigh_adj)
        
    if os.path.exists(mcmc_output_figures_fp) == False:
        os.makedirs(mcmc_output_figures_fp)        
    fig.savefig(mcmc_output_figures_fp + glacier_str + '_chains_' + str(int(iters/1000)) + 'k' + str_ending + '.png', 
                bbox_inches='tight', dpi=300)
    fig.clf()

#    # ===== LOG POSTERIOR PLOT ================================================================
#    fig, ax = plt.subplots(1, 1, squeeze=False, figsize=(6, 4))
#    fig.suptitle('log(posterior)_' + glacier_str, y=0.94)
#
#    chain_legend = []
#    for n_df, df in enumerate(dfs):
#        chain = df[vn].values
#        runs = np.arange(0,chain.shape[0])
#        
#        # LOG POSTERIOR
#        chain_massbal = df['massbal'].values
#        chain_tempchange = df['tempchange'].values
#        chain_precfactor = df['precfactor'].values
#        chain_ddfsnow = df['ddfsnow'].values
#        
#        logpost_massbal = -(1 / (2 * observed_error**2)) * (observed_massbal - chain_massbal)**2 
#        
#        if 'tempchange' in variables: 
#            if tempchange_disttype == 'truncnormal':
#                logpost_tempchange = - (1 / (2 * tempchange_sigma**2)) * (chain_tempchange - tempchange_mu)**2
#            elif tempchange_disttype == 'uniform':
#                logpost_tempchange = 0
#
#        if 'precfactor' in variables:
#            if precfactor_disttype == 'lognormal':
#                precfactor_lognorm_sigma = (1/precfactor_lognorm_tau)**2
#                logpost_precfactor = (np.exp(-1/(2 * precfactor_lognorm_sigma**2) * (np.log(chain_precfactor) - 
#                                                 precfactor_lognorm_mu)**2) * (1 / chain_precfactor))                
#            elif precfactor_disttype == 'uniform':
#                logpost_precfactor = 0
#            elif precfactor_disttype == 'custom':
#                print('\nNEED TO UPDATE CUSTOM LOG(POSTERIOR) CALCULATIONS\n')
#                logpost_precfactor = 0
#                
#        if 'ddfsnow' in variables:
#            if ddfsnow_disttype == 'truncnormal':
#                logpost_ddfsnow = - (1 / (2 * ddfsnow_sigma**2)) * (chain_ddfsnow - ddfsnow_mu)**2
#            elif ddfsnow_disttype == 'uniform':
#                logpost_ddfsnow = 0
#
#        logposterior = logpost_massbal + logpost_tempchange + logpost_precfactor + logpost_ddfsnow
#                        
#        if n_df == 0:
#            ax[0,0].plot(runs, logposterior, color='b', linewidth=0.2)
#        elif n_df == 1:
#            ax[0,0].plot(runs, logposterior, color='r', linewidth=0.2)
#        else:
#            ax[0,0].plot(runs, logposterior, color='y', linewidth=0.2)
#        chain_legend.append('chain' + str(n_df + 1))
#    
#    ax[0,0].legend(chain_legend)
#    ax[0,0].set_xlabel('Step Number', size=14)
#    ax[0,0].set_ylabel('Log(posterior)', size=14)
#    ax[0,0].set_xlim(0, len(chain))
#    
#    # Save figure            
##    fig_logposterior_fp = mcmc_output_figures_fp + 'logposterior/'
##    if os.path.exists(fig_logposterior_fp) == False:
##        os.makedirs(fig_logposterior_fp)
#    fig.savefig(mcmc_output_figures_fp + glacier_str + '_logpost_' + str(int(iters/1000)) + 'k' + str_ending + '.png', 
#                bbox_inches='tight', dpi=300)
#    fig.clf()
#        
#    # ===== PAIRWISE SCATTER PLOTS ===========================================================
#    fig, ax = plt.subplots(len(variables), len(variables), squeeze=False, figsize=(12, len(variables)*3), 
#                           gridspec_kw={'wspace':0, 'hspace':0})
#    fig.suptitle('mcmc_pairwise_scatter_' + glacier_str, y=0.94)
#
#    df = dfs[0]
#    for h, vn1 in enumerate(variables):
#        v1 = df[vn1].values
#        for j, vn2 in enumerate(variables):
#            v2 = df[vn2].values
#            if h == j:
#                # Histogram
#                hist, bins, bin_spacing = calc_histogram(v1, nbins)  
#                ax[h,j].bar(bins[1:], hist, width=bin_spacing, align='center', alpha=0.2, edgecolor='black', color='b')
#                if h == 0:
#                    ax[h,j].text(bins[-1], int(0.98*hist.max()), 'n='+str(int(len(v1)/1000)) + 'k', fontsize=14, 
#                                 verticalalignment='center', horizontalalignment='right')
#            elif h > j:
#                # Scatterplot
#                subset_idx1 = int(v1.shape[0]/3)
#                subset_idx2 = 2*int(v1.shape[0]/3)
#                v1_subset1 = v1[0:subset_idx1]
#                v1_subset2 = v1[subset_idx1:subset_idx2]
#                v1_subset3 = v1[subset_idx2:]
#                v2_subset1 = v2[0:subset_idx1]
#                v2_subset2 = v2[subset_idx1:subset_idx2]
#                v2_subset3 = v2[subset_idx2:]
#                #ax[h,j].plot(v2, v1, 'o', mfc='none', mec='black')
#                ax[h,j].plot(v2_subset1, v1_subset1, 'o', mfc='none', mec='b', ms=1, alpha=1)
#                ax[h,j].plot(v2_subset2, v1_subset2, 'o', mfc='none', mec='r', ms=1, alpha=1)
#                ax[h,j].plot(v2_subset3, v1_subset3, 'o', mfc='none', mec='y', ms=1, alpha=1)
#            else:
#                # Correlation coefficient
#                slope, intercept, r_value, p_value, std_err = linregress(v2, v1)
#                text2plot = (vn_label_nounits_dict[vn1] + '/\n' + vn_label_nounits_dict[vn2] + '\n$R^2$=' +
#                             '{:.2f}'.format((r_value**2)))
#                ax[h,j].text(0.5, 0.5, text2plot, fontsize=14, verticalalignment='center', horizontalalignment='center')
#
#            # Only show x-axis on bottom and y-axis on left
#            if h + 1 < len(variables):
#                ax[h,j].xaxis.set_major_formatter(plt.NullFormatter())
#            if h == j or j != 0:
#                ax[h,j].yaxis.set_major_formatter(plt.NullFormatter())
#            
#            # Add labels
#            if j == 0:
#                if h > 0:
#                    ax[h,j].set_ylabel(vn_label_dict[vn1], fontsize=14)
#                else:
#                    ax[h,j].set_ylabel('Mass balance\n ', fontsize=14)
#            if h + 1 == len(variables):            
#                ax[h,j].set_xlabel(vn_label_dict[vn2], fontsize=14)
#
##    fig_autocor_fp = mcmc_output_figures_fp + 'autocorrelation/'
##    if os.path.exists(fig_autocor_fp) == False:
##        os.makedirs(fig_autocor_fp)
#    fig.savefig(mcmc_output_figures_fp + glacier_str + '_scatter_' + str(int(iters/1000)) + 'k' + str_ending + '.png', 
#                bbox_inches='tight', dpi=300)


def plot_mb_vs_parameters(tempchange_iters, precfactor_iters, ddfsnow_iters, modelparameters, glacier_rgi_table, 
                          glacier_area_t0, icethickness_t0, width_t0, elev_bins, glacier_gcm_temp, glacier_gcm_prec, 
                          glacier_gcm_elev, glacier_gcm_lrgcm, glacier_gcm_lrglac, dates_table, observed_massbal, 
                          observed_error, mb_max_acc, mb_max_loss, tempchange_max_acc, tempchange_max_loss, 
                          option_areaconstant=0):
    """
    Plot the mass balance [mwea] versus all model parameters to see how parameters effect mass balance
    """
    #%%
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
                (glac_bin_temp, glac_bin_prec, glac_bin_acc, glac_bin_refreeze, glac_bin_snowpack, glac_bin_melt,
                 glac_bin_frontalablation, glac_bin_massbalclim, glac_bin_massbalclim_annual, glac_bin_area_annual,
                 glac_bin_icethickness_annual, glac_bin_width_annual, glac_bin_surfacetype_annual,
                 glac_wide_massbaltotal, glac_wide_runoff, glac_wide_snowline, glac_wide_snowpack,
                 glac_wide_area_annual, glac_wide_volume_annual, glac_wide_ELA_annual) = (
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


    mb_vs_parameters['massbal_norm'] = (mb_vs_parameters['massbal'] - mb_max_loss) / (mb_max_acc - mb_max_loss)
    mb_vs_parameters['tempbias_norm'] = ((mb_vs_parameters['tempbias'] - tempchange_max_acc) / 
                                         (tempchange_max_loss - tempchange_max_acc))

    # Compute the slope
    mb_subset = mb_vs_parameters[(mb_vs_parameters.loc[:,'precfactor'] == 1)]
    mb_subset = mb_subset[mb_subset.loc[:,'ddfsnow'] == 0.0041]
    mb_subset.reset_index(drop=True, inplace=True)
    
    # Compute slope based on wider step to avoid erroneous slopes due to rounding errors
    slope_step = 1
    tc_iter_step = np.round(abs(mb_subset.loc[0,'tempbias'] - mb_subset.loc[1,'tempbias']),2)
    slope_idx = int(slope_step / tc_iter_step / 2)
    if slope_idx < 1:
        slope_idx = int(1)
        
    # Extend mb_subset so slope can be calculated at all tempchange values
    mb_subset_ext = np.tile(mb_subset.loc[0].values, (slope_idx,1))
    mb_subset_ext = np.concatenate((mb_subset_ext, mb_subset.values))
    mb_subset_ext = np.concatenate((mb_subset_ext, np.tile(mb_subset.loc[mb_subset.shape[0]-1].values, (slope_idx,1))))
    mb_subset_ext[0:slope_idx,1] = (
            np.arange(mb_subset_ext[slope_idx,1] - tc_iter_step, 
                      mb_subset_ext[slope_idx,1] - tc_iter_step * (slope_idx + 1), -tc_iter_step)[::-1])
    mb_subset_ext[-slope_idx:,1] = (
            np.arange(mb_subset_ext[-slope_idx,1] + tc_iter_step,  
                      mb_subset_ext[-slope_idx,1] + tc_iter_step * (slope_idx + 1), tc_iter_step))
    mb_subset_ext[:,5] = (mb_subset_ext[:,1] - tempchange_max_acc) / (tempchange_max_loss - tempchange_max_acc)
    mb_subset_ext = pd.DataFrame(mb_subset_ext, columns=['precfactor', 'tempbias', 'ddfsnow', 'massbal', 'massbal_norm', 
                                                         'tempbias_norm'])  
    mb_slope = pd.DataFrame()
    mb_slope['tempbias'] = mb_subset['tempbias'].values
    mb_slope['tempbias_norm'] = mb_subset['tempbias_norm'].values
    tempbias_1 = mb_subset_ext.loc[0:mb_subset_ext.shape[0]-(2*slope_idx+1),'tempbias'].values
    tempbias_2 = mb_subset_ext.loc[2*slope_idx:mb_subset_ext.shape[0],'tempbias'].values
    tempbiasnorm_1 = mb_subset_ext.loc[0:mb_subset_ext.shape[0]-(2*slope_idx+1),'tempbias_norm'].values
    tempbiasnorm_2 = mb_subset_ext.loc[2*slope_idx:mb_subset_ext.shape[0],'tempbias_norm'].values
    mb_1 = mb_subset_ext.loc[0:mb_subset_ext.shape[0]-(2*slope_idx+1),'massbal'].values
    mb_2 = mb_subset_ext.loc[2*slope_idx:mb_subset_ext.shape[0],'massbal'].values
    mbnorm_1 = mb_subset_ext.loc[0:mb_subset_ext.shape[0]-(2*slope_idx+1),'massbal_norm'].values
    mbnorm_2 = mb_subset_ext.loc[2*slope_idx:mb_subset_ext.shape[0],'massbal_norm'].values
    mb_slope['slope'] = (mb_2 - mb_1) / (tempbias_2 - tempbias_1)
    mb_slope['slope_norm'] = (mbnorm_2 - mbnorm_1) / (tempbiasnorm_2 - tempbiasnorm_1)

    # get glacier number
    if glacier_rgi_table.O1Region >= 10:
        glacier_RGIId = glacier_rgi_table['RGIId'][6:]
    else:
        glacier_RGIId = glacier_rgi_table['RGIId'][7:]
    
    np.savetxt(input.output_filepath + 'cal_opt2/figures/' + glacier_RGIId + '_mb_vs_parameters.csv', 
               mb_vs_parameters, delimiter=',')

    # Plot the normalized mass balance versus tempchange
    fig, ax = plt.subplots(figsize=(6,4))
    
    # Subset data for each precfactor
    prec_linedict = {0.5:'--',
                     1:'-',
                     1.5:':',
                     2:':',
                     20: '-.'}
    ddfsnow_colordict = {0.0031:'b',
                         0.0041:'k',
                         0.0051:'r'}

    for precfactor in [1]:
        modelparameters[2] = precfactor
        mb_vs_parameters_subset = mb_vs_parameters.loc[mb_vs_parameters.loc[:,'precfactor'] == precfactor]
        for ddfsnow in [0.0041]:
            mb_vs_parameters_plot =  mb_vs_parameters_subset.loc[mb_vs_parameters_subset.loc[:,'ddfsnow'] == ddfsnow]
            ax.plot(mb_vs_parameters_plot.loc[:,'tempbias'], mb_vs_parameters_plot.loc[:,'massbal_norm'], 
                    linestyle=prec_linedict[precfactor], color=ddfsnow_colordict[ddfsnow], label='model')  
#            ax.plot(mb_vs_parameters_plot.loc[:,'tempbias_norm'], mb_vs_parameters_plot.loc[:,'massbal_norm'], 
#                    linestyle=prec_linedict[precfactor], color=ddfsnow_colordict[ddfsnow], label='model')  
    
    # Add horizontal line of mass balance observations
    ax.axhline((observed_massbal - mb_max_loss) / (mb_max_acc - mb_max_loss), color='gray', linewidth=2, label='obs')    
#    ax.axhline(observed_massbal - 1.96*observed_error, color='gray', linewidth=1) 
#    ax.axhline(observed_massbal + 1.96*observed_error, color='gray', linewidth=1)         
#    ax.set_xlim(np.min(tempchange_iters), np.max(tempchange_iters))
    ax.set_xlim(tempchange_max_acc,tempchange_max_loss)
#    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    
    
    # Add slope on secondary axis
    ax2 = ax.twinx()
    ax2.plot(mb_slope.loc[:,'tempbias'], mb_slope.loc[:,'slope_norm'], color='yellow', label='slope')
#    ax2.plot(mb_slope.loc[:,'tempbias_norm'], mb_slope.loc[:,'slope_norm'], color='yellow', label='slope')
    ax2.set_ylim(-3,0)
    
    # Labels
    ax.set_title('Normalized Mass Balance vs Tempbias' + glacier_RGIId)
    ax.set_xlabel('Tempbias [degC]', fontsize=14)
    ax.set_ylabel('Normalized Mass Balance [-]', fontsize=14)
    ax2.set_ylabel('Slope', color='yellow')
    
    # added these three lines
    ax.legend(loc='lower left', frameon=False)
    fig.savefig(input.output_filepath + 'cal_opt2/figures/' + glacier_RGIId + '_mb_vs_parameters_norm.png', 
                bbox_inches='tight', dpi=300)  
    

    # get glacier number
    if glacier_rgi_table.O1Region >= 10:
        glacier_RGIId = glacier_rgi_table['RGIId'][6:]
    else:
        glacier_RGIId = glacier_rgi_table['RGIId'][7:]
    
    np.savetxt(input.output_filepath + 'cal_opt2/figures/' + glacier_RGIId + '_mb_vs_parameters.csv', 
               mb_vs_parameters, delimiter=',')

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
    ax.axhline(observed_massbal, color='gray', linewidth=2)    
    ax.axhline(observed_massbal - 1.96*observed_error, color='gray', linewidth=1) 
    ax.axhline(observed_massbal + 1.96*observed_error, color='gray', linewidth=1)         
    ax.set_xlim(np.min(tempchange_iters), np.max(tempchange_iters))
#    ax.set_ylim(-6,2)
    
    # Add slope on secondary axis
    ax2 = ax.twinx()
    ax2.plot(mb_slope.loc[:,'tempbias'], mb_slope.loc[:,'slope'], color='yellow')
    
    # Labels
    ax.set_title('Mass balance versus Parameters ' + glacier_RGIId)
    ax.set_xlabel('Tempbias [degC]', fontsize=14)
    ax.set_ylabel('Mass balance [mwea]', fontsize=14)
    ax2.set_ylabel('Slope', color='yellow')
#    ax2.set_ylim(-0.5,0)
    
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
        leg_names.append('DDFsnow ' + str(np.round(ddfsnow,4)))
    ax.legend(leg_lines, leg_names, loc='lower left', frameon=False)
    fig.savefig(input.output_filepath + 'cal_opt2/figures/' + glacier_RGIId + '_mb_vs_parameters.png', 
                bbox_inches='tight', dpi=300)    
    

def plot_mc_results2(netcdf_fn, glacier_cal_data, burns=[0,1000,3000,5000],
                     plot_res=1000, distribution_type='truncnormal'):
    """
    Plot gelman-rubin statistic, effective_n (autocorrelation with lag
    100) and markov chain error plots.

    Takes the output from the Markov Chain model and plots the results
    for the mass balance, temperature change, precipitation factor,
    and degree day factor of snow.  Also, outputs the plots associated
    with the model.

    Parameters
    ----------
    netcdf_fn : str
        Netcdf of MCMC methods with chains of model parameters
    iters : int
        Number of iterations associated with the Markov Chain
    burn : list of ints
        List of burn in values to plot for Gelman-Rubin stats
    plot_res: int
        Interval of points for which GR and MCerror statistic are calculated.
        (Lower value leads to higher plot resolution)
    glacier_RGIId_float : str
    precfactor_mu : float
        Mean of precipitation factor (default assigned from input)
    tempchange_mu : float
        Mean of temperature change (default assigned from input)
    ddfsnow_mu : float
        Mean of degree day factor of snow (default assigned from input)

    Returns
    -------
    .png files
        Saves two figures of (1) trace, histogram, and autocorrelation, and (2) pair-wise scatter plots.
    """
    # Open dataset
    ds = xr.open_dataset(netcdf_fn)

    # Extract calibration information needed for priors
    # Variables to plot
    variables = ds.mp.values[:].tolist()
    for i in parameters_all:
        if i in variables:
            variables.remove(i)
    variables.extend(parameters)
    # Observations data
    obs_type_list = []
    for x in range(glacier_cal_data.shape[0]):
        cal_idx = glacier_cal_data.index.values[x]
        obs_type = glacier_cal_data.loc[cal_idx, 'obs_type']
        obs_type_list.append(obs_type)

    # Titles for plots
    vn_title_dict = {}
    for n, vn in enumerate(variables):
        if vn.startswith('obs'):
            if obs_type_list[n].startswith('mb'):
                vn_title_dict[vn] = 'Mass Balance ' + str(n)
        elif vn == 'massbal':
            vn_title_dict[vn] = 'Mass Balance'
        elif vn == 'precfactor':
            vn_title_dict[vn] = 'Precipitation Factor'
        elif vn == 'tempchange':
            vn_title_dict[vn] = 'Temperature Bias'
        elif vn == 'ddfsnow':
            vn_title_dict[vn] = 'DDF Snow'

    # get variables and burn length for dimension
    v_len = len(variables)
    b_len = len(burns)
    c_len = len(ds.mp_value)
    no_chains = len(ds.chain)


    # ====================== GELMAN-RUBIN PLOTS ===========================

    plt.figure(figsize=(v_len*4, 2))
    plt.subplots_adjust(wspace=0.3, hspace=0.5)
    plt.suptitle('Gelman-Rubin Statistic vs Number of MCMC Steps', y=1.10)

    for v_count, vn in enumerate(variables):

        plt.subplot(1, v_len, v_count+1)

        for b_count, burn in enumerate(burns):

            plot_list = list(range(burn+plot_res, c_len+plot_res, plot_res))
            gr_list = [gelman_rubin(ds, vn, pt, burn) for pt in plot_list]

            # plot GR
            plt.plot(plot_list, gr_list, label='Burn-In ' + str(burn))

            # plot horizontal line for benchmark
            plt.axhline(1.01, color='black', linestyle='--', linewidth=1)

            if v_count == 0:
                plt.ylabel('Gelman-Rubin Value', size=10)

            if b_count == 0:
                plt.title(vn_title_dict[vn], size=10)

            if v_count == v_len-1:
                plt.legend()

            # niceties
            plt.xlabel('Step Number', size=10)

    # Save figure
    plt.savefig(mcmc_output_figures_fp + glacier_str + '_' + distribution_type +
                '_gelman-rubin' + '_plots_' + str(no_chains) + 'chain_' +
                str(c_len) + 'iter' + '.png', bbox_inches='tight')

    # ====================== MC ERROR PLOTS ===========================

    plt.figure(figsize=(v_len*4, 2))
    plt.subplots_adjust(wspace=0.3, hspace=0.5)
    plt.suptitle('MC Error as Percentage of Mean vs Number of MCMC Steps', y=1.10)

    for v_count, vn in enumerate(variables):

        plt.subplot(1, v_len, v_count+1)

        # points to plot at
        plot_list = list(range(0, c_len+plot_res, plot_res))

        # find mean
        total_mean = abs(np.mean(ds['mp_value'].sel(chain=0, mp=vn).values))

        mce_list = []
        #mean_list = []

        # calculate mc error and mean at each point
        for pt in plot_list:

            mce, mean = MC_error(ds, vn, iters=pt)
            mce_list.append(mce)
            #mean_list.append(abs(mean) / 100)

        # plot
        plt.plot(plot_list, mce_list / (total_mean / 100))
        plt.axhline(1, color='orange', label='1% of Mean', linestyle='--')
        plt.axhline(3, color='green', label='3% of Mean', linestyle='--')

        if v_count == 0:
            plt.ylabel('MC Error [% of mean]', size=10)

        if v_count == v_len-1:
            plt.legend()

        # niceties
        plt.xlabel('Step Number', size=10)
        plt.title(vn_title_dict[vn], size=10)

    # Save figure
    plt.savefig(mcmc_output_figures_fp + glacier_str + '_' + distribution_type +
                '_mc-error' + '_plots_' + str(no_chains) + 'chain_' +
                str(c_len) + 'iter' + '.png', bbox_inches='tight')

    # ====================== EFFECTIVE_N PLOTS ===========================

    plt.figure(figsize=(v_len*4, 2))
    plt.subplots_adjust(wspace=0.3, hspace=0.5)
    plt.suptitle('Effective Sample Size vs Number of MCMC Steps', y=1.10)

    # get dataframe
    #df = ds['mp_value'].sel(chain=0).to_pandas()

    for v_count, vn in enumerate(variables):

        plt.subplot(1, v_len, v_count+1)

        for b_count, burn in enumerate(burns):

            # points to plot at
            plot_list = list(range(burn+plot_res, c_len+plot_res, plot_res))
            #en_list = [effective_n(df[burn:pt], vn) for pt in plot_list]
            en_list = [effective_n(ds, vn=vn, iters=pt, burn=burn) for pt in plot_list]
            # plot
            plt.plot(plot_list, en_list, label='Burn-In ' + str(burn))

            if v_count == 0:
                plt.ylabel('Effective Sample Size', size=10)

            if v_count == v_len-1:
                plt.legend()

            if b_count == 0:
                plt.title(vn_title_dict[vn], size=10)

            # niceties
            plt.xlabel('Step Number', size=10)
            plt.xlabel('Step Number', size=10)

    # Save figure
    plt.savefig(mcmc_output_figures_fp + glacier_str + '_' + distribution_type +
                '_effective-n' + '_plots_' + str(no_chains) + 'chain_' +
                str(c_len) + 'iter' + '.png', bbox_inches='tight')

    '''
    Writes a csv table that lists MCMC assessment values for
    each glacier (represented by a netcdf file.

    Writes out the values of effective_n (autocorrelation with
    lag 100), Gelman-Rubin Statistic, MC_error.

    Parameters
    ----------
    netcdf_fn : str
        Netcdf of MCMC methods with chains of model parameters
    vn : str
        Name of variable (massbal, ddfsnow, precfactor, tempchange)
    region_no : int
        number of the glacier region (13, 14 or 15)
    iters : int
        Number of iterations associated with the Markov Chain
    burn : list of ints
        List of burn in values to plot for Gelman-Rubin stats

    Returns
    -------
    dfs : list of pandas.DataFrame
        dataframes containing statistical information for all glaciers
    .csv files
        Saves tables to csv file.

    '''
    # hard code some variable names (dirty solution)
    variables = ['massbal', 'precfactor', 'tempchange', 'ddfsnow']

    # find all netcdf files (representing glaciers)
    filelist = glob.glob(mcmc_output_netcdf_fp + str(region_no) + '*.nc')[0:10]

    # list of dataframes to return
    dfs = []

    for vn in variables:

        # create lists of each value
        glac_no = []
        effective_n_list = []
        gelman_rubin_list = []
        mc_error = []

        # find all netcdf files (representing glaciers)
        filelist = glob.glob(mcmc_output_netcdf_fp + str(region_no) + '*.nc')

        # iterate through each glacier
        for netcdf in filelist:
            # open dataset
            ds = xr.open_dataset(netcdf)

            # find values for this glacier and append to lists
            glac_no.append(netcdf[-11:-3])

            effective_n_list.append(effective_n(ds, vn=vn, iters=iters, burn=burn))
            gelman_rubin_list.append(gelman_rubin(ds, vn=vn, iters=iters, burn=burn))
            mc_error.append(MC_error(ds, vn=vn, iters=iters, burn=burn)[0])

        # create dataframe
        data = {'Glacier': glac_no,
                'Effective N' : effective_n_list,
                'MC Error' : mc_error,
                'Gelman-Rubin' : gelman_rubin_list}
        df = pd.DataFrame(data)
        df.set_index('Glacier', inplace=True)
        #df = pd.DataFrame(data, index = {'Glacier': glac_no})

        # save csv
        df.to_csv(mcmc_output_csv_fp + 'region' + str(region_no) + '_' +
                  str(iters) + 'iterations_' + str(burn) + 'burn_' + str(vn) + '.csv')

        dfs.append(df)

    return dfs


def write_table(region=15, iters=1000, burn=0):
    '''
    Writes a csv table that lists MCMC assessment values for
    each glacier (represented by a netcdf file.

    Writes out the values of effective_n (autocorrelation with
    lag 100), Gelman-Rubin Statistic, MC_error.

    Parameters
    ----------
    region : int
        number of the glacier region (13, 14 or 15)
    iters : int
        Number of iterations associated with the Markov Chain
    burn : list of ints
        List of burn in values to plot for Gelman-Rubin stats

    Returns
    -------
    dfs : list of pandas.DataFrame
        dataframes containing statistical information for all glaciers
    .csv files
        Saves tables to csv file.

    '''

    dfs=[]

    variables = ['massbal', 'precfactor', 'tempchange', 'ddfsnow']

    # find all netcdf files (representing glaciers)
    filelist = glob.glob(mcmc_output_netcdf_fp + str(region) + '*.nc')

    for vn in variables:

        # create lists of each value
        glac_no = []
        effective_n_list = []
        gelman_rubin_list = []
        mc_error = []


        # iterate through each glacier
        for netcdf in filelist:
            print(netcdf)
            # open dataset
            ds = xr.open_dataset(netcdf)

            # find values for this glacier and append to lists
            glac_no.append(netcdf[-11:-3])
            effective_n_list.append(effective_n(ds, vn=vn, iters=iters, burn=burn))
            # test if multiple chains exist
            if len(ds.chain) > 1:
                gelman_rubin_list.append(gelman_rubin(ds, vn=vn, iters=iters, burn=burn))
            mc_error.append(MC_error(ds, vn=vn, iters=iters, burn=burn)[0])

            ds.close()

        mean = abs(np.mean(mc_error))
        mc_error /= mean

        # create dataframe
        data = {'Glacier': glac_no,
                'Effective N' : effective_n_list,
                'MC Error' : mc_error}
        if len(gelman_rubin_list) > 0:
            data['Gelman-Rubin'] = gelman_rubin_list
        df = pd.DataFrame(data)
        df.set_index('Glacier', inplace=True)

        # save csv
        df.to_csv(mcmc_output_csv_fp + 'region' + str(region) + '_' +
                  str(iters) + 'iterations_' + str(burn) + 'burn_' + str(vn) + '.csv')

        dfs.append(df)

    return dfs


def plot_histograms(iters, burn, region=15, dfs=None):
    '''
    Plots histograms to assess mcmc chains for groups of glaciers

    Plots histograms of effective_n, gelman-rubin and mc error for
    the given number of iterations and burn-in and the given variable.

    For this function to work, the appropriate csv file must have already
    been created.

    Parameters
    ----------
    dfs : list of pandas.DataFrame
        list of dataframes containing glacier information to be plotted. If
        none, looks for appropriate csv file
    vn : str
        Name of variable (massbal, ddfsnow, precfactor, tempchange)
    iters : int
        Number of iterations associated with the Markov Chain
    burn : list of ints
        List of burn in values to plot for Gelman-Rubin stats

    Returns
    -------
    .png files
        Saves images to 3 png files.

    '''

    # hard code some variable names (dirty solution)
    variables = ['massbal', 'precfactor', 'tempchange', 'ddfsnow']
    vn_title_dict = {'massbal':'Mass Balance',
                     'precfactor':'Precipitation Factor',
                     'tempchange':'Temperature Bias',
                     'ddfsnow':'DDF Snow'}
    metrics = ['Gelman-Rubin', 'MC Error', 'Effective N']

    vn_df_dict = {}

    # read csv files
    for vn in variables:
        vn_df_dict[vn] = pd.read_csv(mcmc_output_csv_fp + 'region' +
                                     str(region) + '_' + str(iters) +
                                     'iterations_' + str(burn) + 'burn_' +
                                     str(vn) + '.csv')


    # get variables and burn length for dimension
    v_len = len(variables)

    # create plot for each metric
    for metric in metrics:

        plt.figure(figsize=(v_len*4, 3))
        plt.subplots_adjust(wspace=0.3, hspace=0.3)

        if metric is 'MC Error':
            plt.suptitle(metric + ' (as percentage of mean) Histrogram ' +
                         str(iters) + ' iterations ' + str(burn) + ' burn-in', y=1.05,
                         fontsize=14)
        else:
            plt.suptitle(metric + ' Histrogram ' +
                         str(iters) + ' iterations ' + str(burn) + ' burn-in', y=1.10,
                         fontsize=14)

        # create subplot for each variable
        for v_count, vn in enumerate(variables):

            df = vn_df_dict[vn]

            # plot histogram
            plt.subplot(1, v_len, v_count+1)
            n, bins, patches = plt.hist(x=df[metric], bins=30, alpha=.4, edgecolor='black',
                                        color='#0504aa')


            # niceties
            plt.title(vn_title_dict[vn])

            if v_count == 0:
                plt.ylabel('Frequency')


        # Save figure
        plt.savefig(mcmc_output_hist_fp + 'region' + str(region) + '_' + str(iters) +
                    'iterations_' + str(burn) + 'burn_' + str(metric.replace(' ','_')) + '.png')


def plot_histograms_2(iters, burn, region=15, dfs=None):
    '''
    Plots histograms to assess mcmc chains for groups of glaciers.
    Puts them all in one image file.

    Plots histograms of effective_n, gelman-rubin and mc error for
    the given number of iterations and burn-in and the given variable.

    For this function to work, the appropriate csv file must have already
    been created.

    Parameters
    ----------
    dfs : list of pandas.DataFrame
        list of dataframes containing glacier information to be plotted. If
        none, looks for appropriate csv file
    iters : int
        Number of iterations associated with the Markov Chain
    burn : list of ints
        List of burn in values to plot for Gelman-Rubin stats

    Returns
    -------
    .png files
        Saves images to 3 png files.

    '''

    # hard code some variable names (dirty solution)
    variables = ['massbal', 'precfactor', 'tempchange', 'ddfsnow']
    vn_title_dict = {'massbal':'Mass Balance',
                     'precfactor':'Precipitation Factor',
                     'tempchange':'Temperature Bias',
                     'ddfsnow':'DDF Snow'}

    test = pd.read_csv(mcmc_output_csv_fp + 'region' +
                                     str(region) + '_' + str(iters) +
                                     'iterations_' + str(burn) + 'burn_' +
                                     str('massbal') + '.csv')

    # determine whether Gelman-Rubin has been computed
    if 'Gelman-Rubin' in test.columns:
        metrics = ['Gelman-Rubin', 'MC Error', 'Effective N']
    else:
        metrics = ['MC Error', 'Effective N']

    # hard code font sizes
    ticks=10
    suptitle=16
    title=14
    label=12
    plotline=3
    legend=10

    # bins and ticks
    if iters==15000:
        tbins = np.arange(1.00, 1.06, 0.002)
        pbins = np.arange(1.0, 1.006, 0.0002)
        dbins = np.arange(1.0, 1.006, 0.0002)
        mbins = np.arange(1.0, 1.006, 0.0002)
        mcbins = np.arange(0, 3, 0.1)
        grticks = np.arange(0, 30, 5)
        mcticks = np.arange(0, 19, 4)
        nticks = np.arange(0, 19, 4)
    else:
        tbins = np.arange(1.00, 1.12, 0.004)
        pbins = np.arange(1.00, 1.12, 0.004)
        dbins = np.arange(1, 1.018, 0.006)
        mbins = np.arange(1, 1.012, 0.0004)
        mcbins = np.arange(0, 4, 0.125)
        grticks = np.arange(0, 30, 5)
        mcticks = np.arange(0, 19, 4)
        nticks = np.arange(0, 19, 4)


    vn_df_dict = {}
    # read csv files
    for vn in variables:
        vn_df_dict[vn] = pd.read_csv(mcmc_output_csv_fp + 'region' +
                                     str(region) + '_' + str(iters) +
                                     'iterations_' + str(burn) + 'burn_' +
                                     str(vn) + '.csv')

    # get variables and burn length for dimension
    v_len = len(variables)
    m_len = len(metrics)

    # create figure
    #fig=plt.figure(figsize=(6.5, 4))
    fig = plt.figure(figsize=(v_len*5, m_len*3.5), dpi=72)
    plt.subplots_adjust(wspace=0.3, hspace=0.5)

    # write title
    plt.suptitle('MC Metrics Assessment Histograms ' +
                 str(iters) + ' iterations ' + str(burn) + ' burn-in',
                 fontsize=18, y=0.97)

    #create subplot for each metric
    for m_count, metric in enumerate(metrics):

        # create subplot for each variable
        for v_count, vn in enumerate(variables):

            df = vn_df_dict[vn]

            # plot histogram
            ax = plt.subplot(m_len, v_len, v_len*m_count+v_count+1)
            ax2 = ax.twinx()

            # create uniform bins based on matric
            if metric == 'Gelman-Rubin':
                if vn == 'tempchange':
                    bins = tbins
                elif vn == 'precfactor':
                    bins = pbins
                elif vn == 'ddfsnow':
                    bins = dbins
                elif vn == 'massbal':
                    bins = mbins
            elif metric == 'MC Error':
                bins = mcbins
            elif metric == 'Effective N':
                bins = 30

            #print('bins: ', bins)

            # compute histogram and change to percentage of glaciers
            hist, bins = np.histogram(a=df[metric], bins=bins)
            hist = hist * 100.0 / hist.sum()

            # plot histogram
            ax.bar(x=bins[1:], height=hist, width=(bins[1]-bins[0]), align='center',
                   alpha=.4, edgecolor='black', color='#0504aa')

            # create uniform bins based on metric
            if metric == 'Gelman-Rubin':
                ax.set_yticks(grticks)
            elif metric == 'MC Error':
                ax.set_yticks(mcticks)
            elif metric == 'Effective N':
                ax.set_yticks(nticks)

            # find cumulative percentage and plot it
            cum_hist = [hist[0:i].sum() for i in range(len(hist))]
            ax2.plot(bins[:-1], cum_hist, color='#ff6600',
                     linewidth=plotline, label='Cumulative\nPercentage')
            ax2.set_yticks(np.arange(0, 110, 20))

            # set tick sizes
            ax.tick_params(labelsize=ticks)
            ax2.tick_params(labelsize=ticks)

            # niceties
            if m_count == 0:
                plt.title(vn_title_dict[vn], fontsize=title)

            # axis labels
            if v_count == 0:
                ax.set_ylabel('Percentage of Glaciers [%]', fontsize=label, labelpad=10)
            if v_count ==3:
                ax2.set_ylabel('Cumulative Percentage [%]', fontsize=label, rotation = 270, labelpad=35)
            if metric=='MC Error':
                ax.set_xlabel(metric + ' (as percentage of mean)', fontsize=label)
            else:
                ax.set_xlabel(metric + ' value', fontsize=label)

            # legend
            if v_count==3 and m_count==0:
                ax2.legend(loc='center right', fontsize=legend)

    # Save figure
    plt.savefig(mcmc_output_hist_fp + 'region' + str(region) + '_' + str(iters) +
                'iterations_' + str(burn) + 'burn_all.png')


#%% Find files
# ===== LOAD CALIBRATION DATA =====
rgi_glac_number = []

#mcmc_output_netcdf_fp = mcmc_output_netcdf_fp + 'single_obs_inlist/'

##for i in os.listdir(mcmc_output_netcdf_fp):
#for i in ['13.00001.nc']:
#    glacier_str = i.replace('.nc', '')
#    if glacier_str.startswith(str(input.rgi_regionsO1[0])):
#        rgi_glac_number.append(glacier_str.split('.')[1])
#rgi_glac_number = sorted(rgi_glac_number)
rgi_glac_number = input.rgi_glac_number


# Glacier RGI data
main_glac_rgi = modelsetup.selectglaciersrgitable(rgi_regionsO1=input.rgi_regionsO1, rgi_regionsO2 = 'all',
                                                  rgi_glac_number=rgi_glac_number)
# Glacier hypsometry [km**2], total area
main_glac_hyps = modelsetup.import_Husstable(main_glac_rgi, input.rgi_regionsO1, input.hyps_filepath,
                                             input.hyps_filedict, input.hyps_colsdrop)
# Ice thickness [m], average
main_glac_icethickness = modelsetup.import_Husstable(main_glac_rgi, input.rgi_regionsO1, input.thickness_filepath, 
                                                     input.thickness_filedict, input.thickness_colsdrop)
main_glac_hyps[main_glac_icethickness == 0] = 0
# Width [km], average
main_glac_width = modelsetup.import_Husstable(main_glac_rgi, input.rgi_regionsO1, input.width_filepath,
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

#%%
# Select dates including future projections
#dates_table_nospinup = modelsetup.datesmodelrun(startyear=input.startyear, endyear=input.endyear, spinupyears=0)
dates_table_nospinup = modelsetup.datesmodelrun(startyear=2000, endyear=2018, spinupyears=0)
# Calibration data
cal_data = pd.DataFrame()
for dataset in cal_datasets:
    cal_subset = class_mbdata.MBData(name=dataset, rgi_regionO1=input.rgi_regionsO1[0])
    cal_subset_data = cal_subset.retrieve_mb(main_glac_rgi, main_glac_hyps, dates_table_nospinup)
    cal_data = cal_data.append(cal_subset_data, ignore_index=True)
cal_data = cal_data.sort_values(['glacno', 't1_idx'])
cal_data.reset_index(drop=True, inplace=True)
#%%

# ===== PROCESS EACH NETCDF FILE =====
mb_compare_cols = ['RGIId', 'glacno', 'obs_mb_mwea', 'max_loss_mwea', 'max_acc_mwea', 'mod_mb_mwea', 'mb_obs_max', 
                   'PF_low', 'PF_high', 'TC_mu', 'TC_sigma', 'TC_low', 'TC_high']
mb_compare = pd.DataFrame(np.zeros((main_glac_rgi.shape[0], len(mb_compare_cols))), columns=mb_compare_cols)
mb_compare[:] = np.nan
mb_compare['RGIId'] = main_glac_rgi['RGIId']
mb_compare['glacno'] = main_glac_rgi['glacno']
mb_compare['obs_mb_mwea'] = cal_data.mb_mwe / (cal_data.t2 - cal_data.t1)
# Add maximum loss based on glacier volume
mb_compare['max_loss_mwea'] = ((-1 * (main_glac_hyps * main_glac_icethickness).sum(axis=1) / main_glac_hyps.sum(axis=1) 
                               * input.density_ice / input.density_water).values / (cal_data.t2 - cal_data.t1).values)

#%%
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
    ds = xr.open_dataset(mcmc_output_netcdf_fp + glacier_str + '.nc')
    df = pd.DataFrame(ds['mp_value'].values[:,:,0], columns=ds.mp.values)
    mb_compare.loc[n,'mod_mb_mwea'] = df.massbal.mean()    
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
    
    # OLD SETUP
    tempchange_mu = input.tempchange_mu
    tempchange_sigma = input.tempchange_sigma
    tempchange_boundlow = input.tempchange_boundlow
    tempchange_boundhigh = input.tempchange_boundhigh
    tempchange_start = tempchange_mu
        
    # NEW SETUP
    if input.new_setup == 1:
        #%%
        def mb_mwea_calc(modelparameters, option_areaconstant=1):
            """
            Run the mass balance and calculate the mass balance [mwea]
            """
            # Mass balance calculations
            (glac_bin_temp, glac_bin_prec, glac_bin_acc, glac_bin_refreeze, glac_bin_snowpack, glac_bin_melt, 
             glac_bin_frontalablation, glac_bin_massbalclim, glac_bin_massbalclim_annual, glac_bin_area_annual, 
             glac_bin_icethickness_annual, glac_bin_width_annual, glac_bin_surfacetype_annual, 
             glac_wide_massbaltotal, glac_wide_runoff, glac_wide_snowline, glac_wide_snowpack, 
             glac_wide_area_annual, glac_wide_volume_annual, glac_wide_ELA_annual) = (
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
        
        #%%
        # ----- TEMPBIAS: UPPER BOUND -----
        # MAXIMUM LOSS - AREA EVOLVING
        mb_max_loss = (-1 * (glacier_area_t0 * icethickness_t0).sum() / glacier_area_t0.sum() * 
                       input.density_ice / input.density_water / (t2 - t1))
        # Looping forward and backward to ensure optimization does not get stuck
        modelparameters[2] = 1
        modelparameters[7] = tempchange_boundlow
        mb_mwea_1 = mb_mwea_calc(modelparameters, option_areaconstant=0)
        # use absolute value because with area evolving the maximum value is a limit
        while abs(mb_mwea_1 - mb_max_loss) > 0.001:
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
            while mb_mwea_1 > mb_obs_min:
                modelparameters[7] = modelparameters[7] + 1
                mb_mwea_1 = mb_mwea_calc(modelparameters, option_areaconstant=1)
            while mb_mwea_1 < mb_obs_min:
                modelparameters[7] = modelparameters[7] - input.tempchange_step
                mb_mwea_1 = mb_mwea_calc(modelparameters, option_areaconstant=1)
            tempchange_boundhigh = modelparameters[7] + input.tempchange_step
            
        print('mb_max_loss:', np.round(mb_max_loss,2), 'TC_max_loss_AreaEvolve:', np.round(tempchange_max_loss,2),
              '\nmb_AreaConstant:', np.round(mb_tc_boundhigh,2), 'TC_boundhigh:', np.round(tempchange_boundhigh,2), 
              '\nmb_obs_min:', np.round(mb_obs_min,2))
        # Important notes:
        #  - If mb_obs_min < mb_max_loss, then need to use constant area in order to get reasonable bounds
        #  - At high TC values, same TC will produce much more melt for constant area than changing area because 
        #    changing area is able to retreat to higher elevations and melt less over time
        

        # ----- TEMPBIAS: LOWER BOUND -----
        # Lower temperature bound based on max positive mass balance adjusted to avoid edge effects
        # Temperature at the lowest bin
        #  T_bin = T_gcm + lr_gcm * (z_ref - z_gcm) + lr_glac * (z_bin - z_ref) + tempchange
        lowest_bin = np.where(glacier_area_t0 > 0)[0][0]
        tempchange_max_acc = (-1 * (glacier_gcm_temp + glacier_gcm_lrgcm * 
                                    (elev_bins[lowest_bin] - glacier_gcm_elev)).max())
        # Compute max accumulation [mwea]
        modelparameters[2] = 1
        modelparameters[7] = -100
        mb_max_acc = mb_mwea_calc(modelparameters, option_areaconstant=1)
        
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
                
        
        mb_tc_boundlow = mb_mwea_calc(modelparameters, option_areaconstant=1)
        print('\nmb_max_acc:', np.round(mb_max_acc,2), 'TC_max_acc:', np.round(tempchange_max_acc,2),
              '\nmb_TC_boundlow_PF1:', np.round(mb_tc_boundlow,2), 'TC_boundhigh:', np.round(tempchange_boundlow,2),
              '\nmb_obs_max:', np.round(mb_obs_max,2)
              )
        
        #%%
#        # UPPER BOUND ADJUSTMENTS
#        # Loop backward to avoid edge effects
#        if input.tempchange_edge_method == 'mb':
#            modelparameters[7] = tempchange_max_loss
#            modelparameters[2] = 1
#            mb_mwea = mb_max_loss
#            while mb_mwea < mb_max_loss + input.tempchange_edge_mb:
#                modelparameters[7] = modelparameters[7] - input.tempchange_step
#                mb_mwea = mb_mwea_calc(modelparameters)
#            tempchange_boundhigh = modelparameters[7]
#        elif input.tempchange_edge_method == 'mb_norm':
#            modelparameters[7] = tempchange_max_loss
#            mb_norm = mb_norm_calc(mb_max_loss)
#            while mb_norm < 1 - input.tempchange_edge_mbnorm:
#                modelparameters[7] = modelparameters[7] - input.tempchange_step
#                mb_norm = mb_norm_calc(mb_mwea_calc(modelparameters))
#            tempchange_boundhigh = modelparameters[7]
#        elif input.tempchange_edge_method == 'mb_norm_slope':
#            tempchange_boundhigh = tempchange_max_loss
#            mb_slope = 0
#            while mb_slope > input.tempchange_edge_mbnormslope:
#                tempchange_boundhigh -= input.tempchange_step
#                modelparameters[7] = tempchange_boundhigh + 0.5
#                tc_norm_2 = tc_norm_calc(modelparameters[7])
#                mb_norm_2 = mb_norm_calc(mb_mwea_calc(modelparameters))
#                modelparameters[7] = tempchange_boundhigh - 0.5
#                tc_norm_1 = tc_norm_calc(modelparameters[7])
#                mb_norm_1 = mb_norm_calc(mb_mwea_calc(modelparameters))
#                mb_slope = (mb_norm_2 - mb_norm_1) / (tc_norm_2 - tc_norm_1) 
        #%%
        tempchange_sigma = (tempchange_boundhigh - tempchange_boundlow) / input.tempchange_sigma_adj
        if tempchange_sigma > input.tempchange_sigma:
            tempchange_sigma = input.tempchange_sigma
        
        # OPTIMAL TEMPERATURE CHANGE (PF = 1)
        def find_tempchange_opt(tempchange_4opt):
            """
            Find optimal temperature based on observed mass balance
            """
            # Use a subset of model parameters to reduce number of constraints required
            modelparameters[7] = tempchange_4opt[0]
            # Mean annual mass balance [mwea]
            mb_mwea = mb_mwea_calc(modelparameters, option_areaconstant=1)
            return abs(mb_mwea - observed_massbal)
        # Find optimized tempchange in agreement with observed mass balance
        tempchange_opt_init = [np.mean([tempchange_boundlow, tempchange_boundhigh])]
        tempchange_opt_bnds = (tempchange_boundlow, tempchange_boundhigh)
        tempchange_opt_all = minimize(find_tempchange_opt, tempchange_opt_init, 
                                      bounds=[tempchange_opt_bnds], method='L-BFGS-B')
        tempchange_opt = tempchange_opt_all.x[0]
        print(tempchange_opt)
        
        #%%
        
#        # ADJUST TEMPCHANGE AWAY FROM EDGE by 1 sigma and optimize (PF ~= 1)
#        if tempchange_opt < tempchange_boundlow + tempchange_sigma:
#            print('\nAdjust TC_opt away from lower bound')
#            # Alter PF such that TC_opt is reasonable (at least 1 sigma away)
#            tempchange_opt = tempchange_boundlow + tempchange_sigma
#            modelparameters[7] = tempchange_opt
#            precfactor_mu = input.precfactor_boundlow
#            modelparameters[2] = precfactor_mu
#            mb_mwea = mb_mwea_calc(modelparameters, option_areaconstant=1)
#            while mb_mwea < observed_massbal:
#                precfactor_mu += input.precfactor_step
#                modelparameters[2] = precfactor_mu
#                mb_mwea = mb_mwea_calc(modelparameters, option_areaconstant=1)
#                print('TC:', np.round(modelparameters[7],1), 'PF:', np.round(modelparameters[2],1), 
#                      'Mod:', np.round(mb_mwea,2), 'Obs:', np.round(observed_massbal,2))  
#        else:
#            precfactor_mu = np.mean([input.precfactor_boundlow, input.precfactor_boundhigh])
#        
#        
#        # ----- CHECK LOWER BOUND CAN BE REACHED -----
#        # First, check if observation above threshold such that lower bound for precfactor should be set to 1
#        if input.tempchange_edge_method == 'mb':
#            mb_check = observed_massbal
#            mb_threshold = mb_max_acc - input.tempchange_edge_mb
#        elif input.tempchange_edge_method == 'mb_norm':
#            mb_check = mb_norm_calc(observed_massbal)
#            mb_threshold = input.tempchange_edge_mbnorm
#        elif input.tempchange_edge_method == 'mb_norm_slope':
#            mb_check = mb_norm_calc(observed_massbal)
#            mb_threshold = (mb_norm_1 + mb_norm_2) / 2
#            
#        if mb_check > mb_threshold:
#            precfactor_boundlow = 1
#        else:
#            precfactor_boundlow = input.precfactor_boundlow
#
#        # Second, adjust precfactor down to 0 (if needed)
#        modelparameters[7] = tempchange_opt + tempchange_sigma
#        modelparameters[2] = precfactor_boundlow
#        mb_mod_min = mb_mwea_calc(modelparameters, option_areaconstant=1)
#        print('\nCheck lower bound can be reached with TC_opt + 1 sigma, PF_boundlow')
#        while mb_mod_min > mb_obs_min and precfactor_boundlow > 0:
#            precfactor_boundlow -= input.precfactor_step
#            if precfactor_boundlow < 0:
#                precfactor_boundlow = 0
#            modelparameters[2] = precfactor_boundlow
#            mb_mod_min = mb_mwea_calc(modelparameters, option_areaconstant=1)
#            print('TC:', np.round(modelparameters[7],1), 'PF:', np.round(modelparameters[2],1), 
#                  'mod_min:', np.round(mb_mod_min,2), 'obs_min:', np.round(mb_obs_min,2))
#        print('TC:', np.round(modelparameters[7],1), 'PF:', np.round(modelparameters[2],1), 
#                  'mod_min:', np.round(mb_mod_min,2), 'obs_min:', np.round(mb_obs_min,2))
#        
#        # Then shift TC_opt (if needed)
#        while mb_mod_min > mb_obs_min and tempchange_opt < tempchange_boundhigh - tempchange_sigma:
#            modelparameters[2] = precfactor_boundlow
#            tempchange_opt += input.tempchange_step
#            modelparameters[7] = tempchange_opt + tempchange_sigma
#            mb_mod_min = mb_mwea_calc(modelparameters, option_areaconstant=1)
#            print('TC:', np.round(modelparameters[7],1), 'PF:', np.round(modelparameters[2],1), 
#                  'mod_min:', np.round(mb_mod_min,2), 'obs_min:', np.round(mb_obs_min,2))
#        print('TC_opt:', np.round(tempchange_opt,1))
#        
#
#        # ----- CHECK UPPER BOUND CAN BE REACHED -----
#        print('\nCheck upper bound can be reached with TC_opt - 1 sigma, PF_boundhigh')
#        
#        precfactor_boundhigh = precfactor_mu + (precfactor_mu - precfactor_boundlow)
#        modelparameters[7] = tempchange_opt - tempchange_sigma
#        modelparameters[2] = precfactor_boundhigh
#        mb_mod_max = mb_mwea_calc(modelparameters, option_areaconstant=1)
#        while mb_mod_max < mb_obs_max and precfactor_boundhigh < 20:
#            precfactor_boundhigh += input.precfactor_step
#            modelparameters[2] = precfactor_boundhigh
#            mb_mod_max = mb_mwea_calc(modelparameters, option_areaconstant=1)
#            print('TC:', np.round(modelparameters[7],1), 'PF:', np.round(modelparameters[2],1), 
#                  'mod_min:', np.round(mb_mod_max,2), 'obs_min:', np.round(mb_obs_max,2))
#            
#        tempchange_mu = tempchange_opt
#        tempchange_start = tempchange_mu
##%%
#        print('\nParameters:\nPF_low:', np.round(precfactor_boundlow,2), 'PF_high:', np.round(precfactor_boundhigh,2),
#              '\nTC_low:', np.round(tempchange_boundlow,2), 'TC_high:', np.round(tempchange_boundhigh,2),
#              '\nTC_mu:', np.round(tempchange_mu,2), 'TC_sigma:', np.round(tempchange_sigma,2))
#        
#        print('\n\nNEED TO TEST EXAMPLE WHERE ITS AT THE UPPER BOUND!\n\n')
#        
#        mb_compare.loc[n,'max_acc_mwea'] = mb_max_acc
#        mb_compare.loc[n,'mb_obs_max'] = mb_obs_max
#        mb_compare.loc[n,'PF_low'] = precfactor_boundlow
#        mb_compare.loc[n,'PF_high'] = precfactor_boundhigh
#        mb_compare.loc[n,'TC_mu'] = tempchange_mu
#        mb_compare.loc[n,'TC_sigma'] = tempchange_sigma
#        mb_compare.loc[n,'TC_low'] = tempchange_boundlow
#        mb_compare.loc[n,'TC_high'] = tempchange_boundhigh
#        
#        
#    netcdf_fn = mcmc_output_netcdf_fp + glacier_str + '.nc'
#    iters = len(ds.iter.values)
#    burn = 0
#    if input.new_setup == 1:
#        plot_mc_results(netcdf_fn, glacier_cal_data, iters=iters, burn=burn,
#                        newsetup=1, mb_max_acc=mb_max_acc, mb_max_loss=mb_max_loss,
#                        precfactor_boundlow=precfactor_boundlow, precfactor_boundhigh=precfactor_boundhigh,
#                        tempchange_mu=tempchange_mu, tempchange_sigma=tempchange_sigma, 
#                        tempchange_boundlow=tempchange_boundlow, tempchange_boundhigh=tempchange_boundhigh)
#    else:
#        plot_mc_results(netcdf_fn, glacier_cal_data, iters=iters, burn=burn,
#                        tempchange_mu=tempchange_mu, tempchange_sigma=tempchange_sigma, 
#                        tempchange_boundlow=tempchange_boundlow, tempchange_boundhigh=tempchange_boundhigh)

#%%
## Export comparison
#mb_compare_fn = 'mb_compare_' + 'R' + str(input.rgi_regionsO1[0]) + '_' + str(main_glac_rgi.shape[0]) + 'glac.csv'
#mb_compare.to_csv('/Users/davidrounce/Documents/Dave_Rounce/HiMAT/PyGEM/../Output/cal_opt2/' + mb_compare_fn)
    
    #%% Plot mass balance vs parameters
##    tempchange_iters = np.arange(-1.5, 5, 0.01).tolist()
#    tc_iter_step = 0.1
#    tempchange_iters = np.arange(int(tempchange_max_acc), int(tempchange_max_loss)+tc_iter_step, tc_iter_step).tolist()
##    tempchange_iters = np.arange(-15, 25, 1).tolist()
#    
#    ddfsnow_iters = [0.0031, 0.0041, 0.0051]
#    precfactor_iters = [0.5, 1, 1.5]
##    ddfsnow_iters = [0.0041]
##    precfactor_iters = [1]
#    plot_mb_vs_parameters(tempchange_iters, precfactor_iters, ddfsnow_iters, modelparameters, glacier_rgi_table, 
#                          glacier_area_t0, icethickness_t0, width_t0, elev_bins, glacier_gcm_temp, glacier_gcm_prec, 
#                          glacier_gcm_elev, glacier_gcm_lrgcm, glacier_gcm_lrglac, dates_table, observed_massbal, 
#                          observed_error, mb_max_acc, mb_max_loss, tempchange_max_acc, tempchange_max_loss, 
#                          option_areaconstant=1)

    #%% Other plots
#    plot_mc_results(mcmc_output_netcdf_fp + glacier_str + '.nc', glacier_cal_data, iters=15000, burn=0)
#    plot_mc_results2(mcmc_output_netcdf_fp + glacier_str + '.nc', glacier_cal_data, burns=[0,1000,2000], plot_res=500)
#    summary(mcmc_output_netcdf_fp + glacier_str + '.nc', glacier_cal_data,
#            filename = mcmc_output_tables_fp + glacier_str + '.txt')

#%% TEST PREC TRANSFORMATION FUNCTION
#def prec_transformation(precfactor_raw, lowbnd=input.precfactor_boundlow):
#    """
#    Converts raw precipitation factors from normal distribution to correct values.
#
#    Takes raw values from normal distribution and converts them to correct precipitation factors according to:
#        if x >= 0:
#            f(x) = x + 1
#        else:
#            f(x) = 1 - x / lowbnd * (1 - (1/(1-lowbnd)))
#    i.e., normally distributed values from -2 to 2 and converts them to be 1/3 to 3.
#
#    Parameters
#    ----------
#    precfactor_raw : float
#        numpy array of untransformed precipitation factor values
#
#    Returns
#    -------
#    x : float
#        array of corrected precipitation factors
#    """        
#    x = precfactor_raw.copy()
#    x[x >= 0] = x[x >= 0] + 1
#    x[x < 0] = 1 - x[x < 0] / lowbnd * (1 - (1/(1-lowbnd)))        
#    return x
    
#x = np.random.normal(size=(int(1e6)))
#x_prec = prec_transformation(x)
#x_prec_old = prec_transformation_old(x)
#y = np.arange(-2,2,0.01)
#y_prec = prec_transformation(y)
#y_prec_dydx = (y_prec[:-1] - y_prec[1:]) / (y[:-1] - y[1:])
#y_pdf = norm.pdf(y)
#plt.plot(y,y_prec, label='precfactor (x_ax:raw, y_ax:precfactor)')
#plt.plot(y[:-1],y_prec_dydx, label='slope_precfactor (x_ax:raw, y_ax:slope)')
#plt.plot(y, y_pdf, label='untransformed pdf (x_ax:raw, y_ax:pdf)')
#plt.hist(x_prec, density=True, bins=100, label='transformed pdf (x_ax:precfactor, y_ax:pdf)')
#plt.xlim(-2,3)
#plt.legend()
#plt.savefig(input.output_filepath + 'figures/prec_transformation_new.png', bbox_inches='tight', dpi=300)