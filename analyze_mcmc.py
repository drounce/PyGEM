"""Run the model calibration"""
# Spyder cannot run parallels, so always set -option_parallels=0 when testing in Spyder.

# Built-in libraries
import os
#import glob
#import argparse
#import multiprocessing
#import time
#import inspect
#from time import strftime
# External libraries
import pandas as pd
import numpy as np
import xarray as xr
import pymc
import matplotlib.pyplot as plt
from scipy.stats.kde import gaussian_kde
from scipy.stats import norm
from scipy.stats import truncnorm
from scipy.stats import uniform
from scipy.stats import linregress
# Local libraries
import pygem_input as input
import pygemfxns_modelsetup as modelsetup
import class_mbdata

#%% TO-DO LIST 
# EXPORT CSV STATISTICS - SEE EXAMPLES FROM PYMC FOR DATA WE WANT TO INCLUDE



#%% ===== SCRIPT SPECIFIC INPUT DATA =====
cal_datasets = ['shean']
# Label dictionaries for pairwise scatter plots
vn_label_dict = {'massbal':'Mass balance\n[mwea]',
                 'precfactor':'Precipitation factor\n[-]',
                 'tempchange':'Temperature bias\n[degC]',
                 'ddfsnow':'DDFsnow\n[mwe $degC^{-1} d^{-1}$]'}
vn_label_nounits_dict = {'massbal':'Mass balance',
                         'precfactor':'Prec factor',
                         'tempchange':'Temp bias',
                         'ddfsnow':'DDFsnow'}    
# mcmc model parameters
variables = ['massbal', 'precfactor', 'tempchange', 'ddfsnow']
# Autocorrelation lags
acorr_maxlags = 100

# Export option
#output_filepath = input.main_directory + '/../Output/'
mcmc_output_netcdf_fp = input.main_directory + '/../MCMC_data/spc/mcmc_48glac_3ch_25000iter_20180915/netcdf/'
mcmc_output_figures_fp = input.main_directory + '/../MCMC_data/spc/mcmc_48glac_3ch_25000iter_20180915/figures/'

#%%

# ===== Define functions needed for MCMC method =====
def prec_transformation(precfactor_raw):
    """
    Converts raw precipitation factors from normal distribution to correct values.

    Takes raw values from normal distribution and converts them to correct precipitation factors according to:
        if x >= 0:
            f(x) = x + 1
        else:
            f(x) = 1 / (1 - x)
    i.e., normally distributed values from -2 to 2 and converts them to be 1/3 to 3.

    Parameters
    ----------
    precfactor_raw : float
        numpy array of untransformed precipitation factor values

    Returns
    -------
    precfactor : float
        array of corrected precipitation factors
    """        
    precfactor = precfactor_raw.copy()
    precfactor[precfactor >= 0] = precfactor[precfactor >= 0] + 1
    precfactor[precfactor < 0] = 1 / (1 - precfactor[precfactor < 0])            
    return precfactor


def effective_n(df, vn):
    """
    Compute the effective sample size of a trace.
    
    Takes the trace and computes the effective sample size according to its detrended autocorrelation.
    
    Parameters
    ----------
    model : pymc.MCMC.MCMC
        Model containing traces of parameters, summary statistics, etc.
    vn : str
        Parameter variable name

    Returns
    -------
    effective_n : int
        effective sample size
    """
    # Effective sample size      
    x = df[vn].values
    # detrend trace using mean to be consistent with statistics definition of autocorrelation
    x = (x - x.mean())
    # compute autocorrelation (note: only need second half since they are symmetric)
    rho = np.correlate(x, x, mode='full')
    rho = rho[len(rho)//2:]
    # normalize the autocorrelation values
    #  note: rho[0] is the variance * n_samples, so this is consistent with the statistics definition of 
    #        autocorrelation on wikipedia (dividing by n_samples gives you the expected value).
    rho_norm = rho / rho[0]
    # Iterate untile sum of consecutive estimates of autocorrelation is negative to avoid issues with the sum
    # being -0.5, which returns an effective_n of infinity
    negative_autocorr = False
    t = 1
    n = len(x)
    while not negative_autocorr and (t < n):
        if not t % 2:
            negative_autocorr = sum(rho_norm[t-1:t+1]) < 0
        t += 1
    return int(n / (1 + 2*rho_norm[1:t].sum()))


def gelman_rubin(netcdf_fn, vn, iters=50, burn=0):
    """
    Calculate Gelman-Rubin statistic.
    
    Parameters
    ----------
    netcdf_fn : str
        Netcdf of MCMC methods with chains of model parameters
    vn : str
        Parameter variable name
    iters : int
        Number of iterations associated with the Markov Chain
    burn : int
        Number of iterations to burn in with the Markov Chain
        
    Returns
    -------
    gelman_rubin_stat : float
        gelman_rubin statistic (R_hat)
    """
    # Open dataset
    ds = xr.open_dataset(netcdf_fn)    
    # Create list of model output to be used with functions
    dfs = []
    for n_chain in ds.chain.values:
        dfs.append(pd.DataFrame(ds['mp_value'].sel(chain=n_chain).values[burn:burn+iters], columns=ds.mp.values))
    # Gelman-Rubin statistic
    if len(dfs) > 1:
        for n_chain in range(0,len(dfs)):
            df = dfs[n_chain]
            if n_chain == 0:
                chain = df[vn].values
                chain = np.reshape(chain, (1,len(chain)))
            else: 
                chain2add = np.reshape(df[vn].values, (1,chain.shape[1]))
                chain = np.append(chain, chain2add, axis=0)
    return pymc.gelman_rubin(chain)


def write_csv_results(models, distribution_type='truncnormal'):
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

    
def plot_mc_results(netcdf_fn, iters=50, burn=0, distribution_type='truncnormal', 
                    precfactor_mu=input.precfactor_mu, precfactor_sigma=input.precfactor_sigma, 
                    precfactor_boundlow=input.precfactor_boundlow, 
                    precfactor_boundhigh=input.precfactor_boundhigh,
                    tempchange_mu=input.tempchange_mu, tempchange_sigma=input.tempchange_sigma, 
                    tempchange_boundlow=input.tempchange_boundlow, 
                    tempchange_boundhigh=input.tempchange_boundhigh,
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
    distribution_type : str
        Distribution type either 'truncnormal' or 'uniform' (default truncnormal)
    glacier_RGIId_float : str
    precfactor_mu : float
        Mean of precipitation factor (default assigned from input)
    precfactor_sigma : float
        Standard deviation of precipitation factor (default assigned from input)
    precfactor_boundlow : float
        Lower boundary of precipitation factor (default assigned from input)
    precfactor_boundhigh : float
        Upper boundary of precipitation factor (default assigned from input)
    tempchange_mu : float
        Mean of temperature change (default assigned from input)
    tempchange_sigma : float
        Standard deviation of temperature change (default assigned from input)
    tempchange_boundlow : float
        Lower boundary of temperature change (default assigned from input)
    tempchange_boundhigh: float
        Upper boundary of temperature change (default assigned from input)
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
    # Open dataset
    ds = xr.open_dataset(netcdf_fn)    
    # Create list of model output to be used with functions
    dfs = []
    for n_chain in ds.chain.values:
        dfs.append(pd.DataFrame(ds['mp_value'].sel(chain=n_chain).values[burn:burn+iters], columns=ds.mp.values))
        
    # ===== CHAIN, HISTOGRAM, AND AUTOCORRELATION PLOTS ===========================
    plt.figure(figsize=(12, len(variables)*3))
    plt.subplots_adjust(wspace=0.3, hspace=0.5)
    plt.suptitle('mcmc_ensembles_' + glacier_str + '_' + distribution_type, y=0.94)
    
    # Bounds (SciPy convention)
    precfactor_a = (precfactor_boundlow - precfactor_mu) / precfactor_sigma
    precfactor_b = (precfactor_boundhigh - precfactor_mu) / precfactor_sigma
    tempchange_a = (tempchange_boundlow - tempchange_mu) / tempchange_sigma
    tempchange_b = (tempchange_boundhigh - tempchange_mu) / tempchange_sigma
    ddfsnow_a = (ddfsnow_boundlow - ddfsnow_mu) / ddfsnow_sigma
    ddfsnow_b = (ddfsnow_boundhigh - ddfsnow_mu) / ddfsnow_sigma

    for count, vn in enumerate(variables):
        # ===== Chain =====
        plt.subplot(len(variables), 3, 3*count+1)
        chain_legend = []
        for n_df, df in enumerate(dfs):
            chain = df[vn].values
            runs = np.arange(0,chain.shape[0])
            if n_df == 0:
                plt.plot(runs, chain, color='b')
            elif n_df == 1:
                plt.plot(runs, chain, color='r')
            else:
                plt.plot(runs, chain, color='y')
            chain_legend.append('chain' + str(n_df + 1))
        plt.legend(chain_legend)
        plt.xlabel('Step Number', size=10)
        plt.ylabel(vn_label_dict[vn], size=10)
        # ===== Prior and posterior distributions =====
        plt.subplot(len(variables), 3, 3*count+2)
        # Prior distribution
        z_score = np.linspace(norm.ppf(0.01), norm.ppf(0.99), 100)
        if vn == 'massbal':
            x_values = observed_massbal + observed_error * z_score
            y_values = norm.pdf(x_values, loc=observed_massbal, scale=observed_error)
        elif vn == 'precfactor':
            if distribution_type == 'truncnormal':
                z_score = np.linspace(truncnorm.ppf(0.01, precfactor_a, precfactor_b), 
                                      truncnorm.ppf(0.99, precfactor_a, precfactor_b), 100)
                x_values_raw = precfactor_mu + precfactor_sigma * z_score
                y_values = truncnorm.pdf(x_values_raw, precfactor_a, precfactor_b, loc=precfactor_mu, 
                                         scale=precfactor_sigma)
            elif distribution_type == 'uniform':
                z_score = np.linspace(uniform.ppf(0.01), uniform.ppf(0.99), 100)
                x_values_raw = precfactor_boundlow + z_score * (precfactor_boundhigh - precfactor_boundlow)
                y_values = uniform.pdf(x_values_raw, loc=precfactor_boundlow, 
                                       scale=(precfactor_boundhigh - precfactor_boundlow))
            # transform the precfactor values from the truncated normal to the actual values
            x_values = prec_transformation(x_values_raw)
        elif vn == 'tempchange':
            if distribution_type == 'truncnormal':
                z_score = np.linspace(truncnorm.ppf(0.01, tempchange_a, tempchange_b), 
                                      truncnorm.ppf(0.99, tempchange_a, tempchange_b), 100)
                x_values = tempchange_mu + tempchange_sigma * z_score
                y_values = truncnorm.pdf(x_values, tempchange_a, tempchange_b, loc=tempchange_mu, 
                                         scale=tempchange_sigma)
            elif distribution_type == 'uniform':                
                z_score = np.linspace(uniform.ppf(0.01), uniform.ppf(0.99), 100)
                x_values = tempchange_boundlow + z_score * (tempchange_boundhigh - tempchange_boundlow)
                y_values = uniform.pdf(x_values, loc=tempchange_boundlow, 
                                       scale=(tempchange_boundhigh - tempchange_boundlow))
        elif vn == 'ddfsnow':
            if distribution_type == 'truncnormal':
                z_score = np.linspace(truncnorm.ppf(0.01, ddfsnow_a, ddfsnow_b), 
                                      truncnorm.ppf(0.99, ddfsnow_a, ddfsnow_b), 100)
                x_values = ddfsnow_mu + ddfsnow_sigma * z_score
                y_values = truncnorm.pdf(x_values, ddfsnow_a, ddfsnow_b, loc=ddfsnow_mu, scale=ddfsnow_sigma)
            elif distribution_type == 'uniform':                
                z_score = np.linspace(uniform.ppf(0.01), uniform.ppf(0.99), 100)
                x_values = ddfsnow_boundlow + z_score * (ddfsnow_boundhigh - ddfsnow_boundlow)
                y_values = uniform.pdf(x_values, loc=ddfsnow_boundlow, 
                                       scale=(ddfsnow_boundhigh - ddfsnow_boundlow))
        plt.plot(x_values, y_values, color='k')
        # Ensemble/Posterior distribution  
        # extents
        if chain.min() < x_values.min():
            x_min = chain.min()
        else:
            x_min = x_values.min()
        if chain.max() > x_values.max():
            x_max = chain.max()
        else:
            x_max = x_values.max()
        # Chain legend
        if vn == 'massbal':
            chain_legend = ['observed']
        else:
            chain_legend = ['prior']
        # Loop through models
        for n_chain, df in enumerate(dfs):
            chain = df[vn].values
            # gaussian distribution
            if vn == 'massbal':
                kde = gaussian_kde(chain)
                x_values_kde = np.linspace(x_min, x_max, 100)
                y_values_kde = kde(x_values_kde)
                chain_legend.append('ensemble' + str(n_chain + 1))
            elif vn == 'precfactor':
                kde = gaussian_kde(chain)
                x_values_kde = x_values.copy()
                y_values_kde = kde(x_values_kde)
                chain_legend.append('posterior' + str(n_chain + 1))
            else:
                kde = gaussian_kde(chain)
                x_values_kde = x_values.copy()
                y_values_kde = kde(x_values_kde)
                chain_legend.append('posterior' + str(n_chain + 1))
            if n_chain == 0:
                plt.plot(x_values_kde, y_values_kde, color='b')
            elif n_chain == 1:
                plt.plot(x_values_kde, y_values_kde, color='r')
            else:      
                plt.plot(x_values_kde, y_values_kde, color='y')
            plt.xlabel(vn_label_dict[vn], size=10)
            plt.ylabel('PDF', size=10)
            plt.legend(chain_legend)
        
        # ===== Normalized autocorrelation ======
        plt.subplot(len(variables), 3, 3*count+3)
        chain_norm = chain - chain.mean()
        if chain.shape[0] <= acorr_maxlags:
            acorr_lags = chain.shape[0] - 1
        else:
            acorr_lags = acorr_maxlags
        plt.acorr(chain_norm, maxlags=acorr_lags)
        plt.xlim(0,acorr_lags)
        plt.xlabel('lag')
        plt.ylabel('autocorrelation')
        chain_neff = effective_n(dfs[0], vn)
        plt.text(int(0.6*acorr_lags), 0.85, 'n_eff=' + str(chain_neff))                
    # Save figure
    plt.savefig(mcmc_output_figures_fp + glacier_str + '_' + distribution_type + '_plots_' + str(len(dfs)) + 'chain_' 
                + str(iters) + 'iter_' + str(burn) + 'burn' + '.png', bbox_inches='tight')
    
    # ===== PAIRWISE SCATTER PLOTS ===========================================================
    fig = plt.figure(figsize=(10,12))
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.suptitle('mcmc_pairwise_scatter_' + glacier_str + '_' + distribution_type, y=0.94)
    
    df = dfs[0]
    nvars = len(variables)
    for h, vn1 in enumerate(variables):
        v1 = chain = df[vn].values
        for j, vn2 in enumerate(variables):
            v2 = chain = df[vn2].values
            nsub = h * nvars + j + 1
            ax = fig.add_subplot(nvars, nvars, nsub)
            if h == j:
                plt.hist(v1)
                plt.tick_params(axis='both', bottom=False, left=False, labelleft=False, labelbottom=False)
            elif h > j:
                plt.plot(v2, v1, 'o', mfc='none', mec='black')
            else:
                # Need to plot blank, so axis remain correct
                plt.plot(v2, v1, 'o', mfc='none', mec='none')
                slope, intercept, r_value, p_value, std_err = linregress(v2, v1)
                text2plot = (vn_label_nounits_dict[vn2] + '/\n' + vn_label_nounits_dict[vn1] + '\n$R^2$=' + 
                             '{:.2f}'.format((r_value**2)))
                ax.text(0.5, 0.5, text2plot, transform=ax.transAxes, fontsize=14, 
                        verticalalignment='center', horizontalalignment='center')
            # Plot bottom left
            if (h+1 == nvars) and (j == 0):
                plt.tick_params(axis='both', which='both', left=True, right=False, labelbottom=True, 
                                labelleft=True, labelright=False)
                plt.xlabel(vn_label_dict[vn2])
                plt.ylabel(vn_label_dict[vn1])
            # Plot bottom only
            elif h + 1 == nvars:
                plt.tick_params(axis='both', which='both', left=False, right=False, labelbottom=True, 
                                labelleft=False, labelright=False)
                plt.xlabel(vn_label_dict[vn2])
            # Plot left only (exclude histogram values)
            elif (h !=0) and (j == 0):
                plt.tick_params(axis='both', which='both', left=True, right=False, labelbottom=False, 
                                labelleft=True, labelright=False)
                plt.ylabel(vn_label_dict[vn1])
            else:
                plt.tick_params(axis='both', left=False, right=False, labelbottom=False, 
                                labelleft=False, labelright=False)
    plt.savefig(mcmc_output_figures_fp + glacier_str + '_' + distribution_type + '_pairwisescatter_' + str(len(dfs)) + 
                'chain_' + str(iters) + 'iter_' + str(burn) + 'burn' + '.png', bbox_inches='tight')
     
    
#%% Find files
# ===== LOAD CALIBRATION DATA =====
rgi_glac_number = []
for i in os.listdir(mcmc_output_netcdf_fp):
#for i in ['15.00621.nc']:
    glacier_str = i.replace('.nc', '')
    if glacier_str.startswith(str(input.rgi_regionsO1[0])):
        rgi_glac_number.append(glacier_str.split('.')[1])
rgi_glac_number = sorted(rgi_glac_number)
# Glacier RGI data
main_glac_rgi = modelsetup.selectglaciersrgitable(rgi_regionsO1=input.rgi_regionsO1, rgi_regionsO2 = 'all',
                                                  rgi_glac_number=rgi_glac_number)
# Glacier hypsometry [km**2], total area
main_glac_hyps = modelsetup.import_Husstable(main_glac_rgi, input.rgi_regionsO1, input.hyps_filepath,
                                             input.hyps_filedict, input.hyps_colsdrop)
# Select dates including future projections
dates_table_nospinup, start_date, end_date = modelsetup.datesmodelrun(startyear=input.startyear, endyear=input.endyear, 
                                                                      spinupyears=0)
# Calibration data
cal_data = pd.DataFrame()
for dataset in cal_datasets:
    cal_subset = class_mbdata.MBData(name=dataset, rgi_regionO1=input.rgi_regionsO1[0])
    cal_subset_data = cal_subset.retrieve_mb(main_glac_rgi, main_glac_hyps, dates_table_nospinup)
    cal_data = cal_data.append(cal_subset_data, ignore_index=True)
cal_data = cal_data.sort_values(['glacno', 't1_idx'])
cal_data.reset_index(drop=True, inplace=True)

# ===== PROCESS EACH NETCDF FILE =====
for n, glac_str_noreg in enumerate(rgi_glac_number):
    # Glacier string
    glacier_str = str(input.rgi_regionsO1[0]) + '.' + glac_str_noreg
    # Mass balance data
    observed_massbal = cal_data.mb_mwe[n] / (cal_data.t2[n] - cal_data.t1[n])
    observed_error = cal_data.mb_mwe_err[n] / (cal_data.t2[n] - cal_data.t1[n])    
    # MCMC plots
    plot_mc_results(mcmc_output_netcdf_fp + glacier_str + '.nc', iters=25000, burn=0)
    
    # Example of Gelman-Rubin statistic
    print(glacier_str)
    for vn in variables:
        print(vn, ':', gelman_rubin(mcmc_output_netcdf_fp + glacier_str + '.nc', vn, iters=25000, burn=0))
        