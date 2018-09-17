"""Run the model calibration"""
# Spyder cannot run parallels, so always set -option_parallels=0 when testing in Spyder.

# Built-in libraries
import os
import glob
import argparse
import multiprocessing
import time
import inspect
from time import strftime
# External libraries
import pandas as pd
import numpy as np
import xarray as xr
import pymc
from pymc import deterministic
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats.kde import gaussian_kde
from scipy.stats import norm
from scipy.stats import truncnorm
from scipy.stats import uniform
from scipy.stats import linregress
# Local libraries
import pygem_input as input
import pygemfxns_modelsetup as modelsetup
import pygemfxns_massbalance as massbalance
import class_climate
import class_mbdata

#%% ===== SCRIPT SPECIFIC INPUT DATA =====
# Calibration datasets
cal_datasets = ['shean']
#cal_datasets = ['shean', 'wgms_d', 'wgms_ee', 'group']

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
option_export = 1
output_filepath = input.main_directory + '/../Output/'

# Debugging boolean (if true, a number of print statements are activated through the running of the model)
debug = False

#%% FUNCTIONS
def getparser():
    """
    Use argparse to add arguments from the command line
    
    Parameters
    ----------
    ref_gcm_name (optional) : str
        reference gcm name
    num_simultaneous_processes (optional) : int
        number of cores to use in parallels
    option_parallels (optional) : int
        switch to use parallels or not
        
    Returns
    -------
    Object containing arguments and their respective values.
    """
    parser = argparse.ArgumentParser(description="run calibration in parallel")
    # add arguments
    parser.add_argument('-ref_gcm_name', action='store', type=str, default=input.ref_gcm_name,
                        help='reference gcm name')
    parser.add_argument('-num_simultaneous_processes', action='store', type=int, default=4,
                        help='number of simultaneous processes (cores) to use')
    parser.add_argument('-option_parallels', action='store', type=int, default=1,
                        help='Switch to use or not use parallels (1 - use parallels, 0 - do not)')
    return parser


def main(list_packed_vars):
    """
    Model calibration
    
    Parameters
    ----------
    list_packed_vars : list
        list of packed variables that enable the use of parallels
        
    Returns
    -------
    netcdf files of the calibration output (specific output is dependent on which calibration scheme is used)
    """
    # Unpack variables
    count = list_packed_vars[0]
    chunk = list_packed_vars[1]
    chunk_size = list_packed_vars[2]
    main_glac_rgi_all = list_packed_vars[3]
    gcm_name = list_packed_vars[4]

    time_start = time.time()
    parser = getparser()
    args = parser.parse_args()

    # ===== LOAD GLACIER DATA =====
    #  'raw' refers to the glacier subset that includes glaciers with and without calibration data
    #  after the calibration data has been imported, then all glaciers without data will be dropped
    # Glacier RGI data
    main_glac_rgi_raw = main_glac_rgi_all.iloc[chunk:chunk + chunk_size, :].copy()
    # Glacier hypsometry [km**2], total area
    main_glac_hyps_raw = modelsetup.import_Husstable(main_glac_rgi_raw, input.rgi_regionsO1, input.hyps_filepath,
                                                     input.hyps_filedict, input.hyps_colsdrop)
    # Ice thickness [m], average
    main_glac_icethickness_raw = modelsetup.import_Husstable(main_glac_rgi_raw, input.rgi_regionsO1, 
                                                             input.thickness_filepath, input.thickness_filedict, 
                                                             input.thickness_colsdrop)
    main_glac_hyps_raw[main_glac_icethickness_raw == 0] = 0
    # Width [km], average
    main_glac_width_raw = modelsetup.import_Husstable(main_glac_rgi_raw, input.rgi_regionsO1, input.width_filepath,
                                                      input.width_filedict, input.width_colsdrop)
    elev_bins = main_glac_hyps_raw.columns.values.astype(int)
    # Add volume [km**3] and mean elevation [m a.s.l.]
    main_glac_rgi_raw['Volume'], main_glac_rgi_raw['Zmean'] = (
            modelsetup.hypsometrystats(main_glac_hyps_raw, main_glac_icethickness_raw))
    # Select dates including future projections
    #  - nospinup dates_table needed to get the proper time indices
    dates_table_nospinup, start_date, end_date = modelsetup.datesmodelrun(startyear=input.startyear, 
                                                                          endyear=input.endyear, spinupyears=0)
    dates_table, start_date, end_date = modelsetup.datesmodelrun(startyear=input.startyear, 
                                                                 endyear=input.endyear, spinupyears=input.spinupyears)

    # ===== LOAD CALIBRATION DATA =====
    cal_data = pd.DataFrame()
    for dataset in cal_datasets:
        cal_subset = class_mbdata.MBData(name=dataset, rgi_regionO1=input.rgi_regionsO1[0])
        cal_subset_data = cal_subset.retrieve_mb(main_glac_rgi_raw, main_glac_hyps_raw, dates_table_nospinup)
        cal_data = cal_data.append(cal_subset_data, ignore_index=True)
    cal_data = cal_data.sort_values(['glacno', 't1_idx'])
    cal_data.reset_index(drop=True, inplace=True)
    # If group data is included, then add group dictionary and add group name to main_glac_rgi
    if set(['group']).issubset(cal_datasets) == True:
        # Group dictionary
        group_dict_raw = pd.read_csv(input.mb_group_fp + input.mb_group_dict_fn)
        # Remove groups that have no data
        group_names_wdata = np.unique(cal_data[np.isnan(cal_data.glacno)].group_name.values).tolist()
        group_dict_raw_wdata = group_dict_raw.loc[group_dict_raw.group_name.isin(group_names_wdata)]
        # Create dictionary to map group names to main_glac_rgi
        group_dict = dict(zip(group_dict_raw_wdata['RGIId'], group_dict_raw_wdata['group_name']))
        group_names_unique = list(set(group_dict.values()))
        group_dict_keyslist = [[] for x in group_names_unique]
        for n, group in enumerate(group_names_unique):
            group_dict_keyslist[n] = [group, [k for k, v in group_dict.items() if v == group]]
        # Add group name to main_glac_rgi
        main_glac_rgi_raw['group_name'] = main_glac_rgi_raw['RGIId'].map(group_dict)
    else:
        main_glac_rgi_raw['group_name'] = np.nan
    

    # Drop glaciers that do not have any calibration data (individual or group)    
    main_glac_rgi = ((main_glac_rgi_raw.iloc[np.unique(
            np.append(main_glac_rgi_raw[main_glac_rgi_raw['group_name'].notnull() == True].index.values, 
                      np.where(main_glac_rgi_raw[input.rgi_O1Id_colname].isin(cal_data['glacno']) == True)[0])), :])
            .copy())
    # select glacier data
    main_glac_hyps = main_glac_hyps_raw.iloc[main_glac_rgi.index.values]
    main_glac_icethickness = main_glac_icethickness_raw.iloc[main_glac_rgi.index.values]
    main_glac_width = main_glac_width_raw.iloc[main_glac_rgi.index.values]
    # Reset index
    main_glac_rgi.reset_index(drop=True, inplace=True)
    main_glac_hyps.reset_index(drop=True, inplace=True)
    main_glac_icethickness.reset_index(drop=True, inplace=True)
    main_glac_width.reset_index(drop=True, inplace=True)

    # ===== LOAD CLIMATE DATA =====
    gcm = class_climate.GCM(name=gcm_name)
    # Air temperature [degC]
    gcm_temp, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(gcm.temp_fn, gcm.temp_vn, main_glac_rgi, dates_table)
    # Precipitation [m]
    gcm_prec, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(gcm.prec_fn, gcm.prec_vn, main_glac_rgi, dates_table)
    # Elevation [m asl]
    gcm_elev = gcm.importGCMfxnearestneighbor_xarray(gcm.elev_fn, gcm.elev_vn, main_glac_rgi)
    # Lapse rate
    if gcm_name == 'ERA-Interim':
        gcm_lr, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(gcm.lr_fn, gcm.lr_vn, main_glac_rgi, dates_table)
    else:
        # Mean monthly lapse rate
        ref_lr_monthly_avg = np.genfromtxt(gcm.lr_fp + gcm.lr_fn, delimiter=',')
        gcm_lr = np.tile(ref_lr_monthly_avg, int(gcm_temp.shape[1]/12))

    # ===== CALIBRATION =====
    # Option 2: use MCMC method to determine posterior probability distributions of the three parameters tempchange,
    #           ddfsnow and precfactor. Then create an ensemble of parameter sets evenly sampled from these 
    #           distributions, and output these sets of parameters and their corresponding mass balances to be used in 
    #           the simulations.
    if input.option_calibration == 2:

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
        
        def run_MCMC(distribution_type='truncnormal', 
                     precfactor_mu=input.precfactor_mu, precfactor_sigma=input.precfactor_sigma, 
                     precfactor_boundlow=input.precfactor_boundlow, precfactor_boundhigh=input.precfactor_boundhigh, 
                     precfactor_start=input.precfactor_start,
                     tempchange_mu=input.tempchange_mu, tempchange_sigma=input.tempchange_sigma, 
                     tempchange_boundlow=input.tempchange_boundlow, tempchange_boundhigh=input.tempchange_boundhigh,
                     tempchange_start=input.tempchange_start,
                     ddfsnow_mu=input.ddfsnow_mu, ddfsnow_sigma=input.ddfsnow_sigma, 
                     ddfsnow_boundlow=input.ddfsnow_boundlow, ddfsnow_boundhigh=input.ddfsnow_boundhigh,
                     ddfsnow_start=input.ddfsnow_start,
                     iterations=10, burn=0, thin=input.thin_interval, tune_interval=1000, step=None, 
                     tune_throughout=True, save_interval=None, burn_till_tuned=False, stop_tuning_after=5, verbose=0, 
                     progress_bar=False, dbname=None):
            """
            Runs the MCMC algorithm.

            Runs the MCMC algorithm by setting the prior distributions and calibrating the probability distributions of 
            three parameters for the mass balance function.

            Parameters
            ----------
            distribution_type : str
                Distribution type either 'truncnormal' or 'uniform' (default normal)
            precfactor_mu : float
                Mean of precipitation factor (default assigned from input)
            precfactor_sigma : float
                Standard deviation of precipitation factor (default assigned from input)
            precfactor_boundlow : float
                Lower boundary of precipitation factor (default assigned from input)
            precfactor_boundhigh : float
                Upper boundary of precipitation factor (default assigned from input)
            precfactor_start : float
                Starting value of precipitation factor for sampling iterations (default assigned from input)
            tempchange_mu : float
                Mean of temperature change (default assigned from input)
            tempchange_sigma : float
                Standard deviation of temperature change (default assigned from input)
            tempchange_boundlow : float
                Lower boundary of temperature change (default assigned from input)
            tempchange_boundhigh: float
                Upper boundary of temperature change (default assigned from input)
            tempchange_start : float
                Starting value of temperature change for sampling iterations (default assigned from input)
            ddfsnow_mu : float
                Mean of degree day factor of snow (default assigned from input)
            ddfsnow_sigma : float 
                Standard deviation of degree day factor of snow (default assigned from input)
            ddfsnow_boundlow : float
                Lower boundary of degree day factor of snow (default assigned from input)
            ddfsnow_boundhigh : float
                Upper boundary of degree day factor of snow (default assigned from input)
            ddfsnow_start : float
                Starting value of degree day factor of snow for sampling iterations (default assigned from input)
            iterations : int
                Total number of iterations to do (default 10).
            burn : int
                Variables will not be tallied until this many iterations are complete (default 0).
            thin : int
                Variables will be tallied at intervals of this many iterations (default 1).
            tune_interval : int
                Step methods will be tuned at intervals of this many iterations (default 1000).
            step : str
                Choice of step method to use (default metropolis-hastings).
            tune_throughout : boolean
                If true, tuning will continue after the burnin period; otherwise tuning will halt at the end of the 
                burnin period (default True).    
            save_interval : int or None
                If given, the model state will be saved at intervals of this many iterations (default None).
            burn_till_tuned: boolean
                If True the Sampler will burn samples until all step methods are tuned. A tuned step methods is one 
                that was not tuned for the last `stop_tuning_after` tuning intervals. The burn-in phase will have a 
                minimum of 'burn' iterations but could be longer if tuning is needed. After the phase is done the 
                sampler will run for another (iter - burn) iterations, and will tally the samples according to the 
                'thin' argument. This means that the total number of iteration is update throughout the sampling 
                procedure.  If True, it also overrides the tune_thorughout argument, so no step method will be tuned 
                when sample are being tallied (default False).    
            stop_tuning_after: int
                The number of untuned successive tuning interval needed to be reached in order for the burn-in phase to 
                be done (if burn_till_tuned is True) (default 5).
            verbose : int
                An integer controlling the verbosity of the models output for debugging (default 0).
            progress_bar : boolean
                Display progress bar while sampling (default True).
            dbname : str
                Choice of database name the sample should be saved to (default 'trial.pickle').

            Returns
            -------
            pymc.MCMC.MCMC
                Returns a model that contains sample traces of tempchange, ddfsnow, precfactor and massbalance. These 
                samples can be accessed by calling the trace attribute. For example:

                    model.trace('ddfsnow')[:]

                gives the trace of ddfsnow values.

                A trace, or Markov Chain, is an array of values outputed by the MCMC simulation which defines the
                posterior probability distribution of the variable at hand.
            """        
            # Assign prior distributions
            # Temperature change and precipitation factor depend on distribution type
            if distribution_type == 'truncnormal':
                # Precipitation factor [-]
                #  truncated normal distribution (-2 to 2) to reflect that we have confidence in the data, but allow for 
                #  bias (following the transformation) to range from 1/3 to 3.  
                #  Transformation is if x >= 0, x+1; else, 1/(1-x)
                precfactor = pymc.TruncatedNormal('precfactor', mu=precfactor_mu, tau=1/(precfactor_sigma**2), 
                                                  a=precfactor_boundlow, b=precfactor_boundhigh, value=precfactor_start)
                # Temperature change [degC]
                #  truncated normal distribution (-10 to 10) to reflect that we have confidence in the data, but allow
                #  for bias to still be present.
                tempchange = pymc.TruncatedNormal('tempchange', mu=tempchange_mu, tau=1/(tempchange_sigma**2), 
                                                  a=tempchange_boundlow, b=tempchange_boundhigh, value=tempchange_start)
                # Degree day factor of snow [mwe degC-1 d-1]
                #  truncated normal distribution with mean 0.0041 mwe degC-1 d-1 and standard deviation of 0.0015 
                #  (Braithwaite, 2008)
                ddfsnow = pymc.TruncatedNormal('ddfsnow', mu=ddfsnow_mu, tau=1/(ddfsnow_sigma**2), a=ddfsnow_boundlow, 
                                               b=ddfsnow_boundhigh, value=ddfsnow_start)
            elif distribution_type == 'uniform':
                # Precipitation factor [-]
                precfactor = pymc.Uniform('precfactor', lower=precfactor_boundlow, upper=precfactor_boundhigh, 
                                          value=precfactor_start)
                # Temperature change [degC]
                tempchange = pymc.Uniform('tempchange', lower=tempchange_boundlow, upper=tempchange_boundhigh, 
                                          value=tempchange_start)
                # Degree day factor of snow [mwe degC-1 d-1]
                ddfsnow = pymc.Uniform('ddfsnow', lower=ddfsnow_boundlow, upper=ddfsnow_boundhigh, value=ddfsnow_start)
            
            # Define deterministic function for MCMC model based on our a priori probobaility distributions.
            @deterministic(plot=False)
            def massbal(precfactor=precfactor, ddfsnow=ddfsnow, tempchange=tempchange):
                # Copy model parameters and change them based on the probability distribtions we have given
                modelparameters_copy = modelparameters.copy()
                if precfactor is not None:
                    modelparameters_copy[2] = float(precfactor)
                if ddfsnow is not None:
                    modelparameters_copy[4] = float(ddfsnow)
                if tempchange is not None:
                    modelparameters_copy[7] = float(tempchange)
                # Precipitation factor transformation
                if modelparameters_copy[2] >= 0:
                    modelparameters_copy[2] = modelparameters_copy[2] + 1
                else:
                    modelparameters_copy[2] = 1 / (1 - modelparameters_copy[2])
                # Degree day factor of ice is proportional to ddfsnow
                modelparameters[5] = modelparameters[4] / input.ddfsnow_iceratio
                # Mass balance calculations
                (glac_bin_temp, glac_bin_prec, glac_bin_acc, glac_bin_refreeze, glac_bin_snowpack, glac_bin_melt,
                 glac_bin_frontalablation, glac_bin_massbalclim, glac_bin_massbalclim_annual, glac_bin_area_annual,
                 glac_bin_icethickness_annual, glac_bin_width_annual, glac_bin_surfacetype_annual,
                 glac_wide_massbaltotal, glac_wide_runoff, glac_wide_snowline, glac_wide_snowpack,
                 glac_wide_area_annual, glac_wide_volume_annual, glac_wide_ELA_annual) = (
                    massbalance.runmassbalance(modelparameters_copy, glacier_rgi_table, glacier_area_t0, 
                                               icethickness_t0, width_t0, elev_bins, glacier_gcm_temp, glacier_gcm_prec,
                                               glacier_gcm_elev, glacier_gcm_lrgcm, glacier_gcm_lrglac, dates_table,
                                               option_areaconstant=1))
                # Return glacier-wide mass balance [mwea] for comparison
                return glac_wide_massbaltotal[t1_idx:t2_idx].sum() / (t2 - t1)            
            # Observed distribution
            #  This observation data defines the observed likelihood of the mass balances, and allows us to fit the 
            #  probability distribution of the mass balance to the results.
            obs_massbal = pymc.Normal('obs_massbal', mu=massbal, tau=(1/(observed_error**2)), 
                                      value=float(observed_massbal), observed=True)
            # Set model
            if dbname is None:
                model = pymc.MCMC({'precfactor':precfactor, 'tempchange':tempchange, 'ddfsnow':ddfsnow, 
                                   'massbal':massbal, 'obs_massbal':obs_massbal})
            else:
                model = pymc.MCMC({'precfactor':precfactor, 'tempchange':tempchange, 'ddfsnow':ddfsnow, 
                                   'massbal':massbal, 'obs_massbal':obs_massbal}, db='pickle', dbname=dbname)
            # set step method if specified
            if step == 'am':
                model.use_step_method(pymc.AdaptiveMetropolis, precfactor, delay = 1000)
                model.use_step_method(pymc.AdaptiveMetropolis, tempchange, delay = 1000)
                model.use_step_method(pymc.AdaptiveMetropolis, ddfsnow, delay = 1000)
            # sample
            #  note: divide by zero warning here that does not affect model run
            model.sample(iter=iterations, burn=burn, thin=thin,
                         tune_interval=tune_interval, tune_throughout=tune_throughout,
                         save_interval=save_interval, verbose=verbose, progress_bar=progress_bar)
            #close database
            model.db.close()

            return model
        
        def effective_n(model, vn):
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
            x = model.trace(vn)[:]
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
        
        
        def gelman_rubin(models, vn):
            """
            Calculate Gelman-Rubin statistic.
            
            Parameters
            ----------
            models : list of pymc.MCMC.MCMC
                Models containing traces of parameters, summary statistics, etc.
            vn : str
                Parameter variable name
                
            Returns
            -------
            gelman_rubin_stat : float
                gelman_rubin statistic (R_hat)
            """
            # Gelman-Rubin statistic
            if len(models) > 1:
                for n_chain in range(0,len(models)):
                    model = models[n_chain]
                    if n_chain == 0:
                        chain = model.trace(vn)[:]
                        chain = np.reshape(chain, (1,len(chain)))
                    else: 
                        chain2add = np.reshape(model.trace(vn)[:], (1,chain.shape[1]))
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

            
        def plot_mc_results(models, distribution_type='truncnormal', 
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
            models : list of pymc.MCMC.MCMC
                Models containing traces of parameters, summary statistics, etc.
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
            model = models[0]
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
                for n_model, model in enumerate(models):
                    chain = model.trace(vn)[:]
                    if vn == 'precfactor':
                        chain_raw = chain.copy()
                        chain = prec_transformation(chain_raw)
                    runs = np.arange(0,chain.shape[0])
                    if n_model == 0:
                        plt.plot(runs, chain, color='b')
                    elif n_model == 1:
                        plt.plot(runs, chain, color='r')
                    else:
                        plt.plot(runs, chain, color='y')
                    chain_legend.append('chain' + str(n_model + 1))
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
                for n_model, model in enumerate(models):
                    chain = model.trace(vn)[:]
                    # If precipitation factor, then transform chain
                    if vn == 'precfactor':
                        chain_raw = chain.copy()
                        chain = prec_transformation(chain_raw)
                    # gaussian distribution
                    if vn == 'massbal':
                        kde = gaussian_kde(chain)
                        x_values_kde = np.linspace(x_min, x_max, 100)
                        y_values_kde = kde(x_values_kde)
                        chain_legend.append('ensemble' + str(n_model + 1))
                    elif vn == 'precfactor':
                        kde = gaussian_kde(chain_raw)
                        x_values_kde = x_values.copy()
                        y_values_kde = kde(x_values_raw)
                        chain_legend.append('posterior' + str(n_model + 1))
                    else:
                        kde = gaussian_kde(chain)
                        x_values_kde = x_values.copy()
                        y_values_kde = kde(x_values_kde)
                        chain_legend.append('posterior' + str(n_model + 1))
                    if n_model == 0:
                        plt.plot(x_values_kde, y_values_kde, color='b')
                    elif n_model == 1:
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
                chain_neff = effective_n(model, vn)
                plt.text(int(0.6*acorr_lags), 0.85, 'n_eff=' + str(chain_neff))                
            # Save figure
            plt.savefig(input.mcmc_output_figs_fp + glacier_str + '_' + distribution_type + '_plots_' + 
                        str(len(models)) + 'chain_' + str(input.mcmc_sample_no) + 'iter_' + str(input.mcmc_burn_no) 
                        + 'burn' + '.png', bbox_inches='tight')
            
            # ===== PAIRWISE SCATTER PLOTS ===========================================================
            fig = plt.figure(figsize=(10,12))
            plt.subplots_adjust(wspace=0.1, hspace=0.1)
            plt.suptitle('mcmc_pairwise_scatter_' + glacier_str + '_' + distribution_type, y=0.94)
            
            nvars = len(variables)
            for i, vn1 in enumerate(variables):
                v1 = model.trace(vn1)[:]
                if vn1 == 'precfactor':
                    v1 = prec_transformation(v1.copy())
                for j, vn2 in enumerate(variables):
                    v2 = model.trace(vn2)[:]
                    if vn2 == 'precfactor':
                        v2 = prec_transformation(v2.copy())
                    nsub = i * nvars + j + 1
                    ax = fig.add_subplot(nvars, nvars, nsub)
                    if i == j:
                        plt.hist(v1)
                        plt.tick_params(axis='both', bottom=False, left=False, labelleft=False, labelbottom=False)
                    elif i > j:
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
                    if (i+1 == nvars) and (j == 0):
                        plt.tick_params(axis='both', which='both', left=True, right=False, labelbottom=True, 
                                        labelleft=True, labelright=False)
                        plt.xlabel(vn_label_dict[vn2])
                        plt.ylabel(vn_label_dict[vn1])
                    # Plot bottom only
                    elif i + 1 == nvars:
                        plt.tick_params(axis='both', which='both', left=False, right=False, labelbottom=True, 
                                        labelleft=False, labelright=False)
                        plt.xlabel(vn_label_dict[vn2])
                    # Plot left only (exclude histogram values)
                    elif (i !=0) and (j == 0):
                        plt.tick_params(axis='both', which='both', left=True, right=False, labelbottom=False, 
                                        labelleft=True, labelright=False)
                        plt.ylabel(vn_label_dict[vn1])
                    else:
                        plt.tick_params(axis='both', left=False, right=False, labelbottom=False, 
                                        labelleft=False, labelright=False)
            plt.savefig(input.mcmc_output_figs_fp + glacier_str + '_' + distribution_type + '_pairwisescatter_' + 
                        str(len(models)) + 'chain_' + str(input.mcmc_sample_no) + 'iter_' + str(input.mcmc_burn_no) 
                        + 'burn' + '.png', bbox_inches='tight')


        def process_df(df):
            """
            Processes the dataframe to  include only relevant information needed for future model runs.
            
            Parameters
            ----------
            df : pandas dataframe
                Dataframe outputed by stratified sampling
                
            Returns
            -------
            df : pandas dataframe
                Dataframe with the other uncalibrated parameters (lrgcm, lrglac, precgrad, ddfice, tempsnow) added.

            Note: Creates an index for the dataframe (from zero to 1 less than number of ensemble runs) and names the 
                  index 'runs'. Names the columns axis 'variables'
            """
            # set columns for static variables
            df['lrgcm'] = np.full(len(df), input.lrgcm)
            df['lrglac'] = np.full(len(df), input.lrglac)
            df['precgrad'] = np.full(len(df), input.precgrad)
            df['ddfice'] = np.full(len(df), input.ddfice)
            df['tempsnow'] = np.full(len(df), input.tempsnow)
            # drop unnecesary info
            df = df.drop('sorted_index', 1)
            # name column axis
            df.columns.name = 'variables'
            # create a new index
            df['runs'] = np.arange(len(df))
            df = df.set_index('runs')
            return df


        # ===== Begin MCMC process =====
        # loop through each glacier selected
        for glac in range(main_glac_rgi.shape[0]):

            if debug:
                print(count, main_glac_rgi.loc[main_glac_rgi.index.values[glac],'RGIId_float'])

            # Set model parameters
            modelparameters = [input.lrgcm, input.lrglac, input.precfactor,
                               input.precgrad, input.ddfsnow, input.ddfice,
                               input.tempsnow, input.tempchange]

            # Select subsets of data
            glacier_rgi_table = main_glac_rgi.loc[main_glac_rgi.index.values[glac], :]
            glacier_gcm_elev = gcm_elev[glac]
            glacier_gcm_prec = gcm_prec[glac,:]
            glacier_gcm_temp = gcm_temp[glac,:]
            glacier_gcm_lrgcm = gcm_lr[glac,:]
            glacier_gcm_lrglac = glacier_gcm_lrgcm.copy()
            glacier_area_t0 = main_glac_hyps.iloc[glac,:].values.astype(float)
            icethickness_t0 = main_glac_icethickness.iloc[glac,:].values.astype(float)
            width_t0 = main_glac_width.iloc[glac,:].values.astype(float)
            glacier_cal_data = ((cal_data.iloc[np.where(
                    glacier_rgi_table[input.rgi_O1Id_colname] == cal_data['glacno'])[0],:]).copy())
            glacier_str = '{0:0.5f}'.format(glacier_rgi_table['RGIId_float'])

            # Select observed mass balance, error, and time data
            cal_idx = glacier_cal_data.index.values[0]
            #  Note: index to main_glac_rgi may differ from cal_idx
            t1 = glacier_cal_data.loc[cal_idx, 't1']
            t2 = glacier_cal_data.loc[cal_idx, 't2']
            t1_idx = int(glacier_cal_data.loc[cal_idx,'t1_idx'])
            t2_idx = int(glacier_cal_data.loc[cal_idx,'t2_idx'])
            observed_massbal = glacier_cal_data.loc[cal_idx,'mb_mwe'] / (t2 - t1)
            observed_error = glacier_cal_data.loc[cal_idx,'mb_mwe_err'] / (t2 - t1)

            if debug:
                print('observed_massbal:',observed_massbal, 'observed_error:',observed_error)


            # ===== RUN MARKOV CHAIN MONTE CARLO METHOD ====================            
            # specify distribution type
            distribution_type = input.mcmc_distribution_type
            # fit the MCMC model
            for n_chain in range(0,input.n_chains):
                print(glacier_str + ' chain' + str(n_chain))
                if n_chain == 0:
                    model = run_MCMC(distribution_type=distribution_type, iterations=input.mcmc_sample_no, 
                                     burn=input.mcmc_burn_no, step=input.mcmc_step)
                elif n_chain == 1:
                    # Chains start at lowest values
                    model = run_MCMC(distribution_type=distribution_type, precfactor_start=input.precfactor_boundlow,
                                     tempchange_start=input.tempchange_boundlow, ddfsnow_start=input.ddfsnow_boundlow, 
                                     iterations=input.mcmc_sample_no, burn=input.mcmc_burn_no, step=input.mcmc_step)
                elif n_chain == 2:
                    # Chains start at highest values
                    model = run_MCMC(distribution_type=distribution_type, precfactor_start=input.precfactor_boundhigh,
                                     tempchange_start=input.tempchange_boundlow, ddfsnow_start=input.ddfsnow_boundlow, 
                                     iterations=input.mcmc_sample_no, burn=input.mcmc_burn_no, step=input.mcmc_step)
                    
                df = pd.DataFrame({'tempchange': model.trace('tempchange')[:],
                               'precfactor': prec_transformation(model.trace('precfactor')[:]),
                               'ddfsnow': model.trace('ddfsnow')[:], 
                               'massbal': model.trace('massbal')[:]})
                # set columns for other variables
                df['ddfice'] = df['ddfsnow'] / input.ddfsnow_iceratio
                df['lrgcm'] = np.full(df.shape[0], input.lrgcm)
                df['lrglac'] = np.full(df.shape[0], input.lrglac)
                df['precgrad'] = np.full(df.shape[0], input.precgrad)
                df['tempsnow'] = np.full(df.shape[0], input.tempsnow)
                if n_chain == 0:
                    df_chains = df.values[:, :, np.newaxis]
                else:
                    df_chains = np.dstack((df_chains, df.values))
            
            ds = xr.Dataset({'mp_value': (('iter', 'mp', 'chain'), df_chains)},
                            coords={'iter': df.index.values,
                                    'mp': df.columns.values,
                                    'chain': np.arange(0,n_chain+1)})
       
            ds.to_netcdf(input.mcmc_output_netcdf_fp + glacier_str + '.nc')
            
            # NEED TO RE-STRUCTURE PLOTS AND CSVs to be done in post-processing with netcdf files
            
#            #%%
#            # Example of accessing netcdf file and putting it back into pandas dataframe
#            A = xr.open_dataset(input.mcmc_output_netcdf_fp + '15.03473.nc')
#            B = pd.DataFrame(A['mp_value'].sel(chain=0).values, columns=A.mp.values)
#            #%%
            
            # Plot parameters
#            plot_mc_results(models, distribution_type=distribution_type)
            # Write statisticis to csv
#            write_csv_results(models, distribution_type=distribution_type)

        # ==============================================================
        
    # Option 1: mimize mass balance difference using three-step approach to expand solution space
    elif input.option_calibration == 1:
        
        # ===== FUNCTIONS USED IN CALIBRATION OPTION ===== 
        # Optimization function: Define the function that you are trying to minimize
        #  - modelparameters are the parameters that will be optimized
        #  - return value is the value is the value used to run the optimization
        def objective(modelparameters_subset):
            """
            Objective function for independent glacier data.
            
            Uses a z-score to enable use of different datasets (mass balance, snowline, etc.)

            Parameters
            ----------
            modelparameters_subset : np.float64
                List of model parameters to calibrate
                [precipitation factor, precipitation gradient, degree-day factor of snow, temperature change]

            Returns
            -------
            sum_abs_zscore
                Returns the sum of the absolute z-scores, which represents how well the model matches observations
            """
            # Use a subset of model parameters to reduce number of constraints required
            modelparameters[2] = modelparameters_subset[0]
            modelparameters[3] = modelparameters_subset[1]
            modelparameters[4] = modelparameters_subset[2]
            modelparameters[5] = modelparameters[4] / input.ddfsnow_iceratio
            modelparameters[7] = modelparameters_subset[3]
            # Mass balance calculations
            (glac_bin_temp, glac_bin_prec, glac_bin_acc, glac_bin_refreeze, glac_bin_snowpack, glac_bin_melt, 
             glac_bin_frontalablation, glac_bin_massbalclim, glac_bin_massbalclim_annual, glac_bin_area_annual, 
             glac_bin_icethickness_annual, glac_bin_width_annual, glac_bin_surfacetype_annual, 
             glac_wide_massbaltotal, glac_wide_runoff, glac_wide_snowline, glac_wide_snowpack, 
             glac_wide_area_annual, glac_wide_volume_annual, glac_wide_ELA_annual) = (
                massbalance.runmassbalance(modelparameters, glacier_rgi_table, glacier_area_t0, icethickness_t0, 
                                           width_t0, elev_bins, glacier_gcm_temp, glacier_gcm_prec, 
                                           glacier_gcm_elev, glacier_gcm_lrgcm, glacier_gcm_lrglac, dates_table, 
                                           option_areaconstant=1))  
            # Loop through all measurements
            for x in range(glacier_cal_data.shape[0]):
                cal_idx = glacier_cal_data.index.values[x]
                # Mass balance comparisons
                if ((glacier_cal_data.loc[cal_idx, 'obs_type'] == 'mb_geo') or 
                    (glacier_cal_data.loc[cal_idx, 'obs_type'] == 'mb_glac')):
                    # Observed mass balance [mwe]
                    glacier_cal_compare.loc[cal_idx, 'obs'] = glacier_cal_data.loc[cal_idx, 'mb_mwe']
                    glacier_cal_compare.loc[cal_idx, 'obs_unit'] = 'mwe'
                    # Modeled mass balance [mwe]
                    #  Sum(mass balance x area) / total area
                    t1_idx = glacier_cal_data.loc[cal_idx, 't1_idx'].astype(int)
                    t2_idx = glacier_cal_data.loc[cal_idx, 't2_idx'].astype(int)
                    z1_idx = glacier_cal_data.loc[cal_idx, 'z1_idx'].astype(int)
                    z2_idx = glacier_cal_data.loc[cal_idx, 'z2_idx'].astype(int)
                    year_idx = int(t1_idx / 12)
                    bin_area_subset = glac_bin_area_annual[z1_idx:z2_idx, year_idx]
                    glacier_cal_compare.loc[cal_idx, 'model'] = (
                            (glac_bin_massbalclim[z1_idx:z2_idx, t1_idx:t2_idx] * 
                             bin_area_subset[:,np.newaxis]).sum() / bin_area_subset.sum())
                    # Z-score for modeled mass balance based on observed mass balance and uncertainty
                    #  z-score = (model - measured) / uncertainty
                    glacier_cal_compare.loc[cal_idx, 'uncertainty'] = (input.massbal_uncertainty_mwea * 
                            (glacier_cal_data.loc[cal_idx, 't2'] - glacier_cal_data.loc[cal_idx, 't1']))
                    glacier_cal_compare.loc[cal_idx, 'zscore'] = (
                            (glacier_cal_compare.loc[cal_idx, 'model'] - glacier_cal_compare.loc[cal_idx, 'obs']) /
                            glacier_cal_compare.loc[cal_idx, 'uncertainty'])
            # Minimize the sum of differences
            sum_abs_zscore = abs(glacier_cal_compare['zscore']).sum()
            return sum_abs_zscore
        
        # Group optimization function 
        def objective_group(modelparameters_subset):
            """
            Objective function for grouped glacier data.
            
            The group objective cycles through all the glaciers in a group.
            Uses a z-score to enable use of different datasets (mass balance, snowline, etc.).
            
            Parameters
            ----------
            modelparameters_subset : np.float64
                List of model parameters to calibrate
                [precipitation factor, precipitation gradient, degree-day factor of snow, temperature change]

            Returns
            -------
            abs_zscore
                Returns the absolute z-score, which represents how well the model matches observations
            """
            # Record group's cumulative area and mass balance for comparison
            group_cum_area_km2 = 0
            group_cum_mb_mkm2 = 0    
            # Loop through all glaciers
            for glac in range(main_glac_rgi.shape[0]):
                # Check if glacier is included in group
                if main_glac_rgi.loc[glac, 'group_name'] == group_name:    
                    # Set model parameters
                    # if model parameters already exist for the glacier, then use those instead of group parameters
                    modelparameters = [input.lrgcm, input.lrglac, input.precfactor, input.precgrad, input.ddfsnow, 
                                       input.ddfice, input.tempsnow, input.tempchange]
                    if np.all(main_glac_modelparamsopt[glac] == 0) == False:
                        modelparameters = main_glac_modelparamsopt[glac]
                    else:
                        # Use a subset of model parameters to reduce number of constraints required
                        modelparameters[2] = modelparameters_subset[0]
                        modelparameters[3] = modelparameters_subset[1]
                        modelparameters[4] = modelparameters_subset[2]
                        modelparameters[5] = modelparameters[4] / input.ddfsnow_iceratio
                        modelparameters[7] = modelparameters_subset[3]
                    # Select subsets of data
                    glacier_rgi_table = main_glac_rgi.loc[main_glac_rgi.index.values[glac], :]   
                    glacier_gcm_elev = gcm_elev[glac]
                    glacier_gcm_prec = gcm_prec[glac,:]
                    glacier_gcm_temp = gcm_temp[glac,:]
                    glacier_gcm_lrgcm = gcm_lr[glac,:]
                    glacier_gcm_lrglac = glacier_gcm_lrgcm.copy()
                    glacier_area_t0 = main_glac_hyps.iloc[glac,:].values.astype(float)   
                    icethickness_t0 = main_glac_icethickness.iloc[glac,:].values.astype(float)
                    width_t0 = main_glac_width.iloc[glac,:].values.astype(float)
                    # Mass balance calculations
                    (glac_bin_temp, glac_bin_prec, glac_bin_acc, glac_bin_refreeze, glac_bin_snowpack, glac_bin_melt, 
                     glac_bin_frontalablation, glac_bin_massbalclim, glac_bin_massbalclim_annual, glac_bin_area_annual, 
                     glac_bin_icethickness_annual, glac_bin_width_annual, glac_bin_surfacetype_annual, 
                     glac_wide_massbaltotal, glac_wide_runoff, glac_wide_snowline, glac_wide_snowpack, 
                     glac_wide_area_annual, glac_wide_volume_annual, glac_wide_ELA_annual) = (
                        massbalance.runmassbalance(modelparameters, glacier_rgi_table, glacier_area_t0, icethickness_t0, 
                                                   width_t0, elev_bins, glacier_gcm_temp, glacier_gcm_prec, 
                                                   glacier_gcm_elev, glacier_gcm_lrgcm, glacier_gcm_lrglac, dates_table, 
                                                   option_areaconstant=1, warn_calving=0))  
                    # Mass balance comparisons
                    # Modeled mass balance [mwe]
                    #  Sum(mass balance x area) / total area
                    t1_idx = cal_data.loc[cal_idx, 't1_idx'].astype(int)
                    t2_idx = cal_data.loc[cal_idx, 't2_idx'].astype(int)
                    z1_idx = 0
                    z2_idx = glac_bin_area_annual.shape[0]
                    year_idx = int(t1_idx / 12)
                    bin_area_subset = glac_bin_area_annual[z1_idx:z2_idx, year_idx]                    
                    group_cum_area_km2 = group_cum_area_km2 + bin_area_subset.sum()
                    group_cum_mb_mkm2 = (
                            group_cum_mb_mkm2 + 
                            (glac_bin_massbalclim[z1_idx:z2_idx, t1_idx:t2_idx] * bin_area_subset[:,np.newaxis]).sum())
            # Z-score for modeled mass balance based on observed mass balance and uncertainty
            #  z-score = (model - measured) / uncertainty
            glacier_cal_compare.model = group_cum_mb_mkm2 / group_cum_area_km2
            glacier_cal_compare.zscore = (
                    (glacier_cal_compare.model - glacier_cal_compare.obs) / glacier_cal_compare.uncertainty)
            # Minimize the sum of differences
            abs_zscore = abs(glacier_cal_compare.zscore)
            print(abs_zscore)
            return abs_zscore

        # Output to record
        # Observations vs. model
        output_cols = ['glacno', 'obs_type', 'obs_unit', 'obs', 'model', 'uncertainty', 'zscore', 'calround']
        main_glac_cal_compare = pd.DataFrame(np.zeros((cal_data.shape[0],len(output_cols))),
                                             columns=output_cols)
        main_glac_cal_compare.index = cal_data.index.values
        # Model parameters
        main_glac_modelparamsopt = np.zeros((main_glac_rgi.shape[0], len(input.modelparams_colnames)))
        # Glacier-wide climatic mass balance (required for transfer fucntions)
        main_glacwide_mbclim_mwe = np.zeros((main_glac_rgi.shape[0], 1))
 
        # Loop through glaciers that have unique cal_data
        cal_individual_glacno = np.unique(cal_data.loc[cal_data['glacno'].notnull(), 'glacno'])
        for n in range(cal_individual_glacno.shape[0]):
            glac = np.where(main_glac_rgi[input.rgi_O1Id_colname].isin([cal_individual_glacno[n]]) == True)[0][0]
#            if glac%200 == 0:
#                print(count,':', main_glac_rgi.loc[main_glac_rgi.index.values[glac],'RGIId'])  
#            print(count,':', main_glac_rgi.loc[main_glac_rgi.index.values[glac], 'RGIId'])
            
            # Set model parameters
            modelparameters = [input.lrgcm, input.lrglac, input.precfactor, input.precgrad, input.ddfsnow, input.ddfice,
                               input.tempsnow, input.tempchange]
            # Select subsets of data
            glacier_rgi_table = main_glac_rgi.loc[main_glac_rgi.index.values[glac], :]   
            glacier_gcm_elev = gcm_elev[glac]
            glacier_gcm_prec = gcm_prec[glac,:]
            glacier_gcm_temp = gcm_temp[glac,:]
            glacier_gcm_lrgcm = gcm_lr[glac,:]
            glacier_gcm_lrglac = glacier_gcm_lrgcm.copy()
            glacier_area_t0 = main_glac_hyps.iloc[glac,:].values.astype(float)
            icethickness_t0 = main_glac_icethickness.iloc[glac,:].values.astype(float)
            width_t0 = main_glac_width.iloc[glac,:].values.astype(float)
            glacier_cal_data = ((cal_data.iloc[np.where(
                    glacier_rgi_table[input.rgi_O1Id_colname] == cal_data['glacno'])[0],:]).copy())

            # Modeled vs. Measured comparison dataframe
            glacier_cal_compare = pd.DataFrame(np.zeros((glacier_cal_data.shape[0], len(output_cols))),
                                               columns=output_cols)
            glacier_cal_compare.index = glacier_cal_data.index.values
            glacier_cal_compare[['glacno', 'obs_type']] = glacier_cal_data[['glacno', 'obs_type']]
    
            # Record the calibration round
            calround = 0

            # OPTIMIZATION FUNCTION: Define the function that you are trying to minimize
            #  - modelparameters are the parameters that will be optimized
            #  - return value is the value is the value used to run the optimization
            # One way to improve objective function to include other observations (snowlines, etc.) is to normalize the
            # measured and modeled difference by the estimated error - this would mean we are minimizing the cumulative
            # absolute z-score.
            def objective(modelparameters_subset):
                # Use a subset of model parameters to reduce number of constraints required
                modelparameters[2] = modelparameters_subset[0]
                modelparameters[3] = modelparameters_subset[1]
                modelparameters[4] = modelparameters_subset[2]
                modelparameters[5] = modelparameters[4] / input.ddfsnow_iceratio
                modelparameters[7] = modelparameters_subset[3]
                # Mass balance calculations
                (glac_bin_temp, glac_bin_prec, glac_bin_acc, glac_bin_refreeze, glac_bin_snowpack, glac_bin_melt,
                 glac_bin_frontalablation, glac_bin_massbalclim, glac_bin_massbalclim_annual, glac_bin_area_annual,
                 glac_bin_icethickness_annual, glac_bin_width_annual, glac_bin_surfacetype_annual,
                 glac_wide_massbaltotal, glac_wide_runoff, glac_wide_snowline, glac_wide_snowpack,
                 glac_wide_area_annual, glac_wide_volume_annual, glac_wide_ELA_annual) = (
                    massbalance.runmassbalance(modelparameters, glacier_rgi_table, glacier_area_t0, icethickness_t0,
                                               width_t0, elev_bins, glacier_gcm_temp, glacier_gcm_prec,
                                               glacier_gcm_elev, glacier_gcm_lrgcm, glacier_gcm_lrglac, dates_table,
                                               option_areaconstant=1))
                # Loop through all measurements
                for x in range(glacier_cal_data.shape[0]):
                    cal_idx = glacier_cal_data.index.values[x]
                    # Mass balance comparisons
                    if ((glacier_cal_data.loc[cal_idx, 'obs_type'] == 'mb_geo') or
                        (glacier_cal_data.loc[cal_idx, 'obs_type'] == 'mb_glac')):
                        # Observed mass balance [mwe]
                        glacier_cal_compare.loc[cal_idx, 'obs'] = glacier_cal_data.loc[cal_idx, 'mb_mwe']
                        glacier_cal_compare.loc[cal_idx, 'obs_unit'] = 'mwe'
                        # Modeled mass balance [mwe]
                        #  Sum(mass balance x area) / total area
                        t1_idx = glacier_cal_data.loc[cal_idx, 't1_idx'].astype(int)
                        t2_idx = glacier_cal_data.loc[cal_idx, 't2_idx'].astype(int)
                        z1_idx = glacier_cal_data.loc[cal_idx, 'z1_idx'].astype(int)
                        z2_idx = glacier_cal_data.loc[cal_idx, 'z2_idx'].astype(int)
                        year_idx = int(t1_idx / 12)
                        bin_area_subset = glac_bin_area_annual[z1_idx:z2_idx, year_idx]
                        glacier_cal_compare.loc[cal_idx, 'model'] = (
                                (glac_bin_massbalclim[z1_idx:z2_idx, t1_idx:t2_idx] *
                                 bin_area_subset[:,np.newaxis]).sum() / bin_area_subset.sum())
                        # Z-score for modeled mass balance based on observed mass balance and uncertainty
                        #  z-score = (model - measured) / uncertainty
                        glacier_cal_compare.loc[cal_idx, 'uncertainty'] = (input.massbal_uncertainty_mwea *
                                (glacier_cal_data.loc[cal_idx, 't2'] - glacier_cal_data.loc[cal_idx, 't1']))
                        glacier_cal_compare.loc[cal_idx, 'zscore'] = (
                                (glacier_cal_compare.loc[cal_idx, 'model'] - glacier_cal_compare.loc[cal_idx, 'obs']) /
                                glacier_cal_compare.loc[cal_idx, 'uncertainty'])
                # Minimize the sum of differences
                sum_abs_zscore = abs(glacier_cal_compare['zscore']).sum()
                return sum_abs_zscore

            # INITIAL GUESS
            modelparameters_init = [input.precfactor, input.precgrad, input.ddfsnow, input.tempchange]
            # PARAMETER BOUNDS (Braithwaite, 2008 for DDFsnow)
            precfactor_bnds = (0.9,1.2)
            precgrad_bnds = (0.0001,0.00025)
            ddfsnow_bnds = (0.0036, 0.0046)
            tempchange_bnds = (-1,1)
            modelparameters_bnds = (precfactor_bnds, precgrad_bnds, ddfsnow_bnds, tempchange_bnds)
            # OPTIMIZATION ROUND #1: optimize precfactor, DDFsnow, tempchange
            # Run the optimization
            #  'L-BFGS-B' - much slower
            #  'SLSQP' did not work for some geodetic measurements using the sum_abs_zscore.  One work around was to
            #    divide the sum_abs_zscore by 1000, which made it work in all cases.  However, methods were switched
            #    to 'L-BFGS-B', which may be slower, but is still effective.
            modelparameters_opt = minimize(objective, modelparameters_init, method=input.method_opt,
                                           bounds=modelparameters_bnds, options={'ftol':input.ftol_opt})
            # Record the calibration round
            calround = calround + 1
            # Record the optimized parameters
            modelparameters_init = modelparameters_opt.x
            main_glac_modelparamsopt[glac] = [modelparameters[0], modelparameters[1], modelparameters_init[0],
                     modelparameters_init[1], modelparameters_init[2], modelparameters_init[2] / input.ddfsnow_iceratio,
                     modelparameters[6], modelparameters_init[3]]
            modelparameters = main_glac_modelparamsopt[glac]
            # Re-run the optimized parameters in order to see the mass balance
            # Mass balance calculations
            (glac_bin_temp, glac_bin_prec, glac_bin_acc, glac_bin_refreeze, glac_bin_snowpack, glac_bin_melt,
             glac_bin_frontalablation, glac_bin_massbalclim, glac_bin_massbalclim_annual, glac_bin_area_annual,
             glac_bin_icethickness_annual, glac_bin_width_annual, glac_bin_surfacetype_annual,
             glac_wide_massbaltotal, glac_wide_runoff, glac_wide_snowline, glac_wide_snowpack,
             glac_wide_area_annual, glac_wide_volume_annual, glac_wide_ELA_annual) = (
                massbalance.runmassbalance(modelparameters, glacier_rgi_table, glacier_area_t0, icethickness_t0,
                                           width_t0, elev_bins, glacier_gcm_temp, glacier_gcm_prec,
                                           glacier_gcm_elev, glacier_gcm_lrgcm, glacier_gcm_lrglac, dates_table,
                                           option_areaconstant=1))
            # Loop through all measurements
            for x in range(glacier_cal_data.shape[0]):
                cal_idx = glacier_cal_data.index.values[x]
                # Mass balance comparisons
                if ((glacier_cal_data.loc[cal_idx, 'obs_type'] == 'mb_geo') or
                    (glacier_cal_data.loc[cal_idx, 'obs_type'] == 'mb_glac')):
                    # Observed mass balance [mwe]
                    glacier_cal_compare.loc[cal_idx, 'obs'] = glacier_cal_data.loc[cal_idx, 'mb_mwe']
                    glacier_cal_compare.loc[cal_idx, 'obs_unit'] = 'mwe'
                    # Modeled mass balance [mwe]
                    #  Sum(mass balance x area) / total area
                    t1_idx = glacier_cal_data.loc[cal_idx, 't1_idx'].astype(int)
                    t2_idx = glacier_cal_data.loc[cal_idx, 't2_idx'].astype(int)
                    z1_idx = glacier_cal_data.loc[cal_idx, 'z1_idx'].astype(int)
                    z2_idx = glacier_cal_data.loc[cal_idx, 'z2_idx'].astype(int)
                    year_idx = int(t1_idx / 12)
                    bin_area_subset = glac_bin_area_annual[z1_idx:z2_idx, year_idx]
                    glacier_cal_compare.loc[cal_idx, 'model'] = (
                            (glac_bin_massbalclim[z1_idx:z2_idx, t1_idx:t2_idx] *
                             bin_area_subset[:,np.newaxis]).sum() / bin_area_subset.sum())
                    # Z-score for modeled mass balance based on observed mass balance and uncertainty
                    #  z-score = (model - measured) / uncertainty
                    glacier_cal_compare.loc[cal_idx, 'uncertainty'] = (input.massbal_uncertainty_mwea *
                            (glacier_cal_data.loc[cal_idx, 't2'] - glacier_cal_data.loc[cal_idx, 't1']))
                    glacier_cal_compare.loc[cal_idx, 'zscore'] = (
                            (glacier_cal_compare.loc[cal_idx, 'model'] - glacier_cal_compare.loc[cal_idx, 'obs']) /
                            glacier_cal_compare.loc[cal_idx, 'uncertainty'])

            # OPTIMIZATION ROUND #2:
            # Set zscore to compare and the tolerance
            # if only one calibration point, then zscore should be small
            if glacier_cal_compare.shape[0] == 1:
                zscore_compare = glacier_cal_compare.loc[cal_idx, 'zscore']
                zscore_tolerance = input.zscore_tolerance_single
            # else if multiple calibration points and one is a geodetic MB, check that geodetic MB is within 1
            elif (glacier_cal_compare.obs_type.isin(['mb_geo']).any() == True) and (glacier_cal_compare.shape[0] > 1):
                zscore_compare = glacier_cal_compare.loc[glacier_cal_compare.index.values[np.where(
                        glacier_cal_compare['obs_type'] == 'mb_geo')[0][0]], 'zscore']
                zscore_tolerance = input.zscore_tolerance_all
            # otherwise, check mean zscore
            else:
                zscore_compare = abs(glacier_cal_compare['zscore']).sum() / glacier_cal_compare.shape[0]
                zscore_tolerance = input.zscore_tolerance_all
            # Check if need to expand the bounds
            if abs(zscore_compare) > zscore_tolerance:
                # Bounds
                precfactor_bnds = (0.75,1.5)
                precgrad_bnds = (0.0001,0.00025)
                ddfsnow_bnds = (0.0031, 0.0051)
                tempchange_bnds = (-2,2)
                modelparameters_bnds = (precfactor_bnds, precgrad_bnds, ddfsnow_bnds, tempchange_bnds)
                # Run optimization
                modelparameters_opt = minimize(objective, modelparameters_init, method=input.method_opt,
                                               bounds=modelparameters_bnds, options={'ftol':input.ftol_opt})
                # Record the calibration round
                calround = calround + 1
                # Record the optimized parameters
                modelparameters_init = modelparameters_opt.x
                main_glac_modelparamsopt[glac] = [modelparameters[0], modelparameters[1], modelparameters_init[0],
                         modelparameters_init[1], modelparameters_init[2],
                         modelparameters_init[2] / input.ddfsnow_iceratio, modelparameters[6], modelparameters_init[3]]
                modelparameters = main_glac_modelparamsopt[glac]
                # Re-run the optimized parameters in order to see the mass balance
                # Mass balance calculations
                (glac_bin_temp, glac_bin_prec, glac_bin_acc, glac_bin_refreeze, glac_bin_snowpack, glac_bin_melt,
                 glac_bin_frontalablation, glac_bin_massbalclim, glac_bin_massbalclim_annual, glac_bin_area_annual,
                 glac_bin_icethickness_annual, glac_bin_width_annual, glac_bin_surfacetype_annual,
                 glac_wide_massbaltotal, glac_wide_runoff, glac_wide_snowline, glac_wide_snowpack,
                 glac_wide_area_annual, glac_wide_volume_annual, glac_wide_ELA_annual) = (
                    massbalance.runmassbalance(modelparameters, glacier_rgi_table, glacier_area_t0, icethickness_t0,
                                               width_t0, elev_bins, glacier_gcm_temp, glacier_gcm_prec,
                                               glacier_gcm_elev, glacier_gcm_lrgcm, glacier_gcm_lrglac, dates_table,
                                               option_areaconstant=1))
                # Loop through all measurements
                for x in range(glacier_cal_data.shape[0]):
                    cal_idx = glacier_cal_data.index.values[x]
                    # Mass balance comparisons
                    if ((glacier_cal_data.loc[cal_idx, 'obs_type'] == 'mb_geo') or
                        (glacier_cal_data.loc[cal_idx, 'obs_type'] == 'mb_glac')):
                        # Observed mass balance [mwe]
                        glacier_cal_compare.loc[cal_idx, 'obs'] = glacier_cal_data.loc[cal_idx, 'mb_mwe']
                        glacier_cal_compare.loc[cal_idx, 'obs_unit'] = 'mwe'
                        # Modeled mass balance [mwe]
                        #  Sum(mass balance x area) / total area
                        t1_idx = glacier_cal_data.loc[cal_idx, 't1_idx'].astype(int)
                        t2_idx = glacier_cal_data.loc[cal_idx, 't2_idx'].astype(int)
                        z1_idx = glacier_cal_data.loc[cal_idx, 'z1_idx'].astype(int)
                        z2_idx = glacier_cal_data.loc[cal_idx, 'z2_idx'].astype(int)
                        year_idx = int(t1_idx / 12)
                        bin_area_subset = glac_bin_area_annual[z1_idx:z2_idx, year_idx]
                        glacier_cal_compare.loc[cal_idx, 'model'] = (
                                (glac_bin_massbalclim[z1_idx:z2_idx, t1_idx:t2_idx] *
                                 bin_area_subset[:,np.newaxis]).sum() / bin_area_subset.sum())
                        # Z-score for modeled mass balance based on observed mass balance and uncertainty
                        #  z-score = (model - measured) / uncertainty
                        glacier_cal_compare.loc[cal_idx, 'uncertainty'] = (input.massbal_uncertainty_mwea *
                                (glacier_cal_data.loc[cal_idx, 't2'] - glacier_cal_data.loc[cal_idx, 't1']))
                        glacier_cal_compare.loc[cal_idx, 'zscore'] = (
                                (glacier_cal_compare.loc[cal_idx, 'model'] - glacier_cal_compare.loc[cal_idx, 'obs']) /
                                glacier_cal_compare.loc[cal_idx, 'uncertainty'])

            # OPTIMIZATION ROUND #3:
            # Set zscore to compare and the tolerance
            # if only one calibration point, then zscore should be small
            if glacier_cal_compare.shape[0] == 1:
                zscore_compare = glacier_cal_compare.loc[cal_idx, 'zscore']
                zscore_tolerance = input.zscore_tolerance_single
            # else if multiple calibration points and one is a geodetic MB, check that geodetic MB is within 1
            elif (glacier_cal_compare.obs_type.isin(['mb_geo']).any() == True) and (glacier_cal_compare.shape[0] > 1):
                zscore_compare = glacier_cal_compare.loc[glacier_cal_compare.index.values[np.where(
                        glacier_cal_compare['obs_type'] == 'mb_geo')[0][0]], 'zscore']
                zscore_tolerance = input.zscore_tolerance_all
            # otherwise, check mean zscore
            else:
                zscore_compare = abs(glacier_cal_compare['zscore']).sum() / glacier_cal_compare.shape[0]
                zscore_tolerance = input.zscore_tolerance_all
            # Check if need to expand the bounds
            if abs(zscore_compare) > zscore_tolerance:
                # Bounds
                precfactor_bnds = (0.5,2)
                precgrad_bnds = (0.0001,0.00025)
                ddfsnow_bnds = (0.0026, 0.0056)
                tempchange_bnds = (-5,5)
                modelparameters_bnds = (precfactor_bnds, precgrad_bnds, ddfsnow_bnds, tempchange_bnds)
                # Run optimization
                modelparameters_opt = minimize(objective, modelparameters_init, method=input.method_opt,
                                               bounds=modelparameters_bnds, options={'ftol':input.ftol_opt})
                # Record the calibration round
                calround = calround + 1
                # Record the optimized parameters
                modelparameters_init = modelparameters_opt.x
                main_glac_modelparamsopt[glac] = [modelparameters[0], modelparameters[1], modelparameters_init[0],
                         modelparameters_init[1], modelparameters_init[2],
                         modelparameters_init[2] / input.ddfsnow_iceratio, modelparameters[6], modelparameters_init[3]]
                modelparameters = main_glac_modelparamsopt[glac]
                # Re-run the optimized parameters in order to see the mass balance
                (glac_bin_temp, glac_bin_prec, glac_bin_acc, glac_bin_refreeze, glac_bin_snowpack, glac_bin_melt,
                 glac_bin_frontalablation, glac_bin_massbalclim, glac_bin_massbalclim_annual, glac_bin_area_annual,
                 glac_bin_icethickness_annual, glac_bin_width_annual, glac_bin_surfacetype_annual,
                 glac_wide_massbaltotal, glac_wide_runoff, glac_wide_snowline, glac_wide_snowpack,
                 glac_wide_area_annual, glac_wide_volume_annual, glac_wide_ELA_annual) = (
                    massbalance.runmassbalance(modelparameters, glacier_rgi_table, glacier_area_t0, icethickness_t0,
                                               width_t0, elev_bins, glacier_gcm_temp, glacier_gcm_prec,
                                               glacier_gcm_elev, glacier_gcm_lrgcm, glacier_gcm_lrglac, dates_table,
                                               option_areaconstant=1))
                # Loop through all measurements
                for x in range(glacier_cal_data.shape[0]):
                    cal_idx = glacier_cal_data.index.values[x]
                    # Mass balance comparisons
                    if ((glacier_cal_data.loc[cal_idx, 'obs_type'] == 'mb_geo') or
                        (glacier_cal_data.loc[cal_idx, 'obs_type'] == 'mb_glac')):
                        # Observed mass balance [mwe]
                        glacier_cal_compare.loc[cal_idx, 'obs'] = glacier_cal_data.loc[cal_idx, 'mb_mwe']
                        glacier_cal_compare.loc[cal_idx, 'obs_unit'] = 'mwe'
                        # Modeled mass balance [mwe]
                        #  Sum(mass balance x area) / total area
                        t1_idx = glacier_cal_data.loc[cal_idx, 't1_idx'].astype(int)
                        t2_idx = glacier_cal_data.loc[cal_idx, 't2_idx'].astype(int)
                        z1_idx = glacier_cal_data.loc[cal_idx, 'z1_idx'].astype(int)
                        z2_idx = glacier_cal_data.loc[cal_idx, 'z2_idx'].astype(int)
                        year_idx = int(t1_idx / 12)
                        bin_area_subset = glac_bin_area_annual[z1_idx:z2_idx, year_idx]
                        glacier_cal_compare.loc[cal_idx, 'model'] = (
                                (glac_bin_massbalclim[z1_idx:z2_idx, t1_idx:t2_idx] *
                                 bin_area_subset[:,np.newaxis]).sum() / bin_area_subset.sum())
                        # Z-score for modeled mass balance based on observed mass balance and uncertainty
                        #  z-score = (model - measured) / uncertainty
                        glacier_cal_compare.loc[cal_idx, 'uncertainty'] = (input.massbal_uncertainty_mwea *
                                (glacier_cal_data.loc[cal_idx, 't2'] - glacier_cal_data.loc[cal_idx, 't1']))
                        glacier_cal_compare.loc[cal_idx, 'zscore'] = (
                                (glacier_cal_compare.loc[cal_idx, 'model'] - glacier_cal_compare.loc[cal_idx, 'obs']) /
                                glacier_cal_compare.loc[cal_idx, 'uncertainty'])

            # OPTIMIZATION ROUND #4: Isolate geodetic MB if necessary
            #  if there are multiple measurements and geodetic measurement still has a zscore greater than 1, then
            #  only calibrate the geodetic measurement since this provides longest snapshot of glacier
            if (glacier_cal_compare.obs_type.isin(['mb_geo']).any() == True) and (glacier_cal_compare.shape[0] > 1):
                zscore_compare = glacier_cal_compare.loc[glacier_cal_compare.index.values[np.where(
                        glacier_cal_compare['obs_type'] == 'mb_geo')[0][0]], 'zscore']
                zscore_tolerance = input.zscore_tolerance_all
                # Important to remain within this if loop as this is a special case
                if abs(zscore_compare) > zscore_tolerance:
                    # Select only geodetic for glacier calibration data
                    glacier_cal_data = pd.DataFrame(glacier_cal_data.loc[glacier_cal_data.index.values[np.where(
                            glacier_cal_data['obs_type'] == 'mb_geo')[0][0]]]).transpose()
                    # Run optimization
                    modelparameters_opt = minimize(objective, modelparameters_init, method=input.method_opt,
                                                   bounds=modelparameters_bnds, options={'ftol':input.ftol_opt})
                    # Record the calibration round
                    calround = calround + 1
                    # Record the optimized parameters
                    modelparameters_init = modelparameters_opt.x
                    main_glac_modelparamsopt[glac] = [modelparameters[0], modelparameters[1], modelparameters_init[0],
                             modelparameters_init[1], modelparameters_init[2],
                             modelparameters_init[2] / input.ddfsnow_iceratio, modelparameters[6], modelparameters_init[3]]
                    modelparameters = main_glac_modelparamsopt[glac]
                    # Re-run the optimized parameters in order to see the mass balance
                    # Mass balance calculations
                    (glac_bin_temp, glac_bin_prec, glac_bin_acc, glac_bin_refreeze, glac_bin_snowpack, glac_bin_melt,
                     glac_bin_frontalablation, glac_bin_massbalclim, glac_bin_massbalclim_annual, glac_bin_area_annual,
                     glac_bin_icethickness_annual, glac_bin_width_annual, glac_bin_surfacetype_annual,
                     glac_wide_massbaltotal, glac_wide_runoff, glac_wide_snowline, glac_wide_snowpack,
                     glac_wide_area_annual, glac_wide_volume_annual, glac_wide_ELA_annual) = (
                        massbalance.runmassbalance(modelparameters, glacier_rgi_table, glacier_area_t0, icethickness_t0,
                                                   width_t0, elev_bins, glacier_gcm_temp, glacier_gcm_prec,
                                                   glacier_gcm_elev, glacier_gcm_lrgcm, glacier_gcm_lrglac, dates_table,
                                                   option_areaconstant=1))
                    # Reset calibration data to all values for comparison
                    glacier_cal_data = ((cal_data.iloc[np.where(
                            glacier_rgi_table[input.rgi_O1Id_colname] == cal_data['glacno'])[0],:]).copy())
                    # Loop through all measurements
                    for x in range(glacier_cal_data.shape[0]):
                        cal_idx = glacier_cal_data.index.values[x]
                        # Mass balance comparisons
                        if ((glacier_cal_data.loc[cal_idx, 'obs_type'] == 'mb_geo') or
                            (glacier_cal_data.loc[cal_idx, 'obs_type'] == 'mb_glac')):
                            # Observed mass balance [mwe]
                            glacier_cal_compare.loc[cal_idx, 'obs'] = glacier_cal_data.loc[cal_idx, 'mb_mwe']
                            glacier_cal_compare.loc[cal_idx, 'obs_unit'] = 'mwe'
                            # Modeled mass balance [mwe]
                            #  Sum(mass balance x area) / total area
                            t1_idx = glacier_cal_data.loc[cal_idx, 't1_idx'].astype(int)
                            t2_idx = glacier_cal_data.loc[cal_idx, 't2_idx'].astype(int)
                            z1_idx = glacier_cal_data.loc[cal_idx, 'z1_idx'].astype(int)
                            z2_idx = glacier_cal_data.loc[cal_idx, 'z2_idx'].astype(int)
                            year_idx = int(t1_idx / 12)
                            bin_area_subset = glac_bin_area_annual[z1_idx:z2_idx, year_idx]
                            glacier_cal_compare.loc[cal_idx, 'model'] = (
                                    (glac_bin_massbalclim[z1_idx:z2_idx, t1_idx:t2_idx] *
                                     bin_area_subset[:,np.newaxis]).sum() / bin_area_subset.sum())
                            # Z-score for modeled mass balance based on observed mass balance and uncertainty
                            #  z-score = (model - measured) / uncertainty
                            glacier_cal_compare.loc[cal_idx, 'uncertainty'] = (input.massbal_uncertainty_mwea *
                                    (glacier_cal_data.loc[cal_idx, 't2'] - glacier_cal_data.loc[cal_idx, 't1']))
                            glacier_cal_compare.loc[cal_idx, 'zscore'] = (
                                    (glacier_cal_compare.loc[cal_idx, 'model'] - glacier_cal_compare.loc[cal_idx, 'obs']) /
                                    glacier_cal_compare.loc[cal_idx, 'uncertainty'])

            # Record output
            # Calibration round
            glacier_cal_compare['calround'] = calround
            # Model vs. observations
            main_glac_cal_compare.loc[glacier_cal_data.index.values] = glacier_cal_compare
            # Glacier-wide climatic mass balance over study period (used by transfer functions)
            main_glacwide_mbclim_mwe[glac] = (
                    (glac_bin_massbalclim * glac_bin_area_annual[:, 0][:,np.newaxis]).sum() / 
                    glac_bin_area_annual[:, 0].sum())
            
            print(count, main_glac_rgi.loc[main_glac_rgi.index.values[glac],'RGIId'])
            print('precfactor:', modelparameters[2])
            print('precgrad:', modelparameters[3])
            print('ddfsnow:', modelparameters[4])
            print('ddfice:', modelparameters[5])
            print('tempchange:', modelparameters[7])
            print('calround:', calround)
            print('modeled mass balance [mwe]:', glacier_cal_compare.loc[glacier_cal_data.index.values, 'model'].values)
            print('measured mass balance [mwe]:', glacier_cal_compare.loc[glacier_cal_data.index.values, 'obs'].values)
            print('zscore:', glacier_cal_compare.loc[glacier_cal_data.index.values, 'zscore'].values)
            print(' ')
            
        # ===== GROUP CALIBRATION ====
        # Indices of group calibration data
        cal_data_idx_groups = cal_data.loc[cal_data['group_name'].notnull()].index.values.tolist()
        # Indices of glaciers that have already been calibrated
        cal_individual_glacno_idx = [main_glac_rgi[main_glac_rgi.glacno == x].index.values[0] 
                                     for x in cal_individual_glacno.tolist()]
        # List of name of each group
        group_dict_keyslist_names = [item[0] for item in group_dict_keyslist]
        for cal_idx in cal_data_idx_groups:
            group_name = cal_data.loc[cal_idx, 'group_name']
            print(group_name)
    
            # Group dictionary keys list index
            group_dict_idx = group_dict_keyslist_names.index(group_name)
            # Indices of all glaciers in group
            group_dict_glaciers_idx_all = [main_glac_rgi[main_glac_rgi.RGIId == x].index.values[0] 
                                           for x in group_dict_keyslist[group_dict_idx][1]]
            # Indices of all glaciers in group excluding those already calibrated
            group_dict_glaciers_idx = [x for x in group_dict_glaciers_idx_all if x not in cal_individual_glacno_idx]
    
            # Observed mass balance [km3]
            glacier_cal_compare = main_glac_cal_compare.loc[cal_idx].copy()
            glacier_cal_compare.glacno = group_name
            glacier_cal_compare.obs_type = cal_data.loc[cal_idx, 'obs_type']
            if glacier_cal_compare.obs_type == 'mb':
                glacier_cal_compare.obs_unit = 'mwe'
                glacier_cal_compare.obs = cal_data.loc[cal_idx, 'mb_mwe']
                glacier_cal_compare.uncertainty = cal_data.loc[cal_idx, 'mb_mwe_err']
            
            # Record the calibration round
            calround = 0        
    
            # Loop through the glaciers in the group
            #  For model parameters, check if main_glac_modelparamsopt is zeros!
            #   --> if all zeros, then use the group model parameter
            #   --> if already has values, then use those values
     
            # INITIAL GUESS
            modelparameters_init = [input.precfactor, input.precgrad, input.ddfsnow, input.tempchange]
            # PARAMETER BOUNDS (Braithwaite, 2008 for DDFsnow)
            precfactor_bnds = (0.9,1.2)
            precgrad_bnds = (0.0001,0.00025)
            ddfsnow_bnds = (0.0036, 0.0046) 
            tempchange_bnds = (-1,1)
            modelparameters_bnds = (precfactor_bnds, precgrad_bnds, ddfsnow_bnds, tempchange_bnds)
            # OPTIMIZATION ROUND #1: optimize precfactor, DDFsnow, tempchange
            # Run the optimization
            #  'L-BFGS-B' - much slower
            #  'SLSQP' did not work for some geodetic measurements using the sum_abs_zscore.  One work around was to
            #    divide the sum_abs_zscore by 1000, which made it work in all cases.  However, methods were switched
            #    to 'L-BFGS-B', which may be slower, but is still effective.
            modelparameters_opt = minimize(objective_group, modelparameters_init, method=input.method_opt, 
                                           bounds=modelparameters_bnds, options={'ftol':input.ftol_opt})
            # Record the calibration round
            calround = calround + 1
            # Record the optimized parameters
            modelparameters_init = modelparameters_opt.x
            # Re-run to get data
            # Record group's cumulative area and mass balance for comparison
            group_cum_area_km2 = 0
            group_cum_mb_mkm2 = 0    
            # Loop through all glaciers
            for glac in range(main_glac_rgi.shape[0]):
                # Check if glacier is included in group
                if main_glac_rgi.loc[glac, 'group_name'] == group_name: 
                    # Set model parameters
                    if np.all(main_glac_modelparamsopt[glac] == 0) == False:
                        modelparameters = main_glac_modelparamsopt[glac]
                    else:
                        # if model parameters already exist for the glacier, then use those instead of group parameters
                        modelparameters = [input.lrgcm, input.lrglac, input.precfactor, input.precgrad, input.ddfsnow, 
                                           input.ddfice, input.tempsnow, input.tempchange]
                        # Use a subset of model parameters to reduce number of constraints required
                        modelparameters[2] = modelparameters_init[0]
                        modelparameters[3] = modelparameters_init[1]
                        modelparameters[4] = modelparameters_init[2]
                        modelparameters[5] = modelparameters[4] / input.ddfsnow_iceratio
                        modelparameters[7] = modelparameters_init[3]
                    # Select subsets of data
                    glacier_rgi_table = main_glac_rgi.loc[main_glac_rgi.index.values[glac], :]   
                    glacier_gcm_elev = gcm_elev[glac]
                    glacier_gcm_prec = gcm_prec[glac,:]
                    glacier_gcm_temp = gcm_temp[glac,:]
                    glacier_gcm_lrgcm = gcm_lr[glac,:]
                    glacier_gcm_lrglac = glacier_gcm_lrgcm.copy()
                    glacier_area_t0 = main_glac_hyps.iloc[glac,:].values.astype(float)   
                    icethickness_t0 = main_glac_icethickness.iloc[glac,:].values.astype(float)
                    width_t0 = main_glac_width.iloc[glac,:].values.astype(float)
                    # Mass balance calculations
                    (glac_bin_temp, glac_bin_prec, glac_bin_acc, glac_bin_refreeze, glac_bin_snowpack, glac_bin_melt, 
                     glac_bin_frontalablation, glac_bin_massbalclim, glac_bin_massbalclim_annual, glac_bin_area_annual, 
                     glac_bin_icethickness_annual, glac_bin_width_annual, glac_bin_surfacetype_annual, 
                     glac_wide_massbaltotal, glac_wide_runoff, glac_wide_snowline, glac_wide_snowpack, 
                     glac_wide_area_annual, glac_wide_volume_annual, glac_wide_ELA_annual) = (
                        massbalance.runmassbalance(modelparameters, glacier_rgi_table, glacier_area_t0, icethickness_t0, 
                                                   width_t0, elev_bins, glacier_gcm_temp, glacier_gcm_prec, 
                                                   glacier_gcm_elev, glacier_gcm_lrgcm, glacier_gcm_lrglac, dates_table, 
                                                   option_areaconstant=1, warn_calving=0))  
                    # Mass balance comparisons
                    # Modeled mass balance [mwe]
                    #  Sum(mass balance x area) / total area
                    t1_idx = cal_data.loc[cal_idx, 't1_idx'].astype(int)
                    t2_idx = cal_data.loc[cal_idx, 't2_idx'].astype(int)
                    z1_idx = 0
                    z2_idx = glac_bin_area_annual.shape[0]
                    year_idx = int(t1_idx / 12)
                    bin_area_subset = glac_bin_area_annual[z1_idx:z2_idx, year_idx]                    
                    group_cum_area_km2 = group_cum_area_km2 + bin_area_subset.sum()
                    group_cum_mb_mkm2 = (
                            group_cum_mb_mkm2 + 
                            (glac_bin_massbalclim[z1_idx:z2_idx, t1_idx:t2_idx] * bin_area_subset[:,np.newaxis]).sum())
                    # Glacier-wide climatic mass balance over study period (used by transfer functions)
                    main_glacwide_mbclim_mwe[glac] = (
                            (glac_bin_massbalclim * glac_bin_area_annual[:, 0][:,np.newaxis]).sum() / 
                             glac_bin_area_annual[:, 0].sum())
            # Z-score for modeled mass balance based on observed mass balance and uncertainty
            #  z-score = (model - measured) / uncertainty
            glacier_cal_compare.model = group_cum_mb_mkm2 / group_cum_area_km2
            
            glacier_cal_compare.zscore = (
                    (glacier_cal_compare.model - glacier_cal_compare.obs) / glacier_cal_compare.uncertainty)
            # Minimize the sum of differences
            abs_zscore = abs(glacier_cal_compare.zscore)
            
            # OPTIMIZATION ROUND #2:
            # Check if need to expand the bounds
            if abs_zscore > input.zscore_tolerance_single:
                # Bounds
                precfactor_bnds = (0.75,1.5)
                precgrad_bnds = (0.0001,0.00025)
                ddfsnow_bnds = (0.0031, 0.0051)
                tempchange_bnds = (-2,2)
                modelparameters_bnds = (precfactor_bnds, precgrad_bnds, ddfsnow_bnds, tempchange_bnds) 
                # Run optimization
                modelparameters_opt = minimize(objective_group, modelparameters_init, method=input.method_opt, 
                                               bounds=modelparameters_bnds, options={'ftol':input.ftol_opt})
                # Record the calibration round
                calround = calround + 1
                # Record the optimized parameters
                modelparameters_init = modelparameters_opt.x
                # Re-run to get data
                # Record group's cumulative area and mass balance for comparison
                group_cum_area_km2 = 0
                group_cum_mb_mkm2 = 0    
                # Loop through all glaciers
                for glac in range(main_glac_rgi.shape[0]):
                    # Check if glacier is included in group
                    if main_glac_rgi.loc[glac, 'group_name'] == group_name:        
                        # Set model parameters
                        if np.all(main_glac_modelparamsopt[glac] == 0) == False:
                            modelparameters = main_glac_modelparamsopt[glac]
                        else:
                            # if model parameters already exist for the glacier, then use those instead of group parameters
                            modelparameters = [input.lrgcm, input.lrglac, input.precfactor, input.precgrad, input.ddfsnow, 
                                               input.ddfice, input.tempsnow, input.tempchange]
                            # Use a subset of model parameters to reduce number of constraints required
                            modelparameters[2] = modelparameters_init[0]
                            modelparameters[3] = modelparameters_init[1]
                            modelparameters[4] = modelparameters_init[2]
                            modelparameters[5] = modelparameters[4] / input.ddfsnow_iceratio
                            modelparameters[7] = modelparameters_init[3]
                        # Select subsets of data
                        glacier_rgi_table = main_glac_rgi.loc[main_glac_rgi.index.values[glac], :]   
                        glacier_gcm_elev = gcm_elev[glac]
                        glacier_gcm_prec = gcm_prec[glac,:]
                        glacier_gcm_temp = gcm_temp[glac,:]
                        glacier_gcm_lrgcm = gcm_lr[glac,:]
                        glacier_gcm_lrglac = glacier_gcm_lrgcm.copy()
                        glacier_area_t0 = main_glac_hyps.iloc[glac,:].values.astype(float)   
                        icethickness_t0 = main_glac_icethickness.iloc[glac,:].values.astype(float)
                        width_t0 = main_glac_width.iloc[glac,:].values.astype(float)
                        # Mass balance calculations
                        (glac_bin_temp, glac_bin_prec, glac_bin_acc, glac_bin_refreeze, glac_bin_snowpack, glac_bin_melt, 
                         glac_bin_frontalablation, glac_bin_massbalclim, glac_bin_massbalclim_annual, glac_bin_area_annual, 
                         glac_bin_icethickness_annual, glac_bin_width_annual, glac_bin_surfacetype_annual, 
                         glac_wide_massbaltotal, glac_wide_runoff, glac_wide_snowline, glac_wide_snowpack, 
                         glac_wide_area_annual, glac_wide_volume_annual, glac_wide_ELA_annual) = (
                            massbalance.runmassbalance(modelparameters, glacier_rgi_table, glacier_area_t0, icethickness_t0, 
                                                       width_t0, elev_bins, glacier_gcm_temp, glacier_gcm_prec, 
                                                       glacier_gcm_elev, glacier_gcm_lrgcm, glacier_gcm_lrglac, dates_table, 
                                                       option_areaconstant=1, warn_calving=0))  
                        # Mass balance comparisons
                        # Modeled mass balance [mwe]
                        #  Sum(mass balance x area) / total area
                        t1_idx = cal_data.loc[cal_idx, 't1_idx'].astype(int)
                        t2_idx = cal_data.loc[cal_idx, 't2_idx'].astype(int)
                        z1_idx = 0
                        z2_idx = glac_bin_area_annual.shape[0]
                        year_idx = int(t1_idx / 12)
                        bin_area_subset = glac_bin_area_annual[z1_idx:z2_idx, year_idx]                    
                        group_cum_area_km2 = group_cum_area_km2 + bin_area_subset.sum()
                        group_cum_mb_mkm2 = (
                                group_cum_mb_mkm2 + 
                                (glac_bin_massbalclim[z1_idx:z2_idx, t1_idx:t2_idx] * bin_area_subset[:,np.newaxis]).sum())
                        # Glacier-wide climatic mass balance over study period (used by transfer functions)
                        main_glacwide_mbclim_mwe[glac] = (
                                (glac_bin_massbalclim * glac_bin_area_annual[:, 0][:,np.newaxis]).sum() / 
                                 glac_bin_area_annual[:, 0].sum())
                # Z-score for modeled mass balance based on observed mass balance and uncertainty
                #  z-score = (model - measured) / uncertainty
                glacier_cal_compare.model = group_cum_mb_mkm2 / group_cum_area_km2
                
                glacier_cal_compare.zscore = (
                        (glacier_cal_compare.model - glacier_cal_compare.obs) / glacier_cal_compare.uncertainty)
                # Minimize the sum of differences
                abs_zscore = abs(glacier_cal_compare.zscore)
            
            # OPTIMIZATION ROUND #3:
            # Check if need to expand the bounds
            if abs_zscore > input.zscore_tolerance_single:
                # Bounds
                precfactor_bnds = (0.5,2)
                precgrad_bnds = (0.0001,0.00025)
                ddfsnow_bnds = (0.0026, 0.0056)
                tempchange_bnds = (-5,5)
                modelparameters_bnds = (precfactor_bnds, precgrad_bnds, ddfsnow_bnds, tempchange_bnds) 
                # Run optimization
                modelparameters_opt = minimize(objective_group, modelparameters_init, method=input.method_opt, 
                                               bounds=modelparameters_bnds, options={'ftol':input.ftol_opt})
                # Record the calibration round
                calround = calround + 1
                # Record the optimized parameters
                modelparameters_init = modelparameters_opt.x
                modelparameters = modelparameters_init
                # Re-run to get data
                # Record group's cumulative area and mass balance for comparison
                group_cum_area_km2 = 0
                group_cum_mb_mkm2 = 0    
                # Loop through all glaciers
                for glac in range(main_glac_rgi.shape[0]):
                    # Check if glacier is included in group
                    if main_glac_rgi.loc[glac, 'group_name'] == group_name:        
                        # Set model parameters
                        if np.all(main_glac_modelparamsopt[glac] == 0) == False:
                            modelparameters = main_glac_modelparamsopt[glac]
                        else:
                            # if model parameters already exist for the glacier, then use those instead of group parameters
                            modelparameters = [input.lrgcm, input.lrglac, input.precfactor, input.precgrad, input.ddfsnow, 
                                               input.ddfice, input.tempsnow, input.tempchange]
                            # Use a subset of model parameters to reduce number of constraints required
                            modelparameters[2] = modelparameters_init[0]
                            modelparameters[3] = modelparameters_init[1]
                            modelparameters[4] = modelparameters_init[2]
                            modelparameters[5] = modelparameters[4] / input.ddfsnow_iceratio
                            modelparameters[7] = modelparameters_init[3]
                        # Select subsets of data
                        glacier_rgi_table = main_glac_rgi.loc[main_glac_rgi.index.values[glac], :]   
                        glacier_gcm_elev = gcm_elev[glac]
                        glacier_gcm_prec = gcm_prec[glac,:]
                        glacier_gcm_temp = gcm_temp[glac,:]
                        glacier_gcm_lrgcm = gcm_lr[glac,:]
                        glacier_gcm_lrglac = glacier_gcm_lrgcm.copy()
                        glacier_area_t0 = main_glac_hyps.iloc[glac,:].values.astype(float)   
                        icethickness_t0 = main_glac_icethickness.iloc[glac,:].values.astype(float)
                        width_t0 = main_glac_width.iloc[glac,:].values.astype(float)
                        # Mass balance calculations
                        (glac_bin_temp, glac_bin_prec, glac_bin_acc, glac_bin_refreeze, glac_bin_snowpack, glac_bin_melt, 
                         glac_bin_frontalablation, glac_bin_massbalclim, glac_bin_massbalclim_annual, glac_bin_area_annual, 
                         glac_bin_icethickness_annual, glac_bin_width_annual, glac_bin_surfacetype_annual, 
                         glac_wide_massbaltotal, glac_wide_runoff, glac_wide_snowline, glac_wide_snowpack, 
                         glac_wide_area_annual, glac_wide_volume_annual, glac_wide_ELA_annual) = (
                            massbalance.runmassbalance(modelparameters, glacier_rgi_table, glacier_area_t0, icethickness_t0, 
                                                       width_t0, elev_bins, glacier_gcm_temp, glacier_gcm_prec, 
                                                       glacier_gcm_elev, glacier_gcm_lrgcm, glacier_gcm_lrglac, dates_table, 
                                                       option_areaconstant=1, warn_calving=0))  
                        # Mass balance comparisons
                        # Modeled mass balance [mwe]
                        #  Sum(mass balance x area) / total area
                        t1_idx = cal_data.loc[cal_idx, 't1_idx'].astype(int)
                        t2_idx = cal_data.loc[cal_idx, 't2_idx'].astype(int)
                        z1_idx = 0
                        z2_idx = glac_bin_area_annual.shape[0]
                        year_idx = int(t1_idx / 12)
                        bin_area_subset = glac_bin_area_annual[z1_idx:z2_idx, year_idx]                    
                        group_cum_area_km2 = group_cum_area_km2 + bin_area_subset.sum()
                        group_cum_mb_mkm2 = (
                                group_cum_mb_mkm2 + 
                                (glac_bin_massbalclim[z1_idx:z2_idx, t1_idx:t2_idx] * bin_area_subset[:,np.newaxis]).sum())
                        # Glacier-wide climatic mass balance over study period (used by transfer functions)
                        main_glacwide_mbclim_mwe[glac] = (
                                (glac_bin_massbalclim * glac_bin_area_annual[:, 0][:,np.newaxis]).sum() / 
                                 glac_bin_area_annual[:, 0].sum())
                # Z-score for modeled mass balance based on observed mass balance and uncertainty
                #  z-score = (model - measured) / uncertainty
                glacier_cal_compare.model = group_cum_mb_mkm2 / group_cum_area_km2
                
                glacier_cal_compare.zscore = (
                        (glacier_cal_compare.model - glacier_cal_compare.obs) / glacier_cal_compare.uncertainty)
                # Minimize the sum of differences
                abs_zscore = abs(glacier_cal_compare.zscore)
                
            main_glac_modelparamsopt[group_dict_glaciers_idx] = (
                    [modelparameters[0], modelparameters[1], modelparameters_init[0], 
                     modelparameters_init[1], modelparameters_init[2], modelparameters_init[2] / input.ddfsnow_iceratio,
                     modelparameters[6], modelparameters_init[3]])
    
            glacier_cal_compare.calround = calround
            main_glac_cal_compare.loc[cal_idx] = glacier_cal_compare
            print(group_name,'(zscore):', abs_zscore)
            print('precfactor:', modelparameters[2])
            print('precgrad:', modelparameters[3])
            print('ddfsnow:', modelparameters[4])
            print('ddfice:', modelparameters[5])
            print('tempchange:', modelparameters[7])
            print('calround:', calround)
            print(' ')
    
        # ===== EXPORT OUTPUT =====
        # Export (i) main_glac_rgi w optimized model parameters and glacier-wide climatic mass balance,
        #        (ii) comparison of model vs. observations
        # Concatenate main_glac_rgi, optimized model parameters, glacier-wide climatic mass balance
        main_glac_output = main_glac_rgi.copy()
        main_glac_modelparamsopt_pd = pd.DataFrame(main_glac_modelparamsopt, columns=input.modelparams_colnames)
        main_glac_modelparamsopt_pd.index = main_glac_rgi.index.values
        main_glacwide_mbclim_pd = pd.DataFrame(main_glacwide_mbclim_mwe, columns=[input.mbclim_cn])
        main_glac_output = pd.concat([main_glac_output, main_glac_modelparamsopt_pd, main_glacwide_mbclim_pd], axis=1)
        
        # Export output
        if (input.option_calibration == 1) and (option_export == 1) and (('group' in cal_datasets) == True):
            # main_glac_rgi w model parameters
            modelparams_fn = ('cal_modelparams_opt' + str(input.option_calibration) + '_R' + str(input.rgi_regionsO1[0]) + '_' + 
                              gcm_name + '_' + str(input.startyear - input.spinupyears) + '_' + str(input.endyear) + 
                              '.csv')
            main_glac_output.to_csv(input.output_filepath + modelparams_fn)
            # calibration comparison
            calcompare_fn = ('cal_compare_opt' + str(input.option_calibration) + '_R' + str(input.rgi_regionsO1[0]) + '_' + 
                              gcm_name + '_' + str(input.startyear - input.spinupyears) + '_' + str(input.endyear) + 
                              '.csv')
            main_glac_cal_compare.to_csv(input.output_filepath + calcompare_fn)
        elif (input.option_calibration == 1) and (option_export == 1) and (('group' not in cal_datasets) == True):
            # main_glac_rgi w model parameters
            modelparams_fn = ('cal_modelparams_opt' + str(input.option_calibration) + '_R' + str(input.rgi_regionsO1[0]) + '_' +
                              gcm_name + '_' + str(input.startyear - input.spinupyears) + '_' + str(input.endyear) + 
                              '_' + str(count) + '.csv')
            main_glac_output.to_csv(input.output_filepath + modelparams_fn)
            # calibration comparison
            calcompare_fn = ('cal_compare_opt' + str(input.option_calibration) + '_R' + str(input.rgi_regionsO1[0]) + '_' +
                              gcm_name + '_' + str(input.startyear - input.spinupyears) + '_' + str(input.endyear) + 
                              '_' + str(count) + '.csv')
            main_glac_cal_compare.to_csv(input.output_filepath + calcompare_fn)
        
    # Export variables as global to view in variable explorer
    if (args.option_parallels == 0) or (main_glac_rgi_all.shape[0] < 2 * args.num_simultaneous_processes):
        global main_vars
        main_vars = inspect.currentframe().f_locals

    print('\nProcessing time of', gcm_name, 'for', count,':',time.time()-time_start, 's')

#%% PARALLEL PROCESSING
if __name__ == '__main__':
    time_start = time.time()
    parser = getparser()
    args = parser.parse_args()

    # Reference GCM name
    gcm_name = args.ref_gcm_name
    print('Reference climate data is:', gcm_name)

    # Select all glaciers in a region
    main_glac_rgi_all = modelsetup.selectglaciersrgitable(rgi_regionsO1=input.rgi_regionsO1, rgi_regionsO2 = 'all',
                                                          rgi_glac_number=input.rgi_glac_number)
    # Define chunk size for parallel processing
    if args.option_parallels != 0:
        num_cores = int(np.min([main_glac_rgi_all.shape[0], args.num_simultaneous_processes]))
        chunk_size = int(np.ceil(main_glac_rgi_all.shape[0] / num_cores))
    else:
        # if not running in parallel, chunk size is all glaciers
        chunk_size = main_glac_rgi_all.shape[0]

    # Pack variables for parallel processing
    list_packed_vars = []
    n = 0
    for chunk in range(0, main_glac_rgi_all.shape[0], chunk_size):
        n = n + 1
        list_packed_vars.append([n, chunk, chunk_size, main_glac_rgi_all, gcm_name])

    # if MCMC option, clear files from previous run
    if input.option_calibration == 2:
        # clear MCMC/config/ directory for storing netcdf files
        # for each glacier run. These files will then
        # be combined for the final output, but need to be
        # cleared from the previous run.
        filelist = glob.glob(os.path.join(input.mcmc_output_netcdf_fp, '*.nc'))
        for f in filelist:
            os.remove(f)

    # Parallel processing
    if args.option_parallels != 0:
        print('Processing in parallel with ' + str(num_cores) + ' cores...')
        with multiprocessing.Pool(args.num_simultaneous_processes) as p:
            p.map(main,list_packed_vars)
    # If not in parallel, then only should be one loop
    else:
        for n in range(len(list_packed_vars)):
            main(list_packed_vars[n])

    # if MCMC option, consolidate output into one file for netcdf and csv results
    if input.option_calibration == 2:
        print('combine netcdfs in post-processing for mcmc methods')
#        # Combine csv output into single csv
#        csv_list = []
#        for i in os.listdir(input.mcmc_output_csv_fp):
#            # Append results
#            if i.endswith('.csv') == True:
#                csv_list.append(i)
#        # Column names
#        csv_cn_endings = ['_mean', '_std', '_mc_err', '_95lowhpd', '_95highhpd', '_q2pt5', '_q25', '_q50', '_q75', 
#                          '_q97pt5', 'n_eff']
#        if input.n_chains > 1:
#            csv_cn_endings.append('_gelmanrubin')
#        csv_cns = ['RGIId']
#        for vn in variables:
#            for ending in csv_cn_endings:
#                csv_cns.append(vn + ending)    
#        # Output file
#        count = 0
#        csv_output = pd.DataFrame(columns=csv_cns)
#        for i in csv_list:
#            # Load csv
#            csv_input = pd.read_csv(input.mcmc_output_csv_fp + i)
#            # Append to output csv
#            csv_output.loc[count,'RGIId'] = i[0:8]
#            ncols = csv_input.shape[1] - 1
#            csv_output.iloc[count, 0*ncols+1:1*ncols+1] = csv_input.iloc[0,1:].values
#            csv_output.iloc[count, 1*ncols+1:2*ncols+1] = csv_input.iloc[1,1:].values
#            csv_output.iloc[count, 2*ncols+1:3*ncols+1] = csv_input.iloc[2,1:].values
#            csv_output.iloc[count, 3*ncols+1:4*ncols+1] = csv_input.iloc[3,1:].values
#            count = count + 1
#            # Remove files
#            os.remove(input.mcmc_output_csv_fp + i)
#        # Export csv
#        csv_output.to_csv(input.mcmc_output_fp + input.mcmc_output_csv_fn, index=False)
#
#        # create a dict for dataarrays
#        da_dict = {}
#
#        # for each .nc file in folder, upload dataset
#        for i in os.listdir(input.mcmc_output_netcdf_fp):
#            if i.endswith('.nc'):
#                glacier_RGIId = i[:-3]
#                ds = xr.open_dataset(input.mcmc_output_netcdf_fp + i)
#
#                # get dataarray, add to dictionary
#                da = ds[glacier_RGIId]
#                da_dict[glacier_RGIId] = da
#                
#                ds.close()
#
#                if debug:
#                    print(da)
#
#        # create final dataset with each glacier, make netcdf file
#        ds = xr.Dataset(da_dict)
#        ds.to_netcdf(input.mcmc_output_fp + input.mcmc_output_filename)
#
#        if debug:
#            print(ds)

    else:
        # Combine output into single csv
        if ((args.option_parallels != 0) and (main_glac_rgi_all.shape[0] >= 2 * args.num_simultaneous_processes) and
            (option_export == 1)):
            # Model parameters
            output_prefix = ('cal_modelparams_opt' + str(input.option_calibration) + '_R' + str(input.rgi_regionsO1[0]) + 
                             '_' + gcm_name + '_' + str(input.startyear - input.spinupyears) + '_' + str(input.endyear) 
                             + '_')
            output_list = []
            for i in os.listdir(output_filepath):
                # Append results
                if i.startswith(output_prefix) == True:
                    output_list.append(i)
                    if len(output_list) == 1:
                        output_all = pd.read_csv(output_filepath + i, index_col=0)
                    else:
                        output_2join = pd.read_csv(output_filepath + i, index_col=0)
                        output_all = output_all.append(output_2join, ignore_index=True)
                    # Remove file after its been merged
                    os.remove(output_filepath + i)
            # Export joined files
            output_all_fn = (str(strftime("%Y%m%d")) + '_cal_modelparams_opt' + str(input.option_calibration) + '_R' +
                             str(input.rgi_regionsO1[0]) + '_' + gcm_name + '_' + 
                             str(input.startyear - input.spinupyears) + '_' + str(input.endyear) + '.csv')
            output_all.to_csv(output_filepath + output_all_fn)

            # Calibration comparison
            output_prefix2 = ('cal_compare_opt' + str(input.option_calibration) + '_R' + str(input.rgi_regionsO1[0]) + '_' +
                              gcm_name + '_' + str(input.startyear - input.spinupyears) + '_' + str(input.endyear) + 
                              '_')
            output_list = []
            for i in os.listdir(output_filepath):
                # Append results
                if i.startswith(output_prefix2) == True:
                    output_list.append(i)
                    if len(output_list) == 1:
                        output_all = pd.read_csv(output_filepath + i, index_col=0)
                    else:
                        output_2join = pd.read_csv(output_filepath + i, index_col=0)
                        output_all = output_all.append(output_2join, ignore_index=True)
                    # Remove file after its been merged
                    os.remove(output_filepath + i)
            # Export joined files
            output_all_fn = (str(strftime("%Y%m%d")) + '_cal_compare_opt' + str(input.option_calibration) + '_R' +
                             str(input.rgi_regionsO1[0]) + '_' + gcm_name + '_' + 
                             str(input.startyear - input.spinupyears) + '_' + str(input.endyear) + '.csv')
            output_all.to_csv(output_filepath + output_all_fn)

    print('Total processing time:', time.time()-time_start, 's')

    #%% ===== PLOTTING AND PROCESSING FOR MODEL DEVELOPMENT =====
#    # Place local variables in variable explorer
#    if input.option_calibration == 1:
#        if (args.option_parallels == 0) or (main_glac_rgi_all.shape[0] < 2 * args.num_simultaneous_processes):
#            main_vars_list = list(main_vars.keys())
#            gcm_name = main_vars['gcm_name']
#            main_glac_rgi = main_vars['main_glac_rgi']
#            main_glac_hyps = main_vars['main_glac_hyps']
#            main_glac_icethickness = main_vars['main_glac_icethickness']
#            main_glac_width = main_vars['main_glac_width']
#            elev_bins = main_vars['elev_bins']
#            dates_table = main_vars['dates_table']
#            dates_table_nospinup = main_vars['dates_table_nospinup']
#            cal_data = main_vars['cal_data']
#            gcm_temp = main_vars['gcm_temp']
#            gcm_prec = main_vars['gcm_prec']
#            gcm_elev = main_vars['gcm_elev']
#            glac_bin_acc = main_vars['glac_bin_acc']
#            glac_bin_temp = main_vars['glac_bin_temp']
#            glac_bin_massbalclim = main_vars['glac_bin_massbalclim']
#            modelparameters = main_vars['modelparameters']
#            glac_bin_area_annual = main_vars['glac_bin_area_annual']
#            glacier_cal_compare = main_vars['glacier_cal_compare']
#            main_glac_cal_compare = main_vars['main_glac_cal_compare']
#            main_glac_modelparamsopt = main_vars['main_glac_modelparamsopt']
#            main_glac_output = main_vars['main_glac_output']
#            main_glac_modelparamsopt_pd = main_vars['main_glac_modelparamsopt_pd']
#            main_glacwide_mbclim = main_vars['main_glacwide_mbclim']
#            glac_wide_massbaltotal = main_vars['glac_wide_massbaltotal']
#            glac_wide_area_annual = main_vars['glac_wide_area_annual']
#            glac_wide_volume_annual = main_vars['glac_wide_volume_annual']
#            glacier_rgi_table = main_vars['glacier_rgi_table']
#            main_glac_modelparamsopt = main_vars['main_glac_modelparamsopt']
#            main_glac_massbal_compare = main_vars['main_glac_massbal_compare']
#            main_glac_output = main_vars['main_glac_output']
