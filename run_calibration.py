"""Run the model calibration"""
# Spyder cannot run parallels, so always set -option_parallels=0 when testing in Spyder.

# Built-in libraries
import os
import argparse
import multiprocessing
import resource
import time
import inspect
# External libraries
from datetime import datetime
import pandas as pd
import numpy as np
import xarray as xr
import pymc
from pymc import deterministic
from scipy.optimize import minimize
import pickle
from scipy import stats
# Local libraries
import pygem.pygem_input as pygem_prms
import pygemfxns_modelsetup as modelsetup
import pygemfxns_massbalance as massbalance
import class_climate
import class_mbdata

#from memory_profiler import profile

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
#    spc_region (optional) : str
#        RGI region number for supercomputer
    rgi_glac_number_fn : str
        filename of .pkl file containing a list of glacier numbers which is used to run batches on the supercomputer
    rgi_glac_number : str
        rgi glacier number to run for supercomputer
    progress_bar : int
        Switch for turning the progress bar on or off (default = 0 (off))
    debug : int
        Switch for turning debug printing on or off (default = 0 (off))

    Returns
    -------
    Object containing arguments and their respective values.
    """
    parser = argparse.ArgumentParser(description="run calibration in parallel")
    # add arguments
    parser.add_argument('-ref_gcm_name', action='store', type=str, default=pygem_prms.ref_gcm_name,
                        help='reference gcm name')
    parser.add_argument('-num_simultaneous_processes', action='store', type=int, default=4,
                        help='number of simultaneous processes (cores) to use')
    parser.add_argument('-option_parallels', action='store', type=int, default=1,
                        help='Switch to use or not use parallels (1 - use parallels, 0 - do not)')
    parser.add_argument('-rgi_glac_number_fn', action='store', type=str, default=None,
                        help='Filename containing list of rgi_glac_number, helpful for running batches on spc')
    parser.add_argument('-progress_bar', action='store', type=int, default=0,
                        help='Boolean for the progress bar to turn it on or off (default 0 is off)')
    parser.add_argument('-debug', action='store', type=int, default=0,
                        help='Boolean for debugging to turn it on or off (default 0 is off)')
    parser.add_argument('-rgi_glac_number', action='store', type=str, default=None,
                        help='rgi glacier number for supercomputer')
    return parser


def mb_mwea_calc(modelparameters, glacier_rgi_table, glacier_area_initial, icethickness_initial, width_initial, 
                 elev_bins, glacier_gcm_temp, glacier_gcm_tempstd, glacier_gcm_prec, glacier_gcm_elev, 
                 glacier_gcm_lrgcm, glacier_gcm_lrglac, dates_table, t1_idx, t2_idx, t1, t2,
                 option_areaconstant=1, return_tc_mustmelt=0, return_volremaining=0):
    """
    Run the mass balance and calculate the mass balance [mwea]

    Parameters
    ----------
    option_areaconstant : int
        Switch to keep area constant (1) or not (0)

    Returns
    -------
    mb_mwea : float
        mass balance [m w.e. a-1]
    """
    # Number of constant years
    startyear_doy = (pd.to_datetime(pd.DataFrame({'year':[dates_table.loc[0,'date'].year], 
                                                 'month':[dates_table.loc[0,'date'].month], 
                                                 'day':[dates_table.loc[0,'date'].day]}))
                                   .dt.strftime("%j").astype(float).values[0])
    startyear_daysinyear = (
            (pd.to_datetime(pd.DataFrame({'year':[dates_table.loc[0,'date'].year], 'month':[12], 'day':[31]})) - 
             pd.to_datetime(pd.DataFrame({'year':[dates_table.loc[0,'date'].year], 'month':[1], 'day':[1]})))
            .dt.days + 1).values[0]
    startyear_decimal = dates_table.loc[0,'date'].year + startyear_doy / startyear_daysinyear        
    constantarea_years = int(t1 - startyear_decimal)
    
    # Mass balance calculations
    (glac_bin_temp, glac_bin_prec, glac_bin_acc, glac_bin_refreeze, glac_bin_snowpack, glac_bin_melt,
     glac_bin_frontalablation, glac_bin_massbalclim, glac_bin_massbalclim_annual, glac_bin_area_annual,
     glac_bin_icethickness_annual, glac_bin_width_annual, glac_bin_surfacetype_annual,
     glac_wide_massbaltotal, glac_wide_runoff, glac_wide_snowline, glac_wide_snowpack,
     glac_wide_area_annual, glac_wide_volume_annual, glac_wide_ELA_annual, offglac_wide_prec,
     offglac_wide_refreeze, offglac_wide_melt, offglac_wide_snowpack, offglac_wide_runoff) = (
        massbalance.runmassbalance(modelparameters, glacier_rgi_table, glacier_area_initial, icethickness_initial,
                                   width_initial, elev_bins, glacier_gcm_temp, glacier_gcm_tempstd, glacier_gcm_prec,
                                   glacier_gcm_elev, glacier_gcm_lrgcm, glacier_gcm_lrglac, dates_table,
                                   option_areaconstant=option_areaconstant, constantarea_years=constantarea_years,
                                   debug=False)) 
    # Option to return must melt condition
    if return_tc_mustmelt == 1:
        # Climatic mass balance of lowermost bin must be negative at some point
        glac_bin_area_annual_mask = glac_bin_area_annual.copy()
        glac_bin_area_annual_mask[glac_bin_area_annual_mask>0] = 1
        lowestbin_idx = np.argmax(glac_bin_area_annual_mask > 0, axis=0)
        lowestbin_mbclim_annual = (
                glac_bin_massbalclim_annual[list(lowestbin_idx)[:-1], np.arange(0,lowestbin_idx.shape[0]-1)])
        nyears_negmbclim = np.sum([1 if x < 0 else 0 for x in lowestbin_mbclim_annual])
        return nyears_negmbclim
    elif return_volremaining == 1:
        # Ensure volume by end of century is zero
        # Compute glacier volume change for every time step and use this to compute mass balance
        glac_wide_area = glac_wide_area_annual[:-1].repeat(12)
        # Mass change [km3 mwe]
        #  mb [mwea] * (1 km / 1000 m) * area [km2]
        glac_wide_masschange = glac_wide_massbaltotal / 1000 * glac_wide_area
        # Mean annual mass balance [mwea]
        mb_mwea = glac_wide_masschange[t1_idx:t2_idx+1].sum() / glac_wide_area[0] * 1000 / (t2 - t1)
        t2_yearidx = int(np.ceil(t2 - startyear_decimal))
        return mb_mwea, glac_wide_volume_annual[t2_yearidx]
    # Return mass balance
    else:
        # Compute glacier volume change for every time step and use this to compute mass balance
        glac_wide_area = glac_wide_area_annual[:-1].repeat(12)
        # Mass change [km3 mwe]
        #  mb [mwea] * (1 km / 1000 m) * area [km2]
        glac_wide_masschange = glac_wide_massbaltotal / 1000 * glac_wide_area
        # Mean annual mass balance [mwea]
        mb_mwea = glac_wide_masschange[t1_idx:t2_idx+1].sum() / glac_wide_area[0] * 1000 / (t2 - t1)
        return mb_mwea


def retrieve_priors(modelparameters, glacier_rgi_table, glacier_area_initial, icethickness_initial, width_initial, 
                    elev_bins, glacier_gcm_temp, glacier_gcm_tempstd, glacier_gcm_prec, glacier_gcm_elev, 
                    glacier_gcm_lrgcm, glacier_gcm_lrglac, dates_table, t1_idx, t2_idx, t1, t2, debug=False):
    """
    Calculate parameters for prior distributions for the MCMC analysis

    Parameters
    ----------
    modelparameters : np.array
        glacier model parameters
    glacier_rgi_table : pd.DataFrame
        table of RGI information for a particular glacier
    glacier_area_initial, icethickness_initial, width_initial, elev_bins : np.arrays
        relevant glacier properties data
    glacier_gcm_temp, glacier_gcm_prec, glacier_gcm_elev, glacier_gcm_lrgcm, glacier_gcm_lrglac : np.arrays
        relevant glacier climate data
    dates_table : pd.DataFrame
        table of date/time information
    observed_massbal, mb_obs_min, mb_obs_max, t1, t2, t1_idx, t2_idx: floats (all except _idx) and integers (_idx)
        values related to the mass balance observations and their proper date/time indices
    debug : boolean
        switch to debug the function or not (default=False)
    return_tc_mustmelt : integer
        switch to return the mass balance (default, 0) or number of years lowermost bin has negative mass balance (1)

    Returns
    -------
    precfactor_boundlow, precfactor_boundhigh, precfactor_mu, precfactor_start : floats
        data for precipitation factor's prior distribution
    tempchange_boundlow, tempchange_boundhigh, tempchange_mu, tempchange_sigma, tempchange_start : floats
        data for temperature bias' prior distribution
    tempchange_max_loss, tempchange_max_acc, mb_max_loss, mb_max_acc : floats
        temperature change and mass balance associated with maximum accumulation and maximum loss
    """

    # ----- TEMPBIAS: max accumulation -----
    # Lower temperature bound based on max positive mass balance adjusted to avoid edge effects
    # Temperature at the lowest bin
    #  T_bin = T_gcm + lr_gcm * (z_ref - z_gcm) + lr_glac * (z_bin - z_ref) + tempchange
    lowest_bin = np.where(glacier_area_initial > 0)[0][0]
    tempchange_max_acc = (-1 * (glacier_gcm_temp + glacier_gcm_lrgcm *
                                (elev_bins[lowest_bin] - glacier_gcm_elev)).max())

    if debug:
        print('tc_max_acc:', np.round(tempchange_max_acc,2))
        
    # ----- TEMPBIAS: UPPER BOUND -----
    # MAXIMUM LOSS - AREA EVOLVING
    #  note: the mb_mwea_calc function ensures the area is constant until t1 such that the glacier is not completely
    #        lost before t1; otherwise, this will fail at high TC values
    mb_max_loss = (-1 * (glacier_area_initial * icethickness_initial).sum() / glacier_area_initial.sum() *
                   pygem_prms.density_ice / pygem_prms.density_water / (t2 - t1))

    if debug:
        print('mb_max_loss:', np.round(mb_max_loss,2), 'precfactor:', np.round(modelparameters[2],2))

    # Looping forward and backward to ensure optimization does not get stuck
    modelparameters[7] = tempchange_max_acc    
    mb_mwea_1, vol_remaining = mb_mwea_calc(
            modelparameters, glacier_rgi_table, glacier_area_initial, icethickness_initial, width_initial, elev_bins, 
            glacier_gcm_temp, glacier_gcm_tempstd, glacier_gcm_prec, glacier_gcm_elev, glacier_gcm_lrgcm, 
            glacier_gcm_lrglac, dates_table, t1_idx, t2_idx, t1, t2, 
            option_areaconstant=0, return_volremaining=1)
    
    # use absolute value because with area evolving the maximum value is a limit
    while vol_remaining > 0:
        modelparameters[7] = modelparameters[7] + 1        
        mb_mwea_1, vol_remaining = mb_mwea_calc(
                modelparameters, glacier_rgi_table, glacier_area_initial, icethickness_initial, width_initial, 
                elev_bins, glacier_gcm_temp, glacier_gcm_tempstd, glacier_gcm_prec, glacier_gcm_elev, glacier_gcm_lrgcm, 
                glacier_gcm_lrglac, dates_table, t1_idx, t2_idx, t1, t2, 
                option_areaconstant=0, return_volremaining=1)

        if debug:
            print('mb_mwea_1:', np.round(mb_mwea_1,2), 'TC:', np.round(modelparameters[7],2), 
                  'mb_max_loss:', np.round(mb_max_loss,2), 'vol_left:', np.round(vol_remaining,4))
    # Looping backward for tempchange at max loss
    while vol_remaining == 0:
        modelparameters[7] = modelparameters[7] - 0.05
        mb_mwea_1, vol_remaining = mb_mwea_calc(
                modelparameters, glacier_rgi_table, glacier_area_initial, icethickness_initial, width_initial, 
                elev_bins, glacier_gcm_temp, glacier_gcm_tempstd, glacier_gcm_prec, glacier_gcm_elev, glacier_gcm_lrgcm, 
                glacier_gcm_lrglac, dates_table, t1_idx, t2_idx, t1, t2, 
                option_areaconstant=0, return_volremaining=1)

        if debug:
            print('vol_left:', np.round(vol_remaining,4), 'mb_mwea_1:', np.round(mb_mwea_1,2), 
                  'TC:', np.round(modelparameters[7],2))

    tempchange_max_loss = modelparameters[7]
    tempchange_boundhigh = tempchange_max_loss

    if debug:
        print('tc_max_loss:', np.round(tempchange_max_loss,2), 'mb_max_loss:', np.round(mb_max_loss,2))

    # Lower bound based on must melt condition
    #  note: since the mass balance ablation is conditional on the glacier evolution, there can be cases where higher
    #  temperature biases still have 0 for nyears_negmbclim. Hence, the need to loop beyond the first instance, and
    #  then go back and check that you're using the good cases from there onward. This ensures starting point is good
    modelparameters[7] = tempchange_max_acc
    nyears_negmbclim = mb_mwea_calc(
                modelparameters, glacier_rgi_table, glacier_area_initial, icethickness_initial, width_initial, 
                elev_bins, glacier_gcm_temp, glacier_gcm_tempstd, glacier_gcm_prec, glacier_gcm_elev, glacier_gcm_lrgcm, 
                glacier_gcm_lrglac, dates_table, t1_idx, t2_idx, t1, t2, 
                option_areaconstant=0, return_tc_mustmelt=1)
    
    nyears_negmbclim_list = [nyears_negmbclim]
    tc_negmbclim_list = [modelparameters[7]]
    tc_smallstep_switch = False
    while nyears_negmbclim < 10 and modelparameters[7] < tempchange_max_loss:
        # Switch from large to small step sizes to speed up calculations
        if tc_smallstep_switch == False:
            tc_stepsize = 1
        else:
            tc_stepsize = 0.05
            
        modelparameters_old = modelparameters[7] 
        modelparameters[7] += tc_stepsize
        nyears_negmbclim = mb_mwea_calc(
                modelparameters, glacier_rgi_table, glacier_area_initial, icethickness_initial, width_initial, 
                elev_bins, glacier_gcm_temp, glacier_gcm_tempstd, glacier_gcm_prec, glacier_gcm_elev, glacier_gcm_lrgcm, 
                glacier_gcm_lrglac, dates_table, t1_idx, t2_idx, t1, t2, 
                option_areaconstant=0, return_tc_mustmelt=1)
        
        # Record if using big step and no there is no melt or if using small step and there is melt
        if nyears_negmbclim == 0 or (nyears_negmbclim > 0 and tc_smallstep_switch == True):
            nyears_negmbclim_list.append(nyears_negmbclim)
            tc_negmbclim_list.append(modelparameters[7])
        
        # First time nyears_negmbclim is > 0, flip the switch to use smalll step and restart with last tempchange
        if nyears_negmbclim > 0 and tc_smallstep_switch == False:
            tc_smallstep_switch = True
            modelparameters[7] = modelparameters_old
            nyears_negmbclim = 0

        if debug:
            print('TC:', np.round(modelparameters[7],2), 'nyears_negmbclim:', nyears_negmbclim)

    tempchange_boundlow = tc_negmbclim_list[np.where(np.array(nyears_negmbclim_list) == 0)[0][-1] + 1]

    return tempchange_boundlow, tempchange_boundhigh, mb_max_loss


def main(list_packed_vars):
    """
    Model calibration

    Parameters
    ----------
    list_packed_vars : list
        list of packed variables that enable the use of parallels

    Returns
    -------
    netcdf files of the calibration output
        Depending on the calibration scheme additional output may be exported as well
    """
    # Unpack variables
    count = list_packed_vars[0]
    gcm_name = list_packed_vars[1]
    main_glac_rgi = list_packed_vars[2]
    main_glac_hyps = list_packed_vars[3]
    main_glac_icethickness = list_packed_vars[4]
    main_glac_width = list_packed_vars[5]
    gcm_temp = list_packed_vars[6]
    gcm_tempstd = list_packed_vars[7]
    gcm_prec = list_packed_vars[8]
    gcm_elev = list_packed_vars[9]
    gcm_lr = list_packed_vars[10]
    cal_data = list_packed_vars[11]

    time_start = time.time()
    parser = getparser()
    args = parser.parse_args()

    if args.debug == 1:
        debug = True
    else:
        debug = False

    # ===== CALIBRATION =====
    # Option 2: use MCMC method to determine posterior probability distributions of the three parameters tempchange,
    #           ddfsnow and precfactor. Then create an ensemble of parameter sets evenly sampled from these
    #           distributions, and output these sets of parameters and their corresponding mass balances to be used in
    #           the simulations.
    if pygem_prms.option_calibration == 2:

        # ===== Define functions needed for MCMC method =====
        def run_MCMC(precfactor_disttype=pygem_prms.precfactor_disttype,
                     precfactor_gamma_alpha=pygem_prms.precfactor_gamma_alpha,
                     precfactor_gamma_beta=pygem_prms.precfactor_gamma_beta,
                     precfactor_lognorm_mu=pygem_prms.precfactor_lognorm_mu,
                     precfactor_lognorm_tau=pygem_prms.precfactor_lognorm_tau,
                     precfactor_mu=pygem_prms.precfactor_mu, precfactor_sigma=pygem_prms.precfactor_sigma,
                     precfactor_boundlow=pygem_prms.precfactor_boundlow, precfactor_boundhigh=pygem_prms.precfactor_boundhigh,
                     precfactor_start=pygem_prms.precfactor_start,
                     tempchange_disttype=pygem_prms.tempchange_disttype,
                     tempchange_mu=pygem_prms.tempchange_mu, tempchange_sigma=pygem_prms.tempchange_sigma,
                     tempchange_boundlow=pygem_prms.tempchange_boundlow, tempchange_boundhigh=pygem_prms.tempchange_boundhigh,
                     tempchange_start=pygem_prms.tempchange_start,
                     ddfsnow_disttype=pygem_prms.ddfsnow_disttype,
                     ddfsnow_mu=pygem_prms.ddfsnow_mu, ddfsnow_sigma=pygem_prms.ddfsnow_sigma,
                     ddfsnow_boundlow=pygem_prms.ddfsnow_boundlow, ddfsnow_boundhigh=pygem_prms.ddfsnow_boundhigh,
                     ddfsnow_start=pygem_prms.ddfsnow_start,
                     iterations=10, burn=0, thin=pygem_prms.thin_interval, tune_interval=1000, step=None,
                     tune_throughout=True, save_interval=None, burn_till_tuned=False, stop_tuning_after=5, verbose=0,
                     progress_bar=args.progress_bar, dbname=None,
                     use_potentials=1, mb_max_loss=None
                     ):
            """
            Runs the MCMC algorithm.

            Runs the MCMC algorithm by setting the prior distributions and calibrating the probability distributions of
            three parameters for the mass balance function.

            Parameters
            ----------
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
            precfactor_start : float
                Starting value of precipitation factor for sampling iterations (default assigned from input)
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
            tempchange_start : float
                Starting value of temperature change for sampling iterations (default assigned from input)
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
                'thin' argument. This means that the total number of iteration is updated throughout the sampling
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
            use_potentials : int
                Switch to turn off(0) or on (1) use of potential functions to further constrain likelihood functionns
            mb_max_loss : float
                Mass balance [mwea] at which the glacier completely melts

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
            # ===== PRIOR DISTRIBUTIONS =====
            # Precipitation factor [-]
            if precfactor_disttype == 'gamma':
                precfactor = pymc.Gamma('precfactor', alpha=precfactor_gamma_alpha, beta=precfactor_gamma_beta,
                                        value=precfactor_start)
            elif precfactor_disttype =='lognormal':
                #  lognormal distribution (roughly 0.3 to 3)
                precfactor_start = np.exp(precfactor_start)
                precfactor = pymc.Lognormal('precfactor', mu=precfactor_lognorm_mu, tau=precfactor_lognorm_tau,
                                            value=precfactor_start)
            elif precfactor_disttype == 'uniform':
                precfactor = pymc.Uniform('precfactor', lower=precfactor_boundlow, upper=precfactor_boundhigh,
                                          value=precfactor_start)
            # Temperature change [degC]
            if tempchange_disttype == 'normal':
                tempchange = pymc.Normal('tempchange', mu=tempchange_mu, tau=1/(tempchange_sigma**2),
                                         value=tempchange_start)
            elif tempchange_disttype =='truncnormal':
                tempchange = pymc.TruncatedNormal('tempchange', mu=tempchange_mu, tau=1/(tempchange_sigma**2),
                                                  a=tempchange_boundlow, b=tempchange_boundhigh, value=tempchange_start)
            elif tempchange_disttype =='uniform':
                tempchange = pymc.Uniform('tempchange', lower=tempchange_boundlow, upper=tempchange_boundhigh,
                                          value=tempchange_start)

            # Degree day factor of snow [mwe degC-1 d-1]
            #  always truncated normal distribution with mean 0.0041 mwe degC-1 d-1 and standard deviation of 0.0015
            #  (Braithwaite, 2008), since it's based on data; uniform should only be used for testing
            if ddfsnow_disttype == 'truncnormal':
                ddfsnow = pymc.TruncatedNormal('ddfsnow', mu=ddfsnow_mu, tau=1/(ddfsnow_sigma**2), a=ddfsnow_boundlow,
                                               b=ddfsnow_boundhigh, value=ddfsnow_start)
            if ddfsnow_disttype == 'uniform':
                ddfsnow = pymc.Uniform('ddfsnow', lower=ddfsnow_boundlow, upper=ddfsnow_boundhigh,
                                       value=ddfsnow_start)

            # ===== DETERMINISTIC FUNCTION ====
            # Define deterministic function for MCMC model based on our a priori probobaility distributions.
            @deterministic(plot=False)
            def massbal(tempchange=tempchange, precfactor=precfactor, ddfsnow=ddfsnow):
                """
                Likelihood function for mass balance [mwea] based on model parameters
                """
                modelparameters_copy = modelparameters.copy()
                if tempchange is not None:
                    modelparameters_copy[7] = float(tempchange)
                if precfactor is not None:
                    modelparameters_copy[2] = float(precfactor)
                if ddfsnow is not None:
                    modelparameters_copy[4] = float(ddfsnow)
                    # Degree day factor of ice is proportional to ddfsnow
                    modelparameters_copy[5] = modelparameters_copy[4] / pygem_prms.ddfsnow_iceratio
                # Mass balance calculations
                (glac_bin_temp, glac_bin_prec, glac_bin_acc, glac_bin_refreeze, glac_bin_snowpack, glac_bin_melt,
                 glac_bin_frontalablation, glac_bin_massbalclim, glac_bin_massbalclim_annual, glac_bin_area_annual,
                 glac_bin_icethickness_annual, glac_bin_width_annual, glac_bin_surfacetype_annual,
                 glac_wide_massbaltotal, glac_wide_runoff, glac_wide_snowline, glac_wide_snowpack,
                 glac_wide_area_annual, glac_wide_volume_annual, glac_wide_ELA_annual, offglac_wide_prec,
                 offglac_wide_refreeze, offglac_wide_melt, offglac_wide_snowpack, offglac_wide_runoff) = (
                    massbalance.runmassbalance(modelparameters_copy, glacier_rgi_table, glacier_area_initial,
                                               icethickness_initial, width_initial, elev_bins, glacier_gcm_temp, 
                                               glacier_gcm_tempstd, glacier_gcm_prec, glacier_gcm_elev, 
                                               glacier_gcm_lrgcm, glacier_gcm_lrglac, dates_table,
                                               option_areaconstant=0))
                # Compute glacier volume change for every time step and use this to compute mass balance
                glac_wide_area = glac_wide_area_annual[:-1].repeat(12)
                # Mass change [km3 mwe]
                #  mb [mwea] * (1 km / 1000 m) * area [km2]
                glac_wide_masschange = glac_wide_massbaltotal / 1000 * glac_wide_area
                # Mean annual mass balance [mwea]
                mb_mwea = glac_wide_masschange[t1_idx:t2_idx+1].sum() / glac_wide_area[0] * 1000 / (t2 - t1)

                return mb_mwea

            # ===== POTENTIAL FUNCTION =====
            # Potential functions are used to impose additional constrains on the model
            @pymc.potential
            def mb_max(mb_max_loss=mb_max_loss, massbal=massbal):
                """Model parameters cannot completely melt the glacier, i.e., reject any parameter set within 0.01 mwea
                   of completely melting the glacier"""
                if massbal < mb_max_loss:
                    return -np.inf
                else:
                    return 0

            @pymc.potential
            def must_melt(tempchange=tempchange, precfactor=precfactor, ddfsnow=ddfsnow):
                """
                Likelihood function for mass balance [mwea] based on model parameters
                """
                modelparameters_copy = modelparameters.copy()
                if tempchange is not None:
                    modelparameters_copy[7] = float(tempchange)
                if precfactor is not None:
                    modelparameters_copy[2] = float(precfactor)
                if ddfsnow is not None:
                    modelparameters_copy[4] = float(ddfsnow)
                    # Degree day factor of ice is proportional to ddfsnow
                    modelparameters_copy[5] = modelparameters_copy[4] / pygem_prms.ddfsnow_iceratio
                # Mass balance calculations
                (glac_bin_temp, glac_bin_prec, glac_bin_acc, glac_bin_refreeze, glac_bin_snowpack, glac_bin_melt,
                 glac_bin_frontalablation, glac_bin_massbalclim, glac_bin_massbalclim_annual, glac_bin_area_annual,
                 glac_bin_icethickness_annual, glac_bin_width_annual, glac_bin_surfacetype_annual,
                 glac_wide_massbaltotal, glac_wide_runoff, glac_wide_snowline, glac_wide_snowpack,
                 glac_wide_area_annual, glac_wide_volume_annual, glac_wide_ELA_annual, offglac_wide_prec,
                 offglac_wide_refreeze, offglac_wide_melt, offglac_wide_snowpack, offglac_wide_runoff) = (
                    massbalance.runmassbalance(modelparameters_copy, glacier_rgi_table, glacier_area_initial,
                                               icethickness_initial, width_initial, elev_bins, glacier_gcm_temp, 
                                               glacier_gcm_tempstd, glacier_gcm_prec, glacier_gcm_elev, 
                                               glacier_gcm_lrgcm, glacier_gcm_lrglac, dates_table,
                                               option_areaconstant=0))
                # Climatic mass balance of lowermost bin must be negative at some point
                glac_idx = np.where(glac_bin_area_annual > 0)[0][0]
                lower_massbalclim_annual = glac_bin_massbalclim_annual[glac_idx,:].tolist()
                # Number of years with negative climatic mass balance
                nyears_negmbclim = np.sum([1 if x < 0 else 0 for x in lower_massbalclim_annual])
                if nyears_negmbclim > 0:
                    return 0
                else:
                    return -np.inf

            # ===== OBSERVED DATA =====
            #  Observed data defines the observed likelihood of mass balances (based on geodetic observations)
            obs_massbal = pymc.Normal('obs_massbal', mu=massbal, tau=(1/(observed_error**2)),
                                      value=float(observed_massbal), observed=True)

            # Set model
            if use_potentials == 1:
                model = pymc.MCMC([{'precfactor':precfactor, 'tempchange':tempchange, 'ddfsnow':ddfsnow,
                                   'massbal':massbal, 'obs_massbal':obs_massbal}, mb_max, must_melt])
            else:
                model = pymc.MCMC({'precfactor':precfactor, 'tempchange':tempchange, 'ddfsnow':ddfsnow,
                                   'massbal':massbal, 'obs_massbal':obs_massbal})
#            if dbname is not None:
#                model = pymc.MCMC({'precfactor':precfactor, 'tempchange':tempchange, 'ddfsnow':ddfsnow,
#                                   'massbal':massbal, 'obs_massbal':obs_massbal}, db='pickle', dbname=dbname)

            # Step method (if changed from default)
            #  Adaptive metropolis is supposed to perform block update, i.e., update all model parameters together based
            #  on their covariance, which would reduce autocorrelation; however, tests show doesn't make a difference.
            if step == 'am':
                model.use_step_method(pymc.AdaptiveMetropolis, [precfactor, tempchange, ddfsnow], delay = 1000)

            # Sample
            if args.progress_bar == 1:
                progress_bar_switch = True
            else:
                progress_bar_switch = False
            model.sample(iter=iterations, burn=burn, thin=thin,
                         tune_interval=tune_interval, tune_throughout=tune_throughout,
                         save_interval=save_interval, verbose=verbose, progress_bar=progress_bar_switch)

            # Close database
            model.db.close()
            return model


        #%%
        # ===== Begin MCMC process =====
        # loop through each glacier selected
        for glac in range(main_glac_rgi.shape[0]):

#            if debug:
            print(count, main_glac_rgi.loc[main_glac_rgi.index.values[glac],'RGIId_float'])

            # Set model parameters
            modelparameters = [pygem_prms.lrgcm, pygem_prms.lrglac, pygem_prms.precfactor, pygem_prms.precgrad, pygem_prms.ddfsnow, pygem_prms.ddfice,
                               pygem_prms.tempsnow, pygem_prms.tempchange]

            # Select subsets of data
            glacier_rgi_table = main_glac_rgi.loc[main_glac_rgi.index.values[glac], :]
            glacier_gcm_elev = gcm_elev[glac]
            glacier_gcm_prec = gcm_prec[glac,:]
            glacier_gcm_temp = gcm_temp[glac,:]
            glacier_gcm_tempstd = gcm_tempstd[glac,:]
            glacier_gcm_lrgcm = gcm_lr[glac,:]
            glacier_gcm_lrglac = glacier_gcm_lrgcm.copy()
            glacier_area_initial = main_glac_hyps.iloc[glac,:].values.astype(float)
            icethickness_initial = main_glac_icethickness.iloc[glac,:].values.astype(float)
            width_initial = main_glac_width.iloc[glac,:].values.astype(float)
            glacier_cal_data = ((cal_data.iloc[np.where(
                    glacier_rgi_table['rgino_str'] == cal_data['glacno'])[0],:]).copy())
            glacier_str = '{0:0.5f}'.format(glacier_rgi_table['RGIId_float'])

            # Select observed mass balance, error, and time data
            cal_idx = glacier_cal_data.index.values[0]
            #  Note: index to main_glac_rgi may differ from cal_idx
            t1 = glacier_cal_data.loc[cal_idx, 't1']
            t2 = glacier_cal_data.loc[cal_idx, 't2']
            t1_idx = int(glacier_cal_data.loc[cal_idx,'t1_idx'])
            t2_idx = int(glacier_cal_data.loc[cal_idx,'t2_idx'])
            # Observed mass balance [mwea]
            observed_massbal = glacier_cal_data.loc[cal_idx,'mb_mwe'] / (t2 - t1)
            observed_error = glacier_cal_data.loc[cal_idx,'mb_mwe_err'] / (t2 - t1)

            if debug:
                print('observed_massbal:', np.round(observed_massbal,2), 'observed_error:',np.round(observed_error,2))

            # ===== RUN MARKOV CHAIN MONTE CARLO METHOD ====================
            if icethickness_initial.max() > 0:

                # Regional priors
                precfactor_gamma_alpha = pygem_prms.precfactor_gamma_region_dict[glacier_rgi_table.loc['region']][0]
                precfactor_gamma_beta = pygem_prms.precfactor_gamma_region_dict[glacier_rgi_table.loc['region']][1]
                tempchange_mu = pygem_prms.tempchange_norm_region_dict[glacier_rgi_table.loc['region']][0]
                tempchange_sigma = pygem_prms.tempchange_norm_region_dict[glacier_rgi_table.loc['region']][1]

                # fit the MCMC model
                for n_chain in range(0,pygem_prms.n_chains):

                    if debug:
                        print('\n', glacier_str, ' chain' + str(n_chain))

                    if n_chain == 0:
                        # Starting values: middle
                        tempchange_start = tempchange_mu
                        precfactor_start = precfactor_gamma_alpha / precfactor_gamma_beta
                        ddfsnow_start = pygem_prms.ddfsnow_mu

                    elif n_chain == 1:
                        # Starting values: lowest
                        tempchange_start = tempchange_mu - 1.96 * tempchange_sigma
                        ddfsnow_start = pygem_prms.ddfsnow_mu - 1.96 * pygem_prms.ddfsnow_sigma
                        precfactor_start = stats.gamma.ppf(0.05,precfactor_gamma_alpha, scale=1/precfactor_gamma_beta)

                    elif n_chain == 2:
                        # Starting values: high
                        tempchange_start = tempchange_mu + 1.96 * tempchange_sigma
                        ddfsnow_start = pygem_prms.ddfsnow_mu + 1.96 * pygem_prms.ddfsnow_sigma
                        precfactor_start = stats.gamma.ppf(0.95,precfactor_gamma_alpha, scale=1/precfactor_gamma_beta)


                    # Determine bounds to check TC starting values and estimate maximum mass loss
                    modelparameters[2] = precfactor_start
                    modelparameters[4] = ddfsnow_start
                    modelparameters[5] = ddfsnow_start / pygem_prms.ddfsnow_iceratio

                    tempchange_boundlow, tempchange_boundhigh, mb_max_loss = (
                            retrieve_priors(modelparameters, glacier_rgi_table, glacier_area_initial, icethickness_initial,
                                            width_initial, elev_bins, glacier_gcm_temp, glacier_gcm_tempstd, 
                                            glacier_gcm_prec, glacier_gcm_elev, glacier_gcm_lrgcm, glacier_gcm_lrglac, 
                                            dates_table, t1_idx, t2_idx, t1, t2, debug=False))

                    if debug:
                        print('\nTC_low:', np.round(tempchange_boundlow,2),
                              'TC_high:', np.round(tempchange_boundhigh,2),
                              'mb_max_loss:', np.round(mb_max_loss,2))

                    # Check that tempchange mu and sigma is somewhat within bndhigh and bndlow
                    if tempchange_start > tempchange_boundhigh:
                        tempchange_start = tempchange_boundhigh
                    elif tempchange_start < tempchange_boundlow:
                        tempchange_start = tempchange_boundlow

#                    # Check that tempchange mu and sigma is somewhat within bndhigh and bndlow
#                    if ((tempchange_boundhigh < tempchange_mu - 3 * tempchange_sigma) or
#                        (tempchange_boundlow > tempchange_mu + 3 * tempchange_sigma)):
#                        tempchange_mu = np.mean([tempchange_boundlow, tempchange_boundhigh])
#                        tempchange_sigma = (tempchange_boundhigh - tempchange_boundlow) / 6

                    if debug:
                        print('\ntc_start:', np.round(tempchange_start,3),
                              '\npf_start:', np.round(precfactor_start,3),
                              '\nddf_start:', np.round(ddfsnow_start,4))

                    model = run_MCMC(iterations=pygem_prms.mcmc_sample_no, burn=pygem_prms.mcmc_burn_no, step=pygem_prms.mcmc_step,
                                     precfactor_gamma_alpha=precfactor_gamma_alpha,
                                     precfactor_gamma_beta=precfactor_gamma_beta,
                                     precfactor_start=precfactor_start,
                                     tempchange_mu=tempchange_mu, tempchange_sigma=tempchange_sigma,
                                     tempchange_start=tempchange_start,
                                     ddfsnow_start=ddfsnow_start,
                                     mb_max_loss=mb_max_loss)

                    if debug:
                        print('\nacceptance ratio:', model.step_method_dict[next(iter(model.stochastics))][0].ratio)

                    # Select data from model to be stored in netcdf
                    df = pd.DataFrame({'tempchange': model.trace('tempchange')[:],
                                       'precfactor': model.trace('precfactor')[:],
                                       'ddfsnow': model.trace('ddfsnow')[:],
                                       'massbal': model.trace('massbal')[:]})
                    # set columns for other variables
                    df['ddfice'] = df['ddfsnow'] / pygem_prms.ddfsnow_iceratio
                    df['lrgcm'] = np.full(df.shape[0], pygem_prms.lrgcm)
                    df['lrglac'] = np.full(df.shape[0], pygem_prms.lrglac)
                    df['precgrad'] = np.full(df.shape[0], pygem_prms.precgrad)
                    df['tempsnow'] = np.full(df.shape[0], pygem_prms.tempsnow)

                    if debug:
                        print('mb_mwea:', np.round(df.massbal.mean(),2), 'mb_mwea_std:', np.round(df.massbal.std(),2))

                    if n_chain == 0:
                        df_chains = df.values[:, :, np.newaxis]
                    else:
                        df_chains = np.dstack((df_chains, df.values))

                ds = xr.Dataset({'mp_value': (('iter', 'mp', 'chain'), df_chains),
                                 },
                                coords={'iter': df.index.values,
                                        'mp': df.columns.values,
                                        'chain': np.arange(0,n_chain+1),
                                        })

                if not os.path.exists(pygem_prms.output_fp_cal):
                    os.makedirs(pygem_prms.output_fp_cal)
                ds.to_netcdf(pygem_prms.output_fp_cal + glacier_str + '.nc')
                ds.close()

    #            #%%
    #            # Example of accessing netcdf file and putting it back into pandas dataframe
    #            ds = xr.open_dataset(pygem_prms.output_fp_cal + '13.00014.nc')
    #            df = pd.DataFrame(ds['mp_value'].sel(chain=0).values, columns=ds.mp.values)
    #            priors = pd.Series(ds.priors, index=ds.prior_cns)
    #            #%%

        # ==============================================================

    #%%
    # Huss and Hock (2015) model calibration steps
    elif pygem_prms.option_calibration == 3:

        def objective(modelparameters_subset):
            """
            Objective function for mass balance data.

            Parameters
            ----------
            modelparameters_subset : np.float64
                List of model parameters to calibrate
                [precipitation factor, precipitation gradient, degree-day factor of snow, temperature bias]

            Returns
            -------
            mb_dif_mwea
                Returns the difference in modeled vs observed mass balance [mwea]
            """
            # Use a subset of model parameters to reduce number of constraints required
            modelparameters[2] = modelparameters_subset[0]
            modelparameters[3] = modelparameters_subset[1]
            modelparameters[4] = modelparameters_subset[2]
            modelparameters[5] = modelparameters[4] / ddfsnow_iceratio
            modelparameters[7] = modelparameters_subset[3]
            # Mass balance calculations
            (glac_bin_temp, glac_bin_prec, glac_bin_acc, glac_bin_refreeze, glac_bin_snowpack, glac_bin_melt,
             glac_bin_frontalablation, glac_bin_massbalclim, glac_bin_massbalclim_annual, glac_bin_area_annual,
             glac_bin_icethickness_annual, glac_bin_width_annual, glac_bin_surfacetype_annual,
             glac_wide_massbaltotal, glac_wide_runoff, glac_wide_snowline, glac_wide_snowpack,
             glac_wide_area_annual, glac_wide_volume_annual, glac_wide_ELA_annual, offglac_wide_prec,
             offglac_wide_refreeze, offglac_wide_melt, offglac_wide_snowpack, offglac_wide_runoff) = (
                massbalance.runmassbalance(modelparameters, glacier_rgi_table, glacier_area_initial, 
                                           icethickness_initial, width_initial, elev_bins, glacier_gcm_temp, 
                                           glacier_gcm_tempstd, glacier_gcm_prec, glacier_gcm_elev, glacier_gcm_lrgcm, 
                                           glacier_gcm_lrglac, dates_table, option_areaconstant=1))
            # Use a subset of model parameters to reduce number of constraints required
            modelparameters[2] = modelparameters_subset[0]
            modelparameters[3] = modelparameters_subset[1]
            modelparameters[4] = modelparameters_subset[2]
            modelparameters[5] = modelparameters[4] / ddfsnow_iceratio
            modelparameters[7] = modelparameters_subset[3]
            # Mass balance calculations
            (glac_bin_temp, glac_bin_prec, glac_bin_acc, glac_bin_refreeze, glac_bin_snowpack, glac_bin_melt,
             glac_bin_frontalablation, glac_bin_massbalclim, glac_bin_massbalclim_annual, glac_bin_area_annual,
             glac_bin_icethickness_annual, glac_bin_width_annual, glac_bin_surfacetype_annual,
             glac_wide_massbaltotal, glac_wide_runoff, glac_wide_snowline, glac_wide_snowpack,
             glac_wide_area_annual, glac_wide_volume_annual, glac_wide_ELA_annual, offglac_wide_prec,
             offglac_wide_refreeze, offglac_wide_melt, offglac_wide_snowpack, offglac_wide_runoff) = (
                massbalance.runmassbalance(modelparameters, glacier_rgi_table, glacier_area_initial, 
                                           icethickness_initial, width_initial, elev_bins, glacier_gcm_temp, 
                                           glacier_gcm_tempstd, glacier_gcm_prec, glacier_gcm_elev, glacier_gcm_lrgcm, 
                                           glacier_gcm_lrglac, dates_table, option_areaconstant=1))
            # Compute glacier volume change for every time step and use this to compute mass balance
            glac_wide_area = glac_wide_area_annual[:-1].repeat(12)
            # Mass change [km3 mwe]
            #  mb [mwea] * (1 km / 1000 m) * area [km2]
            glac_wide_masschange = glac_wide_massbaltotal[t1_idx:t2_idx+1] / 1000 * glac_wide_area[t1_idx:t2_idx+1]
            # Mean annual mass balance [mwea]
            mb_mwea = (glac_wide_masschange.sum() / glac_wide_area[0] * 1000 /
                       (glac_wide_masschange.shape[0] / 12))

            # Differnece [mwea] = Observed mass balance [mwea] - mb_mwea
            mb_dif_mwea_abs = abs(observed_massbal - mb_mwea)

#            print('Obs[mwea]:', np.round(observed_massbal,2), 'Model[mwea]:', np.round(mb_mwea,2))
#            print('Dif[mwea]:', np.round(mb_dif_mwea,2))

            return mb_dif_mwea_abs


        def run_objective(modelparameters_init, observed_massbal, precfactor_bnds=(0.33,3), tempchange_bnds=(-10,10),
                          ddfsnow_bnds=(0.0026,0.0056), precgrad_bnds=(0.0001,0.0001), run_opt=True):
            """
            Run the optimization for the single glacier objective function.

            Parameters
            ----------
            modelparams_init : list
                List of model parameters to calibrate
                [precipitation factor, precipitation gradient, degree day factor of snow, temperature change]
            glacier_cal_data : pd.DataFrame
                Table containing calibration data for a single glacier
            precfactor_bnds : tuple
                Lower and upper bounds for precipitation factor (default is (0.33, 3))
            tempchange_bnds : tuple
                Lower and upper bounds for temperature bias (default is (0.33, 3))
            ddfsnow_bnds : tuple
                Lower and upper bounds for degree day factor of snow (default is (0.0026, 0.0056))
            precgrad_bnds : tuple
                Lower and upper bounds for precipitation gradient (default is constant (0.0001,0.0001))
            run_opt : boolean
                Boolean statement allowing one to bypass the optimization and run through with initial parameters
                (default is True - run the optimization)

            Returns
            -------
            modelparameters_opt : optimize.optimize.OptimizeResult
                Returns result of scipy optimization, which includes optimized parameters and other information
            glacier_cal_compare : pd.DataFrame
                Table recapping calibration results: observation, model, calibration round, etc.
            """
            # Bounds
            modelparameters_bnds = (precfactor_bnds, precgrad_bnds, ddfsnow_bnds, tempchange_bnds)
            # Run the optimization
            #  'L-BFGS-B' - much slower
            #  'SLSQP' did not work for some geodetic measurements using the sum_abs_zscore.  One work around was to
            #    divide the sum_abs_zscore by 1000, which made it work in all cases.  However, methods were switched
            #    to 'L-BFGS-B', which may be slower, but is still effective.
            # note: switch enables running through with given parameters
            if run_opt:
                modelparameters_opt = minimize(objective, modelparameters_init, method=pygem_prms.method_opt,
                                               bounds=modelparameters_bnds, options={'ftol':pygem_prms.ftol_opt})
                # Record the optimized parameters
                modelparameters_subset = modelparameters_opt.x
            else:
                modelparameters_subset = modelparameters_init.copy()
            modelparams = (
                    [modelparameters[0], modelparameters[1], modelparameters_subset[0], modelparameters_subset[1],
                     modelparameters_subset[2], modelparameters_subset[2] / ddfsnow_iceratio, modelparameters[6],
                     modelparameters_subset[3]])
            # Re-run the optimized parameters in order to see the mass balance
            # Mass balance calculations
            (glac_bin_temp, glac_bin_prec, glac_bin_acc, glac_bin_refreeze, glac_bin_snowpack, glac_bin_melt,
             glac_bin_frontalablation, glac_bin_massbalclim, glac_bin_massbalclim_annual, glac_bin_area_annual,
             glac_bin_icethickness_annual, glac_bin_width_annual, glac_bin_surfacetype_annual,
             glac_wide_massbaltotal, glac_wide_runoff, glac_wide_snowline, glac_wide_snowpack,
             glac_wide_area_annual, glac_wide_volume_annual, glac_wide_ELA_annual, offglac_wide_prec,
             offglac_wide_refreeze, offglac_wide_melt, offglac_wide_snowpack, offglac_wide_runoff) = (
                massbalance.runmassbalance(modelparams, glacier_rgi_table, glacier_area_initial, icethickness_initial,
                                           width_initial, elev_bins, glacier_gcm_temp, glacier_gcm_tempstd, glacier_gcm_prec,
                                           glacier_gcm_elev, glacier_gcm_lrgcm, glacier_gcm_lrglac, dates_table,
                                           option_areaconstant=1))
            # Compute glacier volume change for every time step and use this to compute mass balance
            glac_wide_area = glac_wide_area_annual[:-1].repeat(12)
            # Mass change [km3 mwe]
            #  mb [mwea] * (1 km / 1000 m) * area [km2]
            glac_wide_masschange = glac_wide_massbaltotal[t1_idx:t2_idx+1] / 1000 * glac_wide_area[t1_idx:t2_idx+1]
            # Mean annual mass balance [mwea]
            mb_mwea = (glac_wide_masschange.sum() / glac_wide_area[0] * 1000 /
                       (glac_wide_masschange.shape[0] / 12))

            # Differnece [mwea] = Observed mass balance [mwea] - mb_mwea
#            mb_dif_mwea = observed_massbal - mb_mwea
#
#            print('Obs[mwea]:', np.round(observed_massbal,2), 'Model[mwea]:', np.round(mb_mwea,2))
#            print('Dif[mwea]:', np.round(mb_dif_mwea,2))

            return modelparams, mb_mwea


        def write_netcdf_modelparams(output_fullfn, modelparameters, mb_mwea, observed_massbal):
            """
            Export glacier model parameters and modeled observations to netcdf file.

            Parameters
            ----------
            output_fullfn : str
                Full filename (path included) of the netcdf to be exported
            modelparams : list
                model parameters
            mb_mwea : float
                modeled mass balance for given parameters

            Returns
            -------
            None
                Exports file to netcdf
            """
            # Select data from model to be stored in netcdf
            df = pd.DataFrame(index=[0])
            df['lrgcm'] = np.full(df.shape[0], pygem_prms.lrgcm)
            df['lrglac'] = np.full(df.shape[0], pygem_prms.lrglac)
            df['precfactor'] = modelparameters[2]
            df['precgrad'] = np.full(df.shape[0], pygem_prms.precgrad)
            df['ddfsnow'] = modelparameters[4]
            df['ddfice'] = df['ddfsnow'] / ddfsnow_iceratio
            df['tempsnow'] = np.full(df.shape[0], pygem_prms.tempsnow)
            df['tempchange'] = modelparameters[7]
            df['mb_mwea'] = mb_mwea
            df['obs_mwea'] = observed_massbal
            df['dif_mwea'] = mb_mwea - observed_massbal
            df_export = df.values[:, :, np.newaxis]
            # Set up dataset and export to netcdf
            ds = xr.Dataset({'mp_value': (('iter', 'mp', 'chain'), df_export)},
                            coords={'iter': df.index.values,
                                    'mp': df.columns.values,
                                    'chain': [0]})
            ds.to_netcdf(output_fullfn)
            ds.close()

        # Huss and Hock (2015) parameters and bounds
        tempchange_init = 0
        tempchange_bndlow = -10
        tempchange_bndhigh = 10
        precfactor_init = 1.5
        precfactor_bndlow = 0.8
        precfactor_bndhigh = 2
        ddfsnow_init = 0.003
        ddfsnow_bndlow = 0.00175
        ddfsnow_bndhigh = 0.0045
        ddfsnow_iceratio = 0.5

        # ===== Begin processing =====
        # loop through each glacier selected
        for glac in range(main_glac_rgi.shape[0]):

            if debug:
                print(count, main_glac_rgi.loc[main_glac_rgi.index.values[glac],'RGIId_float'])
            elif glac%500 == 0:
                print(count, main_glac_rgi.loc[main_glac_rgi.index.values[glac],'RGIId_float'])

            # Set model parameters
            modelparameters = [pygem_prms.lrgcm, pygem_prms.lrglac, precfactor_init, pygem_prms.precgrad, ddfsnow_init, pygem_prms.ddfice,
                               pygem_prms.tempsnow, tempchange_init]
            modelparameters[5] = modelparameters[4] / ddfsnow_iceratio

            # Select subsets of data
            glacier_rgi_table = main_glac_rgi.loc[main_glac_rgi.index.values[glac], :]
            glacier_gcm_elev = gcm_elev[glac]
            glacier_gcm_prec = gcm_prec[glac,:]
            glacier_gcm_temp = gcm_temp[glac,:]
            glacier_gcm_tempstd = gcm_tempstd[glac,:]
            glacier_gcm_lrgcm = gcm_lr[glac,:]
            glacier_gcm_lrglac = glacier_gcm_lrgcm.copy()
            glacier_area_initial = main_glac_hyps.iloc[glac,:].values.astype(float)
            icethickness_initial = main_glac_icethickness.iloc[glac,:].values.astype(float)
            width_initial = main_glac_width.iloc[glac,:].values.astype(float)
            glacier_cal_data = ((cal_data.iloc[np.where(
                    glacier_rgi_table['rgino_str'] == cal_data['glacno'])[0],:]).copy())
            glacier_str = '{0:0.5f}'.format(glacier_rgi_table['RGIId_float'])

            # Select observed mass balance, error, and time data
            cal_idx = glacier_cal_data.index.values[0]
            #  Note: index to main_glac_rgi may differ from cal_idx
            t1 = glacier_cal_data.loc[cal_idx, 't1']
            t2 = glacier_cal_data.loc[cal_idx, 't2']
            t1_idx = int(glacier_cal_data.loc[cal_idx,'t1_idx'])
            t2_idx = int(glacier_cal_data.loc[cal_idx,'t2_idx'])
            # Observed mass balance [mwea]
            observed_massbal = glacier_cal_data.loc[cal_idx,'mb_mwe'] / (t2 - t1)

            if debug:
                print('obs_mwea:', np.round(observed_massbal,2))

            # Round 1: optimize precipitation factor
            if debug:
                print('Round 1:')
            modelparameters_subset = [modelparameters[2], modelparameters[3], modelparameters[4], modelparameters[7]]
            precfactor_bnds = (precfactor_bndlow, precfactor_bndhigh)
            ddfsnow_bnds = (ddfsnow_init, ddfsnow_init)
            tempchange_bnds = (tempchange_init, tempchange_init)
            modelparams, mb_mwea = run_objective(modelparameters_subset, observed_massbal,
                                                 precfactor_bnds=precfactor_bnds, tempchange_bnds=tempchange_bnds,
                                                 ddfsnow_bnds=ddfsnow_bnds)
            precfactor_opt = modelparams[2]
            if debug:
                print('mb_mwea:', np.round(mb_mwea,2), 'precfactor:', np.round(precfactor_opt,2))

            # Round 2: optimize DDFsnow
            if debug:
                print('Round 2:')
            modelparameters_subset = [precfactor_opt, modelparameters[3], modelparameters[4], modelparameters[7]]
            precfactor_bnds = (precfactor_opt, precfactor_opt)
            ddfsnow_bnds = (ddfsnow_bndlow, ddfsnow_bndhigh)
            tempchange_bnds = (tempchange_init, tempchange_init)
            modelparams, mb_mwea = run_objective(modelparameters_subset, observed_massbal,
                                                 precfactor_bnds=precfactor_bnds, tempchange_bnds=tempchange_bnds,
                                                 ddfsnow_bnds=ddfsnow_bnds)

            ddfsnow_opt = modelparams[4]
            if debug:
                print('mb_mwea:', np.round(mb_mwea,2), 'precfactor:', np.round(precfactor_opt,2),
                      'ddfsnow:', np.round(ddfsnow_opt,5))

            # Round 3: optimize tempbias
            if debug:
                print('Round 3:')

            # ----- TEMPBIAS: max accumulation -----
            # Lower temperature bound based on no positive temperatures
            # Temperature at the lowest bin
            #  T_bin = T_gcm + lr_gcm * (z_ref - z_gcm) + lr_glac * (z_bin - z_ref) + tempchange
            lowest_bin = np.where(glacier_area_initial > 0)[0][0]
            tempchange_max_acc = (-1 * (glacier_gcm_temp + glacier_gcm_lrgcm *
                                        (elev_bins[lowest_bin] - glacier_gcm_elev)).max())
            tempchange_bndlow = tempchange_max_acc

            if debug:
                print('tempchange_bndlow:', np.round(tempchange_bndlow,2))

            dif_mb_mwea = abs(observed_massbal - mb_mwea)
            if debug:
                print('dif:', np.round(dif_mb_mwea,2))
            count = 0
            while dif_mb_mwea > 0.1 and count < 20:
                if count > 0:
                    if mb_mwea - observed_massbal > 0:
                        modelparameters[7] += 1
                    else:
                        modelparameters[7] -= 1
                    # Temperature cannot exceed lower bound
                    if modelparameters[7] < tempchange_bndlow:
                        modelparameters[7] = tempchange_bndlow

                modelparameters_subset = [precfactor_opt, modelparameters[3], ddfsnow_opt, modelparameters[7]]
                precfactor_bnds = (precfactor_opt, precfactor_opt)
                ddfsnow_bnds = (ddfsnow_opt, ddfsnow_opt)
                tempchange_bnds = (tempchange_bndlow, tempchange_bndhigh)
                modelparams, mb_mwea = run_objective(modelparameters_subset, observed_massbal,
                                                     precfactor_bnds=precfactor_bnds, tempchange_bnds=tempchange_bnds,
                                                     ddfsnow_bnds=ddfsnow_bnds)
                dif_mb_mwea = abs(observed_massbal - mb_mwea)

                count += 1
                if debug:
                    print('dif:', np.round(dif_mb_mwea,2), 'count:', count, 'tc:', np.round(modelparameters[7],2))

                # Break loop if at lower bound
                if abs(tempchange_bndlow - modelparams[7]) < 0.1:
                    count=20

            # Record optimal temperature bias
            tempchange_opt = modelparams[7]

            if debug:
                print('mb_mwea:', np.round(mb_mwea,2), 'precfactor:', np.round(precfactor_opt,2),
                      'ddfsnow:', np.round(ddfsnow_opt,5), 'tempchange:', np.round(tempchange_opt,2))


            # EXPORT TO NETCDF
            netcdf_output_fp = (pygem_prms.output_fp_cal)
            if not os.path.exists(netcdf_output_fp):
                os.makedirs(netcdf_output_fp)
            write_netcdf_modelparams(netcdf_output_fp + glacier_str + '.nc', modelparameters, mb_mwea, observed_massbal)
        # ==============================================================
    #%%
    # MODIFIED Huss and Hock (2015) model calibration steps
    # - glacier able to evolve
    # - precipitaiton factor, then temperature bias (no ddfsnow)
    # - ranges different
    elif pygem_prms.option_calibration == 4:
        
        if pygem_prms.params2opt.sort() == ['tempbias', 'precfactor'].sort():
            def objective(modelparameters_subset):
                """
                Objective function for mass balance data.
    
                Parameters
                ----------
                modelparameters_subset : np.float64
                    List of model parameters to calibrate
                    [precipitation factor, precipitation gradient, degree-day factor of snow, temperature bias]
    
                Returns
                -------
                mb_dif_mwea
                    Returns the difference in modeled vs observed mass balance [mwea]
                """
                # Use a subset of model parameters to reduce number of constraints required
                modelparameters[2] = modelparameters_subset[0]
                modelparameters[7] = modelparameters_subset[1]
                # Mass balance calculation
                mb_mwea = mb_mwea_calc(
                        modelparameters, glacier_rgi_table, glacier_area_initial, icethickness_initial, width_initial, 
                        elev_bins, glacier_gcm_temp, glacier_gcm_tempstd, glacier_gcm_prec, glacier_gcm_elev, 
                        glacier_gcm_lrgcm, glacier_gcm_lrglac, dates_table, t1_idx, t2_idx, t1, t2, option_areaconstant=0)
                print('model params:', modelparameters[2], modelparameters[7], 
                      '\n  mb_mwea:', mb_mwea)
                # Difference [mwea]
                mb_dif_mwea_abs = abs(observed_massbal - mb_mwea)
                return mb_dif_mwea_abs
    
            def run_objective(modelparameters_init, observed_massbal, precfactor_bnds=(0.33,3), tempchange_bnds=(-10,10),
                              run_opt=True, eps_opt=pygem_prms.eps_opt, ftol_opt=pygem_prms.ftol_opt):
                """
                Run the optimization for the single glacier objective function.
    
                Parameters
                ----------
                modelparams_init : list
                    List of model parameters to calibrate
                    [precipitation factor, precipitation gradient, degree day factor of snow, temperature change]
                glacier_cal_data : pd.DataFrame
                    Table containing calibration data for a single glacier
                precfactor_bnds : tuple
                    Lower and upper bounds for precipitation factor (default is (0.33, 3))
                tempchange_bnds : tuple
                    Lower and upper bounds for temperature bias (default is (0.33, 3))
                ddfsnow_bnds : tuple
                    Lower and upper bounds for degree day factor of snow (default is (0.0026, 0.0056))
                precgrad_bnds : tuple
                    Lower and upper bounds for precipitation gradient (default is constant (0.0001,0.0001))
                run_opt : boolean
                    Boolean statement allowing one to bypass the optimization and run through with initial parameters
                    (default is True - run the optimization)
    
                Returns
                -------
                modelparameters_opt : optimize.optimize.OptimizeResult
                    Returns result of scipy optimization, which includes optimized parameters and other information
                glacier_cal_compare : pd.DataFrame
                    Table recapping calibration results: observation, model, calibration round, etc.
                """
                # Bounds
                modelparameters_bnds = (precfactor_bnds, tempchange_bnds)
                # Run the optimization
                #  'L-BFGS-B' - much slower
                #  'SLSQP' did not work for some geodetic measurements using the sum_abs_zscore.  One work around was to
                #    divide the sum_abs_zscore by 1000, which made it work in all cases.  However, methods were switched
                #    to 'L-BFGS-B', which may be slower, but is still effective.
                # note: switch enables running through with given parameters
                if run_opt:
                    modelparameters_opt = minimize(objective, modelparameters_init, method=pygem_prms.method_opt,
                                                   bounds=modelparameters_bnds, 
                                                   options={'ftol':ftol_opt, 'eps':eps_opt})
                    # Record the optimized parameters
                    modelparameters_subset = modelparameters_opt.x
                else:
                    modelparameters_subset = modelparameters_init.copy()
                modelparams = (
                        [modelparameters[0], modelparameters[1], modelparameters_subset[0], modelparameters[3],
                         modelparameters[4], modelparameters[4] / ddfsnow_iceratio, modelparameters[6],
                         modelparameters_subset[1]])
                # Re-run the optimized parameters in order to see the mass balance
                mb_mwea = mb_mwea_calc(
                        modelparameters, glacier_rgi_table, glacier_area_initial, icethickness_initial, width_initial, 
                        elev_bins, glacier_gcm_temp, glacier_gcm_tempstd, glacier_gcm_prec, glacier_gcm_elev, 
                        glacier_gcm_lrgcm, glacier_gcm_lrglac, dates_table, t1_idx, t2_idx, t1, t2, option_areaconstant=0)
                return modelparams, mb_mwea
        else:
            def objective(modelparameters_subset):
                """
                Objective function for mass balance data.
    
                Parameters
                ----------
                modelparameters_subset : np.float64
                    List of model parameters to calibrate
                    [precipitation factor, precipitation gradient, degree-day factor of snow, temperature bias]
    
                Returns
                -------
                mb_dif_mwea
                    Returns the difference in modeled vs observed mass balance [mwea]
                """
                # Use a subset of model parameters to reduce number of constraints required
                modelparameters[2] = modelparameters_subset[0]
                modelparameters[3] = modelparameters_subset[1]
                modelparameters[4] = modelparameters_subset[2]
                modelparameters[5] = modelparameters[4] / ddfsnow_iceratio
                modelparameters[7] = modelparameters_subset[3]
                # Mass balance calculation
                mb_mwea = mb_mwea_calc(
                        modelparameters, glacier_rgi_table, glacier_area_initial, icethickness_initial, width_initial, 
                        elev_bins, glacier_gcm_temp, glacier_gcm_tempstd, glacier_gcm_prec, glacier_gcm_elev, 
                        glacier_gcm_lrgcm, glacier_gcm_lrglac, dates_table, t1_idx, t2_idx, t1, t2, option_areaconstant=0)
                print('model params:', modelparameters[2], modelparameters[7], 
                      '\n  mb_mwea:', mb_mwea)
                # Difference [mwea]
                mb_dif_mwea_abs = abs(observed_massbal - mb_mwea)
                return mb_dif_mwea_abs
    
            def run_objective(modelparameters_init, observed_massbal, precfactor_bnds=(0.33,3), tempchange_bnds=(-10,10),
                              ddfsnow_bnds=(0.0026,0.0056), precgrad_bnds=(0.0001,0.0001), run_opt=True, 
                              eps_opt=pygem_prms.eps_opt):
                """
                Run the optimization for the single glacier objective function.
    
                Parameters
                ----------
                modelparams_init : list
                    List of model parameters to calibrate
                    [precipitation factor, precipitation gradient, degree day factor of snow, temperature change]
                glacier_cal_data : pd.DataFrame
                    Table containing calibration data for a single glacier
                precfactor_bnds : tuple
                    Lower and upper bounds for precipitation factor (default is (0.33, 3))
                tempchange_bnds : tuple
                    Lower and upper bounds for temperature bias (default is (0.33, 3))
                ddfsnow_bnds : tuple
                    Lower and upper bounds for degree day factor of snow (default is (0.0026, 0.0056))
                precgrad_bnds : tuple
                    Lower and upper bounds for precipitation gradient (default is constant (0.0001,0.0001))
                run_opt : boolean
                    Boolean statement allowing one to bypass the optimization and run through with initial parameters
                    (default is True - run the optimization)
    
                Returns
                -------
                modelparameters_opt : optimize.optimize.OptimizeResult
                    Returns result of scipy optimization, which includes optimized parameters and other information
                glacier_cal_compare : pd.DataFrame
                    Table recapping calibration results: observation, model, calibration round, etc.
                """
                # Bounds
                modelparameters_bnds = (precfactor_bnds, precgrad_bnds, ddfsnow_bnds, tempchange_bnds)
                # Run the optimization
                #  'L-BFGS-B' - much slower
                #  'SLSQP' did not work for some geodetic measurements using the sum_abs_zscore.  One work around was to
                #    divide the sum_abs_zscore by 1000, which made it work in all cases.  However, methods were switched
                #    to 'L-BFGS-B', which may be slower, but is still effective.
                # note: switch enables running through with given parameters
                if run_opt:
                    modelparameters_opt = minimize(objective, modelparameters_init, method=pygem_prms.method_opt,
                                                   bounds=modelparameters_bnds, 
                                                   options={'ftol':pygem_prms.ftol_opt, 'eps':eps_opt})
                    # Record the optimized parameters
                    modelparameters_subset = modelparameters_opt.x
                else:
                    modelparameters_subset = modelparameters_init.copy()
                modelparams = (
                        [modelparameters[0], modelparameters[1], modelparameters_subset[0], modelparameters_subset[1],
                         modelparameters_subset[2], modelparameters_subset[2] / ddfsnow_iceratio, modelparameters[6],
                         modelparameters_subset[3]])
                # Re-run the optimized parameters in order to see the mass balance
                mb_mwea = mb_mwea_calc(
                        modelparameters, glacier_rgi_table, glacier_area_initial, icethickness_initial, width_initial, 
                        elev_bins, glacier_gcm_temp, glacier_gcm_tempstd, glacier_gcm_prec, glacier_gcm_elev, 
                        glacier_gcm_lrgcm, glacier_gcm_lrglac, dates_table, t1_idx, t2_idx, t1, t2, option_areaconstant=0)
                return modelparams, mb_mwea

        def write_netcdf_modelparams(output_fullfn, modelparameters, mb_mwea, observed_massbal):
            """
            Export glacier model parameters and modeled observations to netcdf file.

            Parameters
            ----------
            output_fullfn : str
                Full filename (path included) of the netcdf to be exported
            modelparams : list
                model parameters
            mb_mwea : float
                modeled mass balance for given parameters

            Returns
            -------
            None
                Exports file to netcdf
            """
            # Select data from model to be stored in netcdf
            df = pd.DataFrame(index=[0])
            df['lrgcm'] = np.full(df.shape[0], pygem_prms.lrgcm)
            df['lrglac'] = np.full(df.shape[0], pygem_prms.lrglac)
            df['precfactor'] = modelparameters[2]
            df['precgrad'] = np.full(df.shape[0], pygem_prms.precgrad)
            df['ddfsnow'] = modelparameters[4]
            df['ddfice'] = df['ddfsnow'] / ddfsnow_iceratio
            df['tempsnow'] = np.full(df.shape[0], pygem_prms.tempsnow)
            df['tempchange'] = modelparameters[7]
            df['mb_mwea'] = mb_mwea
            df['obs_mwea'] = observed_massbal
            df['dif_mwea'] = mb_mwea - observed_massbal
            df_export = df.values[:, :, np.newaxis]
            # Set up dataset and export to netcdf
            ds = xr.Dataset({'mp_value': (('iter', 'mp', 'chain'), df_export)},
                            coords={'iter': df.index.values,
                                    'mp': df.columns.values,
                                    'chain': [0]})
            ds.to_netcdf(output_fullfn)
            ds.close()

        # ===== Begin processing =====
        # loop through each glacier selected
        for glac in range(main_glac_rgi.shape[0]):

            # ===== BOUNDS ======
            tempchange_init = 0
            precfactor_init = 1
            precfactor_boundlow = 0.5
            precfactor_boundhigh = 5
            ddfsnow_init = 0.0041
            ddfsnow_iceratio = 0.7

            if debug:
                print(main_glac_rgi.loc[main_glac_rgi.index.values[glac],'RGIId_float'])

            # Set model parameters
            modelparameters = [pygem_prms.lrgcm, pygem_prms.lrglac, precfactor_init, pygem_prms.precgrad, ddfsnow_init, pygem_prms.ddfice,
                               pygem_prms.tempsnow, tempchange_init]
            modelparameters[5] = modelparameters[4] / ddfsnow_iceratio

            # Select subsets of data
            glacier_rgi_table = main_glac_rgi.loc[main_glac_rgi.index.values[glac], :]
            glacier_gcm_elev = gcm_elev[glac]
            glacier_gcm_prec = gcm_prec[glac,:]
            glacier_gcm_temp = gcm_temp[glac,:]
            glacier_gcm_tempstd = gcm_tempstd[glac,:]
            glacier_gcm_lrgcm = gcm_lr[glac,:]
            glacier_gcm_lrglac = glacier_gcm_lrgcm.copy()
            glacier_area_initial = main_glac_hyps.iloc[glac,:].values.astype(float)
            icethickness_initial = main_glac_icethickness.iloc[glac,:].values.astype(float)
            width_initial = main_glac_width.iloc[glac,:].values.astype(float)
            glacier_cal_data = ((cal_data.iloc[np.where(
                    glacier_rgi_table['rgino_str'] == cal_data['glacno'])[0],:]).copy())
            glacier_str = '{0:0.5f}'.format(glacier_rgi_table['RGIId_float'])

            # Select observed mass balance, error, and time data
            cal_idx = glacier_cal_data.index.values[0]
            #  Note: index to main_glac_rgi may differ from cal_idx
            t1 = glacier_cal_data.loc[cal_idx, 't1']
            t2 = glacier_cal_data.loc[cal_idx, 't2']
            t1_idx = int(glacier_cal_data.loc[cal_idx,'t1_idx'])
            t2_idx = int(glacier_cal_data.loc[cal_idx,'t2_idx'])
            # Observed mass balance [mwea]
            observed_massbal = glacier_cal_data.loc[cal_idx,'mb_mwe'] / (t2 - t1)
#            observed_massbal_err = glacier_cal_data.loc[cal_idx,'mb_mwe_err'] / (t2 - t1)

            if debug:
                print('obs_mwea:', np.round(observed_massbal,2))

            if icethickness_initial.max() > 0:

                # Temperature bias bounds and maximum mass loss
                tempchange_boundlow, tempchange_boundhigh, mb_max_loss = (
                        retrieve_priors(modelparameters, glacier_rgi_table, glacier_area_initial, icethickness_initial,
                                        width_initial, elev_bins, glacier_gcm_temp, glacier_gcm_tempstd, 
                                        glacier_gcm_prec, glacier_gcm_elev, glacier_gcm_lrgcm, glacier_gcm_lrglac, 
                                        dates_table, t1_idx, t2_idx, t1, t2,
                                        debug=True
                                        ))
                if debug:
                    print('\nTC_low:', np.round(tempchange_boundlow,2), 'TC_high:', np.round(tempchange_boundhigh,2),
                          'mb_max_loss:', np.round(mb_max_loss,2))

                # If observed mass balance < max loss, then skip calibration and set to upper bound
                if observed_massbal < mb_max_loss:
                    modelparameters[7] = tempchange_boundhigh
                    mb_mwea = mb_max_loss
                    pf_opt = precfactor_init
                    tc_opt = tempchange_init

                # Otherwise, run the calibration
                else:

                    # ROUND 1: PRECIPITATION FACTOR
                    # Adjust bounds based on range of temperature bias
                    if tempchange_init > tempchange_boundhigh:
                        tempchange_init = tempchange_boundhigh
                    elif tempchange_init < tempchange_boundlow:
                        tempchange_init = tempchange_boundlow
                    modelparameters[7] = tempchange_init

                    tc_bndlow_opt = tempchange_init
                    tc_bndhigh_opt = tempchange_init

                    # Constrain bounds of precipitation factor and temperature bias
                    mb_mwea = mb_mwea_calc(
                            modelparameters, glacier_rgi_table, glacier_area_initial, icethickness_initial, 
                            width_initial, elev_bins, glacier_gcm_temp, glacier_gcm_tempstd, glacier_gcm_prec, 
                            glacier_gcm_elev, glacier_gcm_lrgcm, glacier_gcm_lrglac, dates_table, t1_idx, t2_idx, t1, 
                            t2, option_areaconstant=0)

                    if debug:
                        print('\nTC:', np.round(modelparameters[7],2), 'PF:', np.round(modelparameters[2],2),
                              'mb_mwea:', np.round(mb_mwea,2), 'obs_mwea:', np.round(observed_massbal,2))

                    # Adjust lower or upper bound based on the observed mass balance
                    test_count = 0
                    tc_step = 0.5
                    if mb_mwea > observed_massbal:
                        if debug:
                            print('increase TC, decrease PF')

                        precfactor_boundhigh = 1

                        # Check if lowest bound causes good agreement
                        modelparameters[2] = precfactor_boundlow

                        if debug:
                            print(modelparameters[2], precfactor_boundlow, precfactor_boundhigh)

                        while mb_mwea > observed_massbal and test_count < 50:
                            mb_mwea = mb_mwea_calc(
                                modelparameters, glacier_rgi_table, glacier_area_initial, icethickness_initial, 
                                width_initial, elev_bins, glacier_gcm_temp, glacier_gcm_tempstd, glacier_gcm_prec, 
                                glacier_gcm_elev, glacier_gcm_lrgcm, glacier_gcm_lrglac, dates_table, t1_idx, t2_idx, 
                                t1, t2, option_areaconstant=0)

                            if debug:
                                print('\nTC:', np.round(modelparameters[7],2), 'PF:', np.round(modelparameters[2],2),
                                      'mb_mwea:', np.round(mb_mwea,2), 'obs_mwea:', np.round(observed_massbal,2))

                            if test_count > 0:
                                tc_bndlow_opt = modelparameters[7] - tc_step
                                tc_bndhigh_opt = modelparameters[7]

                            modelparameters[7] += tc_step
                            test_count += 1
                            pf_init = np.mean([precfactor_boundlow, precfactor_boundhigh])

                    else:
                        if debug:
                            print('decrease TC, increase PF')

                        precfactor_boundlow = 1

                        # Check if upper bound causes good agreement
                        modelparameters[2] = precfactor_boundhigh

                        while mb_mwea < observed_massbal and test_count < 20:
                            mb_mwea = mb_mwea_calc(
                                modelparameters, glacier_rgi_table, glacier_area_initial, icethickness_initial, 
                                width_initial,  elev_bins, glacier_gcm_temp, glacier_gcm_tempstd, glacier_gcm_prec, 
                                glacier_gcm_elev, glacier_gcm_lrgcm, glacier_gcm_lrglac, dates_table, t1_idx, 
                                t2_idx, t1, t2, option_areaconstant=0)

                            if debug:
                                print('\nTC:', np.round(modelparameters[7],2), 'PF:', np.round(modelparameters[2],2),
                                      'mb_mwea:', np.round(mb_mwea,2), 'obs_mwea:', np.round(observed_massbal,2))

                            if test_count > 0:
                                tc_bndlow_opt = modelparameters[7]
                                tc_bndhigh_opt = modelparameters[7] + tc_step

                            modelparameters[7] -= tc_step
                            test_count += 1
                            pf_init = np.mean([precfactor_boundlow, precfactor_boundhigh])

                    # ===== RUN OPTIMIZATION WITH CONSTRAINED BOUNDS =====
                    if pygem_prms.params2opt.sort() == ['tempbias', 'precfactor'].sort():
                        # Temperature change bounds
                        tempchange_bnds = (tc_bndlow_opt, tc_bndhigh_opt)
                        precfactor_bnds = (precfactor_boundlow, precfactor_boundhigh)
                        tc_init = np.mean([tc_bndlow_opt, tc_bndhigh_opt])
                        pf_init = pf_init
    
                        modelparameters_subset = [pf_init, tc_init]
                        modelparams, mb_mwea = run_objective(modelparameters_subset, observed_massbal,
                                                             precfactor_bnds=precfactor_bnds,
                                                             tempchange_bnds=tempchange_bnds)
                        pf_opt = modelparams[2]
                        tc_opt = modelparams[7]
                        if debug:
                            print('\nmb_mwea:', np.round(mb_mwea,2), 'obs_mb:', np.round(observed_massbal,2),
                                  '\nPF:', np.round(pf_opt,2), 'TC:', np.round(tc_opt,2))
                        
                        # Epsilon (the amount the variable change to calculate the jacobian) can be too small, which causes
                        #  the minimization to believe it has reached a local minima and stop. Therefore, adjust epsilon
                        #  to ensure this is not the case.
                        eps_opt_new = pygem_prms.eps_opt
                        ftol_opt_new = pygem_prms.ftol_opt
                        nround = 0
                        while np.absolute(mb_mwea - observed_massbal) > 0.1 and eps_opt_new <= 0.1:
                            nround += 1
                            if debug:
                                print('DIDNT WORK SO TRYING NEW INITIAL CONDITIONS')
                                print('  old eps_opt:', eps_opt_new)
                                
                            eps_opt_new = eps_opt_new * 10
                            if debug:    
                                print('  new eps_opt:', eps_opt_new)
                            
                            modelparameters_subset = [pf_init, tc_init]
                            modelparams, mb_mwea = run_objective(modelparameters_subset, observed_massbal,
                                                                 precfactor_bnds=precfactor_bnds,
                                                                 tempchange_bnds=tempchange_bnds,
                                                                 eps_opt=eps_opt_new, ftol_opt=ftol_opt_new)
                            pf_opt = modelparams[2]
                            tc_opt = modelparams[7]
                            if debug:
                                print('\nmb_mwea:', np.round(mb_mwea,2), 'obs_mb:', np.round(observed_massbal,2),
                                      '\nPF:', np.round(pf_opt,2), 'TC:', np.round(tc_opt,2))
                    else:
                        # Temperature change bounds
                        tempchange_bnds = (tc_bndlow_opt, tc_bndhigh_opt)
                        precfactor_bnds = (precfactor_boundlow, precfactor_boundhigh)
                        ddfsnow_bnds = (ddfsnow_init, ddfsnow_init)
                        tc_init = np.mean([tc_bndlow_opt, tc_bndhigh_opt])
                        pf_init = pf_init
    
                        modelparameters_subset = [pf_init, modelparameters[3], modelparameters[4], tc_init]
                        modelparams, mb_mwea = run_objective(modelparameters_subset, observed_massbal,
                                                             precfactor_bnds=precfactor_bnds,
                                                             tempchange_bnds=tempchange_bnds,
                                                             ddfsnow_bnds=ddfsnow_bnds)
                        pf_opt = modelparams[2]
                        tc_opt = modelparams[7]
                        if debug:
                            print('\nmb_mwea:', np.round(mb_mwea,2), 'obs_mb:', np.round(observed_massbal,2),
                                  '\nPF:', np.round(pf_opt,2), 'TC:', np.round(tc_opt,2))
                        
                        # Epsilon (the amount the variable change to calculate the jacobian) can be too small, which causes
                        #  the minimization to believe it has reached a local minima and stop. Therefore, adjust epsilon
                        #  to ensure this is not the case.
                        eps_opt_new = pygem_prms.eps_opt
                        nround = 0
                        while np.absolute(mb_mwea - observed_massbal) > 0.3 and eps_opt_new <= 0.1:
                            nround += 1
                            if debug:
                                print('DIDNT WORK SO TRYING NEW INITIAL CONDITIONS')
                                print('  old eps_opt:', eps_opt_new)
                                
                            eps_opt_new = eps_opt_new * 10
                            if debug:    
                                print('  new eps_opt:', eps_opt_new)
                            
                            modelparameters_subset = [pf_init, modelparameters[3], modelparameters[4], tc_init]
                            modelparams, mb_mwea = run_objective(modelparameters_subset, observed_massbal,
                                                                 precfactor_bnds=precfactor_bnds,
                                                                 tempchange_bnds=tempchange_bnds,
                                                                 ddfsnow_bnds=ddfsnow_bnds,
                                                                 eps_opt = eps_opt_new)
                            pf_opt = modelparams[2]
                            tc_opt = modelparams[7]
                            if debug:
                                print('\nmb_mwea:', np.round(mb_mwea,2), 'obs_mb:', np.round(observed_massbal,2),
                                      '\nPF:', np.round(pf_opt,2), 'TC:', np.round(tc_opt,2))

                modelparameters[2] = pf_opt
                modelparameters[7] = tc_opt

            else:
                mb_mwea = 0


            # EXPORT TO NETCDF
            netcdf_output_fp = (pygem_prms.output_fp_cal)
            if not os.path.exists(netcdf_output_fp):
                os.makedirs(netcdf_output_fp)
            write_netcdf_modelparams(netcdf_output_fp + glacier_str + '.nc', modelparameters, mb_mwea, observed_massbal)

            if debug:
                print('model parameters:', tc_opt, pf_opt)
                ds = xr.open_dataset(pygem_prms.output_fp_cal + glacier_str + '.nc')
                df = pd.DataFrame(ds['mp_value'].sel(chain=0).values, columns=ds.mp.values)
                print('ds PF:', np.round(df['precfactor'].values[0],2),
                      'ds TC:', np.round(df['tempchange'].values[0],2))

        # ==============================================================

    #%%
    # Option 1: mimize mass balance difference using multi-step approach to expand solution space
    elif pygem_prms.option_calibration == 1:

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
            modelparameters[5] = modelparameters[4] / pygem_prms.ddfsnow_iceratio
            modelparameters[7] = modelparameters_subset[3]
            # Mass balance calculations
            (glac_bin_temp, glac_bin_prec, glac_bin_acc, glac_bin_refreeze, glac_bin_snowpack, glac_bin_melt,
             glac_bin_frontalablation, glac_bin_massbalclim, glac_bin_massbalclim_annual, glac_bin_area_annual,
             glac_bin_icethickness_annual, glac_bin_width_annual, glac_bin_surfacetype_annual,
             glac_wide_massbaltotal, glac_wide_runoff, glac_wide_snowline, glac_wide_snowpack,
             glac_wide_area_annual, glac_wide_volume_annual, glac_wide_ELA_annual, offglac_wide_prec,
             offglac_wide_refreeze, offglac_wide_melt, offglac_wide_snowpack, offglac_wide_runoff) = (
                massbalance.runmassbalance(modelparameters, glacier_rgi_table, glacier_area_initial, 
                                           icethickness_initial, width_initial, elev_bins, glacier_gcm_temp, 
                                           glacier_gcm_tempstd, glacier_gcm_prec, glacier_gcm_elev, glacier_gcm_lrgcm, 
                                           glacier_gcm_lrglac, dates_table, option_areaconstant=1,
                                           debug=False
                                           ))
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
                    #  sum(mass balance * area) / total area
                    t1_idx = glacier_cal_data.loc[cal_idx, 't1_idx'].astype(int)
                    t2_idx = glacier_cal_data.loc[cal_idx, 't2_idx'].astype(int)
                    z1_idx = glacier_cal_data.loc[cal_idx, 'z1_idx'].astype(int)
                    z2_idx = glacier_cal_data.loc[cal_idx, 'z2_idx'].astype(int)
                    year_idx = int(t1_idx / 12)
                    bin_area_subset = glac_bin_area_annual[z1_idx:z2_idx+1, year_idx]
                    glac_bin_massbaltotal = glac_bin_massbalclim - glac_bin_frontalablation
                    glacier_cal_compare.loc[cal_idx, 'model'] = (
                            (glac_bin_massbaltotal[z1_idx:z2_idx+1, t1_idx:t2_idx] *
                             bin_area_subset[:,np.newaxis]).sum() / bin_area_subset.sum())
                    # Fractional glacier area used to weight z-score
                    glacier_area_total = glac_bin_area_annual[:, year_idx].sum()
                    glacier_cal_compare.loc[cal_idx, 'area_frac'] = bin_area_subset.sum() / glacier_area_total
                    # Z-score for modeled mass balance based on observed mass balance and uncertainty
                    #  z-score = (model - measured) / uncertainty
                    glacier_cal_compare.loc[cal_idx, 'uncertainty'] = (pygem_prms.massbal_uncertainty_mwea *
                            (glacier_cal_data.loc[cal_idx, 't2'] - glacier_cal_data.loc[cal_idx, 't1']))
                    glacier_cal_compare.loc[cal_idx, 'zscore'] = (
                            (glacier_cal_compare.loc[cal_idx, 'model'] - glacier_cal_compare.loc[cal_idx, 'obs']) /
                            glacier_cal_compare.loc[cal_idx, 'uncertainty'])
            # Weighted z-score according to timespan and fraction of glacier area covered
            glacier_cal_compare['zscore_weighted'] = (
                    glacier_cal_compare['zscore'] * (glacier_cal_compare['t2'] - glacier_cal_compare['t1'])
                    * glacier_cal_compare['area_frac'])
            # Minimize the sum of differences
            mean_abs_zscore = abs(glacier_cal_compare['zscore_weighted']).sum() / glacier_cal_compare.shape[0]
            return mean_abs_zscore


        def run_objective(modelparameters_init, glacier_cal_data, precfactor_bnds=(0.33,3), tempchange_bnds=(-10,10),
                          ddfsnow_bnds=(0.0026,0.0056), precgrad_bnds=(0.0001,0.0001), run_opt=True):
            """
            Run the optimization for the single glacier objective function.

            Uses a z-score to enable use of different datasets (mass balance, snowline, etc.).

            Parameters
            ----------
            modelparams_init : list
                List of model parameters to calibrate
                [precipitation factor, precipitation gradient, degree day factor of snow, temperature change]
            glacier_cal_data : pd.DataFrame
                Table containing calibration data for a single glacier
            precfactor_bnds : tuple
                Lower and upper bounds for precipitation factor (default is (0.33, 3))
            tempchange_bnds : tuple
                Lower and upper bounds for temperature bias (default is (0.33, 3))
            ddfsnow_bnds : tuple
                Lower and upper bounds for degree day factor of snow (default is (0.0026, 0.0056))
            precgrad_bnds : tuple
                Lower and upper bounds for precipitation gradient (default is constant (0.0001,0.0001))
            run_opt : boolean
                Boolean statement allowing one to bypass the optimization and run through with initial parameters
                (default is True - run the optimization)

            Returns
            -------
            modelparameters_opt : optimize.optimize.OptimizeResult
                Returns result of scipy optimization, which includes optimized parameters and other information
            glacier_cal_compare : pd.DataFrame
                Table recapping calibration results: observation, model, calibration round, etc.
            """
            # Bounds
            modelparameters_bnds = (precfactor_bnds, precgrad_bnds, ddfsnow_bnds, tempchange_bnds)
            # Run the optimization
            #  'L-BFGS-B' - much slower
            #  'SLSQP' did not work for some geodetic measurements using the sum_abs_zscore.  One work around was to
            #    divide the sum_abs_zscore by 1000, which made it work in all cases.  However, methods were switched
            #    to 'L-BFGS-B', which may be slower, but is still effective.
            # note: switch enables running through with given parameters
            if run_opt:
                modelparameters_opt = minimize(objective, modelparameters_init, method=pygem_prms.method_opt,
                                               bounds=modelparameters_bnds, options={'ftol':pygem_prms.ftol_opt})
                # Record the optimized parameters
                modelparameters_subset = modelparameters_opt.x
            else:
                modelparameters_subset = modelparameters_init.copy()
            modelparams = (
                    [modelparameters[0], modelparameters[1], modelparameters_subset[0], modelparameters_subset[1],
                     modelparameters_subset[2], modelparameters_subset[2] / pygem_prms.ddfsnow_iceratio, modelparameters[6],
                     modelparameters_subset[3]])
            # Re-run the optimized parameters in order to see the mass balance
            # Mass balance calculations
            (glac_bin_temp, glac_bin_prec, glac_bin_acc, glac_bin_refreeze, glac_bin_snowpack, glac_bin_melt,
             glac_bin_frontalablation, glac_bin_massbalclim, glac_bin_massbalclim_annual, glac_bin_area_annual,
             glac_bin_icethickness_annual, glac_bin_width_annual, glac_bin_surfacetype_annual,
             glac_wide_massbaltotal, glac_wide_runoff, glac_wide_snowline, glac_wide_snowpack,
             glac_wide_area_annual, glac_wide_volume_annual, glac_wide_ELA_annual, offglac_wide_prec,
             offglac_wide_refreeze, offglac_wide_melt, offglac_wide_snowpack, offglac_wide_runoff) = (
                massbalance.runmassbalance(modelparams, glacier_rgi_table, glacier_area_initial, icethickness_initial,
                                           width_initial, elev_bins, glacier_gcm_temp, glacier_gcm_tempstd, 
                                           glacier_gcm_prec, glacier_gcm_elev, glacier_gcm_lrgcm, glacier_gcm_lrglac, 
                                           dates_table, option_areaconstant=1))
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
                    bin_area_subset = glac_bin_area_annual[z1_idx:z2_idx+1, year_idx]
                    # Fractional glacier area used to weight z-score
                    glacier_area_total = glac_bin_area_annual[:, year_idx].sum()
                    glacier_cal_compare.loc[cal_idx, 'area_frac'] = bin_area_subset.sum() / glacier_area_total


                    glac_bin_massbaltotal = glac_bin_massbalclim - glac_bin_frontalablation
                    glacier_cal_compare.loc[cal_idx, 'model'] = (
                            (glac_bin_massbaltotal[z1_idx:z2_idx+1, t1_idx:t2_idx] *
                             bin_area_subset[:,np.newaxis]).sum() / bin_area_subset.sum())

#                    glacier_cal_compare.loc[cal_idx, 'model'] = (
#                            (glac_bin_massbalclim[z1_idx:z2_idx+1, t1_idx:t2_idx] *
#                             bin_area_subset[:,np.newaxis]).sum() / bin_area_subset.sum())
                    # Z-score for modeled mass balance based on observed mass balance and uncertainty
                    #  z-score = (model - measured) / uncertainty
                    glacier_cal_compare.loc[cal_idx, 'uncertainty'] = (pygem_prms.massbal_uncertainty_mwea *
                            (glacier_cal_data.loc[cal_idx, 't2'] - glacier_cal_data.loc[cal_idx, 't1']))
                    glacier_cal_compare.loc[cal_idx, 'zscore'] = (
                            (glacier_cal_compare.loc[cal_idx, 'model'] - glacier_cal_compare.loc[cal_idx, 'obs']) /
                            glacier_cal_compare.loc[cal_idx, 'uncertainty'])
                # Weighted z-score according to timespan and fraction of glacier area covered
                glacier_cal_compare['zscore_weighted'] = (
                        glacier_cal_compare['zscore'] * (glacier_cal_compare['t2'] - glacier_cal_compare['t1'])
                        * glacier_cal_compare['area_frac'])
            return modelparams, glacier_cal_compare


        def objective_group(modelparameters_subset):
            """
            Objective function for grouped glacier data.

            The group objective cycles through all the glaciers in a group.
            Uses a z-score to enable use of different datasets (mass balance, snowline, etc.).

            Parameters
            ----------
            modelparameters_subset : np.float64
                List of model parameters to calibrate
                [precipitation factor, precipitation gradient, degree day factor of snow, temperature change]

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
                    modelparameters = [pygem_prms.lrgcm, pygem_prms.lrglac, pygem_prms.precfactor, pygem_prms.precgrad, pygem_prms.ddfsnow,
                                       pygem_prms.ddfice, pygem_prms.tempsnow, pygem_prms.tempchange]
                    if np.all(main_glac_modelparamsopt[glac] == 0) == False:
                        modelparameters = main_glac_modelparamsopt[glac]
                    else:
                        # Use a subset of model parameters to reduce number of constraints required
                        modelparameters[2] = modelparameters_subset[0]
                        modelparameters[3] = modelparameters_subset[1]
                        modelparameters[4] = modelparameters_subset[2]
                        modelparameters[5] = modelparameters[4] / pygem_prms.ddfsnow_iceratio
                        modelparameters[7] = modelparameters_subset[3]
                    # Select subsets of data
                    glacier_rgi_table = main_glac_rgi.loc[main_glac_rgi.index.values[glac], :]
                    glacier_gcm_elev = gcm_elev[glac]
                    glacier_gcm_prec = gcm_prec[glac,:]
                    glacier_gcm_temp = gcm_temp[glac,:]
                    glacier_gcm_tempstd = gcm_tempstd[glac,:]
                    glacier_gcm_lrgcm = gcm_lr[glac,:]
                    glacier_gcm_lrglac = glacier_gcm_lrgcm.copy()
                    glacier_area_initial = main_glac_hyps.iloc[glac,:].values.astype(float)
                    icethickness_initial = main_glac_icethickness.iloc[glac,:].values.astype(float)
                    width_initial = main_glac_width.iloc[glac,:].values.astype(float)
                    # Mass balance calculations
                    (glac_bin_temp, glac_bin_prec, glac_bin_acc, glac_bin_refreeze, glac_bin_snowpack, glac_bin_melt,
                     glac_bin_frontalablation, glac_bin_massbalclim, glac_bin_massbalclim_annual, glac_bin_area_annual,
                     glac_bin_icethickness_annual, glac_bin_width_annual, glac_bin_surfacetype_annual,
                     glac_wide_massbaltotal, glac_wide_runoff, glac_wide_snowline, glac_wide_snowpack,
                     glac_wide_area_annual, glac_wide_volume_annual, glac_wide_ELA_annual, offglac_wide_prec,
                     offglac_wide_refreeze, offglac_wide_melt, offglac_wide_snowpack, offglac_wide_runoff) = (
                        massbalance.runmassbalance(modelparameters, glacier_rgi_table, glacier_area_initial, 
                                                   icethickness_initial, width_initial, elev_bins, glacier_gcm_temp, 
                                                   glacier_gcm_tempstd, glacier_gcm_prec, glacier_gcm_elev, 
                                                   glacier_gcm_lrgcm, glacier_gcm_lrglac, dates_table, 
                                                   option_areaconstant=1))
                    # Mass balance comparisons
                    # Modeled mass balance [mwe]
                    #  sum(mass balance * area) / total area
                    t1_idx = cal_data.loc[cal_idx, 't1_idx'].astype(int)
                    t2_idx = cal_data.loc[cal_idx, 't2_idx'].astype(int)
                    z1_idx = 0
                    z2_idx = glac_bin_area_annual.shape[0]
                    year_idx = int(t1_idx / 12)
                    bin_area_subset = glac_bin_area_annual[z1_idx:z2_idx, year_idx]
                    group_cum_area_km2 = group_cum_area_km2 + bin_area_subset.sum()

                    print('NEED TO CHANGE TO TOTAL MASS BALANCE TO INCLUDE FRONTAL ABLATION')

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
            return abs_zscore


        def run_objective_group(modelparameters_init, precfactor_bnds=(0.33,3), tempchange_bnds=(-10,10),
                                ddfsnow_bnds=(0.0026,0.0056), precgrad_bnds=(0.0001,0.0001), run_opt=True):
            """
            Run the optimization for the group of glacier objective function.

            The group objective cycles through all the glaciers in a group.
            Uses a z-score to enable use of different datasets (mass balance, snowline, etc.).

            Parameters
            ----------
            modelparams_init : list
                List of model parameters to calibrate
                [precipitation factor, precipitation gradient, degree day factor of snow, temperature change]
            precfactor_bnds : tuple
                Lower and upper bounds for precipitation factor (default is (0.33, 3))
            tempchange_bnds : tuple
                Lower and upper bounds for temperature bias (default is (0.33, 3))
            ddfsnow_bnds : tuple
                Lower and upper bounds for degree day factor of snow (default is (0.0026, 0.0056))
            precgrad_bnds : tuple
                Lower and upper bounds for precipitation gradient (default is constant (0.0001,0.0001))
            run_opt : boolean
                Boolean statement allowing one to bypass the optimization and run through with initial parameters
                (default is True - run the optimization)

            Returns
            -------
            modelparameters_opt : optimize.optimize.OptimizeResult
                Returns result of scipy optimization, which includes optimized parameters and other information
            glacier_cal_compare : pd.DataFrame
                Table recapping calibration results: observation, model, calibration round, etc.
            glacwide_mbclim_mwe : np.ndarray
                Glacier-wide climatic mass balance [mwe] for duration of model run
                This information is used for transfer functions
            """
            # Bounds
            modelparameters_bnds = (precfactor_bnds, precgrad_bnds, ddfsnow_bnds, tempchange_bnds)
            # Run the optimization
            #  'L-BFGS-B' - much slower
            #  'SLSQP' did not work for some geodetic measurements using the sum_abs_zscore.  One work around was to
            #    divide the sum_abs_zscore by 1000, which made it work in all cases.  However, methods were switched
            #    to 'L-BFGS-B', which may be slower, but is still effective.
            # note: switch enables running through with given parameters
            if run_opt:
                modelparameters_opt = minimize(objective_group, modelparameters_init, method=pygem_prms.method_opt,
                                               bounds=modelparameters_bnds, options={'ftol':pygem_prms.ftol_opt})
                # Record the optimized parameters
                modelparameters_subset = modelparameters_opt.x
            else:
                modelparameters_subset = modelparameters_init.copy()
            modelparameters_group = (
                    [pygem_prms.lrgcm, pygem_prms.lrglac, modelparameters_subset[0], modelparameters_subset[1],
                     modelparameters_subset[2], modelparameters_subset[2] / pygem_prms.ddfsnow_iceratio, pygem_prms.tempsnow,
                     modelparameters_subset[3]])
            # Re-run the optimized parameters in order to see the mass balance
            # Record group's cumulative area and mass balance for comparison
            group_cum_area_km2 = 0
            group_cum_mb_mkm2 = 0
            glacwide_mbclim_mwe = np.zeros(len(group_dict_glaciers_idx_all))
            glac_count = 0
            # Loop through the glaciers in the group
            #  For model parameters, check if main_glac_modelparamsopt is zeros!
            #   --> if all zeros, then use the group model parameter
            #   --> if already has values, then use those values
            for glac in range(main_glac_rgi.shape[0]):
                # Check if glacier is included in group
                if main_glac_rgi.loc[glac, 'group_name'] == group_name:
                    # Set model parameter
                    # if model parameters already exist for the glacier, then use those instead of group parameters
                    if np.all(main_glac_modelparamsopt[glac] == 0) == False:
                        modelparameters = main_glac_modelparamsopt[glac]
                    else:
                        modelparameters = modelparameters_group
                    # Select subsets of data
                    glacier_rgi_table = main_glac_rgi.loc[main_glac_rgi.index.values[glac], :]
                    glacier_gcm_elev = gcm_elev[glac]
                    glacier_gcm_prec = gcm_prec[glac,:]
                    glacier_gcm_temp = gcm_temp[glac,:]
                    glacier_gcm_tempstd = gcm_tempstd[glac,:]
                    glacier_gcm_lrgcm = gcm_lr[glac,:]
                    glacier_gcm_lrglac = glacier_gcm_lrgcm.copy()
                    glacier_area_initial = main_glac_hyps.iloc[glac,:].values.astype(float)
                    icethickness_initial = main_glac_icethickness.iloc[glac,:].values.astype(float)
                    width_initial = main_glac_width.iloc[glac,:].values.astype(float)
                    # Mass balance calculations
                    (glac_bin_temp, glac_bin_prec, glac_bin_acc, glac_bin_refreeze, glac_bin_snowpack, glac_bin_melt,
                     glac_bin_frontalablation, glac_bin_massbalclim, glac_bin_massbalclim_annual, glac_bin_area_annual,
                     glac_bin_icethickness_annual, glac_bin_width_annual, glac_bin_surfacetype_annual,
                     glac_wide_massbaltotal, glac_wide_runoff, glac_wide_snowline, glac_wide_snowpack,
                     glac_wide_area_annual, glac_wide_volume_annual, glac_wide_ELA_annual, offglac_wide_prec,
                     offglac_wide_refreeze, offglac_wide_melt, offglac_wide_snowpack, offglac_wide_runoff) = (
                        massbalance.runmassbalance(modelparameters, glacier_rgi_table, glacier_area_initial, 
                                                   icethickness_initial, width_initial, elev_bins, glacier_gcm_temp, 
                                                   glacier_gcm_tempstd, glacier_gcm_prec, glacier_gcm_elev, 
                                                   glacier_gcm_lrgcm, glacier_gcm_lrglac, dates_table,
                                                   option_areaconstant=1))
                    # Mass balance comparisons
                    # Modeled mass balance [mwe]
                    #  sum(mass balance * area) / total area
                    t1_idx = cal_data.loc[cal_idx, 't1_idx'].astype(int)
                    t2_idx = cal_data.loc[cal_idx, 't2_idx'].astype(int)
                    z1_idx = 0
                    z2_idx = glac_bin_area_annual.shape[0]
                    year_idx = int(t1_idx / 12)
                    bin_area_subset = glac_bin_area_annual[z1_idx:z2_idx, year_idx]
                    group_cum_area_km2 = group_cum_area_km2 + bin_area_subset.sum()

                    print('NEED TO CHANGE TO TOTAL MASS BALANCE NOT CLIMATIC FOR TIDEWATER GLACIERS')

                    group_cum_mb_mkm2 = (
                            group_cum_mb_mkm2 +
                            (glac_bin_massbalclim[z1_idx:z2_idx, t1_idx:t2_idx] * bin_area_subset[:,np.newaxis]).sum())
                    # Glacier-wide climatic mass balance over study period (used by transfer functions)
                    glacwide_mbclim_mwe[glac_count] = (
                            (glac_bin_massbalclim * glac_bin_area_annual[:, 0][:,np.newaxis]).sum() /
                             glac_bin_area_annual[:, 0].sum())
                    glac_count += 1
            # Z-score for modeled mass balance based on observed mass balance and uncertainty
            #  z-score = (model - measured) / uncertainty
            glacier_cal_compare.model = group_cum_mb_mkm2 / group_cum_area_km2
            glacier_cal_compare.zscore = (
                    (glacier_cal_compare.model - glacier_cal_compare.obs) / glacier_cal_compare.uncertainty)
            return modelparameters_group, glacier_cal_compare, glacwide_mbclim_mwe


        def zscore_compare(glacier_cal_compare, cal_idx):
            """
            Compare z-scores to determine if need to increase solution space for the calibration

            Parameters
            ----------
            glacier_cal_compare : pd.DataFrame
                Table recapping calibration results: observation, model, calibration round, etc.
            cal_idx : list
                Indices of calibration data

            Returns
            -------
            zscore_compare <= zscore_tolerance : Boolean
                Returns True or False depending on if the zscore is less than the specified tolerance
            """
            # Set zscore to compare and the tolerance
            # if only one calibration point, then zscore should be small
            if glacier_cal_compare.shape[0] == 1:
                zscore4comparison = glacier_cal_compare.loc[cal_idx[0], 'zscore']
                zscore_tolerance = pygem_prms.zscore_tolerance_single
            # else if multiple calibration points and one is a geodetic MB, check that geodetic MB is within 1
            elif (glacier_cal_compare.obs_type.isin(['mb_geo']).any() == True) and (glacier_cal_compare.shape[0] > 1):
                zscore4comparison = glacier_cal_compare.loc[glacier_cal_compare.index.values[np.where(
                        glacier_cal_compare['obs_type'] == 'mb_geo')[0][0]], 'zscore']
                zscore_tolerance = pygem_prms.zscore_tolerance_all
            # otherwise, check mean zscore
            else:
                zscore4comparison = abs(glacier_cal_compare['zscore']).sum() / glacier_cal_compare.shape[0]
                zscore_tolerance = pygem_prms.zscore_tolerance_all
            return abs(zscore4comparison) <= zscore_tolerance


        def init_guess_frombounds(bnds_list, calround, bnd_idx):
            """
            Set initial guess of a parameter based on the current and previous bounds

            This sets the initial guess somewhere in the middle of the new solution space as opposed to being
            set in the middle of the solution space.

            Parameters
            ----------
            bnds_list : list
                List of tuples containing the bounds of each parameter for each round
            calround : int
                Calibration round
            bnd_idx : int
                Index for whether to use the upper or lower bounds

            Returns
            -------
            initial guess : float
                Returns initial guess based on the previous and new bounds for a given parameters
            """
            return (bnds_list[calround][bnd_idx] + bnds_list[calround-1][bnd_idx]) / 2


        def write_netcdf_modelparams(output_fullfn, modelparameters, glacier_cal_compare=pd.DataFrame()):
                """
                Export glacier model parameters and modeled observations to netcdf file.

                Parameters
                ----------
                output_fullfn : str
                    Full filename (path included) of the netcdf to be exported
                modelparams_init : list
                    List of model parameters to calibrate
                    [precipitation factor, precipitation gradient, degree day factor of snow, temperature change]
                glacier_cal_compare : pd.DataFrame
                    Table recapping calibration results: observation, model, calibration round, etc.
                    (default is empty dataframe)

                Returns
                -------
                None
                    Exports file to netcdf.  Does not return anything.
                """
                # Select data from model to be stored in netcdf
                df = pd.DataFrame(index=[0])
                df['lrgcm'] = np.full(df.shape[0], pygem_prms.lrgcm)
                df['lrglac'] = np.full(df.shape[0], pygem_prms.lrglac)
                df['precfactor'] = modelparameters[2]
                df['precgrad'] = np.full(df.shape[0], pygem_prms.precgrad)
                df['ddfsnow'] = modelparameters[4]
                df['ddfice'] = df['ddfsnow'] / pygem_prms.ddfsnow_iceratio
                df['tempsnow'] = np.full(df.shape[0], pygem_prms.tempsnow)
                df['tempchange'] = modelparameters[7]
                # Loop through observations to help create dataframe
                for x in range(glacier_cal_compare.shape[0]):
                    obs_cn = 'obs_' + str(x)
                    df[obs_cn] = glacier_cal_compare.loc[glacier_cal_data.index.values,'model'].values[x]
                df_export = df.values[:, :, np.newaxis]
                # Set up dataset and export to netcdf
                ds = xr.Dataset({'mp_value': (('iter', 'mp', 'chain'), df_export)},
                                coords={'iter': df.index.values,
                                        'mp': df.columns.values,
                                        'chain': [0]})
                ds.to_netcdf(output_fullfn)
                ds.close()

        # ==============================================================
        # ===== Individual glacier optimization using objective minimization =====
        # Output
        output_cols = ['glacno', 'obs_type', 'obs_unit', 'obs', 'model', 'uncertainty', 'zscore', 't1', 't2',
                       'area_frac', 'zscore_weighted', 'calround']
        main_glac_cal_compare = pd.DataFrame(np.zeros((cal_data.shape[0],len(output_cols))),
                                             columns=output_cols)
        main_glac_cal_compare.index = cal_data.index.values
        # Model parameters
        main_glac_modelparamsopt = np.zeros((main_glac_rgi.shape[0], len(pygem_prms.modelparams_colnames)))
        # Glacier-wide climatic mass balance (required for transfer fucntions)
        main_glacwide_mbclim_mwe = np.zeros((main_glac_rgi.shape[0], 1))

        # Loop through glaciers that have unique cal_data
        cal_individual_glacno = np.unique(cal_data.loc[cal_data['glacno'].notnull(), 'glacno'])
        for n in range(cal_individual_glacno.shape[0]):
            glac = np.where(main_glac_rgi['rgino_str'].isin([cal_individual_glacno[n]]) == True)[0][0]
            if debug:
                print(count, ':', main_glac_rgi.loc[main_glac_rgi.index.values[glac], 'RGIId'])
            elif glac%100 == 0:
                print(count, ':', main_glac_rgi.loc[main_glac_rgi.index.values[glac], 'RGIId'])
            # Set model parameters
            modelparameters = [pygem_prms.lrgcm, pygem_prms.lrglac, pygem_prms.precfactor, pygem_prms.precgrad, pygem_prms.ddfsnow, pygem_prms.ddfice,
                               pygem_prms.tempsnow, pygem_prms.tempchange]
            # Select subsets of data
            glacier_rgi_table = main_glac_rgi.loc[main_glac_rgi.index.values[glac], :]
            glacier_gcm_elev = gcm_elev[glac]
            glacier_gcm_prec = gcm_prec[glac,:]
            glacier_gcm_temp = gcm_temp[glac,:]
            glacier_gcm_tempstd = gcm_tempstd[glac,:]
            glacier_gcm_lrgcm = gcm_lr[glac,:]
            glacier_gcm_lrglac = glacier_gcm_lrgcm.copy()
            glacier_area_initial = main_glac_hyps.iloc[glac,:].values.astype(float)
            icethickness_initial = main_glac_icethickness.iloc[glac,:].values.astype(float)
            width_initial = main_glac_width.iloc[glac,:].values.astype(float)
            glacier_cal_data = ((cal_data.iloc[np.where(
                    glacier_rgi_table['rgino_str'] == cal_data['glacno'])[0],:]).copy())
            cal_idx = glacier_cal_data.index.values
            glacier_str = '{0:0.5f}'.format(glacier_rgi_table['RGIId_float'])
            # Comparison dataframe
            glacier_cal_compare = pd.DataFrame(np.zeros((glacier_cal_data.shape[0], len(output_cols))),
                                               columns=output_cols)
            glacier_cal_compare.index = glacier_cal_data.index.values
            glacier_cal_compare[['glacno', 'obs_type', 't1', 't2']] = (
                    glacier_cal_data[['glacno', 'obs_type', 't1', 't2']])
            calround = 0
            # Initial bounds and switch to manipulate bounds
            precfactor_bnds_list = pygem_prms.precfactor_bnds_list_init.copy()
            tempchange_bnds_list = pygem_prms.tempchange_bnds_list_init.copy()
            ddfsnow_bnds_list = pygem_prms.ddfsnow_bnds_list_init.copy()
            precgrad_bnds_list = pygem_prms.precgrad_bnds_list_init.copy()
            manipulate_bnds = False

            modelparameters_init = [pygem_prms.precfactor, pygem_prms.precgrad, pygem_prms.ddfsnow, pygem_prms.tempchange]
            modelparameters, glacier_cal_compare = (
                    run_objective(modelparameters_init, glacier_cal_data, run_opt=False))
            zscore_sum = glacier_cal_compare.zscore_weighted.sum()
            precfactor_bnds_list = [pygem_prms.precfactor_bnds_list_init[0]]
            tempchange_bnds_list = [pygem_prms.tempchange_bnds_list_init[0]]
            ddfsnow_bnds_list = [pygem_prms.ddfsnow_bnds_list_init[0]]

            if debug:
                print('precfactor_bnds_list:', precfactor_bnds_list)

            precgrad_init_idx = 0
            if zscore_sum < 0:
                # Index used for calculating initial guess each round
                precfactor_init_idx = 1
                tempchange_init_idx = 0
                ddfsnow_init_idx = 0
                # First tuple will not be changed in case calibrated parameters are near the center,
                # subsequent bounds will be modified
                precfactor_bnds_list_modified = [(1,i[1]) for i in pygem_prms.precfactor_bnds_list_init[1:]]
                tempchange_bnds_list_modified = [(i[0],0) for i in pygem_prms.tempchange_bnds_list_init[1:]]
                ddfsnow_bnds_list_modified = [(i[0],0.0041) for i in pygem_prms.ddfsnow_bnds_list_init[1:]]
                precfactor_bnds_list.extend(precfactor_bnds_list_modified)
                tempchange_bnds_list.extend(tempchange_bnds_list_modified)
                ddfsnow_bnds_list.extend(ddfsnow_bnds_list_modified)
            else:
                # Index used for calculating initial guess each round
                precfactor_init_idx = 0
                tempchange_init_idx = 1
                ddfsnow_init_idx = 1
                # First tuple will not be changed in case calibrated parameters are near the center,
                # subsequent bounds will be modified
                precfactor_bnds_list_modified = [(i[0],1) for i in pygem_prms.precfactor_bnds_list_init[1:]]
                tempchange_bnds_list_modified = [(0,i[1]) for i in pygem_prms.tempchange_bnds_list_init[1:]]
                ddfsnow_bnds_list_modified = [(0.0041,i[1]) for i in pygem_prms.ddfsnow_bnds_list_init[1:]]
                precfactor_bnds_list.extend(precfactor_bnds_list_modified)
                tempchange_bnds_list.extend(tempchange_bnds_list_modified)
                ddfsnow_bnds_list.extend(ddfsnow_bnds_list_modified)

            if debug:
                print('modified precfactor_bnds_list:', precfactor_bnds_list)


            zscore_weighted_total = None
            init_calrounds = len(pygem_prms.precfactor_bnds_list_init)
            continue_loop = True
            while continue_loop:
                # Bounds
                if calround < init_calrounds:
                    precfactor_bnds = precfactor_bnds_list[calround]
                    precgrad_bnds = precgrad_bnds_list[calround]
                    ddfsnow_bnds = ddfsnow_bnds_list[calround]
                    tempchange_bnds = tempchange_bnds_list[calround]
                # Initial guess
                if calround == 0:
                    modelparameters_init = [pygem_prms.precfactor, pygem_prms.precgrad, pygem_prms.ddfsnow, pygem_prms.tempchange]
                elif manipulate_bnds:
                    modelparameters_init = (
                                [init_guess_frombounds(precfactor_bnds_list, calround, precfactor_init_idx),
                                 init_guess_frombounds(precgrad_bnds_list, calround, precgrad_init_idx),
                                 init_guess_frombounds(ddfsnow_bnds_list, calround, ddfsnow_init_idx),
                                 init_guess_frombounds(tempchange_bnds_list, calround, tempchange_init_idx)])
                else:
                    modelparameters_init = (
                            [modelparameters[2], modelparameters[3], modelparameters[4], modelparameters[7]])

                # For additional calibration rounds if the optimization gets stuck, manually push the temperature bias
                if calround >= init_calrounds:
                    # Adjust temperature bias
                    if zscore_weighted_total > 0:
                        modelparameters_init[3] = modelparameters_init[3] + (calround - init_calrounds + 1) * 1
                    else:
                        modelparameters_init[3] = modelparameters_init[3] - (calround - init_calrounds + 1) * 1

                # Run optimization
                modelparameters, glacier_cal_compare = (
                        run_objective(modelparameters_init, glacier_cal_data, precfactor_bnds, tempchange_bnds,
                                      ddfsnow_bnds, precgrad_bnds))
                calround += 1

                # Update model parameters if significantly better
                zscore_weighted_total = glacier_cal_compare['zscore_weighted'].sum()
                mean_zscore = abs(glacier_cal_compare['zscore_weighted']).sum() / glacier_cal_compare.shape[0]
                if calround == 1:
                    mean_zscore_best = mean_zscore
                    modelparameters_best = modelparameters
                    glacier_cal_compare_best = glacier_cal_compare
                    cal_round_best = calround
                elif (mean_zscore_best - mean_zscore) > pygem_prms.zscore_update_threshold:
                    mean_zscore_best = mean_zscore
                    modelparameters_best = modelparameters
                    glacier_cal_compare_best = glacier_cal_compare
                    cal_round_best = calround

                total_calrounds = init_calrounds + pygem_prms.extra_calrounds
                # Break loop if gone through all iterations
                if (calround == total_calrounds) or zscore_compare(glacier_cal_compare, cal_idx):
                    continue_loop = False

                if debug:
                    print('\nCalibration round:', calround,
                          '\nInitial parameters:\n  Precfactor:', modelparameters_init[0],
                          '\n  Tempbias:', modelparameters_init[3], '\n  DDFsnow:', modelparameters_init[2])
                    print('Calibrated parameters:\n  Precfactor:', modelparameters[2],
                          '\n  Tempbias:', modelparameters[7], '\n  DDFsnow:', modelparameters[4])
                    if glacier_cal_compare.shape[0] == 1:
                        print('model:', np.round(glacier_cal_compare.loc[0, 'model'],2),
                              'obs:', np.round(glacier_cal_compare.loc[0, 'obs'],2))
                    print('zscore_weighted_total:', np.round(zscore_weighted_total,2), '\n')

            # RECORD OUTPUT
            mean_zscore = mean_zscore_best
            modelparameters = modelparameters_best
            glacier_cal_compare = glacier_cal_compare_best
            calround = cal_round_best
            # Run mass balance with optimized parameters
            (glac_bin_temp, glac_bin_prec, glac_bin_acc, glac_bin_refreeze, glac_bin_snowpack, glac_bin_melt,
             glac_bin_frontalablation, glac_bin_massbalclim, glac_bin_massbalclim_annual, glac_bin_area_annual,
             glac_bin_icethickness_annual, glac_bin_width_annual, glac_bin_surfacetype_annual,
             glac_wide_massbaltotal, glac_wide_runoff, glac_wide_snowline, glac_wide_snowpack,
             glac_wide_area_annual, glac_wide_volume_annual, glac_wide_ELA_annual, offglac_wide_prec,
             offglac_wide_refreeze, offglac_wide_melt, offglac_wide_snowpack, offglac_wide_runoff) = (
                massbalance.runmassbalance(modelparameters, glacier_rgi_table, glacier_area_initial, 
                                           icethickness_initial, width_initial, elev_bins, glacier_gcm_temp, 
                                           glacier_gcm_tempstd, glacier_gcm_prec, glacier_gcm_elev, glacier_gcm_lrgcm, 
                                           glacier_gcm_lrglac, dates_table, option_areaconstant=1, debug=False))
            # Calibration round
            glacier_cal_compare['calround'] = calround
            # Model vs. observations
            main_glac_cal_compare.loc[glacier_cal_data.index.values] = glacier_cal_compare
            # Glacier-wide climatic mass balance over study period (used by transfer functions)
            main_glacwide_mbclim_mwe[glac] = (
                    (glac_bin_massbalclim * glac_bin_area_annual[:, 0][:,np.newaxis]).sum() /
                    glac_bin_area_annual[:, 0].sum())
            main_glac_modelparamsopt[glac] = modelparameters

            # EXPORT TO NETCDF
            netcdf_output_fp = (pygem_prms.output_fp_cal + 'reg' + str(glacier_rgi_table.O1Region) + '/')
            if not os.path.exists(netcdf_output_fp):
                os.makedirs(netcdf_output_fp)
            # Loop through glaciers calibrated from group and export to netcdf
            glacier_str = '{0:0.5f}'.format(main_glac_rgi.loc[main_glac_rgi.index.values[glac], 'RGIId_float'])
            write_netcdf_modelparams(netcdf_output_fp + glacier_str + '.nc', modelparameters, glacier_cal_compare)

        # ==============================================================
        # ===== GROUP CALIBRATION =====
        if set(['group']).issubset(pygem_prms.cal_datasets) == True:
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
                # Note: glacier_cal_compare is a pd.Series!

                # Calibration round
                calround = 0
                # Initial bounds and switch to manipulate bounds
                precfactor_bnds_list = pygem_prms.precfactor_bnds_list_init.copy()
                tempchange_bnds_list = pygem_prms.tempchange_bnds_list_init.copy()
                ddfsnow_bnds_list = pygem_prms.ddfsnow_bnds_list_init.copy()
                precgrad_bnds_list = pygem_prms.precgrad_bnds_list_init.copy()
                manipulate_bnds = False

                modelparameters_init = [pygem_prms.precfactor, pygem_prms.precgrad, pygem_prms.ddfsnow, pygem_prms.tempchange]
                modelparameters_group, glacier_cal_compare, glacwide_mbclim_mwe = (
                        run_objective_group(modelparameters_init, run_opt=False))
                zscore_sum = glacier_cal_compare.zscore.sum()

                precfactor_bnds_list = [pygem_prms.precfactor_bnds_list_init[0]]
                tempchange_bnds_list = [pygem_prms.tempchange_bnds_list_init[0]]
                ddfsnow_bnds_list = [pygem_prms.ddfsnow_bnds_list_init[0]]

                if debug:
                    print('precfactor_bnds_list:', precfactor_bnds_list)

                precgrad_init_idx = 0
                if zscore_sum < 0:
                    # Index used for calculating initial guess each round
                    precfactor_init_idx = 1
                    tempchange_init_idx = 0
                    ddfsnow_init_idx = 0
                    # First tuple will not be changed in case calibrated parameters are near the center,
                    # subsequent bounds will be modified
                    precfactor_bnds_list_modified = [(1,i[1]) for i in pygem_prms.precfactor_bnds_list_init[1:]]
                    tempchange_bnds_list_modified = [(i[0],0) for i in pygem_prms.tempchange_bnds_list_init[1:]]
                    ddfsnow_bnds_list_modified = [(i[0],0.0041) for i in pygem_prms.ddfsnow_bnds_list_init[1:]]
                    precfactor_bnds_list.extend(precfactor_bnds_list_modified)
                    tempchange_bnds_list.extend(tempchange_bnds_list_modified)
                    ddfsnow_bnds_list.extend(ddfsnow_bnds_list_modified)
                else:
                    # Index used for calculating initial guess each round
                    precfactor_init_idx = 0
                    tempchange_init_idx = 1
                    ddfsnow_init_idx = 1
                    # First tuple will not be changed in case calibrated parameters are near the center,
                    # subsequent bounds will be modified
                    precfactor_bnds_list_modified = [(i[0],1) for i in pygem_prms.precfactor_bnds_list_init[1:]]
                    tempchange_bnds_list_modified = [(0,i[1]) for i in pygem_prms.tempchange_bnds_list_init[1:]]
                    ddfsnow_bnds_list_modified = [(0.0041,i[1]) for i in pygem_prms.ddfsnow_bnds_list_init[1:]]
                    precfactor_bnds_list.extend(precfactor_bnds_list_modified)
                    tempchange_bnds_list.extend(tempchange_bnds_list_modified)
                    ddfsnow_bnds_list.extend(ddfsnow_bnds_list_modified)

                if debug:
                    print('modified precfactor_bnds_list:', precfactor_bnds_list)

                continue_loop = True
                while continue_loop:
                    # Bounds
                    precfactor_bnds = precfactor_bnds_list[calround]
                    precgrad_bnds = precgrad_bnds_list[calround]
                    ddfsnow_bnds = ddfsnow_bnds_list[calround]
                    tempchange_bnds = tempchange_bnds_list[calround]
                    # Initial guess
                    if calround == 0:
                        modelparameters_init = [pygem_prms.precfactor, pygem_prms.precgrad, pygem_prms.ddfsnow, pygem_prms.tempchange]
                    elif manipulate_bnds:
                        modelparameters_init = (
                                    [init_guess_frombounds(precfactor_bnds_list, calround, precfactor_init_idx),
                                     init_guess_frombounds(precgrad_bnds_list, calround, precgrad_init_idx),
                                     init_guess_frombounds(ddfsnow_bnds_list, calround, ddfsnow_init_idx),
                                     init_guess_frombounds(tempchange_bnds_list, calround, tempchange_init_idx)])
                    else:
                        modelparameters_init = (
                                [modelparameters_group[2], modelparameters_group[3], modelparameters_group[4],
                                 modelparameters_group[7]])
                    # Run optimization
                    modelparameters_group, glacier_cal_compare, glacwide_mbclim_mwe = (
                        run_objective_group(modelparameters_init, precfactor_bnds, tempchange_bnds, ddfsnow_bnds,
                                            precgrad_bnds))
                    calround += 1

                    # Update model parameters if significantly better
                    zscore_abs = abs(glacier_cal_compare['zscore']).sum()
                    if calround == 1:
                        zscore_abs_best = zscore_abs
                        modelparameters_group_best = modelparameters_group
                        glacier_cal_compare_best = glacier_cal_compare
                        cal_round_best = calround
                    elif (zscore_abs_best - zscore_abs) > pygem_prms.zscore_update_threshold:
                        zscore_abs_best = zscore_abs
                        modelparameters_group_best = modelparameters_group
                        glacier_cal_compare_best = glacier_cal_compare
                        cal_round_best = calround

                    # Break loop if gone through all iterations
                    if ((calround == len(pygem_prms.precfactor_bnds_list_init)) or
                        (abs(glacier_cal_compare.zscore) < pygem_prms.zscore_tolerance_single)):
                        continue_loop = False

                    if debug:
                        print('\nCalibration round:', calround,
                              '\nInitial parameters:\n  Precfactor:', modelparameters_init[0],
                              '\n  Tempbias:', modelparameters_init[3], '\n  DDFsnow:', modelparameters_init[2])
                        print('Calibrated parameters:\n  Precfactor:', modelparameters_group[2],
                              '\n  Tempbias:', modelparameters_group[7], '\n  DDFsnow:', modelparameters_group[4])

                # Glacier-wide climatic mass balance over study period (used by transfer functions)
                # Record model parameters and mbclim
                group_count = 0
                for glac in range(main_glac_rgi.shape[0]):
                    if main_glac_rgi.loc[glac, 'group_name'] == group_name:
                        main_glacwide_mbclim_mwe[glac] = glacwide_mbclim_mwe[group_count]
                        group_count += 1
                main_glac_modelparamsopt[group_dict_glaciers_idx] = modelparameters_group_best
                glacier_cal_compare.calround = calround
                main_glac_cal_compare.loc[cal_idx] = glacier_cal_compare_best

                # EXPORT TO NETCDF
                netcdf_output_fp = (pygem_prms.output_fp_cal + 'reg' + str(glacier_rgi_table.O1Region) + '/')
                if not os.path.exists(netcdf_output_fp):
                    os.makedirs(netcdf_output_fp)
                # Loop through glaciers calibrated from group (skipping those calibrated indepdently)
                for glac_idx in group_dict_glaciers_idx:
                    glacier_str = '{0:0.5f}'.format(main_glac_rgi.loc[glac_idx, 'RGIId_float'])
                    write_netcdf_modelparams(netcdf_output_fp + glacier_str + '.nc', modelparameters_group)

                print(group_name,'(zscore):', abs(glacier_cal_compare.zscore))
                print('precfactor:', modelparameters_group[2])
                print('precgrad:', modelparameters_group[3])
                print('ddfsnow:', modelparameters_group[4])
                print('ddfice:', modelparameters_group[5])
                print('tempchange:', modelparameters_group[7])
                print('calround:', calround, '\n')

        # Export (i) main_glac_rgi w optimized model parameters and glacier-wide climatic mass balance,
        #        (ii) comparison of model vs. observations
        # Concatenate main_glac_rgi, optimized model parameters, glacier-wide climatic mass balance
        main_glac_output = main_glac_rgi.copy()
        main_glac_modelparamsopt_pd = pd.DataFrame(main_glac_modelparamsopt, columns=pygem_prms.modelparams_colnames)
        main_glac_modelparamsopt_pd.index = main_glac_rgi.index.values
        main_glacwide_mbclim_pd = pd.DataFrame(main_glacwide_mbclim_mwe, columns=['mbclim_mwe'])
        main_glac_output = pd.concat([main_glac_output, main_glac_modelparamsopt_pd, main_glacwide_mbclim_pd], axis=1)

        # Export output
        if not os.path.exists(pygem_prms.output_fp_cal + 'temp/'):
            os.makedirs(pygem_prms.output_fp_cal + 'temp/')
        regions_str = 'R'
        for region in pygem_prms.rgi_regionsO1:
            regions_str += str(region)
        output_modelparams_fn = (
                regions_str + '_modelparams_opt' + str(pygem_prms.option_calibration) + '_' + gcm_name
                + str(pygem_prms.startyear) + str(pygem_prms.endyear) + '--' + str(count) + '.csv')
        output_calcompare_fn = (
                regions_str + '_calcompare_opt' + str(pygem_prms.option_calibration) + '_' + gcm_name
                + str(pygem_prms.startyear) + str(pygem_prms.endyear) + '--' + str(count) + '.csv')
        main_glac_output.to_csv(pygem_prms.output_fp_cal + 'temp/' + output_modelparams_fn )
        main_glac_cal_compare.to_csv(pygem_prms.output_fp_cal + 'temp/' + output_calcompare_fn)

    # Export variables as global to view in variable explorer
    if args.option_parallels == 0:
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

    if pygem_prms.option_calibration == 2:
        print('Chains:', pygem_prms.n_chains, 'Iterations:', pygem_prms.mcmc_sample_no)
        
    # RGI glacier number
    if args.rgi_glac_number_fn is not None:
        with open(args.rgi_glac_number_fn, 'rb') as f:
            glac_no = pickle.load(f)
        rgi_regionsO1 = sorted(list(set([int(x.split('.')[0]) for x in glac_no])))
        regions_str = args.rgi_glac_number_fn.split('_')[0]
    else:
        glac_no = pygem_prms.glac_no
        rgi_regionsO1 = pygem_prms.rgi_regionsO1
        regions_str = 'R'
        for region in pygem_prms.rgi_regionsO1:
            regions_str += str(region)

    # Select all glaciers in a region
    main_glac_rgi_all = modelsetup.selectglaciersrgitable(
            rgi_regionsO1=rgi_regionsO1, rgi_regionsO2 =pygem_prms.rgi_regionsO2, rgi_glac_number=pygem_prms.rgi_glac_number,  
            glac_no=glac_no)
    # Add regions
    main_glac_rgi_all['region'] = main_glac_rgi_all.RGIId.map(pygem_prms.reg_dict)

    # Define chunk size for parallel processing
    if args.option_parallels != 0:
        num_cores = int(np.min([main_glac_rgi_all.shape[0], args.num_simultaneous_processes]))
        chunk_size = int(np.ceil(main_glac_rgi_all.shape[0] / num_cores))
    else:
        # if not running in parallel, chunk size is all glaciers
        chunk_size = main_glac_rgi_all.shape[0]

#    # Pack variables for parallel processing
#    list_packed_vars = []
#    n = 0
#    for chunk in range(0, main_glac_rgi_all.shape[0], chunk_size):
#        n += 1
#        list_packed_vars.append([n, chunk, chunk_size, main_glac_rgi_all, gcm_name])


    #%%===================================================
    # Pack variables for multiprocessing
#    list_packed_vars = []
#    for chunk in range(0, main_glac_rgi_all.shape[0], chunk_size):
#        main_glac_rgi_raw = main_glac_rgi_all.loc[chunk:chunk+chunk_size-1].copy()
#        list_packed_vars.append([main_glac_rgi_raw, gcm_name])

    # ===== LOAD GLACIER DATA =====
    # Glacier hypsometry [km**2], total area
    main_glac_hyps_all = modelsetup.import_Husstable(main_glac_rgi_all, pygem_prms.hyps_filepath,
                                                     pygem_prms.hyps_filedict, pygem_prms.hyps_colsdrop)

    # Ice thickness [m], average
    main_glac_icethickness_all = modelsetup.import_Husstable(main_glac_rgi_all, pygem_prms.thickness_filepath, 
                                                             pygem_prms.thickness_filedict, pygem_prms.thickness_colsdrop)
    main_glac_icethickness_all[main_glac_icethickness_all < 0] = 0
    main_glac_hyps_all[main_glac_icethickness_all == 0] = 0
    # Width [km], average
    main_glac_width_all = modelsetup.import_Husstable(main_glac_rgi_all, pygem_prms.width_filepath,
                                                      pygem_prms.width_filedict, pygem_prms.width_colsdrop)
    elev_bins = main_glac_hyps_all.columns.values.astype(int)
    # Add volume [km**3] and mean elevation [m a.s.l.]
    main_glac_rgi_all['Volume'], main_glac_rgi_all['Zmean'] = (
            modelsetup.hypsometrystats(main_glac_hyps_all, main_glac_icethickness_all))
    # Select dates including future projections
    #  - nospinup dates_table needed to get the proper time indices
    dates_table_nospinup  = modelsetup.datesmodelrun(startyear=pygem_prms.startyear, endyear=pygem_prms.endyear,
                                                     spinupyears=0)
    dates_table = modelsetup.datesmodelrun(startyear=pygem_prms.startyear, endyear=pygem_prms.endyear,
                                           spinupyears=pygem_prms.spinupyears)

    # ===== LOAD CALIBRATION DATA =====
    cal_data = pd.DataFrame()
    for dataset in pygem_prms.cal_datasets:
        cal_subset = class_mbdata.MBData(name=dataset)
        cal_subset_data = cal_subset.retrieve_mb(main_glac_rgi_all, main_glac_hyps_all, dates_table_nospinup)
        cal_data = cal_data.append(cal_subset_data, ignore_index=True)
    cal_data = cal_data.sort_values(['glacno', 't1_idx'])
    cal_data.reset_index(drop=True, inplace=True)

    # If group data is included, then add group dictionary and add group name to main_glac_rgi
    if set(['group']).issubset(pygem_prms.cal_datasets) == True:
        # Group dictionary
        group_dict_raw = pd.read_csv(pygem_prms.mb_group_fp + pygem_prms.mb_group_dict_fn)
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
        main_glac_rgi_all['group_name'] = main_glac_rgi_all['RGIId'].map(group_dict)
    else:
        main_glac_rgi_all['group_name'] = np.nan

    # Drop glaciers that do not have any calibration data (individual or group)
    main_glac_rgi = ((main_glac_rgi_all.iloc[np.unique(
            np.append(main_glac_rgi_all[main_glac_rgi_all['group_name'].notnull() == True].index.values,
                      np.where(main_glac_rgi_all['rgino_str'].isin(cal_data['glacno']) == True)[0])), :])
            .copy())
    # select glacier data
    main_glac_hyps = main_glac_hyps_all.iloc[main_glac_rgi.index.values]
    main_glac_icethickness = main_glac_icethickness_all.iloc[main_glac_rgi.index.values]
    main_glac_width = main_glac_width_all.iloc[main_glac_rgi.index.values]
    # Reset index
    main_glac_rgi.reset_index(drop=True, inplace=True)
    main_glac_hyps.reset_index(drop=True, inplace=True)
    main_glac_icethickness.reset_index(drop=True, inplace=True)
    main_glac_width.reset_index(drop=True, inplace=True)

    # ===== LOAD CLIMATE DATA =====
    gcm = class_climate.GCM(name=gcm_name)
    # Air temperature [degC], Air temperature Std [K], Precipitation [m], Elevation [masl], Lapse rate [K m-1]
    gcm_temp, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(gcm.temp_fn, gcm.temp_vn, main_glac_rgi, dates_table)
    gcm_prec, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(gcm.prec_fn, gcm.prec_vn, main_glac_rgi, dates_table)
    gcm_elev = gcm.importGCMfxnearestneighbor_xarray(gcm.elev_fn, gcm.elev_vn, main_glac_rgi)
    # Air temperature standard deviation [K]
    if pygem_prms.option_ablation != 2 or gcm_name not in ['ERA5']:
        gcm_tempstd = np.zeros(gcm_temp.shape)
    elif gcm_name in ['ERA5']:
        gcm_tempstd, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(gcm.tempstd_fn, gcm.tempstd_vn, 
                                                                        main_glac_rgi, dates_table)
    # Lapse rate [K m-1]
    if gcm_name in ['ERA-Interim', 'ERA5']:
        gcm_lr, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(gcm.lr_fn, gcm.lr_vn, main_glac_rgi, dates_table)
    else:
        # Mean monthly lapse rate
        ref_lr_monthly_avg = np.genfromtxt(gcm.lr_fp + gcm.lr_fn, delimiter=',')
        gcm_lr = np.tile(ref_lr_monthly_avg, int(gcm_temp.shape[1]/12))
    # COAWST data has two domains, so need to merge the two domains
    if gcm_name == 'COAWST':
        gcm_temp_d01, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(gcm.temp_fn_d01, gcm.temp_vn, main_glac_rgi,
                                                                         dates_table)
        gcm_prec_d01, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(gcm.prec_fn_d01, gcm.prec_vn, main_glac_rgi,
                                                                         dates_table)
        gcm_elev_d01 = gcm.importGCMfxnearestneighbor_xarray(gcm.elev_fn_d01, gcm.elev_vn, main_glac_rgi)
        # Check if glacier outside of high-res (d02) domain
        for glac in range(main_glac_rgi.shape[0]):
            glac_lat = main_glac_rgi.loc[glac,pygem_prms.rgi_lat_colname]
            glac_lon = main_glac_rgi.loc[glac,pygem_prms.rgi_lon_colname]
            if (~(pygem_prms.coawst_d02_lat_min <= glac_lat <= pygem_prms.coawst_d02_lat_max) or
                ~(pygem_prms.coawst_d02_lon_min <= glac_lon <= pygem_prms.coawst_d02_lon_max)):
                gcm_prec[glac,:] = gcm_prec_d01[glac,:]
                gcm_temp[glac,:] = gcm_temp_d01[glac,:]
                gcm_elev[glac] = gcm_elev_d01[glac]

    # Pack variables for multiprocessing
    list_packed_vars = []
    n = 0
    for chunk in range(0, main_glac_rgi.shape[0], chunk_size):
        n += 1
        main_glac_rgi_chunk = main_glac_rgi.loc[chunk:chunk+chunk_size-1].copy()
        main_glac_hyps_chunk = main_glac_hyps.loc[chunk:chunk+chunk_size-1].copy()
        main_glac_icethickness_chunk = main_glac_icethickness.loc[chunk:chunk+chunk_size-1].copy()
        main_glac_width_chunk = main_glac_width.loc[chunk:chunk+chunk_size-1].copy()
        gcm_temp_chunk = gcm_temp[chunk:chunk+chunk_size]
        gcm_tempstd_chunk = gcm_tempstd[chunk:chunk+chunk_size]
        gcm_prec_chunk = gcm_prec[chunk:chunk+chunk_size]
        gcm_elev_chunk = gcm_elev[chunk:chunk+chunk_size]
        gcm_lr_chunk = gcm_lr[chunk:chunk+chunk_size]
        cal_data_chunk = cal_data.loc[chunk:chunk+chunk_size-1]

        list_packed_vars.append([n,
                                 gcm_name,
                                 main_glac_rgi_chunk,
                                 main_glac_hyps_chunk,
                                 main_glac_icethickness_chunk,
                                 main_glac_width_chunk,
                                 gcm_temp_chunk,
                                 gcm_tempstd_chunk,
                                 gcm_prec_chunk,
                                 gcm_elev_chunk,
                                 gcm_lr_chunk,
                                 cal_data_chunk
                                 ])

    #%%===================================================
    # Parallel processing
    if args.option_parallels != 0:
        print('Processing in parallel with ' + str(num_cores) + ' cores...')
        with multiprocessing.Pool(args.num_simultaneous_processes) as p:
            p.map(main,list_packed_vars)
    # If not in parallel, then only should be one loop
    else:
        for n in range(len(list_packed_vars)):
            main(list_packed_vars[n])

#    print('memory:', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 10**6, 'GB')

    #%%

#    # Combine output (if desired)
#    if pygem_prms.option_calibration == 1:
#        # Merge csv summary files into single file
#        # Count glaciers
#        glac_count = 0
#        output_temp = pygem_prms.output_fp_cal + 'temp/'
#        regions_str = 'R'
#        for region in rgi_regionsO1:
#            regions_str += str(region)
#        for i in os.listdir(output_temp):
#            if i.startswith(regions_str) and i.endswith('.nc'):
#                glac_count += 1
#
#        # Model parameters - combine into single file
#        check_modelparams_str = (
#                'modelparams_opt' + str(pygem_prms.option_calibration) + '_' + gcm_name + str(pygem_prms.startyear) +
#                str(pygem_prms.endyear))
#        output_modelparams_all_fn = (
#                regions_str + '_' + str(glac_count) + check_modelparams_str + '.csv')
#        # Sorted list of files to merge
#        output_list = []
#        for i in os.listdir(output_temp):
#            if check_modelparams_str in i:
#                output_list.append(i)
#        output_list = sorted(output_list)
#        # Merge model parameters csv summary file
#        list_count = 0
#        for i in output_list:
#            list_count += 1
#            # Append results
#            if list_count == 1:
#                output_all = pd.read_csv(pygem_prms.output_fp_cal + 'temp/' + i, index_col=0)
#            else:
#                output_2join = pd.read_csv(pygem_prms.output_fp_cal + 'temp/' + i, index_col=0)
#                output_all = output_all.append(output_2join, ignore_index=True)
#            # Remove file after its been merged
#            os.remove(pygem_prms.output_fp_cal + 'temp/' + i)
#        # Export joined files
#        output_all.to_csv(pygem_prms.output_fp_cal + output_modelparams_all_fn)
#
#        # Calibration comparison - combine into single file
#        check_calcompare_str = (
#                'calcompare_opt' + str(pygem_prms.option_calibration) + '_' + gcm_name + str(pygem_prms.startyear) +
#                str(pygem_prms.endyear))
#        output_calcompare_all_fn = (
#                regions_str + '_' + str(glac_count) + check_calcompare_str + '.csv')
#        # Sorted list of files to merge
#        output_list = []
#        for i in os.listdir(output_temp):
#            if check_calcompare_str in i:
#                output_list.append(i)
#        output_list = sorted(output_list)
#        # Merge cal compare csv summary file
#        list_count = 0
#        for i in output_list:
#            list_count += 1
#            # Append results
#            if list_count == 1:
#                output_all = pd.read_csv(pygem_prms.output_fp_cal + 'temp/' + i, index_col=0)
#            else:
#                output_2join = pd.read_csv(pygem_prms.output_fp_cal + 'temp/' + i, index_col=0)
#                output_all = output_all.append(output_2join, ignore_index=True)
#            # Remove file after its been merged
#            os.remove(pygem_prms.output_fp_cal + 'temp/' + i)
#        # Export joined files
#        output_all.to_csv(pygem_prms.output_fp_cal + output_calcompare_all_fn)
#
#    print('Total processing time:', time.time()-time_start, 's')

    #%% ===== PLOTTING AND PROCESSING FOR MODEL DEVELOPMENT =====
#    # Place local variables in variable explorer
#    if (args.option_parallels == 0):
#        main_vars_list = list(main_vars.keys())
#        main_glac_rgi = main_vars['main_glac_rgi']
#        main_glac_hyps = main_vars['main_glac_hyps']
#        main_glac_icethickness = main_vars['main_glac_icethickness']
#        main_glac_width = main_vars['main_glac_width']
#        elev_bins = main_vars['elev_bins']
#        dates_table = main_vars['dates_table']
#        dates_table_nospinup = main_vars['dates_table_nospinup']
#        cal_data = main_vars['cal_data']
#        gcm_temp = main_vars['gcm_temp']
#        gcm_prec = main_vars['gcm_prec']
#        gcm_elev = main_vars['gcm_elev']
#        gcm_lr = main_vars['gcm_lr']
#        modelparameters = main_vars['modelparameters']
#        glacier_area_initial = main_vars['glacier_area_initial']
#        glacier_cal_data = main_vars['glacier_cal_data']
#        cal_idx = main_vars['cal_idx']
#        modelparameters = main_vars['modelparameters']
#        icethickness_initial = main_vars['icethickness_initial']
#        width_initial = main_vars['width_initial']
#
#        if pygem_prms.option_calibration == 2 and pygem_prms.new_setup == 1:
#            observed_massbal=main_vars['observed_massbal']
#            observed_error=main_vars['observed_error']
#            mb_max_loss = main_vars['mb_max_loss']
#            tempchange_boundlow = main_vars['tempchange_boundlow']
#            tempchange_boundhigh = main_vars['tempchange_boundhigh']
#            tempchange_mu = main_vars['tempchange_mu']
#            tempchange_sigma = main_vars['tempchange_sigma']
#            tempchange_start = main_vars['tempchange_start']
#            t1_idx = main_vars['t1_idx']
#            t2_idx = main_vars['t2_idx']
#            t1 = main_vars['t1']
#            t2 = main_vars['t2']
#            iterations=pygem_prms.mcmc_sample_no
#            burn=pygem_prms.mcmc_burn_no
#            step=pygem_prms.mcmc_step
#            precfactor_boundlow = main_vars['precfactor_boundlow']
#            precfactor_boundhigh = main_vars['precfactor_boundhigh']
#            glacier_gcm_prec = main_vars['glacier_gcm_prec']
#            glacier_gcm_temp = main_vars['glacier_gcm_temp']
#            glacier_gcm_elev = main_vars['glacier_gcm_elev']
#            glacier_rgi_table = main_vars['glacier_rgi_table']
#            glacier_gcm_lrgcm = main_vars['glacier_gcm_lrgcm']
#            glacier_gcm_lrglac = main_vars['glacier_gcm_lrglac']
#
#        if pygem_prms.option_calibration == 1:
#            glacier_cal_compare = main_vars['glacier_cal_compare']
#            main_glac_cal_compare = main_vars['main_glac_cal_compare']
#            main_glac_modelparamsopt = main_vars['main_glac_modelparamsopt']
#            main_glac_output = main_vars['main_glac_output']
#            main_glac_modelparamsopt_pd = main_vars['main_glac_modelparamsopt_pd']
#            main_glacwide_mbclim_mwe = main_vars['main_glacwide_mbclim_mwe']
#            if set(['group']).issubset(pygem_prms.cal_datasets):
#                group_dict_keyslist = main_vars['group_dict_keyslist']
#                group_dict_keyslist_names = main_vars['group_dict_keyslist_names']
#                cal_data_idx_groups = main_vars['cal_data_idx_groups']
#                cal_data = main_vars['cal_data']
#                cal_individual_glacno = main_vars['cal_individual_glacno']