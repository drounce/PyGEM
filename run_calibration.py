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
    kp_bndlow, kp_bndhigh, kp_mu, kp_start : floats
        data for precipitation factor's prior distribution
    tbias_bndlow, tbias_bndhigh, tbias_mu, tbias_sigma, tbias_start : floats
        data for temperature bias' prior distribution
    tbias_max_loss, tbias_max_acc, mb_max_loss, mb_max_acc : floats
        temperature change and mass balance associated with maximum accumulation and maximum loss
    """

    # ----- TEMPBIAS: max accumulation -----
    # Lower temperature bound based on max positive mass balance adjusted to avoid edge effects
    # Temperature at the lowest bin
    #  T_bin = T_gcm + lr_gcm * (z_ref - z_gcm) + lr_glac * (z_bin - z_ref) + tbias
    lowest_bin = np.where(glacier_area_initial > 0)[0][0]
    tbias_max_acc = (-1 * (glacier_gcm_temp + glacier_gcm_lrgcm *
                                (elev_bins[lowest_bin] - glacier_gcm_elev)).max())

    if debug:
        print('tc_max_acc:', np.round(tbias_max_acc,2))
        
    # ----- TEMPBIAS: UPPER BOUND -----
    # MAXIMUM LOSS - AREA EVOLVING
    #  note: the mb_mwea_calc function ensures the area is constant until t1 such that the glacier is not completely
    #        lost before t1; otherwise, this will fail at high TC values
    mb_max_loss = (-1 * (glacier_area_initial * icethickness_initial).sum() / glacier_area_initial.sum() *
                   pygem_prms.density_ice / pygem_prms.density_water / (t2 - t1))

    if debug:
        print('mb_max_loss:', np.round(mb_max_loss,2), 'kp:', np.round(modelparameters[2],2))

    # Looping forward and backward to ensure optimization does not get stuck
    modelparameters[7] = tbias_max_acc    
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
    # Looping backward for tbias at max loss
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

    tbias_max_loss = modelparameters[7]
    tbias_bndhigh = tbias_max_loss

    if debug:
        print('tc_max_loss:', np.round(tbias_max_loss,2), 'mb_max_loss:', np.round(mb_max_loss,2))

    # Lower bound based on must melt condition
    #  note: since the mass balance ablation is conditional on the glacier evolution, there can be cases where higher
    #  temperature biases still have 0 for nyears_negmbclim. Hence, the need to loop beyond the first instance, and
    #  then go back and check that you're using the good cases from there onward. This ensures starting point is good
    modelparameters[7] = tbias_max_acc
    nyears_negmbclim = mb_mwea_calc(
                modelparameters, glacier_rgi_table, glacier_area_initial, icethickness_initial, width_initial, 
                elev_bins, glacier_gcm_temp, glacier_gcm_tempstd, glacier_gcm_prec, glacier_gcm_elev, glacier_gcm_lrgcm, 
                glacier_gcm_lrglac, dates_table, t1_idx, t2_idx, t1, t2, 
                option_areaconstant=0, return_tc_mustmelt=1)
    
    nyears_negmbclim_list = [nyears_negmbclim]
    tc_negmbclim_list = [modelparameters[7]]
    tc_smallstep_switch = False
    while nyears_negmbclim < 10 and modelparameters[7] < tbias_max_loss:
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
        
        # First time nyears_negmbclim is > 0, flip the switch to use smalll step and restart with last tbias
        if nyears_negmbclim > 0 and tc_smallstep_switch == False:
            tc_smallstep_switch = True
            modelparameters[7] = modelparameters_old
            nyears_negmbclim = 0

        if debug:
            print('TC:', np.round(modelparameters[7],2), 'nyears_negmbclim:', nyears_negmbclim)

    tbias_bndlow = tc_negmbclim_list[np.where(np.array(nyears_negmbclim_list) == 0)[0][-1] + 1]

    return tbias_bndlow, tbias_bndhigh, mb_max_loss


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
    # Option 2: use MCMC method to determine posterior probability distributions of the three parameters tbias,
    #           ddfsnow and kp. Then create an ensemble of parameter sets evenly sampled from these
    #           distributions, and output these sets of parameters and their corresponding mass balances to be used in
    #           the simulations.
    if pygem_prms.option_calibration == 'MCMC':

        # ===== Define functions needed for MCMC method =====
        def run_MCMC(kp_disttype=pygem_prms.kp_disttype,
                     kp_gamma_alpha=pygem_prms.kp_gamma_alpha,
                     kp_gamma_beta=pygem_prms.kp_gamma_beta,
                     kp_lognorm_mu=pygem_prms.kp_lognorm_mu,
                     kp_lognorm_tau=pygem_prms.kp_lognorm_tau,
                     kp_mu=pygem_prms.kp_mu, kp_sigma=pygem_prms.kp_sigma,
                     kp_bndlow=pygem_prms.kp_bndlow, kp_bndhigh=pygem_prms.kp_bndhigh,
                     kp_start=pygem_prms.kp_start,
                     tbias_disttype=pygem_prms.tbias_disttype,
                     tbias_mu=pygem_prms.tbias_mu, tbias_sigma=pygem_prms.tbias_sigma,
                     tbias_bndlow=pygem_prms.tbias_bndlow, tbias_bndhigh=pygem_prms.tbias_bndhigh,
                     tbias_start=pygem_prms.tbias_start,
                     ddfsnow_disttype=pygem_prms.ddfsnow_disttype,
                     ddfsnow_mu=pygem_prms.ddfsnow_mu, ddfsnow_sigma=pygem_prms.ddfsnow_sigma,
                     ddfsnow_bndlow=pygem_prms.ddfsnow_bndlow, ddfsnow_bndhigh=pygem_prms.ddfsnow_bndhigh,
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
            kp_disttype : str
                Distribution type of precipitation factor (either 'lognormal', 'uniform', or 'custom')
            kp_lognorm_mu : float
                Lognormal mean of precipitation factor (default assigned from input)
            kp_lognorm_tau : float
                Lognormal tau (1/variance) of precipitation factor (default assigned from input)
            kp_mu : float
                Mean of precipitation factor (default assigned from input)
            kp_sigma : float
                Standard deviation of precipitation factor (default assigned from input)
            kp_bndlow : float
                Lower boundary of precipitation factor (default assigned from input)
            kp_bndhigh : float
                Upper boundary of precipitation factor (default assigned from input)
            kp_start : float
                Starting value of precipitation factor for sampling iterations (default assigned from input)
            tbias_disttype : str
                Distribution type of tbias (either 'truncnormal' or 'uniform')
            tbias_mu : float
                Mean of temperature change (default assigned from input)
            tbias_sigma : float
                Standard deviation of temperature change (default assigned from input)
            tbias_bndlow : float
                Lower boundary of temperature change (default assigned from input)
            tbias_bndhigh: float
                Upper boundary of temperature change (default assigned from input)
            tbias_start : float
                Starting value of temperature change for sampling iterations (default assigned from input)
            ddfsnow_disttype : str
                Distribution type of degree day factor of snow (either 'truncnormal' or 'uniform')
            ddfsnow_mu : float
                Mean of degree day factor of snow (default assigned from input)
            ddfsnow_sigma : float
                Standard deviation of degree day factor of snow (default assigned from input)
            ddfsnow_bndlow : float
                Lower boundary of degree day factor of snow (default assigned from input)
            ddfsnow_bndhigh : float
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
                Returns a model that contains sample traces of tbias, ddfsnow, kp and massbalance. These
                samples can be accessed by calling the trace attribute. For example:

                    model.trace('ddfsnow')[:]

                gives the trace of ddfsnow values.

                A trace, or Markov Chain, is an array of values outputed by the MCMC simulation which defines the
                posterior probability distribution of the variable at hand.
            """
            # ===== PRIOR DISTRIBUTIONS =====
            # Precipitation factor [-]
            if kp_disttype == 'gamma':
                kp = pymc.Gamma('kp', alpha=kp_gamma_alpha, beta=kp_gamma_beta,
                                        value=kp_start)
            elif kp_disttype =='lognormal':
                #  lognormal distribution (roughly 0.3 to 3)
                kp_start = np.exp(kp_start)
                kp = pymc.Lognormal('kp', mu=kp_lognorm_mu, tau=kp_lognorm_tau,
                                            value=kp_start)
            elif kp_disttype == 'uniform':
                kp = pymc.Uniform('kp', lower=kp_bndlow, upper=kp_bndhigh,
                                          value=kp_start)
            # Temperature change [degC]
            if tbias_disttype == 'normal':
                tbias = pymc.Normal('tbias', mu=tbias_mu, tau=1/(tbias_sigma**2),
                                         value=tbias_start)
            elif tbias_disttype =='truncnormal':
                tbias = pymc.TruncatedNormal('tbias', mu=tbias_mu, tau=1/(tbias_sigma**2),
                                                  a=tbias_bndlow, b=tbias_bndhigh, value=tbias_start)
            elif tbias_disttype =='uniform':
                tbias = pymc.Uniform('tbias', lower=tbias_bndlow, upper=tbias_bndhigh,
                                          value=tbias_start)

            # Degree day factor of snow [mwe degC-1 d-1]
            #  always truncated normal distribution with mean 0.0041 mwe degC-1 d-1 and standard deviation of 0.0015
            #  (Braithwaite, 2008), since it's based on data; uniform should only be used for testing
            if ddfsnow_disttype == 'truncnormal':
                ddfsnow = pymc.TruncatedNormal('ddfsnow', mu=ddfsnow_mu, tau=1/(ddfsnow_sigma**2), a=ddfsnow_bndlow,
                                               b=ddfsnow_bndhigh, value=ddfsnow_start)
            if ddfsnow_disttype == 'uniform':
                ddfsnow = pymc.Uniform('ddfsnow', lower=ddfsnow_bndlow, upper=ddfsnow_bndhigh,
                                       value=ddfsnow_start)

            # ===== DETERMINISTIC FUNCTION ====
            # Define deterministic function for MCMC model based on our a priori probobaility distributions.
            @deterministic(plot=False)
            def massbal(tbias=tbias, kp=kp, ddfsnow=ddfsnow):
                """
                Likelihood function for mass balance [mwea] based on model parameters
                """
                modelparameters_copy = modelparameters.copy()
                if tbias is not None:
                    modelparameters_copy[7] = float(tbias)
                if kp is not None:
                    modelparameters_copy[2] = float(kp)
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
            def must_melt(tbias=tbias, kp=kp, ddfsnow=ddfsnow):
                """
                Likelihood function for mass balance [mwea] based on model parameters
                """
                modelparameters_copy = modelparameters.copy()
                if tbias is not None:
                    modelparameters_copy[7] = float(tbias)
                if kp is not None:
                    modelparameters_copy[2] = float(kp)
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
                model = pymc.MCMC([{'kp':kp, 'tbias':tbias, 'ddfsnow':ddfsnow,
                                   'massbal':massbal, 'obs_massbal':obs_massbal}, mb_max, must_melt])
            else:
                model = pymc.MCMC({'kp':kp, 'tbias':tbias, 'ddfsnow':ddfsnow,
                                   'massbal':massbal, 'obs_massbal':obs_massbal})
#            if dbname is not None:
#                model = pymc.MCMC({'kp':kp, 'tbias':tbias, 'ddfsnow':ddfsnow,
#                                   'massbal':massbal, 'obs_massbal':obs_massbal}, db='pickle', dbname=dbname)

            # Step method (if changed from default)
            #  Adaptive metropolis is supposed to perform block update, i.e., update all model parameters together based
            #  on their covariance, which would reduce autocorrelation; however, tests show doesn't make a difference.
            if step == 'am':
                model.use_step_method(pymc.AdaptiveMetropolis, [kp, tbias, ddfsnow], delay = 1000)

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
            modelparameters = [pygem_prms.lrgcm, pygem_prms.lrglac, pygem_prms.kp, pygem_prms.precgrad, pygem_prms.ddfsnow, pygem_prms.ddfice,
                               pygem_prms.tsnow_threshold, pygem_prms.tbias]

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
                kp_gamma_alpha = pygem_prms.kp_gamma_region_dict[glacier_rgi_table.loc['region']][0]
                kp_gamma_beta = pygem_prms.kp_gamma_region_dict[glacier_rgi_table.loc['region']][1]
                tbias_mu = pygem_prms.tbias_norm_region_dict[glacier_rgi_table.loc['region']][0]
                tbias_sigma = pygem_prms.tbias_norm_region_dict[glacier_rgi_table.loc['region']][1]

                # fit the MCMC model
                for n_chain in range(0,pygem_prms.n_chains):

                    if debug:
                        print('\n', glacier_str, ' chain' + str(n_chain))

                    if n_chain == 0:
                        # Starting values: middle
                        tbias_start = tbias_mu
                        kp_start = kp_gamma_alpha / kp_gamma_beta
                        ddfsnow_start = pygem_prms.ddfsnow_mu

                    elif n_chain == 1:
                        # Starting values: lowest
                        tbias_start = tbias_mu - 1.96 * tbias_sigma
                        ddfsnow_start = pygem_prms.ddfsnow_mu - 1.96 * pygem_prms.ddfsnow_sigma
                        kp_start = stats.gamma.ppf(0.05,kp_gamma_alpha, scale=1/kp_gamma_beta)

                    elif n_chain == 2:
                        # Starting values: high
                        tbias_start = tbias_mu + 1.96 * tbias_sigma
                        ddfsnow_start = pygem_prms.ddfsnow_mu + 1.96 * pygem_prms.ddfsnow_sigma
                        kp_start = stats.gamma.ppf(0.95,kp_gamma_alpha, scale=1/kp_gamma_beta)


                    # Determine bounds to check TC starting values and estimate maximum mass loss
                    modelparameters[2] = kp_start
                    modelparameters[4] = ddfsnow_start
                    modelparameters[5] = ddfsnow_start / pygem_prms.ddfsnow_iceratio

                    tbias_bndlow, tbias_bndhigh, mb_max_loss = (
                            retrieve_priors(modelparameters, glacier_rgi_table, glacier_area_initial, icethickness_initial,
                                            width_initial, elev_bins, glacier_gcm_temp, glacier_gcm_tempstd, 
                                            glacier_gcm_prec, glacier_gcm_elev, glacier_gcm_lrgcm, glacier_gcm_lrglac, 
                                            dates_table, t1_idx, t2_idx, t1, t2, debug=False))

                    if debug:
                        print('\nTC_low:', np.round(tbias_bndlow,2),
                              'TC_high:', np.round(tbias_bndhigh,2),
                              'mb_max_loss:', np.round(mb_max_loss,2))

                    # Check that tbias mu and sigma is somewhat within bndhigh and bndlow
                    if tbias_start > tbias_bndhigh:
                        tbias_start = tbias_bndhigh
                    elif tbias_start < tbias_bndlow:
                        tbias_start = tbias_bndlow

#                    # Check that tbias mu and sigma is somewhat within bndhigh and bndlow
#                    if ((tbias_bndhigh < tbias_mu - 3 * tbias_sigma) or
#                        (tbias_bndlow > tbias_mu + 3 * tbias_sigma)):
#                        tbias_mu = np.mean([tbias_bndlow, tbias_bndhigh])
#                        tbias_sigma = (tbias_bndhigh - tbias_bndlow) / 6

                    if debug:
                        print('\ntc_start:', np.round(tbias_start,3),
                              '\npf_start:', np.round(kp_start,3),
                              '\nddf_start:', np.round(ddfsnow_start,4))

                    model = run_MCMC(iterations=pygem_prms.mcmc_sample_no, burn=pygem_prms.mcmc_burn_no, step=pygem_prms.mcmc_step,
                                     kp_gamma_alpha=kp_gamma_alpha,
                                     kp_gamma_beta=kp_gamma_beta,
                                     kp_start=kp_start,
                                     tbias_mu=tbias_mu, tbias_sigma=tbias_sigma,
                                     tbias_start=tbias_start,
                                     ddfsnow_start=ddfsnow_start,
                                     mb_max_loss=mb_max_loss)

                    if debug:
                        print('\nacceptance ratio:', model.step_method_dict[next(iter(model.stochastics))][0].ratio)

                    # Select data from model to be stored in netcdf
                    df = pd.DataFrame({'tbias': model.trace('tbias')[:],
                                       'kp': model.trace('kp')[:],
                                       'ddfsnow': model.trace('ddfsnow')[:],
                                       'massbal': model.trace('massbal')[:]})
                    # set columns for other variables
                    df['ddfice'] = df['ddfsnow'] / pygem_prms.ddfsnow_iceratio
                    df['lrgcm'] = np.full(df.shape[0], pygem_prms.lrgcm)
                    df['lrglac'] = np.full(df.shape[0], pygem_prms.lrglac)
                    df['precgrad'] = np.full(df.shape[0], pygem_prms.precgrad)
                    df['tsnow_threshold'] = np.full(df.shape[0], pygem_prms.tsnow_threshold)

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
    elif pygem_prms.option_calibration == 'HH2015':

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


        def run_objective(modelparameters_init, observed_massbal, kp_bnds=(0.33,3), tbias_bnds=(-10,10),
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
            kp_bnds : tuple
                Lower and upper bounds for precipitation factor (default is (0.33, 3))
            tbias_bnds : tuple
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
            modelparameters_bnds = (kp_bnds, precgrad_bnds, ddfsnow_bnds, tbias_bnds)
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
            df['kp'] = modelparameters[2]
            df['precgrad'] = np.full(df.shape[0], pygem_prms.precgrad)
            df['ddfsnow'] = modelparameters[4]
            df['ddfice'] = df['ddfsnow'] / ddfsnow_iceratio
            df['tsnow_threshold'] = np.full(df.shape[0], pygem_prms.tsnow_threshold)
            df['tbias'] = modelparameters[7]
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
        tbias_init = 0
        tbias_bndlow = -10
        tbias_bndhigh = 10
        kp_init = 1.5
        kp_bndlow = 0.8
        kp_bndhigh = 2
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
            modelparameters = [pygem_prms.lrgcm, pygem_prms.lrglac, kp_init, pygem_prms.precgrad, ddfsnow_init, pygem_prms.ddfice,
                               pygem_prms.tsnow_threshold, tbias_init]
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
            kp_bnds = (kp_bndlow, kp_bndhigh)
            ddfsnow_bnds = (ddfsnow_init, ddfsnow_init)
            tbias_bnds = (tbias_init, tbias_init)
            modelparams, mb_mwea = run_objective(modelparameters_subset, observed_massbal,
                                                 kp_bnds=kp_bnds, tbias_bnds=tbias_bnds,
                                                 ddfsnow_bnds=ddfsnow_bnds)
            kp_opt = modelparams[2]
            if debug:
                print('mb_mwea:', np.round(mb_mwea,2), 'kp:', np.round(kp_opt,2))

            # Round 2: optimize DDFsnow
            if debug:
                print('Round 2:')
            modelparameters_subset = [kp_opt, modelparameters[3], modelparameters[4], modelparameters[7]]
            kp_bnds = (kp_opt, kp_opt)
            ddfsnow_bnds = (ddfsnow_bndlow, ddfsnow_bndhigh)
            tbias_bnds = (tbias_init, tbias_init)
            modelparams, mb_mwea = run_objective(modelparameters_subset, observed_massbal,
                                                 kp_bnds=kp_bnds, tbias_bnds=tbias_bnds,
                                                 ddfsnow_bnds=ddfsnow_bnds)

            ddfsnow_opt = modelparams[4]
            if debug:
                print('mb_mwea:', np.round(mb_mwea,2), 'kp:', np.round(kp_opt,2),
                      'ddfsnow:', np.round(ddfsnow_opt,5))

            # Round 3: optimize tempbias
            if debug:
                print('Round 3:')

            # ----- TEMPBIAS: max accumulation -----
            # Lower temperature bound based on no positive temperatures
            # Temperature at the lowest bin
            #  T_bin = T_gcm + lr_gcm * (z_ref - z_gcm) + lr_glac * (z_bin - z_ref) + tbias
            lowest_bin = np.where(glacier_area_initial > 0)[0][0]
            tbias_max_acc = (-1 * (glacier_gcm_temp + glacier_gcm_lrgcm *
                                        (elev_bins[lowest_bin] - glacier_gcm_elev)).max())
            tbias_bndlow = tbias_max_acc

            if debug:
                print('tbias_bndlow:', np.round(tbias_bndlow,2))

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
                    if modelparameters[7] < tbias_bndlow:
                        modelparameters[7] = tbias_bndlow

                modelparameters_subset = [kp_opt, modelparameters[3], ddfsnow_opt, modelparameters[7]]
                kp_bnds = (kp_opt, kp_opt)
                ddfsnow_bnds = (ddfsnow_opt, ddfsnow_opt)
                tbias_bnds = (tbias_bndlow, tbias_bndhigh)
                modelparams, mb_mwea = run_objective(modelparameters_subset, observed_massbal,
                                                     kp_bnds=kp_bnds, tbias_bnds=tbias_bnds,
                                                     ddfsnow_bnds=ddfsnow_bnds)
                dif_mb_mwea = abs(observed_massbal - mb_mwea)

                count += 1
                if debug:
                    print('dif:', np.round(dif_mb_mwea,2), 'count:', count, 'tc:', np.round(modelparameters[7],2))

                # Break loop if at lower bound
                if abs(tbias_bndlow - modelparams[7]) < 0.1:
                    count=20

            # Record optimal temperature bias
            tbias_opt = modelparams[7]

            if debug:
                print('mb_mwea:', np.round(mb_mwea,2), 'kp:', np.round(kp_opt,2),
                      'ddfsnow:', np.round(ddfsnow_opt,5), 'tbias:', np.round(tbias_opt,2))


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
    elif pygem_prms.option_calibration == 'HH2015_modified':
        
        if pygem_prms.params2opt.sort() == ['tbias', 'kp'].sort():
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
    
            def run_objective(modelparameters_init, observed_massbal, kp_bnds=(0.33,3), tbias_bnds=(-10,10),
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
                kp_bnds : tuple
                    Lower and upper bounds for precipitation factor (default is (0.33, 3))
                tbias_bnds : tuple
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
                modelparameters_bnds = (kp_bnds, tbias_bnds)
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
    
            def run_objective(modelparameters_init, observed_massbal, kp_bnds=(0.33,3), tbias_bnds=(-10,10),
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
                kp_bnds : tuple
                    Lower and upper bounds for precipitation factor (default is (0.33, 3))
                tbias_bnds : tuple
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
                modelparameters_bnds = (kp_bnds, precgrad_bnds, ddfsnow_bnds, tbias_bnds)
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
            df['kp'] = modelparameters[2]
            df['precgrad'] = np.full(df.shape[0], pygem_prms.precgrad)
            df['ddfsnow'] = modelparameters[4]
            df['ddfice'] = df['ddfsnow'] / ddfsnow_iceratio
            df['tsnow_threshold'] = np.full(df.shape[0], pygem_prms.tsnow_threshold)
            df['tbias'] = modelparameters[7]
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
            tbias_init = 0
            kp_init = 1
            kp_bndlow = 0.5
            kp_bndhigh = 5
            ddfsnow_init = 0.0041
            ddfsnow_iceratio = 0.7

            if debug:
                print(main_glac_rgi.loc[main_glac_rgi.index.values[glac],'RGIId_float'])

            # Set model parameters
            modelparameters = [pygem_prms.lrgcm, pygem_prms.lrglac, kp_init, pygem_prms.precgrad, ddfsnow_init, pygem_prms.ddfice,
                               pygem_prms.tsnow_threshold, tbias_init]
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
                tbias_bndlow, tbias_bndhigh, mb_max_loss = (
                        retrieve_priors(modelparameters, glacier_rgi_table, glacier_area_initial, icethickness_initial,
                                        width_initial, elev_bins, glacier_gcm_temp, glacier_gcm_tempstd, 
                                        glacier_gcm_prec, glacier_gcm_elev, glacier_gcm_lrgcm, glacier_gcm_lrglac, 
                                        dates_table, t1_idx, t2_idx, t1, t2,
                                        debug=True
                                        ))
                if debug:
                    print('\nTC_low:', np.round(tbias_bndlow,2), 'TC_high:', np.round(tbias_bndhigh,2),
                          'mb_max_loss:', np.round(mb_max_loss,2))

                # If observed mass balance < max loss, then skip calibration and set to upper bound
                if observed_massbal < mb_max_loss:
                    modelparameters[7] = tbias_bndhigh
                    mb_mwea = mb_max_loss
                    pf_opt = kp_init
                    tc_opt = tbias_init

                # Otherwise, run the calibration
                else:

                    # ROUND 1: PRECIPITATION FACTOR
                    # Adjust bounds based on range of temperature bias
                    if tbias_init > tbias_bndhigh:
                        tbias_init = tbias_bndhigh
                    elif tbias_init < tbias_bndlow:
                        tbias_init = tbias_bndlow
                    modelparameters[7] = tbias_init

                    tc_bndlow_opt = tbias_init
                    tc_bndhigh_opt = tbias_init

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

                        kp_bndhigh = 1

                        # Check if lowest bound causes good agreement
                        modelparameters[2] = kp_bndlow

                        if debug:
                            print(modelparameters[2], kp_bndlow, kp_bndhigh)

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
                            pf_init = np.mean([kp_bndlow, kp_bndhigh])

                    else:
                        if debug:
                            print('decrease TC, increase PF')

                        kp_bndlow = 1

                        # Check if upper bound causes good agreement
                        modelparameters[2] = kp_bndhigh

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
                            pf_init = np.mean([kp_bndlow, kp_bndhigh])

                    # ===== RUN OPTIMIZATION WITH CONSTRAINED BOUNDS =====
                    if pygem_prms.params2opt.sort() == ['tbias', 'kp'].sort():
                        # Temperature change bounds
                        tbias_bnds = (tc_bndlow_opt, tc_bndhigh_opt)
                        kp_bnds = (kp_bndlow, kp_bndhigh)
                        tc_init = np.mean([tc_bndlow_opt, tc_bndhigh_opt])
                        pf_init = pf_init
    
                        modelparameters_subset = [pf_init, tc_init]
                        modelparams, mb_mwea = run_objective(modelparameters_subset, observed_massbal,
                                                             kp_bnds=kp_bnds,
                                                             tbias_bnds=tbias_bnds)
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
                                                                 kp_bnds=kp_bnds,
                                                                 tbias_bnds=tbias_bnds,
                                                                 eps_opt=eps_opt_new, ftol_opt=ftol_opt_new)
                            pf_opt = modelparams[2]
                            tc_opt = modelparams[7]
                            if debug:
                                print('\nmb_mwea:', np.round(mb_mwea,2), 'obs_mb:', np.round(observed_massbal,2),
                                      '\nPF:', np.round(pf_opt,2), 'TC:', np.round(tc_opt,2))
                    else:
                        # Temperature change bounds
                        tbias_bnds = (tc_bndlow_opt, tc_bndhigh_opt)
                        kp_bnds = (kp_bndlow, kp_bndhigh)
                        ddfsnow_bnds = (ddfsnow_init, ddfsnow_init)
                        tc_init = np.mean([tc_bndlow_opt, tc_bndhigh_opt])
                        pf_init = pf_init
    
                        modelparameters_subset = [pf_init, modelparameters[3], modelparameters[4], tc_init]
                        modelparams, mb_mwea = run_objective(modelparameters_subset, observed_massbal,
                                                             kp_bnds=kp_bnds,
                                                             tbias_bnds=tbias_bnds,
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
                                                                 kp_bnds=kp_bnds,
                                                                 tbias_bnds=tbias_bnds,
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
                print('ds PF:', np.round(df['kp'].values[0],2),
                      'ds TC:', np.round(df['tbias'].values[0],2))

        # ==============================================================

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

    if pygem_prms.option_calibration == 'MCMC':
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
    dates_table_nospinup  = modelsetup.datesmodelrun(startyear=pygem_prms.ref_startyear, endyear=pygem_prms.ref_endyear,
                                                     spinupyears=0)
    dates_table = modelsetup.datesmodelrun(startyear=pygem_prms.ref_startyear, endyear=pygem_prms.ref_endyear,
                                           spinupyears=pygem_prms.ref_spinupyears)

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
#            tbias_bndlow = main_vars['tbias_bndlow']
#            tbias_bndhigh = main_vars['tbias_bndhigh']
#            tbias_mu = main_vars['tbias_mu']
#            tbias_sigma = main_vars['tbias_sigma']
#            tbias_start = main_vars['tbias_start']
#            t1_idx = main_vars['t1_idx']
#            t2_idx = main_vars['t2_idx']
#            t1 = main_vars['t1']
#            t2 = main_vars['t2']
#            iterations=pygem_prms.mcmc_sample_no
#            burn=pygem_prms.mcmc_burn_no
#            step=pygem_prms.mcmc_step
#            kp_bndlow = main_vars['kp_bndlow']
#            kp_bndhigh = main_vars['kp_bndhigh']
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