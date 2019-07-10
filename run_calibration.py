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
import pandas as pd
import numpy as np
import xarray as xr
import pymc
from pymc import deterministic
from scipy.optimize import minimize
import pickle
# Local libraries
import pygem_input as input
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
    spc_region (optional) : str
        RGI region number for supercomputer 
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
    parser.add_argument('-ref_gcm_name', action='store', type=str, default=input.ref_gcm_name,
                        help='reference gcm name')
    parser.add_argument('-num_simultaneous_processes', action='store', type=int, default=4,
                        help='number of simultaneous processes (cores) to use')
    parser.add_argument('-option_parallels', action='store', type=int, default=1,
                        help='Switch to use or not use parallels (1 - use parallels, 0 - do not)')
    parser.add_argument('-spc_region', action='store', type=int, default=None,
                        help='rgi region number for supercomputer')
    parser.add_argument('-rgi_glac_number_fn', action='store', type=str, default=None,
                        help='Filename containing list of rgi_glac_number, helpful for running batches on spc')
    parser.add_argument('-progress_bar', action='store', type=int, default=0,
                        help='Boolean for the progress bar to turn it on or off (default 0 is off)')
    parser.add_argument('-debug', action='store', type=int, default=0,
                        help='Boolean for debugging to turn it on or off (default 0 is off)')
    parser.add_argument('-rgi_glac_number', action='store', type=str, default=None,
                        help='rgi glacier number for supercomputer')
    return parser


def retrieve_prior_parameters(modelparameters, glacier_rgi_table, glacier_area_t0, icethickness_t0, width_t0, elev_bins, 
                              glacier_gcm_temp, glacier_gcm_prec, glacier_gcm_elev, glacier_gcm_lrgcm, 
                              glacier_gcm_lrglac, dates_table, t1_idx, t2_idx, t1, t2, observed_massbal, mb_obs_min,
                              mb_obs_max, debug=False): 
    """
    Calculate parameters for prior distributions for the MCMC analysis
    
    Parameters
    ----------
    modelparameters : np.array
        glacier model parameters
    glacier_rgi_table : pd.DataFrame
        table of RGI information for a particular glacier
    glacier_area_t0, icethickness_t0, width_t0, elev_bins : np.arrays
        relevant glacier properties data
    glacier_gcm_temp, glacier_gcm_prec, glacier_gcm_elev, glacier_gcm_lrgcm, glacier_gcm_lrglac : np.arrays
        relevant glacier climate data
    dates_table : pd.DataFrame
        table of date/time information
    observed_massbal, mb_obs_min, mb_obs_max, t1, t2, t1_idx, t2_idx: floats (all except _idx) and integers (_idx)
        values related to the mass balance observations and their proper date/time indices
    debug : boolean
        switch to debug the function or not (default=False)
    
    Returns
    -------
    precfactor_boundlow, precfactor_boundhigh, precfactor_mu, precfactor_start : floats
        data for precipitation factor's prior distribution
    tempchange_boundlow, tempchange_boundhigh, tempchange_mu, tempchange_sigma, tempchange_start : floats
        data for temperature bias' prior distribution
    tempchange_max_loss, tempchange_max_acc, mb_max_loss, mb_max_acc : floats
        temperature change and mass balance associated with maximum accumulation and maximum loss
    precfactor_opt1, tempchange_opt1 : floats
        initial optimized values of precipitation factor and temperature change
    """
    def mb_mwea_calc(modelparameters, option_areaconstant=1):
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
        
    if debug:
        print('mb_max_loss:', np.round(mb_max_loss,2), 
              'TC_max_loss_AreaEvolve:', np.round(tempchange_max_loss,2),
              '\nmb_AreaConstant:', np.round(mb_tc_boundhigh,2), 
              'TC_boundhigh:', np.round(tempchange_boundhigh,2), 
              '\nmb_obs_min:', np.round(mb_obs_min,2))
    
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
    
    if debug: 
        mb_tc_boundlow = mb_mwea_calc(modelparameters, option_areaconstant=1)
        print('\nmb_max_acc:', np.round(mb_max_acc,2), 'TC_max_acc:', np.round(tempchange_max_acc,2),
              '\nmb_TC_boundlow_PF1:', np.round(mb_tc_boundlow,2), 
              'TC_boundlow:', np.round(tempchange_boundlow,2),
              '\nmb_obs_max:', np.round(mb_obs_max,2)
              )
        
    # ----- OTHER PARAMETERS -----
    # Assign TC_sigma
    tempchange_sigma = input.tempchange_sigma
    if (tempchange_boundhigh - tempchange_boundlow) / 6 < tempchange_sigma:
        tempchange_sigma = (tempchange_boundhigh - tempchange_boundlow) / 6
    
    tempchange_init = 0
    if tempchange_boundlow > 0:
        tempchange_init = tempchange_boundlow 
    elif tempchange_boundhigh < 0:
        tempchange_init = tempchange_boundhigh
        
    # OPTIMAL PRECIPITATION FACTOR (TC = 0 or TC_boundlow)
    # Find optimized tempchange in agreement with observed mass balance
    tempchange_4opt = tempchange_init
    precfactor_opt_init = [1]
    precfactor_opt_bnds = (0, 10)
    precfactor_opt_all = minimize(find_precfactor_opt, precfactor_opt_init, args=(tempchange_4opt), 
                                  bounds=[precfactor_opt_bnds], method='L-BFGS-B')
    precfactor_opt = precfactor_opt_all.x[0]
    
    try:
        precfactor_opt1 = precfactor_opt[0]
    except:
        precfactor_opt1 = precfactor_opt
        
    if debug:    
        print('tempchange_4opt:', tempchange_init)
        print('\nprecfactor_opt:', precfactor_opt)
    
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
        
    try:
        tempchange_opt1 = tempchange_opt[0]
    except:
        tempchange_opt1 = tempchange_opt
        
    if debug:
        print('\nprecfactor_opt (adjusted):', precfactor_opt)
        print('tempchange_opt:', tempchange_opt)

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
        
    if debug:
        print('tempchange_adj_4min:', tempchange_adj)
        
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
    
    if debug:
        print('tempchange_adj_4max:', tempchange_adj)

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
            tempchange_boundhigh, tempchange_mu, tempchange_sigma, tempchange_start, tempchange_max_loss, 
            tempchange_max_acc, mb_max_loss, mb_max_acc, precfactor_opt1, tempchange_opt1)
    

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
    gcm_prec = list_packed_vars[7]
    gcm_elev = list_packed_vars[8]
    gcm_lr = list_packed_vars[9]
    cal_data = list_packed_vars[10]
    
    time_start = time.time()
    parser = getparser()
    args = parser.parse_args()
    
    if args.debug == 1:
        debug = True
    else:
        debug = False
        
    # RGI region
    if args.spc_region is not None:
        rgi_regionsO1 = [int(args.spc_region)]
    else:
        rgi_regionsO1 = input.rgi_regionsO1

    # ===== CALIBRATION =====
    # Option 2: use MCMC method to determine posterior probability distributions of the three parameters tempchange,
    #           ddfsnow and precfactor. Then create an ensemble of parameter sets evenly sampled from these 
    #           distributions, and output these sets of parameters and their corresponding mass balances to be used in 
    #           the simulations.
    if input.option_calibration == 2:

        # ===== Define functions needed for MCMC method =====        
        def run_MCMC(precfactor_disttype=input.precfactor_disttype,
                     precfactor_lognorm_mu=input.precfactor_lognorm_mu, 
                     precfactor_lognorm_tau=input.precfactor_lognorm_tau, 
                     precfactor_mu=input.precfactor_mu, precfactor_sigma=input.precfactor_sigma,
                     precfactor_boundlow=input.precfactor_boundlow, precfactor_boundhigh=input.precfactor_boundhigh, 
                     precfactor_start=input.precfactor_start,
                     tempchange_disttype=input.tempchange_disttype,
                     tempchange_mu=input.tempchange_mu, tempchange_sigma=input.tempchange_sigma, 
                     tempchange_boundlow=input.tempchange_boundlow, tempchange_boundhigh=input.tempchange_boundhigh,
                     tempchange_start=input.tempchange_start,
                     ddfsnow_disttype=input.ddfsnow_disttype,
                     ddfsnow_mu=input.ddfsnow_mu, ddfsnow_sigma=input.ddfsnow_sigma, 
                     ddfsnow_boundlow=input.ddfsnow_boundlow, ddfsnow_boundhigh=input.ddfsnow_boundhigh,
                     ddfsnow_start=input.ddfsnow_start,
                     iterations=10, burn=0, thin=input.thin_interval, tune_interval=1000, step=None, 
                     tune_throughout=True, save_interval=None, burn_till_tuned=False, stop_tuning_after=5, verbose=0, 
                     progress_bar=args.progress_bar, dbname=None):
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
            # PRIOR DISTRIBUTIONS
            # Precipitation factor [-]
            if precfactor_disttype =='lognormal':
                #  lognormal distribution (roughly 0.3 to 3)
                precfactor_start = np.exp(precfactor_start)
                precfactor = pymc.Lognormal('precfactor', mu=precfactor_lognorm_mu, tau=precfactor_lognorm_tau,
                                            value=precfactor_start)
            elif precfactor_disttype == 'uniform':
                precfactor = pymc.Uniform('precfactor', lower=precfactor_boundlow, upper=precfactor_boundhigh, 
                                          value=precfactor_start)
            # Temperature change [degC]            
            if tempchange_disttype =='truncnormal':
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
                    modelparameters_copy[5] = modelparameters_copy[4] / input.ddfsnow_iceratio
                # Mass balance calculations
                (glac_bin_temp, glac_bin_prec, glac_bin_acc, glac_bin_refreeze, glac_bin_snowpack, glac_bin_melt,
                 glac_bin_frontalablation, glac_bin_massbalclim, glac_bin_massbalclim_annual, glac_bin_area_annual,
                 glac_bin_icethickness_annual, glac_bin_width_annual, glac_bin_surfacetype_annual,
                 glac_wide_massbaltotal, glac_wide_runoff, glac_wide_snowline, glac_wide_snowpack,
                 glac_wide_area_annual, glac_wide_volume_annual, glac_wide_ELA_annual, offglac_wide_prec, 
                 offglac_wide_refreeze, offglac_wide_melt, offglac_wide_snowpack, offglac_wide_runoff) = (
                    massbalance.runmassbalance(modelparameters_copy, glacier_rgi_table, glacier_area_t0, 
                                               icethickness_t0, width_t0, elev_bins, glacier_gcm_temp, glacier_gcm_prec,
                                               glacier_gcm_elev, glacier_gcm_lrgcm, glacier_gcm_lrglac, dates_table,
                                               option_areaconstant=1))
                # Compute glacier volume change for every time step and use this to compute mass balance
                glac_wide_area = glac_wide_area_annual[:-1].repeat(12)
                # Mass change [km3 mwe]
                #  mb [mwea] * (1 km / 1000 m) * area [km2]
                glac_wide_masschange = glac_wide_massbaltotal / 1000 * glac_wide_area
                # Mean annual mass balance [mwea]
                mb_mwea = glac_wide_masschange[t1_idx:t2_idx+1].sum() / glac_wide_area[0] * 1000 / (t2 - t1)

                return mb_mwea
            
            
            # Observed distribution
            #  Observed data defines the observed likelihood of mass balances (based on geodetic observations)
            obs_massbal = pymc.Normal('obs_massbal', mu=massbal, tau=(1/(observed_error**2)), 
                                      value=float(observed_massbal), observed=True)
            
            if debug:
                print('obs_massbal:', observed_massbal, 'obs_error:', observed_error)
            
            # Set model
            if dbname is None:
                model = pymc.MCMC({'precfactor':precfactor, 'tempchange':tempchange, 'ddfsnow':ddfsnow, 
                                   'massbal':massbal, 'obs_massbal':obs_massbal})
            else:
                model = pymc.MCMC({'precfactor':precfactor, 'tempchange':tempchange, 'ddfsnow':ddfsnow, 
                                   'massbal':massbal, 'obs_massbal':obs_massbal}, db='pickle', dbname=dbname)
    
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

            if debug:
                print(count, main_glac_rgi.loc[main_glac_rgi.index.values[glac],'RGIId_float'])

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
            mb_obs_max = observed_massbal + 3 * observed_error
            mb_obs_min = observed_massbal - 3 * observed_error

            if debug:
                print('observed_massbal:',observed_massbal, 'observed_error:',observed_error)
            
            # ===== RUN MARKOV CHAIN MONTE CARLO METHOD ====================            
            # OLD SETUP
            tempchange_mu = input.tempchange_mu
            tempchange_sigma = input.tempchange_sigma
            tempchange_boundlow = input.tempchange_boundlow
            tempchange_boundhigh = input.tempchange_boundhigh
            tempchange_start = tempchange_mu
            precfactor_boundlow = input.precfactor_boundlow
            precfactor_boundhigh = input.precfactor_boundhigh
            precfactor_mu = (precfactor_boundlow + precfactor_boundhigh) / 2
            precfactor_start = precfactor_mu
            ddfsnow_boundlow = input.ddfsnow_boundlow
            ddfsnow_boundhigh=input.ddfsnow_boundhigh
            ddfsnow_mu = input.ddfsnow_mu
            ddfsnow_sigma = input.ddfsnow_sigma
            
            # NEW SETUP
            if icethickness_t0.max() > 0:             
                (precfactor_boundlow, precfactor_boundhigh, precfactor_mu, precfactor_start, tempchange_boundlow, 
                 tempchange_boundhigh, tempchange_mu, tempchange_sigma, tempchange_start, tempchange_max_loss, 
                 tempchange_max_acc, mb_max_loss, mb_max_acc, precfactor_opt_init, tempchange_opt_init) = (
                         retrieve_prior_parameters(modelparameters, glacier_rgi_table, glacier_area_t0, icethickness_t0, 
                                                   width_t0, elev_bins, glacier_gcm_temp, glacier_gcm_prec, 
                                                   glacier_gcm_elev, glacier_gcm_lrgcm, glacier_gcm_lrglac, dates_table, 
                                                   t1_idx, t2_idx, t1, t2, observed_massbal, mb_obs_min, mb_obs_max))
        
                print('\nParameters:\nPF_low:', np.round(precfactor_boundlow,2), 'PF_high:', 
                      np.round(precfactor_boundhigh,2), '\nTC_low:', np.round(tempchange_boundlow,2), 
                      'TC_high:', np.round(tempchange_boundhigh,2),
                      '\nTC_mu:', np.round(tempchange_mu,2), 'TC_sigma:', np.round(tempchange_sigma,2))

            
            # fit the MCMC model
            for n_chain in range(0,input.n_chains):
                print(glacier_str + ' chain' + str(n_chain))
                print('\nProcessing time prior to chain:',time.time()-time_start, 's')
                if n_chain == 0:                    
                    model = run_MCMC(iterations=input.mcmc_sample_no, burn=input.mcmc_burn_no, step=input.mcmc_step, 
                                     precfactor_boundlow=precfactor_boundlow, precfactor_boundhigh=precfactor_boundhigh,
                                     precfactor_start=precfactor_start,
                                     tempchange_mu=tempchange_mu, tempchange_sigma=tempchange_sigma, 
                                     tempchange_boundlow=tempchange_boundlow, tempchange_boundhigh=tempchange_boundhigh,
                                     tempchange_start=tempchange_start)   
                elif n_chain == 1:
                    # Chains start at lowest values
                    model = run_MCMC(iterations=input.mcmc_sample_no, burn=input.mcmc_burn_no, step=input.mcmc_step,
                                     precfactor_boundlow=precfactor_boundlow, precfactor_boundhigh=precfactor_boundhigh,
                                     tempchange_mu=tempchange_mu, tempchange_sigma=tempchange_sigma, 
                                     tempchange_boundlow=tempchange_boundlow, tempchange_boundhigh=tempchange_boundhigh,
                                     tempchange_start=tempchange_boundlow, 
                                     precfactor_start=precfactor_boundlow, 
                                     ddfsnow_start=ddfsnow_boundlow)
                elif n_chain == 2:
                    # Chains start at highest values
                    model = run_MCMC(iterations=input.mcmc_sample_no, burn=input.mcmc_burn_no, step=input.mcmc_step,
                                     precfactor_boundlow=precfactor_boundlow, precfactor_boundhigh=precfactor_boundhigh,
                                     tempchange_mu=tempchange_mu, tempchange_sigma=tempchange_sigma, 
                                     tempchange_boundlow=tempchange_boundlow, tempchange_boundhigh=tempchange_boundhigh,
                                     tempchange_start=tempchange_boundhigh, 
                                     precfactor_start=precfactor_boundhigh, 
                                     ddfsnow_start=ddfsnow_boundhigh)
                    
                if debug:
                    print('acceptance ratio:', model.step_method_dict[next(iter(model.stochastics))][0].ratio)
                   
                # Select data from model to be stored in netcdf
                df = pd.DataFrame({'tempchange': model.trace('tempchange')[:],
                                   'precfactor': model.trace('precfactor')[:],
                                   'ddfsnow': model.trace('ddfsnow')[:],
                                   'massbal': model.trace('massbal')[:]})
                # set columns for other variables
                df['ddfice'] = df['ddfsnow'] / input.ddfsnow_iceratio
                df['lrgcm'] = np.full(df.shape[0], input.lrgcm)
                df['lrglac'] = np.full(df.shape[0], input.lrglac)
                df['precgrad'] = np.full(df.shape[0], input.precgrad)
                df['tempsnow'] = np.full(df.shape[0], input.tempsnow)
                
                if debug:
                    print('mb_mwea:', np.round(df.massbal.mean(),2), 'mb_mwea_std:', np.round(df.massbal.std(),2))
                
                if n_chain == 0:
                    df_chains = df.values[:, :, np.newaxis]
                else:
                    df_chains = np.dstack((df_chains, df.values))
                    
            # Record calibration data
            prior_cns = ['pf_bndlow', 'pf_bndhigh', 'pf_mu', 'tc_bndlow', 'tc_bndhigh', 'tc_mu', 'tc_std', 
                         'ddfsnow_bndlow', 'ddfsnow_bndhigh', 'ddfsnow_mu', 'ddfsnow_std', 'mb_max_loss', 'mb_max_acc', 
                         'tc_max_loss', 'tc_max_acc','pf_opt_init', 'tc_opt_init']
            prior_values = [precfactor_boundlow, precfactor_boundhigh, precfactor_mu, tempchange_boundlow, 
                            tempchange_boundhigh, tempchange_mu, tempchange_sigma, ddfsnow_boundlow, ddfsnow_boundhigh, 
                            ddfsnow_mu, ddfsnow_sigma, mb_max_loss, mb_max_acc, tempchange_max_loss, tempchange_max_acc,
                            precfactor_opt_init, tempchange_opt_init]
                    
            ds = xr.Dataset({'mp_value': (('iter', 'mp', 'chain'), df_chains),
                             'priors': (('prior_cns'), prior_values)
                             },
                            coords={'iter': df.index.values,
                                    'mp': df.columns.values,
                                    'chain': np.arange(0,n_chain+1),
                                    'prior_cns': prior_cns
                                    })
                
            if not os.path.exists(input.output_fp_cal):
                os.makedirs(input.output_fp_cal)
            ds.to_netcdf(input.output_fp_cal + glacier_str + '.nc')
            ds.close()
            
#            #%%
#            # Example of accessing netcdf file and putting it back into pandas dataframe
#            A = xr.open_dataset(input.mcmc_output_netcdf_fp + 'reg' + str(rgi_regionsO1[0]) + '/15.03734.nc')
#            B = pd.DataFrame(A['mp_value'].sel(chain=0).values, columns=A.mp.values)
#            priors = pd.Series(ds.priors, index=ds.prior_cns)
#            #%%

        # ==============================================================
    #%%
    # Huss and Hock (2015) model calibration steps
    elif input.option_calibration == 3:
        
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
                massbalance.runmassbalance(modelparameters, glacier_rgi_table, glacier_area_t0, icethickness_t0, 
                                           width_t0, elev_bins, glacier_gcm_temp, glacier_gcm_prec, 
                                           glacier_gcm_elev, glacier_gcm_lrgcm, glacier_gcm_lrglac, dates_table, 
                                           option_areaconstant=1))  
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
                massbalance.runmassbalance(modelparameters, glacier_rgi_table, glacier_area_t0, icethickness_t0, 
                                           width_t0, elev_bins, glacier_gcm_temp, glacier_gcm_prec, 
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
                modelparameters_opt = minimize(objective, modelparameters_init, method=input.method_opt,
                                               bounds=modelparameters_bnds, options={'ftol':input.ftol_opt})
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
                massbalance.runmassbalance(modelparams, glacier_rgi_table, glacier_area_t0, icethickness_t0,
                                           width_t0, elev_bins, glacier_gcm_temp, glacier_gcm_prec,
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
            df['lrgcm'] = np.full(df.shape[0], input.lrgcm)
            df['lrglac'] = np.full(df.shape[0], input.lrglac)
            df['precfactor'] = modelparameters[2]
            df['precgrad'] = np.full(df.shape[0], input.precgrad)
            df['ddfsnow'] = modelparameters[4]
            df['ddfice'] = df['ddfsnow'] / ddfsnow_iceratio
            df['tempsnow'] = np.full(df.shape[0], input.tempsnow)
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
            modelparameters = [input.lrgcm, input.lrglac, precfactor_init, input.precgrad, ddfsnow_init, input.ddfice,
                               input.tempsnow, tempchange_init]
            modelparameters[5] = modelparameters[4] / ddfsnow_iceratio

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
            lowest_bin = np.where(glacier_area_t0 > 0)[0][0]
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
            netcdf_output_fp = (input.output_fp_cal)
            if not os.path.exists(netcdf_output_fp):
                os.makedirs(netcdf_output_fp)
            write_netcdf_modelparams(netcdf_output_fp + glacier_str + '.nc', modelparameters, mb_mwea, observed_massbal)
        # ==============================================================
        
    #%%
    # Option 1: mimize mass balance difference using multi-step approach to expand solution space
    elif input.option_calibration == 1:

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
             glac_wide_area_annual, glac_wide_volume_annual, glac_wide_ELA_annual, offglac_wide_prec, 
             offglac_wide_refreeze, offglac_wide_melt, offglac_wide_snowpack, offglac_wide_runoff) = (
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
                    #  sum(mass balance * area) / total area
                    t1_idx = glacier_cal_data.loc[cal_idx, 't1_idx'].astype(int)
                    t2_idx = glacier_cal_data.loc[cal_idx, 't2_idx'].astype(int)
                    z1_idx = glacier_cal_data.loc[cal_idx, 'z1_idx'].astype(int)
                    z2_idx = glacier_cal_data.loc[cal_idx, 'z2_idx'].astype(int)
                    year_idx = int(t1_idx / 12)
                    bin_area_subset = glac_bin_area_annual[z1_idx:z2_idx+1, year_idx]
                    glacier_cal_compare.loc[cal_idx, 'model'] = (
                            (glac_bin_massbalclim[z1_idx:z2_idx+1, t1_idx:t2_idx] * 
                             bin_area_subset[:,np.newaxis]).sum() / bin_area_subset.sum())
                    # Fractional glacier area used to weight z-score
                    glacier_area_total = glac_bin_area_annual[:, year_idx].sum()
                    glacier_cal_compare.loc[cal_idx, 'area_frac'] = bin_area_subset.sum() / glacier_area_total
                    # Z-score for modeled mass balance based on observed mass balance and uncertainty
                    #  z-score = (model - measured) / uncertainty
                    glacier_cal_compare.loc[cal_idx, 'uncertainty'] = (input.massbal_uncertainty_mwea * 
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
                modelparameters_opt = minimize(objective, modelparameters_init, method=input.method_opt,
                                               bounds=modelparameters_bnds, options={'ftol':input.ftol_opt})
                # Record the optimized parameters
                modelparameters_subset = modelparameters_opt.x
            else:
                modelparameters_subset = modelparameters_init.copy()
            modelparams = (
                    [modelparameters[0], modelparameters[1], modelparameters_subset[0], modelparameters_subset[1], 
                     modelparameters_subset[2], modelparameters_subset[2] / input.ddfsnow_iceratio, modelparameters[6], 
                     modelparameters_subset[3]])
            # Re-run the optimized parameters in order to see the mass balance
            # Mass balance calculations
            (glac_bin_temp, glac_bin_prec, glac_bin_acc, glac_bin_refreeze, glac_bin_snowpack, glac_bin_melt,
             glac_bin_frontalablation, glac_bin_massbalclim, glac_bin_massbalclim_annual, glac_bin_area_annual,
             glac_bin_icethickness_annual, glac_bin_width_annual, glac_bin_surfacetype_annual,
             glac_wide_massbaltotal, glac_wide_runoff, glac_wide_snowline, glac_wide_snowpack,
             glac_wide_area_annual, glac_wide_volume_annual, glac_wide_ELA_annual, offglac_wide_prec, 
             offglac_wide_refreeze, offglac_wide_melt, offglac_wide_snowpack, offglac_wide_runoff) = (
                massbalance.runmassbalance(modelparams, glacier_rgi_table, glacier_area_t0, icethickness_t0,
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
                    bin_area_subset = glac_bin_area_annual[z1_idx:z2_idx+1, year_idx]
                    # Fractional glacier area used to weight z-score
                    glacier_area_total = glac_bin_area_annual[:, year_idx].sum()
                    glacier_cal_compare.loc[cal_idx, 'area_frac'] = bin_area_subset.sum() / glacier_area_total
                    glacier_cal_compare.loc[cal_idx, 'model'] = (
                            (glac_bin_massbalclim[z1_idx:z2_idx+1, t1_idx:t2_idx] *
                             bin_area_subset[:,np.newaxis]).sum() / bin_area_subset.sum())
                    # Z-score for modeled mass balance based on observed mass balance and uncertainty
                    #  z-score = (model - measured) / uncertainty
                    glacier_cal_compare.loc[cal_idx, 'uncertainty'] = (input.massbal_uncertainty_mwea *
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
                     glac_wide_area_annual, glac_wide_volume_annual, glac_wide_ELA_annual, offglac_wide_prec, 
                     offglac_wide_refreeze, offglac_wide_melt, offglac_wide_snowpack, offglac_wide_runoff) = (
                        massbalance.runmassbalance(modelparameters, glacier_rgi_table, glacier_area_t0, icethickness_t0, 
                                                   width_t0, elev_bins, glacier_gcm_temp, glacier_gcm_prec, 
                                                   glacier_gcm_elev, glacier_gcm_lrgcm, glacier_gcm_lrglac, dates_table, 
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
                modelparameters_opt = minimize(objective_group, modelparameters_init, method=input.method_opt, 
                                               bounds=modelparameters_bnds, options={'ftol':input.ftol_opt})
                # Record the optimized parameters
                modelparameters_subset = modelparameters_opt.x
            else:
                modelparameters_subset = modelparameters_init.copy()
            modelparameters_group = (
                    [input.lrgcm, input.lrglac, modelparameters_subset[0], modelparameters_subset[1], 
                     modelparameters_subset[2], modelparameters_subset[2] / input.ddfsnow_iceratio, input.tempsnow, 
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
                     glac_wide_area_annual, glac_wide_volume_annual, glac_wide_ELA_annual, offglac_wide_prec, 
                     offglac_wide_refreeze, offglac_wide_melt, offglac_wide_snowpack, offglac_wide_runoff) = (
                        massbalance.runmassbalance(modelparameters, glacier_rgi_table, glacier_area_t0, icethickness_t0, 
                                                   width_t0, elev_bins, glacier_gcm_temp, glacier_gcm_prec, 
                                                   glacier_gcm_elev, glacier_gcm_lrgcm, glacier_gcm_lrglac, dates_table, 
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
                zscore_tolerance = input.zscore_tolerance_single
            # else if multiple calibration points and one is a geodetic MB, check that geodetic MB is within 1
            elif (glacier_cal_compare.obs_type.isin(['mb_geo']).any() == True) and (glacier_cal_compare.shape[0] > 1):
                zscore4comparison = glacier_cal_compare.loc[glacier_cal_compare.index.values[np.where(
                        glacier_cal_compare['obs_type'] == 'mb_geo')[0][0]], 'zscore']
                zscore_tolerance = input.zscore_tolerance_all
            # otherwise, check mean zscore
            else:
                zscore4comparison = abs(glacier_cal_compare['zscore']).sum() / glacier_cal_compare.shape[0]
                zscore_tolerance = input.zscore_tolerance_all
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
                df['lrgcm'] = np.full(df.shape[0], input.lrgcm)
                df['lrglac'] = np.full(df.shape[0], input.lrglac)
                df['precfactor'] = modelparameters[2]
                df['precgrad'] = np.full(df.shape[0], input.precgrad)
                df['ddfsnow'] = modelparameters[4]
                df['ddfice'] = df['ddfsnow'] / input.ddfsnow_iceratio
                df['tempsnow'] = np.full(df.shape[0], input.tempsnow)
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
        main_glac_modelparamsopt = np.zeros((main_glac_rgi.shape[0], len(input.modelparams_colnames)))
        # Glacier-wide climatic mass balance (required for transfer fucntions)
        main_glacwide_mbclim_mwe = np.zeros((main_glac_rgi.shape[0], 1))
 
        # Loop through glaciers that have unique cal_data
        cal_individual_glacno = np.unique(cal_data.loc[cal_data['glacno'].notnull(), 'glacno'])
        for n in range(cal_individual_glacno.shape[0]):
            glac = np.where(main_glac_rgi[input.rgi_O1Id_colname].isin([cal_individual_glacno[n]]) == True)[0][0]
            if debug:
                print(count, ':', main_glac_rgi.loc[main_glac_rgi.index.values[glac], 'RGIId'])    
            elif glac%100 == 0:
                print(count, ':', main_glac_rgi.loc[main_glac_rgi.index.values[glac], 'RGIId'])
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
            precfactor_bnds_list = input.precfactor_bnds_list_init.copy()
            tempchange_bnds_list = input.tempchange_bnds_list_init.copy()
            ddfsnow_bnds_list = input.ddfsnow_bnds_list_init.copy()
            precgrad_bnds_list = input.precgrad_bnds_list_init.copy()
            manipulate_bnds = False
            
            modelparameters_init = [input.precfactor, input.precgrad, input.ddfsnow, input.tempchange]
            modelparameters, glacier_cal_compare = (
                    run_objective(modelparameters_init, glacier_cal_data, run_opt=False))
            zscore_sum = glacier_cal_compare.zscore_weighted.sum()            
            precfactor_bnds_list = [input.precfactor_bnds_list_init[0]]
            tempchange_bnds_list = [input.tempchange_bnds_list_init[0]]
            ddfsnow_bnds_list = [input.ddfsnow_bnds_list_init[0]]
            
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
                precfactor_bnds_list_modified = [(1,i[1]) for i in input.precfactor_bnds_list_init[1:]]
                tempchange_bnds_list_modified = [(i[0],0) for i in input.tempchange_bnds_list_init[1:]]
                ddfsnow_bnds_list_modified = [(i[0],0.0041) for i in input.ddfsnow_bnds_list_init[1:]]
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
                precfactor_bnds_list_modified = [(i[0],1) for i in input.precfactor_bnds_list_init[1:]]
                tempchange_bnds_list_modified = [(0,i[1]) for i in input.tempchange_bnds_list_init[1:]]
                ddfsnow_bnds_list_modified = [(0.0041,i[1]) for i in input.ddfsnow_bnds_list_init[1:]]
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
                    modelparameters_init = [input.precfactor, input.precgrad, input.ddfsnow, input.tempchange]
                elif manipulate_bnds:
                    modelparameters_init = (
                                [init_guess_frombounds(precfactor_bnds_list, calround, precfactor_init_idx),
                                 init_guess_frombounds(precgrad_bnds_list, calround, precgrad_init_idx),
                                 init_guess_frombounds(ddfsnow_bnds_list, calround, ddfsnow_init_idx),
                                 init_guess_frombounds(tempchange_bnds_list, calround, tempchange_init_idx)])
                else:
                    modelparameters_init = (
                            [modelparameters[2], modelparameters[3], modelparameters[4], modelparameters[7]])
                # Run optimization
                modelparameters, glacier_cal_compare = (
                        run_objective(modelparameters_init, glacier_cal_data, precfactor_bnds, tempchange_bnds, 
                                      ddfsnow_bnds, precgrad_bnds))                
                calround += 1
                
                # Update model parameters if significantly better
                mean_zscore = abs(glacier_cal_compare['zscore_weighted']).sum() / glacier_cal_compare.shape[0]
                if calround == 1:
                    mean_zscore_best = mean_zscore
                    modelparameters_best = modelparameters
                    glacier_cal_compare_best = glacier_cal_compare
                    cal_round_best = calround
                elif (mean_zscore_best - mean_zscore) > input.zscore_update_threshold:
                    mean_zscore_best = mean_zscore
                    modelparameters_best = modelparameters
                    glacier_cal_compare_best = glacier_cal_compare
                    cal_round_best = calround
            
                # Break loop if gone through all iterations
                if (calround == len(input.precfactor_bnds_list_init)) or zscore_compare(glacier_cal_compare, cal_idx):
                    continue_loop = False
                    
                if debug:
                    print('\nCalibration round:', calround,
                          '\nInitial parameters:\nPrecfactor:', modelparameters_init[0], 
                          '\nTempbias:', modelparameters_init[3], '\nDDFsnow:', modelparameters_init[2])
                    print('\nCalibrated parameters:\nPrecfactor:', modelparameters[2], 
                          '\nTempbias:', modelparameters[7], '\nDDFsnow:', modelparameters[4])
                    print('\nmean Zscore:', mean_zscore, '\n')
                          
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
                massbalance.runmassbalance(modelparameters, glacier_rgi_table, glacier_area_t0, icethickness_t0,
                                           width_t0, elev_bins, glacier_gcm_temp, glacier_gcm_prec,
                                           glacier_gcm_elev, glacier_gcm_lrgcm, glacier_gcm_lrglac, dates_table,
                                           option_areaconstant=1))
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
            netcdf_output_fp = (input.output_fp_cal + 'reg' + str(rgi_regionsO1[0]) + '/')
            if not os.path.exists(netcdf_output_fp):
                os.makedirs(netcdf_output_fp)
            # Loop through glaciers calibrated from group and export to netcdf
            glacier_str = '{0:0.5f}'.format(main_glac_rgi.loc[main_glac_rgi.index.values[glac], 'RGIId_float'])
            write_netcdf_modelparams(netcdf_output_fp + glacier_str + '.nc', modelparameters, glacier_cal_compare)
            
        # ==============================================================
        # ===== GROUP CALIBRATION =====
        if set(['group']).issubset(input.cal_datasets) == True:
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
                precfactor_bnds_list = input.precfactor_bnds_list_init.copy()
                tempchange_bnds_list = input.tempchange_bnds_list_init.copy()
                ddfsnow_bnds_list = input.ddfsnow_bnds_list_init.copy()
                precgrad_bnds_list = input.precgrad_bnds_list_init.copy()
                manipulate_bnds = False
                
                modelparameters_init = [input.precfactor, input.precgrad, input.ddfsnow, input.tempchange]
                modelparameters_group, glacier_cal_compare, glacwide_mbclim_mwe = (
                        run_objective_group(modelparameters_init, run_opt=False))
                zscore_sum = glacier_cal_compare.zscore.sum()
                
                precfactor_bnds_list = [input.precfactor_bnds_list_init[0]]
                tempchange_bnds_list = [input.tempchange_bnds_list_init[0]]
                ddfsnow_bnds_list = [input.ddfsnow_bnds_list_init[0]]
                
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
                    precfactor_bnds_list_modified = [(1,i[1]) for i in input.precfactor_bnds_list_init[1:]]
                    tempchange_bnds_list_modified = [(i[0],0) for i in input.tempchange_bnds_list_init[1:]]
                    ddfsnow_bnds_list_modified = [(i[0],0.0041) for i in input.ddfsnow_bnds_list_init[1:]]
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
                    precfactor_bnds_list_modified = [(i[0],1) for i in input.precfactor_bnds_list_init[1:]]
                    tempchange_bnds_list_modified = [(0,i[1]) for i in input.tempchange_bnds_list_init[1:]]
                    ddfsnow_bnds_list_modified = [(0.0041,i[1]) for i in input.ddfsnow_bnds_list_init[1:]]
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
                        modelparameters_init = [input.precfactor, input.precgrad, input.ddfsnow, input.tempchange]
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
                    elif (zscore_abs_best - zscore_abs) > input.zscore_update_threshold:
                        zscore_abs_best = zscore_abs
                        modelparameters_group_best = modelparameters_group
                        glacier_cal_compare_best = glacier_cal_compare
                        cal_round_best = calround
                        
                    # Break loop if gone through all iterations
                    if ((calround == len(input.precfactor_bnds_list_init)) or 
                        (abs(glacier_cal_compare.zscore) < input.zscore_tolerance_single)):
                        continue_loop = False
                    
                    if debug:
                        print('\nCalibration round:', calround,
                              '\nInitial parameters:\nPrecfactor:', modelparameters_init[0], 
                              '\nTempbias:', modelparameters_init[3], '\nDDFsnow:', modelparameters_init[2])
                        print('\nCalibrated parameters:\nPrecfactor:', modelparameters_group[2], 
                              '\nTempbias:', modelparameters_group[7], '\nDDFsnow:', modelparameters_group[4])
                        print('\nZscore:', zscore_abs, '\n')

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
                netcdf_output_fp = (input.output_fp_cal + 'reg' + str(rgi_regionsO1[0]) + '/')
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
        main_glac_modelparamsopt_pd = pd.DataFrame(main_glac_modelparamsopt, columns=input.modelparams_colnames)
        main_glac_modelparamsopt_pd.index = main_glac_rgi.index.values
        main_glacwide_mbclim_pd = pd.DataFrame(main_glacwide_mbclim_mwe, columns=['mbclim_mwe'])
        main_glac_output = pd.concat([main_glac_output, main_glac_modelparamsopt_pd, main_glacwide_mbclim_pd], axis=1)
        
        # Export output
        if not os.path.exists(input.output_fp_cal + 'temp/'):
            os.makedirs(input.output_fp_cal + 'temp/')
        output_modelparams_fn = (
                'R' + str(rgi_regionsO1[0]) + '_modelparams_opt' + str(input.option_calibration) + '_' + gcm_name 
                + str(input.startyear) + str(input.endyear) + '--' + str(count) + '.csv')
        output_calcompare_fn = (
                'R' + str(rgi_regionsO1[0]) + '_calcompare_opt' + str(input.option_calibration) + '_' + gcm_name 
                + str(input.startyear) + str(input.endyear) + '--' + str(count) + '.csv')
        main_glac_output.to_csv(input.output_fp_cal + 'temp/' + output_modelparams_fn )
        main_glac_cal_compare.to_csv(input.output_fp_cal + 'temp/' + output_calcompare_fn)       
        
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
    
    if input.option_calibration == 2:
        print('Chains:', input.n_chains, 'Iterations:', input.mcmc_sample_no)
        
    # RGI region
    if args.spc_region is not None:
        rgi_regionsO1 = [int(args.spc_region)]
    else:
        rgi_regionsO1 = input.rgi_regionsO1
    
    # RGI glacier number
    if args.rgi_glac_number_fn is not None:
        with open(args.rgi_glac_number_fn, 'rb') as f:
            rgi_glac_number = pickle.load(f)
    elif args.rgi_glac_number is not None:
        rgi_glac_number = [args.rgi_glac_number]
    else:
        rgi_glac_number = input.rgi_glac_number    

    # Select all glaciers in a region
    main_glac_rgi_all = modelsetup.selectglaciersrgitable(rgi_regionsO1=rgi_regionsO1, rgi_regionsO2='all',
                                                          rgi_glac_number=rgi_glac_number)
    
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
        
        
    #%% =================================================
#    # Define chunk size for parallel processing
#    if args.option_parallels != 0:
#        num_cores = int(np.min([len(rgi_glac_number), args.num_simultaneous_processes]))
#        chunk_size = int(np.ceil(len(rgi_glac_number) / num_cores))
#    else:
#        # if not running in parallel, chunk size is all glaciers
#        num_cores = 1
#        chunk_size = len(rgi_glac_number)
#    
#    def split_list(lst, n=1):
#        """
#        Split list of glaciers into batches for the supercomputer.
#        
#        Parameters
#        ----------
#        lst : list
#            List that you want to split into separate batches
#        n : int
#            Number of batches to split glaciers into.
#        
#        Returns
#        -------
#        lst_batches : list
#            list of n lists that have sequential values in each list
#        """
#        # If batches is more than list, then there will be one glacier in each batch
#        if n > len(lst):
#            n = len(lst)
#        n_perlist_low = int(len(lst)/n)
#        n_perlist_high = int(np.ceil(len(lst)/n))
#        lst_copy = lst.copy()
#        count = 0
#        lst_batches = []
#        for x in np.arange(n):
#            count += 1
#            if count <= n_perlist_low:
#                lst_subset = lst_copy[0:n_perlist_high]
#                lst_batches.append(lst_subset)
#                [lst_copy.remove(i) for i in lst_subset]
#            else:
#                lst_subset = lst_copy[0:n_perlist_low]
#                lst_batches.append(lst_subset)
#                [lst_copy.remove(i) for i in lst_subset]
#        return lst_batches    
#    
#    rgi_glac_number_batches = split_list(rgi_glac_number, n=num_cores)
#    
#    list_packed_vars = []
#    n = 0
#    for batch in rgi_glac_number_batches:
#        n += 1
#        list_packed_vars.append([batch, gcm_name, n])
        
    
    #%%===================================================
    # Pack variables for multiprocessing
#    list_packed_vars = []
#    for chunk in range(0, main_glac_rgi_all.shape[0], chunk_size):
#        main_glac_rgi_raw = main_glac_rgi_all.loc[chunk:chunk+chunk_size-1].copy()
#        list_packed_vars.append([main_glac_rgi_raw, gcm_name])
        
    # ===== LOAD GLACIER DATA =====
    # Glacier hypsometry [km**2], total area
    main_glac_hyps_all = modelsetup.import_Husstable(main_glac_rgi_all, rgi_regionsO1, input.hyps_filepath,
                                                     input.hyps_filedict, input.hyps_colsdrop)

    # Ice thickness [m], average
    main_glac_icethickness_all = modelsetup.import_Husstable(main_glac_rgi_all, rgi_regionsO1, 
                                                             input.thickness_filepath, input.thickness_filedict, 
                                                             input.thickness_colsdrop)
    main_glac_hyps_all[main_glac_icethickness_all == 0] = 0
    # Width [km], average
    main_glac_width_all = modelsetup.import_Husstable(main_glac_rgi_all, rgi_regionsO1, input.width_filepath,
                                                      input.width_filedict, input.width_colsdrop)
    elev_bins = main_glac_hyps_all.columns.values.astype(int)
    # Add volume [km**3] and mean elevation [m a.s.l.]
    main_glac_rgi_all['Volume'], main_glac_rgi_all['Zmean'] = (
            modelsetup.hypsometrystats(main_glac_hyps_all, main_glac_icethickness_all))
    # Select dates including future projections
    #  - nospinup dates_table needed to get the proper time indices
    dates_table_nospinup  = modelsetup.datesmodelrun(startyear=input.startyear, endyear=input.endyear, 
                                                     spinupyears=0)
    dates_table = modelsetup.datesmodelrun(startyear=input.startyear, endyear=input.endyear, 
                                           spinupyears=input.spinupyears)

    # ===== LOAD CALIBRATION DATA =====
    cal_data = pd.DataFrame()
    for dataset in input.cal_datasets:
        cal_subset = class_mbdata.MBData(name=dataset, rgi_regionO1=rgi_regionsO1[0])
        cal_subset_data = cal_subset.retrieve_mb(main_glac_rgi_all, main_glac_hyps_all, dates_table_nospinup)
        cal_data = cal_data.append(cal_subset_data, ignore_index=True)
    cal_data = cal_data.sort_values(['glacno', 't1_idx'])
    cal_data.reset_index(drop=True, inplace=True)
    
    # If group data is included, then add group dictionary and add group name to main_glac_rgi
    if set(['group']).issubset(input.cal_datasets) == True:
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
        main_glac_rgi_all['group_name'] = main_glac_rgi_all['RGIId'].map(group_dict)
    else:
        main_glac_rgi_all['group_name'] = np.nan

    # Drop glaciers that do not have any calibration data (individual or group)    
    main_glac_rgi = ((main_glac_rgi_all.iloc[np.unique(
            np.append(main_glac_rgi_all[main_glac_rgi_all['group_name'].notnull() == True].index.values, 
                      np.where(main_glac_rgi_all[input.rgi_O1Id_colname].isin(cal_data['glacno']) == True)[0])), :])
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
    # Air temperature [degC], Precipitation [m], Elevation [masl], Lapse rate [K m-1]
    gcm_temp, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(gcm.temp_fn, gcm.temp_vn, main_glac_rgi, dates_table)
    gcm_prec, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(gcm.prec_fn, gcm.prec_vn, main_glac_rgi, dates_table)
    gcm_elev = gcm.importGCMfxnearestneighbor_xarray(gcm.elev_fn, gcm.elev_vn, main_glac_rgi)
    # Lapse rate [K m-1]
    if gcm_name == 'ERA-Interim':
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
            glac_lat = main_glac_rgi.loc[glac,input.rgi_lat_colname]
            glac_lon = main_glac_rgi.loc[glac,input.rgi_lon_colname]
            if (~(input.coawst_d02_lat_min <= glac_lat <= input.coawst_d02_lat_max) or 
                ~(input.coawst_d02_lon_min <= glac_lon <= input.coawst_d02_lon_max)):
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
#    if input.option_calibration == 1:
#        # Merge csv summary files into single file    
#        # Count glaciers
#        glac_count = 0
#        output_temp = input.output_fp_cal + 'temp/'
#        for i in os.listdir(output_temp):
#            if i.startswith(str(rgi_regionsO1[0])) and i.endswith('.nc'):
#                glac_count += 1
#        
#        # Model parameters - combine into single file
#        check_modelparams_str = (
#                'modelparams_opt' + str(input.option_calibration) + '_' + gcm_name + str(input.startyear) + 
#                str(input.endyear))
#        output_modelparams_all_fn = (
#                'R' + str(rgi_regionsO1[0]) + '_' + str(glac_count) + check_modelparams_str + '.csv')
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
#                output_all = pd.read_csv(input.output_fp_cal + 'temp/' + i, index_col=0)
#            else:
#                output_2join = pd.read_csv(input.output_fp_cal + 'temp/' + i, index_col=0)
#                output_all = output_all.append(output_2join, ignore_index=True)
#            # Remove file after its been merged
#            os.remove(input.output_fp_cal + 'temp/' + i)
#        # Export joined files
#        output_all.to_csv(input.output_fp_cal + output_modelparams_all_fn)
#        
#        # Calibration comparison - combine into single file
#        check_calcompare_str = (
#                'calcompare_opt' + str(input.option_calibration) + '_' + gcm_name + str(input.startyear) + 
#                str(input.endyear))
#        output_calcompare_all_fn = (
#                'R' + str(rgi_regionsO1[0]) + '_' + str(glac_count) + check_calcompare_str + '.csv')
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
#                output_all = pd.read_csv(input.output_fp_cal + 'temp/' + i, index_col=0)
#            else:
#                output_2join = pd.read_csv(input.output_fp_cal + 'temp/' + i, index_col=0)
#                output_all = output_all.append(output_2join, ignore_index=True)
#            # Remove file after its been merged
#            os.remove(input.output_fp_cal + 'temp/' + i)
#        # Export joined files
#        output_all.to_csv(input.output_fp_cal + output_calcompare_all_fn)
#
#    print('Total processing time:', time.time()-time_start, 's')

    #%% ===== PLOTTING AND PROCESSING FOR MODEL DEVELOPMENT =====
#    # Place local variables in variable explorer
#    if (args.option_parallels == 0):
#        main_vars_list = list(main_vars.keys())
##            gcm_name = main_vars['gcm_name']
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
#        glacier_area_t0 = main_vars['glacier_area_t0']
#        glacier_cal_data = main_vars['glacier_cal_data']
#        cal_idx = main_vars['cal_idx']
#        modelparameters = main_vars['modelparameters']
#        icethickness_t0 = main_vars['icethickness_t0']
#        width_t0 = main_vars['width_t0']
#        
#        if input.option_calibration == 2 and input.new_setup == 1:
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
#            iterations=input.mcmc_sample_no
#            burn=input.mcmc_burn_no
#            step=input.mcmc_step
#            precfactor_boundlow = main_vars['precfactor_boundlow']
#            precfactor_boundhigh = main_vars['precfactor_boundhigh']
#            glacier_gcm_prec = main_vars['glacier_gcm_prec']
#            glacier_gcm_temp = main_vars['glacier_gcm_temp']
#            glacier_gcm_elev = main_vars['glacier_gcm_elev']
#            glacier_rgi_table = main_vars['glacier_rgi_table']
#            glacier_gcm_lrgcm = main_vars['glacier_gcm_lrgcm']
#            glacier_gcm_lrglac = main_vars['glacier_gcm_lrglac']
#            
#        if input.option_calibration == 1:
#            glacier_cal_compare = main_vars['glacier_cal_compare']
#            main_glac_cal_compare = main_vars['main_glac_cal_compare']
#            main_glac_modelparamsopt = main_vars['main_glac_modelparamsopt']
#            main_glac_output = main_vars['main_glac_output']
#            main_glac_modelparamsopt_pd = main_vars['main_glac_modelparamsopt_pd']
#            main_glacwide_mbclim_mwe = main_vars['main_glacwide_mbclim_mwe']
#    #            glac_wide_massbaltotal = main_vars['glac_wide_massbaltotal']
#    #            glac_wide_area_annual = main_vars['glac_wide_area_annual']
#    #            glac_wide_volume_annual = main_vars['glac_wide_volume_annual']
#    #            glacier_rgi_table = main_vars['glacier_rgi_table']
#    #            main_glac_modelparamsopt = main_vars['main_glac_modelparamsopt']
#    #            main_glac_massbal_compare = main_vars['main_glac_massbal_compare']
#    #            main_glac_output = main_vars['main_glac_output']
#            if set(['group']).issubset(input.cal_datasets):
#                group_dict_keyslist = main_vars['group_dict_keyslist']
#                group_dict_keyslist_names = main_vars['group_dict_keyslist_names']
#                cal_data_idx_groups = main_vars['cal_data_idx_groups']
#                cal_data = main_vars['cal_data']
#                cal_individual_glacno = main_vars['cal_individual_glacno']