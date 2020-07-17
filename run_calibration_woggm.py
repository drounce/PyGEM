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
import spc_split_glaciers as split_glaciers
import class_climate
import class_mbdata

from oggm import cfg
from pygem.oggm_compat import single_flowline_glacier_directory
from pygem.massbalance import PyGEMMassBalance

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
    parser.add_argument('-option_ordered', action='store', type=int, default=1,
                        help='switch to keep lists ordered or not')
    parser.add_argument('-debug', action='store', type=int, default=0,
                        help='Boolean for debugging to turn it on or off (default 0 is off)')
    parser.add_argument('-rgi_glac_number', action='store', type=str, default=None,
                        help='rgi glacier number for supercomputer')
    return parser



def retrieve_priors(gdir, glacier_rgi_table, modelprms, fls, debug=False):
    """
    Calculate parameters for prior distributions for the MCMC analysis
    """
    # Maximum mass loss [mwea] (based on consensus ice thickness estimate)
    with open(gdir.get_filepath('mass_consensus'), 'rb') as f:
        mass_consensus = pickle.load(f)    
    mb_max_loss = -1 * mass_consensus / pygem_prms.density_water / gdir.rgi_area_m2 / (gdir.dates_table.shape[0] / 12)
    
    # ----- TEMPERATURE BIAS UPPER BOUND -----
    # Temperature where no melt
    tbias_max_acc = (-1 * (gdir.historical_climate['temp'] + gdir.historical_climate['lr'] *
                     (fls[0].surface_h.min() - gdir.historical_climate['elev'])).max())
    if debug:
        print('tbias_max_acc:', np.round(tbias_max_acc,2))
        
    # Looping forward and backward to ensure optimization does not get stuck
    modelprms['tbias'] = tbias_max_acc
    mb_mwea_1 = mb_mwea_calc(gdir, modelprms, glacier_rgi_table, fls=fls)
    # use absolute value because with area evolving the maximum value is a limit
    while mb_mwea_1 > mb_max_loss:
        modelprms['tbias'] = modelprms['tbias'] + 1
        mb_mwea_1 = mb_mwea_calc(gdir, modelprms, glacier_rgi_table, fls=fls)
        if debug:
            print('TC:', np.round(modelprms['tbias'],2), 'mb_mwea_1:', np.round(mb_mwea_1,2),
                  'mb_max_loss:', np.round(mb_max_loss,2))
    # Looping backward for tempchange at max loss
    while mb_mwea_1 < mb_max_loss:
        modelprms['tbias'] = modelprms['tbias'] - 0.05
        mb_mwea_1 = mb_mwea_calc(gdir, modelprms, glacier_rgi_table, fls=fls)
        if debug:
            print('tbias:', np.round(modelprms['tbias'],2), 'mb_mwea_1:', np.round(mb_mwea_1,2),
                  'mb_max_loss:', np.round(mb_max_loss,2))
    tbias_max_loss = modelprms['tbias']
    tbias_bndhigh = tbias_max_loss
    if debug:
        print('tbias_bndhigh:', np.round(tbias_bndhigh,2), 'mb_max_loss:', np.round(mb_max_loss,2))

    # ----- TEMPERATURE BIAS LOWER BOUND -----
    #  Since the mass balance ablation is conditional on the glacier evolution, there can be cases where higher 
    # temperature biases still have 0 for nbinyears_negmbclim. Hence, the need to loop beyond the first instance, and 
    # then go back and check that you're using the good cases from there onward. This ensures starting point is good.
    modelprms['tbias'] = tbias_max_acc
    nbinyears_negmbclim = mb_mwea_calc(gdir, modelprms, glacier_rgi_table, fls=fls, return_tbias_mustmelt=True)
    nbinyears_negmbclim_list = [nbinyears_negmbclim]
    tbias_negmbclim_list = [modelprms['tbias']]
    tbias_smallstep_switch = False
    while nbinyears_negmbclim < 10 and modelprms['tbias'] < tbias_bndhigh:
        # Switch from large to small step sizes to speed up calculations
        if tbias_smallstep_switch == False:
            tbias_stepsize = 1
        else:
            tbias_stepsize = 0.05
        tbias_old = modelprms['tbias']
        modelprms['tbias'] = modelprms['tbias'] + tbias_stepsize
        nbinyears_negmbclim = mb_mwea_calc(gdir, modelprms, glacier_rgi_table, fls=fls, return_tbias_mustmelt=True)
        # Record if using big step and no there is no melt or if using small step and there is melt
        if nbinyears_negmbclim == 0 or (nbinyears_negmbclim > 0 and tbias_smallstep_switch == True):
            nbinyears_negmbclim_list.append(nbinyears_negmbclim)
            tbias_negmbclim_list.append(modelprms['tbias'])
        # First time nbinyears_negmbclim is > 0, flip switch to use smalll step and restart with last tbias
        if nbinyears_negmbclim > 0 and tbias_smallstep_switch == False:
            tbias_smallstep_switch = True
            modelprms['tbias'] = tbias_old
            nbinyears_negmbclim = 0
        if debug:
            print('tbias:', np.round(modelprms['tbias'],2), 'nbinyears_negmbclim:', nbinyears_negmbclim)
    tbias_bndlow = tbias_negmbclim_list[np.where(np.array(nbinyears_negmbclim_list) == 0)[0][-1] + 1]
    if debug:
        print('tbias_bndlow:', np.round(tbias_bndlow,2))
        
    return tbias_bndlow, tbias_bndhigh, mb_max_loss



#%%
print('NEED TO ADD INDICES HERE consistent with mbdata!!!!')
def mb_mwea_calc(gdir, modelprms, glacier_rgi_table, fls=None, t1=None, t2=None,
                 option_areaconstant=1, return_tbias_mustmelt=False, 
#                 return_volremaining=False
                 ):
    """
    Run the mass balance and calculate the mass balance [mwea]

    Parameters
    ----------
    option_areaconstant : Boolean

    Returns
    -------
    mb_mwea : float
        mass balance [m w.e. a-1]
    """
    # RUN MASS BALANCE MODEL
    mbmod = PyGEMMassBalance(gdir, modelprms, glacier_rgi_table, fls=fls, option_areaconstant=True,
                             debug=pygem_prms.debug_mb, debug_refreeze=pygem_prms.debug_refreeze)
    years = np.arange(0, int(gdir.dates_table.shape[0]/12))
    for fl_id, fl in enumerate(fls):
        for year in years:
            mbmod.get_annual_mb(fls[0].surface_h, fls=fls, fl_id=fl_id, year=year, debug=False)
    
    # Option to return must melt condition
    if return_tbias_mustmelt:
        # Number of years and bins with negative climatic mass balance
        nbinyears_negmbclim =  len(np.where(mbmod.glac_bin_massbalclim_annual < 0)[0])
        return nbinyears_negmbclim
#    elif return_volremaining:
#        assert True==False, 'Error: need to code return_volremaining option'
#        # Ensure volume by end of century is zero
#        # Compute glacier volume change for every time step and use this to compute mass balance
#        glac_wide_area = glac_wide_area_annual[:-1].repeat(12)
#        # Mass change [km3 mwe]
#        #  mb [mwea] * (1 km / 1000 m) * area [km2]
#        glac_wide_masschange = glac_wide_massbaltotal / 1000 * glac_wide_area
#        # Mean annual mass balance [mwea]
#        mb_mwea = glac_wide_masschange[t1_idx:t2_idx+1].sum() / glac_wide_area[0] * 1000 / (t2 - t1)
#        t2_yearidx = int(np.ceil(t2 - startyear_decimal))
#        return mb_mwea, glac_wide_volume_annual[t2_yearidx]
    else:        
        # Specific mass balance
        mb_mwea = mbmod.glac_wide_massbaltotal.sum() / mbmod.glacier_area_initial.sum() / (mbmod.nmonths / 12)        
        return mb_mwea


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
    # Unpack variables
    count = list_packed_vars[0]
    glac_no = list_packed_vars[1]
    gcm_name = list_packed_vars[2]

    time_start = time.time()
    parser = getparser()
    args = parser.parse_args()

    if args.debug == 1:
        debug = True
    else:
        debug = False
        
    # ===== LOAD GLACIERS =====
    main_glac_rgi = modelsetup.selectglaciersrgitable(glac_no=glac_no)
    
    # ===== TIME PERIOD =====
    dates_table = modelsetup.datesmodelrun(
            startyear=pygem_prms.ref_startyear, endyear=pygem_prms.ref_endyear, spinupyears=pygem_prms.ref_spinupyears, 
            option_wateryear=pygem_prms.ref_wateryear)
    
    # ===== LOAD CLIMATE DATA =====
    # Climate class
    assert gcm_name in ['ERA5', 'ERA-Interim'], 'Error: Climate class not set up for ' + gcm_name
    gcm = class_climate.GCM(name=gcm_name)    
    # Air temperature [degC]
    gcm_temp, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(gcm.temp_fn, gcm.temp_vn, main_glac_rgi, dates_table)
    if pygem_prms.option_ablation == 2 and gcm_name in ['ERA5']:
        gcm_tempstd, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(gcm.tempstd_fn, gcm.tempstd_vn,
                                                                        main_glac_rgi, dates_table)
    else:
        gcm_tempstd = np.zeros(gcm_temp.shape)
    # Precipitation [m]
    gcm_prec, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(gcm.prec_fn, gcm.prec_vn, main_glac_rgi, dates_table)
    # Elevation [m asl]
    gcm_elev = gcm.importGCMfxnearestneighbor_xarray(gcm.elev_fn, gcm.elev_vn, main_glac_rgi)
    # Lapse rate [degC m-1]
    gcm_lr, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(gcm.lr_fn, gcm.lr_vn, main_glac_rgi, dates_table)
    
    # Loop through glaciers
    for glac in range(main_glac_rgi.shape[0]):
        if glac == 0 or glac == main_glac_rgi.shape[0]:
            print(gcm_name,':', main_glac_rgi.loc[main_glac_rgi.index.values[glac],'RGIId'])
        # Select subsets of data
        glacier_rgi_table = main_glac_rgi.loc[main_glac_rgi.index.values[glac], :]
        glacier_str = '{0:0.5f}'.format(glacier_rgi_table['RGIId_float'])
        
        # Glacier data
        gdir = single_flowline_glacier_directory(glacier_str)
        fls = gdir.read_pickle('inversion_flowlines')
        
        # Add climate data to glacier directory
        gdir.historical_climate = {'elev': gcm_elev[glac],
                                   'temp': gcm_temp[glac,:],
                                   'tempstd': gcm_tempstd[glac,:],
                                   'prec': gcm_prec[glac,:],
                                   'lr': gcm_lr[glac,:]}
        gdir.dates_table = dates_table
        
        # Calibration data
        try:
            mbdata_fn = gdir.get_filepath('mb_obs')
            with open(mbdata_fn, 'rb') as f:
                mbdata = pickle.load(f)
        except:
            mbdata = None
        
        mb_obs_mwea = mbdata['mb_mwea']
        mb_obs_mwea_err = mbdata['mb_mwea_err']
        
        # Mass balance model
        glacier_area = fls[0].widths_m * fls[0].dx_meter
        if glacier_area.sum() > 0 and mbdata is not None:

            modelprms = {'kp': pygem_prms.kp, 
                         'tbias': pygem_prms.tbias,
                         'ddfsnow': pygem_prms.ddfsnow,
                         'ddfice': pygem_prms.ddfice,
                         'tsnow_threshold': pygem_prms.tsnow_threshold,
                         'precgrad': pygem_prms.precgrad}

            if debug:
                print(glacier_str + '  kp: ' + str(np.round(modelprms['kp'],2)) + 
                      ' ddfsnow: ' + str(np.round(modelprms['ddfsnow'],4)) + 
                      ' tbias: ' + str(np.round(modelprms['tbias'],2)))
                                                        
            print('- Do all comparisons based on total mass loss (kg), not mwea?')
            
            
            # ===== OBJECTIVE FUNCTIONS (used for HH2015 and modified HH2015) ======
            if pygem_prms.option_calibration in ['HH2015', 'HH2015_modified']:
                def objective(modelprms_subset):
                    """ Objective function for mass balance data (mimize difference between model and observation).
                    
                    Parameters
                    ----------
                    modelprms_subset : list of model parameters [kp, ddfsnow, tbias]
                    """
                    # Use a subset of model parameters to reduce number of constraints required
                    modelprms['kp'] = modelprms_subset[0]
                    modelprms['ddfsnow'] = modelprms_subset[1]
                    modelprms['ddfice'] = modelprms['ddfsnow'] / pygem_prms.ddfsnow_iceratio
                    modelprms['tbias'] = modelprms_subset[2]
                    # Mass balance
                    mb_mwea = mb_mwea_calc(gdir, modelprms, glacier_rgi_table, fls=fls)
                    # Difference with observation (mwea)
                    mb_dif_mwea_abs = abs(mb_obs_mwea - mb_mwea)    
                    return mb_dif_mwea_abs
                
                
                def run_objective(modelprms_init, mb_obs_mwea, kp_bnds=(0.33,3), tbias_bnds=(-10,10),
                                  ddfsnow_bnds=(0.0026,0.0056), run_opt=True, eps_opt=pygem_prms.eps_opt, 
                                  ftol_opt=pygem_prms.ftol_opt):
                    """ Run the optimization for the single glacier objective function.
                    
                    Parameters
                    ----------
                    modelparams_init : list of model parameters to calibrate [kp, ddfsnow, tbias]
                    kp_bnds, tbias_bnds, ddfsnow_bnds, precgrad_bnds : tuples (lower & upper bounds)
                    run_opt : Boolean statement run optimization or bypass optimization and run with initial parameters
                        
                    Returns
                    -------
                    modelparams : model parameters dict and specific mass balance (mwea)
                    """
                    # Bounds
                    modelprms_bnds = (kp_bnds, ddfsnow_bnds, tbias_bnds)
                    # Run the optimization
                    if run_opt:
                        modelprms_opt = minimize(objective, modelprms_init, method=pygem_prms.method_opt,
                                                 bounds=modelprms_bnds, options={'ftol':ftol_opt, 'eps':eps_opt})
                        # Record the optimized parameters
                        modelprms_subset = modelprms_opt.x
                    else:
                        modelprms_subset = modelprms.copy()
                    modelprms['kp'] = modelprms_subset[0]
                    modelprms['ddfsnow'] = modelprms_subset[1]
                    modelprms['ddfice'] = modelprms['ddfsnow'] / pygem_prms.ddfsnow_iceratio
                    modelprms['tbias'] = modelprms_subset[2]
                    # Re-run the optimized parameters in order to see the mass balance
                    mb_mwea = mb_mwea_calc(gdir, modelprms, glacier_rgi_table, fls=fls)
                    return modelprms, mb_mwea
            
            ##    # ===== OLD CALIBRATION OPTIONS =====
            # Option 2: use MCMC method to determine posterior probability distributions of the three parameters tbias,
            #           ddfsnow and kp. Then create an ensemble of parameter sets evenly sampled from these
            #           distributions, and output these sets of parameters and their corresponding mass balances to be used in
            #           the simulations.
            if pygem_prms.option_calibration == 'MCMC':

                # ===== Define functions needed for MCMC method =====
                def run_MCMC(kp_disttype=pygem_prms.kp_disttype, 
                             kp_gamma_alpha=pygem_prms.kp_gamma_alpha, kp_gamma_beta=pygem_prms.kp_gamma_beta, 
                             kp_lognorm_mu=pygem_prms.kp_lognorm_mu, kp_lognorm_tau=pygem_prms.kp_lognorm_tau, 
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
                             tune_throughout=True, save_interval=None, burn_till_tuned=False, stop_tuning_after=5, 
                             verbose=0, progress_bar=args.progress_bar, dbname=None,
                             use_potentials=True, mb_max_loss=None):
                    """
                    Runs the MCMC algorithm.
        
                    Runs the MCMC algorithm by setting the prior distributions and calibrating the probability 
                    distributions of three model parameters for the mass balance function.
        
                    Parameters
                    ----------
                    kp_disttype : str
                        Distribution type of precipitation factor (either 'lognormal', 'uniform', or 'custom')
                    kp_lognorm_mu, kp_lognorm_tau : float
                        Lognormal mean and tau (1/variance) of precipitation factor
                    kp_mu, kp_sigma, kp_bndlow, kp_bndhigh, kp_start : float
                        Mean, stdev, lower bound, upper bound, and start value of precipitation factor
                    tbias_disttype : str
                        Distribution type of tbias (either 'truncnormal' or 'uniform')
                    tbias_mu, tbias_sigma, tbias_bndlow, tbias_bndhigh, tbias_start : float
                        Mean, stdev, lower bound, upper bound, and start value of temperature bias
                    ddfsnow_disttype : str
                        Distribution type of degree day factor of snow (either 'truncnormal' or 'uniform')
                    ddfsnow_mu, ddfsnow_sigma, ddfsnow_bndlow, ddfsnow_bndhigh, ddfsnow_start : float
                        Mean, stdev, lower bound, upper bound, and start value of degree day factor of snow
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
                        If true, tuning will continue after the burnin period; otherwise tuning will halt at the end of 
                        the burnin period (default True).
                    save_interval : int or None
                        If given, the model state will be saved at intervals of this many iterations (default None).
                    burn_till_tuned: boolean
                        If True the Sampler will burn samples until all step methods are tuned. A tuned step methods is 
                        one that was not tuned for the last `stop_tuning_after` tuning intervals. The burn-in phase will 
                        have a minimum of 'burn' iterations but could be longer if tuning is needed. After the phase is 
                        done the sampler will run for another (iter - burn) iterations, and will tally the samples 
                        according to the 'thin' argument. This means that the total number of iteration is updated 
                        throughout the sampling procedure.  If True, it also overrides the tune_thorughout argument, so 
                        no step method will be tuned when sample are being tallied (default False).
                    stop_tuning_after: int
                        The number of untuned successive tuning interval needed to be reached in order for the burn-in 
                        phase to be done (if burn_till_tuned is True) (default 5).
                    verbose : int
                        An integer controlling the verbosity of the models output for debugging (default 0).
                    progress_bar : boolean
                        Display progress bar while sampling (default True).
                    dbname : str
                        Choice of database name the sample should be saved to (default None).
                    use_potentials : Boolean
                        Boolean to use of potential functions to further constrain likelihood functionns
                    mb_max_loss : float
                        Mass balance [mwea] at which the glacier completely melts
        
                    Returns
                    -------
                    pymc.MCMC.MCMC
                        Returns a model that contains sample traces of tbias, ddfsnow, kp and massbalance. These
                        samples can be accessed by calling the trace attribute. For example:
        
                            model.trace('ddfsnow')[:]
        
                        gives the trace of ddfsnow values.
        
                        A trace, or Markov Chain, is an array of values outputed by the MCMC simulation which defines 
                        the posterior probability distribution of the variable at hand.
                    """
                    # ===== PRIOR DISTRIBUTIONS =====
                    # Priors dict to record values for export
                    priors_dict = {}
                    priors_dict['kp_disttype'] = kp_disttype
                    priors_dict['tbias_disttype'] = tbias_disttype
                    priors_dict['ddfsnow_disttype'] = ddfsnow_disttype
                    # Precipitation factor [-]
                    if kp_disttype == 'gamma':
                        kp = pymc.Gamma('kp', alpha=kp_gamma_alpha, beta=kp_gamma_beta, value=kp_start)
                        priors_dict['kp_gamma_alpha'] = kp_gamma_alpha
                        priors_dict['kp_gamma_beta'] = kp_gamma_beta
                    elif kp_disttype =='lognormal':
                        #  lognormal distribution (roughly 0.3 to 3)
                        kp_start = np.exp(kp_start)
                        kp = pymc.Lognormal('kp', mu=kp_lognorm_mu, tau=kp_lognorm_tau, value=kp_start)
                        priors_dict['kp_lognorm_mu'] = kp_lognorm_mu
                        priors_dict['kp_lognorm_tau'] = kp_lognorm_tau
                    elif kp_disttype == 'uniform':
                        kp = pymc.Uniform('kp', lower=kp_bndlow, upper=kp_bndhigh, value=kp_start)
                        priors_dict['kp_bndlow'] = kp_bndlow
                        priors_dict['kp_bndhigh'] = kp_bndhigh
                        
                    # Temperature bias [degC]
                    if tbias_disttype == 'normal':
                        tbias = pymc.Normal('tbias', mu=tbias_mu, tau=1/(tbias_sigma**2), value=tbias_start)
                        priors_dict['tbias_mu'] = tbias_mu
                        priors_dict['tbias_sigma'] = tbias_sigma
                    elif tbias_disttype =='truncnormal':
                        tbias = pymc.TruncatedNormal('tbias', mu=tbias_mu, tau=1/(tbias_sigma**2),
                                                     a=tbias_bndlow, b=tbias_bndhigh, value=tbias_start)
                        priors_dict['tbias_mu'] = tbias_mu
                        priors_dict['tbias_sigma'] = tbias_sigma
                        priors_dict['tbias_bndlow'] = tbias_bndlow
                        priors_dict['tbias_bndhigh'] = tbias_bndhigh
                    elif tbias_disttype =='uniform':
                        tbias = pymc.Uniform('tbias', lower=tbias_bndlow, upper=tbias_bndhigh, value=tbias_start)
                        priors_dict['tbias_bndlow'] = tbias_bndlow
                        priors_dict['tbias_bndhigh'] = tbias_bndhigh
        
                    # Degree day factor of snow [mwe degC-1 d-1]
                    #  always truncated normal distribution with mean 0.0041 mwe degC-1 d-1 and standard deviation of 
                    #  0.0015 (Braithwaite, 2008), since it's based on data; uniform should only be used for testing
                    if ddfsnow_disttype == 'truncnormal':
                        ddfsnow = pymc.TruncatedNormal('ddfsnow', mu=ddfsnow_mu, tau=1/(ddfsnow_sigma**2), 
                                                       a=ddfsnow_bndlow, b=ddfsnow_bndhigh, value=ddfsnow_start)
                        priors_dict['ddfsnow_mu'] = ddfsnow_mu
                        priors_dict['ddfsnow_sigma'] = ddfsnow_sigma
                        priors_dict['ddfsnow_bndlow'] = ddfsnow_bndlow
                        priors_dict['ddfsnow_bndhigh'] = ddfsnow_bndhigh
                    elif ddfsnow_disttype == 'uniform':
                        ddfsnow = pymc.Uniform('ddfsnow', lower=ddfsnow_bndlow, upper=ddfsnow_bndhigh, 
                                               value=ddfsnow_start)
                        priors_dict['ddfsnow_bndlow'] = ddfsnow_bndlow
                        priors_dict['ddfsnow_bndhigh'] = ddfsnow_bndhigh 
        
                    # ===== DETERMINISTIC FUNCTION ====
                    # Define deterministic function for MCMC model based on our a priori probobaility distributions.
                    @deterministic(plot=False)
                    def massbal(tbias=tbias, kp=kp, ddfsnow=ddfsnow):
                        """
                        Likelihood function for mass balance [mwea] based on model parameters
                        """
                        modelprms_copy = modelprms.copy()
                        if tbias is not None:
                            modelprms_copy['tbias'] = float(tbias)
                        if kp is not None:
                            modelprms_copy['kp'] = float(kp)
                        if ddfsnow is not None:
                            modelprms_copy['ddfsnow'] = float(ddfsnow)
                            modelprms_copy['ddfice'] = modelprms_copy['ddfsnow'] / pygem_prms.ddfsnow_iceratio
                        mb_mwea = mb_mwea_calc(gdir, modelprms_copy, glacier_rgi_table, fls=fls)
                        return mb_mwea
        
                    # ===== POTENTIAL FUNCTIONS =====
                    # Potential functions are used to impose additional constrains on the model
                    @pymc.potential
                    def mb_max(mb_max_loss=mb_max_loss, massbal=massbal):
                        """Model parameters cannot completely melt the glacier, 
                           i.e., reject any parameter set within 0.01 mwea of completely melting the glacier"""
                        if massbal < mb_max_loss:
                            return -np.inf
                        else:
                            return 0
        
                    @pymc.potential
                    def must_melt(tbias=tbias, kp=kp, ddfsnow=ddfsnow):
                        """
                        Likelihood function for mass balance [mwea] based on model parameters
                        """
                        modelprms_copy = modelprms.copy()
                        if tbias is not None:
                            modelprms_copy['tbias'] = float(tbias)
                        if kp is not None:
                            modelprms_copy['kp'] = float(kp)
                        if ddfsnow is not None:
                            modelprms_copy['ddfsnow'] = float(ddfsnow)
                            modelprms_copy['ddfice'] = modelprms_copy['ddfsnow'] / pygem_prms.ddfsnow_iceratio
                        nbinyears_negmbclim = mb_mwea_calc(gdir, modelprms_copy, glacier_rgi_table, fls=fls, 
                                                           return_tbias_mustmelt=True)
                        if nbinyears_negmbclim > 0:
                            return 0
                        else:
                            return -np.inf
        
                    # ===== OBSERVED DATA =====
                    #  Observed data defines the observed likelihood of mass balances (based on geodetic observations)
                    obs_massbal = pymc.Normal('obs_massbal', mu=massbal, tau=(1/(mb_obs_mwea_err**2)),
                                              value=float(mb_obs_mwea), observed=True)
                    # Set model
                    if use_potentials:
                        model = pymc.MCMC([{'kp':kp, 'tbias':tbias, 'ddfsnow':ddfsnow,
                                           'massbal':massbal, 'obs_massbal':obs_massbal}, mb_max, must_melt])
                    else:
                        model = pymc.MCMC({'kp':kp, 'tbias':tbias, 'ddfsnow':ddfsnow,
                                           'massbal':massbal, 'obs_massbal':obs_massbal})
        
                    # Step method (if changed from default)
                    #  Adaptive metropolis is supposed to perform block update, i.e., update all model parameters 
                    #  together based on their covariance, which would reduce autocorrelation; however, tests show 
                    #  doesn't make a difference.
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
                    
                    return model, priors_dict
        
                # ===== RUNNING MCMC =====
                # Regional priors
                kp_gamma_alpha = pygem_prms.kp_gamma_alpha
                kp_gamma_beta = pygem_prms.kp_gamma_beta
                tbias_mu = pygem_prms.tbias_mu
                tbias_sigma = pygem_prms.tbias_sigma   
#                kp_gamma_alpha = pygem_prms.kp_gamma_region_dict[glacier_rgi_table.loc['region']][0]
#                kp_gamma_beta = pygem_prms.kp_gamma_region_dict[glacier_rgi_table.loc['region']][1]
#                tbias_mu = pygem_prms.tbias_norm_region_dict[glacier_rgi_table.loc['region']][0]
#                tbias_sigma = pygem_prms.tbias_norm_region_dict[glacier_rgi_table.loc['region']][1]

                modelprms_export = {}
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
                    modelprms['kp'] = kp_start
                    modelprms['ddfsnow'] = ddfsnow_start
                    modelprms['ddfice'] = modelprms['ddfsnow'] / pygem_prms.ddfsnow_iceratio

                    # Retrieve temperature bias bounds
                    tbias_bndlow, tbias_bndhigh, mb_max_loss = retrieve_priors(
                            gdir, glacier_rgi_table, modelprms, fls, debug=False)
                    
                    if debug:
                        print('tbias_bndlow:', np.round(tbias_bndlow,2), 'tbias_bndhigh:', np.round(tbias_bndhigh,2),
                              'mb_max_loss:', np.round(mb_max_loss,2))

                    # Check that tbias mu and sigma is somewhat within bndhigh and bndlow
                    if tbias_start > tbias_bndhigh:
                        tbias_start = tbias_bndhigh
                    elif tbias_start < tbias_bndlow:
                        tbias_start = tbias_bndlow

                    if debug:
                        print('\ntbias_start:', np.round(tbias_start,3), 'pf_start:', np.round(kp_start,3),
                              'ddf_start:', np.round(ddfsnow_start,4))

                    model, priors_dict = run_MCMC(
                            iterations=pygem_prms.mcmc_sample_no, burn=pygem_prms.mcmc_burn_no, 
                            step=pygem_prms.mcmc_step,
                            kp_gamma_alpha=kp_gamma_alpha, kp_gamma_beta=kp_gamma_beta, kp_start=kp_start,
                            tbias_mu=tbias_mu, tbias_sigma=tbias_sigma, tbias_start=tbias_start,
                            ddfsnow_start=ddfsnow_start, mb_max_loss=mb_max_loss,
                            use_potentials=True)

                    if debug:
                        print('\nacceptance ratio:', model.step_method_dict[next(iter(model.stochastics))][0].ratio)


                    # Store data from model to be exported
                    chain_str = 'chain_' + str(n_chain)
                    modelprms_export['tbias'] = {chain_str : list(model.trace('tbias')[:])}
                    modelprms_export['kp'] = {chain_str  : list(model.trace('kp')[:])}
                    modelprms_export['ddfsnow'] = {chain_str : list(model.trace('ddfsnow')[:])}
                    modelprms_export['ddfice'] = {chain_str : list(model.trace('ddfsnow')[:] / 
                                                               pygem_prms.ddfsnow_iceratio)}
                    modelprms_export['mb_mwea'] = {chain_str : list(model.trace('massbal')[:])}

                # Export model parameters
                modelprms_export['precgrad'] = [pygem_prms.precgrad]
                modelprms_export['tsnow_threshold'] = [pygem_prms.tsnow_threshold]
                modelprms_export['mb_obs_mwea'] = [mb_obs_mwea]
                modelprms_export['mb_obs_mwea_err'] = [mb_obs_mwea_err]
                modelprms_export['priors'] = priors_dict
                modelprms_fullfn = gdir.get_filepath('pygem_modelprms')
                if os.path.exists(modelprms_fullfn):
                    with open(modelprms_fullfn, 'rb') as f:
                        modelprms_dict = pickle.load(f)                        
                    modelprms_dict[pygem_prms.option_calibration] = modelprms_export
                else:
                    modelprms_dict = {pygem_prms.option_calibration: modelprms_export}
                with open(modelprms_fullfn, 'wb') as f:
                    pickle.dump(modelprms_dict, f)
                    

            
            #%% ===== HUSS AND HOCK (2015) CALIBRATION =====
            elif pygem_prms.option_calibration == 'HH2015':
                tbias_init = 0
                tbias_bndlow = -10
                tbias_bndhigh = 10
                tbias_step = 1
                kp_init = 1.5
                kp_bndlow = 0.8
                kp_bndhigh = 2
                ddfsnow_init = 0.003
                ddfsnow_bndlow = 0.00175
                ddfsnow_bndhigh = 0.0045
                assert pygem_prms.ddfsnow_iceratio == 0.5, 'Error: ddfsnow_iceratio for HH2015 must be 0.5'
                
                continue_param_search = True
                # ===== ROUND 1: PRECIPITATION FACTOR ======
                if debug:
                    print('Round 1:')
                # Lower bound
                modelprms['kp'] = kp_bndlow
                mb_mwea_kp_low = mb_mwea_calc(gdir, modelprms, glacier_rgi_table, fls=fls)
                # Upper bound
                modelprms['kp'] = kp_bndhigh
                mb_mwea_kp_high = mb_mwea_calc(gdir, modelprms, glacier_rgi_table, fls=fls)
                # Optimimum precipitation factor
                if mb_obs_mwea < mb_mwea_kp_low:
                    kp_opt = kp_bndlow
                    mb_mwea = mb_mwea_kp_low
                elif mb_obs_mwea > mb_mwea_kp_high:
                    kp_opt = kp_bndhigh
                    mb_mwea = mb_mwea_kp_high
                else:
                    modelprms['kp'] = kp_init
                    modelprms_subset = [modelprms['kp'], modelprms['ddfsnow'], modelprms['tbias']]
                    kp_bnds = (kp_bndlow, kp_bndhigh)
                    ddfsnow_bnds = (ddfsnow_init, ddfsnow_init)
                    tbias_bnds = (tbias_init, tbias_init)
                    modelparams_opt, mb_mwea = run_objective(modelprms_subset, mb_obs_mwea, kp_bnds=kp_bnds, 
                                                             tbias_bnds=tbias_bnds, ddfsnow_bnds=ddfsnow_bnds, 
                                                             ftol_opt=1e-3)
                    kp_opt = modelparams_opt['kp']
                    continue_param_search = False
                # Update parameter values
                modelprms['kp'] = kp_opt
                if debug:
                    print('  kp:', np.round(kp_opt,2), 'mb_mwea:', np.round(mb_mwea,2))
        
                # ===== ROUND 2: DEGREE-DAY FACTOR OF SNOW ======
                if continue_param_search:
                    if debug:
                        print('Round 2:')
                    # Lower bound
                    modelprms['ddfsnow'] = ddfsnow_bndlow
                    modelprms['ddfice'] = modelprms['ddfsnow'] / pygem_prms.ddfsnow_iceratio
                    mb_mwea_ddflow = mb_mwea_calc(gdir, modelprms, glacier_rgi_table, fls=fls)
                    # Upper bound
                    modelprms['ddfsnow'] = ddfsnow_bndhigh
                    modelprms['ddfice'] = modelprms['ddfsnow'] / pygem_prms.ddfsnow_iceratio
                    mb_mwea_ddfhigh = mb_mwea_calc(gdir, modelprms, glacier_rgi_table, fls=fls)
                    # Optimimum degree-day factor of snow
                    if mb_obs_mwea < mb_mwea_ddfhigh:
                        ddfsnow_opt = ddfsnow_bndhigh
                        mb_mwea = mb_mwea_ddfhigh
                    elif mb_obs_mwea > mb_mwea_ddflow:
                        ddfsnow_opt = ddfsnow_bndlow
                        mb_mwea = mb_mwea_ddflow
                    else:
                        modelprms_subset = [kp_opt, modelprms['ddfsnow'], modelprms['tbias']]
                        kp_bnds = (kp_opt, kp_opt)
                        ddfsnow_bnds = (ddfsnow_bndlow, ddfsnow_bndhigh)
                        tbias_bnds = (tbias_init, tbias_init)
                        modelparams_opt, mb_mwea = run_objective(modelprms_subset, mb_obs_mwea, kp_bnds=kp_bnds, 
                                                             tbias_bnds=tbias_bnds, ddfsnow_bnds=ddfsnow_bnds, 
                                                             ftol_opt=1e-3)
                        ddfsnow_opt = modelparams_opt['ddfsnow']
                        continue_param_search = False
                    # Update parameter values
                    modelprms['ddfsnow'] = ddfsnow_opt
                    modelprms['ddfice'] = modelprms['ddfsnow'] / pygem_prms.ddfsnow_iceratio
                    if debug:
                        print('  ddfsnow:', np.round(ddfsnow_opt,4), 'mb_mwea:', np.round(mb_mwea,2))
                else:
                    ddfsnow_opt = modelprms['ddfsnow']                    
        
                # ===== ROUND 3: TEMPERATURE BIAS ======
                if continue_param_search:
                    if debug:
                        print('Round 3:')
                    # ----- TEMPBIAS: max accumulation -----
                    # Lower temperature bound based on no positive temperatures
                    # Temperature at the lowest bin
                    #  T_bin = T_gcm + lr_gcm * (z_ref - z_gcm) + lr_glac * (z_bin - z_ref) + tbias
                    tbias_max_acc = (-1 * (gdir.historical_climate['temp'] + gdir.historical_climate['lr'] *
                                     (fls[0].surface_h.min() - gdir.historical_climate['elev'])).max())
                    tbias_bndlow = tbias_max_acc
                    modelprms['tbias'] = tbias_bndlow
                    mb_mwea = mb_mwea_calc(gdir, modelprms, glacier_rgi_table, fls=fls)
                    
                    if debug:
                        print('  tbias_bndlow:', np.round(tbias_bndlow,2), 'mb_mwea:', np.round(mb_mwea,2))
            
                    # Upper bound
                    while mb_mwea > mb_obs_mwea and modelprms['tbias'] < 20:
                        modelprms['tbias'] = modelprms['tbias'] + tbias_step
                        mb_mwea = mb_mwea_calc(gdir, modelprms, glacier_rgi_table, fls=fls)
                        if debug:
                            print('  tc:', np.round(modelprms['tbias'],2), 'mb_mwea:', np.round(mb_mwea,2))
                        tbias_bndhigh = modelprms['tbias']
                        
                    modelprms_subset = [kp_opt, ddfsnow_opt, modelprms['tbias'] - tbias_step/2]
                    kp_bnds = (kp_opt, kp_opt)
                    ddfsnow_bnds = (ddfsnow_opt, ddfsnow_opt)
                    tbias_bnds = (tbias_bndhigh-tbias_step, tbias_bndhigh)
                    
                    modelparams_opt, mb_mwea = run_objective(modelprms_subset, mb_obs_mwea, kp_bnds=kp_bnds, 
                                                             tbias_bnds=tbias_bnds, ddfsnow_bnds=ddfsnow_bnds, 
                                                             ftol_opt=1e-3)
                    # Update parameter values
                    tbias_opt = modelparams_opt['tbias']
                    modelprms['tbias'] = tbias_opt            
                    if debug:
                        print('  tbias:', np.round(tbias_opt,3), 'mb_mwea:', np.round(mb_mwea,3))
                    
                else:
                    tbias_opt = modelprms['tbias']

                # Export model parameters
                modelprms = modelparams_opt
                for vn in ['ddfice', 'ddfsnow', 'kp', 'precgrad', 'tbias', 'tsnow_threshold']:
                    modelprms[vn] = [modelprms[vn]]
                modelprms['mb_mwea'] = [mb_mwea]
                modelprms['mb_obs_mwea'] = [mb_obs_mwea]
                modelprms['mb_obs_mwea_err'] = [mb_obs_mwea_err]
                modelprms_fullfn = gdir.get_filepath('pygem_modelprms')
                if os.path.exists(modelprms_fullfn):
                    with open(modelprms_fullfn, 'rb') as f:
                        modelprms_dict = pickle.load(f)                        
                    modelprms_dict[pygem_prms.option_calibration] = modelprms
                else:
                    modelprms_dict = {pygem_prms.option_calibration: modelprms}
                with open(modelprms_fullfn, 'wb') as f:
                    pickle.dump(modelprms_dict, f)



            #%% ===== MODIFIED HUSS AND HOCK (2015) CALIBRATION =====
            # - precipitation factor, then temperature bias (no ddfsnow)
            # - ranges different
            elif pygem_prms.option_calibration == 'HH2015_modified':
                tbias_init = 0
                tbias_step = 1
                kp_init = 1
                kp_bndlow = 0.5
                kp_bndhigh = 5
                ddfsnow_init = 0.0041
                
                # ----- Temperature bias bounds -----
                # Tbias lower bound based on no positive temperatures
                tbias_bndlow = (-1 * (gdir.historical_climate['temp'] + gdir.historical_climate['lr'] *
                                 (fls[0].surface_h.min() - gdir.historical_climate['elev'])).max())
                modelprms['tbias'] = tbias_bndlow
                mb_mwea = mb_mwea_calc(gdir, modelprms, glacier_rgi_table, fls=fls)        
                if debug:
                    print('  tbias_bndlow:', np.round(tbias_bndlow,2), 'mb_mwea:', np.round(mb_mwea,2))
                # Tbias upper bound (based on kp_bndhigh)
                modelprms['kp'] = kp_bndhigh
                while mb_mwea > mb_obs_mwea and modelprms['tbias'] < 20:
                    modelprms['tbias'] = modelprms['tbias'] + tbias_step
                    mb_mwea = mb_mwea_calc(gdir, modelprms, glacier_rgi_table, fls=fls)
                    if debug:
                        print('  tc:', np.round(modelprms['tbias'],2), 'mb_mwea:', np.round(mb_mwea,2))
                    tbias_bndhigh = modelprms['tbias']

                # ROUND 1: PRECIPITATION FACTOR
                # Adjust bounds based on range of temperature bias
                if tbias_init > tbias_bndhigh:
                    tbias_init = tbias_bndhigh
                elif tbias_init < tbias_bndlow:
                    tbias_init = tbias_bndlow
                modelprms['tbias'] = tbias_init
                modelprms['kp'] = kp_init

                tbias_bndlow_opt = tbias_init
                tbias_bndhigh_opt = tbias_init

                # Constrain bounds of precipitation factor and temperature bias
                mb_mwea = mb_mwea_calc(gdir, modelprms, glacier_rgi_table, fls=fls)

                if debug:
                    print('\ntbias:', np.round(modelprms['tbias'],2), 'kp:', np.round(modelprms['kp'],2),
                          'mb_mwea:', np.round(mb_mwea,2), 'obs_mwea:', np.round(mb_obs_mwea,2))

                # Adjust lower or upper bound based on the observed mass balance
                test_count = 0
                if mb_mwea > mb_obs_mwea:
                    if debug:
                        print('increase tbias, decrease kp')
                    kp_bndhigh = 1
                    # Check if lowest bound causes good agreement
                    modelprms['kp'] = kp_bndlow
                        
                    while mb_mwea > mb_obs_mwea and test_count < 20:
                        mb_mwea = mb_mwea_calc(gdir, modelprms, glacier_rgi_table, fls=fls)
                        if debug:
                            print('\ntbias:', np.round(modelprms['tbias'],2), 'kp:', np.round(modelprms['kp'],2),
                                  'mb_mwea:', np.round(mb_mwea,2), 'obs_mwea:', np.round(mb_obs_mwea,2))
                        if test_count > 0:
                            tbias_bndlow_opt = modelprms['tbias'] - tbias_step
                            tbias_bndhigh_opt = modelprms['tbias']
                        modelprms['tbias'] = modelprms['tbias'] + tbias_step
                        test_count += 1
                else:
                    if debug:
                        print('decrease tbias, increase kp')
                    kp_bndlow = 1
                    # Check if upper bound causes good agreement
                    modelprms['kp'] = kp_bndhigh
                    
                    while mb_mwea < mb_obs_mwea and test_count < 20:
                        mb_mwea = mb_mwea_calc(gdir, modelprms, glacier_rgi_table, fls=fls)
                        if debug:
                            print('\ntbias:', np.round(modelprms['tbias'],2), 'kp:', np.round(modelprms['kp'],2),
                                  'mb_mwea:', np.round(mb_mwea,2), 'obs_mwea:', np.round(mb_obs_mwea,2))
                        if test_count > 0:
                            tbias_bndlow_opt = modelprms['tbias']
                            tbias_bndhigh_opt = modelprms['tbias'] + tbias_step
                        modelprms['tbias'] = modelprms['tbias'] - tbias_step
                        test_count += 1

                # ===== RUN OPTIMIZATION WITH CONSTRAINED BOUNDS =====                
                tbias_bnds = (tbias_bndlow_opt, tbias_bndhigh_opt)
                kp_bnds = (kp_bndlow, kp_bndhigh)
                ddfsnow_bnds = (ddfsnow_init, ddfsnow_init)
                tbias_init = np.mean([tbias_bndlow_opt, tbias_bndhigh_opt])
                kp_init = kp_init
                
                if debug:
                    print('tbias bounds:', tbias_bnds)
                    print('kp bounds:', kp_bnds)
                
                modelprms_subset = [kp_init, ddfsnow_init, tbias_init]                
                modelparams_opt, mb_mwea = run_objective(modelprms_subset, mb_obs_mwea, kp_bnds=kp_bnds, 
                                                         tbias_bnds=tbias_bnds, ddfsnow_bnds=ddfsnow_bnds, 
                                                         ftol_opt=1e-3)

                kp_opt = modelparams_opt['kp']
                tbias_opt = modelparams_opt['tbias']
                if debug:
                    print('\nmb_mwea:', np.round(mb_mwea,2), 'obs_mb:', np.round(mb_obs_mwea,2),
                          '\nkp:', np.round(kp_opt,2), 'tbias:', np.round(tbias_opt,2))
                
                # Epsilon (the amount the variable change to calculate the jacobian) can be too small, which causes
                #  the minimization to believe it has reached a local minima and stop. Therefore, adjust epsilon
                #  to ensure this is not the case.
                eps_opt_new = pygem_prms.eps_opt
                nround = 0
                while np.absolute(mb_mwea - mb_obs_mwea) > 0.3 and eps_opt_new <= 0.1:
                    nround += 1
                    if debug:
                        print('DIDNT WORK SO TRYING NEW INITIAL CONDITIONS')
                        print('  old eps_opt:', eps_opt_new)
                        
                    eps_opt_new = eps_opt_new * 10
                    if debug:    
                        print('  new eps_opt:', eps_opt_new)
                        
                    modelprms_subset = [kp_init, ddfsnow_init, tbias_init]                
                    modelparams_opt, mb_mwea = run_objective(modelprms_subset, mb_obs_mwea, kp_bnds=kp_bnds, 
                                                             tbias_bnds=tbias_bnds, ddfsnow_bnds=ddfsnow_bnds, 
                                                             ftol_opt=eps_opt_new)
                    kp_opt = modelparams_opt['kp']
                    tbias_opt = modelparams_opt['tbias']
                    if debug:
                        print('\nmb_mwea:', np.round(mb_mwea,2), 'obs_mb:', np.round(mb_obs_mwea,2),
                              '\nkp:', np.round(kp_opt,2), 'tbias:', np.round(tbias_opt,2))
                        
                # Export model parameters
                modelprms = modelparams_opt
                for vn in ['ddfice', 'ddfsnow', 'kp', 'precgrad', 'tbias', 'tsnow_threshold']:
                    modelprms[vn] = [modelprms[vn]]
                modelprms['mb_mwea'] = [mb_mwea]
                modelprms['mb_obs_mwea'] = [mb_obs_mwea]
                modelprms['mb_obs_mwea_err'] = [mb_obs_mwea_err]
                modelprms_fullfn = gdir.get_filepath('pygem_modelprms')
                if os.path.exists(modelprms_fullfn):
                    with open(modelprms_fullfn, 'rb') as f:
                        modelprms_dict = pickle.load(f)                        
                    modelprms_dict[pygem_prms.option_calibration] = modelprms
                else:
                    modelprms_dict = {pygem_prms.option_calibration: modelprms}
                with open(modelprms_fullfn, 'wb') as f:
                    pickle.dump(modelprms_dict, f)

# Removed "group calibration"
# Removed old "option 1" of expanding solution space

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

    if args.debug == 1:
        debug = True
    else:
        debug = False
        
    print("\nHow to set cfg.BASENAMES['mb_obs'] permanently?\n")
    if not 'mb_obs' in cfg.BASENAMES:
        cfg.BASENAMES['mb_obs'] = ('mb_data.pkl', 'Mass balance observations')
    if not 'pygem_modelprms' in cfg.BASENAMES:
        cfg.BASENAMES['pygem_modelprms'] = ('pygem_modelprms.pkl', 'PyGEM model parameters')
    if not 'mass_consensus' in cfg.BASENAMES:
        cfg.BASENAMES['mass_consensus'] = ('mass_consensus.pkl', 'Glacier mass from consensus ice thickness estimate')

    # RGI glacier number
    if args.rgi_glac_number_fn is not None:
        with open(args.rgi_glac_number_fn, 'rb') as f:
            glac_no = pickle.load(f)
    elif pygem_prms.glac_no is not None:
        glac_no = pygem_prms.glac_no
    else:
        main_glac_rgi_all = modelsetup.selectglaciersrgitable(
                rgi_regionsO1=pygem_prms.rgi_regionsO1, rgi_regionsO2=pygem_prms.rgi_regionsO2,
                rgi_glac_number=pygem_prms.rgi_glac_number)
        glac_no = list(main_glac_rgi_all['rgino_str'].values)

    # Number of cores for parallel processing
    if args.option_parallels != 0:
        num_cores = int(np.min([len(glac_no), args.num_simultaneous_processes]))
    else:
        num_cores = 1

    # Glacier number lists to pass for parallel processing
    glac_no_lsts = split_glaciers.split_list(glac_no, n=num_cores, option_ordered=args.option_ordered)
            
    # Reference GCM name
    gcm_name = args.ref_gcm_name
    print('Reference climate data is:', gcm_name)

    if pygem_prms.option_calibration == 'MCMC':
        print('Chains:', pygem_prms.n_chains, 'Iterations:', pygem_prms.mcmc_sample_no)
        
    # Pack variables for multiprocessing
    list_packed_vars = []
    for count, glac_no_lst in enumerate(glac_no_lsts):
        list_packed_vars.append([count, glac_no_lst, gcm_name])

    # Parallel processing
    if args.option_parallels != 0:
        print('Processing in parallel with ' + str(args.num_simultaneous_processes) + ' cores...')
        with multiprocessing.Pool(args.num_simultaneous_processes) as p:
            p.map(main,list_packed_vars)
    # If not in parallel, then only should be one loop
    else:
        # Loop through the chunks and export bias adjustments
        for n in range(len(list_packed_vars)):
            main(list_packed_vars[n])



    print('Total processing time:', time.time()-time_start, 's')

    #%% ===== PLOTTING AND PROCESSING FOR MODEL DEVELOPMENT =====
#    # Place local variables in variable explorer
    if (args.option_parallels == 0):
        main_vars_list = list(main_vars.keys())
        gdir = main_vars['gdir']
        fls = main_vars['fls']
        fl = fls[0]
#        mbmod = main_vars['mbmod']
#        dates_table = main_vars['dates_table']
        mbdata = main_vars['mbdata']
        modelprms = main_vars['modelprms']
        modelprms_dict = main_vars['modelprms_dict']
