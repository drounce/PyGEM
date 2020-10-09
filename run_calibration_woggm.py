"""Run a model simulation."""
# Default climate data is ERA-Interim; specify CMIP5 by specifying a filename to the argument:
#    (Command line) python run_simulation_list_multiprocess.py -gcm_list_fn=C:\...\gcm_rcpXX_filenames.txt
#      - Default is running ERA-Interim in parallel with five processors.
#    (Spyder) %run run_simulation_list_multiprocess.py C:\...\gcm_rcpXX_filenames.txt -option_parallels=0
#      - Spyder cannot run parallels, so always set -option_parallels=0 when testing in Spyder.
# Spyder cannot run parallels, so always set -option_parallels=0 when testing in Spyder.

# Built-in libraries
import argparse
import collections
import inspect
import multiprocessing
import os
import time
# External libraries
import pandas as pd
import pickle
import pymc
from pymc import deterministic
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy import stats
import xarray as xr

# Local libraries
import class_climate
#import class_mbdata
import pygem.pygem_input as pygem_prms
from pygem.massbalance import PyGEMMassBalance
#from pygem.glacierdynamics import MassRedistributionCurveModel
from pygem.oggm_compat import single_flowline_glacier_directory, single_flowline_glacier_directory_with_calving
import pygemfxns_gcmbiasadj as gcmbiasadj
import pygemfxns_modelsetup as modelsetup
import spc_split_glaciers as split_glaciers

from oggm import cfg
from oggm import graphics
from oggm import tasks
from oggm import utils
from oggm.core import climate
from oggm.core.flowline import FluxBasedModel
from oggm.core.inversion import calving_flux_from_depth


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
    option_ordered : int
        option to keep glaciers ordered or to grab every n value for the batch
        (the latter helps make sure run times on each core are similar as it removes any timing differences caused by
         regional variations)
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
    parser.add_argument('-option_ordered', action='store', type=int, default=1,
                        help='switch to keep lists ordered or not')
    parser.add_argument('-progress_bar', action='store', type=int, default=0,
                        help='Boolean for the progress bar to turn it on or off (default 0 is off)')
    parser.add_argument('-debug', action='store', type=int, default=0,
                        help='Boolean for debugging to turn it on or off (default 0 is off)')
    parser.add_argument('-rgi_glac_number', action='store', type=str, default=None,
                        help='rgi glacier number for supercomputer')
    return parser


#%%
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
    for year in years:
        mbmod.get_annual_mb(fls[0].surface_h, fls=fls, fl_id=0, year=year, debug=False)
    
    # Option for must melt condition
    if return_tbias_mustmelt:
        # Number of years and bins with negative climatic mass balance
        nbinyears_negmbclim =  len(np.where(mbmod.glac_bin_massbalclim_annual < 0)[0])
        return nbinyears_negmbclim
    # Otherwise return specific mass balance
    else:        
        # Specific mass balance [mwea]
        t1_idx = gdir.mbdata['t1_idx']
        t2_idx = gdir.mbdata['t2_idx']
        nyears = gdir.mbdata['nyears']
        mb_mwea = mbmod.glac_wide_massbaltotal[t1_idx:t2_idx+1].sum() / mbmod.glac_wide_area_annual[0] / nyears        
        return mb_mwea
    
    
def retrieve_tbias_bnds(gdir, modelprms, glacier_rgi_table, fls=None, debug=False):
    """
    Calculate parameters for prior distributions for the MCMC analysis
    """
    # Maximum mass loss [mwea] (based on consensus ice thickness estimate)
    with open(gdir.get_filepath('consensus_mass'), 'rb') as f:
        consensus_mass = pickle.load(f)
    mb_max_loss = -1 * consensus_mass / pygem_prms.density_water / gdir.rgi_area_m2 / (gdir.dates_table.shape[0] / 12)

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
def main(list_packed_vars):
    """
    Model simulation

    Parameters
    ----------
    list_packed_vars : list
        list of packed variables that enable the use of parallels

    Returns
    -------
    netcdf files of the simulation output (specific output is dependent on the output option)
    """
    # Unpack variables
    count = list_packed_vars[0]
    glac_no = list_packed_vars[1]
    gcm_name = list_packed_vars[2]

    parser = getparser()
    args = parser.parse_args()
    time_start = time.time()

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
    assert gcm_name in ['ERA5', 'ERA-Interim'], 'Error: Calibration not set up for ' + gcm_name
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
    

    # ===== LOOP THROUGH GLACIERS TO RUN CALIBRATION =====
    for glac in range(main_glac_rgi.shape[0]):
        if glac == 0 or glac == main_glac_rgi.shape[0]:
            print(gcm_name,':', main_glac_rgi.loc[main_glac_rgi.index.values[glac],'RGIId'])
        # Select subsets of data
        glacier_rgi_table = main_glac_rgi.loc[main_glac_rgi.index.values[glac], :]
        glacier_str = '{0:0.5f}'.format(glacier_rgi_table['RGIId_float'])

        # ===== Load glacier data: area (km2), ice thickness (m), width (km) =====
        if glacier_rgi_table['TermType'] == 1:
            gdir = single_flowline_glacier_directory_with_calving(glacier_str)
        else:
            gdir = single_flowline_glacier_directory(glacier_str)
        fls = gdir.read_pickle('inversion_flowlines')

        # Add climate data to glacier directory
        gdir.historical_climate = {'elev': gcm_elev[glac],
                                   'temp': gcm_temp[glac,:],
                                   'tempstd': gcm_tempstd[glac,:],
                                   'prec': gcm_prec[glac,:],
                                   'lr': gcm_lr[glac,:]}
        gdir.dates_table = dates_table
        
        
        # ----- Calibration data -----
        try:
            mbdata_fn = gdir.get_filepath('mb_obs')
            with open(mbdata_fn, 'rb') as f:
                gdir.mbdata = pickle.load(f)
            # Load data
            mb_obs_mwea = gdir.mbdata['mb_mwea']
            mb_obs_mwea_err = gdir.mbdata['mb_mwea_err']
            # Add time indices consistent with dates_table for mb calculations
            t1_year = gdir.mbdata['t1_datetime'].year
            t1_month = gdir.mbdata['t1_datetime'].month
            t2_year = gdir.mbdata['t2_datetime'].year
            t2_month = gdir.mbdata['t2_datetime'].month
            t1_idx = dates_table[(t1_year == dates_table['year']) & (t1_month == dates_table['month'])].index.values[0]
            t2_idx = dates_table[(t2_year == dates_table['year']) & (t2_month == dates_table['month'])].index.values[0]
            # Record indices
            gdir.mbdata['t1_idx'] = t1_idx
            gdir.mbdata['t2_idx'] = t2_idx
            
            if debug:
                print('  mb_data (mwea): ' + str(np.round(mb_obs_mwea,2)) + ' +/- ' + str(np.round(mb_obs_mwea_err,2)))

        except:
            gdir.mbdata = None
            
        
        # ----- Mass balance model ------
        glacier_area = fls[0].widths_m * fls[0].dx_meter
        if glacier_area.sum() > 0 and gdir.mbdata is not None:
            
            modelprms = {'kp': pygem_prms.kp,
                         'tbias': pygem_prms.tbias,
                         'ddfsnow': pygem_prms.ddfsnow,
                         'ddfice': pygem_prms.ddfice,
                         'tsnow_threshold': pygem_prms.tsnow_threshold,
                         'precgrad': pygem_prms.precgrad}
            
            
            # ===== CALIBRATION OPTIONS =====
            # 'MCMC' : use MCMC method to determine posterior probability distributions of the three parameters tbias,
            #          ddfsnow and kp. Then create an ensemble of parameter sets evenly sampled from these
            #          distributions, and output these sets of parameters and their corresponding mass balances to be
            #          used in the simulations.
            if pygem_prms.option_calibration == 'MCMC':

                # ===== Define functions needed for MCMC method =====
                def run_MCMC(gdir,
                             kp_disttype=pygem_prms.kp_disttype,
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
                # Prior distribution (specified or informed by regions)
                # Precipitation factor prior
                if pygem_prms.kp_gamma_region_dict_fullfn is not None:
                    kp_gamma_alpha = pygem_prms.kp_gamma_region_dict[glacier_rgi_table.loc['region']][0]
                    kp_gamma_beta = pygem_prms.kp_gamma_region_dict[glacier_rgi_table.loc['region']][1]
                else:
                    kp_gamma_alpha = pygem_prms.kp_gamma_alpha
                    kp_gamma_beta = pygem_prms.kp_gamma_beta
                # Temperature bias prior
                if pygem_prms.tbias_norm_region_dict_fullfn is not None:
                    tbias_mu = pygem_prms.tbias_norm_region_dict[glacier_rgi_table.loc['region']][0]
                    tbias_sigma = pygem_prms.tbias_norm_region_dict[glacier_rgi_table.loc['region']][1]
                else:
                    tbias_mu = pygem_prms.tbias_mu
                    tbias_sigma = pygem_prms.tbias_sigma

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
                    tbias_bndlow, tbias_bndhigh, mb_max_loss = retrieve_tbias_bnds(
                            gdir, modelprms, glacier_rgi_table, fls, debug=False)

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
                            gdir,
                            iterations=pygem_prms.mcmc_sample_no, burn=pygem_prms.mcmc_burn_no,
                            step=pygem_prms.mcmc_step,
                            kp_gamma_alpha=kp_gamma_alpha, kp_gamma_beta=kp_gamma_beta, kp_start=kp_start,
                            tbias_mu=tbias_mu, tbias_sigma=tbias_sigma, tbias_start=tbias_start,
                            ddfsnow_start=ddfsnow_start, mb_max_loss=mb_max_loss,
                            use_potentials=True)

                    if debug:
                        print('\nacceptance ratio:', model.step_method_dict[next(iter(model.stochastics))][0].ratio)
                        print('mb_mwea_mean:', np.round(np.mean(model.trace('massbal')[:]),3),
                              'mb_mwea_std:', np.round(np.std(model.trace('massbal')[:]),3),
                              '\nmb_obs_mean:', np.round(mb_obs_mwea,3), 'mb_obs_std:', np.round(mb_obs_mwea_err,3))


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
            if pygem_prms.option_calibration == 'HH2015':
                tbias_init = pygem_prms.tbias_init
                tbias_step = pygem_prms.tbias_step
                kp_init = pygem_prms.kp_init
                kp_bndlow = pygem_prms.kp_bndlow
                kp_bndhigh = pygem_prms.kp_bndhigh
                ddfsnow_init = pygem_prms.ddfsnow_init
                ddfsnow_bndlow = pygem_prms.ddfsnow_bndlow
                ddfsnow_bndhigh = pygem_prms.ddfsnow_bndhigh
                assert pygem_prms.ddfsnow_iceratio == 0.5, 'Error: ddfsnow_iceratio for HH2015 must be 0.5'

                # ----- Initialize model parameters -----
                modelprms['tbias'] = tbias_init
                modelprms['kp'] = kp_init
                modelprms['ddfsnow'] = ddfsnow_init
                modelprms['ddfice'] = modelprms['ddfsnow'] / pygem_prms.ddfsnow_iceratio
                continue_param_search = True
                
                # ----- FUNCTIONS: COMPUTATIONALLY FASTER AND MORE ROBUST THAN SCIPY MINIMIZE -----
                def update_bnds(prm2opt, prm_bndlow, prm_bndhigh, prm_mid, mb_mwea_low, mb_mwea_high, mb_mwea_mid,
                                debug=False):
                    # If mass balance less than observation, reduce tbias
                    if prm2opt == 'kp':
                        if mb_mwea_mid < mb_obs_mwea:
                            prm_bndlow_new, mb_mwea_low_new = prm_mid, mb_mwea_mid
                            prm_bndhigh_new, mb_mwea_high_new = prm_bndhigh, mb_mwea_high
                        else:
                            prm_bndlow_new, mb_mwea_low_new = prm_bndlow, mb_mwea_low
                            prm_bndhigh_new, mb_mwea_high_new = prm_mid, mb_mwea_mid
                    elif prm2opt == 'ddfsnow':
                        if mb_mwea_mid < mb_obs_mwea:
                            prm_bndlow_new, mb_mwea_low_new = prm_bndlow, mb_mwea_low
                            prm_bndhigh_new, mb_mwea_high_new = prm_mid, mb_mwea_mid
                        else:
                            prm_bndlow_new, mb_mwea_low_new = prm_mid, mb_mwea_mid
                            prm_bndhigh_new, mb_mwea_high_new = prm_bndhigh, mb_mwea_high
                    elif prm2opt == 'tbias':
                        if mb_mwea_mid < mb_obs_mwea:
                            prm_bndlow_new, mb_mwea_low_new = prm_bndlow, mb_mwea_low
                            prm_bndhigh_new, mb_mwea_high_new = prm_mid, mb_mwea_mid
                        else:
                            prm_bndlow_new, mb_mwea_low_new = prm_mid, mb_mwea_mid
                            prm_bndhigh_new, mb_mwea_high_new = prm_bndhigh, mb_mwea_high
                    
                    prm_mid_new = (prm_bndlow_new + prm_bndhigh_new) / 2
                    modelprms[prm2opt] = prm_mid_new
                    modelprms['ddfice'] = modelprms['ddfsnow'] / pygem_prms.ddfsnow_iceratio
                    mb_mwea_mid_new = mb_mwea_calc(gdir, modelprms, glacier_rgi_table, fls=fls)

                    if debug:
                        print(prm2opt + '_bndlow:', np.round(prm_bndlow_new,2), 
                              'mb_mwea_low:', np.round(mb_mwea_low_new,2))                        
                        print(prm2opt + '_bndhigh:', np.round(prm_bndhigh_new,2), 
                              'mb_mwea_high:', np.round(mb_mwea_high_new,2))                        
                        print(prm2opt + '_mid:', np.round(prm_mid_new,2), 
                              'mb_mwea_mid:', np.round(mb_mwea_mid_new,3))
                    
                    return (prm_bndlow_new, prm_bndhigh_new, prm_mid_new, 
                            mb_mwea_low_new, mb_mwea_high_new, mb_mwea_mid_new)
                    
                
                def single_param_optimizer(modelprms_subset, mb_obs_mwea, prm2opt=None,
                                           kp_bnds=None, tbias_bnds=None, ddfsnow_bnds=None,
                                           mb_mwea_threshold=0.005, debug=False):
                    assert prm2opt is not None, 'For single_param_optimizer you must specify parameter to optimize'
               
                    if prm2opt == 'kp':
                        prm_bndlow = kp_bnds[0]
                        prm_bndhigh = kp_bnds[1]
                        modelprms['tbias'] = modelprms_subset['tbias']
                        modelprms['ddfsnow'] = modelprms_subset['ddfsnow']
                    elif prm2opt == 'ddfsnow':
                        prm_bndlow = ddfsnow_bnds[0]
                        prm_bndhigh = ddfsnow_bnds[1]
                        modelprms['kp'] = modelprms_subset['kp']
                        modelprms['tbias'] = modelprms_subset['tbias']
                    elif prm2opt == 'tbias':
                        prm_bndlow = tbias_bnds[0]
                        prm_bndhigh = tbias_bnds[1]
                        modelprms['kp'] = modelprms_subset['kp']
                        modelprms['ddfsnow'] = modelprms_subset['ddfsnow']

                    # Lower bound
                    modelprms[prm2opt] = prm_bndlow
                    modelprms['ddfice'] = modelprms['ddfsnow'] / pygem_prms.ddfsnow_iceratio
                    mb_mwea_low = mb_mwea_calc(gdir, modelprms, glacier_rgi_table, fls=fls)
                    # Upper bound
                    modelprms[prm2opt] = prm_bndhigh
                    modelprms['ddfice'] = modelprms['ddfsnow'] / pygem_prms.ddfsnow_iceratio
                    mb_mwea_high = mb_mwea_calc(gdir, modelprms, glacier_rgi_table, fls=fls)
                    # Middle bound
                    prm_mid = (prm_bndlow + prm_bndhigh) / 2
                    modelprms[prm2opt] = prm_mid
                    modelprms['ddfice'] = modelprms['ddfsnow'] / pygem_prms.ddfsnow_iceratio
                    mb_mwea_mid = mb_mwea_calc(gdir, modelprms, glacier_rgi_table, fls=fls)
                    
                    if debug:
                        print(prm2opt + '_bndlow:', np.round(prm_bndlow,2), 'mb_mwea_low:', np.round(mb_mwea_low,2))
                        print(prm2opt + '_bndhigh:', np.round(prm_bndhigh,2), 'mb_mwea_high:', np.round(mb_mwea_high,2))
                        print(prm2opt + '_mid:', np.round(prm_mid,2), 'mb_mwea_mid:', np.round(mb_mwea_mid,3))
                    
                    # Optimize the model parameter
                    if np.absolute(mb_mwea_low - mb_obs_mwea) <= mb_mwea_threshold:
                        modelprms[prm2opt] = prm_bndlow
                        mb_mwea_mid = mb_mwea_low
                    elif np.absolute(mb_mwea_low - mb_obs_mwea) <= mb_mwea_threshold:
                        modelprms[prm2opt] = prm_bndhigh
                        mb_mwea_mid = mb_mwea_high
                    else:
                        ncount = 0
                        while np.absolute(mb_mwea_mid - mb_obs_mwea) > mb_mwea_threshold:
                            if debug:
                                print('\n ncount:', ncount)
                            (prm_bndlow, prm_bndhigh, prm_mid, mb_mwea_low, mb_mwea_high, mb_mwea_mid) = (
                                    update_bnds(prm2opt, prm_bndlow, prm_bndhigh, prm_mid, 
                                                mb_mwea_low, mb_mwea_high, mb_mwea_mid, debug=debug))
                            ncount += 1
                        
                    return modelprms, mb_mwea_mid
                
                
                # ===== ROUND 1: PRECIPITATION FACTOR ======
                if debug:
                    print('Round 1:')
                    
                if debug:
                    print(glacier_str + '  kp: ' + str(np.round(modelprms['kp'],2)) +
                          ' ddfsnow: ' + str(np.round(modelprms['ddfsnow'],4)) +
                          ' tbias: ' + str(np.round(modelprms['tbias'],2)))
                
                # Lower bound
                modelprms['kp'] = kp_bndlow
                mb_mwea_kp_low = mb_mwea_calc(gdir, modelprms, glacier_rgi_table, fls=fls)
                # Upper bound
                modelprms['kp'] = kp_bndhigh
                mb_mwea_kp_high = mb_mwea_calc(gdir, modelprms, glacier_rgi_table, fls=fls)
                
                # Optimal precipitation factor
                if mb_obs_mwea < mb_mwea_kp_low:
                    kp_opt = kp_bndlow
                    mb_mwea = mb_mwea_kp_low
                elif mb_obs_mwea > mb_mwea_kp_high:
                    kp_opt = kp_bndhigh
                    mb_mwea = mb_mwea_kp_high
                else:
                    # Single parameter optimizer (computationally more efficient and less prone to fail)
                    modelprms_subset = {'kp':kp_init, 'ddfsnow': ddfsnow_init, 'tbias': tbias_init}
                    kp_bnds = (kp_bndlow, kp_bndhigh)
                    modelprms_opt, mb_mwea = single_param_optimizer(
                            modelprms_subset, mb_obs_mwea, prm2opt='kp', kp_bnds=kp_bnds, debug=True)
                    kp_opt = modelprms_opt['kp']
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
                    # Optimal degree-day factor of snow
                    if mb_obs_mwea < mb_mwea_ddfhigh:
                        ddfsnow_opt = ddfsnow_bndhigh
                        mb_mwea = mb_mwea_ddfhigh
                    elif mb_obs_mwea > mb_mwea_ddflow:
                        ddfsnow_opt = ddfsnow_bndlow
                        mb_mwea = mb_mwea_ddflow
                    else:
                        # Single parameter optimizer (computationally more efficient and less prone to fail)
                        modelprms_subset = {'kp':kp_opt, 'ddfsnow': ddfsnow_init, 'tbias': tbias_init}
                        ddfsnow_bnds = (ddfsnow_bndlow, ddfsnow_bndhigh)
                        modelprms_opt, mb_mwea = single_param_optimizer(
                                modelprms_subset, mb_obs_mwea, prm2opt='ddfsnow', ddfsnow_bnds=ddfsnow_bnds, debug=True)
                        ddfsnow_opt = modelprms_opt['ddfsnow']
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

                    # Single parameter optimizer (computationally more efficient and less prone to fail)
                    modelprms_subset = {'kp':kp_opt, 
                                        'ddfsnow': ddfsnow_opt, 
                                        'tbias': modelprms['tbias'] - tbias_step/2}
                    tbias_bnds = (tbias_bndhigh-tbias_step, tbias_bndhigh)
                    modelprms_opt, mb_mwea = single_param_optimizer(
                            modelprms_subset, mb_obs_mwea, prm2opt='tbias', tbias_bnds=tbias_bnds, debug=True)
                    
                    # Update parameter values
                    tbias_opt = modelprms_opt['tbias']
                    modelprms['tbias'] = tbias_opt
                    if debug:
                        print('  tbias:', np.round(tbias_opt,3), 'mb_mwea:', np.round(mb_mwea,3))

                else:
                    tbias_opt = modelprms['tbias']
                
                    
                # Export model parameters
                modelprms = modelprms_opt
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
            elif pygem_prms.option_calibration == 'HH2015mod':
                tbias_init = pygem_prms.tbias_init
                tbias_step = pygem_prms.tbias_step
                kp_init = pygem_prms.kp_init
                kp_bndlow = pygem_prms.kp_bndlow
                kp_bndhigh = pygem_prms.kp_bndhigh
                ddfsnow_init = pygem_prms.ddfsnow_init
                
                # ----- Initialize model parameters -----
                modelprms['tbias'] = tbias_init
                modelprms['kp'] = kp_init
                modelprms['ddfsnow'] = ddfsnow_init
                modelprms['ddfice'] = modelprms['ddfsnow'] / pygem_prms.ddfsnow_iceratio
                
                # ----- FUNCTIONS -----
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

                # ===== ROUND 1: PRECIPITATION FACTOR =====
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

                # ----- RUN OPTIMIZATION WITH CONSTRAINED BOUNDS -----
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
        
            
        
#%%

#        glacier_area_km2 = fls[0].widths_m * fls[0].dx_meter / 1e6
#        if glacier_area_km2.sum() > 0:
##            if pygem_prms.hindcast == 1:
##                glacier_gcm_prec = glacier_gcm_prec[::-1]
##                glacier_gcm_temp = glacier_gcm_temp[::-1]
##                glacier_gcm_lrgcm = glacier_gcm_lrgcm[::-1]
##                glacier_gcm_lrglac = glacier_gcm_lrglac[::-1]
#
#            # Load model parameters
#            if pygem_prms.use_calibrated_modelparams:
#                with open(gdir.get_filepath('pygem_modelprms'), 'rb') as f:
#                    modelprms_dict = pickle.load(f)
#
#                assert pygem_prms.option_calibration in modelprms_dict, ('Error: ' + pygem_prms.option_calibration +
#                                                                         ' not in modelprms_dict')
#                modelprms_all = modelprms_dict[pygem_prms.option_calibration]
#                # MCMC needs model parameters to be selected
#                if pygem_prms.option_calibration == 'MCMC':
#                    sim_iters = pygem_prms.sim_iters
#                    if sim_iters == 1:
#                        modelprms_all = {'kp': [np.median(modelprms_all['kp']['chain_0'])],
#                                         'tbias': [np.median(modelprms_all['tbias']['chain_0'])],
#                                         'ddfsnow': [np.median(modelprms_all['ddfsnow']['chain_0'])],
#                                         'ddfice': [np.median(modelprms_all['ddfice']['chain_0'])],
#                                         'tsnow_threshold': modelprms_all['tsnow_threshold'],
#                                         'precgrad': modelprms_all['precgrad']}
#                    else:
#                        # Select every kth iteration to use for the ensemble
#                        mcmc_sample_no = len(modelprms_all['kp']['chain_0'])
#                        mp_spacing = int((mcmc_sample_no - pygem_prms.sim_burn) / sim_iters)
#                        mp_idx_start = np.arange(pygem_prms.sim_burn, pygem_prms.sim_burn + mp_spacing)
#                        np.random.shuffle(mp_idx_start)
#                        mp_idx_start = mp_idx_start[0]
#                        mp_idx_all = np.arange(mp_idx_start, mcmc_sample_no, mp_spacing)
#                        modelprms_all = {
#                                'kp': [modelprms_all['kp']['chain_0'][mp_idx] for mp_idx in mp_idx_all],
#                                'tbias': [modelprms_all['tbias']['chain_0'][mp_idx] for mp_idx in mp_idx_all],
#                                'ddfsnow': [modelprms_all['ddfsnow']['chain_0'][mp_idx] for mp_idx in mp_idx_all],
#                                'ddfice': [modelprms_all['ddfice']['chain_0'][mp_idx] for mp_idx in mp_idx_all],
#                                'tsnow_threshold': modelprms_all['tsnow_threshold'] * sim_iters,
#                                'precgrad': modelprms_all['precgrad'] * sim_iters}
#                else:
#                    sim_iters = 1
#            else:
#                modelprms_all = {'kp': [pygem_prms.kp],
#                                 'tbias': [pygem_prms.tbias],
#                                 'ddfsnow': [pygem_prms.ddfsnow],
#                                 'ddfice': [pygem_prms.ddfice],
#                                 'tsnow_threshold': [pygem_prms.tsnow_threshold],
#                                 'precgrad': [pygem_prms.precgrad]}
#
#            # Loop through model parameters
#            for n_iter in range(sim_iters):
#
#                modelprms = {'kp': modelprms_all['kp'][n_iter],
#                             'tbias': modelprms_all['tbias'][n_iter],
#                             'ddfsnow': modelprms_all['ddfsnow'][n_iter],
#                             'ddfice': modelprms_all['ddfice'][n_iter],
#                             'tsnow_threshold': modelprms_all['tsnow_threshold'][n_iter],
#                             'precgrad': modelprms_all['precgrad'][n_iter]}
#
#                if debug:
#                    print(glacier_str + '  kp: ' + str(np.round(modelprms['kp'],2)) +
#                          ' ddfsnow: ' + str(np.round(modelprms['ddfsnow'],4)) +
#                          ' tbias: ' + str(np.round(modelprms['tbias'],2)))
#
#                # OGGM WANTS THIS FUNCTION TO SIMPLY RETURN THE MASS BALANCE AS A FUNCTION OF HEIGHT AND THAT'S IT
#                mbmod = PyGEMMassBalance(gdir, modelprms, glacier_rgi_table,
#                                         hindcast=pygem_prms.hindcast,
#                                         debug=pygem_prms.debug_mb,
#                                         debug_refreeze=pygem_prms.debug_refreeze,
#                                         fls=fls, option_areaconstant=True)
#
#                # ----- MODEL RUN WITH CONSTANT GLACIER AREA -----
#                years = np.arange(pygem_prms.gcm_startyear, pygem_prms.gcm_endyear + 1)
#                mb_all = []
#                for fl_id, fl in enumerate(fls):
#                    for year in years - years[0]:
#                        mb_annual = mbmod.get_annual_mb(fls[0].surface_h, fls=fls, fl_id=fl_id, year=year,
#                                                        debug=True)
#                        mb_mwea = (mb_annual * 365 * 24 * 3600 * pygem_prms.density_ice /
#                                       pygem_prms.density_water)
#                        glac_wide_mb_mwea = ((mb_mwea * mbmod.glacier_area_initial).sum() /
#                                              mbmod.glacier_area_initial.sum())
#                        print('year:', year, np.round(glac_wide_mb_mwea,3))
#
#                        mb_all.append(glac_wide_mb_mwea)
#
#                print('iter:', n_iter, 'massbal (mean, std):', np.round(np.mean(mb_all),3), np.round(np.std(mb_all),3))
#
#
#                #%%
#                # ----- MODEL THAT IS NOT CONSTANT AREA (INVERSION AND OGGM DYNAMICS) -----
###                # Make a working copy of the glacier directory
###                tmp_dir = os.path.join(cfg.PATHS['working_dir'], 'tmp_dir')
###                utils.mkdir(tmp_dir, reset=True)
###                # new glacier directory is copy of old directory (copy everything)
###                ngd = utils.copy_to_basedir(gdir, base_dir=tmp_dir, setup='all')
##
##                # Perform inversion based on PyGEM MB
##                # Add thickness, width_m, and dx_meter to inversion flowlines so they are compatible with PyGEM's
##                #  mass balance model (necessary because OGGM's inversion flowlines use pixel distances; however,
##                #  this will likely be rectified in the future)
###                    def apply_on_fls(fls):
###                        for fl in fls:
###                            fl.widths_m = fl.widths * gd.grid.dx
###                            fl.dx_meter = fl.dx * gd.grid.dx
###                        return fls
###                    fls_inv = apply_on_fls(ngd.read_pickle('inversion_flowlines'))
##
##                fls_inv = gdir.read_pickle('inversion_flowlines')
##                mbmod_inv = PyGEMMassBalance(gdir, modelprms, glacier_rgi_table,
##                                             hindcast=pygem_prms.hindcast,
##                                             debug=pygem_prms.debug_mb,
##                                             debug_refreeze=pygem_prms.debug_refreeze,
##                                             fls=fls, option_areaconstant=True)
###                # Inversion with OGGM model showing the diffences in the mass balance vs. altitude relationships,
###                #  which is going to drive different ice thickness estimates
###                from oggm.core.massbalance import PastMassBalance
###                oggm_mod = PastMassBalance(gdir)
###                h, w = gdir.get_inversion_flowline_hw()
###                oggm = oggm_mod.get_annual_mb(h, year=2000) * cfg.SEC_IN_YEAR * 900
###                pyg  = mbmod_inv.get_annual_mb(h, year=0, fl_id=0, fls=fls_inv) * cfg.SEC_IN_YEAR * 900
###                plt.plot(oggm, h, '.')
###                plt.plot(pyg, h, '.')
###                plt.show()
##                # Want to reset model parameters
##                climate.apparent_mb_from_any_mb(gdir, mb_model=mbmod_inv, mb_years=np.arange(18),
###                                                    apply_on_fls=apply_on_fls
##                                                )
##                tasks.prepare_for_inversion(gdir)
##                print('setting model parameters here')
###                    fs = 5.7e-20
##                fs = 0
##                glen_a_multiplier = 1
##                tasks.mass_conservation_inversion(gdir, glen_a=cfg.PARAMS['glen_a']*glen_a_multiplier, fs=fs)
##                tasks.filter_inversion_output(gdir)
##                tasks.init_present_time_glacier(gdir) # adds bins below (can use thhe 136 on-glacier for offglac calcs)
##
##                nfls = gdir.read_pickle('model_flowlines')
##                mbmod = PyGEMMassBalance(gdir, modelprms, glacier_rgi_table,
##                                         hindcast=pygem_prms.hindcast,
##                                         debug=pygem_prms.debug_mb,
##                                         debug_refreeze=pygem_prms.debug_refreeze,
##                                         fls=fls, option_areaconstant=True)
##                # Model parameters
##                #  fs=5.7e-20
##                #  glen_a=cfg.PARAMS['glen_a'], OGGM * 1.3 for Farinotti2019: glen_a=cfg.PARAMS['glen_a']*1.3
##                old_model = FluxBasedModel(fls, y0=0, mb_model=mbmod)
##                ev_model = FluxBasedModel(nfls, y0=0, mb_model=mbmod, glen_a=cfg.PARAMS['glen_a']*glen_a_multiplier,
##                                          fs=fs)
##
##                print('Old glacier vol', old_model.volume_km3)
##                print('New glacier vol', ev_model.volume_km3)
##
##                graphics.plot_modeloutput_section(ev_model)
##                _, diag = ev_model.run_until_and_store(400)
##                graphics.plot_modeloutput_section(ev_model)
##                plt.figure()
##                diag.volume_m3.plot()
##                # plt.figure()
##                # diag.area_m2.plot()
##                plt.show()
#
#
#
#                #%% ===== Adding functionality for calving =====
#                water_level = None
#                if glacier_rgi_table['TermType'] == 1:
##                    # Calving params - default is 2.4, which is quite high
##                    cfg.PARAMS['inversion_calving_k'] = 1
##                    # Find out the calving values
##                    calving_output = tasks.find_inversion_calving(gdir)
##                    # Get ready
##                    tasks.init_present_time_glacier(gdir)
#
#                    #%%
#                    # FluxBasedMODEL
##                    for flux_gate in [0.06, 0.10, 0.16]:
##                        model = FluxBasedModel(bu_tidewater_bed(), mb_model=mbmod,
##                                               is_tidewater=True,
##                                               flux_gate=flux_gate,  # default is 0
##                                               calving_k=0.2,  # default is 2.4
##                                              )
#
#
##                    old_model = FluxBasedModel(fls, y0=0, mb_model=mbmod)
##                    ev_model = FluxBasedModel(nfls, y0=0, mb_model=mbmod, glen_a=cfg.PARAMS['glen_a']*glen_a_multiplier,
##                                              fs=fs)
#
#                    # ----- MY VERSION OF FLUXBASEDMODEL W FRONTAL ABLATION -----
#                    # THIS REQUIRES ICE THICKNESS ESTIMATE!
##                    assert True == False, 'Need ice thickness inversion for this code'
##                    to_plot = None
##                    keys = []
##                    for flux_gate in [0, 0.1]:
##                        model = FluxBasedModel(fls, mb_model=mbmod, is_tidewater=True,
##                                               flux_gate=flux_gate,  # default is 0
##                                               calving_k=0.2,  # default is 2.4
##                                              )
##
##                        # long enough to reach approx. equilibrium
##                        _, ds = model.run_until_and_store(6000)
##                        df_diag = model.get_diagnostics()
##
##                        if to_plot is None:
##                            to_plot = df_diag
##
##                        key = 'Flux gate={:.02f}. Calving rate: {:.0f} m yr-1'.format(flux_gate, model.calving_rate_myr)
##                        to_plot[key] = df_diag['surface_h']
##                        keys.append(key)
##
##                        # Plot of volume
##                        (ds.volume_m3 * 1e-9).plot(label=key);
##                    plt.legend(); plt.ylabel('Volume [km$^{3}$]');
###                    to_plot.index = xc
#
#
#
#                    #%%
#
#                    # Calving flux (km3 yr-1; positive value refers to amount of calving, need to subtract from mb)
#                    for calving_k in [1]:
#                        # Calving parameter
#                        cfg.PARAMS['calving_k'] = calving_k
#                        cfg.PARAMS['inversion_calving_k'] = cfg.PARAMS['calving_k']
#
#
#                        # INVERSION TO RUN
#                        # Find out the calving values
#                        out_calving = tasks.find_inversion_calving(gdir)
#                        # Get ready
#                        tasks.init_present_time_glacier(gdir)
#                        # print output
#                        print(out_calving)
#
#                        if debug:
#                            fl = gdir.read_pickle('model_flowlines')[-1]
#                            xc = fl.dis_on_line * fl.dx_meter / 1000
#                            plt.plot(xc, fl.surface_h, '-', color='C1', label='Surface')
#                            plt.plot(xc, gdir.read_pickle('model_flowlines')[-1].bed_h, '--', color='k',
#                                     label='Glacier bed (with calving)')
#                            plt.hlines(0, 0, xc[-1], color='C0', linestyle=':'), plt.legend()
#                            plt.show()
#
#                        print(cfg.PARAMS['calving_front_slope'])
#
#
#
##                        assert True == False, 'Need ice thickness inversion for this code'
#                        to_plot = None
#                        keys = []
#                        fls = gdir.read_pickle('model_flowlines')
#                        for flux_gate in [0, 0.1]:
#                            model = FluxBasedModel(fls, mb_model=mbmod, is_tidewater=True,
#                                                   flux_gate=flux_gate,  # default is 0
#                                                   calving_k=0.2,  # default is 2.4
#                                                  )
#
#                            # long enough to reach approx. equilibrium
#                            _, ds = model.run_until_and_store(6000)
#                            df_diag = model.get_diagnostics()
#
#                            if to_plot is None:
#                                to_plot = df_diag
#
#                            key = 'Flux gate={:.02f}. Calving rate: {:.0f} m yr-1'.format(flux_gate, model.calving_rate_myr)
#                            to_plot[key] = df_diag['surface_h']
#                            keys.append(key)
#
#                            # Plot of volume
#                            (ds.volume_m3 * 1e-9).plot(label=key);
#                        plt.legend(); plt.ylabel('Volume [km$^{3}$]');
#                        plt.show()
#
##                        model_wc = tasks.run_constant_climate(gdir, y0=2000, nyears=100)
#                        # switch to tasks.run_from_climate_data
#
#
##                        # Water level (max based on 50 m freeboard)
##                        if water_level is None:
##                            th = fls[-1].surface_h[-1]
##                            vmin, vmax = cfg.PARAMS['free_board_marine_terminating']
##                            water_level = utils.clip_scalar(0, th - vmax, th - vmin)
##
##
##
##                        calving_output = calving_flux_from_depth(gdir, water_level=water_level)
##                        calving_flux = calving_output['flux']
##
##                        print('\n calving_k:', calving_k)
##                        print('  calving flux (km3 yr-1):', calving_flux)
##                        print('  freeboard:', calving_output['free_board'])
##                        print('  water_level:', calving_output['water_level'])
#
#
#                #%%
##
#
#                # ===== END INVERSION AND MODEL RUN WITH OGGM DYNAMICS =====                #%%
#
##                print('\nrunning mass redistribution model...')
##                model = MassRedistributionCurveModel(fls, mb_model=mbmod, y0=0)
##                model.run_until(pygem_prms.gcm_endyear - pygem_prms.gcm_startyear)
##                model.run_until(1)
#
#                if args.option_parallels == 0:
#                    print('\nTO-DO LIST:')
#                    print(' - add frontal ablation to be removed in mass redistribution curves glacierdynamics')
#                    print(' - switch daily_std ablation to a gaussian to ensure repeatability without seeding')
#                    print(' - export netcdf files')
#                    print(' - update supercomputer environment to ensure code still runs on spc')
#                    print('    --> get set up with Pittsburgh Supercomputing Center')
#                    print(' - make two refreeze (potential?) options stand-alone functions like frontal ablation')
#                    print(' - climate data likely mismatch with OGGM, e.g., prec in m for the month')
#
#                # # RECORD PARAMETERS TO DATASET
#                # if pygem_prms.output_package == 2:
#                #     output_temp_glac_monthly[:, n_iter] = mbmod.glac_wide_temp
#                #     output_prec_glac_monthly[:, n_iter] = mbmod.glac_wide_prec
#                #     output_acc_glac_monthly[:, n_iter] = mbmod.glac_wide_acc
#                #     output_refreeze_glac_monthly[:, n_iter] = mbmod.glac_wide_refreeze
#                #     output_melt_glac_monthly[:, n_iter] = mbmod.glac_wide_melt
#                #     output_frontalablation_glac_monthly[:, n_iter] = mbmod.glac_wide_frontalablation
#                #     output_massbaltotal_glac_monthly[:, n_iter] = mbmod.glac_wide_massbaltotal
#                #     output_runoff_glac_monthly[:, n_iter] = mbmod.glac_wide_runoff
#                #     output_snowline_glac_monthly[:, n_iter] = mbmod.glac_wide_snowline
#                #     output_area_glac_annual[:, n_iter] = mbmod.glac_wide_area_annual
#                #     output_volume_glac_annual[:, n_iter] = mbmod.glac_wide_volume_annual
#                #     output_ELA_glac_annual[:, n_iter] = mbmod.glac_wide_ELA_annual
#                #     output_offglac_prec_monthly[:, n_iter] = mbmod.offglac_wide_prec
#                #     output_offglac_refreeze_monthly[:, n_iter] = mbmod.offglac_wide_refreeze
#                #     output_offglac_melt_monthly[:, n_iter] = mbmod.offglac_wide_melt
#                #     output_offglac_snowpack_monthly[:, n_iter] = mbmod.offglac_wide_snowpack
#                #     output_offglac_runoff_monthly[:, n_iter] = mbmod.offglac_wide_runoff
#
##            # ===== Export Results =====
##            output_ds, encoding = create_xrdataset(glacier_rgi_table, dates_table, record_stats=1,
##                                                       option_wateryear=pygem_prms.gcm_wateryear)
##            output_ds['glac_temp_monthly'].values = output_temp_glac_monthly.mean(1)[np.newaxis,:] + 273.15
##            output_ds['glac_temp_monthly_std'].values = output_temp_glac_monthly.std(1)[np.newaxis,:]
##            output_ds['glac_prec_monthly'].values = output_prec_glac_monthly.mean(1)[np.newaxis,:]
##            output_ds['glac_prec_monthly_std'].values = output_prec_glac_monthly.std(1)[np.newaxis,:]
##            output_ds['glac_acc_monthly'].values = output_acc_glac_monthly.mean(1)[np.newaxis,:]
##            output_ds['glac_acc_monthly_std'].values = output_acc_glac_monthly.std(1)[np.newaxis,:]
##            output_ds['glac_refreeze_monthly'].values = output_refreeze_glac_monthly.mean(1)[np.newaxis,:]
##            output_ds['glac_refreeze_monthly_std'].values = output_refreeze_glac_monthly.std(1)[np.newaxis,:]
##            output_ds['glac_melt_monthly'].values = output_melt_glac_monthly.mean(1)[np.newaxis,:]
##            output_ds['glac_melt_monthly_std'].values = output_melt_glac_monthly.std(1)[np.newaxis,:]
##            output_ds['glac_frontalablation_monthly'].values = output_frontalablation_glac_monthly.mean(1)[np.newaxis,:]
##            output_ds['glac_frontalablation_monthly_std'].values = (
##                    output_frontalablation_glac_monthly.std(1)[np.newaxis,:])
##            output_ds['glac_massbaltotal_monthly'].values = output_massbaltotal_glac_monthly.mean(1)[np.newaxis,:]
##            output_ds['glac_massbaltotal_monthly_std'].values = output_massbaltotal_glac_monthly.std(1)[np.newaxis,:]
##            output_ds['glac_runoff_monthly'].values = output_runoff_glac_monthly.mean(1)[np.newaxis,:]
##            output_ds['glac_runoff_monthly_std'].values = output_runoff_glac_monthly.std(1)[np.newaxis,:]
##            output_ds['glac_snowline_monthly'].values = output_snowline_glac_monthly.mean(1)[np.newaxis,:]
##            output_ds['glac_snowline_monthly_std'].values = output_snowline_glac_monthly.std(1)[np.newaxis,:]
##            output_ds['glac_area_annual'].values = output_area_glac_annual.mean(1)[np.newaxis,:]
##            output_ds['glac_area_annual_std'].values = output_area_glac_annual.std(1)[np.newaxis,:]
##            output_ds['glac_volume_annual'].values = output_volume_glac_annual.mean(1)[np.newaxis,:]
##            output_ds['glac_volume_annual_std'].values = output_volume_glac_annual.std(1)[np.newaxis,:]
##            output_ds['glac_ELA_annual'].values = output_ELA_glac_annual.mean(1)[np.newaxis,:]
##            output_ds['glac_ELA_annual_std'].values = output_ELA_glac_annual.std(1)[np.newaxis,:]
##            output_ds['offglac_prec_monthly'].values = output_offglac_prec_monthly.mean(1)[np.newaxis,:]
##            output_ds['offglac_prec_monthly_std'].values = output_offglac_prec_monthly.std(1)[np.newaxis,:]
##            output_ds['offglac_refreeze_monthly'].values = output_offglac_refreeze_monthly.mean(1)[np.newaxis,:]
##            output_ds['offglac_refreeze_monthly_std'].values = output_offglac_refreeze_monthly.std(1)[np.newaxis,:]
##            output_ds['offglac_melt_monthly'].values = output_offglac_melt_monthly.mean(1)[np.newaxis,:]
##            output_ds['offglac_melt_monthly_std'].values = output_offglac_melt_monthly.std(1)[np.newaxis,:]
##            output_ds['offglac_snowpack_monthly'].values =  output_offglac_snowpack_monthly.mean(1)[np.newaxis,:]
##            output_ds['offglac_snowpack_monthly_std'].values =  output_offglac_snowpack_monthly.std(1)[np.newaxis,:]
##            output_ds['offglac_runoff_monthly'].values = output_offglac_runoff_monthly.mean(1)[np.newaxis,:]
##            output_ds['offglac_runoff_monthly_std'].values = output_offglac_runoff_monthly.std(1)[np.newaxis,:]
##
##            # Export statistics to netcdf
##            if pygem_prms.output_package == 2:
##                output_sim_fp = pygem_prms.output_sim_fp + gcm_name + '/'
##                if gcm_name not in ['ERA-Interim', 'ERA5', 'COAWST']:
##                    output_sim_fp += rcp_scenario + '/'
##                # Create filepath if it does not exist
##                if os.path.exists(output_sim_fp) == False:
##                    os.makedirs(output_sim_fp)
##                # Netcdf filename
##                if gcm_name in ['ERA-Interim', 'ERA5', 'COAWST']:
##                    # Filename
##                    netcdf_fn = (glacier_str + '_' + gcm_name + '_c' + str(pygem_prms.option_calibration) + '_ba' +
##                                 str(pygem_prms.option_bias_adjustment) + '_' +  str(sim_iters) + 'sets' + '_' +
##                                 str(pygem_prms.gcm_startyear) + '_' + str(pygem_prms.gcm_endyear) + '.nc')
##                else:
##                    netcdf_fn = (glacier_str + '_' + gcm_name + '_' + rcp_scenario + '_c' +
##                                 str(pygem_prms.option_calibration) + '_ba' + str(pygem_prms.option_bias_adjustment) +
##                                 '_' +  str(sim_iters) + 'sets' + '_' + str(pygem_prms.gcm_startyear) + '_' +
##                                 str(pygem_prms.gcm_endyear) + '.nc')
##                if pygem_prms.option_synthetic_sim==1:
##                    netcdf_fn = (netcdf_fn.split('--')[0] + '_T' + str(pygem_prms.synthetic_temp_adjust) + '_P' +
##                                 str(pygem_prms.synthetic_prec_factor) + '--' + netcdf_fn.split('--')[1])
##
##                # Export netcdf
##                output_ds.to_netcdf(output_sim_fp + netcdf_fn, encoding=encoding)
##
##            # Close datasets
##            output_ds.close()


    # Global variables for Spyder development
    if args.option_parallels == 0:
        global main_vars
        main_vars = inspect.currentframe().f_locals


#%% PARALLEL PROCESSING
if __name__ == '__main__':
    time_start = time.time()
    parser = getparser()
    args = parser.parse_args()

    if args.debug == 1:
        debug = True
    else:
        debug = False

    if not 'pygem_modelprms' in cfg.BASENAMES:
        cfg.BASENAMES['pygem_modelprms'] = ('pygem_modelprms.pkl', 'PyGEM model parameters')

    # RGI glacier number
    if args.rgi_glac_number_fn is not None:
        with open(args.rgi_glac_number_fn, 'rb') as f:
            glac_no = pickle.load(f)
    elif pygem_prms.glac_no is not None:
        glac_no = pygem_prms.glac_no
    else:
        main_glac_rgi_all = modelsetup.selectglaciersrgitable(
                rgi_regionsO1=pygem_prms.rgi_regionsO1, rgi_regionsO2 =pygem_prms.rgi_regionsO2,
                rgi_glac_number=pygem_prms.rgi_glac_number)
        glac_no = list(main_glac_rgi_all['rgino_str'].values)

    # Number of cores for parallel processing
    if args.option_parallels != 0:
        num_cores = int(np.min([len(glac_no), args.num_simultaneous_processes]))
    else:
        num_cores = 1

    # Glacier number lists to pass for parallel processing
    glac_no_lsts = split_glaciers.split_list(glac_no, n=num_cores, option_ordered=args.option_ordered)

    # Read GCM names from argument parser
    gcm_name = args.ref_gcm_name
    print('Processing:', gcm_name)
    
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
    # Place local variables in variable explorer
    if args.option_parallels == 0:
        main_vars_list = list(main_vars.keys())
        gcm_name = main_vars['gcm_name']
        main_glac_rgi = main_vars['main_glac_rgi']
        if pygem_prms.hyps_data in ['Huss', 'Farinotti']:
            main_glac_hyps = main_vars['main_glac_hyps']
            main_glac_icethickness = main_vars['main_glac_icethickness']
            main_glac_width = main_vars['main_glac_width']
        dates_table = main_vars['dates_table']
        if pygem_prms.option_synthetic_sim == 1:
            dates_table_synthetic = main_vars['dates_table_synthetic']
            gcm_temp_tile = main_vars['gcm_temp_tile']
            gcm_prec_tile = main_vars['gcm_prec_tile']
            gcm_lr_tile = main_vars['gcm_lr_tile']
        gcm_temp = main_vars['gcm_temp']
        gcm_tempstd = main_vars['gcm_tempstd']
        gcm_prec = main_vars['gcm_prec']
        gcm_elev = main_vars['gcm_elev']
        gcm_lr = main_vars['gcm_lr']
        gcm_temp_lrglac = main_vars['gcm_lr']
#        output_ds_all_stats = main_vars['output_ds_all_stats']
#        modelprms = main_vars['modelprms']
        glacier_rgi_table = main_vars['glacier_rgi_table']
        glacier_str = main_vars['glacier_str']
        if pygem_prms.hyps_data in ['oggm']:
            gdir = main_vars['gdir']
            fls = main_vars['fls']
            elev_bins = fls[0].surface_h
#            icethickness_initial = fls[0].thick
            width_initial = fls[0].widths_m / 1000
            glacier_area_initial = width_initial * fls[0].dx / 1000
#            mbmod = main_vars['mbmod']
#            if pygem_prms.use_calibrated_modelparams:
#                modelprms_dict = main_vars['modelprms_dict']
#            model = main_vars['model']
