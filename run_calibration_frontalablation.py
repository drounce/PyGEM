"""
Calibrate frontal ablation parameters for tidewater glaciers

@author: davidrounce
"""
# Built-in libraries
import os
import pickle

# External libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress
import xarray as xr

# Local libraries
import class_climate
import pygem.pygem_input as pygem_prms
import pygem.pygem_modelsetup as modelsetup
from pygem.massbalance import PyGEMMassBalance
from pygem.glacierdynamics import MassRedistributionCurveModel
from pygem.oggm_compat import single_flowline_glacier_directory_with_calving
from pygem.shop import debris 

from oggm import utils, cfg
from oggm import tasks
from oggm.core import climate
from oggm.core.flowline import FluxBasedModel
from oggm.core.inversion import find_inversion_calving_from_any_mb


#%% ----- MANUAL INPUT DATA -----
regions = [1,3,4,5,7,9,17,19]
#regions = [19]

overwrite = False
output_fp = pygem_prms.main_directory + '/../calving_data/analysis/'

option_merge_data = False        # Merge frontal ablation datasets and add mbclim data
option_ind_calving_k = False    # Calibrate individual glaciers
option_reg_calving_k = False    # Calibrate all glaciers regionally
if option_reg_calving_k:
    drop_ind_glaciers = False # For region 9 decide if using individual glacier data or regional data
option_merge_calving_k = False   # Merge all regions together
option_update_mb_data = False   # Update gdirs with the new mass balance data
option_plot_calving_k = True    # Plots of the calibration performance
option_scrap = False             # Scrap calculations



frontal_ablation_Gta_cn = 'fa_gta_obs'
frontal_ablation_Gta_unc_cn = 'fa_gta_obs_unc'

# Frontal ablation calibration parameter (yr-1)
calving_k_init = 0.1
calving_k_bndlow = 0.001
calving_k_bndhigh = 5
calving_k_step = 0.2
nround_max = 5
cfl_number = 0.01
invert_standard=False

perc_threshold_agreement = 0.05     # Threshold (%) at which to stop optimization and consider good agreement
fa_threshold = 1e-4                 # Absolute threshold at which to stop optimization (Gta)

debug=True
debug_reg_calving_fxn = True
prms_from_reg_priors=False
prms_from_glac_cal=True

#%% ----- CONVERSION FUNCTIONS -----
def mwea_to_gta(mwea, area_m2):
    return mwea * pygem_prms.density_water * area_m2 / 1e12
def gta_to_mwea(gta, area_m2):
    return gta * 1e12 / pygem_prms.density_water / area_m2

#%% ----- CALIBRATE FRONTAL ABLATION -----
def reg_calving_flux(main_glac_rgi, calving_k, fa_glac_data_reg=None,
                     frontal_ablation_Gta_cn=None,
                     prms_from_reg_priors=False, prms_from_glac_cal=False, ignore_nan=True, debug=False,
                     invert_standard=invert_standard,
                     calc_mb_geo_correction=False, reset_gdir=True):
    """
    Compute the calving flux for a group of glaciers
    
    Parameters
    ----------
    main_glac_rgi : pd.DataFrame
        rgi summary statistics of each glacier
    calving_k : np.float
        calving model parameter (typical values on order of 1)
    prms_from_reg_priors : Boolean
        use model parameters from regional priors
    prms_from_glac_cal : Boolean
        use model parameters from initial calibration

    Returns
    -------
    output_df : pd.DataFrame
        Dataframe containing information pertaining to each glacier's calving flux
    """    
    # ===== TIME PERIOD =====
    dates_table = modelsetup.datesmodelrun(
            startyear=pygem_prms.ref_startyear, endyear=pygem_prms.ref_endyear, spinupyears=pygem_prms.ref_spinupyears,
            option_wateryear=pygem_prms.ref_wateryear)

    # ===== LOAD CLIMATE DATA =====
    # Climate class
    assert pygem_prms.ref_gcm_name in ['ERA5', 'ERA-Interim'], (
            'Error: Calibration not set up for ' + pygem_prms.ref_gcm_name)
    gcm = class_climate.GCM(name=pygem_prms.ref_gcm_name)
    # Air temperature [degC]
    gcm_temp, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(gcm.temp_fn, gcm.temp_vn, main_glac_rgi, dates_table)
    if pygem_prms.option_ablation == 2 and pygem_prms.ref_gcm_name in ['ERA5']:
        gcm_tempstd, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(gcm.tempstd_fn, gcm.tempstd_vn,
                                                                        main_glac_rgi, dates_table)
    else:
        gcm_tempstd = np.zeros(gcm_temp.shape)
    # Precipitation [m]
    gcm_prec, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(gcm.prec_fn, gcm.prec_vn, main_glac_rgi, dates_table)
    # Elevation [m asl]
    gcm_elev = gcm.importGCMfxnearestneighbor_xarray(gcm.elev_fn, gcm.elev_vn, main_glac_rgi)
    # Lapse rate [degC m-1]
    if pygem_prms.use_constant_lapserate:
        gcm_lr = np.zeros(gcm_temp.shape) + pygem_prms.lapserate
    else:
        gcm_lr, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(gcm.lr_fn, gcm.lr_vn, main_glac_rgi, dates_table)

    # ===== CALIBRATE ALL THE GLACIERS AT ONCE =====
    output_cns = ['RGIId', 'calving_k', 'calving_thick', 'calving_flux_Gta_inv', 'calving_flux_Gta', 'no_errors', 'oggm_dynamics']
    output_df = pd.DataFrame(np.zeros((main_glac_rgi.shape[0],len(output_cns))), columns=output_cns)
    output_df['RGIId'] = main_glac_rgi.RGIId
    output_df['calving_k'] = calving_k
    output_df['calving_thick'] = np.nan
    output_df['calving_flux_Gta'] = np.nan
    output_df['oggm_dynamics'] = 0
    output_df['mb_mwea_fa_asl_lost'] = 0
    for nglac in np.arange(main_glac_rgi.shape[0]):
        
        print('\n',main_glac_rgi.loc[main_glac_rgi.index.values[nglac],'RGIId'])
#        if main_glac_rgi.loc[nglac,'RGIId'] in ['RGI60-09.00855']:
        
        # Select subsets of data
        glacier_rgi_table = main_glac_rgi.loc[main_glac_rgi.index.values[nglac], :]
        glacier_str = '{0:0.5f}'.format(glacier_rgi_table['RGIId_float'])
        
        gdir = single_flowline_glacier_directory_with_calving(glacier_str, 
                                                              logging_level='CRITICAL',
#                                                              logging_level='WORKFLOW'
                                                              reset=reset_gdir
                                                              )
        gdir.is_tidewater = True

        try:
            fls = gdir.read_pickle('inversion_flowlines')
            glacier_area = fls[0].widths_m * fls[0].dx_meter
            debris.debris_binned(gdir, fl_str='inversion_flowlines', ignore_debris=True)
        except:
            fls = None
              
        # Add climate data to glacier directory
        gdir.historical_climate = {'elev': gcm_elev[nglac],
                                   'temp': gcm_temp[nglac,:],
                                   'tempstd': gcm_tempstd[nglac,:],
                                   'prec': gcm_prec[nglac,:],
                                   'lr': gcm_lr[nglac,:]}
        gdir.dates_table = dates_table
        
        # ----- Invert ice thickness and run simulation ------
        if (fls is not None) and (glacier_area.sum() > 0):
            
            # ----- Model parameters -----
            kp_value = None
            tbias_value = None
            # Use most likely parameters from initial calibration to force the mass balance gradient for the inversion
            if prms_from_reg_priors:
                if pygem_prms.priors_reg_fullfn is not None:
                    # Load priors
                    priors_df = pd.read_csv(pygem_prms.priors_reg_fullfn)
                    priors_idx = np.where((priors_df.O1Region == glacier_rgi_table['O1Region']) & 
                                          (priors_df.O2Region == glacier_rgi_table['O2Region']))[0][0]
                    kp_value = priors_df.loc[priors_idx,'kp_med']
                    tbias_value = priors_df.loc[priors_idx,'tbias_med']
            # Use the calibrated model parameters (although they were calibrated without accounting for calving)
            elif prms_from_glac_cal:
                modelprms_fn = glacier_str + '-modelprms_dict.pkl'
                modelprms_fp = (pygem_prms.output_filepath + 'calibration/' + glacier_str.split('.')[0].zfill(2) 
                                + '/')
                assert os.path.exists(modelprms_fp + modelprms_fn), 'modelprms_dict file does not exist'
                with open(modelprms_fp + modelprms_fn, 'rb') as f:
                    modelprms_dict = pickle.load(f)
                modelprms_em = modelprms_dict['emulator']
                kp_value = modelprms_em['kp'][0]
                tbias_value = modelprms_em['tbias'][0]
                
            # Otherwise use input parameters
            if kp_value is None:
                kp_value = pygem_prms.kp
            if tbias_value is None:
                tbias_value = pygem_prms.tbias
            
            # Set model parameters
            modelprms = {'kp': kp_value,
                         'tbias': tbias_value,
                         'ddfsnow': pygem_prms.ddfsnow,
                         'ddfice': pygem_prms.ddfice,
                         'tsnow_threshold': pygem_prms.tsnow_threshold,
                         'precgrad': pygem_prms.precgrad}                
                
            # Calving and dynamic parameters
            cfg.PARAMS['calving_k'] = calving_k
            cfg.PARAMS['inversion_calving_k'] = cfg.PARAMS['calving_k']
            
            if pygem_prms.use_reg_glena:
                glena_df = pd.read_csv(pygem_prms.glena_reg_fullfn)
                glena_idx = np.where(glena_df.O1Region == glacier_rgi_table.O1Region)[0][0]
                glen_a_multiplier = glena_df.loc[glena_idx,'glens_a_multiplier']
                fs = glena_df.loc[glena_idx,'fs']
            else:
                fs = pygem_prms.fs
                glen_a_multiplier = pygem_prms.glen_a_multiplier
                
#            print('\n\nOVERWRITING GLEN_A\n\n')
#            # Increasing glen_a_multiplier, decreases frontal ablation by decreasing ice thickness
#            glen_a_multiplier=3
            
            # cfl_number of 0.01 is more conservative than the default of 0.02 (less issues)
            if cfl_number is None:
                cfg.PARAMS['cfl_number'] = pygem_prms.cfl_number
            else:
                cfg.PARAMS['cfl_number'] = cfl_number
            
            # ----- Mass balance model for ice thickness inversion using OGGM -----
            mbmod_inv = PyGEMMassBalance(gdir, modelprms, glacier_rgi_table,
                                         hindcast=pygem_prms.hindcast,
                                         debug=pygem_prms.debug_mb,
                                         debug_refreeze=pygem_prms.debug_refreeze,
                                         fls=fls, option_areaconstant=False,
                                         inversion_filter=False)
            h, w = gdir.get_inversion_flowline_hw()
            
#            if debug:
#                mb_t0 = (mbmod_inv.get_annual_mb(h, year=0, fl_id=0, fls=fls) * cfg.SEC_IN_YEAR * 
#                         pygem_prms.density_ice / pygem_prms.density_water) 
#                plt.plot(mb_t0, h, '.')
#                plt.ylabel('Elevation')
#                plt.xlabel('Mass balance (mwea)')
#                plt.show()
            
            # ----- CALVING -----
            # Number of years (for OGGM's run_until_and_store)
            if pygem_prms.timestep == 'monthly':
                nyears = int(dates_table.shape[0]/12)
            else:
                assert True==False, 'Adjust nyears for non-monthly timestep'
            mb_years=np.arange(nyears)
            
            # Perform inversion
            # - find_inversion_calving_from_any_mb will do the inversion with calving, but if it fails
            #   then it will do the inversion assuming land-terminating
            if invert_standard:
                climate.apparent_mb_from_any_mb(gdir, mb_model=mbmod_inv, mb_years=np.arange(nyears))
                tasks.prepare_for_inversion(gdir)
                tasks.mass_conservation_inversion(gdir, glen_a=cfg.PARAMS['glen_a']*glen_a_multiplier, fs=fs)
#                tasks.filter_inversion_output(gdir)
            else:
                out_calving = find_inversion_calving_from_any_mb(gdir, mb_model=mbmod_inv, mb_years=mb_years,
                                                                 glen_a=cfg.PARAMS['glen_a']*glen_a_multiplier, fs=fs)
                
            # ------ MODEL WITH EVOLVING AREA ------
            tasks.init_present_time_glacier(gdir) # adds bins below
            debris.debris_binned(gdir, fl_str='model_flowlines')  # add debris enhancement factors to flowlines
            nfls = gdir.read_pickle('model_flowlines')
            # Mass balance model
            mbmod = PyGEMMassBalance(gdir, modelprms, glacier_rgi_table,
                                     hindcast=pygem_prms.hindcast,
                                     debug=pygem_prms.debug_mb,
                                     debug_refreeze=pygem_prms.debug_refreeze,
                                     fls=nfls, option_areaconstant=True)
            # Water Level
            # Check that water level is within given bounds
            cls = gdir.read_pickle('inversion_input')[-1]
            th = cls['hgt'][-1]
            vmin, vmax = cfg.PARAMS['free_board_marine_terminating']
            water_level = utils.clip_scalar(0, th - vmax, th - vmin)
    
            #%%
            ev_model = FluxBasedModel(nfls, y0=0, mb_model=mbmod, 
                                      glen_a=cfg.PARAMS['glen_a']*glen_a_multiplier, fs=fs,
                                      is_tidewater=gdir.is_tidewater,
                                      water_level=water_level
                                      )
            
            try:
                _, diag = ev_model.run_until_and_store(nyears)
                ev_model.mb_model.glac_wide_volume_annual[-1] = diag.volume_m3[-1]
                ev_model.mb_model.glac_wide_area_annual[-1] = diag.area_m2[-1]
                
                # Record frontal ablation for tidewater glaciers and update total mass balance
                if gdir.is_tidewater:
                    # Glacier-wide frontal ablation (m3 w.e.)
                    # - note: diag.calving_m3 is cumulative calving
#                    if debug:
#                        print('\n\ndiag.calving_m3:', diag.calving_m3.values)
#                        print('calving_m3_since_y0:', ev_model.calving_m3_since_y0)
                    calving_m3_annual = ((diag.calving_m3.values[1:] - diag.calving_m3.values[0:-1]) * 
                                         pygem_prms.density_ice / pygem_prms.density_water)
                    for n in np.arange(calving_m3_annual.shape[0]):
                        ev_model.mb_model.glac_wide_frontalablation[12*n+11] = calving_m3_annual[n]

                    # Glacier-wide total mass balance (m3 w.e.)
                    ev_model.mb_model.glac_wide_massbaltotal = (
                            ev_model.mb_model.glac_wide_massbaltotal  - ev_model.mb_model.glac_wide_frontalablation)
                    
#                    if debug:
#                        print('avg calving_m3:', calving_m3_annual.sum() / nyears)
#                        print('avg frontal ablation [Gta]:', 
#                              np.round(ev_model.mb_model.glac_wide_frontalablation.sum() / 1e9 / nyears,4))
#                        print('avg frontal ablation [Gta]:', 
#                              np.round(ev_model.calving_m3_since_y0 * pygem_prms.density_ice / 1e12 / nyears,4))
                        
                    # Output of calving
                    out_calving_forward = {}
                    # calving flux (km3 ice/yr)
                    out_calving_forward['calving_flux'] = calving_m3_annual.sum() / nyears / 1e9
                    # calving flux (Gt/yr)
                    calving_flux_Gta = out_calving_forward['calving_flux'] * pygem_prms.density_ice / pygem_prms.density_water
                    
                    # calving front thickness at start of simulation
                    thick = nfls[0].thick
                    last_idx = np.nonzero(thick)[0][-1]
                    out_calving_forward['calving_front_thick'] = thick[last_idx]
                    
                    
                    # Record in dataframe
                    output_df.loc[nglac,'calving_flux_Gta'] = calving_flux_Gta
                    output_df.loc[nglac,'calving_thick'] = out_calving_forward['calving_front_thick']
                    output_df.loc[nglac,'no_errors'] = 1
                    output_df.loc[nglac,'oggm_dynamics'] = 1
                    
                    if debug:               
                        print('OGGM dynamics, calving_k:', np.round(calving_k,4), 'glen_a:', np.round(glen_a_multiplier,2))                 
                        print('    calving front thickness [m]:', np.round(out_calving_forward['calving_front_thick'],1))
                        print('    calving flux model [Gt/yr]:', np.round(calving_flux_Gta,5))
                
            except:
                if gdir.is_tidewater:
                    if debug:
                        print('OGGM dynamics failed, using mass redistribution curves')
                                                    # Mass redistribution curves glacier dynamics model
                    ev_model = MassRedistributionCurveModel(
                                    nfls, mb_model=mbmod, y0=0,
                                    glen_a=cfg.PARAMS['glen_a']*glen_a_multiplier, fs=fs,
                                    is_tidewater=gdir.is_tidewater,
                                    water_level=water_level
                                    )
                    _, diag = ev_model.run_until_and_store(nyears)
                    ev_model.mb_model.glac_wide_volume_annual = diag.volume_m3.values
                    ev_model.mb_model.glac_wide_area_annual = diag.area_m2.values
    
                    # Record frontal ablation for tidewater glaciers and update total mass balance
                    # Update glacier-wide frontal ablation (m3 w.e.)
                    ev_model.mb_model.glac_wide_frontalablation = ev_model.mb_model.glac_bin_frontalablation.sum(0)
                    # Update glacier-wide total mass balance (m3 w.e.)
                    ev_model.mb_model.glac_wide_massbaltotal = (
                            ev_model.mb_model.glac_wide_massbaltotal - ev_model.mb_model.glac_wide_frontalablation)

                    calving_flux_km3a = (ev_model.mb_model.glac_wide_frontalablation.sum() * pygem_prms.density_water / 
                                         pygem_prms.density_ice / nyears / 1e9)

#                    if debug:
#                        print('avg frontal ablation [Gta]:', 
#                              np.round(ev_model.mb_model.glac_wide_frontalablation.sum() / 1e9 / nyears,4))
#                        print('avg frontal ablation [Gta]:', 
#                              np.round(ev_model.calving_m3_since_y0 * pygem_prms.density_ice / 1e12 / nyears,4))
                    
                    # Output of calving
                    out_calving_forward = {}
                    # calving flux (km3 ice/yr)
                    out_calving_forward['calving_flux'] = calving_flux_km3a
                    # calving flux (Gt/yr)
                    calving_flux_Gta = out_calving_forward['calving_flux'] * pygem_prms.density_ice / pygem_prms.density_water
                    # calving front thickness at start of simulation
                    thick = nfls[0].thick
                    last_idx = np.nonzero(thick)[0][-1]
                    out_calving_forward['calving_front_thick'] = thick[last_idx]
                    
                    # Record in dataframe
                    output_df.loc[nglac,'calving_flux_Gta'] = calving_flux_Gta
                    output_df.loc[nglac,'calving_thick'] = out_calving_forward['calving_front_thick']
                    output_df.loc[nglac,'no_errors'] = 1
                    
                    if debug:          
                        print('Mass Redistribution curve, calving_k:', np.round(calving_k,1), 'glen_a:', np.round(glen_a_multiplier,2))                       
                        print('    calving front thickness [m]:', np.round(out_calving_forward['calving_front_thick'],0))
                        print('    calving flux model [Gt/yr]:', np.round(calving_flux_Gta,5))


            if calc_mb_geo_correction:
                # Mass balance correction from mass loss above sea level due to calving retreat 
                #  (i.e., what the geodetic signal should see)
                last_yr_idx = np.where(mbmod.glac_wide_area_annual > 0)[0][-1]
                if last_yr_idx == mbmod.glac_bin_area_annual.shape[1]-1:
                    last_yr_idx = -2
                bin_last_idx = np.where(mbmod.glac_bin_area_annual[:,last_yr_idx] > 0)[0][-1]
                bin_area_lost = mbmod.glac_bin_area_annual[bin_last_idx:,0] - mbmod.glac_bin_area_annual[bin_last_idx:,-2]
                height_asl = mbmod.heights - water_level
                height_asl[mbmod.heights<0] = 0
                mb_mwea_fa_asl_geo_correction = ((bin_area_lost * height_asl[bin_last_idx:]).sum() / 
                                        mbmod.glac_wide_area_annual[0] *
                                        pygem_prms.density_ice / pygem_prms.density_water / nyears)
                mb_mwea_fa_asl_geo_correction_max = 0.3*gta_to_mwea(calving_flux_Gta, glacier_rgi_table['Area']*1e6)
                if mb_mwea_fa_asl_geo_correction > mb_mwea_fa_asl_geo_correction_max:
                    mb_mwea_fa_asl_geo_correction = mb_mwea_fa_asl_geo_correction_max
                    
                # Below sea-level correction due to calving that geodetic mass balance doesn't see
#                print('test:', mbmod.glac_bin_icethickness_annual.shape, height_asl.shape, bin_area_lost.shape)
#                height_bsl = mbmod.glac_bin_icethickness_annual - height_asl
                
                # Area for retreat
                if debug:
#                    print('\n----- area calcs -----')
#                    print(mbmod.glac_bin_area_annual[bin_last_idx:,0])
#                    print(mbmod.glac_bin_icethickness_annual[bin_last_idx:,0])
#                    print(mbmod.glac_bin_area_annual[bin_last_idx:,-2])
#                    print(mbmod.glac_bin_icethickness_annual[bin_last_idx:,-2])
#                    print(mbmod.heights.shape, mbmod.heights[bin_last_idx:])
                    print('  mb_mwea_fa_asl_geo_correction:', np.round(mb_mwea_fa_asl_geo_correction,2))
#                    print('  mb_mwea_fa_asl_geo_correction:', mb_mwea_fa_asl_geo_correction)
#                    print(glacier_rgi_table, glacier_rgi_table['Area'])
                    
                    
                output_df.loc[nglac,'mb_mwea_fa_asl_lost'] = mb_mwea_fa_asl_geo_correction

            if out_calving_forward is None:
                output_df.loc[nglac,['calving_k', 'calving_thick', 'calving_flux_Gta', 'no_errors']] = (
                        np.nan, np.nan, np.nan, 0)
                
    # Remove glaciers that failed to run
    if fa_glac_data_reg is None:
        reg_calving_gta_obs_good = None
        output_df_good = output_df.dropna(axis=0, subset=['calving_flux_Gta'])
        reg_calving_gta_mod_good = output_df_good.calving_flux_Gta.sum()
    elif ignore_nan:
        output_df_good = output_df.dropna(axis=0, subset=['calving_flux_Gta'])
        reg_calving_gta_mod_good = output_df_good.calving_flux_Gta.sum()
        rgiids_data = list(fa_glac_data_reg.RGIId.values)
        rgiids_mod = list(output_df_good.RGIId.values)
        fa_data_idx = [rgiids_data.index(x) for x in rgiids_mod]
        reg_calving_gta_obs_good = fa_glac_data_reg.loc[fa_data_idx,frontal_ablation_Gta_cn].sum()
    else:
        reg_calving_gta_mod_good = output_df.calving_flux_Gta.sum()
        reg_calving_gta_obs_good = fa_glac_data_reg[frontal_ablation_Gta_cn].sum()
            
    return output_df, reg_calving_gta_mod_good, reg_calving_gta_obs_good
            


def run_opt_fa(main_glac_rgi_ind, calving_k, calving_k_bndlow, calving_k_bndhigh, fa_glac_data_ind,
               calving_k_step=calving_k_step,
              frontal_ablation_Gta_cn=frontal_ablation_Gta_cn, 
              prms_from_reg_priors=prms_from_reg_priors, prms_from_glac_cal=prms_from_glac_cal,
              ignore_nan=False, debug=debug_reg_calving_fxn, calc_mb_geo_correction=False):
    """
    Run the optimization of the frontal ablation for an individual glacier
    """
    output_df, reg_calving_gta_mod, reg_calving_gta_obs = (
            reg_calving_flux(main_glac_rgi_ind, calving_k, fa_glac_data_reg=fa_glac_data_ind,
                             frontal_ablation_Gta_cn=frontal_ablation_Gta_cn, 
                             prms_from_reg_priors=prms_from_reg_priors, prms_from_glac_cal=prms_from_glac_cal,
                             ignore_nan=False, debug=debug_reg_calving_fxn,
                             calc_mb_geo_correction=calc_mb_geo_correction))
    
    calving_k_bndlow_hold = np.copy(calving_k_bndlow)
    
    if debug:
        print('  fa_model_init [Gt/yr] :', np.round(reg_calving_gta_mod,4))
        
    # ----- Rough optimizer using calving_k_step to loop through parameters within bounds ------
    calving_k_last = calving_k
#    output_df_last = output_df.copy()
    reg_calving_gta_mod_last = reg_calving_gta_mod.copy()
    
    if reg_calving_gta_mod < reg_calving_gta_obs:
        
        if debug:
            print('\nincrease calving_k')
        
#        print('reg_calving_gta_mod:', reg_calving_gta_mod)
#        print('reg_calving_gta_obs:', reg_calving_gta_obs)
#        print('calving_k:', calving_k)
#        print('calving_k_bndhigh:', calving_k_bndhigh)
#        print('calving_k_bndlow:', calving_k_bndlow)
            
        while ((reg_calving_gta_mod < reg_calving_gta_obs and np.round(calving_k,2) < calving_k_bndhigh
               and calving_k > calving_k_bndlow
               and (np.abs(reg_calving_gta_mod - reg_calving_gta_obs) / reg_calving_gta_obs > perc_threshold_agreement
                    and np.abs(reg_calving_gta_mod - reg_calving_gta_obs) > fa_threshold))):
            # Record previous output
            calving_k_last = np.copy(calving_k)
#            output_df_last = output_df.copy()
            reg_calving_gta_mod_last = reg_calving_gta_mod.copy()
            
            if debug:
                print(' increase calving_k_step:', calving_k_step)
            
            # Increase calving k
            calving_k += calving_k_step
            
            if calving_k > calving_k_bndhigh:
                calving_k = calving_k_bndhigh
                
            # Re-run the regional frontal ablation estimates
            output_df, reg_calving_gta_mod, reg_calving_gta_obs = (
                reg_calving_flux(main_glac_rgi_ind, calving_k, fa_glac_data_reg=fa_glac_data_ind,
                                 frontal_ablation_Gta_cn=frontal_ablation_Gta_cn, 
                                 prms_from_reg_priors=prms_from_reg_priors, prms_from_glac_cal=prms_from_glac_cal,
                                 ignore_nan=False, debug=debug_reg_calving_fxn,
                                 calc_mb_geo_correction=calc_mb_geo_correction))
            if debug:
                print('  fa_data  [Gt/yr]:', np.round(reg_calving_gta_obs,4))
                print('  fa_model [Gt/yr] :', np.round(reg_calving_gta_mod,4))    
        
        # Set lower bound
        calving_k_bndlow = calving_k_last
        reg_calving_gta_mod_bndlow = reg_calving_gta_mod_last
        # Set upper bound
        calving_k_bndhigh = calving_k
        reg_calving_gta_mod_bndhigh = reg_calving_gta_mod
            
    else:
        
        if debug:
            print('\ndecrease calving_k')      
            
        print('-----')
        print('reg_calving_gta_mod:', reg_calving_gta_mod)
        print('reg_calving_gta_obs:', reg_calving_gta_obs)
        print('calving_k:', calving_k)
        print('calving_k_bndlow:', calving_k_bndlow)
        print('fa perc:', (np.abs(reg_calving_gta_mod - reg_calving_gta_obs) / reg_calving_gta_obs))
        print('fa thres:', np.abs(reg_calving_gta_mod - reg_calving_gta_obs))
        print('good values:', output_df.loc[0,'calving_flux_Gta'])
            
        while ((reg_calving_gta_mod > reg_calving_gta_obs and calving_k > calving_k_bndlow
                and (np.abs(reg_calving_gta_mod - reg_calving_gta_obs) / reg_calving_gta_obs > perc_threshold_agreement
                    and np.abs(reg_calving_gta_mod - reg_calving_gta_obs) > fa_threshold))
               and not np.isnan(output_df.loc[0,'calving_flux_Gta'])):
            # Record previous output
            calving_k_last = np.copy(calving_k)
#            output_df_last = output_df.copy()
            reg_calving_gta_mod_last = reg_calving_gta_mod.copy()
                
            # Decrease calving k
            calving_k -= calving_k_step
            
            if calving_k < calving_k_bndlow:
                calving_k = calving_k_bndlow
                
            # Re-run the regional frontal ablation estimates
            output_df, reg_calving_gta_mod, reg_calving_gta_obs = (
                reg_calving_flux(main_glac_rgi_ind, calving_k, fa_glac_data_reg=fa_glac_data_ind,
                                 frontal_ablation_Gta_cn=frontal_ablation_Gta_cn, 
                                 prms_from_reg_priors=prms_from_reg_priors, prms_from_glac_cal=prms_from_glac_cal,
                                 ignore_nan=False, debug=debug_reg_calving_fxn,
                                 calc_mb_geo_correction=calc_mb_geo_correction))
            if debug:
                print('  fa_data  [Gt/yr]:', np.round(reg_calving_gta_obs,4))
                print('  fa_model [Gt/yr] :', np.round(reg_calving_gta_mod,4))

        # Set lower bound
        calving_k_bndlow = calving_k
        reg_calving_gta_mod_bndlow = reg_calving_gta_mod
        # Set upper bound
        calving_k_bndhigh = calving_k_last
        reg_calving_gta_mod_bndhigh = reg_calving_gta_mod_last
        
        print('bnds:', calving_k_bndlow, calving_k_bndhigh)
        print('bnds gt/yr:', reg_calving_gta_mod_bndlow, reg_calving_gta_mod_bndhigh)

    # ----- Optimize further using mid-point "bisection" method -----
    # Consider replacing with scipy.optimize.brent
    if not np.isnan(output_df.loc[0,'calving_flux_Gta']):
        
        # Check if upper bound causes good fit
        if (np.abs(reg_calving_gta_mod_bndhigh - reg_calving_gta_obs) / reg_calving_gta_obs < perc_threshold_agreement
            or np.abs(reg_calving_gta_mod_bndhigh - reg_calving_gta_obs) < fa_threshold):
            
            # If so, calving_k equals upper bound and re-run to get proper estimates for output
            calving_k = calving_k_bndhigh
            
            # Re-run the regional frontal ablation estimates
            output_df, reg_calving_gta_mod, reg_calving_gta_obs = (
                reg_calving_flux(main_glac_rgi_ind, calving_k, fa_glac_data_reg=fa_glac_data_ind,
                                 frontal_ablation_Gta_cn=frontal_ablation_Gta_cn, 
                                 prms_from_reg_priors=prms_from_reg_priors, prms_from_glac_cal=prms_from_glac_cal,
                                 ignore_nan=False, debug=debug_reg_calving_fxn,
                                 calc_mb_geo_correction=calc_mb_geo_correction))
            if debug:
                print('upper bound:')
                print('  calving_k:', np.round(calving_k,4))
                print('  fa_data  [Gt/yr]:', np.round(reg_calving_gta_obs,4))
                print('  fa_model [Gt/yr] :', np.round(reg_calving_gta_mod,4))
                
        # Check if lower bound causes good fit
        elif (np.abs(reg_calving_gta_mod_bndlow - reg_calving_gta_obs) / reg_calving_gta_obs < perc_threshold_agreement
              or np.abs(reg_calving_gta_mod_bndlow - reg_calving_gta_obs) < fa_threshold):
            
            calving_k = calving_k_bndlow
            # Re-run the regional frontal ablation estimates
            output_df, reg_calving_gta_mod, reg_calving_gta_obs = (
                reg_calving_flux(main_glac_rgi_ind, calving_k, fa_glac_data_reg=fa_glac_data_ind,
                                 frontal_ablation_Gta_cn=frontal_ablation_Gta_cn, 
                                 prms_from_reg_priors=prms_from_reg_priors, prms_from_glac_cal=prms_from_glac_cal,
                                 ignore_nan=False, debug=debug_reg_calving_fxn,
                                 calc_mb_geo_correction=calc_mb_geo_correction))
            if debug:
                print('lower bound:')
                print('  calving_k:', np.round(calving_k,4))
                print('  fa_data  [Gt/yr]:', np.round(reg_calving_gta_obs,4))
                print('  fa_model [Gt/yr] :', np.round(reg_calving_gta_mod,4))
                
        else:
            # Calibrate between limited range
            nround = 0
            # Set initial calving_k
            calving_k = (calving_k_bndlow + calving_k_bndhigh) / 2
            
#            print('fa_perc:', np.abs(reg_calving_gta_mod_bndlow - reg_calving_gta_obs) / reg_calving_gta_obs)
#            print('fa_dif:', np.abs(reg_calving_gta_mod_bndlow - reg_calving_gta_obs))
#            print('calving_k_bndlow:', calving_k_bndlow)
#            print('nround:', nround, 'nround_max:', nround_max)
#            print('calving_k:', calving_k, 'calving_k_bndlow_set:', calving_k_bndlow_hold)
            
            while ((np.abs(reg_calving_gta_mod - reg_calving_gta_obs) / reg_calving_gta_obs > perc_threshold_agreement and 
                    np.abs(reg_calving_gta_mod - reg_calving_gta_obs) > fa_threshold) and nround <= nround_max
                   and calving_k > calving_k_bndlow_hold):
                
                nround += 1
                if debug:
                    print('\nRound', nround)
                # Update calving_k using midpoint
                calving_k = (calving_k_bndlow + calving_k_bndhigh) / 2
                output_df, reg_calving_gta_mod, reg_calving_gta_obs = (
                    reg_calving_flux(main_glac_rgi_ind, calving_k, fa_glac_data_reg=fa_glac_data_ind,
                                     frontal_ablation_Gta_cn=frontal_ablation_Gta_cn, 
                                     prms_from_reg_priors=prms_from_reg_priors, prms_from_glac_cal=prms_from_glac_cal,
                                     ignore_nan=False, debug=debug_reg_calving_fxn,
                                     calc_mb_geo_correction=calc_mb_geo_correction))
                if debug:
                    print('  calving_k:', np.round(calving_k,4))
                    print('  fa_data  [Gt/yr]:', np.round(reg_calving_gta_obs,4))
                    print('  fa_model [Gt/yr] :', np.round(reg_calving_gta_mod,4))
                
                # Update bounds
                if reg_calving_gta_mod < reg_calving_gta_obs:
                    # Update lower bound
                    reg_calving_gta_mod_bndlow = reg_calving_gta_mod
                    calving_k_bndlow = np.copy(calving_k)
                else:
                    # Update upper bound
                    reg_calving_gta_mod_bndhigh = reg_calving_gta_mod
                    calving_k_bndhigh = np.copy(calving_k)
                    
#                if debug:
#                    print('fa_perc:', np.abs(reg_calving_gta_mod_bndlow - reg_calving_gta_obs) / reg_calving_gta_obs)
#                    print('fa_dif:', np.abs(reg_calving_gta_mod_bndlow - reg_calving_gta_obs))
#                    print('calving_k_bndlow:', calving_k_bndlow)
#                    print('nround:', nround, 'nround_max:', nround_max)
#                    print('  calving_k:', calving_k)
                    
        if calving_k < calving_k_bndlow:
            calving_k = calving_k_bndlow
                    
    return output_df, calving_k


#%%
if option_merge_data:
    calving_fp = pygem_prms.main_directory + '/../calving_data/'
#    calving_fn1 = 'Northern_hemisphere_calving_flux_Kochtitzky_et_al_for_David_Rounce_with_melt_v8-wromainMB.csv'
    calving_fn1 = 'Northern_hemisphere_calving_flux_Kochtitzky_et_al_for_David_Rounce_with_melt_v13-wromainMB.csv'
    calving_fn2 = 'frontalablation_glacier_data_minowa2021.csv'
    calving_fn3 = 'frontalablation_glacier_data_osmanoglu.csv'
    
    fa_glac_data_cns_subset = ['RGIId','fa_gta_obs', 'fa_gta_obs_unc',
                               'Romain_gta_mbtot', 'Romain_gta_mbclim','Romain_mwea_mbtot', 'Romain_mwea_mbclim', 
                               'thick_measured_yn', 'start_date', 'end_date', 'source']
    
    # Load datasets
    fa_glac_data1 = pd.read_csv(calving_fp + calving_fn1)
    fa_glac_data2 = pd.read_csv(calving_fp + calving_fn2)
    fa_glac_data3 = pd.read_csv(calving_fp + calving_fn3)
    
    # Process datasets
    mb_data_df = pd.read_csv(pygem_prms.hugonnet_fp + pygem_prms.hugonnet_fn)
    mb_data_df['mb_mwea_romain'] = mb_data_df['mb_mwea'].copy() 
    mb_data_df['mb_mwea_err_romain'] = mb_data_df['mb_mwea_err'].copy() 

    # Kochtitzky data
    fa_data_df1 = pd.DataFrame(np.zeros((fa_glac_data1.shape[0],len(fa_glac_data_cns_subset))), columns=fa_glac_data_cns_subset)
    fa_data_df1['RGIId'] = fa_glac_data1['RGIId']
    fa_data_df1['fa_gta_obs'] = fa_glac_data1['Frontal_ablation_2000_to_2020_gt_per_yr_mean']
    fa_data_df1['fa_gta_obs_unc'] = fa_glac_data1['Frontal_ablation_2000_to_2020_gt_per_yr_mean_err']
    fa_data_df1['Romain_gta_mbtot'] = fa_glac_data1['Romain_gta_mbtot']
    fa_data_df1['Romain_gta_mbclim'] = fa_glac_data1['Romain_gta_mbclim']
    fa_data_df1['Romain_mwea_mbtot'] = fa_glac_data1['Romain_mwea_mbtot']
    fa_data_df1['Romain_mwea_mbclim'] = fa_glac_data1['Romain_mwea_mbclim']
    fa_data_df1['thick_measured_yn'] = fa_glac_data1['thick_measured_yn']
    fa_data_df1['start_date'] = '20009999'
    fa_data_df1['end_date'] = '20199999'
    fa_data_df1['source'] = 'Kochtitzky et al.'

    # Minowa data
    fa_data_df2 = pd.DataFrame(np.zeros((fa_glac_data2.shape[0],len(fa_glac_data_cns_subset))), columns=fa_glac_data_cns_subset)
    fa_data_df2['RGIId'] = fa_glac_data2['RGIId']
    fa_data_df2['fa_gta_obs'] = fa_glac_data2['frontal_ablation_Gta']
    fa_data_df2['fa_gta_obs_unc'] = fa_glac_data2['frontal_ablation_unc_Gta']
#    fa_data_df2['thick_measured_yn'] = np.nan
    fa_data_df2['start_date'] = fa_glac_data2['start_date']
    fa_data_df2['end_date'] = fa_glac_data2['end_date']
    fa_data_df2['source'] = fa_glac_data2['Source']
    fa_data_df2.sort_values('RGIId', inplace=True)
    
    rgiids_fa_data2 = sorted(list(fa_data_df2.RGIId))
    
    # Osmanoglu data
    fa_data_df3 = pd.DataFrame(np.zeros((fa_glac_data3.shape[0],len(fa_glac_data_cns_subset))), columns=fa_glac_data_cns_subset)
    fa_data_df3['RGIId'] = fa_glac_data3['RGIId']
    fa_data_df3['fa_gta_obs'] = fa_glac_data3['frontal_ablation_Gta']
    fa_data_df3['fa_gta_obs_unc'] = fa_glac_data3['frontal_ablation_unc_Gta']
#    fa_data_df3['thick_measured_yn'] = np.nan
    fa_data_df3['start_date'] = fa_glac_data3['start_date']
    fa_data_df3['end_date'] = fa_glac_data3['end_date']
    fa_data_df3['source'] = fa_glac_data3['Source']
    fa_data_df3.sort_values('RGIId', inplace=True)
    
#    rgiids_fa_data3 = sorted(list(fa_data_df3.RGIId))
#    
#    rgiids_mb_data = list(mb_data_df.RGIId.values)
#    
#    # Process data
#    fa_data_process = pd.concat((fa_data_df2, fa_data_df3), axis=0)
#    fa_data_process.reset_index(drop=True, inplace=True)
#    rgiids_process = rgiids_fa_data2 + rgiids_fa_data3
#    for nglac, rgiid in enumerate(rgiids_process):
#        for batman in [0]:
##        if rgiid in ['RGI60-19.02066']:
#            
#            if debug:
#                print('\n' + str(nglac) + ' ' + rgiid)
#            
##            # Aggregate data from multiple glaciers if needed, since some include multiple glaciers
##            if ',' in rgiid:
##                
##                rgiids_multiple_list = rgiid.split(',')
##                
##                # Combine mass balance from both glaciers, remove calving, and set both to be average
##                fa_idx = rgiids_fa_data_process.index(rgiid)
##                fa_gta = fa_data_process.loc[fa_idx,'fa_gta_obs']
###                fa_gta_err = fa_data_process.loc[fa_idx,fa_gta_err_cn]
##                mb_gta_list = []
##                mb_gta_err_list = []
##                area_m2_list = []
##                for rgiid_single in rgiids_multiple_list:
##                    print(rgiid_single)
##                    mb_idx = rgiids_mb_data.index(rgiid_single)
##                    
##                    print(rgiid_single, mb_data_df.loc[mb_idx,'mb_mwea_romain'], mb_data_df.loc[mb_idx,'area'])
##                    
##                    mb_gta_single = mwea_to_gta(mb_data_df.loc[mb_idx,'mb_mwea_romain'], 
##                                                mb_data_df.loc[mb_idx,'area'] * 1e6)
###                    mb_gta_err_single = mwea_to_gta(mb_data_df.loc[mb_idx,'mb_mwea_err_romain'],
###                                                    mb_data_df.loc[mb_idx,'area'] * 1e6)                    
##                    mb_gta_list.append(mb_gta_single)
###                    mb_gta_err_list.append(mb_gta_err_single)
##                    area_m2_list.append(mb_data_df.loc[mb_idx,'area'] * 1e6)
##    
##                mb_gta = np.array(mb_gta_list).sum()
###                mb_gta_err = (np.array(mb_gta_err_list)**2).sum()**0.5
##                area_m2 = np.array(area_m2_list).sum()
##                
##                mb_mwea = gta_to_mwea(mb_gta, area_m2)
###                mb_mwea_err = gta_to_mwea(mb_gta_err, area_m2)
##                
##                fa_mwea = gta_to_mwea(fa_gta, area_m2)
###                fa_mwea_err = gta_to_mwea(fa_gta_err, area_m2)
##                    
##                assert True==False, 'here'
##                
##            # Otherwise load individual glacier data
##            else:
#                
#            mb_idx = rgiids_mb_data.index(rgiid)
#        
#            # Mass balance
#            mb_mwea = mb_data_df.loc[mb_idx,pygem_prms.hugonnet_mb_cn]
#            mb_mwea_err = mb_data_df.loc[mb_idx,pygem_prms.hugonnet_mb_err_cn]
#            area_m2 = mb_data_df.loc[mb_idx,'area'] * 1e6
#            mb_gta = mwea_to_gta(mb_mwea, area_m2)
#            fa_data_process.loc[nglac,'Romain_mwea_mbtot'] = mb_mwea
#            fa_data_process.loc[nglac,'Romain_gta_mbtot'] = mb_gta
##                fa_data_process.loc[nglac,'Romain_mwea_mbtot_err'] = mb_data_df.loc[mb_idx,pygem_prms.hugonnet_mb_err_cn]
#        
#            # Frontal Ablation (gta)
#            fa_gta = fa_data_process.loc[nglac,'fa_gta_obs']
#            # convert to mwea
#            fa_mwea = gta_to_mwea(fa_gta, area_m2)
#                
#            # Climatic mass balance correct for frontal ablation
#            #  - equals total mass balance minus frontal ablation
#            mb_gta_mbclim = mb_gta + fa_gta
#            fa_data_process.loc[nglac,'Romain_gta_mbclim'] = mb_gta_mbclim
##            # sum of squares to aggregate error
##            mb_gta_mbclim_err = (mb_gta_mbtot_err**2 + fa_gta_err**2)**0.5
##            fa_data_df.loc[nglac,'Romain_gta_mbclim_err'] = mb_gta_mbclim_err
#                
#            # Convert to mwea 
#            fa_data_process.loc[nglac,'Romain_mwea_mbtot'] = gta_to_mwea(mb_gta, area_m2) 
##            fa_data_df.loc[nglac,'Romain_mwea_mbtot_err'] = gta_to_mwea(mb_gta_mbtot_err, area_m2) 
#            fa_data_process.loc[nglac,'Romain_mwea_mbclim'] = gta_to_mwea(mb_gta_mbclim, area_m2)
##            fa_data_df.loc[nglac,'Romain_mwea_mbclim_err'] = gta_to_mwea(mb_gta_mbclim_err, area_m2)
#            
#            if debug:
#                print('  mb_tot (mwea):', np.round(gta_to_mwea(mb_gta, area_m2),2))
#                print('  mb_clim (mwea):', np.round(gta_to_mwea(mb_gta_mbclim, area_m2),2))
#                
##            # Record area
##            mb_idx = rgiids_mb_data.index(rgiid)
##            fa_data_df.loc[nglac,'area_km2'] = mb_data_df.loc[mb_idx,'area']

    # Concatenate datasets
#    fa_data_df = pd.concat([fa_data_df1, fa_data_process], axis=0)
    fa_data_df = pd.concat([fa_data_df1, fa_data_df2, fa_data_df3], axis=0)
    area_dict = dict(zip(mb_data_df.RGIId, mb_data_df.area))
    fa_data_df['area_km2'] = fa_data_df['RGIId'].map(area_dict)

    # Export frontal ablation data for Will
    fa_data_df.to_csv(calving_fp + calving_fn1.replace('.csv','-w17_19.csv'), index=False)
    
                    
#%%
if option_reg_calving_k:
    # Load calving glacier data
    fa_glac_data = pd.read_csv(pygem_prms.frontalablation_glacier_data_fullfn)
    
    for reg in regions:
        # Regional data
        fa_glac_data_reg = fa_glac_data.loc[fa_glac_data['O1Region'] == reg, :].copy()
        fa_glac_data_reg.reset_index(inplace=True, drop=True)
        # Drop individual data points
        if drop_ind_glaciers:
            fa_idxs = []
            for nglac, rgiid in enumerate(fa_glac_data_reg.RGIId):
                # Avoid regional data and observations from multiple RGIIds (len==14)
                if fa_glac_data_reg.loc[nglac,'RGIId'] == 'all':
                    fa_idxs.append(nglac)
            fa_glac_data_reg = fa_glac_data_reg.loc[fa_idxs,:]
            fa_glac_data_reg.reset_index(inplace=True, drop=True)

        if fa_glac_data_reg.loc[0,'RGIId'] == 'all':
            main_glac_rgi_all = modelsetup.selectglaciersrgitable(rgi_regionsO1=[reg], rgi_regionsO2='all', 
                                                                  rgi_glac_number='all', 
                                                                  include_landterm=False, include_laketerm=False, 
                                                                  include_tidewater=True)
            # Set ignore_nan to False because we don't know individual glaciers
            ignore_nan = False
        else:
            fa_glac_data_reg['glacno'] = np.nan
            for nglac, rgiid in enumerate(fa_glac_data_reg.RGIId):
                # Avoid regional data and observations from multiple RGIIds (len==14)
#                if (not fa_glac_data_reg.loc[nglac,'RGIId'] == 'all' and len(fa_glac_data_reg.loc[nglac,'RGIId']) == 14
#                    and fa_glac_data_reg.loc[nglac,'RGIId'] in ['RGI60-03.00191']):
                if not fa_glac_data_reg.loc[nglac,'RGIId'] == 'all' and len(fa_glac_data_reg.loc[nglac,'RGIId']) == 14:
                    fa_glac_data_reg.loc[nglac,'glacno'] = (str(int(rgiid.split('-')[1].split('.')[0])) + '.' + 
                                                            rgiid.split('-')[1].split('.')[1])
            # Drop observations that aren't of individual glaciers
            fa_glac_data_reg = fa_glac_data_reg.dropna(axis=0, subset=['glacno'])
            fa_glac_data_reg.reset_index(inplace=True, drop=True)
            reg_calving_gta_obs = fa_glac_data_reg[frontal_ablation_Gta_cn].sum()
            glacno_reg_wdata = sorted(list(fa_glac_data_reg.glacno.values))
            main_glac_rgi_all = modelsetup.selectglaciersrgitable(glac_no=glacno_reg_wdata)
            # Ignore nan for individual glaciers
            ignore_nan = True
        
        # Tidewater glaciers
        termtype_list = [1,5]
        main_glac_rgi = main_glac_rgi_all.loc[main_glac_rgi_all['TermType'].isin(termtype_list)]
        main_glac_rgi.reset_index(inplace=True, drop=True)
        
        # ----- OPTIMIZE CALVING_K BASED ON REGIONAL FRONTAL ABLATION DATA -----
        calving_k = calving_k_init
        output_df, reg_calving_gta_mod, reg_calving_gta_obs = (
                reg_calving_flux(main_glac_rgi, calving_k, fa_glac_data_reg=fa_glac_data_reg, 
                                 frontal_ablation_Gta_cn=frontal_ablation_Gta_cn, ignore_nan=ignore_nan,
                                 prms_from_reg_priors=prms_from_reg_priors, prms_from_glac_cal=prms_from_glac_cal, 
                                 debug=True))
        
        if debug:
            print('calving_k:', calving_k)
            print('  fa_data  [Gt/yr]:', np.round(reg_calving_gta_obs,5))
            print('  fa_model [Gt/yr] :', np.round(reg_calving_gta_mod,5))
    
        # ----- Rough optimizer using calving_k_step to loop through parameters within bounds ------
        if reg_calving_gta_mod < reg_calving_gta_obs:
            
            if debug:
                print('increase calving_k')
                
            while reg_calving_gta_mod < reg_calving_gta_obs and np.round(calving_k,2) < calving_k_bndhigh:
                # Record previous output
                calving_k_last = calving_k
#                output_df_last = output_df.copy()
                reg_calving_gta_mod_last = reg_calving_gta_mod.copy()
                
                # Increase calving k
                calving_k += calving_k_step
                
                if calving_k > calving_k_bndhigh:
                    calving_k = calving_k_bndhigh
                
                # Re-run the regional frontal ablation estimates
                output_df, reg_calving_gta_mod, reg_calving_gta_obs = (
                    reg_calving_flux(main_glac_rgi, calving_k, fa_glac_data_reg=fa_glac_data_reg, 
                                     frontal_ablation_Gta_cn=frontal_ablation_Gta_cn, ignore_nan=ignore_nan,
                                     prms_from_reg_priors=prms_from_reg_priors, prms_from_glac_cal=prms_from_glac_cal, 
                                     debug=True))
                if debug:
                    print('calving_k:', calving_k)
                    print('  fa_data  [Gt/yr]:', np.round(reg_calving_gta_obs,2))
                    print('  fa_model [Gt/yr] :', np.round(reg_calving_gta_mod,2))    
            
            # Set lower bound
            calving_k_bndlow = calving_k_last
            reg_calving_gta_mod_bndlow = reg_calving_gta_mod_last
            # Set upper bound
            calving_k_bndhigh = calving_k
            reg_calving_gta_mod_bndhigh = reg_calving_gta_mod
                
        else:
            
            if debug:
                print('decrease calving_k')      
                
            while reg_calving_gta_mod > reg_calving_gta_obs and np.round(calving_k,2) > calving_k_bndlow:
                # Record previous output
                calving_k_last = calving_k
#                output_df_last = output_df.copy()
                reg_calving_gta_mod_last = reg_calving_gta_mod.copy()
                
                # Increase calving k
                calving_k -= calving_k_step
                
                if calving_k < calving_k_bndlow:
                    calving_k = calving_k_bndlow
                
                # Re-run the regional frontal ablation estimates
                output_df, reg_calving_gta_mod, reg_calving_gta_obs = (
                    reg_calving_flux(main_glac_rgi, calving_k, fa_glac_data_reg=fa_glac_data_reg, 
                                     frontal_ablation_Gta_cn=frontal_ablation_Gta_cn, ignore_nan=ignore_nan,
                                     prms_from_reg_priors=prms_from_reg_priors, prms_from_glac_cal=prms_from_glac_cal, 
                                     debug=True))
                if debug:
                    print('calving_k:', calving_k)
                    print('  fa_data  [Gt/yr]:', np.round(reg_calving_gta_obs,2))
                    print('  fa_model [Gt/yr] :', np.round(reg_calving_gta_mod,2))
    
            # Set lower bound
            calving_k_bndlow = calving_k
            reg_calving_gta_mod_bndlow = reg_calving_gta_mod
            # Set upper bound
            calving_k_bndhigh = calving_k_last
            reg_calving_gta_mod_bndhigh = reg_calving_gta_mod_last
    
    
        # ----- Optimize further using mid-point "bisection" method -----
        # Consider replacing with scipy.optimize.brent
        if np.abs(reg_calving_gta_mod_bndhigh - reg_calving_gta_obs) / reg_calving_gta_obs < 0.01:
            calving_k = calving_k_bndhigh
            reg_calving_gta_mod = reg_calving_gta_mod_bndhigh
        elif np.abs(reg_calving_gta_mod_bndlow - reg_calving_gta_obs) / reg_calving_gta_obs < 0.01:
            calving_k = calving_k_bndlow
            reg_calving_gta_mod = reg_calving_gta_mod_bndlow
        else:
            # Calibrate between limited range
            nround = 0
            while np.abs(reg_calving_gta_mod - reg_calving_gta_obs) / reg_calving_gta_obs > 0.01 and nround <= nround_max:
                
                nround += 1
                if debug:
                    print('Round', nround)
                    
                # Update calving_k using midpoint
                calving_k = (calving_k_bndlow + calving_k_bndhigh) / 2
                output_df, reg_calving_gta_mod, reg_calving_gta_obs = (
                    reg_calving_flux(main_glac_rgi, calving_k, fa_glac_data_reg=fa_glac_data_reg, 
                                     frontal_ablation_Gta_cn=frontal_ablation_Gta_cn, ignore_nan=ignore_nan,
                                     prms_from_reg_priors=prms_from_reg_priors, prms_from_glac_cal=prms_from_glac_cal, 
                                     debug=True))
                
                if debug:
                    print('calving_k:', calving_k)
                    print('  fa_data  [Gt/yr]:', np.round(reg_calving_gta_obs,2))
                    print('  fa_model [Gt/yr] :', np.round(reg_calving_gta_mod,2))
                
                # Update bounds
                if reg_calving_gta_mod < reg_calving_gta_obs:
                    # Update lower bound
                    reg_calving_gta_mod_bndlow = reg_calving_gta_mod
                    calving_k_bndlow = calving_k
                else:
                    # Update upper bound
                    reg_calving_gta_mod_bndhigh = reg_calving_gta_mod
                    calving_k_bndhigh = calving_k
                    
                    
        #%%
        # ----- EXPORT MODEL RESULTS -----
        if fa_glac_data_reg.loc[0,'RGIId'] == 'all':
            output_df['fa_gta_obs'] = np.nan
            output_df['fa_gta_obs_unc'] = np.nan
        else:
            fa_obs_dict = dict(zip(fa_glac_data_reg.RGIId, fa_glac_data_reg[frontal_ablation_Gta_cn].values))
#            fa_obs_unc_dict = dict(zip(fa_glac_data_reg.RGIId, fa_glac_data_reg.frontal_ablation_unc_Gta))
#            fa_glacname_dict = dict(zip(fa_glac_data_reg.RGIId, fa_glac_data_reg.glacier_name))
            output_df['fa_gta_obs'] = output_df['RGIId'].map(fa_obs_dict)
#            output_df['fa_gta_obs_unc'] = output_df['RGIId'].map(fa_obs_unc_dict)
#            output_df['name'] = output_df['RGIId'].map(fa_glacname_dict)
        
        rgi_area_dict = dict(zip(main_glac_rgi.RGIId, main_glac_rgi.Area))
        output_df['area_km2'] = output_df['RGIId'].map(rgi_area_dict)
        
        if cfl_number is None:
            output_df['cfl_number'] = pygem_prms.cfl_number
        else:
            output_df['cfl_number'] = cfl_number
        
        
        output_fp = pygem_prms.main_directory + '/../calving_data/analysis/'
        if not os.path.exists(output_fp):
            os.makedirs(output_fp)
        output_fn = str(reg) + '-calving_cal_reg.csv'
        output_df.to_csv(output_fp + output_fn, index=False)
        #%%
        # ----- PLOT RESULTS FOR EACH GLACIER -----
        if not fa_glac_data_reg.loc[0,'RGIId'] == 'all':
            
            plot_max_raw = np.max([output_df.calving_flux_Gta.max(), output_df.fa_gta_obs.max()])
            plot_max = 10**np.ceil(np.log10(plot_max_raw))
    
            plot_min_raw = np.max([output_df.calving_flux_Gta.min(), output_df.fa_gta_obs.min()])
            plot_min = 10**np.floor(np.log10(plot_min_raw))
    
            x_min, x_max = plot_min, plot_max
            
        
            fig, ax = plt.subplots(1, 1, squeeze=False, gridspec_kw = {'wspace':0, 'hspace':0})
            
            # Marker size
            glac_area_all = output_df['area_km2'].values
            s_sizes = [10,50,250,1000]
            s_byarea = np.zeros(glac_area_all.shape) + s_sizes[3]
            s_byarea[(glac_area_all < 10)] = s_sizes[0]
            s_byarea[(glac_area_all >= 10) & (glac_area_all < 100)] = s_sizes[1]
            s_byarea[(glac_area_all >= 100) & (glac_area_all < 1000)] = s_sizes[2]
            
            sc = ax[0,0].scatter(output_df['fa_gta_obs'], output_df['calving_flux_Gta'], 
                                 color='k', marker='o', linewidth=1, facecolor='none', 
                                 s=s_byarea, clip_on=True)
            
            # Labels
            ax[0,0].set_xlabel('Observed $A_{f}$ (Gt/yr)', size=12)    
            ax[0,0].set_ylabel('Modeled $A_{f}$ (Gt/yr)', size=12)
            ax[0,0].set_xlim(x_min,x_max)
            ax[0,0].set_ylim(x_min,x_max)
            ax[0,0].plot([x_min, x_max], [x_min, x_max], color='k', linewidth=0.5, zorder=1)
            
            calving_k = np.nanmean(output_df.calving_k.values)
            ax[0,0].text(0.97, 0.03, '$k_{f} =$' + str(np.round(calving_k,3)), size=10, color='grey',
                         horizontalalignment='right', verticalalignment='bottom',
                         transform=ax[0,0].transAxes)
            
            # Log scale
            ax[0,0].set_xscale('log')
            ax[0,0].set_yscale('log')
            
            # Legend
            obs_labels = ['< 10', '10-100', '100-1000', '> 1000']
            for nlabel, obs_label in enumerate(obs_labels):
                ax[0,0].scatter([-10],[-10], color='grey', marker='o', linewidth=1, 
                                facecolor='none', s=s_sizes[nlabel], zorder=3, label=obs_label)
            ax[0,0].text(1.08, 0.97, 'Area (km$^{2}$)', size=12, horizontalalignment='left', verticalalignment='top', 
                         transform=ax[0,0].transAxes, color='grey')
            leg = ax[0,0].legend(loc='upper left', ncol=1, fontsize=10, frameon=False,
                                 handletextpad=1, borderpad=0.25, labelspacing=1, labelcolor='grey',
                                 bbox_to_anchor=(1.035, 0.9))
            
    #        # Legend (over plot)
    #        obs_labels = ['< 10', '10-10$^{2}$', '10$^{2}$-10$^{3}$', '> 10$^{3}$']
    #        for nlabel, obs_label in enumerate(obs_labels):
    #            ax[0,0].scatter([-10],[-10], color='grey', marker='o', linewidth=1, 
    #                            facecolor='none', s=s_sizes[nlabel], zorder=3, label=obs_label)
    #        ax[0,0].text(0.06, 0.98, 'Area (km$^{2}$)', size=12, horizontalalignment='left', verticalalignment='top', 
    #                     transform=ax[0,0].transAxes, color='grey')
    #        leg = ax[0,0].legend(loc='upper left', ncol=1, fontsize=10, frameon=False,
    #                             handletextpad=1, borderpad=0.25, labelspacing=0.4, bbox_to_anchor=(0.0, 0.93),
    #                             labelcolor='grey')
            
            # Save figure
            fig.set_size_inches(3.45,3.45)
            fig_fullfn = output_fp + str(reg) + '-calving_glac_compare-cal_reg.png'
            fig.savefig(fig_fullfn, bbox_inches='tight', dpi=300)


#%% ===== INDIVIDUAL CALIBRATION =========================================================================================        
if option_ind_calving_k:
    
    # Load calving glacier data
    calving_fp = pygem_prms.main_directory + '/../calving_data/'
#    calving_fn = 'Northern_hemisphere_calving_flux_Kochtitzky_et_al_for_David_Rounce_with_melt_v8-wromainMB-w17_19.csv'
    calving_fn = 'Northern_hemisphere_calving_flux_Kochtitzky_et_al_for_David_Rounce_with_melt_v13-wromainMB-w17_19.csv'
    fa_glac_data = pd.read_csv(calving_fp + calving_fn)
    hugonnet_fn = 'df_pergla_global_20yr-filled.csv'
    mb_data = pd.read_csv(pygem_prms.hugonnet_fp + hugonnet_fn)
    
    fa_glac_data['O1Region'] = [int(x.split('-')[1].split('.')[0]) for x in fa_glac_data.RGIId.values]
    
    calving_k_bndhigh_set = np.copy(calving_k_bndhigh)
    calving_k_bndlow_set = np.copy(calving_k_bndlow)
    calving_k_step_set = np.copy(calving_k_step)
    
    for reg in regions:
        
        if not os.path.exists(output_fp):
            os.makedirs(output_fp)
        output_fn = str(reg) + '-calving_cal_ind.csv'

        # Regional data
        fa_glac_data_reg = fa_glac_data.loc[fa_glac_data['O1Region'] == reg, :].copy()
        fa_glac_data_reg.reset_index(inplace=True, drop=True)
        
        
        fa_glac_data_reg['glacno'] = np.nan
#        fa_glac_data_reg['glacno'] = [str(int(x.split('-')[1].split('.')[0])) + '.' + x.split('-')[1].split('.')[1]
#                                      for x in fa_glac_data_reg.RGIId]
        
        for nglac, rgiid in enumerate(fa_glac_data_reg.RGIId):
            # Avoid regional data and observations from multiple RGIIds (len==14)
            if not fa_glac_data_reg.loc[nglac,'RGIId'] == 'all' and len(fa_glac_data_reg.loc[nglac,'RGIId']) == 14:
                fa_glac_data_reg.loc[nglac,'glacno'] = (str(int(rgiid.split('-')[1].split('.')[0])) + '.' + 
                                                        rgiid.split('-')[1].split('.')[1])
        
        # Drop observations that aren't of individual glaciers
        fa_glac_data_reg = fa_glac_data_reg.dropna(axis=0, subset=['glacno'])
        fa_glac_data_reg.reset_index(inplace=True, drop=True)
        reg_calving_gta_obs = fa_glac_data_reg[frontal_ablation_Gta_cn].sum()
        
        # Glacier numbers for model runs
        glacno_reg_wdata = sorted(list(fa_glac_data_reg.glacno.values))
        
        main_glac_rgi_all = modelsetup.selectglaciersrgitable(glac_no=glacno_reg_wdata)
        # Tidewater glaciers
        termtype_list = [1,5]
        main_glac_rgi = main_glac_rgi_all.loc[main_glac_rgi_all['TermType'].isin(termtype_list)]
        main_glac_rgi.reset_index(inplace=True, drop=True)
        
        # ----- QUALITY CONTROL USING MB_CLIM COMPARED TO REGIONAL MASS BALANCE -----
        mb_data['O1Region'] = [int(x.split('-')[1].split('.')[0]) for x in mb_data.RGIId.values]
        mb_data_reg = mb_data.loc[mb_data['O1Region'] == reg, :]
        mb_data_reg.reset_index(inplace=True)

        mb_clim_reg_avg = np.mean(mb_data_reg.mb_mwea)
        mb_clim_reg_std = np.std(mb_data_reg.mb_mwea)
        mb_clim_reg_3std = mb_clim_reg_avg + 3*mb_clim_reg_std
        mb_clim_reg_max = np.max(mb_data_reg.mb_mwea)
        mb_clim_reg_3std_min = mb_clim_reg_avg - 3*mb_clim_reg_std
        print('mb_clim_reg_avg:', np.round(mb_clim_reg_avg,2), '+/-', np.round(mb_clim_reg_std,2))
        print('mb_clim_3std (neg):', np.round(mb_clim_reg_3std_min,2))
        print('mb_clim_3std (pos):', np.round(mb_clim_reg_3std,2))
        print('mb_clim_min:', np.round(mb_data_reg.mb_mwea.min(),2))
        print('mb_clim_max:', np.round(mb_clim_reg_max,2))
            
        if not os.path.exists(output_fp + output_fn) or overwrite:

            output_cns = ['RGIId', 'calving_k', 'calving_k_nmad', 'calving_thick', 'calving_flux_Gta', 'fa_gta_obs', 'fa_gta_obs_unc', 'fa_gta_max', 
                          'calving_flux_Gta_bndlow', 'calving_flux_Gta_bndhigh', 'no_errors', 'oggm_dynamics', 
                          'mb_clim_gta', 'mb_total_gta', 'mb_clim_mwea', 'mb_total_mwea']
            
            output_df_all = pd.DataFrame(np.zeros((main_glac_rgi.shape[0],len(output_cns))), columns=output_cns)
            output_df_all['RGIId'] = main_glac_rgi.RGIId
            output_df_all['calving_k_nmad'] = 0
            
            #%%
            # Load observations 
            fa_obs_dict = dict(zip(fa_glac_data_reg.RGIId, fa_glac_data_reg[frontal_ablation_Gta_cn]))
            fa_obs_unc_dict = dict(zip(fa_glac_data_reg.RGIId, fa_glac_data_reg[frontal_ablation_Gta_unc_cn]))
    #        fa_glacname_dict = dict(zip(fa_glac_data_reg.RGIId, fa_glac_data_reg.glacier_name))
            rgi_area_dict = dict(zip(main_glac_rgi.RGIId, main_glac_rgi.Area))
            
            output_df_all['fa_gta_obs'] = output_df_all['RGIId'].map(fa_obs_dict)
            output_df_all['fa_gta_obs_unc'] = output_df_all['RGIId'].map(fa_obs_unc_dict)
    #        output_df_all['name'] = output_df_all['RGIId'].map(fa_glacname_dict)
            output_df_all['area_km2'] = output_df_all['RGIId'].map(rgi_area_dict)
            
            # ----- LOAD DATA ON MB_CLIM CORRECTED FOR FRONTAL ABLATION -----
            # use this to assess reasonableness of results and see if calving_k values affected
            fa_rgiids_list = list(fa_glac_data_reg.RGIId)
            output_df_all['mb_total_gta_obs'] = np.nan
            output_df_all['mb_clim_gta_obs'] = np.nan
            output_df_all['mb_total_mwea_obs'] = np.nan
            output_df_all['mb_clim_mwea_obs'] = np.nan
#            output_df_all['thick_measured_yn'] = np.nan
            for nglac, rgiid in enumerate(list(output_df_all.RGIId)):
                fa_idx = fa_rgiids_list.index(rgiid)
                output_df_all.loc[nglac, 'mb_total_gta_obs'] = fa_glac_data_reg.loc[fa_idx, 'Romain_gta_mbtot']
                output_df_all.loc[nglac, 'mb_clim_gta_obs'] = fa_glac_data_reg.loc[fa_idx, 'Romain_gta_mbclim']
                output_df_all.loc[nglac, 'mb_total_mwea_obs'] = fa_glac_data_reg.loc[fa_idx, 'Romain_mwea_mbtot']
                output_df_all.loc[nglac, 'mb_clim_mwea_obs'] = fa_glac_data_reg.loc[fa_idx, 'Romain_mwea_mbclim']
#                output_df_all.loc[nglac, 'thick_measured_yn'] = fa_glac_data_reg.loc[fa_idx, 'thick_measured_yn']

            # ----- CORRECT TOO POSITIVE CLIMATIC MASS BALANCES -----
            output_df_all['mb_clim_gta'] = output_df_all['mb_clim_gta_obs']
            output_df_all['mb_total_gta'] = output_df_all['mb_total_gta_obs']
            output_df_all['mb_clim_mwea'] = output_df_all['mb_clim_mwea_obs']
            output_df_all['mb_total_mwea'] = output_df_all['mb_total_mwea_obs']
            output_df_all['fa_gta_max'] = output_df_all['fa_gta_obs']
            
            output_df_badmbclim = output_df_all.loc[output_df_all.mb_clim_mwea_obs > mb_clim_reg_3std]
            # Correct by using mean + 3std as maximum climatic mass balance
            if output_df_badmbclim.shape[0] > 0:
                rgiids_toopos = list(output_df_badmbclim.RGIId)

                for nglac, rgiid in enumerate(list(output_df_all.RGIId)):
                    if rgiid in rgiids_toopos:
                        # Specify maximum frontal ablation based on maximum climatic mass balance
                        mb_clim_mwea = mb_clim_reg_3std
                        area_m2 = output_df_all.loc[nglac,'area_km2'] * 1e6
                        mb_clim_gta = mwea_to_gta(mb_clim_mwea, area_m2)
                        mb_total_gta = output_df_all.loc[nglac,'mb_total_gta_obs']
                        
                        fa_gta_max = mb_clim_gta - mb_total_gta
                        
                        output_df_all.loc[nglac,'fa_gta_max'] = fa_gta_max
                        output_df_all.loc[nglac,'mb_clim_mwea'] = mb_clim_mwea
                        output_df_all.loc[nglac,'mb_clim_gta'] = mb_clim_gta
            
            # ---- FIRST ROUND CALIBRATION -----
            # ----- OPTIMIZE CALVING_K BASED ON INDIVIDUAL GLACIER FRONTAL ABLATION DATA -----
            for nglac in np.arange(main_glac_rgi.shape[0]):

#                if main_glac_rgi.loc[nglac,'RGIId'] in ['RGI60-03.00108']:
                for batman in [0]:
                    # Reset bounds
                    calving_k = calving_k_init
                    calving_k_bndlow = np.copy(calving_k_bndlow_set)
                    calving_k_bndhigh = np.copy(calving_k_bndhigh_set)
                    calving_k_step = np.copy(calving_k_step_set)
                    
                    # Select individual glacier
                    main_glac_rgi_ind = main_glac_rgi.loc[[nglac],:]
                    main_glac_rgi_ind.reset_index(inplace=True, drop=True)
                    rgiid_ind = main_glac_rgi_ind.loc[0,'RGIId']
                    fa_glac_data_ind = fa_glac_data_reg.loc[fa_glac_data_reg.RGIId == rgiid_ind, :]
                    fa_glac_data_ind.reset_index(inplace=True, drop=True)
                    
                    # Update the data
                    fa_gta_max = output_df_all.loc[nglac,'fa_gta_max']
                    if fa_glac_data_ind.loc[0,frontal_ablation_Gta_cn] > fa_gta_max:
                        reg_calving_gta_obs = fa_gta_max
                        fa_glac_data_ind.loc[0,frontal_ablation_Gta_cn] = fa_gta_max

                    # Check bounds
                    bndlow_good = True
                    bndhigh_good = True
                    try:
                        output_df_bndhigh, reg_calving_gta_mod_bndhigh, reg_calving_gta_obs = (
                                reg_calving_flux(main_glac_rgi_ind, calving_k_bndhigh, fa_glac_data_reg=fa_glac_data_ind, 
                                                 frontal_ablation_Gta_cn=frontal_ablation_Gta_cn, 
                                                 prms_from_reg_priors=prms_from_reg_priors, prms_from_glac_cal=prms_from_glac_cal,
                                                 ignore_nan=False, debug=debug_reg_calving_fxn))
                    except:
                        bndhigh_good = False
                        reg_calving_gta_mod_bndhigh = None
                    try:
                        output_df_bndlow, reg_calving_gta_mod_bndlow, reg_calving_gta_obs = (
                                reg_calving_flux(main_glac_rgi_ind, calving_k_bndlow, fa_glac_data_reg=fa_glac_data_ind,
                                                 frontal_ablation_Gta_cn=frontal_ablation_Gta_cn, 
                                                 prms_from_reg_priors=prms_from_reg_priors, prms_from_glac_cal=prms_from_glac_cal,
                                                 ignore_nan=False, debug=debug_reg_calving_fxn))
                    except:
                        bndlow_good = False
                        reg_calving_gta_mod_bndlow = None
                        
                    # Record bounds
                    output_df_all.loc[nglac,'calving_flux_Gta_bndlow'] = reg_calving_gta_mod_bndlow
                    output_df_all.loc[nglac,'calving_flux_Gta_bndhigh'] = reg_calving_gta_mod_bndhigh
                    
                    if debug:
                        print('  fa_data  [Gt/yr]:', np.round(reg_calving_gta_obs,4))
                        print('  fa_model_bndlow [Gt/yr] :', reg_calving_gta_mod_bndlow)
                        print('  fa_model_bndhigh [Gt/yr] :', reg_calving_gta_mod_bndhigh)
                    
                        
                    run_opt = False
                    if bndhigh_good and bndlow_good:
                        if reg_calving_gta_obs < reg_calving_gta_mod_bndlow:
                            output_df_all.loc[nglac,'calving_k'] = output_df_bndlow.loc[0,'calving_k']
                            output_df_all.loc[nglac,'calving_thick'] = output_df_bndlow.loc[0,'calving_thick']
                            output_df_all.loc[nglac,'calving_flux_Gta'] = output_df_bndlow.loc[0,'calving_flux_Gta']
                            output_df_all.loc[nglac,'no_errors'] = output_df_bndlow.loc[0,'no_errors']
                            output_df_all.loc[nglac,'oggm_dynamics'] = output_df_bndlow.loc[0,'oggm_dynamics']
                        elif reg_calving_gta_obs > reg_calving_gta_mod_bndhigh:
                            output_df_all.loc[nglac,'calving_k'] = output_df_bndhigh.loc[0,'calving_k']
                            output_df_all.loc[nglac,'calving_thick'] = output_df_bndhigh.loc[0,'calving_thick']
                            output_df_all.loc[nglac,'calving_flux_Gta'] = output_df_bndhigh.loc[0,'calving_flux_Gta']
                            output_df_all.loc[nglac,'no_errors'] = output_df_bndhigh.loc[0,'no_errors']
                            output_df_all.loc[nglac,'oggm_dynamics'] = output_df_bndhigh.loc[0,'oggm_dynamics']
                        else:
                            run_opt = True
                    else:
                        run_opt = True
                    
                    if run_opt:
                        output_df, calving_k = run_opt_fa(main_glac_rgi_ind, calving_k, calving_k_bndlow, calving_k_bndhigh, 
                                                          fa_glac_data_ind, frontal_ablation_Gta_cn=frontal_ablation_Gta_cn, 
                                                          prms_from_reg_priors=prms_from_reg_priors, prms_from_glac_cal=prms_from_glac_cal,
                                                          ignore_nan=False, debug=debug_reg_calving_fxn)
                        calving_k_med = np.copy(calving_k)
                        output_df_all.loc[nglac,'calving_k'] = output_df.loc[0,'calving_k']
                        output_df_all.loc[nglac,'calving_thick'] = output_df.loc[0,'calving_thick']
                        output_df_all.loc[nglac,'calving_flux_Gta'] = output_df.loc[0,'calving_flux_Gta']
                        output_df_all.loc[nglac,'no_errors'] = output_df.loc[0,'no_errors']
                        output_df_all.loc[nglac,'oggm_dynamics'] = output_df.loc[0,'oggm_dynamics']
                    
                        #%% ----- ADD UNCERTAINTY -----
                        # Upper uncertainty
                        print('\n\n----- upper uncertainty:')
                        fa_glac_data_ind_high = fa_glac_data_ind.copy()
                        fa_gta_obs_high = fa_glac_data_ind.loc[0,'fa_gta_obs'] + fa_glac_data_ind.loc[0,'fa_gta_obs_unc']
                        fa_glac_data_ind_high.loc[0,'fa_gta_obs'] = fa_gta_obs_high
                        calving_k_bndlow_upper = np.copy(calving_k_med) - 0.01
                        calving_k_start = np.copy(calving_k_med)
                        output_df, calving_k = run_opt_fa(main_glac_rgi_ind, calving_k_start, calving_k_bndlow_upper, calving_k_bndhigh, 
                                                          fa_glac_data_ind_high, frontal_ablation_Gta_cn=frontal_ablation_Gta_cn, 
                                                          prms_from_reg_priors=prms_from_reg_priors, prms_from_glac_cal=prms_from_glac_cal,
                                                          ignore_nan=False, debug=debug_reg_calving_fxn)
                        calving_k_nmadhigh = np.copy(calving_k)
                        
                        if debug:
                            print('calving_k:', np.round(calving_k,2), 'fa_data high:', np.round(fa_glac_data_ind_high.loc[0,'fa_gta_obs'],4),
                                  'fa_mod high:', np.round(output_df.loc[0,'calving_flux_Gta'],4))
                            
                        # Lower uncertainty
                        print('\n\n----- lower uncertainty:')

                        fa_glac_data_ind_low = fa_glac_data_ind.copy()
                        fa_gta_obs_low = fa_glac_data_ind.loc[0,'fa_gta_obs'] - fa_glac_data_ind.loc[0,'fa_gta_obs_unc']
                        if fa_gta_obs_low < 0:
                            calving_k_nmadlow = calving_k_med - abs(calving_k_nmadhigh - calving_k_med)
                            if debug:
                                print('calving_k:', np.round(calving_k_nmadlow,2), 'fa_data low:', np.round(fa_gta_obs_low,4))
                        else:
                            fa_glac_data_ind_low.loc[0,'fa_gta_obs'] = fa_gta_obs_low
                            calving_k_bndhigh_lower = np.copy(calving_k_med) + 0.01
                            calving_k_start = np.copy(calving_k_med)
                            output_df, calving_k = run_opt_fa(main_glac_rgi_ind, calving_k_start, calving_k_bndlow, calving_k_bndhigh_lower,
                                                              fa_glac_data_ind_low, 
                                                              calving_k_step=(calving_k_med - calving_k_bndlow) / 10,
                                                              frontal_ablation_Gta_cn=frontal_ablation_Gta_cn, 
                                                              prms_from_reg_priors=prms_from_reg_priors, prms_from_glac_cal=prms_from_glac_cal,
                                                              ignore_nan=False, debug=debug_reg_calving_fxn)
                            calving_k_nmadlow = np.copy(calving_k)
                            if debug:
                                print('calving_k:', np.round(calving_k,2), 'fa_data low:', np.round(fa_glac_data_ind_low.loc[0,'fa_gta_obs'],4),
                                      'fa_mod low:', np.round(output_df.loc[0,'calving_flux_Gta'],4))
                        
                        
                        calving_k_nmad = np.mean([abs(calving_k_nmadhigh - calving_k_med), abs(calving_k_nmadlow - calving_k_med)])
                    
                        # Final
                        if debug:
                            print('----- final -----')
                            print(rgiid, 'calving_k (med/high/low/nmad):', np.round(calving_k_med,2), 
                                  np.round(calving_k_nmadhigh,2), np.round(calving_k_nmadlow,2), np.round(calving_k_nmad,2))
                    
                        output_df_all.loc[nglac,'calving_k_nmad'] = calving_k_nmad

#            # Glaciers at bounds, have calving_k_nmad based on regional mean
#            output_df_all_subset = output_df_all.loc[output_df_all.calving_k_nmad > 0, :]
#            calving_k_nmad = 1.4826 * median_abs_deviation(output_df_all_subset.calving_k)
#            output_df_all.loc[output_df_all['calving_k_nmad']==0,'calving_k_nmad'] = calving_k_nmad
                        
            
                        
            # ----- EXPORT MODEL RESULTS -----
            output_df_all.to_csv(output_fp + output_fn, index=False)
        
        else:
            output_df_all = pd.read_csv(output_fp + output_fn)


        output_df_all_rnd1 = output_df_all.copy()
        
        
        #%%
        # ----- VIEW DIAGNOSTICS OF 'GOOD' GLACIERS -----
        # special for 17 because so few 'good' glaciers
        if reg in [17]:
            output_df_all_good = output_df_all.loc[(output_df_all['calving_k'] < calving_k_bndhigh_set), :]
        else:
            output_df_all_good = output_df_all.loc[(output_df_all['fa_gta_obs'] == output_df_all['fa_gta_max']) & 
                                                   (output_df_all['calving_k'] < calving_k_bndhigh_set), :]
        
        rgiids_good = list(output_df_all_good.RGIId)
            
        calving_k_reg_mean = output_df_all_good.calving_k.mean()
        print(' calving_k mean/med:', np.round(calving_k_reg_mean,2), 
                                      np.round(np.median(output_df_all_good.calving_k),2))
        
        output_df_all['calving_flux_Gta_rnd1'] = output_df_all['calving_flux_Gta'].copy()
        output_df_all['calving_k_rnd1'] = output_df_all['calving_k'].copy()
            
        
        # ----- PLOT RESULTS FOR EACH GLACIER -----
        plot_max_raw = np.max([output_df_all_good.calving_flux_Gta.max(), output_df_all_good.fa_gta_obs.max()])
        plot_max = 10**np.ceil(np.log10(plot_max_raw))

        plot_min_raw = np.max([output_df_all_good.calving_flux_Gta.min(), output_df_all_good.fa_gta_obs.min()])
        plot_min = 10**np.floor(np.log10(plot_min_raw))
        if plot_min < 1e-3:
            plot_min = 1e-4

        x_min, x_max = plot_min, plot_max
        
        fig, ax = plt.subplots(2, 2, squeeze=False, gridspec_kw = {'wspace':0.4, 'hspace':0.4})
        
        # ----- Scatter plot -----
        # Marker size
        glac_area_all = output_df_all_good['area_km2'].values
        s_sizes = [10,50,250,1000]
        s_byarea = np.zeros(glac_area_all.shape) + s_sizes[3]
        s_byarea[(glac_area_all < 10)] = s_sizes[0]
        s_byarea[(glac_area_all >= 10) & (glac_area_all < 100)] = s_sizes[1]
        s_byarea[(glac_area_all >= 100) & (glac_area_all < 1000)] = s_sizes[2]
        
        sc = ax[0,0].scatter(output_df_all_good['fa_gta_obs'], output_df_all_good['calving_flux_Gta'], 
                             color='k', marker='o', linewidth=1, facecolor='none', 
                             s=s_byarea, clip_on=True)
        # Labels
        ax[0,0].set_xlabel('Observed $A_{f}$ (Gt/yr)', size=12)    
        ax[0,0].set_ylabel('Modeled $A_{f}$ (Gt/yr)', size=12)
        ax[0,0].set_xlim(x_min,x_max)
        ax[0,0].set_ylim(x_min,x_max)
        ax[0,0].plot([x_min, x_max], [x_min, x_max], color='k', linewidth=0.5, zorder=1)
        # Log scale
        ax[0,0].set_xscale('log')
        ax[0,0].set_yscale('log')
        
        # Legend
        obs_labels = ['< 10', '10-10$^{2}$', '10$^{2}$-10$^{3}$', '> 10$^{3}$']
        for nlabel, obs_label in enumerate(obs_labels):
            ax[0,0].scatter([-10],[-10], color='grey', marker='o', linewidth=1, 
                            facecolor='none', s=s_sizes[nlabel], zorder=3, label=obs_label)
        ax[0,0].text(0.06, 0.98, 'Area (km$^{2}$)', size=12, horizontalalignment='left', verticalalignment='top', 
                     transform=ax[0,0].transAxes, color='grey')
        leg = ax[0,0].legend(loc='upper left', ncol=1, fontsize=10, frameon=False,
                             handletextpad=1, borderpad=0.25, labelspacing=0.4, bbox_to_anchor=(0.0, 0.93),
                             labelcolor='grey')

        # ----- Histogram -----
#        nbins = 25
#        ax[0,1].hist(output_df_all_good['calving_k'], bins=nbins, color='grey', edgecolor='k')
        vn_bins = np.arange(0, np.max([1,output_df_all_good.calving_k.max()]) + 0.1, 0.1)
        hist, bins = np.histogram(output_df_all_good.loc[output_df_all_good['no_errors'] == 1, 'calving_k'], bins=vn_bins)
        ax[0,1].bar(x=vn_bins[:-1] + 0.1/2, height=hist, width=(bins[1]-bins[0]), 
                             align='center', edgecolor='black', color='grey')
        ax[0,1].set_xticks(np.arange(0,np.max([1,vn_bins.max()])+0.1, 1))
        ax[0,1].set_xticks(vn_bins, minor=True)
        ax[0,1].set_xlim(vn_bins.min(), np.max([1,vn_bins.max()]))
        if hist.max() < 40:
            y_major_interval = 5
            y_max = np.ceil(hist.max()/y_major_interval)*y_major_interval
            ax[0,1].set_yticks(np.arange(0,y_max+y_major_interval,y_major_interval))
        elif hist.max() > 40:
            y_major_interval = 10
            y_max = np.ceil(hist.max()/y_major_interval)*y_major_interval
            ax[0,1].set_yticks(np.arange(0,y_max+y_major_interval,y_major_interval))
        
        # Labels
        ax[0,1].set_xlabel('$k_{f}$ (yr$^{-1}$)', size=12)
        ax[0,1].set_ylabel('Count (glaciers)', size=12)
        
        
        # ----- CALVING_K VS MB_CLIM -----
        ax[1,0].scatter(output_df_all_good['calving_k'], output_df_all_good['mb_clim_mwea'], 
                        color='k', marker='o', linewidth=1, facecolor='none', 
                        s=s_byarea, clip_on=True)
        ax[1,0].set_xlabel('$k_{f}$ (yr$^{-1}$)', size=12)
        ax[1,0].set_ylabel('$B_{clim}$ (mwea)', size=12)
        
        # ----- CALVING_K VS AREA -----
        ax[1,1].scatter(output_df_all_good['area_km2'], output_df_all_good['calving_k'], 
                        color='k', marker='o', linewidth=1, facecolor='none', 
                        s=s_byarea, clip_on=True)
        ax[1,1].set_xlabel('Area (km2)', size=12)
        ax[1,1].set_ylabel('$k_{f}$ (yr$^{-1}$)', size=12)
        
        # Correlation
        slope, intercept, r_value, p_value, std_err = linregress(output_df_all_good['area_km2'], 
                                                                 output_df_all_good['calving_k'],)
        print('  r_value =', np.round(r_value,2), 'slope = ', np.round(slope,5), 
              'intercept = ', np.round(intercept,5), 'p_value = ', np.round(p_value,6))
        area_min = 0
        area_max = output_df_all_good.area_km2.max()
        ax[1,1].plot([area_min, area_max], [intercept+slope*area_min, intercept+slope*area_max], color='k')
        
        # Save figure
        fig.set_size_inches(6,6)
        if not os.path.exists(output_fp):
            os.makedirs(output_fp)
        fig_fullfn = output_fp + str(reg) + '-calving_glac_compare-cal_ind-good.png'
        fig.savefig(fig_fullfn, bbox_inches='tight', dpi=300)
        
        
        #%%
        # ----- REPLACE UPPER BOUND CALVING_K WITH MEDIAN CALVING_K -----
        rgiids_bndhigh = list(output_df_all.loc[output_df_all['calving_k'] == calving_k_bndhigh_set,'RGIId'].values)
        for nglac, rgiid in enumerate(output_df_all.RGIId):
            if rgiid in rgiids_bndhigh:
                # Estimate frontal ablation for poor glaciers extrapolated from good ones
                main_glac_rgi_ind = main_glac_rgi.loc[main_glac_rgi.RGIId == rgiid,:]
                main_glac_rgi_ind.reset_index(inplace=True, drop=True)
                fa_glac_data_ind = fa_glac_data_reg.loc[fa_glac_data_reg.RGIId == rgiid, :]
                fa_glac_data_ind.reset_index(inplace=True, drop=True)
                
                calving_k = np.median(output_df_all_good.calving_k)
#                calving_k = intercept + slope * main_glac_rgi_ind.loc[0,'Area']
#                if calving_k > output_df_all_good.calving_k.max():
#                    calving_k = output_df_all_good.calving_k.max()
                
                output_df, reg_calving_gta_mod, reg_calving_gta_obs = (
                        reg_calving_flux(main_glac_rgi_ind, calving_k, fa_glac_data_reg=fa_glac_data_ind,
                                         frontal_ablation_Gta_cn=frontal_ablation_Gta_cn, 
                                         prms_from_reg_priors=prms_from_reg_priors, prms_from_glac_cal=prms_from_glac_cal,
                                         ignore_nan=False, debug=debug_reg_calving_fxn))
                
                output_df_all.loc[nglac,'calving_flux_Gta'] = output_df.loc[0,'calving_flux_Gta']
                output_df_all.loc[nglac,'calving_k'] = output_df.loc[0,'calving_k']
                output_df_all.loc[nglac,'calving_k_nmad'] = np.median(output_df_all_good.calving_k_nmad)
                
        # ----- EXPORT MODEL RESULTS -----
        output_df_all.to_csv(output_fp + output_fn, index=False)
        
        
        #%% ----- PROCESS MISSING GLACIERS WHERE GEODETIC MB IS NOT CORRECTED FOR AREA ABOVE SEA LEVEL LOSSES
        if reg in [1,3,4,5,7,9,17]:
            output_fn_missing = output_fn.replace('.csv','-missing.csv')
            
            main_glac_rgi_all = modelsetup.selectglaciersrgitable(rgi_regionsO1=[reg], rgi_regionsO2='all', 
                                                                  rgi_glac_number='all', 
                                                                  include_landterm=False, include_laketerm=False, 
                                                                  include_tidewater=True)
            rgiids_processed = list(output_df_all.RGIId)
            rgiids_all = list(main_glac_rgi_all.RGIId)
            rgiids_missing = [x for x in rgiids_all if x not in rgiids_processed]
            
            glac_no_missing = [x.split('-')[1] for x in rgiids_missing]
            main_glac_rgi_missing = modelsetup.selectglaciersrgitable(glac_no=glac_no_missing)
            
            print(reg, len(glac_no_missing), main_glac_rgi_missing.Area.sum(), glac_no_missing)
            
            if not os.path.exists(output_fp + output_fn_missing) or overwrite:
    
                # Add regions for median subsets
                output_df_all['O1Region'] = [int(x.split('-')[1].split('.')[0]) for x in output_df_all.RGIId]
                
                # Update mass balance data
                output_df_missing = pd.DataFrame(np.zeros((len(rgiids_missing),len(output_df_all.columns))), columns=output_df_all.columns)
                output_df_missing['RGIId'] = rgiids_missing
                output_df_missing['fa_gta_obs'] = np.nan
                rgi_area_dict = dict(zip(main_glac_rgi_missing.RGIId, main_glac_rgi_missing.Area))
                output_df_missing['area_km2'] = output_df_missing['RGIId'].map(rgi_area_dict)
                rgi_mbobs_dict = dict(zip(mb_data['RGIId'],mb_data['mb_mwea']))
                output_df_missing['mb_clim_mwea_obs'] = output_df_missing['RGIId'].map(rgi_mbobs_dict)
                output_df_missing['mb_clim_gta_obs'] = [mwea_to_gta(output_df_missing.loc[x,'mb_clim_mwea_obs'],
                                                                    output_df_missing.loc[x,'area_km2']*1e6) for x in output_df_missing.index]
                output_df_missing['mb_total_mwea_obs'] = output_df_missing['mb_clim_mwea_obs']
                output_df_missing['mb_total_gta_obs'] = output_df_missing['mb_total_gta_obs']
    
                # Start with median value
                calving_k_med = np.median(output_df_all.loc[output_df_all['O1Region']==reg,'calving_k'])
                for nglac, rgiid in enumerate(rgiids_missing):
                    
#                    if rgiid in ['RGI60-01.21008']:
                    for batman in [0]:
                        main_glac_rgi_ind = modelsetup.selectglaciersrgitable(glac_no=[rgiid.split('-')[1]])
                        # Estimate frontal ablation for missing glaciers
                        output_df, reg_calving_gta_mod, reg_calving_gta_obs = (
                                reg_calving_flux(main_glac_rgi_ind, calving_k_med, debug=True, calc_mb_geo_correction=True))
                        
                        # Adjust climatic mass balance to account for the losses due to frontal ablation
                        #  add this loss because it'll come from frontal ablation instead of climatic mass balance
                        mb_clim_fa_corrected = (output_df_missing.loc[nglac,'mb_clim_mwea_obs'] + 
                                                output_df.loc[0,'mb_mwea_fa_asl_lost'])
                        
                        mb_clim_reg_95 = (mb_clim_reg_avg + 1.96*mb_clim_reg_std)
    
                        print('mb_clim (raw):', np.round(output_df_missing.loc[nglac,'mb_clim_mwea_obs'],2))
                        print('mb_clim (fa_corrected):', np.round(mb_clim_fa_corrected,2))
                        print('mb_clim (reg 95%):', np.round(mb_clim_reg_95,2))
                        print('mb_total (95% min):', np.round(mb_clim_reg_3std_min,2))
                        
                        # Set nmad to median value - correct if value reduced
#                        calving_k_nmad_missing = 1.4826*median_abs_deviation(output_df_all_good.calving_k)
                        calving_k_nmad_missing = np.median(output_df_all_good.calving_k_nmad)
                        output_df_missing.loc[nglac,'calving_k_nmad'] = calving_k_nmad_missing
                        
                        if mb_clim_fa_corrected < mb_clim_reg_95:
                            for cn in ['calving_k', 'calving_thick', 'calving_flux_Gta', 'no_errors', 'oggm_dynamics']:
                                output_df_missing.loc[nglac,cn] = output_df.loc[0,cn]
                            output_df_missing.loc[nglac,'mb_clim_mwea'] = mb_clim_fa_corrected
                            output_df_missing.loc[nglac,'mb_clim_gta'] = mwea_to_gta(output_df_missing.loc[nglac,'mb_clim_mwea'], 
                                                                                     output_df_missing.loc[nglac,'area_km2']*1e6)
                            output_df_missing.loc[nglac,'mb_total_gta'] = (output_df_missing.loc[nglac,'mb_clim_gta'] - 
                                                                           output_df_missing.loc[nglac,'calving_flux_Gta'])
                            output_df_missing.loc[nglac,'mb_total_mwea'] = gta_to_mwea(output_df_missing.loc[nglac,'mb_total_gta'], 
                                                                                       output_df_missing.loc[nglac,'area_km2']*1e6)
                        
                        if mb_clim_fa_corrected > mb_clim_reg_95:
                            
                            # Calibrate frontal ablation based on fa_mwea_max
                            #  i.e., the maximum frontal ablation that is consistent with reasonable mb_clim
                            fa_mwea_max = mb_clim_reg_95 - output_df_missing.loc[nglac,'mb_clim_mwea_obs']
                            
                            # Reset bounds
                            calving_k = calving_k_med
                            calving_k_bndlow = np.copy(calving_k_bndlow_set)
                            calving_k_bndhigh = np.copy(calving_k_bndhigh_set)
                            calving_k_step = np.copy(calving_k_step_set)
                            
                            # Select individual glacier
                            rgiid_ind = main_glac_rgi_ind.loc[0,'RGIId']
                            fa_glac_data_ind = pd.DataFrame(np.zeros((1,len(fa_glac_data_reg.columns))), 
                                                            columns=fa_glac_data_reg.columns)
                            fa_glac_data_ind.loc[0,'RGIId'] = rgiid_ind
        
                            # Check bounds
                            bndlow_good = True
                            bndhigh_good = True
                            try:
                                output_df_bndhigh, reg_calving_gta_mod_bndhigh, reg_calving_gta_obs = (
                                        reg_calving_flux(main_glac_rgi_ind, calving_k_bndhigh, fa_glac_data_reg=fa_glac_data_ind, 
                                                         frontal_ablation_Gta_cn=frontal_ablation_Gta_cn, 
                                                         prms_from_reg_priors=prms_from_reg_priors, prms_from_glac_cal=prms_from_glac_cal,
                                                         ignore_nan=False, debug=debug_reg_calving_fxn, calc_mb_geo_correction=True))
                            except:
                                bndhigh_good = False
                                reg_calving_gta_mod_bndhigh = None
            
                            try:
                                output_df_bndlow, reg_calving_gta_mod_bndlow, reg_calving_gta_obs = (
                                        reg_calving_flux(main_glac_rgi_ind, calving_k_bndlow, fa_glac_data_reg=fa_glac_data_ind,
                                                         frontal_ablation_Gta_cn=frontal_ablation_Gta_cn, 
                                                         prms_from_reg_priors=prms_from_reg_priors, prms_from_glac_cal=prms_from_glac_cal,
                                                         ignore_nan=False, debug=debug_reg_calving_fxn, calc_mb_geo_correction=True))
                            except:
                                bndlow_good = False
                                reg_calving_gta_mod_bndlow = None
                                
                            print('mb_mwea_fa_asl_lost_bndhigh:', output_df_bndhigh.loc[0,'mb_mwea_fa_asl_lost'])
                            print('mb_mwea_fa_asl_lost_bndlow:', output_df_bndlow.loc[0,'mb_mwea_fa_asl_lost'])
                  
                            # Record bounds
                            output_df_missing.loc[nglac,'calving_flux_Gta_bndlow'] = reg_calving_gta_mod_bndlow
                            output_df_missing.loc[nglac,'calving_flux_Gta_bndhigh'] = reg_calving_gta_mod_bndhigh
                            
                            if debug:
                                print('  fa_model_bndlow [Gt/yr] :', reg_calving_gta_mod_bndlow)
                                print('  fa_model_bndhigh [Gt/yr] :', reg_calving_gta_mod_bndhigh)
                                
                                    
                            run_opt = True
                            if fa_mwea_max > 0:
                                if bndhigh_good and bndlow_good:
                                    if fa_mwea_max < output_df_bndlow.loc[0,'mb_mwea_fa_asl_lost']:
                                        # Adjust climatic mass balance to note account for the losses due to frontal ablation
                                        mb_clim_fa_corrected = (output_df_missing.loc[nglac,'mb_clim_mwea_obs'] + 
                                                                output_df_bndlow.loc[0,'mb_mwea_fa_asl_lost'])
                                        # Record output
                                        for cn in ['calving_k', 'calving_thick', 'calving_flux_Gta', 'no_errors', 'oggm_dynamics']:
                                            output_df_missing.loc[nglac,cn] = output_df_bndlow.loc[0,cn]
                                        output_df_missing.loc[nglac,'mb_clim_mwea'] = mb_clim_fa_corrected
                                        output_df_missing.loc[nglac,'mb_clim_gta'] = mwea_to_gta(output_df_missing.loc[nglac,'mb_clim_mwea'], 
                                                                                                 output_df_missing.loc[nglac,'area_km2']*1e6)
                                        output_df_missing.loc[nglac,'mb_total_gta'] = (output_df_missing.loc[nglac,'mb_clim_gta'] - 
                                                                                       output_df_missing.loc[nglac,'calving_flux_Gta'])
                                        output_df_missing.loc[nglac,'mb_total_mwea'] = gta_to_mwea(output_df_missing.loc[nglac,'mb_total_gta'], 
                                                                                                   output_df_missing.loc[nglac,'area_km2']*1e6)
                                        
                                        run_opt = False
                                    
                                    elif output_df_bndhigh.loc[0,'mb_mwea_fa_asl_lost'] == output_df_bndlow.loc[0,'mb_mwea_fa_asl_lost']:
                                        # Adjust climatic mass balance to note account for the losses due to frontal ablation
                                        mb_clim_fa_corrected = (output_df_missing.loc[nglac,'mb_clim_mwea_obs'] + 
                                                                output_df.loc[0,'mb_mwea_fa_asl_lost'])
                                        # Record output
                                        for cn in ['calving_k', 'calving_thick', 'calving_flux_Gta', 'no_errors', 'oggm_dynamics']:
                                            output_df_missing.loc[nglac,cn] = output_df.loc[0,cn]
                                        output_df_missing.loc[nglac,'mb_clim_mwea'] = mb_clim_fa_corrected
                                        output_df_missing.loc[nglac,'mb_clim_gta'] = mwea_to_gta(output_df_missing.loc[nglac,'mb_clim_mwea'], 
                                                                                                 output_df_missing.loc[nglac,'area_km2']*1e6)
                                        output_df_missing.loc[nglac,'mb_total_gta'] = (output_df_missing.loc[nglac,'mb_clim_gta'] - 
                                                                                       output_df_missing.loc[nglac,'calving_flux_Gta'])
                                        output_df_missing.loc[nglac,'mb_total_mwea'] = gta_to_mwea(output_df_missing.loc[nglac,'mb_total_gta'], 
                                                                                                   output_df_missing.loc[nglac,'area_km2']*1e6)
                                        run_opt = False
                                     
                                if run_opt:
            #                        mb_clim_fa_corrected = (output_df_missing.loc[nglac,'mb_clim_mwea_obs'] + 
            #                                                output_df.loc[0,'mb_mwea_fa_asl_lost'])
                                    print('\n\n\n-------')
                                    print('mb_clim_obs:', np.round(output_df_missing.loc[nglac,'mb_clim_mwea_obs'],2))
                                    print('mb_clim_fa_corrected:', np.round(mb_clim_fa_corrected,2))
                                    
                                    calving_k_step_missing = (calving_k_med - calving_k_bndlow) / 20
                                    calving_k_next = calving_k - calving_k_step_missing
                                    while output_df.loc[0,'mb_mwea_fa_asl_lost'] > fa_mwea_max and calving_k_next > 0:
                                        calving_k -= calving_k_step_missing
                                        
                                        # Estimate frontal ablation for missing glaciers
                                        output_df, reg_calving_gta_mod, reg_calving_gta_obs = (
                                                reg_calving_flux(main_glac_rgi_ind, calving_k, debug=True, calc_mb_geo_correction=True))
                                        
                                        calving_k_next = calving_k - calving_k_step_missing
            
                                    # Adjust climatic mass balance to note account for the losses due to frontal ablation
                                    mb_clim_fa_corrected = (output_df_missing.loc[nglac,'mb_clim_mwea_obs'] + 
                                                            output_df.loc[0,'mb_mwea_fa_asl_lost'])
                                    # Record output
                                    for cn in ['calving_k', 'calving_thick', 'calving_flux_Gta', 'no_errors', 'oggm_dynamics']:
                                        output_df_missing.loc[nglac,cn] = output_df.loc[0,cn]
                                    output_df_missing.loc[nglac,'mb_clim_mwea'] = mb_clim_fa_corrected
                                    output_df_missing.loc[nglac,'mb_clim_gta'] = mwea_to_gta(output_df_missing.loc[nglac,'mb_clim_mwea'], 
                                                                                             output_df_missing.loc[nglac,'area_km2']*1e6)
                                    output_df_missing.loc[nglac,'mb_total_gta'] = (output_df_missing.loc[nglac,'mb_clim_gta'] - 
                                                                                   output_df_missing.loc[nglac,'calving_flux_Gta'])
                                    output_df_missing.loc[nglac,'mb_total_mwea'] = gta_to_mwea(output_df_missing.loc[nglac,'mb_total_gta'], 
                                                                                               output_df_missing.loc[nglac,'area_km2']*1e6)     
                                    
                                    print('mb_clim_fa_corrected (updated):', np.round(mb_clim_fa_corrected,2))
                            
                            # If mass balance is higher than 95% threshold, then just make sure correction is reasonable (no more than 10%)
                            else:
                                calving_k = calving_k_med
                                calving_k_step_missing = (calving_k_med - calving_k_bndlow) / 20
                                calving_k_next = calving_k - calving_k_step_missing
                                while (output_df.loc[0,'mb_mwea_fa_asl_lost'] > 0.1*output_df_missing.loc[nglac,'mb_clim_mwea_obs'] and 
                                       calving_k_next > 0):
                                    calving_k -= calving_k_step_missing
                                    
                                    # Estimate frontal ablation for missing glaciers
                                    output_df, reg_calving_gta_mod, reg_calving_gta_obs = (
                                            reg_calving_flux(main_glac_rgi_ind, calving_k, debug=True, calc_mb_geo_correction=True))
                                    
                                    calving_k_next = calving_k - calving_k_step_missing
        
                                # Adjust climatic mass balance to note account for the losses due to frontal ablation
                                mb_clim_fa_corrected = (output_df_missing.loc[nglac,'mb_clim_mwea_obs'] + 
                                                        output_df.loc[0,'mb_mwea_fa_asl_lost'])
                                # Record output
                                for cn in ['calving_k', 'calving_thick', 'calving_flux_Gta', 'no_errors', 'oggm_dynamics']:
                                    output_df_missing.loc[nglac,cn] = output_df.loc[0,cn]
                                output_df_missing.loc[nglac,'mb_clim_mwea'] = mb_clim_fa_corrected
                                output_df_missing.loc[nglac,'mb_clim_gta'] = mwea_to_gta(output_df_missing.loc[nglac,'mb_clim_mwea'], 
                                                                                         output_df_missing.loc[nglac,'area_km2']*1e6)
                                output_df_missing.loc[nglac,'mb_total_gta'] = (output_df_missing.loc[nglac,'mb_clim_gta'] - 
                                                                               output_df_missing.loc[nglac,'calving_flux_Gta'])
                                output_df_missing.loc[nglac,'mb_total_mwea'] = gta_to_mwea(output_df_missing.loc[nglac,'mb_total_gta'], 
                                                                                           output_df_missing.loc[nglac,'area_km2']*1e6)       
                                
                                print('mb_clim_fa_corrected (updated):', np.round(mb_clim_fa_corrected,2))
                                
                                
                                # Adjust calving_k_nmad if calving_k is very low to avoid poor values
                                if output_df_missing.loc[nglac,'calving_k'] < calving_k_nmad_missing:
                                    output_df_missing.loc[nglac,'calving_k_nmad'] = output_df_missing.loc[nglac,'calving_k'] - calving_k_bndlow_set    
          
                # Export 
                output_df_missing.to_csv(output_fp + output_fn_missing, index=False)
            else:
                output_df_missing = pd.read_csv(output_fp + output_fn_missing)
            
        
        #%% ----- CORRECTION FOR TOTAL MASS BALANCE IN ANTARCTICA/SUBANTARCTIC -----
        if reg in [19]:
            
#            #%% STATISTICS FOR CALIBRATED GLACIERS 
#            for nglac, rgiid in enumerate(output_df_all.RGIId.values):
#                mb_rgiids = list(mb_data_reg.RGIId.values)
#                mb_idx = mb_rgiids.index(rgiid)
#                output_df_all.loc[nglac,'mb_clim_mwea_obs'] = mb_data_reg.loc[mb_idx,'mb_mwea']
#                output_df_all.loc[nglac,'mb_clim_gta_obs'] = mwea_to_gta(output_df_all.loc[nglac,'mb_clim_mwea_obs'], 
#                                                                         output_df_all.loc[nglac,'area_km2']*1e6)
#                output_df_all.loc[nglac,'mb_total_gta_obs'] = output_df_all.loc[nglac,'mb_clim_gta_obs'] - output_df_all.loc[nglac,'calving_flux_Gta']
#                output_df_all.loc[nglac,'mb_total_mwea_obs'] = gta_to_mwea(output_df_all.loc[nglac,'mb_total_gta_obs'], 
#                                                                           output_df_all.loc[nglac,'area_km2']*1e6)
#                print(mb_data_reg.loc[mb_idx,'RGIId'], rgiid)
#                
##            mb_clim_reg_avg_1std = mb_clim_reg_avg + mb_clim_reg_std
##            print('clim threshold:', np.round(mb{}))
#            #%%
            
            output_fn_missing = output_fn.replace('.csv','-missing.csv')
            
            main_glac_rgi_all = modelsetup.selectglaciersrgitable(rgi_regionsO1=[reg], rgi_regionsO2='all', 
                                                                  rgi_glac_number='all', 
                                                                  include_landterm=False, include_laketerm=False, 
                                                                  include_tidewater=True)
            rgiids_processed = list(output_df_all.RGIId)
            rgiids_all = list(main_glac_rgi_all.RGIId)
            rgiids_missing = [x for x in rgiids_all if x not in rgiids_processed]
            
            glac_no_missing = [x.split('-')[1] for x in rgiids_missing]
            main_glac_rgi_missing = modelsetup.selectglaciersrgitable(glac_no=glac_no_missing)
            
            print(reg, len(glac_no_missing), main_glac_rgi_missing.Area.sum(), glac_no_missing)
            
            if not os.path.exists(output_fp + output_fn_missing) or overwrite:
                
                # Add regions for median subsets
                output_df_all['O1Region'] = [int(x.split('-')[1].split('.')[0]) for x in output_df_all.RGIId]
                
                # Update mass balance data
                output_df_missing = pd.DataFrame(np.zeros((len(rgiids_missing),len(output_df_all.columns))), columns=output_df_all.columns)
                output_df_missing['RGIId'] = rgiids_missing
                output_df_missing['fa_gta_obs'] = np.nan
                rgi_area_dict = dict(zip(main_glac_rgi_missing.RGIId, main_glac_rgi_missing.Area))
                output_df_missing['area_km2'] = output_df_missing['RGIId'].map(rgi_area_dict)
                rgi_mbobs_dict = dict(zip(mb_data['RGIId'],mb_data['mb_mwea']))
                output_df_missing['mb_clim_mwea_obs'] = output_df_missing['RGIId'].map(rgi_mbobs_dict)
                output_df_missing['mb_clim_gta_obs'] = [mwea_to_gta(output_df_missing.loc[x,'mb_clim_mwea_obs'],
                                                                    output_df_missing.loc[x,'area_km2']*1e6) for x in output_df_missing.index]
                output_df_missing['mb_total_mwea_obs'] = output_df_missing['mb_clim_mwea_obs']
                output_df_missing['mb_total_gta_obs'] = output_df_missing['mb_clim_gta_obs']
                
                # Uncertainty with calving_k based on regional calibration
#                calving_k_nmad_missing = 1.4826 * median_abs_deviation(output_df_all.calving_k)
                calving_k_nmad_missing = np.median(output_df_all_good.calving_k_nmad)
                output_df_missing['calving_k_nmad'] = calving_k_nmad_missing
                
                # Check that climatic mass balance is reasonable
                mb_clim_reg_95 = (mb_clim_reg_avg + 1.96*mb_clim_reg_std)
                
                # Start with median value
                calving_k_med = np.median(output_df_all.loc[output_df_all['O1Region']==reg,'calving_k'])
                for nglac, rgiid in enumerate(rgiids_missing):
                    
#                    if rgiid in ['RGI60-19.00227']:
#                    if rgiid in ['RGI60-19.01721']:
#                    glacno = rgiid.split('-')[1]
#                    if glacno in ['19.01721', '19.00418', '19.00169', '19.00156', '19.00029', '19.00746', '19.00707', 
#                                 '19.00748', '19.00113', '19.00562', '19.00160', '19.00432', '19.00417', '19.00103']:
#                    if glacno in ['19.00707']:
                    for batman in [0]:
                        main_glac_rgi_ind = modelsetup.selectglaciersrgitable(glac_no=[rgiid.split('-')[1]])
                        area_km2 = main_glac_rgi_ind.loc[0,'Area']
                        # Estimate frontal ablation for missing glaciers
                        output_df, reg_calving_gta_mod, reg_calving_gta_obs = (
                                reg_calving_flux(main_glac_rgi_ind, calving_k_med, debug=True, calc_mb_geo_correction=True))
                        
                        # ASSUME THE TOTAL MASS BALANCE EQUALS THE GEODETIC MASS BALANCE CORRECTED FOR THE FA BELOW SEA LEVEL
                        mb_total_mwea = output_df_missing.loc[nglac,'mb_total_mwea_obs']
                        mb_fa_mwea = gta_to_mwea(output_df.loc[0,'calving_flux_Gta'], area_km2*1e6)
                        mb_clim_mwea = mb_total_mwea + mb_fa_mwea
                        
                        print('mb_total_mwea:', np.round(mb_total_mwea,2))
                        print('mb_clim_mwea:', np.round(mb_clim_mwea,2))
                        print('mb_fa_mwea:', np.round(mb_fa_mwea,2))
                        print('mb_clim (reg 95%):', np.round(mb_clim_reg_95,2))
#                        print('mb_total (95% min):', np.round(mb_clim_reg_3std_min,2))                        
                        
                        if mb_clim_mwea < mb_clim_reg_95:
                            for cn in ['calving_k', 'calving_thick', 'calving_flux_Gta', 'no_errors', 'oggm_dynamics']:
                                output_df_missing.loc[nglac,cn] = output_df.loc[0,cn]
                            output_df_missing.loc[nglac,'mb_clim_mwea'] = mb_clim_mwea
                            output_df_missing.loc[nglac,'mb_clim_gta'] = mwea_to_gta(output_df_missing.loc[nglac,'mb_clim_mwea'], area_km2*1e6)
                            output_df_missing.loc[nglac,'mb_total_mwea'] = mb_total_mwea 
                            output_df_missing.loc[nglac,'mb_total_gta'] = mwea_to_gta(output_df_missing.loc[nglac,'mb_total_gta'], area_km2*1e6)
                        else:

                            # Calibrate frontal ablation based on fa_mwea_max
                            #  i.e., the maximum frontal ablation that is consistent with reasonable mb_clim
                            fa_mwea_max = mb_fa_mwea - (mb_clim_mwea - mb_clim_reg_95)
                            
                            # If mb_clim_mwea is already greater than mb_clim_reg_95, then going to have this be positive
                            #  therefore, correct it to only let it be 10% of the positive mb_total such that it stays "reasonable"
                            if fa_mwea_max < 0:
                                print('\n  too positive, limiting fa_mwea_max to 10% mb_total_mwea')
                                fa_mwea_max = 0.1*mb_total_mwea
                            
                            # Reset bounds
                            calving_k = np.copy(calving_k_med)
                            calving_k_bndlow = np.copy(calving_k_bndlow_set)
                            calving_k_bndhigh = np.copy(calving_k_bndhigh_set)
                            calving_k_step = np.copy(calving_k_step_set)
                            
                            # Select individual glacier
                            rgiid_ind = main_glac_rgi_ind.loc[0,'RGIId']
                            fa_glac_data_ind = pd.DataFrame(np.zeros((1,len(fa_glac_data_reg.columns))), 
                                                            columns=fa_glac_data_reg.columns)
                            fa_glac_data_ind.loc[0,'RGIId'] = rgiid_ind
        
                            # Check bounds
                            bndlow_good = True
                            bndhigh_good = True
                            try:
                                output_df_bndhigh, reg_calving_gta_mod_bndhigh, reg_calving_gta_obs = (
                                        reg_calving_flux(main_glac_rgi_ind, calving_k_bndhigh, fa_glac_data_reg=fa_glac_data_ind, 
                                                         frontal_ablation_Gta_cn=frontal_ablation_Gta_cn, 
                                                         prms_from_reg_priors=prms_from_reg_priors, prms_from_glac_cal=prms_from_glac_cal,
                                                         ignore_nan=False, debug=debug_reg_calving_fxn, calc_mb_geo_correction=True))
                            except:
                                bndhigh_good = False
                                reg_calving_gta_mod_bndhigh = None
            
                            try:
                                output_df_bndlow, reg_calving_gta_mod_bndlow, reg_calving_gta_obs = (
                                        reg_calving_flux(main_glac_rgi_ind, calving_k_bndlow, fa_glac_data_reg=fa_glac_data_ind,
                                                         frontal_ablation_Gta_cn=frontal_ablation_Gta_cn, 
                                                         prms_from_reg_priors=prms_from_reg_priors, prms_from_glac_cal=prms_from_glac_cal,
                                                         ignore_nan=False, debug=debug_reg_calving_fxn, calc_mb_geo_correction=True))
                            except:
                                bndlow_good = False
                                reg_calving_gta_mod_bndlow = None
                                
                            # Record bounds
                            output_df_missing.loc[nglac,'calving_flux_Gta_bndlow'] = reg_calving_gta_mod_bndlow
                            output_df_missing.loc[nglac,'calving_flux_Gta_bndhigh'] = reg_calving_gta_mod_bndhigh
                            
                            if debug:
                                print('  fa_model_bndlow [mwea] :', np.round(gta_to_mwea(reg_calving_gta_mod_bndlow, area_km2*1e6),2))
                                print('  fa_model_bndhigh [mwea] :', np.round(gta_to_mwea(reg_calving_gta_mod_bndhigh,area_km2*1e6),2))
                                print('  fa_mwea_cal [mwea]:', np.round(fa_mwea_max,2))
                                 
                            if bndhigh_good and bndlow_good:

                                print('\n-------')
                                print('mb_clim_mwea:', np.round(mb_clim_mwea,2))

                                calving_k_step_missing = (calving_k_med - calving_k_bndlow) / 20
                                calving_k_next = calving_k - calving_k_step_missing
                                ncount = 0
                                while mb_fa_mwea > fa_mwea_max and calving_k_next > 0:
                                    calving_k -= calving_k_step_missing
                                    
                                    if ncount == 0:
                                        reset_gdir=True
                                    else:
                                        reset_gdir=False
                                    # Estimate frontal ablation for missing glaciers
                                    output_df, reg_calving_gta_mod, reg_calving_gta_obs = (
                                            reg_calving_flux(main_glac_rgi_ind, calving_k, debug=True, calc_mb_geo_correction=True,
                                                             reset_gdir=reset_gdir))
                                    
                                    mb_fa_mwea = gta_to_mwea(output_df.loc[0,'calving_flux_Gta'],  area_km2*1e6)
                                    
                                    calving_k_next = calving_k - calving_k_step_missing
                                    
                                    print(calving_k, 'mb_fa_mwea:', np.round(mb_fa_mwea,2), 'mb_fa_mwea_max:', np.round(fa_mwea_max,2))
                                    
                                # Record output
                                for cn in ['calving_k', 'calving_thick', 'calving_flux_Gta', 'no_errors', 'oggm_dynamics']:
                                    output_df_missing.loc[nglac,cn] = output_df.loc[0,cn]
                                
                                mb_clim_mwea = mb_total_mwea + mb_fa_mwea
                                
                                print('mb_total_mwea:', np.round(mb_total_mwea,2))
                                print('mb_clim_mwea:', np.round(mb_clim_mwea,2))
                                print('mb_fa_mwea:', np.round(mb_fa_mwea,2))
                                print('mb_clim (reg 95%):', np.round(mb_clim_reg_95,2))

                                for cn in ['calving_k', 'calving_thick', 'calving_flux_Gta', 'no_errors', 'oggm_dynamics']:
                                    output_df_missing.loc[nglac,cn] = output_df.loc[0,cn]
                                output_df_missing.loc[nglac,'mb_clim_mwea'] = mb_clim_mwea
                                output_df_missing.loc[nglac,'mb_clim_gta'] = mwea_to_gta(output_df_missing.loc[nglac,'mb_clim_mwea'], area_km2*1e6)
                                output_df_missing.loc[nglac,'mb_total_mwea'] = mb_total_mwea 
                                output_df_missing.loc[nglac,'mb_total_gta'] = mwea_to_gta(output_df_missing.loc[nglac,'mb_total_mwea'], area_km2*1e6)

                        # Adjust calving_k_nmad if calving_k is very low to avoid poor values
                        if output_df_missing.loc[nglac,'calving_k'] < calving_k_nmad_missing:
                            output_df_missing.loc[nglac,'calving_k_nmad'] = output_df_missing.loc[nglac,'calving_k'] - calving_k_bndlow_set  
                                
#                        # Check uncertainty based on NMAD
#                        calving_k_plusnmad = calving_k_med + calving_k_nmad
#                        output_df, reg_calving_gta_mod, reg_calving_gta_obs = (
#                                reg_calving_flux(main_glac_rgi_ind, calving_k_plusnmad, debug=True, calc_mb_geo_correction=True))
#                        mb_fa_mwea = gta_to_mwea(output_df.loc[0,'calving_flux_Gta'],  area_km2*1e6)
#                        print('mb_fa_mwea (calving_k + nmad):', np.round(mb_fa_mwea,2))
#                        
#                        calving_k_minusnmad = calving_k_med - calving_k_nmad
#                        output_df, reg_calving_gta_mod, reg_calving_gta_obs = (
#                                reg_calving_flux(main_glac_rgi_ind, calving_k_minusnmad, debug=True, calc_mb_geo_correction=True))
#                        mb_fa_mwea = gta_to_mwea(output_df.loc[0,'calving_flux_Gta'],  area_km2*1e6)
#                        print('mb_fa_mwea (calving_k - nmad):', np.round(mb_fa_mwea,2))

#                        calving_k_values = [0.001, 0.01, 0.1, 0.15, 0.18, 0.2, 0.25, 0.3, 0.35]
#                        mb_fa_mwea_list = []
#                        for calving_k in calving_k_values:
#                            output_df, reg_calving_gta_mod, reg_calving_gta_obs = (
#                                reg_calving_flux(main_glac_rgi_ind, calving_k, debug=True, calc_mb_geo_correction=True))
#                            mb_fa_mwea = gta_to_mwea(output_df.loc[0,'calving_flux_Gta'],  area_km2*1e6)
#                            mb_fa_mwea_list.append(mb_fa_mwea)
#                            print(calving_k, 'mb_fa_mwea (calving_k - nmad):', np.round(mb_fa_mwea,2))
#                        # Set up your plot (and/or subplots)
#                        fig, ax = plt.subplots(1, 1, squeeze=False, sharex=False, sharey=False, gridspec_kw = {'wspace':0.4, 'hspace':0.15})
#                        ax[0,0].scatter(calving_k_values, mb_fa_mwea_list, color='k', linewidth=1, zorder=2, label='plot1')
#                        ax[0,0].set_xlabel('calving_k', size=12)     
#                        ax[0,0].set_ylabel('mb_fa_mwea', size=12)
#                        plt.show()

                # Export 
                output_df_missing.to_csv(output_fp + output_fn_missing, index=False)
            else:
                output_df_missing = pd.read_csv(output_fp + output_fn_missing)

        #%% ----- CORRECTION FOR TOTAL MASS BALANCE IN ANTARCTICA/SUBANTARCTIC  (OLD)-----
#        #  only apply in this region because poorer data quality (i.e., few observations)
#        if reg in [19]:
#            
##            mb_mwea_fa_good_all = gta_to_mwea(output_df_all_good.calving_flux_Gta.values, 
##                                              output_df_all_good.area_km2.values * 1e6)
##            mb_mwea_fa_good_3std = mb_mwea_fa_good_all.mean() + 3*mb_mwea_fa_good_all.std()
#            
#            # Total mass balance is 95% lower bound for climatic mass balance minus specific frontal ablation
#            #  from calibrated glaciers
##            mb_total_reg_min = mb_clim_reg_3std_min - mb_mwea_fa_good_3std
#            mb_total_reg_min = mb_clim_reg_3std_min
#            
#            print('mb total 95% min:', np.round(mb_total_reg_min,2))
#            
#            for nglac, rgiid in enumerate(rgiids_missing):
#                
#                if output_df_missing.loc[nglac,'mb_total_mwea'] < mb_total_reg_min:
#                    
##                    if rgiid in ['RGI60-19.01826']:
##                    if rgiid in ['RGI60-19.00232']:
#                    for batman in [0]:
#                        
#                        calving_k = output_df_missing.loc[nglac,'calving_k']
#                        calving_k_step_missing = (calving_k - calving_k_bndlow) / 20
#                        if calving_k_step_missing < 0.001:
#                            calving_k_step_missing = 0.001
#                        calving_k_next = calving_k - calving_k_step_missing
#                        
#                        # Estimate frontal ablation for missing glaciers
#                        main_glac_rgi_ind = modelsetup.selectglaciersrgitable(glac_no=[rgiid.split('-')[1]])
#                        output_df, reg_calving_gta_mod, reg_calving_gta_obs = (
#                                reg_calving_flux(main_glac_rgi_ind, calving_k, debug=True, calc_mb_geo_correction=True))
#                        
#                        # Adjust climatic mass balance to account for the losses due to frontal ablation
#                        #  add this loss because it'll come from frontal ablation instead of climatic mass balance
#                        mb_clim_fa_corrected = (output_df_missing.loc[nglac,'mb_clim_mwea_obs'] + 
#                                                output_df.loc[0,'mb_mwea_fa_asl_lost'])
#                        
#                        # Account for frontal ablation
#                        mb_mwea_fa = gta_to_mwea(output_df.loc[0,'calving_flux_Gta'], 
#                                                 main_glac_rgi_ind.loc[0,'Area'] * 1e6)
#                        
#                        mb_total_corrected = mb_clim_fa_corrected - mb_mwea_fa
#                        
#                        print('mb total corrected:', np.round(mb_total_corrected,2))
#                        print('mb mwea frontal ablation:', np.round(mb_mwea_fa,2))
#                        
#                        while mb_total_corrected < mb_total_reg_min and calving_k_next > calving_k_bndlow:
#                            
#                            calving_k -= calving_k_step_missing
#                            
#                            # Estimate frontal ablation for missing glaciers
#                            output_df, reg_calving_gta_mod, reg_calving_gta_obs = (
#                                    reg_calving_flux(main_glac_rgi_ind, calving_k, debug=True, calc_mb_geo_correction=True))
#                            
#                            # Adjust climatic mass balance to account for the losses due to frontal ablation
#                            #  add this loss because it'll come from frontal ablation instead of climatic mass balance
#                            mb_clim_fa_corrected = (output_df_missing.loc[nglac,'mb_clim_mwea_obs'] + 
#                                                    output_df.loc[0,'mb_mwea_fa_asl_lost'])
#                            
#                            # Account for frontal ablation
#                            mb_mwea_fa = gta_to_mwea(output_df.loc[0,'calving_flux_Gta'], 
#                                                     main_glac_rgi_ind.loc[0,'Area'] * 1e6)
#                            
#                            mb_total_corrected = mb_clim_fa_corrected - mb_mwea_fa
#                             
#                            calving_k_next = calving_k - calving_k_step_missing
#                            
#                            
#                            
#                        print('mb total corrected:', np.round(mb_total_corrected,2))
#                        print('mb mwea frontal ablation:', np.round(mb_mwea_fa,2))
#
#
#                        # Record output
#                        for cn in ['calving_k', 'calving_thick', 'calving_flux_Gta', 'no_errors', 'oggm_dynamics']:
#                            output_df_missing.loc[nglac,cn] = output_df.loc[0,cn]
#                        output_df_missing.loc[nglac,'mb_clim_mwea'] = mb_clim_fa_corrected
#                        output_df_missing.loc[nglac,'mb_clim_gta'] = mwea_to_gta(output_df_missing.loc[nglac,'mb_clim_mwea'], 
#                                                                                 output_df_missing.loc[nglac,'area_km2']*1e6)
#                        output_df_missing.loc[nglac,'mb_total_mwea'] = mb_total_corrected
#                        output_df_missing.loc[nglac,'mb_total_gta'] = mwea_to_gta(output_df_missing.loc[nglac,'mb_total_mwea'], 
#                                                                                   output_df_missing.loc[nglac,'area_km2']*1e6)
#                        
#                     
#            # Fill nan
#            nan_idx = np.where(np.isnan(output_df_missing['mb_total_gta']))[0]
#            if len(nan_idx) > 0:
#                output_df_missing.loc[nan_idx,'calving_k'] = 0
#                output_df_missing.loc[nan_idx,'mb_total_gta'] = output_df_missing.loc[nan_idx,'mb_clim_gta']
#                output_df_missing.loc[nan_idx,'mb_total_mwea'] = output_df_missing.loc[nan_idx,'mb_clim_mwea']
#     
#            # Export csv
#            output_df_missing.to_csv(output_fp + output_fn_missing.replace('.csv','_wmbtotal_correction.csv'), index=False)
#            
#        print('total calving flux (Gta):', output_df_missing.calving_flux_Gta.sum())
#        
##        #%%
##        output_fn = 'all-calving_cal_ind.csv'
##        output_df_missing = pd.read_csv(output_fp + output_fn)
##        output_df_missing_old = pd.read_csv(output_fp + output_fn.replace('.csv','-old.csv'))
##        
##        rgiids_new_calving_k = []
##        for nglac, rgiid in enumerate(output_df_missing.RGIId):
##            rgiids_old = list(output_df_missing_old.RGIId) 
##            if rgiid in rgiids_old:
##                rgiid_idx = rgiids_old.index(rgiid)
##                
##                if not output_df_missing.loc[nglac,'calving_k'] == output_df_missing_old.loc[rgiid_idx,'calving_k']:
##                    print(rgiid, output_df_missing.loc[nglac,'RGIId'], output_df_missing_old.loc[rgiid_idx,'calving_k'],
##                          output_df_missing.loc[nglac,'calving_k'])
##                    rgiids_new_calving_k.append(rgiid)
##        glacno_redo = [x.split('-')[1] for x in rgiids_new_calving_k]
##        print(glacno_redo)
                
        #%%
        # ----- PLOT RESULTS FOR EACH GLACIER -----
        plot_max_raw = np.max([output_df_all.calving_flux_Gta.max(), output_df_all.fa_gta_obs.max()])
        plot_max = 10**np.ceil(np.log10(plot_max_raw))

        plot_min_raw = np.max([output_df_all.calving_flux_Gta.min(), output_df_all.fa_gta_obs.min()])
        plot_min = 10**np.floor(np.log10(plot_min_raw))
        if plot_min < 1e-3:
            plot_min = 1e-4

        x_min, x_max = plot_min, plot_max
        
        fig, ax = plt.subplots(2, 2, squeeze=False, gridspec_kw = {'wspace':0.3, 'hspace':0.3})
        
        # ----- Scatter plot -----
        # Marker size
        glac_area_all = output_df_all['area_km2'].values
        s_sizes = [10,50,250,1000]
        s_byarea = np.zeros(glac_area_all.shape) + s_sizes[3]
        s_byarea[(glac_area_all < 10)] = s_sizes[0]
        s_byarea[(glac_area_all >= 10) & (glac_area_all < 100)] = s_sizes[1]
        s_byarea[(glac_area_all >= 100) & (glac_area_all < 1000)] = s_sizes[2]
        
        sc = ax[0,0].scatter(output_df_all['fa_gta_obs'], output_df_all['calving_flux_Gta'], 
                             color='k', marker='o', linewidth=1, facecolor='none', 
                             s=s_byarea, clip_on=True)
        # Labels
        ax[0,0].set_xlabel('Observed $A_{f}$ (Gt/yr)', size=12)    
        ax[0,0].set_ylabel('Modeled $A_{f}$ (Gt/yr)', size=12)
        ax[0,0].set_xlim(x_min,x_max)
        ax[0,0].set_ylim(x_min,x_max)
        ax[0,0].plot([x_min, x_max], [x_min, x_max], color='k', linewidth=0.5, zorder=1)
        # Log scale
        ax[0,0].set_xscale('log')
        ax[0,0].set_yscale('log')
        
        # Legend
        obs_labels = ['< 10', '10-10$^{2}$', '10$^{2}$-10$^{3}$', '> 10$^{3}$']
        for nlabel, obs_label in enumerate(obs_labels):
            ax[0,0].scatter([-10],[-10], color='grey', marker='o', linewidth=1, 
                            facecolor='none', s=s_sizes[nlabel], zorder=3, label=obs_label)
        ax[0,0].text(0.06, 0.98, 'Area (km$^{2}$)', size=12, horizontalalignment='left', verticalalignment='top', 
                     transform=ax[0,0].transAxes, color='grey')
        leg = ax[0,0].legend(loc='upper left', ncol=1, fontsize=10, frameon=False,
                             handletextpad=1, borderpad=0.25, labelspacing=0.4, bbox_to_anchor=(0.0, 0.93),
                             labelcolor='grey')
#        ax[0,0].text(1.08, 0.97, 'Area (km$^{2}$)', size=12, horizontalalignment='left', verticalalignment='top', 
#                     transform=ax[0,0].transAxes)
#        leg = ax[0,0].legend(loc='upper left', ncol=1, fontsize=10, frameon=False,
#                             handletextpad=1, borderpad=0.25, labelspacing=1, bbox_to_anchor=(1.035, 0.9))
        
        # ----- Histogram -----
#        nbins = 25
#        ax[0,1].hist(output_df_all['calving_k'], bins=nbins, color='grey', edgecolor='k')
        vn_bins = np.arange(0, np.max([1,output_df_all.calving_k.max()]) + 0.1, 0.1)
        hist, bins = np.histogram(output_df_all.loc[output_df_all['no_errors'] == 1, 'calving_k'], bins=vn_bins)
        ax[0,1].bar(x=vn_bins[:-1] + 0.1/2, height=hist, width=(bins[1]-bins[0]), 
                             align='center', edgecolor='black', color='grey')
        ax[0,1].set_xticks(np.arange(0,np.max([1,vn_bins.max()])+0.1, 1))
        ax[0,1].set_xticks(vn_bins, minor=True)
        ax[0,1].set_xlim(vn_bins.min(), np.max([1,vn_bins.max()]))
        if hist.max() < 40:
            y_major_interval = 5
            y_max = np.ceil(hist.max()/y_major_interval)*y_major_interval
            ax[0,1].set_yticks(np.arange(0,y_max+y_major_interval,y_major_interval))
        elif hist.max() > 40:
            y_major_interval = 10
            y_max = np.ceil(hist.max()/y_major_interval)*y_major_interval
            ax[0,1].set_yticks(np.arange(0,y_max+y_major_interval,y_major_interval))
        
        # Labels
        ax[0,1].set_xlabel('$k_{f}$ (yr$^{-1}$)', size=12)
        ax[0,1].set_ylabel('Count (glaciers)', size=12)
        
        
        # ----- CALVING_K VS MB_CLIM -----
        ax[1,0].scatter(output_df_all['calving_k'], output_df_all['mb_clim_mwea'], 
                        color='k', marker='o', linewidth=1, facecolor='none', 
                        s=s_byarea, clip_on=True)
        ax[1,0].set_xlabel('$k_{f}$ (yr$^{-1}$)', size=12)
        ax[1,0].set_ylabel('$B_{clim}$ (mwea)', size=12)
        
        # ----- CALVING_K VS AREA -----
        ax[1,1].scatter(output_df_all['area_km2'], output_df_all['calving_k'], 
                        color='k', marker='o', linewidth=1, facecolor='none', 
                        s=s_byarea, clip_on=True)
        ax[1,1].set_xlabel('Area (km2)', size=12)
        ax[1,1].set_ylabel('$k_{f}$ (yr$^{-1}$)', size=12)
        
        
        # Save figure
        fig.set_size_inches(6,6)
        fig_fullfn = output_fp + str(reg) + '-calving_glac_compare-cal_ind.png'
        fig.savefig(fig_fullfn, bbox_inches='tight', dpi=300)
        


#%% ----- MERGE CALIBRATED CALVING DATASETS -----
if option_merge_calving_k:
    
    for nreg, reg in enumerate(regions):
        
        # Load quality controlled frontal ablation data 
        output_fn_reg = str(reg) + '-calving_cal_ind.csv'
        
        assert os.path.exists(output_fp + output_fn_reg), 'Calibrated frontal ablation output dataset does not exist'
        
        output_df_reg = pd.read_csv(output_fp + output_fn_reg)
        
        if not 'calving_k_nmad' in list(output_df_reg.columns):
            output_df_reg['calving_k_nmad'] = 0
        
        if nreg == 0:
            output_df_all = output_df_reg
        else:
            output_df_all = pd.concat([output_df_all, output_df_reg], axis=0)
            
        output_fn_reg_missing = output_fn_reg.replace('.csv','-missing.csv')
        if os.path.exists(output_fp + output_fn_reg_missing):
            
            # Check if second correction exists
            output_fn_reg_missing_v2 = output_fn_reg_missing.replace('.csv','_wmbtotal_correction.csv')
            if os.path.exists(output_fp + output_fn_reg_missing_v2):            
                output_df_reg_missing = pd.read_csv(output_fp + output_fn_reg_missing_v2)
            else:
                output_df_reg_missing = pd.read_csv(output_fp + output_fn_reg_missing)
                
            if not 'calving_k_nmad' in list(output_df_reg_missing.columns):
                output_df_reg_missing['calving_k_nmad'] = 0
            
            output_df_all = pd.concat([output_df_all, output_df_reg_missing], axis=0)
        
    output_fn_all = 'all-calving_cal_ind.csv'
    output_df_all.to_csv(output_fp + output_fn_all, index=0)


#%% ----- UPDATE MASS BALANCE DATA WITH FRONTAL ABLATION ESTIMATES -----
if option_update_mb_data:
    
    # Load calving glacier data (already quality controlled during calibration)
    calving_fp = pygem_prms.main_directory + '/../calving_data/analysis/'
    calving_fn = 'all-calving_cal_ind.csv'
    assert os.path.exists(calving_fp + calving_fn), 'Calibrated frontal ablation output dataset does not exist'
    fa_glac_data = pd.read_csv(calving_fp + calving_fn)

    # Load mass balance data
    hugonnet_fn = 'df_pergla_global_20yr-filled.csv'
    mb_data = pd.read_csv(pygem_prms.hugonnet_fp + hugonnet_fn)
    mb_rgiids = list(mb_data.RGIId)

    # Record prior data
    mb_data['mb_romain_mwea'] = mb_data['mb_mwea']
    mb_data['mb_romain_mwea_err'] = mb_data['mb_mwea_err']
    mb_data['mb_clim_mwea'] = mb_data['mb_mwea']
    mb_data['mb_clim_mwea_err'] = mb_data['mb_mwea_err']

    # Update mass balance data
    for nglac, rgiid in enumerate(fa_glac_data.RGIId):
        
        O1region = int(rgiid.split('-')[1].split('.')[0])
        if O1region in regions:        

            # Update the mass balance data in Romain's file
            mb_idx = mb_rgiids.index(rgiid)
            mb_data.loc[mb_idx,'mb_mwea'] = fa_glac_data.loc[nglac,'mb_total_mwea']
            mb_data.loc[mb_idx,'mb_clim_mwea'] = fa_glac_data.loc[nglac,'mb_clim_mwea']
            
            print(rgiid, 'mb_mwea:', np.round(mb_data.loc[mb_idx,'mb_mwea'],2), 
                  'mb_clim:', np.round(mb_data.loc[mb_idx,'mb_clim_mwea'],2), 
                  'mb_romain:', np.round(mb_data.loc[mb_idx,'mb_romain_mwea'],2))

    # Export the updated dataset
    mb_data.to_csv(pygem_prms.hugonnet_fp + hugonnet_fn.replace('.csv','-facorrected.csv'), index=False)

#%%
    # Update gdirs
#    rgiids = ['RGI60-' + x for x in pygem_prms.glac_no]
#    for nglac, rgiid in enumerate(rgiids):
    for nglac, rgiid in enumerate(fa_glac_data.RGIId):
        
        O1region = int(rgiid.split('-')[1].split('.')[0])
        if O1region in regions:    

            print(rgiid)
        
            # Select subsets of data
            glacier_str = rgiid.split('-')[1]
    
            gdir = single_flowline_glacier_directory_with_calving(glacier_str, 
                                                                  logging_level='CRITICAL',
                                                                  reset=True
                                                                  )


#%%
if option_plot_calving_k:
    
    rgi_reg_dict = {'all':'Global',
                    'global':'Global',
                    1:'Alaska',
                    2:'W Canada/USA',
                    3:'Arctic Canada North',
                    4:'Arctic Canada South',
                    5:'Greenland',
                    6:'Iceland',
                    7:'Svalbard',
                    8:'Scandinavia',
                    9:'Russian Arctic',
                    10:'North Asia',
                    11:'Central Europe',
                    12:'Caucasus/Middle East',
                    13:'Central Asia',
                    14:'South Asia West',
                    15:'South Asia East',
                    16:'Low Latitudes',
                    17:'Southern Andes',
                    18:'New Zealand',
                    19:'Antarctica/Subantarctic'
                    }
    
    fig_fp = output_fp + '../figs/'

    fig = plt.figure()
    gs = fig.add_gridspec(nrows=3,ncols=3,wspace=0.4,hspace=0.4)
    ax1 = fig.add_subplot(gs[0,0])
    ax2 = fig.add_subplot(gs[0,1])
    ax3 = fig.add_subplot(gs[0,2])
    ax4 = fig.add_subplot(gs[1,0])
    ax5 = fig.add_subplot(gs[1,1])
    ax6 = fig.add_subplot(gs[1,2])
    ax7 = fig.add_subplot(gs[2,0])
    ax8 = fig.add_subplot(gs[2,1])
    ax9 = fig.add_subplot(gs[2,2])
    
    regions_ordered = [1,3,4,5,7,9,17,19]
    for nax, ax in enumerate([ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8, ax9]):
        
        if ax not in [ax9]:
            reg = regions_ordered[nax]  
            
            calving_k_fn = str(reg) + '-calving_cal_ind.csv'
            calving_k_fn_missing = calving_k_fn.replace('.csv','-missing.csv')
        
            output_df_all_good = pd.read_csv(output_fp + calving_k_fn)
    #        calving_k_df_missing = pd.read_csv(output_fp + calving_k_fn_missing)
            
            
            # ----- PLOT RESULTS FOR EACH GLACIER -----
    #        plot_max_raw = np.max([output_df_all_good.calving_flux_Gta.max(), output_df_all_good.fa_gta_obs.max()])
    #        plot_max = 10**np.ceil(np.log10(plot_max_raw))
    #
    #        plot_min_raw = np.max([output_df_all_good.calving_flux_Gta.min(), output_df_all_good.fa_gta_obs.min()])
    #        plot_min = 10**np.floor(np.log10(plot_min_raw))
    #        if plot_min < 1e-3:
            plot_min = 1e-4
            plot_max = 10
    
            x_min, x_max = plot_min, plot_max
            
            # ----- Scatter plot -----
            # Marker size
            glac_area_all = output_df_all_good['area_km2'].values
            s_sizes = [10,40,120,240]
            s_byarea = np.zeros(glac_area_all.shape) + s_sizes[3]
            s_byarea[(glac_area_all < 10)] = s_sizes[0]
            s_byarea[(glac_area_all >= 10) & (glac_area_all < 100)] = s_sizes[1]
            s_byarea[(glac_area_all >= 100) & (glac_area_all < 1000)] = s_sizes[2]
            
            sc = ax.scatter(output_df_all_good['fa_gta_obs'], output_df_all_good['calving_flux_Gta'], 
                                 color='k', marker='o', linewidth=0.5, facecolor='none', 
                                 s=s_byarea, clip_on=True)
            
            ax.plot([x_min, x_max], [x_min, x_max], color='k', linewidth=0.5, zorder=1)
            
            ax.text(0.98, 1.02, rgi_reg_dict[reg], size=10, horizontalalignment='right', 
                 verticalalignment='bottom', transform=ax.transAxes)
    
        # Labels
        ax.set_xlim(x_min,x_max)
        ax.set_ylim(x_min,x_max)
        # Log scale
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        ax.tick_params(axis='both', which='major', direction='inout', right=True)
        ax.tick_params(axis='both', which='minor', direction='in', right=True)

            
    
    #        # ----- Histogram -----
    ##        nbins = 25
    ##        ax[0,1].hist(output_df_all_good['calving_k'], bins=nbins, color='grey', edgecolor='k')
    #        vn_bins = np.arange(0, np.max([1,output_df_all_good.calving_k.max()]) + 0.1, 0.1)
    #        hist, bins = np.histogram(output_df_all_good.loc[output_df_all_good['no_errors'] == 1, 'calving_k'], bins=vn_bins)
    #        ax[0,1].bar(x=vn_bins[:-1] + 0.1/2, height=hist, width=(bins[1]-bins[0]), 
    #                             align='center', edgecolor='black', color='grey')
    #        ax[0,1].set_xticks(np.arange(0,np.max([1,vn_bins.max()])+0.1, 1))
    #        ax[0,1].set_xticks(vn_bins, minor=True)
    #        ax[0,1].set_xlim(vn_bins.min(), np.max([1,vn_bins.max()]))
    #        if hist.max() < 40:
    #            y_major_interval = 5
    #            y_max = np.ceil(hist.max()/y_major_interval)*y_major_interval
    #            ax[0,1].set_yticks(np.arange(0,y_max+y_major_interval,y_major_interval))
    #        elif hist.max() > 40:
    #            y_major_interval = 10
    #            y_max = np.ceil(hist.max()/y_major_interval)*y_major_interval
    #            ax[0,1].set_yticks(np.arange(0,y_max+y_major_interval,y_major_interval))
    #        
    #        # Labels
    #        ax[0,1].set_xlabel('$k_{f}$ (yr$^{-1}$)', size=12)
    #        ax[0,1].set_ylabel('Count (glaciers)', size=12)
            
            
            
            
            
            
        
                   
            # Plot
    #        ax.plot(years, reg_vol_med_norm, color=temp_colordict[deg_group], linestyle='-', 
    #                linewidth=1, zorder=4, label=deg_group)
    #        ax.plot(years, reg_vol_med_norm_nocalving, color=temp_colordict[deg_group], linestyle=':', 
    #                linewidth=1, zorder=3, label=None)
    #        
    #        if ax in [ax1, ax4, ax7]:
    #            ax.set_ylabel('Mass (rel. to 2015)')
    #        ax.set_xlim(startyear, endyear)
    #        ax.xaxis.set_major_locator(MultipleLocator(40))
    #        ax.xaxis.set_minor_locator(MultipleLocator(10))
    #        ax.set_ylim(0,1.1)
    #        ax.yaxis.set_major_locator(MultipleLocator(0.2))
    #        ax.yaxis.set_minor_locator(MultipleLocator(0.1))
            
        
        # Legend
        if ax in [ax9]:
            obs_labels = ['< 10', '10-10$^{2}$', '10$^{2}$-10$^{3}$', '> 10$^{3}$']
            for nlabel, obs_label in enumerate(obs_labels):
                ax.scatter([-10],[-10], color='grey', marker='o', linewidth=1, 
                                facecolor='none', s=s_sizes[nlabel], zorder=3, label=obs_label)
            ax.text(0.1, 1.06, 'Area (km$^{2}$)', size=12, horizontalalignment='left', verticalalignment='top', 
                         transform=ax.transAxes, color='grey')
            leg = ax.legend(loc='upper left', ncol=1, fontsize=10, frameon=False,
                                 handletextpad=1, borderpad=0.25, labelspacing=0.4, bbox_to_anchor=(0.0, 0.93),
                                 labelcolor='grey')
        
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
            
    # Labels
    fig.text(0.5,0.04,'Observed frontal ablation (Gt yr$^{-1}$)', fontsize=12, horizontalalignment='center', verticalalignment='bottom')
    fig.text(0.04,0.5,'Modeled frontal ablation (Gt yr$^{-1}$)', size=12, horizontalalignment='center', verticalalignment='center', rotation=90)
        
        
    # Save figure
    fig_fn = ('allregions_calving_ObsMod.png')
    fig.set_size_inches(6.5,5.5)
    fig.savefig(output_fp + fig_fn, bbox_inches='tight', dpi=300)


#%%     
if option_scrap:
    # GCMs and RCP scenarios
    regions = [19]
#    gcm_names = ['BCC-CSM2-MR']
#    rcps = ['ssp126']
    
    gcm_names = ['BCC-CSM2-MR', 'CESM2', 'CESM2-WACCM', 'EC-Earth3', 'EC-Earth3-Veg', 'FGOALS-f3-L', 
                      'GFDL-ESM4', 'INM-CM4-8', 'INM-CM5-0']
    rcps = ['ssp126', 'ssp245', 'ssp370', 'ssp585']
    
    netcdf_fp_cmip5 = '/Users/drounce/Documents/HiMAT/spc_backup/simulations_calving_v4/'
    
    for reg in regions:
        # Load glaciers
        glacno_list = []
        for rcp in rcps:
            for gcm_name in gcm_names:
                
                # Filepath where glaciers are stored
                netcdf_fp = netcdf_fp_cmip5 + str(reg).zfill(2) + '/' + gcm_name + '/' + rcp + '/stats/'
                
                # Load the glaciers
                glacno_list_gcmrcp = []
                for i in os.listdir(netcdf_fp):
                    if i.endswith('.nc'):
                        glacno_list_gcmrcp.append(i.split('_')[0])
                glacno_list_gcmrcp = sorted(glacno_list_gcmrcp)
                
                print(gcm_name, rcp, 'simulated', len(glacno_list_gcmrcp), 'glaciers')
                
                # Only include the glaciers that were simulated by all GCM/RCP combinations
                if len(glacno_list) == 0:
                    glacno_list = glacno_list_gcmrcp
                else:
                    glacno_list = list(set(glacno_list).intersection(glacno_list_gcmrcp))
                glacno_list = sorted(glacno_list)
        
        # All glaciers for fraction
        main_glac_rgi_all = modelsetup.selectglaciersrgitable(rgi_regionsO1=[reg], 
                                                              rgi_regionsO2='all', rgi_glac_number='all', 
                                                              glac_no=None)
        # Glaciers with successful runs to process
        main_glac_rgi = modelsetup.selectglaciersrgitable(glac_no=glacno_list)
        
        #%%
        df_calving = pd.read_csv('/Users/drounce/Documents/HiMAT/calving_data/analysis/all-calving_cal_ind.csv')
        
        for gcm_name in gcm_names:
            for rcp in rcps:
            
                
                # Tidewater glaciers
                # Filepath where glaciers are stored
                netcdf_fp_binned = netcdf_fp_cmip5 + str(reg).zfill(2) + '/' + gcm_name + '/' + rcp + '/binned/'
                netcdf_fp_stats = netcdf_fp_cmip5 + str(reg).zfill(2) + '/' + gcm_name + '/' + rcp + '/stats/'
                
                # ----- Process glacier volume and losses -----
                print(gcm_name, rcp, 'simulated', len(glacno_list_gcmrcp), 'glaciers')
                
                fa_gta_total = 0
                fa_gta_total_cal = 0
                fa_gta_total_2080_2100 = 0
                for glacno in glacno_list:

                    netcdf_fn_stats_ending = 'MCMC_ba1_50sets_2000_2100_all.nc'
                    netcdf_fn_stats = '_'.join([glacno, gcm_name, rcp, netcdf_fn_stats_ending])
                                        
                    # Open files
                    ds_stats = xr.open_dataset(netcdf_fp_stats + '/' + netcdf_fn_stats)
                    
                    fa_monthly_2000_2020 = ds_stats.glac_frontalablation_monthly.values[0,0:12*20]
                    fa_monthly_2080_2100 = ds_stats.glac_frontalablation_monthly.values[0,-12*20:]
                    fa_gta_2000_2020 = fa_monthly_2000_2020.sum() / 1e9 / 20
                    fa_gta_2080_2100 = fa_monthly_2080_2100.sum() / 1e9 / 20
                    fa_gta_total += fa_gta_2000_2020
                    fa_gta_total_2080_2100 += fa_gta_2080_2100
                    
                    # Calibration
                    rgiid = 'RGI60-' + glacno
                    df_idx = list(df_calving.RGIId).index(rgiid)
                    fa_gta_2000_2020_cal = df_calving.loc[df_idx,'calving_flux_Gta']
                    fa_gta_total_cal += fa_gta_2000_2020_cal

#                    print(glacno, 'fa_gta 2000-2020:', np.round(fa_gta_2000_2020,2), 'cal:', np.round(fa_gta_2000_2020_cal,2))
                    
                print('\n', gcm_name, rcp, 'fa total gta:', np.round(fa_gta_total,1), 'cal:', np.round(fa_gta_total_cal,1), 
                      '2080-2100:', np.round(fa_gta_total_2080_2100,1))
                    
#        assert 1==0, 'why more? due to calving_k variations?'