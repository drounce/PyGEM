"""
Calibrate frontal ablation parameters for tidewater glaciers

@author: davidrounce
"""
# Built-in libraries
import argparse
import os
import pickle
# External libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#from scipy.stats import median_abs_deviation
# Local libraries
import class_climate
import pygem.pygem_input as pygem_prms
import pygem.pygem_modelsetup as modelsetup
from pygem.massbalance import PyGEMMassBalance
from pygem.oggm_compat import single_flowline_glacier_directory, single_flowline_glacier_directory_with_calving
from pygem.shop import debris 

from oggm import cfg
from oggm import graphics
from oggm import tasks
from oggm.core.flowline import FluxBasedModel
from oggm.core.inversion import find_inversion_calving_from_any_mb


#%% ----- MANUAL INPUT DATA -----
regions = [3]
# For region 9 decide if using individual glacier data or regional data
drop_ind_glaciers = False

# Frontal ablation calibration parameter (yr-1)
calving_k_init = 0.1
calving_k_bndlow = 0.01
calving_k_bndhigh = 5
calving_k_step = 0.01
nround_max = 5

debug=True
prms_from_reg_priors=True
prms_from_glac_cal=False

option_reg_calving_k = True    # Calibrate all glaciers regionally
option_ind_calving_k = False     # Calibrate individual glaciers



#%% ----- CALIBRATE FRONTAL ABLATION -----
def reg_calving_flux(main_glac_rgi, calving_k, fa_glac_data_reg, 
                     prms_from_reg_priors=False, prms_from_glac_cal=False, ignore_nan=True, debug=False):
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
    gcm_lr, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(gcm.lr_fn, gcm.lr_vn, main_glac_rgi, dates_table)

    # ===== CALIBRATE ALL THE GLACIERS AT ONCE =====
    output_cns = ['RGIId', 'calving_k', 'calving_thick', 'calving_flux_Gta_inv', 'calving_flux_Gta', 'no_errors']
    output_df = pd.DataFrame(np.zeros((main_glac_rgi.shape[0],len(output_cns))), columns=output_cns)
    output_df['RGIId'] = main_glac_rgi.RGIId
    output_df['calving_k'] = calving_k
    output_df['calving_thick'] = np.nan
    output_df['calving_flux_Gta_inv'] = np.nan
    output_df['calving_flux_Gta'] = np.nan
    for nglac in np.arange(main_glac_rgi.shape[0]):
        
        print('\ncalving_k:', calving_k)
        print(' ',main_glac_rgi.loc[main_glac_rgi.index.values[nglac],'RGIId'])
        
        # Select subsets of data
        glacier_rgi_table = main_glac_rgi.loc[main_glac_rgi.index.values[nglac], :]
        glacier_str = '{0:0.5f}'.format(glacier_rgi_table['RGIId_float'])
        
        gdir = single_flowline_glacier_directory_with_calving(glacier_str, 
                                                              logging_level='CRITICAL'
#                                                              logging_level='WORKFLOW'
#                                                              reset=True
                                                              )

        try:
            fls = gdir.read_pickle('inversion_flowlines')
            glacier_area = fls[0].widths_m * fls[0].dx_meter
            debris.debris_binned(gdir, fl_str='inversion_flowlines', ignore_debris=True)
        except:
            fls = None
            
#        print('\n\nsurface height:', fls[0].surface_h)
            
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
                
                assert glacier_rgi_table.O1Region in glena_df.O1Region, glacier_str + ' O1 region not in glena_df'
                
                glena_idx = np.where(glena_df.O1Region == glacier_rgi_table.O1Region)[0][0]
                
                glen_a_multiplier = glena_df.loc[glena_idx,'glens_a_multiplier']
                fs = glena_df.loc[glena_idx,'fs']
            else:
                fs = pygem_prms.fs
                glen_a_multiplier = pygem_prms.glen_a_multiplier
            
            # cfl_number of 0.01 is more conservative than the default of 0.02 (less issues)
            cfg.PARAMS['cfl_number'] = pygem_prms.cfl_number
            
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
            
            out_calving = find_inversion_calving_from_any_mb(gdir, mb_model=mbmod_inv, mb_years=mb_years,
                                                             glen_a=cfg.PARAMS['glen_a']*glen_a_multiplier, fs=fs)
            
            if debug:
                print('out_calving:', out_calving)
            
            if out_calving is not None:
                # calving_flux output is in km3/yr, so need to convert
                calving_flux_Gta_inv = out_calving['calving_flux'] * pygem_prms.density_ice / pygem_prms.density_water
                # Record output
                output_df.loc[nglac,'calving_flux_Gta_inv'] = calving_flux_Gta_inv
                output_df.loc[nglac,'calving_thick'] = out_calving['calving_front_thick']
                
                if debug:                                    
                    print('  inversion:')
                    print('    calving front thickness [m]:', np.round(gdir.get_diagnostics()['calving_front_thick'],0))
                    print('    calving flux [Gt/yr]:', calving_flux_Gta_inv)
    
                # ----- FORWARD MODEL TO ACCOUNT FOR DYNAMICAL FEEDBACKS ----
                # Set up flowlines
                tasks.init_present_time_glacier(gdir) # adds bins below
                debris.debris_binned(gdir, fl_str='model_flowlines', ignore_debris=True)
                nfls = gdir.read_pickle('model_flowlines')
                
                # Mass balance model
                mbmod = PyGEMMassBalance(gdir, modelprms, glacier_rgi_table,
                                         hindcast=pygem_prms.hindcast,
                                         debug=pygem_prms.debug_mb,
                                         debug_refreeze=pygem_prms.debug_refreeze,
                                         fls=nfls, ignore_debris=True)
               
#                try:
                # Glacier dynamics model
                if pygem_prms.option_dynamics == 'OGGM':
                    ev_model = FluxBasedModel(nfls, y0=0, mb_model=mbmod, 
                                              glen_a=cfg.PARAMS['glen_a']*glen_a_multiplier, fs=fs,
                                              is_tidewater=gdir.is_tidewater,
                                              water_level=gdir.get_diagnostics().get('calving_water_level', None),
#                                              calving_use_limiter=False
                                              )
                    
                    if debug:
                        print('New glacier vol', ev_model.volume_m3)
                        graphics.plot_modeloutput_section(ev_model)
                        plt.show()
                        
                        
#                        print(dir(nfls[0]))
                        print('flowline volume:', nfls[0].volume_km3)
                        bin_volume = nfls[0].section * fls[0].dx_meter
                        last_bin_idx = np.where(nfls[0].section > 0)[0][-1]
                        print('last bin:', last_bin_idx)
                        print('surface_h', nfls[0].surface_h[last_bin_idx])
                        print('bin volume (Gt):', bin_volume[last_bin_idx] / 1e9 * 0.9)
           
                    _, diag = ev_model.run_until_and_store(nyears)
                    ev_model.mb_model.glac_wide_volume_annual[-1] = diag.volume_m3[-1]
                    ev_model.mb_model.glac_wide_area_annual[-1] = diag.area_m2[-1]
                
                # Calving flux (Gt/yr) from simulation
                calving_flux_gta = diag.calving_m3.values[-1] * pygem_prms.density_ice / 1e12 / nyears
                    
#                    area_initial = mbmod.glac_bin_area_annual[:,0].sum()
#                    mb_mod_mwea = ((diag.volume_m3.values[-1] - diag.volume_m3.values[0]) 
#                                    / area_initial / nyears * pygem_prms.density_ice / pygem_prms.density_water)
                
                if debug:
                    print('  calving_flux sim (Gt/yr):', np.round(calving_flux_gta,5))
                                  
#                        fl = nfls[-1]
#                        xc = fl.dis_on_line * fl.dx_meter / 1000
#                        f, ax = plt.subplots(1, 1, figsize=(8, 5))
#                        plt.plot(xc, fl.surface_h, '-', color='C1', label='Surface')
#                        plt.plot(xc, gdir.read_pickle('model_flowlines')[-1].bed_h, '--', color='k', label='Glacier bed')
#                        plt.hlines(0, 0, xc[-1], color='C0', linestyle=':'), plt.legend();
#                        plt.show()
#                        
#                        graphics.plot_modeloutput_section(ev_model)
#                        plt.show()
                    
                output_df.loc[nglac,'calving_flux_Gta'] = calving_flux_gta
                output_df.loc[nglac,'no_errors'] = 1
                    
#                except:
#                    output_df.loc[nglac,'calving_flux_Gta'] = np.nan
#                    output_df.loc[nglac,'no_errors'] = 0
                    
                    
            else:
                output_df.loc[nglac,['calving_thick', 'calving_flux_Gta_inv', 'calving_flux_Gta', 'no_errors']] = (
                        np.nan, np.nan, np.nan, 0)
                
    # Remove glaciers that failed to run
    if ignore_nan:
        output_df_good = output_df.dropna(axis=0, subset=['calving_flux_Gta'])
        reg_calving_gta_mod_good = output_df_good.calving_flux_Gta.sum()
        rgiids_data = list(fa_glac_data_reg.RGIId.values)
        rgiids_mod = list(output_df_good.RGIId.values)
        fa_data_idx = [rgiids_data.index(x) for x in rgiids_mod]
        reg_calving_gta_obs_good =fa_glac_data_reg.loc[fa_data_idx,'frontal_ablation_Gta'].sum()
    else:
        reg_calving_gta_mod_good = output_df.calving_flux_Gta.sum()
        reg_calving_gta_obs_good = fa_glac_data_reg.frontal_ablation_Gta.sum()
            
    return output_df, reg_calving_gta_mod_good, reg_calving_gta_obs_good
                
                    
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
                if (not fa_glac_data_reg.loc[nglac,'RGIId'] == 'all' and len(fa_glac_data_reg.loc[nglac,'RGIId']) == 14
                    and fa_glac_data_reg.loc[nglac,'RGIId'] in ['RGI60-03.00191']):
#                if not fa_glac_data_reg.loc[nglac,'RGIId'] == 'all' and len(fa_glac_data_reg.loc[nglac,'RGIId']) == 14:
                    fa_glac_data_reg.loc[nglac,'glacno'] = (str(int(rgiid.split('-')[1].split('.')[0])) + '.' + 
                                                            rgiid.split('-')[1].split('.')[1])
            # Drop observations that aren't of individual glaciers
            fa_glac_data_reg = fa_glac_data_reg.dropna(axis=0, subset=['glacno'])
            fa_glac_data_reg.reset_index(inplace=True, drop=True)
            reg_calving_gta_obs = fa_glac_data_reg.frontal_ablation_Gta.sum()
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
                reg_calving_flux(main_glac_rgi, calving_k, fa_glac_data_reg, ignore_nan=ignore_nan,
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
                output_df_last = output_df.copy()
                reg_calving_gta_mod_last = reg_calving_gta_mod.copy()
                
                # Increase calving k
                calving_k += calving_k_step
                
                if calving_k > calving_k_bndhigh:
                    calving_k = calving_k_bndhigh
                
                # Re-run the regional frontal ablation estimates
                output_df, reg_calving_gta_mod, reg_calving_gta_obs = (
                    reg_calving_flux(main_glac_rgi, calving_k, fa_glac_data_reg, ignore_nan=ignore_nan,
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
                output_df_last = output_df.copy()
                reg_calving_gta_mod_last = reg_calving_gta_mod.copy()
                
                # Increase calving k
                calving_k -= calving_k_step
                
                if calving_k < calving_k_bndlow:
                    calving_k = calving_k_bndlow
                
                # Re-run the regional frontal ablation estimates
                output_df, reg_calving_gta_mod, reg_calving_gta_obs = (
                    reg_calving_flux(main_glac_rgi, calving_k, fa_glac_data_reg, ignore_nan=ignore_nan,
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
                    reg_calving_flux(main_glac_rgi, calving_k, fa_glac_data_reg, ignore_nan=ignore_nan,
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
#        # ----- EXPORT MODEL RESULTS -----
#        if fa_glac_data_reg.loc[0,'RGIId'] == 'all':
#            output_df['fa_gta_obs'] = np.nan
#            output_df['fa_gta_obs_unc'] = np.nan
#        else:
#            fa_obs_dict = dict(zip(fa_glac_data_reg.RGIId, fa_glac_data_reg.frontal_ablation_Gta))
#            fa_obs_unc_dict = dict(zip(fa_glac_data_reg.RGIId, fa_glac_data_reg.frontal_ablation_unc_Gta))
#            fa_glacname_dict = dict(zip(fa_glac_data_reg.RGIId, fa_glac_data_reg.glacier_name))
#            output_df['fa_gta_obs'] = output_df['RGIId'].map(fa_obs_dict)
#            output_df['fa_gta_obs_unc'] = output_df['RGIId'].map(fa_obs_unc_dict)
#            output_df['name'] = output_df['RGIId'].map(fa_glacname_dict)
#        
#        rgi_area_dict = dict(zip(main_glac_rgi.RGIId, main_glac_rgi.Area))
#        output_df['area_km2'] = output_df['RGIId'].map(rgi_area_dict)
#        
#        output_fp = pygem_prms.output_filepath + 'calibration/calving/'
#        if not os.path.exists(output_fp):
#            os.makedirs(output_fp)
#        output_fn = str(reg) + '-calving_cal_reg.csv'
#        output_df.to_csv(output_fp + output_fn, index=False)
#        #%%
#        # ----- PLOT RESULTS FOR EACH GLACIER -----
#        if not fa_glac_data_reg.loc[0,'RGIId'] == 'all':
#            
#            plot_max_raw = np.max([output_df.calving_flux_Gta.max(), output_df.fa_gta_obs.max()])
#            plot_max = 10**np.ceil(np.log10(plot_max_raw))
#    
#            plot_min_raw = np.max([output_df.calving_flux_Gta.min(), output_df.fa_gta_obs.min()])
#            plot_min = 10**np.floor(np.log10(plot_min_raw))
#    
#            x_min, x_max = plot_min, plot_max
#            
#        
#            fig, ax = plt.subplots(1, 1, squeeze=False, gridspec_kw = {'wspace':0, 'hspace':0})
#            
#            # Marker size
#            glac_area_all = output_df['area_km2'].values
#            s_sizes = [10,50,250,1000]
#            s_byarea = np.zeros(glac_area_all.shape) + s_sizes[3]
#            s_byarea[(glac_area_all < 10)] = s_sizes[0]
#            s_byarea[(glac_area_all >= 10) & (glac_area_all < 100)] = s_sizes[1]
#            s_byarea[(glac_area_all >= 100) & (glac_area_all < 1000)] = s_sizes[2]
#            
#            sc = ax[0,0].scatter(output_df['fa_gta_obs'], output_df['calving_flux_Gta'], 
#                                 color='k', marker='o', linewidth=1, facecolor='none', 
#                                 s=s_byarea, clip_on=True)
#            
#            # Labels
#            ax[0,0].set_xlabel('Observed $A_{f}$ (Gt/yr)', size=12)    
#            ax[0,0].set_ylabel('Modeled $A_{f}$ (Gt/yr)', size=12)
#            ax[0,0].set_xlim(x_min,x_max)
#            ax[0,0].set_ylim(x_min,x_max)
#            ax[0,0].plot([x_min, x_max], [x_min, x_max], color='k', linewidth=0.5, zorder=1)
#            
#            ax[0,0].text(0.97, 0.03, '$k_{f} =$' + str(np.round(calving_k,3)), size=10, color='grey',
#                         horizontalalignment='right', verticalalignment='bottom',
#                         transform=ax[0,0].transAxes)
#            
#            # Log scale
#            ax[0,0].set_xscale('log')
#            ax[0,0].set_yscale('log')
#            
#            # Legend
#            obs_labels = ['< 10', '10-100', '100-1000', '> 1000']
#            for nlabel, obs_label in enumerate(obs_labels):
#                ax[0,0].scatter([-10],[-10], color='grey', marker='o', linewidth=1, 
#                                facecolor='none', s=s_sizes[nlabel], zorder=3, label=obs_label)
#            ax[0,0].text(1.08, 0.97, 'Area (km$^{2}$)', size=12, horizontalalignment='left', verticalalignment='top', 
#                         transform=ax[0,0].transAxes, color='grey')
#            leg = ax[0,0].legend(loc='upper left', ncol=1, fontsize=10, frameon=False,
#                                 handletextpad=1, borderpad=0.25, labelspacing=1, labelcolor='grey',
#                                 bbox_to_anchor=(1.035, 0.9))
#            
#    #        # Legend (over plot)
#    #        obs_labels = ['< 10', '10-10$^{2}$', '10$^{2}$-10$^{3}$', '> 10$^{3}$']
#    #        for nlabel, obs_label in enumerate(obs_labels):
#    #            ax[0,0].scatter([-10],[-10], color='grey', marker='o', linewidth=1, 
#    #                            facecolor='none', s=s_sizes[nlabel], zorder=3, label=obs_label)
#    #        ax[0,0].text(0.06, 0.98, 'Area (km$^{2}$)', size=12, horizontalalignment='left', verticalalignment='top', 
#    #                     transform=ax[0,0].transAxes, color='grey')
#    #        leg = ax[0,0].legend(loc='upper left', ncol=1, fontsize=10, frameon=False,
#    #                             handletextpad=1, borderpad=0.25, labelspacing=0.4, bbox_to_anchor=(0.0, 0.93),
#    #                             labelcolor='grey')
#            
#            # Save figure
#            fig.set_size_inches(3.45,3.45)
#            fig_fullfn = output_fp + str(reg) + '-calving_glac_compare-cal_reg.png'
#            fig.savefig(fig_fullfn, bbox_inches='tight', dpi=300)


#%% ===== INDIVIDUAL CALIBRATION =========================================================================================        
if option_ind_calving_k:
    # Load calving glacier data
    fa_glac_data = pd.read_csv(pygem_prms.frontalablation_glacier_data_fullfn)
    
    for reg in regions:
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
        reg_calving_gta_obs = fa_glac_data_reg.frontal_ablation_Gta.sum()
        
        # Glacier numbers for model runs
        glacno_reg_wdata = sorted(list(fa_glac_data_reg.glacno.values))
        
        main_glac_rgi_all = modelsetup.selectglaciersrgitable(glac_no=glacno_reg_wdata)
        # Tidewater glaciers
        termtype_list = [1,5]
        main_glac_rgi = main_glac_rgi_all.loc[main_glac_rgi_all['TermType'].isin(termtype_list)]
        main_glac_rgi.reset_index(inplace=True, drop=True)
        
        output_cns = ['RGIId', 'calving_k', 'calving_thick', 'calving_flux_Gta_inv', 'calving_flux_Gta', 'no_errors']
        output_df_all = pd.DataFrame(np.zeros((main_glac_rgi.shape[0],len(output_cns))), columns=output_cns)
        output_df_all['RGIId'] = main_glac_rgi.RGIId
        
        #%%
        # ----- OPTIMIZE CALVING_K BASED ON INDIVIDUAL GLACIER FRONTAL ABLATION DATA -----
        for nglac in np.arange(main_glac_rgi.shape[0]):
#        for nglac in [14]:
            
            # Reset bounds
            calving_k = calving_k_init
            calving_k_bndlow = 0.01
            calving_k_bndhigh = 5
            calving_k_step = 0.2
            
            # Select individual glacier
            main_glac_rgi_ind = main_glac_rgi.loc[[nglac],:]
            main_glac_rgi_ind.reset_index(inplace=True, drop=True)
            rgiid_ind = main_glac_rgi_ind.loc[0,'RGIId']
            fa_glac_data_ind = fa_glac_data_reg.loc[fa_glac_data_reg.RGIId == rgiid_ind, :]
            fa_glac_data_ind.reset_index(inplace=True, drop=True)
            
            # Estimate frontal ablation
            output_df, reg_calving_gta_mod, reg_calving_gta_obs = (
                    reg_calving_flux(main_glac_rgi_ind, calving_k, fa_glac_data_ind,
                                     prms_from_reg_priors=prms_from_reg_priors, prms_from_glac_cal=prms_from_glac_cal,
                                     ignore_nan=False, debug=True))
            if debug:
                print('  fa_data  [Gt/yr]:', np.round(reg_calving_gta_obs,4))
                print('  fa_model [Gt/yr] :', np.round(reg_calving_gta_mod,4))

#%%
            # ----- Rough optimizer using calving_k_step to loop through parameters within bounds ------
            calving_k_last = calving_k
            output_df_last = output_df.copy()
            reg_calving_gta_mod_last = reg_calving_gta_mod.copy()
            if reg_calving_gta_mod < reg_calving_gta_obs:
                
                if debug:
                    print('\nincrease calving_k')
                    
                while ((reg_calving_gta_mod < reg_calving_gta_obs and np.round(calving_k,2) < calving_k_bndhigh
                       and (np.abs(reg_calving_gta_mod - reg_calving_gta_obs) / reg_calving_gta_obs > 0.1
                            and np.abs(reg_calving_gta_mod - reg_calving_gta_obs) > 1e-3))
                       and not np.isnan(output_df.loc[0,'calving_flux_Gta'])):
                    # Record previous output
                    calving_k_last = calving_k
                    output_df_last = output_df.copy()
                    reg_calving_gta_mod_last = reg_calving_gta_mod.copy()
                    
                    # Increase calving k
                    calving_k += calving_k_step
                    
                    if calving_k > calving_k_bndhigh:
                        calving_k = calving_k_bndhigh
                        
                    # Re-run the regional frontal ablation estimates
                    output_df, reg_calving_gta_mod, reg_calving_gta_obs = (
                        reg_calving_flux(main_glac_rgi_ind, calving_k, fa_glac_data_ind,
                                         prms_from_reg_priors=prms_from_reg_priors, prms_from_glac_cal=prms_from_glac_cal,
                                         ignore_nan=False, debug=True))
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
                    
                while ((reg_calving_gta_mod > reg_calving_gta_obs and np.round(calving_k,2) > calving_k_bndlow
                        and (np.abs(reg_calving_gta_mod - reg_calving_gta_obs) / reg_calving_gta_obs > 0.1
                            and np.abs(reg_calving_gta_mod - reg_calving_gta_obs) > 1e-3))
                       and not np.isnan(output_df.loc[0,'calving_flux_Gta'])):
                    # Record previous output
                    calving_k_last = calving_k
                    output_df_last = output_df.copy()
                    reg_calving_gta_mod_last = reg_calving_gta_mod.copy()
                    
                    # Increase calving k
                    calving_k -= calving_k_step
                    
                    if calving_k < calving_k_bndlow:
                        calving_k = calving_k_bndlow
                        
                    # Re-run the regional frontal ablation estimates
                    output_df, reg_calving_gta_mod, reg_calving_gta_obs = (
                        reg_calving_flux(main_glac_rgi_ind, calving_k, fa_glac_data_ind,
                                         prms_from_reg_priors=prms_from_reg_priors, prms_from_glac_cal=prms_from_glac_cal,
                                         ignore_nan=False, debug=True))
                    if debug:
                        print('  fa_data  [Gt/yr]:', np.round(reg_calving_gta_obs,4))
                        print('  fa_model [Gt/yr] :', np.round(reg_calving_gta_mod,4))
        
                # Set lower bound
                calving_k_bndlow = calving_k
                reg_calving_gta_mod_bndlow = reg_calving_gta_mod
                # Set upper bound
                calving_k_bndhigh = calving_k_last
                reg_calving_gta_mod_bndhigh = reg_calving_gta_mod_last


            # ----- Optimize further using mid-point "bisection" method -----
            # Consider replacing with scipy.optimize.brent
            if not np.isnan(output_df.loc[0,'calving_flux_Gta']):
                
                # Check if upper bound causes good fit
                if (np.abs(reg_calving_gta_mod_bndhigh - reg_calving_gta_obs) / reg_calving_gta_obs < 0.1
                    or np.abs(reg_calving_gta_mod_bndhigh - reg_calving_gta_obs) < 1e-3):
                    
                    # If so, calving_k equals upper bound and re-run to get proper estimates for output
                    calving_k = calving_k_bndhigh
                    
                    # Re-run the regional frontal ablation estimates
                    output_df, reg_calving_gta_mod, reg_calving_gta_obs = (
                        reg_calving_flux(main_glac_rgi_ind, calving_k, fa_glac_data_ind,
                                         prms_from_reg_priors=prms_from_reg_priors, prms_from_glac_cal=prms_from_glac_cal,
                                         ignore_nan=False, debug=True))
                    if debug:
                        print('  fa_data  [Gt/yr]:', np.round(reg_calving_gta_obs,4))
                        print('  fa_model [Gt/yr] :', np.round(reg_calving_gta_mod,4))
                        
                # Check if lower bound causes good fit
                elif (np.abs(reg_calving_gta_mod_bndlow - reg_calving_gta_obs) / reg_calving_gta_obs < 0.1
                      or np.abs(reg_calving_gta_mod_bndlow - reg_calving_gta_obs) < 1e-3):
                    
                    calving_k = calving_k_bndlow
                    # Re-run the regional frontal ablation estimates
                    output_df, reg_calving_gta_mod, reg_calving_gta_obs = (
                        reg_calving_flux(main_glac_rgi_ind, calving_k, fa_glac_data_ind,
                                         prms_from_reg_priors=prms_from_reg_priors, prms_from_glac_cal=prms_from_glac_cal,
                                         ignore_nan=False, debug=True))
                    if debug:
                        print('  fa_data  [Gt/yr]:', np.round(reg_calving_gta_obs,4))
                        print('  fa_model [Gt/yr] :', np.round(reg_calving_gta_mod,4))
                        
                else:
                    # Calibrate between limited range
                    nround = 0
                    while ((np.abs(reg_calving_gta_mod - reg_calving_gta_obs) / reg_calving_gta_obs > 0.1 and 
                            np.abs(reg_calving_gta_mod - reg_calving_gta_obs) > 1e-3) and nround <= nround_max):
                        
                        nround += 1
                        if debug:
                            print('\nRound', nround)
                        # Update calving_k using midpoint
                        calving_k = (calving_k_bndlow + calving_k_bndhigh) / 2
                        output_df, reg_calving_gta_mod, reg_calving_gta_obs = (
                            reg_calving_flux(main_glac_rgi_ind, calving_k, fa_glac_data_ind,
                                             prms_from_reg_priors=prms_from_reg_priors, prms_from_glac_cal=prms_from_glac_cal,
                                             ignore_nan=False, debug=True))
                        if debug:
                            print('  fa_data  [Gt/yr]:', np.round(reg_calving_gta_obs,4))
                            print('  fa_model [Gt/yr] :', np.round(reg_calving_gta_mod,4))
                        
                        # Update bounds
                        if reg_calving_gta_mod < reg_calving_gta_obs:
                            # Update lower bound
                            reg_calving_gta_mod_bndlow = reg_calving_gta_mod
                            calving_k_bndlow = calving_k
                        else:
                            # Update upper bound
                            reg_calving_gta_mod_bndhigh = reg_calving_gta_mod
                            calving_k_bndhigh = calving_k    
                
            output_df_all.loc[nglac,:] = output_df.loc[0,:]
                    
        #%% 
        # ----- EXPORT MODEL RESULTS -----
        fa_obs_dict = dict(zip(fa_glac_data_reg.RGIId, fa_glac_data_reg.frontal_ablation_Gta))
        fa_obs_unc_dict = dict(zip(fa_glac_data_reg.RGIId, fa_glac_data_reg.frontal_ablation_unc_Gta))
        fa_glacname_dict = dict(zip(fa_glac_data_reg.RGIId, fa_glac_data_reg.glacier_name))
        rgi_area_dict = dict(zip(main_glac_rgi.RGIId, main_glac_rgi.Area))
        
        output_df_all['fa_gta_obs'] = output_df_all['RGIId'].map(fa_obs_dict)
        output_df_all['fa_gta_obs_unc'] = output_df_all['RGIId'].map(fa_obs_unc_dict)
        output_df_all['name'] = output_df_all['RGIId'].map(fa_glacname_dict)
        output_df_all['area_km2'] = output_df_all['RGIId'].map(rgi_area_dict)
        
        output_fp = pygem_prms.output_filepath + 'calibration/calving/'
        if not os.path.exists(output_fp):
            os.makedirs(output_fp)
        output_fn = str(reg) + '-calving_cal_ind.csv'
        output_df_all.to_csv(output_fp + output_fn, index=False)

    #%%
        # ----- PLOT RESULTS FOR EACH GLACIER -----
        plot_max_raw = np.max([output_df_all.calving_flux_Gta.max(), output_df_all.fa_gta_obs.max()])
        plot_max = 10**np.ceil(np.log10(plot_max_raw))

        plot_min_raw = np.max([output_df_all.calving_flux_Gta.min(), output_df_all.fa_gta_obs.min()])
        plot_min = 10**np.floor(np.log10(plot_min_raw))
        if plot_min < 1e-3:
            plot_min = 1e-4

        x_min, x_max = plot_min, plot_max
        
        fig, ax = plt.subplots(1, 2, squeeze=False, gridspec_kw = {'wspace':0.3, 'hspace':0})
        
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
        
        # Save figure
        fig.set_size_inches(6,3.45)
        fig_fullfn = output_fp + str(reg) + '-calving_glac_compare-cal_ind.png'
        fig.savefig(fig_fullfn, bbox_inches='tight', dpi=300)
    

#%%             
print('\n\n------TO-DO LIST:------')
print('  - record frontal ablation in md_mod from diagnostic')
print('  - get data for other regions')
print('  - calibrate the model parameters and export those for tidewater glaciers')
print('    --> this needs to include all tidewater glaciers; not just those with data')
print('    --> use the regional calibration parameter to do so')
print('  - test emulator performance for tidewater glaciers')

print('  - add check case to ensure lower bound is actually tested/run because calving_step right now prevents it')