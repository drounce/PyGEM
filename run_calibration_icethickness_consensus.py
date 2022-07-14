"""Find the optimal values of glens_a_multiplier to match the consensus ice thickness estimates """

# Built-in libraries
import argparse
from collections import OrderedDict
import os
import time

# External libraries
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

# Local libraries
import class_climate
import pygem.pygem_input as pygem_prms
from pygem.massbalance import PyGEMMassBalance
from pygem.oggm_compat import single_flowline_glacier_directory
import pygem.pygem_modelsetup as modelsetup

from oggm import cfg
from oggm import tasks
from oggm.core import climate

   
#%% ----- MANUAL INPUT DATA -----
regions = [12]

print('setting glacier dynamic model parameters here')
fs = 0                 # keep this set at 0
a_multiplier = 1       # calibrate this based on ice thickness data or the consensus estimates
a_multiplier_bndlow = 0.1
a_multiplier_bndhigh = 10

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


def plot_nfls_section(nfls):
    """
    Modification of OGGM's plot_modeloutput_section()
    """
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_axes([0.07, 0.08, 0.7, 0.84])

    # Compute area histo
    area = np.array([])
    height = np.array([])
    bed = np.array([])
    for cls in nfls:
        a = cls.widths_m * cls.dx_meter * 1e-6
        a = np.where(cls.thick > 0, a, 0)
        area = np.concatenate((area, a))
        height = np.concatenate((height, cls.surface_h))
        bed = np.concatenate((bed, cls.bed_h))
    ylim = [bed.min(), height.max()]

    # Plot histo
    posax = ax.get_position()
    posax = [posax.x0 + 2 * posax.width / 3.0,
             posax.y0,  posax.width / 3.0,
             posax.height]
    axh = fig.add_axes(posax, frameon=False)

    axh.hist(height, orientation='horizontal', range=ylim, bins=20,
             alpha=0.3, weights=area)
    axh.invert_xaxis()
    axh.xaxis.tick_top()
    axh.set_xlabel('Area incl. tributaries (km$^2$)')
    axh.xaxis.set_label_position('top')
    axh.set_ylim(ylim)
    axh.yaxis.set_ticks_position('right')
    axh.set_yticks([])
    axh.axhline(y=ylim[1], color='black', alpha=1)  # qick n dirty trick

    # plot Centerlines
    cls = nfls[-1]
    x = np.arange(cls.nx) * cls.dx * cls.map_dx

    # Plot the bed
    ax.plot(x, cls.bed_h, color='k', linewidth=2.5, label='Bed (Parab.)')

    # Where trapezoid change color
    if hasattr(cls, '_do_trapeze') and cls._do_trapeze:
        bed_t = cls.bed_h * np.NaN
        pt = cls.is_trapezoid & (~cls.is_rectangular)
        bed_t[pt] = cls.bed_h[pt]
        ax.plot(x, bed_t, color='rebeccapurple', linewidth=2.5,
                label='Bed (Trap.)')
        bed_t = cls.bed_h * np.NaN
        bed_t[cls.is_rectangular] = cls.bed_h[cls.is_rectangular]
        ax.plot(x, bed_t, color='crimson', linewidth=2.5, label='Bed (Rect.)')

    # Plot glacier
    def surf_to_nan(surf_h, thick):
        t1 = thick[:-2]
        t2 = thick[1:-1]
        t3 = thick[2:]
        pnan = ((t1 == 0) & (t2 == 0)) & ((t2 == 0) & (t3 == 0))
        surf_h[np.where(pnan)[0] + 1] = np.NaN
        return surf_h
    
    surfh = surf_to_nan(cls.surface_h, cls.thick)
    ax.plot(x, surfh, color='#003399', linewidth=2, label='Glacier')

    ax.set_ylim(ylim)

    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xlabel('Distance along flowline (m)')
    ax.set_ylabel('Altitude (m)')

    # Legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    ax.legend(list(by_label.values()), list(by_label.keys()),
              bbox_to_anchor=(0.5, 1.0),
              frameon=False)
    plt.show()

def reg_vol_comparison(gdirs, mbmods, a_multiplier=1, fs=0, debug=False):
    """ Calculate the modeled volume [km3] and consensus volume [km3] for the given set of glaciers """
    
    reg_vol_km3_consensus = 0
    reg_vol_km3_modeled = 0
    for nglac, gdir in enumerate(gdirs):
        if nglac%2000 == 0:
            print(gdir.rgi_id)
        mbmod_inv = mbmods[nglac]
        
        # Arbitrariliy shift the MB profile up (or down) until mass balance is zero (equilibrium for inversion)
        climate.apparent_mb_from_any_mb(gdir, mb_model=mbmod_inv, mb_years=np.arange(nyears))
    
        tasks.prepare_for_inversion(gdir)
        tasks.mass_conservation_inversion(gdir, glen_a=cfg.PARAMS['glen_a']*a_multiplier, fs=fs)
        tasks.init_present_time_glacier(gdir) # adds bins below
        nfls = gdir.read_pickle('model_flowlines')
            
        # Load consensus volume
        if os.path.exists(gdir.get_filepath('consensus_mass')):
            consensus_fn = gdir.get_filepath('consensus_mass')
            with open(consensus_fn, 'rb') as f:
                consensus_km3 = pickle.load(f) / pygem_prms.density_ice / 1e9
            
            reg_vol_km3_consensus += consensus_km3
            reg_vol_km3_modeled += nfls[0].volume_km3
        
            if debug:   
                plot_nfls_section(nfls)
                print('\n\n  Modeled vol [km3]:  ', nfls[0].volume_km3)
                print('  Consensus vol [km3]:', consensus_km3,'\n\n')
                
    return reg_vol_km3_modeled, reg_vol_km3_consensus

    
#%%
parser = getparser()
args = parser.parse_args()
time_start = time.time()

if args.debug == 1:
    debug = True
else:
    debug = False

# Check that input file set up properly to record results of successful calibration
try:
    os.path.exists(pygem_prms.glena_reg_fullfn)
except:
    assert True==False, "pygem_prms.glena_reg_fullfn is not specified in input file. You may need to set option_dynamics='OGGM'"

# Calibrate each region
for reg in regions:
    
    print('Region:', reg)
    
    # ===== LOAD GLACIERS =====
    main_glac_rgi_all = modelsetup.selectglaciersrgitable(
                rgi_regionsO1=[reg], rgi_regionsO2='all', rgi_glac_number='all', 
                include_landterm=True,include_laketerm=True, include_tidewater=True)
        
    
    main_glac_rgi_all = main_glac_rgi_all.sort_values('Area', ascending=False)
    main_glac_rgi_all.reset_index(inplace=True, drop=True)
    main_glac_rgi_all['Area_cum'] = np.cumsum(main_glac_rgi_all['Area'])
    main_glac_rgi_all['Area_cum_frac'] = main_glac_rgi_all['Area_cum'] / main_glac_rgi_all.Area.sum()
    
    glac_idx = np.where(main_glac_rgi_all.Area_cum_frac > pygem_prms.icethickness_cal_frac_byarea)[0][0]
    main_glac_rgi_subset = main_glac_rgi_all.loc[0:glac_idx, :]
    main_glac_rgi_subset = main_glac_rgi_subset.sort_values('O1Index', ascending=True)
    main_glac_rgi_subset.reset_index(inplace=True, drop=True)
    
    
    print('But only the largest 90% of the glaciers by area, which includes', main_glac_rgi_subset.shape[0], 'glaciers.')
    
    # ===== TIME PERIOD =====
    dates_table = modelsetup.datesmodelrun(
            startyear=pygem_prms.ref_startyear, endyear=pygem_prms.ref_endyear, spinupyears=pygem_prms.ref_spinupyears,
            option_wateryear=pygem_prms.ref_wateryear)
    
    # ===== LOAD CLIMATE DATA =====
    # Climate class
    gcm_name = pygem_prms.ref_gcm_name
    assert gcm_name in ['ERA5', 'ERA-Interim'], 'Error: Calibration not set up for ' + gcm_name
    gcm = class_climate.GCM(name=gcm_name)
    # Air temperature [degC]
    gcm_temp, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(gcm.temp_fn, gcm.temp_vn, main_glac_rgi_subset, dates_table)
    if pygem_prms.option_ablation == 2 and gcm_name in ['ERA5']:
        gcm_tempstd, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(gcm.tempstd_fn, gcm.tempstd_vn,
                                                                        main_glac_rgi_subset, dates_table)
    else:
        gcm_tempstd = np.zeros(gcm_temp.shape)
    # Precipitation [m]
    gcm_prec, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(gcm.prec_fn, gcm.prec_vn, main_glac_rgi_subset, dates_table)
    # Elevation [m asl]
    gcm_elev = gcm.importGCMfxnearestneighbor_xarray(gcm.elev_fn, gcm.elev_vn, main_glac_rgi_subset)
    # Lapse rate [degC m-1]
    if pygem_prms.use_constant_lapserate:
        gcm_lr = np.zeros(gcm_temp.shape) + pygem_prms.lapserate
    else:
        gcm_lr, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(gcm.lr_fn, gcm.lr_vn, main_glac_rgi, dates_table)
    
    # ===== RUN MASS BALANCE =====
    # Number of years (for OGGM's run_until_and_store)
    if pygem_prms.timestep == 'monthly':
        nyears = int(dates_table.shape[0]/12)
    else:
        assert True==False, 'Adjust nyears for non-monthly timestep'
    
    reg_vol_km3_consensus = 0
    reg_vol_km3_modeled = 0
    mbmods = []
    gdirs = []
    for glac in range(main_glac_rgi_subset.shape[0]):
        # Select subsets of data
        glacier_rgi_table = main_glac_rgi_subset.loc[main_glac_rgi_subset.index.values[glac], :]
        glacier_str = '{0:0.5f}'.format(glacier_rgi_table['RGIId_float'])
        
        if glac%1000 == 0:
            print(glacier_str)
    
        # ===== Load glacier data: area (km2), ice thickness (m), width (km) =====
        try:
            gdir = single_flowline_glacier_directory(glacier_str, logging_level='CRITICAL')
            
            # Flowlines
            fls = gdir.read_pickle('inversion_flowlines')
            
            # Add climate data to glacier directory
            gdir.historical_climate = {'elev': gcm_elev[glac],
                                        'temp': gcm_temp[glac,:],
                                        'tempstd': gcm_tempstd[glac,:],
                                        'prec': gcm_prec[glac,:],
                                        'lr': gcm_lr[glac,:]}
            gdir.dates_table = dates_table
        
            glacier_area_km2 = fls[0].widths_m * fls[0].dx_meter / 1e6
            if (fls is not None) and (glacier_area_km2.sum() > 0):
        
                modelprms_fn = glacier_str + '-modelprms_dict.pkl'
                modelprms_fp = pygem_prms.output_filepath + 'calibration/' + glacier_str.split('.')[0].zfill(2) + '/'
                modelprms_fullfn = modelprms_fp + modelprms_fn
                assert os.path.exists(modelprms_fullfn), glacier_str + ' calibrated parameters do not exist.'            
                with open(modelprms_fullfn, 'rb') as f:
                    modelprms_dict = pickle.load(f)
                
                assert 'emulator' in modelprms_dict, ('Error: ' + glacier_str + ' emulator not in modelprms_dict')
                modelprms_all = modelprms_dict['emulator']
        
                # Loop through model parameters
                modelprms = {'kp': modelprms_all['kp'][0],
                              'tbias': modelprms_all['tbias'][0],
                              'ddfsnow': modelprms_all['ddfsnow'][0],
                              'ddfice': modelprms_all['ddfice'][0],
                              'tsnow_threshold': modelprms_all['tsnow_threshold'][0],
                              'precgrad': modelprms_all['precgrad'][0]}
                
                # ----- ICE THICKNESS INVERSION using OGGM -----
                # Apply inversion_filter on mass balance with debris to avoid negative flux
                if pygem_prms.include_debris:
                    inversion_filter = True
                else:
                    inversion_filter = False
                        
                # Perform inversion based on PyGEM MB
                mbmod_inv = PyGEMMassBalance(gdir, modelprms, glacier_rgi_table,
                                              hindcast=pygem_prms.hindcast,
                                              debug=pygem_prms.debug_mb,
                                              debug_refreeze=pygem_prms.debug_refreeze,
                                              fls=fls, option_areaconstant=True,
                                              inversion_filter=inversion_filter)
        
        #        if debug:
        #            h, w = gdir.get_inversion_flowline_hw()
        #            mb_t0 = (mbmod_inv.get_annual_mb(h, year=0, fl_id=0, fls=fls) * cfg.SEC_IN_YEAR * 
        #                     pygem_prms.density_ice / pygem_prms.density_water) 
        #            plt.plot(mb_t0, h, '.')
        #            plt.ylabel('Elevation')
        #            plt.xlabel('Mass balance (mwea)')
        #            plt.show()
                
                mbmods.append(mbmod_inv)
                gdirs.append(gdir)
        except:
            print(glacier_str + ' failed - likely no gdir')
            
    print('\n\n------\nModel setup time:', time.time()-time_start, 's')
                     
    # ===== CHECK BOUNDS =====
    reg_vol_km3_mod, reg_vol_km3_con = reg_vol_comparison(gdirs, mbmods, a_multiplier=a_multiplier, fs=fs, 
                                                          debug=False)
    # Lower bound
    reg_vol_km3_mod_bndlow, reg_vol_km3_con = reg_vol_comparison(gdirs, mbmods, 
                                                                 a_multiplier=a_multiplier_bndlow, fs=fs, 
                                                                 debug=False)
    # Higher bound
    reg_vol_km3_mod_bndhigh, reg_vol_km3_con = reg_vol_comparison(gdirs, mbmods,
                                                                  a_multiplier=a_multiplier_bndhigh, fs=fs, 
                                                                  debug=False)
    
    print('Region:', reg)
    print('Consensus [km3]    :', reg_vol_km3_con)
    print('Model [km3]        :', reg_vol_km3_mod)
    print('Model bndlow [km3] :', reg_vol_km3_mod_bndlow)
    print('Model bndhigh [km3]:', reg_vol_km3_mod_bndhigh)
    
    # ===== OPTIMIZATION =====
    # Check consensus is within bounds
    if reg_vol_km3_con < reg_vol_km3_mod_bndhigh:
        a_multiplier_opt = a_multiplier_bndhigh
    elif reg_vol_km3_con > reg_vol_km3_mod_bndlow:
        a_multiplier_opt = a_multiplier_bndhigh
    # If so, then find optimal glens_a_multiplier
    else:
        def to_minimize(a_multiplier):
            """Objective function to minimize"""
            reg_vol_km3_mod, reg_vol_km3_con = reg_vol_comparison(gdirs, mbmods, a_multiplier=a_multiplier, fs=fs, 
                                                                  debug=False)
            return reg_vol_km3_mod - reg_vol_km3_con
        # Brentq minimization
        a_multiplier_opt, r = brentq(to_minimize, a_multiplier_bndlow, a_multiplier_bndhigh, rtol=1e-2,
                                         full_output=True)
        # Re-run to get estimates
        reg_vol_km3_mod, reg_vol_km3_con = reg_vol_comparison(gdirs, mbmods, a_multiplier=a_multiplier_opt, fs=fs, 
                                                              debug=False)
        
        print('\n\nOptimized:\n  glens_a_multiplier:', np.round(a_multiplier_opt,3))
        print('  Consensus [km3]:', reg_vol_km3_con)
        print('  Model [km3]    :', reg_vol_km3_mod)
            
    # ===== EXPORT RESULTS =====
    glena_cns = ['O1Region', 'count', 'glens_a_multiplier', 'fs', 'reg_vol_km3_consensus', 'reg_vol_km3_modeled']
    glena_df_single = pd.DataFrame(np.zeros((1,len(glena_cns))), columns=glena_cns)
    glena_df_single.loc[0,:] = [reg, main_glac_rgi_subset.shape[0], a_multiplier_opt, fs, reg_vol_km3_con, reg_vol_km3_mod]
    
    if os.path.exists(pygem_prms.glena_reg_fullfn):
        glena_df = pd.read_csv(pygem_prms.glena_reg_fullfn)
        
        # Add or overwrite existing file
        glena_idx = np.where((glena_df.O1Region == reg))[0]
        if len(glena_idx) > 0:
            glena_df.loc[glena_idx,:] = glena_df_single.values
        else:
            glena_df = pd.concat([glena_df, glena_df_single], axis=0)
            
    else:
        glena_df = glena_df_single
        
    glena_df = glena_df.sort_values('O1Region', ascending=True)
    glena_df.reset_index(inplace=True, drop=True)
    glena_df.to_csv(pygem_prms.glena_reg_fullfn, index=False)
    
    
print('\n\n------\nTotal processing time:', time.time()-time_start, 's')
        