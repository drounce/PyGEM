import os
import logging

import numpy as np
import pandas as pd
import pickle
from scipy.optimize import minimize

from oggm import cfg
from oggm.utils import entity_task, clip_min, clip_scalar
#from oggm.core.inversion import calving_flux_from_depth
import pygem.pygem_input as pygem_prms


if not 'calving_k_opt' in cfg.BASENAMES:
    cfg.BASENAMES['calving_k_opt'] = ('calving_k_opt.pkl', 'calving k optimized with frontal ablation data')

# Module logger
log = logging.getLogger(__name__)

# REPLACE THIS WITH OGGM ONCE OGGM IS FIXED!
def calving_flux_from_depth(gdir, k=None, water_level=None, water_depth=None,
                            thick=None, fixed_water_depth=False):
    """Finds a calving flux from the calving front thickness.

    Approach based on Huss and Hock, (2015) and Oerlemans and Nick (2005).
    We take the initial output of the model and surface elevation data
    to calculate the water depth of the calving front.

    Parameters
    ----------
    gdir : GlacierDirectory
    k : float
        calving constant
    water_level : float
        in case water is not at 0 m a.s.l
    water_depth : float (mandatory)
        the default is to compute the water_depth from ice thickness
        at the terminus and altitude. Set this to force the water depth
        to a certain value
    thick :
        Set this to force the ice thickness to a certain value (for
        sensitivity experiments).
    fixed_water_depth :
        If we have water depth from Bathymetry we fix the water depth
        and forget about the free-board

    Returns
    -------
    A dictionary containing:
    - the calving flux in [km3 yr-1]
    - the frontal width in m
    - the frontal thickness in m
    - the frontal water depth in m
    - the frontal free board in m
    """

    # Defaults
    if k is None:
        k = cfg.PARAMS['inversion_calving_k']

    # Read inversion output
    fl = gdir.read_pickle('inversion_flowlines')[-1]

    # Altitude at the terminus and frontal width
    free_board = clip_min(fl.surface_h[-1], 0) - water_level
    width = fl.widths[-1] * gdir.grid.dx

    # Calving formula
    if thick is None:
        cl = gdir.read_pickle('inversion_output')[-1]
        thick = cl['thick'][-1]
    if water_depth is None:
        water_depth = thick - free_board
    elif not fixed_water_depth:
        # Correct thickness with prescribed water depth
        # If fixed_water_depth=True then we forget about t_altitude
        thick = water_depth + free_board
        
    flux = k * thick * water_depth * width / 1e9
    
    print('freeboard:', free_board)
    print('water depth:', water_depth)
    print('thickness:', thick)
    print('width:', width)
    print('k:', k)
    print('flux:', flux)

    if fixed_water_depth:
        # Recompute free board before returning
        free_board = thick - water_depth

    return {'flux': clip_min(flux, 0),
            'width': width,
            'thick': thick,
            'inversion_calving_k': k,
            'water_depth': water_depth,
            'water_level': water_level,
            'free_board': free_board}


@entity_task(log, writes=['calving_k_opt'])
def calibrate_calving_k_single_wconsensus(gdir, calving_data_fn=pygem_prms.calving_data_fullfn):
    """Calibrate calving parameter k with frontal ablation data and export to the given glacier directory
    
    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        where to write the data
    """
    assert os.path.exists(calving_data_fn), 'Error: ' + calving_data_fn + ' does not exist.'
    
    #%%
    
    # Load calving data
    calving_data = pd.read_csv(pygem_prms.calving_data_fullfn)
    
    if gdir.rgi_id in list(calving_data.RGIId):
        
        # Load glacier data
        gidx = np.where(gdir.rgi_id == calving_data.RGIId)[0][0]
        calving_flux_obs_gta = calving_data.loc[gidx,'frontal_ablation_Gta'] 
        calving_flux_obs_km3a = calving_flux_obs_gta * pygem_prms.density_water / pygem_prms.density_ice
        
        fls = gdir.read_pickle('inversion_flowlines')
        # Ice thickness at calving front from consensus ice thickness
        h_calving = fls[-1].consensus_h[-1]
        # Water level (max based on 50 m freeboard)
        th = fls[-1].surface_h[-1]
        vmin, vmax = cfg.PARAMS['free_board_marine_terminating']
        # Constrain water level such that the freeboard is somewhere between 10 and 50 m
        water_level = clip_scalar(0, th - vmax, th - vmin)

        # ----- Optimize calving_k ------
        def objective(calving_k_obj):
            """ Objective function for calving data (mimize difference between model and observation).
            
            Parameters
            ----------
            calving_k : calving parameter
            """
            # Calving flux (km3 yr-1; positive value refers to amount of calving, need to subtract from mb)
            calving_output = calving_flux_from_depth(gdir, water_level=water_level, thick=h_calving, k=calving_k_obj)
            calving_flux_km3a = calving_output['flux']
            # Difference with observation (km3 yr-1)
            calving_flux_dif_abs = abs(calving_flux_km3a - calving_flux_obs_km3a)
            return calving_flux_dif_abs
        
        # Run objective
        calving_k_init = 1
        calving_k_bnds = ((0,50),)
        calving_k_opt_all = minimize(objective, calving_k_init, method=pygem_prms.method_opt,
                                     bounds=calving_k_bnds, 
#                                     options={'ftol':ftol_opt, 'eps':eps_opt}
                                     )
        # Record the optimized parameters
        calving_k_opt = calving_k_opt_all.x[0]
        calving_k = calving_k_opt
        
        # Calving flux (km3 yr-1; positive value refers to amount of calving, need to subtract from mb)
        calving_output = calving_flux_from_depth(gdir, water_level=water_level, thick=h_calving, k=calving_k)
        calving_flux = calving_output['flux']
            
        print('\n', gdir.rgi_id)
        print('  calving_k:', np.round(calving_k_opt,2))
        print('  calving flux (km3 yr-1):', np.round(calving_flux,3))
        print('  calving flux obs (km3 yr-1):', np.round(calving_flux_obs_km3a,3))
        print('  freeboard:', np.round(calving_output['free_board'],1))
        print('  water_level:', np.round(calving_output['water_level'],1))
        print('  h_calving (m):', np.round(h_calving,1))
        print('  w_calving (m):', np.round(fls[-1].widths[-1] * gdir.grid.dx, 1),'\n')
        
        
        # Pickle data
        output_fn = gdir.get_filepath('calving_k_opt')
        with open(output_fn, 'wb') as f:
            pickle.dump(calving_k_opt, f)