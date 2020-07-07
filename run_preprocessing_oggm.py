""" PRE-PROCESSING FOR MODEL RUNS USING OGGM """


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
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
# Local libraries
import class_climate
#import class_mbdata
import pygem.pygem_input as pygem_prms
import pygemfxns_gcmbiasadj as gcmbiasadj
import pygemfxns_modelsetup as modelsetup
import spc_split_glaciers as split_glaciers
from oggm import cfg
from oggm import graphics
from oggm import tasks, utils, workflow
from oggm.core import climate
from oggm.core.flowline import FluxBasedModel
from oggm.shop import rgitopo
from pygem.massbalance import PyGEMMassBalance
from pygem.glacierdynamics import MassRedistributionCurveModel
from pygem.oggm_compat import single_flowline_glacier_directory
from pygem.shop import debris

#%%
# ===== OGGM CONFIG FILE =====
# Initialize OGGM and set up the default run parameters
cfg.initialize(logging_level='WORKFLOW')
cfg.PARAMS['border'] = 10
# Usually we recommend to set dl_verify to True - here it is quite slow
# because of the huge files so we just turn it off.
# Switch it on for real cases!
cfg.PARAMS['dl_verify'] = True
cfg.PARAMS['use_multiple_flowlines'] = False
# temporary directory for testing (deleted on computer restart)
#cfg.PATHS['working_dir'] = utils.get_temp_dir('PyGEM_ex') 
cfg.PATHS['working_dir'] = pygem_prms.oggm_gdir_fp

# ===== LOAD GLACIERS =====
if pygem_prms.glac_no is not None:
    glac_no = pygem_prms.glac_no
else:
    main_glac_rgi_all = modelsetup.selectglaciersrgitable(
            rgi_regionsO1=pygem_prms.rgi_regionsO1, rgi_regionsO2 =pygem_prms.rgi_regionsO2,
            rgi_glac_number=pygem_prms.rgi_glac_number)
    glac_no = list(main_glac_rgi_all['rgino_str'].values)
    
rgi_ids = ['RGI60-' + x.split('.')[0].zfill(2) + '.' + x.split('.')[1] for x in glac_no]


#%% ===== SELECT BEST DEM =====
# Get the pre-processed topography data
gdirs = rgitopo.init_glacier_directories_from_rgitopo(rgi_ids)


# ===== FLOWLINES ===== 
gdirs = workflow.init_glacier_directories(rgi_ids)

# Compute all the stuff
list_tasks = [
    tasks.glacier_masks,
    tasks.compute_centerlines,
    tasks.initialize_flowlines,
    tasks.compute_downstream_line,
    tasks.catchment_area,
    tasks.catchment_width_geom,
    tasks.catchment_width_correction,
#    tasks.compute_downstream_bedshape,
    debris.debris_to_gdir,
    debris.debris_binned
]

for task in list_tasks:
    workflow.execute_entity_task(task, gdirs)
    # equivalent to:
    #for gdir in gdirs:
    #    tasks.glacier_masks(gdir)
    #    tasks.compute_centerlines(gdir)
    #    ...

# Task format with arguments (unable to use list_tasks if require function arguments)
#workflow.execute_entity_task(debris.debris_to_gdir, gdirs, debris_dir=pygem_prms.debris_fp, add_to_gridded=True)
#workflow.execute_entity_task(debris.debris_binned, gdirs)

# Perform inversion based on PyGEM MB
## Add thickness, width_m, and dx_meter to inversion flowlines so they are compatible with PyGEM's
##  mass balance model (necessary because OGGM's inversion flowlines use pixel distances; however, 
##  this will likely be rectified in the future)
fls_inv = gdirs[0].read_pickle('inversion_flowlines')

#%%


    
#%%
#                    # Perform inversion based on PyGEM MB
#                    # Add thickness, width_m, and dx_meter to inversion flowlines so they are compatible with PyGEM's
#                    #  mass balance model (necessary because OGGM's inversion flowlines use pixel distances; however, 
#                    #  this will likely be rectified in the future)
#                    fls_inv = ngd.read_pickle('inversion_flowlines')
#                    
#                    mbmod_inv = PyGEMMassBalance(modelparameters[0:8], glacier_rgi_table, glacier_gcm_temp,
#                                                 glacier_gcm_tempstd, glacier_gcm_prec, glacier_gcm_elev,
#                                                 glacier_gcm_lrgcm, glacier_gcm_lrglac, dates_table,
#                                                 option_areaconstant=0, hindcast=pygem_prms.hindcast,
#                                                 debug=pygem_prms.debug_mb,
#                                                 debug_refreeze=pygem_prms.debug_refreeze,
#                                                 fls=fls_inv, repeat_period=False,
#                                                 )
#                    
#                    
#                    # Inversion with OGGM model showing the diffences in the mass balance vs. altitude relationships, 
#                    #  which is going to drive different ice thickness estimates
#                    from oggm.core.massbalance import PastMassBalance
#                    oggm_mod = PastMassBalance(gd)
#                    h, w = gd.get_inversion_flowline_hw()
#                    oggm = oggm_mod.get_annual_mb(h, year=2000) * cfg.SEC_IN_YEAR * 900
#                    pyg  = mbmod_inv.get_annual_mb(h, year=0, fl_id=0, fls=fls_inv) * cfg.SEC_IN_YEAR * 900
#                    plt.plot(oggm, h, '.')
#                    plt.plot(pyg, h, '.')
#                    plt.show()
#
#
#                    # Want to reset model parameters
#                    climate.apparent_mb_from_any_mb(ngd, mb_model=mbmod_inv, mb_years=np.arange(18))
#                    tasks.prepare_for_inversion(ngd)
#                    print('setting model parameters here')
##                    fs = 5.7e-20
#                    fs = 0
#                    glen_a_multiplier = 1
#                    tasks.mass_conservation_inversion(ngd, glen_a=cfg.PARAMS['glen_a']*glen_a_multiplier, fs=fs)
#                    tasks.filter_inversion_output(ngd)
#                    tasks.init_present_time_glacier(ngd)
#
#                    nfls = ngd.read_pickle('model_flowlines')
#                    mbmod = PyGEMMassBalance(modelparameters[0:8], glacier_rgi_table, glacier_gcm_temp,
#                                             glacier_gcm_tempstd, glacier_gcm_prec, glacier_gcm_elev,
#                                             glacier_gcm_lrgcm, glacier_gcm_lrglac, dates_table,
#                                             option_areaconstant=0, hindcast=pygem_prms.hindcast,
#                                             debug=pygem_prms.debug_mb,
#                                             debug_refreeze=pygem_prms.debug_refreeze,
#                                             fls=nfls, repeat_period=True)

    

#%%    
## ===== CLIMATE DATA: HISTORICAL (FROM OGGM) ======
## Process the ECMWF climate data from David
#from oggm.shop import ecmwf
#workflow.execute_entity_task(ecmwf.process_ecmwf_data, gdirs, dataset='ERA5dr')
#
## This creates a "climate_historical.nc" file in each glacier directory with the data in it:
#fpath = gdirs[0].get_filepath('climate_historical')
#print(fpath)
