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
from pygem.shop import calving, debris, mbdata, icethickness

#%%
# ===== OGGM CONFIG FILE =====
# Initialize OGGM and set up the default run parameters
cfg.initialize(logging_level='WORKFLOW')
cfg.PARAMS['use_multiprocessing'] = False
#cfg.PARAMS['mp_processes'] = 1
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
            rgi_regionsO1=pygem_prms.rgi_regionsO1, rgi_regionsO2=pygem_prms.rgi_regionsO2,
            rgi_glac_number=pygem_prms.rgi_glac_number)
    glac_no = list(main_glac_rgi_all['rgino_str'].values)
    
rgi_ids = ['RGI60-' + x.split('.')[0].zfill(2) + '.' + x.split('.')[1] for x in glac_no]

#%% ===== SELECT BEST DEM =====
# Get the pre-processed topography data
# - creates directories from scratch
gdirs = rgitopo.init_glacier_directories_from_rgitopo(rgi_ids)


# ===== FLOWLINES (w debris) ===== 
# - checks if directories are created (only use if you're on an already prepared directory)
#gdirs = workflow.init_glacier_directories(rgi_ids)

print('\nTO-DO LIST:')
print(' - reinstall from git\n\n')

# Compute all the stuff
list_tasks = [
        
    # Tasks for OGGM
    tasks.glacier_masks,
    tasks.compute_centerlines,
    tasks.initialize_flowlines,
    tasks.compute_downstream_line,
    tasks.catchment_area,
    tasks.catchment_width_geom,
    tasks.catchment_width_correction,
#    tasks.compute_downstream_line, # check??
#    tasks.compute_downstream_bedshape,
    # OGGM needs this to advance the glacier - it will be the exact same simply with additional bins below
    # - init_present_time_glacier does this!

#    # New workflow following Huss and Farinotti (2012) - squeezed flowline
#    # - squeezed flowline averages slow of all branches over a bin
#    # - OGGM does it based on the main flowline where most of the mass is; also we have more control with frontal ablation width
    
    
    # Debris tasks
    debris.debris_to_gdir,
    debris.debris_binned,
    # Consensus ice thickness
    icethickness.consensus_gridded,
    icethickness.consensus_binned,
    # Mass balance data
    mbdata.mb_df_to_gdir,
]

for task in list_tasks:
    workflow.execute_entity_task(task, gdirs)     



## ===== Mass balance data =====
##mbdata.mb_bins_to_reg_glacierwide(mb_binned_fp=pygem_prms.mb_binned_fp, O1Regions=['01'])
##workflow.execute_entity_task(mbdata.mb_bins_to_glacierwide, gdirs)
#workflow.execute_entity_task(mbdata.mb_df_to_gdir, gdirs)

# ===== CALVING CALIBRATION =====
# Individual glaciers
#for gdir in gdirs:
#    if gdir.is_tidewater:
#        calving.calibrate_calving_k_single_wconsensus(gdir)


## Perform inversion based on PyGEM MB
### Add thickness, width_m, and dx_meter to inversion flowlines so they are compatible with PyGEM's
###  mass balance model (necessary because OGGM's inversion flowlines use pixel distances; however, 
###  this will likely be rectified in the future)
#fls_inv = gdirs[0].read_pickle('inversion_flowlines')

#%%
# ----- Alternative to use squeezed flowlines from Huss and Farinotti (2012) -----
#tasks.simple_glacier_masks, # much more robust mask than the one used for flowlines
#tasks.elevation_band_flowline, # same as Huss and Farinotti; produces the binned elevation (30m), length, and width
#tasks.fixed_dx_elevation_band_flowline, # converts the binned elevation, length, width to the fixed dx grid in OGGM
#                                        # output is the same flowline object
    
# ----- Alternative way of running tasks -----
#for rgi_id in rgi_ids:
#    gdirs = rgitopo.init_glacier_directories_from_rgitopo([rgi_id])
#    gdir = gdirs[0]
#    tasks.glacier_masks(gdir)
#    tasks.compute_centerlines(gdir)
#    tasks.glacier_masks(gdir)
#    tasks.compute_centerlines(gdir)
#    tasks.initialize_flowlines(gdir)
#    tasks.compute_downstream_line(gdir)
#    tasks.catchment_area(gdir)
#    tasks.catchment_width_geom(gdir)
#    tasks.catchment_width_correction(gdir)    
#    # Debris tasks
#    debris.debris_to_gdir(gdir)
#    debris.debris_binned(gdir)
#    # Consensus ice thickness
#    icethickness.consensus_gridded(gdir)
#    icethickness.consensus_binned(gdir)
#    # Tidewater
#    if gdir.is_tidewater:
#        calving.calibrate_calving_k_single_wconsensus(gdir)