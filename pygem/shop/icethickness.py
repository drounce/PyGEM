import os
import logging

import numpy as np
import pandas as pd
import pickle
import rasterio
import xarray as xr

from oggm import cfg
from oggm.utils import entity_task
#from oggm.core.gis import rasterio_to_gdir
#from oggm.utils import ncDataset
import pygem.pygem_input as pygem_prms

if not 'mass_consensus' in cfg.BASENAMES:
    cfg.BASENAMES['mass_consensus'] = ('mass_consensus.pkl', 'Glacier mass from consensus ice thickness estimate')

# Module logger
log = logging.getLogger(__name__)


@entity_task(log, writes=['mass_consensus'])
def consensus_mass_estimate(gdir, h_consensus_fp=pygem_prms.h_consensus_fp):
    """Convert consensus ice thickness estimate to glacier mass and add to the given glacier directory
    
    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        where to write the data
    """
    
    assert os.path.exists(h_consensus_fp), "Error: h_consensus_fp does not exist."
    
    # If binned mb data exists, then write to glacier directory
    if os.path.exists(h_consensus_fp + 'RGI60-' + gdir.rgi_region + '/' + gdir.rgi_id + '_thickness.tif'):
        h_fn = h_consensus_fp + 'RGI60-' + gdir.rgi_region + '/' + gdir.rgi_id + '_thickness.tif'
    else: 
        h_fn = None
        
    if h_fn is not None:
        # open consensus ice thickness estimate
        h_dr = rasterio.open(h_fn, 'r', driver='GTiff')
        h = h_dr.read(1).astype(rasterio.float32)
        # Glacier mass [kg]
        glacier_mass = (h * h_dr.res[0] * h_dr.res[1]).sum() * pygem_prms.density_ice

        # Pickle data
        consensus_fn = gdir.get_filepath('mass_consensus')
        with open(consensus_fn, 'wb') as f:
            pickle.dump(glacier_mass, f)