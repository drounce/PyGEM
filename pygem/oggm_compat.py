""" PYGEM-OGGGM COMPATIBILITY FUNCTIONS """
# External libraries
import numpy as np
import pandas as pd
import netCDF4
# Local libraries
import pygem_input as pygem_prms
from oggm import cfg, utils
from oggm import workflow
#from oggm import tasks
from oggm.cfg import SEC_IN_YEAR
from oggm.core.massbalance import MassBalanceModel
#from oggm.shop import rgitopo
from pygem.shop import debris, mbdata, icethickness

# Troubleshooting:
#  - EXCEPT: PASS is the key to the issues that is being experienced when running code Fabien provides on mac
#  - also have changed temporary working directories (wd), but the true problem may be the except:pass

class CompatGlacDir:
    def __init__(self, rgiid):
        self.rgiid = rgiid
        
        
        
def single_flowline_glacier_directory(rgi_id, reset=pygem_prms.overwrite_gdirs, prepro_border=80, logging_level='WORKFLOW'):
    """Prepare a GlacierDirectory for PyGEM (single flowline to start with)

    Parameters
    ----------
    rgi_id : str
        the rgi id of the glacier (RGIv60-)
    reset : bool
        set to true to delete any pre-existing files. If false (the default),
        the directory won't be re-downloaded if already available locally in
        order to spare time.
    prepro_border : int
        the size of the glacier map: 10, 80, 160, 250

    Returns
    -------
    a GlacierDirectory object
    """

    if type(rgi_id) != str:
        raise ValueError('We expect rgi_id to be a string')
    if rgi_id.startswith('RGI60-') == False:
        rgi_id = 'RGI60-' + rgi_id.split('.')[0].zfill(2) + '.' + rgi_id.split('.')[1]
    else:
        raise ValueError('Check RGIId is correct')

    # Initialize OGGM and set up the default run parameters
    cfg.initialize(logging_level=logging_level)
#    cfg.initialize(logging_level='CRITICAL')
#    cfg.initialize(logging_level='ERROR')
    # Set multiprocessing to false; otherwise, causes daemonic error due to PyGEM's multiprocessing
    #  - avoids having multiple multiprocessing going on at the same time
    cfg.PARAMS['use_multiprocessing']  = False
    
    # Avoid erroneous glaciers (e.g., Centerlines too short or other issues)
    cfg.PARAMS['continue_on_error'] = True
    
    # Set border boundary
    cfg.PARAMS['border'] = 10
    # Usually we recommend to set dl_verify to True - here it is quite slow
    # because of the huge files so we just turn it off.
    # Switch it on for real cases!
    cfg.PARAMS['dl_verify'] = True
    cfg.PARAMS['use_multiple_flowlines'] = False
    # temporary directory for testing (deleted on computer restart)
    cfg.PATHS['working_dir'] = pygem_prms.oggm_gdir_fp

    # Check if folder is already processed
    if not reset:
        try:
            gdir = utils.GlacierDirectory(rgi_id)
            gdir.read_pickle('inversion_flowlines')
            # If the above works the directory is already processed, return
            return gdir
        except:
            process_gdir = True
        
    else:
        process_gdir = True
    
    if process_gdir:
        # Download preprocessed data
#        gdirs = workflow.init_glacier_directories([rgi_id], from_prepro_level=2, prepro_border=40)
        
        # Start after the prepro task level
#        base_url = 'https://cluster.klima.uni-bremen.de/~fmaussion/gdirs/prepro_l2_202010/single_fl'
#        base_url = 'https://cluster.klima.uni-bremen.de/~fmaussion/gdirs/prepro_l2_202010/elevbands_fl'
        base_url = pygem_prms.oggm_base_url
        
#        try:
        gdirs = workflow.init_glacier_directories([rgi_id], from_prepro_level=2, prepro_border=40, 
                                                  prepro_base_url=base_url, prepro_rgi_version='62',
#                                                  use_demo_glaciers=False
                                                  )
        # Compute all the stuff
        list_tasks = [
#                tasks.glacier_masks,
#                tasks.compute_centerlines,
#                tasks.initialize_flowlines,
#                tasks.compute_downstream_line,
#                tasks.compute_downstream_bedshape,
#                tasks.catchment_area,
#                tasks.catchment_intersections,      
#                tasks.catchment_width_geom,
#                tasks.catchment_width_correction,            
            # Consensus ice thickness
            icethickness.consensus_gridded,
#            icethickness.consensus_binned,
            # Mass balance data
            mbdata.mb_df_to_gdir
        ]
        
        # Debris tasks
        if pygem_prms.include_debris:
            list_tasks.append(debris.debris_to_gdir)
            list_tasks.append(debris.debris_binned)
            
    
        for task in list_tasks:
            workflow.execute_entity_task(task, gdirs)
            
        gdir = gdirs[0]
        
#        except:
#            gdir = None
    
        return gdir
        


def single_flowline_glacier_directory_with_calving(rgi_id, reset=pygem_prms.overwrite_gdirs, prepro_border=80, k_calving=1,
                                                   logging_level='WORKFLOW', 
#                                                   use_demo_glaciers=False
                                                   ):
    """Prepare a GlacierDirectory for PyGEM (single flowline to start with)

    k_calving is free variable!

    Parameters
    ----------
    rgi_id : str
        the rgi id of the glacier
    reset : bool
        set to true to delete any pre-existing files. If false (the default),
        the directory won't be re-downloaded if already available locally in
        order to spare time.
    prepro_border : int
        the size of the glacier map: 10, 80, 160, 250
    Returns
    -------
    a GlacierDirectory object
    """
    if type(rgi_id) != str:
        raise ValueError('We expect rgi_id to be a string')
    if rgi_id.startswith('RGI60-') == False:
        rgi_id = 'RGI60-' + rgi_id.split('.')[0].zfill(2) + '.' + rgi_id.split('.')[1]
    else:
        raise ValueError('Check RGIId is correct')

    # Initialize OGGM and set up the default run parameters
    cfg.initialize(logging_level=logging_level)
    # Set multiprocessing to false; otherwise, causes daemonic error due to PyGEM's multiprocessing
    #  - avoids having multiple multiprocessing going on at the same time
    cfg.PARAMS['use_multiprocessing']  = False
    
#    cfg.PARAMS['has_internet'] = True
    
    # Avoid erroneous glaciers (e.g., Centerlines too short or other issues)
    cfg.PARAMS['continue_on_error'] = True
    
    # Set border boundary
    cfg.PARAMS['border'] = 10
    # Usually we recommend to set dl_verify to True - here it is quite slow
    # because of the huge files so we just turn it off.
    # Switch it on for real cases!
    cfg.PARAMS['dl_verify'] = True
    cfg.PARAMS['use_multiple_flowlines'] = False
    # temporary directory for testing (deleted on computer restart)
    cfg.PATHS['working_dir'] = pygem_prms.oggm_gdir_fp
    
    # Check if folder is already processed
    if not reset:
        try:
            gdir = utils.GlacierDirectory(rgi_id)
            gdir.read_pickle('inversion_flowlines')
            # previously was model_flowlines and not inversion_flowlines
#            gdir.read_pickle('model_flowlines')
            # If the above works the directory is already processed, return
            return gdir
        except:
            process_gdir = True
        
    else:
        process_gdir = True
    
    if process_gdir:
        # Download preprocessed data
#        gdirs = workflow.init_glacier_directories([rgi_id], from_prepro_level=2, prepro_border=40)
        
        # Start after the prepro task level
#        base_url = 'https://cluster.klima.uni-bremen.de/~fmaussion/gdirs/prepro_l2_202010/single_fl'
#        base_url = 'https://cluster.klima.uni-bremen.de/~fmaussion/gdirs/prepro_l2_202010/elevbands_fl'
        base_url = pygem_prms.oggm_base_url
        gdirs = workflow.init_glacier_directories([rgi_id], from_prepro_level=2, prepro_border=40, 
                                                  prepro_base_url=base_url, prepro_rgi_version='62')
        
        if not gdirs[0].is_tidewater:
            raise ValueError('This glacier is not tidewater!')
            
        # Compute all the stuff
        list_tasks = [
            # Consensus ice thickness
            icethickness.consensus_gridded,
#            icethickness.consensus_binned,
            # Mass balance data
            mbdata.mb_df_to_gdir
        ]
        
        for task in list_tasks:
            # The order matters!
            workflow.execute_entity_task(task, gdirs)
            
        return gdirs[0]        


def create_empty_glacier_directory(rgi_id):
    """Create empty GlacierDirectory for PyGEM's alternative ice thickness products

    Parameters
    ----------
    rgi_id : str
        the rgi id of the glacier (RGIv60-)

    Returns
    -------
    a GlacierDirectory object
    """
    # RGIId check
    if type(rgi_id) != str:
        raise ValueError('We expect rgi_id to be a string')
    assert rgi_id.startswith('RGI60-'), 'Check RGIId starts with RGI60-'

    # Create empty directory
    gdir = CompatGlacDir(rgi_id)

    return gdir


def get_glacier_zwh(gdir):
    """Computes this glaciers altitude, width and ice thickness.

    Parameters
    ----------
    gdir : GlacierDirectory
        the glacier to compute

    Returns
    -------
    a dataframe with the requested data
    """

    fls = gdir.read_pickle('model_flowlines')
    z = np.array([])
    w = np.array([])
    h = np.array([])
    for fl in fls:
        # Widths (in m)
        w = np.append(w, fl.widths_m)
        # Altitude (in m)
        z = np.append(z, fl.surface_h)
        # Ice thickness (in m)
        h = np.append(h, fl.thick)
    # Distance between two points
    dx = fl.dx_meter

    # Output
    df = pd.DataFrame()
    df['z'] = z
    df['w'] = w
    df['h'] = h
    df['dx'] = dx

    return df


class RandomLinearMassBalance(MassBalanceModel):
    """Mass-balance as a linear function of altitude with random ELA.

    This is a dummy MB model to illustrate how to program one.

    The reference ELA is taken at a percentile altitude of the glacier.
    It then varies randomly from year to year.

    This class implements the MassBalanceModel interface so that the
    dynamical model can use it. Even if you are not familiar with object
    oriented programming, I hope that the example below is simple enough.
    """

    def __init__(self, gdir, grad=3., h_perc=60, sigma_ela=100., seed=None):
        """ Initialize.

        Parameters
        ----------
        gdir : oggm.GlacierDirectory
            the working glacier directory
        grad: float
            Mass-balance gradient (unit: [mm w.e. yr-1 m-1])
        h_perc: int
            The percentile of the glacier elevation to choose the ELA
        sigma_ela: float
            The standard deviation of the ELA (unit: [m])
        seed : int, optional
            Random seed used to initialize the pseudo-random number generator.

        """
        super(RandomLinearMassBalance, self).__init__()
        self.valid_bounds = [-1e4, 2e4]  # in m
        self.grad = grad
        self.sigma_ela = sigma_ela
        self.hemisphere = 'nh'
        self.rng = np.random.RandomState(seed)

        # Decide on a reference ELA
        grids_file = gdir.get_filepath('gridded_data')
        with netCDF4.Dataset(grids_file) as nc:
            glacier_mask = nc.variables['glacier_mask'][:]
            glacier_topo = nc.variables['topo_smoothed'][:]

        self.orig_ela_h = np.percentile(glacier_topo[glacier_mask == 1],
                                        h_perc)
        self.ela_h_per_year = dict()  # empty dictionary

    def get_random_ela_h(self, year):
        """This generates a random ELA for the requested year.

        Since we do not know which years are going to be asked for we generate
        them on the go.
        """

        year = int(year)
        if year in self.ela_h_per_year:
            # If already computed, nothing to be done
            return self.ela_h_per_year[year]

        # Else we generate it for this year
        ela_h = self.orig_ela_h + self.rng.randn() * self.sigma_ela
        self.ela_h_per_year[year] = ela_h
        return ela_h

    def get_annual_mb(self, heights, year=None, fl_id=None):

        # Compute the mass-balance gradient
        ela_h = self.get_random_ela_h(year)
        mb = (np.asarray(heights) - ela_h) * self.grad

        # Convert to units of [m s-1] (meters of ice per second)
        return mb / SEC_IN_YEAR / cfg.PARAMS['ice_density']
