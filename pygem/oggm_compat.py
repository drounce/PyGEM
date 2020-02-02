from oggm import cfg, utils
from oggm import workflow
from oggm import tasks
from oggm.cfg import SEC_IN_YEAR
from oggm.core.massbalance import MassBalanceModel
import numpy as np
import pandas as pd
import netCDF4


def single_flowline_glacier_directory(rgi_id, reset=False, prepro_border=80):
    """Prepare a GlacierDirectory for PyGEM (single flowline to start with)

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
    if 'RGI60-' not in rgi_id:
        raise ValueError('OGGM currently expects IDs to start with RGI60-')

    cfg.initialize()
    wd = utils.gettempdir(dirname='pygem-{}-b{}'.format(rgi_id, prepro_border),
                          reset=reset)
    cfg.PATHS['working_dir'] = wd
    cfg.PARAMS['use_multiple_flowlines'] = False

    # Check if folder is already processed
    try:
        gdir = utils.GlacierDirectory(rgi_id)
        gdir.read_pickle('model_flowlines')
        # If the above works the directory is already processed, return
        return gdir
    except OSError:
        pass

    # If not ready, we download the preprocessed data for this glacier
    gdirs = workflow.init_glacier_regions([rgi_id],
                                          from_prepro_level=2,
                                          prepro_border=prepro_border)
    # Compute all the stuff
    list_talks = [
        tasks.glacier_masks,
        tasks.compute_centerlines,
        tasks.initialize_flowlines,
        tasks.compute_downstream_line,
        tasks.catchment_area,
        tasks.catchment_width_geom,
        tasks.catchment_width_correction,
        tasks.compute_downstream_bedshape,
        tasks.local_t_star,
        tasks.mu_star_calibration,
        tasks.prepare_for_inversion,
        tasks.mass_conservation_inversion,
        tasks.filter_inversion_output,
        tasks.init_present_time_glacier,
    ]
    for task in list_talks:
        # The order matters!
        workflow.execute_entity_task(task, gdirs)

    return gdirs[0]


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
