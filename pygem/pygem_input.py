"""Model inputs to run PyGEM"""

# Built-in libraries
import os
# External libraries
import numpy as np
# Local libaries
#from pygem.utils._funcs_selectglaciers import get_same_glaciers, glac_num_fromrange, glac_fromcsv, glac_wo_cal


#%% ===== MODEL SETUP DIRECTORY =====
main_directory = os.getcwd()
# Output directory
output_filepath = main_directory + '/../Output/'
model_run_date = 'January 30 2021'

#%% ===== GLACIER SELECTION =====
rgi_regionsO1 = [1]                 # 1st order region number (RGI V6.0)
rgi_regionsO2 = 'all'               # 2nd order region number (RGI V6.0)
# RGI glacier number (RGI V6.0)
#  Three options: (1) use glacier numbers for a given region (or 'all'), must have glac_no set to None
#                 (2) glac_no is not None, e.g., ['1.00001', 13.0001'], overrides rgi_glac_number
#                 (3) use one of the functions from  utils._funcs_selectglaciers
rgi_glac_number = 'all'
#rgi_glac_number = ['00001']
#rgi_glac_number = glac_num_fromrange(1,48)

glac_no_skip = None
glac_no = None
glac_no = ['1.00570']
glac_no = ['18.02397']

if glac_no is not None:
    rgi_regionsO1 = sorted(list(set([int(x.split('.')[0]) for x in glac_no])))

# Types of glaciers to include (True) or exclude (False)
include_landterm = True                # Switch to include land-terminating glaciers
include_laketerm = True                # Switch to include lake-terminating glaciers
include_tidewater = True               # Switch to include tidewater glaciers
ignore_calving = False                 # Switch to ignore calving and treat tidewater glaciers as land-terminating

oggm_base_url = 'https://cluster.klima.uni-bremen.de/~fmaussion/gdirs/prepro_l2_202010/elevbands_fl_with_consensus'
#oggm_base_url = 'https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.4/L1-L2_files/elev_bands/'
logging_level = 'DEBUG' # DEBUG, INFO, WARNING, ERROR, WORKFLOW, CRITICAL (recommended WORKFLOW)

#%% ===== CLIMATE DATA ===== 
# Reference period runs (reference period refers to the calibration period)
ref_gcm_name = 'ERA5'               # reference climate dataset
ref_startyear = 2000                # first year of model run (reference dataset)
ref_endyear = 2019                  # last year of model run (reference dataset)
ref_wateryear = 'calendar'          # options for years: 'calendar', 'hydro', 'custom'
ref_spinupyears = 0                 # spin up years
if ref_spinupyears > 0:
    assert 0==1, 'Code needs to be tested to ensure spinup years are correctly accounted for in output files'

# Simulation runs (refers to period of simulation and needed separately from reference year to account for bias adjustments)
gcm_startyear = 2000            # first year of model run (simulation dataset)
gcm_endyear = 2019              # last year of model run (simulation dataset)
gcm_wateryear = 'calendar'      # options for years: 'calendar', 'hydro', 'custom'
gcm_spinupyears = 0             # spin up years for simulation (output not set up for spinup years at present)
constantarea_years = 0          # number of years to not let the area or volume change
if gcm_spinupyears > 0:
    assert 0==1, 'Code needs to be tested to enure spinup years are correctly accounted for in output files'

# Hindcast option (flips array so 1960-2000 would run 2000-1960 ensuring that glacier area at 2000 is correct)
hindcast = False                # True: run hindcast simulation, False: do not
if hindcast:
    gcm_startyear = 1980        # first year of model run (simulation dataset)
    gcm_endyear = 2000          # last year of model run (simulation dataset)


#%% ===== CALIBRATION OPTIONS =====
# Calibration option ('emulator', 'MCMC', 'MCMC_fullsim' 'HH2015', 'HH2015mod')
option_calibration = 'MCMC'
#option_calibration = 'emulator'
#option_calibration = 'MCMC_fullsim'

# Prior distribution (specify filename or set equal to None)
priors_reg_fullfn = main_directory + '/../Output/calibration/priors_region.csv'

# Calibration-specific information for each calibration option
if option_calibration == 'HH2015':
    tbias_init = 0
    tbias_step = 1
    kp_init = 1.5
    kp_bndlow = 0.8
    kp_bndhigh = 2
    ddfsnow_init = 0.003
    ddfsnow_bndlow = 0.00175
    ddfsnow_bndhigh = 0.0045
    
elif option_calibration == 'HH2015mod':
    # Initial parameters
    tbias_init = 0
    tbias_step = 0.5
    kp_init = 1
    kp_bndlow = 0.5
    kp_bndhigh = 3
    ddfsnow_init = 0.0041
    # Minimization details
    method_opt = 'SLSQP'            # SciPy optimization scheme ('SLSQP' or 'L-BFGS-B')
    params2opt = ['tbias', 'kp']    # parameters to optimize
    ftol_opt = 1e-3                 # tolerance for SciPy optimization scheme
    eps_opt = 0.01                  # epsilon (adjust variables for jacobian) for SciPy optimization scheme (1e-6 works)
    
elif option_calibration == 'emulator':
    emulator_sims = 100             # Number of simulations to develop the emulator
    overwrite_em_sims = False       # Overwrite emulator simulations
    opt_hh2015_mod = True           # Option to also perform the HH2015_mod calibration using the emulator
    emulator_fp = output_filepath + 'emulator/'
    tbias_step = 0.5                # tbias step size
    tbias_init = 0                  # tbias initial value
    kp_init = 1                     # kp initial value
    kp_bndlow = 0.5                 # kp lower bound
    kp_bndhigh = 3                  # kp upper bound
    ddfsnow_init = 0.0041           # ddfsnow initial value
    option_areaconstant = True      # Option to keep area constant or evolve
    # Distributions
    tbias_disttype = 'truncnormal'  # Temperature bias distribution ('truncnormal', 'uniform')
    tbias_sigma = 3                 # tbias standard deviation for truncnormal distribution
    kp_gamma_alpha = 2              # Precipitation factor gamma distribution alpha
    kp_gamma_beta = 1               # Precipitation factor gamma distribution beta
    ddfsnow_disttype = 'truncnormal'# Degree-day factor of snow distribution ('truncnormal')
    ddfsnow_mu = 0.0041             # ddfsnow mean
    ddfsnow_sigma = 0.0015          # ddfsnow standard deviation
    ddfsnow_bndlow = 0              # ddfsnow lower bound
    ddfsnow_bndhigh = np.inf        # ddfsnow upper bound
    # Minimization details 
    method_opt = 'SLSQP'            # SciPy optimization scheme ('SLSQP' or 'L-BFGS-B')
    params2opt = ['tbias', 'kp']    # parameters to optimize
    ftol_opt = 1e-6                 # tolerance for SciPy optimization scheme
    eps_opt = 0.01                  # epsilon (adjust variables for jacobian) for SciPy optimization scheme
    
elif option_calibration in ['MCMC', 'MCMC_fullsim']:
    emulator_fp = output_filepath + 'emulator/'
    emulator_sims = 100
    tbias_step = 1
    tbias_stepsmall = 0.1
    option_areaconstant = True      # Option to keep area constant or evolve
    # Chain options
    n_chains = 1                    # number of chains (min 1, max 3)
    mcmc_sample_no = 10000          # number of steps (10000 was found to be sufficient in HMA)
    mcmc_burn_no = 200              # number of steps to burn-in (0 records all steps in chain)
#    mcmc_sample_no = 100          # number of steps (10000 was found to be sufficient in HMA)
#    mcmc_burn_no = 0              # number of steps to burn-in (0 records all steps in chain)
    mcmc_step = None                # step option (None or 'am')
    thin_interval = 10              # thin interval if need to reduce file size (best to leave at 1 if space allows)
    # Degree-day factor of snow distribution options
    ddfsnow_disttype = 'truncnormal'# distribution type ('truncnormal', 'uniform')
    ddfsnow_mu = 0.0041             # ddfsnow mean
    ddfsnow_sigma = 0.0015          # ddfsnow standard deviation
    ddfsnow_bndlow = 0              # ddfsnow lower bound
    ddfsnow_bndhigh = np.inf        # ddfsnow upper bound
    ddfsnow_start=ddfsnow_mu        # ddfsnow initial chain value
    # Precipitation factor distribution options
    kp_disttype = 'gamma'           # distribution type ('gamma' (recommended), 'lognormal', 'uniform')
    if priors_reg_fullfn is None:
        kp_gamma_alpha = 9          # precipitation factor alpha value of gamma distribution
        kp_gamma_beta = 4           # precipitation factor beta value of gamme distribution
        kp_lognorm_mu = 0           # precipitation factor mean of log normal distribution
        kp_lognorm_tau = 4          # precipitation factor tau of log normal distribution
        kp_mu = 0                   # precipitation factor mean of normal distribution
        kp_sigma = 1.5              # precipitation factor standard deviation of normal distribution
        kp_bndlow = 0.5             # precipitation factor lower bound
        kp_bndhigh = 1.5            # precipitation factor upper bound
        kp_start = 1                # precipitation factor initial chain value
    # Temperature bias distribution options
    tbias_disttype = 'normal'       # distribution type ('normal' (recommended), 'truncnormal', 'uniform')
    if priors_reg_fullfn is None:
        tbias_mu = 0                # temperature bias mean of normal distribution
        tbias_sigma = 1             # temperature bias mean of standard deviation
        tbias_bndlow = -10          # temperature bias lower bound
        tbias_bndhigh = 10          # temperature bias upper bound
        tbias_start = tbias_mu      # temperature bias initial chain value


# ----- Calibration Dataset -----
# Hugonnet geodetic mass balance data
hugonnet_fp = main_directory + '/../DEMs/Hugonnet2020/'
#hugonnet_fn = 'df_pergla_global_20yr-filled.csv'
hugonnet_fn = 'df_pergla_global_20yr-filled-FAcorrected.csv'
if '-filled' in hugonnet_fn:
    hugonnet_mb_cn = 'mb_mwea'
    hugonnet_mb_err_cn = 'mb_mwea_err'
    hugonnet_rgi_glacno_cn = 'RGIId'
    hugonnet_mb_clim_cn = 'mb_clim_mwea'
    hugonnet_mb_clim_err_cn = 'mb_clim_mwea_err'
else:
    hugonnet_mb_cn = 'dmdtda'
    hugonnet_mb_err_cn = 'err_dmdtda'
    hugonnet_rgi_glacno_cn = 'rgiid'
hugonnet_time1_cn = 't1'
hugonnet_time2_cn = 't2'
hugonnet_area_cn = 'area_km2'

# ----- Ice thickness calibration parameter -----
icethickness_cal_frac_byarea = 0.9  # Regional glacier area fraction that is used to calibrate the ice thickness
                                    #  e.g., 0.9 means only the largest 90% of glaciers by area will be used to calibrate
                                    #  glen's a for that region.

#%% ===== SIMULATION AND GLACIER DYNAMICS OPTIONS =====
# Glacier dynamics scheme (options: 'OGGM', 'MassRedistributionCurves', None)
option_dynamics = 'OGGM'
    
# MCMC options
if option_calibration == 'MCMC':
    sim_iters = 1                  # number of simulations
    sim_burn = 0                    # number of burn-in (if burn-in is done in MCMC sampling, then don't do here)
else:
    sim_iters = 1                   # number of simulations

# Output filepath of simulations
output_sim_fp = output_filepath + 'simulations/'
# Output statistics of simulation (options include any of the following 'mean', 'std', '2.5%', '25%', 'median', '75%', '97.5%')
sim_stat_cns = ['median', 'mad']

# Output options
export_essential_data = True        # Export essential data (ex. mass balance components, ElA, etc.)
export_binned_thickness = False      # Export binned ice thickness
export_binned_area_threshold = 0    # Area threshold for exporting binned ice thickness
export_extra_vars = True            # Option to export extra variables (temp, prec, melt, acc, etc.)

# Bias adjustment option (0: no adjustment, 1: new prec scheme and temp building on HH2015, 2: HH2015 methods)
option_bias_adjustment = 1

# OGGM glacier dynamics parameters
if option_dynamics in ['OGGM', 'MassRedistributionCurves']:
    cfl_number = 0.02
    cfl_number_calving = 0.01
    glena_reg_fullfn = main_directory + '/../Output/calibration/glena_region.csv'
    use_reg_glena = True
    if use_reg_glena:
        assert os.path.exists(glena_reg_fullfn), 'Regional glens a calibration file does not exist.'
    else:
        fs = 0
        glen_a_multiplier = 1

# Mass redistribution / Glacier geometry change options
icethickness_advancethreshold = 5   # advancing glacier ice thickness change threshold (5 m in Huss and Hock, 2015)
terminus_percentage = 20            # glacier (%) considered terminus (20% in HH2015), used to size advancing new bins


#%% ===== MODEL PARAMETERS =====
use_calibrated_modelparams = True   # False: use input values, True: use calibrated model parameters
use_constant_lapserate = False      # False: use spatially and temporally varying lapse rate, True: use constant value specified below
if not use_calibrated_modelparams:
    print('\nWARNING: using non-calibrated model parameters\n')
    sim_iters = 1
    
kp = 1                              # precipitation factor [-] (referred to as k_p in Radic etal 2013; c_prec in HH2015)
tbias = 5                           # temperature bias [deg C]
ddfsnow = 0.0041                    # degree-day factor of snow [m w.e. d-1 degC-1]
ddfsnow_iceratio = 0.7              # Ratio degree-day factor snow snow to ice
ddfice = ddfsnow / ddfsnow_iceratio # degree-day factor of ice [m w.e. d-1 degC-1]
precgrad = 0.0001                   # precipitation gradient on glacier [m-1]
lapserate = -0.0065                 # temperature lapse rate for both gcm to glacier and on glacier between elevation bins [K m-1]
tsnow_threshold = 1                 # temperature threshold for snow [deg C] (HH2015 used 1.5 degC +/- 1 degC)
calving_k = 0.7                     # frontal ablation rate [yr-1]

# Frontal ablation calibrated file
calving_fp = main_directory + '/../calving_data/analysis/'
calving_fn = 'all-calving_cal_ind.csv'


#%% ===== MASS BALANCE MODEL OPTIONS =====
# Initial surface type options
option_surfacetype_initial = 1
#  option 1 (default) - use median elevation to classify snow/firn above the median and ice below.
#   > Sakai et al. (2015) found that the decadal ELAs are consistent with the median elevation of nine glaciers in High
#     Mountain Asia, and Nuimura et al. (2015) also found that the snow line altitude of glaciers in China corresponded
#     well with the median elevation.  Therefore, the use of the median elevation for defining the initial surface type
#     appears to be a fairly reasonable assumption in High Mountain Asia.
#  option 2 - use mean elevation
include_firn = True                 # True: firn included, False: firn is modeled as snow
include_debris = True               # True: account for debris with melt factors, False: do not account for debris

# Downscaling model options
# Reference elevation options for downscaling climate variables
option_elev_ref_downscale = 'Zmed'  # 'Zmed', 'Zmax', or 'Zmin' for median, maximum or minimum glacier elevations
# Downscale temperature to bins options
option_temp2bins = 1                # 1: lr_gcm and lr_glac to adjust temp from gcm to the glacier bins
option_adjusttemp_surfelev = 1      # 1: adjust temps based on surface elev changes; 0: no adjustment
# Downscale precipitation to bins options
option_prec2bins = 1                # 1: prec_factor and prec_grad to adjust precip from gcm to the glacier bins
option_preclimit = 0                # 1: limit the uppermost 25% using an expontial fxn

# Accumulation model options
option_accumulation = 2             # 1: single threshold, 2: threshold +/- 1 deg using linear interpolation

# Ablation model options
option_ablation = 1                 # 1: monthly temp, 2: superimposed daily temps enabling melt near 0 (HH2015)
option_ddf_firn = 1                 # 0: ddf_firn = ddf_snow; 1: ddf_firn = mean of ddf_snow and ddf_ice
ddfdebris = ddfice                  # add options for handling debris-covered glaciers

# Refreezing model option (options: 'Woodward' or 'HH2015')
#  Woodward refers to Woodward et al. 1997 based on mean annual air temperature
#  HH2015 refers to heat conduction in Huss and Hock 2015
option_refreezing = 'Woodward'      # Woodward: annual air temp (Woodward etal 1997)
if option_refreezing == 'Woodward':
    rf_month = 10                   # refreeze month
elif option_refreezing == 'HH2015':
    rf_layers = 5                   # number of layers for refreezing model (8 is sufficient - Matthias)
#    rf_layers_max = 8               # number of layers to include for refreeze calculation
    rf_dz = 10/rf_layers            # layer thickness (m)
    rf_dsc = 3                      # number of time steps for numerical stability (3 is sufficient - Matthias)
    rf_meltcrit = 0.002             # critical amount of melt [m w.e.] for initializing refreezing module
    pp = 0.3                        # additional refreeze water to account for water refreezing at bare-ice surface
    rf_dens_top = 300               # snow density at surface (kg m-3)
    rf_dens_bot = 650               # snow density at bottom refreezing layer (kg m-3)
    option_rf_limit_meltsnow = 1
    
    
#%% CLIMATE DATA
# ERA5 (default reference climate data)
if ref_gcm_name == 'ERA5':
    era5_fp = main_directory + '/../climate_data/ERA5/'
    era5_temp_fn = 'ERA5_temp_monthly.nc'
    era5_tempstd_fn = 'ERA5_tempstd_monthly.nc'
    era5_prec_fn = 'ERA5_totalprecip_monthly.nc'
    era5_elev_fn = 'ERA5_geopotential.nc'
    era5_pressureleveltemp_fn = 'ERA5_pressureleveltemp_monthly.nc'
    era5_lr_fn = 'ERA5_lapserates_monthly.nc'
    assert os.path.exists(era5_fp), 'ERA5 filepath does not exist'
    assert os.path.exists(era5_fp + era5_temp_fn), 'ERA5 temperature filepath does not exist'
    assert os.path.exists(era5_fp + era5_prec_fn), 'ERA5 precipitation filepath does not exist'
    assert os.path.exists(era5_fp + era5_elev_fn), 'ERA5 elevation data does not exist'
    if not use_constant_lapserate:
        assert os.path.exists(era5_fp + era5_lr_fn), 'ERA5 lapse rate data does not exist'
    if option_ablation == 2:
        assert os.path.exists(era5_fp + era5_tempstd_fn), 'ERA5 temperature std filepath does not exist'

# CMIP5 (GCM data)
cmip5_fp_var_prefix = main_directory + '/../climate_data/cmip5/'
cmip5_fp_var_ending = '_r1i1p1_monNG/'
cmip5_fp_fx_prefix = main_directory + '/../climate_data/cmip5/'
cmip5_fp_fx_ending = '_r0i0p0_fx/'

# CMIP6 (GCM data)
cmip6_fp_prefix = main_directory + '/../climate_data/cmip6/'


#%% ===== GLACIER DATA (RGI, ICE THICKNESS, ETC.) =====
# ----- RGI DATA -----
# Filepath for RGI files
rgi_fp = main_directory + '/../RGI/rgi60/00_rgi60_attribs/'
assert os.path.exists(rgi_fp), 'RGI filepath does not exist. PyGEM requires RGI data to run.'
# Column names
rgi_lat_colname = 'CenLat'
rgi_lon_colname = 'CenLon_360' # REQUIRED OTHERWISE GLACIERS IN WESTERN HEMISPHERE USE 0 deg
elev_colname = 'elev'
indexname = 'GlacNo'
rgi_O1Id_colname = 'glacno'
rgi_glacno_float_colname = 'RGIId_float'
# Column names from table to drop (list names or accept an empty list)
rgi_cols_drop = ['GLIMSId','BgnDate','EndDate','Status','Linkages','Name']

# ----- ADDITIONAL DATA (hypsometry, ice thickness, width, debris) -----
h_consensus_fp = main_directory + '/../IceThickness_Farinotti/composite_thickness_RGI60-all_regions/'
# Filepath for the hypsometry files
binsize = 10            # Elevation bin height [m]
hyps_data = 'OGGM'      # Hypsometry dataset (OGGM; Maussion etal 2019)
                        # Other options to program are 'Huss' (GlacierMIP; Hock etal 2019) and Farinotti (Farinotti etal 2019) 

# Hypsometry data pre-processed by OGGM
if hyps_data == 'OGGM':
    oggm_gdir_fp = main_directory + '/../oggm_gdirs/'
    overwrite_gdirs = False

# Debris datasets
if include_debris:
    debris_fp = main_directory + '/../debris_data/'
    assert os.path.exists(debris_fp), 'Debris filepath does not exist. Turn off include_debris or add filepath.'
else:
    debris_fp = None


#%% MODEL TIME FRAME DATA
# Models require complete data for each year such that refreezing, scaling, etc. can be calculated
# Leap year option
option_leapyear = 0         # 1: include leap year days, 0: exclude leap years so February always has 28 days
# User specified start/end dates
#  note: start and end dates must refer to whole years
startmonthday = '06-01'     # Only used with custom calendars
endmonthday = '05-31'       # Only used with custom calendars
wateryear_month_start = 10  # water year starting month
winter_month_start = 10     # first month of winter (for HMA winter is October 1 - April 30)
summer_month_start = 5      # first month of summer (for HMA summer is May 1 - Sept 30)
option_dates = 1            # 1: use dates from date table (first of each month), 2: dates from climate data
timestep = 'monthly'        # time step ('monthly' only option at present)


#%% MODEL PROPERTIES
density_ice = 900           # Density of ice [kg m-3] (or Gt / 1000 km3)
density_water = 1000        # Density of water [kg m-3]
area_ocean = 362.5 * 1e12   # Area of ocean [m2] (Cogley, 2012 from Marzeion et al. 2020)
k_ice = 2.33                # Thermal conductivity of ice [J s-1 K-1 m-1] recall (W = J s-1)
k_air = 0.023               # Thermal conductivity of air [J s-1 K-1 m-1] (Mellor, 1997)
#k_air = 0.001               # Thermal conductivity of air [J s-1 K-1 m-1]
ch_ice = 1890000            # Volumetric heat capacity of ice [J K-1 m-3] (density=900, heat_capacity=2100 J K-1 kg-1)
ch_air = 1297               # Volumetric Heat capacity of air [J K-1 m-3] (density=1.29, heat_capacity=1005 J K-1 kg-1)
Lh_rf = 333550              # Latent heat of fusion [J kg-1]
tolerance = 1e-12           # Model tolerance (used to remove low values caused by rounding errors)
gravity = 9.81              # Gravity [m s-2]
pressure_std = 101325       # Standard pressure [Pa]
temp_std = 288.15           # Standard temperature [K]
R_gas = 8.3144598           # Universal gas constant [J mol-1 K-1]
molarmass_air = 0.0289644   # Molar mass of Earth's air [kg mol-1]


#%% DEBUGGING OPTIONS
debug_refreeze = False
debug_mb = False


# Pass variable to shell script
if __name__ == '__main__':
    reg_str = ''
    for region in rgi_regionsO1:
        reg_str += str(region)
    print(reg_str)
