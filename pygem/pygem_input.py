"""Model inputs to run PyGEM"""

# Built-in libraries
import os
# External libraries
import pandas as pd
import numpy as np


#%% Functions to select specific glacier numbers
def get_same_glaciers(glac_fp):
    """
    Get same 1000 glaciers for testing of priors

    Parameters
    ----------
    glac_fp : str
        filepath to where netcdf files of individual glaciers are held

    Returns
    -------
    glac_list : list
        list of rgi glacier numbers
    """
    glac_list = []
    for i in os.listdir(glac_fp):
        if i.endswith('.nc'):
            glac_list.append(i.split('.')[1])
    glac_list = sorted(glac_list)
    return glac_list


def get_shean_glacier_nos(region_no, number_glaciers=0, option_random=0):
    """
    Generate list of glaciers that have calibration data and select number of glaciers to include.

    The list is currently sorted in terms of area such that the largest glaciers are modeled first.

    Parameters
    ----------
    region_no : int
        region number (Shean data available for regions 13, 14, and 15)
    number_glaciers : int
        number of glaciers to include in model run (default = 0)
    option_random : int
        option to select glaciers randomly for model run (default = 0, not random)

    Returns
    -------
    num : list of strings
        list of rgi glacier numbers
    """
    # safety, convert input to int
    region_no = int(region_no)
    # get shean's data, convert to dataframe, get
    # glacier numbers
    current_directory = os.getcwd()
    csv_path = current_directory + '/../DEMs/Shean_2019_0213/hma_mb_20190215_0815_std+mean_all_filled_bolch.csv'
    ds_all = pd.read_csv(csv_path)
    ds_reg = ds_all[(ds_all['RGIId'] > region_no) & (ds_all['RGIId'] < region_no + 1)].copy()
    if option_random == 1:
        ds_reg = ds_reg.sample(n=number_glaciers)
    else:
        ds_reg = ds_reg.sort_values('area_m2', ascending=False)
    ds_reg.reset_index(drop=True, inplace=True)
    
    
    # Glacier number and index for comparison
    ds_reg['glacno'] = ((ds_reg['RGIId'] % 1) * 10**5).round(0).astype(int)
    ds_reg['glacno_str'] = (ds_reg['glacno'] / 10**5).apply(lambda x: '%.5f' % x).astype(str).str.split('.').str[1]
    num = list(ds_reg['glacno_str'].values)
    num = sorted(num)
    return num


def glac_num_fromrange(int_low, int_high):
    """
    Generate list of glaciers for all numbers between two integers.

    Parameters
    ----------
    int_low : int64
        low value of range
    int_high : int64
        high value of range

    Returns
    -------
    y : list
        list of rgi glacier numbers
    """
    x = (np.arange(int_low, int_high+1)).tolist()
    y = [str(i).zfill(5) for i in x]
    return y

def glac_fromcsv(csv_fullfn, cn='RGIId'):
    """
    Generate list of glaciers from csv file
    
    Parameters
    ----------
    csv_fp, csv_fn : str
        csv filepath and filename
    
    Returns
    -------
    y : list
        list of glacier numbers, e.g., ['14.00001', 15.00001']
    """
    df = pd.read_csv(csv_fullfn)
    return [x.split('-')[1] for x in df['RGIId'].values]


#%%
# Model setup directory
main_directory = os.getcwd()
# Output directory
output_filepath = main_directory + '/../Output/'
model_run_date = 'June 14, 2020'

# ===== GLACIER SELECTION =====
#rgi_regionsO1 = [13, 14, 15]            # 1st order region number (RGI V6.0)
rgi_regionsO1 = [1]            # 1st order region number (RGI V6.0)
rgi_regionsO2 = 'all'           # 2nd order region number (RGI V6.0)
# RGI glacier number (RGI V6.0)
#  Two options: (1) use glacier numbers for a given region (or 'all'), must have glac_no set to None
#               (2) glac_no is not None, e.g., ['1.00001', 13.0001'], overrides rgi_glac_number
rgi_glac_number = 'all'
#rgi_glac_number = ['00013']
#rgi_glac_number = glac_num_fromrange(1,5)
#rgi_glac_number = get_same_glaciers(output_filepath + 'cal_opt1/reg1/')
#rgi_glac_number = get_shean_glacier_nos(rgi_regionsO1[0], 1, option_random=1)
glac_no = ['15.03733']
#glac_no = ['13.42413']
#glac_no = ['1.00570','1.15645','11.00897','14.06794','15.03733','18.02342']
#glac_no = None
if glac_no is not None:
    rgi_regionsO1 = sorted(list(set([int(x.split('.')[0]) for x in glac_no])))

# ===== CLIMATE DATA ===== 
# Reference period runs
#ref_gcm_name = 'ERA-Interim'       # reference climate dataset
ref_gcm_name = 'ERA5'               # reference climate dataset
ref_startyear = 2000                # first year of model run (reference dataset)
ref_endyear = 2019                  # last year of model run (reference dataset)
ref_wateryear = 2                   # 1: water year, 2: calendar year, 3: custom defined 
ref_spinupyears = 0                 # spin up years
constantarea_years = 0              # number of years to not let the area or volume change
if ref_spinupyears > 0:
    assert 0==1, 'Code needs to be tested to enure spinup years are correctly accounted for in output files'
if constantarea_years > 0:
    print('\nConstant area years > 0\n')

# Simulation runs (separate so calibration and simulations can be run at same time; also needed for bias adjustments)
gcm_startyear = 2000            # first year of model run (simulation dataset)
gcm_endyear = 2019              # last year of model run (simulation dataset)
#gcm_startyear = 2000            # first year of model run (simulation dataset)
#gcm_endyear = 2100              # last year of model run (simulation dataset)
gcm_spinupyears = 0             # spin up years for simulation (output not set up for spinup years at present)
if gcm_spinupyears > 0:
    assert 0==1, 'Code needs to be tested to enure spinup years are correctly accounted for in output files'
gcm_wateryear = 2               # water year for simmulation

# Hindcast option (flips array so 1960-2000 would run 2000-1960 ensuring that glacier area at 2000 is correct)
hindcast = 0                    # 1: run hindcast simulation, 0: do not
if hindcast == 1:
    constantarea_years = 0     # number of years to not let the area or volume change
    gcm_startyear = 1980        # first year of model run (simulation dataset)
    gcm_endyear = 2000          # last year of model run (simulation dataset)

# Synthetic options (synthetic refers to created climate data, e.g., repeat 1995-2015 for the next 100 years)
option_synthetic_sim = 0        # 1: run synthetic simulation, 0: do not
if option_synthetic_sim == 1:
    synthetic_startyear = 1995      # synthetic start year
    synthetic_endyear = 2015        # synethetic end year
    synthetic_spinupyears = 0       # synthetic spinup years
    synthetic_temp_adjust = 3       # Temperature adjustment factor for synthetic runs
    synthetic_prec_factor = 1.12    # Precipitation adjustment factor for synthetic runs

#%% SIMULATION OPTIONS
# MCMC options
sim_iters = 100     # number of simulations (needed for cal_opt 2)
#sim_iters = 1     # number of simulations (needed for cal_opt 2)
#print('\n\nDELETE ME! - SWITCH SIM_ITERS BACK TO 100\n\n')
sim_burn = 200      # number of burn-in (needed for cal_opt 2)

# Simulation output filepath
output_sim_fp = output_filepath + 'simulations/'
# Simulation output statistics (can include 'mean', 'std', '2.5%', '25%', 'median', '75%', '97.5%')
sim_stat_cns = ['mean', 'std']
# Bias adjustment options (0: no adjustment, 1: new prec scheme and temp from HH2015, 2: HH2015 methods)
option_bias_adjustment = 1

#%% ===== CALIBRATION OPTIONS =====
# Calibration option ('MCMC', 'HH2015', 'HH2015mod')
#option_calibration = 'MCMC'
#option_calibration = 'HH2015'
option_calibration = 'HH2015mod'
#option_calibration = 'emulator'
# Calibration datasets ('shean', 'larsen', 'mcnabb', 'wgms_d', 'wgms_ee', 'group')
cal_datasets = ['shean']
#cal_datasets = ['shean']
# Calibration output filepath
output_fp_cal = output_filepath + 'cal_' + option_calibration + '/'

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
    tbias_step = 1
    kp_init = 1
    kp_bndlow = 0.5
    kp_bndhigh = 5
    ddfsnow_init = 0.0041
    # Minimization details 
    method_opt = 'SLSQP'            # SciPy optimization scheme ('SLSQP' or 'L-BFGS-B')
    params2opt = ['tbias', 'kp']
    ftol_opt = 1e-3                 # tolerance for SciPy optimization scheme
    eps_opt = 0.01                  # epsilon (adjust variables for jacobian) for SciPy optimization scheme (1e-6 works)
    
elif option_calibration == 'emulator':
    emulator_sims = 10000            # Number of simulations to develop the emulator
    tbias_step = 1                   # tbias step size
    tbias_init = 0                   # tbias initial value
    kp_init = 1                      # kp initial value
    ddfsnow_init = 0.0041            # ddfsnow initial value
    # Distributions
    tbias_disttype = 'truncnormal'   # Temperature bias distribution ('truncnormal', 'uniform')
    tbias_sigma = 3                  # tbias standard deviation for truncnormal distribution
    kp_gamma_alpha = 2               # Precipitation factor gamma distribution alpha
    kp_gamma_beta = 1                # Precipitation factor gamma distribution beta
    ddfsnow_disttype = 'truncnormal' # Degree-day factor of snow distribution ('truncnormal')
    ddfsnow_mu = 0.0041              # ddfsnow mean
    ddfsnow_sigma = 0.0015           # ddfsnow standard deviation
    ddfsnow_bndlow = 0               # ddfsnow lower bound
    ddfsnow_bndhigh = np.inf         # ddfsnow upper bound

elif option_calibration == 'MCMC':
    # Chain options
    n_chains = 1                    # number of chains (min 1, max 3)
    mcmc_sample_no = 10           # number of steps (10000 was found to be sufficient in HMA)
    mcmc_burn_no = 0                # number of steps to burn-in (0 records all steps in chain)
    mcmc_step = None                # step option (None or 'am')
    thin_interval = 1               # thin interval if need to reduce file size (best to leave at 1 if space allows)
    
    # Degree-day factor of snow distribution options
    ddfsnow_disttype = 'truncnormal' # distribution type ('truncnormal', 'uniform')
    ddfsnow_mu = 0.0041
    ddfsnow_sigma = 0.0015
    ddfsnow_bndlow = 0
    ddfsnow_bndhigh = np.inf
    ddfsnow_start=ddfsnow_mu
    
    # Precipitation factor distribution options
    kp_disttype = 'gamma'   # distribution type ('gamma', 'lognormal', 'uniform')
    kp_gamma_region_dict_fullfn = None
#    kp_gamma_region_dict_fullfn = main_directory + '/../Output/precfactor_gamma_region_dict.csv'
    if kp_gamma_region_dict_fullfn is not None:
        assert os.path.exists(kp_gamma_region_dict_fullfn), ('Using option_calibration: ' + option_calibration +
                             '.  Precfactor regional dictionary does not exist.')
        kp_gamma_region_df = pd.read_csv(kp_gamma_region_dict_fullfn)
        kp_gamma_region_dict = dict(zip(kp_gamma_region_df.Region.values, 
                                        [[kp_gamma_region_df.loc[x,'alpha'], kp_gamma_region_df.loc[x,'beta']] 
                                        for x in kp_gamma_region_df.index.values]))
    else:
        kp_gamma_alpha = 9
        kp_gamma_beta = 4
        kp_lognorm_mu = 0
        kp_lognorm_tau = 4
        kp_mu = 0
        kp_sigma = 1.5
        kp_bndlow = 0.5
        kp_bndhigh = 1.5
        kp_start = 1
        
    # Temperature bias distribution options
    tbias_disttype = 'normal'  # distribution type ('normal', 'truncnormal', 'uniform')
    tbias_norm_region_dict_fullfn = None
#    tbias_norm_region_dict_fullfn = main_directory + '/../Output/tempchange_norm_region_dict.csv'
    if tbias_norm_region_dict_fullfn is not None:
        assert os.path.exists(tbias_norm_region_dict_fullfn), ('Using option_calibration: ' + option_calibration +
                             '.  Tempbias regional dictionary does not exist.')
        tbias_norm_region_df = pd.read_csv(tbias_norm_region_dict_fullfn)
        tbias_norm_region_dict = dict(zip(tbias_norm_region_df.Region.values, 
                                          [[tbias_norm_region_df.loc[x,'mu'], tbias_norm_region_df.loc[x,'sigma']] 
                                          for x in tbias_norm_region_df.index.values]))
    else:
        tbias_mu = 0
        tbias_sigma = 1
        tbias_bndlow = -10
        tbias_bndhigh = 10
        tbias_start = tbias_mu


#%% ===== MODEL PARAMETERS =====
use_calibrated_modelparams = True   # False: use input values, True: use calibrated model parameters
#print('\nWARNING: using non-calibrated model parameters\n')
kp = 1                              # precipitation factor [-] (k_p in Radic etal 2013; c_prec in HH2015)
precgrad = 0.0001                   # precipitation gradient on glacier [m-1]
ddfsnow = 0.0041                    # degree-day factor of snow [m w.e. d-1 degC-1]
ddfsnow_iceratio = 0.7              # Ratio degree-day factor snow snow to ice
if ddfsnow_iceratio != 0.7:
    print('\n\n  Warning: ddfsnow_iceratio is', ddfsnow_iceratio, '\n\n')
ddfice = ddfsnow / ddfsnow_iceratio # degree-day factor of ice [m w.e. d-1 degC-1]
tbias = 0                           # temperature bias [deg C]
lrgcm = -0.0065                     # lapse rate from gcm to glacier [K m-1]
lrglac = -0.0065                    # lapse rate on glacier for bins [K m-1]
tsnow_threshold = 1.0               # temperature threshold for snow [deg C] (HH2015 used 1.5 degC +/- 1 degC)
frontalablation_k = 2               # frontal ablation rate [yr-1]
af = 0.7                            # Bulk flow parameter for frontal ablation (m^-0.5)
option_frontalablation_k = 1        # Calving option (1: values from HH2015, 
                                    #                 2: calibrate glaciers independently, use transfer fxns for others)
frontalablation_k0dict = {1:3.4,    # Calving dictionary with values from HH2015
                          2:0, 3:0.2, 4:0.2, 5:0.5, 6:0.3, 7:0.5, 8:0, 9:0.2, 10:0, 11:0, 12:0, 13:0, 14:0, 15:0, 16:0,
                          17:6, 18:0, 19:1}

# Calving width dictionary to override RGI elevation bins, which can be highly inaccurate at the calving front
width_calving_dict_fullfn = main_directory + '/../Calving_data/calvingfront_widths.csv'
if os.path.exists(width_calving_dict_fullfn):
    width_calving_df = pd.read_csv(width_calving_dict_fullfn)
    width_calving_dict = dict(zip(width_calving_df.RGIId, width_calving_df.front_width_m))
else:
    width_calving_dict = {}

## Calving parameter dictionary (according to Supplementary Table 3 in HH2015)
#frontalablation_k0dict_fullfn = main_directory + '/../Calving_data/frontalablation_k0_dict.csv'
#if os.path.exists(frontalablation_k0dict_fullfn):
#    frontalablation_k0dict_df = pd.read_csv(frontalablation_k0dict_fullfn)
#    frontalablation_k0dict = dict(zip(frontalablation_k0dict_df.O1Region, frontalablation_k0dict_df.k0))
#else:
#    frontalablation_k0dict = None
    
# Calving glacier data
calving_data_fullfn = main_directory + '/../Calving_data/calving_glacier_data.csv'

# Model parameter column names and filepaths
modelparams_colnames = ['lrgcm', 'lrglac', 'precfactor', 'precgrad', 'ddfsnow', 'ddfice', 'tempsnow', 'tempchange']
# Model parameter filepath
modelparams_fp = output_filepath + 'cal_' + option_calibration + '/'
#modelparams_fp = output_filepath + 'cal_opt2_spc_20190806/'

#%% ===== MASS BALANCE MODEL OPTIONS =====
# Initial surface type options
option_surfacetype_initial = 1
#  option 1 (default) - use median elevation to classify snow/firn above the median and ice below.
#   > Sakai et al. (2015) found that the decadal ELAs are consistent with the median elevation of nine glaciers in High
#     Mountain Asia, and Nuimura et al. (2015) also found that the snow line altitude of glaciers in China corresponded
#     well with the median elevation.  Therefore, the use of the median elevation for defining the initial surface type
#     appears to be a fairly reasonable assumption in High Mountain Asia.
#  option 2 - use mean elevation
#  option 3 (Need to code) - specify an AAR ratio and apply this to estimate initial conditions
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
option_ablation = 2                 # 1: monthly temp, 2: superimposed daily temps enabling melt near 0 (HH2015)
option_ddf_firn = 1                 # 0: ddf_firn = ddf_snow; 1: ddf_firn = mean of ddf_snow and ddf_ice
ddfdebris = ddfice                  # add options for handling debris-covered glaciers

# Refreezing model options
#option_refreezing = 'HH2015'        # HH2015: heat conduction (Huss and Hock, 2015)
option_refreezing = 'Woodward'      # Woodward: annual air temp (Woodward etal 1997)
if option_refreezing == 'HH2015':
    rf_layers = 5                   # number of layers for refreezing model (8 is sufficient - Matthias)
#    rf_layers_max = 8               # number of layers to include for refreeze calculation
    rf_dz = 10/rf_layers            # layer thickness (m)
    rf_dsc = 3                      # number of time steps for numerical stability (3 is sufficient - Matthias)
    rf_meltcrit = 0.002             # critical amount of melt [m w.e.] for initializing refreezing module
    pp = 0.3                        # additional refreeze water to account for water refreezing at bare-ice surface
    rf_dens_top = 300               # snow density at surface (kg m-3)
    rf_dens_bot = 650               # snow density at bottom refreezing layer (kg m-3)
    option_rf_limit_meltsnow = 1
    
elif option_refreezing == 'Woodward':
    rf_month = 10                   # refreeze month

# Mass redistribution / Glacier geometry change options
option_massredistribution = 1       # 1: mass redistribution (Huss and Hock, 2015)
option_glaciershape = 1             # 1: parabolic (Huss and Hock, 2015), 2: rectangular, 3: triangular
option_glaciershape_width = 1       # 1: include width, 0: do not include
icethickness_advancethreshold = 5   # advancing glacier ice thickness change threshold (5 m in Huss and Hock, 2015)
terminus_percentage = 20            # glacier (%) considered terminus (20% in HH2015), used to size advancing new bins
    
#%% CLIMATE DATA
# ERA-INTERIM (Reference data)
# Variable names
era_varnames = ['temperature', 'precipitation', 'geopotential', 'temperature_pressurelevels']
#  Note: do not change variable names as these are set to run with the download_erainterim_data.py script.
#        If option 2 is being used to calculate the lapse rates, then the pressure level data is unnecessary.
# Dates
eraint_start_date = '19790101'
eraint_end_date = '20180501'
# Resolution
grid_res = '0.5/0.5'
# Bounding box (N/W/S/E)
#bounding_box = '90/0/-90/360'
bounding_box = '50/70/25/105'
# Lapse rate option
#  option 0 - lapse rates are constant defined by input
#  option 1 (default) - lapse rates derived from gcm pressure level temperature data (varies spatially and temporally)
#  option 2 - lapse rates derived from surrounding pixels (varies spatially and temporally)
#    Note: Be careful with option 2 as the ocean vs land/glacier temperatures can cause unrealistic inversions
#          This is the option used by Marzeion et al. (2012)
option_lr_method = 1

# ERA5
if ref_gcm_name == 'ERA5':
    era5_fp = main_directory + '/../Climate_data/ERA5/'
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
    assert os.path.exists(era5_fp + era5_lr_fn), 'ERA5 lapse rate data does not exist'
    if option_ablation == 2:
        assert os.path.exists(era5_fp + era5_tempstd_fn), 'ERA5 temperature std filepath does not exist'

# ERA-Interim
elif ref_gcm_name == 'ERA-Interim':
    eraint_fp = main_directory + '/../Climate_data/ERA_Interim/download/'
    eraint_temp_fn = 'ERAInterim_Temp2m_DailyMeanMonthly_' + eraint_start_date + '_' + eraint_end_date + '.nc'
    eraint_prec_fn = 'ERAInterim_TotalPrec_DailyMeanMonthly_' + eraint_start_date + '_' + eraint_end_date + '.nc'
    eraint_elev_fn = 'ERAInterim_geopotential.nc'
    eraint_pressureleveltemp_fn = 'ERAInterim_pressureleveltemp_' + eraint_start_date + '_' + eraint_end_date + '.nc'
    eraint_lr_fn = ('ERAInterim_lapserates_' + eraint_start_date + '_' + eraint_end_date + '_opt' + 
                    str(option_lr_method) + '_world.nc')
    assert os.path.exists(eraint_fp), 'ERA-Interim filepath does not exist'
    assert os.path.exists(eraint_temp_fn), 'ERA-Interim temperature filepath does not exist'
    assert os.path.exists(eraint_prec_fn), 'ERA-Interim precipitation filepath does not exist'
    assert os.path.exists(eraint_elev_fn), 'ERA-Interim elevation data does not exist'
    assert os.path.exists(eraint_lr_fn), 'ERA-Interim lapse rate data does not exist'
    if option_ablation == 2:
        assert 0==1, 'ERA-Interim not set up to use option_ablation 2 (temperature std data not downloaded)'

# CMIP5 (GCM data)
cmip5_fp_var_prefix = main_directory + '/../Climate_data/cmip5/'
cmip5_fp_var_ending = '_r1i1p1_monNG/'
cmip5_fp_fx_prefix = main_directory + '/../Climate_data/cmip5/'
cmip5_fp_fx_ending = '_r0i0p0_fx/'
cmip5_fp_lr = main_directory + '/../Climate_data/cmip5/bias_adjusted_1995_2100/2018_0524/'
cmip5_lr_fn = 'biasadj_mon_lravg_1995_2015_R15.csv'

# COAWST (High-resolution climate data over HMA)
coawst_fp_unmerged = main_directory + '/../Climate_data/coawst/Monthly/'
coawst_fp = main_directory + '/../Climate_data/coawst/'
coawst_fn_prefix_d02 = 'wrfout_d02_Monthly_'
coawst_fn_prefix_d01 = 'wrfout_d01_Monthly_'
coawst_temp_fn_d02 = 'wrfout_d02_Monthly_T2_1999100100-2006123123.nc'
coawst_prec_fn_d02 = 'wrfout_d02_Monthly_TOTPRECIP_1999100100-2006123123.nc'
coawst_elev_fn_d02 = 'wrfout_d02_Monthly_HGHT.nc'
coawst_temp_fn_d01 = 'wrfout_d01_Monthly_T2_1999100100-2006123123.nc'
coawst_prec_fn_d01 = 'wrfout_d01_Monthly_TOTPRECIP_1999100100-2006123123.nc'
coawst_elev_fn_d01 = 'wrfout_d01_Monthly_HGHT.nc'
coawst_vns = ['T2', 'TOTPRECIP', 'HGHT']
coawst_d02_lon_min = 65
coawst_d02_lon_max = 99
coawst_d02_lat_min = 20
coawst_d02_lat_max = 38

#%% GLACIER DATA (RGI, ICE THICKNESS, ETC.)
# ===== RGI DATA =====
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
# Column names from table to drop
rgi_cols_drop = ['GLIMSId','BgnDate','EndDate','Status','Connect','Linkages','Name']
#rgi_cols_drop = []

# ===== ADDITIONAL DATA (hypsometry, ice thickness, width) =====
h_consensus_fp = main_directory + '/../IceThickness_Farinotti/composite_thickness_RGI60-all_regions/'
# Filepath for the hypsometry files
binsize = 10            # Elevation bin height [m]
#hyps_data = 'Huss'      # Hypsometry dataset (GlacierMIP; Hock etal 2019)
#hyps_data = 'Farinotti' # Hyspsometry dataset (Farinotti etal 2019)
hyps_data = 'OGGM'       # Hypsometry dataset (OGGM; Maussion etal 2019)

# Data from Farinotti et al. (2019): Consensus ice thickness estimates
if hyps_data == 'Farinotti':
    option_shift_elevbins_20m = 0   # option to shift bins by 20 m (needed since off by 20 m, seem email 5/24/2018)
    # Dictionary of hypsometry filenames
    hyps_filepath = main_directory + '/../IceThickness_Farinotti/output/'
    hyps_filedict = {1:  'area_km2_01_Farinotti2019_10m.csv',
                     13: 'area_km2_13_Farinotti2019_10m.csv',
                     14: 'area_km2_14_Farinotti2019_10m.csv',
                     15: 'area_km2_15_Farinotti2019_10m.csv'}
    hyps_colsdrop = ['RGIId']
    # Thickness data
    thickness_filepath = main_directory + '/../IceThickness_Farinotti/output/'
    thickness_filedict = {1:  'thickness_m_01_Farinotti2019_10m.csv',
                          13: 'thickness_m_13_Farinotti2019_10m.csv',
                          14: 'thickness_m_14_Farinotti2019_10m.csv',
                          15: 'thickness_m_15_Farinotti2019_10m.csv'}
    thickness_colsdrop = ['RGIId']
    # Width data
    width_filepath = main_directory + '/../IceThickness_Farinotti/output/'
    width_filedict = {1:  'width_km_01_Farinotti2019_10m.csv',
                      13: 'width_km_13_Farinotti2019_10m.csv',
                      14: 'width_km_14_Farinotti2019_10m.csv',
                      15: 'width_km_15_Farinotti2019_10m.csv'}
    width_colsdrop = ['RGIId']
# Data from GlacierMIP
elif hyps_data == 'Huss':
    option_shift_elevbins_20m = 1   # option to shift bins by 20 m (needed since off by 20 m, seem email 5/24/2018)
    # Dictionary of hypsometry filenames
    # (Files from Matthias Huss should be manually pre-processed to be 'RGI-ID', 'Cont_range', and bins starting at 5)
    hyps_filepath = main_directory + '/../IceThickness_Huss/bands_10m_DRR/'
    hyps_filedict = {
                    1:  'area_01_Huss_Alaska_10m.csv',
                    3:  'area_RGI03_10.csv',
                    4:  'area_RGI04_10.csv',
                    6:  'area_RGI06_10.csv',
                    7:  'area_RGI07_10.csv',
                    8:  'area_RGI08_10.csv',
                    9:  'area_RGI09_10.csv',
                    13: 'area_13_Huss_CentralAsia_10m.csv',
                    14: 'area_14_Huss_SouthAsiaWest_10m.csv',
                    15: 'area_15_Huss_SouthAsiaEast_10m.csv',
                    16: 'area_16_Huss_LowLatitudes_10m.csv',
                    17: 'area_17_Huss_SouthernAndes_10m.csv'}
    hyps_colsdrop = ['RGI-ID','Cont_range']
    # Thickness data
    thickness_filepath = main_directory + '/../IceThickness_Huss/bands_10m_DRR/'
    thickness_filedict = {
                    1:  'thickness_01_Huss_Alaska_10m.csv',
                    3:  'thickness_RGI03_10.csv',
                    4:  'thickness_RGI04_10.csv',
                    6:  'thickness_RGI06_10.csv',
                    7:  'thickness_RGI07_10.csv',
                    8:  'thickness_RGI08_10.csv',
                    9:  'thickness_RGI09_10.csv',
                    13: 'thickness_13_Huss_CentralAsia_10m.csv',
                    14: 'thickness_14_Huss_SouthAsiaWest_10m.csv',
                    15: 'thickness_15_Huss_SouthAsiaEast_10m.csv',
                    16: 'thickness_16_Huss_LowLatitudes_10m.csv',
                    17: 'thickness_17_Huss_SouthernAndes_10m.csv'}
    thickness_colsdrop = ['RGI-ID','Cont_range']
    # Width data
    width_filepath = main_directory + '/../IceThickness_Huss/bands_10m_DRR/'
    width_filedict = {
                    1:  'width_01_Huss_Alaska_10m.csv',
                    3:  'width_RGI03_10.csv',
                    4:  'width_RGI04_10.csv',
                    6:  'width_RGI06_10.csv',
                    7:  'width_RGI07_10.csv',
                    8:  'width_RGI08_10.csv',
                    9:  'width_RGI09_10.csv',
                    13: 'width_13_Huss_CentralAsia_10m.csv',
                    14: 'width_14_Huss_SouthAsiaWest_10m.csv',
                    15: 'width_15_Huss_SouthAsiaEast_10m.csv',
                    16: 'width_16_Huss_LowLatitudes_10m.csv',
                    17: 'width_17_Huss_SouthernAndes_10m.csv'}
    width_colsdrop = ['RGI-ID','Cont_range']
elif hyps_data == 'OGGM':
    oggm_gdir_fp = main_directory + '/../oggm_gdirs/'
    overwrite_gdirs = False
    
# Glacier dynamics options ('OGGM', 'MassRedistributionCurves', None??)
#option_dynamics = 'OGGM'
option_dynamics = 'MassRedistributionCurves'
    
# Debris datasets
if include_debris:
    debris_fp = main_directory + '/../debris_data/'
    assert os.path.exists(debris_fp), 'Debris filepath does not exist. Turn off include_debris or add filepath.'
else:
    debris_fp = None


#%% MODEL TIME FRAME DATA
# Models require complete data for each year such that refreezing, scaling, etc. can be calculated
# Leap year option
option_leapyear = 1         # 1: include leap year days, 0: exclude leap years so February always has 28 days
# User specified start/end dates
#  note: start and end dates must refer to whole years
startmonthday = '06-01'
endmonthday = '05-31'
wateryear_month_start = 10  # water year starting month
winter_month_start = 10     # first month of winter (for HMA winter is October 1 - April 30)
summer_month_start = 5      # first month of summer (for HMA summer is May 1 - Sept 30)
option_dates = 1            # 1: use dates from date table (first of each month), 2: dates from climate data
timestep = 'monthly'        # time step ('monthly' only option at present)

# Seasonal dictionaries for WGMS data that is not provided
lat_threshold = 75
# Winter (start/end) and Summer (start/end)
monthdict = {'northernmost': [9, 5, 6, 8],
             'north': [10, 4, 5, 9],
             'south': [4, 9, 10, 3],
             'southernmost': [3, 10, 11, 2]}
# Latitude threshold
# 01 - Alaska - < 75
# 02 - W Can - < 75
# 03 - N Can - > 74
# 04 - S Can - < 74
# 05 - Greenland - 60 - 80
# 06 - Iceland - < 75
# 07 - Svalbard - 70 - 80
# 08 - Scandinavia - < 70
# 09 - Russia - 72 - 82
# 10 - N Asia - 46 - 77


#%% CALIBRATION DATASETS
mb_binned_fp = main_directory + '/../DEMs/mb_bins_all-20200430/'

# ===== HUGONNET GEODETIC =====
hugonnet_fp = main_directory + '/../DEMs/Hugonnet2020/'
hugonnet_fn = 'df_pergla_global_20yr.csv'
hugonnet_rgi_glacno_cn = 'rgiid'
hugonnet_mb_cn = 'dmdtda'
hugonnet_mb_err_cn = 'err_dmdtda'
hugonnet_time1_cn = 't1'
hugonnet_time2_cn = 't2'
hugonnet_area_cn = 'area_km2'

# ===== SHEAN GEODETIC =====
shean_fp = main_directory + '/../DEMs/Shean_2019_0213/'
shean_fn = 'hma_mb_20190215_0815_std+mean_all_filled_bolch.csv'
shean_rgi_glacno_cn = 'RGIId'
shean_mb_cn = 'mb_mwea'
shean_mb_err_cn = 'mb_mwea_sigma'
shean_time1_cn = 't1'
shean_time2_cn = 't2'
shean_area_cn = 'area_m2'

# ===== BERTHIER GEODETIC =====
berthier_fp = main_directory + '/../DEMs/Berthier/output/'
#berthier_fn = 'AK_all_20190913_wextrapolations_1980cheat.csv'
berthier_fn = 'AK_all_20190913.csv'
berthier_rgi_glacno_cn = 'RGIId'
berthier_mb_cn = 'mb_mwea'
berthier_mb_err_cn = 'mb_mwea_sigma'
berthier_time1_cn = 't1'
berthier_time2_cn = 't2'
berthier_area_cn = 'area_km2'

# ===== BRAUN GEODETIC =====
braun_fp = main_directory + '/../DEMs/Braun/output/'
braun_fn = 'braun_AK_all_20190924_wlarsen_mcnabb_best.csv'
#braun_fn = 'braun_AK_all_20190924_wextrapolations.csv'
#braun_fn = 'braun_AK_all_20190924.csv'
braun_rgi_glacno_cn = 'RGIId'
braun_mb_cn = 'mb_mwea'
braun_mb_err_cn = 'mb_mwea_sigma'
braun_time1_cn = 't1'
braun_time2_cn = 't2'
braun_area_cn = 'area_km2'

# ===== BRUN GEODETIC =====
brun_fp = main_directory + '/../DEMs/'
brun_fn = 'Brun_Nature2017_MB_glacier-wide.csv'
brun_rgi_glacno_cn = 'GLA_ID'
brun_mb_cn = 'MB [m w.a a-1]'
brun_mb_err_cn = 'err. on MB [m w.e a-1]'
# NEED TO FINISH SETTING UP BRUN WITH CLASS_MBDATA

# ===== MAUER GEODETIC =====
mauer_fp = main_directory + '/../DEMs/'
mauer_fn = 'Mauer_geoMB_HMA_1970s_2000_min80pctCov.csv'
mauer_rgi_glacno_cn = 'RGIId'
mauer_mb_cn = 'geoMassBal'
mauer_mb_err_cn = 'geoMassBalSig'
mauer_time1_cn = 't1'
mauer_time2_cn = 't2'

# ===== MCNABB GEODETIC =====
mcnabb_fp = main_directory + '/../DEMs/McNabb_data/wgms_dv/'
mcnabb_fn = 'McNabb_data_all_preprocessed.csv'
mcnabb_rgiid_cn = 'RGIId'
mcnabb_mb_cn = 'mb_mwea'
mcnabb_mb_err_cn = 'mb_mwea_sigma'
mcnabb_time1_cn = 'date0'
mcnabb_time2_cn = 'date1'
mcnabb_area_cn = 'area'

# ===== LARSEN GEODETIC =====
larsen_fp = main_directory + '/../DEMs/larsen/'
larsen_fn = 'larsen2015_supplementdata_wRGIIds_v3.csv'
larsen_rgiid_cn = 'RGIId'
larsen_mb_cn = 'mb_mwea'
larsen_mb_err_cn = 'mb_mwea_sigma'
larsen_time1_cn = 'date0'
larsen_time2_cn = 'date1'
larsen_area_cn = 'area'

# ===== WGMS =====
wgms_datasets = ['wgms_d', 'wgms_ee']
#wgms_datasets = ['wgms_d']
wgms_fp = main_directory + '/../WGMS/DOI-WGMS-FoG-2018-06/'
wgms_rgi_glacno_cn = 'glacno'
wgms_obs_type_cn = 'obs_type'
# WGMS lookup tables information
wgms_lookup_fn = 'WGMS-FoG-2018-06-AA-GLACIER-ID-LUT.csv'
rgilookup_fullfn = main_directory + '/../RGI/rgi60/00_rgi60_links/00_rgi60_links.csv'
rgiv6_fn_prefix = main_directory + '/../RGI/rgi60/00_rgi60_attribs/' + '*'
rgiv5_fn_prefix = main_directory + '/../RGI/00_rgi50_attribs/' + '*'

# WGMS (d) geodetic mass balance information
wgms_d_fn = 'WGMS-FoG-2018-06-D-CHANGE.csv'
wgms_d_fn_preprocessed = 'wgms_d_rgiv6_preprocessed.csv'
wgms_d_thickness_chg_cn = 'THICKNESS_CHG'
wgms_d_thickness_chg_err_cn = 'THICKNESS_CHG_UNC'
wgms_d_volume_chg_cn = 'VOLUME_CHANGE'
wgms_d_volume_chg_err_cn = 'VOLUME_CHANGE_UNC'
wgms_d_z1_cn = 'LOWER_BOUND'
wgms_d_z2_cn = 'UPPER_BOUND'

# WGMS (e/ee) glaciological mass balance information
wgms_e_fn = 'WGMS-FoG-2018-06-E-MASS-BALANCE-OVERVIEW.csv'
wgms_ee_fn = 'WGMS-FoG-2018-06-EE-MASS-BALANCE.csv'
wgms_ee_fn_preprocessed = 'wgms_ee_rgiv6_preprocessed.csv'
wgms_ee_mb_cn = 'BALANCE'
wgms_ee_mb_err_cn = 'BALANCE_UNC'
wgms_ee_t1_cn = 'YEAR'
wgms_ee_z1_cn = 'LOWER_BOUND'
wgms_ee_z2_cn = 'UPPER_BOUND'
wgms_ee_period_cn = 'period'

# ===== COGLEY DATA =====
cogley_fp = main_directory + '/../Calibration_datasets/'
cogley_fn_preprocessed = 'Cogley_Arctic_processed_wInfo.csv'
cogley_rgi_glacno_cn = 'glacno'
cogley_mass_chg_cn = 'geo_mass_kgm2a'
cogley_mass_chg_err_cn = 'geo_mass_unc'
cogley_z1_cn = 'Zmin'
cogley_z2_cn = 'Zmax'
cogley_obs_type_cn = 'obs_type'

# ===== REGIONAL DATA =====
# Regional data refers to all measurements that have lumped multiple glaciers together
#  - a dictionary linking the regions to RGIIds is required
mb_group_fp = main_directory + '/../Calibration_datasets/'
mb_group_dict_fn = 'mb_group_dict.csv'
mb_group_data_fn = 'mb_group_data.csv'
mb_group_t1_cn = 'begin_period'
mb_group_t2_cn = 'end_period'

#%% REGIONS
grouping = None
#grouping = 'himap'
if grouping == 'watershed':
    reg_vn = 'watershed'
    reg_dict_fn = main_directory + '/../qgis_himat/rgi60_HMA_dict_watershed.csv'
    reg_csv = pd.read_csv(reg_dict_fn)
    reg_dict = dict(zip(reg_csv.RGIId, reg_csv[reg_vn]))
elif grouping == 'kaab':
    reg_vn = 'kaab_name'
    reg_dict_fn = main_directory + '/../qgis_himat/rgi60_HMA_dict_kaab.csv'
    reg_csv = pd.read_csv(reg_dict_fn)
    reg_dict = dict(zip(reg_csv.RGIId, reg_csv[reg_vn]))
elif grouping == 'himap':
    reg_vn = 'bolch_name'
    reg_dict_fn = main_directory + '/../qgis_himat/rgi60_HMA_dict_bolch.csv'
    reg_csv = pd.read_csv(reg_dict_fn)
    reg_dict = dict(zip(reg_csv.RGIId, reg_csv[reg_vn]))
else:
    reg_dict = {}


#%% OUTPUT OPTIONS
# Output package
#  option 0 - no netcdf package
#  option 1 - "raw package" [preferred units: m w.e.]
#              monthly variables for each bin (temp, prec, acc, refreeze, snowpack, melt, frontalablation,
#                                              massbal_clim)
#              annual variables for each bin (area, icethickness, surfacetype)
#  option 2 - "Glaciologist Package" output [units: m w.e. unless otherwise specified]:
#              monthly glacier-wide variables (prec, acc, refreeze, melt, frontalablation, massbal_total, runoff,
#                                              snowline)
#              annual glacier-wide variables (area, volume, ELA)
output_package = 2
output_glacier_attr_vns = ['glacno', 'RGIId_float', 'CenLon', 'CenLat', 'O1Region', 'O2Region', 'Area', 'Zmin', 'Zmax',
                           'Zmed', 'Slope', 'Aspect', 'Lmax', 'Form', 'TermType', 'Surging']
time_names = ['time', 'year', 'year_plus1']
# Output package variables
output_variables_package2 = ['temp_glac_monthly', 'prec_glac_monthly', 'acc_glac_monthly',
                            'refreeze_glac_monthly', 'melt_glac_monthly', 'frontalablation_glac_monthly',
                            'massbaltotal_glac_monthly', 'runoff_glac_monthly', 'snowline_glac_monthly',
                            'area_glac_annual', 'volume_glac_annual', 'ELA_glac_annual',
                            'offglac_prec_monthly', 'offglac_refreeze_monthly', 'offglac_melt_monthly',
                            'offglac_snowpack_monthly', 'offglac_runoff_monthly']

#%% MODEL PROPERTIES
density_ice = 900           # Density of ice [kg m-3] (or Gt / 1000 km3)
density_water = 1000        # Density of water [kg m-3]
area_ocean = 362.5 * 10**6  # Area of ocean [km2]
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
#    print(rgi_glac_number[0:10])
