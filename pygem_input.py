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
    csv_path = current_directory + '/../DEMs/Shean_2019_0213/hma_mb_20190215_0815_std+mean.csv'
    ds_all = pd.read_csv(csv_path)
    ds_reg = ds_all[(ds_all['RGIId'] > region_no) & (ds_all['RGIId'] < region_no + 1)].copy()
    if option_random == 1:
        ds_reg = ds_reg.sample(n=number_glaciers)
        ds_reg.reset_index(drop=True, inplace=True)
    else:
        ds_reg = ds_reg.sort_values('area_m2', ascending=False)
        ds_reg.reset_index(drop=True, inplace=True)
    rgi = ds_reg['RGIId']
    # get only glacier numbers, convert to string
    num = rgi % 1
    num = num.round(5)
    num = num.astype(str)
    # slice string to remove decimal
    num = [n[2:] for n in num]
    # make sure there are 5 digits
    for i in range(len(num)):
        while len(num[i]) < 5:
            num[i] += '0'
    if number_glaciers > 0:
        num = num[0:number_glaciers]
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


#%%
# Model setup directory
main_directory = os.getcwd()
# Output directory
output_filepath = main_directory + '/../Output/'

# ===== GLACIER SELECTION =====
# Region number 1st order (RGI V6.0) - HMA is 13, 14, 15
rgi_regionsO1 = [13]
# 2nd order region numbers (RGI V6.0)
rgi_regionsO2 = 'all'
# RGI glacier number (RGI V6.0)
#rgi_glac_number = 'all'
rgi_glac_number = ['45048']
#rgi_glac_number = glac_num_fromrange(25419,25424)
#rgi_glac_number = get_same_glaciers(output_filepath + 'cal_opt1/reg1/')
#rgi_glac_number = get_shean_glacier_nos(rgi_regionsO1[0], 2, option_random=1)

# ===== Bias adjustment option =====
option_bias_adjustment = 1
#  Option 0 - ignore bias adjustments
#  Option 1 - new precipitation, temperature from Huss and Hock [2015]
#  Option 2 - Huss and Hock [2015] methods

# Reference climate dataset
ref_gcm_name = 'ERA-Interim' # used as default for argument parsers

# First year of model run (change these to calibration)
#startyear = 1980
startyear = 2000
#  water year example: 2000 would start on October 1999, since October 1999 - September 2000 is the water year 2000
#  calendar year example: 2000 would start on January 2000
# Last year of model run
#endyear = 2017
endyear = 2018
# Spin up time [years]
spinupyears = 0
# Water year option
#option_wateryear = 1
option_wateryear = 3
#  Option 1 (default) - water year (ex. 2000: Oct 1 1999 - Sept 30 2000)
#  Option 2 - calendar year
#  Option 3 - define start/end months and days (BE CAREFUL WHEN CUSTOMIZING USING OPTION 3 - DOUBLE CHECK YOUR DATES)
#constantarea_years = 37
constantarea_years = 0

# Simulation runs
#  simulation runs are separate such that calibration runs can be run at same time as simulations
gcm_startyear = 1980
gcm_endyear = 2017
#gcm_startyear = 2000
#gcm_endyear = 2100
gcm_spinupyears = 0
gcm_wateryear = 1

# Hindcast flips the array such that 1960 - 2000 would go from 2000-1960 ensuring that the glacier area at 2000 is 
# what it's supposed to be.
hindcast = 0
if hindcast == 1:
    constantarea_years = 18 # constant years so glacier doesn't evolve until before 2000
    gcm_startyear = 1960
    gcm_endyear = 2017

# Synthetic simulation options
#  synthetic simulations refer to climate data that is created (ex. repeat 1990-2000 for the next 100 years) 
option_synthetic_sim = 0
synthetic_startyear = 1995
synthetic_endyear = 2015
synthetic_spinupyears = 0
synthetic_temp_adjust = 3
synthetic_prec_factor = 1.12

#%% ===== CALIBRATION OPTIONS =====
# Calibration option (1 = minimization, 2 = MCMC, 3=HH2015, 4=modified HH2015)
option_calibration = 2
# Calibration datasets
cal_datasets = ['shean']
#cal_datasets = ['mcnabb']
#cal_datasets = ['larsen']
#cal_datasets = ['mcnabb', 'larsen']
#cal_datasets = ['wgms_d', 'group']
#cal_datasets = ['shean', 'wgms_d', 'wgms_ee', 'group']
# Calibration output filepath (currently only for option 1)
output_fp_cal = output_filepath + 'cal_opt' + str(option_calibration) + '/'

# OPTION 1: Minimization
# Model parameter bounds for each calibration round
#  first tuple will run as expected; 
#precfactor_bnds_list_init = [(0.9, 1.125), (0.8,1.25), (0.5,2), (0.33,3)]
#precgrad_bnds_list_init = [(0.0001,0.0001), (0.0001,0.0001), (0.0001,0.0001), (0.0001,0.0001)]
#ddfsnow_bnds_list_init = [(0.0036, 0.0046), (0.0036, 0.0046), (0.0026, 0.0056), (0.00185, 0.00635)]
#tempchange_bnds_list_init = [(-1,1), (-2,2), (-5,5), (-10,10)]
precfactor_bnds_list_init = [(0.8, 2.0), (0.8,2), (0.8,2), (0.2,5)]
precgrad_bnds_list_init = [(0.0001,0.0001), (0.0001,0.0001), (0.0001,0.0001), (0.0001,0.0001)]
ddfsnow_bnds_list_init = [(0.003, 0.003), (0.00175, 0.0045), (0.00175, 0.0045), (0.00175, 0.0045)]
tempchange_bnds_list_init = [(0,0), (0,0), (-2.5,2.5), (-10,10)]
# Threshold to update the model parameters (based on the difference in zscores)
zscore_update_threshold = 0.1
# Additional calibration rounds in case optimization is getting stuck
extra_calrounds = 3

# OPTION 2: MCMC 
# Chain options 
n_chains = 1 # (min 1, max 3)
mcmc_sample_no = 500
mcmc_burn_no = 0
ensemble_no = mcmc_sample_no - mcmc_burn_no
mcmc_step = None
#mcmc_step = 'am'
thin_interval = 1

# MCMC distribution parameters
#precfactor_disttype = 'lognormal'
#precfactor_disttype = 'uniform'
precfactor_disttype = 'gamma'
precfactor_gamma_region_dict = {'Karakoram': [2.53, 1.69],
                                'Western Kunlun Shan': [2.41, 1.88],
                                'Nyainqentanglha': [9.28, 3.01],
                                'Eastern Himalaya': [3.82, 1.95],
                                'Central Himalaya': [3.11, 1.54],
                                'Western Himalaya': [2.95, 1.92],
                                'Tibetan Interior Mountains': [3.64, 1.58],
                                'Dzhungarsky Alatau': [4.62, 1.88],
                                'Central Tien Shan': [3.16, 1.50],
                                'Northern/Western Tien Shan': [3.88, 1.80],
                                'Western Pamir': [2.6, 1.78],
                                'Pamir Alay': [3.99, 2.25],
                                'Eastern Pamir': [1.8, 1.64],
                                'Eastern Tibetan Mountains': [6.76, 2.76],
                                'Qilian Shan': [4.74, 1.69],
                                'Tanggula Shan': [11.18, 3.53],
                                'Eastern Kunlun Shan': [3.71, 1.54],
                                'Hengduan Shan': [7.17, 2.48],
                                'Gangdise Mountains': [5.89, 2.12],
                                'Eastern Tien Shan': [3.0, 0.84],
                                'Altun Shan': [13.13, 2.67],
                                'Eastern Hindu Kush': [3.9, 2.39]
                                }
precfactor_gamma_alpha = 3.0
precfactor_gamma_beta = 0.84
precfactor_lognorm_mu = 0
precfactor_lognorm_tau = 4
precfactor_mu = 0
precfactor_sigma = 1.5
#precfactor_boundlow = 0
#precfactor_boundhigh = 2
precfactor_boundlow = 0.5
precfactor_boundhigh = 1.5
precfactor_start = 1
precfactor_step = 0.1
precfactor_boundhigh_adj = 0
#tempchange_disttype = 'normal'
tempchange_disttype = 'truncnormal'
#tempchange_disttype = 'uniform'
tempchange_norm_region_dict = {'Karakoram': [2.43, 1.95],
                               'Western Kunlun Shan': [3.39, 1.79],
                               'Nyainqentanglha':[-0.37, 0.97],
                               'Eastern Himalaya': [0.08, 1.02],
                               'Central Himalaya': [0.31, 1.03],
                               'Western Himalaya': [0.4, 1.08],
                               'Tibetan Interior Mountains': [0.61, 0.95],
                               'Dzhungarsky Alatau': [0.48, 0.80],
                               'Central Tien Shan': [1.08, 1.25],
                               'Northern/Western Tien Shan': [0.52, 0.99],
                               'Western Pamir': [0.91, 1.4],
                               'Pamir Alay': [0.14, 0.97],
                               'Eastern Pamir': [1.49, 1.28],
                               'Eastern Tibetan Mountains': [0.41, 0.77],
                               'Qilian Shan': [0.7, 0.96],
                               'Tanggula Shan': [-0.11, 0.45],
                               'Eastern Kunlun Shan': [0.75, 1.01],
                               'Hengduan Shan': [-0.25, 0.79],
                               'Gangdise Mountains': [0.25, 0.55],
                               'Eastern Tien Shan': [1.46, 1.82],
                               'Altun Shan': [0.11, 0.96],
                               'Eastern Hindu Kush': [0.5, 1.58]
                               }
tempchange_mu = 0.91
tempchange_sigma = 1.4
tempchange_boundlow = -10
tempchange_boundhigh = 10
tempchange_start = tempchange_mu
tempchange_step = 0.1
tempchange_sigma_adj = 6
tempchange_mu_adj = 0.12
#tempchange_edge_method = 'mb'
tempchange_edge_method = 'mb_norm'
#tempchange_edge_method = 'mb_norm_slope'
tempchange_edge_mb = 1
tempchange_edge_mbnorm = 0.9
tempchange_edge_mbnormslope = -0.75
ddfsnow_disttype = 'truncnormal'
#ddfsnow_disttype = 'uniform'
ddfsnow_mu = 0.0041
ddfsnow_sigma = 0.0015
ddfsnow_boundlow = ddfsnow_mu - 1.96 * ddfsnow_sigma
ddfsnow_boundhigh = ddfsnow_mu + 1.96 * ddfsnow_sigma
ddfsnow_start=ddfsnow_mu

#%% SIMULATION OUTPUT
# Number of model parameter sets for simulation
#  if 1, the median is used
sim_iters = 100
sim_burn = 200
# Simulation output filepath
output_sim_fp = output_filepath + 'simulations/'
# Simulation output statistics
#sim_stat_cns = ['mean', 'std', '2.5%', '25%', 'median', '75%', '97.5%']
sim_stat_cns = ['mean', 'std']


#%% MODEL PARAMETERS 
# Option to import calibration parameters for each glacier
option_import_modelparams = 1
#print('\nSWITCH OPTION IMPORT MODEL PARAMS BACK!\n')
#  Option 1 (default) - calibrated model parameters in netcdf files
#  Option 0 - use the parameters set by the input
precfactor = 1
#  range 0.5 - 2
# Precipitation gradient on glacier [m-1]
precgrad = 0.0001
#  range 0.0001 - 0.0010
# Degree-day factor of snow [m w.e. d-1 degC-1]
ddfsnow = 0.0041
#  range 2.6 - 5.1 * 10^-3
# Temperature adjustment [deg C]
tempchange = 0
#  range -10 to 10
# Lapse rate from gcm to glacier [K m-1]
lrgcm = -0.0065
# Lapse rate on glacier for bins [K m-1]
lrglac = -0.0065
#  k_p in Radic et al. (2013)
#  c_prec in Huss and Hock (2015)
# Degree-day factor of ice [m w.e. d-1 degC-1]
ddfice = ddfsnow / 0.7
#  note: '**' means to the power, so 10**-3 is 0.001
# Ratio degree-day factor snow snow to ice
ddfsnow_iceratio = 0.7
# Temperature threshold for snow [deg C]
tempsnow = 1.0
#   Huss and Hock (2015) T_snow = 1.5 deg C with +/- 1 deg C for ratios
#  facilitates calibration similar to Huss and Hock (2015)
# Frontal ablation  dictating rate [yr-1]
frontalablation_k = 2
# Calving width dictionary to override RGI elevation bins, which can be highly inaccurate at the calving front
width_calving_dict = {'RGI60-01.01390':5730,
                      'RGI60-01.03622':1860,
                      'RGI60-01.10689':4690,
                      'RGI60-01.13638':940,
                      'RGI60-01.14443':6010,
                      'RGI60-01.14683':2240,
                      'RGI60-01.14878':1570,
                      'RGI60-01.17807':2130,
                      'RGI60-01.17840':980,
                      'RGI60-01.17843':1030,
                      'RGI60-01.17876':1390,
                      'RGI60-01.20470':1200,
                      'RGI60-01.20783':760,
                      'RGI60-01.20841':420,
                      'RGI60-01.20891':2050,
                      'RGI60-01.21001':1580,
                      'RGI60-01.23642':2820,
                      'RGI60-01.26736':3560}

# Calving option
option_frontalablation_k = 1
#  Option 1 (default) - use values as Huss and Hock (2015)
#  Option 2 - calibrate each glacier independently, use transfer functions for uncalibrated glaciers
# Calving parameter dictionary
#  according to Supplementary Table 3 in Huss and Hock (2015)
frontalablation_k0dict = {
            1:  3.4,
            2:  0,
            3:  0.2,
            4:  0.2,
            5:  0.5,
            6:  0.3,
            7:  0.5,
            8:  0,
            9:  0.2,
            10: 0,
            11: 0,
            12: 0,
            13: 0,
            14: 0,
            15: 0,
            16: 0,
            17: 6,
            18: 0,
            19: 1}

# Model parameters column names and filepaths
modelparams_colnames = ['lrgcm', 'lrglac', 'precfactor', 'precgrad', 'ddfsnow', 'ddfice', 'tempsnow', 'tempchange']
# Model parameter filepath
if option_calibration == 1:
    modelparams_fp_dict = {
            1:  output_filepath + 'cal_opt1/reg1/',
            3:  output_filepath + 'cal_opt1/',
            4:  output_filepath + 'cal_opt1/',
            6:  output_filepath + 'cal_opt1/reg6/',
            13: output_filepath + 'cal_opt1/reg13/',
            14: output_filepath + 'cal_opt1/reg14/',
            15: output_filepath + 'cal_opt1/reg15/'}
elif option_calibration == 2:
#    modelparams_fp_dict = {
#            13: output_filepath + 'cal_opt2/',
#            14: output_filepath + 'cal_opt2/',
#            15: output_filepath + 'cal_opt2/'}
    modelparams_fp_dict = {
            13: output_filepath + 'cal_opt2_spc_20190308_adjp12_wpriors/cal_opt2/',
            14: output_filepath + 'cal_opt2_spc_20190308_adjp12_wpriors/cal_opt2/',
            15: output_filepath + 'cal_opt2_spc_20190308_adjp12_wpriors/cal_opt2/'}
#    modelparams_fp_dict = {
#            13: output_filepath + 'cal_opt2_spc_3000glac_3chain_adjp12/',
#            14: output_filepath + 'cal_opt2_spc_3000glac_3chain_adjp12/',
#            15: output_filepath + 'cal_opt2_spc_3000glac_3chain_adjp12/'}

#%% CLIMATE DATA
# ERA-INTERIM (Reference data)
# Variable names
era_varnames = ['temperature', 'precipitation', 'geopotential', 'temperature_pressurelevels']
#era_varnames = ['temperature']
#  Note: do not change variable names as these are set to run with the download_erainterim_data.py script.
#        If option 2 is being used to calculate the lapse rates, then the pressure level data is unnecessary.
# Dates
eraint_start_date = '19790101'
eraint_end_date = '20180501'
era5_downloadyearstart = 2017
era5_downloadyearend = 2018
# Resolution
grid_res = '0.5/0.5'
# Bounding box (N/W/S/E)
#bounding_box = '90/0/-90/360'
bounding_box = '50/70/25/105'
# Lapse rate option
option_lr_method = 1
#  Option 0 - lapse rates are constant defined by input
#  Option 1 (default) - lapse rates derived from gcm pressure level temperature data (varies spatially and temporally)
#  Option 2 - lapse rates derived from surrounding pixels (varies spatially and temporally)
#    Note: Be careful with option 2 as the ocean vs land/glacier temperatures can causeƒ unrealistic inversions
#    This is the option used by Marzeion et al. (2012)

# ERA-Interim
era5_fp = main_directory + '/../Climate_data/ERA5/'
#era5_temp_fn = 'ERA5_Temp2m_' + str(era5_downloadyearstart) + '_' + str(era5_downloadyearend) + '.nc'
era5_temp_fn = 'ERA5_Temp2m_test.nc'
era5_prec_fn = 'ERA5_TotalPrec_' + str(era5_downloadyearstart) + '_' + str(era5_downloadyearend) + '.nc'
era5_elev_fn = 'ERA5_geopotential.nc'
era5_pressureleveltemp_fn = ('ERA5_pressureleveltemp_' + str(era5_downloadyearstart) + '_' + str(era5_downloadyearend) 
                             + '.nc')
era5_lr_fn = ('ERA5_lapserates_' + str(era5_downloadyearstart) + '_' + str(era5_downloadyearend) +'_opt' + 
              str(option_lr_method) + '_HMA.nc')

# ERA-Interim
eraint_fp = main_directory + '/../Climate_data/ERA_Interim/download/'
eraint_temp_fn = 'ERAInterim_Temp2m_DailyMeanMonthly_' + eraint_start_date + '_' + eraint_end_date + '.nc'
eraint_prec_fn = 'ERAInterim_TotalPrec_DailyMeanMonthly_' + eraint_start_date + '_' + eraint_end_date + '.nc'
eraint_elev_fn = 'ERAInterim_geopotential.nc'
eraint_pressureleveltemp_fn = 'ERAInterim_pressureleveltemp_' + eraint_start_date + '_' + eraint_end_date + '.nc'
eraint_lr_fn = ('ERAInterim_lapserates_' + eraint_start_date + '_' + eraint_end_date + '_opt' + str(option_lr_method) + 
                '_world.nc')

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
# Glacier selection option
option_glacier_selection = 1
#  Option 1 (default) - enter numbers associated with RGI V6.0
#  Option 2 - glaciers/regions selected via shapefile
#  Option 3 - glaciers/regions selected via new table (other inventory)
# Filepath for RGI files
rgi_filepath = main_directory + '/../RGI/rgi60/00_rgi60_attribs/'
# Column names
rgi_lat_colname = 'CenLat'
rgi_lon_colname = 'CenLon'
elev_colname = 'elev'
indexname = 'GlacNo'
rgi_O1Id_colname = 'glacno'
rgi_glacno_float_colname = 'RGIId_float'
# Column names from table to drop
rgi_cols_drop = ['GLIMSId','BgnDate','EndDate','Status','Connect','Linkages','Name']
# Dictionary of hypsometry filenames
rgi_dict = {
            1:  '01_rgi60_Alaska.csv',
            3:  '03_rgi60_ArcticCanadaNorth.csv',
            4:  '04_rgi60_ArcticCanadaSouth.csv',
            6:  '06_rgi60_Iceland.csv',
            7:  '07_rgi60_Svalbard.csv',
            8:  '08_rgi60_Scandinavia.csv',
            9:  '09_rgi60_RussianArctic.csv',
            13: '13_rgi60_CentralAsia.csv',
            14: '14_rgi60_SouthAsiaWest.csv',
            15: '15_rgi60_SouthAsiaEast.csv'}

# ===== ADDITIONAL DATA (hypsometry, ice thickness, width) =====
# Option to shift all elevation bins by 20 m
#  (required for Matthias' ice thickness and area since they are 20 m off, see email from May 24 2018)
option_shift_elevbins_20m = 1
# Elevation band height [m]
binsize = 10
# Filepath for the hypsometry files
hyps_filepath = main_directory + '/../IceThickness_Huss/bands_10m_DRR/'
# Dictionary of hypsometry filenames 
# (Files from Matthias Huss should be manually pre-processed to be 'RGI-ID', 'Cont_range', and bins starting at 5)
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
                15: 'area_15_Huss_SouthAsiaEast_10m.csv'}
# Extra columns in hypsometry data that will be dropped
hyps_colsdrop = ['RGI-ID','Cont_range']
# Filepath for the ice thickness files
thickness_filepath = main_directory + '/../IceThickness_Huss/bands_10m_DRR/'
# Dictionary of thickness filenames
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
                15: 'thickness_15_Huss_SouthAsiaEast_10m.csv'}
# Extra columns in ice thickness data that will be dropped
thickness_colsdrop = ['RGI-ID','Cont_range']
# Filepath for the width files
width_filepath = main_directory + '/../IceThickness_Huss/bands_10m_DRR/'
# Dictionary of thickness filenames
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
                15: 'width_15_Huss_SouthAsiaEast_10m.csv'}
# Extra columns in ice thickness data that will be dropped
width_colsdrop = ['RGI-ID','Cont_range']

#%% MODEL TIME FRAME DATA
# Note: models are required to have complete data for each year such that refreezing, scaling, etc. are consistent for
#       all time periods.
# Leap year option
option_leapyear = 1
#  Option 1 (default) - leap year days are included, i.e., every 4th year Feb 29th is included in the model, so
#                       days_in_month = 29 for these years.
#  Option 0 - exclude leap years, i.e., February always has 28 days
# User specified start/end dates
#  note: start and end dates must refer to whole years 
startmonthday = '06-01'
endmonthday = '05-31'
# Water year starting month
wateryear_month_start = 10
# First month of winter
winter_month_start = 10
#  for HMA, winter is considered  October 1 - April 30
# First month of summer
summer_month_start = 5
#  for HMA, summer is considered May 1 - September 30
# Option to use dates based on first of each month or those associated with the climate data
option_dates = 1
#  Option 1 (default) - use dates associated with the dates_table that user generates (first of each month)
#  Option 2 - use dates associated with the climate data (problem here is that this may differ between products)
# Model timestep
timestep = 'monthly'
#  enter 'monthly' or 'daily'
#  water year example: 2000 would end on September 2000
#  calendar year example: 2000 would end on December 2000

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


#%% CALIBRATION DATA
# ===== SHEAN GEODETIC =====
shean_fp = main_directory + '/../DEMs/Shean_2019_0213/'
#shean_fn = 'hma_mb_20190215_0815_std+mean.csv'
#shean_fn = 'hma_mb_20190215_0815_std+mean_all_filled_kaab.csv'
shean_fn = 'hma_mb_20190215_0815_std+mean_all_filled_bolch.csv'

shean_rgi_glacno_cn = 'RGIId'
shean_mb_cn = 'mb_mwea'
shean_mb_err_cn = 'mb_mwea_sigma'
shean_time1_cn = 't1'
shean_time2_cn = 't2'
shean_area_cn = 'area_m2'
#shean_vol_cn = 'mb_m3wea'
#shean_vol_err_cn = 'mb_m3wea_sigma'

# ===== BRUN GEODETIC =====
brun_fp = main_directory + '/../DEMs/'
brun_fn = 'Brun_Nature2017_MB_glacier-wide.csv'
brun_rgi_glacno_cn = 'GLA_ID'
brun_mb_cn = 'MB [m w.a a-1]'
brun_mb_err_cn = 'err. on MB [m w.e a-1]'
# NEED TO FINISH SETTING UP BRUN WITH CLASS_MBDATA

# ===== MAUER GEODETIC =====
mauer_fp = main_directory + '/../DEMs/'
#mauer_fn = 'RupperMauer_GeodeticMassBalance_Himalayas_2000_2016.csv'
mauer_fn = 'Mauer_geoMB_HMA_1970s_2000_min80pctCov.csv'
mauer_rgi_glacno_cn = 'RGIId'
mauer_mb_cn = 'geoMassBal'
mauer_mb_err_cn = 'geoMassBalSig'
mauer_time1_cn = 't1'
mauer_time2_cn = 't2'

# ===== MCNABB GEODETIC =====
mcnabb_fp = main_directory + '/../DEMs/McNabb_data/wgms_dv/'
mcnabb_fn = 'Alaska_dV_17jun_preprocessed.csv'
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


# Minimization details
method_opt = 'SLSQP'
ftol_opt = 1e-3

# Limit potential mass balance for future simulations option
option_mb_envelope = 1

# Mass change tolerance [%] - required for calibration
masschange_tolerance = 0.1

# Mass balance uncertainty [mwea]
massbal_uncertainty_mwea = 0.1
# Z-score tolerance
#  all refers to tolerance if multiple calibration points
#  single refers to tolerance if only a single calibration point since we want this to be more exact
zscore_tolerance_all = 1
zscore_tolerance_single = 0.1

#%% REGIONS
grouping = 'himap'
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

#%% TRANSFER FUNCTIONS
# Slope of line of best fit for parameter vs. median elevation
#  These are derived from run_preprocessing.py option_parameter_relationships
#  If the relationship is not significant, then set the slope to 0
tempchange_lobf_property_cn = 'Zmed'
tempchange_lobf_slope = 0.0028212
precfactor_lobf_property_cn = 'Zmed'
precfactor_lobf_slope = -0.004693
ddfsnow_lobf_property_cn = 'Zmed'
ddfsnow_lobf_slope = 1.112333e-06
precgrad_lobf_property_cn = 'Zmed'
precgrad_lobf_slope = 0

#%% BIAS ADJUSTMENT OPTIONS (required for future simulations)
biasadj_fp = output_filepath + 'biasadj/'
#biasadj_fn = 
#biasadj_params_filepath = main_directory + '/../Climate_data/cmip5/bias_adjusted_1995_2100/'
#biasadj_fn_lr = 'biasadj_mon_lravg_1995_2100.csv'
#biasadj_fn_ending = '_biasadj_opt1_1995_2100.csv'

#%% Mass balance model options
# Initial surface type options
option_surfacetype_initial = 1
#  Option 1 (default) - use median elevation to classify snow/firn above the median and ice below.
#   > Sakai et al. (2015) found that the decadal ELAs are consistent with the median elevation of nine glaciers in High
#     Mountain Asia, and Nuimura et al. (2015) also found that the snow line altitude of glaciers in China corresponded
#     well with the median elevation.  Therefore, the use of the median elevation for defining the initial surface type
#     appears to be a fairly reasonable assumption in High Mountain Asia.
#  Option 2 (Need to code) - use mean elevation instead
#  Option 3 (Need to code) - specify an AAR ratio and apply this to estimate initial conditions
# Firn surface type option
option_surfacetype_firn = 1
#  Option 1 (default) - firn is included
#  Option 0 - firn is not included
# Debris surface type option
option_surfacetype_debris = 0
#  Option 0 (default) - debris cover is not included
#  Option 1 - debris cover is included
#   > Load in Batu's debris maps and specify for each glacier
#   > Determine how DDF_debris will be included

# DDF firn
option_DDF_firn = 1
#  Option 1 (default) - DDF_firn is average of DDF_ice and DDF_snow (Huss and Hock, 2015)
#  Option 0 - DDF_firn equal to DDF_snow (m w.e. d-1 degC-1)
# DDF debris
ddfdebris = ddfice
# Reference elevation options for downscaling climate variables
option_elev_ref_downscale = 'Zmed'
#  Option 1 (default) - 'Zmed', median glacier elevation
#  Option 2 - 'Zmax', maximum glacier elevation
#  Option 3 - 'Zmin', minimum glacier elevation (terminus)
# Downscale temperature to bins options
option_temp2bins = 1
#  Option 1 (default) - lr_gcm and lr_glac to adjust temperature from gcm to the glacier bins
# Adjust temperatures based on changes in surface elevation option
option_adjusttemp_surfelev = 1
#  Option 1 (default) - yes, adjust temperature
#  Option 0 - do not adjust temperature
# Downscale precipitation to bins options
option_prec2bins = 1
#  Option 1 (default) - prec_factor and prec_grad to adjust precipitation from gcm to the glacier bins
# Accumulation erosion
option_preclimit = 1
#  Option 1 (default) - limit the uppermost 25% using an expontial fxn
# Accumulation options
option_accumulation = 2
#  Option 1 (default) - Single threshold (<= snow, > rain)
#  Option 2 - single threshold +/- 1 deg uses linear interpolation

# Surface type options
option_surfacetype = 1
#  How is surface type considered, annually?
# Surface ablation options
option_surfaceablation = 1
#  Option 1 (default) - DDF for snow, ice, and debris
# Refreezing model options
option_refreezing = 2
#  Option 1 (default) - heat conduction approach (Huss and Hock, 2015)
#  Option 2 - annual air temperature appraoch (Woodward et al., 1997)
# Refreeze depth [m]
refreeze_depth = 10
# Refreeze month
refreeze_month = 10
#  required for air temperature approach to set when the refreeze is included
# Melt model options
option_melt_model = 1
#  Option 1 (default) DDF
# Mass redistribution / Glacier geometry change options
option_massredistribution = 1
#  Option 1 (default) - Mass redistribution based on Huss and Hock (2015), i.e., volume gain/loss redistributed over the
#                       glacier using empirical normalized ice thickness change curves
# Cross-sectional glacier shape options
option_glaciershape = 1
#  Option 1(default) - parabolic (used by Huss and Hock, 2015)
#  Option 2 - rectangular, i.e., glacier lowering but area and width does not change
#  Option 3 - triangular
# Glacier width option
option_glaciershape_width = 1
#  Option 0 (default) - do not include
#  Option 1 - include
# Advancing glacier ice thickness change threshold
icethickness_advancethreshold = 5
#  Huss and Hock (2015) use a threshold of 5 m
# Percentage of glacier considered to be terminus
terminus_percentage = 20
#  Huss and Hock (2015) use 20% to calculate new area and ice thickness

#%% OUTPUT OPTIONS
output_package = 2
    # Option 0 - no netcdf package
    # Option 1 - "raw package" [preferred units: m w.e.]
    #             monthly variables for each bin (temp, prec, acc, refreeze, snowpack, melt, frontalablation,
    #                                             massbal_clim)
    #             annual variables for each bin (area, icethickness, surfacetype)
    # Option 2 - "Glaciologist Package" output [units: m w.e. unless otherwise specified]:
    #             monthly glacier-wide variables (prec, acc, refreeze, melt, frontalablation, massbal_total, runoff, 
    #                                             snowline)
    #             annual glacier-wide variables (area, volume, ELA)
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

#%% WARNING MESSAGE OPTION
option_warningmessages = 1
#  Warning messages are a good check to make sure that the script is running properly, and small nuances due to
#  differences in input data (e.g., units associated with GCM air temperature data are correct)
#  Option 1 (default) - print warning messages within script that are meant to assist user
#                       currently these messages are only included in a few scripts (e.g., climate data)
#  Option 0 - do not print warning messages within script

#%% MODEL PROPERTIES 
# Density of ice [kg m-3] (or Gt / 1000 km3)
density_ice = 900
# Density of water [kg m-3]
density_water = 1000
# Area of ocean [km2]
area_ocean = 362.5 * 10**6
# Heat capacity of ice [J K-1 kg-1]
ch_ice = 1.89 * 10**6
# Thermal conductivity of ice [W K-1 m-1]
k_ice = 2.33
# Model tolerance (used to remove low values caused by rounding errors)
tolerance = 1e-12
# Gravity [m s-2]
gravity = 9.81
# Standard pressure [Pa]
pressure_std = 101325
# Standard temperature [K]
temp_std = 288.15
# Universal gas constant [J mol-1 K-1]
R_gas = 8.3144598
# Molar mass of Earth's air [kg mol-1]
molarmass_air = 0.0289644
# Bulk flow parameter for frontal ablation (m^-0.5)
af = 0.7

# Pass variable to shell script
if __name__ == '__main__':
    print(rgi_regionsO1[0])