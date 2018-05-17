"""
pygem_input.py is a list of the model inputs that are required to run PyGEM.

These inputs are separated from the main script, so that they can easily be
configured. Other modules can also import these variables to reduce the number
of parameters required to run a function.
"""
# DEVELOPER'S NOTE: Consider structuring the input via steps and/or as required input or have a separate area for
#   variables, parameters that don't really change.
import os
import numpy as np

#%% ===== MODEL PARAMETERS T0 ADJUST ==========================================
# ===== CALIBRATION OPTIONS =====
option_calibration = 0
#  Option 0 (default) - regular model simulation (variables defined)
#  Option 1 - calibration run (output differs and calibration data selected)
# ===== PARALLELS OPTION =====
option_parallels = 1
#  Option 0 - do not use parallels
#  Option 1 - Use parallels for a single gcm run
#  Option 2 - Use parallels for multiple sets of gcms simulations
# ===== GLACIER SELECTION =====
# Region number 1st order (RGI V6.0) - HMA is 13, 14, 15
rgi_regionsO1 = [15]
# 2nd order region numbers (RGI V6.0)
rgi_regionsO2 = 'all'
#rgi_regionsO2 = [2]
#  do not change this
# RGI glacier number (RGI V6.0)
#rgi_glac_number = 'all'
#rgi_glac_number = ['03473', '03733']
#rgi_glac_number = ['03473']
#rgi_glac_number = ['02760']
rgi_glac_number = ['06881']
#rgi_glac_number = ['09249']
#rgi_glac_number = ['01204']
#rgi_glac_number = ['09991']
#rgi_glac_number = ['10070']
#  example numbers are associated with rgi_regionsO1 [15]; 'all' includes all glaciers iwthin a region

# First year of model run
startyear = 2000
#  water year example: 2000 would start on October 1999, since October 1999 - September 2000 is the water year 2000
#  calendar year example: 2000 would start on January 2000
# Last year of model run
endyear = 2100

# Remove NaN values (glaciers without calibration data)
option_removeNaNcal = 1
#  Option 0 (default) - do not remove these glaciers
#  Option 1 - remove glaciers without cal data
# Model setup directory
main_directory = os.getcwd()
modelsetup_dir = main_directory + '/../PyGEM_cal_setup/'
# Glacier list name
#glacier_list_name = 'glacier_list_R15_all'
#  CAN DELETE!

# Limit potential mass balance for future simulations option
option_mb_envelope = 1

# ===== MODEL PARAMETERS =====
# Option to load calibration parameters for each glacier
option_loadparameters = 1
#  Option 1 (default) - csv of glacier parameters
#  Option 0 - use the parameters set by the input
precfactor = 1
#  range 0.5 - 2
# Precipitation gradient on glacier [% m-1]
precgrad = 0.0001
#  range 0.0001 - 0.0010
# Degree-day factor of snow [m w.e. d-1 degC-1]
ddfsnow = 0.0041
#  range 2.6 - 5.1 * 10^-3
# Temperature adjustment [deg C]
tempchange = 0
#  range -10 to 10


# Precipitation correction factor [-]
# ===== Grid search parameters =====
grid_precfactor = np.arange(0.75, 2, 0.25)
grid_tempbias = np.arange(-4, 6, 2)
grid_ddfsnow = np.arange(0.0031, 0.0056, 0.0005)
grid_precgrad = np.arange(0.0001, 0.0007, 0.0002)

#%% ===== MODEL PARAMETERS THAT ARE CONSTANT ==================================
# Lapse rate from gcm to glacier [K m-1]
lrgcm = -0.0065
# Lapse rate on glacier for bins [K m-1]
lrglac = -0.0065
#  k_p in Radic et al. (2013)
#  c_prec in Huss and Hock (2015)
# Degree-day factor of ice [m w.e. d-1 degC-1]
ddfice = 0.0041 / 0.7
#  note: '**' means to the power, so 10**-3 is 0.001
# Ratio degree-day factor snow snow to ice
ddfsnow_iceratio = 0.7
# Temperature threshold for snow [deg C]
tempsnow = 1.0
#   Huss and Hock (2015) T_snow = 1.5 deg C with +/- 1 deg C for ratios
#  facilitates calibration similar to Huss and Hock (2015)

#%% ------- INPUT FOR STEP ONE --------------------------------------------------
# STEP ONE: Model Region/Glaciers
#   The user needs to define the region/glaciers that will be used in the model run. The user has the option of choosing
#   the standard RGI regions or defining their own regions.
#   Note: Make sure that all input variables are defined for the chosen option

# ----- Input required for glacier selection options -----
# Glacier selection option
option_glacier_selection = 1
#  Option 1 (default) - enter numbers associated with RGI V6.0
#  Option 2 - glaciers/regions selected via shapefile
#  Option 3 - glaciers/regions selected via new table (other inventory)
# OPTION 1: RGI glacier inventory information
# Filepath for RGI files
rgi_filepath = main_directory + '/../RGI/rgi60/00_rgi60_attribs/'
#  file path where the rgi tables are located on the computer
#  os.path.dirname(__file__) is getting the directory where the pygem model is running.  '..' goes up a folder and then
#  allows it to enter RGI and find the folders from there.
# Latitude column name
lat_colname = 'CenLat'
# Longitude column name
lon_colname = 'CenLon'
# Elevation column name
elev_colname = 'elev'
# Index name
indexname = 'GlacNo'
# Dictionary of hypsometry filenames
rgi_dict = {
            13: '13_rgi60_CentralAsia.csv',
            14: '14_rgi60_SouthAsiaWest.csv',
            15: '15_rgi60_SouthAsiaEast.csv'}
# Columns in the RGI tables that are not necessary to include in model run.
rgi_cols_drop = ['GLIMSId','BgnDate','EndDate','Status','Connect','Surging','Linkages','Name']
#  this will change as model develops to include ice caps, calving, etc.
rgi_O1Id_colname = 'RGIId-O1No'
# OPTION 2: Select/customize regions based on shapefile(s)
# Enter shapefiles, etc.

#%% ------- INPUT FOR STEP TWO --------------------------------------------------
# STEP TWO: Additional model setup
#   Additional model setup that has been separated from the glacier selection in step one in order to keep the input
#   organized and easy to read.
#
# ----- Input required for glacier hypsometry -----
# extract glacier hypsometry according to Matthias Huss's ice thickness files (area and width included as well)
#  Potential option - automatically extract 50m hypsometry from RGI60
#  Potential option - extract glacier hypsometry and mass balance from David Shean's measurements using high-res DEMs
# Elevation band height [m]
binsize = 10
# Filepath for the hypsometry files
hyps_filepath = main_directory + '/../IceThickness_Huss/bands_10m_DRR/'
# Dictionary of hypsometry filenames 
# (FILES SHOULD BE PRE-PROCESSED TO BE 'RGI-ID', 'Cont_range' and bins starting at 5)
hyps_filedict = {
                13: 'area_13_Huss_CentralAsia_10m.csv',
                14: 'area_14_Huss_SouthAsiaWest_10m.csv',
                15: 'area_15_Huss_SouthAsiaEast_10m.csv'}
# Extra columns in hypsometry data that will be dropped
hyps_colsdrop = ['RGI-ID','Cont_range']
# Filepath for the ice thickness files
thickness_filepath = main_directory + '/../IceThickness_Huss/bands_10m_DRR/'
# Dictionary of thickness filenames
thickness_filedict = {
                13: 'thickness_13_Huss_CentralAsia_10m.csv',
                14: 'thickness_14_Huss_SouthAsiaWest_10m.csv',
                15: 'thickness_15_Huss_SouthAsiaEast_10m.csv'}
# Extra columns in ice thickness data that will be dropped
thickness_colsdrop = ['RGI-ID','Cont_range']
# Filepath for the width files
width_filepath = main_directory + '/../IceThickness_Huss/bands_10m_DRR/'
# Dictionary of thickness filenames
width_filedict = {
                13: 'width_13_Huss_CentralAsia_10m.csv',
                14: 'width_14_Huss_SouthAsiaWest_10m.csv',
                15: 'width_15_Huss_SouthAsiaEast_10m.csv'}
# Extra columns in ice thickness data that will be dropped
width_colsdrop = ['RGI-ID','Cont_range']

# ----- Input required for model time frame -----
# Note: models are required to have complete data for each year such that refreezing, scaling, etc. are consistent for
#       all time periods.
# Leap year option
option_leapyear = 1
#  Option 1 (default) - leap year days are included, i.e., every 4th year Feb 29th is included in the model, so
#                       days_in_month = 29 for these years.
#  Option 0 - exclude leap years, i.e., February always has 28 days
# Water year option
option_wateryear = 1
#  Option 1 (default) - use water year instead of calendar year (ex. 2000: Oct 1 1999 - Sept 1 2000)
#  Option 0 - use calendar year
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
# Number of years for model spin up [years]
spinupyears = 5

# ----- Input required for initial surface type -----
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


#%% ------- INPUT FOR STEP THREE ------------------------------------------------
# STEP THREE: Climate Data
#   User has the climate data and how it is downscaled to glacier and bins.
# Information regarding climate data
#  - netcdf files downloaded from cmip5-archive at ethz or ERA-Interim reanalysis data (ECMWF)
#  - NG refers to New Generation of CMIP5 data, i.e., a homogenized dataset
#  - _var refers to variables
#  - _fx refers to time invariant (constant/fixed) data
#  - temp      variable name is 't2m'       for ERAInterim and 'tas'  for CMIP5
#  - prec      variable name is 'tp'        for ERAInterim and 'pr'   for CMIP5
#  - elev      variable name is 'z'         for ERAInterim and 'orog' for CMIP5 
#  - latitude  variable name is 'latitude'  for ERAInterim and 'lat'  for CMIP5
#  - longitude variable name is 'longitude' for ERAInterim and 'lon'  for CMIP5
#  - time      variable name is 'time'      for both

# Downscale GCM data option
option_gcm_downscale = 1
#  Option 1 (default): select climate data based on nearest neighbor
#  Option 2: import prepared csv files (saves time)
# Lapse rate option
option_lapserate_fromgcm = 1
#  Option 0 - lapse rates are constant defined by input
#  Option 1 (default) - lapse rates derived from gcm pressure level temperature data (varies spatially and temporally)
#  Option 2 (NEED TO CODE) - lapse rates derived from surrounding pixels (varies spatially and temporally)

# REFERENCE DATA INFORMATION
# Reference climate data filepath
filepath_ref = main_directory + '/../Climate_data/ERA_Interim/'
# Reference climate data csvs (THESE ARE ALL GENERATED IN PRE-PROCESSING)
# Dictionary of filenames for temperature, precipitation, lapse rate, and elevation data
gcmtemp_filedict = {
                    13: 'csv_ERAInterim_temp_19952015_13_CentralAsia.csv',
                    14: 'csv_ERAInterim_temp_19952015_14_SouthAsiaWest.csv',
                    15: 'csv_ERAInterim_temp_19952015_15_SouthAsiaEast.csv'}
gcmprec_filedict = {
                    13: 'csv_ERAInterim_prec_19952015_13_CentralAsia.csv',
                    14: 'csv_ERAInterim_prec_19952015_14_SouthAsiaWest.csv',
                    15: 'csv_ERAInterim_prec_19952015_15_SouthAsiaEast.csv'}
gcmelev_filedict = {
                    13: 'csv_ERAInterim_elev_13_CentralAsia.csv',
                    14: 'csv_ERAInterim_elev_14_SouthAsiaWest.csv',
                    15: 'csv_ERAInterim_elev_15_SouthAsiaEast.csv'}
gcmlapserate_filedict = {
                         13: 'csv_ERAInterim_lapserate_19952015_13_CentralAsia.csv',
                         14: 'csv_ERAInterim_lapserate_19952015_14_SouthAsiaWest.csv',
                         15: 'csv_ERAInterim_lapserate_19952015_15_SouthAsiaEast.csv'}

## CLIMATE DATA INFORMATION
## ERAINTERIM CLIMATE DATA (Reference data)
## Climate data filepath
#gcm_filepath_var = main_directory + '/../Climate_data/ERA_Interim/'
#gcm_filepath_fx = main_directory + '/../Climate_data/ERA_Interim/'
## Climate file and variable names
#gcm_temp_filename = 'ERAInterim_AirTemp2m_DailyMeanMonthly_1995_2016.nc'
#gcm_temp_varname = 't2m'
#gcm_prec_filename = 'ERAInterim_TotalPrec_DailyMeanMonthly_1979_2017.nc'
#gcm_prec_varname = 'tp'
#gcm_elev_filename = 'ERAInterim_geopotential.nc'
#gcm_elev_varname = 'z'
#gcm_lapserate_filename = 'HMA_Regions13_14_15_ERAInterim_lapserates_1979_2017.nc' # GENERATED IN PRE-PROCESSING
#gcm_lapserate_varname = 'lapserate'
#gcm_lat_varname = 'latitude'
#gcm_lon_varname = 'longitude'
#gcm_time_varname = 'time'


# CMIP5 INPUT CLIMATE DATA
#gcm_name = 'MPI-ESM-LR'
#rcp_scenario = 'rcp26'
# Climate data filepath
gcm_filepath_var = main_directory + '/../Climate_data/cmip5/rcp26_r1i1p1_monNG/'
gcm_filepath_fx = main_directory + '/../Climate_data/cmip5/rcp26_r0i0p0_fx/'
# Climate file and variable names
#gcm_temp_filename = 'tas_mon_' + gcm_name + '_' + rcp_scenario + '_r1i1p1_native.nc'
gcm_temp_varname = 'tas'
#gcm_prec_filename = 'pr_mon_' + gcm_name + '_' + rcp_scenario + '_r1i1p1_native.nc'
gcm_prec_varname = 'pr'
#gcm_elev_filename = 'orog_fx_' + gcm_name + '_' + rcp_scenario + '_r0i0p0.nc'
gcm_elev_varname = 'orog'
gcm_lat_varname = 'lat'
gcm_lon_varname = 'lon'
gcm_time_varname = 'time'

# Bias adjustments option (required for future simulations)
option_bias_adjustment = 2
#  Option 0 - ignore bias adjustments
#  Option 1 - bias adjustments using new technique 
#  Option 2 - bias adjustments using Huss and Hock [2015] methods
biasadj_data_filepath = main_directory + '/../Climate_data/cmip5/R15_rcp26_1995_2100/'
biasadj_params_filepath = main_directory + '/../Climate_data/cmip5/bias_adjusted_1995_2100/'
biasadj_fn_lr = 'biasadj_mon_lravg_1995_2100.csv'
biasadj_fn_ending = '_biasadj_opt1_1995_2100.csv'

# Calibration datasets
# Geodetic mass balance dataset
# Filepath
cal_mb_filepath = main_directory + '/../DEMs/'
# Filename
cal_mb_filedict = {
                   13: 'geodetic_glacwide_DShean20170207_13_CentralAsia.csv',
                   14: 'geodetic_glacwide_DShean20170207_14_SouthAsiaWest.csv',
                   15: 'geodetic_glacwide_DShean20170207_15_SouthAsiaEast.csv'}
# RGIId column name
cal_rgi_colname = 'RGIId'
# Mass balance column name
massbal_colname = 'mb_mwea'
# Mass balance uncertainty column name
massbal_uncertainty_colname = 'mb_mwea_sigma'
# Mass balance date 1 column name
massbal_time1 = 'year1'
# Mass balance date 1 column name
massbal_time2 = 'year2'
# Mass balance tolerance [m w.e.a]
massbal_tolerance = 0.1
# Calibration optimization tolerance
#cal_tolerance = 1e-4

#%% ------- INPUT FOR STEP FOUR -------------------------------------------------
# STEP FOUR: Glacier Evolution
#   Enter brief description of user options here.

# Model parameters filepath, filename, and column names
modelparams_filepath = main_directory + '/../Calibration_datasets/'
#modelparams_filename = 'calparams_R15_20180306_nearest.csv'
#modelparams_filename = 'calparams_R15_20180305_fillnanavg.csv'
#modelparams_filename = 'calparams_R15_20180403_nearest.csv'
modelparams_filename = 'calparams_R15_20180403_nnbridx.csv'
#modelparams_filename = 'calparams_R14_20180313_fillnanavg.csv'
modelparams_colnames = ['lrgcm', 'lrglac', 'precfactor', 'precgrad', 'ddfsnow', 'ddfice', 'tempsnow', 'tempchange']

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

#------- INPUT FOR STEP FOUR -------------------------------------------------
# STEP FIVE: Output
# Output package number
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
output_filepath = '../Output/'
calibrationcsv_filenameprefix = 'calibration_'
calibrationnetcdf_filenameprefix = 'calibration_gridsearchcoarse_R'
netcdf_fn_prefix = 'PyGEM_R'

#%% ========== LIST OF MODEL INPUT ==============================================
#------- INPUT FOR CODE ------------------------------------------------------
# Warning message option
option_warningmessages = 1
#  Warning messages are a good check to make sure that the script is running properly, and small nuances due to
#  differences in input data (e.g., units associated with GCM air temperature data are correct)
#  Option 1 (default) - print warning messages within script that are meant to
#                      assist user
#  Option 0 - do not print warning messages within script

#%% ------- MODEL PROPERTIES ----------------------------------------------------
# Density of ice [kg m-3]
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
