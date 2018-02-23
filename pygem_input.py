"""
pygem_input.py is a list of the model inputs that are required to run PyGEM.

These inputs are separated from the main script, so that they can easily be
configured. Other modules can also import these variables to reduce the number
of parameters required to run a function.
"""
# DEVELOPER'S NOTE: Consider structuring the input via steps and/or as required input or have a separate area for 
#   variables, parameters that don't really change.
import os

# ========== LIST OF MODEL INPUT ==============================================
#------- INPUT FOR CODE ------------------------------------------------------
# Warning message option
option_warningmessages = 1
#  Warning messages are a good check to make sure that the script is running properly, and small nuances due to 
#  differences in input data (e.g., units associated with GCM air temperature data are correct)
#  Option 1 (default) - print warning messages within script that are meant to
#                      assist user
#  Option 0 - do not print warning messages within script

#------- MODEL PROPERTIES ----------------------------------------------------
# Density of ice [km m-3]
density_ice = 900
# Density of water [km m-3]
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

#------- INPUT FOR STEP ONE --------------------------------------------------
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
# Directory name
main_directory = os.getcwd()
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
# 1st order region numbers (RGI V6.0)
rgi_regionsO1 = [15]
#  enter integer(s) in brackets, e.g., [13, 14]
# 2nd order region numbers (RGI V6.0)
rgi_regionsO2 = 'all'
#rgi_regionsO2 = [2]
#  enter 'all' to include all subregions or enter integer(s) in brackets to specify specific subregions, e.g., [5, 6]. 
# RGI glacier number (RGI V6.0)
rgi_glac_number = ['03473', '03733']
#rgi_glac_number = 'all'
#  enter 'all' to include all glaciers within (sub)region(s) or enter a string of complete glacier number for specific 
#  glaciers, e.g., ['05000', '07743'] for glaciers '05000' and '07743'
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


#------- INPUT FOR STEP TWO --------------------------------------------------
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

""" NEED TO CODE VOLUME-LENGTH SCALING """
# Option - volume-length scaling
#   V_init = c_v * Area_init ^ VA_constant_exponent
#   where L is the change in
#   Need to define c_l and q, which are volume length scaling constants


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
# First year of model run
startyear = 2000
#  water year example: 2000 would start on October 1999, since October 1999 - September 2000 is the water year 2000
#  calendar year example: 2000 would start on January 2000
# Last year of model run
endyear = 2015
#  water year example: 2000 would end on September 2000
#  calendar year example: 2000 would end on December 2000
# Number of years for model spin up [years]
spinupyears = 0

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


#------- INPUT FOR STEP THREE ------------------------------------------------
# STEP THREE: Climate Data
#   The user has the option to choose the type of climate data being used in the
#   model run, and how that data will be downscaled to the glacier and bins.
# Option to downscale GCM data
option_gcm_downscale = 1
#  Option 1 (default): select climate data based on nearest neighbor
#  Option 2: import prepared csv files (saves time)
# Filepath to GCM variable files
gcm_filepath_var = main_directory + '/../Climate_data/ERA_Interim/'
#  _var refers to variable data; NG refers to New Generation of CMIP5 data, i.e., a homogenized dataset
# Filepath to GCM fixed variable files
gcm_filepath_fx = main_directory + '/../Climate_data/ERA_Interim/'
#  _fx refers to time invariant (constant) data
# Temperature filename
gcm_temp_filename = 'ERAInterim_AirTemp2m_DailyMeanMonthly_1995_2016.nc'
#  netcdf files downloaded from cmip5-archive at ethz or ERA-Interim reanalysis data (ECMWF)
# Precipitation filename
gcm_prec_filename = 'ERAInterim_TotalPrec_DailyMeanMonthly_1979_2017.nc'
# Elevation filename
gcm_elev_filename = 'ERAInterim_geopotential.nc'
# Temperature variable name given by GCM
gcm_temp_varname = 't2m'
#  't2m' for ERA Interim, 'tas' for CMIP5
# Precipitation variable name given by GCM
gcm_prec_varname = 'tp'
#  'tp' for ERA Interim, 'pr' for CMIP5
# Elevation variable name given by GCM
gcm_elev_varname = 'z'
#  'z' for ERA Interim, 'orog' for CMIP5
# Latitude variable name given by GCM
gcm_lat_varname = 'latitude'
#  'latitude' for ERA Interim, 'lat' for CMIP5
# Longitude variable name given by GCM
gcm_lon_varname = 'longitude'
#  'longitude' for ERA Interim, 'lon' for CMIP5
# Time variable name given by GCM
gcm_time_varname = 'time'

# Calibration option
option_calibration = 0
#  Option 0 (default) - regular model simulation (variables defined)
#  Option 1 - calibration run (glacier area remains constant)
# Calibration datasets
# Geodetic mass balance dataset
# Filepath
cal_mb_filepath = main_directory + '/../DEMs/'
# Filename
cal_mb_filename = 'geodetic_glacwide_DShean20170207_15_SouthAsiaEast.csv'
# RGIId column name
cal_rgi_colname = 'RGIId'
# Mass balance column name
massbal_colname = 'mb_mwea'
# Mass balance date 1 column name
massbal_time1 = 'year1'
# Mass balance date 1 column name
massbal_time2 = 'year2'
# Mass balance tolerance [m w.e.a]
massbal_tolerance = 0.1
# Calibration optimization tolerance
#cal_tolerance = 1e-4

#------- INPUT FOR STEP FOUR -------------------------------------------------
# STEP FOUR: Glacier Evolution
#   Enter brief description of user options here.

# Lapse rate from gcm to glacier [K m-1]
lr_gcm = -0.0065
# Lapse rate on glacier for bins [K m-1]
lr_glac = -0.0065
# Precipitation correction factor [-]
prec_factor = 1
#  k_p in Radic et al. (2013)
#  c_prec in Huss and Hock (2015)
# Precipitation gradient on glacier [% m-1]
prec_grad = 0.0001
# Degree-day factor of ice [m w.e. d-1 degC-1]
ddf_ice = 7.2 * 10**-3
#  note: '**' means to the power, so 10**-3 is 0.001
# Degree-day factor of snow [m w.e. d-1 degC-1]
ddf_snow = 3.6 * 10**-3
# Temperature threshold for snow (C)
temp_snow = 1.5
#   Huss and Hock (2015) T_snow = 1.5 deg C with +/- 1 deg C for ratios

# DDF firn 
option_DDF_firn = 1
#  Option 1 (default) - DDF_firn is average of DDF_ice and DDF_snow (Huss and Hock, 2015)
#  Option 0 - DDF_firn equal to DDF_snow (m w.e. d-1 degC-1)
# DDF debris
ddf_debris = ddf_ice
# Reference elevation options for downscaling climate variables
option_elev_ref_downscale = 'Zmed'
#  Option 1 (default) - 'Zmed', median glacier elevation
#  Option 2 - 'Zmax', maximum glacier elevation
#  Option 3 - 'Zmin', minimum glacier elevation (terminus)
# Downscale temperature to bins options
option_temp2bins = 1
#  Option 1 (default) - lr_gcm and lr_glac to adjust temperature from gcm to the glacier bins
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
# Temperature adjustment options
option_adjusttemp_surfelev = 1
#  Option 1 (default) - yes, adjust temperature
#  Option 2 - do not adjust temperature
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
#  Option 2 - rectangular
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
## Constant area or allow geometry changes
#option_areaconstant = 1
##  Option 0 (default simulation) - area is not constant, glacier can widen/narrow and retreat/advance
##  Option 1 (default calibration) - area is constant to avoid retreat/advance mass balance feedbacks  

## Calibration constraint packages
#option_calibration_constraints = 2
##  Option 1 (default): optimize all values within their given bounds
## Option 1 - optimize all parameters
## Option 2 - only optimize precfactor
## Option 3 - only optimize precfactor, DDFsnow, DDFice
## Option 4 - only optimize precfactor, DDFsnow, DDFice; DDFice = 2 x DDFsnow
## Option 5 - only optimize precfactor, DDFsnow, DDFice; DDFice > DDFsnow

#------- INPUT FOR STEP FOUR -------------------------------------------------
# STEP FIVE: Output
# Output package number
output_package = 0
    # Option 0 - no netcdf package
    # Option 1 - "raw package" [preferred units: m w.e.]
    #             monthly variables for each bin (temp, prec, acc, refreeze, snowpack, melt, frontalablation, 
    #                                             massbal_clim)
    #             annual variables for each bin (area, icethickness, surfacetype)
netcdf_filenameprefix = 'PyGEM_output_rgiregion'
netcdf_filepath = '../Output/'









