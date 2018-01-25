"""
pygem_input.py is a list of the model inputs that are required to run PyGEM.

These inputs are separated from the main script, so that they can easily be
configured. Other modules can also import these variables to reduce the number
of parameters required to run a function.
"""
# Note: (DEVELOPMENT) Consider structuring the input via steps and/or as
#       required input or have a separate area for variables, parameters that
#       don't really change.
import numpy as np

# ========== LIST OF MODEL INPUT ==============================================
#------- INPUT FOR CODE ------------------------------------------------------
option_modelrun_type = 0
# Option 0 (default) - calibration run (glacier area remains constant)
# Option 1 - regular model run
option_warningmessages = 1
# Warning messages are a good check to make sure that the script is running 
# properly, and small nuances due to differences in input data (e.g., units 
# associated with GCM air temperature data are correct)
# Option 1 (default) - print warning messages within script that are meant to
#                      assist user
# Option 0 - do not print warning messages within script

#------- MODEL PROPERTIES ----------------------------------------------------
density_ice = 900
#  kg m**3
density_water = 1000
#  kg m**-3
area_ocean = 362.5 * 10**6
#  km**2
tolerance = 1e-12
#  used to remove very low values caused by rounding errors in calculations

#------- INPUT FOR STEP ONE --------------------------------------------------
# STEP ONE: Model Region/Glaciers
#   The user needs to define the region/glaciers that will be used in the model
#   run. The user has the option of choosing the standard RGI regions or
#   defining their own regions.
#   Note: Make sure that all input variables are defined for the chosen option
# In step one, the model will:
#   > select glaciers included in model run
import os
# ----- Input required for glacier selection options -----
option_glacier_selection = 1
# Option 1 (default) - enter numbers associated with RGI V6.0
# Option 2 - glaciers/regions selected via shapefile
# Option 3 - glaciers/regions selected via new table (other inventory)
# OPTION 1: RGI glacier inventory information
# NOW USING RELATIVE PATHWAY!
rgi_filepath = os.path.dirname(__file__) + '/../RGI/rgi60/00_rgi60_attribs/'
# os.path.dirname(__file__) is getting the directory where the pygem model
# is running.  '..' goes up a folder and then allows it to enter RGI and
# find the folders from there.
# file path where the rgi tables are located on the computer
lat_colname = 'CenLat'
# center latitude
lon_colname = 'CenLon'
# center longitude
elev_colname = 'elev'
# elevation
indexname = 'GlacNo'
# glacier number specific to each model run
rgi_regionsO1 = [15]
# 1st order regions defined by RGI V6.0
# Enter integer(s) in brackets, e.g., [13, 14]
rgi_regionsO2 = 'all'
# rgi_regionsO2 = [1]
# 2nd order regions defined by RGI V6.0
# Enter 'all' to include all subregions or enter integer(s) in brackets
# to specify specific subregions, e.g., [5, 6]. If entering individual
# glaciers (rgi_glac_number != 'all'), then rgi_regionsO2 should be 'all'.
rgi_glac_number = ['03473', '03733']
#rgi_glac_number = 'all'
#     glacier numbers defined by RGI V6.0
#     Enter 'all' to include all glaciers within (sub)region(s) or enter a
#     string of complete glacier number for specific glaciers, e.g.,
#     ['05000', '07743'] for glaciers '05000' and '07743'
# Dictionary of hypsometry filenames
rgi_dict = {
            13: '13_rgi60_CentralAsia.csv',
            14: '14_rgi60_SouthAsiaWest.csv',
            15: '15_rgi60_SouthAsiaEast.csv'}
# Columns in the RGI tables that are not necessary to include in model run.
#  This will change as model develops to include ice caps, calving, etc.
rgi_cols_drop = ['GLIMSId', 'BgnDate', 'EndDate', 'Status', 'Connect', 'Form',
                'TermType', 'Surging', 'Linkages', 'Name']
# OPTION 2: Select/customize regions based on shapefile(s)
# Enter shapefiles, etc.


#------- INPUT FOR STEP TWO --------------------------------------------------
# STEP TWO: Additional model setup
#   Additional model setup that has been separated from the glacier selection in
#   step one in order to keep the input organized and easy to read.
# In step two, the model will:
#   > select glacier hypsometry
#   > define the model time frame
#   > define the initial surface type
#
# ----- Input required for glacier hypsometry -----
option_glacier_hypsometry = 1
# Option 1 (default) - automatically extract 50m hypsometry from RGI60
# Option 2 - extract glacier hypsometry according to Matthias Huss's ice
#            thickness files (area and width included as well)
# Option 3 - extract glacier hypsometry and mass balance from David Shean's
#            measurements using high-res DEMs

# Elevation band height [m]
binsize = 10
# Filepath for the hypsometry files
hyps_filepath = os.path.dirname(__file__) + '/../IceThickness_Huss/bands_10m_DRR/'
# Dictionary of hypsometry filenames
hypsfile_dict = {
                13: 'area_13_Huss_CentralAsia_10m.csv',
                14: 'area_14_Huss_SouthAsiaWest_10m.csv',
                15: 'area_15_Huss_SouthAsiaEast_10m.csv'}
# Extra columns in hypsometry data that will be dropped
hyps_cols_drop = ['RGI-ID','Cont_range']
# Filepath for the ice thickness files
thickness_filepath = os.path.dirname(__file__) + '/../IceThickness_Huss/bands_10m_DRR/'
# Dictionary of thickness filenames
thicknessfile_dict = {
                13: 'thickness_13_Huss_CentralAsia_10m.csv',
                14: 'thickness_14_Huss_SouthAsiaWest_10m.csv',
                15: 'thickness_15_Huss_SouthAsiaEast_10m.csv'}
# Extra columns in ice thickness data that will be dropped
thickness_cols_drop = ['RGI-ID','Cont_range']

# ----- Input required for model time frame -----
# Note: models are required to have complete data for each year, i.e., the
#       model will start on January 1st and end December 31st for the given
#       start and end year, respectively.
option_leapyear = 1
# Option 1 (default) - leap year days are included, i.e., every 4th year Feb 29th is
#                      included in the model, so days_in_month = 29 for these years.
# Option 0 - exclude leap years, i.e., February always has 28 days
option_wateryear = 1
# Option 1 (default) - use water year instead of calendar year
#                      (ex. 2017 extends from Oct 1 2016 - Sept 1 2017)
# Option 0 - use calendar year
""" NEED TO CODE THIS IN """
# Note: the model will run on full years in order to ensure that refreezing,
#       scaling, etc. are consistent for all time periods. Therefore,
#       output options are where specific dates for calibration periods,
#       etc. will be entered.
wateryear_month_start = 10
option_dates = 1
# Option 1 (default) - use dates associated with the dates_table that user generates (first of each month)
# Option 2 - use dates associated with the climate data (problem here is that this may differ between products)
timestep = 'monthly'
# model time step ('monthly' or 'daily')
startyear = 2000
# first year of model run
""" DEVELOPER'S NOTE: NEED TO MAKE DIFFERENCE BETWEEN WATER YEAR AND CALENDAR YEAR CRYSTAL CLEAR """
endyear = 2015
# last year of model run
spinupyears = 0
# model spin up period (years)
""" NEED TO CODE THIS IN, ESPECIALLY FOR THE OUTPUT """
winter_month_start = 10
#  winter is considered November 1 - April 30
summer_month_start = 5
#  summer is considered May 1 - Sept 30

# ----- Input required for initial surface type -----
""" ADD THIS INPUT HERE """
option_surfacetype_initial = 1
# Option 1 (default) - use median elevation to classify snow/firn above the median and ice below.
#   > Sakai et al. (2015) found that the decadal ELAs are consistent with the median elevation of nine glaciers in High 
#     Mountain Asia, and Nuimura et al. (2015) also found that the snow line altitude of glaciers in China corresponded
#     well with the median elevation.  Therefore, the use of the median elevation for defining the initial surface type
#     appears to be a fairly reasonable assumption in High Mountain Asia. 
# Option 2 (Need to code) - use mean elevation instead
# Option 3 (Need to code) - specify an AAR ratio and apply this to estimate initial conditions
option_surfacetype_firn = 1
# Option 1 (default) - firn is included
# Option 0 - firn is not included
option_surfacetype_debris = 0
# Option 0 (default) - debris cover is not included
# Option 1 - debris cover is included
#   > Load in Batu's debris maps and specify for each glacier
#   > Determine how DDF_debris will be included

# ----- Input required for ice thickness estimates
option_glaciervolume = 1
# Option 1 (default) - ice thickness estimates provided by Matthias Huss

# Option 2 - volume-length scaling
#   V_init = c_v * Area_init ^ VA_constant_exponent
#   where L is the change in
#   Need to define c_l and q, which are volume length scaling constants
""" NEED TO CODE VOLUME-LENGTH SCALING """


#------- INPUT FOR STEP THREE ------------------------------------------------
# STEP THREE: Climate Data
#   The user has the option to choose the type of climate data being used in the
#   model run, and how that data will be downscaled to the glacier and bins.
option_gcm_downscale = 1
# OPTION 1 (default): NEAREST NEIGHBOR
    # Thoughts on 2017/08/21:
    #   > Pre-processing functions should be coded and added after the initial
    #     import such that the initial values can be printed if necessary.
    #   > Data imported here is monthly, i.e., it is 1 value per month. If the
    #     data is going to be subsampled to a daily resolution in order to
    #     estimate melt in areas with low monthly mean temperature as is done in
    #     Huss and Hock (2015), then those calculations should be performed in
    #     the ablation section.

# OPTION 1: Nearest neighbor to select climate data
gcm_filepath_var = os.path.dirname(__file__) + '/../Climate_data/ERA_Interim/'
# File path to directory where the gcm data is located
# Note: _var refers to data associated with a variable and NG refers to New
#       Generation of CMIP5 data, i.e., a homogenized dataset
gcm_filepath_fx = os.path.dirname(__file__) + '/../Climate_data/ERA_Interim/'
# File path to directory where the gcm data is located
# Note: _fx refers to time invariant (constant) data
gcm_temp_filename = 'ERAInterim_AirTemp2m_DailyMeanMonthly_1995_2016.nc'
#  netcdf files downloaded from cmip5-archive at ethz or ERA-Interim reanalysis data (ECMWF)
gcm_prec_filename = 'ERAInterim_TotalPrec_DailyMeanMonthly_1979_2017.nc'
gcm_elev_filename = 'ERAInterim_geopotential.nc'
gcm_temp_varname = 't2m'
#  variable name for temperature in the GCM 
#  ('t2m' for ERA Interim, 'tas' for CMIP5)
gcm_prec_varname = 'tp'
#  variable name for precipitation in the GCM 
#  ('tp' for ERA Interim, 'pr' for CMIP5)
gcm_elev_varname = 'z'
#  variable name for model surface altitude in the GCM 
#  ('z' for ERA Interim, 'orog' for CMIP5)
gcm_lat_varname = 'latitude'
#  variable name for latitude in the GCM
#  ('latitude' for ERA Interim, 'lat' for CMIP5)
gcm_lon_varname = 'longitude'
#  variable name for longitude in the GCM
#  ('longitude' for ERA Interim, 'lon' for CMIP5)
gcm_time_varname = 'time'
#  variable name for longitude in the GCM

## Details for CMIP5 data, so can test model changing between CMIP5 and ERA-Interim quickly
#gcm_filepath_var = os.path.dirname(__file__) + '/../Climate_data/cmip5/rcp85_r1i1p1_monNG/'
#gcm_filepath_fx = os.path.dirname(__file__) + '/../Climate_data/cmip5/rcp85_r0i0p0_fx/'
#gcm_temp_filename = 'tas_mon_MPI-ESM-LR_rcp85_r1i1p1_native.nc'
#gcm_prec_filename = 'pr_mon_MPI-ESM-LR_rcp85_r1i1p1_native.nc'
#gcm_elev_filename = 'orog_fx_MPI-ESM-LR_rcp85_r0i0p0.nc'
#gcm_temp_varname = 'tas'
#gcm_prec_varname = 'pr'
#gcm_elev_varname = 'orog'
#gcm_lat_varname = 'lat'
#gcm_lon_varname = 'lon'
#gcm_time_varname = 'time'

#------- INPUT FOR STEP FOUR -------------------------------------------------
# STEP FOUR: Glacier Evolution
#   Enter brief description of user options here.

# lapse rate (K m-1) for gcm to glacier
lr_gcm = -0.0065
# lapse rate (K m-1) on glacier for bins
lr_glac = -0.0065
# precipitation correction factor (-)
prec_factor = 0.3
#   k_p in Radic et al. (2013)
#   c_prec in Huss and Hock (2015)
prec_grad = 0.0001
# precipitation gradient on glacier (% m-1)
DDF_ice = 7.2 * 10**-3
# DDF ice (m w.e. d-1 degC-1)
# note: '**' means to the power, so 10**-3 is 0.001
DDF_snow = 4.0 * 10**-3
# DDF snow (m w.e. d-1 degC-1)
T_snow = 0
# temperature threshold for snow (C)
#   Huss and Hock (2015) T_snow = 1.5 deg C with +/- 1 deg C for ratios
DDF_firn = np.mean([DDF_ice, DDF_snow])
# DDF firn (m w.e. d-1 degC-1)
# DDF_firn is average of DDF_ice and DDF_snow (Huss and Hock, 2015)
DDF_debris = DDF_ice
# DDF debris currently equivalent to ice

option_elev_ref_downscale = 'Zmed'
# Option 1 (default) - 'Zmed', median glacier elevation
# Option 2 - 'Zmax', maximum glacier elevation
# Option 3 - 'Zmin', minimum glacier elevation (terminus)
option_surfacetype = 1
# How is surface type considered, annually?
option_surfaceablation = 1
# Option 1 (default) - DDF for snow, ice, and debris
option_accumulation = 1
# Option 1 (default) - Single threshold (<= snow, > rain)
# Option 2 - single threshold +/- 1 deg uses linear interpolation
option_refreezing = 2
# Option 1 (default) - refreezing according to Huss and Hock (2015)
# Option 2 - annual potential refreezing according to Woodward et al. (1997)
refreeze_month = 10
# refreeze_month is the month where the refreeze is included
option_prec2bins = 1
# Option 1 (default) - use of a precipitation bias factor to adjust GCM
#                      value and a precipitation gradient on the glacier
# Option 2 (need to code) - Huss and Hock (2015), exponential limits, etc.
option_temp2bins = 1
# Option 1 (default) - lr_gcm and lr_glac to adjust temperature from gcm to
#                      the glacier reference (default: median), and then
#                      use the glacier lapse rate to adjust temperatures
#                      on the glacier for each bin.
# Option 2 (no other options at this moment)
option_melt_model = 1
# Option 1 (default) DDF
option_massredistribution = 1
# Option 1 (default) - Mass redistribution based on Huss and Hock (2015),
#                      i.e., volume gain/loss redistributed over the glacier
#                      using empirical normalized ice thickness change curves
option_glaciershape = 3
# glacier cross-sectional shape
# Option 1(default) - parabolic (used by Huss and Hock, 2015)
# Option 2 - rectangular
# Option 3 - triangular
icethickness_surgethreshold = 5
# ice thickness threshold for what defines a surge, i.e., adding a new elevation bin
#  Huss and Hock (2015) use a threshold of 5 m
terminus_percentage = 20
# percentage of glacier that is used to define the terminus for surges
#  Huss and Hock (2015) use 20% to calculate new area and ice thickness
#------- INPUT FOR STEP FOUR -------------------------------------------------
# STEP FIVE: Output
netcdf_filenameprefix = 'PyGEM_output_rgiregion'
netcdf_filepath = '../Output/'