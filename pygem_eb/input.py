# Built-in libraries
import os
# External libraries
import numpy as np
import pandas as pd
import pygem.oggm_compat as oggm

debug=True          # Print monthly outputs?
store_data=False     # Save file?
new_file=True       # Write to scratch file?

# ========== USER OPTIONS ========== 
glac_no = ['01.00570']
n_bins = 1              # Number of elevation bins
parallel = False        # Run parallel processing?
store_vars = ['MB','EB','Temp','Layers']  # Variables to store of the possible set: ['MB','EB','Temp','Layers']
storage_freq = 'H'      # Frequency to store data using pandas offset aliases

# ========== GLACIER INFO ========== 
glac_props = {'01.00570':{'name':'Gulkana',
                            'AWS_fn':'Preprocessed/gulkanaD/gulkana_merra2.csv', # gulkanaD_wERA5.csv
                            # 'AWS_elev':1854,
                            'AWS_elev':1546, # 1854 is D, B is 1693, AB is 1546
                            'init_filepath':''},
            '01.01104':{'name':'Lemon Creek',
                            'AWS_fn':'LemonCreek1285_hourly.csv'},
            '01.16195':{'name':'South',
                            'AWS_elev':2280,
                            'AWS_fn':'Preprocessed/south/south2280_hourly_2008_wNR.csv'},
            '08.00213':{'name':'Storglaciaren',
                            'AWS_fn':'Storglaciaren/SITES_MET_TRS_SGL_dates_15MIN.csv'},
            '11.03674':{'name':'Saint-Sorlin',
                            'AWS_elev':2720,
                            'AWS_fn':'Preprocessed/saintsorlin/saintsorlin_hourly.csv'},
            '16.02444':{'name':'Artesonraju',
                            'AWS_elev':4797,
                            'AWS_fn':'Preprocessed/artesonraju/Artesonraju_hourly.csv'}}
gdir = oggm.single_flowline_glacier_directory(glac_no[0], logging_level='CRITICAL',has_internet=False)
all_fls = oggm.get_glacier_zwh(gdir)
fls = all_fls.iloc[np.nonzero(all_fls['h'].to_numpy())] # remove empty bins
bin_indices = np.linspace(len(fls.index)-1,0,n_bins,dtype=int)
bin_elev = fls.iloc[bin_indices]['z'].to_numpy()
bin_ice_depth = fls.iloc[bin_indices]['h'].to_numpy()

# FORCE MANUAL BECAUSE OGGM ISNT WORKING******
# bin_elev = np.array([1546])
# bin_ice_depth = np.array([200])

# ========== DIRECTORIES AND FILEPATHS ========== 
main_directory = os.getcwd()
output_filepath = main_directory + '/../Output/'
output_sim_fp = output_filepath + 'simulations/'
model_run_date = str(pd.Timestamp.today()).replace('-','_')[0:10]
glac_name = glac_props[glac_no[0]]['name']

# Define output filepath
output_name = f'{output_filepath}EB/{glac_name}_{model_run_date}_'
if new_file:
    i = '0'
    while os.path.exists(output_name+str(i)+'.nc'):
        i = int(i) + 1
    output_name = output_name + str(i)
else:
    output_name = output_name+'scratch'
# output_name = f'{output_filepath}EB/{glac_name}_{model_run_date}_base'

# Define input filepaths
glac_no_str = str(glac_no[0]).replace('.','_')
init_filepath = main_directory + f'/pygem_eb/sample_init_data/{glac_no_str}.nc'
grainsize_fp = main_directory + '/pygem_eb/data/drygrainsize(SSAin=60).nc'
initial_temp_fp = main_directory + '/pygem_eb/sample_init_data/gulkanaBtemp.csv'
initial_density_fp = main_directory + '/pygem_eb/sample_init_data/gulkanaBdensity.csv'
snicar_input_fp = main_directory + '/biosnicar-py/src/biosnicar/inputs.yaml'

# ========== CLIMATE AND TIME INPUTS ========== 
climate_input = 'AWS' # GCM or AWS
if climate_input in ['AWS']:
    AWS_fp = main_directory + '/../climate_data/AWS/'
    AWS_fn = AWS_fp+glac_props[glac_no[0]]['AWS_fn']
    glac_name = glac_props[glac_no[0]]['name']
    n_bins = 1
    bin_elev = [int(glac_props[glac_no[0]]['AWS_elev'])]
    assert os.path.exists(AWS_fn), 'Check AWS filepath or glac_no in input.py'

dates_from_data = False
if dates_from_data and climate_input in ['AWS']:
    cdf = pd.read_csv(AWS_fn,index_col=0)
    startdate = pd.to_datetime(cdf.index[0])
    enddate = pd.to_datetime(cdf.index.to_numpy()[-1])
else:
    startdate = pd.to_datetime('2023-04-21 00:00')
    enddate = pd.to_datetime('2023-08-09 00:00')
    # startdate = pd.to_datetime('2016-10-01 00:00') # weighing gage installed in 2015
    # enddate = pd.to_datetime('2018-05-01 00:00')
n_months = np.round((enddate-startdate)/pd.Timedelta(days=30))
print(f'Running {n_bins} bin(s) at {bin_elev} m a.s.l. for {n_months} months starting in {startdate.month_name()}, {startdate.year}')

#  ========== MODEL OPTIONS ========== 
# INITIALIATION
initialize_water = 'zero_w0'      # 'zero_w0' or 'initial_w0'
initialize_temp = 'interp'        # 'piecewise' or 'interp'
initialize_dens = 'interp'        # 'piecewise' or 'interp'
surftemp_guess =  -10             # guess for surface temperature of first timestep
startssn = 'endaccum'             # 'endaccum' or 'endmelt' -- sample density/temp data provided for Gulkana
initial_snowdepth = [2.2]*n_bins
initial_firndepth = [0]*n_bins
icelayers = 'multiple'

# TIMESTEP
dt = 3600                   # Model timestep [s]
dt_heateq = 3600/5          # Time resolution of heat eq [s], should be integer multiple of 3600s so data can be stored on the hour

# METHODS
method_turbulent = 'MO-similarity'      # 'MO-similarity' 
method_heateq = 'Crank-Nicholson'       # 'Crank-Nicholson'
method_densification = 'off'            # 'Boone', 'off', 'DEBAM' (broken)
method_cooling = 'iterative'            # 'minimize' (slow) or 'iterative' (fast)
method_ground = 'MolgHardy'             # 'MolgHardy'
method_percolation = 'w_LAPs'           # 'w_LAPs' or 'no_LAPs'
method_conductivity = 'OstinAndersson'  # 'OstinAndersson', 'VanDusen','Sturm','Douville','Jansson'
method_grainsizetable = 'interpolate'            # 'interpolate' (slow) or 'ML' (fast)

# CONSTANT SWITCHES
constant_snowfall_density = False
constant_conductivity = False

# ALBEDO SWITCHES
switch_snow = 1             # 0 to turn off fresh snow feedback; 1 to include it
switch_melt = 2             # 0 to turn off melt feedback; 1 for simple degradation; 2 for grain size evolution
switch_LAPs = 1             # 0 to turn off LAPs; 1 to turn on
BC_freshsnow = 1e-7         # concentration of BC in fresh snow [kg m-3]
dust_freshsnow = 2e-4       # concentration of dust in fresh snow [kg m-3]
# 1 kg m-3 = 1e6 ppb

# ========== PARAMETERS ==========
precgrad = 0.0001           # precipitation gradient on glacier [m-1]
lapserate = -0.0065         # temperature lapse rate for both gcm to glacier and on glacier between elevation bins [C m-1]
tsnow_threshold = 0         # Threshold below which snowfall occurs [C]
kp = 1                      # precipitation factor [-] 
fresh_grainsize = 54.5      # Grainsize of fresh snow [um]
albedo_ice = 0.3            # Albedo of ice [-] 
roughness_ice = 1.7         # surface roughness length for ice [mm] (Moelg et al. 2012, TC)
ksp_BC = 0.1                # meltwater scavenging efficiency of BC (from CLM5)
ksp_dust = 0.015            # meltwater scavenging efficiency of dust (from CLM5)
dz_toplayer = 0.03          # Thickness of the uppermost bin [m]
layer_growth = 0.6          # Rate of exponential growth of bin size (smaller layer growth = more layers) recommend 0.2-.6
k_ice = 2.33                # Thermal conductivity of ice [W K-1 m-1]
aging_factor_roughness = 0.06267 # effect of aging on roughness length: 60 days from 0.24 to 4.0 => 0.06267
albedo_TOD = 0              # Time of day to calculate albedo [hr]
initSSA = 80                # initial estimate of Specific Surface Area of fresh snowfall (interpolation tables)

# ========== CONSTANTS ===========
daily_dt = 3600*24          # Seconds in a day [s]
density_ice = 900           # Density of ice [kg m-3] (or Gt / 1000 km3)
density_water = 1000        # Density of water [kg m-3]
density_fresh_snow = 100    # ** For assuming constant density of fresh snowfall [kg m-3]
density_firn = 700          # Density threshold for firn
# k_ice = 2.33                # Thermal conductivity of ice [W K-1 m-1]
k_air = 0.023               # Thermal conductivity of air [W K-1 m-1] (Mellor, 1997)
lapserate_dew = -0.002       # dew point temperature lapse rate [C m-1]
Lh_rf = 333550              # Latent heat of fusion of ice [J kg-1]
gravity = 9.81              # Gravity [m s-2]
pressure_std = 101325       # Standard pressure [Pa]
temp_std = 288.15           # Standard temperature [K]
R_gas = 8.3144598           # Universal gas constant [J mol-1 K-1]
molarmass_air = 0.0289644   # Molar mass of Earth's air [kg mol-1]
Cp_water = 4184             # Isobaric heat capacity of water [J kg-1 K-1]
Cp_air = 1005               # Isobaric heat capacity of air [J kg-1 K-1]
Cp_ice = 2050               # Isobaric heat capacity of ice [J kg-1 K-1]
Lv_evap = 2.514e6           # latent heat of evaporation [J kg-1]
Lv_sub = 2.849e6            # latent heat of sublimation [J kg-1]
karman = 0.4                # von Karman's constant
density_std = 1.225         # Air density at sea level [kg m-3]
viscosity_snow = 3.7e7      # Viscosity of snow [Pa-s] 
# dz_toplayer = 0.03          # Thickness of the uppermost bin [m]
# layer_growth = 0.6          # Rate of exponential growth of bin size (smaller layer growth = more layers) recommend 0.2-.6
sigma_SB = 5.67037e-8       # Stefan-Boltzmann constant [W m-2 K-4]
max_nlayers = 20            # Maximum number of vertical layers allowed
max_dz = 1                  # Max layer height
albedo_deg_rate = 15        # Rate of exponential decay of albedo
wet_snow_C = 4.22e-13       # Constant for wet snow metamorphosis [m3 s-1]
# fresh_grainsize = 54.5      # Grainsize of fresh snow [um]
constant_grainsize = 1000   # Grainsize to treat as constant if switch_melt is 0 [um]
Sr = 0.033                  # for irreducible water content flow method
rainBC = BC_freshsnow       # concentration of BC in rain
raindust = dust_freshsnow   # concentration of dust in rain
temp_temp = -3              # temperature of temperate ice [C]
temp_depth = 100            # depth of temperate ice [m]
albedo_fresh_snow = 0.85    # Albedo of fresh snow [-] (Moelg et al. 2012, TC)
albedo_firn = 0.55          # Albedo of firn [-]
roughness_fresh_snow = 0.24 # surface roughness length for fresh snow [mm] (Moelg et al. 2012, TC)
roughness_firn = 4          # surface roughness length for firn [mm] (Moelg et al. 2012, TC)

# ========== OTHER PYGEM INPUTS ========== 
rgi_regionsO1 = [1]
rgi_regionsO2 = [2]
rgi_glac_number = 'all'

# Types of glaciers to include (True) or exclude (False)
include_landterm = True                # Switch to include land-terminating glaciers
include_laketerm = True                # Switch to include lake-terminating glaciers
include_tidewater = True               # Switch to include tidewater glaciers
ignore_calving = False                 # Switch to ignore calving and treat tidewater glaciers as land-terminating
oggm_base_url = 'https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.4/L1-L2_files/elev_bands/'
logging_level = 'DEBUG' # DEBUG, INFO, WARNING, ERROR, WORKFLOW, CRITICAL (recommended WORKFLOW)
option_leapyear = 1 # 0 to exclude leap years

# Reference period runs (runs up to present)
ref_gcm_name = 'MERRA2'        # reference climate dataset ('ERA5-hourly' or 'MERRA2')
ref_startyear = 1980                # first year of model run (reference dataset)
ref_endyear = 2019                  # last year of model run (reference dataset)
ref_wateryear = 'calendar'          # options for years: 'calendar', 'hydro', 'custom'
ref_spinupyears = 0                 # spin up years

# This is where the simulation runs climate data will be set up once we're there
# Simulation runs (refers to period of simulation and needed separately from reference year to account for bias adjustments)
gcm_startyear = 1980            # first year of model run (simulation dataset)
gcm_endyear = 2019              # last year of model run (simulation dataset)
gcm_wateryear = 'calendar'      # options for years: 'calendar', 'hydro', 'custom'
gcm_spinupyears = 0             # spin up years for simulation (output not set up for spinup years at present)
# constantarea_years = 0          # number of years to not let the area or volume change
# if gcm_spinupyears > 0:
#     assert 0==1, 'Code needs to be tested to enure spinup years are correctly accounted for in output files'