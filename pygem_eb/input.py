# Built-in libraries
import os
# External libraries
import numpy as np
import pandas as pd
import pygem.oggm_compat as oggm

debug=True
store_data=True

#%% ===== GLACIER SELECTION =====
rgi_regionsO1 = [1]                 # 1st order region number (RGI V6.0)
rgi_regionsO2 = [2]                 # 2nd order region number (RGI V6.0)
rgi_glac_number = 'all'
glac_no = ['01.00570']

#%% ===== MODEL SETUP DIRECTORY =====
new_file=True
glac_props = {'01.00570':{'name':'Gulkana',
                            'AWS_fn':'Preprocessed/gulkanaD/gulkanaD_wERA5.csv',
                            'AWS_elev':1693, # 1854 is AWS (D), B is 1693, AB is 1546
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
                            'AWS_fn':'Preprocessed/artesonraju/Artesonraju_hourly.csv'}}

main_directory = os.getcwd()
output_filepath = main_directory + '/../Output/'
output_sim_fp = output_filepath + 'simulations/'
model_run_date = str(pd.Timestamp.today()).replace('-','_')[0:10]
glac_name = glac_props[glac_no[0]]['name']
output_name = f'{output_filepath}EB/{glac_name}_{model_run_date}_'
if new_file:
    i = '0'
    while os.path.exists(output_name+str(i)+'.nc'):
        i = int(i) + 1
    output_name = output_name + str(i)
else:
    output_name = output_name+'scratch'
glac_no_str = str(glac_no[0]).replace('.','_')
init_filepath = main_directory + f'/pygem_eb/sample_init_data/{glac_no_str}.nc'

#%% MODEL OPTIONS
n_bins = 1
parallel = False

# Set up bins
gdir = oggm.single_flowline_glacier_directory(glac_no[0], logging_level='CRITICAL')
fls = oggm.get_glacier_zwh(gdir)
fls = fls.iloc[np.nonzero(fls['h'].to_numpy())] #filter out zero bins to get only initial glacier volume
med_idx = np.where(fls['z'].to_numpy()==np.median(fls['z'].to_numpy()))[0]
bin_indices = np.linspace(len(fls.index)-1,0,n_bins,dtype=int)
bin_elev = fls.iloc[bin_indices]['z'].to_numpy()
icelayers = 'multiple'
bin_ice_depth = fls.iloc[bin_indices]['h'].to_numpy()

# Types of glaciers to include (True) or exclude (False)
include_landterm = True                # Switch to include land-terminating glaciers
include_laketerm = True                # Switch to include lake-terminating glaciers
include_tidewater = True               # Switch to include tidewater glaciers
ignore_calving = False                 # Switch to ignore calving and treat tidewater glaciers as land-terminating

oggm_base_url = 'https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.4/L1-L2_files/elev_bands/'
logging_level = 'DEBUG' # DEBUG, INFO, WARNING, ERROR, WORKFLOW, CRITICAL (recommended WORKFLOW)

#%% ===== CLIMATE DATA ===== 
# Specify dataset
climate_input = 'AWS' # GCM or AWS
if climate_input in ['AWS']:
    AWS_fp = main_directory + '/../climate_data/AWS/'
    AWS_fn = AWS_fp+glac_props[glac_no[0]]['AWS_fn']
    glac_name = glac_props[glac_no[0]]['name']
    bin_elev = [int(glac_props[glac_no[0]]['AWS_elev'])]
    assert os.path.exists(AWS_fn), 'Check AWS filepath or glac_no in input.py'
print(f'{len(bin_elev)} bins at elevations: {bin_elev} [m]')

# Dates
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
option_leapyear = 1 # 0 to exclude leap years
# Reference period runs (runs up to present)
ref_gcm_name = 'ERA5-hourly'        # reference climate dataset
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

# Filepaths
# init_filepath = main_directory + '/pygem_eb/sample_init_data/startssn_initialTp.nc'.replace('startssn',startssn)
grainsize_fp = '~/research/PyGEM-EB/pygem_eb/data/drygrainsize(SSAin=60).nc'
initial_temp_fp = '~/research/PyGEM-EB/pygem_eb/sample_init_data/gulkanaBtemp.csv'
initial_density_fp = '~/research/PyGEM-EB/pygem_eb/sample_init_data/gulkanaBdensity.csv'
snicar_input_fp = '/home/claire/research/PyGEM-EB/biosnicar-py/src/biosnicar/inputs.yaml'

# Initialization
option_initWater = 'zero_w0'            # 'zero_w0' or 'initial_w0'
option_initTemp = 'interp'           # 'piecewise' or 'interp'
option_initDensity = 'interp'        # 'piecewise' or 'interp'
startssn = 'endaccum'                    # 'endaccum' or 'endmelt' -- sample density/temp data provided for Gulkana
initial_snowdepth = [3.3]*n_bins
initial_firndepth = [0]*n_bins

# Simulation options
dt = 3600
daily_dt = 3600*24
dt_heateq = 3600/5         # Time resolution of heat eq [s], should be integer multiple of 3600s so data can be stored on the hour
method_turbulent = 'MO-similarity'  # 'MO-similarity' or *****
# option_SW
# option_LW
method_heateq = 'what' # 'Crank-Nicholson': neglects penetrating shortwave
method_densification = 'Boone'
method_cooling = 'iterative' # 'minimize' (slow) or 'iterative' (fast)
method_ground = 'MolgHardy'
method_percolation = 'w_LAPs'
method_grainsizetable = 'ML'
method_albedo = 'SNICAR'
surftemp_guess =  -10   # guess for surface temperature of first timestep

# Albedo switches
switch_snow = 1             # 0 to turn off fresh snow feedback; 1 to include it
switch_melt = 0
switch_LAPs = 0
initLAPs = [[0,0],[0,0]]    # initial LAP concentrations. Set to None to use fresh snow values
BC_freshsnow = 1e-7          # concentration of BC in fresh snow [kg m-3]. Only used if switch_LAPs is not 2
dust_freshsnow = 2e-4        # concentration of dust in fresh snow [kg m-3]. Only used if switch_LAPs is not 2

# Output
store_vars = ['MB','EB','Temp','Layers']        # Variables to store of the possible set: ['MB','EB','Temp','Layers']
storage_freq = 'H'          # frequency to store data using pandas offset aliases
vars_to_store = 'all'       # list of variables to store or 'all'

#%% MODEL PROPERTIES THAT MAY NEED TO BE ADJUSTED
precgrad = 0.0001           # precipitation gradient on glacier [m-1]
lapserate = -0.0065         # temperature lapse rate for both gcm to glacier and on glacier between elevation bins [K m-1]
lapserate_dew = -0.002      # dew point temperature lapse rate [K m-1]
tsnow_threshold = 1         # Threshold to consider freezing
kp = 1                      # precipitation factor [-] 
temp_temp = -3               # temperature of temperate ice in Celsius
depth_temp = 30
#%% MODEL PROPERTIES
density_ice = 900           # Density of ice [kg m-3] (or Gt / 1000 km3)
density_firn = 700          # Density threshold for firn
density_water = 1000        # Density of water [kg m-3]
density_fresh_snow = 100    # ** For assuming constant density of fresh snowfall [kg m-3]
k_ice = 2.33                # Thermal conductivity of ice [J s-1 K-1 m-1] recall (W = J s-1)
k_air = 0.023               # Thermal conductivity of air [J s-1 K-1 m-1] (Mellor, 1997)
Lh_rf = 333550              # Latent heat of fusion [J kg-1]
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
density_std = 1.225         # air density at sea level [kg m^-3]
albedo_fresh_snow = 0.85    # albedo of fresh snow [-] (Moelg et al. 2012, TC)
albedo_firn = 0.55          # albedo of firn [-] (Moelg et al. 2012, TC)
albedo_ice = 0.3            # albedo of ice [-] (Moelg et al. 2012, TC)
viscosity_snow = 1          # viscosity of snow Pa-s  
dz_toplayer = 0.03          # thickness of the uppermost bin [m]
layer_growth = 0.6          # rate of exponential growth of bin size (smaller layer growth = more layers) recommend 0.2-.6
sigma_SB = 5.67037e-8       # Stefan-Boltzmann constant [W m-2 K-4]
max_nlayers = 20            # maximum number of vertical layers allowed
max_dz = 1                  # max layer height
albedo_deg_rate = 30
wet_snow_C = 4.22e-13       # m3 s-1
fresh_grainsize = 300
max_pen_depth = 2           # maximum depth of shortwave penetration [m]
Sr = 0.033                  # for irreducible water content flow method
rainBC = BC_freshsnow             # concentration of BC in rain
raindust = dust_freshsnow           # concentration of dust in rain
ksp_BC = 0.1                # meltwater scavenging efficiency of BC (from CLM5)
ksp_dust = 0.015            # meltwater scavenging efficiency of dust (from CLM5)

def get_uptime():
    with open('/proc/uptime', 'r') as f:
        uptime_seconds = float(f.readline().split()[0])

    return uptime_seconds

def interpzh(fls,n_bins,bin_elev):
    print(fls['h'][0],fls['h'][np.where(fls['z']==np.median(fls['z']))[0]],fls['h'][len(fls.index)-1])
    print(fls['h'])