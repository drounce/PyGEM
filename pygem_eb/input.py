# Built-in libraries
import os
# External libraries
import numpy as np
import pandas as pd
import xarray as xr
# import pygem.oggm_compat as oggm

debug=True           # Print monthly outputs?
store_data=False      # Save file?
new_file=True        # Write to scratch file?

# ========== USER OPTIONS ========== 
glac_no = ['01.00570']  # List of RGI glacier IDs
parallel = False        # Run parallel processing?
n_bins = 1              # Number of elevation bins
timezone = pd.Timedelta(hours=-8)   # local GMT time zone
use_AWS = False          # Use AWS data? (or just reanalysis)

# ========== GLACIER INFO ========== 
glac_props = {'01.00570':{'name':'Gulkana',
                            'site_elev':1693,
                            'AWS_fn':'Preprocessed/gulkanaB24.csv'}, 
            '01.01104':{'name':'Lemon Creek',
                            'site_elev':1285,
                            'AWS_fn':'LemonCreek1285_hourly.csv'},
            '01.00709':{'name':'Mendenhall',
                            'site_elev':1316},
            '01.16195':{'name':'South',
                            'site_elev':2280,
                            'AWS_fn':'Preprocessed/south/south2280_2008.csv'},
            '08.00213':{'name':'Storglaciaren',
                            'AWS_fn':'Storglaciaren/SITES_MET_TRS_SGL_dates_15MIN.csv'},
            '11.03674':{'name':'Saint-Sorlin',
                            'site_elev':2720,
                            'AWS_fn':'Preprocessed/saintsorlin/saintsorlin_hourly.csv'},
            '16.02444':{'name':'Artesonraju',
                            'site_elev':4797,
                            'AWS_fn':'Preprocessed/artesonraju/Artesonraju_hourly.csv'}}

# WAYS OF MAKING BIN_ELEV
# dynamics = False
# if dynamics:
#     gdir = oggm.single_flowline_glacier_directory(glac_no[0], logging_level='CRITICAL') #,has_internet=False
#     all_fls = oggm.get_glacier_zwh(gdir)
#     fls = all_fls.iloc[np.nonzero(all_fls['h'].to_numpy())] # remove empty bins
#     bin_indices = np.linspace(len(fls.index)-1,0,n_bins,dtype=int)
#     bin_elev = fls.iloc[bin_indices]['z'].to_numpy()
#     bin_ice_depth = fls.iloc[bin_indices]['h'].to_numpy()
# bin_elev = np.array([1270,1385,1470,1585,1680,1779]) # From Takeuchi 2009
# bin_elev = np.array([1526,1693,1854])
# bin_ice_depth = np.ones(len(bin_elev)) * 200

if glac_no == ['01.00570']:
    # Gulkana runs have specific sites with associated elevation / shading
    site = 'B'
    site_fp = os.path.join(os.getcwd(),'pygem_eb/sample_data/gulkana/site_constants.csv')
    site_df = pd.read_csv(site_fp,index_col='site')
    bin_elev = [site_df.loc[site]['elevation']]
    kp = site_df.loc[site]['kp']
    slope = site_df.loc[site]['slope']
    aspect = site_df.loc[site]['aspect']
    sky_view = site_df.loc[site]['sky_view']
    initial_snowdepth = [site_df.loc[site]['snowdepth']]
    initial_firndepth = [site_df.loc[site]['firndepth']]
else:
    # Manually specify for other glaciers
    sky_view = 0.936
    bin_elev = [2280]
    kp = 1
    site = 'AWS'
    initial_snowdepth = [2.18]*n_bins   # initial depth of snow; array of length n_bins
    initial_firndepth = [0]*n_bins      # initial depth of firn; array of length n_bins
bin_ice_depth = np.ones(len(bin_elev)) * 200

assert len(bin_elev) == n_bins, 'Check n_bins in input'

# ========== DIRECTORIES AND FILEPATHS ========== 
machine = 'Torch'
main_directory = os.getcwd()
output_filepath = main_directory + '/../Output/'
output_sim_fp = output_filepath + 'simulations/'
model_run_date = str(pd.Timestamp.today()).replace('-','_')[0:10]
glac_name = glac_props[glac_no[0]]['name']

# Find new filepath 
output_name = f'{output_filepath}EB/{glac_name}_{model_run_date}_'
if new_file:
    i = '0'
    while os.path.exists(output_name+str(i)+'.nc'):
        i = int(i) + 1
    output_name = output_name + str(i)
else:
    output_name = output_name+'scratch'
# output_name = f'{output_filepath}EB/{glac_name}_{model_run_date}_BCred4'

# Define input filepaths
glac_no_str = str(glac_no[0]).replace('.','_')
grainsize_fp = main_directory + '/pygem_eb/sample_data/grainsize/drygrainsize(SSAin=60).nc'
initial_temp_fp = main_directory + '/pygem_eb/sample_data/gulkanaBtemp.csv'
initial_density_fp = main_directory + '/pygem_eb/sample_data/gulkanaBdensity.csv'
snicar_input_fp = main_directory + '/biosnicar-py/biosnicar/inputs.yaml'
shading_fp = main_directory + f'/shading/out/{glac_name}{site}_shade.csv'
temp_bias_fp = main_directory + '/pygem_eb/sample_data/gulkana/Gulkana_MERRA2_temp_bias.csv'
albedo_out_fp = main_directory + '/../Output/EB/albedo.csv'

# ========== CLIMATE AND TIME INPUTS ========== 
reanalysis = 'MERRA2' # 'MERRA2' (or 'ERA5-hourly' -- BROKEN)
temp_bias_adjust = True # adjust MERRA-2 temperatures according to bias?
MERRA2_filetag = False    # False or string to follow 'MERRA2_VAR_' in MERRA2 filename
AWS_fp = main_directory + '/../climate_data/AWS/'
AWS_fn = AWS_fp+glac_props[glac_no[0]]['AWS_fn']
glac_name = glac_props[glac_no[0]]['name']
wind_ref_height = 10 if reanalysis in ['ERA5-hourly'] else 2
if use_AWS:
    assert os.path.exists(AWS_fn), 'Check AWS filepath or glac_no in input.py'

dates_from_data = False
if dates_from_data:
    cdf = pd.read_csv(AWS_fn,index_col=0)
    cdf = cdf.set_index(pd.to_datetime(cdf.index))
    if glac_no != ['01.00570']:
        bin_elev = np.array([cdf['z'].iloc[0]])
    startdate = pd.to_datetime(cdf.index[0])
    enddate = pd.to_datetime(cdf.index.to_numpy()[-1])
    if reanalysis == 'MERRA2' and startdate.minute != 30:
        startdate += pd.Timedelta(minutes=30)
        enddate -= pd.Timedelta(minutes=30)
else:
    # startdate = pd.to_datetime('2000-05-01 00:30') 
    # enddate = pd.to_datetime('2002-07-31 23:30')
    startdate = pd.to_datetime('2023-04-20 00:30')    # Gulkana AWS dates
    enddate = pd.to_datetime('2023-08-10 00:30')
    # startdate = pd.to_datetime('2008-05-04 18:30')    # South dates
    # enddate = pd.to_datetime('2008-09-14 00:30')
    # startdate = pd.to_datetime('2016-05-11 00:30') # JIF sample dates
    # enddate = pd.to_datetime('2016-07-18 00:30')
    
n_months = np.round((enddate-startdate)/pd.Timedelta(days=30))
print(f'Running {n_bins} bin(s) at {bin_elev} m a.s.l. for {n_months} months starting in {startdate.month_name()}, {startdate.year}')

#  ========== MODEL OPTIONS ========== 
# INITIALIATION
initialize_water = 'zero_w0'        # 'zero_w0' or 'initial_w0'
initialize_temp = 'interp'          # 'piecewise', 'interp' or 'ripe' (all temps=0)
initialize_dens = 'interp'          # 'piecewise' or 'interp'
surftemp_guess =  -10               # guess for surface temperature of first timestep
if 6 < startdate.month < 9:         # initialize without snow
    initial_snowdepth = np.array([0]*n_bins).ravel()

# OUTPUT
store_vars = ['MB','EB','Temp','Layers']  # Variables to store of the possible set: ['MB','EB','Temp','Layers']
store_bands = False     # Store spectral albedo .csv
store_climate = False   # Store climate dataset .nc

# TIMESTEP
dt = 3600                   # Model timestep [s]
dt_heateq = 3600/5          # Time resolution of heat eq [s], should be integer multiple of 3600s so data can be stored on the hour

# METHODS
method_turbulent = 'MO-similarity'      # 'MO-similarity' or 'BulkRichardson' 
method_heateq = 'Crank-Nicholson'       # 'Crank-Nicholson'
method_densification = 'Boone'          # 'Boone', 'HerronLangway', 'Kojima'
method_cooling = 'iterative'            # 'minimize' (slow) or 'iterative' (fast)
method_ground = 'MolgHardy'             # 'MolgHardy'
method_conductivity = 'OstinAndersson'  # 'OstinAndersson', 'VanDusen','Sturm','Douville','Jansson'
# method_grainsizetable = 'interpolate' # unused

# CONSTANT SWITCHES
constant_snowfall_density = False       # False or density in kg m-3
constant_conductivity = k_ice = 2.33       # False or conductivity in W K-1 m-1
constant_freshgrainsize = False          # False or grain size in um (54.5 is standard)
constant_drdry = 1e-4                  # False or dry metamorphism grain size growth rate [um s-1] (1e-4 seems reasonable)

# ALBEDO SWITCHES
switch_snow = 1             # 0 to turn off fresh snow feedback; 1 to include it
switch_melt = 2             # 0 to turn off melt feedback; 1 for simple degradation; 2 for grain size evolution
switch_LAPs = 1             # 0 to turn off LAPs; 1 to turn on
if switch_snow + switch_melt + switch_LAPs < 4:
    snow_on = 'ON' if switch_snow == 1 else 'OFF'
    melt_on = 'ON' if switch_melt == 2 else 'OFF'
    LAPs_on = 'ON' if switch_LAPs == 1 else 'OFF'
    if switch_melt == 1:
        LAPs_on = melt_on = 'ON (DECAY)'
    print(f'SWITCH RUN WITH SNOW {snow_on}, MELT {melt_on} and LAPs {LAPs_on}')
    output_name = f'{output_filepath}EB/{glac_name}_{model_run_date}_{switch_snow}{switch_melt}{switch_LAPs}'

# ALBEDO BANDS
wvs = np.round(np.arange(0.2,5,0.01),2) # 480 bands used by SNICAR
band_indices = {}
for i in np.arange(0,480):
    band_indices['Band '+str(i)] = np.array([i])
grainsize_ds = xr.open_dataset(grainsize_fp)

# ========== PARAMETERS ==========
# play with
snow_threshold_low = 0      # lower threshold for linear snow-rain scaling [C]
snow_threshold_high = 1     # upper threshold for linear snow-rain scaling [C]
albedo_ice = 0.2            # albedo of ice [-] 
dz_toplayer = 0.05          # Thickness of the uppermost bin [m]
layer_growth = 0.4          # Rate of exponential growth of bin size (smaller layer growth = more layers) recommend 0.3-.6
# leave
precgrad = 0.0001           # precipitation gradient on glacier [m-1]
lapserate = -0.0065         # temperature lapse rate for both gcm to glacier and on glacier between elevation bins [C m-1]
roughness_ice = 1.7         # surface roughness length for ice [mm] (Moelg et al. 2012, TC)
ksp_BC = 0.5                  # 0.1-0.2 meltwater scavenging efficiency of BC (from CLM5)
ksp_dust = 0.015            # 0.015 meltwater scavenging efficiency of dust (from CLM5)
roughness_aging_rate = 0.1  # effect of aging on roughness length: 60 days from 0.24 to 4.0 => 0.06267
albedo_TOD = [12]           # List of time(s) of day to calculate albedo [hr] 
initSSA = 80                # initial estimate of Specific Surface Area of fresh snowfall (interpolation tables)
dep_factor = 0.5            # multiplicative factor to adjust MERRA-2 deposition
BC_freshsnow = 9e-7         # concentration of BC in fresh snow [kg m-3]
dust_freshsnow = 6e-4       # concentration of dust in fresh snow [kg m-3]
# 1 kg m-3 = 1e6 ppb = ng g-1 = ug L-1

# ========== CONSTANTS ===========
daily_dt = 3600*24          # Seconds in a day [s]
density_ice = 900           # Density of ice [kg m-3] (or Gt / 1000 km3)
density_water = 1000        # Density of water [kg m-3]
density_firn = 700          # Density threshold for firn
k_air = 0.023               # Thermal conductivity of air [W K-1 m-1] (Mellor, 1997)
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
sigma_SB = 5.67037e-8       # Stefan-Boltzmann constant [W m-2 K-4]
max_nlayers = 30            # Maximum number of vertical layers allowed
max_dz = 1                  # Max layer height
albedo_deg_rate = 15        # Rate of exponential decay of albedo
wet_snow_C = 4.22e-13       # Constant for wet snow metamorphosis [m3 s-1]
average_grainsize = 1000    # Grainsize to treat as constant if switch_melt is 0 [um]
rfz_grainsize = 1500        # Grainsize of refrozen snow [um]
Sr = 0.033                  # for irreducible water content flow method
rainBC = BC_freshsnow       # concentration of BC in rain
raindust = dust_freshsnow   # concentration of dust in rain
temp_temp = 0               # temperature of temperate ice [C]
temp_depth = 100            # depth of temperate ice [m]
albedo_fresh_snow = 0.9     # Albedo of fresh snow [-] (Moelg et al. 2012, TC - 0.85)
albedo_firn = 0.55          # Albedo of firn [-]
albedo_ground = 0.3         # Albedo of ground [-]
roughness_fresh_snow = 0.24 # surface roughness length for fresh snow [mm] (Moelg et al. 2012, TC)
roughness_firn = 4          # surface roughness length for firn [mm] (Moelg et al. 2012, TC)
ratio_BC2_BCtot = 2.08      # Ratio to transform BC bin 2 deposition to total BC
ratio_DU3_DUtot = 3         # Ratio to transform dust bin 3 deposition to total dust
ratio_DU_bin1 = 0.0834444   # Ratio to transform total dust to SNICAR Bin 1 (0.05-0.5um)
ratio_DU_bin2 = 0.19784     # " SNICAR Bin 2 (0.5-1.25um)
ratio_DU_bin3 = 0.481675    # " SNICAR Bin 3 (1.25-2.5um)
ratio_DU_bin4 = 0.203786    # " SNICAR Bin 4 (2.5-5um)
ratio_DU_bin5 = 0.034       # " SNICAR Bin 5 (5-50um)
diffuse_cloud_limit = 0.6   # Threshold to consider cloudy vs clear-sky in SNICAR
mb_threshold = 1e-3         # Threshold to consider not conserving mass

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