(pygem_input_overview_target)=
# pygem_input.py
This script is where the user is able to specify the glaciers to model, choose model parameterizations and calibration options, specify relative filepaths and filenames for the model to function, and specify other model details and constants. The script is loosely organized to have the most frequently changed items at the top of the file, while also separating the file into organized chunks. The general organization is:

* [Model setup directory](input_model_setup_target)
* [Glacier selection](input_glacier_selection_target)
* [Climate data and time periods](input_climate_data_time_target)
* [Calibration options](input_cal_options_target)
* [Simulation and glacier dynamics options](input_sim_dyn_options_target)
* [Model parameters](input_model_prms_target)
* [Mass balance model options](input_mb_model_options_target)
* [Climate data filepaths and filenames](input_climate_data_files_target)
* [Glacier data](input_glacier_data_target)
* [Model time period details](input_model_time_details_target)
* [Model constants](input_model_constants_target)
* [Debugging options](input_debugging_options)

```{note}
**pygem_input.py** is heavily commented, so this information should hopefully be clear when modifying variables within the file itself.
```
(input_model_setup_target)=
## Model Setup Directory
| Variable | Format/Options | Description |
| :--- | :--- | :--- |
| main_directory | os.getcwd() | main directory used for relative filepaths |
| output_filepath | str | output filepath |
| model_run_date | str | date associated with model runs <br>(useful for knowing version of model used) |

(input_glacier_selection_target)=
## Glacier Selection
Several options exist to specify the glaciers, but they generally fall into specifying based on the RGI regions or glacier numbers. Additional options exist to include/exclude certain types of glaciers for detailed studies.

**Specify glaciers**

| Variable | Format/Options | Description |
| :--- | :--- | :--- |
| rgi_regionsO1 | list of int | 1st order region number |
| rgi_regionsO2 | list of int or 'all' | 2nd order region number ('all' means to include all subregions) |
| rgi_glac_number | list of str or 'all' | glacier number (e.g., '00001') ('all' means to include all glaciers in a given region/subregion) |
| glac_no_skip | list of str or None | glacier numbers (e.g., '1.00001') of any glaciers to exclude |
| glac_no | list or None | glacier numbers (e.g., '1.00001') of glaciers to run |

```{warning}
Set glac_no will always overwrite the rgi regions, so if you want to use the rgi region options, then set glac_no=None
```

**Specify types of glaciers**

| Variable | Format/Options | Description |
| :--- | :--- | :--- |
| include_landterm | [True, False] | switch to include land-terminating glaciers |
| include_laketerm | [True, False] | switch to include lake-terminating glaciers |
| include_tidewater | [True, False] | switch to include marine-terminating glaciers |
| ignore_calving | [True, False] | switch to ignore calving and treat marine-terminating glaciers as land-terminating |
  
**OGGM Glacier Directory Filepath**

| Variable | Format/Options | Description |
| :--- | :--- | :--- |
| oggm_base_url | str | filepath to OGGM's server with glacier directories |
| logging_level | str | logging level for OGGM. Options: DEBUG, INFO, WARNING, ERROR, WORKFLOW, CRITICAL (recommended WORKFLOW) |


(input_climate_data_time_target)=
## Climate Data and Time Periods
**Reference climate data**

| Variable | Format/Options | Description |
| :--- | :--- | :--- |
| ref_gcm_name | ['ERA5'] | reference climate dataset |
| ref_startyear | int | first year of model run (reference dataset) |
| ref_endyear | int | last year of model run (reference dataset) |
| ref_wateryear | ['calendar', 'hydro', 'custom'] | defining the calendar being used. If using custom, additional details required (see [Model Time Period Details](input_model_time_details_target)) |
| ref_spinupyears | int | number of spin up years (suggest 0) |

**Future climate data**
<br>Note that this is separate to account for bias corrections between reference and future climate data.

| Variable | Format/Options | Description |
| :--- | :--- | :--- |
| gcm_startyear | int | first year of model run |
| gcm_endyear | int | last year of model run |
| gcm_wateryear | ['calendar', 'hydro', 'custom'] | defining the calendar being used. If using custom, additional details required (see [Model Time Period Details](input_model_time_details_target) |
| gcm_spinupyears | int | number of spin up years (suggest 0) |
| constantarea_years | int | number of years to not let the area or volume change (suggest 0) |

**Hindcast options**
<br>These options will flip the climate data array so 1960-2000 would run 2000-1960 ensuring that glacier area at 2000 is correct; however, due to nonlinearities a run starting at 1960 would not provide the same mass change and area at 2000.

| Variable | Format/Options | Description |
| :--- | :--- | :--- |
| hindcast | [True, False] | switch to run hindcast simulation |


(input_cal_options_target)=
## Calibration Options
<br>These variables specify the calibration option as well as important details concerning the options or output files.

| Variable | Format/Options | Description |
| :--- | :--- | :--- |
| option_calibration | ['emulator', 'MCMC', 'MCMC_fullsim' 'HH2015', 'HH2015mod'] | calibration option |
| priors_reg_fullfn | str | filepath to where the prior distributions for the MCMC are stored. Note this is used in the run_calibration.py for MCMC as well as for the run_calibration_frontalablation.py in case preference is to use regional parameters instead of individual parameters. |

**HH2015-specific options**

| Variable | Format/Options | Description |
| :--- | :--- | :--- |
| tbias_init | float |  initial temperature bias |
| tbias_step | float | step for coarse optimization search |
| kp_init | float | initial precipitation factor |
| kp_bndlow | float | lower bound for precipitation factor |
| kp_bndhigh | float | upper bound for precipitation factor |
| ddfsnow_init | float | initial degree-day factor of snow |
| ddfsnow_bndlow | float | lower bound for degree-day factor of snow |
| ddfsnow_bndhigh | float | upper bound for degree-day factor of snow |
  
```{warning}
Huss and Hock (2015) uses a ratio of the degree-day factor of ice to snow of 2. If you want to use this same calibration framework, then you need to set ddfsnow_iceratio=0.5 in [model parameters below](input_model_prms_target)
```
    
**HH2015mod-specific options**
<br>Some variables have been described above. Only variables shown below:

| Variable | Format/Options | Description |
| :--- | :--- | :--- |
| method_opt | ['SLSQP', 'L-BFGS-B'] | SciPy optimization scheme |
| params2opt | ['tbias', 'kp'] | parameters to optimize |
| ftol_opt | float | tolerance for SciPy optimization scheme |
| eps_opt | float | epsilon (adjust variables for jacobian) for SciPy optimization scheme (0.01 works) |
    
**emulator-specific options**
<br>Some variables have been described above. Only variables shown below: 

| Variable | Format/Options | Description |
| :--- | :--- | :--- |
| emulator_sims | int |  Number of simulations to develop the emulator |
| overwrite_em_sims | [True, False] | switch to overwrite emulator simulations |
| opt_hh2015_mod | [True, False] | switch to also perform the HH2015_mod calibration using the emulator |
| emulator_fp | str | filepath to store emulator details |
| option_areaconstant | [True, False] | switch to keep area constant or evolve |

Prior distributions:

| Variable | Format/Options | Description |
| :--- | :--- | :--- |
| tbias_disttype | ['truncnormal', 'uniform'] | temperature bias prior distribution |
| tbias_sigma | float | temperature bias standard deviation for truncnormal distribution |
| kp_gamma_alpha | float | precipitation factor gamma distribution alpha |
| kp_gamma_beta | float | precipitation factor gamma distribution beta |
| ddfsnow_disttype | ['truncnormal']  | degree-day factor of snow distribution |
| ddfsnow_mu | float | degree-day factor of snow mean |
| ddfsnow_sigma | float | degree-day factor of snow standard deviation |
| ddfsnow_bndlow | float | degree-day factor of snow lower bound |
| ddfsnow_bndhigh | float | degree-day factor of snow upper bound |

**MCMC and MCMC_fullsim - specific options**
<br>Some variables have been described above. Only variables shown below:

| Variable | Format/Options | Description |
| :--- | :--- | :--- |
| tbias_stepsmall | float | temperature bias small stepsize to avoid infeasible starting set of parameters for Markov Chain |
| n_chains | int | number of chains (min 1, max 3) |
| mcmc_sample_no | int | number of steps (10000 was found to be sufficient in HMA) |
| mcmc_burn_no | int | number of steps to burn-in (0 records all steps in chain) |
| mcmc_step | [None, 'am'] | Markov Chain step option (None uses default, while am refers to adaptive metropolis algorithm) |
| thin_interval | int | thin interval if need to reduce file size (best to leave at 1 if space allows) |
| ddfsnow_start | ddfsnow_mu | degree-day factor of snow initial chain value |
| kp_disttype | ['gamma', 'lognormal', 'uniform'] | precipitation factor distribution type |
| kp_start | float | precipitation factor initial chain value |
| tbias_disttype | ['normal', 'truncnormal', 'uniform'] | temperature bias distribution type |
| tbias_start | float | temperature bias initial chain value |

**Calibration Dataset**
<br>This section will be updated as additional datasets are incorporated. For now, the following are used:

| Variable | Format/Options | Description |
| :--- | :--- | :--- |
| hugonnet_fp | str | filepath to pre-processed Hugonnet et al. (2021) data |
| hugonnet_fn | str | filename of pre-processed Hugonnet et al. (2021) data |
| hugonnet_mb_cn | str | mass balance column name |
| hugonnet_mb_err_cn | str | mass balance uncertainty column name |
| hugonnet_rgi_glacno_cn | str | RGIId or glacier number column name |
| hugonnet_mb_clim_cn | str | climatic mass balance column name |
| hugonnet_mb_clim_err_cn | str | climatic mass balance uncertainty column name |
| hugonnet_time1_cn | str | observation start time column name |
| hugonnet_time2_cn | str | observation end time column name |
| hugonnet_area_cn | str | glacier area column name |
  
**Calibration Frontal Ablation Parameter**

| Variable | Format/Options | Description |
| :--- | :--- | :--- |
| calving_fp | str | filepath to pre-processed frontal ablation parameters |
| calving_fn | str | filename of pre-processed frontal ablation parameters |
| icethickness_cal_frac_byarea | float | regional glacier area fraction that is used to calibrate the ice thickness (e.g., 0.9 means only the largest 90% of glaciers by area will be used to calibrate glen's a for that region) |


(input_sim_dyn_options_target)=
# Simulation and Glacier Dynamics Options
| Variable | Format/Options | Description |
| :--- | :--- | :--- |
| option_dynamics | ['OGGM', 'MassRedistributionCurves', None] | glacier dynamics scheme option |
    
**MCMC-specific**

| Variable | Format/Options | Description |
| :--- | :--- | :--- |
| sim_iters | int | number of simulations |
| sim_burn | int | number of burn-in (if burn-in is done in MCMC sampling, then don't do here) |

**General Parameters**

| Variable | Format/Options | Description |
| :--- | :--- | :--- |
| output_sim_fp | str | output filepath to store simulations |
| sim_stat_cns | ['mean', 'std', '2.5%', '25%', 'median', '75%', '97.5%'] | names of statistics to record for simulations with multiple parameter sets |
| export_essential_data | [True, False] | switch to export essential data (ex. mass balance components, ElA, etc.) |
| export_binned_thickness | [True, False] | switch to export binned datasets |
| export_binned_area_threshold | float | area threshold for exporting binned thicknesses |
| export_extra_vars | [True, False] | switch to export extra variables (temp, prec, melt, acc, etc.) |

**Bias Correction**

| Variable | Format/Options | Description |
| :--- | :--- | :--- |
| option_bias_adjustment | [0, 1, 2] | bias correction option (0: no adjustment, 1: new prec scheme and temp building on HH2015, 2: HH2015 methods) |

**OGGM Glacier Dynamics Parameters**

| Variable | Format/Options | Description |
| :--- | :--- | :--- |
| cfl_number | float | time step threshold (seconds) |
| cfl_number_calving | float | time step threshold (seconds) for marine-terimating glaciers |
| glena_reg_fullfn | str | full filepath and name of Glen's parameter A values |
| use_reg_glena | [True, False] | switch to use regionally calibrated Glen's parameter A |
| fs | float | sliding parameter |
| glen_a_multiplier | float | Glen's parameter A multiplier |

**Mass Redistribution Curve Parameters**

| Variable | Format/Options | Description |
| :--- | :--- | :--- |
| icethickness_advancethreshold | float | advancing glacier ice thickness change threshold (m) |
| terminus_percentage | float | precentage of glacier area considered to be terminus (used to size advancing new bins) |


(input_model_prms_target)=
## Model Parameters
| Variable | Format/Options | Description |
| :--- | :--- | :--- |
| use_calibrated_modelparams | [True, False] | switch to use calibrated model parameters |
| use_constant_lapserate | [True, False] | switch to use constant value specified below or not |
| kp | float | precipitation factor (-) |
| tbias | float | temperature bias (deg C) |
| ddfsnow | float | degree-day factor of snow (m w.e. d$^{-1}$ degC$^{-1}$) |
| ddfsnow_iceratio | float | ratio degree-day factor snow snow to ice |
| ddfice | ddfsnow / ddfsnow_iceratio | degree-day factor of ice (m w.e. d$^{-1}$ degC$^{-1}$) |
| precgrad | float | precipitation gradient on glacier (m$^{-1}$) |
| lapserate | float | temperature lapse rate for both gcm to glacier and on glacier between elevation bins (K m$^{-1}$) |
| tsnow_threshold | float | temperature threshold for snow (deg C) |
| calving_k | float | frontal ablation rate (yr$^{-1}$) |


(input_mb_model_options_target)=
## Mass Balance Model Options
**Surface Type Options**

| Variable | Format/Options | Description |
| :--- | :--- | :--- |
| option_surfacetype_initial | 1 | initial surface type option (1: median elevation distinguishes firn/ice, 2: mean elevation) |
| include_firn | [True, False] | switch to include firn or treat it as snow |
| include_debris | [True, False] | switch to account for debris with sub-debris melt factors or not |
  
  
**Downscaling Options**

| Variable | Format/Options | Description |
| :--- | :--- | :--- |
| option_elev_ref_downscale | ['Zmed', 'Zmax', 'Zmin'] | reference elevation for downscaling climate variables to glacier |
| option_temp2bins | [1] | downscale temperature to bins options (1: use lr_gcm and lr_glac to adjust temp from gcm to the glacier bins) |
| option_adjusttemp_surfelev | [0, 1] | switch to adjust temperatures based on surface elevation changes or not |
| option_prec2bins | [1] | downscale precipitation to bins (currently only based on precipitation factor and precipitation gradient) |
| option_preclimit | [0, 1] | switch to limit the uppermost 25% using an expontial function |

**Accumulation Options**

| Variable | Format/Options | Description |
| :--- | :--- | :--- |
| option_accumulation | [1, 2] | accumulation threshold option: (1) single threshold or (2) +/- 1 degree linear interpolation |

**Ablation Options**

| Variable | Format/Options | Description |
| :--- | :--- | :--- |
| option_ablation | [1, 2] | compute ablation using (1) monthly temperature or (2) superimposed daily temperatures |
| option_ddf_firn | [0, 1] | estimate degree-day factor of firn by (0) degree-day factor of snow or (2) mean of degree-day factor of snow and ice |

**Refreezing Options** (options: 'Woodward' or 'HH2015')

| Variable | Format/Options | Description |
| :--- | :--- | :--- |
| option_refreezing | ['Woodward', 'HH2015'] | compute refreezing using annual air temperatures ('Woodward') or heat conduction ('HH2015') |

Woodward-specific options:
| Variable | Format/Options | Description |
| :--- | :--- | :--- |
| rf_month | int | month to reset refreeze |

HH2015-specific options:
| Variable | Format/Options | Description |
| :--- | :--- | :--- |
| rf_layers | int | number of layers for refreezing model (8 is sufficient according to Matthias Huss and some tests) |
| rf_dz | float | layer thickness (m) |
| rf_dsc | int | number of time steps for numerical stability (3 is sufficient - Matthias) |
| rf_meltcrit | float | critical amount of melt (m w.e.) for initializing refreezing module |
| pp | float | additional refreeze water to account for water refreezing at bare-ice surface |
| rf_dens_top | float | snow density at surface (kg m$^{-3}$) |
| rf_dens_bot | float | snow density at bottom refreezing layer (kg m$^{-3}$) |
| option_rf_limit_meltsnow | [0, 1] | switch to limit the amount of refreezing to the amount of snow melt |


(input_climate_data_files_target)=
## Climate Data Filepaths and Filenames
Here is where you should add information for any new datasets as well as adding functionality to class_climate.py

**ERA5**

| Variable | Format/Options | Description |
| :--- | :--- | :--- |
| era5_fp | str | ERA5 filepath |
| era5_temp_fn |  str | ERA5 temperature filename | 
| era5_tempstd_fn |  str | ERA5 temperature daily standard deviation filename | 
| era5_prec_fn | str | ERA5 precipitation filename | 
|  era5_elev_fn |  str | ERA5 elevation filename | 
|  era5_pressureleveltemp_fn |  str | ERA5 pressure level temperature filename | 
|  era5_lr_fn |  str | ERA5 lapse rate filename | 

**CMIP5 (GCM Data)**

| Variable | Format/Options | Description |
| :--- | :--- | :--- |
| cmip5_fp_var_prefix | str | filepath prefix for CMIP5 variables (temperature, precipitation) |
| cmip5_fp_var_ending | str | filepath ending for CMIP5 variables (temperature, precipitation) |
| cmip5_fp_fx_prefix | str | filepath prefix for CMIP5 fixed variables (elevation) |
| cmip5_fp_fx_ending | str | filepath ending for CMIP5 fixed variables (elevation |

**CMIP6 (GCM Data)**

| Variable | Format/Options | Description |
| :--- | :--- | :--- |
|cmip6_fp_prefix | str | filepath prefix for CMIP6 variables |


(input_glacier_data_target)=
## Glacier Data
**Randolph Glacier Inventory**

| Variable | Format/Options | Description |
| :--- | :--- | :--- |
| rgi_fp | str | filepath to regional attributes files |
| rgi_lat_colname | str | RGI latitude column name |
| rgi_lon_colname | str | RGI longitude column name' |
| elev_colname | str | elevation column name |
| indexname | str | index column name |
| rgi_O1Id_colname | str | glacier number column name |
| rgi_glacno_float_colname | str | glacier number as float column name |
| rgi_cols_drop | list of str | column names from table to drop <br>(e.g., ['GLIMSId','BgnDate','EndDate','Status','Linkages','Name'] or []) |

**Ice Thickness Data**

| Variable | Format/Options | Description |
| :--- | :--- | :--- |
| h_consensus_fp | str | filepath to consensus ice thickness estimates |
| binsize | int | elevation bin height (m) |
| hyps_data | 'OGGM' | hypsometry dataset |
| oggm_gdir_fp | str | OGGM glacier directories filepath |
| overwrite_gdirs | [True, False] | switch to overwrite glacier directories |

**Debris Data**

| Variable | Format/Options | Description |
| :--- | :--- | :--- |
| debris_fp | str | filepath to debris data |
    
    
(input_model_time_details_target)=
## Model Time Period Details
| Variable | Format/Options | Description |
| :--- | :--- | :--- |
| option_leapyear | [0, 1] | option to (1) include leap year days or (0) exclude leap years so February always has 28 days |
| startmonthday | str | start month and day for custom calendars (e.g., '06-01') |
| endmonthday | str | end month and day for custom calendars (e.g., '05-31') |
| wateryear_month_start | int | month starting for hydrological calendar
| winter_month_start | int | first month of winter |
| summer_month_start | int | first month of summer |
| option_dates | [1, 2] | use dates from (1) date table (first of each month) or (2) dates from climate data
| timestep | 'monthly' | model timestep |


(input_model_constants_target)=
## Model Constants
| Variable | Format/Options | Description |
| :--- | :--- | :--- |
| density_ice | float | density of ice (kg m$^{-3}$) |
| density_water | float | density of water (kg m$^{-3}$) |
| area_ocean | float | area of ocean |
| k_ice | float | thermal conductivity of ice (J s$^{-1}$ K$^{-1}$ m$^{-1}$) |
| k_air | float | thermal conductivity of air (J s$^{-1}$ K$^{-1}$ m$^{-1}$) |
| ch_ice | float | volumetric heat capacity of ice (J K$^{-1}$ m$^{-3}$) |
| ch_air | float | volumetric Heat capacity of air (J K$^{-1}$ m$^{-3}$) |
| Lh_rf | float | latent heat of fusion (J kg$^{-1}$) |
| tolerance | float | model tolerance <br>(used to remove low values caused by rounding errors) |
| gravity | float | gravity (m s$^{-2}$) |
| pressure_std | float | standard pressure (Pa) |
| temp_std | float | standard temperature (K) |
| R_gas | float | universal gas constant (J mol$^{-1}$ K$^{-1}$) |
| molarmass_air | float | molar mass of Earth's air (kg mol$^{-1}$) |

(input_debugging_options)=
## Debugging Options
| Variable | Format/Options | Description |
| :--- | :--- | :--- |
| debug_refreeze | [True, False] | Used for separate debugging of the heat conduction refreezing scheme |
| debug_mb | [True, False] | Used for separate debugging of the mass balance calculations |
