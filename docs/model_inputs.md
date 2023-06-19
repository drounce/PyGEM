(model_inputs_target)=
# Model Inputs
The minimum data required to run the model is a glacier inventory, glacier characteristics (ice thickness, hypsometry, etc.), climate data, and mass balance data for calibration ([Table 1](model_input_table_target)). The model uses glacier outlines provided by the [Randolph Glacier Inventory](https://www.glims.org/RGI/) (RGI Consortium 2017; RGI 6.0 contains 215,547 glaciers globally). For debris-covered glaciers, spatially distributed sub-debris melt enhancement factors can be used to account for the enhanced or suppressed melting depending on the debris thickness [(Rounce et al. 2021)](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2020GL091311). OGGM is used to select a digital elevation model (DEM) for each glacier and bin the data according to the glacier central flowlines. OGGM can also be used to estimate each glacier’s initial ice thickness or use existing ice thickness estimates available from the [OGGM Shop](https://docs.oggm.org/en/stable/shop.html).

For present-day (2000-2019) model runs, PyGEM currently uses monthly near-surface air temperature and precipitation data from ERA5 [(Hersbach et al. 2020)](https://rmets.onlinelibrary.wiley.com/doi/full/10.1002/qj.3803). Air temperature lapse rates are estimated using monthly air temperature data from various pressure levels. A second option is available to derive lapse rates from the neighboring pixels; however, this is for glaciers near the coast and therefore is not recommended. Additionally, the monthly temperature standard deviation is required if incorporating sub-monthly temperature variability in ablation (see below). Note that historical runs (e.g., 1980-2000) are challenging due to the lack of glacier inventories in the past; hence, one can assume the glacier area is constant or run the model in reverse (e.g., 1999, 1998, ... , 1980). However, due to nonlinearities associated with the glacier dynamics running if the model is run in reverse and then forward, the glacier areas will be different in 2000 than the initial dataset; hence, caution must be used and the results should be evaluated carefully.

For future (e.g., 2000-2100) model runs, PyGEM currently uses an ensemble of General Circulation Models (GCMs) and Shared Socioeconomic Pathways (SSPs) from the Coupled Model Intercomparison Project Phase 6 (CMIP6). The model can also be run using Representative Concentration Pathways (RCPs) associated with CMIP5. Future simulations are adjusted using additive factors for air temperature and multiplicative factors for precipitation to remove any bias between the GCMs and ERA5 data over the calibration period (2000-2019). Additional bias corrections options will be available in the future.

(model_input_table_target)=
**Table 1.** Data requirements for PyGEM. Optional datasets are shown in italics.

| Component | Dataset | References |
| :--- | :--- | :--- |
| [Glacier data](input_glacier_inventory_target) | Randolph Glacier Inventory Version 6.0 | [RGI Consortium (2017)](https://www.glims.org/RGI/) |
| [Climate data](climate_data_target) | 1: ERA5 monthly air temperature, monthly precipitation, and orography <br><em>2: ERA5 monthly lapse rates (from pressure level data) and monthly air temperature variance (from ERA5 hourly data)</em> <br>3: CMIP5 or CMIP6 monthly air temperature, monthly precipitation, and orography | [Hersbach et al. (2020)](https://rmets.onlinelibrary.wiley.com/doi/full/10.1002/qj.3803) |
| [Mass balance data](input_mb_data_target) | 1: Geodetic glacier-wide mass balance (m w.e. yr$^{-1}$) <br>2: Glaciological glacier-wide mass balance (m w.e. yr$^{-1}$) <br><em>3: All other data need to be programmed</em> | [Hugonnet et al. (2021)](https://www.nature.com/articles/s41586-021-03436-z) <br> [WGMS (2021)](https://wgms.ch/data_databaseversions/) |
| [Debris data](input_debris_data_target) (optional) | <em>Sub-debris melt factors </em> | [Rounce et al. (2021)](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2020GL091311) |
| [Frontal ablation data](input_fa_data_target) (optional) | <em>1: Frontal ablation per glacier (Gt yr-1) <br>Used to calibrate marine-terminating glaciers </em> | [Osmanoglu et al. (2013;](https://www.cambridge.org/core/journals/annals-of-glaciology/article/surface-velocity-and-ice-discharge-of-the-ice-cap-on-king-george-island-antarctica/62E511405ADD31A43FF52CDBC727A9D0) [2014)](https://tc.copernicus.org/articles/8/1807/2014/); [Minowa et al. (2021)](https://www.sciencedirect.com/science/article/pii/S0012821X21000704); [Kochtitzky et al. (2022)](https://www.nature.com/articles/s41467-022-33231-x) |
| [Ice thickness data](input_thickness_data_target) (optional) </em>| <em>Spatially distributed ice thickness data <br>Used to calibrate creep parameter to match volume of existing ice thickness dataset </em>| [Farinotti et al. (2019)](https://www.nature.com/articles/s41561-019-0300-3)|

```{note}
A sample of all relevant data to perform a test run of the model is provided [here](https://drive.google.com/file/d/159zS-oGWLHG9nzkFdsf6Uh4-w9lJSt8H/view?usp=sharing). The different sources of data required to run the model are provided below including instructions on where to download the data. We recommend reviewing the sample data to address any questions you may have concerning the structure of the datasets.
```

(input_glacier_inventory_target)=
## Glacier Inventory
The current model structure of defining the glaciers uses the Randolph Glacier Inventory version 6.0 (RGI Consortium 2017), but theoretically any glacier inventory that uses the same format and provides the same information (e.g., RGIId, Area, Terminus Type, Median elevation) would be applicable. The glacier inventory is formatted as a .csv file. The latest version of the RGI can be downloaded [here](https://www.glims.org/RGI/).

(input_glacier_data_target)=
## Glacier Data
The mass balance model can be run with information concerning glacier hypsometry; however, to account for glacier dynamics, the model requires information concerning the glacier's ice thickness as well. These data are available through the Glacier Model Intercomparison Project (GlacierMIP) (Marzeion et al. 2020) or can be derived workflows from OGGM ([Maussion et al. 2019](https://gmd.copernicus.org/articles/12/909/2019/)). We recommend using pre-processed glacier direcotry from the [OGGM Shop](https://docs.oggm.org/en/stable/shop.html).

There are several options for pre-processed data from OGGM and we recommend you read the documentation in [OGGM Shop](https://docs.oggm.org/en/stable/shop.html). The model is currently configured to use Level 2 data with a border value of 40 m using elevation bands.  OGGM will automatically download the glacier directories based on the link you specify in the input file; however, if you would like to download them in advance (e.g., if your supercomputing environment does not allow you to access the internet within the script), then you may use the following as an example of how you can download these data to your local computer from OGGM’s server:

```
wget -r --no-parent https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.4/L1-L2_files/elev_bands/RGI62/b_240/L2/ 
```

(climate_data_target)=
## Climate Data (Reference)
PyGEM requires a reference climate dataset, i.e., climate data that covers the calibration period. These data are provided as netcdf (.nc) files. The model is currently configured to use monthly data.

For reference climate data (e.g., 2000-2020), PyGEM uses ERA5, which superseded ERA-Interim due to its higher resolution and other strengths. The following data is required with optional data noted accordingly:
* Air temperature
* Precipitation (total, i.e., liquid and solid)
* Orography (geopotential)
* Air temperature lapse rates
* Air temperature daily standard deviation <em>(optional)</em>

These data can be downloaded, or are derived from downloaded data, from the [Copernicus Climate Change Service (C3S) Climate Data Store](https://cds.climate.copernicus.eu/#!/search?text=ERA5&type=dataset). Instructions for downloading these climate data may be found [here](https://confluence.ecmwf.int/display/CKB/How+to+download+ERA5). Specific detail on each dataset is provided below.

### Monthly temperature
Monthly near-surface (2 m) air temperature can be downloaded directly. This is used to calculate the positive degree days for melt, annual air temperature for refreezing, and to differentiate liquid and solid precipitation.

### Monthly precipitation
Monthly precipitation can be downloaded directly. This is used to calculate accumulation and runoff and is the total precipitation over the time period, which is currently monthly.

### Orography
Orography can be downloaded directly. This is used to account for elevation differences between the climate data pixel and the glaciers.

### Lapse rates
Monthly air temperature lapse rates are derived from monthly air temperature at each pressure level from 300 to 1000 hPa ([Huss and Hock, 2015](https://www.frontiersin.org/articles/10.3389/feart.2015.00054/full)). The air temperature at various pressure levels can be downloaded and then processed using the 'createlapserates' option from **preprocess_ecmwf_data.py** in the **PyGEM-Scripts** respoitory. These lapse rates are used to adjust the air temperature for differences in elevations between the climate pixel and the glacier as well as for various elevation bins on the glacier.

### Monthly temperature standard deviation
Monthly air temperature standard deviation is an optional product, which is only required if you plan to account for the monthly temperature variations within ablation like was done in [Huss and Hock (2015)](https://www.frontiersin.org/articles/10.3389/feart.2015.00054/full). These monthly data are derived from hourly near-surface air temperature. Hourly near-surface air temperature can be downloaded and processed using the 'createtempstd' option from **preprocess_ecmwf_data.py** in the **PyGEM-Scripts** respoitory. Note that the hourly data is quite large and therefore this step will take considerable time to download and process the data.

## Climate Data (Future)
The projected climate data has been developed using general circulation models (GCMs) from the Coupled Model Intercomparison Project Phase 5 (CMIP5), which uses Representative Concentration Pathways (RCPs) and CMIP Phase 6 (CMIP6), which uses Shared Socioeconomic Pathways (SSPs). While the use of RCPs may be of interest for comparison with previous studies, we recommend using SSPs moving forward.

The climate data required is:
* Air temperature
* Precipitation
* Orography

These data are provided as netcdf (.nc) files and can be downloaded from [OGGM’s server](https://cluster.klima.uni-bremen.de/~oggm/). 

To account for differences between the reference and future climate data, bias adjustments are made to the future temperature and precipitation data. These are described in the [Section on Bias Corrections].

(input_mb_data_target)=
## Model Calibration and Validation Data
The choice of calibration and validation data is entirely up to the modeler. However, PyGEM is currently configured to be calibrated with glacier-wide geodetic mass balance data and validated using glacier-wide glaciological data.

### Calibration data
Model parameters need to be calibrated and results should be validated using some form of mass balance (glaciological, geodetic, or gravimetric), glacier runoff, snowline, or equilibrium line altitude data ([Table 1](model_input_table_target)). Additional calibration data is required to account for frontal ablation associated with marine-terminating glaciers. We envision the model continuously incorporating new large-scale systematic datasets as they become available. 

The model was originally developed to integrate large-scale systematic glacier-wide mass balance data from 2000-2018 in High Mountain Asia [(Shean et al. 2020)](https://www.frontiersin.org/articles/10.3389/feart.2019.00363/full) and now uses a global dataset from 2000-2019 [(Hugonnet et al. 2021)](https://www.nature.com/articles/s41586-021-03436-z). The default frontal ablation data is currently from various datasets spanning 2000 to 2020 from the Northern Hemisphere [(Kochtitzky et al. 2022)](https://www.nature.com/articles/s41467-022-33231-x), South America [(Minowa et al. 2021)](https://www.sciencedirect.com/science/article/pii/S0012821X21000704), and Antarctica [(Osmanoglu et al. 2013;](https://www.cambridge.org/core/journals/annals-of-glaciology/article/surface-velocity-and-ice-discharge-of-the-ice-cap-on-king-george-island-antarctica/62E511405ADD31A43FF52CDBC727A9D0) [2014)](https://tc.copernicus.org/articles/8/1807/2014/). For model validation, annual and seasonal glaciological glacier-wide mass balance data from 1979 to 2019 have been used [(WGMS 2021)](https://wgms.ch/data_databaseversions/).

The current file structure for both the geodetic and frontal ablation data are .csv files. It's important to enter all the relevant information within the “Calibration Datasets” section of **pygem_input.py**. The key information is the glacier's RGIId, mass balance in m water equivalent (w.e.) yr$^{-1}$, mass balance error/uncertainty, initial time, end time, and area of the glacier. Processing of these data for each glacier is done in the "shop" directory under the **mbdata.py** script. We recommend reviewing the sample mass balance data for additional information on file structure.

### Validation data
The model is currently developed to use glacier-wide annual and seasonal mass balance data for validation from the World Glacier Monitoring Service ([WGMS](https://wgms.ch/data_databaseversions/)).

In addition to its use for model validation, the WGMS winter mass balance data is also used to constrain the initial calibration that is used to estimate the prior distributions for the Markov Chain Monte Carlo simulations. This processing is performed in **‘run_preprocessing_wgms_mbdata.py’** using the ‘-estimate_kp’ option.

(input_debris_data_target)=
## Debris Thickness Data
For debris-covered glaciers, spatially distributed sub-debris melt enhancement factors are required. These files are provided by [Rounce et al. (2021)](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2020GL091311) as .tif files. There are currently scripts in the “shop” directory that process the .tif files and aggregate them to elevation bins to be consistent with the glacier directory structure of OGGM. If using a different dataset, this pre-processing would need to be added.

(input_fa_data_target)=
## Frontal Ablation Data
The default frontal ablation data is currently from various datasets spanning 2000 to 2020 from the Northern Hemisphere ([Kochtitzky et al. 2022](https://www.nature.com/articles/s41467-022-33231-x)), South America ([Minowa et al. 2021](https://www.sciencedirect.com/science/article/pii/S0012821X21000704)), and Antarctic/Subantarctic ([Osmanoglu et al. 2013](https://www.cambridge.org/core/journals/annals-of-glaciology/article/surface-velocity-and-ice-discharge-of-the-ice-cap-on-king-george-island-antarctica/62E511405ADD31A43FF52CDBC727A9D0); [2014](https://tc.copernicus.org/articles/8/1807/2014/)).

(input_thickness_data_target)=
## Ice Thickness Data
The default ice thickness data is currently the "consensus" ice thickness estiamtes from [Farinotti et al. (2019)](https://www.nature.com/articles/s41561-019-0300-3). These estimates are used to calibrate the ice viscosity parameter such that the modeled ice thickness estimates roughly match the "consensus" ice thickness estimates at a regional scale.