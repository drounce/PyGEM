(model_structure_and_workflow_target)=
# Model Structure and Workflow
The model is written in Python. The model is currently available as two repositories on github:

1. **PyGEM** ([https://github.com/drounce/PyGEM](https://github.com/drounce/PyGEM)) contains the main model code.  This repository can now be installed in your environment with PyPI ([https://pypi.org/project/pygem/](https://pypi.org/project/pygem/)) using "pip install pygem".

2. **PyGEM-scripts** ([https://github.com/drounce/PyGEM-scripts](https://github.com/drounce/PyGEM-scripts)) contains general scripts used to run the model (e.g., run_calibration.py, run_simulation.py) as well as the post-processing, analysis scripts (e.g., analyze_simulations.py). To use these files and run the model one must clone the github repository onto your local machine.

All input parameters are specified in the pygem_input.py file, which needs to be adjusted according to the specific datasets and model options of each user. The input file is well documented inline and [sample files](https://drive.google.com/drive/folders/13kiU00Zz2swN5OzwXiWIQTj_JLEHnDgZ) are produced to support trial runs.

## Directory structure
Currently, the model does not have a “required” set of directories. For simplicity with github, we highly recommend keeping the forked version of the code in its own directory. Furthermore, since many of the datasets that will be used for regional and global model runs are of considerable size, we encourage users to develop their own organized file structure. The code has been developed to automatically set up all file paths using relative paths from the PyGEM-Scripts directory, which is where the code is run from the command line. For example, a directory with the following subdirectories is recommended (see [sample files](https://drive.google.com/drive/folders/13kiU00Zz2swN5OzwXiWIQTj_JLEHnDgZ)):

* climate_data
  - This directory contains the reference and future climate data
* debris_data (optional)
  - This directory contains data concerning the debris thickness and sub-debris melt enhancement factors. 
* DEMs
  - This directory contains the geodetic mass balance data derived from DEM differencing that is used for calibration.
* IceThickness_Farinotti (optional)
  - This directory includes the consensus ice thickness estimates (Farinotti et al. 2019). The directory is optional unless you want to match the consensus ice thickness estimates.
* oggm_gdirs
  - This directory contains the glacier directories downloaded from OGGM. This directory will be automatically generated by the pre-processing steps from OGGM.
* Output
  - This directory contains all model output
* PyGEM-scripts
  - This directory contains scripts to run the model and process output
* RGI
  - This directory contains the RGI glacier information
* WGMS (optional)
  - This directory contains the WGMS mass balance data for validation. The directory is optional in case you prefer to validate your model with different data.

## Model Code Workflow
The model code itself is heavily commented with the hope that the code is easy to follow and develop further. Broadly speaking, the current steps include:
* Pre-processing any required data <em>(optional)</em>
For example, the following script corrects the geodetic mass balance from [Hugonnet et al. (2021)](https://www.nature.com/articles/s41586-021-03436-z) to account for frontal ablation from [Kochtitzky et al. (2022)](https://www.nature.com/articles/s41467-022-33231-x).
```
python run_preprocessing_wgms_mbdata.py -mb_data_removeFA=1
```

* Setting up **pygem_input.py** <br>This requires the user to state the glaciers/regions to model; model physics, calibration, and simulation options; relative filepaths for relevant datasets; etc.
* Calibrating frontal ablation parameter <em>(optional)</em> using **run_calibration_frontalablation.py** <br>This will calibrate the frontal ablation model parameter for marine-terminating glaciers; however, multiple steps are required including the following:
```
python run_calibration_frontalablation.py   (set option_merge_data = True)
python run_calibration_frontalablation.py   (set option_ind_calving_k = True)
python run_calibration_frontalablation.py   (set option_merge_calving_k = True)
python run_calibration_frontalablation.py   (set option_update_mb_data = True)
```
```{note}
The run_calibration_frontalablation.py script is hard-coded with True/False options so one must manually go into the script and adjust the options. 
```
* Calibrating the model parameters using **run_calibration.py** <br>This will calibrate model parameters based on the calibration option and mass balance data specified in **pygem_input.py**. If using an option besides the Markov Chain Monte Carlo (MCMC) methods, then run the following:
```
python run_calibration.py
```
If using the Markov Chain Monte Carlo (MCMC) methods, then multiple steps are required. First, set the <em>option_calibration = ‘emulator’</em> in **pygem_input.py**. This creates an emulator that helps speed up the simulations within the MCMC methods and helps generate an initial calibration to generate the regional priors. The regional priors are then determined by running the following:
```
python run_mcmc_prior.py
```
Once the regional priors are set, the MCMC methods can be performed.  Change the <em>option_calibration = ‘MCMC’</em> in **pygem_input.py**, then run the following:
```
python run_calibration.py
```
* Calibrate the ice viscocity (glen_a) model parameter such that the ice volume estimated using the calibrated mass balance gradients are consistent with the consensus ice volume estimates ([Farinotti et al. 2019]((https://www.nature.com/articles/s41561-019-0300-3))) for each RGI region by running the following:
```
python run_calibration_icethickness_consensus.py
```
* Run model simulations for historical or future climate scenarios. The default will be to run the model with the reference data (e.g., ERA5). <br><em>**Historical simulations**</em> are currently performed without evolving the glacier geometry; thus, <em>option_dynamics = None</em> in **pygem_input.py** and the <em>ref_startyear</em> and <em>ref_endyear</em> will be used to set the length of the simulation. The simulation can then be run using the following:
```
python run_simulation.py
```
<em>**Future simulations**</em> require specifying a GCM and scenario, which is passed to the script through the argument parser. For example, the following will run a simulation for CESM2 SSP2-4.5:
```
python run_simulation.py -gcm_name='CESM2' -scenario='ssp245'
```
```{note}
For future simulations, at a minimum the user should specify the dynamical option (<em>option_dynamics</em>), start year (<em>gcm_startyear</em>), end year (<em>gcm_endyear</em>), bias correction option (<em>option_bias_adjustment</em>).
```
* Post-process the data. <br>There are currently several scripts available to process the datasets (e.g., merge them into regional files, create multi-GCM means and standard deviations for each SSP). While these scripts are well documented in line, they still need to be cleaned up for more widespread use.  An example:
```
python process_simulation.py
```
* Share your data with beautiful figures. <br>All users will analyze PyGEM in different ways; however, we aim to provide some general scripts to produce publication-quality figures of mass, area, and runoff change such as those within the analysis_Science_figs.py. Figure options and pathways will be hard-coded within these scripts for the present moment. An example:
```
python analysis_Science_figs.py
```