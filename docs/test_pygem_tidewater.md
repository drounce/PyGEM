(test_tidewater_target)=
# Advanced Test - Tidewater Glaciers
Here we will go over testing the calibration of the frontal ablation parameterization. <br><br>Note that in a typical regional modeling workflow, the ice viscocity ("Glen A") model parameter would already be calibrated for the land-terminating glaciers uch that the modeled ice volume roughly matches the ice volume estimates [(see run_calibration_icethickness)](run_calibration_icethickness_overview_target). Here, we have already provided you with output files, which include this calibration.
```{warning}
The frontal ablation calibration relies on a mass balance emulator; therefore, you will need to install GPyTorch [(see Installing PyGEM)](install_pygem_target).
```

## Frontal Ablation Calibration
Open **pygem_input.py** and check the following (changing as needed):
* glac_no = ['1.03622']
* ref_startyear = 2000
* ref_endyear = 2019
* include_calving = False
* option_calibration = 'emulator'
* hugonnet_fn = 'df_pergla_global_20yr-filled.csv'
* calving_fp = main_directory + '/../calving_data/'
* calving_fn = 'frontalablation_data_test.csv'
* option_dynamics = 'OGGM'
* include_debris = False

```{note}
It may feel counterintuitive to set <em>include_calving = False</em>; however, this is needed in the initial steps to avoid circularity issues.
```
Then proceed with running an initial calibration to create the emulator:
```
python run_calibration.py -option_parallels=0
```
```{note}
Command line considerations:
<br>Look at arguments in getparser() function for additional command line options, which include options for running in parallel (i.e., we set option_parallels=0 to turn this option off), debugging, etc.
```
If successful, the script will run without errors and the following datasets will be generated:
* ../Output/calibration/01/1.03622-modelprms_dict.pkl
* ../emulator/sims/01/1.03622-100_emulator_sims.csv
* ../emulator/models/01/1.03622-emulator-mb_mwea.pth
* ../emulator/models/01/1.03622-emulator-mb_mwea_extra.pkl
* ../oggm_gdirs/per_glacier/RGI60-01/RGI60-01.03/RGI60-01.03622/*

These contain the calibration data, simulations used to create the emulator, and information needed to recreate the emulator. The glacier directory (oggm_gdirs) with relevant glacier information is created automatically as well.

<br><br>Next, run the frontal ablation calibration. Open the **pygem_input.py** and check/change the following:
* include_calving = True

Then open **run_calibration_frontalablation.py** and check/change the following:
* regions = [1]
* option_ind_calving_k = True
* calving_fp = pygem_prms.main_directory + '/../calving_data/'
* calving_fn = 'frontalablation_data_test.csv'

```{warning}
This script is currently hard-coded. We hope to change this to be automated from pygem_input.py and the command line in the future. For now, please be patient and open and modify these scripts as instructed.
```
If successful, the script will run without errors and the following datasets will be generated:
* ../calving_data/analysis/1-calving_cal_ind.csv

Then open **run_calibration_frontalablation.py** and check/change the following:
* option_ind_calving_k = False
* option_merge_calving_k = True
```
python run_calibration.py -option_parallels=0
```
If successful, the script will run without errors and the following dataset will be generated: 
* ../calving_data/analysis/all-calving_cal_ind.csv

This data simply merges all the different regions together, which is essentially pointless for a single glacier test, but useful to get into good workflow habits.

<br><br> Next, update the climatic-basal mass balance data by removing the frontal ablation from the total mass change. Open **run_calibration_frontalablation.py** and check/change the following:
* option_merge_calving_k = False
* option_update_mb_data = True

If successful, the script will run without errors and the following dataset will be generated:
* ../DEMs/Hugonnet2020/df_pergla_global_20yr-filled-facorrected.csv

CONGRATULATIONS! You are now ready to run simulations with frontal ablation included. However, now that frontal ablation is included, it's important to update the calibration data and use the latest datasets produced.

## Update datasets and recalibrate model parameters
Update the datasets in **pygem_input.py** and check/change:
* hugonnet_fn = 'df_pergla_global_20yr-filled-FAcorrected.csv'
* calving_fp = main_directory + '/../calving_data/analysis/'
* calving_fn = 'all-calving_cal_ind.csv'

Then recalibrate the model parameters by running:
```
python run_calibration.py -option_parallels=0
```

If successful, the script will run without errors and no new datasets will be generated. Why? It's because the model parameters dictionary already exists, so it will simply update the parameters within there.


## Run simulation
Then proceed with running the simulation, while specifying the GCM and scenario through the command line as follows:
```
python run_simulation.py -option_parallels=0 -gcm_name='CESM2' -scenario='ssp245'
```
If successful, the script will run without errors and the following with be generated: 
* ../Output/simulation/01/CESM2/ssp245/stats/1.03622_CESM2_ssp245_emulator_ba1_1sets_2000_2100_all.nc.nc

This is a netcdf file that stores model output from the simulation.

CONGRATULATIONS! You are now ready to run PyGEM for your study region and consider frontal ablation!