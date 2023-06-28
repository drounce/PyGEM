(run_calibration_icethickness_overview_target)=
# run_calibration_icethickness.py
This script will calibrate the ice viscosity ("Glen A") model parameter such that the modeled ice volume roughly matches the ice volume estimates from [Farinotti et al. (2019)](https://www.nature.com/articles/s41561-019-0300-3) for each RGI region.

If successful, the script will run without error and output the following:
* ../Output/calibration/‘glena_region.csv’ 

## Script Structure
The ice thickness calibration is currently hard-coded for the user to specify the regions, and the initial, upper and lower bounds for the Glen's A multiplier, which is the parameter to be calibrated.

Broadly speaking, the script follows:
* Load glaciers
* Select subset of glaciers to reduce computational expense
* Load climate data
* Run mass balance and invert for initial ice thickness
* Use minimization to find agreement between our modeled and [Farinotti et al. (2019)](https://www.nature.com/articles/s41561-019-0300-3) modeled ice thickness estimates for each RGI region
* Export the calibrated parameters.

## Special Considerations
* This code is currently not set up to run automatically as it has the regions hard-coded within the script. The reason for this hard-coding is to be able to run the script, while other scripts are running too (e.g., calibration). This should be changed in the future to facilitate automation though.
* In  pygem_input.py, you need to set option_dynamics=‘OGGM’. Otherwise, you’ll get an assertion error telling you to do so, since you won’t be able to record the output otherwise.
* While the code runs at the RGI Order 1 region scale, it will only calibrate for the glaciers that have calibration data and run successfully.
* pygem_input.py has a parameter ‘icethickness_cal_frac_byarea’ that is used to set the fraction of glaciers by area to include in this calibration. The default is 0.9 (i.e., the largest 90% of glaciers by area). This is to reduce computational expense, since the smallest 10% of glaciers by area contribute very little to the regional volume.