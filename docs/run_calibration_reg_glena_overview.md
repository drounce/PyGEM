(run_calibration_reg_glena_overview_target)=
# run_calibration_reg_glena.py
This script will calibrate the ice viscosity ("Glen A") model parameter such that the modeled ice volume roughly matches the ice volume estimates from [Farinotti et al. (2019)](https://www.nature.com/articles/s41561-019-0300-3) for each RGI region. Run the script as follows:

```
run_calibration_reg_glena -rgi_region01 <region>
```

If successful, the script will run without error and output the following:
* ../Output/calibration/‘glena_region.csv’ 

## Script Structure

Broadly speaking, the script follows:
* Load glaciers
* Select subset of glaciers to reduce computational expense
* Load climate data
* Run mass balance and invert for initial ice thickness
* Use minimization to find agreement between our modeled and [Farinotti et al. (2019)](https://www.nature.com/articles/s41561-019-0300-3) modeled ice thickness estimates for each RGI region
* Export the calibrated parameters.

## Special Considerations
* While the code runs at the RGI Order 1 region scale, it will only calibrate for the glaciers that have calibration data and run successfully.
* *~/PyGEM/config.yaml* has a parameter `calib.icethickness_cal_frac_byarea` that is used to set the fraction of glaciers by area to include in this calibration. The default is 0.9 (i.e., the largest 90% of glaciers by area). This is to reduce computational expense, since the smallest 10% of glaciers by area contribute very little to the regional volume.