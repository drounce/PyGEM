(run_calibration_icethickness_overview_target)=
# run_calibration_icethickness.py
**Insert Details**

Considerations:
* This code is currently not set up to run automatically as it has the regions hard-coded within the script. The reason for this hard-coding is to be able to run the script, while other scripts are running too (e.g., calibration). This should be changed in the future to facilitate automation though.
* In  pygem_input.py, you need to set option_dynamics=‘OGGM’. Otherwise, you’ll get an assertion error telling you to do so, since you won’t be able to record the output otherwise.
* While the code runs at the RGI Order 1 region scale, it will only calibrate for the glaciers that have calibration data and run successfully.
* pygem_input.py has a parameter ‘icethickness_cal_frac_byarea’ that is used to set the fraction of glaciers by area to include in this calibration. The default is 0.9 (i.e., the largest 90% of glaciers by area). This is to reduce computational expense, since the smallest 10% of glaciers by area contribute very little to the regional volume.

If successful, the script will run without error and output the following:
* ../Output/calibration/‘glena_region.csv’ 