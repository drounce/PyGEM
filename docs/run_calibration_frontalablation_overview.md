(run_calibration_fa_overview_target)=
# run_calibration_frontalablation.py
This script will perform all pre-processing and calibration required to calibrate the frontal ablation parameterization for marine-terminating glaciers. If successful, the script will run without errors, generate numerous diagnostic plots, and most importantly, produce a .csv file with the calibration outputs.

## Script Structure
The frontal ablation calibration runs through several steps on a regional basis. First, the frontal ablation data are merged together into a single dataset.

Next, the frontal ablation parameter is calibrated for each marine-terminating glacier. The geodetic mass balance dataset is used to ensure that the frontal ablation estimates do not produce unrealistic climatic mass balance estimates. In the event that they do, a correction is performed to limit the frontal ablation based on a lower bound for the climatic mass balance based on regional mass balance data. This exports a .csv file for each region that includes the calibrated model parameters as well as numerous other columns to be able to compare the observed and modeled frontal ablation as well as the climatic mass balance.

All the frontal ablation parameters for each marine-terminating glacier in a given region are then merged together into a single file.

Lastly, the climatic-basal mass balance data is updated by removing the frontal ablation from the total mass change. This is an important step since the geodetic mass balance data do not account for the mass loss below sea level; however, some of the observed mass change is due to frontal ablation that is above the water level, which is accounted for by this script. This script will export a new "corrected" .csv file to replace the previous mass balance .csv file.


To perform the frontal ablation calibration steps outlined above, simply call the run_calibration_frontalablation python script (by default all regions are calibrated):
```
run_calibration_frontalablation  #optionally pass -rgi_region01 <region>
```

## Special Considerations
Circularity issues exist in calibrating the frontal ablation parameter as the mass balance model parameters are required to estimate the ice thickness, but the frontal ablation will affect the mass balance estimates and thus the mass balance model parameters. We suggest taking an iterative approach: calibrate the mass balance model parameters, calibrate the frontal ablation parameter, update the glacier-wide climatic mass balance, and recalibrate the mass balance model parameters.

Currently only one iteration has been used, but this could be investigated further in the future.