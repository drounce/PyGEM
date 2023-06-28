(run_calibration_fa_overview_target)=
# run_calibration_frontalablation.py
This script will perform all pre-processing and calibration required to calibrate the frontal ablation parameterization for marine-terminating glaciers. If successful, the script will run without errors, generate numerous diagnostic plots, and most importantly, produce a .csv file with the calibration outputs.

## Script Structure
The frontal ablation calibration is a hard-coded script that requires several steps. 

First, you need to change the region within the run_calibration_frontalablation.py script. 

Then merge the frontal ablation data together into a single file by setting <em>option_merge_data=True</em>. The hard-coded portion of this option is located in the if statement, so you need to go to where the if statement begins and specify the three filenames that you want to merge together. Additionally, modify the datasets to ensure the data is labeled the same for all three datasets, thereby enabling the merging of the datasets. The output is a merged .csv file that has all the data standardized. This can be considered a pre-processing step.
```
python run_calibration_frontalablation.py   (set option_merge_data = True)
```
Next, calibrate the frontal ablation parameter for each marine-terminating glacier by setting <em>option_ind_calving_k=True</em>. The hard-coded portion of this option is also lcoated in the if statement, so you need to go to where the if statement begins. Here, specify the merged calving filepath and filename as well as the geodetic mass balance dataset. The geodetic mass balance dataset is used to ensure that the frontal ablation estimates do not produce unrealistic climatic mass balance estimates. In the event that they do, a correction is performed to limit the frontal ablation based on a lower bound for the climatic mass balance based on regional mass balance data. This exports a .csv file for each region that includes the calibrated model parameters as well as numerous other columns to be able to compare the observed and modeled frontal ablation as well as the climatic mass balance.
```
python run_calibration_frontalablation.py   (set option_ind_calving_k = True)
```
Then merge all the frontal ablation parameters, which are currently in multiple regional files, into a single file by setting <em>option_merge_calving_k=True</em>.
```
python run_calibration_frontalablation.py   (set option_merge_calving_k = True)
```
Lastly, update the climatic-basal mass balance data by removing the frontal ablation from the total mass change by setting <em>option_update_mb_data=True</em>. This is an important step since the geodetic mass balance data do not account for the mass loss below sea level; however, some of the observed mass change is due to frontal ablation that is above the water level, which is accounted for by this script. This script will export a new "corrected" .csv file to replace the previous mass balance .csv file.
```
python run_calibration_frontalablation.py   (set option_update_mb_data = True)
```
```{warning}
As mentioned, the run_calibration_frontalablation.py script is hard-coded with True/False options so one must manually go into the script and adjust the options. Additionally, there is specific information within the if statements, which is provided right below the if statements.
```
```{note}
As we continue developing PyGEM, we plan to make this process automated as well and move the pre-processing to a separate script.
```

## Special Considerations
Circularity issues exist in calibrating the frontal ablation parameter as the mass balance model parameters are required to estimate the ice thickness, but the frontal ablation will affect the mass balance estimates and thus the mass balance model parameters. We suggest taking an iterative approach: calibrate the mass balance model parameters, calibrate the frontal ablation parameter, update the glacier-wide climatic mass balance, and recalibrate the mass balance model parameters.

Currently only one iteration has been used, but this could be investigated further in the future.