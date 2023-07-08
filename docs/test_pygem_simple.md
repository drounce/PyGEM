(test_simple_target)=
# Simple Test
## Simple calibration
Open **pygem_input.py** and check the following (changing as needed):
* glac_no = ['15.03733']
* ref_startyear = 2000
* ref_endyear = 2019
* option_calibration = 'HH2015'
* option_dynamics = None

Then proceed with running the calibration as follows:
```
python run_calibration.py -option_parallels=0
```
```{note}
Command line considerations:
<br>Look at arguments in getparser() function for additional command line options, which include options for running in parallel (i.e., we set option_parallels=0 to turn this option off), debugging, etc.
```
If successful, the script will run without errors and the following will be generated:
* ../Output/calibration/15/15.03733-modelprms_dict.pkl

This is a .pkl file that contains the calibration data.

## Simple present-day simulation
You are now ready to run a simulation. Go back to **pygem_input.py** and check/change:
* option_dynamics = 'OGGM'
* use_reg_glena = False

Then proceed with running the simulation for a reference time period as follows:
```
python run_simulation.py -option_parallels=0
```
If successful, the script will run without errors and the following will be generated:
* ../Output/simulation/15/ERA5/stats/15.03733_ERA5_HH2015_ba1_1sets_2000_2020_all.nc

This is a netcdf file that stores model output from the simulation.

## Simple future simulation
Now you can try simulating the glacier into the future. Got back to **pygem_input.py** and check/change:
* gcm_startyear = 2000
* gcm_endyear = 2100

Then proceed with running the simulation, while specifying the GCM and scenario through the command line as follows:
```
python run_simulation.py -option_parallels=0 -gcm_name='CESM2' -scenario='ssp245'
```

If successful, the script will run without errors and the following with be generated: 
* ../Output/simulation/15/CESM2/ssp245/stats/15.03733_CESM2_ssp245_HH2015_ba1_1sets_2000_2020_all.nc


CONGRATULATIONS! You are now ready to run PyGEM for your study region!