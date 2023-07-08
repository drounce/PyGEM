(test_advanced_target)=
# Advanced Test
Here we will go over testing both the emulator and Bayesian inference.

## MCMC Calibration
Open **pygem_input.py** and check the following (changing as needed):
* glac_no = ['15.03733']
* ref_startyear = 2000
* ref_endyear = 2019
* option_calibration = 'emulator'
* option_dynamics = None

Then proceed with running the calibration:
```
python run_calibration.py -option_parallels=0
```
```{note}
Command line considerations:
<br>Look at arguments in getparser() function for additional command line options, which include options for running in parallel (i.e., we set option_parallels=0 to turn this option off), debugging, etc.
```
If successful, the script will run without errors and the following datasets will be generated:
* ../Output/calibration/15/15.03733-modelprms_dict.pkl
* ../emulator/sims/15/15.03733-100_emulator_sims.csv
* ../emulator/models/15/15.03733-emulator-mb_mwea.pth
* ../emulator/models/15/15.03733-emulator-mb_mwea_extra.pkl

These contain the calibration data, simulations used to create the emulator, and information needed to recreate the emulator.

```{note}
Normally the next step would be to run this for all glaciers in a region and then determine the prior distributions for the MCMC methods; however, given we're just testing on a single glacier, skip this step and use the default priors from the '../Output/calibration/priors.region.csv'.
```

Next, run the calibration again using the Bayesian inference. Open the **pygem_input.py** and check/change the following:
* option_calibration = 'MCMC'

Then proceed with running the calibration:
```
python run_calibration.py -option_parallels=0
```
If successful, the script will run without errors and no new calibration file will be produced. Why? Because the modelprms_dict.pkl file contains all the data. To quickly check that the MCMC was successful, the model will generate the following:
* ../Output/mcmc_success/15/15.03733-mcmc_success.txt

```{warning}
This (i.e., the run_calibration.py with the emulator and MCMC options) is where errors related to missing modules will arise. We recommend that you work through adding the missing modules and use StackOverflow to identify any additional debugging issues related to potential missing modules or module dependencies.
```

## MCMC simulations
You are now ready to run a simulation. We'll skip the simulation for the reference period and go directly to running a future simulation. Go back to **pygem_input.py** and check/change:
* gcm_startyear = 2000
* gcm_endyear = 2100
* option_dynamics = 'OGGM'
* use_reg_glena = False
* sim_iters = 50

Then proceed with running the simulation, while specifying the GCM and scenario through the command line as follows:
```
python run_simulation.py -option_parallels=0 -gcm_name='CESM2' -scenario='ssp245'
```
If successful, the script will run without errors and the following with be generated: 
* ../Output/simulation/15/CESM2/ssp245/stats/15.03733_CESM2_ssp245_MCMC_ba1_1sets_2000_2020_all.nc

This is a netcdf file that stores model output from the simulation.


CONGRATULATIONS! You are now ready to run PyGEM for your study region!