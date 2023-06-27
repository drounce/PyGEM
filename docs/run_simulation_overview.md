(run_simulation_target)=
# run_simulation.py
This script will run the glacier evolution model for the reference climate data or for future climate scenarios. If successful, the script will run without errors and generate one or more netcdf files. The user has the option to export essential statistics (e.g., area, mass, runoff, mass balance components)and/or binned data (e.g., ice thickness, area, mass, mass balance). The output general output will be:
* ../Output/simulations/\[gcm_name\]/\[scenario\]/stats/\[glac_no\]...-all.nc

When running the script, the GCM and scenario need to be passed via the command line as follows:
```
python run_simulation.py -gcm_name=[insert_gcm_name] -scenario=[insert_scenario]
```

## Script Structure
While most users may just want to run the model, those interested in developing new calibration schemes should be aware of the general structure of the script.  Broadly speaking, the script follows:
* Load glaciers
* Load climate data
* Bias correct the climate data
* Load glacier data (area, etc.)
* Load model parameters
* Estimate ice thickness
* Run simulation
* Export model results

## View Output
Various netcdf files may be generated. To view the results, we recommend using xarray as follows:

```
ds = xr.open_dataset(filename)
print(ds)
```

## Special Considerations
There currently exist a series of try/except statements to ensure model runs are successful. The most common causes of failure are (i) advancing glacier exceeds the "borders" defined in OGGM's pre-processing and (ii) numerical instabilities within OGGM's dynamical model. The latter is more common for tidewater glaciers. If you want to avoid these issues, we suggest removing the try/except statements.

```{warning}
<em>sim_iters</em> in the <em>pygem_input.py</em> specifies the number of iterations. If using MCMC as the calibration option, this becomes important. If you set <em>sim_iters=1</em>, the simulation will run using the median value of each model parameter. 
```