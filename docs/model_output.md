# Model Output 
The model outputs a variety of data including monthly mass balance and its components (accumulation, melt, refreezing, frontal ablation, glacier runoff), and annual mass, mass below sea level, and area. Results are written as a netcdf file (.nc) for each glacier. If multiple simulations are performed (e.g., for Monte Carlo simulations), then statistics related to the median and median absolute deviation are output for each parameter. In addition to this standard glacier-wide output, binned output is also available, which include the bin’s surface elevation, volume, thickness, and climatic mass balance annually. Additional output can be exported by modifying the **run_simulation.py** script.

## Post-processing Data
- Add examples of merge and files produced

## Analyzing Results
- Add examples of figures/plots generated (e.g., mass change and cross sections)