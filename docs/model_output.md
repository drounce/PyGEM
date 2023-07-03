(model_output_overview_target)=
# Model Output 
The model outputs a variety of data including monthly mass balance and its components (accumulation, melt, refreezing, frontal ablation, glacier runoff), and annual mass, mass below sea level, and area. Results are written as a netcdf file (.nc) for each glacier. If multiple simulations are performed (e.g., for Monte Carlo simulations), then statistics related to the median and median absolute deviation are output for each parameter. In addition to this standard glacier-wide output, binned output is also available, which include the bin\’s surface elevation, volume, thickness, and climatic mass balance annually. Additional output can be exported by modifying the **run_simulation.py** script.

## Post-processing Data
- Add examples of merge and files produced

## Analyzing Results
Various Jupyter Notebooks are available to view results. Some analyses require additional datasets (e.g., specifying watersheds), which will be described in the files.
- **analyze_glacier_change_byRGIRegion.ipynb** <br>This notebook can be used to plot glacier mass, area, and runoff changes for a given region as shown in the figure below:
```{figure} _static/analyze_glacier_change_region11.png
---
width: 100%
---
```
- **analyze_glacier_change_byWatershed.ipynb** <br>This notebook can be used to aggregate glacier mass, area, and runoff into watersheds; specifically, it will create new netcdf files per watershed such that after the initial aggregation, analyses can be performed much more rapidly. The notebook continues to show an example plot of glacier mass, area, and runoff changes for each watershed in an example region:
```{figure} _static/analyze_glacier_change_watershed11.png
---
width: 100%
---
```
```{note}
This notebook assumes that you have a "dictionary", i.e., a .csv file, that has each glacier of interest and the watershed (or other grouping) name associated with each glacier.
```
- **analyze_glacier_change_CrossSection.ipynb** <br>This notebook can be used to plot cross sections of an individual glacier's ice thickness over time for an ensemble of GCMs:
```{figure} _static/15.03733_profile_2100_ssps.png
---
width: 100%
---
```
```{note}
Want to create a gif of cross sections evolving over time instead? Check out **analyze_glacier_change_CrossSection-gif.ipynb**
```