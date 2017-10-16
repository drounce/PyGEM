# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 09:04:46 2017

@author: David Rounce

pygemfxns_output.py is a list of functions that are used for post-processing the model results.  Specifically, these 
functions are meant to interact with the main model output to extract things like runoff, ELA, SLA, etc., which will
be specified by the user.  This allows the main script to run as quickly as possible and record only the minimum amount
of model results.
"""
#========= LIST OF PACKAGES ==================================================
import pandas as pd
import numpy as np
import xarray as xr
#========= IMPORT COMMON VARIABLES FROM MODEL INPUT ==========================
from pygem_input import *
    # import all data

#========= DESCRIPTION OF VARIABLES (alphabetical order) =====================
#  GlacNo - glacier number associated with the model run.
#     > the user can link this to the RGIID using the main_glac_rgi table
#  netcdf_filename - filename of the netcdf that contains the model results
#  output_interval - 'monthly', 'seasonal', or 'annual'
#     > for seasonal, user specifies the start of the winter and summer months
#     > for annual, user specifies whether using water or calendar year

#========= FUNCTIONS (alphabetical order) ===================================
def runoff_glacier(netcdf_filename, output_interval, GlacNo):
    """
    Compute the monthly, seasonal, or annual runoff for a given glacier based on the data from the netcdf output.
    """
    # Function Options:
    #  > output_interval = 'monthly', 'seasonal', or 'annual'
    #
    # Open the netcdf containing the model output
    #  (using "with" is a clean way of opening and closing the file automatically once it's no longer being used)
    with xr.open_dataset(netcdf_filename) as ds:
    #     print(output_dataset['time'])
        # Select data required to compute runoff
        melt_snow = pd.DataFrame(ds['melt_snow_bin_monthly'][GlacNo,:,:].values, index=ds['binelev'][:], 
                                 columns=ds['time'][:])
        melt_surf = pd.DataFrame(ds['melt_surf_bin_monthly'][GlacNo,:,:].values, index=ds['binelev'][:], 
                                 columns=ds['time'][:])
        refreeze = pd.DataFrame(ds['refreeze_bin_monthly'][GlacNo,:,:].values, index=ds['binelev'][:], 
                                columns=ds['time'][:])
        prec = pd.DataFrame(ds['prec_bin_monthly'][GlacNo,:,:].values, index=ds['binelev'][:], 
                            columns=ds['time'][:])
        glac_bin_area = pd.DataFrame(ds['area_bin_annual'][GlacNo,:,:].values, index=ds['binelev'][:], 
                                     columns=ds['year'][:])
    # Compute runoff
    runoff_monthly = pd.DataFrame(0, index=[GlacNo], columns=melt_snow.columns)
    # Loop through each time step, since glacier area is given as an annual variable
    if timestep == 'monthly':
        for step in range(melt_snow.shape[1]):
            col = runoff_monthly.columns.values[step]
            runoff_monthly.loc[GlacNo, col] = ((melt_snow.loc[:, col] + melt_surf.loc[:, col] - refreeze.loc[:, col] + 
                                                prec.loc[:, col]).multiply(glac_bin_area.iloc[:, step//12] * 1000**2)
                                                .sum())
            #  runoff = Sum of [(melt_surface + melt_snow - refreeze + precipitation) * Area_bin]
            #  units: m * km**2 * (1000 m / 1 km)**2
            #  int(step/12) enables the annual glacier area to be used with the monthly time step
    else:
        print('Runoff for daily timestep has not been developed yet. Runoff was unable to be computed.')
    # Groupby desired output_interval
    if output_interval == 'monthly':
        runoff_output = runoff_monthly
    elif output_interval == 'annual':
        if timestep == 'monthly':
            runoff_output = runoff_monthly.groupby(np.arange(runoff_monthly.shape[1]) // 12, axis=1).sum()
        elif timestep == 'daily':
            print('\nError: need to code the groupbyyearsum for daily timestep for computing annual output products.'
                  'Exiting the model run.\n')
            exit()
        runoff_output.index = [GlacNo]
        runoff_output.columns = glac_bin_area.columns
    elif output_interval == 'seasonal':
        # Determine seasons based on summer and winter start months
        ds_month = pd.Series(runoff_monthly.columns.values).dt.month
        ds_season = (pd.Series(runoff_monthly.columns.values).dt.month
                     .apply(lambda x: 'summer' if (x >= summer_month_start and x < winter_month_start) else 'winter'))
        ds_wateryear = pd.Series(runoff_monthly.columns.values).dt.year
        runoff_monthly_copy = runoff_monthly.transpose()
        runoff_monthly_copy = runoff_monthly_copy.reset_index()
        runoff_monthly = pd.concat([runoff_monthly_copy, ds_month, ds_season, ds_wateryear], axis=1)
        runoff_monthly.columns = ['time','runoff', 'month', 'season', 'wateryear']
        # if water year is being used, then convert the years to water years
        if option_wateryear == 1:
            runoff_monthly['wateryear'] = (runoff_monthly['wateryear']
                                           .where(runoff_monthly['month'] < wateryear_month_start, 
                                                  runoff_monthly['wateryear'] + 1))
            #  df.where is counterintuitive compared to np.where as df.where(condition, value_false)
            #  Hence, this is the same as where('month' > wateryear_month_start) add 1 so that it's with the next year
        # Create empty seasonal dataframe
        runoff_output = pd.DataFrame(0, index=['summer', 'winter'], columns=glac_bin_area.columns)
        # Compute the seasonal runoff for each year
        for yearstep in range(glac_bin_area.columns.values.shape[0]):
            # Track the water years
            wateryear = glac_bin_area.columns.values[0] + yearstep
            # Select the yearly subset in order to use groupby 
            runoff_subset = runoff_monthly.loc[runoff_monthly['wateryear'] == wateryear]
            # Use groupby to sort the yearly subset by the seasons
            runoff_subset_seasonal = runoff_subset.groupby('season')['runoff'].sum()
            # Record the seasonal runoffs for each year
            runoff_output.loc[:,wateryear] = runoff_subset_seasonal
    return runoff_output