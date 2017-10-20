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
def AAR_glacier(netcdf_filename, output_interval, GlacNo):
    """
    Compute the Accumulation-Area Ratio (AAR) for a given glacier
    """
    # Function Options:
    #  > output_interval = 'monthly' or 'annual'
    #
    # Open the netcdf containing the model output
    with xr.open_dataset(netcdf_filename) as ds:
        # Select data required to compute AAR
        area_annual = pd.DataFrame(ds['area_bin_annual'][GlacNo,:,:].values, index=ds['binelev'][:], 
                                   columns=ds['year'][:])
    # Compute the ELA and setup the AAR dataframe
    ELA_output = ELA_glacier(netcdf_filename, output_interval, GlacNo)
    AAR_output = pd.DataFrame(0, index=ELA_output.index, columns=ELA_output.columns)
    
    # Compute the Accumulation-Area Ratio (AAR) [%]
    for step in range(AAR_output.shape[1]):
        if output_interval == 'annual':
            series_area = area_annual.iloc[:, step]
        elif output_interval == 'monthly':
            series_area = area_annual.iloc[:, int(step // 12)]
        # use try and except to avoid errors with the ELA not being specified, which would cause error with indexing
        try:
            AAR_output.loc[GlacNo, AAR_output.columns[step]] = (1 - (np.cumsum(series_area)).divide(series_area.sum())
                .iloc[int(ELA_output.loc[GlacNo, AAR_output.columns[step]] / binsize) - 1]) * 100
            #  ELA_output.iloc[GlacNo, AAR_output.columns[step]] is the elevation associated with the ELA
            #    dividing this by the binsize returns the column position if the indexing started at 1,
            #    the "-1" accounts for the fact that python starts its indexing at 0, so
            #    ".iloc[int(ELA_output.loc[GlacNo,AAR_output.columns[step]]/binsize)-1]" gives the column of the ELA.
            #  np.cumsum gives the cumulative sum of the glacier area for the given year
            #    this is divided by the total area to get the cumulative fraction of glacier area.
            #  The column position is then used to select the cumulative fraction of glacier area of the ELA
            #    since this is the area below the ELA, the value is currently the ablation area as a decimal;
            #    therefore, "1 - (cumulative_fraction)" gives the fraction of the ablation area,
            #    and multiplying this by 100 gives the fraction as a percentage.
        except:
            # if ELA does not exist, then set AAR = -9.99
            AAR_output.loc[GlacNo, step] = -9.99
    return AAR_output


def ELA_glacier(netcdf_filename, output_interval, GlacNo):
    """
    Compute the Equlibrium Line Altitude (ELA) for a given glacier
    """
    # Function Options:
    #  > output_interval = 'monthly', 'seasonal', or 'annual'
    #  > GlacNo = 'all' or a specific GlacNo
    #
    # Open the netcdf containing the model output
    with xr.open_dataset(netcdf_filename) as ds:
        # Select data required to compute ELA
        massbal_spec_monthly = pd.DataFrame(ds['massbal_spec_bin_monthly'][GlacNo,:,:].values, index=ds['binelev'][:], 
                                    columns=ds['time'][:])
        # Set up the input mass balance data and the output dataframe
        if output_interval == 'monthly':
            ELA_output = pd.DataFrame(0, index=[GlacNo], columns=massbal_spec_monthly.columns)
            massbal_input = massbal_spec_monthly
        elif output_interval == 'annual':
            if timestep == 'monthly':
                massbal_spec_annual = massbal_spec_monthly.groupby(np.arange(massbal_spec_monthly.shape[1]) // 12, 
                                                                   axis=1).sum()
                massbal_spec_annual.columns = ds['year'].values
            elif timestep == 'daily':
                print('\nError: need to code the groupbyyearsum for daily timestep for computing annual output'
                      ' products. Exiting the model run.\n')
                exit()
            ELA_output = pd.DataFrame(0, index=[GlacNo], columns=massbal_spec_annual.columns)
            massbal_input = massbal_spec_annual
        # Loop through each timestep
        for step in range(ELA_output.shape[1]):
            # Select subset of the data based on the timestep
            series_massbal_spec = massbal_input.iloc[:, step]
            # Use numpy's sign function to return an array of the sign of the values (1=positive, -1=negative, 0=zero)
            series_ELA_sign = np.sign(series_massbal_spec)                
            # Use numpy's where function to determine where the specific mass balance changes from negative to positive
            series_ELA_signchange = np.where((np.roll(series_ELA_sign,1) - series_ELA_sign) == -2)
            #   roll is a numpy function that performs a circular shift, so in this case all the values are shifted up one 
            #   place. Since we are looking for the change from negative to positive, i.e., a value of -1 to +1, we want to 
            #   find where the value equals -2. numpy's where function is used to find this value of -2.  The ELA will be 
            #   the mean elevation between this bin and the bin below it.
            #   Example: bin 4665 m has a negative mass balance and 4675 m has a positive mass balance. The difference with 
            #            the roll function will give 4675 m a value of -2.  Therefore, the ELA will be 4670 m.
            #   Note: If there is a bin with no glacier area between the min and max height of the glacier (ex. a very steep 
            #     section), then this will not be captured.  This only becomes a problem if this bin is technically the ELA, 
            #     i.e., above it is a positive mass balance, and below it is a negative mass balance.  Using np.roll with a
            #     larger shift would be one way to work around this issue.
            # try and except to avoid errors associated with the entire glacier having a positive or negative mass balance
            try:
                ELA_output.loc[GlacNo, ELA_output.columns.values[step]] = (
                    (series_massbal_spec.index.values[series_ELA_signchange[0]][0] - binsize/2).astype(int))
                #  series_ELA_signchange[0] returns the position of the ELA. series_massbal_annual.index returns an array 
                #  with one value, so the [0] ais used to accesses the element in that array. The binsize is then used to 
                #  determine the median elevation between those two bins.
            except:
                # This may not work in three cases:
                #   > The mass balance of the entire glacier is completely positive or negative.
                #   > The mass balance of the whole glacier is 0 (no accumulation or ablation, i.e., snow=0, temp<0)
                #   > The ELA falls on a band that does not have any glacier (ex. a very steep section) causing the sign 
                #     roll method to fail. In this case, using a large shift may solve the issue.
                try:
                    # if entire glacier is positive, then set to the glacier's minimum
                    if series_ELA_sign.iloc[np.where(series_ELA_sign != 0)[0][0]] == 1:
                            ELA_output.loc[GlacNo, ELA_output.columns.values[step]] = (
                                series_ELA_sign.index.values[np.where(series_ELA_sign != 0)[0][0]] - binsize/2)
                    # if entire glacier is negative, then set to the glacier's maximum
                    elif series_ELA_sign.iloc[np.where((series_ELA_sign != 0))[0][0]] == -1:
                        ELA_output.loc[GlacNo, ELA_output.columns.values[step]] = (
                            series_ELA_sign.index.values[np.where(series_ELA_sign != 0)[0]
                                                         [np.where(series_ELA_sign != 0)[0].shape[0]-1]] + binsize/2)
                except:
                    # if the specific mass balance over the entire glacier is 0, i.e., no ablation or accumulation,
                    #  then the ELA is the same as the previous timestep
                    if series_ELA_sign.sum() == 0 and step != 0:
                        ELA_output.loc[GlacNo, ELA_output.columns.values[step]] = (
                            ELA_output.loc[GlacNo, ELA_output.columns.values[step - 1]])
                    # Otherwise, it's likely due to a problem with the shift
                    else:
                        ELA_output.loc[GlacNo, ELA_output.columns.values[step]] = -9.99
    return ELA_output


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
            #  units: m * km**2 * (1000 m / 1 km)**2 = [m**3]
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