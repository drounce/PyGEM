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
import netCDF4 as nc
from time import strftime
from datetime import datetime
#========= IMPORT COMMON VARIABLES FROM MODEL INPUT ==========================
import pygem_input as input

#========= DESCRIPTION OF VARIABLES (alphabetical order) =====================
#  GlacNo - glacier number associated with the model run.
#     > the user can link this to the RGIID using the main_glac_rgi table
#  netcdf_filename - filename of the netcdf that contains the model results
#  output_interval - 'monthly', 'seasonal', or 'annual'
#     > for seasonal, user specifies the start of the winter and summer months
#     > for annual, user specifies whether using water or calendar year

#========= FUNCTIONS (alphabetical order) ===================================
def netcdfcreate(regionO1_number, main_glac_hyps, dates_table, annual_columns):
    """Create a netcdf file to store the desired output
    Output: empty netcdf file with the proper setup to be filled in by the model
    """
    # Netcdf file path and name
    filename = input.netcdf_filenameprefix + str(regionO1_number) + '_' + str(strftime("%Y%m%d")) + '.nc'
    fullfile = input.netcdf_filepath + filename
    # Create netcdf file ('w' will overwrite existing files, 'r+' will open existing file to write)
    netcdf_output = nc.Dataset(fullfile, 'w', format='NETCDF4')
    # Global attributes
    netcdf_output.description = 'Results from glacier evolution model'
    netcdf_output.history = 'Created ' + str(strftime("%Y-%m-%d %H:%M:%S"))
    netcdf_output.source = 'Python Glacier Evolution Model'
    # Dimensions
    glacier = netcdf_output.createDimension('glacier', None)
    binelev = netcdf_output.createDimension('binelev', main_glac_hyps.shape[1])
    if input.timestep == 'monthly':
        time = netcdf_output.createDimension('time', dates_table.shape[0] - input.spinupyears * 12)
    year = netcdf_output.createDimension('year', annual_columns.shape[0] - input.spinupyears)
    year_plus1 = netcdf_output.createDimension('year_plus1', annual_columns.shape[0] - input.spinupyears + 1)
    glacierparameter = netcdf_output.createDimension('glacierparameter', 10)
    #glaciertable = netcdf_output.createDimension('glaciertable', main_glac_hyps.shape[0])
    # Variables associated with dimensions 
    glaciers = netcdf_output.createVariable('glacier', np.int32, ('glacier',))
    glaciers.long_name = "glacier number associated with model run"
    glaciers.standard_name = "GlacNo"
    glaciers.comment = ("The glacier number is defined for each model run. The user should look at the main_glac_rgi"
                           + " table to determine the RGIID or other information regarding this particular glacier.")
    binelevs = netcdf_output.createVariable('binelev', np.int32, ('binelev',))
    binelevs.standard_name = "center bin_elevation"
    binelevs.units = "m a.s.l."
    binelevs[:] = main_glac_hyps.columns.values
    binelevs.comment = ("binelev are the bin elevations that were used for the model run.")
    times = netcdf_output.createVariable('time', np.float64, ('time',))
    times.standard_name = "date"
    times.units = "days since 1900-01-01 00:00:00"
    times.calendar = "gregorian"
    if input.timestep == 'monthly':
        times[:] = (nc.date2num(dates_table.loc[input.spinupyears*12:dates_table.shape[0]+1,'date'].astype(datetime), 
                                units = times.units, calendar = times.calendar))
    years = netcdf_output.createVariable('year', np.int32, ('year',))
    years.standard_name = "year"
    if input.option_wateryear == 1:
        years.units = 'water year'
    elif input.option_wateryear == 0:
        years.units = 'calendar year'
    years[:] = annual_columns[input.spinupyears:annual_columns.shape[0]]
    # years_plus1 adds an additional year such that the change in glacier dimensions (area, etc.) is recorded
    years_plus1 = netcdf_output.createVariable('year_plus1', np.int32, ('year_plus1',))
    years_plus1.standard_name = "year with additional year to record glacier dimension changes"
    if input.option_wateryear == 1:
        years_plus1.units = 'water year'
    elif input.option_wateryear == 0:
        years_plus1.units = 'calendar year'
    years_plus1 = np.concatenate((annual_columns[input.spinupyears:annual_columns.shape[0]], 
                                  np.array([annual_columns[annual_columns.shape[0]-1]+1])))
    glacierparameters = netcdf_output.createVariable('glacierparameters',str,('glacierparameter',))
    glacierparameters.standard_name = "important parameters associated with the glacier for the model run"
    glacierparameters[:] = np.array(['RGIID','lat','lon','lr_gcm','lr_glac','prec_factor','prec_grad','DDF_ice',
                                     'DDF_snow','T_snow'])
    glacierparameter = netcdf_output.createVariable('glacierparameter',str,('glacier','glacierparameter',))
    # Variables associated with the output
    if input.output_package == 1:
        # Package 1 "Raw Package" output [units: m w.e. unless otherwise specified]:
        #  monthly variables for each bin (temp, prec, acc, refreeze, snowpack, melt, frontalablation, massbal_clim)
        #  annual variables for each bin (area, icethickness, surfacetype)
        temp_bin_monthly = netcdf_output.createVariable('temp_bin_monthly', np.float64, ('glacier', 'binelev', 'time'))
        temp_bin_monthly.standard_name = "air temperature"
        temp_bin_monthly.units = "degC"
        prec_bin_monthly = netcdf_output.createVariable('prec_bin_monthly', np.float64, ('glacier', 'binelev', 'time'))
        prec_bin_monthly.standard_name = "liquid precipitation"
        prec_bin_monthly.units = "m"
        acc_bin_monthly = netcdf_output.createVariable('acc_bin_monthly', np.float64, ('glacier', 'binelev', 'time'))
        acc_bin_monthly.standard_name = "accumulation"
        acc_bin_monthly.units = "m w.e."
        refreeze_bin_monthly = netcdf_output.createVariable('refreeze_bin_monthly', np.float64, ('glacier', 'binelev', 
                                                                                                 'time'))
        refreeze_bin_monthly.standard_name = "refreezing"
        refreeze_bin_monthly.units = "m w.e."
        snowpack_bin_monthly = netcdf_output.createVariable('snowpack_bin_monthly', np.float64, ('glacier', 'binelev', 
                                                                                                  'time'))
        snowpack_bin_monthly.standard_name = "snowpack on the glacier surface"
        snowpack_bin_monthly.units = "m w.e."
        snowpack_bin_monthly.comment = ("snowpack represents the snow depth when units are m w.e.")
        melt_bin_monthly = netcdf_output.createVariable('melt_bin_monthly', np.float64, ('glacier', 'binelev', 'time'))
        melt_bin_monthly.standard_name = 'surface melt'
        melt_bin_monthly.units = "m w.e."
        melt_bin_monthly.comment = ("surface melt is the sum of melt from snow, refreeze, and the underlying glacier")
        frontalablation_bin_monthly = netcdf_output.createVariable('frontalablation_bin_monthly', np.float64, 
                                                                   ('glacier', 'binelev', 'time'))
        frontalablation_bin_monthly.standard_name = "frontal ablation"
        frontalablation_bin_monthly.units = "m w.e."
        frontalablation_bin_monthly.comment = ("mass losses from calving, subaerial frontal melting, sublimation above "
                                                + "the waterline and subaqueous frontal melting below the waterline")
        massbalclim_bin_monthly = netcdf_output.createVariable('massbalclim_bin_monthly', np.float64, 
                                                               ('glacier', 'binelev', 'time'))
        massbalclim_bin_monthly.standard_name = "climatic mass balance"
        massbalclim_bin_monthly.units = "m w.e."
        massbalclim_bin_monthly.comment = ("climatic mass balance is the sum of the surface mass balance and the "
                                           + "internal mass balance and accounts for the climatic mass loss over the "
                                           + "area of the entire bin")
        area_bin_annual = netcdf_output.createVariable('area_bin_annual', np.float64, ('glacier', 'binelev', 
                                                                                       'year_plus1'))
        area_bin_annual.standard_name = "glacier area"
        area_bin_annual.unit = "km**2"
        area_bin_annual.comment = ("the area that was used for the duration of the year")
        icethickness_bin_annual = netcdf_output.createVariable('icethickness_bin_annual', np.float64, 
                                                               ('glacier', 'binelev', 'year_plus1'))
        icethickness_bin_annual.standard_name = "ice thickness"
        icethickness_bin_annual.unit = "m ice"
        icethickness_bin_annual.comment = ("the ice thickness that was used for the duration of the year")
        surfacetype_bin_annual = netcdf_output.createVariable('surfacetype_bin_annual', np.float64, ('glacier', 'binelev', 
                                                                                               'year'))
        surfacetype_bin_annual.standard_name = "surface type"
        surfacetype_bin_annual.comment = "surface types: 0 = off-glacier, 1 = ice, 2 = snow, 3 = firn, 4 = debris"
    elif input.output_package == 2:
        # Package 2 "Glaciologist Package" output [units: m w.e. unless otherwise specified]:
        # Monthly glacier-wide variables (acc, refreeze, snowpack, melt, frontalablation, massbal_clim, massbal_total, area
        #  area, ice thickness, volume, runoff, ELA, AAR, snowline)
        print('create package')
    netcdf_output.close()


def netcdfwrite(regionO1_number, glac, main_glac_rgi, glac_bin_temp, glac_bin_prec, glac_bin_acc, glac_bin_refreeze, 
                glac_bin_snowpack, glac_bin_melt, glac_bin_frontalablation, glac_bin_massbalclim, glac_bin_area_annual, 
                glac_bin_icethickness_annual, glac_bin_surfacetype_annual):
    """Write to the netcdf file that has already been generated to store the desired output
    Output: netcdf with desired variables filled in
    """
    # Netcdf file path and name
    filename = input.netcdf_filenameprefix + str(regionO1_number) + '_' + str(strftime("%Y%m%d")) + '.nc'
    fullfile = input.netcdf_filepath + filename
    # Open netcdf file to write to existing file ('r+')
    netcdf_output = nc.Dataset(fullfile, 'r+')
    # Record the variables for each glacier (remove data associated with spinup years)
    # glaciers parameters
    netcdf_output.variables['glacierparameter'][glac,:] = np.array([main_glac_rgi.loc[glac,'RGIId'],
        main_glac_rgi.loc[glac,input.lat_colname],main_glac_rgi.loc[glac,input.lon_colname], input.lr_gcm, 
        input.lr_glac, input.prec_factor, input.prec_grad, input.DDF_ice, input.DDF_snow, input.T_snow])
    if input.output_package == 1:
        netcdf_output.variables['temp_bin_monthly'][glac,:,:] = (
                glac_bin_temp[:,input.spinupyears*12:glac_bin_temp.shape[1]+1])
        netcdf_output.variables['prec_bin_monthly'][glac,:,:] = (
                glac_bin_prec[:,input.spinupyears*12:glac_bin_temp.shape[1]+1])
        netcdf_output.variables['acc_bin_monthly'][glac,:,:] = (
                glac_bin_acc[:,input.spinupyears*12:glac_bin_temp.shape[1]+1])
        netcdf_output.variables['refreeze_bin_monthly'][glac,:,:] = (
                glac_bin_refreeze[:,input.spinupyears*12:glac_bin_temp.shape[1]+1])
        netcdf_output.variables['snowpack_bin_monthly'][glac,:,:] = (
                glac_bin_snowpack[:,input.spinupyears*12:glac_bin_temp.shape[1]+1])
        netcdf_output.variables['melt_bin_monthly'][glac,:,:] = (
                glac_bin_melt[:,input.spinupyears*12:glac_bin_temp.shape[1]+1])
        netcdf_output.variables['frontalablation_bin_monthly'][glac,:,:] = (
                glac_bin_frontalablation[:,input.spinupyears*12:glac_bin_temp.shape[1]+1])
        netcdf_output.variables['massbalclim_bin_monthly'][glac,:,:] = (
                glac_bin_massbalclim[:,input.spinupyears*12:glac_bin_temp.shape[1]+1])
        netcdf_output.variables['area_bin_annual'][glac,:,:] = (
                glac_bin_area_annual[:,input.spinupyears:glac_bin_area_annual.shape[1]+1])
        netcdf_output.variables['icethickness_bin_annual'][glac,:,:] = (
                glac_bin_icethickness_annual[:,input.spinupyears:glac_bin_area_annual.shape[1]+1])
        netcdf_output.variables['surfacetype_bin_annual'][glac,:,:] = (
                glac_bin_surfacetype_annual[:,input.spinupyears:glac_bin_area_annual.shape[1]+1])
    # Close the netcdf file
    netcdf_output.close()


#========= OLD FUNCTIONS (alphabetical order) =========================================================================
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
                .iloc[int(ELA_output.loc[GlacNo, AAR_output.columns[step]] / input.binsize) - 1]) * 100
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
            if input.timestep == 'monthly':
                massbal_spec_annual = massbal_spec_monthly.groupby(np.arange(massbal_spec_monthly.shape[1]) // 12, 
                                                                   axis=1).sum()
                massbal_spec_annual.columns = ds['year'].values
            elif input.timestep == 'daily':
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
                    (series_massbal_spec.index.values[series_ELA_signchange[0]][0] - input.binsize/2).astype(int))
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
                                series_ELA_sign.index.values[np.where(series_ELA_sign != 0)[0][0]] - input.binsize/2)
                    # if entire glacier is negative, then set to the glacier's maximum
                    elif series_ELA_sign.iloc[np.where((series_ELA_sign != 0))[0][0]] == -1:
                        ELA_output.loc[GlacNo, ELA_output.columns.values[step]] = (
                            series_ELA_sign.index.values[np.where(series_ELA_sign != 0)[0]
                                                         [np.where(series_ELA_sign != 0)[0].shape[0]-1]] + input.binsize
                                                          /2)
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

# Extras for writing netcdf:
#    # Annual variables being recorded for each bin of each glacier
#    acc_bin_annual = netcdf_output.createVariable('acc_bin_annual', np.float64, ('glacier', 'binelev', 'year'))
#    acc_bin_annual.long_name = "solid precipitation"
#    acc_bin_annual.standard_name = "accumulation"
#    acc_bin_annual.units = "m w.e."
#    acc_bin_annual.comment = "annual accumulation is the total snowfall for a given year"
#    refreeze_bin_annual = netcdf_output.createVariable('refreeze_bin_annual', np.float64, ('glacier', 'binelev', 
#                                                                                           'year'))
#    refreeze_bin_annual.long_name = "refreezing of melted snow"
#    refreeze_bin_annual.standard_name = "refreezing"
#    refreeze_bin_annual.units = "m w.e."
#    refreeze_bin_annual.comment = "refreezing is the amount of melted snow that refreezes within the snow"
#    melt_bin_annual = netcdf_output.createVariable('melt_bin_annual', np.float64, ('glacier', 'binelev', 'year'))
#    melt_bin_annual.long_name = 'specific surface melt'
#    melt_bin_annual.standard_name = 'surface melt'
#    melt_bin_annual.units = "m w.e."
#    melt_bin_annual.comment = ("surface melt is the sum of both the melting of the snow on the surface and the "
#                               + "underlying ice/firn/snow")
#    frontal_ablation_bin_annual = netcdf_output.createVariable('frontal_ablation_bin_annual', np.float64, 
#                                                               ('glacier', 'binelev', 'year'))
#    frontal_ablation_bin_annual.standard_name = "annual specific frontal ablation"
#    frontal_ablation_bin_annual.units = "m w.e."
#    frontal_ablation_bin_annual.comment = ("mass losses due to calving, subaerial frontal melting, sublimation above "
#                                           + "the waterline and subaqueous frontal melting below the waterline")
#    massbal_clim_bin_annual = netcdf_output.createVariable('massbal_clim_bin_annual', np.float64, 
#                                                               ('glacier', 'binelev', 'year'))
#    massbal_clim_bin_annual.standard_name = "annual specific climatic mass balance"
#    massbal_clim_bin_annual.unit = "m w.e."
#    massbal_clim_bin_annual.comment = ("climatic mass balance is the sum of the surface mass balance and the "
#                                           + "internal mass balance and accounts for the climatic mass loss over the "
#                                           + "area of the entire bin") 
#    massbal_total_bin_annual = netcdf_output.createVariable('massbal_total_mwe_bin_annual', np.float64, 
#                                                              ('glacier', 'binelev', 'year'))
#    massbal_total_bin_annual.standard_name = "annual specific total mass balance"
#    massbal_total_bin_annual.unit = "m w.e."
#    massbal_total_bin_annual.comment = ("specific total mass balance is the total conservation of mass within an "
#                                            + "elevation bin, i.e., it includes the climatic mass balance, flow of ice "
#                                            + "into or out of the bin, calving, bed losses/gains, and any additional "
#                                            + "sources of mass loss or gain.")
#    volume_bin_annual = netcdf_output.createVariable('volume_bin_annual', np.float64, ('glacier', 'binelev', 'year'))
#    volume_bin_annual.long_name = "volume of ice in each bin updated annually"
#    volume_bin_annual.standard_name = "glacier volume"
#    volume_bin_annual.unit = "km**3"
#    volume_bin_annual.comment = ("volume is the total volume of ice in the elevation bin that was used for the duration"
#                                 + " of the timestep, i.e., it is the volume at the start of the time step")
#    width_bin_annual = netcdf_output.createVariable('width_bin_annual', np.float64, ('glacier', 'binelev', 'year'))
#    width_bin_annual.long_name = "width of glacier in each bin updated annually"
#    width_bin_annual.standard_name = "width"
#    width_bin_annual.unit = "m"
#    # Monthly glacier-wide variables being recorded for each glacier
#    acc_glac_monthly = netcdf_output.createVariable('acc_glac_monthly', np.float64, ('glaciertable', 'time'))
#    acc_glac_monthly.standard_name = "monthly glacier-wide specific accumulation"
#    acc_glac_monthly.units = "m w.e."
#    acc_glac_monthly.comment = "total monthly accumulation over the entire glacier divided by the glacier area"
#    refreeze_glac_monthly = netcdf_output.createVariable('refreeze_glac_monthly', np.float64, ('glaciertable', 'time'))
#    refreeze_glac_monthly.standard_name = "monthly glacier-wide specific refreeze"
#    refreeze_glac_monthly.units = "m w.e."
#    refreeze_glac_monthly.comment = "total monthly refreeze over the entire glacier divided by the glacier area"
#    melt_glac_monthly = netcdf_output.createVariable('melt_glac_monthly', np.float64, ('glaciertable', 'time'))
#    melt_glac_monthly.standard_name = "monthly glacier-wide specific melt"
#    melt_glac_monthly.units = "m w.e."
#    melt_glac_monthly.comment = "total monthly melt over the entire glacier divided by the glacier area"
#    frontal_ablation_glac_monthly = netcdf_output.createVariable('frontal_ablation_glac_monthly', np.float64, 
#                                                                 ('glaciertable', 'time'))
#    frontal_ablation_glac_monthly.standard_name = "monthly glacier-wide specific frontal ablation"
#    frontal_ablation_glac_monthly.units = "m w.e."
#    frontal_ablation_glac_monthly.comment = ("mass losses due to calving, subaerial frontal melting, sublimation above "
#                                             + "the waterline and subaqueous frontal melting below the waterline")
#    massbal_clim_glac_monthly = netcdf_output.createVariable('massbal_clim_glac_monthly', np.float64, 
#                                                                 ('glaciertable', 'time'))
#    massbal_clim_glac_monthly.standard_name = "monthly glacier-wide specific climatic mass balance"
#    massbal_clim_glac_monthly.units = "m w.e."
#    massbal_clim_glac_monthly.comment = ("climatic mass balance is the sum of the surface mass balance and the "
#                                             + "internal mass balance and accounts for the climatic mass loss over the "
#                                             + "area of the entire bin") 
#    area_glac_monthly = netcdf_output.createVariable('area_glac_monthly', np.float64, ('glaciertable', 'time'))
#    area_glac_monthly.standard_name = "glacier area"
#    area_glac_monthly.units = "km**2"
#    area_glac_monthly.comment = ("area for a given timestep is the area that was used for the duration of the timestep,"
#                                 + " i.e., it is the area at the start of the time step")
#    ELA_glac_monthly = netcdf_output.createVariable('ELA_glac_monthly', np.float64, ('glaciertable', 'time'))
#    ELA_glac_monthly.standard_name = "transient equilibrium line altitude"
#    ELA_glac_monthly.units = "m a.s.l."
#    ELA_glac_monthly.comment = ("equilibrium line altitude is the elevation that separates a positive and negative mass"
#                                + "balance on the glacier")
#    AAR_glac_monthly = netcdf_output.createVariable('AAR_glac_monthly', np.float64, ('glaciertable', 'time'))
#    AAR_glac_monthly.standard_name = "transient accumulation-area ratio"
#    AAR_glac_monthly.units = "%"
#    AAR_glac_monthly.comment = ("accumulation-area ratio is the percentage of the area of the accumulation zone to the "
#                                + "area of the glacier")
#    snowline_glac_monthly = netcdf_output.createVariable('snowline_glac_monthly', np.float64, ('glaciertable', 'time'))
#    snowline_glac_monthly.standard_name = "transient snowline"
#    snowline_glac_monthly.units = "m a.s.l."
#    snowline_glac_monthly.comment = ("transient snowline is the line separating the snow from ice/firn at the end of "
#                                     + "each month")
#    runoff_glac_monthly = netcdf_output.createVariable('runoff_glac_monthly', np.float64, ('glaciertable', 'time'))
#    runoff_glac_monthly.long_name = "monthly glacier runoff"
#    runoff_glac_monthly.standard_name = "glacier runoff"
#    runoff_glac_monthly.units = "m**3"
#    runoff_glac_monthly.comment = "runoff is the total volume of water that is leaving the glacier each month"
#    # Annual glacier-wide variables being recorded for each glacier
#    acc_glac_annual = netcdf_output.createVariable('acc_glac_annual', np.float64, ('glaciertable', 'year'))
#    acc_glac_annual.standard_name = "annual glacier-wide specific accumulation"
#    acc_glac_annual.units = "m w.e."
#    acc_glac_annual.comment = "total annual accumulation over the entire glacier divided by the glacier area"
#    refreeze_glac_annual = netcdf_output.createVariable('refreeze_glac_annual', np.float64, ('glaciertable', 'year'))
#    refreeze_glac_annual.standard_name = "annual glacier-wide specific refreeze"
#    refreeze_glac_annual.units = "m w.e."
#    refreeze_glac_annual.comment = "total annual refreeze over the entire glacier divided by the glacier area"
#    melt_glac_annual = netcdf_output.createVariable('melt_glac_annual', np.float64, ('glaciertable', 'year'))
#    melt_glac_annual.standard_name = "annual glacier-wide specific melt"
#    melt_glac_annual.units = "m w.e."
#    melt_glac_annual.comment = "total annual melt over the entire glacier divided by the glacier area"
#    frontal_ablation_glac_annual = netcdf_output.createVariable('frontal_ablation_glac_annual', np.float64, 
#                                                                ('glaciertable', 'year'))
#    frontal_ablation_glac_annual.standard_name = "annual glacier-wide specific frontal ablation"
#    frontal_ablation_glac_annual.units = "m w.e."
#    frontal_ablation_glac_annual.comment = ("mass losses due to calving, subaerial frontal melting, sublimation above "
#                                            + "the waterline and subaqueous frontal melting below the waterline")
#    massbal_clim_glac_annual = netcdf_output.createVariable('massbal_clim_glac_annual', np.float64, 
#                                                                ('glaciertable', 'year'))
#    massbal_clim_glac_annual.standard_name = "annual glacier-wide specific climatic mass balance"
#    massbal_clim_glac_annual.units = "m w.e."
#    massbal_clim_glac_annual.comment = ("climatic mass balance is the sum of the surface mass balance and the "
#                                            + "internal mass balance and accounts for the climatic mass loss over the "
#                                            + "area of the entire bin") 
#    massbal_total_glac_annual = netcdf_output.createVariable('massbal_total_glac_annual', np.float64, 
#                                                                 ('glaciertable', 'year'))
#    massbal_total_glac_annual.standard_name = "annual glacier-wide specific climatic mass balance"
#    massbal_total_glac_annual.units = "m w.e."
#    massbal_total_glac_annual.comment = ("specific total mass balance is the total conservation of mass within an "
#                                             + "elevation bin, i.e., it includes the climatic mass balance, flow of ice"
#                                             + " into or out of the bin, calving, bed losses/gains, and any additional "
#                                             + "sources of mass loss or gain.")
#    area_glac_annual = netcdf_output.createVariable('area_glac_annual', np.float64, ('glaciertable', 'year'))
#    area_glac_annual.standard_name = "glacier area"
#    area_glac_annual.units = "km**2"
#    area_glac_annual.comment = ("area for a given timestep is the area that was used for the duration of the timestep, "
#                                + "i.e., it is the area at the start of the time step")
#    volume_glac_annual = netcdf_output.createVariable('volume_glac_annual', np.float64, ('glaciertable', 'year'))
#    volume_glac_annual.standard_name = "glacier volume"
#    volume_glac_annual.units = "km**3"
#    volume_glac_annual.comment = ("volume is the total volume of ice in the elevation bin that was used for the "
#                                  + "duration of the timestep, i.e., it is the volume at the start of the time step")
#    ELA_glac_annual = netcdf_output.createVariable('ELA_glac_annual', np.float64, ('glaciertable', 'year'))
#    ELA_glac_annual.standard_name = "annual equilibrium line altitude"
#    ELA_glac_annual.units = "m a.s.l."
#    ELA_glac_annual.comment = ("annual equilibrium line altitude is the elevation that separates a positive and "
#                               + "negative annual mass balance on the glacier")
#    AAR_glac_annual = netcdf_output.createVariable('AAR_glac_annual', np.float64, ('glaciertable', 'year'))
#    AAR_glac_annual.standard_name = "annual accumulation-area ratio"
#    AAR_glac_annual.units = "m a.s.l."
#    AAR_glac_annual.comment = ("annual accumulation-area ratio is the percentage of the area of the accumulation zone "
#                               + "to the area of the glacier based on the annual mass balance")
#    snowline_glac_annual = netcdf_output.createVariable('snowline_glac_annual', np.float64, ('glaciertable', 'year'))
#    snowline_glac_annual.standard_name = "annual snowline"
#    snowline_glac_annual.units = "m a.s.l."
#    snowline_glac_annual.comment = ("annual snowline is the snowline during the month of minimum snow coverm, which "
#                                    + "usually occurs at the end of the ablation season.")
#    runoff_glac_annual = netcdf_output.createVariable('runoff_glac_annual', np.float64, ('glaciertable', 'year'))
#    runoff_glac_annual.standard_name = "annual glacier runoff"
#    runoff_glac_annual.units = "m a.s.l."
#    runoff_glac_annual.comment = "annual runoff is jthe total volume of water that is leaving the glacier each year"