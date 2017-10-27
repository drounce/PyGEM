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
def netcdf_output_create(regionO1_number, main_glac_hyps, dates_table, annual_columns):
    """Create a netcdf file to store the desired output
    
    Output: empty netcdf file with the proper setup to be filled in by the model
    """
    # netcdf file path, name, and format
    filename = input.netcdf_filenameprefix + str(regionO1_number) + '_' + str(strftime("%Y%m%d")) + '.nc'
    fullfile = input.netcdf_filepath + filename
    fileformat = 'NETCDF4_CLASSIC'
    # Create the netcdf file open to write ('w') with the netCDF4 classic file format
    netcdf_output = nc.Dataset(fullfile, 'w', format=fileformat)
    # Create global attributes
    netcdf_output.description = 'Monthly specific mass balance for each glacier in the model run'
    netcdf_output.history = 'Created ' + str(strftime("%Y-%m-%d %H:%M:%S"))
    netcdf_output.source = 'Python Global Glacier Evolution Model'
    # Create dimensions
    glacier = netcdf_output.createDimension('glacier', None)
    binelev = netcdf_output.createDimension('binelev', main_glac_hyps.shape[1])
    time = netcdf_output.createDimension('time', dates_table.shape[0])
    year = netcdf_output.createDimension('year', annual_columns.shape[0])
    glaciertable = netcdf_output.createDimension('glaciertable', main_glac_hyps.shape[0])
    # Create the variables associated with the dimensions
    glacier_out = netcdf_output.createVariable('glacier', np.int32, ('glacier',))
    glacier_out.long_name = "glacier number associated with model run"
    glacier_out.standard_name = "GlacNo"
    glacier_out.comment = ("The glacier number is defined for each model run. The user should look at the main_glac_rgi"
                           + " table to determine the RGIID or other information regarding this particular glacier.")
    binelev_out = netcdf_output.createVariable('binelev', np.int32, ('binelev',))
    binelev_out.long_name = "center bin elevation"
    binelev_out.standard_name = "bin_elevation"
    binelev_out.units = "m a.s.l."
    binelev_out[:] = main_glac_hyps.columns.values
    binelev_out.comment = ("binelev are the bin elevations that were used for the model run. By default, they span from"
                           + " 0 - 9000 and their spacing is based on the input binsize.")
    time_out = netcdf_output.createVariable('time', np.float64, ('time',))
    time_out.long_name = "date of model run"
    time_out.standard_name = "date"
    time_out.units = "days since 1900-01-01 00:00:00"
    time_out.calendar = "gregorian"
    time_out[:] = nc.date2num(dates_table['date'].astype(datetime), units = time_out.units, 
                              calendar = time_out.calendar)
    year_out = netcdf_output.createVariable('year', np.int32, ('year',))
    year_out.long_name = "year of model run"
    year_out.standard_name = "year"
    if input.option_wateryear == 1:
        year_out.units = 'water year'
    elif input.option_wateryear == 0:
        year_out.units = 'calendar year'
    year_out[:] = annual_columns
    # Monthly variables being recorded for each bin of each glacier
    acc_bin_monthly = netcdf_output.createVariable('acc_bin_monthly', np.float64, ('glacier', 'binelev', 'time'))
    acc_bin_monthly.long_name = "solid precipitation"
    acc_bin_monthly.standard_name = "specific accumulation"
    acc_bin_monthly.units = "m w.e."
    acc_bin_monthly.comment = "accumulation is the total snowfall for a given timestep"
    refreeze_bin_monthly = netcdf_output.createVariable('refreeze_bin_monthly', np.float64, ('glacier', 'binelev', 
                                                                                             'time'))
    refreeze_bin_monthly.long_name = "refreezing of melted snow"
    refreeze_bin_monthly.standard_name = "refreezing"
    refreeze_bin_monthly.units = "m w.e."
    refreeze_bin_monthly.comment = "refreezing is the amount of melted snow that refreezes within the snow"
    melt_bin_monthly = netcdf_output.createVariable('melt_bin_monthly', np.float64, ('glacier', 'binelev', 'time'))
    melt_bin_monthly.long_name = 'specific surface melt'
    melt_bin_monthly.standard_name = 'surface melt'
    melt_bin_monthly.units = "m w.e."
    melt_bin_monthly.comment = ("surface melt is the sum of both the melting of the snow on the surface and the "
                                + "underlying ice/firn/snow")
    frontal_ablation_bin_monthly = netcdf_output.createVariable('frontal_ablation_bin_monthly', np.float64, 
                                                                ('glacier', 'binelev', 'time'))
    frontal_ablation_bin_monthly.standard_name = "specific frontal ablation"
    frontal_ablation_bin_monthly.units = "m w.e."
    frontal_ablation_bin_monthly.comment = ("mass losses due to calving, subaerial frontal melting, sublimation above "
                                            + "the waterline and subaqueous frontal melting below the waterline")
    massbal_clim_mwe_bin_monthly = netcdf_output.createVariable('massbal_clim_mwe_bin_monthly', np.float64, 
                                                                ('glacier', 'binelev', 'time'))
    massbal_clim_mwe_bin_monthly.standard_name = "monthly specific climatic mass balance"
    massbal_clim_mwe_bin_monthly.units = "m w.e."
    massbal_clim_mwe_bin_monthly.comment = ("climatic mass balance is the sum of the surface mass balance and the "
                                            + "internal mass balance and accounts for the climatic mass loss over the "
                                            + "area of the entire bin") 
    area_bin_monthly = netcdf_output.createVariable('area_bin_monthly', np.float64, ('glacier', 'binelev', 'time'))
    area_bin_monthly.long_name = "monthly glacier area of each elevation bin updated annually"
    area_bin_monthly.standard_name = "area"
    area_bin_monthly.unit = "km**2"
    area_bin_monthly.comment = ("area for a given year is the area that was used for the duration of the timestep, "
                                + "i.e., it is the area at the start of the time step")
    temp_bin_monthly = netcdf_output.createVariable('temp_bin_monthly', np.float64, ('glacier', 'binelev', 'time'))
    temp_bin_monthly.long_name = "air temperature "
    temp_bin_monthly.standard_name = "temperature"
    temp_bin_monthly.units = "degC"
    prec_bin_monthly = netcdf_output.createVariable('prec_bin_monthly', np.float64, ('glacier', 'binelev', 'time'))
    prec_bin_monthly.long_name = "liquid precipitation"
    prec_bin_monthly.standard_name = "precipitation"
    prec_bin_monthly.units = "m"
    prec_bin_monthly.comment = "precipitation is the total liquid precipitation that fell during that month"
    snowdepth_bin_monthly = netcdf_output.createVariable('snowdepth_bin_monthly', np.float64, 
                                                             ('glacier', 'binelev', 'time'))
    snowdepth_bin_monthly.standard_name = "snow depth on surface"
    snowdepth_bin_monthly.units = "m w.e."
    snowdepth_bin_monthly.comment = ("snow depth is the depth of snow at the end of the timestep, since the snow on the"
                                     + " surface depends on the snowfall, melt, etc. during that timestep")
    
    melt_snowcomponent_bin_monthly = netcdf_output.createVariable('melt_snow_bin_monthly', np.float64, 
                                                                  ('glacier', 'binelev', 'time'))
    melt_snowcomponent_bin_monthly.long_name = ("specific snow melt above the surface regardless of underlying surface "
                                                + "type")
    melt_snowcomponent_bin_monthly.standard_name = "snow melt"
    melt_snowcomponent_bin_monthly.units = "m w.e."
    melt_snowcomponent_bin_monthly.comment = ("only the melt associated with the snow on the surface and refreezing in "
                                              + "the snow regardless of whether the underlying surface type is snow or "
                                              + "not")
    melt_glaccomponent_bin_monthly = netcdf_output.createVariable('melt_glac_bin_monthly', np.float64, 
                                                                  ('glacier', 'binelev', 'time'))
    melt_glaccomponent_bin_monthly.long_name = "specific glacier melt"
    melt_glaccomponent_bin_monthly.standard_name = "glacier melt"
    melt_glaccomponent_bin_monthly.units = "m w.e."
    melt_glaccomponent_bin_monthly.comment = ("melt of the glacier (firn/ice) after all the snow on the surface has "
                                              + "melted")
    # Annual variables being recorded for each bin of each glacier
    acc_bin_annual = netcdf_output.createVariable('acc_bin_annual', np.float64, ('glacier', 'binelev', 'year'))
    acc_bin_annual.long_name = "solid precipitation"
    acc_bin_annual.standard_name = "accumulation"
    acc_bin_annual.units = "m w.e."
    acc_bin_annual.comment = "annual accumulation is the total snowfall for a given year"
    refreeze_bin_annual = netcdf_output.createVariable('refreeze_bin_annual', np.float64, ('glacier', 'binelev', 
                                                                                           'year'))
    refreeze_bin_annual.long_name = "refreezing of melted snow"
    refreeze_bin_annual.standard_name = "refreezing"
    refreeze_bin_annual.units = "m w.e."
    refreeze_bin_annual.comment = "refreezing is the amount of melted snow that refreezes within the snow"
    melt_bin_annual = netcdf_output.createVariable('melt_bin_annual', np.float64, ('glacier', 'binelev', 'year'))
    melt_bin_annual.long_name = 'specific surface melt'
    melt_bin_annual.standard_name = 'surface melt'
    melt_bin_annual.units = "m w.e."
    melt_bin_annual.comment = ("surface melt is the sum of both the melting of the snow on the surface and the "
                               + "underlying ice/firn/snow")
    frontal_ablation_bin_annual = netcdf_output.createVariable('frontal_ablation_bin_annual', np.float64, 
                                                               ('glacier', 'binelev', 'year'))
    frontal_ablation_bin_annual.standard_name = "annual specific frontal ablation"
    frontal_ablation_bin_annual.units = "m w.e."
    frontal_ablation_bin_annual.comment = ("mass losses due to calving, subaerial frontal melting, sublimation above "
                                           + "the waterline and subaqueous frontal melting below the waterline")
    massbal_clim_mwe_bin_annual = netcdf_output.createVariable('massbal_clim_mwe_bin_annual', np.float64, 
                                                               ('glacier', 'binelev', 'year'))
    massbal_clim_mwe_bin_annual.standard_name = "annual specific climatic mass balance"
    massbal_clim_mwe_bin_annual.unit = "m w.e."
    massbal_clim_mwe_bin_annual.comment = ("climatic mass balance is the sum of the surface mass balance and the "
                                           + "internal mass balance and accounts for the climatic mass loss over the "
                                           + "area of the entire bin") 
    massbal_total_mwe_bin_annual = netcdf_output.createVariable('massbal_total_mwe_bin_annual', np.float64, 
                                                              ('glacier', 'binelev', 'year'))
    massbal_total_mwe_bin_annual.standard_name = "annual specific total mass balance"
    massbal_total_mwe_bin_annual.unit = "m w.e."
    massbal_total_mwe_bin_annual.comment = ("specific total mass balance is the total conservation of mass within an "
                                            + "elevation bin, i.e., it includes the climatic mass balance, flow of ice "
                                            + "into or out of the bin, calving, bed losses/gains, and any additional "
                                            + "sources of mass loss or gain.")
    area_bin_annual = netcdf_output.createVariable('area_bin_annual', np.float64, ('glacier', 'binelev', 'year'))
    area_bin_annual.long_name = "glacier area of each elevation bin updated annually"
    area_bin_annual.standard_name = "area"
    area_bin_annual.unit = "km**2"
    area_bin_annual.comment = ("area for a given year is the area that was used for the duration of the timestep, i.e.,"
                               + " it is the area at the start of the time step")
    icethickness_bin_annual = netcdf_output.createVariable('icethickness_bin_annual', np.float64, ('glacier', 'binelev',
                                                                                                   'year'))
    icethickness_bin_annual.long_name = "ice thickness of each bin updated annually"
    icethickness_bin_annual.standard_name = "ice thickness"
    icethickness_bin_annual.unit = "m"
    icethickness_bin_annual.comment = ("ice thickness is the ice thickness that was used for the duration of the "
                                       + "timestep, i.e., it is the ice thickness at the start of the time step")
    volume_bin_annual = netcdf_output.createVariable('volume_bin_annual', np.float64, ('glacier', 'binelev', 'year'))
    volume_bin_annual.long_name = "volume of ice in each bin updated annually"
    volume_bin_annual.standard_name = "glacier volume"
    volume_bin_annual.unit = "km**3"
    volume_bin_annual.comment = ("volume is the total volume of ice in the elevation bin that was used for the duration"
                                 + " of the timestep, i.e., it is the volume at the start of the time step")
    surftype_bin_annual = netcdf_output.createVariable('surftype_bin_annual', np.float64, ('glacier', 'binelev', 
                                                                                           'year'))
    surftype_bin_annual.standard_name = "surface type"
    surftype_bin_annual.comment = "surface type defined as 1 = ice, 2 = snow, 3 = firn, 4 = debris, 0 = off glacier"
    width_bin_annual = netcdf_output.createVariable('width_bin_annual', np.float64, ('glacier', 'binelev', 'year'))
    width_bin_annual.long_name = "width of glacier in each bin updated annually"
    width_bin_annual.standard_name = "width"
    width_bin_annual.unit = "m"
    # Monthly glacier-wide variables being recorded for each glacier
    acc_glac_monthly = netcdf_output.createVariable('acc_glac_monthly', np.float64, ('glaciertable', 'time'))
    acc_glac_monthly.standard_name = "monthly glacier-wide specific accumulation"
    acc_glac_monthly.units = "m w.e."
    acc_glac_monthly.comment = "total monthly accumulation over the entire glacier divided by the glacier area"
    refreeze_glac_monthly = netcdf_output.createVariable('refreeze_glac_monthly', np.float64, ('glaciertable', 'time'))
    refreeze_glac_monthly.standard_name = "monthly glacier-wide specific refreeze"
    refreeze_glac_monthly.units = "m w.e."
    refreeze_glac_monthly.comment = "total monthly refreeze over the entire glacier divided by the glacier area"
    melt_glac_monthly = netcdf_output.createVariable('melt_glac_monthly', np.float64, ('glaciertable', 'time'))
    melt_glac_monthly.standard_name = "monthly glacier-wide specific melt"
    melt_glac_monthly.units = "m w.e."
    melt_glac_monthly.comment = "total monthly melt over the entire glacier divided by the glacier area"
    frontal_ablation_glac_monthly = netcdf_output.createVariable('frontal_ablation_glac_monthly', np.float64, 
                                                                 ('glaciertable', 'time'))
    frontal_ablation_glac_monthly.standard_name = "monthly glacier-wide specific frontal ablation"
    frontal_ablation_glac_monthly.units = "m w.e."
    frontal_ablation_glac_monthly.comment = ("mass losses due to calving, subaerial frontal melting, sublimation above "
                                             + "the waterline and subaqueous frontal melting below the waterline")
    massbal_clim_mwe_glac_monthly = netcdf_output.createVariable('massbal_clim_mwe_glac_monthly', np.float64, 
                                                                 ('glaciertable', 'time'))
    massbal_clim_mwe_glac_monthly.standard_name = "monthly glacier-wide specific climatic mass balance"
    massbal_clim_mwe_glac_monthly.units = "m w.e."
    massbal_clim_mwe_glac_monthly.comment = ("climatic mass balance is the sum of the surface mass balance and the "
                                             + "internal mass balance and accounts for the climatic mass loss over the "
                                             + "area of the entire bin") 
    area_glac_monthly = netcdf_output.createVariable('area_glac_monthly', np.float64, ('glaciertable', 'time'))
    area_glac_monthly.standard_name = "glacier area"
    area_glac_monthly.units = "km**2"
    area_glac_monthly.comment = ("area for a given timestep is the area that was used for the duration of the timestep,"
                                 + " i.e., it is the area at the start of the time step")
    ELA_glac_monthly = netcdf_output.createVariable('ELA_glac_monthly', np.float64, ('glaciertable', 'time'))
    ELA_glac_monthly.standard_name = "transient equilibrium line altitude"
    ELA_glac_monthly.units = "m a.s.l."
    ELA_glac_monthly.comment = ("equilibrium line altitude is the elevation that separates a positive and negative mass"
                                + "balance on the glacier")
    AAR_glac_monthly = netcdf_output.createVariable('AAR_glac_monthly', np.float64, ('glaciertable', 'time'))
    AAR_glac_monthly.standard_name = "transient accumulation-area ratio"
    AAR_glac_monthly.units = "%"
    AAR_glac_monthly.comment = ("accumulation-area ratio is the percentage of the area of the accumulation zone to the "
                                + "area of the glacier")
    snowline_glac_monthly = netcdf_output.createVariable('snowline_glac_monthly', np.float64, ('glaciertable', 'time'))
    snowline_glac_monthly.standard_name = "transient snowline"
    snowline_glac_monthly.units = "m a.s.l."
    snowline_glac_monthly.comment = ("transient snowline is the line separating the snow from ice/firn at the end of "
                                     + "each month")
    runoff_glac_monthly = netcdf_output.createVariable('runoff_glac_monthly', np.float64, ('glaciertable', 'time'))
    runoff_glac_monthly.long_name = "monthly glacier runoff"
    runoff_glac_monthly.standard_name = "glacier runoff"
    runoff_glac_monthly.units = "m**3"
    runoff_glac_monthly.comment = "runoff is the total volume of water that is leaving the glacier each month"
    # Annual glacier-wide variables being recorded for each glacier
    acc_glac_annual = netcdf_output.createVariable('acc_glac_annual', np.float64, ('glaciertable', 'year'))
    acc_glac_annual.standard_name = "annual glacier-wide specific accumulation"
    acc_glac_annual.units = "m w.e."
    acc_glac_annual.comment = "total annual accumulation over the entire glacier divided by the glacier area"
    refreeze_glac_annual = netcdf_output.createVariable('refreeze_glac_annual', np.float64, ('glaciertable', 'year'))
    refreeze_glac_annual.standard_name = "annual glacier-wide specific refreeze"
    refreeze_glac_annual.units = "m w.e."
    refreeze_glac_annual.comment = "total annual refreeze over the entire glacier divided by the glacier area"
    melt_glac_annual = netcdf_output.createVariable('melt_glac_annual', np.float64, ('glaciertable', 'year'))
    melt_glac_annual.standard_name = "annual glacier-wide specific melt"
    melt_glac_annual.units = "m w.e."
    melt_glac_annual.comment = "total annual melt over the entire glacier divided by the glacier area"
    frontal_ablation_glac_annual = netcdf_output.createVariable('frontal_ablation_glac_annual', np.float64, 
                                                                ('glaciertable', 'year'))
    frontal_ablation_glac_annual.standard_name = "annual glacier-wide specific frontal ablation"
    frontal_ablation_glac_annual.units = "m w.e."
    frontal_ablation_glac_annual.comment = ("mass losses due to calving, subaerial frontal melting, sublimation above "
                                            + "the waterline and subaqueous frontal melting below the waterline")
    massbal_clim_mwe_glac_annual = netcdf_output.createVariable('massbal_clim_mwe_glac_annual', np.float64, 
                                                                ('glaciertable', 'year'))
    massbal_clim_mwe_glac_annual.standard_name = "annual glacier-wide specific climatic mass balance"
    massbal_clim_mwe_glac_annual.units = "m w.e."
    massbal_clim_mwe_glac_annual.comment = ("climatic mass balance is the sum of the surface mass balance and the "
                                            + "internal mass balance and accounts for the climatic mass loss over the "
                                            + "area of the entire bin") 
    massbal_total_mwe_glac_annual = netcdf_output.createVariable('massbal_total_mwe_glac_annual', np.float64, 
                                                                 ('glaciertable', 'year'))
    massbal_total_mwe_glac_annual.standard_name = "annual glacier-wide specific climatic mass balance"
    massbal_total_mwe_glac_annual.units = "m w.e."
    massbal_total_mwe_glac_annual.comment = ("specific total mass balance is the total conservation of mass within an "
                                             + "elevation bin, i.e., it includes the climatic mass balance, flow of ice"
                                             + " into or out of the bin, calving, bed losses/gains, and any additional "
                                             + "sources of mass loss or gain.")
    area_glac_annual = netcdf_output.createVariable('area_glac_annual', np.float64, ('glaciertable', 'year'))
    area_glac_annual.standard_name = "glacier area"
    area_glac_annual.units = "km**2"
    area_glac_annual.comment = ("area for a given timestep is the area that was used for the duration of the timestep, "
                                + "i.e., it is the area at the start of the time step")
    volume_glac_annual = netcdf_output.createVariable('volume_glac_annual', np.float64, ('glaciertable', 'year'))
    volume_glac_annual.standard_name = "glacier volume"
    volume_glac_annual.units = "km**3"
    volume_glac_annual.comment = ("volume is the total volume of ice in the elevation bin that was used for the "
                                  + "duration of the timestep, i.e., it is the volume at the start of the time step")
    ELA_glac_annual = netcdf_output.createVariable('ELA_glac_annual', np.float64, ('glaciertable', 'year'))
    ELA_glac_annual.standard_name = "annual equilibrium line altitude"
    ELA_glac_annual.units = "m a.s.l."
    ELA_glac_annual.comment = ("annual equilibrium line altitude is the elevation that separates a positive and "
                               + "negative annual mass balance on the glacier")
    AAR_glac_annual = netcdf_output.createVariable('AAR_glac_annual', np.float64, ('glaciertable', 'year'))
    AAR_glac_annual.standard_name = "annual accumulation-area ratio"
    AAR_glac_annual.units = "m a.s.l."
    AAR_glac_annual.comment = ("annual accumulation-area ratio is the percentage of the area of the accumulation zone "
                               + "to the area of the glacier based on the annual mass balance")
    snowline_glac_annual = netcdf_output.createVariable('snowline_glac_annual', np.float64, ('glaciertable', 'year'))
    snowline_glac_annual.standard_name = "annual snowline"
    snowline_glac_annual.units = "m a.s.l."
    snowline_glac_annual.comment = ("annual snowline is the snowline during the month of minimum snow coverm, which "
                                    + "usually occurs at the end of the ablation season.")
    runoff_glac_annual = netcdf_output.createVariable('runoff_glac_annual', np.float64, ('glaciertable', 'year'))
    runoff_glac_annual.standard_name = "annual glacier runoff"
    runoff_glac_annual.units = "m a.s.l."
    runoff_glac_annual.comment = "annual runoff is jthe total volume of water that is leaving the glacier each year"
    print('output netcdf file has been created.')


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
    if input.timestep == 'monthly':
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
        if input.timestep == 'monthly':
            runoff_output = runoff_monthly.groupby(np.arange(runoff_monthly.shape[1]) // 12, axis=1).sum()
        elif input.timestep == 'daily':
            print('\nError: need to code the groupbyyearsum for daily timestep for computing annual output products.'
                  'Exiting the model run.\n')
            exit()
        runoff_output.index = [GlacNo]
        runoff_output.columns = glac_bin_area.columns
    elif output_interval == 'seasonal':
        # Determine seasons based on summer and winter start months
        ds_month = pd.Series(runoff_monthly.columns.values).dt.month
        ds_season = (pd.Series(runoff_monthly.columns.values).dt.month
                     .apply(lambda x: 'summer' if (x >= input.summer_month_start and x < input.winter_month_start) else 
                            'winter'))
        ds_wateryear = pd.Series(runoff_monthly.columns.values).dt.year
        runoff_monthly_copy = runoff_monthly.transpose()
        runoff_monthly_copy = runoff_monthly_copy.reset_index()
        runoff_monthly = pd.concat([runoff_monthly_copy, ds_month, ds_season, ds_wateryear], axis=1)
        runoff_monthly.columns = ['time','runoff', 'month', 'season', 'wateryear']
        # if water year is being used, then convert the years to water years
        if input.option_wateryear == 1:
            runoff_monthly['wateryear'] = (runoff_monthly['wateryear']
                                           .where(runoff_monthly['month'] < input.wateryear_month_start, 
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