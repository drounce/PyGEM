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
import numpy as np
import netCDF4 as nc
from time import strftime
from datetime import datetime
#========= IMPORT COMMON VARIABLES FROM MODEL INPUT ==========================
import pygem_input as input

#========= DESCRIPTION OF VARIABLES (alphabetical order) =====================
# Add description of variables...

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
        #  annual variables for each bin (area, icethickness, width, surfacetype)
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
        area_bin_annual.comment = "the area that was used for the duration of the year"
        icethickness_bin_annual = netcdf_output.createVariable('icethickness_bin_annual', np.float64, 
                                                               ('glacier', 'binelev', 'year_plus1'))
        icethickness_bin_annual.standard_name = "ice thickness"
        icethickness_bin_annual.unit = "m ice"
        icethickness_bin_annual.comment = "the ice thickness that was used for the duration of the year"
        width_bin_annual = netcdf_output.createVariable('width_bin_annual', np.float64, 
                                                        ('glacier', 'binelev', 'year_plus1'))
        width_bin_annual.standard_name = "glacier width"
        width_bin_annual.unit = "km"
        width_bin_annual.comment = "the width that was used for the duration of the year"
        surfacetype_bin_annual = netcdf_output.createVariable('surfacetype_bin_annual', np.float64, 
                                                              ('glacier', 'binelev', 'year'))
        surfacetype_bin_annual.standard_name = "surface type"
        surfacetype_bin_annual.comment = "surface types: 0 = off-glacier, 1 = ice, 2 = snow, 3 = firn, 4 = debris"
    elif input.output_package == 2:
        # Package 2 "Glaciologist Package" output [units: m w.e. unless otherwise specified]:
        #  monthly glacier-wide variables (acc, refreeze, melt, frontalablation, massbal_total, runoff, snowline)
        #  annual glacier-wide variables (area, volume, ELA)
        acc_glac_monthly = netcdf_output.createVariable('acc_glac_monthly', np.float64, ('glacier', 'time'))
        acc_glac_monthly.standard_name = "glacier-wide accumulation"
        acc_glac_monthly.units = "m w.e."
        refreeze_glac_monthly = netcdf_output.createVariable('refreeze_glac_monthly', np.float64, ('glacier', 'time'))
        refreeze_glac_monthly.standard_name = "glacier-wide refreeze"
        refreeze_glac_monthly.units = "m w.e."
        melt_glac_monthly = netcdf_output.createVariable('melt_glac_monthly', np.float64, ('glacier', 'time'))
        melt_glac_monthly.standard_name = "glacier-wide melt"
        melt_glac_monthly.units = "m w.e."
        frontalablation_glac_monthly = netcdf_output.createVariable('frontalablation_glac_monthly', np.float64, 
                                                                    ('glacier', 'time'))
        frontalablation_glac_monthly.standard_name = "glacier-wide frontal ablation"
        frontalablation_glac_monthly.units = "m w.e."
        frontalablation_glac_monthly.comment = ("mass losses from calving, subaerial frontal melting, sublimation above"
                                                + " the waterline and subaqueous frontal melting below the waterline")
        massbaltotal_glac_monthly = netcdf_output.createVariable('massbaltotal_glac_monthly', np.float64, ('glacier', 
                                                                                                           'time'))
        massbaltotal_glac_monthly.standard_name = "glacier-wide total mass balance"
        massbaltotal_glac_monthly.units = "m w.e."
        massbaltotal_glac_monthly.comment = ("total mass balance is the sum of the climatic mass balance and frontal "
                                             + "ablation.") 
        runoff_glac_monthly = netcdf_output.createVariable('runoff_glac_monthly', np.float64, ('glacier', 'time'))
        runoff_glac_monthly.standard_name = "glacier runoff"
        runoff_glac_monthly.units = "m**3"
        runoff_glac_monthly.comment = "runoff from the glacier terminus, which moves over time"
        snowline_glac_monthly = netcdf_output.createVariable('snowline_glac_monthly', np.float64, ('glacier', 'time'))
        snowline_glac_monthly.standard_name = "transient snowline"
        snowline_glac_monthly.units = "m a.s.l."
        snowline_glac_monthly.comment = "transient snowline is the line separating the snow from ice/firn"
        area_glac_annual = netcdf_output.createVariable('area_glac_annual', np.float64, ('glacier', 'year_plus1'))
        area_glac_annual.standard_name = "glacier area"
        area_glac_annual.units = "km**2"
        area_glac_annual.comment = "the area that was used for the duration of the year"
        volume_glac_annual = netcdf_output.createVariable('volume_glac_annual', np.float64, ('glacier', 'year_plus1'))
        volume_glac_annual.standard_name = "glacier volume"
        volume_glac_annual.units = "km**3 ice"
        volume_glac_annual.comment = "the volume based on area and ice thickness used for that year"
        ELA_glac_annual = netcdf_output.createVariable('ELA_glac_annual', np.float64, ('glacier', 'year'))
        ELA_glac_annual.standard_name = "annual equilibrium line altitude"
        ELA_glac_annual.units = "m a.s.l."
        ELA_glac_annual.comment = "equilibrium line altitude is the elevation where the climatic mass balance is zero"
    netcdf_output.close()


def netcdfwrite(regionO1_number, glac, main_glac_rgi, elev_bins, glac_bin_temp, glac_bin_prec, glac_bin_acc, 
                glac_bin_refreeze, glac_bin_snowpack, glac_bin_melt, glac_bin_frontalablation, glac_bin_massbalclim, 
                glac_bin_massbalclim_annual, glac_bin_area_annual, glac_bin_icethickness_annual, glac_bin_width_annual,
                glac_bin_surfacetype_annual):
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
        # Package 1 "Raw Package" output [units: m w.e. unless otherwise specified]:
        #  monthly variables for each bin (temp, prec, acc, refreeze, snowpack, melt, frontalablation, massbal_clim)
        #  annual variables for each bin (area, icethickness, surfacetype)
        # Column start and end for monthly variables
        colstart = input.spinupyears * 12
        colend = glac_bin_temp.shape[1] + 1
        # Record all the output variables
        netcdf_output.variables['temp_bin_monthly'][glac,:,:] = glac_bin_temp[:,colstart:colend]
        netcdf_output.variables['prec_bin_monthly'][glac,:,:] = glac_bin_prec[:,colstart:colend]
        netcdf_output.variables['acc_bin_monthly'][glac,:,:] = glac_bin_acc[:,colstart:colend]
        netcdf_output.variables['refreeze_bin_monthly'][glac,:,:] = glac_bin_refreeze[:,colstart:colend]
        netcdf_output.variables['snowpack_bin_monthly'][glac,:,:] = glac_bin_snowpack[:,colstart:colend]
        netcdf_output.variables['melt_bin_monthly'][glac,:,:] = glac_bin_melt[:,colstart:colend]
        netcdf_output.variables['frontalablation_bin_monthly'][glac,:,:] = glac_bin_frontalablation[:,colstart:colend]
        netcdf_output.variables['massbalclim_bin_monthly'][glac,:,:] = glac_bin_massbalclim[:,colstart:colend]
        netcdf_output.variables['area_bin_annual'][glac,:,:] = (
                glac_bin_area_annual[input.spinupyears:glac_bin_area_annual.shape[1]+1])
        netcdf_output.variables['icethickness_bin_annual'][glac,:,:] = (
                glac_bin_icethickness_annual[input.spinupyears:glac_bin_area_annual.shape[1]+1])
        netcdf_output.variables['width_bin_annual'][glac,:,:] = (
                glac_bin_width_annual[input.spinupyears:glac_bin_area_annual.shape[1]+1])
        netcdf_output.variables['surfacetype_bin_annual'][glac,:,:] = (
                glac_bin_surfacetype_annual[:,input.spinupyears:glac_bin_area_annual.shape[1]+1])
    elif input.output_package == 2:
        # Package 2 "Glaciologist Package" output [units: m w.e. unless otherwise specified]:
        #  monthly glacier-wide variables (acc, refreeze, melt, frontalablation, massbal_total, runoff, snowline)
        #  annual glacier-wide variables (area, volume, ELA)
        # Compute that desired output
        glac_bin_area = glac_bin_area_annual[:,0:glac_bin_area_annual.shape[1]-1].repeat(12,axis=1)
        glac_wide_area = glac_bin_area.sum(axis=0)
        glac_wide_prec = (glac_bin_prec * glac_bin_area).sum(axis=0) / glac_wide_area[np.newaxis,:]
        glac_wide_acc = (glac_bin_acc * glac_bin_area).sum(axis=0) / glac_wide_area[np.newaxis,:]
        glac_wide_refreeze = (glac_bin_refreeze * glac_bin_area).sum(axis=0) / glac_wide_area[np.newaxis,:]
        glac_wide_melt = (glac_bin_melt * glac_bin_area).sum(axis=0) / glac_wide_area[np.newaxis,:]
        glac_wide_frontalablation = ((glac_bin_frontalablation * glac_bin_area).sum(axis=0) / 
                                     glac_wide_area[np.newaxis,:])
        glac_wide_massbalclim = glac_wide_acc + glac_wide_refreeze - glac_wide_melt
        glac_wide_massbaltotal = glac_wide_massbalclim - glac_wide_frontalablation
        glac_wide_runoff = (glac_wide_prec + glac_wide_melt - glac_wide_refreeze) * glac_wide_area * (1000)**2
        #  units: (m + m w.e. - m w.e.) * km**2 * (1000 m / 1 km)**2 = m**3
        glac_wide_snowline = (glac_bin_snowpack > 0).argmax(axis=0)
        glac_wide_snowline[glac_wide_snowline > 0] = (elev_bins[glac_wide_snowline[glac_wide_snowline > 0]] - 
                                                      input.binsize/2)
        glac_wide_area_annual = glac_bin_area_annual.sum(axis=0)
        glac_wide_volume_annual = (glac_bin_area_annual * glac_bin_icethickness_annual / 1000).sum(axis=0)
        glac_wide_ELA_annual = (glac_bin_massbalclim_annual > 0).argmax(axis=0)
        glac_wide_ELA_annual[glac_wide_ELA_annual > 0] = (elev_bins[glac_wide_ELA_annual[glac_wide_ELA_annual > 0]] - 
                                                          input.binsize/2)
        # Column start and end for monthly variables
        colstart = input.spinupyears * 12
        colend = glac_bin_temp.shape[1] + 1
        netcdf_output.variables['acc_glac_monthly'][glac,:] = glac_wide_acc[:,colstart:colend]
        netcdf_output.variables['refreeze_glac_monthly'][glac,:] = glac_wide_refreeze[:,colstart:colend]
        netcdf_output.variables['melt_glac_monthly'][glac,:] = glac_wide_melt[:,colstart:colend]
        netcdf_output.variables['frontalablation_glac_monthly'][glac,:] = glac_wide_frontalablation[:,colstart:colend]
        netcdf_output.variables['massbaltotal_glac_monthly'][glac,:] = glac_wide_massbaltotal[:,colstart:colend]
        netcdf_output.variables['runoff_glac_monthly'][glac,:] = glac_wide_runoff[:,colstart:colend]
        netcdf_output.variables['snowline_glac_monthly'][glac,:] = glac_wide_snowline[colstart:colend]
        netcdf_output.variables['area_glac_annual'][glac,:] = (
                glac_wide_area_annual[input.spinupyears:glac_bin_area_annual.shape[1]+1])
        netcdf_output.variables['volume_glac_annual'][glac,:] = (
                glac_wide_volume_annual[input.spinupyears:glac_bin_area_annual.shape[1]+1])
        netcdf_output.variables['ELA_glac_annual'][glac,:] = (
                glac_wide_ELA_annual[input.spinupyears:glac_bin_area_annual.shape[1]])
    # Close the netcdf file
    netcdf_output.close()


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
    
#%% Compute different variables for different output packages based on model-generated variables
#    # Record variables that are specified by the user to be output
#    # Monthly area [km**2] at each bin
#    glac_bin_area = glac_bin_area_annual[:,0:glac_bin_area_annual.shape[1]-1].repeat(12,axis=1)
#    # Monthly ice thickness [m ice] at each bin
#    glac_bin_icethickness = glac_bin_icethickness_annual[:,0:glac_bin_icethickness_annual.shape[1]-1].repeat(12,axis=1)
#
#    # Annual outputs at each bin:
#    # Annual volume [km**3]
#    glac_bin_volume_annual = glac_bin_area_annual * glac_bin_icethickness_annual / 1000
#    # Annual total specific mass balance [m3] (mass balance after mass redistribution)
#    glac_bin_massbal_total_m3_annual = (glac_bin_volume_annual - np.roll(glac_bin_volume_annual,-1,axis=1))
#    # Annual accumulation [m3]
##    glac_bin_acc_annual = glac_bin_acc.reshape(-1,12).sum(axis=1).reshape(-1,annual_columns.shape[0])
#    glac_bin_acc_annual = (glac_bin_acc.reshape(-1,12).sum(axis=1).reshape(-1,annual_columns.shape[0]) * 
#                           glac_bin_area_annual[:,0:glac_bin_area_annual.shape[1]-1] * 1000**2)
#    # Annual refreeze [m3]
##    glac_bin_refreeze_annual = glac_bin_refreeze.reshape(-1,12).sum(axis=1).reshape(-1,annual_columns.shape[0])
#    glac_bin_refreeze_annual = (glac_bin_refreeze.reshape(-1,12).sum(axis=1).reshape(-1,annual_columns.shape[0]) * 
#                                glac_bin_area_annual[:,0:glac_bin_area_annual.shape[1]-1] * 1000**2)
#    # Annual melt [m3]
##    glac_bin_melt_annual = glac_bin_melt.reshape(-1,12).sum(axis=1).reshape(-1,annual_columns.shape[0])
#    glac_bin_melt_annual = (glac_bin_melt.reshape(-1,12).sum(axis=1).reshape(-1,annual_columns.shape[0]) * 
#                            glac_bin_area_annual[:,0:glac_bin_area_annual.shape[1]-1] * 1000**2)
#    # Annual frontal ablation [m3]
##    glac_bin_frontalablation_annual = (
##            glac_bin_frontalablation.reshape(-1,12).sum(axis=1).reshape(-1,annual_columns.shape[0]))
#    glac_bin_frontalablation_annual = (
#            glac_bin_frontalablation.reshape(-1,12).sum(axis=1).reshape(-1,annual_columns.shape[0]) * 
#            glac_bin_area_annual[:,0:glac_bin_area_annual.shape[1]-1] * 1000**2)
#    # Annual precipitation [m3]
##    glac_bin_prec_annual = glac_bin_prec.reshape(-1,12).sum(axis=1).reshape(-1,annual_columns.shape[0])
#    glac_bin_prec_annual = (glac_bin_prec.reshape(-1,12).sum(axis=1).reshape(-1,annual_columns.shape[0]) * 
#                            glac_bin_area_annual[:,0:glac_bin_area_annual.shape[1]-1] * 1000**2)
#    
#


#    # Refreeze [m w.e.]
#    glac_wide_refreeze = np.zeros(glac_wide_area.shape)
#    glac_wide_refreeze[glac_wide_area > 0] = ((glac_bin_refreeze * glac_bin_area).sum(axis=0)[glac_wide_area > 0] / 
#                                              glac_wide_area[glac_wide_area > 0])
#    # Melt [m w.e.]
#    glac_wide_melt = np.zeros(glac_wide_area.shape)
#    glac_wide_melt[glac_wide_area > 0] = ((glac_bin_melt * glac_bin_area).sum(axis=0)[glac_wide_area > 0] / 
#                                          glac_wide_area[glac_wide_area > 0])
#    # Frontal ablation [m w.e.]
#    glac_wide_frontalablation = np.zeros(glac_wide_area.shape)
#    glac_wide_frontalablation[glac_wide_area > 0] = (
#            (glac_bin_frontalablation * glac_bin_area).sum(axis=0)[glac_wide_area > 0] / 
#            glac_wide_area[glac_wide_area > 0])
#    # Mass balance [m w.e.]
#    #  glacier-wide climatic and total mass balance are the same; use climatic since its required to run the model
#    glac_wide_massbal_mwe = np.zeros(glac_wide_area.shape)
#    glac_wide_massbal_mwe[glac_wide_area > 0] = (
#            (glac_bin_massbal_clim_mwe * glac_bin_area).sum(axis=0)[glac_wide_area > 0] / 
#            glac_wide_area[glac_wide_area > 0])
#    # Melt [m w.e.]
#    glac_wide_melt = np.zeros(glac_wide_area.shape)
#    glac_wide_melt[glac_wide_area > 0] = ((glac_bin_melt * glac_bin_area).sum(axis=0)[glac_wide_area > 0] / 
#                                          glac_wide_area[glac_wide_area > 0])
#    # Precipitation [m]
#    glac_wide_prec = np.zeros(glac_wide_area.shape)
#    glac_wide_prec[glac_wide_area > 0] = ((glac_bin_prec * glac_bin_area).sum(axis=0)[glac_wide_area > 0] / 
#                                          glac_wide_area[glac_wide_area > 0])
#    # Volume [km**3]
#    glac_wide_volume = (glac_bin_area * glac_bin_icethickness / 1000).sum(axis=0)
#            
#    # Annual glacier-wide Parameters:
#    # Annual volume [km**3]
#    glac_wide_volume_annual = (glac_bin_area_annual * glac_bin_icethickness_annual / 1000).sum(axis=0)
    
    
#    # Annual accumulation [m w.e.]
#    glac_wide_acc_annual = 
#    # Annual refreeze [m w.e.]
#    glac_wide_refreeze_annual = 
#    # Annual melt [m w.e.]
#    glac_wide_melt_annual = 
#    # Annual frontal ablation [m w.e.]
#    glac_wide_frontalablation_annual = 
#    # Annual precipitation [m]
#    glac_wide_prec_annual = 