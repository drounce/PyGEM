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
import pandas as pd 
import netCDF4 as nc
from time import strftime
from datetime import datetime
import matplotlib.pyplot as plt
#========= IMPORT COMMON VARIABLES FROM MODEL INPUT ==========================
import pygem_input as input
import cartopy

#========= DESCRIPTION OF VARIABLES (alphabetical order) =====================
# Add description of variables...

#========= FUNCTIONS (alphabetical order) ===================================
#%%===== NETCDF FUNCTIONS =============================================================================================
def netcdfcreate(regionO1_number, main_glac_hyps, dates_table, annual_columns):
    """Create a netcdf file to store the desired output
    Output: empty netcdf file with the proper setup to be filled in by the model
    """
    # Netcdf file path and name
    filename = input.netcdf_filenameprefix + str(regionO1_number) + '_' + str(strftime("%Y%m%d")) + '.nc'
    fullfile = input.output_filepath + filename
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
    glacierparameter = netcdf_output.createDimension('glacierparameter', 11)
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
    glacierparameters[:] = np.array(['RGIID','lat','lon','lrgcm','lrglac','precfactor','precgrad','ddfice',
                                     'ddfsnow','tempsnow', 'tempchange'])
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
        #  monthly glacier-wide variables (prec, acc, refreeze, melt, frontalablation, massbal_total, runoff, snowline)
        #  annual glacier-wide variables (area, volume, ELA)
        prec_glac_monthly = netcdf_output.createVariable('prec_glac_monthly', np.float64, ('glacier', 'time'))
        prec_glac_monthly.standard_name = "glacier-wide precipitation"
        prec_glac_monthly.units = "m"
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


def netcdfwrite(regionO1_number, glac, modelparameters, glacier_rgi_table, elev_bins, glac_bin_temp, glac_bin_prec, 
                glac_bin_acc, glac_bin_refreeze, glac_bin_snowpack, glac_bin_melt, glac_bin_frontalablation, 
                glac_bin_massbalclim, glac_bin_massbalclim_annual, glac_bin_area_annual, glac_bin_icethickness_annual, 
                glac_bin_width_annual,glac_bin_surfacetype_annual):
    """Write to the netcdf file that has already been generated to store the desired output
    Output: netcdf with desired variables filled in
    """
    # Netcdf file path and name
    filename = input.netcdf_filenameprefix + str(regionO1_number) + '_' + str(strftime("%Y%m%d")) + '.nc'
    fullfile = input.output_filepath + filename
    # Open netcdf file to write to existing file ('r+')
    netcdf_output = nc.Dataset(fullfile, 'r+')
    # Record the variables for each glacier (remove data associated with spinup years)
    # glaciers parameters
    netcdf_output.variables['glacierparameter'][glac,:] = np.array([glacier_rgi_table.loc['RGIId'],
        glacier_rgi_table.loc[input.lat_colname],glacier_rgi_table.loc[input.lon_colname], modelparameters[0], 
        modelparameters[1], modelparameters[2], modelparameters[3], modelparameters[4], modelparameters[5], 
        modelparameters[6], modelparameters[7]])
    if input.output_package == 1:
        # Package 1 "Raw Package" output [units: m w.e. unless otherwise specified]:
        #  monthly variables for each bin (temp, prec, acc, refreeze, snowpack, melt, frontalablation, massbal_clim)
        #  annual variables for each bin (area, icethickness, surfacetype)
        # Write variables to netcdf
        netcdf_output.variables['temp_bin_monthly'][glac,:,:] = glac_bin_temp
        netcdf_output.variables['prec_bin_monthly'][glac,:,:] = glac_bin_prec
        netcdf_output.variables['acc_bin_monthly'][glac,:,:] = glac_bin_acc
        netcdf_output.variables['refreeze_bin_monthly'][glac,:,:] = glac_bin_refreeze
        netcdf_output.variables['snowpack_bin_monthly'][glac,:,:] = glac_bin_snowpack
        netcdf_output.variables['melt_bin_monthly'][glac,:,:] = glac_bin_melt
        netcdf_output.variables['frontalablation_bin_monthly'][glac,:,:] = glac_bin_frontalablation
        netcdf_output.variables['massbalclim_bin_monthly'][glac,:,:] = glac_bin_massbalclim
        netcdf_output.variables['area_bin_annual'][glac,:,:] = glac_bin_area_annual
        netcdf_output.variables['icethickness_bin_annual'][glac,:,:] = glac_bin_icethickness_annual
        netcdf_output.variables['width_bin_annual'][glac,:,:] = glac_bin_width_annual
        netcdf_output.variables['surfacetype_bin_annual'][glac,:,:] = glac_bin_surfacetype_annual
    elif input.output_package == 2:
        # Package 2 "Glaciologist Package" output [units: m w.e. unless otherwise specified]:
        #  monthly glacier-wide variables (prec, acc, refreeze, melt, frontalablation, massbal_total, runoff, snowline)
        #  annual glacier-wide variables (area, volume, ELA)
        # Preset desired output (needed to avoid dividing by zero)
        glac_wide_prec = np.zeros(glac_bin_temp.shape[1])
        glac_wide_acc = np.zeros(glac_bin_temp.shape[1])
        glac_wide_refreeze = np.zeros(glac_bin_temp.shape[1])
        glac_wide_melt = np.zeros(glac_bin_temp.shape[1])
        glac_wide_frontalablation = np.zeros(glac_bin_temp.shape[1])
        # Compute desired output
        glac_bin_area = glac_bin_area_annual[:,0:glac_bin_area_annual.shape[1]-1].repeat(12,axis=1)
        glac_wide_area = glac_bin_area.sum(axis=0)
        glac_wide_prec_mkm2 = (glac_bin_prec * glac_bin_area).sum(axis=0)
        glac_wide_prec[glac_wide_prec_mkm2 > 0] = (glac_wide_prec_mkm2[glac_wide_prec_mkm2 > 0] / 
                                                   glac_wide_area[glac_wide_prec_mkm2 > 0])
        glac_wide_acc_mkm2 = (glac_bin_acc * glac_bin_area).sum(axis=0)
        glac_wide_acc[glac_wide_acc_mkm2 > 0] = (glac_wide_acc_mkm2[glac_wide_acc_mkm2 > 0] / 
                                                 glac_wide_area[glac_wide_acc_mkm2 > 0])
        glac_wide_refreeze_mkm2 = (glac_bin_refreeze * glac_bin_area).sum(axis=0)
        glac_wide_refreeze[glac_wide_refreeze_mkm2 > 0] = (glac_wide_refreeze_mkm2[glac_wide_refreeze_mkm2 > 0] / 
                                                           glac_wide_area[glac_wide_refreeze_mkm2 > 0])
        glac_wide_melt_mkm2 = (glac_bin_melt * glac_bin_area).sum(axis=0)
        glac_wide_melt[glac_wide_melt_mkm2 > 0] = (glac_wide_melt_mkm2[glac_wide_melt_mkm2 > 0] / 
                                                   glac_wide_area[glac_wide_melt_mkm2 > 0])
        glac_wide_frontalablation_mkm2 = (glac_bin_frontalablation * glac_bin_area).sum(axis=0)
        glac_wide_frontalablation[glac_wide_frontalablation_mkm2 > 0] = (
                glac_wide_frontalablation_mkm2[glac_wide_frontalablation_mkm2 > 0] / 
                glac_wide_area[glac_wide_frontalablation_mkm2 > 0])
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
        # Write variables to netcdf
        netcdf_output.variables['prec_glac_monthly'][glac,:] = glac_wide_prec
        netcdf_output.variables['acc_glac_monthly'][glac,:] = glac_wide_acc
        netcdf_output.variables['refreeze_glac_monthly'][glac,:] = glac_wide_refreeze
        netcdf_output.variables['melt_glac_monthly'][glac,:] = glac_wide_melt
        netcdf_output.variables['frontalablation_glac_monthly'][glac,:] = glac_wide_frontalablation
        netcdf_output.variables['massbaltotal_glac_monthly'][glac,:] = glac_wide_massbaltotal
        netcdf_output.variables['runoff_glac_monthly'][glac,:] = glac_wide_runoff
        netcdf_output.variables['snowline_glac_monthly'][glac,:] = glac_wide_snowline
        netcdf_output.variables['area_glac_annual'][glac,:] = glac_wide_area_annual
        netcdf_output.variables['volume_glac_annual'][glac,:] = glac_wide_volume_annual
        netcdf_output.variables['ELA_glac_annual'][glac,:] = glac_wide_ELA_annual
    # Close the netcdf file
    netcdf_output.close()


#%%===== PLOT FUNCTIONS =============================================================================================
def plot_latlonvar(lons, lats, variable, rangelow, rangehigh, title, xlabel, ylabel, colormap, east, west, south, north, 
                   xtick, ytick):
    """
    Plot a variable according to its latitude and longitude
    """
    # Create the projection
    ax = plt.axes(projection=cartopy.crs.PlateCarree())
    # Add country borders for reference
    ax.add_feature(cartopy.feature.BORDERS)
    # Set the extent
    ax.set_extent([east, west, south, north], cartopy.crs.PlateCarree())
    # Label title, x, and y axes
    plt.title(title)
    ax.set_xticks(np.arange(east,west+1,xtick), cartopy.crs.PlateCarree())
    ax.set_yticks(np.arange(south,north+1,ytick), cartopy.crs.PlateCarree())
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # Plot the data 
    plt.scatter(lons, lats, c=variable, cmap=colormap)
    #  plotting x, y, size [s=__], color bar [c=__]
    plt.clim(rangelow,rangehigh)
    #  set the range of the color bar
    plt.colorbar(fraction=0.02, pad=0.04)
    #  fraction resizes the colorbar, pad is the space between the plot and colorbar
    plt.show()
    

def plot_caloutput(data):
    """
    Plot maps and histograms of the calibration parameters to visualize results
    """
    # Set extent
    east = int(round(data['CenLon'].min())) - 1
    west = int(round(data['CenLon'].max())) + 1
    south = int(round(data['CenLat'].min())) - 1
    north = int(round(data['CenLat'].max())) + 1
    xtick = 1
    ytick = 1
    # Select relevant data
    lats = data['CenLat'][:]
    lons = data['CenLon'][:]
    precfactor = data['precfactor'][:]
    tempchange = data['tempchange'][:]
    ddfsnow = data['ddfsnow'][:]
    calround = data['calround'][:]
    massbal = data['MB_geodetic_mwea']
    # Plot regional maps
    plot_latlonvar(lons, lats, massbal, 'Geodetic mass balance [mwea]', 'longitude [deg]', 'latitude [deg]', east, west, 
               south, north, xtick, ytick)
    plot_latlonvar(lons, lats, precfactor, 'precipitation factor', 'longitude [deg]', 'latitude [deg]', east, west, 
                   south, north, xtick, ytick)
    plot_latlonvar(lons, lats, tempchange, 'Temperature bias [degC]', 'longitude [deg]', 'latitude [deg]', east, west, 
                   south, north, xtick, ytick)
    plot_latlonvar(lons, lats, ddfsnow, 'DDF_snow [m w.e. d-1 degC-1]', 'longitude [deg]', 'latitude [deg]', east, west, 
                   south, north, xtick, ytick)
    plot_latlonvar(lons, lats, calround, 'Calibration round', 'longitude [deg]', 'latitude [deg]', east, west, 
                   south, north, xtick, ytick)
    # Plot histograms
    data.hist(column='MB_difference_mwea', bins=50)
    plt.title('Mass Balance Difference [mwea]')
    data.hist(column='precfactor', bins=50)
    plt.title('Precipitation factor [-]')
    data.hist(column='tempchange', bins=50)
    plt.title('Temperature bias [degC]')
    data.hist(column='ddfsnow', bins=50)
    plt.title('DDFsnow [mwe d-1 degC-1]')
    plt.xticks(rotation=60)
    data.hist(column='calround', bins = [0.5, 1.5, 2.5, 3.5])
    plt.title('Calibration round')
    plt.xticks([1, 2, 3])

#%%===== GRID SEARCH ======
#data = nc.Dataset(input.main_directory + '/../Output/calibration_gridsearchcoarse_R15_20180319.nc', 'r+')
glac = 3732
massbaltotal_mwea = data['massbaltotal_glac_monthly'][glac,:,:].sum(axis=1) / (main_glac_calmassbal[glac,2] - main_glac_calmassbal[glac,1] + 1)
massbaltotal_mwea_cal = main_glac_calmassbal[glac,0]
difference = massbaltotal_mwea - massbaltotal_mwea_cal
plt.scatter(massbaltotal_mwea, grid_modelparameters[:,2])
plt.show()


#%%===== PLOTTING ===========================================================================================
##netcdf_output15 = nc.Dataset(input.main_directory + '/../Output/PyGEM_output_rgiregion15_ERAInterim_calSheanMB_nearest_20180306.nc', 'r+')
#netcdf_output15 = nc.Dataset(input.main_directory + '/../Output/PyGEM_output_rgiregion15_ERAInterim_calSheanMB_transferAvg_20180306.nc', 'r+')
##netcdf_output14 = nc.Dataset(input.main_directory + '/../Output/PyGEM_output_rgiregion14_ERAInterim_calSheanMB_nearest_20180313.nc', 'r+')
#netcdf_output14 = nc.Dataset(input.main_directory + '/../Output/PyGEM_output_rgiregion14_ERAInterim_calSheanMB_transferAvg_20180313.nc', 'r+')
#
## Select relevant data
#glacier_data15 = pd.DataFrame(netcdf_output15['glacierparameter'][:])
#glacier_data15.columns = netcdf_output15['glacierparameters'][:]
#lats15 = glacier_data15['lat'].values.astype(float)
#lons15 = glacier_data15['lon'].values.astype(float)
#massbal_total15 = netcdf_output15['massbaltotal_glac_monthly'][:]
#massbal_total_mwea15 = massbal_total15.sum(axis=1)/(massbal_total15.shape[1]/12)
#volume_glac_annual15 = netcdf_output15['volume_glac_annual'][:]
#volume_reg_annual15 = volume_glac_annual15.sum(axis=0)
#volume_reg_annualnorm15 = volume_reg_annual15 / volume_reg_annual15[0]
#runoff_glac_monthly15 = netcdf_output15['runoff_glac_monthly'][:]
#runoff_reg_monthly15 = runoff_glac_monthly15.mean(axis=0)
#acc_glac_monthly15 = netcdf_output15['acc_glac_monthly'][:]
#acc_reg_monthly15 = acc_glac_monthly15.mean(axis=0)
#acc_reg_annual15 = np.sum(acc_reg_monthly15.reshape(-1,12), axis=1)
#refreeze_glac_monthly15 = netcdf_output15['refreeze_glac_monthly'][:]
#refreeze_reg_monthly15 = refreeze_glac_monthly15.mean(axis=0)
#refreeze_reg_annual15 = np.sum(refreeze_reg_monthly15.reshape(-1,12), axis=1)
#melt_glac_monthly15 = netcdf_output15['melt_glac_monthly'][:]
#melt_reg_monthly15 = melt_glac_monthly15.mean(axis=0)
#melt_reg_annual15 = np.sum(melt_reg_monthly15.reshape(-1,12), axis=1)
#massbaltotal_glac_monthly15 = netcdf_output15['massbaltotal_glac_monthly'][:]
#massbaltotal_reg_monthly15 = massbaltotal_glac_monthly15.mean(axis=0)
#massbaltotal_reg_annual15 = np.sum(massbaltotal_reg_monthly15.reshape(-1,12), axis=1)
#glacier_data14 = pd.DataFrame(netcdf_output14['glacierparameter'][:])
#glacier_data14.columns = netcdf_output14['glacierparameters'][:]
#lats14 = glacier_data14['lat'].values.astype(float)
#lons14 = glacier_data14['lon'].values.astype(float)
#massbal_total14 = netcdf_output14['massbaltotal_glac_monthly'][:]
#massbal_total_mwea14 = massbal_total14.sum(axis=1)/(massbal_total14.shape[1]/12)
#volume_glac_annual14 = netcdf_output14['volume_glac_annual'][:]
#volume_reg_annual14 = volume_glac_annual14.sum(axis=0)
#volume_reg_annualnorm14 = volume_reg_annual14 / volume_reg_annual14[0]
#runoff_glac_monthly14 = netcdf_output14['runoff_glac_monthly'][:]
#runoff_reg_monthly14 = runoff_glac_monthly14.mean(axis=0)
#acc_glac_monthly14 = netcdf_output14['acc_glac_monthly'][:]
#acc_reg_monthly14 = acc_glac_monthly14.mean(axis=0)
#acc_reg_annual14 = np.sum(acc_reg_monthly14.reshape(-1,12), axis=1)
#refreeze_glac_monthly14 = netcdf_output14['refreeze_glac_monthly'][:]
#refreeze_reg_monthly14 = refreeze_glac_monthly14.mean(axis=0)
#refreeze_reg_annual14 = np.sum(refreeze_reg_monthly14.reshape(-1,12), axis=1)
#melt_glac_monthly14 = netcdf_output14['melt_glac_monthly'][:]
#melt_reg_monthly14 = melt_glac_monthly14.mean(axis=0)
#melt_reg_annual14 = np.sum(melt_reg_monthly14.reshape(-1,12), axis=1)
#massbaltotal_glac_monthly14 = netcdf_output14['massbaltotal_glac_monthly'][:]
#massbaltotal_reg_monthly14 = massbaltotal_glac_monthly14.mean(axis=0)
#massbaltotal_reg_annual14 = np.sum(massbaltotal_reg_monthly14.reshape(-1,12), axis=1)
#years = np.arange(2000, 2016 + 1)
#month = np.arange(2000, 2016, 1/12)
#plt.plot(years,volume_reg_annualnorm15, label='Region 15')
#plt.plot(years,volume_reg_annualnorm14, label='Region 14')
#plt.ylabel('Volume normalized [-]', size=15)
#plt.legend()
#plt.show()
#plt.plot(month,runoff_reg_monthly15, label='Region 15')
#plt.ylabel('Runoff [m3 / month]', size=15)
#plt.legend()
#plt.show()
##plt.plot(month, massbaltotal_reg_monthly, label='massbal_total')
##plt.plot(month, acc_reg_monthly, label='accumulation')
##plt.plot(month, refreeze_reg_monthly, label='refreeze')
##plt.plot(month, -1*melt_reg_monthly, label='melt')
##plt.ylabel('monthly regional mean [m.w.e.] / month')
##plt.legend()
##plt.show()
#plt.plot(years[0:16], massbaltotal_reg_annual15, label='massbal_total')
#plt.plot(years[0:16], acc_reg_annual15, label='accumulation')
#plt.plot(years[0:16], refreeze_reg_annual15, label='refreeze')
#plt.plot(years[0:16], -1*melt_reg_annual15, label='melt')
#plt.ylabel('Region 15 annual mean [m.w.e.]', size=15)
#plt.legend()
#plt.show()
#
#lons = np.concatenate((lons14, lons15), axis=0)
#lats = np.concatenate((lats14, lats15), axis=0)
#massbal_total_mwea = np.concatenate((massbal_total_mwea14, massbal_total_mwea15), axis=0)
#
## Set extent
#east = int(round(lons.min())) - 1
#west = int(round(lons.max())) + 1
#south = int(round(lats.min())) - 1
#north = int(round(lats.max())) + 1
#xtick = 1
#ytick = 1
## Plot regional maps
#plot_latlonvar(lons, lats, massbal_total_mwea, -1.5, 0.5, 'Modeled mass balance [mwea]', 'longitude [deg]', 
#               'latitude [deg]', 'jet_r', east, west, south, north, xtick, ytick)
#
##%%### ====== PLOTTING FOR CALIBRATION FUNCTION ======================================================================
### Plot histograms and regional variations
#data13 = pd.read_csv(input.main_directory + '/../Output/calibration_R13_20180318_Opt01solutionspaceexpanding.csv')
#data13 = data13.dropna()
##data14 = pd.read_csv(input.main_directory + '/../Output/calibration_R14_20180313_Opt01solutionspaceexpanding.csv')
##data14 = data14.dropna()
##data15 = pd.read_csv(input.main_directory + '/../Output/calibration_R15_20180306_Opt01solutionspaceexpanding.csv')
##data15 = data15.dropna()
#data = data13
#
## Concatenate the data
##frames = [data13, data14, data15]
##data = pd.concat(frames)
#
### Fill in values with average 
### Subset all values that have data
##data_subset = data.dropna()
##data_subset_params = data_subset[['lrgcm','lrglac','precfactor','precgrad','ddfsnow','ddfice','tempsnow','tempchange']]
##data_subset_paramsavg = data_subset_params.mean()
##paramsfilled = data[['lrgcm','lrglac','precfactor','precgrad','ddfsnow','ddfice','tempsnow','tempchange']]
##paramsfilled = paramsfilled.fillna(data_subset_paramsavg)    
#
## Set extent
#east = int(round(data['CenLon'].min())) - 1
#west = int(round(data['CenLon'].max())) + 1
#south = int(round(data['CenLat'].min())) - 1
#north = int(round(data['CenLat'].max())) + 1
#xtick = 1
#ytick = 1
## Select relevant data
#lats = data['CenLat'][:]
#lons = data['CenLon'][:]
#precfactor = data['precfactor'][:]
#tempchange = data['tempchange'][:]
#ddfsnow = data['ddfsnow'][:]
#calround = data['calround'][:]
#massbal = data['MB_geodetic_mwea']
## Plot regional maps
#plot_latlonvar(lons, lats, massbal, -1.5, 0.5, 'Geodetic mass balance [mwea]', 'longitude [deg]', 'latitude [deg]', 
#               'jet_r', east, west, south, north, xtick, ytick)
#plot_latlonvar(lons, lats, precfactor, 0.8, 1.3, 'Precipitation factor [-]', 'longitude [deg]', 'latitude [deg]', 
#               'jet_r', east, west, south, north, xtick, ytick)
#plot_latlonvar(lons, lats, tempchange, -4, 2, 'Temperature bias [degC]', 'longitude [deg]', 'latitude [deg]', 
#               'jet', east, west, south, north, xtick, ytick)
#plot_latlonvar(lons, lats, ddfsnow, 0.003, 0.005, 'DDF_snow [m w.e. d-1 degC-1]', 'longitude [deg]', 'latitude [deg]', 
#               'jet', east, west, south, north, xtick, ytick)
#plot_latlonvar(lons, lats, calround, 1, 3, 'Calibration round', 'longitude [deg]', 'latitude [deg]', 
#               'jet_r', east, west, south, north, xtick, ytick)
## Plot histograms
#data.hist(column='MB_difference_mwea', bins=50)
#plt.title('Mass Balance Difference [mwea]')
#data.hist(column='precfactor', bins=50)
#plt.title('Precipitation factor [-]')
#data.hist(column='tempchange', bins=50)
#plt.title('Temperature bias [degC]')
#data.hist(column='ddfsnow', bins=50)
#plt.title('DDFsnow [mwe d-1 degC-1]')
#plt.xticks(rotation=60)
#data.hist(column='calround', bins = [0.5, 1.5, 2.5, 3.5])
#plt.title('Calibration round')
#plt.xticks([1, 2, 3])
#    
### run plot function
##output.plot_caloutput(data)


#%%===== EXTRAS =============================================================================================
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