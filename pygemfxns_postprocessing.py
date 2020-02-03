# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 09:04:46 2017

@author: David Rounce

pygemfxns_output_postprocessing.py is a mix of post-processing for things like plots, relationships between variables,
and any other comparisons between output or input data.
"""

# Built-in Libraries
import os
import collections
# External Libraries
import numpy as np
import pandas as pd 
import netCDF4 as nc
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import scipy
import cartopy
import xarray as xr
# Local Libraries
import pygem.pygem_input as pygem_prms
import pygemfxns_modelsetup as modelsetup
import pygemfxns_massbalance as massbalance
import class_mbdata
import run_simulation

# Script options
option_plot_futuresim = 0
option_mb_shean_analysis = 0
option_mb_shean_regional = 0
option_geodeticMB_loadcompare = 0
option_check_biasadj = 0
option_parameter_relationships = 0
option_MCMC_ensembles = 0
option_calcompare_w_geomb = 0
option_add_metadata2netcdf = 0
option_var_mon2annual = 0


#%% SUBSET RESULTS INTO EACH VARIABLE NAME SO EASIER TO TRANSFER
if option_var_mon2annual == 1:
    netcdf_fp_prefix = pygem_prms.output_filepath + 'simulations/spc/20181108_vars/'
    vns = ['acc_glac_monthly', 'melt_glac_monthly', 'refreeze_glac_monthly', 'frontalablation_glac_monthly', 
           'massbaltotal_glac_monthly', 'temp_glac_monthly', 'prec_glac_monthly', 'runoff_glac_monthly']
#    vns = ['runoff_glac_monthly']
    
    def coords_attrs_dict(ds, vn):
        """
        Retrieve dictionaries containing coordinates, attributes, and encoding for the dataset and variable name
        
        Parameters
        ----------
        ds : xr.Dataset
            dataset of a variable of interest
        vn : str
            variable name
            
        Returns
        -------
        output_coords_dict : dictionary
            coordiantes for the modified variable
        output_attrs_dict: dictionary
            attributes to add to the modified variable
        encoding : dictionary
            encoding used with exporting xarray dataset to netcdf
        """
        # Variable coordinates dictionary
        output_coords_dict = {
                'temp_glac_annual': collections.OrderedDict(
                        [('glac', ds.glac.values), ('year', ds.year.values), ('stats', ds.stats.values)]),
                'prec_glac_annual': collections.OrderedDict(
                        [('glac', ds.glac.values), ('year', ds.year.values), ('stats', ds.stats.values)]),
                'runoff_glac_annual': collections.OrderedDict(
                        [('glac', ds.glac.values), ('year', ds.year.values), ('stats', ds.stats.values)]),
                'acc_glac_annual': collections.OrderedDict(
                        [('glac', ds.glac.values), ('year', ds.year.values), ('stats', ds.stats.values)]),
                'acc_glac_summer': collections.OrderedDict(
                        [('glac', ds.glac.values), ('year', ds.year.values), ('stats', ds.stats.values)]),
                'acc_glac_winter': collections.OrderedDict(
                        [('glac', ds.glac.values), ('year', ds.year.values), ('stats', ds.stats.values)]),
                'melt_glac_annual': collections.OrderedDict(
                        [('glac', ds.glac.values), ('year', ds.year.values), ('stats', ds.stats.values)]),
                'melt_glac_summer': collections.OrderedDict(
                        [('glac', ds.glac.values), ('year', ds.year.values), ('stats', ds.stats.values)]),
                'melt_glac_winter': collections.OrderedDict(
                        [('glac', ds.glac.values), ('year', ds.year.values), ('stats', ds.stats.values)]),
                'refreeze_glac_annual': collections.OrderedDict(
                        [('glac', ds.glac.values), ('year', ds.year.values), ('stats', ds.stats.values)]),
                'refreeze_glac_summer': collections.OrderedDict(
                        [('glac', ds.glac.values), ('year', ds.year.values), ('stats', ds.stats.values)]),
                'refreeze_glac_winter': collections.OrderedDict(
                        [('glac', ds.glac.values), ('year', ds.year.values), ('stats', ds.stats.values)]),
                'frontalablation_glac_annual': collections.OrderedDict(
                        [('glac', ds.glac.values), ('year', ds.year.values), ('stats', ds.stats.values)]),
                'frontalablation_glac_summer': collections.OrderedDict(
                        [('glac', ds.glac.values), ('year', ds.year.values), ('stats', ds.stats.values)]),
                'frontalablation_glac_winter': collections.OrderedDict(
                        [('glac', ds.glac.values), ('year', ds.year.values), ('stats', ds.stats.values)]),
                'massbaltotal_glac_annual': collections.OrderedDict(
                        [('glac', ds.glac.values), ('year', ds.year.values), ('stats', ds.stats.values)]),
                'massbaltotal_glac_summer': collections.OrderedDict(
                        [('glac', ds.glac.values), ('year', ds.year.values), ('stats', ds.stats.values)]),
                'massbaltotal_glac_winter': collections.OrderedDict(
                        [('glac', ds.glac.values), ('year', ds.year.values), ('stats', ds.stats.values)])     
                }
        # Attributes dictionary
        output_attrs_dict = {
                'temp_glac_annual': {
                        'long_name': 'glacier-wide mean air temperature',
                        'units': 'degC',
                        'temporal_resolution': 'annual',
                        'comment': (
                                'annual mean has each month weight equally, each elevation bin is weighted equally'
                                ' to compute the mean temperature, and bins where the glacier no longer exists due to '
                                'retreat have been removed')},
                'prec_glac_annual': {
                        'long_name': 'glacier-wide precipitation (liquid)',
                        'units': 'm',
                        'temporal_resolution': 'annual',
                        'comment': 'only the liquid precipitation, solid precipitation excluded'},
                'acc_glac_annual': {
                        'long_name': 'glacier-wide accumulation',
                        'units': 'm w.e.',
                        'temporal_resolution': 'annual',
                        'comment': 'only the solid precipitation'},
                'acc_glac_summer': {
                        'long_name': 'glacier-wide accumulation',
                        'units': 'm w.e.',
                        'temporal_resolution': 'annual summer',
                        'comment': 'only the solid precipitation'},
                'acc_glac_winter': {
                        'long_name': 'glacier-wide accumulation',
                        'units': 'm w.e.',
                        'temporal_resolution': 'annual winter',
                        'comment': 'only the solid precipitation'},
                'melt_glac_annual': {
                        'long_name': 'glacier-wide melt',
                        'units': 'm w.e.',
                        'temporal_resolution': 'annual'},
                'melt_glac_summer': {
                        'long_name': 'glacier-wide melt',
                        'units': 'm w.e.',
                        'temporal_resolution': 'annual summer'},
                'melt_glac_winter': {
                        'long_name': 'glacier-wide melt',
                        'units': 'm w.e.',
                        'temporal_resolution': 'annual winter'},
                'refreeze_glac_annual': {
                        'long_name': 'glacier-wide refreeze',
                        'units': 'm w.e.',
                        'temporal_resolution': 'annual'},
                'refreeze_glac_summer': {
                        'long_name': 'glacier-wide refreeze',
                        'units': 'm w.e.',
                        'temporal_resolution': 'annual summer'},
                'refreeze_glac_winter': {
                        'long_name': 'glacier-wide refreeze',
                        'units': 'm w.e.',
                        'temporal_resolution': 'annual winter'},
                'frontalablation_glac_annual': {
                        'long_name': 'glacier-wide frontal ablation',
                        'units': 'm w.e.',
                        'temporal_resolution': 'annual',
                        'comment': (
                                'mass losses from calving, subaerial frontal melting, sublimation above the '
                                'waterline and subaqueous frontal melting below the waterline')},
                'frontalablation_glac_summer': {
                        'long_name': 'glacier-wide frontal ablation',
                        'units': 'm w.e.',
                        'temporal_resolution': 'annual summer',
                        'comment': (
                                'mass losses from calving, subaerial frontal melting, sublimation above the '
                                'waterline and subaqueous frontal melting below the waterline')},
                'frontalablation_glac_winter': {
                        'long_name': 'glacier-wide frontal ablation',
                        'units': 'm w.e.',
                        'temporal_resolution': 'annual winter',
                        'comment': (
                                'mass losses from calving, subaerial frontal melting, sublimation above the '
                                'waterline and subaqueous frontal melting below the waterline')},
                'massbaltotal_glac_annual': {
                        'long_name': 'glacier-wide total mass balance',
                        'units': 'm w.e.',
                        'temporal_resolution': 'annual',
                        'comment': 'total mass balance is the sum of the climatic mass balance and frontal ablation'},
                'massbaltotal_glac_summer': {
                        'long_name': 'glacier-wide total mass balance',
                        'units': 'm w.e.',
                        'temporal_resolution': 'annual summer',
                        'comment': 'total mass balance is the sum of the climatic mass balance and frontal ablation'},
                'massbaltotal_glac_winter': {
                        'long_name': 'glacier-wide total mass balance',
                        'units': 'm w.e.',
                        'temporal_resolution': 'annual winter',
                        'comment': 'total mass balance is the sum of the climatic mass balance and frontal ablation'},
                'runoff_glac_annual': {
                        'long_name': 'glacier-wide runoff',
                        'units': 'm**3',
                        'temporal_resolution': 'annual',
                        'comment': 'runoff from the glacier terminus, which moves over time'},
                }
                
        encoding = {}  
        noencoding_vn = ['stats', 'glac_attrs']
        # Encoding (specify _FillValue, offsets, etc.)
        if vn not in noencoding_vn:
            encoding[vn] = {'_FillValue': False}
        return output_coords_dict, output_attrs_dict, encoding
    
    for vn in vns:
        netcdf_fp = netcdf_fp_prefix + vn + '/'
        for i in os.listdir(netcdf_fp):
            if i.endswith('.nc'):
                print(i)
                               
                # Open dataset and extract annual values
                ds = xr.open_dataset(netcdf_fp + i)      
                ds_mean = ds[vn].values[:,:,0]
                ds_std = ds[vn].values[:,:,1]
                ds_var = ds_std**2
                
                # Compute annual/seasonal mean/sum and standard deviation for the variable of interest
                if vn is 'temp_glac_monthly':
                    output_list = ['annual']
                    vn_annual = 'temp_glac_annual'
                    # Mean annual temperature, standard deviation, and variance
                    ds_mean_annual = ds_mean.reshape(-1,12).mean(axis=1).reshape(-1,int(ds_mean.shape[1]/12))
                    ds_var_annual = ds_var.reshape(-1,12).mean(axis=1).reshape(-1,int(ds_std.shape[1]/12))
                    ds_std_annual = ds_var_annual**0.5
                    ds_values_annual = np.concatenate((ds_mean_annual[:,:,np.newaxis], ds_std_annual[:,:,np.newaxis]), 
                                                      axis=2)
                elif vn in ['prec_glac_monthly', 'runoff_glac_monthly']:
                    output_list = ['annual']
                    vn_annual = 'prec_glac_annual'
                    # Total annual precipitation, standard deviation, and variance
                    ds_sum_annual = ds_mean.reshape(-1,12).sum(axis=1).reshape(-1,int(ds_mean.shape[1]/12))
                    ds_var_annual = ds_var.reshape(-1,12).sum(axis=1).reshape(-1,int(ds_std.shape[1]/12))
                    ds_std_annual = ds_var_annual**0.5
                    ds_values_annual = np.concatenate((ds_sum_annual[:,:,np.newaxis], ds_std_annual[:,:,np.newaxis]), 
                                                      axis=2)
                elif vn in ['acc_glac_monthly', 'melt_glac_monthly', 'refreeze_glac_monthly', 
                            'frontalablation_glac_monthly', 'massbaltotal_glac_monthly']:
                    output_list = ['annual', 'summer', 'winter']
                    # Annual total, standard deviation, and variance
                    ds_sum_annual = ds_mean.reshape(-1,12).sum(axis=1).reshape(-1,int(ds_mean.shape[1]/12))
                    ds_var_annual = ds_var.reshape(-1,12).sum(axis=1).reshape(-1,int(ds_std.shape[1]/12))
                    ds_std_annual = ds_var_annual**0.5
                    ds_values_annual = np.concatenate((ds_sum_annual[:,:,np.newaxis], ds_std_annual[:,:,np.newaxis]), 
                                                      axis=2)
                    # Seasonal total, standard deviation, and variance
                    if ds.time.year_type == 'water year':
                        option_wateryear = 1
                    elif ds.time.year_type == 'calendar year':
                        option_wateryear = 2
                    else:
                        option_wateryear = 3
                    dates_table = modelsetup.datesmodelrun(startyear=ds.year.values[0], endyear=ds.year.values[-1], 
                                                           spinupyears=0, option_wateryear=option_wateryear)
                    # For seasonal calculations copy monthly values and remove the other season's values
                    ds_mean_summer = ds_mean.copy()
                    ds_var_summer = ds_var.copy()                    
                    ds_mean_summer[:,dates_table.season.values == 'winter'] = 0
                    ds_sum_summer = ds_mean_summer.reshape(-1,12).sum(axis=1).reshape(-1, int(ds_mean.shape[1]/12))
                    ds_var_summer = ds_var_summer.reshape(-1,12).sum(axis=1).reshape(-1,int(ds_std.shape[1]/12))
                    ds_std_summer = ds_var_summer**0.5
                    ds_values_summer = np.concatenate((ds_sum_summer[:,:,np.newaxis], ds_std_summer[:,:,np.newaxis]), 
                                                      axis=2)
                    ds_mean_winter = ds_mean.copy()
                    ds_var_winter = ds_var.copy()                    
                    ds_mean_winter[:,dates_table.season.values == 'summer'] = 0
                    ds_sum_winter = ds_mean_winter.reshape(-1,12).sum(axis=1).reshape(-1, int(ds_mean.shape[1]/12))
                    ds_var_winter = ds_var_winter.reshape(-1,12).sum(axis=1).reshape(-1,int(ds_std.shape[1]/12))
                    ds_std_winter = ds_var_winter**0.5
                    ds_values_winter = np.concatenate((ds_sum_winter[:,:,np.newaxis], ds_std_winter[:,:,np.newaxis]), 
                                                      axis=2)
                # Create modified dataset
                for temporal_res in output_list:
                    vn_new = vn.split('_')[0] + '_glac_' + temporal_res
                    output_fp = netcdf_fp_prefix + vn_new + '/'
                    output_fn = i.split('.nc')[0][:-7] + temporal_res + '.nc'                
                    output_coords_dict, output_attrs_dict, encoding = coords_attrs_dict(ds, vn_new)
                    if temporal_res is 'annual':
                        ds_new = xr.Dataset({vn_new: (list(output_coords_dict[vn_new].keys()), ds_values_annual)},
                                             coords=output_coords_dict[vn_new])
                    elif temporal_res is 'summer':
                        ds_new = xr.Dataset({vn_new: (list(output_coords_dict[vn_new].keys()), ds_values_summer)},
                                             coords=output_coords_dict[vn_new])
                    elif temporal_res is 'winter':
                        ds_new = xr.Dataset({vn_new: (list(output_coords_dict[vn_new].keys()), ds_values_winter)},
                                             coords=output_coords_dict[vn_new])
                    ds_new[vn_new].attrs = output_attrs_dict[vn_new]
                    # Merge new dataset into the old to retain glacier table and other attributes
                    output_ds = xr.merge((ds, ds_new))
                    output_ds = output_ds.drop(vn)
                    # Export netcdf
                    if not os.path.exists(output_fp):
                        os.makedirs(output_fp)
                    output_ds.to_netcdf(output_fp + output_fn, encoding=encoding)
                
                # Remove file
                os.remove(netcdf_fp + i)
        


#%%===== PLOT FUNCTIONS =============================================================================================
def plot_latlonvar(lons, lats, variable, rangelow, rangehigh, title, xlabel, ylabel, colormap, east, west, south, north, 
                   xtick=1, 
                   ytick=1, 
                   marker_size=2,
                   option_savefig=0, 
                   fig_fn='Samplefig_fn.png',
                   output_filepath = pygem_prms.main_directory + '/../Output/'):
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
    plt.scatter(lons, lats, s=marker_size, c=variable, cmap='RdBu', marker='o', edgecolor='black', linewidths=0.25)
    #  plotting x, y, size [s=__], color bar [c=__]
    plt.clim(rangelow,rangehigh)
    #  set the range of the color bar
    plt.colorbar(fraction=0.02, pad=0.04)
    #  fraction resizes the colorbar, pad is the space between the plot and colorbar
    if option_savefig == 1:
        plt.savefig(output_filepath + fig_fn)
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


#%% ===== PARAMETER RELATIONSHIPS ======
if option_parameter_relationships == 1:
    # Load csv
    ds = pd.read_csv(pygem_prms.main_directory + '/../Output/20180710_cal_modelparams_opt1_R15_ERA-Interim_1995_2015.csv', 
                     index_col=0)
    property_cn = 'Zmed'
    
    # Relationship between model parameters and glacier properties
    plt.figure(figsize=(6,10))
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.suptitle('Model parameters vs. ' + property_cn, y=0.94)
    # Temperature change
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(ds[property_cn], ds['tempchange'])
    xplot = np.arange(4000,6500)
    line = slope*xplot+intercept
    plt.subplot(4,1,1)
    plt.plot(ds[property_cn], ds['tempchange'], 'o', mfc='none', mec='black')
    plt.plot(xplot, line)
    plt.xlabel(property_cn + ' [masl]', size=10)
    plt.ylabel('tempchange \n[degC]', size=12)
    equation = 'tempchange = ' + str(round(slope,7)) + ' * ' + property_cn + ' + ' + str(round(intercept,5))
    plt.text(0.15, 0.85, equation, fontsize=12, transform=plt.gcf().transFigure, 
             bbox=dict(facecolor='white', edgecolor='none', alpha=0.85))
    print(equation, ' , R2 =', round(r_value**2,2))
    # Precipitation factor
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(ds[property_cn], ds['precfactor'])
    xplot = np.arange(4000,6500)
    line = slope*xplot+intercept
    plt.subplot(4,1,2)
    plt.plot(ds[property_cn], ds['precfactor'], 'o', mfc='none', mec='black')
    plt.plot(xplot, line)
    plt.xlabel(property_cn + ' [masl]', size=12)
    plt.ylabel('precfactor \n[-]', size=12)
    equation = 'precfactor = ' + str(round(slope,7)) + ' * ' + property_cn + ' + ' + str(round(intercept,5)) 
    plt.text(0.15, 0.65, equation, fontsize=12, transform=plt.gcf().transFigure, 
             bbox=dict(facecolor='white', edgecolor='none', alpha=0.85))
    print(equation, ' , R2 =', round(r_value**2,2))
    # Degree day factor of snow    
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(ds[property_cn], ds['ddfsnow'])
    xplot = np.arange(4000,6500)
    line = slope*xplot+intercept
    plt.subplot(4,1,3)
    plt.plot(ds[property_cn], ds['ddfsnow'], 'o', mfc='none', mec='black')
    plt.plot(xplot, line)
    plt.xlabel(property_cn + ' [masl]', size=12)
    plt.ylabel('ddfsnow \n[mwe d-1 degC-1]', size=12)
#    plt.legend()
    equation = 'ddfsnow = ' + str(round(slope,12)) + ' * ' + property_cn + ' + ' + str(round(intercept,5)) 
    plt.text(0.15, 0.45, equation, fontsize=12, transform=plt.gcf().transFigure, 
             bbox=dict(facecolor='white', edgecolor='none', alpha=0.85))
    print(equation, ' , R2 =', round(r_value**2,2))  
    # Precipitation gradient
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(ds[property_cn], ds['precgrad'])
    xplot = np.arange(4000,6500)
    line = slope*xplot+intercept
    plt.subplot(4,1,4)
    plt.plot(ds[property_cn], ds['precgrad'], 'o', mfc='none', mec='black')
    plt.plot(xplot, line)
    plt.xlabel(property_cn + ' [masl]', size=12)
    plt.ylabel('precgrad \n[% m-1]', size=12)
#    plt.legend()
    equation = 'precgrad = ' + str(round(slope,12)) + ' * ' + property_cn + ' + ' + str(round(intercept,5)) 
    plt.text(0.15, 0.25, equation, fontsize=12, transform=plt.gcf().transFigure, 
             bbox=dict(facecolor='white', edgecolor='none', alpha=0.85))
    print(equation, ' , R2 =', round(r_value**2,2))  
    # Plot and save figure
    if option_savefigs == 1:
        plt.savefig(pygem_prms.output_filepath + 'figures/' + 'modelparameters_vs_' + property_cn + '.png', 
                    bbox_inches='tight')
    plt.show()

#%% ===== PLOTTING: Future simulations =====
if option_plot_futuresim == 1:
    output_fp = pygem_prms.output_filepath + 'R15_sims_20180530/'
    gcm_list = ['MPI-ESM-LR', 'GFDL-CM3', 'CanESM2', 'GISS-E2-R']
#    gcm_list = ['NorESM1-M']
#    gcm_list = ['MPI-ESM-LR']
    rcp_scenario = 'rcp26'
    rgi_regionO1 = [15]
    output_all = []
    gcm = gcm_list[0]
    for gcm in gcm_list:
#    for rcp_scenario in ['rcp26', 'rcp85']:
        print(gcm)
        output_fn = 'PyGEM_R' + str(rgi_regionO1[0]) + '_' + gcm + '_' + rcp_scenario + '_biasadj_opt1_1995_2100.nc'
        output = nc.Dataset(output_fp + output_fn)
    
        # Select relevant data
        main_glac_rgi = pd.DataFrame(output['glacier_table'][:], columns=output['glacier_table_header'][:])
        main_glac_rgi['RGIId'] = 'RGI60-' + main_glac_rgi['RGIId_float'].astype(str)
        lats = main_glac_rgi['CenLat']
        lons = main_glac_rgi['CenLon']
        months = nc.num2date(output['time'][:], units=output['time'].units, calendar=output['time'].calendar).tolist()
        years = output['year'][:]
        years_plus1 = output['year_plus1'][:]
        massbal_total = output['massbaltotal_glac_monthly'][:]
        massbal_total_mwea = massbal_total.sum(axis=1)/(massbal_total.shape[1]/12)
        volume_glac_annual = output['volume_glac_annual'][:]
        volume_glac_annual[volume_glac_annual[:,0] == 0] = np.nan
        volume_glac_annualnorm = volume_glac_annual / volume_glac_annual[:,0][:,np.newaxis] * 100
        volchange_glac_perc_15yrs = (volume_glac_annual[:,16] - volume_glac_annual[:,0]) / volume_glac_annual[:,0] * 100 
        volchange_glac_perc_15yrs[np.isnan(volchange_glac_perc_15yrs)==True] = 0
        volume_reg_annual = output['volume_glac_annual'][:].sum(axis=0)
        volume_reg_annualnorm = volume_reg_annual / volume_reg_annual[0] * 100
        slr_reg_annual_mm = ((volume_reg_annual[0] - volume_reg_annual) * pygem_prms.density_ice / pygem_prms.density_water / 
                             pygem_prms.area_ocean * 10**6)
        runoff_glac_monthly = output['runoff_glac_monthly'][:]
        runoff_reg_monthly = runoff_glac_monthly.mean(axis=0)
        acc_glac_monthly = output['acc_glac_monthly'][:]
        acc_reg_monthly = acc_glac_monthly.mean(axis=0)
        acc_reg_annual = np.sum(acc_reg_monthly.reshape(-1,12), axis=1)
        refreeze_glac_monthly = output['refreeze_glac_monthly'][:]
        refreeze_reg_monthly = refreeze_glac_monthly.mean(axis=0)
        refreeze_reg_annual = np.sum(refreeze_reg_monthly.reshape(-1,12), axis=1)
        melt_glac_monthly = output['melt_glac_monthly'][:]
        melt_reg_monthly = melt_glac_monthly.mean(axis=0)
        melt_reg_annual = np.sum(melt_reg_monthly.reshape(-1,12), axis=1)
        massbaltotal_glac_monthly = output['massbaltotal_glac_monthly'][:]
        massbaltotal_reg_monthly = massbaltotal_glac_monthly.mean(axis=0)
        massbaltotal_reg_annual = np.sum(massbaltotal_reg_monthly.reshape(-1,12), axis=1)
        
        # PLOT OF ALL GCMS
        #  use subplots to plot all the GCMs on the same figure
        # Plot: Regional volume change [%]
        plt.subplot(2,1,1)
        plt.plot(years_plus1, volume_reg_annualnorm, label=gcm)
        plt.title('Region ' + str(rgi_regionO1[0]))
        plt.ylabel('Volume [%]')
        plt.xlim(2000,2101)
        plt.legend()
        
        # Plot: Regional sea-level rise [mm]
        plt.subplot(2,1,2)
        plt.plot(years_plus1, slr_reg_annual_mm, label=gcm)
        plt.ylabel('Sea-level rise [mm]')
        plt.xlim(2000,2101)        
    plt.show()

    
    # PLOTS FOR LAST GCM
    # Plot: Regional mass balance [mwe]
    plt.plot(years, massbaltotal_reg_annual, label='massbal_total')
    plt.plot(years, acc_reg_annual, label='accumulation')
    plt.plot(years, refreeze_reg_annual, label='refreeze')
    plt.plot(years, -1*melt_reg_annual, label='melt')
    plt.ylabel('Region 15 annual mean [m.w.e.]')
    plt.title(gcm)
    plt.legend()
    plt.show()  

    # Plot: Regional map of volume change by glacier
    volume_change_glac_perc = output['volume_glac_annual'][:][:,0]
    volume_change_glac_perc[volume_change_glac_perc > 0] = (
            (volume_glac_annual[volume_change_glac_perc > 0,-1] - 
             volume_glac_annual[volume_change_glac_perc > 0, 0]) 
            / volume_glac_annual[volume_change_glac_perc > 0, 0] * 100)
    # Set extent
    east = int(round(lons.min())) - 1
    west = int(round(lons.max())) + 1
    south = int(round(lats.min())) - 1
    north = int(round(lats.max())) + 1
    xtick = 1
    ytick = 1
    # Plot regional maps
    plot_latlonvar(lons, lats, volume_change_glac_perc, -100, 100, gcm + ' Volume [%]', 
                   'longitude [deg]', 'latitude [deg]', 'jet_r', east, west, south, north, xtick, ytick, 
                   marker_size=20)
        
    
#%% ===== MASS BALANCE ANALYSIS =====
if option_mb_shean_analysis == 1:
    # Set parameters within this little batch script
    option_nearestneighbor_export = 0
    
    # Load csv
    ds = pd.read_csv(pygem_prms.main_directory + '/../Output/calibration_R15_20180403_Opt02solutionspaceexpanding.csv', 
                     index_col='GlacNo')
    # Select data of interest
    data_all = ds[['RGIId', 'Area', 'CenLon', 'CenLat', 'mb_mwea', 'mb_mwea_sigma', 'lrgcm', 'lrglac', 'precfactor', 
                   'precgrad', 'ddfsnow', 'ddfice', 'tempsnow', 'tempchange']].copy()
    # Drop nan data to retain only glaciers with calibrated parameters
    data = data_all.dropna()
    
    # Compute statistics
    mb_mean = data['mb_mwea'].mean()
    mb_std = data['mb_mwea'].std()
    mb_95 = [mb_mean - 1.96 * mb_std, mb_mean + 1.96 * mb_std]
    # Remove data outside of 95% confidence bounds
    data_95 = data[(data['mb_mwea'] >= mb_95[0]) & (data['mb_mwea'] <= mb_95[1])]
    mb_1std = [mb_mean - 1 * mb_std, mb_mean + 1 * mb_std]
    # Remove data outside of 95% confidence bounds
    data_1std = data[(data['mb_mwea'] >= mb_1std[0]) & (data['mb_mwea'] <= mb_1std[1])]
    
    # Plot Glacier Area vs. MB
    plt.scatter(data['Area'], data['mb_mwea'], facecolors='none', edgecolors='black', label='Region 15')
    plt.ylabel('MB 2000-2015 [mwea]', size=12)
    plt.xlabel('Glacier area [km2]', size=12)
    plt.legend()
    plt.show()
    # Only 95% confidence
    plt.scatter(data_95['Area'], data_95['mb_mwea'], facecolors='none', edgecolors='black', label='Region 15')
    plt.ylabel('MB 2000-2015 [mwea]', size=12)
    plt.xlabel('Glacier area [km2]', size=12)
    plt.ylim(-3,1.5)
    plt.legend()
    plt.show()
    # Only 1 std
    plt.scatter(data_1std['Area'], data_1std['mb_mwea'], facecolors='none', edgecolors='black', label='Region 15')
    plt.ylabel('MB 2000-2015 [mwea]', size=12)
    plt.xlabel('Glacier area [km2]', size=12)
    plt.ylim(-3,1.5)
    plt.legend()
    plt.show()
    
    # Bar plot
    bins = np.array([0.1, 0.25, 0.5, 1, 2.5, 5, 10, 20, 200])
    hist, bin_edges = np.histogram(data.Area,bins) # make the histogram
    fig, ax = plt.subplots()
    # Plot the histogram heights against integers on the x axis
    ax.bar(range(len(hist)),hist,width=1)
    # Set the tickets to the middle of the bars
    ax.set_xticks([i for i,j in enumerate(hist)])
    # Set the xticklabels to a string taht tells us what the bin edges were
    ax.set_xticklabels(['{} - {}'.format(bins[i],bins[i+1]) for i,j in enumerate(hist)], rotation=45)
    plt.show()
    
    # Compute max/min for the various bins
    mb = data_1std['mb_mwea']
    area = data_1std['Area']
    mb_envelope = np.zeros((bins.shape[0]-1,3))
    for n in range(bins.shape[0] - 1):
        mb_envelope[n,0] = bins[n+1]
        mb_subset = mb[(area > bins[n]) & (area <= bins[n+1])]
        mb_envelope[n,1] = mb_subset.min()
        mb_envelope[n,2] = mb_subset.max()
    
    
    # zoomed in
    plt.scatter(data['Area'], data['mb_mwea'], facecolors='none', edgecolors='black', label='Region 15')
    plt.ylabel('MB 2000-2015 [mwea]', size=12)
    plt.xlabel('Glacier area [km2]', size=12)
    plt.xlim(0.1,2)
    plt.legend()
    plt.show()
    
    # Plot Glacier Area vs. MB
    plt.scatter(data['mb_mwea'], data['Area'], facecolors='none', edgecolors='black', label='Region 15')
    plt.ylabel('Glacier area [km2]', size=12)
    plt.xlabel('MB 2000-2015 [mwea]', size=12)
    plt.legend()
    plt.show()
     # Plot Glacier Area vs. MB
    plt.scatter(data_95['mb_mwea'], data_95['Area'], facecolors='none', edgecolors='black', label='Region 15')
    plt.ylabel('Glacier area [km2]', size=12)
    plt.xlabel('MB 2000-2015 [mwea]', size=12)
    plt.xlim(-3,1.75)
    plt.legend()
    plt.show()
    
    # Histogram of MB data
    plt.hist(data['mb_mwea'], bins=50)
    plt.show()
    
    
    main_glac_rgi = modelsetup.selectglaciersrgitable(rgi_glac_number='all')
    # Select calibration data from geodetic mass balance from David Shean
    main_glac_calmassbal = modelsetup.selectcalibrationdata(main_glac_rgi)
    # Concatenate massbal data to the main glacier
    main_glac_rgi = pd.concat([main_glac_rgi, main_glac_calmassbal], axis=1)
    # Drop those with nan values
    main_glac_calmassbal = main_glac_calmassbal.dropna()
    main_glac_rgi = main_glac_rgi.dropna()
    
    main_glac_rgi[['lrgcm', 'lrglac', 'precfactor', 'precgrad', 'ddfsnow', 'ddfice', 'tempsnow', 'tempchange']] = (
            data[['lrgcm', 'lrglac', 'precfactor', 'precgrad', 'ddfsnow', 'ddfice', 'tempsnow', 'tempchange']])
    # Mass balance versus various parameters
    # Median elevation
    plt.scatter(main_glac_rgi['mb_mwea'], main_glac_rgi['Zmed'], facecolors='none', edgecolors='black', 
                label='Region 15')
    plt.ylabel('Median Elevation [masl]', size=12)
    plt.xlabel('MB 2000-2015 [mwea]', size=12)
    plt.legend()
    plt.show()
    # Elevation range
    main_glac_rgi['elev_range'] = main_glac_rgi['Zmax'] - main_glac_rgi['Zmin']
    plt.scatter(main_glac_rgi['mb_mwea'], main_glac_rgi['elev_range'], facecolors='none', edgecolors='black', 
                label='Region 15')
    plt.ylabel('Elevation range [m]', size=12)
    plt.xlabel('MB 2000-2015 [mwea]', size=12)
    plt.legend()
    plt.show()
    plt.scatter(main_glac_rgi['Area'], main_glac_rgi['elev_range'], facecolors='none', edgecolors='black', 
                label='Region 15')
    plt.ylabel('Elevation range [m]', size=12)
    plt.xlabel('Area [km2]', size=12)
    plt.legend()
    plt.show()
    # Length
    plt.scatter(main_glac_rgi['mb_mwea'], main_glac_rgi['Lmax'], facecolors='none', edgecolors='black', 
                label='Region 15')
    plt.ylabel('Length [m]', size=12)
    plt.xlabel('MB 2000-2015 [mwea]', size=12)
    plt.legend()
    plt.show()
    # Slope
    plt.scatter(main_glac_rgi['mb_mwea'], main_glac_rgi['Slope'], facecolors='none', edgecolors='black', 
                label='Region 15')
    plt.ylabel('Slope [deg]', size=12)
    plt.xlabel('MB 2000-2015 [mwea]', size=12)
    plt.legend()
    plt.show()
    # Aspect
    plt.scatter(main_glac_rgi['mb_mwea'], main_glac_rgi['Aspect'], facecolors='none', edgecolors='black', 
                label='Region 15')
    plt.ylabel('Aspect [deg]', size=12)
    plt.xlabel('MB 2000-2015 [mwea]', size=12)
    plt.legend()
    plt.show()
    plt.scatter(main_glac_rgi['Aspect'], main_glac_rgi['precfactor'], facecolors='none', edgecolors='black', 
                label='Region 15')
    plt.ylabel('precfactor [-]', size=12)
    plt.xlabel('Aspect [deg]', size=12)
    plt.legend()
    plt.show()
    # tempchange
    # Line of best fit
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(main_glac_rgi['mb_mwea'], 
                                                                         main_glac_rgi['tempchange'])
    xplot = np.arange(-3,1.5)
    line = slope*xplot+intercept
    plt.plot(main_glac_rgi['mb_mwea'], main_glac_rgi['tempchange'], 'o', mfc='none', mec='black')
    plt.plot(xplot, line)
    plt.ylabel('tempchange [deg]', size=12)
    plt.xlabel('MB 2000-2015 [mwea]', size=12)
    plt.legend()
    plt.show()
    # precfactor
    # Line of best fit
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(main_glac_rgi['mb_mwea'], 
                                                                         main_glac_rgi['precfactor'])
    xplot = np.arange(-3,1.5)
    line = slope*xplot+intercept
    plt.plot(main_glac_rgi['mb_mwea'], main_glac_rgi['precfactor'], 'o', mfc='none', mec='black')
    plt.plot(xplot, line)
    plt.ylabel('precfactor [-]', size=12)
    plt.xlabel('MB 2000-2015 [mwea]', size=12)
    plt.legend()
    plt.show()


#%% ===== ALL GEODETIC MB DATA LOAD & COMPARE (Shean, Brun, Mauer) =====
if option_geodeticMB_loadcompare == 1:    
    
#    rgi_regionsO1 = [15]
    rgi_regionsO1 = ['13, 14, 15'] # 13, 14, 15 - load data from csv
    rgi_glac_number = 'all'
    
    if rgi_regionsO1[0] == '13, 14, 15':
        # Note: this file was created by manually copying the main_glac_rgi for regions 13, 14, 15 into a csv
        main_glac_rgi = pd.read_csv(pygem_prms.main_directory + 
                                    '/../DEMs/geodetic_glacwide_Shean_Maurer_Brun_HMA_20180807.csv')
    else:
        # Mass balance column name
        massbal_colname = 'mb_mwea'
        # Mass balance uncertainty column name
        massbal_uncertainty_colname = 'mb_mwea_sigma'
        # Mass balance date 1 column name
        massbal_t1 = 't1'
        # Mass balance date 1 column name
        massbal_t2 = 't2'
        # Mass balance tolerance [m w.e.a]
        massbal_tolerance = 0.1
        # Calibration optimization tolerance
        
        main_glac_rgi = modelsetup.selectglaciersrgitable(rgi_regionsO1=rgi_regionsO1, rgi_regionsO2='all', 
                                                          rgi_glac_number=rgi_glac_number)
        # SHEAN DATA
        # Load all data
        ds_all_shean = pd.read_csv(pygem_prms.main_directory + '/../DEMs/Shean_2018_0806/hma_mb_20180803_1229.csv')
        ds_all_shean['RegO1'] = ds_all_shean[pygem_prms.shean_rgi_glacno_cn].values.astype(int)
        ds_all_shean['glacno'] = ((ds_all_shean[pygem_prms.shean_rgi_glacno_cn] % 1) * 10**5).round(0).astype(int)
        ds_all_shean['RGIId'] = ('RGI60-' + ds_all_shean['RegO1'].astype(str) + '.' +
                                 (ds_all_shean['glacno'] / 10**5).apply(lambda x: '%.5f' % x).str.split('.').str[1])
        # Select glaciers included in main_glac_rgi
        ds_shean = (ds_all_shean.iloc[np.where(ds_all_shean['RGIId'].isin(main_glac_rgi['RGIId']) == True)[0],:]).copy()
        ds_shean.sort_values(['glacno'], inplace=True)
        ds_shean.reset_index(drop=True, inplace=True)
        ds_shean['O1Index'] = np.where(main_glac_rgi['RGIId'].isin(ds_shean['RGIId']))[0]
        # Select data for main_glac_rgi
        main_glac_calmassbal_shean = np.zeros((main_glac_rgi.shape[0],4))
        ds_subset_shean = ds_shean[[pygem_prms.rgi_O1Id_colname, massbal_colname, massbal_uncertainty_colname, massbal_t1, 
                                    massbal_t2]].values
        rgi_O1Id = main_glac_rgi[pygem_prms.rgi_O1Id_colname].values
        for glac in range(rgi_O1Id.shape[0]):
            try:
                # Grab the mass balance based on the RGIId Order 1 glacier number
                main_glac_calmassbal_shean[glac,:] = (
                        ds_subset_shean[np.where(np.in1d(ds_subset_shean[:,0],rgi_O1Id[glac])==True)[0][0],1:])
                #  np.in1d searches if there is a match in the first array with the second array provided and returns an
                #   array with same length as first array and True/False values. np.where then used to identify the 
                #   index where there is a match, which is then used to select the massbalance value
                #  Use of numpy arrays for indexing and this matching approach is much faster than looping through; 
                #   however, need the for loop because np.in1d does not order the values that match; hence, need to do 
                #   it 1 at a time
            except:
                # If there is no mass balance data available for the glacier, then set as NaN
                main_glac_calmassbal_shean[glac,:] = np.empty(4)
                main_glac_calmassbal_shean[glac,:] = np.nan
        main_glac_calmassbal_shean = pd.DataFrame(main_glac_calmassbal_shean, 
                                                  columns=[massbal_colname, massbal_uncertainty_colname, massbal_t1, 
                                                           massbal_t2])
        main_glac_rgi['Shean_MB_mwea'] = main_glac_calmassbal_shean[pygem_prms.massbal_colname]
        main_glac_rgi['Shean_MB_mwea_sigma'] = main_glac_calmassbal_shean[pygem_prms.massbal_uncertainty_colname]
        main_glac_rgi['Shean_MB_year1'] = main_glac_calmassbal_shean[massbal_t1]
        main_glac_rgi['Shean_MB_year2'] = main_glac_calmassbal_shean[massbal_t2]
            
        # ===== BRUN DATA =====
        # Load all data
        cal_rgi_colname = 'GLA_ID'
        ds_all_raw_brun = pd.read_csv(pygem_prms.brun_fp + pygem_prms.brun_fn)
        ds_all_brun = ds_all_raw_brun[ds_all_raw_brun['Measured GLA area [percent]'] >= 60].copy()
        ds_all_brun[massbal_t1] = 2000
        ds_all_brun[massbal_t2] = 2016
        ds_all_brun.rename(columns={pygem_prms.brun_mb_cn:massbal_colname}, inplace=True)
        ds_all_brun.rename(columns={pygem_prms.brun_mb_err_cn:massbal_uncertainty_colname}, inplace=True)
        # Subset glaciers based on region
        ds_all_brun['RegO1'] = ds_all_brun[pygem_prms.brun_rgi_glacno_cn].values.astype(int)
        ds_all_brun['glacno'] = ((ds_all_brun[pygem_prms.brun_rgi_glacno_cn] % 1) * 10**5).round(0).astype(int)
        ds_all_brun['RGIId'] = ('RGI60-' + ds_all_brun['RegO1'].astype(str) + '.' +
                               (ds_all_brun['glacno'] / 10**5).apply(lambda x: '%.5f' % x).str.split('.').str[1])
        # Select glaciers included in main_glac_rgi
        ds_brun = (ds_all_brun.iloc[np.where(ds_all_brun['RGIId'].isin(main_glac_rgi['RGIId']) == True)[0],:]).copy()
        ds_brun.sort_values(['glacno'], inplace=True)
        ds_brun.reset_index(drop=True, inplace=True)
        ds_brun['O1Index'] = np.where(main_glac_rgi['RGIId'].isin(ds_brun['RGIId']))[0]
        # Select data for main_glac_rgi
        main_glac_calmassbal_brun = np.zeros((main_glac_rgi.shape[0], 6))
        ds_subset_brun = ds_brun[[pygem_prms.rgi_O1Id_colname, massbal_colname, massbal_uncertainty_colname, massbal_t1, 
                                  massbal_t2, 'Tot_GLA_area [km2]', 'Measured GLA area [percent]']].values
        for glac in range(rgi_O1Id.shape[0]):
            try:
                # Grab the mass balance based on the RGIId Order 1 glacier number
                main_glac_calmassbal_brun[glac,:] = (
                        ds_subset_brun[np.where(np.in1d(ds_subset_brun[:,0],rgi_O1Id[glac])==True)[0][0],1:])
            except:
                # If there is no mass balance data available for the glacier, then set as NaN
                main_glac_calmassbal_brun[glac,:] = np.empty(main_glac_calmassbal_brun.shape[1])
                main_glac_calmassbal_brun[glac,:] = np.nan
        main_glac_calmassbal_brun = pd.DataFrame(main_glac_calmassbal_brun, 
                                                  columns=[massbal_colname, massbal_uncertainty_colname, massbal_t1, 
                                                           massbal_t2, 'Tot_GLA_area [km2]', 
                                                           'Measured GLA area [percent]'])
        main_glac_rgi['Brun_MB_mwea'] = main_glac_calmassbal_brun[massbal_colname]
        main_glac_rgi['Brun_MB_err_mwea'] = main_glac_calmassbal_brun[massbal_uncertainty_colname]
        main_glac_rgi['Brun_Tot_GLA_area[km2]'] = main_glac_calmassbal_brun['Tot_GLA_area [km2]']
        main_glac_rgi['Brun_GLA_area_measured[%]'] = main_glac_calmassbal_brun['Measured GLA area [percent]']
        ds_brun['GLA_ID'] = ds_brun['GLA_ID'].astype(str)
    
        
        # ===== MAUER DATA =====
        # Load all data
        cal_rgi_colname = 'id'
        ds_all_raw_mauer = pd.read_csv(pygem_prms.mauer_fp + pygem_prms.mauer_fn)
        ds_all_mauer = ds_all_raw_mauer[ds_all_raw_mauer['percentCov'] >= 60].copy()
        ds_all_mauer.rename(columns={pygem_prms.mauer_mb_cn:massbal_colname}, inplace=True)
        ds_all_mauer.rename(columns={pygem_prms.mauer_mb_err_cn:massbal_uncertainty_colname}, inplace=True)
        ds_all_mauer.rename(columns={pygem_prms.mauer_time1_cn:massbal_t1}, inplace=True)
        ds_all_mauer.rename(columns={pygem_prms.mauer_time2_cn:massbal_t2}, inplace=True)
        # Subset glaciers based on region
        ds_all_mauer['RegO1'] = ds_all_mauer[pygem_prms.mauer_rgi_glacno_cn].values.astype(int)
        ds_all_mauer['glacno'] = ((ds_all_mauer[pygem_prms.mauer_rgi_glacno_cn] % 1) * 10**5).round(0).astype(int)
        ds_all_mauer['RGIId'] = ('RGI60-' + ds_all_mauer['RegO1'].astype(str) + '.' +
                                (ds_all_mauer['glacno'] / 10**5).apply(lambda x: '%.5f' % x).str.split('.').str[1])
        # Select glaciers included in main_glac_rgi
        ds_mauer = (ds_all_mauer.iloc[np.where(ds_all_mauer['RGIId'].isin(main_glac_rgi['RGIId']) == True)[0],:]).copy()
        ds_mauer.sort_values(['glacno'], inplace=True)
        ds_mauer.reset_index(drop=True, inplace=True)
        ds_mauer['O1Index'] = np.where(main_glac_rgi['RGIId'].isin(ds_mauer['RGIId']))[0]
        
        main_glac_calmassbal_mauer = np.zeros((main_glac_rgi.shape[0], 5))
        ds_subset_mauer = ds_mauer[[pygem_prms.rgi_O1Id_colname, massbal_colname, massbal_uncertainty_colname, massbal_t1, 
                                    massbal_t2, 'percentCov']].values
        
        for glac in range(rgi_O1Id.shape[0]):
            try:
                # Grab the mass balance based on the RGIId Order 1 glacier number
                main_glac_calmassbal_mauer[glac,:] = (
                        ds_subset_mauer[np.where(np.in1d(ds_subset_mauer[:,0],rgi_O1Id[glac])==True)[0][0],1:])
            except:
                # If there is no mass balance data available for the glacier, then set as NaN
                main_glac_calmassbal_mauer[glac,:] = np.empty(main_glac_calmassbal_mauer.shape[1])
                main_glac_calmassbal_mauer[glac,:] = np.nan
        main_glac_calmassbal_mauer = pd.DataFrame(main_glac_calmassbal_mauer, 
                                                  columns=[massbal_colname, massbal_uncertainty_colname, massbal_t1, 
                                                           massbal_t2, 'percentCov'])
        main_glac_rgi['Mauer_MB_mwea'] = main_glac_calmassbal_mauer[massbal_colname]
        main_glac_rgi['Mauer_MB_mwea_sigma'] = main_glac_calmassbal_mauer[massbal_uncertainty_colname]
        main_glac_rgi['Mauer_MB_year1'] = main_glac_calmassbal_mauer[massbal_t1]
        main_glac_rgi['Mauer_MB_year2'] = main_glac_calmassbal_mauer[massbal_t2]
        main_glac_rgi['Mauer_GLA_area_measured[%]'] = main_glac_calmassbal_mauer['percentCov']
        ds_mauer['id'] = ds_mauer['id'].astype(str)
        
    # Differences
    main_glac_rgi['Dif_Shean-Mauer[mwea]'] = main_glac_rgi['Shean_MB_mwea'] - main_glac_rgi['Mauer_MB_mwea']
    main_glac_rgi['Dif_Shean-Brun[mwea]'] = main_glac_rgi['Shean_MB_mwea'] - main_glac_rgi['Brun_MB_mwea']
    main_glac_rgi['Dif_Mauer-Brun[mwea]'] = main_glac_rgi['Mauer_MB_mwea'] - main_glac_rgi['Brun_MB_mwea']
    
    # Statistics
    print('Glacier area [total]:', round(main_glac_rgi.Area.sum(),1),'km2')
    print('Glacier count [total]:',main_glac_rgi.shape[0],'\n')
    print('Glacier area [Shean]:', 
          round(main_glac_rgi[np.isfinite(main_glac_rgi['Shean_MB_mwea'])].Area.sum(),1),
          'km2 (',
          round(main_glac_rgi[np.isfinite(main_glac_rgi['Shean_MB_mwea'])].Area.sum()/main_glac_rgi.Area.sum()*100,1), 
          '%)')    
    print('Glacier area [Brun]:', 
          round(main_glac_rgi[np.isfinite(main_glac_rgi['Brun_MB_mwea'])].Area.sum(),1),
          'km2 (',
          round(main_glac_rgi[np.isfinite(main_glac_rgi['Brun_MB_mwea'])].Area.sum()/main_glac_rgi.Area.sum()*100,1), 
          '%)')    
    print('Glacier area [Mauer]:', 
          round(main_glac_rgi[np.isfinite(main_glac_rgi['Mauer_MB_mwea'])].Area.sum(),1),
          'km2 (',
          round(main_glac_rgi[np.isfinite(main_glac_rgi['Mauer_MB_mwea'])].Area.sum()/main_glac_rgi.Area.sum()*100,1), 
          '%)','\n')    
    print('Glacier count [Shean]:',main_glac_rgi['Shean_MB_mwea'].dropna().shape[0],
          '(', round(main_glac_rgi['Shean_MB_mwea'].dropna().shape[0]/main_glac_rgi.shape[0]*100,1),'% )')
    print('Glacier count [Brun]:',main_glac_rgi['Brun_MB_mwea'].dropna().shape[0],
          '(', round(main_glac_rgi['Brun_MB_mwea'].dropna().shape[0]/main_glac_rgi.shape[0]*100,1),'% )')
    print('Glacier count [Mauer]:',main_glac_rgi['Mauer_MB_mwea'].dropna().shape[0],
          '(', round(main_glac_rgi['Mauer_MB_mwea'].dropna().shape[0]/main_glac_rgi.shape[0]*100,1),'% )','\n')   
    print('Comparison:')
    print('# same glaciers (Shean/Mauer):',main_glac_rgi['Dif_Shean-Mauer[mwea]'].copy().dropna().shape[0])
    print('# same glaciers (Shean/Brun)',main_glac_rgi['Dif_Shean-Brun[mwea]'].copy().dropna().shape[0])
    print('# same glaciers (Mauer/Brun)',main_glac_rgi['Dif_Mauer-Brun[mwea]'].copy().dropna().shape[0], '\n')
    print('Mean difference (Shean/Mauer):', main_glac_rgi['Dif_Shean-Mauer[mwea]'].mean())
    print('Std difference (Shean/Mauer)):', main_glac_rgi['Dif_Shean-Mauer[mwea]'].std())
    print('Min difference (Shean/Mauer):', main_glac_rgi['Dif_Shean-Mauer[mwea]'].min())
    print('Max difference (Shean/Mauer):', main_glac_rgi['Dif_Shean-Mauer[mwea]'].max(), '\n')
    print('Mean difference (Shean/Brun):', main_glac_rgi['Dif_Shean-Brun[mwea]'].mean())
    print('Std difference (Shean/Brun):', main_glac_rgi['Dif_Shean-Brun[mwea]'].std())
    print('Min difference (Shean/Brun):', main_glac_rgi['Dif_Shean-Brun[mwea]'].min())
    print('Max difference (Shean/Brun):', main_glac_rgi['Dif_Shean-Brun[mwea]'].max(), '\n')
    print('Mean difference (Mauer/Brun):', main_glac_rgi['Dif_Mauer-Brun[mwea]'].mean())
    print('Std difference (Mauer/Brun):', main_glac_rgi['Dif_Mauer-Brun[mwea]'].std())
    print('Min difference (Mauer/Brun):', main_glac_rgi['Dif_Mauer-Brun[mwea]'].min())
    print('Max difference (Mauer/Brun):', main_glac_rgi['Dif_Mauer-Brun[mwea]'].max())
    # Plot histograms of the differences
#    plt.hist(main_glac_rgi['Dif_Shean-Mauer[mwea]'].copy().dropna().values, label='Reg '+str(rgi_regionsO1[0]))
#    plt.xlabel('MB Shean - Mauer [mwea]', size=12)
#    plt.legend()
#    plt.show()
#    plt.hist(main_glac_rgi['Dif_Shean-Brun[mwea]'].copy().dropna().values, label='Reg '+str(rgi_regionsO1[0]))
#    plt.xlabel('MB Shean - Brun [mwea]', size=12)
#    plt.legend()
#    plt.show()
#    plt.hist(main_glac_rgi['Dif_Mauer-Brun[mwea]'].copy().dropna().values, label='Reg '+str(rgi_regionsO1[0]))
#    plt.xlabel('MB Mauer - Brun [mwea]', size=12)
#    plt.legend()
#    plt.show()
    # Plot differences vs. percent area
    #  Fairly consistent; only two 'outliers' 
    # Shean - Brun
    compare_shean_brun = (
            main_glac_rgi[['RGIId', 'Area', 'Dif_Shean-Brun[mwea]','Brun_GLA_area_measured[%]']].copy().dropna())
    compare_shean_mauer = (
            main_glac_rgi[['RGIId', 'Area', 'Dif_Shean-Mauer[mwea]','Mauer_GLA_area_measured[%]']].copy().dropna())
    compare_mauer_brun = (
            main_glac_rgi[['RGIId', 'Area', 'Dif_Mauer-Brun[mwea]','Mauer_GLA_area_measured[%]', 
                           'Brun_GLA_area_measured[%]']].copy().dropna())
#    plt.scatter(compare_shean_brun['Brun_GLA_area_measured[%]'].values, 
#                compare_shean_brun['Dif_Shean-Brun[mwea]'].values, facecolors='none', edgecolors='black', 
#                label='Reg '+str(rgi_regionsO1[0]))
#    plt.xlabel('Brun % Glacier area measured', size=12)
#    plt.ylabel('MB Shean - Brun [mwea]', size=12)
#    plt.legend()
#    plt.show()
    plt.scatter(compare_shean_brun['Area'].values, compare_shean_brun['Dif_Shean-Brun[mwea]'].values, facecolors='none', 
                edgecolors='black', label='Reg '+str(rgi_regionsO1[0]))
    plt.xlabel('Glacier area [km2]', size=12)
    plt.ylabel('MB Shean - Brun [mwea]', size=12)
    plt.legend()
    plt.show()
    # Shean - Mauer
#    plt.scatter(compare_shean_mauer['Mauer_GLA_area_measured[%]'].values, 
#                compare_shean_mauer['Dif_Shean-Mauer[mwea]'].values, facecolors='none', edgecolors='black', 
#                label='Reg '+str(rgi_regionsO1[0]))
#    plt.xlabel('Mauer % Glacier area measured', size=12)
#    plt.ylabel('MB Shean - Mauer [mwea]', size=12)
#    plt.legend()
#    plt.show()
    plt.scatter(compare_shean_mauer['Area'].values, compare_shean_mauer['Dif_Shean-Mauer[mwea]'].values, 
                facecolors='none', edgecolors='black', label='Reg '+str(rgi_regionsO1[0]))
    plt.xlabel('Glacier area [km2]', size=12)
    plt.ylabel('MB Shean - Mauer [mwea]', size=12)
    plt.legend()
    plt.show()
    # Mauer - Brun
    plt.scatter(compare_mauer_brun['Area'].values, compare_mauer_brun['Dif_Mauer-Brun[mwea]'].values, 
                facecolors='none', edgecolors='black', label='Reg '+str(rgi_regionsO1[0]))
    plt.xlabel('Glacier area [km2]', size=12)
    plt.ylabel('MB Mauer - Brun [mwea]', size=12)
    plt.legend()
    plt.show()
    
    # Record statistics concerning number and area covered per region
    main_glac_summary_colnames = ['reg count', 'count', '% reg count', 'reg area', 'area', '% total area']
    main_glac_summary_idxnames = ['shean', 'mauer', 'brun', 'all']
    main_glac_summary = pd.DataFrame(np.zeros((len(main_glac_summary_idxnames),len(main_glac_summary_colnames))), 
                                     index = main_glac_summary_idxnames, columns=main_glac_summary_colnames)
    main_glac_summary['reg count'] = main_glac_rgi.shape[0]
    main_glac_summary['reg area'] = main_glac_rgi['Area'].sum()
    main_glac_summary.loc['shean', 'count'] = main_glac_rgi['Shean_MB_mwea'].dropna().shape[0]
    main_glac_summary.loc['shean','area'] = (
            main_glac_rgi['Area'].where(pd.isnull(main_glac_rgi['Shean_MB_mwea']) == False).dropna().sum())
    main_glac_summary.loc['mauer', 'count'] = main_glac_rgi['Mauer_MB_mwea'].dropna().shape[0]
    main_glac_summary.loc['mauer','area'] = (
            main_glac_rgi['Area'].where(pd.isnull(main_glac_rgi['Mauer_MB_mwea']) == False).dropna().sum())
    main_glac_summary.loc['brun', 'count'] = main_glac_rgi['Brun_MB_mwea'].dropna().shape[0]
    main_glac_summary.loc['brun','area'] = (
            main_glac_rgi['Area'].where(pd.isnull(main_glac_rgi['Brun_MB_mwea']) == False).dropna().sum())
    main_glac_summary.loc['all', 'count'] = (
            main_glac_rgi['Area'][((pd.isnull(main_glac_rgi['Shean_MB_mwea']) == False) | 
                                   (pd.isnull(main_glac_rgi['Mauer_MB_mwea']) == False) | 
                                   (pd.isnull(main_glac_rgi['Brun_MB_mwea']) == False))].shape[0])
    main_glac_summary.loc['all', 'area'] = (
            main_glac_rgi['Area'][((pd.isnull(main_glac_rgi['Shean_MB_mwea']) == False) | 
                                   (pd.isnull(main_glac_rgi['Mauer_MB_mwea']) == False) | 
                                   (pd.isnull(main_glac_rgi['Brun_MB_mwea']) == False))].sum())
    main_glac_summary['% reg count'] = main_glac_summary['count'] / main_glac_summary['reg count'] * 100
    main_glac_summary['% total area'] = main_glac_summary['area'] / main_glac_summary['reg area'] * 100
  
#    # Percent coverage if exclude glaciers < 1 km2
#    A = main_glac_rgi[np.isfinite(main_glac_rgi['Shean_MB_mwea'])]
#    print(round(A[A['Area'] > 1].Area.sum() / main_glac_rgi.Area.sum() * 100,1))


#%% ====== PLOTTING FOR CALIBRATION FUNCTION ======================================================================
if option_calcompare_w_geomb == 1:
    # Plot histograms and regional variations
    rgi_regionsO1 = [15]
    csv_path = '../DEMs/Shean_2018_0806/hma_mb_20180803_1229_all_filled.csv'
    modelparams_fp_dict = {
                13: pygem_prms.output_filepath + 'cal_opt2_20181018/reg13/',
                14: pygem_prms.output_filepath + 'cal_opt2_20181018/reg14/',
                15: pygem_prms.output_filepath + 'cal_opt2_20181018/reg15/'}
    sims_fp_dict = {
                13: pygem_prms.output_filepath + 'simulations/ERA-Interim_2000_2018_nobiasadj/reg13/stats/',
                14: pygem_prms.output_filepath + 'simulations/ERA-Interim_2000_2018_nobiasadj/reg14/stats/',
                15: pygem_prms.output_filepath + 'simulations/ERA-Interim_2000_2018_nobiasadj/reg15/stats/'}
    
    cal_data_all = pd.read_csv(csv_path)
    
    main_glac_rgi = modelsetup.selectglaciersrgitable(rgi_regionsO1=rgi_regionsO1, rgi_regionsO2 = 'all',
                                                      rgi_glac_number='all')
    
    cal_data_all['RegO1'] = cal_data_all['RGIId'].values.astype(int)
    # Select data for specific region
    cal_data_reg = cal_data_all[cal_data_all['RegO1']==rgi_regionsO1[0]].copy()
    cal_data_reg.reset_index(drop=True, inplace=True)
    # Glacier number and index for comparison
    cal_data_reg['glacno'] = ((cal_data_reg['RGIId'] % 1) * 10**5).round(0).astype(int)
    cal_data_reg['RGIId'] = ('RGI60-' + str(rgi_regionsO1[0]) + '.' + 
                       (cal_data_reg['glacno'] / 10**5).apply(lambda x: '%.5f' % x).astype(str).str.split('.').str[1])
    # Select glaciers with mass balance data
    cal_data = (cal_data_reg.iloc[np.where(cal_data_reg['glacno'].isin(main_glac_rgi['glacno']) == True)[0],:]).copy()
    cal_data.reset_index(drop=True, inplace=True)
    
    # Compare observations, calibration, and simulations
    cal_data['calibrated_mb'] = np.nan
    for glac in range(main_glac_rgi.shape[0]):
        glac_str = main_glac_rgi.loc[glac,'RGIId'].split('-')[1]
        # Add calibrated mass balance
        netcdf_fn_cal = glac_str + '.nc'
        ds_cal = xr.open_dataset(modelparams_fp_dict[rgi_regionsO1[0]] + netcdf_fn_cal)
        df_cal = pd.DataFrame(ds_cal['mp_value'].sel(chain=0).values, columns=ds_cal.mp.values)
        cal_mb = df_cal.massbal.values.mean()
        cal_data.loc[glac,'cal_mb'] = cal_mb
        # Add simulated mass balance (will be more off because has mass loss/area changes feedback)
        netcdf_fn_sim = 'ERA-Interim_c2_ba0_200sets_2000_2018--' + glac_str + '_stats.nc'
        ds_sim = xr.open_dataset(sims_fp_dict[rgi_regionsO1[0]] + netcdf_fn_sim)
        df_sim = pd.DataFrame(ds_cal['mp_value'].sel(chain=0).values, columns=ds_cal.mp.values)
        sim_mb = df_cal.massbal.values.mean()
        cal_data.loc[glac,'sim_mb'] = sim_mb
    
    cal_data['cal_mb_dif'] = cal_data.cal_mb - cal_data.mb_mwea
    cal_data['sim_mb_dif'] = cal_data.sim_mb - cal_data.mb_mwea
    cal_data.hist(column='cal_mb_dif', bins=50)
    cal_data.hist(column='sim_mb_dif', bins=50)