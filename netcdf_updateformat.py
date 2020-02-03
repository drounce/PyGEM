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
from collections import OrderedDict
# External Libraries
import numpy as np
import pandas as pd 
import netCDF4 as nc
import scipy
import cartopy
import xarray as xr
# Local Libraries
import pygem_input as input


netcdf_fn = ('/Users/davidrounce/Documents/Dave_Rounce/HiMAT/Output/simulations/ERA-Interim/' + 
             'R15--all--ERA-Interim_c2_ba1_100sets_2000_2015.nc')
output_fn = netcdf_fn.replace('.nc','-v2.nc')
ds = xr.open_dataset(netcdf_fn)
df = pd.DataFrame(ds.glacier_table.values, columns=ds.glac_attrs)


#%%
#def create_xrdataset(ds):
#    """
#    Create empty xarray dataset that will be used to record simulation runs.
#    
#    Parameters
#    ----------
#    ds : pandas dataframe
#        dataframe containing relevant rgi glacier information
#    dates_table : pandas dataframe
#        table of the dates, months, days in month, etc.
#    sim_iters : int
#        number of simulation runs included
#    stat_cns : list
#        list of strings containing statistics that will be used on simulations
#    record_stats : int
#        Switch to change from recording simulations to statistics
#        
#    Returns
#    -------
#    output_ds_all : xarray Dataset
#        empty xarray dataset that contains variables and attributes to be filled in by simulation runs
#    encoding : dictionary
#        encoding used with exporting xarray dataset to netcdf
#    """    
#    if input.output_package == 2:
        
# Create empty datasets for each variable and merge them
# Coordinate values
output_variables = input.output_variables_package2
glac_values = ds.glac.values
time_values = ds.time.values
year_values = ds.year.values
year_plus1_values = ds.year_plus1.values
year_type = ds.time.year_type
    
## Switch to record simulations or statistics
#if record_stats == 0:
#    record_name = 'sim'
#    record_name_values = np.arange(0,sim_iters)
#elif record_stats == 1:
#    record_name = 'stats'
#    record_name_values = input.sim_stat_cns

# Variable coordinates dictionary
output_coords_dict = collections.OrderedDict()
output_coords_dict['RGIId'] =  collections.OrderedDict([('glac', glac_values)])
output_coords_dict['CenLon'] = collections.OrderedDict([('glac', glac_values)])
output_coords_dict['CenLat'] = collections.OrderedDict([('glac', glac_values)])
output_coords_dict['O1Region'] = collections.OrderedDict([('glac', glac_values)])
output_coords_dict['O2Region'] = collections.OrderedDict([('glac', glac_values)])
output_coords_dict['Area'] = collections.OrderedDict([('glac', glac_values)])
output_coords_dict['glac_prec_monthly'] = collections.OrderedDict([('glac', glac_values), ('time', time_values)])
output_coords_dict['glac_temp_monthly'] = collections.OrderedDict([('glac', glac_values), ('time', time_values)])
output_coords_dict['glac_acc_monthly'] = collections.OrderedDict([('glac', glac_values), ('time', time_values)])
output_coords_dict['glac_refreeze_monthly'] = collections.OrderedDict([('glac', glac_values), ('time', time_values)])
output_coords_dict['glac_melt_monthly'] = collections.OrderedDict([('glac', glac_values), ('time', time_values)])
output_coords_dict['glac_frontalablation_monthly'] = collections.OrderedDict([('glac', glac_values), ('time', time_values)])
output_coords_dict['glac_massbaltotal_monthly'] = collections.OrderedDict([('glac', glac_values), ('time', time_values)])
output_coords_dict['glac_runoff_monthly'] = collections.OrderedDict([('glac', glac_values), ('time', time_values)]) 
output_coords_dict['glac_snowline_monthly'] = collections.OrderedDict([('glac', glac_values), ('time', time_values)])
output_coords_dict['glac_area_annual'] = collections.OrderedDict([('glac', glac_values), ('year_plus1', year_plus1_values)])
output_coords_dict['glac_volume_annual'] = collections.OrderedDict([('glac', glac_values), ('year_plus1', year_plus1_values)])
output_coords_dict['glac_ELA_annual'] = collections.OrderedDict([('glac', glac_values), ('year', year_values)])
output_coords_dict['offglac_prec_monthly'] = collections.OrderedDict([('glac', glac_values), ('time', time_values)])
output_coords_dict['offglac_refreeze_monthly'] = collections.OrderedDict([('glac', glac_values), ('time', time_values)])
output_coords_dict['offglac_melt_monthly'] = collections.OrderedDict([('glac', glac_values), ('time', time_values)])
output_coords_dict['offglac_snowpack_monthly'] = collections.OrderedDict([('glac', glac_values), ('time', time_values)])
output_coords_dict['offglac_runoff_monthly'] = collections.OrderedDict([('glac', glac_values), ('time', time_values)])
output_coords_dict['glac_prec_monthly_std'] = collections.OrderedDict([('glac', glac_values), ('time', time_values)])
output_coords_dict['glac_temp_monthly_std'] = collections.OrderedDict([('glac', glac_values), ('time', time_values)])
output_coords_dict['glac_acc_monthly_std'] = collections.OrderedDict([('glac', glac_values), ('time', time_values)])
output_coords_dict['glac_refreeze_monthly_std'] = collections.OrderedDict([('glac', glac_values), ('time', time_values)])
output_coords_dict['glac_melt_monthly_std'] = collections.OrderedDict([('glac', glac_values), ('time', time_values)])
output_coords_dict['glac_frontalablation_monthly_std'] = collections.OrderedDict([('glac', glac_values), ('time', time_values)])
output_coords_dict['glac_massbaltotal_monthly_std'] = collections.OrderedDict([('glac', glac_values), ('time', time_values)])
output_coords_dict['glac_runoff_monthly_std'] = collections.OrderedDict([('glac', glac_values), ('time', time_values)])
output_coords_dict['glac_snowline_monthly_std'] = collections.OrderedDict([('glac', glac_values), ('time', time_values)])
output_coords_dict['glac_area_annual_std'] = collections.OrderedDict([('glac', glac_values), ('year_plus1', year_plus1_values)])
output_coords_dict['glac_volume_annual_std'] = collections.OrderedDict([('glac', glac_values), ('year_plus1', year_plus1_values)])
output_coords_dict['glac_ELA_annual_std'] = collections.OrderedDict([('glac', glac_values), ('year', year_values)])
output_coords_dict['offglac_prec_monthly_std'] = collections.OrderedDict([('glac', glac_values), ('time', time_values)])
output_coords_dict['offglac_refreeze_monthly_std'] = collections.OrderedDict([('glac', glac_values), ('time', time_values)])
output_coords_dict['offglac_melt_monthly_std'] = collections.OrderedDict([('glac', glac_values), ('time', time_values)])
output_coords_dict['offglac_snowpack_monthly_std'] = collections.OrderedDict([('glac', glac_values), ('time', time_values)])
output_coords_dict['offglac_runoff_monthly_std'] = collections.OrderedDict([('glac', glac_values), ('time', time_values)])

# Attributes dictionary
output_attrs_dict = {
        'time': {
                'long_name': 'time',
                 'year_type':year_type,
                 'comment':'start of the month'},
        'glac': {
                'long_name': 'glacier index',
                 'comment': 'glacier index referring to glaciers properties and model results'},
        'year': {
                'long_name': 'years',
                 'year_type': year_type,
                 'comment': 'years referring to the start of each year'},
        'year_plus1': {
                'long_name': 'years plus one additional year',
                'year_type': year_type,
                'comment': 'additional year allows one to record glacier dimension changes at end of model run'},
        'RGIId': {
                'long_name': 'Randolph Glacier Inventory ID',
                'comment': 'RGIv6.0'},
        'CenLon': {
                'long_name': 'center longitude',
                'units': 'degrees E',
                'comment': 'value from RGIv6.0'},
        'CenLat': {
                'long_name': 'center latitude',
                'units': 'degrees N',
                'comment': 'value from RGIv6.0'},
        'O1Region': {
                'long_name': 'RGI order 1 region',
                'comment': 'value from RGIv6.0'},
        'O2Region': {
                'long_name': 'RGI order 2 region',
                'comment': 'value from RGIv6.0'},
        'Area': {
                'long_name': 'glacier area',
                'units': 'km2',
                'comment': 'value from RGIv6.0'},
        'glac_temp_monthly': {
                'standard_name': 'air_temperature',
                'long_name': 'glacier-wide mean air temperature',
                'units': 'K',
                'temporal_resolution': 'monthly',
                'comment': ('each elevation bin is weighted equally to compute the mean temperature, and '
                            'bins where the glacier no longer exists due to retreat have been removed')},
        'glac_prec_monthly': {
                'long_name': 'glacier-wide precipitation (liquid)',
                'units': 'm',
                'temporal_resolution': 'monthly',
                'comment': 'only the liquid precipitation, solid precipitation excluded'},
        'glac_acc_monthly': {
                'long_name': 'glacier-wide accumulation, in water equivalent',
                'units': 'm',
                'temporal_resolution': 'monthly',
                'comment': 'only the solid precipitation'},
        'glac_refreeze_monthly': {
                'long_name': 'glacier-wide refreeze, in water equivalent',
                'units': 'm',
                'temporal_resolution': 'monthly'},
        'glac_melt_monthly': {
                'long_name': 'glacier-wide melt, in water equivalent',
                'units': 'm',
                'temporal_resolution': 'monthly'},
        'glac_frontalablation_monthly': {
                'long_name': 'glacier-wide frontal ablation, in water equivalent',
                'units': 'm',
                'temporal_resolution': 'monthly',
                'comment': (
                        'mass losses from calving, subaerial frontal melting, sublimation above the '
                        'waterline and subaqueous frontal melting below the waterline')},
        'glac_massbaltotal_monthly': {
                'long_name': 'glacier-wide total mass balance, in water equivalent',
                'units': 'm',
                'temporal_resolution': 'monthly',
                'comment': 'total mass balance is the sum of the climatic mass balance and frontal ablation'},
        'glac_runoff_monthly': {
                'long_name': 'glacier-wide runoff',
                'units': 'm3',
                'temporal_resolution': 'monthly',
                'comment': 'runoff from the glacier terminus, which moves over time'},
        'glac_snowline_monthly': {
                'long_name': 'transient snowline altitude above mean sea level',
                'units': 'm',
                'temporal_resolution': 'monthly',
                'comment': 'transient snowline is altitude separating snow from ice/firn'},
        'glac_area_annual': {
                'long_name': 'glacier area',
                'units': 'km2',
                'temporal_resolution': 'annual',
                'comment': 'area at start of the year'},
        'glac_volume_annual': {
                'long_name': 'glacier volume',
                'units': 'km3',
                'temporal_resolution': 'annual',
                'comment': 'volume of ice based on area and ice thickness at start of the year'}, 
        'glac_ELA_annual': {
                'long_name': 'annual equilibrium line altitude above mean sea level',
                'units': 'm',
                'temporal_resolution': 'annual',
                'comment': (
                        'equilibrium line altitude is the elevation where the climatic mass balance is '
                        'zero')}, 
        'offglac_prec_monthly': {
                'long_name': 'off-glacier-wide precipitation (liquid)',
                'units': 'm',
                'temporal_resolution': 'monthly',
                'comment': 'only the liquid precipitation, solid precipitation excluded'},
        'offglac_refreeze_monthly': {
                'long_name': 'off-glacier-wide refreeze, in water equivalent',
                'units': 'm',
                'temporal_resolution': 'monthly'},
        'offglac_melt_monthly': {
                'long_name': 'off-glacier-wide melt, in water equivalent',
                'units': 'm',
                'temporal_resolution': 'monthly',
                'comment': 'only melt of snow and refreeze since off-glacier'},
        'offglac_runoff_monthly': {
                'long_name': 'off-glacier-wide runoff',
                'units': 'm3',
                'temporal_resolution': 'monthly',
                'comment': 'off-glacier runoff from area where glacier no longer exists'},
        'offglac_snowpack_monthly': {
                'long_name': 'off-glacier-wide snowpack, in water equivalent',
                'units': 'm',
                'temporal_resolution': 'monthly',
                'comment': 'snow remaining accounting for new accumulation, melt, and refreeze'},
        'glac_temp_monthly_std': {
                'standard_name': 'air_temperature',
                'long_name': 'glacier-wide mean air temperature standard deviation',
                'units': 'K',
                'temporal_resolution': 'monthly',
                'comment': (
                        'each elevation bin is weighted equally to compute the mean temperature, and '
                        'bins where the glacier no longer exists due to retreat have been removed')},
        'glac_prec_monthly_std': {
                'long_name': 'glacier-wide precipitation (liquid) standard deviation',
                'units': 'm',
                'temporal_resolution': 'monthly',
                'comment': 'only the liquid precipitation, solid precipitation excluded'},
        'glac_acc_monthly_std': {
                'long_name': 'glacier-wide accumulation, in water equivalent, standard deviation',
                'units': 'm',
                'temporal_resolution': 'monthly',
                'comment': 'only the solid precipitation'},
        'glac_refreeze_monthly_std': {
                'long_name': 'glacier-wide refreeze, in water equivalent, standard deviation',
                'units': 'm',
                'temporal_resolution': 'monthly'},
        'glac_melt_monthly_std': {
                'long_name': 'glacier-wide melt, in water equivalent, standard deviation',
                'units': 'm',
                'temporal_resolution': 'monthly'},
        'glac_frontalablation_monthly_std': {
                'long_name': 'glacier-wide frontal ablation, in water equivalent, standard deviation',
                'units': 'm',
                'temporal_resolution': 'monthly',
                'comment': (
                        'mass losses from calving, subaerial frontal melting, sublimation above the '
                        'waterline and subaqueous frontal melting below the waterline')},
        'glac_massbaltotal_monthly_std': {
                'long_name': 'glacier-wide total mass balance, in water equivalent, standard deviation',
                'units': 'm',
                'temporal_resolution': 'monthly',
                'comment': (
                        'total mass balance is the sum of the climatic mass balance and frontal '
                        'ablation')},
        'glac_runoff_monthly_std': {
                'long_name': 'glacier-wide runoff standard deviation',
                'units': 'm3',
                'temporal_resolution': 'monthly',
                'comment': 'runoff from the glacier terminus, which moves over time'},
        'glac_snowline_monthly_std': {
                'long_name': 'transient snowline above mean sea level standard deviation',
                'units': 'm',
                'temporal_resolution': 'monthly',
                'comment': 'transient snowline is altitude separating snow from ice/firn'},
        'glac_area_annual_std': {
                'long_name': 'glacier area standard deviation',
                'units': 'km2',
                'temporal_resolution': 'annual',
                'comment': 'area at start of the year'},
        'glac_volume_annual_std': {
                'long_name': 'glacier volume standard deviation',
                'units': 'km3',
                'temporal_resolution': 'annual',
                'comment': 'volume of ice based on area and ice thickness at start of the year'}, 
        'glac_ELA_annual_std': {
                'long_name': 'annual equilibrium line altitude above mean sea level standard deviation',
                'units': 'm',
                'temporal_resolution': 'annual',
                'comment': (
                        'equilibrium line altitude is the elevation where the climatic mass balance is '
                        'zero')}, 
        'offglac_prec_monthly_std': {
                'long_name': 'off-glacier-wide precipitation (liquid) standard deviation',
                'units': 'm',
                'temporal_resolution': 'monthly',
                'comment': 'only the liquid precipitation, solid precipitation excluded'},
        'offglac_refreeze_monthly_std': {
                'long_name': 'off-glacier-wide refreeze, in water equivalent, standard deviation',
                'units': 'm',
                'temporal_resolution': 'monthly'},
        'offglac_melt_monthly_std': {
                'long_name': 'off-glacier-wide melt, in water equivalent, standard deviation',
                'units': 'm',
                'temporal_resolution': 'monthly',
                'comment': 'only melt of snow and refreeze since off-glacier'},
        'offglac_runoff_monthly_std': {
                'long_name': 'off-glacier-wide runoff standard deviation',
                'units': 'm3',
                'temporal_resolution': 'monthly',
                'comment': 'off-glacier runoff from area where glacier no longer exists'},
        'offglac_snowpack_monthly_std': {
                'long_name': 'off-glacier-wide snowpack, in water equivalent, standard deviation',
                'units': 'm',
                'temporal_resolution': 'monthly',
                'comment': 'snow remaining accounting for new accumulation, melt, and refreeze'},
        }
        
# Add variables to empty dataset and merge together
count_vn = 0
encoding = {}
for vn in output_coords_dict.keys():
    count_vn += 1
    empty_holder = np.zeros([len(output_coords_dict[vn][i]) for i in list(output_coords_dict[vn].keys())])
    output_ds = xr.Dataset({vn: (list(output_coords_dict[vn].keys()), empty_holder)},
                           coords=output_coords_dict[vn])
    # Merge datasets of stats into one output
    if count_vn == 1:
        output_ds_all = output_ds
    else:
        output_ds_all = xr.merge((output_ds_all, output_ds))
# Add a glacier table so that the glaciers attributes accompany the netcdf file
#main_glac_rgi_float = main_glac_rgi[input.output_glacier_attr_vns].copy()
#main_glac_rgi_xr = xr.Dataset({'glacier_table': (('glac', 'glac_attrs'), main_glac_rgi_float.values)},
#                               coords={'glac': glac_values,
#                                       'glac_attrs': main_glac_rgi_float.columns.values})
#output_ds_all = output_ds_all.combine_first(main_glac_rgi_xr)
#output_ds_all.glacier_table.attrs['long_name'] = 'RGI glacier table'
#output_ds_all.glacier_table.attrs['comment'] = 'table contains attributes from RGI for each glacier'
#output_ds_all.glac_attrs.attrs['long_name'] = 'RGI glacier attributes'
noencoding_vn = ['RGIId']
# Add attributes
for vn in output_ds_all.variables:
    try:
        output_ds_all[vn].attrs = output_attrs_dict[vn]
    except:
        pass
    # Encoding (specify _FillValue, offsets, etc.)
    
    if vn not in noencoding_vn:
        encoding[vn] = {'_FillValue': False,
                        'zlib':True,
                        'complevel':9
                        }

#return output_ds_all, encoding

output_ds_all['RGIId'].values = ['RGI60-' + str(int(df.loc[x,'O1Region'])).zfill(2) + '.' + 
                                 str(int(df.loc[x,'glacno'])).zfill(5) for x in df.index.values]
output_ds_all['CenLon'].values = df.CenLon.values
output_ds_all['CenLat'].values = df.CenLat.values
output_ds_all['O1Region'].values = df.O1Region.values
output_ds_all['O2Region'].values = df.O2Region.values
output_ds_all['Area'].values = df.Area.values
output_ds_all['glac_prec_monthly'].values = ds.prec_glac_monthly.values[:,:,0]
output_ds_all['glac_temp_monthly'].values = ds.temp_glac_monthly.values[:,:,0] + 273.15
output_ds_all['glac_acc_monthly'].values = ds.acc_glac_monthly.values[:,:,0]
output_ds_all['glac_refreeze_monthly'].values = ds.refreeze_glac_monthly.values[:,:,0]
output_ds_all['glac_melt_monthly'].values = ds.melt_glac_monthly.values[:,:,0]
output_ds_all['glac_frontalablation_monthly'].values = ds.frontalablation_glac_monthly.values[:,:,0]
output_ds_all['glac_massbaltotal_monthly'].values = ds.massbaltotal_glac_monthly.values[:,:,0]
output_ds_all['glac_runoff_monthly'].values = ds.runoff_glac_monthly.values[:,:,0]
output_ds_all['glac_snowline_monthly'].values = ds.snowline_glac_monthly.values[:,:,0]
output_ds_all['glac_area_annual'].values = ds.area_glac_annual.values[:,:,0]
output_ds_all['glac_volume_annual'].values = ds.volume_glac_annual.values[:,:,0]
output_ds_all['glac_ELA_annual'].values = ds.ELA_glac_annual.values[:,:,0]
output_ds_all['offglac_prec_monthly'].values = ds.offglac_prec_monthly.values[:,:,0]
output_ds_all['offglac_refreeze_monthly'].values = ds.offglac_refreeze_monthly.values[:,:,0]
output_ds_all['offglac_melt_monthly'].values = ds.offglac_melt_monthly.values[:,:,0]
output_ds_all['offglac_snowpack_monthly'].values = ds.offglac_snowpack_monthly.values[:,:,0]
output_ds_all['offglac_runoff_monthly'].values = ds.offglac_runoff_monthly.values[:,:,0]
output_ds_all['glac_prec_monthly_std'].values = ds.prec_glac_monthly.values[:,:,1]
output_ds_all['glac_temp_monthly_std'].values = ds.temp_glac_monthly.values[:,:,1] + 273.15
output_ds_all['glac_acc_monthly_std'].values = ds.acc_glac_monthly.values[:,:,1]
output_ds_all['glac_refreeze_monthly_std'].values = ds.refreeze_glac_monthly.values[:,:,1]
output_ds_all['glac_melt_monthly_std'].values = ds.melt_glac_monthly.values[:,:,1]
output_ds_all['glac_frontalablation_monthly_std'].values = ds.frontalablation_glac_monthly.values[:,:,1]
output_ds_all['glac_massbaltotal_monthly_std'].values = ds.massbaltotal_glac_monthly.values[:,:,1]
output_ds_all['glac_runoff_monthly_std'].values = ds.runoff_glac_monthly.values[:,:,1]
output_ds_all['glac_snowline_monthly_std'].values = ds.snowline_glac_monthly.values[:,:,1]
output_ds_all['glac_area_annual_std'].values = ds.area_glac_annual.values[:,:,1]
output_ds_all['glac_volume_annual_std'].values = ds.volume_glac_annual.values[:,:,1]
output_ds_all['glac_ELA_annual_std'].values = ds.ELA_glac_annual.values[:,:,1]
output_ds_all['offglac_prec_monthly_std'].values = ds.offglac_prec_monthly.values[:,:,1]
output_ds_all['offglac_refreeze_monthly_std'].values = ds.offglac_refreeze_monthly.values[:,:,1]
output_ds_all['offglac_melt_monthly_std'].values = ds.offglac_melt_monthly.values[:,:,1]
output_ds_all['offglac_snowpack_monthly_std'].values = ds.offglac_snowpack_monthly.values[:,:,1]
output_ds_all['offglac_runoff_monthly_std'].values = ds.offglac_runoff_monthly.values[:,:,1]

output_ds_all.attrs = {
                        'source': 'PyGEMv0.1.0',
                        'institution': 'University of Alaska Fairbanks, Fairbanks, AK',
                        'history': 'Created by David Rounce (drounce@alaska.edu) on September 14 2019',
                        'references': 'doi:10.3389/feart.2019.00331 and doi:10.1017/jog.2019.91' }

output_ds_all.to_netcdf(output_fn, encoding=encoding)
output_ds_all.close()

