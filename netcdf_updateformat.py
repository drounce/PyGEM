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
import zipfile
# External Libraries
import numpy as np
import pandas as pd 
import netCDF4 as nc
import scipy
import cartopy
import xarray as xr
# Local Libraries
import pygem.pygem_input as pygem_prms
#%%
netcdf_fullfn = '/Users/davidrounce/Documents/Dave_Rounce/HiMAT/PyGEM/../Climate_data/ERA5/ERA5_lapserates_monthly.nc'
ds = xr.open_dataset(netcdf_fullfn)

encoding = {}
for vn in ds.variables:
    encoding[vn] = {'_FillValue': False,
                    'zlib':True,
                    'complevel':9
                    }

ds.to_netcdf(netcdf_fullfn.replace('.nc','-compressed.nc'), encoding=encoding)

#%%

option_update_fileformat = 0
option_plot_4eachfile = 0
option_update_fileformat_subset = True

if option_update_fileformat == 1:
    option_files = 0 # 1 is the individual, 0 is the multimodel
    
#    netcdf_fullfns_all = []
#    if option_files == 1:
#        # Process zipped individual GCM files
#        netcdf_fp_cmip5 = '/Volumes/LaCie/HMA_PyGEM/2019_0914/spc_zipped/'
#        for i in os.listdir(netcdf_fp_cmip5):
#            if os.path.isdir(netcdf_fp_cmip5 + i):
#                for j in os.listdir(netcdf_fp_cmip5 + i):
#                    if j.endswith('.nc.zip'):
#                        netcdf_fullfns_all.append(netcdf_fp_cmip5 + i + '/' + j)
#    else:
#        # Process multi-model files
#        netcdf_fp_cmip5 = '/Volumes/LaCie/HMA_PyGEM/2019_0914/'
#        for i in os.listdir(netcdf_fp_cmip5):
#            if i.endswith('2100.nc'):
#                netcdf_fullfns_all.append(netcdf_fp_cmip5 + i)
    netcdf_fullfns_all = ['/Volumes/LaCie/HMA_PyGEM/2019_0914/ERA-Interim/' + 
                           'R15--all--ERA-Interim_c2_ba1_100sets_1980_2017.nc']
    
    for netcdf_fn in netcdf_fullfns_all:
    #for netcdf_fn in [netcdf_fullfns_all[8]]:
        unzip_fp = '/'.join(netcdf_fn.split('/')[:-1]) + '/'
        if option_files == 1:
            unzip_fn = (netcdf_fn.split('/')[-1]).replace('.zip','')
            with zipfile.ZipFile(netcdf_fn, 'r') as zipObj:
                # Extract all the contents of zip file in current directory
                zipObj.extractall(unzip_fp)
        else:
            unzip_fn = netcdf_fn.split('/')[-1]
    
        output_fn = unzip_fp + unzip_fn.replace('.nc','-v2.nc')
    
        if os.path.exists(output_fn) == False:
    
            print(unzip_fn)
            
            ds = xr.open_dataset(unzip_fp + unzip_fn)
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
            #    if pygem_prms.output_package == 2:
                    
            # Create empty datasets for each variable and merge them
            # Coordinate values
            output_variables = pygem_prms.output_variables_package2
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
            #    record_name_values = pygem_prms.sim_stat_cns
            
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
            #main_glac_rgi_float = main_glac_rgi[pygem_prms.output_glacier_attr_vns].copy()
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
            
            print('  processing mean monthly variables...')
            
            output_ds_all['glac_prec_monthly'].values = ds.prec_glac_monthly.values[:,:,0]
            output_ds_all['glac_temp_monthly'].values = ds.temp_glac_monthly.values[:,:,0] + 273.15
            output_ds_all['glac_acc_monthly'].values = ds.acc_glac_monthly.values[:,:,0]
            output_ds_all['glac_refreeze_monthly'].values = ds.refreeze_glac_monthly.values[:,:,0]
            output_ds_all['glac_melt_monthly'].values = ds.melt_glac_monthly.values[:,:,0]
            output_ds_all['glac_frontalablation_monthly'].values = ds.frontalablation_glac_monthly.values[:,:,0]
            output_ds_all['glac_massbaltotal_monthly'].values = ds.massbaltotal_glac_monthly.values[:,:,0]
            output_ds_all['glac_runoff_monthly'].values = ds.runoff_glac_monthly.values[:,:,0]
            output_ds_all['glac_snowline_monthly'].values = ds.snowline_glac_monthly.values[:,:,0]
            
            print('  processing mean annual variables...')
            
            output_ds_all['glac_area_annual'].values = ds.area_glac_annual.values[:,:,0]
            output_ds_all['glac_volume_annual'].values = ds.volume_glac_annual.values[:,:,0]
            output_ds_all['glac_ELA_annual'].values = ds.ELA_glac_annual.values[:,:,0]
            output_ds_all['offglac_prec_monthly'].values = ds.offglac_prec_monthly.values[:,:,0]
            output_ds_all['offglac_refreeze_monthly'].values = ds.offglac_refreeze_monthly.values[:,:,0]
            output_ds_all['offglac_melt_monthly'].values = ds.offglac_melt_monthly.values[:,:,0]
            output_ds_all['offglac_snowpack_monthly'].values = ds.offglac_snowpack_monthly.values[:,:,0]
            output_ds_all['offglac_runoff_monthly'].values = ds.offglac_runoff_monthly.values[:,:,0]
            
            print('  processing std monthly variables...')
            
            output_ds_all['glac_prec_monthly_std'].values = ds.prec_glac_monthly.values[:,:,1]
            output_ds_all['glac_temp_monthly_std'].values = ds.temp_glac_monthly.values[:,:,1]
            output_ds_all['glac_acc_monthly_std'].values = ds.acc_glac_monthly.values[:,:,1]
            output_ds_all['glac_refreeze_monthly_std'].values = ds.refreeze_glac_monthly.values[:,:,1]
            output_ds_all['glac_melt_monthly_std'].values = ds.melt_glac_monthly.values[:,:,1]
            output_ds_all['glac_frontalablation_monthly_std'].values = ds.frontalablation_glac_monthly.values[:,:,1]
            output_ds_all['glac_massbaltotal_monthly_std'].values = ds.massbaltotal_glac_monthly.values[:,:,1]
            output_ds_all['glac_runoff_monthly_std'].values = ds.runoff_glac_monthly.values[:,:,1]
            output_ds_all['glac_snowline_monthly_std'].values = ds.snowline_glac_monthly.values[:,:,1]
            
            print('  processing std annual variables...')
            
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
            
            
#%%
if option_plot_4eachfile == 1:
    netcdf_fp_cmip5 = '/Volumes/LaCie/HMA_PyGEM/2019_0914/multimodel_updated/'  
    
    vn = 'volume_glac_annual'

#    startyear = 2015
#    endyear = 2100
    
    figure_fp = netcdf_fp_cmip5 + 'figures/'
    if os.path.exists(figure_fp) == False:
        os.makedirs(figure_fp)
    
    netcdf_fns = []
    for i in os.listdir(netcdf_fp_cmip5):
        if 'multigcm' in i and i.endswith('_c2_ba1_100sets_2000_2100.nc'):
            netcdf_fns.append(i)
            
    for netcdf_fn in netcdf_fns:
#    for netcdf_fn in netcdf_fns[0:1]:
        print(netcdf_fn)
        ds = xr.open_dataset(netcdf_fp_cmip5 + netcdf_fn)
        rgino_str_all = [x.split('-')[1] for x in ds.RGIId.values]
        
        # Load glaciers
        import pygemfxns_modelsetup as modelsetup
        main_glac_rgi = modelsetup.selectglaciersrgitable(glac_no=rgino_str_all)
        # Add subgroups for uncertainty
        hex55_dict_fn = '/Users/davidrounce/Documents/Dave_Rounce/HiMAT/qgis_himat/rgi60_HMA_dict_hexbins_55km.csv'
        hex55_csv = pd.read_csv(hex55_dict_fn)
        hex55_dict = dict(zip(hex55_csv.RGIId, hex55_csv.hexid55))
        main_glac_rgi['hexid55'] = main_glac_rgi.RGIId.map(hex55_dict)
        subgroups = main_glac_rgi.hexid55.unique().tolist()
        subgroup_cn = 'hexid55'
        
        #%%
        # Load data
        yrs = ds.year.values
        yrs_plus1 = ds.year_plus1.values
        # Volume
        vol_glac_all = ds.glac_volume_annual.values 
        vol_glac_std_all = ds.glac_volume_annual_std.values
        
        # Fixed-gauge runoff
        runoff_onglac_all = ds.glac_runoff_monthly.values
        runoff_onglac_std_all = ds.glac_runoff_monthly_std.values
        runoff_offglac_all = ds.offglac_runoff_monthly.values
        runoff_offglac_std_all = ds.offglac_runoff_monthly_std.values
        # aggregate on and off-glacier runoff
        runoff_glac_all = runoff_onglac_all + runoff_offglac_all
        runoff_glac_std_all = runoff_onglac_std_all + runoff_offglac_std_all
        # convert monthly to annual
        import pygemfxns_gcmbiasadj as gcmbiasadj
        runoff_glac_all_annual = gcmbiasadj.annual_sum_2darray(runoff_glac_all)
        runoff_glac_std_all_annual = gcmbiasadj.annual_sum_2darray(runoff_glac_std_all)
        
        
        # Regional volume change
        reg_vol_annual = vol_glac_all.sum(axis=0)
        reg_runoff_annual = runoff_glac_all_annual.sum(axis=0)        
        # Uncertainty associated with volume change based on subgroups
        #  sum standard deviations in each subgroup assuming that they are uncorrelated
        #  then use the root sum of squares using the uncertainty of each subgroup to get the 
        #  uncertainty of the group
        subgroup_vol_std = np.zeros((len(subgroups), len(yrs_plus1)))
        subgroup_runoff_std = np.zeros((len(subgroups), len(yrs)))
        for nsubgroup, subgroup in enumerate(subgroups):
            main_glac_rgi_subgroup = main_glac_rgi.loc[main_glac_rgi[subgroup_cn] == subgroup]
            subgroup_indices = (
                    main_glac_rgi.loc[main_glac_rgi[subgroup_cn] == subgroup].index.values.tolist())
            # subgroup uncertainty is sum of each glacier since assumed to be perfectly correlated
            subgroup_vol_std[nsubgroup,:] = vol_glac_std_all[subgroup_indices,:].sum(axis=0)
            subgroup_runoff_std[nsubgroup,:] = runoff_glac_std_all_annual[subgroup_indices,:].sum(axis=0)
        reg_vol_annual_std = (subgroup_vol_std**2).sum(axis=0)**0.5  
        reg_runoff_annual_std = (subgroup_runoff_std**2).sum(axis=0)**0.5  
        
        #%%
        # ===== Plot Volume Change =====
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1, squeeze=False, sharex=False, sharey=False, gridspec_kw = {'wspace':0.4, 'hspace':0.15})
                     
        # Plot
        ax[0,0].plot(yrs_plus1, reg_vol_annual, color='k', linewidth=1, zorder=2)
        # Fill between
        ax[0,0].fill_between(yrs_plus1, reg_vol_annual - reg_vol_annual_std, reg_vol_annual + reg_vol_annual_std, 
                             facecolor='k', alpha=0.2, zorder=1)
        # X-label
        ax[0,0].set_xlabel('Year', size=12)
        ax[0,0].set_xlim(2000,2100)
        ax[0,0].xaxis.set_tick_params(labelsize=12)         
        # Y-label
        ax[0,0].set_ylabel('Volume (km$^{3}$)', size=12)
        ax[0,0].yaxis.set_tick_params(labelsize=12)
        if netcdf_fn.startswith('R15'):
            ax[0,0].set_ylim(0,1000)
        else:
            ax[0,0].set_ylim(0,3600)
        # Tick parameters
        ax[0,0].tick_params(axis='both', which='major', labelsize=12, direction='inout')           
        # Save figure
        #  figures can be saved in any format (.jpg, .png, .pdf, etc.)
        fig.set_size_inches(4, 4)
        figure_fn = netcdf_fn.replace('.nc', '-volume.png')
        fig.savefig(figure_fp + figure_fn, bbox_inches='tight', dpi=300)
        
        #%%
        # ===== Plot Runoff Change =====
        fig, ax = plt.subplots(1, 1, squeeze=False, sharex=False, sharey=False, gridspec_kw = {'wspace':0.4, 'hspace':0.15})
                     
        # Plot
        ax[0,0].plot(yrs, reg_runoff_annual / 1e9, color='k', linewidth=1, zorder=2)
        # Fill between
        ax[0,0].fill_between(yrs, (reg_runoff_annual - reg_runoff_annual_std) / 1e9, 
                             (reg_runoff_annual + reg_runoff_annual_std) / 1e9, 
                             facecolor='k', alpha=0.2, zorder=1)
        # X-label
        ax[0,0].set_xlabel('Year', size=12)
        ax[0,0].set_xlim(2000,2100)
        ax[0,0].xaxis.set_tick_params(labelsize=12)         
        # Y-label
        ax[0,0].set_ylabel('Runoff (km$^{3}$)', size=12)
        ax[0,0].yaxis.set_tick_params(labelsize=12)
        if netcdf_fn.startswith('R15'):
            ax[0,0].set_ylim(25,60)
        elif netcdf_fn.startswith('R14'):
            ax[0,0].set_ylim(25,60)
        else:
            ax[0,0].set_ylim(45,90)
        # Tick parameters
        ax[0,0].tick_params(axis='both', which='major', labelsize=12, direction='inout')           
        # Save figure
        #  figures can be saved in any format (.jpg, .png, .pdf, etc.)
        fig.set_size_inches(4, 4)
        figure_fn = netcdf_fn.replace('.nc', '-runoff.png')
        fig.savefig(figure_fp + figure_fn, bbox_inches='tight', dpi=300)
        
        
        
        
if option_update_fileformat_subset:
    option_files = 1 # 1 is the individual, 0 is the multimodel
    
    netcdf_fullfns_all = []
    if option_files == 1:
        # Process zipped individual GCM files
        netcdf_fp_cmip5 = '/Volumes/LaCie/HMA_PyGEM/2019_0914/spc_zipped/'
        for i in os.listdir(netcdf_fp_cmip5):
            if os.path.isdir(netcdf_fp_cmip5 + i):
                for j in os.listdir(netcdf_fp_cmip5 + i):
                    if j.endswith('.nc.zip'):
                        netcdf_fullfns_all.append(netcdf_fp_cmip5 + i + '/' + j)
    else:
        # Process multi-model files
        netcdf_fp_cmip5 = '/Volumes/LaCie/HMA_PyGEM/2019_0914/'
        for i in os.listdir(netcdf_fp_cmip5):
            if i.endswith('2100.nc'):
                netcdf_fullfns_all.append(netcdf_fp_cmip5 + i)
#    netcdf_fullfns_all = ['/Volumes/LaCie/HMA_PyGEM/2019_0914/ERA-Interim/' + 
#                           'R13--all--ERA-Interim_c2_ba1_100sets_1980_2017.nc']
    
    for netcdf_fn in netcdf_fullfns_all:
    #for netcdf_fn in [netcdf_fullfns_all[8]]:
        unzip_fp = '/'.join(netcdf_fn.split('/')[:-1]) + '/'
        if option_files == 1:
            unzip_fn = (netcdf_fn.split('/')[-1]).replace('.zip','')
            with zipfile.ZipFile(netcdf_fn, 'r') as zipObj:
                # Extract all the contents of zip file in current directory
                zipObj.extractall(unzip_fp)
        else:
            unzip_fn = netcdf_fn.split('/')[-1]
    
        output_fn = unzip_fp + unzip_fn.replace('.nc','-v2.nc')
    
        if os.path.exists(output_fn) == False:
    
            print(unzip_fn)
            
            ds = xr.open_dataset(unzip_fp + unzip_fn)
            df = pd.DataFrame(ds.glacier_table.values, columns=ds.glac_attrs)
            
                    
            # Create empty datasets for each variable and merge them
            # Coordinate values
            output_variables = pygem_prms.output_variables_package2
            glac_values = ds.glac.values
            time_values = ds.time.values
            year_values = ds.year.values
            year_plus1_values = ds.year_plus1.values
            year_type = ds.time.year_type
                
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
            output_coords_dict['glac_area_annual'] = collections.OrderedDict([('glac', glac_values), ('year_plus1', year_plus1_values)])
            output_coords_dict['glac_volume_annual'] = collections.OrderedDict([('glac', glac_values), ('year_plus1', year_plus1_values)])
            output_coords_dict['offglac_runoff_monthly'] = collections.OrderedDict([('glac', glac_values), ('time', time_values)])
            output_coords_dict['glac_prec_monthly_std'] = collections.OrderedDict([('glac', glac_values), ('time', time_values)])
            output_coords_dict['glac_temp_monthly_std'] = collections.OrderedDict([('glac', glac_values), ('time', time_values)])
            output_coords_dict['glac_acc_monthly_std'] = collections.OrderedDict([('glac', glac_values), ('time', time_values)])
            output_coords_dict['glac_refreeze_monthly_std'] = collections.OrderedDict([('glac', glac_values), ('time', time_values)])
            output_coords_dict['glac_melt_monthly_std'] = collections.OrderedDict([('glac', glac_values), ('time', time_values)])
            output_coords_dict['glac_frontalablation_monthly_std'] = collections.OrderedDict([('glac', glac_values), ('time', time_values)])
            output_coords_dict['glac_massbaltotal_monthly_std'] = collections.OrderedDict([('glac', glac_values), ('time', time_values)])
            output_coords_dict['glac_runoff_monthly_std'] = collections.OrderedDict([('glac', glac_values), ('time', time_values)])
            output_coords_dict['glac_area_annual_std'] = collections.OrderedDict([('glac', glac_values), ('year_plus1', year_plus1_values)])
            output_coords_dict['glac_volume_annual_std'] = collections.OrderedDict([('glac', glac_values), ('year_plus1', year_plus1_values)])
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
            
            output_ds_all['RGIId'].values = ['RGI60-' + str(int(df.loc[x,'O1Region'])).zfill(2) + '.' + 
                                             str(int(df.loc[x,'glacno'])).zfill(5) for x in df.index.values]
            output_ds_all['CenLon'].values = df.CenLon.values
            output_ds_all['CenLat'].values = df.CenLat.values
            output_ds_all['O1Region'].values = df.O1Region.values
            output_ds_all['O2Region'].values = df.O2Region.values
            output_ds_all['Area'].values = df.Area.values
            
            print('  processing mean monthly variables...')
            
            output_ds_all['glac_prec_monthly'].values = ds.prec_glac_monthly.values[:,:,0]
            output_ds_all['glac_temp_monthly'].values = ds.temp_glac_monthly.values[:,:,0] + 273.15
            output_ds_all['glac_acc_monthly'].values = ds.acc_glac_monthly.values[:,:,0]
            output_ds_all['glac_refreeze_monthly'].values = ds.refreeze_glac_monthly.values[:,:,0]
            output_ds_all['glac_melt_monthly'].values = ds.melt_glac_monthly.values[:,:,0]
            output_ds_all['glac_frontalablation_monthly'].values = ds.frontalablation_glac_monthly.values[:,:,0]
            output_ds_all['glac_massbaltotal_monthly'].values = ds.massbaltotal_glac_monthly.values[:,:,0]
            output_ds_all['glac_runoff_monthly'].values = ds.runoff_glac_monthly.values[:,:,0]
            output_ds_all['offglac_runoff_monthly'].values = ds.offglac_runoff_monthly.values[:,:,0]
            
            print('  processing mean annual variables...')
            
            output_ds_all['glac_area_annual'].values = ds.area_glac_annual.values[:,:,0]
            output_ds_all['glac_volume_annual'].values = ds.volume_glac_annual.values[:,:,0]
            
            
            print('  processing std monthly variables...')
            
            output_ds_all['glac_prec_monthly_std'].values = ds.prec_glac_monthly.values[:,:,1]
            output_ds_all['glac_temp_monthly_std'].values = ds.temp_glac_monthly.values[:,:,1]
            output_ds_all['glac_acc_monthly_std'].values = ds.acc_glac_monthly.values[:,:,1]
            output_ds_all['glac_refreeze_monthly_std'].values = ds.refreeze_glac_monthly.values[:,:,1]
            output_ds_all['glac_melt_monthly_std'].values = ds.melt_glac_monthly.values[:,:,1]
            output_ds_all['glac_frontalablation_monthly_std'].values = ds.frontalablation_glac_monthly.values[:,:,1]
            output_ds_all['glac_massbaltotal_monthly_std'].values = ds.massbaltotal_glac_monthly.values[:,:,1]
            output_ds_all['glac_runoff_monthly_std'].values = ds.runoff_glac_monthly.values[:,:,1]
            output_ds_all['offglac_runoff_monthly_std'].values = ds.offglac_runoff_monthly.values[:,:,1]
            
            print('  processing std annual variables...')
            
            output_ds_all['glac_area_annual_std'].values = ds.area_glac_annual.values[:,:,1]
            output_ds_all['glac_volume_annual_std'].values = ds.volume_glac_annual.values[:,:,1]
            
            
            output_ds_all.attrs = {
                                    'source': 'PyGEMv0.1.0',
                                    'institution': 'University of Alaska Fairbanks, Fairbanks, AK',
                                    'history': 'Created by David Rounce (drounce@alaska.edu) on September 14 2019',
                                    'references': 'doi:10.3389/feart.2019.00331 and doi:10.1017/jog.2019.91' }
            
            output_ds_all.to_netcdf(output_fn, encoding=encoding)
            output_ds_all.close()