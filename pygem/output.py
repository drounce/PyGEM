#!/usr/bin/env python3
"""
Created on Sept 19 2023
Updated Mar 29 2024

@author: btobers mrweathers drounce

PyGEM classes and subclasses for glacier simulation outputs

The two main parent classes are 'single_glacier' and 'compiled_regional'
Both of these have several subclasses which will inherit the necessary parent information
"""
import pygem_input as pygem_prms
from dataclasses import dataclass
import numpy as np
import pandas as pd
import xarray as xr
import cftime
import os
import collections
from datetime import datetime

### single glacier output parent class ###
@dataclass
class single_glacier:
    """
    Single glacier output dataset class for the Python Glacier Evolution Model.
    """
    glacier_rgi_table : pd.DataFrame
    dates_table : pd.DataFrame
    wateryear : bool
    pygem_version : float
    user_info : dict

    def __post_init__(self):
        self.glac_values = np.array([self.glacier_rgi_table.name])
        self.get_time_vals()
        self.init_dict()

    def get_time_vals(self):
        if self.wateryear == 'hydro':
            self.year_type = 'water year'
            self.annual_columns = np.unique(self.dates_table['wateryear'].values)[0:int(self.dates_table.shape[0]/12)]
        elif self.wateryear == 'calendar':
            self.year_type = 'calendar year'
            self.annual_columns = np.unique(self.dates_table['year'].values)[0:int(self.dates_table.shape[0]/12)]
        elif self.wateryear == 'custom':
            self.year_type = 'custom year'
        self.time_values = self.dates_table.loc[pygem_prms.gcm_spinupyears*12:self.dates_table.shape[0]+1,'date'].tolist()
        self.time_values = [cftime.DatetimeNoLeap(x.year, x.month, x.day) for x in self.time_values]
        # append additional year to self.year_values to account for mass and area at end of period
        self.year_values = self.annual_columns[pygem_prms.gcm_spinupyears:self.annual_columns.shape[0]]
        self.year_values = np.concatenate((self.year_values, np.array([self.annual_columns[-1] + 1])))

    def init_dict(self):
        # Variable coordinates dictionary
        self.output_coords_dict = collections.OrderedDict()
        self.output_coords_dict['RGIId'] =  collections.OrderedDict([('glac', self.glac_values)])
        self.output_coords_dict['CenLon'] = collections.OrderedDict([('glac', self.glac_values)])
        self.output_coords_dict['CenLat'] = collections.OrderedDict([('glac', self.glac_values)])
        self.output_coords_dict['O1Region'] = collections.OrderedDict([('glac', self.glac_values)])
        self.output_coords_dict['O2Region'] = collections.OrderedDict([('glac', self.glac_values)])
        self.output_coords_dict['Area'] = collections.OrderedDict([('glac', self.glac_values)])

    def create_xr_ds(self):
        # Add variables to empty dataset and merge together
        count_vn = 0
        self.encoding = {}
        for vn in self.output_coords_dict.keys():
            count_vn += 1
            empty_holder = np.zeros([len(self.output_coords_dict[vn][i]) for i in list(self.output_coords_dict[vn].keys())])
            output_xr_ds_ = xr.Dataset({vn: (list(self.output_coords_dict[vn].keys()), empty_holder)},
                                coords=self.output_coords_dict[vn])
            # Merge datasets of stats into one output
            if count_vn == 1:
                self.output_xr_ds = output_xr_ds_
            else:
                self.output_xr_ds = xr.merge((self.output_xr_ds, output_xr_ds_))
        noencoding_vn = ['RGIId']
        # Add attributes
        for vn in self.output_xr_ds.variables:
            try:
                self.output_xr_ds[vn].attrs = self.output_attrs_dict[vn]
            except:
                pass
            # Encoding (specify _FillValue, offsets, etc.)
        
            if vn not in noencoding_vn:
                self.encoding[vn] = {'_FillValue': None,
                                'zlib':True,
                                'complevel':9
                                }    
        self.output_xr_ds['RGIId'].values = np.array([self.glacier_rgi_table.loc['RGIId']])
        self.output_xr_ds['CenLon'].values = np.array([self.glacier_rgi_table.CenLon])
        self.output_xr_ds['CenLat'].values = np.array([self.glacier_rgi_table.CenLat])
        self.output_xr_ds['O1Region'].values = np.array([self.glacier_rgi_table.O1Region])
        self.output_xr_ds['O2Region'].values = np.array([self.glacier_rgi_table.O2Region])
        self.output_xr_ds['Area'].values = np.array([self.glacier_rgi_table.Area * 1e6])
    
        self.output_xr_ds.attrs = {'source': f'PyGEMv{self.pygem_version}',
                        'institution': self.user_info['institution'],
                        'history': f'Created by {pygem_prms.user_info["name"]} ({self.user_info["email"]}) on ' + datetime.today().strftime('%Y-%m-%d'),
                        'references': 'doi:10.3389/feart.2019.00331 and doi:10.1017/jog.2019.91'}

    def get_xr_ds(self):
        return self.output_xr_ds
    
    def save_xr_ds(self, out_path, netcdf_fn):
        # Create filepath if it does not exist
        if os.path.exists(out_path) == False:
            os.makedirs(out_path, exist_ok=True)
        # export netcdf
        self.output_xr_ds.to_netcdf(out_path + netcdf_fn, encoding=self.encoding) 
        # close datasets
        self.output_xr_ds.close()

@dataclass
class glacierwide_stats(single_glacier):
    """
    Single glacier-wide statistics dataset
    """
    sim_iters : int
    extra_vars : bool

    def __post_init__(self):
        super().__post_init__()
        self.update_dict()
        self.get_dict_atts()
        self.create_xr_ds()

    def update_dict(self):
        self.output_coords_dict['glac_runoff_monthly'] = collections.OrderedDict([('glac', self.glac_values), 
                                                                            ('time', self.time_values)]) 
        self.output_coords_dict['glac_area_annual'] = collections.OrderedDict([('glac', self.glac_values),
                                                                        ('year', self.year_values)])
        self.output_coords_dict['glac_mass_annual'] = collections.OrderedDict([('glac', self.glac_values), 
                                                                            ('year', self.year_values)])
        self.output_coords_dict['glac_mass_bsl_annual'] = collections.OrderedDict([('glac', self.glac_values), 
                                                                                ('year', self.year_values)])
        self.output_coords_dict['glac_ELA_annual'] = collections.OrderedDict([('glac', self.glac_values),
                                                                        ('year', self.year_values)])
        self.output_coords_dict['offglac_runoff_monthly'] = collections.OrderedDict([('glac', self.glac_values), 
                                                                                ('time', self.time_values)])
        if self.sim_iters > 1:
            self.output_coords_dict['glac_runoff_monthly_mad'] = collections.OrderedDict([('glac', self.glac_values), 
                                                                                    ('time', self.time_values)])
            self.output_coords_dict['glac_area_annual_mad'] = collections.OrderedDict([('glac', self.glac_values), 
                                                                                ('year', self.year_values)])
            self.output_coords_dict['glac_mass_annual_mad'] = collections.OrderedDict([('glac', self.glac_values), 
                                                                                    ('year', self.year_values)])
            self.output_coords_dict['glac_mass_bsl_annual_mad'] = collections.OrderedDict([('glac', self.glac_values), 
                                                                                        ('year', self.year_values)])
            self.output_coords_dict['glac_ELA_annual_mad'] = collections.OrderedDict([('glac', self.glac_values), 
                                                                                ('year', self.year_values)])
            self.output_coords_dict['offglac_runoff_monthly_mad'] = collections.OrderedDict([('glac', self.glac_values), 
                                                                                        ('time', self.time_values)])
            
        if self.extra_vars:
            self.output_coords_dict['glac_prec_monthly'] = collections.OrderedDict([('glac', self.glac_values), 
                                                                            ('time', self.time_values)])
            self.output_coords_dict['glac_temp_monthly'] = collections.OrderedDict([('glac', self.glac_values), 
                                                                            ('time', self.time_values)])
            self.output_coords_dict['glac_acc_monthly'] = collections.OrderedDict([('glac', self.glac_values), 
                                                                            ('time', self.time_values)])
            self.output_coords_dict['glac_refreeze_monthly'] = collections.OrderedDict([('glac', self.glac_values), 
                                                                                ('time', self.time_values)])
            self.output_coords_dict['glac_melt_monthly'] = collections.OrderedDict([('glac', self.glac_values), 
                                                                            ('time', self.time_values)])
            self.output_coords_dict['glac_frontalablation_monthly'] = collections.OrderedDict([('glac', self.glac_values), 
                                                                                        ('time', self.time_values)])
            self.output_coords_dict['glac_massbaltotal_monthly'] = collections.OrderedDict([('glac', self.glac_values), 
                                                                                    ('time', self.time_values)])
            self.output_coords_dict['glac_snowline_monthly'] = collections.OrderedDict([('glac', self.glac_values), 
                                                                                ('time', self.time_values)])
            self.output_coords_dict['glac_mass_change_ignored_annual'] = collections.OrderedDict([('glac', self.glac_values),
                                                                                        ('year', self.year_values)])
            self.output_coords_dict['offglac_prec_monthly'] = collections.OrderedDict([('glac', self.glac_values), 
                                                                                ('time', self.time_values)])
            self.output_coords_dict['offglac_refreeze_monthly'] = collections.OrderedDict([('glac', self.glac_values), 
                                                                                    ('time', self.time_values)])
            self.output_coords_dict['offglac_melt_monthly'] = collections.OrderedDict([('glac', self.glac_values), 
                                                                                ('time', self.time_values)])
            self.output_coords_dict['offglac_snowpack_monthly'] = collections.OrderedDict([('glac', self.glac_values), 
                                                                                    ('time', self.time_values)])
            if self.sim_iters > 1:
                self.output_coords_dict['glac_prec_monthly_mad'] = collections.OrderedDict([('glac', self.glac_values), 
                                                                                    ('time', self.time_values)])
                self.output_coords_dict['glac_temp_monthly_mad'] = collections.OrderedDict([('glac', self.glac_values), 
                                                                                    ('time', self.time_values)])
                self.output_coords_dict['glac_acc_monthly_mad'] = collections.OrderedDict([('glac', self.glac_values), 
                                                                                    ('time', self.time_values)])
                self.output_coords_dict['glac_refreeze_monthly_mad'] = collections.OrderedDict([('glac', self.glac_values), 
                                                                                        ('time', self.time_values)])
                self.output_coords_dict['glac_melt_monthly_mad'] = collections.OrderedDict([('glac', self.glac_values), 
                                                                                    ('time', self.time_values)])
                self.output_coords_dict['glac_frontalablation_monthly_mad'] = collections.OrderedDict([('glac', self.glac_values), 
                                                                                                ('time', self.time_values)])
                self.output_coords_dict['glac_massbaltotal_monthly_mad'] = collections.OrderedDict([('glac', self.glac_values), 
                                                                                            ('time', self.time_values)])
                self.output_coords_dict['glac_snowline_monthly_mad'] = collections.OrderedDict([('glac', self.glac_values), 
                                                                                        ('time', self.time_values)])
                self.output_coords_dict['glac_mass_change_ignored_annual_mad'] = collections.OrderedDict([('glac', self.glac_values),
                                                                                                    ('year', self.year_values)])
                self.output_coords_dict['offglac_prec_monthly_mad'] = collections.OrderedDict([('glac', self.glac_values), 
                                                                                        ('time', self.time_values)])
                self.output_coords_dict['offglac_refreeze_monthly_mad'] = collections.OrderedDict([('glac', self.glac_values), 
                                                                                            ('time', self.time_values)])
                self.output_coords_dict['offglac_melt_monthly_mad'] = collections.OrderedDict([('glac', self.glac_values), 
                                                                                        ('time', self.time_values)])
                self.output_coords_dict['offglac_snowpack_monthly_mad'] = collections.OrderedDict([('glac', self.glac_values), 
                                                                                            ('time', self.time_values)])

    def get_dict_atts(self):
        # Attributes dictionary
        self.output_attrs_dict = {
            'time': {
                    'long_name': 'time',
                    'year_type':self.year_type,
                    'comment':'start of the month'},
            'glac': {
                    'long_name': 'glacier index',
                    'comment': 'glacier index referring to glaciers properties and model results'},
            'year': {
                    'long_name': 'years',
                    'year_type': self.year_type,
                    'comment': 'years referring to the start of each year'},
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
                    'units': 'm2',
                    'comment': 'value from RGIv6.0'},
            'glac_runoff_monthly': {
                    'long_name': 'glacier-wide runoff',
                    'units': 'm3',
                    'temporal_resolution': 'monthly',
                    'comment': 'runoff from the glacier terminus, which moves over time'},
            'glac_area_annual': {
                    'long_name': 'glacier area',
                    'units': 'm2',
                    'temporal_resolution': 'annual',
                    'comment': 'area at start of the year'},
            'glac_mass_annual': {
                    'long_name': 'glacier mass',
                    'units': 'kg',
                    'temporal_resolution': 'annual',
                    'comment': 'mass of ice based on area and ice thickness at start of the year'},
            'glac_mass_bsl_annual': {
                    'long_name': 'glacier mass below sea level',
                    'units': 'kg',
                    'temporal_resolution': 'annual',
                    'comment': 'mass of ice below sea level based on area and ice thickness at start of the year'},
            'glac_ELA_annual': {
                    'long_name': 'annual equilibrium line altitude above mean sea level',
                    'units': 'm',
                    'temporal_resolution': 'annual',
                    'comment': 'equilibrium line altitude is the elevation where the climatic mass balance is zero'}, 
            'offglac_runoff_monthly': {
                    'long_name': 'off-glacier-wide runoff',
                    'units': 'm3',
                    'temporal_resolution': 'monthly',
                    'comment': 'off-glacier runoff from area where glacier no longer exists'},
            }
        
        if self.sim_iters > 1:
            output_attrs_dict_mad = {
                'glac_runoff_monthly_mad': {
                        'long_name': 'glacier-wide runoff median absolute deviation',
                        'units': 'm3',
                        'temporal_resolution': 'monthly',
                        'comment': 'runoff from the glacier terminus, which moves over time'},
                'glac_area_annual_mad': {
                        'long_name': 'glacier area median absolute deviation',
                        'units': 'm2',
                        'temporal_resolution': 'annual',
                        'comment': 'area at start of the year'},
                'glac_mass_annual_mad': {
                        'long_name': 'glacier mass median absolute deviation',
                        'units': 'kg',
                        'temporal_resolution': 'annual',
                        'comment': 'mass of ice based on area and ice thickness at start of the year'},
                'glac_mass_bsl_annual_mad': {
                        'long_name': 'glacier mass below sea level median absolute deviation',
                        'units': 'kg',
                        'temporal_resolution': 'annual',
                        'comment': 'mass of ice below sea level based on area and ice thickness at start of the year'},
                'glac_ELA_annual_mad': {
                        'long_name': 'annual equilibrium line altitude above mean sea level median absolute deviation',
                        'units': 'm',
                        'temporal_resolution': 'annual',
                        'comment': 'equilibrium line altitude is the elevation where the climatic mass balance is zero'}, 
                'offglac_runoff_monthly_mad': {
                        'long_name': 'off-glacier-wide runoff median absolute deviation',
                        'units': 'm3',
                        'temporal_resolution': 'monthly',
                        'comment': 'off-glacier runoff from area where glacier no longer exists'},
                }
            self.output_attrs_dict.update(output_attrs_dict_mad)
            
        if self.extra_vars:
            output_attrs_dict_extras = {
                'glac_temp_monthly': {
                        'standard_name': 'air_temperature',
                        'long_name': 'glacier-wide mean air temperature',
                        'units': 'K',
                        'temporal_resolution': 'monthly',
                        'comment': ('each elevation bin is weighted equally to compute the mean temperature, and '
                                    'bins where the glacier no longer exists due to retreat have been removed')},
                'glac_prec_monthly': {
                        'long_name': 'glacier-wide precipitation (liquid)',
                        'units': 'm3',
                        'temporal_resolution': 'monthly',
                        'comment': 'only the liquid precipitation, solid precipitation excluded'},
                'glac_acc_monthly': {
                        'long_name': 'glacier-wide accumulation, in water equivalent',
                        'units': 'm3',
                        'temporal_resolution': 'monthly',
                        'comment': 'only the solid precipitation'},
                'glac_refreeze_monthly': {
                        'long_name': 'glacier-wide refreeze, in water equivalent',
                        'units': 'm3',
                        'temporal_resolution': 'monthly'},
                'glac_melt_monthly': {
                        'long_name': 'glacier-wide melt, in water equivalent',
                        'units': 'm3',
                        'temporal_resolution': 'monthly'},
                'glac_frontalablation_monthly': {
                        'long_name': 'glacier-wide frontal ablation, in water equivalent',
                        'units': 'm3',
                        'temporal_resolution': 'monthly',
                        'comment': (
                                'mass losses from calving, subaerial frontal melting, sublimation above the '
                                'waterline and subaqueous frontal melting below the waterline; positive values indicate mass lost like melt')},
                'glac_massbaltotal_monthly': {
                        'long_name': 'glacier-wide total mass balance, in water equivalent',
                        'units': 'm3',
                        'temporal_resolution': 'monthly',
                        'comment': 'total mass balance is the sum of the climatic mass balance and frontal ablation'},
                'glac_snowline_monthly': {
                    'long_name': 'transient snowline altitude above mean sea level',
                    'units': 'm',
                    'temporal_resolution': 'monthly',
                    'comment': 'transient snowline is altitude separating snow from ice/firn'},
                'glac_mass_change_ignored_annual': { 
                    'long_name': 'glacier mass change ignored',
                    'units': 'kg',
                    'temporal_resolution': 'annual',
                    'comment': 'glacier mass change ignored due to flux divergence'},
                'offglac_prec_monthly': {
                    'long_name': 'off-glacier-wide precipitation (liquid)',
                    'units': 'm3',
                    'temporal_resolution': 'monthly',
                    'comment': 'only the liquid precipitation, solid precipitation excluded'},
                'offglac_refreeze_monthly': {
                        'long_name': 'off-glacier-wide refreeze, in water equivalent',
                        'units': 'm3',
                        'temporal_resolution': 'monthly'},
                'offglac_melt_monthly': {
                        'long_name': 'off-glacier-wide melt, in water equivalent',
                        'units': 'm3',
                        'temporal_resolution': 'monthly',
                        'comment': 'only melt of snow and refreeze since off-glacier'},
                'offglac_snowpack_monthly': {
                    'long_name': 'off-glacier-wide snowpack, in water equivalent',
                    'units': 'm3',
                    'temporal_resolution': 'monthly',
                    'comment': 'snow remaining accounting for new accumulation, melt, and refreeze'}
                }
            self.output_attrs_dict.update(output_attrs_dict_extras)
            
            if self.sim_iters > 1:
                output_attrs_dict_extras_mad = {
                    'glac_temp_monthly_mad': {
                            'standard_name': 'air_temperature',
                            'long_name': 'glacier-wide mean air temperature median absolute deviation',
                            'units': 'K',
                            'temporal_resolution': 'monthly',
                            'comment': (
                                    'each elevation bin is weighted equally to compute the mean temperature, and '
                                    'bins where the glacier no longer exists due to retreat have been removed')},
                    'glac_prec_monthly_mad': {
                            'long_name': 'glacier-wide precipitation (liquid) median absolute deviation',
                            'units': 'm3',
                            'temporal_resolution': 'monthly',
                            'comment': 'only the liquid precipitation, solid precipitation excluded'},
                    'glac_acc_monthly_mad': {
                            'long_name': 'glacier-wide accumulation, in water equivalent, median absolute deviation',
                            'units': 'm3',
                            'temporal_resolution': 'monthly',
                            'comment': 'only the solid precipitation'},
                    'glac_refreeze_monthly_mad': {
                            'long_name': 'glacier-wide refreeze, in water equivalent, median absolute deviation',
                            'units': 'm3',
                            'temporal_resolution': 'monthly'},
                    'glac_melt_monthly_mad': {
                            'long_name': 'glacier-wide melt, in water equivalent, median absolute deviation',
                            'units': 'm3',
                            'temporal_resolution': 'monthly'},
                    'glac_frontalablation_monthly_mad': {
                            'long_name': 'glacier-wide frontal ablation, in water equivalent, median absolute deviation',
                            'units': 'm3',
                            'temporal_resolution': 'monthly',
                            'comment': (
                                    'mass losses from calving, subaerial frontal melting, sublimation above the '
                                    'waterline and subaqueous frontal melting below the waterline')},
                    'glac_massbaltotal_monthly_mad': {
                            'long_name': 'glacier-wide total mass balance, in water equivalent, median absolute deviation',
                            'units': 'm3',
                            'temporal_resolution': 'monthly',
                            'comment': 'total mass balance is the sum of the climatic mass balance and frontal ablation'},
                    'glac_snowline_monthly_mad': {
                            'long_name': 'transient snowline above mean sea level median absolute deviation',
                            'units': 'm',
                            'temporal_resolution': 'monthly',
                            'comment': 'transient snowline is altitude separating snow from ice/firn'},
                    'glac_mass_change_ignored_annual_mad': { 
                            'long_name': 'glacier mass change ignored median absolute deviation',
                            'units': 'kg',
                            'temporal_resolution': 'annual',
                            'comment': 'glacier mass change ignored due to flux divergence'},
                    'offglac_prec_monthly_mad': {
                            'long_name': 'off-glacier-wide precipitation (liquid) median absolute deviation',
                            'units': 'm3',
                            'temporal_resolution': 'monthly',
                            'comment': 'only the liquid precipitation, solid precipitation excluded'},
                    'offglac_refreeze_monthly_mad': {
                            'long_name': 'off-glacier-wide refreeze, in water equivalent, median absolute deviation',
                            'units': 'm3',
                            'temporal_resolution': 'monthly'},
                    'offglac_melt_monthly_mad': {
                            'long_name': 'off-glacier-wide melt, in water equivalent, median absolute deviation',
                            'units': 'm3',
                            'temporal_resolution': 'monthly',
                            'comment': 'only melt of snow and refreeze since off-glacier'},
                    
                    'offglac_snowpack_monthly_mad': {
                            'long_name': 'off-glacier-wide snowpack, in water equivalent, median absolute deviation',
                            'units': 'm3',
                            'temporal_resolution': 'monthly',
                            'comment': 'snow remaining accounting for new accumulation, melt, and refreeze'},
                    }
                self.output_attrs_dict.update(output_attrs_dict_extras_mad)
    

@dataclass
class binned_stats(single_glacier):
    """
    Single glacier binned dataset
    """




### compiled regional output parent class ###
@dataclass
class compiled_regional:
    """
    Compiled regional output dataset for the Python Glacier Evolution Model.
    """

@dataclass
class regional_annual_mass(compiled_regional):
    """
    compiled regional annual mass
    """

@dataclass
class regional_annual_area(compiled_regional):
    """
    compiled regional annual area
    """

@dataclass
class regional_monthly_runoff(compiled_regional):
    """
    compiled regional monthly runoff
    """

@dataclass
class regional_monthly_massbal(compiled_regional):
    """
    compiled regional monthly climatic mass balance
    """
