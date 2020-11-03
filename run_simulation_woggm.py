"""Run a model simulation."""
# Default climate data is ERA-Interim; specify CMIP5 by specifying a filename to the argument:
#    (Command line) python run_simulation_list_multiprocess.py -gcm_list_fn=C:\...\gcm_rcpXX_filenames.txt
#      - Default is running ERA-Interim in parallel with five processors.
#    (Spyder) %run run_simulation_list_multiprocess.py C:\...\gcm_rcpXX_filenames.txt -option_parallels=0
#      - Spyder cannot run parallels, so always set -option_parallels=0 when testing in Spyder.
# Spyder cannot run parallels, so always set -option_parallels=0 when testing in Spyder.

# Built-in libraries
import argparse
import collections
import inspect
import multiprocessing
import os
import time
# External libraries
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
# Local libraries
import class_climate
#import class_mbdata
import pygem.pygem_input as pygem_prms
from pygem.massbalance import PyGEMMassBalance
from pygem.glacierdynamics import MassRedistributionCurveModel
from pygem.oggm_compat import single_flowline_glacier_directory, single_flowline_glacier_directory_with_calving
from pygem.shop import debris
import pygemfxns_gcmbiasadj as gcmbiasadj
import pygemfxns_modelsetup as modelsetup
import spc_split_glaciers as split_glaciers

from oggm import cfg
from oggm import graphics
from oggm import tasks
from oggm import utils
from oggm.core import climate
#print('Switch back to OGGM import of FluxBasedModel')
#from pygem.glacierdynamics import FluxBasedModel
from oggm.core.flowline import FluxBasedModel
from oggm.core.inversion import calving_flux_from_depth


#%% FUNCTIONS
def getparser():
    """
    Use argparse to add arguments from the command line

    Parameters
    ----------
    gcm_list_fn (optional) : str
        text file that contains the climate data to be used in the model simulation
    gcm_name (optional) : str
        gcm name
    rcp (optional) : str
        representative concentration pathway (ex. 'rcp26')
    num_simultaneous_processes (optional) : int
        number of cores to use in parallels
    option_parallels (optional) : int
        switch to use parallels or not
    rgi_glac_number_fn (optional) : str
        filename of .pkl file containing a list of glacier numbers that used to run batches on the supercomputer
    batch_number (optional): int
        batch number used to differentiate output on supercomputer
    option_ordered : int
        option to keep glaciers ordered or to grab every n value for the batch
        (the latter helps make sure run times on each core are similar as it removes any timing differences caused by
         regional variations)
    debug (optional) : int
        Switch for turning debug printing on or off (default = 0 (off))
    debug_spc (optional) : int
        Switch for turning debug printing of spc on or off (default = 0 (off))

    Returns
    -------
    Object containing arguments and their respective values.
    """
    parser = argparse.ArgumentParser(description="run simulations from gcm list in parallel")
    # add arguments
    parser.add_argument('-gcm_list_fn', action='store', type=str, default=pygem_prms.ref_gcm_name,
                        help='text file full of commands to run')
    parser.add_argument('-gcm_name', action='store', type=str, default=None,
                        help='GCM name used for model run')
    parser.add_argument('-rcp', action='store', type=str, default=None,
                        help='rcp scenario used for model run (ex. rcp26)')
    parser.add_argument('-num_simultaneous_processes', action='store', type=int, default=4,
                        help='number of simultaneous processes (cores) to use')
    parser.add_argument('-option_parallels', action='store', type=int, default=1,
                        help='Switch to use or not use parallels (1 - use parallels, 0 - do not)')
    parser.add_argument('-rgi_glac_number_fn', action='store', type=str, default=None,
                        help='Filename containing list of rgi_glac_number, helpful for running batches on spc')
    parser.add_argument('-batch_number', action='store', type=int, default=None,
                        help='Batch number used to differentiate output on supercomputer')
    parser.add_argument('-option_ordered', action='store', type=int, default=1,
                        help='switch to keep lists ordered or not')
    parser.add_argument('-debug', action='store', type=int, default=0,
                        help='Boolean for debugging to turn it on or off (default 0 is off')
    parser.add_argument('-debug_spc', action='store', type=int, default=0,
                        help='Boolean for debugging to turn it on or off (default 0 is off')
    return parser


def calc_stats_array(data, stats_cns=pygem_prms.sim_stat_cns):
    """
    Calculate stats for a given variable

    Parameters
    ----------
    vn : str
        variable name
    ds : xarray dataset
        dataset of output with all ensemble simulations

    Returns
    -------
    stats : np.array
        Statistics related to a given variable
    """
    if 'mean' in stats_cns:
        stats = data.mean(axis=1)[:,np.newaxis]
    if 'std' in stats_cns:
        stats = np.append(stats, data.std(axis=1)[:,np.newaxis], axis=1)
    if '2.5%' in stats_cns:
        stats = np.append(stats, np.percentile(data, 2.5, axis=1)[:,np.newaxis], axis=1)
    if '25%' in stats_cns:
        stats = np.append(stats, np.percentile(data, 25, axis=1)[:,np.newaxis], axis=1)
    if 'median' in stats_cns:
        stats = np.append(stats, np.median(data, axis=1)[:,np.newaxis], axis=1)
    if '75%' in stats_cns:
        stats = np.append(stats, np.percentile(data, 75, axis=1)[:,np.newaxis], axis=1)
    if '97.5%' in stats_cns:
        stats = np.append(stats, np.percentile(data, 97.5, axis=1)[:,np.newaxis], axis=1)
    return stats


def create_xrdataset(glacier_rgi_table, dates_table, option_wateryear=pygem_prms.gcm_wateryear):
    """
    Create empty xarray dataset that will be used to record simulation runs.

    Parameters
    ----------
    main_glac_rgi : pandas dataframe
        dataframe containing relevant rgi glacier information
    dates_table : pandas dataframe
        table of the dates, months, days in month, etc.

    Returns
    -------
    output_ds_all : xarray Dataset
        empty xarray dataset that contains variables and attributes to be filled in by simulation runs
    encoding : dictionary
        encoding used with exporting xarray dataset to netcdf
    """
    if pygem_prms.output_package == 2:
        # Create empty datasets for each variable and merge them
        # Coordinate values
        glac_values = np.array([glacier_rgi_table.name])

        # Time attributes and values
        if option_wateryear == 'hydro':
            year_type = 'water year'
            annual_columns = np.unique(dates_table['wateryear'].values)[0:int(dates_table.shape[0]/12)]
        elif option_wateryear == 'calendar':
            year_type = 'calendar year'
            annual_columns = np.unique(dates_table['year'].values)[0:int(dates_table.shape[0]/12)]
        elif option_wateryear == 'custom':
            year_type = 'custom year'
            
        time_values = dates_table.loc[pygem_prms.gcm_spinupyears*12:dates_table.shape[0]+1,'date'].tolist()
        # append additional year to year_values to account for volume and area at end of period
        year_values = annual_columns[pygem_prms.gcm_spinupyears:annual_columns.shape[0]]
        year_values = np.concatenate((year_values, np.array([annual_columns[-1] + 1])))

        # Variable coordinates dictionary
        output_coords_dict = collections.OrderedDict()
        output_coords_dict['RGIId'] =  collections.OrderedDict([('glac', glac_values)])
        output_coords_dict['CenLon'] = collections.OrderedDict([('glac', glac_values)])
        output_coords_dict['CenLat'] = collections.OrderedDict([('glac', glac_values)])
        output_coords_dict['O1Region'] = collections.OrderedDict([('glac', glac_values)])
        output_coords_dict['O2Region'] = collections.OrderedDict([('glac', glac_values)])
        output_coords_dict['Area'] = collections.OrderedDict([('glac', glac_values)])
        output_coords_dict['glac_prec_monthly'] = collections.OrderedDict([('glac', glac_values), 
                                                                           ('time', time_values)])
        output_coords_dict['glac_temp_monthly'] = collections.OrderedDict([('glac', glac_values), 
                                                                           ('time', time_values)])
        output_coords_dict['glac_acc_monthly'] = collections.OrderedDict([('glac', glac_values), 
                                                                          ('time', time_values)])
        output_coords_dict['glac_refreeze_monthly'] = collections.OrderedDict([('glac', glac_values), 
                                                                               ('time', time_values)])
        output_coords_dict['glac_melt_monthly'] = collections.OrderedDict([('glac', glac_values), 
                                                                           ('time', time_values)])
        output_coords_dict['glac_frontalablation_monthly'] = collections.OrderedDict([('glac', glac_values), 
                                                                                      ('time', time_values)])
        output_coords_dict['glac_massbaltotal_monthly'] = collections.OrderedDict([('glac', glac_values), 
                                                                                   ('time', time_values)])
        output_coords_dict['glac_runoff_monthly'] = collections.OrderedDict([('glac', glac_values), 
                                                                             ('time', time_values)]) 
        output_coords_dict['glac_snowline_monthly'] = collections.OrderedDict([('glac', glac_values), 
                                                                               ('time', time_values)])
        output_coords_dict['glac_area_annual'] = collections.OrderedDict([('glac', glac_values), ('year', year_values)])
        output_coords_dict['glac_volume_annual'] = collections.OrderedDict([('glac', glac_values), 
                                                                            ('year', year_values)])
        output_coords_dict['glac_volume_change_ignored_annual'] = collections.OrderedDict([('glac', glac_values),
                                                                                           ('year', year_values)])
        output_coords_dict['glac_ELA_annual'] = collections.OrderedDict([('glac', glac_values),
                                                                         ('year', year_values)])
        output_coords_dict['offglac_prec_monthly'] = collections.OrderedDict([('glac', glac_values), 
                                                                              ('time', time_values)])
        output_coords_dict['offglac_refreeze_monthly'] = collections.OrderedDict([('glac', glac_values), 
                                                                                  ('time', time_values)])
        output_coords_dict['offglac_melt_monthly'] = collections.OrderedDict([('glac', glac_values), 
                                                                              ('time', time_values)])
        output_coords_dict['offglac_snowpack_monthly'] = collections.OrderedDict([('glac', glac_values), 
                                                                                  ('time', time_values)])
        output_coords_dict['offglac_runoff_monthly'] = collections.OrderedDict([('glac', glac_values), 
                                                                                ('time', time_values)])
        output_coords_dict['glac_prec_monthly_std'] = collections.OrderedDict([('glac', glac_values), 
                                                                               ('time', time_values)])
        output_coords_dict['glac_temp_monthly_std'] = collections.OrderedDict([('glac', glac_values), 
                                                                               ('time', time_values)])
        output_coords_dict['glac_acc_monthly_std'] = collections.OrderedDict([('glac', glac_values), 
                                                                              ('time', time_values)])
        output_coords_dict['glac_refreeze_monthly_std'] = collections.OrderedDict([('glac', glac_values), 
                                                                                   ('time', time_values)])
        output_coords_dict['glac_melt_monthly_std'] = collections.OrderedDict([('glac', glac_values), 
                                                                               ('time', time_values)])
        output_coords_dict['glac_frontalablation_monthly_std'] = collections.OrderedDict([('glac', glac_values), 
                                                                                          ('time', time_values)])
        output_coords_dict['glac_massbaltotal_monthly_std'] = collections.OrderedDict([('glac', glac_values), 
                                                                                       ('time', time_values)])
        output_coords_dict['glac_runoff_monthly_std'] = collections.OrderedDict([('glac', glac_values), 
                                                                                 ('time', time_values)])
        output_coords_dict['glac_snowline_monthly_std'] = collections.OrderedDict([('glac', glac_values), 
                                                                                   ('time', time_values)])
        output_coords_dict['glac_area_annual_std'] = collections.OrderedDict([('glac', glac_values), 
                                                                              ('year', year_values)])
        output_coords_dict['glac_volume_annual_std'] = collections.OrderedDict([('glac', glac_values), 
                                                                                ('year', year_values)])
        output_coords_dict['glac_volume_change_ignored_annual_std'] = collections.OrderedDict([('glac', glac_values),
                                                                                               ('year', year_values)])
        output_coords_dict['glac_ELA_annual_std'] = collections.OrderedDict([('glac', glac_values), 
                                                                             ('year', year_values)])
        output_coords_dict['offglac_prec_monthly_std'] = collections.OrderedDict([('glac', glac_values), 
                                                                                  ('time', time_values)])
        output_coords_dict['offglac_refreeze_monthly_std'] = collections.OrderedDict([('glac', glac_values), 
                                                                                      ('time', time_values)])
        output_coords_dict['offglac_melt_monthly_std'] = collections.OrderedDict([('glac', glac_values), 
                                                                                  ('time', time_values)])
        output_coords_dict['offglac_snowpack_monthly_std'] = collections.OrderedDict([('glac', glac_values), 
                                                                                      ('time', time_values)])
        output_coords_dict['offglac_runoff_monthly_std'] = collections.OrderedDict([('glac', glac_values), 
                                                                                    ('time', time_values)])
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
                            'waterline and subaqueous frontal melting below the waterline')},
            'glac_massbaltotal_monthly': {
                    'long_name': 'glacier-wide total mass balance, in water equivalent',
                    'units': 'm3',
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
                    'units': 'm2',
                    'temporal_resolution': 'annual',
                    'comment': 'area at start of the year'},
            'glac_volume_annual': {
                    'long_name': 'glacier volume',
                    'units': 'm3',
                    'temporal_resolution': 'annual',
                    'comment': 'volume of ice based on area and ice thickness at start of the year'},
            'glac_volume_change_ignored_annual': { 
                    'long_name': 'glacier volume change ignored',
                    'units': 'm3',
                    'temporal_resolution': 'annual',
                    'comment': 'glacier volume change ignored due to flux divergence'},
            'glac_ELA_annual': {
                    'long_name': 'annual equilibrium line altitude above mean sea level',
                    'units': 'm',
                    'temporal_resolution': 'annual',
                    'comment': 'equilibrium line altitude is the elevation where the climatic mass balance is zero'}, 
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
            'offglac_runoff_monthly': {
                    'long_name': 'off-glacier-wide runoff',
                    'units': 'm3',
                    'temporal_resolution': 'monthly',
                    'comment': 'off-glacier runoff from area where glacier no longer exists'},
            'offglac_snowpack_monthly': {
                    'long_name': 'off-glacier-wide snowpack, in water equivalent',
                    'units': 'm3',
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
                    'units': 'm3',
                    'temporal_resolution': 'monthly',
                    'comment': 'only the liquid precipitation, solid precipitation excluded'},
            'glac_acc_monthly_std': {
                    'long_name': 'glacier-wide accumulation, in water equivalent, standard deviation',
                    'units': 'm3',
                    'temporal_resolution': 'monthly',
                    'comment': 'only the solid precipitation'},
            'glac_refreeze_monthly_std': {
                    'long_name': 'glacier-wide refreeze, in water equivalent, standard deviation',
                    'units': 'm3',
                    'temporal_resolution': 'monthly'},
            'glac_melt_monthly_std': {
                    'long_name': 'glacier-wide melt, in water equivalent, standard deviation',
                    'units': 'm3',
                    'temporal_resolution': 'monthly'},
            'glac_frontalablation_monthly_std': {
                    'long_name': 'glacier-wide frontal ablation, in water equivalent, standard deviation',
                    'units': 'm3',
                    'temporal_resolution': 'monthly',
                    'comment': (
                            'mass losses from calving, subaerial frontal melting, sublimation above the '
                            'waterline and subaqueous frontal melting below the waterline')},
            'glac_massbaltotal_monthly_std': {
                    'long_name': 'glacier-wide total mass balance, in water equivalent, standard deviation',
                    'units': 'm3',
                    'temporal_resolution': 'monthly',
                    'comment': 'total mass balance is the sum of the climatic mass balance and frontal ablation'},
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
                    'units': 'm2',
                    'temporal_resolution': 'annual',
                    'comment': 'area at start of the year'},
            'glac_volume_annual_std': {
                    'long_name': 'glacier volume standard deviation',
                    'units': 'm3',
                    'temporal_resolution': 'annual',
                    'comment': 'volume of ice based on area and ice thickness at start of the year'},
            'glac_volume_change_ignored_annual_std': { 
                    'long_name': 'glacier volume change ignored standard deviation',
                    'units': 'm3',
                    'temporal_resolution': 'annual',
                    'comment': 'glacier volume change ignored due to flux divergence'},
            'glac_ELA_annual_std': {
                    'long_name': 'annual equilibrium line altitude above mean sea level standard deviation',
                    'units': 'm',
                    'temporal_resolution': 'annual',
                    'comment': 'equilibrium line altitude is the elevation where the climatic mass balance is zero'}, 
            'offglac_prec_monthly_std': {
                    'long_name': 'off-glacier-wide precipitation (liquid) standard deviation',
                    'units': 'm3',
                    'temporal_resolution': 'monthly',
                    'comment': 'only the liquid precipitation, solid precipitation excluded'},
            'offglac_refreeze_monthly_std': {
                    'long_name': 'off-glacier-wide refreeze, in water equivalent, standard deviation',
                    'units': 'm3',
                    'temporal_resolution': 'monthly'},
            'offglac_melt_monthly_std': {
                    'long_name': 'off-glacier-wide melt, in water equivalent, standard deviation',
                    'units': 'm3',
                    'temporal_resolution': 'monthly',
                    'comment': 'only melt of snow and refreeze since off-glacier'},
            'offglac_runoff_monthly_std': {
                    'long_name': 'off-glacier-wide runoff standard deviation',
                    'units': 'm3',
                    'temporal_resolution': 'monthly',
                    'comment': 'off-glacier runoff from area where glacier no longer exists'},
            'offglac_snowpack_monthly_std': {
                    'long_name': 'off-glacier-wide snowpack, in water equivalent, standard deviation',
                    'units': 'm3',
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
        output_ds_all['RGIId'].values = np.array(['RGI60-' + str(int(glacier_rgi_table.loc['O1Region'])).zfill(2) + 
                                         '.' + str(int(glacier_rgi_table.loc['glacno'])).zfill(5)])
        output_ds_all['CenLon'].values = np.array([glacier_rgi_table.CenLon])
        output_ds_all['CenLat'].values = np.array([glacier_rgi_table.CenLat])
        output_ds_all['O1Region'].values = np.array([glacier_rgi_table.O1Region])
        output_ds_all['O2Region'].values = np.array([glacier_rgi_table.O2Region])
        output_ds_all['Area'].values = np.array([glacier_rgi_table.Area * 1e6])
        
        output_ds.attrs = {'source': 'PyGEMv0.1.0',
                           'institution': 'University of Alaska Fairbanks, Fairbanks, AK',
                           'history': 'Created by David Rounce (drounce@alaska.edu) on ' + pygem_prms.model_run_date,
                           'references': 'doi:10.3389/feart.2019.00331 and doi:10.1017/jog.2019.91'}
        
    return output_ds_all, encoding


def main(list_packed_vars):
    """
    Model simulation

    Parameters
    ----------
    list_packed_vars : list
        list of packed variables that enable the use of parallels

    Returns
    -------
    netcdf files of the simulation output (specific output is dependent on the output option)
    """
    # Unpack variables
    count = list_packed_vars[0]
    glac_no = list_packed_vars[1]
    gcm_name = list_packed_vars[2]

    parser = getparser()
    args = parser.parse_args()

    if (gcm_name != pygem_prms.ref_gcm_name) and (args.rcp is None):
        rcp_scenario = os.path.basename(args.gcm_list_fn).split('_')[1]
    elif args.rcp is not None:
        rcp_scenario = args.rcp

    if debug:
        if 'rcp_scenario' in locals():
            print(rcp_scenario)

    if args.debug_spc == 1:
        debug_spc = True
    else:
        debug_spc = False

    # ===== LOAD GLACIERS =====
    main_glac_rgi = modelsetup.selectglaciersrgitable(glac_no=glac_no)

    # ===== TIME PERIOD =====
    dates_table = modelsetup.datesmodelrun(
            startyear=pygem_prms.gcm_startyear, endyear=pygem_prms.gcm_endyear, spinupyears=pygem_prms.gcm_spinupyears,
            option_wateryear=pygem_prms.gcm_wateryear)

    # ===== LOAD CLIMATE DATA =====
    # Climate class
    if gcm_name in ['ERA5', 'ERA-Interim', 'COAWST']:
        gcm = class_climate.GCM(name=gcm_name)
        if pygem_prms.option_synthetic_sim == 0:
            assert pygem_prms.gcm_endyear <= int(time.strftime("%Y")), 'Climate data not available to gcm_endyear'
    else:
        # GCM object
        gcm = class_climate.GCM(name=gcm_name, rcp_scenario=rcp_scenario)
        # Reference GCM
        ref_gcm = class_climate.GCM(name=pygem_prms.ref_gcm_name)
        # Adjust reference dates in event that reference is longer than GCM data
        if pygem_prms.ref_startyear >= pygem_prms.gcm_startyear:
            ref_startyear = pygem_prms.ref_startyear
        else:
            ref_startyear = pygem_prms.gcm_startyear
        if pygem_prms.ref_endyear <= pygem_prms.gcm_endyear:
            ref_endyear = pygem_prms.ref_endyear
        else:
            ref_endyear = pygem_prms.gcm_endyear
        dates_table_ref = modelsetup.datesmodelrun(startyear=ref_startyear, endyear=ref_endyear,
                                                   spinupyears=pygem_prms.ref_spinupyears,
                                                   option_wateryear=pygem_prms.ref_wateryear)
    
    # Select climate data
    if pygem_prms.option_synthetic_sim == 0:
        # Air temperature [degC]
        gcm_temp, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(gcm.temp_fn, gcm.temp_vn, main_glac_rgi,
                                                                     dates_table)
        if pygem_prms.option_ablation != 2:
            gcm_tempstd = np.zeros(gcm_temp.shape)
        elif pygem_prms.option_ablation == 2 and gcm_name in ['ERA5']:
            gcm_tempstd, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(gcm.tempstd_fn, gcm.tempstd_vn,
                                                                            main_glac_rgi, dates_table)
        elif pygem_prms.option_ablation == 2 and pygem_prms.ref_gcm_name in ['ERA5']:
            # Compute temp std based on reference climate data
            ref_tempstd, ref_dates = ref_gcm.importGCMvarnearestneighbor_xarray(ref_gcm.tempstd_fn, ref_gcm.tempstd_vn,
                                                                                main_glac_rgi, dates_table_ref)
            # Monthly average from reference climate data
            gcm_tempstd = gcmbiasadj.monthly_avg_array_rolled(ref_tempstd, dates_table_ref, dates_table)
        else:
            gcm_tempstd = np.zeros(gcm_temp.shape)

        # Precipitation [m]
        gcm_prec, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(gcm.prec_fn, gcm.prec_vn, main_glac_rgi,
                                                                     dates_table)
        # Elevation [m asl]
        gcm_elev = gcm.importGCMfxnearestneighbor_xarray(gcm.elev_fn, gcm.elev_vn, main_glac_rgi)
        # Lapse rate
        if gcm_name in ['ERA-Interim', 'ERA5']:
            gcm_lr, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(gcm.lr_fn, gcm.lr_vn, main_glac_rgi, dates_table)
        else:
            # Compute lapse rates based on reference climate data
            ref_lr, ref_dates = ref_gcm.importGCMvarnearestneighbor_xarray(ref_gcm.lr_fn, ref_gcm.lr_vn, main_glac_rgi,
                                                                           dates_table_ref)
            # Monthly average from reference climate data
            gcm_lr = gcmbiasadj.monthly_avg_array_rolled(ref_lr, dates_table_ref, dates_table)


    # ===== BIAS CORRECTIONS =====
    # No adjustments
    if pygem_prms.option_bias_adjustment == 0 or gcm_name == pygem_prms.ref_gcm_name:
        gcm_temp_adj = gcm_temp
        gcm_prec_adj = gcm_prec
        gcm_elev_adj = gcm_elev
    # Bias correct based on reference climate data
    else:
        # Air temperature [degC], Precipitation [m], Elevation [masl], Lapse rate [K m-1]
        ref_temp, ref_dates = ref_gcm.importGCMvarnearestneighbor_xarray(ref_gcm.temp_fn, ref_gcm.temp_vn,
                                                                         main_glac_rgi, dates_table_ref)
        ref_prec, ref_dates = ref_gcm.importGCMvarnearestneighbor_xarray(ref_gcm.prec_fn, ref_gcm.prec_vn,
                                                                         main_glac_rgi, dates_table_ref)
        ref_elev = ref_gcm.importGCMfxnearestneighbor_xarray(ref_gcm.elev_fn, ref_gcm.elev_vn, main_glac_rgi)
        
        # OPTION 1: Adjust temp using Huss and Hock (2015), prec similar but addresses for variance and outliers
        if pygem_prms.option_bias_adjustment == 1:
            # Temperature bias correction
            gcm_temp_adj, gcm_elev_adj = gcmbiasadj.temp_biasadj_HH2015(ref_temp, ref_elev, gcm_temp,
                                                                        dates_table_ref, dates_table)
            # Precipitation bias correction
            gcm_prec_adj, gcm_elev_adj = gcmbiasadj.prec_biasadj_opt1(ref_prec, ref_elev, gcm_prec,
                                                                      dates_table_ref, dates_table)
        # OPTION 2: Adjust temp and prec using Huss and Hock (2015)
        elif pygem_prms.option_bias_adjustment == 2:
            # Temperature bias correction
            gcm_temp_adj, gcm_elev_adj = gcmbiasadj.temp_biasadj_HH2015(ref_temp, ref_elev, gcm_temp,
                                                                        dates_table_ref, dates_table)
            # Precipitation bias correction
            gcm_prec_adj, gcm_elev_adj = gcmbiasadj.prec_biasadj_HH2015(ref_prec, ref_elev, gcm_prec,
                                                                        dates_table_ref, dates_table)
            
    # ===== RUN MASS BALANCE =====
    # Number of simulations
    if pygem_prms.option_calibration == 'MCMC':
        sim_iters = pygem_prms.sim_iters
    else:
        sim_iters = 1
    
    # Number of years (for OGGM's run_until_and_store)
    if pygem_prms.timestep == 'monthly':
        nyears = int(dates_table.shape[0]/12)
    else:
        assert True==False, 'Adjust nyears for non-monthly timestep'

    for glac in range(main_glac_rgi.shape[0]):
        if glac == 0 or glac == main_glac_rgi.shape[0]:
            print(gcm_name,':', main_glac_rgi.loc[main_glac_rgi.index.values[glac],'RGIId'])
        # Select subsets of data
        glacier_rgi_table = main_glac_rgi.loc[main_glac_rgi.index.values[glac], :]
        glacier_str = '{0:0.5f}'.format(glacier_rgi_table['RGIId_float'])

        # ===== Load glacier data: area (km2), ice thickness (m), width (km) =====
        if glacier_rgi_table['TermType'] == 1:
            gdir = single_flowline_glacier_directory_with_calving(glacier_str)
        else:
            gdir = single_flowline_glacier_directory(glacier_str)
            # reset glacier directory to overwrite
    #        gd = single_flowline_glacier_directory(glacier_str, reset=True)
    #        fls = gdir.read_pickle('model_flowlines')
        fls = gdir.read_pickle('inversion_flowlines')

        # Add climate data to glacier directory
        gdir.historical_climate = {'elev': gcm_elev_adj[glac],
                                   'temp': gcm_temp_adj[glac,:],
                                   'tempstd': gcm_tempstd[glac,:],
                                   'prec': gcm_prec_adj[glac,:],
                                   'lr': gcm_lr[glac,:]}
        gdir.dates_table = dates_table

        glacier_area_km2 = fls[0].widths_m * fls[0].dx_meter / 1e6
        if glacier_area_km2.sum() > 0:
#            if pygem_prms.hindcast == 1:
#                glacier_gcm_prec = glacier_gcm_prec[::-1]
#                glacier_gcm_temp = glacier_gcm_temp[::-1]
#                glacier_gcm_lrgcm = glacier_gcm_lrgcm[::-1]
#                glacier_gcm_lrglac = glacier_gcm_lrglac[::-1]

            # Load model parameters
            if pygem_prms.use_calibrated_modelparams:
                
                assert os.path.exists(gdir.get_filepath('pygem_modelprms')), 'Calibrated parameters do not exist.'
                with open(gdir.get_filepath('pygem_modelprms'), 'rb') as f:
                    modelprms_dict = pickle.load(f)

                assert pygem_prms.option_calibration in modelprms_dict, ('Error: ' + pygem_prms.option_calibration +
                                                                         ' not in modelprms_dict')
                modelprms_all = modelprms_dict[pygem_prms.option_calibration]
                # MCMC needs model parameters to be selected
                if pygem_prms.option_calibration == 'MCMC':
                    sim_iters = pygem_prms.sim_iters
                    if sim_iters == 1:
                        modelprms_all = {'kp': [np.median(modelprms_all['kp']['chain_0'])],
                                         'tbias': [np.median(modelprms_all['tbias']['chain_0'])],
                                         'ddfsnow': [np.median(modelprms_all['ddfsnow']['chain_0'])],
                                         'ddfice': [np.median(modelprms_all['ddfice']['chain_0'])],
                                         'tsnow_threshold': modelprms_all['tsnow_threshold'],
                                         'precgrad': modelprms_all['precgrad']}
                    else:
                        # Select every kth iteration to use for the ensemble
                        mcmc_sample_no = len(modelprms_all['kp']['chain_0'])
                        mp_spacing = int((mcmc_sample_no - pygem_prms.sim_burn) / sim_iters)
                        mp_idx_start = np.arange(pygem_prms.sim_burn, pygem_prms.sim_burn + mp_spacing)
                        np.random.shuffle(mp_idx_start)
                        mp_idx_start = mp_idx_start[0]
                        mp_idx_all = np.arange(mp_idx_start, mcmc_sample_no, mp_spacing)
                        modelprms_all = {
                                'kp': [modelprms_all['kp']['chain_0'][mp_idx] for mp_idx in mp_idx_all],
                                'tbias': [modelprms_all['tbias']['chain_0'][mp_idx] for mp_idx in mp_idx_all],
                                'ddfsnow': [modelprms_all['ddfsnow']['chain_0'][mp_idx] for mp_idx in mp_idx_all],
                                'ddfice': [modelprms_all['ddfice']['chain_0'][mp_idx] for mp_idx in mp_idx_all],
                                'tsnow_threshold': modelprms_all['tsnow_threshold'] * sim_iters,
                                'precgrad': modelprms_all['precgrad'] * sim_iters}
                else:
                    sim_iters = 1
            else:
                modelprms_all = {'kp': [pygem_prms.kp],
                                 'tbias': [pygem_prms.tbias],
                                 'ddfsnow': [pygem_prms.ddfsnow],
                                 'ddfice': [pygem_prms.ddfice],
                                 'tsnow_threshold': [pygem_prms.tsnow_threshold],
                                 'precgrad': [pygem_prms.precgrad]}
            
            # Time attributes and values
            if pygem_prms.gcm_wateryear == 'hydro':
                annual_columns = np.unique(dates_table['wateryear'].values)[0:int(dates_table.shape[0]/12)]
            else:
                annual_columns = np.unique(dates_table['year'].values)[0:int(dates_table.shape[0]/12)]
            # append additional year to year_values to account for volume and area at end of period
            year_values = annual_columns[pygem_prms.gcm_spinupyears:annual_columns.shape[0]]
            year_values = np.concatenate((year_values, np.array([annual_columns[-1] + 1])))
            output_glac_temp_monthly = np.zeros((dates_table.shape[0], sim_iters))
            output_glac_prec_monthly = np.zeros((dates_table.shape[0], sim_iters))
            output_glac_acc_monthly = np.zeros((dates_table.shape[0], sim_iters))
            output_glac_refreeze_monthly = np.zeros((dates_table.shape[0], sim_iters))
            output_glac_melt_monthly = np.zeros((dates_table.shape[0], sim_iters))
            output_glac_frontalablation_monthly = np.zeros((dates_table.shape[0], sim_iters))
            output_glac_massbaltotal_monthly = np.zeros((dates_table.shape[0], sim_iters))
            output_glac_runoff_monthly = np.zeros((dates_table.shape[0], sim_iters))
            output_glac_snowline_monthly = np.zeros((dates_table.shape[0], sim_iters))
            output_glac_area_annual = np.zeros((year_values.shape[0], sim_iters))
            output_glac_volume_annual = np.zeros((year_values.shape[0], sim_iters))
            output_glac_volume_change_ignored_annual = np.zeros((year_values.shape[0], sim_iters))
            output_glac_ELA_annual = np.zeros((year_values.shape[0], sim_iters))
            output_offglac_prec_monthly = np.zeros((dates_table.shape[0], sim_iters))
            output_offglac_refreeze_monthly = np.zeros((dates_table.shape[0], sim_iters))
            output_offglac_melt_monthly = np.zeros((dates_table.shape[0], sim_iters))
            output_offglac_snowpack_monthly = np.zeros((dates_table.shape[0], sim_iters))
            output_offglac_runoff_monthly = np.zeros((dates_table.shape[0], sim_iters))
            

            # Loop through model parameters
            for n_iter in range(sim_iters):

                modelprms = {'kp': modelprms_all['kp'][n_iter],
                             'tbias': modelprms_all['tbias'][n_iter],
                             'ddfsnow': modelprms_all['ddfsnow'][n_iter],
                             'ddfice': modelprms_all['ddfice'][n_iter],
                             'tsnow_threshold': modelprms_all['tsnow_threshold'][n_iter],
                             'precgrad': modelprms_all['precgrad'][n_iter]}

                if debug:
                    print(glacier_str + '  kp: ' + str(np.round(modelprms['kp'],2)) +
                          ' ddfsnow: ' + str(np.round(modelprms['ddfsnow'],4)) +
                          ' tbias: ' + str(np.round(modelprms['tbias'],2)))

#                # OGGM WANTS THIS FUNCTION TO SIMPLY RETURN THE MASS BALANCE AS A FUNCTION OF HEIGHT AND THAT'S IT
#                mbmod = PyGEMMassBalance(gdir, modelprms, glacier_rgi_table,
#                                         hindcast=pygem_prms.hindcast,
#                                         debug=pygem_prms.debug_mb,
#                                         debug_refreeze=pygem_prms.debug_refreeze,
#                                         fls=fls, option_areaconstant=True)
#
#                # ----- MODEL RUN WITH CONSTANT GLACIER AREA -----
#                years = np.arange(pygem_prms.gcm_startyear, pygem_prms.gcm_endyear + 1)
#                mb_all = []
#                for fl_id, fl in enumerate(fls):
#                    for year in years - years[0]:
#                        mb_annual = mbmod.get_annual_mb(fls[0].surface_h, fls=fls, fl_id=fl_id, year=year,
#                                                        debug=True)
#                        mb_mwea = (mb_annual * 365 * 24 * 3600 * pygem_prms.density_ice /
#                                       pygem_prms.density_water)
#                        glac_wide_mb_mwea = ((mb_mwea * mbmod.glacier_area_initial).sum() /
#                                              mbmod.glacier_area_initial.sum())
#                        print('year:', year, np.round(glac_wide_mb_mwea,3))
#
#                        mb_all.append(glac_wide_mb_mwea)
#
#                print('iter:', n_iter, 'massbal (mean, std):', np.round(np.mean(mb_all),3), np.round(np.std(mb_all),3))


                #%%
                # ----- ICE THICKNESS INVERSION using OGGM -----
                # Perform inversion based on PyGEM MB
#                fls_inv = gdir.read_pickle('inversion_flowlines')
#                mbmod_inv = PyGEMMassBalance(gdir, modelprms, glacier_rgi_table,
#                                             hindcast=pygem_prms.hindcast,
#                                             debug=pygem_prms.debug_mb,
#                                             debug_refreeze=pygem_prms.debug_refreeze,
#                                             fls=fls_inv, option_areaconstant=True)
                mbmod_inv = PyGEMMassBalance(gdir, modelprms, glacier_rgi_table,
                                             hindcast=pygem_prms.hindcast,
                                             debug=pygem_prms.debug_mb,
                                             debug_refreeze=pygem_prms.debug_refreeze,
                                             fls=fls, option_areaconstant=True)
                h, w = gdir.get_inversion_flowline_hw()
                
                if debug:
#                    pyg  = mbmod_inv.get_annual_mb(h, year=0, fl_id=0, fls=fls_inv) * cfg.SEC_IN_YEAR * 900
                    pyg  = mbmod_inv.get_annual_mb(h, year=0, fl_id=0, fls=fls) * cfg.SEC_IN_YEAR * 900
                    plt.plot(pyg, h, '.')
                    plt.show()
                
                
                # Arbitrariliy shift the MB profile up (or down) until mass balance is zero (equilibrium for inversion)
                climate.apparent_mb_from_any_mb(gdir, mb_model=mbmod_inv, mb_years=np.arange(nyears))
                
                print('setting glacier dynamic model parameters here')
                fs = 0                      # keep this set at 0
                glen_a_multiplier = 1       # calibrate this based on ice thickness data or the consensus estimates
                
                tasks.prepare_for_inversion(gdir)
                tasks.mass_conservation_inversion(gdir, glen_a=cfg.PARAMS['glen_a']*glen_a_multiplier, fs=fs)
#                tasks.filter_inversion_output(gdir)
                tasks.init_present_time_glacier(gdir) # adds bins below
                debris.debris_binned(gdir, fl_str='model_flowlines')  # add debris enhancement factors to flowlines
                nfls = gdir.read_pickle('model_flowlines')
                
                #%%
                # ------ MODEL WITH EVOLVING AREA ------
                # Mass balance model
                mbmod = PyGEMMassBalance(gdir, modelprms, glacier_rgi_table,
                                         hindcast=pygem_prms.hindcast,
                                         debug=pygem_prms.debug_mb,
                                         debug_refreeze=pygem_prms.debug_refreeze,
                                         fls=nfls, option_areaconstant=True)
                
                # Glacier dynamics model
                if pygem_prms.option_dynamics == 'OGGM':
                    print('OGGM GLACIER DYNAMICS!')
                    ev_model = FluxBasedModel(nfls, y0=0, mb_model=mbmod, 
                                              glen_a=cfg.PARAMS['glen_a']*glen_a_multiplier, fs=fs)
                    if debug:
                        print('New glacier vol', ev_model.volume_m3)
                        graphics.plot_modeloutput_section(ev_model)
        
                    _, diag = ev_model.run_until_and_store(nyears)
                    ev_model.mb_model.glac_wide_volume_annual[-1] = diag.volume_m3[-1]
                    ev_model.mb_model.glac_wide_area_annual[-1] = diag.area_m2[-1]
                
                # Mass redistribution model                    
                elif pygem_prms.option_dynamics == 'MassRedistributionCurves':
                    print('MASS REDISTRIBUTION CURVES!')
                    ev_model = MassRedistributionCurveModel(nfls, mb_model=mbmod, y0=0)
                    
                    if debug:
                        print('New glacier vol', ev_model.volume_m3)
                        graphics.plot_modeloutput_section(ev_model)
                        
                    _, diag = ev_model.run_until_and_store(nyears)

                    
                if debug:
                    graphics.plot_modeloutput_section(ev_model)
#                    graphics.plot_modeloutput_map(gdir, model=ev_model)
                    plt.figure()
                    diag.volume_m3.plot()
                    plt.figure()
                    diag.area_m2.plot()
                    plt.show()
                        
                
                    
                # Post-process data to ensure mass is conserved and update accordingly for ignored mass losses
                #  ignored mass losses occur because mass balance model does not know ice thickness and flux divergence
                area_initial = mbmod.glac_bin_area_annual[:,0].sum()
                mb_mwea_diag = ((diag.volume_m3.values[-1] - diag.volume_m3.values[0]) 
                                / area_initial / nyears * pygem_prms.density_ice / pygem_prms.density_water)
                mb_mwea_mbmod = mbmod.glac_wide_massbaltotal.sum() / area_initial / nyears
                
                if debug:
#                    vol_change_diag = diag.volume_m3.values[-1] - diag.volume_m3.values[0]
#                    vol_change_mbmod = (mbmod.glac_wide_massbaltotal.sum() * pygem_prms.density_water / 
#                                        pygem_prms.density_ice)
#                    print('  vol change       :', vol_change_diag)
#                    print('  vol change mbmod :', vol_change_mbmod)
                    print('  mb [mwea]:', mb_mwea_diag)
#                    print('  mb_mbmod [mwea]:', mb_mwea_mbmod)
                
                #%%
                if np.abs(mb_mwea_diag - mb_mwea_mbmod) > 1e-6:
                    ev_model.mb_model.ensure_mass_conservation(diag)

                # RECORD PARAMETERS TO DATASET
                output_glac_temp_monthly[:, n_iter] = mbmod.glac_wide_temp
                output_glac_prec_monthly[:, n_iter] = mbmod.glac_wide_prec
                output_glac_acc_monthly[:, n_iter] = mbmod.glac_wide_acc
                output_glac_refreeze_monthly[:, n_iter] = mbmod.glac_wide_refreeze
                output_glac_melt_monthly[:, n_iter] = mbmod.glac_wide_melt
                output_glac_frontalablation_monthly[:, n_iter] = mbmod.glac_wide_frontalablation
                output_glac_massbaltotal_monthly[:, n_iter] = mbmod.glac_wide_massbaltotal
                output_glac_runoff_monthly[:, n_iter] = mbmod.glac_wide_runoff
                output_glac_snowline_monthly[:, n_iter] = mbmod.glac_wide_snowline
                output_glac_area_annual[:, n_iter] = diag.area_m2.values
                output_glac_volume_annual[:, n_iter] = diag.volume_m3.values
                output_glac_volume_change_ignored_annual[:-1, n_iter] = mbmod.glac_wide_volume_change_ignored_annual
                output_glac_ELA_annual[:, n_iter] = mbmod.glac_wide_ELA_annual
                output_offglac_prec_monthly[:, n_iter] = mbmod.offglac_wide_prec
                output_offglac_refreeze_monthly[:, n_iter] = mbmod.offglac_wide_refreeze
                output_offglac_melt_monthly[:, n_iter] = mbmod.offglac_wide_melt
                output_offglac_snowpack_monthly[:, n_iter] = mbmod.offglac_wide_snowpack
                output_offglac_runoff_monthly[:, n_iter] = mbmod.offglac_wide_runoff
                
                    
                    #%%                    
#                    print('volume:', diag.volume_m3.values)
#                    np.savetxt(pygem_prms.output_filepath + 'bin_mbclim.csv', 
#                               mbmod.glac_bin_massbalclim_annual, delimiter=',')
#                    
#                    nyears = dates_table.shape[0] / 12
#                    glacwide_mb_mwea_annual = mbmod.glac_bin_area_annual[:,0:-1] * mbmod.glac_bin_massbalclim_annual
#                    glacwide_mb_mwea_mean = glacwide_mb_mwea_annual.sum() / area_initial / nyears
#                    glacwide_area = mbmod.glac_bin_area_annual.sum(0)
#                    print('area from mbmod:', glacwide_area)
#                    print('area dif :', glacwide_area - diag.area_m2.values)
#                    print('mb_check_from_mbmod 2 mwea:', glacwide_mb_mwea_mean)
#                    glacwide_mb_mwea_mean = glacwide_mb_mwea_annual.sum() / nfls[0].area_m2 / nyears
#                    print('third:', glacwide_mb_mwea_mean)
                    

                #%% ===== Adding functionality for calving =====
#                water_level = None
#                if glacier_rgi_table['TermType'] == 1:
##                    # Calving params - default is 2.4, which is quite high
##                    cfg.PARAMS['inversion_calving_k'] = 1
##                    # Find out the calving values
##                    calving_output = tasks.find_inversion_calving(gdir)
##                    # Get ready
##                    tasks.init_present_time_glacier(gdir)
#
#                    #%%
#                    # FluxBasedMODEL
##                    for flux_gate in [0.06, 0.10, 0.16]:
##                        model = FluxBasedModel(bu_tidewater_bed(), mb_model=mbmod,
##                                               is_tidewater=True,
##                                               flux_gate=flux_gate,  # default is 0
##                                               calving_k=0.2,  # default is 2.4
##                                              )
#
#
##                    old_model = FluxBasedModel(fls, y0=0, mb_model=mbmod)
##                    ev_model = FluxBasedModel(nfls, y0=0, mb_model=mbmod, 
##                                              glen_a=cfg.PARAMS['glen_a']*glen_a_multiplier,fs=fs)
#
#                    # ----- MY VERSION OF FLUXBASEDMODEL W FRONTAL ABLATION -----
#                    # THIS REQUIRES ICE THICKNESS ESTIMATE!
##                    assert True == False, 'Need ice thickness inversion for this code'
##                    to_plot = None
##                    keys = []
##                    for flux_gate in [0, 0.1]:
##                        model = FluxBasedModel(fls, mb_model=mbmod, is_tidewater=True,
##                                               flux_gate=flux_gate,  # default is 0
##                                               calving_k=0.2,  # default is 2.4
##                                              )
##
##                        # long enough to reach approx. equilibrium
##                        _, ds = model.run_until_and_store(6000)
##                        df_diag = model.get_diagnostics()
##
##                        if to_plot is None:
##                            to_plot = df_diag
##
##                        key = 'Flux gate={:.02f}. Calving rate: {:.0f} m yr-1'.format(flux_gate, 
##                                         model.calving_rate_myr)
##                        to_plot[key] = df_diag['surface_h']
##                        keys.append(key)
##
##                        # Plot of volume
##                        (ds.volume_m3 * 1e-9).plot(label=key);
##                    plt.legend(); plt.ylabel('Volume [km$^{3}$]');
###                    to_plot.index = xc
#
#
#
#                    #%%
#
#                    # Calving flux (km3 yr-1; positive value refers to amount of calving, need to subtract from mb)
#                    for calving_k in [1]:
#                        # Calving parameter
#                        cfg.PARAMS['calving_k'] = calving_k
#                        cfg.PARAMS['inversion_calving_k'] = cfg.PARAMS['calving_k']
#
#
#                        # INVERSION TO RUN
#                        # Find out the calving values
#                        out_calving = tasks.find_inversion_calving(gdir)
#                        # Get ready
#                        tasks.init_present_time_glacier(gdir)
#                        # print output
#                        print(out_calving)
#
#                        if debug:
#                            fl = gdir.read_pickle('model_flowlines')[-1]
#                            xc = fl.dis_on_line * fl.dx_meter / 1000
#                            plt.plot(xc, fl.surface_h, '-', color='C1', label='Surface')
#                            plt.plot(xc, gdir.read_pickle('model_flowlines')[-1].bed_h, '--', color='k',
#                                     label='Glacier bed (with calving)')
#                            plt.hlines(0, 0, xc[-1], color='C0', linestyle=':'), plt.legend()
#                            plt.show()
#
#                        print(cfg.PARAMS['calving_front_slope'])
#
#
#
##                        assert True == False, 'Need ice thickness inversion for this code'
#                        to_plot = None
#                        keys = []
#                        fls = gdir.read_pickle('model_flowlines')
#                        for flux_gate in [0, 0.1]:
#                            model = FluxBasedModel(fls, mb_model=mbmod, is_tidewater=True,
#                                                   flux_gate=flux_gate,  # default is 0
#                                                   calving_k=0.2,  # default is 2.4
#                                                  )
#
#                            # long enough to reach approx. equilibrium
#                            _, ds = model.run_until_and_store(6000)
#                            df_diag = model.get_diagnostics()
#
#                            if to_plot is None:
#                                to_plot = df_diag
#
#                            key = 'Flux gate={:.02f}. Calving rate: {:.0f} m yr-1'.format(flux_gate, 
#                                             model.calving_rate_myr)
#                            to_plot[key] = df_diag['surface_h']
#                            keys.append(key)
#
#                            # Plot of volume
#                            (ds.volume_m3 * 1e-9).plot(label=key);
#                        plt.legend(); plt.ylabel('Volume [km$^{3}$]');
#                        plt.show()
#
##                        model_wc = tasks.run_constant_climate(gdir, y0=2000, nyears=100)
#                        # switch to tasks.run_from_climate_data
#
#
##                        # Water level (max based on 50 m freeboard)
##                        if water_level is None:
##                            th = fls[-1].surface_h[-1]
##                            vmin, vmax = cfg.PARAMS['free_board_marine_terminating']
##                            water_level = utils.clip_scalar(0, th - vmax, th - vmin)
##
##
##
##                        calving_output = calving_flux_from_depth(gdir, water_level=water_level)
##                        calving_flux = calving_output['flux']
##
##                        print('\n calving_k:', calving_k)
##                        print('  calving flux (km3 yr-1):', calving_flux)
##                        print('  freeboard:', calving_output['free_board'])
##                        print('  water_level:', calving_output['water_level'])


                #%%
            if args.option_parallels == 0 and debug:
                print('\nTO-DO LIST:')
                print(' - add option to export limited outputs:')
                print(      'area, vol, vol_ignored, mb components, runoff (on/off), SLA, ELA')
                print(' - Check if happy with 18.02342 CanESM2 RCP26, retreat then grows in an overdeepening')
                print(' - automate hemisphere based on region and glacier string')
                print(' - offglac_snowpack_monthly continously grows bc retreat at high altitudes where acc > melt')
                print(' - automate ice thickness inversion for volume to agree with consensus estimate')
                print(' - add frontal ablation to be removed in mass redistribution curves glacierdynamics')
                print(' - emulators for calibration')
                print(' - new environment without PyMC2')
                print(' - update supercomputer environment to ensure code still runs on spc')
                print('    --> get set up with Pittsburgh Supercomputing Center')
                print(' - climate data likely mismatch with OGGM, e.g., prec in m for the month')

                    
            # ===== Export Results =====
            # Create empty dataset
            output_ds_all_stats, encoding = create_xrdataset(glacier_rgi_table, dates_table)
            
            # Output statistics
            output_glac_temp_monthly_stats = calc_stats_array(output_glac_temp_monthly)
            output_glac_prec_monthly_stats = calc_stats_array(output_glac_prec_monthly)
            output_glac_acc_monthly_stats = calc_stats_array(output_glac_acc_monthly)
            output_glac_refreeze_monthly_stats = calc_stats_array(output_glac_refreeze_monthly)
            output_glac_melt_monthly_stats = calc_stats_array(output_glac_melt_monthly)
            output_glac_frontalablation_monthly_stats = calc_stats_array(output_glac_frontalablation_monthly)
            output_glac_massbaltotal_monthly_stats = calc_stats_array(output_glac_massbaltotal_monthly)
            output_glac_runoff_monthly_stats = calc_stats_array(output_glac_runoff_monthly)
            output_glac_snowline_monthly_stats = calc_stats_array(output_glac_snowline_monthly)
            output_glac_area_annual_stats = calc_stats_array(output_glac_area_annual)
            output_glac_volume_annual_stats = calc_stats_array(output_glac_volume_annual)
            output_glac_volume_change_ignored_annual_stats = calc_stats_array(output_glac_volume_change_ignored_annual)
            output_glac_ELA_annual_stats = calc_stats_array(output_glac_ELA_annual)
            output_offglac_prec_monthly_stats = calc_stats_array(output_offglac_prec_monthly)
            output_offglac_melt_monthly_stats = calc_stats_array(output_offglac_melt_monthly)
            output_offglac_refreeze_monthly_stats = calc_stats_array(output_offglac_refreeze_monthly)
            output_offglac_snowpack_monthly_stats = calc_stats_array(output_offglac_snowpack_monthly)
            output_offglac_runoff_monthly_stats = calc_stats_array(output_offglac_runoff_monthly)
            
            # Output Mean
            output_ds_all_stats['glac_temp_monthly'].values[0,:] = output_glac_temp_monthly_stats[:,0] + 273.15
            output_ds_all_stats['glac_prec_monthly'].values[0,:] = output_glac_prec_monthly_stats[:,0]
            output_ds_all_stats['glac_acc_monthly'].values[0,:] = output_glac_acc_monthly_stats[:,0]
            output_ds_all_stats['glac_refreeze_monthly'].values[0,:] = output_glac_refreeze_monthly_stats[:,0]
            output_ds_all_stats['glac_melt_monthly'].values[0,:] = output_glac_melt_monthly_stats[:,0]
            output_ds_all_stats['glac_frontalablation_monthly'].values[0,:] = (
                    output_glac_frontalablation_monthly_stats[:,0])
            output_ds_all_stats['glac_massbaltotal_monthly'].values[0,:] = (
                    output_glac_massbaltotal_monthly_stats[:,0])
            output_ds_all_stats['glac_runoff_monthly'].values[0,:] = output_glac_runoff_monthly_stats[:,0]
            output_ds_all_stats['glac_snowline_monthly'].values[0,:] = output_glac_snowline_monthly_stats[:,0]
            output_ds_all_stats['glac_area_annual'].values[0,:] = output_glac_area_annual_stats[:,0]
            output_ds_all_stats['glac_volume_annual'].values[0,:] = output_glac_volume_annual_stats[:,0]
            output_ds_all_stats['glac_volume_change_ignored_annual'].values[0,:] = (
                    output_glac_volume_change_ignored_annual_stats[:,0])
            output_ds_all_stats['glac_ELA_annual'].values[0,:] = output_glac_ELA_annual_stats[:,0]
            output_ds_all_stats['offglac_prec_monthly'].values[0,:] = output_offglac_prec_monthly_stats[:,0]
            output_ds_all_stats['offglac_melt_monthly'].values[0,:] = output_offglac_melt_monthly_stats[:,0]
            output_ds_all_stats['offglac_refreeze_monthly'].values[0,:] = output_offglac_refreeze_monthly_stats[:,0]
            output_ds_all_stats['offglac_snowpack_monthly'].values[0,:] = output_offglac_snowpack_monthly_stats[:,0]
            output_ds_all_stats['offglac_runoff_monthly'].values[0,:] = output_offglac_runoff_monthly_stats[:,0]
            
            # Output Standard Deviation
            output_ds_all_stats['glac_temp_monthly_std'].values[0,:] = output_glac_temp_monthly_stats[:,1]
            output_ds_all_stats['glac_prec_monthly_std'].values[0,:] = output_glac_prec_monthly_stats[:,1]
            output_ds_all_stats['glac_acc_monthly_std'].values[0,:] = output_glac_acc_monthly_stats[:,1]
            output_ds_all_stats['glac_refreeze_monthly_std'].values[0,:] = output_glac_refreeze_monthly_stats[:,1]
            output_ds_all_stats['glac_melt_monthly_std'].values[0,:] = output_glac_melt_monthly_stats[:,1]
            output_ds_all_stats['glac_frontalablation_monthly_std'].values[0,:] = (
                    output_glac_frontalablation_monthly_stats[:,1])
            output_ds_all_stats['glac_massbaltotal_monthly_std'].values[0,:] = (
                    output_glac_massbaltotal_monthly_stats[:,1])
            output_ds_all_stats['glac_runoff_monthly_std'].values[0,:] = output_glac_runoff_monthly_stats[:,1]
            output_ds_all_stats['glac_snowline_monthly_std'].values[0,:] = output_glac_snowline_monthly_stats[:,1]
            output_ds_all_stats['glac_area_annual_std'].values[0,:] = output_glac_area_annual_stats[:,1]
            output_ds_all_stats['glac_volume_annual_std'].values[0,:] = output_glac_volume_annual_stats[:,1]
            output_ds_all_stats['glac_volume_change_ignored_annual_std'].values[0,:] = (
                    output_glac_volume_change_ignored_annual_stats[:,1])
            output_ds_all_stats['glac_ELA_annual_std'].values[0,:] = output_glac_ELA_annual_stats[:,1]
            output_ds_all_stats['offglac_prec_monthly_std'].values[0,:] = output_offglac_prec_monthly_stats[:,1]
            output_ds_all_stats['offglac_melt_monthly_std'].values[0,:] = output_offglac_melt_monthly_stats[:,1]
            output_ds_all_stats['offglac_refreeze_monthly_std'].values[0,:] = output_offglac_refreeze_monthly_stats[:,1]
            output_ds_all_stats['offglac_snowpack_monthly_std'].values[0,:] = output_offglac_snowpack_monthly_stats[:,1]
            output_ds_all_stats['offglac_runoff_monthly_std'].values[0,:] = output_offglac_runoff_monthly_stats[:,1]
            

            # Export statistics to netcdf
            if pygem_prms.output_package == 2:
                output_sim_fp = pygem_prms.output_sim_fp + gcm_name + '/'
                if gcm_name not in ['ERA-Interim', 'ERA5', 'COAWST']:
                    output_sim_fp += rcp_scenario + '/'
                # Create filepath if it does not exist
                if os.path.exists(output_sim_fp) == False:
                    os.makedirs(output_sim_fp)
                # Netcdf filename
                if gcm_name in ['ERA-Interim', 'ERA5', 'COAWST']:
                    # Filename
                    netcdf_fn = (glacier_str + '_' + gcm_name + '_' + str(pygem_prms.option_calibration) + '_ba' +
                                 str(pygem_prms.option_bias_adjustment) + '_' +  str(sim_iters) + 'sets' + '_' +
                                 str(pygem_prms.gcm_startyear) + '_' + str(pygem_prms.gcm_endyear) + '.nc')
                else:
                    netcdf_fn = (glacier_str + '_' + gcm_name + '_' + rcp_scenario + '_' +
                                 str(pygem_prms.option_calibration) + '_ba' + str(pygem_prms.option_bias_adjustment) + 
                                 '_' + str(sim_iters) + 'sets' + '_' + str(pygem_prms.gcm_startyear) + '_' + 
                                 str(pygem_prms.gcm_endyear) + '.nc')
                if pygem_prms.option_synthetic_sim==1:
                    netcdf_fn = (netcdf_fn.split('--')[0] + '_T' + str(pygem_prms.synthetic_temp_adjust) + '_P' +
                                 str(pygem_prms.synthetic_prec_factor) + '--' + netcdf_fn.split('--')[1])
                # Export netcdf
                output_ds_all_stats.to_netcdf(output_sim_fp + netcdf_fn, encoding=encoding)

            # Close datasets
            output_ds_all_stats.close()


    # Global variables for Spyder development
    if args.option_parallels == 0:
        global main_vars
        main_vars = inspect.currentframe().f_locals


#%% PARALLEL PROCESSING
if __name__ == '__main__':
    time_start = time.time()
    parser = getparser()
    args = parser.parse_args()

    if args.debug == 1:
        debug = True
    else:
        debug = False

    if not 'pygem_modelprms' in cfg.BASENAMES:
        cfg.BASENAMES['pygem_modelprms'] = ('pygem_modelprms.pkl', 'PyGEM model parameters')

    # RGI glacier number
    if args.rgi_glac_number_fn is not None:
        with open(args.rgi_glac_number_fn, 'rb') as f:
            glac_no = pickle.load(f)
    elif pygem_prms.glac_no is not None:
        glac_no = pygem_prms.glac_no
    else:
        main_glac_rgi_all = modelsetup.selectglaciersrgitable(
                rgi_regionsO1=pygem_prms.rgi_regionsO1, rgi_regionsO2 =pygem_prms.rgi_regionsO2,
                rgi_glac_number=pygem_prms.rgi_glac_number)
        glac_no = list(main_glac_rgi_all['rgino_str'].values)

    # Number of cores for parallel processing
    if args.option_parallels != 0:
        num_cores = int(np.min([len(glac_no), args.num_simultaneous_processes]))
    else:
        num_cores = 1

    # Glacier number lists to pass for parallel processing
    glac_no_lsts = split_glaciers.split_list(glac_no, n=num_cores, option_ordered=args.option_ordered)

    # Read GCM names from argument parser
    gcm_name = args.gcm_list_fn
    if args.gcm_name is not None:
        gcm_list = [args.gcm_name]
        rcp_scenario = args.rcp
    elif args.gcm_list_fn == pygem_prms.ref_gcm_name:
        gcm_list = [pygem_prms.ref_gcm_name]
        rcp_scenario = args.rcp
    else:
        with open(args.gcm_list_fn, 'r') as gcm_fn:
            gcm_list = gcm_fn.read().splitlines()
            rcp_scenario = os.path.basename(args.gcm_list_fn).split('_')[1]
            print('Found %d gcms to process'%(len(gcm_list)))

    # Loop through all GCMs
    for gcm_name in gcm_list:
        if args.rcp is None:
            print('Processing:', gcm_name)
        else:
            print('Processing:', gcm_name, rcp_scenario)
        # Pack variables for multiprocessing
        list_packed_vars = []
        for count, glac_no_lst in enumerate(glac_no_lsts):
            list_packed_vars.append([count, glac_no_lst, gcm_name])

        # Parallel processing
        if args.option_parallels != 0:
            print('Processing in parallel with ' + str(args.num_simultaneous_processes) + ' cores...')
            with multiprocessing.Pool(args.num_simultaneous_processes) as p:
                p.map(main,list_packed_vars)
        # If not in parallel, then only should be one loop
        else:
            # Loop through the chunks and export bias adjustments
            for n in range(len(list_packed_vars)):
                main(list_packed_vars[n])



    print('Total processing time:', time.time()-time_start, 's')


#%% ===== PLOTTING AND PROCESSING FOR MODEL DEVELOPMENT =====
    # Place local variables in variable explorer
    if args.option_parallels == 0:
        main_vars_list = list(main_vars.keys())
        gcm_name = main_vars['gcm_name']
        main_glac_rgi = main_vars['main_glac_rgi']
        if pygem_prms.hyps_data in ['Huss', 'Farinotti']:
            main_glac_hyps = main_vars['main_glac_hyps']
            main_glac_icethickness = main_vars['main_glac_icethickness']
            main_glac_width = main_vars['main_glac_width']
        dates_table = main_vars['dates_table']
        if pygem_prms.option_synthetic_sim == 1:
            dates_table_synthetic = main_vars['dates_table_synthetic']
            gcm_temp_tile = main_vars['gcm_temp_tile']
            gcm_prec_tile = main_vars['gcm_prec_tile']
            gcm_lr_tile = main_vars['gcm_lr_tile']
        gcm_temp = main_vars['gcm_temp']
        gcm_tempstd = main_vars['gcm_tempstd']
        gcm_prec = main_vars['gcm_prec']
        gcm_elev = main_vars['gcm_elev']
        gcm_lr = main_vars['gcm_lr']
        gcm_temp_adj = main_vars['gcm_temp_adj']
        gcm_prec_adj = main_vars['gcm_prec_adj']
        gcm_elev_adj = main_vars['gcm_elev_adj']
        gcm_temp_lrglac = main_vars['gcm_lr']
#        output_ds_all_stats = main_vars['output_ds_all_stats']
#        modelprms = main_vars['modelprms']
        glacier_rgi_table = main_vars['glacier_rgi_table']
        glacier_str = main_vars['glacier_str']
        if pygem_prms.hyps_data in ['OGGM']:
            gdir = main_vars['gdir']
            fls = main_vars['fls']
            width_initial = fls[0].widths_m
            glacier_area_initial = width_initial * fls[0].dx
            mbmod = main_vars['mbmod']
            ev_model = main_vars['ev_model']
            diag = main_vars['diag']
            if pygem_prms.use_calibrated_modelparams:
                modelprms_dict = main_vars['modelprms_dict']