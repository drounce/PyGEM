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
import numpy as np
import xarray as xr
# Local libraries
import class_climate
import class_mbdata
import pygem.pygem_input as pygem_prms
import pygemfxns_gcmbiasadj as gcmbiasadj
import pygemfxns_massbalance as massbalance
import pygemfxns_modelsetup as modelsetup
import spc_split_glaciers as split_glaciers
from pygem.massbalance import PyGEMMassBalance
from pygem.glacierdynamics import MassRedistributionCurveModel
from pygem.oggm_compat import single_flowline_glacier_directory


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
    parser.add_argument('-option_parallels', action='store', type=int, default=0,
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


def create_xrdataset(glacier_rgi_table, dates_table, sim_iters=pygem_prms.sim_iters, stat_cns=pygem_prms.sim_stat_cns,
                     record_stats=0, option_wateryear=pygem_prms.gcm_wateryear):
    """
    Create empty xarray dataset that will be used to record simulation runs.

    Parameters
    ----------
    main_glac_rgi : pandas dataframe
        dataframe containing relevant rgi glacier information
    dates_table : pandas dataframe
        table of the dates, months, days in month, etc.
    sim_iters : int
        number of simulation runs included
    stat_cns : list
        list of strings containing statistics that will be used on simulations
    record_stats : int
        Switch to change from recording simulations to statistics

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
        annual_columns = np.unique(dates_table['wateryear'].values)[0:int(dates_table.shape[0]/12)]
        time_values = dates_table.loc[pygem_prms.spinupyears*12:dates_table.shape[0]+1,'date'].tolist()
        year_values = annual_columns[pygem_prms.spinupyears:annual_columns.shape[0]]
        year_plus1_values = np.concatenate((annual_columns[pygem_prms.spinupyears:annual_columns.shape[0]],
                                            np.array([annual_columns[annual_columns.shape[0]-1]+1])))
        # Year type for attributes
        if option_wateryear == 1:
            year_type = 'water year'
        elif option_wateryear == 2:
            year_type = 'calendar year'
        else:
            year_type = 'custom year'

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
        output_coords_dict['glac_area_annual'] = collections.OrderedDict([('glac', glac_values), 
                                                                          ('year_plus1', year_plus1_values)])
        output_coords_dict['glac_volume_annual'] = collections.OrderedDict([('glac', glac_values), 
                                                                            ('year_plus1', year_plus1_values)])
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
                                                                              ('year_plus1', year_plus1_values)])
        output_coords_dict['glac_volume_annual_std'] = collections.OrderedDict([('glac', glac_values), 
                                                                                ('year_plus1', year_plus1_values)])
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
        output_ds_all['RGIId'].values = np.array(['RGI60-' + str(int(glacier_rgi_table.loc['O1Region'])).zfill(2) + '.' + 
                                         str(int(glacier_rgi_table.loc['glacno'])).zfill(5)])
        output_ds_all['CenLon'].values = np.array([glacier_rgi_table.CenLon])
        output_ds_all['CenLat'].values = np.array([glacier_rgi_table.CenLat])
        output_ds_all['O1Region'].values = np.array([glacier_rgi_table.O1Region])
        output_ds_all['O2Region'].values = np.array([glacier_rgi_table.O2Region])
        output_ds_all['Area'].values = np.array([glacier_rgi_table.Area])
        
        output_ds.attrs = {'source': 'PyGEMv0.1.0',
                           'institution': 'University of Alaska Fairbanks, Fairbanks, AK',
                           'history': 'Created by David Rounce (drounce@alaska.edu) on ' + pygem_prms.model_run_date,
                           'references': 'doi:10.3389/feart.2019.00331 and doi:10.1017/jog.2019.91' }
        
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
    regions_str = list_packed_vars[2]
    gcm_name = list_packed_vars[3]

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
    
    # Load glacier data for Huss and Farinotti to avoid repetitively reading the csv file (not needed for OGGM)
    if pygem_prms.hyps_data in ['Huss', 'Farinotti']:
        # Glacier hypsometry [km**2], total area
        main_glac_hyps = modelsetup.import_Husstable(
                main_glac_rgi, pygem_prms.hyps_filepath, pygem_prms.hyps_filedict, pygem_prms.hyps_colsdrop)
        # Ice thickness [m], average
        main_glac_icethickness = modelsetup.import_Husstable(
                main_glac_rgi, pygem_prms.thickness_filepath, pygem_prms.thickness_filedict, 
                pygem_prms.thickness_colsdrop)
        main_glac_icethickness[main_glac_icethickness < 0] = 0
        main_glac_hyps[main_glac_icethickness == 0] = 0
        # Width [km], average
        main_glac_width = modelsetup.import_Husstable(
                main_glac_rgi, pygem_prms.width_filepath, pygem_prms.width_filedict, pygem_prms.width_colsdrop)
        
#        if pygem_prms.option_surfacetype_debris == 1:
#            main_glac_debrisfactor = modelsetup.import_Husstable(
#                    main_glac_rgi, pygem_prms.debris_fp, pygem_prms.debris_filedict, pygem_prms.debris_colsdrop)
#        else:
#            print('\n\nDELETE ME - CHECK THAT THIS IS SAME FORMAT AS MAIN_GLAC_HYPS AND OTHERS\n\n')
#            main_glac_debrisfactor = np.zeros(main_glac_hyps.shape) + 1
#        main_glac_debrisfactor[main_glac_hyps == 0] = 0
    
    # ===== TIME PERIOD =====
    dates_table = modelsetup.datesmodelrun(
            startyear=pygem_prms.gcm_startyear, endyear=pygem_prms.gcm_endyear, spinupyears=pygem_prms.gcm_spinupyears, 
            option_wateryear=pygem_prms.gcm_wateryear)

#    # =================
#    if debug:
#        # Select dates including future projections
#        #  - nospinup dates_table needed to get the proper time indices
#        dates_table_nospinup  = modelsetup.datesmodelrun(
#                startyear=pygem_prms.gcm_startyear, endyear=pygem_prms.gcm_endyear, spinupyears=0, 
#                option_wateryear=pygem_prms.gcm_wateryear)
#
#        # ===== LOAD CALIBRATION DATA =====
#        cal_data = pd.DataFrame()
#        for dataset in pygem_prms.cal_datasets:
#            cal_subset = class_mbdata.MBData(name=dataset)
#            cal_subset_data = cal_subset.retrieve_mb(main_glac_rgi, main_glac_hyps, dates_table_nospinup)
#            cal_data = cal_data.append(cal_subset_data, ignore_index=True)
#        cal_data = cal_data.sort_values(['glacno', 't1_idx'])
#        cal_data.reset_index(drop=True, inplace=True)
#    # =================

    # ===== LOAD CLIMATE DATA =====
    # Set up climate class
    if gcm_name in ['ERA5', 'ERA-Interim', 'COAWST']:
        gcm = class_climate.GCM(name=gcm_name)
        # Check that end year is reasonable
        if (pygem_prms.gcm_endyear > int(time.strftime("%Y"))) and (pygem_prms.option_synthetic_sim == 0):
            print('\n\nEND YEAR BEYOND AVAILABLE DATA FOR ERA-INTERIM. CHANGE END YEAR.\n\n')
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
                                                   spinupyears=pygem_prms.spinupyears,
                                                   option_wateryear=pygem_prms.ref_wateryear)

    # Load climate data
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

        # COAWST data has two domains, so need to merge the two domains
        if gcm_name == 'COAWST':
            gcm_temp_d01, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(gcm.temp_fn_d01, gcm.temp_vn,
                                                                             main_glac_rgi, dates_table)
            gcm_prec_d01, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(gcm.prec_fn_d01, gcm.prec_vn,
                                                                             main_glac_rgi, dates_table)
            gcm_elev_d01 = gcm.importGCMfxnearestneighbor_xarray(gcm.elev_fn_d01, gcm.elev_vn, main_glac_rgi)
            # Check if glacier outside of high-res (d02) domain
            for glac in range(main_glac_rgi.shape[0]):
                glac_lat = main_glac_rgi.loc[glac,pygem_prms.rgi_lat_colname]
                glac_lon = main_glac_rgi.loc[glac,pygem_prms.rgi_lon_colname]
                if (~(pygem_prms.coawst_d02_lat_min <= glac_lat <= pygem_prms.coawst_d02_lat_max) or
                    ~(pygem_prms.coawst_d02_lon_min <= glac_lon <= pygem_prms.coawst_d02_lon_max)):
                    gcm_prec[glac,:] = gcm_prec_d01[glac,:]
                    gcm_temp[glac,:] = gcm_temp_d01[glac,:]
                    gcm_elev[glac] = gcm_elev_d01[glac]

    # ===== Synthetic Simulation =====
    elif pygem_prms.option_synthetic_sim == 1:
        # Synthetic dates table
        dates_table_synthetic = modelsetup.datesmodelrun(
                startyear=pygem_prms.synthetic_startyear, endyear=pygem_prms.synthetic_endyear,
                option_wateryear=pygem_prms.gcm_wateryear, spinupyears=0)
        # Air temperature [degC]
        gcm_temp_tile, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(gcm.temp_fn, gcm.temp_vn, main_glac_rgi,
                                                                          dates_table_synthetic)
        # Precipitation [m]
        gcm_prec_tile, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(gcm.prec_fn, gcm.prec_vn, main_glac_rgi,
                                                                          dates_table_synthetic)
        # Elevation [m asl]
        gcm_elev = gcm.importGCMfxnearestneighbor_xarray(gcm.elev_fn, gcm.elev_vn, main_glac_rgi)
        # Lapse rate
        gcm_lr_tile, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(gcm.lr_fn, gcm.lr_vn, main_glac_rgi,
                                                                        dates_table_synthetic)
        # Future simulation based on synthetic (replicated) data; add spinup years; dataset restarts after spinupyears
        datelength = dates_table.shape[0] - pygem_prms.gcm_spinupyears * 12
        n_tiles = int(np.ceil(datelength / dates_table_synthetic.shape[0]))
        gcm_temp = np.append(gcm_temp_tile[:,:pygem_prms.gcm_spinupyears*12],
                             np.tile(gcm_temp_tile,(1,n_tiles))[:,:datelength], axis=1)
        gcm_prec = np.append(gcm_prec_tile[:,:pygem_prms.gcm_spinupyears*12],
                             np.tile(gcm_prec_tile,(1,n_tiles))[:,:datelength], axis=1)
        gcm_lr = np.append(gcm_lr_tile[:,:pygem_prms.gcm_spinupyears*12], 
                           np.tile(gcm_lr_tile,(1,n_tiles))[:,:datelength], axis=1)
        # Temperature and precipitation sensitivity adjustments
        gcm_temp = gcm_temp + pygem_prms.synthetic_temp_adjust
        gcm_prec = gcm_prec * pygem_prms.synthetic_prec_factor

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
    # Checks on precipitation data
    assert gcm_prec_adj.max() <= 10, 'gcm_prec_adj (precipitation bias adjustment) too high, needs to be modified'
    assert gcm_prec_adj.min() >= 0, 'gcm_prec_adj is producing a negative precipitation value'

    # ===== RUN MASS BALANCE =====
    # Number of simulations
    if pygem_prms.option_calibration == 2:
        sim_iters = pygem_prms.sim_iters
    else:
        sim_iters = 1

    for glac in range(main_glac_rgi.shape[0]):
        if glac == 0 or glac == main_glac_rgi.shape[0]:
            print(gcm_name,':', main_glac_rgi.loc[main_glac_rgi.index.values[glac],'RGIId'])
        # Select subsets of data
        glacier_rgi_table = main_glac_rgi.loc[main_glac_rgi.index.values[glac], :]
        glacier_str = '{0:0.5f}'.format(glacier_rgi_table['RGIId_float'])
        glacier_gcm_elev = gcm_elev_adj[glac]
        glacier_gcm_prec = gcm_prec_adj[glac,:]
        glacier_gcm_temp = gcm_temp_adj[glac,:]
        glacier_gcm_tempstd = gcm_tempstd[glac,:]
        glacier_gcm_lrgcm = gcm_lr[glac,:]
        glacier_gcm_lrglac = glacier_gcm_lrgcm.copy()
        
        # ===== Load glacier data: area (km2), ice thickness (m), width (km) =====
        if pygem_prms.hyps_data in ['oggm']:
            gd = single_flowline_glacier_directory(glacier_str)
            fls = gd.read_pickle('model_flowlines')
            # Hack to run
            icethickness_initial = np.array([1])
        elif pygem_prms.hyps_data in ['Huss', 'Farinotti']:
            glacier_area_initial = main_glac_hyps.iloc[glac,:].values.astype(float)
            icethickness_initial = main_glac_icethickness.iloc[glac,:].values.astype(float)
            width_initial = main_glac_width.iloc[glac,:].values.astype(float)
            elev_bins = main_glac_hyps.columns.values.astype(int)
            print('CONVERT AREA, THHICKNESS, WIDTH, AND ELEV BINS INTO FLOWLINES STRUCTURE')
            
#            if pygem_prms.option_surfacetype_debris == 1:
#                glacier_debrisfactor = main_glac_debrisfactor.iloc[glac,:].values.astype(float)

        # Empty datasets to record output
        annual_columns = np.unique(dates_table['wateryear'].values)[0:int(dates_table.shape[0]/12)]
        year_values = annual_columns[pygem_prms.spinupyears:annual_columns.shape[0]]
        year_plus1_values = np.concatenate((annual_columns[pygem_prms.spinupyears:annual_columns.shape[0]],
                                            np.array([annual_columns[annual_columns.shape[0]-1]+1])))
        output_temp_glac_monthly = np.zeros((dates_table.shape[0], sim_iters))
        output_prec_glac_monthly = np.zeros((dates_table.shape[0], sim_iters))
        output_acc_glac_monthly = np.zeros((dates_table.shape[0], sim_iters))
        output_refreeze_glac_monthly = np.zeros((dates_table.shape[0], sim_iters))
        output_melt_glac_monthly = np.zeros((dates_table.shape[0], sim_iters))
        output_frontalablation_glac_monthly = np.zeros((dates_table.shape[0], sim_iters))
        output_massbaltotal_glac_monthly = np.zeros((dates_table.shape[0], sim_iters))
        output_runoff_glac_monthly = np.zeros((dates_table.shape[0], sim_iters))
        output_snowline_glac_monthly = np.zeros((dates_table.shape[0], sim_iters))
        output_area_glac_annual = np.zeros((year_plus1_values.shape[0], sim_iters))
        output_volume_glac_annual = np.zeros((year_plus1_values.shape[0], sim_iters))
        output_ELA_glac_annual = np.zeros((year_values.shape[0], sim_iters))
        output_offglac_prec_monthly = np.zeros((dates_table.shape[0], sim_iters))
        output_offglac_refreeze_monthly = np.zeros((dates_table.shape[0], sim_iters))
        output_offglac_melt_monthly = np.zeros((dates_table.shape[0], sim_iters))
        output_offglac_snowpack_monthly = np.zeros((dates_table.shape[0], sim_iters))
        output_offglac_runoff_monthly = np.zeros((dates_table.shape[0], sim_iters))

#        if icethickness_initial.max() > 0:
        if fls[0].thick.max() > 0:

            if pygem_prms.hindcast == 1:
                glacier_gcm_prec = glacier_gcm_prec[::-1]
                glacier_gcm_temp = glacier_gcm_temp[::-1]
                glacier_gcm_lrgcm = glacier_gcm_lrgcm[::-1]
                glacier_gcm_lrglac = glacier_gcm_lrglac[::-1]

#            # get glacier number
#            if glacier_rgi_table.O1Region >= 10:
#                glacier_RGIId = main_glac_rgi.iloc[glac]['RGIId'][6:]
#            else:
#                glacier_RGIId = main_glac_rgi.iloc[glac]['RGIId'][7:]

            if pygem_prms.option_import_modelparams == 1:
                ds_mp = xr.open_dataset(pygem_prms.modelparams_fp + glacier_str + '.nc')
                cn_subset = pygem_prms.modelparams_colnames
                modelparameters_all = (pd.DataFrame(ds_mp['mp_value'].sel(chain=0).values,
                                                    columns=ds_mp.mp.values)[cn_subset])
            else:
                modelparameters_all = (
                        pd.DataFrame(np.asarray([
                                pygem_prms.lrgcm, pygem_prms.lrglac, pygem_prms.precfactor, pygem_prms.precgrad,
                                pygem_prms.ddfsnow, pygem_prms.ddfice, pygem_prms.tempsnow, pygem_prms.tempchange])
                                .reshape(1,-1), columns=pygem_prms.modelparams_colnames))

            # Set the number of iterations and determine every kth iteration to use for the ensemble
            if pygem_prms.option_calibration == 2 and modelparameters_all.shape[0] > 1:
                sim_iters = pygem_prms.sim_iters
                # Select every kth iteration
                mp_spacing = int((modelparameters_all.shape[0] - pygem_prms.sim_burn) / sim_iters)
                mp_idx_start = np.arange(pygem_prms.sim_burn, pygem_prms.sim_burn + mp_spacing)
                np.random.shuffle(mp_idx_start)
                mp_idx_start = mp_idx_start[0]
                mp_idx_all = np.arange(mp_idx_start, modelparameters_all.shape[0], mp_spacing)
            else:
                sim_iters = 1

            # Loop through model parameters
            for n_iter in range(sim_iters):

                if sim_iters == 1:
                    modelparameters = modelparameters_all.mean()
                else:
                    mp_idx = mp_idx_all[n_iter]
                    modelparameters = modelparameters_all.iloc[mp_idx,:]

                if debug:
                    print(glacier_str, ('PF: ' + str(np.round(modelparameters[2],2)) + ' ddfsnow: ' +
                          str(np.round(modelparameters[4],4)) + ' tbias: ' + str(np.round(modelparameters[7],2))))

#                print('\n\nDELETE ME! Switch back model parameters\n\n')
#                modelparameters[2] = 1
#                modelparameters[7] = 2
#                print('model params:', modelparameters)
                    
                if pygem_prms.hyps_data in ['oggm']:
                    # OGGM WANTS THIS FUNCITON TO SIMPLY RETURN THE MASS BALANCE AS A FUNCTION OF HEIGHT AND THAT'S IT
#                     years = np.arange(pygem_prms.gcm_startyear, pygem_prms.gcm_endyear + 1)
#                     mb_all = []
#                     for fl_id, fl in enumerate(fls):
#                         for year in years - years[0]:
#                             mb_annual = mbmod.get_annual_mb(fls[0].surface_h, fls=fls, fl_id=fl_id, year=year,
#                                                             debug=False)
# #                            print('year:', year, 'mbclim_annual_sum:', mb_annual.sum())
#                             mb_mwea = (mb_annual * 365 * 24 * 3600 * pygem_prms.density_ice /
#                                            pygem_prms.density_water)
#                             glac_wide_mb_mwea = ((mb_mwea * mbmod.glacier_area_initial).sum() /
#                                                   mbmod.glacier_area_initial.sum())
# #                            print('year:', year, np.round(glac_wide_mb_mwea,3))
#
#                             mb_all.append(glac_wide_mb_mwea)
#                     print('iter:', n_iter, 'massbal (mean, std):',
#                           np.round(np.mean(mb_all),3), np.round(np.std(mb_all),3))
#
# #                            if debug:
# #                                # Convert m ice s-1 to m w.e. a-1
# #                                mb_mwea = (mb_annual * 365 * 24 * 3600 * pygem_prms.density_ice /
# #                                           pygem_prms.density_water)
# #                                print(np.round(mb_mwea,3))

                    from oggm.core.flowline import FluxBasedModel
                    from oggm import cfg
                    from oggm.core import climate
                    from oggm import tasks, utils

                    # Make a working copy of the glacier directory
                    tmp_dir = os.path.join(cfg.PATHS['working_dir'], 'tmp_dir')
                    utils.mkdir(tmp_dir, reset=True)
                    ngd = utils.copy_to_basedir(gd, base_dir=tmp_dir, setup='all')

                    # Redo the inversion based on PyGEM MB
                    tmp_fls = ngd.read_pickle('inversion_flowlines')
                    for fl in tmp_fls:
                        # This trick is necessary because of OGGM's inversion flowlines being stupid
                        fl.thick = fl.widths * 0 + 1
                        fl.widths_m = fl.widths * gd.grid.dx
                        fl.dx_meter = fl.dx * gd.grid.dx
                    mbmod_inv = PyGEMMassBalance(modelparameters[0:8], glacier_rgi_table, glacier_gcm_temp,
                                                 glacier_gcm_tempstd, glacier_gcm_prec, glacier_gcm_elev,
                                                 glacier_gcm_lrgcm, glacier_gcm_lrglac, dates_table,
                                                 option_areaconstant=0, hindcast=pygem_prms.hindcast,
                                                 debug=pygem_prms.debug_mb,
                                                 debug_refreeze=pygem_prms.debug_refreeze,
                                                 fls=tmp_fls, repeat_period=False, option_inversion=True)
                    climate.apparent_mb_from_any_mb(ngd, mb_model=mbmod_inv, mb_years=np.arange(18))
                    tasks.prepare_for_inversion(ngd)
                    tasks.mass_conservation_inversion(ngd, glen_a=cfg.PARAMS['glen_a']*10, fs=5.7e-20) # ,
                    tasks.filter_inversion_output(ngd)
                    tasks.init_present_time_glacier(ngd)

                    nfls = ngd.read_pickle('model_flowlines')
                    mbmod = PyGEMMassBalance(modelparameters[0:8], glacier_rgi_table, glacier_gcm_temp,
                                             glacier_gcm_tempstd, glacier_gcm_prec, glacier_gcm_elev,
                                             glacier_gcm_lrgcm, glacier_gcm_lrglac, dates_table,
                                             option_areaconstant=0, hindcast=pygem_prms.hindcast,
                                             debug=pygem_prms.debug_mb,
                                             debug_refreeze=pygem_prms.debug_refreeze,
                                             fls=nfls, repeat_period=True)
                    old_model = FluxBasedModel(fls, y0=0, mb_model=mbmod)  # , glen_a=cfg.PARAMS['glen_a']*5
                    ev_model = FluxBasedModel(nfls, y0=0, mb_model=mbmod, glen_a=cfg.PARAMS['glen_a']*10, fs=5.7e-20)

                    print('Old glacier vol', old_model.volume_km3)
                    print('New glacier vol', ev_model.volume_km3)

                    from oggm import graphics
                    import matplotlib.pyplot as plt

                    graphics.plot_modeloutput_section(ev_model)
                    _, diag = ev_model.run_until_and_store(400)
                    graphics.plot_modeloutput_section(ev_model)
                    plt.figure()
                    diag.volume_m3.plot()
                    # plt.figure()
                    # diag.area_m2.plot()
                    plt.show()
#
#                    print('\nrunning mass redistribution model...')
#                    model = MassRedistributionCurveModel(fls, mb_model=mbmod, y0=0)
#                    model.run_until(pygem_prms.gcm_endyear - pygem_prms.gcm_startyear)
##                    model.run_until(1)
#                    
##                    print('\n-----')
##                    print(np.round(mbmod.glac_bin_area_annual[84:88,0:3],5))
##                    print('-----\n')
#
                    #  print('\nTO-DO LIST:')
                    #  print(' - add frontal ablation and remove in glacierdynamics')
                    #  print(' - setup flowlines for Huss and Farinotti datasets to work seemlessly')
                    #  print('     (may want to restrict or warn user if not using redistribution curves)')
                    #  print(' - add debris melt factors')
                    #  print(' - update supercomputer environment to ensure code still runs on spc')
                    #  print(' - setup code to run ice thickness inversion with PyGEM MB to avoid spinup issues')
                    #  print(' - make two refreeze (potential?) options stand-alone functions like frontal ablation')
                    #
                    # # RECORD PARAMETERS TO DATASET
                    # if pygem_prms.output_package == 2:
                    #     output_temp_glac_monthly[:, n_iter] = mbmod.glac_wide_temp
                    #     output_prec_glac_monthly[:, n_iter] = mbmod.glac_wide_prec
                    #     output_acc_glac_monthly[:, n_iter] = mbmod.glac_wide_acc
                    #     output_refreeze_glac_monthly[:, n_iter] = mbmod.glac_wide_refreeze
                    #     output_melt_glac_monthly[:, n_iter] = mbmod.glac_wide_melt
                    #     output_frontalablation_glac_monthly[:, n_iter] = mbmod.glac_wide_frontalablation
                    #     output_massbaltotal_glac_monthly[:, n_iter] = mbmod.glac_wide_massbaltotal
                    #     output_runoff_glac_monthly[:, n_iter] = mbmod.glac_wide_runoff
                    #     output_snowline_glac_monthly[:, n_iter] = mbmod.glac_wide_snowline
                    #     output_area_glac_annual[:, n_iter] = mbmod.glac_wide_area_annual
                    #     output_volume_glac_annual[:, n_iter] = mbmod.glac_wide_volume_annual
                    #     output_ELA_glac_annual[:, n_iter] = mbmod.glac_wide_ELA_annual
                    #     output_offglac_prec_monthly[:, n_iter] = mbmod.offglac_wide_prec
                    #     output_offglac_refreeze_monthly[:, n_iter] = mbmod.offglac_wide_refreeze
                    #     output_offglac_melt_monthly[:, n_iter] = mbmod.offglac_wide_melt
                    #     output_offglac_snowpack_monthly[:, n_iter] = mbmod.offglac_wide_snowpack
                    #     output_offglac_runoff_monthly[:, n_iter] = mbmod.offglac_wide_runoff

            # ===== Export Results =====                    
            output_ds, encoding = create_xrdataset(glacier_rgi_table, dates_table, record_stats=1,
                                                       option_wateryear=pygem_prms.gcm_wateryear)
            output_ds['glac_temp_monthly'].values = output_temp_glac_monthly.mean(1)[np.newaxis,:] + 273.15
            output_ds['glac_temp_monthly_std'].values = output_temp_glac_monthly.std(1)[np.newaxis,:]
            output_ds['glac_prec_monthly'].values = output_prec_glac_monthly.mean(1)[np.newaxis,:]
            output_ds['glac_prec_monthly_std'].values = output_prec_glac_monthly.std(1)[np.newaxis,:]
            output_ds['glac_acc_monthly'].values = output_acc_glac_monthly.mean(1)[np.newaxis,:]
            output_ds['glac_acc_monthly_std'].values = output_acc_glac_monthly.std(1)[np.newaxis,:]
            output_ds['glac_refreeze_monthly'].values = output_refreeze_glac_monthly.mean(1)[np.newaxis,:]
            output_ds['glac_refreeze_monthly_std'].values = output_refreeze_glac_monthly.std(1)[np.newaxis,:]
            output_ds['glac_melt_monthly'].values = output_melt_glac_monthly.mean(1)[np.newaxis,:]
            output_ds['glac_melt_monthly_std'].values = output_melt_glac_monthly.std(1)[np.newaxis,:]
            output_ds['glac_frontalablation_monthly'].values = output_frontalablation_glac_monthly.mean(1)[np.newaxis,:]
            output_ds['glac_frontalablation_monthly_std'].values = (
                    output_frontalablation_glac_monthly.std(1)[np.newaxis,:])
            output_ds['glac_massbaltotal_monthly'].values = output_massbaltotal_glac_monthly.mean(1)[np.newaxis,:]
            output_ds['glac_massbaltotal_monthly_std'].values = output_massbaltotal_glac_monthly.std(1)[np.newaxis,:]
            output_ds['glac_runoff_monthly'].values = output_runoff_glac_monthly.mean(1)[np.newaxis,:]
            output_ds['glac_runoff_monthly_std'].values = output_runoff_glac_monthly.std(1)[np.newaxis,:]
            output_ds['glac_snowline_monthly'].values = output_snowline_glac_monthly.mean(1)[np.newaxis,:]
            output_ds['glac_snowline_monthly_std'].values = output_snowline_glac_monthly.std(1)[np.newaxis,:]
            output_ds['glac_area_annual'].values = output_area_glac_annual.mean(1)[np.newaxis,:]
            output_ds['glac_area_annual_std'].values = output_area_glac_annual.std(1)[np.newaxis,:]
            output_ds['glac_volume_annual'].values = output_volume_glac_annual.mean(1)[np.newaxis,:]
            output_ds['glac_volume_annual_std'].values = output_volume_glac_annual.std(1)[np.newaxis,:]
            output_ds['glac_ELA_annual'].values = output_ELA_glac_annual.mean(1)[np.newaxis,:]
            output_ds['glac_ELA_annual_std'].values = output_ELA_glac_annual.std(1)[np.newaxis,:]
            output_ds['offglac_prec_monthly'].values = output_offglac_prec_monthly.mean(1)[np.newaxis,:]
            output_ds['offglac_prec_monthly_std'].values = output_offglac_prec_monthly.std(1)[np.newaxis,:]
            output_ds['offglac_refreeze_monthly'].values = output_offglac_refreeze_monthly.mean(1)[np.newaxis,:]
            output_ds['offglac_refreeze_monthly_std'].values = output_offglac_refreeze_monthly.std(1)[np.newaxis,:]
            output_ds['offglac_melt_monthly'].values = output_offglac_melt_monthly.mean(1)[np.newaxis,:]
            output_ds['offglac_melt_monthly_std'].values = output_offglac_melt_monthly.std(1)[np.newaxis,:]
            output_ds['offglac_snowpack_monthly'].values =  output_offglac_snowpack_monthly.mean(1)[np.newaxis,:]
            output_ds['offglac_snowpack_monthly_std'].values =  output_offglac_snowpack_monthly.std(1)[np.newaxis,:]
            output_ds['offglac_runoff_monthly'].values = output_offglac_runoff_monthly.mean(1)[np.newaxis,:]
            output_ds['offglac_runoff_monthly_std'].values = output_offglac_runoff_monthly.std(1)[np.newaxis,:]                    

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
                    netcdf_fn = (glacier_str + '_' + gcm_name + '_c' + str(pygem_prms.option_calibration) + '_ba' +
                                 str(pygem_prms.option_bias_adjustment) + '_' +  str(sim_iters) + 'sets' + '_' +
                                 str(pygem_prms.gcm_startyear) + '_' + str(pygem_prms.gcm_endyear) + '.nc')
                else:
                    netcdf_fn = (glacier_str + '_' + gcm_name + '_' + rcp_scenario + '_c' +
                                 str(pygem_prms.option_calibration) + '_ba' + str(pygem_prms.option_bias_adjustment) + 
                                 '_' +  str(sim_iters) + 'sets' + '_' + str(pygem_prms.gcm_startyear) + '_' + 
                                 str(pygem_prms.gcm_endyear) + '.nc')
                if pygem_prms.option_synthetic_sim==1:
                    netcdf_fn = (netcdf_fn.split('--')[0] + '_T' + str(pygem_prms.synthetic_temp_adjust) + '_P' +
                                 str(pygem_prms.synthetic_prec_factor) + '--' + netcdf_fn.split('--')[1])
                    
                # Export netcdf
                output_ds.to_netcdf(output_sim_fp + netcdf_fn, encoding=encoding)

            # Close datasets
            output_ds.close()


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

    # Regions
    regions_str = 'R'
    for region in sorted(set([x.split('.')[0] for x in glac_no])):
        regions_str += str(region)

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
            list_packed_vars.append([count, glac_no_lst, regions_str, gcm_name])

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
#        modelparameters = main_vars['modelparameters']
        glacier_rgi_table = main_vars['glacier_rgi_table']
        glacier_str = main_vars['glacier_str']
        if pygem_prms.hyps_data in ['oggm']:
            fls = main_vars['fls']
            elev_bins = fls[0].surface_h
            icethickness_initial = fls[0].thick
            width_initial = fls[0].widths_m / 1000
            glacier_area_initial = width_initial * fls[0].dx / 1000       
            mbmod = main_vars['mbmod']
#            model = main_vars['model']
        glacier_gcm_temp = main_vars['glacier_gcm_temp']
        glacier_gcm_tempstd = main_vars['glacier_gcm_tempstd']
        glacier_gcm_prec = main_vars['glacier_gcm_prec']
        glacier_gcm_elev = main_vars['glacier_gcm_elev']
        glacier_gcm_lrgcm = main_vars['glacier_gcm_lrgcm']
        glacier_gcm_lrglac = glacier_gcm_lrgcm
#        glac_bin_frontalablation = main_vars['glac_bin_frontalablation']
#        glac_bin_area_annual = main_vars['glac_bin_area_annual']
#        glac_bin_massbalclim_annual = main_vars['glac_bin_massbalclim_annual']
#        glac_bin_melt = main_vars['glac_bin_melt']
#        glac_bin_acc = main_vars['glac_bin_acc']
#        glac_bin_refreeze = main_vars['glac_bin_refreeze']
##        glac_bin_snowpack = main_vars['glac_bin_snowpack']
#        glac_bin_temp = main_vars['glac_bin_temp']
#        glac_bin_prec = main_vars['glac_bin_prec']
#        glac_bin_massbalclim = main_vars['glac_bin_massbalclim']
##        glac_wide_massbaltotal = main_vars['glac_wide_massbaltotal']
##        glac_wide_area_annual = main_vars['glac_wide_area_annual']
##        glac_wide_volume_annual = main_vars['glac_wide_volume_annual']
#        glac_wide_runoff = main_vars['glac_wide_runoff']
###        glac_wide_prec = main_vars['glac_wide_prec']
###        glac_wide_refreeze = main_vars['glac_wide_refreeze']
##        modelparameters_all = main_vars['modelparameters_all']
##        sim_iters = main_vars['sim_iters']
