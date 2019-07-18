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
import resource
import time
# External libraries
import pandas as pd
import pickle
import numpy as np
import xarray as xr
# Local libraries
import class_climate
import class_mbdata
import pygem_input as input
import pygemfxns_gcmbiasadj as gcmbiasadj
import pygemfxns_massbalance as massbalance
import pygemfxns_modelsetup as modelsetup


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
    spc_region (optional) : str
        RGI region number for supercomputer 
    rgi_glac_number_fn (optional) : str
        filename of .pkl file containing a list of glacier numbers that used to run batches on the supercomputer
    batch_number (optional): int
        batch number used to differentiate output on supercomputer
    debug (optional) : int
        Switch for turning debug printing on or off (default = 0 (off))
      
        
    Returns
    -------
    Object containing arguments and their respective values.
    """
    parser = argparse.ArgumentParser(description="run simulations from gcm list in parallel")
    # add arguments
    parser.add_argument('-gcm_list_fn', action='store', type=str, default=input.ref_gcm_name,
                        help='text file full of commands to run')
    parser.add_argument('-gcm_name', action='store', type=str, default=None,
                        help='GCM name used for model run')
    parser.add_argument('-rcp', action='store', type=str, default=None,
                        help='rcp scenario used for model run (ex. rcp26)')
    parser.add_argument('-num_simultaneous_processes', action='store', type=int, default=4,
                        help='number of simultaneous processes (cores) to use')
    parser.add_argument('-option_parallels', action='store', type=int, default=1,
                        help='Switch to use or not use parallels (1 - use parallels, 0 - do not)')
    parser.add_argument('-spc_region', action='store', type=int, default=None,
                        help='rgi region number for supercomputer')
    parser.add_argument('-rgi_glac_number_fn', action='store', type=str, default=None,
                        help='Filename containing list of rgi_glac_number, helpful for running batches on spc')
    parser.add_argument('-batch_number', action='store', type=int, default=None,
                        help='Batch number used to differentiate output on supercomputer')
    parser.add_argument('-debug', action='store', type=int, default=0,
                        help='Boolean for debugging to turn it on or off (default 0 is off')
    return parser


def calc_stats(vn, ds, stats_cns=input.sim_stat_cns, glac=0):
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
    data = ds[vn].values[glac,:,:]
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


def create_xrdataset(main_glac_rgi, dates_table, sim_iters=input.sim_iters, stat_cns=input.sim_stat_cns, 
                     record_stats=0, option_wateryear=input.gcm_wateryear):
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
    if input.output_package == 2:
        # Create empty datasets for each variable and merge them
        # Coordinate values
        output_variables = input.output_variables_package2
        glac_values = main_glac_rgi.index.values
        annual_columns = np.unique(dates_table['wateryear'].values)[0:int(dates_table.shape[0]/12)]
        time_values = dates_table.loc[input.spinupyears*12:dates_table.shape[0]+1,'date'].tolist()
        year_values = annual_columns[input.spinupyears:annual_columns.shape[0]]
        year_plus1_values = np.concatenate((annual_columns[input.spinupyears:annual_columns.shape[0]], 
                                            np.array([annual_columns[annual_columns.shape[0]-1]+1])))
        # Year type for attributes
        if option_wateryear == 1:
            year_type = 'water year'
        elif option_wateryear == 2:
            year_type = 'calendar year'
        else:
            year_type = 'custom year'
            
        # Switch to record simulations or statistics
        if record_stats == 0:
            record_name = 'sim'
            record_name_values = np.arange(0,sim_iters)
        elif record_stats == 1:
            record_name = 'stats'
            record_name_values = input.sim_stat_cns
        
        # Variable coordinates dictionary
        output_coords_dict = {
                'prec_glac_monthly': collections.OrderedDict(
                        [('glac', glac_values), ('time', time_values), (record_name, record_name_values)]),
                'temp_glac_monthly': collections.OrderedDict(
                        [('glac', glac_values), ('time', time_values), (record_name, record_name_values)]),
                'acc_glac_monthly': collections.OrderedDict(
                        [('glac', glac_values), ('time', time_values), (record_name, record_name_values)]),
                'refreeze_glac_monthly': collections.OrderedDict(
                        [('glac', glac_values), ('time', time_values), (record_name, record_name_values)]),
                'melt_glac_monthly': collections.OrderedDict(
                        [('glac', glac_values), ('time', time_values), (record_name, record_name_values)]),
                'frontalablation_glac_monthly': collections.OrderedDict(
                        [('glac', glac_values), ('time', time_values), (record_name, record_name_values)]), 
                'massbaltotal_glac_monthly': collections.OrderedDict(
                        [('glac', glac_values), ('time', time_values), (record_name, record_name_values)]),
                'runoff_glac_monthly': collections.OrderedDict(
                        [('glac', glac_values), ('time', time_values), (record_name, record_name_values)]), 
                'snowline_glac_monthly': collections.OrderedDict(
                        [('glac', glac_values), ('time', time_values), (record_name, record_name_values)]),
                'area_glac_annual': collections.OrderedDict(
                        [('glac', glac_values), ('year_plus1', year_plus1_values), (record_name, record_name_values)]),
                'volume_glac_annual': collections.OrderedDict(
                        [('glac', glac_values), ('year_plus1', year_plus1_values), (record_name, record_name_values)]),
                'ELA_glac_annual': collections.OrderedDict(
                        [('glac', glac_values), ('year', year_values), (record_name, record_name_values)]),
                'offglac_prec_monthly': collections.OrderedDict(
                        [('glac', glac_values), ('time', time_values), (record_name, record_name_values)]),
                'offglac_refreeze_monthly': collections.OrderedDict(
                        [('glac', glac_values), ('time', time_values), (record_name, record_name_values)]),
                'offglac_melt_monthly': collections.OrderedDict(
                        [('glac', glac_values), ('time', time_values), (record_name, record_name_values)]),
                'offglac_snowpack_monthly': collections.OrderedDict(
                        [('glac', glac_values), ('time', time_values), (record_name, record_name_values)]),
                'offglac_runoff_monthly': collections.OrderedDict(
                        [('glac', glac_values), ('time', time_values), (record_name, record_name_values)]), 
                }
        # Attributes dictionary
        output_attrs_dict = {
                'time': {
                        'long_name': 'date',
                         'year_type':year_type},
                'glac': {
                        'long_name': 'glacier index',
                         'comment': 'glacier index value that refers to the glacier table'},
                'year': {
                        'long_name': 'years',
                         'year_type': year_type,
                         'comment': 'years referring to the start of each year'},
                'year_plus1': {
                        'long_name': 'years plus one additional year',
                        'year_type': year_type,
                        'comment': ('additional year allows one to record glacier dimension changes at end of '
                                    'model run')},   
                'sim': {
                        'long_name': 'simulation number',
                        'comment': 'simulation numbers only needed for MCMC methods'},
                'stats': {
                        'long_name': 'variable statistics',
                        'comment': '% refers to percentiles'},
                'temp_glac_monthly': {
                        'long_name': 'glacier-wide mean air temperature',
                        'units': 'degC',
                        'temporal_resolution': 'monthly',
                        'comment': (
                                'each elevation bin is weighted equally to compute the mean temperature, and '
                                'bins where the glacier no longer exists due to retreat have been removed')},
                'prec_glac_monthly': {
                        'long_name': 'glacier-wide precipitation (liquid)',
                        'units': 'm',
                        'temporal_resolution': 'monthly',
                        'comment': 'only the liquid precipitation, solid precipitation excluded'},
                'acc_glac_monthly': {
                        'long_name': 'glacier-wide accumulation',
                        'units': 'm w.e.',
                        'temporal_resolution': 'monthly',
                        'comment': 'only the solid precipitation'},
                'refreeze_glac_monthly': {
                        'long_name': 'glacier-wide refreeze',
                        'units': 'm w.e.',
                        'temporal_resolution': 'monthly'},
                'melt_glac_monthly': {
                        'long_name': 'glacier-wide melt',
                        'units': 'm w.e.',
                        'temporal_resolution': 'monthly'},
                'frontalablation_glac_monthly': {
                        'long_name': 'glacier-wide frontal ablation',
                        'units': 'm w.e.',
                        'temporal_resolution': 'monthly',
                        'comment': (
                                'mass losses from calving, subaerial frontal melting, sublimation above the '
                                'waterline and subaqueous frontal melting below the waterline')},
                'massbaltotal_glac_monthly': {
                        'long_name': 'glacier-wide total mass balance',
                        'units': 'm w.e.',
                        'temporal_resolution': 'monthly',
                        'comment': (
                                'total mass balance is the sum of the climatic mass balance and frontal '
                                'ablation')},
                'runoff_glac_monthly': {
                        'long_name': 'glacier-wide runoff',
                        'units': 'm**3',
                        'temporal_resolution': 'monthly',
                        'comment': 'runoff from the glacier terminus, which moves over time'},
                'snowline_glac_monthly': {
                        'long_name': 'transient snowline',
                        'units': 'm a.s.l.',
                        'temporal_resolution': 'monthly',
                        'comment': 'transient snowline is altitude separating snow from ice/firn'},
                'area_glac_annual': {
                        'long_name': 'glacier area',
                        'units': 'km**2',
                        'temporal_resolution': 'annual',
                        'comment': 'area used for the duration of the defined start/end of year'},
                'volume_glac_annual': {
                        'long_name': 'glacier volume',
                        'units': 'km**3 ice',
                        'temporal_resolution': 'annual',
                        'comment': 'volume based on area and ice thickness used for that year'}, 
                'ELA_glac_annual': {
                        'long_name': 'annual equilibrium line altitude',
                        'units': 'm a.s.l.',
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
                        'long_name': 'off-glacier-wide refreeze',
                        'units': 'm w.e.',
                        'temporal_resolution': 'monthly'},
                'offglac_melt_monthly': {
                        'long_name': 'off-glacier-wide melt',
                        'units': 'm w.e.',
                        'temporal_resolution': 'monthly',
                        'comment': 'only melt of snow and refreeze since off-glacier'},
                'offglac_runoff_monthly': {
                        'long_name': 'off-glacier-wide runoff',
                        'units': 'm**3',
                        'temporal_resolution': 'monthly',
                        'comment': 'off-glacier runoff from area where glacier no longer exists'},
                'offglac_snowpack_monthly': {
                        'long_name': 'off-glacier-wide snowpack',
                        'units': 'm w.e.',
                        'temporal_resolution': 'monthly',
                        'comment': 'snow remaining accounting for new accumulation, melt, and refreeze'},
                }
                
        # Add variables to empty dataset and merge together
        count_vn = 0
        encoding = {}
        noencoding_vn = ['stats', 'glac_attrs']
        for vn in output_variables:
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
        main_glac_rgi_float = main_glac_rgi[input.output_glacier_attr_vns].copy()
        main_glac_rgi_xr = xr.Dataset({'glacier_table': (('glac', 'glac_attrs'), main_glac_rgi_float.values)},
                                       coords={'glac': glac_values,
                                               'glac_attrs': main_glac_rgi_float.columns.values})
        output_ds_all = output_ds_all.combine_first(main_glac_rgi_xr)
        output_ds_all.glacier_table.attrs['long_name'] = 'RGI glacier table'
        output_ds_all.glacier_table.attrs['comment'] = 'table contains attributes from RGI for each glacier'
        output_ds_all.glac_attrs.attrs['long_name'] = 'RGI glacier attributes'
        # Add attributes
        for vn in output_ds_all.variables:
            try:
                output_ds_all[vn].attrs = output_attrs_dict[vn]
            except:
                pass
            # Encoding (specify _FillValue, offsets, etc.)
            if vn not in noencoding_vn:
                encoding[vn] = {'_FillValue': False}
    return output_ds_all, encoding


def convert_glacwide_results(elev_bins, glac_bin_temp, glac_bin_prec, glac_bin_acc, glac_bin_refreeze, 
                             glac_bin_snowpack, glac_bin_melt, glac_bin_frontalablation, glac_bin_massbalclim_annual, 
                             glac_bin_area_annual, glac_bin_icethickness_annual):
    """
    Convert raw runmassbalance function output to glacier-wide results for output package 2
    
    Parameters
    ----------
    elev_bins : numpy array
        elevation of each elevation bin
    glac_bin_temp : numpy array
        temperature for each elevation bin for each timestep
    glac_bin_prec : numpy array
        precipitation (liquid) for each elevation bin for each timestep
    glac_bin_acc : numpy array
        accumulation (solid precipitation) for each elevation bin for each timestep
    glac_bin_refreeze : numpy array
        refreeze for each elevation bin for each timestep
    glac_bin_snowpack : numpy array
        snowpack for each elevation bin for each timestep
    glac_bin_melt : numpy array
        glacier melt for each elevation bin for each timestep
    glac_bin_frontalablation : numpy array
        frontal ablation for each elevation bin for each timestep
    glac_bin_massbalclim_annual : numpy array
        annual climatic mass balance for each elevation bin for each timestep   
    glac_bin_area_annual : numpy array
        annual glacier area for each elevation bin for each timestep
    glac_bin_icethickness_annual: numpy array
        annual ice thickness for each elevation bin for each timestep
     
    Returns
    -------
    glac_wide_temp : np.array
        monthly mean glacier-wide temperature (bins weighted equally)
    glac_wide_prec : np.array
        monthly glacier-wide precipitation (liquid only)
    glac_wide_acc : np.array
        monthly glacier-wide accumulation (solid precipitation only)
    glac_wide_refreeze : np.array
        monthly glacier-wide refreeze
    glac_wide_melt : np.array
        monthly glacier-wide melt
    glac_wide_frontalablation : np.array
        monthly glacier-wide frontal ablation
    glac_wide_massbaltotal : np.array
        monthly glacier-wide total mass balance (climatic mass balance + frontal ablation)
    glac_wide_runoff: np.array
        monthly glacier-wide runoff at the terminus of the glacier
    glac_wide_snowline : np.array
        monthly glacier-wide snowline
    glac_wide_area_annual : np.array
        annual glacier area
    glac_wide_volume_annual : np.array
        annual glacier volume
    glac_wide_ELA_annual : np.array
        annual equilibrium line altitude
    """
    # Preset desired output (needed to avoid dividing by zero)
    glac_wide_temp = np.zeros(glac_bin_temp.shape[1])
    glac_wide_prec = np.zeros(glac_bin_temp.shape[1])
    glac_wide_acc = np.zeros(glac_bin_temp.shape[1])
    glac_wide_refreeze = np.zeros(glac_bin_temp.shape[1])
    glac_wide_melt = np.zeros(glac_bin_temp.shape[1])
    glac_wide_frontalablation = np.zeros(glac_bin_temp.shape[1])
    # Compute desired output
    glac_bin_area = glac_bin_area_annual[:,0:glac_bin_area_annual.shape[1]-1].repeat(12,axis=1)
    glac_wide_area = glac_bin_area.sum(axis=0)
    glac_wide_temp_sum = glac_bin_temp.sum(axis=0)
    glac_bin_temp_nonzero = np.zeros(glac_bin_temp.shape)
    glac_bin_temp_nonzero[glac_bin_temp != 0] = 1
    glac_wide_temp_bincount = glac_bin_temp_nonzero.sum(axis=0)
    glac_wide_temp[glac_wide_temp_bincount > 0] = (glac_wide_temp_sum[glac_wide_temp_bincount > 0] / 
                                                   glac_wide_temp_bincount[glac_wide_temp_bincount > 0])
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
    # ELA and snowline can't be below minimum elevation
    glac_zmin_annual = elev_bins[(glac_bin_area_annual > 0).argmax(axis=0)][:-1] - input.binsize/2
    glac_wide_ELA_annual[glac_wide_ELA_annual < glac_zmin_annual] = (
            glac_zmin_annual[glac_wide_ELA_annual < glac_zmin_annual])
    glac_zmin = elev_bins[(glac_bin_area > 0).argmax(axis=0)] - input.binsize/2
    glac_wide_snowline[glac_wide_snowline < glac_zmin] = glac_zmin[glac_wide_snowline < glac_zmin]
    
    return (glac_wide_temp, glac_wide_prec, glac_wide_acc, glac_wide_refreeze, glac_wide_melt, 
            glac_wide_frontalablation, glac_wide_massbaltotal, glac_wide_runoff, glac_wide_snowline, 
            glac_wide_area_annual, glac_wide_volume_annual, glac_wide_ELA_annual)


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
    chunk = list_packed_vars[1]
    main_glac_rgi_all = list_packed_vars[2]
    chunk_size = list_packed_vars[3]
    gcm_name = list_packed_vars[4]
#    rgi_glac_number = list_packed_vars[0]
#    gcm_name = list_packed_vars[1]
#    count = list_packed_vars[2]

    parser = getparser()
    args = parser.parse_args()
        
    if (gcm_name != input.ref_gcm_name) and (args.rcp is None):
        rcp_scenario = os.path.basename(args.gcm_list_fn).split('_')[1]
    elif args.rcp is not None:
        rcp_scenario = args.rcp
        
    # RGI region
    if args.spc_region is not None:
        rgi_regionsO1 = [int(args.spc_region)]
    else:
        rgi_regionsO1 = input.rgi_regionsO1
    
    if debug:
        if 'rcp_scenario' in locals():
            print(rcp_scenario)

    # ===== LOAD GLACIER DATA =====
    main_glac_rgi = main_glac_rgi_all.iloc[chunk:chunk + chunk_size, :].copy()
#    main_glac_rgi = modelsetup.selectglaciersrgitable(rgi_regionsO1=rgi_regionsO1, rgi_regionsO2 = 'all',
#                                                      rgi_glac_number=rgi_glac_number)
    # Glacier hypsometry [km**2], total area
    main_glac_hyps = modelsetup.import_Husstable(main_glac_rgi, rgi_regionsO1, input.hyps_filepath,
                                                 input.hyps_filedict, input.hyps_colsdrop)
    # Ice thickness [m], average
    main_glac_icethickness = modelsetup.import_Husstable(main_glac_rgi, rgi_regionsO1, input.thickness_filepath,
                                                         input.thickness_filedict, input.thickness_colsdrop)
    main_glac_hyps[main_glac_icethickness == 0] = 0
    # Width [km], average
    main_glac_width = modelsetup.import_Husstable(main_glac_rgi, rgi_regionsO1, input.width_filepath,
                                                  input.width_filedict, input.width_colsdrop)
    elev_bins = main_glac_hyps.columns.values.astype(int)
    # Volume [km**3] and mean elevation [m a.s.l.]
    main_glac_rgi['Volume'], main_glac_rgi['Zmean'] = modelsetup.hypsometrystats(main_glac_hyps, main_glac_icethickness)
    
    # Select dates including future projections
    dates_table = modelsetup.datesmodelrun(startyear=input.gcm_startyear, endyear=input.gcm_endyear, 
                                           spinupyears=input.gcm_spinupyears, option_wateryear=input.gcm_wateryear)
    
    # =================
    if debug:
        # Select dates including future projections
        #  - nospinup dates_table needed to get the proper time indices
        dates_table_nospinup  = modelsetup.datesmodelrun(startyear=input.gcm_startyear, endyear=input.gcm_endyear, 
                                                         spinupyears=0, option_wateryear=input.gcm_wateryear)
    
        # ===== LOAD CALIBRATION DATA =====
        cal_data = pd.DataFrame()
        for dataset in input.cal_datasets:
            cal_subset = class_mbdata.MBData(name=dataset, rgi_regionO1=rgi_regionsO1[0])
            cal_subset_data = cal_subset.retrieve_mb(main_glac_rgi, main_glac_hyps, dates_table_nospinup)
            cal_data = cal_data.append(cal_subset_data, ignore_index=True)
        cal_data = cal_data.sort_values(['glacno', 't1_idx'])
        cal_data.reset_index(drop=True, inplace=True)
    
    # =================
    
    
    # Synthetic simulation dates
    if input.option_synthetic_sim == 1:
        dates_table_synthetic = modelsetup.datesmodelrun(
                startyear=input.synthetic_startyear, endyear=input.synthetic_endyear, 
                option_wateryear=input.gcm_wateryear, spinupyears=0)
        
    # ===== LOAD CLIMATE DATA =====
    if gcm_name == 'ERA-Interim' or gcm_name == 'COAWST':
        gcm = class_climate.GCM(name=gcm_name)
        # Check that end year is reasonable
        if (input.gcm_endyear > int(time.strftime("%Y"))) and (input.option_synthetic_sim == 0):
            print('\n\nEND YEAR BEYOND AVAILABLE DATA FOR ERA-INTERIM. CHANGE END YEAR.\n\n')
    else:
        gcm = class_climate.GCM(name=gcm_name, rcp_scenario=rcp_scenario)
    
    if input.option_synthetic_sim == 0:        
        # Air temperature [degC]
        gcm_temp, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(gcm.temp_fn, gcm.temp_vn, main_glac_rgi, 
                                                                     dates_table)
        # Precipitation [m]
        gcm_prec, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(gcm.prec_fn, gcm.prec_vn, main_glac_rgi, 
                                                                     dates_table)
        # Elevation [m asl]
        gcm_elev = gcm.importGCMfxnearestneighbor_xarray(gcm.elev_fn, gcm.elev_vn, main_glac_rgi)          
        # Lapse rate
        if gcm_name == 'ERA-Interim':
            gcm_lr, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(gcm.lr_fn, gcm.lr_vn, main_glac_rgi, dates_table)
        else:
            # Compute lapse rates based on reference climate data
            # Adjust reference dates in event that reference is longer than GCM data
            if input.startyear >= input.gcm_startyear:
                ref_startyear = input.startyear
            else:
                ref_startyear = input.gcm_startyear
            if input.endyear <= input.gcm_endyear:
                ref_endyear = input.endyear
            else:
                ref_endyear = input.gcm_endyear
            dates_table_ref = modelsetup.datesmodelrun(startyear=ref_startyear, endyear=ref_endyear, 
                                                       spinupyears=input.spinupyears, 
                                                       option_wateryear=input.option_wateryear)
            # Monthly average from reference climate data
            ref_gcm = class_climate.GCM(name=input.ref_gcm_name)
            if debug:
                print(ref_startyear, ref_endyear)
            ref_lr, ref_dates = ref_gcm.importGCMvarnearestneighbor_xarray(ref_gcm.lr_fn, ref_gcm.lr_vn, main_glac_rgi, 
                                                                           dates_table_ref)
            ref_lr_monthly_avg = gcmbiasadj.monthly_avg_2darray(ref_lr)
#            ref_lr_monthly_avg = np.roll(ref_lr_monthly_avg, -4)
#            gcm_lr = np.tile(ref_lr_monthly_avg, int(gcm_temp.shape[1]/12))
            gcm_lr = gcmbiasadj.monthly_lr_rolled(ref_lr, dates_table_ref, dates_table)
               
        # COAWST data has two domains, so need to merge the two domains
        if gcm_name == 'COAWST':
            gcm_temp_d01, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(gcm.temp_fn_d01, gcm.temp_vn,
                                                                             main_glac_rgi, dates_table)
            gcm_prec_d01, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(gcm.prec_fn_d01, gcm.prec_vn, 
                                                                             main_glac_rgi, dates_table)
            gcm_elev_d01 = gcm.importGCMfxnearestneighbor_xarray(gcm.elev_fn_d01, gcm.elev_vn, main_glac_rgi)
            # Check if glacier outside of high-res (d02) domain
            for glac in range(main_glac_rgi.shape[0]):
                glac_lat = main_glac_rgi.loc[glac,input.rgi_lat_colname]
                glac_lon = main_glac_rgi.loc[glac,input.rgi_lon_colname]
                if (~(input.coawst_d02_lat_min <= glac_lat <= input.coawst_d02_lat_max) or 
                    ~(input.coawst_d02_lon_min <= glac_lon <= input.coawst_d02_lon_max)):
                    gcm_prec[glac,:] = gcm_prec_d01[glac,:]
                    gcm_temp[glac,:] = gcm_temp_d01[glac,:]
                    gcm_elev[glac] = gcm_elev_d01[glac]
  
    # ===== SYNTHETIC SIMULATION =====
    elif input.option_synthetic_sim == 1:
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
        datelength = dates_table.shape[0] - input.gcm_spinupyears * 12
        n_tiles = int(np.ceil(datelength / dates_table_synthetic.shape[0]))
        gcm_temp = np.append(gcm_temp_tile[:,:input.gcm_spinupyears*12], 
                             np.tile(gcm_temp_tile,(1,n_tiles))[:,:datelength], axis=1)
        gcm_prec = np.append(gcm_prec_tile[:,:input.gcm_spinupyears*12], 
                             np.tile(gcm_prec_tile,(1,n_tiles))[:,:datelength], axis=1)
        gcm_lr = np.append(gcm_lr_tile[:,:input.gcm_spinupyears*12], np.tile(gcm_lr_tile,(1,n_tiles))[:,:datelength], 
                           axis=1)
        # Temperature and precipitation sensitivity adjustments
        gcm_temp = gcm_temp + input.synthetic_temp_adjust
        gcm_prec = gcm_prec * input.synthetic_prec_factor
       
    # ===== BIAS CORRECTIONS =====
    # No adjustments
    if input.option_bias_adjustment == 0 or gcm_name == input.ref_gcm_name:
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
        if input.option_bias_adjustment == 1:
            # Temperature bias correction
            gcm_temp_adj, gcm_elev_adj = gcmbiasadj.temp_biasadj_HH2015(ref_temp, ref_elev, gcm_temp, 
                                                                        dates_table_ref, dates_table)
            # Precipitation bias correction
            gcm_prec_adj, gcm_elev_adj = gcmbiasadj.prec_biasadj_opt1(ref_prec, ref_elev, gcm_prec, 
                                                                      dates_table_ref, dates_table)
        
        # OPTION 2: Adjust temp and prec using Huss and Hock (2015)
        elif input.option_bias_adjustment == 2:
            # Temperature bias correction
            gcm_temp_adj, gcm_elev_adj = gcmbiasadj.temp_biasadj_HH2015(ref_temp, ref_elev, gcm_temp, 
                                                                        dates_table_ref, dates_table)
            # Precipitation bias correction
            gcm_prec_adj, gcm_elev_adj = gcmbiasadj.prec_biasadj_HH2015(ref_prec, ref_elev, gcm_prec, 
                                                                        dates_table_ref, dates_table)
 
    # Checks on precipitation data
    if gcm_prec_adj.max() > 10:
        print('precipitation bias too high, needs to be modified')
        print(np.where(gcm_prec_adj > 10))
    elif gcm_prec_adj.min() < 0:
        print('Negative precipitation value')
        print(np.where(gcm_prec_adj < 0))
    
#%%
    # ===== RUN MASS BALANCE =====
    # Dataset to store model simulations and statistics
    # Number of simulations
    if input.option_calibration == 1:
        sim_iters = 1
    elif input.option_calibration == 2:
        sim_iters = input.sim_iters
    # Create datasets
    output_ds_all, encoding = create_xrdataset(main_glac_rgi, dates_table, sim_iters=sim_iters, 
                                               option_wateryear=input.gcm_wateryear)
    output_ds_all_stats, encoding = create_xrdataset(main_glac_rgi, dates_table, record_stats=1, 
                                                     option_wateryear=input.gcm_wateryear)
    
    for glac in range(main_glac_rgi.shape[0]):
        if glac == 0 or glac == main_glac_rgi.shape[0]:
            print(gcm_name,':', main_glac_rgi.loc[main_glac_rgi.index.values[glac],'RGIId'])
        # Select subsets of data
        glacier_rgi_table = main_glac_rgi.loc[main_glac_rgi.index.values[glac], :]
        glacier_gcm_elev = gcm_elev_adj[glac]
        glacier_gcm_prec = gcm_prec_adj[glac,:]
        glacier_gcm_temp = gcm_temp_adj[glac,:]
        glacier_gcm_lrgcm = gcm_lr[glac,:]
        glacier_gcm_lrglac = glacier_gcm_lrgcm.copy()
        glacier_area_t0 = main_glac_hyps.iloc[glac,:].values.astype(float)
        icethickness_t0 = main_glac_icethickness.iloc[glac,:].values.astype(float)
        width_t0 = main_glac_width.iloc[glac,:].values.astype(float)
        
        if input.hindcast == 1:
            glacier_gcm_prec = glacier_gcm_prec[::-1]
            glacier_gcm_temp = glacier_gcm_temp[::-1]
            glacier_gcm_lrgcm = glacier_gcm_lrgcm[::-1]
            glacier_gcm_lrglac = glacier_gcm_lrglac[::-1]

        # get glacier number
        if rgi_regionsO1[0] >= 10:
            glacier_RGIId = main_glac_rgi.iloc[glac]['RGIId'][6:]
        else:
            glacier_RGIId = main_glac_rgi.iloc[glac]['RGIId'][7:]
        
        if debug:
            print(glacier_RGIId)
            
        if input.option_import_modelparams == 1:
            if input.option_calibration == 1:
                ds_mp = xr.open_dataset(input.modelparams_fp_dict[rgi_regionsO1[0]] + glacier_RGIId + '.nc')
                cn_subset = input.modelparams_colnames
                modelparameters_all = (pd.DataFrame(ds_mp.mp_value.sel(chain=0).values, 
                                                    columns=ds_mp.mp.values)[cn_subset])
            elif input.option_calibration == 2:
                ds_mp = xr.open_dataset(input.modelparams_fp_dict[rgi_regionsO1[0]] + glacier_RGIId + '.nc')
                cn_subset = input.modelparams_colnames
                modelparameters_all = (pd.DataFrame(ds_mp['mp_value'].sel(chain=0).values, 
                                                    columns=ds_mp.mp.values)[cn_subset])
        else:
            modelparameters_all = (
                    pd.DataFrame(np.asarray([input.lrgcm, input.lrglac, input.precfactor, input.precgrad, input.ddfsnow, 
                                             input.ddfice, input.tempsnow, input.tempchange]).reshape(1,-1), 
                                             columns=input.modelparams_colnames))
        
        # Set the number of iterations and determine every kth iteration to use for the ensemble
        if (input.option_calibration == 1) or (modelparameters_all.shape[0] == 1):
            sim_iters = 1
        elif input.option_calibration == 2:
            sim_iters = input.sim_iters
            # Select every kth iteration
            mp_spacing = int((modelparameters_all.shape[0] - input.sim_burn) / sim_iters)
            mp_idx_start = np.arange(input.sim_burn, input.sim_burn + mp_spacing)
            np.random.shuffle(mp_idx_start)
            mp_idx_start = mp_idx_start[0]
            mp_idx_all = np.arange(mp_idx_start, modelparameters_all.shape[0], mp_spacing)
            
        # Loop through model parameters
        for n_iter in range(sim_iters):

            if sim_iters == 1:
                modelparameters = modelparameters_all.mean()  
            else:
                mp_idx = mp_idx_all[n_iter]
                modelparameters = modelparameters_all.iloc[mp_idx,:]
                
            if debug:
                print(glacier_RGIId, ':', [modelparameters[2], modelparameters[4], modelparameters[7]])
                debug_mb = True
            else:
                debug_mb = False
                       
            # run mass balance calculation
            (glac_bin_temp, glac_bin_prec, glac_bin_acc, glac_bin_refreeze, glac_bin_snowpack, glac_bin_melt,
             glac_bin_frontalablation, glac_bin_massbalclim, glac_bin_massbalclim_annual, glac_bin_area_annual,
             glac_bin_icethickness_annual, glac_bin_width_annual, glac_bin_surfacetype_annual,
             glac_wide_massbaltotal, glac_wide_runoff, glac_wide_snowline, glac_wide_snowpack,
             glac_wide_area_annual, glac_wide_volume_annual, glac_wide_ELA_annual, offglac_wide_prec, 
             offglac_wide_refreeze, offglac_wide_melt, offglac_wide_snowpack, offglac_wide_runoff) = (
                massbalance.runmassbalance(modelparameters[0:8], glacier_rgi_table, glacier_area_t0, icethickness_t0,
                                           width_t0, elev_bins, glacier_gcm_temp, glacier_gcm_prec, 
                                           glacier_gcm_elev, glacier_gcm_lrgcm, glacier_gcm_lrglac, dates_table, 
                                           option_areaconstant=0, debug=True,
#                                           debug=debug_mb
                                           ))
            
            if input.hindcast == 1:                
                glac_bin_temp = glac_bin_temp[:,::-1]
                glac_bin_prec = glac_bin_prec[:,::-1]
                glac_bin_acc = glac_bin_acc[:,::-1]
                glac_bin_refreeze = glac_bin_refreeze[:,::-1]
                glac_bin_snowpack = glac_bin_snowpack[:,::-1]
                glac_bin_melt = glac_bin_melt[:,::-1]
                glac_bin_frontalablation = glac_bin_frontalablation[:,::-1]
                glac_bin_massbalclim = glac_bin_massbalclim[:,::-1]
                glac_bin_massbalclim_annual = glac_bin_massbalclim_annual[:,::-1]
                glac_bin_area_annual = glac_bin_area_annual[:,::-1]
                glac_bin_icethickness_annual = glac_bin_icethickness_annual[:,::-1]
                glac_bin_width_annual = glac_bin_width_annual[:,::-1]
                glac_bin_surfacetype_annual = glac_bin_surfacetype_annual[:,::-1]
                glac_wide_massbaltotal = glac_wide_massbaltotal[::-1]
                glac_wide_runoff = glac_wide_runoff[::-1]
                glac_wide_snowline = glac_wide_snowline[::-1]
                glac_wide_snowpack = glac_wide_snowpack[::-1]
                glac_wide_area_annual = glac_wide_area_annual[::-1]
                glac_wide_volume_annual = glac_wide_volume_annual[::-1]
                glac_wide_ELA_annual = glac_wide_ELA_annual[::-1]
                offglac_wide_prec = offglac_wide_prec[::-1]
                offglac_wide_refreeze = offglac_wide_refreeze[::-1]
                offglac_wide_melt = offglac_wide_melt[::-1]
                offglac_wide_snowpack = offglac_wide_snowpack[::-1]
                offglac_wide_runoff = offglac_wide_runoff[::-1]
                
            
            if debug:
                # Compute glacier volume change for every time step and use this to compute mass balance
                #  this will work for any indexing
                glac_wide_area = glac_wide_area_annual[:-1].repeat(12)
                # Mass change [km3 mwe]
                #  mb [mwea] * (1 km / 1000 m) * area [km2]
                glac_wide_masschange = glac_wide_massbaltotal / 1000 * glac_wide_area
                # Mean annual mass balance [mwea]
                mb_mwea = (glac_wide_masschange.sum() / glac_wide_area[0] * 1000 / 
                           (glac_wide_masschange.shape[0] / 12))
                print('mb_model [mwea]:', mb_mwea.round(6))
                
            

            # RECORD PARAMETERS TO DATASET            
            if input.output_package == 2:
                (glac_wide_temp, glac_wide_prec, glac_wide_acc, glac_wide_refreeze, glac_wide_melt, 
                 glac_wide_frontalablation, glac_wide_massbaltotal, glac_wide_runoff, glac_wide_snowline, 
                 glac_wide_area_annual, glac_wide_volume_annual, glac_wide_ELA_annual) = (
                         convert_glacwide_results(elev_bins, glac_bin_temp, glac_bin_prec, glac_bin_acc, 
                                                  glac_bin_refreeze, glac_bin_snowpack, glac_bin_melt, 
                                                  glac_bin_frontalablation, glac_bin_massbalclim_annual, 
                                                  glac_bin_area_annual, glac_bin_icethickness_annual))

                # Record output to xarray dataset
                output_ds_all.temp_glac_monthly[glac, :, n_iter] = glac_wide_temp
                output_ds_all.prec_glac_monthly[glac, :, n_iter] = glac_wide_prec
                output_ds_all.acc_glac_monthly[glac, :, n_iter] = glac_wide_acc
                output_ds_all.refreeze_glac_monthly[glac, :, n_iter] = glac_wide_refreeze
                output_ds_all.melt_glac_monthly[glac, :, n_iter] = glac_wide_melt
                output_ds_all.frontalablation_glac_monthly[glac, :, n_iter] = glac_wide_frontalablation
                output_ds_all.massbaltotal_glac_monthly[glac, :, n_iter] = glac_wide_massbaltotal
                output_ds_all.runoff_glac_monthly[glac, :, n_iter] = glac_wide_runoff
                output_ds_all.snowline_glac_monthly[glac, :, n_iter] = glac_wide_snowline
                output_ds_all.area_glac_annual[glac, :, n_iter] = glac_wide_area_annual
                output_ds_all.volume_glac_annual[glac, :, n_iter] = glac_wide_volume_annual
                output_ds_all.ELA_glac_annual[glac, :, n_iter] = glac_wide_ELA_annual
                output_ds_all.offglac_prec_monthly[glac, :, n_iter] = offglac_wide_prec
                output_ds_all.offglac_refreeze_monthly[glac, :, n_iter] = offglac_wide_refreeze
                output_ds_all.offglac_melt_monthly[glac, :, n_iter] = offglac_wide_melt
                output_ds_all.offglac_snowpack_monthly[glac, :, n_iter] = offglac_wide_snowpack
                output_ds_all.offglac_runoff_monthly[glac, :, n_iter] = offglac_wide_runoff
                
#            if debug:
#                print('glac runoff max:', np.round(glac_wide_runoff.max(),0), 
#                      'glac prec max:', np.round(glac_wide_prec.max(),2),
#                      'glac refr max:', np.round(glac_wide_refreeze.max(),2),
#                      'offglac ref max:', np.round(offglac_wide_refreeze.max(),2))
            
                
        # Calculate statistics of simulations
        # List of variables
        ds_vns = []
        for vn in output_ds_all.variables:
            ds_vns.append(vn)
        for vn in ds_vns:
            if vn in input.output_variables_package2:
                stats = calc_stats(vn, output_ds_all, glac=glac)
                output_ds_all_stats[vn].values[glac,:,:] = stats            
        
#        if debug:
#            mb_mwea_all = ((output_ds_all_stats.massbaltotal_glac_monthly.values[glac,:,0]).sum(axis=0) / 
#                            (dates_table.shape[0] / 12))
#            print('mb_model [mwea] mean:', round(mb_mwea_all,4))   
#            # Calibration
#            cal_idx = np.where(cal_data.glacno == main_glac_rgi.glacno)[0][0]
#            mb_cal_mwea = cal_data.loc[cal_idx, 'mb_mwe'] / (cal_data.loc[cal_idx, 't2'] - cal_data.loc[cal_idx, 't1'])
#            print('mb_cal [mwea]:', round(mb_cal_mwea,4))
                
    # Export statistics to netcdf
    if input.output_package == 2:
        output_sim_fp = input.output_sim_fp + gcm_name + '/'
        # Create filepath if it does not exist
        if os.path.exists(output_sim_fp) == False:
            os.makedirs(output_sim_fp)
        # Netcdf filename
        if (gcm_name == 'ERA-Interim') or (gcm_name == 'COAWST'):
            # Filename
            netcdf_fn = ('R' + str(rgi_regionsO1[0]) + '_' + gcm_name + '_c' + 
                         str(input.option_calibration) + '_ba' + str(input.option_bias_adjustment) + '_' +  
                         str(sim_iters) + 'sets' + '_' + str(input.gcm_startyear) + '_' + str(input.gcm_endyear) + 
                         '--' + str(count) + '.nc')
        else:
            netcdf_fn = ('R' + str(rgi_regionsO1[0]) + '_' + gcm_name + '_' + rcp_scenario + '_c' + 
                         str(input.option_calibration) + '_ba' + str(input.option_bias_adjustment) + '_' +  
                         str(sim_iters) + 'sets' + '_' + str(input.gcm_startyear) + '_' + str(input.gcm_endyear) + 
                         '--' + str(count) + '.nc')
        if input.option_synthetic_sim==1:
            netcdf_fn = (netcdf_fn.split('--')[0] + '_T' + str(input.synthetic_temp_adjust) + '_P' + 
                         str(input.synthetic_prec_factor) + '--' + netcdf_fn.split('--')[1])
        if args.batch_number is not None:
            netcdf_fn_split = netcdf_fn.split('--')  
            netcdf_fn = netcdf_fn_split[0] + '_batch' + str(args.batch_number) + '--' + netcdf_fn_split[1]
        # Export netcdf
        output_ds_all_stats.to_netcdf(output_sim_fp + netcdf_fn, encoding=encoding)
   
    # Close datasets
    output_ds_all_stats.close()
    output_ds_all.close()

    #%% Export variables as global to view in variable explorer
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

    # RGI region number
    if args.spc_region is not None:
        rgi_regionsO1 = [int(args.spc_region)]
    else:
        rgi_regionsO1 = input.rgi_regionsO1

    # RGI glacier number
    if args.rgi_glac_number_fn is not None:
        with open(args.rgi_glac_number_fn, 'rb') as f:
            rgi_glac_number = pickle.load(f)
    else:
        rgi_glac_number = input.rgi_glac_number

    # Select all glaciers in a region
    main_glac_rgi_all = modelsetup.selectglaciersrgitable(rgi_regionsO1=rgi_regionsO1, rgi_regionsO2 = 'all',
                                                          rgi_glac_number=rgi_glac_number)
    
    # Define chunk size for parallel processing
    if args.option_parallels != 0:
        num_cores = int(np.min([len(rgi_glac_number), args.num_simultaneous_processes]))
        chunk_size = int(np.ceil(len(rgi_glac_number) / num_cores))
    else:
        # if not running in parallel, chunk size is all glaciers
        num_cores = 1
        chunk_size = len(rgi_glac_number)
        
    # Read GCM names from argument parser
    gcm_name = args.gcm_list_fn
    if args.gcm_name is not None:
        gcm_list = [args.gcm_name]
        rcp_scenario = args.rcp
    elif args.gcm_list_fn == input.ref_gcm_name:
        gcm_list = [input.ref_gcm_name]
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
        n = 0
        for chunk in range(0, main_glac_rgi_all.shape[0], chunk_size):
            list_packed_vars.append([n, chunk, main_glac_rgi_all, chunk_size, gcm_name])
            n += 1

        # Parallel processing
        if args.option_parallels != 0:
            print('Processing in parallel with ' + str(num_cores) + ' cores...')
            with multiprocessing.Pool(args.num_simultaneous_processes) as p:
                p.map(main,list_packed_vars)
        # If not in parallel, then only should be one loop
        else:
            # Loop through the chunks and export bias adjustments
            for n in range(len(list_packed_vars)):
                main(list_packed_vars[n])
                
        # Merge netcdf files together into one
        # Filenames to merge
        output_list_sorted = []
        output_sim_fp = input.output_sim_fp + gcm_name + '/'
        if input.option_calibration == 1:
            sim_iters = 1
        elif input.option_calibration == 2:
            sim_iters = input.sim_iters
        if (gcm_name == 'ERA-Interim') or (gcm_name == 'COAWST'):
            check_str = ('R' + str(rgi_regionsO1[0]) + '_' + gcm_name + '_c' + 
                         str(input.option_calibration) + '_ba' + str(input.option_bias_adjustment) + '_' +  
                         str(sim_iters) + 'sets' + '_' + str(input.gcm_startyear) + '_' + str(input.gcm_endyear) 
                         + '--')
        else:
            check_str = ('R' + str(rgi_regionsO1[0]) + '_' + gcm_name + '_' + rcp_scenario + '_c' + 
                         str(input.option_calibration) + '_ba' + str(input.option_bias_adjustment) + '_' +  
                         str(sim_iters) + 'sets' + '_' + str(input.gcm_startyear) + '_' + str(input.gcm_endyear) 
                         + '--')
        if input.option_synthetic_sim==1:
            check_str = (check_str.split('--')[0] + '_T' + str(input.synthetic_temp_adjust) + '_P' + 
                         str(input.synthetic_prec_factor) + '--')
        if args.batch_number is not None:
            check_str = check_str.split('--')[0] + '_batch' + str(args.batch_number) + '--'
        for i in os.listdir(output_sim_fp):
            if i.startswith(check_str):
                output_list_sorted.append([int(i.split('--')[1].split('.')[0]), i])
        output_list_sorted = sorted(output_list_sorted)
        output_list = [i[1] for i in output_list_sorted]
        # Open datasets and combine
        count_ds = 0
        for i in output_list:
            count_ds += 1
            
            print(count_ds, 'output_list file:', i)
            
            ds = xr.open_dataset(output_sim_fp + i)
            # Merge datasets of stats into one output
            if count_ds == 1:
                ds_all = ds
            else:
                ds_all = xr.merge((ds_all, ds))
            # Close dataset (NOTE: closing dataset here causes error on supercomputer)
#            ds.close()
                
        # Filename
        ds_all_fn = i.split('--')[0] + '.nc'
        # Encoding
        # Add variables to empty dataset and merge together
        encoding = {}
        noencoding_vn = ['stats', 'glac_attrs']
        if input.output_package == 2:
            for vn in input.output_variables_package2:
                # Encoding (specify _FillValue, offsets, etc.)
                if vn not in noencoding_vn:
                    encoding[vn] = {'_FillValue': False}
        # Export to netcdf
        if input.output_package == 2:
            ds_all.to_netcdf(output_sim_fp + ds_all_fn, encoding=encoding)
        else:
            ds_all.to_netcdf(output_sim_fp + ds_all_fn)
        # Close dataset
        ds.close()
        ds_all.close()
        # Remove files in output_list
        for i in output_list:
            os.remove(output_sim_fp + i)

    print('Total processing time:', time.time()-time_start, 's')
    
#    print('memory:', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 10**6, 'GB')

#%% ===== PLOTTING AND PROCESSING FOR MODEL DEVELOPMENT =====
    # Place local variables in variable explorer
    if args.option_parallels == 0:
        main_vars_list = list(main_vars.keys())
        gcm_name = main_vars['gcm_name']
#        rcp_scenario = main_vars['rcp_scenario']
        main_glac_rgi = main_vars['main_glac_rgi']
        main_glac_hyps = main_vars['main_glac_hyps']
        main_glac_icethickness = main_vars['main_glac_icethickness']
        main_glac_width = main_vars['main_glac_width']
        dates_table = main_vars['dates_table']
        if input.option_synthetic_sim == 1:
            dates_table_synthetic = main_vars['dates_table_synthetic']
            gcm_temp_tile = main_vars['gcm_temp_tile']
            gcm_prec_tile = main_vars['gcm_prec_tile']
            gcm_lr_tile = main_vars['gcm_lr_tile']
        gcm_temp = main_vars['gcm_temp']
        gcm_prec = main_vars['gcm_prec']
        gcm_elev = main_vars['gcm_elev']
        gcm_lr = main_vars['gcm_lr']
        gcm_temp_adj = main_vars['gcm_temp_adj']
        gcm_prec_adj = main_vars['gcm_prec_adj']
        gcm_elev_adj = main_vars['gcm_elev_adj']
#        if input.option_bias_adjustment != 0:
#            main_glac_biasadj = main_vars['main_glac_biasadj']
        gcm_temp_lrglac = main_vars['gcm_lr']
        output_ds_all = main_vars['output_ds_all']
        modelparameters = main_vars['modelparameters']
        glacier_rgi_table = main_vars['glacier_rgi_table']
        glacier_gcm_temp = main_vars['glacier_gcm_temp']
        glacier_gcm_prec = main_vars['glacier_gcm_prec']
        glacier_gcm_elev = main_vars['glacier_gcm_elev']
        glacier_gcm_lrgcm = main_vars['glacier_gcm_lrgcm']
        glacier_gcm_lrglac = glacier_gcm_lrgcm
        glacier_area_t0 = main_vars['glacier_area_t0']
        icethickness_t0 = main_vars['icethickness_t0']
        width_t0 = main_vars['width_t0']
        elev_bins = main_vars['elev_bins']
        glac_bin_frontalablation = main_vars['glac_bin_frontalablation']
        glac_bin_area_annual = main_vars['glac_bin_area_annual']
        glac_bin_massbalclim_annual = main_vars['glac_bin_massbalclim_annual']
        glac_bin_melt = main_vars['glac_bin_melt']
        glac_bin_acc = main_vars['glac_bin_acc']
        glac_bin_refreeze = main_vars['glac_bin_refreeze']
        glac_bin_snowpack = main_vars['glac_bin_snowpack']
        glac_bin_temp = main_vars['glac_bin_temp']
        glac_bin_prec = main_vars['glac_bin_prec']
        glac_bin_massbalclim = main_vars['glac_bin_massbalclim']
        glac_wide_massbaltotal = main_vars['glac_wide_massbaltotal']
        glac_wide_area_annual = main_vars['glac_wide_area_annual']
        glac_wide_volume_annual = main_vars['glac_wide_volume_annual']
        glac_wide_runoff = main_vars['glac_wide_runoff']
        glac_wide_prec = main_vars['glac_wide_prec']
        glac_wide_refreeze = main_vars['glac_wide_refreeze']
        
        
        modelparameters_all = main_vars['modelparameters_all']
        sim_iters = main_vars['sim_iters']
#        cal_data = main_vars['cal_data']
#        if input.option_calibration == 2:
#            mp_idx = main_vars['mp_idx']
#            mp_idx_all = main_vars['mp_idx_all']
#        netcdf_fn = main_vars['netcdf_fn']
        
#%%
#    ds = xr.open_dataset(input.output_sim_fp + 'CanESM2/R13_CanESM2_rcp26_c2_ba1_100sets_2000_2100.nc')
#    vol_annual = ds.volume_glac_annual.values[0,:,0]
#    mb_monthly = ds.massbaltotal_glac_monthly.values[0,:,0]
#    area_annual = ds.area_glac_annual.values[0,:,0]
#
#    # ===== MASS CHANGE CALCULATIONS =====
#    # Compute glacier volume change for every time step and use this to compute mass balance
##    glac_wide_area = np.repeat(area_annual[:,:-1], 12, axis=1)
#    glac_wide_area = np.repeat(area_annual[:-1], 12)
#    
#    # Mass change [km3 mwe]
#    #  mb [mwea] * (1 km / 1000 m) * area [km2]
#    glac_wide_masschange = mb_monthly / 1000 * glac_wide_area
#    
#    print('Average mass balance:', np.round(glac_wide_masschange.sum() / 101, 2), 'Gt/yr')
#    
#    print('Average mass balance:', np.round(mb_monthly.sum() / 101, 2), 'mwea')
#    
#    A = mb_monthly[0:18*12]
#    print(A.sum() / 18)
#    
#    print('Vol change[%]:', vol_annual[-1] / vol_annual[0] * 100)
#    ds.close()
        
#    #%%
#    # ===== MASS CHANGE CALCULATIONS: ISSUE WITH USING AVERAGE MB AND AREA TO COMPUTE VOLUME CHANGE ======
#    # Mean volume change from each volume simulation
#    A = output_ds_all.volume_glac_annual.values[0,:,:]
#    A_volchg = A[-1,:] - A[0,:]
#    A_volchg_mean = np.mean(A_volchg)
#    
#    # Mean volume change from each mass balance and area simulation
#    B = output_ds_all.massbaltotal_glac_monthly.values[0,:,:]
#    B_area = (output_ds_all.area_glac_annual.values[0,:-1,:]).repeat(12,axis=0)
#    B_volchg_monthly = B / 1000 * B_area / 0.9
#    B_volchg = np.sum(B_volchg_monthly, axis=0)
#    B_volchg_mean = np.mean(B_volchg)
#    
#    print('Volume change from each simulation of volume agree with each simulation of mass balance and area:',
#          'from volume:', np.round(A_volchg_mean,9), 'from MB/area:', np.round(B_volchg_mean,9), 
#          'difference:', np.round(A_volchg_mean - B_volchg_mean,9))
#    
#    # Mean volume change based on the mean mass balance and mean area (these are what we output because files would be
#    # too large to output every simulation)
#    B_mean = B.mean(axis=1)
#    B_mean_area = B_area.mean(axis=1)
#    B_mean_volchg_monthly = B_mean / 1000 * B_mean_area / 0.9
#    B_mean_volchg = np.sum(B_mean_volchg_monthly)
#    
#    print('\nVolume change from each simulation of volume is different than using mean mass balance and area',
#          'from volume', np.round(A_volchg_mean,9), 'from mean MB/area:', np.round(B_mean_volchg,9),
#          'difference:', np.round(A_volchg_mean - B_mean_volchg,9))
        