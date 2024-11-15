""" derive monthly glacier mass for PyGEM simulation using annual glacier mass and monthly total mass balance """

# Built-in libraries
import argparse
import collections
import copy
import inspect
import multiprocessing
import os
import sys
import time
# External libraries
import pandas as pd
import pickle
import numpy as np
import xarray as xr
import pygem
import pygem.setup.config as config
# read config
pygem_prms = config.read_config()
import pygem.pygem_modelsetup as modelsetup


def get_monthly_mass(glac_mass_annual, glac_massbaltotal_monthly):
    """
    funciton to calculate the monthly glacier mass
    from annual glacier mass and monthly total mass balance

    Parameters
    ----------
    glac_mass_annual : float
        ndarray containing the annual glacier mass for each year computed by PyGEM
        shape: [#glac, #years]
        unit: kg
    glac_massbaltotal_monthly : float
        ndarray containing the monthly total mass balance computed by PyGEM
        shape: [#glac, #months]
        unit: kg

    Returns
    -------
    glac_mass_monthly: float
        ndarray containing the monthly glacier mass
        shape : [#glac, #months]
        unit: kg

    """
    # get running total monthly mass balance - reshape into subarrays of all values for a given year, then take cumulative sum
    oshape = glac_massbaltotal_monthly.shape
    running_glac_massbaltotal_monthly = np.reshape(glac_massbaltotal_monthly, (-1,12), order='C').cumsum(axis=-1).reshape(oshape)

    # tile annual mass to then superimpose atop running glacier mass balance (trim off final year from annual mass)
    glac_mass_monthly = np.repeat(glac_mass_annual[:,:-1], 12, axis=-1)

    # add annual mass values to running glacier mass balance
    glac_mass_monthly += running_glac_massbaltotal_monthly

    return glac_mass_monthly


def update_xrdataset(input_ds, glac_mass_monthly):
    """
    update xarray dataset to add new fields

    Parameters
    ----------
    xrdataset : xarray Dataset
        existing xarray dataset
    newdata : ndarray 
        new data array
    description: str
        describing new data field

    output_ds : xarray Dataset
        empty xarray dataset that contains variables and attributes to be filled in by simulation runs
    encoding : dictionary
        encoding used with exporting xarray dataset to netcdf
    """
    # coordinates
    glac_values = input_ds.glac.values
    time_values = input_ds.time.values

    output_coords_dict = collections.OrderedDict()
    output_coords_dict['glac_mass_monthly'] = (
            collections.OrderedDict([('glac', glac_values), ('time', time_values)]))

    # Attributes dictionary
    output_attrs_dict = {}
    output_attrs_dict['glac_mass_monthly'] = {
            'long_name': 'glacier mass',
            'units': 'kg',
            'temporal_resolution': 'monthly',
            'comment': 'monthly glacier mass'}


    # Add variables to empty dataset and merge together
    count_vn = 0
    encoding = {}
    for vn in output_coords_dict.keys():
        empty_holder = np.zeros([len(output_coords_dict[vn][i]) for i in list(output_coords_dict[vn].keys())])
        output_ds = xr.Dataset({vn: (list(output_coords_dict[vn].keys()), empty_holder)},
                               coords=output_coords_dict[vn])
        count_vn += 1
        # Merge datasets of stats into one output
        if count_vn == 1:
            output_ds_all = output_ds
        else:
            output_ds_all = xr.merge((output_ds_all, output_ds))
    # Add attributes
    for vn in output_ds_all.variables:
        try:
            output_ds_all[vn].attrs = output_attrs_dict[vn]
        except:
            pass
        # Encoding (specify _FillValue, offsets, etc.)
        encoding[vn] = {'_FillValue': None,
                        'zlib':True,
                        'complevel':9
                        }    

    output_ds_all['glac_mass_monthly'].values = (
            glac_mass_monthly
            )

    return output_ds_all, encoding


def main(list_packed_vars):

    """
    create monthly mass data product
    Parameters
    ----------
    list_packed_vars : list
        list of packed variables that enable the use of parallels
    Returns
    -------
    statsds : netcdf Dataset
        updated stats netcdf containing monthly glacier mass
    """
    # Unpack variables
    count = list_packed_vars[0]
    glac_no = list_packed_vars[1]
    gcm_name = list_packed_vars[2]
    scenario = list_packed_vars[3]
    realization = list_packed_vars[4]
    gcm_bc_startyear = list_packed_vars[5]
    gcm_startyear = list_packed_vars[6]
    gcm_endyear = list_packed_vars[7]
    
    # ===== LOAD GLACIERS =====
    main_glac_rgi = modelsetup.selectglaciersrgitable(glac_no=glac_no)

    for glac in range(main_glac_rgi.shape[0]):
        if glac == 0:
            print(gcm_name,':', main_glac_rgi.loc[main_glac_rgi.index.values[glac],'RGIId'])
        # Select subsets of data
        glacier_rgi_table = main_glac_rgi.loc[main_glac_rgi.index.values[glac], :]
        glacier_str = '{0:0.5f}'.format(glacier_rgi_table['RGIId_float'])
        reg_str = str(glacier_rgi_table.O1Region).zfill(2)
        rgiid = main_glac_rgi.loc[main_glac_rgi.index.values[glac],'RGIId']

        # get datapath to stats datasets produced from run_simulation.py
        output_sim_stats_fp = pygem_prms.output_sim_fp + reg_str + '/' + gcm_name + '/'
        if gcm_name not in ['ERA-Interim', 'ERA5', 'COAWST']:
            output_sim_stats_fp += scenario + '/'
        output_sim_stats_fp += 'stats/'
        # Create filepath if it does not exist
        if os.path.exists(output_sim_stats_fp) == False:
            os.makedirs(output_sim_stats_fp, exist_ok=True)
        # Number of simulations
        if pygem_prms.option_calibration == 'MCMC':
            sim_iters = pygem_prms.sim_iters
        else:
            sim_iters = 1
        # Netcdf filename
        if gcm_name in ['ERA-Interim', 'ERA5', 'COAWST']:
            # Filename
            netcdf_fn = (glacier_str + '_' + gcm_name + '_' + str(pygem_prms.option_calibration) + '_ba0' +
                        '_' +  str(sim_iters) + 'sets' + '_' + str(gcm_startyear) + '_' + str(gcm_endyear) + '_all.nc')
        elif realization is not None:
            netcdf_fn = (glacier_str + '_' + gcm_name + '_' + scenario + '_' + realization + '_' +
                            str(pygem_prms.option_calibration) + '_ba' + str(pygem_prms.option_bias_adjustment) + 
                            '_' + str(sim_iters) + 'sets' + '_' + str(gcm_bc_startyear) + '_' + 
                            str(gcm_endyear) + '_all.nc')
        else:
            netcdf_fn = (glacier_str + '_' + gcm_name + '_' + scenario + '_' +
                            str(pygem_prms.option_calibration) + '_ba' + str(pygem_prms.option_bias_adjustment) + 
                            '_' + str(sim_iters) + 'sets' + '_' + str(gcm_bc_startyear) + '_' + 
                            str(gcm_endyear) + '_all.nc')
        
        if os.path.exists(output_sim_stats_fp + netcdf_fn):

            try:
                # open dataset
                statsds = xr.open_dataset(output_sim_stats_fp + netcdf_fn)

                # calculate monthly mass - pygem glac_massbaltotal_monthly is in units of m3, so convert to mass using density of ice
                glac_mass_monthly = get_monthly_mass(
                                                    statsds.glac_mass_annual.values, 
                                                    statsds.glac_massbaltotal_monthly.values * pygem_prms.density_ice, 
                                                    )
                statsds.close()

                # update dataset to add monthly mass change
                output_ds_stats, encoding = update_xrdataset(statsds, glac_mass_monthly)

                # close input ds before write
                statsds.close()

                # append to existing stats netcdf
                output_ds_stats.to_netcdf(output_sim_stats_fp + netcdf_fn, mode='a', encoding=encoding, engine='netcdf4')

                # close datasets
                output_ds_stats.close()
            
            except:
                pass
        else:
            print('Simulation not found: ',output_sim_stats_fp + netcdf_fn)

    return


#%% PARALLEL PROCESSING
if __name__ == '__main__':
    time_start = time.time()


    # set up CLI
    parser = argparse.ArgumentParser(
    description='''Script to process montly mass for PyGEM simulations file\n\nExample calls:\n$python process_monthly_mass.py -rgi_glac_number=13.40312 -gcm_name=CESM2 -scenario=ssp585\n$python process_monthly_mass.py -rgi_region01=13 -gcm_name=CESM2 -scenario=ssp585''',
    formatter_class=argparse.RawTextHelpFormatter)


    parser.add_argument('-rgi_region01', type=int, default=None,
                        help='Randoph Glacier Inventory region')
    parser.add_argument('-rgi_glac_number', type=str, default=None,
                        help='Randoph Glacier Inventory region')
    parser.add_argument('-rgi_glac_number_fn', action='store', type=str, default=None,
                        help='Filename containing list of rgi_glac_number, helpful for running batches on spc')
    parser.add_argument('-gcm_list_fn', action='store', type=str, default=pygem_prms.ref_gcm_name,
                        help='text file full of commands to run')
    parser.add_argument('-gcm_name', action='store', type=str, default=None,
                        help='GCM name used for model run')
    parser.add_argument('-scenario', action='store', type=str, default=None,
                        help='rcp or ssp scenario used for model run (ex. rcp26 or ssp585)')
    parser.add_argument('-realization', action='store', type=str, default=None,
                        help='realization from large ensemble used for model run (ex. 1011.001 or 1301.020)')
    parser.add_argument('-realization_list', action='store', type=str, default=None,
                        help='text file full of realizations to run')
    parser.add_argument('-gcm_bc_startyear', action='store', type=int, default=pygem_prms.gcm_bc_startyear,
                        help='start year for bias correction')
    parser.add_argument('-gcm_startyear', action='store', type=int, default=pygem_prms.gcm_startyear,
                        help='start year for the model run')
    parser.add_argument('-gcm_endyear', action='store', type=int, default=pygem_prms.gcm_endyear,
                        help='end year for the model run')
    parser.add_argument('-num_simultaneous_processes', action='store', type=int, default=4,
                        help='number of simultaneous processes (cores) to use')
    parser.add_argument('-batch_number', action='store', type=int, default=None,
                        help='Batch number used to differentiate output on supercomputer')
    # flags
    parser.add_argument('-option_ordered', action='store_true',
                        help='Flag to keep glacier lists ordered (default is off)')
    parser.add_argument('-option_parallels', action='store_true',
                        help='Flag to use or not use parallels (default is off)')
    args = parser.parse_args()


    # RGI glacier number
    if args.rgi_glac_number:
        glac_no = [args.rgi_glac_number]
    elif args.rgi_glac_number_fn is not None:
        with open(args.rgi_glac_number_fn, 'rb') as f:
            glac_no = pickle.load(f)
    elif args.rgi_region01:
        main_glac_rgi_all = modelsetup.selectglaciersrgitable(
                rgi_regionsO1=[args.rgi_region01], rgi_regionsO2=pygem_prms.rgi_regionsO2,
                rgi_glac_number=pygem_prms.rgi_glac_number, glac_no=pygem_prms.glac_no,
                include_landterm=pygem_prms.include_landterm, include_laketerm=pygem_prms.include_laketerm, 
                include_tidewater=pygem_prms.include_tidewater, 
                min_glac_area_km2=pygem_prms.min_glac_area_km2)        
        glac_no = list(main_glac_rgi_all['rgino_str'].values)
    elif pygem_prms.glac_no is not None:
        glac_no = pygem_prms.glac_no
    else:
        main_glac_rgi_all = modelsetup.selectglaciersrgitable(
                rgi_regionsO1=pygem_prms.rgi_regionsO1, rgi_regionsO2=pygem_prms.rgi_regionsO2,
                rgi_glac_number=pygem_prms.rgi_glac_number, glac_no=pygem_prms.glac_no,
                include_landterm=pygem_prms.include_landterm, include_laketerm=pygem_prms.include_laketerm, 
                include_tidewater=pygem_prms.include_tidewater, 
                min_glac_area_km2=pygem_prms.min_glac_area_km2)
        glac_no = list(main_glac_rgi_all['rgino_str'].values)

    # Number of cores for parallel processing
    if args.option_parallels != 0:
        num_cores = int(np.min([len(glac_no), args.num_simultaneous_processes]))
    else:
        num_cores = 1

    # Glacier number lists to pass for parallel processing
    glac_no_lsts = modelsetup.split_list(glac_no, n=num_cores, option_ordered=args.option_ordered)

    # Read GCM names from argument parser
    gcm_name = args.gcm_list_fn
    if args.gcm_name is not None:
        gcm_list = [args.gcm_name]
        scenario = args.scenario
    elif args.gcm_list_fn == pygem_prms.ref_gcm_name:
        gcm_list = [pygem_prms.ref_gcm_name]
        scenario = args.scenario
    else:
        with open(args.gcm_list_fn, 'r') as gcm_fn:
            gcm_list = gcm_fn.read().splitlines()
            scenario = os.path.basename(args.gcm_list_fn).split('_')[1]
            print('Found %d gcms to process'%(len(gcm_list)))
  
    # Read realizations from argument parser
    if args.realization is not None:
        realizations = [args.realization]
    elif args.realization_list is not None:
        with open(args.realization_list, 'r') as real_fn:
            realizations = list(real_fn.read().splitlines())
            print('Found %d realizations to process'%(len(realizations)))
    else:
        realizations = None
    
    # Producing realization or realization list. Best to convert them into the same format!
    # Then pass this as a list or None.
    # If passing this through the list_packed_vars, then don't go back and get from arg parser again!
 
    # Loop through all GCMs
    for gcm_name in gcm_list:
        print('Processing:', gcm_name, scenario)
        # Pack variables for multiprocessing
        list_packed_vars = []          
        if realizations is not None:
            for realization in realizations:
                for count, glac_no_lst in enumerate(glac_no_lsts):
                    list_packed_vars.append([count, glac_no_lst, gcm_name, scenario, realization, args.gcm_bc_startyear, args.gcm_startyear, args.gcm_endyear])
        else:
            for count, glac_no_lst in enumerate(glac_no_lsts):
                list_packed_vars.append([count, glac_no_lst, gcm_name, scenario, realizations, args.gcm_bc_startyear, args.gcm_startyear, args.gcm_endyear])
                
        print('len list packed vars:', len(list_packed_vars))
           
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