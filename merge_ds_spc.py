#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 08:21:18 2018
@author: davidrounce
"""
# Built-in libraries
import argparse
import multiprocessing
import os
import zipfile
# External libraries
import numpy as np
import pandas as pd
import xarray as xr
# Local libraries
import pygem.pygem_input as pygem_prms
import spc_split_glaciers as split_glaciers


#%% Functions
def getparser():
    """
    Use argparse to add arguments from the command line
    
    Parameters
    ----------
    gcm_name (optional) : str
        gcm name
    rcp (optional) : str
        representative concentration pathway (ex. 'rcp26')
    merge_batches (optional) : int
        switch to run merge_batches fxn (1 merge, 0 ignore)
    debug : int
        Switch for turning debug printing on or off (default = 0 (off))
        
    Returns
    -------
    Object containing arguments and their respective values.
    """
    parser = argparse.ArgumentParser(description="run calibration in parallel")
    # add arguments
    parser.add_argument('-gcm_name', action='store', type=str, default=None,
                        help='GCM name used for model run')
    parser.add_argument('-rcp', action='store', type=str, default=None,
                        help='rcp scenario used for model run (ex. rcp26)')
    parser.add_argument('-output_sim_fp', action='store', type=str, default=pygem_prms.output_sim_fp,
                        help='output simulation filepath where results are being stored by GCM')
    parser.add_argument('-region', action='store', type=str, default=None,
                        help='region for merging')
    parser.add_argument('-num_simultaneous_processes', action='store', type=int, default=1,
                        help='number of simultaneous processes (cores) to use')
    parser.add_argument('-chunk_start', action='store', type=int, default=None,
                        help='starting chunk point')
    parser.add_argument('-chunk_end', action='store', type=int, default=None,
                        help='ending chunk point')
    parser.add_argument('-chunk_no', action='store', type=int, default=None,
                        help='ending chunk point')
    return parser



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
    glac_fullfn_lst = list_packed_vars[1]
    
    #%%
    # Merge by chunk
    for nglac, i in enumerate(glac_fullfn_lst):
        ds = xr.open_dataset(i)
        if nglac == 0:
            ds_all = ds
        else:
            ds_all = xr.concat([ds_all, ds], 'glac')
            
    # Filename
    region = i.split('/')[-1].split('.')[0]
    glac_no = i.split('/')[-1].split('_')[0]
    ds_chunk_fn = i.split('/')[-1].replace(glac_no + '_','R' + region + '--chunk' + str(count).zfill(2) + '--')
    
    # Encoding
    # Add variables to empty dataset and merge together
    encoding = {}
    noencoding_vn = ['stats', 'glac_attrs']
    for vn in pygem_prms.output_variables_package2:
        # Encoding (specify _FillValue, offsets, etc.)
        if vn not in noencoding_vn:
            encoding[vn] = {'_FillValue': False}
    # Export to netcdf
    ds_all.to_netcdf(ds_all_fp + ds_chunk_fn, encoding=encoding)
    # Close dataset
    ds.close()
    ds_all.close()
    # Remove files in output_list
    for i in glac_fullfn_lst:
        os.remove(i)
    
    
if __name__ == '__main__':
    parser = getparser()
    args = parser.parse_args()
    
    gcm_name = args.gcm_name
    nchunks = args.num_simultaneous_processes

    output_fp = '../Output/simulations/'
    ds_fp = output_fp + gcm_name + '/'
    ds_all_fp = output_fp + 'merged/' + gcm_name + '/'
    
    # Check file path exists
    if os.path.exists(ds_all_fp) == False:
        os.makedirs(ds_all_fp)
    
#    gcm_files = []
#    for i in os.listdir(ds_fp):
#        if i.endswith('.nc'):
#            gcm_files.append(ds_fp + i)
#    gcm_files = sorted(gcm_files)
    
    gcm_files = []
    if args.rcp is not None:
        rcps = [args.rcp]
        ds_fp += args.rcp + '/'
        for i in os.listdir(ds_fp):
            if i.endswith('.nc'):
                full_fn = ds_fp + i
                gcm_files.append(full_fn)
    else:
        for i in os.listdir(ds_fp):
            if i.endswith('.nc'):
                full_fn = ds_fp + i
                gcm_files.append(full_fn)
            elif os.path.isdir(ds_fp + i):
                for j in os.listdir(ds_fp + i):
                    if j.endswith('.nc'):
                        full_fn = ds_fp + i + '/' + j
                        gcm_files.append(full_fn) 
    gcm_files = sorted(gcm_files)
    
    rcps = []
    regions = []
    for i in gcm_files:
        # Regions
        i_region = i.split('/')[-1].split('.')[0]
        if i_region not in regions:
            regions.append(i_region)
        
        # RCPs
        i_rcp = i.split('/')[-1].split('_')[2]
        if i_rcp not in rcps:
            rcps.append(i_rcp)
    
    regions = sorted(regions)
    rcps = sorted(rcps)
    
    if args.rcp is not None:
        rcps = [args.rcp]

    if len(rcps) == 0:
        rcps.append(gcm_name)
        
#    regions = ['14', '15']
#    rcps = ['rcp26', 'rcp45', 'rcp60', 'rcp85']
    
    for rcp in rcps:
        for region in regions:
            print(rcp, region)
            
            glac_fullfn_region = []
            for i in gcm_files:
                fn = i.split('/')[-1]
                if fn.startswith(region) and rcp in fn:
                    glac_fullfn_region.append(i)
            glac_fullfn_region = sorted(glac_fullfn_region)
            
            print(rcp, region, len(glac_fullfn_region), 'glaciers')
            
            # Split into lists for parallel processing
            glac_fullfn_lsts = split_glaciers.split_list(glac_fullfn_region, n=nchunks)
            
            # Pack variables for multiprocessing
            list_packed_vars = []
            for count, glac_fullfn_lst in enumerate(glac_fullfn_lsts):
                list_packed_vars.append([count, glac_fullfn_lst])

            # Parallel processing
            # MERGE INDIVIDUAL FILES
            if nchunks != 0:
                print('Processing in parallel with ' + str(nchunks) + ' cores...')
                with multiprocessing.Pool(args.num_simultaneous_processes) as p:
                    p.map(main,list_packed_vars)
            # If not in parallel, then only should be one loop
            else:
                # Loop through the chunks and export bias adjustments
                for n in range(len(list_packed_vars)):
                    main(list_packed_vars[n])
                
            # MERGE CHUNKS
            chunk_fns = []
            for i in os.listdir(ds_all_fp):
                if i.startswith('R' + region) and i.endswith('.nc') and rcp in i and gcm_name in i and 'chunk' in i:
                    print(i)
                    chunk_fns.append(i)
            chunk_fns = sorted(chunk_fns)
            for nchunk, chunk_fn in enumerate(chunk_fns):
                ds = xr.open_dataset(ds_all_fp + chunk_fn)
                if nchunk == 0:
                    ds_all = ds
                else:
                    ds_all = xr.concat([ds_all, ds], 'glac')
            # Update glacier values
            ds_all.glac.values = ds_all.glacier_table.values[:,0].astype(int)
                    
            # Encoding
            # Add variables to empty dataset and merge together
            encoding = {}
            noencoding_vn = ['stats', 'glac_attrs']
            for vn in pygem_prms.output_variables_package2:
                # Encoding (specify _FillValue, offsets, etc.)
                if vn not in noencoding_vn:
                    encoding[vn] = {'_FillValue': False}
            # Export file 
            chunkno = chunk_fn.split('--')[1]
            ds_all_fn = chunk_fn.replace(chunkno, 'all')
            # Export to netcdf
            ds_all.to_netcdf(ds_all_fp + ds_all_fn, encoding=encoding)
            # Close dataset
            ds.close()
            ds_all.close()
            
            # Remove files
            for chunk_fn in chunk_fns:
                os.remove(ds_all_fp + chunk_fn)
            
            # Zip file to reduce file size
            with zipfile.ZipFile(ds_all_fp + ds_all_fn + '.zip', mode='w', compression=zipfile.ZIP_DEFLATED) as myzip:
                myzip.write(ds_all_fp + ds_all_fn, arcname=ds_all_fn)
            
            # Remove non-zipped file
            os.remove(ds_all_fp + ds_all_fn)