"""Split glaciers into lists to run on separate nodes on the supercomputer"""

# Built-in libraries
import argparse
import collections
import os
import zipfile
# External libraries
import numpy as np
import pandas as pd
import xarray as xr
# Local libraries
import pygem.pygem_input as pygem_prms
import pygemfxns_modelsetup as modelsetup
import pygemfxns_gcmbiasadj as gcmbiasadj
import run_simulation as simulation


#%run run_postprocessing.py -gcm_name='ERA-Interim' -merge_batches=1

option_multimodel = 0
option_merge_era = 1

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
    parser.add_argument('-region', action='store', type=int, default=None,
                        help='RGI region number (order 1)')
    parser.add_argument('-output_sim_fp', action='store', type=str, default=pygem_prms.output_sim_fp,
                        help='output simulation filepath where results are being stored by GCM')
    parser.add_argument('-option_remove_merged_files', action='store', type=int, default=0,
                        help='Switch to delete merged files or not (1-delete)')
    parser.add_argument('-option_remove_batch_files', action='store', type=int, default=1,
                        help='Switch to delete batch files or not (1-delete)')
    parser.add_argument('-merge_batches', action='store', type=int, default=0,
                        help='Switch to merge batches or not (1-merge)')
    parser.add_argument('-extract_subset', action='store', type=int, default=0,
                        help='Switch to extract a subset of variables or not (1-yes)')
    parser.add_argument('-unzip_files', action='store', type=int, default=0,
                        help='Switch to unzip files or not (1-yes)')
    parser.add_argument('-subset_byvar', action='store', type=int, default=0,
                        help='Switch to subset by each variables or not')
    parser.add_argument('-vars_mon2annualseasonal', action='store', type=int, default=0,
                        help='Switch to compute annual and seasonal data or not')
    parser.add_argument('-debug', action='store', type=int, default=0,
                        help='Boolean for debugging to turn it on or off (default 0 is off)')
    return parser

  
#%% ===== MERGE HINDCAST AND PRESENT SIMULATION =====
if option_merge_era == 1:
    print('MERGING ERA...')
    regions = ['13','14','15']
    ds_fp = '/Users/davidrounce/Documents/Dave_Rounce/HiMAT/Output/simulations/spc_20190914/merged/ERA-Interim/'
    
    print('ASSUMES THERE IS ONE YEAR OF OVERLAP')
    
    for region in regions:
        ds_fn1 = 'R' + region + '--all--ERA-Interim_c2_ba1_100sets_1980_2000.nc'
        ds_fn2 = 'R' + region + '--all--ERA-Interim_c2_ba1_100sets_2000_2017.nc'
        ds_merged_fn = 'R' + region + '--all--ERA-Interim_c2_ba1_100sets_1980_2017.nc'
        ds1_raw = xr.open_dataset(ds_fp + ds_fn1)
        ds2_raw = xr.open_dataset(ds_fp + ds_fn2)
        
        time_idx = list(np.arange(12,ds2_raw.time.shape[0]))
        year_idx = list(np.arange(1,ds2_raw.year.shape[0]))
        year_plus1_idx = list(np.arange(1,ds2_raw.year_plus1.shape[0]))
        year_plus1_idx_4ds1 = list(np.arange(0,ds1_raw.year_plus1.shape[0]-1))
        
        ds1 = ds1_raw.isel(year_plus1=year_plus1_idx_4ds1)
        ds2 = ds2_raw.isel(time=time_idx, year=year_idx, year_plus1=year_plus1_idx)
        
        time_vns = ['temp_glac_monthly', 'prec_glac_monthly', 'acc_glac_monthly', 'refreeze_glac_monthly', 
                    'melt_glac_monthly', 'frontalablation_glac_monthly', 'massbaltotal_glac_monthly', 'runoff_glac_monthly', 
                    'snowline_glac_monthly', 'offglac_prec_monthly', 'offglac_refreeze_monthly', 'offglac_melt_monthly', 
                    'offglac_snowpack_monthly', 'offglac_runoff_monthly']
        year_plus1_vns = ['area_glac_annual', 'volume_glac_annual']
        year_vns = ['ELA_glac_annual']
        
        ds1_time = ds1[time_vns]
        ds2_time = ds2[time_vns]
        ds3_time = xr.concat((ds1_time, ds2_time), 'time')
        
        ds1_year = ds1[year_vns]
        ds2_year = ds2[year_vns]
        ds3_year = xr.concat((ds1_year,ds2_year), 'year')
        
        ds1_year_plus1 = ds1[year_plus1_vns]
        ds2_year_plus1 = ds2[year_plus1_vns]
        ds3_year_plus1 = xr.concat((ds1_year_plus1,ds2_year_plus1), 'year_plus1')
        
        ds3_years = ds3_year.merge(ds3_year_plus1)
        ds3 = ds3_years.merge(ds3_time)
        ds3['glacier_table'] = ds1_raw['glacier_table']
        
        # Export merged dataset
        # Encoding
        # add variables to empty dataset and merge together
        encoding = {}
        noencoding_vn = ['stats', 'glac_attrs']
        if pygem_prms.output_package == 2:
            for encoding_vn in pygem_prms.output_variables_package2:
                # Encoding (specify _FillValue, offsets, etc.)
                if encoding_vn not in noencoding_vn:
                    encoding[encoding_vn] = {'_FillValue': False}
        ds3.to_netcdf(ds_fp + ds_merged_fn, encoding=encoding)       

#%% ====================== MULTI-MODEL SCRIPT! ===================
if option_multimodel == 1:
    print('MULTIMODEL CALCULATIONS')
    gcm_names = ['bcc-csm1-1', 'CanESM2', 'CESM1-CAM5', 'CCSM4', 'CNRM-CM5', 'CSIRO-Mk3-6-0', 'FGOALS-g2', 'GFDL-CM3', 
                 'GFDL-ESM2G', 'GFDL-ESM2M', 'GISS-E2-R', 'HadGEM2-ES', 'IPSL-CM5A-LR', 'IPSL-CM5A-MR', 'MIROC-ESM', 
                 'MIROC-ESM-CHEM', 'MIROC5', 'MPI-ESM-LR', 'MPI-ESM-MR', 'MRI-CGCM3', 'NorESM1-M', 'NorESM1-ME']
    #gcm_names = ['bcc-csm1-1']
    #rcps = ['rcp26', 'rcp45', 'rcp60', 'rcp85']
    #regions = [13,14,15]
    rcps = ['rcp85']
    regions = [15]
    zip_fp = '/Volumes/LaCie/HMA_PyGEM/2019_0914/spc_zipped/'
    multimodel_fp = zip_fp + '../multimodel/'
    
    ds_vns = ['temp_glac_monthly', 'prec_glac_monthly', 'acc_glac_monthly', 'refreeze_glac_monthly', 'melt_glac_monthly',
              'frontalablation_glac_monthly', 'massbaltotal_glac_monthly', 'runoff_glac_monthly', 'snowline_glac_monthly', 
              'area_glac_annual', 'volume_glac_annual', 'ELA_glac_annual', 'offglac_prec_monthly', 
              'offglac_refreeze_monthly', 'offglac_melt_monthly', 'offglac_snowpack_monthly', 'offglac_runoff_monthly']
        
    for batman in [0]:
        def sum_multimodel(fn, vn, ds_var_multimodel_sum, count):  
            """ Sum multimodel to avoid creating excessively large np.arrays that crash memory """
            # Open dataset and use first dataset as multimodel dataset to retain attributes
            ds = xr.open_dataset(multimodel_fp + fn)
            print(fn, np.round(ds[vn][1,1,0].values,3))
        
            # Select values of variable
            ds_var = ds[vn][:,:,0].values
            # Concatenate into numpy array
            if ds_var_multimodel_sum is None:
                ds_var_multimodel_sum = ds_var
            else:
                ds_var_multimodel_sum += ds_var
        
            ds.close()
            
            # Record count to divide by to get mean in the end
            count += 1
                           
            return ds_var_multimodel_sum, count
        
        def sum_multimodel_variance(fn, vn, ds_var_multimodel_stdsum, ds_var_multimodel_mean):  
            """ Sum multimodel variance to avoid creating excessively large np.arrays that crash memory """
            # Open dataset and use first dataset as multimodel dataset to retain attributes
            ds = xr.open_dataset(multimodel_fp + fn)
            print(fn, 'std calc')
        
            # Select values of variable
            ds_var = ds[vn][:,:,0].values
            
            ds_var_stdsum = (ds_var - ds_var_multimodel_mean)**2
            
            # Concatenate into numpy array
            if ds_var_multimodel_stdsum is None:
                ds_var_multimodel_stdsum = ds_var_stdsum
            else:
                ds_var_multimodel_stdsum += ds_var_stdsum
        
            ds.close()
                           
            return ds_var_multimodel_stdsum
        
        for region in regions:
            for rcp in rcps:
                
                for gcm_name in gcm_names:
                    gcm_fp = zip_fp + gcm_name + '/'
                    for i in os.listdir(gcm_fp):
                        if str(region) in i and rcp in i and os.path.exists(multimodel_fp + i.replace('.zip','')) == False:
                            print('Extracting ' + i)
                            with zipfile.ZipFile(gcm_fp + i, 'r') as zipObj:
                                # Extract all the contents of zip file in current directory
                                zipObj.extractall(multimodel_fp)
                
    
                list_fns = []
                for i in os.listdir(multimodel_fp):
                    if str(region) in i and rcp in i:
                        list_fns.append(i)
                
                print(len(list_fns), list_fns)
                
                # Use existing dataset to setup multimodel netcdf structure
                ds_multimodel = xr.open_dataset(multimodel_fp + list_fns[0])
                
                for vn in ds_vns:
                    print(vn)
                    
                    ds_var_multimodel_sum = None
                    ds_var_multimodel_stdsum = None
                    count = 0
                    
                    # Multimodel mean
                    # sum data from each array to reduce memory requirements
                    for i in list_fns:
                        ds_var_multimodel_sum, count = sum_multimodel(i, vn, ds_var_multimodel_sum, count)
                    # compute mean
                    ds_var_multimodel_mean = ds_var_multimodel_sum / count
                    
                    print('Mean:', np.round(ds_var_multimodel_mean[1,1],3))
                    
                    # Multimodel standard deviation
                    # sum squared difference
                    for i in list_fns:
                        ds_var_multimodel_stdsum = sum_multimodel_variance(i, vn, ds_var_multimodel_stdsum, 
                                                                           ds_var_multimodel_mean)
                    # compute standard deviation
                    ds_var_multimodel_std = (ds_var_multimodel_stdsum / count)**0.5
                    
                    print('Std:', np.round(ds_var_multimodel_std[1,1],3))
    
                    ds_multimodel[vn][:,:,:] = (
                            np.concatenate((ds_var_multimodel_mean[:,:,np.newaxis], ds_var_multimodel_std[:,:,np.newaxis]), 
                                           axis=2))
                    
                # Export merged dataset
                # Encoding
                # add variables to empty dataset and merge together
                encoding = {}
                noencoding_vn = ['stats', 'glac_attrs']
                if pygem_prms.output_package == 2:
                    for encoding_vn in pygem_prms.output_variables_package2:
                        # Encoding (specify _FillValue, offsets, etc.)
                        if encoding_vn not in noencoding_vn:
                            encoding[encoding_vn] = {'_FillValue': False}
                    
                ds_multimodel_fn = 'R' + str(region) + '_multimodel_' + rcp + '_c2_ba1_100sets_2000_2100.nc'
                ds_multimodel.to_netcdf(multimodel_fp + ds_multimodel_fn, encoding=encoding)       
                
                    #%%

def merge_batches(gcm_name, output_sim_fp=pygem_prms.output_sim_fp, rcp=None,
                  option_remove_merged_files=0, option_remove_batch_files=0, debug=False):   
    """ MERGE BATCHES """
    
#for gcm_name in ['CCSM4', 'GFDL-CM3', 'GFDL-ESM2M', 'GISS-E2-R', 'IPSL-CM5A-LR', 'MIROC5', 'MRI-CGCM3', 'NorESM1-M']:
#    debug=True
#    netcdf_fp = pygem_prms.output_sim_fp + gcm_name + '/'    
    
    splitter = '_batch'
    zipped_fp = output_sim_fp + 'spc_zipped/'
    merged_fp = output_sim_fp + 'spc_merged/'
    netcdf_fp = output_sim_fp + gcm_name + '/'
    
    # Check file path exists
    if os.path.exists(zipped_fp) == False:
        os.makedirs(zipped_fp)
    
    if os.path.exists(merged_fp) == False:
        os.makedirs(merged_fp)
    
    regions = []
    rcps = []
    for i in os.listdir(netcdf_fp):
        if i.endswith('.nc'):
            
            i_region = int(i.split('_')[0][1:])
            if i_region not in regions:
                regions.append(i_region)
            
            if gcm_name not in ['ERA-Interim']:
                i_rcp = i.split('_')[2]
                if i_rcp not in rcps:
                    rcps.append(i_rcp)
    regions = sorted(regions)
    rcps = sorted(rcps)
    
    # Set RCPs if not for GCM and/or if overriding with argument
    if len(rcps) == 0:
        rcps = [None]
    
    if rcp is not None:
        rcps = [rcp]
    
    if debug:
        print('Regions:', regions, 
              '\nRCPs:', rcps)
    
    # Encoding
    # Add variables to empty dataset and merge together
    encoding = {}
    noencoding_vn = ['stats', 'glac_attrs']
    if pygem_prms.output_package == 2:
        for vn in pygem_prms.output_variables_package2:
            # Encoding (specify _FillValue, offsets, etc.)
            if vn not in noencoding_vn:
                encoding[vn] = {'_FillValue': False}
    
    
    for reg in regions:
        
        check_str = 'R' + str(reg) + '_' + gcm_name
        
        for rcp in rcps:
            
            if rcp is not None:
                check_str = 'R' + str(reg) + '_' + gcm_name + '_' + rcp
                
            if debug:
                print('Region(s)', reg, 'RCP', rcp, ':', 'check_str:', check_str)
            
            output_list = []
            merged_list = []
            
            for i in os.listdir(netcdf_fp):
                if i.startswith(check_str) and splitter in i:
                    output_list.append([int(i.split(splitter)[1].split('.')[0]), i])
            output_list = sorted(output_list)
            output_list = [i[1] for i in output_list]
            
            # Open datasets and combine
            count_ds = 0
            for i in output_list:
                if debug:
                    print(i)
                count_ds += 1
                ds = xr.open_dataset(netcdf_fp + i)
                # Merge datasets of stats into one output
                if count_ds == 1:
                    ds_all = ds
                else:
                    ds_all = xr.concat([ds_all, ds], dim='glac')
            ds_all.glac.values = np.arange(0,len(ds_all.glac.values))
            ds_all_fn = i.split(splitter)[0] + '.nc'
            # Export to netcdf
            ds_all.to_netcdf(merged_fp + ds_all_fn, encoding=encoding)
            
            print('Merged ', gcm_name, rcp, 'Region(s)', reg)
            
            merged_list.append(merged_fp + ds_all_fn)
            
            if debug:
                print(merged_list)
            
            # Zip file to reduce file size
            with zipfile.ZipFile(zipped_fp + ds_all_fn + '.zip', mode='w', compression=zipfile.ZIP_DEFLATED) as myzip:
                myzip.write(merged_fp + ds_all_fn, arcname=ds_all_fn)
                
            # Remove unzipped files
            if option_remove_merged_files == 1:
                for i in merged_list:
                    os.remove(i)
            
            if option_remove_batch_files == 1:
                # Remove batch files
                for i in output_list:
                    os.remove(netcdf_fp + i)
  

#def extract_subset(gcm_name, netcdf_fp=pygem_prms.output_sim_fp):
def extract_subset(gcm_name, rcp_scenario=None, region_no=None, netcdf_fp=pygem_prms.output_sim_fp, unzip_files=0):
##gcm_names = ['CanESM2', 'CESM1-CAM5', 'CNRM-CM5', 'CSIRO-Mk3-6-0', 'FGOALS-g2', 
##             'GFDL-ESM2G', 'HadGEM2-ES', 'IPSL-CM5A-MR', 'MIROC-ESM', 
##             'MIROC-ESM-CHEM', 'MPI-ESM-LR', 'MPI-ESM-MR', 'NorESM1-ME']
#gcm_names = ['CanESM2']
#zip_fp = '/Volumes/LaCie/HMA_PyGEM/2019_0914/spc_zipped/'
##netcdf_fp = pygem_prms.output_sim_fp
#rcp_scenario= None
#region_no=None
#unzip_files = 1
#for gcm_name in gcm_names:
#    
#    netcdf_fp = zip_fp + gcm_name + '/'
#    print(netcdf_fp)
    
    vns_all = pygem_prms.output_variables_package2
    
    vns_subset = ['massbaltotal_glac_monthly', 'runoff_glac_monthly', 'offglac_runoff_monthly', 'area_glac_annual', 
                  'volume_glac_annual', 'glacier_table']
    
    # List of variable names to drop from merged file            
    drop_vns = [item for item in vns_all if item not in vns_subset]
    
    if rcp_scenario is None:
        rcp_checkstr = 'rcp'
    else:
        rcp_checkstr = rcp_scenario
    
    if region_no is None:
        region_checkstr = 'R'
    else:
        region_checkstr = 'R' + str(region_no)
    
    # Unzip files
    if unzip_files == 1:
        for i in os.listdir(netcdf_fp):
            # Unzip file if it doesn't exist yet
            if ((i.endswith('.nc.zip')) and (os.path.isfile((netcdf_fp + i).replace('.zip','')) == False)
                and (rcp_checkstr in i) and (region_checkstr in i)):
                with zipfile.ZipFile(netcdf_fp + i, 'r') as zip_ref:
                    zip_ref.extractall(netcdf_fp)            
    
#    # Loop through files to extract filenames
#    regions = []
#    rcps = []
#    for i in os.listdir(netcdf_fp):
#        if i.endswith('.nc') and gcm_name in i:
#            i_region = int(i.split('_')[0][1:])
#            i_rcp = i.split('_')[2]
#        
#            if i_region not in regions:
#                regions.append(i_region)
#            if i_rcp not in rcps:
#                rcps.append(i_rcp)
#    regions = sorted(regions)
#    rcps = sorted(rcps)
    
    # Determine RCPs
    if rcp_scenario is not None:
        rcps = [rcp_scenario]
    else:
        rcps = []
        for i in os.listdir(netcdf_fp):
            if i.endswith('.nc') and gcm_name in i:
#                i_rcp = i.split('_')[2]
                i_rcp = 'rcp' + i.split('rcp')[1].split('_')[0]
                if i_rcp not in rcps:
                    rcps.append(i_rcp)
        rcps = sorted(rcps)
        
    # Determine Regions
    if region_no is not None:
        regions = [region_no]
    else:
        regions = []
        for i in os.listdir(netcdf_fp):
            if i.endswith('.nc') and gcm_name in i:
#                i_region = int(i.split('_')[0][1:]) 
                i_region = i.split('R')[1][0:2]
                if i_region not in regions:
                    regions.append(i_region)
        regions = sorted(regions)
        

    ds_fns = []
    for i in os.listdir(netcdf_fp):
        if (i.endswith('.nc')) and (rcp_checkstr in i) and (region_checkstr in i):
            ds_fns.append(i)
    

    # Extract subsets
    for ds_fn in ds_fns:
        # Encoding
        encoding = {}
        noencoding_vn = ['stats', 'glac_attrs']
        # Encoding (specify _FillValue, offsets, etc.)
        for vn in vns_subset:
            if vn not in noencoding_vn:
                encoding[vn] = {'_FillValue': False}
            
        # Open datasets and combine
        ds = xr.open_dataset(netcdf_fp + ds_fn)
        # Drop variables
        ds = ds.drop(drop_vns)                
        ds_new_fn = ds_fn.replace('.nc', '--subset.nc')
        # Export to netcdf
        subset_fp = netcdf_fp + '../spc_subset/'
        # Add filepath if it doesn't exist
        if not os.path.exists(subset_fp):
            os.makedirs(subset_fp)
        ds.to_netcdf(subset_fp + ds_new_fn, encoding=encoding)
        ds.close()
            
        vol_glac_all = ds.volume_glac_annual.values[:,:,0]
        vol_remain_perc = vol_glac_all[:,vol_glac_all.shape[1]-1].sum() / vol_glac_all[:,0].sum() * 100
        reg = ds_fn.split('--')[0]
        rcp = ds_fn.split('_')[1]
        print(gcm_name, 'Region', reg, rcp, 'Vol remain [%]:', np.round(vol_remain_perc,1))
        
        # Remove file
        os.remove(netcdf_fp + ds_fn)            
        
#    # Extract subsets
#    for reg in regions:
#        for rcp in rcps:
#            check_str = 'R' + str(reg) + '_' + gcm_name + '_' + rcp
#            output_list = []
#            
#            for i in os.listdir(netcdf_fp):
#                if i.startswith(check_str):
#                    ds_fn = i
#                    output_list.append(i)
#                
#                    # Encoding
#                    encoding = {}
#                    noencoding_vn = ['stats', 'glac_attrs']
#                    # Encoding (specify _FillValue, offsets, etc.)
#                    for vn in vns_subset:
#                        if vn not in noencoding_vn:
#                            encoding[vn] = {'_FillValue': False}
#                        
#                    # Open datasets and combine
#                    ds = xr.open_dataset(netcdf_fp + ds_fn)
#                    # Drop variables
#                    ds = ds.drop(drop_vns)                
#                    ds_new_fn = ds_fn.split('.nc')[0] + '--subset.nc'
#                    # Export to netcdf
#                    subset_fp = netcdf_fp + '../spc_subset/'
#                    print(subset_fp)
#                    # Add filepath if it doesn't exist
#                    if not os.path.exists(subset_fp):
#                        os.makedirs(subset_fp)
#                    ds.to_netcdf(subset_fp + ds_new_fn, encoding=encoding)
#                    ds.close()
#                        
#                    vol_glac_all = ds.volume_glac_annual.values[:,:,0]
#                    vol_remain_perc = vol_glac_all[:,vol_glac_all.shape[1]-1].sum() / vol_glac_all[:,0].sum() * 100
#                    print(gcm_name, 'Region', reg, rcp, 'Vol remain [%]:', np.round(vol_remain_perc,1))
#                    
##                    # Remove file
##                    os.remove(netcdf_fp + i)
                
              
def subset_byvar(gcm_name):    
    vns_all = pygem_prms.output_variables_package2
    vns_subset = pygem_prms.output_variables_package2
    
    if pygem_prms.output_package == 2:
        vns_all = pygem_prms.output_variables_package2
    
    netcdf_fp = pygem_prms.output_sim_fp
    
    regions = []
    rcps = []
    for i in os.listdir(netcdf_fp):
        if i.endswith('.nc'):
            i_region = int(i.split('_')[0][1:])
            i_rcp = i.split('_')[2]
        
            if i_region not in regions:
                regions.append(i_region)
            if i_rcp not in rcps:
                rcps.append(i_rcp)
    regions = sorted(regions)
    rcps = sorted(rcps)
    
    for reg in regions:
        for rcp in rcps:
            check_str = 'R' + str(reg) + '_' + gcm_name + '_' + rcp
            output_list = []
            
            for i in os.listdir(netcdf_fp):
                if i.startswith(check_str):
                    ds_fn = i
                    output_list.append(i)

            for vn in vns_subset:
                # List of variable names to drop from merged file            
                drop_vns = [item for item in vns_all if item not in [vn]]
    
                # Encoding
                # Add variables to empty dataset and merge together
                encoding = {}
                noencoding_vn = ['stats', 'glac_attrs']
                # Encoding (specify _FillValue, offsets, etc.)
                if vn not in noencoding_vn:
                    encoding[vn] = {'_FillValue': False}
                    
                # Open datasets and combine
                ds = xr.open_dataset(netcdf_fp + ds_fn)
                # Drop variables
                ds = ds.drop(drop_vns)                
                ds_new_fn = ds_fn.split('.nc')[0] + '--' + vn + '.nc'
                # Export to netcdf
                subset_fp = pygem_prms.output_sim_fp + '/spc_vars/' + vn + '/'
                # Add filepath if it doesn't exist
                if not os.path.exists(subset_fp):
                    os.makedirs(subset_fp)
                ds.to_netcdf(subset_fp + ds_new_fn, encoding=encoding)
                ds.close()
                    
                if vn == 'volume_glac_annual':
                    vol_glac_all = ds.volume_glac_annual.values[:,:,0]
                    vol_remain_perc = vol_glac_all[:,vol_glac_all.shape[1]-1].sum() / vol_glac_all[:,0].sum() * 100
                    print(gcm_name, 'Region', reg, rcp, 'Vol remain [%]:', vol_remain_perc)
            
#            # Delete file
#            for i in output_list:
#                os.remove(netcdf_fp + i)
#            
#    # Delete directory
#    for i in os.listdir(netcdf_fp):
#        if i.endswith('.DS_Store'):
#            os.remove(netcdf_fp + i)
#    os.rmdir(netcdf_fp)
    
    
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
    

def vars_mon2annualseasonal(gcm_name):
    netcdf_fp_prefix = pygem_prms.output_sim_fp + 'spc_vars/'
    vns = ['acc_glac_monthly', 'melt_glac_monthly', 'refreeze_glac_monthly', 'frontalablation_glac_monthly', 
           'massbaltotal_glac_monthly', 'temp_glac_monthly', 'prec_glac_monthly', 'runoff_glac_monthly']

    for vn in vns:
        netcdf_fp = netcdf_fp_prefix + vn + '/'
        for i in os.listdir(netcdf_fp):
            if i.endswith('.nc') and gcm_name in i:
                print(i)
                               
                # Open dataset and extract annual values
                ds = xr.open_dataset(netcdf_fp + i)      
                ds_mean = ds[vn].values[:,:,0]
                ds_std = ds[vn].values[:,:,1]
                ds_var = ds_std**2
                
                # Compute annual/seasonal mean/sum and standard deviation for the variable of interest
                if vn is 'temp_glac_monthly':
                    output_list = ['annual']
                    # Mean annual temperature, standard deviation, and variance
                    ds_mean_annual = ds_mean.reshape(-1,12).mean(axis=1).reshape(-1,int(ds_mean.shape[1]/12))
                    ds_var_annual = ds_var.reshape(-1,12).mean(axis=1).reshape(-1,int(ds_std.shape[1]/12))
                    ds_std_annual = ds_var_annual**0.5
                    ds_values_annual = np.concatenate((ds_mean_annual[:,:,np.newaxis], ds_std_annual[:,:,np.newaxis]), 
                                                      axis=2)
                elif vn in ['prec_glac_monthly', 'runoff_glac_monthly']:
                    output_list = ['annual']
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
                    
                    print('CHANGE BACK OPTION WATER YEAR HERE - DUE TO MANUAL ERROR')
                    option_wateryear=pygem_prms.gcm_wateryear
                    
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
#                os.remove(netcdf_fp + i)

    
#%%
if __name__ == '__main__':
    parser = getparser()
    args = parser.parse_args()
    
    if args.debug == 1:
        debug = True
    else:
        debug = False
    
    if args.merge_batches == 1:
        merge_batches(args.gcm_name, output_sim_fp=args.output_sim_fp, rcp=args.rcp,
                      option_remove_merged_files=args.option_remove_merged_files,
                      option_remove_batch_files=args.option_remove_batch_files)
        
    if args.extract_subset == 1:
        extract_subset(args.gcm_name, rcp_scenario=args.rcp, region_no=args.region, netcdf_fp=args.output_sim_fp,
                       unzip_files=args.unzip_files)    
        
    if args.subset_byvar == 1:
        subset_byvar(args.gcm_name)
        
    if args.vars_mon2annualseasonal == 1:
        vars_mon2annualseasonal(args.gcm_name)
        
##%% RE-ORDER OUTPUT BY GLACIER NUMBER
#for region in regions:
#    for rcp in rcps:
#        
#        output_fp = '../Output/simulations/spc_20190914/merged/'
#        ds_fn = 'R' + str(region) + '--all--IPSL-CM5A-LR_' + rcp + '_c2_ba1_100sets_2000_2100.nc'
#        #ds_fn = 'R15--all--IPSL-CM5A-LR_rcp60_c2_ba1_100sets_2000_2100.nc'
#        ds = xr.open_dataset(output_fp + ds_fn)
#        df = pd.DataFrame(ds.glacier_table.values, columns=ds.glac_attrs.values)
#        
#        glacno = df.glacno.values.astype(int)
#        ds.glac.values = glacno
#        ds2 = ds.sortby('glac')
#        df2 = pd.DataFrame(ds2.glacier_table.values, columns=ds2.glac_attrs.values)
#        
##        glacno_str = [str(region) + '.' + str(x).zfill(5) for x in glacno]
##        main_glac_rgi = modelsetup.selectglaciersrgitable(glac_no = glacno_str)
#        
##        A = ds2.area_glac_annual.values[:,:,0]
##        B = A[:,0] - main_glac_rgi['Area'].values
#        
#        ds2_fn = ds_fn.replace('.nc', '-ordered.nc')
#        # Encoding
#        # Add variables to empty dataset and merge together
#        encoding = {}
#        noencoding_vn = ['stats', 'glac_attrs']
#        for vn in pygem_prms.output_variables_package2:
#            # Encoding (specify _FillValue, offsets, etc.)
#            if vn not in noencoding_vn:
#                encoding[vn] = {'_FillValue': False}
#        # Export to netcdf
#        ds2.to_netcdf(output_fp + ds2_fn, encoding=encoding)
#        # Close dataset
#        ds.close()
#        ds2.close()
#        
#        os.remove(output_fp + ds_fn)