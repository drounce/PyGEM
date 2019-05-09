"""Split glaciers into lists to run on separate nodes on the supercomputer"""

# Built-in libraries
import argparse
import collections
import os
import zipfile
# External libraries
import numpy as np
import xarray as xr
# Local libraries
import pygem_input as input
import pygemfxns_modelsetup as modelsetup


#%run run_postprocessing.py -gcm_name='ERA-Interim' -merge_batches=1

#%% Functions
def getparser():
    """
    Use argparse to add arguments from the command line
    
    Parameters
    ----------
    gcm_name (optional) : str
        gcm name
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
    parser.add_argument('-merge_batches', action='store', type=int, default=0,
                        help='Switch to merge batches or not (1-merge)')
    parser.add_argument('-extract_subset', action='store', type=int, default=0,
                        help='Switch to extract a subset of variables or not (1-yes)')
    parser.add_argument('-subset_byvar', action='store', type=int, default=0,
                        help='Switch to subset by each variables or not')
    parser.add_argument('-vars_mon2annualseasonal', action='store', type=int, default=0,
                        help='Switch to compute annual and seasonal data or not')
    parser.add_argument('-debug', action='store', type=int, default=0,
                        help='Boolean for debugging to turn it on or off (default 0 is off)')
    return parser



def merge_batches(gcm_name):   
    """ MERGE BATCHES """
    
#for gcm_name in ['CanESM2']:
#    debug=True
    
    
    splitter = '_batch'
    netcdf_fp = input.output_sim_fp + gcm_name + '/'
    zipped_fp = netcdf_fp + '../spc_zipped/'
    
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
    
    if len(rcps) == 0:
        rcps = [None]
    
    print('Regions:', regions, 
          '\nRCPs:', rcps)
    
    # Encoding
    # Add variables to empty dataset and merge together
    encoding = {}
    noencoding_vn = ['stats', 'glac_attrs']
    if input.output_package == 2:
        for vn in input.output_variables_package2:
            # Encoding (specify _FillValue, offsets, etc.)
            if vn not in noencoding_vn:
                encoding[vn] = {'_FillValue': False}
    
    
    for reg in regions:
#    for reg in [15]:
        
        check_str = 'R' + str(reg) + '_' + gcm_name
        
        print(check_str)
        
        for rcp in rcps:
#        for rcp in ['rcp85']:
            
            if rcp is not None:
                check_str = 'R' + str(reg) + '_' + gcm_name + '_' + rcp
                
            print('R', reg, 'RCP', rcp, ':', 'check_str:', check_str)
            
            output_list = []
            merged_list = []
            
            for i in os.listdir(netcdf_fp):
                if i.startswith(check_str) and splitter in i:
                    output_list.append([int(i.split(splitter)[1].split('.')[0]), i])
            output_list = sorted(output_list)
            output_list = [i[1] for i in output_list]
            
#            if debug:
            print(output_list)
            
            # Open datasets and combine
            count_ds = 0
            for i in output_list:
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
            ds_all.to_netcdf(input.output_sim_fp + ds_all_fn, encoding=encoding)
            
            print('Merged ', gcm_name, reg, rcp)
            
            merged_list.append(input.output_sim_fp + ds_all_fn)
            
            print(merged_list)
            
            # Zip file to reduce file size
            # Check file path exists
            if os.path.exists(zipped_fp) == False:
                os.makedirs(zipped_fp)
                
            with zipfile.ZipFile(zipped_fp + ds_all_fn + '.zip', mode='w', compression=zipfile.ZIP_DEFLATED) as myzip:
                myzip.write(input.output_sim_fp + ds_all_fn, arcname=ds_all_fn)
                
#            # Remove unzipped files
#            for i in merged_list:
#                os.remove(i)
            
            # Remove batch files
            for i in output_list:
                os.remove(netcdf_fp + i)
  

#def extract_subset(gcm_name):  
gcm_names = ['MIROC-ESM', 'MIROC-ESM-CHEM', 'MIROC5', 'MRI-CGCM3', 'NorESM1-ME']
for gcm_name in gcm_names:
    
    vns_all = input.output_variables_package2
    
    vns_subset = ['massbaltotal_glac_monthly', 'runoff_glac_monthly', 'offglac_runoff_monthly', 'area_glac_annual', 
                  'volume_glac_annual', 'glacier_table']
    
    # List of variable names to drop from merged file            
    drop_vns = [item for item in vns_all if item not in vns_subset]
    
    netcdf_fp = input.output_sim_fp
    
    regions = []
    rcps = []
    for i in os.listdir(netcdf_fp):
        if i.endswith('.nc') and gcm_name in i:
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
                    ds_new_fn = ds_fn.split('.nc')[0] + '--subset.nc'
                    # Export to netcdf
                    subset_fp = input.output_sim_fp + '/spc_subset/'
                    # Add filepath if it doesn't exist
                    if not os.path.exists(subset_fp):
                        os.makedirs(subset_fp)
                    ds.to_netcdf(subset_fp + ds_new_fn, encoding=encoding)
                    ds.close()
                        
                    vol_glac_all = ds.volume_glac_annual.values[:,:,0]
                    vol_remain_perc = vol_glac_all[:,vol_glac_all.shape[1]-1].sum() / vol_glac_all[:,0].sum() * 100
                    print(gcm_name, 'Region', reg, rcp, 'Vol remain [%]:', np.round(vol_remain_perc,1))
                    
#                    # Remove file
#                    os.remove(netcdf_fp + i)
                
              
def subset_byvar(gcm_name):    
    vns_all = input.output_variables_package2
    vns_subset = input.output_variables_package2
    
    if input.output_package == 2:
        vns_all = input.output_variables_package2
    
    netcdf_fp = input.output_sim_fp
    
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
                subset_fp = input.output_sim_fp + '/spc_vars/' + vn + '/'
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
    netcdf_fp_prefix = input.output_sim_fp + 'spc_vars/'
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
                    option_wateryear=input.gcm_wateryear
                    
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
        merge_batches(args.gcm_name)
        
    if args.extract_subset == 1:
        extract_subset(args.gcm_name)
        
    if args.subset_byvar == 1:
        subset_byvar(args.gcm_name)
        
    if args.vars_mon2annualseasonal == 1:
        vars_mon2annualseasonal(args.gcm_name)