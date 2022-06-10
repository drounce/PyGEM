"""Merge batches generated from running simulations in parallel"""

# Built-in Libraries
import os
import argparse
# External Libraries
import numpy as np
import xarray as xr
# Local Libraries
import pygem.pygem_input as pygem_prms

#%%
def getparser():
    """
    Use argparse to add arguments from the command line
    
    Parameters
    ----------
    gcm_name (optional) : str
        gcm name      
        
    Returns
    -------
    Object containing arguments and their respective values.
    """
    parser = argparse.ArgumentParser(description="options for merging batches to single netcdf")
    # add arguments
    parser.add_argument('-gcm_name', action='store', type=str, default=None,
                        help='GCM name used for model run')
    parser.add_argument('-splitter', action='store', type=str, default='_batch',
                        help='string used to split batches')
    parser.add_argument('-netcdf_fp_prefix', action='store', type=str, default=pygem_prms.output_sim_fp,
                        help='string used to split batches')
    return parser


#%%
#gcm_name = 'GFDL-CM3'

# Select options from parser
parser = getparser()
args = parser.parse_args()
gcm_name = args.gcm_name
splitter = args.splitter
netcdf_fp_prefix = args.netcdf_fp_prefix


netcdf_fp = netcdf_fp_prefix + gcm_name + '/'
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
    for rcp in rcps:
        print('R', reg, rcp, ':')
        check_str = 'R' + str(reg) + '_' + gcm_name + '_' + rcp
        output_list = []
        
        for i in os.listdir(netcdf_fp):
            if i.startswith(check_str):
                output_list.append([int(i.split(splitter)[1].split('.')[0]), i])
        output_list = sorted(output_list)
        output_list = [i[1] for i in output_list]

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
        ds_all.to_netcdf(netcdf_fp + '../' + ds_all_fn, encoding=encoding)
#        # Remove files in output_list
#        for i in output_list:
#            os.remove(netcdf_fp + i)
        