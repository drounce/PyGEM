"""
Mass redistribution analysis
"""
import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt

import pygem_input as input
import pygemfxns_modelsetup as modelsetup

#%% Input data
rgi_regionO1 = [13, 14, 15]
search_binnedcsv_fn = input.main_directory + '\\..\\DEMs\\mb_bins_sample_20180323\\*_mb_bins.csv'
search_rgiv6_fn = input.main_directory + '\\..\\RGI\\rgi60\\00_rgi60_attribs\\' + '*'

#%% Select files
# Find files for analysis
rgi_files = glob.glob(search_rgiv6_fn)
binnedcsv_files = glob.glob(search_binnedcsv_fn)

# RGIId's of available glaciers
ds = pd.DataFrame()
ds['reg_glacno'] = [x.split('\\')[-1].split('_')[0] for x in binnedcsv_files]
ds['RGIId'] = 'RGI60-' + ds['reg_glacno'] 
ds['region'] = ds.reg_glacno.astype(float).astype(int)
ds['glacno_str'] = ds.reg_glacno.str.split('.').apply(lambda x: x[1])
ds['glacno'] = ds.reg_glacno.str.split('.').apply(lambda x: x[1]).astype(int)

main_glac_rgi = pd.DataFrame()
#for region in rgi_regionO1:
#for region in [13]:
for n, region in enumerate(rgi_regionO1):
    print('Region', region)
    ds_reg = ds[ds.region == region]
    rgi_glac_number = ds_reg['glacno_str'].tolist()
    
    # only import glaciers that are not 0
    
    if len(rgi_glac_number) > 0: 
        main_glac_rgi_reg = modelsetup.selectglaciersrgitable(rgi_regionsO1=[region], rgi_regionsO2 = 'all', 
                                                              rgi_glac_number=rgi_glac_number)
        # concatenate regions
        main_glac_rgi = main_glac_rgi.append(main_glac_rgi_reg, ignore_index=True)

#%%
#for n in range(len(binnedcsv_files)):
for n in [0]:
    print(binnedcsv_files[n])
    ds_binnedcsv = pd.read_csv(binnedcsv_files[n])
    
    


#%%
#def import_Husstable(rgi_table, rgi_regionsO1, filepath, filedict, drop_col_names,
#                     indexname=input.indexname):
#    """Use the dictionary specified by the user to extract the desired variable.
#    The files must be in the proper units (ice thickness [m], area [km2], width [km]) and need to be pre-processed to 
#    have all bins between 0 - 8845 m.
#    
#    Output is a Pandas DataFrame of the variable for all the glaciers in the model run
#    (rows = GlacNo, columns = elevation bins).
#    
#    Line Profiling: Loading in the table takes the most time (~2.3 s)
#    """
#    ds = pd.read_csv(filepath + filedict[rgi_regionsO1[0]])
#    # Select glaciers based on 01Index value from main_glac_rgi table
#    #  as long as Huss tables have all rows associated with rgi attribute table, then this shortcut works and saves time
#    glac_table = ds.iloc[rgi_table['O1Index'].values]
#    # must make copy; otherwise, drop will cause SettingWithCopyWarning
#    glac_table_copy = glac_table.copy()
#    # Clean up table and re-index
#    # Reset index to be GlacNo
#    glac_table_copy.reset_index(drop=True, inplace=True)
#    glac_table_copy.index.name = indexname
#    # Drop columns that are not elevation bins
#    glac_table_copy.drop(drop_col_names, axis=1, inplace=True)
#    # Change NAN from -99 to 0
#    glac_table_copy[glac_table_copy==-99] = 0.
#    return glac_table_copy