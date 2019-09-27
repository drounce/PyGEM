#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 2019
@author: davidrounce
"""
from subprocess import call

regions = [13, 14, 15]
gcm_names = ['NorESM1-M']
#gcm_names = ['GFDL-CM3', 'GISS-E2-R', 'IPSL-CM5A-LR', 'MIROC5', 'MRI-CGCM3']
rcps = ['rcp26', 'rcp45', 'rcp85']
#rcps = ['rcp26', 'rcp45', 'rcp60', 'rcp85']
#output_fp = '/Users/davidrounce/Documents/Dave_Rounce/HiMAT/Output/simulations/spc_20190914/'
netcdf_fp = '/Volumes/LaCie/HMA_PyGEM/2019_0914/merged/'   

for gcm in gcm_names:
    
    ds_fp = netcdf_fp + gcm + '/'
    
    for rcp in rcps:
        for region in regions:

            # Append arguments to call list
            call_list = ["python", "run_postprocessing.py"]
            call_list.append("-gcm_name={}".format(gcm))
            call_list.append("-rcp={}".format(rcp))
            call_list.append('-region=' + str(region))
            call_list.append('-output_sim_fp={}'.format(ds_fp))
            call_list.append('-extract_subset=1')            
            call_list.append('-unzip_files=1')
            
            print(call_list)

            # Run script
            call(call_list)

        # ADD IN LOOP TO THEN MERGE THE LISTS IN THEIR ENTIRETY INTO A SINGLE DS!
        # DO THIS WITH A SEPARATE CALL...

#%%
#import os
#import pandas as pd
#import numpy as np
#import xarray as xr
##subset_fp = netcdf_fp + 'spc_subset/'
##for gcm in gcm_names:
##    for i in os.listdir(subset_fp):
##        if i.endswith('.nc') and gcm in i:
##            ds = xr.open_dataset(subset_fp + i)
##            vol_glac_all = ds.volume_glac_annual.values[:,:,0]
##            vol_remain_perc = vol_glac_all[:,vol_glac_all.shape[1]-1].sum() / vol_glac_all[:,0].sum() * 100
##            reg = i.split('--')[0]
##            rcp = i.split('_')[1]
##            gcm_name = i.split('--')[2].split('_')[0]
##            print(gcm_name, 'Region', reg, rcp, 'Vol remain [%]:', np.round(vol_remain_perc,1))
#
##ds = xr.open_dataset(subset_fp + 'R15--all--CCSM4_rcp26_c2_ba1_100sets_2000_2100--subset.nc')
##ds = xr.open_dataset('/Users/davidrounce/Documents/Dave_Rounce/HiMAT/Output/simulations/spc_20190914/merged/ERA-Interim/R15--all--ERA-Interim_c2_ba1_100sets_2000_2018.nc')
##ds = xr.open_dataset('/Users/davidrounce/Documents/Dave_Rounce/HiMAT/Output/R15--all--GFDL-CM3_rcp26_c2_ba1_100sets_2000_2100.nc')
#ds = xr.open_dataset('/Volumes/LaCie/HMA_PyGEM/2019_0914/merged/spc_subset/R13--all--MRI-CGCM3_rcp26_c2_ba1_100sets_2000_2100--subset.nc')
#df = pd.DataFrame(ds.glacier_table.values, columns=ds.glac_attrs.values)
#
#vol_glac_all = ds.volume_glac_annual.values[:,:,0]
#print('Vol remain [%]:', np.round(vol_glac_all[:,vol_glac_all.shape[1]-1].sum() / vol_glac_all[:,0].sum() * 100,1))
#vol_remain_perc = vol_glac_all[:,vol_glac_all.shape[1]-1] / vol_glac_all[:,0] * 100
#vol_remain_perc_threshold = 100
#print('% glaciers growing by more than ' + str(vol_remain_perc_threshold), ':', 
#      np.round(len(np.where(vol_remain_perc > vol_remain_perc_threshold)[0]) / df.shape[0] * 100,1))
#
#area_glac_all = ds.area_glac_annual.values[:,:,0]
#area_remaining = area_glac_all[:,-1]
#area_chg = area_glac_all[:,-1] - area_glac_all[:,0]
#area_chg_idx = np.where(area_chg > 20)[0]
#
#
#vol_threshold = 1
#print('Max volume:', vol_glac_all[:,-1].max())
#vol_max_idx = np.where(vol_glac_all[:,-1] > vol_threshold)[0]
#for lrg_idx in vol_max_idx:
#    print(lrg_idx, int(df.iloc[lrg_idx,0]), vol_glac_all[lrg_idx,0], vol_glac_all[lrg_idx,-1])
#
#area_initial = area_glac_all[:,0]
#A = area_initial - df['Area'].values
#
#mb = (vol_glac_all[:,-1] - vol_glac_all[:,0]) / area_glac_all[:,0] / 18 * 1000



