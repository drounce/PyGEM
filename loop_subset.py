#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 2019
@author: davidrounce
"""
from subprocess import call

regions = [13, 14, 15]
#gcm_names = ['CSIRO-Mk3-6-0', 'GFDL-ESM2G',  'IPSL-CM5A-MR', 'MIROC-ESM', 'MIROC-ESM-CHEM', 'NorESM1-ME']
gcm_names = ['FGOALS-g2', 'HadGEM2-ES', 'MPI-ESM-LR', 'MPI-ESM-MR']
rcps = ['rcp26', 'rcp45', 'rcp85']
#rcps = ['rcp26', 'rcp45', 'rcp60', 'rcp85']
#output_fp = '/Users/davidrounce/Documents/Dave_Rounce/HiMAT/Output/simulations/spc_20190914/'
netcdf_fp = '/Volumes/LaCie/HMA_PyGEM/2019_0914/spc_zipped/'   

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
            
#            print(call_list)

            # Run script
            call(call_list)

        # ADD IN LOOP TO THEN MERGE THE LISTS IN THEIR ENTIRETY INTO A SINGLE DS!
        # DO THIS WITH A SEPARATE CALL...