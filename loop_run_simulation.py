#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 08:21:18 2018

@author: davidrounce
"""
from subprocess import call

#regions = [13, 14, 15]
regions = [15]
gcm_names = ['CanESM2', 'CCSM4', 'CNRM-CM5', 'GFDL-CM3', 'GFDL-ESM2M', 'GISS-E2-R', 'IPSL-CM5A-LR', 'IPSL-CM5A-MR', 
             'MIROC5', 'MRI-CGCM3', 'NorESM1-M']
rcps = ['rcp26', 'rcp45', 'rcp85']


for region in regions:
    for rcp in rcps:
        for gcm in gcm_names:
        
            # Append arguments to call list
            call_list = ["python", "run_simulation.py"]
            call_list.append('-spc_region=' + str(region))
            call_list.append("-gcm_name={}".format(gcm))
            call_list.append("-rcp={}".format(rcp))
            call_list.append("-option_parallels=0")
            call_list.append("-rgi_glac_number_fn=R" + str(region) + "_mauer_1970s_2000_rgi_glac_number.pkl")
            
            # Run script
            call(call_list)
