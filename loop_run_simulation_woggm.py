#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 2019
@author: davidrounce
"""
import os
from subprocess import call

#region = [15]
# GCMs and RCP/SSP scenarios
#gcm_names = ['CanESM2', 'CCSM4', 'CNRM-CM5', 'CSIRO-Mk3-6-0', 'GFDL-CM3', 
#             'GFDL-ESM2M', 'GISS-E2-R', 'IPSL-CM5A-LR', 'MPI-ESM-LR', 'NorESM1-M']
#rcps = ['rcp26', 'rcp45', 'rcp85']

#gcm_names = ['BCC-CSM2-MR', 'CESM2', 'CESM2-WACCM', 'EC-Earth3', 'EC-Earth3-Veg', 'FGOALS-f3-L', 
#             'GFDL-ESM4', 'INM-CM4-8', 'INM-CM5-0', 'MPI-ESM1-2-HR', 'MRI-ESM2-0', 'NorESM2-MM']
#rcps = ['ssp126', 'ssp245', 'ssp370', 'ssp585']

gcm_names = ['EC-Earth3', 'EC-Earth3-Veg', 'GFDL-ESM4', 'MRI-ESM2-0']
rcps = ['ssp119']

#%% Using input file
for gcm in gcm_names:
    for rcp in rcps:

        # Append arguments to call list
        call_list = ["python", "run_simulation_woggm.py"]
        call_list.append("-gcm_name={}".format(gcm))
        call_list.append("-scenario={}".format(rcp))
        call_list.append('-option_parallels=0')
        
#        print(call_list)

        # Run script
        call(call_list)
        
        
#%% Using batches
#region = [15]

#call_list_split = ['python', 'spc_split_glaciers.py', '-n_batches=1', '-option_ordered=0']
#call(call_list_split)
#
#check_str="R" + str(region[0]) + "_rgi_glac_number_batch"
#for i in os.listdir():
#    if i.startswith(check_str):
#        rgi_pkl_fn = i
#
#for gcm in gcm_names:
#    for rcp in rcps:
#
#        # Append arguments to call list
#        call_list = ["python", "run_simulation_woggm.py"]
#        call_list.append("-gcm_name={}".format(gcm))
#        call_list.append("-scenario={}".format(rcp))
#        call_list.append('-option_parallels=0')
#        call_list.append("-rgi_glac_number_fn={}".format(rgi_pkl_fn))
#        
##        print(call_list)
#
#        # Run script
#        call(call_list)