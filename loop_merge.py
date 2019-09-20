#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 08:21:18 2018
@author: davidrounce
"""
import os
from subprocess import call
import spc_split_glaciers as split_glaciers

# regions = [13, 14]
# gcm_names = ['CCSM4']
# rcps = ['rcp26']
regions = [13, 14, 15]
gcm_names = ['CCSM4']
rcps = ['rcp45', 'rcp60', 'rcp85']
output_fp = '/Users/davidrounce/Documents/Dave_Rounce/HiMAT/Output/simulations/spc_20190914/'
nchunks = 10        


for gcm in gcm_names:
    for rcp in rcps:

        ds_fp = output_fp + gcm + '/' + rcp + '/'

        for region in regions:

            if region == 15:
                nchunks = 10
            if region == 14:
                nchunks = 15
            elif region == 13:
                nchunks = 25


            # Glacier numbers
            glac_no = []
            for i in os.listdir(ds_fp):
                if i.endswith('.nc') and i.startswith(str(region)):
                    glac_no.append(i.split('_')[0])
                    if len(glac_no) == 1:
                        ds_ending = i.replace(i.split('_')[0],'')
            glac_no = sorted(glac_no)

            # Glacier number lists to pass for parallel processing
            glac_no_lsts = split_glaciers.split_list(glac_no, n=nchunks)


            for nchunk, chunk in enumerate(glac_no_lsts):
                print(nchunk, chunk[0], chunk[-1])
                chunk_start = glac_no.index(chunk[0])
                chunk_end = glac_no.index(chunk[-1])

                # Append arguments to call list
                call_list = ["python", "merge_ds.py"]
                call_list.append('-region=' + str(region))
                call_list.append("-gcm_name={}".format(gcm))
                call_list.append("-rcp={}".format(rcp))
                call_list.append('-chunk_no=' + str(nchunk))
                call_list.append('-chunk_start=' + str(chunk_start))
                call_list.append('-chunk_end=' + str(chunk_end))

                # Run script
                call(call_list)

            # ADD IN LOOP TO THEN MERGE THE LISTS IN THEIR ENTIRETY INTO A SINGLE DS!
            # DO THIS WITH A SEPARATE CALL...
