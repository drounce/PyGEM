#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 09:26:39 2023

@author: mweather btobers drounce
"""

# imports
import os
import glob
import sys
import time
import argparse
import xarray as xr
import numpy as np
import pygem
from datetime import datetime

# Local libraries
import pygem_input as pygem_prms
import pygem.pygem_modelsetup as modelsetup


rgi_reg_dict = {'all':'Global',
                'all_no519':'Global, excl. GRL and ANT',
                'global':'Global',
                1:'Alaska',
                2:'W Canada & US',
                3:'Arctic Canada North',
                4:'Arctic Canada South',
                5:'Greenland Periphery',
                6:'Iceland',
                7:'Svalbard',
                8:'Scandinavia',
                9:'Russian Arctic',
                10:'North Asia',
                11:'Central Europe',
                12:'Caucasus & Middle East',
                13:'Central Asia',
                14:'South Asia West',
                15:'South Asia East',
                16:'Low Latitudes',
                17:'Southern Andes',
                18:'New Zealand',
                19:'Antarctic & Subantarctic'
                }


def main(reg, simpath, gcm, scenario, bias_adj, gcm_startyear, gcm_endyear, vars):

    # #%% ----- PROCESS DATASETS FOR INDIVIDUAL GLACIERS AND ELEVATION BINS -----
    comppath = simpath + 'compile/'

    # define base directory
    base_dir = simpath + "/" + str(reg).zfill(2) + "/"

    # get all glaciers in region to see which fraction ran successfully
    main_glac_rgi_all = modelsetup.selectglaciersrgitable(rgi_regionsO1=[reg], 
                                                        rgi_regionsO2='all', rgi_glac_number='all', 
                                                        glac_no=None,
                                                        debug=True)

    glacno_list_all = list(main_glac_rgi_all['rgino_str'].values)

    ### CREATE BATCHES ###
    # get last glacier number to define number of batches
    lastn = int(sorted(glacno_list_all)[-1].split('.')[1])
    # round up to thosand
    batch_interval = 1000
    last_thous = np.ceil(lastn / batch_interval) * batch_interval
    # get number of batches
    nbatches = last_thous // batch_interval

    # split glaciers into groups of a thousand based on all glaciers in region
    glacno_list_batches = modelsetup.split_list(glacno_list_all, n=nbatches, group_thousands=True)

    # make sure batch sublists are sorted properly and that each goes from NN001 to N(N+1)000
    glacno_list_batches = sorted(glacno_list_batches, key=lambda x:x[0])
    for i in range(len(glacno_list_batches) - 1): 
        glacno_list_batches[i].append(glacno_list_batches[i+1][0])
        glacno_list_batches[i+1].pop(0)

    # open up a simulation file to get time for prepoluating aggregated data below - make sure gcm has scenario of interest
    if gcm:
        gcms = [gcm]
    else:
        gcms = os.listdir(base_dir)
    time_values = None
    while time_values is None:        
        for gcm in gcms:
            if scenario:
                if scenario in os.listdir(base_dir + '/' + gcm):
                    fn = glob.glob(base_dir + gcm  + "/" + scenario  + "/stats/" + '*.nc')[0]
                else:
                    # remove the gcm from our gcm list if the desired scenario is not contained
                    gcms.remove(gcm)
            else:
                fn = glob.glob(base_dir + gcm  + "/stats/" + '*.nc')[0]
            ds_glac = xr.open_dataset(fn)
            year_values = ds_glac.year.values
            time_values = ds_glac.time.values
            # store file ending
            file_ending = fn[-31:]
            # check if desired vars are in ds
            ds_vars = list(ds_glac.keys())
            missing_vars = list(set(vars) - set(ds_vars))
            if len(missing_vars) > 0:
                vars = list(set(vars).intersection(ds_vars))
                raise ValueError(f'Requested variables are missing: {missing_vars}')
    print(f'Compiling GCMS: {gcms}')
    print(f'Variables: {vars}')

    ### LEVEL I ###
    # loop through glacier batches of 1000
    for nbatch, glacno_list in enumerate(glacno_list_batches):
        print(f'Batch {nbatch}:')

        # batch start timer
        loop_start = time.time()

        # get batch start and end numbers
        batch_start = glacno_list[0].split('.')[1]
        batch_end = glacno_list[-1].split('.')[1]

        print(nbatch, batch_start, batch_end)

        # get all glacier info for glaciers in batch
        main_glac_rgi_batch = main_glac_rgi_all.loc[main_glac_rgi_all.apply(lambda x: x.rgino_str in glacno_list, axis=1)]

        # instantiate variables that will hold all concatenated data for GCM
        # monthly vars
        reg_glac_allgcms_runoff_monthly = None
        reg_offglac_allgcms_runoff_monthly = None
        reg_glac_allgcms_acc_monthly = None
        reg_glac_allgcms_melt_monthly = None
        reg_glac_allgcms_refreeze_monthly = None
        reg_glac_allgcms_frontalablation_monthly = None
        reg_glac_allgcms_massbaltotal_monthly = None
        reg_glac_allgcms_prec_monthly = None
        reg_glac_allgcms_mass_monthly = None

        # annual vars
        reg_glac_allgcms_area_annual = None
        reg_glac_allgcms_mass_annual = None    

        ### LEVEL II ###
        # for each batch, loop through GCMs
        for gcm in gcms:

            # get list of glacier simulation files 
            if scenario:
                sim_dir = base_dir + gcm  + '/' + scenario + '/stats/'
            else:
                sim_dir = base_dir + gcm  + '/stats/'

            fps = glob.glob(sim_dir + '*_ba' + str(bias_adj) + '_*' + str(gcm_startyear) + '_' + str(gcm_endyear) + '_all.nc')

            # during 0th batch, print the regional stats of glaciers and area successfully simulated for all regional glaciers for given gcm scenario
            if nbatch==0:
                # Glaciers with successful runs to process
                glacno_ran = [x.split('/')[-1].split('_')[0] for x in fps]
                glacno_ran = [x.split('.')[0].zfill(2) + '.' + x[-5:] for x in glacno_ran]
                main_glac_rgi = main_glac_rgi_all.loc[main_glac_rgi_all.apply(lambda x: x.rgino_str in glacno_ran, axis=1)]
                print(f'{gcm}, glaciers successfully simulated:\n  - {main_glac_rgi.shape[0]} of {main_glac_rgi_all.shape[0]} glaciers ({np.round(main_glac_rgi.shape[0]/main_glac_rgi_all.shape[0]*100,3)}%)')
                print(f'  - {np.round(main_glac_rgi.Area.sum(),0)} km2 of {np.round(main_glac_rgi_all.Area.sum(),0)} km2 ({np.round(main_glac_rgi.Area.sum()/main_glac_rgi_all.Area.sum()*100,3)}%)')

            # instantiate variables that will hold concatenated data for the current GCM
            # monthly vars
            reg_glac_gcm_runoff_monthly = None
            reg_offglac_gcm_runoff_monthly = None
            reg_glac_gcm_acc_monthly = None
            reg_glac_gcm_melt_monthly = None
            reg_glac_gcm_refreeze_monthly = None
            reg_glac_gcm_frontalablation_monthly = None
            reg_glac_gcm_massbaltotal_monthly = None
            reg_glac_gcm_prec_monthly = None
            reg_glac_gcm_mass_monthly = None

            # annual vars
            reg_glac_gcm_area_annual = None
            reg_glac_gcm_mass_annual = None    


            ### LEVEL III ###
            # loop through each glacier in batch list
            for i, glacno in enumerate(glacno_list):
                # get glacier string and file name
                glacier_str = '{0:0.5f}'.format(float(glacno))  
                if scenario:
                    glacno_fn = sim_dir + glacier_str + '_' + gcm + '_' + scenario + '_' + file_ending
                else:
                    glacno_fn = sim_dir + glacier_str + '_' + gcm + '_' + file_ending

                # try to load all glaciers in region
                try:
                    # open netcdf file
                    ds_glac = xr.open_dataset(glacno_fn)
                    # get monthly vars
                    glac_runoff_monthly = ds_glac.glac_runoff_monthly.values
                    offglac_runoff_monthly = ds_glac.offglac_runoff_monthly.values
                    # try extra vars
                    try:
                        glac_acc_monthly = ds_glac.glac_acc_monthly.values
                        glac_melt_monthly = ds_glac.glac_melt_monthly.values
                        glac_refreeze_monthly = ds_glac.glac_refreeze_monthly.values
                        glac_frontalablation_monthly = ds_glac.glac_frontalablation_monthly.values
                        glac_massbaltotal_monthly = ds_glac.glac_massbaltotal_monthly.values
                        glac_prec_monthly = ds_glac.glac_prec_monthly.values
                        glac_mass_monthly = ds_glac.glac_mass_monthly.values
                    except:
                        glac_acc_monthly = np.full((1,len(time_values)), np.nan)
                        glac_melt_monthly = np.full((1,len(time_values)), np.nan)
                        glac_refreeze_monthly = np.full((1,len(time_values)), np.nan)
                        glac_frontalablation_monthly = np.full((1,len(time_values)), np.nan)
                        glac_massbaltotal_monthly = np.full((1,len(time_values)), np.nan)
                        glac_prec_monthly = np.full((1,len(time_values)), np.nan)
                        glac_mass_monthly = np.full((1,len(time_values)), np.nan)
                    # get annual vars
                    glac_area_annual = ds_glac.glac_area_annual.values
                    glac_mass_annual = ds_glac.glac_mass_annual.values


                # if glacier output DNE in sim output file, create empty nan arrays to keep record of missing glaciers
                except:
                    # monthly vars
                    glac_runoff_monthly = np.full((1,len(time_values)), np.nan)
                    offglac_runoff_monthly = np.full((1,len(time_values)), np.nan)
                    glac_acc_monthly = np.full((1,len(time_values)), np.nan)
                    glac_melt_monthly = np.full((1,len(time_values)), np.nan)
                    glac_refreeze_monthly = np.full((1,len(time_values)), np.nan)
                    glac_frontalablation_monthly = np.full((1,len(time_values)), np.nan)
                    glac_massbaltotal_monthly = np.full((1,len(time_values)), np.nan)
                    glac_prec_monthly = np.full((1,len(time_values)), np.nan)
                    glac_mass_monthly = np.full((1,len(time_values)), np.nan)
                    # annual vars
                    glac_area_annual = np.full((1,year_values.shape[0]), np.nan)
                    glac_mass_annual = np.full((1,year_values.shape[0]), np.nan)

                
                # append each glacier output to master regional set of arrays
                if reg_glac_gcm_mass_annual is None:
                    # monthly vars
                    reg_glac_gcm_runoff_monthly = glac_runoff_monthly
                    reg_offglac_gcm_runoff_monthly = offglac_runoff_monthly
                    reg_glac_gcm_acc_monthly = glac_acc_monthly
                    reg_glac_gcm_melt_monthly = glac_melt_monthly
                    reg_glac_gcm_refreeze_monthly = glac_refreeze_monthly
                    reg_glac_gcm_frontalablation_monthly = glac_frontalablation_monthly
                    reg_glac_gcm_massbaltotal_monthly = glac_massbaltotal_monthly
                    reg_glac_gcm_prec_monthly = glac_prec_monthly
                    reg_glac_gcm_mass_monthly = glac_mass_monthly
                    # annual vars
                    reg_glac_gcm_area_annual = glac_area_annual
                    reg_glac_gcm_mass_annual = glac_mass_annual    
                # otherwise concatenate existing arrays
                else:
                    # monthly vars
                    reg_glac_gcm_runoff_monthly = np.concatenate((reg_glac_gcm_runoff_monthly, glac_runoff_monthly), axis=0)
                    reg_offglac_gcm_runoff_monthly = np.concatenate((reg_offglac_gcm_runoff_monthly, offglac_runoff_monthly), axis=0)
                    reg_glac_gcm_acc_monthly = np.concatenate((reg_glac_gcm_acc_monthly, glac_acc_monthly), axis=0)
                    reg_glac_gcm_melt_monthly = np.concatenate((reg_glac_gcm_melt_monthly, glac_melt_monthly), axis=0)
                    reg_glac_gcm_refreeze_monthly = np.concatenate((reg_glac_gcm_refreeze_monthly, glac_refreeze_monthly), axis=0)
                    reg_glac_gcm_frontalablation_monthly = np.concatenate((reg_glac_gcm_frontalablation_monthly, glac_frontalablation_monthly), axis=0)
                    reg_glac_gcm_massbaltotal_monthly = np.concatenate((reg_glac_gcm_massbaltotal_monthly, glac_massbaltotal_monthly), axis=0)
                    reg_glac_gcm_prec_monthly = np.concatenate((reg_glac_gcm_prec_monthly, glac_prec_monthly), axis=0)
                    reg_glac_gcm_mass_monthly = np.concatenate((reg_glac_gcm_mass_monthly, glac_mass_monthly), axis=0)
                    # annual vars
                    reg_glac_gcm_area_annual = np.concatenate((reg_glac_gcm_area_annual, glac_area_annual), axis=0)
                    reg_glac_gcm_mass_annual = np.concatenate((reg_glac_gcm_mass_annual, glac_mass_annual), axis=0)  

            # aggregate gcms
            if reg_glac_allgcms_runoff_monthly is None:
                # monthly vars
                reg_glac_allgcms_runoff_monthly = reg_glac_gcm_runoff_monthly[np.newaxis,:,:]
                reg_offglac_allgcms_runoff_monthly = reg_offglac_gcm_runoff_monthly[np.newaxis,:,:]
                reg_glac_allgcms_acc_monthly = reg_glac_gcm_acc_monthly[np.newaxis,:,:]
                reg_glac_allgcms_melt_monthly = reg_glac_gcm_melt_monthly[np.newaxis,:,:]
                reg_glac_allgcms_refreeze_monthly = reg_glac_gcm_refreeze_monthly[np.newaxis,:,:]
                reg_glac_allgcms_frontalablation_monthly = reg_glac_gcm_frontalablation_monthly[np.newaxis,:,:]
                reg_glac_allgcms_massbaltotal_monthly = reg_glac_gcm_massbaltotal_monthly[np.newaxis,:,:]
                reg_glac_allgcms_prec_monthly = reg_glac_gcm_prec_monthly[np.newaxis,:,:]
                reg_glac_allgcms_mass_monthly = reg_glac_gcm_mass_monthly[np.newaxis,:,:]
                # annual vars
                reg_glac_allgcms_area_annual = reg_glac_gcm_area_annual[np.newaxis,:,:]
                reg_glac_allgcms_mass_annual = reg_glac_gcm_mass_annual[np.newaxis,:,:]
            else:
                # monthly vrs
                reg_glac_allgcms_runoff_monthly  = np.concatenate((reg_glac_allgcms_runoff_monthly, reg_glac_gcm_runoff_monthly[np.newaxis,:,:]), axis=0)
                reg_offglac_allgcms_runoff_monthly  = np.concatenate((reg_offglac_allgcms_runoff_monthly, reg_offglac_gcm_runoff_monthly[np.newaxis,:,:]), axis=0)
                reg_glac_allgcms_acc_monthly  = np.concatenate((reg_glac_allgcms_acc_monthly, reg_glac_gcm_acc_monthly[np.newaxis,:,:]), axis=0)
                reg_glac_allgcms_melt_monthly  = np.concatenate((reg_glac_allgcms_melt_monthly, reg_glac_gcm_melt_monthly[np.newaxis,:,:]), axis=0)
                reg_glac_allgcms_refreeze_monthly  = np.concatenate((reg_glac_allgcms_refreeze_monthly, reg_glac_gcm_refreeze_monthly[np.newaxis,:,:]), axis=0)
                reg_glac_allgcms_frontalablation_monthly  = np.concatenate((reg_glac_allgcms_frontalablation_monthly, reg_glac_gcm_frontalablation_monthly[np.newaxis,:,:]), axis=0)
                reg_glac_allgcms_massbaltotal_monthly  = np.concatenate((reg_glac_allgcms_massbaltotal_monthly, reg_glac_gcm_massbaltotal_monthly[np.newaxis,:,:]), axis=0)
                reg_glac_allgcms_prec_monthly  = np.concatenate((reg_glac_allgcms_prec_monthly, reg_glac_gcm_prec_monthly[np.newaxis,:,:]), axis=0)
                reg_glac_allgcms_mass_monthly  = np.concatenate((reg_glac_allgcms_mass_monthly, reg_glac_gcm_mass_monthly[np.newaxis,:,:]), axis=0)
                # annual vars
                reg_glac_allgcms_area_annual  = np.concatenate((reg_glac_allgcms_area_annual, reg_glac_gcm_area_annual[np.newaxis,:,:]), axis=0)
                reg_glac_allgcms_mass_annual  = np.concatenate((reg_glac_allgcms_mass_annual, reg_glac_gcm_mass_annual[np.newaxis,:,:]), axis=0)

                            
        #===== CREATE NETCDF FILES=====
        rgiid_list = ['RGI60-' + x for x in glacno_list]
        cenlon_list = list(main_glac_rgi_batch.CenLon.values)
        cenlat_list = list(main_glac_rgi_batch.CenLat.values)
        attrs_dict = {'Region':str(reg) + ' - ' + rgi_reg_dict[reg],
                                    'source': f'PyGEMv{pygem.__version__}',
                                    'institution': 'Carnegie Mellon University',
                                    'history': f'Created by {pygem_prms.user_info["name"]} ({pygem_prms.user_info["email"]}) on ' + datetime.today().strftime('%Y-%m-%d'),
                                    'references': 'doi:10.3389/feart.2019.00331 and doi:10.1017/jog.2019.91',
                                    'Conventions': 'CF-1.9',
                                    'featureType': 'timeSeries'}

        for var in vars:
            #glac_runoff_monthly
            if var=='glac_runoff_monthly':
                ds = xr.Dataset(
                        data_vars=dict(
                                glac_runoff_monthly=(["model", "glacier", "time"], reg_glac_allgcms_runoff_monthly),
                                crs = np.nan
                                ),
                                coords=dict(
                                        RGIId=(["glacier"], rgiid_list),
                                        Climate_Model= (["model"], gcms),
                                        lon=(["glacier"], cenlon_list),
                                        lat=(["glacier"], cenlat_list),
                                        time=time_values,
                                ),
                                attrs=attrs_dict
                                )
                ds.glac_runoff_monthly.attrs['long_name'] = 'glacier-wide runoff'
                ds.glac_runoff_monthly.attrs['units'] = 'm3'
                ds.glac_runoff_monthly.attrs['temporal_resolution'] = 'monthly'
                ds.glac_runoff_monthly.attrs['comment'] = 'runoff from the glacier terminus, which moves over time'
                ds.glac_runoff_monthly.attrs['grid_mapping'] = 'crs'

            #offglac_runoff_monthly
            elif var=='offglac_runoff_monthly':
                ds = xr.Dataset(
                        data_vars=dict(
                                offglac_runoff_monthly=(["model", "glacier", "time"], reg_offglac_allgcms_runoff_monthly),
                                crs = np.nan
                                ),
                                coords=dict(
                                        RGIId=(["glacier"], rgiid_list),
                                        Climate_Model= (["model"], gcms),
                                        lon=(["glacier"], cenlon_list),
                                        lat=(["glacier"], cenlat_list),
                                        time=time_values,
                                ),
                                attrs=attrs_dict
                                )
                ds.offglac_runoff_monthly.attrs['long_name'] = 'off-glacier-wide runoff'
                ds.offglac_runoff_monthly.attrs['units'] = 'm3'
                ds.offglac_runoff_monthly.attrs['temporal_resolution'] = 'monthly'
                ds.offglac_runoff_monthly.attrs['comment'] = 'off-glacier runoff from area where glacier no longer exists'
                ds.offglac_runoff_monthly.attrs['grid_mapping'] = 'crs'

            #glac_acc_monthly
            elif var=='glac_acc_monthly':
                ds = xr.Dataset(
                        data_vars=dict(
                                glac_acc_monthly=(["model", "glacier", "time"], reg_glac_allgcms_acc_monthly),
                                crs = np.nan
                                ),
                                coords=dict(
                                        RGIId=(["glacier"], rgiid_list),
                                        Climate_Model= (["model"], gcms),
                                        lon=(["glacier"], cenlon_list),
                                        lat=(["glacier"], cenlat_list),
                                        time=time_values,
                                ),
                                attrs=attrs_dict
                                )
                ds.glac_acc_monthly.attrs['long_name'] = 'glacier-wide accumulation, in water equivalent'
                ds.glac_acc_monthly.attrs['units'] = 'm3'
                ds.glac_acc_monthly.attrs['temporal_resolution'] = 'monthly'
                ds.glac_acc_monthly.attrs['comment'] = 'only the solid precipitation'
                ds.glac_acc_monthly.attrs['grid_mapping'] = 'crs'

            #glac_melt_monthly
            elif var=='glac_melt_monthly':
                ds = xr.Dataset(
                        data_vars=dict(
                                glac_melt_monthly=(["model", "glacier", "time"], reg_glac_allgcms_melt_monthly),
                                crs = np.nan
                                ),
                                coords=dict(
                                        RGIId=(["glacier"], rgiid_list),
                                        Climate_Model= (["model"], gcms),
                                        lon=(["glacier"], cenlon_list),
                                        lat=(["glacier"], cenlat_list),
                                        time=time_values,
                                ),
                                attrs=attrs_dict
                                )
                ds.glac_melt_monthly.attrs['long_name'] = 'glacier-wide melt, in water equivalent'
                ds.glac_melt_monthly.attrs['units'] = 'm3'
                ds.glac_melt_monthly.attrs['temporal_resolution'] = 'monthly'
                ds.glac_melt_monthly.attrs['grid_mapping'] = 'crs'

            #glac_refreeze_monthly
            elif var=='glac_refreeze_monthly':
                ds = xr.Dataset(
                        data_vars=dict(
                                glac_refreeze_monthly=(["model", "glacier", "time"], reg_glac_allgcms_refreeze_monthly),
                                crs = np.nan
                                ),
                                coords=dict(
                                        RGIId=(["glacier"], rgiid_list),
                                        Climate_Model= (["model"], gcms),
                                        lon=(["glacier"], cenlon_list),
                                        lat=(["glacier"], cenlat_list),
                                        time=time_values,
                                ),
                                attrs=attrs_dict
                                )
                ds.glac_refreeze_monthly.attrs['long_name'] = 'glacier-wide refreeze, in water equivalent'
                ds.glac_refreeze_monthly.attrs['units'] = 'm3'
                ds.glac_refreeze_monthly.attrs['temporal_resolution'] = 'monthly'
                ds.glac_refreeze_monthly.attrs['grid_mapping'] = 'crs'

            #glac_frontalablation_monthly
            elif var=='glac_frontalablation_monthly':
                ds = xr.Dataset(
                        data_vars=dict(
                                glac_frontalablation_monthly=(["model", "glacier", "time"], reg_glac_allgcms_frontalablation_monthly),
                                crs = np.nan
                                ),
                                coords=dict(
                                        RGIId=(["glacier"], rgiid_list),
                                        Climate_Model= (["model"], gcms),
                                        lon=(["glacier"], cenlon_list),
                                        lat=(["glacier"], cenlat_list),
                                        time=time_values,
                                ),
                                attrs=attrs_dict
                                )
                ds.glac_frontalablation_monthly.attrs['long_name'] = 'glacier-wide frontal ablation, in water equivalent'
                ds.glac_frontalablation_monthly.attrs['units'] = 'm3'
                ds.glac_frontalablation_monthly.attrs['temporal_resolution'] = 'monthly'
                ds.glac_frontalablation_monthly.attrs['comment'] = 'mass losses from calving, subaerial frontal melting, \
                    sublimation above the waterline and subaqueous frontal melting below the waterline; \
                    positive values indicate mass lost like melt'
                ds.glac_frontalablation_monthly.attrs['grid_mapping'] = 'crs'

            #glac_massbaltotal_monthly
            elif var=='glac_massbaltotal_monthly':
                ds = xr.Dataset(
                        data_vars=dict(
                                glac_massbaltotal_monthly=(["model", "glacier", "time"], reg_glac_allgcms_massbaltotal_monthly),
                                crs = np.nan
                                ),
                                coords=dict(
                                        RGIId=(["glacier"], rgiid_list),
                                        Climate_Model= (["model"], gcms),
                                        lon=(["glacier"], cenlon_list),
                                        lat=(["glacier"], cenlat_list),
                                        time=time_values,
                                ),
                                attrs=attrs_dict
                                )
                ds.glac_massbaltotal_monthly.attrs['long_name'] = 'glacier-wide total mass balance, in water equivalent'
                ds.glac_massbaltotal_monthly.attrs['units'] = 'm3'
                ds.glac_massbaltotal_monthly.attrs['temporal_resolution'] = 'monthly'
                ds.glac_massbaltotal_monthly.attrs['comment'] = 'total mass balance is the sum of the climatic mass balance and frontal ablation'
                ds.glac_massbaltotal_monthly.attrs['grid_mapping'] = 'crs'


            #glac_prec_monthly
            elif var=='glac_prec_monthly':
                ds = xr.Dataset(
                        data_vars=dict(
                                glac_prec_monthly=(["model", "glacier", "time"], reg_glac_allgcms_prec_monthly),
                                crs = np.nan
                                ),
                                coords=dict(
                                        RGIId=(["glacier"], rgiid_list),
                                        Climate_Model= (["model"], gcms),
                                        lon=(["glacier"], cenlon_list),
                                        lat=(["glacier"], cenlat_list),
                                        time=time_values,
                                ),
                                attrs=attrs_dict
                                )
                ds.glac_prec_monthly.attrs['long_name'] = 'glacier-wide precipitation (liquid)'
                ds.glac_prec_monthly.attrs['units'] = 'm3'
                ds.glac_prec_monthly.attrs['temporal_resolution'] = 'monthly'
                ds.glac_prec_monthly.attrs['comment'] = 'only the liquid precipitation, solid precipitation excluded'
                ds.glac_prec_monthly.attrs['grid_mapping'] = 'crs'
    
            #glac_mass_monthly
            elif var=='glac_mass_monthly':
                ds = xr.Dataset(
                        data_vars=dict(
                                glac_mass_monthly=(["model", "glacier", "time"], reg_glac_allgcms_mass_monthly),
                                crs = np.nan
                                ),
                                coords=dict(
                                        RGIId=(["glacier"], rgiid_list),
                                        Climate_Model= (["model"], gcms),
                                        lon=(["glacier"], cenlon_list),
                                        lat=(["glacier"], cenlat_list),
                                        time=time_values,
                                ),
                                attrs=attrs_dict
                                )
                ds.glac_mass_monthly.attrs['long_name'] = 'glacier mass'
                ds.glac_mass_monthly.attrs['units'] = 'kg'
                ds.glac_mass_monthly.attrs['temporal_resolution'] = 'monthly'
                ds.glac_mass_monthly.attrs['comment'] = 'mass of ice based on area and ice thickness at start of the year and the monthly total mass balance'
                ds.glac_mass_monthly.attrs['grid_mapping'] = 'crs'

            #glac_area_annual
            elif var=='glac_area_annual':
                ds = xr.Dataset(
                        data_vars=dict(
                                glac_area_annual=(["model", "glacier", "time"], reg_glac_allgcms_area_annual),
                                crs = np.nan
                                ),
                                coords=dict(
                                        RGIId=(["glacier"], rgiid_list),
                                        Climate_Model= (["model"], gcms),
                                        lon=(["glacier"], cenlon_list),
                                        lat=(["glacier"], cenlat_list),
                                        time=year_values,
                                ),
                                attrs=attrs_dict
                                )
                ds.glac_area_annual.attrs['long_name'] = 'glacier area'
                ds.glac_area_annual.attrs['units'] = 'm2'
                ds.glac_area_annual.attrs['temporal_resolution'] = 'annual'
                ds.glac_area_annual.attrs['comment'] = 'area at start of the year'
                ds.glac_area_annual.attrs['grid_mapping'] = 'crs'

            #glac_mass_annual
            elif var=='glac_mass_annual':
                ds = xr.Dataset(
                        data_vars=dict(
                                glac_mass_annual=(["model", "glacier", "time"], reg_glac_allgcms_mass_annual),
                                crs = np.nan
                                ),
                                coords=dict(
                                        RGIId=(["glacier"], rgiid_list),
                                        Climate_Model= (["model"], gcms),
                                        lon=(["glacier"], cenlon_list),
                                        lat=(["glacier"], cenlat_list),
                                        time=year_values,
                                ),
                                attrs=attrs_dict
                                )
                ds.glac_mass_annual.attrs['long_name'] = 'glacier mass'
                ds.glac_mass_annual.attrs['units'] = 'kg'
                ds.glac_mass_annual.attrs['temporal_resolution'] = 'annual'
                ds.glac_mass_annual.attrs['comment'] = 'mass of ice based on area and ice thickness at start of the year'
                ds.glac_mass_annual.attrs['grid_mapping'] = 'crs'
        
            # crs attributes - same for all vars
            ds.crs.attrs['grid_mapping_name'] = 'latitude_longitude'
            ds.crs.attrs['longitude_of_prime_meridian'] = 0.0
            ds.crs.attrs['semi_major_axis'] = 6378137.0
            ds.crs.attrs['inverse_flattening'] = 298.257223563
            ds.crs.attrs['proj4text'] = '+proj=longlat +datum=WGS84 +no_defs'
            ds.crs.attrs['crs_wkt'] = 'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]]'
            
            # time attributes - different for monthly v annual
            ds.time.attrs['long_name'] = 'time'
            if 'annual' in var:
                ds.time.attrs['range'] = str(year_values[0]) + ' - ' + str(year_values[-1])
                ds.time.attrs['comment'] = 'years referring to the start of each year'
            elif 'monthly' in var:
                ds.time.attrs['range'] = str(time_values[0]) + ' - ' + str(time_values[-1])
                ds.time.attrs['comment'] = 'start of the month'
            
            ds.RGIId.attrs['long_name'] = 'Randolph Glacier Inventory Id'
            ds.RGIId.attrs['comment'] = 'RGIv6.0 (https://nsidc.org/data/nsidc-0770/versions/6)'
            ds.RGIId.attrs['cf_role'] = 'timeseries_id'

            ds.Climate_Model.attrs['long_name'] = 'General Circulation Model'
            
            ds.lon.attrs['standard_name'] = 'longitude'
            ds.lon.attrs['long_name'] = 'longitude of glacier center'
            ds.lon.attrs['units'] = 'degrees_east'
            
            ds.lat.attrs['standard_name'] = 'latitude'
            ds.lat.attrs['long_name'] = 'latitude of glacier center'
            ds.lat.attrs['units'] = 'degrees_north'
            
            
            nsidc_glac_fp = comppath + '/glacier_stats/' + var + '/' + str(reg).zfill(2) + '/'
            if not os.path.exists(nsidc_glac_fp):
                os.makedirs(nsidc_glac_fp, exist_ok=True)
            
            if scenario:
                ds_fn = ('R' + str(reg).zfill(2) + '_' + var + '_' +
                        scenario + '_Batch-' + str(batch_start) + '-' + str(batch_end) + '_' + file_ending)
            else:
                ds_fn = ('R' + str(reg).zfill(2) + '_' + var + '_' +
                        gcm + '_Batch-' + str(batch_start) + '-' + str(batch_end) + '_' + file_ending)

            ds.to_netcdf(nsidc_glac_fp + ds_fn)

        loop_end = time.time()
        print(f'Batch {nbatch} runtime:\t{np.round(loop_end - loop_start,2)} seconds')
            
            
    ### MERGE BATCHES FOR ANNUAL VARS ###
    vns = ['glac_mass_annual', 'glac_area_annual']

    for vn in vns:
        if vn in vars:
            vn_fp = comppath + 'glacier_stats/' + vn + '/' + str(reg).zfill(2) + '/'

            fn_merge_list = []
            fn_merge_list_start = []
            for i in os.listdir(vn_fp):
                if i.endswith('.nc') and 'Batch' in i and file_ending in i:
                    fn_merge_list.append(i)
                    fn_merge_list_start.append(int(i.split('-')[-2]))
        
            if len(fn_merge_list) > 0:
                fn_merge_list = [x for _,x in sorted(zip(fn_merge_list_start,fn_merge_list))]
            
                ds = None
                for fn in fn_merge_list:
                    ds_batch = xr.open_dataset(vn_fp + fn)
                    
                    if ds is None:
                        ds = ds_batch
                    else:
                        ds = xr.concat([ds, ds_batch], dim="glacier")
                
                ds_fn = fn.split('Batch')[0][:-1] + '_' + file_ending
                ds.to_netcdf(vn_fp + ds_fn)
                
                ds_batch.close()
                
                for fn in fn_merge_list:
                    os.remove(vn_fp + fn)


if __name__=='__main__':
    start = time.time()

    # Set up CLI
    parser = argparse.ArgumentParser(
    description="""description: program for compiling regional stats from the python glacier evolution model (PyGEM)\n\nexample call: $python compile_simulations -rgi_region=<##> -scenario=<SCENARIO> -simpath=</path/to/sims/> -gcm_startyear=<YYYY> -gcm_endyear=<YYYY>""",
    formatter_class=argparse.RawTextHelpFormatter)
    requiredNamed = parser.add_argument_group('required named arguments')
    requiredNamed.add_argument('-rgi_region01', type=int, help='Randoph Glacier Inventory 01 regions list', nargs='+', required=True)
    parser.add_argument('-gcm_name', type=str, default=None, help='GCM name to compile results from (ex. ERA5 or CESM2)')
    parser.add_argument('-scenario', type=str, default=None, help='rcp or ssp scenario used for model run (ex. rcp26 or ssp585)')
    parser.add_argument('-gcm_startyear', type=int, default=pygem_prms.gcm_bc_startyear, help='Global Climate Model start year for simulations (ex. 2000)')
    parser.add_argument('-gcm_endyear', type=int, default=pygem_prms.gcm_endyear, help='Global Circulation Model end year for simulations (ex. 2100)')
    parser.add_argument('-sim_path', type=str, default=pygem_prms.output_filepath + '/simulations/', help='PyGEM simulations filepath')
    parser.add_argument('-bias_adj', type=int, default=pygem_prms.option_bias_adjustment, help='bias adjustment type (ex. 1)')
    parser.add_argument('-vars',type=str, help='comm delimited list of PyGEM variables to compile (ex. "monthly_mass","annual_area")', 
                        choices=['glac_runoff_monthly','offglac_runoff_monthly','glac_acc_monthly','glac_melt_monthly','glac_refreeze_monthly','glac_frontalablation_monthly','glac_massbaltotal_monthly','glac_prec_monthly','glac_mass_monthly','glac_mass_annual','glac_area_annual'],
                        nargs='+')
    args = parser.parse_args()

    simpath = args.sim_path
    region = args.rgi_region01
    gcm = args.gcm_name
    scenario = args.scenario
    bias_adj = args.bias_adj
    gcm_startyear = args.gcm_startyear
    gcm_endyear = args.gcm_endyear
    vars = args.vars

    if gcm in ['ERA5', 'ERA-Interim', 'COAWST']:
        scenario = None
        bias_adj = 0
        gcm_startyear = pygem_prms.gcm_startyear

    if scenario:
        gcm = None

    if not simpath:
        simpath = pygem_prms.output_filepath + 'simulations/'

    if not os.path.exists(simpath + 'compile/'):
        os.makedirs(simpath + 'compile/')

    if not isinstance(region, list):
        region = [region]

    if not vars:
        vars = ['glac_runoff_monthly','offglac_runoff_monthly','glac_acc_monthly','glac_melt_monthly','glac_refreeze_monthly','glac_frontalablation_monthly','glac_massbaltotal_monthly','glac_prec_monthly','glac_mass_monthly','glac_mass_annual','glac_area_annual']

    for reg in region:
        main(reg, simpath, gcm, scenario, bias_adj, gcm_startyear, gcm_endyear, vars)

    end = time.time()
    print(f'Total runtime: {np.round(end - start,2)} seconds')
