r"""
run_calibration_list_multiprocess.py runs calibration for glaciers and stores results in csv files.  The script runs 
using the reference climate data.
    
    (Command line) python run_calibration_list_multiprocess.py 
      - Default is running ERA-Interim in parallel with five processors.

    (Spyder) %run run_calibration_list_multiprocess.py -option_parallels=0
      - Spyder cannot run parallels, so always set -option_parallels=0 when testing in Spyder.
      
"""

import pandas as pd
import numpy as np
import os
import argparse
import inspect
#import subprocess as sp
import multiprocessing
from scipy.optimize import minimize
import time
import matplotlib.pyplot as plt
from time import strftime
import xarray as xr
import netCDF4 as nc

import pygem_input as input
import pygemfxns_modelsetup as modelsetup
import pygemfxns_massbalance as massbalance
import pygemfxns_output as output
import class_climate
import class_mbdata

#%% ===== SCRIPT SPECIFIC INPUT DATA ===== 
# Glacier selection
rgi_regionsO1 = [15]
#rgi_glac_number = 'all'
rgi_glac_number = ['03473']
#rgi_glac_number = ['03473', '03733']

# Required input
gcm_name = input.ref_gcm_name
gcm_startyear = 2000
gcm_endyear = 2015
gcm_spinupyears = 5

# Calibration datasets
#cal_datasets = ['shean', 'wgms_ee']
cal_datasets = ['shean']
#cal_datasets = ['wgms_ee']

## Export option
#option_export = 1
#output_filepath = input.main_directory + '/../Output/'


#%% ===== LOAD GLACIER DATA ===== 
#  'raw' refers to the glacier subset that includes glaciers with and without calibration data
#  after the calibration data has been imported, then all glaciers without data will be dropped
# Glacier RGI data
main_glac_rgi_raw = modelsetup.selectglaciersrgitable(rgi_regionsO1=rgi_regionsO1, rgi_regionsO2 = 'all', 
                                                      rgi_glac_number=rgi_glac_number)
# Glacier hypsometry [km**2], total area
main_glac_hyps_raw = modelsetup.import_Husstable(main_glac_rgi_raw, rgi_regionsO1, input.hyps_filepath, 
                                                 input.hyps_filedict, input.hyps_colsdrop)
# Ice thickness [m], average
main_glac_icethickness_raw = modelsetup.import_Husstable(main_glac_rgi_raw, rgi_regionsO1, input.thickness_filepath, 
                                                         input.thickness_filedict, input.thickness_colsdrop)
main_glac_hyps_raw[main_glac_icethickness_raw == 0] = 0
# Width [km], average
main_glac_width_raw = modelsetup.import_Husstable(main_glac_rgi_raw, rgi_regionsO1, input.width_filepath, 
                                                  input.width_filedict, input.width_colsdrop)
elev_bins = main_glac_hyps_raw.columns.values.astype(int)
# Volume [km**3] and mean elevation [m a.s.l.]
main_glac_rgi_raw['Volume'], main_glac_rgi_raw['Zmean'] = (
        modelsetup.hypsometrystats(main_glac_hyps_raw, main_glac_icethickness_raw))
# Select dates including future projections
#  - nospinup dates_table needed to get the proper time indices
dates_table_nospinup, start_date, end_date = modelsetup.datesmodelrun(startyear=gcm_startyear, endyear=gcm_endyear, 
                                                                      spinupyears=0)
dates_table, start_date, end_date = modelsetup.datesmodelrun(startyear=gcm_startyear, endyear=gcm_endyear, 
                                                             spinupyears=gcm_spinupyears)

# ===== LOAD CALIBRATION DATA =====
cal_data = pd.DataFrame()
for dataset in cal_datasets:
    cal_subset = class_mbdata.MBData(name=dataset)
    cal_subset_data = cal_subset.masschange_total(main_glac_rgi_raw, main_glac_hyps_raw, dates_table_nospinup)
    cal_data = cal_data.append(cal_subset_data, ignore_index=True)
cal_data = cal_data.sort_values(['glacno', 't1_idx'])
cal_data.reset_index(drop=True, inplace=True)

# Drop glaciers that do not have any calibration data
main_glac_rgi = ((main_glac_rgi_raw.iloc[np.where(
        main_glac_rgi_raw[input.rgi_O1Id_colname].isin(cal_data['glacno']) == True)[0],:]).copy())
main_glac_hyps = main_glac_hyps_raw.iloc[main_glac_rgi.index.values]
main_glac_icethickness = main_glac_icethickness_raw.iloc[main_glac_rgi.index.values]  
main_glac_width = main_glac_width_raw.iloc[main_glac_rgi.index.values]
# Reset index    
main_glac_rgi.reset_index(drop=True, inplace=True)
main_glac_hyps.reset_index(drop=True, inplace=True)
main_glac_icethickness.reset_index(drop=True, inplace=True)
main_glac_width.reset_index(drop=True, inplace=True)

# ===== LOAD CLIMATE DATA =====
gcm = class_climate.GCM(name=gcm_name)
# Air temperature [degC]
gcm_temp, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(gcm.temp_fn, gcm.temp_vn, main_glac_rgi, dates_table)
# Precipitation [m]
gcm_prec, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(gcm.prec_fn, gcm.prec_vn, main_glac_rgi, dates_table)
# Elevation [m asl]
gcm_elev = gcm.importGCMfxnearestneighbor_xarray(gcm.elev_fn, gcm.elev_vn, main_glac_rgi)  
# Lapse rate
if gcm_name == 'ERA-Interim':
    gcm_lr, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(gcm.lr_fn, gcm.lr_vn, main_glac_rgi, dates_table)
else:
    # Mean monthly lapse rate
    ref_lr_monthly_avg = np.genfromtxt(gcm.lr_fp + gcm.lr_fn, delimiter=',')
    gcm_lr = np.tile(ref_lr_monthly_avg, int(gcm_temp.shape[1]/12))
    
# ===== CALIBRATION =====    
# Insert for loop here
# Glacier number
glac = 0
# Select subsets of data
glacier_rgi_table = main_glac_rgi.loc[main_glac_rgi.index.values[glac], :]
glacier_gcm_elev = gcm_elev[glac]
glacier_gcm_prec = gcm_prec[glac,:]
glacier_gcm_temp = gcm_temp[glac,:]
glacier_gcm_lrgcm = gcm_lr[glac,:]
glacier_gcm_lrglac = glacier_gcm_lrgcm.copy()
glacier_area_t0 = main_glac_hyps.iloc[glac,:].values.astype(float)   
icethickness_t0 = main_glac_icethickness.iloc[glac,:].values.astype(float)
width_t0 = main_glac_width.iloc[glac,:].values.astype(float)
glacier_cal_data = ((cal_data.iloc[np.where(
        glacier_rgi_table[input.rgi_O1Id_colname] == cal_data['glacno'])[0],:]).copy())

# Set model parameters
modelparameters = [input.lrgcm, input.lrglac, input.precfactor, input.precgrad, input.ddfsnow, input.ddfice, 
                   input.tempsnow, input.tempchange]

# Mass balance calculations
(glac_bin_temp, glac_bin_prec, glac_bin_acc, glac_bin_refreeze, glac_bin_snowpack, glac_bin_melt, 
 glac_bin_frontalablation, glac_bin_massbalclim, glac_bin_massbalclim_annual, glac_bin_area_annual, 
 glac_bin_icethickness_annual, glac_bin_width_annual, glac_bin_surfacetype_annual, 
 glac_wide_massbaltotal, glac_wide_runoff, glac_wide_snowline, glac_wide_snowpack, 
 glac_wide_area_annual, glac_wide_volume_annual, glac_wide_ELA_annual) = (
    massbalance.runmassbalance(modelparameters, glacier_rgi_table, glacier_area_t0, icethickness_t0, 
                               width_t0, elev_bins, glacier_gcm_temp, glacier_gcm_prec, 
                               glacier_gcm_elev, glacier_gcm_lrgcm, glacier_gcm_lrglac, dates_table, 
                               option_areaconstant=1))  

#model_glac_mb = glac_wide_massbaltotal.sum()