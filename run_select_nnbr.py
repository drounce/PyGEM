"""
run_select_nnbr.py selects the best parameter set for the uncalibrated glaciers based on the nearest neighbors.
Specifically, the script iterates through the nearest neighbors until the modeled mass balance using the reference 
climate dataset is within +/- 1 stdev from the mb_mwea statistics from the neighbors.
"""

import pandas as pd
import numpy as np
import time
from time import strftime
from sklearn.neighbors import NearestNeighbors

import pygem_input as input
import pygemfxns_modelsetup as modelsetup
#import pygemfxns_climate as climate
import pygemfxns_massbalance as massbalance
#import pygemfxns_output as output
import class_climate

#%% INPUT 
rgi_regionsO1 = [15]
rgi_glac_number = 'all'

startyear = 2000
endyear = 2015
spinupyears = 5
# Calibrated model parameters full filename (needs to include 'all' glaciers in a region)
cal_modelparams_fullfn = input.main_directory + '/../Output/20180710_cal_modelparams_opt1_R15_ERA-Interim_1995_2015.csv'
# Number of nearest neighbors
n_nbrs = 20
# Option to remove marine-terminating glaciers
option_cal_remove_marine_glaciers = 1

# Reference climate data
gcm_name = 'ERA-Interim'
option_gcm_downscale = 2
option_lapserate_fromgcm = 1

option_export = 1

time_start = time.time()

#%% ===== LOAD GLACIER DATA =====
# RGI glacier attributes
main_glac_rgi = modelsetup.selectglaciersrgitable(rgi_regionsO1=rgi_regionsO1, 
                                                  rgi_regionsO2='all', 
                                                  rgi_glac_number='all')
# Glacier hypsometry [km**2], total area
main_glac_hyps = modelsetup.import_Husstable(main_glac_rgi, rgi_regionsO1, input.hyps_filepath, 
                                             input.hyps_filedict, input.hyps_colsdrop)
elev_bins = main_glac_hyps.columns.values.astype(int)
# Ice thickness [m], average
main_glac_icethickness = modelsetup.import_Husstable(main_glac_rgi, rgi_regionsO1, input.thickness_filepath, 
                                                     input.thickness_filedict, input.thickness_colsdrop)
main_glac_hyps[main_glac_icethickness == 0] = 0
# Width [km], average
main_glac_width = modelsetup.import_Husstable(main_glac_rgi, rgi_regionsO1, input.width_filepath, 
                                              input.width_filedict, input.width_colsdrop)
# Add volume [km**3] and mean elevation [m a.s.l.] to the main glaciers table
main_glac_rgi['Volume'], main_glac_rgi['Zmean'] = modelsetup.hypsometrystats(main_glac_hyps, main_glac_icethickness)
# Model time frame
dates_table, start_date, end_date = modelsetup.datesmodelrun(startyear, endyear, spinupyears)
# Quality control - if ice thickness = 0, glacier area = 0 (problem identified by glacier RGIV6-15.00016 03/06/2018)
main_glac_hyps[main_glac_icethickness == 0] = 0

#%% ===== LOAD CLIMATE DATA =====
gcm = class_climate.GCM(name=gcm_name)
if option_gcm_downscale == 1:  
    # Air Temperature [degC] and GCM dates
    gcm_temp, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(gcm.temp_fn, gcm.temp_vn, main_glac_rgi, dates_table)
    # Precipitation [m] and GCM dates
    gcm_prec, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(gcm.prec_fn, gcm.prec_vn, main_glac_rgi, dates_table)
    # Elevation [m a.s.l] associated with air temperature  and precipitation data
    gcm_elev = gcm.importGCMfxnearestneighbor_xarray(gcm.elev_fn, gcm.elev_vn, main_glac_rgi)
    # Add GCM time series to the dates_table
    dates_table['date_gcm'] = gcm_dates
    # Lapse rates [degC m-1]
    if option_lapserate_fromgcm == 1:
        gcm_lr, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(gcm.lr_fn, gcm.lr_vn, main_glac_rgi, dates_table)
        
elif option_gcm_downscale == 2:
    # 32 seconds for Region 15
    # Import air temperature, precipitation, and elevation from pre-processed csv files for a given region
    #  this simply saves time from re-running the fxns above
    gcm_temp_all = np.genfromtxt(gcm.var_fp + input.gcmtemp_filedict[input.rgi_regionsO1[0]], delimiter=',')
    gcm_prec_all = np.genfromtxt(gcm.var_fp + input.gcmprec_filedict[input.rgi_regionsO1[0]], delimiter=',')
    gcm_elev_all = np.genfromtxt(gcm.fx_fp + input.gcmelev_filedict[input.rgi_regionsO1[0]], delimiter=',')
    # Lapse rates [degC m-1]  
    gcm_lr_all = np.genfromtxt(gcm.var_fp + input.gcmlapserate_filedict[input.rgi_regionsO1[0]], 
                                               delimiter=',')
    # Select the climate data for the glaciers included in the study
    gcm_temp = gcm_temp_all[main_glac_rgi['O1Index'].values]
    gcm_prec = gcm_prec_all[main_glac_rgi['O1Index'].values]
    gcm_elev = gcm_elev_all[main_glac_rgi['O1Index'].values]
    gcm_lr = gcm_lr_all[main_glac_rgi['O1Index'].values]

print('Loading time:', time.time()-time_start, 's')

#%% ===== NEAREST NEIGHBOR SELECTION =====
# The selection of nearest neighbors looks at the closest 20 pixels and looks at the 

# Load calibrated parameters
ds_cal = pd.read_csv(cal_modelparams_fullfn, index_col=0)
ds_cal['newidx'] = ds_cal['O1Index']
ds_cal.set_index('newidx', inplace=True, drop=True)

# Dictionary cal index and 'GlacNo'
cal_dict = dict(zip(np.arange(0,ds_cal.shape[0]), ds_cal.index.values))
# Dataframe of all glaciers with model parameters from nearest neighbors
main_glac_modelparamsopt_pd = pd.DataFrame(np.full([main_glac_rgi.shape[0], len(input.modelparams_colnames)], np.nan), 
                                           columns=input.modelparams_colnames)
main_glac_modelparamsopt_pd.index = main_glac_rgi.index.values
ds_cal_all = pd.concat([main_glac_rgi.copy(), main_glac_modelparamsopt_pd], axis=1)
ds_cal_all['mbclim_mwe'] = np.nan
# Fill in the values of the already calibrated glaciers
ds_cal_all = ds_cal_all.combine_first(ds_cal)
# Add columns describing nearest neighbors
ds_cal_all['nbr_idx'] = np.nan
ds_cal_all['nbr_idx_count'] = np.nan
ds_cal_all['nbr_mb_mean'] = np.nan
ds_cal_all['nbr_mb_std'] = np.nan
ds_cal_all['z_score'] = np.nan
nbr_idx_cols = []
for n in range(n_nbrs):
    nbr_col_name = 'nearidx_' + str(n+1)
    ds_cal_all[nbr_col_name] = np.nan
    nbr_idx_cols.append(nbr_col_name)

# AVOID MARINE-TERMINATING GLACIERS UNTIL FRONTAL ABLATION IS INCLUDED, i.e., climatic mass balance is separated
# Remove marine-terminating glaciers from calibration dataset such that they are not included in neighbor calculations
if option_cal_remove_marine_glaciers == 1:
    ds_cal = ds_cal[ds_cal.TermType != 1]

# Loop through each glacier and select the n_nbrs closest glaciers    
for glac in range(main_glac_rgi.shape[0]):
    # Select nnbrs only for uncalibrated glaciers
    if ds_cal_all.loc[glac, input.modelparams_colnames].isnull().values.any() == True:
        # Print every 100th glacier
#        if glac%500 == 0:
#            print(main_glac_rgi.loc[glac,'RGIId'])
        print(main_glac_rgi.loc[glac,'RGIId'])    
        # Select the lon/lat of the glacier
        glac_lonlat = np.zeros((1,2))
        glac_lonlat[:] = ds_cal_all.loc[glac,['CenLon','CenLat']].values
        # Append the lon/lat
        glac_lonlat_wcal = np.append(glac_lonlat, ds_cal.loc[:,['CenLon','CenLat']].values, axis=0)
        # Calculate nearest neighbors (set neighbors + 1 to account for itself)
        nbrs = NearestNeighbors(n_neighbors=n_nbrs+1, algorithm='brute').fit(glac_lonlat_wcal)
        distances_raw, indices_raw = nbrs.kneighbors(glac_lonlat_wcal)
        # Select glacier (row 0) and remove itself (col 0), so left with indices for nearest neighbors
        indices_raw2 = indices_raw[0,:][indices_raw[0,:] > 0] - 1
        indices = np.array([cal_dict[n] for n in indices_raw2])
        # Add indices to columns
        ds_cal_all.loc[glac, nbr_idx_cols] = indices            
        
        # Nearest neighbors: mass balance envelope
        nbrs_data = np.zeros((len(nbr_idx_cols),4))
        #  Col 0 - index count
        #  Col 1 - index value
        #  Col 2 - MB
        #  Col 3 - MB modeled using neighbor's parameter
        nbrs_data[:,0] = np.arange(1,len(nbr_idx_cols)+1) 
        nbrs_data[:,1] = ds_cal_all.loc[glac, nbr_idx_cols].values.astype(int)
        nbrs_data[:,2] = ds_cal_all.loc[nbrs_data[:,1], input.mbclim_cn]
        mb_nbrs_mean = nbrs_data[:,2].mean()
        mb_nbrs_std = nbrs_data[:,2].std()
        mb_envelope_lower = mb_nbrs_mean - mb_nbrs_std
        mb_envelope_upper = mb_nbrs_mean + mb_nbrs_std
    
        # Set nbr_idx_count to -1, since adds 1 at the start        
        nbr_idx_count = -1
        # Loop through nearest neighbors until find set of model parameters that returns MB in MB envelope
        # Break loop used to exit loop if unable to find parameter set that satisfies criteria
        break_while_loop = False
        while break_while_loop == False:
            # Nearest neighbor index
            nbr_idx_count = nbr_idx_count + 1
            # Check if cycled through all neighbors; if so, then choose neighbor with MB closest to the envelope
            if nbr_idx_count == len(nbr_idx_cols):
                break_while_loop = True
                mb_abs = np.zeros((nbrs_data.shape[0],2)) 
                mb_abs[:,0] = abs(nbrs_data[:,3] - mb_envelope_lower)
                mb_abs[:,1] = abs(nbrs_data[:,3] - mb_envelope_upper)
                nbr_idx_count = np.where(mb_abs == mb_abs.min())[0][0]    
            # Nearest neighbor index value
            nbr_idx = nbrs_data[nbr_idx_count,1]
            # Model parameters - apply transfer function adjustments
            modelparameters = ds_cal_all.loc[nbr_idx, input.modelparams_colnames]
            modelparameters['tempchange'] = (
                    modelparameters['tempchange'] + input.tempchange_lobf_slope * 
                    (main_glac_rgi.loc[glac, input.tempchange_lobf_property_cn] - 
                     ds_cal_all.loc[nbr_idx, input.tempchange_lobf_property_cn]))
            modelparameters['precfactor'] = (
                    modelparameters['precfactor'] + input.precfactor_lobf_slope * 
                    (main_glac_rgi.loc[glac, input.precfactor_lobf_property_cn] - 
                     ds_cal_all.loc[nbr_idx, input.precfactor_lobf_property_cn]))
            modelparameters['ddfsnow'] = (
                    modelparameters['ddfsnow'] + input.ddfsnow_lobf_slope * 
                    (main_glac_rgi.loc[glac, input.ddfsnow_lobf_property_cn] - 
                     ds_cal_all.loc[nbr_idx, input.ddfsnow_lobf_property_cn]))
            modelparameters['ddfice'] = modelparameters['ddfsnow'] / input.ddfsnow_iceratio
            modelparameters['precgrad'] = (
                    modelparameters['precgrad'] + input.precgrad_lobf_slope * 
                    (main_glac_rgi.loc[glac, input.precgrad_lobf_property_cn] - 
                     ds_cal_all.loc[nbr_idx, input.precgrad_lobf_property_cn]))
            # Select subsets of data
            glacier_rgi_table = main_glac_rgi.loc[glac, :]
            glacier_gcm_elev = gcm_elev[glac]
            glacier_gcm_prec = gcm_prec[glac,:]
            glacier_gcm_temp = gcm_temp[glac,:]
            glacier_gcm_lrgcm = gcm_lr[glac,:]
            glacier_gcm_lrglac = glacier_gcm_lrgcm.copy()
            glacier_area_t0 = main_glac_hyps.iloc[glac,:].values.astype(float)   
            icethickness_t0 = main_glac_icethickness.iloc[glac,:].values.astype(float)
            width_t0 = main_glac_width.iloc[glac,:].values.astype(float)
            # Avoid error where there is no glacier/ice thickness - give it parameters to avoid issues in simulation
            if glacier_area_t0.max() == 0:
                break_while_loop = True
                nbr_idx = nbrs_data[0,1]
                mbclim_mwe = np.nan
            # As long as there is ice present, then select parameters from nearest neighbors
            else:
                # MASS BALANCE
                # Run the mass balance function (spinup years have been removed from output)
                (glac_bin_temp, glac_bin_prec, glac_bin_acc, glac_bin_refreeze, glac_bin_snowpack, glac_bin_melt, 
                 glac_bin_frontalablation, glac_bin_massbalclim, glac_bin_massbalclim_annual, glac_bin_area_annual, 
                 glac_bin_icethickness_annual, glac_bin_width_annual, glac_bin_surfacetype_annual, 
                 glac_wide_massbaltotal, glac_wide_runoff, glac_wide_snowline, glac_wide_snowpack, 
                 glac_wide_area_annual, glac_wide_volume_annual, glac_wide_ELA_annual) = (
                    massbalance.runmassbalance(modelparameters, glacier_rgi_table, glacier_area_t0, icethickness_t0, 
                                               width_t0, elev_bins, glacier_gcm_temp, glacier_gcm_prec, 
                                               glacier_gcm_elev, glacier_gcm_lrgcm, glacier_gcm_lrglac, dates_table, 
                                               option_areaconstant=1))
                # Glacier-wide climatic mass balance [m w.e.] based on initial area
                mbclim_mwe = (
                        (glac_bin_massbalclim * glac_bin_area_annual[:, 0][:,np.newaxis]).sum() / 
                        glac_bin_area_annual[:, 0].sum())
                nbrs_data[nbr_idx_count,3] = mbclim_mwe
           
            # Prior to breaking loop print RGIId and z score
            if break_while_loop == True:
                # Compute z score and print out value
                z_score = (mbclim_mwe - mb_nbrs_mean) / mb_nbrs_std
                if abs(z_score) > 1.96:
                    print(glacier_rgi_table.RGIId, 'z_score:', mbclim_mwe, np.round(z_score,2))
               
            # If mass balance falls within envelope, then end while loop
            if (mbclim_mwe <= mb_envelope_upper) and (mbclim_mwe >= mb_envelope_lower):
                break_while_loop = True
                
            # Record results
            if break_while_loop == True:
                z_score = (mbclim_mwe - mb_nbrs_mean) / mb_nbrs_std
                ds_cal_all.loc[glac, input.mbclim_cn] = mbclim_mwe
                ds_cal_all.loc[glac, 'nbr_idx'] = nbr_idx
                ds_cal_all.loc[glac, 'nbr_idx_count'] = nbr_idx_count + 1
                ds_cal_all.loc[glac, 'nbr_mb_mean'] = mb_nbrs_mean
                ds_cal_all.loc[glac, 'nbr_mb_std'] = mb_nbrs_std
                ds_cal_all.loc[glac, 'z_score'] = z_score
                ds_cal_all.loc[glac, 'lrgcm'] = ds_cal_all.loc[nbr_idx, 'lrgcm']
                ds_cal_all.loc[glac, 'lrglac'] = ds_cal_all.loc[nbr_idx, 'lrglac']
                ds_cal_all.loc[glac, 'precfactor'] = modelparameters['precfactor']
                ds_cal_all.loc[glac, 'precgrad'] = ds_cal_all.loc[nbr_idx, 'precgrad']
                ds_cal_all.loc[glac, 'ddfsnow'] = modelparameters['ddfsnow']
                ds_cal_all.loc[glac, 'ddfice'] = modelparameters['ddfice']
                ds_cal_all.loc[glac, 'tempsnow'] = ds_cal_all.loc[nbr_idx, 'tempsnow']
                ds_cal_all.loc[glac, 'tempchange'] = modelparameters['tempchange']
                

if option_export == 1:
    output_fullfn = cal_modelparams_fullfn.replace('.csv', '_wnnbrs_' + str(strftime("%Y%m%d")) + '.csv')
    ds_cal_all.to_csv(output_fullfn, sep=',')

print('Processing time:', time.time()-time_start, 's')