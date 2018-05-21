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
import pygemfxns_climate as climate
import pygemfxns_massbalance as massbalance
#import pygemfxns_output as output

#%% INPUT 
rgi_regionsO1 = [15]
rgi_glac_number = 'all'

startyear = 2000
endyear = 2015
spinupyears = 5
# Calibrated model parameters full filename (needs to include 'all' glaciers in a region)
cal_modelparams_fullfn = input.main_directory + '/../Output/calibration_R15_20180403_Opt02solutionspaceexpanding.csv'
modelparams_colnames = ['lrgcm', 'lrglac', 'precfactor', 'precgrad', 'ddfsnow', 'ddfice', 'tempsnow', 'tempchange']
# Number of nearest neighbors
n_nbrs = 20

# Reference climate data
option_gcm_downscale = 2
option_lapserate_fromgcm = 1
gcm_filepath_var = input.main_directory + '/../Climate_data/ERA_Interim/'
gcm_filepath_fx = input.main_directory + '/../Climate_data/ERA_Interim/'
gcm_temp_filename = 'ERAInterim_AirTemp2m_DailyMeanMonthly_1995_2016.nc'
gcm_temp_varname = 't2m'
gcm_prec_filename = 'ERAInterim_TotalPrec_DailyMeanMonthly_1979_2017.nc'
gcm_prec_varname = 'tp'
gcm_elev_filename = 'ERAInterim_geopotential.nc'
gcm_elev_varname = 'z'
gcm_lapserate_filename = 'HMA_Regions13_14_15_ERAInterim_lapserates_1979_2017.nc' # GENERATED IN PRE-PROCESSING
gcm_lapserate_varname = 'lapserate'
gcm_lat_varname = 'latitude'
gcm_lon_varname = 'longitude'
gcm_time_varname = 'time'

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
if option_gcm_downscale == 1:  
    # Air Temperature [degC] and GCM dates
    main_glac_gcmtemp, main_glac_gcmdate = climate.importGCMvarnearestneighbor_xarray(
            gcm_temp_filename, gcm_temp_varname, main_glac_rgi, dates_table, start_date, end_date,
            filepath=gcm_filepath_var, gcm_lon_varname=gcm_lon_varname, gcm_lat_varname=gcm_lat_varname)
    # Precipitation [m] and GCM dates
    main_glac_gcmprec, main_glac_gcmdate = climate.importGCMvarnearestneighbor_xarray(
            gcm_prec_filename, gcm_prec_varname, main_glac_rgi, dates_table, start_date, end_date,
            filepath=gcm_filepath_var, gcm_lon_varname=gcm_lon_varname, gcm_lat_varname=gcm_lat_varname)
    # Elevation [m a.s.l] associated with air temperature  and precipitation data
    main_glac_gcmelev = climate.importGCMfxnearestneighbor_xarray(
            gcm_elev_filename, gcm_elev_varname, main_glac_rgi,
            filepath=gcm_filepath_fx, gcm_lon_varname=gcm_lon_varname, gcm_lat_varname=gcm_lat_varname)
    # Add GCM time series to the dates_table
    dates_table['date_gcm'] = main_glac_gcmdate
    # Lapse rates [degC m-1]  
    if option_lapserate_fromgcm == 1:
        main_glac_gcmlapserate, main_glac_gcmdate = climate.importGCMvarnearestneighbor_xarray(
                gcm_lapserate_filename, gcm_lapserate_varname, main_glac_rgi, dates_table, start_date, end_date,
                filepath=gcm_filepath_var, gcm_lon_varname=gcm_lon_varname, gcm_lat_varname=gcm_lat_varname)
elif option_gcm_downscale == 2:
    # Import air temperature, precipitation, and elevation from pre-processed csv files for a given region
    #  this simply saves time from re-running the fxns above
    main_glac_gcmtemp_all = np.genfromtxt(gcm_filepath_var + input.gcmtemp_filedict[input.rgi_regionsO1[0]], 
                                          delimiter=',')
    main_glac_gcmprec_all = np.genfromtxt(gcm_filepath_var + input.gcmprec_filedict[input.rgi_regionsO1[0]], 
                                          delimiter=',')
    main_glac_gcmelev_all = np.genfromtxt(gcm_filepath_var + input.gcmelev_filedict[input.rgi_regionsO1[0]], 
                                          delimiter=',')
    # Lapse rates [degC m-1]  
    main_glac_gcmlapserate_all = np.genfromtxt(gcm_filepath_var + 
                                               input.gcmlapserate_filedict[input.rgi_regionsO1[0]], delimiter=',')
    # Select the climate data for the glaciers included in the study
    main_glac_gcmtemp = main_glac_gcmtemp_all[main_glac_rgi['O1Index'].values]
    main_glac_gcmprec = main_glac_gcmprec_all[main_glac_rgi['O1Index'].values]
    main_glac_gcmelev = main_glac_gcmelev_all[main_glac_rgi['O1Index'].values]
    main_glac_gcmlapserate = main_glac_gcmlapserate_all[main_glac_rgi['O1Index'].values]

print('Loading time:', time.time()-time_start, 's')

#%% ===== NEAREST NEIGHBOR SELECTION =====
# Load calibrated parameters
ds_cal_all = pd.read_csv(cal_modelparams_fullfn, index_col=0)
ds_cal = ds_cal_all.dropna()
# Dictionary cal index and 'GlacNo'
cal_dict = dict(zip(np.arange(0,ds_cal.shape[0]), ds_cal.index.values))
# Add columns to pandas dataframe
ds_cal_all_wnnbrs = ds_cal_all.copy()
ds_cal_all_wnnbrs['vol_change_perc_model'] = np.nan
ds_cal_all_wnnbrs['nbr_idx'] = np.nan
ds_cal_all_wnnbrs['nbr_idx_count'] = np.nan
ds_cal_all_wnnbrs['nbr_mb_mean'] = np.nan
ds_cal_all_wnnbrs['nbr_mb_std'] = np.nan
ds_cal_all_wnnbrs['z_score'] = np.nan
nbr_idx_cols = []
for n in range(n_nbrs):
    nbr_col_name = 'nearidx_' + str(n+1)
    ds_cal_all_wnnbrs[nbr_col_name] = np.nan
    nbr_idx_cols.append(nbr_col_name)

# Loop through each glacier and select the n_nbrs closest glaciers
for glac in range(main_glac_rgi.shape[0]):
    # Select nnbrs only for uncalibrated glaciers
    if ds_cal_all.loc[glac, modelparams_colnames].isnull().values.any() == True:
        # Print every 100th glacier
        if glac%500 == 0:
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
        ds_cal_all_wnnbrs.loc[glac, nbr_idx_cols] = indices            
        
        # Nearest neighbors: mass balance envelope
        nbrs_data = np.zeros((len(nbr_idx_cols),4))
        #  Col 0 - index count
        #  Col 1 - index value
        #  Col 2 - MB
        #  Col 3 - MB modeled using neighbor's parameter
        nbrs_data[:,0] = np.arange(1,len(nbr_idx_cols)+1) 
        nbrs_data[:,1] = ds_cal_all_wnnbrs.loc[glac, nbr_idx_cols].values.astype(int)
        nbrs_data[:,2] = ds_cal_all_wnnbrs.loc[nbrs_data[:,1], input.massbal_colname]
        mb_nbrs_mean = nbrs_data[:,2].mean()
        mb_nbrs_std = nbrs_data[:,2].std()
        mb_envelope_lower = mb_nbrs_mean - mb_nbrs_std
        mb_envelope_upper = mb_nbrs_mean + mb_nbrs_std
    
        # Glacier properties
        glac_idx = main_glac_rgi.loc[glac,'O1Index']
        glac_zmed = main_glac_rgi.loc[glac,'Zmed']
    
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
            # Model parameters
            modelparameters = ds_cal_all_wnnbrs.loc[nbr_idx,['lrgcm', 'lrglac', 'precfactor', 'precgrad', 'ddfsnow', 
                                                             'ddfice', 'tempsnow', 'tempchange']]
            nbr_zmed = ds_cal_all_wnnbrs.loc[nbr_idx,'Zmed']
            modelparameters['tempchange'] = modelparameters['tempchange'] + 0.002594 * (glac_zmed - nbr_zmed)
            modelparameters['precfactor'] = modelparameters['precfactor'] - 0.0005451 * (glac_zmed - nbr_zmed)
            modelparameters['ddfsnow'] = modelparameters['ddfsnow'] + 1.31e-06 * (glac_zmed - nbr_zmed)
            # Reset variables
            glacier_rgi_table = main_glac_rgi.loc[glac, :]
            glacier_gcm_elev = main_glac_gcmelev[glac]
            glacier_gcm_prec = main_glac_gcmprec[glac,:]
            glacier_gcm_temp = main_glac_gcmtemp[glac,:]
            glacier_gcm_lrgcm = main_glac_gcmlapserate[glac]
            glacier_gcm_lrglac = glacier_gcm_lrgcm.copy()
            glacier_area_t0 = main_glac_hyps.iloc[glac,:].values.astype(float)   
            icethickness_t0 = main_glac_icethickness.iloc[glac,:].values.astype(float)
            width_t0 = main_glac_width.iloc[glac,:].values.astype(float)
            # MASS BALANCE
            # Run the mass balance function (spinup years have been removed from output)
            (glac_bin_temp, glac_bin_prec, glac_bin_acc, glac_bin_refreeze, glac_bin_snowpack, glac_bin_melt, 
             glac_bin_frontalablation, glac_bin_massbalclim, glac_bin_massbalclim_annual, glac_bin_area_annual, 
             glac_bin_icethickness_annual, glac_bin_width_annual, glac_bin_surfacetype_annual, 
             glac_wide_massbaltotal, glac_wide_runoff, glac_wide_snowline, glac_wide_snowpack, glac_wide_area_annual, 
             glac_wide_volume_annual, glac_wide_ELA_annual) = (
                massbalance.runmassbalance(modelparameters, glacier_rgi_table, glacier_area_t0, icethickness_t0, 
                                           width_t0, elev_bins, glacier_gcm_temp, glacier_gcm_prec, glacier_gcm_elev, 
                                           glacier_gcm_lrgcm, glacier_gcm_lrglac, dates_table))
            # Volume change [%]
            if icethickness_t0.max() > 0:
                glac_vol_change_prec = (
                    (glac_wide_volume_annual[-1] - glac_wide_volume_annual[0]) / glac_wide_volume_annual[0] * 100)
            # Annual glacier-wide mass balance [m w.e.]
            glac_wide_massbaltotal_annual = np.sum(glac_wide_massbaltotal.reshape(-1,12), axis=1)
            # Average annual glacier-wide mass balance [m w.e.a.]
            massbal_idx_start = 0
            massbal_idx_end = 16
            mb_mwea = glac_wide_massbaltotal_annual[massbal_idx_start:massbal_idx_end].mean()
            #  units: m w.e. based on initial area
            nbrs_data[nbr_idx_count,3] = mb_mwea
            
#            print(nbr_idx_count, ds_cal_all_wnnbrs.loc[nbr_idx,'RGIId'])
#            print('Mass balance 2000-2015 [mwea]:', mb_mwea)      
           
            # Prior to breaking loop print RGIId and z score
            if break_while_loop == True:
                # Compute z score and print out value
                z_score = (mb_mwea - mb_nbrs_mean) / mb_nbrs_std
                if abs(z_score) > 1.96:
                    print(glacier_rgi_table.RGIId, 'z_score:', mb_mwea, np.round(z_score,2))
               
            # If mass balance falls within envelope, then end while loop
            if (mb_mwea <= mb_envelope_upper) and (mb_mwea >= mb_envelope_lower):
                break_while_loop = True
                
            # Record results
            if break_while_loop == True:
                z_score = (mb_mwea - mb_nbrs_mean) / mb_nbrs_std
                ds_cal_all_wnnbrs.loc[glac, 'MB_model_mwea'] = mb_mwea
                ds_cal_all_wnnbrs.loc[glac, 'vol_change_perc_model'] = glac_vol_change_prec
                ds_cal_all_wnnbrs.loc[glac, 'nbr_idx'] = nbr_idx
                ds_cal_all_wnnbrs.loc[glac, 'nbr_idx_count'] = nbr_idx_count + 1
                ds_cal_all_wnnbrs.loc[glac, 'nbr_mb_mean'] = mb_nbrs_mean
                ds_cal_all_wnnbrs.loc[glac, 'nbr_mb_std'] = mb_nbrs_std
                ds_cal_all_wnnbrs.loc[glac, 'z_score'] = z_score
                ds_cal_all_wnnbrs.loc[glac, 'lrgcm'] = ds_cal_all.loc[nbr_idx, 'lrgcm']
                ds_cal_all_wnnbrs.loc[glac, 'lrglac'] = ds_cal_all.loc[nbr_idx, 'lrglac']
                ds_cal_all_wnnbrs.loc[glac, 'precfactor'] = ds_cal_all.loc[nbr_idx, 'precfactor']
                ds_cal_all_wnnbrs.loc[glac, 'precgrad'] = ds_cal_all.loc[nbr_idx, 'precgrad']
                ds_cal_all_wnnbrs.loc[glac, 'ddfsnow'] = ds_cal_all.loc[nbr_idx, 'ddfsnow']
                ds_cal_all_wnnbrs.loc[glac, 'ddfice'] = ds_cal_all.loc[nbr_idx, 'ddfice']
                ds_cal_all_wnnbrs.loc[glac, 'tempsnow'] = ds_cal_all.loc[nbr_idx, 'tempsnow']
                ds_cal_all_wnnbrs.loc[glac, 'tempchange'] = ds_cal_all.loc[nbr_idx, 'tempchange']

csv_output_fullfn = cal_modelparams_fullfn.replace('.csv', '_wnnbrs_' + str(strftime("%Y%m%d")) + '.csv')
ds_cal_all_wnnbrs.to_csv(csv_output_fullfn, sep=',')

print('Processing time:', time.time()-time_start, 's')