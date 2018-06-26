"""
pygemfxns_preprocessing.py is a list of the model functions that are used to preprocess the data into the proper format.

"""

import pandas as pd
import numpy as np
import os
import glob
import xarray as xr
import netCDF4 as nc
from time import strftime
from datetime import datetime
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
import matplotlib.pyplot as plt

import pygem_input as input
import pygemfxns_modelsetup as modelsetup


#%% INPUT OPTIONS
# Lapse rate function option
option_createlapserates = 0
# WGMS process option
option_wgms = 1

# Input data for lapse rate function
if option_createlapserates == 1:
    # Input data
    gcm_filepath = os.getcwd() + '/../Climate_data/ERA_Interim/HMA_temp_pressurelevel_data/'
    gcm_filename_prefix = 'HMA_EraInterim_temp_pressurelevels_'
    tempname = 't'
    levelname = 'level'
    latname = 'latitude'
    lonname = 'longitude'
    elev_idx_max = 1
    elev_idx_min = 10
    startyear = 1979
    endyear = 2017
    output_filepath = '../Output/'
    output_filename_prefix = 'HMA_Regions13_14_15_ERAInterim_lapserates'
    

# Input data for WGMS processing
if option_wgms == 1:
    # Filenames
    rgilookup_fn = input.main_directory + '/../RGI/rgi60/00_rgi60_links/00_rgi60_links.csv'
    rgiv6_fn_prefix = input.main_directory + '/../RGI/rgi60/00_rgi60_attribs/' + '*'
    rgiv5_fn_prefix = input.main_directory + '/../RGI/00_rgi50_attribs/' + '*'
    
    wgmslookup_fn = input.main_directory + '/../WGMS/DOI-WGMS-FoG-2018-06/WGMS-FoG-2018-06-AA-GLACIER-ID-LUT.csv'
    wgms_mb_d_fn = input.main_directory + '/../WGMS/DOI-WGMS-FoG-2018-06/WGMS-FoG-2018-06-D-CHANGE.csv'
    wgms_mb_ee_fn = input.main_directory + '/../WGMS/DOI-WGMS-FoG-2018-06/WGMS-FoG-2018-06-EE-MASS-BALANCE.csv'
    wgms_mb_e_fn = input.main_directory + '/../WGMS/DOI-WGMS-FoG-2018-06/WGMS-FoG-2018-06-E-MASS-BALANCE-OVERVIEW.csv'

#%% Connect the WGMS mass balance datasets with the RGIIds and relevant elevation bands
# Note: WGMS reports the RGI in terms of V5 as opposed to V6.  Some of the glaciers have changed their RGIId between the
#       two versions, so need to convert WGMS V5 Ids to V6 Ids using the GLIMSID.
# PROBLEMS WITH DATASETS:
#  - D only have a "year" despite being geodetic mass balances that span multiple years.  Therefore, start and end year
#    need to be added during pre-processing.
#  - Furthermore, need to be careful with information describing dataset as some descriptions have been incorrect.
    
# ===== Dictionaries (WGMS --> RGIID V6) =====
# Load RGI version 5 & 6 and create dictionary linking the two
#  -required to avoid errors associated with changes in RGIId between the two versions in some regions
rgiv6_fn_all = glob.glob(rgiv6_fn_prefix)
rgiv5_fn_all = glob.glob(rgiv5_fn_prefix)
# Create dictionary of all regions
#  - regions that didn't change between versions (ex. 13, 14, 15) will all the be same.  Others that have changed may
#    vary greatly.
for n in range(len(rgiv6_fn_all)):
#for n in [14]:
    print(n)
    rgiv6_fn = glob.glob(rgiv6_fn_prefix)[n]
    rgiv6 = pd.read_csv(rgiv6_fn, encoding='latin1')
    rgiv5_fn = glob.glob(rgiv5_fn_prefix)[n]
    rgiv5 = pd.read_csv(rgiv5_fn, encoding='latin1')
    # Dictionary to link versions 5 & 6
    rgi_version_compare = rgiv5[['RGIId', 'GLIMSId']].copy()
    rgi_version_compare['RGIIdv6'] = np.nan
    # Link versions 5 & 6 based on GLIMSID
    for r in range(rgiv5.shape[0]):
        try:
            # Use GLIMSID
            rgi_version_compare.iloc[r,2] = rgiv6.iloc[rgiv6['GLIMSId'].values == rgiv5.loc[r,'GLIMSId'],0].values[0]
    #        # Use Lat/Lon
    #        latlon_dif = abs(rgiv6[['CenLon', 'CenLat']].values - rgiv5[['CenLon', 'CenLat']].values[r,:])
    #        latlon_dif[abs(latlon_dif) < 1e-6] = 0
    #        rgi_version_compare.iloc[r,2] = rgiv6.iloc[np.where(latlon_dif[:,0] + latlon_dif[:,1] < 0.001)[0][0],0]
        except:
            rgi_version_compare.iloc[r,2] = np.nan
    rgiv56_dict_reg = dict(zip(rgi_version_compare['RGIId'], rgi_version_compare['RGIIdv6']))
    if n == 0:
        rgiv56_dict = rgiv56_dict_reg.copy()
    else:
        rgiv56_dict.update(rgiv56_dict_reg)

# RGI Lookup table
rgilookup = pd.read_csv(rgilookup_fn, skiprows=2)
rgidict = dict(zip(rgilookup['FoGId'], rgilookup['RGIId']))
# WGMS Lookup table
wgmslookup = pd.read_csv(wgmslookup_fn, encoding='latin1')
wgmsdict = dict(zip(wgmslookup['WGMS_ID'], wgmslookup['RGI_ID']))
# Manual lookup table
mandict = {10402: 'RGI60-13.10093',
              10401: 'RGI60-15.03734',
              6846: 'RGI60-15.12707'}

## ===== WGMS (D) Geodetic mass balance data =====
#wgms_mb_geo_all = pd.read_csv(wgms_mb_d_fn, encoding='latin1')
#wgms_mb_geo_all['RGIId_rgidict'] = wgms_mb_geo_all['WGMS_ID'].map(rgidict)
#wgms_mb_geo_all['RGIId_mandict'] = wgms_mb_geo_all['WGMS_ID'].map(mandict)
#wgms_mb_geo_all['RGIId_wgmsdict'] = wgms_mb_geo_all['WGMS_ID'].map(wgmsdict)
#wgms_mb_geo_all['RGIId_wgmsdictv6'] = wgms_mb_geo_all['RGIId_wgmsdict'].map(rgiv56_dict)
## Use dictionaries to convert wgms data to RGIIds
#wgms_mb_geo_RGIIds_all_raw_wdicts = wgms_mb_geo_all[['RGIId_rgidict', 'RGIId_mandict','RGIId_wgmsdictv6']]
#wgms_mb_geo_RGIIds_all_raw = wgms_mb_geo_RGIIds_all_raw_wdicts.apply(lambda x: sorted(x, key=pd.isnull), 1).iloc[:,0]
## Select data for specific region
#wgms_mb_geo_all['RGIId'] = wgms_mb_geo_RGIIds_all_raw_wdicts['RGIId_rgidict']
#wgms_mb_geo_all['version'], wgms_mb_geo_all['glacno'] = wgms_mb_geo_RGIIds_all_raw.str.split('-').dropna().str
#wgms_mb_geo_all['glacno'] = wgms_mb_geo_all['glacno'].apply(pd.to_numeric)
#wgms_mb_geo_all['region'] = wgms_mb_geo_all['glacno'].apply(np.floor)
#wgms_mb_geo = wgms_mb_geo_all.loc[wgms_mb_geo_all['region'] == rgi_regionsO1[0]].sort_values('glacno')
#wgms_mb_geo.reset_index(drop=True, inplace=True)
# Export relevant information
#wgms_mb_glac_export = pd.DataFrame()
#wgms_mb_glac_export_cols = ['RGIId', 'YEAR', 'LOWER_BOUND', 'UPPER_BOUND', 'AREA', 'WINTER_BALANCE', 
#                            'WINTER_BALANCE_UNC', 'SUMMER_BALANCE', 'SUMMER_BALANCE_UNC', 'ANNUAL_BALANCE', 
#                            'ANNUAL_BALANCE_UNC', 'WGMS_ID', 'POLITICAL_UNIT', 'NAME', 'REMARKS']
#wgms_mb_glac_export[wgms_mb_glac_export_cols] = wgms_mb_glac[wgms_mb_glac_export_cols]
#wgms_mb_glac_export_fn = input.main_directory + '/../WGMS/DOI-WGMS-FoG-2018-06/R' + str(rgi_regionsO1[0]) + '_wgms_ee.csv'
#wgms_mb_glac_export.to_csv(wgms_mb_glac_export_fn)

# SEE SURVEY DATE AND REFERENCE DATE FOR BEFORE AND AFTER! Link these to dates_table for analysis


# ===== WGMS (EE) Geodetic mass balance data =====
wgms_mb_glac_all = pd.read_csv(wgms_mb_ee_fn, encoding='latin1')
wgms_mb_glac_all['RGIId_rgidict'] = wgms_mb_glac_all['WGMS_ID'].map(rgidict)
wgms_mb_glac_all['RGIId_mandict'] = wgms_mb_glac_all['WGMS_ID'].map(mandict)
wgms_mb_glac_all['RGIId_wgmsdict'] = wgms_mb_glac_all['WGMS_ID'].map(wgmsdict)
wgms_mb_glac_all['RGIId_wgmsdictv6'] = wgms_mb_glac_all['RGIId_wgmsdict'].map(rgiv56_dict)
# Use dictionaries to convert wgms data to RGIIds
wgms_mb_glac_RGIIds_all_raw_wdicts = wgms_mb_glac_all[['RGIId_rgidict', 'RGIId_mandict','RGIId_wgmsdictv6']]
wgms_mb_glac_RGIIds_all_raw = wgms_mb_glac_RGIIds_all_raw_wdicts.apply(lambda x: sorted(x, key=pd.isnull), 1).iloc[:,0]
# Determine regions and glacier numbers
wgms_mb_glac_all['RGIId'] = wgms_mb_glac_RGIIds_all_raw.values
wgms_mb_glac_all['version'], wgms_mb_glac_all['glacno'] = wgms_mb_glac_RGIIds_all_raw.str.split('-').dropna().str
wgms_mb_glac_all['glacno'] = wgms_mb_glac_all['glacno'].apply(pd.to_numeric)
wgms_mb_glac_all['region'] = wgms_mb_glac_all['glacno'].apply(np.floor)
wgms_mb_glac = wgms_mb_glac_all[np.isfinite(wgms_mb_glac_all['glacno'])].sort_values('glacno')
wgms_mb_glac.reset_index(drop=True, inplace=True)
# Import MB overview data to extract survey dates
wgms_mb_overview = pd.read_csv(wgms_mb_e_fn, encoding='latin1')
wgms_mb_glac['BEGIN_PERIOD'] = np.nan 
wgms_mb_glac['END_PERIOD'] = np.nan 
wgms_mb_glac['TIME_SYSTEM'] = np.nan
for x in range(wgms_mb_glac.shape[0]):
    wgms_mb_glac.loc[x,'BEGIN_PERIOD'] = (
            wgms_mb_overview[(wgms_mb_glac.loc[x,'WGMS_ID'] == wgms_mb_overview['WGMS_ID']) & 
                             (wgms_mb_glac.loc[x,'YEAR'] == wgms_mb_overview['Year'])]['BEGIN_PERIOD'].values)
    wgms_mb_glac.loc[x,'END_PERIOD'] = (
            wgms_mb_overview[(wgms_mb_glac.loc[x,'WGMS_ID'] == wgms_mb_overview['WGMS_ID']) & 
                             (wgms_mb_glac.loc[x,'YEAR'] == wgms_mb_overview['Year'])]['END_PERIOD'].values)
    wgms_mb_glac.loc[x,'TIME_SYSTEM'] = (
            wgms_mb_overview[(wgms_mb_glac.loc[x,'WGMS_ID'] == wgms_mb_overview['WGMS_ID']) & 
                             (wgms_mb_glac.loc[x,'YEAR'] == wgms_mb_overview['Year'])]['TIME_SYSTEM'].values)


## Test on a single region
#wgms_mb_glac = wgms_mb_glac_all.loc[wgms_mb_glac_all['region'] == n + 1].sort_values('glacno')

# Split summer, winter, and annual into separate rows such that each becomes a data point in the calibration scheme
#  if summer and winter exist, then discard annual to avoid double-counting the annual measurement
export_cols_annual = ['RGIId', 'glacno', 'WGMS_ID', 'YEAR', 'TIME_SYSTEM', 'BEGIN_PERIOD', 'END_PERIOD', 'LOWER_BOUND', 
                      'UPPER_BOUND', 'ANNUAL_BALANCE', 'ANNUAL_BALANCE_UNC', 'REMARKS']
export_cols_summer = ['RGIId', 'glacno', 'WGMS_ID', 'YEAR', 'TIME_SYSTEM', 'BEGIN_PERIOD', 'END_PERIOD', 'LOWER_BOUND', 
                      'UPPER_BOUND', 'SUMMER_BALANCE', 'SUMMER_BALANCE_UNC', 'REMARKS']
export_cols_winter = ['RGIId', 'glacno', 'WGMS_ID', 'YEAR', 'TIME_SYSTEM', 'BEGIN_PERIOD', 'END_PERIOD', 'LOWER_BOUND', 
                      'UPPER_BOUND', 'WINTER_BALANCE', 'WINTER_BALANCE_UNC', 'REMARKS']
wgms_mb_glac_annual = wgms_mb_glac.loc[((np.isnan(wgms_mb_glac['WINTER_BALANCE'])) & 
                                        (np.isnan(wgms_mb_glac['SUMMER_BALANCE']))), export_cols_annual]
wgms_mb_glac_summer = wgms_mb_glac.loc[np.isfinite(wgms_mb_glac['SUMMER_BALANCE']), export_cols_summer]
wgms_mb_glac_winter = wgms_mb_glac.loc[np.isfinite(wgms_mb_glac['WINTER_BALANCE']), export_cols_winter]
# Assign a time period to each of the measurements, which will be used for comparison with model data 
wgms_mb_glac_annual['period'] = 'annual'
wgms_mb_glac_summer['period'] = 'summer'
wgms_mb_glac_winter['period'] = 'winter'
# Rename columns such that all rows are the same
wgms_mb_glac_annual.rename(columns={'ANNUAL_BALANCE': 'BALANCE', 'ANNUAL_BALANCE_UNC': 'BALANCE_UNC'}, inplace=True)
wgms_mb_glac_summer.rename(columns={'SUMMER_BALANCE': 'BALANCE', 'SUMMER_BALANCE_UNC': 'BALANCE_UNC'}, inplace=True)
wgms_mb_glac_winter.rename(columns={'WINTER_BALANCE': 'BALANCE', 'WINTER_BALANCE_UNC': 'BALANCE_UNC'}, inplace=True)
# Export relevant information
wgms_mb_glac_export = (pd.concat([wgms_mb_glac_annual, wgms_mb_glac_summer, wgms_mb_glac_winter])
                                 .sort_values(['glacno', 'YEAR']))
# Add observation type for comparison (massbalance, snowline, etc.)
wgms_mb_glac_export['obs_type'] = 'mb'
wgms_mb_glac_export.reset_index(drop=True, inplace=True)
wgms_mb_glac_export_fn = input.main_directory + '/../WGMS/DOI-WGMS-FoG-2018-06/wgms_ee_rgiv6_preprocessed.csv'
wgms_mb_glac_export.to_csv(wgms_mb_glac_export_fn)


#%% Create netcdf file of lapse rates from temperature pressure level data
def lapserates_createnetcdf(gcm_filepath, gcm_filename_prefix, tempname, levelname, latname, lonname, elev_idx_max, 
                            elev_idx_min, startyear, endyear, output_filepath, output_filename_prefix):
    """
    Create a netcdf with the lapse rate for every latitude/longitude for each month.  The lapse rates are computed based
    on the slope of a linear line of best fit for the temperature pressure level data.
    Note: prior to running this function, you must explore the temperature pressure level data to determine the
          elevation range indices for a given region, variable names, etc.
    """
    fullfilename = gcm_filepath + gcm_filename_prefix + str(startyear) + '.nc'
    data = xr.open_dataset(fullfilename)    
    # Extract the pressure levels [Pa]
    if data[levelname].attrs['units'] == 'millibars':
        # Convert pressure levels from millibars to Pa
        levels = data[levelname].values * 100
    # Compute the elevation [m a.s.l] of the pressure levels using the barometric pressure formula (pressure in Pa)
    elev = -input.R_gas*input.temp_std/(input.gravity*input.molarmass_air)*np.log(levels/input.pressure_std)
    # Netcdf file for lapse rates ('w' will overwrite existing file)
    output_fullfilename = output_filepath + output_filename_prefix + '_' + str(startyear) + '_' + str(endyear) + '.nc'
    netcdf_output = nc.Dataset(output_fullfilename, 'w', format='NETCDF4')
    # Global attributes
    netcdf_output.description = 'Lapse rates from ERA Interim pressure level data that span the regions elevation range'
    netcdf_output.history = 'Created ' + str(strftime("%Y-%m-%d %H:%M:%S"))
    netcdf_output.source = 'ERA Interim reanalysis data downloaded February 2018'
    # Dimensions
    latitude = netcdf_output.createDimension('latitude', data['latitude'].values.shape[0])
    longitude = netcdf_output.createDimension('longitude', data['longitude'].values.shape[0])
    time = netcdf_output.createDimension('time', None)
    # Create dates in proper format for time dimension
    startdate = str(startyear) + '-01-01'
    enddate = str(endyear) + '-12-31'
    startdate = datetime(*[int(item) for item in startdate.split('-')])
    enddate = datetime(*[int(item) for item in enddate.split('-')])
    startdate = startdate.strftime('%Y-%m')
    enddate = enddate.strftime('%Y-%m')
    dates = pd.DataFrame({'date' : pd.date_range(startdate, enddate, freq='MS')})
    dates = dates['date'].astype(datetime)
    # Variables associated with dimensions 
    latitude = netcdf_output.createVariable('latitude', np.float32, ('latitude',))
    latitude.long_name = 'latitude'
    latitude.units = 'degrees_north'
    latitude[:] = data['latitude'].values
    longitude = netcdf_output.createVariable('longitude', np.float32, ('longitude',))
    longitude.long_name = 'longitude'
    longitude.units = 'degrees_east'
    longitude[:] = data['longitude'].values
    time = netcdf_output.createVariable('time', np.float64, ('time',))
    time.long_name = "time"
    time.units = "hours since 1900-01-01 00:00:00"
    time.calendar = "gregorian"
    time[:] = nc.date2num(dates, units=time.units, calendar=time.calendar)
    lapserate = netcdf_output.createVariable('lapserate', np.float64, ('time', 'latitude', 'longitude'))
    lapserate.long_name = "lapse rate"
    lapserate.units = "degC m-1"
    # Set count to keep track of time position
    count = 0
    for year in range(startyear,endyear+1):
        print(year)
        fullfilename_year = gcm_filepath + gcm_filename_prefix + str(year) + '.nc'
        data_year = xr.open_dataset(fullfilename_year)
        count = count + 1
        for lat in range(0,latitude[:].shape[0]):
            for lon in range(0,longitude[:].shape[0]):
                data_subset = data_year[tempname].isel(level=range(elev_idx_max,elev_idx_min+1), 
                                                       latitude=lat, longitude=lon).values
                lapserate_subset = (((elev[elev_idx_max:elev_idx_min+1] * data_subset).mean(axis=1) - 
                                     elev[elev_idx_max:elev_idx_min+1].mean() * data_subset.mean(axis=1)) / 
                                    ((elev[elev_idx_max:elev_idx_min+1]**2).mean() - 
                                     (elev[elev_idx_max:elev_idx_min+1].mean())**2))
                lapserate[12*(count-1):12*count,lat,lon] = lapserate_subset
                # Takes roughly 4 minutes per year to compute the lapse rate for each lat/lon combo in HMA
    netcdf_output.close()
        
# Application of the lapserate_createnetcdf function
if option_createlapserates == 1:
    lapserates_createnetcdf(gcm_filepath, gcm_filename_prefix, tempname, levelname, latname, lonname, elev_idx_max, 
                            elev_idx_min, startyear, endyear, output_filepath, output_filename_prefix)  


#%% Write csv file from model results
# Create csv such that not importing the air temperature each time (takes 90 seconds for 13,119 glaciers)
#output_csvfullfilename = input.main_directory + '/../Output/ERAInterim_elev_15_SouthAsiaEast.csv'
#climate.createcsv_GCMvarnearestneighbor(input.gcm_prec_filename, input.gcm_prec_varname, dates_table, main_glac_rgi, 
#                                        output_csvfullfilename)
#np.savetxt(output_csvfullfilename, main_glac_gcmelev, delimiter=",") 
    

#%% NEAREST NEIGHBOR CALIBRATION PARAMETERS
## Load csv
#ds = pd.read_csv(input.main_directory + '/../Output/calibration_R15_20180403_Opt02solutionspaceexpanding.csv', 
#                 index_col='GlacNo')
## Select data of interest
#data = ds[['CenLon', 'CenLat', 'lrgcm', 'lrglac', 'precfactor', 'precgrad', 'ddfsnow', 'ddfice', 'tempsnow', 
#           'tempchange']].copy()
## Drop nan data to retain only glaciers with calibrated parameters
#data_cal = data.dropna()
#A = data_cal.mean(0)
## Select latitude and longitude of calibrated parameters for distance estimate
#data_cal_lonlat = data_cal.iloc[:,0:2].values
## Loop through each glacier and select the parameters based on the nearest neighbor
#for glac in range(data.shape[0]):
#    # Avoid applying this to any glaciers that already were optimized
#    if data.iloc[glac, :].isnull().values.any() == True:
#        # Select the latitude and longitude of the glacier's center
#        glac_lonlat = data.iloc[glac,0:2].values
#        # Set point to be compatible with cdist function (from scipy)
#        pt = [[glac_lonlat[0],glac_lonlat[1]]]
#        # scipy function to calculate distance
#        distances = cdist(pt, data_cal_lonlat)
#        # Find minimum index (could be more than one)
#        idx_min = np.where(distances == distances.min())[1]
#        # Set new parameters
#        data.iloc[glac,2:] = data_cal.iloc[idx_min,2:].values.mean(0)
#        #  use mean in case multiple points are equidistant from the glacier
## Remove latitude and longitude to create csv file
#parameters_export = data.iloc[:,2:]
## Export csv file
#parameters_export.to_csv(input.main_directory + '/../Calibration_datasets/calparams_R15_20180403_nearest.csv', 
#                         index=False)    