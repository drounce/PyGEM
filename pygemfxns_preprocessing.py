"""
pygemfxns_preprocessing.py is a list of the model functions that are used to preprocess the data into the proper format.

"""
#========== IMPORT MODULES USED IN FUNCTIONS ==========================================================================
import pandas as pd
import numpy as np
import os
import xarray as xr
import netCDF4 as nc
from time import strftime
from datetime import datetime
#========== IMPORT INPUT AND FUNCTIONS FROM MODULES ===================================================================
import pygem_input as input

#%% Write csv file from model results
# Create csv such that not importing the air temperature each time (takes 90 seconds for 13,119 glaciers)
#output_csvfullfilename = input.main_directory + '/../Output/ERAInterim_elev_15_SouthAsiaEast.csv'
#climate.createcsv_GCMvarnearestneighbor(input.gcm_prec_filename, input.gcm_prec_varname, dates_table, main_glac_rgi, 
#                                        output_csvfullfilename)
#np.savetxt(output_csvfullfilename, main_glac_gcmelev, delimiter=",") 


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
        
## Application of the lapserate_createnetcdf function
#gcm_filepath = os.getcwd() + '/../Climate_data/ERA_Interim/HMA_temp_pressurelevel_data/'
#gcm_filename_prefix = 'HMA_EraInterim_temp_pressurelevels_'
#tempname = 't'
#levelname = 'level'
#latname = 'latitude'
#lonname = 'longitude'
#elev_idx_max = 1
#elev_idx_min = 10
#startyear = 1979
#endyear = 2017
#output_filepath = '../Output/'
#output_filename_prefix = 'HMA_Regions13_14_15_ERAInterim_lapserates'
#lapserates_createnetcdf(gcm_filepath, gcm_filename_prefix, tempname, levelname, latname, lonname, elev_idx_max, 
#                        elev_idx_min, startyear, endyear, output_filepath, output_filename_prefix)  

#%% Connect the WGMS point mass balance datasets with the RGIIds and relevant elevation bands
# Bounding box
lat_bndN = 46
lat_bndS = 26
lon_bndW = 65
lon_bndE = 105

# Load RGI lookup table 
rgilookup_filename = input.main_directory + '/../RGI/rgi60/00_rgi60_links/00_rgi60_links.csv'
rgilookup = pd.read_csv(rgilookup_filename, skiprows=2)
rgidict = dict(zip(rgilookup['FoGId'], rgilookup['RGIId']))
# Load WGMS lookup table
wgmslookup_filename = (input.main_directory + 
                       '/../Calibration_datasets\DOI-WGMS-FoG-2017-10\WGMS-FoG-2017-10-AA-GLACIER-ID-LUT.csv')
wgmslookup = pd.read_csv(wgmslookup_filename, encoding='latin1')
wgmsdict = dict(zip(wgmslookup['WGMS_ID'], wgmslookup['RGI_ID']))

# WGMS POINT MASS BALANCE DATA
# Load WGMS point mass balance data
wgms_massbal_pt_filename = (input.main_directory + 
                            '/../Calibration_datasets\DOI-WGMS-FoG-2017-10/WGMS-FoG-2017-10-EEE-MASS-BALANCE-POINT.csv')
wgms_massbal_pt = pd.read_csv(wgms_massbal_pt_filename, encoding='latin1')

# Select values based on the bounding box
wgms_massbal_pt = wgms_massbal_pt[(wgms_massbal_pt['POINT_LAT'] <= lat_bndN) & 
                                  (wgms_massbal_pt['POINT_LAT'] >= lat_bndS) & 
                                  (wgms_massbal_pt['POINT_LON'] <= lon_bndE) &
                                  (wgms_massbal_pt['POINT_LON'] >= lon_bndW)]
# Remove values without an elevation
wgms_massbal_pt = wgms_massbal_pt[wgms_massbal_pt['POINT_ELEVATION'].isnull() == False]
# Select values within yearly range
wgms_massbal_pt = wgms_massbal_pt[(wgms_massbal_pt['YEAR'] >= input.startyear) & 
                                  (wgms_massbal_pt['YEAR'] <= input.endyear)]
# Count unique values
wgms_massbal_pt_Ids = pd.DataFrame()
wgms_massbal_pt_Ids['WGMS_ID'] = wgms_massbal_pt['WGMS_ID'].value_counts().index.values
wgms_massbal_pt_Ids['RGIId'] = np.nan
wgms_massbal_pt_Ids['RGIId_wgms'] = wgms_massbal_pt_Ids['WGMS_ID'].map(wgmsdict)
wgms_massbal_pt_Ids['RGIId_rgi'] = wgms_massbal_pt_Ids['WGMS_ID'].map(rgidict)
# Manually add additional points
manualdict = {10402: 'RGI60-13.10093'}
wgms_massbal_pt_Ids['RGIId_manual'] = wgms_massbal_pt_Ids['WGMS_ID'].map(manualdict)

for glac in range(wgms_massbal_pt_Ids.shape[0]):
    if pd.isnull(wgms_massbal_pt_Ids.loc[glac,'RGIId_wgms']) == False:
        wgms_massbal_pt_Ids.loc[glac,'RGIId'] = wgms_massbal_pt_Ids.loc[glac,'RGIId_wgms']
    elif pd.isnull(wgms_massbal_pt_Ids.loc[glac,'RGIId_rgi']) == False:
        wgms_massbal_pt_Ids.loc[glac,'RGIId'] = wgms_massbal_pt_Ids.loc[glac,'RGIId_rgi']
    elif pd.isnull(wgms_massbal_pt_Ids.loc[glac,'RGIId_manual']) == False:
        wgms_massbal_pt_Ids.loc[glac,'RGIId'] = wgms_massbal_pt_Ids.loc[glac,'RGIId_manual']

# WGMS GEODETIC MASS BALANCE DATA 
# Load WGMS geodetic mass balance data
wgms_massbal_geo_filename = (input.main_directory + 
                            '/../Calibration_datasets\DOI-WGMS-FoG-2017-10/WGMS-FoG-2017-10-EE-MASS-BALANCE.csv')
wgms_massbal_geo = pd.read_csv(wgms_massbal_geo_filename, encoding='latin1')
# Load WGMS glacier table to look up lat/lon that goes with the glacier
wgms_massbal_geo_lookup_filename = (input.main_directory + 
                            '/../Calibration_datasets\DOI-WGMS-FoG-2017-10/WGMS-FoG-2017-10-A-GLACIER.csv')
wgms_massbal_geo_lookup = pd.read_csv(wgms_massbal_geo_lookup_filename, encoding='latin1')
# Create WGMSID - lat/lon dictionaries
wgms_latdict = dict(zip(wgms_massbal_geo_lookup['WGMS_ID'], wgms_massbal_geo_lookup['LATITUDE']))
wgms_londict = dict(zip(wgms_massbal_geo_lookup['WGMS_ID'], wgms_massbal_geo_lookup['LONGITUDE']))
# Add latitude and longitude to wgms_massbal_measurements
wgms_massbal_geo['LATITUDE'] = wgms_massbal_geo['WGMS_ID'].map(wgms_latdict)
wgms_massbal_geo['LONGITUDE'] = wgms_massbal_geo['WGMS_ID'].map(wgms_londict)
# Select values based on the bounding box
wgms_massbal_geo = wgms_massbal_geo[(wgms_massbal_geo['LATITUDE'] <= lat_bndN) & 
                                    (wgms_massbal_geo['LATITUDE'] >= lat_bndS) & 
                                    (wgms_massbal_geo['LONGITUDE'] <= lon_bndE) &
                                    (wgms_massbal_geo['LONGITUDE'] >= lon_bndW)]
# Select only glacier-wide values ()
#

#%% Conslidate the WGMS data into a single csv file for a given WGMS-defined region  
### Inputs for mass balance glaciological method
###filepath = os.getcwd() + '/../WGMS/Asia_South_East_MB_glac_method/'
##filepath = os.getcwd() + '/../WGMS/Asia_South_West_MB_glac_method/'
##filename_prefix = 'FoG_MB_'
##skiprows_value = 13
#
## Inputs for mass balance (glacier thickness change) from geodetic approach
##filepath = os.getcwd() + '/../WGMS/Asia_South_East_Thickness_change_geodetic/'
#filepath = os.getcwd() + '/../WGMS/Asia_South_West_Thickness_change_geodetic/'
#filename_prefix = 'FoG_TC_'
#skiprows_value = 16
#    
#data = None
#for filename in os.listdir(filepath):
#    print(filename)
#    try:
#        # try reading csv with default encoding
#        data_subset = pd.read_csv(filepath + filename, delimiter = ';', skiprows=skiprows_value, quotechar='"')
#    except:
#        # except try reading with latin1, which handles accents
#        data_subset = pd.read_csv(filepath + filename, delimiter = ';', skiprows=skiprows_value, quotechar='"', encoding='latin1')
#        
#    # Append data to create one dataframe
#    if data is None:
#        data = data_subset
#    else:
#        data = data.append(data_subset)
## Sort data according to ID and survey year
#data = data.sort_values(by=['WGMS_ID', 'SURVEY_YEAR'])     
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    