""" Download ERA5 data from the command line """
# You must activate the ERA5_download environment in your command line first
# Proceed to make sure you have created an account and saved your CDS API key in your home directory

# Built-in libaries
import os
# External libraries
#import cdsapi
import numpy as np
import xarray as xr
# Local libraries
import pygem_input as input

class era5_variable():
    """
    ERA5 data properties used to automatically download data
    
    Attributes
    ----------
    vn : str
        variable name
    properties : dict
        dictionary containing properties associated with the ERA5 variable
    """
    
    def __init__(self, vn, year_list):
        """
        Add variable name and specific properties associated with each variable.
        """
        # Dates formatted properly as a string
#        year_list = np.arange(input.era5_downloadyearstart, input.era5_downloadyearend + 1)
#        year_list = [str(x) for x in year_list]
        
        # Variable name
        self.vn = vn
        self.year_list = year_list
        
        if self.vn == 'temperature':
            self.level = 'reanalysis-era5-single-levels'
            self.properties = {
                    'variable':'2m_temperature',
                    'product_type':'reanalysis',
                    'area':input.bounding_box,
                    'year':year_list,
                    'month':['01','02','03', '04','05','06','07','08','09', '10','11','12'],
                    'day':['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16',
                           '17','18','19','20','21','22','23','24','25','26','27','28','29','30','31'],
                    'time':['00:00','01:00','02:00','03:00','04:00','05:00','06:00','07:00',
                            '08:00', '09:00','10:00','11:00','12:00','13:00','14:00','15:00',
                            '16:00','17:00','18:00','19:00','20:00','21:00','22:00','23:00'],
                    'format':'netcdf'
                    }
            self.fn = input.era5_fp + input.era5_temp_fn
            
            
        elif self.vn == 'precipitation':
            self.level = 'reanalysis-era5-single-levels'
            self.properties = {
                    'variable':'total_precipitation',
                    'product_type':'reanalysis',
                    'area':input.bounding_box,
                    'year':year_list,
                    'month':['01','02','03', '04','05','06','07','08','09', '10','11','12'],
                    'day':['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16',
                           '17','18','19','20','21','22','23','24','25','26','27','28','29','30','31'],
                    'time':['00:00','01:00','02:00','03:00','04:00','05:00','06:00','07:00',
                            '08:00', '09:00','10:00','11:00','12:00','13:00','14:00','15:00',
                            '16:00','17:00','18:00','19:00','20:00','21:00','22:00','23:00'],
                    'format':'netcdf'
                    }
            self.fn = input.era5_fp + input.era5_prec_fn
            
            
        elif self.vn == 'geopotential':
            self.level = 'reanalysis-era5-single-levels'
            self.properties = {
                    'variable':'orography',
                    'product_type':'reanalysis',
                    'area':input.bounding_box,
                    'year':'2018',
                    'month':['01','02','03', '04','05','06','07','08','09', '10','11','12'],
                    'day':['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16',
                           '17','18','19','20','21','22','23','24','25','26','27','28','29','30','31'],
                    'time':['00:00','01:00','02:00','03:00','04:00','05:00','06:00','07:00',
                            '08:00', '09:00','10:00','11:00','12:00','13:00','14:00','15:00',
                            '16:00','17:00','18:00','19:00','20:00','21:00','22:00','23:00'],
                    'format':'netcdf'
                    }
            self.fn = input.era5_fp + input.era5_elev_fn
            
            
        elif self.vn == 'temperature_pressurelevels':
            self.level = 'reanalysis-era5-pressure-levels'
            self.properties = {
                    'variable':'temperature',
                    'product_type':'reanalysis',
                    'area':input.bounding_box,
                    'pressure_level':['300','350','400','450','500','550','600','650','700','750','775','800','825',
                                      '850','875','900','925','950','975','1000'],
                    'year':'2018',
                    'month':['01','02','03', '04','05','06','07','08','09', '10','11','12'],
                    'day':['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16',
                           '17','18','19','20','21','22','23','24','25','26','27','28','29','30','31'],
                    'time':['00:00','01:00','02:00','03:00','04:00','05:00','06:00','07:00',
                            '08:00', '09:00','10:00','11:00','12:00','13:00','14:00','15:00',
                            '16:00','17:00','18:00','19:00','20:00','21:00','22:00','23:00'],
                    'format':'netcdf'
                    }
            self.fn = input.era5_fp + input.era5_pressureleveltemp_fn
    

#%% DOWNLOAD DATA FROM SERVER
# Check directory to store data exists or create it
if not os.path.isdir(input.era5_fp):
    os.makedirs(input.era5_fp)
    
vns = ['temperature']

# Dates formatted properly as a string
year_list = np.arange(input.era5_downloadyearstart, input.era5_downloadyearend + 1)
year_list = [str(x) for x in year_list]

# Download data for each variable
#for vn in input.era_varnames:
for vn in vns:
    
    for year in year_list:
        print(year)
        # Create a Client instance
    #    c = cdsapi.Client()
        class_vn = era5_variable(vn, [year])
        
        # Download data
    #    if not os.path.isfile(class_vn.fn):
    #        c.retrieve(class_vn.level, class_vn.properties, class_vn.fn)
        
#        # Convert to daily mean
#        ds = xr.open_dataset(input.era5_fp + 'ERA5_Temp2m_2017_2018.nc')
#        
#        if vn == 'temperature':
#            ds_monthly = ds.resample(time='1MS').mean()
        
        
#%% LAPSE RATES
## Create netcdf file for lapse rates using temperature data to fill in dimensions
#temp = xr.open_dataset(input.eraint_fp + input.eraint_temp_fn)
## Netcdf file for lapse rates ('w' will overwrite existing file)
#if not os.path.isfile(input.eraint_fp + input.eraint_lr_fn):
#    netcdf_output = nc.Dataset(input.eraint_fp + input.eraint_lr_fn, 'w', format='NETCDF4')
#    # Global attributes
#    netcdf_output.description = 'Lapse rates from ERA Interim pressure level data that span the regions elevation range'
#    netcdf_output.history = 'Created ' + str(strftime("%Y-%m-%d %H:%M:%S"))
#    netcdf_output.source = 'ERA Interim reanalysis data downloaded February 2018'
#    # Dimensions
#    latitude = netcdf_output.createDimension('latitude', temp['latitude'].values.shape[0])
#    longitude = netcdf_output.createDimension('longitude', temp['longitude'].values.shape[0])
#    time = netcdf_output.createDimension('time', None)
#    # Create dates in proper format for time dimension
#    startdate = input.eraint_start_date[0:4] + '-' + input.eraint_start_date[4:6] + '-' + input.eraint_start_date[6:]
#    enddate = input.eraint_end_date[0:4] + '-' + input.eraint_end_date[4:6] + '-' + input.eraint_end_date[6:]
#    startdate = datetime(*[int(item) for item in startdate.split('-')])
#    enddate = datetime(*[int(item) for item in enddate.split('-')])
#    startdate = startdate.strftime('%Y-%m')
#    enddate = enddate.strftime('%Y-%m')
#    dates = pd.DataFrame({'date' : pd.date_range(startdate, enddate, freq='MS')})
#    dates = dates['date'].astype(datetime)
#    # Variables associated with dimensions 
#    latitude = netcdf_output.createVariable('latitude', np.float32, ('latitude',))
#    latitude.long_name = 'latitude'
#    latitude.units = 'degrees_north'
#    latitude[:] = temp['latitude'].values
#    longitude = netcdf_output.createVariable('longitude', np.float32, ('longitude',))
#    longitude.long_name = 'longitude'
#    longitude.units = 'degrees_east'
#    longitude[:] = temp['longitude'].values
#    time = netcdf_output.createVariable('time', np.float64, ('time',))
#    time.long_name = "time"
#    time.units = "hours since 1900-01-01 00:00:00"
#    time.calendar = "gregorian"
#    time[:] = nc.date2num(dates, units=time.units, calendar=time.calendar)
#    lapserate = netcdf_output.createVariable('lapserate', np.float64, ('time', 'latitude', 'longitude'))
#    lapserate.long_name = "lapse rate"
#    lapserate.units = "degC m-1"
#    
#    # Compute lapse rates
#    # Option 1 is based on pressure level data
#    if input.option_lr_method == 1:   
#        # Compute lapse rates from temperature pressure level data
#        data = xr.open_dataset(input.eraint_fp + input.eraint_pressureleveltemp_fn) 
#        # Extract the pressure levels [Pa]
#        if data['level'].attrs['units'] == 'millibars':
#            # Convert pressure levels from millibars to Pa
#            levels = data['level'].values * 100
#        # Compute the elevation [m a.s.l] of the pressure levels using the barometric pressure formula (pressure in Pa)
#        elev = -input.R_gas*input.temp_std/(input.gravity*input.molarmass_air)*np.log(levels/input.pressure_std)
#        # Compute lapse rate
#        for lat in range(0,latitude[:].shape[0]):
#            print(latitude[lat])
#            for lon in range(0,longitude[:].shape[0]):
#                data_subset = data['t'].isel(latitude=lat, longitude=lon).values
#                lapserate[:,lat,lon] = (((elev * data_subset).mean(axis=1) - elev.mean() * data_subset.mean(axis=1)) / 
#                                        ((elev**2).mean() - (elev.mean())**2))    
#    # Option 2 is based on surrouding pixel data
#    elif input.option_lr_method == 2: 
#        # Compute lapse rates from temperature and elevation of surrouding pixels
#        # Elevation data
#        geopotential = xr.open_dataset(input.eraint_fp + input.eraint_elev_fn)
#        if ('units' in geopotential.z.attrs) and (geopotential.z.units == 'm**2 s**-2'):  
#            # Convert m2 s-2 to m by dividing by gravity (ERA Interim states to use 9.80665)
#            elev = geopotential.z.values[0,:,:] / 9.80665
#        # Compute lapse rate
#        for lat in range(1,latitude[:].shape[0]-1):
#            print('latitude:',latitude[lat])
#            for lon in range(1,longitude[:].shape[0]-1):
#                elev_subset = elev[lat-1:lat+2, lon-1:lon+2]
#                temp_subset = temp.t2m[:, lat-1:lat+2, lon-1:lon+2].values
#                #  time, latitude, longitude
#                lapserate[:,lat,lon] = (
#                        ((elev_subset * temp_subset).mean(axis=(1,2)) - elev_subset.mean() * 
#                         temp_subset.mean(axis=(1,2))) / ((elev_subset**2).mean() - (elev_subset.mean())**2))
#    netcdf_output.close()
    
