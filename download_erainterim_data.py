"""Download climate data"""
# Note: converting pressure level data to lapse rates can be a slow process (days) if done globally.

# Built-in libraries
import os
# External libraries
import pandas as pd
import numpy as np
import xarray as xr
import netCDF4 as nc
from ecmwfapi import ECMWFDataServer
from time import strftime
from datetime import datetime
# Local libraries
import pygem_input as input


class eraint_variable2download():
    """ERA Interim data properties used to automatically download data.
    """
    
    def __init__(self, vn):
        """Add variable name and specific properties associated with each variable.
        
        Parameters
        ----------
        vn : str
            variable name.
        output_fn : str
            output file name.
        properties : dict
            dictionary containing properties associated with the ERA-Interim variable.
        """
        
        # Dates formatted properly as a string
        date_list = "/".join([d.strftime('%Y%m%d') for d in 
                              pd.date_range(start=input.eraint_start_date, end=input.eraint_end_date, freq='MS')])
        # Variable name
        self.vn = vn        
        # Variable properties
        if self.vn == 'temperature':
            self.properties = {
                "class": "ei",
                "dataset": "interim",
                "date": date_list,
                "expver": "1",
                "grid": input.grid_res,
                "levtype": "sfc",
                "param": "167.128",
                "area": input.bounding_box,
                "stream": "moda",
                "type": "an",
                "format": "netcdf",
                "target": input.eraint_fp + input.eraint_temp_fn,
                }
            
        elif self.vn == 'precipitation':
            self.properties = {
                "class": "ei",
                "dataset": "interim",
                "date": date_list,
                "expver": "1",
                "grid": input.grid_res,
                "levtype": "sfc",
                "param": "228.128",
                "area": input.bounding_box,
                "step": "0-12",
                "stream": "mdfa",
                "type": "fc",
                "format": "netcdf",
                "target": input.eraint_fp + input.eraint_prec_fn,
                }
            
        elif self.vn == 'geopotential':
            self.properties = {
                "class": "ei",
                "dataset": "interim",
                "date": "1989-01-01",
                "expver": "1",
                "grid": input.grid_res,
                "levtype": "sfc",
                "param": "129.128",
                "area": input.bounding_box,
                "step": "0",
                "stream": "oper",
                "time": "12:00:00",
                "type": "an",
                "format": "netcdf",
                "target": input.eraint_fp + input.eraint_elev_fn,
                }
            
        elif self.vn == 'temperature_pressurelevels':
            self.properties = {
                "class": "ei",
                "dataset": "interim",
                "date": date_list,
                "expver": "1",
                "grid": input.grid_res,
                "levelist": "300/350/400/450/500/550/600/650/700/750/775/800/825/850/875/900/925/950/975/1000",
                "levtype": "pl",
                "param": "130.128",
                "area": input.bounding_box,
                "stream": "moda",
                "type": "an",
                "format": "netcdf",
                "target": input.eraint_fp + input.eraint_pressureleveltemp_fn,
                }
        
def retrieve_data(variable, server):
    """Retrieve ERA-Interim data from server.
    
    Parameters
    ----------
    variable : class
        eraint_variable class, which includes all properties required to retrieve data from server
    server : data server
        ECMWF data server used to access the data
        
    Returns
    -------
    Downloads climate data directly into output filename.
    """       
    server.retrieve(variable.properties)

#%% DOWNLOAD DATA FROM SERVER
# Check directory to store data exists or create it
if not os.path.isdir(input.eraint_fp):
    os.makedirs(input.eraint_fp)
# Open server
server = ECMWFDataServer()
# Download data for each variable
for varname in input.eraint_varnames:
    class_variable = eraint_variable2download(varname)
    # Check if data already downloaded
    if not os.path.isfile(class_variable.properties['target']):
        retrieve_data(class_variable, server)
        
#%% LAPSE RATES
# Create netcdf file for lapse rates using temperature data to fill in dimensions
temp = xr.open_dataset(input.eraint_fp + input.eraint_temp_fn)
# Netcdf file for lapse rates ('w' will overwrite existing file)
if not os.path.isfile(input.eraint_fp + input.eraint_lr_fn):
    netcdf_output = nc.Dataset(input.eraint_fp + input.eraint_lr_fn, 'w', format='NETCDF4')
    # Global attributes
    netcdf_output.description = 'Lapse rates from ERA Interim pressure level data that span the regions elevation range'
    netcdf_output.history = 'Created ' + str(strftime("%Y-%m-%d %H:%M:%S"))
    netcdf_output.source = 'ERA Interim reanalysis data downloaded February 2018'
    # Dimensions
    latitude = netcdf_output.createDimension('latitude', temp['latitude'].values.shape[0])
    longitude = netcdf_output.createDimension('longitude', temp['longitude'].values.shape[0])
    time = netcdf_output.createDimension('time', None)
    # Create dates in proper format for time dimension
    startdate = input.eraint_start_date[0:4] + '-' + input.eraint_start_date[4:6] + '-' + input.eraint_start_date[6:]
    enddate = input.eraint_end_date[0:4] + '-' + input.eraint_end_date[4:6] + '-' + input.eraint_end_date[6:]
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
    latitude[:] = temp['latitude'].values
    longitude = netcdf_output.createVariable('longitude', np.float32, ('longitude',))
    longitude.long_name = 'longitude'
    longitude.units = 'degrees_east'
    longitude[:] = temp['longitude'].values
    time = netcdf_output.createVariable('time', np.float64, ('time',))
    time.long_name = "time"
    time.units = "hours since 1900-01-01 00:00:00"
    time.calendar = "gregorian"
    time[:] = nc.date2num(dates, units=time.units, calendar=time.calendar)
    lapserate = netcdf_output.createVariable('lapserate', np.float64, ('time', 'latitude', 'longitude'))
    lapserate.long_name = "lapse rate"
    lapserate.units = "degC m-1"
    
    # Compute lapse rates
    # Option 1 is based on pressure level data
    if input.option_lr_method == 1:   
        print('remove restriction on lat/lon')
        # Compute lapse rates from temperature pressure level data
        data = xr.open_dataset(input.eraint_fp + input.eraint_pressureleveltemp_fn) 
        # Extract the pressure levels [Pa]
        if data['level'].attrs['units'] == 'millibars':
            # Convert pressure levels from millibars to Pa
            levels = data['level'].values * 100
        # Compute the elevation [m a.s.l] of the pressure levels using the barometric pressure formula (pressure in Pa)
        elev = -input.R_gas*input.temp_std/(input.gravity*input.molarmass_air)*np.log(levels/input.pressure_std)
        # Compute lapse rate
        for lat in range(0,latitude[:].shape[0]):
            if (20 <= latitude[lat] <= 50):
                for lon in range(0,longitude[:].shape[0]):
                    if (60 <= longitude[lon] <= 107):
                        print(latitude[lat], longitude[lon])
                        data_subset = data['t'].isel(latitude=lat, longitude=lon).values
                        lapserate[:,lat,lon] = (((elev * data_subset).mean(axis=1) - elev.mean() * data_subset.mean(axis=1)) / 
                                                ((elev**2).mean() - (elev.mean())**2))    
    # Option 2 is based on surrouding pixel data
    elif input.option_lr_method == 2: 
        # Compute lapse rates from temperature and elevation of surrouding pixels
        # Elevation data
        geopotential = xr.open_dataset(input.eraint_fp + input.eraint_elev_fn)
        if ('units' in geopotential.z.attrs) and (geopotential.z.units == 'm**2 s**-2'):  
            # Convert m2 s-2 to m by dividing by gravity (ERA Interim states to use 9.80665)
            elev = geopotential.z.values[0,:,:] / 9.80665
        # Compute lapse rate
        for lat in range(0,latitude[:].shape[0]):
            if (20 <= latitude[lat] <= 50):
                print('latitude:',latitude[lat])
                for lon in range(0,longitude[:].shape[0]):
                    if (60 <= longitude[lon] <= 107):
                        
                        elev_subset = elev[lat-1:lat+2, lon-1:lon+2]
                        temp_subset = temp.t2m[:, lat-1:lat+2, lon-1:lon+2].values
                        #  time, latitude, longitude
                        lapserate[:,lat,lon] = (
                                ((elev_subset * temp_subset).mean(axis=(1,2)) - elev_subset.mean() * temp_subset.mean(axis=(1,2))) / 
                                ((elev_subset**2).mean() - (elev_subset.mean())**2))
    netcdf_output.close()
