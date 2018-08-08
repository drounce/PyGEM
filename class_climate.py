"""
pygemfxns_climate.py is a list of functions that are used to facilitate the
pre-processing of the climate data associated with each glacier for PyGEM.
"""

import pandas as pd
import numpy as np
import xarray as xr

import pygem_input as input

class GCM():
    # GCM is the global climate model for a glacier evolution model run
    def __init__(self, 
                 name=str(),
                 rcp_scenario=str()):
        
        # Source of climate data
        self.name = name
        # Set parameters for ERA-Interim and CMIP5 netcdf files
        if self.name == 'ERA-Interim':
            # Variable names
            self.temp_vn = 't2m'
            self.prec_vn = 'tp'
            self.elev_vn = 'z'
            self.lat_vn = 'latitude'
            self.lon_vn = 'longitude'
            self.time_vn = 'time'
            self.lr_vn = 'lapserate'
            # Variable filenames
            self.temp_fn = input.eraint_temp_fn
            self.prec_fn = input.eraint_prec_fn
            self.elev_fn = input.eraint_elev_fn
            self.lr_fn = input.eraint_lr_fn
            # Variable filepaths
            self.var_fp = input.eraint_fp_var
            self.fx_fp = input.eraint_fp_fx
            # Extra information
            self.timestep = input.timestep
            self.rgi_lat_colname=input.rgi_lat_colname
            self.rgi_lon_colname=input.rgi_lon_colname
        # Other options are currently all from standardized CMIP5 format
        else:
            # Variable names
            self.temp_vn = 'tas'
            self.prec_vn = 'pr'
            self.elev_vn = 'orog'
            self.lat_vn = 'lat'
            self.lon_vn = 'lon'
            self.time_vn = 'time'
            # Variable filenames
            self.temp_fn = self.temp_vn + '_mon_' + name + '_' + rcp_scenario + '_r1i1p1_native.nc'
            self.prec_fn = self.prec_vn + '_mon_' + name + '_' + rcp_scenario + '_r1i1p1_native.nc'
            self.elev_fn = self.elev_vn + '_fx_' + name + '_' + rcp_scenario + '_r0i0p0.nc'
            self.lr_fn = input.cmip5_lr_fn
            # Variable filepaths
            self.var_fp = input.cmip5_fp_var_prefix + rcp_scenario + input.cmip5_fp_var_ending
            self.fx_fp = input.cmip5_fp_fx_prefix + rcp_scenario + input.cmip5_fp_fx_ending
            self.lr_fp = input.cmip5_fp_lr
            # Extra information
            self.timestep = input.timestep
            self.rgi_lat_colname=input.rgi_lat_colname
            self.rgi_lon_colname=input.rgi_lon_colname
            self.rcp_scenario = rcp_scenario
            
    def importGCMfxnearestneighbor_xarray(self, filename, variablename, main_glac_rgi):
        """
        Import time invariant (constant) variables and extract the nearest neighbor of the variable. Meteorological data 
        from the global climate models were provided by Ben Marzeion and ETH-Zurich for the GlacierMIP Phase II project 
        or from ERA Interim (ECMWF), which uses geopotential instead of surface height.
        
        Output: Numpy array of nearest neighbor time series for all the glaciers in the model run
        (rows = glaciers, column = variable time series)
        """
        # Import netcdf file
        filefull = self.fx_fp + filename
    #    filefull = filepath_fx + filename
        data = xr.open_dataset(filefull)
    #     print('Explore the dataset:\n', data)
    #     print('\nExplore the variable of interest:\n', data[variablename])
    #     print(data[variablename].coords)
    #     print(data.variables[variablename].attrs['units'])
        # If time dimension included, then set the time index (required for ERA Interim, but not for CMIP5 data)
        if 'time' in data[variablename].coords:
            time_idx = 0
            #  ERA Interim has only 1 value of time, so index is 0
        glac_variable = np.zeros(main_glac_rgi.shape[0])
        # Find Nearest Neighbor
        lat_nearidx = (np.abs(main_glac_rgi[self.rgi_lat_colname].values[:,np.newaxis] - 
                              data.variables[self.lat_vn][:].values).argmin(axis=1))
        lon_nearidx = (np.abs(main_glac_rgi[self.rgi_lon_colname].values[:,np.newaxis] - 
                              data.variables[self.lon_vn][:].values).argmin(axis=1))
        #  argmin() is finding the minimum distance between the glacier lat/lon and the GCM pixel
        for glac in range(main_glac_rgi.shape[0]):
            # Select the slice of GCM data for each glacier
            try:
                glac_variable[glac] = data[variablename][time_idx, lat_nearidx[glac], lon_nearidx[glac]].values
            except:
                glac_variable[glac] = data[variablename][lat_nearidx[glac], lon_nearidx[glac]].values
        # Correct units if necessary (CMIP5 already in m a.s.l., ERA Interim is geopotential [m2 s-2])
        if variablename == self.elev_vn:
            # If the variable has units associated with geopotential, then convert to m.a.s.l (ERA Interim)
            if 'units' in data.variables[variablename].attrs and (
                    data.variables[variablename].attrs['units'] == 'm**2 s**-2'):  
                # Convert m2 s-2 to m by dividing by gravity (ERA Interim states to use 9.80665)
                glac_variable = glac_variable / 9.80665
            # Elseif units already in m.a.s.l., then continue
            elif 'units' in data.variables[variablename].attrs and data.variables[variablename].attrs['units'] == 'm':
                pass
            # Otherwise, provide warning
            else:
                print('Check units of elevation from GCM is m.')
        return glac_variable
    
    
    def importGCMvarnearestneighbor_xarray(self, filename, variablename, main_glac_rgi, dates_table):
        """
        Import meteorological variables and extract the nearest neighbor time series of the variable. Meteorological 
        data from the global climate models were provided by Ben Marzeion and ETH-Zurich for the GlacierMIP Phase II 
        project. "NG" refers to their "new generation" products, which are homogenized.  Additionally, ERA-Interim 
        reanalysis data were provided by ECMWF.
        
        Note: The function is setup to select netcdf data using the dimensions: time, latitude, longitude (in that 
              order).Prior to running the script, the user must check that this is the correct order of the dimensions 
              and the user should open the netcdf file to determine the names of each dimension as they may vary.
        """
        # Import netcdf file
        filefull = self.var_fp + filename
        data = xr.open_dataset(filefull)
        glac_variable_series = np.zeros((main_glac_rgi.shape[0],dates_table.shape[0]))
    #     print('Explore the dataset:\n', data)
    #     print('\nExplore the variable of interest:\n', data[variablename])
        # Determine the correct time indices
        if self.timestep == 'monthly':
            start_idx = (np.where(pd.Series(data.variables[self.time_vn])
                                  .apply(lambda x: x.strftime('%Y-%m')) == dates_table['date']
                                  .apply(lambda x: x.strftime('%Y-%m'))[0]))[0][0]
            print()
            end_idx = (np.where(pd.Series(data.variables[self.time_vn])
                                .apply(lambda x: x.strftime('%Y-%m')) == dates_table['date']
                                .apply(lambda x: x.strftime('%Y-%m'))[dates_table.shape[0] - 1]))[0][0]
            #  np.where finds the index position where to values are equal
            #  pd.Series(data.variables[gcm_time_varname]) creates a pandas series of the time variable associated with
            #  the netcdf
            #  .apply(lambda x: x.strftime('%Y-%m')) converts the timestamp to a string with YYYY-MM to enable the 
            #  comparison
            #    > different climate dta can have different date formats, so this standardization for comparison is 
            #      important
            #      ex. monthly data may provide date on 1st of month or middle of month, so YYYY-MM-DD would not work
            #  The same processing is done for the dates_table['date'] to facilitate the comparison
            #  [0] is used to access the first date
            #  dates_table.shape[0] - 1 is used to access the last date
            #  The final indexing [0][0] is used to access the value, which is inside of an array containing extraneous 
            #  information
        elif self.timestep == 'daily':
            start_idx = (np.where(pd.Series(data.variables[self.time_vn])
                                  .apply(lambda x: x.strftime('%Y-%m-%d')) == dates_table['date']
                                  .apply(lambda x: x.strftime('%Y-%m-%d'))[0]))[0][0]
            end_idx = (np.where(pd.Series(data.variables[self.time_vn])
                                .apply(lambda x: x.strftime('%Y-%m-%d')) == dates_table['date']
                                .apply(lambda x: x.strftime('%Y-%m-%d'))[dates_table.shape[0] - 1]))[0][0]
        # Extract the time series
        time_series = pd.Series(data.variables[self.time_vn][start_idx:end_idx+1])
        # Find Nearest Neighbor
        lat_nearidx = (np.abs(main_glac_rgi[self.rgi_lat_colname].values[:,np.newaxis] - 
                              data.variables[self.lat_vn][:].values).argmin(axis=1))
        lon_nearidx = (np.abs(main_glac_rgi[self.rgi_lon_colname].values[:,np.newaxis] - 
                              data.variables[self.lon_vn][:].values).argmin(axis=1))
        #  argmin() is finding the minimum distance between the glacier lat/lon and the GCM pixel; .values is used to 
        #  extract the position's value as opposed to having an array
        for glac in range(main_glac_rgi.shape[0]):
            # Select the slice of GCM data for each glacier
            glac_variable_series[glac,:] = data[variablename][start_idx:end_idx+1, lat_nearidx[glac], 
                                                              lon_nearidx[glac]].values
        # Perform corrections to the data if necessary
        # Surface air temperature corrections
        if (variablename == 'tas') or (variablename == 't2m'):
            if 'units' in data.variables[variablename].attrs and data.variables[variablename].attrs['units'] == 'K':
                # Convert from K to deg C
                glac_variable_series = glac_variable_series - 273.15
            else:
                print('Check units of air temperature from GCM is degrees C.')
        # Precipitation corrections
        # If the variable is precipitation
        elif (variablename == 'pr') or (variablename == 'tp'):
            # If the variable has units and those units are meters (ERA Interim)
            if 'units' in data.variables[variablename].attrs and data.variables[variablename].attrs['units'] == 'm':
                pass
            # Elseif the variable has units and those units are kg m-2 s-1 (CMIP5)
            elif 'units' in data.variables[variablename].attrs and (
                    data.variables[variablename].attrs['units'] == 'kg m-2 s-1'):  
                # Convert from kg m-2 s-1 to m day-1
                glac_variable_series = glac_variable_series/1000*3600*24
                #   (1 kg m-2 s-1) * (1 m3/1000 kg) * (3600 s / hr) * (24 hr / day) = (m day-1)
            # Else check the variables units
            else:
                print('Check units of precipitation from GCM is meters per day.')
            if self.timestep == 'monthly':
                # Convert from meters per day to meters per month
                if 'daysinmonth' in dates_table.columns:
                    glac_variable_series = glac_variable_series * dates_table['daysinmonth'].values[np.newaxis,:]
        elif variablename != self.lr_vn:
            print('Check units of air temperature or precipitation')
        return glac_variable_series, time_series
    

#%% ===== FUNCTIONS FROM PREVIOUS SETUP NOT ASSOCIATED WITH CLASSES =====
#def createcsv_GCMvarnearestneighbor(filename, variablename, dates_table, glac_table, output_csvfullfilename):
#    # Import netcdf file
#    filefull = input.gcm_filepath_var + filename
#    data = xr.open_dataset(filefull)
#    glac_variable_series_nparray = np.zeros((glac_table.shape[0],dates_table.shape[0]))
#    # Determine the correct time indices
#    if input.timestep == 'monthly':
#        start_idx = (np.where(pd.Series(data.variables[input.gcm_time_varname]).apply(lambda x: x.strftime('%Y-%m')) == 
#                             dates_table['date'].apply(lambda x: x.strftime('%Y-%m'))[0]))[0][0]
#        end_idx = (np.where(pd.Series(data.variables[input.gcm_time_varname]).apply(lambda x: x.strftime('%Y-%m')) == 
#                             dates_table['date'].apply(lambda x: x.strftime('%Y-%m'))[dates_table.shape[0] - 1]))[0][0]
#    # Find Nearest Neighbor
#    lat_nearidx = (np.abs(glac_table[input.lat_colname].values[:,np.newaxis] - 
#                          data.variables[input.gcm_lat_varname][:].values).argmin(axis=1))
#    lon_nearidx = (np.abs(glac_table[input.lon_colname].values[:,np.newaxis] - 
#                          data.variables[input.gcm_lon_varname][:].values).argmin(axis=1))
#    for glac in range(glac_table.shape[0]):
#        # Select the slice of GCM data for each glacier
#        glac_variable_series_nparray[glac,:] = (data[variablename][start_idx:end_idx+1, lat_nearidx[glac], 
#                                                                   lon_nearidx[glac]].values)
#    np.savetxt(output_csvfullfilename, glac_variable_series_nparray, delimiter=",") 