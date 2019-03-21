"""class of climate data and functions associated with manipulating the dataset to be in the proper format"""

# External libraries
import pandas as pd
import numpy as np
import xarray as xr
# Local libraries
import pygem_input as input
import pygemfxns_modelsetup as modelsetup


class GCM():
    """
    Global climate model data properties and functions used to automatically retrieve data.
    
    Attributes
    ----------
    name : str
        name of climate dataset.
    rcp_scenario : str
        rcp scenario (example: 'rcp26')
    """
    def __init__(self, 
                 name=str(),
                 rcp_scenario=str()):
        """
        Add variable name and specific properties associated with each gcm.
        """
        
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
            self.var_fp = input.eraint_fp
            self.fx_fp = input.eraint_fp
            # Extra information
            self.timestep = input.timestep
            self.rgi_lat_colname=input.rgi_lat_colname
            self.rgi_lon_colname=input.rgi_lon_colname
        
        elif self.name == 'COAWST':
            # Variable names
            self.temp_vn = 'T2'
            self.prec_vn = 'TOTPRECIP'
            self.elev_vn = 'HGHT'
            self.lat_vn = 'LAT'
            self.lon_vn = 'LON'
            self.time_vn = 'time'
            # Variable filenames
            self.temp_fn = input.coawst_temp_fn_d02
            self.prec_fn = input.coawst_prec_fn_d02
            self.elev_fn = input.coawst_elev_fn_d02
            self.temp_fn_d01 = input.coawst_temp_fn_d01
            self.prec_fn_d01 = input.coawst_prec_fn_d01
            self.elev_fn_d01 = input.coawst_elev_fn_d01
#            self.lr_fn = input.coawst_lr_fn
            # Variable filepaths
            self.var_fp = input.coawst_fp
            self.fx_fp = input.coawst_fp
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
#            self.lr_fn = input.cmip5_lr_fn
            # Variable filepaths
            self.var_fp = input.cmip5_fp_var_prefix + rcp_scenario + input.cmip5_fp_var_ending
            self.fx_fp = input.cmip5_fp_fx_prefix + rcp_scenario + input.cmip5_fp_fx_ending
#            self.lr_fp = input.cmip5_fp_lr
            # Extra information
            self.timestep = input.timestep
            self.rgi_lat_colname=input.rgi_lat_colname
            self.rgi_lon_colname=input.rgi_lon_colname
            self.rcp_scenario = rcp_scenario
            
            
    def importGCMfxnearestneighbor_xarray(self, filename, vn, main_glac_rgi):
        """
        Import time invariant (constant) variables and extract nearest neighbor.
        
        Note: cmip5 data used surface height, while ERA-Interim data is geopotential
        
        Parameters
        ----------
        filename : str
            filename of variable
        variablename : str
            variable name
        main_glac_rgi : pandas dataframe
            dataframe containing relevant rgi glacier information
        
        Returns
        -------
        glac_variable : numpy array
            array of nearest neighbor values for all the glaciers in model run (rows=glaciers, column=variable)
        """
        # Import netcdf file
        data = xr.open_dataset(self.fx_fp + filename)
        glac_variable = np.zeros(main_glac_rgi.shape[0])
        # If time dimension included, then set the time index (required for ERA Interim, but not for CMIP5 or COAWST)
        if 'time' in data[vn].coords:
            time_idx = 0
            #  ERA Interim has only 1 value of time, so index is 0
        # Find Nearest Neighbor
        if self.name == 'COAWST':
            for glac in range(main_glac_rgi.shape[0]):
                latlon_dist = (((data[self.lat_vn].values - main_glac_rgi[self.rgi_lat_colname].values[glac])**2 + 
                                 (data[self.lon_vn].values - main_glac_rgi[self.rgi_lon_colname].values[glac])**2)**0.5)
                latlon_nearidx = [x[0] for x in np.where(latlon_dist == latlon_dist.min())]
                lat_nearidx = latlon_nearidx[0]
                lon_nearidx = latlon_nearidx[1]
                glac_variable[glac] = (
                        data[vn][latlon_nearidx[0], latlon_nearidx[1]].values)
        else:
            #  argmin() finds the minimum distance between the glacier lat/lon and the GCM pixel
            lat_nearidx = (np.abs(main_glac_rgi[self.rgi_lat_colname].values[:,np.newaxis] - 
                                  data.variables[self.lat_vn][:].values).argmin(axis=1))
            lon_nearidx = (np.abs(main_glac_rgi[self.rgi_lon_colname].values[:,np.newaxis] - 
                                  data.variables[self.lon_vn][:].values).argmin(axis=1))
            
            latlon_nearidx = list(zip(lat_nearidx, lon_nearidx))
            latlon_nearidx_unique = list(set(latlon_nearidx))
            
            glac_variable_dict = {}
            for latlon in latlon_nearidx_unique:
                try:
                    glac_variable_dict[latlon] = data[vn][time_idx, latlon[0], latlon[1]].values
                except:
                    glac_variable_dict[latlon] = data[vn][latlon[0], latlon[1]].values
            
            glac_variable = np.array([glac_variable_dict[x] for x in latlon_nearidx])    
            
#            for glac in range(main_glac_rgi.shape[0]):
#                # Select the slice of GCM data for each glacier
#                try:
#                    glac_variable[glac] = data[vn][time_idx, lat_nearidx[glac], lon_nearidx[glac]].values
#                except:
#                    glac_variable[glac] = data[vn][lat_nearidx[glac], lon_nearidx[glac]].values
        # Correct units if necessary (CMIP5 already in m a.s.l., ERA Interim is geopotential [m2 s-2])
        if vn == self.elev_vn:
            # If the variable has units associated with geopotential, then convert to m.a.s.l (ERA Interim)
            if 'units' in data[vn].attrs and (data[vn].attrs['units'] == 'm**2 s**-2'):  
                # Convert m2 s-2 to m by dividing by gravity (ERA Interim states to use 9.80665)
                glac_variable = glac_variable / 9.80665
            # Elseif units already in m.a.s.l., then continue
            elif 'units' in data[vn].attrs and data[vn].attrs['units'] == 'm':
                pass
            # Otherwise, provide warning
            else:
                print('Check units of elevation from GCM is m.')
        return glac_variable

    
    def importGCMvarnearestneighbor_xarray(self, filename, vn, main_glac_rgi, dates_table):
        """
        Import time series of variables and extract nearest neighbor.
        
        Note: "NG" refers to a homogenized "new generation" of products from ETH-Zurich.
              The function is setup to select netcdf data using the dimensions: time, latitude, longitude (in that 
              order). Prior to running the script, the user must check that this is the correct order of the dimensions 
              and the user should open the netcdf file to determine the names of each dimension as they may vary.
        
        Parameters
        ----------
        filename : str
            filename of variable
        vn : str
            variable name
        main_glac_rgi : pandas dataframe
            dataframe containing relevant rgi glacier information
        dates_table: pandas dataframe
            dataframe containing dates of model run
        
        Returns
        -------
        glac_variable_series : numpy array
            array of nearest neighbor values for all the glaciers in model run (rows=glaciers, columns=time series)
        time_series : numpy array
            array of dates associated with the meteorological data (may differ slightly from those in table for monthly
            timestep, i.e., be from the beginning/middle/end of month)
        """
        # Import netcdf file
        data = xr.open_dataset(self.var_fp + filename)
        glac_variable_series = np.zeros((main_glac_rgi.shape[0],dates_table.shape[0]))
        # Determine the correct time indices
        if self.timestep == 'monthly':
            start_idx = (np.where(pd.Series(data[self.time_vn])
                                  .apply(lambda x: x.strftime('%Y-%m')) == dates_table['date']
                                  .apply(lambda x: x.strftime('%Y-%m'))[0]))[0][0]
            end_idx = (np.where(pd.Series(data[self.time_vn])
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
            start_idx = (np.where(pd.Series(data[self.time_vn])
                                  .apply(lambda x: x.strftime('%Y-%m-%d')) == dates_table['date']
                                  .apply(lambda x: x.strftime('%Y-%m-%d'))[0]))[0][0]
            end_idx = (np.where(pd.Series(data[self.time_vn])
                                .apply(lambda x: x.strftime('%Y-%m-%d')) == dates_table['date']
                                .apply(lambda x: x.strftime('%Y-%m-%d'))[dates_table.shape[0] - 1]))[0][0]
        # Extract the time series
        time_series = pd.Series(data[self.time_vn][start_idx:end_idx+1])
        # Find Nearest Neighbor
        if self.name == 'COAWST':
            for glac in range(main_glac_rgi.shape[0]):
                latlon_dist = (((data[self.lat_vn].values - main_glac_rgi[self.rgi_lat_colname].values[glac])**2 + 
                                 (data[self.lon_vn].values - main_glac_rgi[self.rgi_lon_colname].values[glac])**2)**0.5)
                latlon_nearidx = [x[0] for x in np.where(latlon_dist == latlon_dist.min())]
                lat_nearidx = latlon_nearidx[0]
                lon_nearidx = latlon_nearidx[1]
                glac_variable_series[glac,:] = (
                        data[vn][start_idx:end_idx+1, latlon_nearidx[0], latlon_nearidx[1]].values)
        else:
            lat_nearidx = (np.abs(main_glac_rgi[self.rgi_lat_colname].values[:,np.newaxis] - 
                                  data.variables[self.lat_vn][:].values).argmin(axis=1))
            lon_nearidx = (np.abs(main_glac_rgi[self.rgi_lon_colname].values[:,np.newaxis] - 
                                  data.variables[self.lon_vn][:].values).argmin(axis=1))
            #  argmin() is finding the minimum distance between the glacier lat/lon and the GCM pixel; .values is used to 
            #  extract the position's value as opposed to having an array
            for glac in range(main_glac_rgi.shape[0]):
                # Select the slice of GCM data for each glacier
                glac_variable_series[glac,:] = (
                        data[vn][start_idx:end_idx+1, lat_nearidx[glac], lon_nearidx[glac]].values)
        # Perform corrections to the data if necessary
        # Surface air temperature corrections
        if (vn == 'tas') or (vn == 't2m') or (vn == 'T2'):
            if 'units' in data[vn].attrs and data[vn].attrs['units'] == 'K':
                # Convert from K to deg C
                glac_variable_series = glac_variable_series - 273.15
            else:
                print('Check units of air temperature from GCM is degrees C.')
        # Precipitation corrections
        # If the variable is precipitation
        elif (vn == 'pr') or (vn == 'tp') or (vn == 'TOTPRECIP'):
            # If the variable has units and those units are meters (ERA Interim)
            if 'units' in data[vn].attrs and data[vn].attrs['units'] == 'm':
                pass
            # Elseif the variable has units and those units are kg m-2 s-1 (CMIP5)
            elif 'units' in data[vn].attrs and data[vn].attrs['units'] == 'kg m-2 s-1':  
                # Convert from kg m-2 s-1 to m day-1
                glac_variable_series = glac_variable_series/1000*3600*24
                #   (1 kg m-2 s-1) * (1 m3/1000 kg) * (3600 s / hr) * (24 hr / day) = (m day-1)
            # Elseif the variable has units and those units are mm (COAWST)
            elif 'units' in data[vn].attrs and data[vn].attrs['units'] == 'mm':
                glac_variable_series = glac_variable_series/1000
            # Else check the variables units
            else:
                print('Check units of precipitation from GCM is meters per day.')
            if self.timestep == 'monthly' and self.name != 'COAWST':
                # Convert from meters per day to meters per month (COAWST data already 'monthly accumulated precipitation')
                if 'daysinmonth' in dates_table.columns:
                    glac_variable_series = glac_variable_series * dates_table['daysinmonth'].values[np.newaxis,:]
        elif vn != self.lr_vn:
            print('Check units of air temperature or precipitation')
        return glac_variable_series, time_series
    

#%% Testing
if __name__ == '__main__':
    gcm = GCM(name='CanESM2', rcp_scenario='rcp85')
#    gcm = GCM(name='ERA-Interim')
    
    main_glac_rgi = modelsetup.selectglaciersrgitable(rgi_regionsO1=input.rgi_regionsO1, rgi_regionsO2 = 'all',
                                                      rgi_glac_number='all')
    dates_table = modelsetup.datesmodelrun(startyear=2000, endyear=2006, spinupyears=0)

    print('loaded')
    # Air temperature [degC], Precipitation [m], Elevation [masl], Lapse rate [K m-1]
#    gcm_temp, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(gcm.temp_fn, gcm.temp_vn, main_glac_rgi, dates_table)
#    gcm_prec, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(gcm.prec_fn, gcm.prec_vn, main_glac_rgi, dates_table)
    gcm_elev = gcm.importGCMfxnearestneighbor_xarray(gcm.elev_fn, gcm.elev_vn, main_glac_rgi)
#    if gcm.name == 'ERA-Interim':
#        gcm_lr, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(gcm.lr_fn, gcm.lr_vn, main_glac_rgi, dates_table)
#    else:
#        gcm_lr = np.tile(ref_lr_monthly_avg, int(gcm_temp.shape[1]/12))
#    # COAWST data has two domains, so need to merge the two domains
#    if gcm.name == 'COAWST':
#        gcm_temp_d01, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(gcm.temp_fn_d01, gcm.temp_vn, main_glac_rgi, 
#                                                                         dates_table)
#        gcm_prec_d01, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(gcm.prec_fn_d01, gcm.prec_vn, main_glac_rgi, 
#                                                                         dates_table)
#        gcm_elev_d01 = gcm.importGCMfxnearestneighbor_xarray(gcm.elev_fn_d01, gcm.elev_vn, main_glac_rgi)
#        # Check if glacier outside of high-res (d02) domain
#        for glac in range(main_glac_rgi.shape[0]):
#            glac_lat = main_glac_rgi.loc[glac,input.rgi_lat_colname]
#            glac_lon = main_glac_rgi.loc[glac,input.rgi_lon_colname]
#            if (~(input.coawst_d02_lat_min <= glac_lat <= input.coawst_d02_lat_max) or 
#                ~(input.coawst_d02_lon_min <= glac_lon <= input.coawst_d02_lon_max)):
#                gcm_prec[glac,:] = gcm_prec_d01[glac,:]
#                gcm_temp[glac,:] = gcm_temp_d01[glac,:]
#                gcm_elev[glac] = gcm_elev_d01[glac]
    
    #%%
#    filename = gcm.elev_fn
#    vn = gcm.elev_vn
#    
#    # Import netcdf file
#    data = xr.open_dataset(gcm.fx_fp + filename)
#    glac_variable = np.zeros(main_glac_rgi.shape[0])
#    # If time dimension included, then set the time index (required for ERA Interim, but not for CMIP5 or COAWST)
#    if 'time' in data[vn].coords:
#        time_idx = 0
#        #  ERA Interim has only 1 value of time, so index is 0
#    # Find Nearest Neighbor
#     #  argmin() finds the minimum distance between the glacier lat/lon and the GCM pixel
#    lat_nearidx = (np.abs(main_glac_rgi[gcm.rgi_lat_colname].values[:,np.newaxis] - 
#                          data.variables[gcm.lat_vn][:].values).argmin(axis=1))
#    lon_nearidx = (np.abs(main_glac_rgi[gcm.rgi_lon_colname].values[:,np.newaxis] - 
#                          data.variables[gcm.lon_vn][:].values).argmin(axis=1))
#    
#    latlon_nearidx = list(zip(lat_nearidx, lon_nearidx))
#    latlon_nearidx_unique = list(set(latlon_nearidx))
#    
#    glac_variable_dict = {}
#    for latlon in latlon_nearidx_unique:
#        try:
#            glac_variable_dict[latlon] = data[vn][time_idx, latlon[0], latlon[1]].values
#        except:
#            glac_variable_dict[latlon] = data[vn][latlon[0], latlon[1]].values
#    
#    glac_variable = np.array([glac_variable_dict[x] for x in latlon_nearidx])    
#            
#    # Correct units if necessary (CMIP5 already in m a.s.l., ERA Interim is geopotential [m2 s-2])
#    if vn == gcm.elev_vn:
#        # If the variable has units associated with geopotential, then convert to m.a.s.l (ERA Interim)
#        if 'units' in data[vn].attrs and (data[vn].attrs['units'] == 'm**2 s**-2'):  
#            # Convert m2 s-2 to m by dividing by gravity (ERA Interim states to use 9.80665)
#            glac_variable = glac_variable / 9.80665
#        # Elseif units already in m.a.s.l., then continue
#        elif 'units' in data[vn].attrs and data[vn].attrs['units'] == 'm':
#            pass
#        # Otherwise, provide warning
#        else:
#            print('Check units of elevation from GCM is m.')
    
    
    