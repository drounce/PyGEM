""" PROCESS CLIMATE DATA FOR A GIVEN GLACIER DIRECTORY """

# External libraries
import xarray as xr
# Local libraries
import pygem.pygem_input as pygem_prms

print(' check units of all data consistent with PyGEM')
print(' check files exist')
print(' add assert statements for option_ablation, i.e., tempstd must exist to use')
print(' check longitude 0-360!')
#%%

def process_historicalclimatedata(gdir, vns=pygem_prms.climate_dict[pygem_prms.ref_gcm_name]['vns']):
    """
    Import time invariant (constant) variables and extract nearest neighbor.
    
    Note: cmip5 data used surface height, while ERA-Interim data is geopotential
    
    Parameters
    ----------
    gdir : glacier directory
        glacier directory object
    variablename : str
        variable name
    main_glac_rgi : pandas dataframe
        dataframe containing relevant rgi glacier information
    
    Returns
    -------
    glac_variable : numpy array
        array of nearest neighbor values for all the glaciers in model run (rows=glaciers, column=variable)
    """
#    # Import netcdf file
#    data = xr.open_dataset(self.fx_fp + filename)
#    glac_variable = np.zeros(main_glac_rgi.shape[0])
#    # If time dimension included, then set the time index (required for ERA Interim, but not for CMIP5 or COAWST)
#    if 'time' in data[vn].coords:
#        time_idx = 0
#        #  ERA Interim has only 1 value of time, so index is 0
#    # Find Nearest Neighbor
#    if self.name == 'COAWST':
#        for glac in range(main_glac_rgi.shape[0]):
#            latlon_dist = (((data[self.lat_vn].values - main_glac_rgi[self.rgi_lat_colname].values[glac])**2 + 
#                             (data[self.lon_vn].values - main_glac_rgi[self.rgi_lon_colname].values[glac])**2)**0.5)
#            latlon_nearidx = [x[0] for x in np.where(latlon_dist == latlon_dist.min())]
#            lat_nearidx = latlon_nearidx[0]
#            lon_nearidx = latlon_nearidx[1]
#            glac_variable[glac] = (
#                    data[vn][latlon_nearidx[0], latlon_nearidx[1]].values)
#    else:
#        #  argmin() finds the minimum distance between the glacier lat/lon and the GCM pixel
#        lat_nearidx = (np.abs(main_glac_rgi[self.rgi_lat_colname].values[:,np.newaxis] - 
#                              data.variables[self.lat_vn][:].values).argmin(axis=1))
#        lon_nearidx = (np.abs(main_glac_rgi[self.rgi_lon_colname].values[:,np.newaxis] - 
#                              data.variables[self.lon_vn][:].values).argmin(axis=1))
#        
#        latlon_nearidx = list(zip(lat_nearidx, lon_nearidx))
#        latlon_nearidx_unique = list(set(latlon_nearidx))
#    
#        glac_variable_dict = {}
#        for latlon in latlon_nearidx_unique:
#            try:
#                glac_variable_dict[latlon] = data[vn][time_idx, latlon[0], latlon[1]].values
#            except:
#                glac_variable_dict[latlon] = data[vn][latlon[0], latlon[1]].values
#        
#        glac_variable = np.array([glac_variable_dict[x] for x in latlon_nearidx])    
#        
##            for glac in range(main_glac_rgi.shape[0]):
##                # Select the slice of GCM data for each glacier
##                try:
##                    glac_variable[glac] = data[vn][time_idx, lat_nearidx[glac], lon_nearidx[glac]].values
##                except:
##                    glac_variable[glac] = data[vn][lat_nearidx[glac], lon_nearidx[glac]].values
#    # Correct units if necessary (CMIP5 already in m a.s.l., ERA Interim is geopotential [m2 s-2])
#    if vn == self.elev_vn:
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
#            
#    return glac_variable

#%%
#@entity_task(log, writes=['climate_historical'])
#def process_ecmwf_data(gdir, dataset=None, ensemble_member=0,
#                       y0=None, y1=None, output_filesuffix=None):
#    """Processes and writes the ECMWF baseline climate data for this glacier.
#    Extracts the nearest timeseries and writes everything to a NetCDF file.
#    Parameters
#    ----------
#    dataset : str
#        'ERA5', 'ERA5L', 'CERA'. Defaults to cfg.PARAMS['baseline_climate']
#    ensemble_member : int
#        for CERA, pick an ensemble member number (0-9). We might make this
#        more of a clever pick later.
#    y0 : int
#        the starting year of the timeseries to write. The default is to take
#        the entire time period available in the file, but with this kwarg
#        you can shorten it (to save space or to crop bad data)
#    y1 : int
#        the starting year of the timeseries to write. The default is to take
#        the entire time period available in the file, but with this kwarg
#        you can shorten it (to save space or to crop bad data)
#    output_filesuffix : str
#        this add a suffix to the output file (useful to avoid overwriting
#        previous experiments)
#    """
#
#    if cfg.PATHS.get('climate_file', None):
#        warnings.warn("You seem to have set a custom climate file for this "
#                      "run, but are using the ECMWF climate file "
#                      "instead.")
#
#    if dataset is None:
#        dataset = cfg.PARAMS['baseline_climate']
#
#    # Use xarray to read the data
#    lon = gdir.cenlon + 360 if gdir.cenlon < 0 else gdir.cenlon
#    lat = gdir.cenlat
#    with xr.open_dataset(get_ecmwf_file(dataset, 'tmp')) as ds:
#        assert ds.longitude.min() >= 0
#        # set temporal subset for the ts data (hydro years)
#        sm = cfg.PARAMS['hydro_month_' + gdir.hemisphere]
#        em = sm - 1 if (sm > 1) else 12
#        yrs = ds['time.year'].data
#        y0 = yrs[0] if y0 is None else y0
#        y1 = yrs[-1] if y1 is None else y1
#        if dataset == 'ERA5dr':
#            # Last year incomplete
#            assert ds['time.month'][-1] == 5
#            if em > 5:
#                y1 -= 1
#        ds = ds.sel(time=slice('{}-{:02d}-01'.format(y0, sm),
#                               '{}-{:02d}-01'.format(y1, em)))
#        if dataset == 'CERA':
#            ds = ds.sel(number=ensemble_member)
#        try:
#            ds = ds.sel(longitude=lon, latitude=lat, method='nearest')
#        except ValueError:
#            # Flattened ERA5
#            c = (ds.longitude - lon)**2 + (ds.latitude - lat)**2
#            ds = ds.isel(points=c.argmin())
#        temp = ds['t2m'].data - 273.15
#        time = ds.time.data
#        ref_lon = float(ds['longitude'])
#        ref_lon = ref_lon - 360 if ref_lon > 180 else ref_lon
#        ref_lat = float(ds['latitude'])
#    with xr.open_dataset(get_ecmwf_file(dataset, 'pre')) as ds:
#        assert ds.longitude.min() >= 0
#        ds = ds.sel(time=slice('{}-{:02d}-01'.format(y0, sm),
#                               '{}-{:02d}-01'.format(y1, em)))
#        if dataset == 'CERA':
#            ds = ds.sel(number=ensemble_member)
#        try:
#            ds = ds.sel(longitude=lon, latitude=lat, method='nearest')
#        except ValueError:
#            # Flattened ERA5
#            c = (ds.longitude - lon)**2 + (ds.latitude - lat)**2
#            ds = ds.isel(points=c.argmin())
#        prcp = ds['tp'].data * 1000 * ds['time.daysinmonth']
#    with xr.open_dataset(get_ecmwf_file(dataset, 'inv')) as ds:
#        assert ds.longitude.min() >= 0
#        ds = ds.isel(time=0)
#        try:
#            ds = ds.sel(longitude=lon, latitude=lat, method='nearest')
#        except ValueError:
#            # Flattened ERA5
#            c = (ds.longitude - lon)**2 + (ds.latitude - lat)**2
#            ds = ds.isel(points=c.argmin())
#        hgt = ds['z'].data / cfg.G
#
#    gradient = None
#    temp_std = None
#
#    # Should we compute the gradient?
#    if cfg.PARAMS['temp_use_local_gradient']:
#        raise NotImplementedError('`temp_use_local_gradient` not '
#                                  'implemented yet')
#    if dataset == 'ERA5dr':
#        with xr.open_dataset(get_ecmwf_file(dataset, 'lapserates')) as ds:
#            assert ds.longitude.min() >= 0
#            ds = ds.sel(time=slice('{}-{:02d}-01'.format(y0, sm),
#                                   '{}-{:02d}-01'.format(y1, em)))
#            ds = ds.sel(longitude=lon, latitude=lat, method='nearest')
#            gradient = ds['lapserate'].data
#
#        with xr.open_dataset(get_ecmwf_file(dataset, 'tempstd')) as ds:
#            assert ds.longitude.min() >= 0
#            ds = ds.sel(time=slice('{}-{:02d}-01'.format(y0, sm),
#                                   '{}-{:02d}-01'.format(y1, em)))
#            ds = ds.sel(longitude=lon, latitude=lat, method='nearest')
#            temp_std = ds['t2m_std'].data
#
#    # OK, ready to write
#    gdir.write_monthly_climate_file(time, prcp, temp, hgt, ref_lon, ref_lat,
#                                    filesuffix=output_filesuffix,
#                                    gradient=gradient,
#                                    temp_std=temp_std,
#                                    source=dataset)

#%%
#    def importGCMfxnearestneighbor_xarray(self, filename, vn, main_glac_rgi):
#        """
#        Import time invariant (constant) variables and extract nearest neighbor.
#        
#        Note: cmip5 data used surface height, while ERA-Interim data is geopotential
#        
#        Parameters
#        ----------
#        filename : str
#            filename of variable
#        variablename : str
#            variable name
#        main_glac_rgi : pandas dataframe
#            dataframe containing relevant rgi glacier information
#        
#        Returns
#        -------
#        glac_variable : numpy array
#            array of nearest neighbor values for all the glaciers in model run (rows=glaciers, column=variable)
#        """
#        # Import netcdf file
#        data = xr.open_dataset(self.fx_fp + filename)
#        glac_variable = np.zeros(main_glac_rgi.shape[0])
#        # If time dimension included, then set the time index (required for ERA Interim, but not for CMIP5 or COAWST)
#        if 'time' in data[vn].coords:
#            time_idx = 0
#            #  ERA Interim has only 1 value of time, so index is 0
#        # Find Nearest Neighbor
#        if self.name == 'COAWST':
#            for glac in range(main_glac_rgi.shape[0]):
#                latlon_dist = (((data[self.lat_vn].values - main_glac_rgi[self.rgi_lat_colname].values[glac])**2 + 
#                                 (data[self.lon_vn].values - main_glac_rgi[self.rgi_lon_colname].values[glac])**2)**0.5)
#                latlon_nearidx = [x[0] for x in np.where(latlon_dist == latlon_dist.min())]
#                lat_nearidx = latlon_nearidx[0]
#                lon_nearidx = latlon_nearidx[1]
#                glac_variable[glac] = (
#                        data[vn][latlon_nearidx[0], latlon_nearidx[1]].values)
#        else:
#            #  argmin() finds the minimum distance between the glacier lat/lon and the GCM pixel
#            lat_nearidx = (np.abs(main_glac_rgi[self.rgi_lat_colname].values[:,np.newaxis] - 
#                                  data.variables[self.lat_vn][:].values).argmin(axis=1))
#            lon_nearidx = (np.abs(main_glac_rgi[self.rgi_lon_colname].values[:,np.newaxis] - 
#                                  data.variables[self.lon_vn][:].values).argmin(axis=1))
#            
#            latlon_nearidx = list(zip(lat_nearidx, lon_nearidx))
#            latlon_nearidx_unique = list(set(latlon_nearidx))
#        
#            glac_variable_dict = {}
#            for latlon in latlon_nearidx_unique:
#                try:
#                    glac_variable_dict[latlon] = data[vn][time_idx, latlon[0], latlon[1]].values
#                except:
#                    glac_variable_dict[latlon] = data[vn][latlon[0], latlon[1]].values
#            
#            glac_variable = np.array([glac_variable_dict[x] for x in latlon_nearidx])    
#            
##            for glac in range(main_glac_rgi.shape[0]):
##                # Select the slice of GCM data for each glacier
##                try:
##                    glac_variable[glac] = data[vn][time_idx, lat_nearidx[glac], lon_nearidx[glac]].values
##                except:
##                    glac_variable[glac] = data[vn][lat_nearidx[glac], lon_nearidx[glac]].values
#        # Correct units if necessary (CMIP5 already in m a.s.l., ERA Interim is geopotential [m2 s-2])
#        if vn == self.elev_vn:
#            # If the variable has units associated with geopotential, then convert to m.a.s.l (ERA Interim)
#            if 'units' in data[vn].attrs and (data[vn].attrs['units'] == 'm**2 s**-2'):  
#                # Convert m2 s-2 to m by dividing by gravity (ERA Interim states to use 9.80665)
#                glac_variable = glac_variable / 9.80665
#            # Elseif units already in m.a.s.l., then continue
#            elif 'units' in data[vn].attrs and data[vn].attrs['units'] == 'm':
#                pass
#            # Otherwise, provide warning
#            else:
#                print('Check units of elevation from GCM is m.')
#                
#        return glac_variable

    
#    def importGCMvarnearestneighbor_xarray(self, filename, vn, main_glac_rgi, dates_table):
#        """
#        Import time series of variables and extract nearest neighbor.
#        
#        Note: "NG" refers to a homogenized "new generation" of products from ETH-Zurich.
#              The function is setup to select netcdf data using the dimensions: time, latitude, longitude (in that 
#              order). Prior to running the script, the user must check that this is the correct order of the dimensions 
#              and the user should open the netcdf file to determine the names of each dimension as they may vary.
#        
#        Parameters
#        ----------
#        filename : str
#            filename of variable
#        vn : str
#            variable name
#        main_glac_rgi : pandas dataframe
#            dataframe containing relevant rgi glacier information
#        dates_table: pandas dataframe
#            dataframe containing dates of model run
#        
#        Returns
#        -------
#        glac_variable_series : numpy array
#            array of nearest neighbor values for all the glaciers in model run (rows=glaciers, columns=time series)
#        time_series : numpy array
#            array of dates associated with the meteorological data (may differ slightly from those in table for monthly
#            timestep, i.e., be from the beginning/middle/end of month)
#        """
#        # Import netcdf file
#        data = xr.open_dataset(self.var_fp + filename)
#        glac_variable_series = np.zeros((main_glac_rgi.shape[0],dates_table.shape[0]))
#        # Determine the correct time indices
#        if self.timestep == 'monthly':
#            start_idx = (np.where(pd.Series(data[self.time_vn]).apply(lambda x: x.strftime('%Y-%m')) == 
#                                  dates_table['date'].apply(lambda x: x.strftime('%Y-%m'))[0]))[0][0]
#            end_idx = (np.where(pd.Series(data[self.time_vn]).apply(lambda x: x.strftime('%Y-%m')) == 
#                                dates_table['date']
#                                .apply(lambda x: x.strftime('%Y-%m'))[dates_table.shape[0] - 1]))[0][0]              
#            #  np.where finds the index position where to values are equal
#            #  pd.Series(data.variables[gcm_time_varname]) creates a pandas series of the time variable associated with
#            #  the netcdf
#            #  .apply(lambda x: x.strftime('%Y-%m')) converts the timestamp to a string with YYYY-MM to enable the 
#            #  comparison
#            #    > different climate dta can have different date formats, so this standardization for comparison is 
#            #      important
#            #      ex. monthly data may provide date on 1st of month or middle of month, so YYYY-MM-DD would not work
#            #  The same processing is done for the dates_table['date'] to facilitate the comparison
#            #  [0] is used to access the first date
#            #  dates_table.shape[0] - 1 is used to access the last date
#            #  The final indexing [0][0] is used to access the value, which is inside of an array containing extraneous 
#            #  information
#        elif self.timestep == 'daily':
#            start_idx = (np.where(pd.Series(data[self.time_vn])
#                                  .apply(lambda x: x.strftime('%Y-%m-%d')) == dates_table['date']
#                                  .apply(lambda x: x.strftime('%Y-%m-%d'))[0]))[0][0]
#            end_idx = (np.where(pd.Series(data[self.time_vn])
#                                .apply(lambda x: x.strftime('%Y-%m-%d')) == dates_table['date']
#                                .apply(lambda x: x.strftime('%Y-%m-%d'))[dates_table.shape[0] - 1]))[0][0]
#        # Extract the time series
#        time_series = pd.Series(data[self.time_vn][start_idx:end_idx+1])
#        # Find Nearest Neighbor
#        if self.name == 'COAWST':
#            for glac in range(main_glac_rgi.shape[0]):
#                latlon_dist = (((data[self.lat_vn].values - main_glac_rgi[self.rgi_lat_colname].values[glac])**2 + 
#                                 (data[self.lon_vn].values - main_glac_rgi[self.rgi_lon_colname].values[glac])**2)**0.5)
#                latlon_nearidx = [x[0] for x in np.where(latlon_dist == latlon_dist.min())]
#                lat_nearidx = latlon_nearidx[0]
#                lon_nearidx = latlon_nearidx[1]
#                glac_variable_series[glac,:] = (
#                        data[vn][start_idx:end_idx+1, latlon_nearidx[0], latlon_nearidx[1]].values)
#        else:
#            #  argmin() finds the minimum distance between the glacier lat/lon and the GCM pixel; .values is used to 
#            #  extract the position's value as opposed to having an array
#            lat_nearidx = (np.abs(main_glac_rgi[self.rgi_lat_colname].values[:,np.newaxis] - 
#                                  data.variables[self.lat_vn][:].values).argmin(axis=1))
#            lon_nearidx = (np.abs(main_glac_rgi[self.rgi_lon_colname].values[:,np.newaxis] - 
#                                  data.variables[self.lon_vn][:].values).argmin(axis=1))
#            # Find unique latitude/longitudes
#            latlon_nearidx = list(zip(lat_nearidx, lon_nearidx))
#            latlon_nearidx_unique = list(set(latlon_nearidx))
#            # Create dictionary of time series for each unique latitude/longitude
#            glac_variable_dict = {}
#            for latlon in latlon_nearidx_unique:
#                glac_variable_dict[latlon] = data[vn][start_idx:end_idx+1, latlon[0], latlon[1]].values
#            # Convert to series
#            glac_variable_series = np.array([glac_variable_dict[x] for x in latlon_nearidx])    
#
##            for glac in range(main_glac_rgi.shape[0]):
##                # Select the slice of GCM data for each glacier
##                glac_variable_series[glac,:] = (
##                        data[vn][start_idx:end_idx+1, lat_nearidx[glac], lon_nearidx[glac]].values)
#                
#        # Perform corrections to the data if necessary
#        # Surface air temperature corrections
#        if vn in ['tas', 't2m', 'T2']:
#            if 'units' in data[vn].attrs and data[vn].attrs['units'] == 'K':
#                # Convert from K to deg C
#                glac_variable_series = glac_variable_series - 273.15
#            else:
#                print('Check units of air temperature from GCM is degrees C.')
#        elif vn in ['t2m_std']:
#            if 'units' in data[vn].attrs and data[vn].attrs['units'] not in ['C', 'K']:
#                print('Check units of air temperature standard deviation from GCM is degrees C or K')
#        # Precipitation corrections
#        # If the variable is precipitation
#        elif vn in ['pr', 'tp', 'TOTPRECIP']:
#            # If the variable has units and those units are meters (ERA Interim)
#            if 'units' in data[vn].attrs and data[vn].attrs['units'] == 'm':
#                pass
#            # Elseif the variable has units and those units are kg m-2 s-1 (CMIP5)
#            elif 'units' in data[vn].attrs and data[vn].attrs['units'] == 'kg m-2 s-1':  
#                # Convert from kg m-2 s-1 to m day-1
#                glac_variable_series = glac_variable_series/1000*3600*24
#                #   (1 kg m-2 s-1) * (1 m3/1000 kg) * (3600 s / hr) * (24 hr / day) = (m day-1)
#            # Elseif the variable has units and those units are mm (COAWST)
#            elif 'units' in data[vn].attrs and data[vn].attrs['units'] == 'mm':
#                glac_variable_series = glac_variable_series/1000
#            # Else check the variables units
#            else:
#                print('Check units of precipitation from GCM is meters per day.')
#            if self.timestep == 'monthly' and self.name != 'COAWST':
#                # Convert from meters per day to meters per month (COAWST data already 'monthly accumulated precipitation')
#                if 'daysinmonth' in dates_table.columns:
#                    glac_variable_series = glac_variable_series * dates_table['daysinmonth'].values[np.newaxis,:]
#        elif vn != self.lr_vn:
#            print('Check units of air temperature or precipitation')
#        return glac_variable_series, time_series
    
    