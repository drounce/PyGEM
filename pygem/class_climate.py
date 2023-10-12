"""class of climate data and functions associated with manipulating the dataset to be in the proper format"""

import os
# External libraries
import pandas as pd
import numpy as np
import xarray as xr
# Local libraries
try:
    import pygem_input as pygem_prms
except:
    import pygem.pygem_input as pygem_prms


class GCM():
    """
    Global climate model data properties and functions used to automatically retrieve data.
    
    Attributes
    ----------
    name : str
        name of climate dataset.
    scenario : str
        rcp or ssp scenario (example: 'rcp26' or 'ssp585')
    realization : str
        realization from large ensemble (example: '1011.001' or '1301.020')
    """
    def __init__(self, 
                 name=str(),
                 scenario=str(),
                 realization=None):
        """
        Add variable name and specific properties associated with each gcm.
        """
        
        if pygem_prms.rgi_lon_colname not in ['CenLon_360']:
            assert 1==0, 'Longitude does not use 360 degrees. Check how negative values are handled!'
        
        # Source of climate data
        self.name = name
        
        # If multiple realizations from each model+scenario are being used,
        #   then self.realization = realization. 
        # Otherwise, the realization attribute is not considered for single 
        #   realization model+scenario simulations.
        if realization is not None:
            self.realization = realization
            
            # Set parameters for CESM2 Large Ensemble
            if self.name == 'smbb.f09_g17.LE2':
                # Standardized CESM2 Large Ensemble format (GCM/SSP)
                # Variable names
                self.temp_vn = 'tas'
                self.prec_vn = 'pr'
                self.elev_vn = 'orog'
                self.lat_vn = 'lat'
                self.lon_vn = 'lon'
                self.time_vn = 'time'
                # Variable filenames
                self.temp_fn = self.temp_vn + '_mon_' + scenario + '_' + name + '-' + realization + '.cam.h0.1980-2100.nc'
                self.prec_fn = self.prec_vn + '_mon_' + scenario + '_' + name + '-' + realization + '.cam.h0.1980-2100.nc'
                self.elev_fn = self.elev_vn + '_fx_' + scenario + '_' + name + '.cam.h0.nc'
                # Variable filepaths
                self.var_fp = pygem_prms.cesm2_fp_var_prefix + scenario + pygem_prms.cesm2_fp_var_ending
                self.fx_fp = pygem_prms.cesm2_fp_fx_prefix + scenario + pygem_prms.cesm2_fp_fx_ending
                # Extra information
                self.timestep = pygem_prms.timestep
                self.rgi_lat_colname=pygem_prms.rgi_lat_colname
                self.rgi_lon_colname=pygem_prms.rgi_lon_colname
                self.scenario = scenario
                
            # Set parameters for GFDL SPEAR Large Ensemble
            elif self.name == 'GFDL-SPEAR-MED':
                # Standardized GFDL SPEAR Large Ensemble format (GCM/SSP)
                # Variable names
                self.temp_vn = 'tas'
                self.prec_vn = 'pr'
                self.elev_vn = 'zsurf'
                self.lat_vn = 'lat'
                self.lon_vn = 'lon'
                self.time_vn = 'time'
                # Variable filenames
                self.temp_fn = self.temp_vn + '_mon_' + scenario + '_' + name + '-' + realization + 'i1p1f1_gr3_1980-2100.nc'
                self.prec_fn = self.prec_vn + '_mon_' + scenario + '_' + name + '-' + realization + 'i1p1f1_gr3_1980-2100.nc'
                self.elev_fn = self.elev_vn + '_fx_' + scenario + '_' + name + '.nc'
                # Variable filepaths
                self.var_fp = pygem_prms.gfdl_fp_var_prefix + scenario + pygem_prms.gfdl_fp_var_ending
                self.fx_fp = pygem_prms.gfdl_fp_fx_prefix + scenario + pygem_prms.gfdl_fp_fx_ending
                # Extra information
                self.timestep = pygem_prms.timestep
                self.rgi_lat_colname=pygem_prms.rgi_lat_colname
                self.rgi_lon_colname=pygem_prms.rgi_lon_colname
                self.scenario = scenario
            
        else:
            self.realization = []
            
            # Set parameters for ERA5, ERA-Interim, and CMIP5 netcdf files
            if self.name == 'ERA5':
                # Variable names
                self.temp_vn = 't2m'
                self.tempstd_vn = 't2m_std'
                self.prec_vn = 'tp'
                self.elev_vn = 'z'
                self.lat_vn = 'latitude'
                self.lon_vn = 'longitude'
                self.time_vn = 'time'
                self.lr_vn = 'lapserate'
                # Variable filenames
                self.temp_fn = pygem_prms.era5_temp_fn
                self.tempstd_fn = pygem_prms.era5_tempstd_fn
                self.prec_fn = pygem_prms.era5_prec_fn
                self.elev_fn = pygem_prms.era5_elev_fn
                self.lr_fn = pygem_prms.era5_lr_fn
                # Variable filepaths
                self.var_fp = pygem_prms.era5_fp
                self.fx_fp = pygem_prms.era5_fp
                # Extra information
                self.timestep = pygem_prms.timestep
                self.rgi_lat_colname=pygem_prms.rgi_lat_colname
                self.rgi_lon_colname=pygem_prms.rgi_lon_colname
                
            elif self.name == 'ERA-Interim':
                # Variable names
                self.temp_vn = 't2m'
                self.prec_vn = 'tp'
                self.elev_vn = 'z'
                self.lat_vn = 'latitude'
                self.lon_vn = 'longitude'
                self.time_vn = 'time'
                self.lr_vn = 'lapserate'
                # Variable filenames
                self.temp_fn = pygem_prms.eraint_temp_fn
                self.prec_fn = pygem_prms.eraint_prec_fn
                self.elev_fn = pygem_prms.eraint_elev_fn
                self.lr_fn = pygem_prms.eraint_lr_fn
                # Variable filepaths
                self.var_fp = pygem_prms.eraint_fp
                self.fx_fp = pygem_prms.eraint_fp
                # Extra information
                self.timestep = pygem_prms.timestep
                self.rgi_lat_colname=pygem_prms.rgi_lat_colname
                self.rgi_lon_colname=pygem_prms.rgi_lon_colname

            if self.name == 'ERA5-hourly' and pygem_prms.run_eb:
                # Variable names for energy balance
                self.temp_vn = 't2m'
                self.dtemp_vn = 'd2m'
                self.sp_vn = 'sp'
                self.prec_vn = 'tp'
                self.elev_vn = 'z'
                self.tcc_vn = 'tcc'
                self.SWin_vn = 'ssrd'
                self.LWin_vn = 'strd'
                self.uwind_vn = 'u10'
                self.vwind_vn = 'v10'
                self.lat_vn = 'latitude'
                self.lon_vn = 'longitude'
                self.time_vn = 'time'
                # self.lr_vn = 'lapserate'
                # Variable filenames
                self.temp_fn = 'ERA5_temp_hourly.nc'
                self.dtemp_fn = 'ERA5_dtemp_hourly.nc'
                self.sp_fn = 'ERA5_sp_hourly.nc'
                self.tcc_fn = 'ERA5_tcc_hourly.nc'
                self.LWin_fn = 'ERA5_LWin_hourly.nc'
                self.SWin_fn = 'ERA5_SWin_hourly.nc'
                self.vwind_fn = 'ERA5_vwind_hourly.nc'
                self.uwind_fn = 'ERA5_uwind_hourly.nc'
                self.prec_fn = 'ERA5_precip_hourly.nc'
                self.elev_fn = pygem_prms.era5_elev_fn
                # self.lr_fn = 'lapserates_hourly.nc' 
                # Variable filepaths
                self.var_fp = pygem_prms.era5h_fp
                self.fx_fp = pygem_prms.era5h_fp
                # Extra information
                self.timestep = pygem_prms.timestep
                self.rgi_lat_colname=pygem_prms.rgi_lat_colname
                self.rgi_lon_colname=pygem_prms.rgi_lon_colname
                
            # Standardized CMIP5 format (GCM/RCP)
            elif 'rcp' in scenario:
                # Variable names
                self.temp_vn = 'tas'
                self.prec_vn = 'pr'
                self.elev_vn = 'orog'
                self.lat_vn = 'lat'
                self.lon_vn = 'lon'
                self.time_vn = 'time'
                # Variable filenames
                self.temp_fn = self.temp_vn + '_mon_' + name + '_' + scenario + '_r1i1p1_native.nc'
                self.prec_fn = self.prec_vn + '_mon_' + name + '_' + scenario + '_r1i1p1_native.nc'
                self.elev_fn = self.elev_vn + '_fx_' + name + '_' + scenario + '_r0i0p0.nc'
                # Variable filepaths
                self.var_fp = pygem_prms.cmip5_fp_var_prefix + scenario + pygem_prms.cmip5_fp_var_ending
                self.fx_fp = pygem_prms.cmip5_fp_fx_prefix + scenario + pygem_prms.cmip5_fp_fx_ending
                # Extra information
                self.timestep = pygem_prms.timestep
                self.rgi_lat_colname=pygem_prms.rgi_lat_colname
                self.rgi_lon_colname=pygem_prms.rgi_lon_colname
                self.scenario = scenario
            
            # Standardized CMIP6 format (GCM/SSP)
            elif 'ssp' in scenario:
                # Variable names
                self.temp_vn = 'tas'
                self.prec_vn = 'pr'
                self.elev_vn = 'orog'
                self.lat_vn = 'lat'
                self.lon_vn = 'lon'
                self.time_vn = 'time'
                # Variable filenames
                self.temp_fn = name + '_' + scenario + '_r1i1p1f1_' + self.temp_vn + '.nc'
                self.prec_fn = name + '_' + scenario + '_r1i1p1f1_' + self.prec_vn + '.nc'
                self.elev_fn = name + '_' + self.elev_vn + '.nc'
                # Variable filepaths
                self.var_fp = pygem_prms.cmip6_fp_prefix + name + '/'
                self.fx_fp = pygem_prms.cmip6_fp_prefix + name + '/'
                # Extra information
                self.timestep = pygem_prms.timestep
                self.rgi_lat_colname=pygem_prms.rgi_lat_colname
                self.rgi_lon_colname=pygem_prms.rgi_lon_colname
                self.scenario = scenario
            
            
    def importGCMfxnearestneighbor_xarray(self, filename, vn, main_glac_rgi):
        """
        Import time invariant (constant) variables and extract nearest neighbor.
        
        Note: cmip5 data used surface height, while ERA-Interim data is geopotential
        
        Parameters
        ----------
        filename : str
            filename of variable
        vn : str
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

    
    def importGCMvarnearestneighbor_xarray(self, filename, vn, main_glac_rgi, dates_table, realizations=['r1i1p1f1','r4i1p1f1']):
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
        if not os.path.exists(self.var_fp + filename):
            for realization in realizations:
                filename_realization = filename.replace('r1i1p1f1','r4i1p1f1')
                if os.path.exists(self.var_fp + filename_realization):
                    filename = filename_realization
            
        data = xr.open_dataset(self.var_fp + filename)
        glac_variable_series = np.zeros((main_glac_rgi.shape[0],dates_table.shape[0]))
        
        # Check GCM provides required years of data
        years_check = pd.Series(data['time']).apply(lambda x: int(x.strftime('%Y')))
        assert years_check.max() >= dates_table.year.max(), self.name + ' does not provide data out to ' + str(dates_table.year.max())
        assert years_check.min() <= dates_table.year.min(), self.name + ' does not provide data back to ' + str(dates_table.year.min())
        
        # Determine the correct time indices
        if self.timestep == 'monthly' and not pygem_prms.run_eb:
            start_idx = (np.where(pd.Series(data[self.time_vn]).apply(lambda x: x.strftime('%Y-%m')) == 
                                  dates_table['date'].apply(lambda x: x.strftime('%Y-%m'))[0]))[0][0]
            end_idx = (np.where(pd.Series(data[self.time_vn]).apply(lambda x: x.strftime('%Y-%m')) == 
                                dates_table['date']
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
        elif self.timestep == 'daily' and not pygem_prms.run_eb:
            start_idx = (np.where(pd.Series(data[self.time_vn])
                                  .apply(lambda x: x.strftime('%Y-%m-%d')) == dates_table['date']
                                  .apply(lambda x: x.strftime('%Y-%m-%d'))[0]))[0][0]
            end_idx = (np.where(pd.Series(data[self.time_vn])
                                .apply(lambda x: x.strftime('%Y-%m-%d')) == dates_table['date']
                                .apply(lambda x: x.strftime('%Y-%m-%d'))[dates_table.shape[0] - 1]))[0][0]
        elif pygem_prms.run_eb:
            #format start and end dates to match that of the netcdf time variable
            #netCDF from ERA5 hourly should be datetime64 (numpy) so this code will just do that rather than check
            #what the format actually is
            assert data[self.time_vn].dtype != 'datetime64[sn]', 'check GCM time format'
            start_formatted = dates_table.loc[0,'date'].to_datetime64()
            end_formatted = dates_table.loc[dates_table.shape[0]-1,'date'].to_datetime64()
            start_idx = np.where(data[self.time_vn].values == start_formatted)[0][0]
            end_idx = np.where(data[self.time_vn].values == end_formatted)[0][0]
        glac_variable_series = np.zeros((main_glac_rgi.shape[0],end_idx-start_idx+1))
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
        elif pygem_prms.run_eb:
            #this code is for using the data that is already selected to Gulkana; thus nearest neighbor is unnecessary
            for glac in range(main_glac_rgi.shape[0]):
                glac_variable_series[glac,:] = data[vn][start_idx:end_idx+1,0,0].values
        else:
            #  argmin() finds the minimum distance between the glacier lat/lon and the GCM pixel; .values is used to 
            #  extract the position's value as opposed to having an array
            lat_nearidx = (np.abs(main_glac_rgi[self.rgi_lat_colname].values[:,np.newaxis] - 
                                  data.variables[self.lat_vn][:].values).argmin(axis=1))
            lon_nearidx = (np.abs(main_glac_rgi[self.rgi_lon_colname].values[:,np.newaxis] - 
                                  data.variables[self.lon_vn][:].values).argmin(axis=1))
            # Find unique latitude/longitudes
            latlon_nearidx = list(zip(lat_nearidx, lon_nearidx))
            latlon_nearidx_unique = list(set(latlon_nearidx))
            # Create dictionary of time series for each unique latitude/longitude
            glac_variable_dict = {}
            for latlon in latlon_nearidx_unique:                
                if 'expver' in data.keys():
                    expver_idx = 0
                    glac_variable_dict[latlon] = data[vn][start_idx:end_idx+1, expver_idx, latlon[0], latlon[1]].values
                else:
                    glac_variable_dict[latlon] = data[vn][start_idx:end_idx+1, latlon[0], latlon[1]].values
                
            # Convert to series
            glac_variable_series = np.array([glac_variable_dict[x] for x in latlon_nearidx])  

        # Perform corrections to the data if necessary
        # Surface air temperature corrections
        if vn in ['tas', 't2m', 'T2']:
            if 'units' in data[vn].attrs and data[vn].attrs['units'] == 'K':
                # Convert from K to deg C
                glac_variable_series = glac_variable_series - 273.15
            else:
                print('Check units of air temperature from GCM is degrees C.')
        elif vn in ['t2m_std']:
            if 'units' in data[vn].attrs and data[vn].attrs['units'] not in ['C', 'K']:
                print('Check units of air temperature standard deviation from GCM is degrees C or K')
        # Precipitation corrections
        # If the variable is precipitation
        elif vn in ['pr', 'tp', 'TOTPRECIP']:
            # If the variable has units and those units are meters (ERA Interim)
            if 'units' in data[vn].attrs and data[vn].attrs['units'] == 'm':
                pass
            # Elseif the variable has units and those units are kg m-2 s-1 (CMIP5/CMIP6)
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
            if self.timestep == 'monthly' and self.name != 'COAWST' and not pygem_prms.run_eb:
                # Convert from meters per day to meters per month (COAWST data already 'monthly accumulated precipitation')
                if 'daysinmonth' in dates_table.columns:
                    glac_variable_series = glac_variable_series * dates_table['daysinmonth'].values[np.newaxis,:]
        elif vn in ['sp','d2m','tcc','ssrd','u10','v10']:
            # code in units check for these variables
            print('!! Not checking units for any EB variables')
            pass
        elif vn != self.lr_vn:
            print('Check units of air temperature or precipitation')
        return glac_variable_series, time_series
    
class AWS():
    def __init__(self,fp,dates_table):
        # Variable names for energy balance
        self.temp_vn = 'temp'
        self.prec_vn = 'tp'
        self.wind_vn = 'wind'
        self.winddir_vn = 'winddir'
        self.dtemp_vn = 'dtemp'
        self.rh_vn = 'rh'
        self.sp_vn = 'sp'
        self.elev_vn = 'z'
        self.tcc_vn = 'tcc'
        self.SWin_vn = 'SWin'
        self.SWout_vn = 'SWout'
        self.LWin_vn = 'LWin'
        self.LWout_vn = 'LWout'
        self.NR_vn = 'NR'
        # File name
        self.fn = fp
        df = pd.read_csv(fp,index_col=0)

        data_start = pd.to_datetime(df.index.to_numpy()[0])
        data_end = pd.to_datetime(df.index.to_numpy()[-1])
        assert dates_table.date[0] >= data_start, 'Check input dates: start date before range of AWS data'
        assert dates_table.date.to_numpy()[-1] <= data_end, 'Check input dates: end date after range of AWS data'
        df = df.set_index(pd.date_range(data_start,data_end,freq='H'))
        df = df.loc[dates_table.date[0]:dates_table.date.to_numpy()[-1]]
        
        self.temp = df[self.temp_vn].to_numpy()
        self.dtemp = df[self.dtemp_vn].to_numpy()
        self.tp = df[self.prec_vn].to_numpy()
        self.rh = df[self.rh_vn].to_numpy()
        self.SWin = df[self.SWin_vn].to_numpy()
        self.SWout = df[self.SWout_vn].to_numpy()
        self.LWin = df[self.LWin_vn].to_numpy()
        self.LWout = df[self.LWout_vn].to_numpy()
        self.wind = df[self.wind_vn].to_numpy()
        self.sp = df[self.sp_vn].to_numpy()
        self.elev = df[self.elev_vn].to_numpy()[0]

        # DATA WHICH MAY NOT BE IN FILE (NOT NEEDED)
        nans = np.ones_like(self.temp)*np.nan
        if self.NR_vn in df.columns:
            self.NR = df[self.NR_vn].to_numpy()
        else:
            self.NR = nans

        if self.winddir_vn in df.columns:
            self.winddir = df[self.winddir_vn].to_numpy()
        else:
            self.winddir = nans

        if self.tcc_vn in df.columns:
            self.tcc = df[self.tcc_vn].to_numpy()
        else:
            self.tcc = nans
        return