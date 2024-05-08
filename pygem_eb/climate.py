import numpy as np
import xarray as xr
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import threading
import pygem_eb.input as eb_prms

class Climate():
    """
    Climate-related functions. If use_AWS = True in the input, the climate 
    dataset will be filled with as many AWS variables as exist before turning 
    to reanalysis data to fill the necessary variables. If use_AWS = False,
    only reanalysis data will be used.
    """
    def __init__(self,args,glacier_table):
        """
        Initializes glacier information and creates the dataset where 
        climate data will be stored.
        """
        # load args and run information
        self.args = args
        self.dates = pd.date_range(args.startdate,args.enddate,freq='h')
        self.dates_UTC = self.dates - eb_prms.timezone
        n_time = len(self.dates)
        n_bins = args.n_bins
        bin_idx = np.arange(0,n_bins)

        # glacier cenlat and lon
        if eb_prms.glac_no == ['01.00570']:
            self.lat = eb_prms.site_df.loc[eb_prms.site]['lat']
            self.lon = eb_prms.site_df.loc[eb_prms.site]['lon']
        else:
            self.lat = glacier_table['CenLat'].values
            self.lon = glacier_table['CenLon'].values

        # define reanalysis variables
        self.get_vardict()
        self.all_vars = ['temp','tp','rh','wind','sp','SWin',
                            'LWin','bcwet','bcdry','dustwet','dustdry']

        # create empty dataset
        nans = np.ones(n_time)*np.nan
        bin_nans = np.array([nans]*n_bins).reshape(n_bins,n_time)
        self.cds = xr.Dataset(data_vars = dict(
                bin_elev = (['bin'],eb_prms.bin_elev,{'units':'m a.s.l.'}),
                SWin = (['time'],nans,{'units':'J m-2'}),
                SWout = (['time'],nans,{'units':'J m-2'}),
                LWin = (['time'],nans,{'units':'J m-2'}),
                LWout = (['time'],nans,{'units':'J m-2'}),
                NR = (['time'],nans,{'units':'J m-2'}),
                tcc = (['time'],nans,{'units':'1'}),
                rh = (['time'],nans,{'units':'%'}),
                uwind = (['time'],nans,{'units':'m s-1'}),
                vwind = (['time'],nans,{'units':'m s-1'}),
                wind = (['time'],nans,{'units':'m s-1'}),
                winddir = (['time'],nans,{'units':'deg'}),
                bcdry = (['time'],nans,{'units':'kg m-2 s-1'}),
                bcwet = (['time'],nans,{'units':'kg m-2 s-1'}),
                dustdry = (['time'],nans,{'units':'kg m-2 s-1'}),
                dustwet = (['time'],nans,{'units':'kg m-2 s-1'}),
                bin_temp = (['bin','time'],bin_nans,{'units':'C'}),
                bin_tp = (['bin','time'],bin_nans,{'units':'m'}),
                bin_sp = (['bin','time'],bin_nans,{'units':'Pa'})
                ),
                coords = dict(
                    bin=(['bin'],bin_idx),
                    time=(['time'],self.dates)
                    ))
    
        self.n_time = n_time
        self.n_bins = n_bins
        return
    
    def get_AWS(self,fp):
        """
        Loads available AWS data and determines which variables
        are remaining to come from reanalysis.
        """
        # load data
        df = pd.read_csv(fp,index_col=0)
        df = df.set_index(pd.to_datetime(df.index))

        # check dates of data match input dates
        data_start = pd.to_datetime(df.index.to_numpy()[0])
        data_end = pd.to_datetime(df.index.to_numpy()[-1])
        assert self.dates[0] >= data_start, 'Check input dates: start date before range of AWS data'
        assert self.dates[len(self.dates)-1] <= data_end, 'Check input dates: end date after range of AWS data'
        df = df.loc[self.dates]

        # get AWS elevation
        self.AWS_elev = df.iloc[0]['z']

        # get the available variables
        all_AWS_vars = ['temp','tp','rh','wind','sp','SWin','SWout',
                    'LWin','LWout','bcwet','bcdry','dustwet','dustdry']
        AWS_vars = df.columns
        self.measured_vars = list(set(all_AWS_vars) & set(AWS_vars))
        
        # extract and store data
        for var in self.measured_vars:
            data = df[var]
            # adjust elevation-dependent variables
            if var in ['temp','tp','sp']:
                data = self.bin_adjust(data, var, self.AWS_elev)
                varname = 'bin_' + var
            else:
                varname = var
            self.cds[varname].values = data

        # figure out which data is still needed from reanalysis
        need_vars = [e for e in self.all_vars if e not in AWS_vars]
        return need_vars
    
    def get_reanalysis(self,vars):
        """
        Utilizes threading to efficiently fetch reanalysis climate
        data variables.
        """
        dates = self.dates_UTC
        lat = self.lat
        lon = self.lon
        
        # get reanalysis data geopotential
        z_fp = self.reanalysis_fp + self.var_dict['elev']['fn']
        z_vn = self.var_dict['elev']['vn']
        zds = xr.open_dataset(z_fp).sel(lat=lat,lon=lon,method='nearest')
        zds = self.check_units('elev',zds)
        self.reanalysis_elev = zds.isel(time=0)[z_vn].values
        assert self.reanalysis_elev < 8000

        # define worker function for threading
        def access_cell(fn, var, result_dict):
            # open and check units of climate data
            ds = xr.open_dataset(fn)
            if var != 'elev':
                ds = ds.sel(time=dates)
            ds = self.check_units(var,ds)
            # index by lat and lon
            vn = self.var_dict[var]['vn']
            data = ds.sel(lat=lat, lon=lon, method='nearest')[vn].values
            # adjust elevation-dependent variables
            if var in ['temp','tp','sp']:
                data = self.bin_adjust(data, var, self.reanalysis_elev)
            # store result
            result_dict[var] = data
            ds.close()
        
        # initiate variables
        all_data = {}
        threads = []
        # loop through vars to initiate threads
        for var in vars:
            if var == 'wind':
                fn = self.reanalysis_fp + self.var_dict['uwind']['fn']
                thread = threading.Thread(target=access_cell,
                                        args=(fn, 'uwind', all_data))
                thread.start()
                threads.append(thread)
                var = 'vwind'
            fn = self.reanalysis_fp + self.var_dict[var]['fn']
            thread = threading.Thread(target=access_cell,
                                    args=(fn, var, all_data))
            thread.start()
            threads.append(thread)
        # join threads
        for thread in threads:
            thread.join()

        # store data
        for var in vars:
            if var in ['temp','tp','sp']:
                varname = 'bin_'+var
            elif var == 'wind':
                varname = 'uwind'
                self.cds[varname].values = all_data[varname]
                varname = 'vwind'
                var = 'vwind'
            else:
                varname = var
            self.cds[varname].values = all_data[var]
        return

    def bin_adjust(self, data, var, elev_data):
        """
        Adjusts elevation-dependent climate variables (temperature, precip,
        surface pressure).

        Parameters
        =========
        """
        out = np.zeros((self.n_bins,self.n_time))
        
        #Loop through each elevation bin and adjust climate variables
        for idx,z in enumerate(eb_prms.bin_elev):
            # TEMPERATURE: correct according to lapserate
            if var == 'temp':
                out[idx,:] = data + eb_prms.lapserate*(z-elev_data)

            # PRECIP: correct according to lapserate, precipitation factor
            elif var == 'tp':
                if '__iter__' in dir(eb_prms.kp):
                    out[idx,:] = data*(1+eb_prms.precgrad*(z-elev_data))*eb_prms.kp[idx]
                else:
                    out[idx,:] = data*(1+eb_prms.precgrad*(z-elev_data))*eb_prms.kp

            # SURFACE PRESSURE: correct according to barometric law
            elif var == 'sp':
                out[idx,:] = data
                # **** FIGURE THIS OUT TO ACTUALLY ADJUST BY ELEVATION
                # temp_data = self.cds.sel(bin=idx)['bin_temp']
                # out[idx,:] = data*np.power((data + eb_prms.lapserate*(z-elev_data)+273.15)/(temp_data+273.15),
                #                 -eb_prms.gravity*eb_prms.molarmass_air/(eb_prms.R_gas*eb_prms.lapserate))
        return out
    
    def check_ds(self):
        # need to get wind from u/v components in reanalysis data      
        wind = self.cds['wind'].values
        if np.all(np.isnan(wind)):
            uwind = self.cds['uwind'].values
            vwind = self.cds['vwind'].values
            wind = np.sqrt(np.power(uwind,2)+np.power(vwind,2))
            winddir = np.arctan2(-uwind,-vwind) * 180 / np.pi
            self.cds['wind'].values = wind
            self.cds['winddir'].values = winddir

        # adjust MERRA-2 temperature bias (varies by month of the year)
        temp_filled = True if ~eb_prms.use_AWS else 'temp' in self.need_vars
        if eb_prms.reanalysis == 'MERRA2' and temp_filled:
            self.adjust_temp_bias()

        # check all variables are there
        for var in self.all_vars:
            varname = 'bin_'+var if var in ['temp','tp','sp'] else var
            data = self.cds[varname].values
            assert ~np.all(np.isnan(data)), f'{var} failed'

        if eb_prms.store_climate:
            name = eb_prms.glac_name
            out_fp = eb_prms.output_name.replace(name,name+'_climate')
            self.cds.to_netcdf(out_fp+'.nc')
        return
    
    def check_units(self,var,ds):
        model_units = {'temp':'C','uwind':'m s-1','vwind':'m s-1',
                       'rh':'%','sp':'Pa','tp':'m s-1','elev':'m',
                       'SWin':'J m-2', 'LWin':'J m-2', 'tcc':'1',
                       'bcdry':'kg m-2 s-1', 'bcwet':'kg m-2 s-1',
                       'dustdry':'kg m-2 s-1', 'dustwet':'kg m-2 s-1'}
        vn = list(ds.keys())[0]
        units_in =  ds[vn].attrs['units']
        units_out = model_units[var]
        if units_in != units_out:
            if var == 'temp' and units_in == 'K':
                ds[vn] = ds[vn] - 273.15
            elif var == 'rh' and units_in == '1':
                ds[vn] = ds[vn] * 100
            elif var == 'tp' and units_in == 'kg m-2 s-1':
                ds[vn] = ds[vn] / 1000 * 3600
            elif var == 'SWin' and units_in == 'W m-2':
                ds[vn] = ds[vn] * 3600
            elif var == 'LWin' and units_in == 'W m-2':
                ds[vn] = ds[vn] * 3600
            elif var == 'elev' and units_in == 'm+2 s-2':
                ds[vn] = ds[vn] / eb_prms.gravity
            else:
                print(f'WARNING: units did not match for {var} but were not updated')
                print(f'Previously {units_in}; should be {units_out}')
                print('Make a manual change in utils.py')
                quit()
        return ds
    
    def adjust_temp_bias(self):
        bias_df = pd.read_csv(eb_prms.temp_bias_fp)
        for month in bias_df.index:
            bias = bias_df.loc[month]['bias']
            idx = np.where(self.cds.coords['time'].dt.month.values == month)[0]
            self.cds['bin_temp'][{'time':idx}] = self.cds['bin_temp'][{'time':idx}] + bias
        return

    def getVaporPressure(self,tempC):
        """
        Returns vapor pressure in Pa from air temperature in Celsius
        """
        return 610.94*np.exp(17.625*tempC/(tempC+243.04))
    
    def getDewTemp(self,vap):
        """
        Returns dewpoint air temperature in C from vapor pressure in Pa
        """
        return 243.04*np.log(vap/610.94)/(17.625-np.log(vap/610.94))

    def RMSE(self,y,pred):
        N = len(y)
        RMSE = np.sqrt(np.sum(np.square(pred-y))/N)
        return RMSE

    def get_vardict(self):
        # Update filenames for MERRA-2 (need grid lat/lon)
        self.reanalysis_fp = eb_prms.main_directory + '/../climate_data/'
        self.var_dict = {'temp':{'fn':[],'vn':[]},
            'rh':{'fn':[],'vn':[]},'sp':{'fn':[],'vn':[]},
            'tp':{'fn':[],'vn':[]},'tcc':{'fn':[],'vn':[]},
            'SWin':{'fn':[],'vn':[]},'LWin':{'fn':[],'vn':[]},
            'uwind':{'fn':[],'vn':[]},'vwind':{'fn':[],'vn':[]},
            'bcdry':{'fn':[],'vn':[]},'bcwet':{'fn':[],'vn':[]},
            'dustdry':{'fn':[],'vn':[]},'dustwet':{'fn':[],'vn':[]},
            'elev':{'fn':[],'vn':[]},'time':{'fn':'','vn':''},
            'lat':{'fn':'','vn':''}, 'lon':{'fn':'','vn':''}}
        if eb_prms.reanalysis == 'MERRA2':
            self.reanalysis_fp += 'MERRA2/'
            flat = str(int(np.floor(self.lat/10)*10))
            flon = str(int(np.floor(self.lon/10)*10))
            self.var_dict['temp']['vn'] = 'T2M'
            self.var_dict['rh']['vn'] = 'RH2M'
            self.var_dict['sp']['vn'] = 'PS'
            self.var_dict['tp']['vn'] = 'PRECTOTCORR'
            self.var_dict['elev']['vn'] = 'PHIS'
            self.var_dict['tcc']['vn'] = 'CLDTOT'
            self.var_dict['SWin']['vn'] = 'SWGDN'
            self.var_dict['LWin']['vn'] = 'LWGAB'
            self.var_dict['uwind']['vn'] = 'U2M'
            self.var_dict['vwind']['vn'] = 'V2M'
            self.var_dict['bcwet']['vn'] = 'BCWT002'
            self.var_dict['bcdry']['vn'] = 'BCDP002'
            self.var_dict['dustwet']['vn'] = 'DUWT003'
            self.var_dict['dustdry']['vn'] = 'DUDP003'
            self.time_vn = 'time'
            self.lat_vn = 'lat'
            self.lon_vn = 'lon'
            self.elev_vn = self.var_dict['elev']['vn']
            # Variable filenames
            self.var_dict['temp']['fn'] = f'T2M/MERRA2_T2M_{flat}_{flon}.nc'
            self.var_dict['rh']['fn'] = f'RH2M/MERRA2_RH2M_{flat}_{flon}.nc'
            self.var_dict['sp']['fn'] = f'PS/MERRA2_PS_{flat}_{flon}.nc'
            self.var_dict['tcc']['fn'] = f'CLDTOT/MERRA2_CLDTOT_{flat}_{flon}.nc'
            self.var_dict['LWin']['fn'] = f'LWGAB/MERRA2_LWGAB_{flat}_{flon}.nc'
            self.var_dict['SWin']['fn'] = f'SWGDN/MERRA2_SWGDN_{flat}_{flon}.nc'
            self.var_dict['vwind']['fn'] = f'V2M/MERRA2_V2M_{flat}_{flon}.nc'
            self.var_dict['uwind']['fn'] = f'U2M/MERRA2_U2M_{flat}_{flon}.nc'
            self.var_dict['tp']['fn'] = f'PRECTOTCORR/MERRA2_PRECTOTCORR_{flat}_{flon}.nc'
            self.var_dict['elev']['fn'] = f'MERRA2constants.nc4'
            self.var_dict['bcwet']['fn'] = f'BCWT002/MERRA2_BCWT002_{flat}_{flon}.nc'
            self.var_dict['bcdry']['fn'] = f'BCDP002/MERRA2_BCDP002_{flat}_{flon}.nc'
            self.var_dict['dustwet']['fn'] = f'DUWT003/MERRA2_DUWT003_{flat}_{flon}.nc'
            self.var_dict['dustdry']['fn'] = f'DUDP003/MERRA2_DUDP003_{flat}_{flon}.nc'
        elif eb_prms.reanalysis == 'ERA5':
            self.reanalysis_fp += 'ERA5/'
            # Variable names for energy balance
            self.var_dict['temp']['vn'] = 't2m'
            self.var_dict['rh']['vn'] = 'rh'
            self.var_dict['sp']['vn'] = 'sp'
            self.var_dict['tp']['vn'] = 'tp'
            self.var_dict['elev']['vn'] = 'z'
            self.var_dict['tcc']['vn'] = 'tcc'
            self.var_dict['SWin']['vn'] = 'ssrd'
            self.var_dict['LWin']['vn'] = 'strd'
            self.var_dict['uwind']['vn'] = 'u10'
            self.var_dict['vwind']['vn'] = 'v10'
            self.var_dict['bcwet']['vn'] = 'BCWT002'
            self.var_dict['bcdry']['vn'] = 'BCDP002'
            self.var_dict['dustwet']['vn'] = 'DUWT003'
            self.var_dict['dustdry']['vn'] = 'DUDP003'
            self.time_vn = 'time'
            self.lat_vn = 'latitude'
            self.lon_vn = 'longitude'
            self.elev_vn = self.var_dict['elev']['vn']
            # Variable filenames
            self.var_dict['temp']['fn'] = 'ERA5_temp_hourly.nc'
            self.var_dict['rh']['fn'] = 'ERA5_rh_hourly.nc'
            self.var_dict['sp']['fn'] = 'ERA5_sp_hourly.nc'
            self.var_dict['tcc']['fn'] = 'ERA5_tcc_hourly.nc'
            self.var_dict['LWin']['fn'] = 'ERA5_LWin_hourly.nc'
            self.var_dict['SWin']['fn'] = 'ERA5_SWin_hourly.nc'
            self.var_dict['vwind']['fn'] = 'ERA5_vwind_hourly.nc'
            self.var_dict['uwind']['fn'] = 'ERA5_uwind_hourly.nc'
            self.var_dict['tp']['fn'] = 'ERA5_tp_hourly.nc'
            self.var_dict['elev']['fn'] = 'pygems elev path'
            self.var_dict['bcwet']['fn'] = 'BCWT002/MERRA2_BCWT002_LAT_LON.nc'
            self.var_dict['bcdry']['fn'] = 'BCDP002/MERRA2_BCDP002_LAT_LON.nc'
            self.var_dict['dustwet']['fn'] = 'DUWT003/MERRA2_DUWT003_LAT_LON.nc'
            self.var_dict['dustdry']['fn'] = 'DUDP003/MERRA2_DUDP003_LAT_LON.nc'

    # UNUSED --- MOVE TO PYGEM EB GRAVEYARD
    def getGrainSizeModel(self,initSSA,var):
        path = '/home/claire/research/PyGEM-EB/pygem_eb/data/'
        fn = 'drygrainsize(SSAin=##).nc'.replace('##',str(initSSA))
        ds = xr.open_dataset(path+fn)

        # extract X values (temperature, density, temperature gradient)
        T = ds.coords['TVals'].to_numpy()
        p = ds.coords['DENSVals'].to_numpy()
        dTdz = ds.coords['DTDZVals'].to_numpy()
        
        # form meshgrid and corresponding y for model input
        TT,pp,ddTT = np.meshgrid(T,p,dTdz)
        X = np.array([TT.flatten(),pp.flatten(),ddTT.flatten()]).T
        y = []
        for tpd in X:
            tsel = tpd[0]
            psel = tpd[1]
            dtsel = tpd[2]
            y.append(ds.sel(TVals=tsel,DENSVals=psel,DTDZVals=dtsel)[var].to_numpy())
        y = np.array(y)

        # dr0 can be modeled linearly; kappa and tau should be log transformed
        transform = 'none' if var=='dr0mat' else 'log'
        if transform == 'log':
            y = np.log(y)

        # randomly select training and testing data to store statistics
        np.random.seed(9)
        percent_train = 90
        N = len(y)
        train_mask = np.zeros_like(y,dtype=np.bool_)
        train_mask[np.random.permutation(N)[:int(N*percent_train / 100)]] = True
        X_train = X[train_mask,:]
        X_val = X[np.logical_not(train_mask),:]
        y_train = y[train_mask].reshape(-1,1)
        y_val = y[np.logical_not(train_mask)].reshape(-1,1)

        # train rain forest model
        rf = RandomForestRegressor(max_depth=15,n_estimators=10)
        y_train = y_train.flatten()
        rf.fit(X_train,y_train)
        trainloss = self.RMSE(y_train,rf.predict(X_train))
        valloss = self.RMSE(y_val,rf.predict(X_val))

        return rf,trainloss,valloss
        