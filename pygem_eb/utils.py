import numpy as np
import xarray as xr
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import pygem_eb.input as eb_prms

class Utils():
    """
    Utility functions.
    """
    def __init__(self, args,glacier_table):
        self.args = args
        self.lat = glacier_table['CenLat'].values
        self.lon = glacier_table['CenLon'].values
        return

    def getBinnedClimate(self, temp_data, tp_data, sp_data,
                         n_timesteps, elev_data, bin_elev = eb_prms.bin_elev):
        """
        Adjusts elevation-dependent climate variables (temperature, precip,
        surface pressure, dewpoint temperature and relative humidity).

        Parameters
        =========
        """
        n_bins = len(bin_elev)
        temp = np.zeros((n_bins,n_timesteps))
        tp = np.zeros((n_bins,n_timesteps))
        sp = np.zeros((n_bins,n_timesteps))
        
        #Loop through each elevation bin and adjust climate variables
        for idx,z in enumerate(bin_elev):
            # TEMPERATURE: correct according to lapserate
            temp[idx,:] = temp_data + eb_prms.lapserate*(z-elev_data)

            # PRECIP: correct according to lapserate, precipitation factor
            if self.args.climate_input in ['GCM']:
                if '__iter__' in dir(eb_prms.kp):
                    tp[idx,:] = tp_data*(1+eb_prms.precgrad*(z-elev_data))*eb_prms.kp[idx]
                else:
                    tp[idx,:] = tp_data*(1+eb_prms.precgrad*(z-elev_data))*eb_prms.kp
            else:
                tp[idx,:] = tp_data*(1+eb_prms.precgrad*(z-elev_data))

            # SURFACE PRESSURE: correct according to barometric law
            sp[idx,:] = sp_data*np.power((temp_data + eb_prms.lapserate*(z-elev_data)+273.15)/(temp_data+273.15),
                                -eb_prms.gravity*eb_prms.molarmass_air/(eb_prms.R_gas*eb_prms.lapserate))
        
        return temp, tp, sp
    
    def adjust_temp_bias(self,climateds):
        bias_df = pd.read_csv(eb_prms.temp_bias_fp)
        for month in bias_df.index:
            bias = bias_df.loc[month]['bias']
            idx = np.where(climateds.coords['time'].dt.month.values == month)[0]
            climateds['bin_temp'][{'time':idx}] = climateds['bin_temp'][{'time':idx}] + bias
        return climateds

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

    def getGrainSizeModel(self,initSSA,var):
        # UNUSED
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
        
    def RMSE(self,y,pred):
        N = len(y)
        RMSE = np.sqrt(np.sum(np.square(pred-y))/N)
        return RMSE
