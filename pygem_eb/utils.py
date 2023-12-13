import torch
import numpy as np
import xarray as xr
from sklearn.ensemble import RandomForestRegressor
import pygem_eb.input as eb_prms


class Utils():
    """
    Utility functions.
    """
    def __init__(self, args):
        self.args = args
        return

    def getBinnedClimate(self, temp_data, tp_data, sp_data, dtemp_data, rh_data, 
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
        dtemp = np.zeros((n_bins,n_timesteps))
        
        #Loop through each elevation bin and adjust climate variables
        for idx,z in enumerate(bin_elev):
            # TEMPERATURE: correct according to lapserate
            temp[idx,:] = temp_data + eb_prms.lapserate*(z-elev_data)

            # PRECIP: correct according to lapserate, precipitation factor
            if self.args.climate_input in ['GCM']:
                if len(np.array(eb_prms.kp).flatten()) > 1:
                    tp[idx,:] = tp_data*(1+eb_prms.precgrad*(z-elev_data))*eb_prms.kp[idx]
                else:
                    tp[idx,:] = tp_data*(1+eb_prms.precgrad*(z-elev_data))*eb_prms.kp
            else:
                tp[idx,:] = tp_data*(1+eb_prms.precgrad*(z-elev_data))

            # SURFACE PRESSURE: correct according to barometric law
            sp[idx,:] = sp_data*np.power((temp_data + eb_prms.lapserate*(z-elev_data)+273.15)/(temp_data+273.15),
                                -eb_prms.gravity*eb_prms.molarmass_air/(eb_prms.R_gas*eb_prms.lapserate))
            
            # RH / DTEMP: if RH is not empty, get dtemp data from it; or vice versa
            rh_empty = np.all(np.isnan(rh_data))
            dtemp_empty = np.all(np.isnan(dtemp_data))
            assert rh_empty or dtemp_empty, 'Input either dewpoint temperature or humidity data'
            if not rh_empty: 
                vap_data = rh_data / 100 * self.getVaporPressure(temp[idx,:])
                dtemp_data = self.getDewTemp(vap_data)
                dtemp[idx,:] = dtemp_data + eb_prms.lapserate_dew*(z-elev_data)
                rh = np.array([rh_data]*n_bins)
            elif not dtemp_empty:
                rh = self.getVaporPressure(dtemp_data) / self.getVaporPressure(temp) * 100
        return temp, tp, sp, dtemp, rh
    
    def getVaporPressure(self,tempC):
        """
        Returns vapor pressure in Pa from air temperature in Celsius
        """
        return 610.94*np.exp(17.625*tempC/(tempC+243.04))
    
    def getDewTemp(self,vap):
        """
        Returns air temperature in C from vapor pressure in Pa
        """
        return 243.04*np.log(vap/610.94)/(17.625-np.log(vap/610.94))

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

        
    def RMSE(self,y,pred):
        N = len(y)
        RMSE = np.sqrt(np.sum(np.square(pred-y))/N)
        return RMSE
        