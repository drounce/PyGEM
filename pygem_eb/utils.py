import torch
import numpy as np
import xarray as xr
from sklearn.ensemble import RandomForestRegressor

class Utils():
    """
    Utility functions.
    """
    def __init__(self):

        return

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
        