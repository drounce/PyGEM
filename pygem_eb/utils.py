import torch
import numpy as np
import xarray as xr
from sklearn.ensemble import RandomForestRegressor
import pygem_eb.input as eb_prms


class Utils():
    """
    Utility functions.
    """
    def __init__(layers):

        return

    # ============= LAYERS FUNCTIONS ============= 

    def addLayers(self,layers,layers_to_add):
        """
        Adds layers to layers class.

        Parameters
        ----------
        layers_to_add : pd.Dataframe
            Contains temperature 'T', water content 'w', height 'h', type 't', dry mass 'drym'
        """
        layers.nlayers += len(layers_to_add.loc['T'].values)
        layers.ltemp = np.append(layers_to_add.loc['T'].values,layers.ltemp)
        layers.lwater = np.append(layers_to_add.loc['w'].values,layers.lwater)
        layers.lheight = np.append(layers_to_add.loc['h'].values,layers.lheight)
        layers.ltype = np.append(layers_to_add.loc['t'].values,layers.ltype)
        layers.ldrymass = np.append(layers_to_add.loc['drym'].values,layers.ldrymass)
        layers.lrefreeze = np.append(0,layers.lrefreeze)
        # Only way to add a layer is with new snow, so layer new snow = layer dry mass
        layers.lnewsnow = np.append(layers_to_add.loc['drym'].values,layers.lnewsnow)
        layers.grainsize = np.append(eb_prms.fresh_grainsize,layers.grainsize)
        layers.lBC = np.append(eb_prms.BC_freshsnow,layers.lBC)
        layers.ldust = np.append(eb_prms.dust_freshsnow,layers.ldust)
        layers = self.updateLayerProperties()
        return layers
    
    def removeLayer(self,layers,layer_to_remove):
        """
        Removes a single layer from layers class.

        Parameters
        ----------
        layer_to_remove : int
            index of layer to remove
        """
        layers.nlayers -= 1
        layers.ltemp = np.delete(layers.ltemp,layer_to_remove)
        layers.lwater = np.delete(layers.lwater,layer_to_remove)
        layers.lheight = np.delete(layers.lheight,layer_to_remove)
        layers.ltype = np.delete(layers.ltype,layer_to_remove)
        layers.ldrymass = np.delete(layers.ldrymass,layer_to_remove)
        layers.lrefreeze = np.delete(layers.lrefreeze,layer_to_remove)
        layers.lnewsnow = np.delete(layers.lnewsnow,layer_to_remove)
        layers.grainsize = np.delete(layers.grainsize,layer_to_remove)
        layers.lBC = np.delete(layers.lBC,layer_to_remove)
        layers.ldust = np.delete(layers.ldust,layer_to_remove)
        layers = self.updateLayerProperties()
        return
    
    def splitLayer(self,layers,layer_to_split):
        """
        Splits a single layer into two layers with half the height, mass, and water content.
        """
        if (layers.nlayers+1) < eb_prms.max_nlayers:
            l = layer_to_split
            layers.nlayers += 1
            layers.ltemp = np.insert(layers.ltemp,l,layers.ltemp[l])
            layers.ltype = np.insert(layers.ltype,l,layers.ltype[l])
            layers.grainsize = np.insert(layers.grainsize,l,layers.grainsize[l])

            layers.lwater[l] = layers.lwater[l]/2
            layers.lwater = np.insert(layers.lwater,l,layers.lwater[l])
            layers.lheight[l] = layers.lheight[l]/2
            layers.lheight = np.insert(layers.lheight,l,layers.lheight[l])
            layers.ldrymass[l] = layers.ldrymass[l]/2
            layers.ldrymass = np.insert(layers.ldrymass,l,layers.ldrymass[l])
            layers.lrefreeze[l] = layers.lrefreeze[l]/2
            layers.lrefreeze = np.insert(layers.lrefreeze,l,layers.lrefreeze[l])
            layers.lnewsnow[l] = layers.lnewsnow[l]/2
            layers.lnewsnow = np.insert(layers.lnewsnow,l,layers.lnewsnow[l])
            layers.lBC[l] = layers.lBC[l]/2
            layers.lBC = np.insert(layers.lBC,l,layers.lBC[l])
            layers.ldust[l] = layers.ldust[l]/2
            layers.ldust = np.insert(layers.ldust,l,layers.ldust[l])
        layers = self.updateLayerProperties()
        return layers

    def mergeLayers(self,layers,layer_to_merge):
        """
        Merges two layers, summing height, mass and water content and averaging other properties.
        Layers merged will be the index passed and the layer below it
        """
        l = layer_to_merge
        layers.ldensity[l+1] = np.sum(layers.ldensity[l:l+2]*layers.ldrymass[l:l+2])/np.sum(layers.ldrymass[l:l+2])
        layers.lwater[l+1] = np.sum(layers.lwater[l:l+2])
        layers.ltemp[l+1] = np.mean(layers.ltemp[l:l+2])
        layers.lheight[l+1] = np.sum(layers.lheight[l:l+2])
        layers.ldrymass[l+1] = np.sum(layers.ldrymass[l:l+2])
        layers.lrefreeze[l+1] = np.sum(layers.lrefreeze[l:l+2])
        layers.lnewsnow[l+1] = np.sum(layers.lnewsnow[l:l+2])
        layers.grainsize[l+1] = np.mean(layers.grainsize[l:l+2])
        layers.lBC[l+1] = np.mean(layers.lBC[l:l+2])
        layers.ldust[l+1] = np.mean(layers.ldust[l:l+2])
        self.removeLayer(l)
        return layers
    
    def updateLayers(self,layers):
        """
        Checks the layer heights against the initial size scheme. If layers have become too small, they are
        merged with the layer below. If layers have become too large, they are split into two even layers.
        """
        layer = 0
        layer_split = False
        min_heights = lambda i: eb_prms.dz_toplayer * np.exp((i-1)*eb_prms.layer_growth) / 2
        max_heights = lambda i: eb_prms.dz_toplayer * np.exp((i-1)*eb_prms.layer_growth) * 2
        while layer < layers.nlayers:
            layer_split = False
            dz = layers.lheight[layer]
            if layers.ltype[layer] in ['snow','firn']:
                if dz < min_heights(layer) and layers.ltype[layer]==layers.ltype[layer+1]:
                    # Layer too small. Merge but only if it's the same type as the layer underneath
                    layers = self.mergeLayers(layers,layer)
                elif dz > max_heights(layer):
                    # Layer too big. Split into two equal size layers
                    layers = self.splitLayer(layers,layer)
                    layer_split = True
            if not layer_split:
                layer += 1
        return layers
    
    def updateLayerProperties(self,layers,do=['depth','density','irrwater']):
        """
        Recalculates nlayers, depths, density, and irreducible water content from DRY density. 
        Can specify to only update certain properties.

        Parameters
        ----------
        do : list-like
            List of any combination of depth, density and irrwater to be updated
        """
        layers.nlayers = len(layers.lheight)
        layers.snow_idx =  np.where(np.array(layers.ltype)=='snow')[0]
        layers.firn_idx =  np.where(layers.ltype=='firn')[0]
        layers.ice_idx =  np.where(layers.ltype=='ice')[0]
        if 'depth' in do:
            layers.ldepth = np.array([np.sum(layers.lheight[:i+1])-(layers.lheight[i]/2) for i in range(layers.nlayers)])
        if 'density' in do:
            layers.ldensity = layers.ldrymass / layers.lheight
        if 'irrwater' in do:
            layers.irrwatercont = layers.getIrrWaterCont(layers.ldensity)
        return layers
    
    def updateLayerTypes(self,layers):
        """
        Checks if new firn or ice layers have been created by densification.
        """
        layer = 0
        while layer < layers.nlayers:
            dens = layers.ldensity[layer]
            # New FIRN layer
            if dens >= eb_prms.density_firn and layers.ltype[layer] == 'snow':
                layers.ltype[layer] = 'firn'
                layers.ldensity[layer] = eb_prms.density_firn
                # Merge layers if there is firn under the new firn layer
                if layers.ltype[layer+1] in ['firn']: 
                    self.mergeLayers(layers,layer)
                    print('new firn!')
            # New ICE layer
            elif layers.ldensity[layer] >= eb_prms.density_ice and layers.ltype[layer] == 'firn':
                layers.ltype[layer] = 'ice'
                layers.ldensity[layer] = eb_prms.density_ice
                # Merge into ice below
                if layers.ltype[layer+1] in ['ice']:
                    self.mergeLayers(layers,layer)
                    print('new ice!')
            else:
                layer += 1
        return layers
    
    # ============= OTHER ============= 

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
        