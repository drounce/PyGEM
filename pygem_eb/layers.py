import numpy as np
import pandas as pd
import xarray as xr
import pygem_eb.input as eb_prms

class Layers():
    """
    Layer scheme for the 1D snowpack model.
    """
    def __init__(self,bin_no):
        """
        Initialize the temperature, density and water content profile of the vertical layers.

        Parameters
        ----------
        bin_no : int
            Bin number
        """
        # Extract datasets containing pairs of depth,value for temperature and density (elevation independent)
        Tpds = xr.open_dataset(eb_prms.init_filepath)
        depth_data = Tpds.coords['layer_depth'].values
        temp_data = [[depth_data[i],Tpds['snow_temp'].to_numpy()[i]] for i in range(len(depth_data))]
        density_data = [[depth_data[i],Tpds['snow_density'].to_numpy()[i]] for i in range(len(depth_data))]

        # Interpolate dataset for snow, firn and ice depth at bin elevation
        Tpds_interp = Tpds.interp(bin_elev=eb_prms.bin_elev[bin_no],kwargs={'fill_value':'extrapolate'})
        vars = ['snow_depth','firn_depth','ice_depth']
        sfi_h0 = np.array([Tpds_interp[var].values for var in vars])
        sfi_h0[0] = 0.41

        # Calculate the layer depths based on initial snow, firn and ice depths
        lheight,ldepth,ltype = self.getLayers(sfi_h0)
        self.nlayers = len(lheight)
        self.ltype = ltype
        self.lheight = lheight
        self.ldepth = ldepth

        # Initialize layer temperature and density based on chosen method and data 
        ltemp,ldensity,lwater = self.getTpw(sfi_h0,temp_data,density_data)

        # Define dry (solid) mass of each layer [kg m-2]
        ldrymass = ldensity*lheight
        self.ldrymass = ldrymass
        # Define irreducible water content of each layer and set saturated value
        irrwatercont = self.getIrrWaterCont(ldensity)
        # saturated = np.where(watercont == irrwatercont,1,0)

        # Initialize BC and dust content
        if eb_prms.switch_LAPs == 0:
            BC = [0,0]
            dust = [0,0]
        elif eb_prms.switch_LAPs == 2 and eb_prms.initLAPs is not None:
            BC = eb_prms.initLAPs[0,:]
            dust = eb_prms.initLAPs[1,:]
        else:
            BC = [eb_prms.BC_freshsnow,eb_prms.BC_freshsnow]
            dust = [eb_prms.dust_freshsnow,eb_prms.dust_freshsnow]
        
        self.ltemp = ltemp
        self.ldensity = ldensity
        self.lwater = lwater
        self.lheight = lheight
        self.ldepth = ldepth
        self.ltype = ltype
        self.ldrymass = ldrymass
        self.irrwatercont = irrwatercont
        self.BC = BC
        self.dust = dust
        print(self.nlayers,'layers initialized for bin',bin_no)
        return 
    
    def getLayers(self,sfi_h0):
        """
        Initializes layer depths based on an exponential growth function with prescribed rate of growth and 
        initial layer depth (from pygem_input). 

        Parameters
        ----------
        sfi_h0 : np.ndarray
            Initial thicknesses of the snow, firn and ice layers [m]

        Returns
        -------
        layerh : np.ndarray
            Height of the layer [m]
        layerz : np.ndarray
            Depth of the middle of the layer [m]
        layertype : np.ndarray
            Type of layer, 'snow' 'firn' or 'ice'
        """
        dz_toplayer = eb_prms.dz_toplayer
        layer_growth = eb_prms.layer_growth

        #Initialize variables to get looped
        layerh = []
        layertype = []

        snow_depth = 0
        layeridx = 0
        # Make exponentially growing snow layers
        while snow_depth < sfi_h0[0]:
            layerh.append(dz_toplayer * np.exp(layeridx*layer_growth))
            layertype.append('snow')
            layeridx += 1
            snow_depth = np.sum(layerh)
        layerh[-1] = layerh[-1] - (snow_depth-sfi_h0[0])
    
        # Add firn layers
        if sfi_h0[1] > 0.75:
            n_firn_layers = int(round(sfi_h0[1],0))
            layerh.extend([sfi_h0[1]/n_firn_layers]*n_firn_layers)
            layertype.extend(['firn']*n_firn_layers)
        elif sfi_h0[1] > 0:
            layerh.extend([sfi_h0[1]])
            layertype.extend(['firn'])

        # Add ice bin
        if eb_prms.icelayers == 1:
            layerh.append(sfi_h0[2])
            layertype.append('ice')
        else:
            ice_depth = 0
            while ice_depth < sfi_h0[2]:
                layerh.append(dz_toplayer * np.exp(layeridx*layer_growth))
                layertype.append('ice')
                layeridx += 1
                ice_idx = np.where(np.array(layertype)=='ice')[0]
                ice_depth += np.sum(np.array(layerh)[ice_idx])
            ice_idx = np.where(np.array(layertype)=='ice')[0]
            layerh[-1] = layerh[-1] - (ice_depth-sfi_h0[2])

        # Calculate layer depths (mid-points)
        layerz = [np.sum(layerh[:i+1])-(layerh[i]/2) for i in range(len(layerh))]
 
        return np.array(layerh), np.array(layerz), np.array(layertype)

    def getTpw(self,sfi_h0,temp_data,density_data,method=eb_prms.option_initTemp):
        snow_idx =  np.where(self.ltype=='snow')[0]
        firn_idx =  np.where(self.ltype=='firn')[0]
        ice_idx =  np.where(self.ltype=='ice')[0]
        temp_data = np.array(temp_data)
        density_data = np.array(density_data)

        # Initialize temperature profiles from piecewise formulation or interpolating data
        if eb_prms.option_initTemp in ['piecewise']:
            ltemp = self.initProfilesPiecewise(self.ldepth,temp_data,'temp')
        elif eb_prms.option_initTemp in ['interp']:
            ltemp = np.interp(self.ldepth,temp_data[0,:],temp_data[1,:])
        else:
            assert 1==0, "Choose between 'piecewise' and 'interp' methods for temp initialization"
        ltemp[ice_idx] = 0
        print('FIX ICE TEMPERATURE INITIALIZATION')

        # Initialize SNOW density profiles  from piecewise formulation or interpolating data
        if eb_prms.option_initDensity in ['piecewise']:
            ldensity = self.initProfilesPiecewise(self.ldepth[snow_idx],density_data,'density')
        elif eb_prms.option_initDensity in ['interp']:
            ldensity = np.interp(self.ldepth[snow_idx],density_data[0,:],density_data[1,:])
        else:
            assert 1==0, "Choose between 'piecewise' and 'interp' methods for density initialization"

        # Calculate firn density slope that linearly increases density from the bottom snow bin to the top of the ice layer
        if sfi_h0[0] > 0 and sfi_h0[1] > 0:
            pslope = (eb_prms.density_ice - ldensity[-1])/(self.ldepth[ice_idx[0]]-self.ldepth[snow_idx[-1]])
        elif sfi_h0[1] > 0:
            pslope = (eb_prms.density_ice - eb_prms.density_firn)/(sfi_h0[1])
        # Add firn and ice layer densities
        for idx,type in enumerate(self.ltype):
            if type in ['firn']:
                ldensity = np.append(ldensity,ldensity[snow_idx[-1]] + pslope*(self.ldepth[idx]-self.ldepth[snow_idx[-1]]))
            elif type in ['ice']:
                ldensity = np.append(ldensity,eb_prms.density_ice)

        # Initialize water content [kg m-2]
        if eb_prms.option_initWater in ['zero_w0']:
            lwater = np.zeros(self.nlayers)
        elif eb_prms.option_initWater in ['initial_w0']:
            assert 1==0, "Only zero water content method is set up"
        
        self.snow_idx = snow_idx
        self.firn_idx = firn_idx
        self.ice_idx = ice_idx
        return ltemp,ldensity,lwater
    
    def initProfilesPiecewise(self,layerz,snow_var,varname):
        """
        Based on the DEBAM scheme for temperature and density that assumes linear changes with depth 
        in three piecewise sections.

        Parameters
        ----------
        layerz : np.ndarray
            Middles depth of the layers to be filled.
        snow_var : np.ndarray
            Turning point snow temperatures or densities and the associated depths in pairs 
            by (depth,temp/density value). If a surface value (z=0) is not prescribed, temperature 
            is assumed to be 0C, or density to be 100 kg m-3.
        varname : str
            'temp' or 'density': which variable is being calculated
        """
        # Check if inputs are the correct dimensions
        assert np.shape(snow_var) in [(4,2),(3,2)], "! Snow inputs data is improperly formatted"

        # Check if a surface value is given; if not, add a row at z=0, T=0C or p=100kg/m3
        if np.shape(snow_var) == (3,2):
            assert snow_var[0,0] == 1.0, "! Snow inputs data is improperly formatted"
            if varname in ['temp']:
                np.insert(snow_var,0,[0,0],axis=0)
            elif varname in ['density']:
                np.insert(snow_var,0,[0,100],axis=0)

        #calculate slopes and intercepts for the piecewise function
        snow_var = np.array(snow_var)
        slopes = [(snow_var[i,1]-snow_var[i+1,1])/(snow_var[i,0]-snow_var[i+1,0]) for i in range(3)]
        intercepts = [snow_var[i+1,1] - slopes[i]*snow_var[i+1,0] for i in range(3)]

        #solve piecewise functions at each layer depth
        layer_var = np.piecewise(layerz,
                     [layerz <= snow_var[1,0], (layerz <= snow_var[2,0]) & (layerz > snow_var[1,0]),
                      (layerz > snow_var[2,0])],
                      [lambda x: slopes[0]*x+intercepts[0],lambda x:slopes[1]*x+intercepts[1],
                       lambda x: slopes[2]*x+intercepts[2]])
        return layer_var
    
    def addLayers(self,layers_to_add):
        """
        Adds layers to layers class.

        Parameters
        ----------
        layers_to_add : pd.Dataframe
            Contains temperature 'T', water content 'w', height 'h', type 't', dry mass 'drym'
        """
        self.nlayers += len(layers_to_add.loc['T'].values)
        self.ltemp = np.append(layers_to_add.loc['T'].values,self.ltemp)
        self.lwater = np.append(layers_to_add.loc['w'].values,self.lwater)
        self.lheight = np.append(layers_to_add.loc['h'].values,self.lheight)
        self.ltype = np.append(layers_to_add.loc['t'].values,self.ltype)
        self.ldrymass = np.append(layers_to_add.loc['drym'].values,self.ldrymass)
        self.updateLayerProperties()
        return
    
    def removeLayer(self,layer_to_remove):
        """
        Removes a single layer from layers class.

        Parameters
        ----------
        layer_to_remove : int
            index of layer to remove
        """
        self.nlayers -= 1
        self.ltemp = np.delete(self.ltemp,layer_to_remove)
        self.lwater = np.delete(self.lwater,layer_to_remove)
        self.lheight = np.delete(self.lheight,layer_to_remove)
        self.ltype = np.delete(self.ltype,layer_to_remove)
        self.ldrymass = np.delete(self.ldrymass,layer_to_remove)
        self.updateLayerProperties()
        return
    
    def splitLayer(self,layer_to_split):
        """
        Splits a single layer into two layers with half the height, mass, and water content.
        """
        if (self.nlayers+1) < eb_prms.max_nlayers:
            l = layer_to_split
            self.nlayers += 1
            self.ltemp = np.insert(self.ltemp,l,self.ltemp[l])
            self.ltype = np.insert(self.ltype,l,self.ltype[l])

            self.lwater[l] = self.lwater[l]/2
            self.lwater = np.insert(self.lwater,l,self.lwater[l])
            self.lheight[l] = self.lheight[l]/2
            self.lheight = np.insert(self.lheight,l,self.lheight[l])
            self.ldrymass[l] = self.ldrymass[l]/2
            self.ldrymass = np.insert(self.ldrymass,l,self.ldrymass[l])
        self.updateLayerProperties()
        return

    def mergeLayers(self,layer_to_merge):
        """
        Merges two layers, summing height, mass and water content and averaging other properties.
        Layers merged will be the index passed and the layer below it
        """
        l = layer_to_merge
        self.ldensity[l+1] = np.sum(self.ldensity[l:l+2]*self.ldrymass[l:l+2])/np.sum(self.ldrymass[l:l+2])
        self.lwater[l+1] = np.sum(self.lwater[l:l+2])
        self.ltemp[l+1] = np.mean(self.ltemp[l:l+2])
        self.lheight[l+1] = np.sum(self.lheight[l:l+2])
        self.ldrymass[l+1] = np.sum(self.ldrymass[l:l+2])
        self.removeLayer(l)
        return
    
    def updateLayers(self):
        """
        Checks the layer heights against the initial size scheme. If layers have become too small, they are
        merged with the layer below. If layers have become too large, they are split into two even layers.
        """
        layer = 0
        layer_split = False
        min_heights = lambda i: eb_prms.dz_toplayer * np.exp((i-1)*eb_prms.layer_growth) / 2
        max_heights = lambda i: eb_prms.dz_toplayer * np.exp((i-1)*eb_prms.layer_growth) * 2
        while layer < self.nlayers:
            layer_split = False
            dz = self.lheight[layer]
            if self.ltype[layer] in ['snow','firn']:
                if dz < min_heights(layer) and self.ltype[layer]==self.ltype[layer+1]:
                    # Layer too small. Merge but only if it's the same type as the layer underneath
                    self.mergeLayers(layer)
                elif dz > max_heights(layer):
                    # Layer too big. Split into two equal size layers
                    self.splitLayer(layer)
                    layer_split = True
            if not layer_split:
                layer += 1
        return
    
    def updateLayerProperties(self,do=['depth','density','irrwater']):
        """
        Recalculates nlayers, depths, density, and irreducible water content from DRY density. 
        Can specify to only update certain properties.

        Parameters
        ----------
        do : list-like
            List of any combination of depth, density and irrwater to be updated
        """
        self.nlayers = len(self.lheight)
        self.snow_idx =  np.where(np.array(self.ltype)=='snow')[0]
        self.firn_idx =  np.where(self.ltype=='firn')[0]
        self.ice_idx =  np.where(self.ltype=='ice')[0]
        if 'depth' in do:
            self.ldepth = np.array([np.sum(self.lheight[:i+1])-(self.lheight[i]/2) for i in range(self.nlayers)])
        if 'density' in do:
            self.ldensity = self.ldrymass / self.lheight
        if 'irrwater' in do:
            self.irrwatercont = self.getIrrWaterCont(self.ldensity)
        return
    
    def updateLayerTypes(self):
        """
        Checks if new firn or ice layers have been created by densification.
        """
        layer = 0
        while layer < self.nlayers:
            dens = self.ldensity[layer]
            # New FIRN layer
            if dens >= eb_prms.density_firn and self.ltype[layer] == 'snow':
                self.ltype[layer] = 'firn'
                self.ldensity[layer] = eb_prms.density_firn
                # Merge layers if there is firn under the new firn layer
                if self.ltype[layer+1] in ['firn']: 
                    self.mergeLayers(layer)
                    print('new firn!')
            # New ICE layer
            elif self.ldensity[layer] >= eb_prms.density_ice and self.ltype[layer] == 'firn':
                self.ltype[layer] = 'ice'
                self.ldensity[layer] = eb_prms.density_ice
                # Merge into ice below
                if self.ltype[layer+1] in ['ice']:
                    self.mergeLayers(layer)
                    print('new ice!')
            else:
                layer += 1
        return
    
    def addSnow(self,snowfall,airtemp,new_density=eb_prms.density_fresh_snow):
        """
        Adds snowfall to the layer scheme. If the existing top layer is ice, the fresh snow is a new layer,
        otherwise it is merged with the top layer.
        
        Parameters
        ----------
        snowfall : float
            Fresh snow MASS in kg / m2
        airtemp : float
            Air temperature in C
        new_density : float
            Density to use for the new snow
        """
        store_surface = False
        if self.ltype[0] in 'ice':
            new_layer = pd.DataFrame([airtemp,0,snowfall/new_density,'snow',snowfall],index=['T','w','h','t','drym'])
            self.addLayers(new_layer)
        elif self.ldensity[0] > new_density*3:
            # large density difference - create a new layer anyway
            new_layer = pd.DataFrame([airtemp,0,snowfall/new_density,'snow',snowfall],index=['T','w','h','t','drym'])
            self.addLayers(new_layer)
            store_surface = True
        else:
            # take weighted average of density and temperature of surface layer and new snow
            new_layermass = self.ldrymass[0] + snowfall
            self.ldensity[0] = (self.ldensity[0]*self.ldrymass[0] + new_density*snowfall)/(new_layermass)
            self.ltemp[0] = (self.ltemp[0]*self.ldrymass[0] + airtemp*snowfall)/(new_layermass)
            self.ldrymass[0] = new_layermass
            self.lheight[0] += snowfall/new_density
            if self.lheight[0] > (eb_prms.dz_toplayer * 1.5):
                self.splitLayer(0)
        self.updateLayerProperties()
        return store_surface
    
    def getIrrWaterCont(self,density=[0]):
        """
        Calculates the irreducible water content of the layers.
        """
        if sum(density) == 0: # if density is not specified, calculate from self
            density = self.ldrymass / self.lheight
        density = density.astype(float)
        density_ice = eb_prms.density_ice
        ice_idx = np.where(self.ltype=='ice')[0]

        porosity = (density_ice - density[:ice_idx[0]])/density_ice
        irrwatercont = 0.0143*np.exp(3.3*porosity)
        irrwatersat = irrwatercont*density[:ice_idx[0]]/porosity # kg m-3, mass of liquid over pore volume
        irrwatercont = irrwatersat*self.lheight[:ice_idx[0]] # kg m-2, mass of liquid in a layer
        
        irrwatercont = np.append(irrwatercont,np.zeros_like(ice_idx)) # ice layers cannot hold water
        return irrwatercont