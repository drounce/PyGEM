import numpy as np
import pandas as pd
import xarray as xr
import pygem_eb.input as eb_prms

class Layers():
    """
    Layer scheme for the 1D snowpack model.

    All mass terms are stored in kg m-2.
    """
    def __init__(self,bin_no,utils):
        """
        Initialize the layer properties (temperature, density, water content, LAPs, etc.)

        Parameters
        ----------
        bin_no : int
            Bin number
        utils
            class object from utils.py
        """
        # Add utils to layers class
        self.utils = utils 

        # Load in initial depths of snow, firn and ice
        snow_depth = eb_prms.initial_snowdepth[bin_no]
        firn_depth = eb_prms.initial_firndepth[bin_no]
        ice_depth = eb_prms.bin_ice_depth[bin_no]
        initial_sfi = np.array([snow_depth,firn_depth,ice_depth])

        # Calculate the layer depths based on initial snow, firn and ice depths
        lheight,ldepth,ltype,nlayers = self.makeLayers(initial_sfi)
        self.nlayers = nlayers              # NUMBER OF LAYERS
        self.ltype = ltype                  # LAYER TYPE (snow, firn, or ice)
        self.lheight = lheight              # LAYER HEIGHT (dz) [m]
        self.ldepth = ldepth                # LAYER DEPTH (midlayer) [m]

        # Initialize layer temperature, density, water content
        ltemp,ldensity,lwater = self.getTpw(initial_sfi)
        self.ltemp = ltemp                  # LAYER TEMPERATURE [C]
        self.ldensity = ldensity            # LAYER DENSITY [kg m-3]
        self.ldrymass = ldensity*lheight    # LAYER DRY (SOLID) MASS [kg m-2]
        self.lwater = lwater                # LAYER WATER CONTENT [kg m-2]

        # Initialize LAPs (black carbon and dust) - rows are "bins"
        self.lBC = np.ones((2,self.nlayers))*eb_prms.BC_freshsnow*lheight     # BC MASS [kg m-2]
        self.ldust = np.ones((5,self.nlayers))*eb_prms.dust_freshsnow*lheight # DUST MASS [kg m-2]

        # Grain size initial timestep can copy density
        self.grainsize = self.ldensity.copy()   # LAYER GRAIN SIZE [um]
        self.grainsize[np.where(self.grainsize<300)[0]] = 300
        self.grainsize[np.where(self.ltype == 'firn')[0]] = 1000
        self.grainsize[np.where(self.ltype == 'ice')[0]] = 1800

        # Initialize RF model for grain size lookups
        if eb_prms.method_grainsizetable in ['ML']:
            self.tau_rf,_,_ = utils.getGrainSizeModel(initSSA=eb_prms.initSSA,var='taumat')
            self.kap_rf,_,_ = utils.getGrainSizeModel(initSSA=eb_prms.initSSA,var='kapmat')
            self.dr0_rf,_,_ = utils.getGrainSizeModel(initSSA=eb_prms.initSSA,var='dr0mat')
        
        # Additional layer properties
        self.updateLayerProperties()
        self.lrefreeze = np.zeros_like(self.ltemp)   # LAYER MASS OF REFREEZE [kg m-2]
        self.lnewsnow = np.zeros_like(self.ltemp)    # LAYER MASS OF NEW SNOW [kg m-2]
        self.delayed_snow = 0 # Initiate bucket for 'delayed snow' (to avoid making tiny layers)

        print(f'{self.nlayers} layers initialized for bin {bin_no}')
        return 
    
    def makeLayers(self,initial_sfi):
        """
        Initializes layer depths based on an exponential growth function with prescribed rate 
        of growth and initial layer heights (from pygem_input). 

        Parameters
        ----------
        initial_sfi : np.ndarray
            Initial thicknesses of the snow, firn and ice layers [m]

        Returns
        -------
        lheight : np.ndarray
            Height of the layer [m]
        ldepth : np.ndarray
            Depth of the middle of the layer [m]
        ltype : np.ndarray
            Type of layer, 'snow' 'firn' or 'ice'
        """
        dz_toplayer = eb_prms.dz_toplayer
        layer_growth = eb_prms.layer_growth
        snow_height = initial_sfi[0]
        firn_height = initial_sfi[1]
        ice_height = initial_sfi[2]

        #Initialize variables to get looped
        lheight = []
        ltype = []

        current_depth = 0
        layer = 0
        # Make exponentially growing snow layers
        while current_depth < snow_height:
            lheight.append(dz_toplayer * np.exp(layer*layer_growth))
            ltype.append('snow')
            layer += 1
            current_depth = np.sum(lheight)
        lheight[-1] = lheight[-1] - (current_depth-snow_height)
    
        # Add firn layers
        if firn_height > 0.75:
            n_firn_layers = int(round(firn_height,0))
            lheight.extend([firn_height/n_firn_layers]*n_firn_layers)
            ltype.extend(['firn']*n_firn_layers)
        elif firn_height > 0:
            lheight.extend([firn_height])
            ltype.extend(['firn'])

        # Add ice layers
        if eb_prms.icelayers == 'single':
            lheight.append(ice_height)
            ltype.append('ice')
        else:
            current_depth = 0
            # Make exponentially growing ice layers
            while current_depth < ice_height:
                lheight.append(dz_toplayer * np.exp(layer*layer_growth))
                ltype.append('ice')
                layer += 1
                ice_idx = np.where(np.array(ltype)=='ice')[0]
                current_depth = np.sum(np.array(lheight)[ice_idx])
            lheight[-1] = lheight[-1] - (current_depth-ice_height)
        
        # Get depth of layers (distance from surface to midpoint of layer) [m]
        nlayers = len(lheight)
        ldepth = [np.sum(lheight[:i+1])-(lheight[i]/2) for i in range(nlayers)]
        return np.array(lheight), np.array(ldepth), np.array(ltype), nlayers

    def getTpw(self,initial_sfi):
        """
        Initializes the layer temperatures, densities and water content.

        Parameters:
        -----------
        initial_sfi : np.ndarray
            Array containing depth of snow, firn and ice
        Returns:
        --------
        layertemp, layerdensity, layerwater : np.ndarray
            Arrays containing layer temperature, density and water content.
                                        [C]      [kg m-3]       [kg m-2]
        """
        snow_idx =  np.where(self.ltype=='snow')[0]
        firn_idx =  np.where(self.ltype=='firn')[0]
        ice_idx =  np.where(self.ltype=='ice')[0]

        # Read in temp and density data from csv
        temp_data = pd.read_csv(eb_prms.initial_temp_fp)[['depth','temp']].to_numpy()
        density_data = pd.read_csv(eb_prms.initial_density_fp)[['depth','density']].to_numpy()

        # Initialize temperature profiles from piecewise formulation or interpolating data
        if eb_prms.initialize_temp in ['piecewise']:
            ltemp = self.initProfilesPiecewise(self.ldepth,temp_data,'temp')
        elif eb_prms.initialize_temp in ['interp']:
            ltemp = np.interp(self.ldepth,temp_data[:,0],temp_data[:,1])
        else:
            assert 1==0, "Choose between 'piecewise' and 'interp' methods for temp initialization"

        # Initialize SNOW density profiles from piecewise formulation or interpolating data
        if eb_prms.initialize_dens in ['piecewise']:
            ldensity = self.initProfilesPiecewise(self.ldepth[snow_idx],density_data,'density')
        elif eb_prms.initialize_dens in ['interp']:
            ldensity = np.interp(self.ldepth[snow_idx],density_data[:,0],density_data[:,1])
        else:
            assert 1==0, "Choose between 'piecewise' and 'interp' methods for density initialization"

        # Calculate firn density slope that linearly increases density from the bottom snow layer to top ice layer
        if initial_sfi[0] > 0 and initial_sfi[1] > 0:
            pslope = (eb_prms.density_ice - ldensity[-1])/(self.ldepth[ice_idx[0]]-self.ldepth[snow_idx[-1]])
        elif initial_sfi[1] > 0:
            pslope = (eb_prms.density_ice - eb_prms.density_firn)/(initial_sfi[1])
        # Add firn and ice layer densities
        for idx,type in enumerate(self.ltype):
            if type in ['firn']:
                ldensity = np.append(ldensity,ldensity[snow_idx[-1]] + pslope*(self.ldepth[idx]-self.ldepth[snow_idx[-1]]))
            elif type in ['ice']:
                ldensity = np.append(ldensity,eb_prms.density_ice)

        # Initialize water content [kg m-2]
        assert eb_prms.initialize_water in ['zero_w0'], "Only zero water content method is set up"
        lwater = np.zeros(self.nlayers)

        return ltemp,ldensity,lwater
    
    def initProfilesPiecewise(self,ldepth,snow_var,varname):
        """
        Based on the DEBAM scheme for temperature and density that assumes linear changes with depth 
        in three piecewise sections.

        Parameters
        ----------
        ldepth : np.ndarray
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

        # Check if a surface value is given; if not, add a row at z=0 m, T=0 C / p=100 kg m-3
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
        layer_var = np.piecewise(ldepth,
                     [ldepth <= snow_var[1,0], (ldepth <= snow_var[2,0]) & (ldepth > snow_var[1,0]),
                      (ldepth > snow_var[2,0])],
                      [lambda x: slopes[0]*x+intercepts[0],lambda x:slopes[1]*x+intercepts[1],
                       lambda x: slopes[2]*x+intercepts[2]])
        return layer_var
    
    # ========= UTILITY FUNCTIONS ==========

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
        self.ldrymass = np.append(layers_to_add.loc['m'].values,self.ldrymass)
        self.lrefreeze = np.append(0,self.lrefreeze)
        # Only way to add a layer is with new snow, so layer new snow = layer dry mass
        self.lnewsnow = np.append(layers_to_add.loc['new'].values,self.lnewsnow)
        self.grainsize = np.append(layers_to_add.loc['g'].values,self.grainsize)
        new_layer_BC = np.ones((2,1))*layers_to_add.loc['BC'].values*self.lheight[0]
        self.lBC = np.append(new_layer_BC,self.lBC,axis=1)
        new_layer_dust = np.ones((5,1))*layers_to_add.loc['dust'].values*self.lheight[0]
        self.ldust = np.append(new_layer_dust,self.ldust,axis=1)
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
        self.lrefreeze = np.delete(self.lrefreeze,layer_to_remove)
        self.lnewsnow = np.delete(self.lnewsnow,layer_to_remove)
        self.grainsize = np.delete(self.grainsize,layer_to_remove)
        self.lBC = np.delete(self.lBC,layer_to_remove,axis=1)
        self.ldust = np.delete(self.ldust,layer_to_remove,axis=1)
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
            self.grainsize = np.insert(self.grainsize,l,self.grainsize[l])

            self.lwater[l] = self.lwater[l]/2
            self.lwater = np.insert(self.lwater,l,self.lwater[l])
            self.lheight[l] = self.lheight[l]/2
            self.lheight = np.insert(self.lheight,l,self.lheight[l])
            self.ldrymass[l] = self.ldrymass[l]/2
            self.ldrymass = np.insert(self.ldrymass,l,self.ldrymass[l])
            self.lrefreeze[l] = self.lrefreeze[l]/2
            self.lrefreeze = np.insert(self.lrefreeze,l,self.lrefreeze[l])
            self.lnewsnow[l] = self.lnewsnow[l]/2
            self.lnewsnow = np.insert(self.lnewsnow,l,self.lnewsnow[l])
            self.lBC[:,l] = self.lBC[:,l]/2
            self.lBC = np.insert(self.lBC,l,self.lBC[:,l],axis=1)
            self.ldust[:,l] = self.ldust[:,l]/2
            self.ldust = np.insert(self.ldust,l,self.ldust[:,l],axis=1)
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
        self.lrefreeze[l+1] = np.sum(self.lrefreeze[l:l+2])
        self.lnewsnow[l+1] = np.sum(self.lnewsnow[l:l+2])
        self.grainsize[l+1] = np.mean(self.grainsize[l:l+2])
        self.lBC[:,l+1] = np.sum(self.lBC[:,l:l+2],axis=1)
        self.ldust[:,l+1] = np.sum(self.ldust[:,l:l+2],axis=1)
        self.removeLayer(l)
        return
    
    def updateLayers(self):
        """
        Checks the layer heights against the initial size scheme. If layers have become too small, 
        they are merged with the layer below. If layers have become too large, they are split into 
        two identical layers of half the size.
        """
        layer = 0
        min_heights = lambda i: eb_prms.dz_toplayer * np.exp((i-1)*eb_prms.layer_growth) / 2
        max_heights = lambda i: eb_prms.dz_toplayer * np.exp((i-1)*eb_prms.layer_growth) * 2
        while layer < self.nlayers:
            layer_split = False
            dz = self.lheight[layer]
            if self.ltype[layer] in ['snow','firn']:
                if dz < min_heights(layer) and self.ltype[layer]==self.ltype[layer+1]:
                    # Layer too small. Merge if it is the same type as the layer underneath
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
        Recalculates nlayers, depths, density, and irreducible water content from density. 
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
    
    def addSnow(self,snowfall,enbal,surface,args,timestamp):
        """
        Adds snowfall to the layer scheme. If the existing top layer is ice, the fresh snow is a new layer,
        otherwise it is merged with the top layer.
        
        Parameters
        ----------
        snowfall : float
            Fresh snow MASS in kg / m2
        enbal
            class object from pygem_eb.energybalance
        """
        store_surface = False
        snowfall += self.delayed_snow

        if args.switch_snow == 0:
            # Snow falls with the same properties as the current top layer
            # **** What to do when top layer is ice?
            new_density = self.ldensity[0]
            new_height = snowfall/new_density
            new_grainsize = self.grainsize[0]
            new_BC = self.lBC[:,0]/self.lheight[0]*new_height
            new_dust = self.ldust[:,0]/self.lheight[0]*new_height
            new_snow = 0
        elif args.switch_snow == 1:
            if eb_prms.constant_snowfall_density:
                new_density = eb_prms.density_fresh_snow
            else:
                new_density = max(109*6*(enbal.tempC-0.)+26*enbal.wind**0.5,50) # from CROCUS ***** CITE
            new_grainsize = eb_prms.fresh_grainsize
            new_height = snowfall/new_density
            new_BC = eb_prms.BC_freshsnow*new_height
            new_dust = eb_prms.dust_freshsnow*new_height
            new_snow = snowfall
            surface.snow_timestamp = timestamp

        # Conditions: if any are TRUE, create a new layer
        new_layer_conds = np.array([self.ltype[0] in 'ice',
                            self.ltype[0] in 'firn',
                            self.ldensity[0] > new_density*3])
        if np.any(new_layer_conds):
            if snowfall/new_density > 1e-4:
                new_layer = pd.DataFrame([enbal.tempC,0,snowfall/new_density,'snow',snowfall,
                                      new_grainsize,new_BC,new_dust,new_snow],
                                     index=['T','w','h','t','m','g','BC','dust','new'])
                self.addLayers(new_layer)
                self.delayed_snow = 0
            else:
                self.delayed_snow = snowfall
            store_surface = True
        else:
            # take weighted average of density and temperature of surface layer and new snow
            new_layermass = self.ldrymass[0] + snowfall
            self.lnewsnow[0] = snowfall if eb_prms.switch_snow == 1 else 0
            self.ldensity[0] = (self.ldensity[0]*self.ldrymass[0] + new_density*snowfall)/(new_layermass)
            self.ltemp[0] = (self.ltemp[0]*self.ldrymass[0] + enbal.tempC*snowfall)/(new_layermass)
            self.grainsize[0] = (self.grainsize[0]*self.ldrymass[0] + new_grainsize*snowfall)/(new_layermass)
            self.ldrymass[0] = new_layermass
            self.lheight[0] += snowfall/new_density
            self.lBC[:,0] = self.lBC[:,0] + new_BC
            self.ldust[:,0] = self.ldust[:,0] + new_dust
            if self.lheight[0] > (eb_prms.dz_toplayer * 1.5):
                self.splitLayer(0)
        self.updateLayerProperties()
        return store_surface

    def getGrainSize(self,surftemp,bins=None,dt=eb_prms.daily_dt):
        """
        Snow grain size metamorphism
        """
        # CONSTANTS
        WET_C = eb_prms.wet_snow_C
        FRESH_GRAINSIZE = eb_prms.fresh_grainsize

        if len(self.snow_idx) > 0:
            # bins should be a list of indices to calculate grain size on
            if not bins:
                bins = self.snow_idx
            n = len(bins)
            
            # Get fractions of refreeze, new snow and old snow
            refreeze = self.lrefreeze[bins]
            new_snow = self.lnewsnow[bins]
            old_snow = self.ldrymass[bins] - refreeze - new_snow
            f_old = old_snow / self.ldrymass[bins]
            f_new = new_snow / self.ldrymass[bins]
            f_rfz = refreeze / self.ldrymass[bins]
            f_liq = self.lwater[bins] / (self.lwater[bins] + self.ldrymass[bins])

            dz = self.lheight.copy()[bins]
            T = self.ltemp.copy()[bins] + 273.15
            surftempK = surftemp + 273.15
            p = self.ldensity.copy()[bins]
            g = self.grainsize.copy()[bins]

            # Dry metamorphism
            # Calculate temperature gradient
            dTdz = np.zeros_like(T)
            if len(bins) > 2:
                dTdz[0] = (surftempK - (T[0]*dz[0]+T[1]*dz[1]) / (dz[0]+dz[1]))/dz[0]
                dTdz[1:-1] = ((T[:-2]*dz[:-2] + T[1:-1]*dz[1:-1]) / (dz[:-2] + dz[1:-1]) -
                        (T[1:-1]*dz[1:-1] + T[2:]*dz[2:]) / (dz[1:-1] + dz[2:])) / dz[1:-1]
                dTdz[-1] = dTdz[-2] # Bottom temp gradient -- not used
            elif len(bins) == 2: # Use top ice layer for temp gradient
                T_2layer = np.array([surftempK,T[0],T[1],self.ltemp[2]+273.15])
                depth_2layer = np.array([0,self.ldepth[0],self.ldepth[1],self.ldepth[2]])
                dTdz = (T_2layer[0:2] - T_2layer[2:]) / (depth_2layer[0:2] - depth_2layer[2:])
            else: # Single bin
                dTdz = (self.ltemp[2]+273.15-surftempK) / self.ldepth[2]
                dTdz = np.array([dTdz])
            dTdz = np.abs(dTdz[bins])

            # Force to be within lookup table ranges******
            # p[np.where(p > 400)[0]] = 400
            dTdz[np.where(dTdz > 300)[0]] = 300

            if eb_prms.method_grainsizetable in ['interpolate']:
                # Interpolate lookup table at the values of T,dTdz,p
                ds = xr.open_dataset(eb_prms.grainsize_fp)
                ds = ds.interp(TVals=T[bins],DTDZVals=dTdz,DENSVals=p)
                # Extract values
                diag = np.zeros((n,n,n),dtype=bool)
                for i in range(n):
                    diag[i,i,i] = True
                tau = ds.taumat.to_numpy()[diag]
                kap = ds.kapmat.to_numpy()[diag]
                dr0 = ds.dr0mat.to_numpy()[diag]

            elif eb_prms.method_grainsizetable in ['ML']:
                X = np.vstack([T,p,dTdz]).T
                tau = np.exp(self.tau_rf.predict(X))
                kap = np.exp(self.kap_rf.predict(X))
                dr0 = self.dr0_rf.predict(X)

            if min(g)<eb_prms.fresh_grainsize:
                drdrydt = dr0*1e-6*np.power(tau/(tau + 1.0),1/kap)
            else:
                drdrydt = dr0*1e-6*np.power(tau/(tau + 1e6*(g - FRESH_GRAINSIZE)),1/kap)
            drdry = drdrydt * dt * 10**6 # convert m to um

            # Wet metamorphism
            drwetdt = WET_C*f_liq**3/(4*np.pi*g**2)
            drwet = drwetdt * dt * 10**6 # convert m to um

            # Get old grain size
            aged_grainsize = g+drdry+drwet
            aged_grainsize[np.where(aged_grainsize > 1500)[0]] = 1500 
            #**** THIS STATEMENT IS BECAUSE SEOMTIMES YOU GET CRAZY HIGH GRAIN SIZE WITH BIG TEMP GRADIENTS

            # Sum contributions of old snow, new snow and refreeze
            grainsize = aged_grainsize*f_old + 54.5*f_new + 1500*f_rfz
            self.grainsize[bins] = grainsize
            self.grainsize[self.firn_idx] = 2000
            self.grainsize[self.ice_idx] = 5000
        elif len(self.firn_idx) > 0: # no snow, but there is firn
            self.grainsize[self.firn_idx] = 2000 # **** FIRN GRAIN SIZE
            self.grainsize[self.ice_idx] = 5000
        else: # no snow or firn, just ice
            self.grainsize[self.ice_idx] = 5000
        return 
    
    def getIrrWaterCont(self,density=None):
        """
        Calculates the irreducible water content of the layers.
        """
        if not np.all(density):
            density = self.ldrymass / self.lheight
        density = density.astype(float)
        DENSITY_ICE = eb_prms.density_ice
        ice_idx = self.ice_idx

        porosity = (DENSITY_ICE - density[:ice_idx[0]])/DENSITY_ICE
        irrwaterfrac = 0.0143*np.exp(3.3*porosity)
        irrwatersat = irrwaterfrac*density[:ice_idx[0]]/porosity # kg m-3, mass of liquid over pore volume
        irrwatercont = irrwatersat*self.lheight[:ice_idx[0]] # kg m-2, mass of liquid in a layer
        
        irrwatercont = np.append(irrwatercont,np.zeros_like(ice_idx)) # ice layers cannot hold water
        return irrwatercont