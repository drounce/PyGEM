import numpy as np
import pandas as pd
import xarray as xr
import pygem_eb.input as eb_prms
import pygem_eb.energybalance as eb
import pygem_eb.layers as eb_layers
import pygem_eb.surface as eb_surface

class massBalance():
    """
    Mass balance scheme which calculates layer and bin mass balance from melt, 
    refreeze and accumulation. main() executes the core of the model.
    """
    def __init__(self,bin_idx,dates_table,args,utils):
        """
        Initializes the layers and surface classes and model time for the mass balance scheme.

        Parameters
        ----------
        bin_idx : int
            Index value of the elevation bin
        dates_table : pd.Dataframe
            Dataframe containing the dates for the model run
        """
        # Set up model time and bin
        self.dt = eb_prms.dt
        self.days_since_snowfall = 0
        self.time_list = dates_table['date']
        self.bin_idx = bin_idx

        # Initialize layers and surface classes
        self.args = args
        self.layers = eb_layers.Layers(bin_idx,utils)
        self.surface = eb_surface.Surface(self.layers,self.time_list,args)

        # Initialize output class
        self.output = Output(self.time_list,bin_idx,args)
        return
    
    def main(self,climateds):
        """
        Runs the time loop and mass balance scheme to solve for melt, refreeze, accumulation and runoff.

        Parameters
        ----------
        climateds : xr.Dataset
            Dataset containing elevation-adjusted climate data
        """
        # Get classes and time
        layers = self.layers
        surface = self.surface
        dt = self.dt

        # ===== ENTER TIME LOOP =====
        for time in self.time_list:
            # BEGIN MASS BALANCE
            self.time = time

            # Initiate the energy balance to unpack climate data
            enbal = eb.energyBalance(climateds,time,self.bin_idx,dt)

            # Get rain and snowfall amounts [kg m-2]
            rain,snowfall = self.getPrecip(enbal,surface,time)

            # Add fresh snow to layers
            if snowfall > 0:
                store_surface = layers.addSnow(snowfall,enbal)
                if store_surface: # ****Store surface might not be necessary
                    surface.storeSurface()
            # Add dry deposited BC and dust to layers
            enbal.getDryDeposition(layers)

            # Update daily properties
            if time.hour + time.minute < 1: 
                surface.updateSurfaceDaily(layers,time)
                self.days_since_snowfall = surface.days_since_snowfall
                layers.lnewsnow = np.zeros(layers.nlayers)
                surface.getAlbedo(layers)

            # Calculate surface energy balance by updating surface temperature
            surface.getSurfTemp(enbal,layers)

            # Calculate subsurface heating from penetrating SW
            if layers.nlayers > 1: 
                SWin,SWout = enbal.getSW(surface.albedo)
                subsurf_melt = self.heatSubsurface(layers,SWin+SWout)
            else: # If there is only one layer, no subsurface melt occurs
                subsurf_melt = [0]

            # Calculate column melt including the surface
            if surface.Qm > 0:
                layermelt, fully_melted_mass = self.meltSubsurface(layers,subsurf_melt)
            else: # No surface melt
                layermelt = subsurf_melt.copy()
                layermelt[0] = 0
                fully_melted_mass = 0

            # Percolate the meltwater and any liquid precipitation if there are snow or firn layers
            water_in = rain + fully_melted_mass
            if len(np.concatenate([layers.snow_idx,layers.firn_idx])) > 0:
                runoff = self.percolation(layers,layermelt,water_in)
            else:
                runoff = water_in + np.sum(layermelt)
            
            # Update layers (checks for tiny or huge layers)
            layers.updateLayers()

            # Recalculate the temperature profile considering conduction, unless glacier is isothermal
            if np.abs(np.sum(layers.ltemp)) != 0.:
                layers.ltemp = self.solveHeatEq(layers,surface.stemp,eb_prms.dt_heateq)

            # Calculate refreeze
            refreeze = self.refreezing(layers)

            # Run densification daily
            if time.hour + time.minute < 1:
                self.densification(layers,eb_prms.daily_dt)

            # END MASS BALANCE
            self.runoff = runoff
            self.melt = np.sum(layermelt) / eb_prms.density_water
            self.refreeze = refreeze
            self.accum = snowfall / eb_prms.density_water

            # Store timestep data
            self.output.storeTimestep(self,enbal,surface,layers,time)   

            # Debugging: print current state and monthly melt at the end of each month
            if time.is_month_start and time.hour + time.minute == 0 and eb_prms.debug:
                self.current_state(time,enbal.tempC)

            # Advance timestep
            pass

        # Completed bin: store data
        if self.args.store_data:
            self.output.storeData(self.bin_idx)
        return

    def heatSubsurface(self,layers,surface_SW):
        """
        Calculates melt in subsurface layers (excluding layer 0) due to penetrating shortwave radiation.

        Parameters
        ----------
        surface_SW : float
            Incoming SW radiation [W m-2]

        Returns
        -------
        layermelt : np.ndarray
            Array containing subsurface melt amounts [kg m-2]
        """
        # CONSTANTS
        HEAT_CAPACITY_ICE = eb_prms.Cp_ice
        LH_RF = eb_prms.Lh_rf

        # LAYERS IN
        lt = layers.ltype.copy()
        ld = layers.ldepth.copy()
        lT = layers.ltemp.copy()
        ldm = layers.ldrymass.copy()

        # Fraction of radiation absorbed at the surface depends on surface type
        FRAC_ABSRAD = 0.9 if lt[0] in ['snow'] else 0.8

        # Extinction coefficient depends on layer type
        EXTINCT_COEF = np.ones(layers.nlayers)*1e8 # ensures unfilled layers have 0 heat
        for layer,type in enumerate(lt):
            EXTINCT_COEF[layer] = 17.1 if type in ['snow'] else 2.5
            if np.exp(-EXTINCT_COEF[layer]*ld[layer]) < 1e-6:
                break # Cut off when flux reaches ~0
        pen_SW = surface_SW*(1-FRAC_ABSRAD)*np.exp(-EXTINCT_COEF*ld)/self.dt

        # recalculate layer temperatures, leaving out the surface (calculated separately)
        lT[1:] = lT[1:] + pen_SW[1:]/(ldm[1:]*HEAT_CAPACITY_ICE)*self.dt

        # calculate melt from temperatures above 0
        layermelt = np.zeros(layers.nlayers)
        for layer,temp in enumerate(lT):
            # check if temperature is above 0
            if temp > 0.:
                # calculate melt from the energy that raised the temperature above 0
                melt = (temp-0.)*ldm[layer]*HEAT_CAPACITY_ICE/LH_RF
                lT[layer] = 0.
            else:
                melt = 0
                lT[layer] = temp
            layermelt[layer] = melt

        # LAYERS OUT
        layers.ltemp = lT
        return layermelt

    def meltSubsurface(self,layers,subsurf_melt):
        """
        For cases when bins are melting. Can melt multiple surface bins at once if Qm is
        sufficiently high. Otherwise, adds the surface layer melt to the array containing
        subsurface melt to return the total layer melt. (This function does NOT REMOVE MASS
        from layers. That is done in percolation.)

        Parameters
        ----------
        layers
            class object from pygem_eb.layers
        surface
            class object from pygem_eb.surface
        subsurf_melt : np.ndarray
            Array containing melt of subsurface layers [kg m-2]
        Returns
        -------
        layermelt : np.ndarray
            Array containing layer melt amounts [kg m-2]
        
        """
        # CONSTANTS
        LH_RF = eb_prms.Lh_rf

        # LAYERS IN
        ldm = layers.ldrymass.copy()
        layermelt = subsurf_melt.copy()         # mass of melt due to penetrating SW [kg m-2]
        surface_melt = self.surface.Qm*self.dt/LH_RF       # mass of melt due to SEB [kg m-2]

        if surface_melt > ldm[0]:
            # melt by surface energy balance completely melts surface layer, so check if it melts further layers
            fully_melted = np.where(np.cumsum(ldm) <= surface_melt)[0]

            # calculate how much additional melt will occur in the first layer that's not fully melted
            newsurface_melt = surface_melt - np.sum(ldm[fully_melted])
            newsurface_idx = fully_melted[-1] + 1
            # it's possible to fully melt that layer too when combined with penetrating SW melt:
            if newsurface_melt + layermelt[newsurface_idx] > ldm[newsurface_idx]:
                fully_melted = np.append(fully_melted,newsurface_idx)
                # push new surface to the next layer down
                newsurface_melt -= ldm[newsurface_idx]
                newsurface_idx += 1

            # set melt amounts from surface melt into melt array
            layermelt[fully_melted] = ldm[fully_melted] 
            layermelt[newsurface_idx] += newsurface_melt 
        else:
            # only surface layer is melting
            layermelt[0] = surface_melt
            fully_melted = []
        fully_melted_mass = np.sum(layermelt[fully_melted])

        # Remove layers that were completely melted 
        removed = 0 # Indexes of layers change as you loop
        for layer in fully_melted:
            layers.removeLayer(layer-removed)
            layermelt = np.delete(layermelt,layer-removed)
            removed += 1 
        return layermelt, fully_melted_mass
        
    def percolation(self,layers,layermelt,water_in=0):
        """
        Calculates the liquid water content in each layer by downward percolation and applies melt.

        Parameters
        ----------
        layers
            class object from pygem_eb.layers
        layermelt: np.ndarray
            Array containing melt amount for each layer
        water_in : float
            Additional liquid water input (rainfall and fully melted layers) [kg m-2]

        Returns
        -------
        runoff : float
            Runoff that was not absorbed into void space [m w.e.]
        melted_layers : list
            List of layer indices that were fully melted
        """
        # CONSTANTS
        DENSITY_WATER = eb_prms.density_water
        DENSITY_ICE = eb_prms.density_ice
        FRAC_IRREDUC = eb_prms.Sr
        dt = self.dt
        
        # LAYERS IN
        snow_firn_idx = np.concatenate([layers.snow_idx,layers.firn_idx])
        ldm = layers.ldrymass.copy()[snow_firn_idx]
        lw = layers.lwater.copy()[snow_firn_idx]
        lh = layers.lheight.copy()[snow_firn_idx]
        layermelt_sf = layermelt[snow_firn_idx]

        if eb_prms.method_percolation in ['no_LAPs']:
            for layer,melt in enumerate(layermelt_sf):
                # remove melt from the dry mass
                ldm[layer] -= melt
                lh[layer] -= melt / layers.ldensity[layer]

                # add melt and extra_water (melt from above) to layer water content
                added_water = melt + extra_water
                lw[layer] += added_water

                # check if meltwater exceeds the irreducible water content of the layer
                layers.updateLayerProperties('irrwater')
                if lw[layer] >= layers.irrwatercont[layer]:
                    # set water content to irr. water content and add the difference to extra_water
                    extra_water = lw[layer] - layers.irrwatercont[layer]
                    lw[layer] = layers.irrwatercont[layer]
                else: #if not overflowing, extra_water should be set back to 0
                    extra_water = 0
                
                # get the change in layer height due to loss of solid mass (volume only considers solid)
                lh[layer] -= melt/layers.ldensity[layer]
                # need to update layer depths
                layers.updateLayerProperties() 
            # extra water goes to runoff
            runoff = extra_water / DENSITY_WATER
            # LAYERS OUT
            layers.lheight[snow_firn_idx] = lh
            layers.lwater[snow_firn_idx] = lw
            layers.ldrymass[snow_firn_idx] = ldm
            return runoff
        
        elif eb_prms.method_percolation in ['w_LAPs']:
            # Calculate volumetric fractions
            theta_liq = lw / (lh*DENSITY_WATER)
            theta_ice = ldm / (lh*DENSITY_ICE)
            porosity = 1 - theta_ice

            # initialize flow into the top layer
            qi = water_in / dt
            q_in_store = []
            q_out_store = []
            for layer,melt in enumerate(layermelt_sf):
                # set flow in equal to flow out of the previous layer
                q_in = qi

                # calculate flow out of layer i (cannot be negative)
                flow_out = DENSITY_WATER*lh[layer]/dt * (theta_liq[layer]-FRAC_IRREDUC*porosity[layer])
                qi = max(0,flow_out)

                # check limit of qi based on underlying layer holding capacity
                if layer < len(porosity) - 1 and theta_liq[layer] <= 0.3:
                    if porosity[layer] < 0.05 or porosity[layer+1] < 0.05:
                        # no flow if the layer itself or the lower layer has no pore space
                        lim = 0
                    else:
                        lim = DENSITY_WATER*lh[layer+1]/dt * (1-theta_ice[layer+1]-theta_liq[layer+1])
                        lim = max(0,lim)
                else: # no limit on bottom layer (1e6 sufficiently high)
                    lim = 1e6
                q_out = min(qi,lim)

                # layer mass balance
                lw[layer] += (q_in - q_out)*dt + melt
                ldm[layer] -= melt
                lh[layer] -= melt / layers.ldensity[layer]
                q_in_store.append(q_in)
                q_out_store.append(q_out)

            # LAYERS OUT
            layers.lheight[snow_firn_idx] = lh
            layers.lwater[snow_firn_idx] = lw
            layers.ldrymass[snow_firn_idx] = ldm
            runoff = q_out*dt + np.sum(layermelt[layers.ice_idx])

            # Diffuse LAPs 
            if self.args.switch_LAPs == 1:
                self.diffuseLAPs(layers,np.array(q_out_store),water_in/dt)
        return runoff
        
    def diffuseLAPs(self,layers,q_out,rainfall):
        # CONSTANTS
        PARTITION_COEF_BC = eb_prms.ksp_BC
        PARTITION_COEF_DUST = eb_prms.ksp_dust
        RAIN_CONC_BC = eb_prms.rainBC
        RAIN_CONC_DUST = eb_prms.raindust
        dt = eb_prms.dt

        # LAYERS IN
        snow_firn_idx = np.concatenate([layers.snow_idx,layers.firn_idx])
        lw = layers.lwater[snow_firn_idx]
        ldm = layers.ldrymass[snow_firn_idx]

        # mBC/mdust is layer mass of species in kg m-2
        mBC = layers.lBC[snow_firn_idx]
        mdust = layers.ldust[snow_firn_idx]
        # cBC/cdust is layer mass mixing ratio in kg kg-1
        cBC = mBC / (lw + ldm)
        cdust = mdust / (lw + ldm)

        # inward fluxes depend on previous layer or rainfall for top layer
        m_BC_in = PARTITION_COEF_BC*q_out[:-1]*cBC[:-1]
        m_dust_in = PARTITION_COEF_DUST*q_out[:-1]*cdust[:-1]
        m_BC_in = np.append(PARTITION_COEF_BC*rainfall*RAIN_CONC_BC,m_BC_in)
        m_dust_in = np.append(PARTITION_COEF_DUST*rainfall*RAIN_CONC_DUST,m_dust_in)
        # outward fluxes are simply flow out * concentration of the layer
        m_BC_out = PARTITION_COEF_BC*q_out*cBC
        m_dust_out = PARTITION_COEF_DUST*q_out*cdust

        # mass balance on each constituent
        dmBC = (m_BC_in - m_BC_out)*dt
        dmdust = (m_dust_in - m_dust_out)*dt
        mBC += dmBC.astype(float)
        mdust += dmdust.astype(float)
        layers.lBC[snow_firn_idx] = mBC
        layers.ldust[snow_firn_idx] = mdust
        return
    
    def refreezing(self,layers):
        """
        Calculates refreeze in layers due to temperatures below freezing with liquid water content.

        Parameters:
        -----------
        layers
            class object from pygem_eb.layers
        Returns:
        --------
        refreeze : float
            Total amount of refreeze [m w.e.]
        """
        # CONSTANTS
        HEAT_CAPACITY_ICE = eb_prms.Cp_ice
        DENSITY_ICE = eb_prms.density_ice
        DENSITY_WATER = eb_prms.density_water
        LH_RF = eb_prms.Lh_rf

        # LAYERS IN
        lT = layers.ltemp.copy()
        lw = layers.lwater.copy()
        ldm = layers.ldrymass.copy()
        lh = layers.lheight.copy()

        refreeze = np.zeros(layers.nlayers)
        for layer, T in enumerate(lT):
            if T < 0. and lw[layer] > 0:
                # calculate potential for refreeze [J m-2]
                E_temperature = np.abs(T)*ldm[layer]*HEAT_CAPACITY_ICE  # cold content available 
                E_water = lw[layer]*LH_RF  # amount of water to freeze
                E_pore = (DENSITY_ICE*lh[layer]-ldm[layer])*LH_RF # pore space available

                # calculate amount of refreeze in kg m-2
                dm_ref = np.min([abs(E_temperature),abs(E_water),abs(E_pore)])/LH_RF     # cannot be negative

                # add refreeze to array in m w.e.
                refreeze[layer] = dm_ref / DENSITY_WATER

                # add refreeze to layer ice mass
                ldm[layer] += dm_ref
                # update layer temperature from latent heat
                lT[layer] = -(E_temperature-dm_ref*LH_RF)/HEAT_CAPACITY_ICE/layers.ldrymass[layer]
                # update water content
                lw[layer] = max(0,layers.lwater[layer]-dm_ref) # cannot be negative
        
        # LAYERS OUT
        layers.ltemp = lT
        layers.lwater = lw
        layers.ldrymass = ldm
        layers.lheight = lh
        layers.updateLayerProperties()
        
        # Update refreeze with new refreeze content
        layers.lrefreeze += refreeze
        return np.sum(refreeze)
    
    def densification(self,layers,dt_dens):
        """
        Calculates densification of layers due to compression from overlying mass.
        Method Boone follows COSIPY. Other method doesn't work.

        Parameters:
        -----------
        layers
            class object from pygem_eb.layers
        dt_dens : float
            Timestep at which densification is applied [s]
        """
        # CONSTANTS
        GRAVITY = eb_prms.gravity
        VISCOSITY_SNOW = eb_prms.viscosity_snow
        DENSITY_FRESH_SNOW = eb_prms.density_fresh_snow

        # LAYERS IN
        snowfirn_idx = np.append(layers.snow_idx,layers.firn_idx)
        lp = layers.ldensity.copy()
        lT = layers.ltemp.copy()
        ldm = layers.ldrymass.copy()
        lw = layers.lwater.copy()

        if eb_prms.method_densification in ['Boone']:
            # EMPIRICAL PARAMETERS
            c1 = 2.8e-6
            c2 = 0.042
            c3 = 0.046
            c4 = 0.081
            c5 = 0.018

            # Loop through layers
            for layer,height in enumerate(layers.lheight[snowfirn_idx]):
                weight_above = GRAVITY*np.sum(ldm[:layer]+lw[:layer])
                viscosity = VISCOSITY_SNOW * np.exp(c4*(0.-lT[layer])+c5*lp[layer])

                # get change in density and recalculate height 
                dRho = (((weight_above*GRAVITY)/viscosity) + c1*np.exp(-c2*(0.-lT[layer]) 
                                - c3*np.maximum(0.,lp[layer]-DENSITY_FRESH_SNOW)))*lp[layer]*dt_dens
                lp[layer] += dRho

            # LAYERS OUT
            layers.ldensity = lp
            layers.lheight = layers.ldrymass/layers.ldensity
            layers.updateLayerProperties('depth')
            layers.updateLayerTypes()

        elif eb_prms.method_densification in ['off']:
            pass

        # DEBAM method - broken
        elif eb_prms.method_densification in ['DEBAM']:
            # get change in height and recalculate density from resulting compression
            for layer,height in enumerate(layers.lheight[snowfirn_idx]):
                weight_above = eb_prms.gravity*np.sum(layers.ldrymass[:layer])
                dD = height*weight_above/eb_prms.viscosity_snow/dt_dens
                layers.lheight[layer] -= dD
                layers.ldensity[layer] = layers.ldrymass[layer] / layers.lheight[layer]
                layers.updateLayerProperties('depth')
                layers.updateLayerTypes()

        return
    
    def getPrecip(self,enbal,surface,time):
        """
        Determines whether rain or snowfall occurred and outputs amounts.

        Parameters:
        -----------
        enbal
            class object from pygem_eb.energybalance
        surface
            class object from pygem_eb.surface
        time : pd.Datetime
            Current timestep
        Returns:
        --------
        rain, snowfall : float
            Specific mass of liquid and solid precipitation [kg m-2]
        """
        if enbal.tempC <= eb_prms.tsnow_threshold: 
            # there is precipitation and it falls as snow--set fresh snow timestamp
            surface.snow_timestamp = time
            rain = 0
            snow = enbal.prec*eb_prms.density_water*self.dt # kg m-2
        else:
            # precipitation falls as rain
            rain = enbal.prec*eb_prms.density_water*self.dt  # kg m-2
            snow = 0
        return rain,snow
      
    def solveHeatEq(self,layers,surftemp,dt_heat=eb_prms.dt_heateq):
        """
        Resolves the temperature profile from conduction of heat using 
        Forward-in-Time-Central-in-Space (FTCS) scheme

        Parameters:
        -----------
        layers
            class object from pygem_eb.layers
        surftemp : float
            Surface temperature [C]
        dt_heat : int
            Timestep to loop the heat equation solver [s]
        Returns:
        --------
        new_T : np.ndarray
            Array of new layer temperatures
        """
        # CONSTANTS
        HEAT_CAPACITY_ICE = eb_prms.Cp_ice

        # LAYERS IN
        nl = layers.nlayers
        lh = layers.lheight.copy()
        lp = layers.ldensity.copy()
        lT_old = layers.ltemp.copy()
        lT = layers.ltemp.copy()
        # lcond = 2.2*np.power(density/eb_prms.density_ice,1.88)
        lcond = 0.21e-01 + 0.42e-03 * lp + 0.22e-08 * lp ** 3
        Cp_ice = eb_prms.Cp_ice

        # set boundary conditions
        lT[-1] = lT_old[-1] # lowest layer of ice is ALWAYS at 0.*****

        if nl > 2:
            # heights of imaginary average bins between layers
            up_height = np.array([np.mean(lh[i:i+2]) for i in range(nl-2)])  # upper layer 
            dn_height = np.array([np.mean(lh[i+1:i+3]) for i in range(nl-2)])  # lower layer

            # conductivity
            up_cond = np.array([np.mean(lcond[i:i+2]*lh[i:i+2]) for i in range(nl-2)]) / up_height
            dn_cond = np.array([np.mean(lcond[i+1:i+3]*lh[i+1:i+3]) for i in range(nl-2)]) / dn_height

            # density
            up_dens = np.array([np.mean(lp[i:i+2]) for i in range(nl-2)]) / up_height
            dn_dens = np.array([np.mean(lp[i+1:i+3]) for i in range(nl-2)]) / dn_height

            # find temperature of top layer from surftemp boundary condition
            surf_cond = up_cond[0]*2/(up_dens[0]*up_height[0])*(surftemp-lT_old[0])
            subsurf_cond = dn_cond[0]/(up_dens[0]*up_height[0])*(lT_old[0]-lT_old[1])
            lT[0] = lT_old[0] + dt_heat/(Cp_ice*lh[0])*(surf_cond - subsurf_cond)
            if lT[0] > 0 or lT[0] < -50: 
            # If top layer of snow is very thin on top of ice, it can break this calculation
                lT[0] = np.mean([surftemp,lT_old[1]])

            surf_cond = up_cond/(up_dens*up_height)*(lT_old[:-2]-lT_old[1:-1])
            subsurf_cond = dn_cond/(dn_dens*dn_height)*(lT_old[1:-1]-lT_old[2:])
            lT[1:-1] = lT_old[1:-1] + dt_heat/(Cp_ice*lh[1:-1])*(surf_cond - subsurf_cond)

        elif nl > 1:
            lT = np.array([surftemp/2,0])
        else:
            lT = np.array([0])

        return lT

    def current_state(self,time,airtemp):
        layers = self.layers
        surftemp = self.surface.stemp
        albedo = self.surface.albedo
        melte = np.mean(self.output.meltenergy_output[-720:])
        melt = np.sum(self.output.melt_output[-720:])
        accum = np.sum(self.output.accum_output[-720:])

        layers.updateLayerProperties()
        snowdepth = np.sum(layers.lheight[layers.snow_idx])
        firndepth = np.sum(layers.lheight[layers.firn_idx])
        icedepth = np.sum(layers.lheight[layers.ice_idx])

        # Begin prints
        print(f'MONTH COMPLETED: {time.month_name()} {time.year} with +{accum:.2f} and -{melt:.2f} m w.e.')
        print(f'Currently {airtemp:.2f} C with {melte:.0f} W m-2 melt energy')
        print(f'----------surface albedo: {albedo:.3f} -----------')
        print(f'-----------surface temp: {surftemp:.2f} C-----------')
        if len(layers.snow_idx) > 0:
            print(f'|       snow depth: {snowdepth:.2f} m      {len(layers.snow_idx)} layers      |')
        if len(layers.firn_idx) > 0:
            print(f'|       firn depth: {firndepth:.2f} m      {len(layers.firn_idx)} layers      |')
        print(f'|       ice depth: {icedepth:.2f} m      {len(layers.ice_idx)} layers      |')
        for l in range(min(2,layers.nlayers)):
            print(f'--------------------layer {l}---------------------')
            print(f'     T = {layers.ltemp[l]:.1f} C                 h = {layers.lheight[l]:.3f} m ')
            print(f'                 p = {layers.ldensity[l]:.0f} kg/m3')
            print(f'Water Mass : {layers.lwater[l]:.2f} kg/m2   Dry Mass : {layers.ldrymass[l]:.2f} kg/m2')
        print('================================================')
        return

class Output():
    def __init__(self,time,bin_idx,args):
        """
        Creates netcdf file to save the model output.

        Parameters
        ----------
        """
        n_bins = eb_prms.n_bins
        bin_idxs = range(n_bins)
        self.n_timesteps = len(time)
        zeros = np.zeros([self.n_timesteps,n_bins,eb_prms.max_nlayers])

        # Create variable name dict
        vn_dict = {'EB':['SWin','SWout','LWin','LWout','rain','ground',
                         'sensible','latent','meltenergy','albedo'],
                   'MB':['melt','refreeze','runoff','accum','snowdepth'],
                   'Temp':['airtemp','surftemp'],
                   'Layers':['layertemp','layerdensity','layerwater','layerheight',
                             'layerBC','layerdust','layergrainsize']}

        # Create file to store outputs
        all_variables = xr.Dataset(data_vars = dict(
                SWin = (['time','bin'],zeros[:,:,0],{'units':'W m-2'}),
                SWout = (['time','bin'],zeros[:,:,0],{'units':'W m-2'}),
                LWin = (['time','bin'],zeros[:,:,0],{'units':'W m-2'}),
                LWout = (['time','bin'],zeros[:,:,0],{'units':'W m-2'}),
                rain = (['time','bin'],zeros[:,:,0],{'units':'W m-2'}),
                ground = (['time','bin'],zeros[:,:,0],{'units':'W m-2'}),
                sensible = (['time','bin'],zeros[:,:,0],{'units':'W m-2'}),
                latent = (['time','bin'],zeros[:,:,0],{'units':'W m-2'}),
                meltenergy = (['time','bin'],zeros[:,:,0],{'units':'W m-2'}),
                albedo = (['time','bin'],zeros[:,:,0],{'units':'0-1'}),
                melt = (['time','bin'],zeros[:,:,0],{'units':'m w.e.'}),
                refreeze = (['time','bin'],zeros[:,:,0],{'units':'m w.e.'}),
                runoff = (['time','bin'],zeros[:,:,0],{'units':'m w.e.'}),
                accum = (['time','bin'],zeros[:,:,0],{'units':'m w.e.'}),
                airtemp = (['time','bin'],zeros[:,:,0],{'units':'C'}),
                surftemp = (['time','bin'],zeros[:,:,0],{'units':'C'}),
                layertemp = (['time','bin','layer'],zeros,{'units':'C'}),
                layerwater = (['time','bin','layer'],zeros,{'units':'kg m-2'}),
                layerheight = (['time','bin','layer'],zeros,{'units':'m'}),
                layerdensity = (['time','bin','layer'],zeros,{'units':'kg m-3'}),
                layerBC = (['time','bin','layer'],zeros,{'units':'ppb'}),
                layerdust = (['time','bin','layer'],zeros,{'units':'ppm'}),
                layergrainsize = (['time','bin','layer'],zeros,{'units':'um'}),
                snowdepth = (['time','bin'],zeros[:,:,0],{'units':'m'})
                ),
                coords=dict(
                    time=(['time'],time),
                    bin = (['bin'],bin_idxs),
                    layer=(['layer'],np.arange(eb_prms.max_nlayers))
                    ))
        # Select variables from the specified input
        vars_list = vn_dict[eb_prms.store_vars[0]]
        for var in eb_prms.store_vars[1:]:
            vars_list.extend(vn_dict[var])
        # If on the first bin, create the netcdf file to store output
        if bin_idx == 0 and str(args.store_data) == 'True':
            all_variables[vars_list].to_netcdf(eb_prms.output_name+'.nc')

        # Initialize energy balance outputs
        self.SWin_output = []
        self.SWout_output = []
        self.LWin_output = []
        self.LWout_output = []
        self.rain_output = []
        self.ground_output = []
        self.sensible_output = []
        self.latent_output = []
        self.meltenergy_output = []
        self.albedo_output = []

        # Initialize mass balance outputs
        self.melt_output = []
        self.refreeze_output = []
        self.runoff_output = []
        self.accum_output = []
        self.snowdepth_output = []
        self.airtemp_output = []
        self.surftemp_output = []

        # Initialize layer outputs
        self.layertemp_output = dict()
        self.layerwater_output = dict()
        self.layerdensity_output = dict()
        self.layerheight_output = dict()
        self.layerBC_output = dict()
        self.layerdust_output = dict()
        self.layergrainsize_output = dict()
        return
    
    def storeTimestep(self,massbal,enbal,surface,layers,step):
        step = str(step)
        self.SWin_output.append(float(enbal.SWin))
        self.SWout_output.append(float(enbal.SWout))
        self.LWin_output.append(float(enbal.LWin))
        self.LWout_output.append(float(enbal.LWout))
        self.rain_output.append(float(enbal.rain))
        self.ground_output.append(float(enbal.ground))
        self.sensible_output.append(float(enbal.sens))
        self.latent_output.append(float(enbal.lat))
        self.meltenergy_output.append(float(surface.Qm))
        self.albedo_output.append(float(surface.albedo))
        self.melt_output.append(float(massbal.melt))
        self.refreeze_output.append(float(massbal.refreeze))
        self.runoff_output.append(float(massbal.runoff))
        self.accum_output.append(float(massbal.accum))
        self.airtemp_output.append(float(enbal.tempC))
        self.surftemp_output.append(float(surface.stemp))
        self.snowdepth_output.append(np.sum(layers.lheight[layers.snow_idx]))

        self.layertemp_output[step] = layers.ltemp
        self.layerwater_output[step] = layers.lwater
        self.layerheight_output[step] = layers.lheight
        self.layerdensity_output[step] = layers.ldensity
        self.layerBC_output[step] = layers.lBC / layers.lheight * 1e6
        self.layerdust_output[step] = layers.ldust / layers.lheight * 1e3
        self.layergrainsize_output[step] = layers.grainsize

    def storeData(self,bin):    
        with xr.open_dataset(eb_prms.output_name+'.nc') as dataset:
            ds = dataset.load()
            if 'EB' in eb_prms.store_vars:
                ds['SWin'].loc[:,bin] = self.SWin_output
                ds['SWout'].loc[:,bin] = self.SWout_output
                ds['LWin'].loc[:,bin] = self.LWin_output
                ds['LWout'].loc[:,bin] = self.LWout_output
                ds['rain'].loc[:,bin] = self.rain_output
                ds['ground'].loc[:,bin] = self.ground_output
                ds['sensible'].loc[:,bin] = self.sensible_output
                ds['latent'].loc[:,bin] = self.latent_output
                ds['meltenergy'].loc[:,bin] = self.meltenergy_output
                ds['albedo'].loc[:,bin] = self.albedo_output
            if 'MB' in eb_prms.store_vars:
                ds['melt'].loc[:,bin] = self.melt_output
                ds['refreeze'].loc[:,bin] = self.refreeze_output
                ds['runoff'].loc[:,bin] = self.runoff_output
                ds['accum'].loc[:,bin] = self.accum_output
                ds['snowdepth'].loc[:,bin] = self.snowdepth_output
            if 'Temp' in eb_prms.store_vars:
                ds['airtemp'].loc[:,bin] = self.airtemp_output
                ds['surftemp'].loc[:,bin] = self.surftemp_output
            if 'Layers' in eb_prms.store_vars:
                layertemp_output = pd.DataFrame.from_dict(self.layertemp_output,orient='index')
                layerdensity_output = pd.DataFrame.from_dict(self.layerdensity_output,orient='index')
                layerheight_output = pd.DataFrame.from_dict(self.layerheight_output,orient='index')
                layerwater_output = pd.DataFrame.from_dict(self.layerwater_output,orient='index')
                layerBC_output = pd.DataFrame.from_dict(self.layerBC_output,orient='index')
                layerdust_output = pd.DataFrame.from_dict(self.layerdust_output,orient='index')
                layergrainsize_output = pd.DataFrame.from_dict(self.layergrainsize_output,orient='index')
                
                if len(layertemp_output.columns) < eb_prms.max_nlayers:
                    n_columns = len(layertemp_output.columns)
                    for i in range(n_columns,eb_prms.max_nlayers):
                        nans = np.zeros(self.n_timesteps)*np.nan
                        layertemp_output[str(i)] = nans
                        layerdensity_output[str(i)] = nans
                        layerheight_output[str(i)] = nans
                        layerwater_output[str(i)] = nans
                        layerBC_output[str(i)] = nans
                        layerdust_output[str(i)] = nans
                        layergrainsize_output[str(i)] = nans

                ds['layertemp'].loc[:,bin,:] = layertemp_output
                ds['layerheight'].loc[:,bin,:] = layerheight_output
                ds['layerdensity'].loc[:,bin,:] = layerdensity_output
                ds['layerwater'].loc[:,bin,:] = layerwater_output
                ds['layerBC'].loc[:,bin,:] = layerBC_output
                ds['layerdust'].loc[:,bin,:] = layerdust_output
                ds['layergrainsize'].loc[:,bin,:] = layergrainsize_output
        ds.to_netcdf(eb_prms.output_name+'.nc')
        return
    
    def addVars(self):
        with xr.open_dataset(eb_prms.output_name+'.nc') as dataset:
            ds = dataset.load()
            ds['SWnet'] = ds['SWin'] + ds['SWout']
            ds['LWnet'] = ds['LWin'] + ds['LWout']
            ds['NetRad'] = ds['SWnet'] + ds['LWnet']
        ds.to_netcdf(eb_prms.output_name+'.nc')
        return
    
    def addAttrs(self,args,time_elapsed):
        time_elapsed = str(time_elapsed) + ' s'
        with xr.open_dataset(eb_prms.output_name+'.nc') as dataset:
            ds = dataset.load()
            ds = ds.assign_attrs(input_data=str(args.climate_input),
                                 run_start=str(args.startdate),
                                 run_end=str(args.enddate),
                                 n_bins=str(args.n_bins),
                                 model_run_date=str(pd.Timestamp.today()),
                                 switch_melt=str(args.switch_melt),
                                 switch_snow=str(args.switch_snow),
                                 switch_LAPs=str(args.switch_LAPs),
                                 time_elapsed=time_elapsed)
            if len(args.glac_no) > 1:
                reg = args.glac_no[0][0:2]
                ds = ds.assign_attrs(glacier=f'{len(args.glac_no)} glaciers in region {reg}')
            else:
                ds = ds.assign_attrs(glacier=eb_prms.glac_name)
        ds.to_netcdf(eb_prms.output_name+'.nc')
        return