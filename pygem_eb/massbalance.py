import numpy as np
import pandas as pd
import xarray as xr
import pygem_eb.input as eb_prms
import pygem_eb.energybalance as eb
import pygem_eb.layers as eb_layers
import pygem_eb.surface as eb_surface

class massBalance():
    """
    Mass balance scheme which calculates layer and bin mass balance 
    from melt, refreeze and accumulation. main() executes the core 
    of the model.
    """
    def __init__(self,bin_idx,args,climate):
        """
        Initializes the layers and surface classes and model 
        time for the mass balance scheme.

        Parameters
        ----------
        bin_idx : int
            Index value of the elevation bin
        dates : array-like (pd.datetime)
            List of local time dates
        """
        # Set up model time and bin
        self.dt = eb_prms.dt
        self.days_since_snowfall = 0
        self.time_list = climate.dates
        self.bin_idx = bin_idx
        self.elev = eb_prms.bin_elev[bin_idx]

        # Initialize climate, layers and surface classes
        self.args = args
        self.climate = climate
        self.layers = eb_layers.Layers(bin_idx,climate)
        self.surface = eb_surface.Surface(self.layers,self.time_list,args,climate)

        # Initialize output class
        self.output = Output(self.time_list,bin_idx,args)

        # Initialize mass balance check variables
        self.initial_mass = np.sum(self.layers.ldrymass)
        self.delay_time = self.time_list[0] - pd.Timedelta(days=1)
        return
    
    def main(self):
        """
        Runs the time loop and mass balance scheme to solve for melt, refreeze, 
        accumulation and runoff.
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
            enbal = eb.energyBalance(self.climate,time,self.bin_idx,dt)

            # Get rain and snowfall amounts [kg m-2]
            rain,snowfall = self.get_precip(enbal)

            # Add fresh snow to layers
            layers.add_snow(snowfall,enbal,surface,self.args,time)

            # Add dry deposited BC and dust to layers
            enbal.get_dry_deposition(layers)

            # Update daily properties
            if time.hour == 0: 
                surface.daily_updates(layers,enbal.tempC,surface.stemp,time)
                self.days_since_snowfall = surface.days_since_snowfall
                layers.lnewsnow = np.zeros(layers.nlayers)
            if time.hour in eb_prms.albedo_TOD:
                surface.get_albedo(layers,time)

            # Calculate surface energy balance by updating surface temperature
            surface.get_surftemp(enbal,layers)

            # Calculate subsurface heating from penetrating SW
            SWin,SWout = enbal.get_SW(surface)
            subsurf_melt = self.penetrating_SW(layers,SWin+SWout)
            
            # Calculate column melt including the surface
            layermelt = self.melt_layers(layers,subsurf_melt)
            
            # Percolate the meltwater, rain and LAPs
            runoff = self.percolation(enbal,layers,layermelt,rain)
            
            # Recalculate the temperature profile considering conduction
            self.solve_heat_eq(layers,surface.stemp)

            # Calculate refreeze
            refreeze = self.refreezing(layers)

            # Run densification daily
            if time.hour == 0:
                self.densification(layers)

            # Update snow/ice mass from latent heat fluxes (sublimation etc)
            # (***Currently neglected)

            # Check and update layer sizes
            layers.check_layers(time)

            # Check mass conserves
            self.check_mass_conservation(snowfall+rain, runoff)

            # END MASS BALANCE
            self.runoff = runoff
            self.melt = np.sum(layermelt) / eb_prms.density_water
            self.refreeze = refreeze
            self.accum = snowfall / eb_prms.density_water

            # Store timestep data
            self.output.store_timestep(self,enbal,surface,layers,time)   

            # Debugging: print current state and monthly melt at the end of each month
            if time.is_month_start and time.hour == 0 and eb_prms.debug:
                self.current_state(time,enbal.tempC)

            # Advance timestep
            pass

        # Completed bin: store data
        if self.args.store_data:
            self.output.store_data(self.bin_idx)

        if eb_prms.store_bands:
            surface.albedo_df.to_csv(eb_prms.albedo_out_fp.replace('.csv',f'_{self.elev}.csv'))
        return

    def penetrating_SW(self,layers,surface_SW):
        """
        Calculates melt in subsurface layers (excluding layer 0) 
        due to penetrating shortwave radiation.

        Parameters
        ----------
        surface_SW : float
            Incoming SW radiation [W m-2]

        Returns
        -------
        layermelt : np.ndarray
            Array containing subsurface melt amounts [kg m-2]
        """
        if layers.nlayers == 1: 
            return [0.]
        
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

    def melt_layers(self,layers,subsurf_melt):
        """
        For cases when bins are melting. Can melt multiple surface bins
        at once if Qm is sufficiently high. Otherwise, adds the surface
        layer melt to the array containing subsurface melt to return the
        total layer melt. (This function does NOT REMOVE MELTED MASS from 
        layers. That is done in percolation().)

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
        if self.surface.Qm <= 0:
            layermelt = subsurf_melt.copy()
            layermelt[0] = 0
            self.melted_layers = 0
            return layermelt
        
        # CONSTANTS
        LH_RF = eb_prms.Lh_rf

        # LAYERS IN
        ldm = layers.ldrymass.copy()
        lw = layers.lwater.copy()
        layermelt = subsurf_melt.copy()       # mass of melt due to penetrating SW [kg m-2]
        surface_melt = self.surface.Qm*self.dt/LH_RF     # mass of melt due to SEB [kg m-2]
        initial_mass = np.sum(layers.ldrymass + layers.lwater)

        if surface_melt > ldm[0]: # melt by surface energy balance completely melts surface layer
            # check if it melts further layers
            fully_melted = np.where(np.cumsum(ldm) <= surface_melt)[0]

            # calculate how much more melt occurs in the first layer not fully melted
            newsurface_melt = surface_melt - np.sum(ldm[fully_melted])
            newsurface_idx = fully_melted[-1] + 1

            # possible to fully melt that layer too when combined with penetrating SW melt:
            if newsurface_melt + layermelt[newsurface_idx] > ldm[newsurface_idx]:
                fully_melted = np.append(fully_melted,newsurface_idx)
                # push new surface to the next layer down
                newsurface_melt -= ldm[newsurface_idx]
                newsurface_idx += 1

            # set melt amounts from surface melt into melt array
            layermelt[fully_melted] = ldm[fully_melted] + lw[fully_melted]
            layermelt[newsurface_idx] += newsurface_melt 
        else:
            # only surface layer is melting
            layermelt[0] = surface_melt
            fully_melted = []
        
        class MeltedLayers():
            def __init__(self):
                self.mass = layermelt[fully_melted]
                self.BC = layers.lBC[fully_melted]
                self.dust = layers.ldust[fully_melted]

        self.melted_layers = MeltedLayers()

        # Remove layers that were completely melted 
        removed = 0 # Accounts for indexes of layers changing with loop
        for layer in fully_melted:
            layers.remove_layer(layer-removed)
            layermelt = np.delete(layermelt,layer-removed)
            removed += 1 

        # CHECK MASS CONSERVATION
        change = np.sum(layers.ldrymass + layers.lwater) - initial_mass
        if len(fully_melted) > 0:
            change += np.sum(self.melted_layers.mass)
        assert np.abs(change) < eb_prms.mb_threshold, 'melt_layers fails mass conservation'
        
        return layermelt
        
    def percolation(self,enbal,layers,layermelt,rainfall=0):
        """
        Calculates the liquid water content in each layer by downward
        percolation and applies melt.

        Parameters
        ----------
        enbal
            class object from pygem_eb.energybalance
        layers
            class object from pygem_eb.layers
        layermelt: np.ndarray
            Array containing melt amount for each layer
        rainfall : float
            Additional liquid water input from rainfall [kg m-2]

        Returns
        -------
        runoff : float
            Runoff of liquid water lost to system [m w.e.]
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
        initial_mass = np.sum(layers.ldrymass + layers.lwater)
        rain_bool = rainfall > 0

        # Get completely melted layers
        if self.melted_layers != 0:
            water_in = rainfall + np.sum(self.melted_layers.mass)
        else:
            water_in = rainfall

        if len(snow_firn_idx) > 1:
            # Calculate volumetric fractions (theta)
            theta_liq = lw / (lh*DENSITY_WATER)
            theta_ice = ldm / (lh*DENSITY_ICE)
            porosity = 1 - theta_ice

            # Account for melt in dry and wet mass
            ldm -= layermelt_sf
            lw += layermelt_sf

            # initialize flow into the top layer
            q_out = water_in / dt
            q_in_store = []
            q_out_store = []
            for layer,melt in enumerate(layermelt_sf):
                # set flow in equal to flow out of the previous layer
                q_in = q_out

                # calculate flow out of layer i (cannot be negative)
                flow_out = DENSITY_WATER*lh[layer]/dt * (
                    theta_liq[layer]-FRAC_IRREDUC*porosity[layer])
                qi = max(0,flow_out)

                # check limit of qi based on underlying layer holding capacity
                if layer < len(porosity) - 1 and theta_liq[layer] <= 0.3:
                    next = layer+1
                    lim = DENSITY_WATER*lh[next]/dt * (1-theta_ice[next]-theta_liq[next])
                    lim = max(0,lim)
                else: # no limit on bottom layer (1e6 sufficiently high)
                    lim = 1e6
                q_out = min(qi,lim)

                # layer mass balance
                lw[layer] += (q_in - q_out)*dt
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
                self.diffuse_LAPs(layers,np.array(q_out_store),enbal,rain_bool)
            
        else:
            # No percolation, but need to move melt to runoff
            layers.ldrymass -= layermelt
            layers.lheight -= layermelt / DENSITY_ICE
            runoff = water_in + np.sum(layermelt)

        # CHECK MASS CONSERVATION
        ins = water_in
        outs = runoff
        change = np.sum(layers.ldrymass + layers.lwater) - initial_mass
        assert np.abs(change - (ins-outs)) < eb_prms.mb_threshold, 'percolation fails mass conservation'

        return runoff
        
    def diffuse_LAPs(self,layers,q_out,enbal,rain_bool):
        """
        Diffuses LAPs vertically through the snow and firn layers based on
        inter-layer water fluxes from percolation.

        Parameters
        ----------
        layers
            class object from pygem_eb.layers
        q_out : np.ndarray
            Array containing water flow out of a layer [kg m-2 s-1]
        enbal
            class object from pygem_eb.energybalance
        rain_bool : Bool
            Raining or not?
        """
        # CONSTANTS
        PARTITION_COEF_BC = eb_prms.ksp_BC
        PARTITION_COEF_DUST = eb_prms.ksp_dust
        DEP_FACTOR = eb_prms.dep_factor
        dt = eb_prms.dt

        # LAYERS IN
        snow_firn_idx = np.concatenate([layers.snow_idx,layers.firn_idx])
        lw = layers.lwater[snow_firn_idx]
        ldm = layers.ldrymass[snow_firn_idx]

        # mBC/mdust is layer mass of species in kg m-2
        mBC = layers.lBC[snow_firn_idx]
        mdust = layers.ldust[snow_firn_idx]

        # get wet deposition into top layer if it's raining
        if rain_bool:
            mBC[0] += enbal.bcwet * dt * DEP_FACTOR
            mdust[0] += enbal.dustwet * eb_prms.ratio_DU3_DUtot * dt

        # cBC/cdust is layer mass mixing ratio in kg kg-1
        cBC = mBC / (lw + ldm)
        cdust = mdust / (lw + ldm)

        if self.melted_layers != 0:
            m_BC_in_top = np.array(np.sum(self.melted_layers.BC) / dt)
            m_dust_in_top = np.array(np.sum(self.melted_layers.dust) / dt)
        else:
            m_BC_in_top = np.array([0],dtype=float) 
            m_dust_in_top = np.array([0],dtype=float)
        m_BC_in_top *= PARTITION_COEF_BC
        m_dust_in_top *= PARTITION_COEF_DUST

        # inward fluxes = outward fluxes from previous layer
        m_BC_in = PARTITION_COEF_BC*q_out[:-1]*cBC[:-1]
        m_dust_in = PARTITION_COEF_DUST*q_out[:-1]*cdust[:-1]
        m_BC_in = np.append(m_BC_in_top,m_BC_in)
        m_dust_in = np.append(m_dust_in_top,m_dust_in)

        # outward fluxes are simply (flow out)*(concentration of the layer)
        m_BC_out = PARTITION_COEF_BC*q_out*cBC
        m_dust_out = PARTITION_COEF_DUST*q_out*cdust

        # mass balance on each constituent
        dmBC = (m_BC_in - m_BC_out)*dt
        dmdust = (m_dust_in - m_dust_out)*dt
        mBC += dmBC.astype(float)
        mdust += dmdust.astype(float)

        # LAYERS OUT
        layers.lBC[snow_firn_idx] = mBC
        layers.ldust[snow_firn_idx] = mdust
        return
    
    def refreezing(self,layers):
        """
        Calculates refreeze in layers due to temperatures below freezing
        with liquid water content.

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

        initial_mass = np.sum(layers.ldrymass + layers.lwater)
        refreeze = np.zeros(layers.nlayers)
        for layer, T in enumerate(lT):
            if T < 0. and lw[layer] > 0:
                # calculate potential for refreeze [J m-2]
                E_cold = np.abs(T)*ldm[layer]*HEAT_CAPACITY_ICE  # cold content available 
                E_water = lw[layer]*LH_RF  # amount of water to freeze
                E_pore = (DENSITY_ICE*lh[layer]-ldm[layer])*LH_RF # pore space available

                # calculate amount of refreeze in kg m-2
                dm_ref = np.min([abs(E_cold),abs(E_water),abs(E_pore)])/LH_RF

                # add refreeze to array in m w.e.
                refreeze[layer] = dm_ref / DENSITY_WATER

                # add refreeze to layer ice mass
                ldm[layer] += dm_ref
                # update layer temperature from latent heat
                lT[layer] = min(0,-(E_cold-dm_ref*LH_RF)/(HEAT_CAPACITY_ICE*ldm[layer]))
                # update water content
                lw[layer] = max(0,layers.lwater[layer]-dm_ref)
        
        # Update refreeze with new refreeze content
        layers.lrefreeze += refreeze

        # LAYERS OUT
        layers.ltemp = lT
        layers.lwater = lw
        layers.ldrymass = ldm
        layers.lheight = lh
        layers.update_layer_props()

        # CHECK MASS CONSERVATION
        change = np.sum(layers.ldrymass + layers.lwater) - initial_mass
        assert np.abs(change) < eb_prms.mb_threshold, 'refreezing fails mass conservation'

        return np.sum(refreeze)
    
    def densification(self,layers):
        """
        Calculates densification of layers due to compression from overlying mass.

        Parameters:
        -----------
        layers
            class object from pygem_eb.layers
        """
        # CONSTANTS
        GRAVITY = eb_prms.gravity
        R = eb_prms.R_gas
        VISCOSITY_SNOW = eb_prms.viscosity_snow
        rho = eb_prms.constant_snowfall_density
        DENSITY_FRESH_SNOW = rho if rho else 50
        DENSITY_ICE = eb_prms.density_ice
        dt = eb_prms.daily_dt

        # LAYERS IN
        snowfirn_idx = np.append(layers.snow_idx,layers.firn_idx)
        lp = layers.ldensity.copy()
        lT = layers.ltemp.copy()
        ldm = layers.ldrymass.copy()
        lw = layers.lwater.copy()

        initial_mass = np.sum(layers.ldrymass + layers.lwater)

        if eb_prms.method_densification in ['Boone']:
            # EMPIRICAL PARAMETERS
            c1 = 2.7e-6     # s-1 (2.7e-6) --> 2.7e-4
            c2 = 0.042      # K-1 (0.042)
            c3 = 0.046      # m3 kg-1 (0.046)
            c4 = 0.081      # K-1 (0.081)
            c5 = 0.018       # m3 kg-1 (0.018) --> 0.07

            for layer in snowfirn_idx:
                weight_above = GRAVITY*np.sum(ldm[:layer]+lw[:layer])
                viscosity = VISCOSITY_SNOW*np.exp(c4*(0.-lT[layer])+c5*lp[layer])

                # get change in density
                mass_term = (weight_above*GRAVITY)/viscosity
                temp_term = -c2*(0.-lT[layer])
                dens_term = -c3*max(0,lp[layer]-DENSITY_FRESH_SNOW)
                dRho = (mass_term+c1*np.exp(temp_term+dens_term))*lp[layer]*dt
                lp[layer] += dRho

            # LAYERS OUT
            layers.ldensity = lp
            layers.lheight = ldm / lp
            layers.update_layer_props('depth')

        # Herron Langway (1980) method
        elif eb_prms.method_densification in ['HerronLangway']:
            # yearly accumulation is the maximum layer snow mass in mm w.e. yr-1
            a = layers.max_snow / (dt*365) # kg m-2 = mm w.e.
            k = np.zeros_like(lp)
            b = np.zeros_like(lp)
            for layer,density in enumerate(lp[snowfirn_idx]):
                lTK = lT[layer] + 273.15
                if density < 550:
                    b[layer] = 1
                    k[layer] = 11*np.exp(-10160/(R*lTK))
                else:
                    b[layer] = 0.5
                    k[layer] = 575*np.exp(-21400/(R*lTK))
            dRho = k*a**b*(DENSITY_ICE - lp)/DENSITY_ICE*dt
            lp += dRho

            # LAYERS OUT
            layers.ldensity = lp
            layers.lheight = ldm / lp
            layers.update_layer_props('depth')

        # Kojima (1967) method (JULES)
        elif eb_prms.method_densification in ['Kojima']:
            NU_0 = 1e7      # Pa s
            RHO_0 = 50      # kg m-3
            k_S = 4000      # K
            T_m = 0. + 273.15
            for layer in snowfirn_idx:
                weight_above = GRAVITY*np.sum(ldm[:layer]+lw[:layer])

                # get change in density
                T_K = lT[layer] + 273.15
                exp_term = np.exp(k_S/T_m - k_S/T_K - lp[layer]/RHO_0)
                dRho = lp[layer]*weight_above/NU_0*exp_term
                lp[layer] += dRho

            # LAYERS OUT
            layers.ldensity = lp
            layers.lheight = ldm / lp
            layers.update_layer_props('depth')

        # Check if new firn or ice layers were created
        layers.update_layer_types()

        # CHECK MASS CONSERVATION
        change = np.sum(layers.ldrymass + layers.lwater) - initial_mass
        assert np.abs(change) < eb_prms.mb_threshold, 'densification fails mass conservation'
        return
    
    def get_precip(self,enbal):
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
        # CONSTANTS
        SNOW_THRESHOLD_LOW = eb_prms.snow_threshold_low
        SNOW_THRESHOLD_HIGH = eb_prms.snow_threshold_high
        DENSITY_WATER = eb_prms.density_water

        # Define rain vs snow scaling 
        rain_scale = np.arange(0,1,20)
        temp_scale = np.arange(SNOW_THRESHOLD_LOW,SNOW_THRESHOLD_HIGH,20)
        
        if enbal.tempC <= SNOW_THRESHOLD_LOW: 
            # precip falls as snow
            rain = 0
            snow = enbal.tp*DENSITY_WATER
            if 4 < self.time.month < 10: # kp adjusts only winter snowfall
                snow /= eb_prms.kp
        elif SNOW_THRESHOLD_LOW < enbal.tempC < SNOW_THRESHOLD_HIGH:
            # mix of rain and snow
            fraction_rain = np.interp(enbal.tempC,temp_scale,rain_scale)
            rain = enbal.tp*fraction_rain*DENSITY_WATER
            snow = enbal.tp*(1-fraction_rain)*DENSITY_WATER
        else:
            # precip falls as rain
            rain = enbal.tp*DENSITY_WATER
            snow = 0
        return rain,snow  # kg m-2
      
    def solve_heat_eq(self,layers,surftemp,dt_heat=eb_prms.dt_heateq):
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
        if np.sum(layers.ltemp) == 0.:
            return
        
        # CONSTANTS
        CP_ICE = eb_prms.Cp_ice
        DENSITY_ICE = eb_prms.density_ice
        TEMP_TEMP = eb_prms.temp_temp
        TEMP_DEPTH = eb_prms.temp_depth

        # set temperate ice and heat-diffusing layer indices
        temperate_idx = np.where(layers.ldepth > TEMP_DEPTH)[0]
        diffusing_idx = np.arange(temperate_idx[0])
        layers.ltemp[temperate_idx] = TEMP_TEMP

        # LAYERS IN
        nl = len(diffusing_idx)
        lh = layers.lheight[diffusing_idx]
        lp = layers.ldensity[diffusing_idx]
        lT_old = layers.ltemp[diffusing_idx]
        lT = layers.ltemp[diffusing_idx]

        # get conductivity 
        if eb_prms.constant_conductivity:
            lcond = np.ones(nl)*eb_prms.k_ice
        elif eb_prms.method_conductivity in ['VanDusen']:
            lcond = 0.21e-01 + 0.42e-03*lp + 0.22e-08*lp**3
        elif eb_prms.method_conductivity in ['Sturm']:
            lcond = 0.0138 - 1.01e-3*lp + 3.233e-6*lp**2
        elif eb_prms.method_conductivity in ['Douville']:
            lcond = 2.2*np.power(lp/DENSITY_ICE,1.88)
        elif eb_prms.method_conductivity in ['Jansson']:
            lcond = 0.02093 + 0.7953e-3*lp + 1.512e-12*lp**4
        elif eb_prms.method_conductivity in ['OstinAndersson']:
            lcond = -8.71e-3 + 0.439e-3*lp + 1.05e-6*lp**2

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
            lT[0] = lT_old[0] + dt_heat/(CP_ICE*lh[0])*(surf_cond - subsurf_cond)
            if lT[0] > 0 or lT[0] < -50: 
            # If top layer of snow is very thin on top of ice, it can break this calculation
                lT[0] = np.mean([surftemp,lT_old[1]])

            surf_cond = up_cond/(up_dens*up_height)*(lT_old[:-2]-lT_old[1:-1])
            subsurf_cond = dn_cond/(dn_dens*dn_height)*(lT_old[1:-1]-lT_old[2:])
            lT[1:-1] = lT_old[1:-1] + dt_heat/(CP_ICE*lh[1:-1])*(surf_cond - subsurf_cond)

        elif nl > 1:
            lT = np.array([surftemp/2,0])
        else:
            lT = np.array([0])

        # LAYERS OUT
        layers.ltemp[diffusing_idx] = lT
        return 

    def current_state(self,time,airtemp):
        layers = self.layers
        surftemp = self.surface.stemp
        albedo = self.surface.bba
        melte = np.mean(self.output.meltenergy_output[-720:])
        melt = np.sum(self.output.melt_output[-720:])
        accum = np.sum(self.output.accum_output[-720:])
        ended_month = (time - pd.Timedelta(days=1)).month_name()

        layers.update_layer_props()
        snowdepth = np.sum(layers.lheight[layers.snow_idx])
        firndepth = np.sum(layers.lheight[layers.firn_idx])
        icedepth = np.sum(layers.lheight[layers.ice_idx])

        # Begin prints
        print(f'MONTH COMPLETED: {ended_month} {time.year} with +{accum:.2f} and -{melt:.2f} m w.e.')
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
    
    def check_mass_conservation(self,mass_in,mass_out):
        """
        Checks mass was conserved within the last timestep
        mass_in:    sum of precipitation (kg m-2)
        mass_out:   sum of runoff (kg m-2)
        """
        # difference in mass since the last timestep
        current_mass = np.sum(self.layers.ldrymass + self.layers.lwater)
        diff = current_mass - self.initial_mass
        out_in = mass_out - mass_in
        
        # account for delayed snow in mass conservation check
        if self.layers.delayed_snow > 0:
            self.delay_time = self.time
            self.delayed_snow = self.layers.delayed_snow
        
        # delayed snow affects the timestep of delay and one hour after
        if self.time - self.delay_time <= pd.Timedelta(hours=1):
            check = np.abs(diff + out_in) - self.delayed_snow < eb_prms.mb_threshold
        else:
            check = np.abs(diff + out_in) < eb_prms.mb_threshold
        assert check, f'Timestep {self.time} failed mass conservation'
        
        # new initial mass
        self.initial_mass = current_mass
        return

class Output():
    def __init__(self,time,bin_idx,args):
        """
        Creates netcdf file where the model output will be saved.

        Parameters
        ----------
        """
        n_bins = eb_prms.n_bins
        bin_idxs = range(n_bins)
        self.n_timesteps = len(time)
        self.n_bins = n_bins
        zeros = np.zeros([self.n_timesteps,n_bins,eb_prms.max_nlayers])

        # Create variable name dict
        vn_dict = {'EB':['SWin','SWout','LWin','LWout','rain','ground',
                         'sensible','latent','meltenergy','albedo',
                         'SWin_sky','SWin_terr'],
                   'MB':['melt','refreeze','runoff','accum','snowdepth'],
                   'Temp':['airtemp','surftemp'],
                   'Layers':['layertemp','layerdensity','layerwater','layerheight',
                             'layerBC','layerdust','layergrainsize']}

        # Create file to store outputs
        all_variables = xr.Dataset(data_vars = dict(
                SWin = (['time','bin'],zeros[:,:,0],{'units':'W m-2'}),
                SWout = (['time','bin'],zeros[:,:,0],{'units':'W m-2'}),
                SWin_sky = (['time','bin'],zeros[:,:,0],{'units':'W m-2'}),
                SWin_terr = (['time','bin'],zeros[:,:,0],{'units':'W m-2'}),
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
        self.SWin_output = []       # incoming shortwave [W m-2]
        self.SWout_output = []      # outgoing shortwave [W m-2]
        self.LWin_output = []       # incoming longwave [W m-2]
        self.LWout_output = []      # outgoing longwave [W m-2]
        self.SWin_sky_output = []   # incoming sky shortwave [W m-2]
        self.SWin_terr_output = []  # incoming terrain shortwave [W m-2]
        self.rain_output = []       # rain energy [W m-2]
        self.ground_output = []     # ground energy [W m-2]
        self.sensible_output = []   # sensible energy [W m-2]
        self.latent_output = []     # latent energy [W m-2]
        self.meltenergy_output = [] # melt energy [W m-2]
        self.albedo_output = []     # surface broadband albedo [-]

        # Initialize mass balance outputs
        self.snowdepth_output = []  # depth of snow [m]
        self.melt_output = []       # melt by timestep [m w.e.]
        self.refreeze_output = []   # refreeze by timestep [m w.e.]
        self.accum_output = []      # accumulation by timestep [m w.e.]
        self.runoff_output = []     # runoff by timestep [m w.e.]
        self.airtemp_output = []    # downscaled air temperature [C]
        self.surftemp_output = []   # surface temperature [C]

        # Initialize layer outputs
        self.layertemp_output = dict()      # layer temperature [C]
        self.layerwater_output = dict()     # layer water content [kg m-2]
        self.layerdensity_output = dict()   # layer density [kg m-3]
        self.layerheight_output = dict()    # layer height [m]
        self.layerBC_output = dict()        # layer black carbon content [ppb]
        self.layerdust_output = dict()      # layer dust content [ppm]
        self.layergrainsize_output = dict() # layer grain size [um]
        return
    
    def store_timestep(self,massbal,enbal,surface,layers,step):
        step = str(step)
        self.SWin_output.append(float(enbal.SWin))
        self.SWout_output.append(float(enbal.SWout))
        self.LWin_output.append(float(enbal.LWin))
        self.LWout_output.append(float(enbal.LWout))
        self.SWin_sky_output.append(float(enbal.SWin_sky))
        self.SWin_terr_output.append(float(enbal.SWin_terr))
        self.rain_output.append(float(enbal.rain))
        self.ground_output.append(float(enbal.ground))
        self.sensible_output.append(float(enbal.sens))
        self.latent_output.append(float(enbal.lat))
        self.meltenergy_output.append(float(surface.Qm))
        self.albedo_output.append(float(surface.bba))
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

    def store_data(self,bin):    
        with xr.open_dataset(eb_prms.output_name+'.nc') as dataset:
            ds = dataset.load()
            if 'EB' in eb_prms.store_vars:
                ds['SWin'].loc[:,bin] = self.SWin_output
                ds['SWout'].loc[:,bin] = self.SWout_output
                ds['LWin'].loc[:,bin] = self.LWin_output
                ds['LWout'].loc[:,bin] = self.LWout_output
                ds['SWin_sky'].loc[:,bin] = self.SWin_sky_output
                ds['SWin_terr'].loc[:,bin] = self.SWin_terr_output
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
        return ds
    
    def add_vars(self):
        """
        Adds additional variables to the output dataset.
        
        dh: surface height change [m]
        SWnet: net shortwave radiation flux [W m-2]
        LWnet: net longwave radiation flux [W m-2]
        NetRad: net radiation flux (SW and LW) [W m-2]
        """
        with xr.open_dataset(eb_prms.output_name+'.nc') as dataset:
            ds = dataset.load()

            # add surface height change
            height = ds.layerheight.sum(dim='layer')
            diff = height.diff(dim='time')
            initial = np.array([0]*self.n_bins).reshape(-1,1)
            dh = np.append(initial,diff.values.T,axis=1)
            if dh.shape == (len(ds.time),):
                dh = np.array([dh]).reshape(-1,1)
            elif dh.shape[0] == self.n_bins:
                dh = dh.T
            ds['dh'] = (['time','bin'],dh,{'units':'m'})

            # add summed radiation terms
            SWnet = ds['SWin'] + ds['SWout']
            LWnet = ds['LWin'] + ds['LWout']
            NetRad = SWnet + LWnet
            ds['SWnet'] = (['time','bin'],SWnet.values,{'units':'W m-2'})
            ds['LWnet'] = (['time','bin'],LWnet.values,{'units':'W m-2'})
            ds['NetRad'] = (['time','bin'],NetRad.values,{'units':'W m-2'})
        ds.to_netcdf(eb_prms.output_name+'.nc')
        return
    
    def add_basic_attrs(self,args,time_elapsed,climate):
        """
        Adds informational attributes to the output dataset.

        time_elapsed
        run_start and run_end
        from_AWS and from_reanalysis list of variables
        n_bins and bin_elev
        model_run_date
        switch_melt, switch_LAPs, switch_snow
        """
        time_elapsed = str(time_elapsed) + ' s'
        bin_elev = ', '.join([str(z) for z in eb_prms.bin_elev])

        # get information on variable sources
        re_str = eb_prms.reanalysis+': '
        props = eb_prms.glac_props[eb_prms.glac_no[0]]
        if args.use_AWS:
            measured = climate.measured_vars
            AWS_str = props['name']+' '+str(props['site_elev'])+': '
            AWS_str += ', '.join(measured)
            re_vars = [e for e in climate.all_vars if e not in measured]
            re_str += ', '.join(re_vars)
        else:
            re_str += 'all'
            AWS_str = 'none'
        
        with xr.open_dataset(eb_prms.output_name+'.nc') as dataset:
            ds = dataset.load()
            ds = ds.assign_attrs(from_AWS=AWS_str,
                                 from_reanalysis=re_str,
                                 run_start=str(args.startdate),
                                 run_end=str(args.enddate),
                                 n_bins=str(args.n_bins),
                                 bin_elev=bin_elev,
                                 model_run_date=str(pd.Timestamp.today()),
                                 switch_melt=str(args.switch_melt),
                                 switch_snow=str(args.switch_snow),
                                 switch_LAPs=str(args.switch_LAPs),
                                 time_elapsed=time_elapsed,
                                 run_by=eb_prms.machine)
            if len(args.glac_no) > 1:
                reg = args.glac_no[0][0:2]
                ds = ds.assign_attrs(glacier=f'{len(args.glac_no)} glaciers in region {reg}')
            else:
                ds = ds.assign_attrs(glacier=eb_prms.glac_name)
        ds.to_netcdf(eb_prms.output_name+'.nc')
        return
    
    def add_attrs(self,new_attrs):
        """
        Adds new attributes as a dict to the output dataset
        """
        with xr.open_dataset(eb_prms.output_name+'.nc') as dataset:
            ds = dataset.load()
            if not new_attrs:
                return ds
            ds = ds.assign_attrs(new_attrs)
        ds.to_netcdf(eb_prms.output_name+'.nc')
        return ds