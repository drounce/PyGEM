import numpy as np
import pandas as pd
import xarray as xr
import pygem_eb.input as eb_prms
import pygem_eb.energybalance as eb
import pygem_eb.layers as eb_layers
import pygem_eb.surface as eb_surface

class massBalance():
    """
    Mass balance scheme which calculates layer and bin mass balance from melt, refreeze and accumulation.
    Contains main() function which executes the core of the model.
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
        self.last=0
        # Set up model time
        self.dt = eb_prms.dt
        self.days_since_snowfall = 0
        self.time_list = dates_table['date']
        self.bin_idx = bin_idx

        # Initialize layers and surface classes
        self.layers = eb_layers.Layers(bin_idx,utils)
        self.surface = eb_surface.Surface(self.layers,self.time_list)

        # Initialize output class
        self.args = args
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
        # Get classes / time
        layers = self.layers
        surface = self.surface
        dt = self.dt

        # ===== ENTER TIME LOOP =====
        for time in self.time_list:
            # BEGIN MASS BALANCE
            # Initiate the energy balance to unpack climate data
            enbal = eb.energyBalance(climateds,time,self.bin_idx,dt)
            self.depBC,self.depdust = enbal.getDeposition()

            # Check if snowfall or rain occurred and update snow timestamp
            rain,snowfall = self.getPrecip(enbal,surface,time)

            # Add fresh snow to layers
            if snowfall > 0:
                store_surface = layers.addSnow(snowfall,enbal.tempC)
                if store_surface:
                    surface.storeSurface()
                    
            # Update daily properties
            if time.hour < 1 and time.minute < 1: 
                surface.updateSurfaceDaily(layers,time,self.args)
                self.days_since_snowfall = surface.days_since_snowfall
            
                # Update albedo
                surface.getAlbedo(layers,self.args)

            # Calculate surface energy balance by updating surface temperature
            surface.getSurfTemp(enbal,layers)

            # Calculate subsurface heating/melt from penetrating SW
            if layers.nlayers > 1: 
                SWin,SWout = enbal.getSW(surface.albedo)
                subsurf_melt = self.subsurfaceMelt(layers,SWin+SWout)
            else: # If there is only one layer, no subsurface melt occurs
                subsurf_melt = [0]

            # Calculate column melt including the surface
            if surface.Qm > 0:
                layermelt, fully_melted_mass = self.surfaceMelt(layers,surface,subsurf_melt)
            else: # No surface melt
                layermelt = subsurf_melt.copy()
                layermelt[0] = 0
                fully_melted_mass = 0

            # Percolate the meltwater and any liquid precipitation
            water_in = rain + fully_melted_mass
            runoff = self.percolation(layers,layermelt,water_in)
            
            # Update layers (checks for tiny or huge layers)
            layers.updateLayers()

            # Recalculate the temperature profile considering conduction
            if np.abs(np.sum(layers.ltemp)) != 0.:
                # If glacier is isothermal, heat is not conducted, so skip
                layers.ltemp = self.solveHeatEq(layers,surface.temp,eb_prms.dt_heateq)

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
                self.current_state(time,layers,surface.temp,enbal.tempC)
            # Advance timestep

        # Completed bin: store data
        if self.args.store_data:
            self.output.storeData(self.bin_idx)
        return

    def subsurfaceMelt(self,layers,SW_surf):
        """
        Calculates melt in subsurface layers (excluding layer 0) due to penetrating shortwave radiation.

        Parameters
        ----------
        layers
            class object from pygem_eb.layers.py
        SW_surf : float
            Incoming SW radiation [W m-2]
        Returns
        -------
        layermelt : np.ndarray
            Array containing subsurface melt amounts [kg m-2]
        """
        # Fraction of radiation absorbed at the surface depends on surface type
        if layers.ltype[0] in ['snow']:
            frac_absrad = 0.9
        else:
            frac_absrad = 0.8

        # Extinction coefficient depends on layer type
        extinct_coef = np.ones(layers.nlayers)*1e8 # ensures unfilled layers have 0 heat
        for layer,type in enumerate(layers.ltype):
            if type in ['snow']:
                extinct_coef[layer] = 17.1
            else:
                extinct_coef[layer] = 2.5
            # Cut off if the flux reaches zero threshold (1e-6)
            if np.exp(-extinct_coef[layer]*layers.ldepth[layer]) < 1e-6:
                break
        SW_pen = SW_surf*(1-frac_absrad)*np.exp(-extinct_coef*layers.ldepth)/self.dt

        # recalculate layer temperatures, leaving out the surface since surface heating by SW is calculated separately
        new_Tprofile = layers.ltemp.copy()
        new_Tprofile[1:] = layers.ltemp[1:] + SW_pen[1:]/(layers.ldrymass[1:]*eb_prms.Cp_ice)*self.dt

        # calculate melt from temperatures above 0
        layermelt = np.zeros(layers.nlayers)
        for layer,new_T in enumerate(new_Tprofile):
            # check if temperature is above 0
            if new_T > 0.:
                # calculate melt from the energy that raised the temperature above 0
                melt = (new_T-0.)*layers.ldrymass[layer]*eb_prms.Cp_ice/eb_prms.Lh_rf
                layers.ltemp[layer] = 0.
            else:
                melt = 0
                layers.ltemp[layer] = new_T
            layermelt[layer] = melt

        return layermelt

    def surfaceMelt(self,layers,surface,subsurf_melt):
        """
        For cases when bins are melting. Can melt multiple surface bins at once if Qm is
        sufficiently high. Otherwise, adds the surface layer melt to the array containing
        subsurface melt to return the total layer melt.

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
        layermelt = subsurf_melt.copy()                 # mass of melt due to penetrating SW in kg/m2
        surface_melt = surface.Qm*self.dt/eb_prms.Lh_rf # mass of melt due to SEB in kg/m2

        if surface_melt > layers.ldrymass[0]:
            # melt by surface energy balance completely melts surface layer, so check if it melts further layers
            fully_melted = np.where(np.cumsum(layers.ldrymass) <= surface_melt)[0]

            # calculate how much additional melt will occur in the first layer that's not fully melted
            newsurface_melt = surface_melt - np.sum(layers.ldrymass[fully_melted])
            newsurface_idx = fully_melted[-1] + 1
            # it's possible to fully melt that layer too when combined with penetrating SW melt:
            if newsurface_melt + layermelt[newsurface_idx] > layers.ldrymass[newsurface_idx]:
                fully_melted = np.append(fully_melted,newsurface_idx)
                # push new surface to the next layer down
                newsurface_melt -= layers.ldrymass[newsurface_idx]
                newsurface_idx += 1

            # set melt amounts from surface melt into melt array
            layermelt[fully_melted] = layers.ldrymass[fully_melted] 
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
        Calculates the liquid water content in each layer by downward percolation and adjusts 
        layer heights.

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
        rainfall = water_in
        # Percolation occurs only through snow and firn layers
        snow_firn_idx = np.concatenate([layers.snow_idx,layers.firn_idx])
        if len(snow_firn_idx) > 0:
            # ****** neglects subsurface melt if there is not snow or firn
            #  -- this is probably okay?
            ldm = layers.ldrymass.copy()[snow_firn_idx]
            lw = layers.lwater.copy()[snow_firn_idx]
            lh = layers.lheight.copy()[snow_firn_idx]
            layermelt_sf = layermelt[snow_firn_idx]
        
            if eb_prms.method_percolation in ['no_LAPs']:
                for layer,melt in enumerate(layermelt_sf):
                    # remove melt from the dry mass
                    ldm[layer] -= melt

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
                runoff = extra_water / eb_prms.density_water
                layers.lheight[snow_firn_idx] = lh
                layers.lwater[snow_firn_idx] = lw
                layers.ldrymass[snow_firn_idx] = ldm
                return runoff
            
            elif eb_prms.method_percolation in ['w_LAPs']:
                density_water = eb_prms.density_water
                density_ice = eb_prms.density_ice
                Sr = eb_prms.Sr
                dt = eb_prms.dt

                # calculate volumetric fractions
                theta_liq = lw / (lh*density_water)
                theta_ice = ldm / (lh*density_ice)
                porosity = 1 - theta_ice

                # initialize flow into the top layer
                qi = rainfall / eb_prms.dt
                q_in_store = []
                q_out_store = []
                for layer,melt in enumerate(layermelt_sf):
                    # set flow in equal to flow out of the previous layer
                    q_in = qi

                    # calculate flow out of layer i (cannot be negative)
                    flow_out = density_water*lh[layer]/dt * (theta_liq[layer] - Sr*porosity[layer])
                    qi = max(0,flow_out)

                    # check limit of qi based on underlying layer holding capacity
                    if layer < len(porosity) - 1 and theta_liq[layer] <= 0.3:
                        if porosity[layer] < 0.05 or porosity[layer+1] < 0.05:
                            # no flow if the layer itself or the lower layer has no pore space
                            lim = 0
                        else:
                            lim = density_water*lh[layer+1]/dt * (1-theta_ice[layer+1]-theta_liq[layer+1])
                            lim = max(0,lim)
                    else: # no limit on bottom layer (1e6 sufficiently high)
                        lim = 1e6
                    q_out = min(qi,lim)

                    # layer mass balance
                    lw[layer] += (q_in - q_out)*dt + melt
                    ldm[layer] -= melt
                    q_in_store.append(q_in)
                    q_out_store.append(q_out)
                # store new layer heights, water and solid mass
                layers.lheight[snow_firn_idx] = lh
                layers.lwater[snow_firn_idx] = lw
                layers.ldrymass[snow_firn_idx] = ldm
                runoff = q_out*dt + np.sum(layermelt[layers.ice_idx])
                # mass diffusion of LAPs
                self.diffuseLAPs(layers,q_in_store,q_out_store,rainfall)
        else:
            runoff = rainfall + np.sum(layermelt)
        return runoff
        
    def diffuseLAPs(self,layers,q_in,q_out,rainfall):
        # constants
        ksp_BC = eb_prms.ksp_BC
        ksp_dust = eb_prms.ksp_dust

        # load in layer data
        snow_firn_idx = np.concatenate([layers.snow_idx,layers.firn_idx])
        lh = layers.lheight[snow_firn_idx]
        lw = layers.lwater[snow_firn_idx]
        ldm = layers.ldrymass[snow_firn_idx]

        # lBC/dust is concentration of species in kg m-3
        # mBC/dust is layer mass of species in kg m-2
        # cBC/dust is layer mass mixing ratio in kg kg-1
        mBC = layers.lBC[snow_firn_idx] * lh
        mdust = layers.ldust[snow_firn_idx] * lh
        cBC = mBC / (lw + ldm)
        cdust = mdust / (lw+ldm)
        q_in = np.array(q_in)
        q_out = np.array(q_out)

        # inward fluxes depend on previous layer or rainfall for top layer
        m_BC_in = ksp_BC*q_out[:-1]*cBC[:-1]
        m_dust_in = ksp_dust*q_out[:-1]*cdust[:-1]
        m_BC_in = np.append(ksp_BC*rainfall*eb_prms.dt*eb_prms.rainBC,m_BC_in)
        m_dust_in = np.append(ksp_dust*rainfall*eb_prms.dt*eb_prms.raindust,m_dust_in)
        # outward fluxes are simply flow out * concentration of the layer
        m_BC_out = ksp_BC*q_out*cBC
        m_dust_out = ksp_dust*q_out*cdust
        # get deposition in top layer
        depBC = np.zeros_like(m_BC_in)
        depdust = np.zeros_like(m_BC_in)
        depBC[0] = self.depBC
        depdust[0] = self.depdust

        # mass balance on each constituent
        dmBC = (m_BC_in - m_BC_out + depBC)*eb_prms.dt
        dmdust = (m_dust_in - m_dust_out + depdust)*eb_prms.dt
        layers.lBC[snow_firn_idx] = (layers.lBC[snow_firn_idx]*lh + dmBC) / lh
        layers.ldust[snow_firn_idx] = (layers.ldust[snow_firn_idx]*lh + dmdust) / lh
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
        Cp_ice = eb_prms.Cp_ice
        density_ice = eb_prms.density_ice
        Lh_rf = eb_prms.Lh_rf
        refreeze = np.zeros(layers.nlayers)
        for layer, T in enumerate(layers.ltemp):
            if T < 0. and layers.lwater[layer] > 0:
                # calculate potential for refreeze [J m-2]
                E_temperature = np.abs(T)*layers.ldrymass[layer]*Cp_ice  # cold content available 
                E_water = layers.lwater[layer]*Lh_rf  # amount of water to freeze
                E_pore = (density_ice*layers.lheight[layer]-layers.ldrymass[layer])*Lh_rf # pore space available

                # calculate amount of refreeze in kg m-2
                dm_ref = np.min([abs(E_temperature),abs(E_water),abs(E_pore)])/Lh_rf     # cannot be negative

                # add refreeze to array in m w.e.
                refreeze[layer] = dm_ref /  eb_prms.density_water

                # add refreeze to layer ice mass
                layers.ldrymass[layer] += dm_ref
                # update layer temperature from latent heat
                layers.ltemp[layer] = -(E_temperature-dm_ref*Lh_rf)/Cp_ice/layers.ldrymass[layer]
                # update water content
                layers.lwater[layer] = max(0,layers.lwater[layer]-dm_ref) # cannot be negative
                # recalculate layer heights from new mass and update layers
                layers.lheight[layer] = layers.ldrymass[layer]/layers.ldensity[layer]
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
        # Only apply to snow and firn layers
        snowfirn_idx = np.append(layers.snow_idx,layers.firn_idx)

        if eb_prms.method_densification in ['Boone']:
            ldensity = layers.ldensity.copy()
            ltemp = layers.ltemp.copy()
            # Constants
            g = eb_prms.gravity
            c1 = 2.8e-6
            c2 = 0.042
            c3 = 0.046
            c4 = 0.081
            c5 = 0.018
            viscosity_0 = 3.7e7
            density_0 = eb_prms.density_fresh_snow

            # Loop through layers
            for layer,height in enumerate(layers.lheight[snowfirn_idx]):
                weight_above = eb_prms.gravity*np.sum(layers.ldrymass[:layer]+layers.lwater[:layer])
                viscosity = viscosity_0 * np.exp(c4*(0.-ltemp[layer])+c5*ldensity[layer])

                # get change in density and recalculate height 
                dRho = (((weight_above*g)/viscosity) + c1*np.exp(-c2*(0.-ltemp[layer]) - c3*np.maximum(0.,ldensity[layer]-density_0)))*ldensity[layer]*dt_dens
                ldensity[layer] += dRho

            # Update layer properties
            layers.ldensity = ldensity
            layers.lheight = layers.ldrymass/layers.ldensity
            layers.updateLayerProperties('depth')
            layers.updateLayerTypes()

        # DEBAM method - broken
        else:
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
        if enbal.prec > 1e-14 and enbal.tempC <= eb_prms.tsnow_threshold: 
            # there is precipitation and it falls as snow--set fresh snow timestamp
            surface.snow_timestamp = time
            rain = 0
            density_fresh_snow = max(109*6*(enbal.tempC-0.)+26*enbal.wind**0.5,50) # from CROCUS ***** CITE
            snow = enbal.prec*density_fresh_snow*self.dt # kg m-2
            precip_type = 'snow'
        elif enbal.tempC > eb_prms.tsnow_threshold:
            # precipitation falls as rain
            rain = enbal.prec*eb_prms.density_water*self.dt  # kg m-2
            snow = 0
            precip_type = 'rain'
        else: # no precipitation
            precip_type = 'none'
            rain = 0
            snow = 0
        # surface.updatePrecip(precip_type,rain+snow)
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
        nl = layers.nlayers
        height = layers.lheight
        density = layers.ldensity
        old_T = layers.ltemp
        new_T = old_T.copy()
        # conductivity = 2.2*np.power(density/eb_prms.density_ice,1.88)
        conductivity = 0.21e-01 + 0.42e-03 * density + 0.22e-08 * density ** 3
        Cp_ice = eb_prms.Cp_ice

        # set boundary conditions
        new_T[-1] = old_T[-1] # lowest layer of ice is ALWAYS at 0.*****

        if nl > 2:
            # heights of imaginary average bins between layers
            up_height = np.array([np.mean(height[i:i+2]) for i in range(nl-2)])  # upper layer 
            dn_height = np.array([np.mean(height[i+1:i+3]) for i in range(nl-2)])  # lower layer

            # conductivity
            up_cond = np.array([np.mean(conductivity[i:i+2]*height[i:i+2]) for i in range(nl-2)]) / up_height
            dn_cond = np.array([np.mean(conductivity[i+1:i+3]*height[i+1:i+3]) for i in range(nl-2)]) / dn_height

            # density
            up_dens = np.array([np.mean(density[i:i+2]) for i in range(nl-2)]) / up_height
            dn_dens = np.array([np.mean(density[i+1:i+3]) for i in range(nl-2)]) / dn_height

            # find temperature of top layer from surftemp boundary condition
            surf_cond = up_cond[0]*2/(up_dens[0]*up_height[0])*(surftemp-old_T[0])
            subsurf_cond = dn_cond[0]/(up_dens[0]*up_height[0])*(old_T[0]-old_T[1])
            new_T[0] = old_T[0] + dt_heat/(Cp_ice*height[0])*(surf_cond - subsurf_cond)
            if new_T[0] > 0: # If top layer of snow is very thin on top of ice, it breaks the temperature
                new_T[0] = 0 

            surf_cond = up_cond/(up_dens*up_height)*(old_T[:-2]-old_T[1:-1])
            subsurf_cond = dn_cond/(dn_dens*dn_height)*(old_T[1:-1]-old_T[2:])
            new_T[1:-1] = old_T[1:-1] + dt_heat/(Cp_ice*height[1:-1])*(surf_cond - subsurf_cond)

        elif nl > 1:
            if surftemp < 0:
                new_T = np.array([surftemp/2,0])
            else:
                new_T = np.array([0,0])
        else:
            new_T[0] = 0

        return new_T
    
    # def getMassBal(self,running_values,surftemp,enbal,month):
    #     if surftemp < 0:
    #         sublimation = min(enbal.lat/(eb_prms.density_water * eb_prms.Lv_sub), 0)*self.dt
    #         deposition = max(enbal.lat/(eb_prms.density_water * eb_prms.Lv_sub), 0)*self.dt
    #         evaporation = 0
    #         condensation = 0
    #     else:
    #         sublimation = 0
    #         deposition = 0
    #         evaporation = min(enbal.lat/(eb_prms.density_water * eb_prms.Lv_evap), 0)*self.dt
    #         condensation = max(enbal.lat/(eb_prms.density_water * eb_prms.Lv_evap), 0)*self.dt
    #     melt,runoff,refreeze,accum = running_values.loc[['melt','runoff','refreeze','accum']][0]
    #     self.monthly_output.loc[month] = [melt,runoff,refreeze,accum,0]

    #     # calculate total mass balance
    #     MB = accum + refreeze - melt + deposition - evaporation - sublimation
    #     self.monthly_output.loc[month]['MB'] = MB
    #     return

    def current_state(self,time,layers,surftemp,airtemp):
        melte = np.mean(self.output.meltenergy_output[-720:])
        melt = np.sum(self.output.melt_output[-720:])
        accum = np.sum(self.output.accum_output[-720:])

        layers.updateLayerProperties()
        snowdepth = np.sum(layers.lheight[layers.snow_idx])
        firndepth = np.sum(layers.lheight[layers.firn_idx])
        icedepth = np.sum(layers.lheight[layers.ice_idx])

        # Begin prints
        print(f'MONTH COMPLETED: {time.month_name()} {time.year} with +{accum:.2f} and -{melt:.2f} m w.e.')
        print(f'CURRENT STATE: Air temp: {airtemp:.2f} C     Melt energy: {melte:.0f} W m-2')
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
        vn_dict = {'EB':['SWin','SWout','LWin','LWout','rain','ground','sensible','latent','meltenergy'],
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
                layerBC = (['time','bin','layer'],zeros,{'units':'kg m-3'}),
                layerdust = (['time','bin','layer'],zeros,{'units':'kg m-3'}),
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
        if bin_idx == 0 and args.store_data:
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

        # Initialize mass balance outputs
        self.melt_output = []
        self.refreeze_output = []
        self.runoff_output = []
        self.accum_output = []
        self.airtemp_output = []
        self.surftemp_output = []

        # Initialize layer outputs
        self.layertemp_output = dict()
        self.layerwater_output = dict()
        self.layerdensity_output = dict()
        self.layerheight_output = dict()
        self.snowdepth_output = []
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
        self.melt_output.append(float(massbal.melt))
        self.refreeze_output.append(float(massbal.refreeze))
        self.runoff_output.append(float(massbal.runoff))
        self.accum_output.append(float(massbal.accum))
        self.airtemp_output.append(float(enbal.tempC))
        self.surftemp_output.append(float(surface.temp))
        self.snowdepth_output.append(np.sum(layers.lheight[layers.snow_idx]))

        self.layertemp_output[step] = layers.ltemp
        self.layerwater_output[step] = layers.lwater
        self.layerheight_output[step] = layers.lheight
        self.layerdensity_output[step] = layers.ldensity
        self.layerBC_output[step] = layers.lBC
        self.layerdust_output[step] = layers.ldust
        self.layergrainsize_output[step] = layers.grainsize

        # layertemp[eb_prms.max_nlayers-layers.nlayers:] = layers.ltemp
        # layerwater[eb_prms.max_nlayers-layers.nlayers:] = layers.lwater
        # layerheight[eb_prms.max_nlayers-layers.nlayers:] = layers.lheight
        # layerdensity[eb_prms.max_nlayers-layers.nlayers:] = layers.ldensity
        # self.layertemp_output.append(layertemp)
        # self.layerwater_output.append(layerwater)
        # self.layerheight_output.append(layerheight)
        # self.layerdensity_output.append(layerdensity)

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
            ds['albedo'] = -ds['SWout'] / ds['SWin']
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