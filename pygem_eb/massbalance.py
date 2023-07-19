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
    def __init__(self,bin_idx,climateds):
        """
        Initializes the layers and surface classes.

        Parameters
        ----------
        bin_idx : int
            Index value of the elevation bin
        climateds : xr.Dataset
            Dataset containing elevation-adjusted climate data
        """
        # Set up model time
        self.dt = eb_prms.dt
        self.days_since_snowfall = 0
        self.time_list = pd.date_range(eb_prms.startdate,eb_prms.enddate,freq=str(self.dt)+'S')
        self.bin_idx = bin_idx

        # Initialize layers and surface classes
        self.layers = eb_layers.Layers(bin_idx)
        self.surface = eb_surface.Surface(self.layers,self.time_list)

        # Initialize output class
        self.output = Output(self.time_list)
        return
    
    def main(self,climateds):
        """
        Runs the time loop and mass balance scheme to solve for melt, refreeze, accumulation and runoff.

        Parameters
        ----------
        climateds : xr.Dataset
            Dataset containing elevation-adjusted climate data
        """
        # Get layers and surface classes
        layers = self.layers
        surface = self.surface

        # Initialize time variable
        dt = self.dt
        timeidx = 0
        
        # Initialize space for "running" MB terms which sum each timestep -- reset to 0 monthly
        melt_monthly = 0
        refreeze_monthly = 0
        runoff_monthly = 0
        accum_monthly = 0

        # ===== ENTER TIME LOOP =====
        for time in self.time_list:
            # BEGIN MASS BALANCE
            # Initiate the energy balance to unpack climate data
            enbal = eb.energyBalance(climateds,time,self.bin_idx,dt)

            # Check if snowfall or rain occurred and update snow timestamp
            rain,snowfall = self.getPrecip(enbal,surface,time)

            # Update surface daily
            if time.hour < 1 and time.minute < 1:
                surface.days_since_snowfall = (time - surface.snow_timestamp)/pd.Timedelta(days=1)
                self.days_since_snowfall = surface.days_since_snowfall
                surface.updateSurface()

            # Add fresh snow to layers
            if snowfall > 0:
                layers.addSnow(snowfall,enbal.tempC)

            # Calculate surface energy balance by updating surface temperature
            surface.getSurfTemp(enbal,layers)

            # Calculate subsurface heating/melt from penetrating SW
            if layers.nlayers > 1: 
                Sin,Sout = enbal.getSW(surface.albedo)
                subsurf_melt = self.getSubsurfMelt(layers,Sin+Sout)
            else: # If there is bare ice, no subsurface melt occurs
                subsurf_melt = [0]

            # Calculate column melt including the surface
            if surface.Qm > 0:
                layermelt = self.getMelt(layers,surface,subsurf_melt)
            else: # no melt
                layermelt = subsurf_melt.copy()
                layermelt[0] = 0

            # Percolate the meltwater and any liquid precipitation
            runoff,layers_to_remove = self.percolation(layers,layermelt,rain)

            # Remove layers that were completely melted and merge/split layers as needed
            for layer in layers_to_remove:
                layers.removeLayer(layer)
            layers.updateLayers()

            # Recalculate the temperature profile considering conduction
            if surface.temp != 0. or np.abs(np.sum(layers.ltemp)) != 0.:
                # Save time by not resolving temperature profile when glacier is isothermal
                layers.ltemp = self.solveHeatEq(layers,surface.temp,eb_prms.dt_heateq)

            # Calculate refreeze
            refreeze = self.refreezing(layers)

            # Run densification daily **** hard-coded timestep
            if time.hour < 1 and time.minute < 1:
                self.densification(layers,self.dt*24)

            # END MASS BALANCE
            self.runoff = runoff
            self.melt = np.sum(layermelt) / eb_prms.density_water
            self.refreeze = refreeze
            self.accum = snowfall / eb_prms.density_water

            # Store running (monthly) values (all in m w.e.) 
            # **** this is mainly for debugging for the prints section
            # can be removed for the final product
            runoff_monthly += self.runoff
            melt_monthly += self.melt
            refreeze_monthly += self.refreeze
            accum_monthly += self.accum

            # Store data
            self.output.storeTimestep(self,enbal,surface,layers)

            # Monthly time check ***** debugging
            if (time+pd.Timedelta(hours=1)).is_month_start and time.hour == 23 and time.minute == 0:
                if eb_prms.debug: # MONTHLY PRINTS
                    melte = np.mean(self.output.meltenergy_output[-720:])
                    layers.updateLayerProperties
                    snowdepth = np.sum(layers.lheight[layers.snow_idx])
                    print(time.month_name(),time.year,'for bin no',self.bin_idx)
                    print(f'|    Qm: {melte:.0f} W/m2              Melt: {melt_monthly:.2f} m w.e.  |')
                    print(f'| Air temp: {enbal.tempC:.3f} C       Accum: {accum_monthly:.2f} m w.e.   |')
                    print(f'-------------surface temp: {surface.temp:.2f} C-------------')
                    print(f'|             snow height: {snowdepth:.2f} m             |')
                    if layers.nlayers > 3:
                        print(f'|    {layers.nlayers} layers total, only displaying top few  |')
                    for l in range(min(3,layers.nlayers)):
                        print(f'--------------------layer {l}---------------------')
                        print(f'     T = {layers.ltemp[l]:.1f} C                 h = {layers.lheight[l]:.3f} m ')
                        print(f'                 p = {layers.ldensity[l]:.0f} kg/m3')
                        print(f'Water Mass : {layers.lwater[l]:.2f} kg/m2   Dry Mass : {layers.ldrymass[l]:.2f} kg/m2')
                    print('================================================')
                    print(layers.ltype)
                # Reset monthly values
                melt_monthly = 0
                refreeze_monthly = 0
                runoff_monthly = 0
                accum_monthly = 0
            
            # Advance timestep
            timeidx += 1
        
        # Completed bin: store data
        if eb_prms.store_data:
            self.output.storeData(self.bin_idx)
        return

    def getSubsurfMelt(self,layers,Snet_surf):
        """
        Calculates melt in subsurface layers (excluding layer 0) due to penetrating shortwave radiation.

        Parameters
        ----------
        layers
            class object from pygem_eb.layers.py
        Snet_surf : float
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
        for idx,type in enumerate(layers.ltype):
            if type in ['snow']:
                extinct_coef[idx] = 17.1
            else:
                extinct_coef[idx] = 2.5
            # Cut off if the flux reaches zero threshold (1e-6)
            if np.exp(-extinct_coef[idx]*layers.ldepth[idx]) < 1e-6:
                break
        Snet_pen = Snet_surf*(1-frac_absrad)*np.exp(-extinct_coef*layers.ldepth)/self.dt

        # recalculate layer temperatures, leaving out the surface since surface temp is calculated separately
        new_Tprofile = layers.ltemp.copy()
        new_Tprofile[1:] = layers.ltemp[1:] + Snet_pen[1:]/(layers.ldrymass[1:]*eb_prms.Cp_ice)*self.dt

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

    def getMelt(self,layers,surface,subsurf_melt):
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
            Array containing melt amount [kg m-2]
        
        """
        layermelt = subsurf_melt.copy()

        surface_melt = surface.Qm*self.dt/eb_prms.Lh_rf         # melt in kg/m2
        if surface_melt > layers.ldrymass[0]:
            # melt by surface energy balance completely melts surface layer, so check if it melts further layers
            fully_melted = np.where(np.array([np.sum(layers.ldrymass[:i+1]) for i in range(layers.nlayers)]) <= surface_melt)[0]

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
        return layermelt
        
    def percolation(self,layers,layermelt,extra_water=0):
        """
        Calculates the liquid water content in each layer by downward percolation and adjusts 
        layer heights.

        Parameters
        ----------
        layers
            class object from pygem_eb.layers
        layermelt: np.ndarray
            Array containing melt amount for each layer
        extra_water : float
            Additional liquid water input (eg. rainfall) [kg m-2]

        Returns
        -------
        runoff : float
            Runoff that was not absorbed into void space [m w.e.]
        melted_layers : list
            List of layer indices that were fully melted
        """
        melted_layers = []
        for layer,melt in enumerate(layermelt):
            # check if the layer fully melted
            if melt >= layers.ldrymass[layer]:
                melted_layers.append(layer)
                # pass the meltwater to the next layer
                extra_water += layers.ldrymass[layer]+layers.lwater[layer]
            else:
                # remove melt from the dry mass
                layers.ldrymass[layer] -= melt

                # add melt and extra_water (melt from above) to layer water content
                added_water = melt + extra_water
                layers.lwater[layer] += added_water

                # check if meltwater exceeds the irreducible water content of the layer
                layers.updateLayerProperties('irrwater')
                if layers.lwater[layer] >= layers.irrwatercont[layer]:
                    # set water content to irr. water content and add the difference to extra_water
                    extra_water = layers.lwater[layer] - layers.irrwatercont[layer]
                    layers.lwater[layer] = layers.irrwatercont[layer]
                else: #if not overflowing, extra_water should be set back to 0
                    extra_water = 0
                
                # get the change in layer height due to loss of solid mass (volume only considers solid)
                layers.lheight[layer] -= melt/layers.ldensity[layer]
                # need to update layer depths
                layers.updateLayerProperties() 

        # extra water goes to runoff
        runoff = extra_water / eb_prms.density_water

        return runoff,melted_layers

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
        refreeze = 0
        for layer, T in enumerate(layers.ltemp):
            if T < 0. and layers.lwater[layer] > 0:
                # calculate potential for refreeze [J m-2]
                E_temperature = np.abs(T)*layers.ldrymass[layer]*Cp_ice  # cold content available 
                E_water = layers.lwater[layer]*Lh_rf  # amount of water to freeze
                E_pore = (density_ice*layers.lheight[layer]-layers.ldrymass[layer])*Lh_rf # pore space available

                # calculate amount of refreeze in kg m-2
                dm_ref = np.min([abs(E_temperature),abs(E_water),abs(E_pore)])/Lh_rf     # cannot be negative

                # add refreeze to running sum in m w.e.
                refreeze += dm_ref /  eb_prms.density_water

                # add refreeze to layer ice mass
                layers.ldrymass[layer] += dm_ref
                # update layer temperature from latent heat
                layers.ltemp[layer] = -(E_temperature-dm_ref*Lh_rf)/Cp_ice/layers.ldrymass[layer]
                # update water content
                layers.lwater[layer] = max(0,layers.lwater[layer]-dm_ref) # cannot be negative
                # recalculate layer heights from new mass and update layers
                layers.lheight[layer] = layers.ldrymass[layer]/layers.ldensity[layer]
                layers.updateLayerProperties()
        return refreeze
    
    def densification(self,layers,dt_dens):
        """
        Calculates densification of layers due to compression from overlying mass.
        Method Boone follows COSIPY.

        Parameters:
        -----------
        layers
            class object from pygem_eb.layers
        dt_dens : float
            Timestep at which densification is applied [s]
        """
        # only apply to snow and firn layers
        snowfirn_idx = np.append(layers.snow_idx,np.where(layers.ltype == 'firn')[0])

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
                dRho = (((weight_above*g)/viscosity) + c1*np.exp(-c2*(0.-ltemp[layer]) - c3*np.maximum(0.,ldensity[layer]-density_0)))*ldensity[layer]*self.dt
                ldensity[layer] += dRho

                layers.ldensity = ldensity
                layers.lheight[layer] = layers.ldrymass[layer]/layers.ldensity[layer]
                layers.updateLayerProperties('depth')
                layers.updateLayerTypes()

        else:
            # get change in height and recalculate density from resulting compression
            for layer,height in enumerate(layers.lheight[snowfirn_idx]):
                weight_above = eb_prms.gravity*np.sum(layers.ldrymass[:layer])
                dD = height*weight_above/eb_prms.viscosity_snow/self.dt
                layers.lheight[layer] -= dD
                layers.ldensity[layer] = layers.ldrymass[layer] / layers.lheight[layer]
                layers.updateLayerProperties('depth')
                layers.updateLayerTypes()

        return
    
    def getPrecip(self,enbal,surface,time):
        if enbal.prec > 1e-8 and enbal.tempC <= eb_prms.tsnow_threshold: 
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
        surface.updatePrecip(precip_type,rain+snow)
        return rain,snow
      
    def solveHeatEq(self,layers,surftemp,dt_heat):
        """
        Resolves the temperature profile from conduction of heat using Forward-in-Time-Central-in-Space (FTCS) scheme
        """
        nl = layers.nlayers
        height = layers.lheight
        density = layers.ldensity
        old_T = layers.ltemp
        new_T = old_T.copy()
        if nl > 1: #***** check on weird densities
            if np.max(density[:-1]) >= 900 or np.min(density) < 0:
                print(density)
                print('Height',height)
                print('Dry mass',layers.ldrymass)
                print('Water',layers.lwater)
        # conductivity = 2.2*np.power(density/eb_prms.density_ice,1.88)
        conductivity = 0.21e-01 + 0.42e-03 * density + 0.22e-08 * density ** 3
        Cp_ice = eb_prms.Cp_ice

        # set boundary conditions
        new_T[-1] = old_T[-1] # ice is ALWAYS isothermal*****

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
    
class Output():
    def __init__(self,time):
        n_bins = eb_prms.n_bins
        bin_idx = range(n_bins)
        zeros = np.zeros([len(time),n_bins,eb_prms.max_nlayers])

        # Create variable name dict
        vn_dict = {'EB':['SWin','SWout','LWin','LWout','rain','sensible','latent','meltenergy'],
                   'MB':['melt','refreeze','runoff','accum','snowdepth'],
                   'Temp':['airtemp','surftemp'],
                   'Layers':['layertemp','layerdensity','layerwater','layerheight']}

        # Create file to store outputs
        all_variables = xr.Dataset(data_vars = dict(
                SWin = (['time','bin'],zeros[:,:,0],{'units':'W m-2'}),
                SWout = (['time','bin'],zeros[:,:,0],{'units':'W m-2'}),
                LWin = (['time','bin'],zeros[:,:,0],{'units':'W m-2'}),
                LWout = (['time','bin'],zeros[:,:,0],{'units':'W m-2'}),
                rain = (['time','bin'],zeros[:,:,0],{'units':'W m-2'}),
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
                snowdepth = (['time','bin'],zeros[:,:,0],{'units':'m'})
                ),
                coords=dict(
                    time=(['time'],time),
                    bin = (['bin'],bin_idx),
                    layer=(['layer'],np.arange(eb_prms.max_nlayers))
                    ))
        vars_list = vn_dict[eb_prms.store_vars[0]]
        for var in eb_prms.store_vars[1:]:
            vars_list.extend(vn_dict[var])
        all_variables[vars_list].to_netcdf(eb_prms.output_name+'.nc')

        # Initialize energy balance outputs
        self.SWin_output = []
        self.SWout_output = []
        self.LWin_output = []
        self.LWout_output = []
        self.rain_output = []
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
        self.layertemp_output = []
        self.layerwater_output = []
        self.layerdensity_output = []
        self.layerheight_output = []
        self.snowdepth_output = []
        return
    
    def storeTimestep(self,massbal,enbal,surface,layers):
        self.SWin_output.append(enbal.SWin)
        self.SWout_output.append(enbal.SWout)
        self.LWin_output.append(enbal.LWin)
        self.LWout_output.append(enbal.LWout)
        self.rain_output.append(enbal.rain)
        self.sensible_output.append(enbal.sens)
        self.latent_output.append(enbal.lat)
        self.meltenergy_output.append(surface.Qm)
        self.melt_output.append(massbal.melt)
        self.refreeze_output.append(massbal.refreeze)
        self.runoff_output.append(massbal.runoff)
        self.accum_output.append(massbal.accum)
        self.airtemp_output.append(enbal.tempC)
        self.surftemp_output.append(surface.temp)
        self.snowdepth_output.append(np.sum(layers.lheight[layers.snow_idx]))

        layertemp = [None]*eb_prms.max_nlayers
        layerwater = layertemp.copy()
        layerheight = layertemp.copy()
        layerdensity = layertemp.copy()
        layertemp[eb_prms.max_nlayers-layers.nlayers:] = layers.ltemp
        layerwater[eb_prms.max_nlayers-layers.nlayers:] = layers.lwater
        layerheight[eb_prms.max_nlayers-layers.nlayers:] = layers.lheight
        layerdensity[eb_prms.max_nlayers-layers.nlayers:] = layers.ldensity
        self.layertemp_output.append(layertemp)
        self.layerwater_output.append(layerwater)
        self.layerheight_output.append(layerheight)
        self.layerdensity_output.append(layerdensity)
    
    def storeData(self,bin):
        with xr.open_dataset(eb_prms.output_name+'.nc') as dataset:
            ds = dataset.load()
            if 'EB' in eb_prms.store_vars:
                ds['SWin'].loc[:,bin] = self.SWin_output
                ds['SWout'].loc[:,bin] = self.SWout_output
                ds['LWin'].loc[:,bin] = self.LWin_output
                ds['LWout'].loc[:,bin] = self.LWout_output
                ds['rain'].loc[:,bin] = self.rain_output
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
                ds['layertemp'].loc[:,bin,:] = self.layertemp_output
                ds['layerheight'].loc[:,bin,:] = self.layerheight_output
                ds['layerdensity'].loc[:,bin,:] = self.layerdensity_output
                ds['layerwater'].loc[:,bin,:] = self.layerwater_output
        ds.to_netcdf(eb_prms.output_name+'.nc')
        return