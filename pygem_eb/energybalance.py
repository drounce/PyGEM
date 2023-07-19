#import xarray as xr
import numpy as np
import pygem_eb.input as eb_prms

class energyBalance():
    """
    Energy balance scheme that calculates the surface energy balance and penetrating shortwave radiation. 
    This class is updated within MassBalance every timestep, so it stores the current climate data and 
    surface fluxes.
    """ 
    def __init__(self,climateds,time,bin_idx,dt):
        """
        Loads in the climate data at a given timestep to use in the surface energy balance.

        Parameters
        ----------
        climateds : xr.Dataset
            Climate dataset containing temperature, precipitation, pressure, wind speed,
            shortwave radiation, and total cloud cover.
        time : datetime
            Time to index the climate dataset.
        bin_idx : int
            Index number of the bin being run
        dt : float
            Resolution for the time loop [s]
        """
        # Unpack climate variables
        if time.minute == 0:
            # Timestamp is on the hour, no processing needed, just extract the values
            climateds_now = climateds.sel(time=time)
            # Bin-dependent variables indexed by bin_idx
            self.tempC = climateds_now['bin_temp'].to_numpy()[bin_idx]
            self.tp = climateds_now['bin_tp'].to_numpy()[bin_idx]
            self.sp = climateds_now['bin_sp'].to_numpy()[bin_idx]
            self.rH = climateds_now['bin_rh'].to_numpy()[bin_idx]
            # Elevation-invariant variables
            self.wind = climateds_now['wind'].to_numpy()
            self.tcc = climateds_now['tcc'].to_numpy()
            self.SWin = climateds_now['SWin'].to_numpy()
            self.SWout = climateds_now['SWout'].to_numpy()
            self.LWin = climateds_now['LWin'].to_numpy()
            self.LWout = climateds_now['LWout'].to_numpy()
        else:
            # Timestep is between hours, so interpolate using interpClimate function
            # Bin-dependent variables indexed by bin_idx
            self.tempC = self.interpClimate(climateds,time,'bin_temp',bin_idx)
            self.tp = self.interpClimate(climateds,time,'bin_tp',bin_idx)
            self.sp = self.interpClimate(climateds,time,'bin_sp',bin_idx)
            self.rH = self.interpClimate(climateds,time,'bin_rh',bin_idx)
            # Elevation-invariant variables
            self.wind = self.interpClimate(climateds,time,'wind')
            self.tcc = self.interpClimate(climateds,time,'tcc')
            self.SWin = self.interpClimate(climateds,time,'SWin')
            self.SWout = self.interpClimate(climateds,time,'SWout')
            self.LWin = self.interpClimate(climateds,time,'LWin')
            self.LWout = self.interpClimate(climateds,time,'LWout')
        # Define additional useful values
        self.tempK = self.tempC + 273.15
        self.prec =  self.tp / 3600     # tp is hourly total precip, so prec is the rate in m/s
        self.dt = dt
        return

    def surfaceEB(self,surftemp,layers,albedo,days_since_snowfall,mode='sum'):
        """
        Calculates the surface heat fluxes for the current timestep.

        Parameters
        ----------
        surftemp : float
            Temperature of the surface snow in Celsius
        layers
            class object from pygem_eb.layers
        albedo : float
            Albedo of the surface
        days_since_snowfall : int
            Number of days since fresh snowfall
        method_turbulent : str, default: 'MO-similarity'
            'MO-similarity', 'bulk-aerodynamic', or 'Richardson': determines method for calculating 
            turbulent fluxes
        mode : str, default: 'sum'
            'sum' to return sum of heat fluxes, 'list' to return separate fluxes,
            'optim' to return absolute value of heat fluxes (necessary for BFGS optimization)

        Returns
        -------
        Qm : float OR np.ndarray
            If mode is 'sum' or 'optim', returns the sum of heat fluxes
            If mode is 'list', returns list in the order of SWin,SWout,LWin,LWout,rain,sensible,latent
        """
        # SHORTWAVE RADIATION  (Snet_surf)
        SWin,SWout = self.getSW(albedo)
        Snet_surf = SWin + SWout
        self.SWin = SWin
        self.SWout = SWout
                    
        # LONGWAVE RADIATION (Lnet)
        LWin,LWout = self.getLW(surftemp)
        Lnet = LWin + LWout
        self.LWin = LWin
        self.LWout = LWout

        # RAIN FLUX (Qp)
        Qp = self.getRain(surftemp)
        self.rain = Qp

        # TURBULENT FLUXES (Qs and Ql)
        roughness = self.roughness_length(days_since_snowfall,layers.ltype)
        if eb_prms.method_turbulent in ['MO-similarity']:
            Qs, Ql = self.getTurbulentMO(surftemp,roughness)
        else:
            print('Only MO similarity method is set up for turbulent fluxes')
            Qs, Ql = self.getTurbulentMO(surftemp,roughness)
        self.sens = Qs
        self.lat = Ql

        # OUTPUTS
        Qm = Snet_surf + Lnet + Qp + Qs + Ql

        if mode in ['sum']:
            return Qm
        elif mode in ['optim']:
            return np.abs(Qm)
        elif mode in ['list']:
            return np.array([SWin,SWout,LWin,LWout,Qp,Qs,Ql])
        else:
            assert 1==0, 'argument \'mode\' in function surfaceEB should be sum, list or optim'
    
    def getSW(self,albedo):
        """
        Simplest parameterization for shortwave radiation which just adjusts it by modeled albedo.
        Returns the shortwave surface flux and the penetrating shortwave with each layer.
        """
        # sun_pos = solar.get_position(time,glacier_table['CenLon'],glacier_table['CenLat'])
        SWin = self.SWin/self.dt
        if np.isnan(self.SWout):
            SWout = -self.SWin*albedo/self.dt #* (cos(theta))
        else:
            SWout = self.SWout

        return SWin,SWout

    def getLW(self,surftemp):
        """
        Scheme following Klok and Oerlemans (2002) for calculating net longwave radiation
        from the air temperature and cloud cover.
        """
        if np.isnan(self.LWout):
            surftempK = surftemp+273.15
            LWout = -eb_prms.sigma_SB*surftempK**4
        else:
            LWout = self.LWout
        
        if np.isnan(self.LWin):
            ezt = self.vapor_pressure(self.tempC)    # vapor pressure in hPa
            Ecs = .23+ .433*(ezt/self.tempK)**(1/8)  # clear-sky emissivity
            Ecl = 0.984                         # cloud emissivity, Klok and Oerlemans, 2002
            Esky = Ecs*(1-self.tcc**2)+Ecl*self.tcc**2    # sky emissivity
            LWin = eb_prms.sigma_SB*(Esky*self.tempK**4)
        else:
            LWin = self.LWin
        return LWin,LWout
    
    def getRain(self,surftemp):
        """
        Calculates amount of energy supplied by precipitation that falls as rain.
        """
        is_rain = self.tempC > eb_prms.tsnow_threshold
        Qp = is_rain*eb_prms.Cp_water*(self.tempC-surftemp)*self.prec*eb_prms.density_water
        return Qp

    def getTurbulentMO(self,surf_temp,roughness):
        """
        Calculates turbulent fluxes (sensible and latent heat) based on Monin-Obukhov Similarity 
        Theory, requiring iteration.

        Parameters
        ----------
        surf_temp : float
            Surface temperature of snowpack/ice [C]
        roughness : float
            Surface roughness [m]
        """
        chi = lambda zeta: abs(1-16*zeta)**(1/4)
        PsiM = lambda zeta: np.piecewise(zeta,[zeta<0,(zeta>=0)&(zeta<=1),zeta>1],
                            [2*np.log((1+chi(zeta))/2)+np.log((1+chi(zeta)**2)/2)-2*np.arctan(chi(zeta))+np.pi/2,
                            -5*zeta, -4*(1+np.log(zeta))-zeta])
        PsiT = lambda zeta: np.piecewise(zeta,[zeta<0,(zeta>=0)&(zeta<=1),zeta>1],
                            [np.log((1+chi(zeta)**2)/2), -5*zeta, -4*(1+np.log(zeta))-zeta])
        
        Qs = 1000 #initial guess
        Ql = 1000
        converged = False
        zeta = 0.1
        count_iters = 0
        karman = eb_prms.karman
        while not converged:
            previous_zeta = zeta
            z = 2 #reference height, 2m

            Lv = np.piecewise(Ql,[(Ql>0)&(surf_temp<=0),(Ql>0)&(surf_temp>0),Ql<0],
                            [eb_prms.Lv_evap,eb_prms.Lv_sub,eb_prms.Lv_sub])

            z0 = roughness
            z0t = z0/100    # Roughness length for sensible heat
            z0q = z0/10     # Roughness length for moisture

            # calculate friction velocity using previous heat flux to get Obukhov length (L)
            fric_vel = karman*self.wind/(np.log(z/z0)-PsiM(zeta))
            air_dens = self.sp/eb_prms.R_gas/self.tempK*eb_prms.molarmass_air
            L = fric_vel**3*(self.tempC+273.15)*air_dens*eb_prms.Cp_air/(karman*eb_prms.gravity*Qs)
            if L<0.3: # DEBAM uses this correction to ensure it isn't over stablizied
                L = 0.3
            zeta = z/L
                
            cD = karman**2/(np.log(z/z0)-PsiM(zeta)-PsiM(z0/L))**2
            cH = karman*cD**(1/2)/((np.log(z/z0t)-PsiT(zeta)-PsiT(z0t/L)))
            cE = karman*cD**(1/2)/((np.log(z/z0q)-PsiT(zeta)-PsiT(z0q/L)))
            Qs = air_dens*eb_prms.Cp_air*cH*self.wind*(self.tempC-surf_temp)

            Ewz = self.vapor_pressure(self.tempC)  # vapor pressure at 2m
            Ew0 = self.vapor_pressure(surf_temp) # vapor pressure at the surface
            qz = (self.rH/100)*0.622*(Ewz/(self.sp-Ewz))
            q0 = 1.0*0.622*(Ew0/(self.sp-Ew0))
            # qz = (rH2 * 0.622 * (Ew / (p - Ew))) / 100.0
            # q0 = (100.0 * 0.622 * (Ew0 / (p - Ew0))) / 100.0
            Ql = air_dens*Lv*cE*self.wind*(qz-q0)

            count_iters += 1
            if count_iters > 10 or abs(previous_zeta - zeta) < .1:
                converged = True

        return Qs, Ql
    
    def roughness_length(self,days_since_snowfall,layertype):
        """
        Function to determine the roughness length of the surface. This assumes the roughness of snow
        linearly degrades with time in 60 days from that of fresh snow to firn.

        Parameters
        ----------
        days_since_snowfall : int
            Number of days since fresh snow occurred
        layertype : np.ndarray
            List of layer types to determine surface roughness
        """
        roughness_fresh_snow = 0.24                     # surface roughness length for fresh snow [mm] (Moelg et al. 2012, TC)
        roughness_ice = 1.7                             # surface roughness length for ice [mm] (Moelg et al. 2012, TC)
        roughness_firn = 4.0                            # surface roughness length for aged snow [mm] (Moelg et al. 2012, TC)
        aging_factor_roughness = 0.06267                # effect of ageing on roughness length: 60 days from 0.24 to 4.0 => 0.06267

        if layertype[0] in ['snow']:
            sigma = min(roughness_fresh_snow + aging_factor_roughness * days_since_snowfall, roughness_firn)
        elif layertype[0] in ['firn']:
            sigma = roughness_firn
        elif layertype[0] in ['ice']:
            sigma = roughness_ice
        return sigma/1000
    
    def vapor_pressure(self,T,method = 'ARM'):
        if method in ['ARM']:
            P = 0.61094*np.exp(17.625*T/(T+243.04)) # kPa
        return P*1000
 
    def interpClimate(self,climateds,time,varname,bin_idx=-1):
        """
        Interpolates climate variables from the hourly dataset to get sub-hourly data.

        Parameters
        ----------
        climateds : xr.Dataset
            Climate dataset containing temperature, precipitation, pressure, wind speed,
            shortwave radiation, and total cloud cover.
        time : datetime
            Timestamp to interpolate the climate variable.
        varname : str
            Variable name of variable in climateds
        bin_idx : int, default = -1
            Index number of the bin being run. Unspecified for running a variable that is elevation-independent.
        """
        climate_before = self.climateds.sel(time=time.floor('H'))
        climate_after = self.climateds.sel(time=time.ceil('H'))
        if bin_idx == -1:
            before = climate_before[varname].to_numpy()
            after = climate_after[varname].to_numpy()
        else:
            before = climate_before[varname].to_numpy()[bin_idx]
            after = climate_after[varname].to_numpy()[bin_idx]
        return before+(after-before)*(time.minute/60)
    