"""
Energy balance class for PyGEM Energy Balance

@author: cvwilson
"""

import numpy as np
import pygem_eb.input as eb_prms
# import warnings
# warnings.filterwarnings("error")

class energyBalance():
    """
    Energy balance scheme that calculates the surface energy balance
    and penetrating shortwave radiation. This class is updated within
    MassBalance every timestep, so it stores the current climate data and 
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
        if time.minute == 0 or time.minute == 30: #******
            # Timestamp is on the hour, no processing needed, just extract the values
            climateds_now = climateds.sel(time=time)
            # Bin-dependent variables indexed by bin_idx
            self.tempC = climateds_now['bin_temp'].to_numpy()[bin_idx]
            self.tp = climateds_now['bin_tp'].to_numpy()[bin_idx]
            self.sp = climateds_now['bin_sp'].to_numpy()[bin_idx]
            # Elevation-invariant variables
            self.rH = climateds_now['rh'].to_numpy()
            self.wind = climateds_now['wind'].to_numpy()
            self.tcc = climateds_now['tcc'].to_numpy()
            self.SWin_ds = climateds_now['SWin'].to_numpy()
            self.SWout_ds = climateds_now['SWout'].to_numpy()
            self.LWin_ds = climateds_now['LWin'].to_numpy()
            self.LWout_ds = climateds_now['LWout'].to_numpy()
            self.NR_ds = climateds_now['NR'].to_numpy()
            self.bcdry = climateds_now['bcdry'].to_numpy()
            self.bcwet = climateds_now['bcwet'].to_numpy()
            self.dustdry = climateds_now['dustdry'].to_numpy()
            self.dustwet = climateds_now['dustwet'].to_numpy()
        else: # DONT NEED THIS UNLESS I EVER DO SUBHOURLY RUNS
            # Timestep is between hours, so interpolate using interpClimate function
            # Bin-dependent variables indexed by bin_idx
            self.tempC = self.interpClimate(climateds,time,'bin_temp',bin_idx)
            self.tp = self.interpClimate(climateds,time,'bin_tp',bin_idx)
            self.sp = self.interpClimate(climateds,time,'bin_sp',bin_idx)
            self.rH = self.interpClimate(climateds,time,'bin_rh',bin_idx)
            # Elevation-invariant variables
            self.wind = self.interpClimate(climateds,time,'wind')
            self.tcc = self.interpClimate(climateds,time,'tcc')
            self.SWin_ds = self.interpClimate(climateds,time,'SWin')
            self.SWout_ds = self.interpClimate(climateds,time,'SWout')
            self.LWin_ds = self.interpClimate(climateds,time,'LWin')
            self.LWout_ds = self.interpClimate(climateds,time,'LWout')
        # Define additional useful values
        self.tempK = self.tempC + 273.15
        self.prec =  self.tp / 3600     # tp is hourly total precip, prec is the rate in m/s
        self.dt = dt
        self.climateds = climateds

        self.nanLWin = True if np.isnan(self.LWin_ds) else False
        self.nanSWout = True if np.isnan(self.SWout_ds) else False
        self.nanLWout = True if np.isnan(self.LWout_ds) else False
        self.nanNR = True if np.isnan(self.NR_ds) else False
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
        method_turbulent : str
            'MO-similarity', 'bulk-aerodynamic', or 'Richardson': 
            determines method for calculating turbulent fluxes
        mode : str, default: 'sum'
            'sum' to return sum of heat fluxes, 'list' to return separate fluxes,
            'optim' to return absolute value of heat fluxes (needed for BFGS optimization)

        Returns
        -------
        Qm : float OR np.ndarray
            If mode is 'sum' or 'optim', returns the sum of heat fluxes
            If mode is 'list', returns list in the order of 
                    [SWin, SWout, LWin, LWout, sensible, latent, rain, ground]
        """
        # SHORTWAVE RADIATION  (Snet)
        SWin,SWout = self.getSW(albedo)
        Snet_surf = SWin + SWout
        self.SWin = SWin
        self.SWout = SWout[0] if '__iter__' in dir(SWout) else SWout
                    
        # LONGWAVE RADIATION (Lnet)
        LWin,LWout = self.getLW(surftemp)
        Lnet = LWin + LWout
        self.LWin = LWin
        self.LWout = LWout[0] if '__iter__' in dir(LWout) else LWout

        # RAIN FLUX (Qp)
        Qp = self.getRain(surftemp)
        self.rain = Qp[0] if '__iter__' in dir(Qp) else Qp

        # GROUND FLUX (Qg)
        Qg = self.getGround(surftemp)
        self.ground = Qg[0] if '__iter__' in dir(Qg) else Qg

        # TURBULENT FLUXES (Qs and Ql)
        roughness = self.getRoughnessLength(days_since_snowfall,layers.ltype)
        if eb_prms.method_turbulent in ['MO-similarity']:
            Qs, Ql = self.getTurbulentMO(surftemp,roughness)
        else:
            print('Only MO similarity method is set up for turbulent fluxes')
            Qs, Ql = self.getTurbulentMO(surftemp,roughness)
        self.sens = Qs[0] if '__iter__' in dir(Qs) else Qs
        self.lat = Ql[0] if '__iter__' in dir(Qs) else Ql

        # OUTPUTS
        Qm = Snet_surf + Lnet + Qp + Qs + Ql + Qg

        if mode in ['sum']:
            return Qm
        elif mode in ['optim']:
            return np.abs(Qm)
        elif mode in ['list']:
            return np.array([SWin,SWout,LWin,LWout,Qs,Ql,Qp,Qg])
        else:
            assert 1==0, 'argument \'mode\' in function surfaceEB should be sum, list or optim'
    
    def getSW(self,surface):
        """
        Simplest parameterization for shortwave radiation which
        adjusts it by modeled spectral albedo. Returns the shortwave
        surface flux and the penetrating shortwave for each layer.
        
        Parameters
        ----------
        surface
            class object from surface.py
        """
        albedo = surface.albedo
        spectral_weights = surface.spectral_weights
        assert np.abs(1-np.sum(spectral_weights)) < 1e-5, 'Solar weights dont sum to 1'
        SWin = self.SWin_ds/self.dt
        if self.nanSWout:
            SWout = -np.sum(SWin*spectral_weights*albedo)
        else:
            SWout = -self.SWout_ds/self.dt

        return SWin,SWout

    def getLW(self,surftemp):
        """
        If not input in climate data, scheme follows Klok and 
        Oerlemans (2002) for calculating net longwave radiation 
        from the air temperature and cloud cover.
        
        Parameters
        ----------
        surftemp : float
            Surface temperature of snowpack/ice [C]
        """
        if self.nanLWout:
            surftempK = surftemp+273.15
            LWout = -eb_prms.sigma_SB*surftempK**4
        else:
            LWout = -self.LWout_ds/self.dt
        
        if self.nanLWin and self.nanNR:
            ezt = self.vapor_pressure(self.tempC)    # vapor pressure in hPa
            Ecs = .23+ .433*(ezt/self.tempK)**(1/8)  # clear-sky emissivity
            Ecl = 0.984               # cloud emissivity, Klok and Oerlemans, 2002
            Esky = Ecs*(1-self.tcc**2)+Ecl*self.tcc**2    # sky emissivity
            LWin = eb_prms.sigma_SB*(Esky*self.tempK**4)
        elif not self.nanLWin:
            LWin = self.LWin_ds/self.dt
        elif not self.nanNR:
            LWin = self.NR_ds/self.dt - LWout - self.SWin - self.SWout
            
        return LWin,LWout
    
    def getRain(self,surftemp):
        """
        Calculates amount of energy supplied by precipitation that falls as rain.
        
        Parameters
        ----------
        surftemp : float
            Surface temperature of snowpack/ice [C]
        """
        # CONSTANTS
        TEMP_THRESHOLD = eb_prms.tsnow_threshold
        DENSITY_WATER = eb_prms.density_water
        CP_WATER = eb_prms.Cp_water
        
        is_rain = self.tempC > TEMP_THRESHOLD
        Qp = is_rain*(self.tempC-surftemp)*self.prec*DENSITY_WATER*CP_WATER
        return Qp
    
    def getGround(self,surftemp):
        """
        Calculates amount of energy supplied by heat conduction from the temperate ice.
        
        Parameters
        ----------
        surftemp : float
            Surface temperature of snowpack/ice [C]
        """
        if eb_prms.method_ground in ['MolgHardy']:
            Qg = -eb_prms.k_ice * (surftemp - eb_prms.temp_temp) / eb_prms.temp_depth
        else:
            assert 1==0, 'Ground flux method not accepted; choose from [\'MolgHardy\']'
        return Qg
    
    def getTurbulentMO(self,surftemp,roughness):
        """
        Calculates turbulent fluxes (sensible and latent heat) based on 
        Monin-Obukhov Similarity Theory, requiring iteration.

        Parameters
        ----------
        surftemp : float
            Surface temperature of snowpack/ice [C]
        roughness : float
            Surface roughness [m]
        """
        # CONSTANTS
        KARMAN = eb_prms.karman
        GRAVITY = eb_prms.gravity
        R_GAS = eb_prms.R_gas
        MM_AIR = eb_prms.molarmass_air
        CP_AIR = eb_prms.Cp_air

        # ROUGHNESS LENGTHS
        z0 = roughness
        z0t = z0/100    # Roughness length for sensible heat
        z0q = z0/10     # Roughness length for moisture

        # UNIVERSAL FUNCTIONS
        chi = lambda zeta: abs(1-16*zeta)**(1/4)
        PsiM = lambda zeta: np.piecewise(zeta,[zeta<0,(zeta>=0)&(zeta<=1),zeta>1],
                            [2*np.log((1+chi(zeta))/2)+np.log((1+chi(zeta)**2)/2)-2*np.arctan(chi(zeta))+np.pi/2,
                            -5*zeta, -4*(1+np.log(zeta))-zeta])
        PsiT = lambda zeta: np.piecewise(zeta,[zeta<0,(zeta>=0)&(zeta<=1),zeta>1],
                            [np.log((1+chi(zeta)**2)/2), -5*zeta, -4*(1+np.log(zeta))-zeta])
        
        # ADJUST WIND SPEED
        z = 2 # reference height in m
        if eb_prms.wind_ref_height != 2:
            wind *= np.log(2/roughness) / np.log(eb_prms.wind_ref_height/roughness)

        # Transform humidity into mixing ratio (q), get air density from PV=nRT
        Ewz = self.vapor_pressure(self.tempC)  # vapor pressure at 2m
        Ew0 = self.vapor_pressure(surftemp) # vapor pressure at the surface
        qz = (self.rH/100)*0.622*(Ewz/(self.sp-Ewz))
        q0 = 1.0*0.622*(Ew0/(self.sp-Ew0))
        density_air = self.sp/R_GAS/self.tempK*MM_AIR

        # INITIATE LOOP
        loop = True
        zeta = 1e-5 # initial guess, neutral stratification (close to 0 to avoid log issues)

        # Use initial guess to calculate coefficients and fluxes
        L = z / zeta
        cD = KARMAN**2/(np.log(z/z0)-PsiM(zeta)-PsiM(z0/L))**2
        cH = KARMAN*cD**(1/2)/((np.log(z/z0t)-PsiT(zeta)-PsiT(z0t/L)))
        cE = KARMAN*cD**(1/2)/((np.log(z/z0q)-PsiT(zeta)-PsiT(z0q/L)))
        Qs = density_air*CP_AIR*cH*self.wind*(self.tempC-surftemp)
        Ql = density_air*eb_prms.Lv_evap*cE*self.wind*(qz-q0)
        
        count_iters = 0
        while loop:
            previous_zeta = zeta

            if Ql > 0 and surftemp <=0:
                Lv = eb_prms.Lv_evap
            elif Ql > 0 and surftemp > 0:
                Lv = eb_prms.Lv_sub
            elif Ql <= 0:
                Lv = eb_prms.Lv_sub

            # Calculate friction velocity using previous heat flux to get Obukhov length (L)
            fric_vel = KARMAN*self.wind/(np.log(z/z0)-PsiM(zeta))
            Qs = 1e-5 if Qs == 0 else Qs # divide by 0 issue
            L = fric_vel**3*(self.tempC+273.15)*density_air*CP_AIR/(KARMAN*GRAVITY*Qs)
            L = max(L,0.3)  # DEBAM uses this correction to ensure it isn't over stablizied
            zeta = z/L
                
            # Calculate 
            cD = KARMAN**2/(np.log(z/z0)-PsiM(zeta)-PsiM(z0/L))**2
            cH = KARMAN*cD**(1/2)/((np.log(z/z0t)-PsiT(zeta)-PsiT(z0t/L)))
            cE = KARMAN*cD**(1/2)/((np.log(z/z0q)-PsiT(zeta)-PsiT(z0q/L)))

            # Calculate fluxes
            Qs = density_air*CP_AIR*cH*self.wind*(self.tempC-surftemp)
            Ql = density_air*Lv*cE*self.wind*(qz-q0)

            count_iters += 1
            if count_iters > 10 or abs(previous_zeta - zeta) < .1:
                loop = False

        return Qs, Ql
    
    def getDryDeposition(self, layers):
        if np.isnan(self.bcdry):
            self.bcdry = 1e-14 # kg m-2 s-1
        if np.isnan(self.dustdry):
            self.dustdry = 1e-13 # kg m-2 s-1

        layers.lBC[0] += self.bcdry * self.dt * eb_prms.ratio_BC2_BCtot
        layers.ldust[0] += self.dustdry * self.dt * eb_prms.ratio_DU3_DUtot
        return 
    
    def getRoughnessLength(self,days_since_snowfall,layertype):
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
        # CONSTANTS
        ROUGHNESS_FRESH_SNOW = eb_prms.roughness_fresh_snow
        ROUGHNESS_FIRN = eb_prms.roughness_firn
        ROUGHNESS_ICE = eb_prms.roughness_ice
        AGING_ROUGHNESS = eb_prms.aging_factor_roughness

        if layertype[0] in ['snow']:
            sigma = min(ROUGHNESS_FRESH_SNOW + AGING_ROUGHNESS * days_since_snowfall, ROUGHNESS_FIRN)
        elif layertype[0] in ['firn']:
            sigma = ROUGHNESS_FIRN
        elif layertype[0] in ['ice']:
            sigma = ROUGHNESS_ICE
        return sigma/1000
    
    def vapor_pressure(self,T,method = 'ARM'):
        """
        Returns vapor pressure in Pa from temperature in Celsius

        Parameters
        ----------
        T : float
            Temperature in C
        """
        if method in ['ARM']:
            P = 0.61094*np.exp(17.625*T/(T+243.04)) # kPa
        return P*1000
 
    def interpClimate(self,time,varname,bin_idx=-1):
        """
        Interpolates climate variables from the hourly dataset to get sub-hourly data.

        Parameters
        ----------
        time : datetime
            Timestamp to interpolate the climate variable.
        varname : str
            Variable name of variable in climateds
        bin_idx : int, default = -1
            Index number of the bin being run. Default -1 for a variable that is elevation-independent.
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
    