import pygem_eb.input as eb_prms
import numpy as np
from scipy.optimize import minimize

class Surface():
    """
    Surface scheme that tracks the accumulation of LAPs and calculates albedo based on several switches.
    """ 
    def __init__(self,layers,time):
        # Set initial albedo based on surface type
        if layers.ltype[0] in ['snow']:
            self.albedo = eb_prms.albedo_fresh_snow
        elif layers.ltype[0] in ['firn']:
            self.albedo = eb_prms.albedo_firn
        elif layers.ltype[0] in ['ice']:
            self.albedo = eb_prms.albedo_ice

        # Initialize BC, dust, grain_size, etc.
        self.BC = 0
        self.dust = 0
        self.grain_size = 0
        self.temp = eb_prms.surftemp_guess
        self.Qm = 0
        self.days_since_snowfall = 0
        self.type = layers.ltype[0]
        self.snow_timestamp = time[0]
        return
    
    def updateSurface(self,layers):
        """
        Run every timestep to get properties that evolve with time. Keeps track of past surface in the case of fresh snowfall
        after significant melt.
        """
        self.type = layers.ltype[0]
        self.getGrainSize()
        self.getAlbedo()
        return
    
    def getSurfTemp(self,enbal,layers):
        """
        Solves energy balance equation for surface temperature. There are three cases:
        1) LWout data is input - surftemp is derived from data
        2) Qm is positive with surftemp = 0. - excess Qm is used to warm layers to melting point or melt layers
        3) Qm is negative with surftemp = 0. - snowpack is cooling and surftemp is lowered to balance Qm
                Two methods are available (specified in input.py): fast iterative, or slow minimization
        """
        # if not enbal.nanLWout:
        #     self.temp = np.power(np.abs(enbal.LWout_ds/eb_prms.sigma_SB),1/4)
        #     Qm = enbal.surfaceEB(self.temp,layers,self.albedo,self.days_since_snowfall)
        # else:
        if True:
            Qm_check = enbal.surfaceEB(0,layers,self.albedo,self.days_since_snowfall)
            # If Qm is positive with a surface temperature of 0, the surface is either melting or warming to the melting point.
            # If Qm is negative with a surface temperature of 0, the surface temperature needs to be lowered to cool the snowpack.
            cooling = True if Qm_check < 0 else False
            if not cooling:
                # Energy toward the surface: either melting or top layer is heated to melting point
                self.temp = 0
                Qm = Qm_check
                if layers.ltemp[0] < 0: # need to heat surface layer to 0 before it can start melting
                    Qm_check = enbal.surfaceEB(self.temp,layers,self.albedo,self.days_since_snowfall)
                    layers.ltemp[0] += Qm_check*eb_prms.dt/(eb_prms.Cp_ice*layers.ldrymass[0])
                    if layers.ltemp[0] > 0:
                        # if temperature rises above zero, leave excess energy in Qm
                        Qm = layers.ltemp[0]*eb_prms.Cp_ice*layers.ldrymass[0]/eb_prms.dt
                        layers.ltemp[0] = 0
                    elif len(layers.ltype) < 2:
                        self.temp = 0
                        Qm = Qm_check
                    else:
                        # *** This might be a problem area with how surface temperature is being treated
                        Qm = 0
            elif cooling:
                # Energy away from surface: need to change surface temperature to get 0 surface energy flux 
                if eb_prms.method_cooling in ['minimize']:
                    result = minimize(enbal.surfaceEB,self.temp,method='L-BFGS-B',bounds=((-60,0),),tol=1e-3,
                                    args=(layers,self.albedo,self.days_since_snowfall,'optim'))
                    Qm = enbal.surfaceEB(result.x[0],layers,self.albedo,self.days_since_snowfall)
                    if not result.success and abs(Qm) > 10:
                        print('Unsuccessful minimization, Qm = ',Qm)
                        # assert 1==0, 'Surface temperature was not lowered enough by minimization'
                    else:
                        self.temp = result.x[0]
                elif eb_prms.method_cooling in ['iterative']:
                    loop = True
                    n_iters = 0
                    while loop:
                        n_iters += 1
                        # Initial check of Qm comparing to previous surftemp
                        Qm_check = enbal.surfaceEB(self.temp,layers,self.albedo,self.days_since_snowfall)
                        # Check direction of flux at that temperature and adjust
                        if Qm_check > 0.5:
                            self.temp += 0.25
                        elif Qm_check < -0.5:
                            self.temp -= 0.25
                        # surftemp cannot go below -60
                        self.temp = max(-60,self.temp)

                        # break loop if Qm is ~0 or after 10 iterations
                        if abs(Qm_check) < 0.5 or n_iters > 10:
                            # if temp is bottoming out at -60, re-solve minimization
                            if self.temp == -60:
                                result = minimize(enbal.surfaceEB,-50,method='L-BFGS-B',bounds=((-60,0),),tol=1e-3,
                                    args=(layers,self.albedo,self.days_since_snowfall,'optim'))
                                if result.x > -60:
                                    self.temp = result.x[0]
                            # set melt to 0 and break loop
                            Qm = 0
                            loop = False
                Qm = 0

        # Update surface balance terms with new surftemp
        enbal.surfaceEB(self.temp,layers,self.albedo,self.days_since_snowfall)
        self.Qm = Qm
        return
    
    def storeSurface(self):
        previous = self.days_since_snowfall
        # need to keep track of the layer that used to be the surface such that if the snowfall melts, albedo resets to the dirty surface

    def getAlbedo(self):
        if self.type == 'snow':
            if eb_prms.switch_snow == 0:
                self.albedo = eb_prms.albedo_fresh_snow
            else:
                snow_albedo = eb_prms.albedo_firn+(eb_prms.albedo_fresh_snow - eb_prms.albedo_firn)*(np.exp(-self.days_since_snowfall/eb_prms.albedo_deg_rate))
                self.albedo = max(snow_albedo,eb_prms.albedo_firn)
        elif self.type == 'firn':
            self.albedo = eb_prms.albedo_firn
        elif self.type == 'ice':
            self.albedo = eb_prms.albedo_ice
        return 

    def getGrainSize(self):
        return 0
    
    def updatePrecip(self,type,amount):
        if type in 'snow' and amount > 1e-8:
            self.albedo = 0.85
        elif type in 'rain':
            _=0
            # self.albedo = 0.35
        return

