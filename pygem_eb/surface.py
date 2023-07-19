import numpy as np
import pygem_eb.input as eb_prms
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

        self.snow_timestamp = time[0] # ***** what to set snow timestamp at when we start on bare ice???
        return
    
    def updateSurface(self):
        """
        Run every timestep to get properties that evolve with time. Keeps track of past surface in the case of fresh snowfall
        after significant melt.
        """
        self.getGrainSize()
        return
    
    def getSurfTemp(self,enbal,layers):
        Qm_check = enbal.surfaceEB(0,layers,self.albedo,self.days_since_snowfall)
        # If Qm is positive with a surface temperature of 0, the surface is either melting or warming to the melting point.
        # If Qm is negative with a surface temperature of 0, the surface temperature needs to be lowered to cool the snowpack.
        cooling = True if Qm_check < 0 else False
        if not cooling:
            # Energy toward the surface: either melting or top layer is heated to melting point
            self.temp = 0
            Qm = Qm_check
            if layers.ltemp[0] < 0: # need to heat surface layer to 0 before it can start melting
                layers.ltemp[0] += Qm_check*eb_prms.dt/(eb_prms.Cp_ice*layers.ldrymass[0])
                if layers.ltemp[0] > 0:
                    # if temperature rises above zero, leave excess energy in Qm
                    Qm = layers.ltemp[0]*eb_prms.Cp_ice*layers.ldrymass[0]/eb_prms.dt
                    layers.ltemp[0] = 0
                else:
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
                    Qm_check = enbal.surfaceEB(self.temp,layers,self.albedo,self.days_since_snowfall)
                    if Qm_check > 0.5:
                        self.temp += 0.25
                    elif Qm_check < -0.5:
                        self.temp -= 0.25
                    self.temp = max(-60,self.temp)
                    n_iters += 1
                    if abs(Qm_check) < 0.5 or n_iters > 10:
                        if self.temp == -60:
                            result = minimize(enbal.surfaceEB,-50,method='L-BFGS-B',bounds=((-60,0),),tol=1e-3,
                                args=(layers,self.albedo,self.days_since_snowfall,'optim'))
                            if result.x > -60:
                                self.temp = result.x[0]
                        Qm = 0
                        loop = False
        self.Qm = Qm
        return

    def getAlbedo(self):
        self.albedo = 0.85
        return 

    def getGrainSize(self):
        return 0
    
    def updatePrecip(self,type,amount):
        if type in 'snow' and amount > 1e-8:
            self.albedo = 0.85
        elif type in 'rain':
            self.albedo = 0.85
        return

