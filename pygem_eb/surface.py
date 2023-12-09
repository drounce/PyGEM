import pygem_eb.input as eb_prms
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import sys, os
sys.path.append('/home/claire/research/PyGEM-EB/biosnicar-py/src/')
import yaml

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
    
    def updateSurfaceDaily(self,layers,time,args):
        """
        Run every timestep to get properties that evolve with time. Keeps track of past surface in the case of fresh snowfall
        after significant melt.
        """
        self.type = layers.ltype[0]
        if args.switch_melt == 2 and layers.nlayers > 2:
            layers.getGrainSize(self.temp)
        self.days_since_snowfall = (time - self.snow_timestamp)/pd.Timedelta(days=1)
        return
    
    def getSurfTemp(self,enbal,layers):
        """
        Solves energy balance equation for surface temperature. There are three cases:
        (1) LWout data is input - surftemp is derived from data
        (2) Qm is positive with surftemp = 0. - excess Qm is used to warm layers to melting point or melt layers
        (3) Qm is negative with surftemp = 0. - snowpack is cooling and surftemp is lowered to balance Qm
                Two methods are available (specified in input.py): fast iterative, or slow minimization
        """
        if not enbal.nanLWout:
            # CASE (1)
            self.temp = np.power(np.abs(enbal.LWout_ds/eb_prms.sigma_SB),1/4)
            Qm = enbal.surfaceEB(self.temp,layers,self.albedo,self.days_since_snowfall)
        else:
            Qm_check = enbal.surfaceEB(0,layers,self.albedo,self.days_since_snowfall)
            # If Qm is positive with a surface temperature of 0, the surface is either melting or warming to the melting point.
            # If Qm is negative with a surface temperature of 0, the surface temperature needs to be lowered to cool the snowpack.
            cooling = True if Qm_check < 0 else False
            if not cooling:
                # CASE (2): Energy toward the surface: either melting
                #            or top layer is heated to melting point
                self.temp = 0
                Qm = Qm_check
                if layers.ltemp[0] < 0: # need to heat surface layer to 0 before it can start melting
                    Qm_check = enbal.surfaceEB(self.temp,layers,self.albedo,self.days_since_snowfall)
                    temp_change = Qm_check*eb_prms.dt/(eb_prms.Cp_ice*layers.ldrymass[0])
                    layers.ltemp[0] += temp_change
                    if layers.ltemp[0] > 0:
                        # if temperature rises above zero, leave excess energy in the next layer down
                        Qm = layers.ltemp[0]*eb_prms.Cp_ice*layers.ldrymass[0]/eb_prms.dt
                        layers.ltemp[0] = 0
                    elif len(layers.ltype) < 2:
                        self.temp = 0
                        Qm = Qm_check
                    else:
                        # *** This might be a problem area with how surface temperature is being treated
                        Qm = 0
            elif cooling:
                # CASE (3) Energy away from surface: need to change 
                #          surface temperature to get 0 surface energy flux 
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

    def getAlbedo(self,layers,args):
        sfi_albedo = {'snow':eb_prms.albedo_fresh_snow,'firn':eb_prms.albedo_firn,
                               'ice':eb_prms.albedo_ice}
        if args.switch_melt == 0:
            if args.switch_LAPs == 0:
                # SURFACE TYPE ONLY
                self.albedo = sfi_albedo[self.type]
            elif args.switch_LAPs == 1:
                # LAPs ON, GRAIN SIZE OFF
                self.albedo = self.runSNICAR(layers,override_grainsize=True)
        elif args.switch_melt == 1:
            # BASIC DEGRADATION RATE
            if self.type == 'snow':
                snow_albedo = eb_prms.albedo_firn+(eb_prms.albedo_fresh_snow - eb_prms.albedo_firn)*(np.exp(-self.days_since_snowfall/eb_prms.albedo_deg_rate))
                self.albedo = max(snow_albedo,eb_prms.albedo_firn)
            else:
                self.albedo = sfi_albedo[self.type]
        elif args.switch_melt == 2:
            if args.switch_LAPs == 0:
                # LAPs OFF, GRAIN SIZE ON
                self.albedo = self.runSNICAR(layers,override_LAPs=True)
            elif args.switch_LAPs == 1:
                # LAPs ON, GRAIN SIZE ON
                self.albedo = self.runSNICAR(layers)
            
        return 
    
    def runSNICAR(self,layers,n_layers=None,max_depth=None,override_grainsize=None,override_LAPs=None):
        """
        Runs SNICAR model to retrieve broadband albedo. 

        Parameters
        ----------
        layers
            class object from pygem_eb.layers.py
        n_layers : int
            Number of layers to include in the calculation
            * Specify n_layers OR max_depth *
        max_depth : float
            Maximum depth of layers to include in the calculation
        Returns
        -------
        layermelt : np.ndarray
            Array containing subsurface melt amounts [kg m-2]
        """
        import biosnicar as snicar

        # Get layers to include in the calculation
        assert not n_layers and not max_depth, "Specify one of n_layers or max_depth in runSNICAR"
        if not n_layers and max_depth:
            n_layers = np.where(layers.ldepth > max_depth)[0][0] + 1
        elif n_layers and not max_depth:
            n_layers = min(layers.nlayers,n_layers)
        elif not n_layers and not max_depth:
            # Default case if neither is specified: only includes top 1m
            n_layers = np.where(layers.ldepth > 1)[0][0] + 1
        idx = np.arange(n_layers)

        # Unpack layer variables (need to be stored as lists)
        lheight = layers.lheight[idx].astype(float).tolist()
        ldensity = layers.ldensity[idx].astype(float).tolist()
        lgrainsize = layers.grainsize[idx].astype(int)
        # Grain size files are every 1um till 1500um, then every 500
        lgrainsize[np.where(lgrainsize>1500)[0]] = np.round(lgrainsize[np.where(lgrainsize>1500)[0]]/500) * 500
        lgrainsize = lgrainsize.tolist()
        if override_grainsize:
            lgrainsize = [eb_prms.constant_grainsize for i in idx]
        lBC = (layers.lBC[idx].astype(float) * 1e6).tolist()
        ldust = layers.ldust[idx].astype(float) * 1e6
        if override_LAPs:
            lBC = [eb_prms.BC_freshsnow*1e6 for _ in idx]
            ldust = np.array([eb_prms.dust_freshsnow*1e6 for _ in idx])

        # Open and edit yaml input file for SNICAR
        with open(eb_prms.snicar_input_fp) as f:
            list_doc = yaml.safe_load(f)

        # Update changing layer variables
        list_doc['IMPURITIES']['BC']['CONC'] = lBC
        list_doc['IMPURITIES']['DUST1']['CONC'] = (ldust * 0.1).tolist()
        list_doc['IMPURITIES']['DUST2']['CONC'] = (ldust * 0.2).tolist()
        list_doc['IMPURITIES']['DUST3']['CONC'] = (ldust * 0.4).tolist()
        list_doc['IMPURITIES']['DUST4']['CONC'] = (ldust * 0.2).tolist()
        list_doc['IMPURITIES']['DUST5']['CONC'] = (ldust * 0.1).tolist()
        list_doc['ICE']['DZ'] = lheight
        list_doc['ICE']['RHO'] = ldensity
        list_doc['ICE']['RDS'] = lgrainsize

        # Following variables are set to constants, but need to have right number of layers
        ice_variables = ['LAYER_TYPE','SHP','HEX_SIDE','HEX_LENGTH','SHP_FCTR','WATER','AR','CDOM']
        for var in ice_variables:
            list_doc['ICE'][var] = [list_doc['ICE'][var][0]] * n_layers

        # Save SNICAR input file
        with open(eb_prms.snicar_input_fp, 'w') as f:
            yaml.dump(list_doc,f)
        
        # Get albedo from biosnicar "main.py"
        with HiddenPrints():
            self.albedo = snicar.main.get_albedo('adding-doubling',plot=False,validate=False)
        # print('Albedo',self.albedo,'Mean BC',np.mean(lBC))
        # bba = np.sum(illumination.flx_slr * albedo) / np.sum(illumination.flx_slr)
        return self.albedo
    
    def updatePrecip(self,type,amount):
        if type in 'snow' and amount > 1e-8:
            self.albedo = 0.85
        elif type in 'rain':
            _=0
            # self.albedo = 0.35
        return
    
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self,exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
        return