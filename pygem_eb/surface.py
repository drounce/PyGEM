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
    def __init__(self,layers,time,args):
        # Add args to surface class
        self.args = args

        # Set initial albedo based on surface type
        self.albedo_dict = {'snow':eb_prms.albedo_fresh_snow,'firn':eb_prms.albedo_firn,
                                    'ice':eb_prms.albedo_ice}
        self.stype = layers.ltype[0]
        self.albedo = self.albedo_dict[self.stype]

        # Initialize surface properties
        self.stemp = eb_prms.surftemp_guess
        self.days_since_snowfall = 0
        self.snow_timestamp = time[0]
        return
    
    def updateSurfaceDaily(self,layers,time):
        """
        Run every timestep to get albedo-related properties that evolve with time

        Parameters
        ----------
        layers
            class object from layers.py
        time : pd.Datetime
            current timestep
        """
        self.stype = layers.ltype[0]
        if self.args.switch_melt == 2 and layers.nlayers > 2:
            layers.getGrainSize(self.stemp)
        self.days_since_snowfall = (time - self.snow_timestamp)/pd.Timedelta(days=1)
        return
    
    def getSurfTemp(self,enbal,layers):
        """
        Solves energy balance equation for surface temperature. There are three cases:
        (1) LWout data is input - surftemp is derived from data
        (2) Qm is positive with surftemp = 0. - excess Qm is used to warm layers to melting point or melt layers
        (3) Qm is negative with surftemp = 0. - snowpack is cooling and surftemp is lowered to balance Qm
                Two methods are available (specified in input.py): fast iterative, or slow minimization
        
        Parameters
        ----------
        enbal
            class object from energybalance.py
        layers
            class object from layers.py
        """
        # CONSTANTS
        STEFAN_BOLTZMANN = eb_prms.sigma_SB
        HEAT_CAPACITY_ICE = eb_prms.Cp_ice
        dt = eb_prms.dt

        if not enbal.nanLWout:
            # CASE (1)
            self.stemp = np.power(np.abs(enbal.LWout_ds/STEFAN_BOLTZMANN),1/4)
            Qm = enbal.surfaceEB(self.stemp,layers,self.albedo,self.days_since_snowfall)
        else:
            Qm_check = enbal.surfaceEB(0,layers,self.albedo,self.days_since_snowfall)
            # If Qm is positive with a surface temperature of 0, the surface is either melting or warming to the melting point.
            # If Qm is negative with a surface temperature of 0, the surface temperature needs to be lowered to cool the snowpack.
            cooling = True if Qm_check < 0 else False
            if not cooling:
                # CASE (2): Energy toward the surface: either melting or top layer is heated to melting point
                self.stemp = 0
                Qm = Qm_check
                if layers.ltemp[0] < 0: # need to heat surface layer to 0 before it can start melting
                    Qm_check = enbal.surfaceEB(self.stemp,layers,self.albedo,self.days_since_snowfall)
                    temp_change = Qm_check*dt/(HEAT_CAPACITY_ICE*layers.ldrymass[0])
                    layers.ltemp[0] += temp_change
                    if layers.ltemp[0] > 0:
                        # if temperature rises above zero, leave excess energy in the next layer down
                        Qm = layers.ltemp[0]*HEAT_CAPACITY_ICE*layers.ldrymass[0]/dt
                        layers.ltemp[0] = 0
                    elif len(layers.ltype) < 2:
                        self.stemp = 0
                        Qm = Qm_check
                    else:
                        # *** This might be a problem area with how surface temperature is being treated
                        Qm = 0
            elif cooling:
                # CASE (3) Energy away from surface: need to change surface temperature to get 0 surface energy flux 
                if eb_prms.method_cooling in ['minimize']:
                    result = minimize(enbal.surfaceEB,self.stemp,method='L-BFGS-B',bounds=((-60,0),),tol=1e-3,
                                    args=(layers,self.albedo,self.days_since_snowfall,'optim'))
                    Qm = enbal.surfaceEB(result.x[0],layers,self.albedo,self.days_since_snowfall)
                    if not result.success and abs(Qm) > 10:
                        print('Unsuccessful minimization, Qm = ',Qm)
                    else:
                        self.stemp = result.x[0]

                elif eb_prms.method_cooling in ['iterative']:
                    loop = True
                    n_iters = 0
                    while loop:
                        n_iters += 1
                        # Initial check of Qm comparing to previous surftemp
                        Qm_check = enbal.surfaceEB(self.stemp,layers,self.albedo,self.days_since_snowfall)
                        # Check direction of flux at that temperature and adjust
                        if Qm_check > 0.5:
                            self.stemp += 0.25
                        elif Qm_check < -0.5:
                            self.stemp -= 0.25
                        # surftemp cannot go below -60
                        self.stemp = max(-60,self.stemp)

                        # break loop if Qm is ~0 or after 10 iterations
                        if abs(Qm_check) < 0.5 or n_iters > 10:
                            # if temp is still bottoming out at -60, resolve minimization
                            if self.stemp == -60:
                                result = minimize(enbal.surfaceEB,-50,method='L-BFGS-B',bounds=((-60,0),),tol=1e-3,
                                    args=(layers,self.albedo,self.days_since_snowfall,'optim'))
                                if result.x > -60:
                                    self.stemp = result.x[0]
                            break
                # If cooling, Qm must be 0
                Qm = 0

        # Update surface balance terms with new surftemp
        enbal.surfaceEB(self.stemp,layers,self.albedo,self.days_since_snowfall)
        self.Qm = Qm
        return
    
    def storeSurface(self):
        previous = self.days_since_snowfall
        # need to keep track of the layer that used to be the surface such 
        # that if the snowfall melts, albedo resets to the dirty surface

    def getAlbedo(self,layers):
        """
        Checks switches and gets albedo from corresponding method. If LAPs or grain size
        are tracked, albedo comes from SNICAR, otherwise it is parameterized by surface
        type or surface age.
        
        Parameters
        ----------
        layers
            class object from layers.py
        """
        # CONSTANTS
        ALBEDO_FIRN = eb_prms.albedo_firn
        ALBEDO_FRESH_SNOW = eb_prms.albedo_fresh_snow
        DEG_RATE = eb_prms.albedo_deg_rate

        args = self.args
        if self.stype == 'snow':
            if args.switch_melt == 0:
                if args.switch_LAPs == 0:
                    # SURFACE TYPE ONLY
                    self.albedo = self.albedo_dict[self.stype]
                elif args.switch_LAPs == 1:
                    # LAPs ON, GRAIN SIZE OFF
                    self.albedo = self.runSNICAR(layers,override_grainsize=True)
            elif args.switch_melt == 1:
                # BASIC DEGRADATION RATE
                age = self.days_since_snowfall
                aged_albedo = ALBEDO_FIRN+(ALBEDO_FRESH_SNOW-ALBEDO_FIRN)*(np.exp(-age/DEG_RATE))
                self.albedo = max(aged_albedo,ALBEDO_FIRN)
            elif args.switch_melt == 2:
                if args.switch_LAPs == 0:
                    # LAPs OFF, GRAIN SIZE ON
                    self.albedo = self.runSNICAR(layers,override_LAPs=True)
                elif args.switch_LAPs == 1:
                    # LAPs ON, GRAIN SIZE ON
                    self.albedo = self.runSNICAR(layers)
        else:
            self.albedo = self.albedo_dict[self.stype]
        return 
    
    def runSNICAR(self,layers,n_layers=None,max_depth=None,
                  override_grainsize=False,override_LAPs=False):
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
        override_grainsize : Bool
            If True, use constant grainsize specified in input.py
        override_LAPs: Bool
            If True, use constant LAP concentrations specified in input.py

        Returns
        -------
        layermelt : np.ndarray
            Array containing subsurface melt amounts [kg m-2]
        """
        import biosnicar as snicar

        # CONSTANTS
        GRAINSIZE = eb_prms.constant_grainsize

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

        # Grain size files are every 1um till 1500um, then every 500
        lgrainsize = layers.grainsize[idx].astype(int)
        lgrainsize[np.where(lgrainsize>1500)[0]] = np.round(lgrainsize[np.where(lgrainsize>1500)[0]]/500) * 500
        lgrainsize = lgrainsize.tolist()
        if override_grainsize:
            lgrainsize = [GRAINSIZE for _ in idx]

        # LAPs need to be a concentration in ppb
        BC1 = layers.lBC[0,idx] / layers.lheight[idx] * 1e6
        BC2 = layers.lBC[1,idx] / layers.lheight[idx] * 1e6
        dust1 = layers.ldust[0,idx] / layers.lheight[idx] * 1e6
        dust2 = layers.ldust[1,idx] / layers.lheight[idx] * 1e6
        dust3 = layers.ldust[2,idx] / layers.lheight[idx] * 1e6
        dust4 = layers.ldust[3,idx] / layers.lheight[idx] * 1e6
        dust5 = layers.ldust[4,idx] / layers.lheight[idx] * 1e6
        lBC1 = (BC1.astype(float)).tolist()
        lBC2 = (BC2.astype(float)).tolist()
        ldust1 = (dust1.astype(float)).tolist()
        ldust2 = (dust2.astype(float)).tolist()
        ldust3 = (dust3.astype(float)).tolist()
        ldust4 = (dust4.astype(float)).tolist()
        ldust5 = (dust5.astype(float)).tolist()
        if override_LAPs:
            lBC1 = [eb_prms.BC_freshsnow*1e6 for _ in idx]
            ldust1 = np.array([eb_prms.dust_freshsnow*1e6 for _ in idx]).tolist()
            lBC1 = lBC1.copy()
            ldust2 = ldust1.copy()
            ldust3 = ldust1.copy()
            ldust4 = ldust1.copy()
            ldust5 = ldust1.copy()

        # Open and edit yaml input file for SNICAR
        with open(eb_prms.snicar_input_fp) as f:
            list_doc = yaml.safe_load(f)

        # Update changing layer variables
        list_doc['IMPURITIES']['BC1']['CONC'] = lBC1
        list_doc['IMPURITIES']['BC2']['CONC'] = lBC2
        list_doc['IMPURITIES']['DUST1']['CONC'] = ldust1
        list_doc['IMPURITIES']['DUST2']['CONC'] = ldust2
        list_doc['IMPURITIES']['DUST3']['CONC'] = ldust3
        list_doc['IMPURITIES']['DUST4']['CONC'] = ldust4
        list_doc['IMPURITIES']['DUST5']['CONC'] = ldust5
        list_doc['ICE']['DZ'] = lheight
        list_doc['ICE']['RHO'] = ldensity
        list_doc['ICE']['RDS'] = lgrainsize

        # The following variables are set to constants, but need to have right number of layers
        ice_variables = ['LAYER_TYPE','SHP','HEX_SIDE','HEX_LENGTH','SHP_FCTR','WATER','AR','CDOM']
        for var in ice_variables:
            list_doc['ICE'][var] = [list_doc['ICE'][var][0]] * n_layers

        # Save SNICAR input file
        with open(eb_prms.snicar_input_fp, 'w') as f:
            yaml.dump(list_doc,f)
        
        # Get albedo from biosnicar "main.py"
        with HiddenPrints():
            albedo = snicar.main.get_albedo('adding-doubling',plot=False,validate=False)

        # I adjusted SNICAR code to return bba rather than spectral albedo
        # if I want to undo that change so it runs on base SNICAR, need to get bba from spectral
        # bba = np.sum(illumination.flx_slr * albedo) / np.sum(illumination.flx_slr)
        return albedo
    
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self,exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
        return