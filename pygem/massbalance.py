#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 14:00:14 2020

@author: davidrounce
"""
from oggm import cfg, utils
from oggm.core.massbalance import MassBalanceModel
import pygem.pygem_input as pygem_prms
import numpy as np
import pandas as pd
#import netCDF4

cfg.initialize()

#%%
class PyGEMMassBalance(MassBalanceModel):
    """Mass-balance computed from the Python Glacier Evolution Model.

    This mass balance accounts for ablation, accumulation, and refreezing.

    This class implements the MassBalanceModel interface so that the dynamical model can use it.
    """

    def __init__(self, modelparameters, glacier_rgi_table, glacier_area_initial, icethickness_initial, 
                       width_initial, heights, glacier_gcm_temp, glacier_gcm_tempstd, glacier_gcm_prec, 
                       glacier_gcm_elev, glacier_gcm_lrgcm, glacier_gcm_lrglac, dates_table, option_areaconstant=0, 
                       constantarea_years=pygem_prms.constantarea_years, frontalablation_k=None, 
                       debug=False, debug_refreeze=False, hindcast=0, 
#                       use_refreeze=True
                       ):
        """ Initialize.

        Parameters
        ----------
        modelparameters : pd.Series
            Model parameters (lrgcm, lrglac, precfactor, precgrad, ddfsnow, ddfice, tempsnow, tempchange)
            Order of model parameters should not be changed as the run mass balance script relies on this order.
        glacier_rgi_table : pd.Series
            Table of glacier's RGI information
        glacier_area_initial : np.ndarray
            Initial glacier area [km2] for each elevation bin
        icethickness_initial : np.ndarray
            Initial ice thickness [m] for each elevation bin
        width_initial : np.ndarray
            Initial glacier width [km] for each elevation bin
        heights : np.ndarray
            height of elevation bins [masl]
        glacier_gcm_temp : np.ndarray
            GCM temperature [degC] at each time step based on nearest neighbor to the glacier
        glacier_gcm_tempstd : np.ndarray
            GCM temperature daily standard deviation [degC] at each time step based on nearest neighbor to the glacier
        glacier_gcm_prec : np.ndarray
            GCM precipitation (solid and liquid) [m] at each time step based on nearest neighbor to the glacier
        glacier_gcm_elev : float
            GCM elevation [masl] at each time step based on nearest neighbor to the glacier
        glacier_gcm_lrgcm : np.ndarray
            GCM lapse rate [K m-1] from GCM to the glacier for each time step based on nearest neighbor to the glacier
        glacier_gcm_lrglac : np.ndarray
            GCM lapse rate [K m-1] over the glacier for each time step based on nearest neighbor to the glacier
        dates_table : pd.DataFrame
            Table of dates, year, month, daysinmonth, wateryear, and season for each timestep
        option_areaconstant : int
            switch to keep glacier area constant or not (default 0 allows glacier area to change annually)
        constantarea_years : int
            number of years to keep the glacier area constant
        frontalablation_k : float
            frontal ablation parameter
        debug : Boolean
            option to turn on print statements for development or debugging of code
        debug_refreeze : Boolean
            option to turn on print statements for development/debugging of refreezing code
        hindcast : int
            switch to run the model in reverse or not (may be irrelevant after converting to OGGM's setup)
        """
        super(PyGEMMassBalance, self).__init__()
        self.valid_bounds = [-1e4, 2e4]  # in m
        self.hemisphere = 'nh'
        
        # Example of what could be done!
#        self.use_refreeze = use_refreeze
        
        #%%
        self.modelparameters = modelparameters
        self.glacier_rgi_table = glacier_rgi_table
        self.glacier_area_initial = glacier_area_initial
        self.icethickness_initial = icethickness_initial
        self.width_initial = width_initial
        self.heights = heights 
        self.glacier_gcm_temp = glacier_gcm_temp
        self.glacier_gcm_tempstd = glacier_gcm_tempstd
        self.glacier_gcm_prec = glacier_gcm_prec 
        self.glacier_gcm_elev = glacier_gcm_elev
        self.glacier_gcm_lrgcm = glacier_gcm_lrgcm
        self.glacier_gcm_lrglac = glacier_gcm_lrglac
        self.dates_table = dates_table
        
        
        #%%

        if debug:
            print('\n\nDEBUGGING MASS BALANCE FUNCTION\n\n')
            
        # Select annual divisor and columns
#        if pygem_prms.timestep == 'monthly':
#            annual_divisor = 12
        annual_columns = np.arange(0, int(dates_table.shape[0] / 12))
        
        # Variables to store
        nbins = heights.shape[0]
        nmonths = glacier_gcm_temp.shape[0]
        nyears = annual_columns.shape[0]
        
        # Consider storing in xarray
        self.bin_temp = np.zeros((nbins,nmonths))
        self.bin_prec = np.zeros((nbins,nmonths))
        self.bin_acc = np.zeros((nbins,nmonths))
        self.bin_refreezepotential = np.zeros((nbins,nmonths))
        self.bin_refreeze = np.zeros((nbins,nmonths))
        self.bin_meltsnow = np.zeros((nbins,nmonths))
        self.bin_meltglac = np.zeros((nbins,nmonths))
        self.bin_melt = np.zeros((nbins,nmonths))
        self.bin_snowpack = np.zeros((nbins,nmonths))
        self.glac_bin_refreeze = np.zeros((nbins,nmonths))
        self.glac_bin_melt = np.zeros((nbins,nmonths))
        self.glac_bin_frontalablation = np.zeros((nbins,nmonths))
        self.glac_bin_snowpack = np.zeros((nbins,nmonths))
        self.glac_bin_massbalclim = np.zeros((nbins,nmonths))
        self.glac_bin_massbalclim_annual = np.zeros((nbins,nyears))
        self.glac_bin_surfacetype_annual = np.zeros((nbins,nyears))
        self.glac_bin_icethickness_annual = np.zeros((nbins, nyears + 1))
#        self.glac_bin_area_annual = np.zeros((nbins, nyears + 1))
#        self.glac_bin_width_annual = np.zeros((nbins, nyears + 1))
    
        self.offglac_bin_prec = np.zeros((nbins,nmonths))
        self.offglac_bin_melt = np.zeros((nbins,nmonths))
        self.offglac_bin_refreeze = np.zeros((nbins,nmonths))
        self.offglac_bin_snowpack = np.zeros((nbins,nmonths))
        self.offglac_bin_area_annual = np.zeros((nbins, nyears + 1))
        
        # Local variables
        self.bin_precsnow = np.zeros((nbins,nmonths))
        self.refreeze_potential = np.zeros(nbins)
        self.snowpack_remaining = np.zeros(nbins)
        self.dayspermonth = dates_table['daysinmonth'].values
        self.surfacetype_ddf = np.zeros(nbins)
        self.glac_idx_initial = glacier_area_initial.nonzero()[0]
        self.glacier_area_initial = glacier_area_initial.copy()
        self.glacier_area_t0 = glacier_area_initial.copy()
        self.icethickness_t0 = icethickness_initial.copy()
        self.width_t0 = width_initial.copy()
        if pygem_prms.option_refreezing == 1:
            # Refreezing layers density, volumetric heat capacity, and thermal conductivity
            self.rf_dens_expb = (pygem_prms.rf_dens_bot / pygem_prms.rf_dens_top)**(1/(pygem_prms.rf_layers-1))
            self.rf_layers_dens = np.array([pygem_prms.rf_dens_top * self.rf_dens_expb**x for x in np.arange(0,pygem_prms.rf_layers)])
            self.rf_layers_ch = (1 - self.rf_layers_dens/1000) * pygem_prms.ch_air + self.rf_layers_dens/1000 * pygem_prms.ch_ice
            self.rf_layers_k = (1 - self.rf_layers_dens/1000) * pygem_prms.k_air + self.rf_layers_dens/1000 * pygem_prms.k_ice
            self.te_rf = np.zeros((pygem_prms.rf_layers,nbins))     # temp of each layer for each elev bin from present time step
            self.tl_rf = np.zeros((pygem_prms.rf_layers,nbins))     # temp of each layer for each elev bin from previous time step
            self.refr = np.zeros(nbins)                             # refreeze in each bin
            self.rf_cold = np.zeros(nbins)                          # refrezee cold content or "potential" refreeze
        
        # Sea level for marine-terminating glaciers
        self.sea_level = 0
        rgi_region = int(glacier_rgi_table.RGIId.split('-')[1].split('.')[0])
        if frontalablation_k == None:
             self.frontalablation_k0 = pygem_prms.frontalablation_k0dict[rgi_region]
        
        #  glac_idx_initial is used with advancing glaciers to ensure no bands are added in a discontinuous section
        # Run mass balance only on pixels that have an ice thickness (some glaciers do not have an ice thickness)
        #  otherwise causes indexing errors causing the model to fail
        if self.icethickness_t0.max() > 0:    
            if pygem_prms.option_adjusttemp_surfelev == 1:
                # ice thickness initial is used to adjust temps to changes in surface elevation
                self.icethickness_initial = self.icethickness_t0.copy()
                self.icethickness_initial[0:icethickness_initial.nonzero()[0][0]] = (
                        self.icethickness_initial[icethickness_initial.nonzero()[0][0]])
                #  bins that advance need to have an initial ice thickness; otherwise, the temp adjustment will be based on
                #  ice thickness - 0, which is wrong  Since advancing bins take the thickness of the previous bin, set the
                #  initial ice thickness of all bins below the terminus to the ice thickness at the terminus.
            # Compute the initial surface type [0=off-glacier, 1=ice, 2=snow, 3=firn, 4=debris]
#            self.surfacetype, self.firnline_idx = self._surfacetypebinsinitial(self.glacier_area_t0, glacier_rgi_table, heights)
            self.surfacetype, self.firnline_idx = self._surfacetypebinsinitial(heights)
            # Create surface type DDF dictionary (manipulate this function for calibration or for each glacier)
            self.surfacetype_ddf_dict = self._surfacetypeDDFdict(modelparameters)   
        
#        # ANNUAL LOOP (daily or monthly timestep contained within loop)
#        for year in range(0, nyears): 
#            # Check ice still exists:
            
    def get_annual_mb(self, heights, year=None, flowline=None, fl_id=None, 
                      debug=False, option_areaconstant=0):
        """Returns annual climatic mass balance
        
        Parameters
        ----------
        heights : np.array
            elevation bins
        year : int
            year starting with 0 to the number of years in the study
        
        FIXED FOR THE FLOWLINE MODEL"""
        if self.icethickness_t0.max() > 0:    
            print('time for mass balance!')
            
            if debug:
                print(year, 'max ice thickness [m]:', self.icethickness_t0.max())
    
            print('hack until Fabien provides data')
            class Dummy():
                pass
            flowline = Dummy()
            flowline.area_km2 = self.glacier_area_t0
    
            # Glacier indices
            glac_idx_t0 = flowline.area_km2.nonzero()[0]
            
            # Off-glacier area and indices
            if option_areaconstant == 0:
                self.offglac_bin_area_annual[:,year] = self.glacier_area_initial - flowline.area_km2
                offglac_idx = np.where(self.offglac_bin_area_annual[:,year] > 0)[0]            
            
            # Functions currently set up for monthly timestep
            #  only compute mass balance while glacier exists
            if (pygem_prms.timestep == 'monthly') and (glac_idx_t0.shape[0] != 0):      
                
                # AIR TEMPERATURE: Downscale the gcm temperature [deg C] to each bin
                if pygem_prms.option_temp2bins == 1:
                    # Downscale using gcm and glacier lapse rates
                    #  T_bin = T_gcm + lr_gcm * (z_ref - z_gcm) + lr_glac * (z_bin - z_ref) + tempchange
                    self.bin_temp[:,12*year:12*(year+1)] = (self.glacier_gcm_temp[12*year:12*(year+1)] + 
                         self.glacier_gcm_lrgcm[12*year:12*(year+1)] * 
                         (self.glacier_rgi_table.loc[pygem_prms.option_elev_ref_downscale] - self.glacier_gcm_elev) + 
                         self.glacier_gcm_lrglac[12*year:12*(year+1)] * (heights - 
                         self.glacier_rgi_table.loc[pygem_prms.option_elev_ref_downscale])[:,np.newaxis] + self.modelparameters[7])
                    
                    print(self.bin_temp[glac_idx_t0,12*year:12*(year+1)])
                # Option to adjust air temperature based on changes in surface elevation
#                #  note: OGGM automatically updates the bin elevation, so this step is not needed
#                if pygem_prms.option_adjusttemp_surfelev == 1 and pygem_prms.hyps_data in ['Huss', 'Farinotti']:
#                    # T_air = T_air + lr_glac * (icethickness_present - icethickness_initial)
#                    bin_temp[:,12*year:12*(year+1)] = (bin_temp[:,12*year:12*(year+1)] + 
#                                                       glacier_gcm_lrglac[12*year:12*(year+1)] * 
#                                                       (icethickness_t0 - icethickness_initial)[:,np.newaxis])
#                
#                # PRECIPITATION/ACCUMULATION: Downscale the precipitation (liquid and solid) to each bin
#                if pygem_prms.option_prec2bins == 1:
#                    # Precipitation using precipitation factor and precipitation gradient
#                    #  P_bin = P_gcm * prec_factor * (1 + prec_grad * (z_bin - z_ref))
#                    bin_precsnow[:,12*year:12*(year+1)] = (glacier_gcm_prec[12*year:12*(year+1)] * 
#                            modelparameters[2] * (1 + modelparameters[3] * (heights - 
#                            glacier_rgi_table.loc[pygem_prms.option_elev_ref_downscale]))[:,np.newaxis])
#                # Option to adjust prec of uppermost 25% of glacier for wind erosion and reduced moisture content
#                if pygem_prms.option_preclimit == 1:
#                    # If elevation range > 1000 m, apply corrections to uppermost 25% of glacier (Huss and Hock, 2015)
#                    if np.abs(heights[glac_idx_t0[-1]] - heights[glac_idx_t0[0]]) > 1000:
#                        # Indices of upper 25%
#                        glac_idx_upper25 = (
#                                glac_idx_t0[(heights[glac_idx_t0] - heights[glac_idx_t0].min()) / 
#                                            (heights[glac_idx_t0].max() - heights[glac_idx_t0].min()) * 100 >= 75])                        
#                        # Exponential decay according to elevation difference from the 75% elevation
#                        #  prec_upper25 = prec * exp(-(elev_i - elev_75%)/(elev_max- - elev_75%))
#                        # height at 75% of the elevation
#                        height_75 = heights[glac_idx_upper25].min()
#                        glac_idx_75 = np.where(heights == height_75)[0][0]
#                        # exponential decay
#                        bin_precsnow[glac_idx_upper25,12*year:12*(year+1)] = (
#                                bin_precsnow[glac_idx_75,12*year:12*(year+1)] * 
#                                np.exp(-1*(heights[glac_idx_upper25] - height_75) / 
#                                       (heights[glac_idx_upper25].max() - heights[glac_idx_upper25].min()))
#                                [:,np.newaxis])
#                        
#                        # Precipitation cannot be less than 87.5% of the maximum accumulation elsewhere on the glacier
#                        for month in range(0,12):
#                            bin_precsnow[glac_idx_upper25[(bin_precsnow[glac_idx_upper25,month] < 0.875 * 
#                                bin_precsnow[glac_idx_t0,month].max()) & 
#                                (bin_precsnow[glac_idx_upper25,month] != 0)], month] = (
#                                                                0.875 * bin_precsnow[glac_idx_t0,month].max())
#                # Separate total precipitation into liquid (bin_prec) and solid (bin_acc)
#                if pygem_prms.option_accumulation == 1:
#                    # if temperature above threshold, then rain
#                    bin_prec[:,12*year:12*(year+1)][bin_temp[:,12*year:12*(year+1)] > modelparameters[6]] = (
#                        bin_precsnow[:,12*year:12*(year+1)][bin_temp[:,12*year:12*(year+1)] > modelparameters[6]])
#                    # if temperature below threshold, then snow
#                    bin_acc[:,12*year:12*(year+1)][bin_temp[:,12*year:12*(year+1)] <= modelparameters[6]] = (
#                        bin_precsnow[:,12*year:12*(year+1)][bin_temp[:,12*year:12*(year+1)] <= modelparameters[6]])
#                elif pygem_prms.option_accumulation == 2:
#                    # If temperature between min/max, then mix of snow/rain using linear relationship between min/max
#                    bin_prec[:,12*year:12*(year+1)] = ((1/2 + (bin_temp[:,12*year:12*(year+1)] - 
#                            modelparameters[6]) / 2) * bin_precsnow[:,12*year:12*(year+1)])
#                    bin_acc[:,12*year:12*(year+1)] = (bin_precsnow[:,12*year:12*(year+1)] - 
#                           bin_prec[:,12*year:12*(year+1)])
#                    # If temperature above maximum threshold, then all rain
#                    bin_prec[:,12*year:12*(year+1)][bin_temp[:,12*year:12*(year+1)] > modelparameters[6] + 1] = (
#                            bin_precsnow[:,12*year:12*(year+1)][bin_temp[:,12*year:12*(year+1)] > 
#                                         modelparameters[6] + 1])
#                    bin_acc[:,12*year:12*(year+1)][bin_temp[:,12*year:12*(year+1)] > modelparameters[6] + 1] = 0
#                    # If temperature below minimum threshold, then all snow
#                    (bin_acc[:,12*year:12*(year+1)][bin_temp[:,12*year:12*(year+1)] <= modelparameters[6] - 1]
#                        )= (bin_precsnow[:,12*year:12*(year+1)][bin_temp[:,12*year:12*(year+1)] <= 
#                                         modelparameters[6] - 1])
#                    bin_prec[:,12*year:12*(year+1)][bin_temp[:,12*year:12*(year+1)] <= modelparameters[6] - 1] = 0
#                
#                
#                # ENTER MONTHLY LOOP (monthly loop required as )
#                for month in range(0,12):
#                    # Step is the position as a function of year and month, which improves readability
#                    step = 12*year + month
#                    
#                    # ACCUMULATION, MELT, REFREEZE, AND CLIMATIC MASS BALANCE
#                    # Snowpack [m w.e.] = snow remaining + new snow
#                    bin_snowpack[:,step] = snowpack_remaining + bin_acc[:,step]
#                    
#                    # MELT [m w.e.]
#                    # energy available for melt [degC day]   
#                    if pygem_prms.option_ablation == 1:
#                        # option 1: energy based on monthly temperature
#                        melt_energy_available = bin_temp[:,step]*dayspermonth[step]
#                        melt_energy_available[melt_energy_available < 0] = 0
#                    elif pygem_prms.option_ablation == 2:
#                        # option 2: monthly temperature superimposed with daily temperature variability 
#                        # daily temperature variation in each bin for the monthly timestep
#                        bin_tempstd_daily = np.repeat(
#                                np.random.normal(loc=0, scale=glacier_gcm_tempstd[step], size=dayspermonth[step])
#                                .reshape(1,dayspermonth[step]), nbins, axis=0)
#                        # daily temperature in each bin for the monthly timestep
#                        bin_temp_daily = bin_temp[:,step][:,np.newaxis] + bin_tempstd_daily
#                        # remove negative values
#                        bin_temp_daily[bin_temp_daily < 0] = 0
#                        # Energy available for melt [degC day] = sum of daily energy available
#                        melt_energy_available = bin_temp_daily.sum(axis=1)
#                    # SNOW MELT [m w.e.]
#                    bin_meltsnow[:,step] = surfacetype_ddf_dict[2] * melt_energy_available
#                    # snow melt cannot exceed the snow depth
#                    bin_meltsnow[bin_meltsnow[:,step] > bin_snowpack[:,step], step] = (
#                            bin_snowpack[bin_meltsnow[:,step] > bin_snowpack[:,step], step])
#                    # GLACIER MELT (ice and firn) [m w.e.]
#                    # energy remaining after snow melt [degC day]
#                    melt_energy_available = melt_energy_available - bin_meltsnow[:,step] / surfacetype_ddf_dict[2]
#                    # remove low values of energy available caused by rounding errors in the step above
#                    melt_energy_available[abs(melt_energy_available) < pygem_prms.tolerance] = 0
#                    # DDF based on surface type [m w.e. degC-1 day-1]
#                    for surfacetype_idx in surfacetype_ddf_dict: 
#                        surfacetype_ddf[surfacetype == surfacetype_idx] = surfacetype_ddf_dict[surfacetype_idx]
#                    if pygem_prms.option_surfacetype_debris == 1:
#                        print('\n\nLOAD THE MELTFACTOR DATASET over areas that are not firn\n\n')
#                        
#                        if year == 0 and month == 0:
#                            print('\nDELETE ME\n surfacetype_ddf[glac_idx]:', surfacetype_ddf[glac_idx_t0])
#                        
#                    bin_meltglac[glac_idx_t0,step] = surfacetype_ddf[glac_idx_t0] * melt_energy_available[glac_idx_t0]
#                    # TOTAL MELT (snow + glacier)
#                    #  off-glacier need to include melt of refreeze because there are no glacier dynamics,
#                    #  but on-glacier do not need to account for this (simply assume refreeze has same surface type)
#                    bin_melt[:,step] = bin_meltglac[:,step] + bin_meltsnow[:,step]  
#                    
#                    # REFREEZING
#                    if pygem_prms.option_refreezing == 1:
#                        # Refreeze based on heat conduction approach (Huss and Hock 2015)    
#                        # refreeze time step (s)
#                        rf_dt = 3600 * 24 * dates_table.loc[step,'daysinmonth'] / pygem_prms.rf_dsc
#                        
#                        if pygem_prms.option_rf_limit_meltsnow == 1:
#                            bin_meltlimit = bin_meltsnow.copy()
#                        else:
#                            bin_meltlimit = bin_melt.copy()
#                            
#                        # Loop through each elevation bin of glacier
#                        for nbin, gidx in enumerate(glac_idx_t0):
#                                    
#                            # COMPUTE HEAT CONDUCTION - BUILD COLD RESERVOIR                            
#                            # If no melt, then build up cold reservoir (compute heat conduction)
#                            if bin_melt[gidx,step] < pygem_prms.rf_meltcrit:
#                                
#                                if debug_refreeze and nbin == 0 and step < 12:
#                                    print('\nMonth ' + str(dates_table.loc[step,'month']), 'Computing heat conduction')
#                                
#                                # Set refreeze equal to 0
#                                refr[gidx] = 0
#                                # Loop through multiple iterations to converge on a solution
#                                #  -> this will loop through 0, 1, 2
#                                for h in np.arange(0, pygem_prms.rf_dsc):                                  
#                                    # Compute heat conduction in layers (loop through rows)
#                                    #  go from 1 to rf_layers-1 to avoid indexing errors with "j-1" and "j+1"
#                                    #  "j+1" is set to zero, which is fine for temperate glaciers but inaccurate for cold/polythermal glaciers
#                                    for j in np.arange(1, pygem_prms.rf_layers-1):
#                                        # Assume temperature of first layer equals air temperature
#                                        #  assumption probably wrong, but might still work at annual average
#                                        # Since next line uses tl_rf for all calculations, set tl_rf[0] to present mean 
#                                        #  monthly air temperature to ensure the present calculations are done with the 
#                                        #  present time step's air temperature
#                                        tl_rf[0, gidx] = bin_temp[gidx,step]
#                                        # Temperature for each layer
#                                        te_rf[j,gidx] = (tl_rf[j,gidx] + 
#                                             rf_dt * rf_layers_k[j] / rf_layers_ch[j] / pygem_prms.rf_dz**2 * 0.5 * 
#                                             ((tl_rf[j-1,gidx] - tl_rf[j,gidx]) - (tl_rf[j,gidx] - tl_rf[j+1,gidx]))) 
#                                        # Update previous time step
#                                        tl_rf[:,gidx] = te_rf[:,gidx]
#                                   
#                                if debug_refreeze and nbin == 0 and step < 12:
#                                    print('tl_rf:', ["{:.2f}".format(x) for x in tl_rf[:,gidx]])
#                                            
#                            # COMPUTE REFREEZING - TAP INTO "COLD RESERVOIR" or potential refreezing  
#                            else:
#                                
#                                if debug_refreeze and nbin == 0 and step < 12:
#                                    print('\nMonth ' + str(dates_table.loc[step,'month']), 'Computing refreeze')
#                                
#                                # Refreezing over firn surface
#                                if (surfacetype[gidx] == 2) or (surfacetype[gidx] == 3):
#                                    nlayers = pygem_prms.rf_layers-1
#                                # Refreezing over ice surface
#                                else:
#                                    # Approximate number of layers of snow on top of ice
#                                    smax = np.round((bin_snowpack[gidx,step] / (rf_layers_dens[0] / 1000) + pygem_prms.pp) / pygem_prms.rf_dz, 0)
#                                    # if there is very little snow on the ground (SWE > 0.06 m for pp=0.3), then still set smax (layers) to 1
#                                    if bin_snowpack[gidx,step] > 0 and smax == 0:
#                                        smax=1
#                                    # if no snow on the ground, then set to rf_cold to NoData value
#                                    if smax == 0:
#                                        rf_cold[gidx] = 0
#                                    # if smax greater than the number of layers, set to max number of layers minus 1
#                                    if smax > pygem_prms.rf_layers - 1:
#                                        smax = pygem_prms.rf_layers - 1
#                                    nlayers = int(smax)
#                        
#                                # Compute potential refreeze, "cold reservoir", from temperature in each layer 
#                                # only calculate potential refreezing first time it starts melting each year
#                                if rf_cold[gidx] == 0 and tl_rf[:,gidx].min() < 0:
#                                    
#                                    if debug_refreeze and nbin == 0 and step < 12:
#                                        print('calculating potential refreeze from ' + str(nlayers) + ' layers')
#                                    
#                                    for j in np.arange(0,nlayers):
#                                        j += 1
#                                        # units: (degC) * (J K-1 m-3) * (m) * (kg J-1) * (m3 kg-1)
#                                        rf_cold_layer = tl_rf[j,gidx] * rf_layers_ch[j] * pygem_prms.rf_dz / pygem_prms.Lh_rf / pygem_prms.density_water
#                                        rf_cold[gidx] -= rf_cold_layer
#                                        
#                                        if debug_refreeze and nbin == 0 and step < 12:
#                                            print('j:', j, 'tl_rf @ j:', np.round(tl_rf[j,gidx],2), 
#                                                           'ch @ j:', np.round(rf_layers_ch[j],2), 
#                                                           'rf_cold_layer @ j:', np.round(rf_cold_layer,2),
#                                                           'rf_cold @ j:', np.round(rf_cold[gidx],2))
#                                    
#                                    if debug_refreeze and nbin == 0 and step < 12:
#                                        print('rf_cold:', np.round(rf_cold[gidx],2))
#                                        
#                                # Compute refreezing
#                                # If melt and liquid prec < potential refreeze, then refreeze all melt and liquid prec
#                                if (bin_meltlimit[gidx,step] + bin_prec[gidx,step]) < rf_cold[gidx]:
#                                    refr[gidx] = bin_meltlimit[gidx,step] + bin_prec[gidx,step]
#                                # otherwise, refreeze equals the potential refreeze
#                                elif rf_cold[gidx] > 0:
#                                    refr[gidx] = rf_cold[gidx]
#                                else:
#                                    refr[gidx] = 0
#
#                                # Track the remaining potential refreeze                                    
#                                rf_cold[gidx] -= (bin_meltlimit[gidx,step] + bin_prec[gidx,step])
#                                # if potential refreeze consumed, then set to 0 and set temperature to 0 (temperate firn)
#                                if rf_cold[gidx] < 0:
#                                    rf_cold[gidx] = 0
#                                    tl_rf[:,gidx] = 0
#                                    
#                            # Record refreeze
#                            bin_refreeze[gidx,step] = refr[gidx]
#                            
#                            if debug_refreeze and step < 12 and nbin==0:
#                                print('Month ' + str(dates_table.loc[step,'month']),
#                                      'Rf_cold remaining:', np.round(rf_cold[gidx],2),
#                                      'Snow depth:', np.round(bin_snowpack[glac_idx_t0[nbin],step],2),
#                                      'Snow melt:', np.round(bin_meltsnow[glac_idx_t0[nbin],step],2),
#                                      'Rain:', np.round(bin_prec[glac_idx_t0[nbin],step],2),
#                                      'Rfrz:', np.round(bin_refreeze[gidx,step],2)
#                                      )             
#                                
#                    elif pygem_prms.option_refreezing == 2:
#                        # Refreeze based on annual air temperature (Woodward etal. 1997)
#                        #  R(m) = (-0.69 * Tair + 0.0096) * 1 m / 100 cm
#                        # calculate annually and place potential refreeze in user defined month
#                        if dates_table.loc[step,'month'] == pygem_prms.rf_month:                        
#                            bin_temp_annual = annualweightedmean_array(bin_temp[:,12*year:12*(year+1)], 
#                                                                       dates_table.iloc[12*year:12*(year+1),:])
#                            bin_refreezepotential_annual = (-0.69 * bin_temp_annual + 0.0096) * 1/100
#                            # Remove negative refreezing values
#                            bin_refreezepotential_annual[bin_refreezepotential_annual < 0] = 0
#                            bin_refreezepotential[:,step] = bin_refreezepotential_annual
#                            # Reset refreeze potential every year
#                            if bin_refreezepotential[:,step].max() > 0:
#                                refreeze_potential = bin_refreezepotential[:,step]
#
#                        
#                        if debug_refreeze and step < 12:
#                            print('Month' + str(dates_table.loc[step,'month']),
#                                  'Refreeze potential:', np.round(refreeze_potential[glac_idx_t0[0]],3),
#                                  'Snow depth:', np.round(bin_snowpack[glac_idx_t0[0],step],2),
#                                  'Snow melt:', np.round(bin_meltsnow[glac_idx_t0[0],step],2),
#                                  'Rain:', np.round(bin_prec[glac_idx_t0[0],step],2))
#                            
#                            
#                        # Refreeze [m w.e.]
#                        #  refreeze cannot exceed rain and melt (snow & glacier melt)
#                        bin_refreeze[:,step] = bin_meltsnow[:,step] + bin_prec[:,step]
#                        # refreeze cannot exceed snow depth
#                        bin_refreeze[bin_refreeze[:,step] > bin_snowpack[:,step], step] = (
#                                bin_snowpack[bin_refreeze[:,step] > bin_snowpack[:,step], step])
#                        # refreeze cannot exceed refreeze potential
#                        bin_refreeze[bin_refreeze[:,step] > refreeze_potential, step] = (
#                                refreeze_potential[bin_refreeze[:,step] > refreeze_potential])
#                        bin_refreeze[abs(bin_refreeze[:,step]) < pygem_prms.tolerance, step] = 0
#                        # update refreeze potential
#                        refreeze_potential = refreeze_potential - bin_refreeze[:,step]
#                        refreeze_potential[abs(refreeze_potential) < pygem_prms.tolerance] = 0
#                    
#                    if step < 12 and debug_refreeze:
#                        print('refreeze bin ' + str(int(glac_idx_t0[0]*10)) + ':', 
#                                np.round(bin_refreeze[glac_idx_t0[0],step],3))
#                    
#                    # SNOWPACK REMAINING [m w.e.]
#                    snowpack_remaining = bin_snowpack[:,step] - bin_meltsnow[:,step]
#                    snowpack_remaining[abs(snowpack_remaining) < pygem_prms.tolerance] = 0
#                    
#                    # Record values                  
#                    glac_bin_melt[glac_idx_t0,step] = bin_melt[glac_idx_t0,step]
#                    glac_bin_refreeze[glac_idx_t0,step] = bin_refreeze[glac_idx_t0,step]
#                    glac_bin_snowpack[glac_idx_t0,step] = bin_snowpack[glac_idx_t0,step]
#                    # CLIMATIC MASS BALANCE [m w.e.]
#                    glac_bin_massbalclim[glac_idx_t0,step] = (
#                            bin_acc[glac_idx_t0,step] + glac_bin_refreeze[glac_idx_t0,step] - 
#                            glac_bin_melt[glac_idx_t0,step])
#                    
#                    # OFF-GLACIER ACCUMULATION, MELT, REFREEZE, AND SNOWPAC
#                    if option_areaconstant == 0:
#                        # precipitation, refreeze, and snowpack are the same both on- and off-glacier
#                        offglac_bin_prec[offglac_idx,step] = bin_prec[offglac_idx,step]
#                        offglac_bin_refreeze[offglac_idx,step] = bin_refreeze[offglac_idx,step]
#                        offglac_bin_snowpack[offglac_idx,step] = bin_snowpack[offglac_idx,step]
#                        # Off-glacier melt includes both snow melt and melting of refreezing
#                        #  (this is not an issue on-glacier because energy remaining melts underlying snow/ice)
#                        # melt of refreezing (assumed to be snow)
#                        offglac_meltrefreeze = surfacetype_ddf_dict[2] * melt_energy_available
#                        # melt of refreezing cannot exceed refreezing
#                        offglac_meltrefreeze[offglac_meltrefreeze > bin_refreeze[:,step]] = (
#                                bin_refreeze[:,step][offglac_meltrefreeze > bin_refreeze[:,step]])
#                        # off-glacier melt = snow melt + refreezing melt
#                        offglac_bin_melt[offglac_idx,step] = (bin_meltsnow[offglac_idx,step] + 
#                                                              offglac_meltrefreeze[offglac_idx])
#                
#                # ===== RETURN TO ANNUAL LOOP =====
#                # Mass loss cannot exceed glacier volume
#                #  mb [mwea] = -1 * sum{area [km2] * ice thickness [m]} / total area [km2] * density_ice / density_water
#                mb_max_loss = (-1 * (glacier_area_t0 * icethickness_t0 * pygem_prms.density_ice / pygem_prms.density_water).sum() 
#                               / glacier_area_t0.sum())
#                # Check annual climatic mass balance
#                mb_mwea = ((glacier_area_t0 * glac_bin_massbalclim[:,12*year:12*(year+1)].sum(1)).sum() / 
#                            glacier_area_t0.sum()) 
#                
#                
#                    
#                # If mass loss exceeds glacier mass, reduce melt to ensure the glacier completely melts without excess
#                if mb_mwea < mb_max_loss:  
#                    
#                    if debug:
#                         print('mb_mwea (before):', np.round(mb_mwea,3), 'mb_max_loss:', np.round(mb_max_loss,3))
#                         
#                    mb_dif = mb_max_loss - mb_mwea
#                   
#                    glac_wide_melt = ((glac_bin_melt[:,12*year:12*(year+1)] * glacier_area_t0[:,np.newaxis]).sum() / 
#                                       glacier_area_t0.sum())
#                    # adjust using tolerance to avoid any rounding errors that would leave a little glacier volume left
#                    glac_bin_melt[:,12*year:12*(year+1)] = (glac_bin_melt[:,12*year:12*(year+1)] * 
#                                                            (1 + pygem_prms.tolerance - mb_dif / glac_wide_melt))
#                    glac_bin_massbalclim[:,12*year:12*(year+1)] = (
#                            bin_acc[:,12*year:12*(year+1)] + glac_bin_refreeze[:,12*year:12*(year+1)] - 
#                            glac_bin_melt[:,12*year:12*(year+1)])    
#                    # Check annual climatic mass balance
#                    mb_mwea = ((glacier_area_t0 * glac_bin_massbalclim[:,12*year:12*(year+1)].sum(1)).sum() / 
#                                glacier_area_t0.sum()) 
#                    
#                    if debug:
#                        print('mb_check after adjustment (should equal mass loss):', np.round(mb_mwea,3))
#                    
#                
#                # FRONTAL ABLATION
#                # Glacier bed altitude [masl]
#                glac_idx_minelev = np.where(heights == heights[glac_idx_t0].min())[0][0]
#                glacier_bedelev = (heights[glac_idx_minelev] - icethickness_initial[glac_idx_minelev])
#                
#                if debug and glacier_rgi_table['TermType'] != 0:
#                    print('\nyear:', year, '\n sea level:', sea_level, 'bed elev:', np.round(glacier_bedelev, 2))
#                                
#                # If glacier bed below sea level, compute frontal ablation
#                if glacier_bedelev < sea_level:
#                    # Volume [m3] and bed elevation [masl] of each bin
#                    glac_bin_volume = glacier_area_t0 * 10**6 * icethickness_t0
#                    glac_bin_bedelev = np.zeros((glacier_area_t0.shape))
#                    glac_bin_bedelev[glac_idx_t0] = heights[glac_idx_t0] - icethickness_initial[glac_idx_t0]
#                    
#                    # Option 1: Use Huss and Hock (2015) frontal ablation parameterizations
#                    #  Frontal ablation using width of lowest bin can severely overestimate the actual width of the
#                    #  calving front. Therefore, use estimated calving width from satellite imagery as appropriate.
#                    if pygem_prms.option_frontalablation_k == 1 and frontalablation_k == None:
#                        # Calculate frontal ablation parameter based on slope of lowest 100 m of glacier
#                        glac_idx_slope = np.where((heights <= sea_level + 100) & 
#                                                  (heights >= heights[glac_idx_t0].min()))[0]
#                        elev_change = np.abs(heights[glac_idx_slope[0]] - heights[glac_idx_slope[-1]])
#                        # length of lowest 100 m of glacier
#                        length_lowest100m = (glacier_area_t0[glac_idx_slope] / 
#                                             width_t0[glac_idx_slope] * 1000).sum()
#                        # slope of lowest 100 m of glacier
#                        slope_lowest100m = np.rad2deg(np.arctan(elev_change/length_lowest100m))
#                        # Frontal ablation parameter
#                        frontalablation_k = frontalablation_k0 * slope_lowest100m
#
#                    # Calculate frontal ablation
#                    # Bed elevation with respect to sea level
#                    #  negative when bed is below sea level (Oerlemans and Nick, 2005)
#                    waterdepth = sea_level - glacier_bedelev
#                    # Glacier length [m]
#                    length = (glacier_area_t0[width_t0 > 0] / width_t0[width_t0 > 0]).sum() * 1000
#                    # Height of calving front [m]
#                    height_calving = np.max([pygem_prms.af*length**0.5, 
#                                             pygem_prms.density_water / pygem_prms.density_ice * waterdepth])
#                    # Width of calving front [m]
#                    if pygem_prms.hyps_data in ['oggm']:
#                        width_calving = width_t0[np.where(heights == heights[glac_idx_t0].min())[0][0]] * 1000
#                    elif pygem_prms.hyps_data in ['Huss', 'Farinotti']:
#                        if glacier_rgi_table.RGIId in pygem_prms.width_calving_dict:
#                            width_calving = np.float64(pygem_prms.width_calving_dict[glacier_rgi_table.RGIId])
#                        else:
#                            width_calving = width_t0[glac_idx_t0[0]] * 1000                    
#                    # Volume loss [m3] due to frontal ablation
#                    frontalablation_volumeloss = (
#                            np.max([0, (frontalablation_k * waterdepth * height_calving)]) * width_calving)
#                    # Maximum volume loss is volume of bins with their bed elevation below sea level
#                    glac_idx_fa = np.where((glac_bin_bedelev < sea_level) & (glacier_area_t0 > 0))[0]
#                    frontalablation_volumeloss_max = glac_bin_volume[glac_idx_fa].sum()
#                    if frontalablation_volumeloss > frontalablation_volumeloss_max:
#                        frontalablation_volumeloss = frontalablation_volumeloss_max
#                        
#                    
#
#                    if debug:
#                        print('frontalablation_k:', frontalablation_k)
#                        print('width calving:', width_calving)
#                        print('frontalablation_volumeloss [m3]:', frontalablation_volumeloss)
#                        print('frontalablation_massloss [Gt]:', frontalablation_volumeloss * pygem_prms.density_water / 
#                              pygem_prms.density_ice / 10**9)
#                        print('frontalalabion_volumeloss_max [Gt]:', frontalablation_volumeloss_max * pygem_prms.density_water / 
#                              pygem_prms.density_ice / 10**9)
##                        print('glac_idx_fa:', glac_idx_fa)
##                        print('glac_bin_volume:', glac_bin_volume[0])
##                        print('glac_idx_fa[bin_count]:', glac_idx_fa[0])
##                        print('glac_bin_volume[glac_idx_fa[bin_count]]:', glac_bin_volume[glac_idx_fa[0]])
##                        print('glacier_area_t0[glac_idx_fa[bin_count]]:', glacier_area_t0[glac_idx_fa[0]])
##                        print('glac_bin_frontalablation:', glac_bin_frontalablation[glac_idx_fa[0], step])
#                    
#                    # Frontal ablation [mwe] in each bin
#                    bin_count = 0
#                    while (frontalablation_volumeloss > pygem_prms.tolerance) and (bin_count < len(glac_idx_fa)):
#                        # Sort heights to ensure it's universal (works with OGGM and Huss)
#                        heights_calving_sorted = np.argsort(heights[glac_idx_fa])
#                        calving_bin_idx = heights_calving_sorted[bin_count]
#                        # Check if entire bin removed or not
#                        if frontalablation_volumeloss >= glac_bin_volume[glac_idx_fa[calving_bin_idx]]:
#                            glac_bin_frontalablation[glac_idx_fa[calving_bin_idx], step] = (
#                                    glac_bin_volume[glac_idx_fa[calving_bin_idx]] / 
#                                    (glacier_area_t0[glac_idx_fa[calving_bin_idx]] * 10**6) 
#                                    * pygem_prms.density_ice / pygem_prms.density_water)   
#                        else:
#                            glac_bin_frontalablation[glac_idx_fa[calving_bin_idx], step] = (
#                                    frontalablation_volumeloss / (glacier_area_t0[glac_idx_fa[calving_bin_idx]] * 10**6)
#                                    * pygem_prms.density_ice / pygem_prms.density_water)
#                        frontalablation_volumeloss += (
#                                -1 * glac_bin_frontalablation[glac_idx_fa[calving_bin_idx], step] * pygem_prms.density_water 
#                                / pygem_prms.density_ice * glacier_area_t0[glac_idx_fa[calving_bin_idx]] * 10**6)                        
#                                                
#                        if debug:
#                            print('glacier idx:', glac_idx_fa[calving_bin_idx], 
#                                  'volume loss:', (glac_bin_frontalablation[glac_idx_fa[calving_bin_idx], step] * 
#                                  glacier_area_t0[glac_idx_fa[calving_bin_idx]] * pygem_prms.density_water / 
#                                  pygem_prms.density_ice * 10**6).round(0))
#                            print('remaining volume loss:', frontalablation_volumeloss, 'tolerance:', pygem_prms.tolerance)
#                        
#                        bin_count += 1         
#                            
#                    if debug:
#                        print('frontalablation_volumeloss remaining [m3]:', frontalablation_volumeloss)
#                        print('ice thickness:', icethickness_t0[glac_idx_fa[0]].round(0), 
#                              'waterdepth:', waterdepth.round(0), 
#                              'height calving front:', height_calving.round(0), 
#                              'width [m]:', (width_calving).round(0))   
#                        
#
#                # SURFACE TYPE
#                # Annual surface type [-]
#                # Annual climatic mass balance [m w.e.], which is used to determine the surface type
#                glac_bin_massbalclim_annual[:,year] = glac_bin_massbalclim[:,12*year:12*(year+1)].sum(1)
#                
#                if debug:
#                    print('glacier indices:', glac_idx_t0)
#                    print('glac_bin_massbalclim:', glac_bin_massbalclim[:,12*year:12*(year+1)].sum(1))
#                    print('glacier-wide climatic mass balance:', 
#                          (glac_bin_massbalclim_annual[:,year] * glacier_area_t0).sum() / glacier_area_t0.sum())
##                    print('Climatic mass balance:', glac_bin_massbalclim_annual[:,year].sum())
#                
##                # Compute the surface type for each bin
##                print('surface type2:', surfacetype[glac_idx_t0])
##                surfacetype, firnline_idx = surfacetypebinsannual(surfacetype, glac_bin_massbalclim_annual, year)
##                print('surface type3:', surfacetype[glac_idx_t0])
##                print('WHATS GOING ON WITH THIS??\n')
####                print(elev_bins)
#
#                # MASS REDISTRIBUTION
#                # Mass redistribution ignored for calibration and spinup years (glacier properties constant)
#                if (option_areaconstant == 1) or (year < pygem_prms.spinupyears) or (year < constantarea_years):
#                    glacier_area_t1 = glacier_area_t0
#                    icethickness_t1 = icethickness_t0
#                    width_t1 = width_t0                    
#                else:
#                    
#                    if debug:
#                        print('area is changing')
#                        
#                    # First, remove volume lost to frontal ablation
#                    #  changes to _t0 not _t1, since t1 will be done in the mass redistribution
#                    if glac_bin_frontalablation[:,step].max() > 0:
#                        # Frontal ablation loss [mwe]
#                        #  fa_change tracks whether entire bin is lost or not
#                        fa_change = abs(glac_bin_frontalablation[:, step] * pygem_prms.density_water / pygem_prms.density_ice
#                                        - icethickness_t0)
#                        fa_change[fa_change <= pygem_prms.tolerance] = 0
#                        
#                        if debug:
#                            bins_wfa = np.where(glac_bin_frontalablation[:,step] > 0)[0]
#                            print('glacier area t0:', glacier_area_t0[bins_wfa].round(3))
#                            print('ice thickness t0:', icethickness_t0[bins_wfa].round(1))
#                            print('frontalablation [m ice]:', (glac_bin_frontalablation[bins_wfa, step] * 
#                                  pygem_prms.density_water / pygem_prms.density_ice).round(1))
#                            print('frontal ablation [mice] vs icethickness:', fa_change[bins_wfa].round(1))
#                        
#                        # Check if entire bin is removed
#                        glacier_area_t0[np.where(fa_change == 0)[0]] = 0
#                        icethickness_t0[np.where(fa_change == 0)[0]] = 0
#                        width_t0[np.where(fa_change == 0)[0]] = 0
#                        # Otherwise, reduce glacier area such that glacier retreats and ice thickness remains the same
#                        #  A_1 = (V_0 - V_loss) / h_1,  units: A_1 = (m ice * km2) / (m ice) = km2
#                        glacier_area_t0[np.where(fa_change != 0)[0]] = (
#                                (glacier_area_t0[np.where(fa_change != 0)[0]] * 
#                                 icethickness_t0[np.where(fa_change != 0)[0]] - 
#                                 glacier_area_t0[np.where(fa_change != 0)[0]] * 
#                                 glac_bin_frontalablation[np.where(fa_change != 0)[0], step] * pygem_prms.density_water 
#                                 / pygem_prms.density_ice) / icethickness_t0[np.where(fa_change != 0)[0]])
#                        
#                        if debug:
#                            print('glacier area t1:', glacier_area_t0[bins_wfa].round(3))
#                            print('ice thickness t1:', icethickness_t0[bins_wfa].round(1))
#                    
#                    # Redistribute mass if glacier was not fully removed by frontal ablation
#                    if glacier_area_t0.max() > 0:
#                        # Mass redistribution according to Huss empirical curves
#                        glacier_area_t1, icethickness_t1, width_t1 = (
#                                massredistributionHuss(glacier_area_t0, icethickness_t0, width_t0, 
#                                                       glac_bin_massbalclim_annual, year, glac_idx_initial, 
#                                                       glacier_area_initial, heights,
#                                                       debug=False, 
#                                                       hindcast=hindcast))
#                        # update surface type for bins that have retreated
#                        surfacetype[glacier_area_t1 == 0] = 0
#                        
##                        print('\n-----')
##                        print(len(np.where(glacier_area_t0 > 0)[0]))
##                        print(len(np.where(glacier_area_t1 > 0)[0]))
##                        print(surfacetype)
###                        print(glacier_area_t1)
##                        print('-----\n')
##                        print('delete me')
##                        print(elev_bins)
#                        if year == 0:
#                            print('\n\nHERE IS WHERE I LEFT OFF CHECKING!\n\n')
#                        
#                        # update surface type for bins that have advanced 
#                        surfacetype[(surfacetype == 0) & (glacier_area_t1 != 0)] = (
#                                surfacetype[glacier_area_t0.nonzero()[0][0]])
#                    else:
#                        glacier_area_t1 = np.zeros(glacier_area_t0.shape)
#                        icethickness_t1 = np.zeros(glacier_area_t0.shape)
#                        width_t1 = np.zeros(glacier_area_t0.shape)
#                        surfacetype = np.zeros(glacier_area_t0.shape)
#                        
#                # Record glacier properties (area [km**2], thickness [m], width [km])
#                # if first year, record initial glacier properties (area [km**2], ice thickness [m ice], width [km])
#                if year == 0:
#                    glac_bin_area_annual[:,year] = glacier_area_initial
#                    glac_bin_icethickness_annual[:,year] = icethickness_initial
#                    glac_bin_width_annual[:,year] = width_initial
#                # record the next year's properties as well
#                # 'year + 1' used so the glacier properties are consistent with mass balance computations
#                glac_bin_icethickness_annual[:,year + 1] = icethickness_t1
#                glac_bin_area_annual[:,year + 1] = glacier_area_t1
#                glac_bin_width_annual[:,year + 1] = width_t1
#                # Update glacier properties for the mass balance computations
#                icethickness_t0 = icethickness_t1.copy()
#                glacier_area_t0 = glacier_area_t1.copy()
#                width_t0 = width_t1.copy()   
#                glac_bin_surfacetype_annual[:,year] = surfacetype
                
        # Example of modularity
#        if self.use_refreeze:
#            mb += self._refreeze_term(heights, year)
            
#        # Example of how to store variables from within the other functions (ex. mass balance components)
#        self.diag_df.loc[year, 'mb'] = np.average(mb)
        
        mb = heights * 0.

        return mb  # m of ice per second
        
        #%%
        
#        # Example of how to store variables from within the other functions (ex. mass balance components)
#        self.diag_df = pd.DataFrame()
        
#    # Example of what could be done!
#    def _refreeze_term(self, heights, year):
#        
#        return 0
    
    
    def _surfacetypebinsinitial(self, elev_bins):
        """
        Define initial surface type according to median elevation such that the melt can be calculated over snow or ice.
        
        Convention: (0 = off-glacier, 1 = ice, 2 = snow, 3 = firn, 4 = debris).
        Function options: 1 =
        
        Function options specified in pygem_pygem_prms.py:
        - option_surfacetype_initial
            > 1 (default) - use median elevation to classify snow/firn above the median and ice below
            > 2 - use mean elevation instead
        - option_surfacetype_firn = 1
            > 1 (default) - firn is included
            > 0 - firn is not included
        - option_surfacetype_debris = 0
            > 0 (default) - debris cover is not included
            > 1 - debris cover is included
        
        To-do list
        ----------
        Add option_surfacetype_initial to specify an AAR ratio and apply this to estimate initial conditions
        
        Parameters
        ----------
        elev_bins : np.ndarray
            Elevation bins [masl]
        Returns
        -------
        surfacetype : np.ndarray
            Updated surface type for each elevation bin
        firnline_idx : int
            Firn line index
        """        
        surfacetype = np.zeros(self.glacier_area_t0.shape)
        # Option 1 - initial surface type based on the median elevation
        if pygem_prms.option_surfacetype_initial == 1:
            surfacetype[(elev_bins < self.glacier_rgi_table.loc['Zmed']) & (self.glacier_area_t0 > 0)] = 1
            surfacetype[(elev_bins >= self.glacier_rgi_table.loc['Zmed']) & (self.glacier_area_t0 > 0)] = 2
        # Option 2 - initial surface type based on the mean elevation
        elif pygem_prms.option_surfacetype_initial ==2:
            surfacetype[(elev_bins < self.glacier_rgi_table['Zmean']) & (self.glacier_area_t0 > 0)] = 1
            surfacetype[(elev_bins >= self.glacier_rgi_table['Zmean']) & (self.glacier_area_t0 > 0)] = 2
        else:
            print("This option for 'option_surfacetype' does not exist. Please choose an option that exists. "
                  + "Exiting model run.\n")
            exit()
        # Compute firnline index
        try:
            # firn in bins >= firnline_idx
            firnline_idx = np.where(surfacetype==2)[0][0]
        except:
            # avoid errors if there is no firn, i.e., the entire glacier is melting
            firnline_idx = np.where(surfacetype!=0)[0][-1]
        # If firn is included, then specify initial firn conditions
        if pygem_prms.option_surfacetype_firn == 1:
            surfacetype[surfacetype == 2] = 3
            #  everything initially considered snow is considered firn, i.e., the model initially assumes there is no snow 
            #  on the surface anywhere.
        if pygem_prms.option_surfacetype_debris == 1:
            print("Need to code the model to include debris. This option does not currently exist.  Please choose an option"
                  + " that exists.\nExiting the model run.")
            exit()
            # One way to include debris would be to simply have debris cover maps and state that the debris retards melting 
            # as a fraction of melt.  It could also be DDF_debris as an additional calibration tool. Lastly, if debris 
            # thickness maps are generated, could be an exponential function with the DDF_ice as a term that way DDF_debris 
            # could capture the spatial variations in debris thickness that the maps supply.
        return surfacetype, firnline_idx
    
    def _surfacetypeDDFdict(self, modelparameters, 
                            option_surfacetype_firn=pygem_prms.option_surfacetype_firn,
                            option_ddf_firn=pygem_prms.option_ddf_firn):
        """
        Create a dictionary of surface type and its respective DDF.
        
        Convention: [0=off-glacier, 1=ice, 2=snow, 3=firn, 4=debris]
        
        modelparameters[lr_gcm, lr_glac, prec_factor, prec_grad, DDF_snow, DDF_ice, T_snow]
        
        To-do list
        ----------
        Add option_surfacetype_initial to specify an AAR ratio and apply this to estimate initial conditions
        
        Parameters
        ----------
        modelparameters : pd.Series
            Model parameters (lrgcm, lrglac, precfactor, precgrad, ddfsnow, ddfice, tempsnow, tempchange)
            Order of model parameters should not be changed as the run mass balance script relies on this order
        option_surfacetype_firn : int
            Option to include or exclude firn (specified in pygem_pygem_prms.py)
        option_ddf_firn : int
            Option for the degree day factor of firn to be the average of snow and ice or a different value
        Returns
        -------
        surfacetype_ddf_dict : dictionary
            Dictionary relating the surface types with their respective degree day factors
        """        
        surfacetype_ddf_dict = {
    #            0: 0,
                0: modelparameters[4],
                1: modelparameters[5],
                2: modelparameters[4]}
        if option_surfacetype_firn == 1:
            if option_ddf_firn == 0:
                surfacetype_ddf_dict[3] = modelparameters[4]
            elif option_ddf_firn == 1:
                surfacetype_ddf_dict[3] = np.mean([modelparameters[4],modelparameters[5]])
        if pygem_prms.option_surfacetype_debris == 1:
            surfacetype_ddf_dict[4] = pygem_prms.DDF_debris
        return surfacetype_ddf_dict