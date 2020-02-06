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
print('\n\nMOVE ANNUALWEIGHTEDMEAN_ARRAY function elsewhere - utilities?\n\n')
def annualweightedmean_array(var, dates_table):
    """
    Calculate annual mean of variable according to the timestep.
    
    Monthly timestep will group every 12 months, so starting month is important.
    
    Parameters
    ----------
    var : np.ndarray
        Variable with monthly or daily timestep
    dates_table : pd.DataFrame
        Table of dates, year, month, daysinmonth, wateryear, and season for each timestep
    Returns
    -------
    var_annual : np.ndarray
        Annual weighted mean of variable
    """        
    if pygem_prms.timestep == 'monthly':
        dayspermonth = dates_table['daysinmonth'].values.reshape(-1,12)
        #  creates matrix (rows-years, columns-months) of the number of days per month
        daysperyear = dayspermonth.sum(axis=1)
        #  creates an array of the days per year (includes leap years)
        weights = (dayspermonth / daysperyear[:,np.newaxis]).reshape(-1)
        #  computes weights for each element, then reshapes it from matrix (rows-years, columns-months) to an array,
        #  where each column (each monthly timestep) is the weight given to that specific month
        var_annual = (var*weights[np.newaxis,:]).reshape(-1,12).sum(axis=1).reshape(-1,daysperyear.shape[0])
        #  computes matrix (rows - bins, columns - year) of weighted average for each year
        #  explanation: var*weights[np.newaxis,:] multiplies each element by its corresponding weight; .reshape(-1,12) 
        #    reshapes the matrix to only have 12 columns (1 year), so the size is (rows*cols/12, 12); .sum(axis=1) 
        #    takes the sum of each year; .reshape(-1,daysperyear.shape[0]) reshapes the matrix back to the proper 
        #    structure (rows - bins, columns - year)
        # If averaging a single year, then reshape so it returns a 1d array
        if var_annual.shape[1] == 1:
            var_annual = var_annual.reshape(var_annual.shape[0])
    elif pygem_prms.timestep == 'daily':
        print('\nError: need to code the groupbyyearsum and groupbyyearmean for daily timestep.'
              'Exiting the model run.\n')
        exit()
    return var_annual

#%%
class PyGEMMassBalance(MassBalanceModel):
    """Mass-balance computed from the Python Glacier Evolution Model.

    This mass balance accounts for ablation, accumulation, and refreezing.

    This class implements the MassBalanceModel interface so that the dynamical model can use it.
    """

#    def __init__(self, modelparameters, glacier_rgi_table, glacier_area_initial, icethickness_initial, 
#                       width_initial, heights, glacier_gcm_temp, glacier_gcm_tempstd, glacier_gcm_prec, 
#                       glacier_gcm_elev, glacier_gcm_lrgcm, glacier_gcm_lrglac, dates_table, option_areaconstant=0, 
#                       constantarea_years=pygem_prms.constantarea_years, frontalablation_k=None, 
#                       debug=False, debug_refreeze=False, hindcast=0, 
##                       use_refreeze=True
#                       ):
    def __init__(self, modelparameters, glacier_rgi_table, glacier_gcm_temp, glacier_gcm_tempstd, glacier_gcm_prec,
                       glacier_gcm_elev, glacier_gcm_lrgcm, glacier_gcm_lrglac, dates_table, option_areaconstant=0,
                       constantarea_years=pygem_prms.constantarea_years, frontalablation_k=None, 
                       debug=False, debug_refreeze=False, hindcast=0, 
                       fls=None, fl_id=0, glacier_area_initial=None, icethickness_initial=None, width_initial=None,
                       heights=None,
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
        
#        current_flowline = fls[fl_id]
#        icethickness_t0 = current_flowline.thick
        
        # Can pass all of them here
        
        
        # Glacier data
        self.modelparameters = modelparameters
        self.glacier_rgi_table = glacier_rgi_table
        # glac_idx_initial is used with advancing glaciers to ensure no bands are added in a discontinuous section
        self.glac_idx_initial = fls[fl_id].thick.nonzero()
        self.icethickness_initial = fls[fl_id].thick
        self.width_initial = fls[fl_id].widths_m / 1000
        self.glacier_area_initial = fls[fl_id].widths_m / 1000 * fls[fl_id].dx_meter / 1000
        print('initial glacier area:', self.glacier_area_initial.sum())
        
        self.heights = fls[fl_id].surface_h
        
        # Climate data
        self.dates_table = dates_table
        self.glacier_gcm_temp = glacier_gcm_temp
        self.glacier_gcm_tempstd = glacier_gcm_tempstd
        self.glacier_gcm_prec = glacier_gcm_prec 
        self.glacier_gcm_elev = glacier_gcm_elev
        self.glacier_gcm_lrgcm = glacier_gcm_lrgcm
        self.glacier_gcm_lrglac = glacier_gcm_lrglac
        
        self.debug_refreeze = debug_refreeze

        if debug:
            print('\n\nDEBUGGING MASS BALANCE FUNCTION\n\n')
            
            
        # Variables to store (consider storing in xarray)
        nbins = self.icethickness_initial.shape[0]
        nmonths = self.glacier_gcm_temp.shape[0]
        nyears = len(self.dates_table.wateryear.unique())
            
        self.bin_temp = np.zeros((nbins,nmonths))
        self.bin_prec = np.zeros((nbins,nmonths))
        self.bin_acc = np.zeros((nbins,nmonths))
        self.bin_refreezepotential = np.zeros((nbins,nmonths))
        self.bin_refreeze = np.zeros((nbins,nmonths))
        self.bin_meltglac = np.zeros((nbins,nmonths))
        self.bin_meltsnow = np.zeros((nbins,nmonths))
        self.bin_melt = np.zeros((nbins,nmonths))
        self.bin_snowpack = np.zeros((nbins,nmonths))
        self.snowpack_remaining = np.zeros((nbins, nmonths))
        self.glac_bin_refreeze = np.zeros((nbins,nmonths))
        self.glac_bin_melt = np.zeros((nbins,nmonths))
        self.glac_bin_frontalablation = np.zeros((nbins,nmonths))
        self.glac_bin_snowpack = np.zeros((nbins,nmonths))
        self.glac_bin_massbalclim = np.zeros((nbins,nmonths))
        self.glac_bin_massbalclim_annual = np.zeros((nbins,nyears))
        self.glac_bin_surfacetype_annual = np.zeros((nbins,nyears))
        self.glac_bin_icethickness_annual = np.zeros((nbins,nyears+1))
#        self.glac_bin_area_annual = np.zeros((nbins, nyears + 1))
#        self.glac_bin_width_annual = np.zeros((nbins, nyears + 1))
        
        self.offglac_bin_prec = np.zeros((nbins,nmonths))
        self.offglac_bin_melt = np.zeros((nbins,nmonths))
        self.offglac_bin_refreeze = np.zeros((nbins,nmonths))
        self.offglac_bin_snowpack = np.zeros((nbins,nmonths))
        self.offglac_bin_area_annual = np.zeros((nbins,nyears+1))
        
        self.dayspermonth = self.dates_table['daysinmonth'].values
        self.surfacetype_ddf = np.zeros((nbins))
        
        # sSrface type DDF dictionary (manipulate this function for calibration or for each glacier)
        self.surfacetype_ddf_dict = self._surfacetypeDDFdict(self.modelparameters)   
        
        # Refreezing specific layers
        if pygem_prms.option_refreezing == 1:
            # Refreezing layers density, volumetric heat capacity, and thermal conductivity
            self.rf_dens_expb = (pygem_prms.rf_dens_bot / pygem_prms.rf_dens_top)**(1/(pygem_prms.rf_layers-1))
            self.rf_layers_dens = np.array([pygem_prms.rf_dens_top * self.rf_dens_expb**x 
                                            for x in np.arange(0,pygem_prms.rf_layers)])
            self.rf_layers_ch = ((1 - self.rf_layers_dens/1000) * pygem_prms.ch_air + self.rf_layers_dens/1000 * 
                                 pygem_prms.ch_ice)
            self.rf_layers_k = ((1 - self.rf_layers_dens/1000) * pygem_prms.k_air + self.rf_layers_dens/1000 * 
                                pygem_prms.k_ice)
            self.refr = np.zeros(nbins)                             # refreeze in each bin
            self.rf_cold = np.zeros(nbins)                          # refrezee cold content or "potential" refreeze
            
            self.te_rf = np.zeros((pygem_prms.rf_layers,nbins,nmonths))     # layer temp of each elev bin for present time step
            self.tl_rf = np.zeros((pygem_prms.rf_layers,nbins,nmonths))     # layer temp of each elev bin for previous time step
        
        # Sea level for marine-terminating glaciers
        self.sea_level = 0
        rgi_region = int(glacier_rgi_table.RGIId.split('-')[1].split('.')[0])
        if frontalablation_k == None:
             self.frontalablation_k0 = pygem_prms.frontalablation_k0dict[rgi_region]

        # Run mass balance only on pixels that have an ice thickness (some glaciers do not have an ice thickness)
        #  otherwise causes indexing errors causing the model to fail
#        if self.icethickness_t0.max() > 0:    
#            if pygem_prms.option_adjusttemp_surfelev == 1:
#                # ice thickness initial is used to adjust temps to changes in surface elevation
#                self.icethickness_initial = self.icethickness_t0.copy()
#                self.icethickness_initial[0:self.icethickness_initial.nonzero()[0][0]] = (
#                        self.icethickness_initial[self.icethickness_initial.nonzero()[0][0]])
#                print('this indexing is wrong!  Need to find min_elevbin')
#                #  bins that advance need to have an initial ice thickness; otherwise, the temp adjustment will be by
#                #  ice thickness - 0, which is wrong; since advancing bins take thickness of the previous bin, set the
#                #  initial ice thickness of all bins below the terminus to the ice thickness at the terminus.
            
            
            
    def get_annual_mb(self, heights, year=None, fls=None, fl_id=None, 
                      debug=False, option_areaconstant=0):
        """FIXED FORMAT FOR THE FLOWLINE MODEL
        
        Returns annual climatic mass balance
        
        Parameters
        ----------
        heights : np.array
            elevation bins
        year : int
            year starting with 0 to the number of years in the study
        
        """
        fl = fls[fl_id]
        heights = fl.surface_h
        icethickness_t0 = fl.thick
        glacier_area_t0 = fl.widths_m / 1000 * fl.dx_meter / 1000
        glacier_area_t0[icethickness_t0 == 0] = 0
        icethickness_initial = self.icethickness_initial
        glacier_area_initial = self.glacier_area_initial
        
        nbins = heights.shape[0]
        nmonths = self.glacier_gcm_temp.shape[0]
        
        # Local variables
        bin_precsnow = np.zeros((nbins,nmonths))
        
        # Refreezing specific layers
        if pygem_prms.option_refreezing == 1 and year == 0:
            self.te_rf[:,:,0] = 0     # layer temp of each elev bin for present time step
            self.tl_rf[:,:,0] = 0     # layer temp of each elev bin for previous time step
        
        
        if icethickness_t0.max() > 0:    
            if debug:
                print(year, 'max ice thickness [m]:', icethickness_t0.max()) 
            
            # Glacier indices
            glac_idx_t0 = icethickness_t0.nonzero()[0]
            
            # Surface type [0=off-glacier, 1=ice, 2=snow, 3=firn, 4=debris]
            if year == 0:
                self.surfacetype, self.firnline_idx = self._surfacetypebinsinitial(self.heights)
            self.glac_bin_surfacetype_annual[:,year] = self.surfacetype
            
            # Off-glacier area and indices
            if option_areaconstant == 0:
                self.offglac_bin_area_annual[:,year] = glacier_area_initial - glacier_area_t0
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
                         self.glacier_rgi_table.loc[pygem_prms.option_elev_ref_downscale])[:,np.newaxis] + 
                                                self.modelparameters[7])
                # Option to adjust air temperature based on changes in surface elevation
                #  note: OGGM automatically updates the bin elevation, so this step is not needed
                if pygem_prms.option_adjusttemp_surfelev == 1 and pygem_prms.hyps_data in ['Huss', 'Farinotti']:
                    # T_air = T_air + lr_glac * (icethickness_present - icethickness_initial)
                    self.bin_temp[:,12*year:12*(year+1)] = (self.bin_temp[:,12*year:12*(year+1)] + 
                                                       self.glacier_gcm_lrglac[12*year:12*(year+1)] * 
                                                       (icethickness_t0 - icethickness_initial)[:,np.newaxis])
                # PRECIPITATION/ACCUMULATION: Downscale the precipitation (liquid and solid) to each bin
                if pygem_prms.option_prec2bins == 1:                    
                    # Precipitation using precipitation factor and precipitation gradient
                    #  P_bin = P_gcm * prec_factor * (1 + prec_grad * (z_bin - z_ref))
                    bin_precsnow[:,12*year:12*(year+1)] = (self.glacier_gcm_prec[12*year:12*(year+1)] * 
                            self.modelparameters[2] * (1 + self.modelparameters[3] * (heights - 
                            self.glacier_rgi_table.loc[pygem_prms.option_elev_ref_downscale]))[:,np.newaxis])
                # Option to adjust prec of uppermost 25% of glacier for wind erosion and reduced moisture content
                if pygem_prms.option_preclimit == 1:
                    # Elevation range based on all flowlines
                    raw_min_elev = []
                    raw_max_elev = []
                    for raw_fl in fls:
                        if len(raw_fl.surface_h[raw_fl.thick > 0]):
                            raw_min_elev.append(raw_fl.surface_h[raw_fl.thick > 0].min())
                            raw_max_elev.append(raw_fl.surface_h[raw_fl.thick > 0].max())
                    elev_range = np.max(raw_max_elev) - np.min(raw_min_elev)
                    elev_75 = np.min(raw_min_elev) + 0.75 * (elev_range)
                    
                    # If elevation range > 1000 m, apply corrections to uppermost 25% of glacier (Huss and Hock, 2015)
                    if elev_range > 1000:
                        # Indices of upper 25%
                        glac_idx_upper25 = glac_idx_t0[heights[glac_idx_t0] >= elev_75]
                        # Exponential decay according to elevation difference from the 75% elevation
                        #  prec_upper25 = prec * exp(-(elev_i - elev_75%)/(elev_max- - elev_75%))
                        # height at 75% of the elevation
                        height_75 = heights[glac_idx_upper25].min()
                        glac_idx_75 = np.where(heights == height_75)[0][0]
                        # exponential decay
                        bin_precsnow[glac_idx_upper25,12*year:12*(year+1)] = (
                                bin_precsnow[glac_idx_75,12*year:12*(year+1)] * 
                                np.exp(-1*(heights[glac_idx_upper25] - height_75) / 
                                       (heights[glac_idx_upper25].max() - heights[glac_idx_upper25].min()))
                                [:,np.newaxis])
                        # Precipitation cannot be less than 87.5% of the maximum accumulation elsewhere on the glacier
                        for month in range(0,12):
                            (bin_precsnow
                                [glac_idx_upper25[(bin_precsnow[glac_idx_upper25,month] < 0.875 * 
                                 bin_precsnow[glac_idx_t0,month].max()) & 
                                (bin_precsnow[glac_idx_upper25,month] != 0)], month]) = (
                                               0.875 * bin_precsnow[glac_idx_t0,month].max())
                                
                # Separate total precipitation into liquid (bin_prec) and solid (bin_acc)
                if pygem_prms.option_accumulation == 1:
                    # if temperature above threshold, then rain
                    (self.bin_prec[:,12*year:12*(year+1)]
                                  [self.bin_temp[:,12*year:12*(year+1)] > self.modelparameters[6]]) = (
                        bin_precsnow[:,12*year:12*(year+1)]
                                         [self.bin_temp[:,12*year:12*(year+1)] > self.modelparameters[6]])
                    # if temperature below threshold, then snow
                    (self.bin_acc[:,12*year:12*(year+1)]
                                 [self.bin_temp[:,12*year:12*(year+1)] <= self.modelparameters[6]]) = (
                        bin_precsnow[:,12*year:12*(year+1)]
                                         [self.bin_temp[:,12*year:12*(year+1)] <= self.modelparameters[6]])
                elif pygem_prms.option_accumulation == 2:
                    # If temperature between min/max, then mix of snow/rain using linear relationship between min/max
                    self.bin_prec[:,12*year:12*(year+1)] = (
                            (1/2 + (self.bin_temp[:,12*year:12*(year+1)] - 
                             self.modelparameters[6]) / 2) * bin_precsnow[:,12*year:12*(year+1)])
                    self.bin_acc[:,12*year:12*(year+1)] = (
                            bin_precsnow[:,12*year:12*(year+1)] - self.bin_prec[:,12*year:12*(year+1)])
                    # If temperature above maximum threshold, then all rain
                    (self.bin_prec[:,12*year:12*(year+1)]
                                  [self.bin_temp[:,12*year:12*(year+1)] > self.modelparameters[6] + 1]) = (
                            bin_precsnow[:,12*year:12*(year+1)]
                                             [self.bin_temp[:,12*year:12*(year+1)] > self.modelparameters[6] + 1])
                    (self.bin_acc[:,12*year:12*(year+1)]
                                 [self.bin_temp[:,12*year:12*(year+1)] > self.modelparameters[6] + 1]) = 0
                    # If temperature below minimum threshold, then all snow
                    (self.bin_acc[:,12*year:12*(year+1)]
                                 [self.bin_temp[:,12*year:12*(year+1)] <= self.modelparameters[6] - 1]) = (
                            bin_precsnow[:,12*year:12*(year+1)]
                                             [self.bin_temp[:,12*year:12*(year+1)] <= self.modelparameters[6] - 1])
                    (self.bin_prec[:,12*year:12*(year+1)]
                                  [self.bin_temp[:,12*year:12*(year+1)] <= self.modelparameters[6] - 1]) = 0
                
                # ENTER MONTHLY LOOP (monthly loop required since surface type changes)
                for month in range(0,12):
                    # Step is the position as a function of year and month, which improves readability
                    step = 12*year + month
                    
                    # ACCUMULATION, MELT, REFREEZE, AND CLIMATIC MASS BALANCE
                    # Snowpack [m w.e.] = snow remaining + new snow
                    if step > 0:
                        self.bin_snowpack[:,step] = self.snowpack_remaining[:,step-1] + self.bin_acc[:,step]
                    else:
                        self.bin_snowpack[:,step] = self.bin_acc[:,step]
                    
                    # MELT [m w.e.]
                    # energy available for melt [degC day]   
                    if pygem_prms.option_ablation == 1:
                        # option 1: energy based on monthly temperature
                        melt_energy_available = self.bin_temp[:,step]*self.dayspermonth[step]
                        melt_energy_available[melt_energy_available < 0] = 0
                    elif pygem_prms.option_ablation == 2:
                        # option 2: monthly temperature superimposed with daily temperature variability 
                        # daily temperature variation in each bin for the monthly timestep
                        bin_tempstd_daily = np.repeat(
                                np.random.normal(loc=0, scale=self.glacier_gcm_tempstd[step], 
                                                 size=self.dayspermonth[step])
                                .reshape(1,self.dayspermonth[step]), heights.shape[0], axis=0)
                        # daily temperature in each bin for the monthly timestep
                        bin_temp_daily = self.bin_temp[:,step][:,np.newaxis] + bin_tempstd_daily
                        # remove negative values
                        bin_temp_daily[bin_temp_daily < 0] = 0
                        # Energy available for melt [degC day] = sum of daily energy available
                        melt_energy_available = bin_temp_daily.sum(axis=1)
                    # SNOW MELT [m w.e.]
                    self.bin_meltsnow[:,step] = self.surfacetype_ddf_dict[2] * melt_energy_available
                    # snow melt cannot exceed the snow depth
                    self.bin_meltsnow[self.bin_meltsnow[:,step] > self.bin_snowpack[:,step], step] = (
                            self.bin_snowpack[self.bin_meltsnow[:,step] > self.bin_snowpack[:,step], step])
                    # GLACIER MELT (ice and firn) [m w.e.]
                    # energy remaining after snow melt [degC day]
                    melt_energy_available = (
                            melt_energy_available - self.bin_meltsnow[:,step] / self.surfacetype_ddf_dict[2])
                    # remove low values of energy available caused by rounding errors in the step above
                    melt_energy_available[abs(melt_energy_available) < pygem_prms.tolerance] = 0
                    # DDF based on surface type [m w.e. degC-1 day-1]
                    for surfacetype_idx in self.surfacetype_ddf_dict: 
                        self.surfacetype_ddf[self.surfacetype == surfacetype_idx] = (
                                self.surfacetype_ddf_dict[surfacetype_idx])
                    if pygem_prms.option_surfacetype_debris == 1:
                        print('\n\nLOAD THE MELTFACTOR DATASET over areas that are not firn\n\n')
                        
                        if year == 0 and month == 0:
                            print('\nDELETE ME\n surfacetype_ddf[glac_idx]:', self.surfacetype_ddf[glac_idx_t0])
                        
                    self.bin_meltglac[glac_idx_t0,step] = (
                            self.surfacetype_ddf[glac_idx_t0] * melt_energy_available[glac_idx_t0])
                    # TOTAL MELT (snow + glacier)
                    #  off-glacier need to include melt of refreeze because there are no glacier dynamics,
                    #  but on-glacier do not need to account for this (simply assume refreeze has same surface type)
                    self.bin_melt[:,step] = self.bin_meltglac[:,step] + self.bin_meltsnow[:,step]  
                    
                    # REFREEZING
                    if pygem_prms.option_refreezing == 1:
                        if step > 0:
                            self.tl_rf[:,:,step] = self.tl_rf[:,:,step-1]
                            self.te_rf[:,:,step] = self.te_rf[:,:,step-1]
                        
                        # Refreeze based on heat conduction approach (Huss and Hock 2015)    
                        # refreeze time step (s)
                        rf_dt = 3600 * 24 * self.dates_table.loc[step,'daysinmonth'] / pygem_prms.rf_dsc
                        
                        if pygem_prms.option_rf_limit_meltsnow == 1:
                            bin_meltlimit = self.bin_meltsnow.copy()
                        else:
                            bin_meltlimit = self.bin_melt.copy()

                        # Debug lowest bin                            
                        if self.debug_refreeze:
                            gidx_debug = np.where(heights == heights[glac_idx_t0].min())[0]
                            
                        # Loop through each elevation bin of glacier
                        for nbin, gidx in enumerate(glac_idx_t0):
                                    
                            # COMPUTE HEAT CONDUCTION - BUILD COLD RESERVOIR                            
                            # If no melt, then build up cold reservoir (compute heat conduction)
                            if self.bin_melt[gidx,step] < pygem_prms.rf_meltcrit:
                                
                                if self.debug_refreeze and gidx == gidx_debug and step < 12:
                                    print('\nMonth ' + str(self.dates_table.loc[step,'month']), 
                                          'Computing heat conduction')
                                
                                # Set refreeze equal to 0
                                self.refr[gidx] = 0
                                # Loop through multiple iterations to converge on a solution
                                #  -> this will loop through 0, 1, 2
                                for h in np.arange(0, pygem_prms.rf_dsc):    
                                    # Compute heat conduction in layers (loop through rows)
                                    #  go from 1 to rf_layers-1 to avoid indexing errors with "j-1" and "j+1"
                                    #  "j+1" is set to zero, which is fine for temperate glaciers but inaccurate for 
                                    #  cold/polythermal glaciers
                                    for j in np.arange(1, pygem_prms.rf_layers-1):
                                        # Assume temperature of first layer equals air temperature
                                        #  assumption probably wrong, but might still work at annual average
                                        # Since next line uses tl_rf for all calculations, set tl_rf[0] to present mean 
                                        #  monthly air temperature to ensure the present calculations are done with the 
                                        #  present time step's air temperature
                                        self.tl_rf[0, gidx,step] = self.bin_temp[gidx,step]
                                        # Temperature for each layer
                                        self.te_rf[j,gidx,step] = (self.tl_rf[j,gidx,step] + 
                                             rf_dt * self.rf_layers_k[j] / self.rf_layers_ch[j] / pygem_prms.rf_dz**2 * 
                                             0.5 * ((self.tl_rf[j-1,gidx,step] - self.tl_rf[j,gidx,step]) - 
                                                     (self.tl_rf[j,gidx,step] - self.tl_rf[j+1,gidx,step]))) 
                                        # Update previous time step
                                        self.tl_rf[:,gidx,step] = self.te_rf[:,gidx,step]
                                   
                                if self.debug_refreeze and gidx == gidx_debug and step < 12:
                                    print('tl_rf:', ["{:.2f}".format(x) for x in self.tl_rf[:,gidx,step]])
                                            
                            # COMPUTE REFREEZING - TAP INTO "COLD RESERVOIR" or potential refreezing  
                            else:
                                
                                if self.debug_refreeze and gidx == gidx_debug and step < 12:
                                    print('\nMonth ' + str(self.dates_table.loc[step,'month']), 'Computing refreeze')
                                
                                # Refreezing over firn surface
                                if (self.surfacetype[gidx] == 2) or (self.surfacetype[gidx] == 3):
                                    nlayers = pygem_prms.rf_layers-1
                                # Refreezing over ice surface
                                else:
                                    # Approximate number of layers of snow on top of ice
                                    smax = np.round((self.bin_snowpack[gidx,step] / (self.rf_layers_dens[0] / 1000) + 
                                                     pygem_prms.pp) / pygem_prms.rf_dz, 0)
                                    # if there is very little snow on the ground (SWE > 0.06 m for pp=0.3), 
                                    #  then still set smax (layers) to 1
                                    if self.bin_snowpack[gidx,step] > 0 and smax == 0:
                                        smax=1
                                    # if no snow on the ground, then set to rf_cold to NoData value
                                    if smax == 0:
                                        self.rf_cold[gidx] = 0
                                    # if smax greater than the number of layers, set to max number of layers minus 1
                                    if smax > pygem_prms.rf_layers - 1:
                                        smax = pygem_prms.rf_layers - 1
                                    nlayers = int(smax)
                        
                                # Compute potential refreeze, "cold reservoir", from temperature in each layer 
                                # only calculate potential refreezing first time it starts melting each year
                                if self.rf_cold[gidx] == 0 and self.tl_rf[:,gidx,step].min() < 0:
                                    
                                    if self.debug_refreeze and gidx == gidx_debug and step < 12:
                                        print('calculating potential refreeze from ' + str(nlayers) + ' layers')
                                    
                                    for j in np.arange(0,nlayers):
                                        j += 1
                                        # units: (degC) * (J K-1 m-3) * (m) * (kg J-1) * (m3 kg-1)
                                        rf_cold_layer = (self.tl_rf[j,gidx,step] * self.rf_layers_ch[j] * pygem_prms.rf_dz / 
                                                         pygem_prms.Lh_rf / pygem_prms.density_water)
                                        self.rf_cold[gidx] -= rf_cold_layer
                                        
                                        if self.debug_refreeze and gidx == gidx_debug and step < 12:
                                            print('j:', j, 'tl_rf @ j:', np.round(self.tl_rf[j,gidx,step],2), 
                                                           'ch @ j:', np.round(self.rf_layers_ch[j],2), 
                                                           'rf_cold_layer @ j:', np.round(rf_cold_layer,2),
                                                           'rf_cold @ j:', np.round(self.rf_cold[gidx],2))
                                    
                                    if self.debug_refreeze and gidx == gidx_debug and step < 12:
                                        print('rf_cold:', np.round(self.rf_cold[gidx],2))
                                        
                                # Compute refreezing
                                # If melt and liquid prec < potential refreeze, then refreeze all melt and liquid prec
                                if (bin_meltlimit[gidx,step] + self.bin_prec[gidx,step]) < self.rf_cold[gidx]:
                                    self.refr[gidx] = bin_meltlimit[gidx,step] + self.bin_prec[gidx,step]
                                # otherwise, refreeze equals the potential refreeze
                                elif self.rf_cold[gidx] > 0:
                                    self.refr[gidx] = self.rf_cold[gidx]
                                else:
                                    self.refr[gidx] = 0

                                # Track the remaining potential refreeze                                    
                                self.rf_cold[gidx] -= (bin_meltlimit[gidx,step] + self.bin_prec[gidx,step])
                                # if potential refreeze consumed, set to 0 and set temperature to 0 (temperate firn)
                                if self.rf_cold[gidx] < 0:
                                    self.rf_cold[gidx] = 0
                                    self.tl_rf[:,gidx,step] = 0
                                    
                            # Record refreeze
                            self.bin_refreeze[gidx,step] = self.refr[gidx]
                            
                            if self.debug_refreeze and step < 12 and gidx == gidx_debug:
                                print('Month ' + str(self.dates_table.loc[step,'month']),
                                      'Rf_cold remaining:', np.round(self.rf_cold[gidx],2),
                                      'Snow depth:', np.round(self.bin_snowpack[glac_idx_t0[nbin],step],2),
                                      'Snow melt:', np.round(self.bin_meltsnow[glac_idx_t0[nbin],step],2),
                                      'Rain:', np.round(self.bin_prec[glac_idx_t0[nbin],step],2),
                                      'Rfrz:', np.round(self.bin_refreeze[gidx,step],2)
                                      )             
                                
                    elif pygem_prms.option_refreezing == 2:
                        # Refreeze based on annual air temperature (Woodward etal. 1997)
                        #  R(m) = (-0.69 * Tair + 0.0096) * 1 m / 100 cm
                        # calculate annually and place potential refreeze in user defined month
                        if self.dates_table.loc[step,'month'] == pygem_prms.rf_month:                        
                            bin_temp_annual = annualweightedmean_array(self.bin_temp[:,12*year:12*(year+1)], 
                                                                       self.dates_table.iloc[12*year:12*(year+1),:])
                            bin_refreezepotential_annual = (-0.69 * bin_temp_annual + 0.0096) * 1/100
                            # Remove negative refreezing values
                            bin_refreezepotential_annual[bin_refreezepotential_annual < 0] = 0
                            self.bin_refreezepotential[:,step] = bin_refreezepotential_annual
                            # Reset refreeze potential every year
                            if self.bin_refreezepotential[:,step].max() > 0:
                                self.refreeze_potential = self.bin_refreezepotential[:,step]

                        if self.debug_refreeze and step < 12:
                            print('Month' + str(self.dates_table.loc[step,'month']),
                                  'Refreeze potential:', np.round(self.refreeze_potential[glac_idx_t0[0]],3),
                                  'Snow depth:', np.round(self.bin_snowpack[glac_idx_t0[0],step],2),
                                  'Snow melt:', np.round(self.bin_meltsnow[glac_idx_t0[0],step],2),
                                  'Rain:', np.round(self.bin_prec[glac_idx_t0[0],step],2))

                        # Refreeze [m w.e.]
                        #  refreeze cannot exceed rain and melt (snow & glacier melt)
                        self.bin_refreeze[:,step] = self.bin_meltsnow[:,step] + self.bin_prec[:,step]
                        # refreeze cannot exceed snow depth
                        self.bin_refreeze[self.bin_refreeze[:,step] > self.bin_snowpack[:,step], step] = (
                                self.bin_snowpack[self.bin_refreeze[:,step] > self.bin_snowpack[:,step], step])
                        # refreeze cannot exceed refreeze potential
                        self.bin_refreeze[self.bin_refreeze[:,step] > self.refreeze_potential, step] = (
                                self.refreeze_potential[self.bin_refreeze[:,step] > self.refreeze_potential])
                        self.bin_refreeze[abs(self.bin_refreeze[:,step]) < pygem_prms.tolerance, step] = 0
                        # update refreeze potential
                        self.refreeze_potential -= self.bin_refreeze[:,step]
                        self.refreeze_potential[abs(self.refreeze_potential) < pygem_prms.tolerance] = 0
                    
                    if step < 12 and self.debug_refreeze:
                        print('refreeze bin ' + str(int(glac_idx_t0[0]*10)) + ':', 
                                np.round(self.bin_refreeze[glac_idx_t0[0],step],3))

                    # SNOWPACK REMAINING [m w.e.]
                    self.snowpack_remaining[:,step] = self.bin_snowpack[:,step] - self.bin_meltsnow[:,step]
                    self.snowpack_remaining[abs(self.snowpack_remaining[:,step]) < pygem_prms.tolerance, step] = 0
                    
                    # Record values                  
                    self.glac_bin_melt[glac_idx_t0,step] = self.bin_melt[glac_idx_t0,step]
                    self.glac_bin_refreeze[glac_idx_t0,step] = self.bin_refreeze[glac_idx_t0,step]
                    self.glac_bin_snowpack[glac_idx_t0,step] = self.bin_snowpack[glac_idx_t0,step]
                    # CLIMATIC MASS BALANCE [m w.e.]
                    self.glac_bin_massbalclim[glac_idx_t0,step] = (
                            self.bin_acc[glac_idx_t0,step] + self.glac_bin_refreeze[glac_idx_t0,step] - 
                            self.glac_bin_melt[glac_idx_t0,step])
                    
                    # OFF-GLACIER ACCUMULATION, MELT, REFREEZE, AND SNOWPAC
                    if option_areaconstant == 0:
                        # precipitation, refreeze, and snowpack are the same both on- and off-glacier
                        self.offglac_bin_prec[offglac_idx,step] = self.bin_prec[offglac_idx,step]
                        self.offglac_bin_refreeze[offglac_idx,step] = self.bin_refreeze[offglac_idx,step]
                        self.offglac_bin_snowpack[offglac_idx,step] = self.bin_snowpack[offglac_idx,step]
                        # Off-glacier melt includes both snow melt and melting of refreezing
                        #  (this is not an issue on-glacier because energy remaining melts underlying snow/ice)
                        # melt of refreezing (assumed to be snow)
                        self.offglac_meltrefreeze = self.surfacetype_ddf_dict[2] * melt_energy_available
                        # melt of refreezing cannot exceed refreezing
                        self.offglac_meltrefreeze[self.offglac_meltrefreeze > self.bin_refreeze[:,step]] = (
                                self.bin_refreeze[:,step][self.offglac_meltrefreeze > self.bin_refreeze[:,step]])
                        # off-glacier melt = snow melt + refreezing melt
                        self.offglac_bin_melt[offglac_idx,step] = (self.bin_meltsnow[offglac_idx,step] + 
                                                                   self.offglac_meltrefreeze[offglac_idx])

                # ===== RETURN TO ANNUAL LOOP =====
                # Mass loss cannot exceed glacier volume
                #  mb [mwea] = -1 * sum{area [km2] * ice thickness [m]} / total area [km2] * density_ice / density_water
                mb_max_loss = (-1 * (glacier_area_t0 * icethickness_t0 * pygem_prms.density_ice / 
                                     pygem_prms.density_water).sum() 
                               / glacier_area_t0.sum())
                # Check annual climatic mass balance
                mb_mwea = ((glacier_area_t0 * self.glac_bin_massbalclim[:,12*year:12*(year+1)].sum(1)).sum() / 
                            glacier_area_t0.sum()) 
                if debug:
                    print('mb_mwea:', np.round(mb_mwea,3))

                # If mass loss exceeds glacier mass, reduce melt to ensure the glacier completely melts without excess
                if mb_mwea < mb_max_loss:  
                    
                    if debug:
                         print('mb_mwea (before):', np.round(mb_mwea,3), 'mb_max_loss:', np.round(mb_max_loss,3))
                         
                    mb_dif = mb_max_loss - mb_mwea
                   
                    glac_wide_melt = ((self.glac_bin_melt[:,12*year:12*(year+1)] * 
                                       glacier_area_t0[:,np.newaxis]).sum() / glacier_area_t0.sum())
                    # adjust using tolerance to avoid any rounding errors that would leave a little glacier volume left
                    self.glac_bin_melt[:,12*year:12*(year+1)] = (self.glac_bin_melt[:,12*year:12*(year+1)] * 
                                                                 (1 + pygem_prms.tolerance - mb_dif / glac_wide_melt))
                    self.glac_bin_massbalclim[:,12*year:12*(year+1)] = (
                            self.bin_acc[:,12*year:12*(year+1)] + self.glac_bin_refreeze[:,12*year:12*(year+1)] - 
                            self.glac_bin_melt[:,12*year:12*(year+1)])
                    # Check annual climatic mass balance
                    mb_mwea = ((glacier_area_t0 * self.glac_bin_massbalclim[:,12*year:12*(year+1)].sum(1)).sum() / 
                                glacier_area_t0.sum()) 
                    
                    
                    if debug:
                        print('mb_check after adjustment (should equal mass loss):', np.round(mb_mwea,3))
                        
#                print(year, 'mb_mwea:', np.round(mb_mwea,3))
#                print(year, 'vol_chg:', (glacier_area_t0 * self.glac_bin_massbalclim[:,12*year:12*(year+1)].sum(1) / 1000 / 0.9).sum())
#                print('area_sum:', glacier_area_t0.sum())
#                print('mbclim_sum:', self.glac_bin_massbalclim[:,12*year:12*(year+1)].sum(1).sum())
                

                # SURFACE TYPE (-)
                # Annual climatic mass balance [m w.e.] used to determine the surface type
                self.glac_bin_massbalclim_annual[:,year] = self.glac_bin_massbalclim[:,12*year:12*(year+1)].sum(1)
                # Update surface type for each bin
                self.surfacetype, firnline_idx = self._surfacetypebinsannual(self.surfacetype,
                                                                        self.glac_bin_massbalclim_annual, year)
                
                if debug:
                    print('glac_bin_massbalclim:', self.glac_bin_massbalclim[12:20,12*year:12*(year+1)].sum(1))
                    print('surface type present:', self.glac_bin_surfacetype_annual[12:20,year])
                    print('surface type updated:', self.surfacetype[12:20])
                    
    
        # Example of modularity
#        if self.use_refreeze:
#            mb += self._refreeze_term(heights, year)
            
#        # Example of how to store variables from within the other functions (ex. mass balance components)
#        self.diag_df.loc[year, 'mb'] = np.average(mb)
        
        # Mass balance for each bin [m ice per second]
        seconds_in_year = self.dates_table.loc[12*year:12*(year+1)-1,'daysinmonth'].values.sum() * 24 * 3600
        mb = (self.glac_bin_massbalclim[:,12*year:12*(year+1)].sum(1) 
              * pygem_prms.density_water / pygem_prms.density_ice / seconds_in_year)
        
        return mb

    
    #%%
    def _convert_glacwide_results(self, year, flowline
#                                 elev_bins, glac_bin_temp, glac_bin_prec, glac_bin_acc, glac_bin_refreeze,
#                             glac_bin_snowpack, glac_bin_melt, glac_bin_frontalablation, glac_bin_massbalclim_annual,
#                             glac_bin_area_annual, glac_bin_icethickness_annual
                             ):
        """
        Convert raw runmassbalance function output to glacier-wide results for output package 2
        
        Parameters
        ----------
        year : int
            the year of the model run starting from zero
        """
        
        # Compute desired output
        print('\n\nNEED GLACIER AREA FOR THE GLACIER-WIDE RESULTS\n\n')
#        glac_bin_area = glac_bin_area_annual[:,0:glac_bin_area_annual.shape[1]-1].repeat(12,axis=1)
#        glac_wide_area = glac_bin_area.sum(axis=0)
        

#        self.glac_wide_prec = np.zeros(bin_temp.shape[1])
#        self.glac_wide_acc = np.zeros(bin_temp.shape[1])
#        self.glac_wide_refreeze = np.zeros(bin_temp.shape[1])
#        self.glac_wide_melt = np.zeros(bin_temp.shape[1])
#        self.glac_wide_frontalablation = np.zeros(bin_temp.shape[1])
        
        # Preset desired output (needed to avoid dividing by zero)
        if 'bin_temp' in self.__dict__.keys():
            print('SHOULD AREA-WEIGHT THIS')
            
            self.glac_wide_temp = np.zeros(bin_temp.shape[1])
            glac_wide_temp_sum = bin_temp[:,year].sum(axis=0)
            print('glac_wide_temp_sum:', glac_wide_temp_sum)
            if glac_wide_temp_sum > 0:
                flowline.area_km2
                
#            glac_bin_temp_nonzero = np.zeros(bin_temp.shape)
#            glac_bin_temp_nonzero[bin_temp != 0] = 1
#            glac_wide_temp_bincount = glac_bin_temp_nonzero.sum(axis=0)
#            self.glac_wide_temp[glac_wide_temp_bincount > 0] = (glac_wide_temp_sum[glac_wide_temp_bincount > 0] /
#                                                                glac_wide_temp_bincount[glac_wide_temp_bincount > 0])
        if 'bin_prec' in self.__dict__.keys():
            print('add prec')
#            glac_wide_prec_mkm2 = (glac_bin_prec * glac_bin_area).sum(axis=0)
#            glac_wide_prec[glac_wide_prec_mkm2 > 0] = (glac_wide_prec_mkm2[glac_wide_prec_mkm2 > 0] /
#                                                       glac_wide_area[glac_wide_prec_mkm2 > 0])
            
        

#        glac_wide_acc_mkm2 = (glac_bin_acc * glac_bin_area).sum(axis=0)
#        glac_wide_acc[glac_wide_acc_mkm2 > 0] = (glac_wide_acc_mkm2[glac_wide_acc_mkm2 > 0] /
#                                                 glac_wide_area[glac_wide_acc_mkm2 > 0])
#        glac_wide_refreeze_mkm2 = (glac_bin_refreeze * glac_bin_area).sum(axis=0)
#        glac_wide_refreeze[glac_wide_refreeze_mkm2 > 0] = (glac_wide_refreeze_mkm2[glac_wide_refreeze_mkm2 > 0] /
#                                                           glac_wide_area[glac_wide_refreeze_mkm2 > 0])
#        glac_wide_melt_mkm2 = (glac_bin_melt * glac_bin_area).sum(axis=0)
#        glac_wide_melt[glac_wide_melt_mkm2 > 0] = (glac_wide_melt_mkm2[glac_wide_melt_mkm2 > 0] /
#                                                   glac_wide_area[glac_wide_melt_mkm2 > 0])
#        glac_wide_frontalablation_mkm2 = (glac_bin_frontalablation * glac_bin_area).sum(axis=0)
#        glac_wide_frontalablation[glac_wide_frontalablation_mkm2 > 0] = (
#                glac_wide_frontalablation_mkm2[glac_wide_frontalablation_mkm2 > 0] /
#                glac_wide_area[glac_wide_frontalablation_mkm2 > 0])
#        glac_wide_massbalclim = glac_wide_acc + glac_wide_refreeze - glac_wide_melt
#        glac_wide_massbaltotal = glac_wide_massbalclim - glac_wide_frontalablation
#        glac_wide_runoff = (glac_wide_prec + glac_wide_melt - glac_wide_refreeze) * glac_wide_area * (1000)**2
#        #  units: (m + m w.e. - m w.e.) * km**2 * (1000 m / 1 km)**2 = m**3
#        glac_wide_snowline = (glac_bin_snowpack > 0).argmax(axis=0)
#        glac_wide_snowline[glac_wide_snowline > 0] = (elev_bins[glac_wide_snowline[glac_wide_snowline > 0]] -
#                                                      pygem_prms.binsize/2)
#        glac_wide_area_annual = glac_bin_area_annual.sum(axis=0)
#        glac_wide_volume_annual = (glac_bin_area_annual * glac_bin_icethickness_annual / 1000).sum(axis=0)
#        glac_wide_ELA_annual = (glac_bin_massbalclim_annual > 0).argmax(axis=0)
#        glac_wide_ELA_annual[glac_wide_ELA_annual > 0] = (elev_bins[glac_wide_ELA_annual[glac_wide_ELA_annual > 0]] -
#                                                          pygem_prms.binsize/2)
#        # ELA and snowline can't be below minimum elevation
#        glac_zmin_annual = elev_bins[(glac_bin_area_annual > 0).argmax(axis=0)][:-1] - pygem_prms.binsize/2
#        glac_wide_ELA_annual[glac_wide_ELA_annual < glac_zmin_annual] = (
#                glac_zmin_annual[glac_wide_ELA_annual < glac_zmin_annual])
#        glac_zmin = elev_bins[(glac_bin_area > 0).argmax(axis=0)] - pygem_prms.binsize/2
#        glac_wide_snowline[glac_wide_snowline < glac_zmin] = glac_zmin[glac_wide_snowline < glac_zmin]
#    
#    #    print('DELETE ME - TESTING')
#    #    # Compute glacier volume change for every time step and use this to compute mass balance
#    #    #  this will work for any indexing
#    #    glac_wide_area = glac_wide_area_annual[:-1].repeat(12)
#    #
#    ##    print('glac_wide_area_annual:', glac_wide_area_annual)
#    #
#    #    # Mass change [km3 mwe]
#    #    #  mb [mwea] * (1 km / 1000 m) * area [km2]
#    #    glac_wide_masschange = glac_wide_massbaltotal / 1000 * glac_wide_area
#    #
#    #    print('glac_wide_melt:', glac_wide_melt)
#    ##    print('glac_wide_massbaltotal:', glac_wide_massbaltotal)
#    ##    print('glac_wide_masschange:', glac_wide_masschange)
#    ##    print('glac_wide_masschange.shape[0] / 12:', glac_wide_masschange.shape[0] / 12)
#    #
#    #    # Mean annual mass balance [mwea]
#    #    mb_mwea = (glac_wide_masschange.sum() / glac_wide_area[0] * 1000 /
#    #               (glac_wide_masschange.shape[0] / 12))
#    #    print('  mb_model [mwea]:', mb_mwea.round(3))
    
#        return (glac_wide_temp, glac_wide_prec, glac_wide_acc, glac_wide_refreeze, glac_wide_melt,
#                glac_wide_frontalablation, glac_wide_massbaltotal, glac_wide_runoff, glac_wide_snowline,
#                glac_wide_area_annual, glac_wide_volume_annual, glac_wide_ELA_annual)
            
                 
    #%%
    def get_annual_frontalablation(self, heights, year=None, flowline=None, fl_id=None, 
                                   sea_level=0, debug=False): 
        """NEED TO DETERMINE HOW TO INTEGRATE FRONTAL ABLATION WITH THE FLOWLINE MODEL
    
        Returns annual climatic mass balance
        
        Parameters
        ----------
        heights : np.array
            elevation bins
        year : int
            year starting with 0 to the number of years in the study
        
        """
        print('hack until Fabien provides data')
        class Dummy():
            pass
        flowline = Dummy()
        flowline.area_km2 = self.glacier_area_t0
        
        # Glacier indices
        glac_idx_t0 = flowline.area_km2.nonzero()[0]
        
        # FRONTAL ABLATION
        # Glacier bed altitude [masl]
        glac_idx_minelev = np.where(self.heights == self.heights[glac_idx_t0].min())[0][0]
        glacier_bedelev = (self.heights[glac_idx_minelev] - self.icethickness_initial[glac_idx_minelev])
        
        print('\n-----')
        print(self.heights[glac_idx_minelev], self.icethickness_initial[glac_idx_t0], self.glacier_area_t0)
        print('-----\n')
        
        print('\nDELETE ME! Switch sea level back to zero\n')
        sea_level = 200
        
        if debug and self.glacier_rgi_table['TermType'] != 0:
            print('\nyear:', year, '\n sea level:', sea_level, 'bed elev:', np.round(glacier_bedelev, 2))
                            
            # If glacier bed below sea level, compute frontal ablation
            if glacier_bedelev < sea_level:
                # Volume [m3] and bed elevation [masl] of each bin
                print('estimate ablation')
#                glac_bin_volume = glacier_area_t0 * 10**6 * icethickness_t0
#                glac_bin_bedelev = np.zeros((glacier_area_t0.shape))
#                glac_bin_bedelev[glac_idx_t0] = heights[glac_idx_t0] - icethickness_initial[glac_idx_t0]
#                
#                # Option 1: Use Huss and Hock (2015) frontal ablation parameterizations
#                #  Frontal ablation using width of lowest bin can severely overestimate the actual width of the
#                #  calving front. Therefore, use estimated calving width from satellite imagery as appropriate.
#                if pygem_prms.option_frontalablation_k == 1 and frontalablation_k == None:
#                    # Calculate frontal ablation parameter based on slope of lowest 100 m of glacier
#                    glac_idx_slope = np.where((heights <= sea_level + 100) & 
#                                              (heights >= heights[glac_idx_t0].min()))[0]
#                    elev_change = np.abs(heights[glac_idx_slope[0]] - heights[glac_idx_slope[-1]])
#                    # length of lowest 100 m of glacier
#                    length_lowest100m = (glacier_area_t0[glac_idx_slope] / 
#                                         width_t0[glac_idx_slope] * 1000).sum()
#                    # slope of lowest 100 m of glacier
#                    slope_lowest100m = np.rad2deg(np.arctan(elev_change/length_lowest100m))
#                    # Frontal ablation parameter
#                    frontalablation_k = frontalablation_k0 * slope_lowest100m
#
#                # Calculate frontal ablation
#                # Bed elevation with respect to sea level
#                #  negative when bed is below sea level (Oerlemans and Nick, 2005)
#                waterdepth = sea_level - glacier_bedelev
#                # Glacier length [m]
#                length = (glacier_area_t0[width_t0 > 0] / width_t0[width_t0 > 0]).sum() * 1000
#                # Height of calving front [m]
#                height_calving = np.max([pygem_prms.af*length**0.5, 
#                                         pygem_prms.density_water / pygem_prms.density_ice * waterdepth])
#                # Width of calving front [m]
#                if pygem_prms.hyps_data in ['oggm']:
#                    width_calving = width_t0[np.where(heights == heights[glac_idx_t0].min())[0][0]] * 1000
#                elif pygem_prms.hyps_data in ['Huss', 'Farinotti']:
#                    if glacier_rgi_table.RGIId in pygem_prms.width_calving_dict:
#                        width_calving = np.float64(pygem_prms.width_calving_dict[glacier_rgi_table.RGIId])
#                    else:
#                        width_calving = width_t0[glac_idx_t0[0]] * 1000                    
#                # Volume loss [m3] due to frontal ablation
#                frontalablation_volumeloss = (
#                        np.max([0, (frontalablation_k * waterdepth * height_calving)]) * width_calving)
#                # Maximum volume loss is volume of bins with their bed elevation below sea level
#                glac_idx_fa = np.where((glac_bin_bedelev < sea_level) & (glacier_area_t0 > 0))[0]
#                frontalablation_volumeloss_max = glac_bin_volume[glac_idx_fa].sum()
#                if frontalablation_volumeloss > frontalablation_volumeloss_max:
#                    frontalablation_volumeloss = frontalablation_volumeloss_max
#                    
#                
#
#                if debug:
#                    print('frontalablation_k:', frontalablation_k)
#                    print('width calving:', width_calving)
#                    print('frontalablation_volumeloss [m3]:', frontalablation_volumeloss)
#                    print('frontalablation_massloss [Gt]:', frontalablation_volumeloss * pygem_prms.density_water / 
#                          pygem_prms.density_ice / 10**9)
#                    print('frontalalabion_volumeloss_max [Gt]:', frontalablation_volumeloss_max * 
#                          pygem_prms.density_water / pygem_prms.density_ice / 10**9)
##                        print('glac_idx_fa:', glac_idx_fa)
##                        print('glac_bin_volume:', glac_bin_volume[0])
##                        print('glac_idx_fa[bin_count]:', glac_idx_fa[0])
##                        print('glac_bin_volume[glac_idx_fa[bin_count]]:', glac_bin_volume[glac_idx_fa[0]])
##                        print('glacier_area_t0[glac_idx_fa[bin_count]]:', glacier_area_t0[glac_idx_fa[0]])
##                        print('glac_bin_frontalablation:', glac_bin_frontalablation[glac_idx_fa[0], step])
#                
#                # Frontal ablation [mwe] in each bin
#                bin_count = 0
#                while (frontalablation_volumeloss > pygem_prms.tolerance) and (bin_count < len(glac_idx_fa)):
#                    # Sort heights to ensure it's universal (works with OGGM and Huss)
#                    heights_calving_sorted = np.argsort(heights[glac_idx_fa])
#                    calving_bin_idx = heights_calving_sorted[bin_count]
#                    # Check if entire bin removed or not
#                    if frontalablation_volumeloss >= glac_bin_volume[glac_idx_fa[calving_bin_idx]]:
#                        glac_bin_frontalablation[glac_idx_fa[calving_bin_idx], step] = (
#                                glac_bin_volume[glac_idx_fa[calving_bin_idx]] / 
#                                (glacier_area_t0[glac_idx_fa[calving_bin_idx]] * 10**6) 
#                                * pygem_prms.density_ice / pygem_prms.density_water)   
#                    else:
#                        glac_bin_frontalablation[glac_idx_fa[calving_bin_idx], step] = (
#                                frontalablation_volumeloss / (glacier_area_t0[glac_idx_fa[calving_bin_idx]] * 10**6)
#                                * pygem_prms.density_ice / pygem_prms.density_water)
#                    frontalablation_volumeloss += (
#                            -1 * glac_bin_frontalablation[glac_idx_fa[calving_bin_idx],step] * pygem_prms.density_water 
#                            / pygem_prms.density_ice * glacier_area_t0[glac_idx_fa[calving_bin_idx]] * 10**6)                        
#                                            
#                    if debug:
#                        print('glacier idx:', glac_idx_fa[calving_bin_idx], 
#                              'volume loss:', (glac_bin_frontalablation[glac_idx_fa[calving_bin_idx], step] * 
#                              glacier_area_t0[glac_idx_fa[calving_bin_idx]] * pygem_prms.density_water / 
#                              pygem_prms.density_ice * 10**6).round(0))
#                        print('remaining volume loss:', frontalablation_volumeloss, 'tolerance:', pygem_prms.tolerance)
#                    
#                    bin_count += 1         
#                        
#                if debug:
#                    print('frontalablation_volumeloss remaining [m3]:', frontalablation_volumeloss)
#                    print('ice thickness:', icethickness_t0[glac_idx_fa[0]].round(0), 
#                          'waterdepth:', waterdepth.round(0), 
#                          'height calving front:', height_calving.round(0), 
#                          'width [m]:', (width_calving).round(0))  
            
        return 0
            
            #%%
                

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
#                        fa_change = abs(glac_bin_frontalablation[:, step] * pygem_prms.density_water / 
#                                        pygem_prms.density_ice - icethickness_t0)
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
                
        
        
        #%%
        
#        # Example of how to store variables from within the other functions (ex. mass balance components)
#        self.diag_df = pd.DataFrame()
        
#    # Example of what could be done!
#    def _refreeze_term(self, heights, year):
#        
#        return 0
    
    # ===== SURFACE TYPE FUNCTIONS =====
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
        surfacetype = np.zeros(self.icethickness_initial.shape)
        # Option 1 - initial surface type based on the median elevation
        if pygem_prms.option_surfacetype_initial == 1:
            surfacetype[(elev_bins < self.glacier_rgi_table.loc['Zmed']) & (self.icethickness_initial > 0)] = 1
            surfacetype[(elev_bins >= self.glacier_rgi_table.loc['Zmed']) & (self.icethickness_initial > 0)] = 2
        # Option 2 - initial surface type based on the mean elevation
        elif pygem_prms.option_surfacetype_initial ==2:
            surfacetype[(elev_bins < self.glacier_rgi_table['Zmean']) & (self.icethickness_initial > 0)] = 1
            surfacetype[(elev_bins >= self.glacier_rgi_table['Zmean']) & (self.icethickness_initial > 0)] = 2
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
            #  everything initially considered snow is considered firn, i.e., the model initially assumes there is no 
            #  snow on the surface anywhere.
        if pygem_prms.option_surfacetype_debris == 1:
            print("Need to code the model to include debris. This option does not currently exist. Please choose option"
                  + " that exists.\nExiting the model run.")
            exit()
            # One way to include debris would be to have debris cover maps and state that the debris retards melting 
            # as a fraction of melt.
        return surfacetype, firnline_idx
    
    
    def _surfacetypebinsannual(self, surfacetype, glac_bin_massbalclim_annual, year_index):
        """
        Update surface type according to climatic mass balance over the last five years.  
        
        If 5-year climatic balance is positive, then snow/firn.  If negative, then ice/debris.
        Convention: 0 = off-glacier, 1 = ice, 2 = snow, 3 = firn, 4 = debris
        
        Function Options:
          > 1 (default) - update surface type according to Huss and Hock (2015)
          > 2 - Radic and Hock (2011)
        Huss and Hock (2015): Initially, above median glacier elevation is firn, below is ice. Surface type updated for
          each elevation band and month depending on specific mass balance.  If the cumulative balance since the start 
          of mass balance year is positive, then snow is assigned. If the cumulative mass balance is negative (i.e., 
          all snow of current mass balance year has melted), then bare ice or firn exposed. Surface type is assumed to 
          be firn if the elevation band's average annual balance over the preceding 5 years (B_t-5_avg) is positive. If
          B_t-5_avg is negative, surface type is ice.
              > climatic mass balance calculated at each bin and used with the mass balance over the last 5 years to 
                determine whether the surface is firn or ice.  Snow is separate based on each month.
        Radic and Hock (2011): "DDF_snow is used above the ELA regardless of snow cover.  Below the ELA, use DDF_ice is 
          used only when snow cover is 0.  ELA is calculated from observed annual mass balance profiles averaged over 
          the observational period and is kept constant in time for the calibration period.  For the future projections, 
          ELA is set to the mean glacier height and is time dependent since glacier volume, area, and length are time 
          dependent (volume-area-length scaling).
          Bliss et al. (2014) uses the same as Valentina's model
        
        Parameters
        ----------
        surfacetype : np.ndarray
            Surface type for each elevation bin
        glac_bin_massbalclim_annual : np.ndarray
            Annual climatic mass balance for each year and each elevation bin
        year_index : int
            Count of the year of model run (first year is 0)
        Returns
        -------
        surfacetype : np.ndarray
            Updated surface type for each elevation bin
        firnline_idx : int
            Firn line index
        """        
        # Next year's surface type is based on the bin's average annual climatic mass balance over the last 5 years.  If 
        #  less than 5 years, then use the average of the existing years.
        if year_index < 5:
            # Calculate average annual climatic mass balance since run began
            massbal_clim_mwe_runningavg = glac_bin_massbalclim_annual[:,0:year_index+1].mean(1)
        else:
            massbal_clim_mwe_runningavg = glac_bin_massbalclim_annual[:,year_index-4:year_index+1].mean(1)
        # If the average annual specific climatic mass balance is negative, then the surface type is ice (or debris)
        surfacetype[(surfacetype !=0 ) & (massbal_clim_mwe_runningavg <= 0)] = 1
        # If the average annual specific climatic mass balance is positive, then the surface type is snow (or firn)
        surfacetype[(surfacetype != 0) & (massbal_clim_mwe_runningavg > 0)] = 2
        # Compute the firnline index
        try:
            # firn in bins >= firnline_idx
            firnline_idx = np.where(surfacetype==2)[0][0]
        except:
            # avoid errors if there is no firn, i.e., the entire glacier is melting
            firnline_idx = np.where(surfacetype!=0)[0][-1]
        # Apply surface type model options
        # If firn surface type option is included, then snow is changed to firn
        if pygem_prms.option_surfacetype_firn == 1:
            surfacetype[surfacetype == 2] = 3
        if pygem_prms.option_surfacetype_debris == 1:
            print('Need to code the model to include debris.  Please choose an option that currently exists.\n'
                  'Exiting the model run.')
            exit()
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