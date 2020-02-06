#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 14:00:14 2020

@author: davidrounce
"""
from oggm import cfg, utils
from oggm.core.flowline import FlowlineModel
from oggm.exceptions import InvalidParamsError
import pygem.pygem_input as pygem_prms
import numpy as np
import pandas as pd
#import netCDF4

cfg.initialize()

#%%
class MassRedistributionCurveModel(FlowlineModel):
    """Glacier geometry updated using mass redistribution curves; also known as the "delta-h method"

    This uses mass redistribution curves from Huss et al. (2010) to update the glacier geometry
    """

    def __init__(self, flowlines, mb_model=None, y0=0., 
                 inplace=False,
                 debug=True,
                 option_areaconstant=False, spinupyears=pygem_prms.spinupyears, 
                 constantarea_years=pygem_prms.constantarea_years,
                 **kwargs):
        """ Instanciate the model.
        
        Parameters
        ----------
        flowlines : list
            the glacier flowlines
        mb_model : MassBalanceModel
            the mass-balance model
        y0 : int
            initial year of the simulation
        inplace : bool
            whether or not to make a copy of the flowline objects for the run
            setting to True implies that your objects will be modified at run
            time by the model (can help to spare memory)
        is_tidewater: bool, default: False
            use the very basic parameterization for tidewater glaciers
        mb_elev_feedback : str, default: 'annual'
            'never', 'always', 'annual', or 'monthly': how often the
            mass-balance should be recomputed from the mass balance model.
            'Never' is equivalent to 'annual' but without elevation feedback
            at all (the heights are taken from the first call).
        check_for_boundaries: bool, default: True
            raise an error when the glacier grows bigger than the domain
            boundaries
        """
        super(MassRedistributionCurveModel, self).__init__(flowlines, mb_model=mb_model, y0=y0, inplace=inplace,
                                                           mb_elev_feedback='annual', **kwargs)
        self.option_areaconstant = option_areaconstant
        self.constantarea_years = constantarea_years
        self.spinupyears = spinupyears
        self.glac_idx_initial = [fl.thick.nonzero()[0] for fl in flowlines]
        
        # HERE IS THE STUFF TO RECORD FOR EACH FLOWLINE!
        self.calving_m3_since_y0 = 0.  # total calving since time y0
        
        assert len(flowlines) == 1, 'MassRedistributionCurveModel is not set up for multiple flowlines'


    def run_until(self, y1):
        """Runs the model from the current year up to a given year date y1.
        This function runs the model for the time difference y1-self.y0
        If self.y0 has not been specified at some point, it is 0 and y1 will
        be the time span in years to run the model for.
        Parameters
        ----------
        y1 : float
            Upper time span for how long the model should run
        """
        
        # Consider running backwards within here
        
        # We force timesteps to yearly timesteps
        years = np.arange(0, y1 + 1)
        for year in years:
            self.updategeometry(year)

        # Check for domain bounds
        if self.check_for_boundaries:
            if self.fls[-1].thick[-1] > 10:
                raise RuntimeError('Glacier exceeds domain boundaries, '
                                   'at year: {}'.format(self.yr))
        # Check for NaNs
        for fl in self.fls:
            if np.any(~np.isfinite(fl.thick)):
                raise FloatingPointError('NaN in numerical solution.')
    
    
    def updategeometry(self, year):
        """Update geometry for a given year"""
        
        if year == 0:
            print('\nrecord annual data to the "diagnostic_dataset" - diag_ds - like OGGM\n')
            
        
        
        # Loop over flowlines
        for fl_id, fl in enumerate(self.fls):

            # Flowline state
            # !! Change "heights" to "surface_h" for consistency with OGGM
            heights = fl.surface_h
            section = fl.section
            dx = fl.dx_meter
            icethickness_t0 = fl.thick
            width_t0 = fl.widths_m / 1000
            glacier_area_t0 = width_t0 * dx / 1000
            
            # Glacier indices
            glac_idx_t0 = icethickness_t0.nonzero()[0]
            
            # MASS REDISTRIBUTION
            # Mass redistribution ignored for calibration and spinup years (glacier properties constant)
            if (self.option_areaconstant == 1) or (year < self.spinupyears) or (year < self.constantarea_years):
                # run mass balance
                glac_bin_massbalclim_annual = self.mb_model.get_annual_mb(heights, fls=self.fls, fl_id=fl_id, 
                                                                              year=year, debug=False
                                                                              )
                # glacier stays the same
                icethickness_t1 = icethickness_t0.copy()
                width_t1 = width_t0.copy()
#                glacier_area_t1 = glacier_area_t0.copy()
                
                
            # Redistribute mass according to curves
            else:
#                if year == 0:
#                    print('\nHERE WE NEED THE GET FRONTAL ABLATION!\n')
#                # First, remove volume lost to frontal ablation
#                #  changes to _t0 not _t1, since t1 will be done in the mass redistribution
#                if glac_bin_frontalablation[:,step].max() > 0:
#                    # Frontal ablation loss [mwe]
#                    #  fa_change tracks whether entire bin is lost or not
#                    fa_change = abs(glac_bin_frontalablation[:, step] * pygem_prms.density_water / pygem_prms.density_ice
#                                    - icethickness_t0)
#                    fa_change[fa_change <= pygem_prms.tolerance] = 0
#                    
#                    if debug:
#                        bins_wfa = np.where(glac_bin_frontalablation[:,step] > 0)[0]
#                        print('glacier area t0:', glacier_area_t0[bins_wfa].round(3))
#                        print('ice thickness t0:', icethickness_t0[bins_wfa].round(1))
#                        print('frontalablation [m ice]:', (glac_bin_frontalablation[bins_wfa, step] * 
#                              pygem_prms.density_water / pygem_prms.density_ice).round(1))
#                        print('frontal ablation [mice] vs icethickness:', fa_change[bins_wfa].round(1))
#                    
#                    # Check if entire bin is removed
#                    glacier_area_t0[np.where(fa_change == 0)[0]] = 0
#                    icethickness_t0[np.where(fa_change == 0)[0]] = 0
#                    width_t0[np.where(fa_change == 0)[0]] = 0
#                    # Otherwise, reduce glacier area such that glacier retreats and ice thickness remains the same
#                    #  A_1 = (V_0 - V_loss) / h_1,  units: A_1 = (m ice * km2) / (m ice) = km2
#                    glacier_area_t0[np.where(fa_change != 0)[0]] = (
#                            (glacier_area_t0[np.where(fa_change != 0)[0]] * 
#                             icethickness_t0[np.where(fa_change != 0)[0]] - 
#                             glacier_area_t0[np.where(fa_change != 0)[0]] * 
#                             glac_bin_frontalablation[np.where(fa_change != 0)[0], step] * pygem_prms.density_water 
#                             / pygem_prms.density_ice) / icethickness_t0[np.where(fa_change != 0)[0]])
#                    
#                    if debug:
#                        print('glacier area t1:', glacier_area_t0[bins_wfa].round(3))
#                        print('ice thickness t1:', icethickness_t0[bins_wfa].round(1))
                
                # Redistribute mass if glacier was not fully removed by frontal ablation
                if len(glac_idx_t0) > 0:
                    # Mass redistribution according to Huss empirical curves
                    glac_bin_massbalclim_annual = self.mb_model.get_annual_mb(heights, fls=self.fls, fl_id=fl_id, 
                                                                              year=year, debug=False
                                                                              )                    
                    sec_in_year = (self.mb_model.dates_table.loc[12*year:12*(year+1)-1,'daysinmonth'].values.sum() 
                                   * 24 * 3600)
                    glacier_area_t1, icethickness_t1, width_t1 = (
                            self._massredistributionHuss(glacier_area_t0, icethickness_t0, width_t0, 
                                                         glac_bin_massbalclim_annual, self.glac_idx_initial[fl_id], 
                                                         heights, sec_in_year=sec_in_year))
                    
                else:
#                    glacier_area_t1 = np.zeros(heights.shape)
                    icethickness_t1 = np.zeros(heights.shape)
#                    width_t1 = np.zeros(heights.shape)
                    
                    
#            # Record glacier properties (area [km**2], thickness [m], width [km])
#            # if first year, record initial glacier properties (area [km**2], ice thickness [m ice], width [km])
#            if year == 0:
#                glac_bin_area_annual[:,year] = glacier_area_initial
#                glac_bin_icethickness_annual[:,year] = icethickness_initial
#                glac_bin_width_annual[:,year] = width_initial
#            # record the next year's properties as well
#            # 'year + 1' used so the glacier properties are consistent with mass balance computations
#            glac_bin_icethickness_annual[:,year + 1] = icethickness_t1
#            glac_bin_area_annual[:,year + 1] = glacier_area_t1
#            glac_bin_width_annual[:,year + 1] = width_t1
            
            
            # UPDATE THICKNESS, WIDTH, AND SURFACE ELEVATION
            fl.thick = icethickness_t1.copy()
#            fl._widths_m = width_t1.copy() / 1000
#            fl.surface_h = heights + (icethickness_t1 - icethickness_t0)
#            fl.section = 
            
#            glac_bin_surfacetype_annual[:,year] = surfacetype

        # Next step
#        self.yr += 1
    
    
    def update_surfacetype():
        """NEED TO CODE SOMETHING TO UPDATE THE SURFACE TYPE!"""
        print('update surface type')
        
        print('How am I handling updating the surface type??')
        # update surface type for bins that have retreated
        surfacetype[glacier_area_t1 == 0] = 0
        # update surface type for bins that have advanced 
        print('PROBLEM BELOW: can no longer use first non-zero because this would put snow at terminus')
        surfacetype[(surfacetype == 0) & (glacier_area_t1 != 0)] = (
                surfacetype[glacier_area_t0.nonzero()[0][0]])
                

            
            #%%%% ====== START OF MASS REDISTRIBUTION CURVE  
    def _massredistributionHuss(self, glacier_area_t0, icethickness_t0, width_t0, glac_bin_massbalclim_annual, 
                                glac_idx_initial, heights, debug=False, hindcast=0, sec_in_year=365*24*3600):
        """
        Mass redistribution according to empirical equations from Huss and Hock (2015) accounting for retreat/advance.
        glac_idx_initial is required to ensure that the glacier does not advance to area where glacier did not exist before
        (e.g., retreat and advance over a vertical cliff)
        
        Parameters
        ----------
        glacier_area_t0 : np.ndarray
            Glacier area [km2] from previous year for each elevation bin
        icethickness_t0 : np.ndarray
            Ice thickness [m] from previous year for each elevation bin
        width_t0 : np.ndarray
            Glacier width [km] from previous year for each elevation bin
        glac_bin_massbalclim_annual : np.ndarray
            Climatic mass balance [m w.e.] for each elevation bin and year
        glac_idx_initial : np.ndarray
            Initial glacier indices
        debug : Boolean
            option to turn on print statements for development or debugging of code (default False)
        Returns
        -------
        glacier_area_t1 : np.ndarray
            Updated glacier area [km2] for each elevation bin
        icethickness_t1 : np.ndarray
            Updated ice thickness [m] for each elevation bin
        width_t1 : np.ndarray
            Updated glacier width [km] for each elevation bin
        """        
        # Reset the annual glacier area and ice thickness
        glacier_area_t1 = np.zeros(glacier_area_t0.shape)
        icethickness_t1 = np.zeros(glacier_area_t0.shape)
        width_t1 = np.zeros(glacier_area_t0.shape)
        # Annual glacier-wide volume change [km**3]
        glacier_volumechange = (glac_bin_massbalclim_annual / 1000 * sec_in_year * glacier_area_t0).sum()
        
        # For hindcast simulations, volume change is the opposite
        if hindcast == 1:
            glacier_volumechange = -1 * glacier_volumechange
            
        if debug:
            print('\nDebugging Mass Redistribution Huss function\n')
            print('glacier volume change:', glacier_volumechange)
        #  units: [m w.e.] * (1 km / 1000 m) * (1000 kg / (1 m water * m**2) * (1 m ice * m**2 / 900 kg) * [km**2] 
        #         = km**3 ice          
        # If volume loss is less than the glacier volume, then redistribute mass loss/gains across the glacier;
        #  otherwise, the glacier disappears (area and thickness were already set to zero above)
        if -1 * glacier_volumechange < (icethickness_t0 / 1000 * glacier_area_t0).sum():
            # Determine where glacier exists
            
            # Check for negative glacier areas
            #  shouldn't need these three lines, but Anna mentioned she was finding negative areas somehow? 2019/01/30
            glacier_area_t0[glacier_area_t0 < 0] = 0
            icethickness_t0[glacier_area_t0 < 0] = 0
            width_t0[glacier_area_t0 < 0] = 0
            
            glac_idx_t0 = glacier_area_t0.nonzero()[0]
            # Compute ice thickness [m ice], glacier area [km**2] and ice thickness change [m ice] after 
            #  redistribution of gains/losses
            if pygem_prms.option_massredistribution == 1:
                # Option 1: apply mass redistribution using Huss' empirical geometry change equations
                icethickness_t1, glacier_area_t1, width_t1, icethickness_change, glacier_volumechange_remaining = (
                        self._massredistributioncurveHuss(icethickness_t0, glacier_area_t0, width_t0, glac_idx_t0,
                                                          glacier_volumechange, glac_bin_massbalclim_annual,
                                                          heights, debug=False))
                
                if debug:
                    print(icethickness_t0.max(), icethickness_t1.max(), glacier_area_t0.max(), glacier_area_t1.max())
    
            # Glacier retreat
            #  if glacier retreats (ice thickness < 0), then ice thickness is set to zero, and some volume change will need 
            #   to be redistributed across the rest of the glacier
            while glacier_volumechange_remaining < 0:
                glacier_area_t0_retreated = glacier_area_t1.copy()
                icethickness_t0_retreated = icethickness_t1.copy()
                width_t0_retreated = width_t1.copy()
                glacier_volumechange_remaining_retreated = glacier_volumechange_remaining.copy()
                glac_idx_t0_retreated = glacier_area_t0_retreated.nonzero()[0]            
                # Set climatic mass balance for the case when there are less than 3 bins  
                #  distribute the remaining glacier volume change over the entire glacier (remaining bins)
                massbal_clim_retreat = np.zeros(glacier_area_t0_retreated.shape)
                massbal_clim_retreat[glac_idx_t0_retreated] = (glacier_volumechange_remaining / 
                                                               glacier_area_t0_retreated.sum() * 1000)
                # Mass redistribution 
                if pygem_prms.option_massredistribution == 1:
                    # Option 1: apply mass redistribution using Huss' empirical geometry change equations
                    icethickness_t1, glacier_area_t1, width_t1, icethickness_change, glacier_volumechange_remaining = (
                            self._massredistributioncurveHuss(icethickness_t0_retreated, glacier_area_t0_retreated, 
                                                              width_t0_retreated, glac_idx_t0_retreated, 
                                                              glacier_volumechange_remaining_retreated, 
                                                              massbal_clim_retreat, heights)) 
    
            # Glacier advances 
            #  based on ice thickness change exceeding threshold
            #  Overview:
            #    1. If last bin is not full, i.e., area >= area of average terminus bin, then fill it up
            #    2. Add new bin and fill it up
            #    3. If additional volume after adding new bin, then redistribute mass gain across all bins again,
            #       i.e., increase the ice thickness and width
            #    4. Repeat adding a new bin and redistributing the mass until no addiitonal volume is left
            while (icethickness_change > pygem_prms.icethickness_advancethreshold).any() == True: 
                if debug:
                    print('advancing glacier')
                # Record glacier area and ice thickness before advance corrections applied
                glacier_area_t1_raw = glacier_area_t1.copy()
                icethickness_t1_raw = icethickness_t1.copy()
                width_t1_raw = width_t1.copy()
                # Index bins that are advancing
                icethickness_change[icethickness_change <= pygem_prms.icethickness_advancethreshold] = 0
                glac_idx_advance = icethickness_change.nonzero()[0]
                # Update ice thickness based on maximum advance threshold [m ice]
                icethickness_t1[glac_idx_advance] = (icethickness_t1[glac_idx_advance] - 
                               (icethickness_change[glac_idx_advance] - pygem_prms.icethickness_advancethreshold))
                # Update glacier area based on reduced ice thicknesses [km**2]
                if pygem_prms.option_glaciershape == 1:
                    # Glacier area for parabola [km**2] (A_1 = A_0 * (H_1 / H_0)**0.5)
                    glacier_area_t1[glac_idx_advance] = (glacier_area_t1_raw[glac_idx_advance] * 
                                   (icethickness_t1[glac_idx_advance] / icethickness_t1_raw[glac_idx_advance])**0.5)
                    # Glacier width for parabola [km] (w_1 = w_0 * A_1 / A_0)
                    width_t1[glac_idx_advance] = (width_t1_raw[glac_idx_advance] * glacier_area_t1[glac_idx_advance] 
                                                  / glacier_area_t1_raw[glac_idx_advance])
                elif pygem_prms.option_glaciershape == 2:
                    # Glacier area constant for rectangle [km**2] (A_1 = A_0)
                    glacier_area_t1[glac_idx_advance] = glacier_area_t1_raw[glac_idx_advance]
                    # Glacier with constant for rectangle [km] (w_1 = w_0)
                    width_t1[glac_idx_advance] = width_t1_raw[glac_idx_advance]
                elif pygem_prms.option_glaciershape == 3:
                    # Glacier area for triangle [km**2] (A_1 = A_0 * H_1 / H_0)
                    glacier_area_t1[glac_idx_t0] = (glacier_area_t1_raw[glac_idx_t0] * 
                                   icethickness_t1[glac_idx_t0] / icethickness_t1_raw[glac_idx_t0])
                    # Glacier width for triangle [km] (w_1 = w_0 * A_1 / A_0)
                    width_t1[glac_idx_advance] = (width_t1_raw[glac_idx_advance] * glacier_area_t1[glac_idx_advance] 
                                                  / glacier_area_t1_raw[glac_idx_advance])
                # Advance volume [km**3]
                advance_volume = ((glacier_area_t1_raw[glac_idx_advance] * 
                                  icethickness_t1_raw[glac_idx_advance] / 1000).sum() - 
                                  (glacier_area_t1[glac_idx_advance] * icethickness_t1[glac_idx_advance] / 
                                   1000).sum())
    
                # Advance characteristics
                # Indices that define the glacier terminus
                glac_idx_terminus = (
                        glac_idx_t0[(heights[glac_idx_t0] - heights[glac_idx_t0].min()) / 
                                    (heights[glac_idx_t0].max() - heights[glac_idx_t0].min()) * 100 
                                    < pygem_prms.terminus_percentage])
                # For glaciers with so few bands that the terminus is not identified (ex. <= 4 bands for 20% threshold),
                #  then use the information from all the bands
                if glac_idx_terminus.shape[0] <= 1:
                    glac_idx_terminus = glac_idx_t0.copy()
                
                if debug:
                    print('glacier index terminus:',glac_idx_terminus)
                    
                # Average area of glacier terminus [km**2]
                #  exclude the bin at the terminus, since this bin may need to be filled first
                try:
                    minelev_idx = np.where(heights == heights[glac_idx_terminus].min())[0][0]
                    glac_idx_terminus_removemin = list(glac_idx_terminus)
                    glac_idx_terminus_removemin.remove(minelev_idx)
                    terminus_area_avg = np.mean(glacier_area_t0[glac_idx_terminus_removemin])
                except:  
                    glac_idx_terminus_initial = (
                        glac_idx_initial[(heights[glac_idx_initial] - heights[glac_idx_initial].min()) / 
                                    (heights[glac_idx_initial].max() - heights[glac_idx_initial].min()) * 100 
                                    < pygem_prms.terminus_percentage])
                    if glac_idx_terminus_initial.shape[0] <= 1:
                        glac_idx_terminus_initial = glac_idx_initial.copy()
                        
                    minelev_idx = np.where(heights == heights[glac_idx_terminus_initial].min())[0][0]
                    glac_idx_terminus_removemin = list(glac_idx_terminus_initial)
                    glac_idx_terminus_removemin.remove(minelev_idx)
                    terminus_area_avg = np.mean(glacier_area_t0[glac_idx_terminus_removemin])
    
                # Check if the last bin's area is below the terminus' average and fill it up if it is
                if (glacier_area_t1[minelev_idx] < terminus_area_avg) and (icethickness_t0[minelev_idx] <
                   icethickness_t0[glac_idx_terminus].mean()):
                    # Volume required to fill the bin at the terminus
                    advance_volume_fillbin = (icethickness_t1[minelev_idx] / 1000 * 
                                              (terminus_area_avg - glacier_area_t1[minelev_idx]))
                    # If the advance volume is less than that required to fill the bin, then fill the bin as much as
                    #  possible by adding area (thickness remains the same - glacier front is only thing advancing)
                    if advance_volume < advance_volume_fillbin:
                        # add advance volume to the bin (area increases, thickness and width constant)
                        glacier_area_t1[minelev_idx] = (glacier_area_t1[minelev_idx] + 
                                                        advance_volume / (icethickness_t1[minelev_idx] / 1000))
                        # set advance volume equal to zero
                        advance_volume = 0
                    else:
                        # fill the bin (area increases, thickness and width constant)
                        glacier_area_t1[minelev_idx] = (glacier_area_t1[minelev_idx] + 
                                                        advance_volume_fillbin / (icethickness_t1[minelev_idx] / 1000))
                        advance_volume = advance_volume - advance_volume_fillbin                    
    
                # With remaining advance volume, add a bin or redistribute over existing bins if no bins left
                if advance_volume > 0:
                    # Indices for additional bins below the terminus
                    below_glac_idx = np.where(heights < heights[glacier_area_t1 > 0].min())[0]
                    # if no more bins below, then distribute volume over the glacier without further adjustments
                    if len(below_glac_idx) == 0:
                        glacier_area_t1 = glacier_area_t1_raw
                        icethickness_t1 = icethickness_t1_raw
                        width_t1 = width_t1_raw
                        advance_volume = 0
                        
                    # otherwise, add a bin with thickness and width equal to the previous bin and fill it up
                    else:
                        # Sort heights below terminus to ensure it's universal (works with OGGM and Huss)
                        elev_bin2add_sorted = heights[below_glac_idx][np.argsort(heights[below_glac_idx])[::-1]]
                        bin2add_count = 0
                        glac_idx_bin2add = np.where(heights == elev_bin2add_sorted[bin2add_count])[0][0] 
                        # Check if bin2add is in a discontinuous section of the initial glacier
                        while (heights[glac_idx_bin2add] > heights[glac_idx_initial].min() and 
                               glac_idx_bin2add in list(glac_idx_initial)):
                            # Advance should not occur in a discontinuous section of the glacier (e.g., vertical drop),
                            #  so change the bin2add to the next bin down valley
                            bin2add_count += 1
                            glac_idx_bin2add = np.where(heights == elev_bin2add_sorted[bin2add_count])[0][0]
                            
                        # ice thickness of new bin equals ice thickness of bin at the terminus
                        icethickness_t1[glac_idx_bin2add] = icethickness_t1[minelev_idx]
                        width_t1[glac_idx_bin2add] = width_t1[minelev_idx]
                        # volume required to fill the bin at the terminus
                        advance_volume_fillbin = icethickness_t1[glac_idx_bin2add] / 1000 * terminus_area_avg 
                        # If the advance volume is unable to fill entire bin, then fill it as much as possible
                        if advance_volume < advance_volume_fillbin:
                            # add advance volume to the bin (area increases, thickness and width constant)
                            glacier_area_t1[glac_idx_bin2add] = (advance_volume / (icethickness_t1[glac_idx_bin2add]
                                                                 / 1000))
                            advance_volume = 0
                        else:
                            # fill the bin (area increases, thickness and width constant)
                            glacier_area_t1[glac_idx_bin2add] = terminus_area_avg
                            advance_volume = advance_volume - advance_volume_fillbin
                            
                # update the glacier indices
                glac_idx_t0 = glacier_area_t1.nonzero()[0]
                massbal_clim_advance = np.zeros(glacier_area_t1.shape)
                # Record glacier area and ice thickness before advance corrections applied
                glacier_area_t1_raw = glacier_area_t1.copy()
                icethickness_t1_raw = icethickness_t1.copy()
                width_t1_raw = width_t1.copy()
                # If a full bin has been added and volume still remains, then redistribute mass across the
                #  glacier, thereby enabling the bins to get thicker once again prior to adding a new bin.
                #  This is important for glaciers that have very thin ice at the terminus as this prevents the 
                #  glacier from having a thin layer of ice advance tremendously far down valley without thickening.
                if advance_volume > 0:
                    if pygem_prms.option_massredistribution == 1:
                        # Option 1: apply mass redistribution using Huss' empirical geometry change equations
                        (icethickness_t1, glacier_area_t1, width_t1, icethickness_change, 
                         glacier_volumechange_remaining) = (
                                self._massredistributioncurveHuss(
                                        icethickness_t1, glacier_area_t1, width_t1, glac_idx_t0, 
                                        advance_volume, massbal_clim_advance, heights))
                # update ice thickness change
                icethickness_change = icethickness_t1 - icethickness_t1_raw
    
#        np.set_printoptions(suppress=True)
#        print('ice thickness change:', np.round(icethickness_change[glac_idx_t0],3))
    
        return glacier_area_t1, icethickness_t1, width_t1
    
    
    def _massredistributioncurveHuss(self, icethickness_t0, glacier_area_t0, width_t0, glac_idx_t0, 
                                     glacier_volumechange, massbalclim_annual, heights, debug=False):
        """
        Apply the mass redistribution curves from Huss and Hock (2015).
        This is paired with massredistributionHuss, which takes into consideration retreat and advance.
        
        To-do list
        ----------
        - volume-length scaling
        - volume-area scaling
        - pair with OGGM flow model
        
        Parameters
        ----------
        icethickness_t0 : np.ndarray
            Ice thickness [m] from previous year for each elevation bin
        glacier_area_t0 : np.ndarray
            Glacier area [km2] from previous year for each elevation bin
        width_t0 : np.ndarray
            Glacier width [km] from previous year for each elevation bin
        massbalclim_annual : np.ndarray
            Annual climatic mass balance [m w.e.] for each elevation bin for a single year
        glac_idx_t0 : np.ndarray
            glacier indices for present timestep
        glacier_volumechange : float
            glacier-wide volume change [km3] based on the annual climatic mass balance
        Returns
        -------
        glacier_area_t1 : np.ndarray
            Updated glacier area [km2] for each elevation bin
        icethickness_t1 : np.ndarray
            Updated ice thickness [m] for each elevation bin
        width_t1 : np.ndarray
            Updated glacier width [km] for each elevation bin
        icethickness_change : np.ndarray
            Ice thickness change [m] for each elevation bin
        glacier_volumechange_remaining : float
            Glacier volume change remaining, which could occur if there is less ice in a bin than melt, i.e., retreat
        """           
        if debug:
            print('\nDebugging mass redistribution curve Huss\n')
        # Apply Huss redistribution if there are at least 3 elevation bands; otherwise, use the mass balance
        # reset variables
        icethickness_t1 = np.zeros(glacier_area_t0.shape)
        glacier_area_t1 = np.zeros(glacier_area_t0.shape)
        width_t1 = np.zeros(glacier_area_t0.shape) 
        if glac_idx_t0.shape[0] > 3:
            # Select the factors for the normalized ice thickness change curve based on glacier area
            if glacier_area_t0.sum() > 20:
                [gamma, a, b, c] = [6, -0.02, 0.12, 0]
            elif glacier_area_t0.sum() > 5:
                [gamma, a, b, c] = [4, -0.05, 0.19, 0.01]
            else:
                [gamma, a, b, c] = [2, -0.30, 0.60, 0.09]
            # reset variables
            elevrange_norm = np.zeros(glacier_area_t0.shape)
            icethicknesschange_norm = np.zeros(glacier_area_t0.shape)
            # Normalized elevation range [-]
            #  (max elevation - bin elevation) / (max_elevation - min_elevation)
            elevrange_norm[glacier_area_t0 > 0] = ((heights[glac_idx_t0].max() - heights[glac_idx_t0]) / 
                                                   (heights[glac_idx_t0].max() - heights[glac_idx_t0].min()))
            
            #  using indices as opposed to elevations automatically skips bins on the glacier that have no area
            #  such that the normalization is done only on bins where the glacier lies
            # Normalized ice thickness change [-]
            icethicknesschange_norm[glacier_area_t0 > 0] = ((elevrange_norm[glacier_area_t0 > 0] + a)**gamma + 
                                                            b*(elevrange_norm[glacier_area_t0 > 0] + a) + c)
            #  delta_h = (h_n + a)**gamma + b*(h_n + a) + c
            #  indexing is faster here
            # limit the icethicknesschange_norm to between 0 - 1 (ends of fxns not exactly 0 and 1)
            icethicknesschange_norm[icethicknesschange_norm > 1] = 1
            icethicknesschange_norm[icethicknesschange_norm < 0] = 0
            # Huss' ice thickness scaling factor, fs_huss [m ice]         
            #  units: km**3 / (km**2 * [-]) * (1000 m / 1 km) = m ice
            fs_huss = glacier_volumechange / (glacier_area_t0 * icethicknesschange_norm).sum() * 1000
            if debug:
                print('fs_huss:', fs_huss)
            # Volume change [km**3 ice]
            bin_volumechange = icethicknesschange_norm * fs_huss / 1000 * glacier_area_t0
            
            print('here you need to update the section')
            print('update thickness using this: icethicknesschange_norm * fs_huss', )
    
        # Otherwise, compute volume change in each bin based on the climatic mass balance
        else:
            bin_volumechange = massbalclim_annual / 1000 * glacier_area_t0        
        if pygem_prms.option_glaciershape == 1:
            # Ice thickness at end of timestep for parabola [m ice]
            #  run in two steps to avoid errors with negative numbers and fractional exponents
            #  H_1 = (H_0**1.5 + delta_Vol * H_0**0.5 / A_0)**(2/3)
            icethickness_t1[glac_idx_t0] = ((icethickness_t0[glac_idx_t0] / 1000)**1.5 + 
                           (icethickness_t0[glac_idx_t0] / 1000)**0.5 * bin_volumechange[glac_idx_t0] / 
                           glacier_area_t0[glac_idx_t0])
            icethickness_t1[icethickness_t1 < 0] = 0
            icethickness_t1[glac_idx_t0] = icethickness_t1[glac_idx_t0]**(2/3) * 1000        
            # Glacier area for parabola [km**2]
            #  A_1 = A_0 * (H_1 / H_0)**0.5
            glacier_area_t1[glac_idx_t0] = (glacier_area_t0[glac_idx_t0] * (icethickness_t1[glac_idx_t0] / 
                                            icethickness_t0[glac_idx_t0])**0.5)
            # Glacier width for parabola [km]
            #  w_1 = w_0 * (A_1 / A_0)
            width_t1[glac_idx_t0] = width_t0[glac_idx_t0] * glacier_area_t1[glac_idx_t0] / glacier_area_t0[glac_idx_t0]
        elif pygem_prms.option_glaciershape == 2:
            # Ice thickness at end of timestep for rectangle [m ice]
            #  H_1 = H_0 + delta_Vol / A_0
            icethickness_t1[glac_idx_t0] = (((icethickness_t0[glac_idx_t0] / 1000) + 
                                             bin_volumechange[glac_idx_t0] / glacier_area_t0[glac_idx_t0]) * 1000)
            # Glacier area constant for rectangle [km**2]
            #  A_1 = A_0
            glacier_area_t1[glac_idx_t0] = glacier_area_t0[glac_idx_t0]
            # Glacier width constant for rectangle [km]
            #  w_1 = w_0
            width_t1[glac_idx_t0] = width_t0[glac_idx_t0]
        elif pygem_prms.option_glaciershape == 3:
            # Ice thickness at end of timestep for triangle [m ice]
            #  run in two steps to avoid errors with negative numbers and fractional exponents
            icethickness_t1[glac_idx_t0] = ((icethickness_t0[glac_idx_t0] / 1000)**2 + 
                           bin_volumechange[glac_idx_t0] * (icethickness_t0[glac_idx_t0] / 1000) / 
                           glacier_area_t0[glac_idx_t0])                                   
            icethickness_t1[icethickness_t1 < 0] = 0
            icethickness_t1[glac_idx_t0] = icethickness_t1[glac_idx_t0]**(1/2) * 1000
            # Glacier area for triangle [km**2]
            #  A_1 = A_0 * H_1 / H_0
            glacier_area_t1[glac_idx_t0] = (glacier_area_t0[glac_idx_t0] * icethickness_t1[glac_idx_t0] / 
                                            icethickness_t0[glac_idx_t0])
            # Glacier width for triangle [km]
            #  w_1 = w_0 * (A_1 / A_0)
            width_t1[glac_idx_t0] = width_t0[glac_idx_t0] * glacier_area_t1[glac_idx_t0] / glacier_area_t0[glac_idx_t0]
        # Ice thickness change [m ice]
        icethickness_change = icethickness_t1 - icethickness_t0
        # Compute the remaining volume change
        bin_volumechange_remaining = bin_volumechange - ((glacier_area_t1 * icethickness_t1 - glacier_area_t0 * 
                                                          icethickness_t0) / 1000)
        # remove values below tolerance to avoid rounding errors
        bin_volumechange_remaining[abs(bin_volumechange_remaining) < pygem_prms.tolerance] = 0
        # Glacier volume change remaining - if less than zero, then needed for retreat
        glacier_volumechange_remaining = bin_volumechange_remaining.sum()    
        return icethickness_t1, glacier_area_t1, width_t1, icethickness_change, glacier_volumechange_remaining