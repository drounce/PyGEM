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
            heights = fl.surface_h.copy()
            section_t0 = fl.section.copy()
            thick_t0 = fl.thick.copy()
            width_t0 = fl.widths_m.copy()
            
            # CONSTANT AREAS
            #  Mass redistribution ignored for calibration and spinup years (glacier properties constant)
            if (self.option_areaconstant == 1) or (year < self.spinupyears) or (year < self.constantarea_years):
                # run mass balance
                glac_bin_massbalclim_annual = self.mb_model.get_annual_mb(heights, fls=self.fls, fl_id=fl_id, 
                                                                              year=year, debug=False)                                
            # MASS REDISTRIBUTION
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
                if len(section_t0.nonzero()[0]) > 0:
                    # Mass redistribution according to Huss empirical curves
                    glac_bin_massbalclim_annual = self.mb_model.get_annual_mb(heights, fls=self.fls, fl_id=fl_id, 
                                                                              year=year, debug=False)        
                    sec_in_year = (self.mb_model.dates_table.loc[12*year:12*(year+1)-1,'daysinmonth'].values.sum() 
                                   * 24 * 3600)
                    self._massredistributionHuss(section_t0, thick_t0, width_t0, glac_bin_massbalclim_annual, 
                                                 self.glac_idx_initial[fl_id], heights, sec_in_year=sec_in_year)                
                    
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

            
    #%%%% ====== START OF MASS REDISTRIBUTION CURVE  
    def _massredistributionHuss(self, section_t0, thick_t0, width_t0, glac_bin_massbalclim_annual, 
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
        glacier_area_t0 = width_t0 * self.fls[0].dx_meter / 1e6
        glacier_area_t0[thick_t0 == 0] = 0
        
        # Annual glacier-wide volume change [km**3]
        #  units: [m w.e.] * (1 km / 1000 m) * (1000 kg / (1 m water * m**2) * (1 m ice * m**2 / 900 kg) * [km**2] 
        #         = km**3 ice    
        glacier_volumechange = (glac_bin_massbalclim_annual / 1000 * sec_in_year * glacier_area_t0).sum()
        
        # For hindcast simulations, volume change is the opposite
        if hindcast == 1:
            glacier_volumechange = -1 * glacier_volumechange
            
        if debug:
            print('\nDebugging Mass Redistribution Huss function\n')
            print('glacier volume change:', glacier_volumechange)
              
        # If volume loss is less than the glacier volume, then redistribute mass loss/gains across the glacier;
        #  otherwise, the glacier disappears (area and thickness were already set to zero above)
        glacier_volume_total = (self.fls[0].section * self.fls[0].dx_meter / 1e9).sum()
        if -1 * glacier_volumechange < glacier_volume_total:
             # Determine where glacier exists            
            glac_idx_t0 = self.fls[0].thick.nonzero()[0]
            
            # Compute ice thickness [m ice], glacier area [km**2] and ice thickness change [m ice] after 
            #  redistribution of gains/losses
            if pygem_prms.option_massredistribution == 1:
                icethickness_change, glacier_volumechange_remaining = (
                        self._massredistributioncurveHuss(section_t0, thick_t0, width_t0, glac_idx_t0,
                                                          glacier_volumechange, glac_bin_massbalclim_annual,
                                                          heights, debug=False))
                if debug:
                    print('\nmax icethickness change:', np.round(icethickness_change.max(),3), 
                          '\nmin icethickness change:', np.round(icethickness_change.min(),3), 
                          '\nvolume remaining:', glacier_volumechange_remaining)
    
            # Glacier retreat
            #  if glacier retreats (ice thickness == 0), volume change needs to be redistributed over glacier again
            while glacier_volumechange_remaining < 0:
                section_t0_retreated = self.fls[0].section.copy()
                thick_t0_retreated = self.fls[0].thick.copy()
                width_t0_retreated = self.fls[0].widths_m.copy()
                glacier_volumechange_remaining_retreated = glacier_volumechange_remaining.copy()
                glac_idx_t0_retreated = thick_t0_retreated.nonzero()[0]  
                glacier_area_t0_retreated = width_t0_retreated * self.fls[0].dx_meter / 1e6
                glacier_area_t0_retreated[thick_t0 == 0] = 0
                # Set climatic mass balance for the case when there are less than 3 bins  
                #  distribute the remaining glacier volume change over the entire glacier (remaining bins)
                massbalclim_retreat = np.zeros(thick_t0_retreated.shape)
                massbalclim_retreat[glac_idx_t0_retreated] = (glacier_volumechange_remaining / 
                                                               glacier_area_t0_retreated.sum() * 1000)
                # Mass redistribution 
                if pygem_prms.option_massredistribution == 1:
                    # Option 1: apply mass redistribution using Huss' empirical geometry change equations
                    icethickness_change, glacier_volumechange_remaining = (
                        self._massredistributioncurveHuss(
                                section_t0_retreated, thick_t0_retreated, width_t0_retreated, glac_idx_t0_retreated, 
                                glacier_volumechange_remaining_retreated, massbalclim_retreat, heights, debug=False))

            # Glacier advances 
            #  based on ice thickness change exceeding threshold
            #  Overview:
            #    1. Add new bin and fill it up to a maximum of terminus average ice thickness
            #    2. If additional volume after adding new bin, then redistribute mass gain across all bins again,
            #       i.e., increase the ice thickness and width
            #    3. Repeat adding a new bin and redistributing the mass until no addiitonal volume is left
            while (icethickness_change > pygem_prms.icethickness_advancethreshold).any() == True: 
                if debug:
                    print('advancing glacier')

                # Record glacier area and ice thickness before advance corrections applied
                section_t0_raw = self.fls[0].section.copy()
                thick_t0_raw = self.fls[0].thick.copy()
                width_t0_raw = self.fls[0].widths_m.copy()
                glacier_area_t0_raw = width_t0_raw * self.fls[0].dx_meter / 1e6
                
                # Index bins that are advancing
                icethickness_change[icethickness_change <= pygem_prms.icethickness_advancethreshold] = 0
                glac_idx_advance = icethickness_change.nonzero()[0]
                
                # Update ice thickness based on maximum advance threshold [m ice]
#                print('before:', np.round(self.fls[0].thick[85:92],1))
                self.fls[0].thick[glac_idx_advance] = (self.fls[0].thick[glac_idx_advance] - 
                               (icethickness_change[glac_idx_advance] - pygem_prms.icethickness_advancethreshold))
                glacier_area_t1 = self.fls[0].widths_m.copy() * self.fls[0].dx_meter / 1e6
                
                # Advance volume [km**3]
                advance_volume = ((glacier_area_t0_raw[glac_idx_advance] * thick_t0_raw[glac_idx_advance] / 1000).sum() 
                                  - (glacier_area_t1[glac_idx_advance] * self.fls[0].thick[glac_idx_advance] 
                                      / 1000).sum())
                # Set the cross sectional area of the next bin
                advance_section = advance_volume * 1e9 / self.fls[0].dx_meter
                
                # Index of bin to add
                glac_idx_t0 = self.fls[0].thick.nonzero()[0]
                min_elev = self.fls[0].surface_h[glac_idx_t0].min()
                glac_idx_bin2add = (
                        np.where(self.fls[0].surface_h == 
                                 self.fls[0].surface_h[np.where(self.fls[0].surface_h < min_elev)[0]].max())[0][0])
                section_2add = self.fls[0].section.copy()
                section_2add[glac_idx_bin2add] = advance_section
                self.fls[0].section = section_2add
#                print('after:', np.round(self.fls[0].thick[85:92],1))                

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
                    terminus_thickness_avg = np.mean(self.fls[0].thick[glac_idx_terminus_removemin])
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
                    terminus_thickness_avg = np.mean(self.fls[0].thick[glac_idx_terminus_removemin])
                
                # If last bin exceeds terminus thickness average then fill up the bin to average and redistribute mass
                if self.fls[0].thick[glac_idx_bin2add] > terminus_thickness_avg:
                    self.fls[0].thick[glac_idx_bin2add] = terminus_thickness_avg
                    # Redistribute remaining mass
                    volume_added2bin = self.fls[0].section[glac_idx_bin2add] * self.fls[0].dx_meter / 1e9
                    advance_volume -= volume_added2bin
    
                # With remaining advance volume, add a bin or redistribute over existing bins if no bins left
                if advance_volume > 0:
                    # Indices for additional bins below the terminus
                    below_glac_idx = np.where(heights < heights[glacier_area_t1 > 0].min())[0]
                    # if no more bins below, then distribute volume over the glacier without further adjustments
                    if len(below_glac_idx) == 0:
                        self.fls[0].section = section_t0_raw
                        advance_volume = 0
                        
                    # otherwise, redistribute mass
                    else:
                        glac_idx_t0 = self.fls[0].thick.nonzero()[0]
                        glacier_area_t0 = self.fls[0].widths_m.copy() * self.fls[0].dx_meter / 1e6
                        glac_bin_massbalclim_annual = np.zeros(self.fls[0].thick.shape)
                        glac_bin_massbalclim_annual[glac_idx_t0] = (glacier_volumechange_remaining / 
                                                                    glacier_area_t0.sum() * 1000)
                        icethickness_change, glacier_volumechange_remaining = (
                            self._massredistributioncurveHuss(
                                    self.fls[0].section.copy(), self.fls[0].thick.copy(), self.fls[0].widths_m.copy(), 
                                    glac_idx_t0, advance_volume, glac_bin_massbalclim_annual, heights, debug=False))
    
    
    def _massredistributioncurveHuss(self, section_t0, thick_t0, width_t0, glac_idx_t0, glacier_volumechange, 
                                     massbalclim_annual, heights, debug=False):
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
        section_t0 : np.ndarray
            Glacier cross-sectional area [m2] from previous year for each elevation bin
        thick_t0 : np.ndarray
            Glacier ice thickness [m] from previous year for each elevation bin
        width_t0 : np.ndarray
            Glacier width [m] from previous year for each elevation bin
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
        # Glacier area used to select parameters
        glacier_area_t0 = width_t0 * self.fls[0].dx_meter / 1e6
        glacier_area_t0[thick_t0 == 0] = 0
        # Apply mass redistribution curve
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
            
        # Otherwise, compute volume change in each bin based on the climatic mass balance
        else:
            bin_volumechange = massbalclim_annual / 1000 * glacier_area_t0
            
#        # Testing heavy retreat
#        print('\n\ndelete me!')
#        bin_volumechange[88] = -500 * self.fls[0].section[88] * self.fls[0].dx_meter / 1e9
#        print('to here')
            
        if debug:
            print('-----\n')
            vol_before = section_t0 * self.fls[0].dx_meter
            np.set_printoptions(suppress=True)
            
#        print('before:', np.round(self.fls[0].thick[85:92],1))

        # Update cross sectional area (updating thickness does not conserve mass in OGGM!) 
        #  volume change divided by length (dx); units m2
        section_change = bin_volumechange * 1e9 / self.fls[0].dx_meter
        self.fls[0].section = utils.clip_min(self.fls[0].section + section_change, 0)
        # Ice thickness change [m ice]
        icethickness_change = self.fls[0].thick - thick_t0
        # Glacier volume
        vol_after = self.fls[0].section * self.fls[0].dx_meter
        
        if debug:
            print('vol_chg_wanted:', bin_volumechange.sum())
            print('vol_chg:', (vol_after - vol_before)/1e9)
            print('\n-----')
        
#        print('after:', np.round(self.fls[0].thick[85:92],1))
        
        # Compute the remaining volume change
        bin_volumechange_remaining = (bin_volumechange - (self.fls[0].section * self.fls[0].dx_meter - 
                                                          section_t0 * self.fls[0].dx_meter) * 1e-9)
        # remove values below tolerance to avoid rounding errors
        bin_volumechange_remaining[abs(bin_volumechange_remaining) < pygem_prms.tolerance] = 0
        # Glacier volume change remaining - if less than zero, then needed for retreat
        glacier_volumechange_remaining = bin_volumechange_remaining.sum()  
        
        if debug:
            print(glacier_volumechange_remaining)

        return icethickness_change, glacier_volumechange_remaining