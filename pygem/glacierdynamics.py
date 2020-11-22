#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 14:00:14 2020

@author: davidrounce
"""
from collections import OrderedDict
from time import gmtime, strftime

import numpy as np
#import pandas as pd
#import netCDF4
import xarray as xr

from oggm import cfg, utils
from oggm.core.flowline import FlowlineModel
from oggm.exceptions import InvalidParamsError
from oggm import __version__
import pygem.pygem_input as pygem_prms

cfg.initialize()


#%%
class MassRedistributionCurveModel(FlowlineModel):
    """Glacier geometry updated using mass redistribution curves; also known as the "delta-h method"

    This uses mass redistribution curves from Huss et al. (2010) to update the glacier geometry
    """

    def __init__(self, flowlines, mb_model=None, y0=0., 
                 inplace=False,
                 debug=True,
                 option_areaconstant=False, spinupyears=pygem_prms.ref_spinupyears, 
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
        self.y0 = 0
        
#        widths_t0 = flowlines[0].widths_m
#        area_v1 = widths_t0 * flowlines[0].dx_meter
#        print('area v1:', area_v1.sum())
#        area_v2 = np.copy(area_v1)
#        area_v2[flowlines[0].thick == 0] = 0
#        print('area v2:', area_v2.sum())
        
        # HERE IS THE STUFF TO RECORD FOR EACH FLOWLINE!
        self.calving_m3_since_y0 = 0.  # total calving since time y0
        
        assert len(flowlines) == 1, 'MassRedistributionCurveModel is not set up for multiple flowlines'
        
        
    def run_until(self, y1, run_single_year=False):
        """Runs the model from the current year up to a given year date y1.

        This function runs the model for the time difference y1-self.y0
        If self.y0 has not been specified at some point, it is 0 and y1 will
        be the time span in years to run the model for.

        Parameters
        ----------
        y1 : float
            Upper time span for how long the model should run
        """                                   
                    
        # We force timesteps to yearly timesteps
        if run_single_year:
            self.updategeometry(y1)
        else:
            years = np.arange(self.yr, y1)
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
        
                    

    def run_until_and_store(self, y1, run_path=None, diag_path=None,
                            store_monthly_step=None):
        """Runs the model and returns intermediate steps in xarray datasets.

        This function repeatedly calls FlowlineModel.run_until for either
        monthly or yearly time steps up till the upper time boundary y1.

        Parameters
        ----------
        y1 : int
            Upper time span for how long the model should run (needs to be
            a full year)
        run_path : str
            Path and filename where to store the model run dataset
        diag_path : str
            Path and filename where to store the model diagnostics dataset
        store_monthly_step : Bool
            If True (False)  model diagnostics will be stored monthly (yearly).
            If unspecified, we follow the update of the MB model, which
            defaults to yearly (see __init__).

        Returns
        -------
        run_ds : xarray.Dataset
            stores the entire glacier geometry. It is useful to visualize the
            glacier geometry or to restart a new run from a modelled geometry.
            The glacier state is stored at the begining of each hydrological
            year (not in between in order to spare disk space).
        diag_ds : xarray.Dataset
            stores a few diagnostic variables such as the volume, area, length
            and ELA of the glacier.
        """

        if int(y1) != y1:
            raise InvalidParamsError('run_until_and_store only accepts '
                                     'integer year dates.')

        if not self.mb_model.hemisphere:
            raise InvalidParamsError('run_until_and_store needs a '
                                     'mass-balance model with an unambiguous '
                                     'hemisphere.')
        # time
        yearly_time = np.arange(np.floor(self.yr), np.floor(y1)+1)

        if store_monthly_step is None:
            store_monthly_step = self.mb_step == 'monthly'

        if store_monthly_step:
            monthly_time = utils.monthly_timeseries(self.yr, y1)
        else:
            monthly_time = np.arange(np.floor(self.yr), np.floor(y1)+1)

        sm = cfg.PARAMS['hydro_month_' + self.mb_model.hemisphere]

        yrs, months = utils.floatyear_to_date(monthly_time)
        cyrs, cmonths = utils.hydrodate_to_calendardate(yrs, months,
                                                        start_month=sm)

        # init output
        if run_path is not None:
            self.to_netcdf(run_path)
        ny = len(yearly_time)
        if ny == 1:
            yrs = [yrs]
            cyrs = [cyrs]
            months = [months]
            cmonths = [cmonths]
        nm = len(monthly_time)
        sects = [(np.zeros((ny, fl.nx)) * np.NaN) for fl in self.fls]
        widths = [(np.zeros((ny, fl.nx)) * np.NaN) for fl in self.fls]
        bucket = [(np.zeros(ny) * np.NaN) for _ in self.fls]
        diag_ds = xr.Dataset()

        # Global attributes
        diag_ds.attrs['description'] = 'OGGM model output'
        diag_ds.attrs['oggm_version'] = __version__
        diag_ds.attrs['calendar'] = '365-day no leap'
        diag_ds.attrs['creation_date'] = strftime("%Y-%m-%d %H:%M:%S",
                                                  gmtime())
        diag_ds.attrs['hemisphere'] = self.mb_model.hemisphere
        diag_ds.attrs['water_level'] = self.water_level

        # Coordinates
        diag_ds.coords['time'] = ('time', monthly_time)
        diag_ds.coords['hydro_year'] = ('time', yrs)
        diag_ds.coords['hydro_month'] = ('time', months)
        diag_ds.coords['calendar_year'] = ('time', cyrs)
        diag_ds.coords['calendar_month'] = ('time', cmonths)

        diag_ds['time'].attrs['description'] = 'Floating hydrological year'
        diag_ds['hydro_year'].attrs['description'] = 'Hydrological year'
        diag_ds['hydro_month'].attrs['description'] = 'Hydrological month'
        diag_ds['calendar_year'].attrs['description'] = 'Calendar year'
        diag_ds['calendar_month'].attrs['description'] = 'Calendar month'

        # Variables and attributes
        diag_ds['volume_m3'] = ('time', np.zeros(nm) * np.NaN)
        diag_ds['volume_m3'].attrs['description'] = 'Total glacier volume'
        diag_ds['volume_m3'].attrs['unit'] = 'm 3'
        if self.is_marine_terminating:
            diag_ds['volume_bsl_m3'] = ('time', np.zeros(nm) * np.NaN)
            diag_ds['volume_bsl_m3'].attrs['description'] = ('Glacier volume '
                                                             'below '
                                                             'sea-level')
            diag_ds['volume_bsl_m3'].attrs['unit'] = 'm 3'
            diag_ds['volume_bwl_m3'] = ('time', np.zeros(nm) * np.NaN)
            diag_ds['volume_bwl_m3'].attrs['description'] = ('Glacier volume '
                                                             'below '
                                                             'water-level')
            diag_ds['volume_bwl_m3'].attrs['unit'] = 'm 3'

        diag_ds['area_m2'] = ('time', np.zeros(nm) * np.NaN)
        diag_ds['area_m2'].attrs['description'] = 'Total glacier area'
        diag_ds['area_m2'].attrs['unit'] = 'm 2'
        diag_ds['length_m'] = ('time', np.zeros(nm) * np.NaN)
        diag_ds['length_m'].attrs['description'] = 'Glacier length'
        diag_ds['length_m'].attrs['unit'] = 'm 3'
        diag_ds['ela_m'] = ('time', np.zeros(nm) * np.NaN)
        diag_ds['ela_m'].attrs['description'] = ('Annual Equilibrium Line '
                                                 'Altitude  (ELA)')
        diag_ds['ela_m'].attrs['unit'] = 'm a.s.l'
        if self.is_tidewater:
            diag_ds['calving_m3'] = ('time', np.zeros(nm) * np.NaN)
            diag_ds['calving_m3'].attrs['description'] = ('Total accumulated '
                                                          'calving flux')
            diag_ds['calving_m3'].attrs['unit'] = 'm 3'
            diag_ds['calving_rate_myr'] = ('time', np.zeros(nm) * np.NaN)
            diag_ds['calving_rate_myr'].attrs['description'] = 'Calving rate'
            diag_ds['calving_rate_myr'].attrs['unit'] = 'm yr-1'

        # Run
        j = 0
        for i, (yr, mo) in enumerate(zip(yearly_time[:-1], months[:-1])):

            # Record initial parameters
            if i == 0:
                diag_ds['volume_m3'].data[i] = self.volume_m3
                diag_ds['area_m2'].data[i] = self.area_m2
                diag_ds['length_m'].data[i] = self.length_m
            
            self.run_until(yr, run_single_year=True)
            # Model run
            if mo == 1:
                for s, w, b, fl in zip(sects, widths, bucket, self.fls):
                    s[j, :] = fl.section
                    w[j, :] = fl.widths_m
                    if self.is_tidewater:
                        try:
                            b[j] = fl.calving_bucket_m3
                        except AttributeError:
                            pass
                j += 1
            # Diagnostics
            diag_ds['volume_m3'].data[i+1] = self.volume_m3
            diag_ds['area_m2'].data[i+1] = self.area_m2
            diag_ds['length_m'].data[i+1] = self.length_m

            if self.is_tidewater:
                diag_ds['calving_m3'].data[i] = self.calving_m3_since_y0
                diag_ds['calving_rate_myr'].data[i] = self.calving_rate_myr
                if self.is_marine_terminating:
                    diag_ds['volume_bsl_m3'].data[i] = self.volume_bsl_m3
                    diag_ds['volume_bwl_m3'].data[i] = self.volume_bwl_m3

        # to datasets
        run_ds = []
        for (s, w, b) in zip(sects, widths, bucket):
            ds = xr.Dataset()
            ds.attrs['description'] = 'OGGM model output'
            ds.attrs['oggm_version'] = __version__
            ds.attrs['calendar'] = '365-day no leap'
            ds.attrs['creation_date'] = strftime("%Y-%m-%d %H:%M:%S",
                                                 gmtime())
            ds.coords['time'] = yearly_time
            ds['time'].attrs['description'] = 'Floating hydrological year'
            varcoords = OrderedDict(time=('time', yearly_time),
                                    year=('time', yearly_time))
            ds['ts_section'] = xr.DataArray(s, dims=('time', 'x'),
                                            coords=varcoords)
            ds['ts_width_m'] = xr.DataArray(w, dims=('time', 'x'),
                                            coords=varcoords)
            if self.is_tidewater:
                ds['ts_calving_bucket_m3'] = xr.DataArray(b, dims=('time', ),
                                                          coords=varcoords)
            run_ds.append(ds)

        # write output?
        if run_path is not None:
            encode = {'ts_section': {'zlib': True, 'complevel': 5},
                      'ts_width_m': {'zlib': True, 'complevel': 5},
                      }
            for i, ds in enumerate(run_ds):
                ds.to_netcdf(run_path, 'a', group='fl_{}'.format(i),
                             encoding=encode)
            # Add other diagnostics
            diag_ds.to_netcdf(run_path, 'a')

        if diag_path is not None:
            diag_ds.to_netcdf(diag_path)

        return run_ds, diag_ds
    
    
    def updategeometry(self, year):
        """Update geometry for a given year"""
        
#        print('year:', year)
            
        # Loop over flowlines
        for fl_id, fl in enumerate(self.fls):

            # Flowline state
            heights = fl.surface_h.copy()
            section_t0 = fl.section.copy()
            thick_t0 = fl.thick.copy()
            width_t0 = fl.widths_m.copy()
            
            # CONSTANT AREAS
            #  Mass redistribution ignored for calibration and spinup years (glacier properties constant)
            if (self.option_areaconstant) or (year < self.spinupyears) or (year < self.constantarea_years):
                # run mass balance
                glac_bin_massbalclim_annual = self.mb_model.get_annual_mb(heights, fls=self.fls, fl_id=fl_id, 
                                                                              year=year, debug=False)                                
            # MASS REDISTRIBUTION
            else:
                # ----- FRONTAL ABLATION!!! -----
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
                    # Annual glacier mass balance [m ice s-1]
                    glac_bin_massbalclim_annual = self.mb_model.get_annual_mb(heights, fls=self.fls, fl_id=fl_id, 
                                                                              year=year, debug=False)   
                    sec_in_year = (self.mb_model.dates_table.loc[12*year:12*(year+1)-1,'daysinmonth'].values.sum() 
                                   * 24 * 3600)
                    
#                    print(' volume change [m3]:', (glac_bin_massbalclim_annual * sec_in_year * 
#                                                  (width_t0 * fl.dx_meter)).sum())
#                    print(glac_bin_masssbalclim_annual)
#                    print(sec_in_year)
#                    print(width_t0.sum())
#                    print(fl.dx_meter)
#                    print(width_t0 * fl.dx_meter)
                    
#                    # Debugging block
#                    debug_years = [71]
#                    if year in debug_years:
#                        print(year, glac_bin_massbalclim_annual)
#                        print('section t0:', section_t0)
#                        print('thick_t0:', thick_t0)
#                        print('width_t0:', width_t0)
#                        print(self.glac_idx_initial[fl_id])
#                        print('heights:', heights)
                    
                    self._massredistributionHuss(section_t0, thick_t0, width_t0, glac_bin_massbalclim_annual, 
                                                 self.glac_idx_initial[fl_id], heights, sec_in_year=sec_in_year)     
                    
            # Record glacier properties (volume [m3], area [m2], thickness [m], width [km])
            #  record the next year's properties as well
            #  'year + 1' used so the glacier properties are consistent with mass balance computations
            year = int(year)  # required to ensure proper indexing with run_until_and_store (10/21/2020)
            glacier_area = fl.widths_m * fl.dx_meter
            glacier_area[fl.thick == 0] = 0
            self.mb_model.glac_bin_area_annual[:,year+1] = glacier_area
            self.mb_model.glac_bin_icethickness_annual[:,year+1] = fl.thick
            self.mb_model.glac_bin_width_annual[:,year+1] = fl.widths_m
            self.mb_model.glac_wide_area_annual[year+1] = glacier_area.sum()
            self.mb_model.glac_wide_volume_annual[year+1] = (fl.section * fl.dx_meter).sum()

            
    #%%%% ====== START OF MASS REDISTRIBUTION CURVE  
    def _massredistributionHuss(self, section_t0, thick_t0, width_t0, glac_bin_massbalclim_annual, 
                                glac_idx_initial, heights, debug=False, hindcast=0, sec_in_year=365*24*3600):
        """
        Mass redistribution according to empirical equations from Huss and Hock (2015) accounting for retreat/advance.
        glac_idx_initial is required to ensure that the glacier does not advance to area where glacier did not exist 
        before (e.g., retreat and advance over a vertical cliff)
        
        Note: since OGGM uses the DEM, heights along the flowline do not necessarily decrease, i.e., there can be
        overdeepenings along the flowlines that occur as the glacier retreats. This is problematic for 'adding' a bin 
        downstream in cases of glacier advance because you'd be moving new ice to a higher elevation. To avoid this 
        unrealistic case, in the event that this would occur, the overdeepening will simply fill up with ice first until
        it reaches an elevation where it would put new ice into a downstream bin.

        Parameters
        ----------
        section_t0 : np.ndarray
            Glacier cross-sectional area (m2) from previous year for each elevation bin
        thick_t0 : np.ndarray
            Glacier ice thickness [m] from previous year for each elevation bin
        width_t0 : np.ndarray
            Glacier width [m] from previous year for each elevation bin
        glac_bin_massbalclim_annual : np.ndarray
            Climatic mass balance [m ice s-1] for each elevation bin and year
        glac_idx_initial : np.ndarray
            Initial glacier indices
        debug : Boolean
            option to turn on print statements for development or debugging of code (default False)
            
        Returns
        -------
        Updates the flowlines automatically, so does not return anything
        """        
        # Glacier area [m2]
        glacier_area_t0 = width_t0 * self.fls[0].dx_meter
        glacier_area_t0[thick_t0 == 0] = 0
        
        # Annual glacier-wide volume change [m3]
        #  units: [m ice / s] * [s] * [m2] = m3 ice
        glacier_volumechange = (glac_bin_massbalclim_annual * sec_in_year * glacier_area_t0).sum()
        
        # For hindcast simulations, volume change is the opposite
        if hindcast == 1:
            glacier_volumechange = -1 * glacier_volumechange
            
        if debug:
            print('\nDebugging Mass Redistribution Huss function\n')
            print('glacier volume change:', glacier_volumechange)
              
        # If volume loss is less than the glacier volume, then redistribute mass loss/gains across the glacier;
        #  otherwise, the glacier disappears (area and thickness were already set to zero above)
        glacier_volume_total = (self.fls[0].section * self.fls[0].dx_meter).sum()
        if -1 * glacier_volumechange < glacier_volume_total:
             # Determine where glacier exists            
            glac_idx_t0 = self.fls[0].thick.nonzero()[0]
            
            # Compute ice thickness [m ice], glacier area [m2], ice thickness change [m ice] after redistribution
            if pygem_prms.option_massredistribution == 1:
                icethickness_change, glacier_volumechange_remaining = (
                        self._massredistributioncurveHuss(section_t0, thick_t0, width_t0, glac_idx_t0,
                                                          glacier_volumechange, glac_bin_massbalclim_annual,
                                                          heights, debug=False))
                if debug:
#                    print('ice thickness change:', icethickness_change)
                    print('\nmax icethickness change:', np.round(icethickness_change.max(),3), 
                          '\nmin icethickness change:', np.round(icethickness_change.min(),3), 
                          '\nvolume remaining:', glacier_volumechange_remaining)
                    nloop = 0
    
            # Glacier retreat
            #  if glacier retreats (ice thickness == 0), volume change needs to be redistributed over glacier again
            while glacier_volumechange_remaining < 0:
                
                if debug:
                    print('\n\nGlacier retreating (loop ' + str(nloop) + '):')
                
                section_t0_retreated = self.fls[0].section.copy()
                thick_t0_retreated = self.fls[0].thick.copy()
                width_t0_retreated = self.fls[0].widths_m.copy()
                glacier_volumechange_remaining_retreated = glacier_volumechange_remaining.copy()
                glac_idx_t0_retreated = thick_t0_retreated.nonzero()[0]  
                glacier_area_t0_retreated = width_t0_retreated * self.fls[0].dx_meter
                glacier_area_t0_retreated[thick_t0 == 0] = 0
                # Set climatic mass balance for the case when there are less than 3 bins  
                #  distribute the remaining glacier volume change over the entire glacier (remaining bins)
                massbalclim_retreat = np.zeros(thick_t0_retreated.shape)
                massbalclim_retreat[glac_idx_t0_retreated] = (glacier_volumechange_remaining / 
                                                               glacier_area_t0_retreated.sum() / sec_in_year)
                # Mass redistribution 
                if pygem_prms.option_massredistribution == 1:
                    # Option 1: apply mass redistribution using Huss' empirical geometry change equations
                    icethickness_change, glacier_volumechange_remaining = (
                        self._massredistributioncurveHuss(
                                section_t0_retreated, thick_t0_retreated, width_t0_retreated, glac_idx_t0_retreated, 
                                glacier_volumechange_remaining_retreated, massbalclim_retreat, heights, debug=False))
                    # Avoid rounding errors that get loop stuck
                    if abs(glacier_volumechange_remaining) < 1:
                        glacier_volumechange_remaining = 0
                    
                    if debug:
                        print('ice thickness change:', icethickness_change)
                        print('\nmax icethickness change:', np.round(icethickness_change.max(),3), 
                              '\nmin icethickness change:', np.round(icethickness_change.min(),3), 
                              '\nvolume remaining:', glacier_volumechange_remaining)
                        nloop += 1

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
                glacier_area_t0_raw = width_t0_raw * self.fls[0].dx_meter
                
                if debug:
                    print('\n\nthickness t0:', thick_t0_raw)
                    print('glacier area t0:', glacier_area_t0_raw)
                    print('width_t0_raw:', width_t0_raw,'\n\n')
                
                # Index bins that are advancing
                icethickness_change[icethickness_change <= pygem_prms.icethickness_advancethreshold] = 0
                glac_idx_advance = icethickness_change.nonzero()[0]
                
                # Update ice thickness based on maximum advance threshold [m ice]
                self.fls[0].thick[glac_idx_advance] = (self.fls[0].thick[glac_idx_advance] - 
                               (icethickness_change[glac_idx_advance] - pygem_prms.icethickness_advancethreshold))
                glacier_area_t1 = self.fls[0].widths_m.copy() * self.fls[0].dx_meter
                
                # Advance volume [m3]
                advance_volume = ((glacier_area_t0_raw[glac_idx_advance] * thick_t0_raw[glac_idx_advance]).sum() 
                                  - (glacier_area_t1[glac_idx_advance] * self.fls[0].thick[glac_idx_advance]).sum())
                
                # Set the cross sectional area of the next bin
                advance_section = advance_volume / self.fls[0].dx_meter
                
                # Index of bin to add
                glac_idx_t0 = self.fls[0].thick.nonzero()[0]
                min_elev = self.fls[0].surface_h[glac_idx_t0].min()
                glac_idx_bin2add = (
                        np.where(self.fls[0].surface_h == 
                                 self.fls[0].surface_h[np.where(self.fls[0].surface_h < min_elev)[0]].max())[0][0])
                section_2add = self.fls[0].section.copy()
                section_2add[glac_idx_bin2add] = advance_section
                self.fls[0].section = section_2add              

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

                # Average area of glacier terminus [m2]
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
                    volume_added2bin = self.fls[0].section[glac_idx_bin2add] * self.fls[0].dx_meter
                    advance_volume -= volume_added2bin
    
                # With remaining advance volume, add a bin or redistribute over existing bins if no bins left
                if advance_volume > 0:
                    # Indices for additional bins below the terminus
                    glac_idx_t1 = np.where(glacier_area_t1 > 0)[0]
                    below_glac_idx = np.where(heights < heights[glac_idx_t1].min())[0]

                    # if no more bins below, then distribute volume over the glacier without further adjustments
                    #  this occurs with OGGM flowlines when the terminus is in an overdeepening, so we just fill up 
                    #  the overdeepening
                    if len(below_glac_idx) == 0:
                        # Revert to the initial section, which also updates the thickness and width automatically
                        self.fls[0].section = section_t0_raw
                        
                        # set icethickness change and advance_volume to 0 to break the loop
                        icethickness_change[icethickness_change > 0] = 0
                        advance_volume = 0
                        
                    # otherwise, redistribute mass
                    else:
                        glac_idx_t0 = self.fls[0].thick.nonzero()[0]
                        glacier_area_t0 = self.fls[0].widths_m.copy() * self.fls[0].dx_meter
                        glac_bin_massbalclim_annual = np.zeros(self.fls[0].thick.shape)
                        glac_bin_massbalclim_annual[glac_idx_t0] = (glacier_volumechange_remaining / 
                                                                    glacier_area_t0.sum() / sec_in_year)
                        icethickness_change, glacier_volumechange_remaining = (
                            self._massredistributioncurveHuss(
                                    self.fls[0].section.copy(), self.fls[0].thick.copy(), self.fls[0].widths_m.copy(), 
                                    glac_idx_t0, advance_volume, glac_bin_massbalclim_annual, heights, debug=False))
    
    
    def _massredistributioncurveHuss(self, section_t0, thick_t0, width_t0, glac_idx_t0, glacier_volumechange, 
                                     massbalclim_annual, heights, debug=False):
        """
        Apply the mass redistribution curves from Huss and Hock (2015).
        This is paired with massredistributionHuss, which takes into consideration retreat and advance.
        
        Parameters
        ----------
        section_t0 : np.ndarray
            Glacier cross-sectional area [m2] from previous year for each elevation bin
        thick_t0 : np.ndarray
            Glacier ice thickness [m] from previous year for each elevation bin
        width_t0 : np.ndarray
            Glacier width [m] from previous year for each elevation bin
        glac_idx_t0 : np.ndarray
            glacier indices for present timestep
        glacier_volumechange : float
            glacier-wide volume change [m3 ice] based on the annual climatic mass balance
        massbalclim_annual : np.ndarray
            Annual climatic mass balance [m ice s-1] for each elevation bin for a single year
        Returns
        -------
        icethickness_change : np.ndarray
            Ice thickness change [m] for each elevation bin
        glacier_volumechange_remaining : float
            Glacier volume change remaining [m3 ice]; occurs if there is less ice than melt in a bin, i.e., retreat
        """ 
          
        if debug:
            print('\nDebugging mass redistribution curve Huss\n')

        # Apply Huss redistribution if there are at least 3 elevation bands; otherwise, use the mass balance        
        # Glacier area used to select parameters
        glacier_area_t0 = width_t0 * self.fls[0].dx_meter
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
            #  units: m3 / (m2 * [-]) * (1000 m / 1 km) = m ice
            fs_huss = glacier_volumechange / (glacier_area_t0 * icethicknesschange_norm).sum()
            if debug:
                print('fs_huss:', fs_huss)
            # Volume change [m3 ice]
            bin_volumechange = icethicknesschange_norm * fs_huss * glacier_area_t0
            
        # Otherwise, compute volume change in each bin based on the climatic mass balance
        else:
            bin_volumechange = massbalclim_annual * glacier_area_t0
            
        if debug:
            print('-----\n')
            vol_before = section_t0 * self.fls[0].dx_meter

        # Update cross sectional area (updating thickness does not conserve mass in OGGM!) 
        #  volume change divided by length (dx); units m2
        section_change = bin_volumechange / self.fls[0].dx_meter
        self.fls[0].section = utils.clip_min(self.fls[0].section + section_change, 0)
        # Ice thickness change [m ice]
        icethickness_change = self.fls[0].thick - thick_t0
        # Glacier volume
        vol_after = self.fls[0].section * self.fls[0].dx_meter
        
        if debug:
            print('vol_chg_wanted:', bin_volumechange.sum())
            print('vol_chg:', (vol_after.sum() - vol_before.sum()))
            print('\n-----')
        
        # Compute the remaining volume change
        bin_volumechange_remaining = (bin_volumechange - (self.fls[0].section * self.fls[0].dx_meter - 
                                                          section_t0 * self.fls[0].dx_meter))
        # remove values below tolerance to avoid rounding errors
        bin_volumechange_remaining[abs(bin_volumechange_remaining) < pygem_prms.tolerance] = 0
        # Glacier volume change remaining - if less than zero, then needed for retreat
        glacier_volumechange_remaining = bin_volumechange_remaining.sum()  
        
        if debug:
            print(glacier_volumechange_remaining)

        return icethickness_change, glacier_volumechange_remaining
    
    
#%%
## ------ FLOWLINEMODEL FOR MODEL DIAGNOSTICS WITH OGGM (10/30/2020) -----
#import copy
#from functools import partial
#from oggm.core.inversion import find_sia_flux_from_thickness
#
#class FlowlineModel(object):
#    """Interface to the actual model"""
#
#    def __init__(self, flowlines, mb_model=None, y0=0., glen_a=None,
#                 fs=None, inplace=False, smooth_trib_influx=True,
#                 is_tidewater=False, is_lake_terminating=False,
#                 mb_elev_feedback='annual', check_for_boundaries=None,
#                 water_level=None):
#        """Create a new flowline model from the flowlines and a MB model.
#        Parameters
#        ----------
#        flowlines : list
#            a list of :py:class:`oggm.Flowline` instances, sorted by order
#        mb_model : :py:class:`oggm.core.massbalance.MassBalanceModel`
#            the MB model to use
#        y0 : int
#            the starting year of the simulation
#        glen_a : float
#            glen's parameter A
#        fs: float
#            sliding parameter
#        inplace : bool
#            whether or not to make a copy of the flowline objects for the run
#            setting to True implies that your objects will be modified at run
#            time by the model (can help to spare memory)
#        smooth_trib_influx : bool
#            whether to smooth the mass influx from the incoming tributary.
#            The fefault is to use a gaussian kernel on a 9 grid points
#            window.
#        is_tidewater: bool, default: False
#            is this a tidewater glacier?
#        is_lake_terminating: bool, default: False
#            is this a lake terminating glacier?
#        mb_elev_feedback : str, default: 'annual'
#            'never', 'always', 'annual', or 'monthly': how often the
#            mass-balance should be recomputed from the mass balance model.
#            'Never' is equivalent to 'annual' but without elevation feedback
#            at all (the heights are taken from the first call).
#        check_for_boundaries : bool
#            whether the model should raise an error when the glacier exceeds
#            the domain boundaries. The default is to follow
#            PARAMS['error_when_glacier_reaches_boundaries']
#        """
#        
#        widths_t0 = flowlines[0].widths_m
#        area_v1 = widths_t0 * flowlines[0].dx_meter
#        print('area v1:', area_v1.sum())
#        area_v2 = np.copy(area_v1)
#        area_v2[flowlines[0].thick == 0] = 0
#        print('area v2:', area_v2.sum())
#        print('thickness:', flowlines[0].thick)
#
#        self.is_tidewater = is_tidewater
#        self.is_lake_terminating = is_lake_terminating
#        self.is_marine_terminating = is_tidewater and not is_lake_terminating
#
#        if water_level is None:
#            self.water_level = 0
#            if self.is_lake_terminating:
#                if not flowlines[-1].has_ice():
#                    raise InvalidParamsError('Set `water_level` for lake '
#                                             'terminating glaciers in '
#                                             'idealized runs')
#                # Arbitrary water level 1m below last grid points elevation
#                min_h = flowlines[-1].surface_h[flowlines[-1].thick > 0][-1]
#                self.water_level = (min_h -
#                                    cfg.PARAMS['free_board_lake_terminating'])
#        else:
#            self.water_level = water_level
#
#        # Mass balance
#        self.mb_elev_feedback = mb_elev_feedback.lower()
#        if self.mb_elev_feedback in ['never', 'annual']:
#            self.mb_step = 'annual'
#        elif self.mb_elev_feedback in ['always', 'monthly']:
#            self.mb_step = 'monthly'
#        self.mb_model = mb_model
#
#        # Defaults
#        if glen_a is None:
#            glen_a = cfg.PARAMS['glen_a']
#        if fs is None:
#            fs = cfg.PARAMS['fs']
#        self.glen_a = glen_a
#        self.fs = fs
#        self.glen_n = cfg.PARAMS['glen_n']
#        self.rho = cfg.PARAMS['ice_density']
#        if check_for_boundaries is None:
#            check_for_boundaries = cfg.PARAMS[('error_when_glacier_reaches_'
#                                               'boundaries')]
#        self.check_for_boundaries = check_for_boundaries
#
#        # we keep glen_a as input, but for optimisation we stick to "fd"
#        self._fd = 2. / (cfg.PARAMS['glen_n']+2) * self.glen_a
#
#        # Calving shenanigans
#        self.calving_m3_since_y0 = 0.  # total calving since time y0
#        self.calving_rate_myr = 0.
#
#        self.y0 = None
#        self.t = None
#        self.reset_y0(y0)
#
#        self.fls = None
#        self._tributary_indices = None
#        self.reset_flowlines(flowlines, inplace=inplace,
#                             smooth_trib_influx=smooth_trib_influx)
#
#    @property
#    def mb_model(self):
#        return self._mb_model
#
#    @mb_model.setter
#    def mb_model(self, value):
#        # We need a setter because the MB func is stored as an attr too
#        _mb_call = None
#        if value:
#            if self.mb_elev_feedback in ['always', 'monthly']:
#                _mb_call = value.get_monthly_mb
#            elif self.mb_elev_feedback in ['annual', 'never']:
#                _mb_call = value.get_annual_mb
#            else:
#                raise ValueError('mb_elev_feedback not understood')
#        self._mb_model = value
#        self._mb_call = _mb_call
#        self._mb_current_date = None
#        self._mb_current_out = dict()
#        self._mb_current_heights = dict()
#
#    def reset_y0(self, y0):
#        """Reset the initial model time"""
#        self.y0 = y0
#        self.t = 0
#
#    def reset_flowlines(self, flowlines, inplace=False,
#                        smooth_trib_influx=True):
#        """Reset the initial model flowlines"""
#
#        if not inplace:
#            flowlines = copy.deepcopy(flowlines)
#
#        try:
#            len(flowlines)
#        except TypeError:
#            flowlines = [flowlines]
#
#        self.fls = flowlines
#
#        # list of tributary coordinates and stuff
#        trib_ind = []
#        for fl in self.fls:
#            # Important also
#            fl.water_level = self.water_level
#            if fl.flows_to is None:
#                trib_ind.append((None, None, None, None))
#                continue
#            idl = self.fls.index(fl.flows_to)
#            ide = fl.flows_to_indice
#            if not smooth_trib_influx:
#                gk = 1
#                id0 = ide
#                id1 = ide+1
#            elif fl.flows_to.nx >= 9:
#                gk = cfg.GAUSSIAN_KERNEL[9]
#                id0 = ide-4
#                id1 = ide+5
#            elif fl.flows_to.nx >= 7:
#                gk = cfg.GAUSSIAN_KERNEL[7]
#                id0 = ide-3
#                id1 = ide+4
#            elif fl.flows_to.nx >= 5:
#                gk = cfg.GAUSSIAN_KERNEL[5]
#                id0 = ide-2
#                id1 = ide+3
#            trib_ind.append((idl, id0, id1, gk))
#
#        self._tributary_indices = trib_ind
#
#    @property
#    def yr(self):
#        return self.y0 + self.t / cfg.SEC_IN_YEAR
#
#    @property
#    def area_m2(self):
#        return np.sum([f.area_m2 for f in self.fls])
#
#    @property
#    def volume_m3(self):
#        return np.sum([f.volume_m3 for f in self.fls])
#
#    @property
#    def volume_km3(self):
#        return self.volume_m3 * 1e-9
#
#    @property
#    def volume_bsl_m3(self):
#        return np.sum([f.volume_bsl_m3 for f in self.fls])
#
#    @property
#    def volume_bsl_km3(self):
#        return self.volume_bsl_m3 * 1e-9
#
#    @property
#    def volume_bwl_m3(self):
#        return np.sum([f.volume_bwl_m3 for f in self.fls])
#
#    @property
#    def volume_bwl_km3(self):
#        return self.volume_bwl_m3 * 1e-9
#
#    @property
#    def area_km2(self):
#        return self.area_m2 * 1e-6
#
#    @property
#    def length_m(self):
#        return self.fls[-1].length_m
#
#    def get_mb(self, heights, year=None, fl_id=None, fls=None):
#        """Get the mass balance at the requested height and time.
#        Optimized so that no mb model call is necessary at each step.
#        """
#
#        # Do we even have to optimise?
#        if self.mb_elev_feedback == 'always':
#            return self._mb_call(heights, year=year, fl_id=fl_id, fls=fls)
#
#        # Ok, user asked for it
#        if fl_id is None:
#            raise ValueError('Need fls_id')
#
#        if self.mb_elev_feedback == 'never':
#            # The very first call we take the heights
#            if fl_id not in self._mb_current_heights:
#                # We need to reset just this tributary
#                self._mb_current_heights[fl_id] = heights
#            # All calls we replace
#            heights = self._mb_current_heights[fl_id]
#
#        date = utils.floatyear_to_date(year)
#        if self.mb_elev_feedback in ['annual', 'never']:
#            # ignore month changes
#            date = (date[0], date[0])
#
#        if self._mb_current_date == date:
#            if fl_id not in self._mb_current_out:
#                # We need to reset just this tributary
#                self._mb_current_out[fl_id] = self._mb_call(heights,
#                                                            year=year,
#                                                            fl_id=fl_id,
#                                                            fls=fls)
#        else:
#            # We need to reset all
#            self._mb_current_date = date
#            self._mb_current_out = dict()
#            self._mb_current_out[fl_id] = self._mb_call(heights,
#                                                        year=year,
#                                                        fl_id=fl_id,
#                                                        fls=fls)
#
#        return self._mb_current_out[fl_id]
#
#    def to_netcdf(self, path):
#        """Creates a netcdf group file storing the state of the model."""
#
#        flows_to_id = []
#        for trib in self._tributary_indices:
#            flows_to_id.append(trib[0] if trib[0] is not None else -1)
#
#        ds = xr.Dataset()
#        try:
#            ds.attrs['description'] = 'OGGM model output'
#            ds.attrs['oggm_version'] = __version__
#            ds.attrs['calendar'] = '365-day no leap'
#            ds.attrs['creation_date'] = strftime("%Y-%m-%d %H:%M:%S", gmtime())
#            ds['flowlines'] = ('flowlines', np.arange(len(flows_to_id)))
#            ds['flows_to_id'] = ('flowlines', flows_to_id)
#            ds.to_netcdf(path)
#            for i, fl in enumerate(self.fls):
#                ds = fl.to_dataset()
#                ds.to_netcdf(path, 'a', group='fl_{}'.format(i))
#        finally:
#            ds.close()
#
#    def check_domain_end(self):
#        """Returns False if the glacier reaches the domains bound."""
#        return np.isclose(self.fls[-1].thick[-1], 0)
#
#    def step(self, dt):
#        """Advance the numerical simulation of one single step.
#        Important: the step dt is a maximum boundary that is *not* guaranteed
#        to be met if dt is too large for the underlying numerical
#        implementation. However, ``step(dt)`` should never cross the desired
#        time step, i.e. if dt is small enough to ensure stability, step
#        should match it.
#        The caller will know how much has been actually advanced by looking
#        at the output of ``step()`` or by monitoring ``self.t`` or `self.yr``
#        Parameters
#        ----------
#        dt : float
#             the step length in seconds
#        Returns
#        -------
#        the actual dt chosen by the numerical implementation. Guaranteed to
#        be dt or lower.
#        """
#        raise NotImplementedError
#
#    def run_until(self, y1):
#        """Runs the model from the current year up to a given year date y1.
#        This function runs the model for the time difference y1-self.y0
#        If self.y0 has not been specified at some point, it is 0 and y1 will
#        be the time span in years to run the model for.
#        Parameters
#        ----------
#        y1 : float
#            Upper time span for how long the model should run
#        """
#
#        # We force timesteps to monthly frequencies for consistent results
#        # among use cases (monthly or yearly output) and also to prevent
#        # "too large" steps in the adaptive scheme.
#        ts = utils.monthly_timeseries(self.yr, y1)
#
#        # Add the last date to be sure we end on it
#        ts = np.append(ts, y1)
#
#        # Loop over the steps we want to meet
#        for y in ts:
#            t = (y - self.y0) * cfg.SEC_IN_YEAR
#            # because of CFL, step() doesn't ensure that the end date is met
#            # lets run the steps until we reach our desired date
#            while self.t < t:
#                self.step(t-self.t)
#
#            # Check for domain bounds
#            if self.check_for_boundaries:
#                if self.fls[-1].thick[-1] > 10:
#                    raise RuntimeError('Glacier exceeds domain boundaries, '
#                                       'at year: {}'.format(self.yr))
#
#            # Check for NaNs
#            for fl in self.fls:
#                if np.any(~np.isfinite(fl.thick)):
#                    raise FloatingPointError('NaN in numerical solution, '
#                                             'at year: {}'.format(self.yr))
#
#    def run_until_and_store(self, y1, run_path=None, diag_path=None,
#                            store_monthly_step=None):
#        """Runs the model and returns intermediate steps in xarray datasets.
#        This function repeatedly calls FlowlineModel.run_until for either
#        monthly or yearly time steps up till the upper time boundary y1.
#        Parameters
#        ----------
#        y1 : int
#            Upper time span for how long the model should run (needs to be
#            a full year)
#        run_path : str
#            Path and filename where to store the model run dataset
#        diag_path : str
#            Path and filename where to store the model diagnostics dataset
#        store_monthly_step : Bool
#            If True (False)  model diagnostics will be stored monthly (yearly).
#            If unspecified, we follow the update of the MB model, which
#            defaults to yearly (see __init__).
#        Returns
#        -------
#        run_ds : xarray.Dataset
#            stores the entire glacier geometry. It is useful to visualize the
#            glacier geometry or to restart a new run from a modelled geometry.
#            The glacier state is stored at the begining of each hydrological
#            year (not in between in order to spare disk space).
#        diag_ds : xarray.Dataset
#            stores a few diagnostic variables such as the volume, area, length
#            and ELA of the glacier.
#        """
#
#        if int(y1) != y1:
#            raise InvalidParamsError('run_until_and_store only accepts '
#                                     'integer year dates.')
#
#        if not self.mb_model.hemisphere:
#            raise InvalidParamsError('run_until_and_store needs a '
#                                     'mass-balance model with an unambiguous '
#                                     'hemisphere.')
#        # time
#        yearly_time = np.arange(np.floor(self.yr), np.floor(y1)+1)
#
#        if store_monthly_step is None:
#            store_monthly_step = self.mb_step == 'monthly'
#
#        if store_monthly_step:
#            monthly_time = utils.monthly_timeseries(self.yr, y1)
#        else:
#            monthly_time = np.arange(np.floor(self.yr), np.floor(y1)+1)
#
#        sm = cfg.PARAMS['hydro_month_' + self.mb_model.hemisphere]
#
#        yrs, months = utils.floatyear_to_date(monthly_time)
#        cyrs, cmonths = utils.hydrodate_to_calendardate(yrs, months,
#                                                        start_month=sm)
#
#        # init output
#        if run_path is not None:
#            self.to_netcdf(run_path)
#        ny = len(yearly_time)
#        if ny == 1:
#            yrs = [yrs]
#            cyrs = [cyrs]
#            months = [months]
#            cmonths = [cmonths]
#        nm = len(monthly_time)
#        sects = [(np.zeros((ny, fl.nx)) * np.NaN) for fl in self.fls]
#        widths = [(np.zeros((ny, fl.nx)) * np.NaN) for fl in self.fls]
#        bucket = [(np.zeros(ny) * np.NaN) for _ in self.fls]
#        diag_ds = xr.Dataset()
#
#        # Global attributes
#        diag_ds.attrs['description'] = 'OGGM model output'
#        diag_ds.attrs['oggm_version'] = __version__
#        diag_ds.attrs['calendar'] = '365-day no leap'
#        diag_ds.attrs['creation_date'] = strftime("%Y-%m-%d %H:%M:%S",
#                                                  gmtime())
#        diag_ds.attrs['hemisphere'] = self.mb_model.hemisphere
#        diag_ds.attrs['water_level'] = self.water_level
#
#        # Coordinates
#        diag_ds.coords['time'] = ('time', monthly_time)
#        diag_ds.coords['hydro_year'] = ('time', yrs)
#        diag_ds.coords['hydro_month'] = ('time', months)
#        diag_ds.coords['calendar_year'] = ('time', cyrs)
#        diag_ds.coords['calendar_month'] = ('time', cmonths)
#
#        diag_ds['time'].attrs['description'] = 'Floating hydrological year'
#        diag_ds['hydro_year'].attrs['description'] = 'Hydrological year'
#        diag_ds['hydro_month'].attrs['description'] = 'Hydrological month'
#        diag_ds['calendar_year'].attrs['description'] = 'Calendar year'
#        diag_ds['calendar_month'].attrs['description'] = 'Calendar month'
#
#        # Variables and attributes
#        diag_ds['volume_m3'] = ('time', np.zeros(nm) * np.NaN)
#        diag_ds['volume_m3'].attrs['description'] = 'Total glacier volume'
#        diag_ds['volume_m3'].attrs['unit'] = 'm 3'
#        if self.is_marine_terminating:
#            diag_ds['volume_bsl_m3'] = ('time', np.zeros(nm) * np.NaN)
#            diag_ds['volume_bsl_m3'].attrs['description'] = ('Glacier volume '
#                                                             'below '
#                                                             'sea-level')
#            diag_ds['volume_bsl_m3'].attrs['unit'] = 'm 3'
#            diag_ds['volume_bwl_m3'] = ('time', np.zeros(nm) * np.NaN)
#            diag_ds['volume_bwl_m3'].attrs['description'] = ('Glacier volume '
#                                                             'below '
#                                                             'water-level')
#            diag_ds['volume_bwl_m3'].attrs['unit'] = 'm 3'
#
#        diag_ds['area_m2'] = ('time', np.zeros(nm) * np.NaN)
#        diag_ds['area_m2'].attrs['description'] = 'Total glacier area'
#        diag_ds['area_m2'].attrs['unit'] = 'm 2'
#        diag_ds['length_m'] = ('time', np.zeros(nm) * np.NaN)
#        diag_ds['length_m'].attrs['description'] = 'Glacier length'
#        diag_ds['length_m'].attrs['unit'] = 'm 3'
#        diag_ds['ela_m'] = ('time', np.zeros(nm) * np.NaN)
#        diag_ds['ela_m'].attrs['description'] = ('Annual Equilibrium Line '
#                                                 'Altitude  (ELA)')
#        diag_ds['ela_m'].attrs['unit'] = 'm a.s.l'
#        if self.is_tidewater:
#            diag_ds['calving_m3'] = ('time', np.zeros(nm) * np.NaN)
#            diag_ds['calving_m3'].attrs['description'] = ('Total accumulated '
#                                                          'calving flux')
#            diag_ds['calving_m3'].attrs['unit'] = 'm 3'
#            diag_ds['calving_rate_myr'] = ('time', np.zeros(nm) * np.NaN)
#            diag_ds['calving_rate_myr'].attrs['description'] = 'Calving rate'
#            diag_ds['calving_rate_myr'].attrs['unit'] = 'm yr-1'
#
#        # Run
#        j = 0
#        for i, (yr, mo) in enumerate(zip(monthly_time, months)):
#            self.run_until(yr)
#            # Model run
#            if mo == 1:
#                for s, w, b, fl in zip(sects, widths, bucket, self.fls):
#                    s[j, :] = fl.section
#                    w[j, :] = fl.widths_m
#                    if self.is_tidewater:
#                        try:
#                            b[j] = fl.calving_bucket_m3
#                        except AttributeError:
#                            pass
#                j += 1
#            # Diagnostics
#            diag_ds['volume_m3'].data[i] = self.volume_m3
#            diag_ds['area_m2'].data[i] = self.area_m2
#            diag_ds['length_m'].data[i] = self.length_m
#            try:
#                ela_m = self.mb_model.get_ela(year=yr, fls=self.fls,
#                                              fl_id=len(self.fls)-1)
#                diag_ds['ela_m'].data[i] = ela_m
#            except BaseException:
#                # We really don't want to stop the model for some ELA issues
#                diag_ds['ela_m'].data[i] = np.NaN
#
#            if self.is_tidewater:
#                diag_ds['calving_m3'].data[i] = self.calving_m3_since_y0
#                diag_ds['calving_rate_myr'].data[i] = self.calving_rate_myr
#                if self.is_marine_terminating:
#                    diag_ds['volume_bsl_m3'].data[i] = self.volume_bsl_m3
#                    diag_ds['volume_bwl_m3'].data[i] = self.volume_bwl_m3
#
#        # to datasets
#        run_ds = []
#        for (s, w, b) in zip(sects, widths, bucket):
#            ds = xr.Dataset()
#            ds.attrs['description'] = 'OGGM model output'
#            ds.attrs['oggm_version'] = __version__
#            ds.attrs['calendar'] = '365-day no leap'
#            ds.attrs['creation_date'] = strftime("%Y-%m-%d %H:%M:%S",
#                                                 gmtime())
#            ds.coords['time'] = yearly_time
#            ds['time'].attrs['description'] = 'Floating hydrological year'
#            varcoords = OrderedDict(time=('time', yearly_time),
#                                    year=('time', yearly_time))
#            ds['ts_section'] = xr.DataArray(s, dims=('time', 'x'),
#                                            coords=varcoords)
#            ds['ts_width_m'] = xr.DataArray(w, dims=('time', 'x'),
#                                            coords=varcoords)
#            if self.is_tidewater:
#                ds['ts_calving_bucket_m3'] = xr.DataArray(b, dims=('time', ),
#                                                          coords=varcoords)
#            run_ds.append(ds)
#
#        # write output?
#        if run_path is not None:
#            encode = {'ts_section': {'zlib': True, 'complevel': 5},
#                      'ts_width_m': {'zlib': True, 'complevel': 5},
#                      }
#            for i, ds in enumerate(run_ds):
#                ds.to_netcdf(run_path, 'a', group='fl_{}'.format(i),
#                             encoding=encode)
#            # Add other diagnostics
#            diag_ds.to_netcdf(run_path, 'a')
#
#        if diag_path is not None:
#            diag_ds.to_netcdf(diag_path)
#
#        return run_ds, diag_ds
#
#    def run_until_equilibrium(self, rate=0.001, ystep=5, max_ite=200):
#        """ Runs the model until an equilibrium state is reached.
#        Be careful: This only works for CONSTANT (not time-dependant)
#        mass-balance models.
#        Otherwise the returned state will not be in equilibrium! Don't try to
#        calculate an equilibrium state with a RandomMassBalance model!
#        """
#
#        ite = 0
#        was_close_zero = 0
#        t_rate = 1
#        while (t_rate > rate) and (ite <= max_ite) and (was_close_zero < 5):
#            ite += 1
#            v_bef = self.volume_m3
#            self.run_until(self.yr + ystep)
#            v_af = self.volume_m3
#            if np.isclose(v_bef, 0., atol=1):
#                t_rate = 1
#                was_close_zero += 1
#            else:
#                t_rate = np.abs(v_af - v_bef) / v_bef
#        if ite > max_ite:
#            raise RuntimeError('Did not find equilibrium.')
#
#def flux_gate_with_build_up(year, flux_value=None, flux_gate_yr=None):
#    """Default scalar flux gate with build up period"""
#    fac = 1 - (flux_gate_yr - year) / flux_gate_yr
#    return flux_value * utils.clip_scalar(fac, 0, 1)
#
#class FluxBasedModel(FlowlineModel):
#    """The flowline model used by OGGM in production.
#    It solves for the SIA along the flowline(s) using a staggered grid. It
#    computes the *ice flux* between grid points and transports the mass
#    accordingly (also between flowlines).
#    This model is numerically less stable than fancier schemes, but it
#    is fast and works with multiple flowlines of any bed shape (rectangular,
#    parabolic, trapeze, and any combination of them).
#    We test that it conserves mass in most cases, but not on very stiff cliffs.
#    """
#
#    def __init__(self, flowlines, mb_model=None, y0=0., glen_a=None,
#                 fs=0., inplace=False, fixed_dt=None, cfl_number=None,
#                 min_dt=None, flux_gate_thickness=None,
#                 flux_gate=None, flux_gate_build_up=100,
#                 do_kcalving=None, calving_k=None, calving_use_limiter=None,
#                 calving_limiter_frac=None, water_level=None,
#                 **kwargs):
#        """Instanciate the model.
#        Parameters
#        ----------
#        flowlines : list
#            the glacier flowlines
#        mb_model : MassBakanceModel
#            the mass-balance model
#        y0 : int
#            initial year of the simulation
#        glen_a : float
#            Glen's creep parameter
#        fs : float
#            Oerlemans sliding parameter
#        inplace : bool
#            whether or not to make a copy of the flowline objects for the run
#            setting to True implies that your objects will be modified at run
#            time by the model (can help to spare memory)
#        fixed_dt : float
#            set to a value (in seconds) to prevent adaptive time-stepping.
#        cfl_number : float
#            Defaults to cfg.PARAMS['cfl_number'].
#            For adaptive time stepping (the default), dt is chosen from the
#            CFL criterion (dt = cfl_number * dx / max_u).
#            To choose the "best" CFL number we would need a stability
#            analysis - we used an empirical analysis (see blog post) and
#            settled on 0.02 for the default cfg.PARAMS['cfl_number'].
#        min_dt : float
#            Defaults to cfg.PARAMS['cfl_min_dt'].
#            At high velocities, time steps can become very small and your
#            model might run very slowly. In production, it might be useful to
#            set a limit below which the model will just error.
#        is_tidewater: bool, default: False
#            is this a tidewater glacier?
#        is_lake_terminating: bool, default: False
#            is this a lake terminating glacier?
#        mb_elev_feedback : str, default: 'annual'
#            'never', 'always', 'annual', or 'monthly': how often the
#            mass-balance should be recomputed from the mass balance model.
#            'Never' is equivalent to 'annual' but without elevation feedback
#            at all (the heights are taken from the first call).
#        check_for_boundaries: bool, default: True
#            raise an error when the glacier grows bigger than the domain
#            boundaries
#        flux_gate_thickness : float or array
#            flux of ice from the left domain boundary (and tributaries).
#            Units of m of ice thickness. Note that unrealistic values won't be
#            met by the model, so this is really just a rough guidance.
#            It's better to use `flux_gate` instead.
#        flux_gate : float or function or array of floats or array of functions
#            flux of ice from the left domain boundary (and tributaries)
#            (unit: m3 of ice per second). If set to a high value, consider
#            changing the flux_gate_buildup time. You can also provide
#            a function (or an array of functions) returning the flux
#            (unit: m3 of ice per second) as a function of time.
#            This is overriden by `flux_gate_thickness` if provided.
#        flux_gate_buildup : int
#            number of years used to build up the flux gate to full value
#        do_kcalving : bool
#            switch on the k-calving parameterisation. Ignored if not a
#            tidewater glacier. Use the option from PARAMS per default
#        calving_k : float
#            the calving proportionality constant (units: yr-1). Use the
#            one from PARAMS per default
#        calving_use_limiter : bool
#            whether to switch on the calving limiter on the parameterisation
#            makes the calving fronts thicker but the model is more stable
#        calving_limiter_frac : float
#            limit the front slope to a fraction of the calving front.
#            "3" means 1/3. Setting it to 0 limits the slope to sea-level.
#        water_level : float
#            the water level. It should be zero m a.s.l, but:
#            - sometimes the frontal elevation is unrealistically high (or low).
#            - lake terminating glaciers
#            - other uncertainties
#            The default is 0. For lake terminating glaciers,
#            it is inferred from PARAMS['free_board_lake_terminating'].
#            The best way to set the water level for real glaciers is to use
#            the same as used for the inversion (this is what
#            `robust_model_run` does for you)
#        """
#        super(FluxBasedModel, self).__init__(flowlines, mb_model=mb_model,
#                                             y0=y0, glen_a=glen_a, fs=fs,
#                                             inplace=inplace,
#                                             water_level=water_level,
#                                             **kwargs)
#
#        self.fixed_dt = fixed_dt
#        if min_dt is None:
#            min_dt = cfg.PARAMS['cfl_min_dt']
#        if cfl_number is None:
#            cfl_number = cfg.PARAMS['cfl_number']
#        self.min_dt = min_dt
#        self.cfl_number = cfl_number
#
#        # Do we want to use shape factors?
#        self.sf_func = None
#        use_sf = cfg.PARAMS.get('use_shape_factor_for_fluxbasedmodel')
#        if use_sf == 'Adhikari' or use_sf == 'Nye':
#            self.sf_func = utils.shape_factor_adhikari
#        elif use_sf == 'Huss':
#            self.sf_func = utils.shape_factor_huss
#
#        # Calving params
#        if do_kcalving is None:
#            do_kcalving = cfg.PARAMS['use_kcalving_for_run']
#        self.do_calving = do_kcalving and self.is_tidewater
#        if calving_k is None:
#            calving_k = cfg.PARAMS['calving_k']
#        self.calving_k = calving_k / cfg.SEC_IN_YEAR
#        if calving_use_limiter is None:
#            calving_use_limiter = cfg.PARAMS['calving_use_limiter']
#        self.calving_use_limiter = calving_use_limiter
#        if calving_limiter_frac is None:
#            calving_limiter_frac = cfg.PARAMS['calving_limiter_frac']
#        if calving_limiter_frac > 0:
#            raise NotImplementedError('calving limiter other than 0 not '
#                                      'implemented yet')
#        self.calving_limiter_frac = calving_limiter_frac
#
#        # Flux gate
#        self.flux_gate = utils.tolist(flux_gate, length=len(self.fls))
#        self.flux_gate_m3_since_y0 = 0.
#        if flux_gate_thickness is not None:
#            # Compute the theoretical ice flux from the slope at the top
#            flux_gate_thickness = utils.tolist(flux_gate_thickness,
#                                               length=len(self.fls))
#            self.flux_gate = []
#            for fl, fgt in zip(self.fls, flux_gate_thickness):
#                # We set the thickness to the desired value so that
#                # the widths work ok
#                fl = copy.deepcopy(fl)
#                fl.thick = fl.thick * 0 + fgt
#                slope = (fl.surface_h[0] - fl.surface_h[1]) / fl.dx_meter
#                if slope == 0:
#                    raise ValueError('I need a slope to compute the flux')
#                flux = find_sia_flux_from_thickness(slope,
#                                                    fl.widths_m[0],
#                                                    fgt,
#                                                    shape=fl.shape_str[0],
#                                                    glen_a=self.glen_a,
#                                                    fs=self.fs)
#                self.flux_gate.append(flux)
#
#        # convert the floats to function calls
#        for i, fg in enumerate(self.flux_gate):
#            if fg is None:
#                continue
#            try:
#                # Do we have a function? If yes all good
#                fg(self.yr)
#            except TypeError:
#                # If not, make one
#                self.flux_gate[i] = partial(flux_gate_with_build_up,
#                                            flux_value=fg,
#                                            flux_gate_yr=(flux_gate_build_up +
#                                                          self.y0))
#
#        # Optim
#        self.slope_stag = []
#        self.thick_stag = []
#        self.section_stag = []
#        self.u_stag = []
#        self.shapefac_stag = []
#        self.flux_stag = []
#        self.trib_flux = []
#        for fl, trib in zip(self.fls, self._tributary_indices):
#            nx = fl.nx
#            # This is not staggered
#            self.trib_flux.append(np.zeros(nx))
#            # We add an additional fake grid point at the end of tributaries
#            if trib[0] is not None:
#                nx = fl.nx + 1
#            # +1 is for the staggered grid
#            self.slope_stag.append(np.zeros(nx+1))
#            self.thick_stag.append(np.zeros(nx+1))
#            self.section_stag.append(np.zeros(nx+1))
#            self.u_stag.append(np.zeros(nx+1))
#            self.shapefac_stag.append(np.ones(nx+1))  # beware the ones!
#            self.flux_stag.append(np.zeros(nx+1))
#
#    def step(self, dt):
#        """Advance one step."""
#
#        # Just a check to avoid useless computations
#        if dt <= 0:
#            raise InvalidParamsError('dt needs to be strictly positive')
#
#        # Simple container
#        mbs = []
#
#        # Loop over tributaries to determine the flux rate
#        for fl_id, fl in enumerate(self.fls):
#
#            # This is possibly less efficient than zip() but much clearer
#            trib = self._tributary_indices[fl_id]
#            slope_stag = self.slope_stag[fl_id]
#            thick_stag = self.thick_stag[fl_id]
#            section_stag = self.section_stag[fl_id]
#            sf_stag = self.shapefac_stag[fl_id]
#            flux_stag = self.flux_stag[fl_id]
#            trib_flux = self.trib_flux[fl_id]
#            u_stag = self.u_stag[fl_id]
#            flux_gate = self.flux_gate[fl_id]
#
#            # Flowline state
#            surface_h = fl.surface_h
#            thick = fl.thick
#            section = fl.section
#            dx = fl.dx_meter
#
#            # If it is a tributary, we use the branch it flows into to compute
#            # the slope of the last grid point
#            is_trib = trib[0] is not None
#            if is_trib:
#                fl_to = self.fls[trib[0]]
#                ide = fl.flows_to_indice
#                surface_h = np.append(surface_h, fl_to.surface_h[ide])
#                thick = np.append(thick, thick[-1])
#                section = np.append(section, section[-1])
#            elif self.do_calving and self.calving_use_limiter:
#                # We lower the max possible ice deformation
#                # by clipping the surface slope here. It is completely
#                # arbitrary but reduces ice deformation at the calving front.
#                # I think that in essence, it is also partly
#                # a "calving process", because this ice deformation must
#                # be less at the calving front. The result is that calving
#                # front "free boards" are quite high.
#                # Note that 0 is arbitrary, it could be any value below SL
#                surface_h = utils.clip_min(surface_h, self.water_level)
#
#            # Staggered gradient
#            slope_stag[0] = 0
#            slope_stag[1:-1] = (surface_h[0:-1] - surface_h[1:]) / dx
#            slope_stag[-1] = slope_stag[-2]
#
#            # Staggered thick
#            thick_stag[1:-1] = (thick[0:-1] + thick[1:]) / 2.
#            thick_stag[[0, -1]] = thick[[0, -1]]
#
#            if self.sf_func is not None:
#                # TODO: maybe compute new shape factors only every year?
#                sf = self.sf_func(fl.widths_m, fl.thick, fl.is_rectangular)
#                if is_trib:
#                    # for inflowing tributary, the sf makes no sense
#                    sf = np.append(sf, 1.)
#                sf_stag[1:-1] = (sf[0:-1] + sf[1:]) / 2.
#                sf_stag[[0, -1]] = sf[[0, -1]]
#
#            # Staggered velocity (Deformation + Sliding)
#            # _fd = 2/(N+2) * self.glen_a
#            N = self.glen_n
#            rhogh = (self.rho*cfg.G*slope_stag)**N
#            u_stag[:] = (thick_stag**(N+1)) * self._fd * rhogh * sf_stag**N + \
#                        (thick_stag**(N-1)) * self.fs * rhogh
#
#            # Staggered section
#            section_stag[1:-1] = (section[0:-1] + section[1:]) / 2.
#            section_stag[[0, -1]] = section[[0, -1]]
#
#            # Staggered flux rate
#            flux_stag[:] = u_stag * section_stag
#
#            # Add boundary condition
#            if flux_gate is not None:
#                flux_stag[0] = flux_gate(self.yr)
#
#            # CFL condition
#            if not self.fixed_dt:
#                maxu = np.max(np.abs(u_stag))
#                if maxu > cfg.FLOAT_EPS:
#                    cfl_dt = self.cfl_number * dx / maxu
#                else:
#                    cfl_dt = dt
#
#                # Update dt only if necessary
#                if cfl_dt < dt:
#                    dt = cfl_dt
#                    if cfl_dt < self.min_dt:
#                        raise RuntimeError(
#                            'CFL error: required time step smaller '
#                            'than the minimum allowed: '
#                            '{:.1f}s vs {:.1f}s.'.format(cfl_dt, self.min_dt))
#
#            # Since we are in this loop, reset the tributary flux
#            trib_flux[:] = 0
#
#            # We compute MB in this loop, before mass-redistribution occurs,
#            # so that MB models which rely on glacier geometry to decide things
#            # (like PyGEM) can do wo with a clean glacier state
#            mbs.append(self.get_mb(fl.surface_h, self.yr,
#                                   fl_id=fl_id, fls=self.fls))
#
#        # Time step
#        if self.fixed_dt:
#            # change only if step dt is larger than the chosen dt
#            if self.fixed_dt < dt:
#                dt = self.fixed_dt
#
#        # A second loop for the mass exchange
#        for fl_id, fl in enumerate(self.fls):
#            
#            flx_stag = self.flux_stag[fl_id]
#            trib_flux = self.trib_flux[fl_id]
#            tr = self._tributary_indices[fl_id]
#
#            dx = fl.dx_meter
#
#            is_trib = tr[0] is not None
#
#            # For these we had an additional grid point
#            if is_trib:
#                flx_stag = flx_stag[:-1]
#
#            # Mass-balance
#            widths = fl.widths_m
#            mb = mbs[fl_id]
#            
#            # Allow parabolic beds to grow
#            mb = dt * mb * np.where((mb > 0.) & (widths == 0), 10., widths)
#            
##            print('mass balance (m ice for time step):', mb)
#            
#            ice_thickness = np.where(widths > 0, fl.section / widths, 0)
#            
#            ice_thickness_plus_mb = ice_thickness + mb
#            
#            ice_thickness_missing = np.where(ice_thickness_plus_mb > 0, 0, ice_thickness_plus_mb)
#            
##            print('missing ice thickness w/o flux (m3):', (ice_thickness_missing * widths * dx).sum())
#
#
#            # Update section with ice flow and mass balance
#            new_section = (fl.section + (flx_stag[0:-1] - flx_stag[1:])*dt/dx +
#                           trib_flux*dt/dx + mb)
#            
#            volume_change_unaccounted = np.where(new_section > 0, 0, new_section * dx)
##            print(volume_change_unaccounted)
##            print('surface h:', surface_h)
##            print('slope:', ((surface_h[1:] - surface_h[:-1]) / dx))
##            print('dx:', dx)
##            print('volume change unaccounted:', volume_change_unaccounted.sum())
#
#            # Keep positive values only and store
#            
#            old_section = np.copy(fl.section)
#            
#            fl.section = utils.clip_min(new_section, 0)
#            
##            old_volume = (old_section * dx).sum()
##            updated_volume = (fl.section * dx).sum()
##            print('\nvolume prior  (section * dx):', old_volume)
##            print('  volume updated            :', updated_volume)
##            volume_change = updated_volume - old_volume
##            print('  volume change before/after:', updated_volume - old_volume)
##            
##            vol_change_from_mb = (mb * widths * dx).sum()
##            print('  volume change from mb     :', vol_change_from_mb)
##            
##            mb_4timestep = volume_change / (widths * dx).sum() / dt * 365 * 24 * 3600 * 0.9
##            print('  mb (mwea) for time step   :', mb_4timestep)
#            
#            
#            
#            
##            vol_change = (mb * widths * dx).sum()
##            print('volume change from mb (m3):', vol_change)
##            print('difference volume change:', vol_change - (updated_volume - old_volume))
#            
#
#            # If we use a flux-gate, store the total volume that came in
#            self.flux_gate_m3_since_y0 += flx_stag[0] * dt
#
#            # Add the last flux to the tributary
#            # this works because the lines are sorted in order
#            if is_trib:
#                # tr tuple: line_index, start, stop, gaussian_kernel
#                self.trib_flux[tr[0]][tr[1]:tr[2]] += \
#                    utils.clip_min(flx_stag[-1], 0) * tr[3]
#
#            # --- The rest is for calving only ---
#            self.calving_rate_myr = 0.
#
#            # If tributary, do calving only if we are not transferring mass
#            if is_trib and flx_stag[-1] > 0:
#                continue
#
#            # No need to do calving in these cases either
#            if not self.do_calving or not fl.has_ice():
#                continue
#
#            # We do calving only if the last glacier bed pixel is below water
#            # (this is to avoid calving elsewhere than at the front)
#            if fl.bed_h[fl.thick > 0][-1] > self.water_level:
#                continue
#
#            # We do calving only if there is some ice above wl
#            last_above_wl = np.nonzero((fl.surface_h > self.water_level) &
#                                       (fl.thick > 0))[0][-1]
#            if fl.bed_h[last_above_wl] > self.water_level:
#                continue
#
#            # OK, we're really calving
#            section = fl.section
#
#            # Calving law
#            h = fl.thick[last_above_wl]
#            d = h - (fl.surface_h[last_above_wl] - self.water_level)
#            k = self.calving_k
#            q_calving = k * d * h * fl.widths_m[last_above_wl]
#            # Add to the bucket and the diagnostics
#            fl.calving_bucket_m3 += q_calving * dt
#            self.calving_m3_since_y0 += q_calving * dt
#            self.calving_rate_myr = (q_calving / section[last_above_wl] *
#                                     cfg.SEC_IN_YEAR)
#
#            # See if we have ice below sea-water to clean out first
#            below_sl = (fl.surface_h < self.water_level) & (fl.thick > 0)
#            to_remove = np.sum(section[below_sl]) * fl.dx_meter
#            if 0 < to_remove < fl.calving_bucket_m3:
#                # This is easy, we remove everything
#                section[below_sl] = 0
#                fl.calving_bucket_m3 -= to_remove
#            elif to_remove > 0:
#                # We can only remove part of if
#                section[below_sl] = 0
#                section[last_above_wl+1] = ((to_remove - fl.calving_bucket_m3)
#                                            / fl.dx_meter)
#                fl.calving_bucket_m3 = 0
#
#            # The rest of the bucket might calve an entire grid point
#            vol_last = section[last_above_wl] * fl.dx_meter
#            if fl.calving_bucket_m3 > vol_last:
#                fl.calving_bucket_m3 -= vol_last
#                section[last_above_wl] = 0
#
#            # We update the glacier with our changes
#            fl.section = section
#
#        # Next step
#        self.t += dt
#        return dt
#
#    def get_diagnostics(self, fl_id=-1):
#        """Obtain model diagnostics in a pandas DataFrame.
#        Parameters
#        ----------
#        fl_id : int
#            the index of the flowline of interest, from 0 to n_flowline-1.
#            Default is to take the last (main) one
#        Returns
#        -------
#        a pandas DataFrame, which index is distance along flowline (m). Units:
#            - surface_h, bed_h, ice_tick, section_width: m
#            - section_area: m2
#            - slope: -
#            - ice_flux, tributary_flux: m3 of *ice* per second
#            - ice_velocity: m per second (depth-section integrated)
#        """
#        import pandas as pd
#
#        fl = self.fls[fl_id]
#        nx = fl.nx
#
#        df = pd.DataFrame(index=fl.dx_meter * np.arange(nx))
#        df.index.name = 'distance_along_flowline'
#        df['surface_h'] = fl.surface_h
#        df['bed_h'] = fl.bed_h
#        df['ice_thick'] = fl.thick
#        df['section_width'] = fl.widths_m
#        df['section_area'] = fl.section
#
#        # Staggered
#        var = self.slope_stag[fl_id]
#        df['slope'] = (var[1:nx+1] + var[:nx])/2
#        var = self.flux_stag[fl_id]
#        df['ice_flux'] = (var[1:nx+1] + var[:nx])/2
#        var = self.u_stag[fl_id]
#        df['ice_velocity'] = (var[1:nx+1] + var[:nx])/2
#        var = self.shapefac_stag[fl_id]
#        df['shape_fac'] = (var[1:nx+1] + var[:nx])/2
#
#        # Not Staggered
#        df['tributary_flux'] = self.trib_flux[fl_id]
#
#        return df