#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Derive timing of terminus retreat (assuming clean ice retreat - Mass Redistribution Curves)

Created on Mon Apr 12 09:09:12 2021

@author: drounce
"""

# Built-in libraries
import os
import pickle
# External libraries
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd
import xarray as xr
# Local libraries
import pygem.pygem_input as pygem_prms
import pygem.pygem_modelsetup as modelsetup


#%% =====  Inputs =====
debug=False
csv_fp = '/Users/drounce/Documents/HiMAT/HMA_output/csv/'
fig_fp = '/Users/drounce/Documents/HiMAT/HMA_output/fig/'
if not os.path.exists(csv_fp):
    os.makedirs(csv_fp)
if not os.path.exists(fig_fp):
    os.makedirs(fig_fp)

# Load RGIIds
rgiids_fullfn = '/Users/drounce/Documents/HiMAT/HMA_output/RGIUK_DavidRounce_rgiids.csv'
rgiids_df = pd.read_csv(rgiids_fullfn)
rgiids_list = [x.split('-')[1] for x in rgiids_df['RGIId'].values]

#rgiids_list = ['15.03733']
#rgiids_list = ['14.12145'] # Example of glacier that gets down to 3 bins

rcps = ['rcp26', 'rcp45', 'rcp60', 'rcp85']
#rcps = ['rcp85']

# Dictionary of hypsometry filenames
hyps_filepath = pygem_prms.main_directory + '/../IceThickness_Huss/bands_10m_DRR/'
hyps_filedict = {
                13: 'area_13_Huss_CentralAsia_10m.csv',
                14: 'area_14_Huss_SouthAsiaWest_10m.csv',
                15: 'area_15_Huss_SouthAsiaEast_10m.csv'}
hyps_colsdrop = ['RGI-ID','Cont_range']
# Thickness data
thickness_filepath = pygem_prms.main_directory + '/../IceThickness_Huss/bands_10m_DRR/'
thickness_filedict = {
                13: 'thickness_13_Huss_CentralAsia_10m.csv',
                14: 'thickness_14_Huss_SouthAsiaWest_10m.csv',
                15: 'thickness_15_Huss_SouthAsiaEast_10m.csv'}
thickness_colsdrop = ['RGI-ID','Cont_range']
# Width data
width_filepath = pygem_prms.main_directory + '/../IceThickness_Huss/bands_10m_DRR/'
width_filedict = {
                13: 'width_13_Huss_CentralAsia_10m.csv',
                14: 'width_14_Huss_SouthAsiaWest_10m.csv',
                15: 'width_15_Huss_SouthAsiaEast_10m.csv'}
width_colsdrop = ['RGI-ID','Cont_range']

option_glaciershape=1
icethickness_advancethreshold=5
terminus_percentage=20


#%% ===== FUNCTIONS =====
def massredistributionHuss(glacier_area_t0, icethickness_t0, width_t0, glacier_volumechange, year, 
                           glac_idx_initial, glac_area_initial, debug=False):
    """
    Mass redistribution according to empirical equations from Huss and Hock (2015) accounting for retreat/advance.
    glac_idx_initial is required to ensure that the glacier does not advance to area where glacier did not exist before
    (e.g., retreat and advance over a vertical cliff)
    
    Code is based on Release 0.1.0
    
    Parameters
    ----------
    glacier_area_t0 : np.ndarray
        Glacier area [km2] from previous year for each elevation bin
    icethickness_t0 : np.ndarray
        Ice thickness [m] from previous year for each elevation bin
    width_t0 : np.ndarray
        Glacier width [km] from previous year for each elevation bin
    glacier_volumechange : np.float
        Glacier volume change [km3 ice]
    year : int
        Count of the year of model run (first year is 0)
    glac_idx_initial : np.ndarray
        Initial glacier indices
    glac_area_initial : np.ndarray
        Initial glacier array used to determine average terminus area in event that glacier is only one bin
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
        icethickness_t1, glacier_area_t1, width_t1, icethickness_change, glacier_volumechange_remaining = (
                massredistributioncurveHuss(icethickness_t0, glacier_area_t0, width_t0, glac_idx_t0, glacier_volumechange))
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
            icethickness_t1, glacier_area_t1, width_t1, icethickness_change, glacier_volumechange_remaining = (
                    massredistributioncurveHuss(icethickness_t0_retreated, glacier_area_t0_retreated, 
                                                width_t0_retreated, glac_idx_t0_retreated, 
                                                glacier_volumechange_remaining_retreated))                   
        # Glacier advances
        #  if glacier advances (ice thickness change exceeds threshold), then redistribute mass gain in new bins
        while (icethickness_change > icethickness_advancethreshold).any() == True:  
            # Record glacier area and ice thickness before advance corrections applied
            glacier_area_t1_raw = glacier_area_t1.copy()
            icethickness_t1_raw = icethickness_t1.copy()
            width_t1_raw = width_t1.copy()
            # Index bins that are advancing
            icethickness_change[icethickness_change <= icethickness_advancethreshold] = 0
            glac_idx_advance = icethickness_change.nonzero()[0]
            # Update ice thickness based on maximum advance threshold [m ice]
            icethickness_t1[glac_idx_advance] = (icethickness_t1[glac_idx_advance] - 
                           (icethickness_change[glac_idx_advance] - icethickness_advancethreshold))
            # Update glacier area based on reduced ice thicknesses [km**2]
            if option_glaciershape == 1:
                # Glacier area for parabola [km**2] (A_1 = A_0 * (H_1 / H_0)**0.5)
                glacier_area_t1[glac_idx_advance] = (glacier_area_t1_raw[glac_idx_advance] * 
                               (icethickness_t1[glac_idx_advance] / icethickness_t1_raw[glac_idx_advance])**0.5)
                # Glacier width for parabola [km] (w_1 = w_0 * A_1 / A_0)
                width_t1[glac_idx_advance] = (width_t1_raw[glac_idx_advance] * glacier_area_t1[glac_idx_advance] 
                                              / glacier_area_t1_raw[glac_idx_advance])
            elif option_glaciershape == 2:
                # Glacier area constant for rectangle [km**2] (A_1 = A_0)
                glacier_area_t1[glac_idx_advance] = glacier_area_t1_raw[glac_idx_advance]
                # Glacier with constant for rectangle [km] (w_1 = w_0)
                width_t1[glac_idx_advance] = width_t1_raw[glac_idx_advance]
            elif option_glaciershape == 3:
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
            glac_idx_terminus = (glac_idx_t0[(glac_idx_t0 - glac_idx_t0[0] + 1) / 
                                             glac_idx_t0.shape[0] * 100 < terminus_percentage])
            if debug:
                print('glacier index terminus:',glac_idx_terminus)
                print('glacier index:',glac_idx_t0)
                print('glacier indx initial:', glac_idx_initial)
            # For glaciers with so few bands that the terminus is not identified (ex. <= 4 bands for 20% threshold),
            #  then use the information from all the bands
            if glac_idx_terminus.shape[0] <= 1:
                glac_idx_terminus = glac_idx_t0.copy()
            # Average area of glacier terminus [km**2]
            #  exclude the bin at the terminus, since this bin may need to be filled first
            try:
                terminus_area_avg = (
                        glacier_area_t0[glac_idx_terminus[1]:glac_idx_terminus[glac_idx_terminus.shape[0]-1]+1].mean())
            except:  
                glac_idx_terminus_initial = (
                        glac_idx_initial[(glac_idx_initial - glac_idx_initial[0] + 1) / glac_idx_initial.shape[0] * 100 
                                          < terminus_percentage])
                if glac_idx_terminus_initial.shape[0] <= 1:
                    glac_idx_terminus_initial = glac_idx_initial.copy()
                terminus_area_avg = (
                        glac_area_initial[glac_idx_terminus_initial[1]:
                                          glac_idx_terminus_initial[glac_idx_terminus_initial.shape[0]-1]+1].mean())
            # Check if the last bin's area is below the terminus' average and fill it up if it is
            if (glacier_area_t1[glac_idx_terminus[0]] < terminus_area_avg) and (icethickness_t0[glac_idx_terminus[0]] <
               icethickness_t0[glac_idx_t0].mean()):
#            if glacier_area_t1[glac_idx_terminus[0]] < terminus_area_avg:
                # Volume required to fill the bin at the terminus
                advance_volume_fillbin = (icethickness_t1[glac_idx_terminus[0]] / 1000 * (terminus_area_avg - 
                                          glacier_area_t1[glac_idx_terminus[0]]))
                # If the advance volume is less than that required to fill the bin, then fill the bin as much as
                #  possible by adding area (thickness remains the same - glacier front is only thing advancing)
                if advance_volume < advance_volume_fillbin:
                    # add advance volume to the bin (area increases, thickness and width constant)
                    glacier_area_t1[glac_idx_terminus[0]] = (glacier_area_t1[glac_idx_terminus[0]] + 
                                   advance_volume / (icethickness_t1[glac_idx_terminus[0]] / 1000))
                    # set advance volume equal to zero
                    advance_volume = 0
                else:
                    # fill the bin (area increases, thickness and width constant)
                    glacier_area_t1[glac_idx_terminus[0]] = (glacier_area_t1[glac_idx_terminus[0]] + 
                                   advance_volume_fillbin / (icethickness_t1[glac_idx_terminus[0]] / 1000))
                    advance_volume = advance_volume - advance_volume_fillbin
            # With remaining advance volume, add a bin
            if advance_volume > 0:
                # Index for additional bin below the terminus
                glac_idx_bin2add = np.array([glac_idx_terminus[0] - 1])
                # Check if bin2add is in a discontinuous section of the initial glacier
                while ((glac_idx_bin2add > glac_idx_initial.min()) & 
                       ((glac_idx_bin2add == glac_idx_initial).any() == False)):
                    # Advance should not occur in a discontinuous section of the glacier (e.g., vertical drop),
                    #  so change the bin2add to the next bin down valley
                    glac_idx_bin2add = glac_idx_bin2add - 1
                # if the added bin would be below sea-level, then volume is distributed over the glacier without
                #  any adjustments
                if glac_idx_bin2add < 0:
                    glacier_area_t1 = glacier_area_t1_raw
                    icethickness_t1 = icethickness_t1_raw
                    width_t1 = width_t1_raw
                    advance_volume = 0
                # otherwise, add a bin with thickness and width equal to the previous bin and fill it up
                else:
                    # ice thickness of new bin equals ice thickness of bin at the terminus
                    icethickness_t1[glac_idx_bin2add] = icethickness_t1[glac_idx_terminus[0]]
                    width_t1[glac_idx_bin2add] = width_t1[glac_idx_terminus[0]]
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
            # Record glacier area and ice thickness before advance corrections applied
            glacier_area_t1_raw = glacier_area_t1.copy()
            icethickness_t1_raw = icethickness_t1.copy()
            width_t1_raw = width_t1.copy()
            # If a full bin has been added and volume still remains, then redistribute mass across the
            #  glacier, thereby enabling the bins to get thicker once again prior to adding a new bin.
            #  This is important for glaciers that have very thin ice at the terminus as this prevents the 
            #  glacier from having a thin layer of ice advance tremendously far down valley without thickening.
            if advance_volume > 0:
                icethickness_t1, glacier_area_t1, width_t1, icethickness_change, glacier_volumechange_remaining = (
                        massredistributioncurveHuss(icethickness_t1, glacier_area_t1, width_t1, glac_idx_t0, 
                                                    advance_volume))
            # update ice thickness change
            icethickness_change = icethickness_t1 - icethickness_t1_raw
    return glacier_area_t1, icethickness_t1, width_t1


def massredistributioncurveHuss(icethickness_t0, glacier_area_t0, width_t0, glac_idx_t0, glacier_volumechange):
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
        elevrange_norm[glacier_area_t0 > 0] = (glac_idx_t0[-1] - glac_idx_t0) / (glac_idx_t0[-1] - glac_idx_t0[0])
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
        fs_huss = glacier_volumechange / (glacier_area_t0 * icethicknesschange_norm).sum() * 1000
        #  units: km**3 / (km**2 * [-]) * (1000 m / 1 km) = m ice
        # Volume change [km**3 ice]
        bin_volumechange = icethicknesschange_norm * fs_huss / 1000 * glacier_area_t0
    # Otherwise, compute volume change in each bin based on the climatic mass balance
    else:
        # Calculate climatic mass balance to be equal for all bins
        #  (modified from original code to work with outputs that we have)
        mb_clim = (glacier_volumechange / glacier_area_t0.sum() * 1000)
        bin_volumechange = mb_clim / 1000 * glacier_area_t0       
    if option_glaciershape == 1:
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
    elif option_glaciershape == 2:
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
    elif option_glaciershape == 3:
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
    # return desired output
    return icethickness_t1, glacier_area_t1, width_t1, icethickness_change, glacier_volumechange_remaining


#%%
# Load glaciers
main_glac_rgi = modelsetup.selectglaciersrgitable(glac_no=rgiids_list)

# Glacier hypsometry [km**2], total area
main_glac_hyps = modelsetup.import_Husstable(main_glac_rgi, hyps_filepath, hyps_filedict, hyps_colsdrop)
# Ice thickness [m], average
main_glac_icethickness = modelsetup.import_Husstable(main_glac_rgi, thickness_filepath, thickness_filedict, thickness_colsdrop)
main_glac_hyps[main_glac_icethickness == 0] = 0
# Width [km], average
main_glac_width = modelsetup.import_Husstable(main_glac_rgi, width_filepath, width_filedict, width_colsdrop)
elev_bins = main_glac_hyps.columns.values.astype(int)
# Volume [km**3] and mean elevation [m a.s.l.]
main_glac_rgi['Volume'], main_glac_rgi['Zmean'] = modelsetup.hypsometrystats(main_glac_hyps, main_glac_icethickness)


for rcp in rcps:
    
    multimodel_fp = '/Users/drounce/Documents/HiMAT/HMA_output/multimodel/'
    multimodel_fn_14 = 'R14_multigcm_' + rcp + '_c2_ba1_100sets_2000_2100.nc'
    multimodel_fn_15 = 'R15_multigcm_' + rcp + '_c2_ba1_100sets_2000_2100.nc'
    ds_14 = xr.open_dataset(multimodel_fp + multimodel_fn_14)
    ds_15 = xr.open_dataset(multimodel_fp + multimodel_fn_15) 

    for glac in range(main_glac_rgi.shape[0]):
        print(rcp, glac, main_glac_rgi.loc[main_glac_rgi.index.values[glac],'RGIId'])
        # Select subsets of data
        glacier_rgi_table = main_glac_rgi.loc[main_glac_rgi.index.values[glac], :]
        glacier_area_t0 = main_glac_hyps.iloc[glac,:].values.astype(float)
        icethickness_t0 = main_glac_icethickness.iloc[glac,:].values.astype(float)
        width_t0 = main_glac_width.iloc[glac,:].values.astype(float)
        glacier_str = '{0:0.5f}'.format(glacier_rgi_table['RGIId_float'])
        
        glac_idx_initial = glacier_area_t0.nonzero()[0]
        glac_area_initial = glacier_area_t0.copy()
        icethickness_initial = icethickness_t0.copy()
        width_initial = width_t0.copy()
        length_initial = np.zeros(glac_area_initial.shape)
        length_initial[glac_idx_initial] = glac_area_initial[glac_idx_initial] / width_initial[glac_idx_initial]
        
        nyears = len(ds_14.year.values)
        
        # Glacier volume change
        if glacier_rgi_table.O1Region == 14:
            ds_idx = np.where(glacier_rgi_table.RGIId == ds_14.RGIId.values)[0][0]
            glac_volume_annual = ds_14.glac_volume_annual.values[ds_idx,:]
            glac_volchg_annual = glac_volume_annual[1:] - glac_volume_annual[:-1]
        elif glacier_rgi_table.O1Region == 15:
            ds_idx = np.where(glacier_rgi_table.RGIId == ds_15.RGIId.values)[0][0]
            glac_volume_annual = ds_15.glac_volume_annual.values[ds_idx,:]
            glac_volchg_annual = glac_volume_annual[1:] - glac_volume_annual[:-1]
        else:
            assert True==False, 'Need to load dataset that has glacier volume change for region:' + str(glacier_rgi_table.O1Region)
        
        
        #%%
        nbins = elev_bins.shape[0]
        glac_bin_area_annual = np.zeros((nbins, nyears + 1))
        glac_bin_icethickness_annual = np.zeros((nbins, nyears + 1))
        glac_bin_width_annual = np.zeros((nbins, nyears + 1))
        glac_bin_zsurf_annual = np.zeros((nbins, nyears + 1))
        for year in range(0, nyears+1):
            glac_bin_zsurf_annual[glac_idx_initial,year] = (elev_bins - icethickness_initial)[glac_idx_initial]
        
        
        for year in range(0, nyears):
            
            # Check ice still exists:
            if icethickness_t0.max() > 0:    
                
                # Glacier indices
                glac_idx_t0 = glacier_area_t0.nonzero()[0]
                
                # Terminus height
                min_elev_bin = elev_bins[glac_idx_t0[0]]
                # account surface lowering, i.e., not just that a glacier is present in the initial bin
                # Option to adjust air temperature based on changes in surface elevation
                min_elev = min_elev_bin + (icethickness_t0 - icethickness_initial)[glac_idx_t0[0]]
                
                if debug:
                    print('year', year, 'max ice thickness [m]:', np.round(icethickness_t0[glac_idx_t0[0]],1), 'min_elev:', int(min_elev_bin), int(min_elev))
                
                # Glacier volume change
                glacier_volumechange = glac_volchg_annual[year]

                # MASS REDISTRIBUTION
                if glacier_area_t0.max() > 0:
                    # Mass redistribution according to Huss empirical curves
                    glacier_area_t1, icethickness_t1, width_t1 = (
                            massredistributionHuss(glacier_area_t0, icethickness_t0, width_t0, 
                                                   glacier_volumechange, year, glac_idx_initial, 
                                                   glac_area_initial))
                    
                # Record glacier properties (area [km**2], thickness [m], width [km])
                # if first year, record initial glacier properties (area [km**2], ice thickness [m ice], width [km])
                if year == 0:
                    glac_bin_area_annual[:,year] = glacier_area_t0
                    glac_bin_icethickness_annual[:,year] = icethickness_t0
                    glac_bin_width_annual[:,year] = width_t0
                    glac_bin_zsurf_annual[glac_idx_t0,year] = elev_bins[glac_idx_t0]
                # record the next year's properties as well
                # 'year + 1' used so the glacier properties are consistent with mass balance computations
                glac_bin_icethickness_annual[:,year + 1] = icethickness_t1
                glac_bin_area_annual[:,year + 1] = glacier_area_t1
                glac_bin_width_annual[:,year + 1] = width_t1
                glac_bin_zsurf_annual[glac_idx_t0,year+1] = (elev_bins + (icethickness_t1 - icethickness_initial))[glac_idx_t0]
                # Update glacier properties for the mass balance computations
                icethickness_t0 = icethickness_t1.copy()
                glacier_area_t0 = glacier_area_t1.copy()
                width_t0 = width_t1.copy()
        
        # Overdeepening
        glac_bin_overdeepening_annual = glac_bin_zsurf_annual - glac_bin_zsurf_annual[:,0][:,np.newaxis]
        
        # Export results
        # Overdeepening
        csv_fp_overdeepening = csv_fp + 'overdeepening/' + rcp + '/'
        if not os.path.exists(csv_fp_overdeepening):
            os.makedirs(csv_fp_overdeepening)
        df_overdeepening_annual = pd.DataFrame(glac_bin_overdeepening_annual, columns=ds_14.year_plus1.values, index=elev_bins)
        df_overdeepening_annual.to_csv(csv_fp_overdeepening + glacier_str + '_multigcm_' + rcp + '_bin_overdeepening_annual.csv')
        
        # Surface elevation (including ice lowering)
        csv_fp_zsurf = csv_fp + 'zsurface/' + rcp + '/'
        if not os.path.exists(csv_fp_zsurf):
            os.makedirs(csv_fp_zsurf)
        df_zsurf_annual = pd.DataFrame(glac_bin_zsurf_annual, columns=ds_14.year_plus1.values, index=elev_bins)
        df_zsurf_annual.to_csv(csv_fp_zsurf + glacier_str + '_multigcm_' + rcp + '_bin_zsurface_annual.csv')
        
        
        #%%
        # ----- FIGURE: ICE THICKNESS PROFILES ----- 
        fig, ax = plt.subplots(1, 1, squeeze=False, sharex=False, sharey=False, 
                               gridspec_kw = {'wspace':0.7, 'hspace':0.5})

        # VOLUME CHANGE
        length_initial_cumsum = length_initial[::-1].cumsum()[::-1]
        
#        ax[0,0].plot(length_initial_cumsum[glac_idx_initial], 
#                     glac_bin_zsurf_annual[glac_idx_initial,0], 
#                     color='#1b9e77', linewidth=0.75, zorder=2, label='2000')
#        ax[0,0].plot(length_initial_cumsum[glac_idx_initial], 
#                     glac_bin_zsurf_annual[glac_idx_initial,50], 
#                     color='#d95f02', linewidth=0.75, zorder=3, label='2050')
#        ax[0,0].plot(length_initial_cumsum[glac_idx_initial],
#                     glac_bin_zsurf_annual[glac_idx_initial,50], 
#                     glac_bin_zsurf_annual[glac_idx_initial,-1], 
#                     color='#7570b3', linewidth=0.75, zorder=4, label='2100')
        ax[0,0].fill_between(length_initial_cumsum[glac_idx_initial], 
                             glac_bin_zsurf_annual[glac_idx_initial,0], 
                             glac_bin_zsurf_annual[glac_idx_initial,50],
                             color='#3182bd', linewidth=0.75, zorder=4, label='2000')
        ax[0,0].fill_between(length_initial_cumsum[glac_idx_initial], 
                             glac_bin_zsurf_annual[glac_idx_initial,50],
                             glac_bin_zsurf_annual[glac_idx_initial,-1],
                             color='#9ecae1', linewidth=0.75, zorder=4, label='2050')
        ax[0,0].fill_between(length_initial_cumsum[glac_idx_initial], 
                             glac_bin_zsurf_annual[glac_idx_initial,0] - icethickness_initial[glac_idx_initial],
                             glac_bin_zsurf_annual[glac_idx_initial,-1],
                             color='#deebf7', linewidth=0.75, zorder=4, label='2100')
        ax[0,0].plot(length_initial_cumsum[glac_idx_initial], 
                     glac_bin_zsurf_annual[glac_idx_initial,0] - icethickness_initial[glac_idx_initial], 
                     color='k', linewidth=1, zorder=5, label='bed')
        ax[0,0].set_ylabel('Elevation (m a.s.l.)')
        ax[0,0].set_xlabel('Length (km)')
        ax[0,0].set_xlim(0, length_initial_cumsum[glac_idx_initial].max())
        ax[0,0].tick_params(direction='inout', right=True)
        ax[0,0].legend(fontsize=12, labelspacing=0.5, handlelength=2, handletextpad=0.5, borderpad=0, frameon=False)        
        
        # Title
        fig.text(0.5, 0.95, glacier_str + ' (multi-GCM ' + rcp + ')', size=12, ha='center', va='top',)
        
        # Save figure
        fig_fp_rcp = fig_fp + rcp + '/'
        if not os.path.exists(fig_fp_rcp):
            os.makedirs(fig_fp_rcp)
        fig_fn = glacier_str + '_multigcm_' + rcp + '_icethickness_profile.png'
        fig.set_size_inches(8,6)
        fig.savefig(fig_fp_rcp + fig_fn, bbox_inches='tight', dpi=300)