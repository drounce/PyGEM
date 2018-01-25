#%% ###################################################################################################################
"""
Python Glacier Evolution Model "PyGEM" V1.0
Prepared by David Rounce with support from Regine Hock.
This work was funded under the NASA-ROSES program (grant no. NNX17AB27G).

PyGEM is an open source glacier evolution model written in python.  The model expands upon previous models from 
Radic et al. (2013), Bliss et al. (2014), and Huss and Hock (2015).
"""
#######################################################################################################################
# This is the main script that provides the architecture and framework for all of the model runs. All input data is 
# included in a separate module called pygem_input.py. It is recommended to not make any changes to this file unless
# you are a PyGEM developer and making changes to the model architecture.
#
#%%========= IMPORT PACKAGES ==========================================================================================
# Various packages are used to provide the proper architecture and framework for the calculations used in this script. 
# Some packages (e.g., datetime) are included in order to speed up calculations and simplify code.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os # os is used with re to find name matches
import re # see os
import xarray as xr
import netCDF4 as nc
from time import strftime
import timeit

#========== IMPORT INPUT AND FUNCTIONS FROM MODULES ===================================================================
import pygem_input as input
import pygemfxns_modelsetup as modelsetup
import pygemfxns_climate as climate
import pygemfxns_massbalance as massbalance
import pygemfxns_output as output

#%%======== DEVELOPER'S TO-DO LIST ====================================================================================
# > Output log file, i.e., file that states input parameters, date of model run, model options selected, 
#   and any errors that may have come up (e.g., precipitation corrected because negative value, etc.)

# ===== STEP ONE: Select glaciers included in model run ===============================================================
timestart_step1 = timeit.default_timer()
if input.option_glacier_selection == 1:
    # RGI glacier attributes
    main_glac_rgi = modelsetup.selectglaciersrgitable()
elif input.option_glacier_selection == 2:
    print('\n\tMODEL ERROR (selectglaciersrgi): this option to use shapefiles to select glaciers has not been coded '
          '\n\tyet. Please choose an option that exists. Exiting model run.\n')
    exit()
else:
    # Should add options to make regions consistent with Brun et al. (2017), which used ASTER DEMs to get mass 
    # balance of 92% of the HMA glaciers.
    print('\n\tModel Error (selectglaciersrgi): please choose an option that exists for selecting glaciers.'
          '\n\tExiting model run.\n')
    exit()
timeelapsed_step1 = timeit.default_timer() - timestart_step1
print('Step 1 time:', timeelapsed_step1, "s\n")

#%%=== STEP TWO: HYPSOMETRY, ICE THICKNESS, MODEL TIME FRAME, SURFACE TYPE ============================================
timestart_step2 = timeit.default_timer()
# Glacier hypsometry [km**2]
main_glac_hyps = modelsetup.import_hypsometry(main_glac_rgi)
# Ice thickness [m]
main_glac_icethickness = modelsetup.import_icethickness(main_glac_rgi)
# Add volume [km**3] and mean elevation [m a.s.l.] to the main glaciers table
main_glac_rgi['Volume'], main_glac_rgi['Zmean'] = modelsetup.hypsometrystats(main_glac_hyps, main_glac_icethickness)
# Model time frame
dates_table, start_date, end_date, monthly_columns, annual_columns, annual_divisor = modelsetup.datesmodelrun()
# Initial surface type
main_glac_surftypeinit = modelsetup.surfacetypeglacinitial(main_glac_rgi, main_glac_hyps)
# Print time elapsed
timeelapsed_step2 = timeit.default_timer() - timestart_step2
print('Step 2 time:', timeelapsed_step2, "s\n")

#%%=== STEP THREE: IMPORT CLIMATE DATA ================================================================================
timestart_step3 = timeit.default_timer()
if input.option_gcm_downscale == 1:
    # Air Temperature [degC] and GCM dates
    gcm_glac_temp, gcm_time_series = climate.importGCMvarnearestneighbor_xarray(
            input.gcm_temp_filename, input.gcm_temp_varname, main_glac_rgi, dates_table, start_date, end_date)
    # Air Temperature [degC] and GCM dates
    gcm_glac_prec, gcm_time_series = climate.importGCMvarnearestneighbor_xarray(
            input.gcm_prec_filename, input.gcm_prec_varname, main_glac_rgi, dates_table, start_date, end_date)
    # Elevation [m a.s.l] associated with air temperature data
    gcm_glac_elev = climate.importGCMfxnearestneighbor_xarray(
            input.gcm_elev_filename, input.gcm_elev_varname, main_glac_rgi)
else:
    print('\n\tModel Error: please choose an option that exists for downscaling climate data. Exiting model run now.\n')
    exit()
# Add GCM time series to the dates_table
dates_table['date_gcm'] = gcm_time_series
# Print time elapsed
timeelapsed_step3 = timeit.default_timer() - timestart_step3
print('Step 3 time:', timeelapsed_step3, "s\n")

#%%=== STEP FOUR: MASS BALANCE CALCULATIONS ===========================================================================
timestart_step4 = timeit.default_timer()
# Create output netcdf file
#output.netcdf_output_create(input.rgi_regionsO1[0], main_glac_hyps, dates_table, annual_columns)

## Create empty dataframes for monthly glacier-wide output
## Monthly glacier-wide accumulation [m w.e.]
#main_glac_acc_monthly = pd.DataFrame(0, index=main_glac_rgi.index, columns=monthly_columns)
## Monthly glacier-wide refreeze [m w.e.]
#main_glac_refreeze_monthly = pd.DataFrame(0, index=main_glac_rgi.index, columns=monthly_columns)
## Monthly glacier-wide melt [m w.e.]
#main_glac_melt_monthly = pd.DataFrame(0, index=main_glac_rgi.index, columns=monthly_columns)
## Monthly glacier-wide frontal ablation [m w.e.]
#main_glac_frontal_ablation_monthly = pd.DataFrame(0, index=main_glac_rgi.index, columns=monthly_columns)
## Monthly glacier-wide specific mass balance [m w.e.]
#main_glac_massbal_clim_mwe_monthly = pd.DataFrame(0, index=main_glac_rgi.index, columns=monthly_columns)
## Monthly total glacier area [km**2]
#main_glac_area_monthly = pd.DataFrame(0, index=main_glac_rgi.index, columns=monthly_columns)
## Monthly Equilibrium Line Altitude (ELA) [m a.s.l.]
#main_glac_ELA_monthly = pd.DataFrame(0, index=main_glac_rgi.index, columns=monthly_columns)
## Monthly Accumulation-Area Ratio (AAR) [%]
#main_glac_AAR_monthly = pd.DataFrame(0, index=main_glac_rgi.index, columns=monthly_columns)
## Monthly Snow Line Altitude [m a.s.l.]
#main_glac_snowline_monthly = pd.DataFrame(0, index=main_glac_rgi.index, columns=monthly_columns)
## Monthly runoff [m**3]
#main_glac_runoff_monthly = pd.DataFrame(0, index=main_glac_rgi.index, columns=monthly_columns)
## Empty dataframes for annual glacier-wide output (rows = glaciers, columns = years)
## Annual glacier-wide specific accumulation [m w.e.]
#main_glac_acc_annual = pd.DataFrame(0, index=main_glac_rgi.index, columns=annual_columns)
## Annual glacier-wide specific refreeze [m w.e.]
#main_glac_refreeze_annual = pd.DataFrame(0, index=main_glac_rgi.index, columns=annual_columns)
## Annual glacier-wide specific melt [m w.e.]
#main_glac_melt_annual = pd.DataFrame(0, index=main_glac_rgi.index, columns=annual_columns)
## Annual glacier-wide specific frontal ablation [m w.e.]
#main_glac_frontal_ablation_annual = pd.DataFrame(0, index=main_glac_rgi.index, columns=annual_columns)
## Annual glacier-wide specific climatic mass balance [m w.e.]
#main_glac_massbal_clim_mwe_annual = pd.DataFrame(0, index=main_glac_rgi.index, columns=annual_columns)
## Annual glacier-wide specific total mass balance [m w.e.]
#main_glac_massbal_total_mwe_annual = pd.DataFrame(0, index=main_glac_rgi.index, columns=annual_columns)
## Annual glacier area at start of each year [km**2]
#main_glac_area_annual = pd.DataFrame(0, index=main_glac_rgi.index, columns=annual_columns)
## Annual glacier volume at start of each year [km**3]
#main_glac_volume_annual = pd.DataFrame(0, index=main_glac_rgi.index, columns=annual_columns)
## Annual Equilibrium Line Altitude (ELA) [m a.s.l.]
#main_glac_ELA_annual = pd.DataFrame(0, index=main_glac_rgi.index, columns=annual_columns)
## Annual Accumulation-Area Ratio (AAR) [m a.s.l.]
#main_glac_AAR_annual = pd.DataFrame(0, index=main_glac_rgi.index, columns=annual_columns)
## Annual snowline altitude [m a.s.l.]
#main_glac_snowline_annual = pd.DataFrame(0, index=main_glac_rgi.index, columns=annual_columns)
## Annual runoff [m**3]
#main_glac_runoff_annual = pd.DataFrame(0, index=main_glac_rgi.index, columns=annual_columns)

#for glac in range(main_glac_rgi.shape[0]):
for glac in [0]:
    # Downscale the gcm temperature [degC] to each bin
    glac_bin_temp = massbalance.downscaletemp2bins(main_glac_rgi, main_glac_hyps, gcm_glac_temp, gcm_glac_elev, glac)
    # Downscale the gcm precipitation [m] to each bin (includes solid and liquid precipitation)
    glac_bin_precsnow = massbalance.downscaleprec2bins(main_glac_rgi, main_glac_hyps, gcm_glac_prec, gcm_glac_elev,glac)
    # Compute accumulation [m w.e.] and precipitation [m] for each bin
    glac_bin_prec, glac_bin_acc = massbalance.accumulationbins(glac_bin_temp, glac_bin_precsnow)
    # Compute potential refreeze [m w.e.] for each bin
    glac_bin_refreezepotential = massbalance.refreezepotentialbins(glac_bin_temp, dates_table)
    # Set initial surface type for first timestep [0=off-glacier, 1=ice, 2=snow, 3=firn, 4=debris]
    surfacetype = main_glac_surftypeinit.iloc[glac,:].values
    # Create surface type DDF dictionary (manipulate this function for calibration or for each glacier)
    surfacetype_ddf_dict = modelsetup.surfacetypeDDFdict()
    

    # List input matrices to simplify creating a mass balance function:
    #  - glac_bin_temp
    #  - glac_bin_acc
    #  - glac_bin_refreezepotential
    #  - surfacetype
    #  - surfacetype_ddf_dict
    #  - dayspermonth
    #  - main_glac_hyps
    # Variables to export with function
    glac_bin_snowdepth = np.zeros(glac_bin_temp.shape)
    glac_bin_melt = np.zeros(glac_bin_temp.shape)
    glac_bin_meltsnow = np.zeros(glac_bin_temp.shape)
    glac_bin_meltrefreeze = np.zeros(glac_bin_temp.shape)
    glac_bin_meltglac = np.zeros(glac_bin_temp.shape)
    glac_bin_refreeze = np.zeros(glac_bin_temp.shape)
    glac_bin_frontalablation = np.zeros(glac_bin_temp.shape)
    glac_bin_massbal_clim_mwe = np.zeros(glac_bin_temp.shape)
    glac_bin_massbal_clim_mwe_annual = np.zeros((glac_bin_temp.shape[0],annual_columns.shape[0]))
    glac_bin_massbal_total_mwe_annual = np.zeros((glac_bin_temp.shape[0],annual_columns.shape[0])) 
    glac_bin_surfacetype_annual = np.zeros((glac_bin_temp.shape[0],annual_columns.shape[0]))
    glac_bin_icethickness_annual = np.zeros((glac_bin_temp.shape[0],annual_columns.shape[0]))
    glac_bin_area_annual = np.zeros((glac_bin_temp.shape[0],annual_columns.shape[0]))
    glac_bin_volume_annual = np.zeros((glac_bin_temp.shape[0],annual_columns.shape[0]))
    glac_bin_width_annual = np.zeros((glac_bin_temp.shape[0],annual_columns.shape[0]))
    
    # Local variables used within the function
    snowdepth_remaining = np.zeros(glac_bin_temp.shape[0])
    dayspermonth = dates_table['daysinmonth'].values
    surfacetype_ddf = np.zeros(glac_bin_temp.shape[0])
    refreeze_potential = np.zeros(glac_bin_temp.shape[0])
    glacier_area_t0 = main_glac_hyps.iloc[glac,:].values.astype(float)
    icethickness_t0 = main_glac_icethickness.iloc[glac,:].values.astype(float)
    elev_bins = main_glac_hyps.columns.values
    
    
    # Enter loop for each timestep (required to allow for snow accumulation which may alter surface type)
#    for step in range(glac_bin_temp.shape[1]):
#    for step in range(0,26):
    for step in range(0,12):
        # Remove input that is off-glacier (required for each timestep as glacier extent may vary over time)
        glac_bin_temp[surfacetype==0,step] = 0
        glac_bin_acc[surfacetype==0,step] = 0
        glac_bin_refreezepotential[surfacetype==0,step] = 0
        
        # INDEXING INTO GLACIER ROWS SPEEDS UP SOME CALCULATIONS AND SLOWS DOWN OTHERS
        # - currently running operations across entire column is the fastest
        # - using [bot:top] is only tenths of a microsecond slower
        # - using [surfacetype!=0] is 4x slower
        
        # Compute the snow depth and melt for each bin...
        # Snow depth [m w.e.] = snow remaining + new snow
        glac_bin_snowdepth[:,step] = snowdepth_remaining + glac_bin_acc[:,step]
        # Available energy for melt [degC day]
        melt_energy_available = glac_bin_temp[:,step]*dayspermonth[step]
        melt_energy_available[melt_energy_available < 0] = 0
        # Snow melt [m w.e.]
        glac_bin_meltsnow[:,step] = surfacetype_ddf_dict[2] * melt_energy_available
        # snow melt cannot exceed the snow depth
        glac_bin_meltsnow[glac_bin_meltsnow[:,step] > glac_bin_snowdepth[:,step], step] = (
                glac_bin_snowdepth[glac_bin_meltsnow[:,step] > glac_bin_snowdepth[:,step], step])
        # Energy remaining after snow melt [degC day]
        melt_energy_available = melt_energy_available - glac_bin_meltsnow[:,step] / surfacetype_ddf_dict[2]
        melt_energy_available[abs(melt_energy_available) < input.tolerance] = 0
        # Compute the refreeze, refreeze melt, and any changes to the snow depth...
        # Refreeze potential [m w.e.]
        #  timing of refreeze potential will vary with the method, e.g., annual air temperature approach updates 
        #  annually vs heat conduction approach which updates monthly; hence, check if refreeze is being udpated
        if glac_bin_refreezepotential[:,step].max() > 0:
            refreeze_potential = glac_bin_refreezepotential[:,step]
        # Refreeze [m w.e.]
        #  refreeze cannot exceed the amount of snow melt, since it needs a source (accumulation zone modified below)
        glac_bin_refreeze[:,step] = glac_bin_meltsnow[:,step]
        # refreeze cannot exceed refreeze potential
        glac_bin_refreeze[glac_bin_refreeze[:,step] > refreeze_potential, step] = (
                refreeze_potential[glac_bin_refreeze[:,step] > refreeze_potential])
        glac_bin_refreeze[abs(glac_bin_refreeze[:,step]) < input.tolerance, step] = 0
        # Refreeze melt [m w.e.]
        glac_bin_meltrefreeze[:,step] = surfacetype_ddf_dict[2] * melt_energy_available
        # refreeze melt cannot exceed the refreeze
        glac_bin_meltrefreeze[glac_bin_meltrefreeze[:,step] > glac_bin_refreeze[:,step], step] = (
                glac_bin_refreeze[glac_bin_meltrefreeze[:,step] > glac_bin_refreeze[:,step], step])
        # Energy remaining after refreeze melt [degC day]
        melt_energy_available = melt_energy_available - glac_bin_meltrefreeze[:,step] / surfacetype_ddf_dict[2]
        melt_energy_available[abs(melt_energy_available) < input.tolerance] = 0
        # Snow remaining [m w.e.]
        snowdepth_remaining = (glac_bin_snowdepth[:,step] + glac_bin_refreeze[:,step] - glac_bin_meltsnow[:,step] - 
                               glac_bin_meltrefreeze[:,step])
        # Compute any remaining melt and any additional refreeze in the accumulation zone...
        # DDF based on surface type [m w.e. degC-1 day-1]
        for surfacetype_idx in surfacetype_ddf_dict: 
            surfacetype_ddf[surfacetype == surfacetype_idx] = surfacetype_ddf_dict[surfacetype_idx]
        # Glacier melt [m w.e.] based on remaining energy
        glac_bin_meltglac[:,step] = surfacetype_ddf * melt_energy_available
        # Energy remaining after glacier surface melt [degC day]
        melt_energy_available[surfacetype != 0] = (melt_energy_available[surfacetype != 0] - 
                             glac_bin_meltglac[surfacetype != 0, step] / surfacetype_ddf[surfacetype != 0])
        melt_energy_available[abs(melt_energy_available) < input.tolerance] = 0
        #  must specify on-glacier values, otherwise this will divide by zero and cause an error
        # Additional refreeze in the accumulation area [m w.e.]
        #  refreeze in accumulation zone = refreeze of snow + refreeze of underlying snow/firn
        glac_bin_refreeze[(surfacetype == 2) | (surfacetype == 3), step] = (
                glac_bin_refreeze[(surfacetype == 2) | (surfacetype == 3), step] +
                glac_bin_melt[(surfacetype == 2) | (surfacetype == 3), step])
        #  ALTERNATIVE CALCULATION: use ELA and reference into the bins - this requires the ELA and bin size such that
        #  the proper row can be referenced (this would not need to be updated assuming range of bins doesn't change.
        #  This may be an improvement, since this will need to be updated if more surface types are added in the future.
        # refreeze cannot exceed refreeze potential
        glac_bin_refreeze[glac_bin_refreeze[:,step] > refreeze_potential, step] = (
                refreeze_potential[glac_bin_refreeze[:,step] > refreeze_potential])
        # update refreeze potential
        refreeze_potential = refreeze_potential - glac_bin_refreeze[:,step]
        refreeze_potential[abs(refreeze_potential) < input.tolerance] = 0
        # Total melt (snow + refreeze + glacier)
        glac_bin_melt[:,step] = glac_bin_meltglac[:,step] + glac_bin_meltrefreeze[:,step] + glac_bin_meltsnow[:,step]
        # Reset available energy to ensure no energy is carried over into next timestep
#        melt_energy_available = np.zeros(glac_bin_temp.shape[0]) 
        
        # Compute frontal ablation
        #   - INSERT CODE HERE
        
        # Climatic mass balance [m w.e.]
        glac_bin_massbal_clim_mwe[:,step] = (glac_bin_acc[:,step] + glac_bin_refreeze[:,step] - glac_bin_melt[:,step] -
                                             glac_bin_frontalablation[:,step])
        #  climatic mass balance = accumulation + refreeze - melt - frontal ablation
        
        
        # ENTER ANNUAL LOOP
        #  at the end of each year, update glacier characteristics (surface type, length, area, volume)
        if (step + 1) % annual_divisor == 0:
            # % gives the remainder; since step starts at 0, add 1 such that this switches at end of year
            # Index year
            year_index = int(step/annual_divisor)
            #  Note: year_index*annual_divisor gives initial step of the given year
            #        step + 1 gives final step of the given year
            # Annual climatic mass balance [m w.e.]
            glac_bin_massbal_clim_mwe_annual[:,year_index] = (
                glac_bin_massbal_clim_mwe[:,year_index*annual_divisor:step+1].sum(1))
            
            ###### SURFACE TYPE (convert to function) #####
            glac_bin_surfacetype_annual[:,year_index] = surfacetype
            # Compute the surface type for each bin
            #  Next year's surface type is based on the bin's average annual climatic mass balance over the last 5
            #  years.  If less than 5 years, then use the average of the existing years.
            if year_index < 5:
                # Calculate average annual climatic mass balance since run began
                massbal_clim_mwe_runningavg = glac_bin_massbal_clim_mwe_annual[:,0:year_index+1].mean(1)
            else:
                massbal_clim_mwe_runningavg = glac_bin_massbal_clim_mwe_annual[:,year_index-4:year_index+1].mean(1)
            # If the average annual specific climatic mass balance is negative, then the surface type is ice (or debris)
            surfacetype[(surfacetype!=0) & (glac_bin_massbal_clim_mwe_annual[:,year_index]<=0)] = 1
            # If the average annual specific climatic mass balance is positive, then the surface type is snow (or firn)
            surfacetype[(surfacetype!=0) & (glac_bin_massbal_clim_mwe_annual[:,year_index]>0)] = 2
            # Apply surface type model options
            # If firn surface type option is included, then snow is changed to firn
            if input.option_surfacetype_firn == 1:
                surfacetype[surfacetype == 2] = 3
            if input.option_surfacetype_debris == 1:
                print('Need to code the model to include debris.  Please choose an option that currently exists.\n'
                      'Exiting the model run.')
                exit()
            
            ##### GLACIER GEOMETRY CHANGE (convert to function) #####
            # Annual glacier-wide volume change [km**3]
            glacier_volumechange = ((glac_bin_massbal_clim_mwe_annual[:, year_index] / 1000 * input.density_water / 
                                     input.density_ice * glacier_area_t0).sum())
            
            # REMOVE WHEN FINISHED TESTING!!!!
#            glacier_volumechange = 0.2
            glacier_volumechange = -1.5
            
            
            #  units: [m w.e.] * (1 km / 1000 m) * (1000 kg / (1 m water * m**2) * (1 m ice * m**2 / 900 kg) * [km**2] 
            #         = km**3 ice
            # Reset the annual glacier area and ice thickness
            glacier_area_t1 = np.zeros(glacier_area_t0.shape)
            icethickness_t1 = np.zeros(glacier_area_t0.shape)
            # If volume loss is less than the glacier volume, then redistribute mass loss/gains across the glacier;
            #  otherwise, the glacier disappears (area and thickness are set to zero as shown above)
            if -1 * glacier_volumechange < (icethickness_t0 / 1000 * glacier_area_t0).sum():
                # Determine where glacier exists
                glac_idx_t0 = glacier_area_t0.nonzero()[0]
                # Ice thickness [m ice] and ice thickness change [m ice] after redistribution of volume gain/loss
#                icethickness_t1, icethickness_change = massbalance.massredistribution(icethickness_t0, glacier_area_t0, 
#                                                                                      glac_idx_t0, glacier_volumechange)
                
                #Select the factors for the normalized ice thickness change curve based on glacier area
                if glacier_area_t0.sum() > 20:
                    [gamma, a, b, c] = [6, -0.02, 0.12, 0]
                elif glacier_area_t0.sum() > 5:
                    [gamma, a, b, c] = [4, -0.05, 0.19, 0.01]
                else:
                    [gamma, a, b, c] = [2, -0.30, 0.60, 0.09]
                 # reset variables
                elevrange_norm = np.zeros(glacier_area_t0.shape)
                icethicknesschange_norm = np.zeros(glacier_area_t0.shape)
                icethickness_t1 = np.zeros(glacier_area_t0.shape)
                glacier_area_t1 = np.zeros(glacier_area_t0.shape)
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
                if input.option_glaciershape == 1:
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
                elif input.option_glaciershape == 2:
                    # Ice thickness at end of timestep for rectangle [m ice]
                    #  H_1 = H_0 + delta_Vol / A_0
                    icethickness_t1[glac_idx_t0] = (((icethickness_t0[glac_idx_t0] / 1000) + 
                                   bin_volumechange[glac_idx_t0] / glacier_area_t0[glac_idx_t0]) * 1000)
                    # Glacier area constant for rectangle [km**2]
                    #  A_1 = A_0
                    glacier_area_t1[glac_idx_t0] = glacier_area_t0[glac_idx_t0]
                elif input.option_glaciershape == 3:
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
                
                # Ice thickness change [m ice]
                icethickness_change = icethickness_t1 - icethickness_t0


                # Glacier retreat
                #  if glacier retreats (ice thickness < 0), then redistribute mass loss across the rest of the glacier
                while (icethickness_t1[glac_idx_t0] <= 0).any() == True:
                    # Glacier volume change associated with retreat [km**3]
                    glacier_volumechange_retreat = (-1*(icethickness_t0[glac_idx_t0][icethickness_t1[glac_idx_t0] <= 0] 
                            / 1000 * glacier_area_t0[glac_idx_t0][icethickness_t1[glac_idx_t0] <= 0]).sum())
                    #  multiplying by -1 makes volume change negative
                    # Glacier volume change remaining [km**3]
                    glacier_volumechange = glacier_volumechange - glacier_volumechange_retreat
                    # update glacier area and ice thickness to account for retreat
                    glacier_area_t0[icethickness_t1 <= 0] = 0
                    icethickness_t0[icethickness_t1 <= 0] = 0
                    glac_idx_t0 = glacier_area_t0.nonzero()[0]
                    # recalculate ice thickness [m ice] after retreat has been removed
                    icethickness_t1, icethickness_change = massbalance.massredistribution(icethickness_t0, 
                                                                  glacier_area_t0, glac_idx_t0, glacier_volumechange)    
                    
                    
                # Glacier surge
                #  if glacier surges (ice thickness change exceeds threshold), then redistribute mass gain in new bins
                if (icethickness_change > input.icethickness_surgethreshold).any() == True:
                    # Glacier volume change associated with surge [km**3]
                    glacier_volumechange_surge = ((
                            (icethickness_change[icethickness_change > input.icethickness_surgethreshold] - 
                             input.icethickness_surgethreshold) / 1000 * 
                             glacier_area_t0[icethickness_change > input.icethickness_surgethreshold]).sum())
                    #  (Change in ice thickness - threshold) * (1 km / 1000 m) * glacier_area, where threshold exceeded
                    # Surge characteristics
                    # Indices that define the glacier terminus
                    glacier_idx_terminus = (glac_idx_t0[(glac_idx_t0 - glac_idx_t0[0] + 1) / 
                                                           glac_idx_t0.shape[0] * 100 < input.terminus_percentage])
                    # Average area of glacier terminus [km**2]
                    terminus_area_avg = glacier_area_t0[glacier_idx_terminus].mean()                
                    # Average ice thickness of glacier terminus [m]
                    terminus_icethickness_avg = icethickness_t0[glacier_idx_terminus].mean()
                    # Maximum surge bin volume [km**3]
                    surge_bin_volume_max = terminus_icethickness_avg / 1000 * terminus_area_avg
                    # Number of bins to add for present surge [-]
                    surge_bins2add = np.ceil(glacier_volumechange_surge / surge_bin_volume_max).astype(int)
                    # Surge area [km**2]
                    surge_bin_area = glacier_volumechange_surge / (terminus_icethickness_avg / 1000)
                    # Add the surge bins
                    # ice thickness equals the average terminus ice thickness
                    icethickness_t1[(glac_idx_t0[0] - surge_bins2add):glac_idx_t0[0]] = terminus_icethickness_avg 
                    # glacier area for all filled bins is the average terminus glacier area
                    glacier_area_t1[(glac_idx_t0[0] - surge_bins2add + 1):glac_idx_t0[0]] = terminus_area_avg
                    # glacier area for the most downglacier bin is based on the remaining volume change
                    glacier_area_t1[(glac_idx_t0[0] - surge_bins2add)] = ((glacier_volumechange_surge - 
                                    (surge_bins2add - 1) * surge_bin_volume_max) / (terminus_icethickness_avg / 1000))
                    
                    # Simple way to ensure that small areas at terminus don't influence the average area of the bins is:
                    # (1) calculate terminus area and ice thickness average based on all the bins excluding the last one
                    #     which may not be totally full
                    # (2) make sure that you fill up the bottom bin, i.e., bins that have been added during surges, 
                    #     prior to moving to the next bin
                    # Note:
                    # If bin retreats and then surges, the area and ice thickness pre-retreat should be used instead
                    # This will also take care of the cases where you need to skip steep bins at high altitudes, i.e.,
                    # discontinuous glaciers


                A = (glacier_area_t1 * icethickness_t1 / 1000).sum() 
                
                A_hyps = main_glac_hyps.iloc[glac,:].values.astype(float)
                A_ice = main_glac_icethickness.iloc[glac,:].values.astype(float) 
                A1 = (A_hyps * A_ice / 1000).sum()
                print(A-A1)

            # Somewhere we need to update ice thickness and glacier area for the next time step
                     

                     
                 
                 
                 # Total specific mass balance [m w.e.] (mass balance after mass redistribution)
#                 glac_bin_massbal_total_mwe_annual[:,year_index] = (icethicknesschange * input.density_ice / 
#                                                                     input.density_water)
                 #  units: [m ice] * [kg / (m_ice * m**2)] / [kg / (m_water * m**2)] = m w.e.
                 # Ice thickness [m ice]
#                 glac_bin_icethickness_annual[:,year_index] = icethickness_t1
                 
                 
                 
                 
                 

#                     # Update the volume [km**3] for the next year
#                     glac_bin_volume_annual.iloc[:, year_position + 1] = (glac_bin_area_annual.iloc[:, year_position + 1] *
#                         glac_bin_icethickness_annual.iloc[:, year_position + 1] / 1000)
                     #  units: [km**2] * [m] * (1 km / 1000 m) = km**3
        
         # ADD IN AREA CHANGES
         # NOTE: WANT TO HAVE OPTIONS TO USE THESE NORMALIZED ELEVATION CHANGE CURVES OR USE THE REAL THING!
     
        
#            # For calibration run (option_modelruntype = 0), area is constant while ice thickness is changed
#            if input.option_modelrun_type == 0:
#                # glacier ice thickness [m] changes according to specific climatic mass balance
#                #  NOTE: this should also include redistribution!
        
# While in glacier loop, compile the monthly data into a netcdf

timeelapsed_step4 = timeit.default_timer() - timestart_step4
print('Step 4 time:', timeelapsed_step4, "s\n")

#%%=== STEP FIVE: DATA ANALYSIS / OUTPUT ==============================================================================







