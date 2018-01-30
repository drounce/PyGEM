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
# Glacier hypsometry [km**2], total area
main_glac_hyps = modelsetup.import_Husstable(main_glac_rgi, input.rgi_regionsO1, input.hyps_filepath, 
                                             input.hyps_filedict, input.indexname, input.hyps_colsdrop)
# Ice thickness [m], average
main_glac_icethickness = modelsetup.import_Husstable(main_glac_rgi, input.rgi_regionsO1, input.thickness_filepath, 
                                                 input.thickness_filedict, input.indexname, input.thickness_colsdrop)
# Width [km], average
if input.option_glaciershape_width == 1:
    main_glac_width = modelsetup.import_Husstable(main_glac_rgi, input.rgi_regionsO1, input.width_filepath, 
                                                  input.width_filedict, input.indexname, input.width_colsdrop)
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
    # Precipitation [m] and GCM dates
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
    surfacetype = main_glac_surftypeinit.iloc[glac,:].values.copy()
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
    glac_bin_refreeze = np.zeros(glac_bin_temp.shape)
    glac_bin_melt = np.zeros(glac_bin_temp.shape)
    glac_bin_meltsnow = np.zeros(glac_bin_temp.shape)
    glac_bin_meltrefreeze = np.zeros(glac_bin_temp.shape)
    glac_bin_meltglac = np.zeros(glac_bin_temp.shape)
    glac_bin_frontalablation = np.zeros(glac_bin_temp.shape)
    glac_bin_snowpack = np.zeros(glac_bin_temp.shape)
    glac_bin_massbal_clim_mwe = np.zeros(glac_bin_temp.shape)
    glac_bin_massbal_clim_mwe_annual = np.zeros((glac_bin_temp.shape[0],annual_columns.shape[0]))
    glac_bin_surfacetype_annual = np.zeros((glac_bin_temp.shape[0],annual_columns.shape[0]))
    glac_bin_icethickness_annual = np.zeros((glac_bin_temp.shape[0], annual_columns.shape[0] + 1))
    glac_bin_area_annual = np.zeros((glac_bin_temp.shape[0], annual_columns.shape[0] + 1))
    
    # Local variables used within the function
    snowpack_remaining = np.zeros(glac_bin_temp.shape[0])
    dayspermonth = dates_table['daysinmonth'].values
    surfacetype_ddf = np.zeros(glac_bin_temp.shape[0])
    refreeze_potential = np.zeros(glac_bin_temp.shape[0])
    glacier_area_t0 = main_glac_hyps.iloc[glac,:].values.astype(float)
    icethickness_t0 = main_glac_icethickness.iloc[glac,:].values.astype(float)
    icethickness_initial = main_glac_icethickness.iloc[glac,:].values.astype(float)
    elev_bins = main_glac_hyps.columns.values
    
    # Enter loop for each timestep (required to allow for snow accumulation which may alter surface type)
    for step in range(glac_bin_temp.shape[1]):
#    for step in range(0,26):
#    for step in range(0,12):
        # Remove input that is off-glacier (required for each timestep as glacier extent may vary over time)
        glac_bin_temp[surfacetype==0,step] = 0
        glac_bin_acc[surfacetype==0,step] = 0
        glac_bin_refreezepotential[surfacetype==0,step] = 0
        
        # Compute the snow depth and melt for each bin...
        # Snow depth / 'snowpack' [m w.e.] = snow remaining + new snow
        glac_bin_snowpack[:,step] = snowpack_remaining + glac_bin_acc[:,step]
        # Available energy for melt [degC day]
        #  include option to adjust air temperature based on changes in surface elevation as ice melts
        if input.option_adjusttemp_surfelev == 1:
            glac_bin_temp[:,step] = glac_bin_temp[:,step] + input.lr_glac * (icethickness_t0 - icethickness_initial)
            #  T_air = T+air + lr_glac * (icethickness_present - icethickness_initial)
            
            # THIS OPTION ALSO NEEDS TO AFFECT REFREEZE, WHICH IS CURRENTLY COMPUTED OUTSIDE THIS LOOP...
            
        melt_energy_available = glac_bin_temp[:,step]*dayspermonth[step]
        melt_energy_available[melt_energy_available < 0] = 0
        # Snow melt [m w.e.]
        glac_bin_meltsnow[:,step] = surfacetype_ddf_dict[2] * melt_energy_available
        # snow melt cannot exceed the snow depth
        glac_bin_meltsnow[glac_bin_meltsnow[:,step] > glac_bin_snowpack[:,step], step] = (
                glac_bin_snowpack[glac_bin_meltsnow[:,step] > glac_bin_snowpack[:,step], step])
        # Energy remaining after snow melt [degC day]
        melt_energy_available = melt_energy_available - glac_bin_meltsnow[:,step] / surfacetype_ddf_dict[2]
        # remove low values of energy available cause by rounding errors in the step above (e.g., less than 10**-12)
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
        # remove low values of energy available cause by rounding errors
        melt_energy_available[abs(melt_energy_available) < input.tolerance] = 0
        # Snow remaining [m w.e.]
        snowpack_remaining = (glac_bin_snowpack[:,step] + glac_bin_refreeze[:,step] - glac_bin_meltsnow[:,step] - 
                               glac_bin_meltrefreeze[:,step])
        # Compute any remaining melt and any additional refreeze in the accumulation zone...
        # DDF based on surface type [m w.e. degC-1 day-1]
        for surfacetype_idx in surfacetype_ddf_dict: 
            surfacetype_ddf[surfacetype == surfacetype_idx] = surfacetype_ddf_dict[surfacetype_idx]
        # Glacier melt [m w.e.] based on remaining energy
        glac_bin_meltglac[:,step] = surfacetype_ddf * melt_energy_available
        # Energy remaining after glacier surface melt [degC day]
        #  must specify on-glacier values, otherwise this will divide by zero and cause an error
        melt_energy_available[surfacetype != 0] = (melt_energy_available[surfacetype != 0] - 
                             glac_bin_meltglac[surfacetype != 0, step] / surfacetype_ddf[surfacetype != 0])
        # remove low values of energy available cause by rounding errors
        melt_energy_available[abs(melt_energy_available) < input.tolerance] = 0
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
            # for first year, need to record glacier area [km**2] and ice thickness [m ice]
            if year_index == 0:
                glac_bin_area_annual[:,year_index] = main_glac_hyps.iloc[glac,:].values.astype(float)
                glac_bin_icethickness_annual[:,year_index] = main_glac_icethickness.iloc[glac,:].values.astype(float)
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
            
            
            # ADD CALIBRATION OPTION PRE-GEOMETRY CHANGE
            #  Calibration does not require geometry change, so have ability to turn this off with an option!
            #  Simply keep area constant and allow ice thickness to grow
            
            ##### GLACIER GEOMETRY CHANGE (convert to function) #####
            # Reset the annual glacier area and ice thickness
            glacier_area_t1 = np.zeros(glacier_area_t0.shape)
            icethickness_t1 = np.zeros(glacier_area_t0.shape)
            # Annual glacier-wide volume change [km**3]
            glacier_volumechange = ((glac_bin_massbal_clim_mwe_annual[:, year_index] / 1000 * input.density_water / 
                                     input.density_ice * glacier_area_t0).sum())
            #  units: [m w.e.] * (1 km / 1000 m) * (1000 kg / (1 m water * m**2) * (1 m ice * m**2 / 900 kg) * [km**2] 
            #         = km**3 ice            
            
            # If volume loss is less than the glacier volume, then redistribute mass loss/gains across the glacier;
            #  otherwise, the glacier disappears (area and thickness are set to zero as shown above)
            if -1 * glacier_volumechange < (icethickness_t0 / 1000 * glacier_area_t0).sum():
                # Determine where glacier exists
                glac_idx_t0 = glacier_area_t0.nonzero()[0]
                    # Ice thickness [m ice] and ice thickness change [m ice] after redistribution of volume gain/loss
                icethickness_t1, glacier_area_t1, icethickness_change = massbalance.massredistribution(
                        icethickness_t0, glacier_area_t0, glac_idx_t0, glacier_volumechange)
                # Glacier retreat
                #  if glacier retreats (ice thickness < 0), then redistribute mass loss across the rest of the glacier
                glac_idx_t0_raw = glac_idx_t0.copy()
                if (icethickness_t1[glac_idx_t0] <= 0).any() == True:
                    # Record glacier area and ice thickness before retreat corrections applied
                    glacier_area_t0_raw = glacier_area_t0.copy()
                    icethickness_t0_raw = icethickness_t0.copy()
                while (icethickness_t1[glac_idx_t0_raw] <= 0).any() == True:
                    # Glacier volume change associated with retreat [km**3]
                    glacier_volumechange_retreat = (-1*(icethickness_t0[glac_idx_t0][icethickness_t1[glac_idx_t0] <= 0] 
                            / 1000 * glacier_area_t0[glac_idx_t0][icethickness_t1[glac_idx_t0] <= 0]).sum())
                    #  multiplying by -1 makes volume change negative
                    # Glacier volume change remaining [km**3]
                    glacier_volumechange = glacier_volumechange - glacier_volumechange_retreat
                    # update glacier area and ice thickness to account for retreat
                    glacier_area_t0_raw[icethickness_t1 <= 0] = 0
                    icethickness_t0_raw[icethickness_t1 <= 0] = 0
                    glac_idx_t0_raw = glacier_area_t0_raw.nonzero()[0]
                    # recalculate ice thickness [m ice] after retreat has been removed
                    icethickness_t1, glacier_area_t1, icethickness_change = massbalance.massredistribution(
                            icethickness_t0_raw, glacier_area_t0_raw, glac_idx_t0_raw, glacier_volumechange)    
                # Glacier advances
                #  if glacier advancess (ice thickness change exceeds threshold), then redistribute mass gain in new bins
                if (icethickness_change > input.icethickness_advancethreshold).any() == True:                    
                    # Record glacier area and ice thickness before advance corrections applied
                    glacier_area_t1_raw = glacier_area_t1.copy()
                    icethickness_t1_raw = icethickness_t1.copy()
                    # Index bins that are surging
                    icethickness_change[icethickness_change <= input.icethickness_advancethreshold] = 0
                    glac_idx_advance = icethickness_change.nonzero()[0]
                    # Update ice thickness based on maximum advance threshold [m ice]
                    icethickness_t1[glac_idx_advance] = (icethickness_t1[glac_idx_advance] - 
                                   (icethickness_change[glac_idx_advance] - input.icethickness_advancethreshold))
                    # Update glacier area based on reduced ice thicknesses [km**2]
                    if input.option_glaciershape == 1:
                        # Glacier area for parabola [km**2] (A_1 = A_0 * (H_1 / H_0)**0.5)
                        glacier_area_t1[glac_idx_advance] = (glacier_area_t1_raw[glac_idx_advance] * 
                                       (icethickness_t1[glac_idx_advance] / icethickness_t1_raw[glac_idx_advance])**0.5)
                    elif input.option_glaciershape == 2:
                        # Glacier area constant for rectangle [km**2] (A_1 = A_0)
                        glacier_area_t1[glac_idx_advance] = glacier_area_t1_raw[glac_idx_advance]
                    elif input.option_glaciershape == 3:
                        # Glacier area for triangle [km**2] (A_1 = A_0 * H_1 / H_0)
                        glacier_area_t1[glac_idx_t0] = (glacier_area_t1_raw[glac_idx_t0] * 
                                       icethickness_t1[glac_idx_t0] / icethickness_t1_raw[glac_idx_t0])
                    # Advance volume [km**3]
                    advance_volume = ((glacier_area_t1_raw[glac_idx_advance] * 
                                      icethickness_t1_raw[glac_idx_advance] / 1000).sum() - 
                                      (glacier_area_t1[glac_idx_advance] * icethickness_t1[glac_idx_advance] / 
                                       1000).sum())
                    # Advance characteristics
                    # Indices that define the glacier terminus
                    glac_idx_terminus = (glac_idx_t0[(glac_idx_t0 - glac_idx_t0[0] + 1) / 
                                                     glac_idx_t0.shape[0] * 100 < input.terminus_percentage])
                    # Average area of glacier terminus [km**2]
                    terminus_area_avg = glacier_area_t0[glac_idx_terminus].mean()                
                    # Average ice thickness of glacier terminus [m]
                    terminus_icethickness_avg = icethickness_t0[glac_idx_terminus].mean()
                    # Maximum advance bin volume [km**3]
                    advance_bin_volume_max = terminus_icethickness_avg / 1000 * terminus_area_avg
                    # Number of bins to add for present advance [-]
                    advance_bins2add = np.ceil(advance_volume / advance_bin_volume_max).astype(int)
                    # Advance area [km**2]
                    advance_bin_area = advance_volume / (terminus_icethickness_avg / 1000)
                    # Add the advance bins
                    # ice thickness equals the average terminus ice thickness
                    icethickness_t1[(glac_idx_t0[0] - advance_bins2add):glac_idx_t0[0]] = terminus_icethickness_avg 
                    # glacier area for all filled bins is the average terminus glacier area
                    glacier_area_t1[(glac_idx_t0[0] - advance_bins2add + 1):glac_idx_t0[0]] = terminus_area_avg
                    # glacier area for the most downglacier bin is based on the remaining volume change
                    glacier_area_t1[(glac_idx_t0[0] - advance_bins2add)] = ((advance_volume - (advance_bins2add - 1) * 
                                   advance_bin_volume_max) / (terminus_icethickness_avg / 1000))

                    # Simple way to ensure that small areas at terminus don't influence the average area of the bins is:
                    # (1) calculate terminus area and ice thickness average based on all the bins excluding the last one
                    #     which may not be totally full
                    # (2) make sure that you fill up the bottom bin, i.e., bins that have been added during advances, 
                    #     prior to moving to the next bin
                    # Note:
                    # If bin retreats and then advances, the area and ice thickness pre-retreat should be used instead
                    # This will also take care of the cases where you need to skip steep bins at high altitudes, i.e.,
                    # discontinuous glaciers
            
            
            # Record glacier area [km**2] and ice thickness [m ice]
            glac_bin_area_annual[:,year_index + 1] = glacier_area_t1
            glac_bin_icethickness_annual[:,year_index + 1] = icethickness_t1
            # Update glacier area and ice thickness for next year
            glacier_area_t0 = glacier_area_t1.copy()
            icethickness_t0 = icethickness_t1.copy()
            
            
#    # Record variables that are specified by the user to be output
#    # Monthly area [km**2] at each bin
#    glac_bin_area = glac_bin_area_annual[:,0:glac_bin_area_annual.shape[1]-1].repeat(12,axis=1)
#    # Monthly ice thickness [m ice] at each bin
#    glac_bin_icethickness = glac_bin_icethickness_annual[:,0:glac_bin_icethickness_annual.shape[1]-1].repeat(12,axis=1)
#
#    # Annual outputs at each bin:
#    # Annual volume [km**3]
#    glac_bin_volume_annual = glac_bin_area_annual * glac_bin_icethickness_annual / 1000
#    # Annual total specific mass balance [m3] (mass balance after mass redistribution)
#    glac_bin_massbal_total_m3_annual = (glac_bin_volume_annual - np.roll(glac_bin_volume_annual,-1,axis=1))
#    # Annual accumulation [m3]
##    glac_bin_acc_annual = glac_bin_acc.reshape(-1,12).sum(axis=1).reshape(-1,annual_columns.shape[0])
#    glac_bin_acc_annual = (glac_bin_acc.reshape(-1,12).sum(axis=1).reshape(-1,annual_columns.shape[0]) * 
#                           glac_bin_area_annual[:,0:glac_bin_area_annual.shape[1]-1] * 1000**2)
#    # Annual refreeze [m3]
##    glac_bin_refreeze_annual = glac_bin_refreeze.reshape(-1,12).sum(axis=1).reshape(-1,annual_columns.shape[0])
#    glac_bin_refreeze_annual = (glac_bin_refreeze.reshape(-1,12).sum(axis=1).reshape(-1,annual_columns.shape[0]) * 
#                                glac_bin_area_annual[:,0:glac_bin_area_annual.shape[1]-1] * 1000**2)
#    # Annual melt [m3]
##    glac_bin_melt_annual = glac_bin_melt.reshape(-1,12).sum(axis=1).reshape(-1,annual_columns.shape[0])
#    glac_bin_melt_annual = (glac_bin_melt.reshape(-1,12).sum(axis=1).reshape(-1,annual_columns.shape[0]) * 
#                            glac_bin_area_annual[:,0:glac_bin_area_annual.shape[1]-1] * 1000**2)
#    # Annual frontal ablation [m3]
##    glac_bin_frontalablation_annual = (
##            glac_bin_frontalablation.reshape(-1,12).sum(axis=1).reshape(-1,annual_columns.shape[0]))
#    glac_bin_frontalablation_annual = (
#            glac_bin_frontalablation.reshape(-1,12).sum(axis=1).reshape(-1,annual_columns.shape[0]) * 
#            glac_bin_area_annual[:,0:glac_bin_area_annual.shape[1]-1] * 1000**2)
#    # Annual precipitation [m3]
##    glac_bin_prec_annual = glac_bin_prec.reshape(-1,12).sum(axis=1).reshape(-1,annual_columns.shape[0])
#    glac_bin_prec_annual = (glac_bin_prec.reshape(-1,12).sum(axis=1).reshape(-1,annual_columns.shape[0]) * 
#                            glac_bin_area_annual[:,0:glac_bin_area_annual.shape[1]-1] * 1000**2)
#    
#
#    # Monthly glacier-wide Parameters:
#    # Area [km**2]
#    glac_wide_area = glac_bin_area.sum(axis=0)
#    # Accumulation [m w.e.]
#    glac_wide_acc = np.zeros(glac_wide_area.shape)
#    glac_wide_acc[glac_wide_area > 0] = ((glac_bin_acc * glac_bin_area).sum(axis=0)[glac_wide_area > 0] / 
#                                         glac_wide_area[glac_wide_area > 0])
#    # Refreeze [m w.e.]
#    glac_wide_refreeze = np.zeros(glac_wide_area.shape)
#    glac_wide_refreeze[glac_wide_area > 0] = ((glac_bin_refreeze * glac_bin_area).sum(axis=0)[glac_wide_area > 0] / 
#                                              glac_wide_area[glac_wide_area > 0])
#    # Melt [m w.e.]
#    glac_wide_melt = np.zeros(glac_wide_area.shape)
#    glac_wide_melt[glac_wide_area > 0] = ((glac_bin_melt * glac_bin_area).sum(axis=0)[glac_wide_area > 0] / 
#                                          glac_wide_area[glac_wide_area > 0])
#    # Frontal ablation [m w.e.]
#    glac_wide_frontalablation = np.zeros(glac_wide_area.shape)
#    glac_wide_frontalablation[glac_wide_area > 0] = (
#            (glac_bin_frontalablation * glac_bin_area).sum(axis=0)[glac_wide_area > 0] / 
#            glac_wide_area[glac_wide_area > 0])
#    # Mass balance [m w.e.]
#    #  glacier-wide climatic and total mass balance are the same; use climatic since its required to run the model
#    glac_wide_massbal_mwe = np.zeros(glac_wide_area.shape)
#    glac_wide_massbal_mwe[glac_wide_area > 0] = (
#            (glac_bin_massbal_clim_mwe * glac_bin_area).sum(axis=0)[glac_wide_area > 0] / 
#            glac_wide_area[glac_wide_area > 0])
#    # Melt [m w.e.]
#    glac_wide_melt = np.zeros(glac_wide_area.shape)
#    glac_wide_melt[glac_wide_area > 0] = ((glac_bin_melt * glac_bin_area).sum(axis=0)[glac_wide_area > 0] / 
#                                          glac_wide_area[glac_wide_area > 0])
#    # Precipitation [m]
#    glac_wide_prec = np.zeros(glac_wide_area.shape)
#    glac_wide_prec[glac_wide_area > 0] = ((glac_bin_prec * glac_bin_area).sum(axis=0)[glac_wide_area > 0] / 
#                                          glac_wide_area[glac_wide_area > 0])
#    # Runoff [m**3]
#    glac_wide_runoff = (glac_wide_prec + glac_wide_melt - glac_wide_refreeze) * glac_wide_area * (1000)**2
#    #  runoff = precipitation + melt - refreeze
#    #  units: m w.e. * km**2 * (1000 m / 1 km)**2 = m**3
#    # Volume [km**3]
#    glac_wide_volume = (glac_bin_area * glac_bin_icethickness / 1000).sum(axis=0)
#            
#    # Annual glacier-wide Parameters:
#    # Annual volume [km**3]
#    glac_wide_volume_annual = (glac_bin_area_annual * glac_bin_icethickness_annual / 1000).sum(axis=0)
    
    
#    # Annual accumulation [m w.e.]
#    glac_wide_acc_annual = 
#    # Annual refreeze [m w.e.]
#    glac_wide_refreeze_annual = 
#    # Annual melt [m w.e.]
#    glac_wide_melt_annual = 
#    # Annual frontal ablation [m w.e.]
#    glac_wide_frontalablation_annual = 
#    # Annual precipitation [m]
#    glac_wide_prec_annual = 
            
        
         # Options to add:
         # - Refreeze via heat conduction
         # - Volume-Area, Volume-Length scaling
         # - Calibration with constant area, i.e., no mass redistribution, should significantly speed up the code
        
#            # For calibration run (option_modelruntype = 0), area is constant while ice thickness is changed
#            if input.option_modelrun_type == 0:
#                # glacier ice thickness [m] changes according to specific climatic mass balance
#                #  NOTE: this should also include redistribution!
         
         
    # Compute "optional" output, i.e., output that is not required for model to run, but may be desired by user
    # can we pass this into the function? by adding the "-output ______", this would likely make code cleaner
     

        
# While in glacier loop, compile the monthly data into a netcdf

timeelapsed_step4 = timeit.default_timer() - timestart_step4
print('Step 4 time:', timeelapsed_step4, "s\n")

#%%=== STEP FIVE: DATA ANALYSIS / OUTPUT ==============================================================================


