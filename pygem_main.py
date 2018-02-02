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
    elev_bins = main_glac_hyps.columns.values
    glacier_area_t0 = main_glac_hyps.iloc[glac,:].values.astype(float)
    # Inclusion of ice thickness and width, i.e., loading the values may be only required for Huss mass redistribution!
    icethickness_t0 = main_glac_icethickness.iloc[glac,:].values.astype(float)
    width_t0 = main_glac_width.iloc[glac,:].values.astype(float)
    if input.option_adjusttemp_surfelev == 1:
        # ice thickness initial is used to adjust temps to changes in surface elevation
        icethickness_adjusttemp = icethickness_t0.copy()
        icethickness_adjusttemp[0:icethickness_adjusttemp.nonzero()[0][0]] = (
                icethickness_adjusttemp[icethickness_adjusttemp.nonzero()[0][0]])
        #  bins that advance need to have an initial ice thickness; otherwise, the temp adjustment will be based on ice
        #  thickness - 0, which is wrong  Since advancing bins take the thickness of the previous bin, set the initial 
        #  ice thickness of all bins below the terminus to the ice thickness at the terminus.
    
    # Enter loop for each timestep (required to allow for snow accumulation which may alter surface type)
    for step in range(glac_bin_temp.shape[1]):
#    for step in range(0,26):
#    for step in range(0,12):
        
        # Option to adjust air temperature based on changes in surface elevation
        if input.option_adjusttemp_surfelev == 1:
            # Adjust the air temperature
            glac_bin_temp[:,step] = glac_bin_temp[:,step] + input.lr_glac * (icethickness_t0 - icethickness_adjusttemp)
            #  T_air = T+air + lr_glac * (icethickness_present - icethickness_initial)
            # Adjust refreeze as well
            #  refreeze option 2 uses annual temps, so only do this at the start of each year (step % annual_divisor)
            if (input.option_refreezing == 2) & (step % annual_divisor == 0):
                glac_bin_refreezepotential[:,step:step+annual_divisor] = massbalance.refreezepotentialbins(
                        glac_bin_temp[:,step:step+annual_divisor], dates_table.iloc[step:step+annual_divisor,:])
        # Remove input that is off-glacier (required for each timestep as glacier extent may vary over time)
        glac_bin_temp[surfacetype==0,step] = 0
        glac_bin_acc[surfacetype==0,step] = 0
        glac_bin_refreezepotential[surfacetype==0,step] = 0        
        # Compute the snow depth and melt for each bin...
        # Snow depth / 'snowpack' [m w.e.] = snow remaining + new snow
        glac_bin_snowpack[:,step] = snowpack_remaining + glac_bin_acc[:,step]
        # Available energy for melt [degC day]    
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
        # Climatic mass balance [m w.e.]
        glac_bin_massbal_clim_mwe[:,step] = glac_bin_acc[:,step] + glac_bin_refreeze[:,step] - glac_bin_melt[:,step]
        #  climatic mass balance = accumulation + refreeze - melt
        
        # Compute frontal ablation
        if main_glac_rgi.loc[glac,'TermType'] != 0:
            print('Need to code frontal ablation: includes changes to mass redistribution (uses climatic mass balance)')
            # FRONTAL ABLATION IS CALCULATED ANNUALLY IN HUSS AND HOCK (2015)
            # How should frontal ablation pair with geometry changes?
            #  - track the length of the last bin and have the calving losses control the bin length after mass 
            #    redistribution
            #  - the ice thickness will be determined by the mass redistribution
        
        # ENTER ANNUAL LOOP
        #  at the end of each year, update glacier characteristics (surface type, length, area, volume)
        if (step + 1) % annual_divisor == 0:
            # % gives the remainder; since step starts at 0, add 1 such that this switches at end of year
            # Index year
            year_index = int(step/annual_divisor)
            # for first year, need to record glacier area [km**2] and ice thickness [m ice]
            if year_index == 0:
                glac_bin_area_annual[:,year_index] = main_glac_hyps.iloc[glac,:].values.astype(float)
                glac_bin_icethickness_annual[:,year_index] = main_glac_icethickness.iloc[glac,:].values.astype(float)
            # Annual climatic mass balance [m w.e.]
            glac_bin_massbal_clim_mwe_annual[:,year_index] = (
                glac_bin_massbal_clim_mwe[:,year_index*annual_divisor:step+1].sum(1))
            #  year_index*annual_divisor is initial step of the given year; step + 1 is final step of the given year
            
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
            # Note: why would we keep the area constant?  The glaciers are changing over time, so we should allow the
            #       model to change over time as well.
            
            
            ##### GLACIER GEOMETRY CHANGE (convert to function) #####
            # Reset the annual glacier area and ice thickness
            glacier_area_t1 = np.zeros(glacier_area_t0.shape)
            icethickness_t1 = np.zeros(glacier_area_t0.shape)
            width_t1 = np.zeros(glacier_area_t0.shape)
            # Annual glacier-wide volume change [km**3]
            glacier_volumechange = ((glac_bin_massbal_clim_mwe_annual[:, year_index] / 1000 * input.density_water / 
                                     input.density_ice * glacier_area_t0).sum())
            #  units: [m w.e.] * (1 km / 1000 m) * (1000 kg / (1 m water * m**2) * (1 m ice * m**2 / 900 kg) * [km**2] 
            #         = km**3 ice         
            # If volume loss is less than the glacier volume, then redistribute mass loss/gains across the glacier;
            #  otherwise, the glacier disappears (area and thickness were already set to zero above)
            if -1 * glacier_volumechange < (icethickness_t0 / 1000 * glacier_area_t0).sum():
                # Determine where glacier exists
                glac_idx_t0 = glacier_area_t0.nonzero()[0]
                # Compute ice thickness [m ice], glacier area [km**2] and ice thickness change [m ice] after 
                #  redistribution of gains/losses
                if input.option_massredistribution == 1:
                    # Option 1: apply mass redistribution using Huss' empirical geometry change equations
                    icethickness_t1, glacier_area_t1, width_t1, icethickness_change = (
                            massbalance.massredistributionHuss(icethickness_t0, glacier_area_t0, width_t0, glac_idx_t0, 
                                                               glacier_volumechange,
                                                               glac_bin_massbal_clim_mwe_annual[:, year_index]))
                # Glacier retreat
                #  if glacier retreats (ice thickness < 0), then redistribute mass loss across the rest of the glacier
                glac_idx_t0_raw = glac_idx_t0.copy()
                if (icethickness_t1[glac_idx_t0] <= 0).any() == True:
                    # Record glacier area and ice thickness before retreat corrections applied
                    glacier_area_t0_raw = glacier_area_t0.copy()
                    icethickness_t0_raw = icethickness_t0.copy()
                    width_t0_raw = width_t0.copy()
                    #  this is only used when there are less than 3 bins
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
                    width_t0_raw[icethickness_t1 <= 0] = 0
                    glac_idx_t0_raw = glacier_area_t0_raw.nonzero()[0]
                    # Climatic mass balance for the case when there are less than 3 bins and the glacier is retreating, 
                    #  distribute the remaining glacier volume change over the entire glacier (remaining bins)
                    massbal_clim_retreat = np.zeros(glacier_area_t0_raw.shape)
                    massbal_clim_retreat[glac_idx_t0_raw] = glacier_volumechange/glacier_area_t0_raw.sum() * 1000
                    # Compute mass redistribution
                    if input.option_massredistribution == 1:
                        # Option 1: apply mass redistribution using Huss' empirical geometry change equations
                        icethickness_t1, glacier_area_t1, width_t1, icethickness_change = (
                                massbalance.massredistributionHuss(icethickness_t0_raw, glacier_area_t0_raw, 
                                                                   width_t0_raw, glac_idx_t0_raw, glacier_volumechange,
                                                                   massbal_clim_retreat))
                # Glacier advances
                #  if glacier advances (ice thickness change exceeds threshold), then redistribute mass gain in new bins
                while (icethickness_change > input.icethickness_advancethreshold).any() == True:     
                    # Record glacier area and ice thickness before advance corrections applied
                    glacier_area_t1_raw = glacier_area_t1.copy()
                    icethickness_t1_raw = icethickness_t1.copy()
                    width_t1_raw = width_t1.copy()
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
                        # Glacier width for parabola [km] (w_1 = w_0 * A_1 / A_0)
                        width_t1[glac_idx_advance] = (width_t1_raw[glac_idx_advance] * glacier_area_t1[glac_idx_advance] 
                                                      / glacier_area_t1_raw[glac_idx_advance])
                    elif input.option_glaciershape == 2:
                        # Glacier area constant for rectangle [km**2] (A_1 = A_0)
                        glacier_area_t1[glac_idx_advance] = glacier_area_t1_raw[glac_idx_advance]
                        # Glacier with constant for rectangle [km] (w_1 = w_0)
                        width_t1[glac_idx_advance] = width_t1_raw[glac_idx_advance]
                    elif input.option_glaciershape == 3:
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
                                                     glac_idx_t0.shape[0] * 100 < input.terminus_percentage])
                    # Average area of glacier terminus [km**2]
                    terminus_area_avg = glacier_area_t0[glac_idx_terminus[1]:
                                                        glac_idx_terminus[glac_idx_terminus.shape[0]-1]+1].mean()    
                    #  exclude the bin at the terminus, since this bin may need to be filled first
                    # Check if the last bin's area is below the terminus' average and fill it up if it is
                    if glacier_area_t1[glac_idx_terminus[0]] < terminus_area_avg:
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
                        glac_idx_bin2add = np.array([glac_idx_terminus[0] - 1])
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
                    massbal_clim_advance = np.zeros(glacier_area_t1.shape)
                    # Record glacier area and ice thickness before advance corrections applied
                    glacier_area_t1_raw = glacier_area_t1.copy()
                    icethickness_t1_raw = icethickness_t1.copy()
                    width_t1_raw = width_t1.copy()
                    # If a full bin has been added and volume still remains, then redistribute mass across the
                    #  glacier, thereby enabling the bins to get thicker once again prior to adding a new bin.
                    #  This is important for glacier that have very thin ice at the terminus, which would otherwise
                    #  have to keep adding a large number of bins, which would enable a thin layer of ice to advance
                    #  tremendously far down valley.
                    if advance_volume > 0:
                        if input.option_massredistribution == 1:
                            # Option 1: apply mass redistribution using Huss' empirical geometry change equations
                            icethickness_t1, glacier_area_t1, width_t1, icethickness_change = (
                                    massbalance.massredistributionHuss(icethickness_t1, glacier_area_t1, width_t1, 
                                                                       glac_idx_t0, advance_volume,
                                                                       massbal_clim_advance))
                    # update ice thickness change
                    icethickness_change = icethickness_t1 - icethickness_t1_raw

            # Note:
            # If bin retreats and then advances, the area and ice thickness pre-retreat should be used instead
            # This will also take care of the cases where you need to skip steep bins at high altitudes, i.e.,
            # discontinuous glaciers

            # Record glacier area [km**2] and ice thickness [m ice]
            glac_bin_area_annual[:,year_index + 1] = glacier_area_t1
            glac_bin_icethickness_annual[:,year_index + 1] = icethickness_t1
            # Update surface type for bins that have retreated or advanced
            surfacetype[glacier_area_t0 == 0] = 0
            surfacetype[(surfacetype == 0) & (glacier_area_t1 != 0)] = surfacetype[glacier_area_t0.nonzero()[0][0]]
            # Update glacier area and ice thickness for next year
            glacier_area_t0 = glacier_area_t1.copy()
            icethickness_t0 = icethickness_t1.copy()
            
            # NOTE: For glaciers that disappear, need to keep track of the surface type of the top bin such that the
            #       glacier has the ability to grow again...
            
            
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
        
# While in glacier loop, compile the monthly data into a netcdf

timeelapsed_step4 = timeit.default_timer() - timestart_step4
print('Step 4 time:', timeelapsed_step4, "s\n")

#%%=== STEP FIVE: DATA ANALYSIS / OUTPUT ==============================================================================
#
## Must factor in spinup years for model output, i.e., remove spinup years from the model runs

regionO1_number = input.rgi_regionsO1[0]

#output.createnetcdf(regionO1_number, main_glac_hyps, dates_table, annual_columns)

# Note: 'w' creates a new file
#       'r+' opens an existing file for reading and writing

# netcdf file path, name, and format
filename = input.netcdf_filenameprefix + str(regionO1_number) + '_' + str(strftime("%Y%m%d")) + '.nc'
fullfile = input.netcdf_filepath + filename
fileformat = 'NETCDF4_CLASSIC'
# Create the netcdf file open to write ('w') with the netCDF4 classic file format
netcdf_output = nc.Dataset(fullfile, 'w', format=fileformat)
# Create global attributes
netcdf_output.description = 'Results from glacier evolution model'
netcdf_output.history = 'Created ' + str(strftime("%Y-%m-%d %H:%M:%S"))
netcdf_output.source = 'Python Glacier Evolution Model'
# Create dimensions
glacier = netcdf_output.createDimension('glacier', None)
binelev = netcdf_output.createDimension('binelev', main_glac_hyps.shape[1])
time = netcdf_output.createDimension('time', dates_table.shape[0])
year = netcdf_output.createDimension('year', annual_columns.shape[0])
glaciertable = netcdf_output.createDimension('glaciertable', main_glac_hyps.shape[0])
# Create the variables associated with the dimensions
glaciers = netcdf_output.createVariable('glacier', np.int32, ('glacier',))
glaciers.long_name = "glacier number associated with model run"
glaciers.standard_name = "GlacNo"
glaciers.comment = ("The glacier number is defined for each model run. The user should look at the main_glac_rgi"
                       + " table to determine the RGIID or other information regarding this particular glacier.")
binelevs = netcdf_output.createVariable('binelev', np.int32, ('binelev',))
binelevs.long_name = "center bin elevation"
binelevs.standard_name = "bin_elevation"
binelevs.units = "m a.s.l."
binelevs[:] = main_glac_hyps.columns.values
binelevs.comment = ("binelev are the bin elevations that were used for the model run.")
times = netcdf_output.createVariable('time', np.float64, ('time',))
times.long_name = "date of model run"
times.standard_name = "date"
times.units = "days since 1900-01-01 00:00:00"
times.calendar = "gregorian"
times[:] = nc.date2num(dates_table['date'].astype(datetime), units = times.units, calendar = times.calendar)
years = netcdf_output.createVariable('year', np.int32, ('year',))
years.long_name = "year of model run"
years.standard_name = "year"
if input.option_wateryear == 1:
    years.units = 'water year'
elif input.option_wateryear == 0:
    years.units = 'calendar year'
years[:] = annual_columns

if input.output_package == 1:
    # Package 1 output [units: m w.e. unless otherwise specified]:
    # Monthly variables for each bin (temp, prec, acc, refreeze, snowpack, melt, meltglac, meltsnow, meltrefreeze, 
    #  frontalablation, massbal_clim)
    # Annual variables for each bin (massbal_clim, area, icethickness, width, surfacetype)
    temp_bin_monthly = netcdf_output.createVariable('temp_bin_monthly', np.float64, ('glacier', 'binelev', 'time'))
    temp_bin_monthly.standard_name = "air temperature"
    temp_bin_monthly.units = "degC"
    prec_bin_monthly = netcdf_output.createVariable('prec_bin_monthly', np.float64, ('glacier', 'binelev', 'time'))
    prec_bin_monthly.standard_name = "liquid precipitation"
    prec_bin_monthly.units = "m"
    acc_bin_monthly = netcdf_output.createVariable('acc_bin_monthly', np.float64, ('glacier', 'binelev', 'time'))
    acc_bin_monthly.standard_name = "accumulation"
    acc_bin_monthly.units = "m w.e."
    refreeze_bin_monthly = netcdf_output.createVariable('refreeze_bin_monthly', np.float64, ('glacier', 'binelev', 
                                                                                             'time'))
    refreeze_bin_monthly.standard_name = "refreezing"
    refreeze_bin_monthly.units = "m w.e."
    snowpack_bin_monthly = netcdf_output.createVariable('snowdepth_bin_monthly', np.float64, ('glacier', 'binelev', 
                                                                                              'time'))
    snowpack_bin_monthly.standard_name = "snowpack on the glacier surface"
    snowpack_bin_monthly.units = "m w.e."
    snowpack_bin_monthly.comment = ("snowpack represents the snow depth when units are m w.e.")
    melt_bin_monthly = netcdf_output.createVariable('melt_bin_monthly', np.float64, ('glacier', 'binelev', 'time'))
    melt_bin_monthly.standard_name = 'surface melt'
    melt_bin_monthly.units = "m w.e."
    melt_bin_monthly.comment = ("surface melt is the sum of melt from snow, refreeze, and the underlying glacier")
    meltglac_bin_monthly = netcdf_output.createVariable('meltglac_bin_monthly', np.float64, ('glacier', 'binelev', 
                                                                                             'time'))
    meltglac_bin_monthly.standard_name = "glacier melt"
    meltglac_bin_monthly.units = "m w.e."
    meltsnow_bin_monthly = netcdf_output.createVariable('meltsnow_bin_monthly', np.float64, ('glacier', 'binelev', 'time'))
    meltsnow_bin_monthly.standard_name = "snow melt"
    meltsnow_bin_monthly.units = "m w.e."
    meltsnow_bin_monthly.comment = ("only the melt associated with the snow on the surface and refreezing in "
                                              + "the snow regardless of whether the underlying surface type is snow or "
                                              + "not")
    # IN CURRENT FORM IS THERE SEPARATION BETWEEN SNOW MELT AND REFREEZE, WHEN REFREEZE GOES BACK INTO THE SNOW DEPTH?
    
    # FINISH ADDING THE REST, THEN PUT IN PROPER PLACE AND START GETTING THEM SET UP.
    

    
    
    
    frontal_ablation_bin_monthly = netcdf_output.createVariable('frontal_ablation_bin_monthly', np.float64, 
                                                                ('glacier', 'binelev', 'time'))
    frontal_ablation_bin_monthly.standard_name = "specific frontal ablation"
    frontal_ablation_bin_monthly.units = "m w.e."
    frontal_ablation_bin_monthly.comment = ("mass losses due to calving, subaerial frontal melting, sublimation above "
                                            + "the waterline and subaqueous frontal melting below the waterline")
    massbal_clim_mwe_bin_monthly = netcdf_output.createVariable('massbal_clim_mwe_bin_monthly', np.float64, 
                                                                ('glacier', 'binelev', 'time'))
    massbal_clim_mwe_bin_monthly.standard_name = "monthly specific climatic mass balance"
    massbal_clim_mwe_bin_monthly.units = "m w.e."
    massbal_clim_mwe_bin_monthly.comment = ("climatic mass balance is the sum of the surface mass balance and the "
                                            + "internal mass balance and accounts for the climatic mass loss over the "
                                            + "area of the entire bin") 
    area_bin_monthly = netcdf_output.createVariable('area_bin_monthly', np.float64, ('glacier', 'binelev', 'time'))
    area_bin_monthly.long_name = "monthly glacier area of each elevation bin updated annually"
    area_bin_monthly.standard_name = "area"
    area_bin_monthly.unit = "km**2"
    area_bin_monthly.comment = ("area for a given year is the area that was used for the duration of the timestep, "
                                + "i.e., it is the area at the start of the time step")
    
    
    
    
#netcdf_output.close()












