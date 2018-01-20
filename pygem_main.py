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
    dayspermonth = dates_table['daysinmonth'].values
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
    glac_bin_surfacetype_annual = np.zeros((glac_bin_temp.shape[0],annual_columns.shape[0]))
    # Local variables used within the function
    snowdepth_remaining = np.zeros(glac_bin_temp.shape[0])
    snowdepth_refreeze = np.zeros(glac_bin_temp.shape[0])
    surfacetype_ddf = np.zeros(glac_bin_temp.shape[0])
    refreeze_potential = np.zeros(glac_bin_temp.shape[0])
    
    # Enter loop for each timestep (required to allow for snow accumulation which may alter surface type)
#    for step in range(glac_bin_temp.shape[1]):
#    for step in range(0,26):
    for step in range(0,12):
        # Remove input that is off-glacier (required for each timestep as glacier extent may vary over time)
        glac_bin_temp[surfacetype==0,step] = 0
        glac_bin_acc[surfacetype==0,step] = 0
        glac_bin_refreezepotential[surfacetype==0,step] = 0
        
        # Indices for rows of glacier (compute for each step to ensure accounting for glacier geometry changes)
        gbot = np.where(surfacetype!=0)[0][0]
        gtop = np.where(surfacetype!=0)[0][-1] + 1
        #  +1 such that top row of glacier is included
        
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
        for k in surfacetype_ddf_dict: surfacetype_ddf[surfacetype == k] = surfacetype_ddf_dict[k]
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
            
            # Surface type (CONVERT TO ITS OWN FUNCTION)
            glac_bin_surfacetype_annual[:,year_index] = surfacetype
            # Compute the surface type for each bin
            #  Next year's surface type is based on the bin's average annual climatic mass balance over the last 5
            #  years.  If less than 5 years, then use the average of the existing years.
            if year_index < 5:
                # Calculate average annual climatic mass balance since run began
                massbal_clim_mwe_runningavg = glac_bin_massbal_clim_mwe_annual[:,0:year_index+1].mean(1)
#                
#
#                series_surftype_massbal_clim_mwe_runningavg = (
#                        glac_bin_massbal_clim_mwe_annual.iloc[:,0:year_position+1].mean(axis=1))
#            # Otherwise, use 5 year average annual specific climatic mass balance [m w.e.]
#            # if statement used to avoid errors with the last year, since the model run is over
#            elif glac_bin_surftype_annual.columns.values[year_position] != endyear:
#                series_surftype_massbal_clim_mwe_runningavg = (
#                    glac_bin_massbal_clim_mwe_annual.iloc[:, year_position-4:year_position+1].mean(axis=1))
#                #  "year_position - 4 : year_position + 1" provides 5 year running average with current year included
#            # If the average annual specific climatic mass balance is negative, then the surface type is ice (or debris)
#            mask_surftype_accumulationneg = (series_surftype_massbal_clim_mwe_runningavg <= 0) & (series_surftype > 0)
#            series_surftype[mask_surftype_accumulationneg] = 1
#            # If the average annual specific climatic mass balance is positive, then the surface type is snow (or firn)
#            mask_surftype_accumulationpos = (series_surftype_massbal_clim_mwe_runningavg > 0) & (series_surftype > 0)
#            series_surftype[mask_surftype_accumulationpos] = 2
#            # Apply model options for firn and debris
#            if option_surfacetype_firn == 1:
#                series_surftype[series_surftype == 2] = 3
#                #  if firn is included in the model, then change any snow surface to firn
#            if option_surfacetype_debris == 1:
#                print('Need to code the model to include debris. This option does not '
#                      'currently exist.  Please choose an option that does.\nExiting '
#                      'the model run.')
#                exit()
#            # Update the surface type for the following year
#            # if statement used to avoid error in final year, since model run is over
#            if glac_bin_surftype_annual.columns.values[year_position] != endyear:
#                glac_bin_surftype_annual.iloc[:,year_position+1] = series_surftype
            
        


timeelapsed_step4 = timeit.default_timer() - timestart_step4
print('Step 4 time:', timeelapsed_step4, "s\n")









