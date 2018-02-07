"""
fxns_massbalance.py is a list of functions that are used to compute the mass
associated with each glacier for PyGEM.
"""
#========= LIST OF PACKAGES ==================================================
import pandas as pd
import numpy as np
#========= IMPORT COMMON VARIABLES FROM MODEL INPUT ==========================
import pygem_input as input
import pygemfxns_output as output
#========= DESCRIPTION OF VARIABLES (alphabetical order) =====================
    # ablation_annual - annual ablation for each bin on a specific glacier
    # bin_ablation_mon - monthly surface ablation, which is calculated each year
    #                    according to temperature and potentially changing
    #                    surface type
    # climate_elev - table of elevation of the nearest neighbor cell for
    #                each glacier
    # climate_prec - time series of precipitation for every glacier based on
    #                user-specified option (nearest neighbor default)
    # climate_temp - time series of temperature for every glacier based on
    #                user-specified option (nearest neighbor default)
    # dates_table - table of dates (used for column headers), which includes the
    #               year, month, and number of days in each month
    # glac_count - the glacier number, which is used to keep track of the
    #              glaciers within the for loops.
    # glac_ELA - table of annual ELA for all glaciers in model run
    # glac_hyps - table of hypsometry for all the glaciers
    # glac_params - table of calibrated parameters for all the glaciers, which
    #               includes lapse rates, bias factors, etc.
    # glac_surftype - table of surface type for every bin of every glacier
    # glac_table - main table of glaciers in model run with RGI information
    # glac_temp - table of temperature data on the glacier for each bin for the
    #             entire timeseries
    # glac_temp_annual - table of annual mean temperature data on the glacier
    #                    for each bin for the entire timeseries
    # glac_precsnow - table of the total precipitation (liquid and solid) for
    #                 each bin for the entire time series
    # massbal_annual - table of annual specific mass balance for each bin for
    #                  the entire time series for a given glacier
    # option_elev_ref - reference elevation on the glacier (median is the
    #                   default)
    # option_fxn - function option (see specifics within each function)
    # refreeze_annual - table of annual refreezing for each bin of a glacier
    # snow_annual - table of annual total snow data on the glacier for each bin
    #               for the entire timeseries
    # surftype_annual - table of annual surface type on the glacier for each bin
    #                   for the entire timeseries
    # var - generic variable commonly used with simple tasks, e.g., annual mean
    #       or annual sum of a variable
    # year - the year associated with the for loop

    # Commonly used variables within the functions:
    # mask - "masks" refer to a specific condition being met and are used in
    #        functions to apply logical indexing, which speeds up the
    #        computational time. These masks also make it easier to read the
    #        functions.

#========= FUNCTIONS (alphabetical order) ===================================
#    modelparameters = [lr_gcm, lr_glac, prec_factor, prec_grad, ddf_snow, ddf_ice, temp_snow]
#    # Note: use constraints equal to a specific input, if the variable is not being calibrated
#    
#    
#    glacier_rgi_table = main_glac_rgi.loc[glac, :]
#    glacier_gcm_elev = main_glac_gcmelev.iloc[glac]
#    glacier_gcm_prec = main_glac_gcmprec.iloc[glac,:].values
#    glacier_area_t0 = main_glac_hyps.iloc[glac,:].values.astype(float)
#    glac_idx_initial = glacier_area_t0.nonzero()[0]    
#    # Inclusion of ice thickness and width, i.e., loading the values may be only required for Huss mass redistribution!
#    icethickness_t0 = main_glac_icethickness.iloc[glac,:].values.astype(float)
#    width_t0 = main_glac_width.iloc[glac,:].values.astype(float)
    
def runmassbalance(glac, modelparameters, regionO1_number, glacier_rgi_table, glacier_area_t0, icethickness_t0, 
                   width_t0, glacier_gcm_temp, glacier_gcm_prec, glacier_gcm_elev, elev_bins, dates_table, 
                   annual_columns, annual_divisor):
    # Variables to export
    glac_bin_refreeze = np.zeros((elev_bins.shape[0],glacier_gcm_temp.shape[0]))
    glac_bin_melt = np.zeros((elev_bins.shape[0],glacier_gcm_temp.shape[0]))
    glac_bin_meltsnow = np.zeros((elev_bins.shape[0],glacier_gcm_temp.shape[0]))
    glac_bin_meltrefreeze = np.zeros((elev_bins.shape[0],glacier_gcm_temp.shape[0]))
    glac_bin_meltglac = np.zeros((elev_bins.shape[0],glacier_gcm_temp.shape[0]))
    glac_bin_frontalablation = np.zeros((elev_bins.shape[0],glacier_gcm_temp.shape[0]))
    glac_bin_snowpack = np.zeros((elev_bins.shape[0],glacier_gcm_temp.shape[0]))
    glac_bin_massbalclim = np.zeros((elev_bins.shape[0],glacier_gcm_temp.shape[0]))
    glac_bin_massbalclim_annual = np.zeros((elev_bins.shape[0],annual_columns.shape[0]))
    glac_bin_surfacetype_annual = np.zeros((elev_bins.shape[0],annual_columns.shape[0]))
    glac_bin_icethickness_annual = np.zeros((elev_bins.shape[0], annual_columns.shape[0] + 1))
    glac_bin_area_annual = np.zeros((elev_bins.shape[0], annual_columns.shape[0] + 1))
    glac_bin_width_annual = np.zeros((elev_bins.shape[0], annual_columns.shape[0] + 1))
    # Local variables
    glac_idx_initial = glacier_area_t0.nonzero()[0] 
    snowpack_remaining = np.zeros(elev_bins.shape[0])
    dayspermonth = dates_table['daysinmonth'].values
    surfacetype_ddf = np.zeros(elev_bins.shape[0])
    refreeze_potential = np.zeros(elev_bins.shape[0])
    if input.option_adjusttemp_surfelev == 1:
        # ice thickness initial is used to adjust temps to changes in surface elevation
        icethickness_adjusttemp = icethickness_t0.copy()
        icethickness_adjusttemp[0:icethickness_adjusttemp.nonzero()[0][0]] = (
                icethickness_adjusttemp[icethickness_adjusttemp.nonzero()[0][0]])
        #  bins that advance need to have an initial ice thickness; otherwise, the temp adjustment will be based on ice
        #  thickness - 0, which is wrong  Since advancing bins take the thickness of the previous bin, set the initial 
        #  ice thickness of all bins below the terminus to the ice thickness at the terminus.
    # Downscale the gcm temperature [degC] to each bin
    glac_bin_temp = downscaletemp2bins(glacier_rgi_table, glacier_gcm_temp, glacier_gcm_elev, elev_bins,
                                       modelparameters)
    # Downscale the gcm precipitation [m] to each bin (includes solid and liquid precipitation)
    glac_bin_precsnow = downscaleprec2bins(glacier_rgi_table, glacier_gcm_prec, glacier_gcm_elev, elev_bins, 
                                           modelparameters)
    # Compute accumulation [m w.e.] and precipitation [m] for each bin
    glac_bin_prec, glac_bin_acc = accumulationbins(glac_bin_temp, glac_bin_precsnow, modelparameters)
    # Compute potential refreeze [m w.e.] for each bin
    glac_bin_refreezepotential = refreezepotentialbins(glac_bin_temp, dates_table)
    # Compute the initial surface type [0=off-glacier, 1=ice, 2=snow, 3=firn, 4=debris]
    surfacetype, firnline_idx = surfacetypebinsinitial(glacier_area_t0, glacier_rgi_table, elev_bins)
    # Create surface type DDF dictionary (manipulate this function for calibration or for each glacier)
    surfacetype_ddf_dict = surfacetypeDDFdict(modelparameters)
    # Enter loop for each timestep (required to allow for snow accumulation which may alter surface type)
    for step in range(glac_bin_temp.shape[1]):
#    for step in range(0,26):
#    for step in range(0,12):
        # Option to adjust air temperature based on changes in surface elevation
        if input.option_adjusttemp_surfelev == 1:
            # Adjust the air temperature
            glac_bin_temp[:,step] = (glac_bin_temp[:,step] + modelparameters[1] * (icethickness_t0 - 
                                     icethickness_adjusttemp))
            #  T_air = T+air + lr_glac * (icethickness_present - icethickness_initial)
            # Adjust refreeze as well
            #  refreeze option 2 uses annual temps, so only do this at the start of each year (step % annual_divisor)
            if (input.option_refreezing == 2) & (step % annual_divisor == 0):
                glac_bin_refreezepotential[:,step:step+annual_divisor] = refreezepotentialbins(
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
        # Compute the refreeze, refreeze melt, and any changes to the snow depth
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
        snowpack_remaining[abs(snowpack_remaining) < input.tolerance] = 0
        # Compute any remaining melt and any additional refreeze in the accumulation zone
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
        glac_bin_refreeze[elev_bins >= elev_bins[firnline_idx], step] = (
                glac_bin_refreeze[elev_bins >= elev_bins[firnline_idx], step] +
                glac_bin_melt[elev_bins >= elev_bins[firnline_idx], step])
        # refreeze cannot exceed refreeze potential
        glac_bin_refreeze[glac_bin_refreeze[:,step] > refreeze_potential, step] = (
                refreeze_potential[glac_bin_refreeze[:,step] > refreeze_potential])
        # update refreeze potential
        refreeze_potential = refreeze_potential - glac_bin_refreeze[:,step]
        refreeze_potential[abs(refreeze_potential) < input.tolerance] = 0
        # Total melt (snow + refreeze + glacier)
        glac_bin_melt[:,step] = glac_bin_meltglac[:,step] + glac_bin_meltrefreeze[:,step] + glac_bin_meltsnow[:,step]
        # Climatic mass balance [m w.e.]
        glac_bin_massbalclim[:,step] = glac_bin_acc[:,step] + glac_bin_refreeze[:,step] - glac_bin_melt[:,step]
        #  climatic mass balance = accumulation + refreeze - melt
        # Compute frontal ablation
        if glacier_rgi_table.loc['TermType'] != 0:
            print('Need to code frontal ablation: includes changes to mass redistribution (uses climatic mass balance)')
            # FRONTAL ABLATION IS CALCULATED ANNUALLY IN HUSS AND HOCK (2015)
            # How should frontal ablation pair with geometry changes?
            #  - track the length of the last bin and have the calving losses control the bin length after mass 
            #    redistribution
            #  - the ice thickness will be determined by the mass redistribution
            # Note: output functions calculate total mass balance assuming frontal ablation is a positive value that is 
            #       then subtracted from the climatic mass balance.
        # ENTER ANNUAL LOOP
        #  at the end of each year, update glacier characteristics (surface type, length, area, volume)
        if (step + 1) % annual_divisor == 0:
            # % gives the remainder; since step starts at 0, add 1 such that this switches at end of year
            # Index year
            year_index = int(step/annual_divisor)
            # for first year, need to record glacier area [km**2] and ice thickness [m ice]
            if year_index == 0:
                glac_bin_area_annual[:,year_index] = glacier_area_t0
                glac_bin_icethickness_annual[:,year_index] = icethickness_t0
                glac_bin_width_annual[:,year_index] = width_t0
            # Annual climatic mass balance [m w.e.]
            glac_bin_massbalclim_annual[:,year_index] = glac_bin_massbalclim[:,year_index*annual_divisor:step+1].sum(1)
            #  year_index*annual_divisor is initial step of the given year; step + 1 is final step of the given year
            # Annual surface type [-]
            glac_bin_surfacetype_annual[:,year_index] = surfacetype
            # Compute the surface type for each bin
            surfacetype, firnline_idx = surfacetypebinsannual(surfacetype, glac_bin_massbalclim_annual, year_index)
            # Glacier geometry change is dependent on whether model is being calibrated (option_calibration = 1) or not
            if input.option_calibration == 0:
                # Mass redistribution according to Huss empirical curves
                glacier_area_t1, icethickness_t1, width_t1 = massredistributionHuss(glacier_area_t0, icethickness_t0, 
                        width_t0, glac_bin_massbalclim_annual, year_index, glac_idx_initial)
                # Update surface type for bins that have retreated or advanced
                surfacetype[glacier_area_t1 == 0] = 0
                surfacetype[(surfacetype == 0) & (glacier_area_t1 != 0)] = surfacetype[glacier_area_t0.nonzero()[0][0]]
                # Record and update ice thickness and glacier area for next year
            else:
                glacier_area_t1 = glacier_area_t0
                icethickness_t1 = icethickness_t0
                width_t1 = width_t0
            if year_index < input.spinupyears:
                # For spinup years, glacier area and volume are constant
                glac_bin_icethickness_annual[:,year_index + 1] = icethickness_t0
                glac_bin_area_annual[:,year_index + 1] = glacier_area_t0
                glac_bin_width_annual[:,year_index + 1] = width_t0
            else:
                # Record ice thickness [m ice] and glacier area [km**2]
                glac_bin_icethickness_annual[:,year_index + 1] = icethickness_t1
                glac_bin_area_annual[:,year_index + 1] = glacier_area_t1
                glac_bin_width_annual[:,year_index + 1] = width_t1
                # Update glacier area [km**2] and ice thickness [m ice]
                icethickness_t0 = icethickness_t1.copy()
                glacier_area_t0 = glacier_area_t1.copy()
                width_t0 = width_t1.copy()  
    # Remove the spinup years of the variables that are being exported
    if input.timestep == 'monthly':
        colstart = input.spinupyears * annual_divisor
        colend = glacier_gcm_temp.shape[0] + 1
    glac_bin_temp = glac_bin_temp[:,colstart:colend]
    glac_bin_prec = glac_bin_prec[:,colstart:colend]
    glac_bin_acc = glac_bin_acc[:,colstart:colend]
    glac_bin_refreeze = glac_bin_refreeze[:,colstart:colend]
    glac_bin_snowpack = glac_bin_snowpack[:,colstart:colend]
    glac_bin_melt = glac_bin_melt[:,colstart:colend]
    glac_bin_frontalablation = glac_bin_frontalablation[:,colstart:colend]
    glac_bin_massbalclim = glac_bin_massbalclim[:,colstart:colend]
    glac_bin_massbalclim_annual = glac_bin_massbalclim_annual[:,input.spinupyears:annual_columns.shape[0]+1]
    glac_bin_area_annual = glac_bin_area_annual[:,input.spinupyears:annual_columns.shape[0]+1]
    glac_bin_icethickness_annual = glac_bin_icethickness_annual[:,input.spinupyears:annual_columns.shape[0]+1]
    glac_bin_width_annual = glac_bin_width_annual[:,input.spinupyears:annual_columns.shape[0]+1]
    glac_bin_surfacetype_annual = glac_bin_surfacetype_annual[:,input.spinupyears:annual_columns.shape[0]+1]
#    # Update annual_columns last; otherwise, it will influence the indexing above removing spinup years
#    annual_columns = annual_columns[input.spinupyears:annual_columns.shape[0]+1]
#    dates_table = dates_table.iloc[colstart:colend,:]
    # Return the desired output
#    return (glac_bin_temp, glac_bin_prec, glac_bin_acc, glac_bin_refreeze, glac_bin_snowpack, glac_bin_melt, 
#            glac_bin_frontalablation, glac_bin_massbalclim, glac_bin_massbalclim_annual, glac_bin_area_annual, 
#            glac_bin_icethickness_annual, glac_bin_width_annual, glac_bin_surfacetype_annual, annual_columns, 
#            dates_table)
    return (glac_bin_temp, glac_bin_prec, glac_bin_acc, glac_bin_refreeze, glac_bin_snowpack, glac_bin_melt, 
            glac_bin_frontalablation, glac_bin_massbalclim, glac_bin_massbalclim_annual, glac_bin_area_annual, 
            glac_bin_icethickness_annual, glac_bin_width_annual, glac_bin_surfacetype_annual)


#=======================================================================================================================
def accumulationbins(glac_temp, glac_precsnow, modelparameters):
    # Note: this will only work for monthly time step!
    """
    Calculate the accumulation and precipitation for every elevation bin on the glacier.
    Output: numpy array of precipitation [m] and snow [m w.e.] (rows = bins, columns = dates)
    """
    # Surface Ablation Options:
    #   > 1 (default) - single threshold (temp below snow, above rain)
    #   > 2 - linear relationship (fraction of snow prec within +/- 1 deg)
    #         ex. if temp = threshold, then 50% snow and rain
    #    
    bin_prec = np.zeros(glac_precsnow.shape)
    bin_snow = np.zeros(glac_precsnow.shape)
    if input.option_accumulation == 1:
        # If temperature above threshold, then rain
        bin_prec[glac_temp > modelparameters[6]] = glac_precsnow[glac_temp > modelparameters[6]]
        # If temperature below threshold, then snow
        bin_snow[glac_temp <= modelparameters[6]] = glac_precsnow[glac_temp <= modelparameters[6]]
    elif input.option_accumulation == 2:
        # If temperature above maximum threshold, then all rain
        bin_prec[glac_temp >= modelparameters[6] + 1] = glac_precsnow[glac_temp >= modelparameters[6] + 1]
        # If temperature below minimum threshold, then all snow
        bin_snow[glac_temp <= modelparameters[6] - 1] = glac_precsnow[glac_temp <= modelparameters[6] - 1]
        # Otherwise temperature between min/max, then mix of snow/rain using linear relationship between min/max
        bin_prec[(glac_temp < modelparameters[6] + 1) & (glac_temp > modelparameters[6] - 1)] = ((1/2 + (
            glac_temp[(glac_temp < modelparameters[6] + 1) & (glac_temp > modelparameters[6] - 1)] - modelparameters[6])
            / 2) * glac_precsnow[(glac_temp < modelparameters[6] + 1) & (glac_temp > modelparameters[6] - 1)])
        bin_snow[(glac_temp < modelparameters[6] + 1) & (glac_temp > modelparameters[6] - 1)] = ((1 - (1/2 + (
            glac_temp[(glac_temp < modelparameters[6] + 1) & (glac_temp > modelparameters[6] - 1)] - modelparameters[6]) 
            / 2)) * glac_precsnow[(glac_temp < modelparameters[6] + 1) & (glac_temp > modelparameters[6] - 1)])
    else:
        print("This option for 'option_accumulation' does not exist.  Please choose an option that exists."
              "Exiting model run.\n")
    return bin_prec, bin_snow


def annualweightedmean_array(var, dates_table):
    """
    Calculate annual mean of variable according to the timestep.
    Monthly timestep will group every 12 months, so starting month is important.
    """
    if input.timestep == 'monthly':
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
    elif input.timestep == 'daily':
        print('\nError: need to code the groupbyyearsum and groupbyyearmean for daily timestep.'
              'Exiting the model run.\n')
        exit()
    return var_annual


def downscaleprec2bins(glacier_table, gcm_prec, gcm_elev, elev_bins, modelparameters):
    """
    Downscale the global climate model precipitation data to each bin on the glacier using the precipitation bias factor
    (prec_factor) and the glacier precipitation gradient (prec_grad).
    Output: numpy array of precipitation [m] in each bin (rows = bins, columns = dates)
    """
    # Function Options:
    #   > 1 (default) - precip factor bias to correct GCM and a precipitation gradient to adjust precip over the glacier
    #   > 2 (not coded yet) - Huss and Hock (2015), exponential limits, etc.
    if input.option_prec2bins == 1:
        # Option 1 is the default and uses a precipitation factor and precipitation gradient over the glacier.
        bin_prec = (gcm_prec * modelparameters[2] * (1 + modelparameters[3] * (elev_bins - 
                    glacier_table.loc[input.option_elev_ref_downscale]))[:,np.newaxis])
        #   P_bin = P_gcm * prec_factor * (1 + prec_grad * (z_bin - z_ref))
    else:
        print("\nThis option for 'downscaleprec2bins' has not been coded yet. Please choose an existing option."
              "Exiting model run.\n")
        exit()
    return bin_prec


def downscaletemp2bins(glacier_table, gcm_temp, gcm_elev, elev_bins, modelparameters):
    """
    Downscale the global climate model temperature data to each bin on the glacier using the global climate model 
    lapse rate (lr_gcm) and the glacier lapse rate (lr_glac).
    Output: numpy array of temperature [degC] in each bin (rows = bins, columns = dates)
    """
    # Function Options:
    #   > 1 (default) - lapse rate for gcm and glacier
    #   > no other options currently exist
    if input.option_temp2bins == 1:
        # Option 1 is the default and uses a lapse rate for the gcm and a glacier lapse rate.
        bin_temp = (gcm_temp + (modelparameters[0] * (glacier_table.loc[input.option_elev_ref_downscale] - gcm_elev) + 
                    modelparameters[1] * (elev_bins - glacier_table.loc[input.option_elev_ref_downscale]))[:,np.newaxis]
                    )
        #  T_bin = T_gcm + lr_gcm * (z_ref - z_gcm) + lr_glac * (z_bin - z_ref)
        #  Explanation: A + B[:,np.newaxis] adds two one-dimensional matrices together such that the column values of
        #         matrix A is added to all the rows of the matrix B.  This enables all the calculations to be performed
        #         on a single line as opposed to looping through each row of the matrix
        #         ex. A is a 180x1 matrix and B a is 900x1 matrix, then this returns a 900x180 matrix
    else:
        print("\nThis option for 'downscaletemp2bins' has not been coded yet. Please choose an existing option. "
              "Exiting model run.\n")
        exit()
    return bin_temp
   

def massredistributionHuss(glacier_area_t0, icethickness_t0, width_t0, glac_bin_massbalclim_annual, year_index, 
                           glac_idx_initial):
    # Reset the annual glacier area and ice thickness
    glacier_area_t1 = np.zeros(glacier_area_t0.shape)
    icethickness_t1 = np.zeros(glacier_area_t0.shape)
    width_t1 = np.zeros(glacier_area_t0.shape)
    # Annual glacier-wide volume change [km**3]
    glacier_volumechange = ((glac_bin_massbalclim_annual[:, year_index] / 1000 * input.density_water / 
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
            icethickness_t1, glacier_area_t1, width_t1, icethickness_change = massredistributioncurveHuss(
                    icethickness_t0, glacier_area_t0, width_t0, glac_idx_t0, glacier_volumechange,
                    glac_bin_massbalclim_annual[:, year_index])
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
                icethickness_t1, glacier_area_t1, width_t1, icethickness_change = massredistributioncurveHuss(
                        icethickness_t0_raw, glacier_area_t0_raw, width_t0_raw, glac_idx_t0_raw, glacier_volumechange,
                        massbal_clim_retreat)
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
                if input.option_massredistribution == 1:
                    # Option 1: apply mass redistribution using Huss' empirical geometry change equations
                    icethickness_t1, glacier_area_t1, width_t1, icethickness_change = massredistributioncurveHuss(
                            icethickness_t1, glacier_area_t1, width_t1, glac_idx_t0, advance_volume,
                            massbal_clim_advance)
            # update ice thickness change
            icethickness_change = icethickness_t1 - icethickness_t1_raw
    return glacier_area_t1, icethickness_t1, width_t1


def massredistributioncurveHuss(icethickness_t0, glacier_area_t0, width_t0, glac_idx_t0, glacier_volumechange, 
                           massbalclim_annual):
    """ 
    Compute the mass redistribution, otherwise known as glacier geometry changes, based on the glacier volume change
    Function Options:
        > 1 (default) - Huss and Hock (2015); volume gain/loss redistributed over the glacier using empirical normalized
                        ice thickness change curves
        > 2 (Need to code) - volume-length scaling
        > 3 (Need to code) - volume-area scaling
        > 4 (Need to code) - what previous models have done
        > 5 (Need to code) - ice dynamics, simple flow model
        > 6 - no glacier dynamics
    Input:
        > icethickness_t0 - single column array of ice thickness for every bin at the start of the time step
        > glacier_area_t0 - single column array of glacier area for every bin at the start of the time step
        > glac_idx_t0 - single column array of the bin index that is part of the glacier
        > glacier_volumechange - value of glacier-wide volume change [km**3] based on the annual climatic mass balance
        > glac_bin_clim_mwe_annual - single column array of annual climatic mass balance for every bin
    """
    # Apply Huss redistribution if there are at least 3 elevation bands; otherwise, use the mass balance
    # reset variables
    icethickness_t1 = np.zeros(glacier_area_t0.shape)
    glacier_area_t1 = np.zeros(glacier_area_t0.shape)
    width_t1 = np.zeros(glacier_area_t0.shape) 
    if glac_idx_t0.shape[0] > 3:
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
        bin_volumechange = massbalclim_annual / 1000 * glacier_area_t0
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
        # Glacier width for parabola [km]
        #  w_1 = w_0 * (A_1 / A_0)
        width_t1[glac_idx_t0] = width_t0[glac_idx_t0] * glacier_area_t1[glac_idx_t0] / glacier_area_t0[glac_idx_t0]
    elif input.option_glaciershape == 2:
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
        # Glacier width for triangle [km]
        #  w_1 = w_0 * (A_1 / A_0)
        width_t1[glac_idx_t0] = width_t0[glac_idx_t0] * glacier_area_t1[glac_idx_t0] / glacier_area_t0[glac_idx_t0]
    # Ice thickness change [m ice]
    icethickness_change = icethickness_t1 - icethickness_t0
    # return the ice thickness [m ice] and ice thickness change [m ice]
    return icethickness_t1, glacier_area_t1, width_t1, icethickness_change


def refreezepotentialbins(glac_temp, dates_table):
    # Note: this will only work for monthly time step!
    """
    Calculate the refreezing for every elevation bin on the glacier.
    """
    # Refreezing Options:
    #   > 1 (default) - monthly refreezing solved by modeling snow and firm temperatures using heat conduction equations 
    #                   according to Huss and Hock (2015)
    #   > 2 - annual refreezing based on mean air temperature according to Woodward et al. (1997)
    #
    # 1/19/18 - switched to refreeze potential, which is given annually
    bin_refreezepotential = np.zeros(glac_temp.shape)
    if input.option_refreezing == 1:
        print('This option based on Huss and Hock (2015) is intended to be the '
              'default; however, it has not been coded yet due to its '
              'complexity.  For the time being, please choose an option that '
              'exists.\n\nExiting model run.\n\n')
        exit()
    elif input.option_refreezing == 2:
        # Compute annual mean temperature
        glac_temp_annual = annualweightedmean_array(glac_temp, dates_table)
        # Compute bin refreeze potential according to Woodward et al. (1997)
        bin_refreezepotential_annual = (-0.69 * glac_temp_annual + 0.0096) * 1/100
        #   R(m) = -0.69 * Tair + 0.0096 * (1 m / 100 cm)
        #   Note: conversion from cm to m is included
        # Remove negative refreezing values
        bin_refreezepotential_annual[bin_refreezepotential_annual < 0] = 0
        # Place annual refreezing in January for accounting and melt purposes
        if input.timestep == 'monthly':
            # try/except used to avoid errors with option to adjust air temperature based on surface elevation, i.e., 
            #  try works for the regular function
            #  except works when subsets of the glac_temp and dates_table are given
            try:
                placeholder = (12 - dates_table.loc[0,'month'] + input.refreeze_month) % 12
            except:
                placeholder = (12 - dates_table.iloc[0,2] + input.refreeze_month) % 12
            #  using the month of the first timestep and the refreeze month add the annual values to the monthly data
            bin_refreezepotential[:,placeholder::12] = bin_refreezepotential_annual[:,::1]
#            for step in range(glac_temp.shape[1]):
#                if dates_table.loc[step, 'month'] == input.refreeze_month:
#                    bin_refreeze[:,step] = bin_refreeze_annual[:,int(step/12)]
#                    #  int() truncates the value, so int(step/12) selects the position of the corresponding year
        elif input.timestep == 'daily':
            print("MODEL ERROR: daily time step not coded for Woodward et al. "
                  "(1997) refreeze function yet.\n\nExiting model run.\n\n")
            exit()
        else:
            print("MODEL ERROR: please select 'daily' or 'monthly' as the time "
                  "step for the model run.\n\nExiting model run.\n\n")
            exit()
    else:
        print("This option for 'refreezingbins' does not exist.  Please choose "
              "an option that exists. Exiting model run.\n")
        exit()
    return bin_refreezepotential
    

def surfacetypebinsannual(surfacetype, glac_bin_massbalclim_annual, year_index):
    """
    Update surface type according to climatic mass balance over the last five years.  If positive, then snow/firn.  If 
    negative, then ice/debris.
    Convention: 0 = off-glacier, 1 = ice, 2 = snow, 3 = firn, 4 = debris
    """
    # Function Options:
    #   > 1 (default) - update surface type according to Huss and Hock (2015)
    #   > 2 - Radic and Hock (2011)
    # Huss and Hock (2015): Initially, above median glacier elevation is firn and below is ice. Surface type updated for
    #   each elevation band and month depending on the specific mass balance.  If the cumulative balance since the start 
    #   of the mass balance year is positive, then snow is assigned. If the cumulative mass balance is negative (i.e., 
    #   all snow of current mass balance year has melted), then bare ice or firn is exposed. Surface type is assumed to 
    #   be firn if the elevation band's average annual balance over the preceding 5 years (B_t-5_avg) is positive. If
    #   B_t-5_avg is negative, surface type is ice.
    #       > climatic mass balance calculated at each bin and used with the mass balance over the last 5 years to 
    #         determine whether the surface is firn or ice.  Snow is separate based on each month.
    # Radic and Hock (2011): "DDF_snow is used above the ELA regardless of snow cover.  Below the ELA, use DDF_ice is 
    #   used only when snow cover is 0.  ELA is calculated from the observed annual mass balance profiles averaged over 
    #   the observational period and is kept constant in time for the calibration period.  For the future projections, 
    #   ELA is set to the mean glacier height and is time dependent since glacier volume, area, and length are time 
    #   dependent (volume-area-length scaling).
    #   Bliss et al. (2014) uses the same as Valentina's model
    #
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
    if input.option_surfacetype_firn == 1:
        surfacetype[surfacetype == 2] = 3
    if input.option_surfacetype_debris == 1:
        print('Need to code the model to include debris.  Please choose an option that currently exists.\n'
              'Exiting the model run.')
        exit()
    return surfacetype, firnline_idx


def surfacetypebinsinitial(glacier_area, glacier_table, elev_bins):
    """
    Define initial surface type according to median elevation such that the melt can be calculated over snow or ice.
    Convention: (0 = off-glacier, 1 = ice, 2 = snow, 3 = firn, 4 = debris)
    Function Options:
    - option_surfacetype_initial
        > 1 (default) - use median elevation to classify snow/firn above the median and ice below
        > 2 (Need to code) - use mean elevation instead
        > 3 (Need to code) - specify an AAR ratio and apply this to estimate initial conditions
    - option_surfacetype_firn = 1
        > 1 (default) - firn is included
        > 0 - firn is not included
    - option_surfacetype_debris = 0
        > 0 (default) - debris cover is not included
        > 1 - debris cover is included
    Output: Pandas DataFrame of the initial surface type for each glacier in the model run
    (rows = GlacNo, columns = elevation bins)
    """
    surfacetype = np.zeros(glacier_area.shape)
    # Option 1 - initial surface type based on the median elevation
    if input.option_surfacetype_initial == 1:
        surfacetype[(elev_bins < glacier_table.loc['Zmed']) & (glacier_area > 0)] = 1
        surfacetype[(elev_bins >= glacier_table.loc['Zmed']) & (glacier_area > 0)] = 2
    # Option 2 - initial surface type based on the mean elevation
    elif input.option_surfacetype_initial ==2:
        surfacetype[(elev_bins < glacier_table['Zmean']) & (glacier_area > 0)] = 1
        surfacetype[(elev_bins >= glacier_table['Zmean']) & (glacier_area > 0)] = 2
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
    if input.option_surfacetype_firn == 1:
        surfacetype[surfacetype == 2] = 3
        #  everything initially considered snow is considered firn, i.e., the model initially assumes there is no snow 
        #  on the surface anywhere.
    if input.option_surfacetype_debris == 1:
        print("Need to code the model to include debris. This option does not currently exist.  Please choose an option"
              + " that exists.\nExiting the model run.")
        exit()
        # One way to include debris would be to simply have debris cover maps and state that the debris retards melting 
        # as a fraction of melt.  It could also be DDF_debris as an additional calibration tool. Lastly, if debris 
        # thickness maps are generated, could be an exponential function with the DDF_ice as a term that way DDF_debris 
        # could capture the spatial variations in debris thickness that the maps supply.
    return surfacetype, firnline_idx


def surfacetypeDDFdict(modelparameters):
    """
    Create a dictionary of surface type and its respective DDF
    Convention: [0=off-glacier, 1=ice, 2=snow, 3=firn, 4=debris]
    modelparameters[lr_gcm, lr_glac, prec_factor, prec_grad, DDF_snow, DDF_ice, T_snow]
    """
    surfacetype_ddf_dict = {
            1: modelparameters[5],
            2: modelparameters[4]}
    if input.option_surfacetype_firn == 1:
        if input.option_DDF_firn == 0:
            surfacetype_ddf_dict[3] = modelparameters[4]
        elif input.option_DDF_firn == 1:
            surfacetype_ddf_dict[3] = np.mean([modelparameters[4],modelparameters[5]])
    if input.option_surfacetype_debris == 1:
        surfacetype_ddf_dict[4] = input.DDF_debris
    return surfacetype_ddf_dict


#========= OLDER SCRIPTS PRE-JANUARY 11, 2018 =========================================================================
#def AAR_glacier(ELA_value, series_area, GlacNo):
#    """
#    Compute the Accumulation-Area Ratio (AAR) for a given glacier based on its ELA
#    """
#    try:
#        AAR_output = (1 - (np.cumsum(series_area)).divide(series_area.sum())
#            .iloc[int(ELA_value / input.binsize) - 1]) * 100
#        #  ELA_value is the elevation associated with the ELA, so dividing this by the binsize returns the column position 
#        #    if the indexing started at 1, the "-1" accounts for the fact that python starts its indexing at 0, so
#        #    ".iloc[int(ELA_value / binsize) - 1]" gives the column of the ELA.
#        #  np.cumsum gives the cumulative sum of the glacier area for the given year
#        #    this is divided by the total area to get the cumulative fraction of glacier area.
#        #  The column position is then used to select the cumulative fraction of glacier area of the ELA
#        #    since this is the area below the ELA, the value is currently the ablation area as a decimal;
#        #    therefore, "1 - (cumulative_fraction)" gives the fraction of the ablation area,
#        #    and multiplying this by 100 gives the fraction as a percentage.
#    except:
#        # if ELA does not exist, then set AAR = -9.99
#        AAR_output = -9.99
#    return AAR_output    
#
#
#def ELA_glacier(series_massbal_spec, ELA_past):
#    """
#    Compute the Equlibrium Line Altitude (ELA) from a series of specific mass balance, i.e., a single column of the 
#    specific mass balance for each elevation bin
#    """
#    # Use numpy's sign function to return an array of the sign of the values (1=positive, -1=negative, 0=zero)
#    series_ELA_sign = np.sign(series_massbal_spec)                
#    # Use numpy's where function to determine where the specific mass balance changes from negative to positive
#    series_ELA_signchange = np.where((np.roll(series_ELA_sign,1) - series_ELA_sign) == -2)
#    #   roll is a numpy function that performs a circular shift, so in this case all the values are shifted up one 
#    #   place. Since we are looking for the change from negative to positive, i.e., a value of -1 to +1, we want to 
#    #   find where the value equals -2. numpy's where function is used to find this value of -2.  The ELA will be 
#    #   the mean elevation between this bin and the bin below it.
#    #   Example: bin 4665 m has a negative mass balance and 4675 m has a positive mass balance. The difference with 
#    #            the roll function will give 4675 m a value of -2.  Therefore, the ELA will be 4670 m.
#    #   Note: If there is a bin with no glacier area between the min and max height of the glacier (ex. a very steep 
#    #     section), then this will not be captured.  This only becomes a problem if this bin is technically the ELA, 
#    #     i.e., above it is a positive mass balance, and below it is a negative mass balance.  Using np.roll with a
#    #     larger shift would be one way to work around this issue.
#    # try and except to avoid errors associated with the entire glacier having a positive or negative mass balance
#    try:
#        ELA_output = (series_massbal_spec.index.values[series_ELA_signchange[0]][0] - input.binsize/2).astype(int)
#        #  series_ELA_signchange[0] returns the position of the ELA. series_massbal_annual.index returns an array 
#        #  with one value, so the [0] ais used to accesses the element in that array. The binsize is then used to 
#        #  determine the median elevation between those two bins.
#    except:
#        # This may not work in three cases:
#        #   > The mass balance of the entire glacier is completely positive or negative.
#        #   > The mass balance of the whole glacier is 0 (no accumulation or ablation, i.e., snow=0, temp<0)
#        #   > The ELA falls on a band that does not have any glacier (ex. a very steep section) causing the sign 
#        #     roll method to fail. In this case, using a large shift may solve the issue.
#        try:
#            # if entire glacier is positive, then set to the glacier's minimum
#            if series_ELA_sign.iloc[np.where(series_ELA_sign != 0)[0][0]] == 1:
#                ELA_output = series_ELA_sign.index.values[np.where(series_ELA_sign != 0)[0][0]] - input.binsize/2
#            # if entire glacier is negative, then set to the glacier's maximum
#            elif series_ELA_sign.iloc[np.where((series_ELA_sign != 0))[0][0]] == -1:
#                ELA_output = series_ELA_sign.index.values[np.where(series_ELA_sign != 0)[0]
#                             [np.where(series_ELA_sign != 0)[0].shape[0]-1]] + input.binsize/2
#        except:
#            # if the specific mass balance over the entire glacier is 0, i.e., no ablation or accumulation,
#            #  then the ELA is the same as the previous timestep
#            ELA_output = ELA_past
#    return ELA_output