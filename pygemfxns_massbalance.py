"""
fxns_massbalance.py is a list of functions that are used to compute the mass
associated with each glacier for PyGEM.
"""
#========= LIST OF PACKAGES ==================================================
import pandas as pd
import numpy as np
#========= IMPORT COMMON VARIABLES FROM MODEL INPUT ==========================
from pygem_input import *
    # import all data

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
def ablationsurfacebinsmonthly_V2(option_fxn, bin_ablation_mon, glac_temp,
                                 glac_surftype, glac_params, dates_table,
                                 glac_count, year):
    # Note: this will only work for monthly time step!
    """
    Calculate the surface ablation of every elevation bin for the glacier.
    Convention: negative ablation indicates surface lowering.
    """
    # Need to alter the format of the yearly loops
        # Are we using calendar year or water year?
        #   > Provide options to do both.  This should be relatively simple by
        #     applying the "roll"(?) function that was used recently in another
        #     function.
    # Any update to underlying surface type should not impact accumulation rates
    #   or the amount of accumulation that is carried over, i.e., accumulation
    #   needs to be carried over each month regardless of surface type
    # What we really need to do is have a loop at the end that checks if the
    #   last month of the year is reached.  If it is December, then update
    #   surface type and compute refreezing accordingly.  This loop will be
    #   embedded within the monthly for loop.
    # Steps should be as follows:
    #   1. Compute accumulation in the bin based
    #       glac_bin_snow + unmelted snow from previous time step
    #   2. Compute energy required to melt all the snow
    #       E_melt_snow_all = accumulation
    #   3. Compute energy available for melting
    #       E_available = Tair_mean * n_days
    #       (provide option to use standard deviation like Huss and Hock (2015))
    #   4. Compute difference between energy available and for snow melt
    #       E_leftover = E_available - E_melt_snow_all
    #           If E_leftover > 0,
    #               then E_leftover used to melt ice or firn
    #                   melt_ice/firn = DDF_ice/firn * E_leftover
    #                   snow_leftover = 0
    #           If E_leftover < 0,
    #               then the negative energy represents the unmelted snow
    #                   melt_ice/firn = 0
    #                   snow_leftover = -1 * E_leftover * DDF_snow
    #   5. Repeat for each month in the year
    #   6. Reach the end of the year (if month = December)
    #       a. Calculate refreezing
    #       b. Calculate specific mass balance
    #       c. Update surface type according to mass balance over last 5 years
    #       d. Volume-length scaling (add remove elevation bins)
    #            > Simply modify the surface type to remove bins (change to 0)
    #              or to add bins (change 0 to 1[ice])
    #   7. Repeat with the accumulation, etc.
    #
    #   Try to keep the framework in this calling function form as much as
    #   possible.  Many new variable names need to be created.  Keep this
    #   consistent.
    #
    #
    # First see the surface type:
    #   > Did snow fall?  If so, then surface type has temporarily changed
    # Second


def ablationsurfacebinsmonthly(option_fxn, bin_ablation_mon, glac_temp,
                                 glac_surftype, glac_params, dates_table,
                                 glac_count, year):
    # Note: this will only work for monthly time step!
    """
    Calculate the surface ablation of every elevation bin for the glacier.
    Convention: negative ablation indicates surface lowering.
    """
    # Surface Ablation Options:
    #   > 1 (default) - degree day factor model for snow and ice with no
    #                   manipulation of the temperature data
    #   > 2 - degree day factor model for snow and ice with daily variations
    #         built in via daily standard deviations (Huss and Hock, 2015)
    #   > 3 - addition of a third model parameter for debris cover?
    #
    # Create an empty dataframe to record monthly ablation
    bin_ablation_perday = pd.DataFrame(0, columns=glac_temp.columns,
                                       index=glac_temp.index)
    bin_ablation_monraw = pd.DataFrame(0, columns=glac_temp.columns,
                                         index=glac_temp.index)
    if option_fxn == 1:
        # Option 1: DDF model for snow and ice (no adjustments)
        #   ablation_bin,m = DDF_ice,snow,debris * T_m_pos * n_timestep
        # Note: ablation is computed for each element using logical indexing.
        #   Then, a subset of the data is extracted according to the model year.
        #   While the computations may seem longer, the use of logical indexing
        #   instead of for loops greatly speeds up the computational time, which
        #   is why this is done.
        # Note: "-1" is used for convention such that negative ablation
        #   indicates surface lowering.
        # Compute ablation per day over ice
        mask1 = ((glac_surftype == 1) & (glac_temp > 0))
        bin_ablation_perday[mask1] = (-1 * glac_temp[mask1] *
                                      glac_params.loc[glac_count, 'DDF_ice'])
            # a = -1 * DDF_ice * T_m
            # Note: conversion from per day to month below
        # Compute ablation per day over snow
        mask2 = ((glac_surftype == 2) & (glac_temp > 0))
        bin_ablation_perday[mask2] = (-1 * glac_temp[mask2] *
                                      glac_params.loc[glac_count, 'DDF_snow'])
            # a = -1 * DDF_snow * T_m
            # (conversion from per day to month below)
        # Convert daily ablation rate to monthly ablation rate
        bin_ablation_monraw = (
            bin_ablation_perday.mul(list(dates_table['daysinmonth']), axis=1)
            )
            # 'list' needed to multiply each row by the column 'daysinmonth'
        bin_ablation_monsubset = bin_ablation_monraw.filter(regex=str(year))
        bin_ablation_mon[bin_ablation_monsubset.columns] = (
                                                        bin_ablation_monsubset)
    elif option_fxn == 2:
        print("\nThis option for 'option_surfaceablation' has not been coded "
              "yet. Please choose an option that exists. Exiting model run.\n")
        exit()
    else:
        print("\nThis option for 'option_surfaceablation' does not exist. "
              " Please choose an option that exists. Exiting model run.\n")
        exit()
    print("The 'ablationsurfacebins' function has finished.")
    return bin_ablation_mon


def accumulationbins(option_fxn, glac_temp, glac_precsnow, glac_params,
                     glac_count):
    # Note: this will only work for monthly time step!
    """
    Calculate the accumulation for every elevation bin on the glacier.
    """
    # Surface Ablation Options:
    #   > 1 (default) - single threshold (temp below snow, above rain)
    #   > 2 - linear relationship (fraction of snow prec within +/- 1 deg
    #         so if temp = threshold, then 50% snow and rain)
    #
    bin_prec = pd.DataFrame(0, columns=glac_temp.columns, index=glac_temp.index)
    bin_snow = pd.DataFrame(0, columns=glac_temp.columns, index=glac_temp.index)
    if option_fxn == 1:
        mask1 = (glac_temp > glac_params.loc[glac_count,'T_snow'])
        mask2 = (glac_temp <= glac_params.loc[glac_count,'T_snow'])
        bin_prec[mask1] = glac_precsnow[mask1]
        bin_snow[mask2] = glac_precsnow[mask2]
    elif option_fxn == 2:
        mask1 = (glac_temp >= (glac_params.loc[glac_count,'T_snow'] + 1))
        mask2 = (glac_temp <= (glac_params.loc[glac_count,'T_snow'] - 1))
        mask3 = ((glac_temp < (glac_params.loc[glac_count,'T_snow'] + 1)) & (
                glac_temp > (glac_params.loc[glac_count,'T_snow'] - 1)))
        bin_prec[mask1] = glac_precsnow[mask1]
            # If glac_temp >= T_snow + 1, then precipitation
        bin_snow[mask2] = glac_precsnow[mask2]
            # If glac_temp <= T_snow - 1, then snow (precipitation = 0)
        bin_prec[mask3] = ((1/2 + (glac_temp[mask3] -
                          glac_params.loc[glac_count,'T_snow'])/2)
                          * glac_precsnow[mask3])
        bin_snow[mask3] = ((1 - (1/2 + (glac_temp[mask3] -
                          glac_params.loc[glac_count,'T_snow'])/2))
                          * glac_precsnow[mask3])
    else:
        print("This option for 'option_accumulation' does not exist.  Please"
              " choose an option that exists. Exiting model run.\n")
    print("The 'accumulationbins' functions has finished.")
    return bin_prec, bin_snow


def downscaleprec2bins(option_fxn, option_elev_ref, glac_table, glac_hyps,
                       glac_params, climate_prec, climate_elev, glac_count):
    """
    Downscale the global climate model precipitation data to each bin on the
    glacier using the precipitation bias factor (prec_factor) and the glacier
    precipitation gradient (prec_grad).
    """
    # Function Options:
    #   > 1 (default) - precipitation factor bias to correct GCM and a
    #                   precipitation gradient to adjust precip over the glacier
    #   > 2 (not coded yet) - Huss and Hock (2015), exponential limits, etc.
    bin_prec = pd.DataFrame(np.zeros(shape=(len(glac_hyps.loc[0]),
                            len(climate_prec.loc[0]))),
                            columns=list(climate_prec.columns.values),
                            index=list(glac_hyps.columns.values))
    for row in range(len(glac_hyps.iloc[0])):
        # each row is a different elevation bin for a particular glacier
        # the columns are the time series of the precipitation data
        bin_elev = glac_hyps.columns.values[row]
        if option_fxn == 1:
            # Option 1 is the default and uses a precipitation factor and
            # precipitation gradient over the glacier.
            bin_prec.loc[bin_elev] = (climate_prec.loc[glac_count] *
                                 glac_params.loc[glac_count, 'prec_factor'] *
                                 (1 + glac_params.loc[glac_count, 'prec_grad']
                                 * (int(bin_elev) -
                                 glac_table.loc[glac_count, option_elev_ref])))
            # For each elevation bin, this applies the following:
            #   P_bin = P_gcm * prec_factor * (1 + prec_grad * (z_bin - z_ref))
        else:
            print("\nThis option for 'downscaleprec2bins' has not been coded "
                  "yet. Please choose an existing option. Exiting model run.\n")
            exit()
    print("The 'downscaleprec2bins' function has finished.")
    return bin_prec

def downscaletemp2bins(option_fxn, option_elev_ref, glac_table, glac_hyps,
                       glac_params, climate_temp, climate_elev, glac_count):
    """
    Downscale the global climate model temperature data to each bin on the
    glacier using the global climate model lapse rate (lr_gcm) and the glacier
    lapse rate (lr_glac).
    """
    # Function Options:
    #   > 1 (default) - lapse rate for gcm and glacier
    #   > no other options currently exist
    bin_temp = pd.DataFrame(np.zeros(shape=(len(glac_hyps.loc[0]),
                            len(climate_temp.loc[0]))),
                            columns=list(climate_temp.columns.values),
                            index=list(glac_hyps.columns.values))
    for row in range(len(glac_hyps.iloc[0])):
        # each row is a different elevation bin for a particular glacier
        # the columns are the time series of temperature data
        bin_elev = glac_hyps.columns.values[row]
        if option_fxn == 1:
            # Option 1 is the default and uses a lapse rate for the gcm and
            # a glacier lapse rate.
            bin_temp.loc[bin_elev] = (climate_temp.loc[glac_count] +
                                     glac_params.loc[glac_count, 'lr_gcm'] * (
                                     glac_table.loc[glac_count, option_elev_ref]
                                     - climate_elev.loc[glac_count]) +
                                     glac_params.loc[glac_count, 'lr_glac'] * (
                                     int(bin_elev) - glac_table.loc[glac_count,
                                     option_elev_ref]))
            # For each elevation bin, this applies the following:
            #   T_bin = T_gcm + lr_gcm * (z_ref - z_gcm)
            #            + lr_glac * (z_bin - z_ref)
        else:
            print("\nThis option for 'downscaletemp2bins' has not been coded "
                  "yet. Please choose an existing option. Exiting model run.\n")
            exit()
    print("The 'downscaletemp2bins' function has finished.")
    return bin_temp


def groupbyyearmean(var):
    """
    Calculate annual mean of variable according to the year in the column header
    Note: need to add in option to use option_wateryear
    """
    if timestep == 'monthly':
        var_annual = var.groupby(np.arange(var.shape[1]) // 12, axis=1).mean()
    elif timestep == 'daily':
        print('\nError: need to code the groupbyyearsum and groupbyyearmean for daily timestep.'
              'Exiting the model run.\n')
        exit()
    return var_annual


def groupbyyearsum(var):
    """
    Calculate annual sum of variable according to the year in the column header
    """
    if timestep == 'monthly':
        var_annual = var.groupby(np.arange(var.shape[1]) // 12, axis=1).sum()
    elif timestep == 'daily':
        print('\nError: need to code the groupbyyearsum and groupbyyearmean for daily timestep.'
              'Exiting the model run.\n')
        exit()
    return var_annual


def refreezingbins(option_fxn, glac_temp_annual, snow_annual,
                   surftype_annual, glac_temp):
    # Note: this will only work for monthly time step!
    """
    Calculate the refreezing for every elevation bin on the glacier.
    """
    # Refreezing Options:
    #   > 1 (default) - monthly refreezing solved by modeling snow and firm
    #                   temperatures using heat conduction equations according
    #                   to Huss and Hock (2015)
    #   > 2 - annual refreezing based on mean air temperature according to
    #         Woodward et al. (1997)
    #
    bin_refreeze = pd.DataFrame(0, columns=glac_temp.columns,
                                index=glac_temp.index)
    if option_fxn == 1:
        print('This option based on Huss and Hock (2015) is intended to be the '
              'default; however, it has not been coded yet due to its '
              'complexity.  For the time being, please choose an option that '
              'exists.\n\nExiting model run.\n\n')
        exit()
    elif option_fxn == 2:
        # Compute bin refreeze potential according to Woodward et al. (1997)
        bin_refreeze_annual = ((-0.69 * glac_temp_annual + 0.0096)
                                         * 1/100)
            # R(m) = -0.69 * Tair + 0.0096 * (1 m / 100 cm)
            # Note: conversion from cm to m is included
        # Apply bounds for refreezing
        # Bound 1: Refreezing cannot be less than 0
        mask1 = (bin_refreeze_annual < 0)
        bin_refreeze_annual[mask1] = 0
        # Bound 2: In the ablation area (surface type of ice for that year), the
        #          maximum refreezing is equal to the accumulated snow
        mask2 = ((surftype_annual == 1) &
                 (bin_refreeze_annual > snow_annual))
        bin_refreeze_annual[mask2] = snow_annual[mask2]
        # Lastly, refreezing only occurs on glacier (off-glacier = 0)
        mask3 = (surftype_annual == 0)
        bin_refreeze_annual[mask3] = 0
        # Place annual refreezing in January for accounting and melt purposes
        if timestep == 'monthly':
            for col in range(len(bin_refreeze_annual.iloc[0])):
                bin_refreeze.iloc[:,col*12] = bin_refreeze_annual.iloc[:,col]
        elif timestep == 'daily':
            print("MODEL ERROR: daily time step not coded for Woodward et al. "
                  "(1997) refreeze function yet.\n\nExiting model run.\n\n")
            exit()
        else:
            print("MODEL ERROR: please select 'daily' or 'monthly' as the time "
                  "step for the model run.\n\nExiting model run.\n\n")
    else:
        print("This option for 'refreezingbins' does not exist.  Please choose "
              "an option that exists. Exiting model run.\n")
    print("The 'refreezingbins' functions has finished.")
    return bin_refreeze, bin_refreeze_annual


def specificmassbalanceannual(massbal_annual, snow_annual, ablation_annual,
                              refreeze_annual, surftype_annual, year,
                              glac_count, glac_ELA):
    """
    Calculate the annual specific mass balance for every elevation bin on the
    glacier for a given year.  Also, extract the equilibrium line altitude for
    the given year.
    """
    massbal_annual_raw = massbal_annual.copy()
    # Compute surface mass balance for all elevation bins of the glacier
    mask1 = (surftype_annual > 0)
        # elements that are not on the glacier are excluded
        # !!! WARNING: this inherently means that the model is not able to !!!
        # !!!          grow any glaciers from scratch                      !!!
    massbal_annual_raw[mask1] = (snow_annual[mask1] + ablation_annual[mask1] +
                                 refreeze_annual[mask1])
    # Extract the data for the given loop year
    massbal_annual_subset = massbal_annual_raw.filter(regex=str(year))
    # Update the mass balance dataset with the subset
    massbal_annual[massbal_annual_subset.columns] = massbal_annual_subset
    # Determine the ELA
    # Use when the sign of the specific mass balance changes
    ELA_sign = np.sign(massbal_annual_subset)
        # returns array of values for positive(1), negative(-1), or zero(0)
    # ELA_signchange = ((np.roll(ELA_sign,1) - ELA_sign) == -2).astype(int)
    ELA_signchange = np.where((np.roll(ELA_sign,1) - ELA_sign) == -2)
        # roll is a numpy function to perform a circular shift, so in this case
        # all the values are shifted up one place.  Since we are looking for the
        # change from negative to positive, i.e., a value of -1 to +1, we really
        # are looking for a value of -2. numpy where is used to find this value
        # of -2.  The ELA will be the mean elevation between this bin and the
        # bin below it.  It's important to note that we use -2 as opposed to not
        # equal to zero because there will be sign changes at the min glacier
        # altitude and the max glacier altitude. Additionally, if there is a
        # bin, that does not have a glacier (very steep section), then that will
        # not show up as a false ELA.
        #   Example: bin 4665 m has a negative mass balance and 4675 m has a
        #   positive mass balance. The difference with the roll function will
        #   give 4675 m a value of -2.  Therefore, the ELA will be 4670 m.
    glac_ELA.loc[glac_count, str(year)] = int((ELA_sign.index[ELA_signchange[0]]
                                               [0] - binsize/2))
        # ELA_signchange[0] returns the position associated with ELA
        # ELA_sign.index returns an array with one value, i.e., the ELA value
        # the [0] after ELA_sign.index accesses the element in that array
        # The binsize is then reduced
    print("The 'specificmassbalanceannual' function has finished.")
    return massbal_annual, glac_ELA


def surfacetypebinsinitial(glac_surftype, glac_temp, glac_count):
    """
    Create dataframe for initial surface type.  Note: this is needed for every
    timestep to assist with logical indexing in ablation calculations.
    Otherwise, annual timestep would be sufficient as it is constant each year.
    Convention:
        1 - ice
        2 - snow
        3 - firn
        4 - debris
        0 - off-glacier
    """
    # Select initial glacier surface type from the main table that has initial
    # surface type for each glacier based on median altitude
    bin_surftype_series = glac_surftype.iloc[glac_count,:]
    # Create dataframe for surface type for every timestep, which is needed to
    # assist logical indexing in ablation calculations despite being constant
    # for each year.  Note: glac_temp is simply a dummy file that is being used
    # for its dataframe attributes (length and column headers)
    bin_surftype = pd.concat([bin_surftype_series] * len(glac_temp.iloc[0]),
                             axis=1)
    bin_surftype.columns = glac_temp.columns
    return bin_surftype


def surfacetypebinsannual(option_fxn, surftype_annual, massbal_annual, year):
    """
    Update surface type according to mass balance each year.
        mass balance = accumulation - ablation + refreezing
    if mass balance is positive --> snow/firn
    if mass balance is negative --> ice
    where mass balance is zero (or switches) --> ELA
    Output:
        > monthly and annual surface type
        > annual ELA
    Convention:
        1 - ice
        2 - snow
        3 - firn
        4 - debris
        0 - off-glacier
    """
    # Function Options:
    #   > 1 (default) - update surface type according to Huss and Hock (2015)
    #   > 2 - Radic and Hock (2011)
    # Radic and Hock (2011): "DDF_snow is used above the ELA regardless of
    #   snow cover.  Below the ELA, use DDF_ice is used only when snow cover is
    #   0.  ELA is calculated from the observed annual mass balance profiles
    #   averaged over the observational period and is kept constant in time for
    #   the calibration period.  For the future projections, ELA is set to the
    #   mean glacier height and is time dependent since glacier volume, area,
    #   and length are time dependent (volume-area-length scaling).
    #       > 1 - ELA is constant.  ELA is calculated from annual mass balance
    #             profiles over observational period.  However, these profiles
    #             Problem: these profiles will change depending on where you set
    #                      the ELA, so now you're in a loop?
    #       > 2 - ELA is the mean glacier elevation.  Changes every year as the
    #             glacier changes (volume-area-length scaling).
    #       Surface types (2): snow, ice
    # Bliss et al. (2014) uses the same as Valentina's model
    # Huss and Hock (2015): Initially, above median glacier elevation is firn
    #   and below is ice. Surface type updated for each elevation band and month
    #   depending on the specific mass balance.  If the cumulative balance since
    #   the start of the mass balance year is positive, then snow is assigned.
    #   If the cumulative mass balance is negative (i.e., all snow of current
    #   mass balance year has melted), then bare ice or firn is exposed.
    #   Surface type is assumed to be firn if the elevation band's average
    #   annual balance over the preceding 5 years (B_t-5_avg) is positive. If
    #   B_t-5_avg is negative, surface type is ice.
    #       > 1 - specific mass balance calculated at each bin and used with
    #             the mass balance over the last 5 years to determine whether
    #             the surface is firn or ice.  Snow is separate based on each
    #             month.
    #       Surface types (3): Firn, ice, snow
    #           DDF_firn = average(DDF_ice, DDF_snow)
    if option_fxn == 1:
        if year >= startyear + 4:
            # Within 1st 5 years, unable to take average of the preceding 5
            # years, so assume that it is constant.  Don't change anything.
            # Note: this suggests that 5 years for model spinup is a good idea.
            #
            # Compute average mass balance over the last 5 years for each bin
            massbal_annual_avg = massbal_annual.loc[:,year-4:year].mean(axis=1)
                # returns a Series for the given year
            # Update surface type according to the average annual mass balance
            surftype_annual[year][massbal_annual_avg < 0] = 1
            surftype_annual[year][massbal_annual_avg > 0] = 2
                # logical indexing used in conjunction with year column
        # NEED TO ADD OPTION FOR FIRN
        # NEED TO ADD OPTION FOR DEBRIS
    return surftype_annual
