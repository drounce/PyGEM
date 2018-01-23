"""
fxns_massbalance.py is a list of functions that are used to compute the mass
associated with each glacier for PyGEM.
"""
#========= LIST OF PACKAGES ==================================================
import pandas as pd
import numpy as np
#========= IMPORT COMMON VARIABLES FROM MODEL INPUT ==========================
import pygem_input as input

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
def massbalance_binsmonthly():
    # Note: this will only work for monthly time step!
    """
    Calculate the mass balance for every elevation bin on the glacier.
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
    #   1. Compute accumulation in the bin
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
    #   Many new variable names need to be created.  Keep this consistent.
    
    
#========= OLDER SCRIPTS PRE-JANUARY 11, 2018 =========================================================================
def AAR_glacier(ELA_value, series_area, GlacNo):
    """
    Compute the Accumulation-Area Ratio (AAR) for a given glacier based on its ELA
    """
    try:
        AAR_output = (1 - (np.cumsum(series_area)).divide(series_area.sum())
            .iloc[int(ELA_value / input.binsize) - 1]) * 100
        #  ELA_value is the elevation associated with the ELA, so dividing this by the binsize returns the column position 
        #    if the indexing started at 1, the "-1" accounts for the fact that python starts its indexing at 0, so
        #    ".iloc[int(ELA_value / binsize) - 1]" gives the column of the ELA.
        #  np.cumsum gives the cumulative sum of the glacier area for the given year
        #    this is divided by the total area to get the cumulative fraction of glacier area.
        #  The column position is then used to select the cumulative fraction of glacier area of the ELA
        #    since this is the area below the ELA, the value is currently the ablation area as a decimal;
        #    therefore, "1 - (cumulative_fraction)" gives the fraction of the ablation area,
        #    and multiplying this by 100 gives the fraction as a percentage.
    except:
        # if ELA does not exist, then set AAR = -9.99
        AAR_output = -9.99
    return AAR_output    


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


def accumulationbins(glac_temp, glac_precsnow):
    # Note: this will only work for monthly time step!
    """
    Calculate the accumulation for every elevation bin on the glacier.
    
    Output: Pandas dataframes of precipitation [m] and snow [m w.e.] for each bin for each timestep
    (rows = bins, columns = dates)
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
        bin_prec[glac_temp > input.T_snow] = glac_precsnow[glac_temp > input.T_snow]
        # If temperature below threshold, then snow
        bin_snow[glac_temp <= input.T_snow] = glac_precsnow[glac_temp <= input.T_snow]
    elif input.option_accumulation == 2:
        # If temperature above maximum threshold, then all rain
        bin_prec[glac_temp >= input.T_snow + 1] = glac_precsnow[glac_temp >= input.T_snow + 1]
        # If temperature below minimum threshold, then all snow
        bin_snow[glac_temp <= input.T_snow - 1] = glac_precsnow[glac_temp <= input.T_snow - 1]
        # Otherwise temperature between min/max, then mix of snow/rain using linear relationship between min/max
        bin_prec[(glac_temp < input.T_snow + 1) & (glac_temp > input.T_snow - 1)] = ((1/2 + (
            glac_temp[(glac_temp < input.T_snow + 1) & (glac_temp > input.T_snow - 1)] - input.T_snow)/2) * 
            glac_precsnow[(glac_temp < input.T_snow + 1) & (glac_temp > input.T_snow - 1)])
        bin_snow[(glac_temp < input.T_snow + 1) & (glac_temp > input.T_snow - 1)] = ((1 - (1/2 + (
            glac_temp[(glac_temp < input.T_snow + 1) & (glac_temp > input.T_snow - 1)] - input.T_snow)/2)) * 
            glac_precsnow[(glac_temp < input.T_snow + 1) & (glac_temp > input.T_snow - 1)])
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
        weights = (dayspermonth / daysperyear[:,np.newaxis]).reshape(-1)
        #  computes weights for each element, then reshapes it for next step
        var_annual = (var*weights[np.newaxis,:]).reshape(-1,12).sum(axis=1).reshape(-1,daysperyear.shape[0])
        #  computes matrix (rows - bins, columns - year) of weighted average for each year
    elif input.timestep == 'daily':
        print('\nError: need to code the groupbyyearsum and groupbyyearmean for daily timestep.'
              'Exiting the model run.\n')
        exit()
    return var_annual


# re-write groupbyyearsum using the reshape framework above - much faster!
def groupbyyearsum(var):
    print('MUST RE-WRITE FUNCTION USING RESHAPE FRAMEWORK - SEE ANNUALWEIGHTEDMEAN_ARRAY')
#    """
#    NOTE: UPDATE THIS LIKE ANNUALWEIGHTEDMEAN_ARRAY!!!
#    
#    Calculate annual sum of variable according to the timestep.
#    Example monthly timestep will group every 12 months, so starting month is important.
#    """
#    if input.timestep == 'monthly':
#        var_annual = var.groupby(np.arange(var.shape[1]) // 12, axis=1).sum()
#    elif input.timestep == 'daily':
#        print('\nError: need to code the groupbyyearsum and groupbyyearmean for daily timestep.'
#              'Exiting the model run.\n')
#        exit()
#    return var_annual


def downscaleprec2bins(glac_table, glac_hyps, climate_prec, climate_elev, glac_count):
    """
    Downscale the global climate model precipitation data to each bin on the glacier using the precipitation bias factor
    (prec_factor) and the glacier precipitation gradient (prec_grad).
    
    Output: Pandas dataframe of precipitation [m] in each bin for each time step
    (rows = bins, columns = dates)
    """
    # Function Options:
    #   > 1 (default) - precip factor bias to correct GCM and a precipitation gradient to adjust precip over the glacier
    #   > 2 (not coded yet) - Huss and Hock (2015), exponential limits, etc.
    if input.option_prec2bins == 1:
        # Option 1 is the default and uses a precipitation factor and precipitation gradient over the glacier.
        bin_prec = (climate_prec.values[glac_count,:] * input.prec_factor * (1 + input.prec_grad * (
                    glac_hyps.columns.values.astype(int) - glac_table.loc[glac_count, input.option_elev_ref_downscale])
                    )[:,np.newaxis])
        #   P_bin = P_gcm * prec_factor * (1 + prec_grad * (z_bin - z_ref))
    else:
        print("\nThis option for 'downscaleprec2bins' has not been coded yet. Please choose an existing option."
              "Exiting model run.\n")
        exit()
    return bin_prec


def downscaletemp2bins(glac_table, glac_hyps, climate_temp, climate_elev, glac_count):
    """
    Downscale the global climate model temperature data to each bin on the glacier using the global climate model 
    lapse rate (lr_gcm) and the glacier lapse rate (lr_glac).
    
    Output: Pandas dataframe of temperature [degC] in each bin for each time step
    (rows = bins, columns = dates)
    """
    # Function Options:
    #   > 1 (default) - lapse rate for gcm and glacier
    #   > no other options currently exist
    if input.option_temp2bins == 1:
        # Option 1 is the default and uses a lapse rate for the gcm and a glacier lapse rate.
        bin_temp = (climate_temp.values[glac_count,:] + (input.lr_gcm * (
                    glac_table.loc[glac_count, input.option_elev_ref_downscale] - 
                    climate_elev.loc[glac_count]) + input.lr_glac * (glac_hyps.columns.values.astype(int) - 
                    glac_table.loc[glac_count, input.option_elev_ref_downscale]))[:,np.newaxis])
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
    

def ELA_glacier(series_massbal_spec, ELA_past):
    """
    Compute the Equlibrium Line Altitude (ELA) from a series of specific mass balance, i.e., a single column of the 
    specific mass balance for each elevation bin
    """
    # Use numpy's sign function to return an array of the sign of the values (1=positive, -1=negative, 0=zero)
    series_ELA_sign = np.sign(series_massbal_spec)                
    # Use numpy's where function to determine where the specific mass balance changes from negative to positive
    series_ELA_signchange = np.where((np.roll(series_ELA_sign,1) - series_ELA_sign) == -2)
    #   roll is a numpy function that performs a circular shift, so in this case all the values are shifted up one 
    #   place. Since we are looking for the change from negative to positive, i.e., a value of -1 to +1, we want to 
    #   find where the value equals -2. numpy's where function is used to find this value of -2.  The ELA will be 
    #   the mean elevation between this bin and the bin below it.
    #   Example: bin 4665 m has a negative mass balance and 4675 m has a positive mass balance. The difference with 
    #            the roll function will give 4675 m a value of -2.  Therefore, the ELA will be 4670 m.
    #   Note: If there is a bin with no glacier area between the min and max height of the glacier (ex. a very steep 
    #     section), then this will not be captured.  This only becomes a problem if this bin is technically the ELA, 
    #     i.e., above it is a positive mass balance, and below it is a negative mass balance.  Using np.roll with a
    #     larger shift would be one way to work around this issue.
    # try and except to avoid errors associated with the entire glacier having a positive or negative mass balance
    try:
        ELA_output = (series_massbal_spec.index.values[series_ELA_signchange[0]][0] - input.binsize/2).astype(int)
        #  series_ELA_signchange[0] returns the position of the ELA. series_massbal_annual.index returns an array 
        #  with one value, so the [0] ais used to accesses the element in that array. The binsize is then used to 
        #  determine the median elevation between those two bins.
    except:
        # This may not work in three cases:
        #   > The mass balance of the entire glacier is completely positive or negative.
        #   > The mass balance of the whole glacier is 0 (no accumulation or ablation, i.e., snow=0, temp<0)
        #   > The ELA falls on a band that does not have any glacier (ex. a very steep section) causing the sign 
        #     roll method to fail. In this case, using a large shift may solve the issue.
        try:
            # if entire glacier is positive, then set to the glacier's minimum
            if series_ELA_sign.iloc[np.where(series_ELA_sign != 0)[0][0]] == 1:
                ELA_output = series_ELA_sign.index.values[np.where(series_ELA_sign != 0)[0][0]] - input.binsize/2
            # if entire glacier is negative, then set to the glacier's maximum
            elif series_ELA_sign.iloc[np.where((series_ELA_sign != 0))[0][0]] == -1:
                ELA_output = series_ELA_sign.index.values[np.where(series_ELA_sign != 0)[0]
                             [np.where(series_ELA_sign != 0)[0].shape[0]-1]] + input.binsize/2
        except:
            # if the specific mass balance over the entire glacier is 0, i.e., no ablation or accumulation,
            #  then the ELA is the same as the previous timestep
            ELA_output = ELA_past
    return ELA_output


def massredistribution(icethickness_t0, glacier_area, elev_bins, glacier_volumechange):
    """ 
    Compute the mass redistribution, otherwise known as glacier geometry changes, based on the glacier volume change
    Function Options:
    - option_geometrychange
        > 1 (default) - Huss and Hock (2015); volume gain/loss redistributed over the glacier using empirical normalized
                        ice thickness change curves
        > 2 (Need to code) - volume-length scaling
        > 3 (Need to code) - volume-area scaling
        > 4 (Need to code) - what previous models have done
        > 5 (Need to code) - ice dynamics, simple flow model
    Input:
        > glacier_area - single column array of glacier area for every bin
        > elev_bins - single column array of elevation for every bin
        > massbal_clim_mwe_annual - single column array of annual climatic mass balance [m w.e.] for every bin
        > icethickness_t0 - single column array of ice thickness for every bin at the start of the time step
    """
    # Option 1 (default) for mass redistribution, Huss and Hock (2015)
    if input.option_massredistribution == 1:
        #Select the factors for the normalized ice thickness change curve based on glacier areasize
         if glacier_area.sum() > 20:
             [gamma, a, b, c] = [6, -0.02, 0.12, 0]
         elif glacier_area.sum() > 5:
             [gamma, a, b, c] = [4, -0.05, 0.19, 0.01]
         else:
             [gamma, a, b, c] = [2, -0.30, 0.60, 0.09]
         # reset normalized elevation range values, since calculated using indices
         elevrange_norm = np.zeros(elev_bins.shape)
         icethicknesschange_norm = np.zeros(elev_bins.shape)
         # compute position of index of all bins that are nonzero
         elev_idx = glacier_area.nonzero()[0].nonzero()[0]
         #  nonzero()[0] returns the position of all nonzero values, so nonzero()[0].nonzero()[0] returns the
         #  position of all nonzero values with the values ranging from 0 to the total number of nonzero values.
         # Normalized elevation range [-]
         #  (max elevation - bin elevation) / (max_elevation - min_elevation)
         elevrange_norm[glacier_area > 0] = (elev_idx[-1] - elev_idx) / elev_idx[-1]
         #  using indices as opposed to elevations automatically skips bins on the glacier that have no area
         #  such that the normalization is done only on bins where the glacier lies
         # Normalized ice thickness change [-]
         icethicknesschange_norm[glacier_area > 0] = ((elevrange_norm[glacier_area > 0] + a)**gamma + 
                                                      b*(elevrange_norm[glacier_area > 0] + a) + c)
         #  delta_h = (h_n + a)**gamma + b*(h_n + a) + c
         #  indexing is faster here
         # limit the icethicknesschange_norm to between 0 - 1 (ends of fxns not exactly 0 and 1)
         icethicknesschange_norm[icethicknesschange_norm > 1] = 1
         icethicknesschange_norm[icethicknesschange_norm < 0] = 0
         # Huss' ice thickness scaling factor, fs_huss [m ice]         
         fs_huss = glacier_volumechange / (glacier_area * icethicknesschange_norm).sum() * 1000
         #  units: km**3 / (km**2 * [-]) * (1000 m / 1 km) = m ice
         # Ice thickness change [m ice]
         icethicknesschange = icethicknesschange_norm * fs_huss
         #  units: [-] * [m ice]
         # Ice thickness at end of time step [m ice]
         icethickness_t1 = icethickness_t0 + icethicknesschange
         # return the updated ice thickness [m ice]
         return icethickness_t1
    


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
            placeholder = (12 - dates_table.loc[0,'month'] + input.refreeze_month) % 12
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
                                               [0] - input.binsize/2))
        # ELA_signchange[0] returns the position associated with ELA
        # ELA_sign.index returns an array with one value, i.e., the ELA value
        # the [0] after ELA_sign.index accesses the element in that array
        # The binsize is then reduced
    print("The 'specificmassbalanceannual' function has finished.")
    return massbal_annual, glac_ELA


def surfacetypebinsinitial(glac_surftype, glac_temp, glac_count):
    """
    NOTE THIS FUNCTION IS NO LONGER USED - SHOULD BE ABLE TO DELETE THIS
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
    bin_surftype = pd.concat([bin_surftype_series] * glac_temp.shape[1], axis=1)
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
        if year >= input.startyear + 4:
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
