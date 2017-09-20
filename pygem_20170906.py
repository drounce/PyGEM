###############################################################################
"""
Python Glacier Evolution Model "PyGEM" V1.0
Prepared by David Rounce with support from Regine Hock.
This work was funded under the NASA HiMAT project (INSERT PROJECT #).

PyGEM is an open source glacier evolution model written in python.  Model
details come from Radic et al. (2013), Bliss et al. (2014), and Huss and Hock
(2015).
"""
# Trial push
###############################################################################
# This is the main script that provides the architecture and framework for all
# of the model runs. All input data is included in a separate module called
# pygem_input.py. It is recommended to not make any changes to this file unless
# you are a PyGEM developer and making changes to the model architecture.
#
#========== IMPORT PACKAGES ==================================================
# Various packages are used to provide the proper architecture and framework
# for the calculations used in this script. Some packages (e.g., datetime) are
# included in order to speed of calculations and simplify code
import pandas as pd
import numpy as np
from datetime import datetime
import os # os is used with re to find name matches
import re # see os
#========= IMPORT MODEL INPUTS ===============================================
from pygem_input import *
    # import all data
    # pygem_input.py contains all the input data
#========== IMPORT FUNCTIONS FROM MODULES ====================================
import pygemfxns_modelsetup as modelsetup
import pygemfxns_climate as climate
import pygemfxns_massbalance as massbalance

#========== OTHER STEPS ======================================================
# Other steps:
# Geodetic mass balance file path
# ???[Create input script full of option list]???
#       - Select regions: (Option 1 - default regions from RGI inventory)
#                         (Option 2 - custom regions)
#       - Select climate data:
#           - climate data source/file name
#           - climate data resolution
#           - sampling scheme (nearest neighbor, etc.)
#       - Model run type: (Option 1 - calibration)
#                         (Option 2 - simulation)
#

#========== LIST OF OUTPUT ===================================================
# Create a list of outputs (.csv or .txt files) that the user can choose from
# depending on what they are using the model for or want to see:
#    1. Time series of all variables for each grid point (x,y,z) of each glacier
#       and every time step
#       (--> comparing with point balances)
#    2. Time series of all variables for elevation bands if spatial
#       discretization is elevation bands (x,y,z) of each glacier and every time
#       step
#       (--> comparing with profiles)
#    3. as above but all variables averaged over current single glacier surfaces
#       (--> comparing with geodetic mass balances of glaciers)
#    4. Time series of all variables averaged over region
#       (--> comparison to GRACE)
#    5. Time series of seasonal balance for individual glaciers
#       (--> comparison to WGMS data)
# Also develop output log file, i.e., file that states input parameters,
# date of model run, model options selected, and any errors that may have come
# up (e.g., precipitation corrected because negative value, etc.)

#========== MODEL RUN DETAILS ================================================
# The model is run through a series of steps:
#   > Step 01: Region/Glaciers Selection
#              The user needs to define the region/glaciers that will be used in
#              the model run.  The user has the option of choosing the standard
#              RGI regions or defining their own.
#   > Step 02: Model Time Frame
#              The user should consider the time step and duration of the model
#              run along with any calibration product and/or model spinup that
#              may be included as well.
#   > Step 03: Climate Data
#              The user has the option to choose the type of climate data being
#              used in the model run, and how that data will be downscaled to
#              the glacier and bins.
#   > Step 04: Glacier Evolution
#              The heart of the model is the glacier evolution, which includes
#              calculating the specific mass balance, the surface type, and any
#              changes to the size of the glacier (glacier dynamics). The user
#              has many options for how this aspect of the model is run.
#   > Others: model output? model input?

#========== LIST OF MODEL VARIABLES (alphabetical) ===========================
# Prefixes and Suffixes:
#   _annual: dataframe containing information with an annual time step as
#           opposed to the time step specified by the model (daily or monthly).
#   _bin_:  dataframe containing information related to each elevation bin/band
#           on the glacier. These dataframes are indexed such that the main
#           index (rows) are elevation bins and the columns are the time series.
#   _glac_: dataframe containing information related to a glacier. When used by
#           itself (e.g., main_glac_rgi or gcm_glac_temp) it refers to each row
#           being a specific glacier. When used as a prefix followed by a
#           descriptor (e.g., glac_bin_temp), the entire dataframe is for one
#           glacier and the descriptor provides information as to the rows
#   gcm_:   meteorological data from the global climate model or reanalysis
#           dataset
#   main_:  dataframe containing important information related to all the
#           glaciers in the study, where each row represents a glacier (ex.
#           main_glac_rgi).
#   series_: series containing information for a given time step with respect to
#           all the glacier bins. This is needed when cycling through each
#           time step to calculate the mass balance since snow accumulates and
#           alters the surface type.
#
# Variables:
#   dates_table: dataframe of the dates, month, year, and number of days in the
#           month for each date.
#           Rows = dates, Cols = attributes
# > end_date: end date of model run
#           (MAY NOT BE NEEDED - CHECK WHERE IT'S USED)
#   gcm_glac_elev: Series of elevation data associated with the global climate
#           model temperature data
#   gcm_glac_prec: dataframe of the global climate model precipitation data,
#           typically based on the nearest neighbor.
#   gcm_glac_temp: dataframe of the global climate model temperature data,
#           typically based on the nearest neighbor.
#   glac_bin_temp: dataframe of temperature for each bin for each time step on
#           the glacier
#   glac_bin_prec: dataframe of precipitation (liquid) for each bin for each
#           time step on the glacier
#   glac_bin_precsnow: dataframe of the total precipitation (liquid and solid)
#           for each bin for each time step on the glacier
#   glac_bin_snow: dataframe of snow for each bin for each time step on
#           the glacier
#   glac_bin_snowonsurface: dataframe of the snow that has accumulated on the
#           surface of the glacier for each bin for each time step
#   glac_bin_surftype: datframe of the surface type for each bin for each time
#           step on the glacier
#   main_glac_rgi: dataframe containing generic attributes (RGIId, Area, etc.)
#           from the Randolph Glacier Inventory for each glacier.
#           Rows = glaciers, Cols = attributes
#   main_glac_parameters: dataframe containing the calibrated parameters for
#           each glacier
#   main_glac_surftypeinit: dataframe containing in the initial surface type for
#           each glacier
# > start_date: start date of model run
#           (MAY NOT BE NEEDED - CHECK WHERE IT'S USED)

#----- STEP ONE: MODEL REGION/GLACIERS ---------------------------------------
# Step one involves the selection of the regions and glaciers used in the model.
# Regions/glacier included in the model run will be defined using the Randolph
#   Glacier inventory.  For more information, see:
#     https://www.glims.org/RGI/ (RGI Consortium, 2017)
# In step one, the model will:
#   > select glaciers included in model run

# Select glaciers that are included in the model run
# Glacier Selection Options:
#   > 1 (default) - enter numbers associated with RGI V6.0 and select
#                   glaciers accordingly
#   > 2 - glaciers/regions selected via shapefile
#   > 3 - glaciers/regions selected via new table (other inventory)
#
if option_glacier_selection == 1:
    main_glac_rgi = modelsetup.selectglaciersrgitable()
elif option_glacier_selection == 2:
    # OPTION 2: CUSTOMIZE REGIONS USING A SHAPEFILE that specifies the
    #           various regions according to the RGI IDs, i.e., add an
    #           additional column to the RGI table.
    # ??? [INSERT CODE FOR IMPORTING A SHAPEFILE] ???
    #   (1) import shapefile with custom boundaries, (2) grab the RGIIDs
    #   of glaciers that are in these boundaries, (3) perform calibration
    #   using these alternative boundaries that may (or may not) be more
    #   representative of regional processes/climate
    #   Note: this is really only important for calibration purposes and
    #         post-processing when you want to show results over specific
    #         regions.
    # Development Note: if create another method for selecting glaciers,
    #                   make sure that update way to select glacier
    #                   hypsometry as well.
    print('\n\tMODEL ERROR (selectglaciersrgi): this option to use'
          '\n\tshapefiles to select glaciers has not been coded yet.'
          '\n\tPlease choose an option that exists. Exiting model run.\n')
    exit()
else:
    # Should add options to make regions consistent with Brun et al. (2017),
    # which used ASTER DEMs to get mass balance of 92% of the HMA glaciers.
    print('\n\tModel Error (selectglaciersrgi): please choose an option'
          '\n\tthat exists for selecting glaciers. Exiting model run.\n')
    exit()


#----- STEP TWO: ADDITIONAL MODEL SETUP --------------------------------------
# Step two runs more functions related to the model setup. This section has been
#   separated from the selection of the model region/glaciers in order to
#   keep the input organized and easy to read.
# In step two, the model will:
#   > select glacier hypsometry
#   > define the model time frame
#   > define the initial surface type

# Glacier hypsometry
# main_glac_hyps = modelsetup.hypsometryglaciers(main_glac_rgi)
    # Note: need to adjust this hypsometry into separate functions such that it
    #       is easier to follow.
# AUTOMATE THIS TO LOAD THEM IN INSTEAD OF CHOOSING THEM
main_glac_hyps = pd.read_csv(hyps_filepath + 'RGI_13_area_test20170905.csv')

# Glacier initial ice thickness
# AUTOMATE THIS TO LOAD THEM IN INSTEAD OF CHOOSING THEM
main_glac_icethickness = pd.read_csv(hyps_filepath +
                                     'RGI_13_thickness_test20170905.csv')
# ADD OPTION FOR VOLUME-AREA SCALING

# Glacier total initial volume
main_glac_rgi['Volume'] = (
    (main_glac_hyps * main_glac_icethickness/1000).sum(axis=1))
    # volume [km3] = area[km2] * thickness[m] * (1 [km] / 1000 [m])

print(main_glac_rgi.head(),'\n')


# Model time frame
#   Set up table of dates. These dates are used as column headers for many other
#   variables in the model run, so it's important to be an initial step.
dates_table, start_date, end_date = modelsetup.datesmodelrun(option_leapyear)

# Initial surface type
main_glac_surftypeinit = modelsetup.surfacetypeglacinitial(
                                                option_surfacetype_initial,
                                                option_surfacetype_firn,
                                                option_surfacetype_debris,
                                                main_glac_rgi,
                                                main_glac_hyps)


#----- STEP THREE: CLIMATE DATA ----------------------------------------------
# Step three imports the climate data that will be used in the model run.
# Provide options for the selection and downscaling of the data
#    - default: nearest neighbor
#    - alternatives: weighted methods
#      (note: prior to any weighting, lapse rates/biases need to be applied)
# Important to consider the following:
#    - time period of the calibration data or model run
#    - type of model (DDF, EBM, etc.) will dictate meteorological data needed
#   Datasets:
#     Past: Default: ERA reanslysis?
#           Alternatives: COAWST (S.Nichols), NCEP/NCAR(?), others?
#                         automatic weather stations
#     Future: Default: GCMs (see glacierMIP project emails to download data)
#             Alternatives: COAWST (S.Nichols), others?
#
# In step three, the model will:
#   > import meteorological data
#   > select meteorological data for each glacier based on specified option
#       default: nearest neighbor
if option_gcm_downscale == 1:
    # OPTION 1 (default): NEAREST NEIGHBOR
    # Thoughts on 2017/08/21:
    #   > Pre-processing functions should be coded and added after the initial
    #     import such that the initial values can be printed if necessary.
    #   > Data imported here is monthly, i.e., it is 1 value per month. If the
    #     data is going to be subsampled to a daily resolution in order to
    #     estimate melt in areas with low monthly mean temperature as is done in
    #     Huss and Hock (2015), then those calculations should be performed in
    #     the ablation section.
    gcm_glac_temp = climate.importGCMvarnearestneighbor(gcm_temp_varname,
                                                        main_glac_rgi,
                                                        dates_table)
        # gcm nearest neighbor time series for each glacier with GlacNo index
        # rows = # of glaciers, cols = length of time series
    gcm_glac_prec = climate.importGCMvarnearestneighbor(gcm_prec_varname,
                                                        main_glac_rgi,
                                                        dates_table)
        # gcm nearest neighbor time series for each glacier with GlacNo index
        # rows = # of glaciers, cols = length of time series
    gcm_glac_elev = climate.importGCMfxnearestneighbor(gcm_elev_varname,
                                                       main_glac_rgi)
        # gcm nearest neighbor surface altitude for each glacier with GlacNo
        # index, rows = # of glaciers, cols = 1 (Series)
else:
    print('\n\tModel Error: please choose an option that exists for'
          '\n\tdownscaling climate data. Exiting model run now.\n')
    exit() # if you have an error, exit the model run


# ----- STEP FOUR: CALCULATE SPECIFIC MASS BALANCE --------------------------
# Step four computes the specific mass balance for every bin on each glacier
#   over the time span of the model run.
# In step four, the model will:
#   > set parameters or input calibrated parameters
#   > compute specific mass balance for each bin on each glacier (loop)
#
# Specify calibrated parameters:
""" THIS NEEDS TO BE SPECIFIED IN INPUT OR SEPARATE SECTION DEPENDING ON WHAT IS
    CONSIDERED TO BE THE CLEAREST/MOST ORGANIZED MANNER """
main_glac_parameters = main_glac_rgi.loc[:,['RGIId']]
main_glac_parameters['lr_gcm'] = -0.0065
    # lapse rate (K m-1) for gcm to glacier
main_glac_parameters['lr_glac'] = -0.0065
    # lapse rate (K m-1) on glacier for bins
main_glac_parameters['prec_factor'] = 3.0
    # precipitation correction factor (-)
        # k_p in Radic et al. (2013)
        # c_prec in Huss and Hock (2015)
main_glac_parameters['prec_grad'] = 0.0001
    # precipitation gradient on glacier (% m-1)
main_glac_parameters['DDF_ice'] = 7.2 * 10**-3
    # DDF ice (m w.e. d-1 degC-1)
    # note: '**' means to the power, so 10**-3 is 0.001
main_glac_parameters['DDF_snow'] = 4.0 * 10**-3
    # DDF snow (m w.e. d-1 degC-1)
main_glac_parameters['T_snow'] = 1.5
    # temperature threshold for snow (C)
        # Huss and Hock (2015) T_snow = 1.5 deg C with +/- 1 deg C for ratios
main_glac_parameters['DDF_firn'] = main_glac_parameters[['DDF_ice',
                                   'DDF_snow']].mean(axis=1)
    # DDF firn (m w.e. d-1 degC-1)
    # DDF_firn is average of DDF_ice and DDF_snow (Huss and Hock, 2015)
#
# Create dataframe for annual ELA for each glacier
main_glac_ELA = pd.DataFrame(0, columns=pd.date_range(str(startyear),
                             str(endyear+1),freq='A').strftime('%Y'),
                             index=main_glac_rgi.index)

# Compute the mass balance for each glacier (glacier by glacier)
    # Need to code: print out the desired output at the end of each loop
# for glac in range(len(main_glac_rgi)):
for glac in [0]:
    # Downscale the gcm temperature to each bin
    glac_bin_temp = massbalance.downscaletemp2bins(option_temp2bins,
                                                   option_elev_ref_downscale,
                                                   main_glac_rgi,
                                                   main_glac_hyps,
                                                   main_glac_parameters,
                                                   gcm_glac_temp,
                                                   gcm_glac_elev,
                                                   glac)
    # Downscale the gcm precipitation to each bin
    glac_bin_precsnow = massbalance.downscaleprec2bins(option_prec2bins,
                                                    option_elev_ref_downscale,
                                                    main_glac_rgi,
                                                    main_glac_hyps,
                                                    main_glac_parameters,
                                                    gcm_glac_prec,
                                                    gcm_glac_elev,
                                                    glac)
        # glac_bin_precsnow is the precipitation from the gcm for each elevation
        # bin, but has not been separated into precipitation and snow yet.
    # Compute accumulation (snow) and precipitation for each bin
    glac_bin_prec, glac_bin_snow = massbalance.accumulationbins(
                                                        option_accumulation,
                                                        glac_bin_temp,
                                                        glac_bin_precsnow,
                                                        main_glac_parameters,
                                                        glac)
    # Create dataframe for initial surface type
    glac_bin_surftype = massbalance.surfacetypebinsinitial(
                                                        main_glac_surftypeinit,
                                                        glac_bin_temp,
                                                        glac)
    # Create dataframe for snow accumulation on surface and other datasets that
    # need to be recorded
    glac_bin_snowonsurface = pd.DataFrame(0, index = glac_bin_temp.index,
                                          columns=glac_bin_temp.columns)
    glac_bin_melt_snow = pd.DataFrame(0, index = glac_bin_temp.index,
                                      columns=glac_bin_temp.columns)
    glac_bin_melt_surf = pd.DataFrame(0, index = glac_bin_temp.index,
                                      columns=glac_bin_temp.columns)
    glac_bin_massbal = pd.DataFrame(0, index = glac_bin_temp.index,
                                    columns=glac_bin_temp.columns)
    # Mask the variables such that computations are only done on bins that are
    # on the glacier, i.e., glac_bin_surftype != 0
    mask_offglacier = (glac_bin_surftype == 0)
    glac_bin_temp[mask_offglacier] = 0
    glac_bin_prec[mask_offglacier] = 0
    glac_bin_snow[mask_offglacier] = 0

    # Compute annual mean and sum of various datasets, which are needed for
    #   specific models (e.g., refreezing) and for accounting purposes
    """ NEED TO REDO THESE GROUPBY TO BE CONSISTENT WITH WATER YEAR
        (See Valentinas code for setting the start month based on latitude)"""
    # Annual mean air tempearture
    glac_bin_temp_annual = massbalance.groupbyyearmean(glac_bin_temp)
    # Annual total precipitation
    glac_bin_prec_annual = massbalance.groupbyyearsum(glac_bin_prec)
    # Annual total snow
    glac_bin_snow_annual = massbalance.groupbyyearsum(glac_bin_snow)
    # Annual surface type (needs to be updated each year)
    glac_bin_surftype_annual = massbalance.groupbyyearmean(glac_bin_surftype)
    # Annual glacier-bands mass balance
    glac_bin_massbal_annual = pd.DataFrame(0,index=glac_bin_temp_annual.index,
                                           columns=glac_bin_temp_annual.columns)
    # Annual glacier-bands area
    glac_bin_area_annual = pd.DataFrame(0,index=glac_bin_temp_annual.index,
                                           columns=glac_bin_temp_annual.columns)

    # Annual glacier-wide specific mass balance
    main_glac_massbal_annual = pd.DataFrame(0,index=main_glac_rgi.index,
                                           columns=glac_bin_temp_annual.columns)
    # Annual total glacier volume
    main_glac_volume_annual = pd.DataFrame(0,index=main_glac_rgi.index,
                                           columns=glac_bin_temp_annual.columns)
    main_glac_volume_annual.iloc[:,0] = main_glac_rgi['Volume']

    # Annual total glacier area
    main_glac_area_annual = pd.DataFrame(0,index=main_glac_rgi.index,
                                           columns=glac_bin_temp_annual.columns)
    main_glac_area_annual.iloc[:,0] = main_glac_rgi['Area']

    # Compute refreezing since this will affect the snow melt, which ultimately
    # will alter the surface type
    # Compute potential refreezing for each bin
    """Refreeze needs to be placed in annual loop, since it depends on surface
       type"""
    glac_bin_refreeze, glac_bin_refreeze_annual = (
                        massbalance.refreezingbins(option_refreezing,
                                                   glac_bin_temp_annual,
                                                   glac_bin_snow_annual,
                                                   glac_bin_surftype_annual,
                                                   glac_bin_temp)
                                                   )
        # Note: refreezing is a function of the air temperature, and annual
        #       surface type. therefore, it will not be affected by anything
        #       that is within the timestep loop and instead can be calculated
        #       outside the loop.
        #       Refreeze is currently treated as additional snow, i.e., it is
        #       added to the amount of snow on the surface in January of each
        #       year (option_refreezing == 2) and has to melt along with the
        #       rest of the snow before the underlying ice or firn melts.

    ###### testing loop through every timestep ##############################
    print('\n\n Testing new loop through every timestep:\n\n')
    # Loop through each timestep in order to carry over accumulation, which may
    # alter melt rates and surface type

    # for step in range(len(glac_bin_temp.iloc[0,:])):
    for step in range(0,26):
        print('Date:',glac_bin_temp.columns.values[step])
        # step_date = glac_bin_temp.columns.values[step]
            # date of the timestep
        # Extract data associated with each bin for each time step, which will
        # be used in the calculations that follow.  Extracting the data and
        # recording it at the end is most effective as it enables the use of
        # logical indexing on a column by column basis.
        series_temp = glac_bin_temp.iloc[:,step]
        series_prec = glac_bin_prec.iloc[:,step]
        series_snow = glac_bin_snow.iloc[:,step]
        series_surftype = glac_bin_surftype.iloc[:,step]
        series_refreeze = glac_bin_refreeze.iloc[:,step]
        series_daysinmonth = dates_table.loc[step,'daysinmonth']
        # Calculate the accumulation of snow on the surface
        if step == 0:
            series_snowonsurface = series_snow + series_refreeze
        else:
            series_snowonsurface = (glac_bin_snowonsurface.iloc[:,step-1] +
                                    series_snow + series_refreeze)
        # Calculate the melt of snow and ice/firn if all the snow melted
        # Set empty series that will be used
        series_melt_snow = pd.Series(0, index=series_temp.index)
        series_melt_surf = pd.Series(0, index=series_temp.index)
        series_Emelt_remaining = pd.Series(0, index=series_temp.index)

        series_Emelt_available = series_temp * series_daysinmonth
            # Calculate the energy (E) available for melting snow/ice/firn
        """ DEVELOPER'S NOTE: HERE IS WHERE THE OPTION TO INCORPORATE THE
            DAILY STDEV TO ESTIMATE MONTHLY MELT SHOULD BE APPLIED """
        series_Emelt_available[series_Emelt_available < 0] = 0
        # Calculate energy required to melt all the snow
        series_Emelt_snowonsurface_max = (series_snowonsurface /
                                main_glac_parameters.loc[glac,'DDF_snow'])
        # Calculate the melt of snow
        # if energy available is greater than energy required to melt all
        # the snow on the surface, then all the snow melts and there is
        # energy leftover to melt
        mask1 = (series_Emelt_available >= series_Emelt_snowonsurface_max)
        series_melt_snow[mask1] = series_snowonsurface[mask1]
        series_snowonsurface[mask1] = 0
        series_Emelt_remaining[mask1] = (series_Emelt_available[mask1] -
                                         series_Emelt_snowonsurface_max[mask1])
        mask2 = ((series_Emelt_remaining > 0) & (series_surftype == 1))
            # energy left over and surface type is ice
        series_melt_surf[mask2] = (series_Emelt_remaining[mask2] *
                                   main_glac_parameters.loc[glac,'DDF_ice'])
            # surface melt is considered all melt beneath the snow and is
            # dependent on the surface beneath the snow
        if option_surfacetype_firn == 0:
            mask3 = ((series_Emelt_remaining > 0) & (series_surftype == 2))
                # energy left over and surface type is snow
            series_melt_surf[mask3] = (series_Emelt_remaining[mask3] *
                                      main_glac_parameters.loc[glac,'DDF_snow'])
        elif option_surfacetype_firn == 1:
            mask4 = ((series_Emelt_remaining > 0) & (series_surftype == 3))
                # energy left over and surface type is firn
            series_melt_surf[mask4] = (series_Emelt_remaining[mask4] *
                                      main_glac_parameters.loc[glac,'DDF_firn'])
        if option_surfacetype_debris == 1:
            mask5 = ((series_Emelt_remaining > 0) & (series_surftype == 4))
                # energy left over and surface type is debris
            series_melt_surf[mask5] = (series_Emelt_remaining[mask5] *
                                    main_glac_parameters.loc[glac,'DDF_debris'])

        # if energy available is less than energy required to melt all the snow,
        # then only some of the snow melts (and no ice/firn melts)
        mask6 = ((series_Emelt_available < series_Emelt_snowonsurface_max) &
                 (series_Emelt_available > 0))
        series_melt_snow[mask6] = (series_Emelt_available[mask6] *
                                   main_glac_parameters.loc[glac,'DDF_snow'])
        series_snowonsurface[mask6] = (series_snowonsurface[mask6] -
                                       series_melt_snow[mask6])
        series_Emelt_remaining[mask6] = 0
        # Calculate the specific mass balance for each time step
        series_massbal = series_snow + series_refreeze - series_melt_surf
        # Update the snow on the surface for each time step.  This is needed to
        # keep track of snow accumulation for the next time step.
        glac_bin_snowonsurface.iloc[:,step] = series_snowonsurface
        # Record additional datasets of interest
        glac_bin_melt_snow.iloc[:,step] = series_melt_snow
        glac_bin_melt_surf.iloc[:,step] = series_melt_surf
        glac_bin_massbal.iloc[:,step] = series_massbal
        # print(series_Emelt_snowonsurface_max[460:470])
        # print(series_Emelt_available.iloc[460])
        # print(series_melt_snow.iloc[460])
        # print(series_melt_surf.iloc[460])
        # print('Air_Temp:\n', glac_bin_temp.iloc[460:470,0:10])
        # print('Snow on surface:\n',glac_bin_snowonsurface.iloc[460:470,0:10])
        # print('Surface melt:\n',glac_bin_melt_surf.iloc[460:470,0:12])
        # print('Mass balance:\n',glac_bin_massbal.iloc[460:470,step])

        if timestep == 'monthly':
            annual_divisor = 12
        elif timestep == 'daily':
            print('Need to write according to leapyear.  Code this.'
                  'Exiting now.')
            exit()

        if (step + 1) % annual_divisor == 0:
            # Note: "%" gives the remainder.  Since step starts at 0, give it
            #       plus 1 such that for the regular calendar year, December has
            #       a remainder of 0.
            year_position = int((step + 1)/12 - 1)
                # track the position for indexing associated with each year
            print("\n\t\tCOMPUTING ANNUAL CALCULATIONS\n")
            # Various outputs:
            # print('Snow on surface:\n',glac_bin_snowonsurface.iloc[460:470,12:24])
            # print('Snow melt:\n',glac_bin_melt_snow.iloc[460:470,12:24])
            # print('Air_Temp:\n', glac_bin_temp.iloc[460:470,12:24])
            # print('Surface melt:\n',glac_bin_melt_surf.iloc[460:470,12:24])
            # print('Mass balance:\n',glac_bin_massbal.iloc[460:470,12:24])

            # Compute specific annual mass balance for each bin
            glac_bin_massbal_annual.iloc[:,year_position] = (
                glac_bin_massbal.iloc[:,(step-11):(step+1)].sum(axis=1))

            # Compute specific annual glacier-wide mass balance
            if main_glac_area_annual.iloc[glac,year_position] > 0:
                main_glac_massbal_annual.iloc[glac,year_position] = (
                    (glac_bin_massbal_annual.iloc[:,year_position] *
                     main_glac_hyps.iloc[glac,:].T).sum() /
                     main_glac_area_annual.iloc[glac,year_position])
                     # b_a,glac = sum(b_a,bin * area_bin) / area_total
            # !!!!Note the area is changing, so need to adjust here!!!

            # Update volume of next year
            if (main_glac_area_annual.iloc[glac,year_position]) > 0:
                main_glac_volume_annual.iloc[glac,year_position + 1] = (
                    main_glac_volume_annual.iloc[glac,year_position] +
                    (main_glac_massbal_annual.iloc[glac,year_position]/1000 *
                    main_glac_area_annual.iloc[glac,year_position]))
                # V(t+1)[km3] = V(t)[km3] + (B_annual[m] * (1 [km] / 1000 [m])
                #                           * Area[km2])

            # Update area for the next year based on new volume
            # Note: this will fail for the very last year, so need to adjust
            #       this.  Also, need to change from constant area to
            #       some sort of scaling mechanism
            #   > Options:
            #       - Volume-length scaling
            #       - Volume-area scaling
            #       - Volume-length-area scaling
            #       - Normalized empirical adjustments (Huss and Hock, 2015)
            main_glac_area_annual.iloc[glac, year_position + 1] = (
                    main_glac_area_annual.iloc[glac, year_position])

            print('main_glac_massbal_annual:\n'),
            print(main_glac_massbal_annual.iloc[:, 0:5])
            print('\nmain_glac_volume_annual:\n')
            print(main_glac_volume_annual.iloc[:, 0:5])


        # PROPERLY RECORDING THINGS.
        # > COMPUTE ANNUAL MASS BALANCE AS THIS WILL CHANGE VOLUME
        #   FOR VOLUME-LENGTH SCALING
        # > APPLY VOLUME-LENGTH SCALING
        # > UPDATE SURFACE TYPE
        # note: use loop after every december to calculate and update these things



#         # NEXT STEPS:
#         # > code for glacier dynamics
#         #       Glacier volume-length scaling?
#         # Note: THIS SHOULD BE BEFORE SURFACE TYPE IN CASE WE ARE USING THE
#         #       MEAN HEIGHT TO DEFINE THE ELA LIKE VALENTINA DID
#
#         # for every year do the following:
#         #   > ablation
#         #   > accumulation
#         #   > refreezing
#         #   > scaling
#         #   > surface type
# print("\n\nNEED TO CODE: How does snow and refreezing roll over from "
#       "one year to the next??\n\n")

# ----- STEP SIX: ICE DYNAMICS
# volume-length or volume-area scaling, 1-D flow line / shallow flow line
# models

# ----- PRINT DESIRED OUTPUT
    # Model output: It's important to output everything at the
    # yearly/monthly/daily scale to provide the option of outputting various
    # data of interest.

#----- STEP X: MODEL CALIBRATION ---------------------------------------------
# The products that are being used for calibration are going to dictate the
# meteorological data that will be used in this code.  For example, for
# geodetic mass balances the dates of the DEMs are going to drive the time
# period that is being modeled.  The same is true for any GRACE products,
# stake measurements, etc.  If multiple products are being used for
# calibration, select a time period that spans all of the products.
#
# Calibration options:
#  - Geodetic mass balances
#       DEMs from HiMAT team
#         > 1960/70s, Hexagon, S.Rupper's team
#         > 2006/07, Cartosat, D.Shean
#         > 2000-15, WorldView/GeoEye, D.Shean
#         > 2000-15, ASTER, P.Montesano (quality is uncertain at present)
#         > 2000-16, Brun etal (2017) - ASTER DEMs
#  - GRACE products (I.Velicogna or A.Arendt's team)
#  - In-situ measurements (ICIMOD may have data available)
#  - World Glacier Inventory by Cogley (2009) that has been used by Radic
#    et al. (2011, 2013) and Bliss et al. (2014) - 36 glaciers with observed
#    seasonal mass balance profiles.  Only 0-2 may be relevant for HMA.
#  - Combination (e.g., Gardner et al. 2013)
#  - Snow extent/snow line altitude (products being generated by Batu's team?)
#
# ??? [Insert code] ???
# Is the current model run being used for calibration or not?
#   > If not (no), then enter the time period that will be modeled.
#   > If yes, then select the calibration product.  This is going to dictate
#             the time period and the areal extent (e.g., GRACE products will
#             will require a group of glaciers together, while geodetic mass
#             balances can be done on a glacier by glacier basis).  This is
#             going to set the loop that is being used for the calibration.
#
# MAIN QUESTION_1
# At what level should the parameters be calibrated?  This is going to depend
# on the calibration data available, but the model should be given options:
# Option 1 - subregion level
#   > Calibrated parameters are the same over the entire region.
#   > Mass balance at each glacier/bin will differ at the glacier/bin level
#     because parameters are getting good agreement at broad level.  This will
#     make it interesting to look at glaciers that performed poorly and try to
#     understand the cause of the poor performance.
#   > Good for GRACE or other (sub)regional products in addition to geodetic
#     mass balances.
# Option 2 - glacier level
#   > Mass balance will agree well with each glacier.
#   > Calibrated parameters will differ for each glacier in the region.  In
#     contrast to Option 1, glaciers where we see large changes in parameters
#     will indicate that something on that specific glacier isn't performing
#     properly.  For example, the temperature may be too hot or cold, or the
#     glacier could be receiving too much or too little precipitation/snow.
# If BOTH of these options are done, it should reveal exactly what is causing
# some glaciers to not agree well with the mass balance estimates.
#
# MAIN QUESTION_2
# Can we use "dependencies" to reduce the number of calibrated parameters?
#   ex. f_ice should be related to f_snow based on differences in albedos
#
# MAIN QUESTION_3
# Do we have enough data that we can calibrate these parameters physically?
#   ex. lr_era calibrated by GCM and air temperature at weather stations
#   ex. f_snow/prec based on snow extent each year
#   ex. T_snow - could just be held based on physical reality
#           (<0 = snow, 1 degC = 50/50 snow/rain, 2+ degC = rain)
#
# MAIN QUESTION_4
# Do we have enough data that we can calibrate something realistic at the
# glacier level?  Would those parameters hold over a localized region (e.g.,
# Everest Region)?  Let's try this once the model is fully developed.
#> I'm thinking here that if we were to allow the DDF_debris
#  to vary over each elevation bin, then it should reflect the debris thickness,
#  and if it reflects the debris thickness, then we're really starting to tune
#  the model to an individual glacier.
#
# STRUCTURAL CONSIDERATION
# Provide enough freedom/flexibility in the model structure such that the
# parameters can be calibrated at different levels, e.g., hold the downscaling
# lr_era, and kp constant over the region, but then the others can vary.
