"""
pygemfxns_modelsetup.py is a list of functions that are used to set up the model with the required input for the main
script.
"""
#========= LIST OF PACKAGES ==================================================
import pandas as pd
import numpy as np
import os # os is used with re to find name matches
import re # see os
from datetime import datetime

#========= IMPORT COMMON VARIABLES FROM MODEL INPUT ==========================
import pygem_input as input

#========= DESCRIPTION OF VARIABLES (alphabetical order) =====================
    # glac_hyps - table of hypsometry for all the glaciers
    # glac_table - main table of glaciers in model run with RGI information
    # option_fxn - function option (see specifics within each function)

#========= FUNCTIONS (alphabetical order) ===================================
def datesmodelrun():
    """
    Set up a table using the start and end year that has the year, month, day, water year, and number of days in the 
    month.
    
    Output is a Pandas DataFrame with a table of dates (rows = timesteps, columns = timestep attributes), as well as 
    the start date, and end date of the model run.  These two things are useful for grabbing the correct climate data.
    
    Function Options:
    - option_wateryear:
        > 1 (default) - use water year
        > 2 - use calendar year
    -  option_leapyear:
        > 1 (default) - leap years are included
        > 2 - leap years are excluded (February always has 28 days)
        
    Developer's note: ADD OPTIONS FOR CHANGING WATER YEAR FROM OCT 1 - SEPT 30 FOR VARIOUS REGIONS
    """

    # Include spinup time in start year
    startyear_wspinup = input.startyear - input.spinupyears
    # Convert start year into date depending on option_wateryear
    if input.option_wateryear == 1:
        startdate = str(startyear_wspinup) + '-10-01'
        enddate = str(input.endyear) + '-09-30'
    elif input.option_wateryear == 0:
        startdate = str(startyear_wspinup) + '-01-01'
        enddate = str(input.endyear) + '-12-31'
    else:
        print("\n\nError: Please select an option_wateryear that exists. Exiting model run now.\n")
        exit()
    # Convert input format into proper datetime format
    startdate = datetime(*[int(item) for item in startdate.split('-')])
    enddate = datetime(*[int(item) for item in enddate.split('-')])
    if input.timestep == 'monthly':
        startdate = startdate.strftime('%Y-%m')
        enddate = enddate.strftime('%Y-%m')
    elif input.timestep == 'daily':
        startdate = startdate.strftime('%Y-%m-%d')
        enddate = enddate.strftime('%Y-%m-%d')
    # Generate dates_table using date_range function
    if input.timestep == 'monthly':
        # Automatically generate dates from start date to end data using a monthly frequency (MS), which generates
        # monthly data using the 1st of each month 
        dates_table = pd.DataFrame({'date' : pd.date_range(startdate, enddate, freq='MS')})
        # Select attributes of DateTimeIndex (dt.year, dt.month, and dt.daysinmonth)
        dates_table['year'] = dates_table['date'].dt.year
        dates_table['month'] = dates_table['date'].dt.month
        dates_table['daysinmonth'] = dates_table['date'].dt.daysinmonth
        dates_table['timestep'] = np.arange(len(dates_table['date']))
        # Set date as index
        dates_table.set_index('timestep', inplace=True)
        # Remove leap year days if user selected this with option_leapyear
        if input.option_leapyear == 0:
            mask1 = (dates_table['daysinmonth'] == 29)
            dates_table.loc[mask1,'daysinmonth'] = 28
    elif input.timestep == 'daily':
        # Automatically generate daily (freq = 'D') dates
        dates_table = pd.DataFrame({'date' : pd.date_range(startdate, enddate, freq='D')})
        # Extract attributes for dates_table
        dates_table['year'] = dates_table['date'].dt.year
        dates_table['month'] = dates_table['date'].dt.month
        dates_table['day'] = dates_table['date'].dt.day
        dates_table['daysinmonth'] = dates_table['date'].dt.daysinmonth
        # Set date as index
        dates_table.set_index('date', inplace=True)
        # Remove leap year days if user selected this with option_leapyear
        if input.option_leapyear == 0:
            # First, change 'daysinmonth' number
            mask1 = dates_table['daysinmonth'] == 29
            dates_table.loc[mask1,'daysinmonth'] = 28
            # Next, remove the 29th days from the dates
            mask2 = ((dates_table['month'] == 2) & (dates_table['day'] == 29))
            dates_table.drop(dates_table[mask2].index, inplace=True)
    else:
        print("\n\nError: Please select 'daily' or 'monthly' for gcm_timestep. Exiting model run now.\n")
        exit()
    # Add column for water year
    # Water year for northern hemisphere using USGS definition (October 1 - September 30th),
    # e.g., water year for 2000 is from October 1, 1999 - September 30, 2000
    dates_table['wateryear'] = dates_table['year']
    for step in range(dates_table.shape[0]):
        if dates_table.loc[step, 'month'] >= 10:
            dates_table.loc[step, 'wateryear'] = dates_table.loc[step, 'year'] + 1
    # Add column for seasons
    # create a season dictionary to assist groupby functions
    seasondict = {}
    month_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    season_list = []
    for i in range(len(month_list)):
        if (month_list[i] >= input.summer_month_start and month_list[i] < input.winter_month_start):
            season_list.append('summer')
            seasondict[month_list[i]] = season_list[i]
        else:
            season_list.append('winter')
            seasondict[month_list[i]] = season_list[i]
    dates_table['season'] = dates_table['month'].apply(lambda x: seasondict[x])
    # Extract monthly columns
    monthly_columns = dates_table['date']
    # Extract annual columns
    if input.option_wateryear == 1:
        annual_columns = np.arange(dates_table.loc[0,'wateryear'], dates_table.loc[dates_table.shape[0]-1,'wateryear'] 
                                   + 1)
    elif input.option_wateryear == 0:
        annual_columns = np.arange(dates_table.loc[0,'year'], dates_table.loc[dates_table.shape[0]-1,'year'] + 1)
    # Compute annual divisor (used to perform calculations)
    if input.timestep == 'monthly':
        annual_divisor = 12
    elif input.timestep == 'daily':
        print('Need to write according to leapyear.  Code this. Exiting now.')
        exit()
    return dates_table, startdate, enddate, monthly_columns, annual_columns, annual_divisor
    # note from previous version:
        # dates = dates.to_series().apply(lambda x: x.strftime("%Y-%m"))
        # removes the -DD such that its only YYYY-MM


def hypsmassbalDShean(glac_table):
    """
    Select hypsometry and mass balance of all glaciers that are being used in
    the model from the csv files David Shean is producing. This function returns
    hypsometry and mass balance for each elevation bin.
    """
    # Create an empty dataframe. main_glac_hyps will store the hypsometry of
    # all the glaciers with the bin size specified by user input. Set index to
    # be consistent with main_glac_rgi as well as 'RGIId'
    col_bins = (np.arange(int(input.binsize/2),9001,input.binsize))
        # creating all columns from 0 to 9000 within given bin size, which
        # enables this to be used for every glacier in the world
    glac_hyps = pd.DataFrame(index=glac_table.index, columns=col_bins)
        # rows of table will be the glacier index
        # columns will be the center elevation of each bin
    glac_mb = glac_hyps.copy()
        # geodetic mass balance will also be extracted
    for glac in range(len(glac_hyps)):
        # Select full RGIId string, which needs to be compatible with mb format
        hyps_ID_full = glac_table.loc[glac,'RGIId']
        # Remove the 'RGI60-' from the name to agree with David Shean's
        # naming convention (This needs to be automatized!)
        hyps_ID_split = hyps_ID_full.split('-')
        # Choose the end of the string
        hyps_ID_short = hyps_ID_split[1]
        hyps_ID_findname = hyps_ID_short + '_mb_bins.csv'
        for hyps_file in os.listdir(input.hyps_filepath):
            # For all the files in the give directory (hyps_fil_path) see if
            # there is a match.  If there is, then geodetic mass balance is
            # available for calibration.
            if re.match(hyps_ID_findname, hyps_file):
                hyps_ID_fullfile = (input.hyps_filepath + '/' + hyps_file)
                hyps_ID_csv = pd.read_csv(hyps_ID_fullfile)
        # Insert the hypsometry data into the main hypsometry row
        for nrow in range(len(hyps_ID_csv)):
            # convert elevation bins into integers
            elev_bin = int(hyps_ID_csv.loc[nrow,'# bin_center_elev'])
            # add bin count to each elevation bin (or area)
            glac_hyps.loc[glac,elev_bin] = hyps_ID_csv.loc[nrow,'bin_count']
            glac_mb.loc[glac,elev_bin] = hyps_ID_csv.loc[nrow,'mb_bin_med']
    # Fill NaN values with 0
    glac_hyps.fillna(0, inplace=True)
    glac_mb.fillna(0, inplace=True)
    print("The 'hypsmassbalDShean' function has finished.")
    return glac_hyps, glac_mb


def hypsometrystats(hyps_table, thickness_table):
    """Calculate the volume and mean associated with the hypsometry data.
    
    Output is a series of the glacier volume [km**3] and mean elevation values [m a.s.l.]. 
    """
    # Glacier volume [km**3]
    glac_volume = (hyps_table * thickness_table/1000).sum(axis=1)
    # Glacier area [km**2]
    glac_area = hyps_table.sum(axis=1)
    # Mean glacier elevation
    glac_hyps_mean = round(((hyps_table.multiply(hyps_table.columns.values, axis=1)).sum(axis=1))/glac_area).astype(int)
    # Median computations
#    main_glac_hyps_cumsum = np.cumsum(hyps_table, axis=1)
#    for glac in range(hyps_table.shape[0]):
#        # Median glacier elevation
#        # Computed as the elevation when the normalized cumulative sum of the glacier area exceeds 0.5 (50%)
#        series_glac_hyps_cumsumnorm = main_glac_hyps_cumsum.loc[glac,:].copy() / glac_area.iloc[glac]
#        series_glac_hyps_cumsumnorm_positions = (np.where(series_glac_hyps_cumsumnorm > 0.5))[0]
#        glac_hyps_median = main_glac_hyps.columns.values[series_glac_hyps_cumsumnorm_positions[0]]
#    NOTE THERE IS A 20 m (+/- 5 m) OFFSET BETWEEN THE 10 m PRODUCT FROM HUSS AND THE RGI INVENTORY """
    return glac_volume, glac_hyps_mean


def import_hypsometry(rgi_table):
    """Use the hypsometry dictionary specified by the user in the input to extract the correct hypsometry.
    The hypsometry files must be in the proper units of square kilometers [km**2] and need to be pre-processed to have 
    all bins between 0 - 9000 m.
    
    Output is a Pandas DataFrame of the hypsometry for all the glaciers in the model run
    (rows = GlacNo, columns = elevation bins).
    
    Line Profiling: Loading in the table takes the most time (~2.3 s)
    """
    ds = pd.read_csv(input.hyps_filepath + input.hypsfile_dict[input.rgi_regionsO1[0]])
    # Select glaciers based on 01Index value from main_glac_rgi table
    glac_hyps_table = pd.DataFrame()
    for glacier in range(len(rgi_table)):
        if glac_hyps_table.empty:
            glac_hyps_table = ds.loc[rgi_table.loc[glacier,'O1Index']]
        else:
            glac_hyps_table = pd.concat([glac_hyps_table, ds.loc[rgi_table.loc[glacier,'O1Index']]], axis=1)
    glac_hyps_table = glac_hyps_table.transpose()
    # Clean up table and re-index
    # Reset index to be GlacNo
    glac_hyps_table.reset_index(drop=True, inplace=True)
    glac_hyps_table.index.name = input.indexname
    # Drop columns that are not elevation bins
    glac_hyps_table.drop(input.hyps_cols_drop, axis=1, inplace=True)
    # Make sure columns are integers
    glac_hyps_table.columns = glac_hyps_table.columns.values.astype(int)
    # Change NAN from -99 to 0
    glac_hyps_table[glac_hyps_table==-99] = 0.
    return glac_hyps_table


def import_icethickness(rgi_table):
    """Use the thickness dictionary specified by the user in the input to extract the ice thickness.
    The ice thickness files must be in the proper units [m] and need to be pre-processed to have all bins
    between 0 - 9000 m.
    
    Output is a Pandas DataFrame of the ice thickness for all the glaciers in the model run
    (rows = GlacNo, columns = elevation bins).
    
    Line Profiling: Loading in the table takes the most time (~2.3 s)
    """
    ds = pd.read_csv(input.thickness_filepath + input.thicknessfile_dict[input.rgi_regionsO1[0]])
    # Select glaciers based on 01Index value from main_glac_rgi table
    glac_thickness_table = pd.DataFrame()
    for glacier in range(len(rgi_table)):
        if glac_thickness_table.empty:
            glac_thickness_table = ds.loc[rgi_table.loc[glacier,'O1Index']]
        else:
            glac_thickness_table = pd.concat([glac_thickness_table, ds.loc[rgi_table.loc[glacier,'O1Index']]], axis=1)
    glac_thickness_table = glac_thickness_table.transpose()
    # Clean up table and re-index
    # Reset index to be GlacNo
    glac_thickness_table.reset_index(drop=True, inplace=True)
    glac_thickness_table.index.name = input.indexname
    # Drop columns that are not elevation bins
    glac_thickness_table.drop(input.thickness_cols_drop, axis=1, inplace=True)
    # Make sure columns are integers
    glac_thickness_table.columns = glac_thickness_table.columns.values.astype(int)
    # Change NAN from -99 to 0
    glac_thickness_table[glac_thickness_table==-99] = 0.
    return glac_thickness_table


def importHussfile(filename):
    filepath = input.hyps_filepath + 'bands_' + str(input.binsize) + 'm_DRR/'
    fullpath = filepath + filename
    data_Huss = pd.read_csv(fullpath)
    # # THIS SECTION CHANGES THE RGIID TO MATCH RGI FORMAT
    # ID_split = data_Huss['RGIID'].str.split('.').apply(pd.Series).loc[:,2]
    #     # Grabs the value ex. "13-00001", but need to replace '-' with '.'
    #     # use .apply(pd.Series) to turn the tuples into a DataFrame,
    #     # then access normally
    # ID_split2 = ID_split.str.split('-').apply(pd.Series)
    #
    # ID_RGI = 'RGI60-' + ID_split2.loc[:,0] + '.' + ID_split2.loc[:,1]
    # print(ID_RGI.head())
    # # OUTPUT TO CSV
    # ID_RGI.to_csv('RGIIDs_13.csv')

    # Create new dataframe with all of it
    bins = np.arange(int(input.binsize/2),9000,input.binsize)
        # note provide 3 rows of negative numbers to rename columns
    data_table = pd.DataFrame(0, index=data_Huss.index, columns=bins)
    data_Huss.drop(['RGIID'], axis=1, inplace=True)
    columns_int = data_Huss.columns.values.astype(int)
        # convert columns from string to integer such that they can be used
        # in downscaling computations
    data_Huss.columns = columns_int
    data_Huss = data_Huss.astype(float)
    mask1 = (data_Huss == -99)
    data_Huss[mask1] = 0
    print(len(data_Huss.iloc[0]))
    print(data_Huss.iloc[:,0:760])
    # data_Huss.iloc[0,0] = 50
    # data_table.iloc[0,3] = data_Huss.iloc[0,0]
    # print(data_table)
    # exit()
    print('hello')
    data_table.iloc[:,3:(3+(len(data_Huss.iloc[0])))] = data_Huss.iloc[:,:]
    # print(data_table)
    print('again')
    data_table.to_csv('RGI_13_thickness.csv')
    # Note: 7625 elevation did not have a value of 0, but was blank


def selectglaciersrgitable():
    """
    Select all glaciers to be used in the model run according to the regions and glacier numbers defined by the RGI 
    glacier inventory. This function returns the rgi table associated with all of these glaciers.
    
    Output: Pandas DataFrame of the glacier statistics for each glacier in the model run
    (rows = GlacNo, columns = glacier statistics)
    """
    # Glacier Selection Options:
#   > 1 (default) - enter numbers associated with RGI V6.0 and select
#                   glaciers accordingly
#   > 2 - glaciers/regions selected via shapefile
#   > 3 - glaciers/regions selected via new table (other inventory)

    # Create an empty dataframe
    glacier_table = pd.DataFrame()
    for x_region in input.rgi_regionsO1:
        csv_regionO1 = pd.read_csv(input.rgi_filepath + input.rgi_dict[x_region])
        # Populate glacer_table with the glaciers of interest
        if input.rgi_regionsO2 == 'all' and input.rgi_glac_number == 'all':
            print(f"\nAll glaciers within region(s) {input.rgi_regionsO1} are included in this model run.")
            if glacier_table.empty:
                glacier_table = csv_regionO1
            else:
                glacier_table = pd.concat([glacier_table, csv_regionO1], axis=0)
        elif input.rgi_regionsO2 != 'all' and input.rgi_glac_number == 'all':
            print(f"\nAll glaciers within subregion(s) {input.rgi_regionsO2} in region {input.rgi_regionsO1} "
                  "are included in this model run.")
            for x_regionO2 in input.rgi_regionsO2:
                if glacier_table.empty:
                    glacier_table = (csv_regionO1.loc[csv_regionO1['O2Region'] == x_regionO2])
                else:
                   glacier_table = (pd.concat([glacier_table, csv_regionO1.loc[csv_regionO1['O2Region'] == x_regionO2]],
                                              axis=0))
        else:
            print(f"\nThis study is only focusing on glaciers {input.rgi_glac_number} in region "
                  f"{input.rgi_regionsO1}.")
            for x_glac in input.rgi_glac_number:
                glac_id = ('RGI60-' + str(input.rgi_regionsO1)[1:-1] + '.' + x_glac)
                if glacier_table.empty:
                    glacier_table = (csv_regionO1.loc[csv_regionO1['RGIId'] == glac_id])
                else:
                    glacier_table = (pd.concat([glacier_table, csv_regionO1.loc[csv_regionO1['RGIId'] == glac_id]], 
                                               axis=0))
    glacier_table.reset_index(inplace=True)
        # reset the index so that it is in sequential order (0, 1, 2, etc.)
    glacier_table.rename(columns={'index': 'O1Index'}, inplace=True)
        # change old index to 'O1Index' to be easier to recall what it is
    glacier_table_copy = glacier_table.copy()
        # must make copy; otherwise, drop will cause SettingWithCopyWarning
    glacier_table_copy.drop(input.rgi_cols_drop, axis=1, inplace=True)
        # drop columns of data that is not being used
    glacier_table_copy.index.name = input.indexname
    return glacier_table_copy        
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


# EXAMPLE CODE OF SEARCHING FOR A FILENAME
#def selectglaciersrgitable_old():
#    """
#    The upper portion of this code was replaced by a dictionary based on the user input to speed up computation time.
#    This has been kept as an example code of searching for a filename.
#    """
#    # Select glaciers according to RGI V60 tables.
#    glacier_table = pd.DataFrame()
#    for x_region in input.rgi_regionsO1:
#        # print(f"\nRegion: {x_region}")
#        rgi_regionsO1_findname = str(x_region) + '_rgi60_'
#        # print(f"Looking for region {x_region} filename...")
#        # Look up the RGI tables associated with the specific regions
#        # defined above and concatenate all the areas of interest into a
#        # single file
#        for rgi_regionsO1_file in os.listdir(input.rgi_filepath):
#            if re.match(rgi_regionsO1_findname, rgi_regionsO1_file):
#                # if a match for the region is found, then open the file and
#                # select the subregions and/or glaciers from that file
#                rgi_regionsO1_fullfile = input.rgi_filepath + rgi_regionsO1_file
#                csv_regionO1 = pd.read_csv(rgi_regionsO1_fullfile)


def surfacetypeglacinitial(glac_table, glac_hyps):
    """
    Define initial surface type according to median elevation such that the
    melt can be calculated over snow or ice.
    Convention:
        1 - ice
        2 - snow
        3 - firn
        4 - debris
        0 - off-glacier
        
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
    
    Developer's note: need to add debris maps and determine how DDF_debris will be included.
    
    Output: Pandas DataFrame of the initial surface type for each glacier in the model run
    (rows = GlacNo, columns = elevation bins)
    """
    glac_surftype = glac_hyps.copy()
    series_elev = glac_surftype.columns.values
    for glac in range(glac_surftype.shape[0]):
        # Option 1 - initial surface type based on the median elevation
        if input.option_surfacetype_initial == 1:
            glac_surftype.loc[glac, :][(series_elev < glac_table.loc[glac, 'Zmed']) & (glac_hyps.loc[glac, :] > 0)] = 1
            glac_surftype.loc[glac, :][(series_elev >= glac_table.loc[glac, 'Zmed']) & (glac_hyps.loc[glac, :] > 0)] = 2
        # Option 2 - initial surface type based on the mean elevation
        elif input.option_surfacetype_initial ==2:
            glac_surftype.loc[glac, :][(series_elev < glac_table.loc[glac, 'Zmean']) & (glac_hyps.loc[glac, :] > 0)] = 1
            glac_surftype.loc[glac, :][(series_elev >= glac_table.loc[glac, 'Zmean']) & (
                    glac_hyps.loc[glac, :] > 0)] = 2
        else:
            print("This option for 'option_surfacetype' does not exist. Please choose an option that exists. "
                  + "Exiting model run.\n")
            exit()
    # If firn is included, then specify initial firn conditions
    if input.option_surfacetype_firn == 1:
        glac_surftype[glac_surftype == 2] = 3
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
    # Make sure surface type is integer values
    glac_surftype = glac_surftype.astype(int)
    return glac_surftype


def surfacetypeDDFdict():
    """
    Create a dictionary of surface type and its respective DDF
    Convention: [0=off-glacier, 1=ice, 2=snow, 3=firn, 4=debris]
    """
    surfacetype_ddf_dict = {
            1: input.DDF_ice,
            2: input.DDF_snow}
    if input.option_surfacetype_firn == 1:
        surfacetype_ddf_dict[3] = input.DDF_firn
    if input.option_surfacetype_debris == 1:
        surfacetype_ddf_dict[4] = input.DDF_debris
    return surfacetype_ddf_dict
