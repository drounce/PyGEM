"""
fxns_modelsetup.py is a list of functions that are used with the model setup
for PyGEM.
"""
#========= LIST OF PACKAGES ==================================================
import pandas as pd
import numpy as np
import os # os is used with re to find name matches
import re # see os
from datetime import datetime

#========= IMPORT COMMON VARIABLES FROM MODEL INPUT ==========================
from pygem_input import *
    # import all data

#========= DESCRIPTION OF VARIABLES (alphabetical order) =====================
    # glac_hyps - table of hypsometry for all the glaciers
    # glac_table - main table of glaciers in model run with RGI information
    # option_fxn - function option (see specifics within each function)

#========= FUNCTIONS (alphabetical order) ===================================
def datesmodelrun(option_wateryear, option_leapyear):
    """
    Set up a table using the start and end date that has the year, month, day,
    and number of days in the month.
    Developer's note: ADD OPTIONS FOR CHANGING WATER YEAR FROM OCT 1 - SEPT 30 FOR VARIOUS REGIONS
    """
    # Function Options:
    #  option_wateryear:
    #   > 1 (default) - use water year
    #   > 2 - use calendar year
    #  option_leapyear:
    #   > 1 (default) - leap years are included
    #   > 2 - leap years are excluded (February always has 28 days)
    #
    # Include spinup time in start year
    startyear_wspinup = startyear - spinupyears
    # Convert start year into date depending on option_wateryear
    if option_wateryear == 1:
        startdate = str(startyear) + '-10-01'
        enddate = str(endyear) + '-09-30'
    elif option_wateryear == 0:
        startdate = str(startyear) + '-01-01'
        enddate = str(endyear) + '-12-31'
    else:
        print("\n\nError: Please select an option_wateryear that exists. Exiting model run now.\n")
        exit()
    # Convert input format into proper datetime format
    startdate = datetime(*[int(item) for item in startdate.split('-')])
    enddate = datetime(*[int(item) for item in enddate.split('-')])
    if timestep == 'monthly':
        startdate = startdate.strftime('%Y-%m')
        enddate = enddate.strftime('%Y-%m')
    elif timestep == 'daily':
        startdate = startdate.strftime('%Y-%m-%d')
        enddate = enddate.strftime('%Y-%m-%d')
    # Generate dates_table using date_range function
    if timestep == 'monthly':
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
        if option_leapyear == 2:
            mask1 = (dates_table['daysinmonth'] == 29)
            dates_table.loc[mask1,'daysinmonth'] = 28
    elif timestep == 'daily':
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
        if option_leapyear == 2:
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
    print("The 'datesmodelrun' function has finished.")
    return dates_table, startdate, enddate
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
    col_bins = (np.arange(int(binsize/2),9001,binsize))
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
        for hyps_file in os.listdir(hyps_filepath):
            # For all the files in the give directory (hyps_fil_path) see if
            # there is a match.  If there is, then geodetic mass balance is
            # available for calibration.
            if re.match(hyps_ID_findname, hyps_file):
                hyps_ID_fullfile = (hyps_filepath + '/' + hyps_file)
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


def hypsometryglaciers(glac_table):
    """
    Select hypsometry from the RGI V6 of all glaciers that are being used in the
    model run. This function returns hypsometry for each 50 m elevation bin.
    """
    bins = np.arange(int(binsize/2)-binsize*3,9000,binsize)
        # note provide 3 rows of negative numbers to rename columns
    hyps_table = pd.DataFrame(0, index=glac_table.index, columns=bins)
    hyps_table.rename(columns={hyps_table.columns.values[0]: 'RGIId',
                               hyps_table.columns.values[1]: 'GLIMSId',
                               hyps_table.columns.values[2]: 'Area'},
                               inplace=True)
    row_count = 0
        # used to get proper indexing when concatenating over multiple regions
    if option_glacier_hypsometry == 1:
        # Check if glacier selection was done using option 1.  If so, then the
        # rgi_regionsO1 or 'O1Index', which is the index associated with the
        # RGIId's can be used to extract glacier hypsometry.
        # This should be much faster computationally.
        if option_glacier_selection == 1:
            for x_region in rgi_regionsO1:
                rgi_regionsO1_findname = str(x_region) + '_rgi60_'
                # Look up the RGI tables associated with the specific regions
                # defined above and concatenate all the areas of interest into a
                # single file
                for rgi_regionsO1_file in os.listdir(hyps_filepath):
                    if re.match(rgi_regionsO1_findname, rgi_regionsO1_file):
                        # if a match for the region is found, then open the file
                        # and select the subregion(s) and/or glaciers from that
                        # file
                        rgi_regionsO1_fullpath = (hyps_filepath +
                                                  rgi_regionsO1_file)
                        rgi_regionsO1_fullfile = (rgi_regionsO1_fullpath + '/' +
                                                  rgi_regionsO1_file +
                                                  '_hypso.csv')
                        csv_regionO1 = pd.read_csv(rgi_regionsO1_fullfile)
                        csv_regionO1.rename(columns={
                                csv_regionO1.columns.values[0]: 'RGIId',
                                csv_regionO1.columns.values[2]: 'Area'},
                                inplace=True)
                            # Rename 'RGIId' and 'Area' columns to remove the
                            # extra spaces that exist currently in the column
                            # headers (ex. 08/25/2017 the header was 'RGIId   '
                            # as opposed to 'RGIId'
                        # Populate hyps_table with all glaciers in the study
                        if rgi_regionsO2 == 'all' and rgi_glac_number == 'all':
                            hyps_table.iloc[row_count:row_count+len(
                                csv_regionO1),0:len(csv_regionO1.columns)] = (
                                csv_regionO1.values)
                            row_count = row_count + len(csv_regionO1)
                            # Note: need to use row_count, specify column
                            # length, and use '.values' in order to populate the
                            # hyps_table because all of the regional hypsometry
                            # files have different column lengths
                        else:
                            # copy index to enable logical indexing
                            csv_regionO1['Indexcopy'] = (
                                csv_regionO1.index.values)
                            # use logical indexing to select rows based on the
                            # initial index ('O1Index') and the copied index
                            hyps_table_raw = csv_regionO1.loc[
                                            csv_regionO1['Indexcopy'].isin(
                                            glac_table['O1Index'])]
                            # Copy hyps_table_raw to avoid issues with
                            # 'SettingWithCopyWarning'
                            hyps_table = hyps_table_raw.copy()
                            # drop the copied index
                            hyps_table.drop(['Indexcopy'], axis=1, inplace=True)
                            # reset the index to agree with glac_table
                            hyps_table.reset_index(drop=True, inplace=True)
        else:
            print("If option_glacier_selection != 1, then need to code this "
                  "portion. Right now, no other options exist and this is the "
                  "fastest way of doing things.\n Exiting model run.")
            exit()
        # Format the hyps table and perform calculations such that each bin
        # shows the glacier area.
        hyps_table.drop(['RGIId', 'Area', 'GLIMSId'], axis=1, inplace=True)
            # drop columns such that only have GlacNo index and area hypsometry
        hyps_table = hyps_table.mul(glac_table.loc[:,'Area'], axis=0)/1000
            # Convert hyps_table (RGI gives integer in thousandths of the total
            # area with area in a separate column) to area (km2) in each bin.
        columns_int = hyps_table.columns.values.astype(int)
            # convert columns from string to integer such that they can be used
            # in downscaling computations
        hyps_table.columns = columns_int
        print("The 'hypsometryglaciers' function has finished.")
        return hyps_table
    elif option_glacier_hypsometry == 2:
        # Create an empty dataframe glac_hyps that will store the hypsometry of
        # all the glaciers with the bin size specified by user input. Set index
        # to be consistent with main_glac_rgi as well as 'RGIId'
        col_bins = (np.arange(int(binsize/2),9001,binsize))
            # creating all columns from 0 to 9000 within given bin size, which
            # enables this to be used for every glacier in the world
        glac_hyps = pd.DataFrame(index=glac_table.index, columns=col_bins)
            # rows of table will be the glacier index
            # columns will be the center elevation of each bin
        for glac in range(len(glac_hyps)):
            # Select full RGIId string (needs to be compatible with mb format)
            hyps_ID_full = glac_table.loc[glac,'RGIId']
            # Remove the 'RGI60-' from the name to agree with David Shean's
            # naming convention (This needs to be automatized!)
            hyps_ID_split = hyps_ID_full.split('-')
            # Choose the end of the string
            hyps_ID_short = hyps_ID_split[1]
            hyps_ID_findname = hyps_ID_short + '_mb_bins.csv'
            for hyps_file in os.listdir(hyps_filepath):
                # For all the files in the give directory (hyps_fil_path) see if
                # there is a match.  If there is, then geodetic mass balance is
                # available for calibration.
                if re.match(hyps_ID_findname, hyps_file):
                    hyps_ID_fullfile = (hyps_filepath + hyps_file)
                    hyps_ID_csv = pd.read_csv(hyps_ID_fullfile)
            # Insert the hypsometry data into the main hypsometry row
            for nrow in range(len(hyps_ID_csv)):
                # convert elevation bins into integers
                elev_bin = int(hyps_ID_csv.loc[nrow,'# bin_center_elev'])
                # add bin count to each elevation bin (or area)
                glac_hyps.loc[glac,elev_bin] = hyps_ID_csv.loc[nrow,'bin_count']
        # Fill NaN values with 0
        glac_hyps.fillna(0, inplace=True)
        print("The 'hypsometryglaciers' function has finished.")
        return glac_hyps
    else:
        print('\n\tModel Error: please choose an option that exists for'
              '\n\tglacier hypsometry. Exiting model run now.\n')
        exit() # if you have an error, exit the model run


def importHussfile(filename):
    filepath = hyps_filepath + 'bands_' + str(binsize) + 'm_DRR/'
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
    bins = np.arange(int(binsize/2),9000,binsize)
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
    Select all glaciers to be used in the model run according to the regions
    and glacier numbers defined by the RGI glacier inventory. This function
    returns the rgi table associated with all of these glaciers.
    """
    # Select glaciers according to RGI V60 tables.
    glacier_table = pd.DataFrame()
        # Create an empty dataframe. glacier_table is the main rgi glacier
        # dataframe that will be used as a reference that has all the
        # important generic glacier characteristics.
    for x_region in rgi_regionsO1:
        # print(f"\nRegion: {x_region}")
        rgi_regionsO1_findname = str(x_region) + '_rgi60_'
        # print(f"Looking for region {x_region} filename...")
        # Look up the RGI tables associated with the specific regions
        # defined above and concatenate all the areas of interest into a
        # single file
        for rgi_regionsO1_file in os.listdir(rgi_filepath):
            if re.match(rgi_regionsO1_findname, rgi_regionsO1_file):
                # if a match for the region is found, then open the file and
                # select the subregions and/or glaciers from that file
                rgi_regionsO1_fullfile = rgi_filepath + rgi_regionsO1_file
                csv_regionO1 = pd.read_csv(rgi_regionsO1_fullfile)
                # Populate glacer_table with the glaciers of interest
                if rgi_regionsO2 == 'all' and rgi_glac_number == 'all':
                    print(f"\nAll glaciers within region(s) {rgi_regionsO1}"
                          " are included in this model run.")
                    if glacier_table.empty:
                        glacier_table = csv_regionO1
                    else:
                        glacier_table = pd.concat([glacier_table,
                                                   csv_regionO1], axis=0)
                elif rgi_regionsO2 != 'all' and rgi_glac_number == 'all':
                    print(f"\nAll glaciers within subregion(s) "
                          f"{rgi_regionsO2} in region {rgi_regionsO1} are "
                          "included in this model run.")
                    for x_regionO2 in rgi_regionsO2:
                        if glacier_table.empty:
                            glacier_table = (csv_regionO1.loc[
                                             csv_regionO1['O2Region']
                                             == x_regionO2])
                        else:
                           glacier_table = (pd.concat([glacier_table,
                                             csv_regionO1.loc[
                                             csv_regionO1['O2Region']
                                             == x_regionO2]], axis=0))
                else:
                    print(f"\nThis study is only focusing on glaciers "
                          f"{rgi_glac_number} in region {rgi_regionsO1}.")
                    for x_glac in rgi_glac_number:
                        glac_id = ('RGI60-' + str(rgi_regionsO1)[1:-1] +
                                   '.' + x_glac)
                        if glacier_table.empty:
                            glacier_table = (csv_regionO1.loc[
                                             csv_regionO1['RGIId'] ==
                                             glac_id])
                        else:
                            glacier_table = (pd.concat([glacier_table,
                                             csv_regionO1.loc[
                                             csv_regionO1['RGIId'] ==
                                             glac_id]], axis=0))
    # glacier_table['O1Index'] = glacier_table.index.values
    # exit()
    glacier_table.reset_index(inplace=True)
        # reset the index so that it is in sequential order (0, 1, 2, etc.)
    glacier_table.rename(columns={'index': 'O1Index'}, inplace=True)
        # change old index to 'O1Index' to be easier to recall what it is
    glacier_table_copy = glacier_table.copy()
        # must make copy; otherwise, drop will cause SettingWithCopyWarning
    glacier_table_copy.drop(rgi_cols_drop, axis=1, inplace=True)
        # drop columns of data that is not being used
    glacier_table_copy.index.name = indexname
    print("The 'select_rgi_glaciers' function has finished.")
    return glacier_table_copy


def surfacetypeglacinitial(option_fxn, option_firn, option_debris, glac_table,
                           glac_hyps):
    """
    Define initial surface type according to median elevation such that the
    melt can be calculated over snow or ice.
    Convention:
        1 - ice
        2 - snow
        3 - firn
        4 - debris
        0 - off-glacier
    """
    # Function Options:
    # option_convention: What surface types will be included?
    #   > 1 (default) - only snow and ice
    #   > 2 - include firn
    #   > 3 - include debris
    #   > 4 - include firn and debris
    # BETTER TO HAVE OPTION_SURFACETYPE_DEBRIS & OPTION_SURFACETYPE_FIRN, which
    # will be used to state whether or not debris and firn are included


    #   > 1 (default) - only snow, firn and ice. Ice below the median elevation
    #                   and snow below it.
    #   > 2 (not coded yet) - snow, firn, ice, and debris
    #           need to import Batu's debris maps to get this started.  Perhaps
    #           need other options to account for how debris changes?
    #   > 3 (idea) - snow, firn, ice (and/or debris)
    #                set snow/ice elevation based on a snow product/imagery from
    #                initial date, e.g., SRTM - Feb 2002 (ModScag/Landsat?)

    glac_surftype = glac_hyps.copy()
    # Rows are each glacier, and columns are elevation bins
    # Loop through each glacier (row) and elevation bin (column) and define the
    # initial surface type (snow, firn, ice, or debris)
    if option_fxn == 1:
        # glac_surftype[(glac_hyps > 0) &
        for glac in range(glac_surftype.shape[0]):
            elev_med = glac_table.loc[glac,'Zmed']
            for col in range(glac_surftype.shape[1]):
                elev_bin = int(glac_surftype.columns.values[col])
                if glac_hyps.iloc[glac,col] > 0:
                    if elev_bin <= elev_med:
                        glac_surftype.iloc[glac,col] = 1
                    else:
                        glac_surftype.iloc[glac,col] = 2
    elif option_fxn ==2:
        print("This option to include debris has not been coded yet. Please "
              "choose an option that exists. Exiting model run.\n")
        exit()
    else:
        print("This option for 'option_surfacetype' does not exist. Please "
              "choose an option that exists. Exiting model run.\n")
        exit()
    # Make sure surface type is integer values
    glac_surftype = glac_surftype.astype(int)
    # If firn is included, then specify initial firn conditions
    if option_firn == 1:
        glac_surftype[glac_surftype == 2] = 3
            # everything initially considered snow is considered firn, i.e., the
            # model initially assumes there is no snow on the surface anywhere.
    if option_debris == 1:
        print('Need to code the model to include debris. This option does not '
              'currently exist.  Please choose an option that does.\nExiting '
              'the model run.')
        exit()
        # One way to include debris would be to simply have debris cover maps
        # and state that the debris retards melting as a fraction of melt.  It
        # could also be DDF_debris as an additional calibration tool. Lastly,
        # if debris thickness maps are generated, could be an exponential
        # function with the DDF_ice as a term that way DDF_debris could capture
        # the spatial variations in debris thickness that the maps supply.
    print("The 'initialsurfacetype' function has finished.")
    return glac_surftype
