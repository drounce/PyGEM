"""
pygemfxns_modelsetup.py is a list of functions that are used to set up the model with the required input for the main
script.
"""

import pandas as pd
import numpy as np
from datetime import datetime

import pygem_input as input

#========= FUNCTIONS (alphabetical order) ===================================
def datesmodelrun(startyear=input.startyear, endyear=input.endyear, spinupyears=input.spinupyears):
    """
    Create table of year, month, day, water year, season and number of days in the month.
    
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
    startyear_wspinup = startyear - spinupyears
    # Convert start year into date depending on option_wateryear
    if input.option_wateryear == 1:
        startdate = str(startyear_wspinup - 1) + '-10-01'
        enddate = str(endyear) + '-09-30'
    elif input.option_wateryear == 0:
        startdate = str(startyear_wspinup) + '-01-01'
        enddate = str(endyear) + '-12-31'
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
    return dates_table, startdate, enddate


def hypsometrystats(hyps_table, thickness_table):
    """Calculate the volume and mean associated with the hypsometry data.
    
    Output is a series of the glacier volume [km**3] and mean elevation values [m a.s.l.]. 
    """
    # Glacier volume [km**3]
    glac_volume = (hyps_table * thickness_table/1000).sum(axis=1)
    # Mean glacier elevation
    glac_hyps_mean = np.zeros(glac_volume.shape)
    glac_hyps_mean[glac_volume > 0] = ((hyps_table[glac_volume > 0].values * 
                                        hyps_table[glac_volume > 0].columns.values.astype(int)).sum(axis=1) / 
                                       hyps_table[glac_volume > 0].values.sum(axis=1))
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


def import_Husstable(rgi_table, rgi_regionsO1, filepath, filedict, drop_col_names,
                     indexname=input.indexname):
    """Use the dictionary specified by the user to extract the desired variable.
    The files must be in the proper units (ice thickness [m], area [km2], width [km]) and need to be pre-processed to 
    have all bins between 0 - 8845 m.
    
    Output is a Pandas DataFrame of the variable for all the glaciers in the model run
    (rows = GlacNo, columns = elevation bins).
    
    Line Profiling: Loading in the table takes the most time (~2.3 s)
    """
    ds = pd.read_csv(filepath + filedict[rgi_regionsO1[0]])
    # Select glaciers based on 01Index value from main_glac_rgi table
    #  as long as Huss tables have all rows associated with rgi attribute table, then this shortcut works and saves time
    glac_table = ds.iloc[rgi_table['O1Index'].values]
#    glac_table = pd.DataFrame()
#    if input.rgi_regionsO2 == 'all' and input.rgi_glac_number == 'all':
#        glac_table = ds   
#    elif input.rgi_regionsO2 != 'all' and input.rgi_glac_number == 'all':
#        glac_table = ds.iloc[rgi_table['O1Index'].values]
#    elif input.rgi_regionsO2 == 'all' and input.rgi_glac_number != 'all':
#        for glacier in range(len(rgi_table)):
#            if glac_table.empty:
#                glac_table = ds.loc[rgi_table.loc[glacier,'O1Index']]
#            else:
#                glac_table = pd.concat([glac_table, ds.loc[rgi_table.loc[glacier,'O1Index']]], axis=1)
#        glac_table = glac_table.transpose()
    # must make copy; otherwise, drop will cause SettingWithCopyWarning
    glac_table_copy = glac_table.copy()
    # Clean up table and re-index
    # Reset index to be GlacNo
    glac_table_copy.reset_index(drop=True, inplace=True)
    glac_table_copy.index.name = indexname
    # Drop columns that are not elevation bins
    glac_table_copy.drop(drop_col_names, axis=1, inplace=True)
    # Change NAN from -99 to 0
    glac_table_copy[glac_table_copy==-99] = 0.
    return glac_table_copy


def selectcalibrationdata(main_glac_rgi):
    """
    Select geodetic mass balance of all glaciers in the model run that have a geodetic mass balance.  The geodetic mass
    balances are stored in a csv file.
    """
    # Import .csv file
    ds = pd.read_csv(input.cal_mb_filepath + input.cal_mb_filedict[input.rgi_regionsO1[0]])
    main_glac_calmassbal = np.zeros((main_glac_rgi.shape[0],4))
    ds[input.rgi_O1Id_colname] = ((ds[input.cal_rgi_colname] % 1) * 10**5).round(0).astype(int) 
    ds_subset = ds[[input.rgi_O1Id_colname, input.massbal_colname, input.massbal_uncertainty_colname, 
                    input.massbal_time1, input.massbal_time2]].values
    rgi_O1Id = main_glac_rgi[input.rgi_O1Id_colname].values
    for glac in range(rgi_O1Id.shape[0]):
        try:
            # Grab the mass balance based on the RGIId Order 1 glacier number
            main_glac_calmassbal[glac,:] = ds_subset[np.where(np.in1d(ds_subset[:,0],rgi_O1Id[glac])==True)[0][0],1:]
            #  np.in1d searches if there is a match in the first array with the second array provided and returns an
            #   array with same length as first array and True/False values. np.where then used to identify the index
            #   where there is a match, which is then used to select the massbalance value
            #  Use of numpy arrays for indexing and this matching approach is much faster than looping through; however,
            #   need the for loop because np.in1d does not order the values that match; hence, need to do it 1 at a time
        except:
            # If there is no mass balance data available for the glacier, then set as NaN
            main_glac_calmassbal[glac,:] = np.empty(4)
            main_glac_calmassbal[glac,:] = np.nan
    main_glac_calmassbal = pd.DataFrame(main_glac_calmassbal, 
                                        columns=[input.massbal_colname, input.massbal_uncertainty_colname, 
                                                 input.massbal_time1, input.massbal_time2])
    return main_glac_calmassbal


def selectglaciersrgitable(rgi_regionsO1=input.rgi_regionsO1, 
                           rgi_regionsO2=input.rgi_regionsO2, 
                           rgi_glac_number=input.rgi_glac_number,
                           rgi_filepath=input.rgi_filepath,
                           rgi_dict=input.rgi_dict,
                           rgi_cols_drop=input.rgi_cols_drop,
                           rgi_O1Id_colname=input.rgi_O1Id_colname,
                           rgi_glacno_float_colname=input.rgi_glacno_float_colname,
                           indexname=input.indexname):
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
    for x_region in rgi_regionsO1:
        try:
            csv_regionO1 = pd.read_csv(rgi_filepath + rgi_dict[x_region])
        except:
            csv_regionO1 = pd.read_csv(rgi_filepath + rgi_dict[x_region], encoding='latin1')
        # Populate glacer_table with the glaciers of interest
        if rgi_regionsO2 == 'all' and rgi_glac_number == 'all':
            print(f"All glaciers within region(s) {rgi_regionsO1} are included in this model run.")
            if glacier_table.empty:
                glacier_table = csv_regionO1
            else:
                glacier_table = pd.concat([glacier_table, csv_regionO1], axis=0)
        elif rgi_regionsO2 != 'all' and rgi_glac_number == 'all':
            print(f"All glaciers within subregion(s) {rgi_regionsO2} in region {rgi_regionsO1} are included.")
            for x_regionO2 in rgi_regionsO2:
                if glacier_table.empty:
                    glacier_table = csv_regionO1.loc[csv_regionO1['O2Region'] == x_regionO2]
                else:
                    glacier_table = (pd.concat([glacier_table, csv_regionO1.loc[csv_regionO1['O2Region'] == 
                                                                                x_regionO2]], axis=0))
        else:
            print(f"This study is only focusing on glaciers {rgi_glac_number} in region {rgi_regionsO1}.")
            for x_glac in rgi_glac_number:
                glac_id = ('RGI60-' + str(rgi_regionsO1)[1:-1] + '.' + x_glac)
                if glacier_table.empty:
                    glacier_table = csv_regionO1.loc[csv_regionO1['RGIId'] == glac_id]
                else:
                    glacier_table = (pd.concat([glacier_table, csv_regionO1.loc[csv_regionO1['RGIId'] == glac_id]], 
                                               axis=0))
    # must make copy; otherwise, drop will cause SettingWithCopyWarning
    glacier_table_copy = glacier_table.copy()
    # reset the index so that it is in sequential order (0, 1, 2, etc.)
    glacier_table_copy.reset_index(inplace=True)
    # change old index to 'O1Index' to be easier to recall what it is
    glacier_table_copy.rename(columns={'index': 'O1Index'}, inplace=True)
    # drop columns of data that is not being used
    glacier_table_copy.drop(rgi_cols_drop, axis=1, inplace=True)
    # add column with the O1 glacier numbers
    glacier_table_copy[rgi_O1Id_colname] = (
            glacier_table_copy['RGIId'].str.split('.').apply(pd.Series).loc[:,1].astype(int))
    glacier_table_copy[rgi_glacno_float_colname] = (np.array([np.str.split(glacier_table_copy['RGIId'][x],'-')[1] 
                                                    for x in range(glacier_table_copy.shape[0])]).astype(float))
    # set index name
    glacier_table_copy.index.name = indexname
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
    

#========= FUNCTIONS NO LONGER USED (alphabetical order) ==============================================================
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