"""
fxns_climte.py is a list of functions that are used to facilitate the
pre-processing of the climate data associated with each glacier for PyGEM.
"""
#========= LIST OF PACKAGES ==================================================
import pandas as pd
import numpy as np
import netCDF4 as nc
from datetime import datetime

#========= IMPORT COMMON VARIABLES FROM MODEL INPUT ==========================
from pygem_input import *
    # import all data

#========= DESCRIPTION OF VARIABLES (alphabetical order) =====================
    # dates_table - main dataframe of the dates including the year, month, and
    #               number of days in the month
    # glac_table - main table of glaciers in model run with RGI information
    # variablename - name of variable of interest for a given GCM

    # Variables associated with fxns specific to working with Valentina's data:
    # climate_data - global climate model data for a meteorological variable
    #                provided in an excel format (Valentina's data)
    # climate_dict - dictionary to retrieve gcm data from lat/long in a table
    #                format (Valentina's data)
    # output_csv - define output filename for .csv file (Valentina's data)
    # resolution - resolution of the gcm data (required for Valentina's data)

#========= FUNCTIONS (alphabetical order) ===================================
def importGCMvarnearestneighbor_xarray(filename, variablename, glac_table, dates_table, start_date, end_date):
    # OPTION 1: Nearest neighbor to select climate data
    """
    Import meteorological variables and extract the nearest neighbor time series of the variable. Meteorological data 
    from the global climate models were provided by Ben Marzeion and ETH-Zurich for the GlacierMIP Phase II project.
    "NG" refers to their "new generation" products, which are homogenized.  Additionally, ERA-Interim reanalysis data
    were provided by ECMWF.
    Note: prior to running the script, the user should open the netcdf file in python to determine the names of the keys
    """
    # Import netcdf file
    filefull = gcm_filepath_var + filename
    data = xr.open_dataset(filefull)
    
    # Explore the dataset properties
#     print('Explore the dataset:\n', data)
    # Explore the variable of interest
#     print('\nExplore the variable of interest:\n', data[variablename])
    # Extract the variable's attributes (ex. units)
#     print(data.variables[variablename].attrs['units'])
#     print('\n\nExplore the data in more detail:')
#     print(data[variablename].isel(time=0, latitude=0, longitude=0))
    
    # Determine the correct time indices
    if timestep == 'monthly':
        start_idx = (np.where(pd.Series(data.variables[gcm_time_varname]).apply(lambda x: x.strftime('%Y-%m')) == 
                             dates_table['date'].apply(lambda x: x.strftime('%Y-%m'))[0]))[0][0]
        end_idx = (np.where(pd.Series(data.variables[gcm_time_varname]).apply(lambda x: x.strftime('%Y-%m')) == 
                             dates_table['date'].apply(lambda x: x.strftime('%Y-%m'))[dates_table.shape[0] - 1]))[0][0]
        #  np.where finds the index position where to values are equal
        #  pd.Series(data.variables[gcm_time_varname]) creates a pandas series of the time variable associated with the netcdf
        #  .apply(lambda x: x.strftime('%Y-%m')) converts the timestamp to a string with YYYY-MM to enable the comparison
        #    > different climate dta can have different date formats, so this standardization for comparison is important
        #      ex. monthly data may provide date from 1st of month or from middle of month, so YYYY-MM-DD would not work
        #  The same processing is done for the dates_table['date'] to facilitate the comparison
        #  [0] is used to access the first date
        #  dates_table.shape[0] - 1 is used to access the last date
        #  The final indexing [0][0] is used to access the value, which is inside of an array containing extraneous information
    elif timestep == 'daily':
        start_idx = (np.where(pd.Series(data.variables[gcm_time_varname]).apply(lambda x: x.strftime('%Y-%m-%d')) == 
                             dates_table['date'].apply(lambda x: x.strftime('%Y-%m-%d'))[0]))[0][0]
        end_idx = (np.where(pd.Series(data.variables[gcm_time_varname]).apply(lambda x: x.strftime('%Y-%m-%d')) == 
                             dates_table['date'].apply(lambda x: x.strftime('%Y-%m-%d'))[dates_table.shape[0] - 1]))[0][0]
    # Extract the time series
    time_series = pd.Series(data.variables[gcm_time_varname][start_idx:end_idx+1])
#     print(time_series)
    
#     # Create empty dataset for the variable
#     print(dates_table.shape[0])
#     if option_dates == 1:
#         glac_variable_series = pd.DataFrame(0, index=glac_table.index, columns=dates_table['date'])
#     else:
#         glac_variable_series = pd.DataFrame(0, index=glac_table.index, columns=time_series)
    # Find Nearest Neighbor
    for glac in range(len(glac_table)):
#     for glac in range(0,1):
        # Find index of nearest lat/lon
        lat_nearidx = np.asscalar((abs(data.variables[gcm_lat_varname][:] - glac_table.loc[glac,lat_colname])).argmin().values)
        lon_nearidx = np.asscalar((abs(data.variables[gcm_lon_varname][:] - glac_table.loc[glac,lon_colname])).argmin().values)
        #  argmin() is finding the minimum distance between the glacier lat/lon and the GCM pixel
        #  .values is used to extract the position's value as opposed to having an array
        #  np.asscalar() is used to convert from a np.array to an integer such that the value can be used as an index
        # Print the lat/lon indexes
#         print(lat_nearidx, lon_nearidx)
        # Print the actual latitude and longitude as a check
#         print(data.variables[gcm_lat_varname][lat_nearidx].values, data.variables[gcm_lon_varname][lon_nearidx].values)
        
        # Select the slice of GCM data for each glacier
#         print(data[variablename][start_idx:end_idx+1,lat_nearidx,lon_nearidx].values.shape)
#         print(data[variablename][start_idx:end_idx+1,lat_nearidx,lon_nearidx].values)
        if glac == 0:
            glac_variable_series_nparray = data[variablename][start_idx:end_idx+1,lat_nearidx,lon_nearidx].values
        else:
            glac_variable_series_nparray = np.stack((glac_variable_series_nparray, 
                                                     data[variablename][start_idx:end_idx+1,lat_nearidx,lon_nearidx].values))
    # Create DataFrame from stacked np.arrays
    if option_dates == 1:
        glac_variable_series = pd.DataFrame(glac_variable_series_nparray, index=glac_table.index, columns=dates_table['date'])
    else:
        glac_variable_series = pd.DataFrame(glac_variable_series_nparray, index=glac_table.index, columns=time_series)
        
    # Perform corrections to the data if necessary
    # Surface air temperature corrections
    if variablename == gcm_temp_varname:
        if 'units' in data.variables[variablename].attrs and data.variables[variablename].attrs['units'] == 'K':
            glac_variable_series = glac_variable_series - 273.15
            #   Convert from K to deg C
        elif option_warningmessages == 1:
            print('Check units of air temperature from GCM is degrees C.')

    # Precipitation corrections
    # If the variable is precipitation
    if variablename == gcm_prec_varname:
        # If the variable has units and those units are meters (ERA Interim)
        if 'units' in data.variables[variablename].attrs and data.variables[variablename].attrs['units'] == 'm':
            pass
        # Elseif the variable has units and those units are kg m-2 s-1 (CMIP5)
        elif 'units' in data.variables[variablename].attrs and data.variables[variablename].attrs['units'] == 'kg m-2 s-1':  
            # Convert from kg m-2 s-1 to m day-1
            glac_variable_series = glac_variable_series/1000*3600*24
            #   (1 kg m-2 s-1) * (1 m3/1000 kg) * (3600 s / hr) * (24 hr / day) = (m day-1)
        # Else check the variables units
        else:
            print('Check units of precipitation from GCM is meters per day.')
        if timestep == 'monthly':
            # Convert from meters per day to meters per month
            if 'daysinmonth' in dates_table.columns:
                glac_variable_series = (glac_variable_series.mul(list(dates_table['daysinmonth']), axis=1))
            else:
                print("\nMODEL ERROR: 'daysinmonth' DOES NOT EXIST IN THE DATES TABLE.\n" 
                      " Please check that the dates_table is formatted properly such that a \n'daysinmonth' column"
                      " exists.\n\n"
                      "Exiting the model run.\n")
                exit()

    print(f"\nThe 'importGCMvarnearestneighbor' fxn for '{variablename}' has finished.")
    return glac_variable_series, time_series


def importGCMfxnearestneighbor_netcdf4(variablename, glac_table):
    """
    Import time invariant (constant) variables and extract the nearest neighbor of the variable. Meteorological data 
    from the global climate models were provided by Ben Marzeion and ETH-Zurich for the GlacierMIP Phase II project or
    from ERA Interim (ECMWF), which uses geopotential instead of surface height
    """
    # Import netcdf file
    filefull = gcm_filepath_fx + gcm_elev_filename
    data = nc.Dataset(filefull)
    # Print the keys (the variables used within the netcdf file)
    # print(data.variables.keys())
    # print(data.variables[variablename])
    # Extract the variable of interest
    variable_data = data.variables[variablename][:]
    # Extract the latitude and longitude
    lat = data.variables[gcm_lat_varname][:]
    lon = data.variables[gcm_lon_varname][:]
    # Try to extract time index (required for ERA Interim, but not given for CMIP5 data)
    try:
        time_idx = 0
        #  In ERA Interim, time only has 1 value, so the index is 0
    except:
        # If time does not exist, then lat/lon should be the only keys requried to extract elevation
        pass
    # Nearest neighbor to extract data for each glacier
    glac_variable = pd.Series(0, index=glac_table.index)
    for glac in range(glac_table.shape[0]):
        # Find index of nearest lat/lon
        lat_nearidx = (abs(lat-glac_table.loc[glac,lat_colname])).argmin()
        lon_nearidx = (abs(lon-glac_table.loc[glac,lon_colname])).argmin()
        # Extract time series of the variable for the given lat/lon
        try: 
            glac_variable.loc[glac] = variable_data[time_idx, lat_nearidx, lon_nearidx]
        except:
            glac_variable.loc[glac] = variable_data[lat_nearidx, lon_nearidx]
    if hasattr(data.variables[variablename], 'units') and data.variables[variablename].units == 'm':
        pass      
    elif hasattr(data.variables[variablename], 'units') and data.variables[variablename].units == 'm**2 s**-2':
        # Convert m2 s-2 to m by dividing by gravity (ERA Interim states to use 9.80665)
        glac_variable = glac_variable / 9.80665
    else:
        if option_warningmessages == 1:
            print('Check units of elevation from GCM is m.')
    print(f"The 'importGCMfxnearestneighbor' fxn for '{variablename}' has finished.")
    return glac_variable


#def importGCMvarnearestneighbor_netcdf4(filename, variablename, glac_table, dates_table, start_date, end_date):
#    # OPTION 1: Nearest neighbor to select climate data
#    """
#    Import meteorological variables and extract the nearest neighbor time series of the variable. Meteorological data 
#    from the global climate models were provided by Ben Marzeion and ETH-Zurich for the GlacierMIP Phase II project.
#    "NG" refers to their "new generation" products, which are homogenized.  Additionally, ERA-Interim reanalysis data
#    were provided by ECMWF.
#    Note: prior to running the script, the user should open the netcdf file in python to determine the names of the keys
#    """
#    # Import netcdf file
#    filefull = gcm_filepath_var + filename
#    data = nc.Dataset(filefull)
#    # Print the keys (the variables used within the netcdf file)
##    print(data.variables.keys())
#    # Print the information about the variable of interest
##    print(data.variables[variablename])
#    # Extract the variable of interest
#    variable_data = data.variables[variablename][:]
#    # Extract the latitude and longitude
#    lat = data.variables[gcm_lat_varname][:]
#    lon = data.variables[gcm_lon_varname][:]
#    # Extract all data associated with the time variable
#    time_var = data.variables[gcm_time_varname]
#    time = nc.num2date(time_var[:],time_var.units,time_var.calendar)
#    # Convert time to appropriate format for comparison with start and end dates (YYYY-MM or YYYY-MM-DD)
#    # For monthly timestep convert to YYYY-MM to extract proper positions within the netcdf
#    if timestep == 'monthly':
#        for step in range(len(time)):
#            time[step] = time[step].strftime('%Y-%m')
#        # Find the index position for the start date within the netcdf
#        start_idx = np.where(time == start_date)[0][0]
#        end_idx = np.where(time == end_date)[0][0]
#        #   np.where finds where the index where the condition is met and returns a tuple.  To access, the value need to select
#        #   the first value in the first object. Hence, the use of [0][0]
#        # Extract the correct time series from the netcdf
#        time_series = nc.num2date(time_var[:],time_var.units,time_var.calendar)[start_idx:end_idx+1]
##        time_series = pd.Series(time_series).dt.date
#        # This will truncate the time from the series; however, we want all this data for the output netcdf
#        #   need to "+ 1" such that the end date is included (python convention excludes the last value in a range)
#    elif timestep == 'daily':
#        if (data.variables['time'][1] - data.variables['time'][0]) == 1:
#            for step in range(len(time)):
#                time[step] = time[step].strftime('%Y-%m-%d')
#            start_idx = np.where(time == start_date)[0][0]
#            end_idx = np.where(time == end_date)[0][0]
#            time_series = nc.num2date(time_var[:],time_var.units,time_var.calendar)[start_idx:end_idx+1]
##            time_series = pd.Series(time_series).dt.date
#        else:
#            # print error if the model time step is not daily as this will cause the wrong data to be extracted
#            print("\nMODEL ERROR: TIME STEP OF GCM DOES NOT APPEAR TO BE 'DAILY'.\n"
#                  "Please check the GCM time variable and make sure that it is consistent with the time step.\n"
#                  "specified in model input. Exiting the model run now.\n")
#            exit()
#    # Nearest neighbor used to extract data for each glacier
#    if option_dates == 1:
#        glac_variable_series = pd.DataFrame(0, index=glac_table.index, columns=dates_table['date'])
#    else:
#        glac_variable_series = pd.DataFrame(0, index=glac_table.index, columns=time_series)
#    for glac in range(len(glac_table)):
#        # Find index of nearest lat/lon
#        lat_nearidx = (abs(lat-glac_table.loc[glac,lat_colname])).argmin()
#        lon_nearidx = (abs(lon-glac_table.loc[glac,lon_colname])).argmin()
#        # Extract time series of the variable for the given lat/lon
#        glac_variable_series.loc[glac,:] = variable_data[start_idx:end_idx+1, lat_nearidx, lon_nearidx]
#        #   > Use positional indexing to extract data properly
#    # Perform corrections to the data if necessary
#    # Surface air temperature corrections
#    if variablename == gcm_temp_varname:
#        if hasattr(data.variables[variablename], 'units') and data.variables[variablename].units == 'K':
#            glac_variable_series = glac_variable_series - 273.15
#            #   Convert from K to deg C
#        elif option_warningmessages == 1:
#            print('Check units of air temperature from GCM is degrees C.')
#    # Precipitation corrections
#    if variablename == gcm_prec_varname:
#        if hasattr(data.variables[variablename], 'units') and data.variables[variablename].units == 'm':
#            pass           
#        elif hasattr(data.variables[variablename], 'units') and data.variables[variablename].units == 'kg m-2 s-1':  
#            # Convert from kg m-2 s-1 to m day-1
#            glac_variable_series = glac_variable_series/1000*3600*24
#            #   (1 kg m-2 s-1) * (1 m3/1000 kg) * (3600 s / hr) * (24 hr / day) = (m day-1)
#        else:
#            print('Check units of precipitation from GCM is meters per day.')
#        if timestep == 'monthly':
#            # Convert from meters per day to meters per month
#            if 'daysinmonth' in dates_table.columns:
#                glac_variable_series = (glac_variable_series.mul(list(dates_table['daysinmonth']), axis=1))
#            else:
#                print("\nMODEL ERROR: 'daysinmonth' DOES NOT EXIST IN THE DATES TABLE.\n" 
#                      " Please check that the dates_table is formatted properly such that a \n'daysinmonth' column"
#                      " exists.\n\n"
#                      "Exiting the model run.\n")
#                exit()
#    print(f"The 'importGCMvarnearestneighbor' fxn for '{variablename}' has finished.")
#    return glac_variable_series, time_series


#========= FUNCTIONS SPECIFIC TO VALENTINA'S DATA (alphabetical order) =======
##def col_change_Val_to_YearMonth(climate_data, output_csv):
##    """
##    Convert date-time from Valentina's old datasets (two rows) into one row
##    such that the new "year-month" row can be used as the column headings.
##    """
##    for column in climate_data:
##        # print the year (row 0) and month (row 1) for each column
##        print(column, int(climate_data.loc[0,column]),
##              int(climate_data.loc[1,column]))
##        if column == 0:
##            # first column is center latitude
##            climate_data.rename(columns = {0:'CenLat'}, inplace=True)
##        elif column == 1:
##            # second column is center longitude
##            climate_data.rename(columns = {1:'CenLon'}, inplace=True)
##        else:
##            # other columns are year (row 0) and month (row 1)
##            year_str = str(int(climate_data.loc[0,column]))
##            month_int = int(climate_data.loc[1,column])
##                # computes integer value of the month
##            if month_int < 10:
##                month_str = '0' + str(month_int)
##                # makes all strings 2 digits to ensure proper order of columns
##            else:
##                month_str = str(month_int)
##                # 10, 11, 12 already have 2 digits
##            climate_data.rename(columns={column:(year_str + '_' + month_str)},
##                        inplace=True)
##    climate_data.to_csv(output_csv)
##
##def latlongdict(climate_data):
##    # Transform Valetina's excel sheets into a dictionary to speed up processing
##    # time in python
##    """
##    Create a dictionary where lat/long can be entered to retrieve data.
##    """
##    climate_data_lat = climate_data['CenLat']
##    climate_data_lon = climate_data['CenLon']
##    # Separate lat/long from data to make dictionary processing easier
##    climate_data_x = climate_data.copy()
##    climate_data_x.drop(['CenLat', 'CenLon'], axis=1, inplace=True)
##    # Iterate through the data to make a dictionary with two keys (lat/long)
##    my_dict = dict()
##    for nrow in range(len(climate_data_x)):
##        lat = climate_data_lat.loc[nrow]
##        lon = climate_data_lon.loc[nrow]
##        x = climate_data_x.loc[nrow]
##        # print(lat, lon, x)
##        my_dict[(lat, lon)] = x
##    print("The 'lat_long_dict' function has finished.")
##    return my_dict
##
##def nearestneighbor(glac_table, climate_dict, resolution):
##    # nearest neighbor is embedded within importing GCM; hence, this only
##    # only applies to working with examples from Valentina's data.
##    """
##    Use nearest neighbor to select the meteorological data that will be
##    used in the model run.  Note: this meteorological data is raw (corrections
##    are applied until future steps).
##    """
##    main_output = pd.DataFrame()
##    for nrow in range(len(glac_table)):
##        # round the lat/long to agree with climate dictionaries
##        cenlat_rd = (round(glac_table.loc[nrow,'CenLat'] / resolution) *
##                     resolution)
##        cenlon_rd = (round(glac_table.loc[nrow,'CenLon'] / resolution) *
##                     resolution)
##        output_row = pd.DataFrame(climate_dict[cenlat_rd,cenlon_rd]).T
##            # transposed to switch format back to a row with date as columns
##        if main_output.empty:
##            main_output = output_row
##        else:
##            main_output = pd.concat([main_output, output_row], axis=0)
##        # reset index to align index values with the glac_table
##        main_output.reset_index(drop=True, inplace=True)
##        main_output.index.name = glac_table.index.name
##    print("The 'climate_nearestneighbor' function has finished.")
##    return main_output