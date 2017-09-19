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
def importGCMfxnearestneighbor(variablename, glac_table):
    """
    Import time invariant (constant) variables and extract the nearest neighbor
    of the variable. Meteorological data from the global climate models were
    provided by Ben Marzeion and ETH-Zurich for the GlacierMIP Phase II project.
    """
    # Import netcdf file
    filefull = gcm_filepath_fx + variablename + gcm_filename_fx
    data = nc.Dataset(filefull)
    # Extract the keys (the variables used within the netcdf file)
    keys = data.variables.keys()
    # print('\n',keys)
        # prints all information about a given variables
    # Extract the variable of interest
    variable_data = data.variables[variablename][:]
    # Extract the latitude and longitude
    lat = data.variables['lat'][:]
    lon = data.variables['lon'][:]
    # Nearest neighbor to extract data for each glacier
    glac_variable = pd.Series(0, index=glac_table.index)
    for glac in range(len(glac_table)):
        # Find index of nearest lat/lon
        lat_nearidx = (abs(lat-glac_table.loc[glac,lat_colname])).argmin()
        lon_nearidx = (abs(lon-glac_table.loc[glac,lon_colname])).argmin()
        # Extract time series of the variable for the given lat/lon
        glac_variable.loc[glac] = variable_data[lat_nearidx, lon_nearidx]
    print(f"The 'importGCMfxnearestneighbor' fxn for '{variablename}' has "
          "finished.")
    return glac_variable


def importGCMvarnearestneighbor(variablename, glac_table, dates_table):
    """
    Import meteorological variables and extract the nearest neighbor time series
    of the variable. Meteorological data from the global climate models were
    provided by Ben Marzeion and ETH-Zurich for the GlacierMIP Phase II project.
     "NG" refers to their "new generation" products, which are homogenized.
    """
    # Import netcdf file
    filefull = gcm_filepath_var + variablename + gcm_filename_var
    data = nc.Dataset(filefull)
    # Extract the keys (the variables used within the netcdf file)
    keys = data.variables.keys()
    # print(keys)
    # print(data.variables['tas'])
        # prints all information about a given variables
    # Extract the variable of interest
    variable_data = data.variables[variablename][:]
    # Extract the latitude and longitude
    lat = data.variables['lat'][:]
    lon = data.variables['lon'][:]
    # Extract all data associated with the time variable
    time_var = data.variables['time']
    time = nc.num2date(time_var[:],time_var.units,time_var.calendar)
    # Convert time from datetime format to YYYY-MM-DD
    if timestep == 'monthly':
        for step in range(len(time)):
            time[step] = time[step].strftime('%Y-%m')
        # Find the index position for the start date
        # Convert start and end year into proper format
        startdate = str(startyear) + '-01'
        enddate = str(endyear) + '-12'
        # Then find the position
        start_idx = np.where(time == startdate)[0][0]
        end_idx = np.where(time == enddate)[0][0]
            # np.where finds where the index where the condition is met and
            # returns a tuple.  To access, the value need to select the first
            # value in the first object via [0][0]
        time_series = time[start_idx:end_idx+1]
            # need to "+ 1" such that the end date is included (python
            # convention excludes the last value in a range)
    elif timestep == 'daily':
        if (data.variables['time'][1] - data.variables['time'][0]) == 1:
            for step in range(len(time)):
                time[step] = time[step].strftime('%Y-%m-%d')
            startdate = str(startyear) + '-01-01'
            enddate = str(endyear) + '-12-31'
            start_idx = np.where(time == startdate)[0][0]
            end_idx = np.where(time == enddate)[0][0]
            time_series = time[start_idx:end_idx+1]
        else:
            print("\nMODEL ERROR: TIME STEP DOES NOT APPEAR TO BE 'DAILY'.\n"
                  "Please check the GCM time variable and make sure that it is "
                  "consistent with the \ntime step specified in model input."
                  "\n\nExiting the model run.\n\n")
                # print error if the model time step is not daily as this will
                # cause the wrong data to be extracted from the GCM
            exit()
    # Nearest neighbor to extract data for each glacier
    glac_variable_series = pd.DataFrame(0, index=glac_table.index,
                                        columns=time_series)
    for glac in range(len(glac_table)):
        # Find index of nearest lat/lon
        lat_nearidx = (abs(lat-glac_table.loc[glac,lat_colname])).argmin()
        lon_nearidx = (abs(lon-glac_table.loc[glac,lon_colname])).argmin()
        # Extract time series of the variable for the given lat/lon
        glac_variable_series.loc[glac,:] = variable_data[start_idx:end_idx+1,
                                                     lat_nearidx, lon_nearidx]
    # Perform necessary corrections to the data
    # Corrections for surface air temperature
    if variablename == 'tas':
        glac_variable_series = glac_variable_series - 273.15
            # Convert from K to deg C
    # Corrections for precipitation
    if variablename == 'pr':
        glac_variable_series = glac_variable_series/1000*3600*24
        # Convert from kg m-2 s-1 to m day-1
        # (1 kg m-2 s-1) * (1 m3/1000 kg) * (3600 s / hr) * (24 hr / day)
        if timestep == 'monthly':
            if 'daysinmonth' in dates_table.columns:
                glac_variable_series = (glac_variable_series.mul(
                                        list(dates_table['daysinmonth']),
                                        axis=1))
            else:
                    # Convert from meters per day to meters per month
                print("\nMODEL ERROR: 'daysinmonth' DOES NOT EXIST IN THE DATES"
                      " TABLE.\n Please check that the dates_table is formatted"
                      " properly such that a \n'daysinmonth' column exists.\n\n"
                      "Exiting the model run.\n\n")
                exit()
    print(f"The 'importGCMvarnearestneighbor' fxn for '{variablename}' has "
          "finished.")
    return glac_variable_series


#========= FUNCTIONS SPECIFIC TO VALENTINA'S DATA (alphabetical order) =======
def col_change_Val_to_YearMonth(climate_data, output_csv):
    """
    Convert date-time from Valentina's old datasets (two rows) into one row
    such that the new "year-month" row can be used as the column headings.
    """
    for column in climate_data:
        # print the year (row 0) and month (row 1) for each column
        print(column, int(climate_data.loc[0,column]),
              int(climate_data.loc[1,column]))
        if column == 0:
            # first column is center latitude
            climate_data.rename(columns = {0:'CenLat'}, inplace=True)
        elif column == 1:
            # second column is center longitude
            climate_data.rename(columns = {1:'CenLon'}, inplace=True)
        else:
            # other columns are year (row 0) and month (row 1)
            year_str = str(int(climate_data.loc[0,column]))
            month_int = int(climate_data.loc[1,column])
                # computes integer value of the month
            if month_int < 10:
                month_str = '0' + str(month_int)
                # makes all strings 2 digits to ensure proper order of columns
            else:
                month_str = str(month_int)
                # 10, 11, 12 already have 2 digits
            climate_data.rename(columns={column:(year_str + '_' + month_str)},
                        inplace=True)
    climate_data.to_csv(output_csv)


def latlongdict(climate_data):
    # Transform Valetina's excel sheets into a dictionary to speed up processing
    # time in python
    """
    Create a dictionary where lat/long can be entered to retrieve data.
    """
    climate_data_lat = climate_data['CenLat']
    climate_data_lon = climate_data['CenLon']
    # Separate lat/long from data to make dictionary processing easier
    climate_data_x = climate_data.copy()
    climate_data_x.drop(['CenLat', 'CenLon'], axis=1, inplace=True)
    # Iterate through the data to make a dictionary with two keys (lat/long)
    my_dict = dict()
    for nrow in range(len(climate_data_x)):
        lat = climate_data_lat.loc[nrow]
        lon = climate_data_lon.loc[nrow]
        x = climate_data_x.loc[nrow]
        # print(lat, lon, x)
        my_dict[(lat, lon)] = x
    print("The 'lat_long_dict' function has finished.")
    return my_dict


def nearestneighbor(glac_table, climate_dict, resolution):
    # nearest neighbor is embedded within importing GCM; hence, this only
    # only applies to working with examples from Valentina's data.
    """
    Use nearest neighbor to select the meteorological data that will be
    used in the model run.  Note: this meteorological data is raw (corrections
    are applied until future steps).
    """
    main_output = pd.DataFrame()
    for nrow in range(len(glac_table)):
        # round the lat/long to agree with climate dictionaries
        cenlat_rd = (round(glac_table.loc[nrow,'CenLat'] / resolution) *
                     resolution)
        cenlon_rd = (round(glac_table.loc[nrow,'CenLon'] / resolution) *
                     resolution)
        output_row = pd.DataFrame(climate_dict[cenlat_rd,cenlon_rd]).T
            # transposed to switch format back to a row with date as columns
        if main_output.empty:
            main_output = output_row
        else:
            main_output = pd.concat([main_output, output_row], axis=0)
        # reset index to align index values with the glac_table
        main_output.reset_index(drop=True, inplace=True)
        main_output.index.name = glac_table.index.name
    print("The 'climate_nearestneighbor' function has finished.")
    return main_output
