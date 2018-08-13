from ecmwfapi import ECMWFDataServer
import pandas as pd

#%% INPUT DATA
# Variable
varname = 'temperature'
#varname = 'precipitation'
#varname = 'geopotential'
#varname = 'pressurelevel_temp'
# Dates
start_date = '19790101'
end_date = '20180501'
# Resolution
grid_res = '0.125/0.125'
# Bounding box (N/W/S/E)
bounding_box = '50/60/20/107'


#%% SET UP DATA DOWNLOAD FILE
# Dates formatted properly as a string
date_list = "/".join([d.strftime('%Y%m%d') for d in pd.date_range(start=start_date,end=end_date, freq='MS')])

# Details for each variable download
# Temperature
if varname == 'temperature':
    file_dict = {
        "class": "ei",
        "dataset": "interim",
        "date": date_list,
        "expver": "1",
        "grid": grid_res,
        "levtype": "sfc",
        "param": "167.128",
        "stream": "moda",
        "type": "an",
        "target": "output",
        }
# Precipitation
elif varname == 'precipitation':
    file_dict = {
        "class": "ei",
        "dataset": "interim",
        "date": date_list,
        "expver": "1",
        "grid": grid_res,
        "levtype": "sfc",
        "param": "228.128",
        "step": "0-12",
        "stream": "mdfa",
        "type": "fc",
        "target": "output",
        }
# Geopotential
elif varname == 'geopotential':
    file_dict = {
        "class": "ei",
        "dataset": "interim",
        "date": "1989-01-01",
        "expver": "1",
        "grid": grid_res,
        "levtype": "sfc",
        "param": "129.128",
        "step": "0",
        "stream": "oper",
        "time": "12:00:00",
        "type": "an",
        "target": "output",
        }
# Pressure level temperature
elif varname == 'pressurelevel_temp':
    file_dict = {
        "class": "ei",
        "dataset": "interim",
        "date": date_list,
        "expver": "1",
        "grid": grid_res,
        "levelist": "300/350/400/450/500/550/600/650/700/750/775/800/825/850/875/900/925/950/975/1000",
        "levtype": "pl",
        "param": "130.128",
        "area": bounding_box,
        "stream": "moda",
        "type": "an",
        "target": "output",
        }

#%% DOWNLOAD DATA FROM SERVER
server = ECMWFDataServer()
# tells python to access the ECMWF Data server 

server.retrieve(file_dict)

##the following code specifies what data to retrieve and in what format to retrieve it
#server.retrieve({
#    "class": "ei",
#    #  specifies the code for the dataset 'ei'=era-int 
#    "dataset": "interim",
#    #  specifies dataset 
#    "date": date_list,
#    #  specifies the timespan of data to download: YYYYMMDD (monthly)
#    "expver": "1",
#    #  data version 
#    "grid": "0.125/0.125",
#    #  size of individual pixels, measured in degrees of lat/long
#    "levtype": "sfc",
#    #  from which level to download data; i.e. sfc=surface, pl=pressure level
#    "levelist": "300/350/400/450/500",
#    #  if there are multiple levels (i.e. pressure), specificy which levels need to be extracted 
#    "param": "167.128",
#    #  type of data you want to download; all datasets are coded.
#    #  PyGEM - specific parameters: 
#    #   2M Temperature: 167.128
#    #   Precipitation: 228.128
#    #   Geopotential:129.128
#    #   Pressure-Level Temperature:130.128
#    "area":"85/-165/57/170",
#    #  create a bounding box around the region from which you would like to extract data: N/W/S/E
#    "stream": "moda",
#    #  forecasting system used to collect data
#    #  PyGEM-specific streams: 
#    #   2M Temperature: moda
#    #   Precipitation: mdfa
#    #   Geopotential: oper
#    #   Pressure-Level Temperature: moda
#    "type": "an",
#    #  type of field to be extracted. i.e. analysis (an) or forecast (fc)
#    #  PyGEM-specific types: 
#    #   2M Temperature: an
#    #   Precipitation: fc
#    #   Geopotential: an
#    #   Pressure-Level Temperature: an
#    "format":"netcdf",
#    #  the format in which you would like to save the file. Netcdf is reccomended
#    #  for PyGEM processing 
#    "target": "eraInt_2mTemp",
#    #  output file name 
#})

