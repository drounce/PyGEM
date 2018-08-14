from ecmwfapi import ECMWFDataServer
import pandas as pd

import pygem_input as input

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
grid_res = '0.5/0.5'
# Bounding box (N/W/S/E)
bounding_box = '50/60/20/107'
# Output file name
output_fp = input.main_directory + '/../Climate_data/ERA_Interim/download/'
output_fn = 'ERA_Interim_' + varname + '_' + start_date + '_' + end_date + '.nc'
output_fullfn = output_fp + output_fn


#%% SET UP DATA DOWNLOAD FILE
# Dates formatted properly as a string
date_list = "/".join([d.strftime('%Y%m%d') for d in pd.date_range(start=start_date,end=end_date, freq='MS')])

# Details for each variable download
# Temperature
if varname == 'temperature':
    download_dict = {
        "class": "ei",
        "dataset": "interim",
        "date": date_list,
        "expver": "1",
        "grid": grid_res,
        "levtype": "sfc",
        "param": "167.128",
        "stream": "moda",
        "type": "an",
        "format": "netcdf",
        "target": output_fullfn,
        }
# Precipitation
elif varname == 'precipitation':
    download_dict = {
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
        "format": "netcdf",
        "target": output_fullfn,
        }
# Geopotential
elif varname == 'geopotential':
    download_dict = {
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
        "format": "netcdf",
        "target": output_fullfn,
        }
# Pressure level temperature
elif varname == 'pressurelevel_temp':
    download_dict = {
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
        "format": "netcdf",
        "target": output_fullfn,
        }

#%% DOWNLOAD DATA FROM SERVER
server = ECMWFDataServer()
server.retrieve(download_dict)
