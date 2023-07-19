import cdsapi
import os
import pandas as pd

lon_min = 134
lon_max = 180
lat_min = 50
lat_max = 72

all_vars = ['2m_temperature','total_precipitation','surface_pressure','2m_dewpoint_temperature',
            '10m_u_component_of_wind','10m_v_component_of_wind','surface_solar_radiation_downwards']
var = '10m_v_component_of_wind'

# Output information
file_format = 'netcdf'
# folder_out = 'D:/ERA5_hourly/'+var+'/'
folder_out = '/home/claire/research/CDS/'
downloaded_file = 'ERA5_'+var+'_Alaskayear_hourly.nc'
downloaded_file = os.path.join(folder_out, downloaded_file)

# Set up time
start_year = 2012
end_year = 2021
years = [ str(start_year +i ) for i in range(end_year - start_year + 1)] 
start_day = 1
end_day = 31
days = [ str(start_day +i ).zfill(2) for i in range(end_day - start_day + 1)]

c = cdsapi.Client()

for year in years:
    c.retrieve('reanalysis-era5-land',
        {
            'year': year,
            'variable': var,
            'month': [ '01', '02', '03',
                    '04', '05', '06',
                    '07', '08', '09',
                    '10', '11', '12',],
            'day': days,
            'time': [
                '00:00', '01:00', '02:00',
                '03:00', '04:00', '05:00',
                '06:00', '07:00', '08:00',
                '09:00', '10:00', '11:00',
                '12:00', '13:00', '14:00',
                '15:00', '16:00', '17:00',
                '18:00', '19:00', '20:00',
                '21:00', '22:00', '23:00',
            ],
            'area': [ lat_min, lon_min, lat_max, lon_max ],
            'format': file_format,
        },
        downloaded_file.replace('year',year))
    print('finished year',year)