import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import os

# INPUTS
glac_no = '01.00570' 
glacier_name = 'Gulkana'
site = 'A'
region = glac_no[:2]
GCM = 'MERRA2'

# FILEPATHS
fp_clim = '/home/claire/research/climate_data/'
fp_MERRA2 = os.path.join(fp_clim,'MERRA2/VAR/MERRA2_VAR_LAT_LON.nc')
fp_ERA5 = os.path.join(fp_clim,'ERA5/ERA5_hourly/ERA5_VAR_hourly.nc')
fp_AWS = os.path.join(fp_clim,'AWS/Gulkana/LVL2/gulkana1480_hourly_LVL2.csv')
list_RGI = os.listdir('/home/claire/research/RGI/rgi60/00_rgi60_attribs/')
for file in list_RGI:
    if region in file:
        fp_RGI = '/home/claire/research/RGI/rgi60/00_rgi60_attribs/' + file
fp_MERRA2_constants = '~/research/climate_data/MERRA2/MERRA2constants.nc4'
fp_ERA5_constants = '~/research/climate_data/ERA5/ERA5_hourly/ERA5_geopotential.nc'
fp_MB = '/home/claire/research/MB_data/Gulkana/Input_Gulkana_Glaciological_Data.csv'

# NECESSARY INFO
start = pd.to_datetime('2000-01-01 00:30')
end = pd.to_datetime('2023-12-31 23:30')
all_vars = {'temp':{'MERRA2':'T2M','ERA5':'t2m'},
            'tp':{'MERRA2':'PRECTOTCORR','ERA5':'tp'},
            'rh':{'MERRA2':'RH2M','ERA5':'rh'},
            'SWin':{'MERRA2':'SWGDN','ERA5':'ssrd'},
            'LWin':{'MERRA2':'LWGAB','ERA5':'strd'},
            'uwind':{'MERRA2':'U2M','ERA5':'u10'},
            'vwind':{'MERRA2':'V2M','ERA5':'v10'},
            'sp':{'MERRA2':'PS','ERA5':'sp'},
            'tcc':{'MERRA2':'CLDTOT','ERA5':'tcc'},
            'bcwet':{'MERRA2':'BCWT002'},'bcdry':{'MERRA2':'BCDP002'},
            'dustwet':{'MERRA2':'DUWT002'},'dustdry':{'MERRA2':'DUDP002'}}
AWS_elev = 1480
LAPSE_RATE = -0.0065

# FIND LAT LON OF GLACIER CENTERPOINT
glacier_table = pd.read_csv(fp_RGI)
glacier_table = glacier_table.loc[glacier_table['RGIId'] == 'RGI60-'+glac_no]
cenlat = glacier_table['CenLat'].to_numpy()[0]
cenlon = glacier_table['CenLon'].to_numpy()[0]

# DEFINE FUNCTION TO SELECT DATASET
def get_point_ds(GCM,var):
    """
    Selects the closest latitude and longitude gridcell
    from a GCM dataset to the input cenlat, cenlon coordinates.
    Also returns the geopotential of that gridcell converted
    to elevation units (m)
    """
    # open GCM dataset
    if GCM in ['MERRA2','BOTH']:
        file_lat = str(int(np.floor(cenlat/10)*10))
        file_lon = str(int(np.floor(cenlon/10)*10))
        fn_MERRA2 = fp_MERRA2.replace('LAT',file_lat).replace('LON',file_lon)
        fn_MERRA2 = fn_MERRA2.replace('VAR',all_vars[var]['MERRA2'])
        ds = xr.open_dataset(fn_MERRA2)
        eds = xr.open_dataset(fp_MERRA2_constants)
        latname = 'lat'
        lonname = 'lon'
        elevname = 'PHIS'
    if GCM in ['ERA5','BOTH']:
        fn_ERA5 = fp_ERA5.replace('VAR',var)
        ds = xr.open_dataset(fn_ERA5)
        eds = xr.open_dataset(fp_ERA5_constants)
        latname = 'latitude'
        lonname = 'longitude'
        elevname = 'z'

    # get latitude and longitude of nearest GCM point
    datalat = ds.coords[latname][:].values
    datalon = ds.coords[lonname][:].values
    lat_nearidx = np.abs(cenlat - datalat).argmin()
    lon_nearidx = np.abs(cenlon - datalon).argmin()
    lat = datalat[lat_nearidx]
    lon = datalon[lon_nearidx]

    if GCM == 'ERA5':
        lat = lat.round(2)
        lon = lon.round(2)
        if var == 'tcc':
            lat = lat.round(1)
            lon = lon.round(1)

    # select dataset by closest lat/lon to glacier center
    ds = ds.sel({latname:lat,lonname:lon}).drop_vars([latname,lonname])
    eds = eds.sel({latname:lat,lonname:lon})
    elev = eds[elevname].to_numpy()[0] / 9.81
    return ds,elev

# LOAD MASS BALANCE DATA
df_mb = pd.read_csv(fp_MB)
df_mb = df_mb.loc[df_mb['site_name'] == site]
years_mb = np.unique(df_mb['Year'])
site_elev = df_mb.loc[df_mb['Year']==2011,'elevation'].values[0]
# chose 2011 because sites A, AB, B, D all measured

# LOAD CLIMATE DATA
# precipitation from GCM
ds_tp,elev = get_point_ds(GCM,'tp')
df_tp = ds_tp.to_dataframe()
df_tp['Date'] = np.array([time.date() for time in df_tp.index])
df_tp = df_tp.rename(columns={all_vars['tp'][GCM]:'tp'})

# temp from AWS
df_AWS = pd.read_csv(fp_AWS)
df_AWS = df_AWS.set_index(pd.to_datetime(df_AWS['local_time']))
df_AWS = df_AWS.loc[start-pd.Timedelta(minutes=30):]
df_AWS['Year'] = np.array([time.year for time in df_AWS.index])
df_temp = df_AWS.rename(columns={'site_temp_USGS':'temp'})
years_merra2 = np.unique(df_temp['Year'])

# adjust temperature by lapserate
df_temp['temp'] += (LAPSE_RATE)*(site_elev - AWS_elev)

# get intersecting years
years = np.array(list(set(years_mb) & set(years_merra2)))
years = np.sort(years)[1:]

# index MB data by years
winter_mb_data = df_mb['bw'].loc[df_mb['Year'].isin(years)]

tp_measured = []
for year in years:
    # get dates
    previous_fall_date = df_mb.loc[df_mb['Year']==year-1]['fall_date'].to_numpy()[0]
    if str(previous_fall_date) == 'nan':
        previous_fall_date = str(year-1) + '-08-10 00:00'
    spring_date = df_mb.loc[df_mb['Year']==year]['spring_date'].to_numpy()[0]
    acc_dates = pd.date_range(previous_fall_date,spring_date,freq='d')

    df_temp_year = df_temp.loc[acc_dates]
    snow_days = df_temp_year.loc[df_temp_year['temp'] < 0].index
    snow_hours = []
    for d in snow_days:
        day = str(d)
        snow_hours.append(pd.date_range(day+' 00:30',day+' 23:30',freq='h'))
    snow_hours = np.array(snow_hours).flatten()

    # sum mass balance
    tp = df_tp['tp'].loc[snow_hours].to_numpy() * 3600 / 1000
    bw = np.sum(tp[~np.isnan(tp)])
    tp_measured.append(bw)
tp_measured = np.array(tp_measured)

# REGRESSION
include_intercept = False
if include_intercept:
    X = np.vstack([tp_measured,np.ones(len(tp_measured))]).T
    y = winter_mb_data.values
    result,resid = np.linalg.lstsq(X,y,rcond=None)[:2]
    slope = result[0]
    intercept = result[1]
else:
    X = tp_measured.reshape(-1,1)
    y = winter_mb_data.values
    result,resid = np.linalg.lstsq(X,y,rcond=None)[:2]
    slope = result[0]
    intercept = 0
R2 = 1 - resid[0] / (y.size * y.var())

# PLOT RESULTS
fig,ax = plt.subplots(2)
ax[0].plot(years,winter_mb_data,label='MB Data')
ax[0].plot(years,tp_measured * slope + intercept,label='Adj. precip Data')
ax[0].plot(years,tp_measured,'--',label='Raw precip Data')
ax[0].legend()
ax[0].set_xlim(years[0],years[-1])
ax[0].set_ylabel('MB (m w.e.)')

ax[1].scatter(tp_measured,winter_mb_data)
x = np.arange(0,0.6,.05)
line_label = f'{slope:.2f} x + {intercept:.2f}, R$^2$ = {R2:.3f}'
ax[1].plot(x, x*slope+intercept,label=line_label)
ax[1].legend()
ax[1].set_xlabel('Measured Acc (m w.e.)')
ax[1].set_ylabel('Measured Prec (m w.e.)')
ax[1].set_xlim(0,np.max(X))
ax[1].set_ylim(0,np.max(y))

fig.suptitle(f'Regression for MERRA-2 precipitation factor at {glacier_name} site {site}')
plt.show()