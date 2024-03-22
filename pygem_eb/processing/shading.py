"""
Created on Tue Mar 19 11:30:50 2024

Shading model for PyGEM-EB
Requirements: - DEM, slope and aspect rasters surrounding glacier
              - Coordinates for point to perform calculations

1. Input site coordinates and time zone
2. Load DEM, slope and aspect grid
3. Determine horizon angles
        Optional: plot horizon search
4. Calculate sky-view factor
5. Calculate direct clear-sky slope-corrected irradiance and
                           shading for each hour of the year
6. Store .csv of Islope and shade 
        Optional: plot results

@author: clairevwilson
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import rioxarray as rxr
import xarray as xr
import pandas as pd
import suncalc
import geopandas as gpd
from pyproj import Transformer
from numpy import pi, cos, sin, arctan

# =================== INPUTS ===================
site = 'AB'                          # name of site for indexing .csv
timezone = pd.Timedelta(hours=-9)   # time zone of location
site_name = 'Gulkana' + site        # user-defined site name

# plot options
# horizon
plot_horizon = True         # plot horizon search?
store_plot_horizon = True   # save .png horizon plot?

# result
# plot_result = False       # plot results? options: 'dirirrslope','shaded'
plot_result = ['shaded','dirirrslope']
store_plot_result = True    # store .png result plot?
store_result = False        # store .csv output file?

# by angle
plot_by_angle = False

# model options
angle_step = 5              # step to calculate horizon angle (degrees)
search_length = 5000        # distance to search from center point (m)
sub_dt = 10                 # timestep to calculate solar corrections (minutes)

# get site lat and lon    
latlon_df = pd.read_csv('~/GulkanaDEM/gulkana_sites.csv',index_col=0)
lat = latlon_df.loc[site]['lat']    # latitude of point of interest
lon = latlon_df.loc[site]['lon']    # longitude of point of interest

# =================== FILEPATHS ===================
fp = '/home/claire/'
# in
dem_fp = fp + 'GulkanaDEM/Gulkana_DEM_20m.tif'
aspect_fp = fp + 'GulkanaDEM/Gulkana_Aspect_20m.tif'
slope_fp = fp + 'GulkanaDEM/Gulkana_Slope_20m.tif'
# dem_fp = fp + 'GulkanaDEM/2m/Gulkana_2m_DEM.tif'
# aspect_fp = fp + 'GulkanaDEM/2m/Gulkana_2m_aspect.tif'
# slope_fp = fp + 'GulkanaDEM/2m/Gulkana_2m_slope.tif'
# optional shapefile for visualizing
shp_fp = fp + 'GulkanaDEM/Gulkana.shp'

# out
out_fp = fp + f'GulkanaDEM/Out/{site_name}.csv'
out_image_fp = fp + f'GulkanaDEM/Outputs/{site_name}.png'
out_horizon_fp = fp + f'GulkanaDEM/Outputs/{site_name}_horizon.png'

# =================== SETUP ===================
# open files
dem = rxr.open_rasterio(dem_fp).isel(band=0)
aspect = rxr.open_rasterio(aspect_fp).isel(band=0)
slope = rxr.open_rasterio(slope_fp).isel(band=0)
dem_res = dem.rio.resolution()[0]

# ensure same coordinates
shapefile = gpd.read_file(shp_fp).set_crs(epsg=4326)
shapefile = shapefile.to_crs(dem.rio.crs)

# filter nans
dem = dem.where(dem > 0)
aspect = aspect.where(aspect > 0)
slope = slope.where(slope > 0)

# get min/max elevation for plotting
min_elev = int(np.round(np.min(dem.values)/100,0)*100)
max_elev = int(np.round(np.max(dem.values)/100,0)*100)

# get UTM coordinates from lat/lon
transformer = Transformer.from_crs('EPSG:4326', dem.rio.crs, always_xy=True)
xx, yy = transformer.transform(lon, lat)
# get elevation of point from grid
point_elev = dem.sel(x=xx, y=yy, method="nearest").values

# get slope and aspect at point of interest
asp = aspect.sel(x=xx, y=yy, method="nearest").values * pi/180
slp = slope.sel(x=xx, y=yy, method="nearest").values * pi/180
print(f'{site_name} point aspect: {asp*180/pi:.1f} o     slope: {slp*180/pi:.2f} o')

# =================== CONSTANTS ===================
I0 = 1368       # solar constant in W m-2
P0 = 101325     # sea-level pressure in Pa
PSI = 0.75      # vertical atmospheric clear-sky transmissivity
MEAN_RAD = 1    # mean earth-sun radius in AU

# =================== FUNCTIONS ===================
def r_sun(time):
    """From DEBAM manual, gets earth-to-sun radius in AU"""
    doy = time.day_of_year
    # From DEBAM manual
    # theta = (2*pi*doy) / 365 * pi/180
    # radius = 1.000110 + 0.34221*cos(theta) + 1.280e-3*sin(theta) + 7.19e-4*cos(2*theta) + 7.7e-5*sin(2*theta)
    radius = 1 - 0.01672*cos(0.9856*(doy-4))
    return radius

def pressure(elev):
    """Adjusts air pressure by elevation"""
    P = np.exp(-0.0001184*elev)*P0
    return P

def zenith(time):
    """Calculates solar zenith angle for time, lat and lon"""
    time_UTC = time - timezone
    altitude_angle = suncalc.get_position(time_UTC,lon,lat)['altitude']
    zenith = pi/2 - altitude_angle if altitude_angle > 0 else np.nan
    return zenith

def declination(time):
    """Calculates solar declination"""
    doy = time.day_of_year
    delta = -23.4*cos(360*(doy+10)/365) * pi/180
    return delta

hour_angle = lambda t: 15*(12-t.day_of_year)

def select_coordinates(angle,step_size,length):
    """Creates a line of points from the starting cell
    to select grid cells in a given direction (angle in 
    deg 0-360 where 0 is North)"""
    # get starting coordinates
    start_x = xx
    start_y = yy

    # convert angle to radians and make 0 north
    rad = angle * pi/180 + pi/2

    # get change in x and y for each step
    dx = - step_size * cos(rad) # negative so it travels clockwise
    dy = step_size * sin(rad)

    # define end
    n_steps = np.ceil(length / step_size).astype(int)
    end_x = start_x + dx*n_steps
    end_y = start_y + dy*n_steps
    
    # create lines
    xs = np.linspace(start_x,end_x,n_steps)
    ys = np.linspace(start_y,end_y,n_steps)
    if xs.shape > ys.shape:
        ys = np.ones(n_steps) * start_y
    elif ys.shape > xs.shape:
        xs = np.ones(n_steps) * start_x
    
    return xs,ys

def find_horizon(elev,xs,ys,buffer = 30):
    """Finds the horizon along a line of elevation
    values paired to x,y coordinates
    - elev: array of elevations in a single direction
    - xs, ys: coordinates corresponding to elev
    - buffer: defines the minimum number of gridcells
                away the horizon can be found 
                (needed when looking uphill)"""
    # calculate distance from origin and height relative to origin
    distances = np.sqrt((xs-xx)**2+(ys-yy)**2)
    distances[np.where(distances < 1e-3)[0]] = 1e-6
    heights = elev - point_elev
    heights[np.where(heights < 0)[0]] = 0

    # identify maximum horizon elevation angle
    elev_angles = arctan(heights/distances)
    idx = np.argmax(elev_angles[buffer:]) + buffer

    # index out information about the horizon point
    horizon_angle = elev_angles[idx]
    # horizon_distance = distances[idx]
    # horizon_height = heights[idx]
    horizon_x = xs[idx]
    horizon_y = ys[idx]
    return horizon_angle,horizon_x,horizon_y,elev_angles

# =================== MODEL RUN ===================
# plot DEM as background
if plot_horizon:
    fig,ax = plt.subplots(figsize=(6,6))
    dem.plot(ax=ax,cmap='viridis')
    shapefile.plot(ax=ax,color='none',edgecolor='black',linewidth=1.5)
    plt.axis('equal')

# loop through angles
angles = np.arange(0,360,angle_step)
horizons = {}
for ang in angles:
    # set up dict to store
    horizons[ang] = {'horizon_elev':[],'hz_x':[],'hz_y':[]}
    
    # get line in the direction of choice
    xs, ys = select_coordinates(ang,dem_res,search_length)
    
    # select elevation gridcells along the line
    x_select = xr.DataArray(xs,dims=['location'])
    y_select = xr.DataArray(ys,dims=['location'])
    elev = dem.sel(x=x_select,y=y_select,method='nearest').values

    # filter out nans
    xs = xs[~np.isnan(elev)]
    ys = ys[~np.isnan(elev)]
    elev = elev[~np.isnan(elev)]
    
    # find the horizon
    hz_ang,hz_x,hz_y,all_angles = find_horizon(elev,xs,ys)
    horizons[ang]['horizon_elev'] = hz_ang
    horizons[ang]['hz_x'] = hz_x
    horizons[ang]['hz_y'] = hz_y

    # visualize elevations 
    if plot_horizon:
        norm = mpl.colors.Normalize(vmin=min_elev,vmax=max_elev)
        cmap = plt.cm.viridis
        scalar_map = mpl.cm.ScalarMappable(norm=norm,cmap=cmap)
        colors = scalar_map.to_rgba(elev)
        plt.scatter(xs,ys,color=colors,s=1,marker='.',alpha=0.7)
        plt.scatter(hz_x,hz_y,color='red',marker='x',s=50)

# calculate sky-view factor
horizon_elev = np.array([horizons[ang]['horizon_elev'] for ang in angles])
angle_step_rad = angle_step * pi/180
sky_view = np.sum(cos(horizon_elev)**2 * angle_step_rad) / (2*pi)

# scatter center point and plot
if plot_horizon:
    plt.scatter(xx,yy,marker='*',color='orange')
    plt.title(f'{site_name} Horizon Search \n Sky-view Factor = {sky_view:.3f}')
    plt.ylabel('Northing')
    plt.xlabel('Easting')
    if store_plot_horizon:
        plt.savefig(out_horizon_fp)
    else:
        plt.show()

# loop through hours of the year and store data
store_vars = ['dirirrslope','shaded','corr_factor','sun_elev','horizon_elev','sun_az']
year_hours = pd.date_range('2024-01-01 00:00','2024-12-31 23:00',freq='h')
df = pd.DataFrame(data = np.ones((8784,len(store_vars))),
                  columns=store_vars,index=year_hours)
for time in year_hours:
    # loop to get sub-hourly values and average
    sub_vars = ['shaded','Islope','corr_factor','zenith']
    period_dict = {}
    for var in sub_vars:
        period_dict[var] = np.array([])
    for minutes in np.arange(0,60,sub_dt):
        # calculate time-dependent variables
        time_UTC = time - timezone + pd.Timedelta(minutes = minutes)
        P = pressure(point_elev)
        r = r_sun(time)
        Z = zenith(time)
        d = declination(time)
        h = hour_angle(time)
        period_dict['zenith'] = np.append(period_dict['zenith'],Z)

        # calculate direct clear-sky irradiance (not slope corrected)
        I = I0 * (MEAN_RAD/r)**2 * PSI**(P/P0/np.cos(Z)) * np.cos(Z)

        # get sun elevation and azimuth angle
        sunpos = suncalc.get_position(time_UTC,lon,lat)
        sun_elev = sunpos['altitude']       # solar elevation angle
        # suncalc gives azimuth with 0 = South, we want 0 = North
        sun_az = sunpos['azimuth'] + pi     # solar azimuth angle

        # get nearest angle of horizon calculations to the sun azimuth
        sun_az = 2*pi + sun_az if sun_az < 0 else sun_az
        idx = np.argmin(np.abs(angles*pi/180 - sun_az))
        
        # check if the sun elevation angle is below the horizon angle
        shaded = 1 if sun_elev < horizon_elev[idx] else 0
        period_dict['shaded'] = np.append(period_dict['shaded'],shaded)

        # incident angle calculation
        cosTHETA = cos(slp)*cos(Z) + sin(slp)*sin(Z)*cos(sun_az - asp)
        corr_factor = min(cosTHETA/cos(Z),5) * (shaded-1)*-1
        Islope = I * corr_factor
        period_dict['corr_factor'] = np.append(period_dict['corr_factor'],corr_factor)
        period_dict['Islope'] = np.append(period_dict['Islope'],Islope)

    # find hourly mean (median for shaded to remain in Bool)
    I = period_dict['Islope']
    cosZ = cos(period_dict['zenith'])
    corrf = period_dict['corr_factor']
    dt = np.ones(len(I)) * sub_dt
    if ~np.any(np.isnan(I)):
        mean_I = np.sum(I*cosZ*corrf*dt) / np.sum(dt)
        if np.sum(I*cosZ) > 0:
            mean_corr_factor = np.sum(I*cosZ*corrf) / np.sum(I*cosZ)
        else:
            mean_corr_factor = 1
    else:
        mean_I = 0
        mean_corr_factor = 0
    median_shaded = int(np.median(period_dict['shaded']))
    assert ~np.isnan(mean_I)

    # store data
    df.loc[time,'shaded'] = median_shaded
    df.loc[time,'dirirrslope'] = mean_I
    df.loc[time,'corr_factor'] = mean_corr_factor

    # unnecessary, just for plotting
    df.loc[time,'sun_az'] = sun_az * 180/pi
    df.loc[time,'horizon_elev'] = horizon_elev[idx] * 180/pi
    df.loc[time,'sun_elev'] = sun_elev * 180/pi

# variable properties for plotting
varprops = {'dirirr':{'label':'direct flat-surface irradiance [W m-2]','cmap':'plasma'},
              'dirirrslope':{'label':'direct slope-corrected irradiance [W m-2]','cmap':'plasma'},
              'shaded':{'label':'shading [black = shaded]','cmap':'binary'},
              'sun_elev':{'label':'sun elevation angle [$\circ$]','cmap':'Spectral_r'},
              'horizon_elev':{'label':'horizon elevation angle [[$\circ$]','cmap':'YlOrRd'},
              'sun_az':{'label':'sun azimuth angle [$\circ$]','cmap':'twilight'}}
if plot_result:
    # initialize plot
    nrows = 2 if len(plot_result) > 2 else 1
    ncols = len(plot_result) if len(plot_result) <= 2 else int(len(plot_result)/2)
    fig,axes = plt.subplots(nrows,ncols,figsize=(ncols*4,nrows*3),layout='constrained')
    axes = axes.flatten()
    fig.suptitle(f'Annual plots for {site_name} \n Sky-view factor: {sky_view:.3f}')
    
    # get days and hours of the year
    days = np.arange(366)
    hours = np.arange(0,24)
    
    # loop through variables to plot
    for i,var in enumerate(plot_result):
        # gather plot information
        label = varprops[var]['label']
        cmap = varprops[var]['cmap']
        ax = axes[i]

        # reshape and plot data
        vardata = df[var].to_numpy().reshape((len(days),len(hours)))
        pc = ax.pcolormesh(days,hours,vardata.T, cmap=cmap)

        # add colorbar ('shaded' is binary)
        if var != 'shaded':
            clb = fig.colorbar(pc,ax=ax,aspect=10,pad=0.02)
            clb.set_label(var,loc='top')

        # add labels
        ax.set_ylabel('Hour of day')
        ax.set_xlabel('Day of year')
        site_name = site_name if site_name else f'{lat:.2f},{lon:.2f}'
        ax.set_title(label)
    
    # store or show plot
    if store_plot_result:
        plt.savefig(out_image_fp,dpi=150)
    else: 
        plt.show()

if plot_by_angle:
    all_az = np.arange(0,360,1) * pi/180
    elev_angle_list = []
    for sun_az in all_az:
        # get nearest angle of horizon calculations to the sun azimuth
        sun_az = 2*pi + sun_az if sun_az < 0 else sun_az
        idx = np.argmin(np.abs(angles*pi/180 - sun_az))
        elev_angle_list.append(horizon_elev[idx]*180/pi)
    elev_angle_arr = np.array(elev_angle_list)

    fig,ax = plt.subplots()
    ax.plot(all_az*180/pi,elev_angle_arr,color='black')
    ax.fill_between(all_az*180/pi,elev_angle_arr,color='black',alpha=0.6)
    ax.set_xlabel('Azimuth angle ($\circ$)')
    ax.set_ylabel('Horizon angle ($\circ$)')
    fig.suptitle(site_name+' shading by azimuth angle (0$^{\circ}$N)',fontsize=14)
    plt.savefig(f'/home/claire/GulkanaDEM/Outputs/{site_name}_angles.png')

if store_result:
    df[['dirirrslope','shaded']].to_csv(out_fp)