import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import rioxarray as rxr
import xarray as xr
import pandas as pd
import suncalc
from pyproj import Transformer
from numpy import pi, cos, sin, arctan

# =================== INPUTS ===================
# model options
plot_horizon = False   # plot horizon search?
plot_result = ['sun_elev','horizon_elev','shaded','dirirrslope']  # 'dirirr','dirirrslope','shaded' or False to plot result
store_result = True     # store .csv output file?
angle_step = 10         # step to calculate horizon angle in degrees
search_length = 5000    # distance to search from center point (m)

# site information
lat = 63.260281         # latitude of point of interest
lon = -145.425720       # longitude of point of interest
timezone = pd.Timedelta(hours=-9)   # time zone of location
site_name = 'GulkanaCenter'
min_elev = 1200         # rough estimate of lower elevation bound of DEM
max_elev = 2000         # rough estimate of upper elevation bound of DEM

# =================== FILEPATHS ===================
fp = '/home/claire/research/'
# dem_fp = fp + '../GulkanaDEM/Gulkana_DEM_20m.tif'
# aspect_fp = fp + '../GulkanaDEM/Gulkana_Aspect_20m.tif'
# slope_fp = fp + '../GulkanaDEM/Gulkana_Slope_20m.tif'
dem_fp = fp + '../GulkanaDEM/2m/Gulkana_USGS_DEM_smooth.tif'
aspect_fp = fp + '../GulkanaDEM/2m/Gulkana_USGS_aspect_from_smooth_DEM.tif'
slope_fp = fp + '../GulkanaDEM/2m/Gulkana_USGS_slope_from_smooth_DEM.tif'

out_fp = fp + '../GulkanaDEM/Out/gulkana_centerpoint.csv'

# =================== SETUP ===================
# open files
dem = rxr.open_rasterio(dem_fp).isel(band=0)
aspect = rxr.open_rasterio(aspect_fp).isel(band=0)
slope = rxr.open_rasterio(slope_fp).isel(band=0)
dem_res = dem.rio.resolution()[0]

# filter nans
dem = dem.where(dem > 0)
aspect = aspect.where(dem > 0)
slope = slope.where(dem > 0)

# get UTM coordinates from lat/lon
transformer = Transformer.from_crs('EPSG:4326', dem.rio.crs, always_xy=True)
xx, yy = transformer.transform(lon, lat)
# get elevation of point from grid
point_elev = dem.sel(x=xx, y=yy, method="nearest").values

# =================== CONSTANTS ===================
I0 = 1368       # solar constant in W m-2
P0 = 101325     # sea-level pressure in Pa
PSI = 0.75      # vertical atmospheric clear-sky transmissivity
r_m = 1         # mean earth-sun radius in AU

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
    plt.imshow(dem,cmap='viridis',alpha=0.8)
    legend = plt.colorbar()
    legend.ax.set_ylabel('Elevation (m)', rotation=270, fontsize=14, labelpad=20)
    plt.axis('equal') 

# loop through angles
angles = np.arange(0,360,angle_step)
horizons = {}
for ang in angles:
    # set up dict to store
    horizons[ang] = {'horizon_elang':[],'hz_x':[],'hz_y':[],
                     'elev_arr':[],'elang_arr':[],
                     'xs':[],'ys':[]}
    
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
    horizons[ang]['horizon_elang'] = hz_ang
    horizons[ang]['hz_x'] = hz_x
    horizons[ang]['hz_y'] = hz_y
    horizons[ang]['elev_arr'] = elev
    horizons[ang]['elang_arr'] = all_angles
    horizons[ang]['xs'] = xs
    horizons[ang]['ys'] = ys

    # visualize elevations 
    if plot_horizon:
        norm = mpl.colors.Normalize(vmin=min_elev,vmax=max_elev)
        cmap = plt.cm.viridis
        scalar_map = mpl.cm.ScalarMappable(norm=norm,cmap=cmap)
        colors = scalar_map.to_rgba(elev)
        sc = plt.scatter(xs,ys,color=colors,s=5)
        plt.scatter(hz_x,hz_y,color='red',marker='x',s=50)

# calculate sky-view factor
horizon_elev = np.array([horizons[ang]['horizon_elang'] for ang in angles])
angle_step_rad = angle_step * pi/180
sky_view = np.sum(cos(horizon_elev)**2 * angle_step_rad) / (2*pi)

# scatter center point and plot
if plot_horizon:
    plt.scatter(xx,yy,color='yellow')
    plt.title(f'{site_name} Sky-view Factor = {sky_view:.3f}')
    plt.ylabel('Northing')
    plt.xlabel('Easting')
    plt.show()

# loop through hours of the year and store data
store_vars = ['dirirr','dirirrslope','shaded','sun_elev','horizon_elev','sun_az']
year_hours = pd.date_range('2000-01-01 00:00','2000-12-31 23:00',freq='h')
df = pd.DataFrame(data = np.ones((8784,len(store_vars))),
                  columns=store_vars,index=year_hours)
for time in year_hours:
    # calculate time-dependent variables
    time_UTC = time - timezone
    P = pressure(point_elev)
    r = r_sun(time)
    Z = zenith(time)
    d = declination(time)
    h = hour_angle(time)

    # calculate direct clear-sky irradiance (not slope corrected)
    I = I0 * (r_m/r)**2 * PSI**(P/P0/np.cos(Z)) * np.cos(Z)

    # get sun and slope angles
    sunpos = suncalc.get_position(time_UTC,lon,lat)
    sun_elev = sunpos['altitude']       # solar elevation angle
    sun_az = sunpos['azimuth']          # solar azimuth angle
    asp_north = aspect.sel(x=xx, y=yy, method="nearest").values * pi/180   # slope azimuth angle (aspect)
    asp = asp_north # + pi # aspect DEM treats 0 as north, we need as south
    slp = slope.sel(x=xx, y=yy, method="nearest").values * pi/180    # slope angle

    # incident angle calculation
    cosTHETA = cos(slp)*cos(Z) + sin(slp)*sin(Z)*cos(sun_az - asp)
    Islope = I*cosTHETA/cos(Z)
    Islope = max(0,Islope)

    # get nearest angle of horizon calculations to the sun azimuth
    sun_az = 2*pi + sun_az if sun_az < 0 else sun_az
    idx = np.argmin(np.abs(angles*pi/180 - sun_az))
    
    # check if the sun elevation angle is below the horizon angle
    if sun_elev < horizon_elev[idx]:
        shaded = 1  # shaded
    else:
        shaded = 0  # not shaded
    
    df.loc[time,'sun_az'] = sun_az * 180/pi
    df.loc[time,'horizon_elev'] = horizon_elev[idx] * 180/pi
    df.loc[time,'sun_elev'] = sun_elev * 180/pi
    df.loc[time,'shaded'] = shaded
    df.loc[time,'dirirrslope'] = Islope
    df.loc[time,'dirirr'] = I

if plot_result:
    nrows = 2 if len(plot_result) > 2 else 1
    ncols = len(plot_result) if len(plot_result) <= 2 else int(len(plot_result)/2)
    fig,axes = plt.subplots(nrows,ncols,figsize=(ncols*4,nrows*3),layout='constrained')
    axes = axes.flatten()
    days = np.arange(366)
    hours = np.arange(0,24)
    labels = {'dirirr':'direct flat-surface irradiance [W m-2]',
              'dirirrslope':'direct slope-corrected irradiance [W m-2]',
              'shaded':'shading [1 = shaded]',
              'sun_elev':'sun elevation angle [$\circ$]',
              'horizon_elev':'horizon elevation angle [[$\circ$]',
              'sun_az':'sun azimuth angle [$\circ$]'}
    for i,var in enumerate(plot_result):
        ax = axes[i]
        vardata = df[var].to_numpy().reshape((len(days),len(hours)))
        label = labels[var]
        cmap = 'twilight' if var == 'sun_az' else 'plasma'
        pc = ax.pcolormesh(days,hours,vardata.T, cmap=cmap)
        clb = fig.colorbar(pc,ax=ax,aspect=10,pad=0.02)
        clb.set_label(var,loc='top')
        ax.set_ylabel('Hour of day')
        ax.set_xlabel('Day of year')
        site_name = site_name if site_name else f'{lat:.2f},{lon:.2f}'
        ax.set_title(label)
    fig.suptitle(f'Annual plots for {site_name} \n Sky-view factor: {sky_view:.3f}')
    plt.show()

if store_result:
    df[['dirirrslope','shaded']].to_csv(out_fp)