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
# model options
plot_horizon = True         # plot horizon search?
store_plot_horizon = True   # save .png horizon plot?
# ['sun_az','sun_elev','horizon_elev','shaded','dirirrslope','dirirr']
# plot_result = False         # plot results? False or list from ^
plot_result = ['sun_elev','dirirr','shaded','sun_az','dirirrslope','horizon_elev']
store_plot_result = True    # store .png result plot?
store_result = False        # store .csv output file?
angle_step = 5             # step to calculate horizon angle in degrees
search_length = 5000        # distance to search from center point (m)

# site information
# lat = 63.260281         # latitude of point of interest
# lon = -145.425720       # longitude of point of interest
site = 'D'
latlon_df = pd.read_csv('/home/claire/GulkanaDEM/gulkana_sites.csv',index_col=0)
lat = latlon_df.loc[site]['lat']
lon = latlon_df.loc[site]['lon']
timezone = pd.Timedelta(hours=-9)   # time zone of location
site_name = 'Gulkana' + site
min_elev = 1200         # rough estimate of lower elevation bound of DEM
max_elev = 2000         # rough estimate of upper elevation bound of DEM

# =================== FILEPATHS ===================
fp = '/home/claire/research/'
# in: need DEM, slope and aspect
# dem_fp = fp + '../GulkanaDEM/Gulkana_DEM_20m.tif'
# aspect_fp = fp + '../GulkanaDEM/Gulkana_Aspect_20m.tif'
# slope_fp = fp + '../GulkanaDEM/Gulkana_Slope_20m.tif'
dem_fp = fp + '../GulkanaDEM/2m/Gulkana_2m_DEM.tif'
aspect_fp = fp + '../GulkanaDEM/2m/Gulkana_2m_aspect.tif'
slope_fp = fp + '../GulkanaDEM/2m/Gulkana_2m_slope.tif'
# optional shapefile
shp_fp = fp + '../GulkanaDEM/Gulkana.shp'

# out
out_fp = fp + '../GulkanaDEM/Out/gulkana_centerpoint.csv'
out_image_fp = fp + f'../GulkanaDEM/Outputs/{site_name}.png'
out_horizon_fp = fp + f'../GulkanaDEM/Outputs/{site_name}_horizon.png'

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

# get UTM coordinates from lat/lon
transformer = Transformer.from_crs('EPSG:4326', dem.rio.crs, always_xy=True)
xx, yy = transformer.transform(lon, lat)
# get elevation of point from grid
point_elev = dem.sel(x=xx, y=yy, method="nearest").values

# get slope and aspect at point of interest
asp_deg = aspect.sel(x=xx, y=yy, method="nearest").values # slope azimuth angle (aspect)
asp = asp_deg * pi/180 + pi # aspect DEM treats 0 as north, we need as south
slp_deg = slope.sel(x=xx, y=yy, method="nearest").values # slope angle
slp = slp_deg * pi/180  # angles in radians
print(f'{site_name} point aspect: {asp_deg:.1f} o     slope: {slp_deg:.2f} o')

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
        plt.scatter(xs,ys,color=colors,s=5,marker='.',alpha=0.8)
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
store_vars = ['dirirr','dirirrslope','shaded','sun_elev','horizon_elev','sun_az']
year_hours = pd.date_range('2024-01-01 00:00','2024-12-31 23:00',freq='h')
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

    # get sun elevation and azimuth angle
    sunpos = suncalc.get_position(time_UTC,lon,lat)
    sun_elev = sunpos['altitude']       # solar elevation angle
    sun_az = sunpos['azimuth']          # solar azimuth angle

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
    varprops = {'dirirr':{'label':'direct flat-surface irradiance [W m-2]','cmap':'plasma'},
              'dirirrslope':{'label':'direct slope-corrected irradiance [W m-2]','cmap':'plasma'},
              'shaded':{'label':'shading [black = shaded]','cmap':'binary'},
              'sun_elev':{'label':'sun elevation angle [$\circ$]','cmap':'Spectral_r'},
              'horizon_elev':{'label':'horizon elevation angle [[$\circ$]','cmap':'YlOrRd'},
              'sun_az':{'label':'sun azimuth angle [$\circ$]','cmap':'twilight'}}
    for i,var in enumerate(plot_result):
        ax = axes[i]
        vardata = df[var].to_numpy().reshape((len(days),len(hours)))
        label = varprops[var]['label']
        cmap = varprops[var]['cmap']
        pc = ax.pcolormesh(days,hours,vardata.T, cmap=cmap)
        if var != 'shaded':
            clb = fig.colorbar(pc,ax=ax,aspect=10,pad=0.02)
            clb.set_label(var,loc='top')
        ax.set_ylabel('Hour of day')
        ax.set_xlabel('Day of year')
        site_name = site_name if site_name else f'{lat:.2f},{lon:.2f}'
        ax.set_title(label)
    fig.suptitle(f'Annual plots for {site_name} \n Sky-view factor: {sky_view:.3f}')
    if store_plot_result:
        plt.savefig(out_image_fp,dpi=150)
    else: 
        plt.show()

if store_result:
    df[['dirirrslope','shaded']].to_csv(out_fp)