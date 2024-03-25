"""
Created on Tue Mar 19 11:32:50 2024

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
7. Optional: calculate diffuse fraction from measured solar

@author: clairevwilson
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import rioxarray as rxr
import xarray as xr
import pandas as pd
import geopandas as gpd
import argparse
import suncalc
from pyproj import Transformer
from numpy import pi, cos, sin, arctan

# =================== INPUTS ===================
site_by = 'id'                      # method to choose lat/lon ('id' or 'latlon')
site = 'D'                         # name of site for indexing .csv OR
lat,lon = [63.3,-143.3]             # site lat/lon
timezone = pd.Timedelta(hours=-9)   # time zone of location
glacier_name = 'Gulkana'            # name of glacier for labeling

# model options
plot = ['result']           # False or list from ['result','search','horizon']
store = ['result','result_plot']          # False or list from ['result','result_plot','search_plot','horizon_plot']
result_vars = ['dirirrslope','shaded']
get_diffuse = True

# model parameters
angle_step = 5              # step to calculate horizon angle (degrees)
search_length = 5000        # distance to search from center point (m)
sub_dt = 10                 # timestep to calculate solar corrections (minutes)

# =================== PARSE ARGS ===================
parser = argparse.ArgumentParser(description='pygem-eb shading model')
parser.add_argument('-latitude','--lat',action='store',default=lat)
parser.add_argument('-longitude','--lon',action='store',default=lon)
parser.add_argument('-site',action='store',default=site)
parser.add_argument('-site_name',action='store',default=glacier_name+site)
parser.add_argument('-site_by',action='store',default=site_by)
parser.add_argument('-plot',action='store',default=plot)
parser.add_argument('-store',action='store',default=store)
args = parser.parse_args()

if args.site_by == 'id':
    # get site lat and lon    
    latlon_df = pd.read_csv('~/GulkanaDEM/gulkana_sites.csv',index_col=0)
    args.lat = latlon_df.loc[args.site]['lat']    # latitude of point of interest
    args.lon = latlon_df.loc[args.site]['lon']    # longitude of point of interest
    args.site_name = glacier_name + args.site

# =================== FILEPATHS ===================
fp = '/home/claire/'
# in
dem_fp = fp + 'GulkanaDEM/Gulkana_DEM_20m.tif'
# dem_fp = fp + 'GulkanaDEM/McCone/McCone_7km_1200-2500m_50m_surface.tif'
aspect_fp = fp + 'GulkanaDEM/Gulkana_Aspect_20m.tif'
slope_fp = fp + 'GulkanaDEM/Gulkana_Slope_20m.tif'
# optional shapefile for visualizing
shp_fp = fp + 'GulkanaDEM/Gulkana.shp'
# optional solar radiation file for diffuse fraction
solar_fp = fp + 'research/climate_data/AWS/CNR4/cnr4_2023.csv'

# out
out_fp = fp + f'GulkanaDEM/Out/{args.site_name}.csv'
out_image_fp = fp + f'GulkanaDEM/Outputs/{args.site_name}.png'
out_horizon_fp = fp + f'GulkanaDEM/Outputs/{args.site_name}_horizon.png'

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
xx, yy = transformer.transform(args.lon, args.lat)
# check point is in bounds
bounds = np.array(dem.rio.bounds())
x_in = xx >= bounds[0] and xx <= bounds[2]
y_in = yy >= bounds[1] and yy <= bounds[3]
assert x_in and y_in,'point out of raster bounds'
# get elevation of point from grid
point_elev = dem.sel(x=xx, y=yy, method='nearest').values

# get slope and aspect at point of interest
asp = aspect.sel(x=xx, y=yy, method='nearest').values * pi/180
slp = slope.sel(x=xx, y=yy, method='nearest').values * pi/180
print(f'{args.site_name} point aspect: {asp*180/pi:.1f} o     slope: {slp*180/pi:.2f} o')

# =================== CONSTANTS ===================
I0 = 1368       # solar constant in W m-2
P0 = 101325     # sea-level pressure in Pa
PSI = 0.75      # vertical atmospheric clear-sky transmissivity
MEAN_RAD = 1    # mean earth-sun radius in AU

# =================== FUNCTIONS ===================
def r_sun(time):
    """Gets earth-to-sun radius in AU"""
    doy = time.day_of_year
    radius = 1 - 0.01672*cos(0.9856*(doy-4))
    return radius

def pressure(elev):
    """Adjusts air pressure by elevation"""
    P = np.exp(-0.0001184*elev)*P0
    return P

def zenith(time):
    """Calculates solar zenith angle for time, lat and lon"""
    time_UTC = time - timezone
    lon = args.lon
    lat = args.lat
    altitude_angle = suncalc.get_position(time_UTC,lon,lat)['altitude']
    zenith = pi/2 - altitude_angle if altitude_angle > 0 else np.nan
    return zenith

def declination(time):
    """Calculates solar declination"""
    doy = time.day_of_year
    delta = -23.4*cos(360*(doy+10)/365) * pi/180
    return delta

hour_angle = lambda t: 15*(12-t.hour)

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

def find_horizon(elev,xs,ys,buffer = 10):
    """Finds the horizon along a line of elevation
    values paired to x,y coordinates
    - elev: array of elevations in a single direction
    - xs, ys: coordinates corresponding to elev
    - buffer: minimum number of gridcells away the horizon 
            can be found (needed when looking uphill)"""
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
    horizon_x = xs[idx]
    horizon_y = ys[idx]
    return horizon_angle,horizon_x,horizon_y,elev_angles

# =================== HORIZON ===================
# plot DEM as background
if 'search' in args.plot:
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
    if 'search' in args.plot:
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
if 'search' in args.plot:
    plt.scatter(xx,yy,marker='*',color='orange')
    plt.title(f'{args.site_name} Horizon Search \n Sky-view Factor = {sky_view:.3f}')
    plt.ylabel('Northing')
    plt.xlabel('Easting')
    if 'search_plot' in args.store:
        plt.savefig(out_horizon_fp)
    else:
        plt.show()

# =================== IRRADIANCE ===================
# loop through hours of the year and store data
store_vars = ['dirirr','dirirrslope','shaded','corr_factor','sun_elev','horizon_elev','sun_az']
year_hours = pd.date_range('2024-01-01 00:00','2024-12-31 23:00',freq='h')
df = pd.DataFrame(data = np.ones((8784,len(store_vars))),
                  columns=store_vars,index=year_hours)
for time in year_hours:
    # loop to get sub-hourly values and average
    sub_vars = ['shaded','Islope','I','corr_factor','zenith']
    period_dict = {}
    for var in sub_vars:
        period_dict[var] = np.array([])
    for minutes in np.arange(0,60,sub_dt):
        # calculate time-dependent variables
        time_UTC = time - timezone + pd.Timedelta(minutes = minutes)
        P = pressure(point_elev)
        r = r_sun(time + pd.Timedelta(minutes = minutes))
        Z = zenith(time + pd.Timedelta(minutes = minutes))
        d = declination(time + pd.Timedelta(minutes = minutes))
        h = hour_angle(time + pd.Timedelta(minutes = minutes))
        period_dict['zenith'] = np.append(period_dict['zenith'],Z)

        # calculate direct clear-sky irradiance (not slope corrected)
        I = I0 * (MEAN_RAD/r)**2 * PSI**(P/P0/np.cos(Z)) * np.cos(Z)
        period_dict['I'] = np.append(period_dict['I'],I)

        # get sun elevation and azimuth angle
        sunpos = suncalc.get_position(time_UTC,args.lon,args.lat)
        sun_elev = sunpos['altitude']       # solar elevation angle
        # suncalc gives azimuth with 0 = South, we want 0 = North
        sun_az = sunpos['azimuth'] + pi     # solar azimuth angle

        # get nearest angle of horizon calculations to the sun azimuth
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

    # extract sub-hourly-timestep arrays
    I = period_dict['I'][~np.isnan(period_dict['I'])]
    Islope = period_dict['Islope']
    cosZ = cos(period_dict['zenith'])
    corrf = period_dict['corr_factor']
    dt = np.ones(len(Islope)) * sub_dt

    # find hourly means (avoid nans)
    if ~np.any(np.isnan(Islope)):
        mean_I = np.sum(Islope*cosZ*corrf*dt) / np.sum(dt)
        if np.sum(Islope*cosZ) > 0:
            mean_corr_factor = np.sum(Islope*cosZ*corrf) / np.sum(Islope*cosZ)
        else:
            mean_corr_factor = 5
    else:
        mean_I = 0
        mean_corr_factor = 0
    median_shaded = int(np.median(period_dict['shaded']))

    # store data
    df.loc[time,'shaded'] = median_shaded
    df.loc[time,'dirirrslope'] = mean_I
    df.loc[time,'corr_factor'] = mean_corr_factor
    try:
        df.loc[time,'dirirr'] = np.mean(I)
    except:
        df.loc[time,'dirirr'] = np.nan

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
if 'result' in args.plot:
    # initialize plot
    nrows = 2 if len(result_vars) > 2 else 1
    ncols = len(result_vars) if len(result_vars) <= 2 else int(len(result_vars)/2)
    fig,axes = plt.subplots(nrows,ncols,figsize=(ncols*4,nrows*3),layout='constrained')
    axes = axes.flatten()
    fig.suptitle(f'Annual plots for {args.site_name} \n Sky-view factor: {sky_view:.3f}')
    
    # get days and hours of the year
    days = np.arange(366)
    hours = np.arange(0,24)
    
    # loop through variables to plot
    for i,var in enumerate(result_vars):
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
        ax.set_title(label)
    
    # store or show plot
    if 'result_plot' in args.store:
        plt.savefig(out_image_fp,dpi=150)
    else: 
        plt.show()

if 'horizon' in args.plot:
    fig,ax = plt.subplots()
    ax.fill_between(angles,horizon_elev*180/pi,color='black',alpha=0.6)
    ax.plot(angles,horizon_elev*180/pi,color='black')
    ax.set_xlabel('Azimuth angle ($\circ$)')
    ax.set_ylabel('Horizon angle ($\circ$)')
    fig.suptitle(args.site_name+' shading by azimuth angle (0$^{\circ}$N)',fontsize=14)
    if 'horizon_plot' in args.store:
        plt.savefig(f'/home/claire/GulkanaDEM/Outputs/{args.site_name}_angles.png')
    else:
        plt.show()

if 'result' in args.store:
    df[['dirirrslope','shaded']].to_csv(out_fp)

# =================== DIFFUSE ===================
if get_diffuse:
    # load solar dataset
    solar_df = pd.read_csv(solar_fp,index_col=0)
    solar_df.index = pd.to_datetime(solar_df.index) + pd.Timedelta(days=365,hours=9)

    # get hours where station is in the shade
    measured_list = []
    modeled_list = []
    for hour in solar_df.index:
        shaded = bool(df.loc[hour,'shaded'])
        modeled = df.loc[hour,'dirirr']
        if shaded and ~np.isnan(modeled):
            measured = solar_df.loc[hour,'sw_down_Avg']
            if modeled > 1e-3:
                measured_list.append(measured)
                modeled_list.append(modeled)
                print(measured,modeled)

    # plot
    mean_frac = np.mean(measured / modeled)
    plt.figure(figsize=(8,5))
    plt.scatter(measured,modeled)
    plt.plot([0,np.max(measured)],[0,np.max(measured)])
    plt.title(f'Mean diffuse fraction: {mean_frac}')
    plt.savefig(fp + 'GulkanaDEM/Outputs/diffuseD.png',dpi=150)