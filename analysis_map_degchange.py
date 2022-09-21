#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created based off of Hugonnet et al. (2021):
    https://github.com/rhugonnet/ww_tvol_study/blob/main/figures/fig_2_world_dh_vectorized_smallfonts.py
"""


import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.path as mpath
from matplotlib.patheffects import Stroke
import shapely.geometry as sgeom
import matplotlib.patches as mpatches
import os
import pandas as pd
import numpy as np
import gdal, osr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pickle
from scipy.ndimage import uniform_filter
from scipy.spatial import ConvexHull
#from pyddem.vector_tools import SRTMGL1_naming_to_latlon, latlon_to_SRTMGL1_naming, geoimg_mask_on_feat_shp_ds, create_mem_shp, poly_from_coords
#from pybob.image_tools import create_mask_from_shapefile
#from pybob.GeoImg import GeoImg
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature

import pygem.pygem_input as pygem_prms
import pygem.pygem_modelsetup as modelsetup


option_global_vol_remaining_byscenario = False   # Option to plot global map of volume remaining by rcp/ssp scenarios
option_global_vol_remaining_bydeg = True       # Option to plot global map of volume remaining by degrees (e.g, +2, +3, etc.)

rgi_shp_fn = '/Users/drounce/Documents/Papers/pygem_oggm_global/qgis/rgi60_all_simplified2_robinson.shp'

mpl.use('Agg')

plt.rcParams.update({'font.size': 5})
plt.rcParams.update({'lines.linewidth':0.5})
plt.rcParams.update({'axes.linewidth':0.5})
plt.rcParams.update({'pdf.fonttype':42})

group_by_spec = True



class MidpointNormalize(mpl.colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        mpl.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # Note that I'm ignoring clipping and other edge cases here.
        result, is_scalar = self.process_value(value)
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.array(np.interp(value, x, y), mask=result.mask, copy=False)


#%%
if option_global_vol_remaining_byscenario:
    
    regions = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
#    regions = [11]

    add_rgi_glaciers = True
    
    normyear = 2015
    
    # GCMs and RCP scenarios
    gcm_names_rcps = ['CanESM2', 'CCSM4', 'CNRM-CM5', 'CSIRO-Mk3-6-0', 'GFDL-CM3', 
                      'GFDL-ESM2M', 'GISS-E2-R', 'IPSL-CM5A-LR', 'MPI-ESM-LR', 'NorESM1-M']
    gcm_names_ssps = ['BCC-CSM2-MR', 'CESM2', 'CESM2-WACCM', 'EC-Earth3', 'EC-Earth3-Veg', 'FGOALS-f3-L', 
                      'GFDL-ESM4', 'INM-CM4-8', 'INM-CM5-0', 'MPI-ESM1-2-HR', 'MRI-ESM2-0', 'NorESM2-MM']
    gcm_names_ssp119 = ['EC-Earth3', 'EC-Earth3-Veg', 'GFDL-ESM4', 'MRI-ESM2-0']
    #rcps = ['rcp26', 'rcp45', 'rcp85']
    rcps = ['ssp126', 'ssp245', 'ssp370', 'ssp585']
#    rcps = ['ssp119', 'ssp126', 'ssp245', 'ssp370', 'ssp585']
#    rcps = ['ssp126']
    
    years = np.arange(2000,2102)
    
    # Colors and bounds
    col_bounds = np.array([0, 0.005, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    color_shades_to_inset = 'black'
#    color_water = 'lightcyan'
#    color_land = 'gainsboro'
#    color_water = 'gainsboro'
#    color_land = 'darkgrey'
    color_water='lightblue'
    color_land='white'
    
    
    netcdf_fp_cmip5_land = '/Users/drounce/Documents/HiMAT/spc_backup/simulations/'
    netcdf_fp_cmip5 = '/Users/drounce/Documents/HiMAT/spc_backup/simulations_calving_v5/'
    analysis_fp = netcdf_fp_cmip5.replace('simulations','analysis')
    fig_fp = analysis_fp + 'figures/multi_gcm/'
    if not os.path.exists(fig_fp):
        os.makedirs(fig_fp, exist_ok=True)
    csv_fp = analysis_fp + 'csv/'
    if not os.path.exists(csv_fp):
        os.makedirs(csv_fp, exist_ok=True)
    pickle_fp = analysis_fp + 'pickle/'
    pickle_fp_land = netcdf_fp_cmip5_land.replace('simulations','analysis') + 'pickle/'
    
    
    reg_vol_all_deg = {}
    reg_area_all_deg = {}
    ds_multigcm_area = {}
    ds_multigcm_vol = {}
    latlon_df = {}
    tiles_all = []
    
    # Set up all
    ds_multigcm_vol['all'] = {}
    ds_multigcm_area['all'] = {}
    for rcp in rcps:
        ds_multigcm_vol['all'][rcp] = None
        ds_multigcm_area['all'][rcp] = None
    
    # Process regions
    for reg in regions:

        # Glaciers
        fn_reg_glacno_list = 'R' + str(reg) + '_glacno_list.pkl'
        with open(pickle_fp_land + str(reg).zfill(2) + '/' + fn_reg_glacno_list, 'rb') as f:
            glacno_list = pickle.load(f)
        
        # Degree ID dict
        fn_unique_degids = 'R' + str(reg) + '_unique_degids.pkl'
        with open(pickle_fp_land + str(reg).zfill(2) + '/' + fn_unique_degids, 'rb') as f:
            unique_degids = pickle.load(f)
            
        # All glaciers for fraction
        # Glaciers with successful runs to process
        main_glac_rgi = modelsetup.selectglaciersrgitable(glac_no=glacno_list)
            
        # ----- Add Groups -----
        # Degrees (based on degree_size)
        degree_size_pkl = 0.1
        main_glac_rgi['CenLon_round'] = np.floor(main_glac_rgi.CenLon.values/degree_size_pkl) * degree_size_pkl
        main_glac_rgi['CenLat_round'] = np.floor(main_glac_rgi.CenLat.values/degree_size_pkl) * degree_size_pkl
        deg_groups = main_glac_rgi.groupby(['CenLon_round', 'CenLat_round']).size().index.values.tolist()
        deg_dict = dict(zip(deg_groups, np.arange(0,len(deg_groups))))
        main_glac_rgi.reset_index(drop=True, inplace=True)
        cenlon_cenlat = [(main_glac_rgi.loc[x,'CenLon_round'], main_glac_rgi.loc[x,'CenLat_round']) 
                         for x in range(len(main_glac_rgi))]
        main_glac_rgi['CenLon_CenLat'] = cenlon_cenlat
        main_glac_rgi['deg_id'] = main_glac_rgi.CenLon_CenLat.map(deg_dict)
    
        reg_vol_all_deg[reg] = {}
        reg_area_all_deg[reg] = {}
        ds_multigcm_vol[reg] = {}
        ds_multigcm_area[reg] = {}
        latlon_df[reg] = None
        for nrcp, rcp in enumerate(rcps):
            reg_vol_all_deg[reg][rcp] = {}
            reg_area_all_deg[reg][rcp] = {}
          
            if 'rcp' in rcp:
                gcm_names = gcm_names_rcps
            elif 'ssp' in rcp:
                if rcp in ['ssp119']:
                    gcm_names = gcm_names_ssp119
                else:
                    gcm_names = gcm_names_ssps
                
            for ngcm, gcm_name in enumerate(gcm_names):
#                netcdf_fp_cmip5_reg = netcdf_fp_cmip5 + str(reg).zfill(2)
#                if '_calving' in netcdf_fp_cmip5 and not os.path.exists(netcdf_fp_cmip5_reg):
#                    pickle_fp_degid =  (netcdf_fp_cmip5_land + '../analysis/pickle/' + str(reg).zfill(2) + 
#                                        '/degids/' + gcm_name + '/' + rcp + '/')
#                else:
#                    pickle_fp_degid =  pickle_fp + str(reg).zfill(2) + '/degids/' + gcm_name + '/' + rcp + '/'
                    
                if '_calving' in netcdf_fp_cmip5 and reg in [2,6,8,10,11,12,13,14,15,16,18]:
#                if '_calving' in netcdf_fp_cmip5 and reg in [2,6,8,10,11,12,13,14,15,16,18]:
                    pickle_fp_degid =  pickle_fp_land + str(reg).zfill(2) + '/degids/' + gcm_name + '/' + rcp + '/'
                else:
                    pickle_fp_degid =  pickle_fp + str(reg).zfill(2) + '/degids/' + gcm_name + '/' + rcp + '/'
                
                # ----- GCM/RCP PICKLE FILEPATHS AND FILENAMES -----
                # Region string prefix
                degid_rcp_gcm_str = 'R' + str(reg) + '_degids_' + rcp + '_' + gcm_name            
                # Pickle Filenames
                fn_degid_vol_annual = degid_rcp_gcm_str + '_vol_annual.pkl'
                fn_degid_area_annual = degid_rcp_gcm_str + '_area_annual.pkl'
                    
                # Volume
                with open(pickle_fp_degid + fn_degid_vol_annual, 'rb') as f:
                    degid_vol_annual = pickle.load(f)
                # Area
                with open(pickle_fp_degid + fn_degid_area_annual, 'rb') as f:
                    degid_area_annual = pickle.load(f)
                
                # Aggregate to desired scale
                lat_min = main_glac_rgi['CenLat_round'].min()
                lat_max = main_glac_rgi['CenLat_round'].max()
                lon_min = main_glac_rgi['CenLon_round'].min()
                lon_max = main_glac_rgi['CenLon_round'].max()
                if ngcm == 0:
                    print('lat/lon min/max:', lat_min, lat_max, lon_min, lon_max)
                
                degree_size_lon = 1
                degree_size_lat= 1
                lat_start = np.round(lat_min/degree_size_lat)*degree_size_lat
                lat_end = np.round(lat_max/degree_size_lat)*degree_size_lat
                lon_start = np.round(lon_min/degree_size_lon)*degree_size_lon
                lon_end = np.round(lon_max/degree_size_lon)*degree_size_lon
                
                df_degid_vol_annual = pd.DataFrame(degid_vol_annual, columns=years)
                df_degid_vol_annual['deg_id'] = unique_degids
                
                degid_latlon_df = pd.DataFrame(np.zeros((len(unique_degids),2)), columns=['CenLon_round','CenLat_round'])
                for nrow, degid in enumerate(unique_degids):
                    main_glac_rgi_subset = main_glac_rgi.loc[main_glac_rgi['deg_id'] == degid]
                    main_glac_rgi_subset.reset_index(inplace=True, drop=True)
                    degid_latlon_df.loc[nrow,'CenLon_round'] = main_glac_rgi_subset.loc[0,'CenLon_round']
                    degid_latlon_df.loc[nrow,'CenLat_round'] = main_glac_rgi_subset.loc[0,'CenLat_round']
                    
                agg_degid_vol_annual = None
                agg_degid_area_annual = None
                agg_lonlat_list = []
                count = 0
                for lon in np.arange(lon_start, lon_end+degree_size_lon/2, degree_size_lon):
                    for lat in np.arange(lat_start, lat_end+degree_size_lat/2, degree_size_lat):
                        degid_latlon_df_subset = degid_latlon_df.loc[(degid_latlon_df['CenLon_round'] >= lon - degree_size_lon/2) & 
                                                                     (degid_latlon_df['CenLon_round'] < lon + degree_size_lon/2) &
                                                                     (degid_latlon_df['CenLat_round'] >= lat - degree_size_lat/2) &
                                                                     (degid_latlon_df['CenLat_round'] < lat + degree_size_lat/2)]
                        array_idx = degid_latlon_df_subset.index.values
                        
                        if len(array_idx) > 0:
                            agg_degid_vol_annual_single = degid_vol_annual[array_idx,:].sum(0)
                            agg_degid_area_annual_single = degid_area_annual[array_idx,:].sum(0)
                            agg_lonlat_list.append([lon, lat])
                            
                            if agg_degid_vol_annual is None:
                                agg_degid_vol_annual = agg_degid_vol_annual_single
                                agg_degid_area_annual = agg_degid_area_annual_single
                            else:
                                agg_degid_vol_annual = np.vstack([agg_degid_vol_annual, agg_degid_vol_annual_single])
                                agg_degid_area_annual = np.vstack([agg_degid_area_annual, agg_degid_area_annual_single])
                            
                            count += 1
    #                        print(count, lon, lat, np.round(agg_degid_area_annual_single[0]/1e6,1))
                
    #            agg_degid_vol_df_cns = ['CenLon_round', 'CenLat_round', 'Lon_size', 'Lat_size']
    #            for year in years:
    #                agg_degid_vol_df_cns.append(year)
    #            agg_degid_vol_df = pd.DataFrame(np.zeros((agg_degid_vol_annual.shape[0],len(agg_degid_vol_df_cns))), 
    #                                            columns=agg_degid_vol_df_cns)
    #            agg_degid_vol_df['CenLon_round'] = [x[0] for x in agg_lonlat_list]
    #            agg_degid_vol_df['CenLat_round'] = [x[1] for x in agg_lonlat_list]
    #            agg_degid_vol_df['Lon_size'] = degree_size_lon
    #            agg_degid_vol_df['Lat_size'] = degree_size_lat
    #            agg_degid_vol_df.loc[:,years] = agg_degid_vol_annual 
                
                    
                # Record datasets
    #            if latlon_df[reg] is None:
    #                latlon_df[reg] = agg_degid_vol_df.loc[:,['CenLon_round', 'CenLat_round', 'Lon_size', 'Lat_size']]
                reg_vol_all_deg[reg][rcp][gcm_name] = agg_degid_vol_annual
                reg_area_all_deg[reg][rcp][gcm_name] = agg_degid_area_annual
                
                if ngcm == 0:
                    reg_vol_gcm_all = agg_degid_vol_annual[np.newaxis,:,:]
                    reg_area_gcm_all = agg_degid_area_annual[np.newaxis,:,:]
                else:
                    reg_vol_gcm_all = np.vstack((reg_vol_gcm_all, agg_degid_vol_annual[np.newaxis,:,:]))
                    reg_area_gcm_all = np.vstack((reg_area_gcm_all, agg_degid_area_annual[np.newaxis,:,:]))
                    
            ds_multigcm_vol[reg][rcp] = reg_vol_gcm_all
            ds_multigcm_area[reg][rcp] = reg_area_gcm_all
            
            if ds_multigcm_vol['all'][rcp] is None:
                ds_multigcm_vol['all'][rcp] = np.median(reg_vol_gcm_all, axis=0)
                ds_multigcm_area['all'][rcp] = np.median(reg_area_gcm_all, axis=0)
            else:
                ds_multigcm_vol['all'][rcp] = np.concatenate((ds_multigcm_vol['all'][rcp], np.median(reg_vol_gcm_all, axis=0)), axis=0)
                ds_multigcm_area['all'][rcp] = np.concatenate((ds_multigcm_area['all'][rcp], np.median(reg_area_gcm_all, axis=0)), axis=0)
    
            # Tiles are a list of the lat/lon 
            if nrcp == 0:
                tiles = agg_lonlat_list
                for tile in tiles:
                    tiles_all.append(tile)

#%%
    for rcp in rcps:
        print(rcp)
        # Tiles are lat/lon
        normyear_idx = np.where(years == normyear)[0][0]
        
        # Area are in km2
        areas = ds_multigcm_area['all'][rcp][:,normyear_idx] / 1e6
        areas_2100 = ds_multigcm_area['all'][rcp][:,-1] / 1e6
        
        # dh is the elevation change that is used for color; we'll use volume remaining by end of century for now
        vols_remaining_frac = ds_multigcm_vol['all'][rcp][:,-1] / ds_multigcm_vol['all'][rcp][:,normyear_idx]
        dhs = vols_remaining_frac.copy()
    
        areas = [area for _, area in sorted(zip(tiles_all,areas))]
        areas_2100 = [area for _, area in sorted(zip(tiles_all,areas_2100))]
        dhs = [dh for _, dh in sorted(zip(tiles_all,dhs))]
        tiles = sorted(tiles_all)
        
        def latlon_extent_to_axes_units(extent):
        
            extent = np.array(extent)
        
            lons = (extent[0:2] + 179.9) / 359.8
            lats = (extent[2:4] + 89.9) / 179.8
        
            return [lons[0],lons[1],lats[0],lats[1]]
        
        def axes_pos_to_rect_units(units):
        
            return [min(units[0:2]),min(units[2:4]),max(units[0:2])-min(units[0:2]),max(units[2:4])-min(units[2:4])]
        
        def rect_units_to_verts(rect_u):
        
            return np.array([[rect_u[0],rect_u[1]],[rect_u[0]+rect_u[2],rect_u[1]],[rect_u[0]+rect_u[2],rect_u[1] +rect_u[3]],[rect_u[0],rect_u[1]+rect_u[3]],[rect_u[0],rect_u[1]]])
        
        def coordXform(orig_crs, target_crs, x, y):
            return target_crs.transform_points( orig_crs, x, y )
        
        def poly_from_extent(ext):
        
            poly = np.array([(ext[0],ext[2]),(ext[1],ext[2]),(ext[1],ext[3]),(ext[0],ext[3]),(ext[0],ext[2])])
        
            return poly
        
        def latlon_extent_to_robinson_axes_verts(polygon_coords):
        
            list_lat_interp = []
            list_lon_interp = []
            for i in range(len(polygon_coords)-1):
                lon_interp = np.linspace(polygon_coords[i][0],polygon_coords[i+1][0],50)
                lat_interp =  np.linspace(polygon_coords[i][1],polygon_coords[i+1][1],50)
        
                list_lon_interp.append(lon_interp)
                list_lat_interp.append(lat_interp)
        
            all_lon_interp = np.concatenate(list_lon_interp)
            all_lat_interp = np.concatenate(list_lat_interp)
        
            robin = coordXform(ccrs.PlateCarree(),ccrs.Robinson(),all_lon_interp,all_lat_interp)
        
            limits_robin = coordXform(ccrs.PlateCarree(),ccrs.Robinson(),np.array([-179.99,179.99,0,0]),np.array([0,0,-89.99,89.99]))
        
            ext_robin_x = limits_robin[1][0] - limits_robin[0][0]
            ext_robin_y = limits_robin[3][1] - limits_robin[2][1]
        
            verts = robin.copy()
            verts[:,0] = (verts[:,0] + limits_robin[1][0])/ext_robin_x
            verts[:,1] = (verts[:,1] + limits_robin[3][1])/ext_robin_y
        
            return verts[:,0:2]
        
        def shades_main_to_inset(main_pos,inset_pos,inset_verts,label):
        
            center_x = main_pos[0] + main_pos[2]/2
            center_y = main_pos[1] + main_pos[3]/2
        
            left_x = center_x - inset_pos[2]/2
            left_y = center_y - inset_pos[3]/2
        
            shade_ax = fig.add_axes([left_x,left_y,inset_pos[2],inset_pos[3]],projection=ccrs.Robinson(),label=label+'shade')
            shade_ax.set_extent([-179.99,179.99,-89.99,89.99],ccrs.PlateCarree())
        
            #first, get the limits of the manually positionned exploded polygon in projection coordinates
            limits_robin = coordXform(ccrs.PlateCarree(), ccrs.Robinson(), np.array([-179.99, 179.99, 0, 0]),
                                      np.array([0, 0, -89.99, 89.99]))
        
            ext_robin_x = limits_robin[1][0] - limits_robin[0][0]
            ext_robin_y = limits_robin[3][1] - limits_robin[2][1]
        
            inset_mod_x = inset_verts[:,0] +  (inset_pos[0]-left_x)/inset_pos[2]
            inset_mod_y = inset_verts[:,1] +  (inset_pos[1]-left_y)/inset_pos[3]
        
            #then, get the limits of the polygon in the manually positionned center map
            main_mod_x = (inset_verts[:, 0]*main_pos[2] - left_x + main_pos[0])/inset_pos[2]
            main_mod_y = (inset_verts[:, 1]*main_pos[3] - left_y + main_pos[1])/inset_pos[3]
        
            points = np.array(list(zip(np.concatenate((inset_mod_x,main_mod_x)),np.concatenate((inset_mod_y,main_mod_y)))))
        
            chull = ConvexHull(points)
        
            chull_robin_x = points[chull.vertices,0]*ext_robin_x - limits_robin[1][0]
            chull_robin_y = points[chull.vertices,1]*ext_robin_y - limits_robin[3][1]
        
        #    col_contour = mpl.cm.Greys(0.8)
        
#            shade_ax.plot(main_mod_x*ext_robin_x - limits_robin[1][0],main_mod_y*ext_robin_y - limits_robin[3][1],color='white',linewidth=0.75)
            shade_ax.plot(main_mod_x*ext_robin_x - limits_robin[1][0],main_mod_y*ext_robin_y - limits_robin[3][1],color='k',linewidth=0.75)
#            shade_ax.fill(chull_robin_x, chull_robin_y, transform=ccrs.Robinson(), color=color_shades_to_inset, alpha=0.05, zorder=1)
            verts = mpath.Path(np.column_stack((chull_robin_x,chull_robin_y)))
            shade_ax.set_boundary(verts, transform=shade_ax.transAxes)
        
        def only_shade(position,bounds,label,polygon=None):
            main_pos = [0.375, 0.21, 0.25, 0.25]
        
            if polygon is None and bounds is not None:
                polygon = poly_from_extent(bounds)
        
            shades_main_to_inset(main_pos, position, latlon_extent_to_robinson_axes_verts(polygon), label=label)
        
        def add_inset(fig,extent,position,bounds=None,label=None,polygon=None,shades=True, hillshade=True, list_shp=None, main=False, markup=None,markpos='left',markadj=0,markup_sub=None,sub_pos='lt',
                      col_bounds=None, color_water=color_water, color_land=color_land, add_rgi_glaciers=False):
            main_pos = [0.375, 0.21, 0.25, 0.25]
        
            if polygon is None and bounds is not None:
                polygon = poly_from_extent(bounds)
        
            if shades:
                shades_main_to_inset(main_pos, position, latlon_extent_to_robinson_axes_verts(polygon), label=label)
        
            sub_ax = fig.add_axes(position,
                                  projection=ccrs.Robinson(),label=label)
            sub_ax.set_extent(extent, ccrs.Geodetic())
        
            sub_ax.add_feature(cfeature.NaturalEarthFeature('physical', 'ocean', '50m', facecolor=color_water))
            sub_ax.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '50m', facecolor=color_land))
            
            # Add RGI glacier outlines
            if add_rgi_glaciers:
                shape_feature = ShapelyFeature(Reader(rgi_shp_fn).geometries(), ccrs.Robinson(),alpha=1,facecolor='indigo',linewidth=0.35,edgecolor='indigo')
                sub_ax.add_feature(shape_feature)
        
            if bounds is not None:
                verts = mpath.Path(latlon_extent_to_robinson_axes_verts(polygon))
                sub_ax.set_boundary(verts, transform=sub_ax.transAxes)
        
            # HERE IS WHERE VALUES APPEAR TO BE PROVIDED
            if not main:
                print(label)
                for i in range(len(tiles)):
                    lon = tiles[i][0]
                    lat = tiles[i][1]
        
                    if label=='Arctic West' and ((lat < 71 and lon > 60) or (lat <76 and lon>100)):
                        continue
        
                    if label=='HMA' and lat >=46:
                        continue
        
                    # fac = 0.02
                    fac = 1000
                    
                    area_2100 = areas_2100[i]
        
                    if areas[i] > 10:
                        rad = 15000 + np.sqrt(areas[i]) * fac
                    else:
                        rad = 15000 + 10 * fac
                    cb = []
                    cb_val = np.linspace(0, 1, len(col_bounds))
                    for j in range(len(cb_val)):
                        cb.append(mpl.cm.RdYlBu(cb_val[j]))
##                    cb[5] = cb[4]
##                    cb[4] = cb[3]
##                    cb[3] = cb[2]
##                    cb[2] = cb[1]
##                    cb[1] = cb[0]
#                    cb[11] = (78/256, 179/256, 211/256, 1)
#                    cb[10] = (123/256, 204/256, 196/256, 1)
#                    cb[9] = (168/256, 221/256, 181/256, 1)
#                    cb[8] = (204/256, 235/256, 197/256, 1)
#                    cb[7] = (224/256, 243/256, 219/256, 1)
#                    cb[6] = (254/256, 178/256, 76/256, 1)
#                    cb[5] = (253/256, 141/256, 60/256, 1)
#                    cb[4] = (252/256, 78/256, 42/256, 1)
#                    cb[3] = (227/256, 26/256, 28/256, 1)
#                    cb[2] = (189/256, 0/256, 38/256, 1)
#                    cb[1] = (128/256, 0/256, 38/256, 1)
#                    cb[0] = (1,1,1,1)
                    cb[11] = (94/256, 79/256, 162/256, 1)
                    cb[10] = (50/256, 136/256, 189/256, 1)
                    cb[9] = (102/256, 194/256, 165/256, 1)
                    cb[8] = (171/256, 221/256, 164/256, 1)
                    cb[7] = (230/256, 245/256, 152/256, 1)
                    cb[6] = (255/256, 255/256, 191/256, 1)
                    cb[5] = (254/256, 224/256, 139/256, 1)
                    cb[4] = (253/256, 174/256, 97/256, 1)
                    cb[3] = (244/256, 109/256, 67/256, 1)
                    cb[2] = (213/256, 62/256, 79/256, 1)
                    cb[1] = (158/256, 1/256, 66/256, 1)
                    cb[0] = (1,1,1,1)
                    cmap_cus = mpl.colors.LinearSegmentedColormap.from_list('my_cb', list(
                        zip(col_bounds, cb)))
        
                    # xy = [lon,lat]
                    xy = coordXform(ccrs.PlateCarree(),ccrs.Robinson(),np.array([lon]),np.array([lat]))[0][0:2]
                    
                    # Less than percent threshold and area threshold
                    dhdt = dhs[i]
                    col = cmap_cus(dhdt)
                    if dhdt < col_bounds[1] or area_2100 < 0.005:
                        sub_ax.add_patch(
                            mpatches.Circle(xy=xy, radius=rad, facecolor='white', edgecolor='black', linewidth=0.5, alpha=1, transform=ccrs.Robinson(), zorder=30))
                    else:
                        sub_ax.add_patch(
                            mpatches.Circle(xy=xy, radius=rad, facecolor=col, edgecolor='None', alpha=1, transform=ccrs.Robinson(), zorder=30))
                    
            if markup is not None:
                if markpos=='left':
                    lon_upleft = np.min(list(zip(*polygon))[0])
                    lat_upleft = np.max(list(zip(*polygon))[1])
                else:
                    lon_upleft = np.max(list(zip(*polygon))[0])
                    lat_upleft = np.max(list(zip(*polygon))[1])
        
                robin = coordXform(ccrs.PlateCarree(),ccrs.Robinson(),np.array([lon_upleft]),np.array([lat_upleft]))
        
                rob_x = robin[0][0]
                rob_y = robin[0][1]
        
                size_y = 200000
                size_x = 80000 * len(markup) + markadj
        
                if markpos=='right':
                    rob_x = rob_x-50000
                else:
                    rob_x = rob_x+50000
        
                sub_ax_2 = fig.add_axes(position,
                                        projection=ccrs.Robinson(), label=label+'markup')
        
                # adds the white box to the region
#                sub_ax_2.add_patch(mpatches.Rectangle((rob_x, rob_y), size_x, size_y , linewidth=1, edgecolor='grey', facecolor='white',transform=ccrs.Robinson()))
        
                sub_ax_2.set_extent(extent, ccrs.Geodetic())
                verts = mpath.Path(rect_units_to_verts([rob_x,rob_y,size_x,size_y]))
                sub_ax_2.set_boundary(verts, transform=sub_ax.transAxes)
        
                sub_ax_2.text(rob_x,rob_y+50000,markup,
                         horizontalalignment=markpos, verticalalignment='bottom',
                         transform=ccrs.Robinson(), color='black',fontsize=4.5, fontweight='bold',bbox= dict(facecolor='white', alpha=1,linewidth=0.35,pad=1.5))
        
            if markup_sub is not None:
        
                lon_min = np.min(list(zip(*polygon))[0])
                lon_max = np.max(list(zip(*polygon))[0])
                lon_mid = 0.5*(lon_min+lon_max)
        
                lat_min = np.min(list(zip(*polygon))[1])
                lat_max = np.max(list(zip(*polygon))[1])
                lat_mid = 0.5*(lat_min+lat_max)
        
                size_y = 150000
                size_x = 150000
        
                lat_midup = lat_min+0.87*(lat_max-lat_min)
        
                robin = coordXform(ccrs.PlateCarree(),ccrs.Robinson(),np.array([lon_min,lon_min,lon_min,lon_mid,lon_mid,lon_max,lon_max,lon_max,lon_min]),np.array([lat_min,lat_mid,lat_max,lat_min,lat_max,lat_min,lat_mid,lat_max,lat_midup]))
        
                if sub_pos=='lb':
                    rob_x = robin[0][0]
                    rob_y = robin[0][1]
                    ha='left'
                    va='bottom'
                elif sub_pos=='lm':
                    rob_x = robin[1][0]
                    rob_y = robin[1][1]
                    ha='left'
                    va='center'
                elif sub_pos == 'lm2':
                    rob_x = robin[8][0]
                    rob_y = robin[8][1]
                    ha = 'left'
                    va = 'center'
                elif sub_pos=='lt':
                    rob_x = robin[2][0]
                    rob_y = robin[2][1]
                    ha='left'
                    va='top'
                elif sub_pos=='mb':
                    rob_x = robin[3][0]
                    rob_y = robin[3][1]
                    ha='center'
                    va='bottom'
                elif sub_pos=='mt':
                    rob_x = robin[4][0]
                    rob_y = robin[4][1]
                    ha='center'
                    va='top'
                elif sub_pos=='rb':
                    rob_x = robin[5][0]
                    rob_y = robin[5][1]
                    ha='right'
                    va='bottom'
                elif sub_pos=='rm':
                    rob_x = robin[6][0]
                    rob_y = robin[6][1]
                    ha='right'
                    va='center'
                elif sub_pos=='rt':
                    rob_x = robin[7][0]
                    rob_y = robin[7][1]
                    ha='right'
                    va='top'
        
                if sub_pos[0] == 'r':
                    rob_x = rob_x - 50000
                elif sub_pos[0] == 'l':
                    rob_x = rob_x + 50000
        
                if sub_pos[1] == 'b':
                    rob_y = rob_y + 50000
                elif sub_pos[1] == 't':
                    rob_y = rob_y - 50000
        
                sub_ax_3 = fig.add_axes(position,
                                        projection=ccrs.Robinson(), label=label+'markup2')
        
                # sub_ax_3.add_patch(mpatches.Rectangle((rob_x, rob_y), size_x, size_y , linewidth=1, edgecolor='grey', facecolor='white',transform=ccrs.Robinson()))
        
                sub_ax_3.set_extent(extent, ccrs.Geodetic())
                verts = mpath.Path(rect_units_to_verts([rob_x,rob_y,size_x,size_y]))
                sub_ax_3.set_boundary(verts, transform=sub_ax.transAxes)
        
                sub_ax_3.text(rob_x,rob_y,markup_sub,
                         horizontalalignment=ha, verticalalignment=va,
                         transform=ccrs.Robinson(), color='black',fontsize=4.5,bbox=dict(facecolor='white', alpha=1,linewidth=0.35,pad=1.5),fontweight='bold',zorder=25)
        
            if not main:
        #        sub_ax.outline_patch.set_edgecolor('white')
#                sub_ax.spines['geo'].set_edgecolor('white')
                sub_ax.spines['geo'].set_edgecolor('k')
            else:
        #        sub_ax.outline_patch.set_edgecolor('lightgrey')
                sub_ax.spines['geo'].set_edgecolor('lightgrey')
        
    
        #TODO: careful here! figure size determines everything else, found no way to do it otherwise in cartopy
        fig_width_inch=7.2
        fig = plt.figure(figsize=(fig_width_inch,fig_width_inch/1.9716))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
    #    ax = fig.add_axes([0,0.12,1,0.88], projection=ccrs.Robinson())
        
        ax.set_global()
        ax.spines['geo'].set_linewidth(0)
        
        add_inset(fig,[-179.99,179.99,-89.99,89.99],[0.375, 0.21, 0.25, 0.25],bounds=[-179.99,179.99,-89.99,89.99],shades=False, hillshade = False, main=True, col_bounds=col_bounds,
                  color_water='lightblue', color_land='white', add_rgi_glaciers=add_rgi_glaciers
                  )
#        #add_inset(fig,[-179.99,179.99,-89.99,89.99],[0.375, 0.21, 0.25, 0.25],bounds=[-179.99,179.99,-89.99,89.99],shades=False, hillshade = False, main=True, list_shp=shp_buff)
        
        if 19 in regions:
            poly_aw = np.array([(-158,-79),(-135,-60),(-110,-60),(-50,-60),(-50,-79.25),(-158,-79.25)])
            add_inset(fig,[-179.99,179.99,-89.99,89.99],[-0.4,-0.065,2,2],bounds=[-158, -45, -40, -79],
                      label='Antarctic_West', polygon=poly_aw,shades=True,markup_sub='West and Peninsula',sub_pos='mb', col_bounds=col_bounds)
            poly_ae = np.array([(135,-81.5),(152,-63.7),(165,-65),(175,-70),(175,-81.25),(135,-81.75)])
            add_inset(fig,[-179.99,179.99,-89.99,89.99],[-0.71,-0.045,2,2],bounds=[130, 175, -64.5, -81],
                      label='Antarctic_East', polygon=poly_ae,shades=True,markup_sub='East 2',sub_pos='mb', col_bounds=col_bounds)
            
            poly_ac = np.array([(-25,-62),(106,-62),(80,-79.25),(-25,-79.25),(-25,-62)])
            add_inset(fig,[-179.99,179.99,-89.99,89.99],[-0.52,-0.065,2,2],bounds=[-25, 106, -62.5, -79],
                      label='Antarctic_Center',polygon=poly_ac,shades=True,markup='Antarctic and Subantarctic',
                      markpos='right',markadj=0,markup_sub='East 1',sub_pos='mb', col_bounds=col_bounds)
            
            add_inset(fig,[-179.99,179.99,-89.99,89.99],[-0.68,-0.18,2,2],bounds=[64, 78, -48, -56],
                      label='Antarctic_Australes', shades=True,markup_sub='Kerguelen and Heard Islands',sub_pos='lb', col_bounds=col_bounds)
            
            add_inset(fig,[-179.99,179.99,-89.99,89.99],[-0.42,-0.143,2,2],bounds=[-40, -23, -53, -62],
                      label='Antarctic_South_Georgia', shades=True,markup_sub='South Georgia and Central Islands',sub_pos='lb', col_bounds=col_bounds)
        
        if 16 in regions or 17 in regions:
            add_inset(fig, [-179.99,179.99,-89.99,89.99], [-0.52, -0.225, 2, 2],bounds=[-82,-65,13,-57],label='Andes',
                      markup='Low Latitudes &\nSouthern Andes',markadj=0,
#                      markup_sub='a',
                      sub_pos='lm2', col_bounds=col_bounds)
            add_inset(fig, [-179.99,179.99,-89.99,89.99], [-0.352, -0.38, 2, 2],bounds=[-100,-95,22,16],label='Mexico',
                      markup_sub='Mexico',sub_pos='rb', col_bounds=col_bounds)
            add_inset(fig, [-179.99,179.99,-89.99,89.99], [-1.078, -0.22, 2, 2],bounds=[28,42,2,-6],label='Africa',
                      markup_sub='East Africa',sub_pos='rb', col_bounds=col_bounds)
            add_inset(fig, [-179.99,179.99,-89.99,89.99], [-1.64, -0.3, 2, 2],bounds=[133,140,-2,-7],label='Indonesia',
                      markup_sub='New Guinea',sub_pos='lb', col_bounds=col_bounds)
        
        if 3 in regions or 4 in regions or 5 in regions or 6 in regions or 7 in regions or 8 in regions or 9 in regions: 
            poly_arctic = np.array([(-105,84.5),(115,84.5),(110,68),(30,68),(18,57),(-70,57),(-100,75),(-105,84.5)])
            add_inset(fig,[-179.99,179.99,-89.99,89.99],[-0.48,-1.003,2,2],bounds=[-100, 106, 57, 84],label='Arctic West',
                      polygon=poly_arctic,markup='Arctic',markadj=0, col_bounds=col_bounds)
            
        if 18 in regions:
            add_inset(fig,[-179.99,179.99,-89.99,89.99],[-0.92,-0.17,2,2],bounds=[164,176,-47,-40],label='New Zealand',
                      markup='New Zealand',markpos='right',markadj=0, col_bounds=col_bounds)
        
        if 1 in regions or 2 in regions:
            poly_na = np.array([(-170,72),(-140,72),(-120,63),(-101,35),(-126,35),(-165,55),(-170,72)])
            add_inset(fig,[-179.99,179.99,-89.99,89.99],[-0.1,-1.22,2,2],bounds=[-177,-105, 36, 70],label='North America',
                      polygon=poly_na,markup='Alaska & Western\nCanada and US',markadj=0, col_bounds=col_bounds)
        
        if 10 in regions:
            
#            poly_asia_ne = np.array([(142,71),(142,82),(163,82),(155,71),(142,71)])
#            add_inset(fig,[-179.99,179.99,-89.99,89.99],[-0.64,-1.165,2,2],bounds=[142,160,71,80],
#                      polygon=poly_asia_ne,label='North Asia North E',markup_sub='Bulunsky',sub_pos='rt', col_bounds=col_bounds)
            poly_asia_e2 = np.array([(125,57),(125,70.5),(153.8,70.5),(148,57),(125,57)])
            only_shade([-0.71,-1.142,2,2],[125,148,58,72],polygon=poly_asia_e2,
                       label='tmp_NAE2')
            only_shade([-0.517,-1.035,2,2],[53,70,62,69.8],label='tmp_NAW')
            
            add_inset(fig,[-179.99,179.99,-89.99,89.99],[-0.575,-1.109,2,2],bounds=[87,112,68,78.5],
                      label='North Asia North W',markup_sub='North Siberia',sub_pos='rb', col_bounds=col_bounds)
            
            add_inset(fig,[-179.99,179.99,-89.99,89.99],[-0.71,-1.137,2,2],bounds=[125,148,54,68],polygon=poly_asia_e2,
                      label='North Asia East 2',markup_sub='Cherskiy and\nSuntar Khayata',sub_pos='lb',shades=False, col_bounds=col_bounds)
            
            poly_asia = np.array([(148,49),(160,64),(178,64),(170,55),(160,49),(148,49)])
            add_inset(fig,[-179.99,179.99,-89.99,89.99],[-0.823,-1.22,2,2],bounds=[127,179.9,50,64.8],
                      label='North Asia East',polygon=poly_asia,markup_sub='Kamchatka Krai',sub_pos='lb', col_bounds=col_bounds)
        
            
            add_inset(fig,[-179.99,179.99,-89.99,89.99],[-0.75,-1.01,2,2],bounds=[82,120,45.5,58.9],
                      label='South Asia North',markup='North Asia',markup_sub='Altay and Sayan',sub_pos='rb',markadj=0, 
                      col_bounds=col_bounds)
            add_inset(fig,[-179.99,179.99,-89.99,89.99],[-0.525,-1.045,2,2],bounds=[53,68,62,68.5],
                      label='North Asia West',markup_sub='Ural',sub_pos='rb',shades=False, col_bounds=col_bounds)
        
        if 13 in regions or 14 in regions or 15 in regions:
            add_inset(fig,[-179.99,179.99,-89.99,89.99],[-0.685,-1.065,2,2],bounds=[65, 105, 46.5, 25],
                      label='HMA',markup='High Mountain Asia',markadj=0, col_bounds=col_bounds)
        
        if 11 in regions:
            add_inset(fig,[-179.99,179.99,-89.99,89.99],[-0.58,-0.982,2,2],bounds=[-4.9,19,38.2,50.5],
                      label='Europe',markup='Central Europe',markadj=0, col_bounds=col_bounds)
        
        if 12 in regions:
            add_inset(fig,[-179.99,179.99,-89.99,89.99],[-0.66,-0.896,2,2],bounds=[38,54,29.6,43.6],
                      label='Middle East',markup='Caucasus and\nMiddle East',markadj=0, col_bounds=col_bounds)
        
        # ----- Circle sizes -----
#        axleg_background = fig.add_axes([0.001, 0.04, 0.107, 0.2])
        axleg_background = fig.add_axes([0.001, 0.04, 0.09, 0.53])
        axleg_background.get_yaxis().set_visible(False)
        axleg_background.get_xaxis().set_visible(False)
#        axleg_background.axis('off')
        rect1 = mpl.patches.Rectangle((0, 0), 1, 1, color ='white')
        axleg_background.add_patch(rect1)
        
        axleg = fig.add_axes([-0.92,-0.86,2,2],projection=ccrs.Robinson(),label='legend')
        axleg.set_extent([-179.99,179.99,-89.99,89.99], ccrs.Geodetic())
        axleg.outline_patch.set_linewidth(0)
        u=0
        rad_tot = 0
        for a in [10000, 1000, 100]:
            rad = (12000+np.sqrt(a)*1000)
            axleg.add_patch(mpatches.Circle(xy=[-900000,-680000+u*380000],radius=rad,edgecolor='k',label=str(a)+' km$^2$', transform = ccrs.Robinson(),fill=False, zorder=30))
            u=u+1
            rad_tot += rad
        axleg.text(-7.9, 2.4, '10$^{2}$\n10$^{3}$\n10$^{4}$', transform=ccrs.Geodetic(),horizontalalignment='left',verticalalignment='top',fontsize=10)
        axleg.text(-6.2, 9, 'Area', transform=ccrs.Geodetic(),horizontalalignment='center',verticalalignment='top',fontsize=10)
        axleg.text(-6.2, 6.2, '(km$^2$)', transform=ccrs.Geodetic(),horizontalalignment='center',verticalalignment='top',fontsize=9)
        

        
        # ----- Colorbar -----
        cb = []
        cb_val = np.linspace(0, 1, len(col_bounds))
        for j in range(len(cb_val)):
            cb.append(mpl.cm.RdYlBu(cb_val[j]))
##        cb[5] = cb[4]
##        cb[4] = cb[3]
##        cb[3] = cb[2]
##        cb[2] = cb[1]
##        cb[1] = cb[0]
##        cb[0] = (1,1,1,1)
#        cb[11] = (78/256, 179/256, 211/256, 1)
#        cb[10] = (123/256, 204/256, 196/256, 1)
#        cb[9] = (168/256, 221/256, 181/256, 1)
#        cb[8] = (204/256, 235/256, 197/256, 1)
#        cb[7] = (224/256, 243/256, 219/256, 1)
#        cb[6] = (254/256, 178/256, 76/256, 1)
#        cb[5] = (253/256, 141/256, 60/256, 1)
#        cb[4] = (252/256, 78/256, 42/256, 1)
#        cb[3] = (227/256, 26/256, 28/256, 1)
#        cb[2] = (189/256, 0/256, 38/256, 1)
#        cb[1] = (128/256, 0/256, 38/256, 1)
#        cb[0] = (1,1,1,1)
        
        cb[11] = (94/256, 79/256, 162/256, 1)
        cb[10] = (50/256, 136/256, 189/256, 1)
        cb[9] = (102/256, 194/256, 165/256, 1)
        cb[8] = (171/256, 221/256, 164/256, 1)
        cb[7] = (230/256, 245/256, 152/256, 1)
        cb[6] = (255/256, 255/256, 191/256, 1)
        cb[5] = (254/256, 224/256, 139/256, 1)
        cb[4] = (253/256, 174/256, 97/256, 1)
        cb[3] = (244/256, 109/256, 67/256, 1)
        cb[2] = (213/256, 62/256, 79/256, 1)
        cb[1] = (158/256, 1/256, 66/256, 1)
        cb[0] = (1,1,1,1)
        
        cmap_cus = mpl.colors.LinearSegmentedColormap.from_list('my_cb', cb)
        norm = mpl.colors.BoundaryNorm(col_bounds, cmap_cus.N)
        
        cax = fig.add_axes([0.045, 0.275, 0.007, 0.28], facecolor='none')
        sm = plt.cm.ScalarMappable(cmap=cmap_cus,norm=norm )
        cbar = plt.colorbar(sm, ticks=col_bounds, ax=ax, cax=cax, orientation='vertical')
        cax.xaxis.set_ticks_position('bottom')
        cax.xaxis.set_tick_params(pad=0)
        tick_labels = [x for x in col_bounds]
        tick_labels[10] = ''
        tick_labels[8] = ''
        tick_labels[6] = ''
        tick_labels[4] = ''
        tick_labels[2] = ''
        tick_labels[1] = '0.0'
        tick_labels[0] = ''
        cbar.set_ticklabels(tick_labels)
        cbar.ax.tick_params(labelsize=9, pad=0.3)
        
        cax.text(-0.135, 0.215, 'Mass at 2100,\nrel. to 2015 (-)', size=10, horizontalalignment='center',
                verticalalignment='bottom', rotation=90, transform=ax.transAxes)
#        # Switch if running in Spyder
#        cax.text(-0.148, 0.205, 'Mass at 2100,\nrel. to 2015 (-)', size=10, horizontalalignment='center',
#                 verticalalignment='bottom', rotation=90, transform=ax.transAxes)
    
        fig.savefig(fig_fp + 'global_deg_vol_remaining_' + rcp + '.png',dpi=250,transparent=True)
    
    
    
    #%% # ----- PEAKWATER YEAR PLOTS -----
    col_bounds = np.array([2010, 2020, 2030, 2040, 2050, 2060, 2070, 2080, 2090, 2100])
    
    for rcp in rcps:
        print(rcp)
        # Tiles are lat/lon
        normyear_idx = np.where(years == normyear)[0][0]
        
        # Area are in km2
        areas = ds_multigcm_area['all'][rcp][:,normyear_idx] / 1e6
        areas_2100 = ds_multigcm_area['all'][rcp][:,-1] / 1e6
        
        # ----- Peakwater calculation -----
        reg_mass = ds_multigcm_vol['all'][rcp] * pygem_prms.density_ice
        # Record max mb
        reg_mb_gta = (reg_mass[:,1:] - reg_mass[:,0:-1]) / 1e12        
        reg_mb_gta_smoothed = uniform_filter(reg_mb_gta, size=(11))
        
        pw_yrs = []
        for nrow in np.arange(0,reg_mb_gta.shape[0]):
            pw_yrs.append(years[np.where(reg_mb_gta_smoothed[nrow,:] == reg_mb_gta_smoothed[nrow,:].min())[0][0]])
        pw_yrs = np.array(pw_yrs)
        
        # dh is the elevation change that is used for color; we'll use volume remaining by end of century for now
#        vols_remaining_frac = ds_multigcm_vol['all'][warming_group][:,-1] / ds_multigcm_vol['all'][warming_group][:,normyear_idx]
#        dhs = vols_remaining_frac.copy()
        dhs = pw_yrs
    
        areas = [area for _, area in sorted(zip(tiles_all,areas))]
        areas_2100 = [area for _, area in sorted(zip(tiles_all,areas_2100))]
        dhs = [dh for _, dh in sorted(zip(tiles_all,dhs))]
        tiles = sorted(tiles_all)
        
        
        def add_inset(fig,extent,position,bounds=None,label=None,polygon=None,shades=True, hillshade=True, list_shp=None, main=False, markup=None,markpos='left',markadj=0,markup_sub=None,sub_pos='lt',
                      col_bounds=None, color_water=color_water, color_land=color_land, add_rgi_glaciers=False):
            main_pos = [0.375, 0.21, 0.25, 0.25]
        
            if polygon is None and bounds is not None:
                polygon = poly_from_extent(bounds)
        
            if shades:
                shades_main_to_inset(main_pos, position, latlon_extent_to_robinson_axes_verts(polygon), label=label)
        
            sub_ax = fig.add_axes(position,
                                  projection=ccrs.Robinson(),label=label)
            sub_ax.set_extent(extent, ccrs.Geodetic())
        
            sub_ax.add_feature(cfeature.NaturalEarthFeature('physical', 'ocean', '50m', facecolor=color_water))
            sub_ax.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '50m', facecolor=color_land))
            
            # Add RGI glacier outlines
            if add_rgi_glaciers:
                shape_feature = ShapelyFeature(Reader(rgi_shp_fn).geometries(), ccrs.Robinson(),alpha=1,facecolor='indigo',linewidth=0.35,edgecolor='indigo')
                sub_ax.add_feature(shape_feature)
        
            if bounds is not None:
                verts = mpath.Path(latlon_extent_to_robinson_axes_verts(polygon))
                sub_ax.set_boundary(verts, transform=sub_ax.transAxes)
        
            # HERE IS WHERE VALUES APPEAR TO BE PROVIDED
            if not main:
                print(label)
                for i in range(len(tiles)):
                    lon = tiles[i][0]
                    lat = tiles[i][1]
        
                    if label=='Arctic West' and ((lat < 71 and lon > 60) or (lat <76 and lon>100)):
                        continue
        
                    if label=='HMA' and lat >=46:
                        continue
        
                    # fac = 0.02
                    fac = 1000
                    
                    area_2100 = areas_2100[i]
        
                    if areas[i] > 10:
                        rad = 15000 + np.sqrt(areas[i]) * fac
                    else:
                        rad = 15000 + 10 * fac
                        
                    cb = []
                    cb_val = np.linspace(0, 1, len(col_bounds))
                    for j in range(len(cb_val)):
                        cb.append(mpl.cm.RdYlBu(cb_val[j]))
                    cmap_cus = mpl.colors.LinearSegmentedColormap.from_list('my_cb', list(
                        zip((col_bounds - min(col_bounds)) / (max(col_bounds - min(col_bounds))), cb)), N=len(col_bounds)-1)
#                    cb = []
#                    cb_val = np.linspace(0, 1, len(col_bounds))
#                    for j in range(len(cb_val)):
#                        cb.append(mpl.cm.RdYlBu(cb_val[j]))
#                    cb[11] = (94/256, 79/256, 162/256, 1)
#                    cb[10] = (50/256, 136/256, 189/256, 1)
#                    cb[9] = (102/256, 194/256, 165/256, 1)
#                    cb[8] = (171/256, 221/256, 164/256, 1)
#                    cb[7] = (230/256, 245/256, 152/256, 1)
#                    cb[6] = (255/256, 255/256, 191/256, 1)
#                    cb[5] = (254/256, 224/256, 139/256, 1)
#                    cb[4] = (253/256, 174/256, 97/256, 1)
#                    cb[3] = (244/256, 109/256, 67/256, 1)
#                    cb[2] = (213/256, 62/256, 79/256, 1)
#                    cb[1] = (158/256, 1/256, 66/256, 1)
#                    cb[0] = (1,1,1,1)
#                    cmap_cus = mpl.colors.LinearSegmentedColormap.from_list('my_cb', list(
#                        zip(col_bounds, cb)))
        
                    # xy = [lon,lat]
                    xy = coordXform(ccrs.PlateCarree(),ccrs.Robinson(),np.array([lon]),np.array([lat]))[0][0:2]
                    
                    # Less than percent threshold and area threshold
                    dhdt = dhs[i]
                    pw_norm = (dhdt - min(col_bounds)) / (max(col_bounds - min(col_bounds)))
                    col = cmap_cus(pw_norm)
                    if area_2100 < 0.005:
                        sub_ax.add_patch(
                            mpatches.Circle(xy=xy, radius=rad, facecolor=col, edgecolor='black', linewidth=0.5, alpha=1, transform=ccrs.Robinson(), zorder=30))
                    else:
                        sub_ax.add_patch(
                            mpatches.Circle(xy=xy, radius=rad, facecolor=col, edgecolor='None', alpha=1, transform=ccrs.Robinson(), zorder=30))
                    
                    
            if markup is not None:
                if markpos=='left':
                    lon_upleft = np.min(list(zip(*polygon))[0])
                    lat_upleft = np.max(list(zip(*polygon))[1])
                else:
                    lon_upleft = np.max(list(zip(*polygon))[0])
                    lat_upleft = np.max(list(zip(*polygon))[1])
        
                robin = coordXform(ccrs.PlateCarree(),ccrs.Robinson(),np.array([lon_upleft]),np.array([lat_upleft]))
        
                rob_x = robin[0][0]
                rob_y = robin[0][1]
        
                size_y = 200000
                size_x = 80000 * len(markup) + markadj
        
                if markpos=='right':
                    rob_x = rob_x-50000
                else:
                    rob_x = rob_x+50000
        
                sub_ax_2 = fig.add_axes(position,
                                        projection=ccrs.Robinson(), label=label+'markup')
        
                # adds the white box to the region
#                sub_ax_2.add_patch(mpatches.Rectangle((rob_x, rob_y), size_x, size_y , linewidth=1, edgecolor='grey', facecolor='white',transform=ccrs.Robinson()))
        
                sub_ax_2.set_extent(extent, ccrs.Geodetic())
                verts = mpath.Path(rect_units_to_verts([rob_x,rob_y,size_x,size_y]))
                sub_ax_2.set_boundary(verts, transform=sub_ax.transAxes)
        
                sub_ax_2.text(rob_x,rob_y+50000,markup,
                         horizontalalignment=markpos, verticalalignment='bottom',
                         transform=ccrs.Robinson(), color='black',fontsize=4.5, fontweight='bold',bbox= dict(facecolor='white', alpha=1,linewidth=0.35,pad=1.5))
        
            if markup_sub is not None:
        
                lon_min = np.min(list(zip(*polygon))[0])
                lon_max = np.max(list(zip(*polygon))[0])
                lon_mid = 0.5*(lon_min+lon_max)
        
                lat_min = np.min(list(zip(*polygon))[1])
                lat_max = np.max(list(zip(*polygon))[1])
                lat_mid = 0.5*(lat_min+lat_max)
        
                size_y = 150000
                size_x = 150000
        
                lat_midup = lat_min+0.87*(lat_max-lat_min)
        
                robin = coordXform(ccrs.PlateCarree(),ccrs.Robinson(),np.array([lon_min,lon_min,lon_min,lon_mid,lon_mid,lon_max,lon_max,lon_max,lon_min]),np.array([lat_min,lat_mid,lat_max,lat_min,lat_max,lat_min,lat_mid,lat_max,lat_midup]))
        
                if sub_pos=='lb':
                    rob_x = robin[0][0]
                    rob_y = robin[0][1]
                    ha='left'
                    va='bottom'
                elif sub_pos=='lm':
                    rob_x = robin[1][0]
                    rob_y = robin[1][1]
                    ha='left'
                    va='center'
                elif sub_pos == 'lm2':
                    rob_x = robin[8][0]
                    rob_y = robin[8][1]
                    ha = 'left'
                    va = 'center'
                elif sub_pos=='lt':
                    rob_x = robin[2][0]
                    rob_y = robin[2][1]
                    ha='left'
                    va='top'
                elif sub_pos=='mb':
                    rob_x = robin[3][0]
                    rob_y = robin[3][1]
                    ha='center'
                    va='bottom'
                elif sub_pos=='mt':
                    rob_x = robin[4][0]
                    rob_y = robin[4][1]
                    ha='center'
                    va='top'
                elif sub_pos=='rb':
                    rob_x = robin[5][0]
                    rob_y = robin[5][1]
                    ha='right'
                    va='bottom'
                elif sub_pos=='rm':
                    rob_x = robin[6][0]
                    rob_y = robin[6][1]
                    ha='right'
                    va='center'
                elif sub_pos=='rt':
                    rob_x = robin[7][0]
                    rob_y = robin[7][1]
                    ha='right'
                    va='top'
        
                if sub_pos[0] == 'r':
                    rob_x = rob_x - 50000
                elif sub_pos[0] == 'l':
                    rob_x = rob_x + 50000
        
                if sub_pos[1] == 'b':
                    rob_y = rob_y + 50000
                elif sub_pos[1] == 't':
                    rob_y = rob_y - 50000
        
                sub_ax_3 = fig.add_axes(position,
                                        projection=ccrs.Robinson(), label=label+'markup2')
        
                # sub_ax_3.add_patch(mpatches.Rectangle((rob_x, rob_y), size_x, size_y , linewidth=1, edgecolor='grey', facecolor='white',transform=ccrs.Robinson()))
        
                sub_ax_3.set_extent(extent, ccrs.Geodetic())
                verts = mpath.Path(rect_units_to_verts([rob_x,rob_y,size_x,size_y]))
                sub_ax_3.set_boundary(verts, transform=sub_ax.transAxes)
        
                sub_ax_3.text(rob_x,rob_y,markup_sub,
                         horizontalalignment=ha, verticalalignment=va,
                         transform=ccrs.Robinson(), color='black',fontsize=4.5,bbox=dict(facecolor='white', alpha=1,linewidth=0.35,pad=1.5),fontweight='bold',zorder=25)
        
            if not main:
        #        sub_ax.outline_patch.set_edgecolor('white')
#                sub_ax.spines['geo'].set_edgecolor('white')
                sub_ax.spines['geo'].set_edgecolor('k')
            else:
        #        sub_ax.outline_patch.set_edgecolor('lightgrey')
                sub_ax.spines['geo'].set_edgecolor('lightgrey')
        
    
        #TODO: careful here! figure size determines everything else, found no way to do it otherwise in cartopy
        fig_width_inch=7.2
        fig = plt.figure(figsize=(fig_width_inch,fig_width_inch/1.9716))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
    #    ax = fig.add_axes([0,0.12,1,0.88], projection=ccrs.Robinson())
        
        ax.set_global()
        ax.spines['geo'].set_linewidth(0)
        
        add_inset(fig,[-179.99,179.99,-89.99,89.99],[0.375, 0.21, 0.25, 0.25],bounds=[-179.99,179.99,-89.99,89.99],shades=False, hillshade = False, main=True, col_bounds=col_bounds,
                  color_water='lightblue', color_land='white', add_rgi_glaciers=add_rgi_glaciers
                  )
#        #add_inset(fig,[-179.99,179.99,-89.99,89.99],[0.375, 0.21, 0.25, 0.25],bounds=[-179.99,179.99,-89.99,89.99],shades=False, hillshade = False, main=True, list_shp=shp_buff)
        
        if 19 in regions:
            poly_aw = np.array([(-158,-79),(-135,-62),(-110,-62),(-50,-62),(-50,-79.25),(-158,-79.25)])
            add_inset(fig,[-179.99,179.99,-89.99,89.99],[-0.4,-0.065,2,2],bounds=[-158, -50, -62.5, -79],
                      label='Antarctic_West', polygon=poly_aw,shades=True,markup_sub='West and Peninsula',sub_pos='mb', col_bounds=col_bounds)
            
            poly_ae = np.array([(135,-81.5),(152,-63.7),(165,-65),(175,-70),(175,-81.25),(135,-81.75)])
            add_inset(fig,[-179.99,179.99,-89.99,89.99],[-0.71,-0.045,2,2],bounds=[130, 175, -64.5, -81],
                      label='Antarctic_East', polygon=poly_ae,shades=True,markup_sub='East 2',sub_pos='mb', col_bounds=col_bounds)
            
            poly_ac = np.array([(-25,-62),(106,-62),(80,-79.25),(-25,-79.25),(-25,-62)])
            add_inset(fig,[-179.99,179.99,-89.99,89.99],[-0.52,-0.065,2,2],bounds=[-25, 106, -62.5, -79],
                      label='Antarctic_Center',polygon=poly_ac,shades=True,markup='Antarctic and Subantarctic',
                      markpos='right',markadj=0,markup_sub='East 1',sub_pos='mb', col_bounds=col_bounds)
            
            add_inset(fig,[-179.99,179.99,-89.99,89.99],[-0.68,-0.18,2,2],bounds=[64, 78, -48, -56],
                      label='Antarctic_Australes', shades=True,markup_sub='Kerguelen and Heard Islands',sub_pos='lb', col_bounds=col_bounds)
            
            add_inset(fig,[-179.99,179.99,-89.99,89.99],[-0.42,-0.165,2,2],bounds=[-40, -23, -51, -60],
                      label='Antarctic_South_Georgia', shades=True,markup_sub='South Georgia and Central Islands',sub_pos='lt', col_bounds=col_bounds)
        
        if 16 in regions or 17 in regions:
            add_inset(fig, [-179.99,179.99,-89.99,89.99], [-0.52, -0.225, 2, 2],bounds=[-82,-65,13,-57],label='Andes',
                      markup='Low Latitudes &\nSouthern Andes',markadj=0,
#                      markup_sub='a',
                      sub_pos='lm2', col_bounds=col_bounds)
            add_inset(fig, [-179.99,179.99,-89.99,89.99], [-0.352, -0.38, 2, 2],bounds=[-100,-95,22,16],label='Mexico',
                      markup_sub='Mexico',sub_pos='rb', col_bounds=col_bounds)
            add_inset(fig, [-179.99,179.99,-89.99,89.99], [-1.078, -0.22, 2, 2],bounds=[28,42,2,-6],label='Africa',
                      markup_sub='East Africa',sub_pos='rb', col_bounds=col_bounds)
            add_inset(fig, [-179.99,179.99,-89.99,89.99], [-1.64, -0.3, 2, 2],bounds=[133,140,-2,-7],label='Indonesia',
                      markup_sub='New Guinea',sub_pos='lb', col_bounds=col_bounds)
        
        if 3 in regions or 4 in regions or 5 in regions or 6 in regions or 7 in regions or 8 in regions or 9 in regions: 
            poly_arctic = np.array([(-105,84.5),(115,84.5),(110,68),(30,68),(18,57),(-70,57),(-100,75),(-105,84.5)])
            add_inset(fig,[-179.99,179.99,-89.99,89.99],[-0.48,-1.003,2,2],bounds=[-100, 106, 57, 84],label='Arctic West',
                      polygon=poly_arctic,markup='Arctic',markadj=0, col_bounds=col_bounds)
            
        if 18 in regions:
            add_inset(fig,[-179.99,179.99,-89.99,89.99],[-0.92,-0.17,2,2],bounds=[164,176,-47,-40],label='New Zealand',
                      markup='New Zealand',markpos='right',markadj=0, col_bounds=col_bounds)
        
        if 1 in regions or 2 in regions:
            poly_na = np.array([(-170,72),(-140,72),(-120,63),(-101,35),(-126,35),(-165,55),(-170,72)])
            add_inset(fig,[-179.99,179.99,-89.99,89.99],[-0.1,-1.22,2,2],bounds=[-177,-105, 36, 70],label='North America',
                      polygon=poly_na,markup='Alaska & Western\nCanada and US',markadj=0, col_bounds=col_bounds)
        
        if 10 in regions:
            
#            poly_asia_ne = np.array([(142,71),(142,82),(163,82),(155,71),(142,71)])
#            add_inset(fig,[-179.99,179.99,-89.99,89.99],[-0.64,-1.165,2,2],bounds=[142,160,71,80],
#                      polygon=poly_asia_ne,label='North Asia North E',markup_sub='Bulunsky',sub_pos='rt', col_bounds=col_bounds)
            poly_asia_e2 = np.array([(125,57),(125,70.5),(153.8,70.5),(148,57),(125,57)])
            only_shade([-0.71,-1.142,2,2],[125,148,58,72],polygon=poly_asia_e2,
                       label='tmp_NAE2')
            only_shade([-0.517,-1.035,2,2],[53,70,62,69.8],label='tmp_NAW')
            
            add_inset(fig,[-179.99,179.99,-89.99,89.99],[-0.575,-1.109,2,2],bounds=[87,112,68,78.5],
                      label='North Asia North W',markup_sub='North Siberia',sub_pos='lt', col_bounds=col_bounds)
            
            add_inset(fig,[-179.99,179.99,-89.99,89.99],[-0.71,-1.137,2,2],bounds=[125,148,54,68],polygon=poly_asia_e2,
                      label='North Asia East 2',markup_sub='Cherskiy and\nSuntar Khayata',sub_pos='lb',shades=False, col_bounds=col_bounds)
            
            poly_asia = np.array([(148,49),(160,64),(178,64),(170,55),(160,49),(148,49)])
            add_inset(fig,[-179.99,179.99,-89.99,89.99],[-0.823,-1.22,2,2],bounds=[127,179.9,50,64.8],
                      label='North Asia East',polygon=poly_asia,markup_sub='Kamchatka Krai',sub_pos='lb', col_bounds=col_bounds)
        
            
            add_inset(fig,[-179.99,179.99,-89.99,89.99],[-0.75,-1.01,2,2],bounds=[82,120,45.5,58.9],
                      label='South Asia North',markup='North Asia',markup_sub='Altay and Sayan',sub_pos='rb',markadj=0, 
                      col_bounds=col_bounds)
            add_inset(fig,[-179.99,179.99,-89.99,89.99],[-0.525,-1.045,2,2],bounds=[53,68,62,68.5],
                      label='North Asia West',markup_sub='Ural',sub_pos='lt',shades=False, col_bounds=col_bounds)
        
        if 13 in regions or 14 in regions or 15 in regions:
            add_inset(fig,[-179.99,179.99,-89.99,89.99],[-0.685,-1.065,2,2],bounds=[65, 105, 46.5, 25],
                      label='HMA',markup='High Mountain Asia',markadj=0, col_bounds=col_bounds)
        
        if 11 in regions:
            add_inset(fig,[-179.99,179.99,-89.99,89.99],[-0.58,-0.982,2,2],bounds=[-4.9,19,38.2,50.5],
                      label='Europe',markup='Central Europe',markadj=0, col_bounds=col_bounds)
        
        if 12 in regions:
            add_inset(fig,[-179.99,179.99,-89.99,89.99],[-0.66,-0.89,2,2],bounds=[38,54,29,44.75],
                      label='Middle East',markup='Caucasus/Middle East',markadj=0, col_bounds=col_bounds)
        
        # ----- Circle sizes -----
#        axleg_background = fig.add_axes([0.001, 0.04, 0.107, 0.2])
        axleg_background = fig.add_axes([0.001, 0.04, 0.09, 0.53])
        axleg_background.get_yaxis().set_visible(False)
        axleg_background.get_xaxis().set_visible(False)
#        axleg_background.axis('off')
        rect1 = mpl.patches.Rectangle((0, 0), 1, 1, color ='white')
        axleg_background.add_patch(rect1)
        
        axleg = fig.add_axes([-0.92,-0.86,2,2],projection=ccrs.Robinson(),label='legend')
        axleg.set_extent([-179.99,179.99,-89.99,89.99], ccrs.Geodetic())
        axleg.outline_patch.set_linewidth(0)
        u=0
        rad_tot = 0
        for a in [10000, 1000, 100]:
            rad = (12000+np.sqrt(a)*1000)
            axleg.add_patch(mpatches.Circle(xy=[-900000,-680000+u*380000],radius=rad,edgecolor='k',label=str(a)+' km$^2$', transform = ccrs.Robinson(),fill=False, zorder=30))
            u=u+1
            rad_tot += rad
        axleg.text(-7.9, 2.4, '10$^{2}$\n10$^{3}$\n10$^{4}$', transform=ccrs.Geodetic(),horizontalalignment='left',verticalalignment='top',fontsize=10)
        axleg.text(-6.2, 9, 'Area', transform=ccrs.Geodetic(),horizontalalignment='center',verticalalignment='top',fontsize=10)
        axleg.text(-6.2, 6.2, '(km$^2$)', transform=ccrs.Geodetic(),horizontalalignment='center',verticalalignment='top',fontsize=9)
        
        # ----- Colorbar -----
        cax = fig.add_axes([0.028, 0.265, 0.007, 0.28], facecolor='none')
        cb = []
        cb_val = np.linspace(0, 1, len(col_bounds))
        for j in range(len(cb_val)):
            cb.append(mpl.cm.RdYlBu(cb_val[j]))
        cmap_cus = mpl.colors.LinearSegmentedColormap.from_list('my_cb', list(
            zip((col_bounds - min(col_bounds)) / (max(col_bounds - min(col_bounds))), cb)), N=len(col_bounds)-1)
        # norm = mpl.colors.BoundaryNorm(boundaries=col_bounds, ncolors=256)
        norm = mpl.colors.Normalize(vmin=np.min(col_bounds),vmax=np.max(col_bounds))
        sm = plt.cm.ScalarMappable(cmap=cmap_cus, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, cax=cax, ticks=col_bounds, orientation='vertical',extend='both')

        cax.xaxis.set_ticks_position('bottom')
        cax.xaxis.set_tick_params(pad=0)
        tick_labels = [x for x in col_bounds]
        tick_labels[8] = ''
        tick_labels[6] = ''
        tick_labels[4] = ''
        tick_labels[2] = ''
        tick_labels[0] = ''
        cbar.set_ticklabels(tick_labels)
        cbar.ax.tick_params(labelsize=8)
        
        cax.text(-0.147, 0.19, 'Peak Water Year', size=10, horizontalalignment='center',
                verticalalignment='bottom', rotation=90, transform=ax.transAxes)
        # Switch if running in Spyder
#        cax.text(-0.16, 0.175, 'Peak Water Year', size=10, horizontalalignment='center',
#                verticalalignment='bottom', rotation=90, transform=ax.transAxes)
    
        fig.savefig(fig_fp + 'global_deg_peakwater_' + rcp + '.png',dpi=250,transparent=True)
    

#%%
if option_global_vol_remaining_bydeg:
    regions = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
#    regions = [11]

    warming_groups = [1.5,2,2.7,3,4]
    warming_groups_bnds = [0.25, 0.5, 0.5, 0.5, 0.5]
#    warming_groups = [2]
#    warming_groups_bnds = [0.5]
    
    add_rgi_glaciers = True
#    add_rgi_glaciers = False
    
    normyear = 2015
    
    # GCMs and RCP scenarios
    gcm_names_rcps = ['CanESM2', 'CCSM4', 'CNRM-CM5', 'CSIRO-Mk3-6-0', 'GFDL-CM3', 
                      'GFDL-ESM2M', 'GISS-E2-R', 'IPSL-CM5A-LR', 'MPI-ESM-LR', 'NorESM1-M']
    gcm_names_ssps = ['BCC-CSM2-MR', 'CESM2', 'CESM2-WACCM', 'EC-Earth3', 'EC-Earth3-Veg', 'FGOALS-f3-L', 
                      'GFDL-ESM4', 'INM-CM4-8', 'INM-CM5-0', 'MPI-ESM1-2-HR', 'MRI-ESM2-0', 'NorESM2-MM']
    gcm_names_ssp119 = ['EC-Earth3', 'EC-Earth3-Veg', 'GFDL-ESM4', 'MRI-ESM2-0']
    #rcps = ['rcp26', 'rcp45', 'rcp85']
#    rcps = ['ssp126', 'ssp245', 'ssp370', 'ssp585']
    rcps = ['ssp119', 'ssp126', 'ssp245', 'ssp370', 'ssp585']
#    rcps = ['ssp126']
    
    years = np.arange(2000,2102)
    
    # Colors and bounds
    col_bounds = np.array([0, 0.005, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    color_shades_to_inset = 'black'
#    color_water = 'lightcyan'
#    color_land = 'gainsboro'
#    color_water = 'gainsboro'
#    color_land = 'darkgrey'
    color_water='lightblue'
    color_land='white'
    
    
    netcdf_fp_cmip5_land = '/Users/drounce/Documents/HiMAT/spc_backup/simulations/'
    netcdf_fp_cmip5 = '/Users/drounce/Documents/HiMAT/spc_backup/simulations_calving_v5/'
    analysis_fp = netcdf_fp_cmip5.replace('simulations','analysis')
    fig_fp = analysis_fp + 'figures/multi_gcm/'
    if not os.path.exists(fig_fp):
        os.makedirs(fig_fp, exist_ok=True)
    csv_fp = analysis_fp + 'csv/'
    if not os.path.exists(csv_fp):
        os.makedirs(csv_fp, exist_ok=True)
    pickle_fp = analysis_fp + 'pickle/'
    pickle_fp_land = netcdf_fp_cmip5_land.replace('simulations','analysis') + 'pickle/'
     
    reg_vol_all_deg = {}
    reg_area_all_deg = {}
    ds_multigcm_area = {}
    ds_multigcm_vol = {}
    latlon_df = {}
    tiles_all = []
    
    # ----- LOAD CLIMATE DATA -----
    # Filenames
    fn_reg_temp_all = 'reg_temp_all.pkl'
    fn_reg_temp_all_monthly = 'reg_temp_all_monthly.pkl'
    fn_reg_prec_all = 'reg_prec_all.pkl'
    fn_reg_prec_all_monthly = 'reg_prec_all_monthly.pkl'
    fn_reg_datestable = 'policytemp_climate_dates_table_monthly.pkl'
    temp_dev_fn = 'Global_mean_temp_deviation_2081_2100_rel_1850_1900.csv'
            
    assert os.path.exists(pickle_fp + fn_reg_temp_all), 'Global temp data does not exist, run analysis_policy_temp_figs.py script to process'
    with open(pickle_fp + fn_reg_temp_all, 'rb') as f:
        reg_temp_all = pickle.load(f)
    with open(pickle_fp + fn_reg_temp_all_monthly, 'rb') as f:
        reg_temp_all_monthly = pickle.load(f)
    with open(pickle_fp + fn_reg_prec_all, 'rb') as f:
        reg_prec_all = pickle.load(f)
    with open(pickle_fp + fn_reg_prec_all_monthly, 'rb') as f:
        reg_prec_all_monthly = pickle.load(f)
    with open(pickle_fp + fn_reg_datestable, 'rb') as f:
        dates_table = pickle.load(f)
    temp_dev_df = pd.read_csv(csv_fp + temp_dev_fn)
        
    years_climate = np.unique(dates_table.year)
    

    # Set up all
    ds_multigcm_vol['all'] = {}
    ds_multigcm_area['all'] = {}
    for warming_group in warming_groups:
        ds_multigcm_vol['all'][warming_group] = None
        ds_multigcm_area['all'][warming_group] = None
    
    reg_vol_gcm_all_dict = {}
    reg_area_gcm_all_dict = {}
    # Process regions
    for reg in regions:
        
        for warming_group in warming_groups:
            reg_vol_gcm_all_dict[warming_group] = None
            reg_area_gcm_all_dict[warming_group] = None
        
        #%%
        # Glaciers
        fn_reg_glacno_list = 'R' + str(reg) + '_glacno_list.pkl'
        with open(pickle_fp_land + str(reg).zfill(2) + '/' + fn_reg_glacno_list, 'rb') as f:
            glacno_list = pickle.load(f)
        
        # Degree ID dict
        fn_unique_degids = 'R' + str(reg) + '_unique_degids.pkl'
        with open(pickle_fp_land + str(reg).zfill(2) + '/' + fn_unique_degids, 'rb') as f:
            unique_degids = pickle.load(f)

        if reg in [17]:
            # Degree ID dict
            fn_unique_degids_ssp119 = 'R' + str(reg) + '_unique_degids_ssp119.pkl'
            with open(pickle_fp_land + str(reg).zfill(2) + '/' + fn_unique_degids_ssp119, 'rb') as f:
                unique_degids_ssp119 = pickle.load(f)
            # Degree ID dict
            fn_reg_glacno_list_ssp119 = 'R' + str(reg) + '_glacno_list_ssp119.pkl'
            with open(pickle_fp_land + str(reg).zfill(2) + '/' + fn_reg_glacno_list_ssp119, 'rb') as f:
                glacno_list_ssp119 = pickle.load(f)
            main_glac_rgi_ssp119 = modelsetup.selectglaciersrgitable(glac_no=glacno_list_ssp119)
            # ----- Add Groups -----
            # Degrees (based on degree_size)
            degree_size_pkl = 0.1
            main_glac_rgi_ssp119['CenLon_round'] = np.floor(main_glac_rgi_ssp119.CenLon.values/degree_size_pkl) * degree_size_pkl
            main_glac_rgi_ssp119['CenLat_round'] = np.floor(main_glac_rgi_ssp119.CenLat.values/degree_size_pkl) * degree_size_pkl
            deg_groups_ssp119 = main_glac_rgi_ssp119.groupby(['CenLon_round', 'CenLat_round']).size().index.values.tolist()
            deg_dict_ssp119 = dict(zip(deg_groups_ssp119, np.arange(0,len(deg_groups_ssp119))))
            main_glac_rgi_ssp119.reset_index(drop=True, inplace=True)
            cenlon_cenlat_ssp119 = [(main_glac_rgi_ssp119.loc[x,'CenLon_round'], main_glac_rgi_ssp119.loc[x,'CenLat_round']) 
                                     for x in range(len(main_glac_rgi_ssp119))]
            main_glac_rgi_ssp119['CenLon_CenLat'] = cenlon_cenlat_ssp119
            main_glac_rgi_ssp119['deg_id'] = main_glac_rgi_ssp119.CenLon_CenLat.map(deg_dict_ssp119)
            
            # Remove row from SSP119 that is not in other scenarios
            nrow2skip_ssp119 = deg_groups_ssp119.index((-73.8, -48.5))
#            deg_groups_ssp119.remove((-73.8,-48.5))
            
            #%%
            
        # All glaciers for fraction
        # Glaciers with successful runs to process
        main_glac_rgi = modelsetup.selectglaciersrgitable(glac_no=glacno_list)
            
        # ----- Add Groups -----
        # Degrees (based on degree_size)
        degree_size_pkl = 0.1
        main_glac_rgi['CenLon_round'] = np.floor(main_glac_rgi.CenLon.values/degree_size_pkl) * degree_size_pkl
        main_glac_rgi['CenLat_round'] = np.floor(main_glac_rgi.CenLat.values/degree_size_pkl) * degree_size_pkl
        deg_groups = main_glac_rgi.groupby(['CenLon_round', 'CenLat_round']).size().index.values.tolist()
        deg_dict = dict(zip(deg_groups, np.arange(0,len(deg_groups))))
        main_glac_rgi.reset_index(drop=True, inplace=True)
        cenlon_cenlat = [(main_glac_rgi.loc[x,'CenLon_round'], main_glac_rgi.loc[x,'CenLat_round']) 
                         for x in range(len(main_glac_rgi))]
        main_glac_rgi['CenLon_CenLat'] = cenlon_cenlat
        main_glac_rgi['deg_id'] = main_glac_rgi.CenLon_CenLat.map(deg_dict)
    
        reg_vol_all_deg[reg] = {}
        reg_area_all_deg[reg] = {}
        ds_multigcm_vol[reg] = {}
        ds_multigcm_area[reg] = {}
        latlon_df[reg] = None
        for nrcp, rcp in enumerate(rcps):
            reg_vol_all_deg[reg][rcp] = {}
            reg_area_all_deg[reg][rcp] = {}
          
            if 'rcp' in rcp:
                gcm_names = gcm_names_rcps
            elif 'ssp' in rcp:
                if rcp in ['ssp119']:
                    gcm_names = gcm_names_ssp119
                else:
                    gcm_names = gcm_names_ssps
                
            for ngcm, gcm_name in enumerate(gcm_names):
        
                # ----- GCM/RCP PICKLE FILEPATHS AND FILENAMES -----
#                netcdf_fp_cmip5_reg = netcdf_fp_cmip5 + str(reg).zfill(2)
#                if '_calving' in netcdf_fp_cmip5 and not os.path.exists(netcdf_fp_cmip5_reg):
#                    pickle_fp_degid =  (netcdf_fp_cmip5_land + '../analysis/pickle/' + str(reg).zfill(2) + 
#                                        '/degids/' + gcm_name + '/' + rcp + '/')
#                else:
#                    pickle_fp_degid =  pickle_fp + str(reg).zfill(2) + '/degids/' + gcm_name + '/' + rcp + '/'

                if '_calving' in netcdf_fp_cmip5 and reg in [2,6,8,10,11,12,13,14,15,16,18]:
#                if '_calving' in netcdf_fp_cmip5 and reg in [2,6,8,10,11,12,13,14,15,16,18]:
                    pickle_fp_degid =  pickle_fp_land + str(reg).zfill(2) + '/degids/' + gcm_name + '/' + rcp + '/'
                else:
                    pickle_fp_degid =  pickle_fp + str(reg).zfill(2) + '/degids/' + gcm_name + '/' + rcp + '/'
                    
                # Region string prefix
                degid_rcp_gcm_str = 'R' + str(reg) + '_degids_' + rcp + '_' + gcm_name            
                # Pickle Filenames
                fn_degid_vol_annual = degid_rcp_gcm_str + '_vol_annual.pkl'
                fn_degid_area_annual = degid_rcp_gcm_str + '_area_annual.pkl'
                    
                # Volume
                with open(pickle_fp_degid + fn_degid_vol_annual, 'rb') as f:
                    degid_vol_annual = pickle.load(f)
                # Area
                with open(pickle_fp_degid + fn_degid_area_annual, 'rb') as f:
                    degid_area_annual = pickle.load(f)
                
                # Aggregate to desired scale
                lat_min = main_glac_rgi['CenLat_round'].min()
                lat_max = main_glac_rgi['CenLat_round'].max()
                lon_min = main_glac_rgi['CenLon_round'].min()
                lon_max = main_glac_rgi['CenLon_round'].max()
                if ngcm == 0:
                    print('lat/lon min/max:', lat_min, lat_max, lon_min, lon_max)
                
                degree_size_lon = 1
                degree_size_lat= 1
                lat_start = np.round(lat_min/degree_size_lat)*degree_size_lat
                lat_end = np.round(lat_max/degree_size_lat)*degree_size_lat
                lon_start = np.round(lon_min/degree_size_lon)*degree_size_lon
                lon_end = np.round(lon_max/degree_size_lon)*degree_size_lon
                
                print(rcp, gcm_name, len(unique_degids))
                if reg in [17] and rcp in ['ssp119']:
                    unique_degids = np.delete(unique_degids_ssp119,(nrow2skip_ssp119), axis=0)
                    main_glac_rgi = main_glac_rgi_ssp119
                else:
                    unique_degids = unique_degids
                    main_glac_rgi = main_glac_rgi

                degid_latlon_df = pd.DataFrame(np.zeros((len(unique_degids),2)), columns=['CenLon_round','CenLat_round'])
                for nrow, degid in enumerate(unique_degids):
                    main_glac_rgi_subset = main_glac_rgi.loc[main_glac_rgi['deg_id'] == degid]
                    main_glac_rgi_subset.reset_index(inplace=True, drop=True)
                    degid_latlon_df.loc[nrow,'CenLon_round'] = main_glac_rgi_subset.loc[0,'CenLon_round']
                    degid_latlon_df.loc[nrow,'CenLat_round'] = main_glac_rgi_subset.loc[0,'CenLat_round']
                    
                agg_degid_vol_annual = None
                agg_degid_area_annual = None
                agg_lonlat_list = []
                count = 0
                for lon in np.arange(lon_start, lon_end+degree_size_lon/2, degree_size_lon):
                    for lat in np.arange(lat_start, lat_end+degree_size_lat/2, degree_size_lat):
                        degid_latlon_df_subset = degid_latlon_df.loc[(degid_latlon_df['CenLon_round'] >= lon - degree_size_lon/2) & 
                                                                     (degid_latlon_df['CenLon_round'] < lon + degree_size_lon/2) &
                                                                     (degid_latlon_df['CenLat_round'] >= lat - degree_size_lat/2) &
                                                                     (degid_latlon_df['CenLat_round'] < lat + degree_size_lat/2)]
                        array_idx = degid_latlon_df_subset.index.values
                        
                        if len(array_idx) > 0:
                            agg_degid_vol_annual_single = degid_vol_annual[array_idx,:].sum(0)
                            agg_degid_area_annual_single = degid_area_annual[array_idx,:].sum(0)
                            if not [lon, lat] in agg_lonlat_list:
                                agg_lonlat_list.append([lon, lat])
                            
                            if agg_degid_vol_annual is None:
                                agg_degid_vol_annual = agg_degid_vol_annual_single
                                agg_degid_area_annual = agg_degid_area_annual_single
                            else:
                                agg_degid_vol_annual = np.vstack([agg_degid_vol_annual, agg_degid_vol_annual_single])
                                agg_degid_area_annual = np.vstack([agg_degid_area_annual, agg_degid_area_annual_single])
                            
                            count += 1
    #                        print(count, lon, lat, np.round(agg_degid_area_annual_single[0]/1e6,1))
                
                print('list length:', len(agg_lonlat_list))
                    
                # Record datasets
                reg_vol_all_deg[reg][rcp][gcm_name] = agg_degid_vol_annual
                reg_area_all_deg[reg][rcp][gcm_name] = agg_degid_area_annual
                
                # Find temperature deviation
                temp_dev = temp_dev_df.loc[(temp_dev_df['Scenario'] == rcp) & (temp_dev_df['GCM'] == gcm_name), 'global_mean_deviation_degC'].values[0]
                print(' temp dev:', np.round(temp_dev,2))
                
                for nwarming_group, warming_group in enumerate(warming_groups):
                    warming_group_bnd = warming_groups_bnds[nwarming_group]
                    if (temp_dev > warming_group - warming_group_bnd) and (temp_dev <= warming_group + warming_group_bnd):
                        if reg_vol_gcm_all_dict[warming_group] is None:
                            reg_vol_gcm_all_dict[warming_group] = agg_degid_vol_annual[np.newaxis,:,:]
                            reg_area_gcm_all_dict[warming_group] = agg_degid_area_annual[np.newaxis,:,:]
                        else:
                            reg_vol_gcm_all_dict[warming_group] = np.vstack((reg_vol_gcm_all_dict[warming_group], agg_degid_vol_annual[np.newaxis,:,:]))
                            reg_area_gcm_all_dict[warming_group] = np.vstack((reg_area_gcm_all_dict[warming_group], agg_degid_area_annual[np.newaxis,:,:]))
                

        # Save all the data; not just the regional datasets
        for nwarming_group, warming_group in enumerate(warming_groups):
            if not reg_vol_gcm_all_dict[warming_group] is None:
                ds_multigcm_vol[reg][warming_group] = reg_vol_gcm_all_dict[warming_group]
                ds_multigcm_area[reg][warming_group] = reg_area_gcm_all_dict[warming_group]
                if ds_multigcm_vol['all'][warming_group] is None:
                    ds_multigcm_vol['all'][warming_group] = np.median(reg_vol_gcm_all_dict[warming_group], axis=0)
                    ds_multigcm_area['all'][warming_group] = np.median(reg_area_gcm_all_dict[warming_group], axis=0)
                else:
                    ds_multigcm_vol['all'][warming_group] = np.concatenate((ds_multigcm_vol['all'][warming_group], 
                                                                            np.median(reg_vol_gcm_all_dict[warming_group], axis=0)), axis=0)
                    ds_multigcm_area['all'][warming_group] = np.concatenate((ds_multigcm_area['all'][warming_group], 
                                                                             np.median(reg_area_gcm_all_dict[warming_group], axis=0)), axis=0)

        # Tiles are a list of the lat/lon 
        tiles = agg_lonlat_list
        for tile in tiles:
            tiles_all.append(tile)
        
#        assert True==False, 'Something is off with the tiles'
            
            
    #%%
    col_bounds = np.array([0, 0.005, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    for nwarming_group, warming_group in enumerate(warming_groups):
        print(warming_group)

        normyear_idx = np.where(years == normyear)[0][0]
        
        # Area are in km2
        areas = ds_multigcm_area['all'][warming_group][:,normyear_idx] / 1e6
        areas_2100 = ds_multigcm_area['all'][warming_group][:,-1] / 1e6
        
        # dh is the elevation change that is used for color; we'll use volume remaining by end of century for now
        vols_remaining_frac = ds_multigcm_vol['all'][warming_group][:,-1] / ds_multigcm_vol['all'][warming_group][:,normyear_idx]
        dhs = vols_remaining_frac.copy()
    
        areas = [area for _, area in sorted(zip(tiles_all,areas))]
        areas_2100 = [area for _, area in sorted(zip(tiles_all,areas_2100))]
        dhs = [dh for _, dh in sorted(zip(tiles_all,dhs))]
        tiles = sorted(tiles_all)
        
        def latlon_extent_to_axes_units(extent):
        
            extent = np.array(extent)
        
            lons = (extent[0:2] + 179.9) / 359.8
            lats = (extent[2:4] + 89.9) / 179.8
        
            return [lons[0],lons[1],lats[0],lats[1]]
        
        def axes_pos_to_rect_units(units):
        
            return [min(units[0:2]),min(units[2:4]),max(units[0:2])-min(units[0:2]),max(units[2:4])-min(units[2:4])]
        
        def rect_units_to_verts(rect_u):
        
            return np.array([[rect_u[0],rect_u[1]],[rect_u[0]+rect_u[2],rect_u[1]],[rect_u[0]+rect_u[2],rect_u[1] +rect_u[3]],[rect_u[0],rect_u[1]+rect_u[3]],[rect_u[0],rect_u[1]]])
        
        def coordXform(orig_crs, target_crs, x, y):
            return target_crs.transform_points( orig_crs, x, y )
        
        def poly_from_extent(ext):
        
            poly = np.array([(ext[0],ext[2]),(ext[1],ext[2]),(ext[1],ext[3]),(ext[0],ext[3]),(ext[0],ext[2])])
        
            return poly
        
        def latlon_extent_to_robinson_axes_verts(polygon_coords):
        
            list_lat_interp = []
            list_lon_interp = []
            for i in range(len(polygon_coords)-1):
                lon_interp = np.linspace(polygon_coords[i][0],polygon_coords[i+1][0],50)
                lat_interp =  np.linspace(polygon_coords[i][1],polygon_coords[i+1][1],50)
        
                list_lon_interp.append(lon_interp)
                list_lat_interp.append(lat_interp)
        
            all_lon_interp = np.concatenate(list_lon_interp)
            all_lat_interp = np.concatenate(list_lat_interp)
        
            robin = coordXform(ccrs.PlateCarree(),ccrs.Robinson(),all_lon_interp,all_lat_interp)
        
            limits_robin = coordXform(ccrs.PlateCarree(),ccrs.Robinson(),np.array([-179.99,179.99,0,0]),np.array([0,0,-89.99,89.99]))
        
            ext_robin_x = limits_robin[1][0] - limits_robin[0][0]
            ext_robin_y = limits_robin[3][1] - limits_robin[2][1]
        
            verts = robin.copy()
            verts[:,0] = (verts[:,0] + limits_robin[1][0])/ext_robin_x
            verts[:,1] = (verts[:,1] + limits_robin[3][1])/ext_robin_y
        
            return verts[:,0:2]
        
        def shades_main_to_inset(main_pos,inset_pos,inset_verts,label):
        
            center_x = main_pos[0] + main_pos[2]/2
            center_y = main_pos[1] + main_pos[3]/2
        
            left_x = center_x - inset_pos[2]/2
            left_y = center_y - inset_pos[3]/2
        
            shade_ax = fig.add_axes([left_x,left_y,inset_pos[2],inset_pos[3]],projection=ccrs.Robinson(),label=label+'shade')
            shade_ax.set_extent([-179.99,179.99,-89.99,89.99],ccrs.PlateCarree())
        
            #first, get the limits of the manually positionned exploded polygon in projection coordinates
            limits_robin = coordXform(ccrs.PlateCarree(), ccrs.Robinson(), np.array([-179.99, 179.99, 0, 0]),
                                      np.array([0, 0, -89.99, 89.99]))
        
            ext_robin_x = limits_robin[1][0] - limits_robin[0][0]
            ext_robin_y = limits_robin[3][1] - limits_robin[2][1]
        
            inset_mod_x = inset_verts[:,0] +  (inset_pos[0]-left_x)/inset_pos[2]
            inset_mod_y = inset_verts[:,1] +  (inset_pos[1]-left_y)/inset_pos[3]
        
            #then, get the limits of the polygon in the manually positionned center map
            main_mod_x = (inset_verts[:, 0]*main_pos[2] - left_x + main_pos[0])/inset_pos[2]
            main_mod_y = (inset_verts[:, 1]*main_pos[3] - left_y + main_pos[1])/inset_pos[3]
        
            points = np.array(list(zip(np.concatenate((inset_mod_x,main_mod_x)),np.concatenate((inset_mod_y,main_mod_y)))))
        
            chull = ConvexHull(points)
        
            chull_robin_x = points[chull.vertices,0]*ext_robin_x - limits_robin[1][0]
            chull_robin_y = points[chull.vertices,1]*ext_robin_y - limits_robin[3][1]
        
        #    col_contour = mpl.cm.Greys(0.8)
        
#            shade_ax.plot(main_mod_x*ext_robin_x - limits_robin[1][0],main_mod_y*ext_robin_y - limits_robin[3][1],color='white',linewidth=0.75)
            shade_ax.plot(main_mod_x*ext_robin_x - limits_robin[1][0],main_mod_y*ext_robin_y - limits_robin[3][1],color='k',linewidth=0.75)
#            shade_ax.fill(chull_robin_x, chull_robin_y, transform=ccrs.Robinson(), color=color_shades_to_inset, alpha=0.05, zorder=1)
            verts = mpath.Path(np.column_stack((chull_robin_x,chull_robin_y)))
            shade_ax.set_boundary(verts, transform=shade_ax.transAxes)
        
        def only_shade(position,bounds,label,polygon=None):
            main_pos = [0.375, 0.21, 0.25, 0.25]
        
            if polygon is None and bounds is not None:
                polygon = poly_from_extent(bounds)
        
            shades_main_to_inset(main_pos, position, latlon_extent_to_robinson_axes_verts(polygon), label=label)
        
        def add_inset(fig,extent,position,bounds=None,label=None,polygon=None,shades=True, hillshade=True, list_shp=None, main=False, markup=None,markpos='left',markadj=0,markup_sub=None,sub_pos='lt',
                      col_bounds=None, color_water=color_water, color_land=color_land, add_rgi_glaciers=False):
            main_pos = [0.375, 0.21, 0.25, 0.25]
        
            if polygon is None and bounds is not None:
                polygon = poly_from_extent(bounds)
        
            if shades:
                shades_main_to_inset(main_pos, position, latlon_extent_to_robinson_axes_verts(polygon), label=label)
        
            sub_ax = fig.add_axes(position,
                                  projection=ccrs.Robinson(),label=label)
            sub_ax.set_extent(extent, ccrs.Geodetic())
        
            sub_ax.add_feature(cfeature.NaturalEarthFeature('physical', 'ocean', '50m', facecolor=color_water))
            sub_ax.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '50m', facecolor=color_land))
            
            # Add RGI glacier outlines
            if add_rgi_glaciers:
                shape_feature = ShapelyFeature(Reader(rgi_shp_fn).geometries(), ccrs.Robinson(),alpha=1,facecolor='indigo',linewidth=0.35,edgecolor='indigo')
                sub_ax.add_feature(shape_feature)
        
            if bounds is not None:
                verts = mpath.Path(latlon_extent_to_robinson_axes_verts(polygon))
                sub_ax.set_boundary(verts, transform=sub_ax.transAxes)
        
            # HERE IS WHERE VALUES APPEAR TO BE PROVIDED
            if not main:
                print(label)
                for i in range(len(tiles)):
                    lon = tiles[i][0]
                    lat = tiles[i][1]
        
                    if label=='Arctic West' and ((lat < 71 and lon > 60) or (lat <76 and lon>100)):
                        continue
        
                    if label=='HMA' and lat >=46:
                        continue
        
                    # fac = 0.02
                    fac = 1000
                    
                    area_2100 = areas_2100[i]
        
                    if areas[i] > 10:
                        rad = 15000 + np.sqrt(areas[i]) * fac
                    else:
                        rad = 15000 + 10 * fac
                    cb = []
                    cb_val = np.linspace(0, 1, len(col_bounds))
                    for j in range(len(cb_val)):
                        cb.append(mpl.cm.RdYlBu(cb_val[j]))
##                    cb[5] = cb[4]
##                    cb[4] = cb[3]
##                    cb[3] = cb[2]
##                    cb[2] = cb[1]
##                    cb[1] = cb[0]
#                    cb[11] = (78/256, 179/256, 211/256, 1)
#                    cb[10] = (123/256, 204/256, 196/256, 1)
#                    cb[9] = (168/256, 221/256, 181/256, 1)
#                    cb[8] = (204/256, 235/256, 197/256, 1)
#                    cb[7] = (224/256, 243/256, 219/256, 1)
#                    cb[6] = (254/256, 178/256, 76/256, 1)
#                    cb[5] = (253/256, 141/256, 60/256, 1)
#                    cb[4] = (252/256, 78/256, 42/256, 1)
#                    cb[3] = (227/256, 26/256, 28/256, 1)
#                    cb[2] = (189/256, 0/256, 38/256, 1)
#                    cb[1] = (128/256, 0/256, 38/256, 1)
#                    cb[0] = (1,1,1,1)
                    cb[11] = (94/256, 79/256, 162/256, 1)
                    cb[10] = (50/256, 136/256, 189/256, 1)
                    cb[9] = (102/256, 194/256, 165/256, 1)
                    cb[8] = (171/256, 221/256, 164/256, 1)
                    cb[7] = (230/256, 245/256, 152/256, 1)
                    cb[6] = (255/256, 255/256, 191/256, 1)
                    cb[5] = (254/256, 224/256, 139/256, 1)
                    cb[4] = (253/256, 174/256, 97/256, 1)
                    cb[3] = (244/256, 109/256, 67/256, 1)
                    cb[2] = (213/256, 62/256, 79/256, 1)
                    cb[1] = (158/256, 1/256, 66/256, 1)
                    cb[0] = (1,1,1,1)
                    cmap_cus = mpl.colors.LinearSegmentedColormap.from_list('my_cb', list(
                        zip(col_bounds, cb)))
        
                    # xy = [lon,lat]
                    xy = coordXform(ccrs.PlateCarree(),ccrs.Robinson(),np.array([lon]),np.array([lat]))[0][0:2]
                    
                    # Less than percent threshold and area threshold
                    dhdt = dhs[i]
                    col = cmap_cus(dhdt)
                    if dhdt < col_bounds[1] or area_2100 < 0.005:
                        sub_ax.add_patch(
                            mpatches.Circle(xy=xy, radius=rad, facecolor='white', edgecolor='black', linewidth=0.5, alpha=1, transform=ccrs.Robinson(), zorder=30))
                    else:
                        sub_ax.add_patch(
                            mpatches.Circle(xy=xy, radius=rad, facecolor=col, edgecolor='None', alpha=1, transform=ccrs.Robinson(), zorder=30))
                    
            if markup is not None:
                if markpos=='left':
                    lon_upleft = np.min(list(zip(*polygon))[0])
                    lat_upleft = np.max(list(zip(*polygon))[1])
                else:
                    lon_upleft = np.max(list(zip(*polygon))[0])
                    lat_upleft = np.max(list(zip(*polygon))[1])
        
                robin = coordXform(ccrs.PlateCarree(),ccrs.Robinson(),np.array([lon_upleft]),np.array([lat_upleft]))
        
                rob_x = robin[0][0]
                rob_y = robin[0][1]
        
                size_y = 200000
                size_x = 80000 * len(markup) + markadj
        
                if markpos=='right':
                    rob_x = rob_x-50000
                else:
                    rob_x = rob_x+50000
        
                sub_ax_2 = fig.add_axes(position,
                                        projection=ccrs.Robinson(), label=label+'markup')
        
                # adds the white box to the region
#                sub_ax_2.add_patch(mpatches.Rectangle((rob_x, rob_y), size_x, size_y , linewidth=1, edgecolor='grey', facecolor='white',transform=ccrs.Robinson()))
        
                sub_ax_2.set_extent(extent, ccrs.Geodetic())
                verts = mpath.Path(rect_units_to_verts([rob_x,rob_y,size_x,size_y]))
                sub_ax_2.set_boundary(verts, transform=sub_ax.transAxes)
        
                sub_ax_2.text(rob_x,rob_y+50000,markup,
                         horizontalalignment=markpos, verticalalignment='bottom',
                         transform=ccrs.Robinson(), color='black',fontsize=4.5, fontweight='bold',bbox= dict(facecolor='white', alpha=1,linewidth=0.35,pad=1.5))
        
            if markup_sub is not None:
        
                lon_min = np.min(list(zip(*polygon))[0])
                lon_max = np.max(list(zip(*polygon))[0])
                lon_mid = 0.5*(lon_min+lon_max)
        
                lat_min = np.min(list(zip(*polygon))[1])
                lat_max = np.max(list(zip(*polygon))[1])
                lat_mid = 0.5*(lat_min+lat_max)
        
                size_y = 150000
                size_x = 150000
        
                lat_midup = lat_min+0.87*(lat_max-lat_min)
        
                robin = coordXform(ccrs.PlateCarree(),ccrs.Robinson(),np.array([lon_min,lon_min,lon_min,lon_mid,lon_mid,lon_max,lon_max,lon_max,lon_min]),np.array([lat_min,lat_mid,lat_max,lat_min,lat_max,lat_min,lat_mid,lat_max,lat_midup]))
        
                if sub_pos=='lb':
                    rob_x = robin[0][0]
                    rob_y = robin[0][1]
                    ha='left'
                    va='bottom'
                elif sub_pos=='lm':
                    rob_x = robin[1][0]
                    rob_y = robin[1][1]
                    ha='left'
                    va='center'
                elif sub_pos == 'lm2':
                    rob_x = robin[8][0]
                    rob_y = robin[8][1]
                    ha = 'left'
                    va = 'center'
                elif sub_pos=='lt':
                    rob_x = robin[2][0]
                    rob_y = robin[2][1]
                    ha='left'
                    va='top'
                elif sub_pos=='mb':
                    rob_x = robin[3][0]
                    rob_y = robin[3][1]
                    ha='center'
                    va='bottom'
                elif sub_pos=='mt':
                    rob_x = robin[4][0]
                    rob_y = robin[4][1]
                    ha='center'
                    va='top'
                elif sub_pos=='rb':
                    rob_x = robin[5][0]
                    rob_y = robin[5][1]
                    ha='right'
                    va='bottom'
                elif sub_pos=='rm':
                    rob_x = robin[6][0]
                    rob_y = robin[6][1]
                    ha='right'
                    va='center'
                elif sub_pos=='rt':
                    rob_x = robin[7][0]
                    rob_y = robin[7][1]
                    ha='right'
                    va='top'
        
                if sub_pos[0] == 'r':
                    rob_x = rob_x - 50000
                elif sub_pos[0] == 'l':
                    rob_x = rob_x + 50000
        
                if sub_pos[1] == 'b':
                    rob_y = rob_y + 50000
                elif sub_pos[1] == 't':
                    rob_y = rob_y - 50000
        
                sub_ax_3 = fig.add_axes(position,
                                        projection=ccrs.Robinson(), label=label+'markup2')
        
                # sub_ax_3.add_patch(mpatches.Rectangle((rob_x, rob_y), size_x, size_y , linewidth=1, edgecolor='grey', facecolor='white',transform=ccrs.Robinson()))
        
                sub_ax_3.set_extent(extent, ccrs.Geodetic())
                verts = mpath.Path(rect_units_to_verts([rob_x,rob_y,size_x,size_y]))
                sub_ax_3.set_boundary(verts, transform=sub_ax.transAxes)
        
                sub_ax_3.text(rob_x,rob_y,markup_sub,
                         horizontalalignment=ha, verticalalignment=va,
                         transform=ccrs.Robinson(), color='black',fontsize=4.5,bbox=dict(facecolor='white', alpha=1,linewidth=0.35,pad=1.5),fontweight='bold',zorder=25)
        
            if not main:
        #        sub_ax.outline_patch.set_edgecolor('white')
#                sub_ax.spines['geo'].set_edgecolor('white')
                sub_ax.spines['geo'].set_edgecolor('k')
            else:
        #        sub_ax.outline_patch.set_edgecolor('lightgrey')
                sub_ax.spines['geo'].set_edgecolor('lightgrey')
        
    
        #TODO: careful here! figure size determines everything else, found no way to do it otherwise in cartopy
        fig_width_inch=7.2
        fig = plt.figure(figsize=(fig_width_inch,fig_width_inch/1.9716))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
    #    ax = fig.add_axes([0,0.12,1,0.88], projection=ccrs.Robinson())
        
        ax.set_global()
        ax.spines['geo'].set_linewidth(0)
        
        add_inset(fig,[-179.99,179.99,-89.99,89.99],[0.375, 0.21, 0.25, 0.25],bounds=[-179.99,179.99,-89.99,89.99],shades=False, hillshade = False, main=True, col_bounds=col_bounds,
                  color_water='lightblue', color_land='white', add_rgi_glaciers=add_rgi_glaciers
                  )
#        #add_inset(fig,[-179.99,179.99,-89.99,89.99],[0.375, 0.21, 0.25, 0.25],bounds=[-179.99,179.99,-89.99,89.99],shades=False, hillshade = False, main=True, list_shp=shp_buff)
        
        if 19 in regions:
            poly_aw = np.array([(-158,-79),(-135,-60),(-110,-60),(-50,-60),(-50,-79.25),(-158,-79.25)])
            add_inset(fig,[-179.99,179.99,-89.99,89.99],[-0.4,-0.065,2,2],bounds=[-158, -45, -40, -79],
                      label='Antarctic_West', polygon=poly_aw,shades=True,markup_sub='West and Peninsula',sub_pos='mb', col_bounds=col_bounds)
            poly_ae = np.array([(135,-81.5),(152,-63.7),(165,-65),(175,-70),(175,-81.25),(135,-81.75)])
            add_inset(fig,[-179.99,179.99,-89.99,89.99],[-0.71,-0.045,2,2],bounds=[130, 175, -64.5, -81],
                      label='Antarctic_East', polygon=poly_ae,shades=True,markup_sub='East 2',sub_pos='mb', col_bounds=col_bounds)
            
            poly_ac = np.array([(-25,-62),(106,-62),(80,-79.25),(-25,-79.25),(-25,-62)])
            add_inset(fig,[-179.99,179.99,-89.99,89.99],[-0.52,-0.065,2,2],bounds=[-25, 106, -62.5, -79],
                      label='Antarctic_Center',polygon=poly_ac,shades=True,markup='Antarctic and Subantarctic',
                      markpos='right',markadj=0,markup_sub='East 1',sub_pos='mb', col_bounds=col_bounds)
            
            add_inset(fig,[-179.99,179.99,-89.99,89.99],[-0.68,-0.18,2,2],bounds=[64, 78, -48, -56],
                      label='Antarctic_Australes', shades=True,markup_sub='Kerguelen and Heard Islands',sub_pos='lb', col_bounds=col_bounds)
            
            add_inset(fig,[-179.99,179.99,-89.99,89.99],[-0.42,-0.143,2,2],bounds=[-40, -23, -53, -62],
                      label='Antarctic_South_Georgia', shades=True,markup_sub='South Georgia and Central Islands',sub_pos='lb', col_bounds=col_bounds)
        
        if 16 in regions or 17 in regions:
            add_inset(fig, [-179.99,179.99,-89.99,89.99], [-0.52, -0.225, 2, 2],bounds=[-82,-65,13,-57],label='Andes',
                      markup='Low Latitudes &\nSouthern Andes',markadj=0,
#                      markup_sub='a',
                      sub_pos='lm2', col_bounds=col_bounds)
            add_inset(fig, [-179.99,179.99,-89.99,89.99], [-0.352, -0.38, 2, 2],bounds=[-100,-95,22,16],label='Mexico',
                      markup_sub='Mexico',sub_pos='rb', col_bounds=col_bounds)
            add_inset(fig, [-179.99,179.99,-89.99,89.99], [-1.078, -0.22, 2, 2],bounds=[28,42,2,-6],label='Africa',
                      markup_sub='East Africa',sub_pos='rb', col_bounds=col_bounds)
            add_inset(fig, [-179.99,179.99,-89.99,89.99], [-1.64, -0.3, 2, 2],bounds=[133,140,-2,-7],label='Indonesia',
                      markup_sub='New Guinea',sub_pos='lb', col_bounds=col_bounds)
        
        if 3 in regions or 4 in regions or 5 in regions or 6 in regions or 7 in regions or 8 in regions or 9 in regions: 
            poly_arctic = np.array([(-105,84.5),(115,84.5),(110,68),(30,68),(18,57),(-70,57),(-100,75),(-105,84.5)])
            add_inset(fig,[-179.99,179.99,-89.99,89.99],[-0.48,-1.003,2,2],bounds=[-100, 106, 57, 84],label='Arctic West',
                      polygon=poly_arctic,markup='Arctic',markadj=0, col_bounds=col_bounds)
            
        if 18 in regions:
            add_inset(fig,[-179.99,179.99,-89.99,89.99],[-0.92,-0.17,2,2],bounds=[164,176,-47,-40],label='New Zealand',
                      markup='New Zealand',markpos='right',markadj=0, col_bounds=col_bounds)
        
        if 1 in regions or 2 in regions:
            poly_na = np.array([(-170,72),(-140,72),(-120,63),(-101,35),(-126,35),(-165,55),(-170,72)])
            add_inset(fig,[-179.99,179.99,-89.99,89.99],[-0.1,-1.22,2,2],bounds=[-177,-105, 36, 70],label='North America',
                      polygon=poly_na,markup='Alaska & Western\nCanada and US',markadj=0, col_bounds=col_bounds)
        
        if 10 in regions:
            
#            poly_asia_ne = np.array([(142,71),(142,82),(163,82),(155,71),(142,71)])
#            add_inset(fig,[-179.99,179.99,-89.99,89.99],[-0.64,-1.165,2,2],bounds=[142,160,71,80],
#                      polygon=poly_asia_ne,label='North Asia North E',markup_sub='Bulunsky',sub_pos='rt', col_bounds=col_bounds)
            poly_asia_e2 = np.array([(125,57),(125,70.5),(153.8,70.5),(148,57),(125,57)])
            only_shade([-0.71,-1.142,2,2],[125,148,58,72],polygon=poly_asia_e2,
                       label='tmp_NAE2')
            only_shade([-0.517,-1.035,2,2],[53,70,62,69.8],label='tmp_NAW')
            
            add_inset(fig,[-179.99,179.99,-89.99,89.99],[-0.575,-1.109,2,2],bounds=[87,112,68,78.5],
                      label='North Asia North W',markup_sub='North Siberia',sub_pos='rb', col_bounds=col_bounds)
            
            add_inset(fig,[-179.99,179.99,-89.99,89.99],[-0.71,-1.137,2,2],bounds=[125,148,54,68],polygon=poly_asia_e2,
                      label='North Asia East 2',markup_sub='Cherskiy and\nSuntar Khayata',sub_pos='lb',shades=False, col_bounds=col_bounds)
            
            poly_asia = np.array([(148,49),(160,64),(178,64),(170,55),(160,49),(148,49)])
            add_inset(fig,[-179.99,179.99,-89.99,89.99],[-0.823,-1.22,2,2],bounds=[127,179.9,50,64.8],
                      label='North Asia East',polygon=poly_asia,markup_sub='Kamchatka Krai',sub_pos='lb', col_bounds=col_bounds)
        
            
            add_inset(fig,[-179.99,179.99,-89.99,89.99],[-0.75,-1.01,2,2],bounds=[82,120,45.5,58.9],
                      label='South Asia North',markup='North Asia',markup_sub='Altay and Sayan',sub_pos='rb',markadj=0, 
                      col_bounds=col_bounds)
            add_inset(fig,[-179.99,179.99,-89.99,89.99],[-0.525,-1.045,2,2],bounds=[53,68,62,68.5],
                      label='North Asia West',markup_sub='Ural',sub_pos='rb',shades=False, col_bounds=col_bounds)
        
        if 13 in regions or 14 in regions or 15 in regions:
            add_inset(fig,[-179.99,179.99,-89.99,89.99],[-0.685,-1.065,2,2],bounds=[65, 105, 46.5, 25],
                      label='HMA',markup='High Mountain Asia',markadj=0, col_bounds=col_bounds)
        
        if 11 in regions:
            add_inset(fig,[-179.99,179.99,-89.99,89.99],[-0.58,-0.982,2,2],bounds=[-4.9,19,38.2,50.5],
                      label='Europe',markup='Central Europe',markadj=0, col_bounds=col_bounds)
        
        if 12 in regions:
            add_inset(fig,[-179.99,179.99,-89.99,89.99],[-0.66,-0.896,2,2],bounds=[38,54,29.6,43.6],
                      label='Middle East',markup='Caucasus and\nMiddle East',markadj=0, col_bounds=col_bounds)
        
        # ----- Circle sizes -----
#        axleg_background = fig.add_axes([0.001, 0.04, 0.107, 0.2])
        axleg_background = fig.add_axes([0.001, 0.04, 0.09, 0.53])
        axleg_background.get_yaxis().set_visible(False)
        axleg_background.get_xaxis().set_visible(False)
#        axleg_background.axis('off')
        rect1 = mpl.patches.Rectangle((0, 0), 1, 1, color ='white')
        axleg_background.add_patch(rect1)
        
        axleg = fig.add_axes([-0.92,-0.86,2,2],projection=ccrs.Robinson(),label='legend')
        axleg.set_extent([-179.99,179.99,-89.99,89.99], ccrs.Geodetic())
        axleg.outline_patch.set_linewidth(0)
        u=0
        rad_tot = 0
        for a in [10000, 1000, 100]:
            rad = (12000+np.sqrt(a)*1000)
            axleg.add_patch(mpatches.Circle(xy=[-900000,-680000+u*380000],radius=rad,edgecolor='k',label=str(a)+' km$^2$', transform = ccrs.Robinson(),fill=False, zorder=30))
            u=u+1
            rad_tot += rad
        axleg.text(-7.9, 2.4, '10$^{2}$\n10$^{3}$\n10$^{4}$', transform=ccrs.Geodetic(),horizontalalignment='left',verticalalignment='top',fontsize=10)
        axleg.text(-6.2, 9, 'Area', transform=ccrs.Geodetic(),horizontalalignment='center',verticalalignment='top',fontsize=10)
        axleg.text(-6.2, 6.2, '(km$^2$)', transform=ccrs.Geodetic(),horizontalalignment='center',verticalalignment='top',fontsize=9)
        

        
        # ----- Colorbar -----
        cb = []
        cb_val = np.linspace(0, 1, len(col_bounds))
        for j in range(len(cb_val)):
            cb.append(mpl.cm.RdYlBu(cb_val[j]))
##        cb[5] = cb[4]
##        cb[4] = cb[3]
##        cb[3] = cb[2]
##        cb[2] = cb[1]
##        cb[1] = cb[0]
##        cb[0] = (1,1,1,1)
#        cb[11] = (78/256, 179/256, 211/256, 1)
#        cb[10] = (123/256, 204/256, 196/256, 1)
#        cb[9] = (168/256, 221/256, 181/256, 1)
#        cb[8] = (204/256, 235/256, 197/256, 1)
#        cb[7] = (224/256, 243/256, 219/256, 1)
#        cb[6] = (254/256, 178/256, 76/256, 1)
#        cb[5] = (253/256, 141/256, 60/256, 1)
#        cb[4] = (252/256, 78/256, 42/256, 1)
#        cb[3] = (227/256, 26/256, 28/256, 1)
#        cb[2] = (189/256, 0/256, 38/256, 1)
#        cb[1] = (128/256, 0/256, 38/256, 1)
#        cb[0] = (1,1,1,1)
        
        cb[11] = (94/256, 79/256, 162/256, 1)
        cb[10] = (50/256, 136/256, 189/256, 1)
        cb[9] = (102/256, 194/256, 165/256, 1)
        cb[8] = (171/256, 221/256, 164/256, 1)
        cb[7] = (230/256, 245/256, 152/256, 1)
        cb[6] = (255/256, 255/256, 191/256, 1)
        cb[5] = (254/256, 224/256, 139/256, 1)
        cb[4] = (253/256, 174/256, 97/256, 1)
        cb[3] = (244/256, 109/256, 67/256, 1)
        cb[2] = (213/256, 62/256, 79/256, 1)
        cb[1] = (158/256, 1/256, 66/256, 1)
        cb[0] = (1,1,1,1)
        
        cmap_cus = mpl.colors.LinearSegmentedColormap.from_list('my_cb', cb)
        norm = mpl.colors.BoundaryNorm(col_bounds, cmap_cus.N)
        
        cax = fig.add_axes([0.045, 0.275, 0.007, 0.28], facecolor='none')
        sm = plt.cm.ScalarMappable(cmap=cmap_cus,norm=norm )
        cbar = plt.colorbar(sm, ticks=col_bounds, ax=ax, cax=cax, orientation='vertical')
        cax.xaxis.set_ticks_position('bottom')
        cax.xaxis.set_tick_params(pad=0)
        tick_labels = [x for x in col_bounds]
        tick_labels[10] = ''
        tick_labels[8] = ''
        tick_labels[6] = ''
        tick_labels[4] = ''
        tick_labels[2] = ''
        tick_labels[1] = '0.0'
        tick_labels[0] = ''
        cbar.set_ticklabels(tick_labels)
        cbar.ax.tick_params(labelsize=9, pad=0.3)
        
        # Use if running in command line
#        cax.text(-0.135, 0.215, 'Mass at 2100,\nrel. to 2015 (-)', size=10, horizontalalignment='center',
#                verticalalignment='bottom', rotation=90, transform=ax.transAxes)
        # Use if running in Spyder
        cax.text(-0.148, 0.205, 'Mass at 2100,\nrel. to 2015 (-)', size=10, horizontalalignment='center',
                 verticalalignment='bottom', rotation=90, transform=ax.transAxes)
    
        fig.savefig(fig_fp + 'global_deg_vol_remaining_' + str(warming_group) + 'degC.png',dpi=250,transparent=True)
        
    #%%
    # ---- DIFFERENCE WARMING PAIRS -----
    warming_group_difpairs = [(1.5,2.7), (1.5,2), (1.5,3), (2,3), (2,2.7)]
#    col_bounds = np.array([-0.5,-0.4,-0.3,-0.2,-0.1, 0])
#    col_bounds = np.array([-0.25,-0.2,-0.15,-0.1,-0.05, 0])
    col_bounds = np.array([0,0.05,0.1,0.15,0.2,0.25])
    for warming_group_difpair in warming_group_difpairs:
        
        warming_group_1 = warming_group_difpair[0]
        warming_group_2 = warming_group_difpair[1]
        
        print('warming pair:', warming_group_1, warming_group_2)

        normyear_idx = np.where(years == normyear)[0][0]
        
        # Area are in km2
        areas_wg1 = ds_multigcm_area['all'][warming_group_1][:,normyear_idx] / 1e6
        areas_2100_wg1 = ds_multigcm_area['all'][warming_group_1][:,-1] / 1e6
        
        areas_wg2 = ds_multigcm_area['all'][warming_group_2][:,normyear_idx] / 1e6
        areas_2100_wg2 = ds_multigcm_area['all'][warming_group_2][:,-1] / 1e6
        
        # dh is the elevation change that is used for color; we'll use volume remaining by end of century for now
        vols_remaining_frac_wg1 = ds_multigcm_vol['all'][warming_group_1][:,-1] / ds_multigcm_vol['all'][warming_group_1][:,normyear_idx]        
        vols_remaining_frac_wg2 = ds_multigcm_vol['all'][warming_group_2][:,-1] / ds_multigcm_vol['all'][warming_group_2][:,normyear_idx]
        
#        dhs = vols_remaining_frac_wg2 - vols_remaining_frac_wg1
        dhs = vols_remaining_frac_wg1 - vols_remaining_frac_wg2
    
        areas = [area for _, area in sorted(zip(tiles_all,areas_wg1))]
        areas_2100 = [area for _, area in sorted(zip(tiles_all,areas_2100_wg1))]
        areas_2100_wg2 = [area for _, area in sorted(zip(tiles_all,areas_2100_wg2))]
        dhs = [dh for _, dh in sorted(zip(tiles_all,dhs))]
        tiles = sorted(tiles_all)
        
        
        def add_inset(fig,extent,position,bounds=None,label=None,polygon=None,shades=True, hillshade=True, list_shp=None, main=False, markup=None,markpos='left',markadj=0,markup_sub=None,sub_pos='lt',
                      col_bounds=None):
            main_pos = [0.375, 0.21, 0.25, 0.25]
        
            if polygon is None and bounds is not None:
                polygon = poly_from_extent(bounds)
        
            if shades:
                shades_main_to_inset(main_pos, position, latlon_extent_to_robinson_axes_verts(polygon), label=label)
        
            sub_ax = fig.add_axes(position,
                                  projection=ccrs.Robinson(),label=label)
            sub_ax.set_extent(extent, ccrs.Geodetic())
        
            sub_ax.add_feature(cfeature.NaturalEarthFeature('physical', 'ocean', '50m', facecolor=color_water))
            sub_ax.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '50m', facecolor=color_land))
        
            if bounds is not None:
                verts = mpath.Path(latlon_extent_to_robinson_axes_verts(polygon))
                sub_ax.set_boundary(verts, transform=sub_ax.transAxes)
        
            # HERE IS WHERE VALUES APPEAR TO BE PROVIDED
            if not main:
                print(label)
                for i in range(len(tiles)):
                    lon = tiles[i][0]
                    lat = tiles[i][1]
        
                    if label=='Arctic West' and ((lat < 71 and lon > 60) or (lat <76 and lon>100)):
                        continue
        
                    if label=='HMA' and lat >=46:
                        continue
                    
        
                    # fac = 0.02
                    fac = 1000
                    
                    area_2100 = areas_2100[i]
                    area_2100_wg2 = areas_2100_wg2[i]
                    
        
                    if areas[i] > 10:
                        rad = 15000 + np.sqrt(areas[i]) * fac
                    else:
                        rad = 15000 + 10 * fac
                    cb = []
                    cb_val = np.linspace(0, 1, len(col_bounds))
                    for j in range(len(cb_val)):
                        cb.append(mpl.cm.autumn_r(cb_val[j]))
                    cmap_cus = mpl.colors.LinearSegmentedColormap.from_list('my_cb', list(
                        zip((col_bounds - min(col_bounds)) / (max(col_bounds - min(col_bounds))), cb)), N=len(col_bounds))
        
                    # xy = [lon,lat]
                    xy = coordXform(ccrs.PlateCarree(),ccrs.Robinson(),np.array([lon]),np.array([lat]))[0][0:2]
                    
                    # Less than percent threshold and area threshold
                    dhdt = dhs[i]
                    
                    dhdt_col = max(0.0001,min(0.9999,(dhdt - min(col_bounds))/(max(col_bounds)-min(col_bounds))))
                    col = cmap_cus(dhdt_col)
                    # If missing for both, then put as black
                    if area_2100 < 0.005 and area_2100_wg2 < 0.005:
                        sub_ax.add_patch(
                            mpatches.Circle(xy=xy, radius=rad, facecolor='black', edgecolor='black', linewidth=0.5, alpha=1, transform=ccrs.Robinson(), zorder=30))  
                    # If becomes missing, then put as white
                    elif area_2100_wg2 < 0.005:
                        sub_ax.add_patch(
                            mpatches.Circle(xy=xy, radius=rad, facecolor=col, edgecolor='black', linewidth=0.5, alpha=1, transform=ccrs.Robinson(), zorder=30))
                    # Otherwise show color
                    else:
                        sub_ax.add_patch(
                            mpatches.Circle(xy=xy, radius=rad, facecolor=col, edgecolor='None', alpha=1, transform=ccrs.Robinson(), zorder=30))
                    
            if markup is not None:
                if markpos=='left':
                    lon_upleft = np.min(list(zip(*polygon))[0])
                    lat_upleft = np.max(list(zip(*polygon))[1])
                else:
                    lon_upleft = np.max(list(zip(*polygon))[0])
                    lat_upleft = np.max(list(zip(*polygon))[1])
        
                robin = coordXform(ccrs.PlateCarree(),ccrs.Robinson(),np.array([lon_upleft]),np.array([lat_upleft]))
        
                rob_x = robin[0][0]
                rob_y = robin[0][1]
        
                size_y = 200000
                size_x = 80000 * len(markup) + markadj
        
                if markpos=='right':
                    rob_x = rob_x-50000
                else:
                    rob_x = rob_x+50000
        
                sub_ax_2 = fig.add_axes(position,
                                        projection=ccrs.Robinson(), label=label+'markup')
        
                # adds the white box to the region
#                sub_ax_2.add_patch(mpatches.Rectangle((rob_x, rob_y), size_x, size_y , linewidth=1, edgecolor='grey', facecolor='white',transform=ccrs.Robinson()))
        
                sub_ax_2.set_extent(extent, ccrs.Geodetic())
                verts = mpath.Path(rect_units_to_verts([rob_x,rob_y,size_x,size_y]))
                sub_ax_2.set_boundary(verts, transform=sub_ax.transAxes)
        
                sub_ax_2.text(rob_x,rob_y+50000,markup,
                         horizontalalignment=markpos, verticalalignment='bottom',
                         transform=ccrs.Robinson(), color='black',fontsize=4.5, fontweight='bold',bbox= dict(facecolor='white', alpha=1,linewidth=0.35,pad=1.5))
        
            if markup_sub is not None:
        
                lon_min = np.min(list(zip(*polygon))[0])
                lon_max = np.max(list(zip(*polygon))[0])
                lon_mid = 0.5*(lon_min+lon_max)
        
                lat_min = np.min(list(zip(*polygon))[1])
                lat_max = np.max(list(zip(*polygon))[1])
                lat_mid = 0.5*(lat_min+lat_max)
        
                size_y = 150000
                size_x = 150000
        
                lat_midup = lat_min+0.87*(lat_max-lat_min)
        
                robin = coordXform(ccrs.PlateCarree(),ccrs.Robinson(),np.array([lon_min,lon_min,lon_min,lon_mid,lon_mid,lon_max,lon_max,lon_max,lon_min]),np.array([lat_min,lat_mid,lat_max,lat_min,lat_max,lat_min,lat_mid,lat_max,lat_midup]))
        
                if sub_pos=='lb':
                    rob_x = robin[0][0]
                    rob_y = robin[0][1]
                    ha='left'
                    va='bottom'
                elif sub_pos=='lm':
                    rob_x = robin[1][0]
                    rob_y = robin[1][1]
                    ha='left'
                    va='center'
                elif sub_pos == 'lm2':
                    rob_x = robin[8][0]
                    rob_y = robin[8][1]
                    ha = 'left'
                    va = 'center'
                elif sub_pos=='lt':
                    rob_x = robin[2][0]
                    rob_y = robin[2][1]
                    ha='left'
                    va='top'
                elif sub_pos=='mb':
                    rob_x = robin[3][0]
                    rob_y = robin[3][1]
                    ha='center'
                    va='bottom'
                elif sub_pos=='mt':
                    rob_x = robin[4][0]
                    rob_y = robin[4][1]
                    ha='center'
                    va='top'
                elif sub_pos=='rb':
                    rob_x = robin[5][0]
                    rob_y = robin[5][1]
                    ha='right'
                    va='bottom'
                elif sub_pos=='rm':
                    rob_x = robin[6][0]
                    rob_y = robin[6][1]
                    ha='right'
                    va='center'
                elif sub_pos=='rt':
                    rob_x = robin[7][0]
                    rob_y = robin[7][1]
                    ha='right'
                    va='top'
        
                if sub_pos[0] == 'r':
                    rob_x = rob_x - 50000
                elif sub_pos[0] == 'l':
                    rob_x = rob_x + 50000
        
                if sub_pos[1] == 'b':
                    rob_y = rob_y + 50000
                elif sub_pos[1] == 't':
                    rob_y = rob_y - 50000
        
                sub_ax_3 = fig.add_axes(position,
                                        projection=ccrs.Robinson(), label=label+'markup2')
        
                # sub_ax_3.add_patch(mpatches.Rectangle((rob_x, rob_y), size_x, size_y , linewidth=1, edgecolor='grey', facecolor='white',transform=ccrs.Robinson()))
        
                sub_ax_3.set_extent(extent, ccrs.Geodetic())
                verts = mpath.Path(rect_units_to_verts([rob_x,rob_y,size_x,size_y]))
                sub_ax_3.set_boundary(verts, transform=sub_ax.transAxes)
        
                sub_ax_3.text(rob_x,rob_y,markup_sub,
                         horizontalalignment=ha, verticalalignment=va,
                         transform=ccrs.Robinson(), color='black',fontsize=4.5,bbox=dict(facecolor='white', alpha=1,linewidth=0.35,pad=1.5),fontweight='bold',zorder=25)
        
            if not main:
        #        sub_ax.outline_patch.set_edgecolor('white')
                sub_ax.spines['geo'].set_edgecolor('white')
            else:
        #        sub_ax.outline_patch.set_edgecolor('lightgrey')
                sub_ax.spines['geo'].set_edgecolor('lightgrey')
        
    
        #TODO: careful here! figure size determines everything else, found no way to do it otherwise in cartopy
        fig_width_inch=7.2
        fig = plt.figure(figsize=(fig_width_inch,fig_width_inch/1.9716))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
    #    ax = fig.add_axes([0,0.12,1,0.88], projection=ccrs.Robinson())
        
        ax.set_global()
        ax.spines['geo'].set_linewidth(0)
        
        add_inset(fig,[-179.99,179.99,-89.99,89.99],[0.375, 0.21, 0.25, 0.25],bounds=[-179.99,179.99,-89.99,89.99],shades=False, hillshade = False, main=True, col_bounds=col_bounds)
        #add_inset(fig,[-179.99,179.99,-89.99,89.99],[0.375, 0.21, 0.25, 0.25],bounds=[-179.99,179.99,-89.99,89.99],shades=False, hillshade = False, main=True, list_shp=shp_buff)
        
#        add_inset(fig,[-179.99,179.99,-89.99,89.99],[0.375, 0.21, 0.25, 0.25],bounds=[-179.99,179.99,-89.99,89.99],shades=False, hillshade = False, main=True, col_bounds=col_bounds,
#                  color_water='lightblue', color_land='white', add_rgi_glaciers=add_rgi_glaciers
#                  )
#        #add_inset(fig,[-179.99,179.99,-89.99,89.99],[0.375, 0.21, 0.25, 0.25],bounds=[-179.99,179.99,-89.99,89.99],shades=False, hillshade = False, main=True, list_shp=shp_buff)
        
        if 19 in regions:
            poly_aw = np.array([(-158,-79),(-135,-60),(-110,-60),(-50,-60),(-50,-79.25),(-158,-79.25)])
            add_inset(fig,[-179.99,179.99,-89.99,89.99],[-0.4,-0.065,2,2],bounds=[-158, -45, -40, -79],
                      label='Antarctic_West', polygon=poly_aw,shades=True,markup_sub='West and Peninsula',sub_pos='mb', col_bounds=col_bounds)
            poly_ae = np.array([(135,-81.5),(152,-63.7),(165,-65),(175,-70),(175,-81.25),(135,-81.75)])
            add_inset(fig,[-179.99,179.99,-89.99,89.99],[-0.71,-0.045,2,2],bounds=[130, 175, -64.5, -81],
                      label='Antarctic_East', polygon=poly_ae,shades=True,markup_sub='East 2',sub_pos='mb', col_bounds=col_bounds)
            
            poly_ac = np.array([(-25,-62),(106,-62),(80,-79.25),(-25,-79.25),(-25,-62)])
            add_inset(fig,[-179.99,179.99,-89.99,89.99],[-0.52,-0.065,2,2],bounds=[-25, 106, -62.5, -79],
                      label='Antarctic_Center',polygon=poly_ac,shades=True,markup='Antarctic and Subantarctic',
                      markpos='right',markadj=0,markup_sub='East 1',sub_pos='mb', col_bounds=col_bounds)
            
            add_inset(fig,[-179.99,179.99,-89.99,89.99],[-0.68,-0.18,2,2],bounds=[64, 78, -48, -56],
                      label='Antarctic_Australes', shades=True,markup_sub='Kerguelen and Heard Islands',sub_pos='lb', col_bounds=col_bounds)
            
            add_inset(fig,[-179.99,179.99,-89.99,89.99],[-0.42,-0.143,2,2],bounds=[-40, -23, -53, -62],
                      label='Antarctic_South_Georgia', shades=True,markup_sub='South Georgia and Central Islands',sub_pos='lb', col_bounds=col_bounds)
        
        if 16 in regions or 17 in regions:
            add_inset(fig, [-179.99,179.99,-89.99,89.99], [-0.52, -0.225, 2, 2],bounds=[-82,-65,13,-57],label='Andes',
                      markup='Low Latitudes &\nSouthern Andes',markadj=0,
#                      markup_sub='a',
                      sub_pos='lm2', col_bounds=col_bounds)
            add_inset(fig, [-179.99,179.99,-89.99,89.99], [-0.352, -0.38, 2, 2],bounds=[-100,-95,22,16],label='Mexico',
                      markup_sub='Mexico',sub_pos='rb', col_bounds=col_bounds)
            add_inset(fig, [-179.99,179.99,-89.99,89.99], [-1.078, -0.22, 2, 2],bounds=[28,42,2,-6],label='Africa',
                      markup_sub='East Africa',sub_pos='rb', col_bounds=col_bounds)
            add_inset(fig, [-179.99,179.99,-89.99,89.99], [-1.64, -0.3, 2, 2],bounds=[133,140,-2,-7],label='Indonesia',
                      markup_sub='New Guinea',sub_pos='lb', col_bounds=col_bounds)
        
        if 3 in regions or 4 in regions or 5 in regions or 6 in regions or 7 in regions or 8 in regions or 9 in regions: 
            poly_arctic = np.array([(-105,84.5),(115,84.5),(110,68),(30,68),(18,57),(-70,57),(-100,75),(-105,84.5)])
            add_inset(fig,[-179.99,179.99,-89.99,89.99],[-0.48,-1.003,2,2],bounds=[-100, 106, 57, 84],label='Arctic West',
                      polygon=poly_arctic,markup='Arctic',markadj=0, col_bounds=col_bounds)
            
        if 18 in regions:
            add_inset(fig,[-179.99,179.99,-89.99,89.99],[-0.92,-0.17,2,2],bounds=[164,176,-47,-40],label='New Zealand',
                      markup='New Zealand',markpos='right',markadj=0, col_bounds=col_bounds)
        
        if 1 in regions or 2 in regions:
            poly_na = np.array([(-170,72),(-140,72),(-120,63),(-101,35),(-126,35),(-165,55),(-170,72)])
            add_inset(fig,[-179.99,179.99,-89.99,89.99],[-0.1,-1.22,2,2],bounds=[-177,-105, 36, 70],label='North America',
                      polygon=poly_na,markup='Alaska & Western\nCanada and US',markadj=0, col_bounds=col_bounds)
        
        if 10 in regions:
            
#            poly_asia_ne = np.array([(142,71),(142,82),(163,82),(155,71),(142,71)])
#            add_inset(fig,[-179.99,179.99,-89.99,89.99],[-0.64,-1.165,2,2],bounds=[142,160,71,80],
#                      polygon=poly_asia_ne,label='North Asia North E',markup_sub='Bulunsky',sub_pos='rt', col_bounds=col_bounds)
            poly_asia_e2 = np.array([(125,57),(125,70.5),(153.8,70.5),(148,57),(125,57)])
            only_shade([-0.71,-1.142,2,2],[125,148,58,72],polygon=poly_asia_e2,
                       label='tmp_NAE2')
            only_shade([-0.517,-1.035,2,2],[53,70,62,69.8],label='tmp_NAW')
            
            add_inset(fig,[-179.99,179.99,-89.99,89.99],[-0.575,-1.109,2,2],bounds=[87,112,68,78.5],
                      label='North Asia North W',markup_sub='North Siberia',sub_pos='rb', col_bounds=col_bounds)
            
            add_inset(fig,[-179.99,179.99,-89.99,89.99],[-0.71,-1.137,2,2],bounds=[125,148,54,68],polygon=poly_asia_e2,
                      label='North Asia East 2',markup_sub='Cherskiy and\nSuntar Khayata',sub_pos='lb',shades=False, col_bounds=col_bounds)
            
            poly_asia = np.array([(148,49),(160,64),(178,64),(170,55),(160,49),(148,49)])
            add_inset(fig,[-179.99,179.99,-89.99,89.99],[-0.823,-1.22,2,2],bounds=[127,179.9,50,64.8],
                      label='North Asia East',polygon=poly_asia,markup_sub='Kamchatka Krai',sub_pos='lb', col_bounds=col_bounds)
        
            
            add_inset(fig,[-179.99,179.99,-89.99,89.99],[-0.75,-1.01,2,2],bounds=[82,120,45.5,58.9],
                      label='South Asia North',markup='North Asia',markup_sub='Altay and Sayan',sub_pos='rb',markadj=0, 
                      col_bounds=col_bounds)
            add_inset(fig,[-179.99,179.99,-89.99,89.99],[-0.525,-1.045,2,2],bounds=[53,68,62,68.5],
                      label='North Asia West',markup_sub='Ural',sub_pos='rb',shades=False, col_bounds=col_bounds)
        
        if 13 in regions or 14 in regions or 15 in regions:
            add_inset(fig,[-179.99,179.99,-89.99,89.99],[-0.685,-1.065,2,2],bounds=[65, 105, 46.5, 25],
                      label='HMA',markup='High Mountain Asia',markadj=0, col_bounds=col_bounds)
        
        if 11 in regions:
            add_inset(fig,[-179.99,179.99,-89.99,89.99],[-0.58,-0.982,2,2],bounds=[-4.9,19,38.2,50.5],
                      label='Europe',markup='Central Europe',markadj=0, col_bounds=col_bounds)
        
        if 12 in regions:
            add_inset(fig,[-179.99,179.99,-89.99,89.99],[-0.66,-0.896,2,2],bounds=[38,54,29.6,43.6],
                      label='Middle East',markup='Caucasus and\nMiddle East',markadj=0, col_bounds=col_bounds)
        
        
        # ----- Circle sizes -----
#        axleg_background = fig.add_axes([0.001, 0.04, 0.107, 0.2])
        axleg_background = fig.add_axes([0.001, 0.04, 0.09, 0.53])
        axleg_background.get_yaxis().set_visible(False)
        axleg_background.get_xaxis().set_visible(False)
#        axleg_background.axis('off')
        rect1 = mpl.patches.Rectangle((0, 0), 1, 1, color ='white')
        axleg_background.add_patch(rect1)
        
        axleg = fig.add_axes([-0.92,-0.86,2,2],projection=ccrs.Robinson(),label='legend')
        axleg.set_extent([-179.99,179.99,-89.99,89.99], ccrs.Geodetic())
        axleg.outline_patch.set_linewidth(0)
        u=0
        rad_tot = 0
        for a in [10000, 1000, 100]:
            rad = (12000+np.sqrt(a)*1000)
            axleg.add_patch(mpatches.Circle(xy=[-900000,-680000+u*380000],radius=rad,edgecolor='k',label=str(a)+' km$^2$', transform = ccrs.Robinson(),fill=False, zorder=30))
            u=u+1
            rad_tot += rad
        axleg.text(-7.9, 2.4, '10$^{2}$\n10$^{3}$\n10$^{4}$', transform=ccrs.Geodetic(),horizontalalignment='left',verticalalignment='top',fontsize=9.5)
        axleg.text(-6.2, 9, 'Area', transform=ccrs.Geodetic(),horizontalalignment='center',verticalalignment='top',fontsize=9.5)
        axleg.text(-6.2, 6.2, '(km$^2$)', transform=ccrs.Geodetic(),horizontalalignment='center',verticalalignment='top',fontsize=8.5)
        
        
        # ----- Colorbar -----        
        cb = []
        cb_val = np.linspace(0, 1, len(col_bounds))
        for j in range(len(cb_val)):
            cb.append(mpl.cm.autumn_r(cb_val[j]))
        cmap_cus = mpl.colors.LinearSegmentedColormap.from_list('my_cb', list(
            zip((col_bounds - min(col_bounds)) / (max(col_bounds - min(col_bounds))), cb)), N=len(col_bounds))
        norm = mpl.colors.BoundaryNorm(col_bounds, cmap_cus.N)
        
        cax = fig.add_axes([0.045, 0.275, 0.007, 0.28], facecolor='none')
        sm = plt.cm.ScalarMappable(cmap=cmap_cus,norm=norm )
        cbar = plt.colorbar(sm, ticks=col_bounds, ax=ax, cax=cax, orientation='vertical')
        cax.xaxis.set_ticks_position('bottom')
        cax.xaxis.set_tick_params(pad=0)
        tick_labels = [x for x in col_bounds]
        tick_labels[5] = ''
        tick_labels[3] = ''
        tick_labels[1] = ''
        cbar.set_ticklabels(tick_labels)
        cbar.ax.tick_params(labelsize=9, pad=0.3)
        
#        # Use if running in command line
#        cax.text(-0.135, 0.215, 'Mass at 2100,\nrel. to 2015 (-)', size=10, horizontalalignment='center',
#                verticalalignment='bottom', rotation=90, transform=ax.transAxes)
        # Use if running in Spyder
        cax.text(-0.148, 0.205, 'Mass at 2100,\nrel. to 2015 (-)', size=10, horizontalalignment='center',
                 verticalalignment='bottom', rotation=90, transform=ax.transAxes)
    
        fig.savefig(fig_fp + 'global_deg_vol_remaining_dif_' + str(warming_group_2) + '-' + str(warming_group_1) + 'degC.png',dpi=250,transparent=True)
        
        
        
    #%% ----- PEAK WATER PLOTS -----
    col_bounds = np.array([2010, 2020, 2030, 2040, 2050, 2060, 2070, 2080, 2090, 2100])
    
    for nwarming_group, warming_group in enumerate(warming_groups):
        print(warming_group)

        normyear_idx = np.where(years == normyear)[0][0]
        
        # Area are in km2
        areas = ds_multigcm_area['all'][warming_group][:,normyear_idx] / 1e6
        areas_2100 = ds_multigcm_area['all'][warming_group][:,-1] / 1e6
        
        # ----- Peakwater calculation -----
        reg_mass = ds_multigcm_vol['all'][warming_group] * pygem_prms.density_ice
        # Record max mb
        reg_mb_gta = (reg_mass[:,1:] - reg_mass[:,0:-1]) / 1e12        
        reg_mb_gta_smoothed = uniform_filter(reg_mb_gta, size=(11))
        
        pw_yrs = []
        for nrow in np.arange(0,reg_mb_gta.shape[0]):
            pw_yrs.append(years[np.where(reg_mb_gta_smoothed[nrow,:] == reg_mb_gta_smoothed[nrow,:].min())[0][0]])
        pw_yrs = np.array(pw_yrs)
        
        # dh is the elevation change that is used for color; we'll use volume remaining by end of century for now
#        vols_remaining_frac = ds_multigcm_vol['all'][warming_group][:,-1] / ds_multigcm_vol['all'][warming_group][:,normyear_idx]
#        dhs = vols_remaining_frac.copy()
        dhs = pw_yrs
    
        areas = [area for _, area in sorted(zip(tiles_all,areas))]
        areas_2100 = [area for _, area in sorted(zip(tiles_all,areas_2100))]
        dhs = [dh for _, dh in sorted(zip(tiles_all,dhs))]
        tiles = sorted(tiles_all)
        
        
        def add_inset(fig,extent,position,bounds=None,label=None,polygon=None,shades=True, hillshade=True, list_shp=None, main=False, markup=None,markpos='left',markadj=0,markup_sub=None,sub_pos='lt',
                      col_bounds=None, color_water=color_water, color_land=color_land, add_rgi_glaciers=False):
            main_pos = [0.375, 0.21, 0.25, 0.25]
        
            if polygon is None and bounds is not None:
                polygon = poly_from_extent(bounds)
        
            if shades:
                shades_main_to_inset(main_pos, position, latlon_extent_to_robinson_axes_verts(polygon), label=label)
        
            sub_ax = fig.add_axes(position,
                                  projection=ccrs.Robinson(),label=label)
            sub_ax.set_extent(extent, ccrs.Geodetic())
        
            sub_ax.add_feature(cfeature.NaturalEarthFeature('physical', 'ocean', '50m', facecolor=color_water))
            sub_ax.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '50m', facecolor=color_land))
            
            # Add RGI glacier outlines
            if add_rgi_glaciers:
                shape_feature = ShapelyFeature(Reader(rgi_shp_fn).geometries(), ccrs.Robinson(),alpha=1,facecolor='indigo',linewidth=0.35,edgecolor='indigo')
                sub_ax.add_feature(shape_feature)
        
            if bounds is not None:
                verts = mpath.Path(latlon_extent_to_robinson_axes_verts(polygon))
                sub_ax.set_boundary(verts, transform=sub_ax.transAxes)
        
            # HERE IS WHERE VALUES APPEAR TO BE PROVIDED
            if not main:
                print(label)
                for i in range(len(tiles)):
                    lon = tiles[i][0]
                    lat = tiles[i][1]
        
                    if label=='Arctic West' and ((lat < 71 and lon > 60) or (lat <76 and lon>100)):
                        continue
        
                    if label=='HMA' and lat >=46:
                        continue
        
                    # fac = 0.02
                    fac = 1000
                    
                    area_2100 = areas_2100[i]
        
                    if areas[i] > 10:
                        rad = 15000 + np.sqrt(areas[i]) * fac
                    else:
                        rad = 15000 + 10 * fac
                        
                    cb = []
                    cb_val = np.linspace(0, 1, len(col_bounds))
                    for j in range(len(cb_val)):
                        cb.append(mpl.cm.RdYlBu(cb_val[j]))
                    cmap_cus = mpl.colors.LinearSegmentedColormap.from_list('my_cb', list(
                        zip((col_bounds - min(col_bounds)) / (max(col_bounds - min(col_bounds))), cb)), N=len(col_bounds)-1)
                    
                    # xy = [lon,lat]
                    xy = coordXform(ccrs.PlateCarree(),ccrs.Robinson(),np.array([lon]),np.array([lat]))[0][0:2]
                    
                    # Less than percent threshold and area threshold
                    dhdt = dhs[i]
                    pw_norm = (dhdt - min(col_bounds)) / (max(col_bounds - min(col_bounds)))
                    col = cmap_cus(pw_norm)
                    if area_2100 < 0.005:
                        sub_ax.add_patch(
                            mpatches.Circle(xy=xy, radius=rad, facecolor=col, edgecolor='black', linewidth=0.5, alpha=1, transform=ccrs.Robinson(), zorder=30))
                    else:
                        sub_ax.add_patch(
                            mpatches.Circle(xy=xy, radius=rad, facecolor=col, edgecolor='None', alpha=1, transform=ccrs.Robinson(), zorder=30))
            
            if markup is not None:
                if markpos=='left':
                    lon_upleft = np.min(list(zip(*polygon))[0])
                    lat_upleft = np.max(list(zip(*polygon))[1])
                else:
                    lon_upleft = np.max(list(zip(*polygon))[0])
                    lat_upleft = np.max(list(zip(*polygon))[1])
        
                robin = coordXform(ccrs.PlateCarree(),ccrs.Robinson(),np.array([lon_upleft]),np.array([lat_upleft]))
        
                rob_x = robin[0][0]
                rob_y = robin[0][1]
        
                size_y = 200000
                size_x = 80000 * len(markup) + markadj
        
                if markpos=='right':
                    rob_x = rob_x-50000
                else:
                    rob_x = rob_x+50000
        
                sub_ax_2 = fig.add_axes(position,
                                        projection=ccrs.Robinson(), label=label+'markup')
        
                # adds the white box to the region
#                sub_ax_2.add_patch(mpatches.Rectangle((rob_x, rob_y), size_x, size_y , linewidth=1, edgecolor='grey', facecolor='white',transform=ccrs.Robinson()))
        
                sub_ax_2.set_extent(extent, ccrs.Geodetic())
                verts = mpath.Path(rect_units_to_verts([rob_x,rob_y,size_x,size_y]))
                sub_ax_2.set_boundary(verts, transform=sub_ax.transAxes)
        
                sub_ax_2.text(rob_x,rob_y+50000,markup,
                         horizontalalignment=markpos, verticalalignment='bottom',
                         transform=ccrs.Robinson(), color='black',fontsize=4.5, fontweight='bold',bbox= dict(facecolor='white', alpha=1,linewidth=0.35,pad=1.5))
        
            if markup_sub is not None:
        
                lon_min = np.min(list(zip(*polygon))[0])
                lon_max = np.max(list(zip(*polygon))[0])
                lon_mid = 0.5*(lon_min+lon_max)
        
                lat_min = np.min(list(zip(*polygon))[1])
                lat_max = np.max(list(zip(*polygon))[1])
                lat_mid = 0.5*(lat_min+lat_max)
        
                size_y = 150000
                size_x = 150000
        
                lat_midup = lat_min+0.87*(lat_max-lat_min)
        
                robin = coordXform(ccrs.PlateCarree(),ccrs.Robinson(),np.array([lon_min,lon_min,lon_min,lon_mid,lon_mid,lon_max,lon_max,lon_max,lon_min]),np.array([lat_min,lat_mid,lat_max,lat_min,lat_max,lat_min,lat_mid,lat_max,lat_midup]))
        
                if sub_pos=='lb':
                    rob_x = robin[0][0]
                    rob_y = robin[0][1]
                    ha='left'
                    va='bottom'
                elif sub_pos=='lm':
                    rob_x = robin[1][0]
                    rob_y = robin[1][1]
                    ha='left'
                    va='center'
                elif sub_pos == 'lm2':
                    rob_x = robin[8][0]
                    rob_y = robin[8][1]
                    ha = 'left'
                    va = 'center'
                elif sub_pos=='lt':
                    rob_x = robin[2][0]
                    rob_y = robin[2][1]
                    ha='left'
                    va='top'
                elif sub_pos=='mb':
                    rob_x = robin[3][0]
                    rob_y = robin[3][1]
                    ha='center'
                    va='bottom'
                elif sub_pos=='mt':
                    rob_x = robin[4][0]
                    rob_y = robin[4][1]
                    ha='center'
                    va='top'
                elif sub_pos=='rb':
                    rob_x = robin[5][0]
                    rob_y = robin[5][1]
                    ha='right'
                    va='bottom'
                elif sub_pos=='rm':
                    rob_x = robin[6][0]
                    rob_y = robin[6][1]
                    ha='right'
                    va='center'
                elif sub_pos=='rt':
                    rob_x = robin[7][0]
                    rob_y = robin[7][1]
                    ha='right'
                    va='top'
        
                if sub_pos[0] == 'r':
                    rob_x = rob_x - 50000
                elif sub_pos[0] == 'l':
                    rob_x = rob_x + 50000
        
                if sub_pos[1] == 'b':
                    rob_y = rob_y + 50000
                elif sub_pos[1] == 't':
                    rob_y = rob_y - 50000
        
                sub_ax_3 = fig.add_axes(position,
                                        projection=ccrs.Robinson(), label=label+'markup2')
        
                # sub_ax_3.add_patch(mpatches.Rectangle((rob_x, rob_y), size_x, size_y , linewidth=1, edgecolor='grey', facecolor='white',transform=ccrs.Robinson()))
        
                sub_ax_3.set_extent(extent, ccrs.Geodetic())
                verts = mpath.Path(rect_units_to_verts([rob_x,rob_y,size_x,size_y]))
                sub_ax_3.set_boundary(verts, transform=sub_ax.transAxes)
        
                sub_ax_3.text(rob_x,rob_y,markup_sub,
                         horizontalalignment=ha, verticalalignment=va,
                         transform=ccrs.Robinson(), color='black',fontsize=4.5,bbox=dict(facecolor='white', alpha=1,linewidth=0.35,pad=1.5),fontweight='bold',zorder=25)
        
            if not main:
        #        sub_ax.outline_patch.set_edgecolor('white')
#                sub_ax.spines['geo'].set_edgecolor('white')
                sub_ax.spines['geo'].set_edgecolor('k')
            else:
        #        sub_ax.outline_patch.set_edgecolor('lightgrey')
                sub_ax.spines['geo'].set_edgecolor('lightgrey')
        
    
        #TODO: careful here! figure size determines everything else, found no way to do it otherwise in cartopy
        fig_width_inch=7.2
        fig = plt.figure(figsize=(fig_width_inch,fig_width_inch/1.9716))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
    #    ax = fig.add_axes([0,0.12,1,0.88], projection=ccrs.Robinson())
        
        ax.set_global()
        ax.spines['geo'].set_linewidth(0)
        
        add_inset(fig,[-179.99,179.99,-89.99,89.99],[0.375, 0.21, 0.25, 0.25],bounds=[-179.99,179.99,-89.99,89.99],shades=False, hillshade = False, main=True, col_bounds=col_bounds,
                  color_water='lightblue', color_land='white', add_rgi_glaciers=add_rgi_glaciers
                  )
#        #add_inset(fig,[-179.99,179.99,-89.99,89.99],[0.375, 0.21, 0.25, 0.25],bounds=[-179.99,179.99,-89.99,89.99],shades=False, hillshade = False, main=True, list_shp=shp_buff)
        
        if 19 in regions:
            poly_aw = np.array([(-158,-79),(-135,-60),(-110,-60),(-50,-60),(-50,-79.25),(-158,-79.25)])
            add_inset(fig,[-179.99,179.99,-89.99,89.99],[-0.4,-0.065,2,2],bounds=[-158, -45, -40, -79],
                      label='Antarctic_West', polygon=poly_aw,shades=True,markup_sub='West and Peninsula',sub_pos='mb', col_bounds=col_bounds)
            poly_ae = np.array([(135,-81.5),(152,-63.7),(165,-65),(175,-70),(175,-81.25),(135,-81.75)])
            add_inset(fig,[-179.99,179.99,-89.99,89.99],[-0.71,-0.045,2,2],bounds=[130, 175, -64.5, -81],
                      label='Antarctic_East', polygon=poly_ae,shades=True,markup_sub='East 2',sub_pos='mb', col_bounds=col_bounds)
            
            poly_ac = np.array([(-25,-62),(106,-62),(80,-79.25),(-25,-79.25),(-25,-62)])
            add_inset(fig,[-179.99,179.99,-89.99,89.99],[-0.52,-0.065,2,2],bounds=[-25, 106, -62.5, -79],
                      label='Antarctic_Center',polygon=poly_ac,shades=True,markup='Antarctic and Subantarctic',
                      markpos='right',markadj=0,markup_sub='East 1',sub_pos='mb', col_bounds=col_bounds)
            
            add_inset(fig,[-179.99,179.99,-89.99,89.99],[-0.68,-0.18,2,2],bounds=[64, 78, -48, -56],
                      label='Antarctic_Australes', shades=True,markup_sub='Kerguelen and Heard Islands',sub_pos='lb', col_bounds=col_bounds)
            
            add_inset(fig,[-179.99,179.99,-89.99,89.99],[-0.42,-0.143,2,2],bounds=[-40, -23, -53, -62],
                      label='Antarctic_South_Georgia', shades=True,markup_sub='South Georgia and Central Islands',sub_pos='lb', col_bounds=col_bounds)
        
        if 16 in regions or 17 in regions:
            add_inset(fig, [-179.99,179.99,-89.99,89.99], [-0.52, -0.225, 2, 2],bounds=[-82,-65,13,-57],label='Andes',
                      markup='Low Latitudes &\nSouthern Andes',markadj=0,
#                      markup_sub='a',
                      sub_pos='lm2', col_bounds=col_bounds)
            add_inset(fig, [-179.99,179.99,-89.99,89.99], [-0.352, -0.38, 2, 2],bounds=[-100,-95,22,16],label='Mexico',
                      markup_sub='Mexico',sub_pos='rb', col_bounds=col_bounds)
            add_inset(fig, [-179.99,179.99,-89.99,89.99], [-1.078, -0.22, 2, 2],bounds=[28,42,2,-6],label='Africa',
                      markup_sub='East Africa',sub_pos='rb', col_bounds=col_bounds)
            add_inset(fig, [-179.99,179.99,-89.99,89.99], [-1.64, -0.3, 2, 2],bounds=[133,140,-2,-7],label='Indonesia',
                      markup_sub='New Guinea',sub_pos='lb', col_bounds=col_bounds)
        
        if 3 in regions or 4 in regions or 5 in regions or 6 in regions or 7 in regions or 8 in regions or 9 in regions: 
            poly_arctic = np.array([(-105,84.5),(115,84.5),(110,68),(30,68),(18,57),(-70,57),(-100,75),(-105,84.5)])
            add_inset(fig,[-179.99,179.99,-89.99,89.99],[-0.48,-1.003,2,2],bounds=[-100, 106, 57, 84],label='Arctic West',
                      polygon=poly_arctic,markup='Arctic',markadj=0, col_bounds=col_bounds)
            
        if 18 in regions:
            add_inset(fig,[-179.99,179.99,-89.99,89.99],[-0.92,-0.17,2,2],bounds=[164,176,-47,-40],label='New Zealand',
                      markup='New Zealand',markpos='right',markadj=0, col_bounds=col_bounds)
        
        if 1 in regions or 2 in regions:
            poly_na = np.array([(-170,72),(-140,72),(-120,63),(-101,35),(-126,35),(-165,55),(-170,72)])
            add_inset(fig,[-179.99,179.99,-89.99,89.99],[-0.1,-1.22,2,2],bounds=[-177,-105, 36, 70],label='North America',
                      polygon=poly_na,markup='Alaska & Western\nCanada and US',markadj=0, col_bounds=col_bounds)
        
        if 10 in regions:
            
#            poly_asia_ne = np.array([(142,71),(142,82),(163,82),(155,71),(142,71)])
#            add_inset(fig,[-179.99,179.99,-89.99,89.99],[-0.64,-1.165,2,2],bounds=[142,160,71,80],
#                      polygon=poly_asia_ne,label='North Asia North E',markup_sub='Bulunsky',sub_pos='rt', col_bounds=col_bounds)
            poly_asia_e2 = np.array([(125,57),(125,70.5),(153.8,70.5),(148,57),(125,57)])
            only_shade([-0.71,-1.142,2,2],[125,148,58,72],polygon=poly_asia_e2,
                       label='tmp_NAE2')
            only_shade([-0.517,-1.035,2,2],[53,70,62,69.8],label='tmp_NAW')
            
            add_inset(fig,[-179.99,179.99,-89.99,89.99],[-0.575,-1.109,2,2],bounds=[87,112,68,78.5],
                      label='North Asia North W',markup_sub='North Siberia',sub_pos='rb', col_bounds=col_bounds)
            
            add_inset(fig,[-179.99,179.99,-89.99,89.99],[-0.71,-1.137,2,2],bounds=[125,148,54,68],polygon=poly_asia_e2,
                      label='North Asia East 2',markup_sub='Cherskiy and\nSuntar Khayata',sub_pos='lb',shades=False, col_bounds=col_bounds)
            
            poly_asia = np.array([(148,49),(160,64),(178,64),(170,55),(160,49),(148,49)])
            add_inset(fig,[-179.99,179.99,-89.99,89.99],[-0.823,-1.22,2,2],bounds=[127,179.9,50,64.8],
                      label='North Asia East',polygon=poly_asia,markup_sub='Kamchatka Krai',sub_pos='lb', col_bounds=col_bounds)
        
            
            add_inset(fig,[-179.99,179.99,-89.99,89.99],[-0.75,-1.01,2,2],bounds=[82,120,45.5,58.9],
                      label='South Asia North',markup='North Asia',markup_sub='Altay and Sayan',sub_pos='rb',markadj=0, 
                      col_bounds=col_bounds)
            add_inset(fig,[-179.99,179.99,-89.99,89.99],[-0.525,-1.045,2,2],bounds=[53,68,62,68.5],
                      label='North Asia West',markup_sub='Ural',sub_pos='rb',shades=False, col_bounds=col_bounds)
        
        if 13 in regions or 14 in regions or 15 in regions:
            add_inset(fig,[-179.99,179.99,-89.99,89.99],[-0.685,-1.065,2,2],bounds=[65, 105, 46.5, 25],
                      label='HMA',markup='High Mountain Asia',markadj=0, col_bounds=col_bounds)
        
        if 11 in regions:
            add_inset(fig,[-179.99,179.99,-89.99,89.99],[-0.58,-0.982,2,2],bounds=[-4.9,19,38.2,50.5],
                      label='Europe',markup='Central Europe',markadj=0, col_bounds=col_bounds)
        
        if 12 in regions:
            add_inset(fig,[-179.99,179.99,-89.99,89.99],[-0.66,-0.896,2,2],bounds=[38,54,29.6,43.6],
                      label='Middle East',markup='Caucasus and\nMiddle East',markadj=0, col_bounds=col_bounds)
        
        # ----- Circle sizes -----
#        axleg_background = fig.add_axes([0.001, 0.04, 0.107, 0.2])
        axleg_background = fig.add_axes([0.001, 0.04, 0.09, 0.53])
        axleg_background.get_yaxis().set_visible(False)
        axleg_background.get_xaxis().set_visible(False)
#        axleg_background.axis('off')
        rect1 = mpl.patches.Rectangle((0, 0), 1, 1, color ='white')
        axleg_background.add_patch(rect1)
        
        axleg = fig.add_axes([-0.92,-0.86,2,2],projection=ccrs.Robinson(),label='legend')
        axleg.set_extent([-179.99,179.99,-89.99,89.99], ccrs.Geodetic())
        axleg.outline_patch.set_linewidth(0)
        u=0
        rad_tot = 0
        for a in [10000, 1000, 100]:
            rad = (12000+np.sqrt(a)*1000)
            axleg.add_patch(mpatches.Circle(xy=[-900000,-680000+u*380000],radius=rad,edgecolor='k',label=str(a)+' km$^2$', transform = ccrs.Robinson(),fill=False, zorder=30))
            u=u+1
            rad_tot += rad
        axleg.text(-7.9, 2.4, '10$^{2}$\n10$^{3}$\n10$^{4}$', transform=ccrs.Geodetic(),horizontalalignment='left',verticalalignment='top',fontsize=10)
        axleg.text(-6.2, 9, 'Area', transform=ccrs.Geodetic(),horizontalalignment='center',verticalalignment='top',fontsize=10)
        axleg.text(-6.2, 6.2, '(km$^2$)', transform=ccrs.Geodetic(),horizontalalignment='center',verticalalignment='top',fontsize=9)
        
        # ----- Colorbar -----
        cax = fig.add_axes([0.028, 0.265, 0.007, 0.28], facecolor='none')
        cb = []
        cb_val = np.linspace(0, 1, len(col_bounds))
        for j in range(len(cb_val)):
            cb.append(mpl.cm.RdYlBu(cb_val[j]))
        cmap_cus = mpl.colors.LinearSegmentedColormap.from_list('my_cb', list(
            zip((col_bounds - min(col_bounds)) / (max(col_bounds - min(col_bounds))), cb)), N=len(col_bounds)-1)
        # norm = mpl.colors.BoundaryNorm(boundaries=col_bounds, ncolors=256)
        norm = mpl.colors.Normalize(vmin=np.min(col_bounds),vmax=np.max(col_bounds))
        sm = plt.cm.ScalarMappable(cmap=cmap_cus, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, cax=cax, ticks=col_bounds, orientation='vertical',extend='both')

        cax.xaxis.set_ticks_position('bottom')
        cax.xaxis.set_tick_params(pad=0)
        tick_labels = [x for x in col_bounds]
        tick_labels[8] = ''
        tick_labels[6] = ''
        tick_labels[4] = ''
        tick_labels[2] = ''
        tick_labels[0] = ''
        cbar.set_ticklabels(tick_labels)
        cbar.ax.tick_params(labelsize=8)
        
#        cax.text(-0.147, 0.19, 'Peak Mass Loss Year', size=10, horizontalalignment='center',
#                verticalalignment='bottom', rotation=90, transform=ax.transAxes)
        # Switch if running in Spyder
        cax.text(-0.16, 0.17, 'Peak Mass Loss (yr)', size=8.5, horizontalalignment='center',
                verticalalignment='bottom', rotation=90, transform=ax.transAxes)
    
        fig.savefig(fig_fp + 'global_deg_peakwater_' + str(warming_group) + 'degC.png',dpi=250,transparent=True)
        