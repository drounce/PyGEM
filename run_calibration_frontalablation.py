"""
Calibrate frontal ablation parameters for tidewater glaciers

@author: davidrounce
"""
# Built-in libraries
import argparse
import os
# External libraries
import pandas as pd
import numpy as np
#from scipy.stats import median_abs_deviation
# Local libraries
#import class_climate
import pygem.pygem_input as pygem_prms
import pygemfxns_modelsetup as modelsetup
from pygem.oggm_compat import single_flowline_glacier_directory


#%% ----- MANUAL INPUT DATA -----
regions = ['01']
elev_water = 0      # m asl
k_init = 1          # frontal ablation calibration parameter (yr-1)
height_limit = 20   # m


#%% ----- CALIBRATE FRONTAL ABLATION -----
# Load calving glacier data
fa_glacier_data = pd.read_csv(pygem_prms.frontalablation_glacier_data_fullfn)
fa_glacier_rgiids = list(fa_glacier_data.RGIId.values)
fa_regional_data = pd.read_csv(pygem_prms.frontalablation_regional_data_fullfn)

for reg in regions:
    # Load all glaciers
    main_glac_rgi_all = modelsetup.selectglaciersrgitable(
            rgi_regionsO1=[int(reg)], rgi_regionsO2='all', rgi_glac_number='all')
    # Tidewater glaciers
    main_glac_rgi = main_glac_rgi_all.loc[main_glac_rgi_all['TermType'] == 1]
    main_glac_rgi.reset_index(inplace=True, drop=True)
    
    #%%
    # Calibrate individual glaciers
    for glac in range(main_glac_rgi.shape[0]):
#    for glac in [1]:
        rgiid = main_glac_rgi.loc[glac,'RGIId']
        if rgiid in fa_glacier_rgiids:
            fa_idx = fa_glacier_rgiids.index(rgiid)
            
            print('\n\n', glac, rgiid, 'has data')
            
#            base_url = pygem_prms.oggm_base_url
#            gdir = workflow.init_glacier_directories([rgiid], from_prepro_level=2, prepro_border=40, 
#                                                      prepro_base_url=base_url, prepro_rgi_version='62')

            glacier_str = '{0:0.5f}'.format(main_glac_rgi.loc[glac,'RGIId_float'])
            
            print(glacier_str)
            
            gdir = single_flowline_glacier_directory(glacier_str)
            
            fl_df = pd.read_csv(gdir.get_filepath('elevation_band_flowline', filesuffix='_fixed_dx'))
            thick = fl_df['consensus_ice_thickness'].values[-1]
            width = fl_df['widths_m'].values[-1]
            fa_gta_obs = fa_glacier_data.loc[fa_idx,'frontal_ablation_Gta']
            
            # Import flowlines to get the elevation
            fls = gdir.read_pickle('inversion_flowlines')
            elev = fls[0].surface_h[-1]
            
            # Calving law (Oerlemans and Nick 2005; Huss and Hock 2015; Recinos et al. 2019)
            #  q = max(0, k * water_depth * ice_thickness * width)
            #  where k is the calibrated parameter (yr-1)
            
            # Limit water elevation to 100 m (Bassis and Walker 2011 sources)
            if elev - elev_water > height_limit:
                elev_water = elev - height_limit
            # Water depth (m)
            water_depth = thick - elev + elev_water
            
            # Frontal ablation (m3 yr-1)
            fa_m3a = k_init * water_depth * thick * width 
            
            print('  k              :', k_init)
            print('  elev (m)       :', np.round(elev,0))
            print('  elev water (m) :', np.round(elev_water,0))
            print('  water depth (m):', np.round(water_depth,0))
            print('  thickness (m)  :', np.round(thick,0))
            print('  width (m)      :', np.round(width,0))
            print('  fa (m3 yr-1)   :', fa_m3a)
            print('  fa_obs (gta)   :', fa_gta_obs)
            
            fa_gta = fa_m3a * pygem_prms.density_ice / pygem_prms.density_water / 1e9
            
            k_opt = fa_gta_obs * pygem_prms.density_water / pygem_prms.density_ice / (water_depth * thick * width / 1e9)
            
            print('  k_opt (yr-1)   :', k_opt)
    