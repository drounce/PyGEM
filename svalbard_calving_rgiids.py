#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Add RGIIds to data from Blaszczyk et al. (2009)
"""
import numpy as np
import pandas as pd

import pygem.pygem_input as pygem_prms
import pygem.pygem_modelsetup as modelsetup

calving_data_fp = pygem_prms.main_directory + '/../calving_data/'
calving_data_fn = 'frontalablation_glacier_data_blaszczyk2009_4id.csv'
calving_data_fn_export = 'frontalablation_glacier_data_blaszczyk2009.csv'
rgi_region = [7]

df = pd.read_csv(calving_data_fp + calving_data_fn)

main_glac_rgi = modelsetup.selectglaciersrgitable(rgi_regionsO1=rgi_region, rgi_regionsO2='all', 
                                                  rgi_glac_number='all', 
                                                  rgi_cols_drop=[])

#%%
# Manually fix dataframe
df.loc[df.Name=='Stonebren','Name'] = 'Stonebreen'
df.loc[df.Name=='Au Torellbreen','Name'] = 'Austre Torellbreen'
df.loc[df.Name=='Brasvellbreen','Name'] = 'Braasvellbreen'
df.loc[df.Name=='Chauveaubreen','Name'] = 'Chaveauxbreen'
df.loc[df.Name=='Ericabreen','Name'] = 'Erikkabreen'
df.loc[df.Name=='Etonbreen','Name'] = 'Etonbreen: Austfonna'
main_glac_rgi.loc[main_glac_rgi.RGIId == 'RGI60-07.00714', 'Name'] = 'Gullybreen 1'
main_glac_rgi.loc[main_glac_rgi.RGIId == 'RGI60-07.00726', 'Name'] = 'Gullybreen 2'
#main_glac_rgi.loc[main_glac_rgi.RGIId == 'RGI60-07.01328', 'Name'] = 'Hamiltonbreen 2'  # unsure of this one
main_glac_rgi.loc[main_glac_rgi.RGIId == 'RGI60-07.01559', 'Name'] = 'Hinlopenbreen 1'
df.loc[df.Name=='Hochstatterbreen','Name'] = 'Hochstetterbreen'
df.loc[df.Name=='Hyrnerbreen','Name'] = 'Hyrnebreen'
main_glac_rgi.loc[main_glac_rgi.RGIId == 'RGI60-07.01464', 'Name'] = 'Kronebreen2'
df.loc[df.Name=='Liestolbreen','Name'] = 'Liestoelbreen'
main_glac_rgi.loc[main_glac_rgi.RGIId == 'RGI60-07.00990', 'Name'] = 'Loderbreen'
df.loc[df.Name=='Marstranbreen','Name'] = 'Marstrandbreen'
if df.loc[120,'Name'].startswith('Mittag'):
    df.loc[120,'Name'] = 'Mittag-Lefflerbreen'  # had to enter manually based on index
df.loc[df.Name=='Munthebreen','Name'] = 'Munthbreen'
if df.loc[41,'Name'].endswith('hlbacherbreen'):
    df.loc[41,'Name'] = 'Muehlbacherbreen'
df.loc[df.Name=='N Franklinbreen','Name'] = 'Franklinbreen N'
df.loc[df.Name=='Palanderbreen','Name'] = 'Palanderbreen: Vegafonna'
df.loc[df.Name=='S Franklinbreen','Name'] = 'Franklinbreen S'
main_glac_rgi.loc[main_glac_rgi.RGIId == 'RGI60-07.00263', 'Name'] = 'Samarinbreen'
df.loc[df.Name=='Smeerenburgbreen 1','Name'] = 'Smeerenburgbreen'
df.loc[df.Name=='Valhalfonna','Name'] = 'Valhallfonna E'

name_rgiid_dict = dict(zip(main_glac_rgi.Name, main_glac_rgi.RGIId))
rgiid_area_dict = dict(zip(main_glac_rgi.RGIId, main_glac_rgi.Area))

df['RGIId'] = df['Name'].map(name_rgiid_dict)
df['area_km2_rgi'] = df['RGIId'].map(rgiid_area_dict)


# Prepare for calving calibration
df_wrgiid = (df.dropna(axis=0, subset=['RGIId'])).copy()
df_wrgiid['Source'] = 'Blaszczyk et al. (2009), Tidewater glaciers of Svalbard: recent changes and estimates of calving fluxes'
# Blaszcyck et al. (2009) reported values in km3 w.e. yr-1, which is equal to Gt/yr
df_wrgiid['frontal_ablation_Gta'] = df['Qc_km3yr']
df_wrgiid['start_date'] = 20009999
df_wrgiid['end_date'] = 20069999
df_wrgiid['region_no'] = 7
df_wrgiid['frontal_ablation_unc_Gta'] = np.nan
df_wrgiid['glacier_name'] = df_wrgiid['Name']

df_export = df_wrgiid[['region_no', 'RGIId', 'glacier_name', 'frontal_ablation_Gta', 'frontal_ablation_unc_Gta', 
                       'start_date', 'end_date', 'Source']]
df_export.to_csv(calving_data_fp + calving_data_fn_export, index=False)