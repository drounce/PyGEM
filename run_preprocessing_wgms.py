"""
pygemfxns_preprocessing.py is a list of the model functions that are used to preprocess the data into the proper format.

"""

# Built-in libraries
import os
#import glob
import argparse
# External libraries
import pandas as pd
import numpy as np
#import xarray as xr
#import netCDF4 as nc
#from time import strftime
#from datetime import datetime
#from scipy.spatial.distance import cdist
#from scipy.optimize import minimize
#import matplotlib.pyplot as plt
# Local libraries
import pygem.pygem_input as pygem_prms
import pygemfxns_modelsetup as modelsetup

#%% TO-DO LIST:
# - clean up create lapse rate input data (put it all in pygem_prms.py)

#%%
def getparser():
    """
    Use argparse to add arguments from the command line
    
    Parameters
    ----------
    option_createlapserates : int
        Switch for processing lapse rates (default = 0 (no))
    option_wgms : int
        Switch for processing wgms data (default = 0 (no))
        
    Returns
    -------
    Object containing arguments and their respective values.
    """
    parser = argparse.ArgumentParser(description="select pre-processing options")
    # add arguments
    parser.add_argument('-option_wgms', action='store', type=int, default=1,
                        help='option to pre-process wgms data (1=yes, 0=no)')
    return parser

parser = getparser()
args = parser.parse_args()

#%%
#rgi_regionsO1 = [13,14,15]
#main_glac_rgi_all = pd.DataFrame()
#for region in rgi_regionsO1:
#    main_glac_rgi_region = modelsetup.selectglaciersrgitable(rgi_regionsO1=[region], rgi_regionsO2='all', 
#                                                             rgi_glac_number='all')
#    main_glac_rgi_all = main_glac_rgi_all.append(main_glac_rgi_region)


#%%
    
if args.option_wgms == 1:
    
    wgms_fp = '/Users/davidrounce/Documents/Dave_Rounce/HiMAT/WGMS/DOI-WGMS-FoG-2019-12/'
    wgms_ee_fn = 'WGMS-FoG-2019-12-EE-MASS-BALANCE.csv'
    wgms_e_fn = 'WGMS-FoG-2019-12-E-MASS-BALANCE-OVERVIEW.csv'
    
    wgms_id_fn = 'WGMS-FoG-2019-12-AA-GLACIER-ID-LUT.csv'
    
    wgms_e_df = pd.read_csv(wgms_fp + wgms_e_fn, encoding='unicode_escape')
    wgms_ee_df = pd.read_csv(wgms_fp + wgms_ee_fn, encoding='unicode_escape')
    wgms_id_df = pd.read_csv(wgms_fp + wgms_id_fn, encoding='unicode_escape')
    
    wgms_id_dict = dict(zip(wgms_id_df.WGMS_ID, wgms_id_df.RGI_ID))
    
    # Map dictionary
    wgms_ee_df['rgiid_raw'] = wgms_ee_df.WGMS_ID.map(wgms_id_dict)
    wgms_ee_df = wgms_ee_df.dropna(subset=['rgiid_raw'])
    
    # Link RGIv5.0 with RGIv6.0
    rgi60_fp = '/Users/davidrounce/Documents/Dave_Rounce/HiMAT/RGI/rgi60/00_rgi60_attribs/'
    rgi50_fp = '/Users/davidrounce/Documents/Dave_Rounce/HiMAT/RGI/00_rgi50_attribs/'
    
#    regions_str = ['01']
    regions_str = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18']
    rgi60_df = None
    rgi50_df = None
    for reg_str in regions_str:
        # RGI60 data
        for i in os.listdir(rgi60_fp):
            if i.startswith(reg_str) and i.endswith('.csv'):
                rgi60_df_reg = pd.read_csv(rgi60_fp + i, encoding='unicode_escape')
        # append datasets
        if rgi60_df is None:
            rgi60_df = rgi60_df_reg
        else:
            rgi60_df = pd.concat([rgi60_df, rgi60_df_reg], axis=0)
    
        # RGI50 data
        for i in os.listdir(rgi50_fp):
            if i.startswith(reg_str) and i.endswith('.csv'):
                rgi50_df_reg = pd.read_csv(rgi50_fp + i, encoding='unicode_escape')
        # append datasets
        if rgi50_df is None:
            rgi50_df = rgi50_df_reg
        else:
            rgi50_df = pd.concat([rgi50_df, rgi50_df_reg], axis=0)
        
    # Merge based on GLIMSID
    glims_rgi50_dict = dict(zip(rgi50_df.GLIMSId, rgi50_df.RGIId))
    rgi60_df['RGIId_50'] = rgi60_df.GLIMSId.map(glims_rgi50_dict)
    rgi60_df_4dict = rgi60_df.dropna(subset=['RGIId_50'])
    rgi50_rgi60_dict = dict(zip(rgi60_df_4dict.RGIId_50, rgi60_df_4dict.RGIId))
    rgi60_self_dict = dict(zip(rgi60_df.RGIId, rgi60_df.RGIId))
    rgi50_rgi60_dict.update(rgi60_self_dict)
    
    # Add RGIId for version 6 to WGMS
    wgms_ee_df['rgiid'] = wgms_ee_df.rgiid_raw.map(rgi50_rgi60_dict)
    
    # Drop points without data
    wgms_ee_df_wrgiids = wgms_ee_df.dropna(subset=['rgiid'])

    # Winter balances only
    wgms_ee_df_wrgiids_winter = wgms_ee_df_wrgiids.dropna(subset=['WINTER_BALANCE'])
    wgms_ee_df_wrgiids_winter = wgms_ee_df_wrgiids_winter.sort_values('rgiid')
    wgms_ee_df_wrgiids_winter.reset_index(inplace=True, drop=True)
    wgms_ee_winter_fn = 'WGMS-FoG-2019-12-EE-MASS-BALANCE-winter_processed.csv'
    wgms_ee_df_wrgiids_winter.to_csv(wgms_fp + wgms_ee_winter_fn, index=False)
    
    # Add the winter time period using the E-MASS-BALANCE-OVERVIEW file
    wgms_e_cns2add = []
    for cn in wgms_e_df.columns:
        if cn not in wgms_ee_df_wrgiids_winter.columns:
            wgms_e_cns2add.append(cn)
            wgms_ee_df_wrgiids_winter[cn] = np.nan

            
    for nrow in np.arange(wgms_ee_df_wrgiids_winter.shape[0]):
        if nrow%500 == 0:
            print(nrow, 'of', wgms_ee_df_wrgiids_winter.shape[0])
        name = wgms_ee_df_wrgiids_winter.loc[nrow,'NAME']
        wgmsid = wgms_ee_df_wrgiids_winter.loc[nrow,'WGMS_ID']
        year = wgms_ee_df_wrgiids_winter.loc[nrow,'YEAR']
        
        try:
            e_idx = np.where((wgms_e_df['NAME'] == name) & 
                             (wgms_e_df['WGMS_ID'] == wgmsid) & 
                             (wgms_e_df['Year'] == year))[0][0]
        except:
            e_idx = None
        
        if e_idx is not None:
            wgms_ee_df_wrgiids_winter.loc[nrow,wgms_e_cns2add] = wgms_e_df.loc[e_idx,wgms_e_cns2add]
            
    #%%
            
    
