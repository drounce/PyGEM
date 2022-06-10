""" Analyze MCMC output - chain length, etc. """

# Built-in libraries
import os
import pickle
# External libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import median_abs_deviation
import xarray as xr
# Local libraries
import pygem.pygem_input as pygem_prms

#from oggm import cfg, utils
from pygem.oggm_compat import single_flowline_glacier_directory

#%%
regions = [1]

option_regional_mb = True

modelprms_fp = pygem_prms.output_filepath + 'calibration/'


#%%
if option_regional_mb:
    debug = True
    
    reg_df_fp = pygem_prms.output_filepath + 'analysis/'
    reg_df_fn = 'mcmc_reg_comparison.csv'
    
    for reg in regions:
        
        # regional filepath 
        modelprms_fp_reg = modelprms_fp + str(reg).zfill(2) + '/'
        
        # find calibrated glaciers
        glac_strs = []
        for i in os.listdir(modelprms_fp_reg):
            if i.endswith('-modelprms_dict.pkl'):
                glac_strs.append(i.split('-')[0])
        glac_strs = sorted(glac_strs)
        
        # Aggregate regional mass balance data
        reg_mb_m3wea_all = None
        count_glac = 0
        reg_area = 0
        for nglac, glacier_str in enumerate(glac_strs):
            
            if nglac%500 == 0:
                print(glacier_str)
            
            # Load calibration data
            try:
                gdir = single_flowline_glacier_directory(glacier_str, logging_level='CRITICAL')
                fls = gdir.read_pickle('inversion_flowlines')
                glacier_area = np.sum(fls[0].widths_m * fls[0].dx_meter)
                
                mbdata_fn = gdir.get_filepath('mb_obs')
                with open(mbdata_fn, 'rb') as f:
                    gdir.mbdata = pickle.load(f)
                    
                # Load data
                mb_mwea_obs = gdir.mbdata['mb_mwea']
                mb_mwea_err_obs = gdir.mbdata['mb_mwea_err']
                mb_m3wea_obs = mb_mwea_obs * glacier_area
            
            except:
                gdir = None
            
            # Load model parameters
            if gdir is not None:
                try:
                    modelprms_fn = glacier_str + '-modelprms_dict.pkl'
                    modelprms_fullfn = modelprms_fp_reg + modelprms_fn
                    with open(modelprms_fullfn, 'rb') as f:
                        modelprms_dict = pickle.load(f)

                    modelprms_all = modelprms_dict['MCMC']
                    mb_mwea_all = np.array(modelprms_all['mb_mwea']['chain_0'])
                    # convert to m3wea for regional aggregation
                    mb_m3wea_all = mb_mwea_all * glacier_area
                    
                    # add to regional mass balance
                    if reg_mb_m3wea_all is None:
                        reg_mb_m3wea_all = mb_m3wea_all
                        reg_mb_m3wea_obs = mb_m3wea_obs
                    else:
                        reg_mb_m3wea_all += mb_m3wea_all
                        reg_mb_m3wea_obs += mb_m3wea_obs
                    
                    # Track statistics
                    count_glac += 1
                    reg_area += glacier_area
                    
                except:
                    print('\n\nNo calibration data for ' + glacier_str)
            
        # convert back to mwea
        reg_mb_mwea = reg_mb_m3wea_all / reg_area
        reg_mb_mwea_obs = reg_mb_m3wea_obs / reg_area
        
        # Statistics for comparison
        reg_mb_mwea_mean = np.mean(reg_mb_mwea)
        reg_mb_mwea_std = np.std(reg_mb_mwea)
        reg_mb_mwea_med = np.median(reg_mb_mwea)
        reg_mb_mwea_mad = median_abs_deviation(reg_mb_mwea)
        reg_mb_mwea_perc5 = np.percentile(reg_mb_mwea, 5)
        reg_mb_mwea_perc25 = np.percentile(reg_mb_mwea, 25)
        reg_mb_mwea_perc75 = np.percentile(reg_mb_mwea, 75)
        reg_mb_mwea_perc95 = np.percentile(reg_mb_mwea, 95)
        
        print('reg_mb_mwea_med:', np.round(reg_mb_mwea_med,2), 'mb_mwea_obs:', np.round(reg_mb_mwea_obs,2))
        
        reg_df_cns = ['O1Region', 'count', 'area_m2', 'mb_mwea_mean', 'mb_mwea_std', 'mb_mwea_med', 
                      'mb_mwea_mad', 'mb_mwea_perc5', 'mb_mwea_perc25', 'mb_mwea_perc75', 'mb_mwea_perc95']
        reg_df_single = pd.DataFrame(np.zeros((1,len(reg_df_cns))), columns=reg_df_cns)
        reg_df_single.loc[0,:] = [reg, count_glac, reg_area, reg_mb_mwea_mean, reg_mb_mwea_std, 
                                  reg_mb_mwea_med, reg_mb_mwea_mad, reg_mb_mwea_perc5, reg_mb_mwea_perc25, 
                                  reg_mb_mwea_perc75, reg_mb_mwea_perc95]
        
        # Export csv
        if os.path.exists(reg_df_fp + reg_df_fn):
            reg_df = pd.read_csv(reg_df_fp + reg_df_fn)
            
            # Add or overwrite existing file
            reg_idx = np.where((reg_df.O1Region == reg))[0]
            if len(reg_idx) > 0:
                reg_df.loc[reg_idx,:] = reg_df_single.values
            else:
                reg_df = pd.concat([reg_df, reg_df_single], axis=0)
                
        else:
            reg_df = reg_df_single
            
        reg_df = reg_df.sort_values('O1Region', ascending=True)
        reg_df.reset_index(inplace=True, drop=True)
        reg_df.to_csv(reg_df_fp + reg_df_fn, index=False)