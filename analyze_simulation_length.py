""" Analyze simulation output - mass change, runoff, etc. """

# Built-in libraries
#from collections import OrderedDict
#import datetime
#import glob
import os
#import pickle
# External libraries
#import cartopy
#import matplotlib as mpl
#import matplotlib.pyplot as plt
#from matplotlib.pyplot import MaxNLocator
#from matplotlib.lines import Line2D
#import matplotlib.patches as mpatches
#from matplotlib.ticker import MultipleLocator
#from matplotlib.ticker import EngFormatter
#from matplotlib.ticker import StrMethodFormatter
#from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import numpy as np
import pandas as pd
#from scipy.stats import linregress
#from scipy.ndimage import uniform_filter
#import scipy
#from scipy import stats
#from scipy.stats.kde import gaussian_kde
#from scipy.stats import norm
#from scipy.stats import truncnorm
#from scipy.stats import uniform
#from scipy.stats import linregress
#from scipy.stats import lognorm
#from scipy.optimize import minimize
from scipy.stats import median_abs_deviation
import xarray as xr
# Local libraries
#import class_climate
#import class_mbdata
#import pygem.pygem_input as pygem_prms
#import pygemfxns_gcmbiasadj as gcmbiasadj
#import pygemfxns_massbalance as massbalance
#import pygemfxns_modelsetup as modelsetup
#import run_calibration as calibration


#%% ===== Input data =====
netcdf_fp_cmip5 = '/Users/drounce/Documents/HiMAT/spc_backup/simulations_100/'
#netcdf_fp_era = pygem_prms.output_filepath + 'simulations/spc_20190914/merged/ERA-Interim/'

#%%
regions = [1]
# GCMs and RCP scenarios
gcm_names = ['CCSM4']
#rcps = ['rcp26', 'rcp45', 'rcp85']
rcps = ['rcp26']

# Grouping
grouping = 'rgi_region'

for reg in regions:
    for gcm_name in gcm_names:
        for rcp in rcps:
            netcdf_fp = netcdf_fp_cmip5 + gcm_name + '/' + rcp + '/' + 'essential/'
            
            glac_fns = []
            for i in os.listdir(netcdf_fp):
                if i.endswith('.nc'):
                    glac_fns.append(i)
            
            glac_fns = sorted(glac_fns)
            
#            for glac_fn in glac_fns[0:1]:
            for glac_fn in [glac_fns[15519]]:
                ds = xr.open_dataset(netcdf_fp + glac_fn)
                
                # Volume at end of model run
                glac_vol_final = ds.glac_volume_annual.values[0,-1,:]
                # statistics
                glac_vol_final_med = np.median(glac_vol_final)
                glac_vol_final_mad = median_abs_deviation(glac_vol_final)
                
                # Bootstrap method for different lengthed lists
                nsims_list = [100,90,80,70,60,50,40,30,20,10]
                n_iters = 1000
                
                df_columns = ['nsims', 'mean of meds', 'std of meds', 'std of meds [%]', 
                              'mean of mads', 'std of mads', 'std of mads [%]']
                df = pd.DataFrame(np.zeros((len(nsims_list), len(df_columns))), columns=df_columns)
                
                for n, nsims in enumerate(nsims_list):
                    glac_vol_final_meds = []
                    glac_vol_final_mads = []
                    for ncount in np.arange(0,n_iters):
                        rand_idx = np.random.randint(0,glac_vol_final.shape[0],size=nsims)
                        glac_vol_final_sample = glac_vol_final[rand_idx]
                        glac_vol_final_sample_med = np.median(glac_vol_final_sample)
                        glac_vol_final_meds.append(glac_vol_final_sample_med)
                        
                        glac_vol_final_sample_mad = median_abs_deviation(glac_vol_final_sample)
                        glac_vol_final_mads.append(glac_vol_final_sample_mad)
                    
                    
                    glac_vol_final_meds = np.array(glac_vol_final_meds)
                    glac_vol_final_meds_std = glac_vol_final_meds.std()
                    glac_vol_final_meds_mean = glac_vol_final_meds.mean()
                    
                    glac_vol_final_mads = np.array(glac_vol_final_mads)
                    glac_vol_final_mads_std = glac_vol_final_mads.std()
                    glac_vol_final_mads_mean = glac_vol_final_mads.mean()
                    
                    df.loc[n,:] = [nsims, glac_vol_final_meds_mean, glac_vol_final_meds_std,
                                   glac_vol_final_meds_std/glac_vol_final_meds_mean*100,
                                   glac_vol_final_mads_mean, glac_vol_final_mads_std,
                                   glac_vol_final_mads_std/glac_vol_final_mads_mean*100]
                    print('nsims:', nsims, 'std of medians [%]:',
                          np.round(glac_vol_final_meds_std/glac_vol_final_meds_mean*100,1),'%')

                    
                