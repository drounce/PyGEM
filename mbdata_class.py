"""
class of mass balance data and functions associated with manipulating the dataset to be in the proper format
"""

import pandas as pd
import numpy as np
import datetime

import pygem_input as input
import pygemfxns_modelsetup as modelsetup

class MBData():
    # GCM is the global climate model for a glacier evolution model run
    # When calling ERA-Interim, should be able to use defaults
    # When calling 
    def __init__(self, 
                 name='shean',
                 rgi_regionO1=input.rgi_regionsO1[0]
                 ):
        
        # Source of climate data
        self.name = name
        # Set parameters for ERA-Interim and CMIP5 netcdf files
        if self.name == 'shean': 
            self.ds_fp = input.shean_fp
            self.ds_fn = input.shean_fn
            self.rgi_regionO1 = rgi_regionO1
            self.rgi_cn = input.shean_rgi_colname
            self.mb_mwea_cn = input.shean_mb_colname
            self.mb_mwea_err_cn = input.shean_mb_err_colname
            self.t1_cn = input.shean_time1_colname
            self.t2_cn = input.shean_time2_colname
            self.area_cn = input.shean_area_colname
            self.mb_vol_cn = input.shean_vol_colname
            self.mb_vol_err_cn = input.shean_vol_err_colname
            
#        elif self.name == 'brun':
#            self.data_fp = shean_fp,
            
    def masschange_total(self, main_glac_rgi, gcm_startyear, gcm_endyear):
        """
        Calculate the total mass change for various datasets and output it in a format that is useful for calibration
        """       
        if self.name == 'shean':
            # Column names of output
            ds_cols = [input.rgi_O1Id_colname, 'mb_gt', 'mb_gt_err',  self.area_cn, self.t1_cn, self.t2_cn, 't1_idx', 
                       't2_idx']
            # Load all data
            ds_all = pd.read_csv(self.ds_fp + self.ds_fn)
            ds_all['RegO1'] = ds_all['RGIId'].values.astype(int)
            # Select data for specific region
            ds = ds_all[ds_all['RegO1']==self.rgi_regionO1].copy()
            # Glacier number for comparison
            ds[input.rgi_O1Id_colname] = ((ds[self.rgi_cn] % 1) * 10**5).round(0).astype(int)
            # Total mass change [Gt]
            ds['mb_gt'] = (ds_all[self.mb_vol_cn] * (ds_all[self.t2_cn] - ds_all[self.t1_cn]) * (1/1000)**3 * 
                           input.density_water / 1000)
            ds['mb_gt_err'] = (ds_all[self.mb_vol_err_cn] * (ds_all[self.t2_cn] - ds_all[self.t1_cn]) * (1/1000)**3 * 
                               input.density_water / 1000)
            # Determine dates_table_idx that coincides with data
            dates_table, start_date, end_date = modelsetup.datesmodelrun(gcm_startyear, gcm_endyear, spinupyears=0)
            #  ignore spinup years because modeled output used for comparison will not include them
            dates_table['year_decimal'] = dates_table['year'] + dates_table['month'] / 12
            ds['t1_idx'] = [(np.abs(dates_table['year_decimal'] - x)).argmin() for x in ds[self.t1_cn]]
            ds['t2_idx'] = [(np.abs(dates_table['year_decimal'] - x)).argmin() for x in ds[self.t2_cn]]
            # Subset of glaciers in standardized format
            ds_subset = ds[ds_cols]
            # Select glaciers and their mass change data
            rgi_glacno = main_glac_rgi[input.rgi_O1Id_colname].values
            # Mass balance data for each glacier
            main_glac_mbdata = pd.DataFrame(np.empty((main_glac_rgi.shape[0], len(ds_cols))), 
                                            index=main_glac_rgi.index.values, columns=ds_cols)
            for glac in range(rgi_glacno.shape[0]):
                if (np.in1d(ds[input.rgi_O1Id_colname],rgi_glacno[glac])==True).any():
                    main_glac_mbdata.iloc[glac,:] = (
                            ds_subset.iloc[np.where(np.in1d(ds[input.rgi_O1Id_colname],rgi_glacno[glac])==True)[0][0]])
        elif self.name == 'brun':
            print('code brun')
        return main_glac_mbdata


#%% Converting decimal years to month and year
#ds['t1_year'] = ds['t1'].values.astype(int)
#ds['t1_month'] = [(datetime.datetime(int(x),1,1) + datetime.timedelta((x - int(x))*365)).month for x in ds['t1']]   