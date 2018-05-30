"""
class of mass balance data and functions associated with manipulating the dataset to be in the proper format
"""

import pandas as pd
import numpy as np

import pygem_input as input

class MBData():
    # GCM is the global climate model for a glacier evolution model run
    # When calling ERA-Interim, should be able to use defaults
    # When calling 
    def __init__(self, 
                 name='shean',
                 rgi_regionO1=input.rgi_regionsO1,
                 ,
                 ):
        
        # Source of climate data
        self.name = name
        # Set parameters for ERA-Interim and CMIP5 netcdf files
        if self.name == 'shean':
            self.data_fp = shean_fp
            self.data_fn = 'hma_mb_20171211_1343.csv'
            self.mb_colname = 'mb_mwea'
            self.mb_err_colname = 'mb_mwea_sigma'
            self.mb_time1_colname = 'year1'
            self.mb_time2_colname = 'year2'
            self.rgi_colname = 'RGIId'
            self.rgi_regionO1 = rgi_regionO1
            
        elif self.name == 'brun':
            self.data_fp = shean_fp,
            
            
            
            
            
## Mass balance tolerance [m w.e.a]
#massbal_tolerance = 0.1
## Calibration optimization tolerance
##cal_tolerance = 1e-4
            

        # Other options are currently all from standardized CMIP5 format
        elif self.name == 'brun':
            print('provide attributes to various csv file')
            
#    def importGCMfxnearestneighbor_xarray(self, filename, variablename, main_glac_rgi):
#        """
#        Import time invariant (constant) variables and extract the nearest neighbor of the variable. Meteorological data 
#        from the global climate models were provided by Ben Marzeion and ETH-Zurich for the GlacierMIP Phase II project 
#        or from ERA Interim (ECMWF), which uses geopotential instead of surface height.
#        
#        Output: Numpy array of nearest neighbor time series for all the glaciers in the model run
#        (rows = glaciers, column = variable time series)
#        """