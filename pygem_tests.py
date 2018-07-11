r"""
pygem_tests are tests to ensure the model is producing good results.
"""

#import pandas as pd
import numpy as np
#import os
#import argparse
#import inspect
#import subprocess as sp
#import multiprocessing
#from scipy.optimize import minimize
#import time
#from time import strftime

import pygem_input as input
import pygemfxns_modelsetup as modelsetup
#import pygemfxns_climate as climate
#import pygemfxns_massbalance as massbalance

#%% ===== TESTING DATA INPUT: GLACIER AREA, ICE THICKNESS, AND WIDTH =====
# Check that each glacier 
rgi_regionsO1 = [13, 14, 15]

with open(input.main_directory + '/../IceThickness_Huss/' + 'InputData_issues.txt', 'w+') as txtfile:
    txtfile.write('# General issues\n')
    txtfile.write('Matthias\' min elevation is 25 m for 10 m bands, so no glaciers at sea-level\n')
    txtfile.write('Matthias\' max elevation ~20 m more than max elevation from RGIId\n')
    txtfile.write('\nRegional issues')
    for regions in rgi_regionsO1:
        rgi_regionsO1 = [regions]
        txtfile.write('\n#Region' + str(regions) + '\n')
        # Glacier hypsometry [km**2], total area
        main_glac_rgi_all = modelsetup.selectglaciersrgitable(rgi_regionsO1=rgi_regionsO1, rgi_regionsO2 = 'all', 
                                                              rgi_glac_number='all')
        main_glac_hyps_all = modelsetup.import_Husstable(main_glac_rgi_all, rgi_regionsO1, input.hyps_filepath, 
                                                         input.hyps_filedict, input.hyps_colsdrop)
        main_glac_icethickness_all = modelsetup.import_Husstable(main_glac_rgi_all, rgi_regionsO1, 
                                                                 input.thickness_filepath, input.thickness_filedict, 
                                                                 input.thickness_colsdrop)
        main_glac_width_all = modelsetup.import_Husstable(main_glac_rgi_all, rgi_regionsO1, input.width_filepath, 
                                                          input.width_filedict, input.width_colsdrop)
        elev_bins = main_glac_hyps_all.columns.values.astype(int)
        
        for glac in range(main_glac_rgi_all.shape[0]):
            zmin = main_glac_rgi_all.loc[glac,'Zmin']
            zmax = main_glac_rgi_all.loc[glac,'Zmax']
            hyps = main_glac_hyps_all.loc[glac,:]
            ice = main_glac_icethickness_all.loc[glac,:]
            width = main_glac_width_all.loc[glac,:]
            if hyps.max() > 0:
                hyps_elev_min = elev_bins[np.where(hyps>0)[0][0]]
                hyps_elev_max = elev_bins[np.where(hyps>0)[0][-1]]
                if zmin > hyps_elev_min:
                    txtfile.write(str(main_glac_rgi_all.loc[glac,'RGIId'] + ' hyps elevation (' + str(hyps_elev_min) + 
                                      ') < min elevation (' + str(zmin) + ')\n'))
                if hyps_elev_max > zmax + 50:
                    txtfile.write(str(main_glac_rgi_all.loc[glac,'RGIId'] + ' hyps elevation (' + str(hyps_elev_max) + 
                                      ') > max elevation (' + str(zmax) + ')\n'))
            else:
                txtfile.write(str(main_glac_rgi_all.loc[glac,'RGIId'] + ' has no glacier area\n'))
            if ice.max() == 0:
                txtfile.write(str(main_glac_rgi_all.loc[glac,'RGIId'] + ' has no ice thickness\n'))
            if width.max() == 0:
                txtfile.write(str(main_glac_rgi_all.loc[glac,'RGIId'] + ' has no width\n'))
