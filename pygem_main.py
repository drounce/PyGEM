###############################################################################
"""
Python Glacier Evolution Model "PyGEM" V1.0
Prepared by David Rounce with support from Regine Hock.
This work was funded under the NASA HiMAT project (INSERT PROJECT #).

PyGEM is an open source glacier evolution model written in python.  Model details come from Radic et al. (2013), Bliss 
et al. (2014), and Huss and Hock (2015).
"""
#######################################################################################################################
# This is the main script that provides the architecture and framework for all of the model runs. All input data is 
# included in a separate module called pygem_input.py. It is recommended to not make any changes to this file unless
# you are a PyGEM developer and making changes to the model architecture.
#
# ========== IMPORT PACKAGES ==================================================
# Various packages are used to provide the proper architecture and framework for the calculations used in this script. 
# Some packages (e.g., datetime) are included in order to speed of calculations and simplify code.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os # os is used with re to find name matches
import re # see os
import xarray as xr
import netCDF4 as nc
from time import strftime
#import timeit

#========== IMPORT INPUT AND FUNCTIONS FROM MODULES ==========================
import pygem_input as input
import pygemfxns_modelsetup as modelsetup
import pygemfxns_climate as climate
import pygemfxns_massbalance as massbalance
import pygemfxns_output as output

#========== DEVELOPER'S TO-DO LIST =============================================
# > Output log file, i.e., file that states input parameters, date of model run, model options selected, 
#   and any errors that may have come up (e.g., precipitation corrected because negative value, etc.)


# ----- STEP ONE: Select glaciers included in model run ----------------------
if input.option_glacier_selection == 1:
    main_glac_rgi = modelsetup.selectglaciersrgitable()
elif input.option_glacier_selection == 2:
    print('\n\tMODEL ERROR (selectglaciersrgi): this option to use shapefiles to select glaciers has not been coded '
          '\n\tyet. Please choose an option that exists. Exiting model run.\n')
    exit()
else:
    # Should add options to make regions consistent with Brun et al. (2017), which used ASTER DEMs to get mass 
    # balance of 92% of the HMA glaciers.
    print('\n\tModel Error (selectglaciersrgi): please choose an option that exists for selecting glaciers.'
          '\n\tExiting model run.\n')
    exit()

#----- STEP TWO: Hypsometry, Ice thickness, Model time frame, Surface Type ---

""" DEVELOPER'S NOTE: importing hypsometry and ice thickness is very slow since it is adding bins beyond the max/min 
    This could be a preprocessing step as it will cause large delays
"""
# Glacier hypsometry [km**2]
main_glac_hyps = modelsetup.import_hypsometry(main_glac_rgi)
# Ice thickness [m]
main_glac_icethickness = modelsetup.import_icethickness(main_glac_rgi)
# Add volume [km**3] and mean elevation [m a.s.l.] to the main glaciers table
main_glac_rgi['Volume'], main_glac_rgi['Zmean'] = modelsetup.hypsometrystats(main_glac_hyps, main_glac_icethickness)
# Model time frame
dates_table, start_date, end_date = modelsetup.datesmodelrun(input.option_wateryear, input.option_leapyear)
# Initial surface type
modelsetup.surfacetypeglacinitial(main_glac_rgi, main_glac_hyps)


















