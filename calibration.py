import pandas as pd
import numpy as np
import xarray as xr
import os
from itertools import combinations
from sklearn.metrics import mean_squared_error
import pygem_eb.input as eb_prms

# import model
eb_prms.startdate = pd.to_datetime('2023-04-18 00:30')
eb_prms.enddate = eb_prms.startdate + pd.Timedelta(hours=2)
eb_prms.new_file = False
import run_simulation_eb as sim
eb_prms.enddate = pd.to_datetime('2023-08-09 00:30')

# data for comparison
stake_df = pd.read_csv('~/research/MB_data/Stakes/gulkanaAB23_ALL.csv')
stake_df.index = pd.to_datetime(stake_df['Date'])
dates_index = pd.date_range(eb_prms.startdate,eb_prms.enddate) - pd.Timedelta(minutes=30)
stake_df = stake_df.loc[dates_index]

temp_df = pd.read_csv('~/research/MB_data/Gulkana/field_data/iButton_2023_all.csv')
temp_df.index = pd.to_datetime(temp_df['Datetime'])
temp_df = temp_df.loc[eb_prms.startdate:eb_prms.enddate]

# model parameters
params = {
    'kp':[0.5,0.8],
    'albedo_ice':[0.1,0.3],
    'k_ice':[1,4],
    'ksp_BC':[0.01,0.8]
}

i = 0
parser = sim.getparser()
args = parser.parse_args()
for kp in params['kp']:
    for aice in params['albedo_ice']:
        for ksp in params['ksp_BC']:
            for k_ice in params['k_ice']:
                eb_prms.output_name = f'{eb_prms.output_filepath}EB/{eb_prms.glac_name}_cal{i}'
                eb_prms.ksp_BC = ksp
                eb_prms.albedo_ice = aice
                eb_prms.kp = kp
                eb_prms.k_ice = k_ice

                climateds,dates_table,utils = sim.initialize_model('01.00570',args)
                ds_run = sim.run_model(climateds,dates_table,utils,args,
                                    {'ksp_BC':ksp,'albedo_ice':aice,'kp':kp,'k_ice':k_ice})
                
                i += 1