#%% IMPORTS
import pandas as pd
import numpy as np
import xarray as xr
import os
from sklearn.metrics import mean_squared_error
import pygem_eb.input as eb_prms
eb_prms.enddate = eb_prms.startdate + pd.Timedelta(hours=2)
eb_prms.new_file = False
import run_simulation_eb as sim
import copy

#%% LOAD IN PARAMETERS
# Initiate options for parameters to test
param_options = {
          'lapserate':{'options':[-0.003,-0.01],'value':-0.0065,'units':'K m-1'},
          'k_ice':{'options':[1.5,3],'value':2.33,'units':'W K-1 m-1'},
          'precgrad':{'options':[5e-6,5e-4],'value':0.0001,'units':''},
          'kp':{'options':[0.5,2],'value':1,'units':''},
          'albedo_ice':{'options':[0.2,0.4],'value':0.3,'units':''},
          'roughness_ice':{'options':[0.5,3],'value':1.7,'units':'mm'},
          'ksp_BC':{'options':[0.01,0.8],'value':0.1,'units':'kg kg-1'},
          'ksp_dust':{'options':[0.001,0.08],'value':0.015,'units':'kg kg-1'},
          'dz_toplayer':{'options':[0.01,0.05],'value':0.03,'units':'m'},
          'layer_growth':{'options':[0.6,0.9],'value':0.6,'units':''},
          'roughness_aging_rate':{'options':[0.04,0.08],'value':0.06267,'units':'mm d-1'},
          'dry_metamorphism_rate':{'options':[1e-5,1e-4],'value':3.5e-5,'units':'um s-1'}
    # NOTHING BELOW HERE HAS A BIG IMPACT ON MASS BALANCE -- SET CONSTANT
    # MIGHT BE GOOD TO DOUBLE CHECK FIRN DATA ON A BIN THAT HAS FIRN
        #   'initSSA':{'options':[80,100],'value':100}
        #   'lapserate_dew':[-0.001,-0.003],
        #   'temp_temp':[-3,0],
        #   'temp_depth':[50,200],
        #   'roughness_fresh_snow':{'options':[0.1,0.5],'value':0.24},
        #   'roughness_firn':[1,6],
        #   'density_firn':[550,800],
        #   'tsnow_threshold':{'options':[0,2],'value':0,'units':'C'},
        #   'fresh_grainsize':{'options':[50,100],'value':54.5,'units':'um'},
        #   'albedo_TOD':{'options':[8,16],'value':0,'units':'hrs'},
        #   'initSSA':{'options':[60,100],'value':80,'units':'m2 kg-1'},
          }

eb_prms.enddate = pd.to_datetime('2023-08-09 00:30')

# Import data for comparison
stake_df = pd.read_csv('~/research/MB_data/Stakes/gulkanaAB23.csv')
stake_df['Datetime'] = pd.to_datetime(stake_df['Date'])
stake_df = stake_df.set_index('Datetime')
stake_df = stake_df.loc[eb_prms.startdate:eb_prms.enddate]
daily_cum_melt_DATA = np.cumsum(stake_df['melt'].to_numpy())

model_run_date = str(pd.Timestamp.today()).replace('-','_')[0:10]

#%% RUN MODEL
# Loop through various parameters
sens_out = {'RMSE':{},'DATASET':{}}
parser = sim.getparser()
args = parser.parse_args()
for param in list(param_options.keys()):
    for current_value in param_options[param]['options']:
        eb_prms.output_name = f'{eb_prms.output_filepath}EB/{eb_prms.glac_name}_{model_run_date}_{param}_{current_value}'
        if os.path.exists(eb_prms.output_name):
            continue
        # update parameter to adjust
        params = copy.deepcopy(param_options)
        params[param]['value'] = current_value
        eb_prms.lapserate = params['lapserate']['value']  # temperature lapse rate for both gcm to glacier and on glacier between elevation bins [C m-1]
        eb_prms.k_ice = params['k_ice']['value']                # surface roughness length for ice [mm] (Moelg et al. 2012, TC)
        eb_prms.precgrad = params['precgrad']['value']           # precipitation gradient on glacier [m-1]
        eb_prms.kp = params['kp']['value']                      # precipitation factor [-] 
        eb_prms.albedo_ice = params['albedo_ice']['value']                  # Albedo of ice [-] 
        eb_prms.roughness_ice = params['roughness_ice']['value']   
        eb_prms.ksp_BC = params['ksp_BC']['value']
        eb_prms.ksp_dust = params['ksp_dust']['value']
        eb_prms.dz_toplayer = params['dz_toplayer']['value']
        eb_prms.layer_growth = params['layer_growth']['value']
        eb_prms.roughness_aging_rate = params['roughness_aging_rate']['value']
        eb_prms.dry_metamorphism_rate = params['dry_metamorphism_rate']['value']
        # eb_prms.fresh_grainsize = params['fresh_grainsize']['value']
        # eb_prms.tsnow_threshold = params['tsnow_threshold']['value']   # Threshold to consider freezing [C]
        # eb_prms.albedo_TOD = params['albedo_TOD']['value']
        # eb_prms.initSSA = params['initSSA']['value']
        climateds,dates_table,utils = sim.initialize_model(args.glac_no[0],args)
        ds_run = sim.run_model(climateds,dates_table,utils,args,{'Params?':'True'})
        # ds_run = ds_run.sel(bin=0)
        # daily_melt_MODEL = ds_run.resample(time='d').sum()
        # daily_cum_melt_MODEL = daily_melt_MODEL['melt'].cumsum().to_numpy()
        # melt_mse = mean_squared_error(daily_cum_melt_DATA,daily_cum_melt_MODEL)
        # melt_rmse = np.mean(melt_mse)
        # sens_out['RMSE'][param+'='+str(current_value)] = melt_rmse
        # sens_out['DATASET'][param+'='+str(current_value)] = ds_run
print('Done')