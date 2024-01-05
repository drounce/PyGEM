import pandas as pd
import numpy as np
import xarray as xr
from sklearn.metrics import mean_squared_error
import pygem_eb.input as eb_prms
eb_prms.enddate = eb_prms.startdate + pd.Timedelta(hours=2)
eb_prms.new_file = False
import run_simulation_eb as sim

# Initiate options for parameters to test
ds_attrs = {'Params?':'True'}
param_options = {'precgrad':[1e-5,1e-4],
          'lapserate':[-0.005,-0.008],
          'lapserate_dew':[-0.001,-0.003],
          'tsnow_threshold':[-1,1],
          'kp':[1,3],
          'temp_temp':[-3,0],
          'temp_depth':[50,200],
          'albedo_ice':[0.2,0.4],
          'roughness_fresh_snow':[0.1,0.5],
          'roughness_firn':[1,6],
          'roughness_ice':[0.8,2.5],
          'density_firn':[550,800]}

eb_prms.enddate = pd.to_datetime('2023-08-09 00:00')
# Import data for comparison
stake_df = pd.read_csv('~/research/MB_data/Stakes/gulkanaAB23.csv')
stake_df['Datetime'] = pd.to_datetime(stake_df['Date'])
stake_df = stake_df.set_index('Datetime')
stake_df = stake_df.loc[eb_prms.startdate:eb_prms.enddate]
daily_cum_melt_DATA = np.cumsum(stake_df['melt'].to_numpy())

# Loop through various parameters
sens_out = {'RMSE':{},'DS':{}}
for param in list(param_options.keys())[0:2]:
    for current_value in param_options[param]:
        import pygem_eb.input as eb_prms
        params = eb_prms.params
        eb_prms.params[param] = current_value
        # eb_prms.precgrad = params['precgrad']           # precipitation gradient on glacier [m-1]
        # eb_prms.lapserate = params['lapserate']         # temperature lapse rate for both gcm to glacier and on glacier between elevation bins [C m-1]
        # eb_prms.lapserate_dew = params['lapserate_dew'] # dew point temperature lapse rate [C m-1]
        # eb_prms.tsnow_threshold = params['tsnow_threshold']   # Threshold to consider freezing [C]
        # eb_prms.kp = params['kp']                      # precipitation factor [-] 
        # eb_prms.temp_temp = params['temp_temp']        # temperature of temperate ice in Celsius
        # eb_prms.temp_depth = params['temp_depth']      # depth where ice becomes temperate
        # eb_prms.albedo_fresh_snow = params['albedo_fresh_snow']    # Albedo of fresh snow [-] (Moelg et al. 2012, TC)
        # eb_prms.albedo_firn = params['albedo_firn']                # Albedo of firn [-]
        # eb_prms.albedo_ice = params['albedo_ice']                  # Albedo of ice [-] 
        # eb_prms.roughness_fresh_snow = params['roughness_fresh_snow']     # surface roughness length for fresh snow [mm] (Moelg et al. 2012, TC)
        # eb_prms.roughness_ice = params['roughness_ice']                   # surface roughness length for ice [mm] (Moelg et al. 2012, TC)
        # eb_prms.roughness_firn = params['roughness_firn']                 # surface roughness length for aged snow [mm] (Moelg et al. 2012, TC)
        # eb_prms.density_firn = params['density_firn']

        ds_run = sim.run_model(ds_attrs)
        ds_run = ds_run.sel(bin=0)
        daily_melt_MODEL = ds_run.resample(time='d').sum()
        daily_cum_melt_MODEL = daily_melt_MODEL['melt'].cumsum().to_numpy()
        melt_mse = mean_squared_error(daily_cum_melt_DATA,daily_cum_melt_MODEL)
        melt_rmse = np.mean(melt_mse)
        sens_out['RMSE'][param+'='+str(current_value)] = melt_rmse
        sens_out['DS'][param+'='+str(current_value)] = ds_run
print(sens_out['RMSE'])
np.save('sensitivity1.npy',sens_out['RMSE'])