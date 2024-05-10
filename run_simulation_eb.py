import argparse
import time
# External libraries
import numpy as np
import xarray as xr
import pandas as pd
from multiprocessing import Pool
import threading
# Internal libraries
import pygem_eb.input as eb_prms
import pygem_eb.massbalance as mb
import pygem.pygem_modelsetup as modelsetup
import pygem_eb.climate as climutils

# Start timer
start_time = time.time()

# ===== INITIALIZE UTILITIES =====
def get_args():
    parser = argparse.ArgumentParser(description='pygem-eb model runs')
    # add arguments
    parser.add_argument('-glac_no', action='store', default=eb_prms.glac_no,
                        help='',nargs='+')
    parser.add_argument('-start','--startdate', action='store', type=str, 
                        default=eb_prms.startdate,
                        help='pass str like datetime of model run start')
    parser.add_argument('-end','--enddate', action='store', type=str,
                        default=eb_prms.enddate,
                        help='pass str like datetime of model run end')
    parser.add_argument('-use_AWS', action='store_true',
                        default=eb_prms.use_AWS,help='use AWS or just reanalysis?')
    parser.add_argument('-store_data', action='store_true', 
                        default=eb_prms.store_data, help='')
    parser.add_argument('-new_file', action='store_true',
                        default=eb_prms.new_file, help='')
    parser.add_argument('-debug', action='store_true', 
                        default=eb_prms.debug, help='')
    parser.add_argument('-n_bins',action='store',type=int,
                        default=eb_prms.n_bins, help='number of elevation bins')
    parser.add_argument('-switch_LAPs',action='store', type=int,
                        default=eb_prms.switch_LAPs, help='')
    parser.add_argument('-switch_melt',action='store', type=int, 
                        default=eb_prms.switch_melt, help='')
    parser.add_argument('-switch_snow',action='store', type=int,
                        default=eb_prms.switch_snow, help='')
    parser.add_argument('-f', '--fff', help='dummy arg to fool ipython', default='1')
    args = parser.parse_args()
    return args

def initialize_model(glac_no,args,debug=True):
    """
    Loads glacier table and climate dataset for one glacier to initialize
    the model inputs.

    Parameters
    ==========
    glac_no : str
        RGI glacier ID
    
    Returns
    -------
    climate
        Class object from climate.py
    """
    # ===== GLACIER AND TIME PERIOD SETUP =====
    glacier_table = modelsetup.selectglaciersrgitable(np.array([glac_no]),
                    rgi_regionsO1=eb_prms.rgi_regionsO1)
    climate = climutils.Climate(args,glacier_table)

    # Load in available AWS data
    if args.use_AWS:
        need_vars = climate.get_AWS(eb_prms.AWS_fn)
        climate.get_reanalysis(need_vars)
    else:
        climate.get_reanalysis(climate.all_vars)
    climate.check_ds()

    return climate

def run_model(climate,args,store_attrs=None):
    """
    Executes model functions in parallel or series and
    stores output data.

    Parameters
    ==========
    climate
        Class object with climate data from initialize_model
    args
        Command line arguments from get_args
    add_attrs : dict
        Dictionary of additional metadata to store in the .nc
    """
    # ===== RUN ENERGY BALANCE =====
    if eb_prms.parallel:
        def run_mass_balance(bin):
            massbal = mb.massBalance(bin,args,climate)
            massbal.main()
        processes_pool = Pool(args.n_bins)
        processes_pool.map(run_mass_balance,range(args.n_bins))
    for bin in np.arange(args.n_bins):
        massbal = mb.massBalance(bin,args,climate)
        massbal.main()
        
        if bin<args.n_bins-1:
            print('Success: moving onto bin',bin+1)

    # ===== END ENERGY BALANCE =====
    # Get final model run time
    end_time = time.time()
    time_elapsed = end_time-start_time
    print(f'Total Time Elapsed: {time_elapsed:.1f} s')

    # Store metadata in netcdf and save result
    if args.store_data:
        massbal.output.add_vars()
        massbal.output.add_basic_attrs(args,time_elapsed,climate)
        ds_out = massbal.output.add_attrs(store_attrs)
        print('Success: saving to',eb_prms.output_name+'.nc')
    else:
        print('Success: data was not saved')
        ds_out = None
    
    return ds_out

args = get_args()
for gn in args.glac_no:
    climate = initialize_model(gn,args)
    out = run_model(climate,args)
    if out:
        # Get final mass balance
        print(f'Total Mass Loss: {out.melt.sum():.3f} m w.e.')