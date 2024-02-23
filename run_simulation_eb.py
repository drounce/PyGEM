import argparse
import time
# External libraries
import numpy as np
import xarray as xr
import pandas as pd
from multiprocessing import Pool
# Internal libraries
import pygem_eb.input as eb_prms
import pygem.class_climate as class_climate
import pygem_eb.massbalance as mb
import pygem.pygem_modelsetup as modelsetup
import pygem_eb.utils as utilities

# ===== INITIALIZE UTILITIES =====
def getparser():
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
    parser.add_argument('-climate_input', action='store', type=str,
                        default=eb_prms.climate_input,
                        help='pass str of AWS or GCM')
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
    return parser

def initialize_model(debug=True):
    # Initialize argparse and utility functions
    parser = getparser()
    args = parser.parse_args()
    n_bins = args.n_bins

    # ===== GLACIER AND TIME PERIOD SETUP =====
    glacier_table = modelsetup.selectglaciersrgitable(args.glac_no,
                    rgi_regionsO1=eb_prms.rgi_regionsO1,
                    rgi_regionsO2=eb_prms.rgi_regionsO2,
                    rgi_glac_number=eb_prms.rgi_glac_number,
                    include_landterm=eb_prms.include_landterm,
                    include_laketerm=eb_prms.include_laketerm,
                    include_tidewater=eb_prms.include_tidewater)
    dates_table = modelsetup.datesmodelrun(startyear=args.startdate, endyear=args.enddate)
    utils = utilities.Utils(args,glacier_table)

    gcm = class_climate.GCM(name=eb_prms.ref_gcm_name)
    nans = np.empty(len(dates_table))*np.nan
    if args.climate_input in ['GCM']:
        all_vars = list(gcm.var_dict.keys())
        if eb_prms.ref_gcm_name in ['MERRA2']:
            cenlat = glacier_table['CenLat'].to_numpy()
            cenlon = glacier_table['CenLon'].to_numpy()
            file_lat = str(int(np.floor(cenlat/10)*10))
            file_lon = str(int(np.floor(cenlon/10)*10))
            for var in all_vars:
                fn = gcm.var_dict[var]['fn'].replace('LAT',file_lat).replace('LON',file_lon)
                gcm.var_dict[var]['fn'] = fn
        # ===== LOAD CLIMATE DATA =====
        all_data = {}
        for var in all_vars:
            if var not in ['time','lat','lon','elev']:
                data,data_hours = gcm.importGCMvarnearestneighbor_xarray(gcm.var_dict[var]['fn'],
                                                                     gcm.var_dict[var]['vn'],
                                                                     glacier_table,dates_table)
                if var in ['SWin','LWin','tcc','rh','bcdry','bcwet','dustdry','dustwet']:
                    data = data[0]
                if eb_prms.ref_gcm_name in ['MERRA2'] and var in ['SWin','LWin']:
                    data = data * 3600
            elif var == 'elev':
                data = gcm.importGCMfxnearestneighbor_xarray(
                    gcm.var_dict[var]['fn'], gcm.var_dict[var]['vn'], glacier_table)
            all_data[var] = data
        temp_data = all_data['temp']
        tp_data = all_data['prec']
        sp_data = all_data['sp']
        elev_data = all_data['elev']
        rh = all_data['rh']
        uwind = all_data['uwind']
        vwind = all_data['vwind']
        SWin = all_data['SWin']
        LWin = all_data['LWin']
        tcc = all_data['tcc']
        bcdry = all_data['bcdry']
        bcwet = all_data['bcwet']
        dustdry = all_data['dustdry']
        dustwet = all_data['dustwet']
        wind = np.sqrt(np.power(uwind[0],2)+np.power(vwind[0],2))
        winddir = np.arctan2(-uwind[0],-vwind[0]) * 180 / np.pi
        LWout = nans.copy()
        SWout = nans.copy()
        NR = nans.copy()
        ntimesteps = len(data_hours)
    elif args.climate_input in ['AWS']:
        aws = class_climate.AWS(eb_prms.AWS_fn,dates_table)
        temp_data = aws.temp
        tp_data = aws.tp
        rh = aws.rh
        SWin = aws.SWin
        SWout = aws.SWout
        LWin = aws.LWin
        LWout = aws.LWout
        NR = aws.NR
        wind = aws.wind
        winddir = aws.winddir
        tcc = aws.tcc
        sp_data = aws.sp
        elev_data = aws.elev
        bcdry = aws.bcdry
        bcwet = aws.bcwet
        dustdry = aws.dustdry
        dustwet = aws.dustwet
        ntimesteps = len(temp_data)

    # Adjust elevation-dependant climate variables
    temp,tp,sp = utils.getBinnedClimate(temp_data, tp_data, sp_data,
                                            ntimesteps, elev_data)
    dates = pd.date_range(args.startdate,args.enddate,freq='h')

    # ===== SET UP CLIMATE DATASET =====
    bin_idx = np.arange(0,n_bins)
    climateds = xr.Dataset(data_vars = dict(
        bin_elev = (['bin'],eb_prms.bin_elev,{'units':'m a.s.l.'}),
        SWin = (['time'],SWin,{'units':'J m-2'}),
        SWout = (['time'],SWout,{'units':'J m-2'}),
        LWin = (['time'],LWin,{'units':'J m-2'}),
        LWout = (['time'],LWout,{'units':'J m-2'}),
        NR = (['time'],NR,{'units':'J m-2'}),
        tcc = (['time'],tcc,{'units':'1'}),
        rh = (['time'],rh,{'units':'%'}),
        wind = (['time'],wind,{'units':'m s-1'}),
        winddir = (['time'],winddir,{'units':'deg'}),
        bcdry = (['time'],bcdry,{'units':'kg m-2 s-1'}),
        bcwet = (['time'],bcwet,{'units':'kg m-2 s-1'}),
        dustdry = (['time'],dustdry,{'units':'kg m-2 s-1'}),
        dustwet = (['time'],dustwet,{'units':'kg m-2 s-1'}),
        bin_temp = (['bin','time'],temp,{'units':'C'}),
        bin_tp = (['bin','time'],tp,{'units':'m'}),
        bin_sp = (['bin','time'],sp,{'units':'Pa'})
        ),
        coords = dict(
            bin=(['bin'],bin_idx),
            time=(['time'],dates)
            ))
    
    return climateds,dates_table,utils,args

def run_model(climateds,dates_table,utils,args,new_attrs):
    # Start timer
    start_time = time.time()

    # ===== RUN ENERGY BALANCE =====
    if eb_prms.parallel:
        def run_mass_balance(bin):
            massbal = mb.massBalance(bin,dates_table,args,utils)
            massbal.main(climateds)
        processes_pool = Pool(args.n_bins)
        processes_pool.map(run_mass_balance,range(args.n_bins))
    else:
        for bin in np.arange(args.n_bins):
            massbal = mb.massBalance(bin,dates_table,args,utils)
            massbal.main(climateds)
            
            if bin<args.n_bins-1:
                print('Success: moving onto bin',bin+1)

    # ===== END ENERGY BALANCE =====
    # Get final model run time
    end_time = time.time()
    time_elapsed = end_time-start_time
    print(f'Total Time Elapsed: {time_elapsed:.1f} s')

    # Store metadata in netcdf and save result
    if args.store_data:
        massbal.output.addVars()
        massbal.output.addAttrs(args,time_elapsed)
        ds_out = massbal.output.addNewAttrs(new_attrs)
        print('Success: saving to',eb_prms.output_name+'.nc')
    else:
        print('Success: data was not saved')
        ds_out = None
    
    return ds_out

climateds,dates_table,utils,args = initialize_model()
out = run_model(climateds,dates_table,utils,args,{'Params?':'False'})
if out:
    # Get final mass balance
    print(f'Total Mass Loss: {out.melt.sum():.3f} m w.e.')