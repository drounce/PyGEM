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
    parser.add_argument('-start','--startdate', action='store', type=str, default=eb_prms.startdate,
                        help='pass str like datetime of model run start')
    parser.add_argument('-end','--enddate', action='store', type=str, default=eb_prms.enddate,
                        help='pass str like datetime of model run end')
    parser.add_argument('-climate_input', action='store', type=str, default=eb_prms.climate_input,
                        help='pass str-like datetime of model run start')
    parser.add_argument('-store_data', action='store_true', default=eb_prms.store_data,
                        help='')
    parser.add_argument('-new_file', action='store_true', default=eb_prms.new_file,
                        help='')
    parser.add_argument('-debug', action='store_true', default=eb_prms.debug,
                        help='')
    parser.add_argument('-n_bins',action='store',type=int,default=eb_prms.n_bins,
                        help='number of elevation bins')
    parser.add_argument('-switch_LAPs',action='store', type=int, default=eb_prms.switch_LAPs,
                        help='')
    parser.add_argument('-switch_melt',action='store', type=int, default=eb_prms.switch_melt,
                        help='')
    parser.add_argument('-switch_snow',action='store', type=int, default=eb_prms.switch_snow,
                        help='')
    return parser

def run_model(new_attrs):
    # Start timer and initialize argparse and utility functions
    start_time = time.time()
    parser = getparser()
    args = parser.parse_args()
    n_bins = args.n_bins
    utils = utilities.Utils(args)

    # ===== GLACIER AND TIME PERIOD SETUP =====
    glacier_table = modelsetup.selectglaciersrgitable(args.glac_no,
                    rgi_regionsO1=eb_prms.rgi_regionsO1, rgi_regionsO2=eb_prms.rgi_regionsO2,
                    rgi_glac_number=eb_prms.rgi_glac_number, include_landterm=eb_prms.include_landterm,
                    include_laketerm=eb_prms.include_laketerm, include_tidewater=eb_prms.include_tidewater)
    dates_table = modelsetup.datesmodelrun(startyear=args.startdate, endyear=args.enddate)

    gcm = class_climate.GCM(name=eb_prms.ref_gcm_name)
    nans = np.empty(len(dates_table))*np.nan
    if args.climate_input in ['GCM']:
        # ===== LOAD CLIMATE DATA =====
        tp_data, data_hours = gcm.importGCMvarnearestneighbor_xarray(gcm.prec_fn, gcm.prec_vn, glacier_table,dates_table)
        temp_data, data_hours = gcm.importGCMvarnearestneighbor_xarray(gcm.temp_fn, gcm.temp_vn, glacier_table,dates_table)
        dtemp_data, data_hours = gcm.importGCMvarnearestneighbor_xarray(gcm.dtemp_fn, gcm.dtemp_vn, glacier_table,dates_table)
        sp_data, data_hours = gcm.importGCMvarnearestneighbor_xarray(gcm.sp_fn, gcm.sp_vn, glacier_table,dates_table)
        tcc, data_hours = gcm.importGCMvarnearestneighbor_xarray(gcm.tcc_fn, gcm.tcc_vn, glacier_table,dates_table)
        SWin, data_hours = gcm.importGCMvarnearestneighbor_xarray(gcm.SWin_fn, gcm.SWin_vn, glacier_table,dates_table) 
        LWin, data_hours = gcm.importGCMvarnearestneighbor_xarray(gcm.SWin_fn, gcm.SWin_vn, glacier_table,dates_table) 
        uwind, data_hours = gcm.importGCMvarnearestneighbor_xarray(gcm.uwind_fn, gcm.uwind_vn, glacier_table,dates_table)                                                      
        vwind, data_hours = gcm.importGCMvarnearestneighbor_xarray(gcm.vwind_fn, gcm.vwind_vn, glacier_table,dates_table)
        try:
            depBC, data_hours = gcm.importGCMvarnearestneighbor_xarray(gcm.depBC_fn, gcm.depBC_vn, glacier_table,dates_table)
            depdust, data_hours = gcm.importGCMvarnearestneighbor_xarray(gcm.depdust_fn, gcm.depdust_vn, glacier_table,dates_table)
        except:
            depBC = nans.copy()
            depdust = nans.copy()
        elev_data = gcm.importGCMfxnearestneighbor_xarray(gcm.elev_fn, gcm.elev_vn, glacier_table)
        wind = np.sqrt(np.power(uwind[0],2)+np.power(vwind[0],2))
        winddir = np.arctan2(-uwind[0],-vwind[0]) * 180 / np.pi
        LWout = nans.copy()
        SWout = nans.copy()
        rh_data = nans.copy()
        NR = nans.copy()
        SWin = SWin[0]
        LWin = LWin[0]
        tcc = tcc[0]
        ntimesteps = len(data_hours)
    elif args.climate_input in ['AWS']:
        aws = class_climate.AWS(eb_prms.AWS_fn,dates_table)
        temp_data = aws.temp
        tp_data = aws.tp
        dtemp_data = aws.dtemp
        rh_data = aws.rh
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
        bc1dry = aws.bc1dry
        bc2dry = aws.bc2dry
        bc1wet = aws.bc1wet
        bc2wet = aws.bc2wet
        du1dry = aws.du1dry
        du2dry = aws.du2dry
        du3dry = aws.du3dry
        du4dry = aws.du4dry
        du5dry = aws.du5dry
        du1wet = aws.du1wet
        du2wet = aws.du2wet
        du3wet = aws.du3wet
        du4wet = aws.du4wet
        du5wet = aws.du5wet
        ntimesteps = len(temp_data)

    # Adjust elevation-dependant climate variables
    temp,tp,sp,dtemp,rh = utils.getBinnedClimate(temp_data, tp_data, sp_data, dtemp_data, rh_data,
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
        tcc = (['time'],tcc,{'units':'0-1'}),
        wind = (['time'],wind,{'units':'m s-1'}),
        winddir = (['time'],winddir,{'units':'deg'}),
        bc1dry = (['time'],bc1dry,{'units':'kg m-2 s-1'}),
        bc2dry = (['time'],bc2dry,{'units':'kg m-2 s-1'}),
        bc1wet = (['time'],bc1wet,{'units':'kg m-2 s-1'}),
        bc2wet = (['time'],bc2wet,{'units':'kg m-2 s-1'}),
        du1dry = (['time'],du1dry,{'units':'kg m-2 s-1'}),
        du2dry = (['time'],du2dry,{'units':'kg m-2 s-1'}),
        du3dry = (['time'],du3dry,{'units':'kg m-2 s-1'}),
        du4dry = (['time'],du4dry,{'units':'kg m-2 s-1'}),
        du5dry = (['time'],du5dry,{'units':'kg m-2 s-1'}),
        du1wet = (['time'],du1wet,{'units':'kg m-2 s-1'}),
        du2wet = (['time'],du2wet,{'units':'kg m-2 s-1'}),
        du3wet = (['time'],du3wet,{'units':'kg m-2 s-1'}),
        du4wet = (['time'],du4wet,{'units':'kg m-2 s-1'}),
        du5wet = (['time'],du5wet,{'units':'kg m-2 s-1'}),
        bin_temp = (['bin','time'],temp,{'units':'C'}),
        bin_tp = (['bin','time'],tp,{'units':'m'}),
        bin_sp = (['bin','time'],sp,{'units':'Pa'}),
        bin_rh = (['bin','time'],rh,{'units':'%'})
        ),
        coords = dict(
            bin=(['bin'],bin_idx),
            time=(['time'],dates)
            ))

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
    
    return ds_out

out = run_model({'Params?':'False'})