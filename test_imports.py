#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 09:37:38 2018

@author: davidrounce
"""
import os
import linecache
import xarray as xr
import numpy as np
import pygem_input as input
import tracemalloc
from collections import Counter
#import gc

netcdf_fp = input.output_sim_fp + 'spc_zipped/'
netcdf_fn = 'R13_MPI-ESM-LR_rcp26_c2_ba1_100sets_2000_2100.nc'


    
    
def display_top(snapshot, key_type='lineno', limit=3):
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        # replace "/path/to/module/file.py" with "module/file.py"
        filename = os.sep.join(frame.filename.split(os.sep)[-2:])
        print("#%s: %s:%s: %.1f KiB"
              % (index, filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))


tracemalloc.start()

for n in list(np.arange(0,2)):
    print(n)
    ds = xr.open_dataset(netcdf_fp + netcdf_fn)
    runoff = ds.runoff_glac_monthly.values
#    temp = ds.temp_glac_monthly.values
#    prec = ds.prec_glac_monthly.values
#    melt = ds.melt_glac_monthly.values
#    ds = 0
#    temp2 = temp.copy()
#    runoff2 = runoff.copy()
#    gc.collect()

    snapshot = tracemalloc.take_snapshot()
    display_top(snapshot)
    
    