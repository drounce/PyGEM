"""
Python Glacier Evolution Model (PyGEM)

copyright © 2018 David Rounce <drounce@cmu.edu>

Distrubted under the MIT lisence

dynamical spinup
"""
# Built-in libraries
import os
import sys
import math

# External libraries
import numpy as np
# oggm imports
from oggm import cfg
from oggm import graphics
from oggm import tasks
from oggm import utils
from oggm import workflow
from oggm.core.flowline import FluxBasedModel
from oggm.core.massbalance import apparent_mb_from_any_mb
from oggm.core.inversion import find_inversion_calving_from_any_mb
# pygem imports
# load pygem config
import pygem.setup.config as config
# Read the config
pygem_prms = config.read_config()
from pygem.massbalance import PyGEMMassBalance

# pygem dynamical spinup class object
class dynamical_spinup():
    def __init__(self, gdir, modelprms, glacier_rgi_table, fls, ref_startyear, ref_spinupyears, debug=False):
        self.gdir=gdir
        self.modelprms=modelprms
        self.glacier_rgi_table=glacier_rgi_table
        self.fls=fls
        self.ref_startyear=ref_startyear
        self.spinupyrs=ref_spinupyears
        self.debug=debug
        self.mb_mwea=None

    # get glacierwide specific mass balance
    def get_gw_mb(self, yrs=np.arange(1980,2000), debug=False):
        yrs_ = yrs % (self.ref_startyear - self.spinupyrs)
        if yrs_[0] != 0:
            yrs = np.arange(0,yrs_[-1]+1)
        else:
            yrs = yrs_
        # PyGEM mass balance model
        mbmod = PyGEMMassBalance(gdir=self.gdir, modelprms=self.modelprms, glacier_rgi_table=self.glacier_rgi_table, fls=self.fls, option_areaconstant=True)
        for year in yrs:
            mbmod.get_annual_mb(self.fls[0].surface_h, fls=self.fls, fl_id=0, year=year)

        t1_idx = int(yrs_[0] * 12)
        t2_idx = int((yrs_[-1] + 1) * 12) - 1
        self.mb_mwea = mbmod.glac_wide_massbaltotal[t1_idx:t2_idx+1].sum() / mbmod.glac_wide_area_annual[0] / self.spinupyrs
        if debug:
            print(f"{self.gdir.dates_table.iloc[t1_idx]['date']} - {self.gdir.dates_table.iloc[t2_idx]['date']}, {round(self.mb_mwea,2)}")

    # adjust tbias such that glacier-wide mass balance is 0 for spinup period
    def tbias_adjust(self, tbias_step=0.05, debug=False):
        years = np.arange(self.spinupyrs) + (self.ref_startyear - self.spinupyrs)
        self.get_gw_mb(yrs=years)
        if debug:
            print(f"starting mb_mwea: {round(self.mb_mwea,2)}")
            print(f"starting tbias: {round(self.modelprms['tbias'],2)}")

        # mb too high
        while self.mb_mwea > 0:
            self.modelprms['tbias'] += tbias_step
            self.get_gw_mb(yrs=years)

        # mb too low
        while self.mb_mwea < 0:
            self.modelprms['tbias'] -= tbias_step
            self.get_gw_mb(yrs=years)

        if debug:
            print(f"ending mb_mwea: {round(self.mb_mwea,2)}")
            print(f"ending tbias: {round(self.modelprms['tbias'],2)}")

    def ice_thickness_inversion(self, glen_a_mult=1, fs=0):
        nyears = int(self.spinupyrs)

        # perform OGGM ice thickness inversion
        # Perform inversion based on PyGEM MB using reference directory
        mbmod_inv = PyGEMMassBalance(self.gdir, self.modelprms, self.glacier_rgi_table,
                                        fls=self.fls, option_areaconstant=True,
                                        inversion_filter=pygem_prms['mb']['include_debris'])
        if not self.gdir.is_tidewater or not pygem_prms['setup']['include_calving']:
            # Arbitrariliy shift the MB profile up (or down) until mass balance is zero (equilibrium for inversion)
            apparent_mb_from_any_mb(self.gdir, mb_years=np.arange(nyears), mb_model=mbmod_inv)
            tasks.prepare_for_inversion(self.gdir)
            tasks.mass_conservation_inversion(self.gdir, glen_a=cfg.PARAMS['glen_a']*glen_a_mult, fs=fs)
        # Tidewater glaciers
        else:
            cfg.PARAMS['use_kcalving_for_inversion'] = True
            cfg.PARAMS['use_kcalving_for_run'] = True
            out_calving = find_inversion_calving_from_any_mb(self.gdir, mb_model=mbmod_inv, mb_years=np.arange(nyears),
                                                                glen_a=cfg.PARAMS['glen_a']*glen_a_mult, fs=fs)
