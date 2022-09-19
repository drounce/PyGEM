#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 16:31:03 2022

@author: drounce
"""


from oggm import utils
from oggm import cfg

cfg.PARAMS['has_internet'] = True

fp = utils.file_downloader('https://cluster.klima.uni-bremen.de/~oggm/cmip6_tutorials/crop_hma/CESM2/CESM2_ssp245_r1i1p1f1_pr.nc')
ft = utils.file_downloader('https://cluster.klima.uni-bremen.de/~oggm/cmip6_tutorials/crop_hma/CESM2/CESM2_ssp245_r1i1p1f1_tas.nc')
fe = utils.file_downloader('https://cluster.klima.uni-bremen.de/~oggm/cmip6_tutorials/crop_hma/CESM2/CESM2_orog.nc')