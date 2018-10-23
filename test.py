#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 16:27:09 2018

@author: davidrounce
"""

# Built-in libraries
import os
import argparse
import multiprocessing
import time
import inspect
import pygem_input as input
import numpy as np

output_list = []
check_dir = input.main_directory + '/../Output/cal_opt2_20181018/reg13/'
check_str = '13'
# Sorted list of files to merge
output_list = []
for i in os.listdir(check_dir):
    if i.startswith(check_str):
        output_list.append(i)
output_list = sorted(output_list)

A = [float(x.split('.nc')[0]) for x in output_list]
B = [int(round((x-13)*10**5,0)) for x in A]
C = np.array(B)
D = np.roll(C, shift=1)
E = C-D
F = np.where(E!=1)[0]
