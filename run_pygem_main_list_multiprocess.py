"""
Create list to run pygem_main on parallels using multiprocesing and subprocess
"""
#========= LIST OF PACKAGES ==================================================
import numpy as np
import scipy
import os
import glob
import time
import datetime as dt
import argparse
import subprocess as sp
import multiprocessing
#========= IMPORT COMMON VARIABLES FROM MODEL INPUT ==========================
import pygem_input as input

#%% Argument parser to assist breaking down all the glaciers into separate blocks
parser = argparse.ArgumentParser(  \
             description = """run commands in command_file on num_simultaneous_processes processors""",
             epilog = '>> <<',
             formatter_class = argparse.RawDescriptionHelpFormatter)
parser.add_argument('-num_simultaneous_processes',
                    action = 'store',
                    type = int,
                    default = 2,
                    help = 'number of simulataneous processes (cores) to use [%(default)s]')
parser.add_argument('pygem_command_file',
                    action = 'store',
                    default = 'pygem_main.txt',
                    help = 'text file full of pygem commands to run {}'.format(default))
args = parser.parse_args()

with open(args.pygem_command_file, 'r') as pygem_cf:
    process_list = pygem_cf.read().splitlines()

print('found %d commands to run '%(len(process_list)))