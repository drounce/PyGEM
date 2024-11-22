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
# load pygem config
import pygem.setup.config as config
# Read the config
pygem_prms = config.read_config()
