#!/usr/bin/env python3
"""
Created on Sept 19 2023

@author: btobers mrweathers drounce

PyGEM classes and subclasses for glacier simulation outputs

The two main parent classes are single_glacier(object) and compiled_regional(object)
Both of these have several subclasses which will inherit the necessary parent information
"""
import pygem_input as pygem_prms

### single glacier output parent class ###
class single_glacier(object):
    """
    Single glacier output dataset class for the Python Glacier Evolution Model.
    """
    def __init__(self):

class single_glacier_stats(single_glacier):
    """
    Singla glacier statistics dataset
    """
    def __init__(self):

class single_glacier_binned(single_glacier):
    """
    Single glacier binned dataset
    """
    def __init__(self):



### compiled regional output parent class ###
class compiled_regional(object):
    """
    Compiled regional output dataset for the Python Glacier Evolution Model.
    """
    def __init__(self):

class regional_annual_mass(compiled_regional):
    """
    compiled regional annual mass
    """
    def __init__(self):

class regional_annual_area(compiled_regional):
    """
    compiled regional annual area
    """
    def __init__(self):

class regional_monthly_runoff(compiled_regional):
    """
    compiled regional monthly runoff
    """
    def __init__(self):

class regional_monthly_massbal(compiled_regional):
    """
    compiled regional monthly climatic mass balance
    """
    def __init__(self):