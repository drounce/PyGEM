#!/usr/bin/env python3
import setuptools
from setuptools import find_packages  # or find_namespace_packages

if __name__ == "__main__":
    setuptools.setup(
        # ...,
        # package_dir={'':'pygem'}
        # ...
        # packages= []
        # find_packages(
            # All keyword arguments below are optional:
            
            # where='pygem',  # '.' by default
            # include=['mypackage*'],  # ['*'] by default
            # exclude=['mypackage.tests'],  # empty by default
        # ),
        # ...
    )