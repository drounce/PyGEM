Introduction
============

Purpose
**********

The Python Glacier Evolution Model (PyGEM) is an open-source glacier evolution model coded in Python that is designed to model the transient evolution of glaciers on regional and global scales. Each glacier is modeled independently using a given time step and elevation bins. The model computes the climatic mass balance (i.e., snow accumulation minus melt plus refreezing) for each elevation bin and each monthly time step. Glacier geometry is updated annually. The model outputs a variety of data including monthly mass balance and its components (accumulation, melt, refreezing, frontal ablation, glacier runoff),  and annual volume, volume below sea level, and area.

PyGEM has a modular framework that allows different schemes to be used for model calibration or model physics (e.g., ablation, accumulation, refreezing, glacier dynamics). The most recent version of PyGEM has been made compatible with the Open Global Glacier Model (OGGM; https://oggm.org/) to both leverage the pre-processing tools (e.g., digital elevation models, glacier characteristics) and their advances with respect to modeling glacier dynamics and ice thickness inversions.

Spatial and Temporal Resolution
***********

Each glacier is modeled independently. The model is currently set up to use a monthly timestep and elevation bins. We plan to add options to include daily timesteps in the future. The elevation bins can be specified in pre-processing with OGGM.
