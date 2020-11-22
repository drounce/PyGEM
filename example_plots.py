#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 07:58:00 2019

@author: davidrounce
"""

import os
import cartopy
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

#pd.read_csv('')

#%% X-Y PLOT
#ds = pd.read_csv(...)

# X,Y values
x_values = [0,1,2,3]
y_values = [2,-1,1,5]
y2_values = [3, 2, 3, 0]

# Set up your plot (and/or subplots)
fig, ax = plt.subplots(1, 1, squeeze=False, sharex=False, sharey=False, gridspec_kw = {'wspace':0.4, 'hspace':0.15})
             
# Plot
#  zorder controls the order of the plots (higher zorder plots on top)
#  label used to automatically generate legends (legends can be done manually for more control)
ax[0,0].plot(x_values, y_values, color='k', linewidth=1, zorder=2, label='plot1')
ax[0,0].plot(x_values, y2_values, color='b', linewidth=1, zorder=2, label='plot2')

# Fill between
#  fill between is useful for putting colors between plots (e.g., error bounds)
#ax[0,0].fill_between(x_values, y_values, y2_values, facecolor='k', alpha=0.2, zorder=1)

# Text
#  text can be used to manually add labels or to comment on plot
#  transform=ax.transAxes means the x and y are between 0-1
ax[0,0].text(0.5, 0.99, '[insert text]', size=10, horizontalalignment='center', verticalalignment='top', 
             transform=ax[0,0].transAxes)

# X-label
ax[0,0].set_xlabel('X LABEL', size=12)
#ax[0,0].set_xlim(time_values_annual[t1_idx:t2_idx].min(), time_values_annual[t1_idx:t2_idx].max())
#ax[0,0].xaxis.set_tick_params(labelsize=12)
#ax[0,0].xaxis.set_major_locator(plt.MultipleLocator(50))
#ax[0,0].xaxis.set_minor_locator(plt.MultipleLocator(10))
#ax[0,0].set_xticklabels(['2015','2050','2100'])       
 
# Y-label
ax[0,0].set_ylabel('Y LABEL', size=12)
#ax[0,0].set_ylim(0,1.1)
#ax[0,0].yaxis.set_major_locator(plt.MultipleLocator(0.2))
#ax[0,0].yaxis.set_minor_locator(plt.MultipleLocator(0.05))

# Tick parameters
#  controls the plotting of the ticks
#ax[0,0].yaxis.set_ticks_position('both')
#ax[0,0].tick_params(axis='both', which='major', labelsize=12, direction='inout')
#ax[0,0].tick_params(axis='both', which='minor', labelsize=12, direction='inout')               
    
# Example Legend
# Option 1: automatic based on labels
ax[0,0].legend(loc=(0.05, 0.05), fontsize=10, labelspacing=0.25, handlelength=1, handletextpad=0.25, borderpad=0, 
               frameon=False)
# Option 2: manually define legend
#leg_lines = []
#labels = ['plot1', 'plot2']
#label_colors = ['k', 'b']
#for nlabel, label in enumerate(labels):
#    line = Line2D([0,1],[0,1], color=label_colors[nlabel], linewidth=1)
#    leg_lines.append(line)
#ax[0,0].legend(leg_lines, labels, loc=(0.05,0.05), fontsize=10, labelspacing=0.25, handlelength=1, 
#               handletextpad=0.25, borderpad=0, frameon=False)

# Save figure
#  figures can be saved in any format (.jpg, .png, .pdf, etc.)
fig.set_size_inches(4, 4)
figure_fp = os.getcwd() + '/../Output/'
if os.path.exists(figure_fp) == False:
    os.makedirs(figure_fp)
figure_fn = 'example_plot_xy.png'
fig.savefig(figure_fp + figure_fn, bbox_inches='tight', dpi=300)


#%% MAP PLOT
xtick = 5
ytick = 5
xlabel = 'Longitude [$^\circ$]'
ylabel = 'Latitude [$^\circ$]'
labelsize = 12
west = -180
east = 180
south = -90
north = 90

rgiO1_shp_fn = os.getcwd() + '/../RGI/rgi60/00_rgi60_regions/00_rgi60_O1Regions.shp'
        

# Time, Latitude, Longitude
lons = [7.5,86.8,43.2,85.6,6.9,-69.9,10.6,-70.0,170]
lats = [46.0,28.0,42.8,28.2,45.8,-33.6,46.5,-30.1,-43]
values = [2665,5471,3050,4076,2030,3459,2623,4570,900]
#sizes = [100, 50, 20]

#var_change = temp_change

    
# Create the projection
fig, ax = plt.subplots(1, 1, figsize=(10,5), subplot_kw={'projection':cartopy.crs.PlateCarree()})
# Add country borders for reference
ax.add_feature(cartopy.feature.BORDERS, alpha=0.3, zorder=1)
ax.add_feature(cartopy.feature.COASTLINE)

# Set the extent
ax.set_extent([east, west, south, north], cartopy.crs.PlateCarree())    
# Label title, x, and y axes
ax.set_xticks(np.arange(east,west+1,xtick), cartopy.crs.PlateCarree())
ax.set_yticks(np.arange(south,north+1,ytick), cartopy.crs.PlateCarree())
ax.set_xlabel(xlabel, size=labelsize)
ax.set_ylabel(ylabel, size=labelsize)

# Add regions
#  facecolor='none' just plots the lines
#group_shp = cartopy.io.shapereader.Reader(rgiO1_shp_fn)
#group_feature = cartopy.feature.ShapelyFeature(group_shp.geometries(), cartopy.crs.PlateCarree(),
#                                               edgecolor='black', facecolor='grey', alpha=0.2, linewidth=1)
#ax.add_feature(group_feature,zorder=2)

# Add colorbar
cmap = 'RdYlBu_r'
norm = plt.Normalize(int(np.min(values)), np.ceil(np.max(values)))
var_label = 'Elevation [m a.s.l.]'
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm._A = []
plt.colorbar(sm, ax=ax, fraction=0.024, pad=0.01)
fig.text(1, 0.5, var_label, va='center', ha='center', rotation='vertical', size=labelsize)

ax.scatter(lons, lats, c=values, s=50, cmap=cmap, norm=norm, edgecolors='k', linewidth=0.5, alpha=0.9, zorder=3)

#ax.pcolormesh(lons, lats, var_change, cmap=cmap, norm=norm, zorder=2, alpha=0.8)            

# Title
#ax.set_title('[INSERT TITLE]')

# Save figure
fig.set_size_inches(6,4)
figure_fp = os.getcwd() + '/../Output/'
if os.path.exists(figure_fp) == False:
    os.makedirs(figure_fp)
fig_fn = 'example_plot_map.png'
fig.savefig(figure_fp + fig_fn, bbox_inches='tight', dpi=300)


#%%
import xarray as xr

ds1 = 