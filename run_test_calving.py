#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 11:56:59 2020

@author: davidrounce
"""

from pygem import oggm_compat

#def test_single_flowline_glacier_directory_with_calving():
#    rid = 'RGI60-01.03622'
#    gdir = oggm_compat.single_flowline_glacier_directory_with_calving(rid, k_calving=2)
#    diags = gdir.get_diagnostics()
#    print('Calving results:')
#    for k in ['calving_front_width', 'calving_flux', 'calving_thick',
#              'calving_free_board']:
#        print(k + ':', diags[k])
#    if do_plot:
#        from oggm import graphics
#        import matplotlib.pyplot as plt
#        f, (ax1, ax2) = plt.subplots(1, 2)
#        graphics.plot_googlemap(gdir, ax=ax1)
#        graphics.plot_inversion(gdir, ax=ax2)
#        plt.show()
        
rid = 'RGI60-01.03622'
gdir = oggm_compat.single_flowline_glacier_directory_with_calving(rid, k_calving=2)
diags = gdir.get_diagnostics()
print('Calving results:')
for k in ['calving_front_width', 'calving_flux', 'calving_thick',
          'calving_free_board']:
    print(k + ':', diags[k])

do_plot = True
if do_plot:
    from oggm import graphics
    import matplotlib.pyplot as plt
    f, (ax1, ax2) = plt.subplots(1, 2)
    graphics.plot_googlemap(gdir, ax=ax1)
    graphics.plot_inversion(gdir, ax=ax2)
    plt.show()