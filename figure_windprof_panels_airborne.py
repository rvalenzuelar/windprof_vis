'''
    Raul valenzuela
'''

import Windprof2 as wp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import numpy as np
#from matplotlib.gridspec import GridSpecFromSubplotSpec as gssp
#from datetime import datetime

from matplotlib import rcParams
rcParams['xtick.major.pad'] = 3
rcParams['ytick.major.pad'] = 3

def cosd(array):
    return np.cos(np.radians(array))

def sind(array):
    return np.sin(np.radians(array))


homedir = '/localdata'
topdf = False

case = [3,7]
res = 'coarse'
o = 'case{}_total_wind_{}.pdf'

''' creates plot with seaborn style '''
with sns.axes_style("white"):
    sns.set_style('ticks',
              {'xtick.direction': u'in',
               'ytick.direction': u'in'}
              )
    

    scale=1
    plt.figure(figsize=(8*scale, 10*scale))
    
    gs0 = gridspec.GridSpec(2, 1,hspace=0.15)
    
    axes = range(2)    
    axes[0] = plt.subplot(gs0[0],gid='(a) 23-24Jan01')
    axes[1] = plt.subplot(gs0[1],gid='(b) 17Feb01')


labend={3:'25\n00',
        7: '18\n08',
       }



for c, ax in zip(case, axes):

    out = wp.make_arrays2(resolution=res,
                          add_surface=True,
                          case=str(c),
                          interp_hgts=np.linspace(0.160, 3.74, 40))

    wspd, wdir, time, hgt = out


    if ax.get_gid() == '(a) 23-24Jan01':
        cbar = ax
        cbarinvi=False
    else:
        cbar = True
        cbarinvi=True

    ''' wind speed target '''
    ucomp = -wspd*sind(wdir)
    vcomp = -wspd*cosd(wdir)
    x = ucomp*sind(230)
    y = vcomp*cosd(230)
    upslope = -(x+y)
    wspd_target = vcomp
    


    ax, hcbar = wp.plot_time_height(ax        = ax, 
                                    wspd      = wspd_target,
                                    time      = time,
                                    height    = hgt,
                                    spd_range = [-4, 32],
                                    spd_delta = 2,
                                    cmap      = 'nipy_spectral',
                                    cbar      = (ax,cbarinvi),
                                    kind      = 'pcolormesh',
                                    )
    
    wp.add_windstaff(wspd, wdir, time, hgt,
                     color     = (0.4,0.4,0.4),
                     ax        = ax,
                     vdensity  = 2,
                     hdensity  = 1,
                     head_size = 0.08,
                     tail_length = 5)
    

    ''' determine xticks '''
    xtl = ax.get_xticklabels()
    xt  = ax.get_xticks()    

    nmod = 0

    ''' add xtick labels '''
    newxtl = []
    for i, lb in enumerate(xtl):
        if np.mod(i, 6) in [nmod]:
            newxtl.append(lb.get_text())
        else:
            newxtl.append('')
    ax.set_xticklabels(newxtl)

for ax in axes:
    ax.text(0.05,0.9,ax.get_gid(),size=14,va='top',
            weight='bold',transform=ax.transAxes,
            backgroundcolor='w',clip_on=True)


#plt.show()

#fname='/home/raul/Desktop/fig_windprof_panels_airborne.png'
fname='/Users/raulv/Desktop/fig_windprof_panels_airborne.png'
plt.savefig(fname, dpi=300, format='png',papertype='letter',
            bbox_inches='tight')

