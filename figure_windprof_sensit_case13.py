# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 17:03:35 2016

@author: raulv
"""


import Windprof2 as wp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpecFromSubplotSpec as gssp
#from datetime import datetime

from matplotlib import rcParams
rcParams['xtick.major.pad'] = 3
rcParams['ytick.major.pad'] = 3
rcParams['xtick.labelsize'] = 15
rcParams['ytick.labelsize'] = 15
rcParams['axes.labelsize'] = 15
rcParams['legend.handletextpad'] = 0.1
rcParams['legend.handlelength'] = 1.
rcParams['legend.fontsize'] = 15
rcParams['mathtext.default'] = 'sf'

def cosd(array):
    return np.cos(np.radians(array))


homedir = '/localdata'
topdf = False

case = range(13,14)
res = 'coarse'
o = 'case{}_total_wind_{}.pdf'

''' creates plot with seaborn style '''
with sns.axes_style("white"):
    sns.set_style('ticks',
              {'xtick.direction': u'in',
               'ytick.direction': u'in'}
              )

    scale=1.3
    plt.figure(figsize=(8*scale, 6*scale))
    
    gs0 = gridspec.GridSpec(1, 1,
                            hspace = 0.25,
                            )
    
    gs00 = gssp(1, 1,
                subplot_spec=gs0[0],
                )
    
    axes = [plt.subplot(gs00[0],gid='16-18Feb04')]

wprof_range = ('2004-02-16 00:00','2004-02-18 23:00')


''' end time with one more hour to cover all previous hour '''
tta_range = (
                ('2004-02-16 09:00','2004-02-16 16:00'),
                ('2004-02-16 09:00','2004-02-16 16:00'),
                ('2004-02-16 09:00','2004-02-16 16:00'),
                ('2004-02-16 13:00','2004-02-16 14:00'),
                ('2004-02-16 11:00','2004-02-16 15:00'),
                ('2004-02-16 07:00','2004-02-16 17:00'),
               )

''' define ranges for tta and xpol time annotation '''
times = [
            {'tta':[None,None, 1.8]},
            {'tta':[None,None, 1.8]},
            {'tta':[None,None, 1.8]},
            {'tta':[None,None, 1.8]},
            {'tta':[None,None, 1.8]},
            {'tta':[None,None, 1.8]},
        ]

wp_st = wprof_range[0]
wp_en = wprof_range[1]
for tta,time in zip(tta_range, times):
    drange = pd.date_range(start=wp_st,end=wp_en,freq='1H')
    time['tta'][0] = np.where(drange==tta[0])[0]
    time['tta'][1] = np.where(drange==tta[1])[0]
times[3]['tta'][0] = None

''' last xlabel '''
labend={
       13:'19\n00',
       }

for c, ax in zip(case, axes):

    wspd, wdir, time, hgt = wp.make_arrays2(resolution  = res,
                                            add_surface = True,
                                            case        = str(c),
                                            )
    cbar_inv = False

    wspdMerid=-wspd*cosd(wdir);

    if c == 13:
        foo = wspdMerid

    ax, hcbar = wp.plot_time_height(ax        = ax, 
                                    wspd      = wspdMerid,
                                    time      = time,
                                    height    = hgt,
                                    spd_range = [-5, 30],
                                    spd_delta = 4,
                                    cmap      = 'nipy_spectral',
                                    cbar      = (ax,cbar_inv)
                                    )
    
    wp.add_windstaff(wspd, wdir, time, hgt,
                     color=(0.6,0.6,0.6),
                     ax=ax,
                     vdensity=2,
                     hdensity=2)
    
    
    
    ''' add arrow annotations '''    
    vpos1 = -3.11
    vpos2 = -2.0
    arrstyle = '|-|,widthA = 0.5,widthB = 0.5'
    ttacolor = (0,0,0)
    xplcolor = (0.7,0.7,0.7)
            
    for t in times:
        if t['tta'][0] is None:
            ax.text(t['tta'][1]-0.5, vpos2,'None',fontsize=15,
                    weight='bold')
        else:
            st = t['tta'][0]-0.5
            en = t['tta'][1]+0.5
            frac = t['tta'][2]
            ax.annotate('',
                    xy         = (st, vpos2),
                    xytext     = (en, vpos2),
                    xycoords   = 'data',
                    textcoords = 'data',
                    zorder     = 10000,
                    arrowprops=dict(arrowstyle = arrstyle,
                                    ec         = ttacolor,
                                    fc         = ttacolor,
                                    linewidth  = 2)
                    )
        vpos2 -= 2.0
        

    ''' determine xticks '''
    xtl = ax.get_xticklabels()
    xt  = ax.get_xticks()    
    off = 12-np.mod(xt[-1],12)  #offset from 12 hr
    nxt = xt[-1]+off+2
    newxt = range(0,nxt,12)
    ax.set_xticks(newxt)

    ''' append last xlabel '''
    for n in range(off):
        if n ==  off-1:
            xtl.append(labend[c])
        else:
            xtl.append('')
    
    ''' add xtick labels '''
    newxtl = []
    for i, lb in enumerate(xtl):
        if np.mod(i, 12) == 0:
            if isinstance(lb,str):
                newxtl.append(lb)
            else:
                newxtl.append(lb.get_text())
    ax.set_xticklabels(newxtl)


    ''' y labels '''
#    ytl = ax.get_yticklabels()
#    newytl = []
#    for i, lb in enumerate(ytl):
#        if np.mod(i, 2) == 0:
#            newytl.append(lb.get_text())    

    ''' format yticklabels '''
#    yt  = ax.get_yticks()
#    newyt = yt[::2]
#    ax.set_yticks(newyt)            
#    if c in [8, 10, 12,14]:
#        ax.set_yticklabels(newytl)
#    else:
#        ax.set_yticklabels('')


    ''' add axes id '''
    ax.text(0.05,0.95,ax.get_gid(),size=14,va='top',
            weight='bold',transform=ax.transAxes,
            backgroundcolor='w',clip_on=True)

    ax.set_ylim([-15,43])

plt.show()

##fname='/home/raul/Desktop/fig_windprof_panels.png'
#fname='/Users/raulv/Desktop/fig_windprof_panels.png'
#plt.savefig(fname, dpi=100, format='png',papertype='letter',
#            bbox_inches='tight')