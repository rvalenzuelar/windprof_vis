# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 17:03:35 2016

@author: raulv
"""

import Windprof2 as wp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sbn
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpecFromSubplotSpec as gssp
from rv_utilities import add_colorbar, discrete_cmap
from datetime import datetime
import Meteoframes as mf

# sbn.reset_defaults()
sbn.reset_orig()

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

case = range(13, 14)
res = 'coarse'
o = 'case{}_total_wind_{}.pdf'

surf_file = '/Users/raulvalenzuela/Data/SURFACE/case13/czc04047.met'
czd = mf.parse_surface(surf_file)

surf_file = '/Users/raulvalenzuela/Data/SURFACE/case13/bby04047.met'
bby = mf.parse_surface(surf_file)

czdh = czd.preciph[~czd.preciph.isnull()]
bbyh = bby.preciph[~bby.preciph.isnull()]

''' creates plot with seaborn style '''
# with sns.axes_style("white"):
#     sns.set_style('ticks',
#               {'xtick.direction': u'in',
#                'ytick.direction': u'in'}
#               )

scale = 1.3
plt.figure(figsize=(8 * scale, 6 * scale))

gs0 = gridspec.GridSpec(2, 1,
                        height_ratios=[2, 1],
                        hspace=0.05,
                        )

axes = [plt.subplot(gs0[0]),
        plt.subplot(gs0[1])
        ]

wprof_range = ('2004-02-16 00:00', '2004-02-18 23:00')

tta_range = (
    ('2004-02-16 11:00', '2004-02-16 15:00'),
    ('2004-02-16 09:00', '2004-02-16 16:00'),
    ('2004-02-16 07:00', '2004-02-16 17:00'),
    ('2004-02-16 09:00', '2004-02-16 16:00'),
)

''' end time with one more hour to cover all previous hour '''

''' define ranges for tta and xpol time annotation '''
times = [
    {'tta': [None, None, 1.8]},
    {'tta': [None, None, 1.8]},
    {'tta': [None, None, 1.8]},
    {'tta': [None, None, 1.8]},
    {'tta': [None, None, 1.8]},
    {'tta': [None, None, 1.8]},
]

wp_st = wprof_range[0]
wp_en = wprof_range[1]
for tta, time in zip(tta_range, times):
    drange = pd.date_range(start=wp_st, end=wp_en, freq='1H')
    time['tta'][0] = np.where(drange == tta[0])[0]
    time['tta'][1] = np.where(drange == tta[1])[0]

params = [
    'WD$<140$, nh$\geq1\,(2,4)$',
    'WD$<150$, nh$\geq1$',
    'WD$<160$, nh$\geq1$',
    'WD$<150$, nh$\geq2$',
]

for c, ax in zip(case, axes):

    wspd, wdir, time, hgt = wp.make_arrays2(resolution=res,
                                            add_surface=True,
                                            case=str(c),
                                            )
    cbar_inv = False

    wspdMerid = -wspd * cosd(wdir)

    if c == 13:
        foo = wspdMerid

    ax, hcbar, im = wp.plot_time_height(ax=ax,
                                        wspd=wspdMerid,
                                        time=time,
                                        height=hgt,
                                        spd_range=[0, 30],
                                        spd_delta=2,
                                        cmap='jet',
                                        cbar=(ax, cbar_inv),
                                        cbar_label='meridional wind $['
                                                   'm\ s^{-1}]$'
                                        )

    wp.add_windstaff(wspd, wdir, time, hgt,
                     # color=(0.6,0.6,0.6),
                     color='k',
                     ax=ax,
                     vdensity=1,
                     hdensity=0,
                     head_size=0.08,
                     tail_length=5
                     )

    ''' add arrow annotations '''
    vpos1 = -3.11
    vpos2 = -2.3
    arrstyle = '|-|,widthA = 0.5,widthB = 0.5'
    ttacolor = (0, 0, 0)
    xplcolor = (0.7, 0.7, 0.7)

    for t, p in zip(times, params):
        if t['tta'][0] is None:
            vpos2 += 2.3
        else:
            st = t['tta'][0]
            en = t['tta'][1]
            frac = t['tta'][2]
            ax.annotate('',
                        xy=(st, vpos2),
                        xytext=(en, vpos2),
                        xycoords='data',
                        textcoords='data',
                        zorder=10000,
                        arrowprops=dict(arrowstyle=arrstyle,
                                        ec=ttacolor,
                                        fc=ttacolor,
                                        linewidth=2)
                        )
            ax.text(21, vpos2, p, ha='left', va='center')
        vpos2 -= 2.3

    " isolated hours "
    for vp in [-4.3, -6.6]:
        for st, en in [[0, 1], [3, 4]]:
            ax.annotate('',
                        xy=(st, vp),
                        xytext=(en, vp),
                        xycoords='data',
                        textcoords='data',
                        zorder=10000,
                        arrowprops=dict(arrowstyle=arrstyle,
                                        ec=ttacolor,
                                        fc=ttacolor,
                                        linewidth=2)
                        )

    ''' get xtick labels '''
    xtl = ax.get_xticklabels()
    xl = ax.get_xlabel()

    ''' set axis '''
    ax.set_xticks(range(0, 23, 3))
    ax.set_xlim([24, 0])
    ax.set_xticklabels('')

    ax.set_ylim([-13, 30])
    ax.set_xlim([24, 0])

axes[0].text(0.05, 0.93, '(a)',
             ha='right', weight='bold', fontsize=15,
             color='k', backgroundcolor='w',
             transform=axes[0].transAxes)

axes[0].text(0.5, 0.93, 'Wind Profiler at BBY',
             ha='center', weight='bold', fontsize=15,
             color='k', backgroundcolor='w',
             transform=axes[0].transAxes)

axes[0].text(0.95, 0.93, '16 February 2004',
             ha='right', weight='bold', fontsize=15,
             backgroundcolor='w',
             transform=axes[0].transAxes)

''' rainfall '''
cmap = discrete_cmap(7, base_cmap='Set1')

axes[1].plot(np.arange(0.5, 24.5), czdh.values, '-o', lw=2,
             color=cmap(1), label='CZD')

axes[1].plot(np.arange(0.5, 24.5), bbyh.values, '-o', lw=2,
             color=cmap(0), label='BBY')

axes[1].legend(numpoints=1, loc=7)

add_colorbar(axes[1], im, invisible=True)

axes[1].text(0.05, 0.88, '(b)',
             ha='right', weight='bold', fontsize=15,
             transform=axes[1].transAxes)

axes[1].text(0.5, 0.88, 'Rainfall',
             ha='center', weight='bold', fontsize=15,
             color='k', backgroundcolor='w',
             transform=axes[1].transAxes)

axes[1].invert_xaxis()

axes[1].set_xticks(range(0, 26, 3))
axes[1].set_xlim([24, 0])

''' add xtick labels '''
xt = axes[1].get_xticks()
newxtl = []
for i, lb in enumerate(xtl):
    if np.mod(i, 3) == 0:
        if isinstance(lb, str):
            newxtl.append(lb)
        else:
            newxtl.append(lb.get_text())
axes[1].set_xticklabels(newxtl)

axes[1].set_xlabel(xl)
ax.set_xlabel('')
axes[1].set_ylabel('rain rate [mm $h^{-1}$]')

plt.subplots_adjust(bottom=0.13, top=0.98)

# plt.show()


place = '/Users/raulvalenzuela/Documents/'
fname = 'sensit_case13.png'
plt.savefig(place+fname, dpi=100, format='png',papertype='letter',
           bbox_inches='tight')
