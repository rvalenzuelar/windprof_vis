import Windprof2 as wp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import numpy as np
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

case = range(8, 15)
res = 'coarse'
o = 'case{}_total_wind_{}.pdf'

''' creates plot with seaborn style '''
with sns.axes_style("white"):
    sns.set_style('ticks',
              {'xtick.direction': u'in',
               'ytick.direction': u'in'}
              )

    scale=1.3
    plt.figure(figsize=(8*scale, 11*scale))
    
    gs0 = gridspec.GridSpec(4, 1,
                            hspace = 0.25,
                            )
    
    gs00 = gssp(1, 2,
                subplot_spec=gs0[0],
                wspace=0.05
                )
    
    gs01 = gssp(1, 2,
                subplot_spec=gs0[1],
                wspace=0.05
                )

    gs02 = gssp(1, 2,
                subplot_spec=gs0[2],
                wspace=0.05
                )

    gs03 = gssp(1, 2,
                subplot_spec=gs0[3],
                wspace=0.05
                )

    axes = range(7)    
    axes[0] = plt.subplot(gs00[0],gid='(a) 12-14Jan03')
    axes[1] = plt.subplot(gs00[1],gid='(b) 21-23Jan03')
    axes[2] = plt.subplot(gs01[0],gid='(c) 15-16Feb03')
    axes[3] = plt.subplot(gs01[1],gid='(d) 09Jan04')
    axes[4] = plt.subplot(gs02[0],gid='(e) 02Feb04')
    axes[5] = plt.subplot(gs02[1],gid='(f) 16-18Feb04')
    axes[6] = plt.subplot(gs03[0],gid='(g) 25-26Feb04')


''' define ranges for tta and xpol in axis '''
times={8:{ 'tta':[7,  11, 2.14],   'xpol':[6, 63, 0.15]},
       9:{ 'tta':[10, 46, 0.24],   'xpol':[6, 54, 0.18]},
       10:{'tta':[None,None,None], 'xpol':[44, 63, 0.46]},
       11:{'tta':[None,None,None], 'xpol':[38, 47, 0.96]},
       12:{'tta':[32, 36, 2.2], 'xpol':[30, 46, 0.55]},
       13:{'tta':[10, 15, 1.8], 'xpol':[6, 66, 0.15]},
       14:{'tta':[None,None,None], 'xpol':[30, 42, 0.72]}
       }


''' last xlabel '''
labend={8:'15\n00',
       9: '24\n00',
       10:'17\n00',
       11:'11\n00',
       12:'04\n00',
       13:'19\n00',
       14:'27\n00'
       }

for c, ax in zip(case, axes):

    wspd, wdir, time, hgt = wp.make_arrays2(resolution = res,
                                            surface    = True,
                                            case       = str(c),
                                            homedir    = homedir)

    if c == 14:
        cbar_inv = False
    else:
        cbar_inv = True

    wspdMerid=-wspd*cosd(wdir);

    ax, hcbar = wp.plot_time_height(ax        = ax, 
                                    wspd      = wspdMerid,
                                    time      = time,
                                    height    = hgt,
                                    spd_range = [0, 30],
                                    spd_delta = 4,
                                    cmap      = 'nipy_spectral',
                                    cbar      = (ax,cbar_inv)
                                    )
    
    wp.add_windstaff(wspd, wdir, time, hgt, color='k',ax=ax,
                     vdensity=2, hdensity=2)
    
    
    
    ''' add arrow annotations '''    
    vpos = -3.11
    arrstyle = '|-|,widthA = 0.5,widthB = 0.5'
    ttacolor = (0,0,0)
    xplcolor = (0.7,0.7,0.7)
    
    if None not in times[c]['xpol']:
        st = times[c]['xpol'][0]
        en = times[c]['xpol'][1]
        frac = times[c]['xpol'][2]
        ax.annotate('',
                xy         = (st, vpos),
                xytext     = (en, vpos),
                xycoords   = 'data',
                textcoords = 'data',
                zorder     = 10000,
                arrowprops=dict(arrowstyle      = arrstyle,
                                ec              = xplcolor,
                                fc              = xplcolor,
                                linewidth       = 2)
                )
        
    if None not in times[c]['tta']:
        st = times[c]['tta'][0]
        en = times[c]['tta'][1]
        frac = times[c]['tta'][2]
        ax.annotate('',
                xy         = (st, vpos ),
                xytext     = (en, vpos),
                xycoords   = 'data',
                textcoords = 'data',
                zorder     = 10000,
                arrowprops=dict(arrowstyle = arrstyle,
                                ec         = ttacolor,
                                fc         = ttacolor,
                                linewidth  = 2)
                )

    ''' add arrow legend '''
    axes[2].annotate('TTA',
            xy=(20, 33), xycoords='data',
            xytext=(12, 32), textcoords='data',
            zorder=1,
            arrowprops=dict(arrowstyle = arrstyle,
                            ec         = ttacolor,
                            fc         = ttacolor,
                            linewidth  = 2))
    
    axes[2].annotate('X-pol',
            xy=(20, 29), xycoords='data',
            xytext=(12, 28), textcoords='data',
            zorder=1,
            arrowprops=dict(arrowstyle = arrstyle,
                            ec         = xplcolor,
                            fc         = xplcolor,
                            linewidth  = 2))

    if c not in [8]:
        ax.set_ylabel('')   
    
    if c not in [13]:
        ax.set_xlabel('')

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
    ytl = ax.get_yticklabels()
    newytl = []
    for i, lb in enumerate(ytl):
        if np.mod(i, 2) == 0:
            newytl.append(lb.get_text())    

    ''' format yticklabels '''
    yt  = ax.get_yticks()
    newyt = yt[::2]
    ax.set_yticks(newyt)            
    if c in [8, 10, 12,14]:
        ax.set_yticklabels(newytl)
    else:
        ax.set_yticklabels('')


    ''' add axes id '''
    ax.text(0.05,0.95,ax.get_gid(),size=14,va='top',
            weight='bold',transform=ax.transAxes,
            backgroundcolor='w',clip_on=True)



#plt.show()

fname='/home/raul/Desktop/windprof_panels.png'
plt.savefig(fname, dpi=100, format='png',papertype='letter',
            bbox_inches='tight')

