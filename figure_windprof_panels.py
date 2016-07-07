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

def cosd(array):
    return np.cos(np.radians(array))


homedir = '/localdata'
topdf = False

case = range(8, 15)
res = 'coarse'
o = 'case{}_total_wind_{}.pdf'

''' creates plot with seaborn style '''
with sns.axes_style("white"):
    

    scale=1
    plt.figure(figsize=(11*scale, 8.5*scale))
    
    gs0 = gridspec.GridSpec(3, 1,hspace=0.35)
    
    gs00 = gssp(1, 3,
                subplot_spec=gs0[0],
                wspace=0.05)
    
    gs01 = gssp(1, 3,
                subplot_spec=gs0[1],
                wspace=0.05)

    gs02 = gssp(1, 3,
                subplot_spec=gs0[2],
                width_ratios=[1.2,1,1])

    axes = range(7)    
    axes[0] = plt.subplot(gs00[0],gid='(a) Jan 2003')
    axes[1] = plt.subplot(gs00[1],gid='(b) Jan 2003')
    axes[2] = plt.subplot(gs00[2],gid='(c) Feb 2003')
    axes[3] = plt.subplot(gs01[0],gid='(d) Jan 2004')
    axes[4] = plt.subplot(gs01[1],gid='(e) Feb 2004')
    axes[5] = plt.subplot(gs01[2],gid='(f) Feb 2004')
    axes[6] = plt.subplot(gs02[0],gid='(g) Feb 2004')


''' define ranges for tta and xpol in fraction of axis '''
times={8:{'tta':[0.93,0.85,10],'xpol':[0.9,0.1,-38]},
       9:{'tta':[0.95,0.4,10],'xpol':[0.85,0.25,-20]},
       10:{'tta':[None,None],'xpol':[0.4,0.1,10]},
       11:{'tta':[None,None],'xpol':[0.45,0.35,10]},
       12:{'tta':[0.65,0.5,10],'xpol':[0.55,0.4,10]},
       13:{'tta':[0.85,0.77,10],'xpol':[0.9,0.1,-38]},
       14:{'tta':[None,None],'xpol':[0.58,0.43,10]}
       }

labend={8:'15\n00',
       9: '24\n00',
       10:'17\n00',
       11:'11\n00',
       12:'04\n00',
       13:'19\n00',
       14:'27\n00'
       }

for c, ax in zip(case, axes):

    wspd, wdir, time, hgt = wp.make_arrays2(resolution=res,
                                            surface=True,
                                            case=str(c),
                                            homedir=homedir)

    if c == 14:
        cbar = ax
    else:
        cbar = False

    wspdMerid=-wspd*cosd(wdir);

    ax, hcbar = wp.plot_time_height(ax=ax, 
                                      wspd=wspdMerid,
                                      time=time, height=hgt,
                                      spd_range=[0, 30], spd_delta=4,
                                      cmap='nipy_spectral',
                                      cbar=cbar
                                      )
    
    wp.add_windstaff(wspd, wdir, time, hgt, color='k',ax=ax,
                     vdensity=2, hdensity=2)
    
    
    
    ''' add arrow annotations '''    
    scale = 4.1 # use for adjust png output
    alpha = 0.6
    if None not in times[c]['xpol']:
        st = times[c]['xpol'][0]
        en = times[c]['xpol'][1]
#        h = times[c]['xpol'][2]*scale
        h = np.abs(st-en)*(-66.6667*scale)+(15.3333*scale)
        connectst = 'bar,armA={},armB={}'.format(h,h),
        ax.annotate('',
                xy=(st, -0.1 ),
                xycoords='axes fraction',
                xytext=(en, -0.1),
                textcoords='axes fraction',
                zorder=1,
                arrowprops=dict(arrowstyle='<->',
                                connectionstyle=connectst[0],
                                ec=(0.8,0.8,0),
                                fc=(0.8,0.8,0),
                                linewidth=2))
    if None not in times[c]['tta']:
        st = times[c]['tta'][0]
        en = times[c]['tta'][1]
#        h = times[c]['tta'][2]*scale       
        h = np.abs(st-en)*(-66.6667*scale)+(15.3333*scale)
        connectst = 'bar,armA={},armB={}'.format(h,h),
        ax.annotate('',
                xy=(times[c]['tta'][0], -0.1 ),
                xycoords='axes fraction',
                xytext=(times[c]['tta'][1],-0.1),
                textcoords='axes fraction',
                zorder=1,
                arrowprops=dict(arrowstyle="<->",
                                connectionstyle=connectst[0],
                                ec=(0,0,0,alpha),
                                fc=(0,0,0,alpha),
                                linewidth=2))

    if c not in [8]:
        ax.set_ylabel('')   
    
    if c not in [13]:
        ax.set_xlabel('')

    ''' format xticklabels '''
    xtl = ax.get_xticklabels()
    xt = ax.get_xticks()    
    off = 12-np.mod(xt[-1],12)
    newticks = range(xt[-1]+off+1)

    for n in range(off):
        if n ==  off-1:
            xtl.append(labend[c])
        else:
            xtl.append('')
    ax.set_xticks(newticks)
    newxtl = []
    for i, lb in enumerate(xtl):
        if np.mod(i, 12) == 0:
            newxtl.append(lb)
        else:
            newxtl.append('')
    ax.set_xticklabels(newxtl)

    ''' format yticklabels '''
    if c in [8, 11, 14]:
        ytl = ax.get_yticklabels()
        newytl = []
        for i, lb in enumerate(ytl):
            if np.mod(i, 2) == 0:
                newytl.append(lb)
            else:
                newytl.append('')
        ax.set_yticklabels(newytl)
    else:
        ax.set_yticklabels('')


for ax in axes:
    ax.text(0.05,0.95,ax.get_gid(),size=14,va='top',
            weight='bold',transform=ax.transAxes,
            backgroundcolor='w',clip_on=True)


#plt.show()

fname='/home/raul/Desktop/windprof_panels.png'
plt.savefig(fname, dpi=300, format='png',papertype='letter',
            bbox_inches='tight')

