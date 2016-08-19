'''
    Raul valenzuela
'''

import Windprof2 as wp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
#import seaborn as sns
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
#with sns.axes_style("white"):
    

scale=1
plt.figure(figsize=(8*scale, 10*scale))

gs0 = gridspec.GridSpec(2, 1,hspace=0.15)

axes = range(2)    
axes[0] = plt.subplot(gs0[0],gid='(a) 23-24Jan01')
axes[1] = plt.subplot(gs0[1],gid='(b) 17Feb01')

#''' define ranges for tta and xpol in fraction of axis '''
#times={8:{ 'tta':[0.89,0.85], 'xpol':[0.93,0.13]},
#       9:{ 'tta':[0.86,0.40],  'xpol':[0.90,0.25]},
#       10:{'tta':[None,None], 'xpol':[0.41,0.13]},
#       11:{'tta':[None,None], 'xpol':[0.45,0.34]},
#       12:{'tta':[0.54,0.50],  'xpol':[0.56,0.38]},
#       13:{'tta':[0.85,0.77], 'xpol':[0.9,0.1]},
#       14:{'tta':[None,None], 'xpol':[0.58,0.41]}
#       }

labend={3:'25\n00',
        7: '18\n08',
       }



for c, ax in zip(case, axes):

    out = wp.make_arrays2(resolution=res,
                            surface=True,
                            case=str(c),
                            homedir=homedir,
                            interp_hgts=np.linspace(0.160, 3.74, 40))

    wspd, wdir, time, hgt = out


    if ax.get_gid() == '(a) 23-24Jan01':
        cbar = ax
        cbarinvi=False
    else:
        cbar = True
        cbarinvi=True

    ''' wind speed target '''
    ucomp=-wspd*sind(wdir)
    vcomp=-wspd*cosd(wdir)
    x = ucomp*sind(230)
    y = vcomp*cosd(230)
    upslope=-(x+y)
    wspd_target = vcomp
    


    ax, hcbar = wp.plot_time_height(ax=ax, 
                                      wspd=wspd_target,
                                      time=time,
                                      height=hgt,
                                      spd_range=[0, 28],
                                      spd_delta=2,
                                      cmap='nipy_spectral',
                                      cbar=(ax,cbarinvi),
                                      )
    
    wp.add_windstaff(wspd, wdir, time, hgt, color='k',ax=ax,
                     vdensity=1, hdensity=1)
    

    ''' determine xticks '''
    xtl = ax.get_xticklabels()
    xt  = ax.get_xticks()    

    if c == 3:
        nmod = 0
    else:
        nmod = 2

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


plt.show()

#fname='/home/raul/Desktop/windprof_panels_airborne.png'
#plt.savefig(fname, dpi=300, format='png',papertype='letter',
#            bbox_inches='tight')

