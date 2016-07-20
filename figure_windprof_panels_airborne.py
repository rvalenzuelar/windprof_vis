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


homedir = '/localdata'
topdf = False

case = [3,7]
res = 'coarse'
o = 'case{}_total_wind_{}.pdf'

''' creates plot with seaborn style '''
#with sns.axes_style("white"):
    

scale=1
plt.figure(figsize=(8*scale, 10*scale))

gs0 = gridspec.GridSpec(2, 1)

axes = range(2)    
axes[0] = plt.subplot(gs0[0],gid='(a) Jan 2001')
axes[1] = plt.subplot(gs0[1],gid='(b) Feb 2001')



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


    if ax.get_gid() == '(a) Jan 2001':
        cbar = ax
        cbarinvi=False
    else:
        cbar = True
        cbarinvi=True

    wspdMerid=-wspd*cosd(wdir);

    ax, hcbar = wp.plot_time_height(ax=ax, 
                                      wspd=wspdMerid,
                                      time=time, height=hgt,
                                      spd_range=[0, 28], spd_delta=2,
                                      cmap='nipy_spectral',
                                      cbar=cbar,
                                      cbarinvi=cbarinvi,
                                      timelabstep='6H'
                                      )
    
    wp.add_windstaff(wspd, wdir, time, hgt, color='k',ax=ax,
                     vdensity=1, hdensity=1)
    

for ax in axes:
    ax.text(0.05,0.9,ax.get_gid(),size=14,va='top',
            weight='bold',transform=ax.transAxes,
            backgroundcolor='w',clip_on=True)


#plt.show()

fname='/home/raul/Desktop/windprof_panels_airborne.png'
plt.savefig(fname, dpi=300, format='png',papertype='letter',
            bbox_inches='tight')

