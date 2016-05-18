import Windprof2 as wp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os
import numpy as np
# import sys
from matplotlib.backends.backend_pdf import PdfPages
# from matplotlib.transforms import Bbox

# homedir = os.path.expanduser('~')
# homedir = '/Volumes/RauLdisk'
homedir = '/localdata'
topdf = False

case = range(8, 15)
res = 'coarse'
o = 'case{}_total_wind_{}.pdf'

''' creates plot with seaborn style '''
with sns.axes_style("white"):
    fig = plt.figure(figsize=(11, 8.5))
    gs = gridspec.GridSpec(3, 3)
    gs.update(left=0.1, right=0.9,
              top=0.9, bottom=0.1,
              wspace=0.05, hspace=0.25)
    axes = range(9)
    axes[0] = plt.subplot(gs[0, 0])
    axes[1] = plt.subplot(gs[0, 1])
    axes[2] = plt.subplot(gs[0, 2])
    axes[3] = plt.subplot(gs[1, 0])
    axes[4] = plt.subplot(gs[1, 1])
    axes[5] = plt.subplot(gs[1, 2])
    axes[6] = plt.subplot(gs[2, 0])
    axes[7] = plt.subplot(gs[2, 1])


for c, ax in zip(case, axes):

    wspd, wdir, time, hgt = wp.make_arrays2(resolution=res,
                                            surface=True,
                                            case=str(c),
                                            homedir=homedir)

    if c == 14:
        cbar = axes[7]
        axes[7].set_visible(False)
    else:
        cbar = False

    ax, hcbar = wp.plot_colored_staff(ax=ax, wspd=wspd, wdir=wdir, time=time,
                                      height=hgt, spd_range=[0, 20], spd_delta=2,
                                      vdensity=1, hdensity=1, cmap='nipy_spectral',
                                      cbar=cbar)

    ''' format xticklabels '''
    xtl = ax.get_xticklabels()
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

    ''' remove axes labels '''
    ax.set_xlabel('')
    ax.set_ylabel('')


# plt.subplots_adjust(left=0.1, right=0.9,
#                     top = 0.9, bottom=0.1,
#                     wspace=0.05,hspace=0.15)

if topdf:
    savename = o.format(str(c).zfill(2), res)
    wp_pdf = PdfPages(savename)
    wp_pdf.savefig()
    wp_pdf.close()

plt.show(block=False)
# plt.show()
# plt.close('all')
