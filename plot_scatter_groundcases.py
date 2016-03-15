import Windprof2 as wp
import matplotlib.pyplot as plt
# import numpy as np


fig, axes = plt.subplots(3, 3, figsize=(11, 9), sharex=True, sharey=True)
axes = axes.flatten()
wd_lim_surf = 125
wd_lim_aloft = 170
mAGL, color = [120, 'navy']
# mAGL,color=[500,'green']
for c, ax in zip(range(12, 13), axes):

    wspd, wdir, time, hgt = wp.make_arrays(
        resolution='coarse', surface=True, case=str(c))
    wp.plot_scatter2(ax=ax, wdir=wdir, hgt=hgt, time=time, mAGL=mAGL,
                     lim_surf=wd_lim_surf, lim_aloft=wd_lim_aloft, color=color)
    ax.text(0., 0.05, 'Case '+str(c).zfill(2), transform=ax.transAxes)


axes[0].text(
    0, 1.05, 'Altitude: '+str(mAGL) + 'm AGL', transform=axes[0].transAxes)
axes[6].set_xlabel('wind direction surface')
axes[6].text(wd_lim_surf, -20, str(wd_lim_surf), ha='center')
axes[6].text(0, -20, '0', ha='center')
axes[0].set_ylabel('wind direction aloft')
axes[0].text(380, wd_lim_aloft, str(wd_lim_aloft), va='center', rotation=90)
axes[0].text(380, 0, '0', va='center', rotation=90)
plt.subplots_adjust(bottom=0.05, top=0.95, hspace=0.05, wspace=0.05)
fig.delaxes(axes[7])
fig.delaxes(axes[8])
plt.show(block=False)
