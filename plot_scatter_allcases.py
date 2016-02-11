import Windprof2 as wp
import matplotlib.pyplot as plt 
import numpy as np


reload(wp)


# case=13
# wspd,wdir,time,hgt = wp.make_arrays(	resolution='fine',
# 										surface=True,
# 										case=str(case),
# 										period=False)

# wp.plot_scatter(wdir=wdir,hgt=hgt,title='Case '+str(case).zfill(2))
# plt.show(block=False)

fig, axes = plt.subplots(5,3, figsize=(11,15), sharex=True, sharey=True)
axes=axes.flatten()
vline=130
hline=170
mAGL,color=[120,'navy']
# mAGL,color=[500,'green']
for c, ax in zip(range(1,15), axes):
	
	wspd,wdir,time,hgt = wp.make_arrays(resolution='fine', surface=True,
										case=str(c),period=False)
	wp.plot_scatter2(ax=ax,wdir=wdir,hgt=hgt,mAGL=mAGL,vline=vline,hline=hline,color=color)
	ax.text(0.,0.05,'Case '+str(c).zfill(2),transform=ax.transAxes)



axes[0].text(0,1.05,'Altitude: '+str(mAGL) +'m AGL',transform=axes[0].transAxes)
axes[13].set_xlabel('wind direction surface')
axes[13].text(vline,-20,str(vline),ha='center')
axes[13].text(0,-20,'0',ha='center')
axes[6].set_ylabel('wind direction aloft')
axes[6].text(380,hline,str(hline),va='center',rotation=90)
axes[6].text(380,0,'0',va='center', rotation=90)
plt.subplots_adjust(bottom=0.05,top=0.95,hspace=0.05,wspace=0.05)
fig.delaxes(axes[-1])
plt.show(block=False)




