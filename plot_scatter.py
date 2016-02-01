import Windprof2 as wp
import matplotlib.pyplot as plt 


f, ax1 = plt.subplots(figsize=(11,8.5))


c=9
wspd,wdir,time,hgt = wp.make_arrays(	resolution='coarse',
										surface=True,
										case=str(c),
										period=False)

wp.plot_scatter(ax=ax1,wdir=wdir)

plt.show(block=False)





