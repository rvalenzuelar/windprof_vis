import Windprof2 as wp
import matplotlib.pyplot as plt
import seaborn as sns


case=[8]

''' creates plot with seaborn style '''
with sns.axes_style("white"):
	f, ax1 = plt.subplots(figsize=(11,8.5))


for c in case:
	# period = get_period(c)
	period = False
	wspd,wdir,time,hgt = wp.make_arrays(	resolution='coarse',
											surface=True,
											case=str(c),
											period=period)
	wp.plot_time_height(ax=ax1, wspd=wspd, time=time, 
						height=hgt, vrange=[0,25],
						cname='YlGnBu_r',title='Total wind speed')
	# wp.add_windstaff(wspd,wdir,time,hgt,ax=ax1, color=(1,0.2,0))
	# wp.add_soundingTH('bvf_moist',str(c),ax=ax1,wptime=time,wphgt=hgt,sigma=1)

	# f, ax2 = plt.subplots(figsize=(11,8.5))
	# plot_vertical_shear(ax=ax2, wind=wdir, time=time, height=hgt)


	# plt.show(block=False)
	plt.show()