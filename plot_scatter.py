import Windprof2 as wp
import matplotlib.pyplot as plt 
# from mpl_toolkits.mplot3d import Axes3D

reload(wp)

# f, ax1 = plt.subplots(figsize=(11,8.5))

# f=plt.figure(figsize=(11,8.5))
# ax1 = f.add_subplot(111, projection='3d')


case=14
wspd,wdir,time,hgt = wp.make_arrays(	resolution='fine',
										surface=True,
										case=str(case),
										period=False)

wp.plot_scatter(wdir=wdir,hgt=hgt,title='Case '+str(case).zfill(2))
plt.show(block=False)





