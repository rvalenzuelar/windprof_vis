"""
	Plot NOAA wind profiler. 
	Files  have extension HHw, where HH is UTC hour

	Raul Valenzuela
	August, 2015
"""


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

import numpy as np
import os 
import sys

import Meteoframes as mf

from datetime import datetime, timedelta
from matplotlib import colors
import plotSoundTH as ps
from scipy.ndimage.filters import gaussian_filter
from scipy.interpolate import interp1d
from mpl_toolkits.axes_grid1 import make_axes_locatable

''' set directory and input files '''
local_directory='/home/rvalenzuela/'
# local_directory='/Users/raulv/Documents/'

base_directory=local_directory + 'WINDPROF'

def main():

	for c in range(10,15):
		wpfiles = get_filenames(str(c))
		period = get_period(c)
		wspd,wdir,time,hgt = make_arrays(files= wpfiles, 
												resolution='coarse',
												surface=True,
												case=str(c),
												period=period)
		ax=plot_time_height(wspd, time, hgt, vrange=[0,35],cname='YlGnBu_r',title='Total wind speed')
		palette = sns.color_palette()
		color=palette[2]
		add_windstaff(wspd,wdir,time,hgt,ax=ax, color=color)
		plt.show(block=False)


def get_period(case):

	reqdates={ '1': {'ini':[1998,1,18,15],'end':[1998,1,18,20]},
			'2': {'ini':[1998,1,26,4],'end':[1998,1,26,9]},
			'3': {'ini':[2001,1,23,21],'end':[2001,1,24,2]},
			'4': {'ini':[2001,1,25,15],'end':[2001,1,25,20]},
			'5': {'ini':[2001,2,9,10],'end':[2001,2,9,15]},
			'6': {'ini':[2001,2,11,3],'end':[2001,2,11,8]},
			'7': {'ini':[2001,2,17,17],'end':[2001,2,17,22]},
			'8': {'ini':[2003,1,12,15],'end':[2003,1,12,20]},
			'9': {'ini':[2003,1,22,18],'end':[2003,1,22,23]},
			'10': {'ini':[2003,2,16,0],'end':[2003,2,16,5]},
			'11': {'ini':[2004,1,9,17],'end':[2004,1,9,22]},
			'12': {'ini':[2004,2,2,12],'end':[2004,2,2,17]},
			'13': {'ini':[2004,2,17,14],'end':[2004,2,17,19]},
			'14': {'ini':[2004,2,25,8],'end':[2004,2,25,13]}
			}

	return reqdates[str(case)]

def plot_single():

	print base_directory
	usr_case = raw_input('\nIndicate case number (i.e. 1): ')
	wprof_resmod = raw_input('\nIndicate resolution mode (f = fine; c = coarse): ')

	''' get wind profiler file names '''
	wpfiles = get_filenames(usr_case)
	# print wpfiles
	''' make profile arrays '''
	if wprof_resmod == 'f':
		res='fine' # 60 [m]
	elif wprof_resmod == 'c':
		res='coarse' # 100 [m]
	else:
		print 'Error: indicate correct resolution (f or c)'
	wspd,wdir,time,hgt = make_arrays(files= wpfiles, resolution=res,surface=True,case=usr_case)

	''' make time-height section of total wind speed '''
	ax=plot_time_height(wspd, time, hgt, vrange=[0,20],cname='YlGnBu_r',title='Total wind speed')
	l1 = 'BBY wind profiler - Total wind speed (color coded)'

	''' add wind staffs '''
	palette = sns.color_palette()
	color=palette[2]
	u,v = add_windstaff(wspd,wdir,time,hgt,ax=ax, color=color)

	''' add balloon sounding time-height section '''
	# add_soundingTH('bvf_dry',usr_case,ax=ax,wptime=time,sigma=3)
	# # l2 = '\nBBY balloon soundings - Relative humidity (%, contours)'
	# # l2 = '\nBBY balloon soundings - Air pressure (hPa, contours)'
	# # l2 = '\nBBY balloon soundings - Mixing ratio '+r'($g kg^{-1}$, contours)'
	# # l2 = '\nBBY balloon soundings - Air temperature '+r'($^\circ$C, contours)'
	# # l2 = '\nBBY balloon soundings - Brunt-Vaisala freq moist '+r'(x$10^{-4} [s^{-2}]$, contours)'
	# l2 = '\nBBY balloon soundings - Brunt-Vaisala freq dry '+r'(x$10^{-4} [s^{-2}]$, contours)'
	# l3 = '\nDate: '+ time[0].strftime('%Y-%m')
	# plt.suptitle(l1+l2+l3)

	# ''' make time-height section of meridional wind speed '''
	# ax=plot_time_height(v, time, hgt, vrange=range(-20,22,2),cname='BrBG',title='Meridional component')
	# add_windstaff(wspd,wdir,time,hgt,ax=ax, color=color)
	
	# ''' make time-height section of zonal wind speed '''
	# ax=plot_time_height(u, time, hgt, vrange=range(-20,22,2),cname='BrBG',title='Zonal component')
	# add_windstaff(wspd,wdir,time,hgt,ax=ax, color=color)

	plt.show(block=False)
	# plt.show()


def get_filenames(usr_case):

	case='case'+usr_case.zfill(2)
	casedir=base_directory+'/'+case
	out=os.listdir(casedir)
	out.sort()
	file_sound=[]
	for f in out:
		if f[-1:]=='w': 
			file_sound.append(casedir+'/'+f)
	return file_sound

def get_surface_data(usr_case):

	''' set directory and input files '''
	base_directory=local_directory + 'SURFACE'
	case='case'+usr_case.zfill(2)
	casedir=base_directory+'/'+case
	out=os.listdir(casedir)
	out.sort()
	files=[]
	for f in out:
		if f[-3:]=='met': 
			files.append(f)
	file_met=[]
	for f in files:
		if f[:3]=='bby':
			file_met.append(casedir+'/'+f)
	name_field=['press','temp','rh','wspd','wdir','precip','mixr']
	if usr_case in ['1','2']:
		index_field=[3,4,10,5,6,11,13]
	elif usr_case in ['3','4','5','6','7']: 
		index_field=[3,6,9,10,12,17,26]
	else:
		index_field=[3,4,5,6,8,13,15]

	locname='Bodega Bay'
	locelevation = 15 # [m]

	df=[]
	for f in file_met:
		df.append(mf.parse_surface(f,index_field,name_field,locelevation))

	if len(df)>1:
		surface=pd.concat(df)
	else:
		surface=df[0]

	return surface

def plot_time_height(spd_array,time_array,height_array,**kwargs):

	''' NOAA wind profiler files after year 2000 indicate 
	the start time of averaging period; so a timestamp of
	13 UTC indicates average between 13 and 14 UTC '''

	vrange=kwargs['vrange']
	cname=kwargs['cname']
	title=kwargs['title']

	''' creates plot with seaborn style '''
	with sns.axes_style("ticks"):
		f, ax = plt.subplots(figsize=(11,8.5))

	''' make a color map of fixed colors '''
	snsmap=sns.color_palette(cname, 24)
	cmap = colors.ListedColormap(snsmap[2:])
	if len(vrange) == 2:
		vdelta=2
		bounds=range(vrange[0],vrange[1]+vdelta, vdelta)
		vmin=vrange[0]
		vmax=vrange[1]
	else:
		bounds=vrange
		vmin=vrange[0]
		vmax=vrange[-1]
	norm = colors.BoundaryNorm(bounds, cmap.N)

	nrows,ncols = spd_array.shape
	img = plt.imshow(spd_array, interpolation='nearest', origin='lower',
						cmap=cmap, norm=norm,vmin=vmin,vmax=vmax,
						extent=[0,ncols,0,nrows],aspect='auto') #extent helps to make correct timestamp
	
	divider = make_axes_locatable(ax)
	cax = divider.append_axes("right", size="2%", pad=0.09)
	cbar = plt.colorbar(img, cax=cax)

	format_xaxis(ax,time_array)
	format_yaxis(ax,height_array)
	ax.invert_xaxis()
	ax.set_ylabel('Range hight [km]')
	ax.set_xlabel(r'$\Leftarrow$'+' Time [UTC]')

	plt.draw()

	return ax

def make_arrays(**kwargs):

	file_sound = kwargs['files']
	resolution = kwargs['resolution']
	surf = kwargs['surface']

	wp=[] 
	ncols=0 # number of timestamps
	for f in file_sound:
		if resolution=='fine':
			wp.append(mf.parse_windprof(f,'fine'))
		elif resolution=='coarse':
			wp.append(mf.parse_windprof(f,'coarse'))
		else:
			print 'Error: resolution has to be "fine" or "coarse"'
		ncols+=1

	''' creates 2D arrays with spd and dir '''
	nrows = len(wp[0].HT.values) # number of altitude gates (fine same as coarse)
	hgt = wp[0].HT.values
	wspd = np.empty([nrows,ncols])
	wdir = np.empty([nrows,ncols])
	timestamp = []
	for i,p in enumerate(wp):
		timestamp.append(p.timestamp)	
		''' fine resolution '''
		spd=p.SPD.values
		wspd[:,i]=spd
		dirr=p.DIR.values
		wdir[:,i]=dirr

	''' add 2 bottom rows for adding surface obs '''
	bottom_rows=2
	na = np.zeros((bottom_rows,ncols))
	na[:] = np.nan
	wspd = np.flipud(np.vstack((np.flipud(wspd),na)))
	wdir = np.flipud(np.vstack((np.flipud(wdir),na)))
	if surf:
		''' make surface arrays '''
		case = kwargs['case']
		surface = get_surface_data(case)
		hour=pd.TimeGrouper('H')
		surf_wspd = surface.wspd.groupby(hour).mean()	
		surf_wdir = surface.wdir.groupby(hour).mean()
		surf_st = np.where(np.asarray(timestamp) == surf_wspd.index[0])[0][0]
		surf_en = np.where(np.asarray(timestamp) == surf_wspd.index[-1])[0][0]
		wspd[0,surf_st:surf_en+1]=surf_wspd
		wdir[0,surf_st:surf_en+1]=surf_wdir

	hgt = np.hstack(([0.,0.05],hgt))

	''' add last column for 00 UTC of last date '''
	add_left=1
	nrows, _ = wspd.shape
	na = np.zeros((nrows,add_left))
	na[:] = np.nan
	wspd =np.hstack((wspd,na))
	wdir =np.hstack((wdir,na))
	timestamp.append(timestamp[-1]+timedelta(hours=1))

	if 'period' in kwargs:
		period=kwargs['period']
		time=np.asarray(timestamp)
		ini = datetime(*(period['ini'] + [0,0]))
		end = datetime(*(period['end'] + [0,0]))
		idx = np.where((time>= ini) & (time <= end) )[0]
		return wspd[:,idx], wdir[:,idx], time[idx],hgt
	
	return wspd,wdir,timestamp,hgt

def add_windstaff(wspd,wdir,time,hgt,**kwargs):

	if kwargs and kwargs['color']:
		color=kwargs['color']
	else:
		color='k'
	ax = kwargs['ax'] 

	''' derive U and V components '''
	U=-wspd*np.sin(wdir*np.pi/180.)
	V=-wspd*np.cos(wdir*np.pi/180.)
	x=np.array(range(len(time)))+0.5 # wind staff in the middle of pixel
	y=np.array(range(hgt.size))+0.5 # wind staff in the middle of pixel
	X=np.tile(x,(y.size,1)) # repeats x y.size times to make 2D array
	Y=np.tile(y,(x.size,1)).T #repeates y x.size times to make 2D array
	Uzero = U-U
	Vzero = V-V

	ax.barbs(X,Y,U,V,color=color, sizes={'height':0},length=5,linewidth=0.5,barb_increments={'half':1})
	ax.barbs(X,Y,Uzero,Vzero,color=color, sizes={'emptybarb':0.05},fill_empty=True)

	return U,V

def add_soundingTH(soundvar,usr_case,**kwargs):

	try:
		sigma=kwargs['sigma']
	except:
		sigma=None
	ax = kwargs['ax']
	wptime = kwargs['wptime']

	''' call 2D array made from soundings '''
	sarray,shgt, stimestamp,_ = ps.get_interp_array(soundvar,case=usr_case)
	if sigma:
		sarray = gaussian_filter(sarray, sigma,mode='nearest')		

	if soundvar in ['TE','TD']:
		sarray=sarray-273.15
	elif soundvar in ['RH']:
		sarray[sarray>100.]=100.
	elif soundvar in ['bvf_moist','bvf_dry']:
		sarray=sarray*10000.	

	ini = wptime[0].strftime('%Y-%m-%d %H:%M')
	foo = wptime[-1] + timedelta(hours=1)
	end = foo.strftime('%Y-%m-%d %H:%M')
	wp_timestamp=np.arange(ini, end, dtype='datetime64[20m]')

	''' allocate the array in the corresponding 	time '''
	booleans = np.in1d(wp_timestamp,stimestamp)
	idx = np.nonzero(booleans)

	''' scale idx so has same dimensions as ncols of 
	windprof data (usend in imshow-extent); since 
	we have sounding data every 20m there are 
	3 observations per hour in sounding'''
	idx = idx[0]/3.

	''' create TH sounding meshgrid using axes values of
	imshow-extent (cols,rows of windprof image); '''
	x=idx 
	vertical_gates = shgt.shape[0]
	''' y values are correct for wp coarse resolution; check
	modifications when plotting wp fine resolution '''
	y=np.linspace(0,40,vertical_gates)
	X,Y = np.meshgrid(x,y)
	if soundvar == 'theta':
		levels=range(282,298)
	elif soundvar == 'thetaeq':
		levels=range(298,308)
	elif soundvar in ['bvf_moist','bvf_dry']:
		levels=np.arange(-2,2.5,0.5)

	try:
		cs=ax.contour(X,Y,sarray,levels=levels,colors='k',linewidths=0.8)		
		ax.clabel(cs, levels[0::2], fmt='%1.0f', fontsize=12)	
	except UnboundLocalError:
		cs=ax.contour(X,Y,sarray,colors='k',linewidths=0.8)
		ax.clabel(cs, fmt='%1.0f', fontsize=12)	

def format_xaxis(ax,time):

	' time is start hour'
	date_fmt='%d\n%H'
	new_xticks=np.asarray(range(len(time)))
	xtlabel=[]
	for t in time:
		if np.mod(t.hour,3) == 0:
			xtlabel.append(t.strftime(date_fmt))
		else:
			xtlabel.append('')
	ax.set_xticks(new_xticks)
	ax.set_xticklabels(xtlabel)

def format_yaxis(ax,hgt,**kwargs):
	
	hgt_res = np.unique(np.diff(hgt))[0]
	if 'toplimit' in kwargs:
		toplimit=kwargs['toplimit']
		''' extentd hgt to toplimit km so all 
		time-height sections have a common yaxis'''
		hgt=np.arange(hgt[0],toplimit, hgt_res)
	f = interp1d(hgt,range(len(hgt)))
	ys=np.arange(np.ceil(hgt[0]), np.floor(hgt[-1])+1, 0.5)
	new_yticks = f(ys)
	ytlabel = ['{:2.1f}'.format(y) for y in ys]
	ax.set_yticks(new_yticks+0.5)
	ax.set_yticklabels(ytlabel)	

''' start '''
if __name__ == "__main__":
	main()
























