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
import plotSoundTH as ps

from datetime import timedelta
from matplotlib import colors

''' set directory and input files '''
base_directory='/home/rvalenzuela/WINDPROF'
# base_directory='/Users/raulv/Desktop/WINDPROF'

usr_case=None

def main():

	print base_directory
	global usr_case
	usr_case = raw_input('\nIndicate case number (i.e. 1): ')
	wprof_resmod = raw_input('\nIndicate resolution mode (f = fine; c = coarse): ')

	''' get wind profiler file names '''
	wpfiles = get_filenames()

	''' make profile arrays '''
	if wprof_resmod == 'f':
		res='fine' # 60 [m]
	elif wprof_resmod == 'c':
		res='coarse' # 100 [m]
	else:
		print 'Error: indicate correct resolution (f or c)'
	wspd,wdir,time,hgt = make_arrays(files= wpfiles, resolution=res,surface=True)

	''' make time-height section of total wind speed '''
	ax=plot_time_height(wspd, wdir, time, hgt, vrange=[0,20],cname='YlGnBu_r',
											title='Total wind speed',
											with_sounding='bvf_moist',
											with_windstaff=True)

	# ''' get wind components '''
	# U,V=get_wind_components(wspd,wdir)

	# ''' make time-height section of meridional wind speed '''
	# ax=plot_time_height(V, wdir, time, hgt, vrange=range(-20,22,2),cname='BrBG',
	# 									title='Meridional component',
	# 									with_sounding='v',
	# 									with_windstaff=False)
	
	# ''' make time-height section of zonal wind speed '''
	# ax=plot_time_height(U, wdir, time, hgt, vrange=range(-20,22,2),cname='BrBG',
	# 									title='Zonal component',
	# 									with_sounding='u',
	# 									with_windstaff=False)										

	plt.show(block=False)
	# plt.show()


def plot_time_height(spd_array, dir_array, time_array,height_array,**kwargs):

	''' NOAA wind profiler files after year 2000 indicate 
	the start time of averaging period; so a timestamp of
	13 UTC indicates average between 13 and 14 UTC '''

	vrange=kwargs['vrange']
	cname=kwargs['cname']
	title=kwargs['title']
	soundvar=kwargs['with_sounding']
	windstaff=kwargs['with_windstaff']

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
	extent=[0,ncols,-0.05,3.95] #extent helps to make correct timestamp
	img = ax.imshow(spd_array, interpolation='nearest', origin='lower',
						cmap=cmap, norm=norm,vmin=vmin,vmax=vmax,
						extent=extent,aspect='auto') 
	
	cb = plt.colorbar(img, cmap=cmap, norm=norm, 
				boundaries=bounds, ticks=bounds, label='m s-1',fraction=0.046,pad=0.04)

	ax.set_ylim([-0.1,3.95])
	format_yaxis(ax,height_array)
	ax.set_xticks(range(0,49*3,9))
	format_xaxis(ax,time_array)	
	
	if soundvar:
		''' call 2D array made from soundings '''
		sarray,shgt, stimestamp = ps.get_interp_array(soundvar,case=usr_case)
		if soundvar in ['TE','TD']:
			sarray=sarray-273.15
		elif soundvar in ['bvf_moist','bvf_dry']:
			sarray=sarray*10000.
		''' add interpolated sounding array '''
		add_sounding_array(sarray,shgt,stimestamp,ax=ax,wptime=time_array,var=soundvar)
		l1 = 'BBY wind profiler - '+title+' (color coded)'
		vartitle={'thetaeq':'Eq. potential temperature [K]',
				'theta':'Potential temperature [K]',
				'RH':'Relative humidity [%]',
				'u':'Wind speed zonal compoent [ms-1]',
				'v':'Wind speed meridional compoent [ms-1]',
				'P':'Pressure [hPa]',
				'TE':'Air temperature [degC]',
				'TD':'Dew point temperature [degC]',
				'bvf_dry':'Dry Brunt-Vaisala freq. [s-2]'	,
				'bvf_moist':'Moist Brunt-Vaisala freq. [s-2]'	,
				'MR':'Mixing ratio [g kg-1]'	}
		l1 = l1 + '\n BBY balloon soundings - '+vartitle[soundvar]+' (contours)'
	else:
		l1 = 'BBY wind profiler - '+title

	if windstaff:
		''' add wind staffs '''
		palette = sns.color_palette()
		color=palette[2]
		add_windstaff(spd_array,dir_array,time_array,height_array,ax=ax, color=color)

	plt.gca().invert_xaxis()
	plt.ylabel('Range hight [km]')
	plt.xlabel(r'$\Leftarrow$'+' Time [UTC]')
	
	ls = '\nStart: '+time_array[0].strftime('%Y-%m-%d %H:%M UTC')
	le = '\nEnd: '+time_array[-2].strftime('%Y-%m-%d %H:%M UTC')
	plt.suptitle(l1+ls+le)
	plt.draw()

	return ax

def add_sounding_array(array,hgt,time,**kwargs):

	ax = kwargs['ax']
	wptime = kwargs['wptime']
	var = kwargs['var']

	ini = wptime[0].strftime('%Y-%m-%d %H:%M')
	foo = wptime[-1] + timedelta(hours=1)
	end = foo.strftime('%Y-%m-%d %H:%M')
	wp_timestamp=np.arange(ini, end, dtype='datetime64[20m]')

	''' allocate the array in the corresponding 
	time '''
	booleans = np.in1d(wp_timestamp,time)
	idx = np.nonzero(booleans)

	''' contour plot '''
	rows,cols = array.shape
	X,Y = np.meshgrid(idx,(hgt)/1000.)
	if var == 'theta':
		levels=range(282,298)
	elif var == 'thetaeq':
		levels=range(298,308)
	elif var in ['bvf_moist','bvf_dry']:
		levels=np.arange(-2,2.5,0.5)

	try:
		cs=ax.contour(X,Y,array,levels=levels,colors='k',linewidths=0.8)		
		ax.clabel(cs, levels[0::2], fmt='%1.0f', fontsize=12)	
	except UnboundLocalError:
		cs=ax.contour(X,Y,array,colors='k',linewidths=0.8)
		ax.clabel(cs, fmt='%1.0f', fontsize=12)	
	
	plt.draw()

def add_windstaff(wspd,wdir,time,hgt,**kwargs):

	if kwargs and kwargs['color']:
		color=kwargs['color']
	else:
		color='k'
	ax = kwargs['ax'] 

	''' derive U and V components '''
	u,v = get_wind_components(wspd,wdir)

	''' reduce data density '''
	U=np.full(u.shape,np.nan)
	V=np.full(v.shape,np.nan)
	U[:,1::3]=u[:,1::3]
	V[:,1::3]=v[:,1::3]

	nrows,ncols = U.shape
	x=np.array(range(ncols))+0.5 # wind staff in the middle of pixel
	X,Y = np.meshgrid(x,hgt)
	Uzero = U-U
	Vzero = V-V

	''' make staffs '''
	ax.barbs(X,Y,U,V,color=color, sizes={'height':0},length=5,linewidth=0.5,barb_increments={'half':1})
	ax.barbs(X,Y,Uzero,Vzero,color=color, sizes={'emptybarb':0.05},fill_empty=True)

def make_arrays(**kwargs):

	file_sound = kwargs['files']
	resolution = kwargs['resolution']
	surf = kwargs['surface']

	if surf:
		''' make surface arrays '''
		surface = get_surface_data()
		hour=pd.TimeGrouper('H')
		surf_wspd = surface.wspd.groupby(hour).mean()	
		surf_wdir = surface.wdir.groupby(hour).mean()	

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
	nrows = len(wp[0].HT.values) # number of altitude gates (fine=coarse)
	''' gate altitude needed for overlying sounding contours;
	values have offset of 0.1 km but this is later corrected
	when ploting the time-height section'''
	wphgt = np.arange(0.2, 4.0,0.1)
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
		wspd[0,:]=surf_wspd
		wdir[0,:]=surf_wdir
	hgt = np.hstack(([0.,0.1],wphgt))

	''' add last column for 00 UTC of last date '''
	add_left=1
	nrows, _ = wspd.shape
	na = np.zeros((nrows,add_left))
	na[:] = np.nan
	wspd =np.hstack((wspd,na))
	wdir =np.hstack((wdir,na))
	timestamp.append(timestamp[-1]+timedelta(hours=1))

	''' repeat along the x axis so we can
	match the sounding array with 20
	minute resolution '''
	wspd = np.repeat(wspd,3,axis=1)
	wdir = np.repeat(wdir,3,axis=1)

	return wspd,wdir,timestamp,hgt

def get_wind_components(wspd,wdir):

	U=-wspd*np.sin(wdir*np.pi/180.)
	V=-wspd*np.cos(wdir*np.pi/180.)
	return U,V

def get_filenames():

	case='case'+usr_case.zfill(2)
	casedir=base_directory+'/'+case
	out=os.listdir(casedir)
	out.sort()
	file_sound=[]
	for f in out:
		if f[-1:]=='w': 
			file_sound.append(casedir+'/'+f)
	return file_sound

def get_surface_data():

	''' set directory and input files '''
	base_directory='/home/rvalenzuela/SURFACE'
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
	if usr_case =='3':
		index_field=[3,6,9,10,12,17,26]
	elif usr_case == '13':
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

def format_xaxis(ax,time):

	xtlabs = ax.get_xticklabels()
	date_fmt='%H\n%d'
	newxtlabs=[]
	for t in time:
		if np.mod(t.hour,3) == 0:
			newxtlabs.append(t.strftime(date_fmt))
	ax.set_xticklabels(newxtlabs)


def format_yaxis(ax,hgt):

	''' adjust altitude labels such that
	values represent gate altitutudes from 
	first gate, add an empty label, and a 
	surface observation level'''
	ax.set_yticks(hgt)
	hgt=hgt-0.1 # correct range altitude [km]
	ytlabs=np.arange(hgt[-1],hgt[2],-0.1)
	ytlabs=[str(s) for s in ytlabs]
	ytlabs.extend(['','0.0'])
	ytlabs.reverse()
	ax.set_yticklabels(ytlabs)


''' start '''
if __name__ == "__main__":
	main()




