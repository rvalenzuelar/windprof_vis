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

from datetime import timedelta
from matplotlib import colors



''' set directory and input files '''
base_directory='/home/rvalenzuela/WINDPROF'
# base_directory='/Users/raulv/Desktop/WINDPROF'
print base_directory
usr_case = raw_input('\nIndicate case number (i.e. 1): ')

def main():

	''' get wind profiler file names '''
	wpfiles = get_filenames()

	''' make profile arrays '''
	wspd,wdir,time,hgt = make_arrays(files= wpfiles, resolution='fine',surface=True)
	
	''' make time-height section of total wind speed '''
	ax=plot_time_height(wspd, time, hgt, vrange=[0,20],cname='YlGnBu_r',title='Total wind speed')

	''' add wind staffs '''
	palette = sns.color_palette()
	color=palette[2]
	u,v = add_windstaff(wspd,wdir,time,hgt,ax=ax, color=color)

	''' make time-height section of meridional wind speed '''
	ax=plot_time_height(v, time, hgt, vrange=range(-20,22,2),cname='BrBG',title='Meridional component')
	add_windstaff(wspd,wdir,time,hgt,ax=ax, color=color)
	
	''' make time-height section of zonal wind speed '''
	ax=plot_time_height(u, time, hgt, vrange=range(-20,22,2),cname='BrBG',title='Zonal component')
	add_windstaff(wspd,wdir,time,hgt,ax=ax, color=color)


	plt.show(block=False)


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

def plot_time_height(spd_array,time_array,height_array,**kwargs):

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

	img = plt.imshow(spd_array, interpolation='nearest', origin='lower',
						cmap=cmap, norm=norm,vmin=vmin,vmax=vmax)
	cb = plt.colorbar(img, cmap=cmap, norm=norm, 
				boundaries=bounds, ticks=bounds, label='m s-1',fraction=0.046,pad=0.04)

	ax.set_xlim([-1,48])
	ax.set_ylim([-2,40])
	format_xaxis(ax,time_array)
	format_yaxis(ax,height_array)
	plt.gca().invert_xaxis()
	plt.ylabel('Range hight [km]')
	plt.xlabel(r'$\Leftarrow$'+' Time [UTC]')
	l1 = 'BBY wind profiler - '+title
	l2 = '\nStart: '+time_array[0].strftime('%Y-%m-%d %H:%M UTC')
	l3 = '\nEnd: '+time_array[-2].strftime('%Y-%m-%d %H:%M UTC')
	plt.suptitle(l1+l2+l3)
	plt.draw()

	return ax

def make_arrays(**kwargs):

	file_sound = kwargs['files']
	resolution = kwargs['resolution']
	surface = kwargs['surface']

	if surface:
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
		ncols+=1

	''' creates 2D arrays with spd and dir '''
	nrows = len(wp[0].HT.values) # number of altitude gates (fine=coarse)
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
	wspd[0,:]=surf_wspd
	wdir[0,:]=surf_wdir
	hgt = np.hstack(([0.,0.05],hgt))

	''' add last column for 00 UTC of last date '''
	add_left=1
	nrows, _ = wspd.shape
	na = np.zeros((nrows,add_left))
	na[:] = np.nan
	wspd =np.hstack((wspd,na))
	wdir =np.hstack((wdir,na))
	timestamp.append(timestamp[-1]+timedelta(hours=1))

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
	x=np.array(range(len(time)))
	y=np.array(range(hgt.size))
	X=np.tile(x,(y.size,1))
	Y=np.tile(y,(x.size,1)).T	
	Uzero = U-U
	Vzero = V-V

	ax.barbs(X,Y,U,V,color=color, sizes={'height':0},length=5,linewidth=0.5,barb_increments={'half':1})
	ax.barbs(X,Y,Uzero,Vzero,color=color, sizes={'emptybarb':0.05},fill_empty=True)

	return U,V

def format_xaxis(ax,time):

	date_fmt='%d\n%H'
	new_xticks=range(len(time))
	xtlabel=[]
	for t in time:
		if np.mod(t.hour,3) == 0:
			xtlabel.append(t.strftime(date_fmt))
		else:
			xtlabel.append('')
	ax.set_xticks(new_xticks)
	ax.set_xticklabels(xtlabel)

def format_yaxis(ax,hgt):

	new_yticks=range(hgt.size)

	new_labels=[]
	for t in new_yticks:
		if np.mod(t,5)==0:
			new_labels.append(np.around(hgt[t],decimals=2))
		else:
			new_labels.append(' ')


	ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
	ax.set_yticks(new_yticks)
	ax.set_yticklabels(new_labels)


''' start '''
main()
























