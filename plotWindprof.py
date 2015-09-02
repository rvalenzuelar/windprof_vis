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


def main():

	''' get wind profiler file names '''
	wpfiles = get_filenames()

	''' make arrays '''
	wspd,wdir,time,hgt = make_arrays(files= wpfiles, resolution='fine')
	
	ax=plot_time_height(wspd,wdir,time,hgt,vrange=[2,20])
	color=[0.2,0.2,0.2]
	# add_windstaff(ax,wspd,wdir,time,hgt,color=color)

	wspd,wdir,time,hgt = make_arrays(files= wpfiles, resolution='coarse')
	plot_time_height(wspd,wdir,time,hgt,vrange=[2,20])

	plt.show(block=False)


def get_filenames():

	usr_case = raw_input('\nIndicate case number (i.e. 1): ')
	case='case'+usr_case.zfill(2)
	casedir=base_directory+'/'+case
	out=os.listdir(casedir)
	out.sort()
	file_sound=[]
	for f in out:
		if f[-1:]=='w': 
			file_sound.append(casedir+'/'+f)
	return file_sound

def plot_time_height(spd_array,dir_array,time_array,height_array,**kwargs):

	vrange=kwargs['vrange']

	''' creates plot with seaborn style '''
	with sns.axes_style("ticks"):
		f, ax = plt.subplots(figsize=(11,8.5))

	''' make a color map of fixed colors '''
	snsmap=sns.color_palette("YlGnBu_r", 20)
	cmap = colors.ListedColormap(snsmap[2:-1])
	vdelta=2
	bounds=range(vrange[0],vrange[1]+vdelta, vdelta)
	norm = colors.BoundaryNorm(bounds, cmap.N)

	img = plt.imshow(spd_array, interpolation='nearest', origin='lower',
						cmap=cmap, norm=norm,vmin=vrange[0],vmax=vrange[1])
	plt.colorbar(img, cmap=cmap, norm=norm, 
				boundaries=bounds[1:], ticks=bounds[1:])
	ax.set_xlim([-1,48])
	ax.set_ylim([0,39])
	format_xaxis(ax,time_array)
	format_yaxis(ax,height_array)
	plt.gca().invert_xaxis()
	plt.ylabel('Range hight [km]')
	plt.xlabel(r'$\Leftarrow$'+' Time [UTC]')

	plt.draw()

	return ax

def make_arrays(**kwargs):

	file_sound = kwargs['files']
	resolution = kwargs['resolution']

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

	''' add last column for 00 UTC of last date '''
	add_left=1
	na = np.zeros((nrows,add_left))
	na[:] = np.nan
	wspd =np.hstack((wspd,na))
	wdir =np.hstack((wdir,na))
	timestamp.append(timestamp[-1]+timedelta(hours=1))

	''' add 2 bottom rows for adding surface obs '''
	add_bottom=2
	na = np.zeros((add_bottom,ncols+add_left))
	na[:] = np.nan
	wspd = np.flipud(np.vstack((np.flipud(wspd),na)))
	wdir = np.flipud(np.vstack((np.flipud(wdir),na)))
	hgt = np.hstack(([0.,0.05],hgt))

	return wspd,wdir,timestamp,hgt

def add_windstaff(ax,wspd,wdir,time,hgt,**kwargs):

	if kwargs and kwargs['color']:
		color=kwargs['color']
	else:
		color='k'

	''' derive U and V components '''
	U=-wspd*np.sin(wdir*np.pi/180.)
	V=-wspd*np.cos(wdir*np.pi/180.)
	x=np.array(range(len(time)))
	y=np.array(range(hgt.size))
	X=np.tile(x,(y.size,1))
	Y=np.tile(y,(x.size,1)).T	
	Uzero = U-U
	Vzero = V-V

	ax.barbs(X,Y,U,V,color=color, sizes={'height':0},length=5,linewidth=0.5)
	ax.barbs(X,Y,Uzero,Vzero,color=color, sizes={'emptybarb':0.05},fill_empty=True)

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
























