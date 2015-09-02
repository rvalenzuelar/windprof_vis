"""
	Plot NOAA wind profiler. 
	Files  have extension HHw, where HH is UTC hour

	Raul Valenzuela
	August, 2015
"""


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# import matplotlib as mpl
import matplotlib.dates as mdates
from datetime import timedelta
import matplotlib.ticker as mtick

import numpy as np
import os 
# import datetime as dt
import sys

import Thermodyn as thermo
import Meteoframes as mf

''' set directory and input files '''
# base_directory='/home/rvalenzuela/WINDPROF'
base_directory='/Users/raulv/Desktop/WINDPROF'
print base_directory
usr_case = raw_input('\nIndicate case number (i.e. 1): ')
case='case'+usr_case.zfill(2)
casedir=base_directory+'/'+case
out=os.listdir(casedir)
out.sort()
file_sound=[]
for f in out:
	if f[-1:]=='w': 
		file_sound.append(casedir+'/'+f)

def main():
	wpf=[] 
	wpc=[]
	ncols=0 # number of timestamps
	for f in file_sound:
		wpf.append(mf.parse_windprof(f,'fine'))
		wpc.append(mf.parse_windprof(f,'coarse'))
		ncols+=1

	nrows = len(wpf[0].HT.values) # number of altitude gates (fine=coarse)
	hgt_fine = wpf[0].HT.values
	hgt_coar = wpc[0].HT.values
	spd_fine = np.empty([nrows,ncols])
	dir_fine = np.empty([nrows,ncols])
	spd_coar = np.empty([nrows,ncols])
	dir_coar = np.empty([nrows,ncols])
	timestamp = []
	for pf,pc,i in zip(wpf,wpc,range(ncols)):
		timestamp.append(pf.timestamp)	
		''' fine resolution '''
		spdf=pf.SPD.values
		spd_fine[:,i]=spdf 		 
		dirf=pf.DIR.values
		dir_fine[:,i]=dirf

		''' coarse resolution '''
		spdc=pc.SPD.values
		spd_coar[:,i]=spdc
		dirc=pc.DIR.values
		dir_coar[:,i]=dirc
	



	''' add last column for 00 UTC of
	last date and 2 bottom rows for adding surface obs'''
	add_left=1
	na = np.zeros((nrows,add_left))
	na[:] = np.nan
	spd_fine =np.hstack((spd_fine,na))
	add_bottom=2
	na = np.zeros((add_bottom,ncols+add_left))
	na[:] = np.nan
	spd_fine = np.flipud(np.vstack((np.flipud(spd_fine),na)))
	hgt_fine = np.hstack(([0.,0.05],hgt_fine))



	


	''' creates plot with seaborn style '''
	with sns.axes_style("ticks"):
		f, ax = plt.subplots(figsize=(11,8.5))

	im=ax.imshow(spd_fine,interpolation='nearest',cmap="RdYlBu_r",vmin=0,vmax=20,origin='bottom')
	
	''' format yticks '''	
	old_yticks = ax.get_yticks()
	new_yticks=[]
	for yt in old_yticks:
		if int(yt)< 0:
			new_yticks.append(-1)
		elif int(yt) == 0 and add_bottom == 0:
			new_yticks.append(np.around(hgt_fine[int(yt)],decimals=2))
		elif int(yt) == 0 and add_bottom > 0:
			new_yticks.append(0)
		elif int(yt)<40:
			new_yticks.append(np.around(hgt_fine[int(yt)],decimals=2))
		else:
			new_yticks.append(np.around(hgt_fine[-1],decimals=2))	
	ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
	ax.set_yticklabels(new_yticks)

	''' format xticks '''
	date_fmt='%d\n%H'
	new_xticks=range(len(timestamp)+1)
	xtlabel=[]
	for t in timestamp:
		if np.mod(t.hour,3) == 0:
			xtlabel.append(t.strftime(date_fmt))
		else:
			xtlabel.append('')
	last_hour=timestamp[-1]+timedelta(hours=1)
	date_fmt='day %d\n hour %H'
	xtlabel.append(last_hour.strftime(date_fmt))
	ax.set_xticks(new_xticks)
	ax.set_xticklabels(xtlabel)

	plt.colorbar(im)
	plt.gca().invert_xaxis()

	plt.ylabel('Range hight [km]')
	start_day=str(timestamp[0].day)
	end_day=str(timestamp[-1].day)

	# plt.text(48,-4,end_day,horizontalalignment='center')
	# plt.text(0,-4,start_day,horizontalalignment='center')
	plt.xlabel(r'$\Leftarrow$'+' Time [UTC]')

	plt.show(block=False)



def plot(array):

	fig,ax=plt.subplots()
	ax.imshow(array)

	plt.show()

main()






























