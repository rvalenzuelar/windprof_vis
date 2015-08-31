"""
	Plot NOAA wind profiler. 
	Files  have extension HHw, where HH is UTC hour

	Raul Valenzuela
	August, 2015
"""


import pandas as pd
import metpy.plots as metplt
# import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

import numpy as np
import os 
import datetime as dt
import sys

import Thermodyn as thermo
import Meteoframes as mf

''' set directory and input files '''
base_directory='/home/rvalenzuela/WINDPROF'
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
	for f in file_sound:
		wpf.append(mf.parse_windprof(f,'fine'))
		wpc.append(mf.parse_windprof(f,'coarse'))

	for prof in wpf:
		print prof.SPD
	# print wpc

main()
