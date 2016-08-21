"""
    Plot NOAA wind profiler.
    Files  have extension HHw, where HH is UTC hour

    Raul Valenzuela
    August, 2015
"""


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.axes as maxes
import numpy as np
import numpy.ma as ma
import os
import Meteoframes as mf
import sounding as ps

from datetime import datetime, timedelta
from matplotlib import colors
from scipy.ndimage.filters import gaussian_filter
from scipy.interpolate import interp1d
from rv_utilities import add_colorbar, format_xaxis2, \
                         fill2D_with_nans, discrete_cmap

''' set directory and input files '''
# local_directory='/home/rvalenzuela/'
# local_directory='/Users/raulv/Documents/'
local_directory = os.path.expanduser('~')

base_directory = local_directory + '/WINDPROF'


def plot_vertical_shear(ax=None, wind=None, time=None, height=None):

    diff = np.diff(wind, axis=0)
    nrows, ncols = diff.shape
#    cmap = custom_cmap(17)
#    norm = colors.BoundaryNorm(np.arange(-20, 20), cmap.N)

    img = ax.imshow(diff, interpolation='nearest', origin='lower',
                    # cmap=cmap,
                    cmap='RdBu',
                    vmin=-20, vmax=20,
                    # norm=norm,
                    extent=[0, ncols, 0, nrows],
                    aspect='auto')

    add_colorbar(img, ax)
    format_xaxis(ax, time)
    format_yaxis(ax, height)
    ax.invert_xaxis()
    plt.draw()


def plot_single():

    print base_directory
    usr_case = raw_input('\nIndicate case number (i.e. 1): ')
    wprof_resmod = raw_input(
        '\nIndicate resolution mode (f = fine; c = coarse): ')

    ''' get wind profiler file names '''
    wpfiles = get_filenames(usr_case)
    # print wpfiles
    ''' make profile arrays '''
    if wprof_resmod == 'f':
        res = 'fine'  # 60 [m]
    elif wprof_resmod == 'c':
        res = 'coarse'  # 100 [m]
    else:
        print 'Error: indicate correct resolution (f or c)'
    wspd, wdir, time, hgt = make_arrays(
        files=wpfiles, resolution=res, surface=True, case=usr_case)

    ''' make time-height section of total wind speed '''
    ax = plot_time_height(wspd, time, hgt, vrange=[
                          0, 20], cname='YlGnBu_r', title='Total wind speed')
    l1 = 'BBY wind profiler - Total wind speed (color coded)'

    ''' add wind staffs '''
#    palette = sns.color_palette()
#    color = palette[2]
    u, v = add_windstaff(wspd, wdir, time, hgt, ax=ax, color=color)

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

    ''' make time-height section of meridional wind speed '''
    # ax=plot_time_height(v, time, hgt, vrange=range(-20,22,2),cname='BrBG',title='Meridional component')
    # add_windstaff(wspd,wdir,time,hgt,ax=ax, color=color)

    ''' make time-height section of zonal wind speed '''
    # ax=plot_time_height(u, time, hgt, vrange=range(-20,22,2),cname='BrBG',title='Zonal component')
    # add_windstaff(wspd,wdir,time,hgt,ax=ax, color=color)



def plot_time_height(ax=None, wspd=None, time=None, height=None,
                     spd_range=None,spd_delta=None, cmap=None,
                     title=None,cbar=None,cbarinvi=False,
                     timelabstep=None,kind='pcolormesh'):
    
    ''' NOAA wind profiler files after year 2000 indicate
    the start time of averaging period; so a timestamp of
    13 UTC indicates average between 13 and 14 UTC '''
  

    ''' make a color map of fixed colors '''
    if len(spd_range) == 2:
        bounds = range(spd_range[0], spd_range[1]+spd_delta, spd_delta)
    else:
        bounds = spd_range
    cmap = discrete_cmap(len(bounds)+1,base_cmap=cmap,norm_range=[0.1,0.95])    
    norm = colors.BoundaryNorm(bounds, cmap.N)

    x = np.linspace(0,len(time),len(time)+1)
    y = np.linspace(0,height.size, height.size+1)
    
    wspdm = ma.masked_where(np.isnan(wspd),wspd)
#    wspdm = ma.masked_where(wspdm<0, wspdm)
    
    if kind == 'contourf':
        x,y = np.meshgrid(np.linspace(0,len(time),len(time)),
                          np.linspace(0,height.size, height.size))
        img = ax.contourf(x,y,wspdm,cmap=cmap, norm=norm) 
    else:
        img = ax.pcolormesh(x,y,wspdm,cmap=cmap, norm=norm)
   
    hcbar = add_colorbar(cbar[0], img,
                         loc       = 'right',
                         label     = '$[m\ s^{-1}]$',
                         labelpad  = 20,
                         fontsize  = 12,
                         invisible = cbar[1] )

    
    ax.set_xlim([-1.0, len(time) + 1.0])
    
    if timelabstep is None:
        timelabstep='1H'
        
    format_xaxis2(ax, time, timelabstep=timelabstep)
    ax.invert_xaxis()
    ax.set_xlabel(r'$\leftarrow UTC \left[\stackrel{day}{time}\right]$',
                                          fontsize=20)
    ax.set_ylabel('Altitude MSL [km]')
    
    if title is not None:
        ax.text(0., 1.01, title, transform=ax.transAxes)
    
    plt.subplots_adjust(left=0.08, right=0.95)

    return [ax,hcbar]

def add_windstaff(wspd, wdir, time, hgt, ax=None, color='k',
                  vdensity=0,hdensity=0,head_size=0.05,
                  tail_length=4):


    ''' derive U and V components '''
    U = -wspd * np.sin(wdir * np.pi / 180.)
    V = -wspd * np.cos(wdir * np.pi / 180.)

    x = np.linspace(0.5,len(time)-0.5,len(time))
    y = np.linspace(0.5,hgt.size-0.5, hgt.size)
    X,Y=np.meshgrid(x, y)

    ''' change arrays density '''
    if vdensity == 0 or hdensity == 0:
        pass
    else:
        U = fill2D_with_nans(inarray=U,start=[3,0],
            size=[vdensity,hdensity])
        V = fill2D_with_nans(inarray=V,start=[3,0],
            size=[vdensity,hdensity])

    Uzero = U - U
    Vzero = V - V

    Um = ma.masked_where(np.isnan(U),U)
    Vm = ma.masked_where(np.isnan(V),V)

    ''' barb tail '''
    ax.barbs(X, Y, Um, Vm,
             color=color,
             sizes={'height': 0},
             length=tail_length,
             linewidth=0.8,
             barb_increments={'half': 1})
    
    ''' barb head '''
    ax.barbs(X, Y, Uzero, Vzero, color=color, 
             sizes={'emptybarb': head_size},
             fill_empty=True)

#    ax.set_xlim([-3.0, len(time) + 3.0])
#    format_xaxis(ax, time, delta_hours=1)
#    ax.invert_xaxis()

    format_yaxis2(ax, hgt)

#    return U, V


def plot_colored_staff(ax=None, wspd=None, wdir=None, time=None,
                       height=None, cmap=None, spd_range=None,
                       spd_delta=None,
                       vdensity=None, hdensity=None, title=None,
                       cbar=True):
    
    ''' 
        NOAA wind profiler files after year 2000 indicate
        the start time of averaging period; so a timestamp of
        13 UTC indicates average between 13 and 14 UTC 
    
        There is a bug that plots wrong staff when using
        meridional winds only     
    
    '''

    from matplotlib import rcParams
    rcParams['mathtext.default'] = 'sf'

    spd_array = wspd
    time_array = time
    height_array = height

    ''' make a color map of fixed colors '''
    cmap = discrete_cmap(24,base_cmap=cmap,norm_range=[0.1,0.9])  
    if len(spd_range) == 2:
        bounds = range(spd_range[0], spd_range[1] + spd_delta, spd_delta)
#        vmin = spd_range[0]
#        vmax = spd_range[1]
    else:
        bounds = spd_range
#        vmin = spd_range[0]
#        vmax = spd_range[-1]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    nrows, ncols = spd_array.shape

    ''' derive U and V components '''
    U = -wspd * np.sin(wdir * np.pi / 180.)
    V = -wspd * np.cos(wdir * np.pi / 180.)
    x = np.array(range(len(time))) + 0.5  # wind staff in the middle of pixel
    # wind staff in the middle of pixel
    y = np.array(range(height_array.size)) + 0.5
    X = np.tile(x, (y.size, 1))  # repeats x y.size times to make 2D array
    Y = np.tile(y, (x.size, 1)).T  # repeates y x.size times to make 2D array
    Uzero = U - U
    Vzero = V - V

    ''' change arrays density '''
    if vdensity == 0 or hdensity == 0:
        pass
    else:
        U = fill2D_with_nans(inarray=U,start=[3,0],
            size=[vdensity,hdensity])
        V = fill2D_with_nans(inarray=V,start=[3,0],
            size=[vdensity,hdensity])


    ax.barbs(X, Y, U, V, np.sqrt(U * U + V * V), sizes={'height': 0},
             length=5, linewidth=0.5, barb_increments={'half': 1},
             cmap=cmap, norm=norm)
    barb = ax.barbs(X, Y, Uzero, Vzero, np.sqrt(U * U + V * V),
                    sizes={'emptybarb': 0.05}, fill_empty=True,
                    cmap=cmap, norm=norm)

    if isinstance(cbar, maxes._subplots.Axes):
        hcbar = add_colorbar(cbar, barb,loc='right',
                             label='[m s-1]',labelpad=20,
                             fontsize=12)
    elif isinstance(cbar,bool) and cbar:
        hcbar = add_colorbar(ax, barb,loc='right')
    else:
        hcbar = None
    
    ax.set_xlim([-3.0, len(time_array) + 3.0])
    format_xaxis(ax, time_array, delta_hours=1)
    ax.invert_xaxis()
    ax.set_xlabel(r'$\leftarrow UTC \left[\stackrel{day}{time}\right]$',
                                          fontsize=12)

    format_yaxis2(ax, height_array)
    ax.set_ylabel('Altitude MSL [km]')
    
    if title is not None:
        ax.text(0., 1.01, title, transform=ax.transAxes)
    
    plt.subplots_adjust(left=0.08, right=0.95)
    plt.draw()

    return [ax,hcbar]

def plot_scatter(ax=None, wspd=None, wdir=None, hgt=None, title=None):

    if ax is None:
        fig, ax = plt.subplots(
            2, 2, sharex=True, sharey=True, figsize=(11, 8.5))
        axes = [ax[0, 0], ax[0, 1], ax[1, 0], ax[1, 1]]

    f = interp1d(hgt, range(len(hgt)))
    HIDX = f([0.12, 0.5, 1.0, 2.0])
    HIDX = np.round(HIDX, 0).astype(int)

    wd_array = wdir
    x = wd_array[0, :]
    TIDX = ~np.isnan(x)
    x = x[TIDX]
    y1 = wd_array[HIDX[0], TIDX]  # 120 m AGL
    y2 = wd_array[HIDX[1], TIDX]  # 500 m AGL
    y3 = wd_array[HIDX[2], TIDX]  # 1000 m AGL
    y4 = wd_array[HIDX[3], TIDX]  # 2000 m AGL
    ys = [y1, y2, y3, y4]

    s = 100
    hue = 1.0
    alpha = 0.5
    colors = ['navy', 'green', 'red', 'purple']
    labels = ['120m AGL', '500m AGL', '1000m AGL', '2000m AGL']

    for ax, co, y, lab, n in zip(axes, colors, ys, labels, range(4)):

        ax.scatter(x, y, s=s, color=co, edgecolors='none', alpha=alpha)

        ax.text(0, 1.0, lab, transform=ax.transAxes)

        ax.set_xticks(range(0, 360, 30))
        ax.set_yticks(range(0, 360, 30))
        ax.set_xlim([0, 360])
        ax.set_ylim([0, 360])
        if n in [0, 2]:
            ax.set_ylabel('wind aloft')
        if n in [2, 3]:
            ax.set_xlabel('surface wind')
        ax.axvline(180, linewidth=2, color='k')
        ax.axhline(180, linewidth=2, color='k')
        ax.invert_xaxis()
    plt.suptitle(title)
    plt.subplots_adjust(hspace=0.05, wspace=0.05)
    plt.draw()


def plot_scatter2(ax=None, wspd=None, wdir=None, hgt=None, time=None,
                  mAGL=None,  lim_surf=None, lim_aloft=None, color=None):

    x = wdir[0, :]
    TIDX = ~np.isnan(x)  # time index where surf obs is not nan
    x = x[TIDX]  # surf obs

    f = interp1d(hgt, range(len(hgt)))
    HIDX = f(mAGL / 1000.)
    HIDX = np.round(HIDX, 0).astype(int)
    y = wdir[HIDX, TIDX]  # obs aloft

    s = 100
    hue = 1.0
    alpha = 0.5
    colors = ['navy', 'green', 'red', 'purple']

    ax.scatter(x, y, s=s, color=color, edgecolors='none', alpha=alpha)

    ax.set_xticks(range(0, 360, 180))
    ax.set_yticks(range(0, 360, 180))
    ax.set_xticklabels('')
    ax.set_yticklabels('')
    ax.set_xlim([0, 360])
    ax.set_ylim([0, 360])
    ax.axvline(lim_surf, linewidth=2, color=(0.5, 0.5, 0.5))
    ax.axhline(lim_aloft, linewidth=2, color=(0.5, 0.5, 0.5))
    ax.invert_xaxis()
    plt.draw()

    # time=np.asarray(time)
    # timex=time[TIDX]
    # for t,x, y in zip(timex, x, y):
    #   print [t,np.round(x,0), np.round(y,0)]

    if time is not None:
        TTA_IDX = np.where((x <= lim_surf) & (y <= lim_aloft))[0]
        time = np.asarray(time)
        time = time[TIDX]
        # xtta=x[TTA_IDX]
        # ytta=y[TTA_IDX]
        timetta = time[TTA_IDX]
        # for x,y,t in zip(xtta,ytta,timetta):
        #   print [t, np.round(x,1), np.round(y,1)]
        return timetta


def get_tta_times(resolution='coarse', surface=True, case=None,
                  lim_surf=125, lim_aloft=170, mAGL=120,
                  continuous=True, homedir=None):
    '''
    Note:
    I calibrated default values by comparing retrieved times with
    windprof time-height section plots for all ground radar cases (RV)
    '''

    _, wdir, time, hgt = make_arrays(
        resolution=resolution, surface=surface, case=case,
        homedir=homedir)

    x = wdir[0, :]
    TIDX = ~np.isnan(x)  # time index where surf obs is not nan
    x = x[TIDX]  # surf obs

    f = interp1d(hgt, range(len(hgt)))
    HIDX = f(mAGL / 1000.)
    HIDX = np.round(HIDX, 0).astype(int)
    y = wdir[HIDX, TIDX]  # obs aloft

    TTA_IDX = np.where((x <= lim_surf) & (y <= lim_aloft))[0]
    time = np.asarray(time)
    time = time[TIDX]
    timetta = time[TTA_IDX]

    if continuous:
        ''' fills with datetime when there is 1hr gap and remove
            portions that are either post frontal (case 9)
            or shorter than 5hr (case13)'''
        diff = np.diff(timetta)
        onehrgaps = np.where(diff == timedelta(seconds=7200))
        onehr = timedelta(hours=1)
        timetta_cont = np.append(timetta, timetta[onehrgaps] + onehr)
        timetta_cont = np.sort(timetta_cont)

        diff = np.diff(timetta_cont)
        jump_idx = np.where(diff > timedelta(seconds=3600))[0]
        if jump_idx:
            if len(timetta_cont) - jump_idx > jump_idx:
                return timetta_cont[jump_idx + 1:]
            else:
                return timetta_cont[:jump_idx + 1]
        else:
            return timetta_cont
    else:
        return timetta

def get_surface_data(usr_case, homedir=None):
    
    
    
    ''' set directory and input files '''
    case = 'case' + usr_case.zfill(2)
    casedir = homedir + '/' + case
    out = os.listdir(casedir)
    out.sort()
    files = []
        
    for f in out:
        if f[-3:] in ['met','urf']:
            files.append(f)
        
    file_met = []
    for f in files:
        if f[:3] == 'bby':
            file_met.append(casedir + '/' + f)

    df = []
    for f in file_met:
        meteo = mf.parse_surface(f)
        df.append(meteo)

    if len(df) > 1:
        surface = pd.concat(df)
    else:
        surface = df[0]

    return surface


def make_arrays(resolution='coarse', surface=False,
                case=None, period=False,
                homedir=None):

    wpfiles = get_filenames(case, homedir=homedir+'/WINDPROF')

    wp = []
    ncols = 0  # number of timestamps
    for f in wpfiles:
        if resolution == 'fine':
            wp.append(mf.parse_windprof(f, 'fine'))
        elif resolution == 'coarse':
            wp.append(mf.parse_windprof(f, 'coarse'))
        else:
            print 'Error: resolution has to be "fine" or "coarse"'
        ncols += 1

    ''' creates 2D arrays with spd and dir '''
    nrows = len(
        wp[0].HT.values)  # number of altitude gates (fine same as coarse)
    hgt = wp[0].HT.values
#    print len(hgt)
    wspd = np.empty([nrows, ncols])
    wdir = np.empty([nrows, ncols])
    timestamp = []
    for i, p in enumerate(wp):
        timestamp.append(p.timestamp)
        ''' fine resolution '''
        spd = p.SPD.values
        wspd[:, i] = spd
        dirr = p.DIR.values
        wdir[:, i] = dirr

#    print wspd.shape
    ''' add 2 bottom rows for adding surface obs '''
    bottom_rows = 2
    na = np.zeros((bottom_rows, ncols))
    na[:] = np.nan
    wspd = np.flipud(np.vstack((np.flipud(wspd), na)))
    wdir = np.flipud(np.vstack((np.flipud(wdir), na)))
    if surface:
        ''' make surface arrays '''
        surface = get_surface_data(case, homedir=homedir+'/SURFACE')
        hour = pd.TimeGrouper('H')
        surf_wspd = surface.wspd.groupby(hour).mean()
        surf_wdir = surface.wdir.groupby(hour).mean()
        surf_st = np.where(np.asarray(timestamp) == surf_wspd.index[0])[0][0]
        surf_en = np.where(np.asarray(timestamp) == surf_wspd.index[-1])[0][0]
        wspd[0, surf_st:surf_en + 1] = surf_wspd
        wdir[0, surf_st:surf_en + 1] = surf_wdir

    hgt = np.hstack(([0., 0.05], hgt))

    ''' add last column for 00 UTC of last date '''
    add_left = 1
    nrows, _ = wspd.shape
    na = np.zeros((nrows, add_left))
    na[:] = np.nan
    wspd = np.hstack((wspd, na))
    wdir = np.hstack((wdir, na))
    timestamp.append(timestamp[-1] + timedelta(hours=1))

    if period:
        time = np.asarray(timestamp)
        ini = datetime(*(period['ini'] + [0, 0]))
        end = datetime(*(period['end'] + [0, 0]))
        idx = np.where((time >= ini) & (time <= end))[0]
        return wspd[:, idx], wdir[:, idx], time[idx], hgt

    return wspd, wdir, timestamp, hgt


def make_arrays2(resolution='coarse', add_surface=False, case=None,
                 period=False, interp_hgts=None):

    ''' 
    interpolates to grid with 40 gates,
    first and last gate at 160 and 3750 m
    (92 m resolution)
    '''

    import os

    try:
        wprofdir = os.environ['WPROF_PATH']        
    except KeyError:
        print('*** Need to provide datadir or export WPROF_PATH ***')

    try:
        surfdir = os.environ['SURFACE_PATH']
    except KeyError:
        print('*** Need to provide datadir or export SURFACE_PATH ***')


    wpfiles = get_filenames(case, homedir=wprofdir)
    wpfiles.sort()
    
    wp = []
    ncols = 0  # number of timestamps
    for f in wpfiles:
        # print f
        if resolution == 'fine':
            wp.append(mf.parse_windprof(f, 'fine'))
        elif resolution == 'coarse':
            wp.append(mf.parse_windprof(f, 'coarse'))
        else:
            print 'Error: resolution has to be "fine" or "coarse"'
        ncols += 1

    ''' creates 2D arrays with spd and dir '''
    hgt = wp[0].HT.values
    if interp_hgts is None:
        # upper limit exceeds value in cases [3,7]
        newh = np.linspace(0.160, 3.750, 40)
    else:
        # so we use custom limit
        newh = interp_hgts

    wspd = np.zeros(len(newh))
    wdir = np.zeros(len(newh))
    timestamp = []
    first = True
        
    for p in wp:
        ''' for each hourly profile '''
        timestamp.append(p.timestamp)
        fs = interp1d(hgt, p.SPD.values)
        fd = interp1d(hgt, p.DIR.values)
        news = fs(newh)
        newd = fd(newh)
        if first:
            wspd = news
            wdir = newd
            first = False
        else:
            wspd = np.vstack((wspd, news))
            wdir = np.vstack((wdir, newd))

    wspd = wspd.T
    wdir = wdir.T
    timestamp = np.asarray(timestamp)
  

    ''' add 2 bottom rows for adding surface obs '''
    bottom_rows = 2
    na = np.zeros((bottom_rows, ncols)) + np.nan
    wspd = np.flipud(np.vstack((np.flipud(wspd), na)))
    wdir = np.flipud(np.vstack((np.flipud(wdir), na)))
    if add_surface:
        hour = pd.TimeGrouper('H')
        ''' make surface arrays '''
        try:
            surface = get_surface_data(case, homedir=surfdir)
            surf_wspd = surface.wspd
            surf_wdir = surface.wdir            
            u = -surf_wspd*np.sin(np.radians(surf_wdir))
            v = -surf_wspd*np.cos(np.radians(surf_wdir))
            u_mean = u.groupby(hour).mean()
            v_mean = v.groupby(hour).mean()            
            surf_wspd = np.sqrt(u_mean**2 + v_mean**2)
            surf_wdir = 270 - (np.arctan2(v_mean,u_mean)*180/np.pi)
            surf_wdir[surf_wdir>360] = surf_wdir[surf_wdir>360]-360
            surf_st = np.where(surf_wspd.index == timestamp[0])[0][0]
            surf_en = np.where(surf_wspd.index == timestamp[-1])[0][0]
            wspd[0, surf_st:surf_en + 1] = surf_wspd.values
            wdir[0, surf_st:surf_en + 1] = surf_wdir.values
            
        except ValueError:
            wspd[0, surf_st:surf_en + 1] = surf_wspd.iloc[:surf_en+1].values
            wdir[0, surf_st:surf_en + 1] = surf_wdir.iloc[:surf_en+1].values

    hgt = np.hstack(([0., 0.05], newh))

    ''' add last column for 00 UTC of last date 
    (caution with some storms ending at 22UTC)
    '''
    # add_left = 1
    # nrows, _ = wspd.shape
    # na = np.zeros((nrows, add_left))
    # na[:] = np.nan
    # wspd = np.hstack((wspd, na))
    # wdir = np.hstack((wdir, na))
    # timestamp.append(timestamp[-1] + timedelta(hours=1))

    if period:
        time = np.asarray(timestamp)
        ini = datetime(*(period['ini'] + [0, 0]))
        end = datetime(*(period['end'] + [0, 0]))
        idx = np.where((time >= ini) & (time <= end))[0]
        return wspd[:, idx], wdir[:, idx], time[idx], hgt

    return wspd, wdir, timestamp, hgt

def add_soundingTH(soundvar, usr_case, homedir=None, ax=None,
                   sigma=None, wptime=None, wphgt=None):

    ''' call 2D array made from soundings '''
    out = ps.get_interp_array(soundvar, case=usr_case,
                              homedir=homedir)
    sarray, shgt, stimestamp, _ = out

    if sigma is not None:
        sarray = gaussian_filter(sarray, sigma, mode='nearest')

    ' find sounding index corresponding to top of wp '
    f = interp1d(shgt / 1000., range(len(shgt)))
    soundtop_idx = int(f(wphgt[-1]))

    if soundvar in ['TE', 'TD']:
        sarray = sarray - 273.15
    elif soundvar in ['RH']:
        sarray[sarray > 100.] = 100.
    elif soundvar in ['bvf_moist', 'bvf_dry']:
        sarray = sarray * 10000.

    ini = wptime[0].strftime('%Y-%m-%d %H:%M')
    foo = wptime[-1] + timedelta(hours=1)
    end = foo.strftime('%Y-%m-%d %H:%M')
    wp_timestamp = np.arange(ini, end, dtype='datetime64[20m]')

    ''' allocate the array in the corresponding time '''
    booleans = np.in1d(wp_timestamp, stimestamp)
    idx = np.nonzero(booleans)

    ''' scale idx so has same dimensions as ncols of
    windprof data (usend in imshow-extent); since
    we have sounding data every 20m there are
    3 observations per hour in sounding'''
    idx = idx[0] / 3.

    ''' create TH sounding meshgrid using axes values of
    imshow-extent (cols,rows of windprof image); '''
    x = idx
    vertical_gates = shgt[:soundtop_idx].shape[0]
    ''' y values are correct for wp coarse resolution; check
    modifications when plotting wp fine resolution '''
    y = np.linspace(0, 40, vertical_gates)
    X, Y = np.meshgrid(x, y)
    if soundvar == 'theta':
        levels = range(282, 298)
    elif soundvar == 'thetaeq':
        levels = range(280, 311)
    elif soundvar in ['bvf_moist', 'bvf_dry']:
        levels = np.arange(-2.5, 3.5, 1.0)
    # print levels
    try:
        cs = ax.contour(X, Y, sarray[:soundtop_idx, :],
                        levels=levels, colors='k', linewidths=0.8)
        # ax.clabel(cs, levels, fmt='%1.1f', fontsize=12)
        ax.contourf(X, Y, sarray[:soundtop_idx, :],
                    levels=levels, colors='none',
                    hatches=['*', '.', None, '/', '//'], zorder=10000)
    except UnboundLocalError:
        cs = ax.contour(X, Y, sarray, colors='k', linewidths=0.8)
        ax.clabel(cs, fmt='%1.0f', fontsize=12)


def get_filenames(usr_case, homedir=None):

    case = 'case' + usr_case.zfill(2)
    casedir = homedir + '/' + case
    out = os.listdir(casedir)
    out.sort()
    file_list = []
    period = get_period(case=int(usr_case),outfmt='%y%j.%Hw')
    for f in out:
        if f[-9:] in period:
            file_list.append(casedir + '/' + f)
    return file_list

def get_period(case=None, outfmt=None):

    reqdates = {'1': {'ini': [1998, 1, 18, 15], 'end': [1998, 1, 18, 20]},
#                '2': {'ini': [1998, 1, 26, 4], 'end': [1998, 1, 26, 9]},
#                '3': {'ini': [2001, 1, 23, 21], 'end': [2001, 1, 24, 2]},
                '3':  pd.date_range(start='2001-01-23 00:00',periods=31,freq='60T'),
#                '4': {'ini': [2001, 1, 25, 15], 'end': [2001, 1, 25, 20]},
#                '5': {'ini': [2001, 2, 9, 10], 'end': [2001, 2, 9, 15]},
#                '6': {'ini': [2001, 2, 11, 3], 'end': [2001, 2, 11, 8]},
#                '7': {'ini': [2001, 2, 17, 17], 'end': [2001, 2, 17, 22]},
                '7':  pd.date_range(start='2001-02-17 00:00',periods=25,freq='60T'),
                '8':  pd.date_range(start='2003-01-12 00:00',periods=72,freq='60T'),
                '9':  pd.date_range(start='2003-01-21 00:00',periods=72,freq='60T'),
                '10': pd.date_range(start='2003-02-14 00:00',periods=72,freq='60T'),
                '11': pd.date_range(start='2004-01-08 00:00',periods=72,freq='60T'),
                '12': pd.date_range(start='2004-02-01 00:00',periods=72,freq='60T'),
                '13': pd.date_range(start='2004-02-16 00:00',periods=96,freq='60T'),
                '14': pd.date_range(start='2004-02-24 00:00',periods=72,freq='60T')
                }
    if outfmt is None:
        return reqdates[str(case)]
    else:
        outs = []
        dr = reqdates[str(case)]
        for d in dr:
            outs.append(d.strftime(outfmt))

        return outs

def format_yaxis(ax, hgt, **kwargs):

    hgt_res = np.unique(np.diff(hgt))[0]
    if 'toplimit' in kwargs:
        toplimit = kwargs['toplimit']
        ''' extentd hgt to toplimit km so all
        time-height sections have a common yaxis'''
        hgt = np.arange(hgt[0], toplimit, hgt_res)
    f = interp1d(hgt, range(len(hgt)))
    ys = np.arange(np.ceil(hgt[0]), hgt[-1], 0.2)
    new_yticks = f(ys)
    ytlabel = ['{:2.1f}'.format(y) for y in ys]
    ax.set_yticks(new_yticks + 0.5)
    ax.set_yticklabels(ytlabel)

def format_yaxis2(ax, hgt):

    lasthgt = hgt[-1]+(hgt[-1]-hgt[-2])
    hgt = np.append(hgt,lasthgt)
    lenhgt = len(hgt)

    ax.set_ylim(-5.0, lenhgt)
    
    f = interp1d(hgt, range(len(hgt)))
    ys = np.arange(0, 4.0, 0.2)
    new_yticks = f(ys)
    ytlabel = ['{:2.1f}'.format(y) for y in ys]
    ax.set_yticks(new_yticks+0.5)
    ax.set_yticklabels(ytlabel)





