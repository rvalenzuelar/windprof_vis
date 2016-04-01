import Windprof2 as wp
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.backends.backend_pdf import PdfPages

homedir = os.path.expanduser('~')
topdf = False

case = range(8, 15)
res = 'coarse'
o = 'case{}_total_wind_{}.pdf'
t = 'Total wind speed - {} resolution - Case {}'

for c in case:

    ''' creates plot with seaborn style '''
    with sns.axes_style("white"):
        f, ax1 = plt.subplots(figsize=(11, 8.5))

    wspd, wdir, time, hgt = wp.make_arrays(resolution=res,
                                           surface=True,
                                           case=str(c),
                                           homedir=homedir)

    # wp.plot_time_height(ax=ax1, wspd=wspd, time=time,
    #                   height=hgt, vrange=[0,25],
    #                   cname='YlGnBu_r',title='Total wind speed')
    # wp.add_windstaff(wspd,wdir,time,hgt,ax=ax1, color=(1,0.2,0))
    # wp.add_soundingTH('bvf_moist',str(c),ax=ax1,wptime=time,wphgt=hgt,sigma=1)

    wp.plot_colored_staff(ax=ax1, wspd=wspd, wdir=wdir, time=time,
                          height=hgt, spd_range=[0, 20], spd_delta=2,
                          vdensity=1, hdensity=1, cmap='nipy_spectral',
                          title=t.format(res.title(), str(c).zfill(2)))

    wp.add_soundingTH('bvf_moist', str(c), ax=ax1, wptime=time,
                      wphgt=hgt, sigma=1, homedir=homedir)

    if topdf:
        savename = o.format(str(c).zfill(2), res)
        wp_pdf = PdfPages(savename)
        wp_pdf.savefig()
        wp_pdf.close()

    plt.show(block=False)
    # plt.show()
    # plt.close('all')
