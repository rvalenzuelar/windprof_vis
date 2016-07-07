import Windprof2 as wp
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


def cosd(array):
    return np.cos(np.radians(array))

# homedir = os.path.expanduser('~')
#homedir = '/Volumes/RauLdisk'
homedir = '/localdata'

topdf = False

case = range(13, 14)
res = 'coarse'
o = 'case{}_total_wind_{}.pdf'
t = 'Total wind speed - {} resolution - Case {} - Date: {}'

for c in case:

    ''' creates plot with seaborn style '''
    with sns.axes_style("white"):
        scale=0.8
        f, ax1 = plt.subplots(figsize=(11*scale, 8.5*scale))

    wspd, wdir, time, hgt = wp.make_arrays2(resolution=res,
                                           surface=True,
                                           case=str(c),
                                           homedir=homedir)

    wspdMerid=-wspd*cosd(wdir);

    wp.plot_colored_staff(ax=ax1, wspd=wspdMerid, wdir=wdir, time=time,
                          height=hgt, spd_range=[0, 36], spd_delta=2,
                          vdensity=0, hdensity=0, cmap='nipy_spectral',
                          title=t.format(res.title(), str(c).zfill(2),
                          time[1].strftime('%Y-%b')),cbar=ax1)

    # wp.add_soundingTH('bvf_moist', str(c), ax=ax1, wptime=time,
                      # wphgt=hgt, sigma=1, homedir=homedir)

    if topdf:
        savename = o.format(str(c).zfill(2), res)
        wp_pdf = PdfPages(savename)
        wp_pdf.savefig()
        wp_pdf.close()

    plt.show()
