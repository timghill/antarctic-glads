import numpy as np
import matplotlib.colors
from matplotlib.collections import LineCollection
from matplotlib.cm import ScalarMappable
import cmocean

def plotchannels(mesh, Z, vmin=1, vmax=200,
    lmin=1, lmax=3, cmap=cmocean.cm.dense):
    cnorm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    qlist = np.where(Z>vmin)[0]

    print('n channels:', len(qlist))

    lc_colors = []
    lc_lw = []
    lc_xy = []
    for i in qlist:
        Qi = np.abs(Z[i])
        x0,x1 = mesh['x'][mesh['connect_edge'][i,:]]
        y0,y1 =  mesh['y'][mesh['connect_edge'][i,:]]
        lc_xy.append([(x0, y0), (x1, y1)])
        lw = lmin + (lmax-lmin) * cnorm(Qi)
        lw = max(lw, lmax)
        lw = min(lw, lmin)
        lc_lw.append(lw)
        lc_colors.append(cmap(cnorm(Qi)))
    lc = LineCollection(lc_xy, colors=lc_colors, linewidths=lc_lw,
        capstyle='round', zorder=5)
    lc.set(rasterized=True)
    sm = ScalarMappable(cnorm, cmap=cmap)
    return lc, sm
