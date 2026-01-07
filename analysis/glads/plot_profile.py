import numpy as np
from matplotlib import pyplot as plt
from matplotlib.tri import Triangulation
import cmocean
from scipy.interpolate import griddata

basin = 'Cp-D'
flowlines = [0]

mesh = np.load(f'../../issm/{basin}/data/geom/mesh.npy', allow_pickle=True)
levelset = np.load(f'../../issm/{basin}/data/geom/ocean_levelset.npy')

mtri = Triangulation(mesh['x'], mesh['y'], mesh['elements']-1)

fig,ax = plt.subplots()
ax.tripcolor(mtri, levelset)

for flowline in flowlines:
    ss,xx,yy = np.load(f'../../issm/{basin}/data/geom/flowline_{flowline:02d}.npy')
    ax.plot(xx, yy, color='k', linewidth=2)
ax.set_aspect('equal')

ff = np.load(f'../../issm/{basin}/glads/ff.npy')
ff = np.mean(ff, axis=1)
print(ff.shape)
ff[ff>1] = 1
ff[ff<0] = 0


fig,ax = plt.subplots()
ax.tripcolor(mtri, ff, vmin=0, vmax=1, cmap=cmocean.cm.dense)

for flowline in flowlines:
    ss,xx,yy = np.load(f'../../issm/{basin}/data/geom/flowline_{flowline:02d}.npy')
    ax.plot(xx, yy, color='w', linewidth=2)
ax.set_aspect('equal')

# Extend each flowline
for flowline in flowlines:
    ss,xx,yy = np.load(f'../../issm/{basin}/data/geom/flowline_{flowline:02d}.npy')
    xdir = (xx[1] - xx[0])/(ss[1] - ss[0])
    ydir = (yy[1] - yy[0])/(ss[1] - ss[0])
    print(xdir, ydir)
    print(xdir**2 + ydir**2)
    print(ss)
    snew = np.linspace(-50e3, 200e3, 251)
    xnew = xx[0] + snew*xdir
    ynew = yy[0] + snew*ydir
    ax.plot(xnew, ynew, color='w', linewidth=1)

    lsinterp = griddata((mesh['x'], mesh['y']), levelset, (xnew, ynew), method='nearest')
    print(lsinterp)
    
    pfig,pax = plt.subplots()
    finterp = griddata((mesh['x'], mesh['y']), ff, (xnew, ynew), method='nearest')
    pax.plot(snew, finterp)
    pax.plot(snew, lsinterp)
    pax.set_ylim([-1.25, 1.25])

    inew = np.max(np.where(lsinterp==-1))
    pax.axvline(snew[inew], color='k', linestyle='dashed')

    x0 = xnew[inew]
    y0 = ynew[inew]

    print('Old origin:', xx[0], yy[0])
    print('New origin:', np.round(x0),np.round(y0))
# plt.show()