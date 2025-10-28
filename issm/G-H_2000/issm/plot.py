import numpy as np
from matplotlib import pyplot as plt
from matplotlib.tri import Triangulation
import cmocean
from scipy.interpolate import griddata


mesh = np.load('../data/geom/mesh.npy', allow_pickle=True)
mtri = Triangulation(mesh['x'], mesh['y'], mesh['elements']-1)

ice_levelset = np.load('../data/geom/ice_levelset.npy')
ocean_levelset = np.load('../data/geom/ocean_levelset.npy')

present_levelset = np.load('../../G-H/data/geom/ocean_levelset.npy')

surface = np.load('../data/geom/surface.npy')
bed = np.load('../data/geom/bed.npy')
base = np.load('../data/geom/base.npy')
u = np.load('solutions/u_rf_future.npy')
ice = np.load('../data/geom/ice_levelset.npy')

ss,xx,yy = np.load('../data/geom/flowline_00.npy')
xdir = (xx[-1] - xx[0])/(ss[-1] - ss[0])
ydir = (yy[-1] - yy[0])/(ss[-1] - ss[0])

sfull = np.linspace(0, 200e3, 201)
xfull = sfull*xdir + xx[0]
yfull = sfull*ydir + yy[0]

xx = xfull
yy = yfull
ss = sfull

fig,axs = plt.subplots(ncols=2, figsize=(8, 5))
axs[0].tripcolor(mtri, np.load('solutions/u_glads_present.npy'), vmin=0, vmax=2.5e4)
axs[0].set_title('GlaDS')
axs[1].tripcolor(mtri, np.load('solutions/u_rf_future.npy'), vmin=0, vmax=2.5e4)
axs[1].set_title('RF')

for ax in axs.flat:
    ax.set_aspect('equal')
    ax.tricontour(mtri, ice_levelset, levels=(0,), colors=('r',))
    ax.tricontour(mtri, ocean_levelset, levels=(0,), colors=('cyan',))
    ax.tricontour(mtri, present_levelset,levels=(0,),colors=('k',))
    ax.plot(xx, yy, color='k')

fig.savefig('solutions/velocity.png', dpi=400)


# Flowline geometry
fig,ax = plt.subplots()


interp_surface = griddata((mesh['x'], mesh['y']), surface, (xx, yy), method='nearest')
interp_bed = griddata((mesh['x'], mesh['y']), bed, (xx, yy), method='nearest')
interp_base = griddata((mesh['x'], mesh['y']), base, (xx, yy), method='nearest')
interp_speed = griddata((mesh['x'], mesh['y']), u, (xx, yy), method='nearest')
interp_ice = griddata((mesh['x'], mesh['y']), ice, (xx, yy), method='linear')
m = interp_ice<0
# print(interp_ice)
ax.plot(ss[m], interp_bed[m], color='k')
ax.plot(ss[m], interp_base[m], color='blue')
ax.plot(ss[m], interp_surface[m], color='k')
ax2 = ax.twinx()
ax2.plot(ss[m], interp_speed[m], color='r')
ax2.set_ylim([0, 2.5e4])
fig.savefig('solutions/flowline_geom.png', dpi=400)

