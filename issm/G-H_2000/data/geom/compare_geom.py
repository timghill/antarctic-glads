import numpy as np
from matplotlib import pyplot as plt
from matplotlib.tri import Triangulation
from scipy.interpolate import griddata
import cmocean

mesh = np.load('mesh.npy', allow_pickle=True)

mali_ocean_levelset = np.load('ocean_levelset.npy')
mali_ice_levelset = np.load('ice_levelset.npy')
mali_bed = np.load('bed.npy')
mali_base = np.load('base.npy')
mali_surface = np.load('surface.npy')
mali_thick = np.load('thick.npy')


bm_ocean_levelset = np.load('../../../G-H/data/geom/ocean_levelset.npy')
bm_ice_levelset = np.load('../../../G-H/data/geom/ice_levelset.npy')
bm_bed = np.load('../../../G-H/data/geom/bed.npy')
bm_base = np.load('../../../G-H/data/geom/base.npy')
bm_surface = np.load('../../../G-H/data/geom/surface.npy')
bm_thick = np.load('../../../G-H/data/geom/thick.npy')


for linenumber in [0,1]:
    ss,xx,yy = np.load(f'flowline_{linenumber:02d}.npy')
    xdir = (xx[-1] - xx[0])/(ss[-1] - ss[0])
    ydir = (yy[-1] - yy[0])/(ss[-1] - ss[0])
    sfull = np.linspace(-20e3, 200e3, 201)
    xfull = sfull*xdir + xx[0]
    yfull = sfull*ydir + yy[0]

    xx = xfull
    yy = yfull
    ss = sfull/1e3

    interp = lambda z: griddata((mesh['x'], mesh['y']), z, (xx,yy))

    fig, ax = plt.subplots()
    ax.plot(ss, interp(bm_bed), color='brown', linestyle='solid')
    ax.plot(ss, interp(mali_bed), color='brown', linestyle='dashed')

    ax.plot(ss, interp(bm_base), color='blue', linestyle='solid')
    ax.plot(ss, interp(mali_base), color='blue', linestyle='dashed')

    ax.plot(ss, interp(bm_surface), color='black', linestyle='solid', label='BedMachine')
    ax.plot(ss, interp(mali_surface), color='black', linestyle='dashed', label='MALI')
    ax.legend(loc='upper left', frameon=False)
    ax.set_xlabel('Distance from grounding line (km)')
    ax.set_ylabel('Elevation (m)')
    ax.grid()
    ax.axhline(0, color='black', linestyle='dotted', linewidth=0.5)

    fig.savefig(f'compare_geom_{linenumber:02d}.png', dpi=400)

    print('MALI ice shelf elevation:', interp(mali_surface)[:5])
    print('BM ice shelf elevation:', interp(bm_surface)[:5])


mtri = Triangulation(mesh['x'], mesh['y'], mesh['elements']-1)
fig,axs = plt.subplots(figsize=(8, 5), ncols=2, nrows=2, sharey=True)
axs[0,0].tripcolor(mtri, bm_thick, vmin=0, vmax=1000, cmap=cmocean.cm.matter)
pc = axs[0,1].tripcolor(mtri, mali_thick, vmin=0, vmax=1000, cmap=cmocean.cm.matter)

epc = axs[1,0].tripcolor(mtri, mali_thick - bm_thick, vmin=-200, vmax=200, cmap=cmocean.cm.balance)

for ax in axs.flat[:3]:
    ax.set_aspect('equal')
    for linenumber in [0,1]:
        ss,xx,yy = np.load(f'flowline_{linenumber:02d}.npy')
        ax.plot(xx, yy, color='cyan', linewidth=0.5)
fig.colorbar(pc, ax=axs[0], label='Thickness (m)')
fig.colorbar(epc, ax=axs[1], label=r'$\Delta$ Thickness (m)')
fig.savefig('compare_geom_map.png', dpi=400)

