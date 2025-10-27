import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.tri import Triangulation
import netCDF4 as nc
import cmocean

from utils.plotchannels import plotchannels

bedmachine = 'bedmachine/BedMachineAntarctica-v3.nc'

basins = [
    'G-H',
    'Ep-F',
    # 'F-G',
    'Cp-D',
    'C-Cp',
    'B-C',
    'Jpp-K',
    'J-Jpp',
]

Ncmap = cmocean.cm.matter
Zcmap = cmocean.cm.gray
Zalpha = 0.5

dx = 16
with nc.Dataset(bedmachine, 'r') as bm:
    mask = bm['mask'][::dx, ::dx].astype(int)
    x = bm['x'][::dx].astype(np.float32)
    y = bm['y'][::dx].astype(np.float32)
    bed = bm['bed'][::dx, ::dx].astype(np.float32)
    surf = bm['surface'][::dx, ::dx].astype(np.float32)

bed[mask==0] = np.nan
mask[surf>2000] = 2
xx,yy = np.meshgrid(x,y)

fig,ax = plt.subplots(figsize=(10,4*10./6.))
ax.contour(xx, yy, mask, levels=(0.5,2.5,), colors=('k','k'), linewidths=0.5)
pc = ax.pcolormesh(xx, yy, bed, cmap=Zcmap, 
    vmin=-2000, vmax=2000, alpha=Zalpha)
ax.set_aspect('equal')

ax.spines[['left', 'right', 'top', 'bottom']].set_visible(False)
ax.set_xticks([])
ax.set_yticks([])


nsectors = len(basins)
axs_inset = []
for i in range(nsectors):
    basin = basins[i]

    outline = np.load(f'ANT_Basins/basin_{basin}.npy')
    ax.plot(outline[:,0], outline[:,1], color='k', linestyle='solid', linewidth=1)
    mesh = np.load(f'../issm/{basin}/data/geom/mesh.npy', allow_pickle=True)
    mtri = Triangulation(mesh['x'], mesh['y'], mesh['elements']-1)
    N = np.mean(np.load(f'../issm/{basin}/glads/N.npy'), axis=1)
    tpc = ax.tripcolor(mtri, N/1e6,  edgecolor='none',
        rasterized=True, antialiased=False,
        vmin=0, vmax=4, cmap=cmocean.cm.haline,
        )
    # Q = np.quantile(np.abs(np.load(f'../issm/{basin}/glads/Q.npy')), 0.5, axis=1)
    # lc,sm = plotchannels(mesh, Q, vmin=5, vmax=50, cmap=cmocean.cm.ice_r)
    # ax.add_collection(lc)

cax  = ax.inset_axes((-0.125, 0.6, 0.25, 0.025))
cax2 = ax.inset_axes((-0.125, 0.4, 0.25, 0.025))
# cax3 = ax.inset_axes((-0.125, 0.35, 0.25, 0.025))

fig.colorbar(pc, cax=cax, orientation='horizontal', label='Bed elevation (m)',
    extend='both')
cax.xaxis.set_label_position('top')

cbar2 = fig.colorbar(tpc, cax=cax2, orientation='horizontal', 
    label='Effective Pressure (MPa)', extend='both')
cax2.xaxis.set_label_position('top')

# cb3 = fig.colorbar(sm, cax=cax3, orientation='horizontal',
#     label='Channel discharge (m$^3$ s$^{-1}$)', extend='both')

fig.subplots_adjust(left=0.15, right=1.05, bottom=-0.2, top=1.2,
    wspace=0, hspace=0)

fig.savefig('antarctica_overview_N.png', dpi=400)
