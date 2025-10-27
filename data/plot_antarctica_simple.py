import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.tri import Triangulation
import netCDF4 as nc
import cmocean

bedmachine = 'bedmachine/BedMachineAntarctica-v3.nc'

basins = [
    'G-H',
    # 'Ep-F',
    # 'F-G',
    'Cp-D',
    'C-Cp',
    'B-C',
    'Jpp-K',
    # 'J-Jpp',
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

fig,ax = plt.subplots(figsize=(6,4))
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
    tpc = ax.tripcolor(mtri, np.log10(mesh['area']),  edgecolor='none',
        linewidth=0.25, rasterized=True, antialiased=False,
        vmin=6, vmax=9, cmap=cmocean.cm.matter,
        )

cax  = ax.inset_axes((-0.125, 0.6, 0.25, 0.025))
cax2 = ax.inset_axes((-0.125, 0.4, 0.25, 0.025))

fig.colorbar(pc, cax=cax, orientation='horizontal', label='Bed elevation (m)',
    extend='both')
cax.xaxis.set_label_position('top')

cbar2 = fig.colorbar(tpc, cax=cax2, orientation='horizontal', 
    label='Mesh area (m$^2$)', extend='both')
cax2.xaxis.set_label_position('top')

fig.subplots_adjust(left=0.15, right=1.05, bottom=-0.2, top=1.2,
    wspace=0, hspace=0)

fig.savefig('antarctica_overview_simple.png', dpi=400)
