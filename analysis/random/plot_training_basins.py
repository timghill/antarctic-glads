import numpy as np
from matplotlib import pyplot as plt
from matplotlib.tri import Triangulation
import xarray as xr
import cmocean

basins = ['B-C', 'C-Cp', 'Cp-D', 'G-H', 'Jpp-K']
stride = 8
with xr.open_dataset('../../data/bedmachine/BedMachineAntarctica-v3.nc') as bm:
    bed = bm['bed'][::stride, ::stride].values
    x = bm['x'][::stride].values
    y = bm['y'][::stride].values
    mask = bm['mask'][::stride, ::stride].values

bed[mask<2] = np.nan

fig,ax = plt.subplots(figsize=(10, 9))
ax.pcolormesh(x, y, bed, vmin=-4000, vmax=4000, cmap='grey_r')

for basin in basins:
    mesh = np.load(f'../../issm/{basin}/data/geom/mesh.npy', allow_pickle=True)
    bed = np.load(f'../../issm/{basin}/data/geom/bed.npy')

    mtri = Triangulation(mesh['x'], mesh['y'], mesh['elements']-1)
    pc = ax.tripcolor(mtri, bed, vmin=-4000, vmax=4000, cmap=cmocean.cm.topo)

    outline = np.load(f'../../data/ANT_Basins/basin_{basin}.npy')

    ax.plot(outline[:, 0], outline[:, 1], color='k', linewidth=2)

ax.set_aspect('equal')
fig.subplots_adjust(left=-0.15, right=1.15, bottom=-0.15, top=1.15)

cax = ax.inset_axes((0.2, 0.25, 0.3, 0.025))
cbar = fig.colorbar(pc, cax=cax, orientation='horizontal')
cbar.set_label('Bed elevation', fontsize=16)
# cbar.set_ticks([-4000, -2000, 0, 2000, 4000], [-4000, -2000, 0, 2000, 4000], size=14)
cbar.ax.tick_params(labelsize=14) 
# cbar.set_fontsize(16)
ax.spines[['left', 'right', 'top', 'bottom']].set_visible(False)
ax.set_xticks([])
ax.set_yticks([])

fig.savefig('figures/training_basins.png', dpi=400)
