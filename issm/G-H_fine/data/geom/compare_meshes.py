import numpy as np
from matplotlib import pyplot as plt
from matplotlib.tri import Triangulation
from matplotlib.colors import LogNorm
import cmocean

coarse = np.load('../../../G-H/data/geom/mesh.npy', allow_pickle=True)
fine = np.load('mesh.npy', allow_pickle=True)

meshes = [coarse, fine]

fig,axs = plt.subplots(ncols=2, figsize=(8, 4), sharey=True)
for i in range(2):
    mesh = meshes[i]
    mtri = Triangulation(mesh['x']/1e3, mesh['y']/1e3, mesh['elements']-1)
    ax = axs[i]
    cnorm = LogNorm(vmin=1, vmax=1000)
    tpc = ax.tripcolor(mtri, mesh['area']/1e6, norm=cnorm, cmap=cmocean.cm.matter)

    ax.set_aspect('equal')
    ax.tick_params(axis='both', which='major', labelsize=8)

fig.subplots_adjust(left=0.08, bottom=0.08, top=0.95, right=1., wspace=0.1)
cbar = fig.colorbar(tpc, ax=axs, fraction=0.1)
cbar.set_label('Mesh area (km$^2$)', fontsize=8)

axs[0].set_ylabel('Northing (km)', fontsize=8)
axs[0].set_xlabel('Easting (km)', fontsize=8)
axs[1].set_xlabel('Easting (km)', fontsize=8)

axs[0].text(0.025, 0.975, 'a', fontsize=8, fontweight='bold',
    transform=axs[0].transAxes, va='top')
axs[1].text(0.025, 0.975, 'b', fontsize=8, fontweight='bold',
    transform=axs[1].transAxes, va='top')

fig.savefig('refined_mesh.png', dpi=400)

