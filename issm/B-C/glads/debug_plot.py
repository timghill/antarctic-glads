import numpy as np
from matplotlib import pyplot as plt
from matplotlib.tri import Triangulation
import cmocean
import xarray as xr


# bm = xr.open_dataset('../../../data/bedmachine/BedMachineAntarctica-v3.nc')
# dd = 10
# x = np.array(bm['x'][::dd].values).astype(float)
# y = np.array(bm['y'][::dd].values).astype(float)[::-1]
# bm_raster = np.flipud(np.array(bm['mask'][::dd, ::dd].values).astype(int))
# fig,ax = plt.subplots()
# ax.pcolormesh(x, y, bm_raster)


mesh = np.load('../data/geom/mesh.npy', allow_pickle=True)
mtri = Triangulation(mesh['x'], mesh['y'], mesh['elements']-1)

ocean_levelset = np.load('../data/geom/ocean_levelset.npy').squeeze()

N = np.load('RUN/output_001/steady/N.npy', mmap_mode='r')
fig,ax = plt.subplots()
ax.tripcolor(mtri, N[:,-1], cmap=cmocean.cm.haline, vmin=0, vmax=4e6)
ax.set_title('N (MPa)')
ax.set_aspect('equal')
ax.tricontour(mtri, ocean_levelset, levels=(0.,), colors=('w',), linestyles='solid')

plt.show()

