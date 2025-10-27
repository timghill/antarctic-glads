import numpy as np
from matplotlib import pyplot as plt
from matplotlib.tri import Triangulation
import xarray as xr
import zarr as zr
import cmocean

# # Read BedMachine for the grid
# stride = 20
# with xr.open_dataset('../bedmachine/BedMachineAntarctica-v3.nc') as bm:
#     x = bm['x'][::stride]
#     y = bm['y'][::stride]

# xx,yy = np.meshgrid(x, y)
# print(xx.shape)
# print(xx.flatten().shape)

# Read Hillebrand outputs

initstore = zr.storage.LocalStore('AIS_4to20km_r01_20220907_relaxed_q5.zarr')
initroot = zr.open(initstore)
x = initroot['xCell']
y = initroot['yCell']
voc = initroot['verticesOnCell']
print('x:', x.shape)
print(voc.shape)
print(initroot['xVertex'].shape)
# print(list(initroot.keys()))

# print(voc[:10, :])
# print(np.max(voc, axis=0))




store = zr.storage.LocalStore('expAE03_04_q05m50_state.zarr')
root = zr.open(store)
thickness = root['thickness'][:]
print(thickness.shape)


i0 = 0
i1 = 151
s = 0.5
fig,axs = plt.subplots(figsize=(12, 6), ncols=3)
(ax0,ax1,ax2) = axs
ax0.scatter(x, y, s, thickness[i0,:], vmin=0, vmax=4000, cmap=cmocean.cm.matter)
ax1.scatter(x, y, s, thickness[i1,:], vmin=0, vmax=4000, cmap=cmocean.cm.matter)
ax2.scatter(x, y, s, thickness[i1,:] - thickness[i0,:],
    vmin=-2000, vmax=2000, cmap=cmocean.cm.balance_r)

for ax in axs:
    ax.set_aspect('equal')
fig.savefig('thickness.png', dpi=400)
