import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
from matplotlib.tri import Triangulation
from matplotlib.patches import Rectangle
from scipy import interpolate as interp
import cmocean

yts = 365*86400

xclose = [1.4e6, 2.1e6]
yclose = [4.e5, 10e5]

## BASAL VELOCITY
with xr.open_dataset('../../../../data/lanl-mali/AIS_4kmto20km_hist04.nc', engine='scipy') as ais:
    ub = ais['uReconstructX'][0, :, -1]
    vb = ais['uReconstructY'][0, :, -1]

    xMali = ais['xCell']
    yMali = ais['yCell']

vvMali = np.sqrt(ub**2 + vb**2)*yts

mesh = np.load('../geom/mesh.npy', allow_pickle=True)


xyMali = (xMali, yMali)
xyMesh = np.array([mesh['x'], mesh['y']]).T
vvMesh = interp.griddata(xyMali, vvMali, xyMesh, method='nearest')

print(vvMesh.shape)

mtri = Triangulation(mesh['x'], mesh['y'], mesh['elements']-1)

np.save('basal_velocity_mali.npy', vvMesh)
fig,ax = plt.subplots()
tpc = ax.tripcolor(mtri, np.log10(vvMesh), vmin=0, vmax=3, cmap=cmocean.cm.speed)
r = Rectangle((xclose[0], yclose[0]), xclose[1]-xclose[0], yclose[1]-yclose[0],
    edgecolor='k', facecolor='none')
ax.add_patch(r)
fig.colorbar(tpc, label='log10 Basal velocity (m a$^{-1}$)')
fig.savefig('mali_vvMesh.png', dpi=400)

## BASAL MELT RATE
with xr.open_dataset('../../../../data/lanl-mali/output_state_2060.nc', engine='scipy') as output:
    basalmeltMali = output['basalMeltInput'][0]
    print('basalmeltMali', basalmeltMali.shape)

kgm2s_to_mwea = 365*86400/910
basalmeltMali *= kgm2s_to_mwea

basalmeltMesh = interp.griddata(xyMali, basalmeltMali, xyMesh, method='nearest')

np.save('basal_melt_mali.npy', basalmeltMesh)
fig,ax = plt.subplots()
tpc = ax.tripcolor(mtri, np.log10(basalmeltMesh), cmap=cmocean.cm.thermal, vmin=-3, vmax=0)
r = Rectangle((xclose[0], yclose[0]), xclose[1]-xclose[0], yclose[1]-yclose[0],
    edgecolor='k', facecolor='none')
ax.add_patch(r)
fig.colorbar(tpc, label='log10 Basal melt rate (m w.e. a$^{-1}$)')
fig.savefig('mali_basalmeltMesh.png', dpi=400)
