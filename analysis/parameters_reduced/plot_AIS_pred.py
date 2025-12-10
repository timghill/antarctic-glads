import numpy as np
import netCDF4 as nc

from matplotlib import pyplot as plt
import cmocean

Dmean = nc.Dataset('AIS_2km_N_mean.nc')
Dmean.set_auto_mask(False)
Nmean = Dmean['effectivePressure'][:]
Nmean[Nmean>1e12] = np.nan
x = Dmean['x'][:]
y = Dmean['y'][:]
Dmean.close()

Dall = nc.Dataset('AIS_2km_N_ensemble.nc')
Nall = Dall['effectivePressure'][:]
Nall[Nall>1e12] = np.nan
# Nstd = np.std(Nall, axis=0)
Nstd = np.quantile(Nall, 0.84, axis=0) - np.quantile(Nall, 0.16, axis=0)
Dall.close()

fig,ax = plt.subplots()
pc = ax.pcolormesh(Nmean/1e6, vmin=0, vmax=4, cmap=cmocean.cm.haline)
ax.set_aspect('equal')
fig.colorbar(pc, label='Effective pressure (MPa)')
fig.savefig('RF_mean_N.png', dpi=400)

fig,ax = plt.subplots()
pc = ax.pcolormesh(Nstd/1e6, vmin=0, vmax=4, cmap=cmocean.cm.amp)
ax.set_aspect('equal')
fig.colorbar(pc, label='Effective pressure standard deviation (MPa)')
fig.savefig('RF_std_N.png', dpi=400)
