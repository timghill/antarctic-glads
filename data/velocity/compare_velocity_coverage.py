import os
import numpy as np
from matplotlib import pyplot as plt
import xarray as xr

issmdir = os.getenv('ISSM_DIR')
fnamev0 = os.path.join(issmdir, 'examples/Data/Antarctica_ice_velocity.nc')
fnamev1 = 'antarctica_ice_velocity_450m_v2.nc'

with xr.open_dataset(fnamev0) as nc0:
    vx0 = np.flipud(nc0['vx'].values).astype(np.float64)
    vy0 = np.flipud(nc0['vy'].values).astype(np.float64)
    xmin = float(nc0.attrs['xmin'].strip(' m'))
    ymax = float(nc0.attrs['ymax'].strip(' m'))
    spacing = float(nc0.attrs['spacing'].strip(' m'))
    nx = int(nc0.attrs['nx'])
    ny = int(nc0.attrs['ny'])
    x0 = xmin + np.arange(0,nx+1)*spacing
    y0 = ymax - ny*spacing + np.arange(0,ny+1)*spacing

stride = 2
with xr.open_dataset(fnamev1) as nc1:
    vx1 = nc1['VX'][::stride, ::stride].values.astype(np.float64)
    vy1 = nc1['VY'][::stride, ::stride].values.astype(np.float64)
    x1 = nc1['x'][::stride].values
    y1 = nc1['y'][::stride].values

print('vx1:', vx1.shape)
print('x1:', x1.shape)

eps = 1e-6

fig,axs = plt.subplots(figsize=(12, 6), ncols=2)

vv0 = np.sqrt(vx0**2 + vy0**2)
vv1 = np.sqrt(vx1*2 + vy1**2)

print(vv1.dtype)
print(vv1[vv1>0])

print(np.isnan(vv1))

axs[0].pcolormesh(x0, y0, np.log10(vv0 + eps), vmin=0, vmax=3)
axs[1].pcolormesh(x1, y1, np.log10(vv1 + eps), vmin=0, vmax=3)
axs[0].set_title('NSIDC V0')
axs[1].set_title('NSIDC V1')

for ax in axs.flat:
    ax.set_aspect('equal')
    ax.spines[['right', 'top']].set_visible(False)

fig.savefig('compare_velocity_coverage.png', dpi=600)
