import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.tri import Triangulation
import cmocean
import xarray as xr

from scipy.interpolate import griddata

engine = 'scipy'

mesh = np.load('mesh.npy', allow_pickle=True)
vx = np.load('vx.npy')
vy = np.load('vy.npy')
vv = np.sqrt(vx**2 + vy**2)



# Interpolate velocity onto mesh
nsidc_vel=os.path.join(os.getenv('ISSM_DIR'), 
    'examples/Data/Antarctica_ice_velocity.nc')
nsidc = xr.open_dataset(nsidc_vel, engine=engine)
xmin = float(nsidc.attrs['xmin'].strip(' m'))
ymax = float(nsidc.attrs['ymax'].strip(' m'))
spacing = float(nsidc.attrs['spacing'].strip(' m'))
nx = int(nsidc.attrs['nx'])
ny = int(nsidc.attrs['ny'])
vx = np.flipud(nsidc['vx'].values)
vy = np.flipud(nsidc['vy'].values)
x = xmin + np.arange(0,nx+1)*spacing
y = (ymax - ny*spacing + np.arange(0,ny+1)*spacing)
xx,yy = np.meshgrid(x[:-1],y[:-1])
xy = (xx.flatten(), yy.flatten())
vx = griddata(xy, vx.flatten(), (mesh['x'], mesh['y']), method='linear')
vy = griddata(xy, vy.flatten(), (mesh['x'], mesh['y']), method='linear')

np.save('vx.npy', vx)
np.save('vy.npy', vy)

vv = np.sqrt(vx**2 + vy**2)
print('Done interpolating speed')

fig,ax = plt.subplots()
mtri = Triangulation(mesh['x'], mesh['y'], mesh['elements']-1)
pc = ax.tripcolor(mtri, np.log10(vv), vmin=0, vmax=3, cmap=cmocean.cm.speed)
fig.colorbar(pc, label='Speed (m/year)')
ax.set_aspect('equal')
fig.savefig('vv.png', dpi=400)

ss,xx,yy = np.load('flowline_00.npy')

vv[vv==0] = -100

fig,ax = plt.subplots()
vv_linear = griddata((mesh['x'], mesh['y']), vv, (xx,yy), method='linear')
ax.plot(ss/1e3, vv_linear, label='linear')

vv_nearest = griddata((mesh['x'], mesh['y']), vv, (xx,yy), method='nearest')
ax.plot(ss/1e3, vv_nearest, label='nearest')
fig.savefig('vv_flowline.png', dpi=400)
