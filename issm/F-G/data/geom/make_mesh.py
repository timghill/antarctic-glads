# MAKE AMERY MESH

import os
import sys

issmdir = os.getenv('ISSM_DIR')
print(issmdir)
sys.path.append(issmdir + '/bin')
sys.path.append(issmdir + '/lib')

from issmversion import issmversion
from model import model
from triangle import *
from bamg import *
from GetAreas import GetAreas
from InterpFromGridToMesh import InterpFromGridToMesh

import matplotlib
# matplotlib.use('QtAgg')
from matplotlib import pyplot as plt
from matplotlib.tri import Triangulation
from matplotlib.gridspec import GridSpec
import cmocean

import xarray as xr
from scipy import interpolate as interp
from utils.tools import reorder_edges_mesh
import pickle

expfile = 'outline.exp'
meshfile = 'mesh.npy'
engine = 'scipy'

# Mesh parameters
hmax=25e3        # maximum element size of the final mesh
hmin=2.5e3       # minimum element size of the final mesh
gradation=1.2    # maximum size ratio between two neighboring elements
# err=8            # maximum error between interpolated and control field
vel_thresh = 50 # Velocity (m/a) below which to refine the mesh
# aniso = 1.2

# Parameters for an initial mesh
hmax_init = 10e3
hmin_init = 10e3

# Init model instance and create initial mesh
md = model()
md = bamg(md, 'domain', expfile, 'hmax', hmax_init,
    'hmin', hmin_init, 'gradation', gradation)
print('Made draft mesh with numberofvertices:', md.mesh.numberofvertices)

# Interpolate velocity onto mesh
nsidc_vel=os.path.join(os.getenv('ISSM_DIR'), 
    'examples/Data/Antarctica_ice_velocity.nc')
nsidc = xr.open_dataset(nsidc_vel, engine=engine)
xmin = float(nsidc.attrs['xmin'].strip(' m'))
ymax = float(nsidc.attrs['ymax'].strip(' m'))
spacing = float(nsidc.attrs['spacing'].strip(' m'))
nx = int(nsidc.attrs['nx'])
ny = int(nsidc.attrs['ny'])
vx = nsidc['vx'].values
vy = nsidc['vy'].values
x = xmin + np.arange(0,nx+1)*spacing
y = ymax - ny*spacing + np.arange(0,ny+1)*spacing
vx_obs=InterpFromGridToMesh(x,y,np.flipud(vx),md.mesh.x,md.mesh.y,0)
vy_obs=InterpFromGridToMesh(x,y,np.flipud(vy),md.mesh.x,md.mesh.y,0)
vel_obs=np.sqrt(vx_obs**2+vy_obs**2)

# Init area at max element size
area = hmax*np.ones(md.mesh.numberofvertices)

# Refine where fast flowing
area[vel_obs>=vel_thresh] = hmin
md = bamg(md, 'hVertices', area, 'gradation', gradation)
print('Made refined mesh with numberofnodes:', md.mesh.numberofvertices)

# Compute and save additional mesh characteristics
print('Computing additional mesh characteristics')
meshdict = {}
meshdict['x'] = md.mesh.x
meshdict['y'] = md.mesh.y
meshdict['elements'] = md.mesh.elements
meshdict['area'] = GetAreas(md.mesh.elements, md.mesh.x, md.mesh.y)
meshdict['vertexonboundary'] = md.mesh.vertexonboundary
meshdict['numberofelements'] = md.mesh.numberofelements
meshdict['numberofvertices'] = md.mesh.numberofvertices
connect_edge = reorder_edges_mesh(meshdict)
meshdict['connect_edge'] = connect_edge
# Compute edge lengths
x0 = md.mesh.x[connect_edge[:,0]]
x1 = md.mesh.x[connect_edge[:,1]]
dx = x1 - x0
y0 = md.mesh.y[connect_edge[:,0]]
y1 = md.mesh.y[connect_edge[:,1]]
dy = y1 - y0
edge_length = np.sqrt(dx**2 + dy**2)
meshdict['edge_length'] = edge_length
print('Min edge length:', edge_length.min())
print('Med edge length:', np.median(edge_length))
print('Max edge length:', edge_length.max())

with open('mesh.npy', 'wb') as meshout:
    pickle.dump(meshdict, meshout)


# Interpolate velocity
vx_obs=InterpFromGridToMesh(x,y,np.flipud(vx),md.mesh.x,md.mesh.y,0)
vy_obs=InterpFromGridToMesh(x,y,np.flipud(vy),md.mesh.x,md.mesh.y,0)
np.save('vx.npy', vx_obs)
np.save('vy.npy', vy_obs)

# Interpolate geometry features onto the new mesh
print('Interpolating geometry')
bm = xr.open_dataset('../../../../data/bedmachine/BedMachineAntarctica-v3.nc')

stride = 10
x = np.array(bm['x'][::stride].values).astype(int)
y = np.array(bm['y'][::stride].values).astype(int)[::-1]
mask = np.flipud(np.array(bm['mask'][::stride, ::stride].values).astype(int))
bed = np.flipud(np.array(bm['bed'][::stride, ::stride].values).astype(float))
thick = np.flipud(np.array(bm['thickness'][::stride, ::stride].values).astype(float))
surf = np.flipud(np.array(bm['surface'][::stride, ::stride].values).astype(float))
print('surf min:', np.min(surf))
# assert np.all(surf==(bed+thick))

mesh_bm_mask = InterpFromGridToMesh(x, y, mask, md.mesh.x, md.mesh.y, 0)
mesh_ocean_levelset = np.ones(md.mesh.numberofvertices, dtype=int)
mesh_ocean_levelset[mesh_bm_mask>2.5] = -1
mesh_ice_levelset = -1*np.ones(md.mesh.numberofvertices).astype(int)
mesh_bed = InterpFromGridToMesh(x, y, bed, md.mesh.x, md.mesh.y, 0)
mesh_thick = InterpFromGridToMesh(x, y, thick, md.mesh.x, md.mesh.y, 0)
mesh_surface = InterpFromGridToMesh(x, y, surf, md.mesh.x, md.mesh.y, 0)

print('mesh_surface min:', np.min(mesh_surface))

# Post-process thickness where thickness<1
print('min thickness:', np.min(mesh_thick))
min_thick = 10
mesh_base = mesh_surface - mesh_thick
pos = mesh_thick<=min_thick
mesh_thick[pos] = min_thick
mesh_surface = mesh_base + mesh_thick

# Ice shelf base
di = md.materials.rho_ice/md.materials.rho_water
print('di:', di)
iceshelf = mesh_ocean_levelset<0
mesh_thick[iceshelf] = 1/(1-di)*mesh_surface[iceshelf]
print('Min thickness:', np.min(mesh_thick))
print('Min surface:', np.min(mesh_surface))

# Always true
mesh_base = mesh_surface - mesh_thick

print('mesh_ocean_levelset:', mesh_ocean_levelset.shape)
np.save('bed.npy', mesh_bed)
np.save('base.npy', mesh_base)
np.save('thick.npy', mesh_thick)
np.save('surface.npy', mesh_surface)
np.save('ocean_levelset.npy', mesh_ocean_levelset)
np.save('ice_levelset.npy', mesh_ice_levelset)

fig,axs = plt.subplots(nrows=2, ncols=2, figsize=(10,8))
features = [mesh_bed, mesh_base, mesh_thick, mesh_surface]
labels = ['Bed (m)', 'Base (m)', 'Thickness (m)', 'Surface (m)']
mtri = Triangulation(md.mesh.x, md.mesh.y, md.mesh.elements-1)
fig.subplots_adjust(hspace=0.3)


pc = axs.flat[0].tripcolor(mtri, mesh_bed, vmin=-2000, vmax=2000, cmap=cmocean.cm.topo)
fig.colorbar(pc, label='Bed (m)')

pc = axs.flat[1].tripcolor(mtri, mesh_base, vmin=-2000, vmax=2000, cmap=cmocean.cm.topo)
fig.colorbar(pc, label='Base (m)')

pc = axs.flat[2].tripcolor(mtri, mesh_thick, vmin=0, vmax=4000, cmap=cmocean.cm.amp)
fig.colorbar(pc, label='Thickness (m)')

pc = axs.flat[3].tripcolor(mtri, mesh_surface, vmin=0, vmax=4000, cmap=cmocean.cm.haline)
fig.colorbar(pc, label='Surface (m)')


for i in range(4):
    ax = axs.flat[i]
    ax.set_aspect('equal')
    ax.tricontour(mtri, mesh_ocean_levelset, levels=(0,), colors='k',
        linestyles='solid')
fig.savefig('mesh_geometry.png', dpi=400)

# Plot mesh area with grounding line contour
print('Plotting mesh area')
fig,ax = plt.subplots()
mesh = md.mesh
pc = ax.tripcolor(mtri, np.log10(meshdict['area']), vmin=6, vmax=9, 
    cmap=cmocean.cm.matter)
ax.set_aspect('equal')
fig.colorbar(pc, ax=ax, label='log$_{10}$ element area (m$^2$)')
ax.set_xlabel('Easting (km)')
ax.set_ylabel('Northing (km)')
ax.tricontour(mtri, mesh_ocean_levelset, levels=(0,), colors=('k',), linestyles='solid')
fig.savefig('mesh_area.png', dpi=400)
