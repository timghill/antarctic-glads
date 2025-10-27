"""
Make ISSM mesh
"""

import os
import sys

issmdir = os.getenv('ISSM_DIR')
sys.path.append(issmdir + '/bin')
sys.path.append(issmdir + '/lib')

from issmversion import issmversion
from model import model
from triangle import *
from bamg import *
from GetAreas import GetAreas
from InterpFromGridToMesh import InterpFromGridToMesh

from matplotlib import pyplot as plt
from matplotlib.tri import Triangulation
from matplotlib.gridspec import GridSpec
import cmocean

import xarray as xr
import zarr as zr
from scipy import interpolate as interp
from utils.tools import reorder_edges_mesh
import pickle

index = 101
s2year = 365*86400

def make_mesh(expfile, meshfile, engine,
    hmax=25e3, hmin=2.5e3, gradation=1.2, vel_thresh=50):
    """
    # Mesh parameters
    hmax=25e3        # maximum element size of the final mesh
    hmin=2.5e3       # minimum element size of the final mesh
    gradation=1.2    # maximum size ratio between two neighboring elements
    # err=8            # maximum error between interpolated and control field
    vel_thresh = 50 # Velocity (m/a) below which to refine the mesh
    # aniso = 1.2
    """
    # Parameters for an initial mesh
    hmax_init = 10e3
    hmin_init = 10e3

    # Init model instance and create initial mesh
    md = model()
    md = bamg(md, 'domain', expfile, 'hmax', hmax_init,
        'hmin', hmin_init, 'gradation', gradation)
    print('Made draft mesh with numberofvertices:', md.mesh.numberofvertices)

    # Interpolate velocity onto mesh
    ds = zr.open('../../../../data/Hillebrand_geometry/expAE03_04_q05m50_state.zarr')
    print(list(ds.keys()))
    vx = ds['xvelmean'][index,:]*s2year
    vy = ds['yvelmean'][index,:]*s2year

    # initstore = zr.storage.LocalStore('/home/tghill/projects/def-gflowers/tghill/antarctic-glads/data/AIS_4to20km_r01_20220907_relaxed_q5.zarr')
    # print(initstore)
    initroot = zr.open('../../../../data/Hillebrand_geometry/AIS_4to20km_r01_20220907_relaxed_q5.zarr')
    print(list(initroot.keys()))
    x = initroot['xCell'][:]
    y = initroot['yCell'][:]
    bed = initroot['bedTopography'][:].squeeze()
    xv = initroot['xVertex'][:]
    yv = initroot['yVertex'][:]
    print('bed:', bed.shape)

    vx_obs = interp.griddata((x,y), vx, (md.mesh.x, md.mesh.y), method='linear')
    vy_obs = interp.griddata((x,y), vy, (md.mesh.x, md.mesh.y), method='linear')

    # vx_obs=InterpFromGridToMesh(x,y,np.flipud(vx),md.mesh.x,md.mesh.y,100)
    # vy_obs=InterpFromGridToMesh(x,y,np.flipud(vy),md.mesh.x,md.mesh.y,100)
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
    print('   Min edge length:', edge_length.min())
    print('Median edge length:', np.median(edge_length))
    print('   Max edge length:', edge_length.max())

    with open('mesh.npy', 'wb') as meshout:
        pickle.dump(meshdict, meshout)

    # Interpolate velocity

    vx_obs = interp.griddata((x,y), vx, (md.mesh.x, md.mesh.y), method='linear')
    vy_obs = interp.griddata((x,y), vy, (md.mesh.x, md.mesh.y), method='linear')
    np.save('vx.npy', vx_obs)
    np.save('vy.npy', vy_obs)

    # Interpolate geometry features onto the new mesh
    # print('Interpolating geometry')
    # bm = xr.open_dataset('../../../../data/bedmachine/BedMachineAntarctica-v3.nc')

    # stride = 2
    # x = np.array(bm['x'][::stride].values).astype(int)
    # y = np.array(bm['y'][::stride].values).astype(int)[::-1]
    # mask = np.flipud(np.array(bm['mask'][::stride, ::stride].values).astype(int))
    # bed = np.flipud(np.array(bm['bed'][::stride, ::stride].values).astype(float))
    # thick = np.flipud(np.array(bm['thickness'][::stride, ::stride].values).astype(float))
    # surf = np.flipud(np.array(bm['surface'][::stride, ::stride].values).astype(float))

    # mesh_bm_mask = InterpFromGridToMesh(x, y, mask, md.mesh.x, md.mesh.y, 0)
    # mesh_ice_levelset = -1*np.ones(md.mesh.numberofvertices).astype(int)
    # mesh_bed = InterpFromGridToMesh(x, y, bed, md.mesh.x, md.mesh.y, 0)
    # mesh_thick = InterpFromGridToMesh(x, y, thick, md.mesh.x, md.mesh.y, 0)
    # mesh_surface = InterpFromGridToMesh(x, y, surf, md.mesh.x, md.mesh.y, 0)
    # print('mesh_surface min:', np.min(mesh_surface))

    base = ds['lowerSurface'][index,:]
    thickness = ds['thickness'][index, :]
    # surface = ds['surface'][index, :]
    surface = base + thickness
    
    interp2mesh = lambda z: interp.griddata((x,y), z, (md.mesh.x, md.mesh.y))
    mesh_bed = interp.griddata((x, y), bed, (md.mesh.x, md.mesh.y))
    mesh_thick = interp2mesh(thickness)
    mesh_base = interp2mesh(base)
    mesh_surface = interp2mesh(surface)
    mesh_surface = mesh_base + mesh_thick
    # mesh_surfce = interp2mesh(surface)
    # mesh_surface = mesh_base + mesh_thick

    # Define floating/grounded based on flotation
    # rhoice = 917
    # rhowater = 1023
    # hf = rhoice/rhowater*mesh_thick
    # mesh_ocean_levelset[mesh_thick<=hf] = -1
    # mesh_ocean_levelset[mesh_thick>hf] = 1
    mesh_ocean_levelset = np.ones(md.mesh.numberofvertices, dtype=int)
    iceshelf = (mesh_base - 10) > mesh_bed
    mesh_ocean_levelset[iceshelf] = -1
    mesh_ocean_levelset[mesh_surface<5] = -1

    # mesh_base[~iceshelf] = mesh_bed[~iceshelf]
    # mesh_thick = mesh_surface - mesh_base


    # Post-process thickness where thickness<1
    # min_thick = 10
    # mesh_base = mesh_surface - mesh_thick
    # pos = mesh_thick<=min_thick
    # mesh_thick[pos] = min_thick
    # mesh_surface = mesh_base + mesh_thick
    
    # No interior floating ice
    mesh_ocean_levelset[mesh_surface>=1000] = 1


    mesh_ice_levelset = -1*np.ones(md.mesh.numberofvertices).astype(int)
    mesh_ice_levelset[np.logical_and(mesh_ocean_levelset==-1, mesh_thick<5)] = 1

    # Ice shelf base
    # di = md.materials.rho_ice/md.materials.rho_water
    # iceshelf = mesh_ocean_levelset<0
    # mesh_thick[iceshelf] = mesh_surface[iceshelf] - mesh_base[iceshelf]

    # Always true
    # mesh_base = mesh_surface - mesh_thick

    # Save fields
    np.save('bed.npy', mesh_bed)
    np.save('base.npy', mesh_base)
    np.save('thick.npy', mesh_thick)
    np.save('surface.npy', mesh_surface)
    np.save('ocean_levelset.npy', mesh_ocean_levelset)
    np.save('ice_levelset.npy', mesh_ice_levelset)
    return md

def plot_mesh(md, meshfile='mesh.npy'):
    fig,axs = plt.subplots(nrows=2, ncols=2, figsize=(10,8))
    features = [
        np.load('bed.npy'),
        np.load('base.npy'),
        np.load('thick.npy'),
        np.load('surface.npy'),
    ]
    labels = ['Bed (m)', 'Base (m)', 'Thickness (m)', 'Surface (m)']
    mtri = Triangulation(md.mesh.x, md.mesh.y, md.mesh.elements-1)
    fig.subplots_adjust(hspace=0.3)

    pc = axs.flat[0].tripcolor(mtri, features[0], vmin=-2000, vmax=2000, cmap=cmocean.cm.topo)
    fig.colorbar(pc, label='Bed (m)')

    pc = axs.flat[1].tripcolor(mtri, features[1], vmin=-2000, vmax=2000, cmap=cmocean.cm.topo)
    fig.colorbar(pc, label='Base (m)')

    pc = axs.flat[2].tripcolor(mtri, features[2], vmin=0, vmax=4000, cmap=cmocean.cm.amp)
    fig.colorbar(pc, label='Thickness (m)')

    pc = axs.flat[3].tripcolor(mtri, features[3], vmin=0, vmax=4000, cmap=cmocean.cm.haline)
    fig.colorbar(pc, label='Surface (m)')

    mesh_ocean_levelset = np.load('ocean_levelset.npy')
    meshdict = np.load(meshfile, allow_pickle=True)
    for i in range(4):
        ax = axs.flat[i]
        ax.set_aspect('equal')
        ax.tricontour(mtri, mesh_ocean_levelset, levels=(0,), colors='k',
            linestyles='solid')
    fig.savefig('mesh_geometry.png', dpi=400)

    # Plot mesh area with grounding line contour
    print('Plotting mesh area')
    fig,ax = plt.subplots()
    pc = ax.tripcolor(mtri, np.log10(meshdict['area']/1e6), vmin=0, vmax=3, 
        cmap=cmocean.cm.matter)
    ax.set_aspect('equal')
    fig.colorbar(pc, ax=ax, label='log$_{10}$ element area (km$^2$)')
    ax.set_xlabel('Easting (km)')
    ax.set_ylabel('Northing (km)')
    ax.tricontour(mtri, mesh_ocean_levelset, levels=(0,), colors=('k',), linestyles='solid')
    fig.savefig('mesh_area.png', dpi=400)
    return

def main(expfile='outline.exp', meshfile='mesh.npy', engine='scipy'):
    md = make_mesh(expfile, meshfile, engine)
    plot_mesh(md)
    return

if __name__=='__main__':
    main()  
