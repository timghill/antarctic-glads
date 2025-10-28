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

import numpy as np

from matplotlib import pyplot as plt
from matplotlib.tri import Triangulation
from matplotlib.gridspec import GridSpec
import cmocean

import xarray as xr
import zarr as zr
from scipy import interpolate as interp
from utils.tools import reorder_edges_mesh
import pickle

index = 0
s2year = 365*86400

def interpToMesh(mesh):

    # Init model instance and create initial mesh
    md = model()

    # Interpolate velocity onto mesh
    ds = zr.open('../../../../data/Hillebrand_geometry/expAE03_04_q05m50_state.zarr')
    print(list(ds.keys()))
    vx = ds['xvelmean'][index,:]*s2year
    vy = ds['yvelmean'][index,:]*s2year

    initroot = zr.open('../../../../data/Hillebrand_geometry/AIS_4to20km_r01_20220907_relaxed_q5.zarr')
    print(list(initroot.keys()))
    x = initroot['xCell'][:]
    y = initroot['yCell'][:]
    bed = initroot['bedTopography'][:].squeeze()
    print('bed:', bed.shape)

    # vx_obs = interp.griddata((x,y), vx, (mesh['x'], mesh['y']), method='linear')
    # vy_obs = interp.griddata((x,y), vy, (mesh['x'], mesh['y']), method='linear')

    # # vx_obs=InterpFromGridToMesh(x,y,np.flipud(vx),mesh['x'],mesh['y'],100)
    # # vy_obs=InterpFromGridToMesh(x,y,np.flipud(vy),mesh['x'],mesh['y'],100)
    # vel_obs=np.sqrt(vx_obs**2+vy_obs**2)
    # np.save('vx.npy', vx_obs)
    # np.save('vy.npy', vy_obs)

    base = ds['lowerSurface'][index,:]
    thickness = ds['thickness'][index, :]
    
    interp2mesh = lambda z: interp.griddata((x,y), z, (mesh['x'], mesh['y']))
    # Step 1: find floating ice
    mesh_bed = interp2mesh(bed)
    mesh_base = interp2mesh(base)
    iceshelf = (mesh_base - 10) > mesh_bed
    # print('iceshelf:', iceshelf.shape)

    # Step 2: Modelled surface elevation
    mesh_thick = interp2mesh(thickness)
    mesh_surface = mesh_base + mesh_thick

    # Step 3: Use the previously interpolated bed for grounded ice
    # ref_bed = np.load('../../../G-H/data/geom/bed.npy')
    # ref_bed = mesh_bed
    # np.save('bed.npy', ref_bed)
    # mesh_base[~iceshelf] = ref_bed[~iceshelf]

    # Step 4: Adjust ice base for floating ice
    # mesh_thick = mesh_surface - mesh_base
    # floating = np.logical_and(iceshelf, mesh_surface>0)
    # di = md.materials.rho_ice/md.materials.rho_water
    # mesh_thick[floating] = 1/(1-di)*mesh_surface[floating]
    # mesh_base[floating]  = mesh_surface[floating]  - mesh_thick[floating] 

    mesh_ocean_levelset = np.ones((mesh['numberofvertices']), dtype=int)
    # print('mesh_ocean_levelset:', mesh_ocean_levelset.shape)
    mesh_ocean_levelset[iceshelf] = -1
    # mesh_ocean_levelset[mesh_surface<5] = -1
    

    # # No interior floating ice
    # mesh_ocean_levelset[mesh_surface>=1000] = 1

    # # No ice where thickness is ~=0
    mesh_ice_levelset = -1*np.ones(mesh['numberofvertices']).astype(int)
    # # mesh_ice_levelset[np.logical_and(mesh_ocean_levelset==-1, mesh_thick<5)] = 1
    # mesh_ice_levelset[mesh_thick<5] = 1
    # mesh_ice_levelset[mesh_surface<5] = 1
    # mesh_ice_levelset[mesh_surface>1000] = -1

    # Final processing...
    # mesh_base[mesh_ice_levelset==1] = ref_bed[mesh_ice_levelset==1]
    # mesh_surface[mesh_ice_levelset==1] = ref_bed[mesh_ice_levelset==1]
    # mesh_surface[mesh_surface<mesh_base] = mesh_base[mesh_surface<mesh_base]
    # mesh_thick = mesh_surface - mesh_base

    # Save fields
    # np.save('bed.npy', mesh_bed)
    np.save('base.npy', mesh_base)
    np.save('bed.npy', mesh_bed)
    np.save('thick.npy', mesh_thick)
    np.save('surface.npy', mesh_surface)
    # np.save('ocean_levelset.npy', mesh_ocean_levelset)
    # np.save('ice_levelset.npy', mesh_ice_levelset)
    return md

def plot_mesh(meshdict):
    fig,axs = plt.subplots(nrows=2, ncols=2, figsize=(10,8))
    features = [
        np.load('bed.npy'),
        np.load('base.npy'),
        np.load('thick.npy'),
        np.load('surface.npy'),
    ]
    labels = ['Bed (m)', 'Base (m)', 'Thickness (m)', 'Surface (m)']
    mtri = Triangulation(mesh['x'], mesh['y'], mesh['elements']-1)
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
    mesh_ice_levelset = np.load('ice_levelset.npy')
    # meshdict = np.load(meshfile, allow_pickle=True)
    for i in range(4):
        ax = axs.flat[i]
        ax.set_aspect('equal')
        ax.tricontour(mtri, mesh_ocean_levelset, levels=(0,), colors='k',
            linestyles='solid')
        ax.tricontour(mtri, mesh_ice_levelset, levels=(0,), colors='w',
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
    ax.tricontour(mtri, mesh_ice_levelset, levels=(0,), colors=('w',), linestyles='solid')
    fig.savefig('mesh_area.png', dpi=400)
    return

def main(expfile='outline.exp', meshfile='mesh.npy', engine='scipy'):
    md = make_mesh(expfile, meshfile, engine)
    plot_mesh(md)
    return

if __name__=='__main__':
    mesh = np.load('../../../G-H/data/geom/mesh.npy', allow_pickle=True)
    interpToMesh(mesh)
    plot_mesh(mesh)

