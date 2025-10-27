"""
Compute geometric features for all of Antarctica using the
BedMachine product grid
"""
import os
import pickle
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.tri import Triangulation
import cmocean
from scipy import sparse
from scipy import interpolate as interp
import xarray as xr

stride = 1

# def _gldist(mesh, bed, surface):
def _gldist(xx, yy, mask, bed, surface):
    """Euclidean distance from mesh nodes to grounding line
    """
    # rhow = 1028
    # rhoi = 910
    # h_buoyancy = (rhow-rhoi)/rhoi * surface
    # h_boundary = surface[mesh['vertexonboundary']==1]
    # gl = np.where(h_boundary<=(h_buoyancy + 200))[0]

    # glx = mesh['x'][gl,None]
    # gly = mesh['y'][gl,None]
    # print('xx:', xx.shape)

    mask[mask>3] = 2
    xFloating = xx[mask==3].astype(np.float32)[::64]
    yFloating = yy[mask==3].astype(np.float32)[::64]
    xGrounded = xx[mask==2].astype(np.float32)
    yGrounded = yy[mask==2].astype(np.float32)
    print('xGrounded:', xGrounded.shape)
    print('xFloating:', xFloating.shape)

    ddmin = np.zeros(xGrounded.shape, dtype=np.float32)
    batches = 10000
    for i in range(batches):
        print('batch', i)
        dxi = xFloating[:,None] - xGrounded[i::batches]
        dyi = yFloating[:,None] - yGrounded[i::batches]
        dd = np.sqrt(dxi**2 + dyi**2)
        print('dd:', dd.shape)
        ddmin[i::batches] = np.nanmin(dd, axis=0)

    ddmap = np.nan*np.zeros(xx.shape, dtype=np.float32)
    ddmap[mask==2] = ddmin
    return ddmap

# def _flow_accumulation(mesh, phi, melt, levelset, verbose=False, step=250):
#     """Mesh flow accumulation
#     """
#     acc = 0*phi
#     conn = mesh['elements']-1
#     mdot = np.mean(melt[:,None][conn,:], axis=1).squeeze()
#     if verbose:
#         print('mdot:', mdot.shape)
    
#     # Assign initial melt volume from elements to nodes
#     if verbose:
#         print('Assigning initial melt')
#     for element in range(mesh['numberofelements']):
#         phi_neigh = phi[conn[element,:]]
#         if np.all(np.isnan(phi_neigh)):
#             pass
#         else:
#             ixmin = np.nanargmin(phi_neigh)
#             acc[conn[element, ixmin]] = mesh['area'][element]*mdot[element]
    
#     flowacc = 0*phi
    
#     # if verbose:
#         # print('Plotting tricontourf')
#     # fig,ax = plt.subplots()
#     # mtri = Triangulation(mesh['x'], mesh['y'], conn)
#     # ax.tricontourf(mtri, levelset, levels=(-1.5, 0, 1.5), colors=('gray', 'lightgray'))
#     # ax.set_aspect('equal')
    
#     if verbose:
#         print('Looping over starting nodes')
#     yts = 365*86400
#     paths = {}
#     maxiter = 1000
#     groundedice = np.where(levelset==1)[0]
#     for start in groundedice[::step]:
#         phicopy = phi.copy()
#         paths[start] = []
#         nodenum = start.copy()
#         iters = 0
#         done = False
#         while not done and iters<maxiter:
#             flowacc[nodenum] += acc[start]/yts
#             phicopy[nodenum] = np.nan
#             edgenums = np.where(np.any(mesh['connect_edge']==nodenum, axis=1))[0]
#             neigh_nodenums = mesh['connect_edge'][edgenums]
#             neigh_nodenums = neigh_nodenums[neigh_nodenums!=nodenum]

#             phi_neigh = phicopy[neigh_nodenums]
#             if np.all(np.isnan(phi_neigh)) or np.any(levelset[neigh_nodenums]==-1):
#                 done = True
#             else:
#                 next_nodenum = neigh_nodenums[np.nanargmin(phi_neigh)]
#                 paths[start].append(next_nodenum)
#                 nodenum = next_nodenum
#             iters+=1
#         if iters>=maxiter:
#             if verbose:
#                 print('Reached maxiters')
#         # xx = mesh['x'][paths[start]]
#         # yy = mesh['y'][paths[start]]
#         # ax.plot(xx, yy)
    
#     # fig.savefig('init_flowacc.png', dpi=400)
    
#     # fig,ax = plt.subplots()
#     # flowmap = ax.tripcolor(mtri, flowacc)
#     # ax.set_title('Flow accumulation (m^3/s)')
#     # ax.set_aspect('equal')
#     # fig.colorbar(flowmap, label='Discharge (m^3/s)')
#     # fig.savefig('init_flowpaths.png', dpi=400)

#     return flowacc


# def _matrix_flow_accumulation(mesh, phi, melt, levelset, verbose=False, step=250):
#     """Mesh flow accumulation
#     """
#     acc = 0*phi
#     conn = mesh['elements']-1
#     mdot = np.mean(melt[:,None][conn,:], axis=1).squeeze()
#     if verbose:
#         print('mdot:', mdot.shape)
    
#     # Assign initial melt volume from elements to nodes
#     if verbose:
#         print('Assigning initial melt')
#     for element in range(mesh['numberofelements']):
#         phi_neigh = phi[conn[element,:]]
#         if np.all(np.isnan(phi_neigh)):
#             pass
#         else:
#             ixmin = np.nanargmin(phi_neigh)
#             acc[conn[element, ixmin]] = mesh['area'][element]*mdot[element]
    
#     flowacc = 0*phi

#     # Compute node adjacency
#     # print('Constructing node adjacency list')
#     nv = mesh['numberofvertices']
#     adjacent_nodes = []
#     for i in range(nv):
#         if i%1000==0:
#             print(i)
#         edgenums = np.where(np.any(mesh['connect_edge']==i, axis=1))[0]
#         neigh_nodenums = mesh['connect_edge'][edgenums]
#         neigh_nodenums = neigh_nodenums[neigh_nodenums!=i]
#         adjacent_nodes.append(neigh_nodenums)

#     # if verbose:
#         # print('Plotting tricontourf')
#     # fig,ax = plt.subplots()
#     # mtri = Triangulation(mesh['x'], mesh['y'], conn)
#     # ax.tricontourf(mtri, levelset, levels=(-1.5, 0, 1.5), colors=('gray', 'lightgray'))
#     # ax.set_aspect('equal')
    
#     if verbose:
#         print('Looping over starting nodes')
#     yts = 365*86400
#     paths = {}
#     maxiter = 1000
#     groundedice = np.where(levelset==1)[0]
#     for start in groundedice[::step]:
#         # if start%1000==0:
#         #     print(start)
#         phicopy = phi.copy()
#         paths[start] = []
#         nodenum = start.copy()
#         iters = 0
#         done = False
#         while not done and iters<maxiter:
#             flowacc[nodenum] += acc[start]/yts
#             phicopy[nodenum] = np.nan
#             # edgenums = np.where(np.any(mesh['connect_edge']==nodenum, axis=1))[0]
#             # neigh_nodenums = mesh['connect_edge'][edgenums]
#             neigh_nodenums = adjacent_nodes[nodenum]

#             phi_neigh = phicopy[neigh_nodenums]
#             if np.all(np.isnan(phi_neigh)) or np.any(levelset[neigh_nodenums]==-1):
#                 done = True
#             else:
#                 next_nodenum = neigh_nodenums[np.nanargmin(phi_neigh)]
#                 paths[start].append(next_nodenum)
#                 nodenum = next_nodenum
#             iters+=1
#         if iters>=maxiter:
#             if verbose:
#                 print('Reached maxiters')

#     return flowacc

# def _binned_flow_accumulation(mesh, phi, melt):
#     acc = 0*phi
#     yts = 365*86400
#     conn = mesh['elements']-1
#     mdot = np.mean(melt[:,None][conn,:], axis=1).squeeze()/yts
#     area = mesh['area']
#     phiel = np.mean(phi[:,None][conn,:], axis=1).squeeze()

#     groundedice = np.where(~np.isnan(phi))[0]
#     for k in groundedice:
#         acc[k] = np.sum((mdot*area)[phiel>phi[k]])
#     return acc

def _basal_melt(xx, yy):

    datadir = os.path.abspath('../../data/lanl-mali')
    ## BASAL VELOCITY
    ais_outputs = os.path.join(datadir, 'AIS_4kmto20km_hist04.nc')
    with xr.open_dataset(ais_outputs, engine='scipy') as ais:
        ub = ais['uReconstructX'][0, :, -1]
        vb = ais['uReconstructY'][0, :, -1]
        temp = np.mean(ais['temperature'][0, :, :], axis=-1)

        xMali = ais['xCell']
        yMali = ais['yCell']

    # vvMali = np.sqrt(ub**2 + vb**2)*yts
    # vvMesh = interp.griddata((xMali, yMali), vvMali, 
    #     (xx,yy), method='nearest')
    
    # tempMesh = interp.griddata((xMali, yMali), temp, 
    #     (xx,yy), method='nearest')
    # tempMesh[tempMesh==0] = np.median(tempMesh)
    # # np.save('temperature_mali.npy', tempMesh)

    ## BASAL MELT RATE
    output_state = os.path.abspath(os.path.join(datadir, 'output_state_2060.nc'))
    with xr.open_dataset(output_state, engine='scipy') as output:
        basalmeltMali = output['basalMeltInput'][0]

    kgm2s_to_mwea = 365*86400/910
    basalmeltMali *= kgm2s_to_mwea

    basalmeltMesh = interp.griddata((xMali, yMali), basalmeltMali, 
        (xx, yy), method='nearest')

    # np.save('basal_melt_mali.npy', basalmeltMesh)
    return basalmeltMesh
    

def get_features(bedmachine):
    # basin_dir = f'../../issm/{basin}/'
    # print(basin_dir)
    # meshfile = os.path.join(basin_dir, 'data/geom/mesh.npy')
    # mesh = np.load(meshfile, allow_pickle=True)

    # levelset = np.load(
    #     os.path.join(basin_dir, 'data/geom/ocean_levelset.npy')
    # )
    bm = xr.open_dataset(bedmachine)
    x = bm['x'][::stride].values
    y = bm['y'][::stride].values
    xx, yy = np.meshgrid(x, y)
    mask = bm['mask'][::stride, ::stride].values
    mask[mask>3] = 2

    # Store all features in dictionary features
    features = {}

    # Surface, bed and thickness
    print('\tSurface, bed and thickness...', end=' ', flush=True)
    bed = bm['bed'][::stride, ::stride].values
    surface = bm['surface'][::stride, ::stride].values
    thick = bm['thickness'][::stride, ::stride].values
    print('done')

    levelset = np.zeros(mask.shape, dtype=int)
    levelset[mask==2] = 1

    bed[levelset<1] = np.nan
    surface[levelset<1] = np.nan
    thick[levelset<1] = np.nan
    
    features['bed'] = bed
    features['surface'] = surface
    features['thickness'] = thick
    
    # Grounding line distance
    print('\tGrounding line distance...', end=' ', flush=True)
    features['grounding_line_distance'] = _gldist(xx, yy, mask, bed, surface)
    print('done')

    # Local basal melt rate
    print('\tLocal basal melt rate...', end=' ', flush=True)
    # basal_melt = np.load(
    #     os.path.join(basin_dir, 'data/lanl-mali/basal_melt_mali.npy')
    # )
    features['basal_melt'] = np.log10(_basal_melt(xx, yy))
    print('basal_melt:', features['basal_melt'].shape)
    features['basal_melt'][levelset==0] = np.nan
    print('done')

    # Hydraulic potential
    rho_ice = 910
    rho_freshwater = 1000
    g = 9.81
    shreve_potential = rho_freshwater*g*bed + rho_ice*g*thick
    features['potential'] = shreve_potential

    # # Flow accumulation
    # print('\tFlow accumulation subroutine...', end=' ', flush=True)
    # flowacc = _matrix_flow_accumulation(mesh, shreve_potential, basal_melt, levelset, step=1)
    # features['flow_accumulation'] = flowacc
    # print('done')
    # # features['flow_accumulation'] = 0*shreve_potential[levelset>0]

    # print('\tBinned flow accumulation...', end=' ', flush=True)
    # binned_flow = _binned_flow_accumulation(mesh, shreve_potential, basal_melt)
    # features['binned_flow_accumulation'] = binned_flow[levelset>0]
    # print('done')
    return features

def save_all_features(bedmachine):
    basin = 'AIS'
    # for basin in basins:
    features = get_features(bedmachine)
    basinfile = f'features_{basin}.pkl'
    with open(basinfile, 'wb') as basin_pkl:
        pickle.dump(features, basin_pkl)
    return

def plot_features(plotskip=2):
    # for basin in basins:
    basin = 'AIS'
    features = np.load(f'features_{basin}.pkl', allow_pickle=True)
    print('features:', features.keys())
    # basin_dir = f'../../issm/{basin}/'
    # print(basin_dir)
    # meshfile = os.path.join(basin_dir, 'data/geom/mesh.npy')
    # mesh = np.load(meshfile, allow_pickle=True)
    # levelset = np.load(
        # os.path.join(basin_dir, 'data/geom/ocean_levelset.npy')
    # )
    print('Opening BedMachine')
    bm = xr.open_dataset(bedmachine)
    x = bm['x'][::stride].values
    y = bm['y'][::stride].values
    xx, yy = np.meshgrid(x, y)
    mask = bm['mask'][::stride, ::stride].values
    mask[mask>3] = 2

    print('Suface, bed and thickness')
    # Surface, bed and thickness
    # print('\tSurface, bed and thickness...', end=' ', flush=True)
    bed = bm['bed'][::stride, ::stride].values
    surface = bm['surface'][::stride, ::stride].values
    thick = bm['thickness'][::stride, ::stride].values
    # print('done')

    print('Levelset')
    levelset = np.zeros(mask.shape, dtype=int)
    levelset[mask==2] = 1

    print('Masking non-ice areas')
    bed[levelset<1] = np.nan
    surface[levelset<1] = np.nan
    thick[levelset<1] = np.nan

    # mtri = Triangulation(mesh['x'], mesh['y'], mesh['elements']-1)

    print('Init figure')
    fig,axs = plt.subplots(nrows=2, ncols=4, figsize=(10, 8))
    axf = axs.flat
    # bed = np.nan*np.zeros(mesh['numberofvertices'])
    # bed[levelset>0] = features['bed']
    # m0 = axf[0].tripcolor(mtri, bed, 
        # vmin=-2e3, vmax=2e3, cmap=cmocean.cm.topo)
    print('first pcolor bed')
    m0 = axf[0].pcolormesh(xx[::plotskip, ::plotskip], yy[::plotskip, ::plotskip], bed[::plotskip, ::plotskip], 
        vmin=-2e3, vmax=2e3, cmap=cmocean.cm.topo
    )
    # axf[0].set_title('Bed elevation')
    fig.colorbar(m0, location='top', pad=0, shrink=0.8, 
        label='Bed elevation (m)')

    # thick = np.nan*np.zeros(mesh['numberofvertices'])
    # thick[levelset>0] = features['thickness']
    print('pcolor thick')
    m1 = axf[1].pcolormesh(xx[::plotskip, ::plotskip], yy[::plotskip, ::plotskip], thick[::plotskip, ::plotskip],
        vmin=0, vmax=4e3, cmap=cmocean.cm.amp)
    # axf[1].set_title('Thickness')
    fig.colorbar(m1, location='top', pad=0, shrink=0.8, 
        label='Ice thickness (m)')
    
    # surface = np.nan*np.zeros(mesh['numberofvertices'])
    # surface[levelset>0] = features['surface']
    print('pcolor surface')
    m2 = axf[2].pcolormesh(xx[::plotskip, ::plotskip], yy[::plotskip, ::plotskip], surface[::plotskip, ::plotskip],
        vmin=0, vmax=4e3, cmap=cmocean.cm.haline)
    # axf[2].set_title('Surface elevation')
    fig.colorbar(m2, location='top', pad=0, shrink=0.8,
        label='Surface elevation (m)')
    
    # gldist = np.nan*np.zeros(mesh['numberofvertices'])
    print('pcolor gldist')
    gldist = features['grounding_line_distance']/1e3
    m3 = axf[3].pcolormesh(xx[::plotskip, ::plotskip], yy[::plotskip, ::plotskip], gldist[::plotskip, ::plotskip],
        vmin=0, cmap=cmocean.cm.deep)
    # axf[3].set_title('Grounding line distance')
    fig.colorbar(m3, location='top', pad=0, shrink=0.8,
        label='Grounding line distance (km)')
    
    # basal_melt = np.nan*np.zeros(mesh['numberofvertices'])
    basal_melt = features['basal_melt']
    print('pcolor basal melt')
    m4 = axf[4].pcolormesh(xx[::plotskip, ::plotskip], yy[::plotskip, ::plotskip], basal_melt[::plotskip, ::plotskip],
        vmin=-3, cmap=cmocean.cm.rain, vmax=0)
    # axf[4].set_title('Basal melt rate (m/a)')
    fig.colorbar(m4, location='top', pad=0, shrink=0.8,
        label='log$_{10}$ Basal melt rate (m/a)')

    # potential = np.nan*np.zeros(mesh['numberofvertices'])
    print('pcolor potential')
    potential = features['potential']
    m5 = axf[5].pcolormesh(xx[::plotskip, ::plotskip], yy[::plotskip, ::plotskip], potential[::plotskip, ::plotskip],
        vmin=0, cmap=cmocean.cm.dense)
    # axf[5].set_title('Shreve potential')
    fig.colorbar(m5, location='top', pad=0, shrink=0.8,
        label='Shreve potential')

    # flowacc = np.nan*np.zeros(mesh['numberofvertices'])
    # flowacc[levelset>0] = features['flow_accumulation']
    # m6 = axf[6].tripcolor(mtri, flowacc,
    #     vmin=0, vmax=25, cmap=cmocean.cm.thermal)
    # # axf[6].set_title('Flow accumulation')
    # fig.colorbar(m6, location='top', pad=0, shrink=0.8,
    #     label='Flow accumulation (m$^3$ s$^{-1}$)')

    # binacc = np.nan*np.zeros(mesh['numberofvertices'])
    # binacc[levelset>0] = features['binned_flow_accumulation']
    # m6 = axf[7].tripcolor(mtri, binacc,
    #     vmin=0, cmap=cmocean.cm.thermal)
    # # axf[6].set_title('Flow accumulation')
    # fig.colorbar(m6, location='top', pad=0, shrink=0.8,
    #     label='Binned melt rate (m$^3$ s$^{-1}$)')

    for ax in axf:
        ax.set_aspect('equal')
        ax.spines[['left', 'bottom', 'right', 'top']].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.subplots_adjust(hspace=0, wspace=0., left=0.05, right=0.95, 
        top=0.95, bottom=0.05)
    fig.savefig(f'features_{basin}.png', dpi=400)

if __name__=='__main__':
    bedmachine = '../../data/bedmachine/BedMachineAntarctica-v3.nc'
    save_all_features(bedmachine)
    plot_features()
