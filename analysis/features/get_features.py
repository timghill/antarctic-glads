"""
Compute geometric features for each train/test basin
"""
import os
import pickle
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.tri import Triangulation
import cmocean
from scipy import sparse

def _gldist(mesh, mask, bed, surface):
    """Euclidean distance from mesh nodes to grounding line
    """
    # bz = surface[mesh['vertexonboundary']==1]
    # rhow = 1028
    # rhoi = 910
    # h_buoyancy = -(rhow-rhoi)/rhoi * bz
    # h_boundary = surface[mesh['vertexonboundary']==1]
    # gl = np.where(h_boundary<=(h_buoyancy + 200))[0][::4]

    # glx = mesh['x'][gl,None]
    # gly = mesh['y'][gl,None]

    # dx = glx - mesh['x']
    # dy = gly - mesh['y']
    
    # dd = np.sqrt(dx**2 + dy**2)
    # ddmin = np.min(dd, axis=0)
    # return ddmin

    mask[mask>3] = 2
    xFloating = mesh['x'][mask==3].astype(np.float32)
    yFloating = mesh['y'][mask==3].astype(np.float32)

    xGrounded = mesh['x'][mask==2].astype(np.float32)
    yGrounded = mesh['y'][mask==2].astype(np.float32)

    print('Floating:', xFloating.shape)
    print('Grounded:', xGrounded.shape)
    dd = np.sqrt(dx**2 + dy**2)
    ddmin = np.min(dd, axis=0)
    return ddmin

def _flow_accumulation(mesh, phi, melt, levelset, verbose=False, step=250):
    """Mesh flow accumulation
    """
    acc = 0*phi
    conn = mesh['elements']-1
    mdot = np.mean(melt[:,None][conn,:], axis=1).squeeze()
    if verbose:
        print('mdot:', mdot.shape)
    
    # Assign initial melt volume from elements to nodes
    if verbose:
        print('Assigning initial melt')
    for element in range(mesh['numberofelements']):
        phi_neigh = phi[conn[element,:]]
        if np.all(np.isnan(phi_neigh)):
            pass
        else:
            ixmin = np.nanargmin(phi_neigh)
            acc[conn[element, ixmin]] = mesh['area'][element]*mdot[element]
    
    flowacc = 0*phi
    
    # if verbose:
        # print('Plotting tricontourf')
    # fig,ax = plt.subplots()
    # mtri = Triangulation(mesh['x'], mesh['y'], conn)
    # ax.tricontourf(mtri, levelset, levels=(-1.5, 0, 1.5), colors=('gray', 'lightgray'))
    # ax.set_aspect('equal')
    
    if verbose:
        print('Looping over starting nodes')
    yts = 365*86400
    paths = {}
    maxiter = 1000
    groundedice = np.where(levelset==1)[0]
    for start in groundedice[::step]:
        phicopy = phi.copy()
        paths[start] = []
        nodenum = start.copy()
        iters = 0
        done = False
        while not done and iters<maxiter:
            flowacc[nodenum] += acc[start]/yts
            phicopy[nodenum] = np.nan
            edgenums = np.where(np.any(mesh['connect_edge']==nodenum, axis=1))[0]
            neigh_nodenums = mesh['connect_edge'][edgenums]
            neigh_nodenums = neigh_nodenums[neigh_nodenums!=nodenum]

            phi_neigh = phicopy[neigh_nodenums]
            if np.all(np.isnan(phi_neigh)) or np.any(levelset[neigh_nodenums]==-1):
                done = True
            else:
                next_nodenum = neigh_nodenums[np.nanargmin(phi_neigh)]
                paths[start].append(next_nodenum)
                nodenum = next_nodenum
            iters+=1
        if iters>=maxiter:
            if verbose:
                print('Reached maxiters')
        # xx = mesh['x'][paths[start]]
        # yy = mesh['y'][paths[start]]
        # ax.plot(xx, yy)
    
    # fig.savefig('init_flowacc.png', dpi=400)
    
    # fig,ax = plt.subplots()
    # flowmap = ax.tripcolor(mtri, flowacc)
    # ax.set_title('Flow accumulation (m^3/s)')
    # ax.set_aspect('equal')
    # fig.colorbar(flowmap, label='Discharge (m^3/s)')
    # fig.savefig('init_flowpaths.png', dpi=400)

    return flowacc


def _matrix_flow_accumulation(mesh, phi, melt, levelset, verbose=False, step=250):
    """Mesh flow accumulation
    """
    acc = 0*phi
    conn = mesh['elements']-1
    mdot = np.mean(melt[:,None][conn,:], axis=1).squeeze()
    if verbose:
        print('mdot:', mdot.shape)
    
    # Assign initial melt volume from elements to nodes
    if verbose:
        print('Assigning initial melt')
    for element in range(mesh['numberofelements']):
        phi_neigh = phi[conn[element,:]]
        if np.all(np.isnan(phi_neigh)):
            pass
        else:
            ixmin = np.nanargmin(phi_neigh)
            acc[conn[element, ixmin]] = mesh['area'][element]*mdot[element]
    
    flowacc = 0*phi

    # Compute node adjacency
    # print('Constructing node adjacency list')
    nv = mesh['numberofvertices']
    adjacent_nodes = []
    for i in range(nv):
        if i%1000==0:
            print(i)
        edgenums = np.where(np.any(mesh['connect_edge']==i, axis=1))[0]
        neigh_nodenums = mesh['connect_edge'][edgenums]
        neigh_nodenums = neigh_nodenums[neigh_nodenums!=i]
        adjacent_nodes.append(neigh_nodenums)

    # if verbose:
        # print('Plotting tricontourf')
    # fig,ax = plt.subplots()
    # mtri = Triangulation(mesh['x'], mesh['y'], conn)
    # ax.tricontourf(mtri, levelset, levels=(-1.5, 0, 1.5), colors=('gray', 'lightgray'))
    # ax.set_aspect('equal')
    
    if verbose:
        print('Looping over starting nodes')
    yts = 365*86400
    paths = {}
    maxiter = 1000
    groundedice = np.where(levelset==1)[0]
    for start in groundedice[::step]:
        # if start%1000==0:
        #     print(start)
        phicopy = phi.copy()
        paths[start] = []
        nodenum = start.copy()
        iters = 0
        done = False
        while not done and iters<maxiter:
            flowacc[nodenum] += acc[start]/yts
            phicopy[nodenum] = np.nan
            # edgenums = np.where(np.any(mesh['connect_edge']==nodenum, axis=1))[0]
            # neigh_nodenums = mesh['connect_edge'][edgenums]
            neigh_nodenums = adjacent_nodes[nodenum]

            phi_neigh = phicopy[neigh_nodenums]
            if np.all(np.isnan(phi_neigh)) or np.any(levelset[neigh_nodenums]==-1):
                done = True
            else:
                next_nodenum = neigh_nodenums[np.nanargmin(phi_neigh)]
                paths[start].append(next_nodenum)
                nodenum = next_nodenum
            iters+=1
        if iters>=maxiter:
            if verbose:
                print('Reached maxiters')

    return flowacc

def _binned_flow_accumulation(mesh, phi, melt):
    acc = 0*phi
    yts = 365*86400
    conn = mesh['elements']-1
    mdot = np.mean(melt[:,None][conn,:], axis=1).squeeze()/yts
    area = mesh['area']
    phiel = np.mean(phi[:,None][conn,:], axis=1).squeeze()

    groundedice = np.where(~np.isnan(phi))[0]
    for k in groundedice:
        acc[k] = np.sum((mdot*area)[phiel>phi[k]])
    return acc
    

def get_features(basin):
    basin_dir = f'../../issm/{basin}/'
    print(basin_dir)
    meshfile = os.path.join(basin_dir, 'data/geom/mesh.npy')
    mesh = np.load(meshfile, allow_pickle=True)

    levelset = np.load(
        os.path.join(basin_dir, 'data/geom/ocean_levelset.npy')
    )

    # Store all features in dictionary features
    features = {}

    # Surface, bed and thickness
    print('\tSurface, bed and thickness...', end=' ', flush=True)
    surface = np.load(os.path.join(basin_dir, 'data/geom/surface.npy'))
    bed = np.load(os.path.join(basin_dir, 'data/geom/bed.npy'))
    thick = np.load(os.path.join(basin_dir, 'data/geom/thick.npy'))
    print('done')
    
    features['bed'] = bed[levelset>0]
    features['surface'] = surface[levelset>0]
    features['thickness'] = thick[levelset>0]
    
    # Grounding line distance
    print('\tGrounding line distance...', end=' ', flush=True)
    features['grounding_line_distance'] = _gldist(mesh, bed, surface)[levelset>0]
    print('done')

    # Local basal melt rate
    print('\tLocal basal melt rate...', end=' ', flush=True)
    basal_melt = np.load(
        os.path.join(basin_dir, 'data/lanl-mali/basal_melt_mali.npy')
    )
    features['basal_melt'] = np.log10(1e-6 + basal_melt)[levelset>0]
    print('done')

    # Hydraulic potential
    rho_ice = 910
    rho_freshwater = 1000
    g = 9.81
    shreve_potential = rho_freshwater*g*bed + rho_ice*g*thick
    features['potential'] = shreve_potential[levelset>0]

    # Flow accumulation
    print('\tFlow accumulation subroutine...', end=' ', flush=True)
    flowacc = _matrix_flow_accumulation(mesh, shreve_potential, basal_melt, levelset, step=1)
    features['flow_accumulation'] = flowacc[levelset>0]
    print('done')
    # features['flow_accumulation'] = 0*shreve_potential[levelset>0]

    print('\tBinned flow accumulation...', end=' ', flush=True)
    binned_flow = _binned_flow_accumulation(mesh, shreve_potential, basal_melt)
    features['binned_flow_accumulation'] = binned_flow[levelset>0]
    print('done')
    return features

def save_all_features(basins):
    for basin in basins:
        features = get_features(basin)
        basinfile = f'features_{basin}.pkl'
        with open(basinfile, 'wb') as basin_pkl:
            pickle.dump(features, basin_pkl)
    return

def plot_features(basins):
    for basin in basins:
        features = np.load(f'features_{basin}.pkl', allow_pickle=True)
        basin_dir = f'../../issm/{basin}/'
        print(basin_dir)
        meshfile = os.path.join(basin_dir, 'data/geom/mesh.npy')
        mesh = np.load(meshfile, allow_pickle=True)
        levelset = np.load(
            os.path.join(basin_dir, 'data/geom/ocean_levelset.npy')
        )

        mtri = Triangulation(mesh['x'], mesh['y'], mesh['elements']-1)

        fig,axs = plt.subplots(nrows=2, ncols=4, figsize=(10, 8))
        axf = axs.flat
        bed = np.nan*np.zeros(mesh['numberofvertices'])
        bed[levelset>0] = features['bed']
        m0 = axf[0].tripcolor(mtri, bed, 
            vmin=-2e3, vmax=2e3, cmap=cmocean.cm.topo)
        # axf[0].set_title('Bed elevation')
        fig.colorbar(m0, location='top', pad=0, shrink=0.8, 
            label='Bed elevation (m)')

        thick = np.nan*np.zeros(mesh['numberofvertices'])
        thick[levelset>0] = features['thickness']
        m1 = axf[1].tripcolor(mtri, thick,
            vmin=0, vmax=4e3, cmap=cmocean.cm.amp)
        # axf[1].set_title('Thickness')
        fig.colorbar(m1, location='top', pad=0, shrink=0.8, 
            label='Ice thickness (m)')
        
        surface = np.nan*np.zeros(mesh['numberofvertices'])
        surface[levelset>0] = features['surface']
        m2 = axf[2].tripcolor(mtri, surface,
            vmin=0, vmax=4e3, cmap=cmocean.cm.haline)
        # axf[2].set_title('Surface elevation')
        fig.colorbar(m2, location='top', pad=0, shrink=0.8,
            label='Surface elevation (m)')
        
        gldist = np.nan*np.zeros(mesh['numberofvertices'])
        gldist[levelset>0] = features['grounding_line_distance']/1e3
        m3 = axf[3].tripcolor(mtri, gldist,
            vmin=0, cmap=cmocean.cm.deep)
        # axf[3].set_title('Grounding line distance')
        fig.colorbar(m3, location='top', pad=0, shrink=0.8,
            label='Grounding line distance (km)')
        
        basal_melt = np.nan*np.zeros(mesh['numberofvertices'])
        basal_melt[levelset>0] = features['basal_melt']
        m4 = axf[4].tripcolor(mtri, basal_melt,
            vmin=-3, cmap=cmocean.cm.rain, vmax=0)
        # axf[4].set_title('Basal melt rate (m/a)')
        fig.colorbar(m4, location='top', pad=0, shrink=0.8,
            label='log$_{10}$ Basal melt rate (m/a)')

        potential = np.nan*np.zeros(mesh['numberofvertices'])
        potential[levelset>0] = features['potential']
        m5 = axf[5].tripcolor(mtri, potential,
            vmin=0, cmap=cmocean.cm.dense)
        # axf[5].set_title('Shreve potential')
        fig.colorbar(m5, location='top', pad=0, shrink=0.8,
            label='Shreve potential')

        flowacc = np.nan*np.zeros(mesh['numberofvertices'])
        flowacc[levelset>0] = features['flow_accumulation']
        m6 = axf[6].tripcolor(mtri, flowacc,
            vmin=0, vmax=25, cmap=cmocean.cm.thermal)
        # axf[6].set_title('Flow accumulation')
        fig.colorbar(m6, location='top', pad=0, shrink=0.8,
            label='Flow accumulation (m$^3$ s$^{-1}$)')

        binacc = np.nan*np.zeros(mesh['numberofvertices'])
        binacc[levelset>0] = features['binned_flow_accumulation']
        m6 = axf[7].tripcolor(mtri, binacc,
            vmin=0, cmap=cmocean.cm.thermal)
        # axf[6].set_title('Flow accumulation')
        fig.colorbar(m6, location='top', pad=0, shrink=0.8,
            label='Binned melt rate (m$^3$ s$^{-1}$)')

        for ax in axf:
            ax.set_aspect('equal')
            ax.spines[['left', 'bottom', 'right', 'top']].set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])
        fig.subplots_adjust(hspace=0, wspace=0., left=0.05, right=0.95, 
            top=0.95, bottom=0.05)
        fig.savefig(f'features_{basin}.png', dpi=400)

if __name__=='__main__':
    basins = [
        # 'G-H',
        # 'F-G',
        # 'Ep-F',
        'Cp-D',
        # 'C-Cp',
        # 'B-C',
        # 'Jpp-K',
        # 'J-Jpp',
    ]
    save_all_features(basins)
    plot_features(basins)
