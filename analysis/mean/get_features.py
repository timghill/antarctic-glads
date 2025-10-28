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
from scipy import interpolate

def _gldist(mesh, bed, surface, levelset):
    """Euclidean distance from mesh nodes to grounding line
    """
    # ISSUE TODO: this assumes that the grounding line is on the boundary
    # bz = surface[mesh['vertexonboundary']==1]
    # rhow = 1028
    # rhoi = 910
    # h_buoyancy = (rhow-rhoi)/rhoi * surface

    # gl = np.where(levelset<=0)[0]

    xFloating = mesh['x'][levelset<=0,None]
    yFloating = mesh['y'][levelset<=0,None]

    xGrounded = mesh['x'][levelset>=1]
    yGrounded = mesh['y'][levelset>=1]

    # print('Floating:', xFloating.shape)
    # print('Grounded:', xGrounded.shape)

    dx = xFloating - xGrounded
    dy = yFloating - yGrounded
    # print('dx:', dx.shape)
    
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

def _gradient(mesh, z):
    conn = mesh['elements']-1
    xel = np.mean(mesh['x'][conn], axis=1)
    yel = np.mean(mesh['y'][conn], axis=1)
    zel = np.mean(z[conn], axis=1)
    grad = np.zeros((mesh['numberofelements'], 2))
    # for i in range(mesh['numberofelements']):
    #     z_neighbour = z[conn][i]
    #     dx = mesh['x'][conn][i] - xel[i]
    #     dy = mesh['y'][conn][i] - yel[i]
    #     gx = np.mean((zel[i]-z_neighbour)/dx)
    #     gy = np.mean((zel[i]-z_neighbour)/dy)
    #     grad[i,0] = gx
    #     grad[i,1] = gy
    
    # dx = mesh['x'][conn] - xel[:,None]
    # dy = mesh['y'][conn] - yel[:,None]
    # dz = z[conn] - zel[:,None]
    # # dz[dx*dy < 10] = np.nan
    # dz[dx<10] = np.nan
    # dz[dy<10] = np.nan
    # grad[:,0] = np.nanmean((dz/dx), axis=1)
    # grad[:,1] = np.nanmean((dz/dy), axis=1)

    # Cross product to compute slope
    P0x = mesh['x'][conn][:, 0]
    P0y = mesh['y'][conn][:, 0]
    P0z = z[conn][:, 0]
    dx1 = mesh['x'][conn][:, 1] - P0x
    dy1 = mesh['y'][conn][:, 1] - P0y
    dz1 = z[conn][:, 1] - P0z
    dx2 = mesh['x'][conn][:, 2] - P0x
    dy2 = mesh['y'][conn][:, 2] - P0y
    dz2 = z[conn][:, 2] - P0z

    a = (dy1*dz2) - (dz1*dy2)
    b = (dz1*dx2) - (dx1*dz2)
    c = (dx1*dy2) - (dx2*dy1)

    grad[:, 0] = -a/c
    grad[:, 1] = -b/c


    return grad



def _slope(mesh, z):
    grad = _gradient(mesh, z)
    elslope = np.sqrt(np.sum(grad**2, axis=1))
    xel = np.mean(mesh['x'][mesh['elements']-1], axis=1)
    yel = np.mean(mesh['y'][mesh['elements']-1], axis=1)
    nodeslope = interpolate.griddata((xel, yel), elslope, (mesh['x'], mesh['y']))

    return nodeslope
    

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
    features['grounding_line_distance'] = _gldist(mesh, bed, surface, levelset)
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

    # # Flow accumulation
    # print('\tFlow accumulation subroutine...', end=' ', flush=True)
    # flowacc = _matrix_flow_accumulation(mesh, shreve_potential, basal_melt, levelset, step=1)
    # features['flow_accumulation'] = flowacc[levelset>0]
    # print('done')
    # # features['flow_accumulation'] = 0*shreve_potential[levelset>0]

    # print('\tBinned flow accumulation...', end=' ', flush=True)
    # binned_flow = _binned_flow_accumulation(mesh, shreve_potential, basal_melt)
    # features['binned_flow_accumulation'] = binned_flow[levelset>0]
    # print('done')

    print('\tSlopes...', end=' ', flush=True)
    surf_slope = _slope(mesh, surface)
    bed_slope = _slope(mesh, bed)
    shreve_slope = _slope(mesh, shreve_potential)
    print('done')
    features['surface_slope'] = surf_slope[levelset>0]
    features['bed_slope'] = bed_slope[levelset>0]
    features['potential_slope'] = shreve_slope[levelset>0]
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

        fig,axs = plt.subplots(nrows=2, ncols=5, figsize=(12, 8))
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
            label='Shreve potential (Pa)')

        surf_slope = np.nan*np.zeros(mesh['numberofvertices'])
        surf_slope[levelset>0] = features['surface_slope']
        m6 = axf[6].tripcolor(mtri, surf_slope,
            vmin=0, vmax=0.1, cmap=cmocean.cm.speed)
        fig.colorbar(m6, location='top', pad=0, shrink=0.8,
            label='Surface slope (-)')

        bed_slope = np.nan*np.zeros(mesh['numberofvertices'])
        bed_slope[levelset>0] = features['bed_slope']
        m7 = axf[7].tripcolor(mtri, bed_slope,
            vmin=0, vmax=0.2, cmap=cmocean.cm.speed)
        fig.colorbar(m7, location='top', pad=0, shrink=0.8,
            label='Bed slope (-)')

        shreve_slope = np.nan*np.zeros(mesh['numberofvertices'])
        shreve_slope[levelset>0] = features['potential_slope']
        m8 = axf[8].tripcolor(mtri, shreve_slope,
            vmin=0, vmax=500, cmap=cmocean.cm.speed)
        fig.colorbar(m8, location='top', pad=0, shrink=0.8,
            label='Shreve slope (Pa/m)')

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
    basins = [
        # 'G-H',
        # 'F-G',
        # 'Ep-F',
        # 'Cp-D',
        # 'C-Cp',
        # 'B-C',
        # 'Jpp-K',
        # 'J-Jpp',
        # 'G-H_2100'
        'G-H_2050',
    ]
    save_all_features(basins)
    plot_features(basins)

    # mesh = np.load('../../issm/G-H/data/geom/mesh.npy', allow_pickle=True)
    # surface = np.load('../../issm/G-H/data/geom/surface.npy')
    # bed = np.load('../../issm/G-H/data/geom/bed.npy')
    # thick = np.load('../../issm/G-H/data/geom/thick.npy')
    # levelset = np.load('../../issm/G-H/data/geom/ocean_levelset.npy')

    # surface[levelset<1] = np.nan
    # bed[levelset<1] = np.nan
    # thick[levelset<1] = np.nan

    # Shreve = 1000*9.81*bed + 910*9.81*thick

    # slopeSurf = _slope(mesh, surface)
    # slopeBed = _slope(mesh, bed)
    # slopeThick = _slope(mesh, thick)
    # slopeShreve = _slope(mesh, Shreve)

    # mtri = Triangulation(mesh['x'], mesh['y'], mesh['elements']-1)
    # fig,ax = plt.subplots()
    # pc = ax.tripcolor(mtri, slopeSurf, vmin=0, vmax=0.1)
    # ax.set_aspect('equal')
    # fig.colorbar(pc, label='Surface slope (-)')
    # fig.savefig('slopeSurf.png', dpi=400)

    # mtri = Triangulation(mesh['x'], mesh['y'], mesh['elements']-1)
    # fig,ax = plt.subplots()
    # pc = ax.tripcolor(mtri, slopeThick, vmin=0, vmax=0.1)
    # ax.set_aspect('equal')
    # fig.colorbar(pc, label='Thickness slope (-)')
    # fig.savefig('slopeThick.png', dpi=400)

    # fig,ax = plt.subplots()
    # pc = ax.tripcolor(mtri, slopeBed, vmin=0, vmax=0.25)
    # ax.set_aspect('equal')
    # fig.colorbar(pc, label='Bed slope (-)')
    # fig.savefig('slopeBed.png', dpi=400)

    # fig,ax = plt.subplots()
    # pc = ax.tripcolor(mtri, slopeShreve, vmin=0, vmax=1000)
    # ax.set_aspect('equal')
    # fig.colorbar(pc, label='Shreve potential slope (Pa/m)')
    # fig.savefig('slopeShreve.png', dpi=400)
