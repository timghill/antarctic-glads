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
import zarr as zr
from scipy.interpolate import griddata

stride = 4

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
        if i%100==0:
            print('batch', i)
        dxi = xFloating[:,None] - xGrounded[i::batches]
        dyi = yFloating[:,None] - yGrounded[i::batches]
        dd = np.sqrt(dxi**2 + dyi**2)
        # print('dd:', dd.shape)
        ddmin[i::batches] = np.nanmin(dd, axis=0)

    ddmap = np.nan*np.zeros(xx.shape, dtype=np.float32)
    ddmap[mask==2] = ddmin
    return ddmap

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

def _slope(z):
    dx = stride*500
    dy = stride*500
    dzdx = np.nan*np.zeros(z.shape, dtype=z.dtype)
    dzdy = np.nan*np.zeros(z.shape, dtype=z.dtype)
    dzdx[1:-1, 1:-1] = (z[1:-1, 2:] - z[1:-1, :-2])/dx
    dzdy[1:-1, 1:-1] = (z[2:, 1:-1] - z[:-2, 1:-1])/dy
    dz = np.sqrt(dzdx**2 + dzdy**2).astype(np.float32)
    return dz
    

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

    print('dx:', x[1] - x[0])

    # Store all features in dictionary features
    features = {}

    # Surface, bed and thickness
    print('\tSurface, bed and thickness...', end=' ', flush=True)
    bed = bm['bed'][::stride, ::stride].values.astype(np.float32)
    surface = bm['surface'][::stride, ::stride].values.astype(np.float32)
    thick = bm['thickness'][::stride, ::stride].values.astype(np.float32)
    print('done')

    levelset = np.zeros(mask.shape, dtype=int)
    levelset[mask==2] = 1

    bed[levelset<1] = np.nan
    surface[levelset<1] = np.nan
    thick[levelset<1] = np.nan
    
    features['bed'] = bed
    features['surface'] = surface
    features['thickness'] = thick

    features['bed_slope'] = _slope(bed)
    features['surface_slope'] = _slope(surface)
    
    # Grounding line distance
    print('\tGrounding line distance...', end=' ', flush=True)
    features['grounding_line_distance'] = _gldist(xx, yy, mask, bed, surface).astype(np.float32)
    print('done')

    # Local basal melt rate
    print('\tLocal basal melt rate...', end=' ', flush=True)
    # basal_melt = np.load(
    #     os.path.join(basin_dir, 'data/lanl-mali/basal_melt_mali.npy')
    # )
    features['basal_melt'] = np.log10(_basal_melt(xx, yy)).astype(np.float32)
    print('basal_melt:', features['basal_melt'].shape)
    features['basal_melt'][levelset==0] = np.nan
    print('done')

    # Hydraulic potential
    rho_ice = 917
    rho_freshwater = 1000
    g = 9.81
    shreve_potential = rho_freshwater*g*bed + rho_ice*g*thick
    features['potential'] = shreve_potential.astype(np.float32)
    features['potential_slope'] = _slope(features['potential'])

    return features


def get_future_features(bedmachine, year):
    bm = xr.open_dataset(bedmachine)
    x = bm['x'][::stride].values
    y = bm['y'][::stride].values
    print('dx:', x[1] - x[0])
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

    # Thinning
    print('\tApplying thinning...')

    print('Reading MALI thickness change')
    # Import thinning dataset
    root = zr.open('../../data/Hillebrand_geometry/expAE03_04_q05m50_state.zarr')
    index = year-2000
    dH = root['thickness'][index] - root['thickness'][0]

    initroot = zr.open('../../data/Hillebrand_geometry/AIS_4to20km_r01_20220907_relaxed_q5.zarr')
    malix = initroot['xCell']
    maliy = initroot['yCell']
    malixy = (malix, maliy)

    # Interpolate onto bedmachine grid for plotting purpose
    print('Gridding MALI thickness change')
    dH_grid = griddata(malixy, dH, (xx, yy))
    print(dH_grid.shape)
    dH_grid[mask==0] = np.nan

    # Applying the thinning
    _thick = thick + dH_grid
    levelset[_thick<0] = -1
    thick = np.maximum(_thick, 0)

    bed[levelset<1] = np.nan
    surface[levelset<1] = np.nan
    thick[levelset<1] = np.nan

    # NOTE this doesn't do floating ice properly -- but we don't have to here
    features['bed'] = bed.astype(np.float32)
    features['surface'] = surface.astype(np.float32)
    features['thickness'] = thick.astype(np.float32)
    features['bed_slope'] = _slope(bed)
    features['surface_slope'] = _slope(surface)
    
    # Grounding line distance
    print('\tGrounding line distance...', end=' ', flush=True)
    features['grounding_line_distance'] = _gldist(xx, yy, mask, bed, surface).astype(np.float32)
    print('done')

    # Local basal melt rate
    print('\tLocal basal melt rate...', end=' ', flush=True)
    # basal_melt = np.load(
    #     os.path.join(basin_dir, 'data/lanl-mali/basal_melt_mali.npy')
    # )
    features['basal_melt'] = np.log10(_basal_melt(xx, yy)).astype(np.float32)
    print('basal_melt:', features['basal_melt'].shape)
    features['basal_melt'][levelset==0] = np.nan
    print('done')

    # Hydraulic potential
    rho_ice = 917
    rho_freshwater = 1000
    g = 9.81
    shreve_potential = rho_freshwater*g*bed + rho_ice*g*thick
    features['potential'] = shreve_potential.astype(np.float32)
    features['potential_slope'] = _slope(features['potential'])

    return features

def save_all_features(bedmachine, year):
    basin = 'AIS'
    features = get_features(bedmachine)
    basinfile = f'features_{basin}.pkl'
    with open(basinfile, 'wb') as basin_pkl:
        pickle.dump(features, basin_pkl)

    future_features = get_future_features(bedmachine, year)
    basinfile = f'features_{basin}_{year}.pkl'
    with open(basinfile, 'wb') as basin_pkl:
        pickle.dump(future_features, basin_pkl)
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
    save_all_features(bedmachine, 2300)
    # plot_features()
