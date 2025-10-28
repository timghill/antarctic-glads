import numpy as np
from matplotlib import pyplot as plt
from matplotlib.tri import Triangulation
import zarr as zr
from scipy.interpolate import griddata
import cmocean

rhoice = 917
rhowater = 1027
rhofresh = 1000

def main(basin, index):

    # Read MALI model outputs (Hillebrand et al., 2025)
    initstore = zr.storage.LocalStore('AIS_4to20km_r01_20220907_relaxed_q5.zarr')
    initroot = zr.open(initstore)
    x = initroot['xCell']
    y = initroot['yCell']
    xy = (x,y)

    store = zr.storage.LocalStore('expAE03_04_q05m50_state.zarr')
    root = zr.open(store)
    thickness = root['thickness'][:]
    dH = thickness[index] - thickness[0]

    # Interpolate onto mesh
    mesh = np.load(f'../../issm/{basin}/data/geom/mesh.npy', allow_pickle=True)
    meshxy = (mesh['x'], mesh['y'])

    # Apply thinning to existing geometry
    dH_mesh = griddata(xy, dH, meshxy, method='linear')

    ref_surface = np.load(f'../../issm/{basin}/data/geom/surface.npy')
    ref_thick = np.load(f'../../issm/{basin}/data/geom/thick.npy')
    ref_base = np.load(f'../../issm/{basin}/data/geom/base.npy')
    print('min thick:', np.min(ref_thick))

    # new_surface = ref_surface + dH_mesh
    dH_mesh[dH_mesh < -(ref_thick-10)] = -(ref_thick-10)[dH_mesh < -(ref_thick-10)]
    new_thick = ref_thick + dH_mesh
    new_surface = ref_surface + dH_mesh
    hf = (1 - rhoice/rhowater)*new_thick
    print('min thick:', np.min(new_thick))

    # isfloating = new_surface <= (hf + 10)

    buoyancy_adjustment = np.zeros(new_thick.shape)
    buoyancy_adjustment[new_surface<hf] = (hf - new_surface)[new_surface<hf]

    new_surface += buoyancy_adjustment
    # new_base = ref_base + buoyancy_adjustment

    # Ice/ocean masks
    ice_levelset = -1*np.ones(new_surface.shape)
    ice_levelset[new_thick<0.1] = 1

    ocean_levelset = 1*np.ones(new_surface.shape)
    ocean_levelset[new_surface<(hf+10)] = -1

    new_base = new_surface - new_thick

    np.save(f'base_{index:03d}.npy', new_base)
    np.save(f'surface_{index:03d}.npy', new_surface)
    np.save(f'thick_{index:03d}.npy', new_thick)
    np.save(f'ocean_levelset_{index:03d}.npy', ocean_levelset)
    np.save(f'ice_levelset_{index:03d}.npy', ice_levelset)

    mtri = Triangulation(mesh['x'], mesh['y'], mesh['elements']-1)
    fig,axs = plt.subplots(ncols=2, nrows=2, sharex=True, sharey=True)
    pc1 = axs[0,0].tripcolor(mtri, ref_thick, vmin=0, vmax=3000, cmap=cmocean.cm.amp)
    axs[0,1].tripcolor(mtri, new_thick, vmin=0, vmax=3000, cmap=cmocean.cm.amp)
    pc2 = axs[1,1].tripcolor(mtri, dH_mesh, vmin=-500, vmax=500, cmap=cmocean.cm.balance_r)
    axs[0,0].set_title('2000')
    axs[0,1].set_title(str(2000 + int(index-1)))
    axs[1,0].tripcolor(mtri, new_surface-hf, vmin=-3000, vmax=3000, cmap=cmocean.cm.delta)

    # Contours
    for ax in axs.flat:
        ax.tricontour(mtri, ocean_levelset, levels=(0,), colors=('k',))

    axs[1,0].tricontour(mtri, ice_levelset, levels=(0,), colors=('cyan',), linewidths=0.25)
    fig.colorbar(pc1, ax=axs[0], label='Thickness (m)')
    fig.colorbar(pc2, ax=axs[1], label='Thickness change (m)')
    for ax in axs.flat:
        ax.set_aspect('equal')
    # axs[1,0].set_visible(False)


    # axs[0,1].tricontour(mtri, new_surface-hf, levels=(0,), colors=('k',))
    fig.savefig(f'dthickness_{index:03d}.png', dpi=400)


if __name__=='__main__':
    main('G-H', 51)