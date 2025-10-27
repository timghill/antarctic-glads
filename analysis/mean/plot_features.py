import numpy as np
from matplotlib import pyplot as plt
from matplotlib.tri import Triangulation
import cmocean

basins = ['B-C']
features = [
    'bed',
    'surface',
    'thickness',
    'grounding_line_distance',
    'basal_melt',
    'potential',
    # 'binned_flow_accumulation',
]

labels = [
    'Bed elevation (m)',
    'Surface elevation (m)',
    'Ice thickness (m)',
    'GL distance (km)',
    'log$_{10}$ Melt rate (m a$^{-1}$)',
    'Potential (MPa)',
]

lims = [
    [-2000, 2000],
    [0, 4000],
    [0, 4000],
    [0, 1000],
    [-3, 0],
    [0, 40],
]

cmaps = [
    cmocean.cm.topo,
    cmocean.cm.haline,
    cmocean.cm.amp,
    cmocean.cm.deep,
    cmocean.cm.rain,
    cmocean.cm.matter,
]

for basin in basins:
    feats = np.load(f'features_{basin}.pkl', allow_pickle=True)
    mesh = np.load(f'../../issm/{basin}/data/geom/mesh.npy', allow_pickle=True)
    mtri = Triangulation(mesh['x']/1e3, mesh['y']/1e3, mesh['elements']-1)
    levelset = np.load(f'../../issm/{basin}/data/geom/ocean_levelset.npy')

    feats['grounding_line_distance'] = feats['grounding_line_distance']/1e3

    feats['potential'] = feats['potential']/1e6

    fig,axs = plt.subplots(figsize=(10, 8), nrows=2, ncols=3, sharex=True, sharey=True)

    for j,key in enumerate(features):
        yj = np.nan*np.zeros(mesh['numberofvertices'])
        yj[levelset>0] = feats[key]
        # yj[levelset==0] = np.nan
        pc = axs.flat[j].tripcolor(mtri, yj, 
            vmin=lims[j][0], vmax=lims[j][1],
            cmap=cmaps[j])
        cbar = fig.colorbar(pc, ax=axs.flat[j], location='top')
        cbar.set_label(labels[j], size=20, labelpad=16)
        cbar.ax.tick_params(labelsize=20)

        axs.flat[j].set_xticks([])
        axs.flat[j].set_yticks([])
        axs.flat[j].spines[['left', 'right', 'top', 'bottom']].set_visible(False)
        axs.flat[j].set_aspect('equal')

fig.tight_layout()
fig.savefig('figures/features_example.png', dpi=400, transparent=True)
