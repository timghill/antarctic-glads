import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.tri import Triangulation
from matplotlib.gridspec import GridSpec
import netCDF4 as nc
import cmocean


profile_basins = ['G-H', 'G-H', 'C-Cp', 'B-C', 'Jpp-K', 'Cp-D']
profile_numbers = [0, 1, 0, 0, 0, 0]
profile_labels = ['Thwaites', 'PIG', 'Denman', 'Lambert', 'Recovery', 'Totten']
alphabet = ['a', 'b', 'c', 'd']

bedmachine = '../../data/bedmachine/BedMachineAntarctica-v3.nc'

basins = [
    'G-H',
    # 'Ep-F',
    # 'F-G',
    'Cp-D',
    'C-Cp',
    'B-C',
    'Jpp-K',
    # 'J-Jpp',
]

Ncmap = cmocean.cm.matter
Zcmap = cmocean.cm.gray
Zalpha = 0.5

dx = 16
with nc.Dataset(bedmachine, 'r') as bm:
    mask = bm['mask'][::dx, ::dx].astype(int)
    x = bm['x'][::dx].astype(np.float32)
    y = bm['y'][::dx].astype(np.float32)
    bed = bm['bed'][::dx, ::dx].astype(np.float32)
    surf = bm['surface'][::dx, ::dx].astype(np.float32)

bed[mask==0] = np.nan
mask[surf>2000] = 2
xx,yy = np.meshgrid(x,y)

xmin = np.min(xx[~np.isnan(bed)])
xmax = np.max(xx[~np.isnan(bed)])
ymin = np.min(yy[~np.isnan(bed)])
ymax = np.max(yy[~np.isnan(bed)])


fig = plt.figure(figsize=(12, 12))
nrows = 2
ncols = 2
gs = GridSpec(ncols=3*ncols, nrows=2*nrows,
    hspace=0, wspace=0, left=0.05, bottom=0.0, right=0.95, top=0.95,
    height_ratios=(3, 100, 3, 100),
    width_ratios=(10, 80, 10, 10, 80, 10),
)

axs = np.array([[fig.add_subplot(gs[2*i+1,3*j:3*(j+1)], facecolor='none') for j in range(ncols)] for i in range(nrows)])
# print(axs.shape)

caxs = np.array([[fig.add_subplot(gs[2*i,3*j + 1]) for j in range(ncols)] for i in range(nrows)])

def _nanpad(levelset, fname):
    z = np.nan*np.zeros(mesh['numberofvertices'])
    z[levelset>0] = np.load(fname)
    return z

for ax in axs.flat:
    ax.contour(xx, yy, mask, levels=(0.5,2.5,), colors=('k','k'), linewidths=0.5)
    pc = ax.pcolormesh(xx, yy, bed, cmap=Zcmap, 
        vmin=-2000, vmax=2000, alpha=Zalpha)
    ax.set_aspect('equal')

    ax.spines[['left', 'right', 'top', 'bottom']].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

for i,basin in enumerate(basins):
    mesh = np.load(f'../../issm/{basin}/data/geom/mesh.npy', allow_pickle=True)
    levelset = np.load(f'../../issm/{basin}/data/geom/ocean_levelset.npy')
    mtri = Triangulation(mesh['x'], mesh['y'], mesh['elements']-1)

    fglads = np.load(f'../../issm/{basin}/glads/ff.npy').mean(axis=1)
    fglads[levelset<0] = np.nan
    fcv = _nanpad(levelset, f'data/pred_{basin}.npy')
    ferr = fcv - fglads

    Nglads = _nanpad(levelset, f'data/pred_{basin}_N_glads.npy')
    Ncv = _nanpad(levelset, f'data/CV_{basin}_N_glads.npy')
    Nerr = Ncv - Nglads

    pc0 = axs[0,0].tripcolor(mtri, fglads, vmin=0.5, vmax=1, cmap=cmocean.cm.dense)
    pc1 = axs[1,0].tripcolor(mtri, Nglads/1e6, vmin=0, vmax=5, cmap=cmocean.cm.haline)

    pc2 = axs[0,1].tripcolor(mtri, ferr, vmin=-0.1, vmax=0.1, cmap=cmocean.cm.balance)
    pc3 = axs[1,1].tripcolor(mtri, Nerr/1e6, vmin=-1, vmax=1, cmap=cmocean.cm.balance)

    outline = np.load(f'../../data/ANT_Basins/basin_{basin}.npy')
    for ax in axs.flat:
        ax.plot(outline[:,0], outline[:,1], color='k', linestyle='solid', linewidth=1)

for i,ax in enumerate(axs.flat):
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])

    ax.text(0, 1, alphabet[i], transform=ax.transAxes,
        fontweight='bold')

cb0 = fig.colorbar(pc0, cax=caxs[0,0], label='Flotation fraction (-)', orientation='horizontal')
cb1 = fig.colorbar(pc1, cax=caxs[1,0], label='Effective pressure (MPa)', orientation='horizontal')
cb2 = fig.colorbar(pc2, cax=caxs[0,1], label=r'$\Delta$ Flotation fraction (-)', orientation='horizontal')
cb3 = fig.colorbar(pc3, cax=caxs[1,1], label=r'$\Delta$ Effective pressure (MPa)', orientation='horizontal')

for cax in caxs.flat:
    cax.xaxis.set_label_position('top')

# Manual intervention: annotations

# profile_labels = ['Thwaites', 'PIG', 'Denman', 'Lambert', 'Recovery', 'Totten']

dxytext = np.array([
    [-500e3, -400e3],
    [-750e3, 0],
    [200e3, -100e3],
    [500e3, 500e3],
    [-0.75e6, 500e3],
    [0, -500e3],
])

ha = [
    'right',
    'right',
    'left',
    'left',
    'right',
    'left',
]

va = [
    'top',
    'center',
    'bottom',
    'center',
    'bottom',
    'top',
]

for p in range(len(profile_labels)):
    basin = profile_basins[p]
    num = profile_numbers[p]
    flowline = np.load(f'../../issm/{basin}/data/geom/flowline_{num:02d}.npy')
    ss,xx,yy = flowline

    axs[1,0].plot(xx, yy, color='w', linewidth=0.5)
    tx = xx[0] + dxytext[p,0]
    ty = yy[0] + dxytext[p,1]
    
    axs[1,0].text(tx, ty, profile_labels[p],
        ha=ha[p], va=va[p])
    axs[1,0].plot((xx[0], tx), (yy[0], ty), color='k')



fig.savefig('figures/continent_pred_error.png', dpi=400)

