import numpy as np
from scipy.interpolate import griddata
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.tri import Triangulation
from matplotlib import colors
import cmocean
from matplotlib.gridspec import GridSpec
import xarray as xr
import zarr as zr
import scipy.signal

fs = 8
index = 301

basins = [
    'B-C',
    'G-H',
    'Cp-D',
    'C-Cp',
    'Jpp-K',
    'J-Jpp',
    'Ep-F',
]

# Import bedmachine
stride = 4
print('Reading BedMachine')
with xr.open_dataset('../../data/bedmachine/BedMachineAntarctica-v3.nc') as bm:
    bmx = bm['x'][::stride].to_numpy()
    bmy = bm['y'][::stride].to_numpy()[::-1]
    bed = np.flipud(bm['bed'][::stride, ::stride].to_numpy())
    mask = np.flipud(bm['mask'][::stride, ::stride].to_numpy())
    # thick = np.flipud(bm['thickness'][::stride, ::stride].to_numpy())

# print('bmy:', bmy)

# print('Reading MALI thickness change')
# # Import thinning dataset
# root = zr.open('../../data/Hillebrand_geometry/expAE03_04_q05m50_state.zarr')
# dH = root['thickness'][index] - root['thickness'][0]

# initroot = zr.open('../../data/Hillebrand_geometry/AIS_4to20km_r01_20220907_relaxed_q5.zarr')
# malix = initroot['xCell']
# maliy = initroot['yCell']
# malixy = (malix, maliy)


xx,yy = np.meshgrid(bmx, bmy)
print(xx.shape)
bed[mask==0] = np.nan
xmin = np.min(xx[~np.isnan(bed)])
xmax = np.max(xx[~np.isnan(bed)])
ymin = np.min(yy[~np.isnan(bed)])
ymax = np.max(xx[~np.isnan(bed)])


# Interpolate onto bedmachine grid for plotting purpose
# print('Gridding MALI thickness change')
# dH_grid = griddata(malixy, dH, (xx, yy))
# print(dH_grid.shape)
# dH_grid[mask==0] = np.nan

print('Starting figure')
fig = plt.figure(figsize=(7, 6*7/8))
gs = GridSpec(5, 4,
    height_ratios=(100, 100, 2, 100, 100),
    width_ratios=(5, 100, 100, 5),
    bottom=0.05, top=0.95, left=0.085, right=0.9,
    hspace=0.05, wspace=0.05,
)

ax1 = fig.add_subplot(gs[:2, 1])
ax2 = fig.add_subplot(gs[:2, 2])
ax3 = fig.add_subplot(gs[3:, 1])
ax4 = fig.add_subplot(gs[3:, 2])

cax1 = fig.add_subplot(gs[1:4, 0])
cax2 = fig.add_subplot(gs[:2, -1])
cax3 = fig.add_subplot(gs[3:, -1])

# 1. Present N
# F_present = np.load('data/AIS_f.npy')
# N_present = 917*9.81*thick*(1 - F_present)
# print(N_present.shape)
# Npc = ax1.pcolormesh(bmx, bmy, N_present/1e6, vmin=0, vmax=3, 
#     cmap=cmocean.cm.haline, rasterized=True)

# Ncbar = fig.colorbar(Npc, cax=cax1)
# Ncbar.set_label('$N$ (MPa)', fontsize=fs)
# cax1.yaxis.tick_left()
# cax1.yaxis.set_label_position('left')
# cax1.tick_params(labelsize=fs)

# # 2. Imposed thinning
# dH_grid[np.isnan(N_present)] = np.nan
# cnorm = colors.TwoSlopeNorm(vcenter=0, vmin=-2000, vmax=200)
# Hpc = ax2.pcolormesh(bmy, bmy, dH_grid, norm=cnorm, 
#     cmap=cmocean.cm.balance_r, rasterized=True)
# Hcbar = fig.colorbar(Hpc, cax=cax2, extend='both')
# Hcbar.set_label(r'$\Delta$H (m)', fontsize=fs)
# cax2.tick_params(labelsize=fs)
# Hcbar.set_ticks([-2000, -1500, -1000, -500, 0, 50, 100, 150, 200])

# # 3. Future N
# F_future = np.load('data/AIS_2300_f.npy')
# thick_future = dH_grid + thick
# N_future = 917*9.81*thick_future*(1 - F_future)
# N_future[np.logical_and(np.isnan(N_future), ~np.isnan(N_present))] = 0
# Npc = ax3.pcolormesh(bmx, bmy, N_future/1e6, vmin=0, vmax=3, 
#     cmap=cmocean.cm.haline, rasterized=True)

# # 4. Change in N
# N_delta = N_future - N_present
# window = 5
# kern = 1./window/window * np.ones((window,window))
# N_delta = scipy.signal.convolve2d(N_delta, kern, mode='same')
# cnorm = colors.TwoSlopeNorm(vcenter=0, vmin=-1, vmax=0.2)

# ix = np.concatenate((np.linspace(0, 0.5, 128), np.linspace(0.5, 0.9, 128)))
# cc = cmocean.cm.diff(ix)
# cmap = colors.ListedColormap(cc)

# dNpc = ax4.pcolormesh(bmx, bmy, N_delta/1e6, norm=cnorm, 
#     cmap=cmap, rasterized=True)

# deltacbar = fig.colorbar(dNpc, cax=cax3, extend='both')
# deltacbar.set_label(r'$\Delta N$ (MPa)', fontsize=fs)
# cax3.tick_params(labelsize=fs)
# deltacbar.set_ticks([-1, -0.75, -0.5, -0.25, 0, 0.05, 0.1, 0.15, 0.2])

# Scalebar
R = Rectangle((xmin, ymin), 1000e3, 75e3, facecolor='black')
ax1.add_patch(R)
ax1.text(xmin + 500e3, ymin + 75e3, '1000 km', 
    ha='center', va='bottom', fontsize=fs)

axs = np.array([[ax1, ax2], [ax3, ax4]])
alphabet = ['(a)', '(b)', '(c)', '(d)']
for i,ax in enumerate(axs.flat):
    ax.set_aspect('equal')
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    ax.tick_params(labelsize=fs)
    ax.spines[['left', 'right', 'bottom', 'top']].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(0.05, 0.9, alphabet[i], transform=ax.transAxes,
        fontweight='bold', fontsize=fs)

for basin in basins:
    outline = np.load(f'../../data/ANT_Basins/basin_{basin}.npy')
    for ax in axs.flat:
        ax.plot(outline[:,0], outline[:,1], color='k', linestyle='solid', linewidth=0.5)


dxytext = np.array([
    [650e3, -100e3],
    [-1000e3, -300e3],
    [400e3, 0],
    [250e3, 0],
    [-650e3, 1400e3],
    [-400e3, 800e3],
    [0, -900e3],
])

for p in range(len(basins)):
    basin = basins[p]
    mesh = np.load(f'../../issm/{basin}/data/geom/mesh.npy', allow_pickle=True)
    bx = np.mean(mesh['x'])
    by = np.mean(mesh['y'])
    tx = bx + dxytext[p][0]
    ty = by + dxytext[p][1]
    
    axs[0,0].text(tx, ty, basin, fontsize=fs-1)

ax = axs[1,1]
ax.text(2.25e6, -2.0e6, 'Totten', ha='right', va='top', fontsize=fs-1)
ax.plot([2.25e6, 2265369.0], [-2.0e6, -1003529.0], color='k', linewidth=0.5)
ax.text(2.25e6, 2.25e6, 'West Ice\nShelf', ha='right', va='bottom', fontsize=fs-1)
ax.plot([2.25e6, 2.62e6], [2.25e6, 0.26e6], color='k', linewidth=0.5)
ax.text(-2.e6, -1.75, 'WAIS', ha='right', va='center', fontsize=fs-1)
ax.text(-1.4e6, 1.0e6, 'Filchner-\nRonne', va='bottom', ha='center', fontsize=fs-1)
ax.text(0.85e6, -2.36e6, 'Cook', va='top', ha='center', fontsize=fs-1)

fig.savefig('figures/map_delta.png', dpi=400)
# fig.savefig('../../manuscript/f05.png', dpi=300)
# fig.savefig('../../manuscript/f05.pdf', dpi=300)

# plt.show()
