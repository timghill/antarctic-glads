import numpy as np
from scipy.interpolate import griddata
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.tri import Triangulation
import cmocean
from matplotlib.gridspec import GridSpec
import xarray as xr
import zarr as zr

index = 301
# Import bedmachine
stride = 4
print('Reading BedMachine')
with xr.open_dataset('../../data/bedmachine/BedMachineAntarctica-v3.nc') as bm:
    bmx = bm['x'][::stride].to_numpy()
    bmy = bm['y'][::stride].to_numpy()[::-1]
    bed = np.flipud(bm['bed'][::stride, ::stride].to_numpy())
    mask = np.flipud(bm['mask'][::stride, ::stride].to_numpy())
    thick = np.flipud(bm['thickness'][::stride, ::stride].to_numpy())

print('bmy:', bmy)

print('Reading MALI thickness change')
# Import thinning dataset
root = zr.open('../../data/Hillebrand_geometry/expAE03_04_q05m50_state.zarr')
dH = root['thickness'][index] - root['thickness'][0]

initroot = zr.open('../../data/Hillebrand_geometry/AIS_4to20km_r01_20220907_relaxed_q5.zarr')
malix = initroot['xCell']
maliy = initroot['yCell']
malixy = (malix, maliy)


xx,yy = np.meshgrid(bmx, bmy)
print(xx.shape)
bed[mask==0] = np.nan
xmin = np.min(xx[~np.isnan(bed)])
xmax = np.max(xx[~np.isnan(bed)])
ymin = np.min(yy[~np.isnan(bed)])
ymax = np.max(xx[~np.isnan(bed)])


# Interpolate onto bedmachine grid for plotting purpose
print('Gridding MALI thickness change')
dH_grid = griddata(malixy, dH, (xx, yy))
print(dH_grid.shape)
dH_grid[mask==0] = np.nan

print('Starting figure')
fig = plt.figure(figsize=(8, 6))
gs = GridSpec(5, 4,
    height_ratios=(100, 100, 2, 100, 100),
    width_ratios=(5, 100, 100, 5),
    bottom=0.085, top=0.95, left=0.085, right=0.9,
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
F_present = np.load('AISpred.npy')
N_present = 917*9.81*thick*(1 - F_present)
print(N_present.shape)
Npc = ax1.pcolormesh(bmx, bmy, N_present/1e6, vmin=0, vmax=4, cmap=cmocean.cm.haline)

Ncbar = fig.colorbar(Npc, cax=cax1)
Ncbar.set_label('N (MPa)', fontsize=8)
cax1.yaxis.tick_left()
cax1.yaxis.set_label_position('left')
cax1.tick_params(labelsize=8)

# 2. Imposed thinning
dH_grid[np.isnan(N_present)] = np.nan
Hpc = ax2.pcolormesh(bmy, bmy, dH_grid, vmin=-500, vmax=500, cmap=cmocean.cm.balance_r)
Hcbar = fig.colorbar(Hpc, cax=cax2, extend='both')
Hcbar.set_label(r'$\Delta$H (m)', fontsize=8)
cax2.tick_params(labelsize=8)

# 3. Future N
N_future = np.maximum(0, N_present + 917*9.81*dH_grid*(1-F_present))
Npc = ax3.pcolormesh(bmx, bmy, N_future/1e6, vmin=0, vmax=4, cmap=cmocean.cm.haline)

# 4. Change in N
N_delta = N_future - N_present
dNpc = ax4.pcolormesh(bmx, bmy, N_delta/1e6, vmin=-1, vmax=1, cmap=cmocean.cm.diff)

deltacbar = fig.colorbar(dNpc, cax=cax3, extend='both')
deltacbar.set_label(r'$\Delta N$ (MPa)')
cax3.tick_params(labelsize=8)

# Scalebar
R = Rectangle((xmin, ymin), 500e3, 75e3, facecolor='black')
ax1.add_patch(R)
ax1.text(xmin + 250e3, ymin + 75e3, '500 km', 
    ha='center', va='bottom', fontsize=8)

axs = np.array([[ax1, ax2], [ax3, ax4]])
for ax in axs.flat:
    ax.set_aspect('equal')
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    ax.tick_params(labelsize=8)
    ax.spines[['left', 'right', 'bottom', 'top']].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

fig.savefig('figures/map_delta.png', dpi=400)
