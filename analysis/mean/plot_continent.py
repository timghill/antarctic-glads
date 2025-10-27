import numpy as np

from matplotlib import pyplot as plt
from matplotlib.tri import Triangulation
from matplotlib import patches
import cmocean
import xarray as xr

inset_locs = [
    [-0.3, -0.6, 0.65, 0.65],            # Amundsen sea
    # [-0.10, -0.475, 0.6, 0.5],            # Siple coast
    # [0.45, -0.45, 0.6, 0.5],     # Getz
    [0.2, -0.6, 0.65, 0.65],            # Aurora subglacial basin
    [0.7, -0.6, 0.65, 0.65],            # Aurora subglacial basin
    # [0.7, (1-0.65)/2, 0.65, 0.65],           # Denman gl
    [0.6, 0.9, 0.65, 0.65],             # Amery
    [-0.25, 0.9, 0.65, 0.65],              # Recovery ice stream
    # [-0.45, 1.45-0.7, 0.65, 0.65],             # Foundation/Academy
]

scale_locs = [
    [0.05, 0.05],
    [0.7, 0.8],
    [0.15, 0.05],
    [0.8, 0.05],
    [0.05, 0.05],
]
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

Y = np.flipud(np.load('AISpred.npy'))

rhoice = 917
rhowater = 1023
g = 9.81

stride = 4
bedmachine = '../../data/bedmachine/BedMachineAntarctica-v3.nc'
with xr.open_dataset(bedmachine) as bm:
    x = bm['x'][::stride].values
    y = bm['y'][::stride].values
    bed = bm['bed'][::stride, ::stride].values
    thickness = bm['thickness'][::stride, ::stride].values
xx, yy = np.meshgrid(x, y)

xmin = np.min(xx[~np.isnan(Y)])
xmax = np.max(xx[~np.isnan(Y)])
ymin = np.min(yy[~np.isnan(Y)])
ymax = np.max(yy[~np.isnan(Y)])

x0 = xmin + 0.1*(xmax-xmin)
y0 = ymin + 0.2*(ymax-ymin)

# fig,ax = plt.subplots(figsize=(10, 13))
# pc = ax.pcolormesh(xx, yy, Y, vmin=0.5, vmax=1, cmap=cmocean.cm.dense)
# ax.set_aspect('equal')
# ax.set_xticks([])
# ax.set_yticks([])
# ax.spines[['left', 'right', 'top', 'bottom']].set_visible(False)
# ax.set_xlim([xmin, xmax])
# ax.set_ylim([ymin, ymax])

# # INSETS

# for i in range(5):
#     axinset = ax.inset_axes(inset_locs[i], facecolor='none')
#     basin = basins[i]
#     outline = np.load(f'../../data/ANT_Basins/basin_{basin}.npy')
#     xmin = np.min(outline[:,0])
#     xmax = np.max(outline[:,0])
#     ymin = np.min(outline[:,1])
#     ymax = np.max(outline[:,1])
#     axinset.set_xlim([xmin, xmax])
#     axinset.set_ylim([ymin, ymax])
#     axinset.set_aspect('equal')
#     axinset.set_xticks([])
#     axinset.set_yticks([])

#     Yi = np.load(f'pred_{basin}.npy')
#     mesh = np.load(f'../../issm/{basin}/data/geom/mesh.npy', allow_pickle=True)
#     mtri = Triangulation(mesh['x'], mesh['y'], mesh['elements']-1)
#     levelset = np.load(f'../../issm/{basin}/data/geom/ocean_levelset.npy')
#     Yifull = np.nan*np.zeros(mesh['numberofvertices'])
#     Yifull[levelset>0] = Yi
#     axinset.tripcolor(mtri, Yifull, vmin=0.5, vmax=1, cmap=cmocean.cm.dense)
#     axinset.spines[['left', 'right', 'top', 'bottom']].set_visible(False)

#     ax.plot(outline[:, 0], outline[:, 1], color='w', linestyle='dotted', linewidth=3)


# cax = ax.inset_axes([1.1, 0.25, 0.05, 0.5])
# fig.colorbar(pc, cax=cax, orientation='vertical', label='Fraction of overburden')

# # cax = ax.inset_axes()
# pad = 0.1625
# fig.subplots_adjust(left=pad, right=1-pad-0.05, top=1-pad, bottom=pad)

# fig.savefig('figures/AIS_poster_flotfrac.png', dpi=300)

###############################
# Effective pressure
###############################

p_ice = rhoice*g*thickness
phi_bed = rhowater*g*bed
p_water = Y*p_ice
N = p_ice - p_water

fig,ax = plt.subplots(figsize=(10, 13))
pc = ax.pcolormesh(xx, yy, N/1e6, vmin=0, vmax=4, cmap=cmocean.cm.haline)
ax.set_aspect('equal')
ax.set_xticks([])
ax.set_yticks([])
ax.spines[['left', 'right', 'top', 'bottom']].set_visible(False)
ax.set_xlim([xmin, xmax])
ax.set_ylim([ymin, ymax])

# INSETS

for i in range(5):
    axinset = ax.inset_axes(inset_locs[i], facecolor='none')
    basin = basins[i]
    outline = np.load(f'../../data/ANT_Basins/basin_{basin}.npy')
    xmin = np.min(outline[:,0])
    xmax = np.max(outline[:,0])
    ymin = np.min(outline[:,1])
    ymax = np.max(outline[:,1])
    axinset.set_xlim([xmin, xmax])
    axinset.set_ylim([ymin, ymax])
    axinset.set_aspect('equal')
    axinset.set_xticks([])
    axinset.set_yticks([])

    Yi = np.load(f'data/pred_{basin}.npy')
    mesh = np.load(f'../../issm/{basin}/data/geom/mesh.npy', allow_pickle=True)
    mtri = Triangulation(mesh['x'], mesh['y'], mesh['elements']-1)
    levelset = np.load(f'../../issm/{basin}/data/geom/ocean_levelset.npy')
    bed = np.load(f'../../issm/{basin}/data/geom/bed.npy')
    thickness = np.load(f'../../issm/{basin}/data/geom/thick.npy')
    Yifull = np.nan*np.zeros(mesh['numberofvertices'])
    Yifull[levelset>0] = Yi
    Ni = rhoice*g*thickness*(1 - Yifull)
    axinset.tripcolor(mtri, Ni/1e6, vmin=0, vmax=4, cmap=cmocean.cm.haline)
    axinset.spines[['left', 'right', 'top', 'bottom']].set_visible(False)

    ax.plot(outline[:, 0], outline[:, 1], color='w', linestyle='dotted', linewidth=3)
    
    xr = xmin + scale_locs[i][0]*(xmax-xmin)
    yr = ymin + scale_locs[i][1]*(ymax-ymin)
    width = 200e3
    height = 5e3
    scale = patches.Rectangle((xr, yr), width, height, color='black')
    axinset.text(xr + 0.5*width, yr + height + 2.5, '200 km',
        ha='center', va='bottom', color='black', fontsize=10)
    axinset.add_patch(scale)

scale = patches.Rectangle((x0, y0), 500e3, 25e3, color='black')
ax.add_patch(scale)
ax.text(x0 + 250e3, y0 + 40e3, '500 km', fontsize=12, color='black', 
    ha='center', va='bottom')

cax = ax.inset_axes([1.1, 0.25, 0.05, 0.5])
cbar = fig.colorbar(pc, cax=cax, orientation='vertical', label='Effective pressure (MPa)')
cax.tick_params(labelsize=12)
cbar.set_label('Effective pressure (MPa)', fontsize=12)

# cax = ax.inset_axes()
pad = 0.1625
fig.subplots_adjust(left=pad, right=1-pad-0.05, top=1-pad, bottom=pad)
fig.savefig('figures/AIS_poster_N.png', dpi=400)
