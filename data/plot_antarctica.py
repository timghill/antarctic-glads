import numpy as np
import matplotlib
matplotlib.use('QtAgg')
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.tri import Triangulation
import netCDF4 as nc
import cmocean
from utils.plotchannels import plotchannels

bedmachine = 'bedmachine/BedMachineAntarctica-v3.nc'

# mask:
# 0 = ocean
# 1 = ice free land
# 2 = grounded ice
# 3 = floating ice
# 4 = vostok

rect_boxes = [
    [-1800e3, -800e3, 600e3, 800e3],    # Amundsen sea
    [-940e3 , -1.38e6, 800e3, 800e3],   # Siple coast
    [0, 0, 0, 0],                       # Getz
    [1.4e6, -1.5e6, 1e6, 1.4e6],        # Aurora subglacial basin
    [0, 0, 0, 0],                       # Denman gl
    [1.2e6, 2.0e5, 800e3, 800e3],       # Amery grounding line
    [-800e3, 500e3, 1200e3, 1000e3],    # Recovery ice stream
    [-900e3, -50e3, 400e3, 500e3],      # Foundation/Academy glaciers
]

# left = -0.45
# right = 1.45
# bottom = -0.45
# top = 1.45

inset_locs = [
    [-0.45, -0.3, 0.525, 0.625],            # Amundsen sea
    [-0.10, -0.475, 0.6, 0.5],            # Siple coast
    [0.45, -0.45, 0.6, 0.5],     # Getz
    [1.5-0.6, -0.4, 0.6, 0.7],            # Aurora subglacial basin
    [1.45-0.5, 0.5-0.25, 0.5, 0.6],           # Denman gl
    [1.45-0.6, 1.45-0.6, 0.6, 0.6],             # Amery
    [0.1, 1.45-0.6, 0.7, 0.6],              # Recovery ice stream
    [-0.45, 1.45-0.7, 0.65, 0.65],             # Foundation/Academy
]


gl_xy = np.array([
    [-1.5905e6, -0.2536e6],
    [-1.5205e6, -0.4637e6],
    [0, 0],
    [2266.8e3, -999e3],
    [2.5321e6, -0.4105e6],
    [1.6820e6, 0.7106e6],
    [0, 0],
    [0, 0],

])

gl_boxes = np.array([
    [-1.65e6, -3e5,  -1.5e6, -1.55e5],
    [-1.6e6, -5.25e5, -1.3e6, -3.75e5],
    [0, 0, 0, 0],
    [2.0e6, -1.05e6, 2.3e6, -0.9e6],
    [2.6e6, -4.75e5, 2.3e6, -3.25e5],
    [1.75e6, 4.5e5, 1.45e6, 7.5e5],
    [0, 0, 0, 0],    
    [0, 0, 0, 0],
])


# ])

# NN = [
#     None,#'../issm/thwaites/train/train_N.npy',
#     None,
#     None,#'../issm/aurora/glads/N.npy',
#     None,#'../issm/denman/glads/N.npy',
#     None,#'../issm/amery/train/train_N.npy',
#     None,
#     None,
# ]

# QQ = [
#     None,#'../issm/thwaites/train/train_Q.npy',
#     None,
#     None,#'../issm/aurora/glads/Q.npy',
#     None,#'../issm/denman/glads/Q.npy',
#     None,#'../issm/amery/train/train_Q.npy',
#     None,
#     None,
# ]

basins = [
    'G-H',
    'Ep-F',
    'F-G',
    'Cp-D',
    'C-Cp',
    'B-C',
    'Jpp-K',
    'J-Jpp',
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

fig,ax = plt.subplots(figsize=(8,8))
ax.contour(xx, yy, mask, levels=(0.5,2.5,), colors=('k','k'), linewidths=0.5)
# ax.contour(xx, yy, mask, levels=(0.1,), colors=('b',), linewidths=2)
# ax.pcolormesh(xx, yy, mask)
pc = ax.pcolormesh(xx, yy, bed, cmap=Zcmap, 
    vmin=-2000, vmax=2000, alpha=Zalpha)
ax.set_aspect('equal')

ax.spines[['left', 'right', 'top', 'bottom']].set_visible(False)
ax.set_xticks([])
ax.set_yticks([])

# cax.xaxis.tick_top()
# cax.xaxis.set_label_position('top')

nsectors = len(basins)
axs_inset = []
# for i,box in enumerate(rect_boxes):
for i in range(nsectors):
    basin = basins[i]

    ax_ins = ax.inset_axes(inset_locs[i], facecolor='none')
    axs_inset.append(ax_ins)
    
    ax_ins.contour(xx, yy, mask, levels=(0.5,2.5,), colors=('k','k'), linewidths=0.5)

    outline = np.load(f'ANT_Basins/basin_{basin}.npy')
    ax.plot(outline[:,0], outline[:,1], color='k', linestyle='solid', linewidth=1)
    ax_ins.plot(outline[:,0], outline[:,1], color='k', linestyle='solid', linewidth=1)

    # if NN[i]:
    #     N = np.load(NN[i], mmap_mode='r')/1e6
    #     N = np.nanmedian(N[:,-1,:], axis=-1)
    #     mesh = np.load(meshes[i],
    #         allow_pickle=True)
    #     mtri = Triangulation(mesh['x'], mesh['y'], mesh['elements']-1)
    #     Npc = ax_ins.tripcolor(mtri, N, vmin=0, vmax=5, cmap=Ncmap, rasterized=True)
    #     ax_ins.set_xlim(np.min(mesh['x']), np.max(mesh['x']))
    #     ax_ins.set_ylim(np.min(mesh['y']), np.max(mesh['y']))

    #     ax.tripcolor(mtri, N, vmin=0, vmax=5, cmap=Ncmap)

    # elif meshes[i]:
    mesh = np.load(f'../issm/{basin}/data/geom/mesh.npy', allow_pickle=True)
    mtri = Triangulation(mesh['x'], mesh['y'], mesh['elements']-1)
    ins_mappable = ax_ins.tripcolor(mtri, np.log10(mesh['area']),  edgecolor='none',
        linewidth=0.25, rasterized=True, antialiased=False,
        vmin=6, vmax=9, cmap=cmocean.cm.matter,
        )
    x1 = np.min(mesh['x'])
    x2 = np.max(mesh['x'])
    y1 = np.min(mesh['y'])
    y2 = np.max(mesh['y'])
    ax_ins.set_xlim((x1, x2))
    ax_ins.set_ylim((y1, y2))

    rect_boxes[i] = [x1, y1, x2-x1, y2-y1]

    ax.tripcolor(mtri, np.log10(mesh['area']),  edgecolor='none',
        linewidth=0.25, rasterized=True, antialiased=False,
        vmin=6, vmax=9, cmap=cmocean.cm.matter,
        )
    # else:
    #     ax_ins.pcolormesh(xx, yy, bed, cmap=Zcmap, vmin=-2000, vmax=2000,
    #         alpha=Zalpha, rasterized=True)
    #     ax_ins.set_xlim([box[0], box[0] + box[2]])
    #     ax_ins.set_ylim([box[1], box[1] + box[3]])
    
    # if QQ[i]:
    #     mesh = np.load(meshes[i],
    #         allow_pickle=True)
    #     Q = np.load(QQ[i], mmap_mode='r')
    #     Q = np.nanmean(np.abs(Q[:,-1,:]), axis=-1)
    #     print('max:', np.max(Q))
    #     lc,sm = plotchannels(mesh, Q, vmin=5, vmax=50, cmap=cmocean.cm.ice_r)
    #     axs_inset[i].add_collection(lc)

    # ax.contour(xx, yy, mask, levels=(0.1,), colors=('b',), linewidths=2)
    # ax.pcolormesh(xx, yy, mask)
    ax_ins.set_aspect('equal')

    ax_ins.spines[['left', 'right', 'top', 'bottom']].set_visible(False)
    ax_ins.set_xticks([])
    ax_ins.set_yticks([])

for box in gl_boxes:
    if box[0]!=0:
        x1,y1,x2,y2 = box
        R = Rectangle((x1, y1), x2-x1, y2-y1, linestyle='solid',
            facecolor='none', edgecolor='k')
        ax.add_patch(R)

# ax.spines[['left', 'right', 'top', 'bottom']].set_visible(False)
cax  = ax.inset_axes((-0.4, 0.75, 0.45, 0.035))
cax2 = ax.inset_axes((-0.4, 0.575, 0.45, 0.035))
cax3 = ax.inset_axes((-0.4, 0.4, 0.45, 0.035))

fig.colorbar(pc, cax=cax, orientation='horizontal', label='Bed elevation (m)',
    extend='both')
cax.xaxis.set_label_position('top')

cbar2 = fig.colorbar(ins_mappable, cax=cax2, orientation='horizontal', 
    label='Mesh area (m$^2$)', extend='both')
# cbar2.set_ticks([0, 2.5, 5.])
cax2.xaxis.set_label_position('top')

# cbar3 = fig.colorbar(sm, cax=cax3, orientation='horizontal',
#     label='Channel discharge (m$^3$ s$^{-1}$)', extend='both')
# cax3.xaxis.set_label_position('top')
# cbar3.set_ticks([5, 25, 50])
cax3.set_xlabel('Channel discharge (m$^3$ s$^{-1}$)')
cax3.xaxis.set_label_position('top')
cax3.set_yticks([])

# Subplot labels
axs_inset[0].text(rect_boxes[0][0] + 15e3, 
    rect_boxes[0][1] + rect_boxes[0][3] + 15e3,
    'a', fontweight='bold', fontsize=10)
# ax.text(rect_boxes[0][0], rect_boxes[0][1],
#     'a', fontweight='normal', ha='right', va='top')

x1,y1,x2,y2 = gl_boxes[0]
R = Rectangle((x1, y1), x2-x1, y2-y1, linestyle='solid',
    facecolor='none', edgecolor='k')
axs_inset[0].add_patch(R)

x1,y1,x2,y2 = gl_boxes[1]
R = Rectangle((x1, y1), x2-x1, y2-y1, linestyle='solid',
    facecolor='none', edgecolor='k')
axs_inset[0].add_patch(R)

axs_inset[1].text(rect_boxes[1][0] + 250e3, 
    rect_boxes[1][1] + rect_boxes[1][3] + 50e3,
    'b', fontweight='bold', fontsize=10)
# ax.text(rect_boxes[1][0], rect_boxes[1][1],
#     'b', fontweight='normal', ha='right', va='top')

axs_inset[2].text(rect_boxes[2][0] - 15e3, 
    rect_boxes[2][1] + rect_boxes[2][3] + 50e3,
    'c', fontweight='bold', fontsize=10, va='bottom', ha='left')
# ax.text(rect_boxes[2][0] + rect_boxes[2][2] - 25e3, 
#     rect_boxes[2][1] + 15e3,
#     'c', fontweight='normal', ha='right', va='bottom')

axs_inset[3].text(rect_boxes[3][0] - 50e3,
    rect_boxes[3][1] + rect_boxes[3][3] - 150e3,
    'd', fontweight='bold', fontsize=10, ha='right', va='top')

x1,y1,x2,y2 = gl_boxes[3]
R = Rectangle((x1, y1), x2-x1, y2-y1, linestyle='solid',
    facecolor='none', edgecolor='k')
axs_inset[3].add_patch(R)

axs_inset[4].text(rect_boxes[4][0] + 150e3,
    rect_boxes[4][1] + rect_boxes[4][3] - 250e3,
    'e', fontweight='bold', fontsize=10, ha='left', va='top')

x1,y1,x2,y2 = gl_boxes[4]
R = Rectangle((x1, y1), x2-x1, y2-y1, linestyle='solid',
    facecolor='none', edgecolor='k')
axs_inset[4].add_patch(R)
# ax.text(rect_boxes[3][0] + rect_boxes[3][2], 
#     rect_boxes[3][1] + rect_boxes[3][3],
#     'd', fontweight='normal', ha='left', va='top')

# axs_inset[4].text(rect_boxes[4][0] + 15e3, 
#     rect_boxes[4][1] + rect_boxes[4][3] + 15e3,
#     'e', fontweight='bold', fontsize=10)
# ax.text(rect_boxes[4][0] + rect_boxes[4][2] + 15e3,
#     rect_boxes[4][1] + rect_boxes[4][3] + 15e3,
#     'e', fontweight='normal', ha='left', va='bottom')

axs_inset[5].text(rect_boxes[5][0] + 15e3, 
    rect_boxes[5][1] + rect_boxes[5][3] + 15e3,
    'f', fontweight='bold', fontsize=10)

x1,y1,x2,y2 = gl_boxes[5]
R = Rectangle((x1, y1), x2-x1, y2-y1, linestyle='solid',
    facecolor='none', edgecolor='k')
axs_inset[5].add_patch(R)
# ax.text(rect_boxes[5][0] - 15e3,
#     rect_boxes[5][1] + rect_boxes[5][3] + 15e3,
#     'f', fontweight='normal', ha='right', va='bottom')

axs_inset[6].text(rect_boxes[6][0] + 15e3, 
    rect_boxes[6][1] + rect_boxes[6][3] + 15e3,
    'g', fontweight='bold', fontsize=10)
# ax.text(rect_boxes[6][0]  + rect_boxes[6][2] + 15e3,
#     rect_boxes[6][1] - 15e3,
#     'g', fontweight='normal', ha='left', va='top',
#     bbox=dict(boxstyle='square,pad=0.125', fc='white', ec='none'))
    
axs_inset[7].text(rect_boxes[7][0] + 15e3, 
    rect_boxes[7][1] + rect_boxes[7][3] + 25e3,
    'h', fontweight='bold', fontsize=10)

fig.subplots_adjust(left=0.25, right=0.75, bottom=0.25, top=0.75,
    wspace=0, hspace=0)

# plt.show()
fig.savefig('antarctica_overview.png', dpi=400)
