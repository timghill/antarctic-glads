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

bed[mask==0] = np.nan
xx,yy = np.meshgrid(x,y)

fig,ax = plt.subplots(figsize=(8,8))
ax.contour(xx, yy, mask, levels=(2.1,), colors=('k',), linewidths=1.5)
# ax.contour(xx, yy, mask, levels=(0.1,), colors=('b',), linewidths=2)
# ax.pcolormesh(xx, yy, mask)
pc = ax.pcolormesh(xx, yy, bed, cmap=Zcmap, 
    vmin=-2000, vmax=2000, alpha=Zalpha)
ax.set_aspect('equal')

# ax.spines[['left', 'right', 'top', 'bottom']].set_visible(False)
# ax.set_xticks([])
# ax.set_yticks([])
ax.grid()

for xy in gl_xy:
    if xy[0]!=0:
        ax.plot(xy[0], xy[1], 'r*', markersize=10)

for box in gl_boxes:
    if box[0]!=0:
        x1,y1,x2,y2 = box
        R = Rectangle((x1, y1), x2-x1, y2-y1, linestyle='dashed',
            facecolor='none', edgecolor='k')
        ax.add_patch(R)

plt.show()
