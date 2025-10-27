"""
Plot grounding line discharge statistics and compare to literature values

TODO: find max within specified xy-bounds to make sure it really is
the grounding line discharge (more work)
"""

import numpy as np

from matplotlib import pyplot as plt
from matplotlib.tri import Triangulation
from matplotlib.patches import Rectangle
import cmocean
from scipy import interpolate
from scipy import stats
import netCDF4 as nc

from utils.plotchannels import plotchannels

basins = [
    'G-H',
    'Cp-D',
    'C-Cp',
    'B-C',
    'Jpp-K',
    'J-Jpp',
]


glxy = {
    'G-H': np.array([
        [-1.5205e6, -0.4637e6],
        [-1.5905e6, -0.2536e6],
        ]),
    'B-C': np.array([
        [1.6820e6, 0.7106e6],
    ]),
    'C-Cp': np.array([
        [2.5321e6, -0.4105e6],
    ]),
    'Cp-D': np.array([
        [2.275e6, -1.008e6],
    ]),
    'Jpp-K': np.array([
        [-5.88e5, 8.07e5],
    ]),
    'J-Jpp': np.array([
        [-1.261e6, 1.46e5],
        [-9.43e5, 2.66e5]
    ]),
}

xyall = []
for basin in basins:
    xyall.extend(glxy[basin])
xyall = np.array(xyall)
print(xyall)

Qconstraint = {
    'G-H': np.array([92, 42.5]),
    'B-C': np.array([202]),
    'C-Cp': np.array([14.1]),
    'Cp-D': np.array([30.3]),
    'Jpp-K': np.array([82]),
    'J-Jpp': np.array([47, 45.1]),
}

labels = {
    'G-H': ['Thwaites', 'PIG'],
    'B-C': ['Lambert'],
    'C-Cp': ['Denman'],
    'Cp-D': ['Totten'],
    'Jpp-K': ['Recovery'],
    'J-Jpp': ['Rutford', 'Academy'],
}

dthreshold = 50e3
# gl_xy = np.array([
#     [-1.5905e6, -0.2536e6],
#     [-1.5205e6, -0.4637e6],
#     [0, 0],
#     [2266.8e3, -999e3],
#     [2.5321e6, -0.4105e6],
#     [1.6820e6, 0.7106e6],
#     [0, 0],
#     [0, 0],

# ])

# meshes = [
#     '../../issm/thwaites/data/geom/thwaites_mesh.npy',
#     '../../issm/thwaites/data/geom/thwaites_mesh.npy',
#     None,
#     '../../issm/aurora/data/geom/mesh.npy',
#     '../../issm/denman/data/geom/mesh.npy',
#     '../../issm/amery/data/geom/amery_mesh.npy',
#     None,
#     None,
# ]

# N_files = [
#     '../../issm/thwaites/train/train_N.npy',
#     '../../issm/thwaites/train/train_N.npy',
#     None,
#     '../../issm/aurora/glads/N.npy',
#     '../../issm/denman/glads/N.npy',
#     '../../issm/amery/train/train_N.npy',
#     None,
#     None,
# ]

# # nsectors = len(Q_files)
# nsectors = 1

nsectors = len(basins)

N = np.sum([glxy[basin].shape[0] for basin in basins])
print('N:', N)
d = 100

ncols = 4
# nrows = int(np.ceil(N/ncols))
nrows = 2

glnumber = 0
fig,axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(10, 8),
    sharey=False)
constraint_discharge = np.zeros((N,1))
modelled_discharge = np.zeros((N, d))
for i in range(len(basins)):
    basin = basins[i]
    xy = glxy[basin]

    print(basin)

    Q = np.abs(np.load(f'../../issm/{basin}/glads/Q.npy'))
    mesh = np.load(f'../../issm/{basin}/data/geom/mesh.npy', allow_pickle=True)
    levelset = np.load(f'../../issm/{basin}/data/geom/ocean_levelset.npy')
    xc = np.mean(mesh['x'][mesh['connect_edge']], axis=1)
    yc = np.mean(mesh['y'][mesh['connect_edge']], axis=1)

    ni = xy.shape[0]
    for j in range(ni):
        xj,yj = xy[j]
        dist = np.sqrt((xc-xj)**2 + (yc-yj)**2)
        print('min dist:', np.min(dist))
        isclose = dist<dthreshold
        Qmax = np.nanmax(Q[isclose], axis=0)
        ax = axs.flat[glnumber]
        ax.set_title(labels[basin][j])
        ax.hist(Qmax)
        ax.axvline(Qconstraint[basin][j], color='k', label='Literature mean')

        modelled_discharge[glnumber] = Qmax
        constraint_discharge[glnumber] = Qconstraint[basin][j]
        glnumber += 1

Q_rel_error = (np.abs(modelled_discharge - constraint_discharge)/constraint_discharge)**2
print('Q_rel_error:', Q_rel_error.shape)
Q_sum_rel_error = np.mean(Q_rel_error, axis=0)
print('Q_sum_rel_error:', Q_sum_rel_error.shape)
print(np.sort(Q_sum_rel_error))
print('Min error:', np.min(Q_sum_rel_error))
sim_index = np.argmin(Q_sum_rel_error)
print('Sim index:', sim_index)
print(Q_sum_rel_error[sim_index])


for ax in axs[-1].flat:
    ax.set_xlabel('Discharge (m$^3$ s$^{-1}$)')

for ax in axs[:, 0]:
    ax.set_ylabel('Count (n=100)')

for i in range(N):
    ax = axs.flat[i]
    ax.axvline(modelled_discharge[i, sim_index], color='r', label='Best-fit model')

axs.flat[0].legend(loc='upper right', frameon=False)

fig.tight_layout()
fig.savefig('figures/gl_discharge.png', dpi=400)

fig,ax = plt.subplots()
ax.hist(Q_sum_rel_error)
ax.set_xlabel('Relative error')
ax.set_ylabel(f'Count (n={d})')
fig.savefig('figures/gl_discharge_error.png', dpi=400)

print(constraint_discharge)
print(modelled_discharge[:, sim_index])


theta_phys = np.loadtxt('../../issm/theta_physical.csv', delimiter=',', skiprows=1)
print('Winning parameters:', theta_phys[sim_index])
print('With error:', Q_rel_error[:, sim_index])


bedmachine = '../../data/bedmachine/BedMachineAntarctica-v3.nc'

dx = 8
with nc.Dataset(bedmachine, 'r') as bm:
    mask = bm['mask'][::dx, ::dx].astype(int)
    x = bm['x'][::dx].astype(np.float32)
    y = bm['y'][::dx].astype(np.float32)
    bed = bm['bed'][::dx, ::dx].astype(np.float32)

fig,ax = plt.subplots()
ax.contourf(x, y, mask, levels=(0.5,2.5, 4.5), colors=('gray', 'lightgray',))
ax.set_aspect('equal')
ax.spines[['left', 'right', 'top', 'bottom']].set_visible(False)
alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']

ax.set_xticks([])
ax.set_yticks([])
# for i,xy in enumerate(gl_xy):
emin = -1
emax = 1
cmap = cmocean.cm.balance
for i in range(N):
    erri = (modelled_discharge[i, sim_index] - constraint_discharge[i])/constraint_discharge[i]
    sc = ax.scatter(xyall[i,0], xyall[i,1], s=20, c=erri,
        vmin=emin, vmax=emax, cmap=cmap, edgecolor='k')

fig.subplots_adjust(left=0, bottom=0, top=1, right=0.9)
fig.colorbar(sc, label='Relative discharge difference', shrink=0.8)

fig.savefig('figures/gl_discharge_map.png', dpi=400)




# # 0: PIG
# sector = 0
# Q = np.load(Q_files[sector], mmap_mode='r')[:, -1, :]
# mesh = np.load(meshes[sector], allow_pickle=True)
# xc = np.mean(mesh['x'][mesh['connect_edge']], axis=1)
# yc = np.mean(mesh['y'][mesh['connect_edge']], axis=1)
# if gl_xy[sector][0]!=0:
#     dist = np.sqrt((xc-gl_xy[sector][0])**2 +(yc - gl_xy[sector][1])**2)
#     print('min dist:', dist.min())
#     isclose = dist<1e4
# else:
#     isclose = np.isfinite(gl_xy[sector][0])
# Qmax = np.nanmax(Q[isclose], axis=0)
    
# # fig,ax = plt.subplots(figsize=(5, 4))
# ax = axs.flat[ix]
# ix+=1
# ax.hist(Qmax, range=(0, 100), bins=10, density=False,)

# ax.axvline(40, color='k', label='Dow (2022)')
# ylim = ax.get_ylim()
# ax.axvline(45, color='b', label='Ehrenfeucht et al. (2024)')
# # ax.text(151+2, ylim[1]-2, 'Hager et al. (2022)', color='k', va='top')
# ax.set_ylim(ylim)
# ax.set_xlabel('Grounding line discharge (m$^3$ s$^{-1}$)')
# ax.set_ylabel('Number of simulations (n=100)')
# ax.set_title('(a) Pine Island')
# ax.legend(loc='upper right')

# # 0: Thwaites/Amundsen sea sector
# sector = 1
# Q = np.load(Q_files[sector], mmap_mode='r')[:, -1, :]
# mesh = np.load(meshes[sector], allow_pickle=True)
# xc = np.mean(mesh['x'][mesh['connect_edge']], axis=1)
# yc = np.mean(mesh['y'][mesh['connect_edge']], axis=1)
# if gl_xy[sector][0]!=0:
#     dist = np.sqrt((xc-gl_xy[sector][0])**2 +(yc - gl_xy[sector][1])**2)
#     print('min dist:', dist.min())
#     isclose = dist<1e4
# else:
#     isclose = np.isfinite(gl_xy[sector][0])
# Qmax = np.nanmax(Q[isclose], axis=0)
    
# # fig,ax = plt.subplots(figsize=(5, 4))
# ax = axs.flat[ix]
# ix += 1
# ax.hist(Qmax, range=(0, 400), bins=10, density=False,)
# ylim = ax.get_ylim()
# ax.fill_between([103, 151], [0, 0], [ylim[1], ylim[1]],
#     color='gray', alpha=1, zorder=0, edgecolor='none',
#     label='Hager et al. (2022)')

# ax.axvline(80, color='b', label='Dow (2022)')
# ax.axvline(69, color='r', label='Ehrenfeucht et al. (2024)')
# # ax.text(151+2, ylim[1]-2, 'Hager et al. (2022)', color='k', va='top')
# ax.set_ylim(ylim)
# ax.set_xlabel('Grounding line discharge (m$^3$ s$^{-1}$)')
# ax.set_ylabel('Number of simulations (n=100)')
# ax.set_title('(b) Thwaites')
# ax.legend(loc='upper right')

# # 2: Aurora subglacial basin
# sector = 3
# Q = np.load(Q_files[sector], mmap_mode='r')[:, -1, :]
# Qmax = np.nanmax(Q, axis=0)

# # fig,ax = plt.subplots(figsize=(5, 4))
# ax = axs.flat[ix]
# ix += 1
# ax.hist(Qmax, range=(0, 200), bins=10, density=False,)

# ax.axvline(24.7, color='k', label='Dow et al. (2020)')
# ax.axvline(25, color='b', label='Gwyther et al. (2023)')
# ax.axvline(35.82, color='r', label='Pelle et al. (2024)')
# ylim = ax.get_ylim()
# ax.legend(loc='upper right')
# ax.set_xlabel('Grounding line discharge (m$^3$ s$^{-1}$)')
# ax.set_title('(c) Totten')

# # 3. Denman-Scott
# sector = 4
# Q = np.load(Q_files[sector], mmap_mode='r')[:, -1, :]
# mesh = np.load(meshes[sector], allow_pickle=True)
# xc = np.mean(mesh['x'][mesh['connect_edge']], axis=1)
# yc = np.mean(mesh['y'][mesh['connect_edge']], axis=1)
# if gl_xy[sector][0]!=0:
#     dist = np.sqrt((xc-gl_xy[sector][0])**2 +(yc - gl_xy[sector][1])**2)
#     print('min dist:', dist.min())
#     isclose = dist<1e4
# else:
#     isclose = np.isfinite(gl_xy[sector][0])
# Qmax = np.nanmax(Q[isclose], axis=0)

# # fig,ax = plt.subplots(figsize=(5, 4))
# ax = axs.flat[ix]
# ix += 1
# ax.hist(Qmax, range=(0, 200), bins=10, density=False,)

# ax.axvline(9.50, color='k', label='Pelle et al. (2024)')
# ylim = ax.get_ylim()

# ax.axvline(15.8, color='b', label='McArthur et al. (2023)')
# ax.axvline(17, color='r', label='Ehrenfeucht et al. (2024)')
# ylim = ax.get_ylim()
# ax.legend(loc='upper right')
# ax.set_xlabel('Grounding line discharge (m$^3$ s$^{-1}$)')
# ax.set_title('(d) Denman')

# # 4. Amery ice shelf catchment
# sector = 5
# Q = np.load(Q_files[sector], mmap_mode='r')[:, -1, :]
# mesh = np.load(meshes[sector], allow_pickle=True)
# xc = np.mean(mesh['x'][mesh['connect_edge']], axis=1)
# yc = np.mean(mesh['y'][mesh['connect_edge']], axis=1)
# if gl_xy[sector][0]!=0:
#     dist = np.sqrt((xc-gl_xy[sector][0])**2 +(yc - gl_xy[sector][1])**2)
#     print('min dist:', dist.min())
#     isclose = dist<1e4
# else:
#     isclose = np.isfinite(gl_xy[sector][0])
# Qmax = np.nanmax(Q[isclose], axis=0)

# # fig,ax = plt.subplots(figsize=(5, 4))
# ax = axs.flat[ix]
# ix += 1
# ax.hist(Qmax, range=(0,1000), bins=10, density=False,)

# ylim = ax.get_ylim()
# ax.fill_between([50, 70], [0, 0], [ylim[1], ylim[1]],
#     color='gray', alpha=1, zorder=4, edgecolor='none',
#     label='Wearing et al. (2024)')
# ax.axvline(202, color='b', label='Ehrenfeucht et al. (2024)')
# ax.legend(loc='upper right')
# ax.set_title('(e) Lambert (Amery ice shelf)')
# ax.set_xlabel('Grounding line discharge (m$^3$ s$^{-1}$)')
# ax.set_ylim(ylim)


# # ax.fill_between([103, 151], [0, 0], [ylim[1], ylim[1]],
# #     color='gray', alpha=0.5, zorder=0, edgecolor='none')
# # ax.text(151+2, ylim[1]-2, 'Hager et al. (2022)', color='k', va='top')
# # ax.set_ylim(ylim)
# # ax.set_xlabel('Grounding line discharge (m$^3$ s$^{-1}$)')
# # ax.set_ylabel('Number of simulations (n=100)')
# # ax.set_title('Thwaites')

# # Contour the ice front for context

# bedmachine = '../../data/bedmachine/BedMachineAntarctica-v3.nc'

# dx = 8
# with nc.Dataset(bedmachine, 'r') as bm:
#     mask = bm['mask'][::dx, ::dx].astype(int)
#     x = bm['x'][::dx].astype(np.float32)
#     y = bm['y'][::dx].astype(np.float32)
#     bed = bm['bed'][::dx, ::dx].astype(np.float32)

# ax.contourf(x, y, mask, levels=(0.5,2.5, 4.5), colors=('gray', 'lightgray',))
# ax.set_aspect('equal')
# ax.spines[['left', 'right', 'top', 'bottom']].set_visible(False)
# alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
# ax = ax
# ax.set_xticks([])
# ax.set_yticks([])
# # for i,xy in enumerate(gl_xy):
# i = 0
# for xy in (gl_xy):
#     if xy[0]!=0:
#         ax.plot(xy[0], xy[1], '*', color='r')
#         ax.text(xy[0], xy[1], alphabet[i], ha='right', va='top', fontweight='bold')
#         i+=1


# fig.subplots_adjust(left=0.08, bottom=0.1, right=0.975, top=0.955,
#     hspace=0.4)

# fig.savefig('figures/groundingline_statistics.png', dpi=400)

# plt.show()