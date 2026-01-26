"""
Plot grounding line channel discharge statistics and compare to literature values
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

fs = 8

# Approximate coordinates of grounding lines
# From Ehrenfeucht et al. (2025, Geophys Res Letters)
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
# print(xyall)

# Mean Q values from the literature. See Table A1 for sources
Qconstraint = {
    'G-H': np.array([92, 42.5]),
    'B-C': np.array([202]),
    'C-Cp': np.array([14.1]),
    'Cp-D': np.array([30.3]),
    'Jpp-K': np.array([82]),
    'J-Jpp': np.array([47, 45.1]),
}

labels = {
    'G-H': ['(a) Thwaites', '(b) PIG'],
    'B-C': ['(c) Lambert'],
    'C-Cp': ['(d) Denman'],
    'Cp-D': ['(e) Totten'],
    'Jpp-K': ['(f) Recovery'],
    'J-Jpp': ['(g) Rutford', '(h) Academy'],
}

# Maximum distance away from Ehrenfeucht et al. (2025) 
# channel locations to search
dthreshold = 50e3

nsectors = len(basins)

N = np.sum([glxy[basin].shape[0] for basin in basins])
# print('N:', N)
d = 100

ncols = 4
# nrows = int(np.ceil(N/ncols))
nrows = 2

# Search for largest channel near each grounding line
glnumber = 0
fig,axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(6, 4),
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
        # print('min dist:', np.min(dist))
        isclose = dist<dthreshold
        Qmax = np.nanmax(Q[isclose], axis=0)
        iq = np.nanargmax(np.quantile(Q[isclose], 0.95, axis=1))
        # print(iq)
        xi = xc[isclose][iq]
        yi = yc[isclose][iq]
        # print(xi.shape)
        print(xi/1e3, yi/1e3)
        ax = axs.flat[glnumber]
        ax.set_title(labels[basin][j], fontsize=fs)
        ax.hist(Qmax, edgecolor='k')
        ax.axvline(Qconstraint[basin][j], color='k', label='Literature mean')

        modelled_discharge[glnumber] = Qmax
        constraint_discharge[glnumber] = Qconstraint[basin][j]
        glnumber += 1

# Find the simulation with best fit over all grounding lines
Q_rel_error = (np.abs(modelled_discharge - constraint_discharge)/constraint_discharge)**2
print('Q_rel_error:', Q_rel_error.shape)
Q_sum_rel_error = np.mean(Q_rel_error, axis=0)
print('Q_sum_rel_error:', Q_sum_rel_error.shape)
# print(np.sort(Q_sum_rel_error))
print('Min error:', np.min(Q_sum_rel_error))
sim_index = np.argmin(Q_sum_rel_error)
print('Sim index:', sim_index)
print(Q_sum_rel_error[sim_index])

# Plot discharge histogram and map
for ax in axs[-1].flat:
    ax.set_xlabel('Discharge (m$^3$ s$^{-1}$)', fontsize=fs)

for ax in axs[:, 0]:
    ax.set_ylabel('Count (n=100)', fontsize=fs)

for i in range(N):
    ax = axs.flat[i]
    ax.axvline(modelled_discharge[i, sim_index], color='r', label='Best-fit model')
    ax.tick_params(labelsize=fs-1)
    ax.grid()

axs.flat[0].legend(loc='lower left', frameon=False, fontsize=fs-1,
    bbox_to_anchor=(0,1.2, 1, 0.2), ncols=2)

fig.subplots_adjust(wspace=0.2, hspace=0.4, left=0.08, right=0.975,
    bottom=0.125, top=0.85)
fig.savefig('figures/gl_discharge.png', dpi=400)
fig.savefig('../../manuscript/A01.pdf')
fig.savefig('../../manuscript/A01.png', dpi=300)

print('Constant discharge:')
print(constraint_discharge.squeeze())
print('Modelled discharge:')
print(modelled_discharge[:, sim_index].squeeze().round(2))


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
