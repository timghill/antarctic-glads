import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import griddata

u_glads_present = np.load('solutions/u_glads_present.npy')
u_glads_future = np.load('solutions/u_glads_future.npy')
u_rf_present = np.load('solutions/u_rf_present.npy')
u_rf_future = np.load('solutions/u_rf_future.npy')
u_poc_present = np.load('solutions/u_poc_present.npy')
u_poc_future = np.load('solutions/u_poc_future.npy')

ss,xx,yy = np.load('../data/geom/flowline_00.npy')


# colors = ['#89b6bc', '#0d7d87', 'gray', '#ff5a5e', '#c31e23', 'dimgray']
colors = ['#89b6bc', '#ff5a5e', 'gray', '#0d7d87', '#c31e23', 'dimgray']
ls = ['solid', 'solid', 'dashed', 'solid', 'solid', 'dashed']

u = [
    u_glads_present,
    u_rf_present,
    u_poc_present,
    u_glads_future,
    u_rf_future,
    u_poc_future,
]
labels = [
    'GlaDS present',
    'RF present',
    'POC present',
    'GlaDS future',
    'RF future',
    'POC future'
]

levelset = np.load('../data/geom/ocean_levelset.npy')
mesh = np.load('../data/geom/mesh.npy', allow_pickle=True)

levelset_interp = griddata((mesh['x'], mesh['y']), levelset, (xx,yy))
fig,ax = plt.subplots()
for i in range(len(u)):
    u_interp = griddata((mesh['x'], mesh['y']), u[i], (xx,yy))
    u_interp[levelset_interp<=0] = np.nan
    ax.plot(ss/1e3, u_interp, label=labels[i], color=colors[i],
        linestyle=ls[i])
    print(labels[i], np.nanmax(u_interp))

ax.set_xlim([200, 0])
ax.grid()
ax.legend()
ax.set_xlabel('Distance from present terminus (km)')
ax.set_ylabel('Speed (m/year)')
ax.set_title('Lambert')

fig.savefig('u_solutions.png', dpi=400)
