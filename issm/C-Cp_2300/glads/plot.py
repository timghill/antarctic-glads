import numpy as np
from matplotlib import pyplot as plt
from matplotlib.tri import Triangulation
import cmocean


mesh = np.load('../data/geom/mesh.npy', allow_pickle=True)
ff = np.load('ff.npy').mean(axis=1)

ss,xx,yy = np.load('../data/geom/flowline_00.npy')

mtri = Triangulation(mesh['x'], mesh['y'], mesh['elements']-1)

fig,ax = plt.subplots()
pc = ax.tripcolor(mtri, ff, vmin=0.5, vmax=1, cmap=cmocean.cm.dense)
ax.plot(xx, yy, color='white')
ax.set_aspect('equal')
fig.colorbar(pc, label='Flotation fraction')
fig.savefig('ff.png', dpi=400)


fig,ax = plt.subplots()
pc = ax.tripcolor(mtri, ff, vmin=0.5, vmax=1, cmap=cmocean.cm.dense)
ax.plot(xx, yy, color='white')
ax.set_aspect('equal')
ax.set_xlim([xx.min()-20e3, xx.max()+20e3])
ax.set_ylim([yy.min()-100e3, yy.max()+100e3])
fig.colorbar(pc, label='Flotation fraction')
fig.savefig('ff_close.png', dpi=400)


