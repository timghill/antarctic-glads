import numpy as np
from scipy import interpolate

from matplotlib import pyplot as plt
from matplotlib.tri import Triangulation
import cmocean

basins = ['J-Jpp', 'Jpp-K']

mesh1 = np.load('../../J-Jpp/data/geom/mesh.npy', allow_pickle=True)
mesh2 = np.load('../../Jpp-K/data/geom/mesh.npy', allow_pickle=True)

x = np.concatenate((mesh1['x'], mesh2['x']))
y = np.concatenate((mesh1['y'], mesh2['y']))
gladsxy = (x,y)

N1 = np.load('../../J-Jpp/glads/N.npy').mean(axis=1)
N2 = np.load('../../Jpp-K/glads/N.npy').mean(axis=1)
print(N1.shape)
print(N2.shape)
gladsN = np.concatenate((N1,N2))

mesh = np.load('../data/geom/mesh.npy', allow_pickle=True)
ocean_levelset = np.load('../data/geom/ocean_levelset.npy')
issmxy = (mesh['x'], mesh['y'])
issmN = interpolate.griddata(gladsxy, gladsN, issmxy, method='linear')
issmN[ocean_levelset<0] = 0
np.save('glads_N.npy', issmN)

fig,ax = plt.subplots()
mtri = Triangulation(mesh['x'], mesh['y'], mesh['elements']-1)
pc = ax.tripcolor(mtri, issmN/1e6, vmin=0, vmax=5)
ax.set_aspect('equal')
fig.colorbar(pc, label='N (Pa)')
fig.savefig('fris_glads_N.png', dpi=400)
