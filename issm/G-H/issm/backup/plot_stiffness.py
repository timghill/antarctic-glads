import numpy as np
from matplotlib import pyplot as plt
from matplotlib.tri import Triangulation



mesh = np.load('../data/geom/mesh.npy', allow_pickle=True)
levelset = np.load('../data/geom/ocean_levelset.npy')

B = np.load('B.npy')
B[levelset>0] = np.nan

mtri = Triangulation(mesh['x'], mesh['y'], mesh['elements']-1)

fig,ax = plt.subplots()
tpc = ax.tripcolor(mtri, B)
cb = fig.colorbar(tpc, label='Stifness B')
ax.set_aspect('equal')
fig.savefig('stiffness.png', dpi=400)

    