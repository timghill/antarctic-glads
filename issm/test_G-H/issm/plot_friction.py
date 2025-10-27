import numpy as np
from matplotlib import pyplot as plt
from matplotlib.tri import Triangulation
import cmocean


C = np.load('C.npy').squeeze()
mesh = np.load('../data/geom/mesh.npy', allow_pickle=True)
mtri = Triangulation(mesh['x'], mesh['y'], mesh['elements']-1)
fig,ax = plt.subplots()
tpc = ax.tripcolor(mtri, C, vmin=0.05, vmax=2)
fig.colorbar(tpc, label='Friction coefficient')
ax.set_aspect('equal')
fig.savefig('C.png', dpi=400)

print('mean:', np.mean(C[C>0]))