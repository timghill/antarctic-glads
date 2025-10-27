import numpy as np
from matplotlib import pyplot as plt
from matplotlib.tri import Triangulation

mesh = np.load('../data/geom/mesh.npy', allow_pickle=True)
mtri = Triangulation(mesh['x'], mesh['y'], mesh['elements']-1)

C = np.load('solutions/friction_coefficient_POC_nonlinear.npy')
fig,ax = plt.subplots()
tpc = ax.tripcolor(mtri, C, vmin=C.min(), vmax=1)
fig.colorbar(tpc)
ax.set_aspect('equal')
fig.savefig('solutions/friction_coefficient_POC_nonlinear.png', dpi=400)



C = np.load('solutions/friction_coefficient_glads_nonlinear.npy')
fig,ax = plt.subplots()
tpc = ax.tripcolor(mtri, C, vmin=C.min(), vmax=2)
fig.colorbar(tpc)
ax.set_aspect('equal')
fig.savefig('solutions/friction_coefficient_glads_nonlinear.png', dpi=400)



C = np.load('solutions/friction_coefficient_RF_nonlinear.npy')
fig,ax = plt.subplots()
tpc = ax.tripcolor(mtri, C, vmin=C.min(), vmax=2)
fig.colorbar(tpc)
ax.set_aspect('equal')
fig.savefig('solutions/friction_coefficient_RF_nonlinear.png', dpi=400)