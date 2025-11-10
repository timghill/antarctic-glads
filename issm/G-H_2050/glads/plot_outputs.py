import numpy as np
from matplotlib import pyplot as plt
from matplotlib.tri import Triangulation
import cmocean

ff = np.load('/home/tghill/scratch/antarctic-glads/issm/G-H_2050/glads/RUN/output_075/ff.npy')[:,-1]

print(ff.shape)

mesh = np.load('../data/geom/mesh.npy', allow_pickle=True)
levelset = np.load('../data/geom/ocean_levelset.npy')

mtri = Triangulation(mesh['x'], mesh['y'], mesh['elements']-1)

fig,ax = plt.subplots()
pc = ax.tripcolor(mtri, ff, vmin=0, vmax=1, cmap=cmocean.cm.dense)
fig.colorbar(pc, label='Flotation fraction')
ax.set_aspect('equal')
ax.tricontour(mtri, levelset, levels=(0,), colors=('w',))
fig.savefig('ff.png', dpi=400)

