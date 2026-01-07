import numpy as np
from matplotlib import pyplot as plt
from matplotlib.tri import Triangulation
import cmocean

basin = 'G-H'
flowline = 0

ss,xx,yy = np.load(f'../../issm/{basin}/data/geom/flowline_{flowline:02d}.npy')
mesh = np.load(f'../../issm/{basin}/data/geom/mesh.npy', allow_pickle=True)
levelset = np.load(f'../../issm/{basin}/data/geom/ocean_levelset.npy')

mtri = Triangulation(mesh['x'], mesh['y'], mesh['elements']-1)
fig,ax = plt.subplots()
ax.tripcolor(mtri, levelset)
ax.plot(xx, yy, color='w', linewidth=2)
ax.set_aspect('equal')
plt.show()
