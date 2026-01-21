import numpy as np
from matplotlib import pyplot as plt
from matplotlib.tri import Triangulation
import cmocean
from matplotlib.patches import Rectangle

from utils import plotchannels


basin = 'B-C'

mesh = np.load(f'../../issm/{basin}/data/geom/mesh.npy', allow_pickle=True)
N = np.load(f'../../issm/{basin}/glads/N.npy')
N = np.nanmean(N, axis=1)
mtri = Triangulation(mesh['x'], mesh['y'], mesh['elements']-1)
levelset = np.load(f'../../issm/{basin}/data/geom/ocean_levelset.npy')
N[levelset<1] = np.nan

Q = np.abs(np.load(f'../../issm/{basin}/glads/Q.npy').mean(axis=1))

fig,ax = plt.subplots(figsize=(10, 10))
ax.tripcolor(mtri, N/1e6, vmin=0, vmax=4, cmap=cmocean.cm.haline)
ax.set_aspect('equal')
ax.spines[['left', 'right', 'top', 'bottom']].set_visible(False)
ax.set_xticks([])
ax.set_yticks([])

lc,sm = plotchannels.plotchannels(mesh, Q, vmin=2, vmax=200,
    lmin=1, lmax=3, cmap=cmocean.cm.turbid)

ax.add_collection(lc)

xmax = np.max(mesh['x'])
ymax = np.max(mesh['y'])
scale = Rectangle((xmax-250e3, ymax-100e3), 200e3, 10e3, color='black')
ax.add_patch(scale)
ax.text(xmax-150e3, ymax-80e3, '200 km', ha='center', fontsize=24)

# cax = ax.inset_axes()

fig.subplots_adjust(left=0.025, right=0.975, bottom=0.025, top=0.975)
fig.savefig('figures/glads.png', dpi=400)
