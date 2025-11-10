import numpy as np
from matplotlib import pyplot as plt
from matplotlib.tri import Triangulation
from scipy.interpolate import griddata

import cmocean

fine = 'G-H_fine'
coarse = 'G-H'
cases = [coarse, fine]
index = 15

Nfine = np.load(f'../../issm/{fine}/glads/RUN/N.npy')
Lfine = np.load(f'../../issm/{fine}/data/geom/ocean_levelset.npy')
dNdt = Nfine[:, -2] - Nfine[:, -3]

print(np.mean(Nfine[Lfine>0, :], axis=0))
Nfine = Nfine[:, -1]


Ncoarse = np.load(f'../../issm/{coarse}/glads/N.npy')[:, index-1]
Ns = [Ncoarse, Nfine]



xys = []
mtris = []

fig, axs = plt.subplots(figsize=(8, 8), ncols=2, nrows=2, sharey=True, sharex=True)
for i in range(2):
    basin = cases[i]
    mesh = np.load(f'../../issm/{basin}/data/geom/mesh.npy', allow_pickle=True)
    N = Ns[i]
    print(N.shape)

    xys.append((mesh['x'], mesh['y']))

    ax = axs[0, i]
    mtri = Triangulation(mesh['x'], mesh['y'], mesh['elements']-1)
    mtris.append(mtri)
    pc = ax.tripcolor(mtri, N/1e6, vmin=0, vmax=4, cmap=cmocean.cm.haline) 
    
    ax.set_aspect('equal')

print('Interpolating fine -> coarse')
Nfine_interp = griddata(xys[1], Nfine, xys[0], method='linear')
dNdt_interp = griddata(xys[1], dNdt, xys[0], method='linear')
print('done interpolating')

Lcoarse = np.load(f'../../issm/{coarse}/data/geom/ocean_levelset.npy')

ax = axs[1,0]

deltaN = (Nfine_interp - Ncoarse)/1e6
deltaN[Lcoarse<0] = np.nan
diff = ax.tripcolor(mtris[0], deltaN, vmin=-0.5, vmax=0.5, cmap=cmocean.cm.diff)
ax.set_aspect('equal')

dt = axs[1,1].tripcolor(mtris[0], dNdt_interp/1e6*5, vmin=-0.5, vmax=0.5, cmap=cmocean.cm.balance)

cbar = fig.colorbar(pc, ax=axs[0], label='$N$ (MPa)')

cbar = fig.colorbar(diff, ax=axs[1], label=r'$\Delta N$ (MPa)')

# cbar = fig.colorbar(dt, ax=axs)
fig.savefig('mesh_refinement_N.png', dpi=400)

