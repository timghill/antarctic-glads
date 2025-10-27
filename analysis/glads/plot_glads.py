import numpy as np

from matplotlib import pyplot as plt
from matplotlib.tri import Triangulation

basins = [
    'G-H',
    'J-Jpp',
    'Cp-D',
    'C-Cp',
    'B-C',
    'Jpp-K',
]

simindex = 95

fig,ax = plt.subplots()

for basin in basins:
    print(basin)
    N = np.load(f'../../issm/{basin}/glads/N.npy')
    Nbest = N[:, simindex]
    mesh = np.load(f'../../issm/{basin}/data/geom/mesh.npy', allow_pickle=True)
    mtri = Triangulation(mesh['x'], mesh['y'], mesh['elements']-1)
    pc = ax.tripcolor(mtri, Nbest/1e6, vmin=0, vmax=5)

    N_med = np.nanmedian(N, axis=0)
    Nbest_med = np.nanmedian(Nbest)
    N_qntl = len(N_med[N_med<=Nbest_med])/100
    print('N_qntl:', N_qntl)

ax.set_aspect('equal')

fig.colorbar(pc, label='N (MPa)', shrink=0.8)
fig.savefig('figures/N_bestfit.png', dpi=400)
