import numpy as np
from utils.RF import RFData


from matplotlib import pyplot as plt
from matplotlib.tri import Triangulation
from matplotlib.patches import Rectangle
import cmocean

basins = [
    'B-C',
    'C-Cp',
    'Cp-D',
    # 'Ep-F',
    'G-H',
    'Jpp-K',
    'J-Jpp',
]

features = [
    'bed',
]

yy = []
for i,basin in enumerate(basins):
    yi = np.load(f'../../issm/{basin}/glads/N.npy')
    yy.append(yi)


Y = np.concatenate(yy)
print(Y.shape)

Ymedian = np.mean(Y, axis=1)
# Ymedian = np.quantile(Y, 0.5 - 0.34, axis=1)
Ymedian[Ymedian<0] = 0
# Ymedian[Ymedian>1] = 1
# Ymedian = np.quantile(Y, 0.68, axis=1)
print(Ymedian.shape)

norm = np.linalg.norm(Y - Ymedian[:,None], axis=0)
print(norm.shape)

median_number = np.argmin(norm)
print(median_number)
print(norm[median_number])
print(np.sort(norm)[:10])
print(np.argsort(norm)[:10])


fig,ax = plt.subplots()

for basin in basins:
    print(basin)
    N = np.load(f'../../issm/{basin}/glads/N.npy')
    Nbest = N[:, median_number]
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

