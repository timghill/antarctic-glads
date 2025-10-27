import numpy as np
from matplotlib import pyplot as plt
from matplotlib.tri import Triangulation
import cmocean

basin = 'B-C'

Y = np.load(f'../../issm/{basin}/glads/ff.npy')
Y = np.mean(Y, axis=1)
Yhat = np.load(f'data/CV_{basin}.npy')

mask = np.logical_and(Y<=1, Y>=0)

mesh = np.load(f'../../issm/{basin}/data/geom/mesh.npy', allow_pickle=True)
mtri = Triangulation(mesh['x'], mesh['y'], mesh['elements']-1)
levelset = np.load(f'../../issm/{basin}/data/geom/ocean_levelset.npy').astype(int)

fig,axs = plt.subplots(figsize=(10, 8), nrows=2, ncols=2)

Yfull = np.nan*np.zeros(mesh['numberofvertices'])
Yhatfull = np.nan*np.zeros(mesh['numberofvertices'])
Yfull[levelset>0] = Y[levelset>0]
Yhatfull[levelset>0] = Yhat

pc = axs[0,0].tripcolor(mtri, Yfull, vmin=0.5, vmax=1, cmap=cmocean.cm.dense)
axs[0,1].tripcolor(mtri, Yhatfull, vmin=0.5, vmax=1, cmap=cmocean.cm.dense)
epc = axs[1,0].tripcolor(mtri, Yhatfull-Yfull, vmin=-0.25, vmax=0.25, cmap=cmocean.cm.balance)

axs[1,1].scatter(Yfull[mask], Yhatfull[mask], s=2, alpha=0.2)
axs[1,1].grid()

axs[0,0].set_title('GlaDS', fontsize=24)
axs[0,1].set_title('Random Forest', fontsize=24)
axs[1,0].set_title('Random Forest - GlaDS', fontsize=24)
axs[1,1].set_xlabel('GlaDS', fontsize=20)
axs[1,1].set_ylabel('Random Forest', fontsize=20)
R2 = 1 - np.nanvar(Yfull[mask] - Yhatfull[mask])/np.nanvar(Yfull[mask])
axs[1,1].set_title('R$^2$ = {:.3f}'.format(R2), fontsize=24)

axs[1,1].set_xlim([0, 1])
axs[1,1].set_ylim([0, 1])


for ax in axs.flat:
    ax.set_aspect('equal')
    ax.tick_params(labelsize=16)

for ax in axs.flat[:-1]:
    ax.spines[['left', 'right', 'top', 'bottom']].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

fig.subplots_adjust(left=0.07, right=0.95, bottom=0.1, top=0.9, hspace=0.2, wspace=0.2)

cb1 = fig.colorbar(pc, ax=axs[0], location='left', fraction=0.05)
cb2 = fig.colorbar(epc, ax=axs[1], location='left', fraction=0.05)

cb1.set_label('Flotation fraction', fontsize=20)
cb2.set_label(r'$\Delta$ Flotation fraction', fontsize=20)

for cb in (cb1, cb2):
    cb.ax.tick_params(labelsize=16)

fig.savefig('figures/CV_gridded.png', dpi=400, transparent=False)


