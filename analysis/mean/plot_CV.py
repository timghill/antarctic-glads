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

axs[0,0].set_title('GlaDS', fontsize=20)
axs[0,1].set_title('Random Forest', fontsize=20)
axs[1,0].set_title('Random Forest - GlaDS', fontsize=20)
axs[1,1].set_xlabel('GlaDS', fontsize=20)
axs[1,1].set_ylabel('Random Forest', fontsize=20)
R2 = 1 - np.nanvar(Yfull[mask] - Yhatfull[mask])/np.nanvar(Yfull[mask])
axs[1,1].set_title('R$^2$ = {:.3f}'.format(R2), fontsize=20)

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

fig.savefig('figures/CV.png', dpi=400, transparent=True)



## EFFECTIVE PRESSURE
bed = np.load(f'../../issm/{basin}/data/geom/bed.npy')
thick = np.load(f'../../issm/{basin}/data/geom/thick.npy')
rhowater = 1028
rhoice = 910
g = 9.81
N = rhoice*g*thick*(1 - Y)
Nhat = rhoice*g*thick[levelset==1]*(1 - Yhat)
mask = np.logical_and(Y<=1, Y>=0)

mesh = np.load(f'../../issm/{basin}/data/geom/mesh.npy', allow_pickle=True)
mtri = Triangulation(mesh['x'], mesh['y'], mesh['elements']-1)
levelset = np.load(f'../../issm/{basin}/data/geom/ocean_levelset.npy').astype(int)

fig,axs = plt.subplots(figsize=(10, 8), nrows=2, ncols=2)

Nfull = np.nan*np.zeros(mesh['numberofvertices'])
Nhatfull = np.nan*np.zeros(mesh['numberofvertices'])
Nfull[levelset>0] = np.load(f'data/CV_{basin}_N_glads.npy')
Nhatfull[levelset>0] = np.load(f'data/CV_{basin}_N_rf.npy')

pc = axs[0,0].tripcolor(mtri, Nfull/1e6, vmin=0, vmax=4, cmap=cmocean.cm.haline)
axs[0,1].tripcolor(mtri, Nhatfull/1e6, vmin=0, vmax=4, cmap=cmocean.cm.haline)
epc = axs[1,0].tripcolor(mtri, (Nhatfull-Nfull)/1e6, vmin=-1, vmax=1, cmap=cmocean.cm.balance)

axs[1,1].scatter(Nfull[mask]/1e6, Nhatfull[mask]/1e6, s=2, alpha=0.2)
axs[1,1].grid()

axs[0,0].set_title('GlaDS', fontsize=20)
axs[0,1].set_title('Random Forest', fontsize=20)
axs[1,0].set_title('Random Forest - GlaDS', fontsize=20)
axs[1,1].set_xlabel('GlaDS $N$ (MPa)', fontsize=20)
axs[1,1].set_ylabel('Random Forest $N$ (MPa)', fontsize=20)
R2 = 1 - np.nanvar(Nfull[mask] - Nhatfull[mask])/np.nanvar(Nfull[mask])
axs[1,1].set_title('R$^2$ = {:.3f}'.format(R2), fontsize=20)

axs[1,1].set_xlim([0, 4])
axs[1,1].set_ylim([0, 4])
axs[1,1].set_xticks([0, 1, 2, 3, 4])
axs[1,1].set_yticks([0, 1, 2, 3, 4])


for ax in axs.flat:
    ax.set_aspect('equal')
    ax.tick_params(labelsize=20)

for ax in axs.flat[:-1]:
    ax.spines[['left', 'right', 'top', 'bottom']].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

fig.subplots_adjust(left=0.095, right=0.95, bottom=0.1, top=0.9, hspace=0.3, wspace=0.2)

cb1 = fig.colorbar(pc, ax=axs[0], location='left', fraction=0.05)
cb2 = fig.colorbar(epc, ax=axs[1], location='left', fraction=0.05)

# cb1.set_label('Effective pressure (MPa)', fontsize=20)
# cb2.set_label(r'$\Delta$ Effective pressure (MPa)', fontsize=20)
cb1.ax.text(-6., 0.5, 'Effective pressure (MPa)', 
    fontsize=20, ha='right', va='center', rotation=90,
    transform=cb1.ax.transAxes)
cb2.ax.text(-6., 0.5, r'$\Delta$Effective pressure (MPa)', 
    fontsize=20, ha='right', va='center', rotation=90,
    transform=cb2.ax.transAxes)

for cb in (cb1, cb2):
    cb.ax.tick_params(labelsize=20)

fig.savefig('figures/CV_N.png', dpi=400, transparent=False)

