import numpy as np
from matplotlib import pyplot as plt
from matplotlib.tri import Triangulation
import cmocean

basins = [
    'B-C',
    # 'C-Cp',
    'Cp-D',
    'G-H',
    # 'Jpp-K',
]

for basin in basins:
    fig,axs = plt.subplots(figsize=(16,8), ncols=4, nrows=2)

    def pad(levelset, z):
        if z.ndim==1:
            zpad = np.nan*np.zeros(len(levelset))
            zpad[levelset>0] = z
        elif z.ndim==2:
            zpad = np.nan*np.zeros((len(levelset), z.shape[1]))
            zpad[levelset>0,:] = z
        return zpad

    mesh = np.load(f'../../issm/{basin}/data/geom/mesh.npy', allow_pickle=True)
    levelset = np.load(f'../../issm/{basin}/data/geom/ocean_levelset.npy')
    para_cv_f = pad(levelset, np.load(f'data/CV_{basin}_f_rf.npy'))
    para_cv_N = pad(levelset, np.load(f'data/CV_{basin}_N_rf.npy'))
    para_glads_N = pad(levelset, np.load(f'data/CV_{basin}_N_glads.npy'))
    para_glads_f = np.load(f'../../issm/{basin}/glads/ff.npy')

    mean_cv_f = pad(levelset, np.load(f'../mean/data/CV_{basin}.npy'))
    mean_cv_N = pad(levelset, np.load(f'../mean/data/CV_{basin}_N_rf.npy'))

    para_cv_f_mean = np.mean(para_cv_f, axis=1)
    para_cv_N_mean = np.mean(para_cv_N, axis=1)
    glads_N_mean = np.mean(para_glads_N, axis=1)
    glads_f_mean = np.mean(para_glads_f, axis=1)

    print(para_cv_N_mean.shape)
    print(mesh['numberofvertices'])

    mtri = Triangulation(mesh['x'], mesh['y'], mesh['elements']-1)
    axs[0,0].tripcolor(mtri, para_cv_f_mean, vmin=0.5, vmax=1, cmap=cmocean.cm.dense)
    axs[0,0].set_title('Parameters-mean')
    mask = np.logical_and(glads_f_mean>=0, glads_f_mean<=1)
    err = para_cv_f_mean - glads_f_mean
    r2 = 1 - np.nanvar(err[mask])/np.nanvar(glads_f_mean[mask])
    axs[0,1].tripcolor(mtri, para_cv_f_mean - glads_f_mean, vmin=-0.1, vmax=0.1, cmap=cmocean.cm.balance)
    axs[0,1].set_title(f'$R^2 = {r2:.3f}$', fontsize=10)
    axs[0,2].tripcolor(mtri, mean_cv_f, vmin=0.5, vmax=1, cmap=cmocean.cm.dense)
    axs[0,2].set_title('Mean only')
    err = mean_cv_f - glads_f_mean
    r2 = 1 - np.nanvar(err[mask])/np.nanvar(glads_f_mean[mask])
    axs[0,3].tripcolor(mtri, mean_cv_f - glads_f_mean, vmin=-0.1, vmax=0.1, cmap=cmocean.cm.balance)
    axs[0,3].set_title(f'$R^2 = {r2:.3f}$', fontsize=10)
    
    axs[1,0].tripcolor(mtri, para_cv_N_mean/1e6, vmin=0, vmax=4, cmap=cmocean.cm.haline)
    axs[1,1].tripcolor(mtri, para_cv_N_mean/1e6 - glads_N_mean/1e6, vmin=-1, vmax=1, cmap=cmocean.cm.balance)
    err = para_cv_N_mean - glads_N_mean
    r2 = 1 - np.nanvar(err[mask])/np.nanvar(glads_N_mean[mask])
    axs[1,1].set_title(f'$R^2 = {r2:.3f}$', fontsize=10)
    axs[1,2].tripcolor(mtri, mean_cv_N/1e6, vmin=0, vmax=4, cmap=cmocean.cm.haline)
    err = mean_cv_N - glads_N_mean
    r2 = 1 - np.nanvar(err[mask])/np.nanvar(glads_N_mean[mask])
    axs[1,3].tripcolor(mtri, mean_cv_N/1e6 - glads_N_mean/1e6, vmin=-1, vmax=1, cmap=cmocean.cm.balance)
    axs[1,3].set_title(f'$R^2 = {r2:.3f}$', fontsize=10)

    for ax in axs.flat:
        ax.set_aspect('equal')
    fig.savefig(f'figures/CV_compare_{basin}.png', dpi=400)
