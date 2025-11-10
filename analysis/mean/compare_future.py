import numpy as np
from matplotlib import pyplot as plt
from matplotlib.tri import Triangulation
import cmocean

def main(basin, future):
    mesh = np.load(f'../../issm/{basin}/data/geom/mesh.npy', allow_pickle=True)
    present_mask = np.load(f'../../issm/{basin}/data/geom/ocean_levelset.npy')
    future_mask = np.load(f'../../issm/{basin}_{future}/data/geom/ocean_levelset.npy')
    glads_f_present = np.nanmean(np.load(f'../../issm/{basin}/glads/ff.npy'), axis=1)
    # print(glads_f_present)

    glads_N_present = np.nan*np.zeros(present_mask.shape)
    glads_N_present[present_mask>0] = np.load(f'data/pred_{basin}_N_glads.npy')

    glads_f_future = np.nanmean(np.load(f'../../issm/{basin}_{future}/glads/ff.npy'), axis=1)
    glads_N_future = np.nan*np.zeros(future_mask.shape)
    glads_N_future[future_mask>0] = np.load(f'data/pred_{basin}_{future}_N_glads.npy')

    rf_f_present = np.nan*np.zeros(present_mask.shape)
    rf_f_present[present_mask>0] = np.load(f'data/pred_{basin}.npy')
    rf_N_present = np.nan*np.zeros(present_mask.shape)
    rf_N_present[present_mask>0] = np.load(f'data/pred_{basin}_N_rf.npy')

    cv_f_present = np.nan*np.zeros(present_mask.shape)
    cv_f_present[present_mask>0] = np.load(f'data/CV_{basin}.npy')
    cv_N_present = np.nan*np.zeros(present_mask.shape)
    cv_N_present[present_mask>0] = np.load(f'data/CV_{basin}_N_rf.npy')

    rf_f_future = np.nan*np.zeros(future_mask.shape)
    rf_f_future[future_mask>0] = np.load(f'data/pred_{basin}_{future}.npy')
    rf_N_future = np.nan*np.zeros(future_mask.shape)
    rf_N_future[future_mask>0] = np.load(f'data/pred_{basin}_{future}_N_rf.npy')

    u_rf_future = np.load(f'../../issm/{basin}_{future}/issm/solutions/u_rf_future.npy')
    u_glads_future = np.load(f'../../issm/{basin}_{future}/issm/solutions/u_glads_future.npy')

    pm = np.logical_and(glads_f_present<=1, glads_f_present>=0)
    fm = np.logical_and(glads_f_future<=1, glads_f_future>=0)
    R2_f_present = 1 - np.nanvar(rf_f_present[pm] - glads_f_present[pm])/np.nanvar(glads_f_present[pm])
    R2_f_future = 1 - np.nanvar(rf_f_future[fm] - glads_f_future[fm])/np.nanvar(glads_f_future[fm])
    R2_N_present = 1 - np.nanvar(rf_N_present[pm] - glads_N_present[pm])/np.nanvar(glads_N_present[pm])
    R2_N_future = 1 - np.nanvar(rf_N_future[fm] - glads_N_future[fm])/np.nanvar(glads_N_future[fm])
    print(R2_f_present, R2_f_future, R2_N_present, R2_N_future)

    am = np.logical_and(pm, fm)
    rf_deltaf = rf_f_future[am] - rf_f_present[am]
    glads_deltaf = glads_f_future[am] - glads_f_present[am]
    rf_deltaN = rf_N_future[am] - rf_N_present[am]
    glads_deltaN = glads_N_future[am] - glads_N_present[am]
    r2_deltaf = 1 - np.nanvar(rf_deltaf - glads_deltaf)/np.nanvar(glads_deltaf)
    r2_deltaN = 1 - np.nanvar(rf_deltaN - glads_deltaN)/np.nanvar(glads_deltaN)
    print(r2_deltaf, r2_deltaN)

    r2_speed = 1 - np.nanvar(u_rf_future - u_glads_future)/np.nanvar(u_glads_future)
    print(r2_speed)

    mtri = Triangulation(mesh['x'], mesh['y'], mesh['elements']-1)
    fig,axs = plt.subplots(ncols=4, nrows=4)
    fpc = axs[0,0].tripcolor(mtri, glads_f_present, vmin=0.5, vmax=1, cmap=cmocean.cm.dense)
    axs[0,1].tripcolor(mtri, rf_f_present, vmin=0.5, vmax=1, cmap=cmocean.cm.dense)
    axs[0,2].tripcolor(mtri, cv_f_present, vmin=0.5, vmax=1, cmap=cmocean.cm.dense)
    axs[0,3].tripcolor(mtri, rf_f_future, vmin=0.5, vmax=1, cmap=cmocean.cm.dense)

    axs[1,1].set_title(r'$R^2$ = {:.3f}'.format(R2_f_present))
    axs[1,3].set_title(r'$R^2$ = {:.3f}'.format(R2_f_future))

    efpc = axs[1,1].tripcolor(mtri, rf_f_present-glads_f_present, vmin=-0.1, vmax=0.1, cmap=cmocean.cm.balance)
    axs[1,2].tripcolor(mtri, cv_f_present-glads_f_present, vmin=-0.1, vmax=0.1, cmap=cmocean.cm.balance)
    axs[1,3].tripcolor(mtri, rf_f_future-glads_f_future, vmin=-0.1, vmax=0.1, cmap=cmocean.cm.balance)

    Npc = axs[2,0].tripcolor(mtri, glads_N_present, vmin=0, vmax=5e6, cmap=cmocean.cm.haline)
    axs[2,1].tripcolor(mtri, rf_N_present, vmin=0, vmax=5e6, cmap=cmocean.cm.haline)
    axs[2,2].tripcolor(mtri, cv_N_present, vmin=0, vmax=5e6, cmap=cmocean.cm.haline)
    axs[2,3].tripcolor(mtri, rf_N_future, vmin=0, vmax=5e6, cmap=cmocean.cm.haline)

    eNpc = axs[3,1].tripcolor(mtri, rf_N_present-glads_N_present, vmin=-2e6, vmax=2e6, cmap=cmocean.cm.balance)
    axs[3,2].tripcolor(mtri, cv_N_present-glads_N_present, vmin=-2e6, vmax=2e6, cmap=cmocean.cm.balance)
    axs[3,3].tripcolor(mtri, rf_N_present-glads_N_future, vmin=-2e6, vmax=2e6, cmap=cmocean.cm.balance)

    axs[3,1].set_title(r'$R^2$ = {:.3f}'.format(R2_N_present))
    axs[3,3].set_title(r'$R^2$ = {:.3f}'.format(R2_N_future))

    for ax in axs.flat:
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines[['left', 'right', 'top', 'bottom']].set_visible(False)
    
    fig.colorbar(fpc, ax=axs[0], label='Flotation fraction')
    fig.colorbar(efpc, ax=axs[1], label=r'$\Delta$Flotation fraction')
    fig.colorbar(Npc, ax=axs[2], label='N (Pa)')
    fig.colorbar(eNpc, ax=axs[3], label=r'$\Delta$N (Pa)')
    fig.savefig(f'figures/compare_future_{basin}_{future}.png', dpi=400)

if __name__=='__main__':
    main('G-H', 2050)
    main('B-C', 2300)
    main('C-Cp', 2300)
    main('Cp-D', 2300)

    
