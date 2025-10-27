import numpy as np

from matplotlib import pyplot as plt
from matplotlib.tri import Triangulation

import cmocean

def main():
    mesh = np.load('../data/geom/mesh.npy', allow_pickle=True)
    mtri = Triangulation(mesh['x'], mesh['y'], mesh['elements']-1)
    levelset = np.load('../data/geom/ocean_levelset.npy')

    vxobs = np.load('../data/geom/vx.npy')
    vyobs = np.load('../data/geom/vy.npy')
    vvobs = np.sqrt(vxobs**2 + vyobs**2)
    vvobs[vvobs<0.1] = np.nan

    fig, axs = plt.subplots(ncols=4, nrows=3, figsize=(16, 12), sharex=True, sharey=True)

    C_glads = np.load('solutions/friction_coefficient_glads_nonlinear.npy').squeeze()
    C_RF = np.load('solutions/friction_coefficient_RF_nonlinear.npy').squeeze()

    u_glads_glads = np.load('solutions/u_glads_glads_nonlinear.npy').squeeze()
    u_rf_rf = np.load('solutions/u_rf_rf_nonlinear.npy').squeeze()
    u_glads_rf = np.load('solutions/u_glads_rf_nonlinear.npy').squeeze()
    u_rf_glads = np.load('solutions/u_rf_glads_nonlinear.npy').squeeze()

    u_glads_glads[levelset<1] = np.nan
    u_rf_rf[levelset<1] = np.nan
    u_glads_rf[levelset<1] = np.nan
    u_rf_glads[levelset<1] = np.nan

    u_mean = 0.5*(u_glads_glads + u_rf_rf)
    mask = vvobs>0.1

    R2_glads_rf = 1 - np.nanvar(u_glads_rf[mask] - u_glads_glads[mask])/np.nanvar(u_glads_glads[mask])
    # R2_rf_rf = 1 - np.nanvar(u_rf_rf[mask] - u_glads_glads[mask])/np.nanvar(u_glads_glads[mask])
    R2_rf_glads = 1 - np.nanvar(u_rf_glads[mask] - u_rf_rf[mask])/np.nanvar(u_rf_rf[mask])

    obs_R2_glads_glads = 1 - np.nanvar(u_glads_glads[mask] - vvobs[mask])/np.nanvar(vvobs[mask])
    obs_R2_glads_rf = 1 - np.nanvar(u_glads_rf[mask] - vvobs[mask])/np.nanvar(vvobs[mask])
    obs_R2_rf_rf = 1 - np.nanvar(u_rf_rf[mask] - vvobs[mask])/np.nanvar(vvobs[mask])
    obs_R2_rf_glads = 1 - np.nanvar(u_rf_glads[mask] - vvobs[mask])/np.nanvar(vvobs[mask])

    print('R2 vs observations')
    print('glads-glads:', obs_R2_glads_glads)
    print('glads-RF:', obs_R2_glads_rf)
    print('RF-RF:', obs_R2_rf_rf)
    print('RF-glads:', obs_R2_rf_glads)

    Cmax = 500
    mpb = axs[0, 0].tripcolor(mtri, C_glads, vmin=0, vmax=Cmax)
    axs[0, 2].tripcolor(mtri, C_RF, vmin=0, vmax=Cmax)
    fig.colorbar(mpb, ax=axs[0], label='Friction coefficient')
    axs[0,0].set_title('Friction coefficient: GlaDS N')
    axs[0,2].set_title('Friction coefficient: RF N')


    vscale = 200
    mpb = axs[1, 0].tripcolor(mtri, np.log10(u_glads_glads), vmin=0, vmax=3, cmap=cmocean.cm.speed)
    axs[1,0].set_title('Speed: GlaDS C, GlaDS N')
    axs[1, 1].tripcolor(mtri, u_glads_rf - u_glads_glads, vmin=-vscale, vmax=vscale, cmap=cmocean.cm.balance)
    axs[1,1].set_title('RF N, $R^2$ = {:.3f}'.format(R2_glads_rf))

    axs[1, 2].tripcolor(mtri, np.log10(u_rf_rf), vmin=0, vmax=3, cmap=cmocean.cm.speed)
    axs[1,2].set_title('Speed: RF C, RF N')
    axs[1,3].tripcolor(mtri, u_rf_glads - u_rf_rf, vmin=-vscale, vmax=vscale, cmap=cmocean.cm.balance)
    axs[1,3].set_title('GlaDS N, $R^2$ = {:.3f}'.format(R2_rf_glads))
    fig.colorbar(mpb, ax=axs[1], label='log$_{10}$ speed')



    mpb = axs[2,0].tripcolor(mtri, u_glads_glads - vvobs, vmin=-vscale, vmax=vscale, cmap=cmocean.cm.balance)
    axs[2, 0].set_title(r'$R^2$ = {:.3f}'.format(obs_R2_glads_glads))

    axs[2, 1].tripcolor(mtri, u_glads_rf - vvobs, vmin=-vscale, vmax=vscale, cmap=cmocean.cm.balance)
    axs[2, 1].set_title(r'$R^2$ = {:.3f}'.format(obs_R2_glads_rf))

    axs[2, 2].tripcolor(mtri, u_rf_rf - vvobs, vmin=-vscale, vmax=vscale, cmap=cmocean.cm.balance)
    axs[2, 2].set_title(r'$R^2$ = {:.3f}'.format(obs_R2_rf_rf))

    axs[2, 3].tripcolor(mtri, u_rf_glads - vvobs, vmin=-vscale, vmax=vscale, cmap=cmocean.cm.balance)
    axs[2, 3].set_title(r'$R^2$ = {:.3f}'.format(obs_R2_rf_glads))

    fig.colorbar(mpb, ax=axs[2], label='Speed difference (m/yr)')

    for ax in axs.flat:
        ax.set_aspect('equal')

    axs[0,1].set_visible(False)
    axs[0,3].set_visible(False)

    fig.savefig('compare_speed_nonlinear.png', dpi=400)

if __name__=='__main__':
    main()