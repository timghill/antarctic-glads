import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import griddata

colors = ['#89b6bc', '#0d7d87', '#ff5a5e', '#c31e23']

def main(basin):
    nrun = 100
    ss,xx,yy = np.load('../data/geom/flowline_00.npy')
    mesh = np.load('../data/geom/mesh.npy', allow_pickle=True)
    meshxy = (mesh['x'], mesh['y'])
    levelset = np.load('../data/geom/ocean_levelset.npy')
    levelset_interp = griddata(meshxy, levelset, (xx,yy), method='nearest')
    
    uu_glads = np.load('solutions/u_glads_para_sensitivity.npy')[:, :nrun]
    uu_glads_flowline = griddata(meshxy, uu_glads, (xx,yy), method='linear')

    uu_rf = np.load('solutions/u_rf_para_sensitivity.npy')[:, :nrun]
    uu_rf_flowline = griddata(meshxy, uu_rf, (xx,yy), method='linear')

    print('glads:', uu_glads_flowline.shape)

    # print('rf:', uu_rf_flowline.shape, uu_rf_flowline)

    # uu_const_C = np.load('solutions/u_glads_para_sensitivity_const_C.npy')[:, :nrun]
    # uu_const_C_flowline = griddata(meshxy, uu_const_C, (xx,yy), method='linear')

    uu_ref = np.load('solutions/u_glads_future.npy')
    uu_ref_flowline = griddata(meshxy, uu_ref, (xx,yy), method='linear')

    uu_ref_rf = np.load('solutions/u_rf_future.npy')
    uu_ref_rf_flowline = griddata(meshxy, uu_ref_rf, (xx,yy), method='linear')

    u_poc = griddata(meshxy, np.load('solutions/u_poc_future.npy'), (xx, yy), method='linear')
    u_poc_present = griddata(meshxy, np.load('solutions/u_poc_present.npy'), (xx, yy), method='linear')
    u_rf_present = griddata(meshxy, np.load('solutions/u_rf_present.npy'), (xx, yy), method='linear')
    u_glads_present = griddata(meshxy, np.load('solutions/u_glads_present.npy'), (xx, yy), method='linear')


    retreat_mask = levelset_interp>0


    u0_glads = uu_glads_flowline[retreat_mask][0]
    u0_rf = uu_rf_flowline[retreat_mask][0]

    print('Quantiles (5/16/50/84/95)')
    print(np.quantile(u0_glads, (0.05, 0.15, 0.5, 0.84, 0.95)))
    print(np.quantile(u0_rf, (0.05, 0.15, 0.5, 0.84, 0.95)))

    print('Grounding line speed')
    print('glads:', uu_ref_flowline[retreat_mask][0])
    print('rf:', uu_ref_rf_flowline[retreat_mask][0])

    fig,axs = plt.subplots(ncols=2, nrows=1, sharey=True, figsize=(6, 4), width_ratios=(100,100))

    ax = axs[0]
    ax.plot(ss[retreat_mask]/1e3, uu_glads_flowline[retreat_mask],
        color=colors[1], alpha=0.2, label='GlaDS')
    ax.plot(ss[retreat_mask]/1e3, uu_ref_flowline[retreat_mask],
        color='k', label='GlaDS mean')
    ax.plot(ss[retreat_mask]/1e3, uu_ref_rf_flowline[retreat_mask],
        color='k', linestyle='dashed', label='RF mean')
    ax.plot(ss[retreat_mask]/1e3, u_poc[retreat_mask],
        color='dimgray', linestyle='solid', label='POC future')
    ax.plot(ss[retreat_mask]/1e3, u_poc_present[retreat_mask],
        color='gray', linestyle='dashed', label='POC present')
    ax.plot(ss[retreat_mask]/1e3, u_glads_present[retreat_mask],
        color=colors[0], linestyle='solid')
    ax.plot(ss[retreat_mask]/1e3, u_rf_present[retreat_mask],
        color=colors[2], linestyle='solid')
    ax.grid()
    # ax.set_xlabel('Distance from present-day grounding line')
    ax.set_ylabel('Speed (m/year)')
    ax.text(0.025, 1.025, 'a', transform=ax.transAxes,
        fontweight='bold', va='bottom')
    ax.set_xlim([200, 0])
    ax.set_clip_on(False)
    # ax.yaxis.tick_right()
    # ax.yaxis.set_label_position('right')

    ax = axs[1]
    ax.plot(ss[retreat_mask]/1e3, uu_rf_flowline[retreat_mask],
        color=colors[3], alpha=0.2)
    ax.plot(ss[retreat_mask]/1e3, uu_ref_flowline[retreat_mask],
        color='k')
    ax.plot(ss[retreat_mask]/1e3, uu_ref_rf_flowline[retreat_mask],
        color='k', linestyle='dashed')
    ax.plot(ss[retreat_mask]/1e3, u_poc[retreat_mask],
        color='dimgray', linestyle='solid', label='POC future')
    ax.plot(ss[retreat_mask]/1e3, u_poc_present[retreat_mask],
        color='gray', linestyle='dashed', label='POC present')
    ax.plot(ss[retreat_mask]/1e3, u_glads_present[retreat_mask],
        color=colors[0], linestyle='solid')
    ax.plot(ss[retreat_mask]/1e3, u_rf_present[retreat_mask],
        color=colors[2], linestyle='solid')
    ax.grid()
    # ax.set_xlabel('Distance from present-day grounding line')
    # ax.set_ylabel('Speed (m/year)')
    ax.text(0.025, 1.025, 'b', transform=ax.transAxes,
        fontweight='bold', va='bottom')
    ax.set_xlim([200, 0])
    ax.plot([-10, -10], np.quantile(u0_glads, (0.025, 0.975)), color=colors[1], linewidth=1.5, clip_on=False)
    ax.plot([-10, -10], np.quantile(u0_glads, (0.16, 0.84)), color=colors[1], linewidth=3, clip_on=False)
    ax.plot([-20, -20], np.quantile(u0_rf, (0.025, 0.975)), color=colors[3], linewidth=1.5, clip_on=False)
    ax.plot([-20, -20], np.quantile(u0_rf, (0.16, 0.84)), color=colors[3], linewidth=3, clip_on=False)
    ax.plot(-15, u_poc[retreat_mask][0], marker='s', markerfacecolor='dimgray', markeredgecolor='k', clip_on=False)
    ax.plot(-15, u_poc_present[retreat_mask][0], marker='s', color='gray', markeredgecolor='k', clip_on=False)
    ax.plot(-20, u_rf_present[retreat_mask][0], marker='s', color=colors[2], markeredgecolor='k', clip_on=False)
    ax.plot(-10, u_glads_present[retreat_mask][0], marker='s', color=colors[0], markeredgecolor='k', clip_on=False)
    ax.plot(-10, uu_ref_flowline[retreat_mask][0], marker='s', markerfacecolor=colors[1], markeredgecolor='k', clip_on=False)
    ax.plot(-20, uu_ref_rf_flowline[retreat_mask][0], marker='s', markerfacecolor=colors[3], markeredgecolor='k', clip_on=False)
    ax.set_clip_on(False)
    
    fig.text(0.5, 0.02, 'Distance from grounding line (km)', ha='center', va='bottom')

    fig.subplots_adjust(left=0.12, right=0.925, bottom=0.12, top=0.925, wspace=0.05,
        hspace=0.35)
    fig.savefig('solutions/para_sensitivity.png', dpi=400)


    theta = np.loadtxt('../../theta_physical.csv', delimiter=',', skiprows=1)
    print(theta.shape)
    names = np.loadtxt('../../theta_physical.csv', delimiter=',', max_rows=1, dtype=str)
    print(names.shape)
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(8, 6), sharey=True)
    for i in range(len(names)):
        ax = axs.flat[i]
        ax.scatter(theta[:,i], u0_glads)
        ax.set_xlabel(names[i])
        ax.grid()
    
    axs.flat[-1].set_visible(False)
    # for ax in axs.flat[i:]:
    #     ax.set_visible(False)
    for ax in axs[:,0]:
        ax.set_ylabel('Grounding line speed (m/year)')
    fig.tight_layout()
    fig.savefig('solutions/para_scatter.png', dpi=400)


    print('Interpolating effective pressure profiles')
    N_glads_future = np.zeros((len(levelset), 100))
    N_glads_future[levelset>0] = np.load(f'../../../analysis/parameters_full/data/pred_{basin}_{year}_N_glads.npy')/1e6
    N_glads_present =  np.zeros((len(levelset), 100))
    present_levelset = np.load(f'../../{basin}/data/geom/ocean_levelset.npy')
    N_glads_present[present_levelset>0] = np.load(f'../../../analysis/parameters_full/data/pred_{basin}_N_glads.npy')/1e6
    asort = np.argsort(u0_glads)
    slow_index = asort[5]
    fast_index = asort[95]

    dH = np.load('../data/geom/thick.npy') - np.load(f'../../{basin}/data/geom/thick.npy')
    dh_flowline  =griddata(meshxy, dH, (xx,yy))
    dN = 917*9.81*dh_flowline

    fig,ax = plt.subplots()
    N_slow_future = griddata(meshxy, N_glads_future[:,slow_index], (xx,yy))
    N_slow_present = griddata(meshxy, N_glads_present[:,slow_index], (xx,yy))
    N_fast_future = griddata(meshxy, N_glads_future[:,fast_index], (xx,yy))
    N_fast_present = griddata(meshxy, N_glads_present[:,fast_index], (xx,yy))

    print('Plotting effective pressure')
    ax.plot(ss[:]/1e3, N_slow_present[:], color='cornflowerblue', label='Slow present')
    ax.plot(ss[retreat_mask]/1e3, N_slow_future[retreat_mask], color='mediumblue', label='Slow future')

    ax.plot(ss[:]/1e3, N_fast_present[:], color='lightcoral', label='Fast present')
    ax.plot(ss[retreat_mask]/1e3, N_fast_future[retreat_mask], color='firebrick', label='Fast future')
    ax.plot(ss[retreat_mask]/1e3, N_fast_present[retreat_mask] + dN[retreat_mask]/1e6, color='gray',
        label=r'Fast future + $\rho_{\rm{i}} g \Delta H$')
    ax.legend(bbox_to_anchor=(0,1,1,0.3), ncols=3, loc='lower center')
    ax.grid()
    ax.set_xlabel('Distance from present grounding line (km)')
    ax.set_ylabel('N (MPa)')
    fig.savefig('solutions/para_N_profiles.png', dpi=400)

if __name__=='__main__':
    main()
