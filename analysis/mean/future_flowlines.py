import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate

basins = ['G-H', 'G-H', 'C-Cp', 'B-C', 'Cp-D']
future = 2050
template = '{}_{:d}'
linenumbers = [0, 1, 0, 0, 0]
futureruns = [0, 1, 2,3,4]

labels = ['Thwaites', 'PIG', 'Denman', 'Lambert', 'Totten']

colors = ['#89b6bc', '#0d7d87', '#ff5a5e', '#c31e23']

fig2, axs = plt.subplots(figsize=(12, 10), ncols=3, nrows=5, sharex=True)

N = len(basins)
for p in range(N):
# for p in range(2):
# for p in range(2):
# for p in [2, 3]:
# for p in [4]:
    if p<2:
        future = 2050
    else:
        future = 2300
    basin = template.format(basins[p], future)
    present = basins[p]


        # ax.plot(ss/1e3, N_interp_glads/1e6, color='black', label='GlaDS-present')
        # ax.plot(ss/1e3, N_interp_RF_present/1e6, color='red', label='RF (train/present)', linestyle='dashed')
        # ax.plot(ss/1e3, N_interp_CV_present/1e6, color='lightcoral', label='RF (CV/present)', linestyle='dashed')
        # ax.plot(ss/1e3, N_interp_RF/1e6, color='red', label='RF (train/future)', linestyle='dotted')

    N_RF = np.load(f'data/pred_{basin}_N_rf.npy')
    N_RF_present = np.load(f'data/pred_{present}_N_rf.npy')
    N_glads_present = np.load(f'data/pred_{present}_N_glads.npy')
    print(N_glads_present.shape)
    N_CV_present = np.load(f'data/CV_{present}_N_rf.npy')
    print(N_CV_present.shape)

    f_RF = np.load(f'data/pred_{basin}.npy')
    f_RF_present = np.load(f'data/pred_{present}.npy')
    f_CV_present = np.load(f'data/CV_{present}.npy')
    f_glads_present = np.nanmean(np.load(f'../../issm/{present}/glads/ff.npy'), axis=1)
    
    is_iceflow = False
    try:
        # C_glads = np.load(f'../../issm/{basin}/issm/solutions/friction_coefficient_glads_nonlinear.npy').squeeze()
        # C_RF = np.load(f'../../issm/{basin}/issm/solutions/friction_coefficient_RF_nonlinear.npy').squeeze()

        u_poc_present = np.load(f'../../issm/{basin}/issm/solutions/u_poc_present.npy').squeeze()
        u_glads_present = np.load(f'../../issm/{basin}/issm/solutions/u_glads_present.npy').squeeze()
        u_rf_present = np.load(f'../../issm/{basin}/issm/solutions/u_rf_present.npy').squeeze()
        u_cv_present = np.load(f'../../issm/{basin}/issm/solutions/u_cv_present.npy').squeeze()
        u_poc_future = np.load(f'../../issm/{basin}/issm/solutions/u_poc_future.npy').squeeze()
        u_rf_future = np.load(f'../../issm/{basin}/issm/solutions/u_rf_future.npy').squeeze()
        if p in futureruns:
            u_glads_future = np.load(f'../../issm/{basin}/issm/solutions/u_glads_future.npy').squeeze()

        # u_glads_glads_present = np.load(f'../../issm/{basin}/issm/solutions/u_glads_glads_nonlinear.npy').squeeze()
        # u_rf_rf_present = np.load(f'../../issm/{basin}/issm/solutions/u_rf_rf_nonlinear.npy').squeeze()
        # u_glads_rf_present = np.load(f'../../issm/{basin}/issm/solutions/u_glads_rf_nonlinear.npy').squeeze()
        # u_glads_cv = np.load(f'../../issm/{basin}/issm/solutions/u_glads_cv_nonlinear.npy').squeeze()
        # u_rf_glads = np.load(f'../../issm/{basin}/issm/solutions/u_rf_glads_nonlinear.npy').squeeze()
        # u_glads_poc = np.load(f'../../issm/{basin}/issm/solutions/u_glads_poc_nonlinear.npy').squeeze()
        # u_rf_poc = np.load(f'../../issm/{basin}/issm/solutions/u_rf_poc_nonlinear.npy').squeeze()
        # u_poc = np.load(f'../../issm/{basin}/issm/solutions/u_poc_nonlinear.npy').squeeze()

        # vx = np.load(f'../../issm/{basin}/data/geom/vx.npy')
        # vy = np.load(f'../../issm/{basin}/data/geom/vy.npy')
        # vv = np.sqrt(vx**2 + vy**2)
        # vv[vv<0.1] = -999
        is_iceflow = True
    except:
        is_iceflow = False

    mesh = np.load(f'../../issm/{basin}/data/geom/mesh.npy', allow_pickle=True)
    levelset = np.load(f'../../issm/{present}/data/geom/ocean_levelset.npy')
    levelfut = np.load(f'../../issm/{basin}/data/geom/ocean_levelset.npy')

    # thickfut = np.load(f'../../issm/{basin}/data/geom/ocean_levelset.npy')
    # f_glads = f_glads[levelset>0]


    interp = lambda z: interpolate.griddata((mesh['x'][levelset>0], mesh['y'][levelset>0]), z, (xx, yy), method='linear')
    flowline = np.load('../../issm/{}/data/geom/flowline_{:02d}.npy'.format(basin,linenumbers[p]))
    ss,xx,yy = flowline
    N_interp_glads_present = interp(N_glads_present)
    N_interp_RF_present = interp(N_RF_present)
    N_interp_CV_present = interp(N_CV_present)
    N_interp_RF = interpolate.griddata((mesh['x'][levelfut>0], mesh['y'][levelfut>0]), N_RF, (xx, yy), method='linear')
    levelset_interp = interpolate.griddata((mesh['x'], mesh['y']), levelfut, (xx, yy), method='nearest')
    retreat_mask = levelset_interp>0

    f_interp_glads_present = interp(f_glads_present[levelset>0])
    f_interp_RF_present = interp(f_RF_present)
    f_interp_CV_present = interp(f_CV_present)
    f_interp_RF = interpolate.griddata((mesh['x'][levelfut>0], mesh['y'][levelfut>0]), f_RF, (xx, yy), method='linear')

    if p in futureruns:
        f_glads_future = np.nanmean(np.load(f'../../issm/{basin}/glads/ff.npy'), axis=1)

        # f_interp_glads_future = interp(f_glads_future[levelfut>0])
        f_interp_glads_future = interpolate.griddata((mesh['x'][levelfut>0], mesh['y'][levelfut>0]), 
            f_glads_future[levelfut>0], (xx, yy), method='nearest')
        N_interp_glads_future = interpolate.griddata((mesh['x'][levelfut>0], mesh['y'][levelfut>0]), 
            np.load(f'data/pred_{basin}_N_glads.npy'), (xx, yy), method='nearest')
        
        # print(N_interp_glads_future)

    if is_iceflow:
        # C_interp_glads = interpolate.griddata((mesh['x'][levelset>0], mesh['y'][levelset>0]), C_glads[levelset>0], (xx, yy), method='linear')
        # C_interp_RF = interpolate.griddata((mesh['x'][levelset>0], mesh['y'][levelset>0]), C_RF[levelset>0], (xx, yy), method='linear')

        uinterp = lambda z: interpolate.griddata((mesh['x'], mesh['y']), z, (xx, yy), method='linear')
        u_interp_glads_present = uinterp(u_glads_present)
        u_interp_rf_present = uinterp(u_rf_present)
        u_interp_cv_present = uinterp(u_cv_present)
        u_interp_rf_future = uinterp(u_rf_future)
        u_interp_poc_present = uinterp(u_poc_present)
        u_interp_poc_future = uinterp(u_poc_future)
        print('GlaDS-present:', u_interp_glads_present[retreat_mask][0])
        print('RF-present   :', u_interp_rf_present[retreat_mask][0])
        print('POC-present  :', u_interp_poc_present[retreat_mask][0])
        if p in futureruns:
            u_interp_glads_future = uinterp(u_glads_future)

            gl_glads = u_interp_glads_future[retreat_mask][0]
            print('GlaDS-future:', gl_glads)
        gl_rf = u_interp_rf_future[retreat_mask][0]
        print('RF-future   :', gl_rf)
        print('POC-future  :', u_interp_poc_future[retreat_mask][0])
            
            

        # print('max:', np.max(u_interp_glads_present))

        # u_interp_glads_glads = interpolate.griddata((mesh['x'][levelset>0], mesh['y'][levelset>0]), u_glads_glads[levelset>0], (xx, yy), method='linear')
        # u_interp_rf_rf = interpolate.griddata((mesh['x'][levelset>0], mesh['y'][levelset>0]), u_rf_rf[levelset>0], (xx, yy), method='linear')
        # u_interp_glads_rf = interpolate.griddata((mesh['x'][levelset>0], mesh['y'][levelset>0]), u_glads_rf[levelset>0], (xx, yy), method='linear')
        # u_interp_glads_cv = interpolate.griddata((mesh['x'][levelset>0], mesh['y'][levelset>0]), u_glads_cv[levelset>0], (xx, yy), method='linear')
        # # u_interp_rf_glads = interpolate.griddata((mesh['x'][levelset>0], mesh['y'][levelset>0]), u_rf_glads[levelset>0], (xx, yy), method='linear')
        # u_interp_glads_poc = interpolate.griddata((mesh['x'][levelset>0], mesh['y'][levelset>0]), u_glads_poc[levelset>0], (xx, yy), method='linear')
        # # u_interp_rf_poc = interpolate.griddata((mesh['x'][levelset>0], mesh['y'][levelset>0]), u_rf_poc[levelset>0], (xx, yy), method='linear')
        # # u_interp_poc = interpolate.griddata((mesh['x'][levelset>0], mesh['y'][levelset>0]), u_poc[levelset>0], (xx, yy), method='linear')
        # vv_interp = interpolate.griddata((mesh['x'][levelset>0], mesh['y'][levelset>0]), vv[levelset>0], (xx, yy), method='linear')
        # vv_interp[vv_interp<10] = np.nan

    fig,ax1 = plt.subplots()
    for ax in [ax1, axs[p,1]]:
        ax.plot(ss/1e3, N_interp_glads_present/1e6, color=colors[0], label='GlaDS present')
        if p in futureruns:
            ax.plot(ss[retreat_mask]/1e3, N_interp_glads_future[retreat_mask]/1e6, color=colors[1], 
                label='GlaDS future')
        ax.plot(ss/1e3, N_interp_RF_present/1e6, color=colors[2], 
            label='RF present')
        # ax.plot(ss/1e3, N_interp_CV_present/1e6, color=colors[2], 
        #     label='CV present', linestyle='dashed')
        ax.plot(ss[retreat_mask]/1e3, N_interp_RF[retreat_mask]/1e6, color=colors[3], 
            label='RF future')
        ax.set_ylim([0, 4])
        # ax.legend()
        ax.grid()
        ax.set_ylabel('$N$ (MPa)')
        # ax.set_title(labels[p])
    ax1.legend()
    ax1.set_xlabel('Distance from the groundine line (km)')
    # fig.savefig(f'figures/profile_{basin}_{p:02d}_N.png', dpi=400)


    fig,ax1 = plt.subplots()
    for ax in [ax1, axs[p,0]]:
        ax.plot(ss/1e3, f_interp_glads_present, color=colors[0], 
            label='GlaDS-present')
        if p in futureruns:
            ax.plot(ss[retreat_mask]/1e3, f_interp_glads_future[retreat_mask], color=colors[1], 
                label='GlaDS future', linestyle='solid')
        ax.plot(ss/1e3, f_interp_RF_present, color=colors[2], 
            label='RF present')
        # ax.plot(ss/1e3, f_interp_CV_present, color=colors[2], 
        #     label='CV present', linestyle='dashed')
        ax.plot(ss[retreat_mask]/1e3, f_interp_RF[retreat_mask], color=colors[3], 
            label='RF future', linestyle='solid')
        ax.set_ylim([0.6, 1])
        ax.grid()
        ax.set_ylabel('Flotation fraction (-)')
    ax1.set_title(labels[p])
    # axs[p,0].set_ylabel(labels[p])
    axs[p,2].text(0.95, 0.95, '{}, {}'.format(labels[p], future), ha='right', va='top',
        transform=axs[p,2].transAxes, fontweight='bold')
    ax1.legend()
    ax1.set_xlabel('Distance from the groundine line (km)')
    # fig.savefig(f'figures/profile_{basin}_{p:02d}_f.png', dpi=400)

    if is_iceflow:
        # colors = ['gray', 'blue', 'red']
        alpha = 0.9
        fig,ax1 = plt.subplots()
        for ax in [ax1, axs[p,2]]:
            ax.plot(ss[retreat_mask]/1e3, u_interp_poc_present[retreat_mask], label='POC N present', 
                color='gray', linestyle='dashed', alpha=alpha)
            ax.plot(ss[retreat_mask]/1e3, u_interp_rf_present[retreat_mask], label='RF N present', 
                color=colors[2], linestyle='solid', alpha=alpha)
            ax.plot(ss[retreat_mask]/1e3, u_interp_glads_present[retreat_mask], label='GlaDS N present', 
                color=colors[0], linestyle='solid', alpha=alpha)
            # ax.plot(ss/1e3, u_interp_cv_present, label='CV N present', 
            #     color=colors[2], linestyle='dashed', alpha=alpha)
            ax.plot(ss[retreat_mask]/1e3, u_interp_poc_future[retreat_mask], label='POC N future', 
                color='dimgray', linestyle='dashed', alpha=alpha)
            ax.plot(ss[retreat_mask]/1e3, u_interp_rf_future[retreat_mask], label='RF N future', 
                color=colors[3], linestyle='solid', alpha=alpha)
            if p in futureruns:
                ax.plot(ss[retreat_mask]/1e3, u_interp_glads_future[retreat_mask], label='GlaDS N future', 
                    color=colors[1], linestyle='solid', alpha=alpha)
                
            # ax.set_ylim([0.75, 1])
            # ax.legend()
            ax.grid()
            ax.set_ylabel('Speed (m/year)')
            # ax.set_title(labels[p])
            ax.set_ylim(bottom=0)
        ax1.legend()
        ax1.set_xlabel('Distance from the grounding line (km)')
        fig.savefig(f'figures/profile_{basin}_{p:02d}_u_nonlinear.png', dpi=400)
    else:
        axs[p,2].set_visible(False)


    
    fig2.subplots_adjust(left=0.075, right=0.975, bottom=0.05, top=0.9, wspace=0.3, hspace=0.1)

axs[0,2].legend(bbox_to_anchor=(0, 1, 1., 1.0), loc='lower center', frameon=False, ncols=2)
for ax in axs[p]:
    ax.set_xlabel('Distance from grounding line (km)')
axs[0,0].legend(bbox_to_anchor=(0,1,1,0.2), loc='lower left', frameon=False, ncols=3)
fig2.savefig('figures/future_profiles.png', dpi=400)
