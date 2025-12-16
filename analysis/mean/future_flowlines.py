import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate

basins = ['G-H', 'G-H', 'C-Cp', 'B-C', 'Cp-D']
future = 2050
template = '{}_{:d}'
linenumbers = [0, 1, 0, 0, 0]
futureruns = [0, 1, 2, 3, 4]

labels = ['Thwaites', 'PIG', 'Denman', 'Lambert', 'Totten']
alphabet = ['(a)', '(b)', '(c)']

# # colors = ['#89b6bc', '#0d7d87', '#ff5a5e', '#c31e23']

# colors = ['#89b6bc', '#0d7d87', '#ff7966', '#af1e23']
colors = ['#97B4FF', '#5F45D8', '#FFB000', '#FE6100', 'gray', 'dimgray'] # IBM palette



fig, axs = plt.subplots(figsize=(7, 5.8), ncols=3, nrows=5, sharex=True)
fs = 7
N = len(basins)
for p in range(N):
    if p<2:
        future = 2050
    else:
        future = 2300
    basin = template.format(basins[p], future)
    present = basins[p]

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
    # f_glads_present = np.load(f'data/pred_{present}_f_glads.npy').mean(axis=1)
    
    is_iceflow = False
    try:
        dirname = f'{present}_para_{future}'
        u_poc_present = np.load(f'../../issm/{dirname}/issm/solutions/u_poc_present.npy').squeeze()
        u_glads_present = np.load(f'../../issm/{dirname}/issm/solutions/u_glads_present.npy').squeeze()
        u_rf_present = np.load(f'../../issm/{dirname}/issm/solutions/u_rf_present.npy').squeeze()
        u_cv_present = np.load(f'../../issm/{dirname}/issm/solutions/u_cv_present.npy').squeeze()
        u_poc_future = np.load(f'../../issm/{dirname}/issm/solutions/u_poc_future.npy').squeeze()
        u_rf_future = np.load(f'../../issm/{dirname}/issm/solutions/u_rf_future.npy').squeeze()
        # uu_rf_future = np.load(f'../../issm/{dirname}/issm/solutions/u_rf_para_sensitivity.npy')
        if p in futureruns:
            u_glads_future = np.load(f'../../issm/{dirname}/issm/solutions/u_glads_future.npy').squeeze()
            # uu_glads_future = np.load(f'../../issm/{dirname}/issm/solutions/u_glads_para_sensitivity.npy')
        is_iceflow = True
    except Exception as e:
        print(e)
        is_iceflow = False

    mesh = np.load(f'../../issm/{basin}/data/geom/mesh.npy', allow_pickle=True)
    levelset = np.load(f'../../issm/{present}/data/geom/ocean_levelset.npy')
    levelfut = np.load(f'../../issm/{basin}/data/geom/ocean_levelset.npy')

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

    def diff_quantile(f1,f2,q):
        delta = np.zeros((len(levelfut), 100), dtype=np.float32)
        print('delta:', delta.shape)
        y1 = np.zeros(delta.shape)
        y2 = np.zeros(delta.shape)
        y1[levelfut>0,:] = np.load(f1)
        y2[levelset>0,:] = np.load(f2)
        delta = y1 - y2
        dq = np.quantile(delta, q, axis=1)
        return dq

    if p in futureruns:
        f_glads_future = np.nanmean(np.load(f'../../issm/{basin}/glads/ff.npy'), axis=1)
        f_interp_glads_future = interpolate.griddata((mesh['x'][levelfut>0], mesh['y'][levelfut>0]), 
            f_glads_future[levelfut>0], (xx, yy), method='nearest')
        N_interp_glads_future = interpolate.griddata((mesh['x'][levelfut>0], mesh['y'][levelfut>0]), 
            np.load(f'data/pred_{basin}_N_glads.npy'), (xx, yy), method='nearest')

        # levels = (0.5 - 0.68/2, 0.5 + 0.68/2)
        # dN_glads_qntl = diff_quantile(f'data/pred_{basin}_N_glads.npy', 
        #     f'data/pred_{present}_N_glads.npy', levels)
        # dN_rf_qntl = diff_quantile(f'data/pred_{basin}_N_rf.npy',
        #     f'data/pred_{present}_N_rf.npy', levels)
        # df_glads_qntl = diff_quantile(f'data/pred_{basin}_f_glads.npy', 
        #     f'data/pred_{present}_f_glads.npy', levels)
        # df_rf_qntl = diff_quantile(f'data/pred_{basin}_f_rf.npy', 
        #     f'data/pred_{present}_f_rf.npy', levels)

        # xy = (mesh['x'], mesh['y'])
        # dN_glads_qntl = interpolate.griddata(xy, dN_glads_qntl.T, (xx,yy), method='nearest').T
        # dN_rf_qntl = interpolate.griddata(xy, dN_rf_qntl.T, (xx,yy), method='nearest').T
        # df_glads_qntl = interpolate.griddata(xy, df_glads_qntl.T, (xx,yy), method='nearest').T
        # df_rf_qntl = interpolate.griddata(xy, df_rf_qntl.T, (xx,yy), method='nearest').T
        
    if is_iceflow:
        uinterp = lambda z: interpolate.griddata((mesh['x'], mesh['y']), z, (xx, yy), method='linear')
        u_interp_glads_present = uinterp(u_glads_present)
        u_interp_rf_present = uinterp(u_rf_present)
        u_interp_cv_present = uinterp(u_cv_present)
        u_interp_rf_future = uinterp(u_rf_future)
        u_interp_poc_present = uinterp(u_poc_present)
        u_interp_poc_future = uinterp(u_poc_future)

        # uu_interp_rf_future = uinterp(uu_rf_future)[retreat_mask][0]
        print('GlaDS-present:', u_interp_glads_present[retreat_mask][0])
        print('RF-present   :', u_interp_rf_present[retreat_mask][0])
        print('POC-present  :', u_interp_poc_present[retreat_mask][0])
        if p in futureruns:
            u_interp_glads_future = uinterp(u_glads_future)
            # uu_interp_glads_future = uinterp(uu_glads_future)[retreat_mask][0]

            gl_glads = u_interp_glads_future[retreat_mask][0]
            print('GlaDS-future:', gl_glads)
        gl_rf = u_interp_rf_future[retreat_mask][0]
        print('RF-future   :', gl_rf)
        print('POC-future  :', u_interp_poc_future[retreat_mask][0])
            

    ax = axs[p,0]
    # ax.plot(ss/1e3, f_interp_glads_present, color=colors[0], 
    #     label='GlaDS-present')
    if p in futureruns:
        # ax.fill_between(ss[retreat_mask]/1e3,
        #     df_glads_qntl[0,retreat_mask], df_glads_qntl[1,retreat_mask], color=colors[1],
        #     alpha=0.25)
        ax.plot(ss[retreat_mask]/1e3, 
            (f_interp_glads_future - f_interp_glads_present)[retreat_mask], 
            color=colors[1], label='GlaDS future - present', linestyle='solid')
    # ax.plot(ss/1e3, f_interp_RF_present, color=colors[2], 
    #     label='RF present')
    # ax.fill_between(ss[retreat_mask]/1e3,
    #     df_rf_qntl[0,retreat_mask], df_rf_qntl[1,retreat_mask], color=colors[3],
    #     alpha=0.25)
    ax.plot(ss[retreat_mask]/1e3, 
        (f_interp_RF - f_interp_RF_present)[retreat_mask], 
        color=colors[3], label='RF future - present', linestyle='solid')
    ax.set_ylim([-0.15, 0.185])
    ax.grid()
    ax.set_ylabel(r'$\Delta f_{\rm{w}}$ (-)', fontsize=fs, labelpad=0)
    # ax1.set_title(labels[p])
    # axs[p,0].set_ylabel(labels[p])
    axs[p,2].text(0.15, 0.95, '{}, {}'.format(labels[p], future), ha='left', va='top',
        transform=axs[p,2].transAxes, fontweight='bold', fontsize=fs)


    ax = axs[p,1]
    # ax.plot(ss/1e3, N_interp_glads_present/1e6, color=colors[0], label='GlaDS present')
    if p in futureruns:
        # ax.fill_between(ss[retreat_mask]/1e3,
        #     dN_glads_qntl[0,retreat_mask]/1e6, dN_glads_qntl[1,retreat_mask]/1e6, color=colors[1],
        #     alpha=0.25)
        ax.plot(ss[retreat_mask]/1e3, 
            (N_interp_glads_future - N_interp_glads_present)[retreat_mask]/1e6, 
            color=colors[1], label='GlaDS future')
    # ax.plot(ss/1e3, N_interp_RF_present/1e6, color=colors[2], 
    #     label='RF present')
    # ax.fill_between(ss[retreat_mask]/1e3,
    #     dN_rf_qntl[0,retreat_mask]/1e6, dN_rf_qntl[1,retreat_mask]/1e6, color=colors[3],
    #     alpha=0.25)
    ax.plot(ss[retreat_mask]/1e3, 
        (N_interp_RF - N_interp_RF_present)[retreat_mask]/1e6, 
        color=colors[3], label='RF future')
    ax.set_ylim([-2.5, 1])
    # ax.legend()
    ax.grid()
    ax.set_ylabel(r'$\Delta N$ (MPa)', fontsize=fs, labelpad=0)
        # ax.set_title(labels[p])


    if is_iceflow:
        alpha = 0.95
        bold = 1.5
        normal = 1
        # for ax in [ax1, axs[p,2]]:
        ax = axs[p,2]
        ax.plot(ss[retreat_mask]/1e3, u_interp_poc_present[retreat_mask], label=r'$N_{\rm{POC}}$ present', 
            color='gray', linestyle='dashed', alpha=alpha, linewidth=normal)
        ax.plot(ss[retreat_mask]/1e3, u_interp_rf_present[retreat_mask], label=r'$N_{\rm{RF}}$ present', 
            color=colors[2], linestyle='solid', alpha=alpha, linewidth=normal)
        ax.plot(ss[retreat_mask]/1e3, u_interp_glads_present[retreat_mask], label=r'$N_{\rm{GlaDS}}$ present', 
            color=colors[0], linestyle='solid', alpha=alpha, linewidth=normal)
        ax.plot(ss[retreat_mask]/1e3, u_interp_poc_future[retreat_mask], label=r'$N_{\rm{POC}}$ future', 
            color='dimgray', linestyle='dashed', alpha=alpha, linewidth=bold)
        ax.plot(ss[retreat_mask]/1e3, u_interp_rf_future[retreat_mask], label=r'$N_{\rm{RF}}$ future', 
            color=colors[3], linestyle='solid', alpha=alpha, linewidth=bold)
        if p in futureruns:
            ax.plot(ss[retreat_mask]/1e3, u_interp_glads_future[retreat_mask], label=r'$N_{\rm{GlaDS}}$ future', 
                color=colors[1], linestyle='solid', alpha=alpha, linewidth=bold)
            # ax.set_ylim([0.75, 1])
            # ax.legend()
            ax.grid()
            ax.set_ylabel('Speed (m a$^{-1}$)', fontsize=fs, labelpad=0)
            # ax.set_title(labels[p])
            ax.set_ylim(bottom=0)

            # Plot uncertainty
            # ms = 5
            # ax.plot([-5, -5], np.quantile(uu_interp_glads_future, (0.025, 0.975)), 
            #     color=colors[1], linewidth=1.5, clip_on=False)
            # ax.plot([-5, -5], np.quantile(uu_interp_glads_future, (0.16, 0.84)), 
            #     color=colors[1], linewidth=2.5, clip_on=False)
            # ax.plot([-20, -20], np.quantile(uu_interp_rf_future, (0.025, 0.975)), 
            #     color=colors[3], linewidth=1.5, clip_on=False)
            # ax.plot([-20, -20], np.quantile(uu_interp_rf_future, (0.16, 0.84)),
            #     color=colors[3], linewidth=2.5, clip_on=False)
            # ax.plot(-12.5, u_interp_poc_future[retreat_mask][0], 
            #     marker='s', markerfacecolor='dimgray', markeredgecolor='k',
            #     clip_on=False, markersize=ms)
            # ax.plot(-12.5, u_interp_poc_present[retreat_mask][0], 
            #     marker='s', color='gray', markeredgecolor='k', 
            #     clip_on=False, markersize=ms)
            # ax.plot(-20, u_interp_rf_present[retreat_mask][0], 
            #     marker='s', color=colors[2], markeredgecolor='k', 
            #     clip_on=False, markersize=ms)
            # ax.plot(-5, u_interp_glads_present[retreat_mask][0], 
            #     marker='s', color=colors[0], markeredgecolor='k', 
            #     clip_on=False, markersize=ms)
            # ax.plot(-5, u_interp_glads_future[retreat_mask][0], 
            #     marker='s', markerfacecolor=colors[1], markeredgecolor='k', 
            #     clip_on=False, markersize=ms)
            # ax.plot(-20, u_interp_rf_future[retreat_mask][0], 
            #     marker='s', markerfacecolor=colors[3], markeredgecolor='k', 
            #     clip_on=False, markersize=ms)


    else:
        axs[p,2].set_visible(False)
    
    for ax in axs.flat:
        ax.set_xlim([100, 0])
        ax.tick_params(labelsize=fs)
    
    for i,ax in enumerate(axs[0]):
        ax.text(0.025, 0.95, alphabet[i], transform=ax.transAxes,
            fontweight='bold', fontsize=fs,
            ha='left', va='top')


    
    fig.subplots_adjust(left=0.08, right=0.925, bottom=0.075, top=0.89, wspace=0.3, hspace=0.125)

axs[0,2].legend(bbox_to_anchor=(0, 1, 1., 1.0), loc='lower center', 
    frameon=False, ncols=2, fontsize=fs)
# for ax in axs[-1]:
axs[-1,1].set_xlabel('Distance from present grounding line (km)', fontsize=fs)
axs[0,0].legend(bbox_to_anchor=(0,1,1,0.2), loc='lower left', 
    frameon=False, ncols=3, fontsize=fs)
fig.savefig('figures/future_profiles.png', dpi=400)
fig.savefig('../../manuscript/f07.png', dpi=400)
fig.savefig('../../manuscript/f07.pdf')
