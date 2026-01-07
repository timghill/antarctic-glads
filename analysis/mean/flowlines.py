import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate

basins = ['G-H', 'G-H', 'C-Cp', 'B-C', 'Cp-D']
linenumbers = [0, 1, 0, 0, 0]
fs = 7

# colors = ['#89b6bc', '#0d7d87', '#ff5a5e', '#c31e23']
# colors = ['#89b6bc', '#0d7d87', '#FF8C74', '#af1e23']
# colors = ['#BC1E68', '#D674D5', '#177622', '#6BD823','gray', 'dimgray']
colors = ['#8CACFF', '#5F45D8', '#FFB000', '#FE6100', 'gray', 'dimgray'] # IBM palette


# #FF8C74 #FF6B84

labels = ['Thwaites', 'PIG', 'Denman', 'Lambert', 'Totten']
alphabet = ['(a)', '(b)', '(c)']

fig2, axs = plt.subplots(figsize=(7, 5.8), ncols=3, nrows=5, sharex=True)

u_rf = []
u_glads = []

N = len(basins)
for p in range(N):
    basin = basins[p]
    print(basin)

    mesh = np.load(f'../../issm/{basin}/data/geom/mesh.npy', allow_pickle=True)
    levelset = np.load(f'../../issm/{basin}/data/geom/ocean_levelset.npy')

    def load(fname, fill=0):
        z = np.load(fname)
        zpad = np.full(len(levelset), fill, dtype=np.float32)
        zpad[levelset>0] = z
        return zpad

    N_RF = load(f'data/pred_{basin}_N_rf.npy')
    N_CV = load(f'data/CV_{basin}_N_rf.npy')
    N_glads = load(f'data/pred_{basin}_N_glads.npy')

    f_RF = load(f'data/pred_{basin}.npy', fill=1)
    f_CV = load(f'data/CV_{basin}.npy', fill=1)
    f_glads = np.load(f'../../issm/{basin}/glads/ff.npy').mean(axis=1)

    print('f_RF:', f_RF, f_RF[levelset>0])

    f_glads[f_glads>1] = 1
    f_glads[f_glads<0] = 0

    try:
        C_glads = np.load(f'../../issm/{basin}/issm/solutions/friction_coefficient_glads_nonlinear.npy').squeeze()
        C_RF = np.load(f'../../issm/{basin}/issm/solutions/friction_coefficient_RF_nonlinear.npy').squeeze()

        u_glads_glads = np.load(f'../../issm/{basin}/issm/solutions/u_glads_glads_nonlinear.npy').squeeze()
        u_rf_rf = np.load(f'../../issm/{basin}/issm/solutions/u_rf_rf_nonlinear.npy').squeeze()
        u_glads_rf = np.load(f'../../issm/{basin}/issm/solutions/u_glads_rf_nonlinear.npy').squeeze()
        u_glads_cv = np.load(f'../../issm/{basin}/issm/solutions/u_glads_cv_nonlinear.npy').squeeze()
        u_rf_glads = np.load(f'../../issm/{basin}/issm/solutions/u_rf_glads_nonlinear.npy').squeeze()
        u_glads_poc = np.load(f'../../issm/{basin}/issm/solutions/u_glads_poc_nonlinear.npy').squeeze()
        u_rf_poc = np.load(f'../../issm/{basin}/issm/solutions/u_rf_poc_nonlinear.npy').squeeze()
        u_poc = np.load(f'../../issm/{basin}/issm/solutions/u_poc_nonlinear.npy').squeeze()

        u_glads.extend(u_glads_glads.squeeze())
        u_rf.extend(u_glads_rf.squeeze())

        r2 = 1 - np.nanvar(u_glads_rf - u_glads_glads)/np.nanvar(u_glads_glads)
        print('r2:', r2)

        r2 = 1 - np.nanvar(u_glads_cv - u_glads_glads)/np.nanvar(u_glads_glads)
        print('r2:', r2)


        vx = np.load(f'../../issm/{basin}/data/geom/vx.npy')
        vy = np.load(f'../../issm/{basin}/data/geom/vy.npy')
        vv = np.sqrt(vx**2 + vy**2)
        vv[vv<0.1] = -999
        is_iceflow = True
    except:
        is_iceflow = False


    flowline = np.load('../../issm/{}/data/geom/flowline_{:02d}.npy'.format(basin,linenumbers[p]))
    ss,xx,yy = flowline
    N_interp_RF = interpolate.griddata((mesh['x'], mesh['y']), N_RF, (xx, yy), method='nearest')
    N_interp_CV = interpolate.griddata((mesh['x'], mesh['y']), N_CV, (xx, yy), method='nearest')
    N_interp_glads = interpolate.griddata((mesh['x'], mesh['y']), N_glads, (xx, yy), method='nearest')


    f_interp_RF = interpolate.griddata((mesh['x'], mesh['y']), f_RF, (xx, yy), method='nearest')
    f_interp_CV = interpolate.griddata((mesh['x'], mesh['y']), f_CV, (xx, yy), method='nearest')
    f_interp_glads = interpolate.griddata((mesh['x'], mesh['y']), f_glads, (xx, yy), method='nearest')

    print('f_interp_RF:', f_interp_RF)

    if is_iceflow:
        method = 'linear'
        C_interp_glads = interpolate.griddata((mesh['x'], mesh['y']), C_glads, (xx, yy), method='linear')
        C_interp_RF = interpolate.griddata((mesh['x'], mesh['y']), C_RF, (xx, yy), method='linear')

        u_interp_glads_glads = interpolate.griddata((mesh['x'][levelset>0], mesh['y'][levelset>0]), u_glads_glads[levelset>0], (xx, yy), method=method)
        u_interp_rf_rf = interpolate.griddata((mesh['x'][levelset>0], mesh['y'][levelset>0]), u_rf_rf[levelset>0], (xx, yy), method=method)
        u_interp_glads_rf = interpolate.griddata((mesh['x'][levelset>0], mesh['y'][levelset>0]), u_glads_rf[levelset>0], (xx, yy), method=method)
        u_interp_glads_cv = interpolate.griddata((mesh['x'][levelset>0], mesh['y'][levelset>0]), u_glads_cv[levelset>0], (xx, yy), method=method)
        # u_interp_rf_glads = interpolate.griddata((mesh['x'][levelset>0], mesh['y'][levelset>0]), u_rf_glads[levelset>0], (xx, yy), method='linear')
        u_interp_glads_poc = interpolate.griddata((mesh['x'][levelset>0], mesh['y'][levelset>0]), u_glads_poc[levelset>0], (xx, yy), method=method)
        # u_interp_rf_poc = interpolate.griddata((mesh['x'][levelset>0], mesh['y'][levelset>0]), u_rf_poc[levelset>0], (xx, yy), method='linear')
        # u_interp_poc = interpolate.griddata((mesh['x'][levelset>0], mesh['y'][levelset>0]), u_poc[levelset>0], (xx, yy), method='linear')
        vv_interp = interpolate.griddata((mesh['x'][levelset>0], mesh['y'][levelset>0]), vv[levelset>0], (xx, yy), method='nearest')
        vv_interp[vv_interp<10] = np.nan

    # fig,ax1 = plt.subplots()
    # for ax in [ax1, axs[p,1]]:
    ax = axs[p,1]
    ax.plot(ss/1e3, N_interp_glads/1e6, color=colors[1], label='GlaDS')
    ax.plot(ss/1e3, N_interp_RF/1e6, color=colors[3], label='RF', linestyle='solid')
    ax.plot(ss/1e3, N_interp_CV/1e6, color=colors[2], label='CV', linestyle='solid')
    ax.set_ylim([0, 4])
    # ax.legend()
    ax.grid()
    ax.set_ylabel('$N$ (MPa)', fontsize=fs)
    # ax.set_title(labels[p])
    # ax1.legend()
    # ax1.set_xlabel('Distance from the groundine line (km)')
    # fig.savefig(f'figures/profile_{basin}_{p:02d}_N.png', dpi=400)


    # fig,ax1 = plt.subplots()
    # for ax in [ax1, axs[p,0]]:
    ax = axs[p,0]
    ax.plot(ss/1e3, f_interp_glads, color=colors[1], label='GlaDS')
    ax.plot(ss/1e3, f_interp_RF, color=colors[3], label='RF')
    ax.plot(ss/1e3, f_interp_CV, color=colors[2], label='CV', linestyle='solid')
    ax.set_ylim([0.6, 1])
    ax.grid()
    ax.set_ylabel(r'$f_{\rm{w}}$ (-)', fontsize=fs)
    # ax1.set_title(labels[p])
    # axs[p,0].set_ylabel(labels[p])
    axs[p,2].text(0.15, 0.95, labels[p], ha='left', va='top',
        transform=axs[p,2].transAxes, fontweight='bold', fontsize=fs)
    # ax1.legend()
    # ax1.set_xlabel('Distance from the groundine line (km)')
    # fig.savefig(f'figures/profile_{basin}_{p:02d}_f.png', dpi=400)

    if is_iceflow:
        alpha = 0.98
        # fig,ax1 = plt.subplots()
        # for ax in [ax1, axs[p,2]]:
        ax = axs[p,2]
        ax.plot(ss/1e3, vv_interp, color='black', label='Observed', linewidth=1.5)
        # ax.plot(ss/1e3, u_interp_poc, label='C_poc, N_poc', color=colors[0], linestyle=linestyles[0])
        ax.plot(ss/1e3, u_interp_glads_glads, label=r'$C_{\rm{GlaDS}}$, $N_{\rm{GlaDS}}$', color=colors[1], linestyle='solid', alpha=alpha, zorder=4, linewidth=1.25)
        ax.plot(ss/1e3, u_interp_glads_rf, label=r'$C_{\rm{GlaDS}}$, $N_{\rm{RF}}$', color=colors[3], linestyle='dashed', alpha=alpha, zorder=5, linewidth=1)
        ax.plot(ss/1e3, u_interp_glads_cv, label=r'$C_{\rm{GlaDS}}$, $N_{\rm{CV}}$', color=colors[2], linestyle='dashed', alpha=alpha, zorder=5, linewidth=0.75)
        ax.plot(ss/1e3, u_interp_glads_poc, label=r'$C_{\rm{GlaDS}}$, $N_{\rm{POC}}$', color='dimgray', linestyle='dashed', alpha=alpha, zorder=5, linewidth=0.75)
        # ax.plot(ss/1e3, u_interp_rf_glads, label='C_RF, N_glads', color=colors[1], linestyle=linestyles[2])
        ax.plot(ss/1e3, u_interp_rf_rf, label=r'$C_{\rm{RF}}$, $N_{\rm{RF}}$', color=colors[3], linestyle='solid', alpha=alpha, zorder=4, linewidth=1)
        # ax.plot(ss/1e3, u_interp_rf_poc, label='C_RF, N_poc', color=colors[0], linestyle=linestyles[2])
        # ax.set_ylim([0.75, 1])
        # ax.legend()
        ax.grid()
        ax.set_ylabel('Speed (m a$^{-1}$)', fontsize=fs, labelpad=0)
        # ax.set_title(labels[p])
        ax.set_ylim(bottom=0)
        # ax1.legend()
        # ax1.set_xlabel('Distance from the grounding line (km)')
        # fig.savefig(f'figures/profile_{basin}_{p:02d}_u_nonlinear.png', dpi=400)
    else:
        axs[p,2].set_visible(False)
    
    for ax in axs.flat:
        ax.set_xlim([200, 0])
        ax.tick_params(labelsize=fs)


    for i,ax in enumerate(axs[0]):
        ax.text(0.025, 0.95, alphabet[i], transform=ax.transAxes,
            fontweight='bold', fontsize=fs,
            ha='left', va='top')


    
    fig2.subplots_adjust(left=0.08, right=0.96, bottom=0.075, top=0.89, wspace=0.3, hspace=0.125)

axs[0,2].legend(bbox_to_anchor=(0, 1, 1., 1.0), loc='lower center', frameon=False, ncols=2, fontsize=fs)
axs[-1,1].set_xlabel('Distance from grounding line (km)', fontsize=fs)
axs[0,0].legend(bbox_to_anchor=(0,1,1,0.2), loc='lower left', frameon=False, ncols=3, fontsize=fs)
fig2.savefig('figures/profiles.png', dpi=400)
fig2.savefig('../../manuscript/f06.png', dpi=400)
fig2.savefig('../../manuscript/f06.pdf')


r2 = 1 - np.nanvar(np.array(u_rf) - np.array(u_glads))/np.nanvar(np.array(u_glads))
print('OVERALL r2:', r2)
