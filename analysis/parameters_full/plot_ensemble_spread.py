import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate

basins = ['G-H', 'C-Cp', 'B-C']
# future = 2050
# template = '{}_{:d}'
linenumbers = [1, 0, 0]

labels = ['PIG', 'Denman', 'Lambert']
alphabet = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']

# colors = ['#89b6bc', '#0d7d87', '#ff5a5e', '#c31e23']
colors = ['#8CACFF', '#5F45D8', '#FFB000', '#FE6100', 'gray', 'dimgray'] # IBM palette

fig, axs = plt.subplots(figsize=(7, 4), ncols=3, nrows=2, sharex=True)
fs = 7
N = len(basins)
# for p in range(N):
# for p in [3]:
for p in range(N):
    basin = basins[p]

    # N_RF = np.load(f'data/pred_{basin}_N_rf.npy')
    N_glads = np.load(f'data/pred_{basin}_N_glads.npy')/1e6
    print(N_glads.shape)
    # N_CV = np.load(f'data/CV_{basin}_N_rf.npy')

    # f_RF = np.load(f'data/pred_{basin}_f_rf.npy')
    # f_CV = np.load(f'data/CV_{basin}_f_rf.npy')
    f_glads = np.load(f'data/pred_{basin}_f_glads.npy')
    print(f_glads.shape)

    mesh = np.load(f'../../issm/{basin}/data/geom/mesh.npy', allow_pickle=True)
    levelset = np.load(f'../../issm/{basin}/data/geom/ocean_levelset.npy')

    interp = lambda z: interpolate.griddata((mesh['x'][levelset>0], mesh['y'][levelset>0]), z, (xx, yy), method='linear')
    flowline = np.load('../../issm/{}/data/geom/flowline_{:02d}.npy'.format(basin,linenumbers[p]))
    ss,xx,yy = flowline
    N_interp_glads = interp(N_glads)
    print(N_interp_glads.shape)
    # N_interp_RF = interp(N_RF)
    # N_interp_CV = interp(N_CV)

    f_interp_glads = interp(f_glads)
    print(f_interp_glads.shape)
    # f_interp_RF = interp(f_RF)
    # f_interp_CV = interp(f_CV)

    ax = axs[0,p]
    ax.plot(ss/1e3, f_interp_glads.mean(axis=1), color=colors[1], 
        label='Mean', linewidth=1.5)
    ax.fill_between(ss/1e3, np.quantile(f_interp_glads, 0.16, axis=1),
        np.quantile(f_interp_glads, 0.86, axis=1),
        color=colors[0], alpha=0.7, label='68% interval', edgecolor='none')
    ax.fill_between(ss/1e3, np.quantile(f_interp_glads, 0.025, axis=1),
        np.quantile(f_interp_glads, 0.975, axis=1),
        color=colors[0], alpha=0.4, label='95% interval', edgecolor='none')
    ax.set_ylim([0.6, 1.1])
    ax.text(0.025, 1.025, alphabet[p] + ' ' + labels[p], fontsize=fs, fontweight='bold',
        transform=ax.transAxes)

    ax = axs[1,p]
    ax.plot(ss/1e3, N_interp_glads.mean(axis=1), color=colors[1], 
        label='Mean', linewidth=1.5)
    ax.fill_between(ss/1e3, np.quantile(N_interp_glads, 0.16, axis=1),
        np.quantile(N_interp_glads, 0.84, axis=1),
        color=colors[0], alpha=0.7, label='68% interval', edgecolor='none')
    ax.fill_between(ss/1e3, np.quantile(N_interp_glads, 0.025, axis=1),
        np.quantile(N_interp_glads, 0.975, axis=1),
        color=colors[0], alpha=0.4, label='95% interval', edgecolor='none')
    ax.set_ylim([-2, 6.5])
    ax.text(0.025, 1.025, alphabet[p+3], fontsize=fs, fontweight='bold',
        transform=ax.transAxes)

for ax in axs.flat:
    ax.set_xlim([200, 0])
    ax.grid()
    ax.tick_params(labelsize=fs)

for ax in axs[:,1:].flat:
    ax.set_yticklabels([])

axs[0,0].set_ylabel('Flotation fraction (-)', fontsize=fs)
axs[1,0].set_ylabel('Effective pressure (MPa)', fontsize=fs)
axs[1,1].set_xlabel('Distance from grounding line (km)', fontsize=fs)

fig.subplots_adjust(left=0.06, right=0.985, bottom=0.1, top=0.9, wspace=0.15, hspace=0.15)

# axs[0,2].legend(bbox_to_anchor=(0, 1, 1., 1.0), loc='lower center', frameon=False, ncols=2)
# for ax in axs[-1]:
#     ax.set_xlabel('Distance from grounding line (km)')
axs[0,0].legend(bbox_to_anchor=(0,1.1,1,0.2), loc='lower left', frameon=False, ncols=3, fontsize=fs)
fig.savefig('figures/ensemble_spread_profiles.png', dpi=400)
fig.savefig('figures/ensemble_spread_profiles.pdf')
fig.savefig('../../manuscript/f02.png', dpi=400)
fig.savefig('../../manuscript/f02.pdf')
