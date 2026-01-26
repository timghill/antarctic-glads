import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

basins = [
    'B-C',
    'C-Cp',
    'Cp-D',
    'G-H',
    # 'Jpp-K',
]
basin_labels = [
    'Amery',
    'Denman',
    'Aurora',
    'Amundsen\nSea',
]

cases = ['poc', 'glads', 'rf']
case_labels = ['POC', 'GlaDS', 'Random Forest']

colors = ['dimgray', '#0d7d87', '#c31e23']

values = np.array([
    1e-8,
    1e-9,
    1e-9,
    1e-8,
])

fig = plt.figure(figsize=(8, 3))
gs = GridSpec(nrows=1, ncols=4, wspace=0.325, hspace=0.4,
    left=0.06, bottom=0.18, right=0.95, top=0.8)
cols,rows = np.meshgrid(np.arange(3, dtype=int), np.arange(2, dtype=int))
axs = []

alphabet = ['a', 'b', 'c', 'd', 'e', 'f']

for i,basin in enumerate(basins):
    ax = fig.add_subplot(gs[i])
    axs.append(ax)

    for j,case in enumerate(cases):
        J = np.load(f'{basin}/issm/solutions/J{case}.npy')
        alpha = J[:,0]
        Jv = J[:, 1]
        Jr = J[:,-2]/alpha
        for k,aval in enumerate(alpha):
            if aval==values[i]:
                print('CLOSE')
                label=r'$\hat \alpha$' if case=='poc' else '_label'
                ax.plot(Jr[k], Jv[k], label=label, marker='o', markerfacecolor='none', 
                    markeredgecolor='k', markersize=6, color='k', linestyle='none')
        ax.loglog(Jr, Jv, marker='.', label=case_labels[j])

        if case=='poc':
            for j in range(4, len(Jv), 2):
                ax.text(Jr[j], Jv[j], r'$\alpha = {:.1e}$'.format(alpha[j]), rotation=30, fontsize=6)
        
    
    ax.grid()
    ax.set_xlabel(r'$\mathcal{J}_{\rm{reg}}$', fontsize=8)
    # ax.set_title('{} ({})'.format(alphabet[i], basin))

    title = '({}) {} ({})'.format(alphabet[i], basin_labels[i], basin)
    ax.text(0.9, 0.95, title, fontsize=8,
        ha='right', va='top', transform=ax.transAxes)
    

axs[0].set_ylabel(r'$\mathcal{J}_{\rm{u}}$', fontsize=8, labelpad=-12)

# axs = np.array(axs).reshape(rows.shape)
axs[0].legend(bbox_to_anchor=(0,1.1,0.2,1), loc='lower left',
    frameon=False, ncols=4, fontsize=8)

for ax in axs:
    ax.tick_params(labelsize=8)

fig.savefig('tmp.png', dpi=400)

