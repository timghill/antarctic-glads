import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

basins = [
    'B-C',
    'C-Cp',
    'Cp-D',
    'G-H',
    'Jpp-K',
]

cases = ['poc', 'glads', 'rf']

fig = plt.figure(figsize=(8, 6))
gs = GridSpec(nrows=2, ncols=3, wspace=0.3, hspace=0.4,
    left=0.1, bottom=0.1, right=0.95, top=0.9)
cols,rows = np.meshgrid(np.arange(3, dtype=int), np.arange(2, dtype=int))
axs = []

alphabet = ['a', 'b', 'c', 'd', 'e', 'f']

for i,basin in enumerate(basins):
    ax = fig.add_subplot(gs[rows.flat[i], cols.flat[i]])
    axs.append(ax)

    for case in cases:
        J = np.load(f'{basin}/issm/solutions/J{case}.npy')
        alpha = J[:,0]
        Jv = J[:, 1]
        Jr = J[:,-2]/alpha
        ax.loglog(Jr, Jv, marker='.', label=case)

        if case=='poc':
            for j in range(4, len(Jv), 2):
                ax.text(Jr[j], Jv[j], r'$\alpha = {:.1e}$'.format(alpha[j]), rotation=30, fontsize=8)
    
    ax.grid()
    ax.set_xlabel(r'$\mathcal{J}_{\rm{reg}}$', fontsize=10)
    ax.set_ylabel(r'$\mathcal{J}_{\rm{u}}$', fontsize=10)
    # ax.set_title('{} ({})'.format(alphabet[i], basin))
    
    ax.text(-0.1, 1.025, alphabet[i], fontsize=10,
        fontweight='bold', transform=ax.transAxes)
    ax.text(0.5, 1.025, basin, fontsize=10,
        ha='center', transform=ax.transAxes)


# axs = np.array(axs).reshape(rows.shape)
axs[0].legend(bbox_to_anchor=(0,1.1,0.2,1), loc='lower left',
    frameon=False, ncols=3)

fig.savefig('lcurve.png', dpi=400)

