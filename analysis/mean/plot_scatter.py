import numpy as np
from matplotlib import pyplot as plt


fs = 7
plt.rc('font', size=fs)
alphabet = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(i)']

from utils.RF import RFData


def main(basins, features, labels):
    data = RFData(basins, features)
    data.normalizeX()
    msk = np.logical_and(data.Yphys>=0, data.Yphys<=1)
    ncols = 3
    nfeat = data.Xphys.shape[1]
    nrows = int(np.ceil(nfeat/ncols))


    # Print feature correlation
    np.set_printoptions(precision=3, suppress=True)
    print(np.corrcoef(data.X.T))

    # Load RF predictions
    ypred = np.array([])
    for basin in basins:
        yi = np.load(f'data/pred_{basin}.npy')
        ypred = np.concatenate((ypred, yi))
    

    fig,axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7, 6), sharey=True)

    for i in range(nfeat):
        ax = axs.flat[i]
        ax.scatter(data.Xphys[msk, i], data.Yphys[msk], s=1, 
            alpha=0.1, edgecolor='none', label='GlaDS', rasterized=True)
        ax.scatter(data.Xphys[~msk, i], data.Yphys[~msk], s=1, 
            alpha=0.1, edgecolor='none', color='gray', label='GlaDS (masked)', rasterized=True)
        ax.text(0.05, 0.9, alphabet[i], transform=ax.transAxes, fontsize=fs)

        # Bin the RF predictions
        xbinNorm = np.linspace(-3, 3, 51)
        xmu = np.mean(data.Xphys[:, i])
        xsd = np.std(data.Xphys[:, i])
        xcNorm = 0.5*xbinNorm[1:] + 0.5*xbinNorm[:-1]
        X = (data.Xphys[:,i]-xmu)/xsd
        xcPhys = xmu + xsd*xcNorm
        ybin = np.zeros(xcNorm.shape)
        yupper = np.nan*np.zeros(xcNorm.shape)
        ylower = np.nan*np.zeros(xcNorm.shape)
        nbin = len(ybin)
        for k in range(nbin):
            isbin = np.logical_and(
                X>=xbinNorm[k],
                X<xbinNorm[k+1],
            )
            ybin[k] = np.mean(ypred[isbin])
            if len(ypred[isbin]>10):
                yupper[k] = np.quantile(ypred[isbin], 0.975)
                ylower[k] = np.quantile(ypred[isbin], 0.025)
        
        ax.plot(xcPhys, ybin, color='r', label='Random forest mean')
        # ax.fill_between(xcPhys, ylower, yupper, alpha=0.2, color='red', edgecolor='none')
        # ax.scatter(data.Xphys[msk, i], ypred[msk], s=1, alpha=0.1, edgecolor='none', color='red')
        ax.grid()
        ax.set_xlabel(labels[i], fontsize=fs)
        ax.set_ylim([-0.25, 1.25])
        # ax.set_xlim(np.quantile(data.Xphys[:,i], np.array([0.001, 0.999])))
        ax.tick_params(labelsize=fs)

    leg = axs.flat[0].legend(bbox_to_anchor=(0,1,0.3,1), loc='lower left', ncols=3, 
        frameon=False, fontsize=fs, markerscale=7)
    # Source - https://stackoverflow.com/a/42403471
    # Posted by lhuber, modified by community. See post 'Timeline' for change history
    # Retrieved 2026-05-19, License - CC BY-SA 4.0

    for lh in leg.legend_handles: 
        lh.set_alpha(1)

    
    
    for i in range(nfeat, nrows*ncols):
        axs.flat[i].set_visible(False)
    
    for ax in axs[:,0]:
        ax.set_ylabel('Flotation fraction', fontsize=fs)
    
    # Brute force nice x-axis limits
    axs.flat[0].set_xlim([-3000, 2000])
    axs.flat[1].set_xlim([0, 4000])
    axs.flat[2].set_xlim([0, 4500])
    axs.flat[3].set_xlim([0, 1.e6])
    axs.flat[4].set_xlim([-2.5, -0.5])
    axs.flat[5].set_xlim([0, 3.25e7])
    axs.flat[6].set_xlim([0, 0.0625])
    axs.flat[7].set_xlim([0, 0.25])
    axs.flat[8].set_xlim([0, 1500])


    # fig.tight_layout()
    fig.subplots_adjust(right=0.975, bottom=0.085, top=0.95,
        left=0.085, hspace=0.4, wspace=0.2)
    fig.savefig('figures/scatter.png', dpi=400)

    fig.savefig('../../manuscript/D01.png', dpi=400)
    fig.savefig('../../manuscript/D01.pdf', dpi=400)

if __name__=='__main__':
    basins = [
        'G-H',
        'Cp-D',
        'C-Cp',
        'B-C',
        'Jpp-K',
    ]

    features = [
        'bed',
        'surface',
        'thickness',
        'grounding_line_distance',
        'basal_melt',
        'potential',
        'surface_slope',
        'bed_slope',
        'potential_slope',
        # 'binned_flow_accumulation',
    ]

    labels = [
        'Bed elevation (m)',
        'Surface elevation (m)',
        'Ice thickness (m)',
        'Grounding line distance (m)',
        'log Basal melt rate (m w.a. a$^{-1}$)',
        'Hydraulic potential (Pa)',
        'Surface slope (-)',
        'Bed slope (-)',
        'Hydraulic potential slope (Pa m$^{-1}$)'
    ]



    main(basins, features, labels)
