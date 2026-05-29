import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors
import cmocean


fs = 7
plt.rc('font', size=fs)
alphabet = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(i)']

from utils.RF import RFData

    
# Brute force nice x-axis limits
xlim = np.array(
        [[-3000, 2000],
        [0, 4000],
        [0, 4500],
        [0, 1.e3],
        [-2.5, -0.5],
        [0, 3.25e1],
        [0, 0.0625],
        [0, 0.25],
        [0, 1500],
])
ylim = np.array([-0.25, 1.25])



def main(basins, features, labels):
    data = RFData(basins, features)
    data.normalizeX()
    msk = np.logical_and(data.Yphys>=0, data.Yphys<=1)
    data.Xphys[:, 3] = data.Xphys[:, 3]/1e3
    data.Xphys[:, 5] = data.Xphys[:, 5]/1e6
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
    

    fig,axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7, 6), 
        sharey=True)

    mincnt = 1
    maxcnt = 2e3
    lognorm = colors.LogNorm(vmin=mincnt, vmax=maxcnt, clip=True)
    linnorm = colors.Normalize(vmin=mincnt, vmax=maxcnt)
    for i in range(nfeat):
        ax = axs.flat[i]
        extenti = np.array([xlim[i][0], xlim[i][1], ylim[0], ylim[1]])
        mpbl = ax.hexbin(data.Xphys[:, i], data.Yphys[:], rasterized=True,
            gridsize=(30, 20), extent=extenti, mincnt=mincnt, 
            cmap=cmocean.cm.rain, norm=linnorm, edgecolors='none',
            label='GlaDS')
        ax.axhline(y=1, color='k', linestyle='dashed', linewidth=1)
        ax.axhline(y=0, color='k', linestyle='dashed', linewidth=1)

        # mpbl = ax.hexbin(data.Xphys[~msk, i], data.Yphys[~msk], rasterized=True,
        #     gridsize=(30, 20), extent=extenti, mincnt=mincnt, 
        #     cmap=cmocean.cm.gray_r, norm=lognorm)
        # ax.hexbin(data.Xphys[~msk, i], data.Yphys[~msk],  rasterized=True,
        #     gridsize=(15, 10))
        ax.text(0.05, 0.9, alphabet[i], transform=ax.transAxes, fontsize=fs)

        # Bin the RF predictions
        xbinNorm = np.linspace(-3, 3, 51)
        xmu = np.mean(data.Xphys[:, i])
        xsd = np.std(data.Xphys[:, i])
        N = data.Xphys.shape[0]
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
        ax.set_ylim(ylim)
        # ax.set_xlim(np.quantile(data.Xphys[:,i], np.array([0.001, 0.999])))
        ax.tick_params(labelsize=fs)
        ax.set_xlim(xlim[i])

    # leg = axs.flat[0].legend(bbox_to_anchor=(0,1,0.3,1), loc='lower left', ncols=3, 
    #     frameon=False, fontsize=fs, markerscale=7)
    # # Source - https://stackoverflow.com/a/42403471
    # # Posted by lhuber, modified by community. See post 'Timeline' for change history
    # # Retrieved 2026-05-19, License - CC BY-SA 4.0

    # for lh in leg.legend_handles: 
    #     lh.set_alpha(1)

    
    
    for i in range(nfeat, nrows*ncols):
        axs.flat[i].set_visible(False)
    
    for ax in axs[:,0]:
        ax.set_ylabel('Flotation fraction', fontsize=fs)


    # fig.tight_layout()
    fig.subplots_adjust(right=0.95, bottom=0.075, top=0.985,
        left=0.085, hspace=0.4, wspace=0.2)
    cbar = fig.colorbar(mpbl, ax=axs, shrink=0.4, fraction=0.05, pad=0.02,
        label=f'Count (N={N:,})', extend='both', location='right')
    cbar.ax.set_yticks([mincnt, 500, 1000, 1500, 2000])
    fig.savefig('figures/heatmap.png', dpi=400)

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
        'Grounding line distance (km)',
        'log Basal melt rate (m w.a. a$^{-1}$)',
        'Hydraulic potential (MPa)',
        'Surface slope (-)',
        'Bed slope (-)',
        'Hydraulic potential slope (Pa m$^{-1}$)'
    ]



    main(basins, features, labels)
