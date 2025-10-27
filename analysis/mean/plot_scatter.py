import numpy as np
from matplotlib import pyplot as plt
from utils.RF import RFData

def main(basins, features):
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
    

    fig,axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8, 6), sharey=True)

    for i in range(nfeat):
        ax = axs.flat[i]
        ax.scatter(data.Xphys[msk, i], data.Yphys[msk], s=1, alpha=0.1, edgecolor='none')
        ax.scatter(data.Xphys[~msk, i], data.Yphys[~msk], s=1, alpha=0.1, edgecolor='none', color='gray')

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
        
        ax.plot(xcPhys, ybin, color='r')
        # ax.fill_between(xcPhys, ylower, yupper, alpha=0.2, color='red', edgecolor='none')
        # ax.scatter(data.Xphys[msk, i], ypred[msk], s=1, alpha=0.1, edgecolor='none', color='red')
        ax.grid()
        ax.set_xlabel(features[i])
        ax.set_ylim([-0.5, 1.5])
        ax.set_xlim([xmu-3.5*xsd, xmu+3.5*xsd])
        ax.set_xlim()

    
    
    for i in range(nfeat, nrows*ncols):
        axs.flat[i].set_visible(False)
    
    for ax in axs[:,0]:
        ax.set_ylabel('Flotation fraction')


    fig.tight_layout()
    fig.savefig('figures/scatter.png', dpi=400)

if __name__=='__main__':
    basins = [
        'G-H',
        # 'F-G',  # TODO check outputs, look like numerical issues
        # 'Ep-F', # jobs not done
        'Cp-D',
        'C-Cp',
        'B-C',
        'Jpp-K',
        # 'J-Jpp',# TODO check outputs, look like numerical issues
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
    main(basins, features)