
import os
import pickle
import time

import numpy as np

# import scipy.linalg
from matplotlib import pyplot as plt
from matplotlib.tri import Triangulation
from matplotlib import path
from matplotlib import colors
import cmocean
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import r2_score, explained_variance_score
from sklearn.inspection import permutation_importance
from scipy import interpolate
from scipy.stats import gaussian_kde
import xarray as xr

from utils.RF import RFData, AISData

"""
TODO
 - https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html#sklearn.ensemble.HistGradientBoostingRegressor
"""

def _interp2bedmachine(xi, yi, z, stride=5,
    bedmachine='../../data/bedmachine/BedMachineAntarctica-v3.nc'):
    with xr.open_dataset(bedmachine) as bm:
        x = bm['x'][::stride]
        y = bm['y'][::stride]
        bm_mask = bm['mask'][::stride, ::stride]
    
    # Take a rectangular subset of bedmachine
    bm_mask = bm_mask[np.logical_and(y>=yi.min(), y<=yi.max()), np.logical_and(x>=xi.min(), x<=xi.max())]
    x = x[np.logical_and(x>=xi.min(), x<=xi.max())]
    y = y[np.logical_and(y>=yi.min(), y<=yi.max())]
    
    xx,yy = np.meshgrid(x,y)

    # interpfn = interpolate.NearestNDInterpolator((mesh['x'], mesh['y'], z))
    meshxy = (xi, yi)
    zgrid = interpolate.griddata(meshxy, z, (xx,yy), method='linear', fill_value=np.nan)
    zgrid[bm_mask!=2] = np.nan
    return xx,yy,zgrid


def trainRF(basins, feature_keys, Xscale=None, Yscale=None, nPerBasin=1000,
    feature_importance=False, index=None):    
    print('trainRF::', index)
    rfData = RFData(basins, feature_keys, index=index)
    rfData.normalizeX(scale=Xscale)
    rfData.normalizeY(scale=Yscale)
    
    # Only train and evaluate where N>0 and pw>0
    mask = np.logical_and(rfData.Yphys>=0, rfData.Yphys<=1)

    X = rfData.X[mask]
    Y = rfData.Y[mask]

    # Choose a random subset of points
    if nPerBasin:
        rng = np.random.default_rng()
        randIndices = rng.choice(np.arange(len(Y)), len(basins)*nPerBasin)
        print('len(randIndices):', len(randIndices))

        Xsub = X[randIndices]
        Ysub = Y[randIndices]
    else:
        Xsub = X.copy()
        Ysub = Y.copy()

    # scikitlearn random forest
    # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
    regr = RandomForestRegressor(max_depth=10)
    # regr = HistGradientBoostingRegressor(max_iter=10, verbose=2)
    print('Fitting random forest')
    regr.fit(Xsub, Ysub)
    print('Done fitting')

    if feature_importance:
        print('Permutation importance')
        result = permutation_importance(
            regr, Xsub, Ysub, n_repeats=30, random_state=42, n_jobs=1
        )
        print(result)

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.bar(np.arange(len(feature_keys))-0.25, result['importances_mean'], yerr=result['importances_std'], width=0.4)
        ax.set_xticks(np.arange(len(feature_keys)), feature_keys, rotation=45, ha='right')
        ax.set_title('Flotation fraction feature importance')
        ax.set_ylabel(r'Mean $R^2$ decrease')

        ax2 = ax.twinx()
        ax2.bar(np.arange(len(feature_keys))+0.25, regr.feature_importances_, color='tab:orange', width=0.4)
        # ax.set_xticks(np.arange(len(feature_keys)), feature_keys, rotation=45, ha='right')
        # ax.set_title("Feature importances using permutation: flotation fraction")
        ax2.set_ylabel('Impurity-based feature importance', color='tab:orange')

        ax.grid()

        fig.subplots_adjust(bottom=0.4)
        fig.savefig('figures/feature_importance.png', dpi=400)

    return rfData, regr


def crossVal(basins, feature_keys, nPerBasin=1000, index=None):
    # Load all the data for the correct scalings    
    rfData = RFData(basins, feature_keys, index=index)
    rfData.normalizeX(scale=None)
    rfData.normalizeY(scale=None)

    for i in range(len(basins)):
        trainBasins = basins[:i] + basins[i+1:]
        testBasin = basins[i]
        print('CV basins', trainBasins)
        cvData, cvRegr = trainRF(trainBasins, feature_keys, 
            nPerBasin=nPerBasin, Xscale=rfData.Xscale, Yscale=rfData.Yscale, index=index)
        mesh = np.load(f'../../issm/{testBasin}/data/geom/mesh.npy', allow_pickle=True)
        levelset = np.load(f'../../issm/{testBasin}/data/geom/ocean_levelset.npy')
        
        # Make nan-padded arrays for plotting
        Yfull = np.zeros(mesh['numberofvertices'])
        testData = RFData([testBasin], feature_keys, index=index)
        testData.normalizeX(scale=rfData.Xscale)
        testData.normalizeY(scale=rfData.Yscale)
        Yfull[levelset>0] = testData.Yphys
        Yfull[levelset<=0] = np.nan

        mask = np.logical_and(testData.Yphys>=0, testData.Yphys<=1)

        Zhat = cvRegr.predict(testData.X)
        mu,sd = rfData.Yscale
        Yhat = mu + sd*Zhat
        np.save(f'data/CV_{testBasin}.npy', Yhat)

        Y = testData.Y
        Yphys = testData.Yphys

        # Convert to effective pressure
        bed = np.load(f'../../issm/{testBasin}/data/geom/bed.npy')
        thick = np.load(f'../../issm/{testBasin}/data/geom/thick.npy')
        rhow = 1023 # kg.m-3, ISSM default seawater
        rhofw = 1000
        rhoice = 917.0
        g = 9.81    # m.s-2
        phiBed = rhofw*g*bed
        pice = rhoice*g*thick
        N_glads = pice[levelset>0]*(1-Yphys)
        N_rf = pice[levelset>0]*(1 - Yhat)
        np.save(f'data/CV_{testBasin}_N_rf.npy', N_rf)
        np.save(f'data/CV_{testBasin}_N_glads.npy', N_glads)
        

        # Interpolating to bedmachine grid
        Yhat_interp = Yhat.copy()
        Yhat_interp[~mask] = np.nan
        Yphys_interp = Yphys.copy()
        Yphys_interp[~mask] = np.nan

        outline = np.load(f'../../data/ANT_Basins/basin_{testBasin}.npy')
        basinPath = path.Path(outline, closed=True)

        print('Interpolating to bedmachine')
        xi = mesh['x'][levelset==1][mask]
        yi = mesh['y'][levelset==1][mask]
        xx,yy,Yhat_bm = _interp2bedmachine(xi, yi, Yhat[mask])
        _,_,Yphys_bm = _interp2bedmachine(xi, yi, Yphys[mask])
        _,_,Nrf_bm = _interp2bedmachine(xi, yi, N_rf[mask])
        _,_,Nglads_bm = _interp2bedmachine(xi, yi, N_glads[mask])

        # basinMask = np.zeros(xx.shape)
        print('Masking out-of-basin points')
        xy = np.array([xx.flatten(), yy.flatten()]).T
        basinMask = basinPath.contains_points(xy).reshape(xx.shape)

        # Mask out additional points
        Yhat_bm[~basinMask] = np.nan
        Yphys_bm[~basinMask] = np.nan
        Nrf_bm[~basinMask] = np.nan
        Nglads_bm[~basinMask] = np.nan

        # Save the grid and interpolated outputs
        bmgrid = {}
        bmgrid['xx'] = xx
        bmgrid['yy'] = yy
        bmgrid['RF'] = Yhat_bm.astype(np.float32)
        bmgrid['glads'] = Yphys_bm.astype(np.float32)
        bmgrid['N_RF'] = Nrf_bm.astype(np.float32)
        bmgrid['N_glads'] = Nglads_bm.astype(np.float32)
        with open(f'data/CV_{testBasin}_bmgrid.pkl', 'wb') as bmout:
            pickle.dump(bmgrid, bmout)
        print('Done interpolating')

        Yhatfull = np.zeros(mesh['numberofvertices'])
        Yhatfull[levelset>0] = Yhat
        Yhatfull[levelset<=0] = np.nan

        ##########################################################################
        fig,axs = plt.subplots(nrows=2, ncols=2, sharex=False, sharey=False)
        mtri = Triangulation(mesh['x'], mesh['y'], mesh['elements']-1)
        Nmap = axs[0,0].tripcolor(mtri, Yfull, cmap=cmocean.cm.dense, vmin=0.5, vmax=1)
        axs[0,1].tripcolor(mtri, Yhatfull, cmap=cmocean.cm.dense, vmin=0.5, vmax=1)
        diffmap = axs[1,0].tripcolor(mtri, Yhatfull-Yfull, cmap=cmocean.cm.balance, vmin=-0.25, vmax=0.25)

        fig.subplots_adjust(left=0.0625, right=0.925, bottom=0.1, top=0.85,
            hspace=0.3, wspace=0.3)

        cbar1 = fig.colorbar(Nmap, ax=axs[0], label='Fraction of overburden')
        cbar2 = fig.colorbar(diffmap, ax=axs[1], label=r'$\Delta$ Fraction of overburden')

        axs[0,0].set_title('GlaDS')
        axs[0,1].set_title('RF prediction')

        # cnorm = colors.LogNorm(vmin=1, vmax=1e3)
        # axs[1,1].hexbin(Yphys[mask], Yhat[mask], norm=cnorm, cmap='afmhot_r')
        # axs[1,1].set_xlim([0, 1])
        axs[1,1].scatter(Yphys[mask], Yhat[mask], s=0.5, alpha=0.2)
        axs[1,1].set_ylim([0, 1])
        axs[1,1].set_xlabel('GlaDS')
        axs[1,1].set_ylabel('RF prediction')

        r2 = 1 - np.nanvar(Yhat[mask]-Yphys[mask])/np.nanvar(Yphys[mask])
        print('R2:', r2)
        fig.suptitle(f'{testBasin}, R2={r2:.3f}')

        r2_equalarea = 1 - np.nanvar(Yhat_bm - Yphys_bm)/np.nanvar(Yphys_bm)
        print('R2 interpolated:', r2_equalarea)

        for ax in axs.flat[:3]:
            ax.set_aspect('equal')
        
        msk = Yfull.copy()
        zifull = (Yfull - mu)/sd
        msk[np.abs(zifull)<2] = np.nan
        fig.savefig(f'figures/CV_{testBasin}.png', dpi=400)

        ##########################################################################
        fig,axs = plt.subplots(nrows=2, ncols=2, sharex=False, sharey=False)
        # mtri = Triangulation(mesh['x'], mesh['y'], mesh['elements']-1)
        bmgrid = np.load(f'data/CV_{testBasin}_bmgrid.pkl', allow_pickle=True)

        xx = bmgrid['xx']
        yy = bmgrid['yy']
        Nmap = axs[0,0].pcolormesh(xx, yy, bmgrid['glads'], cmap=cmocean.cm.dense, vmin=0.5, vmax=1)
        axs[0,1].pcolormesh(xx, yy, bmgrid['RF'], cmap=cmocean.cm.dense, vmin=0.5, vmax=1)
        diffmap = axs[1,0].pcolormesh(xx, yy, bmgrid['RF'] - bmgrid['glads'], cmap=cmocean.cm.balance, vmin=-0.25, vmax=0.25)

        fig.subplots_adjust(left=0.0625, right=0.925, bottom=0.1, top=0.85,
            hspace=0.3, wspace=0.3)

        cbar1 = fig.colorbar(Nmap, ax=axs[0], label='Fraction of overburden')
        cbar2 = fig.colorbar(diffmap, ax=axs[1], label=r'$\Delta$ Fraction of overburden')

        axs[0,0].set_title('GlaDS')
        axs[0,1].set_title('RF prediction')

        axs[1,1].scatter(bmgrid['glads'].flatten(), bmgrid['RF'].flatten(), s=0.5, alpha=0.2)
        # cnorm = colors.LogNorm(vmin=1, vmax=1e5)
        # axs[1,1].hexbin(bmgrid['glads'].flatten(), bmgrid['RF'].flatten(), norm=cnorm, cmap='afmhot_r')
        axs[1,1].set_xlim([0, 1])
        axs[1,1].set_ylim([0, 1])
        axs[1,1].set_xlabel('GlaDS')
        axs[1,1].set_ylabel('RF prediction')

        fig.suptitle(f'{testBasin}, R2={r2_equalarea:.3f}')

        for ax in axs.flat[:3]:
            ax.set_aspect('equal')
        fig.savefig(f'figures/CV_{testBasin}_gridded.png', dpi=400)


def predictContinent(rfData, rfRegr, feature_keys):
    stride = 10
    AISdata = AISData(feature_keys, stride=stride)
    AISdata.normalizeX(scale=rfData.Xscale)
    XAIS = AISdata.X
    mask = AISdata.mask

    print('XAIS.shape:', XAIS.shape)
    # XAIS = XAIS[:, ::10000]
    # print('XAIS:', XAIS.shape)
    t1 = time.perf_counter()
    Yhat = regr.predict(XAIS)
    t2 = time.perf_counter()
    print('Time for AIS prediction:', t2-t1)

    mu,sd = rfData.Yscale
    YhatPhys = mu + sd*Yhat

    AISpred = np.nan*np.zeros(mask.shape)
    AISpred[mask] = YhatPhys
    AISpred = np.flipud(AISpred)

    np.save('data/AISpred.npy', AISpred)

    fig,ax = plt.subplots()
    pc = ax.pcolormesh(AISpred, vmin=0.5, vmax=1.0, cmap=cmocean.cm.dense)
    ax.set_aspect('equal')
    fig.colorbar(pc, label='Fraction of overburden')
    fig.savefig('figures/AISpred.png', dpi=400)


    fig,ax = plt.subplots()
    pc = ax.pcolormesh(AISpred, vmin=0.8, vmax=1.05, cmap=cmocean.cm.dense)
    ax.set_aspect('equal')
    fig.colorbar(pc, label='Fraction of overburden')
    fig.savefig('figures/AISpredSE.png', dpi=400)
    
def predictBasins(rfData, rfRegr, feature_keys, basins):
    rhow = 1023 # kg.m-3, ISSM default seawater
    rhofresh = 1000
    rhoice = 917.0
    g = 9.81    # m.s-2
    for basin in basins:
        thick = np.load(f'../../issm/{basin}/data/geom/thick.npy')
        levelset = np.load(f'../../issm/{basin}/data/geom/ocean_levelset.npy')
        pice = rhoice*g*thick

        testData = RFData([basin], feature_keys, index=index)
        testData.normalizeX(scale=rfData.Xscale)
        testData.normalizeY(scale=rfData.Yscale)
        Yhat = rfRegr.predict(testData.X)
        mu,sd = rfData.Yscale
        Ypred = mu + sd*Yhat
        np.save(f'data/pred_{basin}.npy', Ypred)

        Yphys = testData.Yphys
        N_glads = pice[levelset>0]*(1-Yphys)
        N_RF = pice[levelset>0]*(1-Ypred)
        np.save(f'data/pred_{basin}_N_rf.npy', N_RF)
        np.save(f'data/pred_{basin}_N_glads.npy', N_glads)

def featureImportance(data, regr, basins, features, repeats=10):
    nfeat, dfeat = data.X.shape
    Yscale = data.Yscale

    rng = np.random.default_rng()

    nBasins = len(basins)
    deltaR2f = np.zeros((nBasins, repeats, dfeat))
    deltaR2N = np.zeros((nBasins, repeats, dfeat)) 
    
       
    
    for basinNum, testBasin in enumerate(basins):
        print(testBasin)

        testData = RFData([testBasin], features, index=index)
        testData.normalizeX(scale=data.Xscale)
        testData.normalizeY(scale=data.Yscale)
        bed = np.load(f'../../issm/{testBasin}/data/geom/bed.npy')
        thick = np.load(f'../../issm/{testBasin}/data/geom/thick.npy')
        levelset = np.load(f'../../issm/{testBasin}/data/geom/ocean_levelset.npy')
        rhow = 1023 # kg.m-3, ISSM default seawater
        rhofresh = 1000
        rhoice = 917.0
        g = 9.81    # m.s-2
        phiBed = rhofresh*g*bed
        pice = rhoice*g*thick
        Yphys = testData.Yphys
        N_glads = pice[levelset>0]*(1-Yphys)

        mask = np.logical_and(testData.Yphys>=0, testData.Yphys<=1)

        Yhat = Yscale[0] + Yscale[1]*regr.predict(testData.X)
        N_rf = pice[levelset>0]*(1 - Yhat)

        print('Base prediction')
        r2fbase = 1 - np.nanvar(Yhat[mask] - Yphys[mask])/np.nanvar(Yphys[mask])
        r2Nbase = 1 - np.nanvar(N_rf[mask] - N_glads[mask])/np.nanvar(N_glads[mask])

        print(r2fbase)
        print(r2Nbase)

        for p in range(dfeat):
            print('\tParameter {}/{}'.format(p+1, dfeat), end='\tRepeats: ', flush=True)
            for r in range(repeats):
                print(r+1, end=', ', flush=True)
                shuffidx = np.arange(testData.X.shape[0])
                rng.shuffle(shuffidx)
                Xpred = testData.X.copy()
                Xpred[:, p] = testData.X[shuffidx, p]
                # print('Xpred:', Xpred[:10, :])
                Yhat = Yscale[0] + Yscale[1]*regr.predict(Xpred)

                N_rf = pice[levelset>0]*(1 - Yhat)

                r2f = 1 - np.nanvar(Yhat[mask] - Yphys[mask])/np.nanvar(Yphys[mask])
                r2N = 1 - np.nanvar(N_rf[mask] - N_glads[mask])/np.nanvar(N_glads[mask])
                deltaR2f[basinNum, r, p] = r2fbase - r2f
                deltaR2N[basinNum, r, p] = r2Nbase - r2N
            print()
    
    return deltaR2f, deltaR2N

def simplifyModel(data, regr, basins, features, repeats=10,
    minFeatures=3, nPerBasin=1000):
    # Baseline: Compute and plot importance using all features
    nfeat = len(features)
    deltaR2f, deltaR2N = featureImportance(data, regr, basins, 
        features, repeats=repeats)
    plotImportance(features, df=deltaR2f, dN=deltaR2N, 
        figname=f'figures/simplify_model_{nfeat}.png')
    # Iteratively remove the least-importance feature
    while nfeat>minFeatures:
        mean_importance = np.mean(deltaR2f, axis=(0,1))
        print('mean_importance:', mean_importance)
        min_importance = np.min(mean_importance)
        min_index = np.argmin(mean_importance)
        print('Removing feature', features[min_index], 'with importance', min_importance)
        features.remove(features[min_index])
        
        nfeat = len(features)
        # Reload data and refit model
        data, regr = trainRF(basins, features, nPerBasin=nPerBasin)
        deltaR2f, deltaR2N = featureImportance(data, regr, basins, 
            features, repeats=repeats, index=index)
        plotImportance(features, df=deltaR2f, dN=deltaR2N, 
            figname=f'figures/simplify_model_{nfeat}.png')


def plotImportance(feature_keys, figname=None, df=None, dN=None):
    if df is None:
        df = np.load('deltaR2f.npy')
    if dN is None:
        dN = np.load('deltaR2N.npy')

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(np.arange(len(feature_keys))-0.25, 
        np.mean(df, axis=(0, 1)), 
        yerr=np.std(df, axis=(0, 1)), 
        width=0.4)

    ax.set_xticks(np.arange(len(feature_keys)), feature_keys, rotation=45, ha='right')
    # ax.set_title('Flotation fraction feature importance')
    ax.set_ylabel(r'Flotation fraction $R^2$ decrease', color='tab:blue')

    ax2 = ax.twinx()
    ax2.bar(np.arange(len(feature_keys))+0.25, 
        np.mean(dN, axis=(0, 1)), 
        yerr=np.std(df, axis=(0, 1)), 
        width=0.4,
        color='red')
    # ax.set_title("Feature importances using permutation: flotation fraction")
    ax2.set_ylabel('Effective pressure $R^2$ decrease', color='red')

    ax.set_ylim(bottom=0)
    ax2.set_ylim(bottom=0)
    ax.grid()
    fig.subplots_adjust(bottom=0.4)
    if figname is None:
        figname = 'figures/manual_feature_importance.png'
    fig.savefig(figname, dpi=400)


def clustering(basins, features):
    """Principal component clusterin
    """
    rfData = RFData(basins, features, index=index)
    rfData.normalizeX()
    rfData.normalizeY()

    U,S,VT = np.linalg.svd(rfData.X, full_matrices=False)
    print('U:', U.shape)
    print('S:', S.shape)
    print('VT:', VT.shape)
    print('Singular values:', S)

    print('VT:', VT[:, :2])

    contrib_var = S**2/np.sum(S**2)
    print('Variance contributions:', contrib_var)

    fig,ax = plt.subplots()

    for i,basin in enumerate(basins):
        basinData = RFData([basin], features, index=index)
        basinData.normalizeX(scale=rfData.Xscale)
        Ustar = basinData.X @ VT.T @ np.diag(1/S) @ np.diag(S)
        print('Ustar:', Ustar.shape)
        ax.scatter(Ustar[:,0], Ustar[:, 1], s=2, alpha=0.2, edgecolor='none', label=basin)
    ax.set_xlabel('PC1 ({:.3%})'.format(contrib_var[0]))
    ax.set_ylabel('PC2 ({:.3%})'.format(contrib_var[1]))

    # Whole-continent
    stride = 30
    featsAIS = np.load('features_AIS.pkl', allow_pickle=True)
    mask = ~np.isnan(featsAIS['bed'][::stride, ::stride])
    # with open('../../bedmachine/')
    print('featsAIS["bed"].shape', featsAIS['bed'].shape)
    Xphys = np.array([featsAIS[key][::stride, ::stride][mask] for key in features]).T
    Xphys = (Xphys - rfData.Xscale[0])/(rfData.Xscale[1] - rfData.Xscale[0])
    print('Xphys:', Xphys.shape)


    UAIS = Xphys @ VT.T @ np.diag(1/S) @ np.diag(S)
    print('KDE estimate')
    KDE = gaussian_kde(UAIS[:,:2].T)
    x = np.linspace(U[:,0].min(), U[:,0].max(), 100)*S[0]
    y = np.linspace(U[:,1].min(), U[:,1].max(), 100)*S[1]
    xx,yy = np.meshgrid(x, y)
    # ax.scatter(UAIS[:, 0], UAIS[:, 1], s=2, alpha=0.2, edgecolor='none', label='AIS', color='#aaaaaa', zorder=0)
    xy = np.array([xx.flatten(), yy.flatten()])
    print('Evaluating KDE')
    density_estimate = KDE(xy).reshape(xx.shape)
    print('Contouring')
    # print(density_estimate)
    # print(density_estimate.max())
    ax.contour(xx, yy, density_estimate, colors='k', zorder=5, levels=5)

    leg = ax.legend(bbox_to_anchor=(0,1,1,0.1), loc='lower left', ncols=5, markerscale=5)
    for lh in leg.legend_handles:
        lh.set_alpha(1)

    fig.savefig('figures/clustering.png', dpi=400)

    # Map the PCs to understand the clusters
    fig,axs = plt.subplots(ncols=2, figsize=(8, 4))
    for i in range(2):
        v = np.nan*np.zeros(mask.shape)
        v[mask] = UAIS[:, i]
        vmax = np.max(np.abs(v[mask]))
        vmin = -vmax

        axs[i].pcolormesh(np.flipud(v), cmap=cmocean.cm.balance, vmin=vmin, vmax=vmax)
        axs[i].set_aspect('equal')
        axs[i].set_title('PC1 {:.3%}'.format(contrib_var[i]))

    fig.savefig('figures/PCmap.png', dpi=400)

def scrambledThicknessPrediction(data, regr, basins, features, index=None):
    nfeat, dfeat = data.X.shape
    Yscale = data.Yscale

    rng = np.random.default_rng()

    nBasins = len(basins)
       
    
    for basinNum, testBasin in enumerate(basins):
        print(testBasin)

        testData = RFData([testBasin], features, index=index)
        testData.normalizeX(scale=data.Xscale)
        testData.normalizeY(scale=data.Yscale)
        bed = np.load(f'../../issm/{testBasin}/data/geom/bed.npy')
        thick = np.load(f'../../issm/{testBasin}/data/geom/thick.npy')
        levelset = np.load(f'../../issm/{testBasin}/data/geom/ocean_levelset.npy')
        rhow = 1023 # kg.m-3, ISSM default seawater
        rhofresh = 1000
        rhoice = 917.0
        g = 9.81    # m.s-2
        phiBed = rhofresh*g*bed
        pice = rhoice*g*thick
        Yphys = testData.Yphys
        N_glads = pice[levelset>0]*(1-Yphys)

        mask = np.logical_and(testData.Yphys>=0, testData.Yphys<=1)

        Yhat = Yscale[0] + Yscale[1]*regr.predict(testData.X)
        N_rf = pice[levelset>0]*(1 - Yhat)

        # print('Base prediction')
        # r2fbase = 1 - np.nanvar(Yhat[mask] - Yphys[mask])/np.nanvar(Yphys[mask])
        # r2Nbase = 1 - np.nanvar(N_rf[mask] - N_glads[mask])/np.nanvar(N_glads[mask])

        print(r2fbase)
        print(r2Nbase)

        rng = np.random.default_rng()
        Xscrambled = testData.X.copy()
        shuffidx = np.arange(testData.X.shape[0])
        rng.shuffle(shuffidx)
        Xscrambled[:, 2] = Xscrambled[shuffidx, 2]

        Zhat = Yscale[0] + Yscale[1]*regr.predict(Xscrambled)
        N_scrambled = pice[levelset>0]*(1 - Zhat)

        mesh = np.load(f'../../issm/{testBasin}/data/geom/mesh.npy', allow_pickle=True)
        Ngladsfull = np.nan*np.zeros(mesh['numberofvertices'])
        Ngladsfull[levelset>0] = N_glads
        Npredfull = np.nan*np.zeros(mesh['numberofvertices'])
        Npredfull[levelset>0] = N_rf
        Nscrambledfull = np.nan*np.zeros(mesh['numberofvertices'])
        Nscrambledfull[levelset>0] = N_scrambled

        Pgladsfull = np.nan*np.zeros(mesh['numberofvertices'])
        Pgladsfull[levelset>0] = Yphys
        Ppredfull = np.nan*np.zeros(mesh['numberofvertices'])
        Ppredfull[levelset>0] = Yhat
        Pscrambledfull = np.nan*np.zeros(mesh['numberofvertices'])
        Pscrambledfull[levelset>0] = Zhat

        mtri = Triangulation(mesh['x'], mesh['y'], mesh['elements']-1)
        fig,axs = plt.subplots(figsize=(12, 8), nrows=2, ncols=3)
        Fargs = dict(vmin=0, vmax=1, cmap=cmocean.cm.dense)
        Nargs = dict(vmin=0, vmax=5e6, cmap=cmocean.cm.haline)
        pc = axs[0,0].tripcolor(mtri, Pgladsfull, **Fargs)
        axs[0,1].tripcolor(mtri, Ppredfull, **Fargs)
        axs[0,2].tripcolor(mtri, Pscrambledfull, **Fargs)
        fig.colorbar(pc, ax=axs[0], label='Flotation fraction')

        pc = axs[1,0].tripcolor(mtri, Ngladsfull, **Nargs)
        axs[1,1].tripcolor(mtri, Npredfull, **Nargs)
        axs[1,2].tripcolor(mtri, Nscrambledfull, **Nargs)
        fig.colorbar(pc, ax=axs[1], label='Effective pressure (Pa)')

        axs[0,0].set_title('GlaDS')
        axs[0,1].set_title('RF: All features')
        axs[0,2].set_title('RF: Scrambled thickness')

        for ax in axs.flat:
            ax.set_aspect('equal')

        fig.savefig(f'figures/scrambled_thickness_{testBasin}.png', dpi=400)



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

    pred_basins = [
        'G-H_2050',
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

    index = None

    rfData, regr = trainRF(basins, features, nPerBasin=10000, feature_importance=False, index=index)
    with open('rf.pkl', 'wb') as rfout:
        pickle.dump(regr, rfout)
    predictBasins(rfData, regr, features, pred_basins)

    # dr2f, dr2N = featureImportance(rfData, regr, basins, features)
    # np.save('deltaR2f.npy', dr2f)
    # np.save('deltaR2N.npy', dr2N)

    # plotImportance(features)

    # scrambledThicknessPrediction(rfData, regr, basins, features)

    # simplifyModel(rfData, regr, basins, features)
    # print('feature_importances_:', regr.feature_importances_)

    # crossVal(basins, features, nPerBasin=10000, index=index)
    # predictContinent(rfData, regr, features)

    # clustering(basins, features)