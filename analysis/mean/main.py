"""
Issues
 - mean, sd of X is not always the same between training/predicting for CV
    -> get_features.py should write the mean and sd

"""

import os
import numpy as np

# import scipy.linalg
from matplotlib import pyplot as plt
from matplotlib.tri import Triangulation
import cmocean
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, explained_variance_score
from sklearn.inspection import permutation_importance
import scipy.interpolate

def reportFailedRuns(basins):
    for basin in basins:
        features = np.load(f'features_{basin}.pkl', allow_pickle=True)

        outputs = np.load(f'../../issm/{basin}/glads/N.npy')
        failed = np.where(outputs[:,0]==np.nan)[0]
        print(f'{basin}: failed runs:', failed)


def loadFeatures(basins, feature_keys, vmin=None, vmax=None):
    feature_list = None
    basin_num = []
    i=0
    for basin in basins:
        print(basin)
        basin_matrix = None
        features = np.load(f'features_{basin}.pkl', allow_pickle=True)
        for key in feature_keys:
            if basin_matrix is None:
                basin_matrix = features[key]
            else:
                basin_matrix = np.vstack((basin_matrix, features[key]))
        
        print(basin_matrix.shape)
        if feature_list is None:
            feature_list = basin_matrix.copy()
        else:
            feature_list = np.hstack((feature_list, basin_matrix))
        basin_num.extend(0*basin_matrix[0] + i)
        i += 1
    
    print('feature_list.shape:', feature_list.shape)
    basin_num = np.array(basin_num)
    print(basin_num.shape)

    if vmin is None:
        vmin = np.min(feature_list, axis=1)
        vmax = np.max(feature_list, axis=1)
    X = (feature_list - vmin[:,None])/(vmax[:,None] - vmin[:,None])
    print('Normalized features min/max:', X.min(axis=1), X.max(axis=1))
    return X, (vmin, vmax)

def loadModelOutputs(basins, field='N'):
    modelOutputs = []
    for basin in basins:

        # Load levelset to find where ice is grounded
        levelset = np.load(
            os.path.join(f'../../issm/{basin}',
                'data/geom/ocean_levelset.npy')
        )
        # Add model outputs to list
        outputs = np.load(f'../../issm/{basin}/glads/{field}.npy')[levelset>0,:]
        meanOutputs = np.nanmean(outputs, axis=1) # Take average over perturbed parameters
        modelOutputs.extend(meanOutputs) 
    return np.array(modelOutputs)


def trainRF(basins, feature_keys, vmin=None, vmax=None, mu=None, sd=None, nPerBasin=1000):
    X, Xscale = loadFeatures(basins, feature_keys, vmin=vmin, vmax=vmax)
    Y = loadModelOutputs(basins, field='N')
    F = loadModelOutputs(basins, field='ff')
    print('X.shape:', X.shape)
    print('Y.shape:', Y.shape)
    assert X.shape[1]==Y.shape[0]

    # TODO
    # Only train and evaluate where N>0 and pw>0

    # First guess: cap max N
    mask = np.logical_and(F>=0, F<=1)
    X = X[:, mask]
    Y = Y[mask]
    

    # Standardize Y values
    if mu is None:
        mu = np.mean(Y)
    if sd is None:
        sd = np.std(Y)
    Z = (Y - mu)/sd

    # Remove total outliers
    # X = X[:,np.abs(Z)<2]
    # Y = Y[np.abs(Z)<2]
    # Z = Z[np.abs(Z)<2]

    # X = X[:, Z>=-mu/sd]
    # Y = Y[Y>=-mu/sd]
    # Z = Z[Z>=-mu/sd]

    rng = np.random.default_rng()
    randIndices = rng.choice(np.arange(len(Y)), len(basins)*nPerBasin)
    print('len(randIndices):', len(randIndices))

    Xsub = X[:, randIndices]
    Ysub = Y[randIndices]
    Zsub = Z[randIndices]



    # scikitlearn random forest
    # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
    regr = RandomForestRegressor(max_depth=10)
    print('Fitting random forest')
    regr.fit(Xsub.T, Zsub)

    print('Predicting random forest')
    Zhat = regr.predict(X.T)

    # R2 = 1 - np.var(Zhat-Z)/np.var(Z)
    R2 = r2_score(Z, Zhat)
    print('R2:', R2)

    # R2_weighted = 

    # fig,ax = plt.subplots()
    # ax.scatter(Z, Zhat)
    # ax.set_xlabel('Model')
    # ax.set_ylabel('Predicted')
    # plt.show()
    # R2 = regr.score(X.T, Z)
    # print('R2:', R2)
    
    # Nhat = regr.predict(joined)


    if False:
        print('Permutation importance')
        result = permutation_importance(
            regr, Xsub.T, Zsub, n_repeats=10, random_state=42, n_jobs=1
        )
        print(result)
        print('Done training')

        fig, ax = plt.subplots()
        ax.bar(np.arange(len(feature_keys)), result['importances_mean'], yerr=result['importances_std'])
        # fig, ax = plt.subplots()
        # forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
        ax.set_xticks(np.arange(len(feature_keys)), feature_keys, rotation=45, ha='right')
        ax.set_title("Feature importances using permutation on full model")
        ax.set_ylabel("Mean accuracy decrease")
        fig.savefig('figures/feature_importance.png', dpi=400)
    
    return regr, Xscale, (mu, sd)

def plotRF(regr, Xscale, Yscale, basins, feature_keys):
    for basin in basins:
        mesh = np.load(f'../../issm/{basin}/data/geom/mesh.npy', allow_pickle=True)
        levelset = np.load(f'../../issm/{basin}/data/geom/ocean_levelset.npy')
        N = loadModelOutputs([basin])
        Nfull = np.zeros(mesh['numberofvertices'])
        Nfull[levelset>0] = N
        Nfull[levelset<0] = np.nan
        X, _ = loadFeatures([basin], feature_keys, vmin=Xscale[0], vmax=Xscale[1])


        # Approximate node area
        mesh = np.load(f'../../issm/{basin}/data/geom/mesh.npy', allow_pickle=True)
        # elx = np.mean(mesh['x'][mesh['elements']-1], axis=1)
        # ely = np.mean(mesh['y'][mesh['elements']-1], axis=1)
        # areaInterpolator = scipy.interpolate.LinearNDInterpolator((elx, ely), mesh['area'])
        # areaInterp = areaInterpolator((mesh['x'], mesh['y']))
        # areaInterp[areaInterp==0] = 1e6
        # areaInterp[np.isnan(areaInterp)] = 1e6
        # areaInterp[np.isinf(areaInterp)] = 1e6

        F = loadModelOutputs([basin], field='ff')
        mask = np.logical_and(F>=0, F<=1)
        # print(areaInterp)

        mu,sd = Yscale
        Zhat = regr.predict(X.T)
        Nhat = mu + sd*Zhat

        # Nhat[N<0] = np.nan
        # Nhat[N>5e6] = np.nan
        Nhatfull = np.zeros(mesh['numberofvertices'])
        Nhatfull[levelset>0] = Nhat
        Nhatfull[levelset<=0] = np.nan
        # print('Nhat:', Nhat.shape)

        # fig,ax = plt.subplots()
        # mtri = Triangulation(mesh['x'], mesh['y'], mesh['elements']-1)
        # ax.tripcolor(mtri, areaInterp)
        # plt.savefig('figures/area.png', dpi=400)


        fig,axs = plt.subplots(nrows=2, ncols=2, sharex=False, sharey=False)
        mtri = Triangulation(mesh['x'], mesh['y'], mesh['elements']-1)
        Nmap = axs[0,0].tripcolor(mtri, Nfull, cmap=cmocean.cm.haline, vmin=0, vmax=5e6)
        axs[0,1].tripcolor(mtri, Nhatfull, cmap=cmocean.cm.haline, vmin=0, vmax=5e6)
        diffmap = axs[1,0].tripcolor(mtri, Nhatfull-Nfull, cmap=cmocean.cm.balance, vmin=-3e6, vmax=3e6)

        cbar1 = fig.colorbar(Nmap, ax=axs[0], label='N (Pa)')
        cbar2 = fig.colorbar(diffmap, ax=axs[1], label='N error (Pa)')

        # axs[1,1].set_visible(False)
        axs[1,1].scatter(N[mask], Nhat[mask], s=0.5, alpha=0.2)
        axs[1,1].set_xlim([0, 5e6])
        axs[1,1].set_ylim([0, 5e6])

        # zi = (N - mu)/sd
        # mask = np.abs(zi<2)
        # zobs = zi[zi<2]
        # zmod = Zhat[zi<2]
        # r2 = 1 - np.var(Nhat[mask] - N[mask])/np.var(N[mask])
        # r2 = 1 - np.var(zmod - zobs)/np.var(zobs)
        # w = np.ones(len(N))
        # w[np.abs(zi)<2] = 0
        # w = areaInterp[levelset>0]
        # print('w:', w.dtype, w.sum())
        # w[np.abs(zi)>2] = 0
        # w = w/len(w)
        # print('weights:', w)
        # print(w.min(), w.max())

        # r2Weighted = r2_score(N, Nhat, sample_weight=w)
        # r2Unweighted = r2_score(N, Nhat, sample_weight=None)
        r2 = 1 - np.nanvar(Nhat[mask]-N[mask])/np.nanvar(N[mask])

        # print('R2: weighted', r2Weighted)
        # print('R2: unweighted', r2Unweighted)
        print('R2:', r2)
        fig.suptitle(f'{basin}, R2={r2:.3f}')

        for ax in axs.flat[:3]:
            ax.set_aspect('equal')
        
        msk = Nfull.copy()
        zifull = (Nfull - mu)/sd
        msk[np.abs(zifull)<2] = np.nan
        # msk[zifull>=-mu/sd] = np.nan
        # msk[zifull>mu] = np.nan
        # print(np.where(np.abs(zifull)>1))
        # print(np.where(zifull<-mu/sd))
        # for ax in axs.flat[:3]:
            # ax.tripcolor(mtri, msk, cmap=cmocean.cm.haline, vmin=0.5, vmax=1, hatch='/')
        fig.savefig(f'figures/overview_{basin}.png', dpi=400)
    
    # plt.show()

def crossVal(basins, feature_keys, Xscale, Yscale, nPerBasin=4000):
    for i in range(len(basins)):
        trainBasins = basins[:i] + basins[i+1:]
        testBasin = basins[i]
        print('CV basins', trainBasins)

        vmin, vmax = Xscale
        mu,sd = Yscale
        regr, _, _ = trainRF(trainBasins, feature_keys, 
            nPerBasin=nPerBasin, mu=mu, sd=sd, vmin=vmin, vmax=vmax)

        mesh = np.load(f'../../issm/{testBasin}/data/geom/mesh.npy', allow_pickle=True)
        levelset = np.load(f'../../issm/{testBasin}/data/geom/ocean_levelset.npy')
        N = loadModelOutputs([testBasin], field='N')
        F = loadModelOutputs([testBasin], field='ff')
        Nfull = np.zeros(mesh['numberofvertices'])
        Nfull[levelset>0] = N
        Nfull[levelset<0] = np.nan
        X, scaleX = loadFeatures([testBasin], feature_keys, vmin=vmin, vmax=vmax)


        mask = np.logical_and(F>=0, F<=1)
        # print(areaInterp)

        Zhat = regr.predict(X.T)
        Nhat = mu + sd*Zhat

        # Nhat[N<0] = np.nan
        # Nhat[N>5e6] = np.nan
        Nhatfull = np.zeros(mesh['numberofvertices'])
        Nhatfull[levelset>0] = Nhat
        Nhatfull[levelset<=0] = np.nan
        # print('Nhat:', Nhat.shape)

        fig,axs = plt.subplots(nrows=2, ncols=2, sharex=False, sharey=False)
        mtri = Triangulation(mesh['x'], mesh['y'], mesh['elements']-1)
        Nmap = axs[0,0].tripcolor(mtri, Nfull, cmap=cmocean.cm.haline, vmin=0, vmax=5e6)
        axs[0,1].tripcolor(mtri, Nhatfull, cmap=cmocean.cm.haline, vmin=0, vmax=5e6)
        diffmap = axs[1,0].tripcolor(mtri, Nhatfull-Nfull, cmap=cmocean.cm.balance, vmin=-2e6, vmax=2e6)

        cbar1 = fig.colorbar(Nmap, ax=axs[0], label='N (Pa)')
        cbar2 = fig.colorbar(diffmap, ax=axs[1], label='N error (Pa)')

        axs[1,1].scatter(N[mask], Nhat[mask], s=0.5, alpha=0.2)
        axs[1,1].set_xlim([0, 1])
        axs[1,1].set_ylim([0, 1])

        r2 = 1 - np.nanvar(Nhat[mask]-N[mask])/np.nanvar(N[mask])

        # print('R2: weighted', r2Weighted)
        # print('R2: unweighted', r2Unweighted)
        print('R2:', r2)
        fig.suptitle(f'{testBasin}, R2={r2:.3f}')

        for ax in axs.flat[:3]:
            ax.set_aspect('equal')
        
        msk = Nfull.copy()
        zifull = (Nfull - mu)/sd
        msk[np.abs(zifull)<2] = np.nan
        # msk[zifull>=-mu/sd] = np.nan
        # msk[zifull>mu] = np.nan
        # print(np.where(np.abs(zifull)>1))
        # print(np.where(zifull<-mu/sd))
        # for ax in axs.flat[:3]:
            # ax.tripcolor(mtri, msk, cmap=cmocean.cm.haline, vmin=0.5, vmax=1, hatch='/')
        fig.savefig(f'figures/CV_{testBasin}.png', dpi=400)

def predictContinent(regr, vmin, vmax, scale):
    # feats = np.load('features_AIS.pkl', mmap_mode='r', allow_pickle=True)
    X, scaleX = loadFeatures(basins, feature_keys)

    
    # feature_list = None
    i=0
    stride = 10
    # for basin in basins:
    feature_matrix = []
    features = np.load('features_AIS.pkl', allow_pickle=True)
    mask = ~np.isnan(features['bed'][::stride, ::stride])
    for key in feature_keys:
        # if feature_matrix is None:
            feature_matrix.append(features[key][::stride, ::stride][mask])
        # else:
            # feature_matrix = np.vstack((feature_matrix, features[key][::stride, ::stride][mask]))
    feature_matrix = np.array(feature_matrix)

    print('features_matrix:', feature_matrix.shape)
    XAIS = (feature_matrix - vmin[:,None])/vmax[:,None]

    print('XAIS:', XAIS.shape)
    # XAIS = XAIS[:, ::10000]
    # print('XAIS:', XAIS.shape)
    Yhat = regr.predict(XAIS.T)

    mu,sd = scale
    YhatPhys = mu + sd*Yhat

    AISpred = np.nan*np.zeros(mask.shape)
    AISpred[mask] = YhatPhys
    AISpred = np.flipud(AISpred)

    fig,ax = plt.subplots()
    pc = ax.pcolormesh(AISpred, vmin=0.8, vmax=1.05, cmap=cmocean.cm.haline)
    ax.set_aspect('equal')
    fig.colorbar(pc, label='Fraction of overburden')
    fig.savefig('AISpred.png', dpi=400)
    
    

if __name__=='__main__':

    basins = [
        'G-H',
        # 'F-G',  # TODO check outputs, look like numerical issues
        # 'Ep-F', # jobs not done, numerical issues?
        'Cp-D',
        'C-Cp',
        'B-C',
        'Jpp-K',
        # 'J-Jpp',# TODO check outputs, look like numerical issues
    ]

    feature_keys = [
        'bed',
        'surface',
        'thickness',
        'grounding_line_distance',
        'basal_melt',
        'potential',
        # 'binned_flow_accumulation',
    ]

    reportFailedRuns(basins)
    RF, Xscale, Yscale = trainRF(basins, feature_keys, nPerBasin=4000)
    plotRF(RF, Xscale, Yscale, basins, feature_keys)

    crossVal(basins, feature_keys, Xscale, Yscale, nPerBasin=4000)

    # _, (vmin, vmax) = loadFeatures(basins, feature_keys)
    # predictContinent(RF, vmin, vmax, Yscale)
