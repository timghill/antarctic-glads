import os
import pickle
import time

import numpy as np

from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.tri import Triangulation
import cmocean


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, explained_variance_score
from sklearn.inspection import permutation_importance
import scipy.interpolate
from scipy.stats import gaussian_kde

from utils.RF import RFDataPara


def trainRF(basins, feature_keys, Xscale=None, Yscale=None, nPerSim=100,
    feature_importance=False):
    """
    Train Random Forest on entire dataset
    """
    data = RFDataPara(basins, feature_keys, field='ff')
    data.normalizeX()
    data.normalizeY()

    npara, dpara = data.Theta.shape
    nfeat = len(features)
    nbasins = len(basins)

    # Randomize nodes for each para combination
    Xtrain = np.zeros((nbasins*nPerSim*npara, nfeat+dpara), dtype=np.float32)
    Ytrain = np.zeros(nbasins*nPerSim*npara, dtype=np.float32)
    rng = np.random.default_rng()
    for i in range(npara):
        mask = np.logical_and(data.Yphys[:, i]>=0, data.Yphys[:, i]<=1)
        Xi = data.X[mask]
        Yi = data.Y[mask, i]
        randIndices = rng.choice(np.arange(len(Yi)), nbasins*nPerSim)
        i1 = i*nbasins*nPerSim
        i2 = (i+1)*nbasins*nPerSim
        Xtrain[i1:i2, :nfeat] = Xi[randIndices,:]
        Xtrain[i1:i2, nfeat:] = data.Theta[i, :]
        Ytrain[i1:i2] = Yi[randIndices]

    regr = RandomForestRegressor()
    print('Fitting random forest')
    t0 = time.perf_counter()
    regr.fit(Xtrain, Ytrain)
    dt = time.perf_counter() - t0
    print(f'Done fitting ({dt:.3f} seconds)')
    return data, regr

def parabasinCV(basins, feature_keys, nPerSim=100, field='ff', k=1):
    """
    Cross-validation over 100x parameter combinations and Nx basins
    """
    data = RFDataPara(basins, feature_keys, field=field)
    data.normalizeX()
    data.normalizeY()
    mu,sd = data.Yscale
    # Keep track of all preds for overall CV R2
    YYpred = np.zeros(data.Y.shape)

    npara, dpara = data.Theta.shape
    nfeat = len(features)
    nbasins = len(basins)-1
    istart = 0

    rhoi = 917
    g = 9.81

    for j,basin in enumerate(basins):
        # Make train/test subsets and normalize in the same
        # way as the whole data
        trainBasins = basins[:j] + basins[j+1:]
        print('trainBasins:', trainBasins)
        testBasin = basins[j]
        print('Test basin:', testBasin)
        trainData = RFDataPara(trainBasins, features, field=field)
        trainData.normalizeX(scale=data.Xscale)
        trainData.normalizeY(scale=data.Yscale)

        testData = RFDataPara([testBasin], features, field=field)
        testData.normalizeX(scale=data.Xscale)
        testData.normalizeY(scale=data.Yscale)

        # Randomize nodes for each para combination
        Xtrain = np.zeros((nbasins*nPerSim*npara, nfeat+dpara), dtype=np.float32)
        Ytrain = np.zeros(nbasins*nPerSim*npara, dtype=np.float32)
        rng = np.random.default_rng()
        for i in range(npara):
            mask = np.logical_and(trainData.Yphys[:, i]>=0, trainData.Yphys[:, i]<=1)
            Xi = trainData.X[mask]
            Yi = trainData.Y[mask, i]
            randIndices = rng.choice(np.arange(len(Yi)), nbasins*nPerSim, replace=False)
            i1 = i*nbasins*nPerSim
            i2 = (i+1)*nbasins*nPerSim
            Xtrain[i1:i2, :nfeat] = Xi[randIndices,:]
            Xtrain[i1:i2, nfeat:] = trainData.Theta[i, :]
            Ytrain[i1:i2] = Yi[randIndices]

        # Parametric k-fold CV
        nfolds = int(npara/k)
        # Randomly assign each parameter vector to one of the folds
        scrambled_inds = rng.choice(np.arange(npara), size=npara, replace=False)
        for fold in range(nfolds):
            cvinds = scrambled_inds[fold*k:(fold+1)*k]
            print('Para CV step {}/{}'.format(fold+1, nfolds))
            # print('cvinds:', cvinds)

            # Compute the indices of the data that we hide for training
            delinds = np.array([np.arange(k*nbasins*nPerSim, (k+1)*nbasins*nPerSim) for k in cvinds])
            trainmask = np.ones((npara*nbasins*nPerSim), dtype=bool)
            trainmask[delinds] = False

            Xcv = Xtrain[trainmask, :]
            Ycv = Ytrain[trainmask]

            # Construct test data subset
            Xtest = np.zeros((testData.X.shape[0]*k, nfeat+dpara), dtype=np.float32)
            # Xtest = data.X[~trainmask]
            for jj,index in enumerate(cvinds):
                i1 = jj*testData.X.shape[0]
                i2 = i1 + testData.X.shape[0]
                Xtest[i1:i2, :nfeat] = testData.X
                Xtest[i1:i2, nfeat:] = testData.Theta[index]
            Ytest = testData.Y[:, cvinds]
            
            # Train RF using just the CV data
            regr = RandomForestRegressor()
            # print('Fitting random forest')
            t0 = time.perf_counter()
            # print('Xcv.shape:', Xcv.shape)
            # print('Ycv.shape:', Xcv.shape)
            regr.fit(Xcv, Ycv)
            dt = time.perf_counter() - t0
            # print(f'Done fitting ({dt:.3f} seconds)')

            # print('Predicting random forest')
            t0 = time.perf_counter()
            Ypred = regr.predict(Xtest)
            dt = time.perf_counter() - t0
            # print(f'Done predicting ({dt:.3f} seconds)')
            YYpred[istart:(istart+testData.X.shape[0]), cvinds] = Ypred.reshape((testData.X.shape[0], k), order='F')
        ypred = YYpred[istart:(istart+testData.X.shape[0]),:]
        ypredphys = mu + sd*ypred
        thick = np.load(f'../../issm/{basin}/data/geom/thick.npy')
        levelset = np.load(f'../../issm/{basin}/data/geom/ocean_levelset.npy')
        N_glads = rhoi*g*thick[levelset>0,None]*(1 - testData.Yphys)
        N_RF = rhoi*g*thick[levelset>0,None]*(1 - ypredphys)
        np.save(f'data/CV_{basin}_f_rf.npy', ypredphys)
        np.save(f'data/CV_{basin}_f_glads.npy', testData.Yphys)
        np.save(f'data/CV_{basin}_N_rf.npy', N_RF)
        np.save(f'data/CV_{basin}_N_glads', N_glads)
        istart += testData.X.shape[0]
    
    print('SUMMARY')
    mask = np.logical_and(data.Yphys>=0, data.Yphys<=1)
    summR2 = 1 - np.var(data.Y[mask] - YYpred[mask])/np.var(data.Y[mask])
    print(summR2)
    return YYpred, summR2

# def featureImportance(data, regr, basins, features, repeats=5):
#     nfeat, dfeat = data.X.shape
#     npara,dpara = data.Theta.shape
#     Yscale = data.Yscale

#     rng = np.random.default_rng()

#     nBasins = len(basins)
#     deltaR2f = np.zeros((nBasins, repeats, dfeat + dpara))
#     deltaR2N = np.zeros((nBasins, repeats, dfeat + dpara)) 

#     # nfeat = data.X.shape[1]
    
       
    
#     for basinNum, testBasin in enumerate(basins):
#     # for basinNum, testBasin in enumerate(basins[:1]):
#         print(testBasin)

#         testData = RFDataPara([testBasin], features)
#         testData.normalizeX(scale=data.Xscale)
#         testData.normalizeY(scale=data.Yscale)
#         bed = np.load(f'../../issm/{testBasin}/data/geom/bed.npy')
#         thick = np.load(f'../../issm/{testBasin}/data/geom/thick.npy')
#         levelset = np.load(f'../../issm/{testBasin}/data/geom/ocean_levelset.npy')
#         rhow = 1023 # kg.m-3, ISSM default seawater
#         rhofresh = 1000
#         rhoice = 917.0
#         g = 9.81    # m.s-2
#         phiBed = rhofresh*g*bed
#         pice = rhoice*g*thick
#         Yphys = testData.Yphys
#         N_glads = pice[levelset>0,None]*(1-Yphys)

#         mask = np.logical_and(testData.Yphys>=0, testData.Yphys<=1)


#         Xpred = np.zeros((len(testData.X)*npara, dfeat+dpara), dtype=np.float32)
#         # Ysim = np.zeros((len(testData.Y)*npara), dtype=np.float32)
#         for i in range(npara):
#             i1 = i*len(testData.X)
#             i2 = (i+1)*len(testData.X)
#             Xpred[i1:i2, :dfeat] = testData.X
#             Xpred[i1:i2, dfeat:] = testData.Theta[i, :]

#         Ysim = testData.Y
#         Yhat = Yscale[0] + Yscale[1]*regr.predict(Xpred).reshape(Ysim.shape, order='F')

#         N_rf = pice[levelset>0,None]*(1 - Yhat)

#         print('Base prediction')
#         r2fbase = 1 - np.nanvar(Yhat[mask] - Yphys[mask])/np.nanvar(Yphys[mask])
#         r2Nbase = 1 - np.nanvar(N_rf[mask] - N_glads[mask])/np.nanvar(N_glads[mask])

#         print(r2fbase)
#         # print(r2Nbase)

#         for p in range(dfeat + dpara):
#             print('\tParameter {}/{}'.format(p+1, dfeat + dpara), end='\tRepeats: ', flush=True)
#             for r in range(repeats):
#                 print(r+1, end=', ', flush=True)
#                 shuffidx = np.arange(testData.X.shape[0]*npara)
#                 rng.shuffle(shuffidx)
#                 xp = Xpred.copy()
#                 xp[:, p] = xp[shuffidx, p]
#                 # print('Xpred:', Xpred[:10, :])
#                 Yhat = Yscale[0] + Yscale[1]*regr.predict(xp).reshape(Ysim.shape, order='F')

#                 N_rf = pice[levelset>0,None]*(1 - Yhat)

#                 r2f = 1 - np.nanvar(Yhat[mask] - Yphys[mask])/np.nanvar(Yphys[mask])
#                 r2N = 1 - np.nanvar(N_rf[mask] - N_glads[mask])/np.nanvar(N_glads[mask])
#                 deltaR2f[basinNum, r, p] = r2fbase - r2f
#                 deltaR2N[basinNum, r, p] = r2Nbase - r2N
#             print()
    
#     return deltaR2f, deltaR2N

def main(basins, features, field='ff', nPerSim=100, k=10):

    print('Training model...', end=' ', flush=True)
    t0 = time.perf_counter()
    data,regr = trainRF(basins, features, nPerSim=nPerSim)
    dt = time.perf_counter() - t0
    print(f'done ({dt:.2f} seconds)')   
    print('Tree-based importance:', regr.feature_importances_)
    print('Saving trained model...', end=' ', flush=True)
    with open('model.pkl', 'wb') as fout:
        pickle.dump(regr, fout)
    print('done')


    print('Cross-validation')
    Ypred, R2 = parabasinCV(basins, features, nPerSim=nPerSim, k=k, field=field)

    return



# def evaluate_error(basins, features, highlight=None):
#     data = RFDataPara(basins, features)
#     data.normalizeY()
#     mu,sd = data.Yscale
#     pred = np.load('CVpred.npy')
#     pred = mu + sd*pred

#     error = pred - data.Yphys

#     mask = np.logical_and(data.Yphys>=0, data.Yphys<=1)
    
#     pred[~mask] = np.nan
#     glads = data.Yphys
#     glads[~mask] = np.nan
#     R2_by_para = 1 - np.nanvar(pred-glads, axis=0)/np.nanvar(glads, axis=0)
#     R2_by_mesh = 1 - np.nanvar(pred-glads, axis=1)/np.nanvar(glads, axis=1)
#     R2 = 1 - np.nanvar(pred-glads)/np.nanvar(glads)

#     Thetaphys = np.loadtxt('../../issm/theta_physical.csv',
#         delimiter=',', skiprows=1)
#     Theta = np.loadtxt('../../issm/theta_standard.csv',
#         delimiter=',', skiprows=1)
#     names = np.loadtxt('../../issm/theta_physical.csv',
#         delimiter=',', dtype=str, max_rows=1)

#     # Plot parameters
#     # ncols = 2
#     # nrows = int(np.ceil(len(names)/2))
#     # fig,axs = plt.subplots(nrows=nrows, ncols=ncols, sharey=True)
#     # for i in range(len(names)):
#     #     ax = axs.flat[i]
#     #     ax.scatter(Theta[:, i], R2_by_para)
#     #     ax.set_xlabel(names[i])
#     #     ax.grid()
#     #     ax.axhline(R2, color='k', linestyle='dashed', linewidth=1.5)
    
#     #     if highlight:
#     #         ax.scatter(Theta[highlight, i], R2_by_para[highlight], color='red')

#     # fig.tight_layout()
#     # fig.savefig('figures/R2_scatter.png', dpi=400)

#     allN = np.array([])
#     allNhat = np.array([])
#     allY = np.array([])
#     allYhat = np.array([])

#     istart = 0
#     for basin in basins:
#         # basinData = RFDataPara([basin], features)
#         mesh = np.load(f'../../issm/{basin}/data/geom/mesh.npy', allow_pickle=True)
#         levelset = np.load(f'../../issm/{basin}/data/geom/ocean_levelset.npy')
#         fig,ax = plt.subplots()
#         mtri = Triangulation(mesh['x'], mesh['y'], mesh['elements']-1)
#         ngrounded = len(levelset[levelset==1])
#         err = np.nan*np.ones(len(mesh['x']))
#         err[levelset==1] = R2_by_mesh[istart:(istart+ngrounded)]
#         # yi = glads[istart:(istart+ngrounded)]
#         # yhati = pred[istart:(istart+ngrounded)]
#         glads_basin = np.nan*np.ones(len(mesh['x']))
#         glads_basin[levelset==1] = np.mean(glads[istart:(istart+ngrounded)], axis=1)
#         istart += ngrounded

#         err[glads_basin<0] = np.nan
#         err[glads_basin>1] = np.nan
#         # err[np.isnan(glads_basin)] = np.nan
#         # err[err<=0] = np.nan

#         Y = np.load(f'../../issm/{basin}/glads/ff.npy')[levelset>0,:]
#         Yhat = np.load(f'data/CV_{basin}_f_rf.npy')
#         N = np.load(f'data/CV_{basin}_N_glads.npy')
#         Nhat = np.load(f'data/CV_{basin}_N_rf.npy')
#         pc = ax.tripcolor(mtri, err, cmap=cmocean.cm.matter, vmin=0, vmax=1)
#         ax.set_aspect('equal')

#         mm = np.logical_and(Y<=1, Y>=0)
#         # r2 = 1 - np.nanvar(yhati-yi)/np.nanvar(yi)
#         r2 = 1 - np.nanvar(Yhat[mm] - Y[mm])/np.nanvar(Y[mm])
#         r2N = 1 - np.nanvar(Nhat[mm] - N[mm])/np.nanvar(N[mm])


#         Ymean = np.mean(Y, axis=1)
#         Yhatmean = np.mean(Yhat, axis=1)
#         Nmean = np.mean(N, axis=1)
#         Nhatmean = np.mean(Nhat, axis=1)
#         mmean = np.logical_and(Ymean<=1, Ymean>=0)

#         r2mean = 1 - np.nanvar(Yhatmean[mmean] - Ymean[mmean])/np.nanvar(Ymean[mmean])
#         r2Nmean = 1 - np.nanvar(Nhatmean[mmean] - Nmean[mmean])/np.nanvar(Nmean[mmean])

#         ax.set_title(f'{basin} ($R^2$={r2:.3f})')
#         print(basin, r2, r2N, r2mean, r2Nmean)
#         fig.colorbar(pc, label=r'$R^2$')
#         fig.savefig(f'figures/R2_map_{basin}.png', dpi=400)

#         allN = np.concatenate((allN, N[mm]))
#         allNhat = np.concatenate((allNhat, Nhat[mm]))
#         allY = np.concatenate((allY, Y[mm]))
#         allYhat = np.concatenate((allYhat, Yhat[mm]))
    
#     fig,axs = plt.subplots(ncols = 2, figsize=(8, 4))

#     ax = axs[0]
#     fmin = 0.75
#     ax.hexbin(allYhat, allY, bins=None, cmap=cmocean.cm.rain, gridsize=50,
#         extent=(fmin, 1, fmin, 1))
#     ax.set_xlabel('GlaDS Flotation Fraction (-)')
#     ax.set_ylabel('Random Forest Flotation Fraction (-)')
#     ax.grid()
#     ax.set_aspect('equal')
#     ax.set_xlim([fmin, 1])
#     ax.set_ylim([fmin, 1])
#     allR2 = 1 - np.nanvar(allYhat - allY)/np.nanvar(allY)
#     ax.set_title('$R^2$ = {:.3f}'.format(allR2))


#     ax = axs[1]
#     hb = ax.hexbin(allNhat/1e6, allN/1e6, bins=None, cmap=cmocean.cm.rain, gridsize=50,
#         extent=(0, 5, 0, 5))
#     ax.set_xlabel('GlaDS Effective Pressure (MPa)')
#     ax.set_ylabel('Random Forest Effective Pressure (MPa)')
#     ax.grid()
#     ax.set_aspect('equal')
#     ax.set_xlim([0, 5])
#     ax.set_ylim([0, 5])
#     allR2 = 1 - np.nanvar(allNhat - allN)/np.nanvar(allN)
#     ax.set_title('$R^2$ = {:.3f}'.format(allR2))

#     fig.subplots_adjust(left=0.085, bottom=0.085, right=0.975, top=0.915, wspace=0.35)

#     cb = fig.colorbar(hb, ax=axs, label='Counts (N={:.3e})'.format(len(allNhat)))
#     fig.savefig('figures/hexbin.png', dpi=400)

        



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
        # 'surface',
        'thickness',
        # 'grounding_line_distance',
        # 'basal_melt',
        'potential',
        'surface_slope',
        # 'bed_slope',
        # 'potential_slope',
        # 'binned_flow_accumulation',
    ]
    # theta: sheet cond, channel cond, r_b, l_c, A
    main(basins, features, field='ff', nPerSim=10, k=10)
    # evaluate_error(basins, features, highlight=95)
