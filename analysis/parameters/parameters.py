"""
TODO
 - Evaluate ff and N performance
 - Evaluate for each basin

"""

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

class RFData:
    def __init__(self, basins, featureKeys, field='ff'):
        self.field = field
        self.features = featureKeys
        self.Xphys = self.loadFeatures(basins, featureKeys)
        self.loadParameters()

        self.Yphys = self.loadModelOutputs(basins, field=field)

    def loadFeatures(self, basins, featureKeys):
        feature_list = None
        for basin in basins:
            basin_matrix = None
            features = np.load(f'../mean/features_{basin}.pkl', allow_pickle=True)
            for key in featureKeys:
                if basin_matrix is None:
                    basin_matrix = features[key][:,None]
                else:
                    basin_matrix = np.hstack((basin_matrix, features[key][:,None]))            
            
            if feature_list is None:
                feature_list = basin_matrix.copy()
            else:
                feature_list = np.vstack((feature_list, basin_matrix))
        return feature_list
    
    def loadParameters(self):
        Thetaphys = np.loadtxt('../../issm/theta_physical.csv',
            delimiter=',', skiprows=1)
        Theta = np.loadtxt('../../issm/theta_standard.csv',
            delimiter=',', skiprows=1)
        names = np.loadtxt('../../issm/theta_physical.csv',
            delimiter=',', dtype=str, max_rows=1)
        self.Thetaphys = Thetaphys
        self.Theta = Theta
        self.Thetanames = names
        return (Thetaphys, Theta, names)
        
    def loadModelOutputs(self, basins, field='ff'):
        modelOutputs = []
        mask = []
        for basin in basins:
            # Load levelset to find where ice is grounded
            levelset = np.load(
                os.path.join(f'../../issm/{basin}',
                    'data/geom/ocean_levelset.npy')
            )
            # Add model outputs to list
            outputs = np.load(f'../../issm/{basin}/glads/{field}.npy')[levelset>0,:]
            modelOutputs.extend(outputs) 
        return np.array(modelOutputs)
    
    def normalizeX(self, scale=None):
        if scale is None:
            xmin = np.min(self.Xphys, axis=0)
            xmax = np.max(self.Xphys, axis=0)
            scale = (xmin, xmax)
        xmin, xmax = scale
        X = (self.Xphys - xmin)/(xmax-xmin)
        self.X = X
        self.Xscale = scale
        return X, scale
    
    def normalizeY(self, scale=None):
        if scale is None:
            mu = np.mean(self.Yphys)
            sd = np.std(self.Yphys)
            scale = (mu, sd)
        mu,sd = scale
        Y = (self.Yphys - mu)/sd
        self.Y = Y
        self.Yscale = scale
        return Y, scale

def trainRF(basins, feature_keys, Xscale=None, Yscale=None, nPerBasin=100,
    feature_importance=False):
    data = RFData(basins, feature_keys, field='ff')
    data.normalizeX()
    data.normalizeY()

    npara, dpara = data.Theta.shape
    nfeat = len(features)
    nbasins = len(basins)

    # Randomize nodes for each para combination
    Xtrain = np.zeros((nbasins*nPerBasin*npara, nfeat+dpara), dtype=np.float32)
    Ytrain = np.zeros(nbasins*nPerBasin*npara, dtype=np.float32)
    rng = np.random.default_rng()
    for i in range(npara):
        mask = np.logical_and(data.Yphys[:, i]>=0, data.Yphys[:, i]<=1)
        Xi = data.X[mask]
        Yi = data.Y[mask, i]
        randIndices = rng.choice(np.arange(len(Yi)), nbasins*nPerBasin)
        i1 = i*nbasins*nPerBasin
        i2 = (i+1)*nbasins*nPerBasin
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

def paraCV(basins, feature_keys, k=1, nPerBasin=100):

    data = RFData(basins, feature_keys, field='ff')
    data.normalizeX()
    data.normalizeY()
    # Keep track of all preds for overall CV R2
    YYpred = np.zeros(data.Y.shape)

    npara, dpara = data.Theta.shape
    nfeat = len(features)
    nbasins = len(basins)

    # Randomize nodes for each para combination
    Xtrain = np.zeros((nbasins*nPerBasin*npara, nfeat+dpara), dtype=np.float32)
    Ytrain = np.zeros(nbasins*nPerBasin*npara, dtype=np.float32)
    rng = np.random.default_rng()
    for i in range(npara):
        mask = np.logical_and(data.Yphys[:, i]>=0, data.Yphys[:, i]<=1)
        Xi = data.X[mask]
        Yi = data.Y[mask, i]
        randIndices = rng.choice(np.arange(len(Yi)), nbasins*nPerBasin, replace=False)
        i1 = i*nbasins*nPerBasin
        i2 = (i+1)*nbasins*nPerBasin
        Xtrain[i1:i2, :nfeat] = Xi[randIndices,:]
        Xtrain[i1:i2, nfeat:] = data.Theta[i, :]
        Ytrain[i1:i2] = Yi[randIndices]

    # Parametric k-fold CV
    nfolds = int(npara/k)
    scrambled_inds = rng.choice(np.arange(npara), size=npara, replace=False)
    for fold in range(nfolds):
        cvinds = scrambled_inds[fold*k:(fold+1)*k]
        print('Para CV step {}/{}'.format(fold+1, nfolds))
        # print('cvinds:', cvinds)

        delinds = np.array([np.arange(k*nbasins*nPerBasin, (k+1)*nbasins*nPerBasin) for k in cvinds])
        trainmask = np.ones((npara*nbasins*nPerBasin), dtype=bool)
        trainmask[delinds] = False

        # Xcv = np.concatenate((Xtrain[:i1], Xtrain[i2:]))
        # Ycv = np.concatenate((Ytrain[:i1], Ytrain[i2:]))
        Xcv = Xtrain[trainmask, :]
        Ycv = Ytrain[trainmask]

        Xtest = np.zeros((data.X.shape[0]*k, nfeat+dpara), dtype=np.float32)
        # Xtest = data.X[~trainmask]
        for jj,index in enumerate(cvinds):
            i1 = jj*data.X.shape[0]
            i2 = i1 + data.X.shape[0]
            Xtest[i1:i2, :nfeat] = data.X
            Xtest[i1:i2, nfeat:] = data.Theta[index]
        Ytest = data.Y[:, cvinds]
        
        regr = RandomForestRegressor()
        # print('Fitting random forest')
        t0 = time.perf_counter()
        regr.fit(Xcv, Ycv)
        dt = time.perf_counter() - t0
        # print(f'Done fitting ({dt:.3f} seconds)')

        # print('Predicting random forest')
        t0 = time.perf_counter()
        Ypred = regr.predict(Xtest)
        dt = time.perf_counter() - t0
        # print(f'Done predicting ({dt:.3f} seconds)')

        YYpred[:, cvinds] = Ypred.reshape((data.X.shape[0], k), order='F')
    
    print('SUMMARY')
    mask = np.logical_and(data.Yphys>=0, data.Yphys<=1)
    summR2 = 1 - np.var(data.Y[mask] - YYpred[mask])/np.var(data.Y[mask])
    return YYpred, summR2

def basinCV(basins, feature_keys, nPerBasin=100, field='ff'):
    data = RFData(basins, feature_keys)
    data.normalizeX()
    data.normalizeY()
    
    for i,basin in enumerate(basins):
        trainBasins = basins[:i] + basins[i+1:]
        print('trainBasins:', trainBasins)
        testBasin = basins[i]
        print(testBasin)
        cvData = RFData(trainBasins, features, field=field)

        cvData.normalizeX(scale=data.Xscale)
        cvData.normalizeY(scale=data.Yscale)

        testData = RFData([testBasin], features, field=field)
        testData.normalizeX(scale=data.Xscale)
        testData.normalizeY(scale=data.Yscale)

        npara = cvData.Theta.shape[0]
        dpara = cvData.Theta.shape[1]
        nfeat = len(features)
        nbasins = len(trainBasins)
        Xtrain = np.zeros((nbasins*nPerBasin*npara, nfeat+dpara), dtype=np.float32)
        Ytrain = np.zeros(nbasins*nPerBasin*npara, dtype=np.float32)
        print(Xtrain.shape)
        rng = np.random.default_rng()
        for i in range(npara):
            mask = np.logical_and(cvData.Yphys[:, i]>=0, cvData.Yphys[:, i]<=1)
            Xi = cvData.X[mask]
            Yi = cvData.Y[mask, i]
            randIndices = rng.choice(np.arange(len(Yi)), nbasins*nPerBasin)
            i1 = i*nbasins*nPerBasin
            i2 = (i+1)*nbasins*nPerBasin
            Xtrain[i1:i2, :nfeat] = Xi[randIndices,:]
            Xtrain[i1:i2, nfeat:] = cvData.Theta[i, :]
            Ytrain[i1:i2] = Yi[randIndices]
        
        regr = RandomForestRegressor()
        print('Fitting random forest')
        t0 = time.perf_counter()
        regr.fit(Xtrain, Ytrain)
        dt = time.perf_counter() - t0
        print(f'Done fitting ({dt:.3f} seconds)')

        # Construct test feature array
        Xpred = np.zeros((len(testData.X)*npara, nfeat+dpara), dtype=np.float32)
        Ysim = np.zeros((len(testData.Y)*npara), dtype=np.float32)
        for i in range(npara):
            i1 = i*len(testData.X)
            i2 = (i+1)*len(testData.X)
            Xpred[i1:i2, :nfeat] = testData.X
            Xpred[i1:i2, nfeat:] = testData.Theta[i, :]
        Ysim = testData.Y
        mask = np.logical_and(testData.Yphys>=0, testData.Yphys<=1)
        print('Predicting random forest')
        t0 = time.perf_counter()
        Ypred = regr.predict(Xpred).reshape(Ysim.shape, order='F')
        dt = time.perf_counter() - t0
        print(f'Done predicting ({dt:.3f} seconds)')

        cvR2 = 1 - np.var(Ypred[mask]-Ysim[mask])/np.var(Ysim[mask])
        print('R2:', cvR2)

def parabasinCV(basins, feature_keys, nPerBasin=100, field='ff', k=1):
    data = RFData(basins, feature_keys, field=field)
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
        trainBasins = basins[:j] + basins[j+1:]
        print('trainBasins:', trainBasins)
        testBasin = basins[j]
        print('Test basin:', testBasin)
        trainData = RFData(trainBasins, features, field=field)
        trainData.normalizeX(scale=data.Xscale)
        trainData.normalizeY(scale=data.Yscale)

        testData = RFData([testBasin], features, field=field)
        testData.normalizeX(scale=data.Xscale)
        testData.normalizeY(scale=data.Yscale)

        # Randomize nodes for each para combination
        Xtrain = np.zeros((nbasins*nPerBasin*npara, nfeat+dpara), dtype=np.float32)
        Ytrain = np.zeros(nbasins*nPerBasin*npara, dtype=np.float32)
        # print('trainData.X:', trainData.X.shape)
        rng = np.random.default_rng()
        for i in range(npara):
            mask = np.logical_and(trainData.Yphys[:, i]>=0, trainData.Yphys[:, i]<=1)
            Xi = trainData.X[mask]
            Yi = trainData.Y[mask, i]
            randIndices = rng.choice(np.arange(len(Yi)), nbasins*nPerBasin, replace=False)
            i1 = i*nbasins*nPerBasin
            i2 = (i+1)*nbasins*nPerBasin
            Xtrain[i1:i2, :nfeat] = Xi[randIndices,:]
            Xtrain[i1:i2, nfeat:] = trainData.Theta[i, :]
            Ytrain[i1:i2] = Yi[randIndices]

        # Parametric k-fold CV
        nfolds = int(npara/k)
        scrambled_inds = rng.choice(np.arange(npara), size=npara, replace=False)
        for fold in range(nfolds):
            cvinds = scrambled_inds[fold*k:(fold+1)*k]
            print('Para CV step {}/{}'.format(fold+1, nfolds))
            # print('cvinds:', cvinds)

            delinds = np.array([np.arange(k*nbasins*nPerBasin, (k+1)*nbasins*nPerBasin) for k in cvinds])
            trainmask = np.ones((npara*nbasins*nPerBasin), dtype=bool)
            trainmask[delinds] = False

            # Xcv = np.concatenate((Xtrain[:i1], Xtrain[i2:]))
            # Ycv = np.concatenate((Ytrain[:i1], Ytrain[i2:]))
            Xcv = Xtrain[trainmask, :]
            Ycv = Ytrain[trainmask]

            Xtest = np.zeros((testData.X.shape[0]*k, nfeat+dpara), dtype=np.float32)
            # Xtest = data.X[~trainmask]
            for jj,index in enumerate(cvinds):
                i1 = jj*testData.X.shape[0]
                i2 = i1 + testData.X.shape[0]
                Xtest[i1:i2, :nfeat] = testData.X
                Xtest[i1:i2, nfeat:] = testData.Theta[index]
            Ytest = testData.Y[:, cvinds]
            
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
        np.save(f'data/CV_{basin}_N_rf.npy', N_RF)
        np.save(f'data/CV_{basin}_N_glads', N_glads)
        istart += testData.X.shape[0]
    
    print('SUMMARY')
    mask = np.logical_and(data.Yphys>=0, data.Yphys<=1)
    summR2 = 1 - np.var(data.Y[mask] - YYpred[mask])/np.var(data.Y[mask])
    print(summR2)
    return YYpred, summR2

def featureImportance(data, regr, basins, features, repeats=5):
    nfeat, dfeat = data.X.shape
    npara,dpara = data.Theta.shape
    Yscale = data.Yscale

    rng = np.random.default_rng()

    nBasins = len(basins)
    deltaR2f = np.zeros((nBasins, repeats, dfeat + dpara))
    deltaR2N = np.zeros((nBasins, repeats, dfeat + dpara)) 

    # nfeat = data.X.shape[1]
    
       
    
    for basinNum, testBasin in enumerate(basins):
    # for basinNum, testBasin in enumerate(basins[:1]):
        print(testBasin)

        testData = RFData([testBasin], features)
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
        N_glads = pice[levelset>0,None]*(1-Yphys)

        mask = np.logical_and(testData.Yphys>=0, testData.Yphys<=1)


        Xpred = np.zeros((len(testData.X)*npara, dfeat+dpara), dtype=np.float32)
        # Ysim = np.zeros((len(testData.Y)*npara), dtype=np.float32)
        for i in range(npara):
            i1 = i*len(testData.X)
            i2 = (i+1)*len(testData.X)
            Xpred[i1:i2, :dfeat] = testData.X
            Xpred[i1:i2, dfeat:] = testData.Theta[i, :]

        Ysim = testData.Y
        Yhat = Yscale[0] + Yscale[1]*regr.predict(Xpred).reshape(Ysim.shape, order='F')

        N_rf = pice[levelset>0,None]*(1 - Yhat)

        print('Base prediction')
        r2fbase = 1 - np.nanvar(Yhat[mask] - Yphys[mask])/np.nanvar(Yphys[mask])
        r2Nbase = 1 - np.nanvar(N_rf[mask] - N_glads[mask])/np.nanvar(N_glads[mask])

        print(r2fbase)
        # print(r2Nbase)

        for p in range(dfeat + dpara):
            print('\tParameter {}/{}'.format(p+1, dfeat + dpara), end='\tRepeats: ', flush=True)
            for r in range(repeats):
                print(r+1, end=', ', flush=True)
                shuffidx = np.arange(testData.X.shape[0]*npara)
                rng.shuffle(shuffidx)
                xp = Xpred.copy()
                xp[:, p] = xp[shuffidx, p]
                # print('Xpred:', Xpred[:10, :])
                Yhat = Yscale[0] + Yscale[1]*regr.predict(xp).reshape(Ysim.shape, order='F')

                N_rf = pice[levelset>0,None]*(1 - Yhat)

                r2f = 1 - np.nanvar(Yhat[mask] - Yphys[mask])/np.nanvar(Yphys[mask])
                r2N = 1 - np.nanvar(N_rf[mask] - N_glads[mask])/np.nanvar(N_glads[mask])
                deltaR2f[basinNum, r, p] = r2fbase - r2f
                deltaR2N[basinNum, r, p] = r2Nbase - r2N
            print()
    
    return deltaR2f, deltaR2N

def main(basins, features, field='ff', nPerBasin=100, k=10,
    feature_importance=False):

    print('Training model...', end=' ', flush=True)
    data,regr = trainRF(basins, features, nPerBasin=nPerBasin)
    print('done')   
    print('Tree-based importance:', regr.feature_importances_)
    print('Saving trained model...', end=' ', flush=True)
    with open('model.pkl', 'wb') as fout:
        pickle.dump(regr, fout)
    
    print('done')

    if feature_importance:
        print('Feature importance')
        dr2f, dr2N = featureImportance(data, regr, basins, features)
        np.save('data/deltaR2f.npy', dr2f)
        np.save('data/deltaR2N.npy', dr2N)
        print(dr2f, dr2N)
        print('done')

    # ypred, R2 = parabasinCV(basins, features, nPerBasin=nPerBasin, k=k)
    # print('ypred:', ypred.shape)
    # np.save('CVpred.npy', ypred)
    
    # print('PARAMETER CROSS-VALIDATION')
    # ypred, paraR2 = paraCV(basins, features, nPerBasin=nPerBasin, k=5)
    # print('R2:', paraR2)
    # data = RFData(basins, features, field='ff')
    # data.normalizeY()
    # data.normalizeX()
    
    # mask = np.logical_and(data.Yphys>=0, data.Yphys<=1)
    # data.Y[~mask] = np.nan
    # data.Yphys[~mask] = np.nan

    # checkR2 = 1 - np.nanvar(ypred-data.Y)/np.nanvar(data.Y)
    # print(checkR2)
    # R2_axis0 = 1 - np.nanvar(ypred-data.Y, axis=0)/np.nanvar(data.Y, axis=0)
    # print(R2_axis0)
    

    # print('BASIN CROSS-VALIDATION')
    # basinCV(basins, features, nPerBasin=nPerBasin)


    return



def evaluate_error(basins, features, highlight=None):
    data = RFData(basins, features)
    data.normalizeY()
    mu,sd = data.Yscale
    pred = np.load('CVpred.npy')
    pred = mu + sd*pred

    error = pred - data.Yphys

    mask = np.logical_and(data.Yphys>=0, data.Yphys<=1)
    
    pred[~mask] = np.nan
    glads = data.Yphys
    glads[~mask] = np.nan
    R2_by_para = 1 - np.nanvar(pred-glads, axis=0)/np.nanvar(glads, axis=0)
    R2_by_mesh = 1 - np.nanvar(pred-glads, axis=1)/np.nanvar(glads, axis=1)
    R2 = 1 - np.nanvar(pred-glads)/np.nanvar(glads)

    Thetaphys = np.loadtxt('../../issm/theta_physical.csv',
        delimiter=',', skiprows=1)
    Theta = np.loadtxt('../../issm/theta_standard.csv',
        delimiter=',', skiprows=1)
    names = np.loadtxt('../../issm/theta_physical.csv',
        delimiter=',', dtype=str, max_rows=1)

    # Plot parameters
    # ncols = 2
    # nrows = int(np.ceil(len(names)/2))
    # fig,axs = plt.subplots(nrows=nrows, ncols=ncols, sharey=True)
    # for i in range(len(names)):
    #     ax = axs.flat[i]
    #     ax.scatter(Theta[:, i], R2_by_para)
    #     ax.set_xlabel(names[i])
    #     ax.grid()
    #     ax.axhline(R2, color='k', linestyle='dashed', linewidth=1.5)
    
    #     if highlight:
    #         ax.scatter(Theta[highlight, i], R2_by_para[highlight], color='red')

    # fig.tight_layout()
    # fig.savefig('figures/R2_scatter.png', dpi=400)

    allN = np.array([])
    allNhat = np.array([])
    allY = np.array([])
    allYhat = np.array([])

    istart = 0
    for basin in basins:
        # basinData = RFData([basin], features)
        mesh = np.load(f'../../issm/{basin}/data/geom/mesh.npy', allow_pickle=True)
        levelset = np.load(f'../../issm/{basin}/data/geom/ocean_levelset.npy')
        fig,ax = plt.subplots()
        mtri = Triangulation(mesh['x'], mesh['y'], mesh['elements']-1)
        ngrounded = len(levelset[levelset==1])
        err = np.nan*np.ones(len(mesh['x']))
        err[levelset==1] = R2_by_mesh[istart:(istart+ngrounded)]
        # yi = glads[istart:(istart+ngrounded)]
        # yhati = pred[istart:(istart+ngrounded)]
        glads_basin = np.nan*np.ones(len(mesh['x']))
        glads_basin[levelset==1] = np.mean(glads[istart:(istart+ngrounded)], axis=1)
        istart += ngrounded

        err[glads_basin<0] = np.nan
        err[glads_basin>1] = np.nan
        # err[np.isnan(glads_basin)] = np.nan
        # err[err<=0] = np.nan

        Y = np.load(f'../../issm/{basin}/glads/ff.npy')[levelset>0,:]
        Yhat = np.load(f'data/CV_{basin}_f_rf.npy')
        N = np.load(f'data/CV_{basin}_N_glads.npy')
        Nhat = np.load(f'data/CV_{basin}_N_rf.npy')
        pc = ax.tripcolor(mtri, err, cmap=cmocean.cm.matter, vmin=0, vmax=1)
        ax.set_aspect('equal')

        mm = np.logical_and(Y<=1, Y>=0)
        # r2 = 1 - np.nanvar(yhati-yi)/np.nanvar(yi)
        r2 = 1 - np.nanvar(Yhat[mm] - Y[mm])/np.nanvar(Y[mm])
        r2N = 1 - np.nanvar(Nhat[mm] - N[mm])/np.nanvar(N[mm])


        Ymean = np.mean(Y, axis=1)
        Yhatmean = np.mean(Yhat, axis=1)
        Nmean = np.mean(N, axis=1)
        Nhatmean = np.mean(Nhat, axis=1)
        mmean = np.logical_and(Ymean<=1, Ymean>=0)

        r2mean = 1 - np.nanvar(Yhatmean[mmean] - Ymean[mmean])/np.nanvar(Ymean[mmean])
        r2Nmean = 1 - np.nanvar(Nhatmean[mmean] - Nmean[mmean])/np.nanvar(Nmean[mmean])

        ax.set_title(f'{basin} ($R^2$={r2:.3f})')
        print(basin, r2, r2N, r2mean, r2Nmean)
        fig.colorbar(pc, label=r'$R^2$')
        fig.savefig(f'figures/R2_map_{basin}.png', dpi=400)

        allN = np.concatenate((allN, N[mm]))
        allNhat = np.concatenate((allNhat, Nhat[mm]))
        allY = np.concatenate((allY, Y[mm]))
        allYhat = np.concatenate((allYhat, Yhat[mm]))
    
    fig,axs = plt.subplots(ncols = 2, figsize=(8, 4))

    ax = axs[0]
    fmin = 0.75
    ax.hexbin(allYhat, allY, bins=None, cmap=cmocean.cm.rain, gridsize=50,
        extent=(fmin, 1, fmin, 1))
    ax.set_xlabel('GlaDS Flotation Fraction (-)')
    ax.set_ylabel('Random Forest Flotation Fraction (-)')
    ax.grid()
    ax.set_aspect('equal')
    ax.set_xlim([fmin, 1])
    ax.set_ylim([fmin, 1])
    allR2 = 1 - np.nanvar(allYhat - allY)/np.nanvar(allY)
    ax.set_title('$R^2$ = {:.3f}'.format(allR2))


    ax = axs[1]
    hb = ax.hexbin(allNhat/1e6, allN/1e6, bins=None, cmap=cmocean.cm.rain, gridsize=50,
        extent=(0, 5, 0, 5))
    ax.set_xlabel('GlaDS Effective Pressure (MPa)')
    ax.set_ylabel('Random Forest Effective Pressure (MPa)')
    ax.grid()
    ax.set_aspect('equal')
    ax.set_xlim([0, 5])
    ax.set_ylim([0, 5])
    allR2 = 1 - np.nanvar(allNhat - allN)/np.nanvar(allN)
    ax.set_title('$R^2$ = {:.3f}'.format(allR2))

    fig.subplots_adjust(left=0.085, bottom=0.085, right=0.975, top=0.915, wspace=0.35)

    cb = fig.colorbar(hb, ax=axs, label='Counts (N={:.3e})'.format(len(allNhat)))
    fig.savefig('figures/hexbin.png', dpi=400)

        



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
    # theta: sheet cond, channel cond, r_b, l_c, A
    main(basins, features, field='ff', nPerBasin=50, k=10,
        feature_importance=True)
    # evaluate_error(basins, features, highlight=95)
