import os
import pickle
import time

import numpy as np

from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.tri import Triangulation
from matplotlib import path
import cmocean
from sklearn.ensemble import RandomForestRegressor
from scipy import interpolate
import xarray as xr

from utils.RF import RFDataPara


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


def trainRF(basins, features, Xscale=None, Yscale=None, nPerSim=100,
    feature_importance=False):
    """
    Train Random Forest on entire dataset
    """
    data = RFDataPara(basins, features, field='ff')
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

def parabasinCV(basins, features, nPerSim=100, field='ff', k=1):
    """
    Cross-validation over 100x parameter combinations and Nx basins
    """
    data = RFDataPara(basins, features, field=field)
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
        np.save(f'data/CV_{basin}_N_rf.npy', N_RF)
        istart += testData.X.shape[0]

        # Interpolating to bedmachine grid
        outline = np.load(f'../../data/ANT_Basins/basin_{testBasin}.npy')
        basinPath = path.Path(outline, closed=True)
        mesh = np.load(f'../../issm/{basin}/data/geom/mesh.npy', allow_pickle=True)
        print('Interpolating to bedmachine')
        xi = mesh['x'][levelset==1]
        yi = mesh['y'][levelset==1]
        xx,yy,Yhat_bm = _interp2bedmachine(xi, yi, ypredphys)
        _,_,Yphys_bm = _interp2bedmachine(xi, yi, testData.Yphys)
        _,_,Nrf_bm = _interp2bedmachine(xi, yi, N_RF)
        _,_,Nglads_bm = _interp2bedmachine(xi, yi, N_glads)

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
    
    print('SUMMARY')
    mask = np.logical_and(data.Yphys>=0, data.Yphys<=1)
    summR2 = 1 - np.var(data.Y[mask] - YYpred[mask])/np.var(data.Y[mask])
    print(summR2)
    return YYpred, summR2

def predictBasins(regr, train_basins, test_basins, features,field='ff'):
    """
    Cross-validation over 100x parameter combinations and Nx basins
    """
    data = RFDataPara(train_basins, features, field=field)
    data.normalizeX()
    data.normalizeY()
    mu,sd = data.Yscale
    # Keep track of all preds for overall CV R2
    YYpred = np.zeros(data.Y.shape)

    rhoi = 917
    g = 9.81

    for j,basin in enumerate(test_basins):
        print(basin)
        # Make train/test subsets and normalize in the same
        # way as the whole data
        testBasin = test_basins[j]
        testData = RFDataPara([testBasin], features, field=field)
        testData.normalizeX(scale=data.Xscale)
        testData.normalizeY(scale=data.Yscale)
        print(testData.X.shape)
        print(testData.Theta.shape)
        
        # print('X:', testData.X[:10])
        # print('T:', testData.Theta[:10])
        # XX,TT = np.meshgrid(testData.X, testData.Theta)
        nT = testData.Theta.shape[0]
        dT = testData.Theta.shape[1]
        nX = testData.X.shape[0]
        dX = testData.X.shape[1]
        XX = np.tensordot(np.ones(nT), testData.X, axes=0).reshape((nT*nX, dX), order='C')
        TT = np.tensordot(np.ones(nX), testData.Theta, axes=0).reshape((nT*nX, dT), order='F')
        # print('XX:', XX.shape)
        # print(XX[:10])
        # print('TT:', TT.shape)
        # print(TT[:10])
        Xpred = np.hstack((XX, TT))
        print('Xpred:', Xpred.shape)
        Ypred = regr.predict(Xpred)
        print('std mean:', np.mean(Ypred))
        Ypred_phys = mu + sd*Ypred
        print('phys mean:', np.mean(Ypred_phys))
        Ypred_phys = Ypred_phys.reshape(testData.Y.shape, order='F')

        levelset = np.load(f'../../issm/{basin}/data/geom/ocean_levelset.npy')
        thick = np.load(f'../../issm/{basin}/data/geom/thick.npy')
        pice = rhoi*g*thick[levelset>0,None]
        Npred_phys = pice*(1-Ypred_phys)
        np.save(f'data/pred_{basin}_f_rf.npy', Ypred_phys)
        np.save(f'data/pred_{basin}_N_rf.npy', Npred_phys)
        np.save(f'data/pred_{basin}_f_glads.npy', testData.Yphys)
        Nglads = pice*(1 - testData.Yphys)
        np.save(f'data/pred_{basin}_N_glads.npy', Nglads)

        mask = np.logical_and(testData.Yphys>=0, testData.Yphys<=1)
        r2 = 1 - np.nanvar(Ypred_phys[mask] - testData.Yphys[mask])/np.nanvar(testData.Yphys[mask])
        print('r2:', r2)
    return


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


    # print('Cross-validation')
    # Ypred, R2 = parabasinCV(basins, features, nPerSim=nPerSim, k=k, field=field)

    # print('Predicting for all basins')
    # predictBasins(basins, features, field=field)

    return


if __name__=='__main__':
    basins = [
        'G-H',
        'Ep-F', # jobs not done
        'Cp-D',
        'C-Cp',
        'B-C',
        'Jpp-K',
        'J-Jpp',# TODO check outputs, look like numerical issues
    ]
    testBasins = [
        'G-H',
        'Cp-D',
        'C-Cp',
        'B-C',
        'Jpp-K',
        'J-Jpp',
        'Ep-F',
        'G-H_2050',
        'Cp-D_2300',
        'C-Cp_2300',
        'B-C_2300',
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
    ]
    # theta: sheet cond, channel cond, r_b, l_c, A
    # main(basins, features, field='ff', nPerSim=250, k=10)
    # evaluate_error(basins, features, highlight=95)
    # regr = np.load('model.pkl', allow_pickle=True)
    # predictBasins(regr, basins, testBasins, features, field='ff')

