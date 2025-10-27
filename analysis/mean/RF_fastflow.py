
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
from scipy.stats import gaussian_kde

from utils.RF import RFData, AISData

def _getSlipSpeed(basins):
    ub = []
    for basin in basins:
        fname= f'../../issm/{basin}/data/lanl-mali/basal_velocity_mali.npy'
        levelset = np.load(f'../../issm/{basin}/data/geom/ocean_levelset.npy')
        uBasin = np.load(fname)
        ub.extend(uBasin[levelset==1])
    return np.array(ub)

def _percentError(Yhat, Y):
    Ylower = np.quantile(Y, 0.1)
    res = Yhat - Y
    mape = np.mean(np.abs(res[Y>=Ylower])/Y[Y>=Ylower])
    return mape


def trainRF(basins, feature_keys, Xscale=None, Yscale=None, nPerBasin=1000,
    feature_importance=False, ubThreshold=None):    
    rfData = RFData(basins, feature_keys)
    rfData.normalizeX(scale=Xscale)
    rfData.normalizeY(scale=Yscale)
    
    # Only train and evaluate where N>0 and pw>0
    mask = np.logical_and(rfData.Yphys>=0, rfData.Yphys<=1)

    if ubThreshold:
        # print(f'Finding subset with ub>={ubThreshold} m/a')
        ub = _getSlipSpeed(basins)
        # print('X:', rfData.X.shape)
        # print('ub:', ub.shape)
        mask = np.logical_and(mask, ub>=ubThreshold)

    X = rfData.X[mask]
    Y = rfData.Y[mask]
    # print('Remaining X:', X.shape)


    # Choose a random subset of points
    if nPerBasin:
        rng = np.random.default_rng()
        randIndices = rng.choice(np.arange(len(Y)), len(basins)*nPerBasin)
        # print('len(randIndices):', len(randIndices))

        Xsub = X[randIndices]
        Ysub = Y[randIndices]
    else:
        Xsub = X.copy()
        Ysub = Y.copy()

    # scikitlearn random forest
    # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
    regr = RandomForestRegressor(max_depth=10)
    # print('Fitting random forest')
    regr.fit(Xsub, Ysub)
    # print('Done fitting')

    if feature_importance:
        print('Permutation importance')
        result = permutation_importance(
            regr, Xsub, Ysub, n_repeats=10, random_state=42, n_jobs=1
        )
        print(result)

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.bar(np.arange(len(feature_keys)), result['importances_mean'], yerr=result['importances_std'])
        ax.set_xticks(np.arange(len(feature_keys)), feature_keys, rotation=45, ha='right')
        ax.set_title(f'Feature importances, ubThreshold={ubThreshold}')
        ax.set_ylabel('Mean accuracy decrease')
        fig.subplots_adjust(bottom=0.4)
        fig.savefig(f'figures/feature_importance_ub_{ubThreshold}.png', dpi=400)

    return rfData, regr

def crossVal(basins, feature_keys, nPerBasin=1000, ubThresholdTrain=None,
    ubThresholdTest=None):
    # Load RF Data
    rfData = RFData(basins, feature_keys)
    rfData.normalizeX(scale=None)
    rfData.normalizeY(scale=None)

    YYhat = []
    YY = []

    for i in range(len(basins)):
        # Test-train split
        trainBasins = basins[:i] + basins[i+1:]
        testBasin = basins[i]
        # print('CV basins', trainBasins)

        # Retrain Random Forest with training split
        cvData, cvRegr = trainRF(trainBasins, feature_keys, 
            nPerBasin=nPerBasin, Xscale=rfData.Xscale, Yscale=rfData.Yscale,
            ubThreshold=ubThresholdTrain)
        
        # Test data
        testData = RFData([testBasin], feature_keys)
        testData.normalizeX(scale=rfData.Xscale)
        testData.normalizeY(scale=rfData.Yscale)

        mask = np.logical_and(testData.Yphys>=0, testData.Yphys<=1)
        if ubThresholdTest:
            # print(f'Finding subset with ub>={ubThresholdTest} m/a')
            ub = _getSlipSpeed([testBasin])
            # print('X:', rfData.X.shape)
            # print('ub:', ub.shape)
            mask = np.logical_and(mask, ub>=ubThresholdTest)

        Zhat = cvRegr.predict(testData.X)
        mu,sd = rfData.Yscale
        Yhat = mu + sd*Zhat
        Yphys = testData.Yphys

        # Keep track of overall data & model
        YY.extend(Yphys[mask])
        YYhat.extend(Yhat[mask])

        R2 = 1 - np.nanvar(Yhat[mask]-Yphys[mask])/np.nanvar(Yphys[mask])
        mape = _percentError(Yhat[mask], Yphys[mask])
        print(f'{testBasin} R2={R2:.3f}, MAPE={mape:.3f}')

        ones = np.ones(Yphys.shape)
        print('Shreve percent error:', _percentError(ones[mask], Yphys[mask]))
    
    YYhat = np.array(YYhat)
    YY = np.array(YY)
    R2all = 1 - np.nanvar(YYhat - YY)/np.nanvar(YY)
    MAPEall = _percentError(YYhat, YY)
    print(f'Overall R2: {R2all:.3f}, MAPE: {MAPEall:.3f}')
    print('Shreve percent error:',
        _percentError(np.ones(YY.shape), YY))


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
        # 'binned_flow_accumulation',
    ]

    ubThreshold = 200
    # rfData, regr = trainRF(basins, features, 
    #     nPerBasin=1000, feature_importance=True,
    #     ubThreshold=None)
    # ubData, ubRF = trainRF(basins, features, 
    #     nPerBasin=None, feature_importance=True, 
    #     ubThreshold=ubThreshold)

    print(f'\nubThresholdTrain=None, ubThresholdTest=None')
    crossVal(basins, features, nPerBasin=2000, 
        ubThresholdTrain=None, ubThresholdTest=None)

    print(f'\nubThresholdTrain=None, ubThresholdTest={ubThreshold}')
    crossVal(basins, features, nPerBasin=2000, 
        ubThresholdTrain=None, ubThresholdTest=ubThreshold)


    print(f'\nubThresholdTrain={ubThreshold}, ubThresholdTest=None')
    crossVal(basins, features, nPerBasin=2000, 
        ubThresholdTrain=ubThreshold, ubThresholdTest=None)

    print(f'\nubThresholdTrain={ubThreshold}, ubThresholdTest={ubThreshold}')
    crossVal(basins, features, nPerBasin=2000, 
        ubThresholdTrain=None, ubThresholdTest=ubThreshold)

    

