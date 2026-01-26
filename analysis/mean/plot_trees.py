import numpy as np
from matplotlib import pyplot as plt
from sklearn.tree import plot_tree


import os
import pickle
import time
import math
import itertools

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




def trainRF(basins, feature_keys, Xscale=None, Yscale=None, nPerBasin=1000,
    index=None):    
    print('trainRF::', index)
    rfData = RFData(basins, feature_keys, index=index)
    rfData.normalizeX(scale=Xscale)
    rfData.normalizeY(scale=Yscale)
    
    # Only train and evaluate where N>0 and pw>0
    mask = np.logical_and(rfData.Yphys>=0, rfData.Yphys<=1)

    X = rfData.Xphys[mask]
    Y = rfData.Yphys[mask]

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
    return regr

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
]    

basins = [
        'G-H',
        # 'F-G',  # TODO check outputs, look like numerical issues
        'Ep-F', # jobs not done
        'Cp-D',
        'C-Cp',
        'B-C',
        'Jpp-K',
        'J-Jpp',# TODO check outputs, look like numerical issues
    ]

print('Fitting random forest')
rf = trainRF(basins, features, nPerBasin=10000)

max_depth = 2
ntrees = 100

for i in range(2):
    print(f'Tree {i}')
    t = rf.estimators_[i]
    fig,ax = plt.subplots(figsize=(6,3))
    plot_tree(t, max_depth=max_depth, ax=ax,
        feature_names=features, fontsize=7, label='all',
        proportion=True, precision=3)
    # fig.tight_layout()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.savefig(f'trees/t_noscale_{i:03d}.png', dpi=600)

    if i==0:
        fig.savefig(f'../../manuscript/D01.png', dpi=300)
        fig.savefig(f'../../manuscript/D01.pdf')
