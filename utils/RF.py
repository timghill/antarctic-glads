
import os
import numpy as np

from matplotlib import pyplot as plt
from matplotlib.tri import Triangulation
import cmocean
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, explained_variance_score
from sklearn.inspection import permutation_importance
import scipy.interpolate
from scipy.stats import gaussian_kde

class RFData:
    def __init__(self, basins, featureKeys, field='ff', index=None):
        self.field = field
        self.features = featureKeys
        self.Xphys = self.loadFeatures(basins, featureKeys)

        self.Yphys = self.loadAverageModelOutputs(basins, field=field, index=index)

    def loadFeatures(self, basins, featureKeys):
        feature_list = None
        for basin in basins:
            basin_matrix = None
            features = np.load(f'../features/features_{basin}.pkl', allow_pickle=True)
            for key in featureKeys:
                if basin_matrix is None:
                    basin_matrix = features[key][:,None]
                else:
                    basin_matrix = np.hstack((basin_matrix, features[key][:,None]))   

                basin_matrix[np.isnan(basin_matrix)] = 0
                basin_matrix[np.isinf(basin_matrix)] = 0
            
            if feature_list is None:
                feature_list = basin_matrix.copy()
            else:
                feature_list = np.vstack((feature_list, basin_matrix))
        return feature_list
        
    def loadAverageModelOutputs(self, basins, field='ff', index=None):
        modelOutputs = []
        mask = []
        for basin in basins:
            # Load levelset to find where ice is grounded
            levelset = np.load(
                os.path.join(f'../../issm/{basin}',
                    'data/geom/ocean_levelset.npy')
            )
            # Add model outputs to list
            try:
                outputs = np.load(f'../../issm/{basin}/glads/{field}.npy')[levelset>0,:]
                if index is None:
                    meanOutputs = np.nanmean(outputs, axis=1) # Take average over perturbed parameters
                    # print('Computing ensemble mean')
                else:
                    meanOutputs = outputs[:, index]
                    # print('Loading index', index)
                modelOutputs.extend(meanOutputs) 
            except:
                print(f'Basin {basin} has no GlaDS outputs, returning nan')
                modelOutputs.extend(np.nan*levelset[levelset>0])

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


class AISData:
    def __init__(self, featureKeys, stride=1, file='features_AIS.pkl'):
        self.Xphys, self.mask = self.loadFeatures(featureKeys, file, stride=stride)
        self.features = featureKeys
        self.stride = stride
    
    def loadFeatures(self, featureKeys, file, stride=1):
        feature_matrix = []
        features = np.load(file, allow_pickle=True)
        mask = ~np.isnan(features['bed'][::stride, ::stride])
        for key in featureKeys:
                feature_matrix.append(features[key][::stride, ::stride][mask])
        feature_matrix = np.array(feature_matrix).T
        return feature_matrix, mask

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


class RFDataPara:
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
            features = np.load(f'../features/features_{basin}.pkl', allow_pickle=True)
            for key in featureKeys:
                if basin_matrix is None:
                    basin_matrix = features[key][:,None]
                else:
                    basin_matrix = np.hstack((basin_matrix, features[key][:,None]))            
            
            if feature_list is None:
                feature_list = basin_matrix.copy()
            else:
                feature_list = np.vstack((feature_list, basin_matrix))
        return feature_list.astype(np.float32)
    
    def loadParameters(self):
        Thetaphys = np.loadtxt('../../issm/theta_physical.csv',
            delimiter=',', skiprows=1)
        Theta = np.loadtxt('../../issm/theta_standard.csv',
            delimiter=',', skiprows=1)
        names = np.loadtxt('../../issm/theta_physical.csv',
            delimiter=',', dtype=str, max_rows=1)
        self.Thetaphys = Thetaphys.astype(np.float32)
        self.Theta = Theta.astype(np.float32)
        self.Thetanames = names
        return (Thetaphys.astype(np.float32), Theta.astype(np.float32), names)
        
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
        return np.array(modelOutputs).astype(np.float32)
    
    def normalizeX(self, scale=None):
        if scale is None:
            xmin = np.min(self.Xphys, axis=0).astype(np.float32)
            xmax = np.max(self.Xphys, axis=0).astype(np.float32)
            scale = (xmin, xmax)
        xmin, xmax = scale
        X = (self.Xphys - xmin)/(xmax-xmin)
        self.X = X.astype(np.float32)
        self.Xscale = scale
        return X.astype(np.float32), scale
    
    def normalizeY(self, scale=None):
        if scale is None:
            mu = np.mean(self.Yphys).astype(np.float32)
            sd = np.std(self.Yphys).astype(np.float32)
            scale = (mu, sd)
        mu,sd = scale
        Y = (self.Yphys - mu)/sd
        self.Y = Y.astype(np.float32)
        self.Yscale = scale
        return Y.astype(np.float32), scale
