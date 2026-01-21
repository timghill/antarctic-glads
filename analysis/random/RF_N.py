
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

class RFData:
    def __init__(self, basins, featureKeys):
        self.Xphys = self.loadFeatures(basins, featureKeys)

        self.Yphys = self.loadAverageModelOutputs(basins)

        self.mask = self.getMask(basins)

    def loadFeatures(self, basins, featureKeys):
        feature_list = None
        for basin in basins:
            basin_matrix = None
            features = np.load(f'features_{basin}.pkl', allow_pickle=True)
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
        
    def loadAverageModelOutputs(self, basins):
        modelOutputs = []
        for basin in basins:
            # Load levelset to find where ice is grounded
            levelset = np.load(
                os.path.join(f'../../issm/{basin}',
                    'data/geom/ocean_levelset.npy')
            )
            # Add model outputs to list
            outputs = np.load(f'../../issm/{basin}/glads/N.npy')[levelset>0,:]
            # meanOutputs = np.nanmean(outputs, axis=1) # Take average over perturbed parameters
            meanOutputs = outputs[:, 95]
            modelOutputs.extend(meanOutputs) 
        return np.array(modelOutputs)
    
    def getMask(self, basins):
        modelOutputs = []
        for basin in basins:
            # Load levelset to find where ice is grounded
            levelset = np.load(
                os.path.join(f'../../issm/{basin}',
                    'data/geom/ocean_levelset.npy')
            )
            # Add model outputs to list
            outputs = np.load(f'../../issm/{basin}/glads/ff.npy')[levelset>0,:]
            meanOutputs = np.nanmean(outputs, axis=1) # Take average over perturbed parameters
            # meanOutputs = outputs[:, 50]
            modelOutputs.extend(meanOutputs) 
        modelOutputs = np.array(modelOutputs)
        mask = np.logical_and(modelOutputs>=0, modelOutputs<=1)
        return mask
    
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



def trainRF(basins, feature_keys, Xscale=None, Yscale=None, nPerBasin=1000,
    feature_importance=False):    
    rfData = RFData(basins, feature_keys)
    rfData.normalizeX(scale=Xscale)
    rfData.normalizeY(scale=Yscale)
    
    # Only train and evaluate where N>0 and pw>0
    # mask = np.logical_and(rfData.Yphys>=0, rfData.Yphys<=1)
    mask = rfData.mask

    print(rfData.X.shape)
    print(rfData.Y.shape)
    print(rfData.Yphys.shape)
    X = rfData.X[mask]
    Y = rfData.Y[mask]

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

    Xsub = X[randIndices]
    Ysub = Y[randIndices]


    # scikitlearn random forest
    # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
    regr = RandomForestRegressor(max_depth=10)
    print('Fitting random forest')
    regr.fit(Xsub, Ysub)

    print('Predicting random forest')
    Zhat = regr.predict(X)
    mu,sd = rfData.Yscale
    Yhat = mu + sd*Zhat
    Yphys = rfData.Yphys[mask]
    # R2 = 1 - np.var(Zhat-Z)/np.var(Z)
    # R2 = r2_score(Z, Zhat)
    # print('R2:', R2)
    R2 = 1 - np.var(Yhat - Yphys)/np.var(Yphys)
    print('R2:', R2)
    print('Done training')


    if feature_importance:
        print('Permutation importance')
        result = permutation_importance(
            regr, Xsub, Ysub, n_repeats=10, random_state=42, n_jobs=1
        )
        print(result)

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.bar(np.arange(len(feature_keys)), result['importances_mean'], yerr=result['importances_std'])
        # fig, ax = plt.subplots()
        # forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
        ax.set_xticks(np.arange(len(feature_keys)), feature_keys, rotation=45, ha='right')
        ax.set_title("Feature importances using permutation on full model")
        ax.set_ylabel("Mean accuracy decrease")
        fig.subplots_adjust(bottom=0.4)
        fig.savefig('figures_N/feature_importance.png', dpi=400)

    return rfData, regr


def crossVal(basins, feature_keys, nPerBasin=1000):
    # Load all the data for the correct scalings    
    rfData = RFData(basins, feature_keys)
    rfData.normalizeX(scale=None)
    rfData.normalizeY(scale=None)

    for i in range(len(basins)):
        trainBasins = basins[:i] + basins[i+1:]
        testBasin = basins[i]
        print('CV basins', trainBasins)
        cvData, cvRegr = trainRF(trainBasins, feature_keys, 
            nPerBasin=nPerBasin, Xscale=rfData.Xscale, Yscale=rfData.Yscale)
        mesh = np.load(f'../../issm/{testBasin}/data/geom/mesh.npy', allow_pickle=True)
        levelset = np.load(f'../../issm/{testBasin}/data/geom/ocean_levelset.npy')
        # N = loadModelOutputs([testBasin])
        Yfull = np.zeros(mesh['numberofvertices'])
        testData = RFData([testBasin], feature_keys)
        testData.normalizeX(scale=rfData.Xscale)
        testData.normalizeY(scale=rfData.Yscale)
        Yfull[levelset>0] = testData.Yphys
        Yfull[levelset<=0] = np.nan

        # mask = np.logical_and(testData.Yphys>=0, testData.Yphys<=1)
        mask = testData.mask
        # X = cvData.X[mask]
        # Y = cvData.Y[mask]

        Zhat = cvRegr.predict(testData.X)
        mu,sd = rfData.Yscale
        Yhat = mu + sd*Zhat
        Y = testData.Y
        Yphys = testData.Yphys

        np.save(f'data_N/CV_{testBasin}.npy', Yhat)

        Yhatfull = np.zeros(mesh['numberofvertices'])
        Yhatfull[levelset>0] = Yhat
        Yhatfull[levelset<=0] = np.nan

        fig,axs = plt.subplots(nrows=2, ncols=2, sharex=False, sharey=False)
        mtri = Triangulation(mesh['x'], mesh['y'], mesh['elements']-1)
        Nmap = axs[0,0].tripcolor(mtri, Yfull, cmap=cmocean.cm.haline, vmin=0, vmax=5e6)
        axs[0,1].tripcolor(mtri, Yhatfull, cmap=cmocean.cm.haline, vmin=0, vmax=5e6)
        diffmap = axs[1,0].tripcolor(mtri, Yhatfull-Yfull, cmap=cmocean.cm.balance, vmin=-2e6, vmax=2e6)

        fig.subplots_adjust(left=0.0625, right=0.925, bottom=0.1, top=0.85,
            hspace=0.3, wspace=0.3)

        cbar1 = fig.colorbar(Nmap, ax=axs[0], label='Effective pressure (Pa)')
        cbar2 = fig.colorbar(diffmap, ax=axs[1], label=r'$\Delta$ Effective Pressure (Pa)')

        axs[0,0].set_title('GlaDS')
        axs[0,1].set_title('RF prediction')
        axs[1,0].set_title('RF - GlaDS')

        axs[1,1].scatter(Yphys[mask], Yhat[mask], s=0.5, alpha=0.2)
        axs[1,1].set_xlim([0, 5e6])
        axs[1,1].set_ylim([0, 5e6])
        axs[1,1].set_xlabel('GlaDS')
        axs[1,1].set_ylabel('RF prediction')
        axs[1,1].grid()

        r2 = 1 - np.nanvar(Yhat[mask]-Yphys[mask])/np.nanvar(Yphys[mask])

        # print('R2: weighted', r2Weighted)
        # print('R2: unweighted', r2Unweighted)
        print('CV R2:', r2)
        fig.suptitle(f'{testBasin}, R2={r2:.3f}')

        for ax in axs.flat[:3]:
            ax.set_aspect('equal')
        
        msk = Yfull.copy()
        zifull = (Yfull - mu)/sd
        msk[np.abs(zifull)<2] = np.nan
        # msk[zifull>=-mu/sd] = np.nan
        # msk[zifull>mu] = np.nan
        # print(np.where(np.abs(zifull)>1))
        # print(np.where(zifull<-mu/sd))
        # for ax in axs.flat[:3]:
            # ax.tripcolor(mtri, msk, cmap=cmocean.cm.haline, vmin=0.5, vmax=1, hatch='/')
        fig.savefig(f'figures_N/CV_{testBasin}.png', dpi=400)


def predictContinent(rfData, rfRegr, feature_keys):
    # feats = np.load('features_AIS.pkl', mmap_mode='r', allow_pickle=True)
    # X, scaleX = loadFeatures(basins, feature_keys)


    # LOAD AIS FEATURES
    # feature_list = None
    # i=0
    stride = 4
    # for basin in basins:
    feature_matrix = []
    features = np.load('features_AIS.pkl', allow_pickle=True)
    mask = ~np.isnan(features['bed'][::stride, ::stride])
    for key in feature_keys:
        # if feature_matrix is None:
            feature_matrix.append(features[key][::stride, ::stride][mask])
        # else:
            # feature_matrix = np.vstack((feature_matrix, features[key][::stride, ::stride][mask]))
    feature_matrix = np.array(feature_matrix).T

    print('features_matrix.shape:', feature_matrix.shape)
    xmin,xmax = rfData.Xscale
    XAIS = (feature_matrix - xmin)/(xmax-xmin)

    print('XAIS.shape:', XAIS.shape)
    # XAIS = XAIS[:, ::10000]
    # print('XAIS:', XAIS.shape)
    Yhat = regr.predict(XAIS)

    mu,sd = rfData.Yscale
    YhatPhys = mu + sd*Yhat

    AISpred = np.nan*np.zeros(mask.shape)
    AISpred[mask] = YhatPhys
    AISpred = np.flipud(AISpred)

    np.save('data_N/AISpred.npy', AISpred)

    fig,ax = plt.subplots()
    pc = ax.pcolormesh(AISpred, vmin=0, vmax=5e6, cmap=cmocean.cm.haline)
    ax.set_aspect('equal')
    fig.colorbar(pc, label='Effective pressure (Pa)')
    fig.savefig('figures_N/AISpred.png', dpi=400)
    
def predictBasins(rfData, rfRegr, feature_keys, basins):
    for basin in basins:
        testData = RFData([basin], feature_keys)
        testData.normalizeX(scale=rfData.Xscale)
        testData.normalizeY(scale=rfData.Yscale)
        Yhat = rfRegr.predict(testData.X)
        mu,sd = rfData.Yscale
        Ypred = mu + sd*Yhat
        np.save(f'data_N/pred_{basin}_N.npy', Ypred)



if __name__=='__main__':
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

    rfData, regr = trainRF(basins, features, nPerBasin=10000, feature_importance=False)
    crossVal(basins, features, nPerBasin=10000)
    predictBasins(rfData, regr, features, basins)
    # predictContinent(rfData, regr, features)

