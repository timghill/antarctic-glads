
import pickle
import os
import time
import numpy as np
import shap
from sklearn.ensemble import RandomForestRegressor

from matplotlib import pyplot as plt
from matplotlib.tri import Triangulation
import cmocean

from utils.RF import RFData


def compute_shap(model, features, basin):
    # shapval = compute_shap(model, [basin])
    
    # Load data
    testData = RFData([basin], features, index=None)

    # Predictions
    Xtest = testData.Xphys
    t0 = time.perf_counter()
    explainer = shap.Explainer(model)
    shap_values = explainer(Xtest)
    t1 = time.perf_counter()
    print('Computed SHAP in {:.2f}s'.format(t1-t0))
    np.save(f'data/SHAP_{basin}.npy', shap_values.values)
    return shap_values.values

def fit_model(basins, features):
    print('Fitting model')

    # Fit the model
    stride = 50
    trainData = RFData(basins, features, index=None)
    mask = np.logical_and(trainData.Yphys>=0, trainData.Yphys<=1)
    Xtrain = trainData.Xphys[mask][::stride]
    Ytrain = trainData.Yphys[mask][::stride]
    model = RandomForestRegressor(max_depth=10)
    model.fit(Xtrain, Ytrain)
    with open('model.pkl', 'wb') as rfout:
        pickle.dump(model, rfout)
    return model

def main(basins, features):
    if not os.path.exists('model.pkl'):
        print('FITTING MODEL')
        fit_model(basins, features)
    model = np.load('model.pkl', allow_pickle=True)

    for basin in basins:
        print(basin)
        if not os.path.exists(f'data/SHAP_{basin}.npy'):
            print('COMPUTING SHAP')
            compute_shap(model, features, basin)

        mesh = np.load(f'../../issm/{basin}/data/geom/mesh.npy', 
            allow_pickle=True)
        levelset = np.load(f'../../issm/{basin}/data/geom/ocean_levelset.npy')
        shapval = np.zeros((len(mesh['x']), len(features)), dtype=np.float32)
        shapval[levelset>0] = np.load(f'data/SHAP_{basin}.npy')
        print('shapval:', shapval.shape)
        mtri = Triangulation(mesh['x'], mesh['y'], mesh['elements']-1)
        fig,axs = plt.subplots(3,3,figsize=(8,8))
        for i in range(len(features)):
        # for i in range(3):
            print('i =', i)
            ax = axs.flat[i]
            tpc = ax.tripcolor(mtri, shapval[:,i], vmin=-0.1, vmax=0.1,
                cmap=cmocean.cm.balance)
            ax.set_aspect('equal')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines[['left', 'right', 'top', 'bottom']].set_visible(False)
            ax.set_title(features[i])
        
        # axs.flat[-1].set_visible(False)
        axs.flat[-1].set_xticks([])
        axs.flat[-1].set_yticks([])
        axs.flat[-1].spines[['left', 'right', 'top', 'bottom']].set_visible(False)
        cax = axs[2,1].inset_axes((0,-0.2, 1, 0.1))
        cax.set_visible(True)
        cbar = fig.colorbar(tpc, cax=cax, orientation='horizontal',
            label='SHAP value')
        
        fig.savefig(f'figures/SHAP_{basin}.png', dpi=400)
            

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
        # 'binned_flow_accumulation',
    ]
    main(basins, features)