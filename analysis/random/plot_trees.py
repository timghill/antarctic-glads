import numpy as np
from matplotlib import pyplot as plt
from sklearn.tree import plot_tree
from sklearn.ensemble import RandomForestRegressor

from utils.RF import RFData

# The default tree
rf = np.load('rf.pkl', allow_pickle=True)

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

max_depth = 3
ntrees = 100

# for i in range(10):
#     print(f'Tree {i}')
#     t = rf.estimators_[i]
#     fig,ax = plt.subplots()
#     plot_tree(t, max_depth=max_depth, ax=ax,
#         feature_names=features)
#     fig.savefig(f'trees/t{i:03d}.png', dpi=600)

# With no thickness
features = [
    'bed',
    'surface',
    # 'thickness',
    'grounding_line_distance',
    'basal_melt',
    'potential',
    'surface_slope',
    'bed_slope',
    'potential_slope',
]
data = RFData(basins, features)
data.normalizeX()
data.normalizeY()

mask = np.logical_and(data.Yphys>=0, data.Yphys<=1)
Xtrain = data.X[mask][::10]
Ytrain = data.Y[mask][::10]

rf = RandomForestRegressor(max_depth=10, max_features=4)
rf.fit(Xtrain, Ytrain)

for i in range(10):
    print(f'Tree {i}')
    t = rf.estimators_[i]
    fig,ax = plt.subplots()
    plot_tree(t, max_depth=max_depth, ax=ax,
        feature_names=features)
    fig.savefig(f'trees/t_nothick_{i:03d}.png', dpi=600)