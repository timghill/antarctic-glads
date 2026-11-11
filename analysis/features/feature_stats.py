import numpy as np


basins = [
    'B-C',
    'C-Cp',
    'Cp-D',
    'Ep-F',
    'G-H',
    'J-Jpp',
    'Jpp-K',
]

features = np.load('features_B-C.pkl', allow_pickle=True)
feature_names = features.keys()

for feat in feature_names:
    print(f'{feat} >>>')
    y = np.array([])
    for basin in basins:
        y = np.concatenate((y, np.load(f'features_{basin}.pkl', allow_pickle=True)[feat]))
    print('\t', 
        np.format_float_scientific(np.nanmin(y), precision=3),
        np.format_float_scientific(np.nanmax(y), precision=3),
    )
