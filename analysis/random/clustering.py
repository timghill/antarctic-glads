import numpy as np
from matplotlib import pyplot as plt

def plot_cluster(basins):
    feature_list = None
    feature_keys = [
        'bed',
        'surface',
        'thickness',
        'grounding_line_distance',
        'basal_melt',
        # 'potential',
        # 'binned_flow_accumulation',
    ]
    basin_num = []
    i=0
    for basin in basins:
        basin_matrix = None
        features = np.load(f'features_{basin}.pkl', allow_pickle=True)
        for key in feature_keys:
            if basin_matrix is None:
                basin_matrix = features[key]
            else:
                basin_matrix = np.vstack((basin_matrix, features[key]))
        
        print(basin_matrix.shape)
        if feature_list is None:
            feature_list = basin_matrix.copy()
        else:
            feature_list = np.hstack((feature_list, basin_matrix))
        basin_num.extend(0*basin_matrix[0] + i)
        i += 1
    
    print('feature_list.shape:', feature_list.shape)
    basin_num = np.array(basin_num)
    print(basin_num.shape)

    vmin = np.min(feature_list, axis=1)
    vmax = np.max(feature_list, axis=1)
    Z = (feature_list - vmin[:,None])/(vmax[:,None] - vmin[:,None])
    print(Z.min(axis=1))
    print(Z.max(axis=1))
    U,S,V = np.linalg.svd(Z, full_matrices=False)
    contrib_var = S**2/np.sum(S**2)
    cumul_var = np.cumsum(contrib_var)
    print('contrib_var:', contrib_var)

    print('U:', U.shape)
    print('V:', V.shape)

    fig,ax = plt.subplots()
    skip = 10
    ax.scatter(V[0, ::skip], V[1, ::skip], s=2, c=basin_num[::skip])
    fig.savefig('cluster.png', dpi=400)



if __name__=='__main__':
    basins = [
        'G-H',
        'F-G',
        'Ep-F',
        'Cp-D',
        'C-Cp',
        # 'B-C',
        # 'Jpp-K',
        # 'J-Jpp',
    ]
    plot_cluster(basins)