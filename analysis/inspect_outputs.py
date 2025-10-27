import numpy as np
from matplotlib import pyplot as plt
from matplotlib.tri import Triangulation
import cmocean

def main(basin):
    # Y = np.load(f'../issm/{basin}/glads/ff.npy')[:,jobid-1]
    Y = np.nanmean(np.load(f'../issm/{basin}/glads/ff.npy'), axis=1)
    print(Y.shape)
    mesh = np.load(f'../issm/{basin}/data/geom/mesh.npy', allow_pickle=True)
    print(mesh['numberofvertices'])
    mtri = Triangulation(mesh['x'], mesh['y'], mesh['elements']-1)
    levelset = np.load(f'../issm/{basin}/data/geom/ocean_levelset.npy')
    Y[levelset<1] = np.nan
    print(levelset)

    fig,ax = plt.subplots()
    pc = ax.tripcolor(mtri, Y, vmin=0, vmax=1, cmap=cmocean.cm.dense) 
    ax.set_aspect('equal')
    fig.colorbar(pc, label='Flotation fraction')
    ax.set_title(basin)
    fig.savefig(f'{basin}.png', dpi=400)

if __name__=='__main__':
    main('F-G')
    main('J-Jpp')
    main('Ep-F')
