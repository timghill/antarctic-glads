import time
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.tri import Triangulation
import cmocean
from scipy.interpolate import griddata

from utils.issm import iceflow

def main(basin):
    # Identify a simulation to use as "control" for friction inversion
    Ncv = np.load(f'../../../analysis/parameters/data/CV_{basin}_N_rf.npy')


    levelset = np.load(f'../data/geom/ocean_levelset.npy')

    # N minimum
    pice = 917*9.8*np.load('../data/geom/thick.npy')[levelset>0][:,None]
    pice = np.repeat(pice, Ncv.shape[1], axis=1)
    Nmin = 0.01*pice
    Ncv[Ncv<0.01*pice] = 0.01*pice[Ncv<0.01*pice]

    t0 = time.perf_counter()
    Nmed = np.median(Ncv, axis=1)
    Ndist = np.linalg.norm(Ncv - Nmed[:,None], axis=0)
    t1 = time.perf_counter()
    simnum = np.argmin(Ndist)
    print('Median simulation:', simnum)

    Nctrl = np.zeros(len(levelset))
    Nctrl[levelset>0] = Ncv[:, simnum]

    md = iceflow.run_friction_inversion(Nctrl,
        coefficients=[1, 1e-3, 1e-4])
    C = md.friction.coefficient
    np.save('C.npy', C)
    C = np.load('C.npy').squeeze() 

    # Do forward runs
    print('Control forward run')
    md = iceflow.run_forward(C, Nctrl)
    uctrl = md.results.StressbalanceSolution.Vel.squeeze()
    np.save('uctrl.npy', uctrl)

    # Perturbed parameter ensemble
    print('Perturbed parameter ensemble')
    nvalues = Ncv.shape[1]
    nvalues = 5
    
    UU = np.zeros((md.mesh.numberofvertices, nvalues))
    # for i in range(nvalues):
    for i in range(5):
        Ni = np.zeros(len(levelset))
        Ni[levelset>0] = Ncv[:,i]
        # Ni[Ni<0] = 100e3

        print('Parameter combination', i+1)
        Ci = C * np.sqrt(Nctrl/Ni)
        _md = iceflow.run_forward(Ci, Ni)
        _u = _md.results.StressbalanceSolution.Vel
        UU[:,i] = _u.squeeze()
    np.save('uu.npy', UU)
    return UU
    
def plot(basin):
    mesh = np.load('../data/geom/mesh.npy', allow_pickle=True)
    leveset = np.load('../data/geom/ocean_levelset.npy')
    mtri = Triangulation(mesh['x'], mesh['y'], mesh['elements']-1)

    ss,xx,yy = np.load('../data/geom/flowline_00.npy')

    vx = np.load('../data/geom/vx.npy')
    vy = np.load('../data/geom/vy.npy')
    vv = np.sqrt(vx**2 + vy**2)
    meshxy = (mesh['x'], mesh['y'])
    xy = (xx, yy)
    C = np.load('C.npy').squeeze()

    interp = lambda z: griddata(meshxy, z, xy, method='linear')
    uobs = interp(vv)
    Cinterp = interp(C)
    
    u_mesh_ctrl = np.load('uctrl.npy')
    uctrl = interp(u_mesh_ctrl)

    u_mesh_all = np.load('uu.npy')[:, :5]
    u_mesh_lower = np.quantile(u_mesh_all, 0.025, axis=1)
    u_mesh_upper = np.quantile(u_mesh_all, 0.975, axis=1)
    ulower = interp(u_mesh_lower)
    uupper = interp(u_mesh_upper)

    fig,ax = plt.subplots()
    ax.plot(ss/1e3, uobs, color='k', linewidth=1.5, label='Observed')
    ax.plot(ss/1e3, uctrl, color='red', label='Ctrl')
    ax.fill_between(ss/1e3, ulower, uupper, color='red', alpha=0.2, edgecolor='none', label='Ensemble')

    ax.legend(loc='upper right')

    ax.grid()
    ax.set_xlabel('Distance from terminus (km)')
    ax.set_ylabel('Speed (m/year)')
    ax2 = ax.twinx()
    ax2.plot(ss/1e3, Cinterp)
    ax2.set_ylabel('Friction coefficient')
    fig.savefig(f'flowline_u_{basin}.png', dpi=400)




if __name__=='__main__':
    basin = 'G-H'
    main(basin)
    plot(basin)