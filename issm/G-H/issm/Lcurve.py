import os
import sys
import pickle

ISSM_DIR = os.getenv('ISSM_DIR')
sys.path.append(os.path.join(ISSM_DIR, 'bin/'))
sys.path.append(os.path.join(ISSM_DIR, 'lib/'))
from issmversion import issmversion

import xarray as xr

from model import *
from triangle import triangle
from setmask import setmask
from parameterize import parameterize
from setflowequation import setflowequation
from generic import generic
from socket import gethostname
from solve import solve
from bamg import bamg
from InterpFromGridToMesh import InterpFromGridToMesh
from verbose import verbose
from toolkits import toolkits
from socket import gethostname
from meshconvert import meshconvert
from m1qn3inversion import m1qn3inversion
from SetMarineIceSheetBC import SetMarineIceSheetBC

from matplotlib import pyplot as plt
from matplotlib.tri import Triangulation

# steps=[1, 2, 3, 4]
engine = 'scipy'

def run_friction_inversion(coupling=2, effective_pressure=None, initialization=None, alpha=1e-7):
    md = model()
    mesh = np.load('../data/geom/mesh.npy', allow_pickle=True)
    md = meshconvert(md, mesh['elements'], mesh['x'], mesh['y'])
    md = parameterize(md, '../../issm_parameterize.py')
    md = setflowequation(md, 'SSA', 'all')

    md.inversion=m1qn3inversion()
    md.inversion.vx_obs=md.initialization.vx
    md.inversion.vy_obs=md.initialization.vy
    md.inversion.vel_obs=md.initialization.vel

    print('   Set boundary conditions')
    md=SetMarineIceSheetBC(md)
    md.basalforcings.floatingice_melting_rate = np.zeros((md.mesh.numberofvertices,1))
    md.basalforcings.groundedice_melting_rate = np.zeros((md.mesh.numberofvertices,1))
    md.thermal.spctemperature                 = md.initialization.temperature
    md.masstransport.spcthickness             = np.nan*np.ones((md.mesh.numberofvertices,1))

    md.friction.coupling = coupling
    if effective_pressure is not None:
        md.friction.effective_pressure = effective_pressure
    
    if initialization is not None:
        md.friction.coefficient = initialization
        #no friction applied on floating ice
        md.friction.coefficient[md.mask.ocean_levelset<0]=0

    md.inversion.iscontrol = 1
    md.inversion.maxsteps = 100
    md.inversion.maxiter = 100
    md.inversion.dxmin = 0.00001
    md.inversion.gttol = 1e-6
    md.verbose = verbose('control', True)

    # Cost functions
    md.inversion.cost_functions=[101, 103, 501]
    md.inversion.cost_functions_coefficients=np.ones((md.mesh.numberofvertices,3))
    md.inversion.cost_functions_coefficients[:,0]=1
    md.inversion.cost_functions_coefficients[:,1]=1e-2
    md.inversion.cost_functions_coefficients[md.inversion.vel_obs<0.1, 0:2] = 0
    md.inversion.cost_functions_coefficients[:,2]=alpha

    # Controls
    md.inversion.control_parameters=['FrictionCoefficient']
    md.inversion.min_parameters=1*np.ones((md.mesh.numberofvertices,1))
    md.inversion.max_parameters=2000*np.ones((md.mesh.numberofvertices,1))

    # SSA solver parameters
    md.stressbalance.restol=0.01
    md.stressbalance.reltol=0.1
    md.stressbalance.abstol=np.nan
    md.stressbalance.maxiter=1000
    
    md.toolkits = toolkits()
    md.cluster = generic('name', gethostname(), 'np', 1)
    md = solve(md, 'Stressbalance')

    md.friction.coefficient = md.results.StressbalanceSolution.FrictionCoefficient

    return md

def run_Lcurve(log_alpha_min, log_alpha_max, nsteps, fname=None):
    levelset = np.load('../data/geom/ocean_levelset.npy')
    Nglads = np.ones(len(levelset))
    Nglads[levelset>0] = np.load('../../../analysis/mean/data/pred_G-H_N_glads.npy')
    Nrf = np.ones(len(levelset))
    Nrf[levelset>0] = np.load('../../../analysis/mean/data/pred_G-H_N_rf.npy')
    
    # Enforce effective pressure caps
    rhoice = 917
    g = 9.81
    
    bed = np.load('../data/geom/bed.npy')
    thick = np.load('../data/geom/thick.npy')
    pice = rhoice*g*thick
    rhow = 1023

    pwater = -rhow*g*bed
    pwater[pwater<0] = 0
    Npoc = pice - pwater
    Npoc[Npoc<0] = 0

    alpha_values = np.logspace(log_alpha_min, log_alpha_max, nsteps)
    print('L-Curve for alpha values:', alpha_values)

    J = np.zeros([nsteps, 5])
    J[:, 0] = alpha_values
    for i in range(nsteps):
        alpha = alpha_values[i]
        print(f'Alpha step {i}/{nsteps}: {alpha:.1e}')
        md = run_friction_inversion(3, Npoc, alpha=alpha)
        Jvalues = md.results.StressbalanceSolution.J
        print('Jvalues:', Jvalues.shape)
        C = md.results.StressbalanceSolution.FrictionCoefficient
        J[i, 1:] = Jvalues[-1,:]
    
    if fname is not None:
        np.save(fname, J)
    return J

def plot_Lcurve(fname):
    J = np.load(fname)
    print('J:', J)
    fig,ax = plt.subplots()
    alpha = J[:,0]
    Jv = J[:, 1]
    Jr = J[:,-2]/alpha
    ax.loglog(Jr, Jv, marker='.')
    nsteps = len(Jv)
    for i in range(0, nsteps, 2):
        ax.text(Jr[i], Jv[i], r'$\alpha = {:.1e}$'.format(alpha[i]), rotation=30)
    ax.grid()
    ax.set_xlabel(r'$\mathcal{J}_{\rm{reg}}$')
    ax.set_ylabel(r'$\mathcal{J}_{\rm{u}}$')
    fig.subplots_adjust(left=0.15, bottom=0.15, right=0.9, top=0.9)
    fig.savefig('Lcurve_POC.png', dpi=400)

if __name__=='__main__':
    # run_Lcurve(log_alpha_min=-10, log_alpha_max=-6, nsteps=9, fname='Jpoc.npy')

    plot_Lcurve('Jpoc.npy')

