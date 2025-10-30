import argparse

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
from inversion import inversion
from m1qn3inversion import m1qn3inversion
from SetMarineIceSheetBC import SetMarineIceSheetBC

from matplotlib import pyplot as plt
from matplotlib.tri import Triangulation

# steps=[1, 2, 4]
engine = 'scipy'

def _load_N_fields(basin):
    levelset = np.load('../data/geom/ocean_levelset.npy')
    Nglads = np.zeros(len(levelset))
    Nglads[levelset>0] = np.load(f'../../../analysis/mean/data/pred_{basin}_N_glads.npy')
    Nrf = np.zeros(len(levelset))
    Nrf[levelset>0] = np.load(f'../../../analysis/mean/data/pred_{basin}_N_rf.npy')
    Ncv = np.zeros(len(levelset))
    Ncv[levelset>0] = np.load(f'../../../analysis/mean/data/CV_{basin}_N_rf.npy')

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
    # Npoc[Npoc<0] = 0
    
    Nglads[Nglads>pice] = pice[Nglads>pice]
    Nrf[Nrf>pice] = pice[Nrf>pice]
    Ncv[Ncv>pice] = pice[Ncv>pice]

    Nglads[Nglads<0.01*pice] = 0.01*pice[Nglads<0.01*pice]
    Nrf[Nrf<0.01*pice] = 0.01*pice[Nrf<0.01*pice]
    Ncv[Ncv<0.01*pice] = 0.01*pice[Ncv<0.01*pice]
    Npoc[Npoc<0.01*pice] = 0.01*pice[Npoc<0.01*pice]

    N = dict(glads=Nglads, rf=Nrf, cv=Ncv, poc=Npoc)
    return N

def set_para(effective_pressure, initialization=None):
    md = model()
    mesh = np.load('../data/geom/mesh.npy', allow_pickle=True)
    md = meshconvert(md, mesh['elements'], mesh['x'], mesh['y'])
    md = parameterize(md, '../../issm_parameterize.py')
    md = setflowequation(md, 'SSA', 'all')

    md.inversion=m1qn3inversion()
    # md.inversion = inversion()
    md.inversion.vx_obs=md.initialization.vx
    md.inversion.vy_obs=md.initialization.vy
    md.inversion.vel_obs=md.initialization.vel
    md.inversion.incomplete_adjoint = 0

    print('   Set boundary conditions')
    md=SetMarineIceSheetBC(md)
    md.basalforcings.floatingice_melting_rate = np.zeros((md.mesh.numberofvertices,1))
    md.basalforcings.groundedice_melting_rate = np.zeros((md.mesh.numberofvertices,1))
    md.thermal.spctemperature                 = md.initialization.temperature
    md.masstransport.spcthickness             = np.nan*np.ones((md.mesh.numberofvertices,1))

    md.friction.coefficient[:] = 225
    md.friction.coefficient[md.mask.ocean_levelset<0]=0
    md.friction.coupling = 3
    md.friction.effective_pressure = effective_pressure
    
    if initialization is not None:
        md.friction.coefficient = initialization
        #no friction applied on floating ice
        md.friction.coefficient[md.mask.ocean_levelset<0]=0

    
    md.friction.p = np.ones((md.mesh.numberofelements,1))
    md.friction.p[:] = 5
    md.friction.q = np.ones((md.mesh.numberofelements,1))
    md.friction.q[:] = 1

    # SSA solver parameters
    md.stressbalance.restol=0.01
    md.stressbalance.reltol=0.1
    md.stressbalance.abstol=np.nan
    md.stressbalance.maxiter=1000
    
    md.toolkits = toolkits()
    md.cluster = generic('name', gethostname(), 'np', 1)
    return md

def run_friction_inversion(effective_pressure, initialization=None,
    coefficients=None, min_para=1, max_para=1e4, x0=None):

    md = set_para(effective_pressure,initialization=initialization)

    # Set inversion-specific parameters
    md.inversion.iscontrol = 1
    md.inversion.maxsteps = 100
    md.inversion.maxiter = 100
    md.inversion.dxmin = 0.00001
    md.inversion.gttol = 1e-6
    md.verbose = verbose('control', True)

    if x0 is not None:
        md.friction.coefficient[md.friction.coefficient>0] = x0

    # Cost functions
    if coefficients is None:
        coefficients = [1, 1e-2, 1e-8]
    # print('Cost function coefficients:', coefficients)
    md.inversion.cost_functions=[101, 103, 501]
    md.inversion.cost_functions_coefficients=np.ones((md.mesh.numberofvertices,3))
    md.inversion.cost_functions_coefficients[:,0]=coefficients[0]
    md.inversion.cost_functions_coefficients[:,1]=coefficients[1]
    md.inversion.cost_functions_coefficients[md.inversion.vel_obs<0.1, 0:2] = 0
    md.inversion.cost_functions_coefficients[:,2]=coefficients[2]

    # Controls
    print('Setting min para:', min_para)
    md.inversion.control_parameters=['FrictionCoefficient']
    md.inversion.min_parameters=min_para*np.ones((md.mesh.numberofvertices,1))
    md.inversion.max_parameters=max_para*np.ones((md.mesh.numberofvertices,1))

    md = solve(md, 'Stressbalance')

    md.friction.coefficient = md.results.StressbalanceSolution.FrictionCoefficient
    return md

def Lcurve(effective_pressure, coefficients, alpha, initialization=None, **kwargs):
    nsteps = len(alpha)
    J = np.zeros([nsteps, 5])
    J[:, 0] = alpha
    for i in range(nsteps):
        coefi = coefficients.copy()
        ai = alpha[i]
        coefi[-1] = ai
        
        print(f'Alpha step {i}/{nsteps}: {ai:.1e}')
        md = run_friction_inversion(effective_pressure,
            initialization=initialization, coefficients=coefi,
            **kwargs)
        Jvalues = md.results.StressbalanceSolution.J
        C = md.results.StressbalanceSolution.FrictionCoefficient
        J[i, 1:] = Jvalues[-1,:]
    return J

def run_forward(friction, effective_pressure):
    md = set_para(effective_pressure, initialization=friction)

    md = solve(md, 'Stressbalance')

    model_vel = md.results.StressbalanceSolution.Vel
    return md

def run_inverse_scenarios(basin, **kwargs):
    Nfields = _load_N_fields(basin)

    if not os.path.exists('solutions/'):
        os.makedirs('solutions')

    coef_poc = run_friction_inversion(Nfields['poc'], **kwargs).friction.coefficient.squeeze()
    np.save('solutions/friction_coefficient_POC_nonlinear.npy', coef_poc)

    coef_poc = np.load('solutions/friction_coefficient_POC_nonlinear.npy').squeeze()
    calc_coef_glads = coef_poc * np.sqrt(Nfields['poc']/Nfields['glads'])
    calc_coef_rf = coef_poc * np.sqrt(Nfields['poc']/Nfields['rf'])

    Nglads = Nfields['glads']
    Nglads[np.isnan(Nglads)] = 1e6
    coef_glads = run_friction_inversion(Nfields['glads'], **kwargs).friction.coefficient.squeeze()
    np.save('solutions/friction_coefficient_glads_nonlinear.npy', coef_glads)

    coef_RF = run_friction_inversion(Nfields['rf'], **kwargs).friction.coefficient.squeeze()
    np.save('solutions/friction_coefficient_RF_nonlinear.npy', coef_RF)

def run_Lcurve_scenarios(basin, coefficients=None):
    log_min = -12
    log_max = -6
    nsteps = 13
    alpha = np.logspace(log_min, log_max, nsteps)
    Nfields = _load_N_fields(basin)
    if coefficients is None:
        coefficients = [1, 1e-2, 1e-8]

    if not os.path.exists('solutions'):
        os.makedirs('solutions')

    Jpoc = Lcurve(Nfields['poc'], coefficients, alpha)
    np.save('solutions/Jpoc.npy', Jpoc)

    Jglads = Lcurve(Nfields['glads'], coefficients, alpha)
    np.save('solutions/Jglads.npy', Jglads)

    Jrf = Lcurve(Nfields['rf'], coefficients, alpha)
    np.save('solutions/Jrf.npy', Jrf)

    plot_Lcurve_scenarios()

def plot_Lcurve_scenarios():
    Jpoc = np.load('solutions/Jpoc.npy')
    Jglads = np.load('solutions/Jglads.npy')
    Jrf = np.load('solutions/Jrf.npy')

    Js = [Jpoc, Jglads, Jrf]
    labels = ['POC', 'GlaDS', 'RF']
    fig,ax = plt.subplots()
    for i in range(len(Js)):
        J = Js[i]
        print(J)
        alpha = J[:,0]
        Jv = J[:, 1]
        Jr = J[:,-2]/alpha

        ax.loglog(Jr, Jv, marker='.', label=labels[i])
        nsteps = len(Jv)
        for j in range(0, nsteps, 2):
            ax.text(Jr[j], Jv[j], r'$\alpha = {:.1e}$'.format(alpha[j]), rotation=30)
    ax.grid()
    ax.set_xlabel(r'$\mathcal{J}_{\rm{reg}}$')
    ax.set_ylabel(r'$\mathcal{J}_{\rm{u}}$')
    ax.legend(loc='upper right')
    fig.savefig('solutions/Lcurve.png')

def run_forward_scenarios(basin):
    Nfields = _load_N_fields(basin)

    C_poc = np.load('solutions/friction_coefficient_POC_nonlinear.npy').squeeze()
    C_glads = np.load('solutions/friction_coefficient_glads_nonlinear.npy').squeeze()
    C_RF = np.load('solutions/friction_coefficient_RF_nonlinear.npy').squeeze()

    # 1. Baseline using POC model
    u_poc = run_forward(C_poc, Nfields['poc']).results.StressbalanceSolution.Vel.squeeze()
    np.save('solutions/u_poc_nonlinear.npy', u_poc)

    # 2. Use C_glads with N_glads
    u_glads_glads = run_forward(C_glads, Nfields['glads']).results.StressbalanceSolution.Vel.squeeze()
    np.save('solutions/u_glads_glads_nonlinear.npy', u_glads_glads)

    # 3. Use C_glads with Nrf
    u_glads_rf = run_forward(C_glads, Nfields['rf']).results.StressbalanceSolution.Vel.squeeze()
    np.save('solutions/u_glads_rf_nonlinear.npy', u_glads_rf)

    # 4. Use C_RF with Nrf
    u_rf_rf = run_forward(C_RF, Nfields['rf']).results.StressbalanceSolution.Vel.squeeze()
    np.save('solutions/u_rf_rf_nonlinear.npy', u_rf_rf)

    # 5. For completeness: use C_glads with Nrf
    u_rf_glads = run_forward(C_RF, Nfields['glads']).results.StressbalanceSolution.Vel.squeeze()
    np.save('solutions/u_rf_glads_nonlinear.npy', u_rf_glads)

    # 6. And for comparison: C_glads with Npoc
    u_glads_poc = run_forward(C_glads, Nfields['poc']).results.StressbalanceSolution.Vel.squeeze()
    np.save('solutions/u_glads_poc_nonlinear.npy', u_glads_poc)

    u_rf_poc = run_forward(C_RF, Nfields['poc']).results.StressbalanceSolution.Vel.squeeze()
    np.save('solutions/u_rf_poc_nonlinear.npy', u_rf_poc)

    # 7. Using the CV fields instead of full prediction
    u_glads_cv = run_forward(C_glads, Nfields['cv']).results.StressbalanceSolution.Vel.squeeze()
    np.save('solutions/u_glads_cv_nonlinear.npy', u_glads_cv)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('direction', type=str)
    parser.add_argument('basin', type=str)
    parser.add_argument('--coefficients', type=float, nargs='*')
    args = parser.parse_args()

    if args.direction=='inverse':
        run_inverse_scenarios(args.basin, coefficients=args.coefficients)
    elif args.direction=='forward':
        run_forward_scenarios(args.basin)
    elif args.direction=='Lcurve':
        run_Lcurve_scenarios(args.basin, coefficients=args.coefficients)
    elif args.direction=='Lplot':
        plot_Lcurve_scenarios()
    else:
        raise ValueError("Expected args.direction to be 'inverse' or 'forward', received '{}'".format(args.direction))
