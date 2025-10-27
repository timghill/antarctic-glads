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
from cuffey import cuffey

from matplotlib import pyplot as plt
from matplotlib.tri import Triangulation

# steps=[1, 2, 3, 4]
engine = 'scipy'

def run_stiffness_inversion(coupling=2, effective_pressure=None, initialization=None, friction=None):
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
    
    if friction is not None:
        md.friction.coefficient = friction

    # if initialization is not None:
    #     # md.friction.coefficient = initialization
    #     #no friction applied on floating ice
    #     # md.friction.coefficient[md.mask.ocean_levelset<0]=0
    #     md.materials.

    md.inversion.iscontrol = 1
    md.inversion.maxsteps = 100
    md.inversion.maxiter = 25
    md.inversion.dxmin = 0.00001
    md.inversion.gttol = 1e-6
    md.verbose = verbose('control', True)

    # Cost functions
    md.inversion.cost_functions=[101, 502]
    md.inversion.cost_functions_coefficients=np.ones((md.mesh.numberofvertices,2))
    md.inversion.cost_functions_coefficients[:,0]=1000
    # md.inversion.cost_functions_coefficients[:,1]=1e-2
    md.inversion.cost_functions_coefficients[md.inversion.vel_obs<0.1, 0] = 0
    # md.inversion.cost_functions_coefficients[:,2]=1e-7
    md.inversion.cost_functions_coefficients[:,1] = 1e-16

    # Controls
    md.inversion.control_parameters = ['MaterialsRheologyBbar']
    md.inversion.min_parameters = cuffey(273) * np.ones((md.mesh.numberofvertices, 1))
    md.inversion.max_parameters = cuffey(200) * np.ones((md.mesh.numberofvertices, 1))

    # SSA solver parameters
    md.stressbalance.restol=0.01
    md.stressbalance.reltol=0.1
    md.stressbalance.abstol=np.nan
    md.stressbalance.maxiter=1000
    
    md.toolkits = toolkits()
    md.cluster = generic('name', gethostname(), 'np', 1)

    mds = md.extract(md.mask.ocean_levelset<0)
    mds = solve(mds, 'Stressbalance')

    md.materials.rheology_B[mds.mesh.extractedvertices]=mds.results.StressbalanceSolution.MaterialsRheologyBbar.squeeze()


    return md.materials.rheology_B

if __name__=='__main__':

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

    coef_poc = np.load('friction_coefficient_POC.npy').squeeze()
    B = run_stiffness_inversion(coupling=3, effective_pressure=Npoc, friction=coef_poc).squeeze()
    np.save('B.npy', B)
    # np.save('friction_coefficient_POC.npy', coef_poc)
    # coef_poc = np.load('friction_coefficient_POC.npy').squeeze()
    
    # Nglads[Nglads>pice] = pice[Nglads>pice]
    # Nrf[Nrf>pice] = pice[Nrf>pice]

    # Nglads[Nglads<0.01*pice] = 0.01*pice[Nglads<0.01*pice]
    # Nrf[Nrf<0.01*pice] = 0.01*pice[Nrf<0.01*pice]

    # calc_coef_glads = coef_poc * np.sqrt(Npoc/Nglads)
    # calc_coef_rf = coef_poc * np.sqrt(Npoc/Nrf)

    # coef_glads = run_friction_inversion(3, Nglads, initialization=calc_coef_glads)
    # np.save('friction_coefficient_glads.npy', coef_glads)

    # coef_RF = run_friction_inversion(3, Nrf, initialization=calc_coef_rf)
    # np.save('friction_coefficient_RF.npy', coef_RF)
