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

from utils.issm import iceflow

def run_friction_inversion(effective_pressure, friction=None,
    coefficients=None):

    md = iceflow.set_para(effective_pressure,initialization=friction)
    md.friction.coefficient = friction

    # Set inversion-specific parameters
    md.inversion.iscontrol = 1
    md.inversion.maxsteps = 100
    md.inversion.maxiter = 10
    md.inversion.dxmin = 0.00001
    md.inversion.gttol = 1e-6
    md.verbose = verbose('control', True)

    # Cost functions
    if coefficients is None:
        coefficients = [1e3, 1, 1e-12]
    # print('Cost function coefficients:', coefficients)
    md.inversion.cost_functions=[101, 103, 502]
    md.inversion.cost_functions_coefficients=np.ones((md.mesh.numberofvertices,3))
    md.inversion.cost_functions_coefficients[:,0]=coefficients[0]
    md.inversion.cost_functions_coefficients[:,1]=coefficients[1]
    md.inversion.cost_functions_coefficients[:,2]=coefficients[2]

    md.inversion.cost_functions_coefficients[md.inversion.vel_obs<0.1, 0:2] = 0
    md.inversion.cost_functions_coefficients[md.mask.ocean_levelset>0, 0:3] = 0
    # Controls
    md.inversion.control_parameters=['MaterialsRheologyBbar']
    md.inversion.min_parameters=cuffey(273)**np.ones((md.mesh.numberofvertices,1))
    md.inversion.max_parameters=cuffey(243)*np.ones((md.mesh.numberofvertices,1))

    md.inversion.min_parameters[md.mask.ocean_levelset>0] = md.materials.rheology_B[md.mask.ocean_levelset>0,None]
    md.inversion.max_parameters[md.mask.ocean_levelset>0] = md.materials.rheology_B[md.mask.ocean_levelset>0,None]

    md.inversion.min_parameters[md.mask.ocean_levelset<0] = md.materials.rheology_B[md.mask.ocean_levelset<0,None]
    md.inversion.max_parameters[md.mask.ocean_levelset<0] = md.materials.rheology_B[md.mask.ocean_levelset<0,None]

    md = solve(md, 'Stressbalance')

    md.materials.rheology_B=md.results.StressbalanceSolution.MaterialsRheologyBbar
    return md

if __name__=='__main__':
    N = np.load('../glads/glads_N.npy')
    
    # Enforce effective pressure caps
    rhoice = 917
    g = 9.81
    friction = np.load('C_glads.npy')
    md = run_friction_inversion(N, friction=friction)
    B = md.materials.rheology_B.squeeze()
    np.save('B.npy', B)

    vel = md.results.StressbalanceSolution.Vel.squeeze()
    np.save('vel_stiffness.npy', vel)