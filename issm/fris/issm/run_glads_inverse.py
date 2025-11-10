import numpy as np

from utils.issm import iceflow

N = np.load('../glads/glads_N.npy')
thick = np.load('../data/geom/thick.npy')
rhoice = 91
g = 9.81
pice = rhoice*g*thick

N = np.maximum(0.05*pice, N)
N = np.minimum(pice, N)

C = np.load('C_glads.npy')

# md = iceflow.run_friction_inversion(N, coefficients=(1, 1e-2, 1e-8), B=np.load('B.npy'))
md = iceflow.run_friction_inversion(N, coefficients=(1, 1e-3, 5e-9), B=None,
    max_para=500, initialization=150*np.ones(N.shape))
C = md.friction.coefficient.squeeze()
np.save('C_glads.npy', C)

vel = md.results.StressbalanceSolution.Vel.squeeze()
np.save('vel.npy', vel)