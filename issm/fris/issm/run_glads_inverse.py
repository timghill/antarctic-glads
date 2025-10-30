import numpy as np

from utils.issm import iceflow

N = np.load('../glads/glads_N.npy')
thick = np.load('../data/geom/thick.npy')
rhoice = 91
g = 9.81
pice = rhoice*g*thick

md = iceflow.run_friction_inversion(N, coefficients=(1, 1e-2, 1e-8), B=np.load('B.npy'))
C = md.friction.coefficient.squeeze()
np.save('C_glads.npy', C)
