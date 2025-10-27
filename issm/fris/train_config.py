import numpy as np
from scipy import stats
import os
import pathlib

## EXPERIMENTAL DESIGN

exp = 'amery'                   # Experiment name, used for paths

# Name the parameter
theta_names = [  
                r'$k_{\rm{s}}$',
                r'$k_{\rm{c}}$',
                r'$r_{\rm{b}}$',
                r'$l_{\rm{c}}$',
                r'$E$',
]

# Define lower, upper bounds in log10 space
theta_bounds = np.array([
                    [-2, 0],                # Sheet conductivity
                    [np.log10(5e-3), 
                        np.log10(0.5)],     # Channel conductivity
                    [1, 2],                 # Bed bump aspect ratio
                    [0, 2],                 # Channel-sheet width
                    [-1, 1],                # Ice-flow factor
])

# How to sample parameters
#   None: Use default sampling (minimized centered discrepancy LH)
#   stats.qmc.QMCEngine
theta_sampler = None

## PATHS
base = pathlib.Path(__file__).parent.resolve()
sim_dir = os.path.join(base, 'glads')
# analysis_dir = os.path.join(base, 'analysis/')
exp_dir = os.path.join(base, 'glads')
mesh = os.path.abspath(os.path.join(sim_dir, '../data/geom/mesh.npy'))

# Paths to use for parameter design
X_physical = os.path.join(exp_dir, 'theta_physical.csv')
X_log = os.path.join(exp_dir, 'theta_log.csv')
X_standard = os.path.join(exp_dir, 'theta_standard.csv')

## ISSM-GlaDS CONFIGURATION
#   Tell ISSM how to set hydrology parameters given the parameter file
def parser(md, jobid):
    X = np.loadtxt(X_physical, delimiter=',', skiprows=1)
    k_s,k_c,r_bed,l_c,E, = X[jobid-1, :]
    h_bed = 0.5
    vertices = np.ones((md.mesh.numberofvertices, 1))
    md.hydrology.sheet_conductivity = k_s*vertices
    md.hydrology.channel_conductivity = k_c*vertices
    md.hydrology.bump_height = h_bed*vertices
    md.hydrology.cavity_spacing = h_bed*r_bed
    md.hydrology.channel_sheet_width = l_c
    md.hydrology.rheology_B_base = (E*2.4e-24)**(-1./3.)*vertices
    md.hydrology.creep_open_flag = 0
    md.hydrology.omega = 1/2000
    md.hydrology.englacial_void_ratio = 1e-5
    md.hydrology.istransition = 1
    md.hydrology.sheet_alpha = 3./2.
    md.hydrology.sheet_beta = 2.
    return md
    

## GP CONFIGURATION
m = 100             # Number of simulations for fitting
# data_dir = os.path.join(analysis_dir, 'data/')
# figures = os.path.join(analysis_dir, 'figures/')
Y_physical = os.path.join(sim_dir, '{exp}_ff.npy'.format(exp=exp))
