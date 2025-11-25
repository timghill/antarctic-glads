import argparse

# import os
# import sys
# import pickle
import numpy as np

from utils.issm.iceflow import run_forward

# ISSM_DIR = os.getenv('ISSM_DIR')
# sys.path.append(os.path.join(ISSM_DIR, 'bin/'))
# sys.path.append(os.path.join(ISSM_DIR, 'lib/'))
# from issmversion import issmversion

# import xarray as xr

# from model import *
# from triangle import triangle
# from setmask import setmask
# from parameterize import parameterize
# from setflowequation import setflowequation
# from generic import generic
# from socket import gethostname
# from solve import solve
# from bamg import bamg
# from InterpFromGridToMesh import InterpFromGridToMesh
# from verbose import verbose
# from toolkits import toolkits
# from socket import gethostname
# from meshconvert import meshconvert
# from inversion import inversion
# from m1qn3inversion import m1qn3inversion
# from SetMarineIceSheetBC import SetMarineIceSheetBC

# from matplotlib import pyplot as plt
# from matplotlib.tri import Triangulation


def _load_N_fields(basin, year=2300):
    future_levelset = np.load('../data/geom/ocean_levelset.npy')
    present_levelset = np.load(f'../../{basin}/data/geom/ocean_levelset.npy')
    
    present = basin.replace('_para', '')
    future = (f'{basin}_{year}')
    future = future.replace('_para', '')

    print('present:', present)
    print('future', future)

    rhow = 1023
    rhoice = 917
    g = 9.81
    present_thick = np.load(f'../../{present}/data/geom/thick.npy')
    future_thick = np.load(f'../../{future}/data/geom/thick.npy')
    bed = np.load('../data/geom/bed.npy')
    pice_present = present_thick*rhoice*g
    pice_future = future_thick*rhoice*g
    # pice = rhoice*g*thick
    pwater = -rhow*g*bed
    pwater[pwater<0] = 0
    N_poc_present = pice_present - pwater
    N_poc_future = pice_future - pwater

    pice = pice_future

    N_glads_present = np.zeros(len(present_levelset))
    N_glads_present[present_levelset>0] = np.load(f'../../../analysis/parameters_reduced/data/pred_{present}_N_glads.npy').mean(axis=1)
    # N_glads_present[future_levelset<0] = 0

    N_rf_present = np.zeros(len(present_levelset))
    N_rf_present[present_levelset>0] = np.load(f'../../../analysis/parameters_reduced/data/pred_{present}_N_rf.npy').mean(axis=1)

    N_cv_present = np.zeros(len(present_levelset))
    N_cv_present[present_levelset>0] = np.load(f'../../../analysis/parameters_reduced/data/CV_{present}_N_rf.npy').mean(axis=1)

    N_rf_future = np.zeros(len(future_levelset))
    N_rf_future[future_levelset>0] = np.load(f'../../../analysis/parameters_reduced/data/pred_{future}_N_rf.npy').mean(axis=1)

    N_glads_future = np.zeros(len(future_levelset))
    N_glads_future[future_levelset>0] =  np.load(f'../../../analysis/parameters_reduced/data/pred_{future}_N_glads.npy').mean(axis=1)

    # Enforce effective pressure caps
    
    
    N_glads_present[N_glads_present>pice] = pice[N_glads_present>pice]
    N_rf_present[N_rf_present>pice] = pice[N_rf_present>pice]
    N_cv_present[N_cv_present>pice] = pice[N_cv_present>pice]
    N_rf_future[N_rf_future>pice] = pice[N_rf_future>pice]
    N_poc_present[N_poc_present>pice] = pice[N_poc_present>pice]
    N_poc_future[N_poc_future>pice] = pice[N_poc_future>pice]
    N_glads_future[N_glads_future>pice] = pice[N_glads_future>pice]

    N_glads_present[N_glads_present<0.01*pice] = 0.01*pice[N_glads_present<0.01*pice]
    N_rf_present[N_rf_present<0.01*pice] = 0.01*pice[N_rf_present<0.01*pice]
    N_cv_present[N_cv_present<0.01*pice] = 0.01*pice[N_cv_present<0.01*pice]
    N_rf_future[N_rf_future<0.01*pice] = 0.01*pice[N_rf_future<0.01*pice]
    N_glads_future[N_glads_future<0.01*pice] = 0.01*pice[N_glads_future<0.01*pice]
    N_poc_present[N_poc_present<0.01*pice] = 0.01*pice[N_poc_present<0.01*pice]
    N_poc_future[N_poc_future<0.01*pice] = 0.01*pice[N_poc_future<0.01*pice]


    N = dict(
        glads_present=N_glads_present,
        rf_present=N_rf_present,
        cv_present=N_cv_present,
        rf_future=N_rf_future,
        glads_future=N_glads_future,
        poc_future=N_poc_future,
        poc_present=N_poc_present,
    )
    return N

def run_scenarios(basin, year):
    Nfields = _load_N_fields(basin, year)

    C_poc = np.load(f'../../{basin}/issm/solutions/friction_coefficient_POC_nonlinear.npy').squeeze()
    C_glads = np.load(f'../../{basin}/issm/solutions/friction_coefficient_glads_nonlinear.npy').squeeze()
    C_RF = np.load(f'../../{basin}/issm/solutions/friction_coefficient_RF_nonlinear.npy').squeeze()

    levelset = np.load('../data/geom/ocean_levelset.npy')
    C_poc[levelset<0] = 0
    C_glads[levelset<0] = 0
    C_RF[levelset<0] = 0

    u_poc_present = run_forward(C_poc, Nfields['poc_present']).results.StressbalanceSolution.Vel.squeeze()
    np.save('solutions/u_poc_present.npy', u_poc_present)

    u_poc_future = run_forward(C_poc, Nfields['poc_future']).results.StressbalanceSolution.Vel.squeeze()
    np.save('solutions/u_poc_future.npy', u_poc_future)

    u_glads_present = run_forward(C_glads, Nfields['glads_present']).results.StressbalanceSolution.Vel.squeeze()
    np.save('solutions/u_glads_present.npy', u_glads_present)
    print('max:', np.quantile(u_glads_present, 0.98))

    u_rf_present = run_forward(C_RF, Nfields['rf_present']).results.StressbalanceSolution.Vel.squeeze()
    np.save('solutions/u_rf_present.npy', u_rf_present)
    print('max:', np.quantile(u_rf_present, 0.98))

    u_cv_present = run_forward(C_glads, Nfields['cv_present']).results.StressbalanceSolution.Vel.squeeze()
    np.save('solutions/u_cv_present.npy', u_cv_present)
    print('max:', np.quantile(u_cv_present, 0.98))

    u_rf_future = run_forward(C_RF, Nfields['rf_future']).results.StressbalanceSolution.Vel.squeeze()
    np.save('solutions/u_rf_future.npy', u_rf_future)
    print('max:', np.quantile(u_rf_future, 0.98))

    u_glads_future = run_forward(C_glads, Nfields['glads_future']).results.StressbalanceSolution.Vel.squeeze()
    np.save('solutions/u_glads_future.npy', u_glads_future)
    print('max:', np.quantile(u_glads_future, 0.98))
    return

if __name__=='__main__':
    run_scenarios('Cp-D_para', '2300')
