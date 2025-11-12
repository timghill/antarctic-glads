
import numpy as np

from utils.issm.iceflow import run_forward


def main(basin, year):
    levelset = np.load(f'../data/geom/ocean_levelset.npy')
    present_levelset = np.load(f'../../{basin}/data/geom/ocean_levelset.npy')
    npara = 100
    nrun = 100
    
    # Load effective pressure
    ff = np.load('../glads/ff.npy')
    rhoi = 917
    thick = np.load('../data/geom/thick.npy')
    g = 9.81
    pice = rhoi*g*thick[:,None]
    N_glads_future = pice*(1 - ff)

    thick_present = np.load(f'../../{basin}/data/geom/thick.npy')
    pice_present = rhoi*g*thick_present[:,None]

    ff_present = np.load(f'../../{basin}/glads/ff.npy')
    N_glads_present = pice_present*(1 - ff_present)

    # N_glads_mean = np.zeros(len(levelset))
    # N_glads_mean[levelset>0] = np.load(f'../../../analysis/mean/data/pred_{basin}_{year}_N_glads.npy')

    N_glads_mean = np.zeros(len(levelset))
    N_glads_mean[present_levelset>0] = np.load(f'../../../analysis/mean/data/pred_{basin}_N_glads.npy')

    print(N_glads_future.shape)
    # N=0 for floating ice
    N_glads_future[levelset<0] = 0
    N_glads_mean[levelset<0] = 0
    # No negative water pressure
    N_glads_future = np.minimum(N_glads_future, pice)
    N_glads_present = np.minimum(N_glads_present, pice)
    N_glads_mean = np.minimum(N_glads_mean, pice.squeeze())
    # No zero-effective-pressure
    N_glads_future = np.maximum(0.01*pice, N_glads_future)
    N_glads_present = np.maximum(0.01*pice, N_glads_present)
    N_glads_mean = np.maximum(0.01*pice.squeeze(), N_glads_mean)

    # Load friction coefficient
    C_glads = np.load(f'../../{basin}/issm/solutions/friction_coefficient_glads_nonlinear.npy').squeeze()
    C_glads[levelset<0] = 0

    uu_calc_C = np.zeros(N_glads_future.shape)
    # for i in range(npara):
    for i in range(nrun):
        print('Calculate C, i=', i)
        Ni_present = N_glads_present[:,i]
        Ci = np.sqrt( N_glads_mean**(1./5.) / Ni_present**(1./5.) ) * C_glads

        ui = run_forward(Ci, N_glads_future[:,i]).results.StressbalanceSolution.Vel.squeeze()
        uu_calc_C[:,i] = ui
    
    np.save('solutions/u_glads_para_sensitivity_calc_C.npy', uu_calc_C)


    # uu_const_C = np.zeros(N_glads_future.shape)
    # # for i in range(npara):
    # for i in range(nrun):
    #     print('Constant C, i=', i)
    #     ui = run_forward(C_glads, N_glads_future[:,i]).results.StressbalanceSolution.Vel.squeeze()
    #     uu_const_C[:,i] = ui
    #     print('MAX:', np.quantile(ui, 0.95))
    # np.save('solutions/u_glads_para_sensitivity_const_C.npy', uu_const_C)





if __name__=='__main__':
    main('B-C', 2300)
