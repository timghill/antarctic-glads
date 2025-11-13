
import numpy as np

from utils.issm.iceflow import run_forward


def main(basin, year):
    levelset = np.load(f'../data/geom/ocean_levelset.npy')
    present_levelset = np.load(f'../../{basin}/data/geom/ocean_levelset.npy')
    npara = 100
    nrun = 100

    thick = np.load('../data/geom/thick.npy')
    pice = 9.81*917*thick

    N_rf_present = np.zeros((len(levelset), npara))
    N_rf_present[present_levelset>0] = np.load(f'../../../analysis/parameters_full/data/pred_{basin}_N_rf.npy')
    
    N_rf_future = np.zeros((len(levelset), npara))
    N_rf_future[levelset>0] = np.load(f'../../../analysis/parameters_full/data/pred_{basin}_{year}_N_rf.npy')

    N_glads_present = np.zeros((len(levelset), npara))
    N_glads_present[present_levelset>0] = np.load(f'../../../analysis/parameters_full/data/pred_{basin}_N_glads.npy')

    N_glads_future = np.zeros((len(levelset), npara))
    N_glads_future[levelset>0] = np.load(f'../../../analysis/parameters_full/data/pred_{basin}_{year}_N_glads.npy')

    N_rf_mean = np.mean(N_rf_present, axis=1)
    N_glads_mean = np.mean(N_glads_present, axis=1)

    # No negative water pressure
    N_glads_future = np.minimum(N_glads_future, pice[:,None])
    N_glads_present = np.minimum(N_glads_present, pice[:,None])
    N_rf_future = np.minimum(N_rf_future, pice[:,None])
    N_rf_present = np.minimum(N_rf_present, pice[:,None])
    N_glads_mean = np.minimum(N_glads_mean, pice)
    N_rf_mean = np.minimum(N_rf_mean, pice)
    
    # No zero-effective-pressure
    N_glads_future = np.maximum(0.01*pice[:,None], N_glads_future)
    N_glads_present = np.maximum(0.01*pice[:,None], N_glads_present)
    N_rf_future = np.maximum(0.01*pice[:,None], N_rf_future)
    N_rf_present = np.maximum(0.01*pice[:,None], N_rf_present)
    N_glads_mean = np.maximum(0.01*pice, N_glads_mean)
    N_rf_mean = np.maximum(0.01*pice, N_rf_mean)

    # Load friction coefficient
    C_glads = np.load(f'../../{basin}/issm/solutions/friction_coefficient_glads_nonlinear.npy').squeeze()
    C_glads[levelset<0] = 0

    C_rf = np.load(f'../../{basin}/issm/solutions/friction_coefficient_RF_nonlinear.npy').squeeze()
    C_rf[levelset<0] = 0

    uu_glads = np.zeros(N_glads_future.shape)
    uu_rf = np.zeros(N_rf_future.shape)
    # for i in range(npara):
    for i in range(nrun):
        print('Calculate C, i=', i)
        Ci_glads = np.sqrt( N_glads_mean**(1./5.) / N_glads_present[:,i]**(1./5.) ) * C_glads
        ui = run_forward(Ci_glads, N_glads_future[:,i]).results.StressbalanceSolution.Vel.squeeze()
        uu_glads[:,i] = ui

        Ci_rf = np.sqrt( N_rf_mean**(1./5.) / N_rf_present[:,i]**(1./5.) ) * C_rf
        ui = run_forward(Ci_rf, N_rf_future[:,i]).results.StressbalanceSolution.Vel.squeeze()
        uu_rf[:,i] = ui
    
    np.save('solutions/u_glads_para_sensitivity.npy', uu_glads)
    np.save('solutions/u_rf_para_sensitivity.npy', uu_rf)


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
