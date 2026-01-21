import numpy as np
from matplotlib import pyplot as plt
from matplotlib.tri import Triangulation
import cmocean

def main(basins, years):
    print('f present, f future, N present, N future')

    all_f_glads = np.array([])
    all_f_rf = np.array([])
    all_N_glads = np.array([])
    all_N_rf = np.array([])
    all_u_glads = np.array([])
    all_u_rf = np.array([])

    for basin,future in zip(basins, years):

        mesh = np.load(f'../../issm/{basin}/data/geom/mesh.npy', allow_pickle=True)
        present_mask = np.load(f'../../issm/{basin}/data/geom/ocean_levelset.npy')
        shape = (len(present_mask),)
        future_mask = np.load(f'../../issm/{basin}_{future}/data/geom/ocean_levelset.npy')
        
        # glads_f_present = np.nan*np.zeros(shape)
        # glads_f_present[present_mask>0] = np.load(f'data/pred_{basin}_f_glads.npy')
        glads_f_present = np.nanmean(np.load(f'../../issm/{basin}/glads/ff.npy'), axis=1)
        glads_N_present = np.nan*np.zeros(shape)
        glads_N_present[present_mask>0] = np.load(f'data/pred_{basin}_N_glads.npy')

        # glads_f_future = np.nan*np.zeros(shape)
        # glads_f_future[future_mask>0] = np.load(f'data/pred_{basin}_{future}_f_glads.npy')
        glads_f_future = np.nanmean(np.load(f'../../issm/{basin}_{future}/glads/ff.npy'), axis=1)
        glads_N_future = np.nan*np.zeros(shape)
        glads_N_future[future_mask>0] = np.load(f'data/pred_{basin}_{future}_N_glads.npy')

        rf_f_present = np.nan*np.zeros(shape)
        rf_f_present[present_mask>0] = np.load(f'data/pred_{basin}.npy')
        rf_N_present = np.nan*np.zeros(shape)
        rf_N_present[present_mask>0] = np.load(f'data/pred_{basin}_N_rf.npy')

        cv_f_present = np.nan*np.zeros(shape)
        cv_f_present[present_mask>0] = np.load(f'data/CV_{basin}.npy')
        cv_N_present = np.nan*np.zeros(shape)
        cv_N_present[present_mask>0] = np.load(f'data/CV_{basin}_N_rf.npy')

        rf_f_future = np.nan*np.zeros(shape)
        rf_f_future[future_mask>0] = np.load(f'data/pred_{basin}_{future}.npy')
        rf_N_future = np.nan*np.zeros(shape)
        rf_N_future[future_mask>0] = np.load(f'data/pred_{basin}_{future}_N_rf.npy')

        u_rf_future = np.load(f'../../issm/{basin}_{future}/issm/solutions/u_rf_future.npy')
        u_glads_future = np.load(f'../../issm/{basin}_{future}/issm/solutions/u_glads_future.npy')
        print('u:', u_rf_future.shape)

        pm = np.logical_and(glads_f_present<=1, glads_f_present>=0)
        fm = np.logical_and(glads_f_future<=1, glads_f_future>=0)
        R2_f_present = 1 - np.nanvar(rf_f_present[pm] - glads_f_present[pm])/np.nanvar(glads_f_present[pm])
        R2_f_future = 1 - np.nanvar(rf_f_future[fm] - glads_f_future[fm])/np.nanvar(glads_f_future[fm])
        R2_N_present = 1 - np.nanvar(rf_N_present[pm] - glads_N_present[pm])/np.nanvar(glads_N_present[pm])
        R2_N_future = 1 - np.nanvar(rf_N_future[fm] - glads_N_future[fm])/np.nanvar(glads_N_future[fm])
        print(basin)
        print(R2_f_present, R2_f_future, R2_N_present, R2_N_future)

        am = np.logical_and(pm, fm)
        rf_deltaf = rf_f_future[am] - rf_f_present[am]
        glads_deltaf = glads_f_future[am] - glads_f_present[am]
        rf_deltaN = rf_N_future[am] - rf_N_present[am]
        glads_deltaN = glads_N_future[am] - glads_N_present[am]
        r2_deltaf = 1 - np.nanvar(rf_deltaf - glads_deltaf)/np.nanvar(glads_deltaf)
        r2_deltaN = 1 - np.nanvar(rf_deltaN - glads_deltaN)/np.nanvar(glads_deltaN)
        # print(r2_deltaf, r2_deltaN)

        r2_speed = 1 - np.nanvar(u_rf_future - u_glads_future)/np.nanvar(u_glads_future)
        print(r2_speed)

        all_f_glads = np.concatenate((all_f_glads, glads_f_future[fm].flatten()))
        all_f_rf = np.concatenate((all_f_rf, rf_f_future[fm].flatten()))
        all_N_glads = np.concatenate((all_N_glads, glads_N_future[fm].flatten()))
        all_N_rf = np.concatenate((all_N_rf, rf_N_future[fm].flatten()))
        all_u_glads = np.concatenate((all_u_glads, u_glads_future))
        all_u_rf = np.concatenate((all_u_rf, u_rf_future))

    R2_f_future = 1 - np.nanvar(all_f_glads - all_f_rf)/np.nanvar(all_f_glads)
    R2_N_future = 1 - np.nanvar(all_N_glads - all_N_rf)/np.nanvar(all_N_glads)
    R2_u = 1 - np.nanvar(all_u_rf - all_u_glads)/np.nanvar(all_u_glads)
    print('OVERALL')
    print(R2_f_future, R2_N_future, R2_u)


if __name__=='__main__':
    basins = [
        'G-H',
        'B-C',
        'C-Cp',
        'Cp-D',
    ]
    years = [
        2050,
        2300,
        2300,
        2300,
    ]
    main(basins, years)

    
