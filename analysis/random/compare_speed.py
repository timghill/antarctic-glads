import numpy as np


def main(basins):
    cv = np.array([])
    glads = np.array([])
    for basin in basins:
        u_cv = np.load(f'../../issm/{basin}/issm/solutions/u_glads_cv_nonlinear.npy')
        u_glads = np.load(f'../../issm/{basin}/issm/solutions/u_glads_glads_nonlinear.npy')
        r2 = 1 - np.nanvar(u_cv - u_glads)/np.nanvar(u_glads)
        print(r2)
        cv = np.concatenate((cv, u_cv))
        glads = np.concatenate((glads, u_glads))
    
    R2 = 1 - np.nanvar(glads-cv)/np.nanvar(glads)
    print('OVERALL:', R2)

if __name__=='__main__':
    main([
        'G-H',
        'B-C',
        'C-Cp',
        'Cp-D',
        'Jpp-K',
    ]
    )
