import os
import numpy as np
import pandas as pd

rhow = 1023.0
rhofw = 1000.0
rhoi = 917.0
g = 9.81

basins = [
    'B-C',
    'G-H',
    'Cp-D',
    'C-Cp',
    'Jpp-K',
    # 'C-Cp', 
    # 'Cp-D', 
    # 'G-H', 
    # 'Jpp-K', 
    # 'J-Jpp', 
    # 'F-G',
]

columns = [
    'R2 (mesh)',
    '% error (mesh)',
    'R2 (grid)',
    '% error (grid)',
]

def _getSlipSpeed(basins):
    ub = []
    for basin in basins:
        fname= f'../../issm/{basin}/data/lanl-mali/basal_velocity_mali.npy'
        levelset = np.load(f'../../issm/{basin}/data/geom/ocean_levelset.npy')
        uBasin = np.load(fname)
        ub.extend(uBasin[levelset==1])
    return np.array(ub)

def _percentError(yhat, y, quantile=0.1):
    lower = np.nanquantile(y, quantile)
    rel = np.nanmean(np.abs((yhat[y>lower] - y[y>lower])/y[y>lower]))
    return rel

def reportR2(basins, train='ff', split='test', ubThreshold=None):
    Y_RF_mesh = []
    Y_RF_grid = []
    Y_glads_mesh = []
    Y_glads_grid = []

    N_RF_mesh = []
    N_RF_grid = []
    N_glads_mesh = []
    N_glads_grid = []

    index = 95

    sum_resid_var = 0
    sum_data_var = 0

    table = np.zeros((len(basins)+1, 4))
    Ntable = np.zeros((len(basins)+1, 4))
    for i,basin in enumerate(basins):
        # Load levelset to find where ice is grounded
        levelset = np.load(
            os.path.join(f'../../issm/{basin}',
                'data/geom/ocean_levelset.npy')
        )
        # Add model outputs to list
        outputs = np.load(f'../../issm/{basin}/glads/ff.npy')[levelset>0,:]
        # Y_glads = np.nanmean(outputs, axis=1) # Take average over perturbed parameters
        # N_glads = np.nanmean(np.load(f'../../issm/{basin}/glads/N.npy')[levelset>0,:], axis=1)

        Y_glads = outputs[:, index]
        # N_glads = np.load(f'../../issm/{basin}/glads/N.npy')[levelset>0,index]
        N_glads = np.load(f'data/CV_{basin}_N_glads.npy')

        mask = np.logical_and(Y_glads>=0, Y_glads<=1)

        if ubThreshold:
            ub = _getSlipSpeed([basin])
            mask = np.logical_and(mask, ub>=ubThreshold)
        
        if train=='ff':
            if split=='test':
                fname = f'data/CV_{basin}.npy'
                Nfname = f'data/CV_{basin}_N_rf.npy'
            elif split=='train':
                # fname = f'data/pred_{basin}.npy'
                raise NotImplementedError

            Y_RF = np.load(fname)
            N_RF = np.load(Nfname)

            R2_f = 1 - np.var(Y_RF[mask]-Y_glads[mask])/np.var(Y_glads[mask])
            R2_N = 1 - np.var(N_RF[mask]-N_glads[mask])/np.var(N_glads[mask])

            # print('Mesh var:', np.nanvar(Y_glads[mask]))

            # Equal-area
            bmgrid = np.load(f'data/CV_{basin}_bmgrid.pkl', allow_pickle=True)
            R2_f_grid = 1 - np.nanvar(bmgrid['RF'] - bmgrid['glads'])/np.nanvar(bmgrid['glads'])
            R2_N_grid = 1 - np.nanvar(bmgrid['N_RF'] - bmgrid['N_glads'])/np.nanvar(bmgrid['N_glads'])

            table[i][0] = R2_f
            table[i][1] = 100*_percentError(Y_RF[mask], Y_glads[mask])
            table[i][2] = R2_f_grid
            table[i][3] = 100*_percentError(bmgrid['RF'], bmgrid['glads'])

            Ntable[i][0] = R2_N
            Ntable[i][1] = 100*_percentError(N_RF[mask], N_glads[mask])
            Ntable[i][2] = R2_N_grid
            Ntable[i][3] = 100*_percentError(bmgrid['N_RF'], bmgrid['N_glads'])

            # print('Gridded var:', np.nanvar(bmgrid['glads']))

            Y_RF_mesh.extend(Y_RF[mask])
            Y_RF_grid.extend(bmgrid['RF'].flatten())
            Y_glads_mesh.extend(Y_glads[mask])
            Y_glads_grid.extend(bmgrid['glads'].flatten())

            N_RF_mesh.extend(N_RF[mask])
            N_RF_grid.extend(bmgrid['N_RF'].flatten())
            N_glads_mesh.extend(N_glads[mask] - np.mean(N_glads[mask]))
            N_glads_grid.extend(bmgrid['N_glads'].flatten())

        elif train=='N':
            # raise NotImplementedError
            if split=='test':
                fname = f'data_N/CV_{basin}.npy'
            elif split=='train':
                raise NotImplementedError
            

            N_RF = np.load(fname)
            # mesh = np.load(f'../../issm/{basin}/data/geom/mesh.npy')
            bed = np.load(f'../../issm/{basin}/data/geom/bed.npy')[levelset>0]
            thick = np.load(f'../../issm/{basin}/data/geom/thick.npy')[levelset>0]

            phi_bed = rhofw*g*bed
            p_ice = rhoi*g*thick
            p_water = p_ice - N_RF
            Y_RF = p_water/p_ice

            p_glads = p_ice - N_glads
            Y_glads = p_glads/p_ice

            R2_f = 1 - np.var(Y_RF[mask]-Y_glads[mask])/np.var(Y_glads[mask])
            R2_N = 1 - np.var(N_RF[mask]-N_glads[mask])/np.var(N_glads[mask])

            # print('Mesh var:', np.nanvar(Y_glads[mask]))

            # Equal-area
            bmgrid = np.load(f'data/CV_{basin}_bmgrid.pkl', allow_pickle=True)
            R2_f_grid = 1 - np.nanvar(bmgrid['RF'] - bmgrid['glads'])/np.nanvar(bmgrid['glads'])
            R2_N_grid = 1 - np.nanvar(bmgrid['N_RF'] - bmgrid['N_glads'])/np.nanvar(bmgrid['N_glads'])

            table[i][0] = R2_f
            table[i][1] = 100*_percentError(Y_RF[mask], Y_glads[mask])
            table[i][2] = R2_f_grid
            table[i][3] = 100*_percentError(bmgrid['RF'], bmgrid['glads'])

            Ntable[i][0] = R2_N
            Ntable[i][1] = 100*_percentError(N_RF[mask], N_glads[mask])
            Ntable[i][2] = R2_N_grid
            Ntable[i][3] = 100*_percentError(bmgrid['N_RF'], bmgrid['N_glads'])

            # print('Gridded var:', np.nanvar(bmgrid['glads']))

            Y_RF_mesh.extend(Y_RF[mask])
            Y_RF_grid.extend(bmgrid['RF'].flatten())
            Y_glads_mesh.extend(Y_glads[mask])
            Y_glads_grid.extend(bmgrid['glads'].flatten())

            N_RF_mesh.extend(N_RF[mask])
            N_RF_grid.extend(bmgrid['N_RF'].flatten())
            N_glads_mesh.extend(N_glads[mask] - N_glads[mask].mean())
            N_glads_grid.extend(bmgrid['N_glads'].flatten())

    # Y_RF_mesh = []
    # Y_RF_grid = []
    # Y_glads_mesh = []
    # Y_glads_grid = []
        
    Y_RF_mesh = np.array(Y_RF_mesh)
    Y_RF_grid = np.array(Y_RF_grid)
    Y_glads_mesh = np.array(Y_glads_mesh)
    Y_glads_grid = np.array(Y_glads_grid)

    N_RF_mesh = np.array(N_RF_mesh)
    N_RF_grid = np.array(N_RF_grid)
    N_glads_mesh = np.array(N_glads_mesh)
    N_glads_grid = np.array(N_glads_grid)

    table[-1,0] = 1 - np.nanvar(Y_RF_mesh - Y_glads_mesh)/np.nanvar(Y_glads_mesh)
    table[-1,1] = 100*_percentError(Y_RF_mesh, Y_glads_mesh)
    table[-1,2] = 1 - np.nanvar(Y_RF_grid - Y_glads_grid)/np.nanvar(Y_glads_grid)
    table[-1,3] = 100*_percentError(Y_RF_grid, Y_glads_grid)
    basinDataFrame = pd.DataFrame(table, index=basins+['Overall'], columns=columns)
    print(basinDataFrame)

    Ntable[-1,0] = 1 - np.nanvar(N_RF_mesh - N_glads_mesh)/np.nanvar(N_glads_mesh)
    Ntable[-1,1] = 100*_percentError(N_RF_mesh, N_glads_mesh)
    Ntable[-1,2] = 1 - np.nanvar(N_RF_grid - N_glads_grid)/np.nanvar(N_glads_grid)
    Ntable[-1,3] = 100*_percentError(N_RF_grid, N_glads_grid)
    basinNFrame = pd.DataFrame(Ntable, index=basins+['Overall'], columns=columns)
    print(basinNFrame)
    
if __name__=='__main__':
    print(80*'*')

    print('\nTRAIN ON FLOT FRAC, ubThreshold=None')
    reportR2(basins, train='ff', ubThreshold=None)

    print('\nTRAIN ON FLOT FRAC, ubThreshold=200')
    reportR2(basins, train='ff', ubThreshold=200)

    print(80*'*')
    
    print('\nTRAIN ON N, ubThreshold=None')
    reportR2(basins, train='N', ubThreshold=None)

    print('\nTRAIN ON N, ubThreshold=200')
    reportR2(basins, train='N', ubThreshold=200)

