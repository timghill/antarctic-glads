"""
Assess whether simulation ensembles meet steady-state criteria
"""

import os
import numpy as np
# from matplotlib import pyplot as plt

basins = [
    'B-C',
    'C-Cp',
    'Cp-D',
    'Ep-F',
    'G-H',
    'J-Jpp',
    'Jpp-K',
    'C-Cp_2300',
]

dhdt_quantiles = np.zeros(len(basins))
dSdt_quantiles = np.zeros(len(basins))

# fig,axs = plt.subplots(ncols=4, nrows=2, figsize=(8, 5))

for i,basin in enumerate(basins):
    gladsdir = f'../../issm/{basin}/glads/'
    # if os.path.exists(os.path.join(gladsdir, 'dhdt.npy')):
    try:
        dhdt = np.load(os.path.join(gladsdir, 'dhdt.npy'))
        dSdt = np.load(os.path.join(gladsdir,'dSdt.npy'))

        dhdt_quantiles[i] = np.quantile(np.abs(dhdt), 0.95)
        dSdt_quantiles[i] = np.quantile(np.abs(dSdt), 0.95)

    except Exception as e:
        # print(e)
        print(f'Basin {basin} has no d/dt data')

print(basins)
print(dhdt_quantiles)
print(dSdt_quantiles)

