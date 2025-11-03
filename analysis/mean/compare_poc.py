import numpy as np
from matplotlib import pyplot as plt

basins = [
    'B-C',
    'C-Cp',
    'Cp-D',
    'G-H',
    'Jpp-K',
]

rhow = 1023
rhoice = 917
g = 9.81

Npoc = []
Nglads = []
allbed = []
allsurf = []
for basin in basins:
    nglads = np.load(f'data/pred_{basin}_N_glads.npy')
    bed = np.load(f'../../issm/{basin}/data/geom/bed.npy')
    thick = np.load(f'../../issm/{basin}/data/geom/thick.npy')
    pice = rhoice*thick*g
    pwater = np.maximum(-rhow*g*bed, 0)

    levelset = np.load(f'../../issm/{basin}/data/geom/ocean_levelset.npy')
    npoc = (pice-pwater)[levelset>0]

    ff = np.nanmean(np.load(f'../../issm/{basin}/glads/ff.npy'), axis=1)[levelset>0]
    mask = np.logical_and(ff<=1, ff>=0)
    print(mask.shape)
    print(ff.shape)

    npoc[~mask] = np.nan
    nglads[~mask] = np.nan
    Npoc.extend(npoc)
    Nglads.extend(nglads)
    allbed.extend(bed[levelset>0])

    allsurf.extend(np.load(f'../../issm/{basin}/data/geom/surface.npy')[levelset>0])

Npoc = np.array(Npoc)
Nglads = np.array(Nglads)
allbed = np.array(allbed)

R2 = 1 - np.nanvar(Npoc-Nglads)/np.nanvar(Nglads)
print('POC R2:', R2)

# R2 = 1 - np.nanvar(Npoc[allbed<0]-Nglads[allbed<0])/np.nanvar(Nglads[allbed<0])
# print('POC R2, bed<0:', R2)
# R2 = 1 - np.nanvar(Npoc[allbed>0]-Nglads[allbed>0])/np.nanvar(Nglads[allbed>0])
# print('POC R2, bed>0:', R2)


fig,ax = plt.subplots()
ax.scatter(Nglads/1e6, Npoc/1e6, s=1, alpha=0.2)
# ax.scatter(Nglads[allbed>0]/1e6, Npoc[allbed>0]/1e6, s=1, alpha=0.2)
ax.set_xlabel('GlaDS N (MPa)')
ax.set_ylabel('Perfect ocean connection N (MPa)')
ax.grid()
ax.plot(Nglads/1e6, Nglads/1e6, color='k')
fig.savefig('figures/scatter_POC.png', dpi=400)


fig,ax = plt.subplots()
ax.scatter((Npoc-Nglads)/1e6, allbed, s=1, alpha=0.2)
ax.set_xlabel('POC - GlaDS N (MPa)')
ax.set_ylabel('Bed elevation (m)')
ax.grid()
fig.savefig('figures/scatter_POC_bed.png', dpi=400)

fig,ax = plt.subplots()
ax.scatter((Npoc-Nglads)/1e6, allsurf, s=1, alpha=0.2)
ax.set_xlabel('POC - GlaDS N (MPa)')
ax.set_ylabel('Surface elevation (m)')
ax.grid()
fig.savefig('figures/scatter_POC_surface.png', dpi=400)
