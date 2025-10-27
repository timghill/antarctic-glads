import numpy as np
import scipy.linalg
from matplotlib import pyplot as plt
from matplotlib.tri import Triangulation
import cmocean

thwaites = np.load('../issm/thwaites/train/train_N.npy', mmap_mode='r')
amery = np.load('../issm/amery/train/train_N.npy', mmap_mode='r')

thwaites = np.mean(thwaites[:, -1, :], axis=-1)
amery = np.mean(amery[:, -1, :], axis=-1)

thwaites_mesh = np.load('../issm/thwaites/data/geom/thwaites_mesh.npy', allow_pickle=True)
amery_mesh = np.load('../issm/amery/data/geom/amery_mesh.npy', allow_pickle=True)

def gldist(mesh, bed, surface):

    bz = surface[mesh['vertexonboundary']==1]
    rhow = 1028
    rhoi = 910
    h_buoyancy = -(rhow-rhoi)/rhoi * bz
    h_boundary = surface[mesh['vertexonboundary']==1]
    gl = np.where(h_boundary<=(h_buoyancy + 200))[0]

    glx = mesh['x'][gl,None]
    gly = mesh['y'][gl,None]

    dx = glx - mesh['x']
    dy = gly - mesh['y']
    
    dd = np.sqrt(dx**2 + dy**2)
    ddmin = np.min(dd, axis=0)
    print(ddmin.shape)
    return ddmin

thwaites_surface = np.load('../issm/thwaites/data/geom/thwaites_surface.npy')
thwaites_bed = np.load('../issm/thwaites/data/geom/thwaites_bed.npy')
thwaites_thick = thwaites_surface - thwaites_bed
thwaites_gldist = gldist(thwaites_mesh, thwaites_bed, thwaites_surface)
thwaites_features = [thwaites_gldist/1e3, thwaites_bed, thwaites_surface, thwaites_thick]
feature_labels = ['Grounding line distance (km)', 'Bed elevation (m)', 'Surface elevation (m)', 'Ice thickness (m)']


amery_surface = np.load('../issm/amery/data/geom/amery_surface.npy')
amery_bed = np.load('../issm/amery/data/geom/amery_bed.npy')
amery_thick = amery_surface - amery_bed
amery_gldist = gldist(amery_mesh, amery_bed, amery_surface)
amery_features = [amery_gldist/1e3, amery_bed, amery_surface, amery_thick]

def bin_average(x, z, bins=10, q=(0.025, 0.975)):
    zmin = np.min(x)
    zmax = np.max(x)
    bin_edges = np.linspace(zmin, zmax, bins+1)
    bin_mid = 0.5*(bin_edges[:-1] + bin_edges[1:])
    bin_mean = np.zeros(bins)
    bin_range = np.zeros((bins, 2))
    for i in range(bins):
        zvals = z[np.logical_and(x>=bin_edges[i], x<bin_edges[i+1])]
        bin_mean[i] = np.nanmean(zvals)
        bin_range[i] = np.nanquantile(zvals, q)
    return bin_mid, bin_mean, bin_range

nfeat = len(thwaites_features)
bins = 25
inc = 1

fig,axs = plt.subplots(nfeat//2, 2, figsize=(10, 7.5))
for i in range(nfeat):
    ax = axs.flat[i]



    amery_bins, amery_mean, amery_range = bin_average(amery_features[i], amery/1e6, bins=bins)
    ax.plot(amery_bins, amery_mean, zorder=5, label='Amery')
    ylim = ax.get_ylim()
    ax.fill_between(amery_bins, amery_range[:,0], amery_range[:,1],
        edgecolor='none', alpha=0.25, zorder=3)
    

    thwaites_bins, thwaites_mean, thwaites_range = bin_average(thwaites_features[i], thwaites/1e6, bins=bins)
    ax.plot(thwaites_bins, thwaites_mean, zorder=5, label='Thwaites')
    ylim = ax.get_ylim()
    ax.fill_between(thwaites_bins, thwaites_range[:,0], thwaites_range[:,1],
        edgecolor='none', alpha=0.25, zorder=3)

    ax.set_ylim(ylim)
    ax.set_xlabel(feature_labels[i])
    ax.set_ylabel('Effective pressure (MPa)')

axs.flat[0].legend()

plt.tight_layout()
fig.subplots_adjust(hspace=0.25)
fig.savefig('amery_thwaites_features.png', dpi=400)


amery_r = np.corrcoef(amery_features)
print('amery_r:', amery_r)
thwaites_r = np.corrcoef(thwaites_features)

rr = (amery_r, thwaites_r)

fig,axs = plt.subplots(ncols=2, figsize=(8, 5), sharey=True)

for i in range(2):
    pc = axs[i].pcolormesh(rr[i], cmap=cmocean.cm.balance, vmin=-1, vmax=1)
    for j in range(nfeat):
        for k in range(nfeat):
            rjk = rr[i][j,k]
            color = 'w' if np.abs(rjk)>0.625 else 'k'
            rfmt = '{:.2f}'.format(rjk)
            axs[i].text(0.5 + k, 0.5 + j, rfmt, ha='center', va='center', color=color)
    axs[i].set_xticks(0.5+np.arange(nfeat), feature_labels, rotation=45, ha='right')
    axs[i].set_yticks(0.5+np.arange(nfeat), feature_labels, rotation=45, va='top')
    # axs[i].invert_yaxis()

axs[0].set_title('Amery')
axs[1].set_title('Thwaites')
fig.subplots_adjust(right=0.95, left=0.25, bottom=0.35, wspace=0.1, top=0.95)
fig.colorbar(pc, ax=axs, label='Correlation coefficient')

fig.savefig('amery_thwaites_features_correlation.png', dpi=400)

# # For Thwaites first!!
# A = np.array([
#     thwaites_gldist, 
#     thwaites_bed, 
#     # thwaites_surface, 
#     thwaites_thick,
#     ]).T
# B = thwaites[:,None]/1e6
# x = scipy.linalg.lstsq(A, B)[0]
# print('x:', x.shape, x)

# thwaites_Nhat = np.squeeze(A @ x)
# fig,ax = plt.subplots()
# ax.scatter(thwaites/1e6, thwaites_Nhat, s=5, edgecolor='none', alpha=0.25)
# ax.grid()
# ax.set_xlabel('Observed N (MPa)')
# ax.set_ylabel('Predicted N (MPa)')
# ax.plot([0, 5], [0, 5], color='k', linewidth=0.5)
# ax.set_xlim([0, 5])
# ax.set_ylim([0, 5])
# fig.savefig('thwaites_linmodel.png', dpi=400)

# r_lin = np.corrcoef(thwaites/1e6, thwaites_Nhat.squeeze())[0,1]
# print('R linear:', r_lin)

# res = thwaites_Nhat - thwaites/1e6
# R2 = 1 - np.var(res)/np.var(thwaites/1e6)
# print('R2:', R2)

# mtri = Triangulation(thwaites_mesh['x'], thwaites_mesh['y'], thwaites_mesh['elements']-1)
# fig,axs = plt.subplots(ncols=2, figsize=(7, 4))
# tpc = axs[0].tripcolor(mtri, thwaites/1e6, vmin=0, vmax=4, cmap=cmocean.cm.haline)
# axs[1].tripcolor(mtri, thwaites_Nhat, vmin=0, vmax=4, cmap=cmocean.cm.haline)
# # axs[2].tripcolor(mtri, thwaites_Nhat-thwaites/1e6, vmin=-2, vmax=2, cmap=cmocean.cm.balance)
# axs[0].set_title('Observed N (MPa)')
# axs[1].set_title('Predicted N (MPa)')
# # axs[2].set_title('Predicted - Observed')
# # plt.tight_layout()
# for ax in axs:
#     ax.set_aspect('equal')
#     ax.set_xticks([])
#     ax.set_yticks([])
#     ax.spines[['left', 'right', 'top', 'bottom']].set_visible(False)
# fig.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.925, wspace=0)
# fig.colorbar(tpc, ax=axs)
# fig.savefig('thwaites_linmodel_map.png', dpi=400)



# For Amery now!!
A = np.array([
    amery_gldist, 
    amery_bed, 
    amery_surface, 
    amery_thick,
    ]).T
B = amery[:,None]/1e6
x = scipy.linalg.lstsq(A, B)[0]
print('x:', x.shape, x)

amery_Nhat = np.squeeze(A @ x)
fig,ax = plt.subplots()
ax.scatter(amery/1e6, amery_Nhat, s=5, edgecolor='none', alpha=0.25)
ax.grid()
ax.set_xlabel('Observed N (MPa)')
ax.set_ylabel('Predicted N (MPa)')
ax.plot([0, 5], [0, 5], color='k', linewidth=0.5)
ax.set_xlim([0, 5])
ax.set_ylim([0, 5])
fig.savefig('amery_linmodel.png', dpi=400)

r_lin = np.corrcoef(amery/1e6, amery_Nhat.squeeze())[0,1]
print('R linear:', r_lin)

res = amery_Nhat - amery/1e6
R2 = 1 - np.var(res)/np.var(amery/1e6)
print('R2:', R2)

mtri = Triangulation(amery_mesh['x'], amery_mesh['y'], amery_mesh['elements']-1)
fig,axs = plt.subplots(ncols=2, figsize=(7, 4))
tpc = axs[0].tripcolor(mtri, amery/1e6, vmin=0, vmax=4, cmap=cmocean.cm.haline)
axs[1].tripcolor(mtri, amery_Nhat, vmin=0, vmax=4, cmap=cmocean.cm.haline)
# axs[2].tripcolor(mtri, amery_Nhat-amery/1e6, vmin=-2, vmax=2, cmap=cmocean.cm.balance)
axs[0].set_title('Observed N (MPa)')
axs[1].set_title('Predicted N (MPa)')
# axs[2].set_title('Predicted - Observed')
# plt.tight_layout()
for ax in axs:
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[['left', 'right', 'top', 'bottom']].set_visible(False)
fig.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.925, wspace=0)
fig.colorbar(tpc, ax=axs)
fig.savefig('amery_linmodel_map.png', dpi=400)

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
# X, y = make_regression(n_features=4, n_informative=2,
#                        random_state=0, shuffle=False)
regr = RandomForestRegressor(max_features=4)
regr.fit(A[::5], amery[::5]/1e6)
print(regr)
Nhat = regr.predict(A)


fig,axs = plt.subplots(ncols=2, figsize=(7, 4))
tpc = axs[0].tripcolor(mtri, amery/1e6, vmin=0, vmax=4, cmap=cmocean.cm.haline)
axs[1].tripcolor(mtri, Nhat, vmin=0, vmax=4, cmap=cmocean.cm.haline)
# axs[2].tripcolor(mtri, amery_Nhat-amery/1e6, vmin=-2, vmax=2, cmap=cmocean.cm.balance)
axs[0].set_title('Observed N (MPa)')
axs[1].set_title('Predicted N (MPa)')
# axs[2].set_title('Predicted - Observed')
# plt.tight_layout()
for ax in axs:
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[['left', 'right', 'top', 'bottom']].set_visible(False)
fig.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.925, wspace=0)
fig.colorbar(tpc, ax=axs)
fig.savefig('amery_rf_map.png', dpi=400)
res = Nhat - amery/1e6
R2 = 1 - np.var(res)/np.var(amery/1e6)
print('RF R2:', R2)

fig,ax = plt.subplots()
ax.scatter(amery/1e6, Nhat, s=5, edgecolor='none', alpha=0.25)
ax.grid()
ax.set_xlim([0, 4])
ax.set_ylim([0, 4])
ax.set_xlabel('Observed')
ax.set_ylabel('Predicted')
fig.savefig('amery_rf_scatter.png', dpi=400)
