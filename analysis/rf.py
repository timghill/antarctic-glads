import numpy as np
import scipy.linalg
from matplotlib import pyplot as plt
from matplotlib.tri import Triangulation
import cmocean
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from scipy import linalg
import alphashape

thwaites = np.load('../issm/thwaites/train/train_N.npy', mmap_mode='r')
amery = np.load('../issm/amery/train/train_N.npy', mmap_mode='r')

thwaites = np.mean(thwaites[:, -1, :], axis=-1)
amery = np.mean(amery[:, -1, :], axis=-1)

thwaites_mesh = np.load('../issm/thwaites/data/geom/thwaites_mesh.npy', allow_pickle=True)
amery_mesh = np.load('../issm/amery/data/geom/amery_mesh.npy', allow_pickle=True)

amery_mtri = Triangulation(amery_mesh['x']/1e3, 
    amery_mesh['y']/1e3, amery_mesh['elements']-1)
thwaites_mtri = Triangulation(thwaites_mesh['x']/1e3, 
    thwaites_mesh['y']/1e3, thwaites_mesh['elements']-1)

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

def tab(mesh, bed, surface):
    thick = surface - bed
    thick_float = surface - thick*(1028 - 910)/910
    # thick_float[surface>thick_float] = thick[surface>thick_float]
    return thick_float


thwaites_surface = np.load('../issm/thwaites/data/geom/thwaites_surface.npy')
thwaites_bed = np.load('../issm/thwaites/data/geom/thwaites_bed.npy')
thwaites_thick = thwaites_surface - thwaites_bed
thwaites_ub = np.load('../issm/thwaites/data/lanl-mali/basal_velocity_mali.npy')
thwaites_gldist = gldist(thwaites_mesh, thwaites_bed, thwaites_surface)
thwaites_features = [thwaites_gldist/1e3, thwaites_bed, thwaites_surface, thwaites_thick, thwaites_ub]
feature_labels = ['Grounding line distance (km)', 'Bed elevation (m)', 
    'Surface elevation (m)', 'Ice thickness (m)', 'Sliding velocity (m/a)']

thwaites_tab = tab(thwaites_mesh, thwaites_bed, thwaites_surface)

amery_surface = np.load('../issm/amery/data/geom/amery_surface.npy')
amery_bed = np.load('../issm/amery/data/geom/amery_bed.npy')
amery_thick = amery_surface - amery_bed
amery_ub = np.load('../issm/amery/data/lanl-mali/basal_velocity_mali.npy')
amery_gldist = gldist(amery_mesh, amery_bed, amery_surface)
amery_features = [amery_gldist/1e3, amery_bed, amery_surface, amery_thick, amery_ub]
A = np.array(amery_features).T
T = np.array(thwaites_features).T

joined = np.vstack((A, T))
print(joined.shape)
yjoined = np.hstack((amery, thwaites)).T
print(yjoined.shape)

xmax = np.max(joined, axis=0)
xmin = np.min(joined, axis=0)

A = (xmax-A)/(xmax-xmin)
T = (xmax-T)/(xmax-xmin)
joined = (xmax-joined)/(xmax-xmin)

plt.tripcolor(thwaites_mtri, thwaites_tab)
plt.colorbar()
plt.savefig('tmp_thwaites_tab.png', dpi=400)

########################################################################
# 0 clustering
########################################################################
ynorm = (joined - np.mean(joined, axis=0))/np.std(joined, axis=0)
u,s,v = linalg.svd(ynorm, full_matrices=False)
print('u:', u.shape)
print('s:', s.shape)
print('v:', v.shape)
print('var:', s**2/np.sum(s**2))
print('v:', v.T)

fig,ax = plt.subplots()
u_amery = u[:len(amery)]
u_thwaites = u[len(amery):]

alpha_shape_amery = alphashape.alphashape(u_amery[:,:2], 250).exterior.coords.xy
alpha_shape_thwaites = alphashape.alphashape(u_thwaites[:,:2], 250).exterior.coords.xy

ax.scatter(u_amery[:,0], u_amery[:,1], label='Amery', alpha=0.5)
ax.scatter(u_thwaites[:,0], u_thwaites[:,1], label='Thwaites', alpha=0.5)
ax.set_xlabel('PC1 ({:.1%})'.format(s[0]**2/np.sum(s**2)))
ax.set_ylabel('PC2 ({:.1%})'.format(s[1]**2/np.sum(s**2)))
ax.plot(alpha_shape_amery[0], alpha_shape_amery[1], color='blue', zorder=5)
ax.plot(alpha_shape_thwaites[0], alpha_shape_thwaites[1], color='red', zorder=5)
ax.grid()
ax.legend()
fig.savefig('amery_thwaites_pc_clusters.png', dpi=400)
fig.tight_layout()


########################################################################
# 1 with Amery only
print('1: AMERY ONLY')
########################################################################
regr = RandomForestRegressor(max_features=1, max_depth=10)
regr.fit(A[::5], amery[::5]/1e6)
print(regr)
Nhat = regr.predict(A)

amery_mtri = Triangulation(amery_mesh['x']/1e3, 
    amery_mesh['y']/1e3, amery_mesh['elements']-1)

amery_xclose = np.array([1.4e6, 2.1e6])/1e3
amery_yclose = np.array([4.e5, 10e5])/1e3

fig,axs = plt.subplots(ncols=2, figsize=(7, 4))
tpc = axs[0].tripcolor(amery_mtri, amery/1e6, vmin=0, vmax=4, cmap=cmocean.cm.haline)
axs[1].tripcolor(amery_mtri, Nhat, vmin=0, vmax=4, cmap=cmocean.cm.haline)
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
for ax in axs:
    ax.set_xlim(amery_xclose)
    ax.set_ylim(amery_yclose)
fig.savefig('amery_rf_map_close.png', dpi=400)

R2 = regr.score(A, amery/1e6)
print('R2:', R2)

fig,ax = plt.subplots()
ax.scatter(amery/1e6, Nhat, s=5, edgecolor='none', alpha=0.25)
ax.grid()
ax.set_xlim([0, 4])
ax.set_ylim([0, 4])
ax.set_xlabel('Observed')
ax.set_ylabel('Predicted')
fig.savefig('amery_rf_scatter.png', dpi=400)


########################################################################
# 2 training on both Amery and Thwaites
print('2: TRAINING ON BOTH AMERY AND THWAITES')
########################################################################


regr = RandomForestRegressor()
regr.fit(joined[::5], yjoined[::5]/1e6)
print(regr)
Nhat = regr.predict(joined)

Nhat_amery = Nhat[:len(amery)]
Nhat_thwaites = Nhat[len(amery):]

amery_xclose = np.array([1.4e6, 2.1e6])/1e3
amery_yclose = np.array([4.e5, 10e5])/1e3

thwaites_xclose = np.array([-1600, -1200])
thwaites_yclose = np.array([-650, -250])

R2 = regr.score(joined, yjoined/1e6)
print('R2:', R2)

fig,ax = plt.subplots()
ax.scatter(amery/1e6, Nhat_amery, s=5, edgecolor='none', alpha=0.25, label='Amery')
ax.scatter(thwaites/1e6, Nhat_thwaites, s=5, edgecolor='none', alpha=0.25, label='Thwaites')
ax.grid()
ax.legend(markerscale=3)
ax.set_xlim([0, 4])
ax.set_ylim([0, 4])
ax.set_xlabel('Observed')
ax.set_ylabel('Predicted')
fig.savefig('both_rf_scatter.png', dpi=400)


########################################################################
# 3.1 training on Thwaites prediction on Amery
print('3.1 TRAIN THWAITES / PREDICT AMERY')
########################################################################

regr = RandomForestRegressor(max_features=1, max_depth=10)
regr.fit(T[::1], thwaites[::1]/1e6)
Nhat = regr.predict(A)

fig,axs = plt.subplots(ncols=2, figsize=(7, 4))
tpc = axs[0].tripcolor(amery_mtri, amery/1e6, vmin=0, vmax=4, cmap=cmocean.cm.haline)
axs[1].tripcolor(amery_mtri, Nhat, vmin=0, vmax=4, cmap=cmocean.cm.haline)
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
fig.savefig('03_amery_rf_map.png', dpi=400)
for ax in axs:
    ax.set_xlim(amery_xclose)
    ax.set_ylim(amery_yclose)
fig.savefig('03_amery_rf_map_close.png', dpi=400)


R2 = regr.score(A, amery/1e6)
print('R2:', R2)

fig,ax = plt.subplots()
ax.scatter(amery/1e6, Nhat, s=5, edgecolor='none', alpha=0.25)
ax.grid()
ax.set_xlim([0, 4])
ax.set_ylim([0, 4])
ax.set_xlabel('Observed')
ax.set_ylabel('Predicted')
fig.savefig('03_amery_rf_scatter.png', dpi=400)

fig,ax = plt.subplots()
ax.scatter(u_amery[:, 0], u_amery[:, 1], s=3, c=np.abs(Nhat - amery/1e6), 
    vmin=0, vmax=2, cmap=cmocean.cm.amp)

ax.plot(alpha_shape_amery[0], alpha_shape_amery[1], color='tab:blue', zorder=5)
ax.plot(alpha_shape_thwaites[0], alpha_shape_thwaites[1], color='tab:orange', zorder=5)
fig.savefig('03_amery_rf_pc.png', dpi=400)


########################################################################
# 3.2 training on Amery prediction on Thwaites
print('3.2 TRAIN AMERY / PREDICT THWAITES')
########################################################################

regr = RandomForestRegressor(max_features=3, max_depth=10)
regr.fit(A[::1], amery[::1]/1e6)
Nhat = regr.predict(T)

fig,axs = plt.subplots(ncols=2, figsize=(7, 4))
tpc = axs[0].tripcolor(thwaites_mtri, thwaites/1e6, vmin=0, vmax=4, cmap=cmocean.cm.haline)
axs[1].tripcolor(thwaites_mtri, Nhat, vmin=0, vmax=4, cmap=cmocean.cm.haline)
for ax in axs:
    ax.tricontour(thwaites_mtri, thwaites_ub, levels=(100,), colors='k')
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
fig.savefig('03_thwaites_rf_map.png', dpi=400)
for ax in axs:
    ax.set_xlim(thwaites_xclose)
    ax.set_ylim(thwaites_yclose)
fig.savefig('03_thwaites_rf_map_close.png', dpi=400)


R2 = regr.score(T, thwaites/1e6)
print('R2:', R2)

fig,ax = plt.subplots()
ax.scatter(thwaites/1e6, Nhat, s=5, edgecolor='none', alpha=0.25)
ax.grid()
ax.set_xlim([0, 4])
ax.set_ylim([0, 4])
ax.set_xlabel('Observed')
ax.set_ylabel('Predicted')
fig.savefig('03_thwaites_rf_scatter.png', dpi=400)

fig,ax = plt.subplots()
ax.scatter(u_thwaites[:, 0], u_thwaites[:, 1], s=3, c=np.abs(Nhat - thwaites/1e6), 
    vmin=0, vmax=2, cmap=cmocean.cm.amp)

ax.plot(alpha_shape_amery[0], alpha_shape_amery[1], color='tab:blue', zorder=5)
ax.plot(alpha_shape_thwaites[0], alpha_shape_thwaites[1], color='tab:orange', zorder=5)
fig.savefig('03_thwaites_rf_pc.png', dpi=400)
