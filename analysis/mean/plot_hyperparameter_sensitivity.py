import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import cmocean

# colors = ['#89b6bc', '#0d7d87', '#ff5a5e', '#c31e23']
# colors = ['#3465CF', '#47577A']
colors = [cmocean.cm.diff(0.2), cmocean.cm.diff(0.4)]

# Feature importance
dN = np.load('deltaR2N_full.npy')
df = np.load('deltaR2f_full.npy')    
dN_red = np.load('deltaR2N_reduced.npy')
df_red = np.load('deltaR2f_reduced.npy')    
basins = [
    'G-H',
    'Ep-F',
    'Cp-D',
    'C-Cp',
    'B-C',
    'Jpp-K',
    'J-Jpp',
]
features = np.array([
    'Bed elevation',
    'Surface elevation',
    'Thickness',
    'Distance to GL',
    'Basal melt rate',
    'Shreve potential',
    'Surface slope',
    'Bed slope',
    'Potential slope',
    # 'binned_flow_accumulation',
])

reduced_features = np.array([0, 2, 5, 6])
features_nothickness = np.array([0,1,3,4,5,6,7,8]).astype(int)

# Sensitivity to particular training basin
basin_sensitivity_R2 = np.load('data/basin_sensitivity_R2_N.npy')

# Sensitivity to number of basins
number_basins_R2 = np.load('data/basin_sensitivity_factorial_R2_N.npy')

fig,(ax1,ax11) = plt.subplots(2, 1, figsize=(4,6), sharex=False, height_ratios=(9, 4))
axs = (ax1,ax11)

# Bar 1
ax = ax1
bar1 = ax.barh(np.arange(len(features))-0.25, 
    np.mean(df, axis=(0, 1)), 
    xerr=np.std(df, axis=(0, 1)), 
    height=0.4,
    color=colors[0],
    label='Flotation fraction',
)
ax.set_xlim([0, 2.2])

ax.set_yticks(np.arange(len(features)), features, rotation=45, ha='right')
# ax.set_title('Flotation fraction feature importance')
# ax.set_xlabel(r'Flotation fraction $R^2$ decrease', color='k', fontsize=8)

axtwin = ax.twiny()
bar2 = axtwin.barh(np.arange(len(features))+0.25, 
    np.mean(dN, axis=(0, 1)), 
    xerr=np.std(dN, axis=(0, 1)), 
    height=0.4,
    color=colors[1],
    label='Effective pressure',
)
axtwin.set_xlim([0, 28])
# ax.set_title("Feature importances using permutation: flotation fraction")
axtwin.set_xlabel('Effective pressure $R^2$ decrease', color='k', fontsize=8)
ax.legend(handles=(bar1, bar2), loc='upper right', frameon=True, fontsize=8)

ax.set_xlim(left=0)
axtwin.set_xlim(left=0)
ax.grid()

# reduced_mean_f = np.zeros(len(features))
# reduced_yerr_f = np.zeros(len(features))

# Bar 2
ax = ax11
bar3 = ax.barh(np.arange(len(reduced_features))-0.25, 
    np.mean(df_red, axis=(0, 1)), 
    xerr=np.std(df_red, axis=(0, 1)), 
    height=0.4,
    color=colors[0],
    label='Flotation fraction',
)
ax.set_xlim([0, 2.2])

# features[np.array([1,4,7,8])] = ''
# feature_mask = np.array([0,2,3,5,6])
ax.set_yticks(np.arange(len(features[reduced_features])), features[reduced_features], 
    rotation=45, ha='right')
# ax.set_yticklabels([])
# ax.set_title('Flotation fraction feature importance')
ax.set_xlabel(r'Flotation fraction $R^2$ decrease', color='k', fontsize=8)

# ax.invert_yaxis()
axtwin2 = ax.twiny()
bar4 = axtwin2.barh(np.arange(len(reduced_features))+0.25, 
    np.mean(dN_red, axis=(0, 1)), 
    xerr=np.std(dN_red, axis=(0, 1)), 
    height=0.4,
    color=colors[1],
    label='Effective pressure',
)
axtwin2.set_xlim([0, 28])
# ax.set_title("Feature importances using permutation: flotation fraction")
# axtwin2.set_xlabel('Effective pressure $R^2$ decrease', color='k', fontsize=8)
# ax.legend(handles=(bar1, bar2), loc='upper right', frameon=False, fontsize=8)

ax.set_xlim(left=0)
axtwin2.set_xlim(left=0)
ax.grid()
# ax.set_ylim(ax1.get_ylim())


ax1.text(-0.05, 1.025, '(a)', transform=ax1.transAxes,
    fontweight='bold', ha='right', va='bottom', fontsize=8)
ax11.text(-0.05, 1.025, '(b)', transform=ax11.transAxes,
    fontweight='bold', ha='right', va='bottom', fontsize=8)
ax11.set_xlim(ax1.get_xlim())
axtwin2.set_xlim(axtwin.get_xlim())

axs = (ax1,ax11,axtwin,axtwin2)
for ax in axs:
    ax.tick_params(labelsize=8)

fig.subplots_adjust(left=0.25, top=0.9, bottom=0.085)
fig.savefig('figures/feature_importance.png', dpi=400)

###################################################################
# Basin sensitivity

fig, (ax2,_,ax3,cax) = plt.subplots(4, 1, figsize=(4,6),
    height_ratios=(100,10,100,10))
axs = (ax2,ax3)

nbasins = len(basins)
x = np.arange(nbasins)
y = np.arange(nbasins)
Rpc = ax2.pcolormesh(x, y, basin_sensitivity_R2 - np.diag(basin_sensitivity_R2), vmin=-0.2, vmax=0.2, cmap=cmocean.cm.diff)
ax2.invert_yaxis()
ax2.set_yticks(y, basins)
ax2.set_xticks(x, basins)
ax2.xaxis.tick_top()
ax2.xaxis.set_label_position('top')
ax2.set_ylabel('Basin excluded from training', fontsize=8)
ax2.set_xlabel('Test basin', fontsize=8)
ax2.text(-0.025, 1.025, '(a)', transform=ax2.transAxes,
    fontweight='bold', ha='right', va='bottom', fontsize=8)
# ax2.set_aspect('equal')
for iy in range(nbasins):
    for ix in range(nbasins):
        if iy!=ix:
            deltar = (basin_sensitivity_R2 - np.diag(basin_sensitivity_R2))[iy,ix]
            color = 'k' if np.abs(deltar)<0.1 else 'w'
            ax2.text(x[ix], y[iy], '{:.3f}'.format(deltar), fontsize=8,
                ha='center', va='center')

y = np.arange(2, nbasins+1)
ax3.pcolormesh(x, y, number_basins_R2 - number_basins_R2[-1], vmin=-0.2, vmax=0.2, cmap=cmocean.cm.diff)
ax3.invert_yaxis()

ax3.xaxis.tick_top()
ax3.xaxis.set_label_position('top')
ax3.set_yticks(y, y)
ax3.set_xticks(x, basins)
ax3.set_xlabel('Test basin', fontsize=8)
ax3.set_ylabel('Number of basins', fontsize=8)
ax3.text(-0.025, 1.025, '(b)', transform=ax3.transAxes,
    fontweight='bold', ha='right', va='bottom', fontsize=8)
# ax3.set_aspect('equal')
for iy in range(len(y)):
    for ix in range(len(x)):
        if iy<(len(y)-1):
            deltar = (number_basins_R2 - number_basins_R2[-1])[iy,ix]
            color = 'k' if np.abs(deltar)<0.1 else 'w'
            ax3.text(x[ix], y[iy], '{:.3f}'.format(deltar), fontsize=8,
                ha='center', va='center', color=color)

cbar = fig.colorbar(Rpc, cax=cax, orientation='horizontal')
cbar.set_label(r'$\Delta R^2$', fontsize=8)

for ax in (ax2,ax3,cax):
    ax.tick_params(labelsize=8)
_.set_visible(False)
fig.subplots_adjust(hspace=0.1, bottom=0.085, top=0.9, left=0.15, right=0.95)
fig.savefig('figures/hyperparameter_sensitivity.png', dpi=400)



# Plot the case with no thickness
dN_nt = np.load('deltaR2N_nothickness.npy')
df_nt = np.load('deltaR2f_nothickness.npy')    
fig,ax = plt.subplots(figsize=(5, 4))

# Bar 1
bar1 = ax.bar(np.arange(len(features_nothickness))-0.25, 
    np.mean(df_nt, axis=(0, 1)), 
    yerr=np.std(df_nt, axis=(0, 1)), 
    width=0.4,
    color=colors[0],
    label='Flotation fraction',
)

ax.set_xticks(np.arange(len(features_nothickness)), features[features_nothickness], rotation=45, ha='right')
# ax.set_title('Flotation fraction feature importance')
ax.set_ylabel(r'Flotation fraction $R^2$ decrease', color='k', fontsize=8)

axtwin = ax.twinx()
bar2 = axtwin.bar(np.arange(len(features_nothickness))+0.25, 
    np.mean(dN_nt, axis=(0, 1)), 
    yerr=np.std(dN_nt, axis=(0, 1)), 
    width=0.4,
    color=colors[1],
    label='Effective pressure',
)
# ax.set_title("Feature importances using permutation: flotation fraction")
axtwin.set_ylabel('Effective pressure $R^2$ decrease', color='k', fontsize=8)
ax.legend(handles=(bar1, bar2), loc='upper right', frameon=True, fontsize=8)

# ax.in()

ax.set_ylim(bottom=0)
axtwin.set_ylim(bottom=0)
ax.grid()

fig.tight_layout()
fig.savefig('figures/para_sensitivity_no_thickness.png', dpi=400)