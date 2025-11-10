import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import cmocean

colors = ['#89b6bc', '#0d7d87', '#ff5a5e', '#c31e23']

# Feature importance
dN = np.load('deltaR2N.npy')
df = np.load('deltaR2f.npy')    
basins = [
    'G-H',
    # 'F-G',  # TODO check outputs, look like numerical issues
    # 'Ep-F', # jobs not done
    'Cp-D',
    'C-Cp',
    'B-C',
    'Jpp-K',
    # 'J-Jpp',# TODO check outputs, look like numerical issues
]
features = [
    'bed',
    'surface',
    # 'thickness',
    'grounding_line_distance',
    'basal_melt',
    'potential',
    'surface_slope',
    'bed_slope',
    'potential_slope',
    # 'binned_flow_accumulation',
]
# Sensitivity to particular training basin
basin_sensitivity_R2 = np.load('data/basin_sensitivity_R2_N.npy')

# Sensitivity to number of basins
number_basins_R2 = np.load('data/basin_sensitivity_factorial_R2_N.npy')

fig = plt.figure(figsize=(8, 8))
gs = GridSpec(2, 4, wspace=0.05, hspace=0.05,
    left=0.09, right=0.92, bottom=0.15, top=0.95,
    width_ratios=(100, 10, 100, 5),
)

ax1 = fig.add_subplot(gs[1,:3])
ax2 = fig.add_subplot(gs[0,0])
ax3 = fig.add_subplot(gs[0,2])

cax = fig.add_subplot(gs[0,-1])

ax = ax1
bar1 = ax.bar(np.arange(len(features))-0.25, 
    np.mean(df, axis=(0, 1)), 
    yerr=np.std(df, axis=(0, 1)), 
    width=0.4,
    color=colors[0],
    label='Flotation fraction',
)

ax.set_xticks(np.arange(len(features)), features, rotation=45, ha='right')
# ax.set_title('Flotation fraction feature importance')
ax.set_ylabel(r'Flotation fraction $R^2$ decrease', color='k', fontsize=8)

axtwin = ax.twinx()
bar2 = axtwin.bar(np.arange(len(features))+0.25, 
    np.mean(dN, axis=(0, 1)), 
    yerr=np.std(df, axis=(0, 1)), 
    width=0.4,
    color=colors[1],
    label='Effective pressure',
)
# ax.set_title("Feature importances using permutation: flotation fraction")
axtwin.set_ylabel('Effective pressure $R^2$ decrease', color='k', fontsize=8)
ax.legend(handles=(bar1, bar2), loc='upper right', frameon=False, fontsize=8)

ax.set_ylim(bottom=0)
axtwin.set_ylim(bottom=0)
ax.grid()
ax1.text(-0.0125, 1.025, 'c', transform=ax1.transAxes,
    fontweight='bold', ha='right', va='bottom')

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
ax2.text(-0.025, 1.025, 'a', transform=ax2.transAxes,
    fontweight='bold', ha='right', va='bottom')
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
ax3.text(-0.025, 1.025, 'b', transform=ax3.transAxes,
    fontweight='bold', ha='right', va='bottom')
for iy in range(len(y)):
    for ix in range(len(x)):
        if iy<(len(y)-1):
            deltar = (number_basins_R2 - number_basins_R2[-1])[iy,ix]
            color = 'k' if np.abs(deltar)<0.1 else 'w'
            ax3.text(x[ix], y[iy], '{:.3f}'.format(deltar), fontsize=8,
                ha='center', va='center', color=color)


cbar = fig.colorbar(Rpc, cax=cax)
cbar.set_label(r'$\Delta R^2$', fontsize=8)

for ax in (ax1,ax2,ax3,cax):
    ax.tick_params(labelsize=8)



fig.savefig('figures/hyperparameter_sensitivity.png', dpi=400)

