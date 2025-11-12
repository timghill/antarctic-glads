import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.tri import Triangulation
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import cmocean
import netCDF4 as nc

rhow = 1023.0
rhofw = 1000.0
rhoi = 917.0
g = 9.81

profile_basins = ['G-H', 'G-H', 'C-Cp', 'B-C', 'Jpp-K', 'Cp-D']
profile_numbers = [0, 1, 0, 0, 0, 0]
profile_labels = ['Thwaites', 'PIG', 'Denman', 'Lambert', 'Recovery', 'Totten']
alphabet = ['a', 'b', 'd', 'e', 'c', 'f']

bedmachine = '../../data/bedmachine/BedMachineAntarctica-v3.nc'

Ncmap = cmocean.cm.matter
Zcmap = cmocean.cm.gray
Zalpha = 0.5
fs = 8

def plot_error(basins, index=15):


    dx = 16
    with nc.Dataset(bedmachine, 'r') as bm:
        mask = bm['mask'][::dx, ::dx].astype(int)
        x = bm['x'][::dx].astype(np.float32)
        y = bm['y'][::dx].astype(np.float32)
        bed = bm['bed'][::dx, ::dx].astype(np.float32)
        surf = bm['surface'][::dx, ::dx].astype(np.float32)

    bed[mask==0] = np.nan
    mask[surf>2000] = 2
    xx,yy = np.meshgrid(x,y)

    xmin = np.min(xx[~np.isnan(bed)])
    xmax = np.max(xx[~np.isnan(bed)])
    ymin = np.min(yy[~np.isnan(bed)])
    ymax = np.max(yy[~np.isnan(bed)])


    fig = plt.figure(figsize=(10, 7))
    nrows = 2
    ncols = 3
    gs = GridSpec(ncols=ncols, nrows=2*nrows,
        hspace=0.1, wspace=0.15, left=0.05, bottom=0.0, right=0.95, top=0.925,
        height_ratios=(3, 100, 3, 100),
        width_ratios=(100, 100, 100),
    )
    axs = np.array([[fig.add_subplot(gs[2*i+1,j], facecolor='none') for j in range(2)] for i in range(nrows)])
    caxs = np.array([[fig.add_subplot(gs[2*i,j]) for j in range(2)] for i in range(nrows)])

    Y_glads = np.array([])
    Y_rf = np.array([])
    N_glads = np.array([])
    N_rf = np.array([])
    phi_glads = np.array([])
    phi_rf = np.array([])
    for i,ax in enumerate(axs.flat):
        ax.contour(xx, yy, mask, levels=(0.5,2.5,), colors=('k','k'), linewidths=0.5)
        pc = ax.pcolormesh(xx, yy, bed, cmap=Zcmap, 
            vmin=-2000, vmax=2000, alpha=Zalpha)
        ax.set_aspect('equal')

        ax.spines[['left', 'right', 'top', 'bottom']].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])
        ax.text(0, 1, alphabet[i], transform=ax.transAxes,
            fontweight='bold', fontsize=fs)
        ax.tick_params(labelsize=fs)

    for basin in basins:
        levelset = np.load(f'../../issm/{basin}/data/geom/ocean_levelset.npy')
        mesh = np.load(f'../../issm/{basin}/data/geom/mesh.npy', allow_pickle=True)
        yi_glads = np.zeros(levelset.shape)*np.nan
        yi_glads[levelset>0] = np.load(f'data/CV_{basin}_f_glads.npy')[:, index]
        yi_rf = np.zeros(levelset.shape)*np.nan
        yi_rf[levelset>0] = np.load(f'data/CV_{basin}_f_rf.npy')[:, index]
        
        ni_glads = np.zeros(levelset.shape)*np.nan
        ni_glads[levelset>0] = np.load(f'data/CV_{basin}_N_glads.npy')[:, index]
        ni_rf = np.zeros(levelset.shape)*np.nan
        ni_rf[levelset>0] = np.load(f'data/CV_{basin}_N_rf.npy')[:, index]

        Y_glads = np.concatenate((Y_glads, np.load(f'data/CV_{basin}_f_glads.npy').flatten()))
        Y_rf = np.concatenate((Y_rf, np.load(f'data/CV_{basin}_f_rf.npy').flatten()))
        N_glads = np.concatenate((N_glads, np.load(f'data/CV_{basin}_N_glads.npy').flatten()))
        N_rf = np.concatenate((N_rf, np.load(f'data/CV_{basin}_N_rf.npy').flatten()))

        phi_bed = rhofw*g*np.load(f'../../issm/{basin}/data/geom/bed.npy')[levelset>0,None]
        thick = np.load(f'../../issm/{basin}/data/geom/thick.npy')[levelset>0,None]

        pot_glads = phi_bed + rhoi*g*thick*np.load(f'data/CV_{basin}_f_glads.npy')
        pot_rf = phi_bed + rhoi*g*thick*np.load(f'data/CV_{basin}_f_rf.npy')
        print(1 - np.nanvar(pot_rf-pot_glads)/np.nanvar(pot_glads))
        phi_rf = np.concatenate((phi_rf, pot_rf.flatten()))
        phi_glads = np.concatenate((phi_glads, pot_glads.flatten()))

        
        mtri = Triangulation(mesh['x'], mesh['y'], mesh['elements']-1)


        pc0 = axs[0,0].tripcolor(mtri, yi_glads, vmin=0.5, vmax=1, cmap=cmocean.cm.dense)
        pc1 = axs[1,0].tripcolor(mtri, ni_glads/1e6, vmin=0, vmax=5, cmap=cmocean.cm.haline)

        pc2 = axs[0,1].tripcolor(mtri, (yi_rf - yi_glads), vmin=-0.1, vmax=0.1, cmap=cmocean.cm.balance)
        pc3 = axs[1,1].tripcolor(mtri, (ni_rf - ni_glads)/1e6, vmin=-1, vmax=1, cmap=cmocean.cm.balance)

        outline = np.load(f'../../data/ANT_Basins/basin_{basin}.npy')
        for ax in axs.flat:
            ax.plot(outline[:,0], outline[:,1], color='k', linestyle='solid', linewidth=1)


    cb0 = fig.colorbar(pc0, cax=caxs[0,0], orientation='horizontal')
    cb0.set_label('Flotation fraction (-)', fontsize=fs)
    cb1 = fig.colorbar(pc1, cax=caxs[1,0], orientation='horizontal')
    cb1.set_label('Effective pressure (MPa)', fontsize=fs)
    cb2 = fig.colorbar(pc2, cax=caxs[0,1], orientation='horizontal')
    cb2.set_label(r'$\Delta$ Flotation fraction (-)', fontsize=fs)
    cb2.set_ticks([-0.1, -0.05, 0, 0.05, 0.1])
    cb3 = fig.colorbar(pc3, cax=caxs[1,1], orientation='horizontal')
    cb3.set_label(r'$\Delta$ Effective pressure (MPa)', fontsize=fs)

    for cax in caxs.flat:
        cax.xaxis.set_label_position('top')
        cax.tick_params(labelsize=fs)
        # cax.xaxis.tick_top()

    dxytext = np.array([
        [-500e3, -400e3],
        [-750e3, 0],
        [200e3, -100e3],
        [650e3, 500e3],
        [-0.75e6, 500e3],
        [0, -500e3],
    ])

    ha = [
        'right',
        'right',
        'left',
        'left',
        'right',
        'left',
    ]

    va = [
        'top',
        'center',
        'bottom',
        'center',
        'bottom',
        'top',
    ]

    for p in range(len(profile_labels)):
        basin = profile_basins[p]
        num = profile_numbers[p]
        flowline = np.load(f'../../issm/{basin}/data/geom/flowline_{num:02d}.npy')
        ss,xx,yy = flowline

        axs[1,0].plot(xx, yy, color='w', linewidth=1)
        tx = xx[0] + dxytext[p,0]
        ty = yy[0] + dxytext[p,1]
        
        axs[1,0].text(tx, ty, profile_labels[p],
            ha=ha[p], va=va[p], fontsize=fs)
        axs[1,0].plot((xx[0], tx), (yy[0], ty), color='k', linewidth=0.65)
    
    ####################################################################3
    ## Hexbin

    gs1 = GridSpecFromSubplotSpec(3, 3, gs[1,2],
        height_ratios=(10, 80, 10),
        width_ratios=(20, 80, 5))
    gs2 = GridSpecFromSubplotSpec(3, 3, gs[3,2],
        height_ratios=(10, 80, 10),
        width_ratios=(20, 80, 5))
    # cgs = GridSpecFromSubplotSpec(3, 1, gs[1:,-1],
    #     height_ratios=(25, 50, 25),
    # )
    ax1 = fig.add_subplot(gs1[1,1])
    ax2 = fig.add_subplot(gs2[1,1])
    # cax = fig.add_subplot(cgs[1:-1])
    cax = fig.add_subplot(gs[0,-1])

    fmin = 0.75
    hb = ax1.hexbin(Y_glads, Y_rf, cmap=cmocean.cm.rain, gridsize=50,
        extent=(fmin, 1, fmin, 1))
    ax1.grid()
    ax1.set_xlim([fmin, 1])
    ax1.set_ylim([fmin, 1])
    mask = np.logical_and(Y_glads>=0, Y_glads<=1)
    r2f = 1 - np.nanvar(Y_rf[mask] - Y_glads[mask])/np.nanvar(Y_glads[mask])
    print('r2f:', r2f)
    ax1.set_title(f'$R^2$ = {r2f:.3f}', fontsize=fs)
    phi_bed = rhofw*g*bed
    
    Nmin = 0
    Nmax = 5
    ax2.hexbin(N_glads/1e6, N_rf/1e6, cmap=cmocean.cm.rain,
        gridsize=50, extent=(Nmin, Nmax, Nmin, Nmax))
    ax2.grid()
    ax2.set_xlim([Nmin, Nmax])
    ax2.set_ylim([Nmin, Nmax])
    r2N = 1 - np.nanvar(N_rf[mask] - N_glads[mask])/np.nanvar(N_glads[mask])
    print('r2N:', r2N)
    ax2.set_title(f'$R^2$ = {r2N:.3f}', fontsize=fs)
    
    ax1.set_aspect('equal')
    ax2.set_aspect('equal')
    
    ax1.set_xlabel('GlaDS flotation fraction (-)', fontsize=fs)
    ax1.set_ylabel('RF flotation fraction (-)', fontsize=fs)
    ax2.set_xlabel('GlaDS effective pressure (MPa)', fontsize=fs)
    ax2.set_ylabel('RF effective pressure (MPa)', fontsize=fs)
    
    cbar = fig.colorbar(hb, cax=cax, 
        shrink=0.65, orientation='horizontal')
    cbar.set_label(label='Counts (N={:.3e})'.format(len(N_rf)), fontsize=fs)
    # cax.xaxis.tick_top()
    cax.xaxis.set_label_position('top')
    cax.tick_params(labelsize=fs)

    for i,ax in enumerate((ax1, ax2)):
        ax.tick_params(labelsize=fs)
        ax.text(1.15, 1, alphabet[i+4], transform=axs[i,-1].transAxes,
            fontweight='bold', fontsize=fs)

    r2phi = 1 - np.nanvar(phi_rf - phi_glads)/np.nanvar(phi_glads)
    print('r2phi:', r2phi)
    r2phi = 1 - np.nanvar(phi_rf[mask] - phi_glads[mask])/np.nanvar(phi_glads[mask])
    print('r2phi:', r2phi)

    fig.savefig('figures/continent_pred_error.png', dpi=400)

if __name__=='__main__':
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
    plot_error(basins)

