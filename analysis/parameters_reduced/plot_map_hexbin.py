import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.tri import Triangulation
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import Rectangle
import cmocean
import netCDF4 as nc

rhow = 1023.0
rhofw = 1000.0
rhoi = 917.0
g = 9.81

# profile_basins = ['G-H', 'G-H', 'C-Cp', 'B-C', 'Jpp-K', 'Cp-D']
# profile_numbers = [0, 1, 0, 0, 0, 0]
# profile_labels = ['Thwaites', 'PIG', 'Denman', 'Lambert', 'Recovery', 'Totten']
# alphabet = ['(a)', '(b)', '(d)', '(e)', '(c)', '(f)']


profile_basins = ['G-H', 'G-H', 'C-Cp', 'B-C', 'Cp-D']
profile_numbers = [0, 1, 0, 0, 0]
profile_labels = ['Thwaites', 'PIG', 'Denman', 'Lambert', 'Totten']
alphabet = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)']

focus_basins = ['B-C', 'Ep-F']

bedmachine = '../../data/bedmachine/BedMachineAntarctica-v3.nc'

Ncmap = cmocean.cm.matter
Zcmap = cmocean.cm.gray
Zalpha = 0.5
fs = 7

def plot_error(basins, index=15):

    # Set a stride length > 1 to approximately plot BM more quickly
    # at continent-wide scale we don't need 5 km resolution
    dx = 16
    with nc.Dataset(bedmachine, 'r') as bm:
        print('Reding bedmachine')
        mask = bm['mask'][::dx, ::dx].astype(int)
        x = bm['x'][::dx].astype(np.float32)
        y = bm['y'][::dx].astype(np.float32)
        bed = bm['bed'][::dx, ::dx].astype(np.float32)
        surf = bm['surface'][::dx, ::dx].astype(np.float32)
    bed[mask==0] = np.nan
    mask[surf>2000] = 2
    xx,yy = np.meshgrid(x,y)
    print('done bedmachine')

    xmin = np.min(xx[~np.isnan(bed)])
    xmax = np.max(xx[~np.isnan(bed)])
    ymin = np.min(yy[~np.isnan(bed)])
    ymax = np.max(yy[~np.isnan(bed)])

    print('Creating figures')
    fig1 = plt.figure(figsize=(7, 3.5))
    nrows = 2
    ncols = 2
    gs = GridSpec(ncols=ncols, nrows=2,
        hspace=0.2, wspace=0.1, left=0.03, bottom=0.0, right=0.925, top=0.925,
        height_ratios=(3, 100),
        width_ratios=(100, 100),
    )
    # axs = np.array([[fig.add_subplot(gs[2*i+1,j], facecolor='none') for j in range(2)] for i in range(nrows)])
    axs = np.array([
        fig1.add_subplot(gs[1,0]),
        fig1.add_subplot(gs[1,1]),
    ])
    caxs = np.array([fig1.add_subplot(gs[0,j]) for j in range(2)])

    efig = plt.figure(figsize=(7,6))
    egs = GridSpec(nrows=4, ncols=3,
        hspace=0.15, wspace=0.1, left=0.025, bottom=0.025, right=0.975, top=0.92,
        height_ratios=(3, 100, 3, 100),
        width_ratios=(110, 90, 90),
    )
    eaxs = np.array([
        [efig.add_subplot(egs[1,0])],
        [efig.add_subplot(egs[3,0])],
    ])
    ecaxs = np.array([
        [efig.add_subplot(egs[0,0])],
        [efig.add_subplot(egs[2,0])],
    ])

    subgs1 = GridSpecFromSubplotSpec(2, 1, egs[:2,1],
        hspace=0)

    subgs2 = GridSpecFromSubplotSpec(2, 1, egs[2:4,1],
        hspace=0)
    faxs = np.array([
        []
    ])

    Y_glads = np.array([])
    Y_rf = np.array([])
    N_glads = np.array([])
    N_rf = np.array([])
    phi_glads = np.array([])
    phi_rf = np.array([])
    print('Contouring bedmachine')
    for i,ax in enumerate(axs.flat):
        ax.contour(xx, yy, mask, levels=(0.5,2.5,), colors=('k','k'), linewidths=0.5)
        pc = ax.pcolormesh(xx, yy, bed, cmap=Zcmap, 
            vmin=-2000, vmax=2000, alpha=Zalpha,
            rasterized=True)
        ax.set_aspect('equal')

        ax.spines[['left', 'right', 'top', 'bottom']].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])
        ax.text(0, 1, alphabet[i], transform=ax.transAxes,
            fontweight='bold', fontsize=fs)
        ax.tick_params(labelsize=fs)
    
    for i,ax in enumerate(eaxs.flat):
        ax.contour(xx, yy, mask, levels=(0.5,2.5,), colors=('k','k'), linewidths=0.5)
        pc = ax.pcolormesh(xx, yy, bed, cmap=Zcmap, 
            vmin=-2000, vmax=2000, alpha=Zalpha,
            rasterized=True)
        ax.set_aspect('equal')

        ax.spines[['left', 'right', 'top', 'bottom']].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])
        ax.text(0, 1, alphabet[4*i], transform=ax.transAxes,
            fontweight='bold', fontsize=fs)
        ax.tick_params(labelsize=fs)
    print('done contouring')

    print('Plotting GlaDS pcolormesh')
    focus_index = 0
    for basin in basins:
        levelset = np.load(f'../../issm/{basin}/data/geom/ocean_levelset.npy')
        mesh = np.load(f'../../issm/{basin}/data/geom/mesh.npy', allow_pickle=True)
        yi_glads = np.zeros(levelset.shape)*np.nan
        yi_glads[levelset>0] = np.load(f'data/pred_{basin}_f_glads.npy')[:, index]
        yi_rf = np.zeros(levelset.shape)*np.nan
        yi_rf[levelset>0] = np.load(f'data/CV_{basin}_f_rf.npy')[:, index]
        
        ni_glads = np.zeros(levelset.shape)*np.nan
        ni_glads[levelset>0] = np.load(f'data/pred_{basin}_N_glads.npy')[:, index]
        ni_rf = np.zeros(levelset.shape)*np.nan
        ni_rf[levelset>0] = np.load(f'data/CV_{basin}_N_rf.npy')[:, index]

        Y_glads = np.concatenate((Y_glads, np.load(f'data/pred_{basin}_f_glads.npy').flatten()))
        Y_rf = np.concatenate((Y_rf, np.load(f'data/CV_{basin}_f_rf.npy').flatten()))
        N_glads = np.concatenate((N_glads, np.load(f'data/pred_{basin}_N_glads.npy').flatten()))
        N_rf = np.concatenate((N_rf, np.load(f'data/CV_{basin}_N_rf.npy').flatten()))

        phi_bed = rhofw*g*np.load(f'../../issm/{basin}/data/geom/bed.npy')[levelset>0,None]
        thick = np.load(f'../../issm/{basin}/data/geom/thick.npy')[levelset>0,None]

        pot_glads = phi_bed + rhoi*g*thick*np.load(f'data/pred_{basin}_f_glads.npy')
        pot_rf = phi_bed + rhoi*g*thick*np.load(f'data/CV_{basin}_f_rf.npy')
        phi_rf = np.concatenate((phi_rf, pot_rf.flatten()))
        phi_glads = np.concatenate((phi_glads, pot_glads.flatten()))

        
        mtri = Triangulation(mesh['x'], mesh['y'], mesh['elements']-1)


        pc0 = axs[0].tripcolor(mtri, yi_glads, vmin=0.75, vmax=1, cmap=cmocean.cm.dense, rasterized=True)
        pc1 = axs[1].tripcolor(mtri, ni_glads/1e6, vmin=0, vmax=4, cmap=cmocean.cm.haline, rasterized=True)

        pc2 = eaxs[0,0].tripcolor(mtri, (yi_rf - yi_glads), vmin=-0.1, vmax=0.1, cmap=cmocean.cm.balance, rasterized=True)
        pc3 = eaxs[1,0].tripcolor(mtri, (ni_rf - ni_glads)/1e6, vmin=-1, vmax=1, cmap=cmocean.cm.balance, rasterized=True)

        outline = np.load(f'../../data/ANT_Basins/basin_{basin}.npy')
        for ax in axs.flat:
            ax.plot(outline[:,0], outline[:,1], color='k', linestyle='solid', linewidth=1)
        for ax in eaxs[:,0].flat:
            ax.plot(outline[:,0], outline[:,1], color='k', linestyle='solid', linewidth=1)
        
        if basin in focus_basins:
            subax = efig.add_subplot(subgs1[focus_index])
            subax.tripcolor(mtri, (yi_rf - yi_glads), vmin=-0.1, vmax=0.1, cmap=cmocean.cm.balance, rasterized=True)
            subax.set_xlim([mesh['x'].min(), mesh['x'].max()])
            subax.set_ylim([mesh['y'].min(), mesh['y'].max()])

            subNax = efig.add_subplot(subgs2[focus_index])
            subNax.tripcolor(mtri, (ni_rf - ni_glads)/1e6, vmin=-1, vmax=1, cmap=cmocean.cm.balance, rasterized=True)
            subNax.set_xlim([mesh['x'].min(), mesh['x'].max()])
            subNax.set_ylim([mesh['y'].min(), mesh['y'].max()])

            subax.text(0,1,alphabet[1+focus_index],va='top',
                transform=subax.transAxes,fontsize=fs,fontweight='bold')
            subNax.text(0,1,alphabet[5+focus_index],va='top',
                transform=subNax.transAxes,fontsize=fs,fontweight='bold')

            for subax in (subax,subNax):
                subax.set_aspect('equal')
                subax.spines[['left', 'right', 'top', 'bottom']].set_visible(False)
                subax.set_xticks([])
                subax.set_yticks([])
                subax.plot(outline[:,0], outline[:,1], 
                    color='k', linestyle='solid', linewidth=1)

                xll = np.min(mesh['x'])
                yll = np.min(mesh['y'])

                rwidth = 250e3
                xll -= 0.75*rwidth
                scalebar = Rectangle((xll, yll), rwidth, 
                    rwidth/10, facecolor='k', edgecolor='k', clip_on=False)
                subax.add_patch(scalebar)
                subax.text(xll + rwidth/2, yll+100e3, f'{rwidth//1000:.0f} km', ha='center', fontsize=fs)

            focus_index += 1

    print('done GlaDS pcolormesh')

    cb0 = fig1.colorbar(pc0, cax=caxs[0], orientation='horizontal')
    cb0.set_label('Flotation fraction (-)', fontsize=fs)
    cb1 = fig1.colorbar(pc1, cax=caxs[1], orientation='horizontal')
    cb1.set_label('Effective pressure (MPa)', fontsize=fs)
    
    cb2 = efig.colorbar(pc2, cax=ecaxs[0,0], orientation='horizontal')
    cb2.set_label(r'$\Delta$ Flotation fraction (-)', fontsize=fs)
    cb2.set_ticks([-0.1, -0.05, 0, 0.05, 0.1])
    cb3 = efig.colorbar(pc3, cax=ecaxs[1,0], orientation='horizontal')
    cb3.set_label(r'$\Delta$ Effective pressure (MPa)', fontsize=fs)

    for cax in caxs.flat:
        cax.xaxis.set_label_position('top')
        cax.tick_params(labelsize=fs)
    for cax in ecaxs.flat:
        cax.xaxis.tick_top()
        cax.xaxis.set_label_position('top')
        cax.tick_params(labelsize=fs)
    
    scalebar = Rectangle((xmin, ymin), 1000e3, 100e3, facecolor='k', edgecolor='k')
    axs[0].add_patch(scalebar)
    axs[0].text(xmin + 500e3, ymin+200e3, '1000 km', ha='center', fontsize=fs)

    scalebar = Rectangle((xmin, ymin+250e3), 1000e3, 100e3, facecolor='k', edgecolor='k')
    eaxs[0,0].add_patch(scalebar)
    eaxs[0,0].text(xmin + 500e3, ymin+250e3+200e3, '1000 km', ha='center', fontsize=fs)

    print('Labelling basins and flowlines')
    dxytext = np.array([
        [-500e3, -400e3],
        [-750e3, 0],
        [200e3, -100e3],
        [650e3, 500e3],
        # [-0.6e6, 500e3],
        [0, -500e3],
    ])

    ha = [
        'right',
        'right',
        'left',
        'left',
        # 'right',
        'left',
    ]

    va = [
        'top',
        'center',
        'bottom',
        'center',
        # 'bottom',
        'top',
    ]

    for p in range(len(profile_labels)):
        basin = profile_basins[p]
        num = profile_numbers[p]
        flowline = np.load(f'../../issm/{basin}/data/geom/flowline_{num:02d}.npy')
        ss,xx,yy = flowline

        axs[1].plot(xx, yy, color='w', linewidth=1)
        tx = xx[0] + dxytext[p,0]
        ty = yy[0] + dxytext[p,1]
        
        axs[1].text(tx, ty, profile_labels[p],
            ha=ha[p], va=va[p], fontsize=fs)
        axs[1].plot((xx[0], tx), (yy[0], ty), color='k', linewidth=0.65)

    ####################################################################3
    ## Label the basins in (a)
    # basins = [
    #     'B-C',
    #     'G-H',
    #     'Cp-D',
    #     'C-Cp',
    #     'Jpp-K',
    #     'J-Jpp',
    #     'Ep-F',
    # ]
    dxytext = np.array([
        [650e3, -100e3],
        [-1000e3, -300e3],
        [400e3, 0],
        [250e3, 0],
        [-650e3, 1400e3],
        [-400e3, 800e3],
        [0, -900e3],
    ])

    for p in range(len(basins)):
        basin = basins[p]
        mesh = np.load(f'../../issm/{basin}/data/geom/mesh.npy', allow_pickle=True)
        bx = np.mean(mesh['x'])
        by = np.mean(mesh['y'])
        tx = bx + dxytext[p][0]
        ty = by + dxytext[p][1]
        
        axs[0].text(tx, ty, basin, fontsize=fs)

    print('done text labels')
    fig1.savefig('figures/glads_map.png', dpi=400)
    efig.savefig('figures/pred_error.png', dpi=400)
    # return 
    print('Done pcolormesh plots')
    ####################################################################3
    ## Hexbin
    print('Starting Hexbin plots')
    pad = 5
    gs1 = GridSpecFromSubplotSpec(3, 3, egs[1,-1],
        height_ratios=(pad, 100-2*pad, pad),
        width_ratios=(1.5*pad, 100-2*pad, 0.5*pad),
        hspace=0, wspace=0)
    gs2 = GridSpecFromSubplotSpec(3, 3, egs[3,-1],
        height_ratios=(pad, 100-2*pad, pad),
        width_ratios=(1.5*pad, 100-2*pad, 0.5*pad),
        hspace=0, wspace=0)
    # cgs = GridSpecFromSubplotSpec(3, 1, gs[1:,-1],
    #     height_ratios=(25, 50, 25),
    # )
    ax1 = efig.add_subplot(gs1[1,1])
    ax2 = efig.add_subplot(gs2[1,1])
    # cax = fig.add_subplot(cgs[1:-1])
    cax = efig.add_subplot(egs[0,-1])

    fmin = 0.75
    ax1.set_xlim([fmin, 1])
    ax1.set_ylim([fmin, 1])
    ax1.set_aspect('equal')
    stride = 1 # For prototyping only; use stride=1 for production
    hb = ax1.hexbin(Y_glads[::stride], Y_rf[::stride], cmap=cmocean.cm.rain, gridsize=100,
        extent=(fmin, 1, fmin, 1), linewidths=(0.2,), vmin=0, vmax=3e4/stride,
        rasterized=True)
    ax1.grid()
    mask = np.logical_and(Y_glads>=0, Y_glads<=1)
    r2f = 1 - np.nanvar(Y_rf[mask] - Y_glads[mask])/np.nanvar(Y_glads[mask])
    print('r2f:', r2f)
    ax1.set_title(f'$R^2$ = {r2f:.3f}', fontsize=fs)
    
    
    Nmin = 0
    Nmax = 5
    ax2.set_xlim([Nmin, Nmax])
    ax2.set_ylim([Nmin, Nmax])
    ax2.set_aspect('equal')
    ax2.hexbin(N_glads[::stride]/1e6, N_rf[::stride]/1e6, cmap=cmocean.cm.rain,
        gridsize=100, extent=(Nmin, Nmax, Nmin, Nmax), linewidths=(0.1,),
        vmin=0, vmax=3e4/stride, rasterized=True)
    ax2.grid()
    r2N = 1 - np.nanvar(N_rf[mask] - N_glads[mask])/np.nanvar(N_glads[mask])
    print('r2N:', r2N)
    ax2.set_title(f'$R^2$ = {r2N:.3f}', fontsize=fs)

    
    ax1.set_xlabel('GlaDS flotation fraction (-)', fontsize=fs)
    ax1.set_ylabel('RF flotation fraction (-)', fontsize=fs)
    ax2.set_xlabel('GlaDS effective pressure (MPa)', fontsize=fs)
    ax2.set_ylabel('RF effective pressure (MPa)', fontsize=fs)
    
    cbar = efig.colorbar(hb, cax=cax, 
        shrink=0.65, orientation='horizontal')
    cbar.set_label(label='Counts (N={:,})'.format(len(N_rf)), fontsize=fs)
    cax.xaxis.tick_top()
    cax.xaxis.set_label_position('top')
    cax.tick_params(labelsize=fs)

    for i,ax in enumerate((ax1, ax2)):
        ax.tick_params(labelsize=fs)
        ax.text(-0.1, 1.1, alphabet[4*i+3], transform=ax.transAxes,
            fontweight='bold', fontsize=fs, ha='left', va='top')

    r2phi = 1 - np.nanvar(phi_rf - phi_glads)/np.nanvar(phi_glads)
    print('r2phi:', r2phi)
    r2phi = 1 - np.nanvar(phi_rf[mask] - phi_glads[mask])/np.nanvar(phi_glads[mask])
    print('r2phi:', r2phi)

    fig1.savefig('figures/glads_map.png', dpi=400)
    # fig1.savefig('../../manuscript/f01.png', dpi=400)
    # fig1.savefig('../../manusscript/f01.pdf', dpi=400)


    efig.savefig('figures/pred_error.png', dpi=400)
    # efig.savefig('../../manuscript/')
    # fig.savefig('figures/continent_pred_error.png', dpi=400)
    # fig.savefig('../../manuscript/f01.png', dpi=400)
    # fig.savefig('../../manuscript/f01.pdf', dpi=400)

if __name__=='__main__':
    basins = [
        'B-C',
        'G-H',
        'Cp-D',
        'C-Cp',
        'Jpp-K',
        'J-Jpp',
        'Ep-F',
    ]
    plot_error(basins, index=14)

