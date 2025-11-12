import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import griddata

colors = ('#89b6bc', '#0d7d87')

def main():
    nrun = 100
    ss,xx,yy = np.load('../data/geom/flowline_00.npy')
    mesh = np.load('../data/geom/mesh.npy', allow_pickle=True)
    meshxy = (mesh['x'], mesh['y'])
    levelset = np.load('../data/geom/ocean_levelset.npy')
    levelset_interp = griddata(meshxy, levelset, (xx,yy), method='nearest')
    
    uu_calc_C = np.load('solutions/u_glads_para_sensitivity_calc_C.npy')[:, :nrun]
    uu_calc_C_flowline = griddata(meshxy, uu_calc_C, (xx,yy), method='linear')

    # uu_const_C = np.load('solutions/u_glads_para_sensitivity_const_C.npy')[:, :nrun]
    # uu_const_C_flowline = griddata(meshxy, uu_const_C, (xx,yy), method='linear')

    uu_ref = np.load('solutions/u_glads_future.npy')
    uu_ref_flowline = griddata(meshxy, uu_ref, (xx,yy), method='linear')

    retreat_mask = levelset_interp>0
    fig,axs = plt.subplots(ncols=2, nrows=1, sharey=True, figsize=(6, 4), width_ratios=(60,100))

    ax = axs[0]
    ax.hist(uu_calc_C_flowline[retreat_mask][0,:], edgecolor='k',
        facecolor=colors[1], orientation='horizontal')
    ax.axhline(uu_ref_flowline[retreat_mask][0], color='k')
    ax.set_ylabel('Grounding line speed (m/year)')
    ax.set_xlabel(f'Count (N={nrun})')
    ax.text(0.025, 1.025, 'a', transform=ax.transAxes,
        fontweight='bold', va='bottom')
    ax.set_xlim([20, 0])
    ax.grid()

    ax = axs[1]
    ax.plot(ss[retreat_mask]/1e3, uu_calc_C_flowline[retreat_mask],
        color=colors[1], alpha=0.2)
    ax.plot(ss[retreat_mask]/1e3, uu_ref_flowline[retreat_mask],
        color='k')
    ax.grid()
    ax.set_xlabel('Distance from present-day grounding line')
    ax.set_ylabel('Speed (m/year)')
    ax.text(0.025, 1.025, 'b', transform=ax.transAxes,
        fontweight='bold', va='bottom')
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position('right')

    # ax = axs[1,1]
    # ax.plot(ss[retreat_mask]/1e3, uu_calc_C_flowline[retreat_mask],
    #     color=colors[1], alpha=0.2)
    # ax.plot(ss[retreat_mask]/1e3, uu_ref_flowline[retreat_mask],
    #     color='k')
    # ax.grid()
    # ax.set_xlabel('Distance from present-day grounding line')
    # ax.set_ylabel('Speed (m/year)')
    # ax.text(0.025, 1.025, 'd', transform=ax.transAxes,
    #     fontweight='bold', va='bottom')
    

    # for ax in axs[:,1]:
    #     ax.yaxis.tick_right()
    #     ax.yaxis.set_label_position('right')
    
    # axs[0,0].set_title('Constant C', fontsize=10)
    # axs[0,1].set_title('Calculate C', fontsize=10)

    fig.subplots_adjust(left=0.12, right=0.925, bottom=0.12, top=0.925, wspace=0.05,
        hspace=0.35)
    fig.savefig('solutions/para_sensitivity.png', dpi=400)

if __name__=='__main__':
    main()
