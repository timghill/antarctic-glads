import os
import argparse

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.tri import Triangulation
from matplotlib import colors as mpc
from matplotlib import cm
import cmocean
import scipy.stats

from utils.tools import import_config

xclose = [1.2e6, 2.1e6]
yclose = [2.e5, 10e5]

def plots(config):
    figures = 'figures'
    if not os.path.exists(figures):
        os.makedirs(figures)
    
    resdir = config.sim_dir

    theta = np.loadtxt('theta_physical.csv', delimiter=',', skiprows=1)
    labels = ['Sheet conductivity', 'Channel conductivity', 'Bump aspect ratio', 'Sheet-channel width', 'Ice-flow factor']

    mesh = np.load(os.path.join(resdir, '../data/geom/mesh.npy'), allow_pickle=True)
    mtri = Triangulation(mesh['x'], mesh['y'], mesh['elements']-1)

    # RUNTIME
    time = np.load(os.path.join(resdir, 'runtime.npy')).squeeze()
    fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(6, 6), sharex=False, sharey=True)
    for i in range(5):
        ax = axs.flat[i]
        ax.scatter(theta[:, i], time/3600)
        ax.set_xlabel(labels[i])
        ax.grid()
        theta_nan = theta[np.isnan(time), i]
        # print(len(theta_nan))
        print(theta_nan)
        for ti in theta_nan:
            # print(ti)
            ax.axvline(ti, color='r', linewidth=0.5)
        # ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_ylim([0, 24])

    for ax in axs[:, 0].flat:
        ax.set_ylabel('Runtime (h)')

    fig.subplots_adjust(bottom=0.1, top=0.95, left=0.1, right=0.95, hspace=0.4)

    fig.savefig(os.path.join(figures, 'ensemble_runtime.png'), dpi=400)
    plt.close(fig)


    # EFFECTIVE PRESSURE
    N = np.load(os.path.join(resdir, 'N.npy'))[:, -1, :]
    print(N.shape)
    N_ensmean = np.nanmean(N, axis=-1)

    Q = np.abs(np.load(os.path.join(resdir, 'Q.npy'))[:,-1,:])
    Q_tot = np.sum(Q, axis=0)

    N_mean = np.load(os.path.join(resdir, 'N_mean.npy'))[-1]
    fw_mean = np.load(os.path.join(resdir, 'ff_mean.npy'))[-1]
    simnum = np.argsort(N_mean)[int(len(N_mean)//2)]
    print('simnum:', simnum)
    N_ensmean = N[:,simnum]

    print('Theta values:')
    print(theta[simnum])

    fw = np.load(os.path.join(resdir, 'ff.npy'))[:,-1,:]
    fw_ensmean = fw[:,simnum]
    Q_ensmean = Q[:,simnum]
    # fw_ensmean = np.nanmean(fw, axis=-1)
    # # N_mean = N_mean[np.isfinite(N_mean)]
    # print(N_mean.shape)
    fig,ax = plt.subplots()
    ax.hist(N_mean/1e3, bins=10, edgecolor='k', range=(0, 2.5e3))
    ax.axvline(N_mean[simnum]/1e3, color='k')
    ax.set_xlabel('N (KPa)')
    ax.set_ylabel('Count')
    fig.savefig(os.path.join(figures, 'ensemble_hist_N.png'), dpi=400)
    plt.close(fig)

    fig,ax = plt.subplots()
    ax.hist(fw_mean, bins=10, edgecolor='k', range=(0.8, 1.1))
    ax.axvline(fw_mean[simnum], color='k')
    ax.set_xlabel('Flotation fraction')
    ax.set_ylabel('Count')
    fig.savefig(os.path.join(figures, 'ensemble_hist_ff.png'), dpi=400)
    plt.close(fig)

    def plot_map(mtri, Z, label=None, **kwargs):
        fig,ax = plt.subplots()
        C = ax.tripcolor(mtri, Z, **kwargs)
        fig.colorbar(C, label=label)
        ax.set_aspect('equal')
        ax.spines[['left', 'right', 'top', 'bottom']].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.subplots_adjust(left=0.025, bottom=0.025, right=0.95, top=0.975)

        channels = np.where(Q_ensmean>10)[0]
        for k in channels:
            channelx = mesh['x'][mesh['connect_edge'][k]]
            channely = mesh['y'][mesh['connect_edge'][k]]
            lwmin = 0.5
            lwmax = 5
            logQ = np.log10(Q_ensmean[k])
            logQmin = 1
            logQmax = 2
            lw = (logQ-logQmin)/(logQmax - logQmin)
            lw = min(lwmax, max(lwmin, lw))
            ax.plot(channelx, channely, color='k', linewidth=lw)
        
        ax.set_xlim(xclose)
        ax.set_ylim(yclose)
        return fig

    # fig,ax = plt.subplots()
    # pc = ax.tripcolor(mtri, N_ensmean/1e3, vmin=0, vmax=5000, cmap=cmocean.cm.haline)
    # fig.colorbar(pc, label='N (KPa)')
    # ax.set_aspect('equal')
    fig = plot_map(mtri, N_ensmean/1e3, vmin=0, vmax=5000, 
        cmap=cmocean.cm.haline, label='N (KPa)')
    fig.savefig(os.path.join(figures, 'ensemble_map_N.png'), dpi=400)
    plt.close(fig)

    # fig,ax = plt.subplots()
    fig = plot_map(mtri, fw_ensmean, vmin=0, vmax=1, 
        cmap=cmocean.cm.dense, label='Flotation fraction')
    fig.savefig(os.path.join(figures, 'ensemble_map_ff.png'), dpi=400)
    plt.close(fig)
    # pc = ax.tripcolor(mtri, fw_ensmean, vmin=0, vmax=1, cmap=cmocean.cm.dense)
    # fig.colorbar(pc, label='Flotation fraction')
    # ax.set_aspect('equal')

    # PLOT QUANTILES OF Q, N
    fig, axs = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(8,5))

    def edge_plot(ax, mtri, Q, vmin=1, vmax=100, **kwargs):
        channels = np.where(Q>vmin)[0]
        lwmin = 0.5
        lwmax = 3
        logQmin = np.log10(vmin)
        logQmax = np.log10(vmax)
        cnorm = mpc.Normalize(vmin=logQmin, vmax=logQmax)
        sm = cm.ScalarMappable(cnorm, cmap=cmocean.cm.turbid)

        for k in channels:
            logQ = np.log10(Q[k])
            cc = cmocean.cm.turbid(cnorm(logQ))
            channelx = mesh['x'][mesh['connect_edge'][k]]
            channely = mesh['y'][mesh['connect_edge'][k]]
            lw = (logQ-logQmin)/(logQmax - logQmin)
            lw = min(lwmax, max(lwmin, lw))
            ax.plot(channelx, channely, linewidth=lw,
                color=cc, **kwargs)
        return sm

    print(Q_tot)
    # Q_finite = Q_tot
    Q_finite = Q_tot.copy()
    nQ = len(Q_finite)
    lq = int(0.05*nQ)
    mq = int(0.5*nQ)
    uq = int(0.95*nQ)
    # Q_sims = np.argsort(Q_tot)[np.array([4, 49, 94])]
    Q_sims = np.argsort(Q_finite)[np.array([lq, mq, uq])]
    print('Q_sims:', Q_sims)
    N_sims = np.argsort(N_mean)[np.array([lq, mq, uq])]

    print('Parameter values')
    print('Q:')
    print(theta[Q_sims])
    print('N:')
    print(theta[N_sims])

    for i in range(3):
        qi = Q[:, Q_sims[i]]
        print(qi.sum())
        sm = edge_plot(axs[0,i], mtri, qi, vmin=10, vmax=1000)
        print(Q_tot[Q_sims[i]])
        axs[0,i].set_aspect('equal')
        axs[0,i].tripcolor(mtri, N[:, Q_sims[i]]/1e3, vmin=0, vmax=5000,
            cmap=cmocean.cm.haline, alpha=0.5, antialiased=True)

        pc = axs[1,i].tripcolor(mtri, N[:, N_sims[i]]/1e3,
            vmin=0, vmax=5000, cmap=cmocean.cm.haline)
        edge_plot(axs[1,i], mtri, qi, vmin=10, vmax=1000,
            alpha=0.3)
        axs[1,i].set_aspect('equal')

    for ax in axs.flat:
        ax.spines[['left', 'right', 'top', 'bottom']].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])


    fig.subplots_adjust(bottom=0.025, left=0.025, right=0.975, top=0.975,
        hspace=0.1, wspace=0.05)

    cbar_S = fig.colorbar(sm, ax=axs[0], label='log$_{10}$ Q (m$^3$ s$^{-1}$)')
    fig.colorbar(pc, ax=axs[1], label='N (KPa)')
    fig.savefig(os.path.join(figures, 'ensemble_map_quantiles.png'), dpi=800)



    # RUNTIME
    time = np.load(os.path.join(resdir, 'runtime.npy')).squeeze()
    fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(6, 6), sharex=False, sharey=True)
    for i in range(5):
        ax = axs.flat[i]
        ax.scatter(theta[:, i], time/3600)
        ax.set_xlabel(labels[i])
        ax.grid()
        theta_nan = theta[np.isnan(time), i]
        # print(len(theta_nan))
        print(theta_nan)
        for ti in theta_nan:
            # print(ti)
            ax.axvline(ti, color='r', linewidth=0.5)
        # ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_ylim([0, 48])

    for ax in axs[:, 0].flat:
        ax.set_ylabel('Runtime (h)')

    fig.subplots_adjust(bottom=0.1, top=0.95, left=0.14, right=0.95, hspace=0.4)

    fig.savefig(os.path.join(figures, 'ensemble_runtime.png'), dpi=400)


    fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(6, 6), sharex=False, sharey=True)
    for i in range(5):
        ax = axs.flat[i]
        ax.scatter(theta[:, i], N_mean/1e3)
        ax.set_xlabel(labels[i])
        ax.grid()
        theta_nan = theta[np.isnan(time), i]
        # print(len(theta_nan))
        for ti in theta_nan:
            # print(ti)
            ax.axvline(ti, color='r', linewidth=0.5)
        # ax.set_yscale('log')
        ax.set_xscale('log')
        # ax.set_ylim([0, 24])

    for ax in axs[:, 0].flat:
        ax.set_ylabel('N (KPa)')

    fig.subplots_adjust(bottom=0.1, top=0.95, left=0.14, right=0.95, hspace=0.4)

    fig.savefig(os.path.join(figures, 'ensemble_N.png'), dpi=400)
    plt.close(fig)
    # plt.show()

    fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(6, 6), sharex=False, sharey=True)
    Q = np.abs(np.load(os.path.join(resdir, 'Q.npy'))[:, :, :])
    Qsum = np.sum(Q, axis=0)

    # print('Q:', Q.shape)
    # print('edge:', mesh['connect_edge'].shape)
    # dx = mesh['x'][mesh['connect_edge'][Q>10, :]]
    # print('dx:', dx.shape)
    # # Q_extent = np.max(
    for i in range(5):
        ax = axs.flat[i]
        ax.scatter(theta[:, i], Qsum[-1,:])
        ax.set_xlabel(labels[i])
        ax.grid()
        theta_nan = theta[np.isnan(time), i]
        # print(len(theta_nan))
        for ti in theta_nan:
            # print(ti)
            ax.axvline(ti, color='r', linewidth=0.5)
        ax.set_yscale('log')
        ax.set_xscale('log')
        # ax.set_ylim([0, 24])

    for ax in axs[:, 0].flat:
        ax.set_ylabel('Total Q (m$^3$ s$^{-1}$)')

    fig.subplots_adjust(bottom=0.1, top=0.95, left=0.14, right=0.95, hspace=0.4)

    fig.savefig(os.path.join(figures, 'ensemble_Q.png'), dpi=400)
    plt.close(fig)

    fig,ax = plt.subplots()
    nt = N.shape[1]
    tt = np.linspace(0, 0.1*(nt-1), nt)
    ax.plot(tt, Qsum)
    fig.savefig(os.path.join(figures, 'ensemble_Q_timeseries.png'), dpi=400)
    # # plt.show()
    return

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    args = parser.parse_args()
    config = import_config(args.config)
    plots(config)
