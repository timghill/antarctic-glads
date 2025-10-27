"""
Assess residual transient behaviour in steady-state simulations
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.tri import Triangulation
import cmocean
from scipy import stats

def trend(x,y):
    lr = stats.linregress(x, y)
    return lr.slope

def compute_change(basin):
    pattern = '/home/tghill/scratch/antarctic-glads/issm/{basin}/glads/RUN/output_{jobid:03d}/steady/{field}.npy'
    dNdt = np.zeros(100)
    dSdt = np.zeros(100)
    dHdt = np.zeros(100)

    for i in range(100):
        jobid = i+1
        print(f'{jobid:03d}/100')
        ff = np.load(pattern.format(basin=basin, jobid=jobid, field='ff'), mmap_mode='r')
        N = np.load(pattern.format(basin=basin, jobid=jobid, field='N'), mmap_mode='r')
        S = np.load(pattern.format(basin=basin, jobid=jobid, field='S'), mmap_mode='r')
        hs = np.load(pattern.format(basin=basin, jobid=jobid, field='h_s'), mmap_mode='r')
        tt = np.load(pattern.format(basin=basin, jobid=jobid, field='time'), mmap_mode='r')

        Nslope = (N[:,-1] - N[:,-10])/(tt[-1] - tt[-10])
        Sslope = (S[:,-1] - S[:,-10])/(tt[-1] - tt[-10])
        Hslope = (hs[:,-1] - hs[:,-10])/(tt[-1] - tt[-10])

        dNdt[i] = np.quantile(np.abs(Nslope), 0.95)
        dSdt[i] = np.quantile(np.abs(Sslope), 0.95)
        dHdt[i] = np.quantile(np.abs(Hslope), 0.95)

    np.save(f'{basin}/quantiles_dNdt.npy', dNdt)
    np.save(f'{basin}/quantiles_dSdt.npy', dSdt)
    np.save(f'{basin}/quantiles_dHdt.npy', dHdt)
    return

def analyze_change(basin):
    dNdt = np.load(f'{basin}/quantiles_dNdt.npy')
    dSdt = np.load(f'{basin}/quantiles_dSdt.npy')
    dHdt = np.load(f'{basin}/quantiles_dHdt.npy')
    
    # Define thresholds
    thresholdN = 0.05*2e6
    thresholdH = 0.05*0.25
    thresholdS = 0.05*1
    
    fractionN = len(dNdt[dNdt<thresholdN])/100
    fractionH = len(dHdt[dHdt<thresholdH])/100
    fractionS = len(dSdt[dSdt<thresholdS])/100
    print('Fraction of converged runs:')
    print('Pressure:', fractionN)
    print('Sheet:', fractionS)
    print('Channel:', fractionH)

    # CDF plots
    fig,axs = plt.subplots(ncols=2, nrows=2)
    # axs[0,0].hist(dNdt/1e3, bins=10)
    axs[0,0].plot(np.sort(dNdt)/1e3, np.arange(1,101))
    # axs[0,0].axvline(thresholdN/1e3, color='red')
    axs[0,0].set_xlabel('dN/dt (kPa/yr)')

    # axs[0,1].hist(dHdt, bins=10)
    axs[0,1].plot(np.sort(dHdt), np.arange(1,101))
    # axs[0,1].axvline(thresholdH, color='red')
    axs[0,1].set_xlabel('dH/dt (m/yr)')

    # axs[1,0].hist(dSdt, bins=10)
    axs[1,0].plot(np.sort(dSdt), np.arange(1,101))
    # axs[1,0].axvline(thresholdS, color='red')
    axs[1,0].set_xlabel('dS/dt (m$^2$/yr)')

    axs[0,0].set_ylabel('Number of converged runs')
    axs[1,0].set_ylabel('Number of converged runs')

    for ax in axs.flat:
        ax.grid()

    fig.tight_layout()

    fig.savefig(f'{basin}/rateofchange.png', dpi=400)

    quantN = np.quantile(dNdt, 0.95)
    quantS = np.quantile(dSdt, 0.95)
    quantH = np.quantile(dHdt, 0.95)
    print('Pressure quantile:', quantN)
    print('Sheet quantile:', quantH)
    print('Channel quantile:', quantS)

if __name__=='__main__':
    basin = 'G-H'
    compute_change(basin)
    analyze_change(basin)
