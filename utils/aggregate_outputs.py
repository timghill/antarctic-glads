"""
Collect outputs from individual simulations into .npy arrays.

Command line interface

    python -m utils.aggregate_outputs njobs
"""

import os
import sys
import importlib
import argparse
import pickle

import numpy as np

from utils.tools import import_config

def collect_issm_results(resdir, njobs, dtype=np.float32):
    """
    Collect outputs from individual simulations into .npy arrays.

    Parameters
    ----------    
    njobs : int
            Number of jobs to look for in the results directory
    
    dtype : type
            Data type to cast outputs into. Recommend np.float32
    """
    mesh = np.load('../data/geom/mesh.npy', allow_pickle=True)
    
    nodes = np.array([mesh['x'], mesh['y']]).T
    connect = mesh['elements'].astype(int)-1
    connect_edge = mesh['connect_edge'].astype(int)
    edge_length = mesh['edge_length']

    # Construct file patterns
    jobids = np.arange(1, njobs+1)

    resdir = os.path.join(resdir, 'RUN/output_{:03d}/steady/')
    respattern = os.path.join(resdir, '{}.npy')
    aggpattern = '{}.npy'
    testout = np.load(respattern.format(1, 'ff'))
    # nt = testout.shape[1]
    all_ff = np.zeros((mesh['numberofvertices'], njobs), dtype=dtype)
    all_S = np.zeros((len(edge_length), njobs), dtype=dtype)
    all_Q = np.zeros((len(edge_length), njobs), dtype=dtype)
    all_hs = np.zeros((mesh['numberofvertices'], njobs), dtype=dtype)
    all_N = np.zeros((mesh['numberofvertices'], njobs), dtype=dtype)
    all_runtime = np.zeros((1, njobs), dtype=dtype)
    for i,jobid in enumerate(jobids):
        print('Job %d' % jobid)

        try:
            ff = np.load(respattern.format(jobid, 'ff'))[:, -1]
            all_ff[:,i] = ff

            N = np.load(respattern.format(jobid, 'N'),mmap_mode='r')[:, -1]
            Q = np.load(respattern.format((jobid), 'Q'),mmap_mode='r')[:, -1]
            S = np.load(respattern.format((jobid), 'S'),mmap_mode='r')[:, -1]
            h_s = np.load(respattern.format((jobid), 'h_s'),mmap_mode='r')[:, -1]
            runtime = np.loadtxt(os.path.join(resdir, 'runtime').format(jobid))
            all_Q[:,i] = Q
            all_S[:,i] = S
            all_hs[:,i] = h_s
            all_runtime[:, i] = runtime
            all_N[:,i] = N
        except Exception as e:
            print(e)
            all_ff[:,i] = np.nan
            all_Q[:,i] = np.nan
            all_S[:,i] = np.nan
            all_hs[:,i] = np.nan
            all_N[:,i]= np.nan
            all_runtime[:, i] = np.nan

        
    np.save(aggpattern.format('ff'), all_ff)
    np.save(aggpattern.format('N'), all_N)
    np.save(aggpattern.format('S'), all_S)
    np.save(aggpattern.format('Q'), all_Q)
    np.save(aggpattern.format('hs'), all_hs)
    np.save(aggpattern.format('runtime'), all_runtime)


    def area_average(z, mesh, mask=0):
        z_el = np.mean(z[mesh['elements']-1], axis=1)
        if mask==0:
            z_el[z_el<0] = np.nan
        # z_el[z_el<0] = np.nan

        print('z:', z.shape)
        print('z_el:', z_el.shape)

        area = mesh['area'].copy()
        # area[np.isnan(np.sum(z_el, axis=(1,2)))] = np.nan

        zmean = np.nansum(z_el*area[:,None], axis=0)/np.nansum(area)

        print('zmean:', zmean.shape)
        return zmean
    
    ff_mean = area_average(all_ff, mesh)
    hs_mean = area_average(all_hs, mesh, mask=None)
    N_mean = area_average(all_N, mesh, mask=None)

    np.save(aggpattern.format('ff_mean'), ff_mean)
    np.save(aggpattern.format('hs_mean'), hs_mean)
    np.save(aggpattern.format('N_mean'), N_mean)

    return 

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('njobs', type=int)
    args = parser.parse_args()
    config = import_config(args.config)
    collect_issm_results(config.sim_dir, args.njobs)
