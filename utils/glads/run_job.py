"""
Run ISSM-GlaDS simultions using experiment config files from command-line

Usage
-----

    python -m src.run_job [config] [id]

config : path to experiment configuration file
id : integer job number
"""

import os
import sys
import argparse
import time

import numpy as np
import pickle

ISSM_DIR = os.getenv('ISSM_DIR')
sys.path.append(os.path.join(ISSM_DIR, 'bin/'))
sys.path.append(os.path.join(ISSM_DIR, 'lib/'))
from issmversion import issmversion

from model import model
from meshconvert import meshconvert
from solve import solve
from setmask import setmask
from parameterize import parameterize

from utils.tools import import_config

def _defaults(config, jobid):
    # Initialize model, set mesh, and set default parameters
    md = model()
    
    # Read in mesh and pass to ISSM
    mesh = np.load(config.mesh, allow_pickle=True)
    md = meshconvert(md, mesh['elements'], mesh['x'], mesh['y'])

    # Parameterize
    md = parameterize(md, '../../issm_parameterize.py')
    md = config.parser(md, jobid)
    md.initialization.ve = np.load('../data/lanl-mali/basal_velocity_mali.npy')

    # GlaDS solver options
    md.stressbalance.restol = 5e-5
    md.stressbalance.reltol = np.nan
    md.stressbalance.abstol = np.nan
    md.stressbalance.maxiter = 25
    md.transient = md.transient.deactivateall()
    md.transient.ishydrology = 1
    md.timestepping.start_time = 0
    md.timestepping.final_time = 5

    md = config.parser(md, jobid)

    # Configure for specific jobid
    md.miscellaneous.name = 'glads_{:03d}'.format(jobid)

    # Make results directory if necessary and save md.hydrology class
    resdir = os.path.join(config.sim_dir, 'RUN/output_{:03d}'.format(jobid))
    if not os.path.exists(resdir):
        os.makedirs(resdir)
    
    # Save hydro instance
    hydro = md.hydrology
    with open(os.path.join(resdir, 'md.hydrology.pkl'), 'wb') as modelinfo:
        pickle.dump(hydro, modelinfo)

    return md,resdir

def _extract_requested_outputs(md):
    """
    Extract arrays of model output fields from ISSM outputs

    Construct a dictionary where the values are ( - , n_timesteps)
    arrays by iterating over the md.results.TransitionSolution
    struct array.

    Parameters
    ----------
    md : model
         Solved ISSM model instance
    
    Returns
    -------
    dict of model output fields
    """
    imin = None
    imax = None
    phi_bed = np.vstack(md.materials.rho_freshwater*md.constants.g*md.geometry.bed)
    phi = np.array([ts.HydraulicPotential[:, 0] for ts in md.results.TransientSolution[imin:imax]]).T
    N = np.array([ts.EffectivePressure[:, 0] for ts in md.results.TransientSolution[imin:imax]]).T
    pw = phi - phi_bed
    ff = pw/(N + pw)
    outputs = dict(
        phi = phi,
        N = N,
        ff = ff,
        h_s = np.array([ts.HydrologySheetThickness[:, 0] for ts in md.results.TransientSolution[imin:imax]]).T,
        S = np.array([ts.ChannelArea[:, 0] for ts in md.results.TransientSolution[imin:imax]]).T,
        Q = np.array([ts.ChannelDischarge[:, 0] for ts in md.results.TransientSolution[imin:imax]]).T,
        time = np.array([ts.time for ts in md.results.TransientSolution[imin:imax]]).T,
        # vx = np.array([ts.HydrologyWaterVx[:, 0] for ts in md.results.TransientSolution[imin:imax]]).T,
        # vy = np.array([ts.HydrologyWaterVy[:, 0] for ts in md.results.TransientSolution[imin:imax]]).T,
    )
    return outputs


def _savemodel(md, resdir):
    requested_outputs = _extract_requested_outputs(md)
    for field in requested_outputs.keys():
        resfile = os.path.join(resdir, f'{field}.npy')
        if os.path.exists(resfile):
            Xold = np.load(resfile)
            if field=='time':
                Xnew = np.hstack((Xold, requested_outputs[field][1:]))
            else:
                Xnew = np.hstack((Xold, requested_outputs[field][:, 1:]))
            np.save(resfile, Xnew)
        else:
            np.save(resfile, requested_outputs[field])
    return md

def run_init(config, jobid):
    """Execute a single ISSM-GlaDS simulation.
    
    Default ISSM parameters are set by the experiment-specific
    defaults file and job-specific parameters are set 
    by the config files.

    Results are saved in directory
        {exp}/RUN/output_XXX/
    The md.hydrology portion of the model class is pickled as
    md.hydrology.pkl and individual fields are saved in .npy format.
    
    Returns
    -------
    md : model
         Solved model
    """
    md,resdir = _defaults(config, 'transient', jobid)

    # Shorter timesteps
    md.timestepping.time_step = 1e-4
    md.timestepping.final_time = 2
    md.settings.output_frequency = 1e4

    # Solve and save output fields to numpy binary files
    t0 = time.perf_counter()
    md = solve(md, 'Transient')
    dt = time.perf_counter() - t0
    np.savetxt(os.path.join(resdir, 'runtime'), [dt], fmt='%.3f')
    md = _savemodel(md, resdir)
    return md


def run_steady(config, jobid):
    """Execute a single ISSM-GlaDS simulation.
    
    Default ISSM parameters are set by the experiment-specific
    defaults file and job-specific parameters are set 
    by the config files.

    Results are saved in directory
        {exp}/RUN/output_XXX/
    The md.hydrology portion of the model class is pickled as
    md.hydrology.pkl and individual fields are saved in .npy format.
    
    Parameters
    ----------
    config  : str
              Experiment configuration file
    
    jobid : int
            Row number in the parameterfile. Note this ID is
            one-indexed, i.e. this should take values [1, n_jobs]
            inclusive
    
    Returns
    -------
    md : model
         Solved model
    """
    md,resdir = _defaults(config, 'steady', jobid)

    # Pick up from the transient file
    read_dir = os.path.join(config.sim_dir, 
        'RUN/output_{:03d}/transient'.format(jobid))
    h_s = np.load(os.path.join(read_dir, 'h_s.npy'))[:, -1:]
    S = np.load(os.path.join(read_dir, 'S.npy'))[:, -1:]
    phi = np.load(os.path.join(read_dir, 'phi.npy'))[:, -1:]

    md.initialization.watercolumn = h_s
    md.initialization.channelarea = S
    md.initialization.hydraulic_potential = phi

    # Longer time steps
    md.timestepping.time_step = 1e-4
    md.timestepping.final_time = 8
    md.settings.output_frequency = 1e4

    print(md.materials)

    # Solve and save output fields to numpy binary files
    t0 = time.perf_counter()
    md = solve(md, 'Transient')
    dt = time.perf_counter() - t0
    np.savetxt(os.path.join(resdir, 'runtime'), [dt], fmt='%.3f')
    md = _savemodel(md,resdir)
    return md



def run_epoch(config, jobid):
    """Execute a single ISSM-GlaDS simulation.
    
    Default ISSM parameters are set by the experiment-specific
    defaults file and job-specific parameters are set 
    by the config files.

    Results are saved in directory
        {exp}/RUN/output_XXX/
    The md.hydrology portion of the model class is pickled as
    md.hydrology.pkl and individual fields are saved in .npy format.
    
    Parameters
    ----------
    config  : str
              Experiment configuration file
    
    jobid : int
            Row number in the parameterfile. Note this ID is
            one-indexed, i.e. this should take values [1, n_jobs]
            inclusive
    
    Returns
    -------
    md : model
         Solved model
    """
    md,resdir = _defaults(config, jobid)
    is_converged = False

    read_dir = os.path.join(config.sim_dir, f'RUN/output_{jobid:03d}')
    if os.path.exists(os.path.join(read_dir, 'phi.npy')):
        h_s = np.load(os.path.join(read_dir, 'h_s.npy'))
        S = np.load(os.path.join(read_dir, 'S.npy'))
        phi = np.load(os.path.join(read_dir, 'phi.npy'))[:, -1:]
        t0 = np.load(os.path.join(read_dir, 'time.npy'))[-1:]

        md.initialization.watercolumn = h_s[:, -1]
        md.initialization.channelarea = S[:, -1]
        md.initialization.hydraulic_potential = phi
        md.timestepping.start_time = np.round(t0, 1)
        md.timestepping.final_time = np.round(t0 + 5, 1)

        # Convergence check!
        dhdt = h_s[:, -1] - h_s[:, -2]
        dSdt = S[:, -1] - S[:, -2]

        dh_threshold = 0.05*md.hydrology.bump_height[0]
        dh_quantile = np.quantile(np.abs(dhdt), 0.95)
        dh_converged = dh_quantile <= dh_threshold
        
        dS_rel_threshold = 0.05
        S[S<1] = 1
        dS_quantile = np.quantile(np.abs(dSdt/S[:, -1]), 0.95)
        dS_converged = dS_quantile < dS_rel_threshold
        is_converged = np.logical_and(dh_converged, dS_converged)

        print('CONVERGENCE CHECK')
        print(is_converged)
        print('dSdt quantile:', dS_quantile)
        print('dhdt quantile:', dh_quantile)
        is_converged = False


    md.timestepping.time_step = 1e-4
    md.settings.output_frequency = 1e4

    if not is_converged:
        # Solve and save output fields to numpy binary files
        t0 = time.perf_counter()
        md = solve(md, 'Transient')
        dt = time.perf_counter() - t0
        np.savetxt(os.path.join(resdir, 'runtime'), [dt], fmt='%.3f')
        md = _savemodel(md,resdir)
    else:
        print('Model already converged!')
    return md

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('jobid', type=int)
    args = parser.parse_args()
    config = import_config(args.config)
    md = run_epoch(config, args.jobid)
    # md = run_init(config, args.jobid)
    # md = run_steady(config, args.jobid)
    return

if __name__=='__main__':
    main()
