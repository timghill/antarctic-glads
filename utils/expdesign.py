"""
Generate parameter experimental designs
"""

import sys
import os
import importlib
import argparse

import numpy as np
from scipy import stats

from matplotlib import pyplot as plt
import cmocean

def log_design(m, bounds, sampler=None):
    """
    Generate log-transformed parameter design.

    Note that bounds are specified in log space. For example,
    bounds = [0, 1] generates samples in physical space between
    1 and 10.

    Parameters
    ----------
    m : int
        Number of samples to draw

    bounds : (n_para, 2) array
             Lower and upper bounds on log parameters
    
    sampler : stats.qmc.QMCEngine, optional
              QMC sampler for generating the design. If not provided,
              defaults to stats.qmc.LatinHypercube
    
    Returns
    -------
    design : dict
             design['standard'] standardized design in [0, 1] hypercube
             design['log'] log design in provided bounds
             design['physical'] physical parameter values
    """
    if sampler is None:
        n_dim = bounds.shape[0]
        sampler = stats.qmc.LatinHypercube(n_dim, 
            optimization='random-cd', scramble=False, seed=42186)
    
    X_std = sampler.random(n=m)

    # Stretch [0, 1] interval into provided log bounds
    X_log = bounds[:, 0] + (bounds[:, 1] - bounds[:, 0])*X_std

    # Convert to physical units
    X_phys = 10**X_log

    print('Generated', X_std.shape, 'design')

    print('Standardized min/max')
    print(X_std.min(axis=0), X_std.max(axis=0))

    print('Log min/max')
    print(X_log.min(axis=0), X_log.max(axis=0))

    print('Physical min/max')
    print(X_phys.min(axis=0), X_phys.max(axis=0))

    design = dict(standard=X_std, log=X_log, physical=X_phys)
    return design

def write_table(design, table_file='table.dat'):
    """Create table.dat for job array, compatible with Digital
    Research Alliance metafarm package.
    
    Parameters
    ----------
    design : dict
             Result of log_design
    
    table_file : str
                 File path to save table"""
    output_str = ''
    _line_template = '%d %d\n'

    m = design['physical'].shape[0]
    output_str = ''
    for i in range(1, m+1):
        output_str = output_str + _line_template % (i, i)
    output_str = output_str

    with open(table_file, 'w') as table_handle:
        table_handle.writelines(output_str)

def main():
    """
    Command-line interface to compute, plot and save experimental design

    python -m src.expdesign config

    where config is the path to a valid configuration file.
    """
    desc = 'Compute, plot and save experimental design'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('config', help='Path to experiment config file')
    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise OSError('Configuration file "{}" does not exist'.format(args.config))
    
    path, name = os.path.split(args.config)
    if path:
        abspath = os.path.abspath(path)
        sys.path.append(abspath)
    module, ext = os.path.splitext(name)
    config = importlib.import_module(module)

    design = log_design(config.m, config.theta_bounds,
        sampler=config.theta_sampler)

    kwargs = dict(delimiter=',', fmt='%.6e',
        header=','.join(config.theta_names), comments='')
    np.savetxt(config.X_physical, design['physical'], **kwargs)
    np.savetxt(config.X_log, design['log'], **kwargs)
    np.savetxt(config.X_standard, design['standard'], **kwargs)

    write_table(design, table_file=os.path.join(config.sim_dir, 'table.dat'))
    return

if __name__=='__main__':
    main()