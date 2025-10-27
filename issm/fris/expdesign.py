"""
Generate parameter experimental designs
"""

import numpy as np
from scipy import stats


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
            optimization='random-cd', scramble=False, seed=632025)
    
    X_std = sampler.random(n=m)

    # Stretch [0, 1] interval into provided log bounds
    X_log = bounds[:, 0] + (bounds[:, 1] - bounds[:, 0])*X_std

    # Convert to physical units
    X_phys = 10**X_log

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
    flux_levels = np.arange(3)
    cavity_levels = np.arange(2)

    m = 16

    n_thetas = m*len(flux_levels)*len(cavity_levels)
    n_dim = 2 + 3
    thetas = np.zeros((n_thetas, n_dim), dtype=np.float32)
    design = dict(
        physical=thetas.copy(),
        standard=thetas.copy(),
    )

    startindex = 0
    endindex = m
    for i in flux_levels:
        for j in cavity_levels:
            if i<2:
                logks_bounds = np.array([-3, -1])
            else:
                logks_bounds = np.array([-4, -2])
            
            theta_bounds = np.array([
                logks_bounds,
                [-2, -1],
                [np.log10(5), np.log10(50)],
            ])

            new_thetas = log_design(m,  bounds=theta_bounds)
            for key in ('physical', 'standard'):
                design[key][startindex:endindex, 0 ] = i
                design[key][startindex:endindex, 1 ] = j
                design[key][startindex:endindex, 2:] = new_thetas[key]

            startindex = endindex
            endindex = startindex+m

    theta_names = ['Sheet-flux model', 'Cavity creep-open', 'k_s', 'k_c', 'r_b']
    kwargs = dict(delimiter=',', fmt='%.4e',
        header=','.join(theta_names), comments='')
    np.savetxt('theta_physical.csv', design['physical'], **kwargs)
    np.savetxt('theta_standard.csv', design['standard'], **kwargs)

    write_table(design, table_file='table.dat')
    return

if __name__=='__main__':
    main()