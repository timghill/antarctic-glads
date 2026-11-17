# Antarctic GlaDS emulation

Tim Hill, 2026 (tim_hill_2@sfu.ca) | https://github.com/timghill/antarctic-glads

## Description

The project structure is:

* `utils/`: shared code for setting up experiments and analyzing outputs (see `pyproject.toml`)
* `issm/`: individual directories for GlaDS-ISSM model runs, including stored GlaDS ensembles and ISSM solutions
* `analysis/`: emulator fitting, evaluation, and all analysis
    * `analysis/glads/`: analysis of the GlaDS ensemble
    * `analysis/mean/`: analysis for the mean of the perturbed-parameter ensemble
    * `analysis/parameters_full/`: RF including all features + parameters
    * `analysis/parameters_reduced/`: simplified RF including 4 features + parameters
* `manuscript/`: final manuscript figures
* `data/`: raw data (BedMachine, ice velocities, and other published datasets not reproduced here. Please download these from the appropriate sources)
* `examples/`: notebook showing how to read the trained random forest and make predictions

Each directory has a README file to describe the contents.

## Installation

The analysis source code has been tested against python 3.11.5. Package requirements are listed in `requirements.txt`, and it is recommended to use a virtual environment to manage versions. For example

```
virtualenv --python 3.11 pyenv/
source pyenv/bin/activate
pip install -r requirements.txt
```

To install the code for this project on your python path, install in editable (`-e`) mode with pip from the project root directory:

```
pip install -e .
```

## Reproducing analysis

All manuscript figures can be reproduced from the script `manuscript/reproduce_figures.sh`. Note that you will need to download source data that are not distributed as part of this repository (BedMachine, etc.) for all figures to run.

Main manuscript figures are made from the following scripts in `analysis/`:

 1. `parameters_reduced/plot_map_hexbin.py`
 2. `parameters_full/plot_ensemble_spread.py`
 3. `mean/plot_hyperparameter_sensitivity.py`
 4. `mean/plot_hyperparameter_sensitivity.py`
 5. `parameters_reduced/plot_map_hexbin.py`
 6. `parameters_reduced/map_delta.py`
 7. `mean/flowlines.py`
 8. `parameters_reduced/future_flowlines.py`

Appendix figures (paths relative to `.`):

- A1: `analysis/glads/groundingline_statistics.py`
- B1: `issm/plot_lcurve.py`
- C1: `analysis/mean/future_flowlines.py`
- D1: `analysis/mean/plot_heatmap.py`
- D2: `analysis/mean/plot_trees.py`
- D3: `analysis/mean/compute_shap.py`
- D4: `analysis/mean/plot_hyperparameter_sensitivity.py`

