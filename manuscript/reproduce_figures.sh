#!/usr/bin/env bash
#
# Script to reproduce all figures from manuscript
#
# These scripts can take a few minutes to run. The heavy
# calculations (GlaDS runs, fitting RF, ISSM runs) are pre-
# computed and stored elsewhere.
#
# Last modified: 17 November 2026
# See: reproduce_figures.out

# Main text
set -x
cd ../
cd analysis

# Fig. 1 and Fig. 5
cd parameters_reduced
python plot_map_hexbin.py

# Fig. 2
cd ../parameters_full
python plot_ensemble_spread.py

# Fig. 3, Fig. 4 and Fig. D4
cd ../mean
python plot_hyperparameter_sensitivity.py

# Fig. 6
cd ../parameters_reduced
python map_delta.py

# Fig. 7
cd ../mean
python flowlines.py

# Fig. 8
cd ../parameters_reduced
python future_flowlines.py
cd ../

# Appendix figures

# Fig. A1
cd glads
python groundingline_statistics.py

# Fig. B1
cd ../../issm
python plot_lcurve.py

# Fig. C1
cd ../analysis/mean
python future_flowlines.py

# Fig. D1
python plot_heatmap.py

# Fig. D2
python plot_trees.py

# Fig. D3
python compute_shap.py
