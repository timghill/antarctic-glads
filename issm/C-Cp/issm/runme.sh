#!/bin/bash
#SBATCH --time=0-01:00
#SBATCH --account=def-gflowers
#SBATCH --mem=4G
#SBATCH --job-name=C-Cp_inversion
#SBATCH --ntasks=1
#SBATCH --output=runme.out

source ~/SFU-code/antarctic-glads/venv/bin/activate

# python -u -m utils.issm.iceflow Lcurve C-Cp --coefficients 1 1e-3 1e-9
python -u -m utils.issm.iceflow inverse C-Cp --coefficients 1 1e-3 1e-9
python -u -m utils.issm.iceflow forward C-Cp

