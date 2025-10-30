#!/bin/bash
#SBATCH --time=0-01:00
#SBATCH --account=def-gflowers
#SBATCH --mem=4G
#SBATCH --job-name=G-H_inversion
#SBATCH --ntasks=1
#SBATCH --output=runme.out

source ~/SFU-code/antarctic-glads/venv/bin/activate

# python -u -m utils.issm.iceflow Lcurve G-H --coefficients 1 1e-2 1e-8
python -u -m utils.issm.iceflow inverse G-H --coefficients 1 1e-2 1e-8
python -u -m utils.issm.iceflow forward G-H

