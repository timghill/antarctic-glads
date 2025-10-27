#!/bin/bash
#SBATCH --time=0-04:00
#SBATCH --account=def-gflowers
#SBATCH --mem=4G
#SBATCH --job-name=Cp-D_inversion
#SBATCH --ntasks=1
#SBATCH --output=runme.out

source ~/SFU-code/antarctic-glads/venv/bin/activate

python -u -m utils.issm.iceflow Lcurve Cp-D --coefficients 1 1e-4 1e-5
# python -u -m utils.issm.iceflow inverse Cp-D --coefficients 1 1e-4 1e-5
# python -u -m utils.issm.iceflow forward Cp-D

