#!/bin/bash
#SBATCH --time=0-08:00
#SBATCH --account=def-gflowers
#SBATCH --mem=8G
#SBATCH --job-name=Jpp-K_inversion
#SBATCH --ntasks=1
#SBATCH --output=runme.out

source ~/SFU-code/antarctic-glads/venv/bin/activate

python -u -m utils.issm.iceflow Lcurve Jpp-K --coefficients 1 1e-3 1e-9
python -u -m utils.issm.iceflow inverse Jpp-K --coefficients 1 1e-3 1e-9
python -u -m utils.issm.iceflow forward Jpp-K

