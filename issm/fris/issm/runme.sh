#!/bin/bash
#SBATCH --time=0-00:60
#SBATCH --account=def-gflowers
#SBATCH --mem=8G
#SBATCH --job-name=inversion
#SBATCH --ntasks=1
#SBATCH --output=runme.out

source ~/SFU-code/antarctic-glads/venv/bin/activate

python -u -m utils.issm.run_friction_inversion

python plot.py
