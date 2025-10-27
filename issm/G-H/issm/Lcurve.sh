#!/bin/bash
#SBATCH --account=def-gflowers
#SBATCH --time=0-02:00
#SBATCH --mem=4G
#SBATCH --ntasks=1
#SBATCH --job-name=Lcurve
#SBATCH --output=Lcurveout

module load mpi4py
source ~/SFU-code/antarctic-glads/venv/bin/activate

python -u Lcurve.py
