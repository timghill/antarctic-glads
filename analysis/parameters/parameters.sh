#!/bin/bash
#SBATCH --time=0-4:00:00
#SBATCH --account=def-gflowers
#SBATCH --mem-per-cpu=16G
#SBATCH --ntasks=1
#SBATCH --job-name=parameters
#SBATCH --output=parameters.out

module load mpi4py
source ~/SFU-code/antarctic-glads/venv/bin/activate

python -u parameters.py
