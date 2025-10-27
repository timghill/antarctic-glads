#!/bin/bash
#SBATCH --account=def-gflowers
#SBATCH --time=0-00:60
#SBATCH --mem-per-cpu=8G
#SBATCH --ntasks=1
#SBATCH --output=inversions.out

module load mpi4py
source ~/SFU-code/antarctic-glads/venv/bin/activate

python -u inversions.py
