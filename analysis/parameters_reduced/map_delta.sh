#!/bin/bash
#SBATCH --time=0-02:00
#SBATCH --account=def-gflowers
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=8G
#SBATCH --job-name=map_delta

module load mpi4py
source ~/SFU-code/antarctic-glads/venv/bin/activate

python -u map_delta.py
