#!/bin/bash
#SBATCH --time=0-2:00:00
#SBATCH --account=def-gflowers
#SBATCH --mem-per-cpu=8G
#SBATCH --ntasks=1
#SBATCH --job-name=write-nc
#SBATCH --output=write.out

module load mpi4py
source ~/SFU-code/antarctic-glads/venv/bin/activate

python -u RF.py
