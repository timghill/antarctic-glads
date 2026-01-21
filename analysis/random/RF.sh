#!/bin/bash
#SBATCH --time=0-8:00:00
#SBATCH --account=def-gflowers
#SBATCH --mem-per-cpu=8G
#SBATCH --ntasks=1
#SBATCH --job-name=RF-mean
#SBATCH --output=RF.out

module load mpi4py
source ~/SFU-code/antarctic-glads/venv/bin/activate

python -u RF.py
