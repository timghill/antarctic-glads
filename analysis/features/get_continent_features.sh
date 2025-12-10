#!/bin/bash
#SBATCH --account=def-gflowers
#SBATCH --time=0-12:00
#SBATCH --mem=32G
#SBATCH --job-name='AISfeatures'
#SBATCH --ntasks=1
#SBATCH --output=get_continent_features.out

module load mpi4py
source ~/SFU-code/antarctic-glads/venv/bin/activate

python -u get_continent_features.py
