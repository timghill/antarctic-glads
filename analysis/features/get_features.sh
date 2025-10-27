#!/bin/bash
#SBATCH --account=def-gflowers
#SBATCH --time=0-01:00
#SBATCH --mem=16G
#SBATCH --job-name='features'
#SBATCH --ntasks=1
#SBATCH --output=get_features.out

source ~/SFU-code/antarctic-glads/venv/bin/activate

python -u get_features.py
