#!/bin/bash
#SBATCH --account=def-gflowers
#SBATCH --time=00-00:10:00
#SBATCH --mem=8G
#SBATCH --ntasks=1

source ~/SFU-code/antarctic-glads/venv/bin/activate

python -u -m utils.plots ../train_config.py

