#!/bin/bash
#SBATCH --account=def-gflowers
#SBATCH --time=0-01:00
#SBATCH --mem=8G
#SBATCH --output=batch_plot.out
#SBATCH --job-name=batch-plot

source ~/SFU-code/antarctic-glads/venv/bin/activate

python -u ../batch_plot.py sensitivity
