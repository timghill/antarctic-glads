#!/bin/bash
#SBATCH --time=0-01:00
#SBATCH --account=def-gflowers
#SBATCH --mem=8G
#SBATCH --job-name=fris-inverse
#SBATCH --ntasks=1
#SBATCH --output=runme.out

source ~/SFU-code/antarctic-glads/venv/bin/activate

python -u -m run_glads_inverse
