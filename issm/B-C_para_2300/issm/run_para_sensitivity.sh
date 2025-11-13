#!/bin/bash
#SBATCH --account=def-gflowers
#SBATCH --job-name=para-sensitivity
#SBATCH --time=0-02:00
#SBATCH --mem=4G
#SBATCH --output=run_para_sensitivity.out

source ~/SFU-code/antarctic-glads/venv/bin/activate

python -u run_para_sensitivity.py
