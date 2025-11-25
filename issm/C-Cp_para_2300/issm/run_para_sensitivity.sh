#!/bin/bash
#SBATCH --account=def-gflowers
#SBATCH --job-name=C-Cp_para
#SBATCH --time=0-02:00
#SBATCH --mem=8G
#SBATCH --output=run_para_sensitivity.out

source ~/SFU-code/antarctic-glads/venv/bin/activate

python -u -m utils.issm.run_para_sensitivity C-Cp 2050

python runme.py
