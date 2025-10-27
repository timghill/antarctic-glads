#!/bin/bash
#SBATCH --account=def-gflowers
#SBATCH --time=0-06:00
#SBATCH --mem=4G
#SBATCH --ntasks=1
#SBATCH --output=assess_steady.out

source ~/SFU-code/antarctic-glads/venv/bin/activate

python  -u assess_steady.py
