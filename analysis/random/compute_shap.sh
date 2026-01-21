#!/bin/bash
#SBATCH --time=0-2:00:00
#SBATCH --account=def-gflowers
#SBATCH --mem-per-cpu=8G
#SBATCH --ntasks=1
#SBATCH --job-name=compute_shap
#SBATCH --output=compute_shap.out

module load mpi4py
source ~/SFU-code/antarctic-glads/venv/bin/activate

rm data/SHAP_*.npy
python -u compute_shap.py
