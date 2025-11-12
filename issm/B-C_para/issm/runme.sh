#!/bin/bash
#SBATCH --time=0-02:00
#SBATCH --account=def-gflowers
#SBATCH --mem=4G
#SBATCH --job-name=B-C_inversion
#SBATCH --ntasks=1
#SBATCH --output=runme.out

source ~/SFU-code/antarctic-glads/venv/bin/activate

# python -u -m utils.issm.iceflow_para Lcurve B-C --coefficients 1 1e-3 1e-9
python -u -m utils.issm.iceflow_para inverse B-C --coefficients 1 1e-3 1e-8
python -u -m utils.issm.iceflow_para forward B-C

