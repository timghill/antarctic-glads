#!/bin/bash
#SBATCH --time=0-12:00
#SBATCH --account=def-gflowers
#SBATCH --mem=1G
#SBATCH --ntasks=1
#SBATCH --job-name=test_F-G

source ~/SFU-code/antarctic-glads/venv/bin/activate

python -u run_job.py ../train_config.py 1
