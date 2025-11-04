#!/bin/bash
#SBATCH --account=def-gflowers
#SBATCH --time=0-02:00
#SBATCH --mem=24G
#SBATCH --job-name=aggregate
#SBATCH --output=aggregate_outputs.out
#SBATCH --mail-user=tha111@sfu.ca
#SBATCH --mail-type=FAIL,END

source /home/tghill/SFU-code/antarctic-glads/venv/bin/activate

python -u -m utils.aggregate_outputs ../train_config.py 100
