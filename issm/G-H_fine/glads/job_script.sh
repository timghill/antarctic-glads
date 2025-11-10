#!/bin/bash
#SBATCH --job-name="G-H_fine"
#SBATCH --time=04-0:00
#SBATCH --mem=4G
#SBATCH --account=def-gflowers
#SBATCH --mail-user=tha111@sfu.ca
#SBATCH --mail-type=FAIL,END,BEGIN

# source ../../setenv.sh

source ~/SFU-code/antarctic-glads/venv/bin/activate

task.run
