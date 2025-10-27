#!/bin/bash
#SBATCH --job-name="gr-train"
#SBATCH --time=03-00:00
#SBATCH --mem=1G
#SBATCH --account=def-gflowers
#SBATCH --mail-user=tha111@sfu.ca
#SBATCH --mail-type=FAIL,END,BEGIN

# source ../../setenv.sh

source ~/SFU-code/antarctic-glads/venv/bin/activate

task.run
