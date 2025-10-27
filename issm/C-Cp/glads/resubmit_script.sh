#!/bin/bash
#SBATCH --job-name="TrainResubmit"
#SBATCH --time=00-16:00:00
#SBATCH --mem=1G
#SBATCH --account=def-gflowers
#SBATCH --mail-user=tha111@sfu.ca
#SBATCH --mail-type=FAIL,END



# Don't change this line:
autojob.run
