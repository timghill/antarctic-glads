#!/bin/bash
#SBATCH --account=def-gflowers
#SBATCH --time=0-00:10
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --output=make_data.out

set -x

module load mpi4py
source ~/SFU-code/antarctic-glads/venv/bin/activate

mkdir -p geom
cd geom
python -u -m utils.mesh.generate_outline C-Cp
python -u -m utils.mesh.make_mesh
cd ../

mkdir -p lanl-mali
cd lanl-mali
python -u -m utils.mesh.interp_mali
