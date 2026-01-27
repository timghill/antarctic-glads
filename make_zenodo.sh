#!/bin/bash

LIVE=$(pwd)
ARCH=$(pwd)/../antarctic-glads-zenodo/
REPO="git@github.com:timghill/antarctic-glads.git"

echo $LIVE
echo $ARCH

# Clean the archive directory
rm -rf $ARCH
mkdir $ARCH

# Clone the repo
cd $ARCH
git clone $REPO .
rm -rf .git

# Copy GlaDS + ISSM outputs (~10 GB)
# cp -rv $LIVE/issm $ARCH/issm
rsync -avI --exclude=slurm-*.out --exclude='status.*' $LIVE/issm $ARCH


# Copy analysis features (~0.25 GB)
cp -rv $LIVE/analysis/features $ARCH/analysis/features/

# Copy analysis output files (~13 GB)
cp -rv $LIVE/analysis/mean/*.pkl $ARCH/analysis/mean/
cp -rv $LIVE/analysis/mean/*.npy $ARCH/analysis/mean/
cp -rv $LIVE/analysis/mean/data $ARCH/analysis/mean/

cp -rv $LIVE/analysis/parameters_reduced/*.pkl $ARCH/analysis/parameters_reduced/
cp -rv $LIVE/analysis/parameters_reduced/data $ARCH/analysis/parameters_reduced/

cp -rv $LIVE/analysis/parameters_full/*.pkl $ARCH/analysis/parameters_full/
cp -rv $LIVE/analysis/parameters_full/data $ARCH/analysis/parameters_full/

# Tar everything
tar -cvzf ../antarctic-glads.tar.gz *
