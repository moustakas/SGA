#! /bin/bash

#SBATCH -A cosmo
#SBATCH -p debug
#SBATCH -N 1
#SBATCH -n 32
#SBATCH -C haswell
#SBATCH -L SCRATCH,project
#SBATCH -t 00:30:00
#SBATCH -o build-drsample-v1.0.log.%j
#SBATCH --mail-user=jmoustakas@siena.edu
#SBATCH --mail-type=ALL

# sbatch ~/scratch/repos/LSLGA/bin/LSLGA-build-drsample.slurm

export IMPY_DIR=$HOME/repos/git/impy
source $IMPY_DIR/bin/LSLGA-env-nersc

time python $LSLGA_CODE_DIR/bin/LSLGA-build-drsample --verbose --clobber --nproc 32
