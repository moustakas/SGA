#! /bin/bash

#SBATCH -A cosmo
#SBATCH -L SCRATCH,project
#SBATCH -o match-hyperleda-allwise.slurm.%j
#SBATCH -p debug
#SBATCH -N 1
#SBATCH -C haswell
#SBATCH -t 00:30:00

# Anything more than 2 cores will likely run out of memory.

# To run the script from an interactive node do:
# salloc -N 1 -C haswell -t 00:60:00 --qos interactive -L SCRATCH,project

source $HOME/repos/git/impy/bin/LSLGA-env-nersc

export ALLWISE_INDIR=/global/project/projectdirs/cosmo/data/wise/allwise-catalog
export ALLWISE_FILELIST=$SCRATCH/tmp/allwise_filelist
time find $ALLWISE_INDIR -name 'wise-allwise-cat-part??.fits' > $ALLWISE_FILELIST

time srun -u --cpu_bind=no python $LSLGA_CODE_DIR/bin/match-hyperleda-allwise -v --numproc 8 \
    -f fits -F $ALLWISE_FILELIST $LSLGA_DIR/sample/hyperleda-d25min10-18nov14.fits \
    $ALLWISE_INDIR $LSLGA_DIR/sample/hyperleda-d25min10-18nov14-allwise.fits 
