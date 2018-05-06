#! /bin/bash

#SBATCH -p debug
#SBATCH -N 1
#SBATCH -L SCRATCH,project
#SBATCH -C haswell
#SBATCH -t 00:30:00
#SBATCH -o match-hyperleda-allwise.slurm.%j
#SBATCH --profile=all

# Anything more than 2 cores will likely run out of memory.

# To run the script from an interactive node do:
# salloc -N 1 -C haswell -t 00:60:00 --qos interactive -L SCRATCH,project

set -x
export ATP_ENABLED=0

desiconda_version=20170818-1.1.12-img
module use /global/common/${NERSC_HOST}/contrib/desi/desiconda/$desiconda_version/modulefiles
module load desiconda

export LSLGA_DIR=$SCRATCH/LSLGA
export LSLGA_CODE_DIR=$SCRATCH/repos/LSLGA
export LEGACYPIPE_DIR=$SCRATCH/repos/legacypipe
export PYTHONPATH=$LEGACYPIPE_DIR/py:${PYTHONPATH}

export ALLWISE_INDIR=/global/project/projectdirs/cosmo/data/wise/allwise-catalog
export ALLWISE_FILELIST=$SCRATCH/tmp/allwise_filelist
time find $ALLWISE_INDIR -name 'wise-allwise-cat-part??.fits' > $ALLWISE_FILELIST

time srun -u --cpu_bind=no python $LSLGA_CODE_DIR/bin/match-hyperleda-allwise -v --numproc 2 \
    -f fits -F allwise_filelist $LSLGA_DIR/sample/hyperleda-d25min10.fits \
    $ALLWISE_INDIR $LSLGA_DIR/sample/hyperleda-d25min10-allwise.fits 
