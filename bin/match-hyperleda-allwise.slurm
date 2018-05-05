#! /bin/bash

#SBATCH -p debug
#SBATCH -N 1
#SBATCH -L SCRATCH,project
#SBATCH -C haswell
#SBATCH -t 00:30:00
#SBATCH -o match-external-catalog.slurm.%j
#SBATCH --profile=all

#source $HOME/software/develop-legacysurveys-code

set -x
export ATP_ENABLED=0

# faster python start-up
module load python/3.5
source /usr/common/contrib/bccp/python-mpi-bcast/nersc/activate.sh

export LSLGA_DIR=$SCRATCH/LSLGA
export LSLGA_CODE_DIR=$SCRATCH/repos/LSLGA

export ALLWISE_INDIR=/global/project/projectdirs/cosmo/data/wise/allwise-catalog
export ALLWISE_FILELIST=$SCRATCH/allwise_filelist
time find $ALLWISE_INDIR -name 'wise-allwise-cat-part??.fits' > $ALLWISE_FILELIST

time srun -u --cpu_bind=no -N 1 python $LSLGA_CODE_DIR/bin/match-hyperleda-allwise -v --numproc 24 \
     -f fits -F $ALLWISE_FILELIST $LSLGA_DIR/sample/hyperleda-d25min10.fits \
     $ALLWISE_INDIR $LSLGA_DIR/sample/hyperleda-d25min10-allwise.fits
