#! /bin/bash

# Shell script for running the various stages of the legacyhalos code using
# MPI+shifter at NERSC. Required arguments:
#   {1} stage [coadds, pipeline-coadds, ellipse, htmlplots]
#   {2} ncores [should match the resources requested.]

# Example: build the coadds using 16 MPI tasks with 8 cores per node (and therefore 16*8/32=4 nodes)

#salloc -N 8 -C haswell -A desi -L cfs,SCRATCH -t 08:00:00 --qos realtime --image=legacysurvey/legacyhalos:v0.0.2 --exclusive
#srun -n 8 -c 32 shifter --module=mpich-cle6 $LEGACYHALOS_CODE_DIR/bin/SGA/SGA-lg9-mpi.sh coadds 8 > coadds-lg9.log.1 2>&1 &

# use the bigmem node--
#salloc -N 1 -C haswell -A desi -L cfs,SCRATCH -t 06:00:00 --qos bigmem --image=legacysurvey/legacyhalos:v0.0.3 --exclusive
#srun -n 1 -c 32 shifter --module=mpich-cle6 $LEGACYHALOS_CODE_DIR/bin/SGA/SGA-lg9-mpi.sh coadds 32 > coadds-lg9.log.1 2>&1 &

# Grab the input arguments--
stage=$1
ncores=$2

source $LEGACYHALOS_CODE_DIR/bin/SGA/SGA-lg9-env
#source /global/u2/i/ioannis/repos/git/legacyhalos/bin/SGA/SGA-lg9-env

if [ $stage = "test" ]; then
    time python $LEGACYHALOS_CODE_DIR/bin/SGA/SGA-mpi --help
elif [ $stage = "coadds" ]; then
    time python $LEGACYHALOS_CODE_DIR/bin/SGA/SGA-mpi --coadds --nproc $ncores --mpi --upercal-sky --verbose # --just-coadds
elif [ $stage = "pipeline-coadds" ]; then
    time python $LEGACYHALOS_CODE_DIR/bin/SGA/SGA-mpi --pipeline-coadds --nproc $ncores --mpi --verbose
elif [ $stage = "ellipse" ]; then
    time python $LEGACYHALOS_CODE_DIR/bin/SGA/SGA-mpi --ellipse --nproc $ncores --mpi --verbose
elif [ $stage = "htmlplots" ]; then
    time python $LEGACYHALOS_CODE_DIR/bin/SGA/SGA-mpi --htmlplots --nproc $ncores --mpi --verbose
else
    echo "Unrecognized stage "$stage
fi
