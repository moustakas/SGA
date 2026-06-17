#! /bin/bash
#SBATCH -A desi
#SBATCH --module=mpich-cle6
#SBATCH -C haswell
#SBATCH -L cfs,SCRATCH
#SBATCH -o SGA-coadds.log-%j
#SBATCH --image=legacysurvey/legacyhalos:v0.0.2
#SBATCH --mail-user=jmoustakas@siena.edu
#SBATCH --mail-type=ALL
#SBATCH -p regular
# #SBATCH -p debug
#SBATCH -N 64
#SBATCH -n 256
#SBATCH -t 04:00:00

# sbatch SGA-mpi.slurm

# coadds
time srun -N 64 -n 256 -c 8 shifter $LEGACYHALOS_CODE_DIR/bin/SGA/SGA-mpi.sh coadds 8

# test
#time srun -N 2 -n 64 -c 1 shifter $LEGACYHALOS_CODE_DIR/bin/SGA/SGA-mpi.sh test 1
