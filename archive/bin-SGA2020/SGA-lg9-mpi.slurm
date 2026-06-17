#!/bin/bash
#SBATCH -A desi
#SBATCH --clusters=escori
#SBATCH --qos=bigmem
#SBATCH --nodes=1
#SBATCH --time=06:00:00
#SBATCH --module=mpich-cle6
#SBATCH --image=legacysurvey/legacyhalos:v0.0.3
#SBATCH --licenses=SCRATCH,cfs
#SBATCH --job-name=SGA-customredux-coadds.log-%j
#SBATCH --mail-user=jmoustakas@siena.edu
#SBATCH --mail-type=ALL
#SBATCH --exclusive
#SBATCH --mem=750GB

# coadds
srun -n 1 -c 32 shifter $LEGACYHALOS_CODE_DIR/bin/SGA/SGA-customredux-mpi.sh coadds 32
#srun -n 1 -c 32 shifter /global/u2/i/ioannis/repos/git/legacyhalos/bin/SGA/SGA-customredux-mpi.sh coadds 32
