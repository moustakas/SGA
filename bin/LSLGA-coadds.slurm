#! /bin/bash

#SBATCH -A cosmo
#SBATCH -L SCRATCH,project
#SBATCH -o coadds.log.%j
#SBATCH --mail-user=jmoustakas@siena.edu
#SBATCH --mail-type=ALL

#SBATCH -p regular
#SBATCH -N 3
#SBATCH -n 3
#SBATCH -t 03:00:00

# #SBATCH -p debug
# #SBATCH -N 32
# #SBATCH -n 48
# #SBATCH -t 00:30:00

# for cori only
#SBATCH -C haswell

# sbatch ~/repos/git/LSLGA/bin/LSLGA-coadds.slurm

# Example calculation of how many nodes I need, given a number of MPI tasks:
  # On Cori:   20 tasks: -c 4 -n 20 -N 20*4/32=2.5
  # On Edison: 20 tasks: -c 4 -n 20 -N 20*4/24=3.3

# Example calculation of scaling to the full sample on edison.  Edison limits
# the number of tasks (processes per node) to 48, so we have to increase the
# number of nodes until the total time is a reasonable number:
#
#   ngal = 1562      # [number of galaxies]
#   hourpergal = 0.5 # [hours per galaxy]
#   ntask = 48
#   nproc = 4
#   nnode = 32
#   totaltime = int(ngal*hourpergal*nproc/nnode/24)
# 
#     print('Edison: -n {} -N {:.0f} -c {} -t 00:{:02d}:00'.format(ntask, nnode, nproc, totaltime))
#       Edison: -n 48 -N 32 -c 4 -t 00:04:00
#
# Some notes: Running over 1562 galaxies with "-n 48 -N 32 -c 4 -t 00:04:00" left 917/1562 galaxies.
#
# Example calculation of scaling to the full sample on cori.  Here, we simply 
# increase the number of tasks (as multiples of 32) until the total time is a
# reasonable number:
#
#   ngal = 3         # [number of galaxies]
#   hourpergal = 3.0 # [hours per galaxy]
#   ntask = ngal # one MPI task per galaxy
#   nproc = 32
#   nnode = ntask*nproc//32
#   totaltime = int(ngal/ntask*hourpergal)
#
#     print('Cori: -n {} -N {:.0f} -c {} -t {:02d}:00:00'.format(ntask, nnode, nproc, totaltime))
#       Cori: -n 3 -N 3 -c 32 -t 03:00:00

source $HOME/repos/git/impy/bin/LSLGA-env-nersc

# do the whole sample!
time srun -N 3 -n 3 -c 32 $LSLGA_CODE_DIR/bin/LSLGA-mpi --coadds --nproc 32 --mpi --chaos --first 1 --verbose

