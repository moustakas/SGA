#! /bin/bash
#SBATCH -A desi
#SBATCH -C cpu
#SBATCH -o /pscratch/sd/i/ioannis/SGA2024/logs/2024cutouts.log.%j
#SBATCH --mail-user=jmoustakas@siena.edu
#SBATCH --image=dstndstn/viewer-cutouts
#SBATCH --mail-type=ALL
#SBATCH -q regular
#SBATCH -N 2
#SBATCH -n 2
#SBATCH -t 00:10:00

# sbatch /global/homes/i/ioannis/code/git/SGA/bin/SGA2024/get-2024cutouts.slurm

time srun -n 2 -c 128 /global/homes/i/ioannis/code/git/SGA/bin/SGA2024/get-2024cutouts.sh sga2020 128
#time srun -n 2 -c 128 /global/homes/i/ioannis/code/git/SGA/bin/SGA2024/get-2024cutouts.sh sga2020-offset 128
#time srun -n 2 -c 128 /global/homes/i/ioannis/code/git/SGA/bin/SGA2024/get-2024cutouts.sh sga2024-missing 128
#time srun -n 2 -c 128 /global/homes/i/ioannis/code/git/SGA/bin/SGA2024/get-2024cutouts.sh sga2024-candidates 128
#time srun -n 2 -c 128 /global/homes/i/ioannis/code/git/SGA/bin/SGA2024/get-2024cutouts.sh skies 128
