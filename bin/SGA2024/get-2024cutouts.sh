#! /bin/bash

#salloc -N 2 -C cpu -A desi -t 01:00:00 --qos interactive --image=dstndstn/viewer-cutouts 
#srun -n 2 -c 128 shifter $HOME/code/git/SGA/bin/SGA2024/get-2024cutouts.sh sga2020 128 > $SCRATCH/SGA2024/logs/cutouts-sga2020.log 2>&1 &
#srun -n 2 -c 128 shifter $HOME/code/git/SGA/bin/SGA2024/get-2024cutouts.sh sga2020-offset 128 > $SCRATCH/SGA2024/logs/cutouts-offset.log 2>&1 &
#srun -n 2 -c 128 shifter $HOME/code/git/SGA/bin/SGA2024/get-2024cutouts.sh sga2020-missing 128 > $SCRATCH/SGA2024/logs/cutouts-missing.log 2>&1 &
#srun -n 2 -c 128 shifter $HOME/code/git/SGA/bin/SGA2024/get-2024cutouts.sh sga2020-candidates 128 > $SCRATCH/SGA2024/logs/cutouts-candidates.log 2>&1 &
#srun -n 2 -c 128 shifter $HOME/code/git/SGA/bin/SGA2024/get-2024cutouts.sh skies 128 > $SCRATCH/SGA2024/logs/cutouts-skies.log 2>&1 &

codedir=$HOME/code/git
mpiscript=$codedir/SGA/bin/SGA2024/get-2024cutouts
outdir_data=$SCRATCH/SGA2024

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export KMP_AFFINITY=disabled
export MPICH_GNI_FORK_MODE=FULLCOPY

catalog_type=$1 # catalog type (sga2020, sga2020-offset, sga2020-missing, sga2024-candidates, skies)
mp=$2           # number of tasks (cores) per MPI rank

args="--outdir-data $outdir_data "

if [[ $catalog_type != " " ]] && [[ $catalog_type != "" ]] && [[ $catalog_type != "-" ]]; then
    args=$args" --catalog-type $catalog_type"
fi
if [[ $mp != " " ]] && [[ $mp != "" ]] && [[ $mp != "-" ]]; then
    args=$args" --mp $mp"
fi

echo $mpiscript $args
time $mpiscript $args
