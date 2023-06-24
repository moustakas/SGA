#! /bin/bash

#salloc -N 2 -C cpu -A desi -t 01:00:00 --qos interactive --image=dstndstn/viewer-cutouts 
#srun -n 2 -c 128 shifter /global/homes/i/ioannis/code/git/SGA/bin/SGA2024/get-2024cutouts.sh sga2020 128 > /pscratch/sd/i/ioannis/SGA2024/cutouts-01.log 2>&1 &

codedir=/global/homes/i/ioannis/code/git
mpiscript=$codedir/SGA/bin/SGA2024/get-2024cutouts
outdir_data=/pscratch/sd/i/ioannis/SGA2024

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export KMP_AFFINITY=disabled
export MPICH_GNI_FORK_MODE=FULLCOPY

catalog_type=$1 # catalog type (sga2020, sga2020-offset, sga2024-candidates, skies)
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
