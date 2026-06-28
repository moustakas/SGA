#!/bin/bash
# SGA SLURM environment setup.
#
# Copy this file to the directory where you will generate and submit SLURM
# scripts, edit as needed, then run generate-sga-slurm from that directory:
#
#   cp /path/to/SGA/etc/sga-env.sh .
#   $EDITOR sga-env.sh

module load conda
conda activate /global/common/software/desi/users/ioannis/SGA

# ---- NERSC shared data paths (year-independent) ----
export COSMO=/dvs_ro/cfs/cdirs/cosmo
export DESI=/global/cfs/cdirs/desicollab/users/ioannis
export LEGACY_SURVEY_BASEDIR=$COSMO/work/legacysurvey
export SKY_TEMPLATE_DIR=$COSMO/work/legacysurvey/dr11/calib/sky_pattern
export GAIA_CAT_DIR=$COSMO/data/gaia/dr3/healpix
export GAIA_CAT_PREFIX=healpix
export GAIA_CAT_SCHEME=nested
export GAIA_CAT_VER=3
export UNWISE_COADDS_DIR=$COSMO/data/unwise/neo11:$COSMO/data/unwise/allwise
export TYCHO2_KD_DIR=$COSMO/staging/tycho2
export PS1CAT_DIR=$COSMO/work/ps1/cats/chunks-qz-star-v3
export DUST_DIR=$COSMO/data/dust/v0_1
export GALEX_DIR=$COSMO/data/galex/images

# ---- SGA 2025 release paths — edit for each new release ----
export SGA_DIR=$DESI/SGA/2025
export SGA_PUBLIC_DIR=$COSMO/www/sga/2025
export SGA_DATA_DIR=${COSMO}/data/sga/2025/data
#export SGA_DATA_DIR=${PSCRATCH}/SGA2025-data
export SGA_HTML_DIR=${PSCRATCH}/SGA2025-html
# Set to the current versioned parent refcat (kd-tree format):
#export LARGEGALAXIES_CAT=$COSMO/work/legacysurvey/sga/2025/SGA2025-beta-parent-refcat-v1.6.kd.fits

# ---- Thread / process controls ----
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONNOUSERSITE=1
export MPICH_GNI_FORK_MODE=FULLCOPY

# ---- Cache directories ----
# Rooted in the submission directory so the cache persists across jobs and is
# reused without rebuilding. mkdir -p and cp -rn are both safe under concurrent
# access (idempotent mkdir; no-clobber copy).
_sga_cache=${SLURM_SUBMIT_DIR:-$PWD}/.sga-cache

export XDG_CACHE_HOME=$_sga_cache/xdg-cache
export XDG_CONFIG_HOME=$_sga_cache/xdg-config
export MPLCONFIGDIR=$_sga_cache/matplotlib
export CUDA_CACHE_PATH=$_sga_cache/.nv/ComputeCache
export CUPY_CACHE_DIR=$_sga_cache/.cupy/kernel_cache

mkdir -p \
    "$XDG_CACHE_HOME/astropy" \
    "$XDG_CONFIG_HOME/astropy" \
    "$MPLCONFIGDIR" \
    "$CUDA_CACHE_PATH" \
    "$CUPY_CACHE_DIR"

# Seed from home dir on first use; subsequent jobs skip silently.
cp -rn ~/.astropy/cache/. "$XDG_CACHE_HOME/astropy/"   2>/dev/null || true
cp -rn ~/.astropy/config/. "$XDG_CONFIG_HOME/astropy/"  2>/dev/null || true
cp -rn ~/.cache/matplotlib/. "$MPLCONFIGDIR/"           2>/dev/null || true
