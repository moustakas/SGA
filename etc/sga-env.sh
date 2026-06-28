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
export UNWISE_COADDS_DIR=$COSMO/data/unwise/neo11:$COSMO/data/unwise/allwise
export TYCHO2_KD_DIR=$COSMO/staging/tycho2
export PS1CAT_DIR=$COSMO/work/ps1/cats/chunks-qz-star-v3
export DUST_DIR=$COSMO/data/dust/v0_1
export GALEX_DIR=$COSMO/data/galex/images

# ---- SGA 2025 release paths — edit for each new release ----
export SGA_DIR=$DESI/SGA/2025
export SGA_PUBLIC_DIR=$COSMO/www/sga/2025
export SGA_DATA_DIR=${PSCRATCH}/SGA2025-data
export SGA_HTML_DIR=${PSCRATCH}/SGA2025-html
# Set to the current versioned parent refcat (kd-tree format):
#export LARGEGALAXIES_CAT=$COSMO/work/legacysurvey/sga/2025/SGA2025-beta-parent-refcat-v1.6.kd.fits

# ---- Thread controls ----
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONNOUSERSITE=1
