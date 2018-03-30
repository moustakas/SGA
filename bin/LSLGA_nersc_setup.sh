#!/bin/bash
# Set up the software environment we need in order to run legacyhalos scripts at
# NERSC.

#desiconda_version=20180103-1.2.3-img
desiconda_version=20170818-1.1.12-img

echo '$desiconda='$desiconda_version
#module use /global/common/software/desi/$NERSC_HOST/desiconda/$desiconda_version/modulefiles
module use /global/common/${NERSC_HOST}/contrib/desi/desiconda/$desiconda_version/modulefiles
module load desiconda

##################################################
# Install tractor and astrometry.net
# cd $CSCRATCH/repos
# git clone https://github.com/dstndstn/astrometry.net.git
# cd astrometry.net/
# python setup.py install --prefix=$CSCRATCH/repos/build

# cd ..
# git clone https://github.com/dstndstn/tractor.git
# cd tractor/
# make
# python setup.py install --prefix=$CSCRATCH/repos/build
##################################################

#export LEGACY_SURVEY_DIR=/global/cscratch1/sd/dstn/dr6plus
#export LEGACY_SURVEY_DIR=/global/cscratch1/sd/desiproc/dr5-new
#export LEGACY_SURVEY_DIR=/global/cscratch1/sd/dstn/dr5-new-sky
#export LEGACY_SURVEY_DIR=/global/project/projectdirs/cosmo/data/legacysurvey/dr5
#export LEGACY_SURVEY_DIR=/global/project/projectdirs/cosmo/work/legacysurvey/dr5
#export LEGACY_SURVEY_DIR=/global/cscratch1/sd/desiproc/dr7

export LEGACYPIPE_DIR=${CSCRATCH}/repos/legacypipe
export LSLGA_DIR=${CSCRATCH}/LSLGA
export LSLGA_CODE_DIR=${CSCRATCH}/repos/LSLGA

echo '$LSLGA_DIR='$LSLGA_DIR
echo '$LSLGA_CODE_DIR='$LSLGA_CODE_DIR
#echo '$LEGACY_SURVEY_DIR='$LEGACY_SURVEY_DIR

export PATH=$LEGACYPIPE_DIR/bin:$PATH
export PATH=$LSLGA_CODE_DIR/bin:$PATH
export PATH=$SCRATCH/repos/build/bin:$PATH

export PYTHONPATH=$LEGACYPIPE_DIR/py:$PYTHONPATH
export PYTHONPATH=$LSLGA_CODE_DIR:$PYTHONPATH
export PYTHONPATH=$CSCRATCH/repos/build/lib/python3.5/site-packages:$PYTHONPATH

# Force MKL single-threaded
# https://software.intel.com/en-us/articles/using-threaded-intel-mkl-in-multi-thread-application
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export KMP_AFFINITY=disabled

# To avoid problems with MPI and Python multiprocessing
export MPICH_GNI_FORK_MODE=FULLCOPY

if [ "$NERSC_HOST" == "cori" ]; then
  module use $LEGACYPIPE_DIR/bin/modulefiles/cori
fi
if [ "$NERSC_HOST" == "edison" ]; then
  module use $LEGACYPIPE_DIR/bin/modulefiles/edison
fi  

export GAIA_CAT_DIR=/project/projectdirs/cosmo/work/gaia/chunks-gaia_rel1

module load unwise_coadds
module load dust

echo '$GAIA_CAT_DIR='$GAIA_CAT_DIR
