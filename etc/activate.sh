#!/bin/bash
# Jupyter kernel activation script for the SGA 2025 environment.
# Called by kernel.json with the connection file as its only argument.
#
# Design: use the SGA conda env's Python explicitly, then layer Dustin's
# tractor/perlmutter-2 module on top via PYTHONPATH/LD_LIBRARY_PATH.
# This ensures tractor and astrometry.net C extensions are importable
# without replacing the conda Python binary.

connection_file=$1

SGA_PREFIX=/global/common/software/desi/users/ioannis/SGA

unset PYTHONPATH
module purge

module use /global/common/software/desi/users/dstn/modulefiles/
module load tractor/perlmutter-2

export LD_LIBRARY_PATH=/lib64:/usr/lib64:${LD_LIBRARY_PATH}

exec ${SGA_PREFIX}/bin/python -m ipykernel -f $connection_file
