#!/bin/bash
# Jupyter kernel activation script for the SGA 2025 environment.
# Called by kernel.json with the connection file as its only argument.

connection_file=$1

SGA_PREFIX=/global/common/software/desi/users/ioannis/SGA

unset PYTHONPATH
module purge

export SGA_DIR=${SGA_DIR:-/dvs_ro/cfs/cdirs/cosmo/data/sga/2025}
export SGA_DATA_DIR=${SGA_DATA_DIR:-/dvs_ro/cfs/cdirs/cosmo/data/sga/2025/data}
export SGA_HTML_DIR=${SGA_HTML_DIR:-/dvs_ro/cfs/cdirs/cosmo/data/sga/2025/html}

exec ${SGA_PREFIX}/bin/python -m ipykernel -f $connection_file
