#!/bin/bash
# Jupyter kernel activation script for the SGA 2025 environment.
# Called by kernel.json with the connection file as its only argument.

connection_file=$1

SGA_PREFIX=/global/common/software/desi/users/ioannis/SGA

unset PYTHONPATH
module purge

exec ${SGA_PREFIX}/bin/python -m ipykernel -f $connection_file
