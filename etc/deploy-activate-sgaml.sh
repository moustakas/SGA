#!/bin/bash
# Write the SGAML Jupyter kernel activation script to SGAML_PREFIX/etc/activate.sh.
# Called by create-env-sgaml.sh and update-env-sgaml.sh; not meant to be run directly.
#
# Usage:
#   bash etc/deploy-activate-sgaml.sh <SGAML_PREFIX> <PT_MODULE> <PYVER>

set -euo pipefail

SGAML_PREFIX=$1
PT_MODULE=$2
PYVER=$3
CLIB=$SGAML_PREFIX/clib

echo "==> Deploying activate.sh..."
mkdir -p "$SGAML_PREFIX/etc"
cat > "$SGAML_PREFIX/etc/activate.sh" << ACTIVATE
#!/bin/bash
# Jupyter kernel activation script for the SGAML environment.
# Loads the NERSC pytorch module for Python/PyTorch, then adds
# pip-installed packages from the SGAML prefix on top.
connection_file=\$1
module purge
module load ${PT_MODULE}
export PYTHONPATH=${SGAML_PREFIX}/lib/python:${SGAML_PREFIX}/lib/python${PYVER}/site-packages
export PATH=${SGAML_PREFIX}/bin:\$PATH
export LD_LIBRARY_PATH=${CLIB}/lib\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}

export SGA_DIR=\${SGA_DIR:-/global/cfs/cdirs/desicollab/users/ioannis/SGA/2025}
export SGA_PUBLIC_DIR=\${SGA_PUBLIC_DIR:-/global/cfs/cdirs/cosmo/www/sga/2025}
export SGA_DATA_DIR=\${SGA_DATA_DIR:-/dvs_ro/cfs/cdirs/cosmo/data/sga/2025/data}
export SGA_HTML_DIR=\${SGA_HTML_DIR:-/global/cfs/cdirs/cosmo/www/sga/2025/html}

# Personal dev overrides (PATH/PYTHONPATH prepends for working branches).
# Create ~/.sga_dev_env to enable; delete it to revert to the installed env.
[ -f "\$HOME/.sga_dev_env" ] && source "\$HOME/.sga_dev_env"
exec python -m ipykernel -f \$connection_file
ACTIVATE
chmod +x "$SGAML_PREFIX/etc/activate.sh"
