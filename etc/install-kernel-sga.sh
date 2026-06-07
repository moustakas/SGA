#!/bin/bash
# Install the SGA 2025 Jupyter kernel for the current NERSC user.
# Run this once; then restart JupyterHub to see the "SGA 2025" kernel.
#
# Usage:
#   bash /path/to/SGA/etc/install-kernel.sh

set -euo pipefail

KERNEL_NAME=SGA2025
KERNEL_DIR=${HOME}/.local/share/jupyter/kernels/${KERNEL_NAME}
SGA_PREFIX=/global/common/software/desi/users/ioannis/SGA

echo "Installing SGA 2025 Jupyter kernel to ${KERNEL_DIR}"
mkdir -p ${KERNEL_DIR}

# kernel.json points to the shared activate.sh inside the conda env prefix.
cat > ${KERNEL_DIR}/kernel.json << EOF
{
 "language": "python",
 "argv": [
  "${SGA_PREFIX}/etc/activate.sh",
  "{connection_file}"
 ],
 "display_name": "SGA 2025"
}
EOF

echo "Done. Restart JupyterHub (or refresh the kernel list) to see 'SGA 2025'."
