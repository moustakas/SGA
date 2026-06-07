#!/bin/bash
# Install the SGAML Jupyter kernel for the current NERSC user.
# Run this once; then restart JupyterHub to see the "SGAML" kernel.
#
# Usage:
#   bash /path/to/SGA/etc/install-kernel-sgaml.sh

set -euo pipefail

KERNEL_NAME=SGAML
KERNEL_DIR=${HOME}/.local/share/jupyter/kernels/${KERNEL_NAME}
SGAML_PREFIX=/global/common/software/desi/users/ioannis/SGAML

echo "Installing SGAML Jupyter kernel to ${KERNEL_DIR}"
mkdir -p ${KERNEL_DIR}

cat > ${KERNEL_DIR}/kernel.json << EOF
{
 "language": "python",
 "argv": [
  "${SGAML_PREFIX}/etc/activate.sh",
  "{connection_file}"
 ],
 "display_name": "SGAML"
}
EOF

echo "Done. Restart JupyterHub (or refresh the kernel list) to see 'SGAML'."
