#!/bin/bash
# Update pip-installed packages in the SGA conda environment.
#
# Usage:
#   module load conda
#   bash etc/update-env.sh                        # update all packages
#   bash etc/update-env.sh sga                    # update SGA only
#   bash etc/update-env.sh legacypipe             # update legacypipe only
#   bash etc/update-env.sh tractor                # update tractor only
#   bash etc/update-env.sh sga legacypipe         # update multiple
#
# For a local editable install (e.g. while working on a branch):
#   bash etc/update-env.sh --editable sga /path/to/SGA
#   bash etc/update-env.sh --editable legacypipe /path/to/legacypipe
#
# --no-deps is used for editable installs because all dependencies are
# already conda-managed; without it, pip may reinstall them from PyPI
# and create version conflicts.

set -euo pipefail

SGA_PREFIX=/global/common/software/desi/users/ioannis/SGA

if command -v micromamba &>/dev/null; then
    MAMBA=micromamba
elif command -v mamba &>/dev/null; then
    MAMBA=mamba
else
    echo "Error: neither mamba nor micromamba found. Run 'module load conda' first."
    exit 1
fi

RUN="$MAMBA run -p $SGA_PREFIX"

update_sga() {
    echo "==> Updating SGA..."
    $RUN pip install --upgrade git+https://github.com/moustakas/SGA
}

update_legacypipe() {
    echo "==> Updating legacypipe..."
    $RUN pip install --no-build-isolation --upgrade git+https://github.com/legacysurvey/legacypipe
}

update_tractor() {
    echo "==> Updating tractor..."
    $RUN pip install --no-build-isolation --upgrade git+https://github.com/dstndstn/tractor
}

editable_install() {
    local pkg=$1
    local path=$2
    echo "==> Editable install of $pkg from $path ..."
    $RUN pip install --no-deps -e "$path"
}

# Parse arguments
if [[ $# -eq 0 ]]; then
    update_sga
    update_legacypipe
    update_tractor
    exit 0
fi

if [[ $1 == "--editable" ]]; then
    [[ $# -lt 3 ]] && { echo "Usage: $0 --editable <sga|legacypipe|tractor> /path/to/checkout"; exit 1; }
    editable_install "$2" "$3"
    exit 0
fi

for pkg in "$@"; do
    case $pkg in
        sga)        update_sga ;;
        legacypipe) update_legacypipe ;;
        tractor)    update_tractor ;;
        *) echo "Unknown package: $pkg (expected sga, legacypipe, or tractor)"; exit 1 ;;
    esac
done
