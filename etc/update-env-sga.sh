#!/bin/bash
# Update pip-installed packages in the SGA conda environment.
#
# Usage:
#   module load conda
#   bash etc/update-env-sga.sh                        # update all packages
#   bash etc/update-env-sga.sh mpi4py                 # rebuild mpi4py (e.g. after MPICH update)
#   bash etc/update-env-sga.sh sga                    # update SGA only
#   bash etc/update-env-sga.sh isoster                # update isoster only
#   bash etc/update-env-sga.sh imagine                # update imagine only
#   bash etc/update-env-sga.sh legacypipe             # update legacypipe only
#   bash etc/update-env-sga.sh tractor                # update tractor only
#   bash etc/update-env-sga.sh sga legacypipe         # update multiple
#
# For a local editable install (e.g. while working on a branch):
#   bash etc/update-env-sga.sh --editable sga /path/to/SGA
#   bash etc/update-env-sga.sh --editable legacypipe /path/to/legacypipe
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

update_mpi4py() {
    # mpi4py must be built against Cray MPICH — never conda/pip install.
    # Re-run the source build whenever the system MPICH version changes.
    #
    # Capture the Cray 'cc' wrapper path NOW, before $MAMBA run prepends the
    # conda env's bin/ to PATH (which would shadow it with the conda-packaged
    # gcc wrapper and break the MPI header search).
    local cray_cc
    cray_cc=$(which cc 2>/dev/null) || {
        echo "Error: 'cc' not found. Is PrgEnv-gnu loaded? Try: module load PrgEnv-gnu"
        return 1
    }
    echo "==> Rebuilding mpi4py against Cray MPICH (MPICC=${cray_cc} -shared)..."
    $RUN env MPICC="${cray_cc} -shared" pip install \
        --force-reinstall --no-cache-dir --no-binary=mpi4py mpi4py
}

update_sga() {
    echo "==> Updating SGA..."
    $RUN pip install --force-reinstall --no-deps --no-cache-dir git+https://github.com/moustakas/SGA
}

update_legacypipe() {
    echo "==> Updating legacypipe..."
    $RUN pip install --no-build-isolation --force-reinstall --no-deps --no-cache-dir git+https://github.com/legacysurvey/legacypipe
}

update_tractor() {
    echo "==> Updating tractor..."
    $RUN pip install --no-build-isolation --force-reinstall --no-deps --no-cache-dir git+https://github.com/dstndstn/tractor
}

update_pydl() {
    echo "==> Updating pydl..."
    $RUN pip install --force-reinstall --no-cache-dir pydl
}

update_isoster() {
    echo "==> Updating isoster..."
    $RUN pip install --force-reinstall --no-deps --no-cache-dir git+https://github.com/MassiveSeaOtters/isoster
}

update_imagine() {
    echo "==> Updating imagine..."
    # imagine is not pip-installable; pull the latest commits in the cloned repo.
    git -C "$SGA_PREFIX/src/imagine" pull
}

editable_install() {
    local pkg=$1
    local path=$2
    echo "==> Editable install of $pkg from $path ..."
    $RUN pip install --no-deps -e "$path"
}

# Parse arguments
if [[ $# -eq 0 ]]; then
    update_mpi4py
    update_pydl
    update_sga
    update_isoster
    update_imagine
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
        mpi4py)     update_mpi4py ;;
        pydl)       update_pydl ;;
        sga)        update_sga ;;
        isoster)    update_isoster ;;
        imagine)    update_imagine ;;
        legacypipe) update_legacypipe ;;
        tractor)    update_tractor ;;
        *) echo "Unknown package: $pkg (expected mpi4py, pydl, sga, isoster, imagine, legacypipe, or tractor)"; exit 1 ;;
    esac
done
