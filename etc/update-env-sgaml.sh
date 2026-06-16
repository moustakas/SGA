#!/bin/bash
# Update pip-installed packages in the SGAML environment.
#
# Usage:
#   module load conda
#   bash etc/update-env-sgaml.sh                          # update all packages
#   bash etc/update-env-sgaml.sh sga                      # update SGA only
#   bash etc/update-env-sgaml.sh ssl-legacysurvey         # update ssl-legacysurvey only
#   bash etc/update-env-sgaml.sh zoobot                   # update Zoobot only
#   bash etc/update-env-sgaml.sh sga ssl-legacysurvey     # update multiple
#   bash etc/update-env-sgaml.sh --local sga /path/to/SGA # install from local checkout

set -euo pipefail

SGAML_PREFIX=/global/common/software/desi/users/ioannis/SGAML
PT_MODULE=pytorch/2.11.0

if ! type module &>/dev/null 2>&1; then
    echo "Error: 'module' command not found. Run this script on a NERSC login node."
    exit 1
fi

if command -v micromamba &>/dev/null; then
    MAMBA=micromamba
elif command -v mamba &>/dev/null; then
    MAMBA=mamba
else
    echo "Error: neither mamba nor micromamba found. Load conda first: module load conda"
    exit 1
fi

module load $PT_MODULE

PYVER=$(python -c "import sys; v=sys.version_info; print(f'{v.major}.{v.minor}')")
export PYTHONPATH=$SGAML_PREFIX/lib/python${PYVER}/site-packages${PYTHONPATH:+:$PYTHONPATH}

PIP_INSTALL="python -m pip install --prefix $SGAML_PREFIX --ignore-installed --upgrade"

update_sga() {
    echo "==> Updating SGA..."
    $PIP_INSTALL --no-deps git+https://github.com/moustakas/SGA
}

local_install() {
    local pkg=$1
    local path=$2
    echo "==> Installing $pkg from local checkout $path ..."
    python -m pip install --prefix "$SGAML_PREFIX" --ignore-installed --no-deps --force-reinstall "$path"
}

update_legacypipe() {
    echo "==> Updating legacypipe..."
    $PIP_INSTALL --no-build-isolation git+https://github.com/legacysurvey/legacypipe
}

update_tractor() {
    echo "==> Updating tractor..."
    # tractor requires swig at build time; spin up a temporary conda env,
    # use it for the build, then delete it.
    SWIG_ENV=$(mktemp -d)
    trap "rm -rf $SWIG_ENV" RETURN
    $MAMBA create --prefix "$SWIG_ENV" --yes -c conda-forge c-compiler swig
    local old_path=$PATH
    local old_ldpath=${LD_LIBRARY_PATH:-}
    export PATH=$SWIG_ENV/bin:$PATH
    export LD_LIBRARY_PATH=$SWIG_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
    $PIP_INSTALL --no-build-isolation git+https://github.com/dstndstn/tractor
    export PATH=$old_path
    export LD_LIBRARY_PATH=$old_ldpath
}

update_pydl() {
    echo "==> Updating pydl..."
    $PIP_INSTALL pydl
}

update_ssl() {
    echo "==> Updating ssl-legacysurvey..."
    SSL_DIR=$(mktemp -d)
    trap "rm -rf $SSL_DIR" EXIT
    git clone --depth=1 https://github.com/georgestein/ssl-legacysurvey "$SSL_DIR"
    $PIP_INSTALL "$SSL_DIR"
}

update_zoobot() {
    echo "==> Updating Zoobot..."
    $PIP_INSTALL "zoobot[pytorch]"
}

# Parse arguments
if [[ $1 == "--local" ]]; then
    [[ $# -lt 3 ]] && { echo "Usage: $0 --local <pkg> /path/to/checkout"; exit 1; }
    local_install "$2" "$3"
    exit 0
fi

if [[ $# -eq 0 ]]; then
    update_pydl
    update_sga
    update_legacypipe
    update_tractor
    update_ssl
    update_zoobot
    exit 0
fi

for pkg in "$@"; do
    case $pkg in
        pydl)             update_pydl ;;
        sga)              update_sga ;;
        legacypipe)       update_legacypipe ;;
        tractor)          update_tractor ;;
        ssl-legacysurvey) update_ssl ;;
        zoobot)           update_zoobot ;;
        *) echo "Unknown package: $pkg (expected pydl, sga, legacypipe, tractor, ssl-legacysurvey, zoobot)"; exit 1 ;;
    esac
done
