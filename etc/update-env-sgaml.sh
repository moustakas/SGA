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

set -euo pipefail

SGAML_PREFIX=/global/common/software/desi/users/ioannis/SGAML
PT_MODULE=pytorch/2.11.0

if ! type module &>/dev/null 2>&1; then
    echo "Error: 'module' command not found. Run this script on a NERSC login node."
    exit 1
fi

module load $PT_MODULE

PYVER=$(python -c "import sys; v=sys.version_info; print(f'{v.major}.{v.minor}')")
export PYTHONPATH=$SGAML_PREFIX/lib/python${PYVER}/site-packages${PYTHONPATH:+:$PYTHONPATH}

PIP_INSTALL="python -m pip install --prefix $SGAML_PREFIX --ignore-installed --upgrade"

update_sga() {
    echo "==> Updating SGA..."
    $PIP_INSTALL git+https://github.com/moustakas/SGA
}

update_legacypipe() {
    echo "==> Updating legacypipe..."
    $PIP_INSTALL --no-build-isolation git+https://github.com/legacysurvey/legacypipe
}

update_tractor() {
    echo "==> Updating tractor..."
    $PIP_INSTALL --no-build-isolation git+https://github.com/dstndstn/tractor
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
