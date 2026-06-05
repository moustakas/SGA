#!/bin/bash
# Build the SGA 2025 shared environment at NERSC.
# Run from the root of the SGA repository after: module load conda
#
# Usage:
#   module load conda
#   bash etc/create-env.sh

set -euo pipefail

SGA_PREFIX=/global/common/software/desi/users/ioannis/SGA

if command -v micromamba &>/dev/null; then
    MAMBA=micromamba
elif command -v mamba &>/dev/null; then
    MAMBA=mamba
else
    echo "Error: neither mamba nor micromamba found. Load conda first: module load conda"
    exit 1
fi

RUN="$MAMBA run -p $SGA_PREFIX"

echo "Using: $MAMBA"
echo "Target: $SGA_PREFIX"
echo ""

# ---------------------------------------------------------------------------
# Step 1: conda packages (compiler, swig, C libraries, numpy, astropy, etc.)
# ---------------------------------------------------------------------------
echo "==> Creating conda environment..."
$MAMBA create --prefix $SGA_PREFIX --file etc/environment.yml --yes

PYVER=$($RUN python -c "import sys; v=sys.version_info; print(f'{v.major}.{v.minor}')")
echo "    python : $PYVER"

# ---------------------------------------------------------------------------
# Step 2: build astrometry.net from source.
# astrometry.net uses a custom Makefile (not pip-installable). Key flags:
#   SYSTEM_GSL=yes              use conda-provided GSL
#   FITSIO_USE_SYSTEM_FITSIO=1  use conda-provided cfitsio
#   NETPBM_INC/LIB              point to conda-provided netpbm
# Parallel build is disabled upstream (known issue).
# PY_BASE_INSTALL_DIR must end in /astrometry so the Makefile installs the
# contents of the source astrometry/ dir into that package directory.
# ---------------------------------------------------------------------------
echo ""
echo "==> Building astrometry.net from source..."

ASTROM_DIR=$(mktemp -d)
trap "rm -rf $ASTROM_DIR" EXIT

git clone --depth=1 https://github.com/dstndstn/astrometry.net "$ASTROM_DIR"

$RUN env \
    SYSTEM_GSL=yes \
    FITSIO_USE_SYSTEM_FITSIO=1 \
    "NETPBM_INC=-I${SGA_PREFIX}/include/netpbm" \
    "NETPBM_LIB=-L${SGA_PREFIX}/lib -lnetpbm" \
    make -C "$ASTROM_DIR" -j1

$RUN env \
    SYSTEM_GSL=yes \
    FITSIO_USE_SYSTEM_FITSIO=1 \
    "NETPBM_INC=-I${SGA_PREFIX}/include/netpbm" \
    "NETPBM_LIB=-L${SGA_PREFIX}/lib -lnetpbm" \
    make -C "$ASTROM_DIR" -j1 py

$RUN env \
    SYSTEM_GSL=yes \
    FITSIO_USE_SYSTEM_FITSIO=1 \
    "NETPBM_INC=-I${SGA_PREFIX}/include/netpbm" \
    "NETPBM_LIB=-L${SGA_PREFIX}/lib -lnetpbm" \
    make -C "$ASTROM_DIR" -j1 install \
        INSTALL_DIR="$SGA_PREFIX" \
        PY_BASE_INSTALL_DIR="$SGA_PREFIX/lib/python${PYVER}/site-packages/astrometry" \
        PY_BASE_LINK_DIR="$SGA_PREFIX/lib/python${PYVER}/site-packages/astrometry"

# ---------------------------------------------------------------------------
# Step 3: pip installs — run inside the activated env so the C compiler and
# numpy headers are on the right paths.
# --no-build-isolation: tractor's setup.py imports numpy at configure time to
# find include dirs for Cython compilation. Without this flag, pip builds in an
# isolated subprocess that doesn't inherit the env's packages, causing
# "ModuleNotFoundError: No module named 'numpy'".
# ---------------------------------------------------------------------------
echo ""
echo "==> Installing pydl..."
$RUN pip install pydl

echo ""
echo "==> Installing tractor (Cython build)..."
$RUN pip install --no-build-isolation git+https://github.com/dstndstn/tractor

echo ""
echo "==> Installing legacypipe..."
$RUN pip install --no-build-isolation git+https://github.com/legacysurvey/legacypipe

echo ""
echo "==> Installing SGA..."
$RUN pip install git+https://github.com/moustakas/SGA

# ---------------------------------------------------------------------------
# Step 4: deploy activate.sh to stable location inside the env prefix
# ---------------------------------------------------------------------------
echo ""
echo "==> Deploying activate.sh..."
mkdir -p $SGA_PREFIX/etc
cp etc/activate.sh $SGA_PREFIX/etc/activate.sh
chmod +x $SGA_PREFIX/etc/activate.sh

echo ""
echo "Done. Environment is at: $SGA_PREFIX"
echo "Run 'bash etc/install-kernel.sh' to register the Jupyter kernel."
