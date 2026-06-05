#!/bin/bash
# Build the SGA 2025 shared environment at NERSC.
# Run from the root of the SGA repository after: module load conda
#
# Usage:
#   module load conda
#   bash etc/create-env.sh

set -euo pipefail

SGA_PREFIX=/global/common/software/desi/users/ioannis/SGA

# Detect mamba vs micromamba
if command -v micromamba &>/dev/null; then
    MAMBA=micromamba
elif command -v mamba &>/dev/null; then
    MAMBA=mamba
else
    echo "Error: neither mamba nor micromamba found. Load conda first: module load conda"
    exit 1
fi

echo "Using: $MAMBA"
echo "Target: $SGA_PREFIX"
echo ""

# ---------------------------------------------------------------------------
# Step 1: conda packages (compiler, numpy, astropy, etc.)
# ---------------------------------------------------------------------------
echo "==> Creating conda environment..."
$MAMBA create --prefix $SGA_PREFIX --file etc/environment.yml --yes

# ---------------------------------------------------------------------------
# Step 2: pip packages — run inside the activated env so the C compiler and
# numpy headers are on the right paths. tractor has Cython extensions;
# legacypipe and SGA are pure Python but depend on tractor at runtime.
# astrometry.net is NOT pip-installed here; it comes from Dustin's module.
#
# --no-build-isolation: tractor's setup.py imports numpy at configure time to
# find include dirs for Cython compilation. Without this flag, pip builds in an
# isolated subprocess that doesn't inherit the env's packages, causing
# "ModuleNotFoundError: No module named 'numpy'".
# ---------------------------------------------------------------------------
echo ""
echo "==> Installing tractor (Cython build)..."
$MAMBA run -p $SGA_PREFIX pip install --no-build-isolation git+https://github.com/dstndstn/tractor

echo ""
echo "==> Installing legacypipe..."
$MAMBA run -p $SGA_PREFIX pip install --no-build-isolation git+https://github.com/legacysurvey/legacypipe

echo ""
echo "==> Installing SGA..."
$MAMBA run -p $SGA_PREFIX pip install git+https://github.com/moustakas/SGA

# ---------------------------------------------------------------------------
# Step 3: deploy activate.sh to stable location inside the env prefix
# ---------------------------------------------------------------------------
echo ""
echo "==> Deploying activate.sh..."
mkdir -p $SGA_PREFIX/etc
cp etc/activate.sh $SGA_PREFIX/etc/activate.sh
chmod +x $SGA_PREFIX/etc/activate.sh

echo ""
echo "Done. Environment is at: $SGA_PREFIX"
echo "Students can now run: bash etc/install-kernel.sh"
