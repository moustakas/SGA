#!/bin/bash
# Build the SGA 2025 environment on a laptop.
# Run from the root of the SGA repository.
#
# Usage:
#   bash etc/create-env-laptop.sh
#
# Requires micromamba (https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html)

set -euo pipefail

ENV_NAME=SGA

if ! command -v micromamba &>/dev/null; then
    echo "Error: micromamba not found."
    echo "Install it from: https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html"
    exit 1
fi

RUN="micromamba run -n $ENV_NAME"

echo "Creating conda environment '$ENV_NAME'..."
micromamba create -n $ENV_NAME --file etc/environment-laptop.yml --yes

# ---------------------------------------------------------------------------
# pip installs — must run after the env exists so the compiler, swig, and
# numpy headers are all on the right paths.
# --no-build-isolation: prevents pip from using an isolated subprocess that
# lacks numpy/swig, which breaks both astrometry.net and tractor builds.
# pydl and legacypipe are pure Python but also installed here for consistency.
# ---------------------------------------------------------------------------

echo ""
echo "==> Installing pydl..."
$RUN pip install pydl

echo ""
echo "==> Installing astrometry.net (C + SWIG build)..."
$RUN pip install --no-build-isolation git+https://github.com/dstndstn/astrometry.net

echo ""
echo "==> Installing tractor (Cython build)..."
$RUN pip install --no-build-isolation git+https://github.com/dstndstn/tractor

echo ""
echo "==> Installing legacypipe..."
$RUN pip install --no-build-isolation git+https://github.com/legacysurvey/legacypipe

echo ""
echo "==> Installing SGA..."
$RUN pip install git+https://github.com/moustakas/SGA

echo ""
echo "==> Registering Jupyter kernel..."
$RUN python -m ipykernel install --user --name SGA --display-name "SGA 2025"

echo ""
echo "Done. Activate with: micromamba activate $ENV_NAME"
