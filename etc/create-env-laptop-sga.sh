#!/bin/bash
# Build the SGA 2025 environment on a laptop (macOS or Linux).
# Run from the root of the SGA repository.
#
# Usage:
#   bash etc/create-env-laptop.sh
#
# Requires micromamba:
#   https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html

set -euo pipefail

ENV_NAME=SGA

if ! command -v micromamba &>/dev/null; then
    echo "Error: micromamba not found."
    echo "Install from: https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html"
    exit 1
fi

RUN="micromamba run -n $ENV_NAME"

# ---------------------------------------------------------------------------
# Step 1: conda packages (compiler, swig, C libraries, numpy, astropy, etc.)
# ---------------------------------------------------------------------------
echo "==> Creating conda environment '$ENV_NAME'..."
micromamba create -n $ENV_NAME --file etc/environment-laptop-sga.yml --yes

# Get the conda env prefix and Python version for use in build steps below
CONDA_PREFIX=$($RUN python -c "import sys; print(sys.prefix)")
PYVER=$($RUN python -c "import sys; v=sys.version_info; print(f'{v.major}.{v.minor}')")
echo "    prefix : $CONDA_PREFIX"
echo "    python : $PYVER"

# ---------------------------------------------------------------------------
# Step 2: build astrometry.net from source.
# astrometry.net uses a custom Makefile (not pip-installable). Key flags:
#   SYSTEM_GSL=yes              use conda-provided GSL
#   FITSIO_USE_SYSTEM_FITSIO=1  use conda-provided cfitsio
#   NETPBM_INC/LIB              point to conda-provided netpbm
# Parallel build is disabled upstream (known issue).
# ---------------------------------------------------------------------------
echo ""
echo "==> Building astrometry.net from source..."

ASTROM_DIR=$(mktemp -d)
trap "rm -rf $ASTROM_DIR" EXIT

git clone --depth=1 https://github.com/dstndstn/astrometry.net "$ASTROM_DIR"

ASTROM_ENV=(
    SYSTEM_GSL=yes
    FITSIO_USE_SYSTEM_FITSIO=1
    "PKG_CONFIG_PATH=${CONDA_PREFIX}/lib/pkgconfig"
    "NETPBM_INC=-I${CONDA_PREFIX}/include/netpbm"
    "NETPBM_LIB=-L${CONDA_PREFIX}/lib -lnetpbm"
)

$RUN env "${ASTROM_ENV[@]}" make -C "$ASTROM_DIR" -j1

$RUN env "${ASTROM_ENV[@]}" make -C "$ASTROM_DIR" -j1 py

$RUN env "${ASTROM_ENV[@]}" make -C "$ASTROM_DIR" -j1 install \
    INSTALL_DIR="$CONDA_PREFIX"

# astrometry.net installs its Python package to $INSTALL_DIR/lib/python/.
# Add a .pth file so the conda env's Python finds it without PYTHONPATH.
echo "${CONDA_PREFIX}/lib/python" \
    > "${CONDA_PREFIX}/lib/python${PYVER}/site-packages/astrometry-path.pth"

# ---------------------------------------------------------------------------
# Step 3: pip installs — must run after the env exists so the C compiler and
# numpy headers are on the right paths.
# --no-build-isolation: prevents pip from using an isolated subprocess that
# lacks numpy/swig, which breaks tractor's Cython build.
# ---------------------------------------------------------------------------
echo ""
echo "==> Installing pydl..."
$RUN pip install pydl

echo ""
echo "==> Installing tractor (Cython build)..."
$RUN pip install --no-build-isolation git+https://github.com/dstndstn/tractor

echo ""
echo "==> Installing legacypipe..."
# legacypipe's version from git describe (e.g. "DR11.1.0.3.g...") is not
# PEP 440 compliant and modern setuptools rejects it. Clone, patch, install.
# Use Python for the patch (cross-platform; avoids sed -i syntax differences
# between macOS and Linux).
LP_DIR=$(mktemp -d)
git clone --depth=1 https://github.com/legacysurvey/legacypipe "$LP_DIR"
python3 -c "
p = open('$LP_DIR/setup.py').read()
p = p.replace(\"version = get_git_version(os.path.dirname(__file__)).replace('-','.')\", \"version = '0.0.0'\")
open('$LP_DIR/setup.py', 'w').write(p)
"
$RUN pip install --no-build-isolation "$LP_DIR"
rm -rf "$LP_DIR"

echo ""
echo "==> Installing SGA..."
$RUN pip install git+https://github.com/moustakas/SGA

# ---------------------------------------------------------------------------
# Step 4: register Jupyter kernel
# ---------------------------------------------------------------------------
echo ""
echo "==> Registering Jupyter kernel..."
$RUN python -m ipykernel install --user --name $ENV_NAME --display-name "SGA 2025"

echo ""
echo "Done. Activate with: micromamba activate $ENV_NAME"
