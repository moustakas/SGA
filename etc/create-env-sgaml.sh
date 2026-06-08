#!/bin/bash
# Build the SGAML (SGA + machine learning) environment at NERSC.
# Uses the NERSC pytorch/2.11.0 module (Python 3.12) as the Python/PyTorch
# base and pip-installs all additional packages into $SGAML_PREFIX.
# A minimal conda env at $SGAML_PREFIX/clib provides the C libraries
# (GSL, cfitsio, netpbm, etc.) needed to build astrometry.net from source
# and at runtime by the compiled astrometry extensions.
#
# Usage:
#   module load conda
#   bash etc/create-env-sgaml.sh

set -euo pipefail

SGAML_PREFIX=/global/common/software/desi/users/ioannis/SGAML
CLIB=$SGAML_PREFIX/clib
PT_MODULE=pytorch/2.11.0

if command -v micromamba &>/dev/null; then
    MAMBA=micromamba
elif command -v mamba &>/dev/null; then
    MAMBA=mamba
else
    echo "Error: neither mamba nor micromamba found. Load conda first: module load conda"
    exit 1
fi

if ! type module &>/dev/null 2>&1; then
    echo "Error: 'module' command not found. Run this script on a NERSC login node."
    exit 1
fi

echo "Using: $MAMBA"
echo "Target: $SGAML_PREFIX"
echo "PyTorch module: $PT_MODULE"
echo ""

# ---------------------------------------------------------------------------
# Step 1: load the NERSC pytorch module.
# This makes python/pip point to Python 3.12 with torch, torchvision,
# lightning, torchmetrics, numpy, scipy, matplotlib, pandas, scikit-learn,
# scikit-image, h5py, pillow, pyarrow, tqdm, wandb, huggingface_hub, and
# many others already available.
# ---------------------------------------------------------------------------
echo "==> Loading $PT_MODULE..."
module load $PT_MODULE

PYVER=$(python -c "import sys; v=sys.version_info; print(f'{v.major}.{v.minor}')")
echo "    python : $PYVER"

# Packages installed with --prefix land here. Export PYTHONPATH so that
# later build steps can find packages installed earlier in this script.
mkdir -p "$SGAML_PREFIX/lib/python${PYVER}/site-packages"
export PYTHONPATH=$SGAML_PREFIX/lib/python${PYVER}/site-packages${PYTHONPATH:+:$PYTHONPATH}

# --ignore-installed prevents pip from attempting to uninstall packages in the
# read-only pytorch module location when a dependency requires a different version.
PIP="python -m pip"
PIP_INSTALL="$PIP install --prefix $SGAML_PREFIX --ignore-installed"

# ---------------------------------------------------------------------------
# Step 2: create a minimal conda env with C build dependencies.
# Provides the compiler, swig, and C libraries that astrometry.net's
# Makefile requires. The C libraries (GSL, cfitsio, netpbm) are also needed
# at runtime by the compiled astrometry extensions, so $CLIB is kept and
# $CLIB/lib is added to LD_LIBRARY_PATH in activate.sh.
# ---------------------------------------------------------------------------
echo ""
echo "==> Creating C build dependencies env at $CLIB..."
$MAMBA create --prefix "$CLIB" --yes -c conda-forge \
    c-compiler \
    swig \
    pkgconf \
    cairo \
    cfitsio \
    gsl \
    libjpeg-turbo \
    libpng \
    netpbm \
    wcslib

# ---------------------------------------------------------------------------
# Step 3: build astrometry.net from source.
# Compiler and C libraries come from $CLIB; Python comes from the pytorch
# module. Parallel build is disabled upstream (known issue).
# ---------------------------------------------------------------------------
echo ""
echo "==> Building astrometry.net from source..."

ASTROM_DIR=$(mktemp -d)
trap "rm -rf $ASTROM_DIR" EXIT

git clone --depth=1 https://github.com/dstndstn/astrometry.net "$ASTROM_DIR"

export PATH=$CLIB/bin:$PATH
export LD_LIBRARY_PATH=$CLIB/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

ASTROM_ENV=(
    SYSTEM_GSL=yes
    FITSIO_USE_SYSTEM_FITSIO=1
    "PKG_CONFIG_PATH=${CLIB}/lib/pkgconfig"
    "NETPBM_INC=-I${CLIB}/include/netpbm"
    "NETPBM_LIB=-L${CLIB}/lib -lnetpbm"
)

env "${ASTROM_ENV[@]}" make -C "$ASTROM_DIR" -j1
env "${ASTROM_ENV[@]}" make -C "$ASTROM_DIR" -j1 py
env "${ASTROM_ENV[@]}" make -C "$ASTROM_DIR" -j1 install \
    INSTALL_DIR="$SGAML_PREFIX"

# ---------------------------------------------------------------------------
# Step 4: pip install Python packages not provided by the pytorch module.
# cython is required for tractor's Cython build in Step 5.
# ---------------------------------------------------------------------------
echo ""
echo "==> Installing Python dependencies..."
$PIP_INSTALL \
    astropy \
    fitsio \
    photutils \
    pydl \
    cython \
    numba \
    umap-learn \
    optuna \
    timm \
    faiss-cpu \
    webdataset \
    litmodels \
    galaxy-datasets

# ---------------------------------------------------------------------------
# Step 5: install tractor (Cython build).
# --no-build-isolation: tractor's setup.py imports numpy at configure time;
# numpy is available from the pytorch module's site-packages.
# ---------------------------------------------------------------------------
echo ""
echo "==> Installing tractor (Cython build)..."
$PIP_INSTALL --no-build-isolation \
    git+https://github.com/dstndstn/tractor

# ---------------------------------------------------------------------------
# Step 6: install legacypipe.
# Its version string (e.g. "DR11.1.0.3.g...") is not PEP 440 compliant;
# clone, patch, and install.
# ---------------------------------------------------------------------------
echo ""
echo "==> Installing legacypipe..."
LP_DIR=$(mktemp -d)
git clone --depth=1 https://github.com/legacysurvey/legacypipe "$LP_DIR"
sed -i "s/version = get_git_version.*/version = '0.0.0'/" "$LP_DIR/setup.py"
$PIP_INSTALL --no-build-isolation "$LP_DIR"
rm -rf "$LP_DIR"

# ---------------------------------------------------------------------------
# Step 7: install SGA, ssl-legacysurvey, and Zoobot.
# zoobot[pytorch] pulls in any remaining deps; torch/torchvision/lightning/
# torchmetrics are already satisfied by the pytorch module.
# ---------------------------------------------------------------------------
echo ""
echo "==> Installing SGA..."
$PIP_INSTALL git+https://github.com/moustakas/SGA

echo ""
echo "==> Installing ssl-legacysurvey..."
SSL_DIR=$(mktemp -d)
git clone --depth=1 https://github.com/georgestein/ssl-legacysurvey "$SSL_DIR"
$PIP_INSTALL "$SSL_DIR"
rm -rf "$SSL_DIR"

echo ""
echo "==> Installing Zoobot..."
$PIP_INSTALL "zoobot[pytorch]"

# ---------------------------------------------------------------------------
# Step 8: deploy activate.sh to stable location inside the prefix.
# ---------------------------------------------------------------------------
echo ""
echo "==> Deploying activate.sh..."
mkdir -p "$SGAML_PREFIX/etc"
cat > "$SGAML_PREFIX/etc/activate.sh" << ACTIVATE
#!/bin/bash
# Jupyter kernel activation script for the SGAML environment.
# Loads the NERSC pytorch module for Python/PyTorch, then adds
# pip-installed packages from the SGAML prefix on top.
connection_file=\$1
module purge
module load ${PT_MODULE}
export PYTHONPATH=${SGAML_PREFIX}/lib/python:${SGAML_PREFIX}/lib/python${PYVER}/site-packages
export PATH=${SGAML_PREFIX}/bin:\$PATH
export LD_LIBRARY_PATH=${CLIB}/lib\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}
export SGA_DIR=\${SGA_DIR:-/dvs_ro/cfs/cdirs/cosmo/data/sga/2025}
export SGA_DATA_DIR=\${SGA_DATA_DIR:-/dvs_ro/cfs/cdirs/cosmo/data/sga/2025/data}
export SGA_HTML_DIR=\${SGA_HTML_DIR:-/dvs_ro/cfs/cdirs/cosmo/data/sga/2025/html}
# Personal dev overrides (PATH/PYTHONPATH prepends for working branches).
# Create ~/.sga_dev_env to enable; delete it to revert to the installed env.
[ -f "\$HOME/.sga_dev_env" ] && source "\$HOME/.sga_dev_env"
exec python -m ipykernel -f \$connection_file
ACTIVATE
chmod +x "$SGAML_PREFIX/etc/activate.sh"

# ---------------------------------------------------------------------------
# Step 9: clean up pip/conda caches to reclaim disk space.
# $CLIB is kept because its shared libraries (GSL, cfitsio, netpbm) are
# needed at runtime by the compiled astrometry extensions.
# ---------------------------------------------------------------------------
echo ""
echo "==> Cleaning up caches..."
python -m pip cache purge
$MAMBA clean --all --yes

echo ""
echo "Done. Environment is at: $SGAML_PREFIX"
echo "Run 'bash etc/install-kernel-sgaml.sh' to register the Jupyter kernel."
