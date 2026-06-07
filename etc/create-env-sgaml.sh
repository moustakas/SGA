#!/bin/bash
# Build the SGAML (SGA + machine learning) shared environment at NERSC.
# Clones the existing SGA environment (avoiding astrometry.net/tractor/legacypipe
# source rebuilds), then layers in PyTorch, ssl-legacysurvey, and Zoobot.
#
# Usage:
#   module load conda
#   bash etc/create-env-sgaml.sh

set -euo pipefail

SGA_PREFIX=/global/common/software/desi/users/ioannis/SGA
SGAML_PREFIX=/global/common/software/desi/users/ioannis/SGAML

if command -v micromamba &>/dev/null; then
    MAMBA=micromamba
elif command -v mamba &>/dev/null; then
    MAMBA=mamba
else
    echo "Error: neither mamba nor micromamba found. Load conda first: module load conda"
    exit 1
fi

RUN="$MAMBA run -p $SGAML_PREFIX"

echo "Using: $MAMBA"
echo "Source: $SGA_PREFIX"
echo "Target: $SGAML_PREFIX"
echo ""

# ---------------------------------------------------------------------------
# Step 1: clone the existing SGA environment.
# mamba --clone copies all packages with binary prefix relocation, so
# compiled packages (astrometry.net, tractor, legacypipe, SGA) are preserved
# without a source rebuild.
# ---------------------------------------------------------------------------
echo "==> Cloning SGA environment..."
$MAMBA create --clone "$SGA_PREFIX" --prefix "$SGAML_PREFIX" --yes

# ---------------------------------------------------------------------------
# Step 2: install PyTorch (CUDA) and ML dependencies.
# pytorch-cuda=12.4 targets Perlmutter A100s; verify against the system
# driver with: nvidia-smi | head -1
# Conda env supports NCCL (multi-GPU) but not MPI; NCCL is sufficient for
# data-parallel training and large-scale inference with both ssl-legacysurvey
# and Zoobot.
# timm, pandas, pillow, pyarrow, tqdm are Zoobot core deps not in the SGA
# base; faiss-gpu, scikit-image, numba, umap-learn, optuna are for
# ssl-legacysurvey similarity search.
# ---------------------------------------------------------------------------
echo ""
echo "==> Installing PyTorch and ML dependencies..."
$MAMBA install -p "$SGAML_PREFIX" --yes \
    -c pytorch -c nvidia -c conda-forge \
    pytorch torchvision torchaudio "pytorch-cuda=12.4" \
    faiss-gpu \
    scikit-image \
    scikit-learn \
    h5py \
    numba \
    umap-learn \
    optuna \
    lightning \
    timm \
    pandas \
    pillow \
    pyarrow \
    tqdm

# ---------------------------------------------------------------------------
# Step 3: install ssl-legacysurvey from source.
# ---------------------------------------------------------------------------
echo ""
echo "==> Installing ssl-legacysurvey..."
SSL_DIR=$(mktemp -d)
trap "rm -rf $SSL_DIR" EXIT
git clone --depth=1 https://github.com/georgestein/ssl-legacysurvey "$SSL_DIR"
$RUN pip install "$SSL_DIR"

# ---------------------------------------------------------------------------
# Step 4: install Zoobot and its remaining pip dependencies.
# The [pytorch] extra pulls in torchmetrics, litmodels, timm, wandb,
# webdataset, huggingface_hub, and galaxy-datasets. torch/torchvision/
# lightning are already conda-installed above; pip checks the version
# constraints and skips reinstalling them.
# ---------------------------------------------------------------------------
echo ""
echo "==> Installing Zoobot..."
$RUN pip install "zoobot[pytorch]"

# ---------------------------------------------------------------------------
# Step 5: deploy activate.sh to stable location inside the env prefix.
# ---------------------------------------------------------------------------
echo ""
echo "==> Deploying activate.sh..."
mkdir -p "$SGAML_PREFIX/etc"
cat > "$SGAML_PREFIX/etc/activate.sh" << 'ACTIVATE'
#!/bin/bash
# Jupyter kernel activation script for the SGAML environment.
connection_file=$1
SGAML_PREFIX=/global/common/software/desi/users/ioannis/SGAML
unset PYTHONPATH
module purge
exec ${SGAML_PREFIX}/bin/python -m ipykernel -f $connection_file
ACTIVATE
chmod +x "$SGAML_PREFIX/etc/activate.sh"

echo ""
echo "Done. Environment is at: $SGAML_PREFIX"
echo "Run 'bash etc/install-kernel-sgaml.sh' to register the Jupyter kernel."
