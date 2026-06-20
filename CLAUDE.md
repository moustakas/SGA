# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

The Siena Galaxy Atlas (SGA) is an astronomical survey project that delivers multicolor images and model fits for a diameter-limited sample of large galaxies using Legacy Survey imaging (deep grz optical + unWISE W1-W4 mid-infrared). The project builds catalogs, performs ellipse photometry, generates image cutouts, and creates quality assurance visualizations.

## Repository Structure

- `py/SGA/` - Main Python package with core modules:
  - `parent.py` - Build parent sample from external catalogs (NED, HyperLeda, LVD, etc.)
  - `SGA.py` - Core definitions: sample bits, mask bits, version control, `read_sga_sample`
  - `ellipse.py` - Ellipse photometry fitting
  - `external.py` - External catalog parsing (HyperLeda, NED, SDSS, Gaia)
  - `qa.py` - QA plot generation
  - `html.py` - Searchable HTML QA page generation
  - `groups.py` - Galaxy group finding via spherical clustering
  - `io.py` - FITS I/O, coordinate conversions
  - `logger.py` - Unified logging (distinct from DESI loggers)
- `bin/` - Active executable scripts; currently only `SGA2025-mpi` (pre-release QA). Future SGA releases add scripts here directly.
- `archive/bin-SGA2025/` - Archived SGA-2025 processing scripts (processing complete)
- `archive/bin-SGA2020/` - Archived SGA-2020 scripts (paper and data release complete)
- `py/SGA/data/SGA2025/` - Reference CSVs used during SGA-2025 processing (overlays, VI lists, etc.)
- `py/SGA/data/SGA2020/` - Small SGA-2020 reference files
- `doc/SGA2025/` - SGA-2025 analysis and calibration notebooks
- `doc/SGA2020/` - SGA-2020 QA notebooks (archived)
- `doc/tutorials/` - User-facing tutorial notebooks
- `science/SGA2025/` - Science analysis notebooks for the SGA-2025 paper
- `science/SGA2020/` - SGA-2020 science figures and scripts
- `etc/` - Conda environment specs and NERSC/laptop setup scripts (see `etc/README.md`)
- `docker/` - Docker configuration for multi-platform builds (production/Shifter use)
- `pyproject.toml` - Package metadata and entry points (PEP 517/518)

## Installation

The SGA package is defined in `pyproject.toml` (PEP 517/518). Install for regular use:

```bash
pip install .
```

For development (editable install from a local checkout):

```bash
pip install --no-deps -e .
```

### Full environment (NERSC or laptop)

The `etc/` directory contains conda environment specs and setup scripts that
build the full dependency stack — astrometry.net (from source), tractor,
legacypipe, pydl, and SGA — into a shared conda environment. See
`etc/README.md` for complete instructions.

```bash
# NERSC
module load conda
bash etc/create-env.sh

# Laptop (requires micromamba)
bash etc/create-env-laptop.sh
```

The shared NERSC environment lives at:
`/global/common/software/desi/users/ioannis/SGA`

## Key Commands

### Running MPI QA (pre-release)
```bash
# Interactive session at NERSC
salloc -N 1 -C cpu -A m3592 -t 04:00:00 --qos interactive
source /global/homes/i/ioannis/code/SGA/archive/bin-SGA2025/SGA2025-shifter
source /global/homes/i/ioannis/code/SGA/archive/bin-SGA2025/SGA2025-env

# QA for specific galaxies
SGA2025-mpi --datadir=/path/to/output --mp=32 --debug --galaxylist="GALAXY_NAME"
```

## Docker

```bash
# Pull and run container locally
docker pull legacysurvey/sga:latest
docker run -it legacysurvey/sga:latest

# At NERSC with Shifter
shifterimg pull docker:legacysurvey/sga:0.8.1
shifter --image docker:legacysurvey/sga:0.8.1 bash
```

## Environment Variables

- `LEGACY_SURVEY_BASEDIR` - Base directory for Legacy Survey data
- `LEGACY_SURVEY_DIR` - Legacy Survey data directory
- `SGA_DIR` - SGA working directory at NERSC (`/dvs_ro/cfs/cdirs/cosmo/work/legacysurvey/sga/2025`)
- `SGA_DATA_DIR` - SGA data directory at NERSC (`/dvs_ro/cfs/cdirs/cosmo/data/sga/2025/data`)
- `SGA_HTML_DIR` - SGA HTML output directory at NERSC (`/dvs_ro/cfs/cdirs/cosmo/work/legacysurvey/sga/2025/html`)

## Architecture Patterns

### Data Model
- Uses `astropy.Table` for catalog data
- FITS I/O via `fitsio` library
- KD-tree structures for fast spatial matching

### Processing
- MPI support via `mpi4py` for distributed processing
- `multiprocessing.Pool` for CPU parallelization
- Region-specific configurations (dr9-north, dr9-south, dr10-south, dr11-south)

### Bit Masking System
Defined in `py/SGA/SGA.py`:
- `SAMPLE` - Sample classification bits (LVD, MCLOUDS, GCLPNE, etc.)
- `OPTMASKBITS` - Optical imaging mask bits
- `GALEXMASKBITS` - GALEX UV mask bits
- `UNWISEMASKBITS` - unWISE IR mask bits

### Logging
Use the unified logger at `SGA.logger.log` to prevent conflicts with DESI and legacypipe loggers.

### Sample Versioning
Catalog versions (v0.10, v0.50, v0.60, etc.) are tracked via `SGA_version()` function in `py/SGA/SGA.py`.

## Testing

No automated test suite. QA is performed through:
- Manual notebook-based analysis in `doc/` and `science/`
- QA plot generation via `py/SGA/qa.py`
- HTML reports for visual inspection

## External Dependencies

- Source detection pipeline: `astrometry.net` (built from source), `tractor`, `legacypipe`
- Astronomy: `astropy`, `fitsio`, `photutils`, `astroquery`, `pydl`
- Computing: `numpy`, `scipy`, `matplotlib`, `mpi4py`

## ML Integration (ssl-legacysurvey, Zoobot)

### Development Workflow
- **Laptop:** develop and test analysis notebooks/scripts using small datasets
- **NERSC:** run GPU-intensive training and large-scale inference jobs in the SGAML environment

### ssl-legacysurvey

**Repo (laptop):** `/Users/ioannis/code/ssl-legacysurvey`  
**GitHub upstream:** `https://github.com/georgestein/ssl-legacysurvey`  
**Package name:** `ssl_legacysurvey`  
**Purpose:** Self-supervised learning (SSL) framework for 76 million DESI Legacy Survey DR9 galaxy images. Used to train MoCo v2 contrastive embeddings, extract per-galaxy representations, and run downstream classification/regression/similarity search. The goal is to extend SGA to leverage these deep-learning tools (e.g., embedding SGA galaxies, anomaly detection, morphology classification).

**Laptop install:** `pip install -e /Users/ioannis/code/ssl-legacysurvey` (after conda env setup via `environment.yml`)  
**NERSC:** pip-installed into the SGAML environment (see below); update with `bash etc/update-env-sgaml.sh ssl-legacysurvey`

### Key Submodules

| Submodule | Key Class / Function | Role |
|---|---|---|
| `ssl_legacysurvey.data_loaders` | `DecalsDataset` | Load galaxy images from HDF5; 152×152 px grz; supports augmentations and EBV correction |
| `ssl_legacysurvey.data_loaders` | `DecalsDataModule` | PyTorch-Lightning DataModule wrapping `DecalsDataset` |
| `ssl_legacysurvey.data_loaders` | `DecalsAugmentations` | Composable augmentation pipeline (crop, rotate, blur, noise, galactic reddening) |
| `ssl_legacysurvey.moco` | `Moco_v2` | PyTorch-Lightning MoCo v2 SSL training; configurable ResNet backbone |
| `ssl_legacysurvey.finetune` | `SSLFineTuner` | Fine-tune backbone or train MLP classification/regression head on top of SSL representations |
| `ssl_legacysurvey.finetune` | `OutputExtractor` | Pass data through network to extract latent representations (before MLP head) |
| `ssl_legacysurvey.utils` | `DecalsDataLoader` | Load images, catalog fields, and pre-computed representations from distributed HDF5 files |
| `ssl_legacysurvey.utils` | `flux_to_mag()` | Convert Legacy Survey flux (nanomaggies) → AB magnitudes |
| `ssl_legacysurvey.utils` | `sdss_rgb()` | Convert g,r,z image arrays → RGB (matches Legacy Survey viewer color stretch) |
| `ssl_legacysurvey.data_analysis` | `pca_transform()`, `umap_transform()` | Dimensionality reduction on representation vectors |

### Data Format
- **HDF5 files**: 152×152 pixel grz image stacks + ~20 catalog fields per galaxy
- **Scale**: up to 20 TB total; chunked multi-file layout indexed by `DecalsDataLoader`
- A small sample dataset lives at `ssl-legacysurvey/data/tiny_dataset.h5` for development/testing

### Entry-Point Scripts (`ssl-legacysurvey/scripts/`)
- `train_ssl.py` — MoCo v2 SSL pre-training (DDP)
- `finetune.py` — Classification/regression head training
- `predict.py` — Batch inference; writes representations to disk
- `similarity_search_nxn.py` — N×N Faiss similarity search on representations
- `dimensionality_reduction.py` — PCA/UMAP on representation vectors
- `scripts/slurm/` — SLURM wrappers for NERSC batch jobs

### Zoobot

Also installed in SGAML: `zoobot[pytorch]` — galaxy morphology classifier pre-trained on GZ DECaLS volunteer labels. Will complement ssl-legacysurvey representations for morphology-based analyses. Update with `bash etc/update-env-sgaml.sh zoobot`.

### SGAML Environment (NERSC)

The SGAML environment layers pip-installed packages on top of the NERSC `pytorch/2.11.0` module (Python 3.12, GPU-enabled PyTorch, Lightning, torchvision, etc.).

**Prefix:** `/global/common/software/desi/users/ioannis/SGAML`  
**C libraries (runtime):** `$SGAML_PREFIX/clib/lib` (GSL, cfitsio, netpbm — needed by astrometry.net)  
**Jupyter kernel activation:** `$SGAML_PREFIX/etc/activate.sh`  
**Personal dev overrides:** `~/.sga_dev_env` (prepends to PATH/PYTHONPATH; create to use a local dev branch, delete to revert)

**Activate manually on a NERSC login/compute node:**
```bash
module load pytorch/2.11.0
export PYTHONPATH=/global/common/software/desi/users/ioannis/SGAML/lib/python3.12/site-packages:$PYTHONPATH
export PATH=/global/common/software/desi/users/ioannis/SGAML/bin:$PATH
export LD_LIBRARY_PATH=/global/common/software/desi/users/ioannis/SGAML/clib/lib:$LD_LIBRARY_PATH
```

**Create / update environment:**
```bash
module load conda
bash etc/create-env-sgaml.sh                        # full build (first time)
bash etc/update-env-sgaml.sh                        # update all packages
bash etc/update-env-sgaml.sh ssl-legacysurvey       # update one package
bash etc/update-env-sgaml.sh --local sga /path/SGA  # install from local checkout
```

**SGAML-specific packages** (beyond the pytorch module): `astropy`, `fitsio`, `photutils`, `pydl`, `numba`, `umap-learn`, `optuna`, `timm`, `faiss-cpu`, `webdataset`, `litmodels`, `galaxy-datasets`, `tractor`, `legacypipe`, `astrometry.net` (from source), `ssl-legacysurvey`, `zoobot`.
