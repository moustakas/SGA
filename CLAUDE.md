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
- `bin/` - Active executable scripts. Key scripts:
  - `SGA2025-mpi` - MPI processing driver (coadds, ellipse, htmlplots, htmlindex)
  - `generate-sga-slurm` - Generate NERSC SLURM batch scripts for any SGA processing stage (see `etc/README.sga-slurm`)
  - `SGA2025-ned-query` - Query NED by name and position; writes per-region CSV files
  - `SGA2025-ned-merge` - Merge byname/bypos NED CSVs into `ned-merged-{region}.fits`
  - `SGA2025-build-catalog` - Merge beta catalog + NED + DESI DR1 + LVD into final per-region FITS, then merge dr11-south + dr11-north into a single deduplicated `SGA2025-{version}.fits`; derives `GALAXY` and `ALTNAMES` columns (see below)
- `archive/bin-SGA2025/` - Archived SGA-2025 processing scripts (processing complete)
- `archive/bin-SGA2020/` - Archived SGA-2020 scripts (paper and data release complete)
- `py/SGA/data/SGA2025/` - Reference CSVs used during SGA-2025 processing (overlays, VI lists, etc.)
- `py/SGA/data/SGA2020/` - Small SGA-2020 reference files
- `doc/` - Sphinx documentation source (RTD, theme: furo). Key files:
  - `conf.py`, `index.rst`, `install.rst` - Sphinx config and landing pages
  - `sga2025.rst` - SGA-2025 data release page (naming, data model, file inventory)
  - `_static/custom.css` - custom CSS (section spacing, table alignment)
  - `requirements.txt` - doc build dependencies (sphinx, furo, sphinx-design)
  - `SGA2025/` - SGA-2025 analysis and calibration notebooks (distinct from `sga2025.rst`)
  - `SGA2020/` - SGA-2020 QA notebooks (archived)
  - `tutorials/` - User-facing tutorial notebooks
- `science/SGA2026/` - SGA-2026/27 planning documents and future science analysis (see `completeness-plan.md`)
- `science/SGA2025/` - Science analysis notebooks for the SGA-2025 paper
- `science/SGA2020/` - SGA-2020 science figures and scripts
- `etc/` - Conda environment specs, NERSC/laptop setup scripts, and SLURM tooling (see `etc/README.md`)
  - `sga-env.sh` - SLURM job environment template; copy to working directory and edit before running `generate-sga-slurm`
  - `README.sga-slurm` - Step-by-step SLURM workflow (copy env → plan → generate → submit)
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
bash etc/create-env-sga.sh

# Laptop (requires micromamba)
bash etc/create-env-laptop-sga.sh
```

The shared NERSC environment lives at:
`/global/common/software/desi/users/ioannis/SGA`

## Documentation

Sphinx docs are built from `doc/` (not `docs/`) and hosted on Read the Docs
under the project slug **SGA** → `https://sga.readthedocs.io`.

Build locally:
```bash
pip install -e ".[doc]"          # installs sphinx, furo, sphinx-design
sphinx-build doc doc/_build/html
```

The `[doc]` optional extras are defined in `pyproject.toml`. RTD build
configuration is in `.readthedocs.yaml` (points at `doc/conf.py`, Python 3.12).

Note: `doc/SGA2025/` (notebooks) and `doc/sga2025.rst` (RTD page) are
different things — don't confuse them.

## Key Commands

### Generating and submitting SLURM jobs
```bash
# Set up a working directory (see etc/README.sga-slurm for full workflow)
mkdir -p ~/runs/sga2025-htmlplots && cd ~/runs/sga2025-htmlplots
cp /path/to/SGA/etc/sga-env.sh . && $EDITOR sga-env.sh   # set release-specific paths

# Preview jobs before generating
generate-sga-slurm --stage htmlplots --nodes 32 --mp 4 --plan
generate-sga-slurm --stage coadds --nodes 16 --plan          # GPU by default

# Generate .slurm files and submit
generate-sga-slurm --stage htmlplots --nodes 32 --mp 4
sbatch sga2025-htmlplots-dr11-south.slurm

# Debug run for a specific index range or galaxy list
generate-sga-slurm --stage htmlplots --nodes 1 --mp 4 \
    --qos debug --time 00:15:00 --first 1024 --last 2048
generate-sga-slurm --stage htmlplots --nodes 1 --mp 4 \
    --qos debug --time 00:15:00 --galaxylist "NGC1068,NGC4258"
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
- `SGA_PUBLIC_DIR` - Location of the final public catalogs (`/dvs_ro/cfs/cdirs/cosmo/www/sga/2025`); accessed via `SGA.sga_public_dir()`

## Architecture Patterns

### Data Model
- Uses `astropy.Table` for catalog data
- FITS I/O via `fitsio` library
- KD-tree structures for fast spatial matching

### Processing
- MPI support via `mpi4py` for distributed processing
- `multiprocessing.Pool` for CPU parallelization
- Region-specific configurations (`dr11-north`, `dr11-south`)

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

### GALAXY and ALTNAMES derivation (`bin/SGA2025-build-catalog`)

`GALAXY` and `ALTNAMES` are derived from NED's pipe-separated `CROSSIDS` field in
this order (all steps applied per object):

1. **HOST fix** — if `CROSSIDS[0]` contains `"HOST"` (e.g. `"SN 2003H HOST"`),
   scan forward for the first non-HOST entry, promote it to position 0, and
   append the original HOST name at the end (deduplicated, case-insensitive).
2. **Famous-name promotion** — `_find_preferred_idx` / `_PREFERRED_PREFIXES`:
   the highest-priority well-known catalog name anywhere in `CROSSIDS` is
   promoted to position 0, regardless of NED's ordering. Priority order:
   Messier > NGC > IC > UGCA/UGC > ESO > ESO-LV > MCG > ARP > MRK > DDO >
   KDG > VCC > CGCG > PGC > PGC1 > AGC > FCC > HCG > HIPASS. Names absent
   from this list (2MASX, SDSS, WISE, ICRF, etc.) remain as GALAXY only when
   nothing higher-priority exists.
3. **Dedup** — remove any entry from `CROSSIDS[1:]` that duplicates `CROSSIDS[0]`
   (case-insensitive) before building ALTNAMES.
4. **Special-character fix** — if `CROSSIDS[0]` fails `_SAFE_GALAXY_RE`
   (`^[a-zA-Z0-9 +\-._:]+$`), scan ALTNAMES candidates for a clean alternative
   and promote it (original moves into ALTNAMES). Common trigger: NED names
   starting with `[`.
5. **Prefix normalization** — uppercase known catalog-name prefixes via
   `_normalize_galaxy_prefix` / `_PREFIX_NORM_RE` (e.g. `Mrk` → `MRK`,
   `Arp` → `ARP`, `UGCa` → `UGCA`). Covered prefixes: 2MASX, UGCA, UGC, NGC,
   IC, PGC, ESO, MCG, ARP, MRK, DDO, KDG, VCC, FCC, CGCG, HIPASS, SDSS, SBS,
   KUG, UZC, MGC, HCG. `Messier` is intentionally excluded (kept mixed-case).
6. **OBJNAME fallback** — if NED never matched, copy `OBJNAME` → `GALAXY`
   (logged as a warning).
7. **Hand-curated overrides** — `GALAXY_OVERRIDES` dict at module level in
   `bin/SGA2025-build-catalog`; add entries there for one-off fixes that the
   rules above cannot handle (e.g. `'MESSIER 109': 'Messier 109'`).
8. **Assert** — `GALAXY` is never empty after all fixes.

`ALTNAMES` (str200) = first three remaining CROSSIDS (after the promoted GALAXY is removed),
pipe-separated, with prefix normalization and case-insensitive deduplication applied;
empty string when none remain.

### Region merging (`bin/SGA2025-build-catalog`)

After both per-region catalogs (`SGA2025-dr11-south-{version}.fits`,
`SGA2025-dr11-north-{version}.fits`) are built, `merge_regions()` combines
them into a single `SGA2025-{version}.fits` (both `SGA2025` and `TRACTOR`
extensions), written to the same `--outdir`:

- Objects present in both regions (`REGION` has both `REGIONBITS['dr11-south']`
  and `REGIONBITS['dr11-north']` set, from `SGA.coadds.REGIONBITS`) are
  deduplicated to a single row, keeping whichever side has more "good" bands
  (`FMASKED_AP00_{G,R,I,Z} < 0.5`). Ties are broken by preferring good r-band
  coverage when only one side has it, then by defaulting to dr11-south
  (dr11-north's z-band is substantially worse, so ties favor south in nearly
  all cases).
- `REGION` self-heal: a row that carries both region bits but has no actual
  counterpart in the other region's per-region file (some pixels overlap the
  other region's footprint but not enough to yield a processed object there)
  has `REGION` corrected down to the single region it actually came from.
- The full run — per-region processing logs (including messages from
  `read_sga_sample`/`_read_catalog` in `SGA.py`, which share the same
  singleton logger) plus the merge summary — is captured to both stdout and
  `SGA2025-merge-{version}.txt` in `--outdir`, via a `logging.Handler` that
  appends every record to a shared list written out at the end of `main()`.
- This merge step runs automatically whenever both regions are processed in
  one invocation (i.e. `--region` is not passed).

`SGA.read_sga_sample(region=None, ...)` (the default) reads this merged
catalog. Pass an explicit `region` to read one of the per-region files
instead. `region` is required (raises `ValueError` if omitted) when
`beta=True`, since there is no merged beta/pre-release catalog.

## Testing

No automated test suite. QA is performed through:
- Manual notebook-based analysis in `doc/` and `science/`
- QA plot generation via `py/SGA/qa.py`
- HTML reports for visual inspection

## External Dependencies

- Source detection pipeline: `astrometry.net` (built from source), `tractor`, `legacypipe`
- Astronomy: `astropy`, `fitsio`, `photutils`, `astroquery`, `pydl`
- Computing: `numpy`, `scipy`, `matplotlib`, `mpi4py`

## SGA-2026/27 Completeness Extension

The next SGA release targets improved completeness at smaller angular diameters (roughly 15–45 arcsec), without repeating the visual inspection (VI) bottleneck that dominated SGA-2025. The full plan is in `science/SGA2026/completeness-plan.md`.

### Two candidate populations

- **Shredded galaxies** (blue/irregular dwarfs): identified via `FRACFLUX > 0.2` in 2+ grz bands in the DR11 Tractor catalogs. Tractor splits these into multiple sub-threshold sources; photometric reconstruction recovers the parent galaxy.
- **Compact spheroidals** (early-type dwarfs, cluster members): single clean Tractor sources (`FRACFLUX ≤ 0.2`, `TYPE = DEV/SER`) below the SGA-2025 diameter threshold. No reconstruction needed; selection is the challenge.

### Pipeline stages

1. **DR11 Tractor pre-filter**: bright star masks (Gaia-based), Galactic cirrus rejection (SFD E(B-V) + bitmasks), extended-source selection.
2. **Shredded galaxy reconstruction**: Manwadkar et al. pipeline (image segmentation → color-based association → curve-of-growth photometry). At low redshift, (g-r)_rest ≈ (g-r)_obs (dust-corrected), so no K-correction is needed. To be re-implemented in `py/SGA/shredding.py`.
3. **SSL embedding similarity**: run ssl-legacysurvey inference on reconstructed cutouts; Faiss kNN against a combined anchor set of SGA-2025 galaxies + Manwadkar's DESI DR9 dwarf catalog embeddings.
4. **ZooBot morphology filter**: independent second signal (galaxy vs. artifact probability). Not yet validated on this application — test on a known sample first.
5. **Targeted VI**: only on the uncertain middle tier; confirmed results feed back into the SSL anchor set.

### Key collaborators and external code

- **Viraj Manwadkar** (Stanford/SLAC): shredding pipeline, DESI DR1 dwarf galaxy catalog, pre-computed DR9 SSL embeddings. Code: `https://github.com/virajvman/desi_dwarfs`. Paper: Manwadkar et al. (in prep.) "When Galaxies Fall Apart".
- **Risa Wechsler** (Stanford/SLAC): co-PI on Manwadkar et al.

### Planned new modules

- `py/SGA/shredding.py` — shredded galaxy reconstruction (based on Manwadkar pipeline)
- `bin/SGA2026-mpi` — MPI driver for SGA-2026 processing

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
