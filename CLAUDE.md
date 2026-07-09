# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

The Siena Galaxy Atlas (SGA) is an astronomical survey project that delivers multicolor images and model fits for a diameter-limited sample of large galaxies using Legacy Survey imaging (deep grz optical + unWISE W1-W4 mid-infrared). The project builds catalogs, performs ellipse photometry, generates image cutouts, and creates quality assurance visualizations.

## Repository Structure

- `py/SGA/` - Main Python package with core modules:
  - `parent.py` - Build parent sample from external catalogs (NED, HyperLeda, LVD, etc.)
  - `SGA.py` - Core definitions: sample bits, mask bits, version control, `read_sample` (parent catalog, pre-ellipse-fitting) and `read_sga_sample` (final ellipse/Tractor catalog)
  - `ellipse.py` - Ellipse photometry fitting
  - `external.py` - External catalog parsing (HyperLeda, NED, SDSS, Gaia)
  - `qa.py` - QA plot generation
  - `html.py` - Searchable HTML QA page generation
  - `groups.py` - Galaxy group finding via spherical clustering
  - `io.py` - FITS I/O, coordinate conversions
  - `logger.py` - Unified logging (distinct from DESI loggers)
  - `photoz.py` - Random-forest photometric-redshift estimation, trained on the spectroscopic subsample (`Z_IVAR>0`)
  - `cosmo.py` - Thin fiducial-cosmology wrapper (flat LambdaCDM, H0=70, Om0=0.3); exact luminosity distance and its inverse, swappable to a different cosmology (e.g. DESI) in one place
  - `brick.py` - Brick tiling geometry (`custom_brickname`, footprint calculations)
  - `calibrate.py` - Rr(26) isophotal-radius calibration
  - `coadds.py` - Coadd/mosaic pipeline; pixel scales (`PIXSCALE`, `GALEX_PIXSCALE`, `UNWISE_PIXSCALE`) and region-run definitions (`RUNS`, `REGIONBITS`)
  - `cutouts.py` - Generate large batches of (annotated) image cutouts
  - `geometry.py` - Ellipse-based geometry: light-weighted moment fitting, elliptical apertures/masks and overlap testing, Tractor ellipticity conversion, literature-catalog geometry selection/merging
  - `mpi.py` - CLI/argument parsing for the `bin/SGA2025-mpi` processing driver
  - `misc.py` - Miscellaneous utilities (e.g. `viewer_inspect` for Legacy Survey viewer VI catalogs)
  - `photo.py` - Early, simpler photometry prototype; not wired into any driver script and currently broken (imports the deliberately-removed `SGA.find_galaxy`), superseded by `ellipse.py`
  - `sky.py` - Sky-position utilities (e.g. Magellanic Cloud membership flagging)
  - `ssl.py` - Build ssl-legacysurvey input catalogs from SGA-2025 mosaics and run MoCo v2 inference
  - `ssl_build_montage.py` - Build sorted PDF montages of galaxy cutouts from SSL nearest-neighbour similarity matrices
  - `ssl_sort.py` - Generate UMAP representations of HDF5 cutout chunks for SSL similarity search
  - `util.py` - General support utilities and constants (e.g. `C_LIGHT`, `TINY`)
  - `webapp/` - Legacy Django-based SGA-2020 catalog web viewer; deployed separately at NERSC (`cosmo/webapp/sga-webapp`), not part of the SGA-2025 pipeline
- `bin/` - Active executable scripts. Note: only `SGA2025-mpi` and `generate-sga-slurm` are installed as entry points via `pyproject.toml`'s `script-files`; the rest are run by adding this checkout's `bin/` to `PATH`. Key scripts:
  - `SGA2025-mpi` - MPI processing driver (coadds, ellipse, htmlplots, htmlindex)
  - `generate-sga-slurm` - Generate NERSC SLURM batch scripts for any SGA processing stage (see `etc/README.sga-slurm`)
  - `SGA2025-ned-query` - Query NED by name and position; writes per-region CSV files
  - `SGA2025-ned-merge` - Merge byname/bypos NED CSVs into `ned-merged-{region}.fits`
  - `SGA2025-ned-lvd-patch` - One-time patch to strip LVD rows from completed `ned-byname-{region}.csv` files so `SGA2025-ned-query --byname` can re-query them with NED-friendly names
  - `SGA2025-build-catalog` - Merge beta catalog + NED + DESI DR1 + LVD (+ optional `SGA2025-photoz` output, see `--photoz-catalog`) into final per-region FITS, then merge dr11-south + dr11-north into a single deduplicated `SGA2025-{version}.fits`; derives `GALAXY` and `ALTNAMES` columns (see below)
  - `SGA2025-photoz` - Train/cross-validate the `SGA.photoz` random-forest model and predict `Z_PHOT`/`Z_PHOT_ERR` for the full sample; writes `SGA2025-photoz-{version}.fits`/`.joblib` to `$SGA_DIR/photoz`, for optional input to `SGA2025-build-catalog --photoz-catalog`
  - `SGA2025-desi-dr1-match` - Match the SGA sample against DESI DR1 (`zall-pix-iron`) or SDSS DR17 spectra within each galaxy's elliptical D26/2 aperture (or a fixed radius for `--sdss`)
  - `SGA2025-qa-redshifts` - Flag high-redshift objects for visual inspection and write a VI worksheet consumable by `SGA2025-build-catalog --vi-redshifts`
  - `SGA2025-ssl-cutouts` - Build HDF5 cutout datasets (152×152 px) from SGA-2025 coadds for SSL inference
  - `SGA2025-ssl-embeddings` - Run MoCo v2 inference on `SGA2025-ssl-cutouts` output to extract ResNet50 backbone/projection embeddings
- `archive/bin-SGA2025/` - Archived SGA-2025 processing scripts (processing complete)
- `archive/bin-SGA2020/` - Archived SGA-2020 scripts (paper and data release complete)
- `py/SGA/data/SGA2025/` - Reference CSVs used during SGA-2025 processing (overlays, VI lists, etc.)
- `py/SGA/data/SGA2020/` - Small SGA-2020 reference files
- `doc/` - Sphinx documentation source (RTD, theme: furo). Key files:
  - `conf.py`, `index.rst`, `install.rst` - Sphinx config and landing pages
  - `sga2025.rst` - SGA-2025 data release page (naming, data model, file inventory)
  - `acknowledgments.rst` - Funding, citation, and external data/tools acknowledgments
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

1. **Bad-primary fix** (`_is_bad_primary`) — if `CROSSIDS[0]` contains `"HOST"`
   (e.g. `"SN 2003H HOST"`) or is itself a transient/event discovery-survey
   designation (`_TRANSIENT_NAME_RE`: `SN ####`, `AT 20##...`, `ZTF##...`,
   `PTF`/`iPTF##...`, `ASASSN-##...`, `ATLAS##...`, `LSQ##...`, `PS##...`,
   `Gaia##...`, `CSS J...`, `MASTER OT J...`, `SNhunt#`, `DES##...`,
   `TCP`/`PNV J...` — these identify the event, not the host galaxy), scan
   forward for the first entry that is neither, promote it to position 0,
   and append the original name at the end (deduplicated, case-insensitive).
   If no such entry exists, the transient/HOST name is kept as-is (no
   OBJNAME fallback for this case — GALAXY is only backfilled from OBJNAME
   when NED returned no match at all; see step 6).
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
5. **Prefix normalization** (`_normalize_galaxy_prefix`) — applied to `GALAXY`
   and every `ALTNAMES` candidate. Four independent fixes, in order:
   - Uppercase known catalog-name prefixes via `_PREFIX_NORM_RE` (e.g. `Mrk`
     → `MRK`, `Arp` → `ARP`, `UGCa` → `UGCA`). Covered prefixes: 2MASX, UGCA,
     UGC, NGC, IC, PGC, ESO, MCG, ARP, MRK, DDO, KDG, KKH, WAS, VCC, FCC,
     CGCG, HIPASS, SDSS, SBS, KUG, UZC, MGC, HCG.
   - Insert a missing separator space via `_add_missing_prefix_space` /
     `_MISSING_SPACE_RE`, for identifiers that arrive glued together (e.g.
     HyperLeda's raw `OBJNAME` `PGC513380` → `PGC 513380`, or NED's
     `SDSSJ130746.83+093729.9` → `SDSS J130746.83+093729.9`); restricted to a
     full-string match so it never touches the unrelated `PGC1` NED catalog
     prefix (which always already carries its own space).
   - Replace NED's LaTeX-escaped `H{alpha}` with the NED-searchable `Halpha`
     (e.g. `H{alpha} Dot 08` → `Halpha Dot 08`).
   - Fix `Messier` casing via `_MESSIER_PREFIX_RE` (e.g. `MESSIER 109` →
     `Messier 109`). Kept separate from `_PREFIX_NORM_RE` because Messier is
     the one prefix that's mixed-case rather than all-caps, and both NED's
     CROSSIDS and HyperLeda's OBJNAME inconsistently use all-caps `MESSIER`.
6. **OBJNAME fallback** — if NED never matched (`GALAXY` still empty), copy
   `_normalize_galaxy_prefix(OBJNAME)` → `GALAXY` (logged as a warning).
7. **Case sync** (`_sync_galaxy_case`) — when `GALAXY` matches `OBJNAME`
   case-insensitively, adopt `_normalize_galaxy_prefix(OBJNAME)` rather than
   `OBJNAME` verbatim: HyperLeda's `OBJNAME` is often the properly-cased
   reference (`HYDRA A` + OBJNAME `Hydra A` → `Hydra A`), but not always
   (OBJNAME `kkh 027`/`MESSIER 083` are themselves badly cased) — routing
   through prefix normalization lets step 5's rules win whenever they apply,
   while still preferring OBJNAME's casing for names normalization doesn't
   touch (Hydra, Leo). Runs twice: once here, and again after step 8, since
   that step can promote an ALTNAMES entry whose casing differs from OBJNAME.
8. **OBJNAME-based promotion** — if an ALTNAMES entry contains OBJNAME as a
   case-insensitive substring (or a weaker last-token/zero-pad match; handles
   e.g. OBJNAME `CenA-MM-Dw1` matching ALTNAMES `Centaurus A:[CSC2014]
   MM-Dw1`), promote it to GALAXY. Two guards, both there to stop this from
   undoing a better choice already made upstream: candidates matching
   `_is_bad_primary` are skipped (else a transient/HOST-tagged OBJNAME, e.g.
   `ASASSN-16pd HOST`, would match the very name step 1 just demoted and
   promote it straight back), and the whole step is skipped when the current
   GALAXY is already a `GALAXY_OVERRIDES` key (else a short/generic OBJNAME,
   e.g. `Fornax`, can substring-match a demoted NED informal name still
   sitting in ALTNAMES, e.g. `Fornax Dwarf SPHEROIDAL`, and promote it back
   over the override-eligible entry, bypassing the override in step 9).
9. **Hand-curated overrides** — `GALAXY_OVERRIDES` dict at module level in
   `bin/SGA2025-build-catalog`; add entries there for one-off fixes that the
   rules above cannot handle (e.g. `'AP Lib': 'ESO 514- G 001'`, promoting a
   catalog identifier over an unrelated variable-star name NED attached to
   the same position).
10. **Assert** — `GALAXY` is never empty after all fixes.

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
