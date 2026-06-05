# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

The Siena Galaxy Atlas (SGA) is an astronomical survey project that delivers multicolor images and model fits for a diameter-limited sample of large galaxies using Legacy Survey imaging (deep grz optical + unWISE W1-W4 mid-infrared). The project builds catalogs, performs ellipse photometry, generates image cutouts, and creates quality assurance visualizations.

## Repository Structure

- `py/SGA/` - Main Python package with core modules:
  - `parent.py` - Build parent sample from external catalogs (NED, HyperLeda, LVD, etc.)
  - `SGA.py` - Core definitions: sample bits, mask bits, version control
  - `ellipse.py` - Ellipse photometry fitting
  - `external.py` - External catalog parsing (HyperLeda, NED, SDSS, Gaia)
  - `qa.py` - QA plot generation
  - `html.py` - Searchable HTML QA page generation
  - `groups.py` - Galaxy group finding via spherical clustering
  - `io.py` - FITS I/O, coordinate conversions
  - `logger.py` - Unified logging (distinct from DESI loggers)
- `bin/SGA2025/` - Executable scripts for SGA-2025 release
- `bin/SGA2020/` - Legacy SGA-2020 scripts
- `etc/` - Conda environment specs and NERSC/laptop setup scripts (see `etc/README.md`)
- `doc/` - Documentation and analysis notebooks
- `science/` - Science analysis notebooks
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

### Building Parent Sample
```bash
SGA2025-build-parent --build-parent-nocuts
SGA2025-build-parent --build-parent-vicuts
SGA2025-build-parent --build-parent-archive
SGA2025-build-parent --in-footprint --region=dr9-north
SGA2025-build-parent --in-footprint --region=dr11-south
SGA2025-build-parent --build-parent
SGA2025-build-parent --qa-parent
```

### Running MPI Processing
```bash
# Interactive session at NERSC
salloc -N 1 -C cpu -A m3592 -t 04:00:00 --qos interactive
SGA2025-shifter
source /global/homes/i/ioannis/code/SGA/bin/SGA2025/SGA2025-env

# Build reference catalog
SGA2025-mpi --build-refcat

# Process specific galaxies
SGA2025-mpi --datadir=/path/to/output --mp=32 --debug --coadds --galaxylist="GALAXY_NAME"

# QA generation
SGA2025-tractor-qa --datadir=/path/to/data
```

### Generating Cutouts
```bash
SGA2025-cutouts --region=dr9-north --ntest=16 --mp=4
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
