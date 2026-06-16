# SGA — Environment Setup and Jupyter Kernels

Two environments are provided: **SGA** (core astronomy stack) and **SGAML**
(SGA + machine learning: PyTorch, ssl-legacysurvey, Zoobot).

## Contents

- [Architecture](#architecture)
- [NERSC — SGA](#nersc--sga)
- [NERSC — SGAML](#nersc--sgaml)
- [Laptop — SGA](#laptop--sga)
- [Laptop — SGAML](#laptop--sgaml)
- [Files](#files)

## Architecture

```
SGA  (/global/common/software/desi/users/ioannis/SGA)
  conda env — python 3.13, astrometry.net (built from source), tractor,
  legacypipe, pydl, numpy, astropy, fitsio, matplotlib, scipy, photutils,
  ipykernel, SGA, isoster, imagine

SGAML  (/global/common/software/desi/users/ioannis/SGAML)
  pytorch/2.11.0 module (python 3.12) — torch, torchvision, lightning,
  numpy, scipy, scikit-learn, pandas, h5py, wandb, huggingface_hub, …
  + pip prefix — astrometry.net, tractor, legacypipe, pydl, SGA,
    astropy, fitsio, photutils, ssl-legacysurvey, Zoobot, faiss-cpu,
    numba, umap-learn, optuna, timm, …
```

---

## NERSC — SGA

### One-time environment setup

Requires write access to `/global/common/software/desi/users/ioannis/`.
Run from the root of the SGA repository:

```bash
module load conda
bash etc/create-env-sga.sh
```

`create-env-sga.sh` runs in order:
1. `mamba create` — conda packages (python 3.13, compiler, swig, C libraries, numpy, astropy, …)
2. Build astrometry.net from source (`make`, `make py`, `make install`)
3. `pip install` — pydl, tractor, legacypipe, SGA, isoster, imagine
4. Deploy `activate-sga.sh` to `SGA_PREFIX/etc/activate.sh`

**Known build notes:**
- `PKG_CONFIG_PATH` must be set explicitly so the linker finds conda-provided cfitsio and GSL
- astrometry.net installs its Python package to `lib/python/`; a `.pth` file is added to site-packages so Python finds it without PYTHONPATH manipulation
- `pydl` is pip-only (not on conda-forge)
- legacypipe's version string from `git describe` is not PEP 440 compliant; the script clones the repo and patches `setup.py` before installing
- imagine has no `setup.py`/`pyproject.toml`; it is cloned to `$SGA_PREFIX/src/imagine` and a `.pth` file makes it importable

### Updating packages

```bash
module load conda
bash etc/update-env-sga.sh              # update all packages
bash etc/update-env-sga.sh sga          # update one package
bash etc/update-env-sga.sh imagine      # (git pull on the cloned repo)
bash etc/update-env-sga.sh tractor      # (needs --no-build-isolation internally)
```

To reclaim disk space after updates (pip cache is otherwise left intact to
speed up subsequent runs):

```bash
module load conda
mamba run -p /global/common/software/desi/users/ioannis/SGA pip cache purge
mamba clean --all --yes
```

### Local checkout (e.g. working on a branch)

```bash
module load conda
bash etc/update-env-sga.sh --editable sga /path/to/SGA
bash etc/update-env-sga.sh --editable legacypipe /path/to/legacypipe
```

`--no-deps` is applied automatically so pip does not reinstall conda-managed
dependencies from PyPI.

### Interactive use

```bash
module load conda
mamba activate /global/common/software/desi/users/ioannis/SGA
python   # or ipython, or run any SGA script directly
```

### One-time kernel setup (per user)

From a NERSC login node or the JupyterHub terminal, with a local clone:

```bash
bash /path/to/SGA/etc/install-kernel-sga.sh
```

Or download and run directly from GitHub:

```bash
wget -q https://raw.githubusercontent.com/moustakas/SGA/main/etc/install-kernel-sga.sh
bash install-kernel-sga.sh && rm install-kernel-sga.sh
```

### Using the kernel

Open (or restart) JupyterHub at https://jupyter.nersc.gov and select
**SGA** from the kernel menu.

---

## NERSC — SGAML

### One-time environment setup

Requires write access to `/global/common/software/desi/users/ioannis/`.
Run from the root of the SGA repository with the conda module loaded
(the script loads `pytorch/2.11.0` internally):

```bash
module load conda
bash etc/create-env-sgaml.sh
```

`create-env-sgaml.sh` runs in order:
1. `module load pytorch/2.11.0` — Python 3.12 with torch, torchvision,
   lightning, numpy, scikit-learn, pandas, and much more already available
2. `mamba create` — minimal C-library env at `SGAML_PREFIX/clib` (GSL,
   cfitsio, netpbm, …) for the astrometry.net build only; not used at runtime
3. Build astrometry.net from source
4. `pip install --prefix` — astropy, fitsio, photutils, pydl, tractor,
   legacypipe, SGA, ssl-legacysurvey, Zoobot, faiss-cpu, numba, umap-learn,
   optuna, timm, and remaining deps
5. Deploy `activate.sh` to `SGAML_PREFIX/etc/`

`--ignore-installed` is passed to all pip invocations so that pip never
attempts to uninstall packages from the read-only pytorch module location.

**Notes:**
- PyTorch and most of the ML stack come from the NERSC-optimized module;
  the conda env supports NCCL (multi-GPU) but not MPI
- `faiss-cpu` is used for similarity search; GPU-accelerated faiss would
  require a separate build against the module's CUDA installation
- To update individual pip-installed packages, re-run the relevant
  `pip install --prefix $SGAML_PREFIX --ignore-installed --upgrade` command
  after `module load conda && module load pytorch/2.11.0`

### Updating packages

```bash
module load conda
bash etc/update-env-sgaml.sh                      # update all packages
bash etc/update-env-sgaml.sh sga                  # update one package
bash etc/update-env-sgaml.sh ssl-legacysurvey     # update ssl-legacysurvey only
bash etc/update-env-sgaml.sh zoobot               # update Zoobot only
```

Supports: `pydl`, `sga`, `legacypipe`, `tractor`, `ssl-legacysurvey`, `zoobot`.

To reclaim disk space after updates:

```bash
module load conda
module load pytorch/2.11.0
python -m pip cache purge
mamba clean --all --yes
```

### One-time kernel setup (per user)

From a NERSC login node or the JupyterHub terminal, with a local clone:

```bash
bash /path/to/SGA/etc/install-kernel-sgaml.sh
```

Or download and run directly from GitHub:

```bash
wget -q https://raw.githubusercontent.com/moustakas/SGA/main/etc/install-kernel-sgaml.sh
bash install-kernel-sgaml.sh && rm install-kernel-sgaml.sh
```

### Using the kernel

Open (or restart) JupyterHub at https://jupyter.nersc.gov and select
**SGAML** from the kernel menu.

---

## Laptop — SGA

`create-env-laptop-sga.sh` handles everything — building astrometry.net
from source, pip-installing tractor, legacypipe, pydl, and SGA, and
registering the Jupyter kernel:

```bash
bash etc/create-env-laptop-sga.sh
```

The conda env name is `SGA` (set by `ENV_NAME` at the top of the script).
The same build notes apply as for NERSC, with one difference: on macOS the
version patch for legacypipe uses Python string replacement instead of
`sed -i` to avoid BSD/GNU sed compatibility issues.

C library dependencies for the astrometry.net build (all from conda-forge):
`cairo`, `cfitsio`, `gsl`, `libjpeg-turbo`, `libpng`, `netpbm`, `wcslib`, `pkgconf`.

Register the Jupyter kernel separately if needed:

```bash
micromamba activate SGA
python -m ipykernel install --user --name SGA --display-name "SGA"
```

---

## Laptop — SGAML

`create-env-laptop-sgaml.sh` builds the full ML stack in a single conda
env. Targets macOS on Apple Silicon (MPS backend; no CUDA). Python 3.12
matches the NERSC `pytorch/2.11.0` module for cross-platform consistency.

```bash
bash etc/create-env-laptop-sgaml.sh
```

`create-env-laptop-sgaml.sh` runs in order:
1. `micromamba create` — conda packages including pytorch (conda-forge build
   with MPS support), faiss-cpu, lightning, timm, and all C build deps
2. Build astrometry.net from source
3. `pip install` — pydl, tractor, legacypipe, SGA, ssl-legacysurvey, Zoobot

**Notes:**
- PyTorch uses the MPS backend on Apple Silicon for supported operations;
  some ops fall back to CPU silently
- This environment is intended for development and testing, not training
  or large-scale inference

Register the Jupyter kernel separately if needed:

```bash
micromamba activate SGAML
python -m ipykernel install --user --name SGAML --display-name "SGAML"
```

---

## Files

| File | Purpose |
|---|---|
| `environment-sga.yml` | Conda env spec for SGA (NERSC) |
| `environment-laptop-sga.yml` | Conda env spec for SGA (laptop) |
| `environment-laptop-sgaml.yml` | Conda env spec for SGAML (laptop) |
| `create-env-sga.sh` | One-time SGA env creation (NERSC) |
| `create-env-sgaml.sh` | One-time SGAML env creation (NERSC) |
| `create-env-laptop-sga.sh` | One-time SGA env creation (laptop) |
| `create-env-laptop-sgaml.sh` | One-time SGAML env creation (laptop) |
| `update-env-sga.sh` | Update or editable-install individual SGA packages (NERSC) |
| `update-env-sgaml.sh` | Update individual SGAML packages (NERSC) |
| `activate-sga.sh` | SGA kernel launch script; deployed to `SGA_PREFIX/etc/activate.sh` |
| `kernel-sga.json` | Example SGA kernel spec (generated by `install-kernel-sga.sh`) |
| `install-kernel-sga.sh` | Per-user SGA kernel registration (NERSC) |
| `install-kernel-sgaml.sh` | Per-user SGAML kernel registration (NERSC) |
