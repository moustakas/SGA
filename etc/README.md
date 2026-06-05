# SGA 2025 — Jupyter Kernel and Environment Setup

This directory contains everything needed to run SGA in a NERSC JupyterHub
notebook or in a local (laptop) environment.  There are two audiences:
**ioannis** (who creates and maintains the shared NERSC environment) and
**students** (who just install the kernel or create their own local env).

---

## NERSC

### Architecture

```
tractor/perlmutter-2 module  (Dustin Lang's build)
  └─ astrometry.net C extensions + shared libraries

SGA conda env  (ioannis, at SGA_PREFIX)
  └─ python 3.13, numpy, astropy, fitsio, matplotlib, scipy,
     photutils, pydl, tractor, legacypipe, ipykernel, SGA

activate.sh  (layers the module on top of the conda env at kernel launch)
kernel.json  (registered per-user via install-kernel.sh)
```

`astrometry.net` is the one package **not** installed into the conda env —
it is provided at kernel-launch time by Dustin's module, which prepends
its paths to `PYTHONPATH` and `LD_LIBRARY_PATH`.  `activate.sh` calls the
conda env's Python binary explicitly so the module cannot override it.

### For ioannis: Creating the shared environment

Do this once (or when rebuilding after a system update).

```bash
# 1. Create the env (tractor and legacypipe are pip-installed automatically)
micromamba create --prefix /global/common/software/desi/users/ioannis/SGA \
                  --file etc/environment.yml

# 2. Copy activate.sh to its stable runtime location inside the env prefix
mkdir -p /global/common/software/desi/users/ioannis/SGA/etc
cp etc/activate.sh /global/common/software/desi/users/ioannis/SGA/etc/activate.sh
chmod +x /global/common/software/desi/users/ioannis/SGA/etc/activate.sh
```

That's it — `tractor`, `legacypipe`, and `SGA` are all handled by the pip
section of `environment.yml`.

### Updating SGA

```bash
micromamba activate /global/common/software/desi/users/ioannis/SGA
pip install --upgrade git+https://github.com/moustakas/SGA
```

For development, use a local editable install instead:

```bash
pip install -e /path/to/local/SGA/clone
```

### For students: Installing the Jupyter kernel

Run once from a NERSC login node or the JupyterHub terminal:

```bash
bash /path/to/SGA/etc/install-kernel.sh
```

Then open (or restart) JupyterHub at https://jupyter.nersc.gov and select
**SGA 2025** from the kernel menu.

---

## Laptop

`astrometry.net` is available on conda-forge, so no module system is needed.
Use `environment-laptop.yml`, which is identical to the NERSC spec except it
adds `astrometry` from conda-forge.

```bash
# Create the env
micromamba create -n SGA --file etc/environment-laptop.yml

# Activate and verify
micromamba activate SGA
python -c "import astrometry; import tractor; import legacypipe; import SGA"
```

> **Note:** The conda-forge `astrometry` package provides the Python bindings
> and core C libraries.  It may not include the full astrometry.net plate-solver
> (index files, `solve-field` binary).  For student notebook use — reading
> catalogs, ellipse photometry, QA plots — this is sufficient.  If you need the
> full solver, build astrometry.net from source following the upstream docs.

For a Jupyter kernel on the laptop, register the env with:

```bash
micromamba activate SGA
python -m ipykernel install --user --name SGA --display-name "SGA 2025"
```

---

## Files in this directory

| File | Purpose |
|---|---|
| `environment.yml` | Micromamba env spec for NERSC (no astrometry.net; comes from module) |
| `environment-laptop.yml` | Micromamba env spec for laptop (astrometry.net from conda-forge) |
| `activate.sh` | Kernel launch script for NERSC; deployed to `SGA_PREFIX/etc/` by ioannis |
| `kernel.json` | Example kernel spec (generated dynamically by `install-kernel.sh`) |
| `install-kernel.sh` | Student one-liner to register the NERSC kernel |
| `perlmutter-2` | Reference copy of Dustin's TCL module file (do not edit) |
