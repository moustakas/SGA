# SGA 2025 — NERSC Jupyter Kernel

This directory contains everything needed to run SGA in a NERSC JupyterHub
notebook.  There are two audiences: **ioannis** (who creates and maintains the
shared environment) and **students** (who just install the kernel).

## Architecture

```
tractor/perlmutter-2 module  (Dustin Lang's build)
  └─ tractor, astrometry.net C extensions, LD_LIBRARY_PATH

SGA conda env  (ioannis, at SGA_PREFIX)
  └─ python 3.13, numpy, astropy, fitsio, matplotlib, scipy,
     photutils, pydl, ipykernel, legacypipe, SGA

activate.sh  (layers the module on top of the conda env)
kernel.json  (registered per-user via install-kernel.sh)
```

`tractor` and `astrometry.net` are **not** installed into the conda env —
they are provided at kernel-launch time by Dustin's module, which prepends
their paths to `PYTHONPATH` and `LD_LIBRARY_PATH`.  The conda env's Python
binary is always used explicitly, so the module cannot hijack the interpreter.

---

## For ioannis: Creating the shared environment

Do this once (or when rebuilding after a system update).

```bash
# 1. Create the micromamba env
micromamba create --prefix /global/common/software/desi/users/ioannis/SGA \
                  --file etc/environment.yml

# 2. Activate it
micromamba activate /global/common/software/desi/users/ioannis/SGA

# 3. Install legacypipe with the tractor module loaded so its deps are
#    available at install time; use --no-deps to avoid pip trying to
#    resolve tractor/astrometry (which aren't on PyPI).
module use /global/common/software/desi/users/dstn/modulefiles/
module load tractor/perlmutter-2
pip install --no-deps git+https://github.com/legacysurvey/legacypipe

# 4. Copy activate.sh to the stable location inside the env prefix
mkdir -p /global/common/software/desi/users/ioannis/SGA/etc
cp etc/activate.sh /global/common/software/desi/users/ioannis/SGA/etc/activate.sh
chmod +x /global/common/software/desi/users/ioannis/SGA/etc/activate.sh
```

### Updating SGA itself

The `environment.yml` installs SGA from GitHub.  To update in-place:

```bash
micromamba activate /global/common/software/desi/users/ioannis/SGA
pip install --upgrade git+https://github.com/moustakas/SGA
```

Or for development, install the local clone in editable mode instead:

```bash
pip install -e /path/to/local/SGA/clone
```

---

## For students: Installing the Jupyter kernel

Run this once from a NERSC login node or terminal:

```bash
bash /path/to/SGA/etc/install-kernel.sh
```

Then open (or restart) JupyterHub at https://jupyter.nersc.gov and select
**SGA 2025** from the kernel menu.

---

## Files in this directory

| File | Purpose |
|---|---|
| `environment.yml` | Micromamba env spec (maintained by ioannis) |
| `activate.sh` | Kernel launch script; source of truth lives at `SGA_PREFIX/etc/activate.sh` |
| `kernel.json` | Example kernel spec (generated dynamically by `install-kernel.sh`) |
| `install-kernel.sh` | Student-facing one-liner to register the kernel |
| `perlmutter-2` | Reference copy of Dustin's TCL module file (do not edit) |
