#!/usr/bin/env bash
set -euo pipefail

# =====================[  Positional arguments  ]=====================
N=${1:-1}                 # nodes
NGPU=${2:-0}              # GPUs per node (0 = CPU-only)
MP=${3:-1}                # multiprocessing cores for SGA
THREADS_PER_GPU=${4:-1}   # threads per GPU for SGA
STAGE=${5:-coadds}        # coadds|ellipse|htmlplots
REGION=${6:-dr11-south}   # dr11-south|dr9-north
DATADIR=${7:-${PSCRATCH:-/tmp}}
HTMLDIR=${8:-${PSCRATCH:-/tmp}}
shift 8

# =====================[  config  ]=====================
DESI="/global/cfs/cdirs/desicollab/users/ioannis"
COSMO="/dvs_ro/cfs/cdirs/cosmo"
SCRATCH=$PSCRATCH

#SGA_DEV="/global/u2/i/ioannis/code/SGA"
SGA_DEV="$DESI/SGA/2025/v0.40-scripts/SGA"
LEGACYPIPE_DEV="/global/common/software/desi/users/ioannis/legacypipe"
#LEGACYPIPE_DEV="$DESI/SGA/2025/v0.40-scripts/legacypipe"
PATH_PREPEND="$SGA_DEV/bin/SGA2025"

LARGEGALAXIES_CAT="$COSMO/work/legacysurvey/sga/2025/SGA2025-beta-parent-refcat-v0.40.kd.fits"
SKY_TEMPLATE_DIR="$COSMO/work/legacysurvey/dr11/calib/sky_pattern"

SGA_DIR="$DESI/SGA/2025"
SGA_DATA_DIR="${PSCRATCH}/SGA2025-data"
SGA_HTML_DIR="${PSCRATCH}/SGA2025-html"

LEGACY_SURVEY_BASEDIR="$COSMO/work/legacysurvey"

GAIA_CAT_DIR="$COSMO/data/gaia/dr3/healpix"
GAIA_CAT_PREFIX="healpix"
GAIA_CAT_SCHEME="nested"
GAIA_CAT_VER="3"

UNWISE_COADDS_DIR="$COSMO/work/wise/outputs/merge/neo11/fulldepth:$COSMO/data/unwise/allwise/unwise-coadds/fulldepth"

TYCHO2_KD_DIR="$COSMO/staging/tycho2"
PS1CAT_DIR="$COSMO/work/ps1/cats/chunks-qz-star-v3"
DUST_DIR="$COSMO/data/dust/v0_1"
GALEX_DIR="$COSMO/data/galex/images"

# Everything remaining goes to SGA2025-mpi unchanged (may be multiple flags)
raw_extra=( "$@" )

# Wrapper-only toggle: dedicate whole node to a single MPI task (1 rank/node)
one_rank_per_node=0
shared=0
sga_extra=()
for a in "${raw_extra[@]}"; do
  case "$a" in
    --one-rank-per-node|--dedicate-node) one_rank_per_node=1 ;;
    --shared) shared=1 ;;
    *) sga_extra+=( "$a" ) ;;
  esac
done

# ---------------- Build program args (forward to SGA2025-mpi) ---------------
args=( --datadir="$DATADIR" --region="$REGION" )
case "$STAGE" in
  coadds)    args+=( --coadds ) ;;
  ellipse)   args+=( --ellipse ) ;;
  htmlplots) args+=( --htmldir="$HTMLDIR" --htmlplots ) ;;
  build_catalog) args+=( --build-catalog ) ;;
  *) echo "Unknown stage: $STAGE" >&2; exit 2 ;;
esac

# GPU args
if (( NGPU > 0 )); then
  if (( one_rank_per_node )); then
    args+=( --use-gpu --threads-per-gpu="$THREADS_PER_GPU" --ngpu="$NGPU" )
  else
    args+=( --use-gpu --threads-per-gpu="$THREADS_PER_GPU" --ngpu=1 )
  fi
fi

# Append any extra SGA flags safely
if (( ${#sga_extra[@]} > 0 )); then
  args+=( "${sga_extra[@]}" )
fi

if [[ "$STAGE" != "build_catalog" ]]; then
    args+=( --mp="$MP" )
fi

# --------------------------- srun config ---------------------------
SRUN=( srun --nodes="$N" )

if (( NGPU > 0 )); then
  if (( one_rank_per_node )); then
    ntasks="$N"
    SRUN+=( --ntasks="$ntasks" --ntasks-per-node=1
            --gpus-per-task="$NGPU" --cpus-per-task=128 --cpu-bind=cores )
  else
    ntasks=$(( N * NGPU ))
    if (( shared )); then
      cpus_per_task=$(( 128 / 4 ))
    else
      cpus_per_task=$(( 128 / NGPU ))
    fi
    SRUN+=( --ntasks="$ntasks" --ntasks-per-node="$NGPU"
            --gpus-per-task=1 --cpus-per-task="$cpus_per_task"
            --cpu-bind=cores )
  fi
else
  # CPU-only path
  if (( one_rank_per_node )); then
    tpn=1
  else
    tpn=$(( 128 / MP )); (( tpn < 1 )) && tpn=1
  fi
  ntasks=$(( N * tpn ))
  SRUN+=( --ntasks="$ntasks" --ntasks-per-node="$tpn" )

  if [[ "$STAGE" != "build_catalog" ]]; then
      if (( MP > 1 )); then
        cpus_per_task=$(( MP * 2 )); cpu_bind="none"
      else
        cpus_per_task=$(( 2 * 128 * N / ntasks )); cpu_bind="cores"   # = 256/tpn
      fi
      SRUN+=( --cpus-per-task="$cpus_per_task" --cpu-bind="$cpu_bind" )
  fi
fi
SRUN+=( --no-kill --kill-on-bad-exit=0 )
#SRUN+=( --network=no_vni --no-kill --kill-on-bad-exit=0 )

if (( NGPU > 0 )); then
  echo "=== GPU resources on this node ==="
  nvidia-smi
  echo "=================================="
fi

# ===================[  Inner script run inside container ]===================
RUNCMD=$(cat <<'EOF'
set -euo pipefail

# tolerate unset dev-path knobs and export as empty if not set by caller
: "${SGA_DEV:=}"; : "${LEGACYPIPE_DEV:=}"; : "${PATH_PREPEND:=}"
export SGA_DEV LEGACYPIPE_DEV PATH_PREPEND

# Basic runtime knobs (optionally inherited from outer env)
: "${PYTHONNOUSERSITE:=1}"
: "${OMP_NUM_THREADS:=1}"
: "${MKL_NUM_THREADS:=1}"
: "${MPICH_GNI_FORK_MODE:=FULLCOPY}"
export PYTHONNOUSERSITE OMP_NUM_THREADS MKL_NUM_THREADS MPICH_GNI_FORK_MODE

# PATH tweaks (optional)
if [ -n "$PATH_PREPEND" ]; then
  export PATH="$PATH_PREPEND:$PATH"
fi

# Dev overrides: prepend local checkouts (if set)
prepend_path ()   { [ -d "$1" ] && export PATH="$1:${PATH:-}"; }
prepend_pypath () { [ -d "$1" ] && export PYTHONPATH="$1:${PYTHONPATH:-}"; }

if [ -n "$SGA_DEV" ]; then
  prepend_pypath "${SGA_DEV%/}/py"
  prepend_path   "${SGA_DEV%/}/bin/SGA2025"
fi
if [ -n "$LEGACYPIPE_DEV" ]; then
  prepend_pypath "${LEGACYPIPE_DEV%/}/py"
fi

# Per-rank caches
TMPCACHE=$(mktemp -d -p "${TMPDIR:-/tmp}" "SGA_${SLURM_JOB_ID}_${SLURM_PROCID}.XXXX") || exit 1
export XDG_CACHE_HOME="$TMPCACHE/cache"
export XDG_CONFIG_HOME="$TMPCACHE/config"
mkdir -p "$XDG_CACHE_HOME/astropy" "$XDG_CONFIG_HOME/astropy"
[ -d "$HOME/.astropy/cache"  ] && cp -r "$HOME/.astropy/cache"  "$XDG_CACHE_HOME/astropy"  || true
[ -d "$HOME/.astropy/config" ] && cp -r "$HOME/.astropy/config" "$XDG_CONFIG_HOME/astropy" || true

export MPLCONFIGDIR="$TMPCACHE/matplotlib"
mkdir -p "$MPLCONFIGDIR"
[ -d "$HOME/.cache/matplotlib" ] && cp -r "$HOME/.cache/matplotlib" "$MPLCONFIGDIR" || true

# CUDA / CuPy caches per-rank
export CUDA_CACHE_MAXSIZE="${CUDA_CACHE_MAXSIZE:-2147483648}"
export CUDA_CACHE_PATH="$TMPCACHE/.nv/ComputeCache/${SLURM_PROCID}"
export CUPY_CACHE_DIR="$TMPCACHE/.cupy/kernel_cache/${SLURM_PROCID}"
mkdir -p "$CUDA_CACHE_PATH" "$CUPY_CACHE_DIR"

# Resolve and exec SGA2025-mpi from *inside* the container
#   $1 is the desired program name (we pass "SGA2025-mpi" from the wrapper)
progname="${0:-SGA2025-mpi}"

# Prefer explicit dev checkout if provided and executable
if [[ -n "${SGA_DEV:-}" ]] && [[ -x "${SGA_DEV%/}/bin/SGA2025/SGA2025-mpi" ]]; then
  mpiscript="${SGA_DEV%/}/bin/SGA2025/SGA2025-mpi"
else
  # Otherwise search PATH that we just constructed in this environment
  mpiscript="$(command -v -- "$progname" || true)"
fi

if [[ -z "${mpiscript:-}" ]]; then
  echo "ERROR: cannot locate '$progname' in PATH, and no usable SGA_DEV at '${SGA_DEV:-}'." >&2
  echo "PATH is: $PATH" >&2
  exit 127
fi

if [[ "${SLURM_PROCID:-0}" -eq 0 ]]; then
  echo "SGA2025-mpi resolved to: $mpiscript" 1>&2
fi

exec "$mpiscript" "$@"
EOF
)

# --------------------------- Launch ---------------------------
ENVFLAGS=()
add_env() { local n="$1" v="$2"; [[ -n "$v" ]] && ENVFLAGS+=( --env "$n=$v" ); }

#ENVFLAGS+=( --env MPICH_ENV_DISPLAY=1 ) # prints MPICH config on startup

# top-level bases
add_env DESI  "$DESI"
add_env COSMO "$COSMO"

# science/catalog
add_env LARGEGALAXIES_CAT "$LARGEGALAXIES_CAT"
add_env SKY_TEMPLATE_DIR   "$SKY_TEMPLATE_DIR"
add_env SGA_DIR            "$SGA_DIR"
add_env SGA_DATA_DIR       "$SGA_DATA_DIR"
add_env SGA_HTML_DIR       "$SGA_HTML_DIR"
add_env LEGACY_SURVEY_BASEDIR "$LEGACY_SURVEY_BASEDIR"
add_env GAIA_CAT_DIR       "$GAIA_CAT_DIR"
add_env GAIA_CAT_PREFIX    "$GAIA_CAT_PREFIX"
add_env GAIA_CAT_SCHEME    "$GAIA_CAT_SCHEME"
add_env GAIA_CAT_VER       "$GAIA_CAT_VER"
add_env UNWISE_COADDS_DIR  "$UNWISE_COADDS_DIR"
add_env TYCHO2_KD_DIR      "$TYCHO2_KD_DIR"
add_env PS1CAT_DIR         "$PS1CAT_DIR"
add_env DUST_DIR           "$DUST_DIR"
add_env GALEX_DIR          "$GALEX_DIR"

# dev toggles / path
add_env SGA_DEV        "$SGA_DEV"
add_env LEGACYPIPE_DEV "$LEGACYPIPE_DEV"
add_env PATH_PREPEND   "$PATH_PREPEND"

echo "Launching:"
printf '%s ' \
  "${SRUN[@]}" \
  shifter \
  "${ENVFLAGS[@]}" \
  bash -lc '<RUNCMD>' \
  "SGA2025-mpi" "${args[@]}"
printf '\n'

#python -m mpi4py --mpi-lib-version
#python -m mpi4py --version
#python -m mpi4py.bench helloworld
time "${SRUN[@]}" \
  shifter \
  "${ENVFLAGS[@]}" \
  bash -lc "$RUNCMD" \
  "SGA2025-mpi" "${args[@]}"
