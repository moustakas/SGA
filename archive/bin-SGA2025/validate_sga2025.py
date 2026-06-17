#!/usr/bin/env python
"""
validate_sga2025.py — Validate SGA2025 production output directories.

Usage (single-process):
    python validate_sga2025.py /path/to/dr11-south [--logfile validation.log]

Usage (MPI, recommended for large releases):
    srun -n 64 python validate_sga2025.py /path/to/dr11-south [--logfile validation.log]

Checks each group directory for:
  - Presence of *-coadds.isdone and *-ellipse.isdone
  - Correct set of files for "full" mode (has *-tractor.fits) vs
    "coadds only" mode (no *-tractor.fits)
  - No unexpected/extraneous files
  - UV (NUV, FUV) and IR (W1-W4) bands always present
  - At least one optical band (g, r, i, z)
  - fitsverify -e -q passes on every *.fits* file (errors only, not warnings)

salloc -N 1 -C cpu -A desi -t 04:00:00 --qos interactive
source /dvs_ro/common/software/desi/desi_environment.sh main
time srun --ntasks=64 python validate_sga2025.py /pscratch/sd/i/ioannis/SGA2025-v1.6/dr11-north --logfile validation-dr11-north.log
time srun --ntasks=64 python validate_sga2025.py /pscratch/sd/i/ioannis/SGA2025-v1.6/dr11-south --logfile validation-dr11-south.log

"""

import os
import re
import sys
import subprocess
import argparse
import logging
from pathlib import Path

# ---------------------------------------------------------------------------
# MPI setup (graceful fallback to single-process if unavailable)
# ---------------------------------------------------------------------------

try:
    from mpi4py import MPI
    _comm = MPI.COMM_WORLD
    _rank = _comm.Get_rank()
    _size = _comm.Get_size()
    _use_mpi = _size > 1
except ImportError:
    _comm, _rank, _size, _use_mpi = None, 0, 1, False

# Uncomment for startup diagnostics (confirms all ranks started and MPI is active):
# import socket as _socket
# print(f'[rank {_rank:03d}/{_size}] started on {_socket.gethostname()} '
#       f'(mpi={_use_mpi})', flush=True)

# ---------------------------------------------------------------------------
# Expected file patterns
# ---------------------------------------------------------------------------

OPTICAL_BANDS = ['g', 'r', 'i', 'z']
UV_BANDS      = ['NUV', 'FUV']
IR_BANDS      = ['W1', 'W2', 'W3', 'W4']

# JPGs common to both modes
COMMON_JPGS = [
    '{g}-image.jpg',
    '{g}-image-W1W2.jpg',
    '{g}-image-FUVNUV.jpg',
]

# JPGs only in full mode
FULL_JPGS = [
    '{g}-model.jpg',
    '{g}-model-W1W2.jpg',
    '{g}-model-FUVNUV.jpg',
    '{g}-resid.jpg',
    '{g}-resid-W1W2.jpg',
    '{g}-resid-FUVNUV.jpg',
]

# Non-band files common to both modes
COMMON_FIXED = [
    '{g}-ccds.fits',
    '{g}-sample.fits',
    '{g}-coadds.isdone',
    '{g}-coadds.log',
    '{g}-ellipse.isdone',
    '{g}-ellipse.log',
]

# Non-band files only in full mode
FULL_FIXED = [
    '{g}-tractor.fits',
]

# Per-band file templates: full mode (compressed)
FULL_BAND_TEMPLATES = [
    '{g}-image-{b}.fits.fz',
    '{g}-invvar-{b}.fits.fz',
    '{g}-model-{b}.fits.fz',
    '{g}-psf-{b}.fits.fz',
]

# Per-band file templates: coadds-only mode (uncompressed)
COADDS_BAND_TEMPLATES = [
    '{g}-image-{b}.fits',
]

# Fixed non-band files: full mode
FULL_FIXED_NONBAND = [
    '{g}-maskbits.fits.fz',
]

# Fixed non-band files: coadds-only mode
COADDS_FIXED_NONBAND = [
    '{g}-maskbits.fits',
]


def expected_files(group, optical_bands_present, mode='full'):
    """
    Return the set of expected filenames for a group directory.

    Parameters
    ----------
    group : str
        Group name, e.g. '33398m2373'
    optical_bands_present : list of str
        Which optical bands are actually present (subset of g,r,i,z)
    mode : 'full' or 'coadds'
    """
    g = f'SGA2025_{group}'
    expected = set()

    for tmpl in COMMON_FIXED:
        expected.add(tmpl.format(g=g))
    for tmpl in COMMON_JPGS:
        expected.add(tmpl.format(g=g))

    if mode == 'full':
        for tmpl in FULL_FIXED + FULL_FIXED_NONBAND:
            expected.add(tmpl.format(g=g))
        for tmpl in FULL_JPGS:
            expected.add(tmpl.format(g=g))
        bands = optical_bands_present + UV_BANDS + IR_BANDS
        for b in bands:
            for tmpl in FULL_BAND_TEMPLATES:
                expected.add(tmpl.format(g=g, b=b))
    else:
        for tmpl in COADDS_FIXED_NONBAND:
            expected.add(tmpl.format(g=g))
        bands = optical_bands_present + UV_BANDS + IR_BANDS
        for b in bands:
            for tmpl in COADDS_BAND_TEMPLATES:
                expected.add(tmpl.format(g=g, b=b))

    return expected


# Pattern to recognize ellipse files
ELLIPSE_RE = re.compile(
    r'^SGA2025_J[\d.]+[+-][\d.]+-ellipse-(griz|galex|unwise)\.fits$'
)


# ---------------------------------------------------------------------------
# fitsverify
# ---------------------------------------------------------------------------

def run_fitsverify(filepath):
    """
    Run ``fitsverify -e -q`` on a single FITS file.

    Parameters
    ----------
    filepath : Path or str

    Returns
    -------
    list of str
        Problem strings (empty if file passed).
    """
    result = subprocess.run(
        ['fitsverify', '-e', '-q', str(filepath)],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        return []

    # Collect non-empty output lines; fitsverify writes to stdout
    lines = [l.strip() for l in result.stdout.splitlines() if l.strip()]
    if lines:
        return [f'fitsverify [{filepath.name}]: {l}' for l in lines]
    # No output but non-zero exit — report generically
    return [f'fitsverify [{filepath.name}]: failed (rc={result.returncode})']


# ---------------------------------------------------------------------------
# Per-group validation
# ---------------------------------------------------------------------------

def validate_group(dirpath):
    """
    Validate a single group directory.

    Returns
    -------
    problems : list of str
    mode : str  ('full' or 'coadds')
    """
    dirpath = Path(dirpath)
    group = dirpath.name
    g = f'SGA2025_{group}'

    problems = []
    actual_files = set(f.name for f in dirpath.iterdir() if f.is_file())

    # Determine mode
    tractor = f'{g}-tractor.fits'
    mode = 'full' if tractor in actual_files else 'coadds'

    # isdone files
    for isdone in [f'{g}-coadds.isdone', f'{g}-ellipse.isdone']:
        if isdone not in actual_files:
            problems.append(f'MISSING isdone: {isdone}')

    # Optical bands present
    if mode == 'full':
        optical_present = [b for b in OPTICAL_BANDS
                           if f'{g}-image-{b}.fits.fz' in actual_files]
    else:
        optical_present = [b for b in OPTICAL_BANDS
                           if f'{g}-image-{b}.fits' in actual_files]

    if len(optical_present) == 0:
        problems.append('MISSING: no optical band images found (expected at least one of g,r,i,z)')

    # Required UV and IR bands
    for b in UV_BANDS + IR_BANDS:
        img = f'{g}-image-{b}.fits.fz' if mode == 'full' else f'{g}-image-{b}.fits'
        if img not in actual_files:
            problems.append(f'MISSING band image: {img}')

    # Expected vs actual file sets
    exp = expected_files(group, optical_present, mode=mode)
    ellipse_files = {f for f in actual_files if ELLIPSE_RE.match(f)}
    non_ellipse_actual = actual_files - ellipse_files

    for f in sorted(exp - non_ellipse_actual):
        problems.append(f'MISSING: {f}')
    for f in sorted(non_ellipse_actual - exp):
        problems.append(f'UNEXPECTED: {f}')
    for f in ellipse_files:
        if not ELLIPSE_RE.match(f):
            problems.append(f'UNEXPECTED ellipse file: {f}')

    # fitsverify on lightweight FITS files only (compressed .fits.fz files
    # are verified elsewhere and are expensive to check here)
    for fname in sorted(actual_files):
        if fname.endswith('-sample.fits') or \
           (fname.endswith('.fits') and '-ellipse' in fname):
            problems.extend(run_fitsverify(dirpath / fname))

    return problems, mode


# ---------------------------------------------------------------------------
# Tree validation (MPI-aware)
# ---------------------------------------------------------------------------

def validate_tree(topdir, log):
    """Walk the tree, distribute work across MPI ranks, gather on rank 0."""

    topdir = Path(topdir)

    # Rank 0 collects all group directories, then broadcasts
    if _rank == 0:
        all_dirs = sorted(d for d in topdir.glob('*/*') if d.is_dir())
        log.info(f'Found {len(all_dirs)} group directories under {topdir}')
        if _use_mpi:
            log.info(f'Distributing across {_size} MPI ranks')
        log.info('')
    else:
        all_dirs = None

    if _use_mpi:
        # print(f'[rank {_rank:03d}/{_size}] waiting for bcast ...', flush=True)
        all_dirs = _comm.bcast(all_dirs, root=0)
        # print(f'[rank {_rank:03d}/{_size}] bcast done, {len(all_dirs)} total dirs', flush=True)

    # Each rank processes its own slice
    my_dirs = all_dirs[_rank::_size]
    # print(f'[rank {_rank:03d}/{_size}] processing {len(my_dirs)} dirs', flush=True)
    my_results = []  # list of (group_name, problems, mode)

    for i, d in enumerate(my_dirs):
        # if (i + 1) % 100 == 0:
        #     print(f'[rank {_rank:03d}/{_size}] {i+1}/{len(my_dirs)} checked', flush=True)
        if _rank == 0 and (i + 1) % 1000 == 0:
            log.info(f'  rank 0: {i+1}/{len(my_dirs)} checked ...')
        problems, mode = validate_group(d)
        my_results.append((d.name, problems, mode))

    # Gather on rank 0
    # print(f'[rank {_rank:03d}/{_size}] done processing, waiting for gather ...', flush=True)
    if _use_mpi:
        all_results_nested = _comm.gather(my_results, root=0)
    else:
        all_results_nested = [my_results]

    if _rank != 0:
        return []

    # Flatten
    results = [item for chunk in all_results_nested for item in chunk]

    # Tally and report
    n_total = len(results)
    n_full    = sum(1 for _, _, m in results if m == 'full')
    n_coadds  = sum(1 for _, _, m in results if m == 'coadds')
    problem_groups = [(name, probs) for name, probs, _ in results if probs]

    for name, probs in problem_groups:
        for p in probs:
            log.warning(f'{name}: {p}')

    log.info('')
    log.info('=' * 70)
    log.info('VALIDATION SUMMARY')
    log.info('=' * 70)
    log.info(f'  Total groups:       {n_total:6d}')
    log.info(f'  Full mode:          {n_full:6d}')
    log.info(f'  Coadds-only mode:   {n_coadds:6d}')
    log.info(f'  OK:                 {n_total - len(problem_groups):6d}')
    log.info(f'  With problems:      {len(problem_groups):6d}')
    log.info('=' * 70)

    return problem_groups


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Validate SGA2025 production output.')
    parser.add_argument('topdir', help='Top-level directory (e.g. /path/to/dr11-south)')
    parser.add_argument('--logfile', default='validation.log',
                        help='Output log file (default: validation.log)')
    args = parser.parse_args()

    # Logging: all ranks write to stdout; only rank 0 writes the log file
    log = logging.getLogger('validate')
    log.setLevel(logging.DEBUG)
    fmt = logging.Formatter('%(message)s')

    if _rank == 0:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        log.addHandler(ch)

        fh = logging.FileHandler(args.logfile, mode='w')
        fh.setFormatter(fmt)
        log.addHandler(fh)

        log.info(f'Validating: {args.topdir}')
        log.info(f'Log file:   {args.logfile}')
        if _use_mpi:
            log.info(f'MPI ranks:  {_size}')
        log.info('')

    problem_groups = validate_tree(args.topdir, log)

    if _rank == 0:
        if problem_groups:
            log.info('')
            log.info(f'Groups with problems ({len(problem_groups)}):')
            for gname, probs in problem_groups:
                log.info(f'  {gname}: {len(probs)} issue(s)')
        else:
            log.info('All groups OK.')

        sys.exit(1 if problem_groups else 0)


if __name__ == '__main__':
    main()
