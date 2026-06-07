#!/usr/bin/env python
"""
validate_sga2025.py — Validate SGA2025 production output directories.

Usage:
    python validate_sga2025.py /path/to/dr11-south [--logfile validation.log]

Checks each group directory for:
  - Presence of *-coadds.isdone and *-ellipse.isdone
  - Correct set of files for "full" mode (has *-tractor.fits) vs
    "coadds only" mode (no *-tractor.fits)
  - No unexpected/extraneous files
  - UV (NUV, FUV) and IR (W1-W4) bands always present
  - At least one optical band (g, r, i, z)
"""

import os
import re
import sys
import argparse
import logging
from pathlib import Path
from collections import defaultdict

# ---------------------------------------------------------------------------
# Expected file patterns
# ---------------------------------------------------------------------------

OPTICAL_BANDS = ['g', 'r', 'i', 'z']
UV_BANDS      = ['NUV', 'FUV']
IR_BANDS      = ['W1', 'W2', 'W3', 'W4']
ALL_BANDS     = OPTICAL_BANDS + UV_BANDS + IR_BANDS

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

    # Common fixed files
    for tmpl in COMMON_FIXED:
        expected.add(tmpl.format(g=g))

    # Common JPGs
    for tmpl in COMMON_JPGS:
        expected.add(tmpl.format(g=g))

    if mode == 'full':
        # Full-mode fixed files
        for tmpl in FULL_FIXED + FULL_FIXED_NONBAND:
            expected.add(tmpl.format(g=g))

        # Full-mode JPGs
        for tmpl in FULL_JPGS:
            expected.add(tmpl.format(g=g))

        # Per-band files for all bands
        bands = optical_bands_present + UV_BANDS + IR_BANDS
        for b in bands:
            for tmpl in FULL_BAND_TEMPLATES:
                expected.add(tmpl.format(g=g, b=b))

    else:  # coadds-only
        for tmpl in COADDS_FIXED_NONBAND:
            expected.add(tmpl.format(g=g))

        bands = optical_bands_present + UV_BANDS + IR_BANDS
        for b in bands:
            for tmpl in COADDS_BAND_TEMPLATES:
                expected.add(tmpl.format(g=g, b=b))

    return expected


# Pattern to recognize ellipse files: SGA2025_Jxxx.xxx[+-]xxx.xxx-ellipse-{type}.fits
ELLIPSE_RE = re.compile(
    r'^SGA2025_J[\d.]+[+-][\d.]+-ellipse-(griz|galex|unwise)\.fits$'
)


def validate_group(dirpath):
    """
    Validate a single group directory.

    Returns
    -------
    list of str
        List of problem descriptions, empty if all OK.
    """
    dirpath = Path(dirpath)
    group = dirpath.name  # e.g. '33398m2373'
    g = f'SGA2025_{group}'

    problems = []
    actual_files = set(f.name for f in dirpath.iterdir() if f.is_file())

    # Determine mode
    tractor = f'{g}-tractor.fits'
    mode = 'full' if tractor in actual_files else 'coadds'

    # Check isdone files first
    for isdone in [f'{g}-coadds.isdone', f'{g}-ellipse.isdone']:
        if isdone not in actual_files:
            problems.append(f'MISSING isdone: {isdone}')

    # Determine which optical bands are present by looking at image files
    if mode == 'full':
        optical_present = [b for b in OPTICAL_BANDS
                           if f'{g}-image-{b}.fits.fz' in actual_files]
    else:
        optical_present = [b for b in OPTICAL_BANDS
                           if f'{g}-image-{b}.fits' in actual_files]

    if len(optical_present) == 0:
        problems.append('MISSING: no optical band images found (expected at least one of g,r,i,z)')

    # Check required UV and IR bands
    for b in UV_BANDS + IR_BANDS:
        if mode == 'full':
            img = f'{g}-image-{b}.fits.fz'
        else:
            img = f'{g}-image-{b}.fits'
        if img not in actual_files:
            problems.append(f'MISSING band image: {img}')

    # Build full expected set and compare
    exp = expected_files(group, optical_present, mode=mode)

    # Separate out ellipse files from actual (they follow a different pattern)
    ellipse_files = {f for f in actual_files if ELLIPSE_RE.match(f)}
    non_ellipse_actual = actual_files - ellipse_files

    missing = exp - non_ellipse_actual
    extra   = non_ellipse_actual - exp

    for f in sorted(missing):
        problems.append(f'MISSING: {f}')

    for f in sorted(extra):
        problems.append(f'UNEXPECTED: {f}')

    # Validate ellipse files: must match the pattern, nothing else
    for f in ellipse_files:
        if not ELLIPSE_RE.match(f):
            problems.append(f'UNEXPECTED ellipse file: {f}')

    return problems, mode


def validate_tree(topdir, log):
    """Walk the tree and validate every group directory."""

    topdir = Path(topdir)
    n_total = 0
    n_ok = 0
    n_problems = 0
    n_full = 0
    n_coadds = 0
    problem_groups = []

    # Group dirs are two levels down: topdir/NNN/group
    group_dirs = sorted(topdir.glob('*/*'))
    group_dirs = [d for d in group_dirs if d.is_dir()]

    log.info(f'Found {len(group_dirs)} group directories under {topdir}')
    log.info('')

    for i, d in enumerate(group_dirs, 1):
        if i % 10000 == 0:
            log.info(f'  ... {i}/{len(group_dirs)} checked, {n_problems} problems so far')

        problems, mode = validate_group(d)
        n_total += 1
        if mode == 'full':
            n_full += 1
        else:
            n_coadds += 1

        if problems:
            n_problems += 1
            problem_groups.append((d.name, problems))
            for p in problems:
                log.warning(f'{d.name}: {p}')
        else:
            n_ok += 1

    log.info('')
    log.info('=' * 70)
    log.info('VALIDATION SUMMARY')
    log.info('=' * 70)
    log.info(f'  Total groups:       {n_total:6d}')
    log.info(f'  Full mode:          {n_full:6d}')
    log.info(f'  Coadds-only mode:   {n_coadds:6d}')
    log.info(f'  OK:                 {n_ok:6d}')
    log.info(f'  With problems:      {n_problems:6d}')
    log.info('=' * 70)

    return problem_groups


def main():
    parser = argparse.ArgumentParser(description='Validate SGA2025 production output.')
    parser.add_argument('topdir', help='Top-level directory (e.g. /path/to/dr11-south)')
    parser.add_argument('--logfile', default='validation.log',
                        help='Output log file (default: validation.log)')
    args = parser.parse_args()

    # Set up logging to both stdout and file
    log = logging.getLogger('validate')
    log.setLevel(logging.DEBUG)
    fmt = logging.Formatter('%(message)s')

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    log.addHandler(ch)

    fh = logging.FileHandler(args.logfile, mode='w')
    fh.setFormatter(fmt)
    log.addHandler(fh)

    log.info(f'Validating: {args.topdir}')
    log.info(f'Log file:   {args.logfile}')
    log.info('')

    problem_groups = validate_tree(args.topdir, log)

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
