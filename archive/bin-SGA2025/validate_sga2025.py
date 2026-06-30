#!/usr/bin/env python
"""
validate_sga2025.py — Validate SGA2025 production output directories.

Three validation stages are supported via ``--stage``:

coadds (default)
    Checks each data group directory for coadd + ellipse outputs.
    - Presence of *-coadds.isdone and *-ellipse.isdone
    - Correct file set for "full" mode (has *-tractor.fits) vs "coadds only"
    - No unexpected/extraneous files
    - UV (NUV, FUV) and IR (W1-W4) bands always present
    - At least one optical band (g, r, i, z)
    - fitsverify -e -q on every *.fits* file (errors only, not warnings)

htmlplots
    Checks each HTML group directory after ``SGA2025-mpi --htmlplots``.
    - Presence of *-html.isdone and *-html.log (and flags *-html.isfail)
    - Presence of *-montage.png and *-thumb.jpg
    - If *-ellipsemask.png exists (full mode): also checks per-galaxy
      *-sbprofiles.png, *-cog.png, and *-sed.png for every SGANAME found
    - PNG integrity check on all .png files (signature + IEND chunk)

htmlindex
    Checks each HTML group directory after ``SGA2025-mpi --htmlindex``.
    - Presence of {group_name}.html

Usage (single-process):
    python validate_sga2025.py /path/to/dr11-south [--logfile validation.log]
    python validate_sga2025.py /path/to/html/dr11-south --stage htmlplots
    python validate_sga2025.py /path/to/html/dr11-south --stage htmlindex

Usage (MPI, recommended for large releases):
    srun -n 64 python validate_sga2025.py /path/to/dr11-south [--logfile validation.log]
    srun -n 64 python validate_sga2025.py /path/to/html/dr11-south --stage htmlplots

NERSC interactive example:
    salloc -N 1 -C cpu -A desi -t 04:00:00 --qos interactive
    source /dvs_ro/common/software/desi/desi_environment.sh main
    time srun --ntasks=64 python validate_sga2025.py /pscratch/sd/i/ioannis/SGA2025-v1.6/dr11-north --logfile validation-coadds-dr11-north.log
    time srun --ntasks=64 python validate_sga2025.py $SGA_HTML_DIR/dr11-south --stage htmlplots --logfile validation-htmlplots-dr11-south.log

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

# Pattern to extract per-galaxy J-coord from HTML plot filenames
_JNAME_RE = re.compile(r'^SGA2025_(J[\d.]+[+-][\d.]+)-.+\.png$')

# Pattern to recognise HTML checksum files (allowed even if generated later)
_SHA256_RE = re.compile(r'^sga_2025_html_.*\.sha256sum$')


# ---------------------------------------------------------------------------
# PNG integrity check
# ---------------------------------------------------------------------------

PNG_SIG  = b'\x89PNG\r\n\x1a\n'
PNG_IEND = b'\x00\x00\x00\x00IEND\xae\x42\x60\x82'


def check_html_tail(filepath):
    """Return an error string if the HTML file is truncated, else None.

    A write interrupted mid-file will not end with ``</html>``.  Reading
    only the last 20 bytes makes this check negligibly cheap.
    """
    try:
        size = filepath.stat().st_size
        if size == 0:
            return f'HTML empty: {filepath.name}'
        with open(filepath, 'rb') as fh:
            fh.seek(max(0, size - 20), 0)
            tail = fh.read().decode('utf-8', errors='replace').rstrip()
        if not tail.endswith('</html>'):
            return f'HTML truncated (does not end with </html>): {filepath.name}'
    except OSError as exc:
        return f'HTML unreadable [{filepath.name}]: {exc}'
    return None


def check_png_integrity(filepath):
    """Return an error string if the PNG is corrupt/truncated, else None.

    Checks the PNG signature (first 8 bytes) and the IEND chunk (last 12
    bytes).  A file whose write was interrupted will be missing the IEND
    chunk.  No pixel data is read, so this is fast even for large files.
    """
    try:
        size = filepath.stat().st_size
        if size < 20:
            return f'PNG too small ({size} B): {filepath.name}'
        with open(filepath, 'rb') as fh:
            sig = fh.read(8)
            if sig != PNG_SIG:
                return f'PNG bad signature: {filepath.name}'
            fh.seek(-12, 2)
            tail = fh.read(12)
        if tail != PNG_IEND:
            return f'PNG truncated/corrupt (missing IEND): {filepath.name}'
    except OSError as exc:
        return f'PNG unreadable [{filepath.name}]: {exc}'
    return None


# ---------------------------------------------------------------------------
# Per-galaxy name discovery for HTML group directories
# ---------------------------------------------------------------------------

def _find_jnames(dirpath):
    """Return the set of per-galaxy J-coord strings found as plot filenames.

    Scans for ``SGA2025_J...-*.png`` files and extracts the J-coordinate
    portion (e.g. ``J123.4567+12.345``).  The three per-galaxy plot types
    (sbprofiles, cog, sed) all share the same prefix, so this discovers
    which galaxies should have plots without needing the sample catalog.
    """
    jnames = set()
    for f in dirpath.iterdir():
        if not f.is_file():
            continue
        m = _JNAME_RE.match(f.name)
        if m:
            jnames.add(m.group(1))
    return jnames


# ---------------------------------------------------------------------------
# HTML per-group validation
# ---------------------------------------------------------------------------

def _html_valid_files(group, actual_files):
    """Return the complete set of known-valid filenames for an HTML group dir.

    Covers outputs from all three HTML stages (htmlplots, htmlindex,
    checksums) so that the UNEXPECTED scan does not flag files written by a
    stage that ran before (or after) the one being validated.
    """
    g = f'SGA2025_{group}'
    valid = {
        # htmlplots
        f'{g}-html.isdone',
        f'{g}-html.log',
        f'{g}-montage.png',
        f'{g}-thumb.jpg',
        f'{g}-ellipsemask.png',   # present only for non-RESOLVED groups
        # htmlindex
        f'{group}.html',
    }
    # Per-galaxy plots: discover J-coords from whatever .png files already exist
    for fname in actual_files:
        m = _JNAME_RE.match(fname)
        if m:
            jname = m.group(1)
            for ptype in ('sbprofiles', 'cog', 'sed'):
                valid.add(f'SGA2025_{jname}-{ptype}.png')
    return valid


def validate_html_group(dirpath, stage):
    """
    Validate a single HTML group directory for ``--stage htmlplots`` or
    ``--stage htmlindex``.

    Returns
    -------
    problems : list of str
    mode : str  ('full', 'skip_ellipse', or 'htmlindex')
    """
    dirpath = Path(dirpath)
    group   = dirpath.name
    g       = f'SGA2025_{group}'
    problems = []
    actual_files = set(f.name for f in dirpath.iterdir() if f.is_file())

    if stage == 'htmlplots':
        # A .isfail marker means the run failed for this group
        if f'{g}-html.isfail' in actual_files:
            problems.append(f'FAILED: {g}-html.isfail present')

        # Completion marker and log
        for marker in [f'{g}-html.isdone', f'{g}-html.log']:
            if marker not in actual_files:
                problems.append(f'MISSING: {marker}')

        # Group-level images always expected when htmlplots ran
        montage = f'{g}-montage.png'
        thumb   = f'{g}-thumb.jpg'
        ellmask = f'{g}-ellipsemask.png'

        if montage not in actual_files:
            problems.append(f'MISSING: {montage}')
        else:
            err = check_png_integrity(dirpath / montage)
            if err:
                problems.append(err)

        if thumb not in actual_files:
            problems.append(f'MISSING: {thumb}')

        # Determine mode from whether ellipsemask was produced
        full_mode = ellmask in actual_files
        if full_mode:
            err = check_png_integrity(dirpath / ellmask)
            if err:
                problems.append(err)

            # Discover per-galaxy J-coords and check all three plot types
            jnames = _find_jnames(dirpath)
            if not jnames:
                problems.append(
                    'MISSING: no per-galaxy plot files found '
                    '(expected *-sbprofiles.png, *-cog.png, *-sed.png)')
            else:
                for jname in sorted(jnames):
                    for ptype in ['sbprofiles', 'cog', 'sed']:
                        fname = f'SGA2025_{jname}-{ptype}.png'
                        if fname not in actual_files:
                            problems.append(f'MISSING: {fname}')
                        else:
                            err = check_png_integrity(dirpath / fname)
                            if err:
                                problems.append(err)

        mode = 'full' if full_mode else 'skip_ellipse'

    elif stage == 'htmlindex':
        html_file = f'{group}.html'
        if html_file not in actual_files:
            problems.append(f'MISSING: {html_file}')
        else:
            err = check_html_tail(dirpath / html_file)
            if err:
                problems.append(err)
        mode = 'htmlindex'

    else:
        mode = stage

    # UNEXPECTED check — runs for every stage.
    # Build the full valid set from actual file names so per-galaxy entries
    # discovered by _find_jnames are included even when not all three plot
    # types are present (those gaps are already caught by the MISSING checks).
    valid = _html_valid_files(group, actual_files)
    for fname in sorted(actual_files):
        if fname in valid:
            continue
        if _SHA256_RE.match(fname):
            continue  # checksum file written by a separate stage
        if stage == 'htmlplots' and fname == f'{g}-html.isfail':
            continue  # already reported as FAILED above; avoid double entry
        problems.append(f'UNEXPECTED: {fname}')

    return problems, mode


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

def validate_tree(topdir, log, stage='coadds'):
    """Walk the tree, distribute work across MPI ranks, gather on rank 0.

    Parameters
    ----------
    topdir : str or Path
        Region-level directory whose ``{raslice}/{group}/`` subdirectories
        will be validated (data dir for ``coadds``; HTML region dir for
        ``htmlplots`` / ``htmlindex``).
    log : logging.Logger
    stage : {'coadds', 'htmlplots', 'htmlindex'}
    """

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
        if stage == 'coadds':
            problems, mode = validate_group(d)
        else:
            problems, mode = validate_html_group(d, stage)
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
    problem_groups = [(name, probs) for name, probs, _ in results if probs]

    for name, probs in problem_groups:
        for p in probs:
            log.warning(f'{name}: {p}')

    log.info('')
    log.info('=' * 70)
    log.info('VALIDATION SUMMARY')
    log.info('=' * 70)
    log.info(f'  Total groups:       {n_total:6d}')
    if stage == 'coadds':
        n_full   = sum(1 for _, _, m in results if m == 'full')
        n_coadds = sum(1 for _, _, m in results if m == 'coadds')
        log.info(f'  Full mode:          {n_full:6d}')
        log.info(f'  Coadds-only mode:   {n_coadds:6d}')
    elif stage == 'htmlplots':
        n_full = sum(1 for _, _, m in results if m == 'full')
        n_skip = sum(1 for _, _, m in results if m == 'skip_ellipse')
        log.info(f'  Full mode:          {n_full:6d}')
        log.info(f'  Skip-ellipse mode:  {n_skip:6d}')
    log.info(f'  OK:                 {n_total - len(problem_groups):6d}')
    log.info(f'  With problems:      {len(problem_groups):6d}')
    log.info('=' * 70)

    return problem_groups


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Validate SGA2025 production output.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Coadds + ellipse (default)
  python validate_sga2025.py /path/to/dr11-south --logfile validation-coadds.log

  # HTML plots (SGA2025-mpi --htmlplots)
  python validate_sga2025.py /path/to/html/dr11-south --stage htmlplots --logfile validation-htmlplots.log

  # HTML index pages (SGA2025-mpi --htmlindex)
  python validate_sga2025.py /path/to/html/dr11-south --stage htmlindex --logfile validation-htmlindex.log

  # MPI (any stage)
  srun -n 64 python validate_sga2025.py /path/to/html/dr11-south --stage htmlplots
""")
    parser.add_argument('topdir',
                        help='Region-level directory to validate.  For --stage coadds '
                             'this is the data directory (e.g. /path/to/dr11-south); '
                             'for HTML stages it is the HTML region directory '
                             '(e.g. /path/to/html/dr11-south).')
    parser.add_argument('--stage', default='coadds',
                        choices=['coadds', 'htmlplots', 'htmlindex'],
                        help='Validation stage (default: coadds)')
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

        log.info(f'Stage:      {args.stage}')
        log.info(f'Validating: {args.topdir}')
        log.info(f'Log file:   {args.logfile}')
        if _use_mpi:
            log.info(f'MPI ranks:  {_size}')
        log.info('')

    problem_groups = validate_tree(args.topdir, log, stage=args.stage)

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
