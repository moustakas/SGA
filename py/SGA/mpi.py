"""
=======
SGA.mpi
=======

Code to deal with the MPI portion of the pipeline.

"""
import os, time
import numpy as np

from SGA.logger import log


def mpi_args():
    """Define and parse the command-line interface for the
    ``bin/SGA2025-mpi`` processing driver.

    Covers pipeline-stage selection (``--coadds``, ``--customsky``,
    ``--ellipse``, ``--htmlplots``, ``--htmlindex``,
    ``--build-refcat``, ``--build-catalog``, ``--count-masked-pixels``,
    ``--parse-tractor-logs``), sample filtering (``--first``/``--last``,
    ``--galaxylist``, ``--mindiam``/``--maxdiam``,
    ``--minmult``/``--maxmult``, ``--region``, ``--lvd``,
    ``--wisesize``, ``--public-release``, ``--test-bricks``),
    processing/fitting options (pixel scales, ellipse-fitting
    Monte Carlo parameters, geometry-selection overrides
    ``--fixgeo``/``--tractorgeo``, source-masking flags), and
    run-control flags (``--force``, ``--clobber``, ``--debug``,
    ``--verbose``, ``--mp``, ``--ngpu``).

    Returns
    -------
    :class:`argparse.Namespace`
        Parsed command-line arguments; every attribute name matches
        the corresponding flag's ``dest`` (equal to the flag name with
        dashes replaced by underscores, unless a flag explicitly sets
        ``dest=`` to invert its sense, e.g. ``--no-unwise`` ->
        ``args.unwise``).

    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--mp', default=1, type=int, help='number of multiprocessing processes per MPI rank.')

    parser.add_argument('--first', type=int, help='Index of first object to process.')
    parser.add_argument('--last', type=int, help='Index of last object to process.')
    parser.add_argument('--galaxylist', type=str, default=None, help='Comma-separated list of galaxy names to process.')

    parser.add_argument('--mindiam', default=0., type=float, help='Minimum diameter (arcmin).')
    parser.add_argument('--maxdiam', default=1e3, type=float, help='Maximum diameter (arcmin).')
    parser.add_argument('--minmult', default=None, type=int, help='Only read primary groups with minmult or more members.')
    parser.add_argument('--maxmult', default=None, type=int, help='Only read primary groups with up to maxmult members.')

    parser.add_argument('--coadds', action='store_true', help='Build the large-galaxy coadds.')
    parser.add_argument('--customsky', action='store_true', help='Build the largest large-galaxy coadds with custom sky-subtraction.')
    parser.add_argument('--just-coadds', action='store_true', help='Just build the coadds and return (using --early-coadds in runbrick.py.')

    parser.add_argument('--ellipse', action='store_true', help='Do the ellipse fitting.')
    parser.add_argument('--htmlplots', action='store_true', help='Build the pipeline figures.')
    parser.add_argument('--htmlindex', action='store_true', help='Build HTML index.html page.')

    parser.add_argument('--htmlhome', default='index.html', type=str, help='Home page file name (use in tandem with --htmlindex).')
    parser.add_argument('--html-raslices', action='store_true',
                        help='Organize HTML pages by RA slice (use in tandem with --htmlindex).')

    parser.add_argument('--pixscale', default=0.262, type=float, help='pixel scale (arcsec/pix).')
    parser.add_argument('--unwise-pixscale', default=2.75, type=float, help='unWISE pixel scale (arcsec/pix).')
    parser.add_argument('--galex-pixscale', default=1.5, type=float, help='GALEX pixel scale (arcsec/pix).')

    parser.add_argument('--nsigma', default=None, type=int, help='detection sigma')
    parser.add_argument('--nmonte', default=100, type=int, help='Number of Monte Carlo draws (ellipsefit_multiband)')
    parser.add_argument('--seed', default=42, type=int, help='Random seed for Monte Carlo draws (ellipsefit_multiband)')
    parser.add_argument('--region', default='dr11-south', choices=['dr9-north', 'dr11-north', 'dr11-south'], type=str, help='Region to analyze.')

    parser.add_argument('--version', type=str, default=None, help='SGA version.')
    parser.add_argument('--datadir', default=None, type=str, help='Override $SGA_DATA_DIR environment variable')
    parser.add_argument('--htmldir', default=None, type=str, help='Override $SGA_HTML_DIR environment variable')

    parser.add_argument('--analyze', action='store_true', help='Analyze the submissions to the queue (in tandem with --coadds).')
    parser.add_argument('--no-groups', action='store_true', help='Ignore angular group parameters; fit individual galaxies (with --coadds).')
    parser.add_argument('--test-bricks', action='store_true', help='Read the sample of test bricks.')
    parser.add_argument('--no-iterative', action='store_true', help='Turn off iterative source detection.')
    parser.add_argument('--use-segmentation', action='store_false', dest='no_segmentation', help='Turn on Tractor segmentation during fitblobs.')
    parser.add_argument('--noradweight', dest='use_radial_weight', action='store_false',
                        help='No radial weighting when determining moment geometry.')
    parser.add_argument('--momentpos', dest='use_tractor_position', action='store_false',
                        help='Use the light-weighted (not Tractor) position during ellipse-fitting.')
    parser.add_argument('--fixgeo', action='store_true', help='Use fixed ellipse geometry (irrespective of if the ELLIPSEMODE bit is set.')
    parser.add_argument('--tractorgeo', action='store_true', help='Use the Tractor geometry (irrespective of if the ELLIPSEMODE bit is set.')
    parser.add_argument('--ignore-galaxy-sources', action='store_true', help='Used for ellipse-fitting for very large mosaics, e.g., M33.')

    parser.add_argument('--no-unwise', action='store_false', dest='unwise', help='Do not build unWISE coadds or do forced unWISE photometry.')
    parser.add_argument('--no-galex', action='store_false', dest='galex', help='Do not build GALEX coadds or do forced GALEX photometry.')
    parser.add_argument('--no-cleanup', action='store_false', dest='cleanup', help='Do not clean up legacypipe files after coadds.')
    parser.add_argument('--large-bricks-first', action='store_false', dest='small_bricks_first', help='Process large mosaics first.')

    parser.add_argument('--diameter-file', default=None, type=str, help='Write a diameter file for use with generate_sga_jobs.sh')
    parser.add_argument('--galaxylist-file', default=None, type=str, help='Write a galaxy list file for use with generate_sga_jobs.sh')
    parser.add_argument('--parse-tractor-logs', action='store_true', help='Parse existing Tractor logs.')

    #parser.add_argument('--ubercal-sky', action='store_true', help='Build the largest large-galaxy coadds with custom (ubercal) sky-subtraction.')
    parser.add_argument('--redo-failures', action='store_true', help='Select (and re-queue) just failures.')
    parser.add_argument('--skip-tractor', action='store_true', help='With --coadds or --ellipse, do not run Tractor.')
    parser.add_argument('--fit-on-coadds', action='store_true', help='Fit on coadds.')

    parser.add_argument('--bright-masking', action='store_true', help='Mask bright sources not on the blob being fitted.')
    parser.add_argument('--galaxy-masking', action='store_true', help='Custom segmentation for large galaxies.')

    parser.add_argument('--force', action='store_true', help='Use with --coadds; ignore previous pickle files.')
    parser.add_argument('--count', action='store_true', help='Count how many objects are left to analyze and then return.')
    parser.add_argument('--debug', action='store_true', help='Log to STDOUT and build debugging plots.')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output.')
    parser.add_argument('--clobber', action='store_true', help='Overwrite existing files.')
    parser.add_argument('--qaplot', action='store_true', help='Build some QA plots (for testing).')

    parser.add_argument('--lvd', action='store_true', help='Read the parent LVD sample.')
    parser.add_argument('--public-release', action='store_true', help='RRead the public (final) sample and generate public (final) QA.')
    parser.add_argument('--wisesize', action='store_true', help='Read the wisesize parent sample.')

    parser.add_argument('--ngpu', default=1, type=int, help='Number of GPUs to use.')
    parser.add_argument('--threads-per-gpu', default=8, type=int, help='Max threads per GPU - CPU will fill remaining threads.')

    parser.add_argument('--build-refcat', action='store_true', help='Build the legacypipe reference catalog.')
    parser.add_argument('--build-catalog', action='store_true', help='Build the final catalog.')
    parser.add_argument('--count-masked-pixels', action='store_true', help='Count masked pixels.')
    args = parser.parse_args()

    return args


def weighted_partition(weights, n):
    '''Partition `weights` into `n` groups with approximately the same
    sum of weights, via a greedy largest-first assignment (each item,
    from heaviest to lightest, goes to whichever group currently has
    the smallest running sum).

    Parameters
    ----------
    weights : array-like
        Weight of each item to partition.
    n : :class:`int`
        Number of groups.

    Returns
    -------
    :class:`list`
        List of `n` lists, each containing the integer indices (into
        `weights`) of the items assigned to that group.

    Notes
    -----
    Compared to `dist_discrete_all`, this function allows
    non-contiguous items to be grouped together, which allows better
    load balancing.

    '''
    #- sumweights will track the sum of the weights that have been assigned
    #- to each group so far
    sumweights = np.zeros(n, dtype=float)

    #- Initialize list of lists of indices for each group
    groups = list()
    for i in range(n):
        groups.append(list())

    #- Assign items from highest weight to lowest weight, always assigning
    #- to whichever group currently has the fewest weights
    weights = np.asarray(weights)
    for i in np.argsort(-weights):
        j = np.argmin(sumweights)
        groups[j].append(i)
        sumweights[j] += weights[i]

    assert len(groups) == n

    return groups


def distribute_work(diameter, itodo=None, size=1, p=2.0, verbose=False,
                    small_bricks_first=True):
    """Partition tasks into `size` buckets with ~equal total weight
    (via greedy longest-processing-time-first / LPT scheduling), then
    sort each bucket so smaller (or larger) bricks are processed
    first.

    Parameters
    ----------
    diameter : array-like
        Brick diameters (arcmin) for all items.
    itodo : array-like of :class:`int`, optional
        Indices to schedule. Defaults to all items.
    size : :class:`int`
        Number of MPI ranks.
    p : :class:`float`
        Exponent for the weight model: ``weight = diameter**p``.
    verbose : :class:`bool`
        If True, print (or, for the ``size == 1`` branch, log) a
        load-balance summary.
    small_bricks_first : :class:`bool`
        If True, sort each rank's indices ascending by diameter
        (smallest first); if False, descending (largest first).

    Returns
    -------
    todo_indices : :class:`list` of :class:`numpy.ndarray`
        One array of indices per rank, sorted per `small_bricks_first`.
    loads : :class:`numpy.ndarray`
        Total weight per rank.

    Notes
    -----
    The ``size == 1`` branch logs its load-balance summary via
    :data:`SGA.logger.log` (``log.info``), while the general
    (``size > 1``) branch uses a bare ``print()`` instead -- an
    inconsistency in how `verbose` output is routed depending on
    `size`.

    """
    import heapq

    if itodo is None:
        itodo = np.arange(len(diameter), dtype=int)
    else:
        itodo = np.asarray(itodo, dtype=int)

    size = int(size)
    if size < 1:
        raise ValueError("size must be >= 1")

    # Trivial or empty cases
    if len(itodo) == 0:
        return [np.array([], dtype=int) for _ in range(size)], np.zeros(size, dtype=float)
    if size == 1:
        br = np.array(itodo, dtype=int)
        if small_bricks_first:
            srt = np.argsort(np.asarray(diameter)[br])
        else:
            srt = np.argsort(np.asarray(diameter)[br])[::-1]
        loads = np.array([np.sum(np.power(np.asarray(diameter)[br], p))] + [0.0]*(size-1), dtype=float)
        if verbose:
            tot = loads.sum()
            target = tot / size
            log.info(f"[scheduler] p={p:g}, total={tot:.6g}, per-rank target={target:.6g}, "
                     f"max load={loads.max():.6g}, rel_imbalance={(loads.max()-target)/target if target>0 else 0.0:.2%}")
        return [br[srt]] + [np.array([], dtype=int) for _ in range(size-1)], loads

    diam = np.asarray(diameter, dtype=float)
    dsub = diam[itodo]
    w = np.power(dsub, p)

    # Stable sort by DESC weight; tie-break on index for determinism
    order = np.lexsort((itodo, -w))
    jobs = itodo[order]
    jw = w[order]

    # Greedy LPT: assign next heaviest job to least-loaded rank
    heap = [(0.0, r) for r in range(size)]  # (load, rank)
    heapq.heapify(heap)
    buckets = [[] for _ in range(size)]
    loads = np.zeros(size, dtype=float)

    for idx, wt in zip(jobs, jw):
        load, r = heapq.heappop(heap)
        buckets[r].append(idx)
        load += wt
        loads[r] = load
        heapq.heappush(heap, (load, r))

    # Within each rank, do large bricks first
    todo_indices = []
    for r in range(size):
        br = np.array(buckets[r], dtype=int)
        if br.size:
            if small_bricks_first:
                srt = np.argsort(diam[br])
            else:
                srt = np.argsort(diam[br])[::-1]
            br = br[srt]
        todo_indices.append(br)

    if verbose:
        tot = loads.sum()
        target = tot / size
        imbalance = (loads.max() - target) / target if target > 0 else 0.0
        print(f"[scheduler] p={p:g}, total={tot:.6g}, per-rank target={target:.6g}, "
              f"max load={loads.max():.6g}, rel_imbalance={imbalance:.2%}")

    return todo_indices, loads


def parse_tractor_log(logfile):
    """Extract key metrics from a Tractor (``legacypipe``/coadds) log
    file, by regex-parsing its stdout for group name, mosaic width,
    CCD/source/blob counts, checkpoint recovery, and runtime.

    If some fields (``nccd``, ``nsources``, ``nblob``) are missing
    from the main log after a checkpoint recovery, falls back to
    reading the rotated logs (``-coadds_log.0``, ``.1``, ...) in order
    until they are found.

    Parameters
    ----------
    logfile : :class:`str` or :class:`pathlib.Path`
        Path to the log file.

    Returns
    -------
    :class:`dict`
        Parsed metrics: ``group_name``, ``width``, ``nccd``, ``nblob``,
        ``nsources``, ``runtime`` (minutes), ``nattempts``,
        ``ncheckpoint``. If `logfile` does not exist, all numeric
        fields are set to 0/0.0 and ``group_name`` is recovered from
        the path itself (via a ``{ra}{p,m}{dec}`` regex match), if
        possible.

    Notes
    -----
    A field left unmatched by any regex (e.g. because the
    corresponding log line was never written) remains ``None`` in the
    returned dict rather than a numeric placeholder -- callers should
    check for `None`, not just falsiness, since 0 is a valid count.

    """
    import re
    import sys
    from pathlib import Path
    import glob


    def count_attempts(logfile):
        """Count the number of fitting attempts by finding rotated log
        files.

        Pattern: ``base-coadds.log`` with rotations as
        ``base-coadds_log.0``, ``base-coadds_log.1``, etc. The ``.0``
        file is the first/original attempt.

        Parameters
        ----------
        logfile : :class:`str` or :class:`pathlib.Path`
            Path to the (possibly rotated) log file.

        Returns
        -------
        :class:`int`
            Number of attempts (1 if only the base log exists, 2+ if
            rotated logs also exist).

        Notes
        -----
        The glob pattern built here (``base_pattern``) is identical
        for the ``'-coadds_log'`` and fallback branches (both produce
        ``logstr + '.*'``), making the ``elif logstr.endswith(
        '-coadds_log'):`` branch redundant with the final ``else:``.

        """
        logfile = Path(logfile)
        logstr = str(logfile)

        # Handle rotation pattern: -coadds.log → -coadds_log.*
        if logstr.endswith('-coadds.log'):
            base_pattern = logstr.replace('-coadds.log', '-coadds.log.*')
        elif logstr.endswith('-coadds_log'):
            base_pattern = logstr + '.*'
        else:
            base_pattern = logstr + '.*'

        # Find all rotated versions
        rotated = glob.glob(base_pattern)

        # Total attempts = main log + rotated logs
        return 1 + len(rotated)


    logfile = Path(logfile)

    result = {
        'group_name': None,
        'width': None,
        'nccd': None,
        'nblob': None,
        'nsources': None,
        'runtime': None,
        'nattempts': None,
        'ncheckpoint': None
    }

    # Handle case where logfile doesn't exist
    if not logfile.exists():
        match = re.search(r'/(\d+[pm]\d+)/', str(logfile))
        if match:
            result['group_name'] = match.group(1)
        result['width'] = 0
        result['nccd'] = 0
        result['nblob'] = 0
        result['nsources'] = 0
        result['runtime'] = 0.0
        result['nattempts'] = 0
        result['ncheckpoint'] = 0
        return result

    # Count attempts
    result['nattempts'] = count_attempts(logfile)

    # Parse main log
    with open(logfile) as f:
        for line in f:
            if 'outdir=' in line and result['group_name'] is None:
                m = re.search(r'--outdir=.+?/(\d+[pm]\d+)', line)
                if m:
                    result['group_name'] = m.group(1)

            if '--width=' in line and result['width'] is None:
                m = re.search(r'--width=(\d+)', line)
                if m:
                    result['width'] = int(m.group(1))

            if 'Keeping' in line and 'CCDs' in line:
                m = re.search(r'Keeping (\d+) CCDs', line)
                if m:
                    result['nccd'] = int(m.group(1))

            if 'Keeping' in line and 'checkpointed results' in line:
                m = re.search(r'Keeping (\d+) of \d+ checkpointed results', line)
                if m:
                    result['ncheckpoint'] = int(m.group(1))

            if 'Sources detected:' in line:
                m = re.search(r'Sources detected: (\d+) in (\d+) blobs', line)
                if m:
                    result['nsources'] = int(m.group(1))
                    result['nblob'] = int(m.group(2))

            if line.startswith('Total runtime:'):
                m = re.search(r'Total runtime: ([\d.]+)', line)
                if m:
                    runtime_sec = float(m.group(1))
                    result['runtime'] = runtime_sec / 60.0

    # If checkpoint recovery, some fields may be missing - check rotated logs
    if result['ncheckpoint'] is not None and result['ncheckpoint'] > 0:
        # Try all rotated logs in order: _log.0, _log.1, _log.2, etc.
        for i in range(10):  # Check up to 10 rotations
            fallback_log = Path(str(logfile).replace('-coadds.log', f'-coadds_log.{i}'))
            if not fallback_log.exists():
                break

            with open(fallback_log) as f:
                for line in f:
                    if result['nccd'] is None and 'Keeping' in line and 'CCDs' in line:
                        m = re.search(r'Keeping (\d+) CCDs', line)
                        if m:
                            result['nccd'] = int(m.group(1))

                    if result['nsources'] is None and 'Sources detected:' in line:
                        m = re.search(r'Sources detected: (\d+) in (\d+) blobs', line)
                        if m:
                            result['nsources'] = int(m.group(1))
                            result['nblob'] = int(m.group(2))

            # Stop if we found everything
            if result['nccd'] is not None and result['nsources'] is not None:
                break

    return result
