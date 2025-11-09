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
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--mp', default=1, type=int, help='number of multiprocessing processes per MPI rank.')

    parser.add_argument('--first', type=int, help='Index of first object to process.')
    parser.add_argument('--last', type=int, help='Index of last object to process.')
    parser.add_argument('--galaxylist', type=str, default=None, help='Comma-separated list of galaxy names to process.')

    parser.add_argument('--mindiam', default=0., type=float, help='Minimum diameter (arcmin).')
    parser.add_argument('--maxdiam', default=1e3, type=float, help='Maximum diameter (arcmin).')
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
    parser.add_argument('--region', default='dr11-south', choices=['dr9-north', 'dr11-south'], type=str, help='Region analyze')

    parser.add_argument('--datadir', default=None, type=str, help='Override $SGA_DATA_DIR environment variable')
    parser.add_argument('--htmldir', default=None, type=str, help='Override $SGA_HTML_DIR environment variable')

    parser.add_argument('--no-groups', action='store_true', help='Ignore angular group parameters; fit individual galaxies (with --coadds).')
    parser.add_argument('--test-bricks', action='store_true', help='Read the sample of test bricks.')

    parser.add_argument('--no-unwise', action='store_false', dest='unwise', help='Do not build unWISE coadds or do forced unWISE photometry.')
    parser.add_argument('--no-galex', action='store_false', dest='galex', help='Do not build GALEX coadds or do forced GALEX photometry.')
    parser.add_argument('--no-cleanup', action='store_false', dest='cleanup', help='Do not clean up legacypipe files after coadds.')

    parser.add_argument('--diameter-file', default=None, type=str, help='Write a diameter file for use with generate_sga_jobs.sh')

    #parser.add_argument('--ubercal-sky', action='store_true', help='Build the largest large-galaxy coadds with custom (ubercal) sky-subtraction.')
    parser.add_argument('--fit-on-coadds', action='store_true', help='Fit on coadds.')
    parser.add_argument('--force', action='store_true', help='Use with --coadds; ignore previous pickle files.')
    parser.add_argument('--count', action='store_true', help='Count how many objects are left to analyze and then return.')
    parser.add_argument('--debug', action='store_true', help='Log to STDOUT and build debugging plots.')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output.')
    parser.add_argument('--clobber', action='store_true', help='Overwrite existing files.')

    parser.add_argument('--lvd', action='store_true', help='Read the parent LVD sample.')
    parser.add_argument('--wisesize', action='store_true', help='Read the wisesize parent sample.')

    parser.add_argument('--use-gpu', action='store_true', help='Launch the GPU version of legacypipe (only with --coadds).')
    parser.add_argument('--ngpu', default=1, type=int, help='Number of GPUs to use.')
    parser.add_argument('--threads-per-gpu', default=8, type=int, help='Max threads per GPU - CPU will fill remaining threads.')

    parser.add_argument('--build-refcat', action='store_true', help='Build the legacypipe reference catalog.')
    parser.add_argument('--build-catalog', action='store_true', help='Build the final catalog.')
    args = parser.parse_args()

    return args


def weighted_partition(weights, n):
    '''
    Partition `weights` into `n` groups with approximately same sum(weights)

    Args:
        weights: array-like weights
        n: number of groups

    Returns list of lists of indices of weights for each group

    Notes:
        compared to `dist_discrete_all`, this function allows non-contiguous
        items to be grouped together which allows better balancing.

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
                    small_bricks_first=False):
    """
    Partition tasks into `size` buckets with ~equal total weight, then
    sort each bucket so smaller bricks are processed first.

    Parameters
    ----------
    diameter : array-like
        Brick diameters (arcmin) for all items.
    itodo : array-like of int or None, optional
        Indices to schedule (default: all items).
    size : int, optional
        Number of MPI ranks (default: 1).
    p : float, optional
        Exponent for weight model: weight = diameter**p (default: 2.0).
    verbose : bool, optional
        If True, print a load-balance summary.

    Returns
    -------
    todo_indices : list of np.ndarray
        One array of indices per rank. Each array is sorted ascending
        by diameter so each rank does small bricks first.
    loads : np.ndarray
        Total weight per rank.

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
