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
    parser.add_argument('--galaxylist', type=str, nargs='*', default=None, help='List of galaxy names to process.')

    parser.add_argument('--d25min', default=0.0, type=float, help='Minimum diameter (arcmin).')
    parser.add_argument('--d25max', default=100.0, type=float, help='Maximum diameter (arcmin).')

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
    parser.add_argument('--nsigma', default=None, type=int, help='detection sigma')
    parser.add_argument('--region', default='dr11-south', choices=['dr9-north', 'dr11-south'], type=str, help='Region analyze')

    parser.add_argument('--datadir', default=None, type=str, help='Override $SGA_DATA_DIR environment variable')
    parser.add_argument('--htmldir', default=None, type=str, help='Override $SGA_HTML_DIR environment variable')

    parser.add_argument('--no-groups', action='store_true', help='Ignore angular group parameters; fit individual galaxies (with --coadds).')

    parser.add_argument('--no-unwise', action='store_false', dest='unwise', help='Do not build unWISE coadds or do forced unWISE photometry.')
    parser.add_argument('--no-galex', action='store_false', dest='galex', help='Do not build GALEX coadds or do forced GALEX photometry.')
    parser.add_argument('--no-cleanup', action='store_false', dest='cleanup', help='Do not clean up legacypipe files after coadds.')

    #parser.add_argument('--ubercal-sky', action='store_true', help='Build the largest large-galaxy coadds with custom (ubercal) sky-subtraction.')
    parser.add_argument('--force', action='store_true', help='Use with --coadds; ignore previous pickle files.')
    parser.add_argument('--count', action='store_true', help='Count how many objects are left to analyze and then return.')
    parser.add_argument('--debug', action='store_true', help='Log to STDOUT and build debugging plots.')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output.')
    parser.add_argument('--clobber', action='store_true', help='Overwrite existing files.')

    parser.add_argument('--lvd', action='store_true', help='Read the parent LVD sample.')

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
