"""
SGA.util
========

General support utilities.

"""
import numpy as np
from time import time

from SGA.logger import log


TINY = np.nextafter(0, 1, dtype=np.float32)
SQTINY = np.sqrt(TINY)
F32MAX = np.finfo(np.float32).max


def get_dt(t0):
    dt = time() - t0
    if dt > 60.:
        dt /= 60.
        unit = 'minutes'
    else:
        unit = 'seconds'
    return dt, unit


def mwdust_transmission(ebv=0., band='r', run='south'):
    """Convert SFD E(B-V) value to dust transmission 0-1 given the
    bandpass.

    Args:
        ebv (float or array-like): SFD E(B-V) value(s)
        band (str): Bandpass name, e.g., 'r'.
        run (str): Photometric system (e.g., 'south').

    Returns:
        Scalar or array (same as ebv input), Milky Way dust
        transmission 0-1.

    Notes:
        Based on `desiutil.dust.mwdust_transmission`.

    """
    k_X = {
        # GALEX - https://github.com/dstndstn/tractor/issues/99
        'FUV': 6.793,
        'NUV': 6.620,
        # WISE - https://github.com/dstndstn/tractor/blob/main/tractor/sfd.py#L23-L35
        'W1': 0.184,
        'W2': 0.113,
        'W3': 0.0241,
        'W4': 0.00910,
        }

    # LS/DR9 - https://desi.lbl.gov/trac/wiki/ImagingStandardBandpass
    if run == 'south':
        k_X.update({
            'g': 3.212, # DECam
            'r': 2.164,
            'i': 1.591,
            'z': 1.211,
        })
    elif run == 'north':
        k_X.update({
            'g': 3.258, # BASS
            'r': 2.176, # BASS
            'i': 1.591, # DECam
            'z': 1.199, # MzLS
        })
    else:
        msg = f'Unrecognized run {run}'
        log.critical(msg)
        raise ValueError(msg)

    if band not in k_X:
        msg = f'Bandpass {band} is missing from dictionary of known bandpasses!'
        log.critical(msg)
        raise ValueError(msg)

    A_X = k_X[band] * ebv
    transmission = 10.**(-0.4 * A_X)

    return transmission


def var2ivar(var, sigma=False):
    """Simple function to safely turn a variance into an inverse
    variance.

    if sigma=True then assume that `var` is a standard deviation

    """
    ivar = np.zeros_like(var)
    if sigma:
        power = 2.
        ISTINY = SQTINY
    else:
        power = 1.
        ISTINY = TINY

    I = var > ISTINY
    if np.any(I):
        ivar[I] = 1. / var[I]**power

    return ivar


def ivar2var(ivar, clip=0., sigma=False, allmasked_ok=False):
    """Safely convert an inverse variance to a variance.

    """
    var = np.zeros_like(ivar)
    goodmask = ivar > clip # True is good
    if np.count_nonzero(goodmask) == 0:
        # Try clipping at zero.
        goodmask = ivar > 0. # True is good
        if np.count_nonzero(goodmask) == 0:
            if allmasked_ok:
                return var, goodmask
            errmsg = 'All values are masked!'
            log.critical(errmsg)
            raise ValueError(errmsg)
    var[goodmask] = 1. / ivar[goodmask]
    if sigma:
        var = np.sqrt(var) # return a sigma
    return var, goodmask


def match_to(A, B, check_for_dups=True):
    """Return indexes where B matches A, holding A in place.

    Parameters
    ----------
    A : :class:`~numpy.ndarray` or `list`
        Array of integers to match TO.
    B : :class:`~numpy.ndarray` or `list`
        Array of integers matched to `A`
    check_for_dups : :class:`bool`, optional, defaults to ``True``
        If ``True`` trigger an exception if there are duplicates in
        either of the `A` or `B` arrays. Passing ``False`` for
        `check_for_dups` isn't recommended, but is retained to facilitate
        comparisons against earlier versions of the function.

    Returns
    -------
    :class:`~numpy.ndarray`
        The integer indexes in A that B matches to.

    Notes
    -----
    - Result should be such that for ii = match_to(A, B)
      np.all(A[ii] == B) is True.
    - We're looking up locations of B in A so B can be
      a shorter array than A (provided the B->A matches are
      unique) but A cannot be a shorter array than B.
    """
    # ADM grab the integer matchers.
    Aii, Bii = match(A, B, check_for_dups=check_for_dups)

    # ADM return, restoring the original order.
    return Aii[np.argsort(Bii)]


def match(A, B, check_for_dups=True):
    """Return matching indexes for two arrays on a unique key.

    Parameters
    ----------
    A : :class:`~numpy.ndarray` or `list`
        An array of integers.
    B : :class:`~numpy.ndarray` or `list`
        An array of integers.
    check_for_dups : :class:`bool`, optional, defaults to ``True``
        If ``True`` trigger an exception if there are duplicates in
        either of the `A` or `B` arrays. Passing ``False`` for
        `check_for_dups` isn't recommended, but is retained to facilitate
        comparisons against earlier versions of the function.

    Returns
    -------
    :class:`~numpy.ndarray`
        The integer indexes in A that match to B.
    :class:`~numpy.ndarray`
        The integer indexes in B that match to A.

    Notes
    -----
    - Result should be such that for Aii, Bii = match(A, B)
      np.all(A[Aii] == B[Bii]) is True.
    - Only works if there is a unique mapping from A->B, i.e
      if A and B do NOT contain duplicates. This is explicitly checked if
      `check_for_dups` is ``True``
    - h/t Anand Raichoor `by way of Stack Overflow`_.
    """
    # AR sorting A,B
    tmpA = np.sort(A)
    tmpB = np.sort(B)

    # ADM via AR rapid check for duplicates in either array.
    if check_for_dups:
        n_Adups = np.count_nonzero(tmpA[1:] == tmpA[:-1])
        n_Bdups = np.count_nonzero(tmpB[1:] == tmpB[:-1])
        msg = []
        if n_Adups > 0:
            msg.append("Array A has {} duplicates".format(n_Adups))
        if n_Bdups > 0:
            msg.append("Array B has {} duplicates".format(n_Bdups))
        if len(msg) > 0:
            msg = "; ".join(msg)
            print(msg)
            raise ValueError(msg)

    # AR mask equivalent to np.in1d(A, B) for unique elements.
    maskA = (
        np.searchsorted(tmpB, tmpA, "right") - np.searchsorted(tmpB, tmpA, "left")
    ) == 1
    maskB = (
        np.searchsorted(tmpA, tmpB, "right") - np.searchsorted(tmpA, tmpB, "left")
    ) == 1

    # AR to get back to original indexes
    return np.argsort(A)[maskA], np.argsort(B)[maskB]
