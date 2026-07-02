"""
SGA.util
========

General support utilities.

"""
import numpy as np
from time import time

from SGA.logger import log

try:  # this fails when building the documentation
    from scipy import constants
    C_LIGHT = constants.c / 1000.0  # [km/s]
except:
    C_LIGHT = 299792.458  # [km/s]


TINY = np.nextafter(0, 1, dtype=np.float32)
SQTINY = np.sqrt(TINY)
F32MAX = np.finfo(np.float32).max


def get_dt(t0):
    """Compute an elapsed time since `t0`, in human-friendly units.

    Parameters
    ----------
    t0 : :class:`float`
        Start time, as returned by :func:`time.time`.

    Returns
    -------
    dt : :class:`float`
        Elapsed time since `t0`, in `unit`.
    unit : :class:`str`
        Either ``'seconds'`` (if the elapsed time is 60 seconds or
        less) or ``'minutes'`` (otherwise).

    """
    dt = time() - t0
    if dt > 60.:
        dt /= 60.
        unit = 'minutes'
    else:
        unit = 'seconds'
    return dt, unit


def filter_effwaves(run='south'):
    """Return the effective wavelengths of the standard Legacy Survey
    (grz), GALEX (FUV/NUV), and unWISE (W1-W4) filters.

    Parameters
    ----------
    run : :class:`str`
        Photometric system to use for the optical (grz) effective
        wavelengths: ``'south'`` (DECam) or ``'north'`` (BASS+MzLS).
        Any other value silently leaves ``g``, ``r``, ``z`` out of the
        returned dictionary (see Notes).

    Returns
    -------
    :class:`dict`
        Mapping of filter name to effective wavelength, in Angstroms.
        Always includes ``FUV``, ``NUV``, ``W1``-``W4``, and ``i``
        (DECam only -- there is no ``i``-band in the BASS+MzLS
        ``'north'`` system, but the key is present regardless of
        `run`); includes ``g``, ``r``, ``z`` only when `run` is
        ``'south'`` or ``'north'``.

    Notes
    -----
    Unlike :func:`mwdust_transmission`, an unrecognized `run` does not
    raise -- it silently returns a dictionary missing ``g``, ``r``,
    ``z``.

    """
    weff = {
        'FUV': 1528.0,
        'NUV': 2271.0,
        'W1': 34002.54044482,
        'W2': 46520.07577119,
        'W3': 128103.3789599,
        'W4': 223752.7751558,
        'i': 7847.78249813, # no i-band in the north
    }
    if run == 'south':
        weff.update({
            'g': 4890.03670428,
            'r': 6469.62203811,
            'z': 9196.46396394,
        })
    elif run == 'north':
        weff.update({
            'g': 4815.95363513,
            'r': 6437.79282937,
            'z': 9229.65786449,
        })

    return weff


def mwdust_transmission(ebv=0., band='r', run='south'):
    """Convert an SFD E(B-V) value to a Milky Way dust transmission
    fraction, given the bandpass.

    Parameters
    ----------
    ebv : :class:`float` or array-like
        SFD E(B-V) value(s).
    band : :class:`str`
        Bandpass name, e.g., ``'r'``. Must be one of ``FUV``, ``NUV``,
        ``W1``-``W4``, or (for ``run='south'``/``'north'``) ``g``,
        ``r``, ``i``, ``z``.
    run : :class:`str`
        Photometric system: ``'south'`` (DECam) or ``'north'``
        (BASS+MzLS).

    Returns
    -------
    :class:`float` or array-like
        Milky Way dust transmission, 0-1 (same shape as `ebv`).

    Raises
    ------
    ValueError
        If `run` is not ``'south'`` or ``'north'``, or if `band` is
        not a recognized bandpass.

    Notes
    -----
    Based on ``desiutil.dust.mwdust_transmission``. Unlike
    :func:`filter_effwaves`, an unrecognized `run` raises instead of
    silently omitting bands.

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
    """Safely convert a variance (or standard deviation) into an
    inverse variance, avoiding division by zero.

    Parameters
    ----------
    var : array-like
        Variance values, or standard-deviation values if `sigma` is
        True.
    sigma : :class:`bool`
        If True, treat `var` as a standard deviation (``ivar = 1 /
        var**2``) rather than a variance (``ivar = 1 / var``).

    Returns
    -------
    :class:`numpy.ndarray`
        Inverse-variance array, same shape as `var`. Entries where
        `var` is at or below a tiny numerical floor (``TINY`` or, if
        `sigma`, ``SQTINY``) are set to zero rather than dividing by
        (near-)zero.

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
    """Safely convert an inverse variance to a variance (or standard
    deviation), avoiding division by zero.

    Parameters
    ----------
    ivar : array-like
        Inverse-variance values.
    clip : :class:`float`
        Threshold below which `ivar` values are treated as masked
        (``ivar > clip`` defines the "good" mask). If no values pass
        this threshold, retries with a threshold of exactly zero
        before giving up (see Notes).
    sigma : :class:`bool`
        If True, return the standard deviation (``sqrt(var)``) instead
        of the variance.
    allmasked_ok : :class:`bool`
        If True, allow every value to be masked (returns an all-zero
        `var` and an all-False `goodmask`) instead of raising.

    Returns
    -------
    var : :class:`numpy.ndarray`
        Variance (or standard deviation, if `sigma`) array, same shape
        as `ivar`; zero wherever ``goodmask`` is False.
    goodmask : :class:`numpy.ndarray`
        Boolean array, True where `ivar` passed the `clip` (or
        zero-floor) threshold.

    Raises
    ------
    ValueError
        If every value is masked (``ivar <= clip`` and ``ivar <= 0``
        everywhere) and `allmasked_ok` is False.

    Notes
    -----
    If `clip` is nonzero and masks out every element, this function
    silently retries with a threshold of exactly zero before raising
    -- so a nonzero `clip` is only a soft preference, not a hard
    floor, when it would otherwise mask the entire array.

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
    A : :class:`numpy.ndarray` or `list`
        Array of integers to match TO.
    B : :class:`numpy.ndarray` or `list`
        Array of integers matched to `A`
    check_for_dups : :class:`bool`, optional, defaults to ``True``
        If ``True`` trigger an exception if there are duplicates in
        either of the `A` or `B` arrays. Passing ``False`` for
        `check_for_dups` isn't recommended, but is retained to facilitate
        comparisons against earlier versions of the function.

    Returns
    -------
    :class:`numpy.ndarray`
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
    A : :class:`numpy.ndarray` or `list`
        An array of integers.
    B : :class:`numpy.ndarray` or `list`
        An array of integers.
    check_for_dups : :class:`bool`, optional, defaults to ``True``
        If ``True`` trigger an exception if there are duplicates in
        either of the `A` or `B` arrays. Passing ``False`` for
        `check_for_dups` isn't recommended, but is retained to facilitate
        comparisons against earlier versions of the function.

    Returns
    -------
    :class:`numpy.ndarray`
        The integer indexes in A that match to B.
    :class:`numpy.ndarray`
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


def find_csv_badline(fname):
    """Locate and print the offending line of a malformed CSV file, by
    attempting to read it as an Astropy table and parsing the row
    number out of the resulting error message.

    Parameters
    ----------
    fname : :class:`str`
        Path to the CSV file to diagnose.

    Returns
    -------
    None
        Prints the offending line number and its contents to stdout
        as a side effect; does not return the line itself.

    Notes
    -----
    Broken: this function uses ``Table`` and ``ascii`` (for
    ``ascii.InconsistentTableError``) without importing either --
    neither ``astropy.table.Table`` nor ``astropy.io.ascii`` is
    imported anywhere in this module. Calling this function always
    raises ``NameError`` (on ``Table`` in the ``try`` block, or on
    ``ascii`` while evaluating the ``except`` clause, whichever is
    reached first). It is also not called anywhere else in the
    codebase.

    """
    try:
        t = Table.read(fname, format='csv', comment='#')
    except ascii.InconsistentTableError as e:
        # Extract the line number from the error message
        msg = str(e)
        # Astropy reports “…data line 157”
        for token in msg.split():
            if token.isdigit():
                bad_line_num = int(token)
                break
        else:
            raise  # did not find a line number; rethrow

        # Print the actual offending line
        with open(fname) as f:
            for i, line in enumerate(f, start=1):
                if i == bad_line_num:
                    print(f"\nOffending line {bad_line_num}:")
                    print(line.rstrip())
                    break
