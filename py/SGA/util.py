"""
SGA.util
========

Support utilities.

"""
import numpy as np


def get_basic_geometry(cat, galaxy_column='OBJNAME', verbose=False):
    """From a catalog containing magnitudes, diameters, position angles, and
    ellipticities, return a "basic" value for each property.

    """
    from astropy.table import Table
    #from SGA.io import read_lvd
    #cat = read_lvd()
    #import fitsio
    #cat = Table(fitsio.read('/Users/ioannis/research/projects/SGA/2024/parent/external/NED-NEDLVS_20210922_v2.fits', rows=np.arange(100)))
    #cat = Table(fitsio.read('/Users/ioannis/research/projects/SGA/2024/parent/external/HyperLeda_meandata_1720804662.fits', rows=np.arange(100)))
    nobj = len(cat)

    basic = Table()
    basic['GALAXY'] = cat[galaxy_column].value

    # HyperLeda
    if 'LOGD25' in cat.columns:
        ref = 'LEDA'
        for prop in ('mag', 'diam', 'ba', 'pa'):
            val = np.zeros(nobj, 'f4') - 99.
            val_ref = np.zeros(nobj, '<U7')
            val_band = np.zeros(nobj, 'U1')

            match prop:
                case 'mag':
                    col = 'BT'
                    band = 'B'
                    I = cat[col] > 0.
                    if np.sum(I) > 0:
                        val[I] = cat[col][I]
                        val_ref[I] = ref
                        val_band[I] = band
                case 'diam':
                    col = 'LOGD25'
                    I = cat[col] > 0.
                    if np.sum(I) > 0:
                        val[I] = 0.1 * 10.**cat[col][I]
                        val_ref[I] = ref
                case 'ba':
                    col = 'LOGR25'
                    I = ~np.isnan(cat[col]) * (cat[col] != 0.)
                    if np.sum(I) > 0:
                        val[I] = 10.**(-cat[col][I])
                        val_ref[I] = ref
                case 'pa':
                    col = 'PA'
                    I = ~np.isnan(cat[col])
                    if np.sum(I) > 0:
                        val[I] = cat[col][I]
                        val_ref[I] = ref

            basic[prop.upper()] = val
            basic[f'{prop.upper()}_REF'] = val_ref
            if prop == 'mag':
                basic['BAND'] = val_band
    # LVD
    elif 'RHALF' in cat.columns:
        ref = 'LVD'
        for prop in ('mag', 'diam', 'ba', 'pa'):
            val = np.zeros(nobj, 'f4') - 99.
            val_ref = np.zeros(nobj, '<U7')
            val_band = np.zeros(nobj, 'U1')

            #import pdb ; pdb.set_trace()
            match prop:
                case 'mag':
                    col = 'APPARENT_MAGNITUDE_V'
                    band = 'V'
                    I = cat[col] > 0.
                    if np.sum(I) > 0:
                        val[I] = cat[col][I]
                        val_ref[I] = ref
                        val_band[I] = band
                case 'diam':
                    col = 'RHALF' # [arcmin]
                    I = cat[col] > 0.
                    if np.sum(I) > 0:
                        val[I] = cat[col][I]
                        val_ref[I] = ref
                case 'ba':
                    col = 'ELLIPTICITY' # =1-b/a
                    I = ~np.isnan(cat[col])
                    if np.sum(I) > 0:
                        val[I] = 1. - cat[col][I]
                        val_ref[I] = ref
                case 'pa':
                    col = 'POSITION_ANGLE'
                    I = ~np.isnan(cat[col])
                    if np.sum(I) > 0:
                        val[I] = cat[col][I] % 180 # put in the range [0, 180]
                        val_ref[I] = ref

            basic[prop.upper()] = val
            basic[f'{prop.upper()}_REF'] = val_ref
            if prop == 'mag':
                basic['BAND'] = val_band
    # NED
    else:
        for prop in ('mag', 'diam', 'ba', 'pa'):
            if prop == 'mag':
                refs = ('SDSS', 'TWOMASS', 'RC3')
                bands = ('R', 'K', 'B')
            else:
                refs = ('ESO', 'SDSS', 'TWOMASS', 'RC3')
                bands = ('B', 'R', 'K', 'B')
            nref = len(refs)

            val = np.zeros(nobj, 'f4') - 99.
            val_ref = np.zeros(nobj, '<U7')
            val_band = np.zeros(nobj, 'U1')

            #allI = np.zeros((nobj, nref), bool)
            for iref, (ref, band) in enumerate(zip(refs, bands)):
                if prop == 'mag':
                    col = f'{ref}_{band}'
                else:
                    col = f'{ref}_{prop.upper()}_{band}'
                I = cat[col] > 0.
                #allI[:, iref] = I

                if np.sum(I) > 0:
                    val[I] = cat[col][I]
                    val_ref[I] = ref
                    val_band[I] = band

            basic[prop.upper()] = val
            basic[f'{prop.upper()}_REF'] = val_ref
            if prop == 'mag':
                basic['BAND'] = val_band

        # supplement any missing values with the "BASIC" data
        I = (basic['MAG'] <= 0.) * (cat['BASIC_MAG'] > 0.)
        if np.any(I):
            basic['MAG'][I] = cat['BASIC_MAG'][I]
            basic['BAND'][I] = 'V'

        I = (basic['DIAM'] <= 0.) * (cat['BASIC_DMAJOR'] > 0.)
        if np.any(I):
            basic['DIAM'][I] = cat['BASIC_DMAJOR'][I]
            basic['DIAM_REF'][I] = 'BASIC'

        I = (basic['BA'] <= 0.) * (cat['BASIC_DMAJOR'] > 0.) * (cat['BASIC_DMINOR'] > 0.)
        if np.any(I):
            basic['BA'][I] = cat['BASIC_DMINOR'][I] / cat['BASIC_DMAJOR'][I]
            basic['BA_REF'][I] = 'BASIC'

    # summarize
    if verbose:
        M = basic['MAG'] > 0.
        D = basic['DIAM'] > 0.
        print(f'Derived photometry for {np.sum(M):,d}/{nobj:,d} objects and ' + \
              f'diameters for {np.sum(D):,d}/{nobj:,d} objects.')

    return basic


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
