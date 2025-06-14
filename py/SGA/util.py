"""
SGA.util
========

Support utilities.

"""
import pdb
import numpy as np
from astropy.table import Table

from astrometry.util.starutil_numpy import arcsec_between
from astrometry.libkd.spherematch import match_radec


def find_close(cat, fullcat, rad_arcsec=1., isolated=False):

    rad = rad_arcsec / 3600.
    allmatches = match_radec(cat['RA'].value, cat['DEC'].value,
                             fullcat['RA'].value, fullcat['DEC'].value,
                             rad, indexlist=True, notself=False)
    primaryindx, groupindx = [], []
    for ii, mm in enumerate(allmatches):
        if mm is not None:
            ngroup = len(mm)
            #print(ngroup, ii, mm)
            if isolated:
                if ngroup == 1:
                    primaryindx.append(ii)
                    groupindx.append(mm)
            else:
                primaryindx.append(ii)
                groupindx.append(mm)

    if len(primaryindx) == 0:
        return [], []
    primaryindx = np.array(primaryindx)
    primaries = cat[primaryindx]
    primaries = primaries[np.argsort(primaries['RA'])]

    groupindx = np.hstack(groupindx)
    groups = fullcat[groupindx]
    groups = groups[np.argsort(groups['RA'])]

    return primaries, groups


def choose_primary(group, verbose=False, keep_all_mergers=False, ignore_objtype=False):
    """Choose the primary member of a group.

    keep_all is helpful for returning a group catalog without dropping any
    sources.

    if keep_all_mergers=True then always keep {GPair,GTrpl} sources, even
      if they do not have a diameter.

    if allow_vetos=True then do not drop systems that are in a 'veto' list,
      even if they overlap with another source.

    """
    if keep_all_mergers:
        IM = np.logical_or(group['OBJTYPE'] == 'GPair', group['OBJTYPE'] == 'GTrpl')
        IG = group['OBJTYPE'] == 'G'
    else:
        IG = np.logical_or(group['OBJTYPE'] == 'G', group['OBJTYPE'] == 'GPair', group['OBJTYPE'] == 'GTrpl')

    #IG = np.logical_or.reduce((group['OBJTYPE'] == 'G', group['OBJTYPE'] == 'GPair', group['OBJTYPE'] == 'GTrpl'))
    #ID = np.vstack((group['DIAM_LIT'] != -99., group['DIAM_HYPERLEDA'] != -99., group['DIAM_SGA2020'] != -99.)).T
    ID = np.vstack((group['DIAM_LIT'] != -99., group['DIAM_HYPERLEDA'] != -99.)).T
    IZ = group['Z'] != -99.
    IS = group['SEP'] == 0.

    if ignore_objtype:
        mask1 = np.any(ID, axis=1)      # any diameter
        mask2 = np.all(ID, axis=1)      # both diameters
        mask3 = np.all(ID, axis=1) * IZ # both diameters, and a redshift
    else:
        mask1 = IG * np.any(ID, axis=1)      # objtype=G and any diameter
        mask2 = IG * np.all(ID, axis=1)      # objtype=G and both diameters
        mask3 = IG * np.all(ID, axis=1) * IZ # objtype=G, both diameters, and a redshift
    mask4 = np.all(ID, axis=1) * IZ      # both diameters and a redshift
    mask5 = np.all(ID, axis=1) * IS      # both diameters and separation=0 (usually PGC is a minimum)
    mask6 = np.all(ID, axis=1)           # both diameters
    mask7 = np.any(ID, axis=1) * IS      # either diameter and separation=0
    mask8 = IS                           # separation=0
    mask9 = IG                           # objtype=G

    if keep_all_mergers:
        mask0 = IM # objtype={GPair,GTrpl}
        allmasks = (mask0, mask1, mask2, mask3, mask4, mask5, mask6, mask7, mask8)
    else:
        allmasks = (mask1, mask2, mask3, mask4, mask5, mask6, mask7, mask8, mask9)

    for mask in allmasks:
        keep = np.where(mask)[0]
        drop = np.where(~mask)[0]
        if len(keep) == 1:
            keep, drop = np.where(mask)[0], np.where(~mask)[0]
            return keep, drop

    print('Warning: cases 1-9 failed; choosing by prefix.')
    prefer_prefix = ['NGC', 'UGC', 'IC', 'MCG', 'CGCG', 'ESO',
                     'I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII']
    prefix = np.array(list(zip(*np.char.split(group['OBJNAME'].value, ' ').tolist()))[0])
    mask = np.array([pre in prefer_prefix for pre in prefix])
    keep = np.where(mask)[0]
    drop = np.where(~mask)[0]
    if len(keep) == 1:
        print(group['OBJNAME', 'OBJTYPE', 'RA', 'DEC', 'DIAM_LIT', 'DIAM_HYPERLEDA', 'Z', 'PGC', 'SEP'])
        keep, drop = np.where(mask)[0], np.where(~mask)[0]
        return keep, drop

    print('Warning: choosing by prefix failed; choosing by minimum separation.')
    print(group['OBJNAME', 'OBJTYPE', 'RA', 'DEC', 'DIAM_LIT', 'DIAM_HYPERLEDA', 'Z', 'PGC', 'SEP'])
    print()
    keep = np.atleast_1d(group['SEP'].argmin())
    return keep, np.delete(np.arange(len(group)), keep)
    #indx = np.arange(len(group))
    #return indx[:1], indx[1:]


def resolve_close(cat, refcat, maxsep=1., keep_all=False, allow_vetos=False,
                  keep_all_mergers=False, objname_column='OBJNAME',
                  ignore_objtype=False, trim=True, verbose=False):
    """Resolve close objects.

    maxsep in arcsec
    cat - smaller catalog
    refcat - full catalog

    """
    VETO = [
        '2MASX J15134005+2607307',  # overlaps with 3C 315, but the coordinates for 3C 315 are wrong
    ]

    allmatches = match_radec(cat['RA'].value, cat['DEC'].value,
                             refcat['RA'].value, refcat['DEC'].value,
                             maxsep/3600., indexlist=True, notself=False)

    nobj = 0
    allindx_cat, allindx_refcat = [], []
    for iobj, onematch in enumerate(allmatches):
        if onematch is None:
            continue
        nmatch = len(onematch)
        if nmatch > 1:
            nobj += nmatch
            allindx_cat.append(iobj)
            allindx_refcat.append(onematch)

    if verbose:
        maxname = len(max(refcat[objname_column], key=len))
        maxtyp = len(max(refcat['OBJTYPE'], key=len))

    refcat['GROUP_ID'] = np.zeros(len(refcat), np.int32) - 99
    refcat['PRIMARY'] = np.ones(len(refcat), bool)
    refcat['NGROUP'] = np.ones(len(refcat), np.int16)
    refcat['SEPARATION'] = np.zeros(len(refcat), 'f4')
    refcat['DONE'] = np.zeros(len(refcat), bool)

    for igroup, (indx_cat, indx_refcat) in enumerate(zip(allindx_cat, allindx_refcat)):
        if verbose and (igroup % 500 == 0):
            print(f'Working on group {igroup+1:,d}/{len(allindx_refcat):,d}')
        indx_cat = np.array(indx_cat)
        indx_refcat = np.array(indx_refcat)
        if np.all(refcat['DONE'][indx_refcat]):
            continue

        group = refcat[indx_refcat]
        dtheta = arcsec_between(Table(cat[indx_cat])['RA'], Table(cat[indx_cat])['DEC'],
                                group['RA'].value, group['DEC'].value)
        group['SEP'] = dtheta.astype('f4')
        ngroup = len(group)

        refcat['GROUP_ID'][indx_refcat] = igroup
        refcat['NGROUP'][indx_refcat] = ngroup
        refcat['SEPARATION'][indx_refcat] = dtheta
        refcat['DONE'][indx_refcat] = True

        if keep_all:
            primary = np.arange(ngroup)
            drop = np.array([])
        else:
            primary, drop = choose_primary(group, verbose=verbose, keep_all_mergers=keep_all_mergers,
                                           ignore_objtype=ignore_objtype)
            refcat['PRIMARY'][indx_refcat[drop]] = False

        #if verbose and (np.any(group['OBJTYPE'] == 'GPair') or np.any(group['OBJTYPE'] == 'GTrpl')):
        if verbose:
            for ii in primary:
                print('Keep: '+'{name: <{W}}'.format(name=group[ii][objname_column], W=maxname)+': ' + \
                      '{typ: <{W}}'.format(typ=group[ii]["OBJTYPE"], W=maxtyp)+', ' + \
                      f'PGC={group[ii]["PGC"]}, z={group[ii]["Z"]:.5f}, D={group[ii]["DIAM_LIT"]:.2f}, ' + \
                      f'D(LEDA)={group[ii]["DIAM_HYPERLEDA"]:.2f} arcmin, (ra,dec)={group[ii]["RA"]:.6f},' + \
                      f'{group[ii]["DEC"]:.6f}')
            for ii in drop:
                print('Drop: '+'{name: <{W}}'.format(name=group[ii][objname_column], W=maxname)+': ' + \
                      '{typ: <{W}}'.format(typ=group[ii]["OBJTYPE"], W=maxtyp)+', ' + \
                      f'PGC={group[ii]["PGC"]}, z={group[ii]["Z"]:.5f}, D={group[ii]["DIAM_LIT"]:.2f}, ' + \
                      f'D(LEDA)={group[ii]["DIAM_HYPERLEDA"]:.2f} arcmin, sep={group[ii]["SEP"]:.3f} arcsec')

        # check for vetos
        if allow_vetos:
            for veto in VETO:
                I = np.where(np.isin(refcat[indx_refcat][objname_column], veto))[0]
                if len(I) == 1:
                    if refcat[indx_refcat[I]]['PRIMARY'] == False:
                        print(f'Restoring vetoed object {refcat[indx_refcat[I[0]]][objname_column]}')
                        refcat['PRIMARY'][indx_refcat[I]] = True
        if verbose:
            print()

    ugroup = np.unique(refcat["GROUP_ID"])
    ugroup = ugroup[ugroup != -99]

    print(f'Found {len(ugroup):,d} group(s) with ' + \
          f'({np.sum(refcat["GROUP_ID"] != -99):,d}/{len(refcat):,d} ' + \
          f'unique objects) within {maxsep:.1f} arcsec.')

    #check = refcat[refcat['GROUP_ID'] != -99]
    #check = check[np.argsort(check['GROUP_ID'])]
    #drop = refcat[(refcat['GROUP_ID'] != -99) * (refcat['PRIMARY'] == False)]
    #prefix = np.array([objname.split(' ')[0] for objname in drop['OBJNAME']])

    if trim:
        out = refcat[refcat['PRIMARY']]
        out.remove_columns(['GROUP_ID', 'PRIMARY', 'NGROUP', 'SEPARATION', 'DONE'])
        return out
    else:
        refcat.remove_column('DONE')
        return refcat


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


