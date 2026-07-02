"""
SGA.sky
=======

Utilities pertaining to operations on the sky.

"""
import pdb
import numpy as np
from astropy.table import Table

from astrometry.util.starutil_numpy import arcsec_between
from astrometry.libkd.spherematch import match_radec

from SGA.coadds import PIXSCALE
from SGA.logger import log


def find_in_mclouds(cat, mcloud='LMC'):
    """Flag catalog objects falling within a Magellanic Cloud's
    elliptical footprint.

    Looks up ``mcloud``'s own row in ``cat`` (by ``OBJNAME``) to get its
    center and geometry (via :func:`SGA.geometry.choose_geometry`), then
    flags every object in ``cat`` inside that ellipse.

    Parameters
    ----------
    cat : :class:`~astropy.table.Table`
        Catalog to search; must contain a row with ``OBJNAME == mcloud``.
    mcloud : :class:`str`
        Name of the Magellanic Cloud to use, e.g. ``'LMC'`` or ``'SMC'``.

    Returns
    -------
    :class:`numpy.ndarray`
        Boolean mask, length ``len(cat)``, True for objects inside the
        cloud's ellipse.

    Raises
    ------
    ValueError
        If ``mcloud`` is not found in ``cat``.

    """
    from SGA.geometry import choose_geometry

    gal = cat[cat['OBJNAME'] == mcloud]
    if len(gal) == 0:
        msg = f'Magellanic Cloud {mcloud} not found in input catalog!'
        log.critical(msg)
        raise ValueError(msg)

    racen, deccen = gal['RA'].value, gal['DEC'].value
    #gal['DIAM_HYPERLEDA', 'PA_HYPERLEDA', 'BA_HYPERLEDA', 'DIAM_LIT', 'DIAM_LIT_REF', 'PA_LIT', 'BA_LIT']

    diam, ba, pa, _ = choose_geometry(gal)
    semia = diam / 2. / 3600. # [degrees]
    semib = ba * semia

    in_mcloud = in_ellipse_mask_sky(
        racen, deccen, semia, semib, pa,
        cat['RA'].value, cat['DEC'].value)

    return in_mcloud


def find_in_gclpne(cat):
    """Flag catalog objects falling within a Galactic globular cluster or
    planetary nebula's footprint.

    Reads legacypipe's packaged globular-cluster/PNe reference catalog
    (``NGC-star-clusters.fits``) and flags every object in ``cat``
    inside any of its elliptical footprints.

    Parameters
    ----------
    cat : :class:`~astropy.table.Table`
        Catalog to search; needs ``RA``/``DEC`` columns.

    Returns
    -------
    :class:`numpy.ndarray`
        Boolean mask, length ``len(cat)``, True for objects inside any
        cluster/PNe ellipse.

    """
    from importlib import resources
    import fitsio

    gclfile = str(resources.files('legacypipe').joinpath('data/NGC-star-clusters.fits'))
    gcl = Table(fitsio.read(gclfile))
    log.info(f'Read {len(gcl):,d} objects from {gclfile}')

    in_gclpne = np.zeros(len(cat), bool)

    for cl in gcl:
        I = in_ellipse_mask_sky(cl['ra'], cl['dec'], cl['radius'], cl['ba']*cl['radius'],
                                cl['pa'], cat['RA'].value, cat['DEC'].value)
        if np.any(I):
            in_gclpne[I] = True

    return in_gclpne


def map_bxby(bx, by, from_wcs, to_wcs):
    """Map ``(bx, by)`` pixel coordinates from one WCS to another (e.g.,
    optical to GALEX).

    Parameters
    ----------
    bx, by : :class:`float` or array-like
        Pixel coordinates in ``from_wcs``.
    from_wcs, to_wcs : WCS
        Source and destination WCS objects (0-indexed pixel convention,
        via ``wcs.pixelxy2radec``/``wcs.radec2pixelxy``).

    Returns
    -------
    to_bx, to_by : :class:`float` or array-like
        Pixel coordinates in ``to_wcs``.

    """
    ra, dec = from_wcs.wcs.pixelxy2radec(bx+1., by+1)
    (_, to_bx, to_by) = to_wcs.wcs.radec2pixelxy(ra, dec)
    return to_bx-1., to_by-1.


def simple_wcs(racenter, deccenter, width, pixscale=0.262):
    """Build a simple, square, tangent-plane (TAN) WCS centered at a
    given sky position.

    Parameters
    ----------
    racenter, deccenter : :class:`float`
        Center of the mosaic, degrees; placed at the reference pixel
        (``CRPIX`` = center of the array, ``CRVAL`` = this position).
    width : :class:`int`
        Mosaic width and height, in pixels (square).
    pixscale : :class:`float`
        Pixel scale, arcsec/pixel. RA increases toward -x (standard sky
        orientation, East left).

    Returns
    -------
    :class:`~astropy.wcs.WCS`
        The constructed WCS.

    """
    from astropy.wcs import WCS
    from astropy.io import fits
    hdr = fits.Header()
    hdr['NAXIS'] = 2
    hdr['NAXIS1'] = width
    hdr['NAXIS2'] = width
    hdr['CTYPE1'] = 'RA---TAN'
    hdr['CTYPE2'] = 'DEC--TAN'
    hdr['CRVAL1'] = racenter
    hdr['CRVAL2'] = deccenter
    hdr['CRPIX1'] = width/2+0.5
    hdr['CRPIX2'] = width/2+0.5
    hdr['CD1_1'] = -pixscale/3600.
    hdr['CD1_2'] = 0.0
    hdr['CD2_1'] = 0.0
    hdr['CD2_2'] = +pixscale/3600.
    return WCS(hdr)


def get_ccds(allccds, onegal, width_pixels, pixscale=PIXSCALE, return_ccds=False):
    """Determine which CCDs (from an already-loaded CCDs table) touch a
    custom brick centered on one galaxy, and augment that galaxy's row
    with the match.

    Mostly taken from ``legacypipe.runbrick.stage_tims``. Distinct from
    :func:`SGA.coadds.get_ccds`, which queries a legacypipe survey
    object directly (``survey.ccds_touching_wcs``) rather than filtering
    an already-loaded ``allccds`` table; the two are not
    interchangeable despite the shared name.

    Parameters
    ----------
    allccds : CCDs table
        Full CCDs table to search (e.g. from a legacypipe survey's
        ``get_ccds_readonly()``), in the format expected by
        ``legacypipe.survey.ccds_touching_wcs``.
    onegal : :class:`~astropy.table.Row`
        Single galaxy row; needs ``RA``, ``DEC``, ``ROW_PARENT``. If at
        least one CCD touches, updated in place with ``NCCD`` (number
        of touching CCDs) and ``FILTERS`` (sorted, concatenated unique
        filter letters).
    width_pixels : :class:`int`
        Mosaic width, in pixels, used to build the search WCS.
    pixscale : :class:`float`
        Pixel scale, arcsec/pixel.
    return_ccds : :class:`bool`
        If True, return a ``(ccds, onegal_table)`` tuple instead of just
        ``onegal_table``.

    Returns
    -------
    :class:`~astropy.table.Table`
        If ``return_ccds=False``: ``onegal`` (updated with
        ``NCCD``/``FILTERS``) wrapped as a single-row Table; empty if no
        CCDs touch. If ``return_ccds=True``: a ``(ccds, onegal_table)``
        tuple, where ``ccds`` holds the matched CCDs (``RA``, ``DEC``,
        ``CAMERA``, ``EXPNUM``, ``PLVER``, ``CCDNAME``, ``FILTER``,
        ``ROW_PARENT``) and both are empty Tables if none touch.

    """
    from SGA.brick import custom_brickname
    from legacypipe.survey import wcs_for_brick, BrickDuck, ccds_touching_wcs

    brickname = f'custom-{custom_brickname(onegal["RA"], onegal["DEC"])}'
    brick = BrickDuck(onegal['RA'], onegal['DEC'], brickname)

    targetwcs = wcs_for_brick(brick, W=float(width_pixels),
                              H=float(width_pixels), pixscale=pixscale)
    I = ccds_touching_wcs(targetwcs, allccds)
    #ccds = survey.ccds_touching_wcs(targetwcs)
    #log.info(len(I))
    #log.info('HACK!!')
    #pdb.set_trace()

    # no CCDs within width_pixels
    if len(I) == 0:
        if return_ccds:
            return Table(), Table()
        else:
            return Table()
    ccds = allccds[I]

    onegal['NCCD'] = len(ccds)
    onegal['FILTERS'] = ''.join(sorted(set(ccds.filter)))

    if return_ccds:
        # convert to an astropy Table so we can vstack
        _ccds = ccds.to_dict()
        ccds = Table()
        for key in _ccds.keys():
            ccds[key.upper()] = _ccds[key]

        ccds = ccds['RA', 'DEC', 'CAMERA', 'EXPNUM', 'PLVER', 'CCDNAME', 'FILTER']
        #ccds['GALAXY'] = [galaxy]
        ccds['ROW_PARENT'] = onegal['ROW_PARENT']

        return ccds, Table(onegal)
    else:
        return Table(onegal)


def in_ellipse_mask_sky(racen, deccen, semia, semib, pa, ras, decs):
    """Test whether points on the sky fall inside an elliptical region,
    using the astronomical position-angle convention.

    Uses a flat-sky (tangent-plane-like) approximation: wraps the RA
    offset into [-180, 180) and scales it by ``cos(deccen)`` to correct
    for meridian convergence, then applies the same rotated-ellipse test
    as :func:`SGA.geometry.in_ellipse_mask`.

    Parameters
    ----------
    racen, deccen : :class:`float`
        Center of the ellipse, in degrees (RA, Dec).
    semia, semib : :class:`float`
        Semi-major and semi-minor axes of the ellipse, in degrees.
    pa : :class:`float`
        Position angle of the major axis, in degrees, measured
        counterclockwise from +Dec (North) toward +RA (East).
    ras, decs : array-like
        Point coordinates to test (RA, Dec), in degrees.

    Returns
    -------
    :class:`numpy.ndarray` of :class:`bool`
        True for points inside or on the ellipse.

    """
    # compute offsets in a “flat‐sky” projection
    # wrap RA difference into [–180, +180]
    dra = (ras - racen + 180) % 360 - 180
    # correct for convergence of meridians
    dra *= np.cos(np.radians(deccen))
    ddec = decs - deccen

    # convert PA to radians
    theta = np.deg2rad(pa)

    # project onto the ellipse axes:
    # major‐axis unit vector = (sinθ, cosθ)
    xp =  dra * np.sin(theta) + ddec * np.cos(theta)
    # minor‐axis unit vector = (–cosθ, sinθ)
    yp = -dra * np.cos(theta) + ddec * np.sin(theta)

    # standard ellipse test (x'/a)^2 + (y'/b)^2 <= 1
    return (xp/semia)**2 + (yp/semib)**2 <= 1


def find_close(cat, fullcat, rad_arcsec=1., isolated=False, return_counts=False):
    """Find objects in ``cat`` with a positional match in ``fullcat``
    within a given search radius.

    For each object in ``cat`` with at least one match in ``fullcat``
    (every object always matches itself at minimum, since
    ``notself=False``), optionally restrict to those whose *only* match
    is itself (``isolated=True``, i.e. no other ``fullcat`` object
    within ``rad_arcsec``).

    Notes
    -----
    When ``isolated=True`` and ``return_counts=True``, the returned
    ``ningroup`` is misaligned with ``primaries``/``groups``: it is
    appended for *every* object with a match (i.e. essentially all of
    ``cat``, since self-matches always count), not just the isolated
    ones that actually end up in ``primaries``/``groups``. So its
    length and per-row correspondence to ``primaries`` only holds when
    ``isolated=False``.

    Parameters
    ----------
    cat : :class:`~astropy.table.Table`
        Objects to search for matches of; needs ``RA``/``DEC``.
    fullcat : :class:`~astropy.table.Table`
        Catalog to match against; needs ``RA``/``DEC``.
    rad_arcsec : :class:`float`
        Search radius, arcsec.
    isolated : :class:`bool`
        If True, keep only objects whose sole match in ``fullcat`` is
        themselves (group size 1).
    return_counts : :class:`bool`
        If True, also return the per-group match count (see Notes for a
        caveat when combined with ``isolated=True``).

    Returns
    -------
    primaries : :class:`~astropy.table.Table` or :class:`list`
        Matched objects from ``cat``, sorted by RA; ``[]`` if none.
    groups : :class:`~astropy.table.Table` or :class:`list`
        All ``fullcat`` matches (including self-matches), sorted by RA;
        ``[]`` if none.
    ningroup : :class:`numpy.ndarray`, only if ``return_counts=True``
        Match-group size per entry (see Notes for the ``isolated=True``
        misalignment caveat).

    """
    rad = rad_arcsec / 3600.
    allmatches = match_radec(cat['RA'].value, cat['DEC'].value,
                             fullcat['RA'].value, fullcat['DEC'].value,
                             rad, indexlist=True, notself=False)
    primaryindx, groupindx, ningroup = [], [], []
    for ii, mm in enumerate(allmatches):
        if mm is not None:
            ngroup = len(mm)
            ningroup.append(ngroup)
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
    ningroup = np.array(ningroup)

    primaries = cat[primaryindx]
    primaries = primaries[np.argsort(primaries['RA'])]

    groupindx = np.hstack(groupindx)
    groups = fullcat[groupindx]
    groups = groups[np.argsort(groups['RA'])]

    if return_counts:
        return primaries, groups, ningroup
    else:
        return primaries, groups


def choose_primary(group, verbose=False, keep_all_mergers=False, ignore_objtype=False):
    """Choose which member(s) of a group of closely-spaced objects to
    keep as primary, trying a sequence of increasingly permissive
    criteria until exactly one candidate is selected.

    Tries masks in order (first one selecting exactly 1 object wins):
    objects with (in order) any diameter, both LIT and HyperLeda
    diameters, both diameters plus a redshift, both diameters plus a
    redshift (ignoring ``OBJTYPE``), both diameters plus zero
    separation, both diameters, either diameter plus zero separation,
    zero separation, or (unless ``ignore_objtype``) simply
    ``OBJTYPE == 'G'``. If ``keep_all_mergers``, an ``OBJTYPE`` in
    ``{'GPair', 'GTrpl'}`` is tried first, ahead of all of the above. If
    none of these narrow the group to exactly one object, falls back to
    preferring well-known catalog name prefixes (``NGC``, ``UGC``,
    etc.), then finally to the object with minimum separation from the
    group's search center.

    Notes
    -----
    When ``keep_all_mergers=False``, the ``IG`` mask is computed as
    ``np.logical_or(group['OBJTYPE'] == 'G', group['OBJTYPE'] == 'GPair',
    group['OBJTYPE'] == 'GTrpl')`` -- but :func:`numpy.logical_or` only
    takes two positional array arguments before its keyword-only ``out``
    parameter, so the third argument here binds to ``out`` instead of
    being OR'd in. The result is that ``IG`` is actually just
    ``(OBJTYPE == 'G') | (OBJTYPE == 'GPair')``, silently excluding
    ``'GTrpl'`` objects from every ``IG``-gated mask (``mask1``-``mask3``,
    ``mask9``). A commented-out line just above
    (``#IG = np.logical_or.reduce((...))``) shows the originally-intended
    3-way OR.

    Parameters
    ----------
    group : :class:`~astropy.table.Table`
        Candidate members of one close-pair/group, from
        :func:`resolve_close`; needs ``OBJTYPE``, ``DIAM_LIT``,
        ``DIAM_HYPERLEDA``, ``Z``, ``SEP``, ``OBJNAME``.
    verbose : :class:`bool`
        If True, print the fallback-selection diagnostic table when the
        primary criteria don't isolate a single object.
    keep_all_mergers : :class:`bool`
        If True, always keep ``OBJTYPE in {'GPair', 'GTrpl'}`` sources
        first, even without a diameter (used when returning a group
        catalog without dropping merger-flagged sources).
    ignore_objtype : :class:`bool`
        If True, drop the ``OBJTYPE == 'G'`` requirement from the
        diameter/redshift-based masks, considering all objects
        regardless of type.

    Returns
    -------
    keep : :class:`numpy.ndarray`
        Indices into ``group`` of the object(s) to keep.
    drop : :class:`numpy.ndarray`
        Indices into ``group`` of the object(s) to drop.

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
    """Group closely-spaced objects in ``refcat`` (searching from the
    positions in ``cat``) and flag which member of each group is
    primary, via :func:`choose_primary`.

    For every group of 2+ ``refcat`` objects within ``maxsep`` of a
    ``cat`` position, assigns a shared ``GROUP_ID``, records
    ``NGROUP``/``SEPARATION`` (separation from the ``cat`` search
    center), and calls :func:`choose_primary` to decide which member(s)
    to keep (unless ``keep_all``, which keeps every member). A
    hardcoded single-entry veto list (currently just
    ``'2MASX J15134005+2607307'``, which overlaps 3C 315 due to a
    coordinate error in the latter) can restore an object to primary
    status regardless of :func:`choose_primary`'s decision, if
    ``allow_vetos`` is set.

    Parameters
    ----------
    cat : :class:`~astropy.table.Table`
        Smaller catalog of search centers; needs ``RA``/``DEC``.
    refcat : :class:`~astropy.table.Table`
        Full catalog to group/flag; needs ``RA``/``DEC``, ``OBJTYPE``,
        ``PGC``, ``Z``, ``DIAM_LIT``, ``DIAM_HYPERLEDA``, and
        ``objname_column``. Modified in place with the bookkeeping
        columns described below (removed again before returning if
        ``trim=True``).
    maxsep : :class:`float`
        Grouping search radius, arcsec.
    keep_all : :class:`bool`
        If True, keep every member of every group as primary (skip
        :func:`choose_primary` entirely).
    allow_vetos : :class:`bool`
        If True, restore hardcoded veto-listed objects to primary status
        even if :func:`choose_primary` dropped them.
    keep_all_mergers : :class:`bool`
        Passed to :func:`choose_primary`.
    objname_column : :class:`str`
        Column in ``refcat``/``cat`` holding each object's name, used
        for veto matching and verbose logging.
    ignore_objtype : :class:`bool`
        Passed to :func:`choose_primary`.
    trim : :class:`bool`
        If True, return only the primary objects, with the bookkeeping
        columns (``GROUP_ID``, ``PRIMARY``, ``NGROUP``, ``SEPARATION``,
        ``DONE``) removed. If False, return the full ``refcat`` with
        those columns intact (minus ``DONE``).
    verbose : :class:`bool`
        If True, print per-group keep/drop diagnostics as grouping
        proceeds.

    Returns
    -------
    :class:`~astropy.table.Table`
        ``refcat`` restricted to primary objects (if ``trim=True``) or
        the full, annotated ``refcat`` (if ``trim=False``).

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

    log.info(f'Found {len(ugroup):,d} group(s) with ' + \
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
