"""
SGA.misc
========

Miscellaneous utility code used by various scripts.

"""
import os, sys
import numpy as np


def viewer_inspect(cat, galaxycolname='GALAXY'):
    """Write a small FITS catalog of names and coordinates that can be
    uploaded to the Legacy Survey viewer's "Load Catalog" tool for
    visual inspection.

    Parameters
    ----------
    cat : :class:`astropy.table.Table`
        Catalog with, at minimum, columns `galaxycolname`, ``RA``, and
        ``DEC``.
    galaxycolname : :class:`str`
        Name of the column in `cat` to rename to ``NAME`` in the
        output catalog.

    Returns
    -------
    None

    Notes
    -----
    Always writes to ``$HOME/tmp/viewer.fits``, overwriting any
    existing file; the output path is not configurable.

    """
    out = cat[galaxycolname, 'RA', 'DEC']
    out.rename_column(galaxycolname, 'NAME')
    outfile = os.path.join(os.getenv('HOME'), 'tmp', 'viewer.fits')
    print('Writing {} objects to {}'.format(len(cat), outfile))
    out.write(outfile, overwrite=True)


def imagetool_inspect(cat, group=False):
    """Write a small whitespace-delimited catalog of names and
    coordinates that can be uploaded to the DECaLS image-list tool
    (https://yymao.github.io/decals-image-list-tool/) for visual
    inspection.

    Parameters
    ----------
    cat : :class:`astropy.table.Table`
        Catalog to write out. If `group` is False, must contain
        ``RA``, ``DEC``, and either ``GALAXY`` or ``NAME`` (used for
        the object name); if the chosen name column has empty
        entries, falls back to ``ALTNAME`` (with spaces stripped), or
        finally the literal string ``'galaxy'``. If `group` is True,
        must instead contain ``GROUP_NAME``, ``GROUP_RA``, and
        ``GROUP_DEC``.
    group : :class:`bool`
        If True, write group-level columns (``GROUP_NAME``,
        ``GROUP_RA``, ``GROUP_DEC``) instead of per-galaxy columns.

    Returns
    -------
    None

    Notes
    -----
    Always writes to ``$HOME/tmp/inspect.txt``, overwriting any
    existing file; the output path is not configurable. When `group`
    is True, the empty-name fallback to ``ALTNAME`` is skipped
    (``ALTNAME`` is a per-galaxy, not per-group, column).

    """
    if group:
        galcol, racol, deccol = 'GROUP_NAME', 'GROUP_RA', 'GROUP_DEC'
    else:
        racol, deccol = 'RA', 'DEC'
        galcol = 'GALAXY'
        if not galcol in cat.colnames:
            galcol = 'NAME'
        
    outfile = os.path.join(os.getenv('HOME'), 'tmp', 'inspect.txt')
    print('Writing {} objects to {}'.format(len(cat), outfile))
    with open(outfile, 'w') as ff:
        ff.write('name ra dec\n')
        for ii, (gal, ra, dec) in enumerate(zip(cat[galcol], cat[racol], cat[deccol])):
            if gal.strip() == '':
                if 'ALTNAME' in cat.colnames:
                    gal = cat['ALTNAME'][ii].strip().replace(' ', '')
                    if gal == '':
                        gal = 'galaxy'
                else:
                    gal = 'galaxy'
            ff.write('{} {:.6f} {:.6f}\n'.format(gal, ra, dec))


def simple_wcs(onegal, radius=None, factor=1.0, pixscale=0.262, zcolumn='Z'):
    '''Build a simple tangent-plane (TAN) WCS centered on a single
    galaxy.

    Parameters
    ----------
    onegal : :class:`astropy.table.Row` or similar
        Single-object record with ``RA``, ``DEC`` and, optionally,
        `zcolumn`.
    radius : :class:`float`, optional
        Half-width of the desired image footprint, in pixels. If not
        given and `zcolumn` is present in `onegal`, computed from the
        redshift (see Notes); otherwise defaults to 100 pixels.
    factor : :class:`float`
        Multiplicative scale factor applied to (twice) `radius` to
        set the final image diameter.
    pixscale : :class:`float`
        Pixel scale, in arcsec/pixel.
    zcolumn : :class:`str`
        Name of the redshift column in `onegal` used to derive
        `radius` when `radius` is not given.

    Returns
    -------
    :class:`astrometry.util.util.Tan`
        WCS object centered on `onegal`'s coordinates, with image
        dimensions ``diam x diam`` pixels.

    Notes
    -----
    The `radius`-from-redshift branch calls a function
    ``cutout_radius_kpc`` that is not defined or imported anywhere in
    this module (nor anywhere else in the package) -- if `radius` is
    left as ``None`` and `zcolumn` is present in `onegal`, this raises
    ``NameError``. This function also appears unused elsewhere in the
    codebase; :func:`SGA.sky.simple_wcs` is a separate, actively-used
    implementation of the same idea.

    '''
    from astrometry.util.util import Tan

    if radius is None:
        if zcolumn in onegal.colnames:
            radius = 2 * cutout_radius_kpc(redshift=onegal[zcolumn], pixscale=pixscale)
        else:
            radius = 100 # hack! [pixels]

    diam = np.ceil(factor * 2 * radius).astype('int') # [pixels]
    simplewcs = Tan(onegal['RA'], onegal['DEC'], diam/2+0.5, diam/2+0.5,
                    -pixscale/3600.0, 0.0, 0.0, pixscale/3600.0, 
                    float(diam), float(diam))
    return simplewcs


def ccdwcs(ccd):
    '''Build a tangent-plane (TAN) WCS object from a single CCD's
    header keywords.

    Parameters
    ----------
    ccd : object
        Single-CCD record exposing ``width``, ``height``, ``crval1``,
        ``crval2``, ``crpix1``, ``crpix2``, ``cd1_1``, ``cd1_2``,
        ``cd2_1``, ``cd2_2`` attributes (e.g. a row from a CCDs
        table).

    Returns
    -------
    W : :class:`int`
        CCD width, in pixels.
    H : :class:`int`
        CCD height, in pixels.
    ccdwcs : :class:`astrometry.util.util.Tan`
        WCS object built from `ccd`'s header keywords.

    '''
    from astrometry.util.util import Tan

    W, H = ccd.width, ccd.height
    ccdwcs = Tan(*[float(xx) for xx in [ccd.crval1, ccd.crval2, ccd.crpix1,
                                        ccd.crpix2, ccd.cd1_1, ccd.cd1_2,
                                        ccd.cd2_1, ccd.cd2_2, W, H]])
    return W, H, ccdwcs


def arcsec2kpc(redshift, cosmo=None):
    """Compute the scale factor to convert a physical axis in
    arcseconds to kpc, at a given redshift.

    Parameters
    ----------
    redshift : :class:`float` or array-like
        Redshift(s) at which to evaluate the angular-diameter-distance
        based conversion.
    cosmo : :class:`astropy.cosmology.FLRW`, optional
        Cosmology to use. Must be supplied by the caller; there is no
        internal default (see Notes).

    Returns
    -------
    :class:`float` or array-like
        Conversion factor, in kpc/arcsec, such that a physical size
        in arcsec times this factor gives the size in kpc.

    Notes
    -----
    Despite the ``cosmo=None`` default, this function does not
    construct a fallback cosmology -- the line that once did so
    (``cosmo = cosmology()``) is commented out, and no cosmology
    helper by that name is imported in this module. If called without
    an explicit `cosmo`, ``cosmo.arcsec_per_kpc_proper(...)`` raises
    ``AttributeError`` on the ``None`` default. At least one call site
    (``SGA.qa``) invokes this function without passing `cosmo`.

    """
    return 1 / cosmo.arcsec_per_kpc_proper(redshift).value # [kpc/arcsec]


def statsinbins(xx, yy, binsize=0.1, minpts=10, xmin=None, xmax=None):
    """Compute per-bin statistics (mean, median, standard deviation,
    25th/75th percentiles) of `yy` in fixed-width bins along `xx`.

    Parameters
    ----------
    xx : array-like
        Independent-variable values used to assign bins.
    yy : array-like
        Dependent-variable values to summarize within each bin.
    binsize : :class:`float`
        Bin width, in the same units as `xx`; used only to set the
        number of bins (``nbin``), not the bin edges themselves (see
        Notes).
    minpts : :class:`int`
        Minimum number of points a bin must contain to be kept in the
        output.
    xmin : :class:`float`, optional
        Lower edge of the binning range. Defaults to ``xx.min()``.
    xmax : :class:`float`, optional
        Upper edge of the binning range. Defaults to ``xx.max()``.

    Returns
    -------
    :class:`numpy.ndarray` or None
        Structured array with fields ``xmean``, ``xmedian``, ``xbin``,
        ``npts``, ``ymedian``, ``ymean``, ``ystd``, ``y25``, ``y75``,
        restricted to bins with ``npts > minpts``. Returns None if no
        bin meets that threshold.

    Notes
    -----
    ``nbin`` is computed from ``(nanmax(xx) - nanmin(xx)) / binsize``
    (i.e. always spans the full data range of `xx`), but the actual
    bin edges (``_xbin = np.linspace(xmin, xmax, nbin)``) span
    `xmin`/`xmax` instead -- if `xmin`/`xmax` are supplied and differ
    from `xx`'s true range, `binsize` no longer accurately describes
    the resulting bin width. ``npts`` per bin is computed via
    ``np.count_nonzero(yy[these])``, which counts nonzero *values* of
    `yy` in the bin, not the number of points in the bin -- a bin
    containing only ``yy == 0`` entries is reported as ``npts == 0``
    and dropped by the `minpts` cut. A dead ``if False:`` branch
    (roughly half the function body) duplicates the live per-bin
    logic using ``scipy.stats.binned_statistic`` instead of
    ``np.digitize``; it is unreachable and appears to be an earlier,
    abandoned implementation.

    """
    from scipy.stats import binned_statistic

    if xmin == None:
        xmin = xx.min()
    if xmax == None:
        xmax = xx.max()

    nbin = int( (np.nanmax(xx) - np.nanmin(xx) ) / binsize )
    stats = np.zeros(nbin, [('xmean', 'f4'), ('xmedian', 'f4'), ('xbin', 'f4'),
                            ('npts', 'i4'), ('ymedian', 'f4'), ('ymean', 'f4'),
                            ('ystd', 'f4'), ('y25', 'f4'), ('y75', 'f4')])

    if False:
        def median(x):
            """Dead code (unreachable ``if False:`` branch); nan-safe median."""
            return np.nanmedian(x)

        def mean(x):
            """Dead code (unreachable ``if False:`` branch); nan-safe mean."""
            return np.nanmean(x)

        def std(x):
            """Dead code (unreachable ``if False:`` branch); nan-safe standard deviation."""
            return np.nanstd(x)

        def q25(x):
            """Dead code (unreachable ``if False:`` branch); nan-safe 25th percentile."""
            return np.nanpercentile(x, 25)

        def q75(x):
            """Dead code (unreachable ``if False:`` branch); nan-safe 75th percentile."""
            return np.nanpercentile(x, 75)

        ystat, bin_edges, _ = binned_statistic(xx, yy, bins=nbin, statistic='median')
        stats['median'] = ystat

        bin_width = (bin_edges[1] - bin_edges[0])
        xmean = bin_edges[1:] - bin_width / 2

        ystat, _, _ = binned_statistic(xx, yy, bins=nbin, statistic='mean')
        stats['mean'] = ystat

        ystat, _, _ = binned_statistic(xx, yy, bins=nbin, statistic=std)
        stats['std'] = ystat

        ystat, _, _ = binned_statistic(xx, yy, bins=nbin, statistic=q25)
        stats['q25'] = ystat

        ystat, _, _ = binned_statistic(xx, yy, bins=nbin, statistic=q75)
        stats['q75'] = ystat

        keep = (np.nonzero( stats['median'] ) * np.isfinite( stats['median'] ))[0]
        xmean = xmean[keep]
        stats = stats[keep]
    else:
        _xbin = np.linspace(xmin, xmax, nbin)
        idx  = np.digitize(xx, _xbin)

        for kk in range(nbin):
            these = idx == kk
            npts = np.count_nonzero( yy[these] )

            stats['xbin'][kk] = _xbin[kk]
            stats['npts'][kk] = npts

            if npts > 0:
                stats['xmedian'][kk] = np.nanmedian( xx[these] )
                stats['xmean'][kk] = np.nanmean( xx[these] )

                stats['ystd'][kk] = np.nanstd( yy[these] )
                stats['ymean'][kk] = np.nanmean( yy[these] )

                qq = np.nanpercentile( yy[these], [25, 50, 75] )
                stats['y25'][kk] = qq[0]
                stats['ymedian'][kk] = qq[1]
                stats['y75'][kk] = qq[2]

        keep = stats['npts'] > minpts
        if np.count_nonzero(keep) == 0:
            return None
        else:
            return stats[keep]


