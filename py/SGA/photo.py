"""
SGA.photo
=========

Utilities for performing simple photometry.

Notes
-----
Not currently wired into any driver script (no callers in ``bin/`` or
``archive/``), and :func:`photo_one` (the per-object worker called by
:func:`do_photo`) cannot currently run: it imports ``from
SGA.find_galaxy import find_galaxy``, but ``py/SGA/find_galaxy.py``
was deliberately removed (commit ``5f9c698``, "remove
mge.find_galaxy") in favor of a different geometry-fitting approach
now used in :mod:`SGA.SGA`/:mod:`SGA.ellipse`; this import was never
updated here. This module appears to be an earlier, simpler
photometry prototype, superseded by the ellipse-fitting pipeline but
left in place.

"""
import pdb
import os, time, sys
import numpy as np
import fitsio
import multiprocessing
from astropy.table import Table, vstack

from SGA.cutouts import cutouts_plan
from SGA.SGA import sga2025_name
from SGA.logger import log


def photo_datamodel(out, ra, dec, diam, ba, pa, bands=['g', 'r', 'i', 'z']):
    """Initialize the simple-photometry output row for one object,
    adding the ``SGANAME`` column and per-band/per-stage placeholder
    columns (all filled with sentinel values, to be overwritten by
    :func:`photo_one`).

    Parameters
    ----------
    out : :class:`astropy.table.Table`
        Single-row input table (typically a subset of the parent
        catalog's columns for one object) to extend in place.
    ra : :class:`float`
        Right ascension, in decimal degrees.
    dec : :class:`float`
        Declination, in decimal degrees.
    diam : array-like
        Initial diameter estimate(s), in arcsec; only ``diam[0]`` is
        used, stored as ``DIAM_INIT``.
    ba : array-like
        Initial axis-ratio estimate(s); only ``ba[0]`` is used.
    pa : array-like
        Initial position-angle estimate(s), in degrees CCW from the
        y-axis; only ``pa[0]`` is used.
    bands : :class:`list`
        Bandpasses to add placeholder photometry columns for.

    Returns
    -------
    :class:`astropy.table.Table`
        `out`, extended in place and also returned for convenience,
        with ``SGANAME``, ``RA_PHOT``/``DEC_PHOT``, ``IN_GAIA``,
        ``NODATA``, ``CENTERMASKED``, ``MGE_FAIL``, ``SEP``,
        ``DIAM_INIT``/``BA_INIT``/``PA_INIT``,
        ``DIAM_PHOT``/``BA_PHOT``/``PA_PHOT``, and per-band
        ``FLUX_{INIT,PHOT}_{BAND}``, ``FLUX_{INIT,PHOT}_ERR_{BAND}``,
        ``GINI_{INIT,PHOT}_{BAND}``, ``FRACMASK_{INIT,PHOT}_{BAND}``
        columns, all initialized to ``-99.`` (or False, for the
        boolean flag columns).

    """
    out.add_column(sga2025_name(ra, dec, unixsafe=False)[0], name='SGANAME', index=0)
    for col in ['RA', 'DEC']:
        out[f'{col}_PHOT'] = [-99.]
    out['IN_GAIA'] = [False]
    out['NODATA'] = [False]
    out['CENTERMASKED'] = [False]
    out['MGE_FAIL'] = [False]
    out['SEP'] = [np.float32(-99.)]
    out['DIAM_INIT'] = diam[0].astype('f4') # [arcsec]
    out['BA_INIT'] = ba[0].astype('f4')
    out['PA_INIT'] = pa[0].astype('f4') # CCW from y-axis
    for col in ['DIAM', 'BA', 'PA']:
        out[f'{col}_PHOT'] = [np.float32(-99.)]
    for col in ['INIT', 'PHOT']:
        for band in bands:
            out[f'FLUX_{col}_{band.upper()}'] = [np.float32(-99.)]
        for band in bands:
            out[f'FLUX_{col}_ERR_{band.upper()}'] = [np.float32(-99.)]
        for band in bands:
            out[f'GINI_{col}_{band.upper()}'] = [np.float32(-99.)]
        for band in bands:
            out[f'FRACMASK_{col}_{band.upper()}'] = [np.float32(-99.)]
    return out


def qaplot_photo_one(qafile, jpgfile, out, ra, dec, pixscale, width,
                     diam, ba, pa, xyinit, wimg, wmask, wcs, xyphot=None,
                     xypeak=None, render_jpeg=True):
    """Build a single-panel QA figure for one object's simple
    photometry: the (masked) cutout image with the initial ellipse
    geometry overlaid in yellow and, if photometry succeeded, the
    derived (MGE-based) geometry overlaid in cyan.

    Parameters
    ----------
    qafile : :class:`str`
        Output QA PNG path.
    jpgfile : :class:`str`
        Path to the pre-rendered color JPEG cutout (used when
        `render_jpeg` is True).
    out : :class:`astropy.table.Table`
        Single-row output table (as built/updated by
        :func:`photo_datamodel`/:func:`photo_one`), used for the title
        text (``SGANAME``, ``OBJNAME``) and, if `xyphot` is given, the
        derived ``RA_PHOT``/``DEC_PHOT``/``DIAM_PHOT``/``BA_PHOT``/
        ``PA_PHOT`` columns.
    ra, dec : :class:`float`
        Object coordinates, in decimal degrees, for the title text.
    pixscale : :class:`float`
        Pixel scale, in arcsec/pixel.
    width : :class:`int`
        Cutout width/height, in pixels (square).
    diam, ba, pa : array-like
        Initial ellipse geometry (diameter in arcsec, axis ratio,
        position angle in degrees); only element 0 of each is used.
    xyinit : :class:`tuple`
        Initial (x, y) pixel position for the yellow ellipse.
    wimg : :class:`numpy.ndarray`
        Inverse-variance-weighted mean image, used only in the
        non-JPEG display branch (see Notes).
    wmask : :class:`numpy.ndarray`
        Boolean mask (True = masked), applied to the JPEG image before
        display.
    wcs : :class:`astropy.wcs.WCS`
        WCS of the cutout, used to project `out`'s derived RA/Dec to
        pixel coordinates when `xyphot` is given.
    xyphot : :class:`tuple`, optional
        If given, also overplot the derived (cyan) ellipse using
        `out`'s ``DIAM_PHOT``/``BA_PHOT``/``PA_PHOT`` columns.
    xypeak : :class:`tuple`, optional
        Unused (see Notes).
    render_jpeg : :class:`bool`
        If True, display the pre-rendered `jpgfile` (with `wmask`
        applied); if False, display a log-stretched `wimg` directly
        (see Notes for a bug in this branch).

    Returns
    -------
    None

    Notes
    -----
    `xypeak` is accepted but never referenced in the function body --
    dead parameter. The ``render_jpeg=False`` branch references
    `xpeak`/`ypeak` (no leading ``x``/``y`` prefix mismatch -- these
    exact names) which are never defined anywhere in this function
    (not parameters, not local variables) -- calling this function
    with ``render_jpeg=False`` always raises ``NameError``. In
    practice this is currently harmless: the sole caller
    (:func:`photo_one`) always calls with the default
    ``render_jpeg=True``, so the broken branch is never exercised.

    """
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    from SGA.qa import overplot_ellipse

    barlen = 15. / pixscale # [pixels]
    barlabel = '15 arcsec'

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, width)
    ax.set_ylim(0, width)
    if render_jpeg:
        jpg = mpimg.imread(jpgfile)
        jpgmask = np.flipud(~wmask)[:, :, np.newaxis]
        #jpgmask = np.flipud(~gaia_mask)[:, :, np.newaxis]
        #jpgmask = np.flipud(apmask_phot)[:, :, np.newaxis]
        im = ax.imshow(jpg * jpgmask, origin='lower', interpolation='nearest')
        ax.invert_yaxis() # JPEG is flipped relative to FITS
    else:
        #ax.imshow(img * apmask_phot, origin='lower', cmap=cmap)
        ax.imshow(np.log(wimg.clip(wimg[xpeak, ypeak]/1e4)) * ~wmask,
                  origin='lower', cmap='inferno', interpolation='nearest')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.margins(0)

    # these two sets of drawings should be identical
    #ap_init.plot(color='red', ls='-', lw=2, ax=ax)
    #for ap_phot in aps_phot:
    #    ap_phot.plot(color='black', ls='-', lw=2, ax=ax)
    overplot_ellipse(major_axis_arcsec=diam[0], ba=ba[0], pa=pa[0], x0=xyinit[0],
                     y0=xyinit[1], height_pixels=width, ax=ax, pixscale=pixscale,
                     color='yellow', linestyle='--', draw_majorminor_axes=True,
                     jpeg=render_jpeg)
    if xyphot is not None:
        xphot, yphot = wcs.wcs_world2pix(out['RA_PHOT'], out['DEC_PHOT'], 1)
        overplot_ellipse(major_axis_arcsec=out[f'DIAM_PHOT'], ba=out['BA_PHOT'],
                         pa=out['PA_PHOT'], x0=xphot, y0=yphot, height_pixels=width,
                         pixscale=pixscale, color='cyan', linestyle='-', linewidth=2,
                         ax=ax, draw_majorminor_axes=True, jpeg=render_jpeg)

    txt = '\n'.join([out['SGANAME'][0], out['OBJNAME'][0], f'{ra:.7f}, {dec:.6f}'])
    ax.text(0.03, 0.93, txt, transform=ax.transAxes, ha='left', va='center',
            color='white', bbox=dict(boxstyle='round', facecolor='k', alpha=0.5),
            linespacing=1.5, fontsize=10)

    # add the scale bar
    xpos, ypos = 0.07, 0.07
    dx = barlen / wimg.shape[0]
    ax.plot([xpos, xpos+dx], [ypos, ypos], transform=ax.transAxes,
            color='white', lw=2)
    ax.text(xpos + dx/2., ypos+0.02, barlabel, transform=ax.transAxes,
            ha='center', va='center', color='white')

    fig.tight_layout()
    fig.savefig(qafile, bbox_inches=0)#, dpi=200)
    plt.close()
    #log.info(f'Wrote {qafile}')


def _photo_one(args):
    """Unpack a single argument tuple and call :func:`photo_one` --
    the top-level callable required by ``multiprocessing.Pool.map``.

    Parameters
    ----------
    args : :class:`tuple`
        Positional arguments to forward to :func:`photo_one`.

    Returns
    -------
    Return value of :func:`photo_one`.

    """
    return photo_one(*args)


def photo_one(fitsfile, jpgfile, photfile, qafile, obj, survey,
              bands=['g', 'r', 'i', 'z'], box_arcsec=5.,
              verbose=False, qaplot=True):
    """Perform simple (non-ellipse-fitting) aperture photometry on a
    single multiband cutout: mask Gaia/Tycho stars, locate the galaxy
    center/geometry via :func:`SGA.find_galaxy.find_galaxy` (an MGE
    fit), then measure aperture flux, masked-pixel fraction, and Gini
    coefficient per band in both the initial and derived elliptical
    apertures.

    Parameters
    ----------
    fitsfile : :class:`str`
        Path to the multiband (image + inverse-variance) FITS cutout.
    jpgfile : :class:`str`
        Path to the pre-rendered color JPEG cutout, for QA.
    photfile : :class:`str`
        Output photometry FITS path.
    qafile : :class:`str`
        Output QA PNG path.
    obj : :class:`astropy.table.Row`
        Single-object catalog row with ``RA``, ``DEC``, ``OBJNAME``,
        and the geometry columns consumed by
        :func:`SGA.geometry.choose_geometry`.
    survey : :class:`legacypipe.survey.LegacySurveyData`
        Survey object used to look up Gaia/Tycho reference sources via
        ``legacypipe.reference.get_reference_sources``.
    bands : :class:`list`
        Bandpasses to measure.
    box_arcsec : :class:`float`
        Side length of a small box (centered on the initial position)
        used to test whether the object's center is fully masked.
    verbose : :class:`bool`
        Unused (see Notes).
    qaplot : :class:`bool`
        If True, write a QA figure via :func:`qaplot_photo_one` after
        a successful measurement.

    Returns
    -------
    None
        Writes `photfile` (and, depending on `qaplot`/failure mode,
        `qafile`) as a side effect; returns early (without raising) if
        the object's center is fully masked or the MGE geometry fit
        fails, in both cases still writing `photfile` with the
        appropriate ``CENTERMASKED``/``MGE_FAIL`` flag set.

    Raises
    ------
    ValueError
        If `fitsfile` is missing its inverse-variance extension, or if
        the WCS solution for the initial position is invalid (NaN).

    Notes
    -----
    Cannot currently run: imports ``from SGA.find_galaxy import
    find_galaxy``, but that module was removed from the package (see
    this module's docstring). `verbose` is accepted but never
    referenced in the function body. The center-masking fallback tries
    progressively looser masks (drop the faint-Gaia mask, then drop
    the bright-Gaia mask too) before giving up and flagging
    ``CENTERMASKED`` -- so a genuinely star-crowded field can still be
    measured with looser masking, which is not obviously indicated in
    the output.

    """
    from astropy.io import fits
    from astropy.wcs import WCS
    from astropy.wcs.utils import proj_plane_pixel_scales as get_pixscale
    from photutils.aperture import EllipticalAperture, CircularAperture
    from photutils.morphology import gini
    from astrometry.libkd.spherematch import match_radec
    from astrometry.util.starutil_numpy import arcsec_between
    from legacypipe.survey import wcs_for_brick, BrickDuck
    from legacypipe.reference import get_reference_sources, get_reference_map

    from SGA.geometry import choose_geometry
    from SGA.find_galaxy import find_galaxy
    from SGA.brick import custom_brickname


    def get_error(ivar):
        """Safely convert an inverse-variance image to a 1-sigma error
        image, leaving non-positive pixels at their original
        (non-positive) value rather than dividing by zero.

        Parameters
        ----------
        ivar : :class:`numpy.ndarray`
            Inverse-variance image.

        Returns
        -------
        :class:`numpy.ndarray`
            Error image, same shape as `ivar`.

        """
        error = ivar.copy()
        I = error > 0.
        error[I] = 1. / np.sqrt(error[I])
        return error

    def write_photfile(out, photfile):
        """Atomically write a single-row photometry table to disk, via
        a temp-file-then-rename, using :mod:`fitsio` instead of
        Astropy's writer (see Notes).

        Parameters
        ----------
        out : :class:`astropy.table.Table`
            Single-row output table to write.
        photfile : :class:`str`
            Destination FITS path.

        Returns
        -------
        None

        Notes
        -----
        Uses ``fitsio.write`` rather than ``out.write`` because, under
        MPI, Astropy 7.0.1's table writer hits
        https://github.com/astropy/astropy/issues/15350 .

        """
        #out.write(photfile, overwrite=True)
        fitsio.write(photfile+'.tmp', out.as_array(), clobber=True)
        os.rename(photfile+'.tmp', photfile)
        log.info(f'Wrote {photfile}')


    ra, dec = obj['RA'], obj['DEC']
    objname = sga2025_name(ra, dec, unixsafe=True)[0]

    diam, ba, pa, ref = choose_geometry(Table(obj), mindiam=15.)
    out = photo_datamodel(Table(obj['OBJNAME', 'STARFDIST', 'STARMAG', 'RA', 'DEC']),
                          ra, dec, diam, ba, pa, bands=bands)

    # read the data
    with fits.open(fitsfile) as H:
        if len(H) != 2:
            msg = f'{fitsfile} ({obj["OBJNAME"]}) is missing an inverse variance extension!'
            log.critical(msg)
            raise ValueError(msg)
        hdr = H[0].header
        imgs = H[0].data
        ivars = H[1].data

    # any data?
    if np.all(imgs == 0.):
        log.warning(f'No data for object {out["OBJNAME"][0]} = {out["SGANAME"][0]}')
        out['NODATA'] = True

    hdr['NAXIS'] = 2
    hdr.pop('NAXIS3')

    wcs = WCS(hdr, naxis=2)
    pixscale = get_pixscale(wcs)[0] * 3600. # [arcsec/pixel]
    minsb = 10.**(-0.4*(30.-22.5)) #/ pixscale**2.

    nband, height, width = imgs.shape

    # build the Gaia/Tycho mask
    gaia_mask = np.zeros((height, width), bool) # True=Gaia star(s)
    gaia_mask_faint = np.zeros_like(gaia_mask)

    brickname = f'custom-{custom_brickname(ra, dec)}'
    brick = BrickDuck(ra, dec, brickname)

    targetwcs = wcs_for_brick(brick, W=float(width), H=float(height), pixscale=pixscale)

    refstars = get_reference_sources(survey, targetwcs, pixscale, bands=bands,
                                     tycho_stars=True, gaia_stars=True,
                                     large_galaxies=False, star_clusters=False)
    refstars = refstars[0]
    if len(refstars) > 0:
        # remove Gaia stars within 5 arcsec of the initial coordinates
        m1, m2, _ = match_radec(refstars.ra, refstars.dec, ra, dec, 5./3600., nearest=True)
        if len(m1) > 0:
            out['IN_GAIA'] = True
            refstars.cut(np.delete(np.arange(len(refstars)), m1))

        # refmap contains just MEDIUM and BRIGHT stars
        refmap = get_reference_map(targetwcs, refstars)
        gaia_mask = np.logical_or(gaia_mask, refmap > 0) # True=Gaia star(s)

        # add fainter stars to the mask
        for star in refstars[refstars.in_bounds]:
            if star.radius_pix <= 0.:
                radius_pix = int(5. / pixscale)
            else:
                radius_pix = star.radius_pix
            ap = CircularAperture((star.ibx, star.iby), radius_pix) # note! (ibx,iby) not (iby,ibx)
            #gaia_mask = np.logical_or(gaia_mask, ap.to_mask().to_image((height, width)) != 0.) # object mask=True
            gaia_mask_faint = np.logical_or(gaia_mask_faint, ap.to_mask().to_image((height, width)) != 0.) # object mask=True

    # initial ellipse geometry and aperture
    xyinit = wcs.wcs_world2pix(ra, dec, 1)
    if np.any(np.isnan(xyinit)):
        msg = f'WCS problem analyzing {photfile}'
        log.critical(msg)
        raise ValueError(msg)
    a_init = diam[0] / 2. / pixscale # [pixels]
    b_init = a_init * ba[0]
    theta_init = np.radians(pa[0] - 90.) # CCW from x-axis

    ap_init = EllipticalAperture(xyinit, a=a_init, b=b_init, theta=theta_init)
    apmask_init = ap_init.to_mask().to_image((height, width)) != 0. # object mask=True

    # 5x5 arcsec box centered on initial position
    box_init = EllipticalAperture(xyinit, a=box_arcsec/2./pixscale,
                                  b=box_arcsec/2./pixscale, theta=0.)
    boxmask_init = box_init.to_mask().to_image((height, width)) != 0. # True=object mask

    # generate the ivar-weighted mean image; flag pixels that are
    # masked in all bandpasses
    wivar = np.sum(ivars, axis=0)
    wimg = np.sum(ivars * imgs, axis=0)
    I = wivar > 0.
    wimg[I] /= wivar[I]
    #wmask = np.sum(ivars <= 0., axis=0) == nband
    wmask = np.logical_or(np.sum(ivars <= 0., axis=0) == nband, gaia_mask, gaia_mask_faint)

    # if the center is fully masked, first try dropping the faint-Gaia
    # mask; if still fully masked, write out and move on
    if np.all(wimg[boxmask_init] == 0.) or np.all(wmask[boxmask_init]):
        wmask = np.logical_or(np.sum(ivars <= 0., axis=0) == nband, gaia_mask)
        if np.all(wimg[boxmask_init] == 0.) or np.all(wmask[boxmask_init]):
            wmask = np.sum(ivars <= 0., axis=0) == nband
            if np.all(wimg[boxmask_init] == 0.) or np.all(wmask[boxmask_init]):
                log.warning(f'Fully masked {box_arcsec:.1f}x{box_arcsec:.1f} arcsec ' + \
                            f'center {out["OBJNAME"][0]} = {out["SGANAME"][0]}')
                out['CENTERMASKED'] = True

                write_photfile(out, photfile)
                qaplot_photo_one(qafile, jpgfile, out, ra, dec, pixscale, width,
                                 diam, ba, pa, xyinit, wimg, wmask, wcs)
                return

    # compute the mean geometry

    ## photutils version which does not perform as well as find_galaxy
    #from photutils.morphology import data_properties
    ##from photutils.segmentation import SourceCatalog, SegmentationImage
    ##src = SourceCatalog(img, SegmentationImage(np.ones_like(img, int)), error=error, mask=mask)[0]
    #src = data_properties(img, mask=wmask)
    #xyphot = (src.xcentroid, src.ycentroid)
    #ba_phot = src.ellipticity.value
    #a_phot = src.semimajor_sigma.value # * 1.5
    #b_phot = a_phot * ba_phot
    #pa_phot = (360. - src.orientation.value) % 180 + 90. # CCW from y-axis
    #theta_phot = np.radians(pa_phot - 90.) # CCW from x-axis

    mge_fail = False
    try:
        mge = find_galaxy(wimg * ~wmask, binning=5, level=minsb, quiet=True)

        # In rare cases, find_galaxy will return invalid parameters, e.g.,
        # CGMW 4-1190. Capture those here and return.
        for param in ('xpeak', 'ypeak', 'xmed', 'ymed', 'majoraxis', 'eps', 'pa', 'theta'):
            if np.isnan(getattr(mge, param)):
                log.warning(f'Problem determing the geometry of {out["OBJNAME"][0]} = {out["SGANAME"][0]}')
                mge_fail = True
                break
    except:
        mge_fail = True
        #mge = find_galaxy(wimg * ~wmask, binning=5, level=minsb, quiet=False, plot=True)
        #import matplotlib.pyplot as plt
        #plt.clf() ; plt.imshow(wimg * ~wmask, origin='lower') ; plt.savefig('ioannis/tmp/junk2.png')

    if mge_fail:
        out['MGE_FAIL'] = True
        write_photfile(out, photfile)
        qaplot_photo_one(qafile, jpgfile, out, ra, dec, pixscale, width,
                         diam, ba, pa, xyinit, wimg, wmask, wcs)
        return


    xypeak = (mge.xpeak, mge.ypeak) # not swapped coordinates!
    xyphot = (mge.ymed, mge.xmed)   # swapped coordinates!
    a_phot = mge.majoraxis # * 1.5 # multiplicative factor? hack?? [pixels]
    ba_phot = 1. - mge.eps
    b_phot = a_phot * ba_phot
    pa_phot = mge.pa # CCW from y-axis
    theta_phot = np.radians((360. - mge.theta) % 180.) # convert from CW from x-axis to CCW from x-axis

    out['DIAM_PHOT'] = a_phot * pixscale # [arcsec]
    out['BA_PHOT'] = ba_phot
    out['PA_PHOT'] = pa_phot

    # aperture photometry in photometric ellipse
    ap_phot = EllipticalAperture(xyphot, a=a_phot, b=b_phot, theta=theta_phot)
    apmask_phot = ap_phot.to_mask().to_image((height, width)) != 0.

    ra_phot, dec_phot = wcs.all_pix2world(xyphot[0], xyphot[1], 1)
    out['RA_PHOT'] = ra_phot
    out['DEC_PHOT'] = dec_phot

    # separation between initial and final coordinates
    out['SEP'] = arcsec_between(ra, dec, ra_phot, dec_phot)

    # next loop on each bandpass
    for iband, band in enumerate(bands):
        img = imgs[iband, :, :]

        ivar = ivars[iband, :, :]
        mask = np.logical_or(ivar <= 0., wmask) # True=masked
        #mask = np.logical_or(ivar <= 0., gaia_mask) # True=masked
        error = get_error(ivar)

        # aperture photometry, fraction of masked pixels, and Gini
        # coefficient in the initial ellipse geometry
        flux_init, ferr_init = ap_init.do_photometry(img, error=error, mask=mask)
        fracmask_init = np.sum(mask[apmask_init]) / mask[apmask_init].size
        gini_init = gini(img * apmask_init, mask=mask)

        out[f'FLUX_INIT_{band.upper()}'] = flux_init[0]
        out[f'FLUX_INIT_ERR_{band.upper()}'] = ferr_init[0]
        out[f'FRACMASK_INIT_{band.upper()}'] = fracmask_init
        if np.isfinite(gini_init):
            out[f'GINI_INIT_{band.upper()}'] = gini_init

        # aperture photometry, fraction of masked pixels, and Gini
        # coefficient in the derived geometry
        flux_phot, ferr_phot = ap_phot.do_photometry(img, error=error, mask=mask)
        fracmask_phot = np.sum(mask[apmask_phot]) / mask[apmask_phot].size
        gini_phot = gini(img * apmask_phot, mask=mask)

        #out[f'FLUX_PHOT_{band.upper()}'] = 22.5 - 2.5 * np.log10(flux_phot[0])
        #out[f'FLUX_PHOT_ERR_{band.upper()}'] = ferr_phot[0] / flux_phot[0] / np.log(10.)
        out[f'FLUX_PHOT_{band.upper()}'] = flux_phot[0]
        out[f'FLUX_PHOT_ERR_{band.upper()}'] = ferr_phot[0]
        out[f'FRACMASK_PHOT_{band.upper()}'] = fracmask_phot
        if np.isfinite(gini_phot):
            out[f'GINI_PHOT_{band.upper()}'] = gini_phot

    #print(out.pprint(max_width=-1))
    #print(out[out.colnames[-8:]])
    write_photfile(out, photfile)

    # build QA
    if qaplot:
        qaplot_photo_one(qafile, jpgfile, out, ra, dec, pixscale, width,
                         diam, ba, pa, xyinit, wimg, wmask, wcs, xyphot=xyphot,
                         xypeak=xypeak)

def _read_one_photfile(args):
    """Unpack a single argument tuple and call :func:`read_one_photfile`
    -- the top-level callable required by ``multiprocessing.Pool.map``.

    Parameters
    ----------
    args : :class:`tuple`
        Positional arguments to forward to :func:`read_one_photfile`.

    Returns
    -------
    Return value of :func:`read_one_photfile`.

    """
    return read_one_photfile(*args)


def read_one_photfile(photfile):
    """Read a single per-object photometry FITS file written by
    :func:`photo_one` into a table.

    Parameters
    ----------
    photfile : :class:`str`
        Path to the photometry FITS file.

    Returns
    -------
    :class:`astropy.table.Table`
        Single-row photometry table.

    """
    return Table(fitsio.read(photfile))


def gather_photo(cat, mp=1, region='dr9-north', cutoutdir='.', photodir='.',
                 photo_version='v1.0'):
    """Gather the individual per-object photometry FITS files written
    by :func:`do_photo`/:func:`photo_one` into one merged catalog.

    Parameters
    ----------
    cat : :class:`astropy.table.Table`
        Parent catalog previously passed to :func:`do_photo`, used
        (via :func:`SGA.cutouts.cutouts_plan`) to recover the same set
        of per-object photometry filenames.
    mp : :class:`int`
        Number of multiprocessing workers used to read the individual
        files in parallel.
    region : :class:`str`
        Imaging region (e.g. ``'dr9-north'``), used only in the output
        catalog filename.
    cutoutdir : :class:`str`
        Unused here except as passed through to
        :func:`SGA.cutouts.cutouts_plan` (not itself referenced for
        gathering).
    photodir : :class:`str`
        Directory containing the per-object photometry FITS files.
    photo_version : :class:`str`
        Version string, used in the output catalog filename.

    Returns
    -------
    None

    Notes
    -----
    Refuses to overwrite an existing output catalog -- if
    ``{SGA_DIR}/parent/photo/parent-photo-{region}-{photo_version}.fits``
    already exists, logs a warning and returns without gathering
    anything (the caller must remove the file by hand first).

    """
    from SGA.SGA import sga_dir

    catdir = os.path.join(sga_dir(), 'parent', 'photo')
    if not os.path.isdir(catdir):
        os.makedirs(catdir, exist_ok=True)

    catfile = os.path.join(catdir, f'parent-photo-{region}-{photo_version}.fits')
    if os.path.isfile(catfile):
        log.warning(f'Existing photo catalog {catfile} must be removed by-hand.')
        return

    _, _, photfiles, _, groups = cutouts_plan(cat, size=1, photodir=photodir,
                                              mp=mp, gather_photo=True)
    indx = groups[0]

    # single-rank only but read in parallel
    mpargs = [(photfile, ) for photfile in photfiles[indx]]
    if mp > 1:
        with multiprocessing.Pool(mp) as P:
            out = P.map(_read_one_photfile, mpargs)
    else:
        out = [read_one_photfile(*mparg) for mparg in mpargs]

    if len(out) > 0:
        out = vstack(out)
        out.write(catfile, overwrite=True)
        log.info(f'Wrote photometry for {len(out):,d} objects to {catfile}')


def do_photo(cat, comm=None, mp=1, bands=['g', 'r', 'i', 'z'],
             region='dr9-north', cutoutdir='.', photodir='.',
             photo_version='v1.0', overwrite=False, verbose=False):
    """MPI/multiprocessing driver: plan cutout/photometry file paths
    for `cat` via :func:`SGA.cutouts.cutouts_plan`, distribute objects
    across MPI ranks, then call :func:`photo_one` (in parallel, if
    `mp` > 1) on this rank's share.

    Parameters
    ----------
    cat : :class:`astropy.table.Table`
        Parent catalog of objects to process.
    comm : :class:`mpi4py.MPI.Intracomm`, optional
        MPI communicator. If None, runs single-rank (``rank=0,
        size=1``).
    mp : :class:`int`
        Number of multiprocessing workers per rank.
    bands : :class:`list`
        Bandpasses to measure, passed through to :func:`photo_one`.
    region : :class:`str`
        Imaging region (e.g. ``'dr9-north'``), used to set the Legacy
        Survey data directory and select the ``RUNS`` survey.
    cutoutdir : :class:`str`
        Directory containing the multiband FITS/JPEG cutouts.
    photodir : :class:`str`
        Output directory for per-object photometry FITS/QA files.
    photo_version : :class:`str`
        Unused (see Notes).
    overwrite : :class:`bool`
        Passed through to :func:`SGA.cutouts.cutouts_plan`.
    verbose : :class:`bool`
        Passed through to :func:`SGA.cutouts.cutouts_plan`.

    Returns
    -------
    None

    Notes
    -----
    `photo_version` is accepted but never referenced in the function
    body -- dead parameter (unlike :func:`gather_photo`, which does
    use its own `photo_version` argument). Reference threshold cuts
    used for past photometry versions, preserved here for provenance::

        'dr11-south'
            # photo-version=v1.0
            I = (primaries['ROW_LVD'] == -99) * (primaries['STARFDIST'] > 1.) * (diam > 0.) * (diam < 10.) # N=4,434

            # photo-version=v1.1
            I = (primaries['ROW_LVD'] == -99) * (primaries['STARFDIST'] > 1.) * (diam >= 10.) * (diam < 12.) # N=434,092

        'dr9-north'
            # photo-version=v1.0
            #I = (primaries['ROW_LVD'] == -99) * (primaries['STARFDIST'] > 1.) * (diam > 0.) * (diam < 11.) # N=94,229

            # photo-version=v1.1
            I = (primaries['ROW_LVD'] == -99) * (primaries['STARFDIST'] > 1.) * (diam >= 11.) * (diam < 15.) # N=212,676

    """
    if comm is None:
        rank, size = 0, 1
    else:
        rank, size = comm.rank, comm.size

    if rank == 0:
        t0 = time.time()
        fitsfiles, jpgfiles, photfiles, qafiles, groups = cutouts_plan(
            cat, size=size, cutoutdir=cutoutdir, photodir=photodir,
            overwrite=overwrite, mp=mp, verbose=verbose, photo=True)
        log.info(f'Planning took {time.time() - t0:.2f} sec')
        #groups = np.array_split(range(len(cat)), size) # unweighted distribution
    else:
        fitsfiles, jpgfiles, photfiles, qafiles, groups = [], [], [], [], []

    if comm:
        fitsfiles = comm.bcast(fitsfiles, root=0)
        jpgfiles = comm.bcast(jpgfiles, root=0)
        photfiles = comm.bcast(photfiles, root=0)
        qafiles = comm.bcast(qafiles, root=0)
        groups = comm.bcast(groups, root=0)
        sys.stdout.flush()

    # all done
    if len(photfiles) == 0 or len(np.hstack(photfiles)) == 0:
        return

    assert(len(groups) == size)

    log.info(f'Rank {rank} started at {time.asctime()}')
    sys.stdout.flush()

    indx = groups[rank]
    if len(indx) == 0:
        return

    if rank == 0:
        from legacypipe.runs import get_survey
        from SGA.coadds import RUNS
        from SGA.io import set_legacysurvey_dir

        set_legacysurvey_dir(region)
        survey = get_survey(RUNS[region])
    else:
        survey = None

    if comm:
        survey = comm.bcast(survey, root=0)

    mpargs = [(fitsfiles[indx[iobj]], jpgfiles[indx[iobj]], photfiles[indx[iobj]],
               qafiles[indx[iobj]], cat[indx[iobj]], survey, bands) for iobj in range(len(indx))]
    if mp > 1:
        with multiprocessing.Pool(mp) as P:
            P.map(_photo_one, mpargs)
    else:
        [photo_one(*mparg) for mparg in mpargs]

    sys.stdout.flush()

    #if comm is not None:
    #    comm.barrier()

    if rank == 0:
        log.info(f'All done at {time.asctime()}')
