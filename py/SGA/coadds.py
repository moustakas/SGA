"""
==========
SGA.coadds
==========

"""
import os, pdb
import numpy as np
from glob import glob

from SGA.logger import log


PIXSCALE = 0.262
GALEX_PIXSCALE = 1.5
UNWISE_PIXSCALE = 2.75

RUNS = {
    'dr9-north': 'north',
    'dr9-south': 'south',
    'dr10-south': 'south',
    'dr11-north': 'north',
    'dr11-south': 'south'
}

# although dr9-north is missing i-band imaging, there are many
# advantages to adopting a consistent data model
GRZ = ['g', 'r', 'z']
GRIZ = ['g', 'r', 'i', 'z']
BANDS = {
    'dr9-north': GRIZ,
    'dr9-south': GRIZ,
    'dr10-south': GRIZ,
    'dr11-north': GRIZ,
    'dr11-south': GRIZ
}

RELEASE = {
    'dr9-north': 9011,
    'dr9-south': 9010,
    'dr10-south': 10000,
    'dr11-north': 11001,
    'dr11-south': 11000
}

REGIONBITS = {
    'dr11-south': 2**0,
    'dr11-north': 2**1,
    'dr9-north': 2**1
}


def srcs2image(cat, wcs, band='r', pixelized_psf=None, psf_sigma=1., patches=None):
    """Render a Tractor model image from a source catalog.

    Builds a blank :class:`tractor.Image` with the given WCS and PSF and
    renders every source in ``cat`` into it via :class:`tractor.Tractor`.
    If pre-rendered ``patches`` are supplied, skips Tractor rendering
    entirely and instead sums the patches directly into a blank image
    (much faster when the same sources are rendered repeatedly, e.g. once
    per candidate mask).

    Parameters
    ----------
    cat : :class:`~astrometry.util.fits.tabledata` or :class:`list`
        Tractor sources to render. If a ``tabledata`` instance (e.g. a
        raw Tractor catalog), sources are built via
        ``legacypipe.catalog.read_fits_catalog``; otherwise treated
        directly as a list of Tractor source objects.
    wcs : WCS
        WCS of the mosaic to render into (a
        ``tractor.wcs.ConstantFitsWcs``, ``legacypipe.survey.LegacySurveyWcs``,
        or plain WCS object); its shape sets the output image shape.
    band : :class:`str`
        Filter name (lower-cased internally) used for photometric
        calibration and to select each source's flux.
    pixelized_psf : PSF, optional
        Pixelized PSF used to render sources. If None, falls back to a
        circular Gaussian PSF with sigma ``psf_sigma``.
    psf_sigma : :class:`float`
        Gaussian PSF sigma, in pixels, used only when ``pixelized_psf``
        is None.
    patches : :class:`list` of Patch or None, optional
        Pre-rendered per-source model patches, row-matched to ``cat``.
        If given, Tractor rendering is skipped and the patches are
        summed directly into the output image (entries may be None).

    Returns
    -------
    :class:`numpy.ndarray`
        Rendered model image, shape of ``wcs``.

    """
    from astrometry.util.fits import tabledata
    from tractor import Image, GaussianMixturePSF, Tractor
    from tractor.basics import LinearPhotoCal
    from tractor.wcs import ConstantFitsWcs
    from tractor.sky import ConstantSky
    from legacypipe.catalog import read_fits_catalog
    from legacypipe.survey import LegacySurveyWcs

    if type(wcs) is ConstantFitsWcs or type(wcs) is LegacySurveyWcs:
        shape = wcs.wcs.shape
    else:
        shape = wcs.shape
    model = np.zeros(shape)

    if patches is not None:
        for patch in patches:
            if patch is not None:
                patch.addTo(model)
        return model

    invvar = np.ones(shape)

    if pixelized_psf is None:
        vv = psf_sigma**2.
        psf = GaussianMixturePSF(1., 0., 0., vv, vv, 0.)
    else:
        psf = pixelized_psf

    photocal = LinearPhotoCal(1., band=band.lower())
    tim = Image(model, invvar=invvar, wcs=wcs, psf=psf,
                photocal=photocal, sky=ConstantSky(0.),
                name=f'model-{band}')

    if type(cat) is tabledata:
        srcs = read_fits_catalog(cat, bands=[band.lower()])
    else:
        srcs = cat

    mod = Tractor([tim], srcs).getModelImage(0)

    return mod


def _mosaic_width(radius_mosaic_arcsec, pixscale=PIXSCALE):
    """Convert a mosaic radius to a pixel width, forced odd so the
    mosaic center lands on a whole pixel (important for ellipse-fitting).

    Parameters
    ----------
    radius_mosaic_arcsec : :class:`float`
        Mosaic radius, in arcsec.
    pixscale : :class:`float`
        Pixel scale, arcsec/pixel.

    Returns
    -------
    :class:`int`
        Mosaic width, in pixels; always odd.

    """
    width = 2 * radius_mosaic_arcsec / pixscale # [pixels]
    width = (np.ceil(width) // 2 * 2 + 1).astype('int') # [pixels]
    return width


def _rearrange_files(galaxy, output_dir, brickname, bands=GRIZ, unwise=True,
                     galex=False, cleanup=False, just_coadds=False,
                     clobber=False, missing_ok=False):
    """Copy legacypipe's per-brick coadd/Tractor/PSF products into the
    SGA per-galaxy file-naming convention, optionally cleaning up the
    raw legacypipe output tree afterward.

    legacypipe writes its outputs under
    ``{output_dir}/coadd/cus/{brickname}/legacysurvey-{brickname}-*``,
    ``{output_dir}/tractor/cus/tractor-{brickname}.fits``, etc.; this
    function copies (see Notes) each expected product into
    ``{output_dir}/{galaxy}-*`` so downstream SGA code
    (e.g. ``SGA.SGA.read_multiband``) can find it by galaxy/group name.
    If the per-brick CCDs file is entirely absent and ``missing_ok`` is
    False, legacypipe found no CCDs touching the brick; this is treated
    as an expected early return (optionally still running cleanup)
    rather than a failure. Copies proceed in stages -- CCDs table,
    optical image/invvar FITS, image JPG (stop here if
    ``just_coadds``), per-band PSFs (optical, then unWISE/GALEX if
    requested), Tractor catalog, maskbits, model FITS, model/resid JPGs,
    then the unWISE and GALEX image/invvar/model FITS and preview JPGs
    -- and each stage aborts the whole function (without cleanup) on
    the first hard copy failure. GALEX image/invvar copies always pass
    ``missing_ok=True`` regardless of the caller's setting, since GALEX
    coverage can be entirely absent around bright stars.

    Notes
    -----
    Despite this docstring's original "move (rename)" description, files
    are copied, not moved: the source file under ``coadd/``/``tractor/``
    is left in place (copied to a temp file at the destination, then
    ``os.rename``d into place) and is only removed if ``cleanup=True``
    deletes the whole raw output tree afterward.

    Parameters
    ----------
    galaxy : :class:`str`
        Galaxy or group name; the filename prefix for every copied
        per-object data product.
    output_dir : :class:`str`
        Directory containing both legacypipe's raw ``coadd``/``tractor``/
        ``metrics`` subdirectories and the destination for the renamed
        per-galaxy files.
    brickname : :class:`str`
        Custom brick name (e.g. from ``SGA.brick.custom_brickname``) used
        in legacypipe's internal file paths.
    bands : :class:`list` of :class:`str`
        Optical bands to copy image/invvar/PSF/model products for.
    unwise : :class:`bool`
        If True, also copy the unWISE W1-W4 PSFs, image/invvar/model
        FITS, and preview JPGs.
    galex : :class:`bool`
        If True, also copy the GALEX FUV/NUV PSFs, image/invvar/model
        FITS, and preview JPGs (image/invvar copies always tolerate
        missing files).
    cleanup : :class:`bool`
        If True, remove the raw ``coadd``/``metrics``/``tractor``/
        ``tractor-i`` directories and any ``{galaxy}-*.p`` pickle files
        under ``output_dir`` after copying (or immediately, in the
        no-CCDs early-return case).
    just_coadds : :class:`bool`
        If True, stop after copying the optical image/invvar FITS and
        image JPG -- skip PSFs, Tractor catalog, maskbits, models, and
        unWISE/GALEX products.
    clobber : :class:`bool`
        If True, overwrite existing destination files; if False and a
        destination file already exists, it is left as-is and treated
        as already copied.
    missing_ok : :class:`bool`
        If True, treat a missing source file as non-fatal (log a
        warning and continue) instead of aborting; also, if True, skip
        the early-return "no CCDs touching brick" short-circuit and
        attempt every copy stage regardless (each guarded by its own
        ``missing_ok`` handling).

    Returns
    -------
    :class:`int`
        ``1`` on success (including the "no CCDs touching brick" and
        ``just_coadds`` early-return cases); ``0`` if a required file
        copy failed (``missing_ok=False`` and the source file is
        missing).

    """
    import fitsio
    import shutil

    def _copyfile(infile, outfile, clobber=False, update_header=False, missing_ok=False):
        """Copy a single file into place via a temp file + atomic
        rename, tolerating a missing source or an already-existing
        destination.

        Notes
        -----
        ``update_header`` is accepted but unused -- the branch guarded
        by it (``if update_header: pass``) is a no-op, so passing
        ``update_header=True`` has no effect on the copied file.

        Parameters
        ----------
        infile : :class:`str`
            Source file path.
        outfile : :class:`str`
            Destination file path.
        clobber : :class:`bool`
            If False and ``outfile`` already exists, skip the copy and
            report success without touching it.
        update_header : :class:`bool`
            Unused (see Notes).
        missing_ok : :class:`bool`
            If True and ``infile`` doesn't exist, log a warning and
            report success rather than failure.

        Returns
        -------
        :class:`int`
            ``1`` on success (copied, already present, or tolerated
            missing source); ``0`` if ``infile`` is missing and
            ``missing_ok`` is False.

        """
        if os.path.isfile(outfile) and not clobber:
            return 1
        if os.path.isfile(infile):
            tmpfile = outfile+'.tmp'
            shutil.copyfile(infile, tmpfile)
            os.rename(tmpfile, outfile)
            if update_header:
                pass
            return 1
        else:
            if missing_ok:
                log.warning(f'Missing file {infile} but missing_ok=True')
                return 1
            else:
                log.warning(f'Missing file {infile}; please check the logfile.')
                return 0

    def _do_cleanup():
        """Remove legacypipe's raw per-brick output directories and any
        leftover per-galaxy pickle files under ``output_dir``.

        """
        import shutil
        shutil.rmtree(os.path.join(output_dir, 'coadd'), ignore_errors=True)
        shutil.rmtree(os.path.join(output_dir, 'metrics'), ignore_errors=True)
        shutil.rmtree(os.path.join(output_dir, 'tractor'), ignore_errors=True)
        shutil.rmtree(os.path.join(output_dir, 'tractor-i'), ignore_errors=True)
        picklefiles = glob(os.path.join(output_dir, f'{galaxy}-*.p'))
        for picklefile in picklefiles:
            if os.path.isfile(picklefile):
                os.remove(picklefile)

    # If we made it here and there is no CCDs file it's because legacypipe
    # exited cleanly with "No photometric CCDs touching brick."
    ccdsfile = os.path.join(output_dir, 'coadd', 'cus', brickname,
                            f'legacysurvey-{brickname}-ccds.fits')
    if not os.path.isfile(ccdsfile) and missing_ok is False:
        log.info('No CCDs touching this brick; nothing to do.')
        if cleanup:
            _do_cleanup()
        return 1

    ccdsfile = os.path.join(output_dir, f'{galaxy}-ccds.fits')
    ok = _copyfile(
        os.path.join(output_dir, 'coadd', 'cus', brickname,
                     f'legacysurvey-{brickname}-ccds.fits'), ccdsfile,
        clobber=clobber, missing_ok=missing_ok)
    if not ok:
        return ok

    ## For objects on the edge of the footprint we can sometimes lose 3-band
    ## coverage if one of the bands is fully masked. Check here and write out all
    ## the files except a
    #if os.path.isfile(ccdsfile): # can be missing during testing if missing_ok=True
    #    allbands = fitsio.read(ccdsfile, columns='filter')
    #    ubands = list(sorted(set(allbands)))

    # image coadds (FITS + JPG)
    for band in bands:
        for imtype, outtype in zip(('image', 'invvar'), ('image', 'invvar')):
            ok = _copyfile(
                os.path.join(output_dir, 'coadd', 'cus', brickname,
                             f'legacysurvey-{brickname}-{imtype}-{band}.fits.fz'),
                             os.path.join(output_dir, f'{galaxy}-{outtype}-{band}.fits.fz'),
                 clobber=clobber, missing_ok=missing_ok, update_header=True)
            if not ok:
                return ok

    # JPG images
    ok = _copyfile(
        os.path.join(output_dir, 'coadd', 'cus', brickname,
                     f'legacysurvey-{brickname}-image.jpg'),
        os.path.join(output_dir, f'{galaxy}-image.jpg'),
        clobber=clobber, missing_ok=missing_ok)
    if not ok:
        return ok

    if just_coadds:
        if cleanup:
            _do_cleanup()
        return 1

    # PSFs
    for band in bands:
        for imtype, outtype in zip(['copsf'], ['psf']):
            ok = _copyfile(
                os.path.join(output_dir, 'coadd', 'cus', brickname,
                             f'legacysurvey-{brickname}-{imtype}-{band}.fits.fz'),
                             os.path.join(output_dir, f'{galaxy}-{outtype}-{band}.fits.fz'),
                clobber=clobber, missing_ok=missing_ok)
            if not ok:
                return ok

    if unwise:
        for band in ['W1', 'W2', 'W3', 'W4']:
            for imtype, outtype in zip(['copsf'], ['psf']):
                ok = _copyfile(
                    os.path.join(output_dir, 'coadd', 'cus', brickname,
                                 f'legacysurvey-{brickname}-{imtype}-{band}.fits.fz'),
                                 os.path.join(output_dir, f'{galaxy}-{outtype}-{band}.fits.fz'),
                    clobber=clobber, missing_ok=missing_ok)
                if not ok:
                    return ok

    if galex:
        for band in ['FUV', 'NUV']:
            for imtype, outtype in zip(['copsf'], ['psf']):
                ok = _copyfile(
                    os.path.join(output_dir, 'coadd', 'cus', brickname,
                                 f'legacysurvey-{brickname}-{imtype}-{band}.fits.fz'),
                                 os.path.join(output_dir, f'{galaxy}-{outtype}-{band}.fits.fz'),
                    clobber=clobber, missing_ok=missing_ok)
                if not ok:
                    return ok

    # tractor catalog
    ok = _copyfile(
        os.path.join(output_dir, 'tractor', 'cus', f'tractor-{brickname}.fits'),
        os.path.join(output_dir, f'{galaxy}-tractor.fits'),
        clobber=clobber, missing_ok=missing_ok)
    if not ok:
        return ok

    # Maskbits, blob images, outlier masks, and depth images.
    ok = _copyfile(
        os.path.join(output_dir, 'coadd', 'cus', brickname,
                     f'legacysurvey-{brickname}-maskbits.fits.fz'),
        os.path.join(output_dir, f'{galaxy}-maskbits.fits.fz'),
        clobber=clobber, missing_ok=missing_ok)
    if not ok:
        return ok

    # model coadds
    for band in bands:
        for imtype in ['model']:
        #for imtype in ('model', 'blobmodel'):
            ok = _copyfile(
                os.path.join(output_dir, 'coadd', 'cus', brickname,
                             f'legacysurvey-{brickname}-{imtype}-{band}.fits.fz'),
                os.path.join(output_dir, f'{galaxy}-{imtype}-{band}.fits.fz'),
                clobber=clobber, missing_ok=missing_ok)
            if not ok:
                return ok

    # JPG images
    for imtype in ('model', 'resid'):
        ok = _copyfile(
            os.path.join(output_dir, 'coadd', 'cus', brickname,
                         f'legacysurvey-{brickname}-{imtype}.jpg'),
            os.path.join(output_dir, f'{galaxy}-{imtype}.jpg'),
            clobber=clobber, missing_ok=missing_ok)
        if not ok:
            return ok

    # WISE
    if unwise:
        for band in ('W1', 'W2', 'W3', 'W4'):
            for imtype in ('image', 'invvar'):
                ok = _copyfile(
                    os.path.join(output_dir, 'coadd', 'cus', brickname,
                                 f'legacysurvey-{brickname}-{imtype}-{band}.fits.fz'),
                    os.path.join(output_dir, f'{galaxy}-{imtype}-{band}.fits.fz'),
                    clobber=clobber, missing_ok=missing_ok)
                if not ok:
                    return ok

            ok = _copyfile(
                os.path.join(output_dir, 'coadd', 'cus', brickname,
                             f'legacysurvey-{brickname}-model-{band}.fits.fz'),
                os.path.join(output_dir, f'{galaxy}-model-{band}.fits.fz'),
                    clobber=clobber, missing_ok=missing_ok)
            if not ok:
                return ok

        for imtype, suffix in zip(('wise', 'wisemodel', 'wiseresid'), ('image', 'model', 'resid')):
            ok = _copyfile(
                os.path.join(output_dir, 'coadd', 'cus', brickname,
                             f'legacysurvey-{brickname}-{imtype}.jpg'),
                os.path.join(output_dir, f'{galaxy}-{suffix}-W1W2.jpg'),
                    clobber=clobber, missing_ok=missing_ok)
            if not ok:
                return ok

    if galex:
        # GALEX imaging can be missing completely around bright stars, so don't fail.
        for band in ('FUV', 'NUV'):
            for imtype in ('image', 'invvar'):
                ok = _copyfile(
                    os.path.join(output_dir, 'coadd', 'cus', brickname,
                                 f'legacysurvey-{brickname}-{imtype}-{band}.fits.fz'),
                    os.path.join(output_dir, f'{galaxy}-{imtype}-{band}.fits.fz'),
                    clobber=clobber, missing_ok=True)
                if not ok:
                    return ok

            ok = _copyfile(
                os.path.join(output_dir, 'coadd', 'cus', brickname,
                             f'legacysurvey-{brickname}-model-{band}.fits.fz'),
                os.path.join(output_dir, f'{galaxy}-model-{band}.fits.fz'),
                    clobber=clobber, missing_ok=missing_ok)
            if not ok:
                return ok

        for imtype, suffix in zip(('galex', 'galexmodel', 'galexresid'), ('image', 'model', 'resid')):
            ok = _copyfile(
                os.path.join(output_dir, 'coadd', 'cus', brickname,
                             f'legacysurvey-{brickname}-{imtype}.jpg'),
                os.path.join(output_dir, f'{galaxy}-{suffix}-FUVNUV.jpg'),
                    clobber=clobber, missing_ok=missing_ok)
            if not ok:
                return ok

    if cleanup:
        _do_cleanup()

    return 1


def _get_ccds(args):
    """Unpack an argument tuple and call :func:`get_ccds`;
    multiprocessing worker.

    Parameters
    ----------
    args : :class:`tuple`
        Positional arguments matching :func:`get_ccds`'s signature.

    Returns
    -------
    See :func:`get_ccds`.

    """
    return get_ccds(*args)


def get_ccds(survey, ra, dec, width_pixels, pixscale=PIXSCALE, bands=BANDS):
    """Quickly determine which CCDs from a legacypipe survey touch a
    custom (non-brick-aligned) mosaic footprint.

    Adapted from ``legacypipe.runbrick.stage_tims``: builds a synthetic
    ``BrickDuck``/WCS centered at ``(ra, dec)`` with the requested
    ``width_pixels``/``pixscale``, queries
    ``survey.ccds_touching_wcs``, and applies the standard
    ``ccd_cuts == 0`` quality selection before optionally restricting to
    a set of filters.

    Notes
    -----
    ``bands`` defaults to the module-level ``BANDS`` dict (survey region
    name -> band list), but the filtering line tests
    ``b in bands`` for each CCD's single-letter filter -- if the default
    dict were ever actually used, membership would be tested against the
    dict's region-name keys, not any band list, so every CCD would fail
    the cut. The one real caller (``custom_coadds``) always passes an
    explicit list of band letters, so this default is latent rather than
    actively triggered.

    Parameters
    ----------
    survey : legacypipe survey object
        Survey object (e.g. ``legacypipe.survey.LegacySurveyData``)
        providing ``ccds_touching_wcs``.
    ra, dec : :class:`float`
        Footprint center, in degrees.
    width_pixels : :class:`int`
        Mosaic width, in pixels (see :func:`_mosaic_width`).
    pixscale : :class:`float`
        Pixel scale, arcsec/pixel.
    bands : :class:`list` of :class:`str`, optional
        Filters to restrict to; if None, skip band filtering entirely
        (see Notes for the default value's caveat).

    Returns
    -------
    CCDs table or :class:`list`
        legacypipe CCDs table cut to quality- and (if given)
        band-selected rows; ``[]`` if no CCDs pass the quality cut.

    """
    from legacypipe.survey import wcs_for_brick, BrickDuck
    from SGA.brick import custom_brickname

    brickname = f'custom-{custom_brickname(ra, dec)}'
    brick = BrickDuck(ra, dec, brickname)

    targetwcs = wcs_for_brick(brick, W=float(width_pixels), H=float(width_pixels), pixscale=pixscale)
    ccds = survey.ccds_touching_wcs(targetwcs)

    if ccds is None or np.sum(ccds.ccd_cuts == 0) == 0:
        return []
    ccds.cut(ccds.ccd_cuts == 0)
    if bands is not None:
        ccds.cut(np.array([b in bands for b in ccds.filter]))

    return ccds


def custom_cutouts(obj, galaxy, output_dir, width, layer, survey, ccds=None,
                   pixscale=0.262, unwise_pixscale=UNWISE_PIXSCALE,
                   galex_pixscale=GALEX_PIXSCALE, bands=GRIZ, galex=False,
                   unwise=False, ivar_cutouts=False, cleanup=True):
    """Build simple image cutouts (no Tractor model-fitting) for one
    galaxy or group, repackaged into SGA's file-naming convention.

    Calls :func:`SGA.cutouts.cutout_one` to fetch coadded image cutouts
    (optionally including unWISE and/or GALEX) from ``layer``, then
    renames/repackages the resulting per-band FITS and JPEG files into
    SGA's naming convention, e.g. for group ``SGA2025_08089m6975``::

        SGA2025_08089m6975-ccds.fits
        SGA2025_08089m6975-image.jpg
        SGA2025_08089m6975-image-g.fits.fz
        SGA2025_08089m6975-image-i.fits.fz
        SGA2025_08089m6975-image-r.fits.fz
        SGA2025_08089m6975-image-z.fits.fz
        SGA2025_08089m6975-invvar-g.fits.fz
        SGA2025_08089m6975-invvar-i.fits.fz
        SGA2025_08089m6975-invvar-r.fits.fz
        SGA2025_08089m6975-invvar-z.fits.fz

    Also converts unWISE images from Vega to AB nanomaggies (unWISE bands
    listed in ``SGA.io.VEGA2AB``), and builds a single ``MASKBITS`` image
    (``BRIGHT``, ``MEDIUM``, ``GALAXY``, ``CLUSTER``, ``RESOLVED``,
    ``MCLOUDS`` bits) from Gaia/Tycho bright-star and large-galaxy
    reference sources via legacypipe's ``get_reference_sources``/
    ``get_reference_map``, using the first optical band's WCS.

    Parameters
    ----------
    obj : :class:`~astropy.table.Table` row
        Group/galaxy row; needs ``GROUP_RA``, ``GROUP_DEC``, ``OBJNAME``.
    galaxy : :class:`str`
        Output filename prefix (group or galaxy name).
    output_dir : :class:`str`
        Directory to write cutouts to.
    width : :class:`int`
        Mosaic width, in pixels (e.g. from :func:`_mosaic_width`).
    layer : :class:`str`
        Legacy Survey imaging layer name (e.g. ``'ls-dr11'``), passed to
        :func:`SGA.cutouts.cutout_one`.
    survey : legacypipe survey object
        Used to look up ``MASKBITS`` values and reference sources.
    ccds : CCDs table, optional
        If given, written verbatim to ``{galaxy}-ccds.fits`` and used to
        get the mean MJD for the maskbits WCS's ``TAITime``.
    pixscale : :class:`float`
        Optical pixel scale, arcsec/pixel.
    unwise_pixscale : :class:`float`
        unWISE pixel scale, arcsec/pixel.
    galex_pixscale : :class:`float`
        GALEX pixel scale, arcsec/pixel.
    bands : :class:`list` of :class:`str`
        Optical bands to fetch.
    galex : :class:`bool`
        If True, also fetch and rearrange GALEX FUV/NUV cutouts.
    unwise : :class:`bool`
        If True, also fetch and rearrange unWISE W1-W4 cutouts.
    ivar_cutouts : :class:`bool`
        If True, also fetch and write inverse-variance cutouts.
    cleanup : :class:`bool`
        If True, delete :func:`SGA.cutouts.cutout_one`'s raw output
        files after rearranging them into the SGA naming convention.

    Returns
    -------
    :class:`int`
        ``1`` on success; ``0`` if the primary cutout FITS file could
        not be read.

    """
    import fitsio
    import shutil
    from time import time
    from astrometry.util.util import Tan
    from tractor.tractortime import TAITime
    from legacypipe.bits import REF_MAP_BITS, maskbits_type
    from legacypipe.reference import get_reference_sources, get_reference_map
    from legacypipe.survey import LegacySurveyWcs
    from SGA.util import get_dt
    from SGA.io import make_header, VEGA2AB
    from SGA.cutouts import cutout_one

    tall = time()

    # CCDs file
    if ccds:
        ccdsfile = os.path.join(output_dir, f'{galaxy}-ccds.fits')
        ccds.writeto(ccdsfile, extname='CCDS', clobber=True)
        log.info(f'Wrote {ccdsfile}')

    dry_run = False
    fits_cutouts = True
    ivar_cutouts = ivar_cutouts
    unwise_cutouts = unwise # False
    galex_cutouts = galex # False

    basefile = os.path.join(output_dir, galaxy)
    cutout_one(basefile, obj['GROUP_RA'], obj['GROUP_DEC'],
               width, pixscale, unwise_pixscale, galex_pixscale,
               layer, bands, dry_run, fits_cutouts, ivar_cutouts,
               unwise_cutouts, galex_cutouts, 0, 0)

    # now rearrange the files to match our file / data model
    fitssuffixes = ['', ]
    jpgsuffixes = ['', ]
    outjpgsuffixes = ['', ]
    allbands = [bands.copy(), ]
    allpixscale = [[pixscale]*len(bands), ]
    if unwise:
        fitssuffixes += ['-unwise']
        jpgsuffixes += ['-W1W2', ]
        outjpgsuffixes += ['-W1W2', ]
        allbands += [['W1', 'W2', 'W3', 'W4', ], ]
        allpixscale += [[UNWISE_PIXSCALE]*4, ]
    if galex:
        fitssuffixes += ['-galex']
        jpgsuffixes += ['-galex', ]
        outjpgsuffixes += ['-FUVNUV', ]
        allbands += [['FUV', 'NUV', ], ]
        allpixscale += [[GALEX_PIXSCALE]*2, ]


    need_maskbits = True
    for fitssuffix, jpgsuffix, outjpgsuffix, allband, allpixscale in zip(
            fitssuffixes, jpgsuffixes, outjpgsuffixes, allbands, allpixscale):
        infile = os.path.join(output_dir, f'{galaxy}{jpgsuffix}.jpeg')
        outfile = os.path.join(output_dir, f'{galaxy}-image{outjpgsuffix}.jpg')
        shutil.copy2(infile, outfile)
        log.info(f'Copying {infile} --> {outfile}')

        fitsfile = os.path.join(output_dir, f'{galaxy}{fitssuffix}.fits')
        try:
            imgs, hdr = fitsio.read(fitsfile, header=True)
            for key in ['VERSION', 'BANDS', 'COMMENT']:
                hdr.delete(key)
            for iband in range(len(bands)):
                hdr.delete(f'BAND{iband}')

            if ivar_cutouts:
                ivars, ivarhdr = fitsio.read(fitsfile, ext=1, header=True)
                for key in ['VERSION', 'BANDS', 'COMMENT']:
                    ivarhdr.delete(key)
                for iband in range(len(bands)):
                    ivarhdr.delete(f'BAND{iband}')
        except:
            msg = f'There was a problem reading {fitsfile} ({obj["OBJNAME"]})'
            log.critical(msg)
            return 0

        for iband, band in enumerate(allband):
            outfile = os.path.join(output_dir, f'{galaxy}-image-{band}.fits')

            # convert WISE images from Vega nanomaggies to AB nanomaggies
            # https://www.legacysurvey.org/dr9/description/#photometry
            if band in VEGA2AB.keys():
                imgs[iband, :, :] *= 10.**(-0.4 * VEGA2AB[band])
                if ivar_cutouts:
                    ivars[iband, :, :] /= (10.**(-0.4 * VEGA2AB[band]))**2.

            primhdr = fitsio.FITSHDR()
            primhdr['EXTEND'] = 'T'

            extra = {
                'PIXSCALE': (allpixscale[iband], 'pixel scale (arcsec/pixel)'),
                'FILTERX': (band, 'Filter short name'),
                'PHOTSYS': ('AB', 'photometric system'),
                'MAGZERO': (22.5, 'Magnitude zeropoint'),
                'BUNIT': ('nanomaggy', 'AB mag = 22.5 - 2.5*log10(nanomaggy)'),
            }
            outhdr = make_header(hdr, keys=hdr.keys(), extra=extra)
            fitsio.write(outfile, None, header=primhdr, clobber=True)
            fitsio.write(outfile, imgs[iband, :, :], header=outhdr, extname=f'IMAGE_{band.upper()}')
            log.info(f'Wrote {outfile}')

            if ivar_cutouts:
                outfile = os.path.join(output_dir, f'{galaxy}-invvar-{band}.fits')
                outhdr = make_header(ivarhdr, keys=ivarhdr.keys(), extra=extra)
                fitsio.write(outfile, None, header=primhdr, clobber=True)
                fitsio.write(outfile, ivars[iband, :, :], header=outhdr, extname=f'INVVAR_{band.upper()}')
                log.info(f'Wrote {outfile}')

            # maskbits image
            if need_maskbits and band in bands:
                wcs = LegacySurveyWcs(Tan(hdr), TAITime(None, mjd=np.mean(ccds.mjd_obs)))

                refstars, _ = get_reference_sources(
                    survey, wcs.wcs, bands=bands, tycho_stars=True,
                    gaia_stars=True, large_galaxies=False, star_clusters=False)
                refmap = get_reference_map(wcs.wcs, refstars)

                # from runbrick.stage_image_coadds
                MASKBITS = survey.get_maskbits()
                maskbits = np.zeros(refmap.shape, dtype=maskbits_type)
                for key in ['BRIGHT', 'MEDIUM', 'GALAXY', 'CLUSTER', 'RESOLVED', 'MCLOUDS']:
                    maskbits |= MASKBITS[key] * ((refmap & REF_MAP_BITS[key]) > 0)

                maskbitsfile = os.path.join(output_dir, f'{galaxy}-maskbits.fits')

                outhdr = make_header(hdr, keys=hdr.keys(),
                                     extra={'PIXSCALE': (allpixscale[iband], 'pixel scale (arcsec/pixel)')})
                for key in ['SURVEY']:
                    outhdr.delete(key)
                fitsio.write(maskbitsfile, None, header=primhdr, clobber=True)
                fitsio.write(maskbitsfile, maskbits, header=outhdr, extname='MASKBITS')
                log.info(f'Wrote {maskbitsfile}')

                need_maskbits = False

    # cleanup...
    if cleanup:
        cleanfiles = [f'{basefile}.fits', f'{basefile}.jpeg']
        if unwise:
            cleanfiles += [f'{basefile}-unwise.fits', f'{basefile}-W1W2.jpeg']
        if galex:
            cleanfiles += [f'{basefile}-galex.fits', f'{basefile}-galex.jpeg']
        for cleanfile in cleanfiles:
            os.remove(cleanfile)

    dt, unit = get_dt(tall)
    log.info(f'Total time for custom cutouts: {dt:.3f} {unit}')

    return 1


def custom_coadds(onegal, galaxy, survey, run, radius_mosaic_arcsec,
                  release=11000, pixscale=PIXSCALE, unwise_pixscale=UNWISE_PIXSCALE,
                  galex_pixscale=GALEX_PIXSCALE, bands=GRIZ, mp=1, layer='ls-dr11',
                  nsigma=None, saddle_fraction=None, saddle_min=None, nsatur=2,
                  rgb_stretch=1.5, no_iterative=False, no_segmentation=False,
                  racolumn='GROUP_RA', deccolumn='GROUP_DEC', force_psf_detection=False,
                  fit_on_coadds=False, bright_masking=False, galaxy_masking=False,
                  just_cutouts=False, ivar_cutouts=False, use_gpu=False, ngpu=1,
                  threads_per_gpu=8, subsky_radii=None, just_coadds=False,
                  missing_ok=False, force=False, cleanup=True, unwise=True, galex=False,
                  no_gaia=False, no_tycho=False, verbose=False):
    """Top-level driver: build a custom set of large-galaxy coadds for
    one group/galaxy by invoking legacypipe's ``runbrick`` pipeline.

    Builds a custom "brick" centered on ``onegal``'s position with a
    width set by ``radius_mosaic_arcsec`` (via :func:`_mosaic_width`),
    quickly checks that any CCDs touch it (:func:`get_ccds`), and then
    either (``just_cutouts=True``) fetches simple image cutouts via
    :func:`custom_cutouts` with no model-fitting, or (the normal case)
    assembles a ``legacypipe.runbrick`` command line from the remaining
    keyword arguments and runs it in-process. On success, renames
    legacypipe's brick-named output files into SGA's per-galaxy naming
    convention via :func:`_rearrange_files`.

    Notes
    -----
    ``verbose`` is accepted but never referenced in this function's
    body -- dead parameter. ``use_gpu`` only appends
    ``--threads-per-gpu``/``--ngpu`` to the runbrick command line; the
    ``--use-gpu`` flag itself (and ``--gpumode=2``/``--verbose``) is
    commented out in the ``cmdargs`` construction, so setting
    ``use_gpu=True`` does not actually enable GPU mode in the invoked
    pipeline as the parameter name would suggest.

    Parameters
    ----------
    onegal : :class:`~astropy.table.Table` row
        Group/galaxy row; needs ``racolumn``, ``deccolumn``.
    galaxy : :class:`str`
        Output filename prefix (group or galaxy name).
    survey : legacypipe survey object
        Provides ``output_dir``, ``survey_dir``, and CCD lookups.
    run : :class:`str`
        legacypipe survey run name (e.g. ``'south'``), passed to
        ``runbrick``.
    radius_mosaic_arcsec : :class:`float`
        Mosaic radius, in arcsec (e.g. from ``SGA.SGA.get_radius_mosaic``),
        converted to a pixel width via :func:`_mosaic_width`.
    release : :class:`int`
        Legacy Survey data release number, passed to ``runbrick``.
    pixscale : :class:`float`
        Optical pixel scale, arcsec/pixel.
    unwise_pixscale : :class:`float`
        unWISE pixel scale, arcsec/pixel (only used by the
        ``just_cutouts`` path).
    galex_pixscale : :class:`float`
        GALEX pixel scale, arcsec/pixel (only used by the
        ``just_cutouts`` path).
    bands : :class:`list` of :class:`str`
        Optical bands to process.
    mp : :class:`int`
        Number of threads passed to ``runbrick`` (``--threads``).
    layer : :class:`str`
        Legacy Survey imaging layer name, passed through to
        :func:`custom_cutouts` on the ``just_cutouts`` path.
    nsigma : :class:`float`, optional
        Detection significance threshold, passed to ``runbrick``
        (``--nsigma``) if given.
    saddle_fraction : :class:`float`, optional
        Source-detection saddle-point fraction, passed to ``runbrick``
        (``--saddle-fraction``) if given.
    saddle_min : :class:`float`, optional
        Source-detection saddle-point minimum, passed to ``runbrick``
        (``--saddle-min``) if given.
    nsatur : :class:`int`, optional
        Saturation-pixel count threshold, passed to ``runbrick``
        (``--nsatur``) if given.
    rgb_stretch : :class:`float`, optional
        RGB JPEG stretch factor, passed to ``runbrick``
        (``--rgb-stretch``) if given.
    no_iterative : :class:`bool`
        If True, disable iterative source detection (``--no-iterative``).
    no_segmentation : :class:`bool`
        If True, disable segmentation-based detection
        (``--no-segmentation``).
    racolumn, deccolumn : :class:`str`
        Column names in ``onegal`` giving the mosaic center.
    force_psf_detection : :class:`bool`
        If False (default), pass ``--no-galaxy-forcepsf`` to disable
        forced PSF-only detection within known large galaxies.
    fit_on_coadds : :class:`bool`
        If True, fit sources on the coadds rather than per-CCD images
        (``--fit-on-coadds --no-ivar-reweighting``).
    bright_masking : :class:`bool`
        If True, pass ``--bright-masking`` to ``runbrick``.
    galaxy_masking : :class:`bool`
        If True, pass ``--galaxy-masking`` to ``runbrick``.
    just_cutouts : :class:`bool`
        If True, skip the ``runbrick`` pipeline entirely and instead
        call :func:`custom_cutouts` for simple, unmodeled image cutouts.
    ivar_cutouts : :class:`bool`
        Passed through to :func:`custom_cutouts` on the ``just_cutouts``
        path.
    use_gpu : :class:`bool`
        See Notes -- does not actually enable ``runbrick``'s GPU mode.
    ngpu : :class:`int`
        Number of GPUs, passed to ``runbrick`` (``--ngpu``) when
        ``use_gpu`` is True.
    threads_per_gpu : :class:`int`
        Threads per GPU, passed to ``runbrick`` (``--threads-per-gpu``)
        when ``use_gpu`` is True.
    subsky_radii : sequence of :class:`float`, optional
        If given, disables the default sky subtraction in favor of
        ubercal sky subtraction with these radii
        (``--no-subsky --ubercal-sky --subsky-radii ...``).
    just_coadds : :class:`bool`
        If True, stop ``runbrick`` after the ``image_coadds`` stage
        (``--stage=image_coadds``), skipping source detection/fitting.
    missing_ok : :class:`bool`
        Passed to :func:`_rearrange_files`; also forced True whenever
        ``cleanup=False``.
    force : :class:`bool`
        If True, force ``runbrick`` to redo every stage
        (``--force-all``) and delete any existing checkpoint file first.
    cleanup : :class:`bool`
        Passed to :func:`_rearrange_files` (and, on the ``just_cutouts``
        path, to :func:`custom_cutouts`) to control whether intermediate
        files are deleted after rearranging.
    unwise : :class:`bool`
        If True, include unWISE W1-W4 coadds (``--save-unwise-psf``); if
        False, disable unWISE entirely (``--no-unwise-coadds --no-wise``).
    galex : :class:`bool`
        If True, include GALEX FUV/NUV coadds (``--galex
        --save-galex-psf``).
    no_gaia : :class:`bool`
        If True, pass ``--no-gaia`` to ``runbrick``.
    no_tycho : :class:`bool`
        If True, pass ``--no-tycho`` to ``runbrick``.
    verbose : :class:`bool`
        Unused (see Notes).

    Returns
    -------
    status : :class:`int`
        ``1`` on success, ``0`` on failure (no CCDs touching the brick
        return ``1`` -- "nothing to do" rather than failure; a
        ``runbrick`` exception or nonzero return code, or a failed
        :func:`_rearrange_files` call, return ``0``).
    stagesuffix : :class:`str`
        Always ``'coadds'``; identifies this processing stage for
        marker-file bookkeeping (see ``SGA.SGA.missing_files``).

    """
    import fitsio
    from legacypipe.runbrick import main as runbrick
    from SGA.brick import custom_brickname

    stagesuffix = 'coadds'

    width = _mosaic_width(radius_mosaic_arcsec, pixscale=pixscale)
    brickname = f'custom-{custom_brickname(onegal[racolumn], onegal[deccolumn])}'

    # Quickly read the input CCDs and check that we have all the
    # colors we need.
    ccds = get_ccds(survey, onegal[racolumn], onegal[deccolumn],
                    width, pixscale, bands=bands)
    if len(ccds) == 0:
        log.info('No CCDs touching this brick; nothing to do.')
        return 1, stagesuffix

    #usebands = np.array(sorted(set(ccds.filter)))
    #log.info(f'Bands touching this brick: {",".join(usebands)}')
    #bands = usebands

    # just cutouts -- no pipeline
    if just_cutouts:
        err = custom_cutouts(onegal, galaxy, survey.output_dir, width, layer,
                             survey, ccds=ccds, pixscale=pixscale, bands=bands,
                             galex=galex, unwise=unwise, unwise_pixscale=unwise_pixscale,
                             galex_pixscale=galex_pixscale, ivar_cutouts=ivar_cutouts,
                             cleanup=cleanup)
        return err, stagesuffix


    # Run the pipeline!
    cmdargs = f'--radec {onegal[racolumn]} {onegal[deccolumn]} '
    cmdargs += f'--width={width} --height={width} --pixscale={pixscale} '
    cmdargs += f'--threads={mp} --outdir={survey.output_dir} --bands={",".join(bands)} '
    cmdargs += f'--survey-dir={survey.survey_dir} --run={run} '
    cmdargs += f'--release={release} '

    if nsatur:
        cmdargs += f'--nsatur={nsatur:.0f} '

    if rgb_stretch:
        cmdargs += f'--rgb-stretch={rgb_stretch:.2f} '
    if nsigma:
        cmdargs += f'--nsigma={nsigma:.0f} '
    if saddle_fraction:
        cmdargs += f'--saddle-fraction={saddle_fraction:.2f} '
    if saddle_min:
        cmdargs += f'--saddle-min={saddle_min:.1f} '

    #cmdargs += '--write-stage=tims --write-stage=srcs '
    cmdargs += '--write-stage=srcs '

    #log.warning('Undo --old-calibs-ok in custom_coadds when ready!')
    #cmdargs += '--old-calibs-ok '
    cmdargs += '--skip-calibs '
    cmdargs += f'--checkpoint={survey.output_dir}/{galaxy}-checkpoint.p '
    cmdargs += f'--pickle={survey.output_dir}/{galaxy}-%%(stage)s.p '
    if just_coadds:
        #unwise = False
        #galex = False
        cmdargs += '--stage=image_coadds '
    if not unwise:
        cmdargs += '--no-unwise-coadds --no-wise '
    else:
        cmdargs += '--save-unwise-psf '
    if galex:
        cmdargs += '--galex '
        cmdargs += '--save-galex-psf '
    if no_gaia:
        cmdargs += '--no-gaia '
    if no_tycho:
        cmdargs += '--no-tycho '
    if force:
        cmdargs += '--force-all '
        checkpointfile = f'{survey.output_dir}/{galaxy}-checkpoint.p'
        if os.path.isfile(checkpointfile):
            os.remove(checkpointfile)
    if subsky_radii is not None: # implies --no-subsky
        #if len(subsky_radii) != 3:
        #    raise ValueError('subsky_radii must be a 3-element vector')
        #cmdargs += '--no-subsky --ubercal-sky --subsky-radii {} {} {} '.format(subsky_radii[0], subsky_radii[1], subsky_radii[2]) # [arcsec]
        cmdargs += '--no-subsky --ubercal-sky --subsky-radii'
        for rad in subsky_radii:
            cmdargs += f' {rad} '
    #if ubercal_sky: # implies --no-subsky
    #    cmdargs += '--no-subsky --ubercal-sky '

    # stage-specific options here--
    cmdargs += '--save-coadd-psf '

    if not force_psf_detection:
        cmdargs += '--no-galaxy-forcepsf '
    if fit_on_coadds:
        cmdargs += '--fit-on-coadds --no-ivar-reweighting '
    if no_iterative:
        cmdargs += '--no-iterative '
    if no_segmentation:
        cmdargs += '--no-segmentation '
    if bright_masking:
        cmdargs += '--bright-masking '
    if galaxy_masking:
        cmdargs += '--galaxy-masking '

    # GPU stuff
    if use_gpu:
        #cmdargs += f'--use-gpu --threads-per-gpu={threads_per_gpu} --ngpu={ngpu} --gpumode=2 '#--verbose '
        cmdargs += f'--threads-per-gpu={threads_per_gpu} --ngpu={ngpu} '#--verbose '

    try:
        log.info(f'runbrick {cmdargs}')
        err = runbrick(args=cmdargs.split())
        #err = 0
    except:
        log.critical(f'Exception raised on {survey.output_dir}/{galaxy}')
        import traceback
        traceback.print_exc()
        return 0, stagesuffix

    # get the updated (final) set of bands
    ccdsfile = os.path.join(
        survey.output_dir, 'coadd', 'cus', brickname,
        f'legacysurvey-{brickname}-ccds.fits')
    if os.path.isfile(ccdsfile):
        bands = np.array(sorted(set(fitsio.read(ccdsfile, columns='filter'))))

    if err != 0:
        log.warning('Something went wrong; please check the logfile.')
        return 0, stagesuffix
    else:
        # Move (rename) files into the desired output directory and clean up.
        if cleanup is False:
            missing_ok = True
        ok = _rearrange_files(galaxy, survey.output_dir, brickname,
                              unwise=unwise, galex=galex, cleanup=cleanup,
                              just_coadds=just_coadds, clobber=True,
                              bands=bands, missing_ok=missing_ok)

        return ok, stagesuffix
