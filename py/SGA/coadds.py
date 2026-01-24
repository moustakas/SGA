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

RUNS = {'dr9-north': 'north', 'dr9-south': 'south',
        'dr10-south': 'south',
        'dr11-north': 'north', 'dr11-south': 'south'}

# although dr9-north is missing i-band imaging, there are many
# advantages to adopting a consistent data model
GRZ = ['g', 'r', 'z']
GRIZ = ['g', 'r', 'i', 'z']
BANDS = {'dr9-north': GRIZ, 'dr9-south': GRIZ,
         'dr10-south': GRIZ,
         'dr11-north': GRIZ, 'dr11-south': GRIZ}

RELEASE = {'dr9-north': 9011, 'dr9-south': 9010,
           'dr10-south': 10000,
           'dr11-north': 11001, 'dr11-south': 11000}

REGIONBITS = {
    'dr11-south': 2**0,
    'dr11-north': 2**1,
    'dr9-north': 2**1
}


def srcs2image(cat, wcs, band='r', pixelized_psf=None, psf_sigma=1.):
    """Build a model image from a Tractor catalog or a list of sources.

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
    invvar = np.ones(shape)

    if pixelized_psf is None:
        vv = psf_sigma**2.
        psf = GaussianMixturePSFGaussianMixturePSF(1., 0., 0., vv, vv, 0.)
    else:
        psf = pixelized_psf

    photocal = LinearPhotoCal(1., band=band.lower())
    tim = Image(model, invvar=invvar, wcs=wcs, psf=psf,
                photocal=photocal, sky=ConstantSky(0.),
                name=f'model-{band}')

    # Do we have a tractor catalog or a list of sources?
    if type(cat) is tabledata:
        srcs = read_fits_catalog(cat, bands=[band.lower()])
    else:
        srcs = cat

    mod = Tractor([tim], srcs).getModelImage(0)

    return mod


def _mosaic_width(radius_mosaic_arcsec, pixscale=PIXSCALE):
    """Ensure the mosaic is an odd number of pixels so the central can
    land on a whole pixel (important for ellipse-fitting).

    """
    width = 2 * radius_mosaic_arcsec / pixscale # [pixels]
    width = (np.ceil(width) // 2 * 2 + 1).astype('int') # [pixels]
    return width


def _rearrange_files(galaxy, output_dir, brickname, bands=GRIZ, unwise=True,
                     galex=False, cleanup=False, just_coadds=False,
                     clobber=False, missing_ok=False):
    """Move (rename) files into the desired output directory and clean up.

    """
    import fitsio
    import shutil

    def _copyfile(infile, outfile, clobber=False, update_header=False, missing_ok=False):
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
    """Wrapper for the multiprocessing."""
    return get_ccds(*args)


def get_ccds(survey, ra, dec, width_pixels, pixscale=PIXSCALE, bands=BANDS):
    """Quickly get the CCDs touching this custom brick.  This code is
    mostly taken from legacypipe.runbrick.stage_tims.

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


def custom_cutouts(obj, galaxy, output_dir, width, layer, pixscale=0.262,
                   unwise_pixscale=UNWISE_PIXSCALE, galex_pixscale=GALEX_PIXSCALE,
                   bands=GRIZ, galex=False, unwise=False, ivar_cutouts=False):
    """
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

    """
    import fitsio
    import shutil
    from SGA.io import make_header, VEGA2AB
    from SGA.cutouts import cutout_one

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

    for fitssuffix, jpgsuffix, outjpgsuffix, allband, allpixscale in zip(
            fitssuffixes, jpgsuffixes, outjpgsuffixes, allbands, allpixscale):
        infile = os.path.join(output_dir, f'{galaxy}{jpgsuffix}.jpeg')
        outfile = os.path.join(output_dir, f'{galaxy}-image{outjpgsuffix}.jpg')
        shutil.copy2(infile, outfile)
        log.info(f'Copying {infile} --> {outfile}')

        fitsfile = os.path.join(output_dir, f'{galaxy}{fitssuffix}.fits')
        try:
            imgs, hdr = fitsio.read(fitsfile, header=True)
        except:
            msg = f'There was a problem reading {fitsfile} ({obj["OBJNAME"]})'
            log.critical(msg)
            return 0

        hdr.delete('VERSION')
        hdr.delete('BANDS')
        hdr.delete('COMMENT')
        for iband in range(len(bands)):
            hdr.delete(f'BAND{iband}')

        for iband, band in enumerate(allband):
            outfile = os.path.join(output_dir, f'{galaxy}-image-{band}.fits')

            # convert WISE images from Vega nanomaggies to AB nanomaggies
            # https://www.legacysurvey.org/dr9/description/#photometry
            if band in VEGA2AB.keys():
                imgs[iband, :, :] *= 10.**(-0.4 * VEGA2AB[band])

            primhdr = fitsio.FITSHDR()
            primhdr['EXTEND'] = 'T'

            extra = {
                'PIXSCALE': (allpixscale[iband], 'pixel scale (arcsec/pixel)'),
                'FILTERX': (band, 'Filter short name'),
                'PHOTSYS': ('AB', 'photometric system'),
                'MAGZERO': (22.5, 'Magnitude zeropoint'),
                'BUNIT': ('nanomaggy', 'AB mag = 22.5 - 2.5*log10(nanomaggy)'),
            }
            outhdr = make_header(hdr, keys=hdr.keys(), extra=extra, extname=f'IMAGE_{band}')
            fitsio.write(outfile, None, header=primhdr, clobber=True)
            fitsio.write(outfile, imgs[iband, :, :], header=outhdr)
            log.info(f'Wrote {outfile}')

    # cleanup...
    cleanfiles = [f'{basefile}.fits', f'{basefile}.jpeg']
    if unwise:
        cleanfiles += [f'{basefile}-unwise.fits', f'{basefile}-W1W2.jpeg']
    if galex:
        cleanfiles += [f'{basefile}-galex.fits', f'{basefile}-galex.jpeg']
    for cleanfile in cleanfiles:
        os.remove(cleanfile)

    return 1


def custom_coadds(onegal, galaxy, survey, run, radius_mosaic_arcsec,
                  release=1000, pixscale=PIXSCALE, unwise_pixscale=UNWISE_PIXSCALE,
                  galex_pixscale=GALEX_PIXSCALE, bands=GRIZ, mp=1, layer='ls-dr11',
                  nsigma=None, nsatur=2, rgb_stretch=1.5, no_iterative=False,
                  racolumn='GROUP_RA', deccolumn='GROUP_DEC', force_psf_detection=False,
                  fit_on_coadds=False, just_cutouts=False, ivar_cutouts=False, use_gpu=False,
                  ngpu=1, threads_per_gpu=8, subsky_radii=None, just_coadds=False,
                  missing_ok=False, force=False, cleanup=True, unwise=True, galex=False,
                  no_gaia=False, no_tycho=False, verbose=False):
    """Build a custom set of large-galaxy coadds.

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
                             pixscale=pixscale, bands=bands, galex=galex,
                             unwise=unwise, unwise_pixscale=unwise_pixscale,
                             galex_pixscale=galex_pixscale, ivar_cutouts=ivar_cutouts)
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

    # GPU stuff
    if use_gpu:
        cmdargs += f'--use-gpu --threads-per-gpu={threads_per_gpu} --ngpu={ngpu} --gpumode=2 '#--verbose '

    try:
        log.info(f'runbrick {cmdargs}')
        err = runbrick(args=cmdargs.split())
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
