"""
==========
SGA.coadds
==========

"""
import os, pdb
import numpy as np
from SGA.io import RACOLUMN, DECCOLUMN
from SGA.logger import log


PIXSCALE = 0.262
GALEX_PIXSCALE = 1.5
UNWISE_PIXSCALE = 2.75

RUNS = {'dr9-north': 'north', 'dr9-south': 'south', 
        'dr10-south': 'south', 'dr11-south': 'south'}

GRZ = ['g', 'r', 'z']
GRIZ = ['g', 'r', 'i', 'z']
BANDS = {'dr9-north': GRZ, 'dr9-south': GRZ,
         'dr10-south': GRIZ, 'dr11-south': GRIZ}


def _mosaic_width(radius_mosaic_arcsec, pixscale=PIXSCALE):
    """Ensure the mosaic is an odd number of pixels so the central can
    land on a whole pixel (important for ellipse-fitting).

    """
    width = 2 * radius_mosaic_arcsec / pixscale # [pixels]
    width = (np.ceil(width) // 2 * 2 + 1).astype('int') # [pixels]
    return width


def _rearrange_files(galaxy, output_dir, brickname, stagesuffix,
                     bands=GRIZ, unwise=True, galex=False, cleanup=False, 
                     just_coadds=False, clobber=False, missing_ok=False):
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
        from glob import glob
        shutil.rmtree(os.path.join(output_dir, 'coadd'), ignore_errors=True)
        shutil.rmtree(os.path.join(output_dir, 'metrics'), ignore_errors=True)
        shutil.rmtree(os.path.join(output_dir, 'tractor'), ignore_errors=True)
        shutil.rmtree(os.path.join(output_dir, 'tractor-i'), ignore_errors=True)
        picklefiles = glob(os.path.join(output_dir, f'{galaxy}-{stagesuffix}-*.p'))
        for picklefile in picklefiles:
            if os.path.isfile(picklefile):
                os.remove(picklefile)

    ## Diagnostic plots (OK if they're missing) (put this above the check for the
    ## CCDs--if the plots have been made then they are useful for testing and
    ## debugging).
    #for qatype in ['ccdpos']:
    ##for qatype in ['ccdpos', 'pipelinesky', 'customsky']:
    #    ok = _copyfile(
    #        os.path.join(output_dir, 'metrics', 'cus', 'fitoncoadds-{}-{}.jpg'.format(brickname, qatype)),
    #                     os.path.join(output_dir, '{}-{}-{}.jpg'.format(galaxy, stagesuffix, qatype)),
    #        clobber=clobber, missing_ok=True)
    #    if not ok:
    #        return ok

    # If we made it here and there is no CCDs file it's because legacypipe
    # exited cleanly with "No photometric CCDs touching brick."
    _ccdsfile = os.path.join(output_dir, 'coadd', 'cus', brickname,
                            f'legacysurvey-{brickname}-ccds.fits')
    if not os.path.isfile(_ccdsfile) and missing_ok is False:
        print('No photometric CCDs touching brick.')
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

    # For objects on the edge of the footprint we can sometimes lose 3-band
    # coverage if one of the bands is fully masked. Check here and write out all
    # the files except a 
    if os.path.isfile(ccdsfile): # can be missing during testing if missing_ok=True
        allbands = fitsio.read(ccdsfile, columns='filter')
        ubands = list(sorted(set(allbands)))

    # image coadds (FITS + JPG)
    for band in bands:
        for imtype, outtype in zip(('image', 'invvar'), ('image', 'invvar')):
            ok = _copyfile(
                os.path.join(output_dir, 'coadd', 'cus', brickname,
                             f'legacysurvey-{brickname}-{imtype}-{band}.fits.fz'),
                             os.path.join(output_dir, f'{galaxy}-{stagesuffix}-{outtype}-{band}.fits.fz'),
                 clobber=clobber, missing_ok=missing_ok, update_header=True)
            if not ok:
                return ok

    # JPG images
    ok = _copyfile(
        os.path.join(output_dir, 'coadd', 'cus', brickname,
                     f'legacysurvey-{brickname}-image.jpg'),
        os.path.join(output_dir, f'{galaxy}-{stagesuffix}-image.jpg'),
        clobber=clobber, missing_ok=missing_ok)
    if not ok:
        return ok

    if just_coadds:
        if cleanup:
            _do_cleanup()
        return 1

    # PSFs (none for stagesuffix=='pipeline')
    if stagesuffix != 'pipeline':
        for band in bands:
            for imtype, outtype in zip(['copsf'], ['psf']):
                ok = _copyfile(
                    os.path.join(output_dir, 'coadd', 'cus', brickname,
                                 f'legacysurvey-{brickname}-{imtype}-{band}.fits.fz'),
                                 os.path.join(output_dir, f'{galaxy}-{stagesuffix}-{outtype}-{band}.fits.fz'),
                    clobber=clobber, missing_ok=missing_ok)
                if not ok:
                    return ok

        if unwise:
            for band in ['W1', 'W2', 'W3', 'W4']:
                for imtype, outtype in zip(['copsf'], ['psf']):
                    ok = _copyfile(
                        os.path.join(output_dir, 'coadd', 'cus', brickname,
                                     f'legacysurvey-{brickname}-{imtype}-{band}.fits.fz'),
                                     os.path.join(output_dir, f'{galaxy}-{stagesuffix}-{outtype}-{band}.fits.fz'),
                        clobber=clobber, missing_ok=missing_ok)
                    if not ok:
                        return ok

        if galex:
            for band in ['FUV', 'NUV']:
                for imtype, outtype in zip(['copsf'], ['psf']):
                    ok = _copyfile(
                        os.path.join(output_dir, 'coadd', 'cus', brickname,
                                     f'legacysurvey-{brickname}-{imtype}-{band}.fits.fz'),
                                     os.path.join(output_dir, f'{galaxy}-{stagesuffix}-{outtype}-{band}.fits.fz'),
                        clobber=clobber, missing_ok=missing_ok)
                    if not ok:
                        return ok

    # tractor catalog
    ok = _copyfile(
        os.path.join(output_dir, 'tractor', 'cus', f'tractor-{brickname}.fits'),
        os.path.join(output_dir, f'{galaxy}-{stagesuffix}-tractor.fits'),
        clobber=clobber, missing_ok=missing_ok)
    if not ok:
        return ok

    # Maskbits, blob images, outlier masks, and depth images.
    ok = _copyfile(
        os.path.join(output_dir, 'coadd', 'cus', brickname,
                     f'legacysurvey-{brickname}-maskbits.fits.fz'),
        os.path.join(output_dir, f'{galaxy}-{stagesuffix}-maskbits.fits.fz'),
        clobber=clobber, missing_ok=missing_ok)
    if not ok:
        return ok

    if False:
        ok = _copyfile(
            os.path.join(output_dir, 'metrics', 'cus', f'blobs-{brickname}.fits.gz'),
            os.path.join(output_dir, f'{galaxy}-{stagesuffix}-blobs.fits.gz'),
            clobber=clobber, missing_ok=missing_ok)
        if not ok:
            return ok

    ok = _copyfile(
        os.path.join(output_dir, 'metrics', 'cus', f'outlier-mask-{brickname}.fits.fz'),
        os.path.join(output_dir, f'{galaxy}-{stagesuffix}-outlier-mask.fits.fz'),
        clobber=clobber, missing_ok=missing_ok)
    if not ok:
        return ok

    if False:
        for band in ['g', 'r', 'z']:
            ok = _copyfile(
                os.path.join(output_dir, 'coadd', 'cus', brickname,
                             f'legacysurvey-{brickname}-depth-{band}.fits.fz'),
                os.path.join(output_dir, f'{galaxy}-depth-{band}.fits.fz'),
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
                os.path.join(output_dir, f'{galaxy}-{stagesuffix}-{imtype}-{band}.fits.fz'),
                clobber=clobber, missing_ok=missing_ok)
            if not ok:
                return ok

    # JPG images
    for imtype in ('model', 'resid'):
        ok = _copyfile(
            os.path.join(output_dir, 'coadd', 'cus', brickname,
                         f'legacysurvey-{brickname}-{imtype}.jpg'),
            os.path.join(output_dir, f'{galaxy}-{stagesuffix}-{imtype}.jpg'),
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
                    os.path.join(output_dir, f'{galaxy}-{stagesuffix}-{imtype}-{band}.fits.fz'),
                    clobber=clobber, missing_ok=missing_ok)
                if not ok:
                    return ok

            ok = _copyfile(
                os.path.join(output_dir, 'coadd', 'cus', brickname,
                             f'legacysurvey-{brickname}-model-{band}.fits.fz'),
                os.path.join(output_dir, f'{galaxy}-{stagesuffix}-model-{band}.fits.fz'),
                    clobber=clobber, missing_ok=missing_ok)
            if not ok:
                return ok

        for imtype, suffix in zip(('wise', 'wisemodel', 'wiseresid'), ('image', 'model', 'resid')):
            ok = _copyfile(
                os.path.join(output_dir, 'coadd', 'cus', brickname,
                             f'legacysurvey-{brickname}-{imtype}.jpg'),
                os.path.join(output_dir, f'{galaxy}-{stagesuffix}-{suffix}-W1W2.jpg'),
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
                    os.path.join(output_dir, f'{galaxy}-{stagesuffix}-{imtype}-{band}.fits.fz'),
                    clobber=clobber, missing_ok=True)
                if not ok:
                    return ok

            ok = _copyfile(
                os.path.join(output_dir, 'coadd', 'cus', brickname,
                             f'legacysurvey-{brickname}-model-{band}.fits.fz'),
                os.path.join(output_dir, f'{galaxy}-{stagesuffix}-model-{band}.fits.fz'),
                    clobber=clobber, missing_ok=missing_ok)
            if not ok:
                return ok

        for imtype, suffix in zip(('galex', 'galexmodel', 'galexresid'), ('image', 'model', 'resid')):
            ok = _copyfile(
                os.path.join(output_dir, 'coadd', 'cus', brickname,
                             f'legacysurvey-{brickname}-{imtype}.jpg'),
                os.path.join(output_dir, f'{galaxy}-{stagesuffix}-{suffix}-FUVNUV.jpg'),
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
    from SGA.io import custom_brickname
    from legacypipe.survey import wcs_for_brick, BrickDuck

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


def custom_coadds(onegal, galaxy, survey, radius_mosaic_arcsec, pixscale=PIXSCALE,
                  bands=GRIZ, mp=1, nsigma=None, subsky_radii=None, just_coadds=False,
                  missing_ok=False, force=False, cleanup=True, unwise=True,
                  galex=False, no_gaia=False, no_tycho=False, verbose=False):
    """Build a custom set of large-galaxy coadds.

    """
    from legacypipe.runbrick import main as runbrick
    from SGA.io import custom_brickname

    stagesuffix = 'coadds'

    width = _mosaic_width(radius_mosaic_arcsec, pixscale=pixscale)
    brickname = f'custom-{custom_brickname(onegal[RACOLUMN], onegal[DECCOLUMN])}'

    # Quickly read the input CCDs and check that we have all the colors we need.
    ccds = get_ccds(survey, onegal[RACOLUMN], onegal[DECCOLUMN], width, pixscale, bands=bands)
    if len(ccds) == 0:
        log.info('No CCDs touching this brick; nothing to do.')
        return 1, stagesuffix
    
    #usebands = list(sorted(set(ccds.filter)))
    #these = [filt in usebands for filt in bands]
    #print('Bands touching this brick, {}'.format(' '.join([filt for filt in usebands])))
    #if np.sum(these) < len(bands) and require_grz:
    #    print('Missing imaging in at least grz and require_grz=True; nothing to do.')
    #    ccdsfile = os.path.join(survey.output_dir, '{}-ccds-{}.fits'.format(galaxy, run))
    #    # should we write out the CCDs file?
    #    print('Writing {} CCDs to {}'.format(len(ccds), ccdsfile))
    #    ccds.writeto(ccdsfile, overrite=True)
    #    return 1, stagesuffix

    # Run the pipeline!
    cmdargs = f'--radec {onegal[RACOLUMN]} {onegal[DECCOLUMN]} '
    cmdargs += f'--width={width} --height={width} --pixscale={pixscale} '
    cmdargs += f'--threads={mp} --outdir={survey.output_dir} --bands={",".join(bands)} '
    cmdargs += f'--survey-dir={survey.survey_dir} '

    #cmdargs += '--write-stage=tims --write-stage=srcs '
    cmdargs += '--write-stage=srcs '

    cmdargs += '--skip-calibs '
    cmdargs += f'--checkpoint={survey.output_dir}/{galaxy}-{stagesuffix}-checkpoint.p '
    cmdargs += f'--pickle={survey.output_dir}/{galaxy}-{stagesuffix}-%%(stage)s.p '
    if just_coadds:
        unwise = False
        galex = False
        cmdargs += '--stage=image_coadds '
    if not unwise:
        cmdargs += '--no-unwise-coadds --no-wise '
    if galex:
        cmdargs += '--galex '
    if no_gaia:
        cmdargs += '--no-gaia '
    if no_tycho:
        cmdargs += '--no-tycho '
    if force:
        cmdargs += '--force-all '
        checkpointfile = f'{survey.output_dir}/{galaxy}-{stagesuffix}-checkpoint.p'
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
    cmdargs += '--fit-on-coadds --no-ivar-reweighting '

    if nsigma:
        cmdargs += f'--nsigma {nsigma:.0f} '

    err = runbrick(args=cmdargs.split())

    # optionally write out the un WISE PSFs
    if unwise or galex:
        import fitsio

        cat = fitsio.read(os.path.join(survey.output_dir, 'tractor', 'cus', f'tractor-{brickname}.fits'),
                          columns=['brick_primary', 'ref_cat', 'ref_id', 'wise_coadd_id', 'brickname'])
        cat = cat[cat['brick_primary']]
        psffile = os.path.join(survey.output_dir, 'coadd', 'cus', brickname,
                               f'legacysurvey-{brickname}-copsf-r.fits.fz')
        if not os.path.isfile(psffile):
            psffile = os.path.join(survey.output_dir, f'{galaxy}-{stagesuffix}-psf-r.fits.fz')
        hdr = fitsio.read_header(psffile)

        for remcard in ('MJD', 'MJD_TAI', 'PSF_SIG', 'INPIXSC'):
            hdr.delete(remcard)

        if unwise:
            import unwise_psf.unwise_psf as unwise_psf

            # FIXME
            thisgal = 0
            coadd_id = cat['wise_coadd_id'][thisgal]

            # coadd_id can be blank in regions around, e.g., Globular Clusters,
            # where we turn off forced photometry.
            if coadd_id == '':
                from legacypipe.unwise import unwise_tiles_touching_wcs
                from legacypipe.survey import wcs_for_brick, BrickDuck
                brick = BrickDuck(onegal[racolumn], onegal[deccolumn], brickname)
                targetwcs = wcs_for_brick(brick, W=float(width), H=float(width), pixscale=pixscale)            
                tiles = unwise_tiles_touching_wcs(targetwcs)
                coadd_id = tiles.coadd_id[0] # grab the first one

            #hdr['PIXSCAL'] = 2.75
            hdr.delete('PIXSCAL')
            hdr.add_record(dict(name='PIXSCAL', value=2.75, comment='pixel scale (arcsec)'))
            hdr.add_record(dict(name='COADD_ID', value=coadd_id, comment='WISE coadd ID'))

            # https://github.com/legacysurvey/legacypipe/blob/main/py/legacypipe/unwise.py#L267-L310        
            fluxrescales = {1: 1.04, 2: 1.005, 3: 1.0, 4: 1.0}
            for band in (1, 2, 3, 4):
                wband = f'W{band}'
                #hdr['BAND'] = wband
                hdr.delete('BAND')
                hdr.add_record(dict(name='BAND', value=wband, comment='Band of this coadd/PSF'))
    
                #psfimg = unwise_psf.get_unwise_psf(band, coadd_id)
                #psfimg /= psfimg.sum()
    
                if (band == 1) or (band == 2):
                    # we only have updated PSFs for W1 and W2
                    psfimg = unwise_psf.get_unwise_psf(band, coadd_id, modelname='neo6_unwisecat')
                    #psfimg = unwise_psf.get_unwise_psf(band, coadd_id, modelname='neo7_unwisecat')
                else:
                    psfimg = unwise_psf.get_unwise_psf(band, coadd_id)
    
                if band == 4:
                    # oversample (the unwise_psf models are at native W4 5.5"/pix,
                    # while the unWISE coadds are made at 2.75"/pix.
                    ph,pw = psfimg.shape
                    subpsf = np.zeros((ph*2-1, pw*2-1), np.float32)
                    from astrometry.util.util import lanczos3_interpolate
                    xx,yy = np.meshgrid(np.arange(0., pw-0.51, 0.5, dtype=np.float32),
                                        np.arange(0., ph-0.51, 0.5, dtype=np.float32))
                    xx = xx.ravel()
                    yy = yy.ravel()
                    ix = xx.astype(np.int32)
                    iy = yy.astype(np.int32)
                    dx = (xx - ix).astype(np.float32)
                    dy = (yy - iy).astype(np.float32)
                    psfimg = psfimg.astype(np.float32)
                    rtn = lanczos3_interpolate(ix, iy, dx, dy, [subpsf.flat], [psfimg])
    
                    psfimg = subpsf
                    del xx, yy, ix, iy, dx, dy
    
                psfimg /= psfimg.sum()
                psfimg *= fluxrescales[band]
                with survey.write_output('copsf', brick=brickname, band=wband) as out:
                    out.fits.write(psfimg, header=hdr)
    
            hdr.delete('COADD_ID')

        if galex:
            from legacypipe.galex import galex_psf
    
            #hdr['PIXSCAL'] = 1.50
            hdr.delete('PIXSCAL')
            hdr.add_record(dict(name='PIXSCAL', value=1.50, comment='pixel scale (arcsec)'))
            gband = {'f': 'FUV', 'n': 'NUV'}
            for band in ('f', 'n'):
                #hdr['BAND'] = gband[band]
                hdr.delete('BAND')
                hdr.add_record(dict(name='BAND', value=gband[band], comment='Band of this coadd/PSF'))
                psfimg = galex_psf(band, os.getenv('GALEX_DIR'))
                psfimg /= psfimg.sum()
                with survey.write_output('copsf', brick=brickname, band=gband[band]) as out:
                    out.fits.write(psfimg, header=hdr)


    if err != 0:
        log.warning('Something went wrong; please check the logfile.')
        return 0, stagesuffix
    else:
        # Move (rename) files into the desired output directory and clean up.
        if cleanup is False:
            missing_ok = True
        ok = _rearrange_files(galaxy, survey.output_dir, brickname, stagesuffix,
                              unwise=unwise, galex=galex, cleanup=cleanup,
                              just_coadds=just_coadds, clobber=True,
                              bands=bands, missing_ok=missing_ok)

        return ok, stagesuffix
