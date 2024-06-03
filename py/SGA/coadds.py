"""
SGA.coadds
==========

"""
import os, pdb
import numpy as np

BANDS = ['g', 'r', 'i', 'z']


def custom_brickname(ra, dec):
    brickname = '{:06d}{}{:05d}'.format(
        int(1000*ra), 'm' if dec < 0 else 'p',
        int(1000*np.abs(dec)))
    return brickname


def _mosaic_width(radius_mosaic, pixscale):
    """Ensure the mosaic is an odd number of pixels so the central can land on a
    whole pixel (important for ellipse-fitting).

    radius_mosaic in arcsec

    """
    #width = np.ceil(2 * radius_mosaic / pixscale).astype('int') # [pixels]
    width = 2 * radius_mosaic / pixscale # [pixels]
    width = (np.ceil(width) // 2 * 2 + 1).astype('int') # [pixels]
    return width


def _rearrange_files(galaxy, output_dir, brickname, stagesuffix, run,
                     bands=BANDS,
                     unwise=True, galex=False, cleanup=False, just_coadds=False,
                     clobber=False, require_grz=True, missing_ok=False,
                     write_wise_psf=False):
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
                print('Warning: missing file {} but missing_ok=True'.format(infile))
                return 1
            else:
                print('Missing file {}; please check the logfile.'.format(infile))
                return 0

    def _do_cleanup():
        import shutil
        from glob import glob
        shutil.rmtree(os.path.join(output_dir, 'coadd'), ignore_errors=True)
        shutil.rmtree(os.path.join(output_dir, 'metrics'), ignore_errors=True)
        shutil.rmtree(os.path.join(output_dir, 'tractor'), ignore_errors=True)
        shutil.rmtree(os.path.join(output_dir, 'tractor-i'), ignore_errors=True)
        picklefiles = glob(os.path.join(output_dir, '{}-{}-*.p'.format(galaxy, stagesuffix)))
        for picklefile in picklefiles:
            if os.path.isfile(picklefile):
                os.remove(picklefile)

    # Diagnostic plots (OK if they're missing) (put this above the check for the
    # CCDs--if the plots have been made then they are useful for testing and
    # debugging).
    for qatype in ['ccdpos']:
    #for qatype in ['ccdpos', 'pipelinesky', 'customsky']:
        ok = _copyfile(
            os.path.join(output_dir, 'metrics', 'cus', 'fitoncoadds-{}-{}.jpg'.format(brickname, qatype)),
                         os.path.join(output_dir, '{}-{}-{}.jpg'.format(galaxy, stagesuffix, qatype)),
            clobber=clobber, missing_ok=True)
        if not ok:
            return ok

    # If we made it here and there is no CCDs file it's because legacypipe
    # exited cleanly with "No photometric CCDs touching brick."
    _ccdsfile = os.path.join(output_dir, 'coadd', 'cus', brickname,
                            'legacysurvey-{}-ccds.fits'.format(brickname))
    if not os.path.isfile(_ccdsfile) and missing_ok is False:
        print('No photometric CCDs touching brick.')
        if cleanup:
            _do_cleanup()
        return 1
    
    ccdsfile = os.path.join(output_dir, '{}-ccds-{}.fits'.format(galaxy, run))
    ok = _copyfile(
        os.path.join(output_dir, 'coadd', 'cus', brickname,
                     'legacysurvey-{}-ccds.fits'.format(brickname)), ccdsfile,
        clobber=clobber, missing_ok=missing_ok)
    if not ok:
        return ok

    # For objects on the edge of the footprint we can sometimes lose 3-band
    # coverage if one of the bands is fully masked. Check here and write out all
    # the files except a 
    if os.path.isfile(ccdsfile): # can be missing during testing if missing_ok=True
        allbands = fitsio.read(ccdsfile, columns='filter')
        ubands = list(sorted(set(allbands)))

        if require_grz and ('g' not in ubands or 'r' not in ubands or 'z' not in ubands):
            print('Lost grz coverage and require_grz=True.')
            if cleanup:
                _do_cleanup()
            return 1
    #else:
    #    bands = ('g', 'r', 'z')

    # image coadds (FITS + JPG)
    for band in bands:
        for imtype, outtype in zip(('image', 'invvar'), ('image', 'invvar')):
            ok = _copyfile(
                os.path.join(output_dir, 'coadd', 'cus', brickname,
                             'legacysurvey-{}-{}-{}.fits.fz'.format(brickname, imtype, band)),
                             os.path.join(output_dir, '{}-{}-{}-{}.fits.fz'.format(galaxy, stagesuffix, outtype, band)),
                clobber=clobber, missing_ok=missing_ok, update_header=True)
            if not ok:
                return ok

    # JPG images
    ok = _copyfile(
        os.path.join(output_dir, 'coadd', 'cus', brickname,
                     'legacysurvey-{}-image.jpg'.format(brickname)),
        os.path.join(output_dir, '{}-{}-image-grz.jpg'.format(galaxy, stagesuffix)),
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
                                 'legacysurvey-{}-{}-{}.fits.fz'.format(brickname, imtype, band)),
                                 os.path.join(output_dir, '{}-{}-{}-{}.fits.fz'.format(galaxy, stagesuffix, outtype, band)),
                    clobber=clobber, missing_ok=missing_ok)
                if not ok:
                    return ok

        if write_wise_psf:
            for band in ['W1', 'W2', 'W3', 'W4', 'FUV', 'NUV']:
                for imtype, outtype in zip(['copsf'], ['psf']):
                    ok = _copyfile(
                        os.path.join(output_dir, 'coadd', 'cus', brickname,
                                     'legacysurvey-{}-{}-{}.fits.fz'.format(brickname, imtype, band)),
                                     os.path.join(output_dir, '{}-{}-{}-{}.fits.fz'.format(galaxy, stagesuffix, outtype, band)),
                        clobber=clobber, missing_ok=missing_ok)
                    if not ok:
                        return ok

    # tractor catalog
    ok = _copyfile(
        os.path.join(output_dir, 'tractor', 'cus', 'tractor-{}.fits'.format(brickname)),
        os.path.join(output_dir, '{}-{}-tractor.fits'.format(galaxy, stagesuffix)),
        clobber=clobber, missing_ok=missing_ok)
    if not ok:
        return ok

    # Maskbits, blob images, outlier masks, and depth images.
    ok = _copyfile(
        os.path.join(output_dir, 'coadd', 'cus', brickname,
                     'legacysurvey-{}-maskbits.fits.fz'.format(brickname)),
        os.path.join(output_dir, '{}-{}-maskbits.fits.fz'.format(galaxy, stagesuffix)),
        clobber=clobber, missing_ok=missing_ok)
    if not ok:
        return ok

    if False:
        ok = _copyfile(
            os.path.join(output_dir, 'metrics', 'cus', 'blobs-{}.fits.gz'.format(brickname)),
            os.path.join(output_dir, '{}-{}-blobs.fits.gz'.format(galaxy, stagesuffix)),
            clobber=clobber, missing_ok=missing_ok)
        if not ok:
            return ok

    ok = _copyfile(
        os.path.join(output_dir, 'metrics', 'cus', 'outlier-mask-{}.fits.fz'.format(brickname)),
        os.path.join(output_dir, '{}-{}-outlier-mask.fits.fz'.format(galaxy, stagesuffix)),
        clobber=clobber, missing_ok=missing_ok)
    if not ok:
        return ok

    if False:
        for band in ['g', 'r', 'z']:
            ok = _copyfile(
                os.path.join(output_dir, 'coadd', 'cus', brickname,
                             'legacysurvey-{}-depth-{}.fits.fz'.format(brickname, band)),
                os.path.join(output_dir, '{}-depth-{}.fits.fz'.format(galaxy, band)),
                clobber=clobber, missing_ok=missing_ok)
            if not ok:
                return ok

    # model coadds
    for band in bands:
        for imtype in ['model']:
        #for imtype in ('model', 'blobmodel'):
            ok = _copyfile(
                os.path.join(output_dir, 'coadd', 'cus', brickname,
                             'legacysurvey-{}-{}-{}.fits.fz'.format(brickname, imtype, band)),
                os.path.join(output_dir, '{}-{}-{}-{}.fits.fz'.format(galaxy, stagesuffix, imtype, band)),
                clobber=clobber, missing_ok=missing_ok)
            if not ok:
                return ok

    # JPG images
    for imtype in ('model', 'resid'):
        ok = _copyfile(
            os.path.join(output_dir, 'coadd', 'cus', brickname,
                         'legacysurvey-{}-{}.jpg'.format(brickname, imtype)),
            os.path.join(output_dir, '{}-{}-{}-grz.jpg'.format(galaxy, stagesuffix, imtype)),
            clobber=clobber, missing_ok=missing_ok)
        if not ok:
            return ok

    # WISE
    if unwise:
        for band in ('W1', 'W2', 'W3', 'W4'):
            for imtype in ('image', 'invvar'):
                ok = _copyfile(
                    os.path.join(output_dir, 'coadd', 'cus', brickname,
                                 'legacysurvey-{}-{}-{}.fits.fz'.format(brickname, imtype, band)),
                    os.path.join(output_dir, '{}-{}-{}-{}.fits.fz'.format(galaxy, stagesuffix, imtype, band)),
                    clobber=clobber, missing_ok=missing_ok)
                if not ok:
                    return ok

            ok = _copyfile(
                os.path.join(output_dir, 'coadd', 'cus', brickname,
                             'legacysurvey-{}-model-{}.fits.fz'.format(brickname, band)),
                os.path.join(output_dir, '{}-{}-model-{}.fits.fz'.format(galaxy, stagesuffix, band)),
                    clobber=clobber, missing_ok=missing_ok)
            if not ok:
                return ok

        for imtype, suffix in zip(('wise', 'wisemodel', 'wiseresid'), ('image', 'model', 'resid')):
            ok = _copyfile(
                os.path.join(output_dir, 'coadd', 'cus', brickname,
                             'legacysurvey-{}-{}.jpg'.format(brickname, imtype)),
                os.path.join(output_dir, '{}-{}-{}-W1W2.jpg'.format(galaxy, stagesuffix, suffix)),
                    clobber=clobber, missing_ok=missing_ok)
            if not ok:
                return ok

    if galex:
        # GALEX imaging can be missing completely around bright stars, so don't fail.
        for band in ('FUV', 'NUV'):
            for imtype in ('image', 'invvar'):
                ok = _copyfile(
                    os.path.join(output_dir, 'coadd', 'cus', brickname,
                                 'legacysurvey-{}-{}-{}.fits.fz'.format(brickname, imtype, band)),
                    os.path.join(output_dir, '{}-{}-{}-{}.fits.fz'.format(galaxy, stagesuffix, imtype, band)),
                    clobber=clobber, missing_ok=True)
                if not ok:
                    return ok

            ok = _copyfile(
                os.path.join(output_dir, 'coadd', 'cus', brickname,
                             'legacysurvey-{}-model-{}.fits.fz'.format(brickname, band)),
                os.path.join(output_dir, '{}-{}-model-{}.fits.fz'.format(galaxy, stagesuffix, band)),
                    clobber=clobber, missing_ok=missing_ok)
            if not ok:
                return ok

        for imtype, suffix in zip(('galex', 'galexmodel', 'galexresid'), ('image', 'model', 'resid')):
            ok = _copyfile(
                os.path.join(output_dir, 'coadd', 'cus', brickname,
                             'legacysurvey-{}-{}.jpg'.format(brickname, imtype)),
                os.path.join(output_dir, '{}-{}-{}-FUVNUV.jpg'.format(galaxy, stagesuffix, suffix)),
                    clobber=clobber, missing_ok=missing_ok)
            if not ok:
                return ok

    if cleanup:
        _do_cleanup()

    return 1


def get_ccds(survey, ra, dec, pixscale, width, bands=['g', 'r', 'z']):
    """Quickly get the CCDs touching this custom brick.  This code is mostly taken
    from legacypipe.runbrick.stage_tims.

    """
    from legacypipe.survey import wcs_for_brick, BrickDuck
    brickname = 'custom-{}'.format(custom_brickname(ra, dec))
    brick = BrickDuck(ra, dec, brickname)

    targetwcs = wcs_for_brick(brick, W=float(width), H=float(width), pixscale=pixscale)
    ccds = survey.ccds_touching_wcs(targetwcs)

    if ccds is None or np.sum(ccds.ccd_cuts == 0) == 0:
        return []
    ccds.cut(ccds.ccd_cuts == 0)
    ccds.cut(np.array([b in bands for b in ccds.filter]))

    return ccds


def detection_coadds(brick, survey, mp=1, pixscale=0.262, run='south',
                     gaussian_kernels='5,10,30', stagesuffix='detection-coadds',
                     overwrite=False):
    """Build the detection coadds.

    brick - table with brick information (brickname, ra, dec, etc.)
    
    """
    import subprocess

    brickname = brick['BRICKNAME']
    bands = ','.join(survey.allbands)
    detection_kernels = ','.join(np.int16(np.array(gaussian_kernels.split(',')).astype(float) / pixscale).astype(str))

    ## Quickly read the input CCDs and check that we have all the colors we need.
    #ccds = get_ccds(survey, onegal[racolumn], onegal[deccolumn], pixscale, width, bands=bands)
    #if len(ccds) == 0:
    #    print('No CCDs touching this brick; nothing to do.')
    #    return 1
    #
    #usebands = list(sorted(set(ccds.filter)))
    #these = [filt in usebands for filt in bands]
    #print('Bands touching this brick, {}'.format(' '.join([filt for filt in usebands])))
    #if np.sum(these) < len(bands) and require_grz:
    #    print('Missing imaging in at least grz and require_grz=True; nothing to do.')
    #    ccdsfile = os.path.join(survey.output_dir, '{}-ccds-{}.fits'.format(galaxy, run))
    #    # should we write out the CCDs file?
    #    print('Writing {} CCDs to {}'.format(len(ccds), ccdsfile))
    #    ccds.writeto(ccdsfile, overwrite=True)
    #    return 1

    # Build the call to runbrick. If 'WIDTH' is in the bricks table, then this
    # is a custom (test) brick, otherwise its a brick in the survey-bricks
    # table.
    cmd = f'python {os.getenv("LEGACYPIPE_CODE_DIR")}/py/legacypipe/runbrick.py '
    if 'WIDTH' in brick.colnames:
        outdir = os.path.join(survey.output_dir, brickname)
        cmd += f'--radec {brick["RA"]} {brick["DEC"]} --width={brick["WIDTH"]} --height={brick["WIDTH"]} '
    else:
        outdir = survey.output_dir
        cmd += f'--brick {bricks["BRICKNAME"]} '
    cmd += f'--pixscale={pixscale} --bands={bands} --threads={mp} '
    cmd += f'--outdir={outdir} --run={run} '
    cmd += '--no-unwise-coadds --stage=image_coadds --stage=srcs '#--force-stage srcs '
    cmd += f'--detection-kernels={detection_kernels} '
    cmd += f'--checkpoint {outdir}/{brickname}-{stagesuffix}-checkpoint.p '
    cmd += f'--pickle {outdir}/{brickname}-{stagesuffix}-%%(stage)s.p '

    # ignore previous checkpoint and pickle files
    if overwrite:
        cmd += '--force-all '
        checkpointfile = f'{outdir}/{brickname}-{stagesuffix}-checkpoint.p'
        if os.path.isfile(checkpointfile):
            os.remove(checkpointfile)
            
    print(cmd)
    #print(cmd, flush=True, file=log)

    err = subprocess.call(cmd.split())#, stdout=log, stderr=log)
    if err != 0:
        print('WARNING: Something went wrong; please check the logfile.')
        return 0
    else:
        return err
    
    
def custom_coadds(onegal, galaxy=None, survey=None, radius_mosaic=None,
                  nproc=1, pixscale=0.262, run='south', racolumn='RA', deccolumn='DEC',
                  bands=['g', 'r', 'z'],
                  nsigma=None, 
                  log=None, apodize=False, custom=True, unwise=True, galex=False, force=False,
                  plots=False, verbose=False, cleanup=True, missing_ok=False,
                  write_all_pickles=False, no_galex_ceres=False, 
                  #no_subsky=False,
                  subsky_radii=None, #ubercal_sky=False,
                  just_coadds=False, require_grz=True, no_gaia=False,
                  no_tycho=False, write_wise_psf=False):
    """Build a custom set of large-galaxy coadds

    radius_mosaic in arcsec

    You must specify *one* of the following:
      * pipeline - standard call to runbrick
      * custom - for the cluster centrals project; calls stage_largegalaxies but
        with custom sky-subtraction

    """
    import subprocess
    
    if survey is None:
        from legacypipe.survey import LegacySurveyData
        survey = LegacySurveyData()
        
    if galaxy is None:
        galaxy = 'galaxy'

    if custom:
        stagesuffix = 'custom'
    else:
        stagesuffix = 'pipeline'

    width = _mosaic_width(radius_mosaic, pixscale)
    brickname = 'custom-{}'.format(custom_brickname(onegal[racolumn], onegal[deccolumn]))

    # Quickly read the input CCDs and check that we have all the colors we need.
    ccds = get_ccds(survey, onegal[racolumn], onegal[deccolumn], pixscale, width, bands=bands)
    if len(ccds) == 0:
        print('No CCDs touching this brick; nothing to do.')
        return 1, stagesuffix
    
    usebands = list(sorted(set(ccds.filter)))
    these = [filt in usebands for filt in bands]
    print('Bands touching this brick, {}'.format(' '.join([filt for filt in usebands])))
    if np.sum(these) < len(bands) and require_grz:
        print('Missing imaging in at least grz and require_grz=True; nothing to do.')
        ccdsfile = os.path.join(survey.output_dir, '{}-ccds-{}.fits'.format(galaxy, run))
        # should we write out the CCDs file?
        print('Writing {} CCDs to {}'.format(len(ccds), ccdsfile))
        ccds.writeto(ccdsfile, overwrite=True)
        return 1, stagesuffix

    # Run the pipeline!
    cmd = 'python {legacypipe_dir}/py/legacypipe/runbrick.py '
    cmd += '--radec {ra} {dec} --width {width} --height {width} --pixscale {pixscale} '
    cmd += '--threads {threads} --outdir {outdir} --bands {bands} '
    cmd += '--survey-dir {survey_dir} --run {run} '
    if write_all_pickles:
        pass
        #cmd += '--write-stage tims --write-stage srcs '
    else:
        cmd += '--write-stage srcs '
    cmd += '--skip-calibs '
    cmd += '--checkpoint {galaxydir}/{galaxy}-{stagesuffix}-checkpoint.p '
    cmd += '--pickle {galaxydir}/{galaxy}-{stagesuffix}-%%(stage)s.p '
    if just_coadds:
        unwise = False
        cmd += '--stage image_coadds --early-coadds '
    if not unwise:
        cmd += '--no-unwise-coadds --no-wise '
    if galex:
        cmd += '--galex '
    if apodize:
        cmd += '--apodize '
    if no_gaia:
        cmd += '--no-gaia '
    if no_tycho:
        cmd += '--no-tycho '
    if no_galex_ceres:
        cmd += '--no-galex-ceres '
        #cmd += '--no-galex-ceres --no-wise-ceres '
    if force:
        cmd += '--force-all '
        checkpointfile = '{galaxydir}/{galaxy}-{stagesuffix}-checkpoint.p'.format(
            galaxydir=survey.output_dir, galaxy=galaxy, stagesuffix=stagesuffix)
        if os.path.isfile(checkpointfile):
            os.remove(checkpointfile)
    if subsky_radii is not None: # implies --no-subsky
        #if len(subsky_radii) != 3:
        #    raise ValueError('subsky_radii must be a 3-element vector')
        #cmd += '--no-subsky --ubercal-sky --subsky-radii {} {} {} '.format(subsky_radii[0], subsky_radii[1], subsky_radii[2]) # [arcsec]
        cmd += '--no-subsky --ubercal-sky --subsky-radii'
        for rad in subsky_radii:
            cmd += ' {} '.format(rad)
    #if ubercal_sky: # implies --no-subsky
    #    cmd += '--no-subsky --ubercal-sky '

    # stage-specific options here--
    if custom:
        cmd += '--fit-on-coadds --no-ivar-reweighting '
        #cmd += '--fit-on-coadds --saddle-fraction 0.2 --saddle-min 4.0 '
        #cmd += '--nsigma 10 '
    else:
        pass # standard pipeline

    if nsigma:
        cmd += '--nsigma {:.0f} '.format(nsigma)

    #cmd += '--stage fit_on_coadds ' ; cleanup = False ; missing_ok = True
    #cmd += '--stage srcs ' ; cleanup = False
    #cmd += '--stage fitblobs ' ; cleanup = False
    #cmd += '--stage coadds ' ; cleanup = False
    #cmd += '--stage wise_forced ' ; cleanup = False

    cmd = cmd.format(legacypipe_dir=os.getenv('LEGACYPIPE_CODE_DIR'), galaxy=galaxy,
                     ra=onegal[racolumn], dec=onegal[deccolumn], width=width,
                     pixscale=pixscale, threads=nproc, outdir=survey.output_dir,
                     bands=','.join(bands),
                     galaxydir=survey.output_dir, survey_dir=survey.survey_dir, run=run,
                     stagesuffix=stagesuffix)
    print(cmd, flush=True, file=log)
    err = subprocess.call(cmd.split(), stdout=log, stderr=log)
    #err = 0

    # optionally write out the GALEX and WISE PSFs
    if unwise and write_wise_psf:
        import fitsio
        import unwise_psf.unwise_psf as unwise_psf
        from legacypipe.galex import galex_psf

        cat = fitsio.read(os.path.join(survey.output_dir, 'tractor', 'cus', 'tractor-{}.fits'.format(brickname)),
                          columns=['brick_primary', 'ref_cat', 'ref_id', 'wise_coadd_id', 'brickname'])
        cat = cat[cat['brick_primary']]
        psffile = os.path.join(survey.output_dir, 'coadd', 'cus', brickname,
                               'legacysurvey-{}-copsf-r.fits.fz'.format(brickname))
        if not os.path.isfile(psffile):
            psffile = os.path.join(survey.output_dir, '{}-{}-psf-r.fits.fz'.format(galaxy, stagesuffix))
        hdr = fitsio.read_header(psffile)
        
        for remcard in ('MJD', 'MJD_TAI', 'PSF_SIG', 'INPIXSC'):
            hdr.delete(remcard)
            
        thisgal = 0 # fix me
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
            wband = 'W{}'.format(band)
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
        print('Something went wrong; please check the logfile.')
        return 0, stagesuffix
    else:
        # Move (rename) files into the desired output directory and clean up.
        if cleanup is False:
            missing_ok = True
        ok = _rearrange_files(galaxy, survey.output_dir, brickname, stagesuffix,
                              run, unwise=unwise, galex=galex, cleanup=cleanup,
                              just_coadds=just_coadds,
                              clobber=True,
                              bands=bands,
                              write_wise_psf=write_wise_psf,
                              #clobber=force,
                              require_grz=require_grz, missing_ok=missing_ok)
        return ok, stagesuffix
