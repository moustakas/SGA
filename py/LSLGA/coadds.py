"""
LSLGA.coadds
============

Code to generate grzW1W2 custom coadds / mosaics.

python -u legacyanalysis/extract-calibs.py --drdir /project/projectdirs/cosmo/data/legacysurvey/dr5 --radec 342.4942 -0.6706 --width 300 --height 300

"""
import os, sys, pdb
import shutil
import numpy as np
from contextlib import redirect_stdout, redirect_stderr

from astrometry.util.multiproc import multiproc

import LSLGA.misc
from LSLGA.misc import custom_brickname

def _copyfile(infile, outfile):
    if os.path.isfile(infile):
        shutil.copy2(infile, outfile)
        return 1
    else:
        print('Missing file {}; please check the logfile.'.format(infile))
        return 0

def pipeline_coadds(onegal, galaxy=None, survey=None, radius_mosaic=None, nproc=1,
                    pixscale=0.262, splinesky=True, log=None, force=False,
                    no_large_galaxies=False, no_gaia=False, no_tycho=False,
                    apodize=False, cleanup=True):
    """Run legacypipe.runbrick on a custom "brick" centered on the galaxy.

    radius_mosaic in arcsec

    """
    import subprocess

    if survey is None:
        from legacypipe.survey import LegacySurveyData
        survey = LegacySurveyData()
        
    galaxydir = survey.output_dir

    if galaxy is None:
        galaxy = 'galaxy'

    cmd = 'python {legacypipe_dir}/py/legacypipe/runbrick.py '
    cmd += '--radec {ra} {dec} --width {width} --height {width} --pixscale {pixscale} '
    cmd += '--threads {threads} --outdir {outdir} '
    cmd += '--survey-dir {survey_dir} '
    #cmd += '--stage image_coadds --early-coadds '
    #cmd += '--write-stage tims '
    cmd += '--write-stage srcs '
    #cmd += '--force-stage wise_forced '
    cmd += '--min-mjd 0 '
    cmd += '--skip-calibs '
    #cmd += '--no-wise --no-wise-ceres '
    cmd += '--checkpoint {galaxydir}/{galaxy}-runbrick-checkpoint.p '
    cmd += '--pickle {galaxydir}/{galaxy}-runbrick-%%(stage)s.p '
    if apodize:
        cmd += '--apodize '
    if no_gaia:
        cmd += '--no-gaia '
    if no_tycho:
        cmd += '--no-tycho '
    if no_large_galaxies:
        cmd += '--no-large-galaxies '
        
    if force:
        cmd += '--force-all '
        checkpointfile = '{galaxydir}/{galaxy}-runbrick-checkpoint.p'.format(galaxydir=galaxydir, galaxy=galaxy)
        if os.path.isfile(checkpointfile):
            os.remove(checkpointfile)
    if not splinesky:
        cmd += '--no-splinesky '
    
    width = np.ceil(2 * radius_mosaic / pixscale).astype('int') # [pixels]

    cmd = cmd.format(legacypipe_dir=os.getenv('LEGACYPIPE_DIR'), galaxy=galaxy,
                     ra=onegal['RA'], dec=onegal['DEC'], width=width,
                     pixscale=pixscale, threads=nproc, outdir=survey.output_dir,
                     galaxydir=galaxydir, survey_dir=survey.survey_dir)
    
    print(cmd, flush=True, file=log)
    err = subprocess.call(cmd.split(), stdout=log, stderr=log)
    if err != 0:
        print('Something went wrong; please check the logfile.')
        return 0
    else:
        # Move (rename) files into the desired output directory and clean up.
        brickname = 'custom-{}'.format(custom_brickname(onegal['RA'], onegal['DEC']))

        ## (Re)package the outliers images into a single MEF -- temporary hack
        ## until we address legacypipe/#271
        #ccdsfile = os.path.join(survey.output_dir, 'coadd', 'cus', brickname,
        #                       'legacysurvey-{}-ccds.fits'.format(brickname))
        #if not os.path.isfile(ccdsfile):
        #    print('CCDs file {} not found!'.format(ccdsfile))
        #    return 0
        #ccds = survey.cleanup_ccds_table(fits_table(ccdsfile))
        #
        #outliersfile = os.path.join(survey.output_dir, '{}-outliers.fits.fz'.format(galaxy))
        #if os.path.isfile(outliersfile):
        #    os.remove(outliersfile)
        #with fitsio.FITS(outliersfile, 'rw') as ff:
        #    for ccd in ccds:
        #        im = survey.get_image_object(ccd)
        #        suffix = '{}-{}-{}'.format(im.camera, im.expnum, im.ccdname)
        #        maskfile = os.path.join(survey.output_dir, 'metrics', 'cus', brickname,
        #                                'outlier-mask-{}.fits.fz'.format(suffix))
        #        if os.path.isfile(maskfile):
        #            mask, hdr = fitsio.read(maskfile, header=True)
        #            key = '{}-{:02d}-{}'.format(im.name, im.hdu, im.band)
        #            ff.write(mask, extname=key, header=hdr)

        # tractor catalog
        ok = _copyfile(
            os.path.join(survey.output_dir, 'tractor', 'cus', 'tractor-{}.fits'.format(brickname)),
            os.path.join(survey.output_dir, '{}-tractor.fits'.format(galaxy)) )
        if not ok:
            return ok

        # CCDs, maskbits, blob images, and depth images
        ok = _copyfile(
            os.path.join(survey.output_dir, 'coadd', 'cus', brickname,
                         'legacysurvey-{}-ccds.fits'.format(brickname)),
            os.path.join(survey.output_dir, '{}-ccds.fits'.format(galaxy)) )
        if not ok:
            return ok

        ok = _copyfile(
            os.path.join(survey.output_dir, 'coadd', 'cus', brickname,
                         'legacysurvey-{}-maskbits.fits.fz'.format(brickname)),
            os.path.join(survey.output_dir, '{}-maskbits.fits.fz'.format(galaxy)) )
        if not ok:
            return ok

        ok = _copyfile(
            os.path.join(survey.output_dir, 'metrics', 'cus', 'blobs-{}.fits.gz'.format(brickname)),
            os.path.join(survey.output_dir, '{}-blobs.fits.gz'.format(galaxy)) )
        if not ok:
            return ok

        for band in ('g', 'r', 'z'):
            ok = _copyfile(
                os.path.join(survey.output_dir, 'coadd', 'cus', brickname,
                             'legacysurvey-{}-depth-{}.fits.fz'.format(brickname, band)),
                os.path.join(survey.output_dir, '{}-depth-{}.fits.fz'.format(galaxy, band)) )
            if not ok:
                return ok
        
        # Data and model images
        for band in ('g', 'r', 'z'):
            for imtype in ('image', 'model'):
                ok = _copyfile(
                    os.path.join(survey.output_dir, 'coadd', 'cus', brickname,
                                 'legacysurvey-{}-{}-{}.fits.fz'.format(brickname, imtype, band)),
                    os.path.join(survey.output_dir, '{}-pipeline-{}-{}.fits.fz'.format(galaxy, imtype, band)) )
                if not ok:
                    return ok

        for band in ('g', 'r', 'z'):
            ok = _copyfile(
                os.path.join(survey.output_dir, 'coadd', 'cus', brickname,
                             'legacysurvey-{}-invvar-{}.fits.fz'.format(brickname, band)),
                os.path.join(survey.output_dir, '{}-invvar-{}.fits.fz'.format(galaxy, band)) )
            if not ok:
                return ok

        # JPG images

        ## Look for WISE stuff in the unwise module--
        #if unwise:
        #    for band in ('W1', 'W2', 'W3', 'W4'):
        #        for imtype in ('image', 'model', 'invvar'):
        #            ok = _copyfile(
        #                os.path.join(survey.output_dir, 'coadd', 'cus', brickname,
        #                             'legacysurvey-{}-{}-{}.fits.fz'.format(brickname, imtype, band)),
        #                os.path.join(survey.output_dir, '{}-{}-{}.fits.fz'.format(galaxy, imtype, band)) )
        #            if not ok:
        #                return ok
        #
        #    for imtype in ('wise', 'wisemodel'):
        #        ok = _copyfile(
        #            os.path.join(survey.output_dir, 'coadd', 'cus', brickname,
        #                         'legacysurvey-{}-{}.jpg'.format(brickname, imtype)),
        #            os.path.join(survey.output_dir, '{}-{}.jpg'.format(galaxy, imtype)) )
        #        if not ok:
        #            return ok

        for imtype in ('image', 'model', 'resid'):
            ok = _copyfile(
                os.path.join(survey.output_dir, 'coadd', 'cus', brickname,
                             'legacysurvey-{}-{}.jpg'.format(brickname, imtype)),
                os.path.join(survey.output_dir, '{}-pipeline-{}-grz.jpg'.format(galaxy, imtype)) )
            if not ok:
                return ok

        if cleanup:
            shutil.rmtree(os.path.join(survey.output_dir, 'coadd'))
            shutil.rmtree(os.path.join(survey.output_dir, 'metrics'))
            shutil.rmtree(os.path.join(survey.output_dir, 'tractor'))
            shutil.rmtree(os.path.join(survey.output_dir, 'tractor-i'))

        return 1

def _pipeline_coadds(onegal, galaxy=None, survey=None, radius=None, nproc=1,
                    pixscale=0.262, splinesky=True, log=None, force=False,
                    archivedir=None, cleanup=True):
    """Run legacypipe.runbrick on a custom "brick" centered on the galaxy.

    radius in arcsec

    """
    import subprocess

    if survey is None:
        from legacypipe.survey import LegacySurveyData
        survey = LegacySurveyData()

    if archivedir is None:
        archivedir = survey.output_dir

    if galaxy is None:
        galaxy = 'galaxy'

    cmd = 'python {legacypipe_dir}/py/legacypipe/runbrick.py '
    cmd += '--radec {ra} {dec} --width {width} --height {width} --pixscale {pixscale} '
    cmd += '--threads {threads} --outdir {outdir} '
    cmd += '--survey-dir {survey_dir} '
    cmd += '--large-galaxies --apodize '
    #cmd += '--unwise-coadds '
    #cmd += '--force-stage coadds '
    cmd += '--write-stage srcs '
    #cmd += '--stage image_coadds --early-coadds '
    cmd += '--no-write --skip --no-wise-ceres '
    cmd += '--checkpoint {archivedir}/{galaxy}-runbrick-checkpoint.p --checkpoint-period 300 '
    cmd += '--pickle {archivedir}/{galaxy}-runbrick-%%(stage)s.p ' 
    if force:
        cmd += '--force-all '
    if not splinesky:
        cmd += '--no-splinesky '

    width = np.ceil(2 * radius / pixscale).astype('int') # [pixels]

    cmd = cmd.format(legacypipe_dir=os.getenv('LEGACYPIPE_DIR'), galaxy=galaxy,
                     ra=onegal['RA'], dec=onegal['DEC'], width=width,
                     pixscale=pixscale, threads=nproc, outdir=survey.output_dir,
                     archivedir=archivedir, survey_dir=survey.survey_dir)
    
    print(cmd, flush=True, file=log)
    err = subprocess.call(cmd.split(), stdout=log, stderr=log)
    if err != 0:
        print('Something we wrong; please check the logfile.')
        return 0
    else:
        # Move (rename) files into the desired output directory and clean up.
        brickname = 'custom-{}'.format(custom_brickname(onegal['RA'], onegal['DEC']))

        # tractor catalog
        ok = _copyfile(
            os.path.join(survey.output_dir, 'tractor', 'cus', 'tractor-{}.fits'.format(brickname)),
            os.path.join(survey.output_dir, '{}-tractor.fits'.format(galaxy)) )
        if not ok:
            return ok

        # CCDs, maskbits, blob images, and depth images
        ok = _copyfile(
            os.path.join(survey.output_dir, 'coadd', 'cus', brickname,
                         'legacysurvey-{}-ccds.fits'.format(brickname)),
            os.path.join(survey.output_dir, '{}-ccds.fits'.format(galaxy)) )
        if not ok:
            return ok

        ok = _copyfile(
            os.path.join(survey.output_dir, 'coadd', 'cus', brickname,
                         'legacysurvey-{}-maskbits.fits.gz'.format(brickname)),
            os.path.join(survey.output_dir, '{}-maskbits.fits.gz'.format(galaxy)) )
        if not ok:
            return ok

        ok = _copyfile(
            os.path.join(survey.output_dir, 'metrics', 'cus', 'blobs-{}.fits.gz'.format(brickname)),
            os.path.join(survey.output_dir, '{}-blobs.fits.gz'.format(galaxy)) )
        if not ok:
            return ok

        for band in ('g', 'r', 'z'):
            ok = _copyfile(
                os.path.join(survey.output_dir, 'coadd', 'cus', brickname,
                             'legacysurvey-{}-depth-{}.fits.fz'.format(brickname, band)),
                os.path.join(survey.output_dir, '{}-depth-{}.fits.fz'.format(galaxy, band)) )
        if not ok:
            return ok
        
        # Data and model images
        for band in ('g', 'r', 'z'):
            for imtype in ('image', 'model'):
                ok = _copyfile(
                    os.path.join(survey.output_dir, 'coadd', 'cus', brickname,
                                 'legacysurvey-{}-{}-{}.fits.fz'.format(brickname, imtype, band)),
                    os.path.join(survey.output_dir, '{}-pipeline-{}-{}.fits.fz'.format(galaxy, imtype, band)) )
                if not ok:
                    return ok

        for band in ('g', 'r', 'z'):
            ok = _copyfile(
                os.path.join(survey.output_dir, 'coadd', 'cus', brickname,
                             'legacysurvey-{}-invvar-{}.fits.fz'.format(brickname, band)),
                os.path.join(survey.output_dir, '{}-invvar-{}.fits.fz'.format(galaxy, band)) )
            if not ok:
                return ok

        #if False: # WISE stuff was moved to the unwise.py module
        #    for band in ('W1', 'W2'):
        #        for imtype in ('image', 'model'):
        #            ok = _copyfile(
        #                os.path.join(survey.output_dir, 'coadd', 'cus', brickname,
        #                             'legacysurvey-{}-{}-{}.fits.fz'.format(brickname, imtype, band)),
        #                os.path.join(survey.output_dir, '{}-{}-{}.fits.fz'.format(galaxy, imtype, band)) )
        #            if not ok:
        #                return ok
        #    for imtype in ('wise', 'wisemodel'):
        #        ok = _copyfile(
        #            os.path.join(survey.output_dir, 'coadd', 'cus', brickname,
        #                         'legacysurvey-{}-{}.jpg'.format(brickname, imtype)),
        #            os.path.join(survey.output_dir, '{}-{}.jpg'.format(galaxy, imtype)) )
        #        if not ok:
        #            return ok

        for imtype in ('image', 'model', 'resid'):
            ok = _copyfile(
                os.path.join(survey.output_dir, 'coadd', 'cus', brickname,
                             'legacysurvey-{}-{}.jpg'.format(brickname, imtype)),
                os.path.join(survey.output_dir, '{}-pipeline-{}-grz.jpg'.format(galaxy, imtype)) )
            if not ok:
                return ok

        if cleanup:
            shutil.rmtree(os.path.join(survey.output_dir, 'coadd'))
            shutil.rmtree(os.path.join(survey.output_dir, 'metrics'))
            shutil.rmtree(os.path.join(survey.output_dir, 'tractor'))
            shutil.rmtree(os.path.join(survey.output_dir, 'tractor-i'))

        return 1

def _custom_sky(skyargs):
    """Perform custom sky-subtraction on a single CCD (with multiprocessing).

    """
    from scipy.stats import sigmaclip
    from scipy.ndimage.morphology import binary_dilation
    from legacypipe.runbrick import stage_srcs

    survey, onegal, radius_arcsec, ccd = skyargs

    #print(ccd.image_filename, ccd.camera)
    #if ccd.image_filename == 'mosaic/CP20160313v2/k4m_160314_074016_ooi_zd_v2.fits.fz':
    #    pdb.set_trace()
    im = survey.get_image_object(ccd)
    print(im, im.band, 'exptime', im.exptime, 'propid', ccd.propid,
          'seeing {:.2f}'.format(ccd.fwhm * im.pixscale), 
          'object', getattr(ccd, 'object', None))

    radius = np.round(radius_arcsec / im.pixscale).astype('int') # [pixels]

    tim = im.get_tractor_image(splinesky=True, subsky=False,
                               hybridPsf=True, normalizePsf=True)

    mp = multiproc()
    targetwcs, bands = tim.subwcs, tim.band
    H, W = targetwcs.shape
    H, W = np.int(H), np.int(W)

    S = stage_srcs(pixscale=im.pixscale, targetwcs=targetwcs, W=W, H=H,
                   bands=bands, tims=[tim], mp=mp, nsigma=5, survey=survey,
                   gaia_stars=True, star_clusters=False)

    mask = S['blobs'] != -1 # 1 = bad
    mask = np.logical_or( mask, tim.getInvvar() <= 0 )
    mask = binary_dilation(mask, iterations=2)

    # Mask the full extent of the central galaxy.
    #http://stackoverflow.com/questions/8647024/how-to-apply-a-disc-shaped-mask-to-a-numpy-array
    _, x0, y0 = targetwcs.radec2pixelxy(onegal['RA'], onegal['DEC'])
    xcen, ycen = np.round(x0 - 1).astype('int'), np.round(y0 - 1).astype('int')

    ymask, xmask = np.ogrid[-ycen:H-ycen, -xcen:W-xcen]
    galmask = (xmask**2 + ymask**2) <= radius**2

    #if im.name == 'decam-00603132-S5':
    #    import matplotlib.pyplot as plt
    #    plt.imshow(mask, origin='lower') ; plt.show()
    #    import pdb ; pdb.set_trace()
    mask = np.logical_or( mask, galmask )

    # Finally estimate the new (constant) sky background.
    # sigma_clipped_stats(image, mask=mask)
    image = tim.getImage()
    good = np.flatnonzero(~mask)
    cimage, _, _ = sigmaclip(image.flat[good], low=2.0, high=2.0)
    newsky = np.median(cimage)
    #newsky = 2.5 * np.median(cimage) - 1.5 * np.mean(cimage)

    out = dict()
    key = '{}-{:02d}'.format(im.name, im.hdu)
    out['{}-mask'.format(key)] = mask
    out['{}-sky'.format(key)] = newsky
    
    return out

def custom_coadds(onegal, galaxy=None, survey=None, radius=None, nproc=1,
                  pixscale=0.262, log=None, plots=False, verbose=False,
                  cleanup=True):
    """Build a custom set of coadds for a single galaxy, with a custom mask and sky
    model.

    radius in arcsec

    """
    from astropy.io import fits

    from astrometry.util.fits import fits_table
    from astrometry.libkd.spherematch import match_radec
    from tractor.sky import ConstantSky

    from legacypipe.runbrick import stage_tims
    from legacypipe.catalog import read_fits_catalog
    from legacypipe.runbrick import _get_mod
    from legacypipe.coadds import make_coadds, write_coadd_images
    from legacypipe.survey import get_rgb, imsave_jpeg
            
    if survey is None:
        from legacypipe.survey import LegacySurveyData
        survey = LegacySurveyData()

    if galaxy is None:
        galaxy = 'galaxy'
        
    brickname = custom_brickname(onegal['RA'], onegal['DEC'])

    if plots:
        from astrometry.util.plotutils import PlotSequence
        ps = PlotSequence('qa-{}'.format(brickname))
    else:
        ps = None

    mp = multiproc(nthreads=nproc)

    width = np.ceil(2 * radius / pixscale).astype('int') # [pixels]

    unwise_dir = os.environ.get('UNWISE_COADDS_DIR', None)    

    # [1] Initialize the "tims" stage of the pipeline, returning a
    # dictionary with the following keys:
    
    #   ['brickid', 'target_extent', 'version_header', 'targetrd',
    #    'brickname', 'pixscale', 'bands', 'survey', 'brick', 'ps',
    #    'H', 'ccds', 'W', 'targetwcs', 'tims']

    def call_stage_tims():
        """Note that we return just the portion of the CCDs centered on the galaxy, and
        that we turn off sky-subtraction.

        """
        return stage_tims(ra=onegal['RA'], dec=onegal['DEC'], brickname=brickname,
                          survey=survey, W=width, H=width, pixscale=pixscale,
                          mp=mp, normalizePsf=True, pixPsf=True, hybridPsf=True,
                          splinesky=True, subsky=False, # note!
                          depth_cut=False, apodize=False, do_calibs=False, rex=True, 
                          unwise_dir=unwise_dir, plots=plots, ps=ps)

    if log:
        with redirect_stdout(log), redirect_stderr(log):
            P = call_stage_tims()
    else:
        P = call_stage_tims()

    tims = P['tims']

    # [2] Derive the custom mask and sky background for each (full) CCD and
    # write out a MEF -custom-mask.fits.gz file.
    skyargs = [(survey, onegal, radius, _ccd) for _ccd in survey.ccds]
    result = mp.map(_custom_sky, skyargs)
    #result = list( zip( *mp.map(_custom_sky, args) ) )
    sky = dict()
    [sky.update(res) for res in result]
    del result

    hx = fits.HDUList()
    for ii, ccd in enumerate(survey.ccds):
        im = survey.get_image_object(ccd)
        key = '{}-{:02d}'.format(im.name, im.hdu)
        mask = sky['{}-mask'.format(key)].astype('uint8')
        hdu = fits.ImageHDU(mask, name=key)
        hdu.header['SKY'] = sky['{}-sky'.format(key)]
        hx.append(hdu)

    maskfile = os.path.join(survey.output_dir, '{}-custom-mask.fits.gz'.format(galaxy))
    print('Writing {}'.format(maskfile))
    hx.writeto(maskfile, overwrite=True)

    # [3] Modify each tim by subtracting our new estimate of the sky.
    newtims = []
    for tim in tims:
        image = tim.getImage()
        newsky = sky['{}-{:02d}-sky'.format(tim.imobj.name, tim.imobj.hdu)]
        if False:
            pipesky = tim.getSky()
            splinesky = np.zeros_like(image)
            pipesky.addTo(splinesky)
            newsky = splinesky
        tim.setImage(image - newsky)
        tim.sky = ConstantSky(0)
        newtims.append(tim)

    # [4] Read the Tractor catalog and render the model image of each CCD, with
    # and without the central large galaxy.
    tractorfile = os.path.join(survey.output_dir, '{}-tractor.fits'.format(galaxy))
    if not os.path.isfile(tractorfile):
        print('Missing Tractor catalog {}'.format(tractorfile))
        return 0
    
    cat = fits_table(tractorfile)
    print('Read {} sources from {}'.format(len(cat), tractorfile), flush=True, file=log)

    # Find and remove all the objects within XX arcsec of the target
    # coordinates.
    m1, m2, d12 = match_radec(cat.ra, cat.dec, onegal['RA'], onegal['DEC'], 5/3600.0, nearest=False)
    if len(d12) == 0:
        print('No matching galaxies found -- probably not what you wanted.')
        #raise ValueError
        keep = np.ones(len(T)).astype(bool)
    else:
        keep = ~np.isin(cat.objid, cat[m1].objid)        

    print('Creating tractor sources...', flush=True, file=log)
    srcs = read_fits_catalog(cat, fluxPrefix='')
    srcs_nocentral = np.array(srcs)[keep].tolist()
    
    if False:
        print('Sources:')
        [print(' ', src) for src in srcs]

    print('Rendering model images with and without surrounding galaxies...', flush=True, file=log)
    modargs = [(tim, srcs) for tim in newtims]
    mods = mp.map(_get_mod, modargs)

    modargs = [(tim, srcs_nocentral) for tim in newtims]
    mods_nocentral = mp.map(_get_mod, modargs)

    # [5] Build the custom coadds, with and without the surrounding galaxies.
    print('Producing coadds...', flush=True, file=log)
    def call_make_coadds(usemods):
        return make_coadds(newtims, P['bands'], P['targetwcs'], mods=usemods, mp=mp,
                           callback=write_coadd_images,
                           callback_args=(survey, brickname, P['version_header'], 
                                          newtims, P['targetwcs']))

    # Custom coadds (all galaxies).
    if log:
        with redirect_stdout(log), redirect_stderr(log):
            C = call_make_coadds(mods)
    else:
        C = call_make_coadds(mods)

    for suffix in ('image', 'model'):
        for band in P['bands']:
            ok = _copyfile(
                os.path.join(survey.output_dir, 'coadd', brickname[:3], 
                                   brickname, 'legacysurvey-{}-{}-{}.fits.fz'.format(
                    brickname, suffix, band)),
                #os.path.join(survey.output_dir, '{}-{}-{}.fits.fz'.format(galaxy, suffix, band)) )
                os.path.join(survey.output_dir, '{}-custom-{}-{}.fits.fz'.format(galaxy, suffix, band)) )
            if not ok:
                return ok

    # Custom coadds (without the central).
    if log:
        with redirect_stdout(log), redirect_stderr(log):
            C_nocentral = call_make_coadds(mods_nocentral)
    else:
        C_nocentral = call_make_coadds(mods_nocentral)

    # Move (rename) the coadds into the desired output directory - no central.
    #for suffix in ('image', 'model'):
    for suffix in np.atleast_1d('model'):
        for band in P['bands']:
            ok = _copyfile(
                os.path.join(survey.output_dir, 'coadd', brickname[:3], 
                                   brickname, 'legacysurvey-{}-{}-{}.fits.fz'.format(
                    brickname, suffix, band)),
                #os.path.join(survey.output_dir, '{}-{}-nocentral-{}.fits.fz'.format(galaxy, suffix, band)) )
                os.path.join(survey.output_dir, '{}-custom-{}-nocentral-{}.fits.fz'.format(galaxy, suffix, band)) )
            if not ok:
                return ok
            
    if cleanup:
        shutil.rmtree(os.path.join(survey.output_dir, 'coadd'))

    # [6] Finally, build png images.
    def call_make_png(C, nocentral=False):
        rgbkwargs = dict(mnmx=(-1, 100), arcsinh=1)
        #rgbkwargs_resid = dict(mnmx=(0.1, 2), arcsinh=1)
        rgbkwargs_resid = dict(mnmx=(-1, 100), arcsinh=1)

        if nocentral:
            coadd_list = [('custom-model-nocentral', C.comods, rgbkwargs),
                          ('custom-image-central', C.coresids, rgbkwargs_resid)]
        else:
            coadd_list = [('custom-image', C.coimgs,   rgbkwargs),
                          ('custom-model', C.comods,   rgbkwargs),
                          ('custom-resid', C.coresids, rgbkwargs_resid)]

        for suffix, ims, rgbkw in coadd_list:
            rgb = get_rgb(ims, P['bands'], **rgbkw)
            kwa = {}
            outfn = os.path.join(survey.output_dir, '{}-{}-grz.jpg'.format(galaxy, suffix))
            print('Writing {}'.format(outfn), flush=True, file=log)
            imsave_jpeg(outfn, rgb, origin='lower', **kwa)
            del rgb

    call_make_png(C, nocentral=False)
    call_make_png(C_nocentral, nocentral=True)

    return 1
