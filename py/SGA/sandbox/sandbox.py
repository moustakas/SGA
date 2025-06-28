def call_detection_coadds(brick, survey, pixscale=0.262, mp=1, run='south',
                          stagesuffix='detection-coadds', gaussian_kernels='5,10,30',
                          debug=False, overwrite=False):
    """Wrapper script to build the detection coadds.

    brick - table with brick information (brickname, ra, dec, etc.)

    """
    from SGA.coadds import detection_coadds

    brickname = brick['BRICKNAME']
    raslice = SGA.io.get_raslice(brick['RA'])
    #brickname, outdir = SGA.io.get_galaxy_galaxydir(bricks=brick)

    outdir = os.path.join(survey.output_dir, raslice)
    logfile = os.path.join(outdir, brickname, f'{brickname}-{stagesuffix}.log')

    t0 = time.time()
    if debug:
        err = detection_coadds(brick, survey, mp=mp, pixscale=pixscale,
                               run=run, stagesuffix=stagesuffix,
                               gaussian_kernels=gaussian_kernels,
                               outdir=outdir, overwrite=overwrite)
    else:
        #with stdouterr_redirected(to=logfile, overwrite=True):#overwrite):
        #with open(logfile, 'w') as mylog:
        #    with redirect_stdout(mylog), redirect_stderr(mylog):
        log.info(f'Started working on brick {brickname} at {time.asctime()}')
        err = detection_coadds(brick, survey, mp=mp, pixscale=pixscale,
                               run=run, stagesuffix=stagesuffix,
                               gaussian_kernels=gaussian_kernels,
                               outdir=outdir, overwrite=overwrite)
        #pdb.set_trace()
        #dt = time.time() - t0
        #log.info('All done, ', dt)

    if err == 0:
        donefile = os.path.join(outdir, brickname, f'{brickname}-{stagesuffix}.isdone')
    else:
        donefile = os.path.join(outdir, brickname, f'{brickname}-{stagesuffix}.isfail')
        
    subprocess.call(f'touch {donefile}'.split())

                
def call_candidate_cutouts(brick, survey, ssl_width=152, pixscale=0.262,
                           stagesuffix='candidate-cutouts',
                           debug=False, overwrite=False):
    """Wrapper script to generate cutouts of all the candidate large galaxies.

    brick - table with brick information (brickname, ra, dec, etc.)

    """
    from SGA.coadds import candidate_cutouts

    brickname = brick['BRICKNAME']
    raslice = SGA.io.get_raslice(brick['RA'])

    coaddsdir = os.path.join(survey.output_dir, raslice, brickname)
    logfile = os.path.join(coaddsdir, f'{brickname}-{stagesuffix}.log')

    t0 = time.time()
    if debug:
        err = candidate_cutouts(brick, coaddsdir, pixscale=pixscale, bands=survey.allbands,
                                ssl_width=ssl_width, stagesuffix=stagesuffix,
                                overwrite=overwrite)
    else:
        err = candidate_cutouts(brick, coaddsdir, pixscale=pixscale, bands=survey.allbands,
                                ssl_width=ssl_width, stagesuffix=stagesuffix,
                                overwrite=overwrite)

    if err == 0:
        donefile = os.path.join(coaddsdir, f'{brickname}-{stagesuffix}.isdone')
    else:
        donefile = os.path.join(coaddsdir, f'{brickname}-{stagesuffix}.isfail')
    subprocess.call(f'touch {donefile}'.split())

                
        if args.detection_coadds or args.candidate_cutouts:
            from legacypipe.runs import get_survey
            outdir = os.path.join(datadir, 'detection')
            survey = get_survey(args.run, allbands=BANDS, output_dir=outdir)

            if True:
                bricks = SGA.io.read_survey_bricks(survey, custom=True)
            else:
                bricks = SGA.io.read_survey_bricks(survey, brickname='0405m085') # '0159m047')
                
            if len(bricks) == 0:
                return
        else:


def detection_coadds(brick, survey, mp=1, pixscale=PIXSCALE, run='south',
                     gaussian_kernels='5,10,30', stagesuffix='detection-coadds',
                     outdir=None, overwrite=False):
    """Build the detection coadds.

    brick - table with brick information (brickname, ra, dec, etc.)
    
    """
    import subprocess

    brickname = brick['BRICKNAME']
    outdir = os.path.join(outdir, brickname)
    
    bands = ','.join(survey.allbands)
    detection_kernels = ','.join(np.int16(np.array(gaussian_kernels.split(',')).astype(float) / pixscale).astype(str))

    if outdir is None:
        outdir = survey.output_dir

    # Build the call to runbrick. If 'WIDTH' is in the bricks table, then this
    # is a custom (test) brick, otherwise its a brick in the survey-bricks
    # table.
    cmd = f'python {os.getenv("LEGACYPIPE_CODE_DIR")}/py/legacypipe/runbrick.py '
    if 'WIDTH' in brick.colnames:
        cmd += f'--radec {brick["RA"]} {brick["DEC"]} --width={brick["WIDTH"]} --height={brick["WIDTH"]} '
    else:
        cmd += f'--brick={brick["BRICKNAME"]} '
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
            
    log.info(cmd)
    err = subprocess.call(cmd.split())#, stdout=log, stderr=log)
    
    if err != 0:
        msg = 'WARNING: Something went wrong; please check the logfile.'
        log.warning(msg)
        return 0
    else:
        return err
    
    
def candidate_cutouts(brick, coaddsdir, ssl_width=152, pixscale=PIXSCALE, bands=BANDS,
                      stagesuffix='candidate-cutouts', overwrite=False):
    """Generate cutouts of all the candidate large galaxies.

    * brick - table with brick information (brickname, ra, dec, etc.)
    * coaddsdir - location of the detection coadds
    
    """
    import fitsio
    import h5py
    from SGA.io import get_raslice
    from astropy.table import Table
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    from PIL import Image

    brickname = brick['BRICKNAME']
    if 'custom' in brickname:
        subdir = 'cus'
    else:
        subdir = brickname[:3]

    #outdir = os.path.join(coaddsdir, 'candidates')
    #if not os.path.isdir(outdir):
    #    os.makedirs(outdir)

    nband = len(bands)

    #...
    srcsfile = os.path.join(coaddsdir, 'metrics', subdir, f'sources-{brickname}.fits')
    srcs = Table(fitsio.read(srcsfile))
    log.info(f'Read {len(srcs)} objects from {srcsfile}')

    jpgfile = os.path.join(coaddsdir, 'coadd', subdir, brickname, f'legacysurvey-{brickname}-image.jpg')
    jpg = mpimg.imread(jpgfile)
    log.info(f'Read {jpgfile}')
    height, width, _ = jpg.shape
                
    # trim off-image (reference) sources 
    I = (srcs['ibx'] > 0) * (srcs['iby'] > 0) * (srcs['iby'] < height) * (srcs['ibx'] < width)
    log.info(f'Trimming to {np.sum(I)}/{len(srcs)} on-image sources.')
    srcs = srcs[I]
    #srcs = srcs[166:]
    nobj = len(srcs)

    # Measure the S/N in a small aperture.
    from photutils.aperture import CircularAperture, aperture_photometry

    rad_arcsec = 3.
    rad_pixels = rad_arcsec / pixscale
    minsnr = 10.

    snrphot = np.zeros((nobj, nband), 'f4')
    for iband, band in enumerate(bands):
        log.debug(f'Working on band {band}')
        coimgfile = os.path.join(coaddsdir, 'coadd', subdir, brickname, f'legacysurvey-{brickname}-image-{band}.fits.fz')
        coivfile = os.path.join(coaddsdir, 'coadd', subdir, brickname, f'legacysurvey-{brickname}-invvar-{band}.fits.fz')
        if not os.path.isfile(coimgfile):
            log.info(f'Missing {coimgfile}')
            continue
        coimg = fitsio.read(coimgfile)
        coiv = fitsio.read(coivfile)
        mask = (coiv == 0)
        coimg[mask] = 0.

        with np.errstate(divide='ignore'):
            imsigma = 1. / np.sqrt(coiv)
            imsigma[mask] = 0.
            
        apxy = np.vstack((srcs['ibx'], srcs['iby'])).T
        aper = CircularAperture(apxy, rad_pixels)
        phot = aperture_photometry(coimg, aper, error=imsigma, mask=mask)
        #phot.rename_columns(['aperture_sum', 'aperture_sum_err'], [f'flux_{band}', f'flux_err_{band}'])
        good = np.where(phot['aperture_sum_err'] > 0)[0]
        if len(good) > 0:
            snrphot[good, iband] = phot['aperture_sum'][good] / phot['aperture_sum_err'][good]

    snrphot = np.max(snrphot, axis=1) # max over any band

    log.info('HACK! - Trim sources!')
    if brickname != 'custom-015823m04663':
        C  = snrphot > minsnr
        srcs = srcs[C]
        snrphot = snrphot[C]

    nobj = len(srcs)
    
    # Read and select objects from the SGA-2020; this should happen once outside
    # of this function!
    sgadir = '/pscratch/sd/i/ioannis/SGA2024-data/SGA2020-cutouts'
    catfile = '/global/cfs/cdirs/cosmo/data/sga/2020/SGA-2020.fits'
    cols = ['SGA_ID', 'GALAXY', 'RA', 'DEC', 'G_MAG_SB25', 'R_MAG_SB25', 'Z_MAG_SB25']
    cat = Table(fitsio.read(catfile, 'ELLIPSE', columns=cols))

    I = (cat['G_MAG_SB25'] != -1.) * (cat['R_MAG_SB25'] != -1.) * (cat['Z_MAG_SB25'] != -1.)
    cat = cat[I]
    sgaras = cat['RA']
    sgadecs = cat['DEC']
    sgaids = cat['SGA_ID']
    rmag = cat['R_MAG_SB25'].data
    grmag = cat['G_MAG_SB25'].data - cat['R_MAG_SB25'].data

    # trim the worse outliers
    rmin, rmax = np.percentile(rmag, [1., 99.])
    grmin, grmax = np.percentile(grmag, [1., 99.])
    I = (rmag > rmin) * (rmag < rmax) * (grmag > grmin) * (grmag < grmax)
    sgaras = sgaras[I]
    sgadecs = sgadecs[I]
    sgaids = sgaids[I]
    rmag = rmag[I]
    grmag = grmag[I]

    # draw uniformly from g-r vs g
    # https://stackoverflow.com/questions/17821458/random-number-from-histogram

    # this code is totally bogus
    from scipy.stats import binned_statistic_2d
    nSGA = 5000
    nbins = 50
    count, xbins, ybins, indx = binned_statistic_2d(rmag, grmag, np.zeros_like(rmag), 'count', bins=(nbins, nbins), expand_binnumbers=True)
    indx -= 1 # https://stackoverflow.com/questions/60779093/scipy-stats-binned-statistic-dd-bin-numbering-has-lots-of-extra-bins

    weights = count / np.sum(count)

    seed = 1
    rand = np.random.default_rng(seed)

    I = []
    for iybin in range(nbins):
        for ixbin in range(nbins):
            if count[iybin, ixbin] > 0:
                Nbin = np.max((int(nSGA * weights[iybin, ixbin]), 1))
                if Nbin > 0:
                    J = np.where((indx[0, :] == iybin) * (indx[1, :] == ixbin))[0]
                    if len(J) != count[iybin, ixbin]:
                        raise ValueError
                    Ibin = rand.choice(J, Nbin, replace=False)
                    I.append(Ibin)
    I = np.hstack(I)
    assert(len(I) == len(np.unique(I)))

    sgaras = sgaras[I]
    sgadecs = sgadecs[I]
    sgaids = sgaids[I]
    nsga = len(sgaids)

    # build a simple QA plot
    if True:
        fig, ax = plt.subplots(figsize=(6, 6))#, sharey=True)
        ax.imshow(jpg)
        S = (srcs['ref_cat'] == 'L3') * (snrphot > minsnr)
        G = (srcs['ref_cat'] == 'GE') * (snrphot > minsnr)
        I = (srcs['ref_cat'] == '  ') * (snrphot > minsnr)
        if np.sum(S) > 0:
            ax.scatter(srcs['ibx'][S], height-srcs['iby'][S], marker='o', color='red')
        if np.sum(G) > 0:
            ax.scatter(srcs['ibx'][G], height-srcs['iby'][G], marker='s', color='green')
        if np.sum(I) > 0:
            ax.scatter(srcs['ibx'][I], height-srcs['iby'][I], marker='+', color='dodgerblue')

        ax.axis('off')
        fig.tight_layout()
        jpgfile = f'/global/cfs/cdirs/desi/users/ioannis/tmp/{os.path.basename(jpgfile)}'
        fig.savefig(jpgfile)
        log.info(f'Wrote {jpgfile}')

        # total hack below here!
        if brickname == 'custom-015823m04663':

            # pad the jpg image with zeros so we can get cutouts with impunity
            bigjpg = np.zeros((height + ssl_width, width + ssl_width, 3), jpg.dtype)
            bigjpg[ssl_width//2:height+ssl_width//2, ssl_width//2:width+ssl_width//2, :] = jpg
            bigheight, bigwidth, _ = bigjpg.shape
        
            nrows, ncols = 4, 3
    
            #nrows, ncols = 20, 10
            #bigimg = np.zeros((nrows * ssl_width, ncols * ssl_width, 3), jpg.dtype)
    
            iobj = 0
            I = np.array([5142, 5143, 5144, 5145, 5146, 5150, 5270, 5303, 5306, 5308]) - nsga
            #I = np.arange(nobj)
    
            snrsort = np.argsort(snrphot[I])[::-1]
            srcs_sorted = srcs[I][snrsort]

            # temporary hack - run 'find_galaxy' on the r-band image
            from SGA.ellipse import find_galaxy
            imgfile = os.path.join(coaddsdir, 'coadd', subdir, brickname, f'legacysurvey-{brickname}-image-r.fits.fz')
            img = np.zeros((height + ssl_width, width + ssl_width), 'f4')
            img[ssl_width//2:height+ssl_width//2, ssl_width//2:width+ssl_width//2] = fitsio.read(imgfile)            
            
            fig, ax = plt.subplots(nrows, ncols, figsize=(2.5*ncols, 2.5*nrows),
                                   sharex=True, sharey=True)
            
            for icol in range(ncols):
                for irow in range(nrows):
                    #print(irow, icol)
                    if iobj < len(I):
                        y1 = srcs_sorted['iby'][iobj] + ssl_width//2 + ssl_width//2
                        y2 = srcs_sorted['iby'][iobj] - ssl_width//2 + ssl_width//2
                        x1 = srcs_sorted['ibx'][iobj] - ssl_width//2 + ssl_width//2
                        x2 = srcs_sorted['ibx'][iobj] + ssl_width//2 + ssl_width//2
                        jpgcutout = bigjpg[bigheight-y1:bigheight-y2, x1:x2, :]

                        imgcutout = img[y2:y1, x1:x2]
                        mge = find_galaxy(imgcutout, plot=False)
                        jpgcutout = Image.fromarray(jpgcutout)
                        
                        
                        pdb.set_trace()
                        
                        ax[irow, icol].imshow(jpgcutout)
                        ax[irow, icol].axis('off')
                    else:
                        ax[irow, icol].axis('off')
    
                    #bigimg[irow*ssl_width:(irow+1)*ssl_width,icol*ssl_width:(icol+1)*ssl_width, :] = cutout
                        
                    iobj += 1
                    
            fig.subplots_adjust(wspace=0.05, hspace=0.05, left=0.05, right=0.95, bottom=0.05, top=0.95)
            jpgfile = f'/global/cfs/cdirs/desi/users/ioannis/tmp/{os.path.basename(jpgfile)}'.replace('-image.jpg', '-cutouts.jpg')
            fig.savefig(jpgfile)
            log.info(f'Wrote {jpgfile}')

    pdb.set_trace()
            
    # Build the hdf5 file needed by ssl-legacysurvey
    h5file = os.path.join(coaddsdir, f'{brickname}-candidate-cutouts.hdf5')
    if os.path.isfile(h5file) and not overwrite:
        log.warning(f'Skipping existing output file {h5file}.')
    try:
        F = h5py.File(h5file, 'w')
        images = F.create_dataset('images', (nsga + nobj, nband, ssl_width, ssl_width))
        F.create_dataset('ra', data=np.hstack((sgaras, srcs['ra'].data)))
        F.create_dataset('dec', data=np.hstack((sgadecs, srcs['dec'].data)))
        #F.create_dataset('ra', data=srcs['ra'].data)
        #F.create_dataset('dec', data=srcs['dec'].data)

        for isga, (sgaid, sgara, sgadec) in enumerate(zip(sgaids, sgaras, sgadecs)):
            coutoutfile = os.path.join(sgadir, get_raslice(sgara), f'SGA2020-{sgaid:06d}.fits')
            images[isga, :] = fitsio.read(coutoutfile)
    
        for iband, band in enumerate(bands):
            log.debug(f'Working on band {band}')
            imgfile = os.path.join(coaddsdir, 'coadd', subdir, brickname, f'legacysurvey-{brickname}-image-{band}.fits.fz')
            if not os.path.isfile(imgfile):
                log.info(f'Missing {imgfile}')
                continue

            # pad the image with zeros so we can get cutouts with impunity
            img = np.zeros((height + ssl_width, width + ssl_width), 'f4')
            img[ssl_width//2:height+ssl_width//2, ssl_width//2:width+ssl_width//2] = fitsio.read(imgfile)            
            
            for isrc, src in enumerate(srcs):
                # one-pixel offset?!?
                y1 = src['iby'] - ssl_width//2 + ssl_width//2
                y2 = src['iby'] + ssl_width//2 + ssl_width//2
                x1 = src['ibx'] - ssl_width//2 + ssl_width//2
                x2 = src['ibx'] + ssl_width//2 + ssl_width//2
                #print(isrc, y1, y2, x1, x2)
                cutout = img[y1:y2, x1:x2]
                assert(cutout.shape[0] == ssl_width and cutout.shape[1] == ssl_width)
                images[nsga + isrc, iband, :] = cutout
                
                #plt.clf() ; plt.imshow(np.log10(cutout), origin='lower') ; plt.savefig('/global/cfs/cdirs/desi/users/ioannis/tmp/junk2.png')
                #pdb.set_trace()
    except:
        msg = f'Problem generating HDF5 dataset {h5file}!'
        log.critical(msg)
        raise ValueError(msg)

    F.close()
    log.info(f'Wrote {nobj} cutout(s) to {h5file}')

    #with h5py.File(h5file, 'r') as F:
    #    print(list(F['ra']))

    return 0 # good


