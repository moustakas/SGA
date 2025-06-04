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
