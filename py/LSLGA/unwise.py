"""
LSLGA.unwise
============

Code to generate unWISE custom coadds / mosaics.

"""

def unwise_coadds(onegal, galaxy=None, radius=30, pixscale=2.75,
                survey=None, unwise_dir=None, verbose=False):
    '''Generate Legacy Survey and WISE cutouts.
    
    radius in arcsec.
    
    pixscale: WISE pixel scale in arcsec/pixel; make this smaller than 2.75
    to oversample.

    '''
    from astrometry.util.util import Tan
    from legacypipe.survey import bricks_touching_wcs

    
    if galaxy is None:
        galaxy = 'galaxy'

    if survey is None:
        from legacypipe.survey import LegacySurveyData
        survey = LegacySurveyData()

    if unwise_dir is None:
        unwise_dir = os.environ.get('UNWISE_COADDS_DIR')
        
    ra, dec = onegal['RA'], onegal['DEC']
    
    npix = int(np.ceil(radius / pixscale))
    W = H = npix
    pix = pixscale / 3600.0
    wcs = Tan(ra, dec, (W+1) / 2.0, (H+1) / 2., -pix,
              0.0, 0.0, pix, float(W), float(H))

    # Find DECaLS bricks overlapping
    B = bricks_touching_wcs(wcs, survey=survey)

    if verbose:
        print('Image size: {}'.format(npix))
        print('Found {} bricks overlapping'.format(len(B))

    TT = []
    for b in B.brickname:
        fn = survey.find_file('tractor', brick=b)
        T = fits_table(fn)
        print('Read', len(T), 'from', b)
        primhdr = fitsio.read_header(fn)
        TT.append(T)
    T = merge_tables(TT)
    print('Total of', len(T), 'sources')
    T.cut(T.brick_primary)
    print(len(T), 'primary')
    margin = 20
    ok,xx,yy = wcs.radec2pixelxy(T.ra, T.dec)
    I = np.flatnonzero((xx > -margin) * (yy > -margin) *
                       (xx < W+margin) * (yy < H+margin))
    T.cut(I)
    print(len(T), 'within ROI')

    # Get the central source.
    #from astrometry.libkd.spherematch import match_radec
    #m1, m2, d12 = match_radec(ra, dec, T.ra, T.dec, 1/3600)
    #print('Half-light radius', T[m2].shapedev_r/60.)

    #return wcs,T

    # Pull out DECaLS coadds (image, model, resid).
    dwcs = wcs.scale(2. * pixscale / 0.262)
    dh,dw = dwcs.shape
    print('DECaLS resampled shape:', dh,dw)
    tags = ['image', 'model', 'resid']
    coimgs = [np.zeros((dh,dw,3), np.uint8) for t in tags]

    fitscoimgs = [np.zeros((dh,dw,3), np.float32) for t in tags]

    for b in B.brickname:
        fn = survey.find_file('image', brick=b, band='r')
        bwcs = Tan(fn, 1) # ext 1: .fz
        try:
            Yo,Xo,Yi,Xi,nil = resample_with_wcs(dwcs, bwcs)
        except ResampleError:
            continue
        if len(Yo) == 0:
            continue
        print('Resampling', len(Yo), 'pixels from', b)
        xl,xh,yl,yh = Xi.min(), Xi.max(), Yi.min(), Yi.max()
        #print('python legacypipe/runbrick.py -b %s --zoom %i %i %i %i --outdir cluster --pixpsf --splinesky --pipe --no-early-coadds' % 
        #      (b, xl-5, xh+5, yl-5, yh+5) + ' -P \'pickles/cluster-%(brick)s-%%(stage)s.pickle\'')
        for i,tag in enumerate(tags):
            fn = survey.find_file(tag+'-jpeg', brick=b)
            img = plt.imread(fn)
            img = np.flipud(img)
            coimgs[i][Yo,Xo,:] = img[Yi,Xi,:]

        # Read FITS images too
        for i,tag in enumerate(tags):
            if tag == 'resid':
                continue
            for iband,band in enumerate('grz'):
                fn = survey.find_file(tag, brick=b, band=band)
                fitsimg = fitsio.read(fn)
                fitscoimgs[i][Yo,Xo,iband] = fitsimg[Yi,Xi]


    #tt = dict(image='Image', model='Model', resid='Resid')
    for img, tag in zip(coimgs, tags):
        #outfn = '{}-grz-{}'.format(prefix, tag)
        #fitsio.write('{}.fits'.format(outfn), img, clobber=True)

        fig, ax = plt.subplots(figsize=(8, 8))
        dimshow(img, ticks=False)
        fig.savefig('{}-grz-{}.png'.format(prefix, tag), bbox_inches='tight', pad_inches=0)

    for img, tag in zip(fitscoimgs, tags):
        # Write FITS coadds
        if tag == 'resid':
            continue
        for iband,band in enumerate('grz'):
            fitsio.write('{}-{}-{}.fits'.format(prefix, band, tag), img[:,:,iband], clobber=True)

    # Find unWISE tiles overlapping
    tiles = unwise_tiles_touching_wcs(wcs)
    print('Cut to', len(tiles), 'unWISE tiles')

    # Here we assume the targetwcs is axis-aligned and that the
    # edge midpoints yield the RA,Dec limits (true for TAN).
    r,d = wcs.pixelxy2radec(np.array([1,   W,   W/2, W/2]),
                            np.array([H/2, H/2, 1,   H  ]))
    # the way the roiradec box is used, the min/max order doesn't matter
    roiradec = [r[0], r[1], d[2], d[3]]

    ra,dec = T.ra, T.dec

    srcs = read_fits_catalog(T)

    wbands = [1,2,3,4]
    wanyband = 'w'

    for band in wbands:
        f = T.get('flux_w%i' % band)
        f *= 10.**(primhdr['WISEAB%i' % band] / 2.5)

    coimgs = [np.zeros((H,W), np.float32) for b in wbands]
    comods = [np.zeros((H,W), np.float32) for b in wbands]
    con    = [np.zeros((H,W), np.uint8) for b in wbands]

    for iband,band in enumerate(wbands):
        print('Photometering WISE band', band)
        wband = 'w%i' % band

        for i,src in enumerate(srcs):
            #print('Source', src, 'brightness', src.getBrightness(), 'params', src.getBrightness().getParams())
            #src.getBrightness().setParams([T.wise_flux[i, band-1]])
            src.setBrightness(NanoMaggies(**{wanyband: T.get('flux_w%i'%band)[i]}))
            # print('Set source brightness:', src.getBrightness())

        # The tiles have some overlap, so for each source, keep the
        # fit in the tile whose center is closest to the source.
        for tile in tiles:
            print('Reading tile', tile.coadd_id)

            tim = get_unwise_tractor_image(unwise_dir, tile.coadd_id, band,
                                           bandname=wanyband,
                                           roiradecbox=roiradec)
            if tim is None:
                print('Actually, no overlap with tile', tile.coadd_id)
                continue
            print('Read image with shape', tim.shape)

            # Select sources in play.
            wisewcs = tim.wcs.wcs
            H,W = tim.shape
            ok,x,y = wisewcs.radec2pixelxy(ra, dec)
            x = (x - 1.).astype(np.float32)
            y = (y - 1.).astype(np.float32)
            margin = 10.
            I = np.flatnonzero((x >= -margin) * (x < W+margin) *
                               (y >= -margin) * (y < H+margin))
            print(len(I), 'within the image + margin')

            subcat = [srcs[i] for i in I]
            tractor = Tractor([tim], subcat)
            mod = tractor.getModelImage(0)

            # plt.clf()
            # dimshow(tim.getImage(), ticks=False)
            # plt.title('WISE %s %s' % (tile.coadd_id, wband))
            # ps.savefig()

            # plt.clf()
            # dimshow(mod, ticks=False)
            # plt.title('WISE %s %s' % (tile.coadd_id, wband))
            # ps.savefig()

            try:
                Yo,Xo,Yi,Xi,nil = resample_with_wcs(wcs, tim.wcs.wcs)
            except ResampleError:
                continue
            if len(Yo) == 0:
                continue
            print('Resampling', len(Yo), 'pixels from WISE', tile.coadd_id,
                  band)

            coimgs[iband][Yo,Xo] += tim.getImage()[Yi,Xi]
            comods[iband][Yo,Xo] += mod[Yi,Xi]
            con   [iband][Yo,Xo] += 1

    for img,mod,n in zip(coimgs, comods, con):
        img /= np.maximum(n, 1)
        mod /= np.maximum(n, 1)

    for band,img,mod in zip(wbands, coimgs, comods):
        outfn = '{}-W{}'.format(prefix, band)
        fitsio.write('{}-{}.fits'.format(outfn, 'image'), img, clobber=True)
        
        lo, hi = np.percentile(img, [25,99])
        resid = img - mod
        mx = np.abs(resid).max()

        if False:
            fig, ax = plt.subplots(figsize=(8, 8))
            dimshow(img, vmin=lo, vmax=hi, ticks=False)
            fig.savefig('{}-{}.png'.format(outfn, 'image'), bbox_inches='tight', pad_inches=0)

            fig, ax = plt.subplots(figsize=(8, 8))
            dimshow(mod, vmin=lo, vmax=hi, ticks=False)
            fig.savefig('{}-{}.png'.format(outfn, 'model'), bbox_inches='tight', pad_inches=0)

            fig, ax = plt.subplots(figsize=(8, 8))
            dimshow(resid, vmin=-mx, vmax=mx, ticks=False)
            fig.savefig('{}-{}.png'.format(outfn, 'resid'), bbox_inches='tight', pad_inches=0)

    # Color WISE images --

    kwa = dict(mn=-0.1, mx=2., arcsinh = 1.)
    #kwa = dict(mn=-0.1, mx=2., arcsinh=None)
    rgb = _unwise_to_rgb(coimgs[:2], **kwa)
    fig, ax = plt.subplots(figsize=(8, 8))
    dimshow(rgb, ticks=False)
    fig.savefig('{}-W1W2-image.png'.format(prefix), bbox_inches='tight', pad_inches=0)


    rgb = _unwise_to_rgb(comods[:2], **kwa)
    fig, ax = plt.subplots(figsize=(8, 8))
    dimshow(rgb, ticks=False)
    fig.savefig('{}-W1W2-model.png'.format(prefix), bbox_inches='tight', pad_inches=0)

    kwa = dict(mn=-1, mx=1, arcsinh=1)
    #kwa = dict(mn=-1, mx=1, arcsinh=None)
    rgb = _unwise_to_rgb([img-mod for img,mod in list(zip(coimgs,comods))[:2]], **kwa)
    fig, ax = plt.subplots(figsize=(8, 8))
    dimshow(rgb, ticks=False)
    fig.savefig('{}-W1W2-resid.png'.format(prefix), bbox_inches='tight', pad_inches=0)

    return wcs, T
