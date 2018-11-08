"""
LSLGA.unwise
============

Code to generate unWISE custom coadds / mosaics.

"""
import os, pdb
import numpy as np

def _unwise_to_rgb(imgs, bands=[1,2], mn=-1, mx=100, arcsinh=1.0):
    """Support routine to generate color unWISE images.

    """
    img = imgs[0]
    H,W = img.shape

    ## FIXME
    w1,w2 = imgs

    rgb = np.zeros((H, W, 3), np.uint8)

    scale1 = 50.
    scale2 = 50.

    #mn,mx = -3.,30.
    #arcsinh = None

    img1 = w1 / scale1
    img2 = w2 / scale2

    print('W1 99th', np.percentile(img1, 99))
    print('W2 99th', np.percentile(img2, 99))

    if arcsinh is not None:
        def nlmap(x):
            return np.arcsinh(x * arcsinh) / np.sqrt(arcsinh)
        #img1 = nlmap(img1)
        #img2 = nlmap(img2)
        mean = (img1 + img2) / 2.
        I = nlmap(mean)
        img1 = img1 / mean * I
        img2 = img2 / mean * I
        mn = nlmap(mn)
        mx = nlmap(mx)
    img1 = (img1 - mn) / (mx - mn)
    img2 = (img2 - mn) / (mx - mn)

    rgb[:,:,2] = (np.clip(img1, 0., 1.) * 255).astype(np.uint8)
    rgb[:,:,0] = (np.clip(img2, 0., 1.) * 255).astype(np.uint8)
    rgb[:,:,1] = rgb[:,:,0]/2 + rgb[:,:,2]/2

    return rgb

def unwise_coadds(onegal, galaxy=None, radius=30, pixscale=2.75, 
                survey=None, unwise_dir=None, verbose=False):
    '''Generate Legacy Survey and WISE cutouts.
    
    radius in arcsec
    
    pixscale: WISE pixel scale in arcsec/pixel; make this smaller than 2.75
    to oversample.

    '''
    import fitsio
    import matplotlib.pyplot as plt
    
    from astrometry.util.util import Tan
    from astrometry.util.plotutils import dimshow
    from astrometry.util.fits import fits_table, merge_tables
    from astrometry.util.resample import resample_with_wcs, ResampleError
    from wise.forcedphot import unwise_tiles_touching_wcs
    from wise.unwise import get_unwise_tractor_image
    from tractor import Tractor, NanoMaggies, Image

    from legacypipe.survey import imsave_jpeg
    from legacypipe.catalog import read_fits_catalog
    
    if galaxy is None:
        galaxy = 'galaxy'

    if survey is None:
        from legacypipe.survey import LegacySurveyData
        survey = LegacySurveyData()

    if unwise_dir is None:
        unwise_dir = os.environ.get('UNWISE_COADDS_DIR')

    npix = np.ceil(2 * radius / pixscale).astype('int') # [pixels]
    W = H = npix
    pix = pixscale / 3600.0
    wcs = Tan(onegal['RA'], onegal['DEC'],
              (W+1) / 2.0, (H+1) / 2.0,
              -pix, 0.0, 0.0, pix,
              float(W), float(H))
    if verbose:
        print('Image size: {}'.format(npix))

    # Read the custom Tractor catalog
    tractorfile = os.path.join(survey.output_dir, '{}-tractor.fits'.format(galaxy))
    if not os.path.isfile(tractorfile):
        print('Missing Tractor catalog {}'.format(tractorfile))
        return 0
    T = fits_table(tractorfile)
    primhdr = fitsio.read_header(tractorfile)

    if verbose:
        print('Total of {} sources'.format(len(T)))
        
    # Find unWISE tiles overlapping
    tiles = unwise_tiles_touching_wcs(wcs)
    print('Cut to', len(tiles), 'unWISE tiles')

    # Assume the targetwcs is axis-aligned and that the edge midpoints yield the
    # RA, Dec limits (true for TAN).
    r, d = wcs.pixelxy2radec(np.array([1,   W,   W/2, W/2]),
                             np.array([H/2, H/2, 1,   H  ]))
    
    # Note: the way the roiradec box is used, the min/max order doesn't matter.
    roiradec = [r[0], r[1], d[2], d[3]]

    ra, dec = T.ra, T.dec
    srcs = read_fits_catalog(T)

    wbands = [1, 2, 3, 4]
    wanyband = 'w'

    for band in wbands:
        f = T.get('flux_w%i' % band)
        f *= 10.**(primhdr['WISEAB%i' % band] / 2.5)

    coimgs = [np.zeros((H,W), np.float32) for b in wbands]
    comods = [np.zeros((H,W), np.float32) for b in wbands]
    con    = [np.zeros((H,W), np.uint8) for b in wbands]

    for iband, band in enumerate(wbands):
        print('Photometering WISE band', band)
        wband = 'w%i' % band

        for i, src in enumerate(srcs):
            #print('Source', src, 'brightness', src.getBrightness(), 'params', src.getBrightness().getParams())
            #src.getBrightness().setParams([T.wise_flux[i, band-1]])
            src.setBrightness(NanoMaggies(**{wanyband: T.get('flux_w%i'%band)[i]}))
            # print('Set source brightness:', src.getBrightness())

        # The tiles have some overlap, so for each source, keep the
        # fit in the tile whose center is closest to the source.
        for tile in tiles:
            print('Reading tile', tile.coadd_id)
            tim = get_unwise_tractor_image(unwise_dir, tile.coadd_id, band,
                                           bandname=wanyband, roiradecbox=roiradec)
            if tim is None:
                print('Actually, no overlap with tile', tile.coadd_id)
                continue
            print('Read image with shape', tim.shape)

            # Select sources in play.
            wisewcs = tim.wcs.wcs
            H, W = tim.shape
            ok, x, y = wisewcs.radec2pixelxy(ra, dec)
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
                Yo, Xo, Yi, Xi, nil = resample_with_wcs(wcs, tim.wcs.wcs)
            except ResampleError:
                continue
            if len(Yo) == 0:
                continue
            print('Resampling', len(Yo), 'pixels from WISE', tile.coadd_id,
                  band)

            coimgs[iband][Yo,Xo] += tim.getImage()[Yi, Xi]
            comods[iband][Yo,Xo] += mod[Yi, Xi]
            con   [iband][Yo,Xo] += 1

    for img, mod, n in zip(coimgs, comods, con):
        img /= np.maximum(n, 1)
        mod /= np.maximum(n, 1)

    for band, img, mod in zip(wbands, coimgs, comods):
        fitsfile = os.path.join(survey.output_dir, '{}-W{}-image.fits'.format(galaxy, band))
        if verbose:
            print('Writing {}'.format(fitsfile))
        fitsio.write(fitsfile, img, clobber=True)
        
    # Color WISE images --
    kwa = dict(mn=-1, mx=100, arcsinh=1)
    #kwa = dict(mn=-0.1, mx=2., arcsinh=1)
    #kwa = dict(mn=-0.1, mx=2., arcsinh=None)

    rgb = _unwise_to_rgb(coimgs[:2], **kwa)
    pngfile = os.path.join(survey.output_dir, '{}-unwise-image.png'.format(galaxy))
    if verbose:
        print('Writing {}'.format(pngfile))
    imsave_jpeg(pngfile, rgb, origin='lower')
    
    fig, ax = plt.subplots(figsize=(8, 8))
    dimshow(rgb, ticks=False)
    fig.savefig(pngfile, bbox_inches='tight', pad_inches=0)

    rgb = _unwise_to_rgb(comods[:2], **kwa)
    pngfile = os.path.join(survey.output_dir, '{}-unwise-model.png'.format(galaxy))
    if verbose:
        print('Writing {}'.format(pngfile))
    imsave_jpeg(pngfile, rgb, origin='lower')

    #kwa = dict(mn=-1, mx=1, arcsinh=1)
    #kwa = dict(mn=-1, mx=1, arcsinh=None)
    rgb = _unwise_to_rgb([img-mod for img, mod in list(zip(coimgs, comods))[:2]], **kwa)
    pngfile = os.path.join(survey.output_dir, '{}-unwise-resid.png'.format(galaxy))
    if verbose:
        print('Writing {}'.format(pngfile))
    imsave_jpeg(pngfile, rgb, origin='lower')

    return 1
