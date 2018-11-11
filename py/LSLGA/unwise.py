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
    H, W = img.shape

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
    rgb[:,:,1] = rgb[:, :, 0] / 2 + rgb[:, :, 2] / 2

    return rgb

def _unwise_images_models(T, srcs, targetwcs, unwise_tims, margin=10):
    """Create the unWISE images and models given a Tractor and sources catalog.

    """
    from astrometry.util.resample import resample_with_wcs, ResampleError
    from tractor import Tractor, Image

    wbands = [1, 2, 3, 4]
    H, W = targetwcs.shape

    coimgs = [np.zeros((H, W), np.float32) for b in wbands]
    comods = [np.zeros((H, W), np.float32) for b in wbands]
    con    = [np.zeros((H, W), np.uint8) for b in wbands]

    for iband, band in enumerate(wbands):
        tims = unwise_tims['w{}'.format(band)]

        # The tiles have some overlap, so for each source, keep the
        # fit in the tile whose center is closest to the source.
        for tim in tims:
            # Select sources in play.
            wisewcs = tim.wcs.wcs
            timH, timW = tim.shape
            ok, x, y = wisewcs.radec2pixelxy(T.ra, T.dec)
            x = (x - 1.).astype(np.float32)
            y = (y - 1.).astype(np.float32)
            I = np.flatnonzero((x >= -margin) * (x < timW + margin) *
                               (y >= -margin) * (y < timH + margin))
            #print('Found {} sources within the image + margin = {} pixels'.format(len(I), margin))

            subcat = [srcs[i] for i in I]
            tractor = Tractor([tim], subcat)
            mod = tractor.getModelImage(0)

            try:
                Yo, Xo, Yi, Xi, nil = resample_with_wcs(targetwcs, tim.wcs.wcs)
            except ResampleError:
                continue
            if len(Yo) == 0:
                continue

            coimgs[iband][Yo, Xo] += tim.getImage()[Yi, Xi]
            comods[iband][Yo, Xo] += mod[Yi, Xi]
            con   [iband][Yo, Xo] += 1

    for img, mod, n in zip(coimgs, comods, con):
        img /= np.maximum(n, 1)
        mod /= np.maximum(n, 1)

    coresids = [img-mod for img, mod in list(zip(coimgs, comods))]

    return coimgs, comods, coresids

def unwise_coadds(onegal, galaxy=None, radius=30, pixscale=2.75, 
                  output_dir=None, unwise_dir=None, verbose=False,
                  log=None):
    '''Generate custom unWISE cutouts.
    
    radius in arcsec
    
    pixscale: WISE pixel scale in arcsec/pixel; make this smaller than 2.75
    to oversample.

    '''
    import fitsio
    import matplotlib.pyplot as plt
    
    from astrometry.util.util import Tan
    from astrometry.util.fits import fits_table
    from astrometry.libkd.spherematch import match_radec
    from wise.forcedphot import unwise_tiles_touching_wcs
    from wise.unwise import get_unwise_tractor_image
    from tractor import NanoMaggies

    from legacypipe.survey import imsave_jpeg
    from legacypipe.catalog import read_fits_catalog
    
    if galaxy is None:
        galaxy = 'galaxy'

    if output_dir is None:
        output_dir = '.'

    if unwise_dir is None:
        unwise_dir = os.environ.get('UNWISE_COADDS_DIR')

    # Initialize the WCS object.
    W = H = np.ceil(2 * radius / pixscale).astype('int') # [pixels]
    targetwcs = Tan(onegal['RA'], onegal['DEC'], (W + 1) / 2.0, (H + 1) / 2.0,
                    -pixscale / 3600.0, 0.0, 0.0, pixscale / 3600.0, float(W), float(H))

    # Read the custom Tractor catalog.
    tractorfile = os.path.join(output_dir, '{}-tractor.fits'.format(galaxy))
    if not os.path.isfile(tractorfile):
        print('Missing Tractor catalog {}'.format(tractorfile), flush=True, file=log)
        return 0
    primhdr = fitsio.read_header(tractorfile)

    T = fits_table(tractorfile)
    srcs = read_fits_catalog(T)

    print('Read {} sources from {}'.format(len(T), tractorfile), flush=True, file=log)

    # Find and read the overlapping unWISE tiles.  Assume the targetwcs is
    # axis-aligned and that the edge midpoints yield the RA, Dec limits (true
    # for TAN).  Note: the way the roiradec box is used, the min/max order
    # doesn't matter.
    r, d = targetwcs.pixelxy2radec(np.array([1,   W,   W/2, W/2]),
                                   np.array([H/2, H/2, 1,   H  ]))
    roiradec = [r[0], r[1], d[2], d[3]]

    tiles = unwise_tiles_touching_wcs(targetwcs)

    wbands = [1, 2, 3, 4]
    unwise_tims = {'w1': [], 'w2': [], 'w3': [], 'w4': []}
    for band in wbands:
        for tile in tiles:
            #print('Reading tile {}'.format(tile.coadd_id))
            tim = get_unwise_tractor_image(unwise_dir, tile.coadd_id, band,
                                           bandname='w', roiradecbox=roiradec)
            if tim is None:
                print('Actually, no overlap with tile {}'.format(tile.coadd_id))
                continue
            print('Read image {} with shape {}'.format(tile.coadd_id, tim.shape))
            unwise_tims['w{}'.format(band)].append(tim)

        for ii, src in enumerate(srcs):
            #print('Source', src, 'brightness', src.getBrightness(), 'params', src.getBrightness().getParams())
            #src.getBrightness().setParams([T.wise_flux[i, band-1]])
            src.setBrightness( NanoMaggies(**{'w': T.get('flux_w{}'.format(band) )[ii]}) )
            # print('Set source brightness:', src.getBrightness())

    #for band in wbands:
    #    f = T.get('flux_w{}'.format(band))
    #    f *= 10**(0.4 * primhdr['WISEAB{}'.format(band)])

    # Find and remove all the objects within XX arcsec of the target
    # coordinates.
    m1, m2, d12 = match_radec(T.ra, T.dec, onegal['RA'], onegal['DEC'], 3/3600.0, nearest=False)
    if len(d12) == 0:
        print('No matching galaxies found -- definitely a problem.')
        raise ValueError
    keep = ~np.isin(T.objid, T[m1].objid)
        
    srcs_nocentral = np.array(srcs)[keep].tolist()
    T_nocentral = T[keep]

    # Build the data and model images with and without the central.
    coimgs, comods, coresids = _unwise_images_models(T, srcs, targetwcs, unwise_tims)
    coimgs_nocentral, comods_nocentral, coresids_nocentral = _unwise_images_models(
        T_nocentral, srcs_nocentral, targetwcs, unwise_tims)
    del unwise_tims

    # Write out the final images with and without the central.
    for coadd, imtype in zip( (coimgs, comods, coresids), ('image', 'model', 'resid') ):
        for img, band in zip(coadd, wbands):
            fitsfile = os.path.join(output_dir, '{}-{}-W{}.fits'.format(galaxy, imtype, band))
            if verbose:
                print('Writing {}'.format(fitsfile))
            fitsio.write(fitsfile, img, clobber=True)

    pdb.set_trace()

    # Color WISE images --
    kwa = dict(mn=-1, mx=100, arcsinh=1)
    #kwa = dict(mn=-0.1, mx=2., arcsinh=1)
    #kwa = dict(mn=-0.1, mx=2., arcsinh=None)

    rgb = _unwise_to_rgb(coimgs[:2], **kwa)
    jpgfile = os.path.join(output_dir, '{}-unwise-image.jpg'.format(galaxy))
    if verbose:
        print('Writing {}'.format(jpgfile))
    imsave_jpeg(jpgfile, rgb, origin='lower')
    
    rgb = _unwise_to_rgb(comods[:2], **kwa)
    jpgfile = os.path.join(output_dir, '{}-unwise-model.jpg'.format(galaxy))
    if verbose:
        print('Writing {}'.format(jpgfile))
    imsave_jpeg(jpgfile, rgb, origin='lower')

    #kwa = dict(mn=-1, mx=1, arcsinh=1)
    #kwa = dict(mn=-1, mx=1, arcsinh=None)
    rgb = _unwise_to_rgb( [img-mod for img, mod in list(zip(coimgs, comods))[:2]], **kwa)
    jpgfile = os.path.join(output_dir, '{}-unwise-resid.jpg'.format(galaxy))
    if verbose:
        print('Writing {}'.format(jpgfile))
    imsave_jpeg(jpgfile, rgb, origin='lower')

    return 1
