"""
LSLGA.galex
===========

Code to generate GALEX custom coadds / mosaics.

"""
import os, pdb
import numpy as np

def _ra_ranges_overlap(ralo, rahi, ra1, ra2):
    import numpy as np
    x1 = np.cos(np.deg2rad(ralo))
    y1 = np.sin(np.deg2rad(ralo))
    x2 = np.cos(np.deg2rad(rahi))
    y2 = np.sin(np.deg2rad(rahi))
    x3 = np.cos(np.deg2rad(ra1))
    y3 = np.sin(np.deg2rad(ra1))
    x4 = np.cos(np.deg2rad(ra2))
    y4 = np.sin(np.deg2rad(ra2))
    cw32 = x2*y3 - x3*y2
    cw41 = x1*y4 - x4*y1
    return np.logical_and(cw32 <= 0, cw41 >= 0)

def _galex_rgb_dstn(imgs, **kwargs):
    nuv,fuv = imgs
    h,w = nuv.shape
    myrgb = np.zeros((h,w,3), np.float32)
    lo,hi = -0.0005, 0.01
    myrgb[:,:,0] = myrgb[:,:,1] = np.clip((nuv - lo) / (hi - lo), 0., 1.)
    lo,hi = -0.00015, 0.003
    myrgb[:,:,2] = np.clip((fuv - lo) / (hi - lo), 0., 1.)
    myrgb[:,:,1] = np.clip((myrgb[:,:,0] + myrgb[:,:,2]*0.2), 0., 1.)
    return myrgb

def _galex_rgb_official(imgs, **kwargs):
    from scipy.ndimage.filters import uniform_filter, gaussian_filter
    nuv,fuv = imgs
    h,w = nuv.shape
    red = nuv * 0.206 * 2297
    blue = fuv * 1.4 * 1525
    #blue = uniform_filter(blue, 3)
    blue = gaussian_filter(blue, 1.)
    green = (0.2*blue + 0.8*red)

    red   *= 0.085
    green *= 0.095
    blue  *= 0.08
    nonlinearity = 2.5
    radius = red + green + blue
    val = np.arcsinh(radius * nonlinearity) / nonlinearity
    with np.errstate(divide='ignore', invalid='ignore'):
        red   = red   * val / radius
        green = green * val / radius
        blue  = blue  * val / radius
    mx = np.maximum(red, np.maximum(green, blue))
    mx = np.maximum(1., mx)
    red   /= mx
    green /= mx
    blue  /= mx
    rgb = np.clip(np.dstack((red, green, blue)), 0., 1.)
    return rgb

def _galex_rgb_moustakas(imgs, **kwargs):
    #from scipy.ndimage.filters import uniform_filter, gaussian_filter
    nuv,fuv = imgs
    h,w = nuv.shape
    red = nuv * 0.206 * 2297
    blue = fuv * 1.4 * 1525
    #blue = uniform_filter(blue, 3)
    #blue = gaussian_filter(blue, 1.)
    green = (0.2*blue + 0.8*red)

    red   *= 0.085
    green *= 0.095
    blue  *= 0.08
    nonlinearity = 0.5 #1.0 # 2.5
    radius = red + green + blue
    val = np.arcsinh(radius * nonlinearity) / nonlinearity
    with np.errstate(divide='ignore', invalid='ignore'):
        red   = red   * val / radius
        green = green * val / radius
        blue  = blue  * val / radius
    mx = np.maximum(red, np.maximum(green, blue))
    mx = np.maximum(1., mx)

    lo = -0.1
    red   = (red - lo) / (mx - lo)
    green = (green - lo) / (mx - lo)
    blue  = (blue - lo) / (mx - lo)
    #red   /= mx
    #green /= mx
    #blue  /= mx
    
    rgb = np.clip(np.dstack((red, green, blue)), 0., 1.)
    return rgb

def galex_coadds(onegal, galaxy=None, radius=30, pixscale=2.75, 
                 output_dir=None, galex_dir=None, verbose=False):
    '''Generate custom GALEX cutouts.
    
    radius in arcsec
    
    pixscale: WISE pixel scale in arcsec/pixel; make this smaller than 2.75
    to oversample.

    '''
    import fitsio
    import matplotlib.pyplot as plt

    from astrometry.util.util import Tan
    from astrometry.util.fits import fits_table
    from astrometry.util.resample import resample_with_wcs, OverlapError
    from tractor import (Tractor, NanoMaggies, Image, LinearPhotoCal,
                         NCircularGaussianPSF, ConstantFitsWcs, ConstantSky)

    from legacypipe.survey import imsave_jpeg
    from legacypipe.catalog import read_fits_catalog

    if galaxy is None:
        galaxy = 'galaxy'

    if galex_dir is None:
        galex_dir = os.environ.get('GALEX_DIR')

    if output_dir is None:
        output_dir = '.'

    W = H = np.ceil(2 * radius / pixscale).astype('int') # [pixels]
    pix = pixscale / 3600.0
    wcs = Tan(onegal['RA'], onegal['DEC'],
              (W+1) / 2.0, (H+1) / 2.0,
              -pix, 0.0, 0.0, pix,
              float(W), float(H))

    # Read the custom Tractor catalog
    tractorfile = os.path.join(output_dir, '{}-tractor.fits'.format(galaxy))
    if not os.path.isfile(tractorfile):
        print('Missing Tractor catalog {}'.format(tractorfile))
        return 0
    T = fits_table(tractorfile)

    ralo,declo = wcs.pixelxy2radec(W,1)
    rahi,dechi = wcs.pixelxy2radec(1,H)
    print('RA',  ralo,rahi)
    print('Dec', declo,dechi)

    fn = os.path.join(galex_dir, 'galex-images.fits')
    print('Reading', fn)
    # galex "bricks" (actually just GALEX tiles)
    galex = fits_table(fn)

    galex.rename('ra_cent', 'ra')
    galex.rename('dec_cent', 'dec')
    galex.rename('have_n', 'has_n')
    galex.rename('have_f', 'has_f')
    cosd = np.cos(np.deg2rad(galex.dec))
    galex.ra1 = galex.ra - 3840*1.5/3600./2./cosd
    galex.ra2 = galex.ra + 3840*1.5/3600./2./cosd
    galex.dec1 = galex.dec - 3840*1.5/3600./2.
    galex.dec2 = galex.dec + 3840*1.5/3600./2.
    bricknames = []
    for tile,subvis in zip(galex.tilename, galex.subvis):
        if subvis == -999:
            bricknames.append(tile.strip())
        else:
            bricknames.append('%s_sg%02i' % (tile.strip(), subvis))
    galex.brickname = np.array(bricknames)

    # bricks_touching_radec_box(self, ralo, rahi, declo, dechi, scale=None):
    I, = np.nonzero((galex.dec1 <= dechi) * (galex.dec2 >= declo))
    ok = _ra_ranges_overlap(ralo, rahi, galex.ra1[I], galex.ra2[I])
    I = I[ok]
    galex.cut(I)
    print('-> bricks', galex.brickname)

    gbands = ['n','f']
    nicegbands = ['NUV', 'FUV']
    coimgs = []
    comods = []
    coresids = []

    srcs = read_fits_catalog(T)
    for src in srcs:
        src.freezeAllBut('brightness')

    for niceband, band in zip(nicegbands, gbands):
        J = np.flatnonzero(galex.get('has_'+band))
        print(len(J), 'GALEX tiles have coverage in band', band)

        coimg = np.zeros((H,W), np.float32)
        comod = np.zeros((H,W), np.float32)
        cowt  = np.zeros((H,W), np.float32)

        for src in srcs:
            src.setBrightness(NanoMaggies(**{band: 1}))

        for j in J:
            brick = galex[j]
            fn = os.path.join(galex_dir, brick.tilename.strip(),
                              '%s-%sd-intbgsub.fits.gz' % (brick.brickname, band))
            print(fn)

            gwcs = Tan(*[float(f) for f in
                         [brick.crval1, brick.crval2, brick.crpix1, brick.crpix2,
                          brick.cdelt1, 0., 0., brick.cdelt2, 3840., 3840.]])
            img = fitsio.read(fn)
            print('Read', img.shape)

            try:
                Yo,Xo,Yi,Xi,nil = resample_with_wcs(wcs, gwcs, [], 3)
            except OverlapError:
                continue

            K = np.flatnonzero(img[Yi,Xi] != 0.)
            if len(K) == 0:
                continue
            Yo = Yo[K]
            Xo = Xo[K]
            Yi = Yi[K]
            Xi = Xi[K]

            #rimg = np.zeros((H,W), np.float32)
            #rimg[Yo,Xo] = img[Yi,Xi]
            #plt.clf()
            #plt.imshow(rimg, interpolation='nearest', origin='lower')
            #ps.savefig()

            wt = brick.get(band + 'exptime')
            coimg[Yo,Xo] += wt * img[Yi,Xi]
            cowt [Yo,Xo] += wt

            x0 = min(Xi)
            x1 = max(Xi)
            y0 = min(Yi)
            y1 = max(Yi)
            subwcs = gwcs.get_subimage(x0, y0, x1-x0+1, y1-y0+1)
            twcs = ConstantFitsWcs(subwcs)
            timg = img[y0:y1+1, x0:x1+1]
            tie = np.ones_like(timg)  ## HACK!
            #hdr = fitsio.read_header(fn)
            #zp = hdr['
            zps = dict(n=20.08, f=18.82)
            zp = zps[band]
            photocal = LinearPhotoCal(NanoMaggies.zeropointToScale(zp),
                                      band=band)
            tsky = ConstantSky(0.)

            # HACK -- circular Gaussian PSF of fixed size...
            # in arcsec
            #fwhms = dict(NUV=6.0, FUV=6.0)
            # -> sigma in pixels
            #sig = fwhms[band] / 2.35 / twcs.pixel_scale()
            sig = 6.0 / 2.35 / twcs.pixel_scale()
            tpsf = NCircularGaussianPSF([sig], [1.])

            tim = Image(data=timg, inverr=tie, psf=tpsf, wcs=twcs, sky=tsky,
                        photocal=photocal, name='GALEX ' + band + brick.brickname)
            tractor = Tractor([tim], srcs)
            mod = tractor.getModelImage(0)

            #print('Tractor image', tim.name)
            #fig, ax = plt.subplots(figsize=(8, 8))
            #dimshow(timg, ticks=False)
            ##plt.imshow(timg, interpolation='nearest', origin='lower')
            #fig.savefig('{}-{}-image.jpg'.format(prefix, niceband))

            #print('Tractor model', tim.name)
            #fig, ax = plt.subplots(figsize=(8, 8))
            ##plt.imshow(mod, interpolation='nearest', origin='lower')
            #dimshow(mod, ticks=False)
            #fig.savefig('{}-{}-model.jpg'.format(prefix, niceband))

            tractor.freezeParam('images')

            #print('Params:')
            #tractor.printThawedParams()

            tractor.optimize_forced_photometry(priors=False, shared_params=False)

            mod = tractor.getModelImage(0)

            #print('Tractor model (forced phot)', tim.name)
            #plt.clf()
            #plt.imshow(mod, interpolation='nearest', origin='lower')
            #ps.savefig()

            comod[Yo,Xo] += wt * mod[Yi-y0,Xi-x0]

        coimg /= np.maximum(cowt, 1e-18)
        comod /= np.maximum(cowt, 1e-18)
        coresid = coimg - comod
        coimgs.append(coimg)
        comods.append(comod)
        coresids.append(coresid)

        fitsfile = os.path.join(output_dir, '{}-image-{}.fits'.format(galaxy, niceband))
        if verbose:
            print('Writing {}'.format(fitsfile))
        fitsio.write(fitsfile, coimg, clobber=True)

        fitsfile = os.path.join(output_dir, '{}-model-{}.fits'.format(galaxy, niceband))
        if verbose:
            print('Writing {}'.format(fitsfile))
        fitsio.write(fitsfile, comod, clobber=True)

        fitsfile = os.path.join(output_dir, '{}-resid-{}.fits'.format(galaxy, niceband))
        if verbose:
            print('Writing {}'.format(fitsfile))
        fitsio.write(fitsfile, coresid, clobber=True)

    _galex_rgb = _galex_rgb_moustakas
    #_galex_rgb = _galex_rgb_dstn
    #_galex_rgb = _galex_rgb_official

    jpgfile = os.path.join(output_dir, '{}-galex-image.jpg'.format(galaxy))
    if verbose:
        print('Writing {}'.format(jpgfile))
    rgb = _galex_rgb(coimgs)
    imsave_jpeg(jpgfile, rgb, origin='lower')

    jpgfile = os.path.join(output_dir, '{}-galex-model.jpg'.format(galaxy))
    if verbose:
        print('Writing {}'.format(jpgfile))
    rgb = _galex_rgb(comods)
    imsave_jpeg(jpgfile, rgb, origin='lower')

    jpgfile = os.path.join(output_dir, '{}-galex-resid.jpg'.format(galaxy))
    if verbose:
        print('Writing {}'.format(jpgfile))
    rgb = _galex_rgb(coresids)
    imsave_jpeg(jpgfile, rgb, origin='lower')

    return 1
