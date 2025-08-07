"""
SGA.ellipse
===========

Code to perform ellipse photometry.

"""
import pdb # for debuggin

import time
import numpy as np
#from scipy.optimize import curve_fit
#import astropy.modeling

from photutils.isophote import (EllipseGeometry, Ellipse, EllipseSample,
                                Isophote, IsophoteList)
from photutils.isophote.sample import CentralEllipseSample
from photutils.isophote.fitter import CentralEllipseFitter

from SGA.logger import log


REF_SBTHRESH = [22, 22.5, 23, 23.5, 24, 24.5, 25, 25.5, 26] # surface brightness thresholds
REF_APERTURES = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 3.0] # multiples of MAJORAXIS

# ndim>1 columns when ellipse-fitting fails; note, this list is used by various
# build_catalog functions (e.g., check virgofilaments.build_catalog), so change
# with care!
FAILCOLS = ['sma', 'intens', 'intens_err', 'eps', 'eps_err',
            'pa', 'pa_err', 'x0', 'x0_err', 'y0', 'y0_err',
            'a3', 'a3_err', 'a4', 'a4_err', 'rms', 'pix_stddev',
            'stop_code', 'ndata', 'nflag', 'niter']
FAILDTYPES = [np.int16, np.float32, np.float32, np.float32, np.float32,
              np.float32, np.float32, np.float32, np.float32, np.float32, np.float32,
              np.float32, np.float32, np.float32, np.float32, np.float32, np.float32,
              np.int16, np.int16, np.int16, np.int16]


def _unpack_isofit(ellipsefit, filt, isofit, failed=False):
    """Unpack the IsophotList objects into a dictionary because the resulting pickle
    files are huge.

    https://photutils.readthedocs.io/en/stable/api/photutils.isophote.IsophoteList.html#photutils.isophote.IsophoteList

    """
    def _fill_failed():
        fail = {}
        for col, dtype in zip(FAILCOLS, FAILDTYPES):
            fail[f'{col}_{filt.lower()}'] = np.array([-1]).astype(dtype)
        return fail

    if failed:
        ellipsefit.update(_fill_failed())
    else:
        I = np.isfinite(isofit.intens) * np.isfinite(isofit.int_err)
        if np.sum(I) == 0:
            ellipsefit.update(_fill_failed())
        else:
            values = [isofit.sma[I], isofit.intens[I], isofit.int_err[I], isofit.eps[I], isofit.ellip_err[I],
                      isofit.pa[I], isofit.pa_err[I], isofit.x0[I], isofit.x0_err[I], isofit.y0[I], isofit.y0_err[I],
                      isofit.a3[I], isofit.a3_err[I], isofit.a4[I], isofit.a4_err[I], isofit.rms[I], isofit.pix_stddev[I],
                      isofit.stop_code[I], isofit.ndata[I], isofit.nflag[I], isofit.niter[I]]
            if len(values) != len(FAILCOLS):
                print('Unanticipated data model change!')
                raise ValueError
            data = {}
            for col, dtype, value in zip(FAILCOLS, FAILDTYPES, values):
                data[f'{col}_{filt.lower()}'] = value.astype(dtype)
            ellipsefit.update(data)
    return ellipsefit


def _integrate_isophot_one(args):
    """Wrapper function for the multiprocessing."""
    return integrate_isophot_one(*args)


def integrate_isophot_one(img, sma, theta, eps, x0, y0,
                          integrmode, sclip, nclip):
    """Integrate the ellipse profile at a single semi-major axis.

    theta in radians

    """
    import copy
    #g = iso.sample.geometry # fixed geometry
    #g = copy.deepcopy(iso.sample.geometry) # fixed geometry
    g = EllipseGeometry(x0=x0, y0=y0, eps=eps, sma=sma, pa=theta)

    # Use the same integration mode and clipping parameters.
    # The central pixel is a special case:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        if g.sma == 0.0:
            gcen = copy.deepcopy(g)
            gcen.sma = 0.0
            gcen.eps = 0.0
            gcen.pa = 0.0
            censamp = CentralEllipseSample(img, 0.0, geometry=gcen,
                                           integrmode=integrmode, sclip=sclip, nclip=nclip)
            out = CentralEllipseFitter(censamp).fit()
        else:
            #g.sma *= pixscalefactor
            #g.x0 *= pixscalefactor
            #g.y0 *= pixscalefactor

            sample = EllipseSample(img, sma=g.sma, geometry=g, integrmode=integrmode,
                                   sclip=sclip, nclip=nclip)
            sample.update(fixed_parameters=True)
            #print(filt, g.sma, sample.mean)

            # Create an Isophote instance with the sample.
            out = Isophote(sample, 0, True, 0)

    return out


def multifit(obj, imgs, varimgs, maskbits, bands=['g', 'r', 'i', 'z'],
             pixscale=0.262, pixfactor=1., delta_logsma=5., delta_sma=5.,
             mp=1, maxsma=None, integrmode='median', nclip=3, sclip=3,
             linearsma=False, sbthresh=REF_SBTHRESH, apertures=REF_APERTURES,
             verbose=False):
    """Multi-band ellipse-fitting, broadly based on--
    https://github.com/astropy/photutils-datasets/blob/master/notebooks/isophote/isophote_example4.ipynb

    """
    import multiprocessing

    def logspaced_integers(limit, n):
        #https://stackoverflow.com/questions/12418234/logarithmically-spaced-integers
        result = [1]
        if n > 1:  # just a check to avoid ZeroDivisionError
            ratio = (float(limit)/result[-1]) ** (1.0/(n-len(result)))
        while len(result) < n:
            next_value = result[-1]*ratio
            if next_value - result[-1] >= 1:
                # safe zone. next_value will be a different integer
                result.append(next_value)
            else:
                # problem! same integer. we need to find
                # next_value by artificially incrementing previous
                # value
                result.append(result[-1]+1)
                # recalculate the ratio so that the remaining
                # values will scale correctly
                ratio = (float(limit)/result[-1]) ** (1.0/(n-len(result)))
                #print(ratio, len(result), n)
                # round, re-adjust to 0 indexing (i.e. minus 1) and return np.uint64 array
        return np.array(list(map(lambda x: round(x)-1, result)), dtype=int)


    # Initialize the output dictionary, starting from the galaxy geometry in the
    # 'data' dictionary.
    ellipsefit = dict()
    nband, width, _ = imgs.shape
    assert(nband == len(bands))

    ## Fix the center to be the peak (pixel) values. Could also use bx,by here
    ## from Tractor.  Also initialize the geometry with the moment-derived
    ## values.  Note that (x,y) are switched between MGE and photutils!!
    #for key, newkey in zip(['largeshift', 'ra_moment', 'dec_moment', 'majoraxis', 'pa', 'eps'],
    #                       ['largeshift', 'ra_moment', 'dec_moment', 'majoraxis', 'pa_moment', 'eps_moment']):
    #    if key == 'majoraxis':
    #        ellipsefit['sma_moment'] = mge['majoraxis'] * refpixscale # [arcsec]
    #        ellipsefit[newkey] = mge[key]
    #
    #if copy_mw_transmission:
    #    ellipsefit['ebv'] = mge['ebv']
    #    for band in bands:
    #        if 'mw_transmission_{}'.format(band.lower()) in mge.keys():
    #            ellipsefit['mw_transmission_{}'.format(band.lower())] = mge['mw_transmission_{}'.format(band.lower())]
    #
    #ellipsefit['ba_moment'] = np.float32(1 - mge['eps']) # note!
    #
    #for mgekey, ellkey in zip(['ymed', 'xmed'], ['x0_moment', 'y0_moment']):
    #    ellipsefit[ellkey] = mge[mgekey]
    #
    #majoraxis = mge['majoraxis'] # [pixel]

    # Get the mean geometry of the system by ellipse-fitting the inner part and
    # taking the mean values of everything.


    # https://photutils.readthedocs.io/en/latest/user_guide/isophote.html
    # NB: (x,y) are switched in photutils and PA is measured CCW from
    # the x-axis while out PA is CCW from the y-axis!
    [bx, by, semia, ba, pa] = list(obj['BX', 'BY', 'SEMIA', 'BA', 'PA'].values())
    eps = 1. - ba

    geo = EllipseGeometry(x0=by, y0=bx, eps=eps, sma=0.5*semia,
                          pa=np.radians(pa-90.))
    geo_cen = EllipseGeometry(x0=by, y0=bx, eps=0., sma=0., pa=0.)

    debug = False#True
    if debug:
        import matplotlib.pyplot as plt
        from photutils.aperture import EllipticalAperture
        aper = EllipticalAperture(
            (geo.x0, geo.y0), geo.sma,
            geo.sma * (1 - geo.eps), geo.pa)
        plt.clf()
        plt.imshow(np.log10(imgs[0, :, :]), origin='lower')
        aper.plot(color='white')
        plt.savefig('ioannis/tmp/junk.png')
        plt.close()
    #ellipse = Ellipse(imgs[0, :, :], geometry=geo)

    # Integrate to the edge [pixels].
    if maxsma is None:
        maxsma = 0.95 * (width / 2.) / np.cos(geo.pa % (np.pi/4.))

    # This algorithm can fail if there are too few points or if
    # all the points aren't unique.
    try:
        if linearsma:
            sma = np.arange(0, np.ceil(maxsma), delta_sma)
            log.info(f'  maxsma={maxsma:.2f} pix, delta_sma={delta_sma:.1f} pix, ' + \
                     f'nsma={len(sma)}')
        else:
            nsma = int(np.ceil(maxsma / delta_logsma))
            sma = logspaced_integers(maxsma, nsma)
            log.info(f'  maxsma={maxsma:.2f} pix, delta_logsma={delta_logsma:.1f} ' + \
                     f'log-pix, nsma={len(sma)}')
        assert(len(sma) == len(np.unique(sma)))
    except:
        msg = 'There was a problem generating the sma vector.'
        log.critical(msg)
        return {}

    # Measure the surface-brightness profile.
    pool = multiprocessing.Pool(mp)
    for iband, filt in enumerate(bands):
        # account for a possible variable pixel scale
        filtx0 = geo.x0 * pixfactor
        filty0 = geo.y0 * pixfactor
        filtsma = sma * pixfactor
        filtsma = np.unique(filtsma)

        isobandfit = pool.map(_integrate_isophot_one, [(
            imgs[iband, :, :], onesma, pa, eps, filtx0, filty0,
            integrmode, sclip, nclip) for onesma in filtsma])
        ellipsefit = _unpack_isofit(ellipsefit, filt, IsophoteList(isobandfit))
    pdb.set_trace()


    tall = time.time()
    for iband, filt in enumerate(bands):
        filtsma = np.unique(filtsma)
        isobandfit = pool.map(_integrate_isophot_one, [(
            imgs[iband, :, :], onesma, pa, eps, x0*pixfactor, y0*pixfactor,
            integrmode, sclip, nclip) for onesma in sma*pixfactor])
        ellipsefit = _unpack_isofit(ellipsefit, filt, IsophoteList(isobandfit))

        #print('Fitting {}-band took...'.format(filt.lower()), end='')
        #img = data['{}_masked'.format(filt.lower())][igal]
        #
        ## handle GALEX and WISE
        #if 'filt2pixscale' in data.keys():
        #    pixscale = data['filt2pixscale'][filt]
        #    if np.isclose(pixscale, refpixscale): # avoid rounding issues
        #        pixscale = refpixscale
        #        pixscalefactor = 1.0
        #    else:
        #        pixscalefactor = refpixscale / pixscale
        #else:
        #    pixscalefactor = 1.0

        x0 = pixfactor * x0
        y0 = pixfactor * y0

        filtsma = np.round(sma * pixscalefactor).astype('f4')
        #filtsma = np.round(sma[::int(1/(pixscalefactor))] * pixscalefactor).astype('f4')
        filtsma = np.unique(filtsma)
        assert(len(np.unique(filtsma)) == len(filtsma))

        # Loop on the reference band isophotes.
        t0 = time.time()
        #isobandfit = pool.map(_integrate_isophot_one, [(iso, img, pixscalefactor, integrmode, sclip, nclip)

        ## In extreme cases, and despite my best effort in io.read_multiband, the
        ## image at the central position of the galaxy can end up masked, which
        ## always points to a deeper issue with the data (e.g., bleed trail,
        ## extremely bright star, etc.). Capture that corner case here.
        #imasked, val = False, []
        #for xb in box:
        #    for yb in box:
        #        val.append(img.mask[int(xb+y0), int(yb+x0)])
        #        #val.append(img.mask[int(xb+x0), int(yb+y0)])
        #if np.any(val):
        #    imasked = True

        # corner case: no data in the image or fully masked
        if np.sum(img.data) == 0 or np.sum(img.mask) == np.product(img.shape):
            ellipsefit = _unpack_isofit(ellipsefit, filt, None, failed=True)
        else:
            if imasked:
                #if img.mask[int(ellipsefit['x0']), int(ellipsefit['y0'])]:
                print(' Central pixel is masked; resorting to extreme measures!')
                #try:
                #    raise ValueError
                #except:
                #    pdb.set_trace()
                ellipsefit = _unpack_isofit(ellipsefit, filt, None, failed=True)
            else:
                isobandfit = pool.map(_integrate_isophot_one, [(
                    img, _sma, ellipsefit['pa_moment'], ellipsefit['eps_moment'], x0,
                    y0, integrmode, sclip, nclip) for _sma in filtsma])
                ellipsefit = _unpack_isofit(ellipsefit, filt, IsophoteList(isobandfit))

        #print('...{:.3f} sec'.format(time.time() - t0))

    #print('Time for all images = {:.3f} min'.format((time.time()-tall)/60))
    pdb.set_trace()

    ellipsefit['success'] = True

    # Perform elliptical aperture photometry--
    print('Performing elliptical aperture photometry.')
    t0 = time.time()
    cog = ellipse_cog(bands, data, ellipsefit, igal=igal,
                      pool=pool, sbthresh=sbthresh, apertures=apertures)
    ellipsefit.update(cog)
    del cog
    print('Time = {:.3f} min'.format( (time.time() - t0) / 60))

    pool.close()

    # Write out
    if not nowrite:
        if galaxyinfo is None:
            outgalaxyinfo = None
        else:
            outgalaxyinfo = galaxyinfo[igal]
            ellipsefit.update(galaxyinfo[igal])

        legacyhalos.io.write_ellipsefit(galaxy, galaxydir, ellipsefit,
                                        galaxy_id=galaxy_id,
                                        galaxyinfo=outgalaxyinfo,
                                        refband=refband,
                                        sbthresh=sbthresh,
                                        apertures=apertures,
                                        bands=ellipsefit['bands'],
                                        verbose=True,
                                        copy_mw_transmission=copy_mw_transmission,
                                        filesuffix=data['filesuffix'])

    return ellipsefit


def ellipsefit_multiband(galaxy, galaxydir, REFIDCOLUMN, read_multiband_function,
                         mp=1, bands=['g', 'r', 'i', 'z'], pixscale=0.262, galex=False,
                         unwise=False, integrmode='median', nclip=3, sclip=3,
                         sbthresh=REF_SBTHRESH, apertures=REF_APERTURES, delta_logsma=5,
                         maxsma=None, refidcolumn=None, verbose=False, nowrite=False,
                         clobber=False):
    """Top-level wrapper script to do ellipse-fitting on a single galaxy.

    fitgeometry - fit for the ellipse parameters (do not use the mean values
      from MGE).

    """
    try:
        data = read_multiband_function(galaxy, galaxydir, bands=bands,
                                       pixscale=pixscale, unwise=unwise,
                                       galex=galex, verbose=verbose)
    except:
        log.warning(f'Problem reading (or missing) data for {galaxydir}/{galaxy}')
        return 0

    dataprefix = ['opt']
    if unwise:
        dataprefix += ['unwise']
    if galex:
        dataprefix += ['galex']

    for prefix in dataprefix:
        maskbits = data[f'{prefix}_maskbits']
        bands = data[f'{prefix}_bands']
        pixscale = data[f'{prefix}_pixscale']
        pixfactor = data['opt_pixscale'] / pixscale
        varimgs = data[f'{prefix}_variance']
        for iobj, obj in enumerate(data['sample']):
            refid = obj[REFIDCOLUMN]
            imgs = data[f'{prefix}_images'][iobj, :, :, :]

            ellipsefit = multifit(obj, imgs, varimgs, maskbits, bands, pixscale=pixscale,
                                  pixfactor=pixfactor, delta_logsma=delta_logsma,
                                  maxsma=maxsma, mp=mp, sbthresh=sbthresh, apertures=apertures,
                                  integrmode=integrmode, nclip=nclip, sclip=sclip,
                                  verbose=verbose)

        # merge all the ellipse-fitting results and write out
        # add to header:
        #  --integrmode
        #  --sclip
        #  --nclip
        #  --fitgeometry

        # output data model:
        #  --use all_bands but do not write to table
        #  --psfdepth, etc.
        #  --maxsma
        #  --combine masking and ellipse-fitting bits into a single bitmask
        #    --largeshift
        #    --...


        pdb.set_trace()

    return 1 # success!
