"""
SGA.ellipse
===========

Code to perform ellipse photometry.

"""
import pdb # for debuggin

import warnings
from time import time
import numpy as np
#from scipy.optimize import curve_fit
#import astropy.modeling
from photutils.isophote import EllipseGeometry, Ellipse, IsophoteList

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
    """Integrate the ellipse profile at a single semi-major axis (in
    pixels).

    theta in radians, CCW from the x-axis

    """
    from photutils.isophote.sample import CentralEllipseSample
    from photutils.isophote.fitter import CentralEllipseFitter
    from photutils.isophote import EllipseSample, Isophote

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        # central pixel is a special case; see
        # https://github.com/astropy/photutils-datasets/blob/main/notebooks/isophote/isophote_example4.ipynb
        if sma == 0.:
            samp = CentralEllipseSample(img, sma=sma, x0=x0, y0=y0, eps=eps,
                                        position_angle=theta, sclip=sclip,
                                        nclip=nclip, integrmode=integrmode)
            samp.update(fixed_parameters=[True]*4) # x0, y0, theta, eps
            out = CentralEllipseFitter(samp).fit()
        else:
            samp = EllipseSample(img, sma=sma, x0=x0, y0=y0, eps=eps,
                                 position_angle=theta, sclip=sclip,
                                 nclip=nclip, integrmode=integrmode)
            samp.update(fixed_parameters=[True]*4) # x0, y0, theta, eps
            out = Isophote(samp, 0, True, 0)

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

    # Integrate to the edge [pixels].
    if maxsma is None:
        maxsma = width / np.sqrt(2.) # =sqrt(2)*(width/2)

    try:
        if linearsma:
            sma = np.arange(0, np.ceil(maxsma), delta_sma)
            log.info(f'maxsma={maxsma:.2f} pix, delta_sma={delta_sma:.1f} pix, ' + \
                     f'nsma={len(sma)}')
        else:
            nsma = int(np.ceil(maxsma / delta_logsma))
            #sma = logspaced_integers(maxsma, nsma)
            sma = np.hstack((0., np.logspace(0., np.log10(maxsma), nsma)))
            log.info(f'maxsma={maxsma:.2f} pix, delta_logsma={delta_logsma:.1f} ' + \
                     f'log-pix, nsma={len(sma)}')
        assert(len(sma) == len(np.unique(sma)))
    except:
        msg = 'There was a problem generating the sma vector.'
        log.critical(msg)
        return {}

    # Measure the surface-brightness profile.
    tall = time()
    for iband, filt in enumerate(bands):
        #log.info(f'Fitting {filt}-band...')
        t0 = time()

        # account for a possible variable pixel scale
        filtx0 = geo.x0 * pixfactor
        filty0 = geo.y0 * pixfactor
        filtsma = sma * pixfactor
        filtsma = np.unique(filtsma)

        with multiprocessing.Pool(mp) as P:
            isobandfit = P.map(_integrate_isophot_one, [(
                imgs[iband, :, :], onesma, geo.pa, geo.eps, filtx0, filty0,
                integrmode, sclip, nclip) for onesma in filtsma])
        ellipsefit = _unpack_isofit(ellipsefit, filt, IsophoteList(isobandfit))

        dt = (time() - t0) / 60.
        if dt > 60.:
            dt /= 60.
            unit = 'minutes'
        else:
            unit = 'seconds'
        log.info(f'Ellipse-fitting {filt}-band took {dt:.3f} {unit}')

    log.info(f'Time to fit all images = {(time()-tall)/60.:.3f} minutes')

    pdb.set_trace()

    # Perform elliptical aperture photometry--
    print('Performing elliptical aperture photometry.')
    t0 = time.time()
    cog = ellipse_cog(bands, data, ellipsefit, igal=igal,
                      pool=pool, sbthresh=sbthresh, apertures=apertures)
    ellipsefit.update(cog)
    del cog
    print('Time = {:.3f} min'.format( (time.time() - t0) / 60))

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

    if not nowrite:
        write_ellipsefit(galaxy, galaxydir, ellipsefit,
                         galaxy_id=galaxy_id,
                         galaxyinfo=outgalaxyinfo,
                         refband=refband,
                         sbthresh=sbthresh,
                         apertures=apertures,
                         bands=ellipsefit['bands'],
                         verbose=True,
                         copy_mw_transmission=copy_mw_transmission,
                         filesuffix=data['filesuffix'])

    return 1 # success!
