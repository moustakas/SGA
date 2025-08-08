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

from SGA.logger import log


REF_SBTHRESH = [22, 22.5, 23, 23.5, 24, 24.5, 25, 25.5, 26] # surface brightness thresholds
REF_APERTURES = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 3.0] # multiples of MAJORAXIS

# ndim>1 columns when ellipse-fitting fails; note, this list is used
# by various build_catalog functions, so change with care!
FAILCOLS = ['sma', 'intens', 'intens_err', 'eps', 'eps_err',
            'pa', 'pa_err', 'x0', 'x0_err', 'y0', 'y0_err',
            'a3', 'a3_err', 'a4', 'a4_err'] + ['ndata']
FAILDTYPES = [np.float32] * 15 + [np.int16]

def unpack_isofit(filt, isofit, failed=False):
    """Unpack a selection of IsophotList attributes into a dictionary.

    https://photutils.readthedocs.io/en/latest/api/photutils.isophote.IsophoteList.html

    """
    def fill_failed():
        fail = {}
        for col, dtype in zip(FAILCOLS, FAILDTYPES):
            fail[f'{col}_{filt}'] = np.array([-1]).astype(dtype)
        return fail

    if failed:
        return fill_failed()
    else:
        I = np.isfinite(isofit.intens) * np.isfinite(isofit.int_err)
        if np.sum(I) == 0:
            return fill_failed()
        else:
            values = [isofit.sma[I], isofit.intens[I], isofit.int_err[I], isofit.eps[I],
                      isofit.ellip_err[I], isofit.pa[I], isofit.pa_err[I], isofit.x0[I],
                      isofit.x0_err[I], isofit.y0[I], isofit.y0_err[I], isofit.a3[I],
                      isofit.a3_err[I], isofit.a4[I], isofit.a4_err[I], isofit.ndata[I]]
            if len(values) != len(FAILCOLS):
                msg = 'Unanticipated data model change in ellipse-fitting code!'
                log.critical(msg)
                raise ValueError(msg)
            out = {}
            for col, dtype, value in zip(FAILCOLS, FAILDTYPES, values):
                out[f'{col}_{filt}'] = value.astype(dtype)
            return out


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


def multifit(obj, imgs, varimgs, maskbits, sma, bands=['g', 'r', 'i', 'z'],
             pixscale=0.262, pixfactor=1., mp=1, integrmode='median', nclip=3,
             sclip=3, sbthresh=REF_SBTHRESH, apertures=REF_APERTURES,
             debug=False):
    """Multi-band ellipse-fitting, broadly based on--
    https://github.com/astropy/photutils-datasets/blob/master/notebooks/isophote/isophote_example4.ipynb

    See also:
    https://photutils.readthedocs.io/en/latest/user_guide/isophote.html

    """
    import multiprocessing
    from photutils.isophote import EllipseGeometry, IsophoteList

    # Initialize the object geometry. NB: (x,y) are switched in
    # photutils and PA is measured CCW from the x-axis while PA is CCW
    # from the y-axis!
    cols = ['BX', 'BY', 'SEMIA', 'BA', 'PA']
    [bx, by, semia, ba, pa] = list(obj[cols].values())
    geo = EllipseGeometry(x0=by, y0=bx, eps=1.-ba,
                          sma=semia, # [pixels]
                          pa=np.radians(pa-90.))

    if debug:
        import matplotlib.pyplot as plt
        from photutils.aperture import EllipticalAperture
        aper = EllipticalAperture((geo.x0, geo.y0), geo.sma,
                                  geo.sma * (1 - geo.eps),
                                  geo.pa)
        plt.clf()
        plt.imshow(np.log10(np.sum(imgs, axis=0)), origin='lower')
        aper.plot(color='white')
        plt.savefig('ioannis/tmp/junk.png')
        plt.close()

    # Measure the surface-brightness profile in each bandpass.
    tall = time()
    out = {}
    for iband, filt in enumerate(bands):
        t0 = time()

        # account for a possible variable pixel scale
        filtx0 = geo.x0 * pixfactor
        filty0 = geo.y0 * pixfactor
        filtsma = sma * pixfactor
        #print(filt, sma)

        with multiprocessing.Pool(mp) as P:
            isobandfit = P.map(_integrate_isophot_one, [(
                imgs[iband, :, :], onesma, geo.pa, geo.eps, filtx0, filty0,
                integrmode, sclip, nclip) for onesma in filtsma])
        out.update(unpack_isofit(filt, IsophoteList(isobandfit)))

        dt = (time() - t0) / 60.
        if dt > 60.:
            dt /= 60.
            unit = 'minutes'
        else:
            unit = 'seconds'
        log.info(f'Ellipse-fitting the {filt}-band took {dt:.3f} {unit}')

    log.info(f'Ellipse-fitting all bandpasses took {(time()-tall)/60.:.3f} minutes')

    ## Perform elliptical aperture photometry--
    #print('Performing elliptical aperture photometry.')
    #t0 = time.time()
    #cog = ellipse_cog(bands, data, ellipsefit, igal=igal,
    #                  pool=pool, sbthresh=sbthresh, apertures=apertures)
    #ellipsefit.update(cog)
    #del cog
    #print('Time = {:.3f} min'.format( (time.time() - t0) / 60))

    return out


def build_sma(width, maxsma=None, delta_logsma=4., delta_sma=1.,
              linearsma=False):
    """Build the semimajor axis array. By default, integrate to the
    edge of the mosaic (in pixels).

    """
    if maxsma is None:
        maxsma = width / np.sqrt(2.) # =sqrt(2)*(width/2)

    try:
        if linearsma:
            sma = np.arange(0, np.ceil(maxsma), delta_sma)
            log.info(f'maxsma={maxsma:.2f} pix, delta_sma={delta_sma:.1f} pix, ' + \
                     f'nsma={len(sma)}')
        else:
            #sma = logspaced_integers(maxsma, nsma)
            nsma = int(np.ceil(maxsma / delta_logsma))
            sma = np.hstack((0., np.logspace(0., np.log10(maxsma), nsma)))
            log.info(f'maxsma={maxsma:.2f} pix, delta_logsma={delta_logsma:.1f} ' + \
                     f'log-pix, nsma={len(sma)}')
        assert(len(sma) == len(np.unique(sma)))
    except:
        msg = 'There was a problem generating the sma vector.'
        log.critical(msg)
        raise ValueError(msg)

    return sma


def qa_ellipsefit(data, ellipsefit, dataprefix=['opt'], title=None):
    """Simple QA.

    """
    from photutils.isophote import EllipseGeometry
    from photutils.aperture import EllipticalAperture
    import matplotlib.pyplot as plt

    qafile = os.path.join('/global/cfs/cdirs/desi/users/ioannis/tmp',
                          f'qa-ellipsemask-{data["galaxy"]}.png')

    refband = data['opt_refband']
    pixscale = data['opt_pixscale']

    obj = data['sample'][0]
    img = data['opt_images'][0, 1, :, :]
    smas = ellipsefit[f'sma_{refband}'] # [pixels]

    cols = ['BX', 'BY', 'SEMIA', 'BA', 'PA']
    [bx, by, semia, ba, pa] = list(obj[cols].values())
    refg = EllipseGeometry(x0=by, y0=bx, eps=1.-ba, # note bx,by swapped
                           sma=semia, # [pixels]
                           pa=np.radians(pa-90.))
    refap = EllipticalAperture((refg.x0, refg.y0), refg.sma,
                                refg.sma*(1. - refg.eps), refg.pa)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4))

    #ax1.imshow(np.flipud(jpg), origin='lower', cmap='inferno')
    ax1.imshow(np.log10(img), origin='lower', cmap='cividis')
    ax1.axis('off')
    for sma in smas: # sma in pixels
        if sma == 0.:
            continue
        ap = EllipticalAperture((refg.x0, refg.y0), sma,
                                sma*(1. - refg.eps), refg.pa)
        ap.plot(color='k', lw=1, ax=ax1)
    refap.plot(color='cyan', lw=2, ls='--', ax=ax1)

    for prefix in dataprefix:
        bands = data[f'{prefix}_bands']
        pixscale = data[f'{prefix}_pixscale']
        for filt in bands:
            ax2.scatter(pixscale*ellipsefit[f'sma_{filt}'],
                        ellipsefit[f'intens_{filt}'], label=filt)
    ax2.set_yscale('log')
    ax2.legend(loc='upper right', ncol=2, fontsize=8)

    for prefix in dataprefix:
        bands = data[f'{prefix}_bands']
        pixscale = data[f'{prefix}_pixscale']
        for filt in bands:
            ax3.scatter(pixscale*ellipsefit[f'sma_{filt}'],
                        ellipsefit[f'intens_{filt}'])
    ax3.axhline(y=0, color='gray')
    #ax3.legend(loc='upper right')

    #last = np.abs(ellipsefit[f'intens_{refband}'][-1])
    #ax3.set_ylim(-3*last, 10*last)
    ax3.set_ylim(-0.01, 0.1)

    for xx in (ax2, ax3):
        xx.axvline(x=(pixscale*semia)**0.25, color='cyan', lw=2, ls='--',
                   label='Second-Moment Diameter')
        xx.set_xlabel('Semi-major axis (arcsec)')
        xx.set_ylabel(r'Surface Brightness (nanomaggies arcsec$^{-2}$)')


    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig('ioannis/tmp/junk.png')


def ellipsefit_multiband(galaxy, galaxydir, REFIDCOLUMN, read_multiband_function,
                         unpack_maskbits_function, MASKBITS, mp=1,
                         bands=['g', 'r', 'i', 'z'], pixscale=0.262, galex=False,
                         unwise=False, integrmode='median', nclip=3, sclip=3,
                         sbthresh=REF_SBTHRESH, apertures=REF_APERTURES, delta_logsma=5.,
                         maxsma=None, refidcolumn=None, verbose=False, nowrite=False,
                         clobber=False, qaplot=True):
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

    # Build the semimajor axis array.
    sma = build_sma(data['width'], maxsma=maxsma, delta_logsma=delta_logsma)

    # iterate over datasetss
    datasets = ['opt']
    if unwise:
        datasets += ['unwise']
    if galex:
        datasets += ['galex']

    # we need as many MASKBITS bit-masks as datasetss
    assert(len(MASKBITS) == len(datasets))

    ellipsefit = []
    for iobj, obj in enumerate(data['sample']):
        refid = obj[REFIDCOLUMN]

        fit = {}
        for idata, dataset in datasets:
            imgs = data[f'{dataset}_images'][iobj, :, :, :]
            bands = data[f'{dataset}_bands']
            pixscale = data[f'{dataset}_pixscale']
            pixfactor = data['opt_pixscale'] / pixscale
            varimgs = data[f'{dataset}_variance']

            # unpack the maskbits image to generate a per-band mask
            maskbits = data[f'{dataset}_maskbits']
            masks = unpack_maskbits(maskbits, imgs.shape, bands=bands,
                                    BITS=MASKBITS[idata])


            pdb.set_trace()


            print('####### Adjust sma by pixfactor')

            out = multifit(obj, imgs, varimgs, maskbits, sma, bands,
                           pixscale=pixscale, pixfactor=pixfactor,
                           mp=mp, sbthresh=sbthresh, apertures=apertures,
                           integrmode=integrmode, nclip=nclip, sclip=sclip)
            fit.update(out)

        ellipsefit.append(fit)

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

    if qaplot:
        qa_ellipsefit(data, ellipsefit, datasets=datasets)

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
