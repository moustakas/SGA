"""
SGA.ellipse
===========

Code to perform ellipse photometry.

"""
import numpy as np
from scipy.optimize import curve_fit
import astropy.modeling

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


def ellipsefit_multiband(galaxy, galaxydir, data, igal=0, galaxy_id='',
                         refband='r', nproc=1,
                         integrmode='median', nclip=3, sclip=3,
                         maxsma=None, logsma=True, delta_logsma=5.0, delta_sma=1.0,
                         sbthresh=REF_SBTHRESH, apertures=REF_APERTURES,
                         copy_mw_transmission=False,
                         galaxyinfo=None, input_ellipse=None,
                         fitgeometry=False, nowrite=False, verbose=False):
    """Multi-band ellipse-fitting, broadly based on--
    https://github.com/astropy/photutils-datasets/blob/master/notebooks/isophote/isophote_example4.ipynb

    Some, but not all hooks for fitgeometry=True are in here, so user beware.

    galaxyinfo - additional dictionary to append to the output file

    galaxy_id - add a unique ID number to the output filename (via
      io.write_ellipsefit).

    """
    import multiprocessing

    bands, refband, refpixscale = data['bands'], data['refband'], data['refpixscale']

    if galaxyinfo is not None:
        galaxyinfo = np.atleast_1d(galaxyinfo)
        assert(len(galaxyinfo)==len(data['mge']))
    
    # If fitgeometry=True then fit for the geometry as a function of semimajor
    # axis, otherwise (the default) use the mean geometry of the galaxy to
    # extract the surface-brightness profile.
    if fitgeometry:
        maxrit = None
    else:
        maxrit = -1

    # Initialize the output dictionary, starting from the galaxy geometry in the
    # 'data' dictionary.
    ellipsefit = dict()
    ellipsefit['integrmode'] = integrmode
    ellipsefit['sclip'] = np.int16(sclip)
    ellipsefit['nclip'] = np.int16(nclip)
    ellipsefit['fitgeometry'] = fitgeometry

    if input_ellipse:
        ellipsefit['input_ellipse'] = True
    else:
        ellipsefit['input_ellipse'] = False

    # This is fragile, but copy over a specific set of keys from the data dictionary--
    copykeys = ['bands', 'refband', 'refpixscale',
                'refband_width', 'refband_height',
                #'psfsigma_g', 'psfsigma_r', 'psfsigma_z',
                'psfsize_g', #'psfsize_min_g', 'psfsize_max_g',
                'psfdepth_g', #'psfdepth_min_g', 'psfdepth_max_g', 
                'psfsize_r', #'psfsize_min_r', 'psfsize_max_r',
                'psfdepth_r', #'psfdepth_min_r', 'psfdepth_max_r',
                'psfsize_z', #'psfsize_min_z', 'psfsize_max_z',
                'psfdepth_z'] #'psfdepth_min_z', 'psfdepth_max_z']
    for key in copykeys:
        if key in data.keys():
            ellipsefit[key] = data[key]

    img = data['{}_masked'.format(refband)][igal]
    mge = data['mge'][igal]

    # Fix the center to be the peak (pixel) values. Could also use bx,by here
    # from Tractor.  Also initialize the geometry with the moment-derived
    # values.  Note that (x,y) are switched between MGE and photutils!!
    for key, newkey in zip(['largeshift', 'ra_moment', 'dec_moment', 'majoraxis', 'pa', 'eps'],
                           ['largeshift', 'ra_moment', 'dec_moment', 'majoraxis', 'pa_moment', 'eps_moment']):
        if key == 'majoraxis':
            ellipsefit['sma_moment'] = mge['majoraxis'] * refpixscale # [arcsec]
        ellipsefit[newkey] = mge[key]

    if copy_mw_transmission:
        ellipsefit['ebv'] = mge['ebv']
        for band in bands:
            if 'mw_transmission_{}'.format(band.lower()) in mge.keys():
                ellipsefit['mw_transmission_{}'.format(band.lower())] = mge['mw_transmission_{}'.format(band.lower())]
        
    ellipsefit['ba_moment'] = np.float32(1 - mge['eps']) # note!
    
    for mgekey, ellkey in zip(['ymed', 'xmed'], ['x0_moment', 'y0_moment']):
        ellipsefit[ellkey] = mge[mgekey]

    majoraxis = mge['majoraxis'] # [pixel]

    # Get the mean geometry of the system by ellipse-fitting the inner part and
    # taking the mean values of everything.

    # http://photutils.readthedocs.io/en/stable/isophote_faq.html#isophote-faq
    # Note: position angle in photutils is measured counter-clockwise from the
    # x-axis, while .pa in MGE measured counter-clockwise from the y-axis.
    geometry0 = EllipseGeometry(x0=ellipsefit['x0_moment'], y0=ellipsefit['y0_moment'],
                                eps=ellipsefit['eps_moment'], sma=0.5*majoraxis, 
                                pa=np.radians(ellipsefit['pa_moment']-90))
    ellipse0 = Ellipse(img, geometry=geometry0)
    #import matplotlib.pyplot as plt
    #plt.imshow(img, origin='lower') ; plt.scatter(ellipsefit['y0'], ellipsefit['x0'], s=50, color='red') ; plt.savefig('junk.png')

    if fitgeometry:
        ellipsefit = _fitgeometry_refband(ellipsefit, geometry0, majoraxis, refband,
                                          integrmode=integrmode, sclip=sclip, nclip=nclip,
                                          verbose=verbose)
    
    # Re-initialize the EllipseGeometry object, optionally using an external set
    # of ellipticity parameters.
    if input_ellipse:
        print('Using input ellipse parameters.')
        ellipsefit['input_ellipse'] = True
        input_eps, input_pa = input_ellipse['eps'], input_ellipse['pa'] % 180
        geometry = EllipseGeometry(x0=ellipsefit['x0_moment'], y0=ellipsefit['y0_moment'],
                                   eps=input_eps, sma=majoraxis, 
                                   pa=np.radians(input_pa-90))
    else:
        # Note: we use the MGE, not fitted geometry here because it's more
        # reliable based on visual inspection.
        geometry = EllipseGeometry(x0=ellipsefit['x0_moment'], y0=ellipsefit['y0_moment'],
                                   eps=ellipsefit['eps_moment'], sma=majoraxis, 
                                   pa=np.radians(ellipsefit['pa_moment']-90))

    geometry_cen = EllipseGeometry(x0=ellipsefit['x0_moment'], y0=ellipsefit['y0_moment'],
                                   eps=0.0, sma=0.0, pa=0.0)
    #ellipsefit['geometry'] = geometry # can't save an object in an .asdf file
    ellipse = Ellipse(img, geometry=geometry)

    # Integrate to the edge [pixels].
    if maxsma is None:
        maxsma = 0.95 * (data['refband_width']/2) / np.cos(geometry.pa % (np.pi/4))
    ellipsefit['maxsma'] = np.float32(maxsma) # [pixels]

    if logsma:
        #https://stackoverflow.com/questions/12418234/logarithmically-spaced-integers
        def _mylogspace(limit, n):
            result = [1]
            if n > 1:  # just a check to avoid ZeroDivisionError
                ratio = (float(limit)/result[-1]) ** (1.0/(n-len(result)))
            while len(result) < n:
                next_value = result[-1]*ratio
                if next_value - result[-1] >= 1:
                    # safe zone. next_value will be a different integer
                    result.append(next_value)
                else:
                    # problem! same integer. we need to find next_value by artificially incrementing previous value
                    result.append(result[-1]+1)
                    # recalculate the ratio so that the remaining values will scale correctly
                    ratio = (float(limit)/result[-1]) ** (1.0/(n-len(result)))
                    #print(ratio, len(result), n)
            # round, re-adjust to 0 indexing (i.e. minus 1) and return np.uint64 array
            return np.array(list(map(lambda x: round(x)-1, result)), dtype=int)

        # this algorithm can fail if there are too few points
        nsma = np.ceil(maxsma / delta_logsma).astype('int')
        sma = _mylogspace(maxsma, nsma).astype('f4')
        assert(len(sma) == len(np.unique(sma)))

        #sma = np.hstack((0, np.logspace(0, np.ceil(np.log10(maxsma)).astype('int'), nsma, dtype=int))).astype('f4')
        print('  maxsma={:.2f} pix, delta_logsma={:.1f} log-pix, nsma={}'.format(maxsma, delta_logsma, len(sma)))
    else:
        sma = np.arange(0, np.ceil(maxsma), delta_sma).astype('f4')
        #ellipsefit['sma'] = np.arange(np.ceil(maxsma)).astype('f4')
        print('  maxsma={:.2f} pix, delta_sma={:.1f} pix, nsma={}'.format(maxsma, delta_sma, len(sma)))

    # this assert will fail when integrating the curve of growth using
    # integrate.simps because the x-axis values have to be unique.
    assert(len(np.unique(sma)) == len(sma))

    nbox = 3
    box = np.arange(nbox)-nbox // 2
    
    refpixscale = data['refpixscale']

    # Now get the surface brightness profile.  Need some more code for this to
    # work with fitgeometry=True...
    pool = multiprocessing.Pool(nproc)

    tall = time.time()
    for filt in bands:
        print('Fitting {}-band took...'.format(filt.lower()), end='')
        img = data['{}_masked'.format(filt.lower())][igal]

        # handle GALEX and WISE
        if 'filt2pixscale' in data.keys():
            pixscale = data['filt2pixscale'][filt]            
            if np.isclose(pixscale, refpixscale): # avoid rounding issues
                pixscale = refpixscale                
                pixscalefactor = 1.0
            else:
                pixscalefactor = refpixscale / pixscale
        else:
            pixscalefactor = 1.0

        x0 = pixscalefactor * ellipsefit['x0_moment']
        y0 = pixscalefactor * ellipsefit['y0_moment']
        filtsma = np.round(sma * pixscalefactor).astype('f4')
        #filtsma = np.round(sma[::int(1/(pixscalefactor))] * pixscalefactor).astype('f4')
        filtsma = np.unique(filtsma)
        assert(len(np.unique(filtsma)) == len(filtsma))
    
        # Loop on the reference band isophotes.
        t0 = time.time()
        #isobandfit = pool.map(_integrate_isophot_one, [(iso, img, pixscalefactor, integrmode, sclip, nclip)

        # In extreme cases, and despite my best effort in io.read_multiband, the
        # image at the central position of the galaxy can end up masked, which
        # always points to a deeper issue with the data (e.g., bleed trail,
        # extremely bright star, etc.). Capture that corner case here.
        imasked, val = False, []
        for xb in box:
            for yb in box:
                val.append(img.mask[int(xb+y0), int(yb+x0)])
                #val.append(img.mask[int(xb+x0), int(yb+y0)])
        if np.any(val):
            imasked = True

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
    
        print('...{:.3f} sec'.format(time.time() - t0))

    print('Time for all images = {:.3f} min'.format((time.time()-tall)/60))

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


def ellipsefit_multiband(galaxy, galaxydir, data, galaxyinfo=None,
                         pixscale=0.262, nproc=1,
                         bands=['g', 'r', 'z'], integrmode='median',
                         nclip=3, sclip=3, sbthresh=REF_SBTHRESH,
                         apertures=REF_APERTURES,
                         delta_sma=1.0, delta_logsma=5, maxsma=None, logsma=True,
                         input_ellipse=None, fitgeometry=False,
                         verbose=False, debug=False, nowrite=False, clobber=False):
    """Top-level wrapper script to do ellipse-fitting on a single galaxy.

    fitgeometry - fit for the ellipse parameters (do not use the mean values
      from MGE).

    """
    from legacyhalos.io import get_ellipsefit_filename

    if bool(data):
        if data['missingdata']:
            if os.path.isfile(os.path.join(galaxydir, '{}-{}-coadds.isdone'.format(galaxy, data['filesuffix']))):
                return 1
            else:
                return 0

        if data['failed']: # all galaxies dropped
            return 1

        if 'galaxy_id' in data.keys():
            galaxy_id = np.atleast_1d(data['galaxy_id'])
        else:
            galaxy_id = ['']

        for igal, galid in enumerate(galaxy_id):
            ellipsefitfile = get_ellipsefit_filename(galaxy, galaxydir, galaxy_id=str(galid),
                                                     filesuffix=data['filesuffix'])
            if os.path.isfile(ellipsefitfile) and not clobber:
                print('Skipping existing catalog {}'.format(ellipsefitfile))
            else:
                ellipsefit = ellipsefit_multiband(galaxy, galaxydir, data,
                                                  galaxyinfo=galaxyinfo,
                                                  igal=igal, galaxy_id=str(galid),
                                                  delta_logsma=delta_logsma, maxsma=maxsma,
                                                  delta_sma=delta_sma, logsma=logsma,
                                                  refband=refband, nproc=nproc, sbthresh=sbthresh,
                                                  apertures=apertures,
                                                  integrmode=integrmode, nclip=nclip, sclip=sclip,
                                                  input_ellipse=input_ellipse,
                                                  verbose=verbose, fitgeometry=False,
                                                  nowrite=False)
        return 1
    else:
        # An object can get here if it's a "known" failure, e.g., if the object
        # falls off the edge of the footprint (and therefore it will never have
        # coadds).
        if os.path.isfile(os.path.join(galaxydir, '{}-{}-coadds.isdone'.format(galaxy, 'custom'))):
            return 1
        else:
            return 0
