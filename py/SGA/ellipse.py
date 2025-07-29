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

def legacyhalos_ellipse(galaxy, galaxydir, data, galaxyinfo=None,
                        pixscale=0.262, nproc=1, refband='r',
                        bands=['g', 'r', 'z'], integrmode='median',
                        nclip=3, sclip=3, sbthresh=REF_SBTHRESH,
                        apertures=REF_APERTURES,
                        delta_sma=1.0, delta_logsma=5, maxsma=None, logsma=True,
                        copy_mw_transmission=False,
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
                                                  copy_mw_transmission=copy_mw_transmission,
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



def ellipse_mask_sky(racen, deccen, semia, semib, phi, ras, decs):
    """Return a mask for points within an elliptical region on the sky.

    Parameters
    ----------
    racen, deccen : float
        Center of the ellipse [degrees].
    semia, semib : float
        Major and minor axes [degrees].
    phi : float
        Position angle of major axis [radians, East of North].
    ras, decs : array_like
        Sky coordinates of the points to test [degrees].

    Returns
    -------
    mask : ndarray of bool
        True for points inside the ellipse.

    """
    # Wrap delta-RA into [-180, +180] range
    dra = (ras - racen + 180) % 360 - 180
    dra *= np.cos(np.radians(deccen))  # account for convergence of RA near poles

    ddec = decs - deccen

    # Rotate into ellipse-aligned coordinates
    xp = dra * np.cos(phi) + ddec * np.sin(phi)
    yp = -dra * np.sin(phi) + ddec * np.cos(phi)

    # Elliptical mask condition
    return (xp / semia)**2 + (yp / semib)**2 <= 1


def ellipse_mask(xcen, ycen, semia, semib, phi, x, y):
    """Simple elliptical mask."""
    xp = (x-xcen) * np.cos(phi) + (y-ycen) * np.sin(phi)
    yp = -(x-xcen) * np.sin(phi) + (y-ycen) * np.cos(phi)
    return (xp / semia)**2 + (yp/semib)**2 <= 1


def get_tractor_ellipse(r50, e1, e2):
    """Convert Tractor epsilon1, epsilon2 values to ellipticity and position angle.

    Taken in part from tractor.ellipses.EllipseE.

    r50 in arcsec

    """
    e = np.hypot(e1, e2)
    ba = (1. - e) / (1. + e)
    #e = (ba + 1.) / (ba - 1.)

    phi = -np.rad2deg(np.arctan2(e2, e1) / 2.)
    #angle = np.deg2rad(-2 * phi)
    #e1 = e * np.cos(angle)
    #e2 = e * np.sin(angle)

    pa = (180. - phi) % 180
    diam = r50 * 2. * 1.2 # [radius-->diameter then 20% higher]

    return diam, ba, pa


def get_basic_geometry(cat, galaxy_column='OBJNAME', verbose=False):
    """From a catalog containing magnitudes, diameters, position angles, and
    ellipticities, return a "basic" value for each property.

    Priority order: RC3, TWOMASS, SDSS, ESO, NED/BASIC

    """
    from astropy.table import Table
    nobj = len(cat)

    basic = Table()
    basic['GALAXY'] = cat[galaxy_column].value

    # default
    magcol = 'MAG_LIT'
    diamcol = 'DIAM_LIT'

    # HyperLeda
    if 'LOGD25' in cat.columns:
        ref = 'HYPERLEDA'
        magcol = f'MAG_{ref}'
        diamcol = f'DIAM_{ref}'
        for prop in ('mag', 'diam', 'ba', 'pa'):
            val = np.zeros(nobj, 'f4') - 99.
            val_ref = np.zeros(nobj, '<U9')
            val_band = np.zeros(nobj, 'U1')

            if prop == 'mag':
                col = 'BT'
                band = 'B'
                I = cat[col] > 0.
                if np.sum(I) > 0:
                    val[I] = cat[col][I]
                    val_ref[I] = ref
                    val_band[I] = band
            elif prop == 'diam':
                col = 'LOGD25'
                I = cat[col] > 0.
                if np.sum(I) > 0:
                    val[I] = 0.1 * 10.**cat[col][I]
                    val_ref[I] = ref
            elif prop == 'ba':
                col = 'LOGR25'
                I = ~np.isnan(cat[col]) * (cat[col] != 0.)
                if np.sum(I) > 0:
                    val[I] = 10.**(-cat[col][I])
                    val_ref[I] = ref
            elif prop == 'pa':
                col = 'PA'
                I = ~np.isnan(cat[col])
                if np.sum(I) > 0:
                    val[I] = cat[col][I]
                    val_ref[I] = ref

            basic[f'{prop.upper()}_{ref}'] = val
            basic[f'{prop.upper()}_{ref}_REF'] = val_ref
            if prop == 'mag':
                basic[f'BAND_{ref}'] = val_band

    # SGA2020
    elif 'D26' in cat.columns:
        ref = 'SGA2020'
        magcol = f'MAG_{ref}'
        diamcol = f'DIAM_{ref}'
        for prop in ('mag', 'diam', 'ba', 'pa'):
            val = np.zeros(nobj, 'f4') - 99.
            val_ref = np.zeros(nobj, '<U9')
            val_band = np.zeros(nobj, 'U1')

            if prop == 'mag':
                col = 'R_MAG_SB26'
                band = 'R'
                I = cat[col] > 0.
                if np.sum(I) > 0:
                    val[I] = cat[col][I]
                    val_ref[I] = ref
                    val_band[I] = band
            elif prop == 'diam':
                col = 'D26'
                I = cat[col] > 0.
                if np.sum(I) > 0:
                    val[I] = cat[col][I]
                    val_ref[I] = ref
            elif prop == 'ba':
                col = 'BA'
                I = ~np.isnan(cat[col]) * (cat[col] != 0.)
                if np.sum(I) > 0:
                    val[I] = cat[col][I]
                    val_ref[I] = ref
            elif prop == 'pa':
                col = 'PA'
                I = ~np.isnan(cat[col])
                if np.sum(I) > 0:
                    val[I] = cat[col][I]
                    val_ref[I] = ref

            basic[f'{prop.upper()}_{ref}'] = val
            basic[f'{prop.upper()}_{ref}_REF'] = val_ref
            if prop == 'mag':
                basic[f'BAND_{ref}'] = val_band
    # LVD
    elif 'RHALF' in cat.columns:
        ref = 'LVD'
        for prop in ('mag', 'diam', 'ba', 'pa'):
            val = np.zeros(nobj, 'f4') - 99.
            val_ref = np.zeros(nobj, '<U9')
            val_band = np.zeros(nobj, 'U1')

            if prop == 'mag':
                col = 'APPARENT_MAGNITUDE_V'
                band = 'V'
                I = cat[col] > 0.
                if np.sum(I) > 0:
                    val[I] = cat[col][I]
                    val_ref[I] = ref
                    val_band[I] = band
            elif prop == 'diam':
                col = 'RHALF' # [arcmin]
                I = cat[col] > 0.
                if np.sum(I) > 0:
                    # see analyze-lvd
                    val[I] = cat[col][I] * 1.2 * 2. # half-light-->full-light; radius-->diameter
                    val_ref[I] = ref
            elif prop == 'ba':
                col = 'ELLIPTICITY' # =1-b/a
                I = ~np.isnan(cat[col])
                if np.sum(I) > 0:
                    val[I] = 1. - cat[col][I]
                    val_ref[I] = ref
            elif prop == 'pa':
                col = 'POSITION_ANGLE'
                I = ~np.isnan(cat[col])
                if np.sum(I) > 0:
                    val[I] = cat[col][I] % 180 # put in the range [0, 180]
                    val_ref[I] = ref

            basic[f'{prop.upper()}_LIT'] = val
            basic[f'{prop.upper()}_LIT_REF'] = val_ref
            if prop == 'mag':
                basic[f'BAND_LIT'] = val_band

    # custom
    elif 'DIAM' in cat.columns:
        ref = 'CUSTOM'
        for prop in ('mag', 'diam', 'ba', 'pa'):
            val = np.zeros(nobj, 'f4') - 99.
            val_ref = np.zeros(nobj, '<U9')
            val_band = np.zeros(nobj, 'U1')

            if prop == 'mag':
                col = 'MAG'
                I = cat[col] > 0.
                if np.sum(I) > 0:
                    val[I] = cat[col][I]
                    val_ref[I] = ref
                    val_band[I] = cat[f'{col}_BAND'][I]
            elif prop == 'diam':
                col = 'DIAM' # [arcmin]
                I = cat[col] > 0.
                if np.sum(I) > 0:
                    val[I] = cat[col][I]
                    val_ref[I] = ref
            elif prop == 'ba':
                col = 'BA'
                I = cat[col] != -99.
                if np.sum(I) > 0:
                    val[I] = cat[col][I]
                    val_ref[I] = ref
            elif prop == 'pa':
                col = 'PA'
                I = cat[col] != -99.
                if np.sum(I) > 0:
                    val[I] = cat[col][I]
                    val_ref[I] = ref

            basic[f'{prop.upper()}_LIT'] = val
            basic[f'{prop.upper()}_LIT_REF'] = val_ref
            if prop == 'mag':
                basic[f'BAND_LIT'] = val_band
    # NED
    else:
        for prop in ('mag', 'diam', 'ba', 'pa'):
            if prop == 'mag':
                refs = ('SDSS', 'TWOMASS', 'RC3')
                bands = ('R', 'K', 'B')
            else:
                refs = ('ESO', 'SDSS', 'TWOMASS', 'RC3')
                bands = ('B', 'R', 'K', 'B')
            nref = len(refs)

            val = np.zeros(nobj, 'f4') - 99.
            val_ref = np.zeros(nobj, '<U9')
            val_band = np.zeros(nobj, 'U1')

            #allI = np.zeros((nobj, nref), bool)
            for iref, (ref, band) in enumerate(zip(refs, bands)):
                if prop == 'mag':
                    col = f'{ref}_{band}'
                else:
                    col = f'{ref}_{prop.upper()}_{band}'
                I = cat[col] > 0.
                #allI[:, iref] = I

                if np.sum(I) > 0:
                    val[I] = cat[col][I]
                    val_ref[I] = ref
                    val_band[I] = band

            basic[f'{prop.upper()}_LIT'] = val
            basic[f'{prop.upper()}_LIT_REF'] = val_ref
            if prop == 'mag':
                basic[f'BAND_LIT'] = val_band

        # supplement any missing values with the "BASIC" data
        I = (basic['MAG_LIT'] <= 0.) * (cat['BASIC_MAG'] > 0.)
        if np.any(I):
            basic['MAG_LIT'][I] = cat['BASIC_MAG'][I]
            basic['BAND_LIT'][I] = 'V'

        I = (basic['DIAM_LIT'] <= 0.) * (cat['BASIC_DMAJOR'] > 0.)
        if np.any(I):
            basic['DIAM_LIT'][I] = cat['BASIC_DMAJOR'][I]
            basic['DIAM_LIT_REF'][I] = 'BASIC'

        I = (basic['BA_LIT'] <= 0.) * (cat['BASIC_DMAJOR'] > 0.) * (cat['BASIC_DMINOR'] > 0.)
        if np.any(I):
            basic['BA_LIT'][I] = cat['BASIC_DMINOR'][I] / cat['BASIC_DMAJOR'][I]
            basic['BA_LIT_REF'][I] = 'BASIC'

    # summarize
    if verbose:
        M = basic[magcol] > 0.
        D = basic[diamcol] > 0.
        log.info(f'Derived photometry for {np.sum(M):,d}/{nobj:,d} objects and ' + \
                 f'diameters for {np.sum(D):,d}/{nobj:,d} objects.')

    return basic


def parse_geometry(cat, ref, mindiam=152*0.262):
    """Parse a specific set of elliptical geometry.

    ref - choose from among SGA2020, HYPERLEDA, RC3, LVD, SMUDGes, or LIT

    """
    nobj = len(cat)
    diam = np.zeros(nobj) - 99. # [arcsec]
    ba = np.ones(nobj)
    pa = np.zeros(nobj)
    outref = np.zeros(nobj, '<U9')

    if ref == 'SGA2020':
        I = cat['DIAM_SGA2020'] > 0.
        if np.any(I):
            diam[I] = cat['DIAM_SGA2020'][I] * 60. # [arcsec]
            ba[I] = cat['BA_SGA2020'][I]
            pa[I] = cat['PA_SGA2020'][I]
            outref[I] = ref
    elif ref == 'HYPERLEDA':
        I = cat['DIAM_HYPERLEDA'] > 0.
        if np.any(I):
            diam[I] = cat['DIAM_HYPERLEDA'][I] * 60. # [arcsec]
            ba[I] = cat['BA_HYPERLEDA'][I]
            pa[I] = cat['PA_HYPERLEDA'][I]
            outref[I] = ref
    elif ref == 'LIT':
        I = cat['DIAM_LIT'] > 0.
        if np.any(I):
            diam[I] = cat['DIAM_LIT'][I] * 60. # [arcsec]
            ba[I] = cat['BA_LIT'][I]
            pa[I] = cat['PA_LIT'][I]
            outref[I] = cat['DIAM_LIT_REF']

    #I = diam <= 0.
    #if np.any(I):
    #    diam[I] = mindiam # [arcsec]
    #    outref[I] = 'NONE'

    # clean up missing values of BA and PA
    ba[ba < 0.] = 1.
    pa[pa < 0.] = 0.

    if nobj == 1:
        return diam[0], ba[0], pa[0], outref[0]
    else:
        return diam, ba, pa, outref


def choose_geometry(cat, mindiam=152*0.262, get_mag=False):
    """Choose an object's geometry, selecting between the
    NED-assembled (literature) values (DIAM, BA, PA), values from the
    SGA2020 (DIAM_SGA2020, BA_SGA2020, PA_SGA2020), and HyperLeda's
    values (DIAM_HYPERLEDA, BA_HYPERLEDA, PA_HYPERLEDA).

    mindiam is ~40 arcsec

    Default values of BA and PA are 1.0 and 0.0.
    Default value of mag is 18.

    """
    nobj = len(cat)
    diam = np.zeros(nobj) - 99.
    ba = np.zeros(nobj) - 99.
    pa = np.zeros(nobj) - 99.
    ref = np.zeros(nobj, '<U9')

    # always prefer LVD because they were all visually determined and
    # inspected
    I = (cat['DIAM_LIT_REF'].value == 'LVD') * (diam == -99.)
    if np.any(I):
        diam[I] = cat['DIAM_LIT'][I] * 60.
        ba[I] = cat['BA_LIT'][I]
        pa[I] = cat['PA_LIT'][I]
        ref[I] = 'LVD'

    # take the largest diameter
    datarefs = np.array(['SGA2020', 'HYPERLEDA', 'LIT'])
    dataindx = np.argmax((cat['DIAM_SGA2020'].value, cat['DIAM_HYPERLEDA'].value, cat['DIAM_LIT'].value), axis=0)

    # first require all of diam, ba, pa...
    for iref, dataref in enumerate(datarefs):
        I = ((dataindx == iref) * (diam == -99.) * (ba == -99.) * (pa == -99.) * 
             (cat[f'DIAM_{dataref}'] != -99.) * (cat[f'BA_{dataref}'] != -99.) * 
             (cat[f'PA_{dataref}'] != -99.))
        if np.any(I):
            diam[I] = cat[f'DIAM_{dataref}'][I] * 60.
            ba[I] = cat[f'BA_{dataref}'][I]
            pa[I] = cat[f'PA_{dataref}'][I]
            ref[I] = datarefs[iref]
            # special-case LVD, RC3, and SMUDGes
            if dataref == 'LIT':
                J = np.where(cat[f'DIAM_{dataref}_REF'][I] == 'SMUDGes')[0]
                if len(J) > 0:
                    ref[I][J] = 'SMUDGes'
                J = np.where(cat[f'DIAM_{dataref}_REF'][I] == 'LVD')[0]
                if len(J) > 0:
                    ref[I][J] = 'LVD'
                J = np.where(cat[f'DIAM_{dataref}_REF'][I] == 'RC3')[0]
                if len(J) > 0:
                    ref[I][J] = 'RC3'

    # ...and then just diam.
    for iref, dataref in enumerate(datarefs):
        I = (dataindx == iref) * (diam == -99.) * (cat[f'DIAM_{dataref}'] != -99.)
        if np.any(I):
            diam[I] = cat[f'DIAM_{dataref}'][I] * 60.
            ba[I] = cat[f'BA_{dataref}'][I]
            pa[I] = cat[f'PA_{dataref}'][I]
            ref[I] = datarefs[iref]
            # special-case LVD, RC3, and SMUDGes
            if dataref == 'LIT':
                J = np.where(cat[f'DIAM_{dataref}_REF'][I] == 'SMUDGes')[0]
                if len(J) > 0:
                    ref[I][J] = 'SMUDGes'
                J = np.where(cat[f'DIAM_{dataref}_REF'][I] == 'LVD')[0]
                if len(J) > 0:
                    ref[I][J] = 'LVD'
                J = np.where(cat[f'DIAM_{dataref}_REF'][I] == 'RC3')[0]
                if len(J) > 0:
                    ref[I][J] = 'RC3'

    # missing diameters
    I = diam <= 0.
    if np.any(I):
        ref[I] = 'NONE'

    # set a minimum floor on the diameter
    I = diam <= mindiam
    if np.any(I):
        diam[I] = mindiam

    # clean up missing values of BA and PA
    ba[ba < 0.] = 1.
    pa[pa < 0.] = 0.

    if get_mag:
        mag = np.zeros(nobj) - 99.
        band = np.zeros(nobj, '<U1')
        for magref in ['SGA2020', 'HYPERLEDA', 'LIT']:
            I = (mag == -99.) * (cat[f'MAG_{magref}'] != -99.)
            #print(magref, np.sum(I))
            if np.any(I):
                mag[I] = cat[f'MAG_{magref}'][I]
                band[I] = cat[f'BAND_{magref}'][I]

        I = (mag == -99.)
        if np.any(I):
            mag[I] = 18.
            #band[I] = ''

    ## return scalars
    #if nobj == 1:
    #    diam = diam[0]
    #    ba = ba[0]
    #    pa = pa[0]
    #    ref = ref[0]

    if get_mag:
        return diam, ba, pa, ref, mag, band
    else:
        return diam, ba, pa, ref
