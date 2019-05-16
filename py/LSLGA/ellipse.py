"""
LSLGA.ellipse
===================

Code to do ellipse fitting on the residual coadds.
"""
from __future__ import absolute_import, division, print_function

import os, pdb
import time, warnings
import multiprocessing

import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt

import LSLGA.io
import LSLGA.misc

from photutils.isophote import (EllipseGeometry, Ellipse, EllipseSample,
                                Isophote, IsophoteList)
from photutils.isophote.sample import CentralEllipseSample
from photutils.isophote.fitter import CentralEllipseFitter

def _apphot_one(args):
    """Wrapper function for the multiprocessing."""
    return apphot_one(*args)

def apphot_one(img, mask, theta, x0, y0, aa, bb, pixscale):
    """Perform aperture photometry in one elliptical annulus.

    """
    from photutils import EllipticalAperture, aperture_photometry

    aperture = EllipticalAperture((x0, y0), aa, bb, theta)
    # Integrate the data to get the total surface brightness (in
    # nanomaggies/arcsec2) and the mask to get the fractional area.
    
    #area = (aperture_photometry(~mask*1, aperture, mask=mask, method='exact'))['aperture_sum'].data * pixscale**2 # [arcsec**2]
    mu_flux = (aperture_photometry(img, aperture, mask=mask, method='exact'))['aperture_sum'].data # [nanomaggies/arcsec2]
    apphot = mu_flux * pixscale**2 # [nanomaggies]
    return apphot

def ellipse_apphot(band, data, ellipsefit, maxsma, filt2pixscalefactor, pool=None):
    """Perform elliptical aperture photometry for the curve-of-growth analysis.

    maxsma in pixels

    """
    import astropy.table
    from astropy.utils.exceptions import AstropyUserWarning

    deltaa = 0.5 # pixel spacing 
    theta = np.radians(ellipsefit['pa']-90)

    results = {}

    for filt in band:
        pixscale = filt2pixscalefactor['{}_pixscale'.format(filt)]
        pixscalefactor = filt2pixscalefactor[filt]

        img = ma.getdata(data['{}_masked'.format(filt)]) # [nanomaggies/arcsec2]
        #img = ma.getdata(data['{}_masked'.format(filt)]) * pixscale**2 # [nanomaggies/arcsec2-->nanomaggies]
        mask = ma.getmask(data['{}_masked'.format(filt)])

        #if filt == 'NUV':
        #    bb = img.copy()
        #    bb[mask] = 0
        #    #plt.imshow(mask*1, origin='lower') ; plt.show()
        #    plt.imshow(bb, origin='lower') ; plt.show()
        #    #plt.imshow(img, origin='lower') ; plt.show()
        #    pdb.set_trace()

        deltaa_filt = deltaa * pixscalefactor
        sma = np.arange(deltaa_filt, maxsma * pixscalefactor, deltaa_filt)
        smb = sma * ellipsefit['eps']

        x0 = ellipsefit['x0'] * pixscalefactor
        y0 = ellipsefit['y0'] * pixscalefactor

        with np.errstate(all='ignore'):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=AstropyUserWarning)
                apphot = pool.map(_apphot_one, [(img, mask, theta, x0, y0, aa, bb, pixscale)
                                                for aa, bb in zip(sma, smb)])
                apphot = np.hstack(apphot)

        results['apphot_smaunit'] = 'arcsec'
        results['apphot_sma_{}'.format(filt)] = sma * pixscale # [arcsec]
        results['apphot_mag_{}'.format(filt)] = apphot

    #fig, ax = plt.subplots()
    #for filt in band:
    #    ax.plot(results['apphot_sma_{}'.format(filt)],
    #            22.5-2.5*np.log10(results['apphot_mag_{}'.format(filt)]), label=filt)
    #ax.set_ylim(30, 5)
    #ax.legend(loc='lower right')
    #plt.show()
    #pdb.set_trace()

    return results

def _unmask_center(img):
    # https://stackoverflow.com/questions/8647024/how-to-apply-a-disc-shaped-mask-to-a-numpy-array
    nn = img.shape[0]
    x0, y0 = geometry.x0, geometry.y0
    rad = geometry.sma # [pixels]
    yy, xx = np.ogrid[-x0:nn-x0, -y0:nn-y0]
    img.mask[xx**2 + yy**2 <= rad**2] = ma.nomask
    return img

def _integrate_isophot_one(args):
    """Wrapper function for the multiprocessing."""
    return integrate_isophot_one(*args)

def integrate_isophot_one(iso, img, pixscalefactor, integrmode, sclip, nclip):
    """Integrate the ellipse profile at a single semi-major axis.

    """
    #g = iso.sample.geometry # fixed geometry
    g = copy.deepcopy(iso.sample.geometry) # fixed geometry
    
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
            g.sma *= pixscalefactor
            g.x0 *= pixscalefactor
            g.y0 *= pixscalefactor

            sample = EllipseSample(img, sma=g.sma, geometry=g, integrmode=integrmode,
                                   sclip=sclip, nclip=nclip)
            sample.update()
            #print(filt, g.sma, sample.mean)

            # Create an Isophote instance with the sample.
            out = Isophote(sample, 0, True, 0)
        
    return out

def ellipsefit_multiband(galaxy, galaxydir, data, sample, maxsma=None, nproc=1,
                         integrmode='median', nclip=2, sclip=3,
                         step=0.1, fflag=0.7, linear=False, zcolumn='Z',
                         nowrite=False, verbose=False, noellipsefit=False,
                         debug=False):
    """Ellipse-fit the multiband data.

    See
    https://github.com/astropy/photutils-datasets/blob/master/notebooks/isophote/isophote_example4.ipynb

    maxsma in (optical) pixels
    zcolumn - name of the redshift column (Z_LAMBDA in redmapper)

    """
    from legacyhalos.mge import find_galaxy

    pool = multiprocessing.Pool(nproc)
    
    # If noellipsefit=True, use the mean geometry of the galaxy to extract the
    # surface-brightness profile (turn off fitting).
    if noellipsefit:
        maxrit = -1
    else:
        maxrit = None

    band, refband, pixscale = data['band'], data['refband'], data['pixscale']
    xcen, ycen = data[refband].shape
    xcen /= 2
    ycen /= 2

    # Get the geometry of the galaxy in the reference band.
    if verbose:
        print('Finding the galaxy in the reference {}-band image.'.format(refband))

    ellipsefit = dict()

    galprops = find_galaxy(data[refband], nblob=1, fraction=0.05,
                           binning=3, plot=debug, quiet=not verbose)
    if debug:
        plt.show()
        
    galprops.centershift = False
    if np.abs(galprops.xpeak-xcen) > 5:
        galprops.xpeak = xcen
        galprops.centershift = True
    if np.abs(galprops.ypeak-ycen) > 5:
        galprops.ypeak = ycen
        galprops.centershift = True

    for key in ('eps', 'majoraxis', 'pa', 'theta', 'centershift',
                'xmed', 'ymed', 'xpeak', 'ypeak'):
        ellipsefit['mge_{}'.format(key)] = float(getattr(galprops, key))

    ellipsefit['success'] = False
    ellipsefit['redshift'] = sample[zcolumn]
    ellipsefit['band'] = band
    ellipsefit['refband'] = refband
    ellipsefit['pixscale'] = pixscale
    for filt in band: # [Gaussian sigma]
        if 'PSFSIZE_{}'.format(filt.upper()) in sample.colnames:
            psfsize = sample['PSFSIZE_{}'.format(filt.upper())]
        else:
            psfsize = 1.1 # [FWHM, arcsec]
        ellipsefit['psfsigma_{}'.format(filt)] = psfsize / np.sqrt(8 * np.log(2)) # [arcsec]
        ellipsefit['psfsigma_{}'.format(filt)] /= pixscale # [pixels]

    # Create a pixel scale mapping to accommodate GALEX and unWISE imaging.
    filt2pixscalefactor = {'g': 1.0, 'r': 1.0, 'z': 1.0, 'g_pixscale': pixscale,
                           'r_pixscale': pixscale, 'z_pixscale': pixscale}
    if 'NUV' in band:
        ellipsefit['galex_pixscale'] = data['galex_pixscale']
        factor = pixscale / data['galex_pixscale']
        filt2pixscalefactor.update({'FUV': factor, 'NUV': factor, 'FUV_pixscale': data['galex_pixscale'],
                                    'NUV_pixscale': data['galex_pixscale']})
        
    if 'W1' in band:
        ellipsefit['unwise_pixscale'] = data['unwise_pixscale']
        factor = pixscale / data['unwise_pixscale']
        filt2pixscalefactor.update({'W1': factor, 'W2': factor, 'W3': factor, 'W4': factor,
                                    'W1_pixscale': data['unwise_pixscale'], 'W2_pixscale': data['unwise_pixscale'],
                                    'W3_pixscale': data['unwise_pixscale'], 'W4_pixscale': data['unwise_pixscale']})

    if maxsma is None:
        maxsma = RADIUS_CLUSTER_KPC / legacyhalos.misc.arcsec2kpc(ellipsefit['redshift']) / pixscale # [pixels]
        # Set the maximum semi-major axis length to XX kpc or XX times the
        # semi-major axis estimated below (whichever is smaller).
        #maxsma_major = 5 * ellipsefit['majoraxis']
        #maxsma = np.min( (maxsma_cluster, maxsma_major) )

    #### ##################################################
    #print('MAXSMA HACK!!!')
    #maxsma = 10
    #nclip = 0
    #integrmode = 'bilinear'
    #### ##################################################
        
    ellipsefit['integrmode'] = integrmode
    ellipsefit['sclip'] = sclip
    ellipsefit['nclip'] = nclip
    ellipsefit['step'] = step
    ellipsefit['fflag'] = fflag
    ellipsefit['linear'] = linear

    # Get the mean geometry of the system by ellipse-fitting the inner part and
    # taking the mean values of everything.
    print('Finding the mean geometry using the reference {}-band image.'.format(refband))

    # http://photutils.readthedocs.io/en/stable/isophote_faq.html#isophote-faq
    # Note: position angle in photutils is measured counter-clockwise from the
    # x-axis, while .pa in MGE measured counter-clockwise from the y-axis.
    t0 = time.time()
    majoraxis = ellipsefit['mge_majoraxis']
    geometry0 = EllipseGeometry(x0=ellipsefit['mge_xpeak'], y0=ellipsefit['mge_ypeak'],
                                eps=ellipsefit['mge_eps'], sma=0.5*majoraxis, 
                                pa=np.radians(ellipsefit['mge_pa']-90))
    
    img = data['{}_masked'.format(refband)]
    ellipse0 = Ellipse(img, geometry=geometry0)

    smamin, smamax = 0.05*majoraxis, 1.2*majoraxis # inner, outer radius
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        factor = (1.1, 1.2, 1.3, 1.4)
        for ii, fac in enumerate(factor): # try a few different starting sma0
            sma0 = smamin*fac
            iso0 = ellipse0.fit_image(sma0, minsma=smamin, maxsma=smamax,
                                      integrmode=integrmode, sclip=sclip, nclip=nclip,
                                      step=0.5, linear=False) # note smaller step size
            if len(iso0) > 0:
                break

    if len(iso0) == 0:
        print('Initial ellipse-fitting failed!')
        return ellipsefit

    good = iso0.stop_code < 4
    ngood = np.sum(good)
    if np.sum(good) == 0:
        print('Too few good measurements to get ellipse geometry!')
        return ellipsefit

    # Fix the center to be the peak (pixel) values.
    ellipsefit['x0'] = ellipsefit['mge_xpeak']
    ellipsefit['y0'] = ellipsefit['mge_ypeak']

    ellipsefit['x0_err'] = np.std(iso0.x0) / np.sqrt(ngood)
    ellipsefit['y0_err'] = np.std(iso0.y0) / np.sqrt(ngood)
    ellipsefit['x0_median'] = np.median(iso0.x0)
    ellipsefit['y0_median'] = np.median(iso0.y0)

    #for key in ('x0', 'y0', 'eps', 'pa'):
    for key in ('eps', 'pa'):
        val = getattr(iso0, key)[good]
        ellipsefit[key] = np.median(val)
        ellipsefit['{}_err'.format(key)] = np.std(val)/np.sqrt(ngood)
        if key == 'pa':
            initval = np.degrees(geometry0.pa) + 90
            ellipsefit[key] = np.degrees(ellipsefit[key]) + 90
            ellipsefit['{}_err'.format(key)] = np.degrees(ellipsefit['{}_err'.format(key)])
        else:
            initval = getattr(geometry0, key)
        if verbose:
            initpa = np.degrees(geometry0.pa)+90
            print(' {} = {:.3f}+/-{:.3f} (initial={:.3f})'.format(
                key, ellipsefit[key], ellipsefit['{}_err'.format(key)],
                initval))
    print('Time = {:.3f} min'.format((time.time() - t0)/60))

    # Re-initialize the EllipseGeometry object.
    geometry = EllipseGeometry(x0=ellipsefit['x0'], y0=ellipsefit['y0'],
                               eps=ellipsefit['eps'], sma=majoraxis, 
                               pa=np.radians(ellipsefit['pa']-90))
    geometry_cen = EllipseGeometry(x0=ellipsefit['x0'], y0=ellipsefit['y0'], eps=0.0, sma=0.0, pa=0.0)
    ellipsefit['geometry'] = geometry
    ellipse = Ellipse(img, geometry=geometry)

    # Fit the reference bands first then the other bands.
    newmask = None
    if verbose:
        print('Ellipse-fitting the reference {}-band image.'.format(refband))

    # First fit with the default parameters.
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        print('Fitting the reference band: {}'.format(refband))
        _sma0 = (1, 3, 6, 9, 12)
        for ii, sma0 in enumerate(_sma0): # try a few different starting minor axes
            if ii > 0:
                print('Failed with sma0={:.1f} pixels, trying sma0={:.1f} pixels.'.format(_sma0[ii-1], sma0))
            try:
                isophot = ellipse.fit_image(sma0, minsma=0.1, maxsma=maxsma,
                                            integrmode=integrmode, sclip=sclip, nclip=nclip)
            except:
                isophot = []
                
            if len(isophot) > 0:
                # Add the central pixel; see
                # https://github.com/astropy/photutils-datasets/blob/master/notebooks/isophote/isophote_example4.ipynb
                #gcen = EllipseGeometry(geometry.x0, geometry.y0, 0.0, 0.0, 0.0)
                #censamp = CentralEllipseSample(img, 0.0, geometry=gcen, integrmode=integrmode, sclip=sclip, nclip=nclip)
                #print(CentralEllipseFitter(censamp).fit().intens)
                censamp = CentralEllipseSample(img, 0.0, geometry=geometry_cen,
                                               integrmode=integrmode, sclip=sclip, nclip=nclip)
                cen = CentralEllipseFitter(censamp).fit()
                isophot.append(cen)
                isophot.sort()
                break # all done!
    print('Time = {:.3f} min'.format( (time.time() - t0) / 60))

    if len(isophot) == 0:
        print('Ellipse-fitting failed.')
        return ellipsefit
    else:
        ellipsefit['success'] = True
        ellipsefit[refband] = isophot

    # Now do forced photometry at the other bandpasses (or do all the bandpasses
    # if we didn't fit above).
    tall = time.time()
    for filt in band:
        t0 = time.time()
        if filt == refband: # we did it already!
            continue

        print('Fitting band {}.'.format(filt))

        img = data['{}_masked'.format(filt)]
        if newmask is not None:
            img.mask = newmask

        pixscalefactor = filt2pixscalefactor[filt]

        # Loop on the reference band isophotes.
        pdb.set_trace()
        isobandfit = pool.map(_integrate_isophot_one, [(iso, img, pixscalefactor, integrmode, sclip, nclip)
                                                       for iso in isophot])

        # Build the IsophoteList instance with the result.
        ellipsefit[filt] = IsophoteList(isobandfit)
        print('Time = {:.3f} min'.format( (time.time() - t0) / 60))

        #if np.all( np.isnan(ellipsefit['g'].intens) ):
        #    print('ERROR: Ellipse-fitting resulted in all NaN; please check the imaging for band {}'.format(filt))
        #    ellipsefit['success'] = False

    print('Time for all images = {:.3f} min'.format( (time.time() - tall) / 60))

    # Perform elliptical aperture photometry.
    print('Performing elliptical aperture photometry.')
    t0 = time.time()
    apphot = ellipse_apphot(band, data, ellipsefit, maxsma, filt2pixscalefactor, pool=pool)
    ellipsefit.update(apphot)
    print('Time = {:.3f} min'.format( (time.time() - t0) / 60))

    # Write out
    if not nowrite:
        legacyhalos.io.write_ellipsefit(galaxy, galaxydir, ellipsefit,
                                        verbose=verbose)

    pool.close()
    
    return ellipsefit

def _ellipse_apphot(band, data, ellipsefit, maxsma, filt2pixscalefactor, warnvalue='ignore'):
    """Perform elliptical aperture photometry for the curve-of-growth analysis.

    maxsma in pixels

    """
    import astropy.table
    from astropy.utils.exceptions import AstropyUserWarning
    from photutils import EllipticalAperture, aperture_photometry

    deltaa = 0.5 # pixel spacing 
    theta = np.radians(ellipsefit['pa']-90)

    results = {}

    for filt in band:
        pixscale = filt2pixscalefactor['{}_pixscale'.format(filt)]
        pixscalefactor = filt2pixscalefactor[filt]

        img = ma.getdata(data['{}_masked'.format(filt)]) # [nanomaggies/arcsec2]
        #img = ma.getdata(data['{}_masked'.format(filt)]) * pixscale**2 # [nanomaggies/arcsec2-->nanomaggies]
        mask = ma.getmask(data['{}_masked'.format(filt)])

        #if filt == 'NUV':
        #    bb = img.copy()
        #    bb[mask] = 0
        #    #plt.imshow(mask*1, origin='lower') ; plt.show()
        #    plt.imshow(bb, origin='lower') ; plt.show()
        #    #plt.imshow(img, origin='lower') ; plt.show()
        #    pdb.set_trace()

        deltaa_filt = deltaa * pixscalefactor
        sma = np.arange(deltaa_filt, maxsma * pixscalefactor, deltaa_filt)
        smb = sma * ellipsefit['eps']

        x0 = ellipsefit['xpeak'] * pixscalefactor
        y0 = ellipsefit['ypeak'] * pixscalefactor

        apphot, apphot_nomask = [], []
        with np.errstate(all='ignore'):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=AstropyUserWarning)
                for aa, bb in zip(sma, smb):
                    aperture = EllipticalAperture((x0, y0), aa, bb, theta)
                    # Integrate the data to get the total surface brightness (in
                    # nanomaggies/arcsec2) and the mask to get the fractional
                    # area.
                    #area = (aperture_photometry(~mask*1, aperture, mask=mask, method='exact'))['aperture_sum'].data * pixscale**2 # [arcsec**2]
                    mu_flux = (aperture_photometry(img, aperture, mask=mask, method='exact'))['aperture_sum'].data # [nanomaggies/arcsec2]
                    apphot.append(mu_flux * pixscale**2) # [nanomaggies]
                apphot = np.array(apphot)
        #if filt == 'NUV':
        #    pdb.set_trace()

        results['apphot_smaunit'] = 'arcsec'
        results['apphot_sma_{}'.format(filt)] = sma * pixscale # [arcsec]
        results['apphot_mag_{}'.format(filt)] = apphot

    #fig, ax = plt.subplots()
    #for filt in band:
    #    ax.plot(results['apphot_sma_{}'.format(filt)],
    #            22.5-2.5*np.log10(results['apphot_mag_{}'.format(filt)]), label=filt)
    #ax.set_ylim(30, 5)
    #ax.legend(loc='lower right')
    #plt.show()
    #pdb.set_trace()

    return results

def _ellipsefit_multiband(galaxy, galaxydir, data, sample, maxsma=None,
                         integrmode='median', nclip=2, sclip=3,
                         step=0.1, fflag=0.7, linear=False, 
                         nowrite=False, verbose=False, noellipsefit=False,
                         debug=False):
    """Ellipse-fit the multiband data.

    See
    https://github.com/astropy/photutils-datasets/blob/master/notebooks/isophote/isophote_example4.ipynb

    maxsma in (optical) pixels

    """
    import copy
    from photutils.isophote import (EllipseGeometry, Ellipse, EllipseSample,
                                    Isophote, IsophoteList)
    from photutils.isophote.sample import CentralEllipseSample
    from photutils.isophote.fitter import CentralEllipseFitter

    from LSLGA.mge import find_galaxy

    if verbose:
        warnvalue = 'ignore' # 'always'
    else:
        warnvalue = 'ignore'

    # If noellipsefit=True, use the mean geometry of the galaxy to extract the
    # surface-brightness profile (turn off fitting).
    if noellipsefit:
        maxrit = -1
    else:
        maxrit = None

    band, refband, pixscale = data['band'], data['refband'], data['pixscale']

    # Get the geometry of the galaxy in the reference band.
    if verbose:
        print('Finding the galaxy in the reference {}-band image.'.format(refband))

    mgegalaxy = find_galaxy(data[refband], nblob=1, binning=3,
                            plot=debug, quiet=not verbose)

    #mgegalaxy.xmed -= 1
    #mgegalaxy.ymed -= 1
    #mgegalaxy.xpeak -= 1
    #mgegalaxy.ypeak -= 1
    
    # Populate the output dictionary
    ellipsefit = dict()
    for key in ('eps', 'majoraxis', 'pa', 'theta',
                'xmed', 'ymed', 'xpeak', 'ypeak'):
        ellipsefit[key] = getattr(mgegalaxy, key)
    
    ellipsefit['success'] = False
    ellipsefit['redshift'] = sample['Z']
    ellipsefit['band'] = band
    ellipsefit['refband'] = refband
    ellipsefit['pixscale'] = pixscale
    for filt in band: # [Gaussian sigma]
        ellipsefit['psfsigma_{}'.format(filt)] = ( sample['PSFSIZE_{}'.format(filt.upper())] /
                                                   np.sqrt(8 * np.log(2)) ) # [arcsec]
        ellipsefit['psfsigma_{}'.format(filt)] /= pixscale # [pixels]

    # Create a pixel scale mapping to accommodate GALEX and unWISE imaging.
    filt2pixscalefactor = {'g': 1.0, 'r': 1.0, 'i': 1.0, 'z': 1.0,
                           'g_pixscale': pixscale, 'r_pixscale': pixscale,
                           'i_pixscale': pixscale, 'z_pixscale': pixscale}
    if 'NUV' in band:
        ellipsefit['galex_pixscale'] = data['galex_pixscale']
        factor = pixscale / data['galex_pixscale']
        filt2pixscalefactor.update({'FUV': factor, 'NUV': factor, 'FUV_pixscale': data['galex_pixscale'],
                                    'NUV_pixscale': data['galex_pixscale']})
        
    if 'W1' in band:
        ellipsefit['unwise_pixscale'] = data['unwise_pixscale']
        factor = pixscale / data['unwise_pixscale']
        filt2pixscalefactor.update({'W1': factor, 'W2': factor, 'W3': factor, 'W4': factor,
                                    'W1_pixscale': data['unwise_pixscale'], 'W2_pixscale': data['unwise_pixscale'],
                                    'W3_pixscale': data['unwise_pixscale'], 'W4_pixscale': data['unwise_pixscale']})

    # Set the maximum semi-major axis length to 100 kpc or XX times the
    # semi-major axis estimated below (whichever is smaller).
    if maxsma is None:
        maxsma_100kpc = 100 / LSLGA.misc.arcsec2kpc(ellipsefit['redshift']) / pixscale # [pixels]
        maxsma_major = 4 * ellipsefit['majoraxis']
        maxsma = np.min( (maxsma_100kpc, maxsma_major) )

    ellipsefit['integrmode'] = integrmode
    ellipsefit['sclip'] = sclip
    ellipsefit['nclip'] = nclip
    ellipsefit['step'] = step
    ellipsefit['fflag'] = fflag
    ellipsefit['linear'] = linear

    # Perform elliptical aperture photometry.
    apphot = ellipse_apphot(band, data, ellipsefit, maxsma, filt2pixscalefactor)
    ellipsefit.update(apphot)

    # http://photutils.readthedocs.io/en/stable/isophote_faq.html#isophote-faq
    # Note: position angle in photutils is measured counter-clockwise from the
    # x-axis, while .pa in MGE measured counter-clockwise from the y-axis.
    geometry = EllipseGeometry(x0=ellipsefit['xpeak'], y0=ellipsefit['ypeak'],
                               eps=ellipsefit['eps'],
                               #sma=0.5*ellipsefit['majoraxis'], 
                               sma=ellipsefit['majoraxis'], 
                               #sma=10,
                               pa=np.radians(ellipsefit['pa']-90))
    ellipsefit['geometry'] = geometry

    def _unmask_center(img):
        # https://stackoverflow.com/questions/8647024/how-to-apply-a-disc-shaped-mask-to-a-numpy-array
        nn = img.shape[0]
        x0, y0 = geometry.x0, geometry.y0
        rad = geometry.sma # [pixels]
        yy, xx = np.ogrid[-x0:nn-x0, -y0:nn-y0]
        img.mask[xx**2 + yy**2 <= rad**2] = ma.nomask
        return img

    # Fit the reference bands first then the other bands.
    newmask = None
    if verbose:
        print('Ellipse-fitting the reference {}-band image.'.format(refband))

    img = data['{}_masked'.format(refband)]
    ellipse = Ellipse(img, geometry=geometry)

    # First fit with the default parameters.
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter(warnvalue)
        for sma0 in (1, 3, 6, 9, 12): # try a few different starting minor axes
            print('  Trying sma0 = {:.1f} pixels.'.format(sma0))
            try:
                isophot = ellipse.fit_image(sma0, minsma=0.1, maxsma=maxsma,
                                            integrmode=integrmode, sclip=sclip, nclip=nclip,
                                            step=step, fflag=fflag, linear=linear,
                                            maxrit=maxrit)
            except:
                isophot = []
            if len(isophot) > 0:
                break

        if len(isophot) == 0:
            print('First iteration of ellipse-fitting failed.')
            # Try unmasking the image centered on the galaxy.
            img = _unmask_center(img)
            newmask = img.mask
            ellipse = Ellipse(img, geometry=geometry)

            for sma0 in (1, 3, 6, 9, 12): # try a few different starting minor axes
                print('  Second iteration: trying sma0 = {:.1f} pixels.'.format(sma0))
                try:
                    isophot = ellipse.fit_image(sma0, minsma=1, maxsma=maxsma,
                                                integrmode=integrmode, sclip=sclip, nclip=nclip,
                                                step=step, fflag=fflag, linear=linear)
                except:
                    isophot = []
                if len(isophot) > 0:
                    break
    if verbose:
        print('Time = {:.3f} sec'.format( (time.time() - t0) / 1))

    if len(isophot) == 0:
        print('Ellipse-fitting failed, likely due to complex morphology or poor initial geometry.')
        return ellipsefit
    else:
        ellipsefit['success'] = True
        ellipsefit[refband] = isophot

    # Now do forced photometry at the other bandpasses (or do all the bandpasses
    # if we didn't fit above).
    tall = time.time()
    for filt in band:
        t0 = time.time()

        if filt == refband: # we did it already!
            continue

        if verbose:
            print('Ellipse-fitting {}-band image.'.format(filt))

        img = data['{}_masked'.format(filt)]
        if newmask is not None:
            img.mask = newmask

        pixscalefactor = filt2pixscalefactor[filt]

        # Loop on the reference band isophotes but skip the first isophote,
        # which is a CentralEllipseSample object (see below).
        isobandfit = []
        with warnings.catch_warnings():
            warnings.simplefilter(warnvalue)
            
            for iso in isophot:
            #for iso in isophot[1:]:
                g = copy.copy(iso.sample.geometry) # fixed geometry
                g.sma *= pixscalefactor
                g.x0 *= pixscalefactor
                g.y0 *= pixscalefactor

                # Use the same integration mode and clipping parameters.
                sample = EllipseSample(img, sma=g.sma, geometry=g, integrmode=integrmode,
                                       sclip=sclip, nclip=nclip)
                sample.update()
                #print(filt, g.sma, sample.mean)

                # Create an Isophote instance with the sample.
                isobandfit.append(Isophote(sample, 0, True, 0))

                # Now deal with the central pixel; see
                # https://github.com/astropy/photutils-datasets/blob/master/notebooks/isophote/isophote_example4.ipynb
                #import pdb ; pdb.set_trace()
                #g = EllipseGeometry(x0=geometry.x0, y0=geometry.y0, eps=ellipsefit['eps'], sma=1.0)
                #g.find_center(img)

                ## Use the same integration mode and clipping parameters.
                #sample = CentralEllipseSample(img, g.sma, geometry=g, integrmode=integrmode,
                #                              sclip=sclip, nclip=nclip)
                #cen = CentralEllipseFitter(sample).fit()
                #isobandfit.append(cen)
                #isobandfit.sort()

                # Build the IsophoteList instance with the result.
                ellipsefit[filt] = IsophoteList(isobandfit)

        if verbose:
            print('Time = {:.3f} sec'.format( (time.time() - t0) / 1))

        #if np.all( np.isnan(ellipsefit['g'].intens) ):
        #    print('ERROR: Ellipse-fitting resulted in all NaN; please check the imaging for band {}'.format(filt))
        #    ellipsefit['success'] = False

    if verbose:
        print('Time for all images = {:.3f} sec'.format( (time.time() - tall) / 1))

    # Write out
    if not nowrite:
        LSLGA.io.write_ellipsefit(galaxy, galaxydir, ellipsefit, verbose=verbose,
                                  noellipsefit=noellipsefit)

    return ellipsefit

def ellipse_sbprofile(ellipsefit, minerr=0.0):
    """Convert ellipse-fitting results to a magnitude, color, and surface brightness
    profiles.

    """
    band, refband = ellipsefit['band'], ellipsefit['refband']
    pixscale, redshift = ellipsefit['pixscale'], ellipsefit['redshift']

    indx = np.ones(len(ellipsefit[refband]), dtype=bool)

    sbprofile = dict()
    for filt in band:
        sbprofile['psfsigma_{}'.format(filt)] = ellipsefit['psfsigma_{}'.format(filt)]
    sbprofile['redshift'] = redshift
    
    sbprofile['minerr'] = minerr
    sbprofile['smaunit'] = 'arcsec'
    sbprofile['sma'] = ellipsefit['r'].sma[indx] * pixscale # [arcsec]

    with np.errstate(invalid='ignore'):
        for filt in band:
            #area = ellipsefit[filt].sarea[indx] * pixscale**2

            sbprofile['mu_{}'.format(filt)] = 22.5 - 2.5 * np.log10(ellipsefit[filt].intens[indx])

            #sbprofile[filt] = 22.5 - 2.5 * np.log10(ellipsefit[filt].intens[indx])
            sbprofile['mu_{}_err'.format(filt)] = 2.5 * ellipsefit[filt].int_err[indx] / \
              ellipsefit[filt].intens[indx] / np.log(10)
            sbprofile['mu_{}_err'.format(filt)] = np.sqrt(sbprofile['mu_{}_err'.format(filt)]**2 + minerr**2)

            # Just for the plot use a minimum uncertainty
            #sbprofile['{}_err'.format(filt)][sbprofile['{}_err'.format(filt)] < minerr] = minerr

    sbprofile['gr'] = sbprofile['mu_g'] - sbprofile['mu_r']
    sbprofile['rz'] = sbprofile['mu_r'] - sbprofile['mu_z']
    sbprofile['gr_err'] = np.sqrt(sbprofile['mu_g_err']**2 + sbprofile['mu_r_err']**2)
    sbprofile['rz_err'] = np.sqrt(sbprofile['mu_r_err']**2 + sbprofile['mu_z_err']**2)

    # Just for the plot use a minimum uncertainty
    #sbprofile['gr_err'][sbprofile['gr_err'] < minerr] = minerr
    #sbprofile['rz_err'][sbprofile['rz_err'] < minerr] = minerr

    # # Add the effective wavelength of each bandpass, although this needs to take
    # # into account the DECaLS vs BASS/MzLS filter curves.
    # from speclite import filters
    # filt = filters.load_filters('decam2014-g', 'decam2014-r', 'decam2014-z', 'wise2010-W1', 'wise2010-W2')
    # for ii, band in enumerate(('g', 'r', 'z', 'W1', 'W2')):
    #     sbprofile.update({'{}_wave_eff'.format(band): filt.effective_wavelengths[ii].value})

    return sbprofile

def LSLGA_ellipse(onegal, galaxy=None, galaxydir=None, pixscale=0.262, nproc=1,
                  refband='r', band=('g', 'r', 'z'), maxsma=None,
                  integrmode='median', nclip=2, sclip=3,
                  galex_pixscale=1.5, unwise_pixscale=2.75,
                  noellipsefit=False, verbose=False, debug=False):
    """Top-level wrapper script to do ellipse-fitting on a single galaxy.

    noellipsefit - do not fit for the ellipse parameters (use the mean values from MGE). 

    """
    if galaxydir is None or galaxy is None:
        galaxy, galaxydir = LSLGA.io.get_galaxy_galaxydir(onegal)

    # Read the data.
    data = LSLGA.io.read_multiband(galaxy, galaxydir, band=band,
                                   refband=refband, pixscale=pixscale,
                                   galex_pixscale=galex_pixscale,
                                   unwise_pixscale=unwise_pixscale)
    if bool(data):
        ## Find the galaxy and (optionally) perform MGE fitting.
        #mgefit = mgefit_multiband(galaxy, galaxydir, data, verbose=verbose,
        #                          noellipsefit=True, debug=debug)

        # Do ellipse-fitting.
        ellipsefit = ellipsefit_multiband(galaxy, galaxydir, data, onegal,
                                          maxsma=maxsma, nproc=nproc,
                                          integrmode=integrmode,
                                          nclip=nclip, sclip=sclip, verbose=verbose,
                                          noellipsefit=noellipsefit)
        if ellipsefit['success']:
            return 1
        else:
            return 0
        
    else:
        return 0
