"""
LSLGA.ellipse
===================

Code to do ellipse fitting on the residual coadds.
"""
from __future__ import absolute_import, division, print_function

import os, pdb
import time, warnings

import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt

import LSLGA.io
import LSLGA.misc

def ellipse_apphot(band, data, mgefit, maxsma, filt2pixscalefactor, warnvalue='ignore'):
    """Perform elliptical aperture photometry for the curve-of-growth analysis.

    maxsma in pixels

    """
    import astropy.table
    from astropy.utils.exceptions import AstropyUserWarning
    from photutils import EllipticalAperture, aperture_photometry

    deltaa = 0.5 # pixel spacing 
    theta = np.radians(mgefit['pa']-90)

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
        smb = sma * mgefit['eps']

        x0 = mgefit['xpeak'] * pixscalefactor
        y0 = mgefit['ypeak'] * pixscalefactor

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

def ellipsefit_multiband(galaxy, galaxydir, data, sample, mgefit, maxsma=None,
                         integrmode='median', nclip=2, sclip=3,
                         step=0.1, fflag=0.7, linear=False, 
                         nowrite=False, verbose=False, noellipsefit=False):
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

    # Populate the output dictionary
    ellipsefit = dict()
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

    # Set the maximum semi-major axis length to 100 kpc or XX times the
    # semi-major axis estimated below (whichever is smaller).
    if maxsma is None:
        maxsma_100kpc = 100 / LSLGA.misc.arcsec2kpc(ellipsefit['redshift']) / pixscale # [pixels]
        maxsma_major = 4 * mgefit['majoraxis']
        maxsma = np.min( (maxsma_100kpc, maxsma_major) )

    ellipsefit['integrmode'] = integrmode
    ellipsefit['sclip'] = sclip
    ellipsefit['nclip'] = nclip
    ellipsefit['step'] = step
    ellipsefit['fflag'] = fflag
    ellipsefit['linear'] = linear

    # Perform elliptical aperture photometry.
    apphot = ellipse_apphot(band, data, mgefit, maxsma, filt2pixscalefactor)
    ellipsefit.update(apphot)

    # http://photutils.readthedocs.io/en/stable/isophote_faq.html#isophote-faq
    # Note: position angle in photutils is measured counter-clockwise from the
    # x-axis, while .pa in MGE measured counter-clockwise from the y-axis.
    geometry = EllipseGeometry(x0=mgefit['xpeak'], y0=mgefit['ypeak'],
                               eps=mgefit['eps'],
                               #sma=0.5*mgefit['majoraxis'], 
                               sma=mgefit['majoraxis'], 
                               #sma=10,
                               pa=np.radians(mgefit['pa']-90))
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
                #g = EllipseGeometry(x0=geometry.x0, y0=geometry.y0, eps=mgefit['eps'], sma=1.0)
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
        LSLGA.io.write_ellipsefit(galaxy, galaxydir, ellipsefit, verbose=verbose)

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

def mgefit_multiband(galaxy, galaxydir, data, debug=False, nowrite=False,
                     noellipsefit=True, verbose=False):
    """MGE-fit the multiband data.

    See http://www-astro.physics.ox.ac.uk/~mxc/software/#mge

    """
    from mge.find_galaxy import find_galaxy
    from mge.sectors_photometry import sectors_photometry
    from mge.mge_fit_sectors import mge_fit_sectors as fit_sectors
    #from mge.mge_print_contours import mge_print_contours as print_contours

    band, refband, pixscale = data['band'], data['refband'], data['pixscale']

    # Get the geometry of the galaxy in the reference band.
    if verbose:
        print('Finding the galaxy in the reference {}-band image.'.format(refband))

    mgegalaxy = find_galaxy(data[refband], nblob=1, binning=3,
                            plot=debug, quiet=not verbose)
    if debug:
        #plt.show()
        pass
    
    #galaxy.xmed -= 1
    #galaxy.ymed -= 1
    #galaxy.xpeak -= 1
    #galaxy.ypeak -= 1
    
    mgefit = dict()
    for key in ('eps', 'majoraxis', 'pa', 'theta',
                'xmed', 'ymed', 'xpeak', 'ypeak'):
        mgefit[key] = getattr(mgegalaxy, key)

    if not noellipsefit:
        t0 = time.time()
        for filt in band:
            if verbose:
                print('Running MGE on the {}-band image.'.format(filt))

            mgephot = sectors_photometry(data[filt], mgegalaxy.eps, mgegalaxy.theta, mgegalaxy.xmed,
                                         mgegalaxy.ymed, n_sectors=11, minlevel=0, plot=debug,
                                         mask=data['{}_mask'.format(filt)])
            if debug:
                #plt.show()
                pass

            mgefit[filt] = fit_sectors(mgephot.radius, mgephot.angle, mgephot.counts,
                                       mgegalaxy.eps, ngauss=None, negative=False,
                                       sigmaPSF=0, normPSF=1, scale=pixscale,
                                       quiet=not debug, outer_slope=4, bulge_disk=False,
                                       plot=debug)
            if debug:
                pass
                #plt.show()

            #_ = print_contours(data[refband], mgegalaxy.pa, mgegalaxy.xpeak, mgegalaxy.ypeak, pp.sol, 
            #                   binning=2, normpsf=1, magrange=6, mask=None, 
            #                   scale=pixscale, sigmapsf=0)

        if verbose:
            print('Time = {:.3f} sec'.format( (time.time() - t0) / 1))

    if not nowrite:
        LSLGA.io.write_mgefit(galaxy, galaxydir, mgefit, band=refband, verbose=verbose)

    return mgefit
    
def LSLGA_ellipse(onegal, galaxy=None, galaxydir=None, pixscale=0.262,
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
        # Find the galaxy and (optionally) perform MGE fitting.
        mgefit = mgefit_multiband(galaxy, galaxydir, data, verbose=verbose,
                                  noellipsefit=True, debug=debug)

        # Do ellipse-fitting.
        ellipsefit = ellipsefit_multiband(galaxy, galaxydir, data, onegal,
                                          mgefit, maxsma=maxsma, integrmode=integrmode,
                                          nclip=nclip, sclip=sclip, verbose=verbose,
                                          noellipsefit=noellipsefit)
        if ellipsefit['success']:
            return 1
        else:
            return 0
        
    else:
        return 0
