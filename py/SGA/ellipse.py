"""
SGA.ellipse
===========

Code to perform ellipse photometry.

"""
import pdb # for debuggin

import os, warnings
from time import time
import numpy as np
import astropy.modeling

from SGA.logger import log


MAXSHIFT_ARCSEC = 3.5

ELLIPSEBIT = dict(
    NOTRACTOR = 2**0,          # SGA source has no corresponding Tractor source
    BLENDED = 2**1,            # SGA center is located within the elliptical mask of another SGA source
    LARGESHIFT = 2**2,         # >MAXSHIFT_ARCSEC shift between the initial and final ellipse position
    LARGESHIFT_TRACTOR = 2**3, # >MAXSHIFT_ARCSEC shift between the Tractor and final ellipse position
)

REF_SBTHRESH = [22, 22.5, 23, 23.5, 24, 24.5, 25, 25.5, 26] # surface brightness thresholds
REF_APERTURES = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 3.0] # multiples of MAJORAXIS

# ndim>1 columns when ellipse-fitting fails; note, this list is used
# by various build_catalog functions, so change with care!
FAILCOLS = ['sma', 'intens', 'intens_err', 'eps', 'eps_err',
            'pa', 'pa_err', 'x0', 'x0_err', 'y0', 'y0_err',
            'a3', 'a3_err', 'a4', 'a4_err'] + ['ndata']
FAILDTYPES = [np.float32] * 15 + [np.int16]


def _get_r0():
    r0 = 10.0 # [arcsec]
    return r0

def cog_model(radius, mtot, m0, alpha1, alpha2):
    r0 = _get_r0()
    #return mtot - m0 * np.expm1(-alpha1*((radius / r0)**(-alpha2)))
    return mtot + m0 * np.log1p(alpha1*(radius/10.0)**(-alpha2))


def cog_dofit(sma, mag, mag_err, bounds=None):
    from scipy.optimize import curve_fit

    chisq = 1e6
    try:
        popt, _ = curve_fit(cog_model, sma, mag, sigma=mag_err,
                            bounds=bounds, max_nfev=10000)
    except RuntimeError:
        popt = None
    else:
        chisq = (((cog_model(sma, *popt) - mag) / mag_err) ** 2).sum()

    return popt, chisq


class CogModel(astropy.modeling.Fittable1DModel):
    """Class to empirically model the curve of growth.

    radius in arcsec
    r0 - constant scale factor (10)

    m(r) = mtot + mcen * (1-exp**(-alpha1*(radius/r0)**(-alpha2))

    """
    mtot = astropy.modeling.Parameter(default=20.0, bounds=(1, 30)) # integrated magnitude (r-->infty)
    m0 = astropy.modeling.Parameter(default=10.0, bounds=(1, 30)) # central magnitude (r=0)
    alpha1 = astropy.modeling.Parameter(default=0.3, bounds=(1e-3, 5)) # scale factor 1
    alpha2 = astropy.modeling.Parameter(default=0.5, bounds=(1e-3, 5)) # scale factor 2

    def __init__(self, mtot=mtot.default, m0=m0.default,
                 alpha1=alpha1.default, alpha2=alpha2.default):
        super(CogModel, self).__init__(mtot, m0, alpha1, alpha2)

        self.r0 = 10 # scale factor [arcsec]

    def evaluate(self, radius, mtot, m0, alpha1, alpha2):
        """Evaluate the COG model."""
        model = mtot + m0 * (1 - np.exp(-alpha1*(radius/self.r0)**(-alpha2)))
        return model


def _apphot_one(args):
    """Wrapper function for the multiprocessing."""
    return apphot_one(*args)


def apphot_one(img, mask, theta, x0, y0, aa, bb, pixscale, variance=False, iscircle=False):
    """Perform aperture photometry in a single elliptical annulus.

    """
    from photutils import EllipticalAperture, CircularAperture, aperture_photometry

    if iscircle:
        aperture = CircularAperture((x0, y0), aa)
    else:
        aperture = EllipticalAperture((x0, y0), aa, bb, theta)

    # Integrate the data to get the total surface brightness (in
    # nanomaggies/arcsec2) and the mask to get the fractional area.

    #area = (aperture_photometry(~mask*1, aperture, mask=mask, method='exact'))['aperture_sum'].data * pixscale**2 # [arcsec**2]
    mu_flux = (aperture_photometry(img, aperture, mask=mask, method='exact'))['aperture_sum'].data # [nanomaggies/arcsec2]
    #print(x0, y0, aa, bb, theta, mu_flux, pixscale, img.shape, mask.shape, aperture)
    if variance:
        apphot = np.sqrt(mu_flux) * pixscale**2 # [nanomaggies]
    else:
        apphot = mu_flux * pixscale**2 # [nanomaggies]

    return apphot


def ellipse_cog(bands, refellipsefit, mp=1,
                seed=1, sbthresh=REF_SBTHRESH, apertures=REF_APERTURES,
                nmonte=30):
    """Measure the curve of growth (CoG) by performing elliptical aperture
    photometry.

    maxsma in pixels
    pixscalefactor - assumed to be constant for all bandpasses!

    """
    import numpy.ma as ma
    import astropy.table
    from astropy.utils.exceptions import AstropyUserWarning
    from scipy import integrate
    from scipy.interpolate import interp1d
    from scipy.stats import sigmaclip

    rand = np.random.RandomState(seed)

    #deltaa = 1.0 # pixel spacing

    #theta, eps = refellipsefit['geometry'].pa, refellipsefit['geometry'].eps
    theta = np.radians(refellipsefit['pa_moment']-90)
    eps = refellipsefit['eps_moment']
    refband = refellipsefit['refband']
    refpixscale = data['refpixscale']

    #maxsma = refellipsefit['maxsma']

    results = {}

    # Build the SB profile and measure the radius (in arcsec) at which mu
    # crosses a few different thresholds like 25 mag/arcsec, etc.
    sbprofile = ellipse_sbprofile(refellipsefit)

    for sbcut in sbthresh:
        if sbprofile['mu_{}'.format(refband)].max() < sbcut or sbprofile['mu_{}'.format(refband)].min() > sbcut:
            print('Insufficient profile to measure the radius at {:.1f} mag/arcsec2!'.format(sbcut))
            results['sma_sb{:0g}'.format(sbcut)] = np.float32(0.0)
            results['sma_ivar_sb{:0g}'.format(sbcut)] = np.float32(0.0)
            continue

        rr = (sbprofile['sma_{}'.format(refband)] * refpixscale)**0.25 # [arcsec]
        sb = sbprofile['mu_{}'.format(refband)] - sbcut
        sberr = sbprofile['muerr_{}'.format(refband)]
        keep = np.where((sb > -1) * (sb < 1))[0]
        if len(keep) < 5:
            keep = np.where((sb > -2) * (sb < 2))[0]
            if len(keep) < 5:
                print('Insufficient profile to measure the radius at {:.1f} mag/arcsec2!'.format(sbcut))
                results['sma_sb{:0g}'.format(sbcut)] = np.float32(0.0)
                results['sma_ivar_sb{:0g}'.format(sbcut)] = np.float32(0.0)
                continue

        # Monte Carlo to get the radius
        rcut = []
        for ii in np.arange(20):
            sbfit = rand.normal(sb[keep], sberr[keep])
            coeff = np.polyfit(sbfit, rr[keep], 1)
            rcut.append((np.polyval(coeff, 0))**4)
        rcut_clipped, _, _ = sigmaclip(rcut, low=3, high=3)
        meanrcut, sigrcut = np.mean(rcut_clipped), np.std(rcut_clipped)
        #meanrcut, sigrcut = np.mean(rcut), np.std(rcut)
        #print(rcut, meanrcut, sigrcut)

        #plt.clf() ; plt.plot((rr[keep])**4, sb[keep]) ; plt.axvline(x=meanrcut) ; plt.savefig('junk.png')
        #plt.clf() ; plt.plot(rr, sb+sbcut) ; plt.axvline(x=meanrcut**0.25) ; plt.axhline(y=sbcut) ; plt.xlim(2, 2.6) ; plt.savefig('junk.png')

        #try:
        #    rcut = interp1d()(sbcut) # [arcsec]
        #except:
        #    print('Warning: extrapolating r({:0g})!'.format(sbcut))
        #    rcut = interp1d(sbprofile['mu_{}'.format(refband)], sbprofile['sma_{}'.format(refband)] * pixscale, fill_value='extrapolate')(sbcut) # [arcsec]
        if meanrcut > 0 and sigrcut > 0:
            # require a minimum S/N
            if meanrcut / sigrcut > 2:
                results['sma_sb{:0g}'.format(sbcut)] = np.float32(meanrcut) # [arcsec]
                results['sma_ivar_sb{:0g}'.format(sbcut)] = np.float32(1.0 / sigrcut**2)
            else:
                print('Dropping profile measured at radius {:.1f} mag/arcsec2 due to S/N<2'.format(sbcut))
                results['sma_sb{:0g}'.format(sbcut)] = np.float32(0.0)
                results['sma_ivar_sb{:0g}'.format(sbcut)] = np.float32(0.0)
        else:
            results['sma_sb{:0g}'.format(sbcut)] = np.float32(0.0)
            results['sma_ivar_sb{:0g}'.format(sbcut)] = np.float32(0.0)

    # aperture radii
    for iap, ap in enumerate(apertures):
        if refellipsefit['sma_moment'] > 0:
            results['sma_ap{:02d}'.format(iap+1)] = np.float32(refellipsefit['sma_moment'] * ap) # [arcsec]
        else:
            results['sma_ap{:02d}'.format(iap+1)] = np.float32(0.0)

    chi2fail = 1e8
    nparams = 4

    if eps == 0.0:
        iscircle = True
    else:
        iscircle = False

    for filt in bands:
        img = ma.getdata(data['{}_masked'.format(filt.lower())][igal]) # [nanomaggies/arcsec2]
        mask = ma.getmask(data['{}_masked'.format(filt.lower())][igal])

        # handle GALEX and WISE
        if 'filt2pixscale' in data.keys():
            pixscale = data['filt2pixscale'][filt]
            if np.isclose(pixscale, refpixscale): # avoid rounding issues
                pixscale = refpixscale
                pixscalefactor = 1.0
            else:
                pixscalefactor = refpixscale / pixscale
        else:
            pixscale = refpixscale
            pixscalefactor = 1.0

        x0 = pixscalefactor * refellipsefit['x0_moment']
        y0 = pixscalefactor * refellipsefit['y0_moment']

        #im = np.log10(img) ; im[mask] = 0 ; plt.clf() ; plt.imshow(im, origin='lower') ; plt.scatter(y0, x0, s=50, color='red') ; plt.savefig('junk.png')

        # First get the elliptical aperture photometry within the threshold
        # radii found above. Also measure aperture photometry in integer
        # multiples of sma_moment.
        smapixels, sbaplist = [], []
        for sbcut in sbthresh:
            # initialize with zeros
            results['flux_sb{:0g}_{}'.format(sbcut, filt.lower())] = np.float32(0.0)
            results['flux_ivar_sb{:0g}_{}'.format(sbcut, filt.lower())] = np.float32(0.0)
            results['fracmasked_sb{:0g}_{}'.format(sbcut, filt.lower())] = np.float32(0.0)
            _smapixels = results['sma_sb{:0g}'.format(sbcut)] / pixscale # [pixels]
            if _smapixels > 0:
                smapixels.append(_smapixels)
                sbaplist.append('sb{:0g}'.format(sbcut))

        for iap, ap in enumerate(apertures):
            # initialize with zeros
            results['flux_ap{:02d}_{}'.format(iap+1, filt.lower())] = np.float32(0.0)
            results['flux_ivar_ap{:02d}_{}'.format(iap+1, filt.lower())] = np.float32(0.0)
            results['fracmasked_ap{:02d}_{}'.format(iap+1, filt.lower())] = np.float32(0.0)
            _smapixels = results['sma_ap{:02d}'.format(iap+1)] / pixscale # [pixels]
            if _smapixels > 0:
                smapixels.append(_smapixels)
                sbaplist.append('ap{:02d}'.format(iap+1))

        if len(smapixels) > 0:
            smapixels = np.hstack(smapixels)
            sbaplist = np.hstack(sbaplist)
            smbpixels = smapixels * (1. - eps)

            with np.errstate(all='ignore'):
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', category=AstropyUserWarning)
                    cogflux = pool.map(_apphot_one, [(img, mask, theta, x0, y0, aa, bb, pixscale, False, iscircle)
                                                     for aa, bb in zip(smapixels, smbpixels)])

                    # compute the fraction of masked pixels
                    nmasked = pool.map(_apphot_one, [(np.ones_like(img), np.logical_not(mask), theta, x0, y0, aa, bb, pixscale, False, iscircle)
                                                       for aa, bb in zip(smapixels, smbpixels)])
                    npix = pool.map(_apphot_one, [(np.ones_like(img), np.zeros_like(mask), theta, x0, y0, aa, bb, pixscale, False, iscircle)
                                                  for aa, bb in zip(smapixels, smbpixels)])

                    if len(cogflux) > 0:
                        cogflux = np.hstack(cogflux)
                        npix = np.hstack(npix) * pixscale**2
                        nmasked = np.hstack(nmasked) * pixscale**2
                        fracmasked = np.zeros_like(cogflux)
                        I = np.where(npix > 0)[0]
                        if len(I) > 0:
                            fracmasked[I] = nmasked[I] / npix[I]
                    else:
                        cogflux = np.array([0.0])
                        fracmasked = np.array([0.0])

                    if '{}_var'.format(filt.lower()) in data.keys():
                        var = data['{}_var'.format(filt.lower())][igal] # [nanomaggies**2/arcsec**4]
                        cogferr = pool.map(_apphot_one, [(var, mask, theta, x0, y0, aa, bb, pixscale, True, iscircle)
                                                        for aa, bb in zip(smapixels, smbpixels)])
                        if len(cogferr) > 0:
                            cogferr = np.hstack(cogferr)
                        else:
                            cogferr = np.array([0.0])
                    else:
                        cogferr = None

            with warnings.catch_warnings():
                if cogferr is not None:
                    ok = np.where(np.isfinite(cogflux) * (cogferr > 0) * np.isfinite(cogferr))[0]
                else:
                    ok = np.where(np.isfinite(cogflux))[0]

            if len(ok) > 0:
                for label, cflux, cferr, fmask in zip(sbaplist[ok], cogflux[ok], cogferr[ok], fracmasked[ok]):
                    results['flux_{}_{}'.format(label, filt.lower())] = np.float32(cflux)
                    results['flux_ivar_{}_{}'.format(label, filt.lower())] = np.float32(1/cferr**2)
                    results['fracmasked_{}_{}'.format(label, filt.lower())] = np.float32(fmask)

        # now get the curve of growth at a wide range of regularly spaced
        # positions along the semi-major axis.

        # initialize
        results['cog_mtot_{}'.format(filt.lower())] = np.float32(0.0)
        results['cog_mtot_ivar_{}'.format(filt.lower())] = np.float32(0.0)
        results['cog_m0_{}'.format(filt.lower())] = np.float32(0.0)
        results['cog_m0_ivar_{}'.format(filt.lower())] = np.float32(0.0)
        results['cog_alpha1_{}'.format(filt.lower())] = np.float32(0.0)
        results['cog_alpha1_ivar_{}'.format(filt.lower())] = np.float32(0.0)
        results['cog_alpha2_{}'.format(filt.lower())] = np.float32(0.0)
        results['cog_alpha2_ivar_{}'.format(filt.lower())] = np.float32(0.0)

        results['cog_chi2_{}'.format(filt.lower())] = np.float32(-1.0)
        results['cog_sma50_{}'.format(filt.lower())] = np.float32(-1.0)

        results['cog_sma_{}'.format(filt.lower())] = np.float32(-1.0) # np.array([])
        results['cog_flux_{}'.format(filt.lower())] = np.float32(0.0) # np.array([])
        results['cog_flux_ivar_{}'.format(filt.lower())] = np.float32(0.0) # np.array([])

        maxsma = np.max(sbprofile['sma_{}'.format(filt.lower())])        # [pixels]
        if maxsma <= 0:
            maxsma = np.max(refellipsefit['sma_{}'.format(filt.lower())])        # [pixels]

        #sma = np.arange(deltaa_filt, maxsma * pixscalefactor, deltaa_filt)

        sma = refellipsefit['sma_{}'.format(filt.lower())] * 1.0 # [pixels]
        keep = np.where((sma > 0) * (sma <= maxsma))[0]
        #keep = np.where(sma < maxsma)[0]
        if len(keep) > 0:
            sma = sma[keep]
        else:
            continue
            #print('Too few good semi-major axis pixels!')
            #raise ValueError

        smb = sma * (1. - eps)

        #print(filt, img.shape, pixscale)
        with np.errstate(all='ignore'):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=AstropyUserWarning)
                #cogflux = [apphot_one(img, mask, theta, x0, y0, aa, bb, pixscale, False, iscircle) for aa, bb in zip(sma, smb)]
                cogflux = pool.map(_apphot_one, [(img, mask, theta, x0, y0, aa, bb, pixscale, False, iscircle)
                                                for aa, bb in zip(sma, smb)])
                if len(cogflux) > 0:
                    cogflux = np.hstack(cogflux)
                else:
                    cogflux = np.array([0.0])

                if '{}_var'.format(filt.lower()) in data.keys():
                    var = data['{}_var'.format(filt.lower())][igal] # [nanomaggies**2/arcsec**4]
                    cogferr = pool.map(_apphot_one, [(var, mask, theta, x0, y0, aa, bb, pixscale, True, iscircle)
                                                    for aa, bb in zip(sma, smb)])
                    if len(cogferr) > 0:
                        cogferr = np.hstack(cogferr)
                    else:
                        cogferr = np.array([0.0])
                else:
                    cogferr = None

        # Store the curve of growth fluxes, included negative fluxes (but check
        # that the uncertainties are positive).
        with warnings.catch_warnings():
            if cogferr is not None:
                ok = np.isfinite(cogflux) * (cogferr > 0) * np.isfinite(cogferr)
            else:
                ok = np.isfinite(cogflux)

        if np.count_nonzero(ok) > 0:
            results['cog_sma_{}'.format(filt.lower())] = np.float32(sma[ok] * pixscale) # [arcsec]
            results['cog_flux_{}'.format(filt.lower())] = np.float32(cogflux[ok])
            results['cog_flux_ivar_{}'.format(filt.lower())] = np.float32(1.0 / cogferr[ok]**2)

            #print('Modeling the curve of growth.')
            # convert to mag
            with warnings.catch_warnings():
                if cogferr is not None:
                    with np.errstate(divide='ignore'):
                        these = np.where((cogflux > 0) * np.isfinite(cogflux) * (cogferr > 0) * np.isfinite(cogferr) * (cogflux / cogferr > 1))[0]
                else:
                    these = np.where((cogflux > 0) * np.isfinite(cogflux))[0]
                    cogmagerr = np.zeros(len(cogflux))+0.1 # hack!

            if len(these) < nparams:
                print('Warning: Too few {}-band pixels to fit the curve of growth; skipping.'.format(filt))
                continue

            sma_arcsec = sma[these] * pixscale             # [arcsec]
            cogmag = 22.5 - 2.5 * np.log10(cogflux[these]) # [mag]
            if cogferr is not None:
                cogmagerr = 2.5 * cogferr[these] / cogflux[these] / np.log(10)

            bounds = ([cogmag[-1]-2.0, 0, 0, 0], np.inf)
            #bounds = ([cogmag[-1]-0.5, 2.5, 0, 0], np.inf)
            #bounds = (0, inf)

            popt, minchi2 = cog_dofit(sma_arcsec, cogmag, cogmagerr, bounds=bounds)
            if minchi2 < chi2fail and popt is not None:
                mtot, m0, alpha1, alpha2 = popt

                print('{} CoG modeling succeeded with a chi^2 minimum of {:.2f}'.format(filt, minchi2))

                results['cog_mtot_{}'.format(filt.lower())] = np.float32(mtot)
                results['cog_m0_{}'.format(filt.lower())] = np.float32(m0)
                results['cog_alpha1_{}'.format(filt.lower())] = np.float32(alpha1)
                results['cog_alpha2_{}'.format(filt.lower())] = np.float32(alpha2)
                results['cog_chi2_{}'.format(filt.lower())] = np.float32(minchi2)

                # Monte Carlo to get the variance
                if nmonte > 0:
                    monte_mtot, monte_m0, monte_alpha1, monte_alpha2 = [], [], [], []
                    for _ in np.arange(nmonte):
                        try:
                            monte_popt, monte_minchi2 = cog_dofit(sma_arcsec, rand.normal(loc=cogmag, scale=cogmagerr),
                                                                  cogmagerr, bounds=bounds)
                        except:
                            monte_popt = None
                        if monte_minchi2 < chi2fail and monte_popt is not None:
                            monte_mtot.append(monte_popt[0])
                            monte_m0.append(monte_popt[1])
                            monte_alpha1.append(monte_popt[2])
                            monte_alpha2.append(monte_popt[3])

                    if len(monte_mtot) > 2:
                        mtot_sig = np.std(monte_mtot)
                        m0_sig = np.std(monte_m0)
                        alpha1_sig = np.std(monte_alpha1)
                        alpha2_sig = np.std(monte_alpha2)

                        if mtot_sig > 0 and m0_sig > 0 and alpha1_sig > 0 and alpha2_sig > 0:
                            results['cog_mtot_ivar_{}'.format(filt.lower())] = np.float32(1/mtot_sig**2)
                            results['cog_m0_ivar_{}'.format(filt.lower())] = np.float32(1/m0_sig**2)
                            results['cog_alpha1_ivar_{}'.format(filt.lower())] = np.float32(1/alpha1_sig**2)
                            results['cog_alpha2_ivar_{}'.format(filt.lower())] = np.float32(1/alpha2_sig**2)

                # get the half-light radius (along the major axis)
                if (m0 != 0) * (alpha1 != 0.0) * (alpha2 != 0.0):
                    #half_light_sma = (- np.log(1.0 - np.log10(2.0) * 2.5 / m0) / alpha1)**(-1.0/alpha2) * _get_r0() # [arcsec]
                    with np.errstate(all='ignore'):
                        half_light_sma = ((np.expm1(np.log10(2.0)*2.5/m0)) / alpha1)**(-1.0 / alpha2) * _get_r0() # [arcsec]
                    results['cog_sma50_{}'.format(filt.lower())] = np.float32(half_light_sma)

    return results


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


def integrate_isophot_one(mimg, sig, msk, sma, theta, eps, x0, y0,
                          integrmode, sclip, nclip):
    """Integrate the ellipse profile at a single semi-major axis (in
    pixels).

    theta in radians, CCW from the x-axis
    mask - True=masked pixel

    """
    from photutils.isophote import EllipseSample, Isophote
    from photutils.isophote.sample import CentralEllipseSample
    from photutils.isophote.fitter import CentralEllipseFitter
    from photutils.aperture import EllipticalAperture, CircularAperture

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        # central pixel is a special case; see
        # https://github.com/astropy/photutils-datasets/blob/main/notebooks/isophote/isophote_example4.ipynb
        if sma == 0.:
            samp = CentralEllipseSample(mimg, sma=sma, x0=x0, y0=y0, eps=eps,
                                        position_angle=theta, sclip=sclip,
                                        nclip=nclip, integrmode=integrmode)
            samp.update(fixed_parameters=[True]*4) # x0, y0, theta, eps
            iso = CentralEllipseFitter(samp).fit()

            flux, ferr, fracmasked = 0., 0., 0.
        else:
            samp = EllipseSample(mimg, sma=sma, x0=x0, y0=y0, eps=eps,
                                 position_angle=theta, sclip=sclip,
                                 nclip=nclip, integrmode=integrmode)
            samp.update(fixed_parameters=[True]*4) # x0, y0, theta, eps
            iso = Isophote(samp, 0, True, 0)

            # aperture photometry
            ap = EllipticalAperture((x0, y0), a=sma, b=sma*(1.-eps), theta=theta)
            flux, ferr = ap.do_photometry(mimg.data, error=sig, mask=msk)
            nmasked, _ = ap.do_photometry(msk)
            fracmasked = nmasked / ap.area

        #W = img.shape[0]
        #apmask = ap.to_mask().to_image((W, W)) != 0
        #np.sum(msk[apmask]) / (W*W)

        #import matplotlib.pyplot as plt
        #fig, ax = plt.subplots()
        #ax.imshow(np.log10(img), origin='lower')
        #ap.plot(ax=ax)
        #fig.savefig('ioannis/tmp/junk.png')
        #plt.close()

    return iso, np.float32(flux), np.float32(ferr), np.float32(fracmasked)


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


def get_dt(t0):
    dt = time() - t0
    if dt > 60.:
        dt /= 60.
        unit = 'minutes'
    else:
        unit = 'seconds'
    return dt, unit


def multifit(obj, images, sigimages, masks, sma_array, bands=['g', 'r', 'i', 'z'],
             opt_wcs=None, wcs=None, opt_pixscale=0.262, pixscale=0.262, mp=1,
             allbands=None, integrmode='median', nclip=3, sclip=3,
             sbthresh=REF_SBTHRESH, apertures=REF_APERTURES, debug=False):
    """Multi-band ellipse-fitting, broadly based on--
    https://github.com/astropy/photutils-datasets/blob/master/notebooks/isophote/isophote_example4.ipynb

    See also:
    https://photutils.readthedocs.io/en/latest/user_guide/isophote.html

    """
    import multiprocessing
    from photutils.isophote import EllipseGeometry, IsophoteList
    from SGA.sky import map_bxby


    def sbprofiles_datamodel(sma, bands):
        import astropy.units as u
        from astropy.table import Table, Column
        nsma = len(sma)

        sbprofiles = Table()
        sbprofiles.add_column(Column(name='sma', unit=u.arcsec, data=sma.astype('f4')))
        for filt in bands:
            sbprofiles.add_column(Column(name=f'sb_{filt}', unit=u.nanomaggy/u.arcsec**2,
                                  data=np.zeros(nsma, 'f4')))
        for filt in bands:
            sbprofiles.add_column(Column(name=f'sb_err_{filt}', unit=u.nanomaggy/u.arcsec**2,
                                  data=np.zeros(nsma, 'f4')))
        for filt in bands:
            sbprofiles.add_column(Column(name=f'flux_{filt}', unit=u.nanomaggy,
                                  data=np.zeros(nsma, 'f4')))
        for filt in bands:
            sbprofiles.add_column(Column(name=f'flux_err_{filt}', unit=u.nanomaggy,
                                  data=np.zeros(nsma, 'f4')))
        for filt in bands:
            sbprofiles.add_column(Column(name=f'fmasked_{filt}', data=np.zeros(nsma, 'f4')))
        return sbprofiles


    def results_datamodel(obj, bands):
        import astropy.units as u
        from astropy.table import Table, Column

        # FIXME - copy everything?
        cols = obj.colnames
        results = Table(obj[cols])
        for thresh in np.array(sbthresh).astype(str):
            for filt in bands:
                results.add_column(Column(name=f'D{thresh}_{filt.upper()}',
                                          unit=u.arcsec, data=np.zeros(1, 'f4')))
        return results


    # Initialize the output table
    if allbands is None:
        allbands = bands

    results = results_datamodel(obj, allbands)
    sbprofiles = sbprofiles_datamodel(sma_array*pixscale, allbands)

    ## Initialize the object geometry. NB: (x,y) are switched in
    ## photutils and PA is measured CCW from the x-axis while PA is CCW
    ## from the y-axis!
    #cols = ['BX_MOMENT', 'BY_MOMENT', 'DIAM_MOMENT', 'BA_MOMENT', 'PA_MOMENT']
    #[opt_bx, opt_by, opt_diam_arcsec, ba, pa] = list(obj[cols].values())

    opt_bx = obj['BX_MOMENT']
    opt_by = obj['BY_MOMENT']
    ellipse_pa = np.radians(obj['PA_MOMENT'] - 90.)
    ellipse_eps = 1 - obj['BA_MOMENT']
    #opt_semia_pix = obj['DIAM_MOMENT'] / 2. / opt_pixscale # [optical pixels]

    #if debug:
    #    import matplotlib.pyplot as plt
    #    from photutils.aperture import EllipticalAperture
    #    aper = EllipticalAperture((geo.x0, geo.y0), geo.sma,
    #                              geo.sma * (1 - geo.eps),
    #                              geo.pa)
    #    plt.clf()
    #    plt.imshow(np.log10(np.sum(images, axis=0)), origin='lower')
    #    aper.plot(color='white')
    #    plt.savefig('ioannis/tmp/junk.png')
    #    plt.close()

    nbands, width, _ = images.shape

    # Measure the surface-brightness profile in each bandpass.
    debug = True
    if debug:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()

    tall = time()
    for iband, filt in enumerate(bands):
        t0 = time()

        bx, by = map_bxby(opt_bx, opt_by, from_wcs=opt_wcs, to_wcs=wcs)

        sig = sigimages[iband, :, :]
        msk = masks[iband, :, :] # True=masked
        mimg = np.ma.array(images[iband, :, :], mask=msk) # ignore masked pixels

        mpargs = [(mimg, sig, msk, onesma, ellipse_pa, ellipse_eps, bx,
                   by, integrmode, sclip, nclip) for onesma in sma_array]
        if mp > 1:
            with multiprocessing.Pool(mp) as P:
                out = P.map(_integrate_isophot_one, mpargs)
        else:
            out = [integrate_isophot_one(*mparg) for mparg in mpargs]

        out = list(zip(*out))
        isobandfit = IsophoteList(out[0])

        # curve of growth
        apflux = np.hstack(out[1]) * pixscale**2. # [nanomaggies]
        apferr = np.hstack(out[2]) * pixscale**2. # [nanomaggies]
        apfmasked = np.hstack(out[3])

        # diameters...
        # results[]

        # populate the output table
        I = np.isfinite(isobandfit.intens) * np.isfinite(isobandfit.int_err)
        if np.sum(I) > 0:
            sbprofiles[f'sb_{filt}'][I] = isobandfit.intens[I]
            sbprofiles[f'sb_err_{filt}'][I] = isobandfit.int_err[I]
        sbprofiles[f'flux_{filt}'] = apflux
        sbprofiles[f'flux_err_{filt}'] = apferr
        sbprofiles[f'fmasked_{filt}'] = apfmasked

        if debug:
            I = apflux > 0.
            mag = 22.5-2.5*np.log10(apflux[I])
            dm = 2.5*apferr[I]/apflux[I]/np.log(10.)
            ax.scatter(sma_array[I]*pixscale, mag, label=filt)

        dt, unit = get_dt(t0)
        log.debug(f'Ellipse-fitting the {filt}-band took {dt:.3f} {unit}')

    dt, unit = get_dt(tall)
    log.info(f'  Fit {"".join(bands)} in a ' + \
             f'{width}x{width} mosaic in {dt:.3f} {unit}')

    if debug:
        ax.invert_yaxis()
        ax.legend()
        fig.savefig('ioannis/tmp/junk.png')
        plt.close()

    #t0 = time()
    #cog = ellipse_cog(bands, data, results, mp=mp,
    #                  sbthresh=sbthresh, apertures=apertures)
    #results.update(cog)
    #log.info('Time = {:.3f} min'.format( (time() - t0) / 60))

    return results, sbprofiles


def build_sma_band(
    opt_sma_array,       # optical edges (pixels)
    opt_pixscale=0.262,  # optical pixel scale
    pixscale=1.5,        # target band pixel scale (e.g., 1.5 UV, 2.75 IR)
    ba=1.0,              # axis ratio b/a for area calc
    min_step_pixels=1.0, # min annulus width in *target* pixels
    min_pixels_per_annulus=150, # min target-band pixels per annulus
    a_min_tgt_px=None,   # optional start radius in target pixels (e.g., 0.5*PSF FWHM)
    a_max_tgt_px=None):  # optional stop radius in target pixels

    # 1) put optical edges in arcsec (common currency)
    a_edges_arcsec = np.asarray(opt_sma_array, float) * opt_pixscale

    # 2) candidate list for the target band (convert to target pixels)
    a_edges_tgt_px_cand = a_edges_arcsec / pixscale

    # optional clamp of start/stop
    if a_min_tgt_px is not None:
        a_edges_tgt_px_cand = a_edges_tgt_px_cand[a_edges_tgt_px_cand >= a_min_tgt_px]
    if a_max_tgt_px is not None:
        a_edges_tgt_px_cand = a_edges_tgt_px_cand[a_edges_tgt_px_cand <= a_max_tgt_px]

    if len(a_edges_tgt_px_cand) < 2:
        return a_edges_tgt_px_cand  # nothing to thin

    # 3) greedy thinning with two constraints: Δa and ΔA (area)
    out = [a_edges_tgt_px_cand[0]]
    for a_out in a_edges_tgt_px_cand[1:]:
        a_in  = out[-1]
        # width constraint
        ok_width = (a_out - a_in) >= min_step_pixels
        # area constraint: ΔA = π q (a_out^2 - a_in^2)
        deltaA = np.pi * max(ba, 1e-6) * (a_out*a_out - a_in*a_in)
        ok_area = deltaA >= min_pixels_per_annulus
        if ok_width and ok_area:
            out.append(a_out)

    # ensure we end at the last candidate if we haven’t reached it yet
    if out[-1] < a_edges_tgt_px_cand[-1]:
        # add last edge if it satisfies width OR area (relax to include outer overlap)
        a_in, a_out_last = out[-1], a_edges_tgt_px_cand[-1]
        npix_annulus = np.pi * max(ba, 1e-6) * (a_out_last*a_out_last - a_in*a_in)
        if (a_out_last - a_in) >= min_step_pixels or (npix_annulus >= min_pixels_per_annulus):
            out.append(a_out_last)

    return np.asarray(out)


def build_sma_opt(s95_pix, ba=1.0, amax_factor=2.0, amax_pix=None,
                  psf_fwhm_pix=None, inner_step_pix=1.0, frac_step=0.15,
                  min_pixels_per_annulus=150, transition_mult=1.5):
    """
    Build a semi-major-axis array for elliptical isophotes.

    Returns
    -------
    a_edges : (N,) ndarray
        Semi-major axis edges (pixels), strictly increasing.
    info : dict of ndarrays
        Per-annulus diagnostics (length N-1):
          - 'a_in', 'a_out', 'delta_a'
          - 'core_step' (bool): linear-core step?
          - 'frac_step' (bool): fractional step proposal?
          - 'area_limited' (bool): min-area constraint enlarged step?
          - 'a_transition' (scalar): core→fractional switch radius
          - 'a_stop' (scalar): stop radius

    """
    # outer limit
    a_stop = float(amax_pix) if amax_pix is not None else float(amax_factor * s95_pix)

    # starting/transition radii
    if psf_fwhm_pix is not None:
        a0 = max(0.5 * psf_fwhm_pix, 1.0)
        a_transition = max(transition_mult * psf_fwhm_pix, a0 + inner_step_pix)
    else:
        a0 = 1.0
        a_transition = 5.0

    a = float(a0)
    a_edges = [a]

    a_in_list, a_out_list, delta_list = [], [], []
    core_list, frac_list, area_list = [], [], []

    while a < a_stop:
        a_in = a
        # propose next step
        if a < a_transition:
            a_next = a + inner_step_pix
            core, frac = True, False
        else:
            a_next = a * (1.0 + frac_step)
            core, frac = False, True

        # enforce minimum pixels in annulus: ΔA = π q (a_out^2 - a_in^2)
        area_limited = False
        if min_pixels_per_annulus and min_pixels_per_annulus > 0:
            need = min_pixels_per_annulus / (np.pi * max(ba, 1e-6))
            a_needed = np.sqrt(a*a + need)
            if a_next < a_needed:
                a_next = a_needed
                area_limited = True

        if a_next <= a:
            a_next = a + max(inner_step_pix, 1e-3)
        if a_next > a_stop:
            break

        a_edges.append(a_next)
        a = a_next

        a_in_list.append(a_in)
        a_out_list.append(a_next)
        delta_list.append(a_next - a_in)
        core_list.append(core)
        frac_list.append(frac)
        area_list.append(area_limited)

    info = {
        "a_in": np.array(a_in_list),
        "a_out": np.array(a_out_list),
        "delta_a": np.array(delta_list),
        "core_step": np.array(core_list, bool),
        "frac_step": np.array(frac_list, bool),
        "area_limited": np.array(area_list, bool),
        "a_transition": a_transition,
        "a_stop": a_stop,
    }
    return np.array(a_edges), info


def qa_sma_grid():
    """Figure to show the derived sma grid.

    """
    import matplotlib.pyplot as plt

    a_edges, info = build_sma_grid(
        s95_pix=80.0, ba=0.6, psf_fwhm_pix=3.0,
        inner_step_pix=1.0, frac_step=0.15,
        min_pixels_per_annulus=200, amax_factor=2.5
    )

    a_mid = 0.5*(info["a_in"] + info["a_out"])
    da = info["delta_a"]

    fig, ax = plt.subplots()
    m_core = info["core_step"]
    m_frac = (~m_core) & (~info["area_limited"])
    m_area = info["area_limited"]

    ax.plot(a_mid[m_core], da[m_core], 'o', label='Linear core')
    ax.plot(a_mid[m_frac], da[m_frac], '^', label='Fractional step')
    ax.plot(a_mid[m_area], da[m_area], 's', label='Area-limited')

    ax.axvline(info["a_transition"], ls='--', label='Transition')
    ax.axvline(info["a_stop"], ls=':', label='Stop')
    ax.set_xlabel('Semi-major axis a (px)')
    ax.set_ylabel('Annulus width Δa (px)')
    ax.legend()
    plt.show()



def qa_ellipsefit(data, sample, results, sbprofiles, unpack_maskbits_function, MASKBITS,
                  REFIDCOLUMN, datasets=['opt', 'unwise', 'galex'], linear=False):
    """Simple QA.

    """
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    from matplotlib.cm import get_cmap
    from photutils.isophote import EllipseGeometry
    from photutils.aperture import EllipticalAperture

    from SGA.sky import map_bxby
    from SGA.qa import overplot_ellipse, get_norm, sbprofile_colors


    def kill_left_y(ax):
        ax.yaxis.set_major_locator(ticker.NullLocator())
        ax.yaxis.set_minor_locator(ticker.NullLocator())
        ax.tick_params(axis='y', which='both', left=False, labelleft=False)
        ax.spines['left'].set_visible(False)  # optional


    nsample = len(sample)
    ndataset = len(datasets)

    opt_wcs = data['opt_wcs']
    opt_pixscale = data['opt_pixscale']
    opt_bands = ''.join(data['opt_bands']) # not general

    ncol = 2 # 3
    nrow = ndataset
    inches_per_panel = 3.

    cmap = plt.cm.cividis
    cmap.set_bad('white')

    sbcolors = sbprofile_colors()

    cmap2 = get_cmap('Dark2')
    colors2 = [cmap2(i) for i in range(5)]

    for iobj, obj in enumerate(sample):

        qafile = os.path.join('/global/cfs/cdirs/desi/users/ioannis/tmp',
                              f'qa-ellipsefit-{obj["SGANAME"]}.png')

        fig, ax = plt.subplots(nrow, ncol,
                               figsize=(inches_per_panel * (1+ncol),
                                        inches_per_panel * nrow),
                               gridspec_kw={
                                   'height_ratios': [1., 1., 1.],
                                   'width_ratios': [1., 2.],
                                   #'width_ratios': [1., 2., 2.],
                                   #'wspace': 0
                               })

        # one row per dataset
        for idata, (dataset, label) in enumerate(zip(datasets, [opt_bands, 'unWISE', 'GALEX'])):

            results_obj = results[idata][iobj]
            sbprofiles_obj = sbprofiles[idata][iobj]

            images = data[f'{dataset}_images'][iobj, :, :, :]
            models = data[f'{dataset}_models'][iobj, :, :, :]
            maskbits = data[f'{dataset}_maskbits'][iobj, :, :]

            bands = data[f'{dataset}_bands']
            pixscale = data[f'{dataset}_pixscale']
            wcs = data[f'{dataset}_wcs']

            opt_bx = obj['BX_MOMENT']
            opt_by = obj['BY_MOMENT']
            ellipse_pa = np.radians(obj['PA_MOMENT'] - 90.)
            ellipse_eps = 1 - obj['BA_MOMENT']
            semia = obj['DIAM_MOMENT'] / 2. # [arcsec]

            bx, by = map_bxby(opt_bx, opt_by, from_wcs=opt_wcs, to_wcs=wcs)
            refg = EllipseGeometry(x0=bx, y0=by, eps=ellipse_eps,
                                   pa=ellipse_pa, sma=semia/pixscale) # sma in pixels
            refap = EllipticalAperture((refg.x0, refg.y0), refg.sma,
                                       refg.sma*(1. - refg.eps), refg.pa)

            # a little wasteful...
            masks = unpack_maskbits_function(data[f'{dataset}_maskbits'], bands=bands,
                                             BITS=MASKBITS[idata])
            masks = masks[iobj, :, :, :]

            wimg = np.sum(images * np.logical_not(masks), axis=0)
            wimg[wimg == 0.] = np.nan

            # col 0 - images
            xx = ax[idata, 0]
            #xx.imshow(np.flipud(jpg), origin='lower', cmap='inferno')
            xx.imshow(wimg, origin='lower', cmap=cmap, interpolation='none',
                      norm=get_norm(wimg), alpha=1.)
            xx.text(0.03, 0.97, label, transform=xx.transAxes,
                    ha='left', va='top', color='white',
                    linespacing=1.5, fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='k', alpha=0.5))
            xx.set_xlim(0, wimg.shape[0]-1)
            xx.set_ylim(0, wimg.shape[0]-1)
            xx.margins(0)
            xx.set_xticks([])
            xx.set_yticks([])

            smas = sbprofiles_obj['sma'] / pixscale # [pixels]
            for sma in smas: # sma in pixels
                if sma == 0.:
                    continue
                ap = EllipticalAperture((refg.x0, refg.y0), sma,
                                        sma*(1. - refg.eps), refg.pa)
                ap.plot(color='k', lw=1, ax=xx)
            refap.plot(color=colors2[1], lw=2, ls='--', ax=xx)

            ## col 1 - linear SB profiles
            #xx = ax[idata, 1]
            #for filt in bands:
            #    xx.fill_between(sbprofiles_obj['sma']**0.25,
            #                    sbprofiles_obj[f'sb_{filt}']-sbprofiles_obj[f'sb_err_{filt}'],
            #                    sbprofiles_obj[f'sb_{filt}']+sbprofiles_obj[f'sb_err_{filt}'],
            #                    label=filt, alpha=0.6)
            #xx.set_xlim(ax[0, 1].get_xlim())
            #if idata == ndataset-1:
            #    xx.set_xlabel(r'(Semi-major axis / arcsec)$^{1/4}$')
            #else:
            #    xx.set_xticks([])
            #
            #xx.relim()
            #xx.autoscale_view()
            #
            #xx_twin = xx.twinx()
            #xx_twin.set_ylim(xx.get_ylim())
            #kill_left_y(xx)
            #
            #if idata == 1:
            #    xx_twin.set_ylabel(r'Surface Brightness (nanomaggies arcsec$^{-2}$)')
            #
            #xx.axvline(x=semia**0.25, color=colors2[1], lw=2, ls='--')

            # col 1 - mag SB profiles
            if linear:
                yminmax = [1e8, -1e8]
            else:
                yminmax = [40, 0]

            xx = ax[idata, 1]
            for filt in bands:
                I = (sbprofiles_obj[f'sb_{filt}'].value > 0.) * (sbprofiles_obj[f'sb_err_{filt}'].value > 0.)
                if np.any(I):
                    mu = 22.5 - 2.5 * np.log10(sbprofiles_obj[f'sb_{filt}'][I].value)
                    muerr = 2.5 * sbprofiles_obj[f'sb_err_{filt}'][I].value / sbprofiles_obj[f'sb_{filt}'][I].value / np.log(10.)

                    col = sbcolors[filt]
                    xx.plot(sbprofiles_obj['sma'][I].value**0.25, mu-muerr, color=col, alpha=0.8)
                    xx.plot(sbprofiles_obj['sma'][I].value**0.25, mu+muerr, color=col, alpha=0.8)
                    xx.fill_between(sbprofiles_obj['sma'][I].value**0.25, mu-muerr, mu+muerr,
                                    label=filt, color=col, alpha=0.7)

                    # robust limits
                    mulo = (mu - muerr)[mu / muerr > 10.]
                    muhi = (mu + muerr)[mu / muerr > 10.]
                    #print(filt, np.min(mulo), np.max(muhi))
                    if len(mulo) > 0:
                        mn = np.min(mulo)
                        if mn < yminmax[0]:
                            yminmax[0] = mn
                    if len(muhi) > 0:
                        mx = np.max(muhi)
                        if mx > yminmax[1]:
                            yminmax[1] = mx
                #print(filt, yminmax[0], yminmax[1])

            xx.margins(x=0)
            xx.set_xlim(ax[0, 1].get_xlim())

            if idata == ndataset-1:
                xx.set_xlabel(r'(Semi-major axis / arcsec)$^{1/4}$')
            else:
                xx.set_xticks([])

            #xx.relim()
            #xx.autoscale_view()
            if linear:
                ylim = [yminmax[0], yminmax[1]]
            else:
                ylim = [yminmax[0]-0.75, yminmax[1]+0.5]
                if ylim[0] < 13:
                    ylim[0] = 13
                if ylim[1] > 34:
                    ylim[1] = 34
            #print(idata, yminmax, ylim)
            xx.set_ylim(ylim)

            xx_twin = xx.twinx()
            xx_twin.set_ylim(ylim)
            kill_left_y(xx)

            xx.invert_yaxis()
            xx_twin.invert_yaxis()

            #y0, y1 = xx.get_ylim()
            #span_dec = abs(np.log10(y1) - np.log10(y0))
            #if span_dec < 1.:
            #    # within ~one decade: 1–2–5 per decade
            #    xx_twin.yaxis.set_major_locator(ticker.LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
            #    xx_twin.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:g}'))
            #    xx_twin.yaxis.set_minor_formatter(ticker.NullFormatter())  # no minor labels
            #else:
            #    # multiple decades: decades only
            #    xx_twin.yaxis.set_major_locator(ticker.LogLocator(base=10))
            #    xx_twin.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:g}'))
            #    xx_twin.yaxis.set_minor_formatter(ticker.NullFormatter())

            if idata == 1:
                xx_twin.set_ylabel(r'Surface Brightness (mag arcsec$^{-2}$)')

            xx.axvline(x=semia**0.25, color=colors2[1], lw=2, ls='--')

            xx.legend(loc='upper right', fontsize=8)


        fig.suptitle(f'{data["galaxy"].replace("_", " ").replace(" GROUP", " Group")}: ' + \
                     f'{obj["OBJNAME"]} ({obj[REFIDCOLUMN]})')
        #fig.suptitle(data['galaxy'].replace('_', ' ').replace(' GROUP', ' Group'))
        fig.tight_layout()
        fig.savefig(qafile, bbox_inches='tight')
        plt.close()
        log.info(f'Wrote {qafile}')


def ellipsefit_datamodel(sma, bands, dataset='opt'):
    """Initialize the ellipsefit data model.

    """
    import astropy.units as u
    from astropy.table import Table, Column

    nsma = len(sma)

    out = Table()
    out.add_column(Column(name=f'sma_{dataset}', unit=u.arcsec, data=sma.astype('f4')))
    for filt in bands:
        out.add_column(Column(name=f'sb_{filt}', unit=u.nanomaggy/u.arcsec**2,
                              data=np.zeros((1, nsma), 'f4')))
        out.add_column(Column(name=f'sb_err_{filt}', unit=u.nanomaggy/u.arcsec**2,
                              data=np.zeros((1, nsma), 'f4')))
        out.add_column(Column(name=f'flux_{filt}', unit=u.nanomaggy,
                              data=np.zeros((1, nsma), 'f4')))
        out.add_column(Column(name=f'flux_err_{filt}', unit=u.nanomaggy,
                              data=np.zeros((1, nsma), 'f4')))
        out.add_column(Column(name=f'fmasked_{filt}', data=np.zeros((1, nsma), 'f4')))

    #for band in bands:
    #    cols.append(('mw_transmission_{}'.format(band.lower()), None))
    #    cols.append(('sma_{}'.format(band.lower()), u.pixel))
    #    cols.append(('intens_{}'.format(band.lower()), 'nanomaggies arcsec-2'))#1e-9*u.maggy/u.arcsec**2))
    #    cols.append(('intens_err_{}'.format(band.lower()), 'nanomaggies arcsec-2'))#1e-9*u.maggy/u.arcsec**2))
    #    cols.append(('eps_{}'.format(band.lower()), None))
    #    cols.append(('eps_err_{}'.format(band.lower()), None))
    #    cols.append(('pa_{}'.format(band.lower()), u.degree))
    #    cols.append(('pa_err_{}'.format(band.lower()), u.degree))
    #    cols.append(('x0_{}'.format(band.lower()), u.pixel))
    #    cols.append(('x0_err_{}'.format(band.lower()), u.pixel))
    #    cols.append(('y0_{}'.format(band.lower()), u.pixel))
    #    cols.append(('y0_err_{}'.format(band.lower()), u.pixel))
    #    cols.append(('a3_{}'.format(band.lower()), None)) # units?
    #    cols.append(('a3_err_{}'.format(band.lower()), None))
    #    cols.append(('a4_{}'.format(band.lower()), None))
    #    cols.append(('a4_err_{}'.format(band.lower()), None))
    #
    #for thresh in sbthresh:
    #    cols.append(('sma_sb{:0g}'.format(thresh), u.arcsec))
    #for thresh in sbthresh:
    #    cols.append(('sma_ivar_sb{:0g}'.format(thresh), 1/u.arcsec**2))
    #for band in bands:
    #    for thresh in sbthresh:
    #        cols.append(('flux_sb{:0g}_{}'.format(thresh, band.lower()), 'nanomaggies'))#1e-9*u.maggy))
    #    for thresh in sbthresh:
    #        cols.append(('flux_ivar_sb{:0g}_{}'.format(thresh, band.lower()), 'nanomaggies-2'))#1e18/u.maggy**2))
    #    for thresh in sbthresh:
    #        cols.append(('fracmasked_sb{:0g}_{}'.format(thresh, band.lower()), None))
    #
    #for iap, ap in enumerate(apertures):
    #    cols.append(('sma_ap{:02d}'.format(iap+1), u.arcsec))
    #for band in bands:
    #    for iap, ap in enumerate(apertures):
    #        cols.append(('flux_ap{:02d}_{}'.format(iap+1, band.lower()), 'nanomaggies'))#1e-9*u.maggy))
    #    for iap, ap in enumerate(apertures):
    #        cols.append(('flux_ivar_ap{:02d}_{}'.format(iap+1, band.lower()), 'nanomaggies-2'))#1e18/u.maggy**2))
    #    for iap, ap in enumerate(apertures):
    #        cols.append(('fracmasked_ap{:02d}_{}'.format(iap+1, band.lower()), None))
    #
    #for band in bands:
    #    cols.append(('cog_sma_{}'.format(band.lower()), u.arcsec))
    #    cols.append(('cog_flux_{}'.format(band.lower()), 'nanomaggies'))#1e-9*u.maggy))
    #    cols.append(('cog_flux_ivar_{}'.format(band.lower()), 'nanomaggies-2'))#1e18/u.maggy**2))
    #
    #for band in bands:
    #    cols.append(('cog_mtot_{}'.format(band.lower()), u.mag))
    #    cols.append(('cog_mtot_ivar_{}'.format(band.lower()), 1/u.mag**2))
    #    cols.append(('cog_m0_{}'.format(band.lower()), u.mag))
    #    cols.append(('cog_m0_ivar_{}'.format(band.lower()), 1/u.mag**2))
    #    cols.append(('cog_alpha1_{}'.format(band.lower()), None))
    #    cols.append(('cog_alpha1_ivar_{}'.format(band.lower()), None))
    #    cols.append(('cog_alpha2_{}'.format(band.lower()), None))
    #    cols.append(('cog_alpha2_ivar_{}'.format(band.lower()), None))
    #    cols.append(('cog_chi2_{}'.format(band.lower()), None))
    #    cols.append(('cog_sma50_{}'.format(band.lower()), u.arcsec))

    return out


def wrap_multifit(data, sample, datasets, unpack_maskbits_function,
                  sbthresh, apertures, SGAMASKBITS, mp=1, debug=False):
    """Simple wrapper on multifit.

    Iterate on objects then datasets (even though some work is
    duplicated).

    """
    REFIDCOLUMN = data['REFIDCOLUMN']

    opt_wcs = data['opt_wcs']
    opt_pixscale = data['opt_pixscale']
    nsample = len(sample)

    results_obj = []
    sbprofiles_obj = []
    for iobj, obj in enumerate(sample):
        refid = obj[REFIDCOLUMN]

        log.info(f'Ellipse-fitting galaxy {iobj+1}/{nsample}.')

        results_dataset = []
        sbprofiles_dataset = []
        for idata, dataset in enumerate(datasets):
            bands = data[f'{dataset}_bands']
            pixscale = data[f'{dataset}_pixscale']
            wcs = data[f'{dataset}_wcs']
            images = data[f'{dataset}_images'][iobj, :, :, :]
            sigimages = data[f'{dataset}_sigma']

            # unpack the maskbits image to generate a per-band mask
            masks = unpack_maskbits_function(data[f'{dataset}_maskbits'],
                                             bands=bands, BITS=SGAMASKBITS[idata])
            masks = masks[iobj, :, :, :]

            # build the sma vector
            if dataset == 'opt':
                ba = obj['BA_MOMENT']
                semia = obj['DIAM_MOMENT'] / 2. / pixscale # [pixels]
                psf_fwhm_pix = 1.1 / pixscale              # [pixels]
                allbands = data['all_opt_bands'] # always griz in north & south

                opt_sma_array, info = build_sma_opt(
                    s95_pix=semia, ba=ba, psf_fwhm_pix=psf_fwhm_pix,
                    inner_step_pix=1., min_pixels_per_annulus=25,
                    frac_step=0.15, amax_factor=3.)
                sma_array = np.copy(opt_sma_array)
            else:
                allbands = bands
                sma_array = build_sma_band(
                    opt_sma_array, opt_pixscale=opt_pixscale,
                    pixscale=pixscale, ba=ba,
                    min_pixels_per_annulus=5) # ~constant S/N per annulus

            #print(sma_array)
            results_dataset1, sbprofiles_dataset1 = multifit(
                obj, images, sigimages, masks, sma_array, bands,
                opt_wcs=opt_wcs, wcs=wcs, opt_pixscale=opt_pixscale,
                pixscale=pixscale, mp=mp, sbthresh=sbthresh,
                apertures=apertures, debug=debug)

            results_dataset.append(results_dataset1)
            sbprofiles_dataset.append(sbprofiles_dataset1)

        results_obj.append(results_dataset)
        sbprofiles_obj.append(sbprofiles_dataset)

    # unpack the SB profiles and results tables
    results = list(zip(*results_obj))       # [ndatasets][nobj]
    sbprofiles = list(zip(*sbprofiles_obj)) # [ndatasets][nobj]

    return results, sbprofiles


def ellipsefit_multiband(galaxy, galaxydir, REFIDCOLUMN, read_multiband_function,
                         unpack_maskbits_function, SGAMASKBITS, run='south', mp=1,
                         bands=['g', 'r', 'i', 'z'], pixscale=0.262, galex_pixscale=1.5,
                         unwise_pixscale=2.75, galex=True, unwise=True,
                         sbthresh=REF_SBTHRESH, apertures=REF_APERTURES,
                         verbose=False, nowrite=False, clobber=False, qaplot=False):
    """Top-level wrapper script to do ellipse-fitting on all galaxies
    in a given group or coadd.

    """
    datasets = ['opt']
    if unwise:
        datasets += ['unwise']
    if galex:
        datasets += ['galex']

    # we need as many MASKBITS bit-masks as datasetss
    assert(len(SGAMASKBITS) == len(datasets))

    #data = read_multiband_function(
    #    galaxy, galaxydir, REFIDCOLUMN, bands=bands, run=run,
    #    pixscale=pixscale, galex_pixscale=galex_pixscale,
    #    unwise_pixscale=unwise_pixscale, unwise=unwise,
    #    galex=galex, verbose=verbose)

    data, sample, err = read_multiband_function(
        galaxy, galaxydir, REFIDCOLUMN, bands=bands, run=run,
        pixscale=pixscale, galex_pixscale=galex_pixscale,
        unwise_pixscale=unwise_pixscale, unwise=unwise,
        galex=galex, verbose=verbose, qaplot=qaplot)
    if err == 0:
        log.warning(f'Problem reading (or missing) data for {galaxydir}/{galaxy}')
        return err

    # ellipse-fit over objects and then datasets
    results, sbprofiles = wrap_multifit(
        data, sample, datasets, unpack_maskbits_function,
        sbthresh, apertures, SGAMASKBITS, mp=mp,
        debug=False)

    if qaplot:
        qa_ellipsefit(data, sample, results, sbprofiles, unpack_maskbits_function,
                      SGAMASKBITS, REFIDCOLUMN, datasets=datasets)

    if not nowrite:
        from SGA.io import write_ellipsefit
        err = write_ellipsefit(data, datasets, results, sbprofiles, verbose=verbose)

    return err
