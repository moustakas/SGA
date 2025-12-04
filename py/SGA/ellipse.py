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

# legacypipe fitting modes
FITMODE = dict(
    FREEZE = 2**0,      # freeze Tractor parameters
    FIXGEO = 2**1,      # fix ellipse geometry
    RESOLVED = 2**2,    # no Tractor catalogs or ellipse-fitting
)

# SGA fitting modes
ELLIPSEMODE = dict(
    FIXGEO = 2**0,      # fix ellipse geometry
    RESOLVED = 2**1,    # no Tractor catalogs or ellipse-fitting (always implies FIXGEO)
    FORCEPSF = 2**2,    # force PSF source detection and photometry within the SGA mask;
    FORCEGAIA = 2**3,   # force Gaia source detection and photometry within the whole field
    LESSMASKING = 2**4, # subtract but do not threshold-mask Gaia stars
    MOREMASKING = 2**5, # threshold-mask extended sources even within the SGA
                        # mask (e.g., within a cluster environment)
    MOMENTPOS = 2**6,   # use the light-weighted (not Tractor) center
    TRACTORGEO = 2**7,  # use the Tractor (not light-weighted) geometry
    RADWEIGHT = 2**8,   # derive the moment geometry after weighting radially as r^1.5
)

ELLIPSEBIT = dict(
    NOTRACTOR = 2**0,          # SGA source has no corresponding Tractor source
    BLENDED = 2**1,            # SGA center is located within the elliptical mask of another SGA source
    LARGESHIFT = 2**2,         # >MAXSHIFT_ARCSEC shift between the initial and final ellipse position
    LARGESHIFT_TRACTOR = 2**3, # >MAXSHIFT_ARCSEC shift between the Tractor and final ellipse position
    MAJORGAL = 2**4,           # nearby bright galaxy (>=XX% of the SGA source) subtracted
    OVERLAP = 2**5,            # any part of the initial SGA ellipse overlaps another SGA ellipse
    SATELLITE = 2**6,          # satellite of another larger galaxy
    TRACTORGEO = 2**7,         # used the Tractor (not light-weighted) geometry
    RADWEIGHT = 2**8,          # moment geometry derived using radial weighting
)

REF_SBTHRESH = [22., 23., 24., 25., 26.]     # surface brightness thresholds
REF_APERTURES = [0.5, 1., 1.25, 1.5, 2., 3.] # multiples of SMA_MOMENT


def to_float32_safe_mapping(d):
    finfo = np.finfo(np.float32)
    out = {}
    for k, v in d.items():
        v = float(v)
        if not np.isfinite(v):
            out[k] = np.float32(np.nan)
        else:
            out[k] = np.float32(np.clip(v, finfo.min, finfo.max))
    return out


def to_float32_safe_scalar(x):
    import math

    try:
        xv = float(x)
    except Exception:
        return np.nan

    if not math.isfinite(xv):
        return np.float32(np.nan)

    f32 = np.finfo(np.float32)
    fmin = float(f32.min)
    fmax = float(f32.max)

    if xv < fmin or xv > fmax:
        return np.float32(np.nan)

    return np.float32(xv)


def cog_model(radius, mtot, dmag, lnalpha1, lnalpha2, r0=10.):
    """
    Curve-of-growth model in magnitudes.

        m(r) = mtot + dmag * (1 - exp(-exp(lnalpha1) * (r/r0)^(-exp(lnalpha2))))

    Parameters
    ----------
    radius   : array-like
    mtot     : float         # asymptotic total magnitude
    dmag     : float         # positive amplitude (~ inner mag - mtot)
    lnalpha1 : float         # log(alpha1), alpha1 = exp(lnalpha1) > 0
    lnalpha2 : float         # log(alpha2), alpha2 = exp(lnalpha2) > 0
    r0       : float         # scale radius

    Returns
    -------
    model magnitudes at 'radius'.

    """
    r = np.asarray(radius, float)
    eps = np.finfo(float).tiny
    a1 = np.exp(lnalpha1)
    a2 = np.exp(lnalpha2)

    # log((r/r0)^(-a2)) = -a2 * log(r/r0)
    logx = -a2 * np.log(np.maximum(r, eps) / r0)

    # z = a1 * x, but do it in logs and clip to avoid overflow in exp
    logz = lnalpha1 + logx
    # safe range for np.exp in float64 (~[-745, 709]); we can be stricter
    logz = np.clip(logz, -100.0, 100.0)
    z = np.exp(logz)

    # 1 - exp(-z) evaluated stably
    return mtot + dmag * (-np.expm1(-z))


def fit_cog(sma_arcsec, flux, ferr=None, r0=10., p0=None, ndrop=0,
            bounds=None, robust=True, minerr=0.02, f_scale=1.,
            debug=False):
    """
    Fit (mtot, dmag, lnalpha1, lnalpha2) in:

        m(r) = mtot + dmag * (1 - exp(-exp(lnalpha1) * (r/r0)^(-exp(lnalpha2))))

    Returns
    -------
    popt : dict
        {'mtot','dmag','lnalpha1','lnalpha2'}
    perr : dict
        1σ uncertainties for the same keys (approximate; from (JᵀJ)^(-1) scaled by χ²/ndof)
    cov  : ndarray or None
        Covariance matrix for (mtot, dmag, lnalpha1, lnalpha2)
    chi2 : float
        Classical χ² = Σ[(m_model - m_data)/σ_m]^2 (unweighted if apferr is None)
    ndof : int
        Degrees of freedom = N - 4 (clamped to ≥1)

    """
    def initial_guesses(sma, mags, r0, bounds, eps=1e-6):
        # bounds order: (mtot, dmag, lnalpha1, lnalpha2)
        (mt_lb, dm_lb, lnA1_lb, lnA2_lb), (mt_ub, dm_ub, lnA1_ub, lnA2_ub) = bounds

        sma = np.asarray(sma, float)
        mags = np.asarray(mags, float)
        good = np.isfinite(sma) & np.isfinite(mags) & (sma > 0)
        a = sma[good]; y = mags[good]
        if a.size < 5:
            # absolute fallback: midpoints in a safe box
            lnalpha2 = np.clip(np.log(0.7), lnA2_lb+eps, lnA2_ub-eps)
            lnalpha1 = np.clip(lnalpha2 * 0.0, lnA1_lb+eps, lnA1_ub-eps)  # r_p=r0 → ln(r_p/r0)=0
            mtot = np.clip(np.median(y), mt_lb+eps, mt_ub-eps)
            dmag = np.clip((np.percentile(y, 80) - mtot), max(dm_lb+eps, 0.1), dm_ub-eps)
            return (mtot, dmag, lnalpha1, lnalpha2)

        # mtot, dmag from ends of the curve
        mtot0 = float(np.median(y[-min(5, y.size):]))
        dmag0 = float(np.percentile(y[:min(5, y.size)], 80) - mtot0)
        dmag0 = max(dmag0, 0.1)

        # default α2 if regression failed or was out of bounds
        lnalpha2 = np.log(0.7)  # α2 ≈ 0.7
        # pivot at geometric mean radius
        rp = float(np.exp(np.mean(np.log(a))))
        # lnalpha1 so that z(rp)=1
        lnalpha1 = np.exp(lnalpha2) * np.log(max(rp, 1e-12) / r0)

        # clip into bounds with a small margin
        mtot0   = float(np.clip(mtot0,   mt_lb+eps, mt_ub-eps))
        dmag0   = float(np.clip(dmag0,   dm_lb+eps, dm_ub-eps))
        lnalpha1 = float(np.clip(lnalpha1, lnA1_lb+eps, lnA1_ub-eps))
        lnalpha2 = float(np.clip(lnalpha2, lnA2_lb+eps, lnA2_ub-eps))
        return (mtot0, dmag0, lnalpha1, lnalpha2)


    def residuals(p):
        mt, dm, lnA1, lnA2 = p
        A2 = np.exp(lnA2)

        # z = exp(lnA1 - A2*log_rr0), computed stably
        logz = lnA1 - A2 * log_rr0
        logz = np.clip(logz, -100.0, 100.0)
        z = np.exp(logz)

        # m = mt + dm * (1 - exp(-z)) = mt + dm * (-expm1(-z))
        yhat = mt + dm * (-np.expm1(-z))

        res = yhat - mags
        if w is not None:
            res = res / w
        return res


    def jacobian(p):
        mt, dm, lnA1, lnA2 = p
        A2 = np.exp(lnA2)

        logz = lnA1 - A2 * log_rr0
        logz = np.clip(logz, -100.0, 100.0)
        z = np.exp(logz)
        emz = np.exp(-z)

        invw = 1.0 if w is None else (1.0 / w)
        J = np.empty((sma.size, 4), float)
        J[:, 0] = 1.0 * invw                           # ∂m/∂mtot
        J[:, 1] = (1.0 - emz) * invw                   # ∂m/∂dmag
        J[:, 2] = (dm * z * emz) * invw                # ∂m/∂lnalpha1
        J[:, 3] = (-dm * z * A2 * log_rr0 * emz) * invw# ∂m/∂lnalpha2
        return J


    from scipy.optimize import least_squares

    nall = np.arange(len(sma_arcsec))

    # check for good values and also ignore the first N inner points,
    # where the PSF dominates (especially in GALEX and WISE)
    if ferr is not None:
        ok = (np.isfinite(sma_arcsec) & np.isfinite(flux) & (sma_arcsec > 0) &
              (flux > 0) & np.isfinite(ferr) & (ferr > 0) &
              (nall > (ndrop-1)))
    else:
        ok = (np.isfinite(sma_arcsec) & np.isfinite(flux) & (sma_arcsec > 0) &
              (flux > 0) & (nall > (ndrop-1)))

    if np.count_nonzero(ok) < 5:
        return {}, {}, None, 0., 0

    sma = sma_arcsec[ok]
    mags = 22.5 - 2.5 * np.log10(flux[ok])

    if ferr is not None:
        w = 2.5 * ferr[ok] / (flux[ok] * np.log(10.))  # mag errors
        w = np.where(w > 0, w, np.nan)
        # add a magnitude floor
        w = np.sqrt(w**2 + minerr**2)
    else:
        w = None

    eps = np.finfo(float).tiny
    log_rr0 = np.log(np.maximum(sma, eps) / r0)   # reused by residuals & jacobian


    # Bounds
    ymin, ymax = float(np.nanmin(mags)), float(np.nanmax(mags))
    if bounds is None:
        # mtot near the data; dmag positive but not absurd; lnalpha*
        # kept in a safe numeric range
        lb = (ymin - 5., 1e-6, -10., -10.)
        ub = (ymax + 5., (ymax - ymin) + 5., 10., 10.)
        bounds = (lb, ub)

    # Initial guess
    if p0 is None:
        p0 = initial_guesses(sma, mags, r0, bounds)

    res = least_squares(
        residuals, x0=np.array(p0, float), bounds=bounds,
        jac=jacobian, f_scale=f_scale, method='trf',
        loss=('soft_l1' if robust else 'linear'),
        x_scale='jac', max_nfev=200, ftol=1e-10, xtol=1e-10,
        gtol=1e-10)

    # Classical chi^2 (independent of robust loss)
    mt, dm, lnA1, lnA2 = res.x
    yhat = cog_model(sma, mt, dm, lnA1, lnA2, r0=r0)
    if w is not None:
        chi2 = float(np.sum(((yhat - mags) / w)**2))
    else:
        chi2 = float(np.sum((yhat - mags)**2))
    ndof = max(1, sma.size - 4)

    # Covariance and 1σ errors (approximate)
    try:
        J = res.jac
        s2 = (res.fun @ res.fun) / ndof
        cov = np.linalg.inv(J.T @ J) * s2
        var = np.diag(cov)
        if np.all(var > 0):
            sig = np.sqrt(var)
            perr = {'mtot': sig[0], 'dmag': sig[1], 'lnalpha1': sig[2], 'lnalpha2': sig[3]}
        else:
            #cov = None
            #perr = {'mtot': np.nan, 'dmag': np.nan, 'lnalpha1': np.nan, 'lnalpha2': np.nan}
            return {}, {}, None, 0., ndof
    except np.linalg.LinAlgError:
        #cov = None
        #perr = {'mtot': np.nan, 'dmag': np.nan, 'lnalpha1': np.nan, 'lnalpha2': np.nan}
        return {}, {}, None, 0., ndof

    popt = {'mtot': mt, 'dmag': dm, 'lnalpha1': lnA1, 'lnalpha2': lnA2}

    # convert to f4
    popt = to_float32_safe_mapping(popt)
    perr = to_float32_safe_mapping(perr)
    chi2 = to_float32_safe_scalar(chi2)

    # check for insane values
    if (any(not np.isfinite(v) for v in popt.values()) or \
        any(not np.isfinite(e) for e in perr.values())):
        return {}, {}, None, 0., ndof
    else:
        return popt, perr, cov, chi2, ndof


def radius_for_fraction(f, dmag, lnalpha1, lnalpha2, r0=10.):
    """
    Invert m(r) = mtot + dmag * (1 - exp(-exp(lnalpha1) * (r/r0)^(-exp(lnalpha2))))
    to get the semi-major axis r_f enclosing flux fraction f (0<f<1).

    Parameters
    ----------
    f        : float in (0,1)   # enclosed flux fraction (e.g., 0.5 for half-light)
    dmag     : float            # amplitude (> 0)
    lnalpha1 : float            # log(alpha1)
    lnalpha2 : float            # log(alpha2)
    r0       : float            # scale radius (same units as desired r)

    Returns
    -------
    r_f : float

    """
    if not (0.0 < f < 1.0):
        raise ValueError("f must be in (0, 1).")
    dm = -2.5 * np.log10(f)          # required mag offset from mtot
    if dmag <= dm:
        raise ValueError(f"dmag={dmag:.6g} must exceed {dm:.6g} mag to reach fraction f.")
    t = 1. - dm/dmag                # in (0, 1)
    y = -np.log(t)                  # > 0
    a1 = np.exp(lnalpha1)
    a2 = np.exp(lnalpha2)
    r = r0 * (a1 / y)**(1.0 / a2)

    return r


def radius_fraction_uncertainty(f, params, cov, r0=10., var_r0=None):
    """
    Error propagation for r_f with analytic derivatives under the new parameterization.

    Parameters
    ----------
    f      : float in (0,1)
    params : sequence (mtot, dmag, lnalpha1, lnalpha2)
    cov    : (4,4) covariance matrix for (mtot, dmag, lnalpha1, lnalpha2)
    r0     : float   # scale radius
    var_r0 : float or None   # optional variance of r0 (assumed independent)

    Returns
    -------
    r_f    : float
    sigma_r: float

    """
    mtot, dmag, lnA1, lnA2 = map(float, params)  # mtot unused, but kept for ordering

    # Compute r_f and intermediates
    dm = -2.5 * np.log10(f)
    if dmag <= dm:
        raise ValueError(f"dmag={dmag:.6g} must exceed {dm:.6g} mag to reach fraction f.")
    t = 1.0 - dm/dmag
    y = -np.log(t)
    a1 = np.exp(lnA1)
    a2 = np.exp(lnA2)
    r = r0 * (a1 / y)**(1.0 / a2)

    # log-derivative approach: dr/dp = r * d(ln r)/dp
    # ln r = ln r0 + (1/a2) * (ln a1 - ln y)
    # derivatives:
    dL_dlnA1 = 1.0 / a2
    dL_dlnA2 = -(lnA1 - np.log(y)) / a2
    # d(ln y)/ddmag = - dm / (dmag^2 * t * y)
    d_ln_y_ddmag = - dm / (dmag*dmag * t * y)
    dL_ddmag = - (1.0 / a2) * d_ln_y_ddmag   # = + dm/(a2 * dmag^2 * t * y)

    # convert to dr/dp
    dr_ddmag  = r * dL_ddmag
    dr_dlnA1  = r * dL_dlnA1
    dr_dlnA2  = r * dL_dlnA2
    dr_dr0    = r / r0

    # gradient in (mtot, dmag, lnalpha1, lnalpha2) order
    g = np.array([0.0, dr_ddmag, dr_dlnA1, dr_dlnA2], dtype=float)

    # propagate
    sigma2 = float(g @ cov @ g)
    if var_r0 is not None:
        sigma2 += (dr_dr0**2) * float(var_r0)
    sigma_r = np.sqrt(max(sigma2, 0.0))

    r = to_float32_safe_scalar(r)
    sigma_r = to_float32_safe_scalar(sigma_r)

    return r, sigma_r


def half_light_radius(params, r0=10.):
    _, dmag, lnA1, lnA2 = params
    return radius_for_fraction(0.5, dmag, lnA1, lnA2, r0=r0)


def half_light_radius_with_uncertainty(params, cov, r0=10., var_r0=None):
    return radius_fraction_uncertainty(0.5, params, cov, r0=r0, var_r0=var_r0)


def _integrate_isophot_one(args):
    """Wrapper function for the multiprocessing."""
    return integrate_isophot_one(*args)


def _boxcar(y, w):
    if w is None or w < 2:
        return y
    w = int(w)
    if w % 2 == 0:
        w += 1
    k = np.ones(w, float) / w
    # reflect at edges to avoid edge dips
    return np.convolve(y, k, mode='same')


def _outer_isophotal_radius(a, mu, mu_iso, smooth_win=None):
    """
    Single-pass outer crossing on a single (a, mu) profile.
    - Assumes a increasing.
    - Enforces non-decreasing mu with radius via running max.
    - Returns np.nan if level is outside data range.

    """
    # optional light smoothing
    mu_work = _boxcar(mu, smooth_win)

    # monotone non-decreasing envelope (magnitudes get fainter outward)
    mu_env = np.maximum.accumulate(mu_work)

    mu_min, mu_max = mu_env[0], mu_env[-1]
    if mu_iso < mu_min: # level brighter than innermost point → inside first bin
        return np.nan
    if mu_iso > mu_max: # level fainter than outermost point → beyond last bin
        return np.nan

    # piecewise-linear monotone interpolation: a(mu_iso)
    # np.interp expects ascending x; mu_env is non-decreasing
    return float(np.interp(mu_iso, mu_env, a))


def isophotal_radius_mc(
    a,                # semi-major axis array
    mu,               # surface brightness [mag/arcsec^2]
    mu_err,           # 1σ uncertainties (same shape as mu)
    mu_iso,           # target isophote (e.g., 25.0)
    nmonte=100,
    sky_sigma=None,   # optional global sky mag error (additive, per draw)
    smooth_win=3,     # odd window for gentle pre-smoothing; set None/1 to disable
    random_state=None,
    return_samples=False):

    """
    Monte Carlo isophotal radius with a monotone outer envelope.

    Returns
    -------
    out : dict with keys
        'a_iso'         : median isophotal radius [same units as a]
        'a_lo','a_hi'   : 16th/84th percentiles
        'success_rate'  : fraction of MC draws with a valid crossing
        'n_success'     : number of valid draws
        'nmonte'       : total draws
        'lower_limit'   : True if even the *nominal* profile never reaches mu_iso outward
        'upper_limit'   : True if the nominal profile is already fainter than mu_iso at innermost bin
        'samples'       : (optional) array of successful radii
    """
    rng = np.random.default_rng(random_state)

    # sanitize + sort by increasing radius
    a = np.asarray(a, float)
    mu = np.asarray(mu, float)
    mu_err = np.asarray(mu_err, float)
    m = np.isfinite(a) & np.isfinite(mu) & np.isfinite(mu_err)
    a, mu, mu_err = a[m], mu[m], mu_err[m]
    order = np.argsort(a)
    a, mu, mu_err = a[order], mu[order], mu_err[order]

    if a.size < 2:
        raise ValueError("Need at least two points in profile.")

    # Quick diagnosis on the *nominal* profile (for limit flags only)
    mu_env_nom = np.maximum.accumulate(_boxcar(mu, smooth_win))
    lower_limit = (mu_iso > mu_env_nom[-1])   # target fainter than outermost measured
    upper_limit = (mu_iso < mu_env_nom[0])    # target brighter than innermost measured

    # Monte Carlo draws
    if nmonte > 0:
        samples = []
        for _ in range(int(nmonte)):
            draw = mu + rng.normal(0.0, mu_err)
            if sky_sigma and sky_sigma > 0:
                draw = draw + rng.normal(0.0, sky_sigma)  # shared offset per draw

            a_iso = _outer_isophotal_radius(a, draw, mu_iso, smooth_win=smooth_win)
            if np.isfinite(a_iso):
                samples.append(a_iso)

        samples = np.array(samples, float)
        n_success = int(np.isfinite(samples).sum())
        success_rate = n_success / float(nmonte)
    else:
        samples = None
        a_iso = _outer_isophotal_radius(a, mu, mu_iso, smooth_win=smooth_win)
        if np.isfinite(a_iso):
            med = a_iso
            sig, lo, hi = 0., 0., 0. # no uncertainty??
            n_success = 1
            success_rate = 1.
        else:
            n_success = 0
            success_rate = 0.

    out = dict(
        nmonte=int(nmonte),
        n_success=n_success,
        success_rate=success_rate,
        lower_limit=bool(lower_limit),
        upper_limit=bool(upper_limit),
    )

    if n_success == 0:
        # No valid crossings in MC → report limits only
        out.update(a_iso=np.nan, a_iso_err=np.nan, a_lo=np.nan, a_hi=np.nan)
        if return_samples:
            out["samples"] = samples
        return out

    if nmonte > 0:
        med = np.nanmedian(samples)
        lo, hi = np.nanpercentile(samples, [25., 75.])
        #lo, hi = np.nanpercentile(samples, [16, 84])
        sig = (hi - lo) / 1.349 # robust sigma

    out.update(a_iso=float(med), a_iso_err=float(sig),
               a_lo=float(lo), a_hi=float(hi))
    if return_samples:
        out["samples"] = samples

    return out


def integrate_isophot_one(mimg, sig, msk, sma, theta, eps, x0, y0,
                          integrmode, sclip, nclip, measure_sb):
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
            if measure_sb:
                samp = CentralEllipseSample(mimg, sma=sma, x0=x0, y0=y0, eps=eps,
                                            position_angle=theta, sclip=sclip,
                                            nclip=nclip, integrmode=integrmode)
                samp.update(fixed_parameters=[True]*4) # x0, y0, theta, eps
                iso = CentralEllipseFitter(samp).fit()
            else:
                iso = None

            flux, ferr, fracmasked = 0., 0., 0.
        else:
            if measure_sb:
                samp = EllipseSample(mimg, sma=sma, x0=x0, y0=y0, eps=eps,
                                     position_angle=theta, sclip=sclip,
                                     nclip=nclip, integrmode=integrmode)
                samp.update(fixed_parameters=[True]*4) # x0, y0, theta, eps
                iso = Isophote(samp, 0, True, 0)
            else:
                iso = None

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


def sbprofiles_datamodel(sma, bands):
    import astropy.units as u
    from astropy.table import Table, Column
    nsma = len(sma)

    ubands = np.char.upper(bands)

    sbprofiles = Table()
    sbprofiles.add_column(Column(name='SMA', unit=u.arcsec, data=sma.astype('f4')))
    for filt in ubands:
        sbprofiles.add_column(Column(name=f'SB_{filt}', unit=u.nanomaggy/u.arcsec**2,
                              data=np.zeros(nsma, 'f4')))
    for filt in ubands:
        sbprofiles.add_column(Column(name=f'SB_ERR_{filt}', unit=u.nanomaggy/u.arcsec**2,
                              data=np.zeros(nsma, 'f4')))
    for filt in ubands:
        sbprofiles.add_column(Column(name=f'FLUX_{filt}', unit=u.nanomaggy,
                              data=np.zeros(nsma, 'f4')))
    for filt in ubands:
        sbprofiles.add_column(Column(name=f'FLUX_ERR_{filt}', unit=u.nanomaggy,
                              data=np.zeros(nsma, 'f4')))
    for filt in ubands:
        sbprofiles.add_column(Column(name=f'FMASKED_{filt}', data=np.ones(nsma, 'f4'))) # NB: initial value
    return sbprofiles


def results_datamodel(obj, bands, dataset, sma_apertures_arcsec, sbthresh):
    import astropy.units as u
    from astropy.table import Table, Column

    ubands = np.char.upper(bands)

    cols = ['SGAID', 'SGAGROUP']
    results = Table(obj[cols])
    #results = Table()

    # Gini coefficient within the MOMENT aperture
    for filt in ubands:
        results.add_column(Column(name=f'GINI_{filt}', data=np.zeros(1, 'f4')))

    # curve of growth model parameters
    for param, unit, dtype in zip(
            #['COG_MTOT', 'COG_M0', 'COG_ALPHA1', 'COG_ALPHA2', 'COG_CHI2', 'COG_NDOF', 'SMA50'],
            ['COG_MTOT', 'COG_DMAG', 'COG_LNALPHA1', 'COG_LNALPHA2', 'COG_CHI2', 'COG_NDOF', 'SMA50'],
            [u.mag, u.mag, None, None, None, None, u.arcsec],
            ['f4', 'f4', 'f4', 'f4', 'f4', np.int32, 'f4']):
        for filt in ubands:
            results.add_column(Column(name=f'{param}_{filt}',
                                      unit=unit, data=np.zeros(1, dtype)))
        if not ('CHI2' in param or 'NDOF' in param):
            for filt in ubands:
                results.add_column(Column(name=f'{param}_ERR_{filt}',
                                          unit=unit, data=np.zeros(1, dtype)))

    # flux within apertures that are multiples of sma_moment
    for iap, ap in enumerate(sma_apertures_arcsec):
        results.add_column(Column(name=f'SMA_AP{iap:02}',
                                  unit=u.arcsec, data=np.float32(ap)))
    for iap in range(len(sma_apertures_arcsec)):
        for filt in ubands:
            results.add_column(Column(name=f'FLUX_AP{iap:02}_{filt}',
                                      unit=u.nanomaggy, data=np.zeros(1, 'f4')))
        for filt in ubands:
            results.add_column(Column(name=f'FLUX_ERR_AP{iap:02}_{filt}',
                                      unit=u.nanomaggy, data=np.zeros(1, 'f4')))
        for filt in ubands:
            results.add_column(Column(name=f'FMASKED_AP{iap:02}_{filt}', # NB: initial value
                                      unit=None, data=np.ones(1, 'f4')))

    # optical isophotal radii
    if dataset == 'opt':
        for thresh in sbthresh:
            for filt in ubands:
                results.add_column(Column(name=f'R{thresh:.0f}_{filt}',
                                          unit=u.arcsec, data=np.zeros(1, 'f4')))
            for filt in ubands:
                results.add_column(Column(name=f'R{thresh:.0f}_ERR_{filt}',
                                          unit=u.arcsec, data=np.zeros(1, 'f4')))
            # flux within apertures based on the optical isophotal radii (deprecated)
            #for filt in ubands:
            #    results.add_column(Column(name=f'FLUX_R{thresh:.0f}_{filt}',
            #                              unit=u.nanomaggy, data=np.zeros(1, 'f4')))
            #for filt in ubands:
            #    results.add_column(Column(name=f'FLUX_ERR_R{thresh:.0f}_{filt}',
            #                              unit=u.nanomaggy, data=np.zeros(1, 'f4')))

    return results


def multifit(obj, images, sigimages, masks, sma_array, dataset='opt',
             bands=['g', 'r', 'i', 'z'], opt_wcs=None, wcs=None,
             opt_pixscale=0.262, pixscale=0.262, mp=1, nmonte=100,
             allbands=None, integrmode='median', nclip=3, sclip=3,
             seed=42, sbthresh=REF_SBTHRESH, sma_apertures_arcsec=None,
             debug=False):
    """Multi-band ellipse-fitting, broadly based on--
    https://github.com/astropy/photutils-datasets/blob/master/notebooks/isophote/isophote_example4.ipynb
    https://photutils.readthedocs.io/en/latest/user_guide/isophote.html

    sma_array in pixels
    sma_apertures_pix in pixels

    """
    import multiprocessing
    from photutils.isophote import EllipseGeometry, IsophoteList
    from photutils.aperture import EllipticalAperture
    from photutils.morphology import gini
    from SGA.util import get_dt
    from SGA.sky import map_bxby


    # Initialize the output table
    if allbands is None:
        allbands = bands

    results = results_datamodel(obj, allbands, dataset, sma_apertures_arcsec, sbthresh)
    sbprofiles = sbprofiles_datamodel(sma_array*pixscale, allbands)

    # Initialize the moment geometry.
    opt_bx = obj['BX']
    opt_by = obj['BY']
    ellipse_pa = np.radians(obj['PA_MOMENT'] - 90.)
    ellipse_eps = 1 - obj['BA_MOMENT']
    sma_moment_arcsec = obj['SMA_MOMENT'] # [arcsec]
    sma_moment_pix = sma_moment_arcsec / pixscale # [pixels]

    nbands, width, _ = images.shape
    sma_array_arcsec = sma_array * pixscale

    # Measure the surface-brightness profile in each bandpass.
    debug = False
    if debug:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()

    tall = time()
    for iband, filt in enumerate(bands):
        t0 = time()

        bx, by = map_bxby(opt_bx, opt_by, from_wcs=opt_wcs, to_wcs=wcs)

        sig = sigimages[iband, :, :]
        msk = masks[iband, :, :] # True=masked
        img = images[iband, :, :]
        mimg = np.ma.array(img, mask=msk) # ignore masked pixels

        # Gini coefficient in the MOMENT ellipse
        ap_moment = EllipticalAperture((bx, by), a=sma_moment_pix, theta=ellipse_pa,
                                       b=sma_moment_pix*(1.-ellipse_eps))
        apmask_moment = ap_moment.to_mask().to_image((width, width)) != 0. # object mask=True
        gini_mask = np.logical_or(msk, np.logical_not(apmask_moment))
        if not np.all(gini_mask): # not all pixels masked
            gin = gini(img, mask=gini_mask)
            results[f'GINI_{filt.upper()}'] = gin

        #import matplotlib.pyplot as plt
        #fig, ax = plt.subplots()
        #ax.imshow(np.log10(img*gini_mask), origin='lower')
        #ap_moment.plot(ax=ax)
        #fig.savefig('ioannis/tmp/junk.png')

        # surface-brightness profile and aperture photometry
        mpargs = [(mimg, sig, msk, onesma, ellipse_pa, ellipse_eps, bx,
                   by, integrmode, sclip, nclip, True)
                  for onesma in sma_array]
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

        I = np.isfinite(apflux) * np.isfinite(apferr) * np.isfinite(apfmasked)
        if np.any(I):
            sbprofiles[f'FLUX_{filt.upper()}'][I] = apflux[I]     # [nanomaggies]
            sbprofiles[f'FLUX_ERR_{filt.upper()}'][I] = apferr[I] # [nanomaggies]
            sbprofiles[f'FMASKED_{filt.upper()}'][I] = apfmasked[I]

        # model
        popt, perr, cov, chi2, ndof = fit_cog(
            sma_array_arcsec, apflux, apferr, r0=sma_moment_arcsec)
        results[f'COG_NDOF_{filt.upper()}'] = ndof # always store ndof

        if bool(popt):
            for key in popt.keys():
                results[f'COG_{key.upper()}_{filt.upper()}'] = popt[key]
                results[f'COG_{key.upper()}_ERR_{filt.upper()}'] = perr[key]
            results[f'COG_CHI2_{filt.upper()}'] = chi2

        # half-light radius and uncertainty
        if bool(popt) and cov is not None:
            try:
                r50, r50_err = half_light_radius_with_uncertainty(
                    (popt['mtot'], popt['dmag'], popt['lnalpha1'], popt['lnalpha2']),
                    cov, r0=sma_moment_arcsec)
                if np.isnan(r50) or np.isnan(r50_err):
                    pass
                else:
                    log.info(f"{filt}(50) = {r50:.3f} ± {r50_err:.3f}")
                    results[f'SMA50_{filt.upper()}'] = r50          # [arcsec]
                    results[f'SMA50_ERR_{filt.upper()}'] = r50_err  # [arcsec]
            except:
                pass

        # aperture photometry within the reference apertures
        mpargs = [(mimg, sig, msk, onesma, ellipse_pa, ellipse_eps, bx,
                   by, integrmode, sclip, nclip, False)
                  for onesma in sma_apertures_arcsec / pixscale]
        if mp > 1:
            with multiprocessing.Pool(mp) as P:
                refout = P.map(_integrate_isophot_one, mpargs)
        else:
            refout = [integrate_isophot_one(*mparg) for mparg in mpargs]
        refout = list(zip(*refout))

        refapflux = np.hstack(refout[1]) * pixscale**2. # [nanomaggies]
        refapferr = np.hstack(refout[2]) * pixscale**2. # [nanomaggies]
        refapfmasked = np.hstack(refout[3])

        for iap in range(len(sma_apertures_arcsec)):
            I = (np.isfinite(refapflux[iap]) * np.isfinite(refapferr[iap]) *
                 np.isfinite(refapfmasked[iap]))
            if np.any(I):
                results[f'FLUX_AP{iap:02}_{filt.upper()}'][I] = refapflux[iap][I]     # [nanomaggies]
                results[f'FLUX_ERR_AP{iap:02}_{filt.upper()}'][I] = refapferr[iap][I] # [nanomaggies]
                results[f'FMASKED_AP{iap:02}_{filt.upper()}'][I] = refapfmasked[iap][I]

        # isophotal radii
        I = np.isfinite(isobandfit.intens) * np.isfinite(isobandfit.int_err)
        if np.sum(I) > 0:
            sbprofiles[f'SB_{filt.upper()}'][I] = isobandfit.intens[I]
            sbprofiles[f'SB_ERR_{filt.upper()}'][I] = isobandfit.int_err[I]

            # isophotal radii
            if dataset == 'opt':
                I = (isobandfit.intens > 0.) * (isobandfit.int_err > 0.)
                if np.sum(I) > 2: # need at least 2 points
                    mu = 22.5 - 2.5 * np.log10(isobandfit.intens[I])
                    mu_err = 2.5 * isobandfit.int_err[I] / isobandfit.intens[I] / np.log(10.)
                    for thresh in sbthresh:
                        res = isophotal_radius_mc(
                            sma_array_arcsec[I], mu=mu, mu_err=mu_err, mu_iso=thresh,
                            nmonte=nmonte, sky_sigma=0.02, smooth_win=3, random_state=seed)
                        if res['lower_limit']:
                            if thresh == 26.:
                                log.warning(f'mu({filt}) never reaches {thresh:.0f} mag/arcsec2.')
                        else:
                            if np.isfinite(res['a_iso']) and np.isfinite(res['a_iso_err']):
                                #log.debug(f"{filt}: R{thresh:.0f} = {res['a_iso']:.2f} ± " + \
                                #          f"{res['a_iso_err']:.2f}")# [success={res['success_rate']:.2%}]")
                                results[f'R{thresh:.0f}_{filt.upper()}'] = res['a_iso']         # [arcsec]
                                results[f'R{thresh:.0f}_ERR_{filt.upper()}'] = res['a_iso_err'] # [arcsec]
                            else:
                                log.debug(f"{filt}: R{thresh:.0f} could not be measured.")

        if debug:
            I = apflux > 0.
            if np.any(I):
                mag = 22.5-2.5*np.log10(apflux[I])
                dm = 2.5*apferr[I]/apflux[I]/np.log(10.)
                ax.scatter(sma_array_arcsec[I], mag, label=filt)

                refsmas = np.hstack([results[f'SMA_AP{iap:02}'] for iap in range(len(sma_apertures_arcsec))])
                refflux = np.hstack([results[f'FLUX_AP{iap:02}_{filt.upper()}'] for iap in range(len(sma_apertures_arcsec))])
                I = (refflux > 0.)
                if np.any(I):
                    refmag = 22.5-2.5*np.log10(refflux[I])
                    ax.scatter(refsmas[I], refmag, marker='s', s=100, facecolor='none')

                if bool(popt):
                    rgrid = np.linspace(min(sma_array_arcsec), max(sma_array_arcsec), 50)
                    mfit = cog_model(rgrid, **popt, r0=sma_moment_arcsec)
                    ax.plot(rgrid, mfit, color='k', alpha=0.8)

        dt, unit = get_dt(t0)
        #log.debug(f'Ellipse-fitting the {filt}-band took {dt:.3f} {unit}')

    dt, unit = get_dt(tall)
    log.info(f'Fit {"".join(bands)} in a ' + \
             f'{width}x{width} mosaic in {dt:.3f} {unit}')

    if debug:
        ax.invert_yaxis()
        ax.legend()
        fig.savefig('ioannis/tmp/junk.png')
        plt.close()


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



def qa_ellipsefit(data, sample, results, sbprofiles, unpack_maskbits_function,
                  MASKBITS, REFIDCOLUMN, datasets=['opt', 'unwise', 'galex'],
                  linear=False, htmlgalaxydir=None):
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

        sganame = obj['SGANAME'].replace(' ', '_')
        qafile = os.path.join(htmlgalaxydir, f'qa-ellipsefit-{sganame}.png')

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

            images = data[f'{dataset}_images'][iobj, :, :, :]
            if np.all(images == 0.):
                have_data = False
            else:
                have_data = True

            results_obj = results[idata][iobj]
            sbprofiles_obj = sbprofiles[idata][iobj]

            models = data[f'{dataset}_models'][iobj, :, :, :]
            maskbits = data[f'{dataset}_maskbits'][iobj, :, :]

            bands = data[f'{dataset}_bands']
            pixscale = data[f'{dataset}_pixscale']
            wcs = data[f'{dataset}_wcs']

            opt_bx = obj['BX']
            opt_by = obj['BY']
            ellipse_pa = np.radians(obj['PA_MOMENT'] - 90.)
            ellipse_eps = 1 - obj['BA_MOMENT']
            semia = obj['SMA_MOMENT'] # [arcsec]

            if have_data:
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
                try:
                    norm = get_norm(wimg)
                except:
                    norm = None

                # col 0 - images
                xx = ax[idata, 0]
                #xx.imshow(np.flipud(jpg), origin='lower', cmap='inferno')
                xx.imshow(wimg, origin='lower', cmap=cmap, interpolation='none',
                          norm=norm, alpha=1.)
                xx.text(0.03, 0.97, label, transform=xx.transAxes,
                        ha='left', va='top', color='white',
                        linespacing=1.5, fontsize=10,
                        bbox=dict(boxstyle='round', facecolor='k', alpha=0.5))
                xx.set_xlim(0, wimg.shape[0]-1)
                xx.set_ylim(0, wimg.shape[0]-1)
                xx.margins(0)
                xx.set_xticks([])
                xx.set_yticks([])

                smas = sbprofiles_obj['SMA'] / pixscale # [pixels]
                for sma in smas: # sma in pixels
                    if sma == 0.:
                        continue
                    ap = EllipticalAperture((refg.x0, refg.y0), sma,
                                            sma*(1. - refg.eps), refg.pa)
                    ap.plot(color='k', lw=1, ax=xx)
                refap.plot(color=colors2[1], lw=2, ls='--', ax=xx)

                # col 1 - mag SB profiles
                if linear:
                    yminmax = [1e8, -1e8]
                else:
                    yminmax = [40, 0]

                xx = ax[idata, 1]
                for filt in bands:
                    I = ((sbprofiles_obj[f'SB_{filt.upper()}'].value > 0.) *
                         (sbprofiles_obj[f'SB_ERR_{filt.upper()}'].value > 0.))
                    if np.any(I):
                        sma = sbprofiles_obj['SMA'][I].value**0.25
                        sb = sbprofiles_obj[f'SB_{filt.upper()}'][I].value
                        sberr = sbprofiles_obj[f'SB_ERR_{filt.upper()}'][I].value
                        mu = 22.5 - 2.5 * np.log10(sb)
                        muerr = 2.5 * sberr / sb / np.log(10.)

                        col = sbcolors[filt]
                        xx.plot(sma, mu-muerr, color=col, alpha=0.8)
                        xx.plot(sma, mu+muerr, color=col, alpha=0.8)
                        xx.fill_between(sma, mu-muerr, mu+muerr,
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

                if idata == 1:
                    xx_twin.set_ylabel(r'Surface Brightness (mag arcsec$^{-2}$)')

                xx.axvline(x=semia**0.25, color=colors2[1], lw=2, ls='--', label=f'R(mom)={semia:.2f} arcsec')
                hndls, _ = xx.get_legend_handles_labels()
                if hndls:
                    xx.legend(loc='upper right', fontsize=8)
            else:
                ax[idata, 0].text(0.03, 0.97, f'{label} - No Data',
                                  transform=ax[idata, 0].transAxes,
                                  ha='left', va='top', color='white',
                                  linespacing=1.5, fontsize=10,
                                  bbox=dict(boxstyle='round', facecolor='k', alpha=0.5))
                ax[idata, 0].set_xticks([])
                ax[idata, 0].set_yticks([])

                ax[idata, 1].set_yticks([])
                ax[idata, 1].margins(x=0)
                ax[idata, 1].set_xlim(ax[0, 1].get_xlim())

                if idata == ndataset-1:
                    ax[idata, 1].set_xlabel(r'(Semi-major axis / arcsec)$^{1/4}$')
                else:
                    ax[idata, 1].set_xticks([])


        fig.suptitle(f'{data["galaxy"].replace("_", " ").replace(" GROUP", " Group")}: ' + \
                     f'{obj["OBJNAME"]} ({obj[REFIDCOLUMN]})')
        #fig.suptitle(data['galaxy'].replace('_', ' ').replace(' GROUP', ' Group'))
        fig.tight_layout()
        fig.savefig(qafile, bbox_inches='tight')
        plt.close()
        log.info(f'Wrote {qafile}')


def wrap_multifit(data, sample, datasets, unpack_maskbits_function,
                  sbthresh, apertures, SGAMASKBITS, mp=1, nmonte=100,
                  seed=42, debug=False):
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
                sma_moment_arcsec = obj['SMA_MOMENT']  # [arsec]
                semia_pix = sma_moment_arcsec / pixscale    # [pixels]
                psf_fwhm_pix = 1.1 / pixscale          # [pixels]
                allbands = data['all_opt_bands'] # always griz in north & south

                # reference apertures
                sma_apertures_arcsec = sma_moment_arcsec * np.array(apertures) # [arcsec]

                opt_sma_array_pix, info = build_sma_opt(
                    s95_pix=max(semia_pix, 3*psf_fwhm_pix), ba=ba, psf_fwhm_pix=psf_fwhm_pix,
                    inner_step_pix=1., min_pixels_per_annulus=15,
                    frac_step=0.15, amax_factor=3.)
                sma_array_pix = np.copy(opt_sma_array_pix)
            else:
                allbands = bands

                sma_array_pix = build_sma_band( # [pixels]
                    opt_sma_array_pix, opt_pixscale=opt_pixscale,
                    pixscale=pixscale, ba=ba,
                    min_pixels_per_annulus=5) # ~constant S/N per annulus

            #print(sma_array_pix)
            results_dataset1, sbprofiles_dataset1 = multifit(
                obj, images, sigimages, masks, sma_array_pix, dataset,
                bands, opt_wcs=opt_wcs, wcs=wcs, opt_pixscale=opt_pixscale,
                pixscale=pixscale, mp=mp, nmonte=nmonte, allbands=allbands,
                sbthresh=sbthresh, sma_apertures_arcsec=sma_apertures_arcsec,
                seed=seed, debug=debug)

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
                         unwise_pixscale=2.75, mask_nearby=None, galex=True, unwise=True,
                         sbthresh=REF_SBTHRESH, apertures=REF_APERTURES, update_geometry=False,
                         nmonte=75, seed=42, verbose=False, skip_ellipse=False,
                         nowrite=False, clobber=False, qaplot=False,
                         htmlgalaxydir=None):
    """Top-level wrapper script to do ellipse-fitting on all galaxies
    in a given group or coadd.

    """
    from astropy.table import Table
    from SGA.SGA import SGA_diameter, build_multiband_mask

    datasets = ['opt']
    if unwise:
        datasets += ['unwise']
    if galex:
        datasets += ['galex']

    # we need as many MASKBITS bit-masks as datasetss
    assert(len(SGAMASKBITS) == len(datasets))

    # In the case that there were "no photometric CCDs touching
    # brick", there will not be a CCDs file (e.g.,
    # dr11-south/203/20337p3381=WISEA J133330.14+334903.2), so we
    # should exit cleanly.
    ccdsfile = os.path.join(galaxydir, f'{galaxy}-ccds.fits')
    if not os.path.isfile(ccdsfile):
        log.info('No CCDs touching this brick; nothing to do.')
        return 1

    data, tractor, sample, samplesrcs, err = read_multiband_function(
        galaxy, galaxydir, REFIDCOLUMN, bands=bands, run=run,
        pixscale=pixscale, galex_pixscale=galex_pixscale,
        unwise_pixscale=unwise_pixscale, unwise=unwise, galex=galex,
        verbose=verbose, skip_ellipse=skip_ellipse)
    if err == 0:
        log.warning(f'Problem reading (or missing) data for {galaxydir}/{galaxy}')
        return err

    FMAJOR_geo = 0.01
    FMAJOR_final = 0.1

    try:
        err = 1

        # mask aggressively to determine the geometry; use FMAJOR_geo
        # plus mask_minor_galaxies=True (outside the ellipse)
        data, sample = build_multiband_mask(
            data, tractor, sample, samplesrcs, qaplot=False, cleanup=False,
            mask_nearby=mask_nearby, niter_geometry=2, FMAJOR_geo=FMAJOR_geo,
            mask_minor_galaxies=True, htmlgalaxydir=htmlgalaxydir)

    except:
        err = 0
        log.critical(f'Exception raised on {galaxydir}/{galaxy}')
        import traceback
        traceback.print_exc()

    if err == 0:
        log.warning(f'Problem building image masks for {galaxydir}/{galaxy}')
        return err

    # special case: completely empty Tractor catalog (e.g.,
    # r9-north/326/32630p0027)
    if err == 1 and not bool(data):
        return err

    # if skipping ellipse-fitting just initialize the data model and write out
    if skip_ellipse:
        results_obj = []
        sbprofiles_obj = []

        for iobj, obj in enumerate(sample):
            sma_moment_arcsec = obj['SMA_MOMENT']
            sma_apertures_arcsec = sma_moment_arcsec * np.array(apertures)

            results_dataset = []
            sbprofiles_dataset = []
            for dataset in datasets:
                if dataset == 'opt':
                    allbands = data['all_opt_bands']
                else:
                    allbands = data[f'{dataset}_bands']

                results_dataset1 = results_datamodel(obj, allbands, dataset,sma_apertures_arcsec, sbthresh)
                sbprofiles_dataset1 = Table()
                #sbprofiles_dataset1 = sbprofiles_datamodel(sma_array*pixscale, allbands)
                results_dataset.append(results_dataset1)
                sbprofiles_dataset.append(sbprofiles_dataset1)

        results = list(zip(*results_obj))       # [ndatasets][nobj]
        sbprofiles = list(zip(*sbprofiles_obj)) # [ndatasets][nobj]
    else:
        # First fit just the optical and then update the mask.
        results, sbprofiles = wrap_multifit(
            data, sample, ['opt'], unpack_maskbits_function,
            sbthresh, apertures, [SGAMASKBITS[0]], mp=mp,
            nmonte=0, seed=seed, debug=False)

        if update_geometry:
            input_geo_initial = None
            niter_geometry = 1 # 2
        else:
            input_geo_initial = np.zeros((len(sample), 5)) # [bx,by,sma,ba,pa]
            niter_geometry = 1 # not used by build_multiband_mask

        #sma_moment0 = sample['SMA_MOMENT'].copy() # original values
        for iobj, obj in enumerate(sample):
            bx, by, sma_mom, ba_mom, pa_mom = [
                obj['BX'], obj['BY'], obj['SMA_MOMENT'], obj['BA_MOMENT'], obj['PA_MOMENT']]

            # if FIXGEO or TRACTORGEO use the input geometry
            if obj['ELLIPSEMODE'] & (ELLIPSEMODE['FIXGEO'] | ELLIPSEMODE['TRACTORGEO']) != 0:
                if not update_geometry:
                    input_geo_initial[iobj, :] = [bx, by, sma_mom/pixscale, ba_mom, pa_mom]
                continue

            # estimate R(26) from first-pass profiles
            tab = Table(obj['SMA_MOMENT', 'ELLIPSEMODE'])
            for thresh in sbthresh:
                for filt in bands:
                    col = f'R{thresh:.0f}_{filt.upper()}'
                    colerr = f'R{thresh:.0f}_ERR_{filt.upper()}'
                    tab[col] = results[0][iobj][col]
                    tab[colerr] = results[0][iobj][colerr]
            radius, radius_err, radius_ref, radius_weight = SGA_diameter(tab, radius_arcsec=True)
            r26_arcsec = float(radius[0])

            # merge R26 with the existing SMA_MASK in arcsec
            sma_moment_arcsec = obj['SMA_MOMENT']
            sma_mask_arcsec = obj['SMA_MASK']

            log.info(f'Initial estimate R(26)={r26_arcsec:.2f} arcsec [previous ' + \
                     f'sma_mask={sma_mask_arcsec:.2f} arcsec].')

            if sma_mask_arcsec <= 0.:
                sma_mask_arcsec = r26_arcsec
            else:
                sma_mask_arcsec = max(sma_mask_arcsec, r26_arcsec)

            sample['SMA_MASK'][iobj] = sma_mask_arcsec
            if not update_geometry:
                # pass explicit/fixed geometry to build_multiband_mask
                input_geo_initial[iobj, :] = [bx, by, sma_moment_arcsec/pixscale, ba_mom, pa_mom]

        # pull back on the masking for the final iteration
        data, sample = build_multiband_mask(data, tractor, sample, samplesrcs,
                                            input_geo_initial=input_geo_initial,
                                            mask_nearby=mask_nearby, qaplot=qaplot,
                                            FMAJOR_geo=FMAJOR_geo, FMAJOR_final=FMAJOR_final,
                                            mask_minor_galaxies=False,
                                            niter_geometry=niter_geometry,
                                            htmlgalaxydir=htmlgalaxydir)

        # ellipse-fit over objects and then datasets
        results, sbprofiles = wrap_multifit(
            data, sample, datasets, unpack_maskbits_function,
            sbthresh, apertures, SGAMASKBITS, mp=mp,
            nmonte=nmonte, seed=seed, debug=False)#qaplot)

        if qaplot:
            qa_ellipsefit(data, sample, results, sbprofiles, unpack_maskbits_function,
                          SGAMASKBITS, REFIDCOLUMN, datasets=datasets,
                          htmlgalaxydir=htmlgalaxydir)

    if not nowrite:
        from SGA.io import write_ellipsefit
        err = write_ellipsefit(data, sample, datasets, results, sbprofiles,
                               SGAMASKBITS, verbose=verbose)

    return err
