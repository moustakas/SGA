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

from SGA.util import get_dt
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
                        # ellipse (e.g., within a cluster environment)
    MOMENTPOS = 2**6,   # use the light-weighted (not Tractor) center
    TRACTORGEO = 2**7,  # use the Tractor (not light-weighted) geometry
    NORADWEIGHT = 2**8, # derive the moment geometry without radial weighting
)

ELLIPSEBIT = dict(
    NOTRACTOR = 2**0,          # SGA source has no corresponding Tractor source
    TRACTORPSF = 2**1,         # SGA source fit by Tractor as a PSF
    FIXGEO = 2**2,             # fixed ellipse geometry (from ELLIPSEMODE)
    BLENDED = 2**3,            # SGA center is located within the elliptical mask of another SGA source
    LARGESHIFT = 2**4,         # >MAXSHIFT_ARCSEC shift between the initial and final ellipse position
    LARGESHIFT_TRACTOR = 2**5, # >MAXSHIFT_ARCSEC shift between the Tractor and final ellipse position
    MAJORGAL = 2**6,           # nearby bright galaxy (>=XX% of the SGA source) subtracted
    OVERLAP = 2**7,            # any part of the initial SGA ellipse overlaps another SGA ellipse
    SATELLITE = 2**8,          # satellite of another larger galaxy
    MOMENTPOS = 2**9,          # light-weighted (not Tractor) center
    TRACTORGEO = 2**10,        # used the Tractor (not light-weighted) geometry
    NORADWEIGHT = 2**11,       # moment geometry derived without radial weighting
    LESSMASKING = 2**12,       # Gaia stars were subtracted but not threshold-masked
    MOREMASKING = 2**13,       # extended sources were threshold-masked even within the SGA ellipse
    FAILGEO = 2**14,           # failed to derive the ellipse geometry (reverted to initial geometry)
    SKIPTRACTOR = 2**15,       # skip Tractor fitting entirely
)

REF_SBTHRESH = [22., 23., 24., 25., 26.]     # surface brightness thresholds
REF_APERTURES = [0.5, 1., 1.25, 1.5, 2., 3.] # multiples of SMA_MOMENT


def to_float32_safe_mapping(d):
    """Clip and cast every value in a mapping to a FITS-safe
    :class:`numpy.float32`.

    Non-finite values (NaN/inf) become ``float32(nan)``; finite values
    outside the representable ``float32`` range are clipped to the
    ``float32`` min/max rather than overflowing to inf.

    Parameters
    ----------
    d : :class:`dict`
        Mapping of key to numeric value (any type castable to
        :class:`float`).

    Returns
    -------
    :class:`dict`
        Same keys, each value cast to :class:`numpy.float32` as described
        above.

    """
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
    """Clip and cast a single scalar to a FITS-safe :class:`numpy.float32`.

    Returns ``float32(nan)`` if ``x`` is not castable to :class:`float`,
    is non-finite, or falls outside the representable ``float32`` range
    (rather than overflowing to inf).

    Parameters
    ----------
    x : scalar
        Value to cast (any type castable to :class:`float`).

    Returns
    -------
    :class:`numpy.float32`
        The clipped/cast value, or ``float32(nan)`` if unrepresentable.

    """
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
    """Evaluate the curve-of-growth magnitude model at a set of radii.

    ``m(r) = mtot + dmag * (1 - exp(-exp(lnalpha1) * (r/r0)^(-exp(lnalpha2))))``

    As ``r -> infinity``, ``m(r) -> mtot``, the asymptotic total
    magnitude; as ``r -> 0``, ``m(r) -> mtot + dmag``, the (fainter)
    innermost-aperture magnitude. Evaluated in log-space with clipping
    to avoid float overflow in the exponential.

    Parameters
    ----------
    radius : array-like
        Semi-major axis (or radius) at which to evaluate the model, same
        units as ``r0``.
    mtot : :class:`float`
        Asymptotic total magnitude (the value ``m(r)`` approaches as
        ``r -> infinity``).
    dmag : :class:`float`
        Positive amplitude, approximately ``m(0) - mtot``.
    lnalpha1 : :class:`float`
        ``log(alpha1)``, where ``alpha1 = exp(lnalpha1) > 0`` sets the
        curve's normalization at ``r0``.
    lnalpha2 : :class:`float`
        ``log(alpha2)``, where ``alpha2 = exp(lnalpha2) > 0`` sets the
        power-law steepness of the curve.
    r0 : :class:`float`
        Scale radius, same units as ``radius``.

    Returns
    -------
    :class:`numpy.ndarray`
        Model magnitude(s) at ``radius``.

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
    """Fit the curve-of-growth magnitude model (see :func:`cog_model`) to
    a curve-of-growth (aperture flux vs. semi-major axis) measurement.

    Converts flux to magnitudes, optionally weights by flux-derived
    magnitude errors (with an ``minerr`` floor added in quadrature), and
    fits ``(mtot, dmag, lnalpha1, lnalpha2)`` via
    :func:`scipy.optimize.least_squares` with an analytic Jacobian,
    optionally using a robust (``'soft_l1'``) loss. Initial guesses are
    computed automatically (from the faint/bright ends of the profile and
    a pivot at the geometric-mean radius) unless ``p0`` is given.
    Parameter uncertainties are the sqrt-diagonal of the covariance
    matrix estimated from ``(J^T J)^{-1}`` scaled by the reduced
    chi-square. Returns empty results (``{}, {}, None, 0., ndof``) if
    fewer than 5 usable points remain, the covariance is singular or has
    non-positive variances, or any fitted parameter/uncertainty is
    non-finite.

    Notes
    -----
    ``debug`` is accepted but not referenced anywhere in this function's
    body -- dead parameter.

    Parameters
    ----------
    sma_arcsec : :class:`numpy.ndarray`
        Semi-major axis of each aperture, in arcsec.
    flux : :class:`numpy.ndarray`
        Enclosed flux at each ``sma_arcsec`` (same length), in linear
        (e.g. nanomaggy) units; converted internally to AB magnitude via
        ``22.5 - 2.5*log10(flux)``.
    ferr : :class:`numpy.ndarray`, optional
        1-sigma flux uncertainty at each point, same length as ``flux``.
        If given, used to build magnitude weights (with the ``minerr``
        floor); if None, the fit is unweighted.
    r0 : :class:`float`
        Scale radius passed to :func:`cog_model`, arcsec.
    p0 : :class:`tuple`, optional
        Initial guess ``(mtot, dmag, lnalpha1, lnalpha2)``; computed
        automatically from the data if not given.
    ndrop : :class:`int`
        Number of innermost points to exclude from the fit (e.g. where
        the PSF dominates, especially for GALEX/unWISE).
    bounds : :class:`tuple`, optional
        ``((mtot_lo, dmag_lo, lnalpha1_lo, lnalpha2_lo), (mtot_hi, ...))``
        parameter bounds; computed automatically from the magnitude range
        of the data if not given.
    robust : :class:`bool`
        If True, use a ``'soft_l1'`` robust loss in the least-squares fit
        (down-weights outliers); if False, use ordinary least squares.
    minerr : :class:`float`
        Magnitude-error floor added in quadrature to the flux-derived
        weights, to prevent near-zero errors from dominating the fit.
    f_scale : :class:`float`
        Scale parameter for the robust loss function, passed to
        :func:`scipy.optimize.least_squares`.
    debug : :class:`bool`
        Unused (see Notes).

    Returns
    -------
    popt : :class:`dict`
        Best-fit parameters, keys ``'mtot'``, ``'dmag'``, ``'lnalpha1'``,
        ``'lnalpha2'`` (:class:`numpy.float32`); empty dict on failure.
    perr : :class:`dict`
        Approximate 1-sigma uncertainties for the same keys, from
        ``(J^T J)^{-1}`` scaled by ``chi2/ndof``; empty dict on failure.
    cov : :class:`numpy.ndarray` or None
        4x4 covariance matrix for ``(mtot, dmag, lnalpha1, lnalpha2)``;
        None on failure.
    chi2 : :class:`float`
        Classical chi-square, ``sum(((model - data)/weight)**2)``
        (unweighted if ``ferr`` is None); ``0.`` on failure.
    ndof : :class:`int`
        Degrees of freedom, ``max(1, npoints - 4)``; ``0`` if fewer than
        5 usable points were available to attempt a fit.

    """
    def initial_guesses(sma, mags, r0, bounds, eps=1e-6):
        """Estimate initial ``(mtot, dmag, lnalpha1, lnalpha2)`` for
        :func:`fit_cog` from the shape of the input profile, clipped into
        ``bounds``.

        """
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
        """Weighted residuals of :func:`cog_model` against the observed
        magnitudes, for :func:`scipy.optimize.least_squares` in
        :func:`fit_cog`.

        """
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
        """Analytic Jacobian of ``residuals`` with respect to
        ``(mtot, dmag, lnalpha1, lnalpha2)``, for
        :func:`scipy.optimize.least_squares` in :func:`fit_cog`.

        """
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
    """Invert the curve-of-growth model (see :func:`cog_model`) to find
    the semi-major axis enclosing a given flux fraction.

    Parameters
    ----------
    f : :class:`float`
        Enclosed flux fraction in (0, 1), e.g. 0.5 for the half-light
        radius.
    dmag : :class:`float`
        Curve-of-growth amplitude (> 0), as in :func:`cog_model`.
    lnalpha1 : :class:`float`
        ``log(alpha1)``, as in :func:`cog_model`.
    lnalpha2 : :class:`float`
        ``log(alpha2)``, as in :func:`cog_model`.
    r0 : :class:`float`
        Scale radius; the returned radius is in the same units.

    Returns
    -------
    :class:`float`
        Semi-major axis ``r_f`` enclosing flux fraction ``f``.

    Raises
    ------
    ValueError
        If ``f`` is not in (0, 1), or if ``dmag`` is too small to reach
        flux fraction ``f`` (i.e. ``dmag <= -2.5*log10(f)``).

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
    """Propagate curve-of-growth parameter uncertainties to the
    uncertainty on the radius enclosing flux fraction ``f`` (see
    :func:`radius_for_fraction`), via analytic derivatives.

    Parameters
    ----------
    f : :class:`float`
        Enclosed flux fraction in (0, 1).
    params : sequence of :class:`float`
        ``(mtot, dmag, lnalpha1, lnalpha2)``; ``mtot`` is accepted for
        positional consistency with :func:`fit_cog`'s ``popt`` ordering
        but is not used in this calculation (``r_f`` does not depend on
        ``mtot``).
    cov : :class:`numpy.ndarray`
        4x4 covariance matrix for ``(mtot, dmag, lnalpha1, lnalpha2)``,
        e.g. from :func:`fit_cog`.
    r0 : :class:`float`
        Scale radius, as in :func:`radius_for_fraction`.
    var_r0 : :class:`float`, optional
        Variance of ``r0`` itself, if it should be propagated as an
        independent source of uncertainty; ignored if None.

    Returns
    -------
    r_f : :class:`numpy.float32`
        Semi-major axis enclosing flux fraction ``f``.
    sigma_r : :class:`numpy.float32`
        Propagated 1-sigma uncertainty on ``r_f``.

    Raises
    ------
    ValueError
        If ``dmag`` is too small to reach flux fraction ``f`` (same
        condition as :func:`radius_for_fraction`).

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
    """Half-light (50% flux) radius from curve-of-growth parameters; a
    thin wrapper around :func:`radius_for_fraction` with ``f=0.5``.

    Parameters
    ----------
    params : sequence of :class:`float`
        ``(mtot, dmag, lnalpha1, lnalpha2)``; ``mtot`` is unused.
    r0 : :class:`float`
        Scale radius, as in :func:`radius_for_fraction`.

    Returns
    -------
    :class:`float`
        Half-light semi-major axis.

    """
    _, dmag, lnA1, lnA2 = params
    return radius_for_fraction(0.5, dmag, lnA1, lnA2, r0=r0)


def half_light_radius_with_uncertainty(params, cov, r0=10., var_r0=None):
    """Half-light radius and its uncertainty; a thin wrapper around
    :func:`radius_fraction_uncertainty` with ``f=0.5``.

    Parameters
    ----------
    params : sequence of :class:`float`
        ``(mtot, dmag, lnalpha1, lnalpha2)``.
    cov : :class:`numpy.ndarray`
        4x4 covariance matrix, as in :func:`radius_fraction_uncertainty`.
    r0 : :class:`float`
        Scale radius.
    var_r0 : :class:`float`, optional
        Variance of ``r0``, if it should be propagated.

    Returns
    -------
    r_f : :class:`numpy.float32`
        Half-light semi-major axis.
    sigma_r : :class:`numpy.float32`
        Propagated 1-sigma uncertainty.

    """
    return radius_fraction_uncertainty(0.5, params, cov, r0=r0, var_r0=var_r0)


def _integrate_isophot_one(args):
    """Unpack an argument tuple and call :func:`integrate_isophot_one`;
    multiprocessing worker.

    Parameters
    ----------
    args : :class:`tuple`
        Positional arguments matching :func:`integrate_isophot_one`'s
        signature.

    Returns
    -------
    See :func:`integrate_isophot_one`.

    """
    return integrate_isophot_one(*args)


def _boxcar(y, w):
    """Apply a simple centered moving-average (boxcar) smoothing to a
    1-D array.

    Parameters
    ----------
    y : :class:`numpy.ndarray`
        Input array to smooth.
    w : :class:`int` or None
        Window width; forced to the next odd integer if even. If None or
        ``< 2``, ``y`` is returned unchanged.

    Returns
    -------
    :class:`numpy.ndarray`
        Smoothed array, same length as ``y`` (``mode='same'`` convolution).

    """
    if w is None or w < 2:
        return y
    w = int(w)
    if w % 2 == 0:
        w += 1
    k = np.ones(w, float) / w
    # reflect at edges to avoid edge dips
    return np.convolve(y, k, mode='same')


def _outer_isophotal_radius(a, mu, mu_iso, smooth_win=None):
    """Find the single outer crossing radius where a surface-brightness
    profile reaches a target isophote level.

    Assumes ``a`` is increasing. Applies optional boxcar smoothing (see
    :func:`_boxcar`), enforces a non-decreasing envelope in ``mu`` via a
    running maximum (surface brightness must get fainter outward), and
    linearly interpolates the radius at which the envelope crosses
    ``mu_iso``.

    Parameters
    ----------
    a : :class:`numpy.ndarray`
        Semi-major axis values, increasing.
    mu : :class:`numpy.ndarray`
        Surface brightness profile, mag/arcsec^2, same length as ``a``.
    mu_iso : :class:`float`
        Target isophote level, mag/arcsec^2.
    smooth_win : :class:`int`, optional
        Boxcar smoothing window passed to :func:`_boxcar`.

    Returns
    -------
    :class:`float`
        Interpolated crossing radius, same units as ``a``; ``numpy.nan``
        if ``mu_iso`` falls outside the range spanned by the (smoothed,
        monotonized) profile.

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
    a,
    mu,
    mu_err,
    mu_iso,
    nmonte=50,
    sky_sigma=None,
    smooth_win=3,
    random_state=None,
    return_samples=False):
    """Monte Carlo isophotal radius at a target surface-brightness level,
    with uncertainty from resampling the profile.

    Sorts and sanitizes the input profile, then (if ``nmonte > 0``) draws
    ``nmonte`` noisy realizations of ``mu`` (per-point Gaussian noise from
    ``mu_err``, plus an optional shared per-draw offset from
    ``sky_sigma``), finds the outer isophote crossing radius for each
    realization via :func:`_outer_isophotal_radius`, and summarizes the
    successful draws by their median and interquartile range. If
    ``nmonte <= 0``, instead evaluates a single crossing on the nominal
    (noiseless) profile with no uncertainty quantification (see Notes).
    Also flags whether the *nominal* profile is a lower limit (never
    reaches ``mu_iso`` at large radius) or upper limit (already fainter
    than ``mu_iso`` at the innermost point).

    Notes
    -----
    Despite the "16th/84th percentiles" phrasing in earlier versions of
    this docstring, the code actually uses the 25th/75th percentiles
    (interquartile range) with ``sigma = (p75 - p25) / 1.349`` (the
    Gaussian IQR-to-sigma conversion factor, consistent with 25/75 rather
    than 16/84). When ``nmonte <= 0`` and a valid crossing is found on
    the nominal profile, ``a_iso_err``/``a_lo``/``a_hi`` are all
    hardcoded to ``0.`` -- no uncertainty is estimated in that mode
    (marked in the source with an explicit "no uncertainty??" comment
    from the original author).

    Parameters
    ----------
    a : :class:`numpy.ndarray`
        Semi-major axis array.
    mu : :class:`numpy.ndarray`
        Surface brightness profile, mag/arcsec^2, same length as ``a``.
    mu_err : :class:`numpy.ndarray`
        1-sigma uncertainty on ``mu``, same shape.
    mu_iso : :class:`float`
        Target isophote level, mag/arcsec^2 (e.g. 25.0).
    nmonte : :class:`int`
        Number of Monte Carlo draws; if ``<= 0``, evaluate only the
        nominal profile (see Notes).
    sky_sigma : :class:`float`, optional
        Additional global sky-magnitude uncertainty, applied as a single
        shared offset per MC draw (not per point).
    smooth_win : :class:`int`
        Boxcar smoothing window (odd; see :func:`_boxcar`) applied before
        computing each draw's envelope crossing; None or 1 disables
        smoothing.
    random_state : :class:`int` or :class:`numpy.random.Generator`, optional
        Seed or generator passed to :func:`numpy.random.default_rng`.
    return_samples : :class:`bool`
        If True, include the array of successful per-draw radii in the
        output under the ``'samples'`` key.

    Returns
    -------
    :class:`dict`
        With keys ``'a_iso'`` (median isophotal radius, same units as
        ``a``), ``'a_iso_err'`` (robust sigma from the IQR, see Notes),
        ``'a_lo'``/``'a_hi'`` (25th/75th percentiles, see Notes),
        ``'success_rate'`` (fraction of MC draws with a valid crossing),
        ``'n_success'`` (number of valid draws), ``'nmonte'`` (total
        draws requested), ``'lower_limit'`` (bool, True if the nominal
        profile never reaches ``mu_iso`` outward), ``'upper_limit'``
        (bool, True if the nominal profile is already fainter than
        ``mu_iso`` at the innermost point), and, if
        ``return_samples=True``, ``'samples'`` (the per-draw radii
        array). ``'a_iso'``/``'a_iso_err'``/``'a_lo'``/``'a_hi'`` are all
        ``numpy.nan`` if no draw produced a valid crossing.

    Raises
    ------
    ValueError
        If fewer than two finite points remain in the input profile.

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
    """Integrate the ellipse profile (surface brightness and/or aperture
    photometry) at a single semi-major axis.

    Elliptical aperture photometry (``flux``, ``ferr``, ``fracmasked``)
    is always computed (via :class:`photutils.aperture.EllipticalAperture`),
    except at ``sma == 0`` where it is set to zero as a placeholder.
    Surface-brightness isophote sampling/fitting (via
    :class:`photutils.isophote.EllipseSample`/:class:`~photutils.isophote.Isophote`,
    or the central-pixel special case via
    :class:`~photutils.isophote.sample.CentralEllipseSample`/
    :class:`~photutils.isophote.fitter.CentralEllipseFitter` at
    ``sma == 0``) is only performed when ``measure_sb`` is True; ``iso``
    is None otherwise.

    Parameters
    ----------
    mimg : :class:`numpy.ma.MaskedArray`
        Masked image to sample/integrate.
    sig : :class:`numpy.ndarray`
        Per-pixel uncertainty (sigma) image, same shape as ``mimg``.
    msk : :class:`numpy.ndarray`
        Boolean mask, same shape as ``mimg``; True = masked pixel.
    sma : :class:`float`
        Semi-major axis, in pixels; ``0.`` selects the central-pixel
        special case.
    theta : :class:`float`
        Position angle, in radians, counterclockwise from the x-axis.
    eps : :class:`float`
        Ellipticity, ``1 - b/a``.
    x0, y0 : :class:`float`
        Ellipse center, in pixel coordinates.
    integrmode : :class:`str`
        Pixel-integration mode passed to
        :class:`photutils.isophote.EllipseSample`
        (e.g. ``'bilinear'``, ``'mean'``, ``'median'``).
    sclip : :class:`float`
        Sigma-clipping threshold for the isophote sample.
    nclip : :class:`int`
        Number of sigma-clipping iterations for the isophote sample.
    measure_sb : :class:`bool`
        If True, also build and fit an isophote sample for surface
        brightness (see above); if False, only aperture photometry is
        computed and ``iso`` is None.

    Returns
    -------
    iso : :class:`~photutils.isophote.Isophote` or None
        Fitted isophote sample, or None if ``measure_sb`` is False.
    flux : :class:`numpy.float32`
        Aperture flux within the ellipse at ``sma``.
    ferr : :class:`numpy.float32`
        Aperture flux uncertainty.
    fracmasked : :class:`numpy.float32`
        Fraction of the aperture's area falling on masked pixels.

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
    """Generate ``n`` monotonically increasing, log-spaced, unique
    0-indexed integers spanning ``[0, limit-1]``.

    Iteratively grows a geometric sequence from 1 toward ``limit``; when
    the geometric step would round to a repeated integer, it is instead
    forced to increment by 1 and the growth ratio for the remaining terms
    is recomputed so the sequence still reaches ``limit``. Adapted from
    https://stackoverflow.com/questions/12418234/logarithmically-spaced-integers.
    Used to select a small, outward-densifying subset of indices (e.g.
    aperture or isophote steps) out of a larger linear range.

    Parameters
    ----------
    limit : :class:`int`
        Upper bound (1-indexed) of the range to span; the returned
        (0-indexed) values reach up to approximately ``limit - 1``.
    n : :class:`int`
        Number of integers to return.

    Returns
    -------
    :class:`numpy.ndarray`
        Length-``n`` array of increasing integers (dtype ``int``),
        0-indexed, log-spaced (denser at the low end).

    """
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
    """Build an empty per-radius surface-brightness profile table.

    One row per input semi-major-axis value; for each band, adds
    zero-filled ``SB_{filt}``/``SB_ERR_{filt}`` (surface brightness) and
    ``FLUX_{filt}``/``FLUX_ERR_{filt}`` (aperture flux) columns, plus a
    ``FMASKED_{filt}`` column initialized to 1 (fully masked) as a
    placeholder until populated by the caller.

    Parameters
    ----------
    sma : :class:`numpy.ndarray`
        Semi-major-axis sampling grid, in arcsec (one row per value).
    bands : :class:`list` of :class:`str`
        Bands to add columns for (upper-cased in the column names).

    Returns
    -------
    :class:`~astropy.table.Table`
        Table with ``len(sma)`` rows: ``SMA`` plus per-band ``SB_*``,
        ``SB_ERR_*``, ``FLUX_*``, ``FLUX_ERR_*``, ``FMASKED_*`` columns.

    """
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
    """Build an empty single-row summary-results table for one object.

    Seeded with ``obj``'s ``SGAID``/``SGAGROUP``, then adds per-band
    ``GINI_{filt}``; curve-of-growth fit parameters and errors
    (``COG_MTOT``, ``COG_DMAG``, ``COG_LNALPHA1``, ``COG_LNALPHA2``,
    ``COG_CHI2``, ``COG_NDOF``, ``SMA50``, each per band, with
    ``_ERR`` counterparts except ``CHI2``/``NDOF``); aperture photometry
    at each radius in ``sma_apertures_arcsec`` (``SMA_AP##``,
    ``FLUX_AP##_{filt}``, ``FLUX_ERR_AP##_{filt}``,
    ``FMASKED_AP##_{filt}``, the latter initialized to 1); and, only
    when ``dataset == 'opt'``, isophotal radii at each threshold in
    ``sbthresh`` (``R{thresh}_{filt}``, ``R{thresh}_ERR_{filt}``).

    Parameters
    ----------
    obj : :class:`~astropy.table.Table` row
        Object row supplying ``SGAID``/``SGAGROUP`` for the output table.
    bands : :class:`list` of :class:`str`
        Bands to add columns for (upper-cased in the column names).
    dataset : :class:`str`
        Imaging dataset name; isophotal-radius columns are only added
        when this is ``'opt'``.
    sma_apertures_arcsec : array-like
        Aperture semi-major-axis radii, in arcsec, defining the
        ``SMA_AP##``/``FLUX_AP##_*`` columns.
    sbthresh : array-like
        Surface-brightness thresholds (mag/arcsec^2) defining the
        isophotal-radius columns; unused when ``dataset != 'opt'``.

    Returns
    -------
    :class:`~astropy.table.Table`
        Single-row, zero-filled results table with the column set
        described above.

    """
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
             opt_pixscale=0.262, pixscale=0.262, mp=1, nmonte=50,
             allbands=None, integrmode='median', nclip=3, sclip=3,
             seed=42, sbthresh=REF_SBTHRESH, sma_apertures_arcsec=None,
             debug=False):
    """Fit elliptical-isophote surface-brightness profiles and
    curve-of-growth photometry for one object, across one imaging
    dataset's bands.

    Broadly based on the photutils isophote-fitting examples --
    https://github.com/astropy/photutils-datasets/blob/master/notebooks/isophote/isophote_example4.ipynb
    https://photutils.readthedocs.io/en/latest/user_guide/isophote.html

    For each band: computes the Gini coefficient within the moment
    ellipse; measures elliptical-aperture photometry at every radius in
    ``sma_array`` (via :func:`integrate_isophot_one`, in parallel across
    ``mp`` workers if requested) to build the per-radius flux and
    surface-brightness profile; fits the curve-of-growth model
    (:func:`fit_cog`) anchored at the object's moment semi-major axis;
    derives the half-light radius and its uncertainty
    (:func:`half_light_radius_with_uncertainty`); performs aperture
    photometry at the fixed reference apertures in
    ``sma_apertures_arcsec``; and, only for ``dataset == 'opt'``, derives
    isophotal radii at each threshold in ``sbthresh`` via Monte Carlo
    (:func:`isophotal_radius_mc`).

    Notes
    -----
    ``opt_pixscale`` is accepted but never referenced in this function's
    body -- only ``pixscale`` is used for every pixel/arcsec conversion.
    ``debug`` is also effectively dead: immediately after the docstring
    the local variable ``debug`` is unconditionally reset to ``False``
    (``debug = False``, shadowing the parameter), so the diagnostic
    matplotlib plotting blocks guarded by ``if debug:`` never execute
    regardless of the value passed in by the caller.

    Parameters
    ----------
    obj : :class:`~astropy.table.Table` row
        Object's sample row; must supply ``BX``, ``BY``, ``PA_MOMENT``,
        ``BA_MOMENT``, ``SMA_MOMENT``, ``SGAID``, ``SGAGROUP``.
    images : :class:`numpy.ndarray`
        Surface-brightness images for this object/dataset, shape
        ``(nband, width, width)``.
    sigimages : :class:`numpy.ndarray`
        Per-pixel uncertainty images, same shape as ``images``.
    masks : :class:`numpy.ndarray`
        Boolean per-band pixel mask (True = masked), same shape as
        ``images``.
    sma_array : :class:`numpy.ndarray`
        Semi-major-axis sampling grid, in pixels (e.g. from
        :func:`build_sma_opt`/:func:`build_sma_band`), at which the
        surface-brightness profile is measured.
    dataset : :class:`str`
        Imaging dataset name (``'opt'``, ``'unwise'``, ``'galex'``);
        only ``'opt'`` triggers isophotal-radius computation.
    bands : :class:`list` of :class:`str`
        Bands to fit; must align with the leading axis of ``images``/
        ``sigimages``/``masks``.
    opt_wcs : WCS
        Optical reference-band WCS, used with :func:`SGA.sky.map_bxby`
        to transform the object's optical ``(BX, BY)`` into this
        dataset's pixel frame.
    wcs : WCS
        This dataset's WCS.
    opt_pixscale : :class:`float`
        Unused (see Notes).
    pixscale : :class:`float`
        This dataset's pixel scale, arcsec/pixel; used for all
        pixel/arcsec conversions.
    mp : :class:`int`
        Number of multiprocessing workers for the per-radius isophote
        integration.
    nmonte : :class:`int`
        Number of Monte Carlo trials passed to
        :func:`isophotal_radius_mc`.
    allbands : :class:`list` of :class:`str`, optional
        Full band set defining the output tables' column set (see
        :func:`results_datamodel`/:func:`sbprofiles_datamodel`); defaults
        to ``bands``. If it includes bands beyond ``bands``, those
        columns exist in the output but are never populated (the fit
        loop only iterates over ``bands``) -- by design, for producing a
        fixed-width table when merging results with other band subsets.
    integrmode : :class:`str`
        Isophote integration mode passed to
        :func:`integrate_isophot_one` (photutils convention, e.g.
        ``'median'``).
    nclip, sclip : :class:`int`
        Sigma-clipping iteration count and threshold passed to
        :func:`integrate_isophot_one`.
    seed : :class:`int`
        Random seed (``random_state``) passed to
        :func:`isophotal_radius_mc`.
    sbthresh : array-like
        Surface-brightness thresholds (mag/arcsec^2) for isophotal-radius
        computation; only used when ``dataset == 'opt'``.
    sma_apertures_arcsec : array-like
        Fixed reference aperture radii, in arcsec, for the
        ``FLUX_AP##_*`` columns.
    debug : :class:`bool`
        Unused (see Notes).

    Returns
    -------
    results : :class:`~astropy.table.Table`
        Single-row summary table (see :func:`results_datamodel`),
        populated with Gini coefficients, curve-of-growth parameters and
        errors, half-light radii, reference-aperture photometry, and
        (for ``dataset == 'opt'``) isophotal radii.
    sbprofiles : :class:`~astropy.table.Table`
        Per-radius surface-brightness profile table (see
        :func:`sbprofiles_datamodel`), populated with flux and surface
        brightness at each ``sma_array`` value.

    """
    import multiprocessing
    from photutils.isophote import EllipseGeometry, IsophoteList
    from photutils.aperture import EllipticalAperture
    from photutils.morphology import gini
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
    """Rescale an optical-band semi-major-axis edge grid onto another
    imaging dataset's (coarser) pixel scale, thinning it so each
    retained annulus has adequate width and area.

    Converts ``opt_sma_array`` (optical pixels) to arcsec, then to
    candidate edges in the target band's pixels, optionally clamps to
    ``[a_min_tgt_px, a_max_tgt_px]``, and greedily keeps a candidate
    edge only if it is both at least ``min_step_pixels`` beyond the
    previously kept edge and encloses an annulus area (assuming axis
    ratio ``ba``) of at least ``min_pixels_per_annulus`` -- since a grid
    fine enough for the optical pixel scale would otherwise be
    oversampled (too few pixels per annulus) at a coarser IR/UV pixel
    scale.

    Parameters
    ----------
    opt_sma_array : array-like
        Semi-major-axis edges from the optical-band grid (e.g. from
        :func:`build_sma_opt`), in optical pixels.
    opt_pixscale : :class:`float`
        Optical pixel scale, arcsec/pixel.
    pixscale : :class:`float`
        Target band's pixel scale, arcsec/pixel (e.g. 1.5 for GALEX,
        2.75 for unWISE).
    ba : :class:`float`
        Axis ratio b/a, used in the annulus-area constraint.
    min_step_pixels : :class:`float`
        Minimum retained radial step, in target-band pixels.
    min_pixels_per_annulus : :class:`float`
        Minimum annulus area, in target-band pixels (assuming axis ratio
        ``ba``), required to keep a candidate edge.
    a_min_tgt_px : :class:`float`, optional
        Discard candidate edges below this radius, in target pixels
        (e.g. a fraction of the PSF FWHM).
    a_max_tgt_px : :class:`float`, optional
        Discard candidate edges above this radius, in target pixels.

    Returns
    -------
    :class:`numpy.ndarray`
        Thinned semi-major-axis edges, in target-band pixels, strictly
        increasing.

    """
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
    """Build a semi-major-axis edge array for elliptical isophote fitting
    in the optical bands.

    Steps outward from a starting radius near the PSF core to a stop
    radius using two regimes: fixed linear steps of ``inner_step_pix``
    while ``a < a_transition`` (to resolve the inner profile finely near
    the PSF), then multiplicative steps ``a_next = a * (1 + frac_step)``
    beyond the transition radius (efficient log-like spacing at large
    radii). At each step, the step is enlarged if needed so the
    resulting annulus encloses at least ``min_pixels_per_annulus``
    pixels (assuming axis ratio ``ba``).

    Parameters
    ----------
    s95_pix : :class:`float`
        Characteristic object radius, in pixels (e.g. an r95-type
        radius), used to set the default stop radius via
        ``amax_factor`` when ``amax_pix`` is not given.
    ba : :class:`float`
        Axis ratio b/a, used in the minimum-annulus-area constraint.
    amax_factor : :class:`float`
        Multiplicative factor applied to ``s95_pix`` for the default
        stop radius.
    amax_pix : :class:`float`, optional
        Explicit stop radius, in pixels, overriding
        ``amax_factor * s95_pix``.
    psf_fwhm_pix : :class:`float`, optional
        PSF FWHM, in pixels. Sets the starting radius (half the FWHM,
        floored at 1 pixel) and the core-to-fractional transition
        radius (``transition_mult * psf_fwhm_pix``). If None, falls back
        to fixed defaults (start at 1 pixel, transition at 5 pixels).
    inner_step_pix : :class:`float`
        Fixed linear step size, in pixels, used in the core
        (``a < a_transition``) regime.
    frac_step : :class:`float`
        Fractional step size used in the outer regime,
        ``a_next = a * (1 + frac_step)``.
    min_pixels_per_annulus : :class:`float`
        Minimum annulus area, in pixels (assuming axis ratio ``ba``),
        enforced by enlarging the step if needed; disabled if 0/None.
    transition_mult : :class:`float`
        Multiple of ``psf_fwhm_pix`` defining the core-to-fractional
        transition radius.

    Returns
    -------
    a_edges : :class:`numpy.ndarray`
        Semi-major-axis edges, in pixels, strictly increasing.
    info : :class:`dict` of :class:`numpy.ndarray`
        Per-annulus diagnostics (length ``len(a_edges) - 1``):
        ``'a_in'``, ``'a_out'``, ``'delta_a'`` (annulus bounds and
        width), ``'core_step'`` (bool, linear-core step),
        ``'frac_step'`` (bool, fractional-step proposal),
        ``'area_limited'`` (bool, step enlarged by the minimum-area
        constraint), plus the scalars ``'a_transition'`` (core to
        fractional switch radius) and ``'a_stop'`` (stop radius).

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
    """Plot diagnostics of a semi-major-axis grid built by
    :func:`build_sma_opt`.

    Notes
    -----
    Currently broken: calls ``build_sma_grid(...)``, a function name
    that does not exist anywhere in this module (the function is named
    :func:`build_sma_opt`). Calling ``qa_sma_grid()`` raises
    ``NameError``; this appears to be stale from a prior rename.

    Returns
    -------
    None

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
    """Build a per-object QA figure comparing each dataset's masked image
    (with fitted isophotes overlaid) to its surface-brightness profile.

    For every object in ``sample``, makes one row per imaging dataset:
    the left column shows the coadded, mask-excluded image with every
    fitted isophote (from ``sbprofiles``) and the reference (moment)
    ellipse overlaid; the right column shows the per-band surface
    brightness profile (always in mag/arcsec^2, computed from
    ``sbprofiles``' ``SB_*``/``SB_ERR_*`` columns) against
    ``SMA**0.25``, with the reference semi-major axis marked. One PNG is
    written per object to
    ``{htmlgalaxydir}/qa-ellipsefit-{SGANAME}.png``.

    Notes
    -----
    Despite its name, ``linear`` does not switch the profile panel
    between linear-flux and magnitude units -- the plotted quantity
    (``mu``) is always ``22.5 - 2.5*log10(SB)`` (a magnitude)
    regardless of this flag. ``linear=True`` only changes the y-axis
    limit calculation: it skips the extra +/-(0.75, 0.5) mag margin and
    the hard 13-34 mag clamp applied when ``linear=False``, and instead
    uses the raw min/max of the plotted magnitudes as axis limits.

    Parameters
    ----------
    data : :class:`dict`
        Per-band image data and metadata for the group mosaic, as
        produced by ``read_multiband``/:func:`SGA.SGA.build_multiband_mask`
        (``opt_wcs``, ``opt_pixscale``, ``opt_bands``, and per-dataset
        ``{dataset}_images``/``{dataset}_models``/``{dataset}_maskbits``/
        ``{dataset}_bands``/``{dataset}_pixscale``/``{dataset}_wcs``).
    sample : :class:`~astropy.table.Table`
        One row per object (``SGANAME``, ``OBJNAME``, ``BX``, ``BY``,
        ``PA_MOMENT``, ``BA_MOMENT``, ``SMA_MOMENT``, and
        ``REFIDCOLUMN``).
    results : :class:`list` of :class:`list` of :class:`~astropy.table.Table`
        Per-dataset, per-object ellipse-fitting result tables, indexed
        ``results[idata][iobj]`` (as returned by :func:`wrap_multifit`).
    sbprofiles : :class:`list` of :class:`list` of :class:`~astropy.table.Table`
        Per-dataset, per-object surface-brightness profile tables
        (``SMA``, ``SB_{filt}``, ``SB_ERR_{filt}`` columns), indexed
        ``sbprofiles[idata][iobj]`` (as returned by :func:`wrap_multifit`).
    unpack_maskbits_function : callable
        Function with the signature of :func:`SGA.SGA.unpack_maskbits`,
        used to unpack each dataset's packed maskbits image into a
        per-band boolean mask.
    MASKBITS : :class:`list` of :class:`dict`
        Per-dataset bit-value dictionaries, indexed to match ``datasets``,
        passed to ``unpack_maskbits_function``.
    REFIDCOLUMN : :class:`str`
        Column name in ``sample`` holding each object's reference ID,
        used in the figure title.
    datasets : :class:`list` of :class:`str`
        Imaging datasets to plot, one row each, in order.
    linear : :class:`bool`
        See Notes -- does not switch plotted units, only affects axis
        limit padding/clamping.
    htmlgalaxydir : :class:`str`, optional
        Output directory for the QA figures.

    Returns
    -------
    None

    """
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    from matplotlib.cm import get_cmap
    from photutils.isophote import EllipseGeometry
    from photutils.aperture import EllipticalAperture

    from SGA.sky import map_bxby
    from SGA.qa import overplot_ellipse, get_norm, sbprofile_colors


    def kill_left_y(ax):
        """Hide the left y-axis of a matplotlib axes (used to pair a
        twin axis showing the same data with different labeling).

        """
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
                  sbthresh, apertures, SGAMASKBITS, mp=1, nmonte=50,
                  seed=42, debug=False):
    """Loop :func:`multifit` over every object and imaging dataset for a
    group, building the semi-major-axis sampling grid for each dataset
    along the way.

    Iterates objects in the outer loop and datasets in the inner loop
    (so some work, e.g. unpacking per-dataset masks, is repeated per
    object rather than shared across objects). For the ``'opt'``
    dataset, builds the pixel-space SMA array via :func:`build_sma_opt`
    from the object's moment geometry; for every other dataset, builds
    it via :func:`build_sma_band`, rescaled from the optical SMA array.

    Notes
    -----
    ``datasets`` must include ``'opt'`` first: the ``ba``,
    ``opt_sma_array_pix``, and ``sma_apertures_arcsec`` locals are only
    assigned inside the ``dataset == 'opt'`` branch, then reused
    unconditionally by every other dataset's branch. Every current call
    site passes ``'opt'`` first, so this ordering dependency is latent
    rather than actually triggered, but it is not enforced or checked
    here.

    Parameters
    ----------
    data : :class:`dict`
        Per-band image data and metadata for the group mosaic (see
        :func:`SGA.SGA.build_multiband_mask`).
    sample : :class:`~astropy.table.Table`
        One row per object (``REFIDCOLUMN`` value, ``BA_MOMENT``,
        ``SMA_MOMENT``).
    datasets : :class:`list` of :class:`str`
        Imaging datasets to fit, in order; must start with ``'opt'``
        (see Notes).
    unpack_maskbits_function : callable
        Function with the signature of :func:`SGA.SGA.unpack_maskbits`,
        used to unpack each dataset's packed maskbits image into a
        per-band boolean mask.
    sbthresh : :class:`list` of :class:`float`
        Surface brightness thresholds (mag/arcsec^2), passed to
        :func:`multifit`.
    apertures : :class:`list` of :class:`float`
        Multiples of the moment semi-major axis defining the reference
        curve-of-growth apertures, passed to :func:`multifit`.
    SGAMASKBITS : :class:`list` of :class:`dict`
        Per-dataset bit-value dictionaries, indexed to match ``datasets``,
        passed to ``unpack_maskbits_function``.
    mp : :class:`int`
        Number of multiprocessing workers, passed to :func:`multifit`.
    nmonte : :class:`int`
        Number of Monte Carlo realizations for uncertainty estimation,
        passed to :func:`multifit`.
    seed : :class:`int`
        Random seed, passed to :func:`multifit`.
    debug : :class:`bool`
        If True, enable debug-mode diagnostics in :func:`multifit`.

    Returns
    -------
    results : :class:`list` of :class:`list` of :class:`~astropy.table.Table`
        Per-dataset, per-object ellipse-fitting result tables, indexed
        ``results[idata][iobj]``.
    sbprofiles : :class:`list` of :class:`list` of :class:`~astropy.table.Table`
        Per-dataset, per-object surface-brightness profile tables,
        indexed ``sbprofiles[idata][iobj]``.

    """
    REFIDCOLUMN = data['REFIDCOLUMN']

    opt_wcs = data['opt_wcs']
    opt_pixscale = data['opt_pixscale']
    nsample = len(sample)

    results_obj = []
    sbprofiles_obj = []

    for iobj, obj in enumerate(sample):
        refid = obj[REFIDCOLUMN]

        t0 = time()
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
                    frac_step=0.15, amax_factor=5.)
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

        dt, unit = get_dt(t0)
        log.info(f'Time for galaxy {iobj+1}/{nsample}: {dt:.3f} {unit}')


    # unpack the SB profiles and results tables
    results = list(zip(*results_obj))       # [ndatasets][nobj]
    sbprofiles = list(zip(*sbprofiles_obj)) # [ndatasets][nobj]

    return results, sbprofiles


def ellipsefit_multiband(galaxy, galaxydir, REFIDCOLUMN, read_multiband_function,
                         unpack_maskbits_function, SGAMASKBITS, region='dr11-south',
                         run='south', mp=1, bands=['g', 'r', 'i', 'z'], pixscale=0.262,
                         galex_pixscale=1.5, unwise_pixscale=2.75, mask_nearby=None,
                         galex=True, unwise=True, use_tractor_position=True, fixgeo=False,
                         tractorgeo=False, use_radial_weight=True, sbthresh=REF_SBTHRESH,
                         apertures=REF_APERTURES, update_geometry=False, nmonte=50, seed=42,
                         verbose=False, skip_tractor=False, skip_ellipse=False, nowrite=False,
                         ignore_galaxy_sources=False, clobber=False, qaplot=False,
                         htmlgalaxydir=None):
    """Top-level driver: read the imaging, build masks, and ellipse-fit
    every galaxy in a group or coadd.

    Reads the group's imaging and catalogs via ``read_multiband_function``
    (e.g. :func:`SGA.SGA.read_multiband`), returning early if there are no
    CCDs touching the brick or the read fails. Unless ``skip_ellipse``,
    runs a two-pass fit: (1) build an aggressive initial mask via
    :func:`SGA.SGA.build_multiband_mask` (``FMAJOR_geo=0.01``,
    ``mask_minor_galaxies=True``) and, unless every object already has
    fixed geometry, fit the optical band only via :func:`wrap_multifit`
    (``nmonte=0``) to get a first-pass R(26) estimate per object via
    :func:`SGA.SGA.SGA_diameter`, merging it into ``SMA_MASK``; (2)
    rebuild the mask with the refined per-object geometry
    (``FMAJOR_final=0.1``, ``mask_minor_galaxies=False``) and run the
    full multi-dataset fit via :func:`wrap_multifit` with the requested
    ``nmonte``. Logs a final isophotal-radius summary per object via
    :func:`SGA.SGA.SGA_diameter`, optionally makes a QA figure via
    :func:`qa_ellipsefit`, and (unless ``nowrite``) writes the results
    via ``SGA.io.write_ellipsefit``.

    When ``skip_ellipse`` is True (RESOLVED or otherwise not
    ellipse-fit objects), no masking or fitting is performed; ``results``
    is instead populated directly from ``results_datamodel`` (empty/
    placeholder values) and ``sbprofiles`` from empty tables, so the
    output data model is still well-formed.

    Notes
    -----
    ``clobber`` is accepted but never referenced in this function's
    body -- clobber/skip-if-done checking is evidently handled upstream
    (e.g. via ``SGA.SGA.missing_files`` in ``bin/SGA2025-mpi``) before
    this function is called. Return-code convention mirrors
    :func:`SGA.SGA.read_multiband`'s ``err``: 0 signals a failure that
    should abort processing for this galaxy, while 1 signals either
    genuine success or one of two "nothing to do" early-return cases (no
    CCDs touching the brick; an entirely empty Tractor catalog after
    masking) -- callers should not treat ``err == 1`` alone as proof that
    fitting actually happened.

    Parameters
    ----------
    galaxy : :class:`str`
        Galaxy or group name; the filename prefix for this object's data
        products.
    galaxydir : :class:`str`
        Directory containing this object's coadd-stage data products.
    REFIDCOLUMN : :class:`str`
        Column name in the sample catalog holding each object's
        reference ID.
    read_multiband_function : callable
        Function with the signature of :func:`SGA.SGA.read_multiband`,
        used to read the imaging and catalogs.
    unpack_maskbits_function : callable
        Function with the signature of :func:`SGA.SGA.unpack_maskbits`,
        passed through to :func:`wrap_multifit` and :func:`qa_ellipsefit`.
    SGAMASKBITS : :class:`list` of :class:`dict`
        Per-dataset bit-value dictionaries (one per entry of the
        internally-built ``datasets`` list); length must equal
        ``len(datasets)``.
    region : :class:`str`
        Survey region, passed to :func:`SGA.SGA.SGA_diameter` for
        region-specific data-quality handling.
    run : :class:`str`
        legacypipe survey run name, passed to ``read_multiband_function``.
    mp : :class:`int`
        Number of multiprocessing workers, passed to
        :func:`SGA.SGA.build_multiband_mask` and :func:`wrap_multifit`.
    bands : :class:`list` of :class:`str`
        Optical bands to read and fit.
    pixscale : :class:`float`
        Optical pixel scale, arcsec/pixel.
    galex_pixscale : :class:`float`
        GALEX pixel scale, arcsec/pixel.
    unwise_pixscale : :class:`float`
        unWISE pixel scale, arcsec/pixel.
    mask_nearby : :class:`list` of :class:`dict`, optional
        Extra ellipses to always mask, passed through to
        :func:`SGA.SGA.build_multiband_mask`.
    galex : :class:`bool`
        If True, include the GALEX FUV/NUV imaging set in ``datasets``.
    unwise : :class:`bool`
        If True, include the unWISE W1-W4 imaging set in ``datasets``.
    use_tractor_position : :class:`bool`
        Passed through to :func:`SGA.SGA.build_multiband_mask`.
    fixgeo : :class:`bool`
        If True, force fixed geometry for every object; also skips the
        first-pass optical-only fit.
    tractorgeo : :class:`bool`
        If True, force Tractor geometry for every object; also skips the
        first-pass optical-only fit.
    use_radial_weight : :class:`bool`
        Passed through to :func:`SGA.SGA.build_multiband_mask`.
    sbthresh : :class:`list` of :class:`float`
        Surface brightness thresholds (mag/arcsec^2) for isophotal radii.
    apertures : :class:`list` of :class:`float`
        Multiples of the moment semi-major axis defining the reference
        curve-of-growth apertures.
    update_geometry : :class:`bool`
        If True, let the final mask/fit pass re-iterate the geometry
        (``niter_geometry=1`` with ``input_geo_initial=None``); if
        False, freeze the geometry at the values derived from the
        first-pass fit (``input_geo_initial`` built explicitly per
        object) before the final mask/fit pass.
    nmonte : :class:`int`
        Number of Monte Carlo realizations for uncertainty estimation in
        the final fit (the first-pass optical-only fit always uses
        ``nmonte=0``).
    seed : :class:`int`
        Random seed for the Monte Carlo realizations.
    verbose : :class:`bool`
        Passed through to ``read_multiband_function`` and
        ``SGA.io.write_ellipsefit``.
    skip_tractor : :class:`bool`
        Passed through to ``read_multiband_function``.
    skip_ellipse : :class:`bool`
        If True, skip masking/fitting entirely and populate a
        placeholder data model instead (see above).
    nowrite : :class:`bool`
        If True, skip writing the output files via
        ``SGA.io.write_ellipsefit``.
    ignore_galaxy_sources : :class:`bool`
        Passed through to :func:`SGA.SGA.build_multiband_mask`; also
        forces the first-pass optical-only fit to be skipped.
    clobber : :class:`bool`
        Unused (see Notes).
    qaplot : :class:`bool`
        If True, generate the QA figure via :func:`qa_ellipsefit`.
    htmlgalaxydir : :class:`str`, optional
        Output directory for QA figures.

    Returns
    -------
    :class:`int`
        Status code; see Notes for the 0/1 convention and its caveats.

    """
    from astropy.table import Table, vstack
    from SGA.util import get_dt
    from SGA.SGA import SGA_diameter, build_multiband_mask

    tall = time()

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
    if not skip_ellipse and not os.path.isfile(ccdsfile):
        log.info('No CCDs touching this brick; nothing to do.')
        return 1

    data, tractor, sample, samplesrcs, err = read_multiband_function(
        galaxy, galaxydir, REFIDCOLUMN, bands=bands, run=run,
        pixscale=pixscale, galex_pixscale=galex_pixscale,
        unwise_pixscale=unwise_pixscale, unwise=unwise, galex=galex,
        verbose=verbose, skip_ellipse=skip_ellipse, skip_tractor=skip_tractor)
    if err == 0:
        log.warning(f'Problem reading (or missing) data for {galaxydir}/{galaxy}')
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

            results_obj.append(results_dataset)
            sbprofiles_obj.append(sbprofiles_dataset)

        results = list(zip(*results_obj))       # [ndatasets][nobj]
        sbprofiles = list(zip(*sbprofiles_obj)) # [ndatasets][nobj]
    else:
        FMAJOR_geo = 0.01
        FMAJOR_final = 0.1

        try:
            err = 1

            # mask aggressively to determine the geometry; use FMAJOR_geo
            # plus mask_minor_galaxies=True (outside the ellipse)
            t0 = time()
            data, sample = build_multiband_mask(
                data, tractor, sample, samplesrcs, qaplot=False, cleanup=False,
                use_tractor_position=use_tractor_position,
                use_radial_weight=use_radial_weight, fixgeo=fixgeo,
                tractorgeo=tractorgeo, mask_nearby=mask_nearby, niter_geometry=2,
                FMAJOR_geo=FMAJOR_geo, mask_minor_galaxies=True,
                ignore_galaxy_sources=ignore_galaxy_sources,
                htmlgalaxydir=htmlgalaxydir, mp=mp)
            dt, unit = get_dt(t0)
            log.info(f'Building the initial multiband mask took {dt:.3f} {unit}')

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

        # First fit just the optical and then update the mask unless
        # skip_initial_fit is set.
        skip_initial_fit = (fixgeo or tractorgeo or
                            ignore_galaxy_sources or
                            all(obj['ELLIPSEMODE'] & (ELLIPSEMODE['FIXGEO'] | ELLIPSEMODE['TRACTORGEO']) != 0
                                for obj in sample)
                            )

        if skip_initial_fit:
            log.info(f'Skipping initial ellipse-fitting.')
        else:
            t0 = time()
            results, sbprofiles = wrap_multifit(
                data, sample, ['opt'], unpack_maskbits_function,
                sbthresh, apertures, [SGAMASKBITS[0]], mp=mp,
                nmonte=0, seed=seed, debug=False)
            dt, unit = get_dt(t0)
            log.info(f'Initial ellipse-fitting took {dt:.3f} {unit}')

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
            if (obj['ELLIPSEMODE'] & (ELLIPSEMODE['FIXGEO'] | ELLIPSEMODE['TRACTORGEO']) != 0) or \
               fixgeo or tractorgeo or skip_initial_fit:
                if not update_geometry:
                    input_geo_initial[iobj, :] = [bx, by, sma_mom/pixscale, ba_mom, pa_mom]
                    log.info(f'Galaxy {iobj+1}/{len(sample)} [{sample["OBJNAME"][iobj]}]: fixed geometry ' + \
                             f'R(26)={obj["SMA_MOMENT"]:.2f} arcsec.')
                continue

            # estimate R(26) from first-pass profiles
            tab = Table(obj['SMA_MOMENT', 'ELLIPSEMODE', 'ELLIPSEBIT', 'SAMPLE'])
            for thresh in sbthresh:
                for filt in bands:
                    col = f'R{thresh:.0f}_{filt.upper()}'
                    colerr = f'R{thresh:.0f}_ERR_{filt.upper()}'
                    tab[col] = results[0][iobj][col]
                    tab[colerr] = results[0][iobj][colerr]
            radius, radius_err, radius_ref, radius_weight = SGA_diameter(
                tab, region, radius_arcsec=True)
            r26_arcsec = float(radius[0])

            # merge R26 with the existing SMA_MASK in arcsec
            sma_moment_arcsec = obj['SMA_MOMENT']
            sma_mask_arcsec = obj['SMA_MASK']

            log.info(f'Galaxy {iobj+1}/{len(sample)} [{sample["OBJNAME"][iobj]}]: Initial estimate ' + \
                     f'R(26)={r26_arcsec:.2f} arcsec [previous ' + \
                     f'sma_mask={sma_mask_arcsec:.2f} arcsec].')
            if sma_mask_arcsec <= 0.:
                sma_mask_arcsec = r26_arcsec
            else:
                sma_mask_arcsec = max(sma_mask_arcsec, r26_arcsec)

            sample['SMA_MASK'][iobj] = sma_mask_arcsec
            if not update_geometry:
                # NB: build_multiband_mask expects sma_moment_arcsec
                # *not* sma_mask_arcsec.
                input_geo_initial[iobj, :] = [bx, by, sma_moment_arcsec/pixscale, ba_mom, pa_mom]

        # pull back on the masking for the final iteration
        t0 = time()
        data, sample = build_multiband_mask(data, tractor, sample, samplesrcs,
                                            input_geo_initial=input_geo_initial,
                                            mask_nearby=mask_nearby, qaplot=qaplot,
                                            FMAJOR_geo=FMAJOR_geo, FMAJOR_final=FMAJOR_final,
                                            mask_minor_galaxies=False,
                                            use_tractor_position=use_tractor_position,
                                            use_radial_weight=use_radial_weight,
                                            fixgeo=fixgeo, tractorgeo=tractorgeo,
                                            niter_geometry=niter_geometry,
                                            ignore_galaxy_sources=ignore_galaxy_sources,
                                            htmlgalaxydir=htmlgalaxydir, mp=mp)
        dt, unit = get_dt(t0)
        log.info(f'Building the final multiband mask took {dt:.3f} {unit}')

        # ellipse-fit over objects and then datasets
        t0 = time()
        results, sbprofiles = wrap_multifit(
            data, sample, datasets, unpack_maskbits_function,
            sbthresh, apertures, SGAMASKBITS, mp=mp,
            nmonte=nmonte, seed=seed, debug=False)#qaplot)
        dt, unit = get_dt(t0)
        log.info(f'Final ellipse-fitting took {dt:.3f} {unit}')

        # nice summary
        for iobj, (res, obj) in enumerate(zip(results[0], sample)):
            res['SAMPLE'] = obj['SAMPLE']
            res['ELLIPSEMODE'] = obj['ELLIPSEMODE']
            res['ELLIPSEBIT'] = obj['ELLIPSEBIT']
            res['SMA_MOMENT'] = obj['SMA_MOMENT']
            log.info(f'Final isophotal radii for galaxy {iobj+1}/{len(sample)}:')
            _ = SGA_diameter(res, region, verbose=True)

        if qaplot:
            qa_ellipsefit(data, sample, results, sbprofiles, unpack_maskbits_function,
                          SGAMASKBITS, REFIDCOLUMN, datasets=datasets,
                          htmlgalaxydir=htmlgalaxydir)

    if not nowrite:
        from SGA.io import write_ellipsefit
        err = write_ellipsefit(data, sample, datasets, results, sbprofiles,
                               SGAMASKBITS, verbose=verbose)

    dt, unit = get_dt(tall)
    log.info(f'Total time for ellipse-fitting: {dt:.3f} {unit}')

    return err
