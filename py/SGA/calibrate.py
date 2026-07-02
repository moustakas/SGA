"""
SGA.calibrate
=============

Code to calibrate Rr(26).

"""
from __future__ import annotations

import os
import json
import warnings
import numpy as np
from astropy.table import Table
from importlib import resources
from dataclasses import dataclass

from SGA.coadds import GRIZ as BANDS
from SGA.logger import log


_CALIB_CACHE = {}

SBTHRESH = [23, 24, 25, 26]

warnings.filterwarnings(
    'ignore',
    message='divide by zero encountered in divide',
    category=RuntimeWarning,
    module='scipy.odr._odrpack')


def clear_calibration_cache():
    """Clear the in-memory calibration cache populated by
    :func:`load_calibration`.

    Returns
    -------
    None

    """
    _CALIB_CACHE.clear()


def _as_float(col):
    """Coerce an astropy Column (or array-like) to a plain
    :class:`numpy.ndarray` of :class:`float`.

    Parameters
    ----------
    col : :class:`~astropy.table.Column` or array-like
        Input column or array.

    Returns
    -------
    :class:`numpy.ndarray`
        Float array with the same values.

    """
    if hasattr(col, 'value'):
        return np.asarray(col.value, dtype=float)
    return np.asarray(col, dtype=float)


def _channel_key(th, band):
    """Build a channel key string from a threshold and band, e.g.
    ``'r26'``, ``'g24'``.

    Parameters
    ----------
    th : :class:`int` or :class:`float`
        Surface-brightness isophotal threshold (e.g. 26).
    band : :class:`str`
        Band letter (any case; lower-cased in the key).

    Returns
    -------
    :class:`str`
        ``f'{band.lower()}{th}'``.

    """
    return f"{band.lower()}{th}"


def _collect_channels_from_table(tbl):
    """Extract per-channel isophotal-radius arrays (linear, arcsec) and
    their errors from an ellipse catalog table.

    Zeros and non-finite values are treated as missing (set to NaN).
    Includes an extra channel ``'moment'`` from column ``SMA_MOMENT`` if
    present.

    Parameters
    ----------
    tbl : :class:`~astropy.table.Table`
        Input table with ``R{TH}_{BAND}``/``R{TH}_ERR_{BAND}`` columns
        for each threshold in ``SBTHRESH`` and band in ``BANDS``, and
        optionally ``SMA_MOMENT``.

    Returns
    -------
    table : :class:`dict`
        Channel name (e.g. ``'r26'``, ``'moment'``) -> array of radii,
        arcsec.
    sigma : :class:`dict`
        Channel name -> array of 1-sigma errors, or None if the
        corresponding ``*_ERR_*`` column is absent.
    N : :class:`int`
        Number of rows in ``tbl``.

    """
    N = len(tbl)
    table = {}
    sigma = {}

    for th in SBTHRESH:
        for band in np.char.upper(BANDS):
            base = f"R{th:.0f}_{band}"
            err = f"R{th:.0f}_ERR_{band}"
            key = _channel_key(th, band)
            if base in tbl.colnames:
                vals = _as_float(tbl[base])
                vals[(~np.isfinite(vals)) | (vals <= 0.0)] = np.nan
                table[key] = vals
                if err in tbl.colnames:
                    e = _as_float(tbl[err])
                    e[(~np.isfinite(e)) | (e <= 0.0)] = np.nan
                    sigma[key] = e
                else:
                    sigma[key] = None

    if "SMA_MOMENT" in tbl.colnames:
        vals = _as_float(tbl["SMA_MOMENT"])
        vals[(~np.isfinite(vals)) | (vals <= 0.0)] = np.nan
        table["moment"] = vals
        sigma["moment"] = None

    return table, sigma, N


@dataclass
class ChannelCalib:
    """Calibration coefficients for a single channel, fit by
    :func:`_fit_channel_bisector_robust` via :func:`calibrate_from_table`.

    The model is ``log(R26) = a + b * log(R_channel) + c . covariates``,
    with intrinsic scatter ``tau`` (log-space).

    Attributes
    ----------
    name : :class:`str`
        Channel key, e.g. ``'r26'``, ``'g24'``, ``'moment'``.
    a : :class:`float`
        Fitted intercept.
    b : :class:`float`
        Fitted slope.
    tau : :class:`float`
        Intrinsic (log-space) scatter, beyond measurement error.
    covar_names : :class:`list`
        Names of the covariates in ``c``, if any. Always empty in the
        SGA-2025 production calibration (no covariates have been used
        to date) -- see the Notes in :func:`calibrate_from_table` for
        what it would take to actually enable this.
    c : :class:`numpy.ndarray`
        Covariate coefficients. Always empty in the SGA-2025 production
        calibration (see :func:`calibrate_from_table`), so
        ``covars_row`` is never actually required by :func:`_infer_one`.
    sigma_obs_default : :class:`float`
        Default (log-space) measurement uncertainty used when an object
        has no per-object error for this channel.

    """
    name: str
    a: float
    b: float
    tau: float
    covar_names: list
    c: np.ndarray
    sigma_obs_default: float


@dataclass
class Calibration:
    """Container for all channel calibrations, as loaded/saved by
    :func:`load_calibration`/:func:`save_calibration`.

    Attributes
    ----------
    channels : :class:`dict`
        Channel name -> :class:`ChannelCalib`.
    target_name : :class:`str`
        Name of the target (anchor) channel the others are calibrated
        against; always ``'r26'`` in the current pipeline.

    """
    channels: dict
    target_name: str = "r26"


def save_calibration(cal, path):
    """Write a :class:`Calibration` to a tab-separated text file.

    One row per channel, columns ``name``, ``a``, ``b``, ``tau``,
    ``sigma_obs_default``, ``covar_names`` (JSON list), ``c_json`` (JSON
    list of covariate coefficients). The model is
    ``log(R26_R) = a + b * log(R_channel)``.

    Parameters
    ----------
    cal : :class:`Calibration`
        Calibration to write.
    path : :class:`str`
        Output file path.

    Returns
    -------
    None

    """
    rows = ["\t".join(["name", "a", "b", "tau", "sigma_obs_default", "covar_names", "c_json"])]
    for ch in cal.channels.values():
        rows.append("\t".join([
            ch.name,
            f"{ch.a:.12g}",
            f"{ch.b:.12g}",
            f"{ch.tau:.12g}",
            "" if ch.sigma_obs_default is None else f"{ch.sigma_obs_default:.12g}",
            json.dumps(ch.covar_names or []),
            json.dumps([float(x) for x in ch.c.tolist()]),
        ]))
    with open(path, "w") as f:
        f.write("\n".join(rows))


def load_calibration(path=None):
    """Load a :class:`Calibration` from a tab-separated text file (see
    :func:`save_calibration`), caching by absolute path.

    Parameters
    ----------
    path : :class:`str`, optional
        Calibration file to read. If None, uses the packaged default at
        ``SGA/data/SGA2025/r26-calibration-coeff.tsv``.

    Returns
    -------
    :class:`Calibration`
        Loaded (or cached) calibration.

    Raises
    ------
    ValueError
        If the file's header doesn't start with ``'name'`` (malformed
        calibration file).

    """
    if path is None:
        path = resources.files("SGA").joinpath("data/SGA2025/r26-calibration-coeff.tsv")

    key = os.path.abspath(str(path))

    if key in _CALIB_CACHE:
        return _CALIB_CACHE[key]

    lines = [ln.strip() for ln in open(path, "r") if ln.strip()]
    if not lines or not lines[0].startswith("name"):
        raise ValueError(f"Malformed calibration file: {path}")

    channels = {}
    for ln in lines[1:]:
        name, a, b, tau, sdef, covars_json, c_json = ln.split("\t")
        channels[name] = ChannelCalib(
            name=name,
            a=float(a),
            b=float(b),
            tau=float(tau),
            covar_names=json.loads(covars_json),
            c=np.array(json.loads(c_json), dtype=float),
            sigma_obs_default=(float(sdef) if sdef != "" else None),
        )

    cal = Calibration(channels=channels, target_name="r26")
    log.debug(f"Read calibration file: {path}")

    _CALIB_CACHE[key] = cal
    return cal


def _to_log_and_sigma(r, sigma_r):
    """Convert a linear radius and its error to log-space.

    Parameters
    ----------
    r : :class:`numpy.ndarray`
        Linear radius, arcsec.
    sigma_r : :class:`numpy.ndarray` or None
        1-sigma linear uncertainty on ``r``; if None, no error is
        propagated.

    Returns
    -------
    y : :class:`numpy.ndarray`
        ``log(r)``.
    sigma_y : :class:`numpy.ndarray` or None
        ``sigma_r / r`` (first-order log-space error propagation), or
        None if ``sigma_r`` is None.

    """
    y = np.log(r)
    if sigma_r is None:
        return y, None
    return y, sigma_r / np.clip(r, 1e-300, None)


def _channel_threshold(name):
    """Return a channel's isophotal surface-brightness threshold.

    Parameters
    ----------
    name : :class:`str`
        Channel key, e.g. ``'r26'``, ``'g24'``, ``'moment'``.

    Returns
    -------
    :class:`int` or None
        Threshold (23, 24, 25, or 26), or None for non-isophotal
        channels (e.g. ``'moment'``).

    """
    if name == "moment":
        return None
    if name == "r26":
        return 26
    try:
        return int(name[-2:])
    except ValueError:
        return None


def _hierarchy_rank(name):
    """Rank a channel for display ordering in
    :func:`_plot_calibration_diagnostics` (lower rank = earlier in the
    panel grid): ``'moment'`` first, then isophotal thresholds 23-26 in
    order, then anything else last.

    Parameters
    ----------
    name : :class:`str`
        Channel key.

    Returns
    -------
    :class:`int`
        Sort rank; 0 for ``'moment'``, 1-4 for thresholds 23-26, 99
        otherwise.

    """
    if name == "moment":
        return 0
    th = _channel_threshold(name)
    if th == 23:
        return 1
    if th == 24:
        return 2
    if th == 25:
        return 3
    if th == 26:
        return 4
    return 99


def _plot_calibration_diagnostics(plot_data, cal, calib_path):
    """Build a multi-panel diagnostic figure of each channel's linear
    radius vs. R26_R, with its fitted calibration curve overlaid.

    One panel per channel present in both ``plot_data`` and
    ``cal.channels``, ordered by :func:`_hierarchy_rank` (ties broken by
    descending total scatter). Each panel shows the raw scatter, the
    fitted ``R26_R = exp(a + b*log(R_channel))`` curve, and a legend
    with the fit coefficients and total scatter
    (``sqrt(tau**2 + sigma_obs_default**2)``). Written to
    ``{calib_path with .tsv replaced by .png}``. No-op if ``plot_data``
    is empty or no channels overlap with ``cal.channels``.

    Parameters
    ----------
    plot_data : :class:`dict`
        Channel name -> ``{'x': array, 'y': array}`` of linear radius
        (channel) and R26_R values, as built by :func:`calibrate_from_table`.
    cal : :class:`Calibration`
        Fitted calibration providing each channel's coefficients.
    calib_path : :class:`str`
        Path of the calibration TSV file; the output PNG is written
        alongside it with the same basename.

    Returns
    -------
    None

    """
    if not plot_data:
        return

    import matplotlib.pyplot as plt

    base, _ = os.path.splitext(str(calib_path))
    png_path = base + ".png"

    scatters = {}
    for name, ch in cal.channels.items():
        sdef = ch.sigma_obs_default or 0.0
        scatters[name] = float(np.sqrt(ch.tau ** 2 + sdef ** 2))

    available = [n for n in plot_data.keys() if n in cal.channels]
    names = sorted(available, key=lambda n: (_hierarchy_rank(n), -scatters.get(n, 0.0)))

    n = len(names)
    if n == 0:
        return

    ncols = 4
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), sharey=True)
    axes_flat = axes.ravel()

    for idx, name in enumerate(names):
        ax = axes_flat[idx]
        d = plot_data[name]
        x = d["x"]
        y = d["y"]

        mask = np.isfinite(x) & np.isfinite(y)
        if not np.any(mask):
            ax.set_title(name)
            ax.set_axis_off()
            continue

        xx = x[mask]
        yy = y[mask]

        ax.scatter(xx, yy, s=4, alpha=0.3)

        ch = cal.channels.get(name)
        if ch is not None:
            a = ch.a
            b = ch.b
            tau = ch.tau
            sdef = ch.sigma_obs_default or 0.0
            s_tot = np.sqrt(tau**2 + sdef**2)

            x_min = np.nanmin(xx)
            x_max = np.nanmax(xx)
            if np.isfinite(x_min) and np.isfinite(x_max) and x_max > x_min:
                x_grid = np.linspace(x_min, x_max, 200)
                xg_pos = np.clip(x_grid, 1e-6, None)
                ylog_fit = a + b * np.log(xg_pos)
                y_fit = np.exp(ylog_fit)
                (line,) = ax.plot(x_grid, y_fit, "k-", lw=1)

                label = f"a={a:.3f}, b={b:.3f}, tau={tau:.3f}, sdef={sdef:.3f}, σ_tot={s_tot:.3f}"
                ax.legend([line], [label], fontsize=7, loc="upper left", frameon=False)

        ax.set_xlabel(f"{name} [arcsec]")
        if idx % ncols == 0:
            ax.set_ylabel("R26_R [arcsec]")
        ax.set_title(f"{name} (σ_tot={s_tot:.3f})")

    for j in range(len(names), len(axes_flat)):
        axes_flat[j].set_axis_off()

    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)

    log.info(f"Wrote calibration diagnostic plot to {png_path}")


def _compute_bisector(xx, yy, eps=1e-12):
    """Compute a least-squares bisector fit for ``y = a + b*x``.

    Uses the Isobe et al. 1990 bisector method: fits the ordinary
    least-squares regressions of ``y`` on ``x`` and of ``x`` on ``y``,
    then takes the line bisecting the two slopes in angle space.

    Parameters
    ----------
    xx, yy : :class:`numpy.ndarray`
        Data to fit, same length.
    eps : :class:`float`
        Numerical floor used when the ``x`` on ``y`` slope is
        near-degenerate (``|denom| < eps``), in which case the
        intersection point falls back to the sample means.

    Returns
    -------
    a : :class:`float`
        Bisector intercept.
    b : :class:`float`
        Bisector slope.

    """
    A1 = np.column_stack([np.ones_like(xx), xx])
    coef1 = np.linalg.lstsq(A1, yy, rcond=None)[0]
    a1, b1 = coef1[0], coef1[1]

    A2 = np.column_stack([np.ones_like(yy), yy])
    coef2 = np.linalg.lstsq(A2, xx, rcond=None)[0]
    a2x, b2x = coef2[0], coef2[1]

    if b2x == 0:
        return a1, b1

    m1 = b1
    m2 = 1.0 / b2x

    denom = m1 - m2
    if abs(denom) < eps:
        x0 = np.mean(xx)
        y0 = np.mean(yy)
    else:
        x0 = ((-a2x / b2x) - a1) / denom
        y0 = a1 + m1 * x0

    theta1 = np.arctan(m1)
    theta2 = np.arctan(m2)
    mb = np.tan(0.5 * (theta1 + theta2))
    a = y0 - mb * x0
    b = mb

    return a, b


def _fit_channel_bisector_robust(x_log, sx_log, y_log, sy_log, Z,
                                  r26_min_anchor=15.0, clip_sigma=3.0, max_iter=5):
    """Robust symmetric fit for ``y = a + b*x`` in log-space, with
    iterative sigma-clipping.

    Fits the bisector (:func:`_compute_bisector`) between ``x_log`` and
    ``y_log``, restricted to points with ``y_log >= log(r26_min_anchor)``
    if given; iteratively re-fits after clipping points whose orthogonal
    residual exceeds ``clip_sigma`` robust-sigma (MAD-based), up to
    ``max_iter`` times or until the clip mask stops changing. Falls back
    to ordinary least squares if fewer than 3 points survive clipping.
    The intrinsic scatter ``tau`` is estimated by subtracting the mean
    measurement variance (from ``sx_log``/``sy_log``) from the total
    residual scatter, floored at 0; the default per-object uncertainty
    ``sigma_obs_default`` falls back to the median ``sx_log`` (if
    available) or an inflated residual scale otherwise.

    Notes
    -----
    ``Z`` is accepted but not yet referenced anywhere in this function's
    body. Covariate support is scaffolded through the calling API
    (:func:`calibrate_from_table`'s ``covariates``/``covariate_names``
    parameters) but the covariate term of the fit is not yet
    implemented here -- adding it is future work, not a bug to fix; see
    the Notes in :func:`calibrate_from_table` for the full picture and
    confirmation that no SGA-2025 production calibration has used
    covariates to date.

    Parameters
    ----------
    x_log, y_log : :class:`numpy.ndarray`
        Log-space channel and anchor (R26) values.
    sx_log, sy_log : :class:`numpy.ndarray` or None
        Log-space 1-sigma uncertainties on ``x_log``/``y_log``; either
        may be None.
    Z : :class:`numpy.ndarray` or None
        Unused (see Notes).
    r26_min_anchor : :class:`float`, optional
        Minimum anchor value (linear arcsec) below which points are
        excluded from the fit; disabled if None or <= 0.
    clip_sigma : :class:`float`
        Sigma-clipping threshold on orthogonal residuals, in robust
        (MAD-based) sigma units.
    max_iter : :class:`int`
        Maximum number of sigma-clipping iterations.

    Returns
    -------
    beta : :class:`numpy.ndarray`
        ``[a, b]`` bisector fit coefficients; ``[0., 1.]`` if fewer than
        3 finite (and, if applicable, anchor-cut) points are available.
    tau : :class:`float`
        Estimated intrinsic (log-space) scatter, floored at 0.
    sigma_obs_default : :class:`float`
        Default log-space measurement uncertainty for objects missing a
        per-object error in this channel.

    """
    eps = 1e-12

    m = np.isfinite(x_log) & np.isfinite(y_log)

    if r26_min_anchor is not None and r26_min_anchor > 0.0:
        y_min_log = np.log(r26_min_anchor)
        m &= (y_log >= y_min_log)

    x = x_log[m]
    y = y_log[m]
    if x.size < 3:
        return np.array([0.0, 1.0], dtype=float), 0.0, 0.1

    mask = np.ones_like(x, dtype=bool)

    for _ in range(max_iter):
        xx = x[mask]
        yy = y[mask]
        if xx.size < 3:
            break

        a, b = _compute_bisector(xx, yy, eps)

        denom = np.sqrt(1.0 + b * b)
        r_orth = (y - (a + b * x)) / (denom + eps)

        mad = np.nanmedian(np.abs(r_orth[mask] - np.nanmedian(r_orth[mask])))
        sigma = 1.4826 * mad if np.isfinite(mad) and mad > 0 else np.nanstd(r_orth[mask])

        if not np.isfinite(sigma) or sigma == 0.0:
            break

        new_mask = np.abs(r_orth) <= clip_sigma * sigma

        if new_mask.sum() == mask.sum():
            mask = new_mask
            break

        mask = new_mask

    xx = x[mask]
    yy = y[mask]
    if xx.size < 3:
        A = np.column_stack([np.ones_like(x), x])
        a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    else:
        a, b = _compute_bisector(xx, yy, eps)

    beta = np.array([float(a), float(b)], dtype=float)

    y_pred = beta[0] + beta[1] * x
    resid_y = y - y_pred

    meas_var = np.zeros_like(resid_y)
    if sy_log is not None:
        sy = sy_log[m]
        meas_var += np.where(np.isfinite(sy), sy**2, 0.0)
    if sx_log is not None:
        sx = sx_log[m]
        meas_var += (beta[1] ** 2) * np.where(np.isfinite(sx), sx**2, 0.0)

    mad_y = np.nanmedian(np.abs(resid_y[mask] - np.nanmedian(resid_y[mask])))
    resid_scale_y = 1.4826 * mad_y if np.isfinite(mad_y) and mad_y > 0 else np.nanstd(resid_y[mask])

    mean_meas_var = np.nanmean(meas_var[mask]) if np.isfinite(np.nanmean(meas_var[mask])) else 0.0
    tau2 = max(0.0, resid_scale_y**2 - mean_meas_var)
    tau = float(np.sqrt(tau2))

    if sx_log is not None:
        sx = sx_log[m]
        med_sx = np.nanmedian(sx[np.isfinite(sx)])
        if np.isfinite(med_sx) and med_sx > 0:
            sdef = float(med_sx)
        else:
            # Conservatively use full residual scale (not subtracting tau^2)
            # when per-object errors are missing
            sdef = float(resid_scale_y / (abs(beta[1]) + 1e-9))
    else:
        # Conservatively use full residual scale (not subtracting tau^2)
        # when per-object errors are missing
        sdef = float(resid_scale_y / (abs(beta[1]) + 1e-9))

    return beta, tau, sdef


def calibrate_from_table(tbl, calib_path, covariates=None, covariate_names=None,
                         r26_min_anchor=10.):
    """Fit R26 calibration coefficients from a table of isophotal radii.

    Uses objects with valid R26_R measurements as anchors to establish the
    relationship ``log(R26) = a + b*log(R_channel)`` for each photometric
    channel. Fitting uses robust bisector regression (Isobe et al. 1990)
    with iterative sigma-clipping to reject outliers (see
    :func:`_fit_channel_bisector_robust`). Writes the resulting
    :class:`Calibration` to ``calib_path`` (:func:`save_calibration`) and
    a diagnostic figure alongside it (:func:`_plot_calibration_diagnostics`).

    Parameters
    ----------
    tbl : :class:`~astropy.table.Table`
        Input table with columns ``R{TH}_{BAND}`` and
        ``R{TH}_ERR_{BAND}`` for each threshold (23, 24, 25, 26) and band
        (G, R, I, Z). Must include ``R26_R`` and ``R26_ERR_R`` for anchor
        selection. May include ``SMA_MOMENT``.
    calib_path : :class:`str`
        Output path for the TSV calibration file.
    covariates : :class:`numpy.ndarray`, optional
        Additional covariates for the fit. Accepted but not actually
        applied -- see Notes.
    covariate_names : :class:`list` of :class:`str`, optional
        Names for the covariates; stored on each resulting
        :class:`ChannelCalib` but otherwise unused (see Notes).
    r26_min_anchor : :class:`float`, optional
        Minimum R26_R radius (arcsec) for anchor selection. Objects with
        smaller R26 are excluded from calibration. Default 10 arcsec.

    Returns
    -------
    :class:`Calibration`
        Calibration object containing fitted coefficients for each channel.

    Notes
    -----
    The calibration model is::

        log(R26) = a + b * log(R_channel) + tau * N(0,1)

    where ``tau`` is the intrinsic scatter. For each channel, the output
    includes ``a``, ``b`` (regression coefficients), ``tau`` (intrinsic
    scatter in log-space), and ``sigma_obs_default`` (default measurement
    uncertainty for objects missing per-object errors).

    The ``covariates``/``covariate_names`` parameters are scaffolding
    for a possible future extension of the model to
    ``log(R26) = a + b*log(R_channel) + c . covariates`` (e.g. to
    absorb an inclination- or surface-brightness-dependent term into
    the channel calibration). That extension is not yet implemented: a
    local ``Z`` array is computed from ``covariates`` but the call to
    :func:`_fit_channel_bisector_robust` always passes ``Z=None``
    regardless, and that function's own ``Z`` parameter is unused in
    its body in any case. As a result ``beta`` from the fit only ever
    has 2 elements (``a``, ``b``), so every :class:`ChannelCalib.c`
    produced here is an empty array, and the covariate term in
    :func:`_infer_one` is correspondingly never exercised.

    Confirmed unused in the SGA-2025 production pipeline: neither
    ``archive/bin-SGA2025/SGA2025-calibrate-r26`` (the calibration
    driver script) nor :func:`SGA.SGA.SGA_diameter` (the production
    caller of :func:`infer_best_r26`) ever passes ``covariates``, and
    every row of the shipped
    ``py/SGA/data/SGA2025/r26-calibration-coeff.tsv`` has
    ``covar_names = []``/``c_json = []``. To actually use covariates,
    the fit in :func:`_fit_channel_bisector_robust` would need a real
    multivariate regression against ``Z`` (currently it only performs
    the univariate bisector fit); the plumbing for storing/applying the
    resulting coefficients is already in place in
    :class:`ChannelCalib`/:func:`_infer_one`.

    """
    table, sigma, N = _collect_channels_from_table(tbl)

    if "r26" not in table:
        raise ValueError("Input table lacks R26_R column required for calibration.")

    y_lin = table["r26"]
    sy_lin = sigma.get("r26", None)

    anchor_mask = np.isfinite(y_lin) & (y_lin > 0.0)
    if sy_lin is not None:
        anchor_mask &= np.isfinite(sy_lin) & (sy_lin > 0.0)

    n_anchor = int(np.sum(anchor_mask))
    if n_anchor < 100:
        log.warning(f"small anchor set size = {n_anchor}")

    y_log, sy_log = _to_log_and_sigma(y_lin[anchor_mask],
                                       None if sy_lin is None else sy_lin[anchor_mask])

    plot_data = {}
    channels = {}

    for name, x_lin_full in table.items():
        if name == "r26":
            continue
        x_lin = x_lin_full[anchor_mask]
        if np.all(~np.isfinite(x_lin)):
            continue

        sx_lin_full = sigma.get(name, None)
        sx_lin = None if sx_lin_full is None else sx_lin_full[anchor_mask]
        x_log, sx_log = _to_log_and_sigma(x_lin, sx_lin)

        m = np.isfinite(x_log)
        if not np.any(m):
            continue

        plot_data[name] = {"x": x_lin[m], "y": y_lin[anchor_mask][m]}

        Z = None
        if covariates is not None:
            Z = np.asarray(covariates, dtype=float)[anchor_mask, :]

        # Pass full arrays; _fit_channel_bisector_robust applies its own masking
        beta, tau, sdef = _fit_channel_bisector_robust(
            x_log, sx_log, y_log, sy_log, Z=None,
            r26_min_anchor=r26_min_anchor, clip_sigma=3.0, max_iter=5)

        a = float(beta[0])
        b = float(beta[1])
        c = beta[2:].copy() if len(beta) > 2 else np.array([], dtype=float)

        channels[name] = ChannelCalib(
            name=name, a=a, b=b, tau=float(tau),
            covar_names=(covariate_names or []), c=c, sigma_obs_default=sdef)

        log.info(f"Calibrated {name:>6s}: a={a:.4f}  b={b:.4f}  tau={tau:.4f}  sdef={sdef:.4f}  (N={np.sum(m)})")

    cal = Calibration(channels=channels, target_name="r26")
    save_calibration(cal, calib_path)
    log.info(f"Wrote calibration to {calib_path}  (channels={len(channels)})")

    _plot_calibration_diagnostics(plot_data, cal, calib_path)

    return cal


def _infer_one(measurements, sigmas, cal, var_floor_log=1e-4, covars_row=None,
               include_direct_r26=True):
    """Combine calibrated channel estimates to infer ``log(R26)`` for a
    single object.

    If ``include_direct_r26`` and a valid direct ``'r26'`` measurement
    with a valid error is available, includes it directly (not run
    through a channel calibration). Otherwise, identifies the deepest
    available isophotal threshold and applies each calibrated channel
    at that threshold (see :func:`ChannelCalib`) to predict
    ``log(R26)``, propagating measurement error and adding the
    channel's intrinsic scatter ``tau`` in quadrature. Falls back to
    non-threshold channels (e.g. ``'moment'``) only if no isophotal
    channel estimate could be formed. Combines all resulting estimates
    by inverse-variance weighting.

    Parameters
    ----------
    measurements : :class:`dict`
        Channel name -> linear radius (arcsec) for this object, as
        extracted from :func:`_collect_channels_from_table` (single-row
        slice).
    sigmas : :class:`dict`
        Channel name -> linear 1-sigma error (arcsec), or None.
    cal : :class:`Calibration`
        Calibration providing each channel's coefficients.
    var_floor_log : :class:`float`
        Minimum log-space variance floor applied to every channel
        estimate (including the direct ``'r26'`` term), to prevent a
        single near-zero-error measurement from dominating the
        inverse-variance combination.
    covars_row : :class:`numpy.ndarray`, optional
        Covariate vector for this object, required only if a channel's
        ``ChannelCalib.c`` is non-empty (in the current pipeline this
        never happens -- see the Notes in :func:`calibrate_from_table`
        -- so this parameter has no practical effect today).
    include_direct_r26 : :class:`bool`
        If True, include the direct ``'r26'`` measurement (when valid)
        in the inverse-variance combination alongside calibrated
        estimates from other channels.

    Returns
    -------
    y_hat : :class:`float`
        Inverse-variance-weighted estimate of ``log(R26)``; ``numpy.nan``
        if no channel estimate could be formed.
    sigma_y : :class:`float`
        1-sigma uncertainty on ``y_hat``; ``numpy.nan`` if no estimate.
    w_dict : :class:`dict`
        Channel name -> inverse-variance weight used in the combination;
        empty if no estimate.

    """
    y_list, v_list, w_dict = [], [], {}

    th_available = []
    for name in measurements.keys():
        th = _channel_threshold(name)
        if th is not None:
            th_available.append(th)
    deepest_th = max(th_available) if th_available else None

    if include_direct_r26 and "r26" in measurements:
        x_lin = measurements["r26"]
        sx_lin = sigmas.get("r26", None)
        if (np.isfinite(x_lin) and x_lin > 0.0 and sx_lin is not None
            and np.isfinite(sx_lin) and sx_lin > 0.0):
            yj = float(np.log(x_lin))
            var_j = float((sx_lin / x_lin) ** 2)
            var_j = max(var_j, var_floor_log)
            w = 1.0 / var_j
            y_list.append(yj)
            v_list.append(var_j)
            w_dict["r26"] = w

    def _add_channel(name):
        """Apply channel ``name``'s calibration to ``measurements``/
        ``sigmas`` and append the resulting ``log(R26)`` estimate (with
        its variance) to the enclosing ``y_list``/``v_list``/``w_dict``,
        for :func:`_infer_one`.

        No-op if the channel's measurement is missing, non-finite, or
        non-positive.

        """
        ch = cal.channels[name]
        x_lin = measurements[name]
        if not np.isfinite(x_lin) or x_lin <= 0.0:
            return

        x_log = float(np.log(x_lin))
        sx_lin = sigmas.get(name, None)

        if sx_lin is None and ch.sigma_obs_default is not None:
            sx_log = ch.sigma_obs_default
        elif sx_lin is None:
            sx_log = None
        else:
            sx_log = float(sx_lin / x_lin)

        z_term = 0.0
        if ch.c.size > 0:
            if covars_row is None or len(covars_row) != len(ch.c):
                raise ValueError(f"Covariate vector required (len {len(ch.c)}) for channel {name}.")
            z_term = float(np.dot(ch.c, covars_row))

        # Forward model: log(R26) = a + b * log(R_channel) + z_term
        yj = ch.a + ch.b * x_log + z_term

        # Variance: Var(y) = b^2 * Var(x) + tau^2
        var_j = (ch.b**2) * (sx_log**2 if sx_log else 0.0) + ch.tau**2
        var_j = max(var_j, var_floor_log)

        w = 1.0 / var_j
        y_list.append(yj)
        v_list.append(var_j)
        w_dict[name] = w

    # First pass: only calibrated isophotal channels at deepest threshold
    if deepest_th is not None:
        for name, ch in cal.channels.items():
            if name not in measurements:
                continue
            th = _channel_threshold(name)
            if th is None:
                continue
            if th < deepest_th:
                continue
            _add_channel(name)

    # Fallback: use non-threshold channels like 'moment' as last resort
    if not y_list:
        for name, ch in cal.channels.items():
            if name not in measurements:
                continue
            th = _channel_threshold(name)
            if th is not None:
                continue
            _add_channel(name)

    if not y_list:
        return np.nan, np.nan, {}

    w_arr = 1.0 / np.asarray(v_list)
    y_arr = np.asarray(y_list)

    y_hat = float(np.sum(w_arr * y_arr) / np.sum(w_arr))
    sigma_y = float(np.sqrt(1.0 / np.sum(w_arr)))

    return y_hat, sigma_y, w_dict


def infer_best_r26(tbl, calib_path=None, covariates=None, add_columns=False,
                   include_direct_r26=True):
    """Infer D26 diameters by applying calibration to available channels.

    For each object, applies the calibration model
    ``log(R26) = a + b*log(R_channel)`` to all available channels at the
    deepest isophotal threshold (see :func:`_infer_one`), then combines
    estimates using inverse-variance weighting.

    Parameters
    ----------
    tbl : :class:`~astropy.table.Table`
        Input table with columns ``R{TH}_{BAND}`` and optionally
        ``R{TH}_ERR_{BAND}``. May include ``SMA_MOMENT`` as a fallback
        channel.
    calib_path : :class:`str`, optional
        Path to calibration TSV file. If None, uses the packaged default.
    covariates : :class:`numpy.ndarray`, optional
        Covariate values, for a calibration fit with covariates (see the
        Notes in :func:`calibrate_from_table`). Raises
        :class:`ValueError` if the calibration expects covariates
        (``ChannelCalib.c`` non-empty for any channel) and none are
        given here; in practice this check never triggers against the
        SGA-2025 production calibration, since covariate fitting is not
        yet implemented in :func:`calibrate_from_table` (no shipped
        calibration populates ``c``).
    add_columns : :class:`bool`, optional
        If True, add ``D26``, ``D26_ERR``, ``D26_REF``, ``D26_WEIGHT``
        columns to ``tbl``.
    include_direct_r26 : :class:`bool`, optional
        If True (default) and R26_R is available with valid errors, include
        the direct R26_R measurement in the inverse-variance weighted
        combination alongside calibrated estimates from other bands (i26, z26,
        g26). If False, use only the calibrated estimates from non-r bands.

    Returns
    -------
    D26 : :class:`numpy.ndarray`
        Inferred diameter at 26 mag/arcsec^2 in arcmin.
    D26_ERR : :class:`numpy.ndarray`
        1-sigma uncertainty on D26 in arcmin.
    D26_REF : :class:`numpy.ndarray`
        Channel name that contributed highest weight for each object.
    D26_WEIGHT : :class:`numpy.ndarray`
        Weight of the highest-contributing channel.

    Raises
    ------
    ValueError
        If the calibration expects covariates but ``covariates`` is None
        (see above; not reachable with calibrations from
        :func:`calibrate_from_table`).

    Notes
    -----
    Channel selection logic:

    1. Identify the deepest available isophotal threshold for each object.
    2. Use only channels at that threshold (e.g., if r25 exists but r26
       doesn't, use all *25 channels but not *24 or *23).
    3. If no isophotal channels are available, fall back to the
       ``'moment'`` channel.
    4. Combine estimates via inverse-variance weighting.

    The variance for each channel estimate includes both measurement error
    (propagated through the calibration slope) and intrinsic scatter (tau).

    """
    cal = load_calibration(calib_path)
    table, sigma, N = _collect_channels_from_table(tbl)

    D26 = np.full(N, np.nan, dtype=np.float32)
    D26_ERR = np.full(N, np.nan, dtype=np.float32)
    D26_REF = np.array([""] * N, dtype="U12")
    D26_WEIGHT = np.full(N, np.nan, dtype=np.float32)

    any_cov = any((ch.c.size > 0) for ch in cal.channels.values())
    if any_cov and covariates is None:
        raise ValueError("Calibration expects covariates, but none were provided.")

    for i in range(N):
        meas_i = {}
        sig_i = {}
        for name, arr in table.items():
            xi = arr[i]
            if np.isfinite(xi) and xi > 0.0:
                meas_i[name] = float(xi)
                si_arr = sigma.get(name, None)
                if si_arr is None:
                    sig_i[name] = None
                else:
                    si = si_arr[i]
                    sig_i[name] = None if (not np.isfinite(si) or si <= 0.0) else float(si)
        zi = None if not any_cov else np.asarray(covariates[i, :], dtype=float)

        y, sy, wdict = _infer_one(meas_i, sig_i, cal, covars_row=zi,
                                   include_direct_r26=include_direct_r26)

        if np.isfinite(y):
            R = np.exp(y)
            sR = R * sy
            D = (2.0 * R) / 60.0
            sD = (2.0 * sR) / 60.0
            D26[i] = np.float32(D)
            D26_ERR[i] = np.float32(sD)
            if wdict:
                kmax = max(wdict, key=wdict.get)
                D26_REF[i] = kmax
                D26_WEIGHT[i] = np.float32(wdict[kmax])

    if add_columns:
        for colname in ("D26", "D26_ERR", "D26_REF", "D26_WEIGHT"):
            if colname in tbl.colnames:
                tbl.remove_column(colname)
        tbl["D26"] = D26.astype(np.float32)
        tbl["D26_ERR"] = D26_ERR.astype(np.float32)
        tbl["D26_REF"] = D26_REF
        tbl["D26_WEIGHT"] = D26_WEIGHT.astype(np.float32)

    return D26, D26_ERR, D26_REF, D26_WEIGHT
