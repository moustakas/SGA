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
    """Clear the in-memory calibration cache."""
    _CALIB_CACHE.clear()


def _as_float(col):
    """Return numpy float array from an astropy Column."""
    if hasattr(col, 'value'):
        return np.asarray(col.value, dtype=float)
    return np.asarray(col, dtype=float)


def _channel_key(th, band):
    """Return channel key string, e.g. 'r26', 'g24'."""
    return f"{band.lower()}{th}"


def _collect_channels_from_table(tbl):
    """Extract per-channel arrays (linear radii in arcsec) and their errors.

    Zeros and non-finite values are treated as missing (set to NaN).
    Includes an extra channel 'moment' from column 'SMA_MOMENT' if present.

    Returns
    -------
    table : dict
        Channel name -> array of radii (arcsec)
    sigma : dict
        Channel name -> array of 1-sigma errors or None
    N : int
        Number of rows in input table
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
    """Calibration coefficients for a single channel."""
    name: str
    a: float
    b: float
    tau: float
    covar_names: list
    c: np.ndarray
    sigma_obs_default: float


@dataclass
class Calibration:
    """Container for all channel calibrations."""
    channels: dict
    target_name: str = "r26"


def save_calibration(cal, path):
    """Write calibration to TSV file.

    Columns: name, a, b, tau, sigma_obs_default, covar_names, c_json

    The model is: log(R26_R) = a + b * log(R_channel)
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
    """Load calibration from TSV file with caching.

    If path is None, uses the packaged default under SGA/data/SGA2025.
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
    """Convert linear radius and error to log-space."""
    y = np.log(r)
    if sigma_r is None:
        return y, None
    return y, sigma_r / np.clip(r, 1e-300, None)


def _channel_threshold(name):
    """Return isophotal threshold (23, 24, 25, 26) or None for non-isophotal."""
    if name == "moment":
        return None
    if name == "r26":
        return 26
    try:
        return int(name[-2:])
    except ValueError:
        return None


def _hierarchy_rank(name):
    """Rank channels for display ordering. Lower rank = earlier in grid."""
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
    """Make multi-panel diagnostic plot of channel vs R26_R with fits."""
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
    """Compute least-squares bisector fit for y = a + b*x.

    Uses the Isobe+ 1990 bisector method: average of Y|X and X|Y regressions.

    Returns (a, b) intercept and slope.
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
    """Robust symmetric fit for y = a + b*x in log-space.

    Uses least-squares bisector with iterative sigma clipping on orthogonal
    residuals. Covariates Z are currently ignored.

    Returns (beta, tau, sigma_obs_default) where beta = [a, b].
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
            sdef = float(resid_scale_y / (abs(beta[1]) + 1e-9))
    else:
        sdef = float(resid_scale_y / (abs(beta[1]) + 1e-9))

    return beta, tau, sdef


def calibrate_from_table(tbl, calib_path, covariates=None, covariate_names=None,
                         r26_min_anchor=10.):
    """Fit calibration from table with valid R26_R measurements.

    Identifies anchor objects (valid R26_R and R26_ERR_R > 0), fits each channel
    via robust bisector regression, and writes calibration TSV.

    Returns Calibration object.
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
            Z = Z[m, :]
        yl = y_log[m]
        syl = None if sy_log is None else sy_log[m]

        beta, tau, sdef = _fit_channel_bisector_robust(
            x_log[m], None if sx_log is None else sx_log[m], yl, syl, Z=None,
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
    """Combine channels to infer log(R_r26) for a single object.

    Uses inverse-variance weighting of calibrated channel estimates.
    Falls back to 'moment' channel only if no isophotal channels available.

    Returns (y_hat, sigma_y, weight_dict).
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
    """Apply calibration to infer D26 diameters for all objects.

    Returns (D26, D26_ERR, D26_REF, D26_WEIGHT) arrays in arcmin.
    D26_REF indicates which channel had highest weight.
    Optionally adds these as columns to the input table.
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
