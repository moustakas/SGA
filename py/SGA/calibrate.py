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
from typing import Dict, List, Optional, Tuple
from astropy.table import Table
from scipy import odr
from sklearn.linear_model import HuberRegressor
from importlib import resources

from SGA.coadds import GRIZ as BANDS
from SGA.logger import log

#from SGA.SGA import SBTHRESH
SBTHRESH = [23, 24, 25, 26]
#SBTHRESH = [24, 25, 26]

from dataclasses import dataclass


warnings.filterwarnings(
    'ignore',
    message='divide by zero encountered in divide',
    category=RuntimeWarning,
    module='scipy.odr._odrpack'
)


def _as_float(col) -> np.ndarray:
    """Return numpy float array from an astropy Column; prefers .value if available."""
    try:
        return np.asarray(col.value, dtype=float)
    except Exception:
        return np.asarray(col, dtype=float)


def _channel_key(th: int, band: str) -> str:
    return f"{band.lower()}{th}"  # e.g., 'r26', 'g24'


def _collect_channels_from_table(tbl: Table) -> Tuple[Dict[str, np.ndarray], Dict[str, Optional[np.ndarray]], int]:
    """Extract per-channel arrays (linear radii in arcsec) and their
    1σ errors, with zeros treated as missing.

    Returns (table_dict, sigma_dict, N).

    Includes an extra channel 'moment' from column 'SMA_MOMENT' (no σ
    column).

    """
    N = len(tbl)
    table: Dict[str, np.ndarray] = {}
    sigma: Dict[str, Optional[np.ndarray]] = {}

    # Isophotal channels
    for th in SBTHRESH:
        for band in np.char.upper(BANDS):
            base = f"R{th:.0f}_{band}"
            err = f"R{th:.0f}_ERR_{band}"
            key = _channel_key(th, band)  # e.g., r26
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

    # Moment channel
    if "SMA_MOMENT" in tbl.colnames:
        vals = _as_float(tbl["SMA_MOMENT"])
        vals[(~np.isfinite(vals)) | (vals <= 0.0)] = np.nan
        table["moment"] = vals
        sigma["moment"] = None

    return table, sigma, N


@dataclass
class ChannelCalib:
    name: str
    a: float
    b: float
    tau: float
    covar_names: List[str]
    c: np.ndarray                  # shape (K,) possibly empty
    sigma_obs_default: Optional[float]  # default σ_log(x) if per-object σ missing


@dataclass
class Calibration:
    channels: Dict[str, ChannelCalib]
    target_name: str = "r26"       # fixed target


def save_calibration(cal: Calibration, path: str) -> None:
    """Write ASCII TSV with columns: name a b tau sigma_obs_default
    covar_names c_json.

    Model:
      logR26,r = = a + b log R25,g + c1 logq + c2 (g−r)

    name - The channel name (e.g., r25, g24, moment, etc.)
    a - Intercept in the log–log linear mapping y = a + bx + c^T z
    b - Slope multiplying the predictor x = log R_X (threshold)
    tau - Extra (“intrinsic”) scatter term for that channel in log-space
    sigma_obs_default - Default uncertainty in log(x) if no per-object σ is available
    covar_names - JSON list of covariate names (if you fit with additional regressors)
    c_json - JSON list of the numerical coefficients c c corresponding to those covariates

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


def load_calibration(path: Optional[str] = None) -> Calibration:
    """Load a Calibration object from a TSV file.
    If no path is provided, use the packaged default under SGA/data/SGA2025.

    """
    if path is None:
        path = resources.files("SGA").joinpath("data/SGA2025/r26-calibration-coeff.tsv")
        log.debug(f"Read calibration file: {path}")

    lines = [ln.strip() for ln in open(path, "r") if ln.strip()]
    if not lines or not lines[0].startswith("name"):
        raise ValueError(f"Malformed calibration file: {path}")
    channels: Dict[str, ChannelCalib] = {}
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

    return Calibration(channels=channels, target_name="r26")


def _to_log_and_sigma(r: np.ndarray, sigma_r: Optional[np.ndarray]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    y = np.log(r)
    if sigma_r is None:
        return y, None
    return y, sigma_r / np.clip(r, 1e-300, None)  # σ_log ≈ σ_R / R


def _channel_threshold(name: str) -> Optional[int]:
    """Return isophotal threshold (e.g. 23,24,25,26) or None for non-isophotal."""
    if name == "r26":
        return 26
    if name == "moment":
        return None
    # expect names like 'g24', 'r25', 'z26', etc.
    try:
        return int(name[-2:])
    except ValueError:
        return None


def _hierarchy_rank(name: str) -> int:
    """
    Rank channels for diagnostic display and combination logic.

    Lower rank = earlier in grid (upper-left).
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

    # Any weird/unclassified channels go last
    return 99


def _plot_calibration_diagnostics(
    plot_data: Dict[str, Dict[str, np.ndarray]],
    cal: Calibration,
    calib_path: str,
) -> None:
    """Make a multi-panel diagnostic plot of channel vs R26_R with fits."""
    if not plot_data:
        return

    import matplotlib.pyplot as plt
    import os
    import numpy as np

    base, _ = os.path.splitext(str(calib_path))
    png_path = base + ".png"

    # Compute total scatter per channel from calibration:
    # σ_tot = sqrt(tau^2 + sigma_obs_default^2)
    scatters: Dict[str, float] = {}
    for name, ch in cal.channels.items():
        sdef = ch.sigma_obs_default or 0.0
        scatters[name] = float(np.sqrt(ch.tau ** 2 + sdef ** 2))

    # Channels we can actually plot
    available = [n for n in plot_data.keys() if n in cal.channels]

    # Sort by (hierarchy rank, descending scatter)
    names = sorted(
        available,
        key=lambda n: (_hierarchy_rank(n), -scatters.get(n, 0.0)),
    )

    n = len(names)
    if n == 0:
        return

    ncols = 4
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(4 * ncols, 4 * nrows),
        sharey=True,
        #squeeze=False
    )
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

            # Model: log R26_R = a + b log x  => R26_R(x) = exp(a + b log x)
            x_min = np.nanmin(xx)
            x_max = np.nanmax(xx)
            if np.isfinite(x_min) and np.isfinite(x_max) and x_max > x_min:
                x_grid = np.linspace(x_min, x_max, 200)
                xg_pos = np.clip(x_grid, 1e-6, None)
                ylog_fit = a + b * np.log(xg_pos)
                y_fit = np.exp(ylog_fit)
                (line,) = ax.plot(x_grid, y_fit, "k-", lw=1)

                label = (
                    f"a={a:.3f}, b={b:.3f}, "
                    f"tau={tau:.3f}, sdef={sdef:.3f}, σ_tot={s_tot:.3f}"
                )
                ax.legend(
                    [line],
                    [label],
                    fontsize=7,
                    loc="upper left",
                    frameon=False,
                )

        ax.set_xlabel(f"{name} [arcsec]")
        if idx % ncols == 0:
            ax.set_ylabel("R26_R [arcsec]")
        ax.set_title(f"{name} (σ_tot={s_tot:.3f})")

    # Turn off unused panels
    for j in range(len(names), len(axes_flat)):
        axes_flat[j].set_axis_off()

    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)

    log.info(f"Wrote calibration diagnostic plot to {png_path}")


def _fit_channel_bisector_robust(
    x_log: np.ndarray,
    sx_log: Optional[np.ndarray],
    y_log: np.ndarray,
    sy_log: Optional[np.ndarray],
    Z: Optional[np.ndarray],
    r26_min_anchor: float = 15.0,
    clip_sigma: float = 3.0,
    max_iter: int = 5,
) -> Tuple[np.ndarray, float, float]:
    """
    Robust symmetric fit for y = a + b*x (+ c^T z) in log-space using a
    least-squares bisector with iterative sigma clipping on orthogonal residuals.

    For now, covariates Z are ignored (the relation is 2-parameter: a, b).
    If Z is not None, they are dropped for this channel.

    Parameters
    ----------
    x_log : array
        log(proxy radius) for anchor objects.
    sx_log : array or None
        1σ uncertainty in log(proxy radius), or None.
    y_log : array
        log(R26_R) for anchor objects.
    sy_log : array or None
        1σ uncertainty in log(R26_R), or None.
    Z : array or None
        Ignored in this implementation (set to None upstream).
    clip_sigma : float
        Sigma-clipping threshold for orthogonal residuals.
    max_iter : int
        Maximum number of clipping iterations.

    Returns
    -------
    beta : array
        [a, b]; no covariates in this implementation.
    tau : float
        Extra scatter term in log-space (in y-direction).
    sigma_obs_default : float
        Default σ_log(x) for this channel when per-object errors are missing.

    """
    eps = 1e-12

    # Base mask: finite and positive radii
    m = np.isfinite(x_log) & np.isfinite(y_log)

    # Apply minimum R26_R cut if requested
    if r26_min_anchor is not None and r26_min_anchor > 0.0:
        y_min_log = np.log(r26_min_anchor)
        m &= (y_log >= y_min_log)

    x = x_log[m]
    y = y_log[m]
    if x.size < 3:
        # Degenerate: default to identity with modest scatter
        return np.array([0.0, 1.0], dtype=float), 0.0, 0.1

    # Ignore covariates for now
    # (If you decide you truly need Z, we can extend this to a robust TLS in higher dimensions.)
    # Iterative sigma-clipped LS-bisector (Isobe+ 1990 style)
    mask = np.ones_like(x, dtype=bool)

    for _ in range(max_iter):
        xx = x[mask]
        yy = y[mask]
        if xx.size < 3:
            break

        # OLS Y|X: yy = a1 + b1 * xx
        A1 = np.column_stack([np.ones_like(xx), xx])
        b1, a1 = np.linalg.lstsq(A1, yy, rcond=None)[0][1], np.linalg.lstsq(A1, yy, rcond=None)[0][0]

        # OLS X|Y: xx = a2 + b2 * yy  => yy = (-a2/b2) + (1/b2) * xx
        A2 = np.column_stack([np.ones_like(yy), yy])
        b2x, a2x = np.linalg.lstsq(A2, xx, rcond=None)[0][1], np.linalg.lstsq(A2, xx, rcond=None)[0][0]
        if b2x == 0:
            # Fallback: if pathological, just use Y|X
            b = b1
            a = a1
        else:
            m1 = b1
            m2 = 1.0 / b2x

            # Intersection of the two regression lines
            # y = a1 + m1 x
            # y = -a2x/b2x + (1/b2x) x
            denom = (m1 - m2)
            if abs(denom) < eps:
                # Nearly parallel; anchor at means
                x0 = np.mean(xx)
                y0 = np.mean(yy)
            else:
                x0 = ((-a2x / b2x) - a1) / (m1 - m2)
                y0 = a1 + m1 * x0

            # Bisector slope: angle-bisector between m1 and m2
            theta1 = np.arctan(m1)
            theta2 = np.arctan(m2)
            mb = np.tan(0.5 * (theta1 + theta2))
            a = y0 - mb * x0
            b = mb

        # Orthogonal residuals to the current bisector line
        # distance = (y - (a + b x)) / sqrt(1 + b^2)
        denom = np.sqrt(1.0 + b * b)
        r_orth = (y - (a + b * x)) / (denom + eps)

        # Robust scale via MAD
        mad = np.nanmedian(np.abs(r_orth[mask] - np.nanmedian(r_orth[mask])))
        sigma = 1.4826 * mad if np.isfinite(mad) and mad > 0 else np.nanstd(r_orth[mask])

        if not np.isfinite(sigma) or sigma == 0.0:
            # No sensible scatter; stop iterating
            break

        new_mask = np.abs(r_orth) <= clip_sigma * sigma

        if new_mask.sum() == mask.sum():
            # Converged
            mask = new_mask
            break

        mask = new_mask

    # Final fit from last iteration
    xx = x[mask]
    yy = y[mask]
    if xx.size < 3:
        # Fallback to simple Y|X on all points
        A = np.column_stack([np.ones_like(x), x])
        a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    else:
        A1 = np.column_stack([np.ones_like(xx), xx])
        b1, a1 = np.linalg.lstsq(A1, yy, rcond=None)[0][1], np.linalg.lstsq(A1, yy, rcond=None)[0][0]

        A2 = np.column_stack([np.ones_like(yy), yy])
        b2x, a2x = np.linalg.lstsq(A2, xx, rcond=None)[0][1], np.linalg.lstsq(A2, xx, rcond=None)[0][0]
        if b2x == 0:
            a, b = a1, b1
        else:
            m1 = b1
            m2 = 1.0 / b2x
            denom = (m1 - m2)
            if abs(denom) < eps:
                x0 = np.mean(xx)
                y0 = np.mean(yy)
            else:
                x0 = ((-a2x / b2x) - a1) / (m1 - m2)
                y0 = a1 + m1 * x0
            theta1 = np.arctan(m1)
            theta2 = np.arctan(m2)
            mb = np.tan(0.5 * (theta1 + theta2))
            a = y0 - mb * x0
            b = mb

    beta = np.array([float(a), float(b)], dtype=float)

    # Now estimate tau, sigma_obs_default in y-direction
    y_pred = beta[0] + beta[1] * x
    resid_y = y - y_pred

    # Measurement variance term (if any)
    meas_var = np.zeros_like(resid_y)
    if sy_log is not None:
        sy = sy_log[m]
        meas_var += np.where(np.isfinite(sy), sy**2, 0.0)
    if sx_log is not None:
        sx = sx_log[m]
        meas_var += (beta[1] ** 2) * np.where(np.isfinite(sx), sx**2, 0.0)

    # Robust residual scale in y
    mad_y = np.nanmedian(np.abs(resid_y[mask] - np.nanmedian(resid_y[mask])))
    resid_scale_y = 1.4826 * mad_y if np.isfinite(mad_y) and mad_y > 0 else np.nanstd(resid_y[mask])

    mean_meas_var = np.nanmean(meas_var[mask]) if np.isfinite(np.nanmean(meas_var[mask])) else 0.0
    tau2 = max(0.0, resid_scale_y**2 - mean_meas_var)
    tau = float(np.sqrt(tau2))

    # Default σ_log(x): if we have sxl, use its median; else infer from residuals
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


def calibrate_from_table(
    tbl: Table,
    calib_path: str,
    *,
    covariates: Optional[np.ndarray] = None,
    covariate_names: Optional[List[str]] = None,
    r26_min_anchor: float = 10.,
) -> Calibration:
    """Identify calibration anchor (objects with valid R26_R and
    R26_ERR_R > 0), fit channels via ODR, and write calibration TSV.

    Returns Calibration object.

    """
    table, sigma, N = _collect_channels_from_table(tbl)

    # Build anchor target y from R26_R
    if "r26" not in table:
        raise ValueError("Input table lacks R26_R column required for calibration.")

    y_lin = table["r26"]
    sy_lin = sigma.get("r26", None)

    # Anchor mask: valid y and sy
    anchor_mask = np.isfinite(y_lin) & (y_lin > 0.0)
    if sy_lin is not None:
        anchor_mask &= np.isfinite(sy_lin) & (sy_lin > 0.0)

    n_anchor = int(np.sum(anchor_mask))
    if n_anchor < 100:
        log.warning(f"small anchor set size = {n_anchor}")

    y_log, sy_log = _to_log_and_sigma(y_lin[anchor_mask], None if sy_lin is None else sy_lin[anchor_mask])

    # For diagnostics: store x,y in linear space for each channel on the anchor set
    plot_data: Dict[str, Dict[str, np.ndarray]] = {}

    # Channels to calibrate: all except the target
    channels: Dict[str, ChannelCalib] = {}
    for name, x_lin_full in table.items():
        if name == "r26":
            continue
        x_lin = x_lin_full[anchor_mask]
        if np.all(~np.isfinite(x_lin)):
            continue

        sx_lin_full = sigma.get(name, None)
        sx_lin = None if sx_lin_full is None else sx_lin_full[anchor_mask]
        x_log, sx_log = _to_log_and_sigma(x_lin, sx_lin)

        # Mask rows where x_log is finite (y_log already finite by construction)
        m = np.isfinite(x_log)
        if not np.any(m):
            continue

        # Save linear-space data for diagnostics (only rows used in the fit)
        plot_data[name] = {
            "x": x_lin[m],
            "y": y_lin[anchor_mask][m],
        }

        Z = None
        if covariates is not None:
            Z = np.asarray(covariates, dtype=float)[anchor_mask, :]
            Z = Z[m, :]
        yl = y_log[m]
        syl = None if sy_log is None else sy_log[m]

        beta, tau, sdef = _fit_channel_bisector_robust(
            x_log[m],
            None if sx_log is None else sx_log[m],
            yl,
            syl,
            Z=None,              # ignore covariates for now; they weren't used anyway
            r26_min_anchor=r26_min_anchor,
            clip_sigma=3.0,
            max_iter=5,
        )
        a = float(beta[0]); b = float(beta[1]); c = beta[2:].copy() if len(beta) > 2 else np.array([], dtype=float)

        channels[name] = ChannelCalib(
            name=name, a=a, b=b, tau=float(tau),
            covar_names=(covariate_names or []), c=c,
            sigma_obs_default=sdef
        )

        log.info(f"Calibrated {name:>6s}: a={a:.4f}  b={b:.4f}  tau={tau:.4f}  sdef={sdef:.4f}  (N={np.sum(m)})")

    cal = Calibration(channels=channels, target_name="r26")
    save_calibration(cal, calib_path)
    log.info(f"Wrote calibration to {calib_path}  (channels={len(channels)})")

    # Generate diagnostic plot
    _plot_calibration_diagnostics(plot_data, cal, calib_path)

    return cal


def _channel_threshold(name: str) -> Optional[int]:
    """Return isophotal threshold (e.g. 23,24,25,26) or None for non-isophotal."""
    if name == "moment":
        return None
    if name == "r26":
        return 26  # direct target measurement
    # names like 'g24', 'r25', 'z26'
    try:
        th = int(name[-2:])
        return th
    except ValueError:
        return None


def _infer_one(
    measurements: Dict[str, float],
    sigmas: Dict[str, Optional[float]],
    cal: Calibration,
    var_floor_log: float = 1e-4,
    covars_row: Optional[np.ndarray] = None,
    include_direct_r26: bool = True,
) -> Tuple[float, float, Dict[str, float]]:
    """Combine channels to infer log R_r26 for a single object."""

    y_list, v_list, w_dict = [], [], {}

    # Determine deepest available isophotal level for this object
    th_available = []
    for name in measurements.keys():
        th = _channel_threshold(name)
        if th is not None:
            th_available.append(th)
    deepest_th = max(th_available) if th_available else None

    # Optionally include direct R26_R as a measurement
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

    # Use calibrated channels with hierarchy: only channels at the
    # deepest available isophotal threshold, plus non-threshold
    # channels like 'moment'.
    for name, ch in cal.channels.items():
        if name not in measurements:
            continue

        th = _channel_threshold(name)

        # Enforce the threshold hierarchy:
        # if we have any isophotal measurements, ignore shallower ones
        if deepest_th is not None and th is not None and th < deepest_th:
            continue  # e.g., if deepest=26, drop 25/24/23; if deepest=25, drop 24/23

        x_lin = measurements[name]
        if not np.isfinite(x_lin) or x_lin <= 0.0:
            continue

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
                raise ValueError(
                    f"Covariate vector required (len {len(ch.c)}) for channel {name}."
                )
            z_term = float(np.dot(ch.c, covars_row))

        b = ch.b if ch.b != 0.0 else 1e-9

        # Invert mapping: x_log = a + b*y + z_term  => y = (x_log - a - z_term)/b
        yj = (x_log - ch.a - z_term) / b

        var_j = ((0.0 if sx_log is None else sx_log**2) + ch.tau**2) / (b**2)
        var_j = max(var_j, var_floor_log)

        w = 1.0 / var_j
        y_list.append(yj)
        v_list.append(var_j)
        w_dict[name] = w

    if not y_list:
        return np.nan, np.nan, {}

    w_arr = 1.0 / np.asarray(v_list)
    y_arr = np.asarray(y_list)

    y_hat = float(np.sum(w_arr * y_arr) / np.sum(w_arr))
    sigma_y = float(np.sqrt(1.0 / np.sum(w_arr)))

    return y_hat, sigma_y, w_dict


def infer_best_r26(
    tbl: Table,
    *,
    calib_path: str = None,
    covariates: Optional[np.ndarray] = None,
    add_columns: bool = False,
    include_direct_r26: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Apply stored calibration to an Astropy Table and return arrays
    of D26 and D26_ERR (diameter in arcmin). Optionally add columns to
    the table.

    Returns (D26, D26_ERR, D26_REF, D26_WEIGHT)

    """
    cal = load_calibration(calib_path)
    table, sigma, N = _collect_channels_from_table(tbl)

    # Prepare output (float32)
    D26 = np.full(N, np.nan, dtype=np.float32)
    D26_ERR = np.full(N, np.nan, dtype=np.float32)
    D26_REF = np.array([""] * N, dtype="U12")
    D26_WEIGHT = np.full(N, np.nan, dtype=np.float32)

    any_cov = any((ch.c.size > 0) for ch in cal.channels.values())
    if any_cov and covariates is None:
        raise ValueError("Calibration expects covariates, but none were provided.")

    for i in range(N):
        meas_i: Dict[str, float] = {}
        sig_i: Dict[str, Optional[float]] = {}
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

        #print("channels in calibration:", sorted(cal.channels.keys()))
        #print("measurements:", meas_i)
        #print("sigmas:", sig_i)

        y, sy, wdict = _infer_one(
            meas_i, sig_i, cal,
            covars_row=zi,
            include_direct_r26=include_direct_r26
        )

        if np.isfinite(y):
            R = np.exp(y)                         # arcsec (radius)
            sR = R * sy                           # arcsec (radius 1σ)
            D = (2.0 * R) / 60.0                  # arcmin (diameter)
            sD = (2.0 * sR) / 60.0                # arcmin
            D26[i] = np.float32(D)
            D26_ERR[i] = np.float32(sD)
            if wdict:
                kmax = max(wdict, key=wdict.get)
                D26_REF[i] = kmax
                D26_WEIGHT[i] = np.float32(wdict[kmax])

    if add_columns:
        # Remove if present, then add with requested names/dtypes
        for colname in ("D26", "D26_ERR", "D26_REF", "D26_WEIGHT"):
            if colname in tbl.colnames:
                tbl.remove_column(colname)
        tbl["D26"] = D26.astype(np.float32)
        tbl["D26_ERR"] = D26_ERR.astype(np.float32)
        tbl["D26_REF"] = D26_REF
        tbl["D26_WEIGHT"] = D26_WEIGHT.astype(np.float32)

    return D26, D26_ERR, D26_REF, D26_WEIGHT
