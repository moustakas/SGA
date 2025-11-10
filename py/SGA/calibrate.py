"""
SGA.calibrate
=============

Code to calibrate Rr(26).

"""
from __future__ import annotations

import json
import warnings
import numpy as np
from typing import Dict, List, Optional, Tuple
from astropy.table import Table
from scipy import odr
from sklearn.linear_model import HuberRegressor

from SGA.SGA import SBTHRESH
from SGA.coadds import GRIZ as BANDS

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
            base = f"R{th}_{band}"
            err = f"R{th}_ERR_{band}"
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


def load_calibration(path: str) -> Calibration:
    lines = [ln.strip() for ln in open(path, "r") if ln.strip()]
    if not lines or not lines[0].startswith("name"):
        raise ValueError("Malformed calibration file")
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


def _fit_channel_odr(
    x_log: np.ndarray,
    sx_log: Optional[np.ndarray],
    y_log: np.ndarray,
    sy_log: Optional[np.ndarray],
    Z: Optional[np.ndarray]) -> Tuple[np.ndarray, float, float]:
    """Fit y = a + b*x + c^T z via SciPy ODR with robust sanitation +
    fallbacks.

    Returns (beta, tau, sigma_obs_default).

    """
    # --- sanitize ---
    eps = 1e-12
    m = np.isfinite(x_log) & np.isfinite(y_log)
    if sx_log is not None:
        sx_log = np.where(np.isfinite(sx_log) & (sx_log >= 0), sx_log, 0.0)
        sx_log = np.where(sx_log < eps, 0.0, sx_log)  # 0 means "unknown"/ignored by ODR
        m &= np.isfinite(sx_log)
    if sy_log is not None:
        sy_log = np.where(np.isfinite(sy_log) & (sy_log >= 0), sy_log, 0.0)
        sy_log = np.where(sy_log < eps, 0.0, sy_log)
        m &= np.isfinite(sy_log)

    if Z is not None:
        Z = Z[np.asarray(m), :]
    xl, yl = x_log[m], y_log[m]
    sxl = None if sx_log is None else sx_log[m]
    syl = None if sy_log is None else sy_log[m]

    # Need at least K+3 points
    K = 0 if Z is None else Z.shape[1]
    if xl.size < max(3, K + 3):
        # Too few points: return OLS with zeros for c if needed
        X_lin = np.hstack([np.ones((xl.size, 1)), xl.reshape(-1, 1)]) if Z is None else np.hstack([np.ones((xl.size, 1)), xl.reshape(-1, 1), Z])
        beta_ols, *_ = np.linalg.lstsq(X_lin, yl, rcond=None)
        beta = beta_ols
        tau = float(np.std(yl - X_lin @ beta)) if xl.size > (K + 2) else 0.0
        sdef = float(np.nanmedian(sxl)) if sxl is not None and sxl.size else 0.1
        return beta, tau, sdef

    # Initial guess
    X_lin = np.hstack([np.ones((xl.size, 1)), xl.reshape(-1, 1)]) if Z is None else np.hstack([np.ones((xl.size, 1)), xl.reshape(-1, 1), Z])
    beta0, *_ = np.linalg.lstsq(X_lin, yl, rcond=None)

    def f(beta, X):
        x = X[0, :]
        yhat = beta[0] + beta[1] * x
        if X.shape[0] > 1:
            c = beta[2:]
            yhat += (X[1:, :].T @ c)
        return yhat

    # Build ODR inputs
    X = np.vstack([xl] if Z is None else [xl, Z.T])
    if sxl is None:
        sx_all = np.vstack([np.zeros_like(xl)])
    else:
        sx_all = np.vstack([sxl])
    if Z is not None:
        sx_all = np.vstack([sx_all, np.zeros_like(Z.T)])

    # --- try ODR ---
    try:
        # Floor any exactly-zero uncertainties to avoid odrpack 1/sd**2 division warnings
        eps_w = 1e-12
        if sx_all is not None:
            sx_all = np.where(np.asarray(sx_all) == 0.0, eps_w, sx_all)
        if syl is not None:
            syl = np.where(np.asarray(syl) == 0.0, eps_w, syl)
        data = odr.RealData(X, yl, sx=sx_all, sy=syl)
        model = odr.Model(f)
        odrobj = odr.ODR(data, model, beta0=beta0, maxit=200)
        odrobj.set_job(fit_type=0)  # explicit ODR
        out = odrobj.run()
        beta = out.beta
    except Exception:
        # --- fallback 1: Deming regression (only if no covariates) ---
        if Z is None:
            sx_med = float(np.nanmedian(sxl)) if sxl is not None and sxl.size else 0.0
            sy_med = float(np.nanmedian(syl)) if syl is not None and syl.size else 0.0
            lam = (sy_med / (sx_med + eps))**2 if (sx_med > 0 and sy_med > 0) else 1.0
            xbar, ybar = np.nanmean(xl), np.nanmean(yl)
            Sxx = np.nanmean((xl - xbar)**2)
            Syy = np.nanmean((yl - ybar)**2)
            Sxy = np.nanmean((xl - xbar)*(yl - ybar))
            disc = (Syy - lam * Sxx)
            b = (disc + np.sqrt(disc**2 + 4 * lam * Sxy**2)) / (2 * Sxy + eps)
            a = ybar - b * xbar
            beta = np.array([a, b], dtype=float)
        else:
            # --- fallback 2: robust OLS (Huber on residual y vs [x,Z]) ---
            hub = HuberRegressor(alpha=0.0, fit_intercept=True)
            hub.fit(X_lin, yl)
            a = float(hub.intercept_); b = float(hub.coef_[0])
            c = hub.coef_[1:] if Z is not None else np.array([], dtype=float)
            beta = np.concatenate([[a, b], c]) if Z is not None else np.array([a, b], dtype=float)

    # Residuals and tau
    if Z is None:
        y_pred = beta[0] + beta[1] * xl
    else:
        y_pred = beta[0] + beta[1] * xl + (Z @ beta[2:])
    resid = yl - y_pred
    meas_var = np.zeros_like(resid)
    if syl is not None:
        meas_var += syl**2
    if sxl is not None:
        meas_var += (beta[1] ** 2) * (sxl**2)

    # robust scale for residuals
    mad = np.nanmedian(np.abs(resid - np.nanmedian(resid)))
    resid_scale = 1.4826 * mad if np.isfinite(mad) else np.nanstd(resid)
    mean_meas_var = np.nanmean(meas_var) if np.isfinite(np.nanmean(meas_var)) else 0.0
    tau2 = max(0.0, resid_scale**2 - mean_meas_var)
    tau = float(np.sqrt(tau2))

    # default σ_log(x)
    if sxl is not None and np.isfinite(np.nanmedian(sxl)):
        sdef = float(np.nanmedian(sxl))
    else:
        sdef = float(resid_scale / (abs(beta[1]) + 1e-9))

    return beta, tau, sdef


def calibrate_from_table(
    tbl: Table,
    calib_path: str,
    *,
    covariates: Optional[np.ndarray] = None,
    covariate_names: Optional[List[str]] = None,
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

        Z = None
        if covariates is not None:
            Z = np.asarray(covariates, dtype=float)[anchor_mask, :]
            Z = Z[m, :]
        yl = y_log[m]
        syl = None if sy_log is None else sy_log[m]

        beta, tau, sdef = _fit_channel_odr(x_log[m], None if sx_log is None else sx_log[m], yl, syl, Z)
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

    return cal


def _infer_one(
    measurements: Dict[str, float],
    sigmas: Dict[str, Optional[float]],
    cal: Calibration,
    var_floor_log: float = 1e-4,
    covars_row: Optional[np.ndarray] = None,
    include_direct_r26: bool = True,
) -> Tuple[float, float, Dict[str, float]]:
    """Combine channels to infer log R_r26 for a single object.

    """
    y_list, v_list, w_dict = [], [], {}

    # Optionally include the direct r26 measurement itself (as y with
    # known σ).
    if include_direct_r26 and "r26" in measurements:
        x_lin = measurements["r26"]
        sx_lin = sigmas.get("r26", None)
        if np.isfinite(x_lin) and (sx_lin is not None) and np.isfinite(sx_lin) and (x_lin > 0.0) and (sx_lin > 0.0):
            yj = float(np.log(x_lin))
            var_j = float((sx_lin / x_lin)**2)
            var_j = max(var_j, var_floor_log)
            w = 1.0 / var_j
            y_list.append(yj); v_list.append(var_j); w_dict["r26"] = w

    for name, ch in cal.channels.items():
        if name not in measurements:
            continue
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
                raise ValueError(f"Covariate vector required (len {len(ch.c)}) for channel {name}.")
            z_term = float(np.dot(ch.c, covars_row))

        # Invert mapping
        b = ch.b if ch.b != 0 else 1e-9
        yj = (x_log - ch.a - z_term) / b
        var_j = ((0.0 if sx_log is None else sx_log**2) + ch.tau**2) / (b**2)
        var_j = max(var_j, var_floor_log)

        w = 1.0 / var_j
        y_list.append(yj); v_list.append(var_j); w_dict[name] = w

    if not y_list:
        return np.nan, np.nan, {}

    w_arr = 1.0 / np.asarray(v_list)
    y_arr = np.asarray(y_list)
    y_hat = float(np.sum(w_arr * y_arr) / np.sum(w_arr))
    sigma_y = float(np.sqrt(1.0 / np.sum(w_arr)))
    return y_hat, sigma_y, w_dict


def infer_best_r26(
    tbl: Table,
    calib_path: str,
    *,
    covariates: Optional[np.ndarray] = None,
    add_columns: bool = True,
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
