
"""
SGA groups.py — overlap-based group finding with optional anisotropic (ellipse-aware) linking.
This version removes any pydl dependency and assumes Astropy is available.
"""

from __future__ import annotations

import math
import numpy as np
from astropy.table import Table, Column

# Logging: use SGA.logger if available, otherwise standard logging
try:
    from SGA.logger import log  # type: ignore
except Exception:
    import logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    class _L:
        @staticmethod
        def info(msg): logging.getLogger("SGA.groups").info(msg)
        @staticmethod
        def warning(msg): logging.getLogger("SGA.groups").warning(msg)
        @staticmethod
        def debug(msg): logging.getLogger("SGA.groups").debug(msg)
    log = _L()

DEG2RAD = np.pi / 180.0
RAD2DEG = 180.0 / np.pi
ARCMIN_PER_DEG = 60.0

def _wrap_deg(x: float | np.ndarray) -> float | np.ndarray:
    return np.mod(x, 360.0)

def _angdiff_deg(a: float, b: float) -> float:
    """Smallest signed difference a-b in (-180, 180]."""
    d = (a - b + 180.0) % 360.0 - 180.0
    return d

def _bearing_pa_deg(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    """
    Astronomical bearing (position angle) from (ra1,dec1) toward (ra2,dec2).
    0° = North, 90° = East, increases CCW. Returns [0,360).
    Uses a small-angle projection appropriate for arcminute scales.
    """
    ra1r, dec1r, ra2r, dec2r = np.radians([ra1, dec1, ra2, dec2])
    dra = (ra2r - ra1r) * np.cos(dec1r)
    ddec = (dec2r - dec1r)
    pa = np.degrees(np.arctan2(dra, ddec))
    return float(_wrap_deg(pa))

def _reff_arcmin(diam_arcmin: float, ba: float, pa_deg: float,
                 ra_i: float, dec_i: float, ra_j: float, dec_j: float,
                 q_floor: float = 0.0) -> float:
    """
    Directional elliptical radius (arcmin) of object i toward j.
    Parameters
    ----------
    diam_arcmin : D25 major-axis diameter (arcmin)
    ba : axis ratio b/a in (0,1]; if invalid, fall back to circular radius
    pa_deg : astronomical PA of major axis, degrees
    ra_i, dec_i : position of object i [deg]
    ra_j, dec_j : position of neighbor j [deg]
    q_floor : minimum axis ratio to avoid extreme shrink along minor axis
    """
    if not np.isfinite(diam_arcmin) or diam_arcmin <= 0.0:
        return 0.0
    a = 0.5 * float(diam_arcmin)
    if not (np.isfinite(ba) and ba > 0.0):
        return a
    if q_floor and ba < q_floor:
        ba = q_floor
    b = ba * a
    if not np.isfinite(pa_deg):
        return a
    bearing = _bearing_pa_deg(ra_i, dec_i, ra_j, dec_j)
    ddeg = _angdiff_deg(bearing, float(pa_deg))
    d = np.radians(ddeg)
    denom = math.hypot(b * math.cos(d), a * math.sin(d))
    if denom <= 0.0:
        return a
    return (a * b) / denom

def _small_angle_sep_arcmin(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    """Approximate small-angle separation in arcmin (accurate at a few arcminutes)."""
    dra = (ra2 - ra1) * math.cos(dec1 * DEG2RAD)
    ddec = (dec2 - dec1)
    return math.hypot(dra, ddec) * ARCMIN_PER_DEG

def _radectoxyz(ra_rad: np.ndarray, dec_rad: np.ndarray) -> np.ndarray:
    """Unit vectors from RA/Dec in radians; shape (N,3)."""
    cosd = np.cos(dec_rad)
    x = cosd * np.cos(ra_rad)
    y = cosd * np.sin(ra_rad)
    z = np.sin(dec_rad)
    return np.vstack((x, y, z)).T

def _xyztoradec(xyz: np.ndarray) -> tuple[float, float]:
    """RA,Dec in radians from a 3-vector."""
    x, y, z = xyz
    ra = math.atan2(y, x)
    rxy = math.hypot(x, y)
    dec = math.atan2(z, rxy)
    if ra < 0:
        ra += 2.0 * np.pi
    return ra, dec

class DSU:
    """Disjoint Set Union (Union-Find) with path compression and union by rank."""
    def __init__(self, n: int):
        self.p = list(range(n))
        self.r = [0] * n
    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x
    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.r[ra] < self.r[rb]:
            self.p[ra] = rb
        elif self.r[ra] > self.r[rb]:
            self.p[rb] = ra
        else:
            self.p[rb] = ra
            self.r[ra] += 1

def build_group_catalog(
    cat: Table,
    group_id_start: int = 0,
    mfac: float = 1.5,
    dmin: float = 36.0/3600.0,   # degrees; 36 arcsec
    dmax: float = 3.0/60.0,      # degrees; 3 arcmin window for candidate pairs
    anisotropic: bool = True,
    link_mode: str = "hybrid",   # "off" | "anisotropic" | "hybrid"
    big_diam: float = 5.0,       # arcmin threshold for "large" galaxies
    mfac_backbone: float = 2.0,  # large–large multiplier
    mfac_sat: float = 1.8,       # large–small multiplier (default tuned)
    k_floor: float = 0.30,       # floor factor: r_eff >= k_floor * r_circ (default tuned)
    q_floor: float = 0.20,       # minimum axis ratio for anisotropy (default tuned)
    name_via: str = "radec",     # "radec" or "none"
) -> Table:
    """
    Build an overlap-based group catalog with optional anisotropy and a hybrid strategy.

    Required input columns in `cat` (degrees and arcminutes):
        RA, DEC [deg], DIAM [arcmin]
    Optional columns:
        BA (=b/a), PA [deg astronomical]
        OBJNAME (kept and returned untouched)

    Output columns added:
        GROUP_ID, MULT (group multiplicity), PRIMARY (bool),
        GROUP_NAME, GROUP_RA, GROUP_DEC, GROUP_DIAM (arcmin).
    """
    required = {'RA', 'DEC', 'DIAM'}
    if not required <= set(cat.colnames):
        raise ValueError(f"Catalog must have columns {required}")
    RA = np.asarray(cat['RA'], dtype=float)
    DEC = np.asarray(cat['DEC'], dtype=float)
    DIAM = np.asarray(cat['DIAM'], dtype=float)
    N = len(cat)
    BA = np.asarray(cat['BA'], dtype=float) if 'BA' in cat.colnames else np.full(N, np.nan)
    PA = np.asarray(cat['PA'], dtype=float) if 'PA' in cat.colnames else np.full(N, np.nan)

    def _radec_to_groupname(ra_deg: float, dec_deg: float) -> str:
        """Best-effort group name following existing SGA convention if available."""
        try:
            from SGA.io import radec_to_groupname  # type: ignore
            return radec_to_groupname(ra_deg, dec_deg)
        except Exception:
            ra_h = ra_deg / 15.0
            h = int(ra_h)
            m = int((ra_h - h) * 60.0 + 1e-6)
            ra_tag = f"{h:02d}{m:02d}"
            s = '+' if dec_deg >= 0 else '-'
            d = int(abs(dec_deg))
            dm = int((abs(dec_deg) - d) * 60.0 + 1e-6)
            dec_tag = f"{d:02d}{dm:02d}"
            return f"G{ra_tag}{s}{dec_tag}"

    def thresh_arcmin(i: int, j: int) -> float:
        """Pair-specific link threshold (arcmin) under the selected mode."""
        ri_c, rj_c = 0.5 * DIAM[i], 0.5 * DIAM[j]
        if not anisotropic or link_mode in ("off", None):
            return 0.5 * mfac * (ri_c + rj_c)

        ri_a = _reff_arcmin(DIAM[i], BA[i], PA[i], RA[i], DEC[i], RA[j], DEC[j], q_floor=q_floor)
        rj_a = _reff_arcmin(DIAM[j], BA[j], PA[j], RA[j], DEC[j], RA[i], DEC[i], q_floor=q_floor)

        if link_mode == "anisotropic":
            return 0.5 * mfac * (ri_a + rj_a)

        big_i = DIAM[i] >= big_diam
        big_j = DIAM[j] >= big_diam
        if big_i and big_j:
            return 0.5 * mfac_backbone * (ri_c + rj_c)
        if big_i or big_j:
            ri = max(ri_a, k_floor * ri_c)
            rj = max(rj_a, k_floor * rj_c)
            return 0.5 * mfac_sat * (ri + rj)
        return 0.5 * mfac * (ri_a + rj_a)

    dsu = DSU(N)
    dmax_arcmin = dmax * ARCMIN_PER_DEG
    dmin_arcmin = dmin * ARCMIN_PER_DEG
    links = 0

    # Pair loop with a quick rectangular pre-filter followed by an accurate separation
    for i in range(N):
        cosdi = math.cos(DEC[i] * DEG2RAD)
        for j in range(i + 1, N):
            if abs(DEC[j] - DEC[i]) > dmax:
                continue
            if abs((RA[j] - RA[i]) * cosdi) > dmax:
                continue
            dd = _small_angle_sep_arcmin(RA[i], DEC[i], RA[j], DEC[j])
            if dd > dmax_arcmin:
                continue
            thr = thresh_arcmin(i, j)
            thr = max(thr, dmin_arcmin)
            if dd <= thr:
                dsu.union(i, j)
                links += 1

    roots = np.array([dsu.find(i) for i in range(N)])
    uniq, inv = np.unique(roots, return_inverse=True)
    group_ids = inv + group_id_start
    mult = np.bincount(inv, minlength=len(uniq))

    # Group centers by DIAM-weighted average on the unit sphere
    grp_ra = np.zeros(len(uniq))
    grp_dec = np.zeros(len(uniq))
    grp_diam = np.zeros(len(uniq))
    grp_primary = np.zeros(len(uniq), dtype=int)

    for g, r in enumerate(uniq):
        idx = np.where(roots == r)[0]
        w = np.clip(DIAM[idx], 1e-3, None)
        ra_rad = RA[idx] * DEG2RAD
        dec_rad = DEC[idx] * DEG2RAD
        xyz = _radectoxyz(ra_rad, dec_rad)
        cen = (xyz * w[:, None]).sum(axis=0) / w.sum()
        cen /= np.linalg.norm(cen)
        ra_c, dec_c = _xyztoradec(cen)
        ra_c *= RAD2DEG
        dec_c *= RAD2DEG
        grp_ra[g], grp_dec[g] = ra_c, dec_c

        # Conservative group footprint: max over (dist to center + member circular radius)
        max_extent = 0.0
        for k in idx:
            dcen = _small_angle_sep_arcmin(RA[k], DEC[k], ra_c, dec_c)
            rk = 0.5 * DIAM[k]
            max_extent = max(max_extent, dcen + rk)
        grp_diam[g] = 2.0 * max_extent
        grp_primary[g] = int(idx[np.argmax(DIAM[idx])])

    # Attach per-row columns
    if 'GROUP_ID' in cat.colnames:
        cat.remove_column('GROUP_ID')
    cat.add_column(Column(group_ids, name='GROUP_ID'))

    row_mult = mult[inv]
    if 'MULT' in cat.colnames:
        cat.remove_column('MULT')
    cat.add_column(Column(row_mult, name='MULT'))

    primary_flags = np.zeros(N, dtype=bool)
    for g, r in enumerate(uniq):
        primary_flags[grp_primary[g]] = True
    if 'PRIMARY' in cat.colnames:
        cat.remove_column('PRIMARY')
    cat.add_column(Column(primary_flags, name='PRIMARY'))

    if 'GROUP_NAME' in cat.colnames:
        cat.remove_column('GROUP_NAME')
    if name_via == "radec":
        names = np.array([
            _radec_to_groupname(grp_ra[inv[i]], grp_dec[inv[i]])
            for i in range(N)
        ])
    else:
        names = np.array([''] * N)
    cat.add_column(Column(names, name='GROUP_NAME'))

    for colname in ('GROUP_RA', 'GROUP_DEC', 'GROUP_DIAM'):
        if colname in cat.colnames:
            cat.remove_column(colname)
    cat.add_column(Column(grp_ra[inv], name='GROUP_RA'))
    cat.add_column(Column(grp_dec[inv], name='GROUP_DEC'))
    cat.add_column(Column(grp_diam[inv], name='GROUP_DIAM'))

    nm = len(uniq)
    n1 = int(np.sum(mult == 1))
    n2 = int(np.sum(mult == 2))
    n3p = int(np.sum(mult >= 3))
    log.info(f"Groups built: {nm} total (singles={n1}, pairs={n2}, 3+={n3p}); links formed: {links}")

    return cat

def qa(*args, **kwargs):
    """Placeholder QA function; intentionally does nothing in this drop-in."""
    log.info("qa(): no-op placeholder.")
