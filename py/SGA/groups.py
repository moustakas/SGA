"""
SGA groups.py — high-performance group finding with hybrid anisotropic linking,
multiprocessing, and ellipse-containment linking.

Assumptions:
  - Astropy available
  - `from SGA.logger import log` succeeds
  - `from SGA.io import radec_to_groupname` succeeds
  - `pydl.pydlutils.spheregroup.spheregroup` installed (pip install pydl)

Appended columns (exact dtypes):
    GROUP_ID        int32
    GROUP_NAME      str10
    GROUP_MULT      int16
    GROUP_PRIMARY   bool
    GROUP_RA        float64
    GROUP_DEC       float64
    GROUP_DIAMETER  float32   # arcmin
"""

from __future__ import annotations

import math
import time
import numpy as np
from astropy.table import Table, Column
from concurrent.futures import ProcessPoolExecutor, as_completed
from SGA.logger import log
from SGA.io import radec_to_groupname
from pydl.pydlutils.spheregroup import spheregroup as _spheregroup

DEG2RAD = np.pi / 180.0
RAD2DEG = 180.0 / np.pi
ARCMIN_PER_DEG = 60.0

# Globals for worker processes (set by _init_pool)
_G_RA = None
_G_DEC = None
_G_DIAM = None
_G_BA = None
_G_PA = None
_G_a_arc = None
_G_b_arc = None
_G_pa_rad = None
_G_has_aniso = None
_G_params = None


def _wrap_deg(x: float | np.ndarray) -> float | np.ndarray:
    return np.mod(x, 360.0)


def _angdiff_deg(a: float, b: float) -> float:
    return (a - b + 180.0) % 360.0 - 180.0


def _bearing_pa_deg_local(dx_deg: float, dy_deg: float) -> float:
    """
    Local-plane bearing (PA-style) from point i toward j using small-angle
    tangent-plane offsets:
        dx_deg = (RA_j - RA_i) * cos(dec0)
        dy_deg = (DEC_j - DEC_i)
    Returns PA in degrees: 0° = North, 90° = East, [0,360).
    """
    pa = math.degrees(math.atan2(dx_deg, dy_deg))
    if pa < 0.0:
        pa += 360.0
    return pa


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


def _init_pool(RA, DEC, DIAM, BA, PA, params_dict):
    """
    Initializer for worker processes. Stores large arrays and common parameters
    in module-level globals to avoid pickling them for every task.
    """
    global _G_RA, _G_DEC, _G_DIAM, _G_BA, _G_PA
    global _G_a_arc, _G_b_arc, _G_pa_rad, _G_has_aniso, _G_params

    _G_RA = RA
    _G_DEC = DEC
    _G_DIAM = DIAM
    _G_BA = BA
    _G_PA = PA

    a_arc = 0.5 * DIAM.astype(float)
    ba_eff = np.where(np.isfinite(BA) & (BA > 0.0), BA, 1.0)
    q_floor = params_dict.get('q_floor', 0.0)
    if q_floor and q_floor > 0.0:
        ba_eff = np.maximum(ba_eff, q_floor)
    b_arc = ba_eff * a_arc
    pa_rad = np.where(np.isfinite(PA), PA, 0.0) * DEG2RAD
    has_aniso = np.isfinite(PA) & np.isfinite(BA) & (BA > 0.0)

    _G_a_arc = a_arc
    _G_b_arc = b_arc
    _G_pa_rad = pa_rad
    _G_has_aniso = has_aniso
    _G_params = params_dict


def _pair_threshold_arcmin(i: int, j: int, dx_deg: float, dy_deg: float) -> float:
    """
    Compute the hybrid/anisotropic link threshold (arcmin) for a pair (i,j),
    given local-plane deltas dx,dy in degrees.
    """
    a_arc = _G_a_arc
    b_arc = _G_b_arc
    pa_rad = _G_pa_rad
    has_aniso = _G_has_aniso
    p = _G_params

    ri_c = a_arc[i]
    rj_c = a_arc[j]

    if (not p['anisotropic']) or (p['link_mode'] in ('off', None)):
        return 0.5 * p['mfac'] * (ri_c + rj_c)

    if has_aniso[i]:
        b_ang = math.atan2(dx_deg, dy_deg)
        d = b_ang - pa_rad[i]
        if d > math.pi:
            d -= 2.0 * math.pi
        elif d < -math.pi:
            d += 2.0 * math.pi
        denom = math.hypot(b_arc[i] * math.cos(d), a_arc[i] * math.sin(d))
        ri_a = (a_arc[i] * b_arc[i]) / denom if denom > 0.0 else ri_c
    else:
        ri_a = ri_c

    if has_aniso[j]:
        b_ang = math.atan2(-dx_deg, -dy_deg)
        d = b_ang - pa_rad[j]
        if d > math.pi:
            d -= 2.0 * math.pi
        elif d < -math.pi:
            d += 2.0 * math.pi
        denom = math.hypot(b_arc[j] * math.cos(d), a_arc[j] * math.sin(d))
        rj_a = (a_arc[j] * b_arc[j]) / denom if denom > 0.0 else rj_c
    else:
        rj_a = rj_c

    if p['link_mode'] == 'anisotropic':
        return 0.5 * p['mfac'] * (ri_a + rj_a)

    big_i = (_G_DIAM[i] >= p['big_diam'])
    big_j = (_G_DIAM[j] >= p['big_diam'])
    if big_i and big_j:
        return 0.5 * p['mfac_backbone'] * (ri_c + rj_c)
    if big_i or big_j:
        ri = ri_a if (ri_a >= p['k_floor'] * ri_c) else (p['k_floor'] * ri_c)
        rj = rj_a if (rj_a >= p['k_floor'] * rj_c) else (p['k_floor'] * rj_c)
        return 0.5 * p['mfac_sat'] * (ri + rj)
    return 0.5 * p['mfac'] * (ri_a + rj_a)


def _contains_pair(i: int, j: int, dx_deg: float, dy_deg: float, scale: float) -> bool:
    """
    Return True if j lies inside the *scaled* ellipse of i.
    Uses local-plane deltas (dx,dy) in degrees and precomputed ellipse params.
    """
    a = _G_a_arc[i]
    b = _G_b_arc[i]
    if a <= 0.0 or b <= 0.0:
        return False
    if not _G_has_aniso[i]:
        r = math.hypot(dx_deg, dy_deg) * ARCMIN_PER_DEG
        return r <= scale * a

    d = math.atan2(dx_deg, dy_deg) - _G_pa_rad[i]
    if d > math.pi:
        d -= 2.0 * math.pi
    elif d < -math.pi:
        d += 2.0 * math.pi
    denom = math.hypot(b * math.cos(d), a * math.sin(d))
    if denom <= 0.0:
        return False
    r_boundary = scale * (a * b) / denom
    r = math.hypot(dx_deg, dy_deg) * ARCMIN_PER_DEG
    return r <= r_boundary


def _process_cluster(members: list[int]) -> np.ndarray:
    """
    Worker: process a single precluster and return an array of edges (i,j)
    that should be unioned at the global DSU. Shape (E,2), dtype int64.
    """
    p = _G_params
    RA = _G_RA
    DEC = _G_DEC

    m = len(members)
    if m < 2:
        return np.empty((0, 2), dtype=np.int64)

    idx = np.asarray(members, dtype=np.int64)
    dec0 = float(np.median(DEC[idx]))
    cosd0 = math.cos(dec0 * DEG2RAD)
    ra0 = float(np.median(RA[idx]))
    dx = (RA[idx] - ra0) * cosd0
    dy = (DEC[idx] - dec0)

    cell_deg = p['cell_arcmin'] / ARCMIN_PER_DEG
    gx = np.floor(dx / cell_deg).astype(np.int64)
    gy = np.floor(dy / cell_deg).astype(np.int64)

    bins = {}
    for k in range(m):
        key = (int(gx[k]), int(gy[k]))
        if key in bins:
            bins[key].append(k)
        else:
            bins[key] = [k]

    edges = []
    dmax_arcmin = p['dmax_arcmin']
    dmin_arcmin = p['dmin_arcmin']
    s = 1.0 + float(p.get('contain_margin', 0.20)) if p.get('contain', True) else None

    for k in range(m):
        i_global = int(idx[k])
        cx = int(gx[k])
        cy = int(gy[k])

        for dxcell in (-1, 0, 1):
            for dycell in (-1, 0, 1):
                key = (cx + dxcell, cy + dycell)
                neigh = bins.get(key)
                if not neigh:
                    continue
                for kk in neigh:
                    if kk <= k:
                        continue
                    j_global = int(idx[kk])

                    ddx = dx[kk] - dx[k]
                    ddy = dy[kk] - dy[k]
                    dd_arcmin = math.hypot(ddx, ddy) * ARCMIN_PER_DEG
                    if dd_arcmin > dmax_arcmin:
                        continue

                    if s is not None and (_contains_pair(i_global, j_global, ddx, ddy, s) or
                                          _contains_pair(j_global, i_global, -ddx, -ddy, s)):
                        edges.append((i_global, j_global))
                        continue

                    thr = _pair_threshold_arcmin(i_global, j_global, ddx, ddy)
                    if thr < dmin_arcmin:
                        thr = dmin_arcmin
                    if dd_arcmin <= thr:
                        edges.append((i_global, j_global))

    if not edges:
        return np.empty((0, 2), dtype=np.int64)
    return np.asarray(edges, dtype=np.int64)


def build_group_catalog(
    cat: Table,
    group_id_start: int = 0,
    mfac: float = 1.5,
    dmin: float = 36.0/3600.0,      # deg (36")
    dmax: float = 3.0/60.0,         # deg (3') maximum candidate separation
    anisotropic: bool = True,
    link_mode: str = "hybrid",      # "off" | "anisotropic" | "hybrid"
    big_diam: float = 3.0,          # arcmin threshold for "large" galaxies  [DEFAULT CHANGED]
    mfac_backbone: float = 2.0,     # large–large multiplier
    mfac_sat: float = 1.5,          # large–small multiplier                 [DEFAULT CHANGED]
    k_floor: float = 0.40,          # floor factor for softened anisotropy   [DEFAULT CHANGED]
    q_floor: float = 0.20,          # minimum BA
    name_via: str = "radec",
    sphere_link_arcmin: float | None = None,  # precluster link length; default uses dmax
    grid_cell_arcmin: float | None = None,    # grid bin size; default = dmax
    mp: int = 1,                               # number of processes for parallel precluster processing
    contain: bool = True,                      # enable ellipse-containment linking
    contain_margin: float = 0.50,              # expansion on axes (1+margin)       [DEFAULT CHANGED]
) -> Table:
    """
    Build a mosaic-friendly group catalog on the sky using a **hybrid anisotropic** rule
    with fast preclustering and optional multiprocessing.

    Parameters
    ----------
    cat : astropy.table.Table
        Input table with required columns: **RA, DEC** [deg] and **DIAM** [arcmin].
        Optional: **BA** (= b/a), **PA** [deg astronomical], **OBJNAME**.

    group_id_start : int, default 0
        Offset added to the 0-based group labels.

    mfac : float, default 1.5
        Base multiplier for **small–small** links (and for all links if `link_mode="anisotropic"`).
        Threshold:  d ≤ 0.5 * mfac * (r_i + r_j).  *Secondary in hybrid.*

    dmin : float, default 36"/deg
        Hard minimum link length (in degrees) to avoid pathological tiny links.

    dmax : float, default 3'/deg
        Hard maximum candidate separation (in degrees). Also sets the search window,
        precluster default, and grid cell default. Larger dmax = more candidates.

    anisotropic : bool, default True
        Whether to use ellipse-aware radii (BA/PA). If False or `link_mode="off"`, falls back
        to circular geometry with `mfac`.

    link_mode : {"off", "anisotropic", "hybrid"}, default "hybrid"
        - "off": circular rule only: d ≤ 0.5 * mfac * (a_i + a_j).
        - "anisotropic": directional ellipses for **all** pairs: d ≤ 0.5 * mfac * (r_i + r_j).
        - "hybrid": use pair-type rules below (recommended).

    big_diam : float [arcmin], default 3.0
        Size threshold for "large" galaxies. Controls which branch fires:
        large–large, large–small, or small–small. **Primary knob** in hybrid.

    mfac_backbone : float, default 2.0
        Multiplier for **large–large** links (circularized). Threshold:
            d ≤ 0.5 * mfac_backbone * (a_i + a_j).
        Keeps obvious bright pairs/triples together. *Secondary.*

    mfac_sat : float, default 1.5
        Multiplier for **large–small** links. Applied to softened directional radii:
            r'_i = max(r_i, k_floor * a_i);  r'_j = max(r_j, k_floor * a_j)
            d ≤ 0.5 * mfac_sat * (r'_i + r'_j).
        **Primary** for attaching satellites to a large host in hybrid.

    k_floor : float, default 0.40
        Soft floor on directional radii in large–small links to prevent thin edge-ons from
        over-suppressing along the minor axis. Higher = more permissive. *Primary for hybrid.*

    q_floor : float, default 0.20
        Global minimum axis ratio used in all anisotropic radii (prevents extreme shrink). *Secondary.*

    name_via : {"radec","none"}, default "radec"
        How to generate GROUP_NAME (fallback uses J-style encoding).

    sphere_link_arcmin : float or None, default None
        Preclustering link length (arcmin) for `spheregroup`. If None, uses `dmax` (3′).
        Tighter = smaller preclusters; looser = larger preclusters. *Affects speed.*

    grid_cell_arcmin : float or None, default None
        Grid cell size (arcmin) for intra-precluster neighbor search. If None, uses `dmax`.
        Larger cells can reduce overhead in sparse fields. *Affects speed.*

    mp : int, default 1
        Number of processes for parallel intra-precluster linking. `mp=8` is typical for
        ~1M rows on a shared node.

    contain : bool, default True
        Enable **containment rule**: if j lies inside i’s ellipse scaled by (1+contain_margin),
        link immediately (or vice versa). **Preempts** distance tests. *Primary for hybrid.*

    contain_margin : float, default 0.50
        Scale factor margin (e.g., 0.50 ⇒ semiaxes ×1.5). Increasing pulls in borderline
        in-ellipse neighbors; decreasing is stricter.

    Returns
    -------
    cat : astropy.table.Table
        The **same table** with columns appended (dtypes fixed):
        GROUP_ID (int32), GROUP_NAME (str10), GROUP_MULT (int16), GROUP_PRIMARY (bool),
        GROUP_RA (float64), GROUP_DEC (float64), GROUP_DIAMETER (float32).

    Notes
    -----
    Core geometry (directional radius toward neighbor j):
        r_eff_i(θ) = (a_i * b_i) / sqrt( (b_i cos(θ − PA_i))^2 + (a_i sin(θ − PA_i))^2 ),
        where a_i = 0.5 * DIAM_i (arcmin), b_i = max(BA_i, q_floor) * a_i,
        θ = atan2(Δx, Δy) on the local tangent plane (deg). If BA/PA missing, r_eff_i = a_i.

    Hybrid pair rules:
        Small–small:    d ≤ 0.5 * mfac * (r_i + r_j)
        Large–small:    r'_i = max(r_i, k_floor*a_i), r'_j = max(r_j, k_floor*a_j);
                         d ≤ 0.5 * mfac_sat * (r'_i + r'_j)
        Large–large:    d ≤ 0.5 * mfac_backbone * (a_i + a_j)

    Guardrails:
        dmin ≤ thresholds ≤ dmax; preclustering uses `sphere_link_arcmin` (default = dmax);
        neighbor search uses a grid of `grid_cell_arcmin` (default = dmax).
    """
    t0 = time.time()

    required = {'RA', 'DEC', 'DIAM'}
    if not required <= set(cat.colnames):
        raise ValueError(f"Catalog must have columns {required}")

    RA = np.asarray(cat['RA'], dtype=float)
    DEC = np.asarray(cat['DEC'], dtype=float)
    DIAM = np.asarray(cat['DIAM'], dtype=float)
    N = len(cat)
    BA = np.asarray(cat['BA'], dtype=float) if 'BA' in cat.colnames else np.full(N, np.nan)
    PA = np.asarray(cat['PA'], dtype=float) if 'PA' in cat.colnames else np.full(N, np.nan)

    ll_arcmin = sphere_link_arcmin if sphere_link_arcmin is not None else dmax * ARCMIN_PER_DEG
    ll_deg = float(ll_arcmin) / ARCMIN_PER_DEG
    grp, mult_pre, frst, nxt = _spheregroup(RA, DEC, ll_deg)
    n_pre = int(grp.max()) + 1 if len(grp) else 0
    t1 = time.time()
    log.info(f"[1/3] Precluster: {n_pre} preclusters, link={ll_arcmin:.2f}' in {t1 - t0:.2f}s")

    ug = np.unique(grp)
    clusters = []
    for g in ug:
        members = []
        i = frst[g]
        while i != -1:
            members.append(i)
            i = nxt[i]
        if len(members) >= 2:
            clusters.append(members)

    dsu = DSU(N)
    dmax_arcmin = dmax * ARCMIN_PER_DEG
    dmin_arcmin = dmin * ARCMIN_PER_DEG

    cell_arcmin = float(grid_cell_arcmin) if grid_cell_arcmin is not None else dmax_arcmin
    params = dict(
        anisotropic=anisotropic,
        link_mode=link_mode,
        mfac=mfac,
        big_diam=big_diam,
        mfac_backbone=mfac_backbone,
        mfac_sat=mfac_sat,
        k_floor=k_floor,
        q_floor=q_floor,
        dmax_arcmin=dmax_arcmin,
        dmin_arcmin=dmin_arcmin,
        cell_arcmin=cell_arcmin,
        contain=contain,
        contain_margin=contain_margin,
    )

    links = 0
    if mp is None or mp <= 1:
        _init_pool(RA, DEC, DIAM, BA, PA, params)
        for members in clusters:
            edges = _process_cluster(members)
            if edges.size:
                for a, b in edges:
                    dsu.union(int(a), int(b))
                links += int(edges.shape[0])
    else:
        mp = int(mp)
        with ProcessPoolExecutor(max_workers=mp, initializer=_init_pool,
                                 initargs=(RA, DEC, DIAM, BA, PA, params)) as ex:
            futures = [ex.submit(_process_cluster, members) for members in clusters]
            for fut in as_completed(futures):
                edges = fut.result()
                if edges.size:
                    for a, b in edges:
                        dsu.union(int(a), int(b))
                    links += int(edges.shape[0])
    t2 = time.time()
    log.info(f"[2/3] Intra-precluster linking (grid+hybrid, mp={mp if mp else 1}): "
             f"{links} links in {t2 - t1:.2f}s (grid={cell_arcmin:.2f}')")

    roots = np.array([dsu.find(i) for i in range(N)], dtype=np.int64)
    uniq, inv = np.unique(roots, return_inverse=True)
    group_ids = (inv + group_id_start).astype(np.int32, copy=False)
    mult = np.bincount(inv, minlength=len(uniq))

    grp_ra = np.zeros(len(uniq), dtype=np.float64)
    grp_dec = np.zeros(len(uniq), dtype=np.float64)
    grp_diam = np.zeros(len(uniq), dtype=np.float32)
    grp_primary = np.zeros(len(uniq), dtype=np.int64)

    for gidx, r in enumerate(uniq):
        members = np.where(roots == r)[0]
        w = np.clip(DIAM[members], 1e-3, None)
        ra_rad = RA[members] * DEG2RAD
        dec_rad = DEC[members] * DEG2RAD
        cosd = np.cos(dec_rad)
        xyz = np.vstack((cosd * np.cos(ra_rad), cosd * np.sin(ra_rad), np.sin(dec_rad))).T
        cen = (xyz * w[:, None]).sum(axis=0) / w.sum()
        nrm = np.linalg.norm(cen)
        if nrm > 0.0:
            cen /= nrm
        x, y, z = cen
        ra_c = math.atan2(y, x)
        if ra_c < 0.0:
            ra_c += 2.0 * math.pi
        dec_c = math.atan2(z, math.hypot(x, y))
        ra_c *= RAD2DEG
        dec_c *= RAD2DEG
        grp_ra[gidx], grp_dec[gidx] = ra_c, dec_c

        max_extent = 0.0
        for k in members:
            dra = (RA[k] - ra_c) * math.cos(dec_c * DEG2RAD)
            ddec = (DEC[k] - dec_c)
            dcen_arcmin = math.hypot(dra, ddec) * ARCMIN_PER_DEG
            rk = 0.5 * DIAM[k]
            if dcen_arcmin + rk > max_extent:
                max_extent = dcen_arcmin + rk
        grp_diam[gidx] = np.float32(2.0 * max_extent)
        grp_primary[gidx] = int(members[np.argmax(DIAM[members])])

    row_group_id = group_ids
    row_mult = mult[inv].astype(np.int16, copy=False)
    row_primary = np.zeros(N, dtype=bool)
    for gidx, r in enumerate(uniq):
        row_primary[grp_primary[gidx]] = True
    row_grp_ra = grp_ra[inv].astype(np.float64, copy=False)
    row_grp_dec = grp_dec[inv].astype(np.float64, copy=False)
    row_grp_diam = grp_diam[inv].astype(np.float32, copy=False)

    if name_via == "radec":
        names = np.array([radec_to_groupname(row_grp_ra[i], row_grp_dec[i]) for i in range(N)], dtype='U10')
    else:
        names = np.full(N, '', dtype='U10')

    def _attach(name: str, data, dtype=None):
        if dtype is not None:
            col = Column(np.asarray(data, dtype=dtype), name=name)
        else:
            col = Column(data, name=name)
        if name in cat.colnames:
            cat.replace_column(name, col)
        else:
            cat.add_column(col)

    _attach('GROUP_ID', row_group_id, np.int32)
    _attach('GROUP_NAME', names, 'U10')
    _attach('GROUP_MULT', row_mult, np.int16)
    _attach('GROUP_PRIMARY', row_primary, bool)
    _attach('GROUP_RA', row_grp_ra, np.float64)
    _attach('GROUP_DEC', row_grp_dec, np.float64)
    _attach('GROUP_DIAMETER', row_grp_diam, np.float32)

    t3 = time.time()

    # Summary buckets
    n_total = len(uniq)
    n1 = int(np.sum(mult == 1))
    n2 = int(np.sum(mult == 2))
    n3_5 = int(np.sum((mult >= 3) & (mult <= 5)))
    n6_10 = int(np.sum((mult >= 6) & (mult <= 10)))
    n10p = int(np.sum(mult > 10))

    log.info(f"[3/3] Aggregate & annotate: {t3 - t2:.2f}s")
    log.info(
        "Summary: groups=%d | singles=%d | pairs=%d | 3–5=%d | 6–10=%d | >10=%d | "
        "links=%d | preclusters=%d | mode=%s | big_diam=%.2f' | mfac=%.2f | "
        "backbone=%.2f | sat=%.2f | k_floor=%.2f | q_floor=%.2f | contain=%s(+%.0f%%) | "
        "dmax=%.2f' | grid=%.2f' | mp=%d",
        n_total, n1, n2, n3_5, n6_10, n10p,
        links, len(ug), link_mode, big_diam, mfac, mfac_backbone,
        mfac_sat, k_floor, q_floor, "on" if contain else "off", contain_margin * 100.0,
        dmax * ARCMIN_PER_DEG, cell_arcmin, (mp if mp else 1)
    )

    return cat


def qa(*args, **kwargs):
    log.info("qa(): no-op placeholder.")
