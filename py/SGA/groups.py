"""
SGA groups.py — group finding for SGA-2025

This module implements high-performance group finding on the sky using:
  • Fast spherical preclustering via pydl's `spheregroup`
  • Intra-precluster linking with a grid-based neighbor search
  • Hybrid anisotropic link thresholds using BA/PA when available
  • Optional ellipse-containment linking
  • Optional post-pass center-merging to remove name collisions
  • Multiprocessing across preclusters

Assumptions
-----------
- Astropy available
- `from SGA.logger import log` succeeds
- `from SGA.io import radec_to_groupname` succeeds
- `pydl.pydlutils.spheregroup.spheregroup` installed

Appended columns (exact dtypes)
--------------------------------
GROUP_ID        int32
GROUP_NAME      str10
GROUP_MULT      int16
GROUP_PRIMARY   bool
GROUP_RA        float64     # deg
GROUP_DEC       float64     # deg
GROUP_DIAMETER  float32     # arcmin
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
    """Wrap angle(s) into [0, 360) degrees."""
    return np.mod(x, 360.0)


def _angdiff_deg(a: float, b: float) -> float:
    """Compute signed difference a-b wrapped into (-180, 180] degrees."""
    return (a - b + 180.0) % 360.0 - 180.0


def _bearing_pa_deg_local(dx_deg: float, dy_deg: float) -> float:
    """
    Bearing (astronomical PA) on the local tangent plane from i→j.

    Parameters
    ----------
    dx_deg : float
        (RA_j − RA_i) * cos(dec0) in degrees.
    dy_deg : float
        (DEC_j − DEC_i) in degrees.

    Returns
    -------
    float
        PA in degrees with 0°=North, 90°=East in [0,360).
    """
    pa = math.degrees(math.atan2(dx_deg, dy_deg))
    if pa < 0.0:
        pa += 360.0
    return pa


class DSU:
    """Disjoint Set Union with path compression and union by rank."""

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
    """Initializer for worker processes: cache arrays and parameters in globals."""
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
    """Hybrid/anisotropic link threshold (arcmin) for pair (i, j) using local-plane deltas."""
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
    True if j lies inside the *scaled* ellipse of i (or circle if BA/PA missing).
    Uses local-plane deltas (dx, dy) in degrees.
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


def _process_cluster(members: list[int], contain_search_arcmin: float=15.0/60.0) -> np.ndarray:
    """Process a single precluster and return edges (i, j) to union (E×2 int64)."""
    p = _G_params
    contain = bool(p.get('contain', True))
    contain_margin = float(p.get('contain_margin', 0.50))
    big_diam = float(p.get('big_diam', 3.0))
    RA = _G_RA
    DEC = _G_DEC

    m = len(members)
    if m < 2:
        return np.empty((0, 2), dtype=np.int64)

    idx = np.asarray(members, dtype=np.int64)
    dec0 = float(np.median(DEC[idx]))
    cosd0 = math.cos(dec0 * DEG2RAD)
    ra0 = float(np.median(RA[idx]))
    dx = _angdiff_deg(RA[idx], ra0) * cosd0
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

                    # make sure we link, e.g., NGC5194 and NGC5195
                    sum_a = (_G_a_arc[i_global] + _G_a_arc[j_global]) * (1.0 + contain_margin)  # arcmin
                    if contain and dd_arcmin <= contain_search_arcmin:
                        if _contains_pair(i_global, j_global, ddx, ddy, 1.0 + contain_margin) or \
                           _contains_pair(j_global, i_global, -ddx, -ddy, 1.0 + contain_margin):
                            edges.append((i_global, j_global))
                            continue

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


def make_singleton_group(cat, group_id_start=0):
    out = cat.copy(copy_data=True)
    n = len(out)
    names = [radec_to_groupname(out['RA'][k], out['DEC'][k]) for k in range(n)]
    out['GROUP_ID'] = np.arange(group_id_start, group_id_start+n, dtype=np.int32)
    out['GROUP_NAME'] = np.array(names, dtype='U10').squeeze()
    out['GROUP_MULT'] = np.ones(n, dtype=np.int16)
    out['GROUP_PRIMARY'] = np.ones(n, dtype=bool)
    out['GROUP_RA'] = out['RA'].astype(np.float64, copy=False)
    out['GROUP_DEC'] = out['DEC'].astype(np.float64, copy=False)
    out['GROUP_DIAMETER'] = out['DIAM'].astype(np.float32, copy=False)
    return out


def build_group_catalog(
    cat: Table,
    group_id_start: int = 0,
    mfac: float = 1.5,
    dmin: float = 36.0/3600.0,      # deg (36")
    dmax: float = 3.0/60.0,         # deg (3') maximum candidate separation
    anisotropic: bool = True,
    link_mode: str = "hybrid",      # "off" | "anisotropic" | "hybrid"
    big_diam: float = 3.0,          # arcmin threshold for "large" galaxies
    mfac_backbone: float = 2.0,     # large–large multiplier
    mfac_sat: float = 1.5,          # large–small multiplier
    k_floor: float = 0.40,          # floor factor for softened anisotropy
    q_floor: float = 0.20,          # minimum BA
    name_via: str = "radec",
    sphere_link_arcmin: float | None = None, # precluster link length; default uses dmax
    grid_cell_arcmin: float | None = None,   # grid bin size; default = dmax
    mp: int = 1,                             # number of processes for parallel precluster processing
    contain: bool = True,                    # enable ellipse-containment linking
    contain_margin: float = 0.50,            # expansion on axes (1+margin)
    merge_centers: bool = True,              # merge groups with centers within threshold
    merge_sep_arcsec: float = 52.,           # merge groups closer than this value
    contain_search_arcmin: float 10.0,       # initial check for wide-separation, large-galaxy pairs
    min_group_diam_arcsec: float = 30.,       # minimum group diameter [arcsec]
    manual_merge_pairs: list[tuple[str, str]] | None = None,
    name_column: str = "OBJNAME",
) -> Table:
    """
    Build a mosaic-friendly group catalog using a hybrid anisotropic rule.

    (Docstring trimmed for brevity; only duplicate-name check and default merge_sep_arcsec changed.)
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
            edges = _process_cluster(members, contain_search_arcmin=contain_search_arcmin)
            if edges.size:
                for a, b in edges:
                    dsu.union(int(a), int(b))
                links += int(edges.shape[0])
    else:
        mp = int(mp)
        with ProcessPoolExecutor(max_workers=mp, initializer=_init_pool,
                                 initargs=(RA, DEC, DIAM, BA, PA, params)) as ex:
            futures = [ex.submit(_process_cluster, members, contain_search_arcmin) for members in clusters]
            for fut in as_completed(futures):
                edges = fut.result()
                if edges.size:
                    for a, b in edges:
                        dsu.union(int(a), int(b))
                    links += int(edges.shape[0])
    t2 = time.time()
    log.info(f"[2/3] Intra-precluster linking (grid+hybrid, mp={mp if mp else 1}): "
             f"{links} links in {t2 - t1:.2f}s (grid={cell_arcmin:.2f}')")

    # Optional: merge provisional groups whose centers are within merge_sep_arcsec
    if merge_centers:
        roots_tmp = np.array([dsu.find(i) for i in range(N)], dtype=np.int64)
        uniq_tmp, inv_tmp = np.unique(roots_tmp, return_inverse=True)

        # Representative member per provisional group (first occurrence)
        reps = np.zeros(len(uniq_tmp), dtype=np.int64)
        first_seen = {}
        for i, g in enumerate(inv_tmp):
            if g not in first_seen:
                first_seen[g] = i
        for g, irep in first_seen.items():
            reps[g] = irep

        # Provisional DIAM-weighted centers on the unit sphere
        ra_c = np.zeros(len(uniq_tmp), dtype=float)
        dec_c = np.zeros(len(uniq_tmp), dtype=float)
        for gi, rlab in enumerate(uniq_tmp):
            idx = np.where(roots_tmp == rlab)[0]
            w = np.clip(DIAM[idx], 1e-3, None)
            ra_rad = RA[idx] * DEG2RAD
            dec_rad = DEC[idx] * DEG2RAD
            cosd = np.cos(dec_rad)
            x = (cosd * np.cos(ra_rad) * w).sum() / w.sum()
            y = (cosd * np.sin(ra_rad) * w).sum() / w.sum()
            z = (np.sin(dec_rad) * w).sum() / w.sum()
            nrm = math.sqrt(x*x + y*y + z*z)
            if nrm > 0.0:
                x, y, z = x/nrm, y/nrm, z/nrm
            ra_c[gi]  = (math.atan2(y, x) % (2.0*math.pi)) * RAD2DEG
            dec_c[gi] = math.atan2(z, math.hypot(x, y)) * RAD2DEG

        # Group-center stitching via spheregroup at merge_sep_arcsec
        ll_cent_deg = float(merge_sep_arcsec) / 3600.0
        gcent, mcent, fcent, nxcent = _spheregroup(ra_c, dec_c, ll_cent_deg)

        # Union all representatives within each center precluster
        for gc in np.unique(gcent):
            i = fcent[gc]
            if i == -1:
                continue
            j = nxcent[i]
            while j != -1:
                dsu.union(int(reps[i]), int(reps[j]))
                j = nxcent[j]

    # manually merge certain groups
    if manual_merge_pairs:
        # map object names -> row indices (assume names are unique; if not, take first match)
        names_arr = np.asarray(cat[name_column]).astype(str)
        index_of = {n: i for i, n in enumerate(names_arr)}

        merged_pairs = 0
        missing = []
        for a_name, b_name in manual_merge_pairs:
            ia = index_of.get(str(a_name))
            ib = index_of.get(str(b_name))
            if ia is None or ib is None:
                missing.append((a_name, b_name))
                continue
            dsu.union(int(ia), int(ib))
            merged_pairs += 1

        if missing:
            log.warning("manual_merge_pairs: %d pair(s) had missing name(s): %s",
                        len(missing), missing[:5])
        if merged_pairs:
            log.info("manual_merge_pairs: merged %d pair(s) by hand.", merged_pairs)


    # Final roots
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
            dra = _angdiff_deg(RA[k], ra_c) * math.cos(dec_c * DEG2RAD)
            ddec = (DEC[k] - dec_c)
            dcen_arcmin = math.hypot(dra, ddec) * ARCMIN_PER_DEG
            rk = 0.5 * DIAM[k]
            if dcen_arcmin + rk > max_extent:
                max_extent = dcen_arcmin + rk
        # minimum group diameter [arcmin]
        #grp_diam[gidx] = np.float32(2.0 * max_extent)
        grp_diam[gidx] = max(2. * max_extent, min_group_diam_arcsec/60.)
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
        """Attach/replace a column with controlled dtype; squeeze to avoid str10[1]."""
        if dtype is not None:
            col = Column(np.asarray(data, dtype=dtype).squeeze(), name=name)
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

    # Group-level duplicate-name warning (not exception)
    if len(cat) > 0:
        gid_unique, idx_first = np.unique(cat['GROUP_ID'], return_index=True)
        group_names = np.asarray(cat['GROUP_NAME'])[idx_first]
        un, counts = np.unique(group_names, return_counts=True)
        dups = un[counts > 1]
        if dups.size > 0:
            #log.warning("Duplicate GROUP_NAME(s) after center-merge: %d unique name(s) duplicated; examples: %s",
            #            int(dups.size), dups[:10].tolist())
            log.warning("Duplicate GROUP_NAME(s) after center-merge.")

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
    """Placeholder QA hook (no-op)."""
    log.info("qa(): no-op placeholder.")


def set_overlap_bit(cat, SAMPLE):
    """
    Flag ellipse-overlap within each group by setting SAMPLE['OVERLAP'].

    Assumes `cat` has columns: GROUP_NAME, GROUP_MULT, RA, DEC, DIAM (arcmin), BA, PA (deg, astronomical).
    Modifies `cat['SAMPLE']` in place by OR'ing the OVERLAP bit for members that
    overlap at least one other member in their group.

    Parameters
    ----------
    cat : astropy.table.Table
        Input catalog, modified in place.
    SAMPLE : dict
        Bitmask dictionary that includes key 'OVERLAP'.

    """
    OVERLAP_BIT = SAMPLE['OVERLAP']
    DEG2RAD = np.pi / 180.0
    ARCMIN_PER_DEG = 60.0

    def _angdiff_deg(a, b):
        """(a-b) wrapped to (-180, 180] deg."""
        return (a - b + 180.0) % 360.0 - 180.0


    def _dir_radius_arcmin(a_arc, b_arc, pa_rad, bearing_rad):
        """
        Radius (arcmin) of an ellipse with semi-axes (a_arc, b_arc) and PA=pa_rad
        along a ray at `bearing_rad` (astronomical: 0°=N, 90°=E).
        """
        d = bearing_rad - pa_rad
        # wrap to [-pi, pi] for numerical stability
        d = np.where(d > np.pi, d - 2.0*np.pi, d)
        d = np.where(d < -np.pi, d + 2.0*np.pi, d)
        denom = np.hypot(b_arc * np.cos(d), a_arc * np.sin(d))
        # fallback to circular if degenerate
        out = (a_arc * b_arc) / np.where(denom > 0.0, denom, 1.0)
        out = np.where(denom > 0.0, out, a_arc)
        return out

    # Work only on groups with >1 member
    mask_mult = (cat['GROUP_MULT'] > 1)
    if not np.any(mask_mult):
        return  # nothing to do

    # Unique group names among multi-member groups
    gnames = np.asarray(cat['GROUP_NAME'][mask_mult]).astype(str)
    ugroups = np.unique(gnames)

    # Column views (avoid repeated table lookups)
    RA_all   = np.asarray(cat['RA'], dtype=float)
    DEC_all  = np.asarray(cat['DEC'], dtype=float)
    DIAM_all = np.asarray(cat['DIAM'], dtype=float)  # arcmin
    BA_all   = np.asarray(cat['BA'], dtype=float) if 'BA' in cat.colnames else np.full(len(cat), np.nan)
    PA_all   = np.asarray(cat['PA'], dtype=float) if 'PA' in cat.colnames else np.full(len(cat), np.nan)
    GNAME    = np.asarray(cat['GROUP_NAME']).astype(str)

    for gname in ugroups:
        I = np.where(GNAME == gname)[0]
        if I.size < 2:
            continue

        # Local tangent-plane scale for this group
        dec0 = float(np.median(DEC_all[I]))
        cosd0 = np.cos(dec0 * DEG2RAD)

        # Per-member geometry
        ra  = RA_all[I]
        dec = DEC_all[I]
        diam = DIAM_all[I]                        # arcmin
        a_arc = 0.5 * diam                        # semi-major (arcmin)

        ba   = BA_all[I]
        pa   = PA_all[I]
        ba_eff = np.where(np.isfinite(ba) & (ba > 0.0), ba, 1.0)  # circular if missing/invalid
        b_arc = ba_eff * a_arc
        pa_rad = np.where(np.isfinite(pa), pa, 0.0) * DEG2RAD

        overlapped = np.zeros(I.size, dtype=bool)

        # Pairwise checks (upper triangle)
        for ii in range(I.size - 1):
            # Local deltas (deg), wrap RA
            dx_deg = _angdiff_deg(ra[ii+1:], ra[ii]) * cosd0
            dy_deg = (dec[ii+1:] - dec[ii])

            # Convert to arcmin and compute separations + bearings
            dx_am = dx_deg * ARCMIN_PER_DEG
            dy_am = dy_deg * ARCMIN_PER_DEG
            sep_am = np.hypot(dx_am, dy_am)
            bearing_ij = np.arctan2(dx_am, dy_am)      # rad, 0=N, 90=E
            bearing_ji = np.arctan2(-dx_am, -dy_am)

            # Directional radii along the center-center line
            ri_dir = _dir_radius_arcmin(a_arc[ii],         b_arc[ii],         pa_rad[ii],         bearing_ij)
            rj_dir = _dir_radius_arcmin(a_arc[ii+1:],      b_arc[ii+1:],      pa_rad[ii+1:],      bearing_ji)

            # Overlap condition: separation <= sum of directional radii
            touches = sep_am <= (ri_dir + rj_dir)
            if np.any(touches):
                overlapped[ii] = True
                overlapped[ii+1:][touches] = True

        if np.any(overlapped):
            cat['SAMPLE'][I[overlapped]] |= OVERLAP_BIT


def remove_small_groups(cat, minmult=2, maxmult=None, mindiam=0.5,
                        diamcolumn='D26', diamerrcolumn=None,
                        exclude_group_names=None):
    """Return a new catalog containing only groups with at least
    `minmult` members whose *all* members have diameters between
    `mindiam` and `maxdiam` based on `diamcolumn` (in arcminutes).

    Early catalogs had duplicate GROUP_IDs, so use GROUP_NAME.

    """
    # Groups to exclude entirely (e.g. LVD groups)
    if exclude_group_names is None:
        exclude_mask = np.zeros(len(cat), dtype=bool)
    else:
        exclude_mask = np.isin(cat['GROUP_NAME'], exclude_group_names)

    # Only consider groups with >= minmult members
    mask_multi = cat['GROUP_MULT'] >= minmult
    if maxmult is not None:
        mask_multi *= (cat['GROUP_MULT'] <= maxmult)
    mask_multi &= ~exclude_mask

    gname = cat['GROUP_NAME'][mask_multi]
    diam = cat[diamcolumn][mask_multi]
    if diamerrcolumn is not None:
        diamerr = cat[diamerrcolumn][mask_multi]
        diam += diamerr # lower limit

    # Sort by group name so each group is contiguous
    order = np.argsort(gname)
    gname_sorted = gname[order]
    diam_sorted = diam[order]

    # For each group, find start index and per-group min/max diameter
    unique_gnames, idx_start = np.unique(gname_sorted, return_index=True)
    group_max = np.maximum.reduceat(diam_sorted, idx_start)

    # Groups where ALL members are < mindiam  ⇒ max < mindiam
    rem_group_names = unique_gnames[group_max < mindiam]
    rem_rows = np.isin(cat['GROUP_NAME'], rem_group_names) & mask_multi
    keep_rows = (~rem_rows) & mask_multi

    rem = cat[rem_rows]
    rem = rem[np.lexsort((rem[diamcolumn], rem['GROUP_NAME']))]

    out = cat[keep_rows]
    out = out[np.lexsort((out[diamcolumn], out['GROUP_NAME']))]

    return out, rem


def find_blended_groups(small_groups, g_sorted, RA_sorted, DEC_sorted,
                        D_sorted, uniq, idx_start, idx_end):
    """Return group names among `small_groups` where at least one
    pair of circularized ellipses overlaps.

    Parameters
    ----------
    small_groups : array-like
        Group names (as in uniq[]) whose members all have D < mindiam.
    g_sorted : array
        GROUP_NAME column sorted by name and filtered to gm2_notLVD rows.
    RA_sorted, DEC_sorted : arrays
        Sorted RA, DEC values aligned with g_sorted.
    D_sorted : array
        Sorted diameters (arcmin) aligned with g_sorted.
    uniq : array
        Unique group names (sorted) corresponding to g_sorted.
    idx_start, idx_end : arrays
        Start and end indices into the sorted arrays for each uniq[] entry.

    Returns
    -------
    np.ndarray
        Sorted unique group names for which overlap was detected.

    """
    blended = []

    for gname in small_groups:
        # locate this group's slice in sorted arrays
        k = np.where(uniq == gname)[0][0]
        i0, i1 = idx_start[k], idx_end[k]

        ra  = RA_sorted[i0:i1]
        dec = DEC_sorted[i0:i1]
        d   = D_sorted[i0:i1]
        r   = 0.5 * d   # circularized radius (arcmin)

        n = len(ra)
        if n < 2:
            continue

        found = False

        for ii in range(n - 1):
            for jj in range(ii + 1, n):

                # Tangent-plane separation (arcmin)
                dec_mean = 0.5 * (dec[ii] + dec[jj]) * np.pi / 180.0
                dx = (ra[jj] - ra[ii]) * np.cos(dec_mean) * 60.0
                dy = (dec[jj] - dec[ii]) * 60.0
                dist2 = dx*dx + dy*dy

                # overlap check
                thresh = r[ii] + r[jj]
                if dist2 <= thresh * thresh:
                    blended.append(gname)
                    found = True
                    break

            if found:
                break

    return np.unique(blended)


def remove_small_groups_and_galaxies(parent, ref_tab, region, REGIONBITS,
                                     SAMPLE, ELLIPSEBIT, mindiam=0.5,
                                     veto_objnames=None):
    """Update parent['REGION'] by removing this region for objects that
    do NOT fall in VI samples 001–007, based on a previous-version catalog
    `ref_tab` for a single region.

    Parameters
    ----------
    parent : astropy.table.Table
        Current parent catalog (modified in-place; also returned).
    ref_tab : astropy.table.Table
        Previous-version regional catalog with GROUP_* and ellipse info
        (equivalent to `fullsample`).
    region : str
        'dr9-north' or 'dr11-south', etc. Must be a key in REGIONBITS.
    REGIONBITS : dict
        Mapping from region_name -> bit value.
    SAMPLE : dict
        SAMPLE bit dictionary with keys LVD, MCLOUDS, GCLPNE, NEARSTAR, INSTAR.

    Returns
    -------
    parent_new : astropy.table.Table
        Parent with this region bit removed where appropriate, and rows
        with REGION==0 dropped.

    """
    bit = REGIONBITS[region]

    LVD      = SAMPLE['LVD']
    MCLOUDS  = SAMPLE['MCLOUDS']
    GCLPNE   = SAMPLE['GCLPNE']
    NEARSTAR = SAMPLE['NEARSTAR']
    INSTAR   = SAMPLE['INSTAR']

    gm1    = (ref_tab['GROUP_MULT'] == 1)
    not_LVD = (ref_tab['SAMPLE'] & LVD) == 0
    is_LVD  = (ref_tab['SAMPLE'] & LVD) != 0

    ellipse_ok  = (ref_tab['ELLIPSEBIT'] == 0)
    ellipse_bad = (ref_tab['ELLIPSEBIT'] != 0)

    sample_flags = (ref_tab['SAMPLE'] & (MCLOUDS | GCLPNE | NEARSTAR | INSTAR)) != 0

    D = ref_tab['D26']

    # 001–004: GM=1, non-LVD, D≥0.5
    mask_001_004 = gm1 & not_LVD & (D >= mindiam)

    # 005: all members of LVD groups
    LVD_group_names = np.unique(ref_tab['GROUP_NAME'][is_LVD])
    mask_005 = np.isin(ref_tab['GROUP_NAME'], LVD_group_names)

    # 006–007: non-LVD, GM≥2, at least one member with D≥mindiam
    #          OR at least one overlapping circularized pair
    gm2_notLVD = (ref_tab['GROUP_MULT'] >= 2) & not_LVD

    D_all   = ref_tab['DIAM_INIT'] # use initial diameter not D26!
    RA_all  = ref_tab['RA_INIT']
    DEC_all = ref_tab['DEC_INIT']
    g_all   = ref_tab['GROUP_NAME']

    D   = D_all[gm2_notLVD]
    RA  = RA_all[gm2_notLVD]
    DEC = DEC_all[gm2_notLVD]
    g   = g_all[gm2_notLVD]

    order = np.argsort(g)
    g_sorted   = g[order]
    D_sorted   = D[order]
    RA_sorted  = RA[order]
    DEC_sorted = DEC[order]

    uniq, idx_start = np.unique(g_sorted, return_index=True)
    idx_end = np.append(idx_start[1:], len(g_sorted))

    group_maxD = np.maximum.reduceat(D_sorted, idx_start) # groups kept by diameter only
    good_groups_diam = uniq[group_maxD >= mindiam]

    # groups with all members below mindiam
    small_groups = uniq[group_maxD < mindiam]

    # find small groups with overlapping circularized ellipses
    blended_groups = find_blended_groups(
        small_groups, g_sorted, RA_sorted, DEC_sorted, D_sorted,
        uniq, idx_start, idx_end)

    good_groups = np.union1d(good_groups_diam, blended_groups) # diameter OR blended
    mask_006_007 = gm2_notLVD & np.isin(g_all, good_groups)

    # union of 001–007 in ref_tab
    mask_all = mask_001_004 | mask_005 | mask_006_007

    # OBJNAMEs in this region that PASS the 001–007 selection
    keep_names = set(ref_tab['OBJNAME'][mask_all])

    # For parent rows in this region and present in ref_tab:
    #parent = parent.copy()
    in_region = (parent['REGION'] & bit) != 0
    in_ref = np.isin(parent['OBJNAME'], ref_tab['OBJNAME'])

    # Objects that fail samples 001–007
    fails = in_region & in_ref & ~np.isin(parent['OBJNAME'], list(keep_names))

    # Apply veto: do NOT drop veto_names
    if veto_objnames is not None:
        veto_mask = np.isin(parent['OBJNAME'], list(veto_objnames))
        fails &= ~veto_mask

    parent['REGION'][fails] -= bit

    # report the numbers but don't actually trim
    I = parent['REGION'] != 0
    log.info(f'Removing {np.sum(~I):,d}/{np.sum(in_region):,d} objects with ' + \
             f'D(26)<0.5 from region {region}')

    return parent
