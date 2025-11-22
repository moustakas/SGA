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


def _process_cluster(members: list[int]) -> np.ndarray:
    """Process a single precluster and return edges (i, j) to union (E×2 int64)."""
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
    sphere_link_arcmin: float | None = None,  # precluster link length; default uses dmax
    grid_cell_arcmin: float | None = None,    # grid bin size; default = dmax
    mp: int = 1,                               # number of processes for parallel precluster processing
    contain: bool = True,                      # enable ellipse-containment linking
    contain_margin: float = 0.50,              # expansion on axes (1+margin)
    merge_centers: bool = True,                # merge groups with centers within threshold
    merge_sep_arcsec: float = 52.,             # merge groups closer than this value
    min_group_diam_arcsec: float = 30.,        # minimum group diameter [arcsec]
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
