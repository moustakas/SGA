"""
SGA groups.py — Refactored group finding for SGA-2025

Key improvements over original:
- Replaced 11 global variables with State class
- Separated concerns: geometry, grid search, linking logic
- Better error handling and validation
- Enhanced statistics reporting
- Maintained identical algorithm behavior

Algorithm: spherical preclustering + grid-based linking with hybrid anisotropic thresholds
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, List, Optional
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


# ============================================================================
# Geometric Utilities (pure functions, easily testable)
# ============================================================================

def wrap_deg(x):
    """Wrap angle(s) into [0, 360) degrees."""
    return np.mod(x, 360.0)


def angdiff_deg(a, b):
    """Signed difference a-b wrapped into (-180, 180] degrees."""
    return (a - b + 180.0) % 360.0 - 180.0


def directional_radius(a_arc, b_arc, pa_rad, bearing_rad):
    """
    Ellipse radius (arcmin) along a bearing direction.

    Works with both scalars and arrays (fully vectorized).

    Parameters
    ----------
    a_arc, b_arc : float or array-like
        Semi-major and semi-minor axes (arcmin)
    pa_rad : float or array-like
        Position angle (radians, astronomical convention)
    bearing_rad : float or array-like
        Bearing direction (radians, astronomical convention)

    Returns
    -------
    float or ndarray
        Radius in arcmin along specified bearing

    """
    # Convert to arrays for uniform handling
    a_arc = np.asarray(a_arc)
    b_arc = np.asarray(b_arc)
    pa_rad = np.asarray(pa_rad)
    bearing_rad = np.asarray(bearing_rad)

    # Determine if output should be scalar (all inputs are 0-d)
    scalar_output = (a_arc.ndim == 0 and b_arc.ndim == 0 and 
                     pa_rad.ndim == 0 and bearing_rad.ndim == 0)

    # Handle degenerate cases for scalar
    if scalar_output:
        if a_arc <= 0.0 or b_arc <= 0.0:
            return float(a_arc)

    # Normalize angle difference to [-pi, pi]
    d = bearing_rad - pa_rad
    d = np.where(d > np.pi, d - 2.0*np.pi, d)
    d = np.where(d < -np.pi, d + 2.0*np.pi, d)

    # Ellipse equation in polar form
    denom = np.hypot(b_arc * np.cos(d), a_arc * np.sin(d))

    # Handle degenerate cases
    out = np.where(denom > 0.0, (a_arc * b_arc) / denom, a_arc)

    return float(out) if scalar_output else out


def contains_point(a_arc, b_arc, pa_rad, dx_deg, dy_deg, scale=1.0):
    """
    Check if point (dx, dy) lies inside scaled ellipse.

    Note: This checks if a POINT is inside an ellipse, not ellipse-ellipse overlap.
    For ellipse overlap, use ellipses_overlap().

    Parameters
    ----------
    a_arc, b_arc : float
        Semi-major and semi-minor axes (arcmin)
    pa_rad : float
        Position angle (radians, astronomical convention)
    dx_deg, dy_deg : float
        Point offset in local tangent plane (degrees)
    scale : float
        Scale factor for ellipse (>1 expands)

    Returns
    -------
    bool
        True if point is inside scaled ellipse

    """
    if a_arc <= 0.0 or b_arc <= 0.0:
        return False

    r_point = math.hypot(dx_deg, dy_deg) * ARCMIN_PER_DEG

    # Circular case
    if np.isclose(a_arc, b_arc):
        return r_point <= scale * a_arc

    bearing = math.atan2(dx_deg, dy_deg)
    r_boundary = directional_radius(a_arc, b_arc, pa_rad, bearing)

    return r_point <= scale * r_boundary


def ellipses_overlap(a1, b1, pa1_rad, a2, b2, pa2_rad, dx_deg, dy_deg, scale=1.0):
    """
    Check if two scaled ellipses overlap.

    This is the proper test for determining if two galaxies are close enough
    to be considered overlapping/interacting.

    Parameters
    ----------
    a1, b1 : float
        Semi-major and semi-minor axes of ellipse 1 (arcmin)
    pa1_rad : float
        Position angle of ellipse 1 (radians, astronomical convention)
    a2, b2 : float
        Semi-major and semi-minor axes of ellipse 2 (arcmin)
    pa2_rad : float
        Position angle of ellipse 2 (radians, astronomical convention)
    dx_deg, dy_deg : float
        Offset from ellipse 1 center to ellipse 2 center (degrees)
    scale : float
        Scale factor applied to both ellipses (e.g., 1.5 = 50% margin)

    Returns
    -------
    bool
        True if the scaled ellipses overlap

    """
    # Separation between centers
    sep_arcmin = math.hypot(dx_deg, dy_deg) * ARCMIN_PER_DEG

    # Bearing from 1 to 2 and vice versa
    bearing_12 = math.atan2(dx_deg, dy_deg)
    bearing_21 = math.atan2(-dx_deg, -dy_deg)

    # Directional radii along the center-center line
    r1_dir = directional_radius(a1, b1, pa1_rad, bearing_12)
    r2_dir = directional_radius(a2, b2, pa2_rad, bearing_21)

    # Ellipses overlap if separation <= sum of scaled directional radii
    return sep_arcmin <= scale * (r1_dir + r2_dir)


# ============================================================================
# Disjoint Set Union
# ============================================================================

class DSU:
    """Union-Find with path compression and union by rank."""

    def __init__(self, n):
        self.p = list(range(n))
        self.r = [0] * n

    def find(self, x):
        if self.p[x] != x:
            self.p[x] = self.find(self.p[x])
        return self.p[x]

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.r[ra] < self.r[rb]:
            self.p[ra] = rb
        elif self.r[ra] > self.r[rb]:
            self.p[rb] = ra
        else:
            self.p[rb] = ra
            self.r[ra] += 1
        return True


# ============================================================================
# Group Finder State (replaces global variables)
# ============================================================================

@dataclass
class Params:
    """Algorithm parameters."""
    anisotropic: bool
    link_mode: str
    mfac: float
    big_diam: float
    mfac_backbone: float
    mfac_sat: float
    k_floor: float
    q_floor: float
    dmax_arcmin: float
    dmin_arcmin: float
    cell_arcmin: float
    contain: bool
    contain_margin: float


class State:
    """Encapsulates all data for group finding (replaces globals)."""

    def __init__(self, cat, params):
        self.n = len(cat)
        self.ra = np.asarray(cat['RA'], dtype=float)
        self.dec = np.asarray(cat['DEC'], dtype=float)
        self.diam = np.asarray(cat['DIAM'], dtype=float)
        self.ba = self._get(cat, 'BA', np.nan)
        self.pa = self._get(cat, 'PA', np.nan)

        self.a_arc = 0.5 * self.diam
        ba_eff = np.where(np.isfinite(self.ba) & (self.ba > 0), self.ba, 1.0)
        if params.q_floor > 0:
            ba_eff = np.maximum(ba_eff, params.q_floor)
        self.b_arc = ba_eff * self.a_arc
        self.pa_rad = np.where(np.isfinite(self.pa), self.pa, 0.0) * DEG2RAD
        self.has_aniso = np.isfinite(self.pa) & np.isfinite(self.ba) & (self.ba > 0)
        self.params = params

    @staticmethod
    def _get(cat, name, default):
        return np.asarray(cat[name], dtype=float) if name in cat.colnames else np.full(len(cat), default, dtype=float)

    def pair_threshold(self, i, j, dx_deg, dy_deg):
        """Compute link threshold for pair (i,j)."""
        p = self.params
        ri_c, rj_c = self.a_arc[i], self.a_arc[j]

        if not p.anisotropic or p.link_mode in ('off', None):
            return 0.5 * p.mfac * (ri_c + rj_c)

        ri_a = self._dir_radius(i, dx_deg, dy_deg)
        rj_a = self._dir_radius(j, -dx_deg, -dy_deg)

        if p.link_mode == 'anisotropic':
            return 0.5 * p.mfac * (ri_a + rj_a)

        # Hybrid mode
        big_i, big_j = self.diam[i] >= p.big_diam, self.diam[j] >= p.big_diam
        if big_i and big_j:
            return 0.5 * p.mfac_backbone * (ri_c + rj_c)
        if big_i or big_j:
            ri = max(ri_a, p.k_floor * ri_c)
            rj = max(rj_a, p.k_floor * rj_c)
            return 0.5 * p.mfac_sat * (ri + rj)
        return 0.5 * p.mfac * (ri_a + rj_a)

    def _dir_radius(self, i, dx_deg, dy_deg):
        if not self.has_aniso[i]:
            return self.a_arc[i]
        bearing = math.atan2(dx_deg, dy_deg)
        return directional_radius(self.a_arc[i], self.b_arc[i], self.pa_rad[i], bearing)

    def contains(self, i, j, dx_deg, dy_deg, scale):
        """Check if j's center is inside i's scaled ellipse."""
        return contains_point(self.a_arc[i], self.b_arc[i], self.pa_rad[i], dx_deg, dy_deg, scale)

    def overlaps(self, i, j, dx_deg, dy_deg, scale):
        """Check if i and j's scaled ellipses overlap."""
        return ellipses_overlap(
            self.a_arc[i], self.b_arc[i], self.pa_rad[i],
            self.a_arc[j], self.b_arc[j], self.pa_rad[j],
            dx_deg, dy_deg, scale
        )


# ============================================================================
# Grid search
# ============================================================================

class Grid:
    """Spatial grid for neighbor search."""

    def __init__(self, dx, dy, cell_deg):
        self.gx = np.floor(dx / cell_deg).astype(np.int64)
        self.gy = np.floor(dy / cell_deg).astype(np.int64)
        self.bins = {}
        for k in range(len(dx)):
            key = (int(self.gx[k]), int(self.gy[k]))
            self.bins.setdefault(key, []).append(k)

    def neighbors(self, k):
        cx, cy = int(self.gx[k]), int(self.gy[k])
        result = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                result.extend(self.bins.get((cx+dx, cy+dy), []))
        return result


# ============================================================================
# Core linking logic
# ============================================================================

def process_cluster(members, state, contain_search_arcmin=15.0/60.0):
    """Find linkages within a precluster.

    """
    if len(members) < 2:
        return np.empty((0, 2), dtype=np.int64)

    idx = np.array(members, dtype=np.int64)
    p = state.params

    dec0 = float(np.median(state.dec[idx]))
    cosd0 = math.cos(dec0 * DEG2RAD)
    ra0 = float(np.median(state.ra[idx]))

    dx = angdiff_deg(state.ra[idx], ra0) * cosd0
    dy = state.dec[idx] - dec0

    grid = Grid(dx, dy, p.cell_arcmin / ARCMIN_PER_DEG)

    edges = []
    scale = 1.0 + p.contain_margin

    for k in range(len(members)):
        i = int(idx[k])
        for kk in grid.neighbors(k):
            if kk <= k:
                continue
            j = int(idx[kk])

            ddx, ddy = dx[kk] - dx[k], dy[kk] - dy[k]
            sep = math.hypot(ddx, ddy) * ARCMIN_PER_DEG

            if should_link(i, j, ddx, ddy, sep, state, scale, contain_search_arcmin):
                edges.append((i, j))

    return np.array(edges, dtype=np.int64) if edges else np.empty((0, 2), dtype=np.int64)


def should_link(i, j, ddx, ddy, sep, state, scale, search_arcmin):
    """Determine if pair should be linked.

    """
    p = state.params

    # Ellipse overlap check for close pairs
    if p.contain and sep <= search_arcmin:
        if state.overlaps(i, j, ddx, ddy, scale):
            return True

    if sep > p.dmax_arcmin:
        return False

    # Standard ellipse overlap check
    if p.contain and state.overlaps(i, j, ddx, ddy, scale):
        return True

    # Threshold check
    thr = max(state.pair_threshold(i, j, ddx, ddy), p.dmin_arcmin)
    return sep <= thr


# ============================================================================
# Multiprocessing support
# ============================================================================

_WORKER_STATE = None

def _init_worker(state):
    global _WORKER_STATE
    _WORKER_STATE = state

def _check_group_pair_overlap(args):
    """
    Worker function to check if two groups have overlapping members.

    Returns
    -------
    tuple or None
        (idx_i, idx_j) if overlap found, else None
    """
    root_i, root_j, pos_i, pos_j, state, scale, max_search_radius = args

    # Quick distance check
    cosd = math.cos(0.5 * (pos_i['dec'] + pos_j['dec']) * DEG2RAD)
    dra = angdiff_deg(pos_j['ra'], pos_i['ra']) * cosd
    ddec = pos_j['dec'] - pos_i['dec']
    sep_deg = math.hypot(dra, ddec)

    # Skip if centers too far apart
    if sep_deg > (pos_i['radius'] + pos_j['radius'] + max_search_radius / ARCMIN_PER_DEG):
        return None

    # Check members for overlap
    for idx_i in pos_i['members']:
        dec0 = state.dec[idx_i]
        cosd0 = math.cos(dec0 * DEG2RAD)
        ra0 = state.ra[idx_i]

        for idx_j in pos_j['members']:
            dx_deg = angdiff_deg(state.ra[idx_j], ra0) * cosd0
            dy_deg = state.dec[idx_j] - dec0

            # Quick distance check before expensive overlap test
            sep = math.hypot(dx_deg, dy_deg) * ARCMIN_PER_DEG
            max_possible_overlap = (state.a_arc[idx_i] + state.a_arc[idx_j]) * scale

            if sep > max_possible_overlap:
                continue

            # Full overlap check
            if ellipses_overlap(
                state.a_arc[idx_i], state.b_arc[idx_i], state.pa_rad[idx_i],
                state.a_arc[idx_j], state.b_arc[idx_j], state.pa_rad[idx_j],
                dx_deg, dy_deg, scale):
                return (int(idx_i), int(idx_j))

    return None


def _merge_overlapping_groups(dsu, state, contain_margin, mp=1):
    """
    Merge groups where members have overlapping ellipses.

    This catches cases like IC 4278 inside NGC 5194 that were missed
    during initial linking due to exceeding dmax.

    Vectorized version for performance with large catalogs.

    Parameters
    ----------
    dsu : DSU
        Disjoint set union structure (modified in place)
    state : State
        Group finder state with galaxy properties
    contain_margin : float
        Margin for ellipse overlap (e.g., 0.75 = 75% expansion)
    mp : int
        Number of parallel processes

    Returns
    -------
    int
        Number of group pairs merged

    """
    # Get current provisional groups
    roots = np.array([dsu.find(i) for i in range(state.n)])
    unique_roots = np.unique(roots)

    scale = 1.0 + contain_margin
    max_search_radius = 10.0  # arcmin

    # Vectorized candidate selection
    group_sizes = np.bincount(roots, minlength=unique_roots.max()+1)
    group_max_diam = np.zeros(unique_roots.max()+1)
    np.maximum.at(group_max_diam, roots, state.diam)

    candidate_mask = (group_max_diam[unique_roots] > 1.0) | (group_sizes[unique_roots] >= 10)
    candidate_roots = unique_roots[candidate_mask]

    if len(candidate_roots) == 0:
        return 0

    # Vectorized group center computation
    group_ra = np.zeros(len(candidate_roots))
    group_dec = np.zeros(len(candidate_roots))
    group_radius = np.zeros(len(candidate_roots))
    group_members = []

    for i, root in enumerate(candidate_roots):
        idx = np.where(roots == root)[0]
        group_members.append(idx)
        group_ra[i] = np.median(state.ra[idx])
        group_dec[i] = np.median(state.dec[idx])
        group_radius[i] = group_max_diam[root] * scale / ARCMIN_PER_DEG

    # Build all pair indices at once
    n_groups = len(candidate_roots)
    n_pairs = n_groups * (n_groups - 1) // 2

    log.info(f"Overlap merge: checking {n_pairs:,d} group pairs from {n_groups:,d} candidate groups")

    # Vectorized distance matrix for quick rejection
    # This avoids building pairs that are obviously too far apart
    cosd = np.cos(group_dec * DEG2RAD)
    ra_rad = group_ra * DEG2RAD
    dec_rad = group_dec * DEG2RAD

    # Cartesian coordinates for distance computation
    x = cosd * np.cos(ra_rad)
    y = cosd * np.sin(ra_rad)
    z = np.sin(dec_rad)

    # Build pair list with spatial filtering
    pair_args = []
    max_sep_deg = 2 * max(group_radius) + max_search_radius / ARCMIN_PER_DEG

    for i in range(n_groups - 1):
        # Vectorized distance to all remaining groups
        dx = x[i+1:] - x[i]
        dy = y[i+1:] - y[i]
        dz = z[i+1:] - z[i]
        chord_dist = np.sqrt(dx*dx + dy*dy + dz*dz)

        # Angular separation (small angle approximation is fine for filtering)
        # For small angles: chord ≈ 2*sin(theta/2) ≈ theta
        ang_sep = 2 * np.arcsin(np.clip(chord_dist / 2, -1, 1)) * RAD2DEG

        # Only check pairs that could plausibly overlap
        max_reach = group_radius[i] + group_radius[i+1:] + max_search_radius / ARCMIN_PER_DEG
        close_enough = ang_sep <= max_reach

        for j_offset in np.where(close_enough)[0]:
            j = i + 1 + j_offset
            pair_args.append((
                candidate_roots[i], candidate_roots[j],
                {'ra': group_ra[i], 'dec': group_dec[i],
                 'radius': group_radius[i], 'members': group_members[i]},
                {'ra': group_ra[j], 'dec': group_dec[j],
                 'radius': group_radius[j], 'members': group_members[j]},
                state, scale, max_search_radius
            ))

    total_pairs = len(pair_args)
    if total_pairs == 0:
        return 0

    log.info(f"  Spatial filtering reduced to {total_pairs:,d} plausible pairs")

    # Process pairs
    n_merges = 0

    if not mp or mp <= 1:
        for pair_idx, args in enumerate(pair_args):
            if total_pairs > 100 and pair_idx % max(1, total_pairs // 10) == 0:
                log.info(f"  Progress: {pair_idx:,d}/{total_pairs:,d} pairs checked")

            result = _check_group_pair_overlap(args)
            if result is not None:
                idx_i, idx_j = result
                if dsu.union(idx_i, idx_j):
                    n_merges += 1
    else:
        with ProcessPoolExecutor(max_workers=mp) as executor:
            futures = {executor.submit(_check_group_pair_overlap, args): i
                      for i, args in enumerate(pair_args)}

            completed = 0
            for future in as_completed(futures):
                completed += 1
                if total_pairs > 100 and completed % max(1, total_pairs // 10) == 0:
                    log.info(f"  Progress: {completed:,d}/{total_pairs:,d} pairs checked")

                result = future.result()
                if result is not None:
                    idx_i, idx_j = result
                    if dsu.union(idx_i, idx_j):
                        n_merges += 1

    return n_merges


def _worker_wrapper(args):
    members, search = args
    return process_cluster(members, _WORKER_STATE, search)


# ============================================================================
# Statistics reporting
# ============================================================================

def report_group_statistics(cat, params, links, n_preclusters, timing):
    """Generate detailed statistics about the grouping results.

    """

    # Get unique groups
    group_ids = cat['GROUP_ID']
    unique_groups = np.unique(group_ids)
    n_groups = len(unique_groups)

    # Multiplicity distribution
    mults = cat['GROUP_MULT']

    # Binned multiplicity statistics
    n_singles = int(np.sum(mults == 1))
    n_pairs = int(np.sum(mults == 2))
    n_3to5 = int(np.sum((mults >= 3) & (mults <= 5)))
    n_6to10 = int(np.sum((mults >= 6) & (mults <= 10)))
    n_11to20 = int(np.sum((mults >= 11) & (mults <= 20)))
    n_21plus = int(np.sum(mults > 20))

    # Group-level statistics (count groups not objects)
    groups_mults = np.array([mults[group_ids == gid][0] for gid in unique_groups])
    n_groups_singles = int(np.sum(groups_mults == 1))
    n_groups_pairs = int(np.sum(groups_mults == 2))
    n_groups_3to5 = int(np.sum((groups_mults >= 3) & (groups_mults <= 5)))
    n_groups_6to10 = int(np.sum((groups_mults >= 6) & (groups_mults <= 10)))
    n_groups_11to20 = int(np.sum((groups_mults >= 11) & (groups_mults <= 20)))
    n_groups_21plus = int(np.sum(groups_mults > 20))

    # Diameter statistics
    diams = cat['GROUP_DIAMETER']
    unique_diams = np.array([diams[group_ids == gid][0] for gid in unique_groups])

    # Large group thresholds (in arcmin)
    large_thresholds = [10.0, 20.0, 30.0, 40.0, 50.0]
    large_counts = {thr: int(np.sum(unique_diams >= thr)) for thr in large_thresholds}

    # Find largest groups
    top_n = 10
    sorted_idx = np.argsort(unique_diams)[::-1]
    largest_groups = []
    for i in range(min(top_n, len(sorted_idx))):
        gid = unique_groups[sorted_idx[i]]
        diam = unique_diams[sorted_idx[i]]
        mult = groups_mults[sorted_idx[i]]
        gname = cat['GROUP_NAME'][group_ids == gid][0]
        largest_groups.append((gid, gname, mult, diam))

    # Print summary
    log.info("="*80)
    log.info("GROUP CATALOG STATISTICS")
    log.info("="*80)
    log.info("")
    log.info("PARAMETERS:")
    log.info(f"  Link mode: {params.link_mode}")
    log.info(f"  dmax: {params.dmax_arcmin:.2f}' ({params.dmax_arcmin/60:.4f}°)")
    log.info(f"  mfac (small-small): {params.mfac:.2f}")
    log.info(f"  mfac_backbone (large-large): {params.mfac_backbone:.2f}")
    log.info(f"  mfac_sat (large-small): {params.mfac_sat:.2f}")
    log.info(f"  big_diam threshold: {params.big_diam:.2f}'")
    log.info(f"  k_floor: {params.k_floor:.2f}")
    log.info(f"  q_floor: {params.q_floor:.2f}")
    log.info(f"  containment: {'enabled' if params.contain else 'disabled'}")
    if params.contain:
        log.info(f"  contain_margin: {params.contain_margin*100:.0f}%")
    log.info("")
    log.info("ALGORITHM PERFORMANCE:")
    log.info(f"  Total objects: {len(cat):,d}")
    log.info(f"  Preclusters: {n_preclusters:,d}")
    log.info(f"  Links formed: {links:,d}")
    log.info(f"  Final groups: {n_groups:,d}")
    log.info(f"  Timing: precluster={timing['precluster']:.2f}s, "
             f"linking={timing['linking']:.2f}s, "
             f"aggregate={timing['aggregate']:.2f}s")
    log.info("")
    log.info("MULTIPLICITY DISTRIBUTION (objects):")
    log.info(f"  Singles:     {n_singles:6,d} objects in {n_groups_singles:6,d} groups")
    log.info(f"  Pairs:       {n_pairs:6,d} objects in {n_groups_pairs:6,d} groups")
    log.info(f"  3-5 members: {n_3to5:6,d} objects in {n_groups_3to5:6,d} groups")
    log.info(f"  6-10:        {n_6to10:6,d} objects in {n_groups_6to10:6,d} groups")
    log.info(f"  11-20:       {n_11to20:6,d} objects in {n_groups_11to20:6,d} groups")
    log.info(f"  21+:         {n_21plus:6,d} objects in {n_groups_21plus:6,d} groups")
    log.info("")
    log.info("GROUP SIZE DISTRIBUTION:")
    for thr in large_thresholds:
        pct = 100.0 * large_counts[thr] / n_groups if n_groups > 0 else 0.0
        log.info(f"  Diameter ≥ {thr:4.0f}' ({thr/60:5.3f}°): {large_counts[thr]:4d} groups ({pct:5.2f}%)")
    log.info("")
    if len(largest_groups) > 0:
        log.info(f"LARGEST {min(top_n, len(largest_groups))} GROUPS:")
        for rank, (gid, gname, mult, diam) in enumerate(largest_groups, 1):
            log.info(f"  {rank:2d}. {gname:10s} (ID={gid:6d}): "
                     f"mult={mult:3d}, diameter={diam:6.2f}' ({diam/60:5.3f}°)")
        log.info("")
    # Warning for very large groups
    critical_threshold = 40.0
    n_critical = large_counts.get(critical_threshold, 0)
    if n_critical > 2:
        log.warning(f"⚠️  {n_critical} groups exceed {critical_threshold:.0f}' (>{critical_threshold/60:.2f}°) - "
                   f"may create problematic mosaics!")
        log.warning(f"   Consider adjusting parameters or using post-processing split")
    log.info("="*80)


# ============================================================================
# Main function
# ============================================================================

def build_group_catalog(
    cat, group_id_start=0, mfac=1.6, dmin=36.0/3600.0, dmax=5.0/60.0,
    anisotropic=True, link_mode="hybrid", big_diam=3.0,
    mfac_backbone=1.3, mfac_sat=1.5, k_floor=0.40, q_floor=0.20,
    name_via="radec", sphere_link_arcmin=None, grid_cell_arcmin=None,
    mp=1, contain=True, contain_margin=0.75, merge_centers=True,
    merge_sep_arcsec=52.0, contain_search_arcmin=5.0,
    min_group_diam_arcsec=30.0, manual_merge_pairs=None, name_column="OBJNAME"):
    """
    Build galaxy group catalog with hybrid anisotropic linking.

    Parameters
    ----------
    cat : Table
        Input catalog (must have RA, DEC, DIAM columns; BA, PA optional)
    group_id_start : int
        Starting group ID number
    mfac : float
        Link multiplier for small-small pairs
    dmin, dmax : float
        Min/max link distances (degrees)
    mp : int
        Number of parallel processes
    manual_merge_pairs : List[Tuple[str, str]]
        Pairs of object names to manually merge

    Returns
    -------
    Table
        Input catalog with added GROUP_* columns

    """
    t0 = time.time()

    # Validate inputs
    required = {'RA', 'DEC', 'DIAM'}
    if not required <= set(cat.colnames):
        raise ValueError(f"Catalog must have columns {required}")

    if link_mode not in ('off', 'anisotropic', 'hybrid', None):
        raise ValueError(f"Invalid link_mode: {link_mode}")

    # Set up parameters
    params = Params(
        anisotropic=anisotropic,
        link_mode=link_mode,
        mfac=mfac,
        big_diam=big_diam,
        mfac_backbone=mfac_backbone,
        mfac_sat=mfac_sat,
        k_floor=k_floor,
        q_floor=q_floor,
        dmax_arcmin=dmax * ARCMIN_PER_DEG,
        dmin_arcmin=dmin * ARCMIN_PER_DEG,
        cell_arcmin=grid_cell_arcmin if grid_cell_arcmin else dmax * ARCMIN_PER_DEG,
        contain=contain,
        contain_margin=contain_margin,
    )

    # Initialize state
    state = State(cat, params)

    # Step 1: Spherical preclustering
    ll_arcmin = sphere_link_arcmin if sphere_link_arcmin else dmax * ARCMIN_PER_DEG
    ll_deg = ll_arcmin / ARCMIN_PER_DEG

    grp, mult_pre, frst, nxt = _spheregroup(state.ra, state.dec, ll_deg)
    n_pre = int(grp.max()) + 1 if len(grp) else 0

    t1 = time.time()
    log.info(f"[1/3] Precluster: {n_pre} preclusters, link={ll_arcmin:.2f}' in {t1-t0:.2f}s")

    # Extract multi-member clusters
    clusters = []
    for g in np.unique(grp):
        members = []
        i = frst[g]
        while i != -1:
            members.append(int(i))
            i = nxt[i]
        if len(members) >= 2:
            clusters.append(members)

    # Step 2: Find linkages within preclusters
    dsu = DSU(state.n)
    links = 0

    if not mp or mp <= 1:
        _init_worker(state)
        for members in clusters:
            edges = process_cluster(members, state, contain_search_arcmin)
            for a, b in edges:
                dsu.union(a, b)
            links += len(edges)
    else:
        with ProcessPoolExecutor(max_workers=mp, initializer=_init_worker, initargs=(state,)) as ex:
            args = [(m, contain_search_arcmin) for m in clusters]
            for edges in ex.map(_worker_wrapper, args):
                for a, b in edges:
                    dsu.union(a, b)
                links += len(edges)

    t2 = time.time()
    log.info(f"[2/3] Linking: {links} links in {t2-t1:.2f}s")

    # Step 3: Merge centers
    if merge_centers:
        roots = np.array([dsu.find(i) for i in range(state.n)])
        uniq = np.unique(roots)
        reps = {r: np.where(roots==r)[0][0] for r in uniq}
        ra_c, dec_c = np.zeros(len(uniq)), np.zeros(len(uniq))

        for gi, r in enumerate(uniq):
            idx = np.where(roots==r)[0]
            w = np.clip(state.diam[idx], 1e-3, None)
            ra_rad, dec_rad = state.ra[idx]*DEG2RAD, state.dec[idx]*DEG2RAD
            cosd = np.cos(dec_rad)
            x = (cosd*np.cos(ra_rad)*w).sum()/w.sum()
            y = (cosd*np.sin(ra_rad)*w).sum()/w.sum()
            z = (np.sin(dec_rad)*w).sum()/w.sum()
            norm = math.sqrt(x*x+y*y+z*z)
            if norm > 0:
                x, y, z = x/norm, y/norm, z/norm
            ra_c[gi] = (math.atan2(y,x)%(2*math.pi))*RAD2DEG
            dec_c[gi] = math.atan2(z,math.hypot(x,y))*RAD2DEG

        gcent, _, fcent, nxcent = _spheregroup(ra_c, dec_c, merge_sep_arcsec/3600.0)
        for gc in np.unique(gcent):
            i = fcent[gc]
            if i == -1:
                continue
            j = nxcent[i]
            while j != -1:
                dsu.union(reps[uniq[i]], reps[uniq[j]])
                j = nxcent[j]

    # Step 4: Manual merges
    n_manual = 0
    if manual_merge_pairs:
        if name_column not in cat.colnames:
            log.warning(f"Manual merge requested but column '{name_column}' not found, skipping")
        else:
            name_map = {str(n): i for i, n in enumerate(cat[name_column])}
            missing = []
            for a, b in manual_merge_pairs:
                ia, ib = name_map.get(str(a)), name_map.get(str(b))
                if ia is None or ib is None:
                    missing.append((a, b))
                    continue
                if dsu.union(ia, ib):
                    n_manual += 1

            if missing:
                log.warning(f"Manual merge: {len(missing)} pair(s) had missing names: {missing[:3]}")
            if n_manual > 0:
                log.info(f"Manual merge: {n_manual} pair(s) successfully merged")

    # Step 5: Merge groups with overlapping ellipses
    if contain:
        t_overlap_start = time.time()
        n_overlap_merges = _merge_overlapping_groups(dsu, state, contain_margin, mp)
        t_overlap_end = time.time()
        if n_overlap_merges > 0:
            log.info(f"Overlap merge: {n_overlap_merges} group pair(s) merged in {t_overlap_end-t_overlap_start:.2f}s")

    # Step 6: Compute final groups
    roots = np.array([dsu.find(i) for i in range(state.n)])
    uniq, inv = np.unique(roots, return_inverse=True)
    group_ids = (inv + group_id_start).astype(np.int32)
    mult = np.bincount(inv, minlength=len(uniq))

    grp_ra, grp_dec, grp_diam = np.zeros(len(uniq)), np.zeros(len(uniq)), np.zeros(len(uniq))
    grp_primary = np.zeros(len(uniq), dtype=np.int64)

    for gi, r in enumerate(uniq):
        idx = np.where(roots==r)[0]
        w = np.clip(state.diam[idx], 1e-3, None)
        ra_rad, dec_rad = state.ra[idx]*DEG2RAD, state.dec[idx]*DEG2RAD
        cosd = np.cos(dec_rad)
        x = (cosd*np.cos(ra_rad)*w).sum()/w.sum()
        y = (cosd*np.sin(ra_rad)*w).sum()/w.sum()
        z = (np.sin(dec_rad)*w).sum()/w.sum()
        norm = math.sqrt(x*x+y*y+z*z)
        if norm > 0:
            x, y, z = x/norm, y/norm, z/norm
        ra_c = (math.atan2(y,x)%(2*math.pi))*RAD2DEG
        dec_c = math.atan2(z,math.hypot(x,y))*RAD2DEG
        grp_ra[gi], grp_dec[gi] = ra_c, dec_c

        max_ext = 0.0
        for k in idx:
            dra = angdiff_deg(state.ra[k], ra_c) * math.cos(dec_c*DEG2RAD)
            sep = math.hypot(dra, state.dec[k]-dec_c) * ARCMIN_PER_DEG
            max_ext = max(max_ext, sep + 0.5*state.diam[k])
        grp_diam[gi] = max(2.0*max_ext, min_group_diam_arcsec/60.0)

        # Primary is the member with the largest DIAM
        grp_primary[gi] = idx[np.argmax(state.diam[idx])]

    # Annotate catalog
    row_primary = np.zeros(state.n, dtype=bool)
    for gi in range(len(uniq)):
        row_primary[grp_primary[gi]] = True

    # Generate group names
    if name_via == 'radec':
        names = [radec_to_groupname(grp_ra[inv[i]], grp_dec[inv[i]]) for i in range(state.n)]
    else:
        names = [''] * state.n

    def add(name, data, dtype=None):
        """Add column with proper dtype and ensure 1D shape."""
        if dtype is not None:
            arr = np.asarray(data, dtype=dtype)
        else:
            arr = np.asarray(data)

        # Ensure 1D (squeeze out any extra dimensions)
        if arr.ndim > 1:
            arr = arr.squeeze()

        col = Column(arr, name=name)

        if name in cat.colnames:
            cat.replace_column(name, col)
        else:
            cat.add_column(col)

    add('GROUP_ID', group_ids, np.int32)
    add('GROUP_NAME', names, 'U10')
    add('GROUP_MULT', mult[inv].astype(np.int16))
    add('GROUP_PRIMARY', row_primary, bool)
    add('GROUP_RA', grp_ra[inv], np.float64)
    add('GROUP_DEC', grp_dec[inv], np.float64)
    add('GROUP_DIAMETER', grp_diam[inv].astype(np.float32))

    t3 = time.time()
    log.info(f"[3/3] Annotate: {t3-t2:.2f}s")

    # Enhanced statistics reporting
    timing = {
        'precluster': t1 - t0,
        'linking': t2 - t1,
        'aggregate': t3 - t2
    }
    report_group_statistics(cat, params, links, n_pre, timing)

    return cat


# ============================================================================
# Utility functions
# ============================================================================

def make_singleton_group(cat, group_id_start=0):
    """
    Create singleton groups (one object per group).

    Ensures data model matches build_group_catalog output for safe vstacking.

    """
    n = len(cat)

    # Generate group names
    names = [radec_to_groupname(cat['RA'][k], cat['DEC'][k]) for k in range(n)]

    # Helper function to add/replace columns with proper dtype and shape
    def add_column(name, data, dtype=None):
        """Add column ensuring 1D shape."""
        if dtype is not None:
            arr = np.asarray(data, dtype=dtype)
        else:
            arr = np.asarray(data)

        # Ensure 1D (squeeze out any extra dimensions)
        if arr.ndim > 1:
            arr = arr.squeeze()

        col = Column(arr, name=name)

        if name in cat.colnames:
            cat.replace_column(name, col)
        else:
            cat.add_column(col)

    # Add GROUP_* columns with exact same dtypes as build_group_catalog
    add_column('GROUP_ID', np.arange(group_id_start, group_id_start + n), np.int32)
    add_column('GROUP_NAME', names, 'U10')
    add_column('GROUP_MULT', np.ones(n), np.int16)
    add_column('GROUP_PRIMARY', np.ones(n), bool)
    add_column('GROUP_RA', cat['RA'], np.float64)
    add_column('GROUP_DEC', cat['DEC'], np.float64)
    add_column('GROUP_DIAMETER', cat['DIAM'], np.float32)

    return cat


def qa(*args, **kwargs):
    """Placeholder QA hook (no-op).

    """
    log.info("qa(): no-op placeholder")


def set_overlap_bit(cat, SAMPLE):
    """
    Flag ellipse-overlap within each group by setting SAMPLE['OVERLAP'].

    Assumes `cat` has columns: GROUP_NAME, GROUP_MULT, RA, DEC, DIAM (arcmin),
    BA, PA (deg, astronomical).

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

    # Work only on groups with >1 member
    mask_mult = (cat['GROUP_MULT'] > 1)
    if not np.any(mask_mult):
        return

    # Preload all columns once (avoid repeated table lookups)
    RA_all = np.asarray(cat['RA'], dtype=float)
    DEC_all = np.asarray(cat['DEC'], dtype=float)
    DIAM_all = np.asarray(cat['DIAM'], dtype=float)
    BA_all = np.asarray(cat['BA'], dtype=float) if 'BA' in cat.colnames else np.full(len(cat), np.nan)
    PA_all = np.asarray(cat['PA'], dtype=float) if 'PA' in cat.colnames else np.full(len(cat), np.nan)
    GNAME = np.asarray(cat['GROUP_NAME']).astype(str)

    # Unique group names among multi-member groups
    ugroups = np.unique(GNAME[mask_mult])

    for gname in ugroups:
        # Get indices for this group
        I = np.where(GNAME == gname)[0]
        if I.size < 2:
            continue

        # Local tangent-plane coordinate system
        dec0 = float(np.median(DEC_all[I]))
        cosd0 = math.cos(dec0 * DEG2RAD)

        # Extract member properties
        ra = RA_all[I]
        dec = DEC_all[I]
        diam = DIAM_all[I]
        a_arc = 0.5 * diam

        # Handle missing BA/PA gracefully
        ba = BA_all[I]
        pa = PA_all[I]
        ba_eff = np.where(np.isfinite(ba) & (ba > 0.0), ba, 1.0)
        b_arc = ba_eff * a_arc
        pa_rad = np.where(np.isfinite(pa), pa, 0.0) * DEG2RAD

        overlapped = np.zeros(I.size, dtype=bool)

        # Pairwise overlap checks (upper triangle only)
        for ii in range(I.size - 1):
            # Compute offsets to all subsequent members
            dx_deg = angdiff_deg(ra[ii+1:], ra[ii]) * cosd0
            dy_deg = dec[ii+1:] - dec[ii]

            # Check ellipse overlap using the same function as linking
            for kk, (dx, dy) in enumerate(zip(dx_deg, dy_deg)):
                jj = ii + 1 + kk
                if ellipses_overlap(a_arc[ii], b_arc[ii], pa_rad[ii],
                                   a_arc[jj], b_arc[jj], pa_rad[jj],
                                   dx, dy, scale=1.0):  # No margin for overlap bit
                    overlapped[ii] = True
                    overlapped[jj] = True

        # Set the OVERLAP bit for overlapping members
        if np.any(overlapped):
            cat['SAMPLE'][I[overlapped]] |= OVERLAP_BIT

