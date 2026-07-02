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
    """Wrap angle(s) into [0, 360) degrees.

    Parameters
    ----------
    x : :class:`float` or array-like
        Angle(s), degrees.

    Returns
    -------
    :class:`float` or :class:`numpy.ndarray`
        Wrapped angle(s), in [0, 360).

    """
    return np.mod(x, 360.0)


def angdiff_deg(a, b):
    """Signed angular difference ``a - b``, wrapped into (-180, 180]
    degrees.

    Parameters
    ----------
    a, b : :class:`float` or array-like
        Angles, degrees.

    Returns
    -------
    :class:`float` or :class:`numpy.ndarray`
        ``a - b``, wrapped into (-180, 180].

    """
    return (a - b + 180.0) % 360.0 - 180.0


def directional_radius(a_arc, b_arc, pa_rad, bearing_rad):
    """Compute an ellipse's radius (arcmin) along a given bearing
    direction from its center.

    Works with both scalars and arrays (fully vectorized).

    Parameters
    ----------
    a_arc, b_arc : :class:`float` or array-like
        Semi-major and semi-minor axes, arcmin.
    pa_rad : :class:`float` or array-like
        Position angle, radians, astronomical convention.
    bearing_rad : :class:`float` or array-like
        Bearing direction, radians, astronomical convention.

    Returns
    -------
    :class:`float` or :class:`numpy.ndarray`
        Ellipse radius, arcmin, along the specified bearing; falls back
        to ``a_arc`` for degenerate (``a_arc <= 0`` or ``b_arc <= 0``)
        input.

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
    """Check whether point ``(dx_deg, dy_deg)`` lies inside a scaled
    ellipse centered at the origin.

    This checks if a POINT is inside an ellipse, not ellipse-ellipse
    overlap. For ellipse overlap, use :func:`ellipses_overlap` (this
    module's version, taking tangent-plane offsets and arcmin axes --
    distinct from :func:`SGA.geometry.ellipses_overlap`, which operates
    in pixel coordinates).

    Notes
    -----
    This function is currently unreachable in the group-finding
    pipeline: its only caller, :meth:`State.contains`, is itself never
    called anywhere in this module (:func:`should_link` uses
    :meth:`State.overlaps` exclusively).

    Parameters
    ----------
    a_arc, b_arc : :class:`float`
        Semi-major and semi-minor axes, arcmin.
    pa_rad : :class:`float`
        Position angle, radians, astronomical convention.
    dx_deg, dy_deg : :class:`float`
        Point offset in the local tangent plane, degrees.
    scale : :class:`float`
        Scale factor for the ellipse (>1 expands).

    Returns
    -------
    :class:`bool`
        True if the point is inside the scaled ellipse.

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
    """Check whether two scaled ellipses overlap, via directional radii
    along the center-center line.

    The proper test for determining if two galaxies are close enough to
    be considered overlapping/interacting: computes each ellipse's
    radius (:func:`directional_radius`) toward the other's center, and
    checks whether the center separation is within the sum of those
    (scaled) directional radii. Distinct from
    :func:`SGA.geometry.ellipses_overlap` (pixel-coordinate,
    boundary-sampling version) despite the shared name.

    Parameters
    ----------
    a1, b1 : :class:`float`
        Semi-major and semi-minor axes of ellipse 1, arcmin.
    pa1_rad : :class:`float`
        Position angle of ellipse 1, radians, astronomical convention.
    a2, b2 : :class:`float`
        Semi-major and semi-minor axes of ellipse 2, arcmin.
    pa2_rad : :class:`float`
        Position angle of ellipse 2, radians, astronomical convention.
    dx_deg, dy_deg : :class:`float`
        Offset from ellipse 1's center to ellipse 2's center, degrees.
    scale : :class:`float`
        Scale factor applied to both ellipses (e.g. 1.5 = 50% margin).

    Returns
    -------
    :class:`bool`
        True if the scaled ellipses overlap.

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
    """Disjoint Set Union (Union-Find) with path compression and union
    by rank, used to accumulate galaxy-group membership as pairwise
    links are discovered.

    Attributes
    ----------
    p : :class:`list` of :class:`int`
        Parent pointer per element (index = element).
    r : :class:`list` of :class:`int`
        Union-by-rank tree-depth heuristic per element.

    """

    def __init__(self, n):
        """Initialize ``n`` singleton sets, each its own root.

        Parameters
        ----------
        n : :class:`int`
            Number of elements.

        """
        self.p = list(range(n))
        self.r = [0] * n

    def find(self, x):
        """Find the root (representative) of ``x``'s set, with path
        compression.

        Parameters
        ----------
        x : :class:`int`
            Element index.

        Returns
        -------
        :class:`int`
            Root index of ``x``'s set.

        """
        if self.p[x] != x:
            self.p[x] = self.find(self.p[x])
        return self.p[x]

    def union(self, a, b):
        """Merge the sets containing ``a`` and ``b``, by rank.

        Parameters
        ----------
        a, b : :class:`int`
            Element indices.

        Returns
        -------
        :class:`bool`
            True if a merge occurred; False if ``a`` and ``b`` were
            already in the same set.

        """
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
    """Group-finding algorithm parameters, assembled once per
    :func:`build_group_catalog` call and threaded through
    :class:`State`.

    Attributes
    ----------
    anisotropic : :class:`bool`
        If False, always use isotropic (circular) link thresholds
        regardless of ``link_mode``.
    link_mode : :class:`str`
        ``'off'``/None (isotropic only), ``'anisotropic'`` (always use
        directional radii), or ``'hybrid'`` (directional radii for
        small-small pairs, size-dependent treatment for pairs involving
        a "big" object -- see :meth:`State.pair_threshold`).
    mfac : :class:`float`
        Link multiplier for small-small pairs.
    big_diam : :class:`float`
        Diameter threshold (arcmin) above which an object is treated as
        "big" in hybrid mode.
    mfac_backbone : :class:`float`
        Link multiplier for big-big pairs (hybrid mode).
    mfac_sat : :class:`float`
        Link multiplier for big-small pairs (hybrid mode).
    k_floor : :class:`float`
        Minimum fraction of an object's circular radius used as a floor
        on its directional radius in big-small pairs (hybrid mode).
    q_floor : :class:`float`
        Minimum effective axis ratio (b/a) floor applied when computing
        anisotropic radii, to avoid pathologically thin ellipses.
    dmax_arcmin : :class:`float`
        Maximum center-to-center separation, arcmin, considered for
        linking (beyond the containment pre-check).
    dmin_arcmin : :class:`float`
        Minimum link threshold, arcmin, applied as a floor regardless of
        the computed pair threshold.
    cell_arcmin : :class:`float`
        Spatial grid cell size, arcmin, for :class:`Grid` neighbor
        search within a precluster.
    contain : :class:`bool`
        If True, also link (and later merge) pairs whose scaled
        ellipses overlap, independent of the threshold/``dmax`` checks.
    contain_margin : :class:`float`
        Fractional margin applied when scaling ellipses for the
        containment/overlap checks (e.g. 0.75 = 75% expansion).

    """
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
    """Encapsulate all per-object data and derived ellipse geometry
    needed for group finding, replacing what used to be module-level
    globals (so the algorithm is safe to run in multiprocessing
    workers).

    Attributes
    ----------
    n : :class:`int`
        Number of objects.
    ra, dec : :class:`numpy.ndarray`
        Object positions, degrees.
    diam : :class:`numpy.ndarray`
        Object diameters, arcmin.
    ba, pa : :class:`numpy.ndarray`
        Object axis ratio and position angle (degrees); ``numpy.nan``
        where the input catalog lacks ``BA``/``PA`` columns.
    a_arc, b_arc : :class:`numpy.ndarray`
        Semi-major/semi-minor axes, arcmin (``b_arc`` uses ``ba``,
        floored at ``params.q_floor`` if positive, defaulting to
        circular (``ba=1``) where ``ba`` is missing/non-positive).
    pa_rad : :class:`numpy.ndarray`
        Position angle, radians (0 where ``pa`` is missing).
    has_aniso : :class:`numpy.ndarray` of :class:`bool`
        True for objects with finite, valid ``ba``/``pa`` (i.e. genuine
        anisotropic geometry available, vs. the circular default).
    params : :class:`Params`
        Algorithm parameters.

    """

    def __init__(self, cat, params):
        """Derive per-object ellipse geometry from a catalog.

        Parameters
        ----------
        cat : :class:`~astropy.table.Table`
            Catalog with ``RA``, ``DEC``, ``DIAM`` (required) and
            ``BA``, ``PA`` (optional) columns.
        params : :class:`Params`
            Algorithm parameters.

        """
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
        """Read a column from ``cat`` as a float array, or a
        constant-filled fallback array if the column is absent.

        Parameters
        ----------
        cat : :class:`~astropy.table.Table`
            Catalog to read from.
        name : :class:`str`
            Column name.
        default : :class:`float`
            Fill value used (for every row) if ``name`` is not a column
            in ``cat``.

        Returns
        -------
        :class:`numpy.ndarray`
            Length ``len(cat)`` float array.

        """
        return np.asarray(cat[name], dtype=float) if name in cat.colnames else np.full(len(cat), default, dtype=float)

    def pair_threshold(self, i, j, dx_deg, dy_deg):
        """Compute the link (pairing) threshold, in arcmin, for object
        pair ``(i, j)``, per the ``link_mode``/anisotropy rules in
        :attr:`params`.

        Isotropic (``not params.anisotropic`` or ``link_mode in ('off',
        None)``): half the mean of the two circular radii, scaled by
        ``mfac``. Anisotropic (``link_mode == 'anisotropic'``): same,
        but using each object's directional radius toward the other
        (:meth:`_dir_radius`) instead of its circular radius. Hybrid
        (``link_mode == 'hybrid'``, the default): big-big pairs (both
        diameters >= ``params.big_diam``) use circular radii scaled by
        ``mfac_backbone``; big-small pairs use directional radii
        (floored at ``k_floor`` times the circular radius) scaled by
        ``mfac_sat``; small-small pairs use directional radii scaled by
        ``mfac``.

        Parameters
        ----------
        i, j : :class:`int`
            Indices of the two objects.
        dx_deg, dy_deg : :class:`float`
            Tangent-plane offset from object ``i`` to object ``j``,
            degrees.

        Returns
        -------
        :class:`float`
            Link threshold, arcmin (not yet floored at
            ``params.dmin_arcmin`` -- see :func:`should_link`).

        """
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
        """Object ``i``'s radius toward a given bearing, or its
        circular radius if it lacks valid anisotropic geometry.

        Parameters
        ----------
        i : :class:`int`
            Object index.
        dx_deg, dy_deg : :class:`float`
            Offset defining the bearing direction, degrees.

        Returns
        -------
        :class:`float`
            Directional (or circular, if ``not has_aniso[i]``) radius,
            arcmin.

        """
        if not self.has_aniso[i]:
            return self.a_arc[i]
        bearing = math.atan2(dx_deg, dy_deg)
        return directional_radius(self.a_arc[i], self.b_arc[i], self.pa_rad[i], bearing)

    def contains(self, i, j, dx_deg, dy_deg, scale):
        """Check if object ``j``'s center (given by the offset from
        ``i``) is inside object ``i``'s scaled ellipse.

        Notes
        -----
        Never called anywhere in this module -- :func:`should_link`
        uses :meth:`overlaps` exclusively, so this method (and
        :func:`contains_point`, which it wraps) is dead code in the
        current pipeline. ``j`` is also accepted but unused in the
        body; only ``dx_deg``/``dy_deg`` (the precomputed offset) and
        ``i``'s geometry are used.

        Parameters
        ----------
        i, j : :class:`int`
            Indices of the two objects.
        dx_deg, dy_deg : :class:`float`
            Offset from object ``i`` to object ``j``, degrees.
        scale : :class:`float`
            Scale factor applied to ``i``'s ellipse.

        Returns
        -------
        :class:`bool`
            True if ``j``'s center is inside ``i``'s scaled ellipse.

        """
        return contains_point(self.a_arc[i], self.b_arc[i], self.pa_rad[i], dx_deg, dy_deg, scale)

    def overlaps(self, i, j, dx_deg, dy_deg, scale):
        """Check if objects ``i`` and ``j``'s scaled ellipses overlap.

        Parameters
        ----------
        i, j : :class:`int`
            Indices of the two objects.
        dx_deg, dy_deg : :class:`float`
            Offset from object ``i`` to object ``j``, degrees.
        scale : :class:`float`
            Scale factor applied to both ellipses.

        Returns
        -------
        :class:`bool`
            True if the scaled ellipses overlap.

        """
        return ellipses_overlap(
            self.a_arc[i], self.b_arc[i], self.pa_rad[i],
            self.a_arc[j], self.b_arc[j], self.pa_rad[j],
            dx_deg, dy_deg, scale
        )


# ============================================================================
# Grid search
# ============================================================================

class Grid:
    """Uniform spatial hash grid over tangent-plane offsets, for fast
    candidate-neighbor lookup within a precluster (see
    :func:`process_cluster`).

    Attributes
    ----------
    gx, gy : :class:`numpy.ndarray`
        Integer grid-cell coordinates of each point.
    bins : :class:`dict`
        ``(gx, gy)`` cell -> list of point indices in that cell.

    """

    def __init__(self, dx, dy, cell_deg):
        """Bin points into a uniform grid.

        Parameters
        ----------
        dx, dy : :class:`numpy.ndarray`
            Tangent-plane offsets of each point, degrees.
        cell_deg : :class:`float`
            Grid cell size, degrees.

        """
        self.gx = np.floor(dx / cell_deg).astype(np.int64)
        self.gy = np.floor(dy / cell_deg).astype(np.int64)
        self.bins = {}
        for k in range(len(dx)):
            key = (int(self.gx[k]), int(self.gy[k]))
            self.bins.setdefault(key, []).append(k)

    def neighbors(self, k):
        """Return the indices of all points in point ``k``'s cell and
        its 8 adjacent cells (3x3 block).

        Parameters
        ----------
        k : :class:`int`
            Point index.

        Returns
        -------
        :class:`list` of :class:`int`
            Candidate neighbor indices (includes ``k`` itself).

        """
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
    """Find pairwise links within one spherical precluster (see
    :func:`build_group_catalog`'s Step 1/2), via a local tangent-plane
    projection and grid-based neighbor search.

    Projects ``members`` onto a tangent plane centered at their median
    position, bins them into a :class:`Grid` (cell size
    ``state.params.cell_arcmin``), and for each candidate neighbor pair
    (from adjacent grid cells, each pair considered once) tests
    :func:`should_link`. Ellipses are scaled by
    ``1 + state.params.contain_margin`` for the containment checks
    inside :func:`should_link`.

    Parameters
    ----------
    members : :class:`list` of :class:`int`
        Indices (into ``state``'s arrays) of objects in this
        precluster. If fewer than 2, returns immediately with no edges.
    state : :class:`State`
        Group-finding state.
    contain_search_arcmin : :class:`float`
        Separation, arcmin, below which the containment/overlap check
        is tried even for pairs that would otherwise fail the standard
        threshold (passed through to :func:`should_link`).

    Returns
    -------
    :class:`numpy.ndarray`
        Shape ``(nlinks, 2)`` array of ``(i, j)`` index pairs (into
        ``state``'s arrays) to be unioned in the caller's :class:`DSU`;
        shape ``(0, 2)`` if none.

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
    scale = 1.0 + p.contain_margin  # Use scaled ellipses for initial linking

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
    """Decide whether object pair ``(i, j)`` should be linked into the
    same group, combining a close-pair ellipse-overlap check, a
    maximum-separation cutoff, and a size-dependent threshold.

    Checked in order: (1) if ``state.params.contain`` and
    ``sep <= search_arcmin``, link if the (scaled) ellipses overlap
    (:meth:`State.overlaps`); (2) reject outright if
    ``sep > state.params.dmax_arcmin``; (3) if ``state.params.contain``,
    link if the (scaled) ellipses overlap, regardless of distance within
    ``dmax``; (4) otherwise link if ``sep`` is within the size-dependent
    pair threshold (:meth:`State.pair_threshold`, floored at
    ``state.params.dmin_arcmin``).

    Parameters
    ----------
    i, j : :class:`int`
        Indices of the two objects.
    ddx, ddy : :class:`float`
        Tangent-plane offset from object ``i`` to object ``j``, degrees.
    sep : :class:`float`
        Center-to-center separation, arcmin (``hypot(ddx, ddy)`` in
        arcmin).
    state : :class:`State`
        Group-finding state.
    scale : :class:`float`
        Ellipse scale factor for the overlap checks.
    search_arcmin : :class:`float`
        Separation below which the close-pair overlap pre-check (step 1
        above) is tried.

    Returns
    -------
    :class:`bool`
        True if the pair should be linked.

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
    """``ProcessPoolExecutor`` initializer: stash ``state`` in a
    process-global so :func:`_worker_wrapper` can access it without
    repickling/copying it on every task.

    Parameters
    ----------
    state : :class:`State`
        Group-finding state, set as the module-global ``_WORKER_STATE``
        in this worker process.

    Returns
    -------
    None

    """
    global _WORKER_STATE
    _WORKER_STATE = state

def _check_group_pair_overlap(args):
    """Check whether any member of provisional group ``i`` overlaps any
    member of provisional group ``j``, for
    :func:`_merge_overlapping_groups`.

    A cheap group-center distance check first rules out pairs whose
    bounding radii plus ``max_search_radius`` can't possibly overlap;
    otherwise, for each member of group ``i``, vectorizes the offset and
    separation to every member of group ``j``, cheaply filters to
    candidates within the sum of circular radii, and only then runs the
    exact :func:`ellipses_overlap` test on those candidates. Returns as
    soon as any overlapping member pair is found.

    Parameters
    ----------
    args : :class:`tuple`
        ``(root_i, root_j, pos_i, pos_j, state, scale, max_search_radius)``,
        where ``root_i``/``root_j`` are the two groups' DSU root indices
        (accepted but unused in this function's body -- the caller uses
        them to interpret the return value, not this function), ``pos_i``/
        ``pos_j`` are dicts with keys ``'ra'``, ``'dec'``, ``'radius'``
        (degrees) and ``'members'`` (index array), ``state`` is the
        :class:`State`, ``scale`` is the ellipse scale factor, and
        ``max_search_radius`` is the group-center pre-check margin,
        arcmin.

    Returns
    -------
    :class:`tuple` of :class:`int` or None
        ``(idx_i, idx_j)`` -- the specific overlapping member indices --
        if an overlap is found, else None.

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

    idx_i = pos_i['members']
    idx_j = pos_j['members']

    # Vectorized setup for group i
    dec0_i = state.dec[idx_i]
    cosd0_i = np.cos(dec0_i * DEG2RAD)
    ra0_i = state.ra[idx_i]

    # For each member of group i, compute offsets to ALL members of group j
    for ki, i_idx in enumerate(idx_i):
        # Vectorized offset calculation to all j members
        dx_deg = angdiff_deg(state.ra[idx_j], ra0_i[ki]) * cosd0_i[ki]
        dy_deg = state.dec[idx_j] - dec0_i[ki]

        # Vectorized separation
        sep = np.hypot(dx_deg, dy_deg) * ARCMIN_PER_DEG
        max_possible = (state.a_arc[i_idx] + state.a_arc[idx_j]) * scale

        # Quick rejection
        candidates = np.where(sep <= max_possible)[0]
        if len(candidates) == 0:
            continue

        # Check only promising candidates
        for kj in candidates:
            j_idx = idx_j[kj]
            if ellipses_overlap(
                state.a_arc[i_idx], state.b_arc[i_idx], state.pa_rad[i_idx],
                state.a_arc[j_idx], state.b_arc[j_idx], state.pa_rad[j_idx],
                dx_deg[kj], dy_deg[kj], scale):
                return (int(i_idx), int(j_idx))

    return None


def _merge_overlapping_groups(dsu, state, contain_margin, mp=1):
    """Merge provisional groups whose members' (unscaled) ellipses
    overlap, catching cases like IC 4278 inside NGC 5194 that initial
    linking (:func:`process_cluster`) missed because the pair exceeded
    ``dmax_arcmin``.

    Restricts the expensive pairwise check to "candidate" groups (max
    member diameter > 1' or >= 10 members), builds a spatial index of
    candidate-group centers, filters to plausibly-overlapping group
    pairs by center separation, then checks each surviving pair via
    :func:`_check_group_pair_overlap` (in parallel across ``mp``
    processes if requested) and unions any pair found to overlap.

    Notes
    -----
    ``contain_margin`` is accepted but never referenced in this
    function's body -- overlap checking here always uses unscaled
    ellipses (``scale = 1.0``, hardcoded), regardless of the value
    passed in.

    This step is expensive (O(N_large * N_small)) and may not be
    necessary if ``merge_centers`` (:func:`build_group_catalog`'s Step
    5) is enabled with an appropriate ``merge_sep_arcsec``; for most
    cases, ``merge_centers`` alone is sufficient to prevent duplicate
    ``GROUP_NAME``s.

    Parameters
    ----------
    dsu : :class:`DSU`
        Disjoint-set-union structure, modified in place.
    state : :class:`State`
        Group-finding state with galaxy properties.
    contain_margin : :class:`float`
        Unused (see Notes).
    mp : :class:`int`
        Number of parallel processes.

    Returns
    -------
    :class:`int`
        Number of group pairs merged.

    """
    # Get current provisional groups
    roots = np.array([dsu.find(i) for i in range(state.n)])
    unique_roots = np.unique(roots)

    # Only check groups that could plausibly overlap
    # (i.e., large groups or groups with many members)
    scale = 1.0  # Use unscaled ellipses for overlap checking
    max_search_radius = 10.0  # arcmin

    # Compute max diameter per group
    group_max_diam = np.zeros(unique_roots.max()+1)
    np.maximum.at(group_max_diam, roots, state.diam)

    # Filter to candidates: large diameter OR many members
    group_sizes = np.bincount(roots, minlength=unique_roots.max()+1)
    candidate_mask = (group_max_diam[unique_roots] > 1.0) | (group_sizes[unique_roots] >= 10)
    candidate_roots = unique_roots[candidate_mask]

    if len(candidate_roots) == 0:
        return 0

    n_candidates = len(candidate_roots)
    log.info(f"Overlap merge: {n_candidates:,d} candidate groups (max_diam>1' or mult>=10)")

    # Build spatial index for candidate groups
    group_ra = np.zeros(n_candidates)
    group_dec = np.zeros(n_candidates)
    group_radius = np.zeros(n_candidates)
    group_members = []

    for i, root in enumerate(candidate_roots):
        idx = np.where(roots == root)[0]
        group_members.append(idx)
        group_ra[i] = np.median(state.ra[idx])
        group_dec[i] = np.median(state.dec[idx])
        group_radius[i] = group_max_diam[root] * scale / ARCMIN_PER_DEG  # Uses scale=1.0

    # Cartesian coordinates for distance computation
    cosd = np.cos(group_dec * DEG2RAD)
    ra_rad = group_ra * DEG2RAD
    dec_rad = group_dec * DEG2RAD
    x = cosd * np.cos(ra_rad)
    y = cosd * np.sin(ra_rad)
    z = np.sin(dec_rad)

    # Build pair list with spatial filtering (vectorized)
    pair_args = []

    if n_candidates > 1:
        # Create index arrays for all pairs
        i_idx, j_idx = np.triu_indices(n_candidates, k=1)

        # Vectorized distance computation for all pairs
        dx = x[j_idx] - x[i_idx]
        dy = y[j_idx] - y[i_idx]
        dz = z[j_idx] - z[i_idx]
        chord_dist = np.sqrt(dx*dx + dy*dy + dz*dz)

        # Angular separation
        ang_sep = 2 * np.arcsin(np.clip(chord_dist / 2, -1, 1)) * RAD2DEG

        # Only check pairs that could plausibly overlap
        max_reach = group_radius[i_idx] + group_radius[j_idx] + max_search_radius / ARCMIN_PER_DEG
        close_enough = ang_sep <= max_reach

        # Build argument list for close pairs
        close_pairs = np.where(close_enough)[0]
        for pair_idx in close_pairs:
            i = i_idx[pair_idx]
            j = j_idx[pair_idx]
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

    log.info(f"  Spatial filtering: {total_pairs:,d} plausible pairs to check")

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
    """Unpack an argument tuple and call :func:`process_cluster` using
    the process-global ``_WORKER_STATE`` (set by :func:`_init_worker`);
    ``ProcessPoolExecutor`` map worker for :func:`build_group_catalog`'s
    Step 2.

    Parameters
    ----------
    args : :class:`tuple`
        ``(members, search)``, matching :func:`process_cluster`'s
        ``members``/``contain_search_arcmin`` parameters.

    Returns
    -------
    See :func:`process_cluster`.

    """
    members, search = args
    return process_cluster(members, _WORKER_STATE, search)


# ============================================================================
# Statistics reporting
# ============================================================================

def report_group_statistics(cat, params, links, n_preclusters, timing):
    """Log a detailed summary of :func:`build_group_catalog`'s results:
    parameters used, algorithm timing, object/group multiplicity
    distribution, group-size (diameter) distribution, and the largest
    groups found.

    Also logs a warning if more than 2 groups exceed a hardcoded 40'
    diameter threshold, since very large groups can produce
    problematic (oversized) mosaics downstream.

    Parameters
    ----------
    cat : :class:`~astropy.table.Table`
        Catalog with ``GROUP_NAME``, ``GROUP_MULT``, ``GROUP_DIAMETER``
        columns, as produced by :func:`build_group_catalog`.
    params : :class:`Params`
        Algorithm parameters used for this run, for the summary header.
    links : :class:`int`
        Total number of pairwise links formed during linking.
    n_preclusters : :class:`int`
        Number of spherical preclusters found in Step 1.
    timing : :class:`dict`
        Keys ``'precluster'``, ``'linking'``, ``'aggregate'`` -> elapsed
        seconds for each stage.

    Returns
    -------
    None

    """

    # Get unique groups
    group_names = cat['GROUP_NAME']
    unique_groups, first_occ, inv = np.unique(group_names, return_index=True, return_inverse=True)
    n_groups = len(unique_groups)

    # Multiplicity distribution
    mults = np.asarray(cat['GROUP_MULT'])
    diams = np.asarray(cat['GROUP_DIAMETER'])

    groups_mults = mults[first_occ]
    unique_diams = diams[first_occ]

    # Binned multiplicity statistics
    n_singles = int(np.sum(mults == 1))
    n_pairs = int(np.sum(mults == 2))
    n_3to5 = int(np.sum((mults >= 3) & (mults <= 5)))
    n_6to10 = int(np.sum((mults >= 6) & (mults <= 10)))
    n_11to20 = int(np.sum((mults >= 11) & (mults <= 20)))
    n_21plus = int(np.sum(mults > 20))

    # Group-level statistics (count groups not objects)
    n_groups_singles = int(np.sum(groups_mults == 1))
    n_groups_pairs = int(np.sum(groups_mults == 2))
    n_groups_3to5 = int(np.sum((groups_mults >= 3) & (groups_mults <= 5)))
    n_groups_6to10 = int(np.sum((groups_mults >= 6) & (groups_mults <= 10)))
    n_groups_11to20 = int(np.sum((groups_mults >= 11) & (groups_mults <= 20)))
    n_groups_21plus = int(np.sum(groups_mults > 20))

    # Diameter statistics

    # Large group thresholds (in arcmin)
    large_thresholds = [10.0, 20.0, 30.0, 40.0, 50.0]
    large_counts = {thr: int(np.sum(unique_diams >= thr)) for thr in large_thresholds}

    # Find largest groups
    top_n = 10
    sorted_idx = np.argsort(unique_diams)[::-1]
    largest_groups = []
    for i in range(min(top_n, len(sorted_idx))):
        gname = unique_groups[sorted_idx[i]]
        diam = unique_diams[sorted_idx[i]]
        mult = groups_mults[sorted_idx[i]]
        largest_groups.append((gname, mult, diam))

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
        for rank, (gname, mult, diam) in enumerate(largest_groups, 1):
            log.info(f"  {rank:2d}. {gname:10s}: mult={mult:3d}, diameter={diam:6.2f}' ({diam/60:5.3f}°)")
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
    cat, mfac=1.6, dmin=36.0/3600.0, dmax=5.0/60.0,
    anisotropic=True, link_mode="hybrid", big_diam=3.0,
    mfac_backbone=1.3, mfac_sat=1.5, k_floor=0.40, q_floor=0.20,
    name_via="radec", sphere_link_arcmin=None, grid_cell_arcmin=None,
    mp=1, contain=True, contain_margin=0.75, merge_centers=True,
    merge_sep_arcsec=52.0, contain_search_arcmin=30.0,
    min_group_diam_arcsec=30.0, manual_merge_pairs=None, name_column="OBJNAME"):
    """Build the galaxy group catalog via spherical preclustering
    followed by grid-based, hybrid anisotropic linking, per
    `CLAUDE.md`'s "Galaxy group finding via spherical clustering".

    Six steps: (1) coarse spherical preclustering via
    ``pydl.pydlutils.spheregroup.spheregroup`` (link length
    ``sphere_link_arcmin``, default ``dmax``) to cheaply bucket objects
    into candidate clusters; (2) within each multi-member precluster,
    fine-grained pairwise linking (:func:`process_cluster`/
    :func:`should_link`, optionally parallelized across ``mp``
    processes), accumulated into a :class:`DSU`; (3) apply any
    ``manual_merge_pairs`` by object name; (4) if ``contain``, merge
    provisional groups whose members' ellipses overlap but were missed
    by step 2 for exceeding ``dmax`` (:func:`_merge_overlapping_groups`);
    (5) if ``merge_centers``, merge any remaining provisional groups
    whose *centers* land within ``merge_sep_arcsec`` of each other (a
    second `spheregroup` pass on the group centers, catching residual
    duplicate/near-duplicate groups); (6) compute final group centers
    (diameter-weighted Cartesian mean), diameters (max member extent
    from center, floored at ``min_group_diam_arcsec``), and primary
    flag (largest-diameter member), then annotate ``cat`` with the
    ``GROUP_*`` columns and log a statistics summary
    (:func:`report_group_statistics`).

    Parameters
    ----------
    cat : :class:`~astropy.table.Table`
        Input catalog; must have ``RA``, ``DEC``, ``DIAM`` columns
        (``BA``, ``PA`` optional -- objects without them are treated as
        circular).
    mfac : :class:`float`
        Link multiplier for small-small pairs (see
        :meth:`State.pair_threshold`).
    dmin, dmax : :class:`float`
        Minimum/maximum link distance, degrees. ``dmax`` also sets the
        default spherical-preclustering link length and grid cell size
        if ``sphere_link_arcmin``/``grid_cell_arcmin`` aren't given.
    anisotropic : :class:`bool`
        Passed to :class:`Params`; if False, always use isotropic
        (circular) thresholds.
    link_mode : {'off', 'anisotropic', 'hybrid', None}
        Linking strategy; see :class:`Params` and
        :meth:`State.pair_threshold`.
    big_diam : :class:`float`
        Diameter threshold (arcmin) for "big" objects in hybrid mode.
    mfac_backbone : :class:`float`
        Link multiplier for big-big pairs (hybrid mode).
    mfac_sat : :class:`float`
        Link multiplier for big-small pairs (hybrid mode).
    k_floor : :class:`float`
        Minimum circular-radius fraction floor for big-small pairs
        (hybrid mode).
    q_floor : :class:`float`
        Minimum effective axis-ratio floor for anisotropic radii.
    name_via : :class:`str`
        If ``'radec'``, generate ``GROUP_NAME`` via
        :func:`SGA.io.radec_to_groupname`; otherwise leave it empty.
    sphere_link_arcmin : :class:`float`, optional
        Link length, arcmin, for the initial spherical preclustering
        (Step 1). Defaults to ``dmax`` (converted to arcmin).
    grid_cell_arcmin : :class:`float`, optional
        :class:`Grid` cell size, arcmin, for within-precluster neighbor
        search (Step 2). Defaults to ``dmax`` (converted to arcmin).
    mp : :class:`int`
        Number of parallel processes for linking (Step 2) and overlap
        merging (Step 4).
    contain : :class:`bool`
        If True, also link/merge pairs whose scaled ellipses overlap,
        independent of the distance-threshold checks (Steps 2 and 4).
    contain_margin : :class:`float`
        Fractional ellipse-scaling margin used for the Step 2
        containment/overlap checks (e.g. 0.75 = 75% expansion). Not
        used by Step 4 (see the Notes in
        :func:`_merge_overlapping_groups`).
    merge_centers : :class:`bool`
        If True, run Step 5 (merge groups whose centers are within
        ``merge_sep_arcsec``).
    merge_sep_arcsec : :class:`float`
        Group-center merge separation, arcsec, for Step 5.
    contain_search_arcmin : :class:`float`
        Separation, arcmin, below which :func:`should_link` tries the
        containment pre-check even for pairs that would otherwise fail
        the distance/threshold checks.
    min_group_diam_arcsec : :class:`float`
        Minimum group diameter floor, arcsec, applied in Step 6.
    manual_merge_pairs : :class:`list` of :class:`tuple` of :class:`str`
        Pairs of ``name_column`` values to force-merge into the same
        group (Step 3), regardless of separation/geometry.
    name_column : :class:`str`
        Column in ``cat`` used to resolve ``manual_merge_pairs`` names
        to row indices.

    Returns
    -------
    :class:`~astropy.table.Table`
        ``cat``, updated in place (and returned) with ``GROUP_NAME``,
        ``GROUP_MULT``, ``GROUP_PRIMARY``, ``GROUP_RA``, ``GROUP_DEC``,
        ``GROUP_DIAMETER`` columns.

    Raises
    ------
    ValueError
        If ``cat`` is missing ``RA``/``DEC``/``DIAM``, or if
        ``link_mode`` is not one of ``'off'``, ``'anisotropic'``,
        ``'hybrid'``, or None.

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

    # Step 3: Manual merges
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

    # Step 4: Merge groups with overlapping ellipses
    if contain:
        t_overlap_start = time.time()
        n_overlap_merges = _merge_overlapping_groups(dsu, state, contain_margin, mp)
        t_overlap_end = time.time()
        if n_overlap_merges > 0:
            log.info(f"Overlap merge: {n_overlap_merges} group pair(s) merged in {t_overlap_end-t_overlap_start:.2f}s")

    # Step 5: Merge centers (after overlap merge to catch all groups)
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

    # Step 6: Compute final groups (vectorized where possible)
    roots = np.array([dsu.find(i) for i in range(state.n)])
    uniq, inv = np.unique(roots, return_inverse=True)
    mult = np.bincount(inv, minlength=len(uniq))

    grp_ra = np.zeros(len(uniq))
    grp_dec = np.zeros(len(uniq))
    grp_diam = np.zeros(len(uniq))
    grp_primary = np.zeros(len(uniq), dtype=np.int64)

    # Vectorized group center computation using bincount
    ra_rad = state.ra * DEG2RAD
    dec_rad = state.dec * DEG2RAD
    cosd = np.cos(dec_rad)
    w = np.clip(state.diam, 1e-3, None)

    # Weighted Cartesian coordinates
    wx = cosd * np.cos(ra_rad) * w
    wy = cosd * np.sin(ra_rad) * w
    wz = np.sin(dec_rad) * w

    # Sum by group using bincount
    sum_wx = np.bincount(inv, weights=wx, minlength=len(uniq))
    sum_wy = np.bincount(inv, weights=wy, minlength=len(uniq))
    sum_wz = np.bincount(inv, weights=wz, minlength=len(uniq))
    sum_w = np.bincount(inv, weights=w, minlength=len(uniq))

    # Normalize and convert back to spherical
    x_norm = sum_wx / sum_w
    y_norm = sum_wy / sum_w
    z_norm = sum_wz / sum_w
    norm = np.sqrt(x_norm*x_norm + y_norm*y_norm + z_norm*z_norm)
    x_norm /= norm
    y_norm /= norm
    z_norm /= norm

    grp_ra = (np.arctan2(y_norm, x_norm) % (2*np.pi)) * RAD2DEG
    grp_dec = np.arctan2(z_norm, np.hypot(x_norm, y_norm)) * RAD2DEG

    # Compute diameters (still needs loop but optimized)
    for gi, r in enumerate(uniq):
        idx = np.where(roots==r)[0]
        ra_c, dec_c = grp_ra[gi], grp_dec[gi]

        # Vectorized distance computation
        cosd_c = math.cos(dec_c * DEG2RAD)
        dra = angdiff_deg(state.ra[idx], ra_c) * cosd_c
        ddec = state.dec[idx] - dec_c
        sep = np.hypot(dra, ddec) * ARCMIN_PER_DEG
        extent = sep + 0.5 * state.diam[idx]
        max_ext = extent.max()

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
        """Add or replace a 1D column on the enclosing ``cat``, casting
        to ``dtype`` and squeezing out any extraneous dimensions.

        Parameters
        ----------
        name : :class:`str`
            Column name.
        data : array-like
            Column data.
        dtype : optional
            Dtype to cast ``data`` to; if None, inferred.

        Returns
        -------
        None

        """
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

def make_singleton_group(cat):
    """Annotate a catalog with trivial one-object-per-group ``GROUP_*``
    columns (each object is its own primary, single-member group).

    Produces the exact same ``GROUP_*`` column set/dtypes as
    :func:`build_group_catalog`, so tables processed either way (e.g.
    a small isolated subsample vs. the full grouped catalog) can be
    safely ``vstack``ed together.

    Parameters
    ----------
    cat : :class:`~astropy.table.Table`
        Input catalog; needs ``RA``, ``DEC``, ``DIAM`` columns.

    Returns
    -------
    :class:`~astropy.table.Table`
        ``cat``, updated in place (and returned) with ``GROUP_NAME``
        (from :func:`SGA.io.radec_to_groupname`), ``GROUP_MULT`` (all
        1), ``GROUP_PRIMARY`` (all True), ``GROUP_RA``/``GROUP_DEC``
        (= ``RA``/``DEC``), ``GROUP_DIAMETER`` (= ``DIAM``).

    """
    n = len(cat)

    # Generate group names
    names = [radec_to_groupname(cat['RA'][k], cat['DEC'][k]) for k in range(n)]

    # Helper function to add/replace columns with proper dtype and shape
    def add_column(name, data, dtype=None):
        """Add or replace a 1D column on the enclosing ``cat``; see
        :func:`build_group_catalog`'s identical nested ``add`` helper.

        """
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
    """Flag ellipse-overlap within each group by setting
    ``cat['SAMPLE']``'s ``OVERLAP`` bit.

    For each group with more than one member (``GROUP_MULT > 1``),
    pairwise-checks every member's (unscaled) ellipse against every
    other member's (via :func:`ellipses_overlap`, with a cheap
    circular-radius pre-filter), and sets the ``OVERLAP`` bit for every
    member involved in at least one overlapping pair. Modifies
    ``cat['SAMPLE']`` in place.

    Parameters
    ----------
    cat : :class:`~astropy.table.Table`
        Input catalog, modified in place; needs ``GROUP_NAME``,
        ``GROUP_MULT``, ``RA``, ``DEC``, ``DIAM`` (arcmin), and
        optionally ``BA``, ``PA`` (degrees, astronomical convention --
        objects without them are treated as circular).
    SAMPLE : :class:`dict`
        Bitmask dictionary that includes key ``'OVERLAP'``.

    Returns
    -------
    None

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

        # Pairwise overlap checks (upper triangle only, vectorized inner loop)
        for ii in range(I.size - 1):
            # Vectorized offset computation to all subsequent members
            dx_deg = angdiff_deg(ra[ii+1:], ra[ii]) * cosd0
            dy_deg = dec[ii+1:] - dec[ii]

            # Vectorized separation
            sep = np.hypot(dx_deg, dy_deg) * ARCMIN_PER_DEG
            max_possible = (a_arc[ii] + a_arc[ii+1:]) * 1.0

            # Quick rejection
            candidates = np.where(sep <= max_possible)[0]

            # Check only promising candidates
            for kk in candidates:
                jj = ii + 1 + kk
                if ellipses_overlap(a_arc[ii], b_arc[ii], pa_rad[ii],
                                   a_arc[jj], b_arc[jj], pa_rad[jj],
                                   dx_deg[kk], dy_deg[kk], scale=1.0):  # No margin for overlap bit
                    overlapped[ii] = True
                    overlapped[jj] = True

        # Set the OVERLAP bit for overlapping members
        if np.any(overlapped):
            cat['SAMPLE'][I[overlapped]] |= OVERLAP_BIT
