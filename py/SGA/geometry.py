"""
SGA.geometry
============

Geometry (mostly ellipse-based) functions: light-weighted moment
ellipse fitting, elliptical apertures/masks and overlap testing,
Tractor ellipticity conversion, and literature-catalog geometry
selection/merging.

"""
import pdb
import numpy as np
from astropy.table import Table

from SGA.logger import log


class EllipseProperties:
    """Fit an ellipse to the flux distribution of the largest labelled
    blob in a 2D image, via light-weighted (optionally radially
    weighted) second moments.

    Call :meth:`fit` to perform the fit; results are stored as instance
    attributes (``x0``, ``y0``, ``a``, ``ba``, ``pa``, etc.) rather than
    returned separately, though :meth:`fit` also returns ``self`` for
    convenience. :meth:`plot` visualizes the fitted blob and ellipse.

    Attributes
    ----------
    x0, y0 : :class:`float`
        Fitted (or fixed) center, in pixel coordinates.
    ba : :class:`float`
        Fitted axis ratio, b/a.
    pa : :class:`float`
        Fitted position angle, degrees, astronomical convention (East of
        North), in [0, 180).
    a : :class:`float`
        Adopted semi-major axis (``a_rms`` or ``a_percentile``,
        depending on ``method``), pixels.
    a_rms : :class:`float`
        Semi-major axis from the second-moment (RMS) method, pixels.
    a_percentile : :class:`float`
        Semi-major axis enclosing a given flux percentile, pixels.
    labels : :class:`numpy.ndarray` or None
        Unused placeholder (never set by :meth:`fit`, which uses a local
        variable of the same name internally instead).
    blob_mask : :class:`numpy.ndarray`
        Boolean mask of the largest contiguous blob used for the fit.

    """

    def __init__(self):
        self.x0 = None
        self.y0 = None
        self.ba = None
        self.pa = None
        self.a = None
        self.a_rms = None
        self.a_percentile = None
        self.labels = None
        self.blob_mask = None

    @staticmethod
    def elliptical_radius(x, y, x0, y0, a, ba=1.0, pa_deg=0.0):
        """Compute the elliptical radius of point(s) relative to a
        given ellipse (1.0 = on the ellipse boundary).

        Parameters
        ----------
        x, y : array-like
            Point coordinates to evaluate.
        x0, y0 : :class:`float`
            Ellipse center.
        a : :class:`float`
            Semi-major axis.
        ba : :class:`float`
            Axis ratio, b/a.
        pa_deg : :class:`float`
            Position angle, degrees.

        Returns
        -------
        :class:`numpy.ndarray`
            Elliptical radius; < 1 inside the ellipse, 1 on the
            boundary, > 1 outside.

        """
        b = a * ba
        theta = np.deg2rad(pa_deg)

        dx = x - x0
        dy = y - y0

        xp =  dx * np.sin(theta) + dy * np.cos(theta)
        yp = -dx * np.cos(theta) + dy * np.sin(theta)

        r_ell = np.sqrt((xp / a)**2 + (yp / b)**2)
        return r_ell

    def fit(self, image, mask=None, method='percentile', percentile=0.95,
            radial_power=0.7, x0y0=None, input_ba_pa=None, smooth_sigma=1.0,
            use_radial_weight=True):
        """Label and smooth the image, then select the largest contiguous
        blob and compute ellipse properties using flux-weighted second
        moments.

        Smooths ``image`` (Gaussian, ``smooth_sigma``), finds the
        largest contiguous blob of positive (and, if given,
        ``mask``-passing) pixels, and computes a flux-weighted centroid
        (unless ``x0y0`` fixes it). If ``input_ba_pa`` is given, solves
        only for a single scale factor ``a`` under that fixed
        axis-ratio/PA (used when the shape should be held at an input,
        e.g. Tractor, value); otherwise diagonalizes the flux-weighted
        inertia tensor to get the semi-major axis, axis ratio, and
        position angle directly. In either case, also computes an
        independent shape-agnostic ``a_percentile`` (the radius
        enclosing the given flux ``percentile``), and ``method``
        selects which of ``a_rms``/``a_percentile`` becomes the adopted
        ``a``. All results are stored on ``self``; on any failure
        (no good pixels, no blobs, too few pixels, degenerate fixed-shape
        solution), falls back to a zeroed/degenerate geometry (``a=0``,
        ``ba=1``, ``pa=0``) with a logged warning, rather than raising.

        Parameters
        ----------
        image : :class:`numpy.ndarray`
            2D image to fit.
        mask : :class:`numpy.ndarray`, optional
            2D boolean array, True = usable pixels; vetoes blob
            detection where False. If None, only ``image > 0`` is used.
        method : {'percentile', 'rms'}
            Which semi-major-axis estimate becomes the adopted ``a``.
        percentile : :class:`float`
            Cumulative flux fraction defining ``a_percentile`` (radial,
            shape-agnostic).
        radial_power : :class:`float`
            Exponent of the radial weighting applied to the flux when
            ``use_radial_weight`` is True (down-weights the core,
            emphasizing outer isophotes in the second-moment
            calculation).
        x0y0 : :class:`tuple`, optional
            ``(x0, y0)`` to hold the center fixed at, instead of solving
            for the flux-weighted centroid.
        input_ba_pa : :class:`tuple`, optional
            ``(ba, pa)`` to hold the shape fixed at (e.g. from Tractor),
            solving only for the overall scale ``a``. If the resulting
            fixed-shape scale is implausibly small relative to
            ``a_percentile`` (< 10%), falls back to ``a_percentile``
            instead, on the assumption the input shape is mismatched to
            the actual flux distribution.
        smooth_sigma : :class:`float`
            Gaussian smoothing sigma, pixels, applied before blob
            detection; disabled if 0/None.
        use_radial_weight : :class:`bool`
            If True, second moments are weighted by ``flux * r**radial_power``
            instead of flux alone.

        Returns
        -------
        :class:`EllipseProperties`
            ``self``, with ``x0``, ``y0``, ``a``, ``a_rms``,
            ``a_percentile``, ``ba``, ``pa``, and ``blob_mask``
            populated.

        """
        from scipy import ndimage

        if smooth_sigma and smooth_sigma > 0:
            smoothed = ndimage.gaussian_filter(image, sigma=smooth_sigma)
        else:
            smoothed = image

        if mask is None:
            good = (smoothed > 0)
        else:
            good = (smoothed > 0) & mask

        if not np.any(good):
            log.warning("No good pixels in EllipseProperties.fit.")
            self.x0 = self.y0 = 0.0
            self.a = self.a_rms = self.a_percentile = 0.0
            self.ba = 1.0
            self.pa = 0.0
            return self

        # segment: pick the largest contiguous blob in the *good* pixels
        labels, nblobs = ndimage.label(good)
        if nblobs < 1:
            log.warning("No labelled blobs in EllipseProperties.fit.")
            self.x0 = self.y0 = 0.0
            self.a = self.a_rms = self.a_percentile = 0.0
            self.ba = 1.0
            self.pa = 0.0
            return self

        sizes = ndimage.sum(good, labels, index=np.arange(1, nblobs + 1))
        largest = np.argmax(sizes) + 1
        self.blob_mask = (labels == largest)

        # now restrict to that blob *only*
        blob_idx = np.flatnonzero(self.blob_mask)
        yy, xx = np.indices(smoothed.shape)
        x_sel = xx.flat[blob_idx].astype(float)
        y_sel = yy.flat[blob_idx].astype(float)
        flux = smoothed.flat[blob_idx].astype(float)
        #print(image.shape, x_sel.shape, flux.shape)

        F = flux.sum()
        if flux.size < 3 or F <= 0.:
            log.warning("Too few pixels in largest blob for ellipse fit.")
            self.x0 = self.y0 = 0.0
            self.a = self.a_rms = self.a_percentile = 0.0
            self.ba = 1.0
            self.pa = 0.0
            return self

        # flux-weighted centroid (optionally fixed)
        if x0y0 is None:
            self.x0 = np.dot(flux, x_sel) / F
            self.y0 = np.dot(flux, y_sel) / F
        else:
            self.x0 = x0y0[0]
            self.y0 = x0y0[1]

        # geometry vectors and radius
        dx = x_sel - self.x0
        dy = y_sel - self.y0
        r = np.hypot(dx, dy)

        # weights for the second moments
        if use_radial_weight:
            wflux = flux * r**radial_power
        else:
            wflux = flux

        F_w = wflux.sum()
        if F_w <= 0:
            # Fallback to unweighted if pathological
            wflux = flux
            F_w = F

        # Fixed-shape branch: use input (b/a, PA) and solve only for
        # scale a.
        if input_ba_pa is not None:
            ba_in, pa_in = input_ba_pa
            # Guard against pathological b/a
            q = float(np.clip(ba_in, 1e-2, 1.0))

            # PA is astronomical (degrees E of N, North-up).
            # In numpy pixel coords (x=col, y=row), North-up means the major axis
            # points along +y, so we rotate by (90 - pa_in) to convert.
            theta = np.deg2rad(pa_in)

            # Astronomical PA is E of N, but in standard FITS images RA increases
            # rightward, so East is left: flip the x-axis sign.
            theta = np.deg2rad(pa_in)
            xp =  dx * np.sin(theta) - dy * np.cos(theta)
            yp = -dx * np.cos(theta) - dy * np.sin(theta)

            Mmaj = np.dot(wflux, xp * xp) / F_w
            Mmin = np.dot(wflux, yp * yp) / F_w

            A = (Mmaj + q*q * Mmin) / (1.0 + q**4)  # a^2
            #A = (Mmaj + q*q * Mmin) / (1.0 + q**4)  # a^2
            if not np.isfinite(A) or A <= 0.0:
                log.warning("Failed fixed-shape scale solution; falling back to zero geometry.")
                self.a = self.a_rms = self.a_percentile = 0.0
                self.ba = 1.0
                self.pa = 0.0
                return self

            a_fixed = np.sqrt(A)

            # Percentile radius: radial distance in pixels, independent of shape
            order_r = np.argsort(r)
            cumflux = np.cumsum(flux[order_r])
            cumfrac = cumflux / cumflux[-1]
            self.a_percentile = float(np.interp(percentile, cumfrac, r[order_r]))

            # Sanity check: if moment-based scale is implausibly small vs
            # percentile radius, the input geometry is probably mismatched
            # to the actual flux distribution — fall back to percentile.
            if self.a_percentile > 0 and a_fixed < 0.1 * self.a_percentile:
                log.warning(f"Fixed-shape a_rms ({a_fixed:.2f}) << a_percentile "
                            f"({self.a_percentile:.2f}); likely geometry mismatch "
                            f"(ba={ba_in:.3f}, pa={pa_in:.1f}). Using a_percentile.")
                a_fixed = self.a_percentile

            # RMS-based “a” is this fixed scale
            self.a_rms = a_fixed

            # Choose final a
            if method == 'rms':
                self.a = self.a_rms
            elif method == 'percentile':
                self.a = self.a_percentile
            else:
                raise ValueError(f"Unknown method: {method}")

            # Adopt Tractor-like shape
            self.ba = q
            self.pa = pa_in % 180.0

            return self

        # central second moments with chosen weights
        Mxx = np.dot(wflux, dx*dx) / F_w
        Myy = np.dot(wflux, dy*dy) / F_w
        Mxy = np.dot(wflux, dx*dy) / F_w

        # diagonalize inertia tensor
        eigvals, eigvecs = np.linalg.eigh([[Mxx, Mxy], [Mxy, Myy]])
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        # RMS-based semi-major axis from largest eigenvalue
        self.a_rms = np.sqrt(max(eigvals[0], 0.0))

        # percentile-based radius (using *flux* weights, not wflux)
        order_r = np.argsort(r)
        cumflux = np.cumsum(flux[order_r])
        cumfrac = cumflux / cumflux[-1]
        self.a_percentile = float(np.interp(percentile, cumfrac, r[order_r]))

        # select final major axis
        if method == 'rms':
            self.a = self.a_rms
        elif method == 'percentile':
            self.a = self.a_percentile
        else:
            raise ValueError(f"Unknown method: {method}")

        # axis ratio and PA
        if self.a_rms > 0. and eigvals[1] > 0.:
            b = np.sqrt(eigvals[1])
            if b < 1e-2:
                log.warning('Unrealistically small semi-minor axis.')
                self.ba = 1.0
                self.pa = 0.0
            else:
                self.ba = b / self.a_rms
                vx, vy = eigvecs[:, 0]
                pa_cart = np.degrees(np.arctan2(vy, vx)) % 180.0
                self.pa = (pa_cart - 90.0) % 180.0
        else:
            log.warning('Unable to determine the ellipse geometry.')
            self.ba = 1.0
            self.pa = 0.0

        return self


    def plot(self, image=None, ax=None, imshow_kwargs=None,
             ellipse_kwargs=None, blob_outline_kwargs=None):
        """Display the provided image (or blob mask) with the outline of
        the selected largest blob and overlay the fitted ellipse.

        Parameters
        ----------
        image : :class:`numpy.ndarray`, optional
            Image to display. If None, uses ``self.blob_mask``.
        ax : :class:`~matplotlib.axes.Axes`, optional
            Axes to plot into. If None, a new figure/axes is created.
        imshow_kwargs : :class:`dict`, optional
            Passed to ``ax.imshow`` (e.g. ``cmap``, ``origin``).
        ellipse_kwargs : :class:`dict`, optional
            Passed to the :class:`~matplotlib.patches.Ellipse` patch
            (e.g. ``edgecolor``).
        blob_outline_kwargs : :class:`dict`, optional
            Passed to ``ax.contour`` when outlining ``self.blob_mask``
            (e.g. ``colors``, ``linewidths``).

        Returns
        -------
        :class:`~matplotlib.axes.Axes`
            The axes plotted into.

        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Ellipse

        if ax is None:
            fig, ax = plt.subplots()
        if imshow_kwargs is None:
            imshow_kwargs = {'origin': 'lower', 'cmap': 'cividis'}
        disp = image if image is not None else self.blob_mask
        ax.imshow(disp, **imshow_kwargs)

        # outline only the largest blob
        if blob_outline_kwargs is None:
            blob_outline_kwargs = {'colors': 'k', 'linewidths': 1, 'alpha': 0.7}
        ax.contour(self.blob_mask, levels=[0.5], **blob_outline_kwargs)

        # overlay ellipse
        if ellipse_kwargs is None:
            ellipse_kwargs = {'edgecolor': 'red', 'facecolor': 'none'}
        angle = self.pa - 90.
        ell = Ellipse((self.x0, self.y0), 2*self.a, 2*self.a*self.ba,
                      angle=angle, **ellipse_kwargs)
        ax.add_patch(ell)

        # centroid marker
        ax.plot(self.x0, self.y0, marker='+', color=ellipse_kwargs.get('edgecolor','red'))
        return ax


def in_ellipse_mask(xcen, ycen, semia, semib, pa, x, y):
    """Test whether points fall inside an ellipse, using the
    astronomical position-angle convention.

    Parameters
    ----------
    xcen, ycen : :class:`float`
        Center of the ellipse.
    semia, semib : :class:`float`
        Semi-major and semi-minor axes lengths, in pixels.
    pa : :class:`float`
        Position angle in degrees, measured CCW from the +y (north) axis.
    x, y : array-like
        Coordinates to test.

    Returns
    -------
    :class:`numpy.ndarray` of :class:`bool`
        True for points inside or on the ellipse.

    Notes
    -----
    If generating a mask for an image of size ``size``, ``x`` and ``y``
    should be defined as::

        xgrid, ygrid = np.meshgrid(np.arange(size), np.arange(size), indexing='xy')

    """
    # convert PA to radians
    theta = np.deg2rad(pa)
    dx = x - xcen
    dy = y - ycen

    # Major-axis direction vector = (sin θ, cos θ)
    xp =  dx * np.sin(theta) + dy * np.cos(theta)
    # Minor-axis (90° CCW from major) = (–cos θ, sin θ)
    yp = -dx * np.cos(theta) + dy * np.sin(theta)

    return (xp/semia)**2 + (yp/semib)**2 <= 1


def ellipses_overlap(bx, by, sma, ba, pa, refbx, refby, refsma,
                     refba, refpa, ntheta=64):
    """Test whether two ellipses plausibly overlap, by sampling points on
    each boundary.

    Checks four conditions (any True means overlap): each ellipse's
    center falls inside the other; or ``ntheta`` boundary-sampled points
    of either ellipse fall inside the other (via :func:`in_ellipse_mask`).
    This is a fast, sampling-based approximation -- it can miss a very
    thin sliver of true overlap between two similarly-oriented
    ellipses, but is robust for the elliptical apertures used elsewhere
    in this pipeline.

    Parameters
    ----------
    bx, by : :class:`float`
        Center of ellipse 1, pixels.
    sma : :class:`float`
        Semi-major axis of ellipse 1, pixels.
    ba : :class:`float`
        Axis ratio (b/a) of ellipse 1.
    pa : :class:`float`
        Position angle of ellipse 1, degrees (astronomical convention).
    refbx, refby : :class:`float`
        Center of ellipse 2, pixels.
    refsma : :class:`float`
        Semi-major axis of ellipse 2, pixels.
    refba : :class:`float`
        Axis ratio (b/a) of ellipse 2.
    refpa : :class:`float`
        Position angle of ellipse 2, degrees.
    ntheta : :class:`int`
        Number of boundary points sampled per ellipse.

    Returns
    -------
    :class:`bool`
        True if the ellipses plausibly overlap.

    """
    semib  = sma * ba
    refsemib = refsma * refba

    # Quick checks: centers inside the other ellipse?
    if in_ellipse_mask(refbx, refby, refsma, refsemib, refpa, bx, by):
        return True
    if in_ellipse_mask(bx, by, sma, semib, pa, refbx, refby):
        return True

    # Sample points on ellipse 1 boundary, test against ellipse 2
    theta = np.deg2rad(pa)
    t = np.linspace(0, 2*np.pi, ntheta, endpoint=False)
    xp = sma * np.cos(t)
    yp = semib * np.sin(t)

    # Map (xp,yp) in ellipse coords -> (x,y) in image coords
    # major axis dir = (sinθ, cosθ), minor = (–cosθ, sinθ)
    dx = xp * np.sin(theta) + yp * (-np.cos(theta))
    dy = xp * np.cos(theta) + yp * np.sin(theta)
    x1 = bx + dx
    y1 = by + dy

    if np.any(in_ellipse_mask(refbx, refby, refsma, refsemib, refpa, x1, y1)):
        return True

    # Sample points on ellipse 2 boundary, test against ellipse 1
    theta2 = np.deg2rad(refpa)
    t2 = np.linspace(0, 2*np.pi, ntheta, endpoint=False)
    xp2 = refsma * np.cos(t2)
    yp2 = refsemib * np.sin(t2)
    dx2 = xp2 * np.sin(theta2) + yp2 * (-np.cos(theta2))
    dy2 = xp2 * np.cos(theta2) + yp2 * np.sin(theta2)
    x2 = refbx + dx2
    y2 = refby + dy2

    if np.any(in_ellipse_mask(bx, by, sma, semib, pa, x2, y2)):
        return True

    return False


def get_tractor_ellipse(r50, e1, e2):
    """Convert a Tractor half-light radius and ellipticity components to
    a diameter, axis ratio, and position angle.

    Taken in part from ``tractor.ellipses.EllipseE``. The returned
    diameter inflates the light-weighted ``r50`` by a factor of 2.4
    (2x radius-to-diameter, x1.2 to extrapolate beyond the half-light
    radius) as an approximate initial guess for the object's full
    extent.

    Parameters
    ----------
    r50 : :class:`float`
        Tractor half-light radius, arcsec.
    e1, e2 : :class:`float`
        Tractor ellipticity components.

    Returns
    -------
    diam : :class:`float`
        Approximate full diameter, same units as ``r50``.
    ba : :class:`float`
        Axis ratio, b/a.
    pa : :class:`float`
        Position angle, degrees, astronomical convention, in [0, 180).

    """
    e = np.hypot(e1, e2)
    ba = (1. - e) / (1. + e)
    #e = (1 - ba) / (1 + ba)

    phi = -np.rad2deg(np.arctan2(e2, e1) / 2.)
    #angle = 2 * np.deg2rad(pa - 180.)
    #e1 = e * np.cos(angle)
    #e2 = e * np.sin(angle)

    pa = (180. - phi) % 180
    diam = r50 * 2. * 1.2 # [radius-->diameter then 20% higher]

    return diam, ba, pa


def get_basic_geometry(cat, galaxy_column='OBJNAME', verbose=False):
    """From a catalog containing magnitudes, diameters, position angles,
    and ellipticities from one or more literature sources, derive a
    single "basic" (best-available) value for each property, per
    object, along with which source it came from.

    Which strategy is used is determined by which columns are present
    in ``cat`` (checked in order, first match wins -- these are treated
    as mutually exclusive catalog types, not merged):

    1. ``LOGD25`` present -> HyperLeda-derived catalog: diameter from
       ``LOGD25``, axis ratio from ``LOGR25``, PA and B-band magnitude
       (``BT``) taken directly.
    2. ``D26`` present -> SGA2020-derived catalog: diameter/BA/PA/mag
       taken directly from the SGA2020 columns.
    3. ``RHALF`` present -> LVD-derived catalog: diameter from
       ``RHALF`` (half-light radius, inflated x2.4 to full diameter),
       BA from ``1 - ELLIPTICITY``, PA and V-band magnitude
       (``APPARENT_MAGNITUDE_V``) taken directly.
    4. ``SHAPE_R`` present -> DR9/DR10 Tractor-derived catalog:
       diameter/BA/PA from Tractor's ``SHAPE_R``/``SHAPE_E1``/``SHAPE_E2``
       (via :func:`get_tractor_ellipse`), r-band magnitude from
       ``FLUX_R``.
    5. ``DIAM`` present -> already-basic/custom catalog: columns taken
       directly (``DIAM``, ``BA``, ``PA``, ``MAG``, ``MAG_BAND``).
    6. Otherwise -> NED-assembled catalog: for magnitude, merges
       ``SDSS``/``TWOMASS``/``RC3`` columns; for diameter/BA/PA, merges
       ``ESO``/``SDSS``/``TWOMASS``/``RC3`` columns. Each property is
       filled from every available reference in turn (later references
       in the loop overwrite earlier ones for objects with more than one
       valid source), which produces an *effective* per-property
       priority order of RC3 > TWOMASS > SDSS > ESO (highest priority
       is checked *last* so it wins the overwrite) -- non-obvious from
       the code, since nothing explicitly says "prefer RC3". Any value
       still missing after that is backfilled from the catalog's
       ``BASIC_*`` columns (``BASIC_MAG`` and V-band; ``BASIC_DMAJOR``/
       ``BASIC_DMINOR`` for diameter/BA), the lowest-priority fallback.

    Parameters
    ----------
    cat : :class:`~astropy.table.Table`
        Input catalog; which columns are present determines which
        strategy above is used.
    galaxy_column : :class:`str`
        Column in ``cat`` to copy into the output ``GALAXY`` column.
    verbose : :class:`bool`
        If True, log the number of objects with a valid magnitude and
        diameter.

    Returns
    -------
    :class:`~astropy.table.Table`
        One row per object in ``cat``, with ``GALAXY`` plus, depending
        on the strategy used, columns named
        ``{MAG,DIAM,BA,PA}_{REF}``/``{MAG,DIAM,BA,PA}_{REF}_REF``/
        ``BAND_{REF}`` (for the HyperLeda/SGA2020/LVD/DR9-10 strategies,
        ``REF`` being that source's name) or
        ``{MAG,DIAM,BA,PA}_LIT``/``{MAG,DIAM,BA,PA}_LIT_REF``/
        ``BAND_LIT`` (for the custom/NED strategies). Missing values are
        ``-99.``/``''`` as appropriate.

    """
    nobj = len(cat)

    basic = Table()
    basic['GALAXY'] = cat[galaxy_column].value

    # default
    magcol = 'MAG_LIT'
    diamcol = 'DIAM_LIT'

    # HyperLeda
    if 'LOGD25' in cat.columns:
        ref = 'HYPERLEDA'
        magcol = f'MAG_{ref}'
        diamcol = f'DIAM_{ref}'
        for prop in ('mag', 'diam', 'ba', 'pa'):
            val = np.zeros(nobj, 'f4') - 99.
            val_ref = np.zeros(nobj, '<U9')
            val_band = np.zeros(nobj, 'U1')

            if prop == 'mag':
                col = 'BT'
                band = 'B'
                I = (cat[col] > 0.) * (cat[col] < 1e20)
                if np.sum(I) > 0:
                    val[I] = cat[col][I]
                    val_ref[I] = ref
                    val_band[I] = band
            elif prop == 'diam':
                col = 'LOGD25'
                I = cat[col] > 0.
                if np.sum(I) > 0:
                    val[I] = 0.1 * 10.**cat[col][I]
                    val_ref[I] = ref
            elif prop == 'ba':
                col = 'LOGR25'
                I = ~np.isnan(cat[col]) * (cat[col] != 0.)
                if np.sum(I) > 0:
                    val[I] = 10.**(-cat[col][I])
                    val_ref[I] = ref
            elif prop == 'pa':
                col = 'PA'
                I = ~np.isnan(cat[col])
                if np.sum(I) > 0:
                    val[I] = cat[col][I]
                    val_ref[I] = ref

            basic[f'{prop.upper()}_{ref}'] = val
            basic[f'{prop.upper()}_{ref}_REF'] = val_ref
            if prop == 'mag':
                basic[f'BAND_{ref}'] = val_band

    # SGA2020
    elif 'D26' in cat.columns:
        ref = 'SGA2020'
        magcol = f'MAG_{ref}'
        diamcol = f'DIAM_{ref}'
        for prop in ('mag', 'diam', 'ba', 'pa'):
            val = np.zeros(nobj, 'f4') - 99.
            val_ref = np.zeros(nobj, '<U9')
            val_band = np.zeros(nobj, 'U1')

            if prop == 'mag':
                col = 'R_MAG_SB26'
                band = 'R'
                I = cat[col] > 0.
                if np.sum(I) > 0:
                    val[I] = cat[col][I]
                    val_ref[I] = ref
                    val_band[I] = band
            elif prop == 'diam':
                col = 'D26'
                I = cat[col] > 0.
                if np.sum(I) > 0:
                    val[I] = cat[col][I]
                    val_ref[I] = ref
            elif prop == 'ba':
                col = 'BA'
                I = ~np.isnan(cat[col]) * (cat[col] != 0.)
                if np.sum(I) > 0:
                    val[I] = cat[col][I]
                    val_ref[I] = ref
            elif prop == 'pa':
                col = 'PA'
                I = ~np.isnan(cat[col])
                if np.sum(I) > 0:
                    val[I] = cat[col][I]
                    val_ref[I] = ref

            basic[f'{prop.upper()}_{ref}'] = val
            basic[f'{prop.upper()}_{ref}_REF'] = val_ref
            if prop == 'mag':
                basic[f'BAND_{ref}'] = val_band

    # LVD
    elif 'RHALF' in cat.columns:
        ref = 'LVD'
        for prop in ('mag', 'diam', 'ba', 'pa'):
            val = np.zeros(nobj, 'f4') - 99.
            val_ref = np.zeros(nobj, '<U9')
            val_band = np.zeros(nobj, 'U1')

            if prop == 'mag':
                col = 'APPARENT_MAGNITUDE_V'
                band = 'V'
                I = cat[col] > 0.
                if np.sum(I) > 0:
                    val[I] = cat[col][I]
                    val_ref[I] = ref
                    val_band[I] = band
            elif prop == 'diam':
                col = 'RHALF' # [arcmin]
                I = cat[col] > 0.
                if np.sum(I) > 0:
                    # see analyze-lvd
                    val[I] = cat[col][I] * 1.2 * 2. # half-light-->full-light; radius-->diameter
                    val_ref[I] = ref
            elif prop == 'ba':
                col = 'ELLIPTICITY' # =1-b/a
                I = ~np.isnan(cat[col])
                if np.sum(I) > 0:
                    val[I] = 1. - cat[col][I]
                    val_ref[I] = ref
            elif prop == 'pa':
                col = 'POSITION_ANGLE'
                I = ~np.isnan(cat[col])
                if np.sum(I) > 0:
                    val[I] = cat[col][I] % 180 # put in the range [0, 180]
                    val_ref[I] = ref

            basic[f'{prop.upper()}_LIT'] = val
            basic[f'{prop.upper()}_LIT_REF'] = val_ref
            if prop == 'mag':
                basic[f'BAND_LIT'] = val_band

    # DR9/DR10 supplement
    elif 'SHAPE_R' in cat.columns:
        ref = 'DR910'
        for prop in ('mag', 'diam', 'ba', 'pa'):
            val = np.zeros(nobj, 'f4') - 99.
            val_ref = np.zeros(nobj, '<U9')
            val_band = np.zeros(nobj, 'U1')

            if prop == 'mag':
                col = 'FLUX_R'
                band = 'r'
                I = cat[col] > 0.
                if np.sum(I) > 0:
                    val[I] = 22.5 - 2.5 * np.log10(cat[col][I])
                    val_ref[I] = ref
                    val_band[I] = band
            elif prop == 'diam':
                col = 'SHAPE_R'
                I = cat[col] > 0.
                if np.sum(I) > 0:
                    val[I] = 2. * 1.2 * cat[col][I] / 60. # [arcmin]
                    val_ref[I] = ref
            elif prop == 'ba' or prop == 'pa':
                I = ~np.isnan(cat['SHAPE_E1']) * ~np.isnan(cat['SHAPE_E2'])
                if np.sum(I) > 0:
                    from SGA.geometry import get_tractor_ellipse
                    _, ba, pa = get_tractor_ellipse(cat['SHAPE_R'][I], cat['SHAPE_E1'][I], cat['SHAPE_E2'][I])
                    if prop == 'ba':
                        val[I] = ba
                    elif prop == 'pa':
                        val[I] = pa
                    val_ref[I] = ref

            basic[f'{prop.upper()}_LIT'] = val
            basic[f'{prop.upper()}_LIT_REF'] = val_ref
            if prop == 'mag':
                basic[f'BAND_LIT'] = val_band

    # custom
    elif 'DIAM' in cat.columns:
        ref = 'CUSTOM'
        for prop in ('mag', 'diam', 'ba', 'pa'):
            val = np.zeros(nobj, 'f4') - 99.
            val_ref = np.zeros(nobj, '<U9')
            val_band = np.zeros(nobj, 'U1')

            if prop == 'mag':
                col = 'MAG'
                I = cat[col] > 0.
                if np.sum(I) > 0:
                    val[I] = cat[col][I]
                    val_ref[I] = ref
                    val_band[I] = cat[f'{col}_BAND'][I]
            elif prop == 'diam':
                col = 'DIAM' # [arcmin]
                I = cat[col] > 0.
                if np.sum(I) > 0:
                    val[I] = cat[col][I]
                    val_ref[I] = ref
            elif prop == 'ba':
                col = 'BA'
                I = cat[col] != -99.
                if np.sum(I) > 0:
                    val[I] = cat[col][I]
                    val_ref[I] = ref
            elif prop == 'pa':
                col = 'PA'
                I = cat[col] != -99.
                if np.sum(I) > 0:
                    val[I] = cat[col][I]
                    val_ref[I] = ref

            basic[f'{prop.upper()}_LIT'] = val
            basic[f'{prop.upper()}_LIT_REF'] = val_ref
            if prop == 'mag':
                basic[f'BAND_LIT'] = val_band
    # NED
    else:
        for prop in ('mag', 'diam', 'ba', 'pa'):
            if prop == 'mag':
                refs = ('SDSS', 'TWOMASS', 'RC3')
                bands = ('R', 'K', 'B')
            else:
                refs = ('ESO', 'SDSS', 'TWOMASS', 'RC3')
                bands = ('B', 'R', 'K', 'B')
            nref = len(refs)

            val = np.zeros(nobj, 'f4') - 99.
            val_ref = np.zeros(nobj, '<U9')
            val_band = np.zeros(nobj, 'U1')

            #allI = np.zeros((nobj, nref), bool)
            for iref, (ref, band) in enumerate(zip(refs, bands)):
                if prop == 'mag':
                    col = f'{ref}_{band}'
                else:
                    col = f'{ref}_{prop.upper()}_{band}'
                I = cat[col] > 0.
                #allI[:, iref] = I

                if np.sum(I) > 0:
                    val[I] = cat[col][I]
                    val_ref[I] = ref
                    val_band[I] = band

            basic[f'{prop.upper()}_LIT'] = val
            basic[f'{prop.upper()}_LIT_REF'] = val_ref
            if prop == 'mag':
                basic[f'BAND_LIT'] = val_band

        # supplement any missing values with the "BASIC" data
        I = (basic['MAG_LIT'] <= 0.) * (cat['BASIC_MAG'] > 0.)
        if np.any(I):
            basic['MAG_LIT'][I] = cat['BASIC_MAG'][I]
            basic['BAND_LIT'][I] = 'V'

        I = (basic['DIAM_LIT'] <= 0.) * (cat['BASIC_DMAJOR'] > 0.)
        if np.any(I):
            basic['DIAM_LIT'][I] = cat['BASIC_DMAJOR'][I]
            basic['DIAM_LIT_REF'][I] = 'BASIC'

        I = (basic['BA_LIT'] <= 0.) * (cat['BASIC_DMAJOR'] > 0.) * (cat['BASIC_DMINOR'] > 0.)
        if np.any(I):
            basic['BA_LIT'][I] = cat['BASIC_DMINOR'][I] / cat['BASIC_DMAJOR'][I]
            basic['BA_LIT_REF'][I] = 'BASIC'

    # summarize
    if verbose:
        M = basic[magcol] > 0.
        D = basic[diamcol] > 0.
        log.info(f'Derived photometry for {np.sum(M):,d}/{nobj:,d} objects and ' + \
                 f'diameters for {np.sum(D):,d}/{nobj:,d} objects.')

    return basic


def parse_geometry(cat, ref, mindiam=152*0.262):
    """Extract one named source's diameter/BA/PA geometry from a
    :func:`get_basic_geometry`-style catalog.

    Parameters
    ----------
    cat : :class:`~astropy.table.Table`
        Catalog with ``DIAM_{ref}``/``BA_{ref}``/``PA_{ref}`` columns
        for the requested ``ref`` (arcmin for diameter).
    ref : :class:`str`
        Which source to extract. Only ``'SGA2020'``, ``'HYPERLEDA'``,
        and ``'LIT'`` are actually implemented as branches -- despite
        what the name might suggest, there are no separate ``'RC3'``,
        ``'LVD'``, or ``'SMUDGes'`` branches (those are just possible
        values *within* ``DIAM_LIT_REF`` when ``ref='LIT'``, set
        upstream by :func:`get_basic_geometry`/:func:`choose_geometry`).
        Passing any other value silently returns all-missing results
        (``diam=-99``, ``ba=1``, ``pa=0``, ``outref=''``) for every
        object, with no warning or error.
    mindiam : :class:`float`
        Accepted but unused -- the code that would apply it as a
        diameter floor is commented out in the function body.

    Returns
    -------
    diam : :class:`float` or :class:`numpy.ndarray`
        Diameter, arcsec (``-99.`` where missing); scalar if
        ``len(cat) == 1``.
    ba : :class:`float` or :class:`numpy.ndarray`
        Axis ratio, b/a (default ``1.`` where missing).
    pa : :class:`float` or :class:`numpy.ndarray`
        Position angle, degrees (default ``0.`` where missing).
    outref : :class:`str` or :class:`numpy.ndarray`
        Source reference string for each object (``''`` where missing).

    """
    nobj = len(cat)
    diam = np.zeros(nobj) - 99. # [arcsec]
    ba = np.ones(nobj)
    pa = np.zeros(nobj)
    outref = np.zeros(nobj, '<U9')

    if ref == 'SGA2020':
        I = cat['DIAM_SGA2020'] > 0.
        if np.any(I):
            diam[I] = cat['DIAM_SGA2020'][I] * 60. # [arcsec]
            ba[I] = cat['BA_SGA2020'][I]
            pa[I] = cat['PA_SGA2020'][I]
            outref[I] = ref
    elif ref == 'HYPERLEDA':
        I = cat['DIAM_HYPERLEDA'] > 0.
        if np.any(I):
            diam[I] = cat['DIAM_HYPERLEDA'][I] * 60. # [arcsec]
            ba[I] = cat['BA_HYPERLEDA'][I]
            pa[I] = cat['PA_HYPERLEDA'][I]
            outref[I] = ref
    elif ref == 'LIT':
        I = cat['DIAM_LIT'] > 0.
        if np.any(I):
            diam[I] = cat['DIAM_LIT'][I] * 60. # [arcsec]
            ba[I] = cat['BA_LIT'][I]
            pa[I] = cat['PA_LIT'][I]
            outref[I] = cat['DIAM_LIT_REF']

    #I = diam <= 0.
    #if np.any(I):
    #    diam[I] = mindiam # [arcsec]
    #    outref[I] = 'NONE'

    # clean up missing values of BA and PA
    ba[ba < 0.] = 1.
    pa[pa < 0.] = 0.

    if nobj == 1:
        return diam[0], ba[0], pa[0], outref[0]
    else:
        return diam, ba, pa, outref


def choose_geometry(cat, mindiam=152*0.262, get_mag=False):
    """Choose an object's final parent-sample geometry, selecting
    between the NED-assembled (literature) values (``DIAM``, ``BA``,
    ``PA``), the SGA2020 values (``DIAM_SGA2020``, ``BA_SGA2020``,
    ``PA_SGA2020``), and HyperLeda's values (``DIAM_HYPERLEDA``,
    ``BA_HYPERLEDA``, ``PA_HYPERLEDA``).

    If ``cat`` already has plain ``DIAM``/``BA``/``PA`` columns (i.e. is
    already a single-geometry parent catalog), those are returned
    directly with ``ref='parent'`` for every object -- see Notes for a
    caveat on this path when ``get_mag=True``. Otherwise, per object:
    LVD-sourced literature values (``DIAM_LIT_REF == 'LVD'``) are always
    preferred first, since they were visually determined and inspected;
    then, among SGA2020/HyperLeda/literature, the source with the
    *largest* diameter is preferred, provided diameter, BA, *and* PA
    are all valid for that source (falling back to a second pass that
    only requires a valid diameter if no source has all three); within
    the literature source, ``DIAM_LIT_REF`` values of ``'SMUDGes'``,
    ``'LVD'``, or ``'RC3'`` are surfaced as the reported ``ref`` in
    place of the generic ``'LIT'`` label. Objects with no valid
    diameter from any source get ``ref='NONE'`` and are floored at
    ``mindiam``; all diameters below ``mindiam`` (~40 arcsec, in
    ``mindiam``'s default) are floored there too. BA is clipped to
    [0.1, 1.0] and PA wrapped to [0, 180).

    Parameters
    ----------
    cat : :class:`~astropy.table.Table`
        Catalog with either plain ``DIAM``/``BA``/``PA`` columns, or the
        ``DIAM_{SGA2020,HYPERLEDA,LIT}``/``BA_{...}``/``PA_{...}``/
        ``DIAM_LIT_REF`` columns produced by :func:`get_basic_geometry`.
    mindiam : :class:`float`
        Minimum diameter floor, arcsec (default ~40 arcsec, i.e.
        ``152 * 0.262`` pixels at the optical pixel scale).
    get_mag : :class:`bool`
        If True, also select and return a magnitude/band (see Notes for
        the early-return-path caveat).

    Returns
    -------
    diam : :class:`numpy.ndarray`
        Diameter, arcsec (arcmin x 60 from the source columns).
    ba : :class:`numpy.ndarray`
        Axis ratio, b/a, clipped to [0.1, 1.0].
    pa : :class:`numpy.ndarray`
        Position angle, degrees, wrapped to [0, 180).
    ref : :class:`numpy.ndarray`
        Source reference string per object (``'LVD'``, ``'SGA2020'``,
        ``'HYPERLEDA'``, ``'LIT'``/``'SMUDGes'``/``'RC3'``, ``'NONE'``,
        or ``'parent'``).
    mag : :class:`numpy.ndarray`, only if ``get_mag=True``
        Magnitude, defaulting to 18.0 where missing or outside (0, 25].
    band : :class:`numpy.ndarray`, only if ``get_mag=True``
        Band letter corresponding to ``mag``.

    Notes
    -----
    In the early-return (plain ``DIAM``/``BA``/``PA`` columns) path,
    ``if 'MAG' in cat.colnames and 'BAND':`` is checked -- since the
    string literal ``'BAND'`` is always truthy, this condition is
    effectively just ``'MAG' in cat.colnames`` regardless of whether a
    ``'BAND'`` column actually exists, so ``cat['BAND'].value`` can
    raise ``KeyError`` if ``'MAG'`` is present without ``'BAND'``.
    Separately, if ``get_mag=True`` and ``'MAG'`` is *not* in
    ``cat.colnames``, ``mag``/``band`` are never assigned before the
    ``return diam, ba, pa, ref, mag, band`` statement, raising
    ``UnboundLocalError``. Neither case is guarded against.

    """
    # Is this a parent catalog with only one set of geometry
    # measurements? If so, take the values and run!
    if 'DIAM' in cat.colnames and 'BA' in cat.colnames and 'PA' in cat.colnames:
        diam = cat['DIAM'].value * 60.
        ba = cat['BA'].value
        pa = cat['PA'].value
        ref = np.array(['parent'] * len(cat))
        if get_mag:
            if 'MAG' in cat.colnames and 'BAND':
                mag = cat['MAG'].value
                band = cat['BAND'].value
            return diam, ba, pa, ref, mag, band
        else:
            return diam, ba, pa, ref


    nobj = len(cat)
    diam = np.zeros(nobj) - 99.
    ba = np.zeros(nobj) - 99.
    pa = np.zeros(nobj) - 99.
    ref = np.zeros(nobj, '<U9')

    # always prefer LVD because they were all visually determined and
    # inspected
    I = (cat['DIAM_LIT_REF'].value == 'LVD') * (diam == -99.)
    if np.any(I):
        diam[I] = cat['DIAM_LIT'][I] * 60.
        ba[I] = cat['BA_LIT'][I]
        pa[I] = cat['PA_LIT'][I]
        ref[I] = 'LVD'

    # take the largest diameter
    datarefs = np.array(['SGA2020', 'HYPERLEDA', 'LIT'])
    dataindx = np.argmax((cat['DIAM_SGA2020'].value, cat['DIAM_HYPERLEDA'].value, cat['DIAM_LIT'].value), axis=0)

    # first require all of diam, ba, pa...
    for iref, dataref in enumerate(datarefs):
        I = ((dataindx == iref) * (diam == -99.) * (ba == -99.) * (pa == -99.) *
             (cat[f'DIAM_{dataref}'] != -99.) * (cat[f'BA_{dataref}'] != -99.) *
             (cat[f'PA_{dataref}'] != -99.))
        if np.any(I):
            diam[I] = cat[f'DIAM_{dataref}'][I] * 60.
            ba[I] = cat[f'BA_{dataref}'][I]
            pa[I] = cat[f'PA_{dataref}'][I]
            ref[I] = datarefs[iref]
            # special-case LVD, RC3, and SMUDGes
            if dataref == 'LIT':
                J = np.where(cat[f'DIAM_{dataref}_REF'][I] == 'SMUDGes')[0]
                if len(J) > 0:
                    ref[I][J] = 'SMUDGes'
                J = np.where(cat[f'DIAM_{dataref}_REF'][I] == 'LVD')[0]
                if len(J) > 0:
                    ref[I][J] = 'LVD'
                J = np.where(cat[f'DIAM_{dataref}_REF'][I] == 'RC3')[0]
                if len(J) > 0:
                    ref[I][J] = 'RC3'

    # ...and then just diam.
    for iref, dataref in enumerate(datarefs):
        I = (dataindx == iref) * (diam == -99.) * (cat[f'DIAM_{dataref}'] != -99.)
        if np.any(I):
            diam[I] = cat[f'DIAM_{dataref}'][I] * 60.
            ba[I] = cat[f'BA_{dataref}'][I]
            pa[I] = cat[f'PA_{dataref}'][I]
            ref[I] = datarefs[iref]
            # special-case LVD, RC3, and SMUDGes
            if dataref == 'LIT':
                J = np.where(cat[f'DIAM_{dataref}_REF'][I] == 'SMUDGes')[0]
                if len(J) > 0:
                    ref[I][J] = 'SMUDGes'
                J = np.where(cat[f'DIAM_{dataref}_REF'][I] == 'LVD')[0]
                if len(J) > 0:
                    ref[I][J] = 'LVD'
                J = np.where(cat[f'DIAM_{dataref}_REF'][I] == 'RC3')[0]
                if len(J) > 0:
                    ref[I][J] = 'RC3'

    # missing diameters
    I = diam <= 0.
    if np.any(I):
        ref[I] = 'NONE'

    # set a minimum floor on the diameter
    I = diam <= mindiam
    if np.any(I):
        diam[I] = mindiam

    # clean up missing (or crazy) values of BA and PA
    pa[pa < 0.] = 0.
    ba[ba < 0.1] = 0.1 # note!
    ba[ba > 1.] = 1.
    pa = pa % 180

    if get_mag:
        mag = np.zeros(nobj) - 99.
        band = np.zeros(nobj, '<U1')
        for magref in ['SGA2020', 'HYPERLEDA', 'LIT']:
            I = (mag == -99.) * (cat[f'MAG_{magref}'] != -99.)
            #print(magref, np.sum(I))
            if np.any(I):
                mag[I] = cat[f'MAG_{magref}'][I]
                band[I] = cat[f'BAND_{magref}'][I]

        I = (mag < 0.) | (mag > 25.)
        if np.any(I):
            mag[I] = 18.
            #band[I] = ''

    ## return scalars
    #if nobj == 1:
    #    diam = diam[0]
    #    ba = ba[0]
    #    pa = pa[0]
    #    ref = ref[0]

    if get_mag:
        return diam, ba, pa, ref, mag, band
    else:
        return diam, ba, pa, ref
