# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""
LSLGA.desiutil
==============

Module of utilities taken from desiutil
"""

from __future__ import (print_function, absolute_import, division,
                        unicode_literals)

import os

import numpy as np
import numpy.ma

try:
    basestring
except NameError:  # For Python 3
    basestring = str


def plot_slices(x, y, x_lo, x_hi, y_cut, num_slices=5, min_count=100, axis=None,
                set_ylim_from_stats=True, scatter=True):
    """Scatter plot with 68, 95 percentiles superimposed in slices.
    Modified from code written by D. Kirkby

    Requires that the matplotlib package is installed.

    Parameters
    ----------
    x : array of :class:`float`
        X-coordinates to scatter plot.  Points outside [x_lo, x_hi] are
        not displayed.
    y : array of :class:`float`
        Y-coordinates to scatter plot.  Y values are assumed to be roughly
        symmetric about zero.
    x_lo : :class:`float`
        Minimum value of `x` to plot.
    x_hi : :class:`float`
        Maximum value of `x` to plot.
    y_cut : :class:`float`
        The target maximum value of :math:`|y|`.  A dashed line at this value
        is added to the plot, and the vertical axis is clipped at
        :math:`|y|` = 1.25 * `y_cut` (but values outside this range are
        included in the percentile statistics).
    num_slices : :class:`int`, optional
        Number of equally spaced slices to divide the interval [x_lo, x_hi]
        into.
    min_count : :class:`int`, optional
        Do not use slices with fewer points for superimposed percentile
        statistics.
    axis : :class:`matplotlib.axes.Axes`, optional
        Uses the current axis if this is not set.
    set_ylim_from_stats : :class:`bool`, optional
        Set ylim of plot from 95% stat.
    scatter : bool, optional
        Show the data as a scatter plot.
        Best to limit to small datasets

    Returns
    -------
    :class:`matplotlib.axes.Axes`
        The Axes object used in the plot.
    """

    import matplotlib.pyplot as plt

    if axis is None:
        axis = plt.gca()

    x_bins = np.linspace(x_lo, x_hi, num_slices + 1)
    x_i = np.digitize(x, x_bins) - 1
    limits = []
    counts = []
    for s in range(num_slices):
        # Calculate percentile statistics for ok fits.
        y_slice = y[(x_i == s)]
        counts.append(len(y_slice))
        if counts[-1] > 0:
            limits.append(np.percentile(y_slice, (2.5, 16, 50, 84, 97.5)))
        else:
            limits.append((0., 0., 0., 0., 0.))
    limits = np.array(limits)
    counts = np.array(counts)

    # Plot points
    if scatter:
        axis.scatter(x, y, s=15, marker='.', lw=0, color='b', alpha=0.5, zorder=1)

    # Plot quantiles in slices with enough fits.
    stepify = lambda y: np.vstack([y, y]).transpose().flatten()
    y_m2 = stepify(limits[:, 0])
    y_m1 = stepify(limits[:, 1])
    y_med = stepify(limits[:, 2])
    y_p1 = stepify(limits[:, 3])
    y_p2 = stepify(limits[:, 4])
    xstack = stepify(x_bins)[1:-1]
    max_yr, max_p2, min_m2 = 0., -1e9, 1e9
    for i in range(num_slices):
        s = slice(2 * i, 2 * i + 2)
        if counts[i] >= min_count:
            axis.fill_between(
                xstack[s], y_m2[s], y_p2[s], alpha=0.15, color='red', zorder=10)
            axis.fill_between(
                xstack[s], y_m1[s], y_p1[s], alpha=0.25, color='red', zorder=10)
            axis.plot(xstack[s], y_med[s], 'r-', lw=2.)
            # For ylim
            max_yr = max(max_yr, np.max(y_p2[s]-y_m2[s]))
            max_p2 = max(max_p2, np.max(y_p2[s]))
            min_m2 = min(min_m2, np.min(y_m2[s]))

    # xlim
    xmin, xmax = np.min(x), np.max(x)
    axis.set_xlim(np.min(x)-(xmax-xmin)*0.02, np.max(x)+(xmax-xmin)*0.02)

    # ylim
    if set_ylim_from_stats:
        axis.set_ylim(min_m2-max_yr/2., max_p2+max_yr/2.)


    # Plot cut lines.
    axis.axhline(+y_cut, ls=':', color='k')
    axis.axhline(0., ls='-', color='k')
    axis.axhline(-y_cut, ls=':', color='k')

    return axis


class MaskedArrayWithLimits(numpy.ma.MaskedArray):
    """Masked array with additional `vmin`, `vmax` attributes.

    This class accepts the same arguments as
    :class:`~numpy.ma.MaskedArray`.

    This is not a general-purpose subclass and is only intended to simplify
    passing `vmin`, `vmax` limits from :func:`~desiutil.plots.prepare_data` to
    the plotting utility methods defined in this module.

    Parameters
    ----------
    vmin : :class:`float`, optional
        Minimum value when used for clipping or masking.
    vmax : :class:`float`, optional
        Maximum value when used for clipping or masking.

    Attributes
    ----------
    vmin : :class:`float`
        Minimum value when used for clipping or masking.
    vmax : :class:`float`
        Maximum value when used for clipping or masking.
    """
    def __new__(cls, *args, **kwargs):
        obj = super(MaskedArrayWithLimits, cls).__new__(cls, *args, **kwargs)
        if 'vmin' in kwargs:
            obj._optinfo['vmin'] = kwargs['vmin']
        #     obj.vmin = kwargs['vmin']
        # else:
        #     obj.vmin = None
        if 'vmax' in kwargs:
            obj._optinfo['vmax'] = kwargs['vmax']
        #     obj.vmax = kwargs['vmax']
        # else:
        #     obj.vmax = None
        return obj

    @property
    def vmin(self):
        return self._optinfo.get('vmin', None)

    @property
    def vmax(self):
        return self._optinfo.get('vmax', None)


def prepare_data(data, mask=None, clip_lo=None, clip_hi=None,
                 save_limits=False):
    """Prepare array data for color mapping.

    Data is clipped and masked to be suitable for passing to matplotlib
    routines that automatically assign colors based on input values.

    Parameters
    ----------
    data : array or masked array
        Array of data values to assign colors for.
    mask : array of bool or None
        Array of bools with same shape as data, where True values indicate
        values that should be ignored when assigning colors.  When None, the
        mask of a masked array will be used or all values of an unmasked
        array will be used.
    clip_lo : float or str
        Data values below clip_lo will be clipped to the minimum color. If
        clip_lo is a string, it should end with "%" and specify a percentile
        of un-masked data to clip below.
    clip_hi : float or str
        Data values above clip_hi will be clipped to the maximum color. If
        clip_hi is a string, it should end with "%" and specify a percentile
        of un-masked data to clip above.
    save_limits : bool
        Save the calculated lo/hi clip values as attributes vmin, vmax of
        the returned masked array.  Use this flag to indicate that plotting
        functions should use these vmin, vmax values when mapping the
        returned data to colors.

    Returns
    -------
    masked array
        Masked numpy array with the same shape as the input data, with any
        input mask applied (or copied from an input masked array) and values
        clipped to [clip_lo, clip_hi].

    Examples
    --------
    If no optional parameters are specified, the input data is returned
    with only non-finite values masked:

    >>> data = np.arange(5.)
    >>> prepare_data(data)
    masked_array(data = [0.0 1.0 2.0 3.0 4.0],
                 mask = [False False False False False],
           fill_value = 1e+20)
    <BLANKLINE>

    Any mask selection is propagated to the output:

    >>> prepare_data(data, data == 2)
    masked_array(data = [0.0 1.0 -- 3.0 4.0],
                 mask = [False False  True False False],
           fill_value = 1e+20)
    <BLANKLINE>

    Values can be clipped by specifying any combination of percentiles
    (specified as strings ending with "%") and numeric values:

    >>> prepare_data(data, clip_lo='25%', clip_hi=3.5)
    masked_array(data = [1.0 1.0 2.0 3.0 3.5],
                 mask = [False False False False False],
           fill_value = 1e+20)
    <BLANKLINE>

    Clipped values are also masked when the clip value or percentile
    is prefixed with "!":

    >>> prepare_data(data, clip_lo='!25%', clip_hi=3.5)
    masked_array(data = [-- 1.0 2.0 3.0 3.5],
                 mask = [ True False False False False],
           fill_value = 1e+20)
    <BLANKLINE>

    An input masked array is passed through without any copying unless
    clipping is requested:

    >>> masked = numpy.ma.arange(5)
    >>> masked is prepare_data(masked)
    True

    Use the save_limits option to store the clipping limits as vmin, vmax
    attributes of the returned object:

    >>> d = prepare_data(data, clip_lo=1, clip_hi=10, save_limits=True)
    >>> d.vmin, d.vmax
    (1.0, 10.0)

    These attributes can then be used by plotting routines to fix the input
    range used for colormapping, independently of the actual range of data.
    """
    data = np.asanyarray(data)
    if mask is None:
        try:
            # Use the mask associated with a MaskedArray.
            cmask = data.mask
            # If no clipping is requested, pass the input through.
            if clip_lo is None and clip_hi is None:
                return data
        except AttributeError:
            # Nothing is masked by default.
            cmask = np.zeros_like(data, dtype=bool)
    else:
        #
        # Make every effort to ensure that modifying the mask of the output
        # does not modify the input mask.
        #
        cmask = np.array(mask)
        if cmask.shape != data.shape:
            raise ValueError('Invalid mask shape.')
    # Mask any non-finite values.
    cmask |= ~np.isfinite(data)
    unmasked_data = data[~cmask]

    # Convert percentile clip values to absolute values.
    def get_clip(value):
        clip_mask = False
        if isinstance(value, basestring):
            if value.startswith('!'):
                clip_mask = True
                value = value[1:]
            if value.endswith('%'):
                value = np.percentile(unmasked_data, float(value[:-1]))
        return float(value), clip_mask

    if clip_lo is None:
        clip_lo, mask_lo = np.min(unmasked_data), False
    else:
        clip_lo, mask_lo = get_clip(clip_lo)
    if clip_hi is None:
        clip_hi, mask_hi = np.max(unmasked_data), False
    else:
        clip_hi, mask_hi = get_clip(clip_hi)

    if save_limits:
        clipped = MaskedArrayWithLimits(
            np.clip(data, clip_lo, clip_hi), cmask, vmin=clip_lo, vmax=clip_hi)
    else:
        clipped = numpy.ma.MaskedArray(
            np.clip(data, clip_lo, clip_hi), cmask)

    # Mask values outside the clip range, if requested.  The comparisons
    # below might trigger warnings for non-finite data.
    settings = np.seterr(all='ignore')
    if mask_lo:
        clipped.mask[data < clip_lo] = True
    if mask_hi:
        clipped.mask[data > clip_hi] = True
    np.seterr(**settings)

    return clipped


def init_sky(projection='eck4', ra_center=120, galactic_plane_color='red',
             ra_labels=np.arange(0, 360, 60),
             dec_labels=np.arange(-60, 90, 30), ax=None):
    """Initialize a basemap projection of the full sky.

    The returned Basemap object is augmented with an ``ellipse()`` method to
    support drawing ellipses or circles on the sky, which is useful for
    representing DESI tiles.

    Note that the projection uses the geographic convention that RA increases
    from left to right rather than the opposite celestial convention because
    otherwise RA labels are drawn incorrectly (see
    https://github.com/matplotlib/basemap/issues/283 for details).

    The DESI footprint would look better with a projection centered at DEC ~ 15,
    which should be possible with basemap but is not current working (see
    https://github.com/matplotlib/basemap/issues/192).

    Requires that matplotlib and basemap are installed.

    Parameters
    ----------
    projection : :class: `string`, optional
        All-sky projection used for coordinate transformations. The default
        'eck4' is recommended for the reasons given `here
        <http://usersguidetotheuniverse.com/index.php/2011/03/03/
        whats-the-best-map-projection/>`__.  Other good choices are
        kav7' and 'moll'.
    ra_center : float
        Map is centered at this RA in degrees. Default is +120, which
        avoids splitting the DESI northern and southern regions.
    galactic_plane_color : color name or None
        Draw a line representing the galactic plane using the specified
        color, or do nothing when None.
    ra_labels : iterable or None
        List of RA values in degrees that will be labeled on the map.
        If None, there will be no RA labels or grid lines.
    dec_labels : iterable or None
        List of DEC values in degrees that will be labeled on the map.
        If None, there will be no DEC labels or grid lines.
    ax : matplotlib axis objet or None
        Axes to use for drawing this map, or use current axes if None.

    Returns
    -------
    :class:`mpl_toolkits.basemap.Basemap`
       The Basemap object created for this plot, which can be used for
       additional projection and plotting operations.
    """
    import matplotlib
    if 'TRAVIS_JOB_ID' in os.environ:
        matplotlib.use('agg')
    from matplotlib.patches import Polygon
    from mpl_toolkits.basemap import pyproj
    from mpl_toolkits.basemap import Basemap
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    # Define a Basemap subclass with an ellipse() method.
    class BasemapWithEllipse(Basemap):
        """Code from http://stackoverflow.com/questions/8161144/
        drawing-ellipses-on-matplotlib-basemap-projections
        It adds ellipses to the class Basemap.
        """
        def ellipse(self, x0, y0, a, b, n, ax=None, **kwargs):
            """Extension to Basemap class from `basemap` to draw ellipses.

            Parameters
            ----------
            x0 : :class: `float`
                Centroid of the ellipse in the X axis.
            y0 : :class: `float`
                Centroid of the ellipse in the Y axis.
            a : :class: `float`
                Semi-major axis of the ellipse.
            b : :class: `float`
                Semi-minor axis of the ellipse.
            n : :class: `int`
                Number of points to draw the ellipse.

            Returns
            -------
            :class: `Basemap`
                It returns one Basemap ellipse at a time.
            """
            ax = kwargs.pop('ax', None) or self._check_ax()
            g = pyproj.Geod(a=self.rmajor, b=self.rminor)
            azf, azb, dist = g.inv([x0, x0], [y0, y0], [x0+a, x0], [y0, y0+b])
            tsid = dist[0] * dist[1]  # a * b
            seg = [self(x0+a, y0)]
            AZ = np.linspace(azf[0], 360. + azf[0], n)
            for i, az in enumerate(AZ):
                # Skips segments along equator (Geod can't handle equatorial arcs).
                if np.allclose(0., y0) and (np.allclose(90., az) or
                                            np.allclose(270., az)):
                    continue

                # In polar coordinates, with the origin at the center of the
                # ellipse and with the angular coordinate ``az`` measured from the
                # major axis, the ellipse's equation  is [1]:
                #
                #                           a * b
                # r(az) = ------------------------------------------
                #         ((b * cos(az))**2 + (a * sin(az))**2)**0.5
                #
                # Azymuth angle in radial coordinates and corrected for reference
                # angle.
                azr = 2. * np.pi / 360. * (az + 90.)
                A = dist[0] * np.sin(azr)
                B = dist[1] * np.cos(azr)
                r = tsid / (B**2. + A**2.)**0.5
                lon, lat, azb = g.fwd(x0, y0, az, r)
                x, y = self(lon, lat)

                # Add segment if it is in the map projection region.
                if x < 1e20 and y < 1e20:
                    seg.append((x, y))

            poly = Polygon(seg, **kwargs)
            ax.add_patch(poly)

            # Set axes limits to fit map region.
            self.set_axes_limits(ax=ax)

            return poly

    # Create an instance of our custom Basemap.
    m = BasemapWithEllipse(
        projection=projection, lon_0=ra_center, resolution=None,
        celestial=False, ax=ax)
    if ra_labels is not None:
        if projection in ('hammer', 'moll'):
            labels = [0, 0, 0, 0]
        else:
            labels = [0, 0, 1, 0]
        m.drawmeridians(
            ra_labels, labels=labels, labelstyle='+/-')
    if dec_labels is not None:
        m.drawparallels(
            dec_labels, labels=[1, 1, 0, 0], labelstyle='+/-')
    m.drawmapboundary()

    # Draw the optional galactic plane.
    if galactic_plane_color is not None:
        # Generate coordinates of a line in galactic coordinates and convert
        # to equatorial coordinates.
        galactic_l = np.linspace(0, 2 * np.pi, 1000)
        galactic_plane = SkyCoord(l=galactic_l*u.radian,
                                  b=np.zeros_like(galactic_l)*u.radian,
                                  frame='galactic').fk5
        # Project to map coordinates and display.  Use a scatter plot to
        # avoid wrap-around complications.
        galactic_x, galactic_y = m(galactic_plane.ra.degree,
                                   galactic_plane.dec.degree)

        paths = m.scatter(
            galactic_x, galactic_y, marker='.', s=10, lw=0, alpha=0.5,
            c=galactic_plane_color)
        # Make sure the galactic plane stays above other displayed objects.
        paths.set_zorder(20)

    return m


def plot_healpix_map(data, nest=False, cmap='viridis', colorbar=True,
                     label=None, basemap=None):
    """Plot a healpix map using an all-sky projection.

    Pass the data array through :func:`prepare_data` to select a subset to plot
    and clip the color map to specified values or percentiles.

    This function is similar to :func:`plot_grid_map` but is generally slower
    at high resolution and has less elegant handling of pixels that wrap around
    in RA, which are not drawn.

    Requires that matplotlib, basemap, and healpy are installed.

    Parameters
    ----------
    data : array or masked array
        1D array of data associated with each healpix.  Must have a size that
        exactly matches the number of pixels for some NSIDE value. Use the
        output of :func:`prepare_data` as a convenient way to specify
        data cuts and color map clipping.
    nest : bool
        If True, assume NESTED pixel ordering.  Otheriwse, assume RING pixel
        ordering.
    cmap : colormap name or object
        Matplotlib colormap to use for mapping data values to colors.
    colorbar : bool
        Draw a colorbar below the map when True.
    label : str or None
        Label to display under the colorbar.  Ignored unless colorbar is True.
    basemap : Basemap object or None
        Use the specified basemap or create a default basemap using
        :func:`init_sky` when None.

    Returns
    -------
    basemap
        The basemap used for the plot, which will match the input basemap
        provided, or be a newly created basemap if None was provided.
    """
    import healpy as hp
    import matplotlib.pyplot as plt
    import matplotlib.colors
    from matplotlib.collections import PolyCollection

    data = prepare_data(data)
    if len(data.shape) != 1:
        raise ValueError('Invalid data array, should be 1D.')
    nside = hp.npix2nside(len(data))

    if basemap is None:
        basemap = init_sky()

    # Get pixel boundaries as quadrilaterals.
    corners = hp.boundaries(nside, np.arange(len(data)), step=1, nest=nest)
    corner_theta, corner_phi = hp.vec2ang(corners.transpose(0, 2, 1))
    corner_ra, corner_dec = (np.degrees(corner_phi),
                             np.degrees(np.pi/2-corner_theta))
    # Convert sky coords to map coords.
    x, y = basemap(corner_ra, corner_dec)
    # Regroup into pixel corners.
    verts = np.array([x.reshape(-1, 4), y.reshape(-1, 4)]).transpose(1, 2, 0)

    # Find and mask any pixels that wrap around in RA.
    uv_verts = np.array([corner_phi.reshape(-1, 4),
                         corner_theta.reshape(-1, 4)]).transpose(1, 2, 0)
    theta_edge = np.unique(uv_verts[:, :, 1])
    phi_edge = np.radians(basemap.lonmax)
    eps = 0.1 * np.sqrt(hp.nside2pixarea(nside))
    wrapped1 = hp.ang2pix(nside, theta_edge, phi_edge - eps, nest=nest)
    wrapped2 = hp.ang2pix(nside, theta_edge, phi_edge + eps, nest=nest)
    wrapped = np.unique(np.hstack((wrapped1, wrapped2)))
    data.mask[wrapped] = True

    # Normalize the data using its vmin, vmax attributes, if present.
    try:
        norm = matplotlib.colors.Normalize(vmin=data.vmin, vmax=data.vmax)
    except AttributeError:
        norm = None

    # Make the collection and add it to the plot.
    collection = PolyCollection(
        verts, array=data, cmap=cmap, norm=norm, edgecolors='none')

    axes = plt.gca() if basemap.ax is None else basemap.ax
    axes.add_collection(collection)
    axes.autoscale_view()

    if colorbar:
        bar = plt.colorbar(
            collection, ax=basemap.ax, orientation='horizontal',
            spacing='proportional', pad=0.01, aspect=50)
        if label:
            bar.set_label(label)

    return basemap


def plot_grid_map(data, ra_edges, dec_edges, cmap='viridis', colorbar=True,
                  label=None, basemap=None):
    """Plot an array of 2D values using an all-sky projection.

    Pass the data array through :func:`prepare_data` to select a subset to plot
    and clip the color map to specified values or percentiles.

    This function is similar to :func:`plot_healpix_map` but is generally faster
    and has better handling of RA wrap around artifacts.

    Requires that matplotlib and basemap are installed.

    Parameters
    ----------
    data : array or masked array
        2D array of data associated with each grid cell, with shape
        (n_ra, n_dec). Use the output of :func:`prepare_data` as a convenient
        way to specify data cuts and color map clipping.
    ra_edges : array
        1D array of n_ra+1 RA grid edge values in degrees, which must span the
        full circle, i.e., ra_edges[0] == ra_edges[-1] - 360. The RA grid
        does not need to match the edges of the basemap projection, in which
        case any wrap-around cells will be duplicated on both edges.
    dec_edges : array
        1D array of n_dec+1 DEC grid edge values in degrees.  Values are not
        required to span the full range [-90, +90].
    cmap : colormap name or object
        Matplotlib colormap to use for mapping data values to colors.
    colorbar : bool
        Draw a colorbar below the map when True.
    label : str or None
        Label to display under the colorbar.  Ignored unless colorbar is True.
    basemap : Basemap object or None
        Use the specified basemap or create a default basemap using
        :func:`init_sky` when None.

    Returns
    -------
    basemap
        The basemap used for the plot, which will match the input basemap
        provided, or be a newly created basemap if None was provided.
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors

    data = prepare_data(data)
    if len(data.shape) != 2:
        raise ValueError('Expected 2D data array.')
    n_dec, n_ra = data.shape

    # Normalize the data using its vmin, vmax attributes, if present.
    # Need to do this before hstack-ing below, which drops vmin, vmax.
    try:
        norm = matplotlib.colors.Normalize(vmin=data.vmin, vmax=data.vmax)
    except AttributeError:
        norm = None

    # Silently flatten, sort, and remove duplicates from the edges arrays.
    ra_edges = np.unique(ra_edges)
    dec_edges = np.unique(dec_edges)
    if len(ra_edges) != n_ra + 1:
        raise ValueError('Invalid ra_edges.')
    if len(dec_edges) != n_dec + 1:
        raise ValueError('Invalid dec_edges.')

    if ra_edges[0] != ra_edges[-1] - 360:
        raise ValueError('Invalid ra_edges, do not span 360 degrees.')

    if basemap is None:
        basemap = init_sky()

    if basemap.lonmin + 360 != basemap.lonmax:
        raise RuntimeError('Can only handle all-sky projections for now.')

    # Shift RA gridlines so they overlap the map's left-edge RA.
    while ra_edges[0] > basemap.lonmin:
        ra_edges -= 360
    while ra_edges[0] <= basemap.lonmin - 360:
        ra_edges += 360

    # Find the first RA gridline that fits within the map's left edge.
    first = np.where(ra_edges >= basemap.lonmin)[0][0]

    if first > 0:
        # Wrap the data beyond the left edge around to the right edge.
        # Remember to use numpy.ma.hstack for the data to preserve the mask.
        if ra_edges[first] > basemap.lonmin:
            # Split a wrap-around column into separate left and right columns.
            ra_edges = np.hstack(([basemap.lonmin], ra_edges[first:],
                                  ra_edges[:first] + 360, [basemap.lonmax]))
            data = numpy.ma.hstack(
                (data[:, first - 1:first], data[:, first:],
                 data[:, :first], data[:, first:first + 1]))
        else:
            ra_edges = np.hstack((ra_edges[first:], ra_edges[:first + 1] + 360))
            data = numpy.ma.hstack((data[:, first:], data[:, :first + 1]))

    # Build a 2D array of grid line intersections.
    grid_ra, grid_dec = np.meshgrid(ra_edges, dec_edges)

    mesh = basemap.pcolormesh(
        grid_ra, grid_dec, data, cmap=cmap, norm=norm,
        edgecolor='none', lw=0, latlon=True)

    if colorbar:
        bar = plt.colorbar(
            mesh, ax=basemap.ax, orientation='horizontal',
            spacing='proportional', pad=0.01, aspect=50)
        if label:
            bar.set_label(label)

    return basemap


def plot_sky_circles(ra_center, dec_center, field_of_view=3.2, data=None,
                     cmap='viridis', facecolors='skyblue', edgecolor='none',
                     colorbar=True, colorbar_ticks=None, label=None,
                     basemap=None):
    """Plot circles on an all-sky projection.

    Pass the optional data array through :func:`prepare_data` to select a
    subset to plot and clip the color map to specified values or percentiles.

    Requires that matplotlib and basemap are installed.

    Parameters
    ----------
    ra_center : array
        1D array of RA in degrees at the centers of each circle to plot.
    dec_center : array
        1D array of DEC in degrees at the centers of each circle to plot.
    field_of_view : array
        Full sky openning angle in degrees of the circles to plot. The default
        is appropriate for a DESI tile.
    data : array or None
        1D array of data associated with each circle, used to set its facecolor.
    cmap : colormap name or object
        Matplotlib colormap to use for mapping data values to colors. Ignored
        unless data is specified.
    facecolors : matplotlib color or array of colors
        Ignored when data is specified. An array must have one entry per circle
        or a single value is used for all circles.
    edgecolor : matplotlib color
        The edge color used for all circles.  Use 'none' to hide edges.
    colorbar : bool
        Draw a colorbar below the map when True and data is provided.
    colorbar_ticks : list or None
        Use the specified colorbar ticks or determine them automatically
        when None.
    label : str or None
        Label to display under the colorbar.  Ignored unless a colorbar is
        displayed.
    basemap : BasemapWithEllipse or None
        An instance of the BasemapWithEllipse class, normally obtained by
        calling :func:`init_sky`.  Create a default basemap when None.

    Returns
    -------
    basemap
        The basemap used for the plot, which will match the input basemap
        provided, or be a newly created basemap if None was provided.
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors
    import matplotlib.cm

    ra_center = np.asarray(ra_center)
    dec_center = np.asarray(dec_center)
    if len(ra_center.shape) != 1:
        raise ValueError('Invalid ra_center, must be a 1D array.')
    if len(dec_center.shape) != 1:
        raise ValueError('Invalid dec_center, must be a 1D array.')
    if len(ra_center) != len(dec_center):
        raise ValueError('Arrays ra_center, dec_center must have same size.')

    if data is not None:
        data = prepare_data(data)
        # Facecolors are determined by the data, when specified.
        if data.shape != ra_center.shape:
            raise ValueError('Invalid data shape, must match ra_center.')
        # Colors associated with masked values in data will be ignored later.
        try:
            # Normalize the data using its vmin, vmax attributes, if present.
            norm = matplotlib.colors.Normalize(vmin=data.vmin, vmax=data.vmax)
        except AttributeError:
            # Otherwise use the data limits.
            norm = matplotlib.colors.Normalize(vmin=data.min(), vmax=data.max())
        cmapper = matplotlib.cm.ScalarMappable(norm, cmap)
        facecolors = cmapper.to_rgba(data)
    else:
        colorbar = False
        # Try to repeat a single fixed color for all circles.
        try:
            facecolors = np.tile(
                [matplotlib.colors.colorConverter.to_rgba(facecolors)],
                (len(ra_center), 1))
        except ValueError:
            # Assume that facecolor is already an array.
            facecolors = np.asarray(facecolors)

    if len(facecolors) != len(ra_center):
        raise ValueError('Invalid facecolor array.')

    if basemap is None:
        basemap = init_sky()

    if basemap.lonmin + 360 != basemap.lonmax:
        raise RuntimeError('Can only handle all-sky projections for now.')

    if len(ra_center) == 0:
        return

    # Convert field-of-view angle into dDEC, dRA.
    dDEC = 0.5 * field_of_view
    dRA = dDEC / np.cos(np.radians(dec_center))

    # Identify circles that wrap around the map edges in RA.
    edge_dist = np.fmod(ra_center - basemap.lonmin, 360)
    wrapped = np.minimum(edge_dist, 360 - edge_dist) < 1.05 * dRA

    # Set the number of vertices for approximating the ellipse based
    # on the field of view.
    n_pt = max(8, int(np.ceil(field_of_view)))

    # Loop over non-wrapped circles.
    for ra, dec, dra, fc in zip(ra_center[~wrapped], dec_center[~wrapped],
                                dRA[~wrapped], facecolors[~wrapped]):
        basemap.ellipse(ra, dec, dra, dDEC, n_pt, facecolor=fc,
                        edgecolor=edgecolor)

    if colorbar:
        mappable = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
        mappable.set_array(data)
        bar = plt.colorbar(
            mappable, ax=basemap.ax, orientation='horizontal',
            spacing='proportional', pad=0.01, aspect=50,
            ticks=colorbar_ticks)
        if label:
            bar.set_label(label)

    return basemap


def plot_sky_binned(ra, dec, weights=None, data=None, plot_type='grid',
                    max_bin_area=5, clip_lo=None, clip_hi=None, verbose=False,
                    cmap='viridis', colorbar=True, label=None, basemap=None,
                    return_grid_data=False):
    """Show objects on the sky using a binned plot.

    Bin values either show object counts per unit sky area or, if an array
    of associated data values is provided, mean data values within each bin.
    Objects can have associated weights.

    Requires that matplotlib and basemap are installed. When plot_type is
    "healpix", healpy must also be installed.

    Parameters
    ----------
    ra : array
        Array of object RA values in degrees. Must have the same shape as
        dec and will be flattened if necessary.
    dec : array
        Array of object DEC values in degrees. Must have the same shape as
        ra and will be flattened if necessary.
    weights : array or None
        Optional of weights associated with each object.  All objects are
        assumed to have equal weight when this is None.
    data : array or None
        Optional array of scalar values associated with each object. The
        resulting plot shows the mean data value per bin when data is
        specified.  Otherwise, the plot shows counts per unit sky area.
    plot_type : str
        Must be either 'grid' or 'healpix', and selects whether data in
        binned in healpix or in (sin(DEC), RA).
    max_bin_area : float
        The bin size will be chosen automatically to be as close as
        possible to this value but not exceeding it.
    clip_lo : float or str
        Clipping is applied to the plot data calculated as counts / area
        or the mean data value per bin. See :func:`prepare_data` for
        details.
    clip_hi : float or str
        Clipping is applied to the plot data calculated as counts / area
        or the mean data value per bin. See :func:`prepare_data` for
        details.
    verbose : bool
        Print information about the automatic bin size calculation.
    cmap : colormap name or object
        Matplotlib colormap to use for mapping data values to colors.
    colorbar : bool
        Draw a colorbar below the map when True.
    label : str or None
        Label to display under the colorbar.  Ignored unless colorbar is True.
    basemap : Basemap object or None
        Use the specified basemap or create a default basemap using
        :func:`init_sky` when None.
    return_grid_data : bool
        If true, return (basemap, grid_data) instead of just basemap

    Returns
    -------
    basemap or (basemap, grid_data)
        The basemap used for the plot, which will match the input basemap
        provided, or be a newly created basemap if None was provided.
        The grid_data plotted is also returned if return_grid_data is True.
    """
    ra = np.asarray(ra).reshape(-1)
    dec = np.asarray(dec).reshape(-1)
    if len(ra) != len(dec):
        raise ValueError('Arrays ra,dec must have same size.')

    plot_types = ('grid', 'healpix',)
    if plot_type not in plot_types:
        raise ValueError(
            'Invalid plot_type, should be one of {0}.'
            .format(', '.join(plot_types)))

    if data is not None and weights is None:
        weights = np.ones_like(data)

    if plot_type == 'grid':
        # Convert the maximum pixel area to steradians.
        max_bin_area = max_bin_area * (np.pi / 180.) ** 2

        # Pick the number of bins in cos(DEC) and RA to use.
        n_cos_dec = int(np.ceil(2 / np.sqrt(max_bin_area)))
        n_ra = int(np.ceil(4 * np.pi / max_bin_area / n_cos_dec))
        # Calculate the actual pixel area in sq. degrees.
        bin_area = 360 ** 2 / np.pi / (n_cos_dec * n_ra)
        if verbose:
            print(
                'Using {0} x {1} grid in cos(DEC) x RA'.format(n_cos_dec, n_ra),
                'with pixel area {:.3f} sq.deg.'.format(bin_area))

        # Calculate the bin edges in degrees.
        ra_edges = np.linspace(-180., +180., n_ra + 1)
        dec_edges = np.degrees(np.arcsin(np.linspace(-1., +1., n_cos_dec + 1)))

        # Put RA values in the range [-180, 180).
        ra = np.fmod(ra, 360.)
        ra[ra >= 180.] -= 360.

        # Histogram the input coordinates.
        counts, _, _ = np.histogram2d(
            dec, ra, [dec_edges, ra_edges], weights=weights)

        if data is None:
            grid_data = counts / bin_area
        else:
            sums, _, _ = np.histogram2d(
                dec, ra, [dec_edges, ra_edges], weights=weights * data)
            # This ratio might result in some nan (0/0) or inf (1/0) values,
            # but these will be masked by prepare_data().
            settings = np.seterr(all='ignore')
            grid_data = sums / counts
            np.seterr(**settings)

        grid_data = prepare_data(
            grid_data, clip_lo=clip_lo, clip_hi=clip_hi)

        basemap = plot_grid_map(
            grid_data, ra_edges, dec_edges, cmap, colorbar, label, basemap)

    elif plot_type == 'healpix':

        import healpy as hp

        for n in range(1, 25):
            nside = 2 ** n
            bin_area = hp.nside2pixarea(nside, degrees=True)
            if bin_area <= max_bin_area:
                break
        npix = hp.nside2npix(nside)
        nest = False
        if verbose:
            print(
                'Using healpix map with NSIDE={0}'.format(nside),
                'and pixel area {:.3f} sq.deg.'.format(bin_area))

        pixels = hp.ang2pix(nside, np.radians(90 - dec), np.radians(ra), nest)
        counts = np.bincount(pixels, weights=weights, minlength=npix)
        if data is None:
            grid_data = counts / bin_area
        else:
            sums = np.bincount(pixels, weights=weights * data, minlength=npix)
            grid_data = np.zeros_like(sums, dtype=float)
            nonzero = counts > 0
            grid_data[nonzero] = sums[nonzero] / counts[nonzero]

        grid_data = prepare_data(grid_data, clip_lo=clip_lo, clip_hi=clip_hi)

        basemap = plot_healpix_map(
            grid_data, nest, cmap, colorbar, label, basemap)

    if return_grid_data:
        return basemap, grid_data
    else:
        return basemap
