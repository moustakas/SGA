"""
SGA.qa
======

Code to do produce various QA (quality assurance) plots.

"""
import os, pdb
import warnings
import time, subprocess
from importlib import resources
import numpy as np
from astropy.table import Table

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl
import matplotlib.ticker as ticker


#sns, _ = plot_style()

fonttype = resources.files('SGA').joinpath('data/Georgia.ttf')
prop = mpl.font_manager.FontProperties(fname=fonttype, size=12)

# color-blind friendly color cycle:
# https://twitter.com/rachel_kurchin/status/1229567059694170115
cb_colors = {'blue': '#377eb8',
             'orange': '#ff7f00',
             'green': '#4daf4a',
             'pink': '#f781bf',
             'brown': '#a65628',
             'purple': '#984ea3',
             'gray': '#999999',
             'red': '#e41a1c',
             'yellow': '#dede00'}

cols = ['OBJNAME', 'OBJNAME_NED', 'OBJNAME_HYPERLEDA', 'MORPH', 'DIAM_LIT', 'DIAM_HYPERLEDA',
        'OBJTYPE', 'RA', 'DEC', 'RA_NED', 'DEC_NED', 'RA_HYPERLEDA', 'DEC_HYPERLEDA',
        'MAG_LIT', 'Z', 'PGC', 'ROW_PARENT']


def to_skyviewer_table(cat, diamcol='DIAM'):
    """Convert an SGA catalog into the minimal column set expected by
    the sky-viewer tool, assigning each object a color that cycles
    spatially (on an ~arcmin scale) so nearby groups/objects are
    visually distinguishable.

    Parameters
    ----------
    cat : :class:`astropy.table.Table`
        Input catalog. Must contain ``OBJNAME``, ``RA``, ``DEC``,
        `diamcol`, ``BA``, ``PA``; if ``GROUP_RA``/``GROUP_DEC`` are
        present, those are used for the spatial color-binning instead
        of ``RA``/``DEC`` (see Notes).
    diamcol : :class:`str`
        Name of the diameter column (in arcmin) to convert to a
        viewer radius (arcsec).

    Returns
    -------
    :class:`astropy.table.Table`
        Table with columns ``name``, ``ra``, ``dec``, ``radius``
        (arcsec), ``abRatio``, ``posAngle``, ``color``.

    Notes
    -----
    When ``GROUP_RA``/``GROUP_DEC`` are present, every member of a
    group is binned (and thus colored) using the *group* center
    rather than its own individual position, so all members of one
    group share a color even though the output table still reports
    each object's individual ``ra``/``dec``.

    """
    # Larger color palette (20-24 colors)
    colors = [
        'red', 'blue', 'green', 'yellow', 'cyan', 'orange', 'magenta', 'lime',
        'pink', 'purple', 'brown', 'navy', 'teal', 'olive', 'maroon', 'aqua',
        'coral', 'gold', 'indigo', 'violet', 'khaki', 'salmon', 'plum', 'tan'
    ]

    # Bin on arcmin scale (~36 arcsec ~ 0.01 deg)
    if 'GROUP_RA' in cat.colnames:
        ra_bin = np.floor(cat['GROUP_RA'] * 100).astype(int) % len(colors)
        dec_bin = np.floor(cat['GROUP_DEC'] * 100).astype(int) % len(colors)
    else:
        ra_bin = np.floor(cat['RA'] * 100).astype(int) % len(colors)
        dec_bin = np.floor(cat['DEC'] * 100).astype(int) % len(colors)

    color_idx = (ra_bin + dec_bin) % len(colors)
    color_array = np.array([colors[i] for i in color_idx])

    # Build output table
    out = Table()
    out['name'] = cat['OBJNAME']
    out['ra'] = cat['RA']
    out['dec'] = cat['DEC']
    out['radius'] = (cat[diamcol] * 60.0 / 2.0).astype(np.float32)
    out['abRatio'] = cat['BA']
    out['posAngle'] = cat['PA']
    out['color'] = color_array

    return out


def plot_style(font_scale=1.2, paper=False, talk=True):
    """Configure global seaborn/matplotlib plotting style (font, style
    context, color palette) for figure generation.

    Parameters
    ----------
    font_scale : :class:`float`
        Scaling factor applied to all font sizes.
    paper : :class:`bool`
        If True, use a "paper" context (smaller fonts, ``deep``
        palette, ``text.usetex`` disabled). Overridden by `talk` if
        both are True (see Notes).
    talk : :class:`bool`
        If True, use a "talk" context (larger fonts, ``deep``
        palette). Defaults to True, so `paper` has no effect unless
        `talk` is explicitly set to False (see Notes).

    Returns
    -------
    sns : module
        The configured :mod:`seaborn` module (for convenience, so
        callers don't need a separate import).
    colors : :class:`list`
        The current seaborn color palette, as a list of RGB tuples.

    Notes
    -----
    `paper` and `talk` are not mutually exclusive in the code -- if
    both are True, the `talk` block runs second and silently
    overrides every setting `paper` made (both set `context`/`palette`
    to overlapping but not identical values). Since `talk` defaults to
    True, passing only `paper=True` (without also passing
    `talk=False`) silently has no effect.

    """
    import seaborn as sns
    rc = {'font.family': 'serif'}#, 'text.usetex': True}
    #rc = {'font.family': 'serif', 'text.usetex': True,
    #       'text.latex.preamble': r'\boldmath'})
    palette, context = 'Set2', 'talk'

    if paper:
        context = 'paper'
        palette = 'deep'
        rc.update({'text.usetex': False})

    if talk:
        context = 'talk'
        palette = 'deep'
        #rc.update({'text.usetex': True})

    sns.set(context=context, style='ticks', font_scale=font_scale, rc=rc)
    sns.set_palette(palette, 12)

    colors = sns.color_palette()
    #sns.reset_orig()

    return sns, colors


def notebook_style():
    """Apply a clean set of :data:`matplotlib.rcParams` (figure size,
    tick direction, font family, DPI) suited for inline display in
    Jupyter notebooks.

    Returns
    -------
    None

    """
    plt.rcParams.update({
        'figure.figsize':  [8., 6.],
        'axes.labelsize':  14,
        'axes.titlesize':  11,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.top':       True,
        'ytick.right':     True,
        'font.family':     'serif',
        'figure.dpi':      150,
    })


def sdss_rgb(imgs, bands, scales=None, m=0.03, Q=20, mnmx=None, clip=True):
    """Convert a list of band images to an RGB array using an arcsinh stretch.

    Default scaling matches the Legacy Survey viewer::

        g -> Blue  (plane 2, scale 6.0)
        r -> Green (plane 1, scale 3.4)
        i -> Red   (plane 0, scale 3.0)
        z -> Red   (plane 0, scale 2.2)

    For ``['g', 'r', 'i', 'z']`` a band-mixing scheme is used so that
    all four bands contribute to all three color channels. For any
    other combination (e.g. ``['g', 'r', 'z']``) each band maps to a
    single plane.

    Parameters
    ----------
    imgs : :class:`list` of :class:`numpy.ndarray`
        Per-band 2-D image arrays, in the same order as `bands`.
    bands : :class:`list` of :class:`str`
        Band names, from ``{'g', 'r', 'i', 'z'}``.
    scales : :class:`dict`, optional
        Override per-band ``(plane, scale)`` tuples.
    m : :class:`float`
        Additive offset applied before the arcsinh stretch.
    Q : :class:`float` or None
        Arcsinh softening parameter. Set to None to skip the stretch
        entirely (linear scaling only).
    mnmx : :class:`tuple`, optional
        Manual ``(min, max)`` linear clip range, used instead of the
        arcsinh/percentile-style normalization.
    clip : :class:`bool`
        If True, clip the output to ``[0, 1]``.

    Returns
    -------
    :class:`numpy.ndarray`
        ``(H, W, 3)`` float32 RGB array.

    """
    _scales = dict(g=(2, 6.0), r=(1, 3.4), i=(0, 3.0), z=(0, 2.2))
    if scales is not None:
        _scales.update(scales)

    I = 0
    for img, band in zip(imgs, bands):
        _, scale = _scales[band]
        I = I + np.maximum(0, img * scale + m)
    I /= len(bands)
    if Q is not None:
        fI = np.arcsinh(Q * I) / np.sqrt(Q)
        I += (I == 0.) * 1e-6
        I = fI / I

    H, W = I.shape
    rgb = np.zeros((H, W, 3), np.float32)

    if list(bands) == ['g', 'r', 'i', 'z']:
        rgbvec = dict(
            g=(0.,   0.,  0.75),
            r=(0.,   0.5, 0.25),
            i=(0.25, 0.5, 0.),
            z=(0.75, 0.,  0.))
        for img, band in zip(imgs, bands):
            _, scale = _scales[band]
            rf, gf, bf = rgbvec[band]
            if mnmx is None:
                v = (img * scale + m) * I
            else:
                v = ((img * scale + m) - mnmx[0]) / (mnmx[1] - mnmx[0])
            if clip:
                v = np.clip(v, 0, 1)
            rgb[:, :, 0] += rf * v
            rgb[:, :, 1] += gf * v
            rgb[:, :, 2] += bf * v
    else:
        for img, band in zip(imgs, bands):
            plane, scale = _scales[band]
            if mnmx is None:
                imgplane = (img * scale + m) * I
            else:
                imgplane = ((img * scale + m) - mnmx[0]) / (mnmx[1] - mnmx[0])
            if clip:
                imgplane = np.clip(imgplane, 0, 1)
            rgb[:, :, plane] = imgplane

    return rgb


def sbprofile_colors():
    """Return the standard per-band color mapping used for surface
    brightness profile plots (optical grz/i + GALEX FUV/NUV + unWISE
    W1-W4).

    Returns
    -------
    :class:`dict`
        Mapping of band name (``'g'``, ``'r'``, ``'i'``, ``'z'``,
        ``'FUV'``, ``'NUV'``, ``'W1'``-``'W4'``) to an RGB color
        tuple.

    """
    import seaborn as sns

    optcolors = sns.color_palette('Set1', n_colors=14, desat=0.75)
    morecolors = sns.color_palette('Set2', n_colors=8, desat=0.75)

    sbcolors = {
        'g': optcolors[1], 'r': optcolors[2], 'i': optcolors[4], 'z': optcolors[0],
        'FUV': morecolors[0], 'NUV': morecolors[1],
        'W1': morecolors[2], 'W2': morecolors[3], 'W3': morecolors[6], 'W4': morecolors[7],
    }

    return sbcolors


def get_norm(img, a=0.9, contrast=0.25, percentile=95.,
             n_samples=1000):
    """Build an asinh-stretch, percentile-clipped image normalization
    for display.

    Parameters
    ----------
    img : :class:`numpy.ndarray`
        Image array to normalize (used only to set the value range;
        not modified).
    a : :class:`float`
        Asinh stretch softening parameter, passed to
        :class:`astropy.visualization.AsinhStretch`.
    contrast : :class:`float`
        Unused (see Notes).
    percentile : :class:`float`
        Percentile interval (e.g. 95 -> [2.5, 97.5]) used to set the
        normalization's clip limits, via
        :class:`astropy.visualization.PercentileInterval`.
    n_samples : :class:`int`
        Number of samples used by
        :class:`astropy.visualization.PercentileInterval` to estimate
        the percentile range.

    Returns
    -------
    :class:`astropy.visualization.ImageNormalize`
        Normalization object combining the percentile interval and
        asinh stretch, suitable for passing as ``norm=`` to
        :func:`matplotlib.pyplot.imshow`.

    Notes
    -----
    `contrast` is accepted but never used in the live code path -- it
    was only referenced in a now-commented-out
    :class:`astropy.visualization.ZScaleInterval` alternative
    (``interval = ZScaleInterval(contrast=contrast, ...)``), which has
    been replaced by :class:`astropy.visualization.PercentileInterval`
    but the unused parameter was never removed.

    """
    #from astropy.visualization import simple_norm
    from astropy.visualization import (AsinhStretch, ImageNormalize,
                                       PercentileInterval)
    #from astropy.visualization import ZScaleInterval

    stretch = AsinhStretch(a=a)
    interval = PercentileInterval(percentile, n_samples=n_samples)
    #interval = ZScaleInterval(contrast=contrast, n_samples=n_samples)

    #norm = simple_norm(img, stretch=stretch, percent=percent, asinh_a=asinh_a)
    #norm = ImageNormalize(img, interval=ZScaleInterval(), stretch=AsinhStretch(a=0.5))
    norm = ImageNormalize(img, interval=interval, stretch=stretch)
    return norm


def matched_norm(data, model, a=0.9, percentile=95.):
    """Build a shared asinh-stretch normalization from the combined
    value range of a data image and its model, so both can be
    displayed on the same color scale for direct comparison.

    Parameters
    ----------
    data : :class:`numpy.ma.MaskedArray`
        Data image; must be a masked array (its mask is used to
        exclude invalid/missing pixels from the percentile
        calculation -- see Notes).
    model : :class:`numpy.ndarray`
        Model image (not required to be masked).
    a : :class:`float`
        Asinh stretch softening parameter.
    percentile : :class:`float`
        Percentile interval used to set the clip limits, computed
        jointly over the unmasked `data` values and all of `model`.

    Returns
    -------
    S : :class:`astropy.visualization.AsinhStretch`
        The stretch function itself (callable), returned separately
        so the caller can apply the identical stretch to other
        arrays (e.g. both `data` and `model` individually).
    norm : :class:`matplotlib.colors.Normalize`
        Linear normalization over the *stretched* value range, to be
        applied after calling `S` on an array.

    Notes
    -----
    `data` is required to be a :class:`numpy.ma.MaskedArray` --
    ``data.mask`` is accessed unconditionally, so passing a plain
    :class:`numpy.ndarray` raises ``AttributeError``. `model` has no
    such requirement and is used in full (unmasked).

    """
    import matplotlib.colors as mcolors
    from astropy.visualization import AsinhStretch, PercentileInterval

    mask = data.mask

    d = np.asarray(data[~mask]).ravel()
    m = np.asarray(model).ravel()
    vals = np.concatenate([d, m])

    p = PercentileInterval(percentile)
    vmin, vmax = p.get_limits(vals)
    S = AsinhStretch(a=a)
    svmin, svmax = S(np.array([vmin, vmax], dtype=float))
    norm = mcolors.Normalize(vmin=float(svmin), vmax=float(svmax), clip=True)

    return S, norm


def overplot_ellipse(major_axis_arcsec, ba, pa, x0, y0,
                     height_pixels=None, ax=None,
                     pixscale=0.262, color='red', linestyle='-',
                     linewidth=1, alpha=1.0, clip=True,
                     jpeg=False, draw_majorminor_axes=True,
                     label=None):
    """Draw an ellipse with astronomical position angle on either a
    numpy image (y-axis up) or a JPEG/PNG image (y-axis down).

    Parameters
    ----------
    major_axis_arcsec : :class:`float`
        Total major-axis length in arcseconds.
    ba : :class:`float`
        Minor-to-major axis ratio (b/a).
    pa : :class:`float`
        Position angle of the major axis in degrees, measured
        counter-clockwise from +y (North) toward +x (East).
    x0, y0 : :class:`float`
        Center coordinates in pixel units (zero-indexed).
    height_pixels : :class:`int`, optional
        Height of the image in pixels; required if `jpeg` is True.
    ax : :class:`matplotlib.axes.Axes`, optional
        Axes on which to draw. Defaults to the current axes.
    pixscale : :class:`float`
        Pixel scale in arcsec/pixel.
    color : color-like
        Edge color for the ellipse and axis lines.
    linestyle : :class:`str`
        Line style for the ellipse and axis lines.
    linewidth : :class:`float`
        Ellipse edge line width (axis lines are drawn at a fixed
        width of 1, independent of this parameter -- see Notes).
    alpha : :class:`float`
        Ellipse transparency (axis lines are always drawn fully
        opaque -- see Notes).
    clip : :class:`bool`
        Whether the ellipse patch is clipped to the axes box
        (``clip_on``).
    jpeg : :class:`bool`
        If True, treat the image as having an inverted (origin
        upper) y-axis: `y0` is flipped using `height_pixels`, and the
        position-angle-to-Matplotlib-angle conversion changes sign
        (see Notes).
    draw_majorminor_axes : :class:`bool`
        Whether to also draw line segments along the major and minor
        axes.
    label : :class:`str`, optional
        Label for the ellipse patch (for legends).

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If `jpeg` is True and `height_pixels` is not given.

    Notes
    -----
    The astronomical-PA-to-Matplotlib-angle conversion differs by
    branch: ``ellipse_angle = 90 - pa`` when `jpeg` is True (inverted
    y-axis), but ``ellipse_angle = pa - 90`` when `jpeg` is False
    (numpy image, y-axis up) -- both are correct for their respective
    coordinate conventions, but note the sign/offset is not simply
    negated between the two branches. `linewidth` and `alpha` control
    only the main ellipse patch; the optional major/minor axis line
    segments are always drawn with ``lw=1`` and full opacity,
    regardless of these parameters.

    """
    from matplotlib.patches import Ellipse

    if ax is None:
        ax = plt.gca()

    # Compute half-lengths in pixels
    major_pix = major_axis_arcsec / pixscale
    minor_pix = major_pix * ba
    semia_pix = major_pix / 2.
    semib_pix = semia_pix * ba

    # Adjust center for JPEG (y-axis down)
    if jpeg:
        if height_pixels is None:
            raise ValueError('`height_pixels` required when jpeg=True')
        ycen = height_pixels - y0
        ellipse_angle = 90. - pa
    else:
        ycen = y0
        ellipse_angle = pa - 90.
    xcen = x0

    # Convert astronomical PA -> Matplotlib angle

    # Create and add patch (width, height are full diameters)
    ell = Ellipse((xcen, ycen), width=major_pix, height=minor_pix,
                  angle=ellipse_angle, facecolor='none', edgecolor=color,
                  linewidth=linewidth, linestyle=linestyle, alpha=alpha,
                  clip_on=clip, label=label)
    ax.add_patch(ell)

    # Optionally draw major/minor axis lines
    if draw_majorminor_axes:
        theta = np.deg2rad(ellipse_angle)
        dx_maj = semia_pix * np.cos(theta)
        dy_maj = semia_pix * np.sin(theta)
        dx_min = semib_pix * np.sin(theta)
        dy_min = semib_pix * np.cos(theta)

        # major axis line
        ax.plot([xcen + dx_maj, xcen - dx_maj],
                [ycen + dy_maj, ycen - dy_maj],
                color=color, lw=1, ls=linestyle, clip_on=True)
        # minor axis line
        ax.plot([xcen + dx_min, xcen - dx_min],
                [ycen - dy_min, ycen + dy_min],
                color=color, lw=1, ls=linestyle, clip_on=True)


def qa_skypatch(primary=None, group=None, racol='RA', deccol='DEC', suffix='group',
                pngsuffix=None, objname=None, racenter=None, deccenter=None,
                layers=None, add_title=True, width_arcmin=2., pngdir='.', jpgdir='.',
                clip=False, verbose=False, overwrite_viewer=False, overwrite=False):
    """Build a QA figure showing one primary object and any nearby
    catalog matches (`group`) overlaid on a Legacy Survey/unWISE
    cutout, with arrows/labels pointing to each match and its
    NED/HyperLeda/LVD cross-match position when it differs from the
    adopted position.

    Downloads (or reuses a cached) cutout JPEG by trying, in order,
    ``ls-dr9``, ``ls-dr11-early`` (dev viewer), ``ls-dr10``, then
    ``unwise-neo7``, stopping at the first non-empty image.

    Parameters
    ----------
    primary : :class:`astropy.table.Table` or :class:`astropy.table.Row`, optional
        Single-row parent-style catalog entry for the object the
        cutout is centered on. Required unless `group` is given (in
        which case ``group[0]`` is used as `primary`).
    group : :class:`astropy.table.Table`, optional
        Parent-style catalog of every object to annotate in the field
        (including `primary`); must contain ``OBJNAME``, ``OBJTYPE``,
        ``DIAM_LIT``, ``PGC``, ``DIAM_HYPERLEDA``, `racol`, `deccol`,
        and ``RA_HYPERLEDA``/``DEC_HYPERLEDA``,
        ``RA_NED``/``DEC_NED``, ``RA_LVD``/``DEC_LVD`` (each using
        ``-99.`` as a missing-value sentinel). Defaults to a
        single-row table built from `primary` if not given.
    racol, deccol : :class:`str`
        Column names in `group` for the adopted RA/Dec used to
        position each object's marker.
    suffix : :class:`str`
        Suffix used when checking for/writing the cached cutout JPEG
        filename (``{outname}-{suffix}-{layer}.jpeg``).
    pngsuffix : :class:`str`, optional
        Suffix used for the output PNG filename
        (``{outname}-{pngsuffix}.png``). Defaults to `suffix`.
    objname : :class:`str`, optional
        Object name used to build the output filenames and (if
        `add_title`) the figure title. Defaults to `primary['OBJNAME']`.
    racenter, deccenter : :class:`float`, optional
        Cutout center coordinates. Defaults to `primary[racol]`/
        `primary[deccol]` (see Notes for a partial-default edge case).
    layers : any, optional
        Unused (see Notes).
    add_title : :class:`bool`
        If True, set the axes title to the object name and coordinates.
    width_arcmin : :class:`float`
        Cutout field of view, in arcmin.
    pngdir : :class:`str`
        Output directory for the QA PNG.
    jpgdir : :class:`str`
        Cache directory for the downloaded cutout JPEG.
    clip : :class:`bool`
        Passed to matplotlib's `clip_on`/`annotation_clip` for the
        per-object markers/arrows/labels, controlling whether
        off-cutout annotations are drawn.
    verbose : :class:`bool`
        If True, print the input `group` table (restricted to the
        module-level ``cols`` column list) before plotting.
    overwrite_viewer : :class:`bool`
        If True, re-download the cutout JPEG even if a cached copy
        exists, and implies `overwrite` (see Notes).
    overwrite : :class:`bool`
        If True, regenerate the output PNG even if it already exists.

    Returns
    -------
    :class:`str` or None
        Path to the written PNG file, or None if the PNG already
        existed and `overwrite` is False (in which case nothing is
        written).

    Raises
    ------
    ValueError
        If neither `primary` nor `group` is given.

    Notes
    -----
    The `layers` parameter is accepted but immediately overridden --
    the function unconditionally sets its own local ``layers =
    ['ls-dr9', 'ls-dr11-early', 'ls-dr10', 'unwise-neo7']`` (and a
    matching ``surveys`` list) a few lines into the body, so any
    caller-supplied value has no effect. ``overwrite_viewer=True``
    forces ``overwrite = True`` as a side effect, so the two flags are
    not independent despite appearing as separate parameters. If only
    one of `racenter`/`deccenter` is given (not both, not neither),
    the ``if racenter is None and deccenter is None:`` guard is False
    and neither is populated from `primary`, leaving the omitted one
    as `None` and causing a downstream `TypeError`/`ValueError` when
    it reaches WCS/pixel-coordinate arithmetic. `verbose` prints
    ``group[cols]`` using ``cols``, a module-level constant list
    defined near the top of this file (not a local or parameter) --
    not a bug, but easy to miss when reading this function in
    isolation.

    """
    import matplotlib.image as mpimg
    from matplotlib.patches import Circle
    from matplotlib.lines import Line2D
    from urllib.request import urlretrieve
    from astropy.wcs import WCS
    from astropy.io import fits

    if overwrite_viewer:
        overwrite = True

    if pngsuffix is None:
        pngsuffix = suffix

    def get_wcs(racenter, deccenter, width_arcmin=2., survey='ls'):
        """Build a tangent-plane (TAN) WCS for a cutout centered on
        `racenter`/`deccenter`, sized for a given survey's pixel scale.

        Parameters
        ----------
        racenter, deccenter : :class:`float`
            Cutout center coordinates, in degrees.
        width_arcmin : :class:`float`
            Desired field of view, in arcmin.
        survey : :class:`str`
            Either ``'ls'`` (0.262 arcsec/pixel) or ``'unwise'``
            (2.75 arcsec/pixel); looked up in a small internal
            ``pix`` dictionary. Any other value raises `KeyError`.

        Returns
        -------
        wcs : :class:`astropy.wcs.WCS`
            WCS for the cutout.
        width : :class:`int`
            Cutout width/height, in pixels (square).
        pixscale : :class:`float`
            Pixel scale used, in arcsec/pixel.

        """
        pix = {'ls': 0.262, 'unwise': 2.75}
        pixscale = pix[survey]
        width = int(60. * width_arcmin / pixscale)
        hdr = fits.Header()
        hdr['NAXIS'] = 2
        hdr['NAXIS1'] = width
        hdr['NAXIS2'] = width
        hdr['CTYPE1'] = 'RA---TAN'
        hdr['CTYPE2'] = 'DEC--TAN'
        hdr['CRVAL1'] = racenter
        hdr['CRVAL2'] = deccenter
        hdr['CRPIX1'] = width/2+0.5
        hdr['CRPIX2'] = width/2+0.5
        hdr['CD1_1'] = -pixscale/3600.
        hdr['CD1_2'] = 0.0
        hdr['CD2_1'] = 0.0
        hdr['CD2_2'] = +pixscale/3600.
        wcs = WCS(hdr)
        width = wcs.pixel_shape[0]
        return wcs, width, pixscale


    def get_url(racenter, deccenter, width, layer, dev=False):
        """Build a Legacy Survey ``jpeg-cutout`` API URL for a given
        center, size, and imaging layer.

        Parameters
        ----------
        racenter, deccenter : :class:`float`
            Cutout center coordinates, in degrees.
        width : :class:`int`
            Cutout width/height, in pixels (square).
        layer : :class:`str`
            Legacy Survey imaging layer name (e.g. ``'ls-dr10'``).
        dev : :class:`bool`
            If True, use the ``viewer-dev`` endpoint instead of the
            production ``viewer`` endpoint.

        Returns
        -------
        :class:`str`
            Fully-formed cutout URL.

        """
        if dev:
            viewer = 'viewer-dev'
        else:
            viewer = 'viewer'
        url = f'https://www.legacysurvey.org/{viewer}/jpeg-cutout?ra={racenter}&dec=' + \
            f'{deccenter}&width={width}&height={width}&layer={layer}'
        #print(url)
        return url

    bbox = dict(boxstyle='round', facecolor='k', alpha=0.5)
    ref_pixscale = 0.262

    if primary is None and group is None:
        raise ValueError('Must specify group *and/or* primary')

    if primary is not None and group is None:
        group = Table(primary)
    if primary is None and group is not None:
        primary = Table(group[0])

    if verbose:
        print(group[cols])
    N = len(group)

    if racenter is None and deccenter is None:
        racenter = primary[racol]
        deccenter = primary[deccol]
    if objname is None:
        objname = primary['OBJNAME']
    if not np.isscalar(objname):
        objname = objname[0]

    outname = objname.replace(' ', '_')
    pngfile = os.path.join(pngdir, f'{outname}-{pngsuffix}.png')
    if os.path.isfile(pngfile) and not overwrite:
        print(f'Output file {pngfile} exists; skipping.')
        return


    # check if the viewer cutout file exists
    surveys = ['ls', 'ls', 'ls', 'unwise']
    layers = ['ls-dr9', 'ls-dr11-early', 'ls-dr10', 'unwise-neo7']
    devs = [False, True, False, False]
    for survey, layer in zip(surveys, layers):
        jpgfile = os.path.join(jpgdir, f'{outname}-{suffix}-{layer}.jpeg')
        if os.path.isfile(jpgfile):
            surveys = [survey]
            layers = [layer]

    for survey, layer, dev in zip(surveys, layers, devs):
        wcs, width, pixscale = get_wcs(racenter, deccenter, survey=survey, width_arcmin=width_arcmin)
        jpgfile = os.path.join(jpgdir, f'{outname}-{suffix}-{layer}.jpeg')
        if os.path.isfile(jpgfile) and not overwrite_viewer:
            img = mpimg.imread(jpgfile)
        else:
            urlretrieve(get_url(racenter, deccenter, width, layer=layer, dev=dev), jpgfile)
            img = mpimg.imread(jpgfile)
            if np.all(img == 32): # no data
                os.remove(jpgfile)
            else:
                break

    decsort = np.argsort(group[deccol])

    leg_colors = ['red']
    leg_lines = ['-']
    leg_labels = ['Adopted']

    #fig = plt.figure()
    #ax = fig.add_subplot()#projection=wcs)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img, origin='lower')
    for imem, (mem, yoffset) in enumerate(zip(group[decsort], (np.arange(0, N)+0.5) * width / N)):

        label = f'{mem["OBJNAME"]} ({mem["OBJTYPE"]}): D={mem["DIAM_LIT"]:.3g}\nPGC {mem["PGC"]}: D(LEDA)={mem["DIAM_HYPERLEDA"]:.3g})'

        if imem % 2 == 0:
            xoffset = 0.2 * width
        else:
            xoffset = width - 0.2 * width

        if width-yoffset > width / 2:
            va = 'top'
        else:
            va = 'bottom'

        pix = wcs.wcs_world2pix(mem[racol], mem[deccol], 1)
        if np.abs(pix[1]-yoffset) < int(0.03*width):
            yoffset += int(0.03*width)

        ax.add_artist(Circle((pix[0], width-pix[1]), radius=4.*ref_pixscale/pixscale,
                             facecolor='none', edgecolor='red', lw=2, ls='-', alpha=0.9, clip_on=clip))
        ax.annotate('', xy=(pix[0], width-pix[1]), xytext=(xoffset, width-yoffset),
                    annotation_clip=clip, arrowprops=dict(
                        facecolor='red', width=3, headwidth=8, shrink=0.01, alpha=0.9))
        ax.annotate(label, xy=(xoffset, width-yoffset), xytext=(xoffset, width-yoffset),
                    va=va, ha='center', color='white', bbox=bbox, fontsize=9,
                    annotation_clip=clip)

        if mem['RA_HYPERLEDA'] != -99. and mem['RA'] != mem['RA_HYPERLEDA']:
            hyper_pix = wcs.wcs_world2pix(mem['RA_HYPERLEDA'], mem['DEC_HYPERLEDA'], 1)
            ax.add_artist(Circle((hyper_pix[0], width-hyper_pix[1]), radius=8.*ref_pixscale/pixscale,
                                 facecolor='none', edgecolor='cyan', lw=1, ls='--', alpha=0.5, clip_on=clip))
            ax.annotate('', xy=(hyper_pix[0], width-hyper_pix[1]), xytext=(xoffset, width-yoffset),
                        annotation_clip=clip, arrowprops=dict(
                            facecolor='cyan', width=1, ls='--', headwidth=4, shrink=0.01, alpha=0.5))
            leg_colors += ['cyan']
            leg_lines += ['--']
            leg_labels += ['HyperLeda']

        if mem['RA_NED'] != -99. and mem['RA'] != mem['RA_NED']:
            ned_pix = wcs.wcs_world2pix(mem['RA_NED'], mem['DEC_NED'], 1)
            ax.add_artist(Circle((ned_pix[0], width-ned_pix[1]), radius=8.*ref_pixscale/pixscale,
                                 facecolor='none', edgecolor='yellow', lw=1, ls='--', alpha=0.5, clip_on=clip))
            ax.annotate('', xy=(ned_pix[0], width-ned_pix[1]), xytext=(xoffset, width-yoffset),
                        annotation_clip=clip, arrowprops=dict(
                            facecolor='yellow', width=1, ls='--', headwidth=4, shrink=0.01, alpha=0.5))
            leg_colors += ['yellow']
            leg_lines += ['--']
            leg_labels += ['NED']

        if mem['RA_LVD'] != -99. and mem['RA'] != mem['RA_LVD']:
            lvd_pix = wcs.wcs_world2pix(mem['RA_LVD'], mem['DEC_LVD'], 1)
            ax.add_artist(Circle((lvd_pix[0], width-lvd_pix[1]), radius=8.*ref_pixscale/pixscale,
                                 facecolor='none', edgecolor='blue', lw=1, ls='--', alpha=0.5, clip_on=clip))
            ax.annotate('', xy=(lvd_pix[0], width-lvd_pix[1]), xytext=(xoffset, width-yoffset),
                        annotation_clip=clip, arrowprops=dict(
                            facecolor='blue', width=1, ls='--', headwidth=4, shrink=0.01, alpha=0.5))
            leg_colors += ['blue']
            leg_lines += ['--']
            leg_labels += ['LVD']

    _, uindx = np.unique(leg_labels, return_index=True)
    ax.legend([Line2D([0], [0], color=col, linewidth=2, linestyle=ls)
               for col, ls in zip(np.array(leg_colors)[uindx], np.array(leg_lines)[uindx])],
              np.array(leg_labels)[uindx], loc='upper left', frameon=True, framealpha=0.7,
              fontsize=8)

    ax.invert_yaxis() # JPEG is flipped relative to my FITS WCS
    if add_title:
        ax.set_title(f'{objname} ({racenter:.8f},{deccenter:.8f})')
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(pngfile, bbox_inches=0)#, dpi=200)
    plt.close()
    #print(f'Wrote {pngfile}')

    return pngfile


def multipage_skypatch(primaries, cat=None, width_arcsec=75., ncol=1, nrow=1,
                       add_title=True, layers=None, pngsuffix='group', jpgdir='.',
                       pngdir='.', pdffile='multipage-skypatch.pdf', clip=True,
                       verbose=False, overwrite_viewer=False, overwrite=True,
                       cleanup=False):
    """Call :func:`qa_skypatch` for every object in `primaries`
    (optionally cross-matched against a companion catalog `cat`), then
    assemble the resulting per-object PNGs into a single multi-page,
    multi-panel PDF.

    Parameters
    ----------
    primaries : :class:`astropy.table.Table`
        Table of objects to build one :func:`qa_skypatch` figure for
        each; must contain ``OBJNAME``, ``OBJTYPE``, ``RA``, ``DEC``.
    cat : :class:`astropy.table.Table`, optional
        Catalog to spatially cross-match against each row of
        `primaries` (via ``astrometry.libkd.spherematch.match_radec``,
        radius `width_arcsec`); matches become that object's `group`
        argument to :func:`qa_skypatch`. If not given, each object is
        plotted with no companions (``group=None``).
    width_arcsec : :class:`float`
        Cross-match radius (and, converted to arcmin, the
        :func:`qa_skypatch` cutout field of view), in arcsec.
    ncol, nrow : :class:`int`
        Grid layout (columns, rows) of QA panels per PDF page.
    add_title : :class:`bool`
        Passed through to :func:`qa_skypatch`.
    layers : any, optional
        Unused (see Notes).
    pngsuffix : :class:`str`
        Passed through to :func:`qa_skypatch` as its `pngsuffix`.
    jpgdir : :class:`str`
        Passed through to :func:`qa_skypatch`; created if missing.
    pngdir : :class:`str`
        Passed through to :func:`qa_skypatch`; created if missing.
    pdffile : :class:`str`
        Output multi-page PDF path.
    clip : :class:`bool`
        Passed through to :func:`qa_skypatch`.
    verbose : :class:`bool`
        Passed through to :func:`qa_skypatch`; also prints a blank
        line after each object.
    overwrite_viewer : :class:`bool`
        Passed through to :func:`qa_skypatch`.
    overwrite : :class:`bool`
        Passed through to :func:`qa_skypatch`.
    cleanup : :class:`bool`
        If True, delete the individual per-object PNGs (and, if
        `pngdir` happens to also satisfy ``os.path.isfile``, attempt
        to ``shutil.rmtree`` it -- see Notes) after the PDF is
        written.

    Returns
    -------
    None

    Notes
    -----
    The `layers` parameter is accepted but never passed to
    :func:`qa_skypatch` (whose own `layers` parameter is itself dead;
    see its Notes) -- doubly unused. The `cleanup` block's final
    check, ``if os.path.isfile(pngdir): shutil.rmtree(pngdir)``, tests
    `pngdir` (a directory path) with ``os.path.isfile``, which is only
    True for a *file*, not a directory -- as written this branch can
    never fire for a normal directory `pngdir`, so the `pngdir` tree
    itself is never actually removed even when `cleanup` is True
    (only the individual PNGs are).

    """
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.image import imread

    from astrometry.libkd.spherematch import match_radec

    if not os.path.isdir(jpgdir):
        os.makedirs(jpgdir)
    if not os.path.isdir(pngdir):
        os.makedirs(pngdir)

    primaries = Table(primaries)
    nobj = len(primaries)

    if cat is None:
        groups = [None] * nobj
    else:
        matches = match_radec(primaries['RA'].value, primaries['DEC'].value, cat['RA'].value,
                              cat['DEC'].value, width_arcsec/3600., indexlist=True, notself=False)
        groups = [cat[onematch] for onematch in matches]


    pngfile = []
    for iobj, (primary, group) in enumerate(zip(primaries, groups)):
        #width = crossid['dtheta_arcsec'] / 60. # [arcmin]
        objname = f'{primary["OBJNAME"]}-{primary["OBJTYPE"]}'
        racenter, deccenter = primary['RA'], primary['DEC']
        #pngfile.append(
        #    qa_skypatch(group=fullcat[match_full[iobj+ss]], pngsuffix='original',
        #                add_title=False,
        #                objname=objname, racenter=racenter, deccenter=deccenter,
        #                pngdir=pngdir, jpgdir=jpgdir, verbose=True, overwrite=True)
        #    )
        pngfile.append(
            qa_skypatch(primary, group=group, pngsuffix=pngsuffix, add_title=add_title,
                        objname=objname, racenter=racenter, deccenter=deccenter,
                        width_arcmin=width_arcsec / 60., clip=clip, pngdir=pngdir,
                        jpgdir=jpgdir, verbose=verbose, overwrite=overwrite,
                        overwrite_viewer=overwrite_viewer)
        )
        if verbose:
            print()

    pngfile = np.array(pngfile)
    allindx = np.arange(len(pngfile))

    nperpage = ncol * nrow
    npage = int(np.ceil(len(pngfile) / nperpage))

    pdf = PdfPages(pdffile)
    for ipage in range(npage):
        indx = allindx[ipage*nperpage:(ipage+1)*nperpage]
        fig, ax = plt.subplots(nrow, ncol, figsize=(2*ncol, 2*nrow))
        for iax, xx in enumerate(np.atleast_1d(ax).flat):
            if iax < len(indx):
                xx.imshow(imread(pngfile[indx[iax]]), interpolation='None')
            xx.axis('off')
        #for xx in np.arange(iax, nperpage):
        fig.subplots_adjust(wspace=0., hspace=0., bottom=0.05, top=0.95, left=0.05, right=0.95)
        pdf.savefig(fig, dpi=150)
        plt.close()
    pdf.close()
    print(f'Wrote {pdffile}')

    # clean up
    if cleanup:
        for png in pngfile:
            if os.path.isfile(png):
                if verbose:
                    print(f'Removing {png}')
                os.remove(png)
        if os.path.isfile(pngdir):
            shutil.rmtree(pngdir)


def addbar_to_png(jpgfile, barlen, barlabel, imtype, pngfile, scaledfont=True,
                  pixscalefactor=1.0, fntsize=20, imtype_fntsize=20):
    """Annotate a JPEG cutout with a scale bar (with label) and/or an
    image-type label, and write the result out as a PNG. Support
    routine used by :mod:`SGA.html`.

    Parameters
    ----------
    jpgfile : :class:`str`
        Path to the input JPEG image.
    barlen : :class:`float` or None
        Scale-bar length, in pixels (scaled by `pixscalefactor`). If
        falsy (e.g. 0 or None), no scale bar/label is drawn.
    barlabel : :class:`str`
        Text label drawn centered under the scale bar.
    imtype : :class:`str` or None
        Image-type text label (e.g. ``'model'``) drawn in the
        lower-left corner. If falsy, no label is drawn.
    pngfile : :class:`str`
        Output PNG path.
    scaledfont : :class:`bool`
        If True, derive `fntsize`/`imtype_fntsize` from the image
        width (overriding whatever was passed in); if False, use the
        given `fntsize`/`imtype_fntsize` values as-is.
    pixscalefactor : :class:`float`
        Scale factor applied to `barlen` and (when `scaledfont`) to
        the derived font sizes -- used to keep the bar/label
        proportionally sized across images with different pixel
        scales.
    fntsize : :class:`int`
        Font size for the scale-bar label. Only used as given if
        `scaledfont` is False.
    imtype_fntsize : :class:`int`
        Font size for the `imtype` label. Only used as given if
        `scaledfont` is False.

    Returns
    -------
    :class:`str`
        `pngfile` (echoed back for convenience/chaining).

    Notes
    -----
    Uses the module-level ``fonttype`` (``SGA/data/Georgia.ttf``,
    defined near the top of this file) for both labels; there is no
    per-call font override.

    """
    from PIL import Image, ImageDraw, ImageFont

    Image.MAX_IMAGE_PIXELS = None

    with Image.open(jpgfile) as im:
        draw = ImageDraw.Draw(im)
        sz = im.size
        width = np.round(pixscalefactor*sz[0]/150).astype('int')
        if scaledfont:
            fntsize = np.round(pixscalefactor*sz[0]/50).astype('int')                
            imtype_fntsize = np.round(pixscalefactor*sz[0]/15).astype('int')                
            #fntsize = np.round(0.05*sz[0]).astype('int')
            #fntsize = np.round(sz[0]/50).astype('int')
        # Bar and label
        if barlen:
            #if fntsize < 56:
            #    fntsize = 56
            font = ImageFont.truetype(fonttype, size=fntsize)
            # Add a scale bar and label--
            x0, x1, y0, y1 = 0+fntsize*2, 0+fntsize*2+barlen*pixscalefactor, sz[1]-fntsize*1.5, sz[1]-fntsize*4
            #print(sz, fntsize, x0, x1, y0, y1, barlen*pixscalefactor)
            draw.line((x0, y1, x1, y1), fill='white', width=width)
            ww = draw.textlength(barlabel, font=font)
            dx = ((x1-x0) - ww)//2
            #print(x0, x1, y0, y1, ww, x0+dx, sz)
            draw.text((x0+dx, y0), barlabel, font=font)
            #print('Writing {}'.format(pngfile))
        # Image type
        if imtype:
            #fntsize = 20 # np.round(sz[0]/20).astype('int')
            font = ImageFont.truetype(fonttype, size=imtype_fntsize)
            ww = draw.textlength(imtype, font=font)
            x0, y0, y1 = imtype_fntsize*1.2, imtype_fntsize*2, imtype_fntsize*1.2#4
            #x0, y0, y1 = sz[0]-ww-imtype_fntsize*2, sz[1]-imtype_fntsize*2, sz[1]-imtype_fntsize*2.5#4
            draw.text((x0, y1), imtype, font=font)
        print('Writing {}'.format(pngfile))
        im.save(pngfile)
    return pngfile


def qa_maskbits(mask, tractor, ellipsefitall, colorimg, largegalaxy=False, png=None):
    """Build a 3-panel QA figure for one brick/group's maskbits: the
    color mosaic, the raw maskbits image, and a source-scatter panel,
    with each fitted galaxy's R(26) and (if available) HyperLeda
    isophotal ellipses overplotted on all three.

    Parameters
    ----------
    mask : :class:`numpy.ndarray`
        2D maskbits image to display in the middle panel.
    tractor : :class:`astropy.table.Table`
        Tractor catalog for the brick/group, with ``BX``, ``BY``,
        ``RA``, ``DEC``, ``FLUX_R`` columns; sources are scattered on
        the third panel, scaled in size by ``FLUX_R`` for objects
        that fall inside each fitted galaxy's ellipse.
    ellipsefitall : :class:`list`
        List of per-galaxy ellipse-fitting result dicts, each with
        ``ra_moment``, ``dec_moment``, ``pa_moment``, ``eps_moment``,
        ``x0_moment``, ``y0_moment``, ``refpixscale``, and, optionally,
        ``d25_leda``/``ba_leda``/``pa_leda`` (see Notes).
    colorimg : :class:`PIL.Image.Image`
        Color mosaic image (e.g. grz JPEG) to display in the first
        panel and to draw ellipse overlays onto in-place via
        :func:`draw_ellipse_on_png`.
    largegalaxy : :class:`bool`
        Unused (see Notes).
    png : :class:`str`, optional
        Output filename. If given, the figure is written to disk and
        closed; otherwise it is shown interactively.

    Returns
    -------
    None

    Notes
    -----
    Colorblind-friendly colors are from
    https://twitter.com/rachel_kurchin/status/1229567059694170115 .
    This function cannot currently run: its ``from SGA.ellipse import
    is_in_ellipse`` import (used to select Tractor sources inside each
    galaxy's ellipse) fails immediately, since no function named
    ``is_in_ellipse`` is defined anywhere in :mod:`SGA.ellipse` or
    :mod:`SGA.geometry` (the closest equivalent is
    :func:`SGA.geometry.in_ellipse_mask`, with a different signature).
    Separately, the per-galaxy loop calls ``_get_diameter(ellipsefit)``,
    which is also not defined or imported anywhere in this module --
    the only trace of it is a commented-out
    ``#from legacyhalos.SGA import _get_diameter``, indicating this
    function predates the SGA/legacyhalos split and was never ported.
    `largegalaxy` is accepted but never referenced in the function
    body. The HyperLeda-ellipse block has a further, independent bug:
    the ``if igal == 0: ellaper.plot(..., label='Hyperleda')`` block
    (plotting in red) is indented one level shallower than the
    ``if 'd25_leda' in ellipsefit.keys():`` check above it, so it
    executes unconditionally for every galaxy, not just those with
    HyperLeda geometry -- for a galaxy lacking ``d25_leda``, `ellaper`
    is left over from the earlier R(26) computation, so the R(26)
    ellipse gets silently redrawn a second time in red, mislabeled as
    "Hyperleda". This function is not called anywhere else in the
    package (no callers in ``py/``, ``bin/``, or ``archive/``); a
    similarly named script exists at ``science/SGA2020/SGA2020-figures``.

    """
    from photutils import EllipticalAperture
    from PIL import ImageDraw, Image

    from tractor.ellipses import EllipseE
    from legacypipe.reference import get_large_galaxy_version
    from SGA.ellipse import is_in_ellipse
    #from legacyhalos.SGA import _get_diameter

    Image.MAX_IMAGE_PIXELS = None
    imgsz = colorimg.size

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(5*3, 5), sharey=True)

    # original maskbits
    ax2.imshow(mask, origin='lower', cmap='gray_r')#, interpolation='none')
    ax2.set_aspect('equal')
    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)
    #ax2.set_title('Original maskbits')
    #ax2.axis('off')
    #ax2.autoscale(False)
    #ax2.scatter(tractor['BX'], tractor['BY'], alpha=0.3, s=10, color='#999999')

    ax3.scatter(tractor['BX'], tractor['BY'], alpha=0.3, s=10, color='#999999',
                label='All Sources')
    ax3.set_aspect('equal')
    sz = mask.shape
    ax3.set_xlim(0, sz[1]-1)
    ax3.set_ylim(0, sz[0]-1)
    #ax3.imshow(mask*0, origin='lower', cmap='gray_r')#, interpolation='none')
    #ax3.plot([0, sz[1]-1], [0, sz[0]-1])
    ax3.get_xaxis().set_visible(False)
    ax3.get_yaxis().set_visible(False)
    ##ax3.set_title('Original maskbits')
    #ax3.axis('off')

    #refcat, _ = get_large_galaxy_version(os.getenv('LARGEGALAXIES_CAT'))
    #ilslga = np.where(tractor['REF_CAT'] == refcat)[0]
    #ax3.scatter(tractor['BX'][ilslga], tractor['BY'][ilslga], s=50,
    #            edgecolor='k', color='blue')
    
    #ax3.autoscale(False)
    ax3.margins(0, tight=True)

    minmarker, maxmarker = 30, 300
    for igal, ellipsefit in enumerate(ellipsefitall):
        diam, diamref = _get_diameter(ellipsefit)
        ragal, decgal, pa, ba = ellipsefit['ra_moment'], ellipsefit['dec_moment'], ellipsefit['pa_moment'], 1-ellipsefit['eps_moment']

        reff, e1, e2 = EllipseE.fromRAbPhi(diam*60/2, ba, 180-pa) # note the 180 rotation
        inellipse = np.where(is_in_ellipse(tractor['RA'], tractor['DEC'], ragal, decgal, reff, e1, e2))[0]
        if len(inellipse) < 3:
            continue

        # scale the size of the marker by flux
        minflx, maxflx = np.percentile(tractor['FLUX_R'][inellipse], [50, 95])
        if maxflx > minflx:
            ss = maxmarker * (tractor['FLUX_R'][inellipse] - minflx) / (maxflx - minflx)
        else:
            ss = np.repeat(maxmarker, len(tractor))
        ss[ss < minmarker] = minmarker
        ss[ss > maxmarker] = maxmarker

        if igal == 0:
            ax3.scatter(tractor['BX'][inellipse], tractor['BY'][inellipse], s=ss,
                        marker='s', edgecolor='k', color=cb_colors['orange'], label='Frozen Sources')
        else:
            ax3.scatter(tractor['BX'][inellipse], tractor['BY'][inellipse], s=ss,
                        marker='s', edgecolor='k', color=cb_colors['orange'])

        # ellipse geometry
        maxis = diam * 60 / ellipsefit['refpixscale'] / 2 # [pixels]
        ellaper = EllipticalAperture((ellipsefit['x0_moment'], ellipsefit['y0_moment']),
                                     maxis, maxis*(1 - ellipsefit['eps_moment']),
                                     np.radians(ellipsefit['pa_moment']-90))
        if igal == 0:
            ellaper.plot(color=cb_colors['blue'], lw=2, ax=ax2, alpha=0.9, label='R(26)')
        else:
            ellaper.plot(color=cb_colors['blue'], lw=2, ax=ax2, alpha=0.9)
        ellaper.plot(color=cb_colors['blue'], lw=2, ls='-', ax=ax3, alpha=0.9)

        draw_ellipse_on_png(colorimg, ellipsefit['x0_moment'], imgsz[1]-ellipsefit['y0_moment'],
                            1-ellipsefit['eps_moment'],
                            ellipsefit['pa_moment'], 2 * maxis * ellipsefit['refpixscale'],
                            ellipsefit['refpixscale'], color=cb_colors['blue']) # '#ffaa33')
        if 'd25_leda' in ellipsefit.keys():
            draw_ellipse_on_png(colorimg, ellipsefit['x0_moment'], imgsz[1]-ellipsefit['y0_moment'],
                                ellipsefit['ba_leda'], ellipsefit['pa_leda'],
                                ellipsefit['d25_leda'] * 60.0, ellipsefit['refpixscale'],
                                color=cb_colors['red'])
        
            # Hyperleda geometry
            maxis = ellipsefit['d25_leda'] * 60 / ellipsefit['refpixscale'] / 2 # [pixels]
            ellaper = EllipticalAperture((ellipsefit['x0_moment'], ellipsefit['y0_moment']),
                                         maxis, maxis * ellipsefit['ba_leda'],
                                         np.radians(ellipsefit['pa_leda']-90))
        if igal == 0:
            ellaper.plot(color=cb_colors['red'], lw=2, ls='-', ax=ax2, alpha=1.0, label='Hyperleda')
        else:
            ellaper.plot(color=cb_colors['red'], lw=2, ls='-', ax=ax2, alpha=1.0)
        ellaper.plot(color=cb_colors['red'], lw=2, ls='-', ax=ax3, alpha=1.0)

    # color mosaic
    draw = ImageDraw.Draw(colorimg)
    ax1.imshow(np.flipud(colorimg), interpolation='none') # not sure why I have to flip here...
    ax1.set_aspect('equal')
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)

    #ax1.axis('off')
    #ax1.autoscale(False)
    #ax1.scatter(tractor['BX'], imgsz[1]-tractor['BY'], alpha=1.0, s=10, color='red')
    #ax1.scatter(tractor['BX'], tractor['BY'], alpha=1.0, s=10, color='#999999')

    hh, ll = ax2.get_legend_handles_labels()
    if len(hh) > 0:
        ax2.legend(loc='lower right', fontsize=12)
        lgnd = ax3.legend(loc='lower right', fontsize=12)
        lgnd.legendHandles[0]._sizes = [40]
        lgnd.legendHandles[1]._sizes = [40]

    fig.subplots_adjust(wspace=0.05, right=0.9)

    if png:
        print('Writing {}'.format(png))
        fig.savefig(png, bbox_inches='tight')#, pad_inches=0)
        plt.close(fig)
    else:
        plt.show()


# adapted from https://github.com/desihub/desiutil/blob/5735fdc34c4e77c7fda84c92c32b9ac41158b8e1/py/desiutil/plots.py#L735-L857
def ar_sky_cbar(ax, sc, label, extend=None, mloc=None):
    """Add a horizontal colorbar below a sky-map axes, styled to match
    `desiutil.plots`-based figures (bottom-anchored ticks, zero-forced
    lower limit).

    Parameters
    ----------
    ax : :class:`matplotlib.axes.Axes`
        Axes to attach the colorbar to.
    sc : :class:`matplotlib.cm.ScalarMappable`
        Mappable (e.g. the return value of ``ax.scatter``/``imshow``)
        that defines the colorbar's data range and colormap.
    label : :class:`str`
        Colorbar label.
    extend : :class:`str`, optional
        Passed to :func:`matplotlib.pyplot.colorbar`'s ``extend``
        keyword (e.g. ``'min'``, ``'max'``, ``'both'``) to draw
        pointed end caps.
    mloc : :class:`float`, optional
        If given, spacing (in data units) for
        :class:`matplotlib.ticker.MultipleLocator` major ticks on the
        colorbar axis, overriding the default tick locator.

    Returns
    -------
    None

    Notes
    -----
    Adapted from ``desiutil.plots`` (see the comment above this
    function for the source URL). Not called anywhere else in this
    package.

    """
    cbar = plt.colorbar(sc, ax=ax, location='bottom',
                        orientation="horizontal",
                        spacing="proportional",
                        extend=extend, extendfrac=0.025,
                        pad=0.1,
                        fraction=0.035, aspect=40)
    cbar.ax.xaxis.set_ticks_position("bottom")
    cbar.ax.set_xlim(0., cbar.ax.get_xlim()[1]) # force zero
    cbar.set_label(label)#, labelpad=10)
    if mloc is not None:
        cbar.ax.xaxis.set_major_locator(ticker.MultipleLocator(mloc))


def plot_sky_binned(ra, dec, weights=None, data=None, plot_type='grid',
                    max_bin_area=5, clip_lo=None, clip_hi=None, verbose=False,
                    cmap='viridis', colorbar=True, label=None, ax=None,
                    return_grid_data=False, **kwargs):
    """Show objects on the sky using a binned plot.

    Bin values either show object counts per unit sky area or, if an array
    of associated data values is provided, mean data values within each bin.
    Objects can have associated weights.

    Requires that matplotlib is installed. When plot_type is
    "healpix", healpy must also be installed.

    Additional keyword parameters will be passed to :func:`init_sky`.

    Parameters
    ----------
    ra : array
        Array of object RA values in degrees. Must have the same shape as
        dec and will be flattened if necessary.
    dec : array
        Array of object Dec values in degrees. Must have the same shape as
        ra and will be flattened if necessary.
    weights : array, optional
        Optional of weights associated with each object.  All objects are
        assumed to have equal weight when this is None.
    data : array, optional
        Optional array of scalar values associated with each object. The
        resulting plot shows the mean data value per bin when data is
        specified.  Otherwise, the plot shows counts per unit sky area.
    plot_type : {'grid', 'healpix'}
        Must be either 'grid' or 'healpix', and selects whether data in
        binned in healpix or in (sin(Dec), RA).
    max_bin_area : :class:`float`, optional
        The bin size will be chosen automatically to be as close as
        possible to this value but not exceeding it.
    clip_lo : :class:`float` or :class:`str`, optional
        Clipping is applied to the plot data calculated as counts / area
        or the mean data value per bin. See :func:`prepare_data` for
        details.
    clip_hi : :class:`float` or :class:`str`, optional
        Clipping is applied to the plot data calculated as counts / area
        or the mean data value per bin. See :func:`prepare_data` for
        details.
    verbose : :class:`bool`, optional
        Print information about the automatic bin size calculation.
    cmap : colormap name or object, optional
        Matplotlib colormap to use for mapping data values to colors.
    colorbar : :class:`bool`, optional
        Draw a colorbar below the map when True.
    label : :class:`str`, optional
        Label to display under the colorbar.  Ignored unless colorbar is ``True``.
    ax : :class:`matplotlib.axes.Axes`, optional
        Axes to use for drawing this map, or create default axes using
        :func:`init_sky` when ``None``.
    return_grid_data : :class:`bool`, optional
        If ``True``, return (ax, grid_data) instead of just ax.

    Returns
    -------
    :class:`matplotlib.axes.Axes` or (ax, grid_data)
        The axis object used for the plot, and the grid_data if
        `return_grid_data` is ``True``.

    Notes
    -----
    Adapted from ``desiutil.plots.plot_sky_binned``. The ``plot_type
    == 'grid'`` branch calls a function ``plot_grid_map`` that is
    never imported or defined anywhere in this module (only
    ``plot_type == 'healpix'``'s counterpart, :func:`plot_healpix_map`,
    is defined here) -- passing ``plot_type='grid'`` (the default)
    always raises ``NameError``. Not called anywhere else in this
    package.

    """
    from desiutil.plots import prepare_data

    ra = np.asarray(ra).reshape(-1)
    dec = np.asarray(dec).reshape(-1)
    if len(ra) != len(dec):
        raise ValueError('Arrays ra,dec must have same size.')

    plot_types = ('grid', 'healpix',)
    if plot_type not in plot_types:
        raise ValueError('Invalid plot_type, should be one of {0}.'.format(', '.join(plot_types)))

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
            print('Using {0} x {1} grid in cos(DEC) x RA'.format(n_cos_dec, n_ra),
                  'with pixel area {:.3f} sq.deg.'.format(bin_area))

        # Calculate the bin edges in degrees.
        # ra_edges = np.linspace(-180., +180., n_ra + 1)
        ra_edges = np.linspace(0.0, 360.0, n_ra + 1)
        dec_edges = np.degrees(np.arcsin(np.linspace(-1., +1., n_cos_dec + 1)))

        # Put RA values in the range [-180, 180).
        # ra = np.fmod(ra, 360.)
        # ra[ra >= 180.] -= 360.

        # Histogram the input coordinates.
        counts, _, _ = np.histogram2d(dec, ra, [dec_edges, ra_edges],
                                      weights=weights)

        if data is None:
            grid_data = counts / bin_area
        else:
            sums, _, _ = np.histogram2d(dec, ra, [dec_edges, ra_edges],
                                        weights=weights * data)
            # This ratio might result in some nan (0/0) or inf (1/0) values,
            # but these will be masked by prepare_data().
            settings = np.seterr(all='ignore')
            grid_data = sums / counts
            np.seterr(**settings)

        grid_data = prepare_data(grid_data, clip_lo=clip_lo, clip_hi=clip_hi)

        ax = plot_grid_map(grid_data, ra_edges, dec_edges,
                           cmap=cmap, colorbar=colorbar, label=label,
                           ax=ax, **kwargs)

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
            print('Using healpix map with NSIDE={0}'.format(nside),
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
        # Hack
        import numpy.ma as ma
        R = np.logical_or(grid_data==0., grid_data <= clip_lo)
        grid_data[R] = ma.masked

        ax = plot_healpix_map(grid_data, nest=nest,
                              cmap=cmap, colorbar=colorbar, label=label,
                              ax=ax, **kwargs)

    if return_grid_data:
        return (ax, grid_data)
    else:
        return ax


def plot_healpix_map(data, nest=False, cmap='viridis', colorbar=True,
                     label=None, ax=None, **kwargs):
    """Plot a healpix map using an all-sky projection.

    Pass the data array through :func:`prepare_data` to select a subset to plot
    and clip the color map to specified values or percentiles.

    This function is similar to :func:`plot_grid_map` but is generally slower
    at high resolution and has less elegant handling of pixels that wrap around
    in RA, which are not drawn.

    Requires that matplotlib and healpy are installed.

    Additional keyword parameters will be passed to :func:`init_sky`.

    Parameters
    ----------
    data : array or masked array
        1D array of data associated with each healpix.  Must have a size that
        exactly matches the number of pixels for some NSIDE value. Use the
        output of :func:`prepare_data` as a convenient way to specify
        data cuts and color map clipping.
    nest : :class:`bool`, optional
        If ``True``, assume NESTED pixel ordering.  Otheriwse, assume RING pixel
        ordering.
    cmap : colormap name or object, optional
        Matplotlib colormap to use for mapping data values to colors.
    colorbar : :class:`bool`, optional
        Draw a colorbar below the map when ``True``.
    label : :class:`str`, optional
        Label to display under the colorbar.  Ignored unless colorbar is ``True``.
    ax : :class:`matplotlib.axes.Axes`, optional
        Axes to use for drawing this map, or create default axes using
        :func:`init_sky` when ``None``.

    Returns
    -------
    :class:`matplotlib.axes.Axes`
        The axis object used for the plot.

    Notes
    -----
    Adapted from ``desiutil.plots.plot_healpix_map``. Calls
    ``init_sky(**kwargs)`` when `ax` is None, but ``init_sky`` is not
    imported anywhere in this function or module -- only imported
    locally inside other functions (:func:`fig_sky`,
    ``qa_binned_radec``). Calling this function (or
    :func:`plot_sky_binned` with ``plot_type='healpix'``) without
    explicitly passing `ax` raises ``NameError``. Not called anywhere
    else in this package.

    """
    import healpy as hp
    from desiutil.plots import prepare_data
    from matplotlib.colors import Normalize, colorConverter
    from matplotlib.collections import PolyCollection

    data = prepare_data(data)
    if len(data.shape) != 1:
        raise ValueError('Invalid data array, should be 1D.')
    nside = hp.npix2nside(len(data))
    #
    # Create axes.
    #
    if ax is None:
        ax = init_sky(**kwargs)
    proj_edge = ax._ra_center - 180
    #
    # Find the projection edge.
    #
    while proj_edge < 0:
        proj_edge += 360
    #
    # Get pixel boundaries as quadrilaterals.
    #
    corners = hp.boundaries(nside, np.arange(len(data)), step=1, nest=nest)
    corner_theta, corner_phi = hp.vec2ang(corners.transpose(0, 2, 1))
    corner_ra, corner_dec = (np.degrees(corner_phi),
                             np.degrees(np.pi/2-corner_theta))
    #
    # Convert sky coords to map coords.
    #
    x, y = ax.projection_ra(corner_ra), ax.projection_dec(corner_dec)
    #
    # Regroup into pixel corners.
    #
    verts = np.array([x.reshape(-1, 4), y.reshape(-1, 4)]).transpose(1, 2, 0)
    #
    # Find and mask any pixels that wrap around in RA.
    #
    uv_verts = np.array([corner_phi.reshape(-1, 4),
                         corner_theta.reshape(-1, 4)]).transpose(1, 2, 0)
    theta_edge = np.unique(uv_verts[:, :, 1])
    phi_edge = np.radians(proj_edge)
    eps = 0.1 * np.sqrt(hp.nside2pixarea(nside))
    wrapped1 = hp.ang2pix(nside, theta_edge, phi_edge - eps, nest=nest)
    wrapped2 = hp.ang2pix(nside, theta_edge, phi_edge + eps, nest=nest)
    wrapped = np.unique(np.hstack((wrapped1, wrapped2)))
    data.mask[wrapped] = True
    #
    # Normalize the data using its vmin, vmax attributes, if present.
    #
    try:
        norm = Normalize(vmin=data.vmin, vmax=data.vmax)
    except AttributeError:
        norm = None
    #
    # Make the collection and add it to the plot.
    #
    good = np.where(~data.mask)[0]
    collection = PolyCollection(verts[good, :, :], array=data[good], cmap=cmap, norm=norm,
                                edgecolors='none')
    ax.add_collection(collection)
    ax.autoscale_view()

    if colorbar:
        bar = plt.colorbar(collection, ax=ax,
                           orientation='horizontal', spacing='proportional',
                           pad=0.11, fraction=0.05, aspect=50)
        if label:
            bar.set_label(label)

    return ax


def fig_sky(S, racolumn='RA', deccolumn='DEC', clip_lo=0., clip_hi=50.,
            max_bin_area=10, mloc=10, pngfile=None):
    """Build a Mollweide-projection sky-density map of a catalog,
    binned into HEALPix cells via :func:`plot_sky_binned` and rendered
    with a colorbar (:func:`ar_sky_cbar`).

    Parameters
    ----------
    S : :class:`astropy.table.Table`
        Catalog to plot; must contain `racolumn` and `deccolumn`.
    racolumn : :class:`str`
        Name of the right ascension column, in degrees.
    deccolumn : :class:`str`
        Name of the declination column, in degrees.
    clip_lo : :class:`float`
        Lower clipping value for the binned density (passed to
        :func:`plot_sky_binned`).
    clip_hi : :class:`float`
        Upper clipping value for the binned density (passed to
        :func:`plot_sky_binned`).
    max_bin_area : :class:`float`
        Maximum HEALPix bin area, in deg2 (passed to
        :func:`plot_sky_binned`).
    mloc : :class:`int`
        Minor-tick locator spacing for the colorbar (passed to
        :func:`ar_sky_cbar`).
    pngfile : :class:`str`, optional
        Output PNG path. If not given, the figure is built but never
        written to disk or returned (see Notes).

    Returns
    -------
    None

    Notes
    -----
    If `pngfile` is not given, the function has no way to return or
    display the figure it built -- both the ``fig``/``ax`` objects
    and the printed density statistics (min/median/mean/std/max, via
    a bare ``print()`` call) are the only observable output, and the
    figure itself is silently discarded (never closed either, in that
    case -- only the `pngfile` branch calls ``plt.close(fig)``). The
    nested helpers ``plot_des``, ``plot_gp_ep``, and
    ``custom_plot_sky_circles`` (DES footprint, Galactic/ecliptic
    plane, and custom sky-circle overlays) are all fully implemented
    but never invoked -- every call site is commented out in the
    function body -- making them dead code in the current version.
    The commented-out ``from desiutil.plots import ...,
    plot_sky_binned`` at the top confirms this module intentionally
    shadows/replaces desiutil's version with its own
    :func:`SGA.qa.plot_sky_binned`.

    """
    import seaborn as sns
    import healpy as hp
    from astropy.coordinates import SkyCoord
    from astropy import units, constants
    from desiutil.plots import init_sky#, prepare_data, plot_sky_binned

    #sns, _ = plot_style(talk=True, font_scale=0.8)

    font = {'size': 14,} #'family': 'normal', 'weight': 'bold'}
    mpl.rc('font', **font)

    # AR DES
    def plot_des(ax, desfn=None, **kwargs):
        """Overplot the DES footprint boundary on a sky-projection axes.

        Parameters
        ----------
        ax : :class:`matplotlib.axes.Axes`
            Sky-projection axes (as returned by
            ``desiutil.plots.init_sky``), exposing ``projection_ra``/
            ``projection_dec``.
        desfn : :class:`str`, optional
            Path to a two-column (RA, Dec) DES footprint text file.
            Defaults to ``$DESI_ROOT/survey/observations/misc/des_footprint.txt``.
        **kwargs
            Passed to ``ax.plot``.

        Returns
        -------
        None

        Notes
        -----
        Dead code in the current version of :func:`fig_sky` -- never
        called (see :func:`fig_sky`'s Notes).

        """
        if desfn is None:
            desfn = os.path.join(os.getenv("DESI_ROOT"), "survey", "observations", "misc", "des_footprint.txt")
        ras, decs = np.loadtxt(desfn, unpack=True)
        ax.plot(ax.projection_ra(ras), ax.projection_dec(decs), **kwargs)

    # AR galactic, ecliptic plane
    def plot_gp_ep(ax, frame, npt=1000, **kwargs):
        """Overplot a Galactic- or ecliptic-plane great circle on a
        sky-projection axes.

        Parameters
        ----------
        ax : :class:`matplotlib.axes.Axes`
            Sky-projection axes, exposing ``projection_ra``/
            ``projection_dec``.
        frame : :class:`str`
            Coordinate frame to draw, e.g. ``"galactic"`` or
            ``"barycentricmeanecliptic"``.
        npt : :class:`int`
            Number of points used to sample the great circle.
        **kwargs
            Passed to ``ax.plot``.

        Returns
        -------
        None

        Notes
        -----
        Dead code in the current version of :func:`fig_sky` -- never
        called (see :func:`fig_sky`'s Notes).

        """
        cs =  SkyCoord(
            np.linspace(0, 360, npt) * units.degree,
            np.zeros(npt) * units.degree,
            frame=frame,
        )
        ras, decs = cs.icrs.ra.degree, cs.icrs.dec.degree
        ii = ax.projection_ra(ras).argsort()
        _ = ax.plot(ax.projection_ra(ras[ii]), ax.projection_dec(decs[ii]), **kwargs)

    def custom_plot_sky_circles(ax, ra_center, dec_center, field_of_view, **kwargs):
        """Draw filled circles of fixed angular size on a
        sky-projection axes, correctly handling the RA wraparound at
        the projection edge.

        Similar to ``desiutil.plots.plot_sky_circles`` but propagates
        `**kwargs` through to the fill call.

        Parameters
        ----------
        ax : :class:`matplotlib.axes.Axes`
            Sky-projection axes, exposing ``_ra_center``,
            ``projection_ra``/``projection_dec``.
        ra_center, dec_center : :class:`float` or array-like
            Circle center coordinate(s), in degrees.
        field_of_view : :class:`float`
            Circle diameter, in degrees (declination-corrected in RA).
        **kwargs
            Passed to ``ax.fill``.

        Returns
        -------
        None

        Notes
        -----
        Dead code in the current version of :func:`fig_sky` -- never
        called (see :func:`fig_sky`'s Notes).

        """
        if (isinstance(ra_center, int) | isinstance(ra_center, float)):
            ra_center, dec_center = np.array([ra_center]), np.array([dec_center])
        proj_edge = ax._ra_center - 180
        while proj_edge < 0:
            proj_edge += 360
        #
        angs = np.linspace(2 * np.pi, 0, 101)
        for ra, dec in zip(ra_center, dec_center):
            ras = ra + 0.5 * field_of_view / np.cos(np.radians(dec)) * np.cos(angs)
            decs = dec + 0.5 * field_of_view * np.sin(angs)
            for sel in [ras > proj_edge, ras <= proj_edge]:
                if sel.sum() > 0:
                    ax.fill(ax.projection_ra(ras[sel]), ax.projection_dec(decs[sel]), **kwargs)

    ras = S[racolumn].data
    decs = S[deccolumn].data
    cmap = 'twilight' # 'Blues' # 'vlag'
    #cmap = sns.color_palette(cmap, as_cmap=True)

    ## Make zero values truly white (cmap.Blue(0) = 0.97,0.98,1.0)
    #cmap = mpl.colormaps.get_cmap(cmap).copy()
    #cmap.set_bad(color='white')
    #cmap.set_under(color='white')
    #cmap.set_over(color='white')

    #nside = 16
    #nest = True
    #
    #bin_area = hp.nside2pixarea(nside, degrees=True)
    #print(bin_area)
    #
    #npix = hp.nside2npix(nside)
    #pixels = hp.ang2pix(nside, np.radians(90 - decs), np.radians(ras), nest=nest)
    #counts = np.bincount(pixels, weights=np.ones_like(ras), minlength=npix)
    #
    #grid_data = counts / bin_area
    #grid_data = prepare_data(grid_data, clip_lo=None, clip_hi=None)
    #
    ##cs = np.random.uniform(size=len(S))
    #sc = ax.scatter(ax.projection_ra(ras), ax.projection_dec(decs), s=1)#, c=cs)

    fig = plt.figure(figsize=(10, 7))#, dpi=300)
    ax = fig.add_subplot(111, projection='mollweide')

    ax = init_sky(galactic_plane_color='k', ecliptic_plane_color='none', ax=ax)

    ax, data = plot_sky_binned(ras, decs, plot_type='healpix', max_bin_area=max_bin_area,
                               clip_lo=clip_lo, clip_hi=clip_hi,
                               verbose=True, ax=ax,
                               cmap=cmap, return_grid_data=True, colorbar=False)
    import numpy.ma as ma
    print(ma.min(data), ma.median(data), ma.mean(data), ma.std(data), ma.max(data))
    #print(np.nanpercentile(data, [5., 10., 25., 50., 75., 90., 95.]))

    ax.set_ylabel('Dec (degrees)')
    ax.set_xlabel('RA (degrees)')

    sc = ax.collections[2]
    ar_sky_cbar(ax, sc, r'Galaxy Surface Density (deg$^{-2}$)',
                extend='both', mloc=mloc)#, clip_lo=clip_lo)

    # AR DES, galactic, ecliptic plane
    #desfn = os.path.join(os.getenv("DESI_ROOT"), "survey", "observations", "misc", "des_footprint.txt")
    #plot_des(ax, desfn=desfn, c="orange", lw=0.5, alpha=1, zorder=1)
    #plot_gp_ep(ax, "galactic", c="k", lw=1, alpha=1, zorder=1)
    #plot_gp_ep(ax, "barycentricmeanecliptic", c="k", lw=0.25, alpha=1, ls="--", zorder=1)

    ## AR circle
    #custom_plot_sky_circles(ax, 0, 0, 2 * 20, color="g", facecolor="none")
    #custom_plot_sky_circles(ax, 290, 0, 2 * 20, color="b", alpha=0.5)
    #ax.set_axisbelow(True)

    fig.subplots_adjust(left=0.1, bottom=0.13, right=0.95, top=0.95)

    if pngfile:
        print(f'Writing {pngfile}')
        fig.savefig(pngfile)
        plt.close(fig)



def fig_size_mag(sample, nocuts=False, pngfile=None):
    """Build a two-panel diameter-vs-magnitude QA figure: a 2D
    density/contour plot of log-diameter vs. apparent magnitude
    (``corner.hist2d``), optionally split by literature reference
    when `nocuts` is True.

    Parameters
    ----------
    sample : :class:`astropy.table.Table`
        Parent-sample catalog. If `nocuts`, must contain
        ``DIAM_LIT_REF``, ``DIAM_LIT``, ``MAG_LIT`` (one contour set
        per unique ``DIAM_LIT_REF`` value, ``-99.`` treated as
        missing); otherwise must contain ``MAG`` and ``DIAM``
        (``DIAM > 0`` required).
    nocuts : :class:`bool`
        If True, plot the uncut literature sample split by reference
        catalog; if False, plot the final adopted ``MAG``/``DIAM``
        columns as a single contour set.
    pngfile : :class:`str`, optional
        Output PNG path. If not given, the figure is built but never
        written to disk (see Notes).

    Returns
    -------
    None

    Notes
    -----
    Unconditionally calls ``pdb.set_trace()`` as its last statement,
    after the `pngfile` save block -- this drops into an interactive
    debugger on *every* call, including non-interactive/batch
    invocations. The sole caller (:func:`SGA.parent.qa_parent`, called
    from ``archive/bin-SGA2025/SGA2025-build-parent --qa-parent``)
    would hang waiting for debugger input in a batch/SLURM context. A
    second axes (``ax2``, built via ``fig.add_gridspec(1, 2, ...)`` as
    a diameter histogram panel) is created and given a shared y-axis
    with `ax1`, but nothing is ever plotted into it in the live code
    path -- the histogram-plotting code is entirely commented out, so
    `ax2` renders as a blank panel occupying roughly a third of the
    figure width. Large blocks of alternate (HyperLeda-based) plotting
    logic are commented out rather than removed, suggesting a partial,
    unfinished refactor away from an SGA-2020-style "leda" branch.

    """
    import corner

    sns, colors = plot_style(talk=True, font_scale=1.2)

    #if leda:
    #    good = np.where((sample['D25_LEDA'] > 0) * (sample['MAG_LEDA'] > 0))[0]
    #
    #    isleda = sample['SGA_ID'][good] < 2000000
    #    notleda = sample['SGA_ID'][good] >= 2000000
    #
    #    mag_leda = sample['MAG_LEDA'][good][isleda]
    #    diam_leda = np.log10(sample['D25_LEDA'][good][isleda]) # [arcmin]
    #
    #    mag_notleda = sample['MAG_LEDA'][good][notleda]
    #    diam_notleda = np.log10(sample['D25_LEDA'][good][notleda]) # [arcmin]
    #
    #    xlabel = r'$b_{t}$ (Vega mag)'
    #    ylabel = r'$D_{\mathrm{L}}(25)$ (arcmin)'
    #    #ylabel = r'$D_{i}(25)$ (arcmin)'
    #else:
    #    good = np.where((sample['RADIUS_SB26'] != -1) * (sample['R_MAG_SB26'] != -1))[0]
    #    mag = sample['R_MAG_SB26'][good]
    #    diam = np.log10(sample['RADIUS_SB26'][good]/60) # [arcmin]
    #    xlabel = r'$m_{r}(<R_{26})$ (AB mag)'
    #    ylabel = r'$R_{26}$ (arcmin)'

    xlabel = r'$b_{t}$ (Vega mag)'
    ylabel = r'Diameter (arcmin)'
    #ylabel = r'$D_{\mathrm{L}}(25)$ (arcmin)'
    xlim, ylim = (25, 0), (-2., 2.8)
    #xlim, ylim = (20, 7), (-1, 1.6)

    @ticker.FuncFormatter
    def major_formatter(x, pos):
        """Format a log10(diameter) axis tick `x` as a linear-diameter
        label (one decimal place below 1 arcmin, none at or above),
        for use as a :class:`matplotlib.ticker.FuncFormatter`.

        Parameters
        ----------
        x : :class:`float`
            Tick value, in log10(arcmin).
        pos : :class:`int`
            Tick position (unused, required by the formatter
            callback signature).

        Returns
        -------
        :class:`str`

        """
        if x >= 0:
            return '{:.0f}'.format(10**x)
        else:
            return '{:.1f}'.format(10**x)

    fig = plt.figure(figsize=(14, 7))

    # https://matplotlib.org/stable/gallery/lines_bars_and_markers/scatter_hist.html
    #fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 7), sharey=True)
    gs = fig.add_gridspec(1, 2,  width_ratios=(2, 1.1),
                          #left=0.1, right=0.9, bottom=0.1, top=0.9,
                          wspace=0.05)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)

    if nocuts:
        for iref, ref in enumerate(np.unique(sample['DIAM_LIT_REF'])):
            I = np.where((sample['DIAM_LIT_REF'] == ref) * (sample['DIAM_LIT'] != -99.) * (sample['MAG_LIT'] != -99.))[0]
            if len(I) == 0:
                continue
            mag = sample['MAG_LIT'][I]
            logdiam = np.log10(sample['DIAM_LIT'][I])
            print(ref, len(I), min(mag), max(mag), min(logdiam), max(logdiam))
            corner.hist2d(mag, logdiam, label=ref,
                          levels=[0.5, 0.75, 0.95, 0.995],
                          bins=100, smooth=True, color=colors[iref], ax=ax1, # mpl.cm.get_cmap('viridis'),
                          plot_density=True, fill_contours=True, range=(xlim, ylim),
                          data_kwargs={'color': colors[iref], 'alpha': 0.2, 'ms': 4, 'alpha': 0.5},
                          contour_kwargs={'colors': 'k'},
                          )
            ax1.legend(loc='upper right', fontsize=14) # frameon=False,
    else:
        I = (sample['MAG'] != -99.) * (sample['DIAM'] > 0.)
        mag = sample['MAG'][I]
        logdiam = np.log10(sample['DIAM'][I])
        corner.hist2d(mag, logdiam,
                      levels=[0.5, 0.75, 0.95, 0.995],
                      bins=100, smooth=True, color=colors[0], ax=ax1,
                      plot_density=True, fill_contours=True, range=(xlim, ylim),
                      data_kwargs={'color': colors[0], 'alpha': 0.2,
                                   'ms': 4, 'alpha': 0.5},
                      contour_kwargs={'colors': 'k'},
                      )
    ax1.yaxis.set_major_formatter(major_formatter)
    ax1.set_yticks(np.log10([0.1, 0.2, 0.5, 1, 2, 5, 10, 25, 40]))
    #ax1.legend(loc='upper right', frameon=False)

    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)

    #big = np.where(sample['RADIUS_SB26'][good]/60 > 2)[0]
    #ingc = np.where(['NGC' in gg or 'UGC' in gg for gg in sample['GALAXY'][big]])[0]
    #ingc = np.where(['NGC' in gg for gg in sample['GALAXY'][good]])[0]
    #ax.scatter(rmag[ingc], radius[ingc], marker='s', edgecolors='k',
    #           s=10, alpha=1.0, lw=1, color='k')

    #ax2.hist(diam_leda, orientation='horizontal', range=ylim, bins=75,
    #         color=colors[0], label='HyperLeda')
    #ax2.hist(diam_notleda, orientation='horizontal', range=ylim, bins=75,
    #         color=colors[2], alpha=0.7, label='Supplemental')
    ##ax1.axhline(y=np.log10(20/60), lw=2, ls='-', color='k')
    ##ax2.axhline(y=np.log10(20/60), lw=2, ls='-', color='k', label=r'$D_{\mathrm{L}}(25)=20$ arcsec')
    #ax2.set_xscale('log')
    #ax2.set_xlabel('Number of Galaxies')
    ##ax2.spines[['right', 'top']].set_visible(False)
    #ax2.tick_params(axis='y', labelleft=False)
    #ax2_twin = ax2.twinx()
    #
    #ax2_twin.yaxis.set_major_formatter(major_formatter)
    #ax2_twin.set_yticks(np.log10([0.1, 0.2, 0.5, 1, 2, 5, 10, 25, 40]))
    ##ax2_twin.set_xscale('log')
    #ax2_twin.set_ylim(ylim)
    #ax2_twin.set_ylabel(ylabel, rotation=270, labelpad=25)
    #
    #ax2.legend(loc='upper right', fontsize=14) # frameon=False,
    ##hh, ll = ax2.get_legend_handles_labels()
    ##ax2.legend([hh[0], hh[1], hh[2]], [ll[0], ll[1], ll[2]],
    ##           loc='upper right', fontsize=14) # frameon=False,

    #fig.tight_layout()
    fig.subplots_adjust(bottom=0.18, top=0.95, right=0.9, left=0.1)

    if pngfile:
        print(f'Writing {pngfile}')
        fig.savefig(pngfile)#, bbox_inches='tight')
        plt.close(fig)

    pdb.set_trace()


def draw_ellipse_on_png(im, x0, y0, ba, pa, major_axis_diameter_arcsec,
                        pixscale, color='#3388ff', linewidth=3):
    """Draw an ellipse overlay directly onto a color image opened with
    PIL (in place).

    See :func:`SGA.qa.draw_ellipse` for a more recent implementation
    that works with matplotlib instead.

    Parameters
    ----------
    im : :class:`PIL.Image.Image`
        Color image to draw on, modified in place via ``im.paste``.
    x0 : :class:`float`
        Ellipse center, x pixel coordinate.
    y0 : :class:`float`
        Ellipse center, y pixel coordinate (PIL's origin is top-left,
        so this is typically ``im.size[1] - ycen`` for a
        bottom-left-origin ``ycen``).
    ba : :class:`float`
        Minor-to-major axis ratio.
    pa : :class:`float`
        Astronomical (MGE) position angle, in degrees, counterclockwise
        from the y-axis.
    major_axis_diameter_arcsec : :class:`float`
        Major-axis diameter, in arcsec.
    pixscale : :class:`float`
        Pixel scale, in arcsec/pixel.
    color : :class:`str`
        Ellipse outline color (hex or PIL color name).
    linewidth : :class:`int`
        Ellipse outline width, in pixels.

    Returns
    -------
    None

    Notes
    -----
    Not called anywhere in the active SGA-2025 codebase; the
    similarly-named ``legacyhalos.qa.draw_ellipse_on_png`` (a
    different package, used by ``science/SGA2020/SGA2020-figures``)
    is a separate function despite the shared name.

    Examples
    --------
    >>> with Image.open('image.jpg') as im:
    ...     draw_ellipse_on_png(im, x0, y0, ba, pa,
    ...                          major_axis_diameter_arcsec, pixscale)

    """
    from PIL import Image, ImageDraw, ImageFont

    Image.MAX_IMAGE_PIXELS = None

    minor_axis_diameter_arcsec = major_axis_diameter_arcsec * ba

    overlay_height = int(major_axis_diameter_arcsec / pixscale)
    overlay_width = int(minor_axis_diameter_arcsec / pixscale)
    overlay = Image.new('RGBA', (overlay_width, overlay_height))

    draw = ImageDraw.ImageDraw(overlay)
    box_corners = (0, 0, overlay_width, overlay_height)
    draw.ellipse(box_corners, fill=None, outline=color, width=linewidth)

    rotated = overlay.rotate(pa, expand=True)
    rotated_width, rotated_height = rotated.size
    paste_shift_x = int(x0 - rotated_width / 2)
    paste_shift_y = int(y0 - rotated_height / 2)
    im.paste(rotated, (paste_shift_x, paste_shift_y), rotated)


def qa_binned_radec(cat, nside=64, png=None):
    """Build a HEALPix-binned sky-density map of a catalog, weighted
    by the inverse DESI tiling pixel-coverage fraction.

    Parameters
    ----------
    cat : :class:`astropy.table.Table`
        Catalog with lowercase ``ra``/``dec`` columns, in degrees.
    nside : :class:`int`
        HEALPix ``NSIDE`` resolution parameter.
    png : :class:`str`, optional
        Output PNG path. If not given, the figure is built but never
        written to disk or returned.

    Returns
    -------
    None

    Notes
    -----
    Not called anywhere in the active SGA-2025 codebase. Depends on
    ``desimodel`` (footprint/pixel-weight lookup), a DESI-specific
    package not otherwise used elsewhere in this module.

    """
    import warnings
    import healpy as hp
    import desimodel.io
    import desimodel.footprint
    from desiutil.plots import init_sky, plot_sky_binned

    ra, dec = cat['ra'].data, cat['dec'].data
    hpix = desimodel.footprint.radec2pix(nside, ra, dec)

    fig, ax = plt.subplots(figsize=(9, 5))

    with warnings.catch_warnings():
        pixweight = desimodel.io.load_pixweight(nside)
        fracarea = pixweight[hpix]
        weight = 1 / fracarea

        warnings.simplefilter('ignore')
        basemap = init_sky(galactic_plane_color='k', ax=ax);
        plot_sky_binned(ra, dec, weights=weight,
                        max_bin_area=hp.nside2pixarea(nside, degrees=True),
                        verbose=False, clip_lo='!1', cmap='viridis',
                        plot_type='healpix', basemap=basemap,
                        label=r'$N$(Large Galaxies) / deg$^2$')
        #plt.suptitle('Parent Sample')

    if png:
        fig.savefig(png)


def qa_multiwavelength_coadds(galaxy, galaxydir, htmlgalaxydir, clobber=False,
                              verbose=True):
    """Build two multiwavelength montage QA PNGs for one galaxy via
    ImageMagick's ``montage`` command-line tool: a 3x1 data-only
    montage (GALEX, Legacy Survey, unWISE) and a 3x3
    data/model/residual montage.

    Parameters
    ----------
    galaxy : :class:`str`
        Galaxy name, used to build input/output filenames.
    galaxydir : :class:`str`
        Directory containing the per-band JPEG cutouts (expects files
        named ``{galaxy}-{suffix}.jpg`` for each of the hardcoded
        `suffix` values).
    htmlgalaxydir : :class:`str`
        Output directory for the montage PNGs.
    clobber : :class:`bool`
        If True, regenerate a montage even if its output file already
        exists.
    verbose : :class:`bool`
        If True, print the output filename before writing it.

    Returns
    -------
    None

    Notes
    -----
    Silently skips (without raising) a montage whose required input
    JPEGs are not all present -- only a ``print()`` warning per
    missing file, no exception or log entry. Shells out to the
    external ``montage`` binary (ImageMagick) via ``subprocess.call``;
    if it is not installed or not on ``PATH``, this fails with a bare
    ``FileNotFoundError`` rather than a descriptive error. Not called
    anywhere in the active SGA-2025 codebase (appears to predate the
    current :func:`SGA.html.multiband_montage`-based QA pipeline).

    """

    # Show the data (GALEX, LS, unWISE from left to right).
    montagefile = os.path.join(htmlgalaxydir, '{}-multiwavelength-data.png'.format(galaxy))

    if not os.path.isfile(montagefile) or clobber:
        # Make sure all the files exist.
        check = True
        jpgfile = []
        for suffix in ('image-FUVNUV', 'custom-image', 'image-W1W2'):
            _jpgfile = os.path.join(galaxydir, '{}-{}.jpg'.format(galaxy, suffix))
            jpgfile.append(_jpgfile)
            if not os.path.isfile(_jpgfile):
                print('File {} not found!'.format(_jpgfile))
                check = False

        if check:
            cmd = 'montage -bordercolor white -borderwidth 1 -tile 3x1 -geometry +0+0 -resize 512 '
            cmd = cmd+' '.join(ff for ff in jpgfile)
            cmd = cmd+' {}'.format(montagefile)

            if verbose:
                print('Writing {}'.format(montagefile))
            subprocess.call(cmd.split())

    # Now make a 3x3 montage which has the data, model (no central), residual
    # (just central) from left to right and GALEX, LS, unWISE from top to
    # bottom.
    montagefile = os.path.join(htmlgalaxydir, '{}-multiwavelength-models.png'.format(galaxy))

    if not os.path.isfile(montagefile) or clobber:
        # Make sure all the files exist.
        check = True
        jpgfile = []
        for suffix in ('image-FUVNUV', 'model-nocentral-FUVNUV', 'image-central-FUVNUV',
                       'custom-image', 'custom-model-nocentral', 'custom-image-central',
                       'image-W1W2', 'model-nocentral-W1W2', 'image-central-W1W2'):
            _jpgfile = os.path.join(galaxydir, f'{galaxy}-{suffix}.jpg')
            jpgfile.append(_jpgfile)
            if not os.path.isfile(_jpgfile):
                print('File {} not found!'.format(_jpgfile))
                check = False

        if check:
            cmd = 'montage -bordercolor white -borderwidth 1 -tile 3x3 -geometry +0+0 -resize 512 '
            cmd = cmd+' '.join(ff for ff in jpgfile)
            cmd = cmd+' {}'.format(montagefile)

            if verbose:
                print('Writing {}'.format(montagefile))
            subprocess.call(cmd.split())


def qa_multiwavelength_sed(ellipsefit, tractor=None, png=None, verbose=True):
    """Plot the multiwavelength (GALEX/optical/unWISE) spectral energy
    distribution for one galaxy: total curve-of-growth magnitude,
    magnitude within R(25), and (optionally) Tractor photometry, vs.
    observed-frame wavelength.

    Parameters
    ----------
    ellipsefit : :class:`dict`
        Ellipse-fitting results for one galaxy (SGA-2020-era data
        model), with keys ``success``, ``sma_r``, ``bands``,
        ``refband``, ``ra_moment``, ``dec_moment``, and per-band
        ``cog_mtot_{filt}``, ``flux_sb25_{filt}``,
        ``flux_ivar_sb25_{filt}``. May optionally have ``redshift``.
    tractor : :class:`dict` or :class:`astropy.table.Row`, optional
        Tractor photometry with per-band ``flux_{filt}``/
        ``flux_ivar_{filt}`` keys/columns. If None, Tractor points are
        omitted from the plot.
    png : :class:`str`, optional
        Output PNG path. If None, the figure is displayed interactively
        (``plt.show()``) instead of being saved.
    verbose : :class:`bool`
        Unused (see Notes).

    Returns
    -------
    None

    Notes
    -----
    Broken: calls an undefined function ``_sbprofile_colors`` (note
    the leading underscore -- the actually-defined function in this
    module is :func:`sbprofile_colors`, with no leading underscore and
    a different, no-argument signature) -- always raises ``NameError``.
    Separately, if ``'redshift' in ellipsefit.keys()``, this function
    also references an undefined name ``cosmo`` (``SGA.misc.arcsec2kpc(
    redshift, cosmo=cosmo)``) -- also always raises ``NameError`` on
    that code path. `verbose` is accepted but never referenced in the
    function body. This function, along with the five others in this
    docstring batch (:func:`ellipse_sbprofile`,
    :func:`display_ellipse_sbprofile`, :func:`display_ellipsefit`,
    :func:`qa_curveofgrowth`, :func:`display_multiband`), is not called
    anywhere else in the codebase and duplicates functionality now
    provided by ``SGA.html``'s SGA-2025 QA functions (e.g.
    :func:`SGA.html.ellipse_sed`) -- it appears to be orphaned
    SGA-2020-era code, left in place but unreachable in the current
    pipeline.

    """
    from copy import deepcopy
    import matplotlib.ticker as ticker
    from astropy.table import Table

    if ellipsefit['success'] is False or np.atleast_1d(ellipsefit['sma_r'])[0] == -1:
        return

    bands, refband = ellipsefit['bands'], ellipsefit['refband']

    galex = 'FUV' in bands
    unwise = 'W1' in bands
    colors = _sbprofile_colors(galex=galex, unwise=unwise)

    if 'redshift' in ellipsefit.keys():
        redshift = ellipsefit['redshift']
        smascale = SGA.misc.arcsec2kpc(redshift, cosmo=cosmo) # [kpc/arcsec]
    else:
        redshift, smascale = None, None

    # see also Morrisey+05
    effwave_north = {'fuv': 1528.0, 'nuv': 2271.0,
                     'w1': 34002.54044482, 'w2': 46520.07577119, 'w3': 128103.3789599, 'w4': 223752.7751558,
                     'g': 4815.95363513, 'r': 6437.79282937, 'i': 7847.78249813, 'z': 9229.65786449}
    effwave_south = {'fuv': 1528.0, 'nuv': 2271.0,
                     'w1': 34002.54044482, 'w2': 46520.07577119, 'w3': 128103.3789599, 'w4': 223752.7751558,
                     'g': 4890.03670428, 'r': 6469.62203811, 'i': 7847.78249813, 'z': 9196.46396394}

    _tt = Table()
    _tt['RA'] = [ellipsefit['ra_moment']]
    _tt['DEC'] = [ellipsefit['dec_moment']]
    run = SGA.io.get_run(_tt)

    if run == 'north':
        effwave = effwave_north
    else:
        effwave = effwave_south

    # build the arrays
    nband = len(bands)
    bandwave = np.array([effwave[filt.lower()] for filt in bands])

    _phot = {'abmag': np.zeros(nband, 'f4')-1,
             'abmagerr': np.zeros(nband, 'f4')+0.5,
             'lower': np.zeros(nband, bool)}
    phot = {'mag_tot': deepcopy(_phot), 'tractor': deepcopy(_phot), 'mag_sb25': deepcopy(_phot)}

    for ifilt, filt in enumerate(bands):
        mtot = ellipsefit['cog_mtot_{}'.format(filt.lower())]
        if mtot > 0:
            phot['mag_tot']['abmag'][ifilt] = mtot
            phot['mag_tot']['abmagerr'][ifilt] = 0.1
            phot['mag_tot']['lower'][ifilt] = False

        flux = ellipsefit['flux_sb25_{}'.format(filt.lower())]
        ivar = ellipsefit['flux_ivar_sb25_{}'.format(filt.lower())]
        #print(filt, mag)

        if flux > 0 and ivar > 0:
            mag = 22.5 - 2.5 * np.log10(flux)
            ferr = 1.0 / np.sqrt(ivar)
            magerr = 2.5 * ferr / flux / np.log(10)
            phot['mag_sb25']['abmag'][ifilt] = mag
            phot['mag_sb25']['abmagerr'][ifilt] = magerr
            phot['mag_sb25']['lower'][ifilt] = False
        if flux <=0 and ivar > 0:
            ferr = 1.0 / np.sqrt(ivar)
            mag = 22.5 - 2.5 * np.log10(ferr)
            phot['mag_sb25']['abmag'][ifilt] = mag
            phot['mag_sb25']['abmagerr'][ifilt] = 0.75
            phot['mag_sb25']['lower'][ifilt] = True

        if tractor is not None:
            flux = tractor['flux_{}'.format(filt.lower())]
            ivar = tractor['flux_ivar_{}'.format(filt.lower())]
            #if filt == 'FUV':
            #    pdb.set_trace()
            if flux > 0 and ivar > 0:
                phot['tractor']['abmag'][ifilt] = 22.5 - 2.5 * np.log10(flux)
                phot['tractor']['abmagerr'][ifilt] = 0.1
            if flux <= 0 and ivar > 0:
                phot['tractor']['abmag'][ifilt] = 22.5 - 2.5 * np.log10(1/np.sqrt(ivar))
                phot['tractor']['abmagerr'][ifilt] = 0.75
                phot['tractor']['lower'][ifilt] = True

    #print(phot['mag_tot']['abmag'])
    #print(phot['mag_sb25']['abmag'])
    #print(phot['tractor']['abmag'])

    def _addphot(thisphot, color, marker, alpha, label):
        """Plot one photometry series as error bars, splitting points
        into lower-limit markers (no legend label) and well-constrained
        markers (with legend label).

        Parameters
        ----------
        thisphot : :class:`dict`
            Per-band photometry with ``abmag``, ``abmagerr``, ``lower``
            arrays (see `phot`, built in the enclosing scope).
        color : color-like
            Marker/error-bar color.
        marker : :class:`str`
            Matplotlib marker style.
        alpha : :class:`float`
            Marker/error-bar transparency.
        label : :class:`str`
            Legend label for the well-constrained points.

        Returns
        -------
        None

        Notes
        -----
        Closes over `ax` and `bandwave` from the enclosing
        :func:`qa_multiwavelength_sed` scope rather than taking them
        as parameters.

        """
        good = np.where((thisphot['abmag'] > 0) * (thisphot['lower'] == True))[0]
        if len(good) > 0:
            ax.errorbar(bandwave[good]/1e4, thisphot['abmag'][good], yerr=thisphot['abmagerr'][good],
                        marker=marker, markersize=11, markeredgewidth=3, markeredgecolor='k',
                        markerfacecolor=color, elinewidth=3, ecolor=color, capsize=4,
                        lolims=True, linestyle='none', alpha=alpha)#, lolims=True)
        good = np.where((thisphot['abmag'] > 0) * (thisphot['lower'] == False))[0]
        if len(good) > 0:
            ax.errorbar(bandwave[good]/1e4, thisphot['abmag'][good], yerr=thisphot['abmagerr'][good],
                        marker=marker, markersize=11, markeredgewidth=3, markeredgecolor='k',
                        markerfacecolor=color, elinewidth=3, ecolor=color, capsize=4,
                        label=label, linestyle='none', alpha=alpha)

    # make the plot
    fig, ax = plt.subplots(figsize=(9, 7))

    # get the plot limits
    good = np.where(phot['mag_tot']['abmag'] > 0)[0]
    ymax = np.min(phot['mag_tot']['abmag'][good])
    ymin = np.max(phot['mag_tot']['abmag'][good])

    good = np.where(phot['tractor']['abmag'] > 0)[0]
    if np.min(phot['tractor']['abmag'][good]) < ymax:
        ymax = np.min(phot['tractor']['abmag'][good])
    if np.max(phot['tractor']['abmag']) > ymin:
        ymin = np.max(phot['tractor']['abmag'][good])
    #print(ymin, ymax)

    ymin += 1.5
    ymax -= 1.5

    wavemin, wavemax = 0.1, 30

    # have to set the limits before plotting since the axes are reversed
    if np.abs(ymax-ymin) > 15:
        ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.set_ylim(ymin, ymax)
    _addphot(phot['mag_tot'], color='red', marker='s', alpha=1.0, label=r'$m_{\mathrm{tot}}$')
    _addphot(phot['mag_sb25'], color='orange', marker='^', alpha=0.9, label=r'$m(r<R_{25})$')
    _addphot(phot['tractor'], color='blue', marker='o', alpha=0.75, label='Tractor')

    #thisphot = phot['tractor']
    #color='blue'
    #marker='o'
    #label='Tractor'

    #good = np.where((thisphot['abmag'] > 0) * (thisphot['lower'] == False))[0]
    #if len(good) > 0:
    #    ax.errorbar(bandwave[good]/1e4, thisphot['abmag'][good], yerr=thisphot['abmagerr'][good],
    #                marker=marker, markersize=11, markeredgewidth=3, markeredgecolor='k',
    #                markerfacecolor=color, elinewidth=3, ecolor=color, capsize=4,
    #                label=label, linestyle='none')

    #good = np.where((thisphot['abmag'] > 0) * (thisphot['lower'] == True))[0]
    ##ax.errorbar(bandwave[good]/1e4, thisphot['abmag'][good], yerr=0.5, #thisphot['abmagerr'][good],
    ##            marker='o', uplims=thisphot['lower'][good], linestyle='none')
    #if len(good) > 0:
    #    ax.errorbar(bandwave[good]/1e4, thisphot['abmag'][good], yerr=0.5, #thisphot['abmagerr'][good][0],
    #                marker=marker, markersize=11, markeredgewidth=3, markeredgecolor='k',
    #                markerfacecolor=color, elinewidth=3, ecolor=color, capsize=4,
    #                uplims=thisphot['lower'][good], linestyle='none')#, lolims=True)

    ax.set_xlabel(r'Observed-frame Wavelength ($\mu$m)')
    ax.set_ylabel(r'Apparent Brightness (AB mag)')
    ax.set_xlim(wavemin, wavemax)
    ax.set_xscale('log')
    ax.legend(loc='lower right')

    def _frmt(value, _):
        """Format an x-axis tick `value` (in microns) as a
        :class:`matplotlib.ticker.FuncFormatter` callback: one decimal
        place below 1, none at or above.

        Parameters
        ----------
        value : :class:`float`
            Tick value to format.
        _ : :class:`int`
            Tick position (unused, required by the ``FuncFormatter``
            callback signature).

        Returns
        -------
        :class:`str`

        """
        if value < 1:
            return '{:.1f}'.format(value)
        else:
            return '{:.0f}'.format(value)

    #ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
    ax.set_xticks([0.1, 0.2, 0.4, 1.0, 3.0, 5.0, 10, 20])
    ax.xaxis.set_major_formatter(plt.FuncFormatter(_frmt))

    if smascale:
        fig.subplots_adjust(left=0.14, bottom=0.15, top=0.85, right=0.95)
        #fig.subplots_adjust(left=0.12, bottom=0.15, top=0.85, right=0.88)
    else:
        fig.subplots_adjust(left=0.14, bottom=0.15, top=0.95, right=0.95)
        #fig.subplots_adjust(left=0.12, bottom=0.15, top=0.95, right=0.88)

    if png:
        #if verbose:
        print('Writing {}'.format(png))
        fig.savefig(png)
        plt.close(fig)
    else:
        plt.show()


def ellipse_sbprofile(ellipsefit, minerr=0.0):
    """Convert isophote-fitting results into surface-brightness,
    color, and semi-major-axis profiles for one galaxy.

    Parameters
    ----------
    ellipsefit : :class:`dict`
        Ellipse-fitting results (SGA-2020-era data model) with keys
        ``band``, ``refband``, ``pixscale``, ``redshift``, per-band
        ``psfsigma_{filt}``, ``{filt}`` (a
        :class:`photutils.isophote.IsophoteList`-like object with
        ``intens``/``int_err`` arrays), and ``r`` (the reference-band
        isophote list, for ``sma``).
    minerr : :class:`float`
        Minimum surface-brightness uncertainty floor, in mag/arcsec2,
        added in quadrature to the propagated photometric error.

    Returns
    -------
    :class:`dict`
        Profile dictionary with ``sma`` (arcsec), ``redshift``,
        ``minerr``, ``smaunit``, per-band ``mu_{filt}``/
        ``mu_{filt}_err`` (surface brightness, mag/arcsec2), and (when
        the relevant bands are present) ``gr``/``gr_err``,
        ``rz``/``rz_err``, ``ri``/``ri_err`` color profiles.

    Notes
    -----
    Not called anywhere in the codebase outside this module (only
    from the similarly-orphaned :func:`display_ellipse_sbprofile`);
    see the orphaned-code note in :func:`qa_multiwavelength_sed`.

    """
    band, refband = ellipsefit['band'], ellipsefit['refband']
    pixscale, redshift = ellipsefit['pixscale'], ellipsefit['redshift']

    indx = np.ones(len(ellipsefit[refband]), dtype=bool)

    sbprofile = dict()
    for filt in band:
        sbprofile['psfsigma_{}'.format(filt)] = ellipsefit['psfsigma_{}'.format(filt)]
    sbprofile['redshift'] = redshift
    sbprofile['minerr'] = minerr
    sbprofile['smaunit'] = 'arcsec'
    sbprofile['sma'] = ellipsefit['r'].sma[indx] * pixscale # [arcsec]

    # Create a pixel scale mapping to accommodate GALEX and unWISE imaging.
    #filt2pixscalefactor = {'g': 1.0, 'r': 1.0, 'z': 1.0}
    #if 'NUV' in band:
    #    sbprofile['sma_galex'] = ellipsefit['r'].sma * ellipsefit['galex_pixscale'] / pixscale # [arcsec]
    #if 'W1' in band:
    #    sbprofile['sma_unwise'] = ellipsefit['r'].sma * ellipsefit['unwise_pixscale'] / pixscale # [arcsec]

    with np.errstate(invalid='ignore'):
        for filt in band:
            #area = ellipsefit[filt].sarea[indx] * pixscale**2
            sbprofile['mu_{}'.format(filt)] = 22.5 - 2.5 * np.log10(ellipsefit[filt].intens[indx])

            #sbprofile[filt] = 22.5 - 2.5 * np.log10(ellipsefit[filt].intens[indx])
            sbprofile['mu_{}_err'.format(filt)] = 2.5 * ellipsefit[filt].int_err[indx] / \
              ellipsefit[filt].intens[indx] / np.log(10)
            sbprofile['mu_{}_err'.format(filt)] = np.sqrt(sbprofile['mu_{}_err'.format(filt)]**2 + minerr**2)

            # Just for the plot use a minimum uncertainty
            #sbprofile['{}_err'.format(filt)][sbprofile['{}_err'.format(filt)] < minerr] = minerr

    if 'mu_g' in sbprofile.keys() and 'mu_r' in sbprofile.keys():
        sbprofile['gr'] = sbprofile['mu_g'] - sbprofile['mu_r']
        sbprofile['gr_err'] = np.sqrt(sbprofile['mu_g_err']**2 + sbprofile['mu_r_err']**2)
    if 'mu_r' in sbprofile.keys() and 'mu_z' in sbprofile.keys():
        sbprofile['rz'] = sbprofile['mu_r'] - sbprofile['mu_z']
        sbprofile['rz_err'] = np.sqrt(sbprofile['mu_r_err']**2 + sbprofile['mu_z_err']**2)
    if 'mu_r' in sbprofile.keys() and 'mu_i' in sbprofile.keys():
        sbprofile['ri'] = sbprofile['mu_r'] - sbprofile['mu_i']
        sbprofile['ri_err'] = np.sqrt(sbprofile['mu_r_err']**2 + sbprofile['mu_i_err']**2)

    return sbprofile


def display_ellipse_sbprofile(ellipsefit, skyellipsefit={}, minerr=0.0,
                              smascale=None, png=None, verbose=True):
    """Plot the multiwavelength surface-brightness and color profiles
    for one galaxy, optionally overlaid with a sky-variance estimate.

    Parameters
    ----------
    ellipsefit : :class:`dict`
        Ellipse-fitting results, as consumed by
        :func:`ellipse_sbprofile`; also needs ``success``.
    skyellipsefit : :class:`dict`
        Optional sky-annulus ellipse-fitting results with ``sma`` and
        per-band flux arrays, used to overplot a dashed sky-variance
        curve (via ``astropy.stats.mad_std``). Empty dict (falsy) by
        default, which skips the sky overlay entirely.
    minerr : :class:`float`
        Minimum surface-brightness uncertainty floor, passed to
        :func:`ellipse_sbprofile`.
    smascale : :class:`float`, optional
        Arcsec-to-kpc conversion factor for the twin top x-axis. If
        None, computed via ``SGA.misc.arcsec2kpc(redshift)`` (see
        Notes).
    png : :class:`str`, optional
        Output PNG path. If None, the figure is displayed interactively.
    verbose : :class:`bool`
        If True, print the output path when writing `png`.

    Returns
    -------
    None

    Notes
    -----
    Broken: calls an undefined function ``_sbprofile_colors()`` (see
    the same anomaly in :func:`qa_multiwavelength_sed`) -- always
    raises ``NameError`` before any plotting happens. Even with a
    correct name, the call site does ``colors = _sbprofile_colors()``
    followed by ``next(colors)`` in a loop, but the analogous,
    actually-defined :func:`sbprofile_colors` returns a plain
    :class:`dict`, not an iterator -- ``next()`` on a dict raises
    ``TypeError``, so this call pattern would still be broken even if
    the name were fixed. If `smascale` is left as `None` (default),
    computes it via ``SGA.misc.arcsec2kpc(redshift)`` without a
    `cosmo` argument, which itself always raises ``AttributeError``
    (see the ``misc.py`` docstring sweep's note on
    :func:`SGA.misc.arcsec2kpc`). See also the general orphaned-code
    note in :func:`qa_multiwavelength_sed`.

    """
    import astropy.stats
    #from legacyhalos.ellipse import ellipse_sbprofile

    if ellipsefit['success']:
        sbprofile = ellipse_sbprofile(ellipsefit, minerr=minerr)

        band, refband = ellipsefit['band'], ellipsefit['refband']
        redshift, pixscale = ellipsefit['redshift'], ellipsefit['pixscale']
        if smascale is None:
            smascale = SGA.misc.arcsec2kpc(redshift) # [kpc/arcsec]

        #if png:
        #    sbfile = png.replace('.png', '.txt')
        #    legacyhalos.io.write_sbprofile(sbprofile, smascale, sbfile)

        yminmax = [40, 0]
        xminmax = [0, 0]
        colors = _sbprofile_colors()

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True,
                                       gridspec_kw = {'height_ratios':[1, 0.5]})

        for filt in band:
            sma = sbprofile['sma']
            mu = sbprofile['mu_{}'.format(filt)]
            muerr = sbprofile['mu_{}_err'.format(filt)]

            #good = (ellipsefit[filt].stop_code < 4)
            #bad = ~good

            #with np.errstate(invalid='ignore'):
            #    good = np.isfinite(mu) * (mu / muerr > 2)
            good = np.isfinite(mu) * (muerr < 0.5)

            sma = sma[good]
            mu = mu[good]
            muerr = muerr[good]

            col = next(colors)
            ax1.fill_between(sma, mu-muerr, mu+muerr, label=r'${}$'.format(filt), color=col,
                             alpha=0.75, edgecolor='k', lw=2)

            if np.nanmin(mu-muerr) < yminmax[0]:
                yminmax[0] = np.nanmin(mu-muerr)
            if np.nanmax(mu+muerr) > yminmax[1]:
                yminmax[1] = np.nanmax(mu+muerr)
            if np.nanmax(sma) > xminmax[1]:
                xminmax[1] = np.nanmax(sma)

            if bool(skyellipsefit):
                skysma = skyellipsefit['sma'] * pixscale

                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    sky = astropy.stats.mad_std(skyellipsefit[filt], axis=1, ignore_nan=True)
                    # sky = np.nanstd(skyellipsefit[filt], axis=1) # / np.sqrt(skyellipsefit[

                skygood = np.isfinite(sky)
                skysma = skysma[skygood]
                skymu = 22.5 - 2.5 * np.log10(sky[skygood])
                ax1.plot( skysma, skymu , color=col, ls='--', alpha=0.75)
                if skymu.max() > yminmax[1]:
                    yminmax[1] = skymu.max()

                ax1.text(0.05, 0.04, 'Sky Variance', ha='left', va='center',
                         transform=ax1.transAxes, fontsize=12)

            #ax1.axhline(y=ellipsefit['mu_{}_sky'.format(filt)], color=col, ls='--')
            #if filt == refband:
            #    ysky = ellipsefit['mu_{}_sky'.format(filt)] - 2.5 * np.log10(0.1) # 10% of sky
            #    ax1.axhline(y=ysky, color=col, ls='--')

        ax1.set_ylabel(r'$\mu(a)$ (mag arcsec$^{-2}$)')
        #ax1.set_ylabel(r'Surface Brightness $\mu(a)$ (mag arcsec$^{-2}$)')
        #ax1.set_ylabel(r'Surface Brightness $\mu(r)$ (mag arcsec$^{-2}$)')

        ylim = [yminmax[0]-0.5, yminmax[1]+0.75]
        if ylim[0] < 17:
            ylim[0] = 10 # 17
        if ylim[1] > 32.5:
            ylim[1] = 35 # 32.5
        ax1.set_ylim(ylim)
        ax1.invert_yaxis()

        xlim = [xminmax[0], xminmax[1]*1.01]
        #ax1.set_xlim(xmin=0)
        #ax1.margins(xmargin=0)

        #ax1.set_ylabel(r'$\mu$ (mag arcsec$^{-2}$)')
        #ax1.set_ylim(31.99, 18)

        ax1_twin = ax1.twiny()
        ax1_twin.set_xlim( (xlim[0]*smascale, xlim[1]*smascale) )
        ax1_twin.set_xlabel('Semi-major Axis $a$ (kpc)')

        ax1.legend(loc='upper right', ncol=1)

        # color vs semi-major axis
        ax2.fill_between(sbprofile['sma'],
                         sbprofile['gr'] - sbprofile['gr_err'],
                         sbprofile['gr'] + sbprofile['gr_err'],
                         label=r'$g - r$', color=next(colors), alpha=0.75,
                         edgecolor='k', lw=2)

        if 'rz' in sbprofile.keys():
            ax2.fill_between(sbprofile['sma'],
                             sbprofile['rz'] - sbprofile['rz_err'],
                             sbprofile['rz'] + sbprofile['rz_err'],
                             label=r'$r - z$', color=next(colors), alpha=0.75,
                             edgecolor='k', lw=2)
        elif 'ri' in sbprofile.keys():
            ax2.fill_between(sbprofile['sma'],
                             sbprofile['ri'] - sbprofile['ri_err'],
                             sbprofile['ri'] + sbprofile['ri_err'],
                             label=r'$r - i$', color=next(colors), alpha=0.75,
                             edgecolor='k', lw=2)

        ax2.set_xlabel(r'Semi-major Axis $a$ (arcsec)')
        #ax2.set_xlabel(r'Galactocentric radius $r$ (arcsec)')
        #ax2.legend(loc='upper left')
        ax2.legend(bbox_to_anchor=(0.25, 0.99))

        ax2.set_ylabel('Color (mag)')
        ax2.set_ylim(-0.5, 2.8)

        for xx in (ax1, ax2):
            xx.set_xlim(xlim)
            ylim = xx.get_ylim()
            xx.fill_between([0, 3*ellipsefit['psfsigma_r']*ellipsefit['pixscale']], [ylim[0], ylim[0]],
                            [ylim[1], ylim[1]], color='grey', alpha=0.1)

        ax2.text(0.03, 0.09, r'PSF\n(3$\sigma$)', ha='center', va='center',
            transform=ax2.transAxes, fontsize=10)

        fig.subplots_adjust(hspace=0.0)

        if png:
            if verbose:
                print('Writing {}'.format(png))
            fig.savefig(png)
            plt.close(fig)
        else:
            plt.show()


def display_ellipsefit(ellipsefit, xlog=False, png=None, verbose=True):
    """Plot the raw isophote-fitting diagnostics (ellipticity, position
    angle, and x/y center) vs. semi-major axis for the reference band.

    Parameters
    ----------
    ellipsefit : :class:`dict`
        Ellipse-fitting results with ``success``, ``band``,
        ``refband``, ``pixscale``, ``redshift``, and a reference-band
        isophote-list object exposing ``sma``, ``stop_code``, ``eps``/
        ``ellip_err``, ``pa``/``pa_err``, ``x0``/``x0_err``, ``y0``/
        ``y0_err``.
    xlog : :class:`bool`
        If True, use a log-scaled x-axis (semi-major axis) on all four
        panels.
    png : :class:`str`, optional
        Output PNG path. If None, the figure is displayed interactively.
    verbose : :class:`bool`
        If True, print the output path when writing `png`.

    Returns
    -------
    None

    Notes
    -----
    Broken: references ``legacyhalos.misc.arcsec2kpc`` -- the
    ``legacyhalos`` module (an ancestor/sibling package this codebase
    evolved from) is never imported anywhere in this file, so this
    always raises ``NameError`` when ``ellipsefit['success']`` is
    True. Also uses ``sns.color_palette()`` (line ``colors =
    iter(sns.color_palette())``) without importing ``seaborn as sns``
    locally or at module level (other functions in this module import
    it locally, but this one does not) -- an independent ``NameError``
    that would fire even before the ``legacyhalos`` one. See also the
    general orphaned-code note in :func:`qa_multiwavelength_sed`.

    """

    from matplotlib.ticker import FormatStrFormatter, ScalarFormatter

    colors = iter(sns.color_palette())

    if ellipsefit['success']:
        band, refband = ellipsefit['band'], ellipsefit['refband']
        pixscale, redshift = ellipsefit['pixscale'], ellipsefit['redshift']
        smascale = legacyhalos.misc.arcsec2kpc(redshift) # [kpc/arcsec]

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 9), sharex=True)

        good = (ellipsefit[refband].stop_code < 4)
        bad = ~good
        ax1.fill_between(ellipsefit[refband].sma[good] * pixscale,
                         ellipsefit[refband].eps[good]-ellipsefit[refband].ellip_err[good],
                         ellipsefit[refband].eps[good]+ellipsefit[refband].ellip_err[good])#,
                         #edgecolor='k', lw=2)
        if np.count_nonzero(bad) > 0:
            ax1.scatter(ellipsefit[refband].sma[bad] * pixscale, ellipsefit[refband].eps[bad],
                        marker='s', s=40, edgecolor='k', lw=2, alpha=0.75)

        #ax1.errorbar(ellipsefit[refband].sma[good] * smascale,
        #             ellipsefit[refband].eps[good],
        #             ellipsefit[refband].ellip_err[good], fmt='o',
        #             markersize=4)#, color=color[refband])
        #ax1.set_ylim(0, 0.5)
        ax1.xaxis.set_major_formatter(ScalarFormatter())

        ax2.fill_between(ellipsefit[refband].sma[good] * pixscale,
                         np.degrees(ellipsefit[refband].pa[good]-ellipsefit[refband].pa_err[good]),
                         np.degrees(ellipsefit[refband].pa[good]+ellipsefit[refband].pa_err[good]))#,
                         #edgecolor='k', lw=2)
        if np.count_nonzero(bad) > 0:
            ax2.scatter(ellipsefit[refband].sma[bad] * pixscale, np.degrees(ellipsefit[refband].pa[bad]),
                        marker='s', s=40, edgecolor='k', lw=2, alpha=0.75)
        #ax2.errorbar(ellipsefit[refband].sma[good] * smascale,
        #             np.degrees(ellipsefit[refband].pa[good]),
        #             np.degrees(ellipsefit[refband].pa_err[good]), fmt='o',
        #             markersize=4)#, color=color[refband])
        #ax2.set_ylim(0, 180)

        ax3.fill_between(ellipsefit[refband].sma[good] * pixscale,
                         ellipsefit[refband].x0[good]-ellipsefit[refband].x0_err[good],
                         ellipsefit[refband].x0[good]+ellipsefit[refband].x0_err[good])#,
                         #edgecolor='k', lw=2)
        if np.count_nonzero(bad) > 0:
            ax3.scatter(ellipsefit[refband].sma[bad] * pixscale, ellipsefit[refband].x0[bad],
                        marker='s', s=40, edgecolor='k', lw=2, alpha=0.75)
        #ax3.errorbar(ellipsefit[refband].sma[good] * smascale, ellipsefit[refband].x0[good],
        #             ellipsefit[refband].x0_err[good], fmt='o',
        #             markersize=4)#, color=color[refband])
        ax3.xaxis.set_major_formatter(ScalarFormatter())
        ax3.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        ax4.fill_between(ellipsefit[refband].sma[good] * pixscale,
                         ellipsefit[refband].y0[good]-ellipsefit[refband].y0_err[good],
                         ellipsefit[refband].y0[good]+ellipsefit[refband].y0_err[good])#,
                         #edgecolor='k', lw=2)
        if np.count_nonzero(bad) > 0:
            ax4.scatter(ellipsefit[refband].sma[bad] * pixscale, ellipsefit[refband].y0[bad],
                        marker='s', s=40, edgecolor='k', lw=2, alpha=0.75)
        #ax4.errorbar(ellipsefit[refband].sma[good] * smascale, ellipsefit[refband].y0[good],
        #             ellipsefit[refband].y0_err[good], fmt='o',
        #             markersize=4)#, color=color[refband])

        ax2.yaxis.tick_right()
        ax2.yaxis.set_label_position('right')
        ax2.yaxis.set_major_formatter(ScalarFormatter())
        ax4.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        ax4.yaxis.tick_right()
        ax4.yaxis.set_label_position('right')
        ax4.xaxis.set_major_formatter(ScalarFormatter())
        ax4.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        for xx in (ax1, ax2, ax3, ax4):
            xx.set_xlim(xmin=0)

        xlim = ax1.get_xlim()
        ax1_twin = ax1.twiny()
        ax1_twin.set_xlim( (xlim[0]*smascale, xlim[1]*smascale) )
        ax1_twin.set_xlabel('Galactocentric radius (kpc)')

        ax2_twin = ax2.twiny()
        ax2_twin.set_xlim( (xlim[0]*smascale, xlim[1]*smascale) )
        ax2_twin.set_xlabel('Galactocentric radius (kpc)')

        ax1.set_ylabel(r'Ellipticity $\epsilon$')
        ax2.set_ylabel('Position Angle (deg)')
        ax3.set_xlabel(r'Galactocentric radius $r$ (arcsec)')
        ax3.set_ylabel(r'$x$ Center')
        ax4.set_xlabel(r'Galactocentric radius $r$ (arcsec)')
        ax4.set_ylabel(r'$y$ Center')

        if xlog:
            for xx in (ax1, ax2, ax3, ax4):
                xx.set_xscale('log')

        fig.subplots_adjust(hspace=0.03, wspace=0.03, bottom=0.15, right=0.85, left=0.15)

        if png:
            if verbose:
                print('Writing {}'.format(png))
            fig.savefig(png)
            plt.close(fig)
        else:
            plt.show()


def qa_curveofgrowth(ellipsefit, png=None, verbose=True):
    """Plot the cumulative (curve-of-growth) brightness vs. semi-major
    axis for each band.

    Parameters
    ----------
    ellipsefit : :class:`dict`
        Ellipse-fitting results with ``band``, ``refband``,
        ``redshift``, and per-band ``apphot_sma_{filt}``/
        ``apphot_mag_{filt}`` (despite the name, the latter holds flux,
        not magnitude -- see Notes) arrays.
    png : :class:`str`, optional
        Output PNG path. If given, the figure is saved (but, unlike
        its sibling functions in this module, never closed or shown
        otherwise -- see Notes).
    verbose : :class:`bool`
        Unused (see Notes).

    Returns
    -------
    None

    Notes
    -----
    `verbose` is accepted but never referenced in the function body.
    Calls ``SGA.misc.arcsec2kpc(redshift)`` without a `cosmo`
    argument, which always raises ``AttributeError`` (see
    :func:`SGA.misc.arcsec2kpc`'s docstring). The
    ``apphot_mag_{filt}`` key is treated as a flux (``mag = 22.5 -
    2.5*np.log10(flux[good])``) despite its name. Unlike the other
    ``png``-writing functions in this module, if `png` is given the
    figure is saved but the function neither calls ``plt.close(fig)``
    nor prints a "Writing" message -- a minor inconsistency with the
    rest of the module's convention. See also the general
    orphaned-code note in :func:`qa_multiwavelength_sed`.

    """
    fig, ax = plt.subplots(figsize=(9, 7))
    band, refband, redshift = ellipsefit['band'], ellipsefit['refband'], ellipsefit['redshift']

    maxsma = ellipsefit['apphot_sma_{}'.format(refband)].max()
    smascale = SGA.misc.arcsec2kpc(redshift) # [kpc/arcsec]

    yfaint, ybright = 0, 50
    for filt in band:
        flux = ellipsefit['apphot_mag_{}'.format(filt)]
        good = np.where( np.isfinite(flux) * (flux > 0) )[0]
        sma = ellipsefit['apphot_sma_{}'.format(filt)][good]
        mag = 22.5-2.5*np.log10(flux[good])
        ax.plot(sma, mag, label=filt)

        #print(filt, np.mean(mag[-5:]))
        #print(filt, mag[-5:], np.mean(mag[-5:])
        #print(filt, np.min(mag))

        if mag.max() > yfaint:
            yfaint = mag.max()
        if mag.min() < ybright:
            ybright = mag.min()

    ax.set_xlabel(r'Semi-major Axis $a$ (arcsec)')
    ax.set_ylabel('Cumulative Brightness (AB mag)')

    ax.set_xlim(0, maxsma)
    ax_twin = ax.twiny()
    ax_twin.set_xlim( (0, maxsma * smascale) )
    ax_twin.set_xlabel('Semi-major Axis $a$ (kpc)')

    yfaint += 0.5
    ybright += -0.5

    ax.set_ylim(yfaint, ybright)
    ax_twin = ax.twinx()
    ax_twin.set_ylim(yfaint, ybright)
    ax_twin.set_ylabel('Cumulative Brightness (AB mag)')#, rotation=-90)

    ax.legend(loc='lower right', fontsize=14, ncol=3)

    fig.subplots_adjust(left=0.12, bottom=0.15, top=0.85, right=0.88)

    if png:
        fig.savefig(png)


def display_multiband(data, geometry=None, mgefit=None, ellipsefit=None, indx=None,
                      magrange=10, inchperband=3, contours=False, png=None,
                      verbose=True, vertical=False):
    """Display multiband cutout images with optional MGE- and/or
    ellipse-fit isophote overlays.

    Parameters
    ----------
    data : :class:`dict`
        Per-band cutout data with ``band`` (list of filter names) and
        per-band ``{filt}_masked`` image arrays.
    geometry : :class:`photutils.isophote.EllipseGeometry`, optional
        Single elliptical-aperture geometry to overplot on every
        panel, if `ellipsefit` is not given.
    mgefit : :class:`dict`, optional
        Per-band MGE (Multi-Gaussian Expansion) fit results with
        ``{filt}.sol``, ``xpeak``, ``ypeak``, ``pa`` keys; if given,
        overplots MGE isophote contours (requires the external ``mge``
        package).
    ellipsefit : :class:`dict`, optional
        Ellipse-fitting results; if ``ellipsefit['success']``, overplots
        a subsample of fitted isophotes per band, else falls back to a
        single elliptical aperture from ``ellipsefit['geometry']``.
    indx : :class:`numpy.ndarray`, optional
        Boolean mask selecting which isophotes to consider from
        `ellipsefit`; defaults to all isophotes in the first band with
        `ellipsefit` results (and is silently *reused* across
        subsequent bands' loops without being recomputed, since it's
        only initialized once via ``if indx is None``).
    magrange : :class:`float`
        Number of magnitudes below the peak surface brightness spanned
        by the (0.5-mag-per-step) MGE contour levels.
    inchperband : :class:`float`
        Per-band figure size scaling, in inches.
    contours : :class:`bool`
        Unused (see Notes).
    png : :class:`str`, optional
        Output PNG path. If None, the figure is displayed interactively.
    verbose : :class:`bool`
        If True, print the output path when writing `png`.
    vertical : :class:`bool`
        If True, stack band panels vertically instead of horizontally
        (intended for talk slides, per an inline comment).

    Returns
    -------
    None

    Notes
    -----
    `contours` is accepted but never referenced in the function body.
    Uses the legacy import form ``from photutils import
    EllipticalAperture`` (twice, in the `geometry`/`ellipsefit`-fallback
    branches) rather than ``from photutils.aperture import
    EllipticalAperture`` used elsewhere in this codebase (e.g.
    ``SGA.ellipse``, ``SGA.html``) -- this form was removed in
    photutils >= 0.7, so these branches will raise ``ImportError``
    under the environment's current photutils version if exercised.
    The `mgefit` branch imports from an external ``mge`` package
    (``mge.mge_print_contours``) not listed among this project's
    dependencies. See also the general orphaned-code note in
    :func:`qa_multiwavelength_sed`.

    """
    from astropy.visualization import AsinhStretch as Stretch
    from astropy.visualization import ImageNormalize

    band = data['band']
    nband = len(band)

    #cmap = 'RdBu_r'
    #from astropy.visualization import PercentileInterval as Interval
    #interval = Interval(0.9)

    cmap = 'viridis'
    from astropy.visualization import ZScaleInterval as Interval
    interval = Interval(contrast=0.9)

    #cmap = {'g': 'winter_r', 'r': 'summer', 'z': 'autumn_r'}
    #cmap = {'g': 'Blues', 'r': 'Greens', 'z': 'Reds'}

    stretch = Stretch(a=0.95)

    if vertical:
        fig, ax = plt.subplots(3, 1, figsize=(nband, inchperband*nband))
    else:
        fig, ax = plt.subplots(1, 3, figsize=(inchperband*nband, nband))

    for filt, ax1 in zip(band, ax):

        img = data['{}_masked'.format(filt)]
        #img = data[filt]

        norm = ImageNormalize(img, interval=interval, stretch=stretch)

        im = ax1.imshow(img, origin='lower', norm=norm, cmap=cmap, #cmap=cmap[filt],
                        interpolation='nearest')
        plt.text(0.1, 0.9, filt, transform=ax1.transAxes, #fontweight='bold',
                 ha='center', va='center', color='k', fontsize=16)

        if mgefit:
            from mge.mge_print_contours import _multi_gauss, _gauss2d_mge

            sigmapsf = np.atleast_1d(0)
            normpsf = np.atleast_1d(1)
            _magrange = 10**(-0.4*np.arange(0, magrange, 1)[::-1]) # 0.5 mag/arcsec^2 steps
            #_magrange = 10**(-0.4*np.arange(0, magrange, 0.5)[::-1]) # 0.5 mag/arcsec^2 steps

            model = _multi_gauss(mgefit[filt].sol, img, sigmapsf, normpsf,
                                 mgefit['xpeak'], mgefit['ypeak'],
                                 mgefit['pa'])

            peak = data[filt][mgefit['xpeak'], mgefit['ypeak']]
            levels = peak * _magrange
            s = img.shape
            extent = [0, s[1], 0, s[0]]

            ax1.contour(model, levels, colors='k', linestyles='solid',
                        extent=extent, alpha=0.5, lw=1)

        if geometry:
            from photutils import EllipticalAperture

            ellaper = EllipticalAperture((geometry.x0, geometry.y0), geometry.sma,
                                         geometry.sma*(1 - geometry.eps), geometry.pa)
            ellaper.plot(color='k', lw=1, ax=ax1, alpha=0.75)

        if ellipsefit:
            if ellipsefit['success']:
                if len(ellipsefit[filt]) > 0:
                    if indx is None:
                        indx = np.ones(len(ellipsefit[filt]), dtype=bool)

                    nfit = len(indx) # len(ellipsefit[filt])
                    nplot = np.rint(0.5*nfit).astype('int')

                    smas = np.linspace(0, ellipsefit[filt].sma[indx].max(), nplot)
                    for sma in smas:
                        efit = ellipsefit[filt].get_closest(sma)
                        x, y, = efit.sampled_coordinates()
                        ax1.plot(x, y, color='k', lw=1, alpha=0.5)
            else:
                from photutils import EllipticalAperture
                geometry = ellipsefit['geometry']
                ellaper = EllipticalAperture((geometry.x0, geometry.y0), geometry.sma,
                                             geometry.sma*(1 - geometry.eps), geometry.pa)
                ellaper.plot(color='k', lw=1, ax=ax1, alpha=0.5)

        ax1.get_xaxis().set_visible(False)
        ax1.get_yaxis().set_visible(False)
        ax1.axis('off')
        #ax1.set_adjustable('box-forced')
        ax1.autoscale(False)

    if vertical:
        fig.subplots_adjust(hspace=0.02, top=0.98, bottom=0.02, left=0.02, right=0.98)
    else:
        fig.subplots_adjust(wspace=0.02, top=0.98, bottom=0.02, left=0.02, right=0.98)

    if png:
        if verbose:
            print('Writing {}'.format(png))
        fig.savefig(png, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
    else:
        plt.show()
