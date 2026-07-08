#!/usr/bin/env python
"""
SGA.html
========

Generate HTML QA pages for SGA galaxy groups with searchable index.

"""
import pdb

import os
import numpy as np
from astropy.table import Table, vstack, join
from pathlib import Path
from glob import glob
import multiprocessing
from SGA.SGA import SAMPLE, Z_FLAG, RACOLUMN, DECCOLUMN, DIAMCOLUMN, REFIDCOLUMN, APERTURES
from SGA.ellipse import FITMODE, ELLIPSEMODE, ELLIPSEBIT, REF_APERTURES

from SGA.logger import log

import warnings
from astropy.utils.exceptions import AstropyDeprecationWarning

# At the top of ellipse_cog function, add:
warnings.filterwarnings('ignore', category=AstropyDeprecationWarning,
                       message=".*Passing 'theta' positionally.*")

# Base URL for the hosted SGA-2025 data-model documentation; per-group pages
# link to specific anchors on this page to explain flag/reference abbreviations
# without repeating a full glossary on every one of ~470,000 pages.
SGA_DOCS_URL = "https://sga.readthedocs.io/en/latest/sga2025.html"
LEGACYPIPE_BITMASKS_URL = "https://www.legacysurvey.org/dr11/bitmasks/"


def multiband_montage(data, sample, htmlgalaxydir, barlen=None,
                      barlabel=None, clobber=False, fullsample=None):
    """Build a 3x3 grid QA montage (data/model/residual rows x
    optical/unWISE/GALEX columns) for one galaxy or group, plus a
    standalone optical thumbnail JPEG.

    Parameters
    ----------
    data : :class:`dict`
        Multiband cutout data for one galaxy/group, as returned by
        :func:`SGA.SGA.read_multiband` -- must contain ``galaxy``,
        ``opt_bands``, ``unwise_bands``, ``galex_bands``,
        ``opt_jpg_image``, ``{opt,unwise,galex}_jpg_{image,model,resid}``,
        ``{opt,unwise,galex}_pixscale``, and ``{opt,unwise,galex}_refband``.
    sample : :class:`astropy.table.Table`
        Per-object sample table for this galaxy/group (currently
        unused in the function body; see Notes).
    htmlgalaxydir : :class:`str`
        Output directory for the montage PNG and thumbnail JPEG
        (created if it does not exist).
    barlen : :class:`float`
        Scale-bar length, in the same pixel units as the optical
        cutout, drawn on the top-left (data, optical) panel. Required
        in practice despite the ``None`` default (see Notes).
    barlabel : :class:`str`
        Text label drawn next to the scale bar (e.g. ``"1'"``).
    clobber : :class:`bool`
        If True, regenerate the montage/thumbnail even if the output
        files already exist.
    fullsample : :class:`astropy.table.Table`, optional
        Final public-catalog sample table, used only to look up a
        human-readable group title (via ``GROUP_NAME`` and
        ``GROUP_PRIMARY``) for the figure's suptitle; falls back to
        `data`'s own galaxy name if not given or no match is found.

    Returns
    -------
    None

    Notes
    -----
    `sample` is accepted but never referenced in the function body --
    a dead parameter. If `barlen` is left as ``None`` (its default),
    the scale-bar block (``dx = barlen / wimg.shape[0]``) raises
    ``TypeError``, since it always executes for the first (data,
    optical) panel; in practice the sole caller
    (:func:`make_plots`, itself called from ``bin/SGA2025-mpi``)
    always supplies a real `barlen` via
    :func:`SGA.SGA.get_radius_mosaic`.

    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec


    if not os.path.isdir(htmlgalaxydir):
        os.makedirs(htmlgalaxydir, exist_ok=True)

    #qafile = os.path.join('ioannis/tmp2/junk.png')
    qafile = os.path.join(htmlgalaxydir, f'{data["galaxy"]}-montage.png')

    thumb_file = os.path.join(htmlgalaxydir, f'{data["galaxy"]}-thumb.jpg')
    if (not os.path.isfile(thumb_file) or clobber) and data.get('opt_jpg_image') is not None:
        plt.imsave(thumb_file, data['opt_jpg_image'])
        log.info(f'Wrote {thumb_file}')

    if os.path.isfile(qafile) and not clobber:
        log.info(f'File {qafile} exists and clobber=False')
        return


    ncol = 3
    nrow = 3 # data, model, residuals
    inches_per_panel = 3.
    fig, ax = plt.subplots(nrow, ncol,
                           figsize=(inches_per_panel*ncol,
                                    inches_per_panel*nrow),
                           gridspec_kw={'wspace': 0.0, 'hspace': 0.0},
                           constrained_layout=True)

    opt_bands = data['opt_bands']
    labels = [''.join(opt_bands), 'unWISE', 'GALEX']
    imgbands = [opt_bands, data['unwise_bands'], data['galex_bands']]

    for ii, imtype in enumerate(['image', 'model', 'resid']):
        for iax, (xx, bands, label, wimg, pixscale, refband) in enumerate(zip(
                ax[ii, :], imgbands, labels,
                [data[f'opt_jpg_{imtype}'], data[f'unwise_jpg_{imtype}'], data[f'galex_jpg_{imtype}']],
                [data['opt_pixscale'], data['unwise_pixscale'], data['galex_pixscale']],
                [data['opt_refband'], data['unwise_refband'], data['galex_refband']])):

            # can be None for --skip-tractor
            if wimg is not None:
                xx.imshow(wimg)

            # add the scale bar
            if ii == 0 and iax == 0:
                xpos, ypos = 0.07, 0.07
                dx = barlen / wimg.shape[0]
                xx.plot([xpos, xpos+dx], [ypos, ypos], transform=xx.transAxes,
                        color='white', lw=2)
                xx.text(xpos + dx/2., ypos+0.05, barlabel, transform=xx.transAxes,
                        ha='center', va='center', color='white')

            if ii == 0:
                label2 = label
            else:
                label2 = imtype.replace('resid', 'residuals')#.replace('model', 'Tractor model')
            if ii == 0 or iax == 0:
                xx.text(0.03, 0.97, label2, transform=xx.transAxes,
                        ha='left', va='top', color='white',
                        linespacing=1.5, fontsize=11,
                        bbox=dict(boxstyle='round', facecolor='k', alpha=0.5))

    for xx in ax.ravel():
        xx.margins(0)
        xx.set_xticks([])
        xx.set_yticks([])

    _suptitle = data['galaxy'].replace('_', ' ').replace(' GROUP', ' Group')
    if fullsample is not None and all(c in fullsample.colnames for c in ('GALAXY', 'SGAID', 'GROUP_NAME')):
        _grp = data['galaxy'].replace('SGA2025_', '', 1)
        _mask = np.array([str(g).strip() == _grp for g in fullsample['GROUP_NAME']])
        if np.any(_mask):
            _rows = fullsample[_mask]
            _pidx = np.where(_rows['GROUP_PRIMARY'].astype(bool))[0] if 'GROUP_PRIMARY' in _rows.colnames else []
            _prim = _rows[_pidx[0]] if len(_pidx) else _rows[0]
            _gname = str(_prim['GALAXY']).strip() or str(_prim['OBJNAME']).strip()
            if _gname:
                _suptitle = f'{_gname}  [SGAID {int(_prim["SGAID"])}]'
    fig.suptitle(_suptitle)
    fig.savefig(qafile)
    plt.close()
    log.info(f'Wrote {qafile}')


def multiband_ellipse_mask(data, ellipse, htmlgalaxydir, unpack_maskbits_function,
                           SGAMASKBITS, barlen=None, barlabel=None, clobber=False, fullsample=None):
    """Build a QA figure showing the optical maskbits and initial/final
    ellipse geometry for every object in a group: one summary row
    (coadded optical/unWISE/GALEX images with initial geometry
    overlaid), followed by one row per object (masked data, model,
    and mask-type breakdown, each with initial and final ellipses
    overlaid).

    Parameters
    ----------
    data : :class:`dict`
        Multiband cutout data for one galaxy/group, as returned by
        :func:`SGA.SGA.read_multiband` -- must contain ``galaxy``,
        ``width``, ``opt_bands``, ``opt_images``, ``opt_maskbits``,
        ``opt_models``, ``opt_invvar``, ``opt_pixscale``, ``opt_wcs``,
        ``unwise_wcs``, ``galex_wcs``, per-band image/invvar arrays,
        and ``{opt,unwise,galex}_{pixscale,refband}``.
    ellipse : :class:`astropy.table.Table`
        Per-object ellipse-fitting results for this group; must
        contain ``SGAID``, ``OBJNAME``, and (when the object has no
        match in `fullsample`) ``BX_INIT``, ``BY_INIT``, ``SMA_INIT``,
        ``BA_INIT``, ``PA_INIT``, ``BX``, ``BY``, ``SMA_MASK``,
        ``BA_MOMENT``, ``PA_MOMENT``.
    htmlgalaxydir : :class:`str`
        Output directory for the QA PNG (created if it does not
        exist).
    unpack_maskbits_function : callable
        Function (in practice :func:`SGA.SGA.unpack_maskbits`) used to
        expand the packed optical ``opt_maskbits`` bitmask array into
        per-object boolean mask layers.
    SGAMASKBITS : :class:`tuple`
        3-tuple of ``(OPTMASKBITS, UNWISEMASKBITS, GALEXMASKBITS)``
        bitmask dictionaries (see :data:`SGA.SGA.OPTMASKBITS` et al.).
        Only ``OPTMASKBITS`` is actually used (see Notes).
    barlen : :class:`float`
        Unused in this function's body (see Notes); accepted for a
        calling-convention symmetric with :func:`multiband_montage`.
    barlabel : :class:`str`
        Unused in this function's body (see Notes).
    clobber : :class:`bool`
        If True, regenerate the QA figure even if the output file
        already exists.
    fullsample : :class:`astropy.table.Table`, optional
        Final public-catalog sample table. When given, used both to
        look up each object's public ``SGAID``/``RA``/``DEC``/``D26``/
        etc. (preferring these over the raw `ellipse` geometry
        columns) and to build a human-readable group title for the
        figure's suptitle.

    Returns
    -------
    None

    Notes
    -----
    `SGAMASKBITS` unpacks ``UNWISEMASKBITS`` and ``GALEXMASKBITS``,
    but neither is referenced again -- only the optical maskbits are
    decoded and displayed (the mask-type column only ever shows
    ``opt_maskbits``-derived masks, even though the summary row
    displays unWISE and GALEX imagery too). `barlen`/`barlabel` are
    accepted but never used in this function (unlike the otherwise
    parallel :func:`multiband_montage`, which does draw a scale bar)
    -- dead parameters.

    """
    import numpy.ma as ma
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.patches import Patch

    from SGA.sky import map_bxby
    from SGA.qa import overplot_ellipse, get_norm, matched_norm

    #import fitsio
    #tt = Table(fitsio.read('/pscratch/sd/i/ioannis/SGA2025-v0.40/dr11-south/140/14041p0867/SGA2025_14041p0867-tractor.fits'))

    if not os.path.isdir(htmlgalaxydir):
        os.makedirs(htmlgalaxydir, exist_ok=True)

    qafile = os.path.join(htmlgalaxydir, f'{data["galaxy"]}-ellipsemask.png')
    if os.path.isfile(qafile) and not clobber:
        log.info(f'File {qafile} exists and clobber=False')
        return

    nsample = len(ellipse)

    alpha = 0.6
    orange = (0.9, 0.6, 0.0, alpha)   # golden-orange
    blue   = (0.0, 0.45, 0.7, alpha)  # muted blue
    purple = (0.8, 0.6, 0.7, alpha)   # soft violet
    magenta = (0.85, 0.2, 0.5, alpha) # vibrant rose

    opt_bands = data['opt_bands']
    opt_images = data['opt_images']
    opt_maskbits = data['opt_maskbits']
    opt_models = data['opt_models']
    opt_invvar = data['opt_invvar']
    opt_pixscale = data['opt_pixscale']

    opt_wcs = data['opt_wcs']
    unwise_wcs = data['unwise_wcs']
    galex_wcs = data['galex_wcs']

    OPTMASKBITS, UNWISEMASKBITS, GALEXMASKBITS = SGAMASKBITS

    ncol = 3
    nrow = 1 + nsample
    inches_per_panel = 3.
    fig, ax = plt.subplots(nrow, ncol,
                           figsize=(inches_per_panel*ncol,
                                    inches_per_panel*nrow),
                           gridspec_kw={'wspace': 0.01, 'hspace': 0.01},
                           constrained_layout=True)

    cmap = plt.cm.cividis
    cmap.set_bad('white')
    #cmap = plt.get_cmap('cividis').copy()
    #cmap.set_bad((1, 1, 1, 1)) # solid white

    cmap1 = plt.get_cmap('tab20') # or tab20b or tab20c
    colors1 = [cmap1(i) for i in range(50)]

    cmap2 = plt.get_cmap('Dark2')
    colors2 = [cmap2(i) for i in range(5)]

    width = data['width']
    sz = (width, width)

    GEOINITCOLS = ['BX_INIT', 'BY_INIT', 'SMA_INIT', 'BA_INIT', 'PA_INIT']
    GEOFINALCOLS = ['BX', 'BY', 'SMA_MASK', 'BA_MOMENT', 'PA_MOMENT']

    # Precompute display names and geometry for each object from the final catalog when available.
    sgaid_map = {}
    if fullsample is not None:
        sgaid_map = {int(s): i for i, s in enumerate(fullsample['SGAID'])}

    def _radec_to_opt(ra, dec):
        """Convert sky coordinates to 0-indexed optical-image pixel
        coordinates using the enclosing scope's `opt_wcs`.

        Parameters
        ----------
        ra : :class:`float`
            Right ascension, in decimal degrees.
        dec : :class:`float`
            Declination, in decimal degrees.

        Returns
        -------
        :class:`tuple`
            ``(bx, by)`` 0-indexed pixel coordinates in the optical
            image.

        """
        _, bx, by = opt_wcs.wcs.radec2pixelxy(ra, dec)
        return float(bx) - 1., float(by) - 1.

    galaxy_names = []
    init_geom = []   # (bx_opt, by_opt, sma_arcsec, ba, pa)
    final_geom = []  # (bx_opt, by_opt, sma_arcsec, ba, pa)
    for _obj in ellipse:
        _idx = sgaid_map.get(int(_obj['SGAID']), -1) if sgaid_map else -1
        if _idx >= 0:
            _pub = fullsample[_idx]
            galaxy_names.append(str(int(_obj['SGAID'])))
            bxi, byi = _radec_to_opt(float(_pub['RA_INIT']), float(_pub['DEC_INIT']))
            sma_i = float(_pub['DIAM_INIT']) * 60. / 2.  # arcmin diameter → arcsec radius
            init_geom.append((bxi, byi, sma_i, float(_pub['BA_INIT']), float(_pub['PA_INIT'])))
            bxf, byf = _radec_to_opt(float(_pub['RA']), float(_pub['DEC']))
            sma_f = float(_pub['D26']) * 60. / 2.         # arcmin diameter → arcsec radius
            final_geom.append((bxf, byf, sma_f, float(_pub['BA']), float(_pub['PA'])))
        else:
            galaxy_names.append(str(int(_obj['SGAID'])) if int(_obj['SGAID']) != 0 else str(_obj['OBJNAME']).strip())
            init_geom.append(tuple(_obj[col] for col in GEOINITCOLS))
            final_geom.append(tuple(_obj[col] for col in GEOFINALCOLS))
    #GEOFINALCOLS = ['BX', 'BY', 'SMA_MOMENT', 'BA_MOMENT', 'PA_MOMENT']

    # coadded optical, IR, and UV images and initial geometry
    imgbands = [opt_bands, data['unwise_bands'], data['galex_bands']]
    labels = [''.join(opt_bands), 'unWISE', 'GALEX']
    for iax, (xx, bands, label, wcs, pixscale, refband) in enumerate(zip(
            ax[0, :], imgbands, labels,
            [opt_wcs, unwise_wcs, galex_wcs],
            [data['opt_pixscale'], data['unwise_pixscale'], data['galex_pixscale']],
            [data['opt_refband'], data['unwise_refband'], data['galex_refband']])):
        wimgs = np.stack([data[filt] for filt in bands])
        wivars = np.stack([data[f'{filt}_invvar'] for filt in bands])
        wimg = np.sum(wivars * wimgs, axis=0)
        wnorm = np.sum(wivars, axis=0)
        wimg[wnorm > 0.] /= wnorm[wnorm > 0.]

        try:
            norm = get_norm(wimg)
        except:
            norm = None
        xx.imshow(wimg, origin='lower', cmap=cmap, interpolation='none',
                  norm=norm, alpha=1.)
        xx.set_xlim(0, wimg.shape[0]-1)
        xx.set_ylim(0, wimg.shape[1]-1)
        xx.margins(0)

        # initial ellipse geometry
        #pixfactor = data['opt_pixscale'] / pixscale
        for iobj, obj in enumerate(ellipse):
            bx_opt, by_opt, sma, ba, pa = init_geom[iobj]
            bx, by = map_bxby(bx_opt, by_opt, from_wcs=opt_wcs, to_wcs=wcs)
            overplot_ellipse(2*sma, ba, pa, bx, by, pixscale=pixscale, ax=xx,
                             color=colors1[iobj], linestyle='-', linewidth=2,
                             draw_majorminor_axes=True, jpeg=False,
                             label=galaxy_names[iobj])

        xx.text(0.03, 0.97, label, transform=xx.transAxes,
                ha='left', va='top', color='white',
                linespacing=1.5, fontsize=8,
                bbox=dict(boxstyle='round', facecolor='k', alpha=0.5))

        if iax == 0:
            xx.legend(loc='lower left', fontsize=8, ncol=1,
                      fancybox=True, framealpha=0.5)
        del wimgs, wivars, wimg

    # unpack the maskbits bitmask
    opt_masks, brightstarmasks, refmasks, gaiamasks, galmasks = \
        unpack_maskbits_function(opt_maskbits, bands=opt_bands, # [nobj,nband,width,width]
                                 BITS=OPTMASKBITS, allmasks=True)

    # one row per object
    for iobj, obj in enumerate(ellipse):
        opt_masks_obj = opt_masks[iobj, :, :, :]
        brightstarmask = brightstarmasks[iobj, :, :]
        gaiamask = gaiamasks[iobj, :, :]
        galmask = galmasks[iobj, :, :]
        refmask = refmasks[iobj, :, :]

        wimg = np.sum(opt_invvar * np.logical_not(opt_masks_obj) * opt_images[iobj, :, :], axis=0)
        wnorm = np.sum(opt_invvar * np.logical_not(opt_masks_obj), axis=0)
        wimg[wnorm > 0.] /= wnorm[wnorm > 0.]
        #wimg[wmask] = np.nan
        #wmask = wimg == 0.
        wimg = ma.masked_array(wimg, mask=wimg==0., fill_value=np.nan)

        wmodel = np.sum(opt_invvar * opt_models[iobj, :, :, :], axis=0)
        wnorm = np.sum(opt_invvar, axis=0)
        wmodel[wnorm > 0.] /= wnorm[wnorm > 0.] / pixscale**2 # [nanomaggies/arcsec**2]

        S, norm = matched_norm(wimg, wmodel)
        #norm = matched_norm(wimg, wmodel)

        #cmap = plt.cm.get_cmap('cividis').copy()
        #cmap.set_bad('white')
        ax[1+iobj, 0].imshow(S(wimg), cmap=cmap, origin='lower',
                             interpolation='none', norm=norm)
        ax[1+iobj, 1].imshow(wmodel, cmap=cmap, origin='lower',
                             interpolation='none', norm=norm)
        #ax[1+iobj, 1].scatter(tt['bx'], tt['by'], color='red', marker='s', s=5)
        #pdb.set_trace()
        #fig, xx = plt.subplots(1, 2, sharex=True, sharey=True)
        #xx[0].imshow(wimg, origin='lower', norm=norm)
        #wnorm = get_norm(wmodel)
        #wnorm.vmin = norm.vmin
        #wnorm.vmax = norm.vmax
        #xx[1].imshow(wmodel, origin='lower', norm=wnorm)
        #fig.savefig('ioannis/tmp/junk.png')

        # masks
        leg = []
        for msk, col, label in zip([brightstarmask, gaiamask, galmask, refmask],
                                   [orange, blue, purple, magenta],
                                   ['Bright Stars', 'Gaia Stars', 'Galaxies', 'Other SGA']):
            rgba = np.zeros((*msk.shape, 4))
            rgba[msk] = col
            ax[1+iobj, 2].imshow(rgba, origin='lower')
            leg.append(Patch(facecolor=col, edgecolor='none', alpha=0.6, label=label))
        if iobj == 0:
            ax[1+iobj, 2].legend(handles=leg, loc='lower right', fontsize=8)

        for col in range(3):
            # initial geometry
            bx, by, sma, ba, pa = init_geom[iobj]
            overplot_ellipse(2*sma, ba, pa, bx, by, pixscale=opt_pixscale,
                             ax=ax[1+iobj, col], color=colors2[0], linestyle='-',
                             linewidth=2, draw_majorminor_axes=True,
                             jpeg=False, label='Initial')

            # final geometry
            bx, by, sma, ba, pa = final_geom[iobj]
            overplot_ellipse(2*sma, ba, pa, bx, by, pixscale=opt_pixscale,
                             ax=ax[1+iobj, col], color=colors2[1], linestyle='--',
                             linewidth=2, draw_majorminor_axes=True,
                             jpeg=False, label='Final')
            ax[1+iobj, col].set_xlim(0, width-1)
            ax[1+iobj, col].set_ylim(0, width-1)
            ax[1+iobj, col].margins(0)

        ax[1+iobj, 0].text(0.03, 0.97, f'{int(obj["SGAID"])}',
                           transform=ax[1+iobj, 0].transAxes,
                           ha='left', va='top', color='white',
                           linespacing=1.5, fontsize=8,
                           bbox=dict(boxstyle='round', facecolor='k', alpha=0.5))

        if iobj == 0:
            ax[1+iobj, 1].text(0.03, 0.97, f'{"".join(opt_bands)} models',
                               transform=ax[1+iobj, 1].transAxes, ha='left', va='top',
                               color='white', linespacing=1.5, fontsize=8,
                               bbox=dict(boxstyle='round', facecolor='k', alpha=0.5))
            ax[1+iobj, 2].text(0.03, 0.97, f'{"".join(opt_bands)} masks',
                               transform=ax[1+iobj, 2].transAxes, ha='left', va='top',
                               color='white', linespacing=1.5, fontsize=8,
                               bbox=dict(boxstyle='round', facecolor='k', alpha=0.5))

        ax[1+iobj, 0].legend(loc='lower left', fontsize=8, fancybox=True,
                             framealpha=0.5)

    for xx in ax.ravel():
        xx.margins(0)
        xx.set_xticks([])
        xx.set_yticks([])

    _suptitle = data['galaxy'].replace('_', ' ').replace(' GROUP', ' Group')
    if fullsample is not None and all(c in fullsample.colnames for c in ('GALAXY', 'SGAID', 'GROUP_NAME')):
        _grp = data['galaxy'].replace('SGA2025_', '', 1)
        _mask = np.array([str(g).strip() == _grp for g in fullsample['GROUP_NAME']])
        if np.any(_mask):
            _rows = fullsample[_mask]
            _pidx = np.where(_rows['GROUP_PRIMARY'].astype(bool))[0] if 'GROUP_PRIMARY' in _rows.colnames else []
            _prim = _rows[_pidx[0]] if len(_pidx) else _rows[0]
            _gname = str(_prim['GALAXY']).strip() or str(_prim['OBJNAME']).strip()
            if _gname:
                _suptitle = f'{_gname}  [SGAID {int(_prim["SGAID"])}]'
    fig.suptitle(_suptitle)
    fig.savefig(qafile)
    plt.close()
    log.info(f'Wrote {qafile}')


def ellipse_sed(data, ellipse, htmlgalaxydir, tractor=None, run='south',
                apertures=REF_APERTURES, clobber=False, fullsample=None):
    """Build and write a per-galaxy spectral energy distribution (SED)
    QA figure, plotting total (curve-of-growth) magnitude, optional
    Tractor photometry, and aperture-photometry magnitudes as a
    function of observed-frame wavelength, for every object in
    `ellipse`.

    Parameters
    ----------
    data : :class:`dict`
        Per-group multiband imaging metadata; must contain
        ``'all_data_bands'`` (the list of bandpasses to plot, spanning
        optical/unWISE/GALEX as available).
    ellipse : :class:`astropy.table.Table`
        Per-object ellipse-photometry catalog with ``SGANAME``,
        ``OBJNAME``, ``SGAID``, and per-band ``COG_MTOT_{BAND}``,
        ``COG_MTOT_ERR_{BAND}``, ``FLUX_AP{NN}_{BAND}``,
        ``FLUX_ERR_AP{NN}_{BAND}`` columns.
    htmlgalaxydir : :class:`str`
        Output directory for the per-object ``{SGANAME}-sed.png``
        figures.
    tractor : :class:`astropy.table.Table` or array-like, optional
        Per-object Tractor catalog(s) (one per row of `ellipse`,
        indexable as ``tractor[iobj][0]``) providing
        ``flux_{band}``/``flux_ivar_{band}`` attributes. If None,
        Tractor photometry is omitted from the plot.
    run : :class:`str`
        Photometric system passed to :func:`SGA.util.filter_effwaves`
        to look up per-band effective wavelengths (``'south'`` or
        ``'north'``).
    apertures : :class:`list`
        Aperture radii/labels (defaults to
        :data:`SGA.ellipse.REF_APERTURES`) used only to set the number
        of aperture-photometry series plotted (``len(apertures)``);
        the aperture *values* themselves are not otherwise used here.
    clobber : :class:`bool`
        If False, skip objects whose output PNG already exists.
    fullsample : :class:`astropy.table.Table`, optional
        Final public catalog, keyed by ``SGAID``, used to look up the
        published galaxy name/SGAID for the plot title when available.
        If None (or an object's ``SGAID`` is not found), falls back to
        ``ellipse``'s own ``OBJNAME``/``SGAID``.

    Returns
    -------
    None
        One PNG figure per object is written to `htmlgalaxydir` as a
        side effect.

    Notes
    -----
    The nested ``_addphot`` helper plots error bars for one photometry
    series (total, Tractor, or one aperture), splitting points into
    lower-limit markers (``lolims=True``, no legend label) and
    well-constrained markers (normal error bars, with legend label);
    it closes over ``ax`` and ``bandwave`` from the enclosing scope
    rather than taking them as arguments.

    """
    from copy import deepcopy
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import seaborn as sns

    from SGA.util import filter_effwaves


    colors1 = sns.color_palette('Set1', n_colors=14, desat=0.75)

    # Build SGAID → index map for fast final-catalog lookups.
    sgaid_map = {}
    if fullsample is not None:
        sgaid_map = {int(s): i for i, s in enumerate(fullsample['SGAID'])}

    marker_mtot = 's'
    marker_tractor = 'o'
    color_mtot = colors1[3]
    color_tractor = colors1[1]

    markers_ap = ['v', '^', 'p', 'P', '*']
    colors_ap = sns.color_palette('Set2', n_colors=5)#, desat=0.75)

    weff = filter_effwaves(run=run)
    bands = data['all_data_bands']
    nband = len(bands)

    # build the arrays
    bandwave = np.array([weff[filt] for filt in bands])
    tphot = {'abmag': np.zeros(nband, 'f4') - 1,
             'abmagerr': np.zeros(nband, 'f4') + 0.5,
             'lower': np.zeros(nband, bool)}
    phot = {'mag_tot': deepcopy(tphot), 'tractor': deepcopy(tphot)}

    for iap in range(len(apertures)):
        phot.update({f'mag_ap{iap:02}': deepcopy(tphot)})


    def _addphot(thisphot, color, marker, alpha, label):
        """Plot one photometry series (total, Tractor, or one
        aperture) as error bars, splitting points into lower-limit
        markers (no legend label) and well-constrained markers (with
        legend label).

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
        :func:`ellipse_sed` scope rather than taking them as
        parameters.

        """
        good = np.where((thisphot['abmag'] > 0.) * thisphot['lower'])[0]
        if len(good) > 0:
            ax.errorbar(bandwave[good]/1e4, thisphot['abmag'][good], yerr=thisphot['abmagerr'][good],
                        marker=marker, markersize=11, markeredgewidth=3, markeredgecolor='k',
                        markerfacecolor=color, elinewidth=3, ecolor=color, capsize=4,
                        lolims=True, linestyle='none', alpha=alpha)#, lolims=True)
        good = np.where((thisphot['abmag'] > 0) * (thisphot['lower'] == False) * (thisphot['abmagerr'] <= 1.))[0]
        if len(good) > 0:
            ax.errorbar(bandwave[good]/1e4, thisphot['abmag'][good], yerr=thisphot['abmagerr'][good],
                        marker=marker, markersize=11, markeredgewidth=3, markeredgecolor='k',
                        markerfacecolor=color, elinewidth=3, ecolor=color, capsize=4,
                        label=label, linestyle='none', alpha=alpha)


    # see also Morrisey+05
    for iobj, obj in enumerate(ellipse):

        sganame = obj['SGANAME'].replace(' ', '_')
        qafile = os.path.join(htmlgalaxydir, f'{sganame}-sed.png')
        if os.path.isfile(qafile) and not clobber:
            log.info(f'File {qafile} exists and clobber=False')
            continue

        pub = None
        if sgaid_map:
            idx = sgaid_map.get(int(obj['SGAID']), -1)
            if idx >= 0:
                pub = fullsample[idx]
        if pub is not None:
            galaxy_name = str(pub['GALAXY']).strip() or str(obj['OBJNAME']).strip()
            title = f'SGAID {int(pub["SGAID"])}'
            #title = f'{galaxy_name}  [SGAID {int(pub["SGAID"])}]'
        else:
            #title = f"{obj['OBJNAME']} ({obj['SGANAME']})"
            title = f"{obj['OBJNAME']} [{obj['SGAID']}]"

        for ifilt, filt in enumerate(bands):
            mtot = ellipse[f'COG_MTOT_{filt.upper()}'][iobj]
            mtoterr = ellipse[f'COG_MTOT_ERR_{filt.upper()}'][iobj]
            if mtot > 0:
                phot['mag_tot']['abmag'][ifilt] = mtot
                phot['mag_tot']['abmagerr'][ifilt] = mtoterr
                phot['mag_tot']['lower'][ifilt] = False

            for iap in range(len(apertures)):
                flux = ellipse[f'FLUX_AP{iap:02}_{filt.upper()}'][iobj]
                ferr = ellipse[f'FLUX_ERR_AP{iap:02}_{filt.upper()}'][iobj]

                if flux > 0. and ferr > 0.:
                    mag = 22.5 - 2.5 * np.log10(flux)
                    magerr = 2.5 * ferr / flux / np.log(10.)
                    phot[f'mag_ap{iap:02}']['abmag'][ifilt] = mag
                    phot[f'mag_ap{iap:02}']['abmagerr'][ifilt] = magerr
                    phot[f'mag_ap{iap:02}']['lower'][ifilt] = False
                if flux <=0 and ferr > 0.:
                    mag = 22.5 - 2.5 * np.log10(ferr)
                    phot[f'mag_ap{iap:02}']['abmag'][ifilt] = mag
                    phot[f'mag_ap{iap:02}']['abmagerr'][ifilt] = 0.75
                    phot[f'mag_ap{iap:02}']['lower'][ifilt] = True

                if tractor is not None:
                    if tractor[iobj] is not None:
                        flux = getattr(tractor[iobj][0], f'flux_{filt.lower()}')
                        ivar = getattr(tractor[iobj][0], f'flux_ivar_{filt.lower()}')
                        if flux > 0. and ivar > 0.:
                            ferr = 1. / np.sqrt(ivar)
                            magerr = 2.5 * ferr / flux / np.log(10.)
                            phot['tractor']['abmag'][ifilt] = 22.5 - 2.5 * np.log10(flux)
                            phot['tractor']['abmagerr'][ifilt] = magerr
                        if flux <= 0. and ivar > 0.:
                            ferr = 1. / np.sqrt(ivar)
                            mag = 22.5 - 2.5 * np.log10(ferr)
                            phot['tractor']['abmag'][ifilt] = mag
                            phot['tractor']['abmagerr'][ifilt] = 0.75
                            phot['tractor']['lower'][ifilt] = True


        # make the plot
        fig, ax = plt.subplots(figsize=(8, 6))

        # get the plot limits
        ymin, ymax = None, None
        good = np.where(phot['mag_tot']['abmag'] > 0)[0]
        if len(good) > 0:
            ymax = np.min(phot['mag_tot']['abmag'][good])
            ymin = np.max(phot['mag_tot']['abmag'][good])

        if tractor is not None:
            good = np.where(phot['tractor']['abmag'] > 0)[0]
            if len(good) > 0:
                if ymax is None:
                    ymax = np.min(phot['tractor']['abmag'][good])
                else:
                    if np.min(phot['tractor']['abmag'][good]) < ymax:
                        ymax = np.min(phot['tractor']['abmag'][good])
                if ymin is None:
                    ymin = np.max(phot['tractor']['abmag'][good])
                else:
                    if np.max(phot['tractor']['abmag']) > ymin:
                        ymin = np.max(phot['tractor']['abmag'][good])

        for iap in range(len(apertures)):
            good = np.where(phot[f'mag_ap{iap:02}']['abmag'] > 0)[0]
            if len(good) > 0:
                if ymax is None:
                    ymax = np.min(phot[f'mag_ap{iap:02}']['abmag'][good])
                else:
                    if np.min(phot[f'mag_ap{iap:02}']['abmag'][good]) < ymax:
                        ymax = np.min(phot[f'mag_ap{iap:02}']['abmag'][good])
                if ymin is None:
                    ymin = np.max(phot[f'mag_ap{iap:02}']['abmag'][good])
                else:
                    if np.max(phot[f'mag_ap{iap:02}']['abmag']) > ymin:
                        ymin = np.max(phot[f'mag_ap{iap:02}']['abmag'][good])

        if ymin is None:
            ymin = 25
        if ymax is None:
            ymax = 10

        ymin += 2.
        ymax -= 1.5

        wavemin, wavemax = 0.1, 30

        # have to set the limits before plotting since the axes are reversed
        if np.abs(ymax-ymin) > 15:
            ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
        ax.set_ylim(ymin, ymax)

        ax.axvspan(wavemin, 0.4,    alpha=0.06, color='#aaaaff', zorder=0)
        ax.axvspan(0.4,    1.0,     alpha=0.06, color='#aaffaa', zorder=0)
        ax.axvspan(1.0,    wavemax, alpha=0.06, color='#ffccaa', zorder=0)

        _addphot(phot['mag_tot'], color=color_mtot, marker=marker_mtot, alpha=1.0,
                 label=r'$m_{\mathrm{tot}}$')
        if tractor is not None:
            _addphot(phot['tractor'], color=color_tractor, marker=marker_tractor, alpha=0.75,
                     label='Tractor')
        for iap in range(len(apertures)):
            _addphot(phot[f'mag_ap{iap:02}'], color=colors_ap[iap], marker=markers_ap[iap],
                     alpha=0.9, label=r'$m(<R_{'+f'AP{iap:02}'+r'})$')

        ax.set_xlabel(r'Observed-frame Wavelength ($\mu$m)', fontsize=14)
        ax.set_ylabel(r'Apparent Brightness (AB mag)', fontsize=14)
        ax.tick_params(axis='both', labelsize=14)
        ax.set_xlim(wavemin, wavemax)
        ax.set_xscale('log')
        ax.legend(loc='lower right', frameon=True, ncol=2)


        def _frmt(value, _):
            """Format an x-axis tick `value` (in microns) as a
            :class:`matplotlib.ticker.FuncFormatter` callback: one
            decimal place below 1, none at or above.

            Parameters
            ----------
            value : :class:`float`
                Tick value to format.
            _ : :class:`int`
                Tick position (unused, required by the
                ``FuncFormatter`` callback signature).

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

        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(qafile, bbox_inches='tight')
        plt.close()
        log.info(f'Wrote {qafile}')



def ellipse_cog(data, ellipse, sbprofiles, region, htmlgalaxydir,
                datasets=['opt', 'unwise', 'galex'], clobber=False,
                fullsample=None):
    """Build and write a per-galaxy curve-of-growth (COG) QA figure,
    plotting cumulative aperture magnitude vs. semi-major axis for
    each band/dataset along with the best-fitting COG model, and
    marking the SGA moment-based and threshold-isophote size scales.

    Parameters
    ----------
    data : :class:`dict`
        Per-group multiband imaging metadata; must contain
        ``'{dataset}_bands'`` for each entry of `datasets`.
    ellipse : :class:`astropy.table.Table`
        Per-object ellipse-photometry catalog with ``SGANAME``,
        ``OBJNAME``, ``SGAID``, ``SMA_MOMENT``, and per-band
        ``COG_MTOT_{BAND}``, ``COG_MTOT_ERR_{BAND}``,
        ``COG_DMAG_{BAND}``, ``COG_LNALPHA1_{BAND}``,
        ``COG_LNALPHA2_{BAND}`` columns.
    sbprofiles : :class:`list` of :class:`astropy.table.Table`
        One surface-brightness-profile table per entry of `datasets`
        (same ordering), each row-matched to `ellipse` and containing
        ``SMA`` and per-band ``FLUX_{BAND}``/``FLUX_ERR_{BAND}``
        columns.
    region : :class:`str`
        Survey region, passed to :func:`SGA.SGA.SGA_diameter` when a
        published D26 is unavailable (see Notes).
    htmlgalaxydir : :class:`str`
        Output directory for the per-object ``{SGANAME}-cog.png``
        figures.
    datasets : :class:`list`
        Ordered list of dataset keys (e.g. ``'opt'``, ``'unwise'``,
        ``'galex'``) identifying which entries of `data` and
        `sbprofiles` to plot; also selects the marker style
        (``markers[idata]``) for each dataset.
    clobber : :class:`bool`
        If False, skip objects whose output PNG already exists.
    fullsample : :class:`astropy.table.Table`, optional
        Final public catalog, keyed by ``SGAID``, used to look up the
        published galaxy name/SGAID and D26 diameter for the plot
        title and size-scale marker. If None (or an object's ``SGAID``
        is not found, or its published ``D26`` is not positive), falls
        back to computing the threshold radius via
        :func:`SGA.SGA.SGA_diameter` on the `ellipse` row itself.

    Returns
    -------
    None
        One PNG figure per object is written to `htmlgalaxydir` as a
        side effect.

    Notes
    -----
    The bare ``import matplotlib`` at the top of this function is
    unused in the live code path -- the only reference to the
    ``matplotlib.lines.Line2D`` type check via the bare module name is
    in a commented-out block; the actually-used symbols
    (``matplotlib.pyplot``, ``blended_transform_factory``,
    ``matplotlib.lines``) are imported separately.

    """
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.transforms import blended_transform_factory
    import matplotlib.lines as mlines

    from SGA.SGA import SGA_diameter
    from SGA.ellipse import cog_model
    from SGA.qa import sbprofile_colors

    colors2 = ['#e91e8c', '#039be5']

    # Build SGAID → index map for fast final-catalog lookups.
    sgaid_map = {}
    if fullsample is not None:
        sgaid_map = {int(s): i for i, s in enumerate(fullsample['SGAID'])}

    for iobj, obj in enumerate(ellipse):

        sganame = obj['SGANAME'].replace(' ', '_')
        qafile = os.path.join(htmlgalaxydir, f'{sganame}-cog.png')
        if os.path.isfile(qafile) and not clobber:
            log.info(f'File {qafile} exists and clobber=False')
            continue

        sbcolors = sbprofile_colors()

        markers = ['s', 'o', 'v']

        # Look up final catalog row for this galaxy.
        pub = None
        if sgaid_map:
            idx = sgaid_map.get(int(obj['SGAID']), -1)
            if idx >= 0:
                pub = fullsample[idx]

        # Build title and sma_sbthresh from final catalog values when available.
        sma_moment = obj['SMA_MOMENT']  # [arcsec]
        label_moment = f'$R(\\mathrm{{mom}})={sma_moment:.1f}$"'
        if pub is not None:
            galaxy_name = str(pub['GALAXY']).strip() or str(pub['OBJNAME']).strip()
            title = f'SGAID {int(pub["SGAID"])}'
            #title = f'{galaxy_name}  [SGAID {int(pub["SGAID"])}]'
            d26 = float(pub['D26'])
            d26_ref = str(pub['D26_REF']).strip()
        else:
            title = f"{obj['OBJNAME']} [{obj['SGAID']}]"
            #title = f"{obj['OBJNAME']} ({obj['SGANAME']})"
            d26 = 0.
            d26_ref = ''

        if pub is not None and d26 > 0.:
            sma_sbthresh = d26 / 2. * 60.  # arcmin → arcsec
            ref_str = d26_ref if d26_ref else 'D26'
        else:
            sma_sbthresh, _, _ref_arr, _ = SGA_diameter(
                Table(obj), region, radius_arcsec=True)
            sma_sbthresh = sma_sbthresh[0]
            ref_str = _ref_arr[0]
        label_sbthresh = f'$R({ref_str})={sma_sbthresh:.1f}$"'

        sma_max = float(np.max(sbprofiles[0][iobj]['SMA'].value))
        xminmax = [0., sma_max]
        #xminmax = [0., sma_max**0.25]

        fig, ax = plt.subplots(figsize=(8, 6))

        # one row per dataset
        yminmax = [40, 0]
        for idata, dataset in enumerate(datasets):

            sbprofiles_obj = sbprofiles[idata][iobj]
            bands = data[f'{dataset}_bands']

            for filt in bands:
                I = ((sbprofiles_obj[f'FLUX_{filt.upper()}'].value > 0.) *
                     (sbprofiles_obj[f'FLUX_ERR_{filt.upper()}'].value > 0.))

                if np.any(I):
                    sma_lin = sbprofiles_obj['SMA'][I].value
                    flux = sbprofiles_obj[f'FLUX_{filt.upper()}'][I].value
                    fluxerr = sbprofiles_obj[f'FLUX_ERR_{filt.upper()}'][I].value
                    mag = 22.5 - 2.5 * np.log10(flux)
                    magerr = 2.5 * fluxerr / flux / np.log(10.)

                    mtot = ellipse[f'COG_MTOT_{filt.upper()}'][iobj]
                    mtoterr = ellipse[f'COG_MTOT_ERR_{filt.upper()}'][iobj]
                    good_cog = (mtot > 0) and (mtoterr <= 2.)
                    if good_cog:
                        label = f'{filt}={mtot:.3f}'+r'$\pm$'+f'{mtoterr:.3f} mag'
                    else:
                        label = '_nolegend_'

                    # only plot points with well-constrained error bars
                    K = magerr < 1.
                    col = sbcolors[filt]
                    if np.any(K):
                        #ax.errorbar(sma_lin[K]**0.25, mag[K], yerr=magerr[K],
                        ax.errorbar(sma_lin[K], mag[K], yerr=magerr[K],
                                    fmt=markers[idata], markersize=5, markeredgewidth=1,
                                    markeredgecolor='k', markerfacecolor=col, elinewidth=3,
                                    ecolor=col, capsize=4, label=label, alpha=0.7)

                    # best-fitting model: start from minimum observed sma to avoid
                    # extrapolation artefacts in r^1/4 space
                    dmag = ellipse[f'COG_DMAG_{filt.upper()}'][iobj]
                    lnalpha1 = ellipse[f'COG_LNALPHA1_{filt.upper()}'][iobj]
                    lnalpha2 = ellipse[f'COG_LNALPHA2_{filt.upper()}'][iobj]
                    if good_cog:
                        sma_min_filt = float(np.min(sma_lin))
                        smagrid_lin = np.logspace(np.log10(sma_min_filt), np.log10(sma_max), 100)
                        #smagrid_lin = np.linspace(sma_min_filt, sma_max, 500)
                        mfit = cog_model(smagrid_lin, mtot, dmag, lnalpha1, lnalpha2, r0=sma_moment)
                        ax.plot(smagrid_lin, mfit, color=col, alpha=0.8)
                        #ax.plot(smagrid_lin**0.25, mfit, color=col, alpha=0.8)

                    # robust limits (use all points, not just K, for y-range)
                    maglo = (mag - magerr)[(magerr < 1.) * (mag / magerr > 8.)]
                    maghi = (mag + magerr)[(magerr < 1.) * (mag / magerr > 8.)]
                    #print(filt, np.min(maglo), np.max(maghi))
                    if len(maglo) > 0:
                        mn = np.min(maglo)
                        if mn < yminmax[0]:
                            yminmax[0] = mn
                    if len(maghi) > 0:
                        mx = np.max(maghi)
                        if mx > yminmax[1]:
                            yminmax[1] = mx
                #print(filt, yminmax[0], yminmax[1])

        ylim = [yminmax[0]-1.5, yminmax[1]+1.5]
        #if ylim[0] < 13:
        #    ylim[0] = 13
        if ylim[1] > 34:
            ylim[1] = 34
        #print(idata, yminmax, ylim)
        ax.set_ylim(ylim)
        ax.invert_yaxis()
        ax.set_xlim(xminmax)
        ax.margins(x=0)

        ax.set_xlabel('Semi-major axis (arcsec)')
        #ax.set_xlabel(r'(Semi-major axis / arcsec)$^{1/4}$')
        ax.set_ylabel('Cumulative Brightness (AB mag)')

        #if sma_sbthresh > 0.:
        #    ax.axvline(x=sma_sbthresh**0.25, color=colors2[1], lw=2, ls='-', label=label_sbthresh)
        #ax.axvline(x=sma_moment**0.25, color=colors2[0], lw=2, ls='--', label=label_moment)

        #hndls, _ = ax.get_legend_handles_labels()
        #if hndls:
        #    # split into two legends
        #    hndls_data = [hndl for hndl in hndls if not type(hndl) is matplotlib.lines.Line2D]
        #    hndls_vline = [hndl for hndl in hndls if type(hndl) is matplotlib.lines.Line2D]
        #    leg1 = ax.legend(handles=hndls_data, loc='lower right', fontsize=8)
        #    ax.legend(handles=hndls_vline, loc='upper left', fontsize=11)
        #    ax.add_artist(leg1)

        # Arrows just inside the axis marking size scales.
        # mfc='none' matches the unfilled outline style of the
        # aperture ellipses.
        trans = blended_transform_factory(ax.transData, ax.transAxes)
        if sma_sbthresh > 0.:
            #ax.plot(sma_sbthresh**0.25, 0.94, 'v', mec=colors2[1], mfc='none', ms=9, mew=1.5,
            ax.plot(sma_sbthresh, 0.94, 'v', mec=colors2[1], mfc='none', ms=9, mew=1.5,
                    transform=trans, clip_on=False, zorder=5)
        #ax.plot(sma_moment**0.25, 0.94, 'v', mec=colors2[0], mfc='none', ms=9, mew=1.5,
        ax.plot(sma_moment, 0.94, 'v', mec=colors2[0], mfc='none', ms=9, mew=1.5,
                transform=trans, clip_on=False, zorder=5)

        band_hndls, _ = ax.get_legend_handles_labels()
        if band_hndls:
            size_hndls = []
            if sma_sbthresh > 0.:
                size_hndls.append(mlines.Line2D(
                    [], [], mec=colors2[1], mfc='none', mew=1.5,
                    marker='v', linestyle='None', ms=9, label=label_sbthresh))
            size_hndls.append(mlines.Line2D(
                [], [], mec=colors2[0], mfc='none', mew=1.5,
                marker='v', linestyle='None', ms=9, label=label_moment))
            ax.legend(handles=size_hndls, loc='lower right', fontsize=11)

        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(qafile, bbox_inches='tight')
        plt.close()
        log.info(f'Wrote {qafile}')



def ellipse_sbprofiles(data, ellipse, sbprofiles, region, htmlgalaxydir,
                       unpack_maskbits_function, MASKBITS, REFIDCOLUMN,
                       datasets=['opt', 'unwise', 'galex'],
                       linear=False, clobber=False, fullsample=None):
    """Build the per-galaxy surface-brightness-profile QA figure: one
    row per dataset (opt/unWISE/GALEX), with a cutout image (with
    isophote and reference-radius apertures overplotted) in the left
    column and the surface-brightness profile(s) vs. ``sma**0.25`` in
    the right column.

    Parameters
    ----------
    data : :class:`dict`
        Multiband cutout data, as returned by a ``read_multiband``-type
        function; must contain, for each entry in `datasets`, the keys
        ``{dataset}_images``, ``{dataset}_models``,
        ``{dataset}_maskbits``, ``{dataset}_bands``,
        ``{dataset}_pixscale``, ``{dataset}_wcs``, plus ``opt_wcs``,
        ``opt_pixscale``, and ``opt_bands``.
    ellipse : :class:`astropy.table.Table`
        Per-galaxy ellipse-fitting results (one row per object), with
        columns including ``SGANAME``, ``SGAID``, ``OBJNAME``, ``BX``,
        ``BY``, ``PA_MOMENT``, ``BA_MOMENT``, ``SMA_MOMENT``.
    sbprofiles : :class:`list`
        Nested list of surface-brightness-profile tables, indexed as
        ``sbprofiles[idata][iobj]``, with columns ``SMA`` and, per
        band, ``SB_{FILT}``/``SB_ERR_{FILT}``.
    region : :class:`str`
        Imaging region (e.g. ``'dr11-south'``), passed to
        :func:`SGA.SGA.SGA_diameter` when the final-catalog ``D26``
        value is unavailable.
    htmlgalaxydir : :class:`str`
        Output directory for the QA PNG.
    unpack_maskbits_function : callable
        Function used to unpack a packed maskbits array into a
        per-band boolean mask cube; called as
        ``unpack_maskbits_function(data[f'{dataset}_maskbits'],
        bands=bands, BITS=MASKBITS[idata])``.
    MASKBITS : :class:`list`
        Per-dataset mask-bit dictionaries, indexed as
        ``MASKBITS[idata]`` and passed through to
        `unpack_maskbits_function`.
    REFIDCOLUMN : :class:`str`
        Unused (see Notes).
    datasets : :class:`list`
        Dataset keys to plot, in row order. Must line up positionally
        with the hardcoded label list ``[opt_bands, 'unWISE',
        'GALEX']`` used for the left-column annotation (see Notes).
    linear : :class:`bool`
        If True, use a linear (rather than magnitude/log) y-axis scale
        for the surface-brightness panels.
    clobber : :class:`bool`
        If True, regenerate the QA PNG even if it already exists.
    fullsample : :class:`astropy.table.Table`, optional
        Final-catalog table (indexed by ``SGAID``) used to look up the
        adopted ``D26``/``BA``/``PA`` for the title and geometry
        annotation and the D(26) reference aperture; when a galaxy is
        not found (or `fullsample` is None), falls back to
        :func:`SGA.SGA.SGA_diameter` for the reference radius and uses
        ``OBJNAME``/``SGAID`` for the title.

    Returns
    -------
    None
        Writes ``{htmlgalaxydir}/{sganame}-sbprofiles.png`` for each
        object in `ellipse`, as a side effect.

    Notes
    -----
    `REFIDCOLUMN` is accepted but never referenced in the function
    body -- dead parameter. The left-column dataset labels
    (``opt_bands``, ``'unWISE'``, ``'GALEX'``) are hardcoded via
    ``zip(datasets, [opt_bands, 'unWISE', 'GALEX'])`` rather than
    derived from `datasets`, so passing a `datasets` list that omits,
    reorders, or extends beyond ``['opt', 'unwise', 'galex']`` produces
    mismatched labels (or is silently truncated by `zip`). ``nsample =
    len(ellipse)`` is computed but never used.

    """
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import matplotlib.lines as mlines
    from matplotlib.transforms import blended_transform_factory
    from photutils.isophote import EllipseGeometry
    from photutils.aperture import EllipticalAperture

    from SGA.sky import map_bxby
    from SGA.SGA import SGA_diameter
    from SGA.qa import overplot_ellipse, get_norm, sbprofile_colors

    def kill_left_y(ax):
        """Hide an axes' left y-axis (ticks, labels, and spine), used
        to keep only the twin right-hand y-axis visible on the
        surface-brightness panels.

        Parameters
        ----------
        ax : :class:`matplotlib.axes.Axes`
            Axes whose left y-axis should be hidden.

        Returns
        -------
        None

        """
        ax.yaxis.set_major_locator(ticker.NullLocator())
        ax.yaxis.set_minor_locator(ticker.NullLocator())
        ax.tick_params(axis='y', which='both', left=False, labelleft=False)
        ax.spines['left'].set_visible(False)  # optional


    nsample = len(ellipse)
    ndataset = len(datasets)

    opt_wcs = data['opt_wcs']
    opt_pixscale = data['opt_pixscale']
    opt_bands = ''.join(data['opt_bands']) # not general

    ncol = 2 # 3
    nrow = ndataset
    inches_per_panel = 3.

    cmap = plt.cm.cividis
    cmap.set_bad('white')

    sbcolors = sbprofile_colors()
    cmap2a = plt.get_cmap('Dark2')
    cmap2b = plt.get_cmap('Paired')
    #colors2 = [cmap2a(1), cmap2b(3)]
    #colors2 = ['#039be5', '#d81b60']
    colors2 = ['#e91e8c', '#039be5']
    #colors2 = ['#7e57c2', '#e91e8c']
    #colors2 = ['#00acc1', '#d81b60']

    # Build SGAID → index map for fast final-catalog lookups.
    sgaid_map = {}
    if fullsample is not None:
        sgaid_map = {int(s): i for i, s in enumerate(fullsample['SGAID'])}

    for iobj, obj in enumerate(ellipse):

        sganame = obj['SGANAME'].replace(' ', '_')
        qafile = os.path.join(htmlgalaxydir, f'{sganame}-sbprofiles.png')
        if os.path.isfile(qafile) and not clobber:
            log.info(f'File {qafile} exists and clobber=False')
            continue

        # Look up final catalog row for this galaxy.
        pub = None
        if sgaid_map:
            idx = sgaid_map.get(int(obj['SGAID']), -1)
            if idx >= 0:
                pub = fullsample[idx]

        # Build figure title and geometry annotation from final catalog values when available.
        geom_str = ''
        if pub is not None:
            galaxy_name = str(pub['GALAXY']).strip() or str(pub['OBJNAME']).strip()
            d26 = float(pub['D26'])
            d26_err = float(pub['D26_ERR'])
            d26_ref = str(pub['D26_REF']).strip()
            ba = float(pub['BA'])
            pa = float(pub['PA'])
            title = f'SGAID {int(pub["SGAID"])}'
            #title = f'{galaxy_name}  [SGAID {int(pub["SGAID"])}]'
            d26_ref_str = f'[{d26_ref}]' if d26_ref else ''
            geom_str = f'$D(26)$ = {d26:.3f}±{d26_err:.3f} arcmin PA={pa:.0f}°  b/a={ba:.2f}'
        else:
            title = f"{obj['OBJNAME']} [{obj['SGAID']}]"
            #title = f"{obj['OBJNAME']} ({obj['SGANAME']})"


        fig, ax = plt.subplots(nrow, ncol,
                               figsize=(inches_per_panel * (1+ncol),
                                        inches_per_panel * nrow),
                               gridspec_kw={
                                   'height_ratios': [1., 1., 1.],
                                   'width_ratios': [1., 2.],
                                   #'width_ratios': [1., 2., 2.],
                                   #'wspace': 0
                               })
        for idata in range(1, ndataset):
            ax[idata, 1].sharex(ax[0, 1])

        # one row per dataset
        sma_sbthresh = 0.
        label_sbthresh = ''
        for idata, (dataset, label) in enumerate(zip(datasets, [opt_bands, 'unWISE', 'GALEX'])):

            images = data[f'{dataset}_images'][iobj, :, :, :]
            if np.all(images == 0.):
                have_data = False
            else:
                have_data = True

            #results_obj = results[idata][iobj]
            sbprofiles_obj = sbprofiles[idata][iobj]

            models = data[f'{dataset}_models'][iobj, :, :, :]
            maskbits = data[f'{dataset}_maskbits'][iobj, :, :]

            bands = data[f'{dataset}_bands']
            pixscale = data[f'{dataset}_pixscale']
            wcs = data[f'{dataset}_wcs']

            opt_bx = obj['BX']
            opt_by = obj['BY']
            ellipse_pa = np.radians(obj['PA_MOMENT'] - 90.)
            ellipse_eps = 1 - obj['BA_MOMENT']

            sma_moment = obj['SMA_MOMENT'] # [arcsec]
            label_moment = f'$R(\\mathrm{{mom}})={sma_moment:.1f}$"'
            if idata == 0:
                if pub is not None and d26 > 0.:
                    sma_sbthresh = d26 / 2. * 60.  # arcmin → arcsec
                    ref_str = d26_ref if d26_ref else 'D26'
                else:
                    sma_sbthresh, _, _d26_ref_arr, _ = SGA_diameter(
                        Table(obj), region, radius_arcsec=True)
                    sma_sbthresh = sma_sbthresh[0]
                    ref_str = _d26_ref_arr[0]
                label_sbthresh = f'$R({ref_str})={sma_sbthresh:.1f}$"'

            if have_data:
                bx, by = map_bxby(opt_bx, opt_by, from_wcs=opt_wcs, to_wcs=wcs)
                refg = EllipseGeometry(x0=bx, y0=by, eps=ellipse_eps,
                                       pa=ellipse_pa, sma=sma_moment/pixscale) # sma in pixels
                refap = EllipticalAperture((refg.x0, refg.y0), refg.sma,
                                           refg.sma*(1. - refg.eps), refg.pa)
                refap_sma_sbthresh = EllipticalAperture((refg.x0, refg.y0), sma_sbthresh/pixscale,
                                               sma_sbthresh/pixscale*(1. - refg.eps), refg.pa)

                # a little wasteful...
                masks = unpack_maskbits_function(data[f'{dataset}_maskbits'], bands=bands,
                                                 BITS=MASKBITS[idata])
                masks = masks[iobj, :, :, :]

                wimg = np.sum(images * np.logical_not(masks), axis=0)
                wimg[wimg == 0.] = np.nan
                try:
                    norm = get_norm(wimg)
                except:
                    norm = None

                # col 0 - images
                xx = ax[idata, 0]
                #xx.imshow(np.flipud(jpg), origin='lower', cmap='inferno')
                xx.imshow(wimg, origin='lower', cmap=cmap, interpolation='none',
                          norm=norm, alpha=1.)
                xx.text(0.03, 0.97, label, transform=xx.transAxes,
                        ha='left', va='top', color='white',
                        linespacing=1.5, fontsize=10,
                        bbox=dict(boxstyle='round', facecolor='k', alpha=0.5))
                xx.set_xlim(0, wimg.shape[0]-1)
                xx.set_ylim(0, wimg.shape[0]-1)
                xx.margins(0)
                xx.set_xticks([])
                xx.set_yticks([])

                smas = sbprofiles_obj['SMA'] / pixscale # [pixels]
                sma_pos = smas[smas > 0]
                min_sep = max(2., float(sma_pos[-1]) / 15.) if len(sma_pos) > 0 else 2.
                prev = 0.
                for s in sma_pos:
                    if s - prev >= min_sep:
                        ap = EllipticalAperture((refg.x0, refg.y0), s,
                                                s*(1. - refg.eps), refg.pa)
                        ap.plot(color='k', lw=1, ax=xx)
                        prev = s
                refap.plot(color=colors2[0], lw=2, ls='--', ax=xx)
                refap_sma_sbthresh.plot(color=colors2[1], lw=2, ls='-', ax=xx)

                # col 1 - mag SB profiles
                if linear:
                    yminmax = [1e8, -1e8]
                else:
                    yminmax = [40, 0]

                xx = ax[idata, 1]
                for filt in bands:
                    I = ((sbprofiles_obj[f'SB_{filt.upper()}'].value > 0.) *
                         (sbprofiles_obj[f'SB_ERR_{filt.upper()}'].value > 0.))
                    if np.any(I):
                        sma = sbprofiles_obj['SMA'][I].value**0.25
                        sb = sbprofiles_obj[f'SB_{filt.upper()}'][I].value
                        sberr = sbprofiles_obj[f'SB_ERR_{filt.upper()}'][I].value
                        mu = 22.5 - 2.5 * np.log10(sb)
                        muerr = 2.5 * sberr / sb / np.log(10.)

                        J = muerr < 1. # don't plot everything
                        if np.any(J):
                            col = sbcolors[filt]
                            #xx.plot(sma[J], mu[J]-muerr[J], color=col, alpha=0.8)
                            #xx.plot(sma[J], mu[J]+muerr[J], color=col, alpha=0.8)
                            #xx.scatter(sma[J], mu[J], label=filt, color=col)
                            xx.fill_between(sma[J], mu[J]-muerr[J], mu[J]+muerr[J],
                                            label=f'$\\mu({filt})$', color=col, alpha=0.7)

                        # robust limits
                        mulo = (mu - muerr)[(muerr < 1.) * (mu / muerr > 8.)]
                        muhi = (mu + muerr)[(muerr < 1.) * (mu / muerr > 8.)]
                        #print(filt, np.min(mulo), np.max(muhi))
                        if len(mulo) > 0:
                            mn = np.min(mulo)
                            if mn < yminmax[0]:
                                yminmax[0] = mn
                        if len(muhi) > 0:
                            mx = np.max(muhi)
                            if mx > yminmax[1]:
                                yminmax[1] = mx
                    #print(filt, yminmax[0], yminmax[1])

                xx.margins(x=0)

                if idata == ndataset-1:
                    xx.set_xlabel(r'(Semi-major axis / arcsec)$^{1/4}$')
                else:
                    xx.tick_params(axis='x', labelbottom=False)

                #xx.relim()
                #xx.autoscale_view()
                if linear:
                    ylim = [yminmax[0], yminmax[1]]
                else:
                    ylim = [yminmax[0]-0.75, yminmax[1]+0.5]
                    if ylim[0] < 13:
                        ylim[0] = 13
                    if ylim[1] > 34:
                        ylim[1] = 34
                #print(idata, yminmax, ylim)
                xx.set_ylim(ylim)

                xx_twin = xx.twinx()
                xx_twin.set_ylim(ylim)
                kill_left_y(xx)

                xx.invert_yaxis()
                xx_twin.invert_yaxis()

                if idata == 1:
                    xx_twin.set_ylabel(r'Surface Brightness $\mu$ (mag arcsec$^{-2}$)')

                # Geometry annotation in the lower-left of the first SB panel.
                if idata == 0 and geom_str:
                    xx.text(0.02, 0.03, geom_str, transform=xx.transAxes,
                            ha='left', va='bottom', fontsize=10)

                # Arrows just inside the axis marking size scales.
                # mfc='none' matches the unfilled outline style of the
                # aperture ellipses.
                trans = blended_transform_factory(xx.transData, xx.transAxes)
                if sma_sbthresh > 0.:
                    xx.plot(sma_sbthresh**0.25, 0.94, 'v', mec=colors2[1], mfc='none', ms=9, mew=1.5,
                            transform=trans, clip_on=False, zorder=5)
                xx.plot(sma_moment**0.25, 0.94, 'v', mec=colors2[0], mfc='none', ms=9, mew=1.5,
                        transform=trans, clip_on=False, zorder=5)

                band_hndls, _ = xx.get_legend_handles_labels()
                if band_hndls:
                    if idata == ndataset - 1:
                        size_hndls = []
                        if sma_sbthresh > 0.:
                            size_hndls.append(mlines.Line2D(
                                [], [], mec=colors2[1], mfc='none', mew=1.5,
                                marker='v', linestyle='None', ms=9, label=label_sbthresh))
                        size_hndls.append(mlines.Line2D(
                            [], [], mec=colors2[0], mfc='none', mew=1.5,
                            marker='v', linestyle='None', ms=9, label=label_moment))
                        leg1 = xx.legend(handles=band_hndls, loc='upper right', fontsize=8)
                        xx.legend(handles=size_hndls, loc='lower left', fontsize=11)
                        xx.add_artist(leg1)
                    else:
                        xx.legend(handles=band_hndls, loc='upper right', fontsize=8)
            else:
                ax[idata, 0].text(0.03, 0.97, f'{label} - No Data',
                                  transform=ax[idata, 0].transAxes,
                                  ha='left', va='top', color='white',
                                  linespacing=1.5, fontsize=10,
                                  bbox=dict(boxstyle='round', facecolor='k', alpha=0.5))
                ax[idata, 0].set_xticks([])
                ax[idata, 0].set_yticks([])

                ax[idata, 1].set_yticks([])
                ax[idata, 1].margins(x=0)

                if idata == ndataset-1:
                    ax[idata, 1].set_xlabel(r'(Semi-major axis / arcsec)$^{1/4}$')
                else:
                    ax[idata, 1].tick_params(axis='x', labelbottom=False)


        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(qafile, bbox_inches='tight')
        plt.close()
        log.info(f'Wrote {qafile}')


def make_plots(galaxy, galaxydir, htmlgalaxydir, REFIDCOLUMN, read_multiband_function,
               unpack_maskbits_function, SGAMASKBITS, APERTURES, region='dr11-south',
               run='south', mp=1, bands=['g', 'r', 'i', 'z'], pixscale=0.262,
               galex_pixscale=1.5, unwise_pixscale=2.75, skip_ellipse=False,
               skip_tractor=False, galex=True, unwise=True, barlen=None, barlabel=None,
               verbose=False, clobber=False, fullsample=None):
    """Generate the full set of per-galaxy QA plots (multiband
    montage, ellipse mask, SED/curve-of-growth, and surface-brightness
    profiles) for one galaxy group, by reading its multiband cutouts
    and ellipse-fitting outputs from disk and dispatching to
    :func:`multiband_montage`, :func:`multiband_ellipse_mask`,
    :func:`ellipse_sed`, :func:`ellipse_cog`, and
    :func:`ellipse_sbprofiles`.

    Parameters
    ----------
    galaxy : :class:`str`
        Group name, passed through to `read_multiband_function`.
    galaxydir : :class:`str`
        Directory containing the group's cutout and ellipse-fitting
        FITS files.
    htmlgalaxydir : :class:`str`
        Output directory for the QA PNGs.
    REFIDCOLUMN : :class:`str`
        Reference-ID column name, passed through to
        `read_multiband_function` and :func:`ellipse_sbprofiles`.
    read_multiband_function : callable
        Function used to read the multiband cutout data; called as
        ``read_multiband_function(galaxy, galaxydir, REFIDCOLUMN,
        bands=bands, run=run, pixscale=pixscale,
        galex_pixscale=galex_pixscale, unwise_pixscale=unwise_pixscale,
        unwise=unwise, galex=galex, skip_ellipse=skip_ellipse,
        skip_tractor=skip_tractor, read_jpg=True)`` and expected to
        return ``(data, tractor, sample, samplesrcs, err)``.
    unpack_maskbits_function : callable
        Passed through to :func:`multiband_ellipse_mask` and
        :func:`ellipse_sbprofiles`.
    SGAMASKBITS : :class:`list`
        Per-dataset mask-bit dictionaries, passed through to
        :func:`multiband_ellipse_mask` and :func:`ellipse_sbprofiles`.
    APERTURES : :class:`list` or :class:`numpy.ndarray`
        Aperture radii, passed through to :func:`ellipse_sed`.
    region : :class:`str`
        Imaging region (e.g. ``'dr11-south'``), passed through to
        :func:`ellipse_cog` and :func:`ellipse_sbprofiles`.
    run : :class:`str`
        Photometric system (``'south'`` or ``'north'``), passed
        through to `read_multiband_function` and :func:`ellipse_sed`.
    mp : :class:`int`
        Unused (see Notes).
    bands : :class:`list`
        Optical bandpasses to read.
    pixscale : :class:`float`
        Optical pixel scale, in arcsec/pixel.
    galex_pixscale : :class:`float`
        GALEX pixel scale, in arcsec/pixel.
    unwise_pixscale : :class:`float`
        unWISE pixel scale, in arcsec/pixel.
    skip_ellipse : :class:`bool`
        If True, generate only the multiband montage and skip all
        ellipse-fitting-dependent QA (SED, curve-of-growth,
        surface-brightness profiles).
    skip_tractor : :class:`bool`
        Passed through to `read_multiband_function`.
    galex : :class:`bool`
        Whether to include the GALEX dataset.
    unwise : :class:`bool`
        Whether to include the unWISE dataset.
    barlen : :class:`float`, optional
        Scale-bar length, passed through to :func:`multiband_montage`
        and :func:`multiband_ellipse_mask`.
    barlabel : :class:`str`, optional
        Scale-bar label, passed through to :func:`multiband_montage`
        and :func:`multiband_ellipse_mask`.
    verbose : :class:`bool`
        Unused (see Notes).
    clobber : :class:`bool`
        If True, regenerate QA PNGs even if they already exist;
        passed through to all the plotting functions called here.
    fullsample : :class:`astropy.table.Table`, optional
        Final-catalog table, passed through to all the plotting
        functions called here for title/geometry annotations.

    Returns
    -------
    :class:`int`
        Always returns 1 (success sentinel), including on several
        early-exit paths (`skip_ellipse`, missing ellipse files); see
        Notes.

    Raises
    ------
    IOError
        If the number of ellipse FITS files found does not match the
        number of objects in `sample`.

    Notes
    -----
    `mp` and `verbose` are accepted but never referenced in the
    function body -- dead parameters. The `err` value returned by
    `read_multiband_function` is never checked, so a failed read is
    not distinguished from a successful one at this level. The return
    value is not a real success/failure signal: both the true success
    path and the "all ellipse files missing" early-return produce the
    same ``return 1``, so callers cannot use it to detect that
    ellipse-dependent plots were skipped.

    """
    import fitsio
    from time import time
    from glob import glob
    from SGA.util import get_dt

    tall = time()

    allbands = ''.join(bands)
    datasets = ['opt']
    if unwise:
        datasets += ['unwise']
    if galex:
        datasets += ['galex']

    data, tractor, sample, samplesrcs, err = read_multiband_function(
        galaxy, galaxydir, REFIDCOLUMN, bands=bands, run=run,
        pixscale=pixscale, galex_pixscale=galex_pixscale,
        unwise_pixscale=unwise_pixscale, unwise=unwise,
        galex=galex, skip_ellipse=skip_ellipse, skip_tractor=skip_tractor,
        read_jpg=True)

    multiband_montage(data, sample, htmlgalaxydir, barlen=barlen,
                      barlabel=barlabel, clobber=clobber, fullsample=fullsample)

    # all done!
    if skip_ellipse:
        dt, unit = get_dt(tall)
        log.info(f'Total time to generate plots: {dt:.3f} {unit}')
        return 1

    nsample = len(sample)

    # read the ellipse-fitting results
    ellipsefiles = glob(os.path.join(galaxydir, f'*-ellipse-{allbands}.fits'))
    if len(ellipsefiles) == 0:
        log.warning(f'All ellipse files missing for {galaxydir}/{galaxy}')
        return 1

    if len(ellipsefiles) != len(sample):
        msg = f'Mismatching number of ellipse files and objects in sample in {galaxydir}'
        log.critical(msg)
        raise IOError(msg)

    ellipse = []
    for ellipsefile in ellipsefiles:
        # Read SGANAME from the file itself rather than parsing the filename.
        ellipse_opt = Table(fitsio.read(ellipsefile, ext='ELLIPSE'))
        sganame = ellipse_opt['SGANAME'][0].replace(' ', '_')

        # loop on datasets and join
        for idata, dataset in enumerate(datasets):
            if dataset == 'opt':
                ellipse_dataset = ellipse_opt
            else:
                suffix = dataset
                ellipsefile_dataset = os.path.join(galaxydir, f'{sganame}-ellipse-{suffix}.fits')
                try:
                    ellipse_dataset = Table(fitsio.read(ellipsefile_dataset, ext='ELLIPSE'))
                except:
                    msg = f'Problem reading {ellipsefile_dataset}!'
                    log.critical(msg)
                    break

            if idata == 0:
                ellipse1 = ellipse_dataset
            else:
                ellipse1 = join(ellipse1, ellipse_dataset)

        ellipse.append(ellipse1)

    if len(ellipse) > 0:
        ellipse = vstack(ellipse)

    # sort by optical flux
    ellipse = ellipse[np.argsort(ellipse['OPTFLUX'])[::-1]]

    # Next, read the original images, models, maskbits, and
    # sbprofiles. Store the models, maskbits, and surface-brightness
    # profiles as a nested ellipse [ndataset][nsample].
    sbprofiles = []
    for idata, dataset in enumerate(datasets):
        bands = data[f'{dataset}_bands']
        refband = data[f'{dataset}_refband']
        sz = data[refband].shape

        images = np.zeros((nsample, len(bands), *sz), 'f4')
        models = np.zeros((nsample, len(bands), *sz), 'f4')
        maskbits = np.zeros((nsample, *sz), np.int32)

        original_images = np.stack([data[filt] for filt in data[f'{dataset}_bands']])

        sbprofiles_obj = []
        for iobj, sganame in enumerate(ellipse['SGANAME'].value):
            if dataset == 'opt':
                suffix = allbands
            else:
                suffix = dataset
            ellipsefile = os.path.join(galaxydir, f'{sganame.replace(" ", "_")}-ellipse-{suffix}.fits')

            sbprofiles_obj.append(Table(fitsio.read(ellipsefile, ext='SBPROFILES')))

            maskbits[iobj, :, :] = fitsio.read(ellipsefile, ext='MASKBITS') # [nsample, ny, ny]
            models[iobj, :, :, :] = fitsio.read(ellipsefile, ext='MODELS')  # [nsample, nband, ny, ny]
            images[iobj, :, :, :] = original_images - models[iobj, :, :, :] # [nsample, nband, ny, ny]

        sbprofiles.append(sbprofiles_obj)

        data[f'{dataset}_images'] = images
        data[f'{dataset}_models'] = models
        data[f'{dataset}_maskbits'] = maskbits

        data[f'{dataset}_invvar'] = np.stack([data[f'{filt}_invvar'] for filt in data[f'{dataset}_bands']])

    # ellipse mask
    multiband_ellipse_mask(data, ellipse, htmlgalaxydir, unpack_maskbits_function,
                           SGAMASKBITS, barlen=barlen, barlabel=barlabel,
                           clobber=clobber, fullsample=fullsample)

    # photometry - curve of growth and SED
    ellipse_sed(data, ellipse, htmlgalaxydir, run=run, tractor=samplesrcs,
                apertures=APERTURES, clobber=clobber, fullsample=fullsample)

    ellipse_cog(data, ellipse, sbprofiles, region, htmlgalaxydir,
                datasets=['opt', 'unwise', 'galex'], clobber=clobber, fullsample=fullsample)

    # surface-brightness profiles
    ellipse_sbprofiles(data, ellipse, sbprofiles, region, htmlgalaxydir,
                       unpack_maskbits_function, SGAMASKBITS,
                       REFIDCOLUMN, datasets=['opt', 'unwise', 'galex'],
                       linear=False, clobber=clobber, fullsample=fullsample)

    dt, unit = get_dt(tall)
    log.info(f'Total time to generate plots: {dt:.3f} {unit}')
    return 1

def decode_bitmask(value, bitdict):
    """Decode a packed bitmask integer into the list of set flag names.

    Parameters
    ----------
    value : :class:`int`
        Packed bitmask value.
    bitdict : :class:`dict`
        Mapping of flag name to bit value (e.g. :data:`SGA.SGA.SAMPLE`,
        :data:`SGA.ellipse.ELLIPSEBIT`, or
        :data:`SGA.ellipse.ELLIPSEMODE`).

    Returns
    -------
    :class:`list`
        Names of every flag in `bitdict` whose bit is set in `value`;
        ``['None']`` if no bits are set.

    Notes
    -----
    A near-duplicate, differently-defaulted implementation
    (``_decode_bits``, returning ``''`` rather than ``['None']`` when
    no bits are set, and joined into a single string) is defined as a
    nested helper inside :func:`generate_group_html`'s Tractor
    section, for MASKBITS/FITBITS decoding.

    """
    flags = []
    for name, bit in bitdict.items():
        if value & bit:
            flags.append(name)
    return flags if flags else ['None']

def get_raslice(ra):
    """Compute the 3-digit, zero-padded RA-slice string for a single
    right ascension.

    Parameters
    ----------
    ra : :class:`float`
        Right ascension, in degrees.

    Returns
    -------
    :class:`str`
        3-digit zero-padded RA-slice string, e.g. ``'068'``.

    Notes
    -----
    A separate, near-identical implementation exists at
    :func:`SGA.io.get_raslice` (which additionally vectorizes over
    array input and omits the ``% 360`` wrap present here). The ``%
    360`` in this version only changes behavior for ``ra >= 360`` or
    negative `ra`, which should not occur for valid J2000 coordinates
    in ``[0, 360)``.

    """
    return "{:03d}".format(int(ra) % 360)

def get_galaxy_names(group_dir):
    """Recover the sorted, unique set of per-galaxy SGA names present
    in a group's output directory, by parsing its QA filenames.

    Parameters
    ----------
    group_dir : :class:`str`
        Path to a group's HTML output directory, expected to contain
        files named ``SGA2025_{sganame}-{suffix}.{ext}``.

    Returns
    -------
    :class:`list`
        Sorted unique galaxy-name strings (the ``{sganame}`` portion
        of each matching filename, i.e. everything before the final
        ``-``-delimited suffix).

    Notes
    -----
    Only matches files starting with the literal, version-specific
    prefix ``SGA2025_J`` (via the glob pattern), then re-checks/strips
    the broader ``SGA2025_`` prefix -- filenames not starting with
    ``SGA2025_J`` are silently skipped by the glob itself. Relies on
    ``rsplit('-', 1)`` to peel off the trailing suffix, so a galaxy
    name that itself contains no ``-`` after the prefix strip is
    dropped (``len(parts) >= 2`` check fails silently, no warning).

    """
    galaxy_names = []
    for onefile in glob(os.path.join(group_dir, "SGA2025_J*")):
        stem = os.path.basename(onefile)
        if stem.startswith("SGA2025_"):
            remainder = stem.replace("SGA2025_", "", 1)
            parts = remainder.rsplit("-", 1)
            if len(parts) >= 2:
                galaxy_name = parts[0]
                galaxy_names.append(galaxy_name)
    return np.unique(galaxy_names).tolist()

def get_sky_viewer_url(ra, dec, diameter, region):
    """Build a Legacy Survey sky-viewer URL centered on a group, with a
    zoom level chosen heuristically from the group diameter.

    Parameters
    ----------
    ra, dec : :class:`float`
        Group center coordinates, in degrees.
    diameter : :class:`float`
        Group angular diameter, in arcmin; used only to set the
        `zoom` heuristic.
    region : :class:`str`
        Processing region; ``'dr11-south'`` selects the
        ``ls-dr11-south`` imaging layer, anything else (including
        ``'dr11-north'``) selects ``ls-dr11-north``.

    Returns
    -------
    :class:`str`
        Fully-formed ``legacysurvey.org/viewer-dev`` URL with the
        ``sga2025-parent`` catalog overlay enabled.

    Notes
    -----
    `zoom` is clamped to ``[11, 16]``; a commented-out alternate
    formula (a different clamp range and coefficient) is left in the
    source as a dead comment, suggesting the heuristic was tuned by
    trial and error and not finalized/cleaned up.

    """
    if region == 'dr11-south':
        layer = 'ls-dr11-south'
    else:
        layer = 'ls-dr11-north'
    diam_arcmin = diameter
    zoom = max(11, min(16, int(14 - np.log10(max(0.3, diam_arcmin)))))
    #zoom = max(10, min(16, int(16 - np.log10(max(1.0, diam_arcmin)))))
    url = "https://www.legacysurvey.org/viewer-dev/?ra={:.4f}&dec={:.4f}&layer={}&zoom={}&sga2025-parent".format(
        ra, dec, layer, zoom)
    return url

def find_group_directory(htmldir, region, group_name):
    """Locate a group's HTML output directory from its name, assuming
    the standard ``{htmldir}/{region}/{raslice}/{group_name}`` sharded
    layout.

    Parameters
    ----------
    htmldir : :class:`pathlib.Path`
        Root HTML output directory.
    region : :class:`str`
        Processing region subdirectory name (e.g. ``'dr11-south'``).
    group_name : :class:`str`
        Group name; its first three characters are used directly as
        the RA-slice subdirectory (rather than recomputing it via
        :func:`get_raslice`).

    Returns
    -------
    :class:`pathlib.Path` or None
        The group directory, or None if it does not exist on disk.

    """
    raslice = group_name[:3]
    group_dir = htmldir / region / raslice / group_name
    if group_dir.exists():
        return group_dir
    return None

def generate_group_html(group_data, fullsample, htmldir, region, prev_group, next_group, clobber=False, fulltractor=None):
    """Build and write the static HTML QA page for a single galaxy
    group: header/thumbnail, group properties, per-object identifiers,
    redshift/distance, multiwavelength montage image, ellipse
    masking/geometry, optional Tractor photometry, aggregate
    curve-of-growth/aperture photometry table, and per-galaxy
    SED/CoG/surface-brightness figure thumbnails.

    Parameters
    ----------
    group_data : :class:`astropy.table.Table`
        Single-group slice of the parent sample (all rows share one
        ``GROUP_NAME``); only its first row is consulted for
        group-level properties (name, RA/Dec, diameter, multiplicity).
    fullsample : :class:`astropy.table.Table`
        Full per-region sample table, used to look up every object
        belonging to this group (via ``GROUP_NAME``) for the
        per-object tables.
    htmldir : :class:`pathlib.Path`
        Root HTML output directory.
    region : :class:`str`
        Processing region (e.g. ``'dr11-south'``).
    prev_group, next_group : :class:`str` or None
        Group names for the "Previous"/"Next" navbar links, or None to
        omit the corresponding link.
    clobber : :class:`bool`
        If True, regenerate the page even if it already exists.
    fulltractor : :class:`astropy.table.Table`, optional
        Full per-region Tractor catalog, row-aligned with
        `fullsample`, used to populate the optional Tractor section.
        If None, that section is omitted entirely.

    Returns
    -------
    :class:`bool`
        False if the group's output directory cannot be found (logged
        as a warning); True otherwise, including when the page already
        exists and `clobber` is False (page is simply skipped).

    Notes
    -----
    `fulltractor` is assumed row-aligned with `fullsample` -- both are
    indexed with the same boolean mask
    (``fullsample['GROUP_NAME'] == group_name``) without any
    additional matching/validation, so if the two tables are ever
    constructed independently or in different orders, the Tractor
    section would silently pair the wrong rows. The per-object sort by
    ``D26``/``DIAM_INIT`` (descending) is applied to both
    `fullgroup_data` and `tractor_group` together, so alignment is at
    least preserved through the sort.

    A local variable named ``galaxy`` is set once from `group_data`
    for the page title/heading (before any per-object loop), then
    reassigned inside the "Identifiers" per-row loop as a loop-body
    temporary reusing the same name -- harmless (the outer use already
    happened), but a naming collision that could confuse future
    edits.

    """
    group_name = group_data['GROUP_NAME'][0]
    group_dir = find_group_directory(htmldir, region, group_name)
    if group_dir is None:
        log.warning("Error: Could not find directory for group {} in region {}".format(group_name, region))
        return False
    output_file = group_dir / "{}.html".format(group_name)
    if output_file.exists() and not clobber:
        log.debug("Skipping (exists): {}".format(output_file))
        return True
    _group_mask = np.isin(fullsample['GROUP_NAME'], group_data['GROUP_NAME'])
    fullgroup_data = fullsample[_group_mask]
    tractor_group = fulltractor[_group_mask] if fulltractor is not None else None

    if len(fullgroup_data) > 1:
        d26_col = 'D26' if 'D26' in fullgroup_data.colnames else ('DIAM_INIT' if 'DIAM_INIT' in fullgroup_data.colnames else None)
        if d26_col is not None:
            _sidx = np.argsort([float(r[d26_col]) for r in fullgroup_data])[::-1]
            fullgroup_data = fullgroup_data[_sidx]
            if tractor_group is not None:
                tractor_group = tractor_group[_sidx]
    if 'GALAXY' in fullgroup_data.columns:
        galaxy = fullgroup_data['GALAXY'][0]
    else:
        galaxy = group_data['OBJNAME'][0]
    sganame = group_data['SGANAME'][0]
    group_ra = group_data['GROUP_RA'][0]
    group_dec = group_data['GROUP_DEC'][0]
    group_diam = group_data['GROUP_DIAMETER'][0]
    group_mult = group_data['GROUP_MULT'][0]

    raslice = group_name[:3]
    sky_url = get_sky_viewer_url(group_ra, group_dec, group_diam, region)
    data_url = "https://portal.nersc.gov/project/cosmo/data/sga/2025/data/{}/{}/{}".format(region, raslice, group_name)
    group_files = [
        "SGA2025_{}-montage.png".format(group_name),
        "SGA2025_{}-ellipsemask.png".format(group_name),
    ]
    per_galaxy_types = ["sbprofiles", "cog", "sed"]
    per_galaxy_titles = ["Surface Brightness", "Curve of Growth", "Spectral Energy Distribution"]
    galaxy_names = get_galaxy_names(str(group_dir))
    if len(galaxy_names) > 1 and len(fullgroup_data) > 0 and 'SGANAME' in fullgroup_data.colnames:
        gname_set = set(galaxy_names)
        ordered = [str(row['SGANAME']).strip().replace('SGA2025 ', '')
                   for row in fullgroup_data
                   if str(row['SGANAME']).strip().replace('SGA2025 ', '') in gname_set]
        ordered += [n for n in galaxy_names if n not in ordered]
        galaxy_names = ordered

    # -----------------------------------------------------------------------
    # formatting helpers
    # -----------------------------------------------------------------------
    _cols = set(fullgroup_data.colnames)

    def _has(col):
        """True if `col` is a column of `fullgroup_data` (checked once
        via the precomputed ``_cols`` set).

        """
        return col in _cols

    def _get(row, col, default=None):
        """Return ``row[col]`` if `col` exists in `fullgroup_data`,
        else `default`, without raising on missing columns.

        """
        return row[col] if col in _cols else default

    def _sf(val, zero_missing=True):
        """Coerce `val` to a Python float, or None if it is
        non-numeric, NaN, or (when `zero_missing`) exactly zero.

        Parameters
        ----------
        val : any
            Value to coerce; typically a table cell that may be
            masked or a fill value.
        zero_missing : :class:`bool`
            If True, treat an exact 0.0 as missing (returns None) --
            appropriate for columns where 0 is a fill/sentinel value
            rather than a real measurement.

        Returns
        -------
        :class:`float` or None

        """
        try:
            v = float(val)
            if np.isnan(v):
                return None
            if zero_missing and v == 0.:
                return None
            return v
        except (TypeError, ValueError):
            return None

    def _fmt_z(row, vcol, icol=''):
        """Format a redshift value (and, if `icol` is given and its
        inverse-variance is positive, its 1-sigma uncertainty) from
        `row` as a display string; ``''`` if `vcol` is missing/zero.

        """
        v = _sf(_get(row, vcol))
        if v is None:
            return ''
        iv = _sf(_get(row, icol), zero_missing=False) if icol else None
        if iv is not None and iv > 0:
            return f'{v:.5f} ± {1./np.sqrt(iv):.5f}'
        return f'{v:.5f}'

    def _fmt_dist(row, vcol, icol=''):
        """Format a distance value (and optional uncertainty from
        `icol`) from `row` as a display string; ``''`` if `vcol` is
        missing, zero, or negative.

        """
        v = _sf(_get(row, vcol), zero_missing=True)
        if v is None or v <= 0:
            return ''
        iv = _sf(_get(row, icol), zero_missing=False) if icol else None
        if iv is not None and iv > 0:
            return f'{v:.2f} ± {1./np.sqrt(iv):.2f}'
        return f'{v:.2f}'   # distance known; ivar=0 is expected in some cases

    def _fmt_diam(row):
        """Format the adopted diameter and its reference string for
        `row`, preferring ``D26``/``D26_ERR``/``D26_REF`` when present
        and positive, else falling back to
        ``DIAM_INIT``/``DIAM``/``INIT_REF``/``DIAM_INIT_REF``.

        Returns
        -------
        :class:`tuple`
            ``(diameter_string, reference_string)``, both ``''`` if no
            diameter is available.

        """
        if _has('D26'):
            d = _sf(_get(row, 'D26'), zero_missing=True)
            if d is not None and d > 0:
                de = _sf(_get(row, 'D26_ERR'), zero_missing=False)
                ref = str(_get(row, 'D26_REF', '') or '').strip()
                if de is not None and de > 0:
                    return f'{d:.3f} ± {de:.3f}', ref
                return f'{d:.3f}', ref
        # fallback to initial diameter
        d = _sf(_get(row, 'DIAM_INIT') or _get(row, 'DIAM'), zero_missing=True)
        ref = str(_get(row, 'INIT_REF', '') or _get(row, 'DIAM_INIT_REF', '') or '').strip()
        if d is not None:
            return f'{d:.3f}', ref
        return '', ''

    def _fmt_mag_cog(row, band):
        """Format the total curve-of-growth magnitude (and
        uncertainty, if available) for `band` from `row`; ``''`` if
        ``COG_MTOT_{band}`` is missing or non-positive.

        """
        m = _sf(_get(row, f'COG_MTOT_{band}'), zero_missing=True)
        if m is None or m <= 0:
            return ''
        me = _sf(_get(row, f'COG_MTOT_ERR_{band}'), zero_missing=False)
        if me is not None and me > 0:
            return f'{m:.3f} ± {me:.3f}'
        return f'{m:.3f}'

    def _fmt_mag_ap(row, ap, band):
        """Convert the aperture flux ``FLUX_AP{ap:02d}_{band}`` from
        `row` to an AB magnitude (and propagated uncertainty, if
        available); ``''`` if the flux is missing or non-positive.

        """
        f = _sf(_get(row, f'FLUX_AP{ap:02d}_{band}'), zero_missing=True)
        if f is None or f <= 0:
            return ''
        mag = 22.5 - 2.5 * np.log10(f)
        fe = _sf(_get(row, f'FLUX_ERR_AP{ap:02d}_{band}'), zero_missing=False)
        if fe is not None and fe > 0:
            return f'{mag:.3f} ± {2.5*fe/f/np.log(10.):.3f}'
        return f'{mag:.3f}'

    def _th(*cells):
        """Build one HTML ``<tr>`` of ``<th>`` header cells.

        Parameters
        ----------
        *cells : :class:`str` or :class:`tuple`
            Each cell is either a plain header-text string, or a
            ``(text, colspan)`` or ``(text, colspan, rowspan)`` tuple;
            a trailing span of exactly 1 is omitted from the emitted
            HTML.

        Returns
        -------
        :class:`str`
            One indented ``<tr>...</tr>`` HTML line.

        """
        parts = []
        for c in cells:
            if isinstance(c, tuple):
                text = c[0]
                attrs = (f" colspan='{c[1]}'" if len(c) > 1 and c[1] != 1 else '') + \
                        (f" rowspan='{c[2]}'" if len(c) > 2 and c[2] != 1 else '')
                parts.append(f'<th{attrs}>{text}</th>')
            else:
                parts.append(f'<th>{c}</th>')
        return '        <tr>' + ''.join(parts) + '</tr>'

    def _td(*cells):
        """Build one HTML ``<tr>`` of plain ``<td>`` data cells from
        `cells`, in order.

        """
        return '        <tr>' + ''.join(f'<td>{c}</td>' for c in cells) + '</tr>'

    def _ned_link(name):
        """Wrap `name` in an HTML link to its NED by-name query page,
        URL-encoding ``+`` and spaces.

        """
        encoded = name.replace('+', '%2B').replace(' ', '+')
        return f"<a href='https://ned.ipac.caltech.edu/byname?objname={encoded}' target='_blank'>{name}</a>"

    def _pgc_link(pgc):
        """Wrap `pgc` in an HTML link to its HyperLeda PGC catalog
        page.

        """
        return f"<a href='http://atlas.obs-hp.fr/hyperleda/ledacat.cgi?o=%23{pgc}' target='_blank'>{pgc}</a>"

    phot_bands = ['G', 'R', 'I', 'Z', 'W1', 'W2', 'FUV', 'NUV']

    # -----------------------------------------------------------------------
    # HTML construction
    # -----------------------------------------------------------------------
    html_lines = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        "    <meta charset='utf-8'>",
        "    <title>SGA2025: {}</title>".format(galaxy),
        "    <style>",
        "        body { font-family: Arial, sans-serif; margin: 0; padding: 60px 20px 20px 20px; }",
        "        .navbar { position: fixed; top: 0; left: 0; right: 0; background-color: #333; padding: 10px 20px; z-index: 1000; }",
        "        .navbar a { color: white; text-decoration: none; margin-right: 20px; font-weight: bold; }",
        "        .navbar a:hover { text-decoration: underline; }",
        "        .breadcrumb { color: #666; margin-bottom: 10px; font-size: 14px; }",
        "        .breadcrumb a { color: #0066cc; text-decoration: none; }",
        "        .breadcrumb a:hover { text-decoration: underline; }",
        "        h1 { color: #333; margin-bottom: 5px; }",
        "        h2 { color: #111; margin-top: 28px; margin-bottom: 8px; font-weight: bold; font-size: 19px; border-bottom: 2px solid #ddd; padding-bottom: 4px; }",
        "        h3 { color: #666; margin-top: 2px; margin-bottom: 16px; font-weight: normal; font-size: 15px; }",
        "        table { border-collapse: collapse; margin: 20px 0; font-size: 14px; }",
        "        th { background-color: #f0f0f0; padding: 6px 10px; border: 1px solid #ddd; text-align: center; white-space: nowrap; font-size: 14px; }",
        "        td { padding: 6px 10px; border: 1px solid #ddd; vertical-align: top; text-align: center; font-size: 14px; }",
        "        td.gal-group { font-weight: bold; white-space: nowrap; }",
        "        tr.gal-even td { background-color: #eef4fb; }",
        "        tr.gal-odd td { background-color: #ffffff; }",
        "        .section { margin: 30px 0; }",
        "        .img-table-row { display: flex; gap: 24px; align-items: flex-start; margin: 20px 0; }",
        "        .img-col { flex: 0 0 56%; max-width: 56%; }",
        "        .img-col img { max-width: 100%; height: auto; display: block; }",
        "        .tables-col { flex: 1; min-width: 0; overflow-x: auto; }",
        "        .galaxy-row { display: flex; gap: 10px; margin: 10px 0; justify-content: space-between; align-items: center; }",
        "        .galaxy-row a { display: block; flex: 0 0 32%; min-height: 200px; }",
        "        .galaxy-row img { width: 100%; height: auto; display: block; }",
        "        .galaxy-row div { flex: 1; max-width: 32%; }",
        "        img { border: 1px solid #ddd; }",
        "        .warn { color: #c0392b; font-weight: bold; }",
        "        .muted { color: #888; font-size: 11px; }",
        "        .toc { position: fixed; top: 52px; right: 10px; background: rgba(255,255,255,0.95); border: 1px solid #ccc; border-radius: 4px; padding: 8px 12px; font-size: 12px; max-width: 175px; z-index: 999; box-shadow: 0 2px 5px rgba(0,0,0,0.12); }",
        "        .toc-title { font-weight: bold; font-size: 10px; text-transform: uppercase; color: #999; letter-spacing: 0.5px; margin-bottom: 5px; }",
        "        .toc a { display: block; color: #555; text-decoration: none; padding: 1px 0; line-height: 1.6; }",
        "        .toc a:hover { color: #0066cc; text-decoration: underline; }",
        "    </style>",
        "</head>",
        "<body>",
        "    <div class='navbar'>",
        "        <a href='../../../index-{}.html'>Home</a>".format(region),
        "        <a href='../../{}/{}/{}.html'>Previous ({})</a>".format(prev_group[:3], prev_group, prev_group, prev_group) if prev_group else "",
        "        <a href='../../{}/{}/{}.html'>Next ({})</a>".format(next_group[:3], next_group, next_group, next_group) if next_group else "",
        "        <a href='{}' target='_blank'>Sky Viewer</a>".format(sky_url),
        "    </div>",
        "    <div class='breadcrumb'>",
        "        <a href='../../../index-{}.html'>Home</a> &gt; ".format(region),
        "        <a href='../../../index-{}.html'>RA {}</a> &gt; {}".format(region, raslice, galaxy),
        "    </div>",
    ]

    _toc = [
        "        <a href='#sec-group'>Group Properties</a>",
        "        <a href='#sec-identifiers'>Identifiers</a>",
        "        <a href='#sec-redshift'>Redshift &amp; Distance</a>",
        "        <a href='#sec-montage'>Multiwavelength Montage</a>",
        "        <a href='#sec-ellipse'>Ellipse Masking</a>",
    ]
    if tractor_group is not None:
        _toc.append("        <a href='#sec-tractor'>Tractor</a>")
    _toc.append("        <a href='#sec-photometry'>Photometry (Data)</a>")
    _toc.append("        <a href='#sec-figures'>Photometry (Figures)</a>")
    html_lines.extend([
        "    <div class='toc'>",
        "        <div class='toc-title'>Contents</div>",
        *_toc,
        "    </div>",
    ])

    thumb_src  = 'SGA2025_{}-thumb.jpg'.format(group_name)
    has_thumb  = (group_dir / thumb_src).exists()
    _links     = "        <p><a href='{}' target='_blank'>Sky Viewer</a> &nbsp;|&nbsp; <a href='{}' target='_blank'>Group Files</a></p>".format(sky_url, data_url)
    if has_thumb:
        html_lines.extend([
            "    <div style='display:flex; align-items:flex-start; gap:20px; margin-bottom:8px;'>",
            "        <img src='{}' style='width:160px; height:160px; object-fit:cover; border:1px solid #ddd; flex-shrink:0;'>".format(thumb_src),
            "        <div style='min-width:0;'>",
            "            <h1 style='margin-top:0;'>{}</h1>".format(galaxy),
            "            <h3>Group: {} | RA Slice: {}</h3>".format(group_name, raslice),
            _links,
            "        </div>",
            "    </div>",
        ])
    else:
        html_lines.extend([
            "    <h1>{}</h1>".format(galaxy),
            "    <h3>Group: {} | RA Slice: {}</h3>".format(group_name, raslice),
            _links,
        ])

    # --- Group summary -------------------------------------------------------
    html_lines.append("    <h2 id='sec-group'>Group Properties</h2>")
    html_lines.append("    <table>")
    html_lines.append(_th(
        ('Group Name', 1, 2), ('Group RA', 1), ('Group Dec', 1), ('Group Diameter', 1),
        ('Group Multiplicity', 1, 2), ('Region', 1, 2), ('Bands', 1, 2),
    ))
    html_lines.append(_th('(deg)', '(deg)', '(arcmin)'))
    _bands_s = str(_get(fullgroup_data[0], 'BANDS', '') or '').strip() if len(fullgroup_data) > 0 else ''
    html_lines.append("        <tr><td>{}</td><td>{:.6f}</td><td>{:.6f}</td><td>{:.2f}</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
        group_name, group_ra, group_dec, group_diam, group_mult, region, _bands_s))
    html_lines.append("    </table>")

    # ---- Section A: Identifiers, Redshift & Distance, Multiwavelength Montage
    if len(fullgroup_data) > 0:

        # --- Identifiers -----------------------------------------------------
        html_lines.append("    <h2 id='sec-identifiers'>Identifiers</h2>")
        html_lines.append("    <table>")
        html_lines.append(_th(
            ('Galaxy', 1, 2), ('SGA ID', 1, 2), ('SGA Name', 1, 2), ('PGC', 1, 2),
            ('Morphology', 1, 2),
            'RA', 'Dec', 'E(B-V)',
            ('Primary', 1, 2), ('Alternate Names', 1, 2),
            'Object Name',
        ))
        html_lines.append(_th('(deg)', '(deg)', '(mag)', '(internal)'))
        for row in fullgroup_data:
            galaxy   = str(_get(row, 'GALAXY', '') or '').strip() or str(row['OBJNAME']).strip()
            altnames = str(_get(row, 'ALTNAMES', '') or '').strip()
            sganame  = str(_get(row, 'SGANAME', '') or '').strip()
            _pgc_v   = _sf(row['PGC'], zero_missing=True)
            pgc      = int(_pgc_v) if _pgc_v is not None and int(_pgc_v) != -99 and int(_pgc_v) != -1 else ''
            morph    = str(_get(row, 'MORPH', '') or '').strip()
            primary  = 'Yes' if row['GROUP_PRIMARY'] else 'No'
            html_lines.append(_td(
                _ned_link(galaxy), row['SGAID'], sganame, _pgc_link(pgc) if pgc else '', morph,
                f"{float(row['RA']):.6f}", f"{float(row['DEC']):.6f}",
                f"{float(row['EBV']):.4f}",
                primary, altnames, row['OBJNAME'],
            ))
        html_lines.append("    </table>")

        # --- Redshift & Distance (only when Z present) -----------------------
        if _has('Z'):
            has_photz = _has('Z_PHOT')
            has_desi  = _has('Z_DESI')
            has_sdss  = _has('Z_SDSS')
            has_ned   = _has('Z_NED')
            has_lvd   = _has('Z_LVD')
            html_lines.append("    <h2 id='sec-redshift'>Redshift &amp; Distance</h2>")
            html_lines.append("    <table>")
            hdr1 = [('Galaxy', 1, 3), ('Adopted', 5)]
            if has_photz:
                hdr1.append(('Photo-z', 1))
            if has_desi:
                hdr1.append(('DESI', 4))
            if has_sdss:
                hdr1.append(('SDSS', 2))
            if has_ned:
                hdr1.append(('NED', 4))
            if has_lvd:
                hdr1.append(('LVD', 2))
            html_lines.append(_th(*hdr1))
            hdr2 = ['', '', 'Distance', '', '']
            hdr3 = ['Redshift', 'Ref', '(Mpc)', 'Ref', 'Method']
            if has_photz:
                hdr2 += ['']
                hdr3 += ['Redshift']
            if has_desi:
                hdr2 += ['', '', '', '']
                hdr3 += ['Redshift', 'ZWARN', 'SPECTYPE', 'N Spectra']
            if has_sdss:
                hdr2 += ['', '']
                hdr3 += ['Redshift', 'CLASS']
            if has_ned:
                hdr2 += ['', 'Mean Dist.', 'Direct Dist.', '']
                hdr3 += ['Redshift', '(Mpc)', '(Mpc)', 'Method']
            if has_lvd:
                hdr2 += ['', 'Distance']
                hdr3 += ['Redshift', '(Mpc)']
            html_lines.append(_th(*hdr2))
            html_lines.append(_th(*hdr3))
            for row in fullgroup_data:
                z_flag = int(_get(row, 'Z_FLAG', 0) or 0)
                if z_flag:
                    _flag_names = decode_bitmask(z_flag, Z_FLAG)
                    _is_bad = bool(z_flag & Z_FLAG['DISCREPANCY'])
                    _icon = '⚠' if _is_bad else 'ⓘ'
                    _cls  = 'warn' if _is_bad else 'muted'
                    z_flag_s = f" <span class='{_cls}' title='{', '.join(_flag_names)}'>{_icon}</span>"
                else:
                    z_flag_s = ''
                z_ref       = str(_get(row, 'Z_REF', '') or '').strip()
                dist_method = str(_get(row, 'DIST_METHOD', '') or '').strip()
                _gname_z = str(_get(row, 'GALAXY', '') or row['OBJNAME']).strip()
                _sgaid_z = f'  [{int(row["SGAID"])}]' if _has('SGAID') else ''
                cells = [
                    _ned_link(_gname_z) + _sgaid_z,
                    _fmt_z(row, 'Z', 'Z_IVAR') + z_flag_s,
                    z_ref,
                    _fmt_dist(row, 'DIST', 'DIST_IVAR'),
                    str(_get(row, 'DIST_REF', '') or '').strip(),
                    dist_method,
                ]
                if has_photz:
                    cells += [_fmt_z(row, 'Z_PHOT', 'Z_PHOT_IVAR')]
                if has_desi:
                    z_desi_s = _fmt_z(row, 'Z_DESI', 'Z_IVAR_DESI')
                    zwarn_raw = _get(row, 'ZWARN_DESI')
                    zwarn    = int(zwarn_raw) if zwarn_raw is not None else -1
                    spectype = str(_get(row, 'SPECTYPE_DESI', '') or '').strip()
                    nspec    = int(_get(row, 'NSPEC_DESI', 0) or 0)
                    if zwarn < 0:
                        zwarn_s = ''
                    elif zwarn == 0:
                        zwarn_s = str(zwarn)
                    else:
                        zwarn_s = f"<span class='warn'>{zwarn}</span>"
                    cells += [z_desi_s, zwarn_s, spectype, nspec if nspec else '']
                if has_sdss:
                    class_sdss = str(_get(row, 'CLASS_SDSS', '') or '').strip()
                    cells += [_fmt_z(row, 'Z_SDSS', 'Z_IVAR_SDSS'), class_sdss]
                if has_ned:
                    cells += [
                        _fmt_z(row, 'Z_NED', 'Z_IVAR_NED'),
                        _fmt_dist(row, 'DIST_NED', 'DIST_IVAR_NED'),
                        _fmt_dist(row, 'DIST_NED_DIRECT', 'DIST_IVAR_NED_DIRECT'),
                        str(_get(row, 'DIST_NED_DIRECT_METHOD', '') or '').strip(),
                    ]
                if has_lvd:
                    cells += [
                        _fmt_z(row, 'Z_LVD', 'Z_IVAR_LVD'),
                        _fmt_dist(row, 'DIST_LVD', 'DIST_IVAR_LVD'),
                    ]
                html_lines.append(_td(*cells))
            html_lines.append("    </table>")
            html_lines.append(
                "    <p class='muted'>Reference abbreviations (<code>Z_REF</code>/<code>DIST_REF</code>/"
                "<code>DIST_METHOD</code>) and the redshift-quality flag (hover the ⚠/ⓘ icon) are defined in the "
                f"<a href='{SGA_DOCS_URL}#redshifts' target='_blank'>Redshifts</a> and "
                f"<a href='{SGA_DOCS_URL}#distances' target='_blank'>Distances</a> sections of the data-model "
                "documentation.</p>"
            )

    # Multiwavelength Montage image (full-width)
    html_lines.append("    <h2 id='sec-montage'>Multiwavelength Montage</h2>")
    filepath = group_dir / group_files[0]
    if filepath.exists():
        html_lines.append("    <a href='{}'><img src='{}' alt='{}' style='max-width: 100%; height: auto; display: block; margin: 10px 0;'></a>".format(group_files[0], group_files[0], group_files[0]))
    else:
        html_lines.append("    <p style='color: #888;'>Missing: {}</p>".format(group_files[0]))

    # ---- Section B: Size & Geometry, Ellipse Mask image, Processing Metadata -
    html_lines.append("")
    html_lines.append("    <h2 id='sec-ellipse'>Ellipse Masking &amp; Geometry</h2>")
    if len(fullgroup_data) > 0:

        # --- Size & Geometry (full-width, above ellipse mask image) ----------
        #html_lines.append("    <h2>Size &amp; Geometry</h2>")
        html_lines.append("    <table>")
        if _has('D26'):
            html_lines.append(_th(
                ('Galaxy', 1, 3),
                ('Fitted', 5), ('Initial', 4), ('Flags', 3),
            ))
            html_lines.append(_th(
                'D(26)', '', '', 'PA', 'Moment SMA',
                'D', '', 'PA', '',
                '', '', '',
            ))
            html_lines.append(_th(
                '(arcmin)', 'Ref', 'b/a', '(deg)', '(arcsec)',
                '(arcmin)', 'b/a', '(deg)', 'Ref',
                'SAMPLE', 'ELLIPSEBIT', 'ELLIPSEMODE',
            ))
        else:
            html_lines.append(_th('Galaxy', 'D (arcmin)', 'b/a', 'PA (°)', 'Ref',
                                  'SAMPLE', 'ELLIPSEBIT', 'ELLIPSEMODE'))
        for row in fullgroup_data:
            d_init       = _sf(_get(row, 'DIAM_INIT') or _get(row, 'DIAM'))
            ba_init      = _sf(_get(row, 'BA_INIT') or _get(row, 'BA'))
            pa_init      = _sf(_get(row, 'PA_INIT') or _get(row, 'PA'))
            init_ref     = str(_get(row, 'INIT_REF', '') or _get(row, 'DIAM_INIT_REF', '') or '').replace('NONE', 'None').strip()
            d_init_s     = f'{d_init:.3f}' if d_init else ''
            ba_init_s    = f'{ba_init:.3f}' if ba_init else ''
            pa_init_s    = f'{pa_init:.1f}' if pa_init is not None else ''
            sample_flags = ', '.join(decode_bitmask(row['SAMPLE'], SAMPLE)) or ''
            ebit_flags   = ', '.join(decode_bitmask(_get(row, 'ELLIPSEBIT', 0) or 0, ELLIPSEBIT)) or ''
            emode_flags  = ', '.join(decode_bitmask(row['ELLIPSEMODE'], ELLIPSEMODE)) or ''
            if _has('D26'):
                d26_s, d26_ref = _fmt_diam(row)
                sma_mom   = _sf(_get(row, 'SMA_MOMENT'), zero_missing=True)
                sma_mom_s = f'{sma_mom:.1f}' if sma_mom else ''
                ba_s  = f'{float(row["BA"]):.3f}' if _sf(_get(row, "BA")) else ''
                pa_s  = f'{float(row["PA"]):.1f}'  if _sf(_get(row, "PA"), zero_missing=False) is not None else ''
                _gname = str(_get(row, 'GALAXY', '') or row['OBJNAME']).strip()
                _sgaid_g = f'  [{int(row["SGAID"])}]' if _has('SGAID') else ''
                html_lines.append(_td(_ned_link(_gname) + _sgaid_g,
                                      d26_s, d26_ref, ba_s, pa_s, sma_mom_s,
                                      d_init_s, ba_init_s, pa_init_s, init_ref,
                                      sample_flags, ebit_flags, emode_flags))
            else:
                html_lines.append(_td(_ned_link(str(row['OBJNAME']).strip()),
                                      d_init_s, ba_init_s, pa_init_s, init_ref,
                                      sample_flags, ebit_flags, emode_flags))
        html_lines.append("    </table>")
        html_lines.append(
            "    <p class='muted'><code>SAMPLE</code>, <code>ELLIPSEBIT</code>, and <code>ELLIPSEMODE</code> "
            "flag names are defined in the "
            f"<a href='{SGA_DOCS_URL}#bitmasks' target='_blank'>Bitmasks</a> section of the data-model "
            "documentation.</p>"
        )

    # ellipse mask image (full-width)
    filepath = group_dir / group_files[1]
    if filepath.exists():
        html_lines.append("    <a href='{}'><img src='{}' alt='{}' style='max-width: 100%; height: auto; display: block; margin: 10px 0;'></a>".format(group_files[1], group_files[1], group_files[1]))
    else:
        html_lines.append("    <p style='color: #888;'>Missing: {}</p>".format(group_files[1]))

    # ---- Section C: Tractor --------------------------------------------------
    if tractor_group is not None:
        try:
            from legacypipe.bits import MASKBITS as LP_MASKBITS, FITBITS as LP_FITBITS
            have_lp_bits = True
        except ImportError:
            log.warning('legacypipe not available; MASKBITS/FITBITS shown as integers')
            have_lp_bits = False
            LP_MASKBITS = {}
            LP_FITBITS  = {}

        _tcols = set(tractor_group.colnames)

        def _fmt_tractor_mag(trow, band):
            """Convert Tractor's ``FLUX_{band}``/``FLUX_IVAR_{band}``
            for `trow` to an AB magnitude with propagated
            uncertainty; ``''`` if the flux column is absent or the
            flux is non-positive.

            """
            fc, ic = f'FLUX_{band}', f'FLUX_IVAR_{band}'
            if fc not in _tcols:
                return ''
            f = float(trow[fc])
            if f <= 0:
                return ''
            mag = 22.5 - 2.5 * np.log10(f)
            if ic in _tcols:
                iv = float(trow[ic])
                if iv > 0:
                    return f'{mag:.3f} ± {2.5 / np.log(10.) / (np.sqrt(iv) * f):.3f}'
            return f'{mag:.3f}'

        def _decode_bits(val, bits_dict):
            """Decode a Tractor MASKBITS/FITBITS integer `val` against
            `bits_dict` into a comma-separated flag-name string;
            falls back to the raw integer string if `val` is nonzero
            but no name matches (e.g. `bits_dict` is empty because
            ``legacypipe`` was not importable).

            """
            v = int(val)
            if v == 0:
                return ''
            names = [k for k, bit in bits_dict.items() if v & bit]
            return ', '.join(names) if names else str(v)

        html_lines.append("    <h2 id='sec-tractor'>Tractor</h2>")
        html_lines.append("    <table>")
        html_lines.append(_th(
            ('Galaxy', 1, 2), 'RA', 'Dec', ('Type', 1, 2),
            ('Sérsic n', 1, 2),
            'r<sub>50</sub>',
            ('Optical (AB mag)', 3),
            ('MASKBITS', 1, 2), ('FITBITS', 1, 2),
        ))
        html_lines.append(_th('(deg)', '(deg', '(arcsec)', 'g', 'r', 'z'))
        for row, tr in zip(fullgroup_data, tractor_group):
            _gname = str(_get(row, 'GALAXY', '') or row['OBJNAME']).strip()
            _sgaid_s = f'  [{int(row["SGAID"])}]' if _has('SGAID') else ''
            ra_s    = f"{float(tr['RA']):.6f}"   if 'RA'      in _tcols else ''
            dec_s   = f"{float(tr['DEC']):.6f}"  if 'DEC'     in _tcols else ''
            typ     = str(tr['TYPE']).strip()     if 'TYPE'    in _tcols else ''
            if 'SERSIC' in _tcols:
                _sv = float(tr['SERSIC'])
                sersic = f'{_sv:.2f}' if _sv > 0 else ''
            else:
                sersic = ''
            shape_r = f"{float(tr['SHAPE_R']):.3f}" if 'SHAPE_R' in _tcols else ''
            mb_s = (_decode_bits(tr['MASKBITS'], LP_MASKBITS)
                    if 'MASKBITS' in _tcols else '')
            fb_s = (_decode_bits(tr['FITBITS'],  LP_FITBITS)
                    if 'FITBITS'  in _tcols else '')
            html_lines.append(_td(
                _ned_link(_gname) + _sgaid_s,
                ra_s, dec_s, typ, sersic, shape_r,
                _fmt_tractor_mag(tr, 'G'),
                _fmt_tractor_mag(tr, 'R'),
                _fmt_tractor_mag(tr, 'Z'),
                mb_s, fb_s,
            ))
        html_lines.append("    </table>")
        html_lines.append(
            "    <p class='muted'><code>MASKBITS</code>/<code>FITBITS</code> are Legacy Survey Tractor flags, "
            f"defined in the Legacy Survey's own <a href='{LEGACYPIPE_BITMASKS_URL}' target='_blank'>bitmasks "
            "reference</a>.</p>"
        )

    # ---- Section D: Photometry — full width ---------------------------------
    if len(fullgroup_data) > 0:
        if _has('COG_MTOT_G'):
            n_ap = sum(1 for ap in range(5) if _has(f'FLUX_AP{ap:02d}_G'))
            n_meas = 1 + n_ap  # CoG row + aperture rows
            html_lines.append("    <h2 id='sec-photometry'>Photometry (AB mag)</h2>")
            html_lines.append("    <table>")
            html_lines.append(_th(
                ('Galaxy', 1, 2), ('Measurement', 1, 2),
                ('Optical', 4), ('IR', 2), ('UV', 2),
            ))
            html_lines.append(_th(*[b.lower() for b in phot_bands]))
            row_idx = 0
            for row in fullgroup_data:
                sma_moment = _sf(_get(row, 'SMA_MOMENT'), zero_missing=True)
                for imeas in range(n_meas):
                    row_cls = 'gal-even' if row_idx % 2 == 0 else 'gal-odd'
                    if imeas == 0:
                        meas_label = 'COG (total)'
                        band_vals  = [_fmt_mag_cog(row, b) for b in phot_bands]
                        _gname_p = str(_get(row, 'GALAXY', '') or row['OBJNAME']).strip()
                        _sgaid_p = f'  [{int(row["SGAID"])}]' if _has('SGAID') else ''
                        name_cell  = f"<td class='gal-group' rowspan='{n_meas}'>{_ned_link(_gname_p) + _sgaid_p}</td>"
                    else:
                        ap = n_ap - imeas  # AP04 first (largest → closest to CoG), AP00 last
                        mult = f'{APERTURES[ap]:g}×'
                        if _has(f'SMA_AP{ap:02d}') and sma_moment:
                            sma_ap = _sf(_get(row, f'SMA_AP{ap:02d}'), zero_missing=False)
                            sma_s  = f' = {sma_ap:.1f}"' if sma_ap else ''
                            meas_label = f'AP{ap:02d} ({mult}{sma_s})'
                        else:
                            meas_label = f'AP{ap:02d} ({mult})'
                        band_vals = [_fmt_mag_ap(row, ap, b) for b in phot_bands]
                        name_cell = ''
                    html_lines.append(f"        <tr class='{row_cls}'>{name_cell}<td>{meas_label}</td>" +
                                      ''.join(f'<td>{v}</td>' for v in band_vals) + '</tr>')
                    row_idx += 1
            html_lines.append("    </table>")

    jname_to_galaxy = {}
    jname_to_sgaid = {}
    if len(fullgroup_data) > 0 and 'SGANAME' in fullgroup_data.colnames:
        for row in fullgroup_data:
            jname = str(row['SGANAME']).strip().replace('SGA2025 ', '')
            jname_to_galaxy[jname] = str(_get(row, 'GALAXY', '') or row['OBJNAME']).strip()
            if _has('SGAID'):
                jname_to_sgaid[jname] = int(row['SGAID'])
    html_lines.extend([
        "",
        "    <div class='section'>",
        "        <div class='galaxy-row'>",
    ])
    for title in per_galaxy_titles:
        html_lines.append("            <div><h2>{}</h2></div>".format(title))
    html_lines.append("        </div>")
    for igal, galaxy_name in enumerate(galaxy_names):
        display_name = jname_to_galaxy.get(galaxy_name, galaxy_name)
        _sgaid_h = jname_to_sgaid.get(galaxy_name)
        _sgaid_suffix = f'  (SGAID {_sgaid_h})' if _sgaid_h else ''
        sec_id = "id='sec-figures' " if igal == 0 else ''
        html_lines.append(f"        <h2 {sec_id}style='margin-bottom: 4px;'>{display_name}{_sgaid_suffix}</h2>")
        html_lines.append("        <div class='galaxy-row'>")
        for img_type in per_galaxy_types:
            filename = "SGA2025_{}-{}.png".format(galaxy_name, img_type)
            filepath = group_dir / filename
            if filepath.exists():
                html_lines.append("            <a href='{}'><img src='{}' alt='{}'></a>".format(filename, filename, filename))
            else:
                log.warning("Missing QA file: {}".format(filepath))
                html_lines.append("            <div style='border:1px solid #ddd; color:#888; font-size:12px; padding:10px; text-align:center; flex:0 0 32%;'>Missing:<br>{}</div>".format(filename))
        html_lines.append("        </div>")
    html_lines.append("    </div>")
    html_lines.extend([
        "",
        "</body>",
        "</html>",
    ])
    with open(output_file, 'w') as f:
        f.write('\n'.join(html_lines))
    log.info("Generated: {}".format(output_file))
    return True

def generate_group_html_wrapper(args):
    """Unpack a single argument tuple and call :func:`generate_group_html`
    for one group -- the top-level callable required by
    ``multiprocessing.Pool.map`` (which cannot pickle/pass keyword
    arguments or bound closures).

    Parameters
    ----------
    args : :class:`tuple`
        9-tuple ``(idx, group_name, sample, fullsample, fulltractor,
        htmldir, region, all_groups, clobber)`` as built by
        :func:`make_html`'s multiprocessing branch: `idx` is
        `group_name`'s position in `all_groups` (used to look up the
        previous/next group for HTML nav links), `sample` is the full
        primary-galaxy table, `fullsample`/`fulltractor` are passed
        straight through, and `htmldir`/`region`/`clobber` configure
        the output location and overwrite behavior.

    Returns
    -------
    Return value of :func:`generate_group_html` for this group.

    Notes
    -----
    Selects this group's rows via ``sample[sample['GROUP_NAME'] ==
    group_name]``, an O(N) boolean-mask scan over the *entire* sample
    table, for every one of the ``mp`` worker calls -- unlike
    :func:`make_html`'s ``mp == 1`` code path, which pre-builds an
    O(1) lookup index once via :func:`_build_group_index`. For large
    regions (dr11-south has ~150k groups) this makes the
    multiprocessing path do O(N x G) work in total where the serial
    path does O(N + G); it is also handed a copy of the full `sample`
    table via the pickled `args` tuple for every one of the G calls.

    """
    idx, group_name, sample, fullsample, fulltractor, htmldir, region, all_groups, clobber = args
    group_data = sample[sample['GROUP_NAME'] == group_name]
    prev_group = all_groups[idx - 1] if idx > 0 else None
    next_group = all_groups[idx + 1] if idx < len(all_groups) - 1 else None
    return generate_group_html(group_data, fullsample, htmldir, region, prev_group, next_group, clobber, fulltractor=fulltractor)

def _build_index_html(region, count, sample_bits, ellipsemode_bits, ellipsebit_bits):
    """Build the complete, self-contained ``index-{region}.html`` document:
    a client-side-filterable/sortable/paginated gallery table of all
    groups in `region`, driven entirely by JavaScript against the
    companion ``groups-{region}.json`` payload written by
    :func:`generate_index`.

    Parameters
    ----------
    region : :class:`str`
        Region name (e.g. ``'dr11-south'``), used in the page title,
        the JS ``REGION`` variable, and to fetch
        ``'groups-{region}.json'`` (a path relative to the HTML file's
        own location).
    count : :class:`int`
        Total number of groups in `region`, shown in the page
        subtitle.
    sample_bits : :class:`dict`
        Mapping of ``SAMPLE`` bit name to integer value (see
        :data:`SGA.SGA.SAMPLE`), rendered as one row of toggle
        buttons and embedded as a JS lookup object.
    ellipsemode_bits : :class:`dict`
        Mapping of ``ELLIPSEMODE`` bit name to integer value (see
        :data:`SGA.ellipse.ELLIPSEMODE`), rendered the same way.
    ellipsebit_bits : :class:`dict`
        Mapping of ``ELLIPSEBIT`` bit name to integer value (see
        :data:`SGA.ellipse.ELLIPSEBIT`), rendered as two rows of
        toggle buttons (split roughly in half) plus embedded JS
        lookup object.

    Returns
    -------
    :class:`str`
        The full HTML document text; the caller (:func:`generate_index`)
        is responsible for writing it to disk.

    Notes
    -----
    The client-side ``getSkyUrl`` JS helper picks the Legacy Survey
    viewer layer with ``(REGION === 'dr11-south') ? 'ls-dr11-south' :
    'ls-dr11-north'`` -- any region other than exactly
    ``'dr11-south'`` (including ``'dr9-north'``, ``'dr9-south'``,
    ``'dr10-south'``) falls through to the ``'ls-dr11-north'`` layer,
    which is only correct for ``'dr11-north'``. All literal JS/CSS
    braces in the returned template are doubled (``{{``/``}}``) to
    survive the trailing ``.format(...)`` call; only the eight named
    placeholders listed above are real substitutions.

    """
    # Pre-compute fragments that contain { } so they don't need double-bracing inside the
    # main format string.
    def _btn_row(bits, css_cls, toggle_fn, descs=None):
        """Render one row of toggle ``<button>`` elements for a bit dictionary.

        Parameters
        ----------
        bits : :class:`dict`
            Mapping of bit name to integer value; one button per entry.
        css_cls : :class:`str`
            CSS class applied to every button (e.g. ``'hbtn-sample'``).
        toggle_fn : :class:`str`
            Name of the client-side JS function invoked (with the bit
            value) on click, e.g. ``'toggleSample'``.
        descs : :class:`dict`, optional
            Mapping of bit name to a human-readable tooltip string
            (shown via the button's ``data-tooltip`` attribute);
            entries not present here get an empty tooltip.

        Returns
        -------
        :class:`str`
            Concatenated HTML for all buttons in `bits`.

        """
        return ''.join(
            '<button class="hbtn {css}" onclick="{fn}({v})" data-bit="{v}" data-tooltip="{tip}">{k}</button>'.format(
                css=css_cls, fn=toggle_fn, k=k, v=v,
                tip=(descs or {}).get(k, ''))
            for k, v in bits.items()
        )
    def _bits_js(bits):
        """Render a bit-name dictionary as a JavaScript object literal.

        Parameters
        ----------
        bits : :class:`dict`
            Mapping of bit name to integer value.

        Returns
        -------
        :class:`str`
            JS object-literal source, e.g. ``"{'LVD': 1, 'MCLOUDS': 2}"``.

        """
        return '{' + ', '.join("'{}': {}".format(k, v) for k, v in bits.items()) + '}'

    _sample_descs = {
        'LVD':      'Local Volume Database dwarf',
        'MCLOUDS':  'In the Magellanic Clouds',
        'GCLPNE':   'In a globular cluster or planetary nebula mask',
        'NEARSTAR': 'Near a bright star',
        'INSTAR':   'Inside a bright star',
        'OVERLAP':  'Initial ellipse overlaps another SGA ellipse',
    }
    _emode_descs = {
        'FIXGEO':      'Fix ellipse geometry to initial values',
        'RESOLVED':    'Milky Way satellite; no Tractor fitting (implies FIXGEO)',
        'FORCEPSF':    'Force PSF detection within the SGA mask',
        'FORCEGAIA':   'Force Gaia source detection',
        'LESSMASKING': 'Subtract but do not threshold-mask Gaia stars',
        'MOREMASKING': 'Threshold-mask extended sources within the SGA ellipse',
        'MOMENTPOS':   'Use light-weighted (not Tractor) center',
        'TRACTORGEO':  'Use Tractor (not light-weighted) geometry',
        'NORADWEIGHT': 'Moment geometry without radial weighting',
    }
    _ebit_descs = {
        'NOTRACTOR':        'No corresponding Tractor source',
        'TRACTORPSF':       'Tractor fit this source as a PSF',
        'FIXGEO':           'Fixed ellipse geometry',
        'BLENDED':          'Center within the ellipse of another SGA source',
        'LARGESHIFT':       'Large positional shift from initial ellipse',
        'LARGESHIFT_TRACTOR': 'Large shift between Tractor and final ellipse',
        'MAJORGAL':         'Nearby bright galaxy subtracted',
        'OVERLAP':          'Fitted ellipse overlaps another SGA ellipse',
        'SATELLITE':        'Satellite of a larger galaxy',
        'MOMENTPOS':        'Light-weighted center adopted',
        'TRACTORGEO':       'Tractor geometry adopted',
        'NORADWEIGHT':      'Moment geometry without radial weighting',
        'LESSMASKING':      'Gaia stars subtracted but not threshold-masked',
        'MOREMASKING':      'Extended sources threshold-masked within the SGA ellipse',
        'FAILGEO':          'Ellipse geometry fit failed; reverted to initial',
        'SKIPTRACTOR':      'Tractor fitting skipped for this group',
    }

    hbtns       = _btn_row(sample_bits,       'hbtn-sample', 'toggleSample', _sample_descs)
    emode_hbtns = _btn_row(ellipsemode_bits,  'hbtn-emode',  'toggleEmode',  _emode_descs)
    _ebit_items     = list(ellipsebit_bits.items())
    _ebit_mid       = (len(_ebit_items) + 1) // 2
    ebit_hbtns_row1 = _btn_row(dict(_ebit_items[:_ebit_mid]), 'hbtn-ebit', 'toggleEbit', _ebit_descs)
    ebit_hbtns_row2 = _btn_row(dict(_ebit_items[_ebit_mid:]), 'hbtn-ebit', 'toggleEbit', _ebit_descs)
    ebit_section = (
        '    <div class="hrow">\n'
        '      <span class="hotlabel">ELLIPSEBIT</span>\n      '
        + ebit_hbtns_row1 + '\n'
        '    </div>\n'
        '    <div class="hrow">\n'
        '      <span class="hotlabel" style="visibility:hidden">ELLIPSEBIT</span>\n      '
        + ebit_hbtns_row2 + '\n'
        '      <button class="hbtn" id="hbtn-ebit-none" onclick="toggleNoneEbit()" style="margin-left:6px;">None</button>\n'
        '      <button class="hbtn" onclick="clearEbit()" style="margin-left:4px;">All</button>\n'
        '    </div>'
    )
    sample_bits_js      = _bits_js(sample_bits)
    ellipsemode_bits_js = _bits_js(ellipsemode_bits)
    ellipsebit_bits_js  = _bits_js(ellipsebit_bits)

    # Double-brace all literal JS braces so .format() doesn't mis-interpret them.
    return """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>SGA-2025 Index &mdash; {region}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 20px; color: #333; }}
    h1   {{ margin-bottom: 4px; }}
    .subtitle {{ color: #666; font-size: 14px; margin-bottom: 18px; }}
    .filter-panel {{ display: flex; gap: 40px; background: #f8f8f8;
                     padding: 18px 20px; border: 1px solid #ddd;
                     border-radius: 4px; margin-bottom: 14px; }}
    .filter-col   {{ flex: 1; }}
    .filter-col h4 {{ margin: 0 0 12px; font-size: 14px; color: #444; }}
    .filter-row   {{ display: flex; align-items: center; margin-bottom: 7px;
                     font-size: 13px; }}
    .filter-row label {{ width: 175px; flex-shrink: 0; }}
    input[type=text]   {{ padding: 4px 6px; font-size: 13px;
                          border: 1px solid #ccc; border-radius: 3px; width: 220px; }}
    input[type=number] {{ padding: 4px 6px; font-size: 13px;
                          border: 1px solid #ccc; border-radius: 3px; width: 90px; }}
    .sep {{ padding: 0 5px; color: #999; }}
    .actions {{ margin: 10px 0 10px; }}
    .btn       {{ padding: 7px 18px; background: #0066cc; color: #fff;
                  border: none; border-radius: 3px; cursor: pointer; font-size: 13px; }}
    .btn:hover {{ background: #0052a3; }}
    .btn-clear       {{ background: #777; margin-left: 8px; }}
    .btn-clear:hover {{ background: #555; }}
    .hotbtns  {{ margin: 0 0 12px; }}
    .hrow     {{ display: flex; flex-wrap: wrap; align-items: center; margin-bottom: 5px; }}
    .hotlabel {{ font-size: 13px; font-weight: bold; margin-right: 8px; color: #444;
                 min-width: 110px; flex-shrink: 0; }}
    .hbtn {{ position: relative; padding: 5px 11px; font-size: 12px; background: #e8e8e8;
             border: 1px solid #bbb; border-radius: 3px; cursor: pointer;
             margin-right: 4px; }}
    .hbtn:hover {{ background: #d0d0d0; }}
    .hbtn.active {{ background: #0066cc; color: #fff; border-color: #0044aa; }}
    .hbtn[data-tooltip]:hover::after {{
        content: attr(data-tooltip);
        position: absolute;
        bottom: calc(100% + 5px);
        left: 50%;
        transform: translateX(-50%);
        background: #333;
        color: #fff;
        padding: 4px 8px;
        border-radius: 3px;
        font-size: 11px;
        white-space: nowrap;
        pointer-events: none;
        z-index: 1000;
    }}
    .summary {{ color: #555; font-size: 13px; margin-bottom: 8px; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th {{ background: #f0f0f0; padding: 8px 10px; border: 1px solid #ddd;
          text-align: center; cursor: pointer; white-space: nowrap; }}
    th:hover        {{ background: #e4e4e4; }}
    th.asc::after   {{ content: " ▲"; font-size: 10px; }}
    th.desc::after  {{ content: " ▼"; font-size: 10px; }}
    td {{ padding: 7px 10px; border: 1px solid #ddd; vertical-align: middle; text-align: center; }}
    td.left  {{ text-align: left; }}
    td.thumb {{ padding: 3px; width: 86px; text-align: center; }}
    td.thumb img {{ width: 80px; height: 80px; object-fit: cover; display: block; margin: auto; }}
    a       {{ color: #0066cc; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    .pager {{ margin: 12px 0; text-align: center; font-size: 13px; }}
    .pager a       {{ margin: 0 4px; cursor: pointer; color: #0066cc; }}
    .pager .cur    {{ margin: 0 8px; color: #333; }}
    .no-results    {{ color: #888; padding: 20px; text-align: center; font-style: italic; }}
  </style>
</head>
<body>
  <h1>SGA-2025 Gallery &mdash; {region}</h1>
  <p class="subtitle">Total: {count:,} groups &nbsp;&mdash;&nbsp; <a href="https://sga.readthedocs.io/en/latest/sga2025.html" target="_blank">SGA-2025 Documentation</a></p>

  <div class="filter-panel">
    <div class="filter-col">
      <h4>Simple Filter</h4>
      <div class="filter-row">
        <label>Name</label>
        <input type="text" id="f-name" placeholder="galaxy, alt name, or group name">
      </div>
      <div class="filter-row">
        <label>D(26) (arcmin)</label>
        <input type="number" id="f-d26-min" placeholder="min" step="0.1" min="0">
        <span class="sep">to</span>
        <input type="number" id="f-d26-max" placeholder="max" step="0.1" min="0">
      </div>
      <div class="filter-row">
        <label>Group Diameter (arcmin)</label>
        <input type="number" id="f-diam-min" placeholder="min" step="0.1" min="0">
        <span class="sep">to</span>
        <input type="number" id="f-diam-max" placeholder="max" step="0.1" min="0">
      </div>
      <div class="filter-row">
        <label>Multiplicity</label>
        <input type="number" id="f-mult-min" placeholder="min" step="1" min="1">
        <span class="sep">to</span>
        <input type="number" id="f-mult-max" placeholder="max" step="1" min="1">
      </div>
      <div class="filter-row">
        <label>Redshift</label>
        <input type="number" id="f-z-min" placeholder="min" step="0.001" min="0">
        <span class="sep">to</span>
        <input type="number" id="f-z-max" placeholder="max" step="0.001" min="0">
      </div>
      <div class="filter-row">
        <label>Distance (Mpc)</label>
        <input type="number" id="f-dist-min" placeholder="min" step="1" min="0">
        <span class="sep">to</span>
        <input type="number" id="f-dist-max" placeholder="max" step="1" min="0">
      </div>
    </div>
    <div class="filter-col">
      <h4>Cone Search</h4>
      <div class="filter-row">
        <label>RA (deg)</label>
        <input type="number" id="f-cone-ra" placeholder="degrees" step="0.0001" min="0" max="360">
      </div>
      <div class="filter-row">
        <label>Dec (deg)</label>
        <input type="number" id="f-cone-dec" placeholder="degrees" step="0.0001" min="-90" max="90">
      </div>
      <div class="filter-row">
        <label>Radius (arcmin)</label>
        <input type="number" id="f-cone-rad" placeholder="arcmin" step="0.1" min="0">
      </div>
    </div>
  </div>

  <div class="actions">
    <button class="btn" onclick="applyFilters()">Filter</button>
    <button class="btn btn-clear" onclick="clearFilters()">Clear</button>
  </div>

  <div class="hotbtns">
    <div class="hrow">
      <span class="hotlabel">SAMPLE</span>
      {hbtns}
      <button class="hbtn" id="hbtn-none" onclick="toggleNone()" style="margin-left:6px;">None</button>
      <button class="hbtn" onclick="clearSample()" style="margin-left:4px;">All</button>
    </div>
    <div class="hrow">
      <span class="hotlabel">ELLIPSEMODE</span>
      {emode_hbtns}
      <button class="hbtn" id="hbtn-emode-none" onclick="toggleNoneEmode()" style="margin-left:6px;">None</button>
      <button class="hbtn" onclick="clearEmode()" style="margin-left:4px;">All</button>
    </div>
{ebit_section}
  </div>

  <div class="summary" id="summary">Loading data&hellip;</div>

  <table>
    <thead><tr>
      <th>Preview</th>
      <th id="th-objname" onclick="sortBy('objname')">Primary Galaxy</th>
      <th id="th-sgaid"   onclick="sortBy('sgaid')">SGA ID</th>
      <th id="th-d26"     onclick="sortBy('d26')">D(26) (arcmin)</th>
      <th id="th-z"       onclick="sortBy('z')">Redshift</th>
      <th id="th-dist"    onclick="sortBy('dist')">Distance (Mpc)</th>
      <th id="th-name"    onclick="sortBy('name')">Group Name</th>
      <th id="th-ra"      onclick="sortBy('ra')">Group RA (deg)</th>
      <th id="th-dec"     onclick="sortBy('dec')">Group Dec (deg)</th>
      <th id="th-diam"    onclick="sortBy('diam')">Group Diameter (arcmin)</th>
      <th id="th-mult"    onclick="sortBy('mult')">N</th>
      <th>Viewer</th>
    </tr></thead>
    <tbody id="results-body">
      <tr><td colspan="12" class="no-results">Loading&hellip;</td></tr>
    </tbody>
  </table>
  <div class="pager" id="pager"></div>

<script>
var DATA           = null;
var currentResults = [];
var currentPage    = 0;
var PAGE_SIZE      = 50;
var sortCol        = 'ra';
var sortAsc        = true;
var activeSampleBits = 0;
var showNone         = false;
var activeEmodeBits  = 0;
var showNoneEmode    = false;
var activeEbitBits   = 0;
var showNoneEbit     = false;
var REGION              = '{region}';
var SAMPLE_BITS         = {sample_bits_js};
var ELLIPSEMODE_BITS    = {ellipsemode_bits_js};
var ELLIPSEBIT_BITS     = {ellipsebit_bits_js};

function val(id)    {{ return document.getElementById(id).value.trim(); }}
function numVal(id) {{ var v = parseFloat(val(id)); return isNaN(v) ? null : v; }}

function angDistArcmin(ra1, dec1, ra2, dec2) {{
    var R   = Math.PI / 180;
    var d1  = dec1 * R, d2 = dec2 * R;
    var dra = (ra1 - ra2) * R / 2, dde = (dec1 - dec2) * R / 2;
    var a   = Math.sin(dde)*Math.sin(dde) +
              Math.cos(d1)*Math.cos(d2)*Math.sin(dra)*Math.sin(dra);
    return 2 * Math.asin(Math.sqrt(a)) / R * 60;
}}

// Range filter for nullable fields: exclude row if value is null when a bound is set.
function inRange(v, lo, hi) {{
    if (lo !== null && (v === null || v < lo)) return false;
    if (hi !== null && (v === null || v > hi)) return false;
    return true;
}}

function applyFilters() {{
    if (!DATA) return;
    var nameQ   = val('f-name').toUpperCase();
    var d26Min  = numVal('f-d26-min'),  d26Max  = numVal('f-d26-max');
    var dMin    = numVal('f-diam-min'), dMax    = numVal('f-diam-max');
    var multMin = numVal('f-mult-min'), multMax = numVal('f-mult-max');
    var zMin    = numVal('f-z-min'),    zMax    = numVal('f-z-max');
    var distMin = numVal('f-dist-min'), distMax = numVal('f-dist-max');
    var cRa     = numVal('f-cone-ra'),  cDec    = numVal('f-cone-dec'),
        cRad    = numVal('f-cone-rad');
    var useCone = cRa !== null && cDec !== null && cRad !== null;
    var results = [];
    var n = DATA.names.length;
    for (var i = 0; i < n; i++) {{
        if (nameQ) {{
            if (DATA.names[i].toUpperCase().indexOf(nameQ)     === -1 &&
                DATA.objnames[i].toUpperCase().indexOf(nameQ)  === -1 &&
                DATA.altnames[i].toUpperCase().indexOf(nameQ)  === -1) continue;
        }}
        if (showNone) {{ if (DATA.sample[i] !== 0) continue; }}
        else if (activeSampleBits !== 0 && (DATA.sample[i] & activeSampleBits) === 0) continue;
        if (showNoneEmode) {{ if (DATA.emode[i] !== 0) continue; }}
        else if (activeEmodeBits !== 0 && (DATA.emode[i] & activeEmodeBits) === 0) continue;
        if (showNoneEbit) {{ if (DATA.ebit[i] !== 0) continue; }}
        else if (activeEbitBits  !== 0 && (DATA.ebit[i]  & activeEbitBits)  === 0) continue;
        var ra = DATA.ra[i], dec = DATA.dec[i];
        if (!inRange(DATA.d26[i],  d26Min,  d26Max))  continue;
        if (!inRange(DATA.diam[i], dMin,    dMax))     continue;
        if (!inRange(DATA.mult[i], multMin, multMax))  continue;
        if (!inRange(DATA.z[i],    zMin,    zMax))     continue;
        if (!inRange(DATA.dist[i], distMin, distMax))  continue;
        if (useCone && angDistArcmin(ra, dec, cRa, cDec) > cRad) continue;
        results.push(i);
    }}
    currentResults = results;
    currentPage    = 0;
    sortResults();
    renderPage();
}}

function sortResults() {{
    var d = DATA, col = sortCol, asc = sortAsc;
    var key;
    if      (col === 'objname') key = function(i) {{ return d.objnames[i]; }};
    else if (col === 'sgaid')   key = function(i) {{ return d.sgaids[i]; }};
    else if (col === 'd26')     key = function(i) {{ return d.d26[i]; }};
    else if (col === 'z')       key = function(i) {{ return d.z[i]; }};
    else if (col === 'dist')    key = function(i) {{ return d.dist[i]; }};
    else if (col === 'name')    key = function(i) {{ return d.names[i]; }};
    else if (col === 'ra')      key = function(i) {{ return d.ra[i]; }};
    else if (col === 'dec')     key = function(i) {{ return d.dec[i]; }};
    else if (col === 'diam')    key = function(i) {{ return d.diam[i]; }};
    else if (col === 'mult')    key = function(i) {{ return d.mult[i]; }};
    else return;
    currentResults.sort(function(a, b) {{
        var av = key(a), bv = key(b);
        // Nulls always sort last regardless of direction
        if (av === null && bv === null) return 0;
        if (av === null) return 1;
        if (bv === null) return -1;
        if (av < bv) return asc ? -1 : 1;
        if (av > bv) return asc ?  1 : -1;
        return 0;
    }});
}}

function sortBy(col) {{
    sortAsc = (sortCol === col) ? !sortAsc : true;
    sortCol = col;
    var cols = ['objname','sgaid','d26','z','dist','name','ra','dec','diam','mult'];
    cols.forEach(function(c) {{
        var th = document.getElementById('th-' + c);
        if (th) th.className = (c === col) ? (sortAsc ? 'asc' : 'desc') : '';
    }});
    sortResults();
    renderPage();
}}

function toggleSample(bit) {{
    showNone = false;
    activeSampleBits ^= bit;
    renderSampleBtns();
    applyFilters();
}}

function toggleNone() {{
    showNone = !showNone;
    if (showNone) activeSampleBits = 0;
    renderSampleBtns();
    applyFilters();
}}

function clearSample() {{
    activeSampleBits = 0;
    showNone = false;
    renderSampleBtns();
    applyFilters();
}}

function renderSampleBtns() {{
    document.querySelectorAll('.hbtn-sample[data-bit]').forEach(function(btn) {{
        var bit = parseInt(btn.getAttribute('data-bit'), 10);
        btn.className = 'hbtn hbtn-sample' + ((activeSampleBits & bit) ? ' active' : '');
    }});
    var nb = document.getElementById('hbtn-none');
    if (nb) nb.className = 'hbtn' + (showNone ? ' active' : '');
}}

function toggleEmode(bit) {{
    showNoneEmode = false; activeEmodeBits ^= bit; renderEmodeBtns(); applyFilters();
}}
function toggleNoneEmode() {{
    showNoneEmode = !showNoneEmode;
    if (showNoneEmode) activeEmodeBits = 0;
    renderEmodeBtns(); applyFilters();
}}
function clearEmode() {{
    activeEmodeBits = 0; showNoneEmode = false; renderEmodeBtns(); applyFilters();
}}
function renderEmodeBtns() {{
    document.querySelectorAll('.hbtn-emode[data-bit]').forEach(function(btn) {{
        var bit = parseInt(btn.getAttribute('data-bit'), 10);
        btn.className = 'hbtn hbtn-emode' + ((activeEmodeBits & bit) ? ' active' : '');
    }});
    var nb = document.getElementById('hbtn-emode-none');
    if (nb) nb.className = 'hbtn' + (showNoneEmode ? ' active' : '');
}}

function toggleEbit(bit) {{
    showNoneEbit = false; activeEbitBits ^= bit; renderEbitBtns(); applyFilters();
}}
function toggleNoneEbit() {{
    showNoneEbit = !showNoneEbit;
    if (showNoneEbit) activeEbitBits = 0;
    renderEbitBtns(); applyFilters();
}}
function clearEbit() {{
    activeEbitBits = 0; showNoneEbit = false; renderEbitBtns(); applyFilters();
}}
function renderEbitBtns() {{
    document.querySelectorAll('.hbtn-ebit[data-bit]').forEach(function(btn) {{
        var bit = parseInt(btn.getAttribute('data-bit'), 10);
        btn.className = 'hbtn hbtn-ebit' + ((activeEbitBits & bit) ? ' active' : '');
    }});
    var nb = document.getElementById('hbtn-ebit-none');
    if (nb) nb.className = 'hbtn' + (showNoneEbit ? ' active' : '');
}}

function escHtml(s) {{
    return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}}

function fmt(v, dec) {{
    return v !== null ? v.toFixed(dec) : '&mdash;';
}}

function getSkyUrl(ra, dec, diam) {{
    var layer = (REGION === 'dr11-south') ? 'ls-dr11-south' : 'ls-dr11-north';
    var zoom  = Math.max(11, Math.min(16, Math.round(14 - Math.log10(Math.max(0.3, diam)))));
    return 'https://www.legacysurvey.org/viewer-dev/?ra=' + ra.toFixed(4) +
           '&dec=' + dec.toFixed(4) + '&layer=' + layer +
           '&zoom=' + zoom + '&sga2025-parent';
}}

function buildRow(i) {{
    var name     = DATA.names[i];
    var raslice  = name.substring(0, 3);
    var grpPath  = REGION + '/' + raslice + '/' + name + '/';
    var htmlPath = grpPath + name + '.html';
    var skyUrl   = getSkyUrl(DATA.ra[i], DATA.dec[i], DATA.diam[i]);
    var thumbCell = DATA.has_thumb[i]
        ? '<td class="thumb"><a href="' + htmlPath + '"><img src="' +
          grpPath + 'SGA2025_' + name + '-thumb.jpg" alt=""></a></td>'
        : '<td class="thumb">&mdash;</td>';
    return '<tr>'
        + thumbCell
        + '<td class="left"><a href="' + htmlPath + '">' + escHtml(DATA.objnames[i]) + '</a></td>'
        + '<td>' + DATA.sgaids[i]           + '</td>'
        + '<td>' + fmt(DATA.d26[i],  2)     + '</td>'
        + '<td>' + fmt(DATA.z[i],    4)     + '</td>'
        + '<td>' + fmt(DATA.dist[i], 1)     + '</td>'
        + '<td>' + escHtml(name)            + '</td>'
        + '<td>' + DATA.ra[i].toFixed(4)    + '</td>'
        + '<td>' + DATA.dec[i].toFixed(4)   + '</td>'
        + '<td>' + DATA.diam[i].toFixed(2)  + '</td>'
        + '<td>' + DATA.mult[i]             + '</td>'
        + '<td><a href="' + skyUrl + '" target="_blank">Sky</a></td>'
        + '</tr>';
}}

function renderPage() {{
    var total = currentResults.length;
    var start = currentPage * PAGE_SIZE;
    var end   = Math.min(start + PAGE_SIZE, total);
    var rows  = '';
    if (total === 0) {{
        rows = '<tr><td colspan="12" class="no-results">No results.</td></tr>';
    }} else {{
        for (var k = start; k < end; k++) rows += buildRow(currentResults[k]);
    }}
    document.getElementById('results-body').innerHTML = rows;
    var rangeStr = total === 0 ? '0' : (start + 1) + '–' + end;
    document.getElementById('summary').textContent =
        'Showing ' + rangeStr + ' of ' + total.toLocaleString() + ' groups';
    renderPager(total);
}}

function renderPager(total) {{
    var nPages = Math.ceil(total / PAGE_SIZE), p = currentPage;
    var html = '';
    if (nPages > 1) {{
        if (p > 0)        html += '<a onclick="goPage(0)">&laquo; First</a> ';
        if (p > 0)        html += '<a onclick="goPage(' + (p-1) + ')">Prev</a>';
        html += '<span class="cur">Page ' + (p+1) + ' of ' + nPages + '</span>';
        if (p < nPages-1) html += '<a onclick="goPage(' + (p+1) + ')">Next</a>';
        if (p < nPages-1) html += ' <a onclick="goPage(' + (nPages-1) + ')">Last &raquo;</a>';
    }}
    document.getElementById('pager').innerHTML = html;
}}

function goPage(n) {{
    currentPage = n;
    renderPage();
    window.scrollTo(0, 0);
}}

function clearFilters() {{
    ['f-name','f-d26-min','f-d26-max','f-diam-min','f-diam-max',
     'f-mult-min','f-mult-max',
     'f-z-min','f-z-max','f-dist-min','f-dist-max',
     'f-cone-ra','f-cone-dec','f-cone-rad'
    ].forEach(function(id) {{ document.getElementById(id).value = ''; }});
    clearSample();
}}

document.addEventListener('keydown', function(e) {{
    if (e.key === 'Enter') applyFilters();
}});

fetch('groups-{region}.json')
    .then(function(r) {{ return r.json(); }})
    .then(function(d) {{
        DATA = d;
        document.getElementById('th-ra').className = 'asc';
        applyFilters();
    }})
    .catch(function(e) {{
        document.getElementById('summary').textContent = 'Error loading data: ' + e;
    }});
</script>
</body>
</html>
""".format(region=region, count=count,
           hbtns=hbtns, emode_hbtns=emode_hbtns, ebit_section=ebit_section,
           sample_bits_js=sample_bits_js,
           ellipsemode_bits_js=ellipsemode_bits_js,
           ellipsebit_bits_js=ellipsebit_bits_js)


def _build_group_index(tbl):
    """Build an O(1)-lookup index from ``GROUP_NAME`` to row indices,
    to avoid repeated O(N) boolean-mask scans (``tbl[tbl['col'] ==
    val]``) when looking up the same table by group many times in a
    loop.

    Parameters
    ----------
    tbl : :class:`astropy.table.Table`
        Table with a ``GROUP_NAME`` column (one row per galaxy).

    Returns
    -------
    :class:`dict`
        Mapping of ``str(GROUP_NAME)`` to a :class:`numpy.ndarray` of
        integer row indices into `tbl` for that group.

    """
    from collections import defaultdict
    raw = defaultdict(list)
    for i, gn in enumerate(tbl['GROUP_NAME']):
        raw[str(gn)].append(i)
    return {k: np.array(v, dtype=int) for k, v in raw.items()}


def generate_index(htmldir, region, sample, fullsample=None):
    """Generate the JS-driven gallery index for one region: write
    ``groups-{region}.json`` (one record per group, primary-galaxy
    properties plus OR-combined bitmasks across all group members)
    and ``index-{region}.html`` (built by :func:`_build_index_html`).

    Parameters
    ----------
    htmldir : :class:`pathlib.Path`
        Base HTML output directory; both output files are written
        directly here.
    region : :class:`str`
        Region name (e.g. ``'dr11-south'``), used in both output
        filenames and passed through to :func:`_build_index_html`.
    sample : :class:`astropy.table.Table`
        Primary-galaxy table (one row per group), used for group-level
        columns (``GROUP_RA``, ``GROUP_DEC``, ``GROUP_DIAMETER``,
        ``GROUP_MULT``, and optionally ``D26``/``Z``/``DIST``/``SAMPLE``/
        ``GROUP_PRIMARY``/``ALTNAMES`` when present).
    fullsample : :class:`astropy.table.Table`, optional
        Full per-galaxy table (all group members, not just primaries);
        when it carries ``SAMPLE``/``ELLIPSEMODE``/``ELLIPSEBIT``
        columns (and a ``GROUP_NAME`` column), those bitmasks are
        OR-reduced across every member of each group instead of using
        only the primary galaxy's value. Without `fullsample` (or if
        it lacks these columns), ``ELLIPSEMODE``/``ELLIPSEBIT`` in the
        JSON payload default to 0 for every group, and ``SAMPLE``
        falls back to OR-reducing across `sample` alone (or to 0 if
        `sample` also lacks a ``SAMPLE`` column).

    Returns
    -------
    None

    Notes
    -----
    Groups whose HTML output directory doesn't exist
    (:func:`find_group_directory` returns None) are silently skipped
    from the index. The per-group thumbnail existence check
    (``has_thumb``) does one ``Path.exists()`` filesystem call per
    group. The nested ``_or`` closure is redefined on every iteration
    of the (potentially ~150k-group) loop, a minor but repeated
    overhead.

    """
    import json
    from SGA.SGA import SAMPLE as SAMPLE_BITS

    unique_groups = np.unique(sample['GROUP_NAME'])
    has_d26    = 'D26'    in sample.colnames
    has_z      = 'Z'      in sample.colnames
    has_dist   = 'DIST'   in sample.colnames
    has_sample = 'SAMPLE' in sample.colnames
    has_prim   = 'GROUP_PRIMARY' in sample.colnames
    # fullsample carries one row per galaxy; use it for bitmask ORs if available
    _fs_cols = set(fullsample.colnames) if fullsample is not None else set()
    _fs_has  = lambda col: col in _fs_cols and 'GROUP_NAME' in _fs_cols
    fs_has_sample = _fs_has('SAMPLE')
    fs_has_emode  = _fs_has('ELLIPSEMODE')
    fs_has_ebit   = _fs_has('ELLIPSEBIT')
    need_fs = fs_has_sample or fs_has_emode or fs_has_ebit

    # Pre-index both tables by GROUP_NAME so the loop below runs in O(N + G)
    # instead of O(N × G).  For dr11-south (~350k rows, ~150k groups) the
    # unindexed version required ~52 billion string comparisons per table.
    _sample_idx = _build_group_index(sample)
    _fs_idx     = _build_group_index(fullsample) if need_fs else {}

    def _nullable(val, decimals, zero_missing=True):
        """Round `val` to `decimals` places, or return None if it is
        missing/non-numeric/NaN (or exactly zero, when `zero_missing`).

        Parameters
        ----------
        val : any
            Value to coerce to float and round; typically a table
            cell that may be masked or a fill value.
        decimals : :class:`int`
            Number of decimal places to round to.
        zero_missing : :class:`bool`
            If True, treat an exact 0.0 as missing (returns None) --
            appropriate for columns where 0 is a fill/sentinel value
            rather than a real measurement.

        Returns
        -------
        :class:`float` or None
            Rounded value, or None if `val` could not be interpreted
            as a finite (and, if `zero_missing`, nonzero) float.

        """
        try:
            v = float(val)
            if np.isnan(v) or (zero_missing and v == 0.0):
                return None
            return round(v, decimals)
        except (TypeError, ValueError):
            return None

    has_altnames = 'ALTNAMES' in sample.colnames

    names, sgaids, objnames, altnames = [], [], [], []
    d26s, zs, dists                   = [], [], []
    ras, decs, diams, mults = [], [], [], []
    samples, emodes, ebits  = [], [], []
    has_thumbs              = []

    for group_name in unique_groups:
        group_dir = find_group_directory(htmldir, region, group_name)
        if group_dir is None:
            continue
        _rows = _sample_idx.get(str(group_name))
        if _rows is None:
            continue
        grp = sample[_rows]
        # Primary galaxy: prefer GROUP_PRIMARY flag, else first row
        if has_prim and np.any(grp['GROUP_PRIMARY'] != 0):
            prim = grp[grp['GROUP_PRIMARY'] != 0][0]
        else:
            prim = grp[0]

        names.append(str(group_name))
        sgaids.append(str(prim['SGAID']))
        objnames.append(str(prim['GALAXY']))
        altnames.append(str(prim['ALTNAMES']).strip() if has_altnames else '')
        ras.append(round(float(prim['GROUP_RA']), 6))
        decs.append(round(float(prim['GROUP_DEC']), 6))
        diams.append(round(float(prim['GROUP_DIAMETER']), 4))
        mults.append(int(prim['GROUP_MULT']))
        d26s.append( _nullable(prim['D26'],  2) if has_d26   else None)
        zs.append(   _nullable(prim['Z'],    5) if has_z     else None)
        dists.append(_nullable(prim['DIST'], 1) if has_dist  else None)
        # OR bitmask columns across all group members using fullsample
        _fs_rows = _fs_idx.get(str(group_name)) if need_fs else None
        fs_grp   = fullsample[_fs_rows] if _fs_rows is not None else None

        def _or(col, fs_flag, arr):
            """Append the bitwise-OR of `col` across this group's members
            (from `fullsample`, if available) to `arr`; append 0 otherwise.

            Parameters
            ----------
            col : :class:`str`
                Column name in `fullsample` to OR-reduce (e.g.
                ``'ELLIPSEMODE'``).
            fs_flag : :class:`bool`
                Whether `fullsample` actually has this column (and a
                matching group of rows); if False, appends 0 without
                touching `fullsample`.
            arr : :class:`list`
                List to append the resulting integer to, in place.

            Returns
            -------
            None

            """
            if fs_flag and fs_grp is not None and len(fs_grp) > 0:
                arr.append(int(np.bitwise_or.reduce(fs_grp[col].astype(np.int32))))
            else:
                arr.append(0)

        if fs_has_sample:
            _or('SAMPLE', True, samples)
        elif has_sample:
            samples.append(int(np.bitwise_or.reduce(grp['SAMPLE'].astype(np.int32))))
        else:
            samples.append(0)
        _or('ELLIPSEMODE', fs_has_emode, emodes)
        _or('ELLIPSEBIT',  fs_has_ebit,  ebits)
        thumb = group_dir / 'SGA2025_{}-thumb.jpg'.format(group_name)
        has_thumbs.append(bool(thumb.exists()))

    payload = {
        'region':    region,
        'names':     names,
        'sgaids':    sgaids,
        'objnames':  objnames,
        'altnames':  altnames,
        'd26':       d26s,
        'z':         zs,
        'dist':      dists,
        'ra':        ras,
        'dec':       decs,
        'diam':      diams,
        'mult':      mults,
        'sample':    samples,
        'emode':     emodes,
        'ebit':      ebits,
        'has_thumb': has_thumbs,
    }
    json_file = htmldir / 'groups-{}.json'.format(region)
    with open(json_file, 'w') as f:
        json.dump(payload, f, separators=(',', ':'))
    log.info('Wrote {} ({} groups)'.format(json_file, len(names)))

    _sample_bits  = {k: v for k, v in SAMPLE_BITS.items()
                      if k not in ('OVERLAP',)}
    _emode_bits   = {k: v for k, v in ELLIPSEMODE.items()
                      if k not in ('FORCEGAIA', 'LESSMASKING', 'MOREMASKING',
                                   'TRACTORGEO', 'NORADWEIGHT')}
    _ebit_bits    = {k: v for k, v in ELLIPSEBIT.items()
                      if k not in ('LESSMASKING', 'MOREMASKING')}

    index_file = htmldir / 'index-{}.html'.format(region)
    with open(index_file, 'w') as f:
        f.write(_build_index_html(region, len(names), _sample_bits, _emode_bits, _ebit_bits))
    log.info('Wrote {}'.format(index_file))


def make_html(sample, fullsample, fulltractor=None, htmldir=None, region='dr11-south', mp=1, clobber=False):
    """Generate per-group HTML QA pages and the region-wide gallery
    index for SGA galaxy groups: the top-level entry point for this
    module, called from ``bin/SGA2025-mpi``'s ``--htmlindex`` stage.

    Parameters
    ----------
    sample : :class:`astropy.table.Table`
        Primary-galaxy table (one row per group).
    fullsample : :class:`astropy.table.Table`
        Full per-galaxy table (all group members), passed through to
        :func:`generate_group_html`/:func:`generate_index` for
        per-member details and OR-combined bitmasks.
    fulltractor : :class:`astropy.table.Table`, optional
        Tractor catalog (e.g. from ``read_sga_sample(tractor=True)``);
        when given, adds a Tractor table to each group page.
    htmldir : :class:`str` or :class:`pathlib.Path`, optional
        Base HTML output directory. Defaults to the ``SGA_HTML_DIR``
        environment variable; raises if neither is set.
    region : :class:`str`
        Region name (e.g. ``'dr11-south'``).
    mp : :class:`int`
        Number of worker processes. If 1, groups are processed
        serially using a pre-built :func:`_build_group_index` lookup
        (fast path); if greater than 1, uses
        ``multiprocessing.Pool.map`` over
        :func:`generate_group_html_wrapper` (see its Notes for a
        per-call lookup-cost caveat).
    clobber : :class:`bool`
        If True, regenerate per-group HTML pages that already exist
        (passed through to :func:`generate_group_html`).

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If `htmldir` is not given and the ``SGA_HTML_DIR`` environment
        variable is not set.

    Notes
    -----
    Only groups whose output directory already exists
    (:func:`find_group_directory` returns non-None) are processed;
    groups with no directory are silently dropped from
    `valid_groups` and never get an HTML page or an index entry.
    :func:`generate_index` is always called at the end (unconditionally,
    regardless of ``mp`` or how many groups were actually
    (re)generated), so the JSON/HTML index stays complete even on a
    partial ``--htmlindex`` run.

    """
    if htmldir is None:
        htmldir_env = os.environ.get("SGA_HTML_DIR")
        if not htmldir_env:
            raise ValueError("htmldir not provided and SGA_HTML_DIR not set")
        htmldir = Path(htmldir_env)
    else:
        htmldir = Path(htmldir)
    unique_groups = np.unique(sample['GROUP_NAME'])
    valid_groups = []

    for group_name in unique_groups:
        if find_group_directory(htmldir, region, group_name) is not None:
            valid_groups.append(group_name)
    valid_groups = np.array(valid_groups)
    log.info("Generating HTML for {} groups in region {}".format(len(valid_groups), region))
    if mp == 1:
        _sample_idx = _build_group_index(sample)
        for idx, group_name in enumerate(valid_groups):
            group_data = sample[_sample_idx[str(group_name)]]
            prev_group = valid_groups[idx - 1] if idx > 0 else None
            next_group = valid_groups[idx + 1] if idx < len(valid_groups) - 1 else None
            generate_group_html(group_data, fullsample, htmldir, region, prev_group, next_group, clobber, fulltractor=fulltractor)
    else:
        pool_args = [(idx, gn, sample, fullsample, fulltractor, htmldir, region, valid_groups, clobber)
                     for idx, gn in enumerate(valid_groups)]
        with multiprocessing.Pool(processes=mp) as pool:
            pool.map(generate_group_html_wrapper, pool_args)
    generate_index(htmldir, region, sample, fullsample)
    log.info("HTML generation complete!")
    return
