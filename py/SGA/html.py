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
from SGA.SGA import SAMPLE, RACOLUMN, DECCOLUMN, DIAMCOLUMN, REFIDCOLUMN, APERTURES
from SGA.ellipse import FITMODE, ELLIPSEMODE, ELLIPSEBIT, REF_APERTURES

from SGA.logger import log

import warnings
from astropy.utils.exceptions import AstropyDeprecationWarning

# At the top of ellipse_cog function, add:
warnings.filterwarnings('ignore', category=AstropyDeprecationWarning,
                       message=".*Passing 'theta' positionally.*")


def multiband_montage(data, sample, htmlgalaxydir, barlen=None,
                      barlabel=None, clobber=False, fullsample=None):
    """Diagnostic QA for the output of build_multiband_mask.

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
    """Diagnostic QA for the output of build_multiband_mask.

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
        """RA, DEC → opt pixel coords (0-indexed)."""
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
    """
    spectral energy distribution

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
    """
    curve of growth

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
    """Surface-brightness profiles.

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
    """Make QA plots.

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
    """Decode a bitmask value into list of flag names."""
    flags = []
    for name, bit in bitdict.items():
        if value & bit:
            flags.append(name)
    return flags if flags else ['None']

def get_raslice(ra):
    """Get RA slice from RA in degrees."""
    return "{:03d}".format(int(ra) % 360)

def get_galaxy_names(group_dir):
    """Extract unique galaxy names from filenames in the directory."""
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
    """Generate Legacy Survey sky viewer URL."""
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
    """Find the directory for a given group using raslice from group_name."""
    raslice = group_name[:3]
    group_dir = htmldir / region / raslice / group_name
    if group_dir.exists():
        return group_dir
    return None

def generate_group_html(group_data, fullsample, htmldir, region, prev_group, next_group, clobber=False, fulltractor=None):
    """Generate HTML QA page for a single galaxy group."""
    group_name = group_data['GROUP_NAME'][0]
    group_dir = find_group_directory(htmldir, region, group_name)
    if group_dir is None:
        log.warning("Error: Could not find directory for group {} in region {}".format(group_name, region))
        return False
    output_file = group_dir / "{}.html".format(group_name)
    if output_file.exists() and not clobber:
        log.info("Skipping (exists): {}".format(output_file))
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
        return col in _cols

    def _get(row, col, default=None):
        return row[col] if col in _cols else default

    def _sf(val, zero_missing=True):
        """Return float or None; treats 0 as missing when zero_missing=True."""
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
        v = _sf(_get(row, vcol))
        if v is None:
            return ''
        iv = _sf(_get(row, icol), zero_missing=False) if icol else None
        if iv is not None and iv > 0:
            return f'{v:.5f} ± {1./np.sqrt(iv):.5f}'
        return f'{v:.5f}'

    def _fmt_dist(row, vcol, icol=''):
        v = _sf(_get(row, vcol), zero_missing=True)
        if v is None or v <= 0:
            return ''
        iv = _sf(_get(row, icol), zero_missing=False) if icol else None
        if iv is not None and iv > 0:
            return f'{v:.2f} ± {1./np.sqrt(iv):.2f}'
        return f'{v:.2f}'   # distance known; ivar=0 is expected in some cases

    def _fmt_diam(row):
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
        m = _sf(_get(row, f'COG_MTOT_{band}'), zero_missing=True)
        if m is None or m <= 0:
            return ''
        me = _sf(_get(row, f'COG_MTOT_ERR_{band}'), zero_missing=False)
        if me is not None and me > 0:
            return f'{m:.3f} ± {me:.3f}'
        return f'{m:.3f}'

    def _fmt_mag_ap(row, ap, band):
        f = _sf(_get(row, f'FLUX_AP{ap:02d}_{band}'), zero_missing=True)
        if f is None or f <= 0:
            return ''
        mag = 22.5 - 2.5 * np.log10(f)
        fe = _sf(_get(row, f'FLUX_ERR_AP{ap:02d}_{band}'), zero_missing=False)
        if fe is not None and fe > 0:
            return f'{mag:.3f} ± {2.5*fe/f/np.log(10.):.3f}'
        return f'{mag:.3f}'

    def _th(*cells):
        """Build a <tr> of <th> cells. A cell may be a plain string or a
        (text, colspan, rowspan) tuple; omit trailing 1s."""
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
        return '        <tr>' + ''.join(f'<td>{c}</td>' for c in cells) + '</tr>'

    def _ned_link(name):
        encoded = name.replace('+', '%2B').replace(' ', '+')
        return f"<a href='https://ned.ipac.caltech.edu/byname?objname={encoded}' target='_blank'>{name}</a>"

    def _pgc_link(pgc):
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
        "    <h1>{}</h1>".format(galaxy),
        "    <h3>Group: {} | RA Slice: {}</h3>".format(group_name, raslice),
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

    html_lines.extend([
        "    <p>",
        "        <a href='{}' target='_blank'>Sky Viewer</a> &nbsp;|&nbsp;".format(sky_url),
        "        <a href='{}' target='_blank'>Group Files</a>".format(data_url),
        "    </p>",
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
            has_desi = _has('Z_DESI')
            has_ned  = _has('Z_NED')
            has_lvd  = _has('Z_LVD')
            html_lines.append("    <h2 id='sec-redshift'>Redshift &amp; Distance</h2>")
            html_lines.append("    <table>")
            hdr1 = [('Galaxy', 1, 3), ('Adopted', 4)]
            if has_desi:
                hdr1.append(('DESI', 4))
            if has_ned:
                hdr1.append(('NED', 2))
            if has_lvd:
                hdr1.append(('LVD', 2))
            html_lines.append(_th(*hdr1))
            hdr2 = ['', '', 'Distance', '']
            hdr3 = ['Redshift', 'Ref', '(Mpc)', 'Ref']
            if has_desi:
                hdr2 += ['', '', '', '']
                hdr3 += ['Redshift', 'ZWARN', 'SPECTYPE', 'N Spectra']
            if has_ned:
                hdr2 += ['', 'Distance']
                hdr3 += ['Redshift', '(Mpc)']
            if has_lvd:
                hdr2 += ['', 'Distance']
                hdr3 += ['Redshift', '(Mpc)']
            html_lines.append(_th(*hdr2))
            html_lines.append(_th(*hdr3))
            for row in fullgroup_data:
                z_flag = int(_get(row, 'Z_FLAG', 0) or 0)
                z_flag_s = " <span class='warn'>⚠</span>" if z_flag & 0x01 else ''
                z_ref = str(_get(row, 'Z_REF', '') or '').strip()
                _gname_z = str(_get(row, 'GALAXY', '') or row['OBJNAME']).strip()
                _sgaid_z = f'  [{int(row["SGAID"])}]' if _has('SGAID') else ''
                cells = [
                    _ned_link(_gname_z) + _sgaid_z,
                    _fmt_z(row, 'Z', 'Z_IVAR') + z_flag_s,
                    z_ref,
                    _fmt_dist(row, 'DIST', 'DIST_IVAR'),
                    str(_get(row, 'DIST_REF', '') or '').strip(),
                ]
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
                if has_ned:
                    cells += [
                        _fmt_z(row, 'Z_NED', 'Z_IVAR_NED'),
                        _fmt_dist(row, 'DIST_NED', 'DIST_IVAR_NED'),
                    ]
                if has_lvd:
                    cells += [
                        _fmt_z(row, 'Z_LVD', 'Z_IVAR_LVD'),
                        _fmt_dist(row, 'DIST_LVD', 'DIST_IVAR_LVD'),
                    ]
                html_lines.append(_td(*cells))
            html_lines.append("    </table>")

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
    """Wrapper for multiprocessing."""
    idx, group_name, sample, fullsample, fulltractor, htmldir, region, all_groups, clobber = args
    group_data = sample[sample['GROUP_NAME'] == group_name]
    prev_group = all_groups[idx - 1] if idx > 0 else None
    next_group = all_groups[idx + 1] if idx < len(all_groups) - 1 else None
    return generate_group_html(group_data, fullsample, htmldir, region, prev_group, next_group, clobber, fulltractor=fulltractor)

def _build_index_html(region, count, sample_bits, ellipsemode_bits, ellipsebit_bits):
    """Return the complete index HTML string for one region."""
    # Pre-compute fragments that contain { } so they don't need double-bracing inside the
    # main format string.
    def _btn_row(bits, css_cls, toggle_fn, descs=None):
        return ''.join(
            '<button class="hbtn {css}" onclick="{fn}({v})" data-bit="{v}" data-tooltip="{tip}">{k}</button>'.format(
                css=css_cls, fn=toggle_fn, k=k, v=v,
                tip=(descs or {}).get(k, ''))
            for k, v in bits.items()
        )
    def _bits_js(bits):
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
        <input type="text" id="f-name" placeholder="galaxy or group name">
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
    var zMin    = numVal('f-z-min'),    zMax    = numVal('f-z-max');
    var distMin = numVal('f-dist-min'), distMax = numVal('f-dist-max');
    var cRa     = numVal('f-cone-ra'),  cDec    = numVal('f-cone-dec'),
        cRad    = numVal('f-cone-rad');
    var useCone = cRa !== null && cDec !== null && cRad !== null;
    var results = [];
    var n = DATA.names.length;
    for (var i = 0; i < n; i++) {{
        if (nameQ) {{
            if (DATA.names[i].toUpperCase().indexOf(nameQ)    === -1 &&
                DATA.objnames[i].toUpperCase().indexOf(nameQ) === -1) continue;
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


def generate_index(htmldir, region, sample, fullsample=None):
    """Generate a JS-driven index-{region}.html and companion groups-{region}.json."""
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

    def _nullable(val, decimals, zero_missing=True):
        """Return rounded float or None for missing/NaN/zero values."""
        try:
            v = float(val)
            if np.isnan(v) or (zero_missing and v == 0.0):
                return None
            return round(v, decimals)
        except (TypeError, ValueError):
            return None

    names, sgaids, objnames = [], [], []
    d26s, zs, dists         = [], [], []
    ras, decs, diams, mults = [], [], [], []
    samples, emodes, ebits  = [], [], []
    has_thumbs              = []

    for group_name in unique_groups:
        group_dir = find_group_directory(htmldir, region, group_name)
        if group_dir is None:
            continue
        grp = sample[sample['GROUP_NAME'] == group_name]
        # Primary galaxy: prefer GROUP_PRIMARY flag, else first row
        if has_prim and np.any(grp['GROUP_PRIMARY'] != 0):
            prim = grp[grp['GROUP_PRIMARY'] != 0][0]
        else:
            prim = grp[0]

        names.append(str(group_name))
        sgaids.append(str(prim['SGAID']))
        objnames.append(str(prim['GALAXY']))
        ras.append(round(float(prim['GROUP_RA']), 6))
        decs.append(round(float(prim['GROUP_DEC']), 6))
        diams.append(round(float(prim['GROUP_DIAMETER']), 4))
        mults.append(int(prim['GROUP_MULT']))
        d26s.append( _nullable(prim['D26'],  2) if has_d26   else None)
        zs.append(   _nullable(prim['Z'],    5) if has_z     else None)
        dists.append(_nullable(prim['DIST'], 1) if has_dist  else None)
        # OR bitmask columns across all group members using fullsample
        need_fs = fs_has_sample or fs_has_emode or fs_has_ebit
        fs_grp  = fullsample[fullsample['GROUP_NAME'] == group_name] if need_fs else None

        def _or(col, fs_flag, arr):
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
    """
    Generate HTML QA pages for SGA galaxy groups.

    Parameters:
    -----------
    sample : astropy.table.Table
        Table containing GROUP_PRIMARY galaxies only
    fullsample : astropy.table.Table
        Table containing all galaxies (including non-primary group members)
    fulltractor : astropy.table.Table, optional
        Tractor catalog (from read_sga_sample(tractor=True)); adds a Tractor table
        to each group page when provided
    htmldir : str
        Base HTML directory (default: $SGA_HTML_DIR)
    region : str
        Region name (default: dr11-south)
    mp : int
        Number of processes for multiprocessing (default: 1)
    clobber : bool
        Overwrite existing HTML files (default: False)

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
        for idx, group_name in enumerate(valid_groups):
            group_data = sample[sample['GROUP_NAME'] == group_name]
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
