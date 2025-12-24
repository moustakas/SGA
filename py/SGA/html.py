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
from SGA.SGA import SAMPLE, RACOLUMN, DECCOLUMN, DIAMCOLUMN, REFIDCOLUMN
from SGA.ellipse import FITMODE, ELLIPSEMODE, REF_APERTURES

from SGA.logger import log


def multiband_montage(data, sample, htmlgalaxydir, barlen=None,
                      barlabel=None, clobber=False):
    """Diagnostic QA for the output of build_multiband_mask.

    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec


    if not os.path.isdir(htmlgalaxydir):
        os.makedirs(htmlgalaxydir, exist_ok=True)

    #qafile = os.path.join('ioannis/tmp2/junk.png')
    qafile = os.path.join(htmlgalaxydir, f'qa-{data["galaxy"]}-montage.png')
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

    fig.suptitle(data['galaxy'].replace('_', ' ').replace(' GROUP', ' Group'))
    fig.savefig(qafile)
    plt.close()
    log.info(f'Wrote {qafile}')


def multiband_ellipse_mask(data, ellipse, htmlgalaxydir, unpack_maskbits_function,
                           SGAMASKBITS, barlen=None, barlabel=None, clobber=False):
    """Diagnostic QA for the output of build_multiband_mask.

    """
    import numpy.ma as ma
    import matplotlib.pyplot as plt
    from matplotlib.cm import get_cmap
    import matplotlib.gridspec as gridspec
    from matplotlib.patches import Patch

    from SGA.sky import map_bxby
    from SGA.qa import overplot_ellipse, get_norm, matched_norm


    if not os.path.isdir(htmlgalaxydir):
        os.makedirs(htmlgalaxydir, exist_ok=True)

    qafile = os.path.join(htmlgalaxydir, f'qa-{data["galaxy"]}-ellipsemask.png')
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
    #cmap = plt.cm.get_cmap('cividis').copy()
    #cmap.set_bad((1, 1, 1, 1)) # solid white

    cmap1 = get_cmap('tab20') # or tab20b or tab20c
    colors1 = [cmap1(i) for i in range(20)]

    cmap2 = get_cmap('Dark2')
    colors2 = [cmap2(i) for i in range(5)]

    width = data['width']
    sz = (width, width)

    GEOINITCOLS = ['BX_INIT', 'BY_INIT', 'SMA_INIT', 'BA_INIT', 'PA_INIT']
    GEOFINALCOLS = ['BX', 'BY', 'SMA_MASK', 'BA_MOMENT', 'PA_MOMENT']
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
            [bx, by, sma, ba, pa] = list(obj[GEOINITCOLS].values())
            bx, by = map_bxby(bx, by, from_wcs=opt_wcs, to_wcs=wcs)
            overplot_ellipse(2*sma, ba, pa, bx, by, pixscale=pixscale, ax=xx,
                             color=colors1[iobj], linestyle='-', linewidth=2,
                             draw_majorminor_axes=True, jpeg=False,
                             label=obj[REFIDCOLUMN])

        xx.text(0.03, 0.97, label, transform=xx.transAxes,
                ha='left', va='top', color='white',
                linespacing=1.5, fontsize=8,
                bbox=dict(boxstyle='round', facecolor='k', alpha=0.5))

        if iax == 0:
            xx.legend(loc='lower left', fontsize=8, ncol=2,
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
        #ax[1+iobj, 1].scatter(allgalsrcs.bx, allgalsrcs.by, color='red', marker='s')
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
            [bx, by, sma, ba, pa] = list(obj[GEOINITCOLS].values())
            overplot_ellipse(2*sma, ba, pa, bx, by, pixscale=opt_pixscale,
                             ax=ax[1+iobj, col], color=colors2[0], linestyle='-',
                             linewidth=2, draw_majorminor_axes=True,
                             jpeg=False, label='Initial')

            # final geometry
            [bx, by, sma, ba, pa] = list(obj[GEOFINALCOLS].values())
            overplot_ellipse(2*sma, ba, pa, bx, by, pixscale=opt_pixscale,
                             ax=ax[1+iobj, col], color=colors2[1], linestyle='--',
                             linewidth=2, draw_majorminor_axes=True,
                             jpeg=False, label='Final')
            ax[1+iobj, col].set_xlim(0, width-1)
            ax[1+iobj, col].set_ylim(0, width-1)
            ax[1+iobj, col].margins(0)

        ax[1+iobj, 0].text(0.03, 0.97, f'{obj["OBJNAME"]} ({obj[REFIDCOLUMN]})',
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

    fig.suptitle(data['galaxy'].replace('_', ' ').replace(' GROUP', ' Group'))
    fig.savefig(qafile)
    plt.close()
    log.info(f'Wrote {qafile}')


def ellipse_sed(data, ellipse, htmlgalaxydir, tractor=None, run='south',
                apertures=REF_APERTURES, clobber=False):
    """
    spectral energy distribution

    """
    from copy import deepcopy
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    from matplotlib.cm import get_cmap
    import seaborn as sns

    from SGA.util import filter_effwaves


    colors1 = sns.color_palette('Set1', n_colors=14, desat=0.75)

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
        good = np.where((thisphot['abmag'] > 0) * (thisphot['lower'] == False))[0]
        if len(good) > 0:
            ax.errorbar(bandwave[good]/1e4, thisphot['abmag'][good], yerr=thisphot['abmagerr'][good],
                        marker=marker, markersize=11, markeredgewidth=3, markeredgecolor='k',
                        markerfacecolor=color, elinewidth=3, ecolor=color, capsize=4,
                        label=label, linestyle='none', alpha=alpha)


    # see also Morrisey+05
    for iobj, obj in enumerate(ellipse):

        sganame = obj['SGANAME'].replace(' ', '_')
        qafile = os.path.join(htmlgalaxydir, f'qa-{sganame}-sed.png')
        if os.path.isfile(qafile) and not clobber:
            log.info(f'File {qafile} exists and clobber=False')
            continue


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

        fig.suptitle(f'{data["galaxy"].replace("_", " ").replace(" GROUP", " Group")}: ' + \
                     f'{obj["OBJNAME"]} ({obj["SGANAME"]})') # ({obj[REFIDCOLUMN]})
        fig.tight_layout()
        fig.savefig(qafile, bbox_inches='tight')
        plt.close()
        log.info(f'Wrote {qafile}')



def ellipse_cog(data, ellipse, sbprofiles, htmlgalaxydir, datasets=['opt', 'unwise', 'galex'],
                clobber=False):
    """
    curve of growth

    """
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.cm import get_cmap

    from SGA.SGA import SGA_diameter
    from SGA.ellipse import cog_model
    from SGA.qa import sbprofile_colors


    for iobj, obj in enumerate(ellipse):

        sganame = obj['SGANAME'].replace(' ', '_')
        qafile = os.path.join(htmlgalaxydir, f'qa-{sganame}-cog.png')
        if os.path.isfile(qafile) and not clobber:
            log.info(f'File {qafile} exists and clobber=False')
            continue

        sbcolors = sbprofile_colors()
        cmap2a = get_cmap('Dark2')
        cmap2b = get_cmap('Paired')
        colors2 = [cmap2a(1), cmap2b(3)]

        markers = ['s', 'o', 'v']

        fig, ax = plt.subplots(figsize=(8, 6))

        # one row per dataset
        yminmax = [40, 0]
        xminmax = [0., max(sbprofiles[0][iobj]['SMA'])] # optical
        for idata, dataset in enumerate(datasets):

            sbprofiles_obj = sbprofiles[idata][iobj]
            bands = data[f'{dataset}_bands']

            sma_moment = obj['SMA_MOMENT'] # [arcsec]
            label_moment = r'$R(mom)='+f'{sma_moment:.1f}'+r'$ arcsec'
            if dataset == 'opt':
                sma_sbthresh, _, label_sbthresh, _ = SGA_diameter(Table(obj), radius_arcsec=True)
                sma_sbthresh = sma_sbthresh[0]
                label_sbthresh = r'$'+label_sbthresh[0]+'='+f'{sma_sbthresh:.1f}'+r'$ arcsec'

            for filt in bands:
                I = ((sbprofiles_obj[f'FLUX_{filt.upper()}'].value > 0.) *
                     (sbprofiles_obj[f'FLUX_ERR_{filt.upper()}'].value > 0.))

                if np.any(I):
                    sma = sbprofiles_obj['SMA'][I].value
                    flux = sbprofiles_obj[f'FLUX_{filt.upper()}'][I].value
                    fluxerr = sbprofiles_obj[f'FLUX_ERR_{filt.upper()}'][I].value
                    mag = 22.5 - 2.5 * np.log10(flux)
                    magerr = 2.5 * fluxerr / flux / np.log(10.)

                    mtot = ellipse[f'COG_MTOT_{filt.upper()}'][iobj]
                    mtoterr = ellipse[f'COG_MTOT_ERR_{filt.upper()}'][iobj]
                    if mtot > 0:
                        label = f'{filt}={mtot:.3f}'+r'$\pm$'+f'{mtoterr:.3f} mag'
                    else:
                        label = filt

                    col = sbcolors[filt]
                    ax.errorbar(sma, mag, yerr=magerr, fmt=markers[idata], markersize=5, markeredgewidth=1,
                                markeredgecolor='k', markerfacecolor=col, elinewidth=3,
                                ecolor=col, capsize=4, label=label, alpha=0.7)

                    # best-fitting model
                    dmag = ellipse[f'COG_DMAG_{filt.upper()}'][iobj]
                    lnalpha1 = ellipse[f'COG_LNALPHA1_{filt.upper()}'][iobj]
                    lnalpha2 = ellipse[f'COG_LNALPHA2_{filt.upper()}'][iobj]
                    if mtot > 0:
                        smagrid = np.linspace(0., xminmax[1], 50)
                        mfit = cog_model(smagrid, mtot, dmag, lnalpha1, lnalpha2, r0=sma_moment)
                        ax.plot(smagrid, mfit, color=col, alpha=0.8)

                    # robust limits
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

        ax.set_xlabel('(Semi-major axis (arcsec)')
        ax.set_ylabel('Cumulative Flux (AB mag)')

        if sma_sbthresh > 0.:
            ax.axvline(x=sma_sbthresh, color=colors2[1], lw=2, ls='-', label=label_sbthresh)
        ax.axvline(x=sma_moment, color=colors2[0], lw=2, ls='--', label=label_moment)

        hndls, _ = ax.get_legend_handles_labels()
        if hndls:
            # split into two legends
            hndls_data = [hndl for hndl in hndls if not type(hndl) is matplotlib.lines.Line2D]
            hndls_vline = [hndl for hndl in hndls if type(hndl) is matplotlib.lines.Line2D]
            leg1 = ax.legend(handles=hndls_data, loc='lower right', fontsize=8)
            ax.legend(handles=hndls_vline, loc='upper left', fontsize=8)
            ax.add_artist(leg1)

        fig.suptitle(f'{data["galaxy"].replace("_", " ").replace(" GROUP", " Group")}: ' + \
                     f'{obj["OBJNAME"]} ({obj["SGANAME"]})') # ({obj[REFIDCOLUMN]})
        #fig.suptitle(data['galaxy'].replace('_', ' ').replace(' GROUP', ' Group'))
        fig.tight_layout()
        fig.savefig(qafile, bbox_inches='tight')
        plt.close()
        log.info(f'Wrote {qafile}')



def ellipse_sbprofiles(data, ellipse, sbprofiles, htmlgalaxydir,
                       unpack_maskbits_function, MASKBITS, REFIDCOLUMN,
                       datasets=['opt', 'unwise', 'galex'],
                       linear=False, clobber=False):
    """Surface-brightness profiles.

    """
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    from matplotlib.cm import get_cmap
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
    cmap2a = get_cmap('Dark2')
    cmap2b = get_cmap('Paired')
    colors2 = [cmap2a(1), cmap2b(3), cmap2a(2)]
    #colors2 = [cmap2(i) for i in range(5)]

    for iobj, obj in enumerate(ellipse):

        sganame = obj['SGANAME'].replace(' ', '_')
        qafile = os.path.join(htmlgalaxydir, f'qa-{sganame}-sbprofiles.png')
        if os.path.isfile(qafile) and not clobber:
            log.info(f'File {qafile} exists and clobber=False')
            continue


        fig, ax = plt.subplots(nrow, ncol,
                               figsize=(inches_per_panel * (1+ncol),
                                        inches_per_panel * nrow),
                               gridspec_kw={
                                   'height_ratios': [1., 1., 1.],
                                   'width_ratios': [1., 2.],
                                   #'width_ratios': [1., 2., 2.],
                                   #'wspace': 0
                               })

        # one row per dataset
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

            sma_mask = obj['SMA_MASK'] # [arcsec]
            sma_moment = obj['SMA_MOMENT'] # [arcsec]
            label_mask = r'$R(mask)='+f'{sma_mask:.1f}'+r'$ arcsec'
            label_moment = r'$R(mom)='+f'{sma_moment:.1f}'+r'$ arcsec'
            if idata == 0:
                sma_sbthresh, _, label_sbthresh, _ = SGA_diameter(Table(obj), radius_arcsec=True)
                sma_sbthresh = sma_sbthresh[0]
                label_sbthresh = r'$'+label_sbthresh[0]+'='+f'{sma_sbthresh:.1f}'+r'$ arcsec'
                #r'$R_{'+filt.lower()+r'}('+f'{thresh:.0f}'+')='+f'{val:.1f}'+r'$ arcsec'

            if have_data:
                bx, by = map_bxby(opt_bx, opt_by, from_wcs=opt_wcs, to_wcs=wcs)
                refg = EllipseGeometry(x0=bx, y0=by, eps=ellipse_eps,
                                       pa=ellipse_pa, sma=sma_moment/pixscale) # sma in pixels
                refap = EllipticalAperture((refg.x0, refg.y0), refg.sma,
                                           refg.sma*(1. - refg.eps), refg.pa)
                refap_sma_mask = EllipticalAperture((refg.x0, refg.y0), sma_mask/pixscale,
                                                    sma_mask/pixscale*(1. - refg.eps), refg.pa)

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
                for sma in smas: # sma in pixels
                    if sma == 0.:
                        continue
                    ap = EllipticalAperture((refg.x0, refg.y0), sma,
                                            sma*(1. - refg.eps), refg.pa)
                    ap.plot(color='k', lw=1, ax=xx)
                refap.plot(color=colors2[0], lw=2, ls='--', ax=xx)
                refap_sma_mask.plot(color=colors2[2], lw=2, ls='-', ax=xx)
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
                                            label=filt, color=col, alpha=0.7)

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
                xx.set_xlim(ax[0, 1].get_xlim())

                if idata == ndataset-1:
                    xx.set_xlabel(r'(Semi-major axis / arcsec)$^{1/4}$')
                else:
                    xx.set_xticks([])

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
                    xx_twin.set_ylabel(r'Surface Brightness (mag arcsec$^{-2}$)')

                if sma_sbthresh > 0.:
                    xx.axvline(x=sma_sbthresh**0.25, color=colors2[1], lw=2, ls='-', label=label_sbthresh)
                xx.axvline(x=sma_mask**0.25, color=colors2[2], lw=2, ls='-', label=label_mask)
                xx.axvline(x=sma_moment**0.25, color=colors2[0], lw=2, ls='--', label=label_moment)

                hndls, _ = xx.get_legend_handles_labels()
                if hndls:
                    if sma_sbthresh > 0.:
                        split = -3
                    else:
                        split = -2
                    if idata == 0:
                        # split into two legends
                        leg1 = xx.legend(handles=hndls[:split], loc='upper right', fontsize=8)
                        xx.legend(handles=hndls[split:], loc='lower left', fontsize=8)
                        xx.add_artist(leg1)
                    else:
                        xx.legend(handles=hndls[:split], loc='upper right', fontsize=8)
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
                ax[idata, 1].set_xlim(ax[0, 1].get_xlim())

                if idata == ndataset-1:
                    ax[idata, 1].set_xlabel(r'(Semi-major axis / arcsec)$^{1/4}$')
                else:
                    ax[idata, 1].set_xticks([])


        fig.suptitle(f'{data["galaxy"].replace("_", " ").replace(" GROUP", " Group")}: ' + \
                     f'{obj["OBJNAME"]} ({obj["SGANAME"]})') # ({obj[REFIDCOLUMN]})
        #fig.suptitle(data['galaxy'].replace('_', ' ').replace(' GROUP', ' Group'))
        fig.tight_layout()
        fig.savefig(qafile, bbox_inches='tight')
        plt.close()
        log.info(f'Wrote {qafile}')


def make_plots(galaxy, galaxydir, htmlgalaxydir, REFIDCOLUMN, read_multiband_function,
               unpack_maskbits_function, SGAMASKBITS, APERTURES, run='south', mp=1,
               bands=['g', 'r', 'i', 'z'], pixscale=0.262, galex_pixscale=1.5,
               skip_ellipse=False, unwise_pixscale=2.75, galex=True, unwise=True,
               barlen=None, barlabel=None, verbose=False, clobber=False):
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
        galex=galex, skip_ellipse=skip_ellipse, read_jpg=True)

    multiband_montage(data, sample, htmlgalaxydir, barlen=barlen,
                      barlabel=barlabel, clobber=clobber)

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
        return Table(), Table()

    if len(ellipsefiles) != len(sample):
        msg = f'Mismatching number of ellipse files and objects in sample in {galaxydir}'
        log.critical(msg)
        raise IOError(msg)

    ellipse = []
    for ellipsefile in ellipsefiles:
        # fragile!!
        sganame = os.path.basename(ellipsefile).split('-')[:-2]
        if len(sganame) == 1:
            sganame = sganame[0]
        else:
            sganame = '-'.join(sganame)

        # loop on datasets and join
        for idata, dataset in enumerate(datasets):
            if dataset == 'opt':
                suffix = allbands
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

    # photometry - curve of growth and SED
    ellipse_sed(data, ellipse, htmlgalaxydir, run=run, tractor=samplesrcs,
                apertures=APERTURES, clobber=clobber)

    ellipse_cog(data, ellipse, sbprofiles, htmlgalaxydir,
                datasets=['opt', 'unwise', 'galex'],
                clobber=clobber)

    # surface-brightness profiles
    ellipse_sbprofiles(data, ellipse, sbprofiles, htmlgalaxydir,
                       unpack_maskbits_function, SGAMASKBITS,
                       REFIDCOLUMN, datasets=['opt', 'unwise', 'galex'],
                       linear=False, clobber=clobber)

    # ellipse mask
    multiband_ellipse_mask(data, ellipse, htmlgalaxydir, unpack_maskbits_function,
                           SGAMASKBITS, barlen=barlen, barlabel=barlabel,
                           clobber=clobber)

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
    for onefile in glob(os.path.join(group_dir, "qa-SGA2025_J*")):
        stem = os.path.basename(onefile)
        if stem.startswith("qa-SGA2025_"):
            remainder = stem.replace("qa-SGA2025_", "", 1)
            parts = remainder.rsplit("-", 1)
            if len(parts) >= 2:
                galaxy_name = parts[0]
                galaxy_names.append(galaxy_name)
    return np.unique(galaxy_names).tolist()

def get_sky_viewer_url(ra, dec, diameter, region):
    """Generate Legacy Survey sky viewer URL."""
    if region == 'dr11-south':
        layer = 'ls-dr11-early-v2'
    else:
        layer = 'ls-dr9-north'
    diam_arcmin = diameter
    zoom = max(11, min(16, int(16 - np.log10(max(1.0, diam_arcmin)))))
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

def generate_group_html(group_data, fullsample, htmldir, region, prev_group, next_group, clobber=False):
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
    fullgroup_data = fullsample[np.isin(fullsample['GROUP_NAME'], group_data['GROUP_NAME'])]
    objname = group_data['OBJNAME'][0]
    group_ra = group_data['GROUP_RA'][0]
    group_dec = group_data['GROUP_DEC'][0]
    group_diam = group_data['GROUP_DIAMETER'][0]
    group_mult = group_data['GROUP_MULT'][0]
    raslice = group_name[:3]
    sky_url = get_sky_viewer_url(group_ra, group_dec, group_diam, region)
    group_files = [
        "qa-SGA2025_{}-montage.png".format(group_name),
        "qa-SGA2025_{}-ellipsemask.png".format(group_name),
    ]
    per_galaxy_types = ["sbprofiles", "cog", "sed"]
    per_galaxy_titles = ["Surface Brightness", "Curve of Growth", "Spectral Energy Distribution"]
    galaxy_names = get_galaxy_names(str(group_dir))
    html_lines = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        "    <title>SGA2025: {}</title>".format(objname),
        "    <style>",
        "        body { font-family: Arial, sans-serif; margin: 0; padding: 60px 20px 20px 20px; }",
        "        .navbar { position: fixed; top: 0; left: 0; right: 0; background-color: #333; padding: 10px 20px; z-index: 1000; }",
        "        .navbar a { color: white; text-decoration: none; margin-right: 20px; font-weight: bold; }",
        "        .navbar a:hover { text-decoration: underline; }",
        "        .breadcrumb { color: #666; margin-bottom: 10px; font-size: 14px; }",
        "        .breadcrumb a { color: #0066cc; text-decoration: none; }",
        "        .breadcrumb a:hover { text-decoration: underline; }",
        "        h1 { color: #333; margin-bottom: 5px; }",
        "        h2 { color: #555; margin-top: 5px; font-weight: normal; font-size: 18px; }",
        "        h3 { color: #555; margin-top: 20px; margin-bottom: 10px; }",
        "        table { border-collapse: collapse; margin: 20px 0; }",
        "        th { background-color: #f0f0f0; padding: 8px; border: 1px solid #ddd; text-align: left; }",
        "        td { padding: 8px; border: 1px solid #ddd; }",
        "        .section { margin: 30px 0; }",
        "        .group-images img { display: block; max-width: 100%; margin: 10px 0; }",
        "        .galaxy-row { display: flex; gap: 10px; margin: 10px 0; justify-content: space-between; }",
        "        .galaxy-row a { display: block; flex: 0 0 32%; }",
        "        .galaxy-row img { width: 100%; height: auto; display: block; }",
        "        .galaxy-row div { flex: 1; max-width: 32%; }",
        "        img { border: 1px solid #ddd; }",
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
        "        <a href='../../../index-{}.html#raslice-{}'>{}</a> &gt; ".format(region, raslice, region),
        "        <a href='../../../index-{}.html#raslice-{}'>RA {}</a> &gt; {}".format(region, raslice, raslice, objname),
        "    </div>",
        "    <h1>{}</h1>".format(objname),
        "    <h2>Group: {} | RA Slice: {}</h2>".format(group_name, raslice),
    ]
    html_lines.append("    <h3>Group Properties</h3>")
    html_lines.append("    <table>")
    html_lines.append("        <tr><th>Property</th><th>Value</th></tr>")
    html_lines.append("        <tr><td>Group Name</td><td>{}</td></tr>".format(group_name))
    html_lines.append("        <tr><td>Object Name</td><td>{}</td></tr>".format(objname))
    html_lines.append("        <tr><td>RA (deg)</td><td>{:.6f}</td></tr>".format(group_ra))
    html_lines.append("        <tr><td>Dec (deg)</td><td>{:.6f}</td></tr>".format(group_dec))
    html_lines.append("        <tr><td>Diameter (arcmin)</td><td>{:.2f}</td></tr>".format(group_diam))
    html_lines.append("        <tr><td>Multiplicity</td><td>{}</td></tr>".format(group_mult))
    html_lines.append("        <tr><td>Region</td><td>{}</td></tr>".format(region))
    html_lines.append("    </table>")
    if len(fullgroup_data) > 0:
        html_lines.append("    <h3>Galaxy Properties</h3>")
        html_lines.append("    <table>")
        html_lines.append("        <tr>")
        html_lines.append("            <th>Object Name</th><th>SGA ID</th><th>RA</th><th>Dec</th>")
        html_lines.append("            <th>Diam (arcmin)</th><th>Mag</th><th>b/a</th><th>PA</th><th>Primary</th>")
        html_lines.append("        </tr>")
        for row in fullgroup_data:
            html_lines.append("        <tr>")
            html_lines.append("            <td>{}</td>".format(row['OBJNAME']))
            html_lines.append("            <td>{}</td>".format(row['SGAID']))
            html_lines.append("            <td>{:.6f}</td>".format(row['RA']))
            html_lines.append("            <td>{:.6f}</td>".format(row['DEC']))
            html_lines.append("            <td>{:.2f}</td>".format(row['DIAM']))
            html_lines.append("            <td>{:.2f}</td>".format(row['MAG']))
            html_lines.append("            <td>{:.3f}</td>".format(row['BA']))
            html_lines.append("            <td>{:.1f}</td>".format(row['PA']))
            html_lines.append("            <td>{}</td>".format('Yes' if row['GROUP_PRIMARY'] else 'No'))
            html_lines.append("        </tr>")
        html_lines.append("    </table>")
        html_lines.append("    <h3>Additional Metadata</h3>")
        html_lines.append("    <table>")
        html_lines.append("        <tr>")
        html_lines.append("            <th>Object Name</th><th>PGC</th><th>SAMPLE</th><th>ELLIPSEMODE</th>")
        html_lines.append("            <th>FITMODE</th><th>E(B-V)</th><th>Diam Ref</th>")
        html_lines.append("        </tr>")
        for row in fullgroup_data:
            sample_flags = ', '.join(decode_bitmask(row['SAMPLE'], SAMPLE))
            ellipse_flags = ', '.join(decode_bitmask(row['ELLIPSEMODE'], ELLIPSEMODE))
            fit_flags = ', '.join(decode_bitmask(row['FITMODE'], FITMODE))
            html_lines.append("        <tr>")
            html_lines.append("            <td>{}</td>".format(row['OBJNAME']))
            html_lines.append("            <td>{}</td>".format(row['PGC'] if row['PGC'] > 0 else '-'))
            html_lines.append("            <td>{}</td>".format(sample_flags))
            html_lines.append("            <td>{}</td>".format(ellipse_flags))
            html_lines.append("            <td>{}</td>".format(fit_flags))
            html_lines.append("            <td>{:.3f}</td>".format(row['EBV']))
            html_lines.append("            <td>{}</td>".format(row['DIAM_REF']))
            html_lines.append("        </tr>")
        html_lines.append("    </table>")
    html_lines.extend([
        "",
        "    <div class='section'>",
        "        <div class='group-images'>",
        "            <h3>Multiwavelength Montage</h3>",
    ])
    filepath = group_dir / group_files[0]
    if filepath.exists():
        html_lines.append("            <a href='{}'><img src='{}' alt='{}' style='max-width: 100%; height: auto;'></a>".format(group_files[0], group_files[0], group_files[0]))
    else:
        html_lines.append("            <img src='' alt='{}' style='display:none;'>".format(group_files[0]))
        html_lines.append("            <p style='color: #888;'>Missing: {}</p>".format(group_files[0]))
    html_lines.append("            <h3>Ellipse Masking & Geometry</h3>")
    filepath = group_dir / group_files[1]
    if filepath.exists():
        html_lines.append("            <a href='{}'><img src='{}' alt='{}' style='max-width: 100%; height: auto;'></a>".format(group_files[1], group_files[1], group_files[1]))
    else:
        html_lines.append("            <img src='' alt='{}' style='display:none;'>".format(group_files[1]))
        html_lines.append("            <p style='color: #888;'>Missing: {}</p>".format(group_files[1]))
    html_lines.extend([
        "        </div>",
        "    </div>",
    ])
    html_lines.extend([
        "",
        "    <div class='section'>",
        "        <div class='galaxy-row'>",
    ])
    for title in per_galaxy_titles:
        html_lines.append("            <div><h3>{}</h3></div>".format(title))
    html_lines.append("        </div>")
    for galaxy_name in galaxy_names:
        html_lines.append("        <div class='galaxy-row'>")
        for img_type in per_galaxy_types:
            filename = "qa-SGA2025_{}-{}.png".format(galaxy_name, img_type)
            filepath = group_dir / filename
            if not filepath.exists():
                raise FileNotFoundError("Missing required file: {}".format(filepath))
            html_lines.append("            <a href='{}'><img src='{}' alt='{}'></a>".format(filename, filename, filename))
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
    idx, group_name, sample, fullsample, htmldir, region, all_groups, clobber = args
    group_data = sample[sample['GROUP_NAME'] == group_name]
    prev_group = all_groups[idx - 1] if idx > 0 else None
    next_group = all_groups[idx + 1] if idx < len(all_groups) - 1 else None
    return generate_group_html(group_data, fullsample, htmldir, region, prev_group, next_group, clobber)

def generate_index(htmldir, region, sample):
    """Generate searchable index.html with links to all group pages."""
    unique_groups = np.unique(sample['GROUP_NAME'])
    groups_by_raslice = {}
    group_info = {}
    for group_name in unique_groups:
        raslice = group_name[:3]
        group_dir = find_group_directory(htmldir, region, group_name)
        if group_dir is None:
            continue
        if raslice not in groups_by_raslice:
            groups_by_raslice[raslice] = []
        groups_by_raslice[raslice].append(group_name)
        group_data = sample[sample['GROUP_NAME'] == group_name][0]
        group_info[group_name] = {
            'objname': group_data['OBJNAME'],
            'ra': group_data['GROUP_RA'],
            'dec': group_data['GROUP_DEC'],
            'diam': group_data['GROUP_DIAMETER'],
            'mult': group_data['GROUP_MULT'],
        }
    for raslice in groups_by_raslice:
        groups_by_raslice[raslice].sort()
    html_lines = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        "    <title>SGA2025 Index - {}</title>".format(region),
        "    <style>",
        "        body { font-family: Arial, sans-serif; margin: 20px; }",
        "        h1 { color: #333; }",
        "        .search-container { margin: 20px 0; }",
        "        #searchInput { width: 100%; max-width: 600px; padding: 10px; font-size: 16px; border: 2px solid #ddd; }",
        "        .summary { color: #666; margin: 10px 0; }",
        "        .nav { background-color: #f0f0f0; padding: 15px; margin-bottom: 20px; }",
        "        .nav a { margin-right: 10px; }",
        "        .raslice-section { margin-bottom: 40px; }",
        "        h2 { color: #555; margin-top: 30px; cursor: pointer; }",
        "        h2:hover { color: #0066cc; }",
        "        table { border-collapse: collapse; width: 100%; margin-bottom: 30px; }",
        "        th { background-color: #f0f0f0; padding: 10px; border: 1px solid #ddd; text-align: left; cursor: pointer; }",
        "        th:hover { background-color: #e0e0e0; }",
        "        td { padding: 10px; border: 1px solid #ddd; }",
        "        td.name { width: 30%; }",
        "        td.thumbnail { padding: 0; width: 20%; }",
        "        td.objname { width: 25%; font-weight: bold; }",
        "        td.coords { width: 25%; font-size: 14px; }",
        "        img { border: 1px solid #ccc; }",
        "        a { text-decoration: none; color: #0066cc; }",
        "        a:hover { text-decoration: underline; }",
        "        .hidden { display: none; }",
        "    </style>",
        "</head>",
        "<body>",
        "    <h1>SGA2025 Group Index - {}</h1>".format(region),
        "    <div class='search-container'>",
        "        <input type='text' id='searchInput' placeholder='Search by object name or group name...' onkeyup='filterTable()'>",
        "    </div>",
        "    <div class='summary' id='summary'>Showing {} groups</div>".format(len(unique_groups)),
        "    <div class='nav'>",
        "        <strong>Jump to an RA slice:</strong><br><br>",
    ]
    for raslice in sorted(groups_by_raslice.keys()):
        html_lines.append("        <a href='#raslice-{}' onclick='openSlice(\"{}\"); return true;'>{}</a>".format(raslice, raslice, raslice))
    html_lines.append("    </div>")
    for raslice in sorted(groups_by_raslice.keys()):
        html_lines.append("    <div class='raslice-section'>")
        html_lines.append("    <h2 id='raslice-{}' onclick='toggleSection(this)'>RA Slice {} ({} groups)</h2>".format(
            raslice, raslice, len(groups_by_raslice[raslice])))
        html_lines.append("    <table class='data-table' style='display: none;'>")
        html_lines.append("        <thead>")
        html_lines.append("            <tr>")
        html_lines.append("                <th onclick='sortTable(this, 0)'>Object Name</th>")
        html_lines.append("                <th onclick='sortTable(this, 1)'>Group Name</th>")
        html_lines.append("                <th onclick='sortTable(this, 2)'>RA / Dec</th>")
        html_lines.append("                <th>Preview</th>")
        html_lines.append("            </tr>")
        html_lines.append("        </thead>")
        html_lines.append("        <tbody>")
        for group_name in groups_by_raslice[raslice]:
            info = group_info[group_name]
            group_dir = find_group_directory(htmldir, region, group_name)
            html_path = "{}/{}/{}/{}.html".format(region, raslice, group_name, group_name)
            montage_file = "qa-SGA2025_{}-montage.png".format(group_name)
            montage_path = group_dir / montage_file
            sky_url = get_sky_viewer_url(info['ra'], info['dec'], info['diam'], region)
            html_lines.append("        <tr>")
            html_lines.append("            <td class='objname'><a href='{}'>{}</a></td>".format(html_path, info['objname']))
            html_lines.append("            <td class='name'>{}</td>".format(group_name))
            html_lines.append("            <td class='coords'>{:.4f}, {:.4f}<br>Diam: {:.2f}' (mult={}) <a href='{}' target='_blank'>[Sky]</a></td>".format(
                info['ra'], info['dec'], info['diam'], info['mult'], sky_url))
            if montage_path.exists():
                thumbnail_path = "{}/{}/{}/{}".format(region, raslice, group_name, montage_file)
                html_lines.append("            <td class='thumbnail'><a href='{}'><div style='width: 100px; height: 75px; overflow: hidden;'><img src='{}' alt='Montage' style='display: block; max-width: none; width: 300px;'></div></a></td>".format(html_path, thumbnail_path))
            else:
                html_lines.append("            <td class='thumbnail'>No preview</td>")
            html_lines.append("        </tr>")
        html_lines.append("        </tbody>")
        html_lines.append("    </table>")
        html_lines.append("    </div>")
    html_lines.extend([
        "",
        "<script>",
        "function filterTable() {",
        "    var input = document.getElementById('searchInput');",
        "    var filter = input.value.toUpperCase();",
        "    var sections = document.getElementsByClassName('raslice-section');",
        "    var totalVisible = 0;",
        "    for (var i = 0; i < sections.length; i++) {",
        "        var table = sections[i].getElementsByClassName('data-table')[0];",
        "        var tbody = table.getElementsByTagName('tbody')[0];",
        "        var rows = tbody.getElementsByTagName('tr');",
        "        var visibleInTable = 0;",
        "        for (var j = 0; j < rows.length; j++) {",
        "            var objname = rows[j].getElementsByTagName('td')[0].textContent;",
        "            var groupname = rows[j].getElementsByTagName('td')[1].textContent;",
        "            if (objname.toUpperCase().indexOf(filter) > -1 || groupname.toUpperCase().indexOf(filter) > -1) {",
        "                rows[j].style.display = '';",
        "                visibleInTable++;",
        "                totalVisible++;",
        "            } else {",
        "                rows[j].style.display = 'none';",
        "            }",
        "        }",
        "        if (filter === '' || visibleInTable === 0) {",
        "            sections[i].style.display = filter === '' ? '' : 'none';",
        "            table.style.display = 'none';",
        "        } else {",
        "            sections[i].style.display = '';",
        "            table.style.display = '';",
        "        }",
        "    }",
        "    document.getElementById('summary').textContent = 'Showing ' + totalVisible + ' groups';",
        "}",
        "function toggleSection(header) {",
        "    var table = header.nextElementSibling;",
        "    if (table.style.display === 'none') {",
        "        table.style.display = '';",
        "    } else {",
        "        table.style.display = 'none';",
        "    }",
        "}",
        "function openSlice(raslice) {",
        "    var header = document.getElementById('raslice-' + raslice);",
        "    var table = header.nextElementSibling;",
        "    table.style.display = '';",
        "}",
        "function sortTable(th, col) {",
        "    var table = th.closest('table');",
        "    var tbody = table.getElementsByTagName('tbody')[0];",
        "    var rows = Array.from(tbody.getElementsByTagName('tr'));",
        "    var ascending = th.dataset.sort !== 'asc';",
        "    rows.sort(function(a, b) {",
        "        var aVal = a.getElementsByTagName('td')[col].textContent.trim();",
        "        var bVal = b.getElementsByTagName('td')[col].textContent.trim();",
        "        if (col === 2) {",
        "            aVal = parseFloat(aVal.split(',')[0]);",
        "            bVal = parseFloat(bVal.split(',')[0]);",
        "        }",
        "        if (aVal < bVal) return ascending ? -1 : 1;",
        "        if (aVal > bVal) return ascending ? 1 : -1;",
        "        return 0;",
        "    });",
        "    rows.forEach(function(row) { tbody.appendChild(row); });",
        "    var headers = table.getElementsByTagName('th');",
        "    for (var i = 0; i < headers.length; i++) {",
        "        headers[i].dataset.sort = '';",
        "    }",
        "    th.dataset.sort = ascending ? 'asc' : 'desc';",
        "}",
        "</script>",
        "</body>",
        "</html>",
    ])
    index_file = htmldir / "index-{}.html".format(region)
    with open(index_file, 'w') as f:
        f.write('\n'.join(html_lines))
    log.info("Generated index: {}".format(index_file))

def make_html(sample, fullsample, htmldir=None, region='dr11-south', mp=1, clobber=False, maketrends=False):
    """
    Generate HTML QA pages for SGA galaxy groups.

    Parameters:
    -----------
    sample : astropy.table.Table
        Table containing GROUP_PRIMARY galaxies only
    fullsample : astropy.table.Table
        Table containing all galaxies (including non-primary group members)
    htmldir : str
        Base HTML directory (default: $SGA_HTML_DIR)
    region : str
        Region name (default: dr11-south)
    mp : int
        Number of processes for multiprocessing (default: 1)
    clobber : bool
        Overwrite existing HTML files (default: False)
    maketrends : bool
        Generate trend plots (not yet implemented)
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
            generate_group_html(group_data, fullsample, htmldir, region, prev_group, next_group, clobber)
    else:
        pool_args = [(idx, gn, sample, fullsample, htmldir, region, valid_groups, clobber)
                     for idx, gn in enumerate(valid_groups)]
        with multiprocessing.Pool(processes=mp) as pool:
            pool.map(generate_group_html_wrapper, pool_args)
    generate_index(htmldir, region, sample)
    log.info("HTML generation complete!")
    return
