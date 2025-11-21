"""
SGA.html
========

Code to generate HTML content.

"""
import pdb

import os, subprocess
import numpy as np
from astropy.table import Table, vstack, join

from SGA.ellipse import REF_APERTURES
from SGA.SGA import RACOLUMN, DECCOLUMN, DIAMCOLUMN, REFIDCOLUMN
from SGA.logger import log


def _get_cutouts_one(args):
    """Wrapper function for the multiprocessing."""
    return get_cutouts_one(*args)


def get_cutouts_one(group, clobber=False):
    """Get viewer cutouts for a single galaxy."""

    layer = get_layer(group)
    groupname = get_groupname(group)

    diam = group_diameter(group) # [arcmin]
    size = np.ceil(diam * 60 / PIXSCALE).astype('int') # [pixels]

    imageurl = '{}/?ra={:.8f}&dec={:.8f}&pixscale={:.3f}&size={:g}&layer={}'.format(
        cutouturl, group['ra'], group['dec'], PIXSCALE, size, layer)

    jpgfile = os.path.join(jpgdir, '{}.jpg'.format(groupname))
    cmd = 'wget --continue -O {:s} "{:s}"' .format(jpgfile, imageurl)
    if os.path.isfile(jpgfile) and not clobber:
        print('File {} exists...skipping.'.format(jpgfile))
    else:
        if os.path.isfile(jpgfile):
            os.remove(jpgfile)
        print(cmd)
        os.system(cmd)


def get_cutouts(groupsample, mp=1, clobber=False):
    """Get viewer cutouts of the whole sample."""

    cutoutargs = list()
    for gg in groupsample:
        cutoutargs.append( (gg, clobber) )

    if mp > 1:
        p = multiprocessing.Pool(mp)
        p.map(_get_cutouts_one, cutoutargs)
        p.close()
    else:
        for args in cutoutargs:
            _get_cutouts_one(args)
    return


def html_javadate():
    """Return a string that embeds a date in a webpage using Javascript.

    """
    import textwrap

    js = textwrap.dedent("""
    <SCRIPT LANGUAGE="JavaScript">
    var months = new Array(13);
    months[1] = "January";
    months[2] = "February";
    months[3] = "March";
    months[4] = "April";
    months[5] = "May";
    months[6] = "June";
    months[7] = "July";
    months[8] = "August";
    months[9] = "September";
    months[10] = "October";
    months[11] = "November";
    months[12] = "December";
    var dateObj = new Date(document.lastModified)
    var lmonth = months[dateObj.getMonth() + 1]
    var date = dateObj.getDate()
    var fyear = dateObj.getYear()
    if (fyear < 2000)
    fyear = fyear + 1900
    document.write(" " + fyear + " " + lmonth + " " + date)
    </SCRIPT>
    """)

    return js


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
        wimg[wimg == 0.] = np.nan

        wmodel = np.sum(opt_invvar * opt_models[iobj, :, :, :], axis=0)
        wnorm = np.sum(opt_invvar, axis=0)
        wmodel[wnorm > 0.] /= wnorm[wnorm > 0.] / pixscale**2 # [nanomaggies/arcsec**2]

        try:
            #norm = matched_norm(wimg, wmodel)
            norm = get_norm(wimg)
        except:
            norm = None
        ax[1+iobj, 0].imshow(wimg, cmap=cmap, origin='lower', interpolation='none',
                             norm=norm)
        norm = get_norm(wmodel)
        ax[1+iobj, 1].imshow(wmodel, cmap=cmap, origin='lower', interpolation='none',
                             norm=norm)
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


def _get_sma_sbthresh(obj, bands):
    val, label = 0., None
    for thresh in [26., 25., 24.]:
        for filt in ['R']:#bands:
            col = f'R{thresh:.0f}_{filt}'
            if col in obj.colnames:
                val = obj[col]
                label = r'$R_{'+filt.lower()+r'}('+f'{thresh:.0f}'+')='+f'{val:.1f}'+r'$ arcsec'
                if val > 0.:
                    return val, label
    return val, label


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
    colors2 = [cmap2a(1), cmap2b(3)]
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

            sma_moment = obj['SMA_MOMENT'] # [arcsec]
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
                xx.axvline(x=sma_moment**0.25, color=colors2[0], lw=2, ls='--', label=label_moment)

                hndls, _ = xx.get_legend_handles_labels()
                if hndls:
                    if sma_sbthresh > 0.:
                        split = -2
                    else:
                        split = -1
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
               unwise_pixscale=2.75, galex=True, unwise=True,
               barlen=None, barlabel=None, verbose=False, clobber=False):
               #radius_mosaic_arcsec=None,
    """Make QA plots.

    """
    from glob import glob
    import fitsio

    allbands = ''.join(bands)
    datasets = ['opt']
    if unwise:
        datasets += ['unwise']
    if galex:
        datasets += ['galex']

    data, tractor, sample, samplesrcs, err = read_multiband_function(
        galaxy, galaxydir, REFIDCOLUMN, bands=bands, run=run,
        niter_geometry=2, pixscale=pixscale, galex_pixscale=galex_pixscale,
        unwise_pixscale=unwise_pixscale, unwise=unwise,
        galex=galex, build_mask=False, read_jpg=True)

    multiband_montage(data, sample, htmlgalaxydir, barlen=barlen,
                      barlabel=barlabel, clobber=clobber)

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

    return 1


def skyserver_link(sdss_objid):
    return 'http://skyserver.sdss.org/dr14/en/tools/explore/summary.aspx?id={:d}'.format(sdss_objid)


# Get the viewer link
def viewer_link(ra, dec, width, sga=False, manga=False, dr10=False):
    baseurl = 'http://legacysurvey.org/viewer-dev/'
    if width > 1200:
        zoom = 13
    elif (width > 400) * (width < 1200):
        zoom = 14
    else:
        zoom = 15

    if dr10:
        drlayer = 'ls-dr10'
    else:
        drlayer = 'ls-dr9'

    layer1 = ''
    if sga:
        layer1 = '&sga&sga-parent'
    if manga:
        layer1 = layer1+'&manga'

    viewer = '{}?ra={:.6f}&dec={:.6f}&zoom={:g}&layer={}{}'.format(
        baseurl, ra, dec, zoom, drlayer, layer1)

    return viewer


def build_htmlhome(sample, htmldir, htmlhome='index.html', pixscale=0.262,
                   racolumn=RACOLUMN, deccolumn=DECCOLUMN, diamcolumn=DIAMCOLUMN,
                   html_raslices=True):
    """Build the home (index.html) page and, optionally, the trends.html top-level
    page.

    """
    htmlhomefile = os.path.join(htmldir, htmlhome)
    print('Building {}'.format(htmlhomefile))

    js = html_javadate()

    # group by RA slices
    raslices = np.array([SGA.io.get_raslice(ra) for ra in sample[racolumn]])
    #rasorted = raslices)

    with open(htmlhomefile, 'w') as html:
        html.write('<html><body>\n')
        html.write('<style type="text/css">\n')
        html.write('table, td, th {padding: 5px; text-align: center; border: 1px solid black;}\n')
        html.write('p {display: inline-block;;}\n')
        html.write('</style>\n')

        html.write('<h1>Virgo Filaments</h1>\n')

        html.write('<p style="width: 75%">\n')
        html.write("""This project is super neat.</p>\n""")

        # The default is to organize the sample by RA slice, but support both options here.
        if html_raslices:
            html.write('<p>The web-page visualizations are organized by one-degree slices of right ascension.</p><br />\n')

            html.write('<table>\n')
            html.write('<tr><th>RA Slice</th><th>Number of Galaxies</th></tr>\n')
            for raslice in sorted(set(raslices)):
                inslice = np.where(raslice == raslices)[0]
                html.write('<tr><td><a href="RA{0}.html"><h3>{0}</h3></a></td><td>{1}</td></tr>\n'.format(raslice, len(inslice)))
            html.write('</table>\n')
        else:
            html.write('<br /><br />\n')
            html.write('<table>\n')
            html.write('<tr>\n')
            html.write('<th> </th>\n')
            #html.write('<th>Index</th>\n')
            html.write('<th>ID</th>\n')
            html.write('<th>Galaxy</th>\n')
            html.write('<th>RA</th>\n')
            html.write('<th>Dec</th>\n')
            html.write('<th>Diameter (arcmin)</th>\n')
            html.write('<th>Viewer</th>\n')
            html.write('</tr>\n')

            galaxy, galaxydir, htmlgalaxydir = SGA.io.get_galaxy_galaxydir(sample, html=True)
            for gal, galaxy1, htmlgalaxydir1 in zip(sample, np.atleast_1d(galaxy), np.atleast_1d(htmlgalaxydir)):

                htmlfile1 = os.path.join(htmlgalaxydir1.replace(htmldir, '')[1:], '{}.html'.format(galaxy1))
                pngfile1 = os.path.join(htmlgalaxydir1.replace(htmldir, '')[1:], '{}-custom-montage-grz.png'.format(galaxy1))
                thumbfile1 = os.path.join(htmlgalaxydir1.replace(htmldir, '')[1:], 'thumb2-{}-custom-montage-grz.png'.format(galaxy1))

                ra1, dec1, diam1 = gal[racolumn], gal[deccolumn], gal[diamcolumn]
                link = viewer_link(ra1, dec1, diam1*2*60/pixscale, sga=True)

                html.write('<tr>\n')
                html.write('<td><a href="{0}"><img src="{1}" height="auto" width="100%"></a></td>\n'.format(pngfile1, thumbfile1))
                #html.write('<td>{}</td>\n'.format(gal['INDEX']))
                html.write('<td>{}</td>\n'.format(gal[REFIDCOLUMN]))
                html.write('<td><a href="{}">{}</a></td>\n'.format(htmlfile1, galaxy1))
                html.write('<td>{:.7f}</td>\n'.format(ra1))
                html.write('<td>{:.7f}</td>\n'.format(dec1))
                html.write('<td>{:.4f}</td>\n'.format(diam1))
                html.write('<td><a href="{}" target="_blank">Link</a></td>\n'.format(link))
                html.write('</tr>\n')
            html.write('</table>\n')

        # close up shop
        html.write('<br /><br />\n')
        html.write('<b><i>Last updated {}</b></i>\n'.format(js))
        html.write('</html></body>\n')

    # Optionally build the individual pages (one per RA slice).
    if html_raslices:
        for raslice in sorted(set(raslices)):
            inslice = np.where(raslice == raslices)[0]
            galaxy, galaxydir, htmlgalaxydir = SGA.io.get_galaxy_galaxydir(sample[inslice], region=region, html=True)

            slicefile = os.path.join(htmldir, 'RA{}.html'.format(raslice))
            print('Building {}'.format(slicefile))

            with open(slicefile, 'w') as html:
                html.write('<html><body>\n')
                html.write('<style type="text/css">\n')
                html.write('table, td, th {padding: 5px; text-align: center; border: 1px solid black;}\n')
                html.write('p {width: "75%";}\n')
                html.write('</style>\n')

                html.write('<h3>RA Slice {}</h3>\n'.format(raslice))

                html.write('<table>\n')
                html.write('<tr>\n')
                #html.write('<th>Number</th>\n')
                html.write('<th> </th>\n')
                #html.write('<th>Index</th>\n')
                html.write('<th>ID</th>\n')
                html.write('<th>Galaxy</th>\n')
                html.write('<th>RA</th>\n')
                html.write('<th>Dec</th>\n')
                html.write('<th>Diameter (arcmin)</th>\n')
                html.write('<th>Viewer</th>\n')

                html.write('</tr>\n')
                for gal, galaxy1, htmlgalaxydir1 in zip(sample[inslice], np.atleast_1d(galaxy), np.atleast_1d(htmlgalaxydir)):

                    htmlfile1 = os.path.join(htmlgalaxydir1.replace(htmldir, '')[1:], '{}.html'.format(galaxy1))
                    pngfile1 = os.path.join(htmlgalaxydir1.replace(htmldir, '')[1:], '{}-custom-montage-grz.png'.format(galaxy1))
                    thumbfile1 = os.path.join(htmlgalaxydir1.replace(htmldir, '')[1:], 'thumb2-{}-custom-montage-grz.png'.format(galaxy1))

                    ra1, dec1, diam1 = gal[racolumn], gal[deccolumn], gal[diamcolumn]
                    link = viewer_link(ra1, dec1, diam1*2*60/pixscale, sga=True)

                    html.write('<tr>\n')
                    #html.write('<td>{:g}</td>\n'.format(count))
                    #print(gal['INDEX'], gal[REFIDCOLUMN], gal['GALAXY'])
                    html.write('<td><a href="{0}"><img src="{1}" height="auto" width="100%"></a></td>\n'.format(pngfile1, thumbfile1))
                    #html.write('<td>{}</td>\n'.format(gal['INDEX']))
                    html.write('<td>{}</td>\n'.format(gal[REFIDCOLUMN]))
                    html.write('<td><a href="{}">{}</a></td>\n'.format(htmlfile1, galaxy1))
                    html.write('<td>{:.7f}</td>\n'.format(ra1))
                    html.write('<td>{:.7f}</td>\n'.format(dec1))
                    html.write('<td>{:.4f}</td>\n'.format(diam1))
                    #html.write('<td>{:.5f}</td>\n'.format(gal[zcolumn]))
                    #html.write('<td>{:.4f}</td>\n'.format(gal['LAMBDA_CHISQ']))
                    #html.write('<td>{:.3f}</td>\n'.format(gal['P_CEN'][0]))
                    html.write('<td><a href="{}" target="_blank">Link</a></td>\n'.format(link))
                    #html.write('<td><a href="{}" target="_blank">Link</a></td>\n'.format(_skyserver_link(gal)))
                    html.write('</tr>\n')
                html.write('</table>\n')
                #count += 1

                html.write('<br /><br />\n')
                html.write('<b><i>Last updated {}</b></i>\n'.format(js))
                html.write('</html></body>\n')



def _build_htmlpage_one(args):
    """Wrapper function for the multiprocessing."""
    return build_htmlpage_one(*args)


def build_htmlpage_one(ii, gal, galaxy1, galaxydir1, htmlgalaxydir1, htmlhome, htmldir,
                       racolumn, deccolumn, diamcolumn, pixscale, nextgalaxy, prevgalaxy,
                       nexthtmlgalaxydir, prevhtmlgalaxydir, verbose, clobber):
    """Build the web page for a single galaxy.

    """
    import fitsio
    from glob import glob
    import SGA.io

    if not os.path.exists(htmlgalaxydir1):
        os.makedirs(htmlgalaxydir1)

    htmlfile = os.path.join(htmlgalaxydir1, '{}.html'.format(galaxy1))
    if os.path.isfile(htmlfile) and not clobber:
        print('File {} exists and clobber=False'.format(htmlfile))
        return

    nexthtmlgalaxydir1 = os.path.join('{}'.format(nexthtmlgalaxydir[ii].replace(htmldir, '')[1:]), '{}.html'.format(nextgalaxy[ii]))
    prevhtmlgalaxydir1 = os.path.join('{}'.format(prevhtmlgalaxydir[ii].replace(htmldir, '')[1:]), '{}.html'.format(prevgalaxy[ii]))

    js = html_javadate()

    # Support routines--

    def _read_ccds_tractor_sample(prefix):
        nccds, tractor, sample = None, None, None

        ccdsfile = glob(os.path.join(galaxydir1, '{}-{}-ccds-*.fits'.format(galaxy1, prefix))) # north or south
        if len(ccdsfile) > 0:
            nccds = fitsio.FITS(ccdsfile[0])[1].get_nrows()

        # samplefile can exist without tractorfile when using --just-coadds
        samplefile = os.path.join(galaxydir1, '{}-sample.fits'.format(galaxy1))
        #samplefile = os.path.join(galaxydir1, '{}-{}-sample.fits'.format(galaxy1, prefix))
        if os.path.isfile(samplefile):
            sample = astropy.table.Table(fitsio.read(samplefile, upper=True))
            if verbose:
                print('Read {} galaxy(ies) from {}'.format(len(sample), samplefile))

        tractorfile = os.path.join(galaxydir1, '{}-{}-tractor.fits'.format(galaxy1, prefix))
        if os.path.isfile(tractorfile):
            cols = ['ref_cat', 'ref_id', 'type', 'sersic', 'shape_r', 'shape_e1', 'shape_e2',
                    'flux_g', 'flux_r', 'flux_z', 'flux_ivar_g', 'flux_ivar_r', 'flux_ivar_z',
                    'flux_fuv', 'flux_nuv', 'flux_ivar_fuv', 'flux_ivar_nuv', 
                    'flux_w1', 'flux_w2', 'flux_w3', 'flux_w4',
                    'flux_ivar_w1', 'flux_ivar_w2', 'flux_ivar_w3', 'flux_ivar_w4']
            tractor = astropy.table.Table(fitsio.read(tractorfile, lower=True, columns=cols))#, rows=irows

            # We just care about the galaxies in our sample
            if prefix == 'custom':
                wt, ws = [], []
                for ii, sid in enumerate(sample[REFIDCOLUMN]):
                    ww = np.where((tractor['ref_cat'] != '  ') * (tractor['ref_id'] == sid))[0]
                    if len(ww) > 0:
                        wt.append(ww)
                        ws.append(ii)
                if len(wt) == 0:
                    print('All galaxy(ies) in {} field dropped from Tractor!'.format(galaxydir1))
                    tractor = None
                else:
                    wt = np.hstack(wt)
                    ws = np.hstack(ws)
                    tractor = tractor[wt]
                    sample = sample[ws]
                    srt = np.argsort(tractor['flux_r'])[::-1]
                    tractor = tractor[srt]
                    sample = sample[srt]
                    assert(np.all(tractor['ref_id'] == sample[REFIDCOLUMN]))

        return nccds, tractor, sample

    def _html_group_properties(html, gal):
        """Build the table of group properties.

        """
        ra1, dec1, diam1 = gal[racolumn], gal[deccolumn], gal[diamcolumn]
        link = viewer_link(ra1, dec1, diam1*2*60/pixscale, sga=True)

        html.write('<h2>Group Properties</h2>\n')

        html.write('<table>\n')
        html.write('<tr>\n')
        #html.write('<th>Number</th>\n')
        #html.write('<th>Index<br />(Primary)</th>\n')
        html.write('<th>ID<br />(Primary)</th>\n')
        html.write('<th>Group Name</th>\n')
        html.write('<th>Group RA</th>\n')
        html.write('<th>Group Dec</th>\n')
        html.write('<th>Group Diameter<br />(arcmin)</th>\n')
        #html.write('<th>Richness</th>\n')
        #html.write('<th>Pcen</th>\n')
        html.write('<th>Viewer</th>\n')
        #html.write('<th>SkyServer</th>\n')
        html.write('</tr>\n')

        html.write('<tr>\n')
        #html.write('<td>{:g}</td>\n'.format(ii))
        #print(gal['INDEX'], gal[REFIDCOLUMN], gal['GALAXY'])
        #html.write('<td>{}</td>\n'.format(gal['INDEX']))
        html.write('<td>{}</td>\n'.format(gal[REFIDCOLUMN]))
        html.write('<td>{}</td>\n'.format(gal['GROUP_NAME']))
        html.write('<td>{:.7f}</td>\n'.format(ra1))
        html.write('<td>{:.7f}</td>\n'.format(dec1))
        html.write('<td>{:.4f}</td>\n'.format(diam1))
        #html.write('<td>{:.5f}</td>\n'.format(gal[zcolumn]))
        #html.write('<td>{:.4f}</td>\n'.format(gal['LAMBDA_CHISQ']))
        #html.write('<td>{:.3f}</td>\n'.format(gal['P_CEN'][0]))
        html.write('<td><a href="{}" target="_blank">Link</a></td>\n'.format(link))
        #html.write('<td><a href="{}" target="_blank">Link</a></td>\n'.format(_skyserver_link(gal)))
        html.write('</tr>\n')
        html.write('</table>\n')

        # Add the properties of each galaxy.
        html.write('<h3>Group Members</h3>\n')
        html.write('<table>\n')
        html.write('<tr>\n')
        html.write('<th>ID</th>\n')
        html.write('<th>Galaxy</th>\n')
        #html.write('<th>Morphology</th>\n')
        html.write('<th>RA</th>\n')
        html.write('<th>Dec</th>\n')
        html.write('<th>D(25)<br />(arcmin)</th>\n')
        #html.write('<th>PA<br />(deg)</th>\n')
        #html.write('<th>e</th>\n')
        html.write('</tr>\n')
        for groupgal in sample:
            #if '031705' in gal['GALAXY']:
            #    print(groupgal['GALAXY'])
            html.write('<tr>\n')
            html.write('<td>{}</td>\n'.format(groupgal[REFIDCOLUMN]))
            html.write('<td>{}</td>\n'.format(groupgal[REFIDCOLUMN]))
            #typ = groupgal['MORPHTYPE'].strip()
            #if typ == '' or typ == 'nan':
            #    typ = '...'
            #html.write('<td>{}</td>\n'.format(typ))
            html.write('<td>{:.7f}</td>\n'.format(groupgal['RA']))
            html.write('<td>{:.7f}</td>\n'.format(groupgal['DEC']))
            html.write('<td>{:.4f}</td>\n'.format(groupgal['DIAM']))
            #if np.isnan(groupgal['PA']):
            #    pa = 0.0
            #else:
            #    pa = groupgal['PA']
            #html.write('<td>{:.2f}</td>\n'.format(pa))
            #html.write('<td>{:.3f}</td>\n'.format(1-groupgal['BA']))
            html.write('</tr>\n')
        html.write('</table>\n')

    def _html_image_mosaics(html):
        html.write('<h2>Image Mosaics</h2>\n')

        if False:
            html.write('<table>\n')
            html.write('<tr><th colspan="3">Mosaic radius</th><th colspan="3">Point-source depth<br />(5-sigma, mag)</th><th colspan="3">Image quality<br />(FWHM, arcsec)</th></tr>\n')
            html.write('<tr><th>kpc</th><th>arcsec</th><th>grz pixels</th><th>g</th><th>r</th><th>z</th><th>g</th><th>r</th><th>z</th></tr>\n')
            html.write('<tr><td>{:.0f}</td><td>{:.3f}</td><td>{:.1f}</td>'.format(
                radius_mosaic_kpc, radius_mosaic_arcsec, radius_mosaic_pixels))
            if bool(ellipse):
                html.write('<td>{:.2f}<br />({:.2f}-{:.2f})</td><td>{:.2f}<br />({:.2f}-{:.2f})</td><td>{:.2f}<br />({:.2f}-{:.2f})</td>'.format(
                    ellipse['psfdepth_g'], ellipse['psfdepth_min_g'], ellipse['psfdepth_max_g'],
                    ellipse['psfdepth_r'], ellipse['psfdepth_min_r'], ellipse['psfdepth_max_r'],
                    ellipse['psfdepth_z'], ellipse['psfdepth_min_z'], ellipse['psfdepth_max_z']))
                html.write('<td>{:.3f}<br />({:.3f}-{:.3f})</td><td>{:.3f}<br />({:.3f}-{:.3f})</td><td>{:.3f}<br />({:.3f}-{:.3f})</td></tr>\n'.format(
                    ellipse['psfsize_g'], ellipse['psfsize_min_g'], ellipse['psfsize_max_g'],
                    ellipse['psfsize_r'], ellipse['psfsize_min_r'], ellipse['psfsize_max_r'],
                    ellipse['psfsize_z'], ellipse['psfsize_min_z'], ellipse['psfsize_max_z']))
            html.write('</table>\n')
            #html.write('<br />\n')

        pngfile, thumbfile = '{}-custom-montage-grz.png'.format(galaxy1), 'thumb-{}-custom-montage-grz.png'.format(galaxy1)
        html.write('<p>Color mosaics showing the data (left panel), model (middle panel), and residuals (right panel).</p>\n')
        html.write('<table width="90%">\n')
        for bandsuffix in ('grz', 'FUVNUV', 'W1W2'):
            pngfile, thumbfile = '{}-custom-montage-{}.png'.format(galaxy1, bandsuffix), 'thumb-{}-custom-montage-{}.png'.format(galaxy1, bandsuffix)
            html.write('<tr><td><a href="{0}"><img src="{1}" alt="Missing file {0}" height="auto" width="100%"></a></td></tr>\n'.format(
                pngfile, thumbfile))
        html.write('</table>\n')

    def _html_ellipsefit_and_photometry(html, tractor, sample):
        html.write('<h2>Elliptical Isophote Analysis</h2>\n')
        if tractor is None:
            html.write('<p>Tractor catalog not available.</p>\n')
            html.write('<h3>Geometry</h3>\n')
            html.write('<h3>Photometry</h3>\n')
            return

        html.write('<h3>Geometry</h3>\n')
        html.write('<table>\n')
        html.write('<tr><th></th>\n')
        html.write('<th colspan="5">Tractor</th>\n')
        html.write('<th colspan="3">Ellipse Moments</th>\n')
        html.write('<th colspan="3">Surface Brightness<br /> Threshold Radii<br />(arcsec)</th>\n')
        html.write('<th colspan="3">Half-light Radii<br />(arcsec)</th>\n')
        html.write('</tr>\n')

        html.write('<tr><th>Galaxy</th>\n')
        html.write('<th>Type</th><th>n</th><th>r(50)<br />(arcsec)</th><th>PA<br />(deg)</th><th>e</th>\n')
        html.write('<th>Size<br />(arcsec)</th><th>PA<br />(deg)</th><th>e</th>\n')
        html.write('<th>R(24)</th><th>R(25)</th><th>R(26)</th>\n')
        html.write('<th>g(50)</th><th>r(50)</th><th>z(50)</th>\n')
        html.write('</tr>\n')

        for ss, tt in zip(sample, tractor):
            ee = np.hypot(tt['shape_e1'], tt['shape_e2'])
            ba = (1 - ee) / (1 + ee)
            pa = 180 - (-np.rad2deg(np.arctan2(tt['shape_e2'], tt['shape_e1']) / 2))
            pa = pa % 180

            html.write('<tr><td>{}</td>\n'.format(ss[REFIDCOLUMN]))
            html.write('<td>{}</td><td>{:.2f}</td><td>{:.3f}</td><td>{:.2f}</td><td>{:.3f}</td>\n'.format(
                tt['type'], tt['sersic'], tt['shape_r'], pa, 1-ba))

            galaxyid = str(tt['ref_id'])
            ellipse = SGA.io.read_ellipsefit(galaxy1, galaxydir1, filesuffix='custom',
                                             galaxy_id=galaxyid, verbose=False)
            if bool(ellipse):
                html.write('<td>{:.3f}</td><td>{:.2f}</td><td>{:.3f}</td>\n'.format(
                    ellipse['sma_moment'], ellipse['pa_moment'], ellipse['eps_moment']))
                    #ellipse['majoraxis']*ellipse['refpixscale'], ellipse['pa_moment'], ellipse['eps_moment']))

                rr = []
                if 'sma_sb24' in ellipse.keys():
                    for rad in [ellipse['sma_sb24'], ellipse['sma_sb25'], ellipse['sma_sb26']]:
                        if rad < 0:
                            rr.append('...')
                        else:
                            rr.append('{:.3f}'.format(rad))
                    html.write('<td>{}</td><td>{}</td><td>{}</td>\n'.format(rr[0], rr[1], rr[2]))
                else:
                    html.write('<td>...</td><td>...</td><td>...</td>\n')

                rr = []
                if 'cog_sma50_g' in ellipse.keys():
                    for rad in [ellipse['cog_sma50_g'], ellipse['cog_sma50_r'], ellipse['cog_sma50_z']]:
                        if rad < 0:
                            rr.append('...')
                        else:
                            rr.append('{:.3f}'.format(rad))
                    html.write('<td>{}</td><td>{}</td><td>{}</td>\n'.format(rr[0], rr[1], rr[2]))
                else:
                    html.write('<td>...</td><td>...</td><td>...</td>\n')                
            else:
                html.write('<td>...</td><td>...</td><td>...</td>\n')
                html.write('<td>...</td><td>...</td><td>...</td>\n')
                html.write('<td>...</td><td>...</td><td>...</td>\n')
                html.write('<td>...</td><td>...</td><td>...</td>\n')
            html.write('</tr>\n')
        html.write('</table>\n')
        
        html.write('<h3>Photometry</h3>\n')
        html.write('<table>\n')
        #html.write('<tr><th></th><th></th>\n')
        #html.write('<th colspan="3"></th>\n')
        #html.write('<th colspan="12">Curve of Growth</th>\n')
        #html.write('</tr>\n')
        html.write('<tr><th></th>\n')
        html.write('<th colspan="9">Tractor</th>\n')
        html.write('<th colspan="9">Curve of Growth</th>\n')
        #html.write('<th colspan="3">&lt R(24)<br />arcsec</th>\n')
        #html.write('<th colspan="3">&lt R(25)<br />arcsec</th>\n')
        #html.write('<th colspan="3">&lt R(26)<br />arcsec</th>\n')
        #html.write('<th colspan="3">Integrated</th>\n')
        html.write('</tr>\n')

        html.write('<tr><th>Galaxy</th>\n')
        html.write('<th>FUV</th><th>NUV</th><th>g</th><th>r</th><th>z</th><th>W1</th><th>W2</th><th>W3</th><th>W4</th>\n')
        html.write('<th>FUV</th><th>NUV</th><th>g</th><th>r</th><th>z</th><th>W1</th><th>W2</th><th>W3</th><th>W4</th>\n')
        html.write('</tr>\n')

        for tt, ss in zip(tractor, sample):
            fuv, nuv, g, r, z, w1, w2, w3, w4 = _get_mags(tt, pipeline=True)
            html.write('<tr><td>{}</td>\n'.format(ss[REFIDCOLUMN]))
            html.write('<td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td>\n'.format(
                fuv, nuv, g, r, z, w1, w2, w3, w4))

            galaxyid = str(tt['ref_id'])
            ellipse = SGA.io.read_ellipsefit(galaxy1, galaxydir1, filesuffix='custom',
                                             galaxy_id=galaxyid, verbose=False)
            if bool(ellipse):# and 'cog_mtot_fuv' in ellipse.keys():
                #g, r, z = _get_mags(ellipse, R24=True)
                #html.write('<td>{}</td><td>{}</td><td>{}</td>\n'.format(g, r, z))
                #g, r, z = _get_mags(ellipse, R25=True)
                #html.write('<td>{}</td><td>{}</td><td>{}</td>\n'.format(g, r, z))
                #g, r, z = _get_mags(ellipse, R26=True)
                #html.write('<td>{}</td><td>{}</td><td>{}</td>\n'.format(g, r, z))
                fuv, nuv, g, r, z, w1, w2, w3, w4 = _get_mags(ellipse, cog=True)                
                html.write('<td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td>\n'.format(
                    fuv, nuv, g, r, z, w1, w2, w3, w4))
                #g, r, z = _get_mags(ellipse, cog=True)
                #html.write('<td>{}</td><td>{}</td><td>{}</td>\n'.format(g, r, z))
            else:
                html.write('<td>...</td><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td>\n')
                html.write('<td>...</td><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td>\n')
            html.write('</tr>\n')
        html.write('</table>\n')

        # Galaxy-specific mosaics--
        for igal in np.arange(len(tractor['ref_id'])):
            galaxyid = str(tractor['ref_id'][igal])
            #html.write('<h4>{}</h4>\n'.format(galaxyid))
            html.write('<h4>{}</h4>\n'.format(sample[REFIDCOLUMN][igal]))

            ellipse = SGA.io.read_ellipsefit(galaxy1, galaxydir1, filesuffix='custom',
                                             galaxy_id=galaxyid, verbose=verbose)
            if not bool(ellipse):
                html.write('<p>Ellipse-fitting not done or failed.</p>\n')
                continue

            html.write('<table width="90%">\n')

            html.write('<tr>\n')
            pngfile = '{}-custom-ellipse-{}-multiband-FUVNUV.png'.format(galaxy1, galaxyid)
            thumbfile = 'thumb-{}-custom-ellipse-{}-multiband-FUVNUV.png'.format(galaxy1, galaxyid)
            html.write('<td><a href="{0}"><img src="{1}" alt="Missing file {1}" height="auto" align="left" width="60%"></a></td>\n'.format(pngfile, thumbfile))
            html.write('</tr>\n')

            html.write('<tr>\n')
            pngfile = '{}-custom-ellipse-{}-multiband.png'.format(galaxy1, galaxyid)
            thumbfile = 'thumb-{}-custom-ellipse-{}-multiband.png'.format(galaxy1, galaxyid)
            html.write('<td><a href="{0}"><img src="{1}" alt="Missing file {1}" height="auto" align="left" width="80%"></a></td>\n'.format(pngfile, thumbfile))
            html.write('</tr>\n')

            html.write('<tr>\n')
            pngfile = '{}-custom-ellipse-{}-multiband-W1W2.png'.format(galaxy1, galaxyid)
            thumbfile = 'thumb-{}-custom-ellipse-{}-multiband-W1W2.png'.format(galaxy1, galaxyid)
            html.write('<td><a href="{0}"><img src="{1}" alt="Missing file {1}" height="auto" align="left" width="100%"></a></td>\n'.format(pngfile, thumbfile))
            html.write('</tr>\n')

            html.write('</table>\n')
            html.write('<br />\n')

            html.write('<table width="90%">\n')
            html.write('<tr>\n')
            pngfile = '{}-custom-ellipse-{}-sbprofile.png'.format(galaxy1, galaxyid)
            html.write('<td width="50%"><a href="{0}"><img src="{0}" alt="Missing file {0}" height="auto" width="100%"></a></td>\n'.format(pngfile))
            pngfile = '{}-custom-ellipse-{}-cog.png'.format(galaxy1, galaxyid)
            html.write('<td><a href="{0}"><img src="{0}" alt="Missing file {0}" height="auto" width="100%"></a></td>\n'.format(pngfile))
            html.write('</tr>\n')

            html.write('<tr>\n')
            pngfile = '{}-custom-ellipse-{}-sed.png'.format(galaxy1, galaxyid)
            html.write('<td width="50%"><a href="{0}"><img src="{0}" alt="Missing file {0}" height="auto" width="100%"></a></td>\n'.format(pngfile))
            html.write('</tr>\n')
            
            html.write('</table>\n')
            #html.write('<br />\n')

    def _html_ccd_diagnostics(html):
        html.write('<h2>CCD Diagnostics</h2>\n')

        html.write('<table width="90%">\n')
        pngfile = '{}-ccdpos.png'.format(galaxy1)
        html.write('<tr><td><a href="{0}"><img src="{0}" alt="Missing file {0}" height="auto" width="100%"></a></td></tr>\n'.format(
            pngfile))
        html.write('</table>\n')
        #html.write('<br />\n')
        
    # Read the catalogs and then build the page--
    nccds, tractor, sample = _read_ccds_tractor_sample(prefix='custom')

    with open(htmlfile, 'w') as html:
        html.write('<html><body>\n')
        html.write('<style type="text/css">\n')
        html.write('table, td, th {padding: 5px; text-align: center; border: 1px solid black}\n')
        html.write('</style>\n')

        # Top navigation menu--
        html.write('<h1>{}</h1>\n'.format(galaxy1))
        raslice = SGA.io.get_raslice(gal[racolumn])
        html.write('<h4>RA Slice {}</h4>\n'.format(raslice))

        html.write('<a href="../../{}">Home</a>\n'.format(htmlhome))
        html.write('<br />\n')
        html.write('<a href="../../{}">Next ({})</a>\n'.format(nexthtmlgalaxydir1, nextgalaxy[ii]))
        html.write('<br />\n')
        html.write('<a href="../../{}">Previous ({})</a>\n'.format(prevhtmlgalaxydir1, prevgalaxy[ii]))

        _html_group_properties(html, gal)
        _html_image_mosaics(html)
        _html_ellipsefit_and_photometry(html, tractor, sample)
        #_html_maskbits(html)
        #_html_ccd_diagnostics(html)

        html.write('<br /><br />\n')
        html.write('<a href="../../{}">Home</a>\n'.format(htmlhome))
        html.write('<br />\n')
        html.write('<a href="../../{}">Next ({})</a>\n'.format(nexthtmlgalaxydir1, nextgalaxy[ii]))
        html.write('<br />\n')
        html.write('<a href="../../{}">Previous ({})</a>\n'.format(prevhtmlgalaxydir1, prevgalaxy[ii]))
        html.write('<br />\n')

        html.write('<br /><b><i>Last updated {}</b></i>\n'.format(js))
        html.write('<br />\n')
        html.write('</html></body>\n')



def make_html(sample=None, datadir=None, htmldir=None, bands=['g', 'r', 'i', 'z'],
              refband='r', region='dr11-south', pixscale=0.262, zcolumn='Z', intflux=None,
              racolumn='GROUP_RA', deccolumn='GROUP_DEC', diamcolumn='GROUP_DIAMETER',
              first=None, last=None, galaxylist=None,
              mp=1, survey=None, makeplots=False,
              htmlhome='index.html', html_raslices=False,
              clobber=False, verbose=True, maketrends=False, ccdqa=False,
              args=None):
    """Make the HTML pages.

    """
    import subprocess
    from astrometry.util.multiproc import multiproc

    import SGA.io
    from SGA.coadds import _mosaic_width

    #datadir = SGA.io.sga_data_dir()
    #htmldir = SGA.io.sga_html_dir()
    datadir = os.path.join(SGA.io.sga_data_dir(), region)
    htmldir = os.path.join(SGA.io.sga_html_dir(), region)
    if not os.path.exists(htmldir):
        os.makedirs(htmldir)

    if sample is None:
        sample = SGA.read_sample(first=first, last=last, galaxylist=galaxylist)

    if type(sample) is astropy.table.row.Row:
        sample = astropy.table.Table(sample)

    # Only create pages for the set of galaxies with a montage.
    keep = np.arange(len(sample))
    _, missing, done, _ = SGA.io.missing_files(sample=sample, region=region, htmldir=htmldir,
                                               htmlindex=True)

    if len(done[0]) == 0:
        print('No galaxies with complete montages!')
        return

    print('Keeping {}/{} galaxies with complete montages.'.format(len(done[0]), len(sample)))
    sample = sample[done[0]]
    #galaxy, galaxydir, htmlgalaxydir = get_galaxy_galaxydir(sample, html=True)

    # Build the home (index.html) page (always, irrespective of clobber)--
    build_htmlhome(sample, htmldir, htmlhome=htmlhome, pixscale=pixscale,
                   racolumn=racolumn, deccolumn=deccolumn, diamcolumn=diamcolumn,
                   html_raslices=html_raslices)

    # Now build the individual pages in parallel.
    if html_raslices:
        raslices = np.array([SGA.io.get_raslice(ra) for ra in sample[racolumn]])
        rasorted = np.argsort(raslices)
        galaxy, galaxydir, htmlgalaxydir = SGA.io.get_galaxy_galaxydir(sample[rasorted], region=region, html=True)
    else:
        rasorted = np.arange(len(sample))
        galaxy, galaxydir, htmlgalaxydir = SGA.io.get_galaxy_galaxydir(sample, region=region, html=True)

    nextgalaxy = np.roll(np.atleast_1d(galaxy), -1)
    prevgalaxy = np.roll(np.atleast_1d(galaxy), 1)
    nexthtmlgalaxydir = np.roll(np.atleast_1d(htmlgalaxydir), -1)
    prevhtmlgalaxydir = np.roll(np.atleast_1d(htmlgalaxydir), 1)

    mp = multiproc(nthreads=mp)
    args = []
    for ii, (gal, galaxy1, galaxydir1, htmlgalaxydir1) in enumerate(zip(
        sample[rasorted], np.atleast_1d(galaxy), np.atleast_1d(galaxydir), np.atleast_1d(htmlgalaxydir))):
        args.append([ii, gal, galaxy1, galaxydir1, htmlgalaxydir1, htmlhome, htmldir,
                     racolumn, deccolumn, diamcolumn, pixscale, nextgalaxy,
                     prevgalaxy, nexthtmlgalaxydir, prevhtmlgalaxydir, verbose,
                     clobber])
    ok = mp.map(_build_htmlpage_one, args)

    return 1
