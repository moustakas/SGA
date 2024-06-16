"""
SGA.qa
======

Code to do produce various QA (quality assurance) plots. 

"""
import os, pdb
import warnings
import time, subprocess
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import SGA.misc

#import seaborn as sns
#sns.set(style='ticks', font_scale=1.4, palette='Set2')
sns, _ = SGA.misc.plot_style()


def draw_ellipse_on_png(im, x0, y0, ba, pa, major_axis_diameter_arcsec,
                        pixscale, color='#3388ff', linewidth=3):
    """Write me.

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


def _sbprofile_colors():
    """Return an iterator of colors good for the surface brightness profile plots.
    https://seaborn.pydata.org/generated/seaborn.color_palette.html#seaborn.color_palette

    """
    _colors = sns.color_palette('Set1', n_colors=11, desat=0.75)
    colors = iter([ _colors[1], _colors[2], _colors[0], _colors[3], _colors[4],
                    _colors[5], _colors[6], _colors[7], _colors[8],
                    _colors[9], _colors[10]])
    return colors

def qa_binned_radec(cat, nside=64, png=None):
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
    """Montage the multiwavelength coadds into a nice QAplot."""

    # Show the data (GALEX, LS, unWISE from left to right).
    montagefile = os.path.join(htmlgalaxydir, '{}-multiwavelength-data.png'.format(galaxy))

    if not os.path.isfile(montagefile) or clobber:
        # Make sure all the files exist.
        check = True
        jpgfile = []
        for suffix in ('image-FUVNUV', 'custom-image-grz', 'image-W1W2'):
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
                       'custom-image-grz', 'custom-model-nocentral-grz', 'custom-image-central-grz',
                       'image-W1W2', 'model-nocentral-W1W2', 'image-central-W1W2'):
            _jpgfile = os.path.join(galaxydir, '{}-{}.jpg'.format(galaxy, suffix))
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

def ellipse_sbprofile(ellipsefit, minerr=0.0):
    """Convert ellipse-fitting results to a magnitude, color, and surface brightness
    profiles.

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
    """Display the multiwavelength surface brightness profile.

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
            
        ax2.text(0.03, 0.09, 'PSF\n(3$\sigma$)', ha='center', va='center',
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
    """Display the isophote fitting results."""

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
    """Plot up the curve of growth versus semi-major axis.

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
    """Display the multi-band images and, optionally, the isophotal fits based on
    either MGE and/or Ellipse.

    vertical -- for talks...

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

def display_ccdpos(onegal, ccds, radius=None, pixscale=0.262, 
                   png=None, verbose=False):
    """Visualize the position of all the CCDs contributing to the image stack of a
    single galaxy.

    """
    if radius is None:
        radius = 100 # [pixels]

    wcs = SGA.misc.simple_wcs(onegal, radius=radius, pixscale=pixscale)
    width, height = wcs.get_width() * pixscale / 3600, wcs.get_height() * pixscale / 3600 # [degrees]
    bb, bbcc = wcs.radec_bounds(), wcs.radec_center() # [degrees]
    pad = 0.2 # [degrees]

    fig, allax = plt.subplots(1, 3, figsize=(12, 5), sharey=True, sharex=True)

    for ax, band in zip(allax, ('g', 'r', 'z')):
        ax.set_aspect('equal')
        ax.set_xlim(bb[0]+width+pad, bb[0]-pad)
        ax.set_ylim(bb[2]-pad, bb[2]+height+pad)
        ax.set_xlabel('RA (deg)')
        ax.text(0.9, 0.05, band, ha='center', va='bottom',
                transform=ax.transAxes, fontsize=18)

        if band == 'g':
            ax.set_ylabel('Dec (deg)')
        ax.get_xaxis().get_major_formatter().set_useOffset(False)
        #ax.add_patch(patches.Rectangle((bb[0], bb[2]), bb[1]-bb[0], bb[3]-bb[2],
        #                               fill=False, edgecolor='black', lw=3, ls='--'))
        ax.add_patch(patches.Circle((bbcc[0], bbcc[1]), radius * pixscale / 3600,
                                    fill=False, edgecolor='black', lw=2))
        ax.add_patch(patches.Circle((bbcc[0], bbcc[1]), 2*radius * pixscale / 3600, # inner sky annulus
                                    fill=False, edgecolor='black', lw=1))
        ax.add_patch(patches.Circle((bbcc[0], bbcc[1]), 5*radius * pixscale / 3600, # outer sky annulus
                                    fill=False, edgecolor='black', lw=1))

        these = np.where(ccds.filter == band)[0]
        col = plt.cm.Set1(np.linspace(0, 1, len(ccds)))
        for ii, ccd in enumerate(ccds[these]):
            #print(ccd.expnum, ccd.ccdname, ccd.filter)
            W, H, ccdwcs = SGA.misc.ccdwcs(ccd)

            cc = ccdwcs.radec_bounds()
            ax.add_patch(patches.Rectangle((cc[0], cc[2]), cc[1]-cc[0],
                                           cc[3]-cc[2], fill=False, lw=2, 
                                           edgecolor=col[these[ii]],
                                           label='ccd{:02d}'.format(these[ii])))
            ax.legend(ncol=2, frameon=False, loc='upper left', fontsize=10)

    plt.subplots_adjust(bottom=0.15, wspace=0.05, left=0.1, right=0.97, top=0.95)

    if png:
        if verbose:
            print('Writing {}'.format(png))
        fig.savefig(png)
        plt.close(fig)
    else:
        plt.show()

