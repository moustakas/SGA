"""
LSLGA.qa
========

Code to do produce various QA (quality assurance) plots. 

"""
import os, pdb
import time, subprocess
import numpy as np
import matplotlib.pyplot as plt

import LSLGA.misc

#import seaborn as sns
#sns.set(style='ticks', font_scale=1.4, palette='Set2')
sns, _ = LSLGA.misc.plot_style()

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

    sbprofile['gr'] = sbprofile['mu_g'] - sbprofile['mu_r']
    sbprofile['rz'] = sbprofile['mu_r'] - sbprofile['mu_z']
    sbprofile['gr_err'] = np.sqrt(sbprofile['mu_g_err']**2 + sbprofile['mu_r_err']**2)
    sbprofile['rz_err'] = np.sqrt(sbprofile['mu_r_err']**2 + sbprofile['mu_z_err']**2)

    return sbprofile

def display_ellipse_sbprofile(ellipsefit, skyellipsefit={}, minerr=0.0,
                              png=None, verbose=True):
    """Display the multiwavelength surface brightness profile.

    """
    import astropy.stats
    #from legacyhalos.ellipse import ellipse_sbprofile

    if ellipsefit['success']:
        sbprofile = ellipse_sbprofile(ellipsefit, minerr=minerr)

        band, refband = ellipsefit['band'], ellipsefit['refband']
        redshift, pixscale = ellipsefit['redshift'], ellipsefit['pixscale']
        smascale = LSLGA.misc.arcsec2kpc(redshift) # [kpc/arcsec]

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

        ax2.fill_between(sbprofile['sma'],
                         sbprofile['rz'] - sbprofile['rz_err'],
                         sbprofile['rz'] + sbprofile['rz_err'],
                         label=r'$r - z$', color=next(colors), alpha=0.75,
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
        

def qa_curveofgrowth(ellipsefit, png=None, verbose=True):
    """Plot up the curve of growth versus semi-major axis.

    """
    fig, ax = plt.subplots(figsize=(9, 7))
    band, refband, redshift = ellipsefit['band'], ellipsefit['refband'], ellipsefit['redshift']

    maxsma = ellipsefit['apphot_sma_{}'.format(refband)].max()
    smascale = LSLGA.misc.arcsec2kpc(redshift) # [kpc/arcsec]

    yfaint, ybright = 0, 50
    for filt in band:
        flux = ellipsefit['apphot_mag_{}'.format(filt)]
        good = np.where( np.isfinite(flux) * (flux > 0) )[0]
        sma = ellipsefit['apphot_sma_{}'.format(filt)][good]
        mag = 22.5-2.5*np.log10(flux[good])
        ax.plot(sma, mag, label=filt)

        print(filt, np.mean(mag[-5:]))
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
