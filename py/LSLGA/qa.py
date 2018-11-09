"""
LSLGA.qa
========

Code to do produce various QA (quality assurance) plots. 

"""
import os, pdb
import time, subprocess
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
sns.set(style='ticks', font_scale=1.4, palette='Set2')

PIXSCALE = 0.262

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

    montagefile = os.path.join(htmlgalaxydir, '{}-multiwavelength-montage.png'.format(galaxy))

    if not os.path.isfile(montagefile) or clobber:
        # Make sure all the files exist.
        check = True
        jpgfile = []
        for suffix in ('galex-image', 'custom-image', 'unwise-image'):
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
