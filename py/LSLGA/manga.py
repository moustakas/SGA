"""
LSLGA.manga
===========

Code to deal with the MaNGA-NSA sample and project.

"""
import os
import pdb
import numpy as np

import fitsio
from astropy.table import Table, Column, hstack
from astrometry.util.fits import fits_table

import LSLGA.qa
import LSLGA.misc

from legacyhalos.html import make_plots, _javastring
from legacyhalos.misc import plot_style

sns = plot_style()

RADIUSFACTOR = 10

def manga_dir():
    """Top-level MaNGA directory (should be an environment variable...)."""
    if 'NERSC_HOST' in os.environ:
        mangadir = os.path.join(os.getenv('SCRATCH'), 'manga-nsa')
    else:
        print('Where am I?')
        raise IOError
    return mangadir

def sample_dir():
    sdir = os.path.join(manga_dir(), 'sample')
    if not os.path.isdir(sdir):
        os.makedirs(sdir, exist_ok=True)
    return sdir

def analysis_dir():
    adir = os.path.join(manga_dir(), 'analysis')
    if not os.path.isdir(adir):
        os.makedirs(adir, exist_ok=True)
    return adir

def html_dir():
    #if 'NERSC_HOST' in os.environ:
    #    htmldir = '/global/project/projectdirs/cosmo/www/temp/ioannis/LSLGA'
    #else:
    #    htmldir = os.path.join(LSLGA_dir(), 'html')

    htmldir = os.path.join(manga_dir(), 'html')
    if not os.path.isdir(htmldir):
        os.makedirs(htmldir, exist_ok=True)
    return htmldir

def read_manga_parent(verbose=False):
    """Read the parent MaNGA-NSA catalog.
    
    """
    sampledir = sample_dir()
    mangafile = os.path.join(sampledir, 'drpall-v2_1_2.fits')
    nsafile = os.path.join(sampledir, 'nsa_v1_0_1.fits')

    allmanga = Table(fitsio.read(mangafile, upper=True))
    _, uindx = np.unique(allmanga['MANGAID'], return_index=True)
    manga = allmanga[uindx]
    if verbose:
        print('Read {}/{} unique galaxies from {}'.format(len(manga), len(allmanga), mangafile), flush=True)
    #plateifu = [pfu.strip() for pfu in manga['PLATEIFU']]

    catid, rowid = [], []
    for mid in manga['MANGAID']:
        cid, rid = mid.split('-')
        catid.append(cid.strip())
        rowid.append(rid.strip())
    catid, rowid = np.hstack(catid), np.hstack(rowid)
    keep = np.where(catid == '1')[0] # NSA
    rows = rowid[keep].astype(np.int32)

    print('Selected {} MaNGA galaxies from the NSA'.format(len(rows)))
    #ww = [np.argwhere(rr[0]==rows) for rr in np.array(np.unique(rows, return_counts=True)).T if rr[1]>=2]

    srt = np.argsort(rows)
    manga = manga[keep][srt]
    nsa = Table(fitsio.read(nsafile, rows=rows[srt], upper=True))
    if verbose:
        print('Read {} galaxies from {}'.format(len(nsa), nsafile), flush=True)
    nsa.rename_column('PLATE', 'PLATE_NSA')
    
    return hstack( (manga, nsa) )

def get_samplefile(dr=None, ccds=False):

    suffix = 'fits'
    if dr is not None:
        if ccds:
            samplefile = os.path.join(sample_dir(), 'manga-nsa-{}-ccds.{}'.format(dr, suffix))
        else:
            samplefile = os.path.join(sample_dir(), 'manga-nsa-{}.{}'.format(dr, suffix))
    else:
        samplefile = os.path.join(sample_dir(), 'manga-nsa.{}'.format(suffix))
        
    return samplefile

def read_sample(columns=None, dr='dr67', ccds=False, verbose=False,
                first=None, last=None):
    """Read the sample."""
    samplefile = get_samplefile(dr=dr, ccds=ccds)
    if ccds:
        sample = Table(fitsio.read(samplefile, columns=columns, upper=True))
        if verbose:
            print('Read {} CCDs from {}'.format(len(sample), samplefile))
    else:
        info = fitsio.FITS(samplefile)
        nrows = info[1].get_nrows()
        if first is None:
            first = 0
        if last is None:
            last = nrows
        if first == last:
            last = last + 1
        rows = np.arange(first, last)

        sample = Table(info[1].read(rows=rows))
        if verbose:
            if len(rows) == 1:
                print('Read galaxy index {} from {}'.format(first, samplefile))
            else:
                print('Read galaxy indices {} through {} (N={}) from {}'.format(
                    first, last-1, len(sample), samplefile))

    return sample

def make_html(sample, analysisdir=None, htmldir=None, band=('g', 'r', 'z'),
              refband='r', pixscale=0.262, nproc=1, dr='dr7', ccdqa=False,
              makeplots=True, survey=None, clobber=False, verbose=True):
    """Make the HTML pages.

    """
    #import legacyhalos.io
    #from legacyhalos.misc import cutout_radius_150kpc

    if analysisdir is None:
        analysisdir = analysis_dir()
    if htmldir is None:
        htmldir = html_dir()

    # Write the last-updated date to a webpage.
    js = _javastring()       

    # Get the viewer link
    def _viewer_link(onegal, dr):
        baseurl = 'http://legacysurvey.org/viewer/'
        width = 3 * onegal['NSA_PETRO_TH50'] / pixscale # [pixels]
        if width > 400:
            zoom = 14
        else:
            zoom = 15
        viewer = '{}?ra={:.6f}&dec={:.6f}&zoom={:g}&layer=decals-{}'.format(
            baseurl, onegal['RA'], onegal['DEC'], zoom, dr)
        return viewer

    homehtml = 'index.html'

    # Build the home (index.html) page--
    if not os.path.exists(htmldir):
        os.makedirs(htmldir)
    htmlfile = os.path.join(htmldir, homehtml)

    with open(htmlfile, 'w') as html:
        html.write('<html><body>\n')
        html.write('<style type="text/css">\n')
        html.write('table, td, th {padding: 5px; text-align: left; border: 1px solid black;}\n')
        html.write('</style>\n')

        html.write('<h1>Central Galaxies: HSC vs DECaLS</h1>\n')
        html.write('<p>\n')
        html.write('<a href="https://github.com/moustakas/legacyhalos">Code and documentation</a>\n')
        html.write('</p>\n')

        html.write('<table>\n')
        html.write('<tr>\n')
        html.write('<th>Number</th>\n')
        html.write('<th>Galaxy</th>\n')
        html.write('<th>RA</th>\n')
        html.write('<th>Dec</th>\n')
        html.write('<th>Redshift</th>\n')
        html.write('<th>Viewer</th>\n')
        html.write('</tr>\n')
        for ii, onegal in enumerate( np.atleast_1d(sample) ):
            galaxy = onegal['MANGAID']
            if type(galaxy) is np.bytes_:
                galaxy = galaxy.decode('utf-8')
                
            htmlfile = os.path.join(galaxy, '{}.html'.format(galaxy))
            html.write('<tr>\n')
            html.write('<td>{:g}</td>\n'.format(ii))
            html.write('<td><a href="{}">{}</a></td>\n'.format(htmlfile, galaxy))
            html.write('<td>{:.7f}</td>\n'.format(onegal['RA']))
            html.write('<td>{:.7f}</td>\n'.format(onegal['DEC']))
            html.write('<td>{:.5f}</td>\n'.format(onegal['Z']))
            html.write('<td><a href="{}" target="_blank">Link</a></td>\n'.format(_viewer_link(onegal, dr)))
            html.write('</tr>\n')
        html.write('</table>\n')
        
        html.write('<br /><br />\n')
        html.write('<b><i>Last updated {}</b></i>\n'.format(js))
        html.write('</html></body>\n')
        html.close()

    # Make a separate HTML page for each object.
    for ii, onegal in enumerate( np.atleast_1d(sample) ):
        galaxy = onegal['MANGAID']
        if type(galaxy) is np.bytes_:
            galaxy = galaxy.decode('utf-8')
        plateifu = onegal['PLATEIFU']
        if type(plateifu) is np.bytes_:
            plateifu = plateifu.decode('utf-8')

        survey.output_dir = os.path.join(analysisdir, galaxy)
        survey.ccds = fits_table(os.path.join(survey.output_dir, '{}-ccds.fits'.format(galaxy)))
        
        htmlgalaxydir = os.path.join(htmldir, '{}'.format(galaxy))
        if not os.path.exists(htmlgalaxydir):
            os.makedirs(htmlgalaxydir)

        htmlfile = os.path.join(htmlgalaxydir, '{}.html'.format(galaxy))
        with open(htmlfile, 'w') as html:
            html.write('<html><body>\n')
            html.write('<style type="text/css">\n')
            html.write('table, td, th {padding: 5px; text-align: left; border: 1px solid black;}\n')
            html.write('</style>\n')

            html.write('<h1>MaNGA ID {}</h1>\n'.format(galaxy))

            html.write('<a href="../{}">Home</a>\n'.format(homehtml))
            html.write('<br />\n')
            html.write('<br />\n')

            # Table of properties
            html.write('<table>\n')
            html.write('<tr>\n')
            html.write('<th>Number</th>\n')
            html.write('<th>MaNGA ID</th>\n')
            html.write('<th>PLATEIFU</th>\n')
            html.write('<th>RA</th>\n')
            html.write('<th>Dec</th>\n')
            html.write('<th>Redshift</th>\n')
            html.write('<th>Viewer</th>\n')
            html.write('</tr>\n')

            html.write('<tr>\n')
            html.write('<td>{:g}</td>\n'.format(ii))
            html.write('<td>{}</td>\n'.format(galaxy))
            html.write('<td>{}</td>\n'.format(plateifu))
            html.write('<td>{:.7f}</td>\n'.format(onegal['RA']))
            html.write('<td>{:.7f}</td>\n'.format(onegal['DEC']))
            html.write('<td>{:.5f}</td>\n'.format(onegal['Z']))
            html.write('<td><a href="{}" target="_blank">Link</a></td>\n'.format(_viewer_link(onegal, dr)))
            html.write('</tr>\n')
            html.write('</table>\n')

            width_kpc = 2 * RADIUSFACTOR * onegal['NSA_PETRO_TH50'] / LSLGA.misc.arcsec2kpc(onegal['Z'])
            
            html.write('<h2>Multiwavelength mosaics</h2>\n')
            html.write("""<p>From left to right: GALEX (FUV/NUV), DESI Legacy Surveys (grz), and unWISE (W1/W2)
            mosaic ({0:.0f} kpc on a side).</p>\n""".format(width_kpc))
            html.write('<table width="90%">\n')
            pngfile = '{}-multiwavelength-montage.png'.format(galaxy)
            html.write('<tr><td><a href="{0}"><img src="{0}" alt="Missing file {0}" height="auto" width="100%"></a></td></tr>\n'.format(
                pngfile))
            html.write('</table>\n')
            #html.write('<br />\n')
            
            ###########################################################################
            html.write('<h2>Image modeling</h2>\n')
            #html.write('<p>Each mosaic (left to right: data, model of all but the central galaxy, residual image containing just the central galaxy) is 300 kpc by 300 kpc.</p>\n')
            html.write('<table width="90%">\n')
            pngfile = '{}-FUVNUV-montage.png'.format(galaxy)
            html.write('<tr><td><a href="{0}"><img src="{0}" alt="Missing file {0}" height="auto" width="100%"></a></td></tr>\n'.format(
                pngfile))
            #html.write('<tr><td>Data, Model, Residuals</td></tr>\n')
            html.write('</table>\n')
            html.write('<br />\n')

            html.write('<table width="90%">\n')
            pngfile = '{}-grz-montage.png'.format(galaxy)
            html.write('<tr><td><a href="{0}"><img src="{0}" alt="Missing file {0}" height="auto" width="100%"></a></td></tr>\n'.format(
                pngfile))
            #html.write('<tr><td>Data, Model, Residuals</td></tr>\n')
            html.write('</table>\n')
            html.write('<br />\n')
            
            html.write('<table width="90%">\n')
            pngfile = '{}-W1W2-montage.png'.format(galaxy)
            html.write('<tr><td><a href="{0}"><img src="{0}" alt="Missing file {0}" height="auto" width="100%"></a></td></tr>\n'.format(
                pngfile))
            #html.write('<tr><td>Data, Model, Residuals</td></tr>\n')
            html.write('</table>\n')
            html.write('<br />\n')
            
            ###########################################################################
            
            html.write('<h2>Elliptical Isophote Analysis</h2>\n')
            html.write('<table width="90%">\n')
            html.write('<tr>\n')
            pngfile = '{}-ellipse-multiband.png'.format(galaxy)
            html.write('<td><a href="{0}"><img src="{0}" alt="Missing file {0}" height="auto" width="100%"></a></td>\n'.format(
                pngfile))
            html.write('</tr>\n')
            html.write('</table>\n')

            html.write('<table width="90%">\n')
            html.write('<tr>\n')
            pngfile = '{}-ellipse-sbprofile.png'.format(galaxy)
            html.write('<td width="50%"><a href="{0}"><img src="{0}" alt="Missing file {0}" height="auto" width="100%"></a></td>\n'.format(
                pngfile))
            html.write('<td></td>\n')
            html.write('</tr>\n')
            html.write('</table>\n')
            
            html.write('<h2>Surface Brightness Profile Modeling</h2>\n')
            html.write('<table width="90%">\n')

            # single-sersic
            html.write('<tr>\n')
            html.write('<th>Single Sersic (No Wavelength Dependence)</th><th>Single Sersic</th>\n')
            html.write('</tr>\n')
            html.write('<tr>\n')
            pngfile = '{}-sersic-single-nowavepower.png'.format(galaxy)
            html.write('<td><a href="{0}"><img src="{0}" alt="Missing file {0}" height="auto" width="100%"></a></td>\n'.format(
                pngfile))
            pngfile = '{}-sersic-single.png'.format(galaxy)
            html.write('<td><a href="{0}"><img src="{0}" alt="Missing file {0}" height="auto" width="100%"></a></td>\n'.format(
                pngfile))
            html.write('</tr>\n')

            # Sersic+exponential
            html.write('<tr>\n')
            html.write('<th>Sersic+Exponential (No Wavelength Dependence)</th><th>Sersic+Exponential</th>\n')
            html.write('</tr>\n')
            html.write('<tr>\n')
            pngfile = '{}-sersic-exponential-nowavepower.png'.format(galaxy)
            html.write('<td><a href="{0}"><img src="{0}" alt="Missing file {0}" height="auto" width="100%"></a></td>\n'.format(
                pngfile))
            pngfile = '{}-sersic-exponential.png'.format(galaxy)
            html.write('<td><a href="{0}"><img src="{0}" alt="Missing file {0}" height="auto" width="100%"></a></td>\n'.format(
                pngfile))
            html.write('</tr>\n')

            # double-sersic
            html.write('<tr>\n')
            html.write('<th>Double Sersic (No Wavelength Dependence)</th><th>Double Sersic</th>\n')
            html.write('</tr>\n')
            html.write('<tr>\n')
            pngfile = '{}-sersic-double-nowavepower.png'.format(galaxy)
            html.write('<td><a href="{0}"><img src="{0}" alt="Missing file {0}" height="auto" width="100%"></a></td>\n'.format(
                pngfile))
            pngfile = '{}-sersic-double.png'.format(galaxy)
            html.write('<td><a href="{0}"><img src="{0}" alt="Missing file {0}" height="auto" width="100%"></a></td>\n'.format(
                pngfile))
            html.write('</tr>\n')
            html.write('</table>\n')
            html.write('<br />\n')

            html.write('<h2>CCD Diagnostics</h2>\n')
            html.write('<table width="90%">\n')
            html.write('<tr>\n')
            pngfile = '{}-ccdpos.png'.format(galaxy)
            html.write('<td><a href="{0}"><img src="{0}" alt="Missing file {0}" height="auto" width="100%"></a></td>\n'.format(
                pngfile))
            html.write('</tr>\n')
            
            for iccd in range(len(survey.ccds)):
                html.write('<tr>\n')
                pngfile = '{}-2d-ccd{:02d}.png'.format(galaxy, iccd)
                html.write('<td><a href="{0}"><img src="{0}" alt="Missing file {0}" height="auto" width="100%"></a></td>\n'.format(
                    pngfile))
                html.write('</tr>\n')
            html.write('</table>\n')
            html.write('<br />\n')
            
            html.write('<a href="../{}">Home</a>\n'.format(homehtml))
            html.write('<br />\n')

            html.write('<br /><b><i>Last updated {}</b></i>\n'.format(js))
            html.write('<br />\n')
            html.write('</html></body>\n')
            html.close()

    if makeplots:
        for onegal in sample:
            galaxy = onegal['MANGAID']
            if type(galaxy) is np.bytes_:
                galaxy = galaxy.decode('utf-8')
            galaxydir = os.path.join(analysisdir, galaxy)
            htmlgalaxydir = os.path.join(htmldir, galaxy)

            survey.output_dir = os.path.join(analysisdir, galaxy)
            survey.ccds = fits_table(os.path.join(survey.output_dir, '{}-ccds.fits'.format(galaxy)))

            # Custom plots
            LSLGA.qa.qa_multiwavelength_coadds(galaxy, galaxydir, htmlgalaxydir,
                                               clobber=clobber, verbose=verbose)
            LSLGA.qa.qa_unwise_coadds(galaxy, galaxydir, htmlgalaxydir,
                                      clobber=clobber, verbose=verbose)
            LSLGA.qa.qa_galex_coadds(galaxy, galaxydir, htmlgalaxydir,
                                     clobber=clobber, verbose=verbose)
            
            # Plots common to legacyhalos
            make_plots([onegal], galaxylist=[galaxy], analysisdir=analysisdir,
                       htmldir=htmldir, clobber=clobber, verbose=verbose,
                       survey=survey, refband=refband, pixscale=pixscale,
                       band=band, nproc=nproc, ccdqa=ccdqa, trends=False)

    print('HTML pages written to {}'.format(htmldir))
