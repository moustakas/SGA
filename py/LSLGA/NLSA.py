"""
LSLGA.NLSA
==========

Code to deal with the NLSA sample and project.

"""
import os
import pdb
import numpy as np

import fitsio
from astropy.table import Table, Column, hstack
from astrometry.util.fits import fits_table

import LSLGA.misc

RADIUSFACTOR = 10
MANGA_RADIUS = 36.75 # / 2 # [arcsec]

def get_galaxy_galaxydir(cat, datadir=None, htmldir=None, html=False):
    """Retrieve the galaxy name and the (nested) directory.

    """
    import astropy
    import healpy as hp
    from LSLGA.misc import radec2pix
    
    nside = 8 # keep hard-coded
    
    if datadir is None:
        datadir = NLSA_dir()
    if htmldir is None:
        htmldir = NLSA_dir()

    def get_healpix_subdir(nside, pixnum, datadir):
        subdir = os.path.join(str(pixnum // 100), str(pixnum))
        return os.path.abspath(os.path.join(datadir, str(nside), subdir))

    if type(cat) is astropy.table.row.Row:
        ngal = 1
        if cat['GALAXY'].strip() != '':
            galaxy = [cat['GALAXY'].strip()]
        else:
            galaxy = ['{}-{}'.format(cat['BRICKNAME'], cat['OBJID'])]
        #galaxy = ['{:08d}'.format(cat['ID'])]
        pixnum = [radec2pix(nside, cat['RA'], cat['DEC'])]
    else:
        ngal = len(cat)
        galaxy = []
        for gg, bb, oo in zip(cat['GALAXY'], cat['BRICKNAME'], cat['OBJID']):
            if gg.strip() != '':
                galaxy.append(gg.strip())
            else:
                galaxy.append('{}-{}'.format(bb, oo))
        galaxy = np.array(galaxy)
        #galaxy = np.array(['{:08d}'.format(mid) for mid in cat['ID']])
        pixnum = radec2pix(nside, cat['RA'], cat['DEC']).data

    galaxydir = np.array([os.path.join(get_healpix_subdir(nside, pix, datadir), gal)
                          for pix, gal in zip(pixnum, galaxy)])
    if html:
        htmlgalaxydir = np.array([os.path.join(get_healpix_subdir(nside, pix, htmldir), gal)
                                  for pix, gal in zip(pixnum, galaxy)])

    if ngal == 1:
        galaxy = galaxy[0]
        galaxydir = galaxydir[0]
        if html:
            htmlgalaxydir = htmlgalaxydir[0]

    if html:
        return galaxy, galaxydir, htmlgalaxydir
    else:
        return galaxy, galaxydir

def NLSA_dir():
    """Top-level NLSA directory (should be an environment variable...)."""
    print('Use an environment variable for NLSA_DIR!')
    nlsadir = os.path.join(os.getenv('LSLGA_DIR'), 'NLSA')
    return nlsadir

def sample_dir():
    sdir = NLSA_dir()
    #sdir = os.path.join(NLSA_dir(), 'sample')
    if not os.path.isdir(sdir):
        os.makedirs(sdir, exist_ok=True)
    return sdir

def html_dir():
    #if 'NERSC_HOST' in os.environ:
    #    htmldir = '/global/project/projectdirs/cosmo/www/temp/ioannis/LSLGA'
    #else:
    #    htmldir = os.path.join(LSLGA_dir(), 'html')
    htmldir = os.path.join(NLSA_dir(), 'html')
    if not os.path.isdir(htmldir):
        os.makedirs(htmldir, exist_ok=True)
    return htmldir

def read_nlsa_parent(verbose=False, camera='90prime-mosaic', first=None,
                     last=None, proposal=False):
    """Read the parent NLSA catalog.
    
    """
    if proposal:
        # Make some multiwavelength mosaics for the proposal.
        sample = Table()
        sample['GALAXY'] = np.array(['NGC3938', 'NGC5322', 'IC4182', 'NGC3719-GROUP'])
        sample['BRICKNAME'] = np.array(['', '', '', ''])
        sample['OBJID'] = np.array([0, 1, 2, 3])
        sample['RA'] = np.array([178.205851, 207.313452, 196.455276, 173.05609]).astype('f8')
        sample['DEC'] = np.array([44.120774, 60.190476, 37.604659, 0.819287]).astype('f8')
        sample['REFF'] = np.array([78.6/2, 78.6/2, 78.6/2, 20.0]).astype('f4') # [arcsec]
        # Move the center of the galaxy group over a bit
        sample['RA'][3] = 173.0722 # move the center over a bit
        sample['DEC'][3] = 0.8194
        sample = sample[0:1]
        #sample = sample[1:2]
        return sample

    sampledir = sample_dir()
    samplefile = os.path.join(sampledir, 'NLSA-{}-v1.0.fits'.format(camera))

    info = fitsio.FITS(samplefile)
    nrows = info[1].get_nrows()

    if first is None:
        first = 0
    if last is None:
        last = nrows
        rows = np.arange(first, last)
    else:
        if last >= nrows:
            print('Index last cannot be greater than the number of rows, {} >= {}'.format(last, nrows))
            raise ValueError()
        rows = np.arange(first, last + 1)
    
    sample = Table(info[1].read(rows=rows, upper=True))
    #sample = sample[np.argsort(reff)]
    #sample = sample[sample['REFF'] > 30]
    #sample = sample[:1]

    # Pick 100 random galaxies, uniformly selected in surface brightness.
    if True:
        print('Choosing 100 random galaxies!')
        seed = 1
        npilot = 64
        keep = np.where((sample['RMAG'] < 20) * (sample['SB'] > 18) * (sample['SB'] < 28))[0]
        sample = sample[keep]
        sb = sample['SB'].data

        nbin = 20
        _xbin = np.linspace(sb.min(), sb.max(), nbin)
        idx  = np.digitize(sb, _xbin)

        prob = np.zeros_like(sb)
        for kk in range(nbin):
            ww = idx == kk
            if np.sum(ww) > 1:
                prob[ww] = 1 / np.sum(ww)
        prob /= np.sum(prob)

        rand = np.random.RandomState(seed=1)
        these = rand.choice(len(sample), npilot, p=prob, replace=False)
        srt = np.argsort(sb[these])
        sample = sample[these[srt]]

    if verbose:
        print('Read {} galaxies from {}'.format(len(sample), samplefile), flush=True)
    
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

        html.write('<h1>Nlsa-NSA</h1>\n')
        html.write('<p>\n')
        html.write('<a href="https://github.com/moustakas/LSLGA">Code and documentation</a>\n')
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
            galaxy = onegal['NLSAID']
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
        galaxy = onegal['NLSAID']
        if type(galaxy) is np.bytes_:
            galaxy = galaxy.decode('utf-8')
        plateifu = onegal['PLATEIFU']
        if type(plateifu) is np.bytes_:
            plateifu = plateifu.decode('utf-8')

        width_arcsec = 2 * NLSA_RADIUS
        #width_arcsec = RADIUSFACTOR * onegal['NSA_PETRO_TH50']
        width_kpc = width_arcsec / LSLGA.misc.arcsec2kpc(onegal['Z'])

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

            html.write('<h1>Nlsa ID {}</h1>\n'.format(galaxy))

            html.write('<a href="../{}">Home</a>\n'.format(homehtml))
            html.write('<br />\n')
            html.write('<br />\n')

            # Table of properties
            html.write('<table>\n')
            html.write('<tr>\n')
            html.write('<th>Number</th>\n')
            html.write('<th>Nlsa ID</th>\n')
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

            html.write('<h2>Multiwavelength mosaics</h2>\n')
            html.write("""<p>From left to right: GALEX (FUV/NUV), DESI Legacy Surveys (grz), and unWISE (W1/W2)
            mosaics ({:.2f} arcsec or {:.0f} kpc on a side).</p>\n""".format(width_arcsec, width_kpc))
            #html.write("""<p>From left to right: GALEX (FUV/NUV), DESI Legacy Surveys (grz), and unWISE (W1/W2)
            #mosaic ({0:.0f} kpc on a side).</p>\n""".format(width_kpc))
            html.write('<table width="90%">\n')
            pngfile = '{}-multiwavelength-data.png'.format(galaxy)
            html.write('<tr><td><a href="{0}"><img src="{0}" alt="Missing file {0}" height="auto" width="100%"></a></td></tr>\n'.format(
                pngfile))
            html.write('</table>\n')
            #html.write('<br />\n')
            
            ###########################################################################
            html.write('<h2>Multiwavelength image modeling</h2>\n')
            html.write("""<p>From left to right: data; model image of all sources except the central, resolved galaxy;
            residual image containing just the central galaxy.</p><p>From top to bottom: GALEX (FUV/NUV), DESI Legacy
            Surveys (grz), and unWISE (W1/W2) mosaic ({:.2f} arcsec or {:.0f} kpc on a side).</p>\n""".format(
                width_arcsec, width_kpc))

            html.write('<table width="90%">\n')
            pngfile = '{}-multiwavelength-models.png'.format(galaxy)
            html.write('<tr><td><a href="{0}"><img src="{0}" alt="Missing file {0}" height="auto" width="100%"></a></td></tr>\n'.format(
                pngfile))
            #html.write('<tr><td>Data, Model, Residuals</td></tr>\n')
            html.write('</table>\n')
            html.write('<br />\n')

            ###########################################################################
            
            html.write('<h2>Elliptical Isophote Analysis</h2>\n')

            if False:
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
            html.write('<td width="100%"><a href="{0}"><img src="{0}" alt="Missing file {0}" height="auto" width="100%"></a></td>\n'.format(
                pngfile))
            html.write('<td></td>\n')
            html.write('</tr>\n')
            html.write('</table>\n')

            if False:
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

            if False:
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
            
            html.write('<br />\n')
            html.write('<br />\n')
            html.write('<a href="../{}">Home</a>\n'.format(homehtml))
            html.write('<br />\n')

            html.write('<br /><b><i>Last updated {}</b></i>\n'.format(js))
            html.write('<br />\n')
            html.write('</html></body>\n')
            html.close()

    if makeplots:
        for onegal in sample:
            galaxy = onegal['NLSAID']
            if type(galaxy) is np.bytes_:
                galaxy = galaxy.decode('utf-8')
            galaxydir = os.path.join(analysisdir, galaxy)
            htmlgalaxydir = os.path.join(htmldir, galaxy)

            survey.output_dir = os.path.join(analysisdir, galaxy)
            #survey.ccds = fits_table(os.path.join(survey.output_dir, '{}-ccds.fits'.format(galaxy)))

            # Custom plots
            ellipsefit = read_ellipsefit(galaxy, galaxydir)
            
            cogfile = os.path.join(htmlgalaxydir, '{}-curve-of-growth.png'.format(galaxy))
            if not os.path.isfile(cogfile) or clobber:
                LSLGA.qa.qa_curveofgrowth(ellipsefit, png=cogfile, verbose=verbose)
                
            #pdb.set_trace()
                
            sbprofilefile = os.path.join(htmlgalaxydir, '{}-ellipse-sbprofile.png'.format(galaxy))
            if not os.path.isfile(sbprofilefile) or clobber:
                LSLGA.qa.display_ellipse_sbprofile(ellipsefit, png=sbprofilefile,
                                                   verbose=verbose)
            
            LSLGA.qa.qa_multiwavelength_coadds(galaxy, galaxydir, htmlgalaxydir,
                                               clobber=clobber, verbose=verbose)

            # Plots common to legacyhalos
            #make_plots([onegal], galaxylist=[galaxy], analysisdir=analysisdir,
            #           htmldir=htmldir, clobber=clobber, verbose=verbose,
            #           survey=survey, refband=refband, pixscale=pixscale,
            #           band=band, nproc=nproc, ccdqa=ccdqa, trends=False)

    print('HTML pages written to {}'.format(htmldir))
