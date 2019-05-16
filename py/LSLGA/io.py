"""
LSLGA.io
========

Code to read and write the various LSLGA files.

"""
import os, warnings
import pickle, pdb
import numpy as np
import numpy.ma as ma
from glob import glob

import fitsio
from astropy.table import Table, Column, hstack
from astropy.io import fits

def LSLGA_dir():
    if 'LSLGA_DIR' not in os.environ:
        print('Required ${LSLGA_DIR environment variable not set.')
        raise EnvironmentError
    return os.path.abspath(os.getenv('LSLGA_DIR'))

def analysis_dir():
    adir = os.path.join(LSLGA_dir(), 'analysis')
    if not os.path.isdir(adir):
        os.makedirs(adir, exist_ok=True)
    return adir

def sample_dir():
    sdir = os.path.join(LSLGA_dir(), 'sample')
    if not os.path.isdir(sdir):
        os.makedirs(sdir, exist_ok=True)
    return sdir

def paper1_dir(figures=True):
    pdir = os.path.join(LSLGA_dir(), 'science', 'paper1')
    if not os.path.ipdir(pdir):
        os.makedirs(pdir, exist_ok=True)
    if figures:
        pdir = os.path.join(pdir, 'figures')
        if not os.path.ipdir(pdir):
            os.makedirs(pdir, exist_ok=True)
    return pdir

def html_dir():
    #if 'NERSC_HOST' in os.environ:
    #    htmldir = '/global/project/projectdirs/cosmo/www/temp/ioannis/LSLGA'
    #else:
    #    htmldir = os.path.join(LSLGA_dir(), 'html')

    htmldir = os.path.join(LSLGA_dir(), 'html')
    if not os.path.isdir(htmldir):
        os.makedirs(htmldir, exist_ok=True)
    return htmldir

def get_galaxy_galaxydir(cat, analysisdir=None, htmldir=None, html=False):
    """Retrieve the galaxy name and the (nested) directory.

    """
    import astropy
    import healpy as hp
    from LSLGA.misc import radec2pix
    
    nside = 8 # keep hard-coded
    
    if analysisdir is None:
        analysisdir = analysis_dir()
    if htmldir is None:
        htmldir = html_dir()

    def get_healpix_subdir(nside, pixnum, analysisdir):
        subdir = os.path.join(str(pixnum // 100), str(pixnum))
        return os.path.abspath(os.path.join(analysisdir, str(nside), subdir))

    if type(cat) is astropy.table.row.Row:
        ngal = 1
        galaxy = [cat['GALAXY']]
        pixnum = [radec2pix(nside, cat['RA'], cat['DEC'])]
    else:
        ngal = len(cat)
        galaxy = np.array([gg.decode('utf-8') for gg in cat['GALAXY'].data])
        pixnum = radec2pix(nside, cat['RA'], cat['DEC']).data

    galaxydir = np.array([os.path.join(get_healpix_subdir(nside, pix, analysisdir), gal)
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

def parent_version(version=None):
    """Version of the parent catalog."""
    if version is None:
        #version = 'v1.0' # 18may13
        version = 'v2.0'  # 18nov14
    return version

def get_parentfile(dr=None, kd=False, ccds=False, d25min=None, d25max=None):
    if kd:
        suffix = 'kd.fits'
    else:
        suffix = 'fits'

    version = parent_version()

    if dr is not None:
        if ccds:
            parentfile = os.path.join(sample_dir(), version, 'LSLGA-{}-{}-ccds.{}'.format(version, dr, suffix))
        else:
            parentfile = os.path.join(sample_dir(), version, 'LSLGA-{}-{}.{}'.format(version, dr, suffix))
    else:
        parentfile = os.path.join(sample_dir(), version, 'LSLGA-{}.{}'.format(version, suffix))

    if d25min is not None:
        parentfile = parentfile.replace('.fits', '-d25min{:.2f}.fits'.format(d25min))
    if d25max is not None:
        parentfile = parentfile.replace('.fits', '-d25max{:.2f}.fits'.format(d25max))
        
    return parentfile

def read_parent(columns=None, dr=None, kd=False, ccds=False, d25min=None,
                d25max=None, verbose=False, first=None, last=None, chaos=False):
    """Read the LSLGA parent catalog.

    """
    parentfile = get_parentfile(dr=dr, kd=kd, ccds=ccds, d25min=d25min, d25max=d25max)

    if kd:
        from astrometry.libkd.spherematch import tree_open
        parent = tree_open(parentfile, 'largegals')
        if verbose:
            print('Read {} galaxies from KD catalog {}'.format(parent.n, parentfile))
    else:
        info = fitsio.FITS(parentfile)

        # Read the CHAOS sample.
        if chaos:
            allgals = info[1].read(columns='GALAXY')
            rows = np.hstack( [np.where(np.isin(allgals, chaosgal.encode('utf-8')))[0]
                               for chaosgal in ('NGC0628', 'NGC5194', 'NGC5457', 'NGC3184')] )
            rows = np.sort(rows)
            nrows = len(rows)
        else:
            nrows = info[1].get_nrows()

        if first is None:
            first = 0
        if last is None:
            last = nrows
        if first == last:
            last = last + 1
            
        if chaos:
            rows = rows[first:last]
        else:
            rows = np.arange(first, last)

        parent = Table(info[1].read(rows=rows))
        if verbose:
            if len(rows) == 1:
                print('Read galaxy index {} from {}'.format(first, parentfile))
            else:
                print('Read galaxy indices {} through {} (N={}) from {}'.format(
                    first, last-1, len(parent), parentfile))

        # Temporary hack to add the data release number, PSF size, and distance.
        if chaos:
            parent.add_column(Column(name='DR', dtype='S3', length=len(parent)))
            gal2dr = {'NGC0628': 'DR7', 'NGC5194': 'DR6', 'NGC5457': 'DR6', 'NGC3184': 'DR6'}
            for ii, gal in enumerate(np.atleast_1d(parent['GALAXY'])):
                if gal in gal2dr.keys():
                    parent['DR'][ii] = gal2dr[gal]
        
    return parent

def read_desi_tiles(verbose=False):
    """Read the latest DESI tile file.
    
    """
    tilefile = os.path.join(sample_dir(), 'desi-tiles.fits')
    tiles = Table(fitsio.read(tilefile, ext=1, upper=True))
    tiles = tiles[tiles['IN_DESI'] > 0]
    
    if verbose:
        print('Read {} DESI tiles from {}'.format(len(tiles), tilefile))
    
    return tiles

def read_tycho(magcut=99, verbose=False):
    """Read the Tycho 2 catalog.
    
    """
    tycho2 = os.path.join(sample_dir(), 'tycho2.kd.fits')
    tycho = Table(fitsio.read(tycho2, ext=1, upper=True))
    tycho = tycho[np.logical_and(tycho['ISGALAXY'] == 0, tycho['MAG_BT'] <= magcut)]
    if verbose:
        print('Read {} Tycho-2 stars with B<{:.1f}.'.format(len(tycho), magcut), flush=True)
    
    # Radius of influence; see eq. 9 of https://arxiv.org/pdf/1203.6594.pdf
    #tycho['RADIUS'] = (0.0802*(tycho['MAG_BT'])**2 - 1.860*tycho['MAG_BT'] + 11.625) / 60 # [degree]

    # From https://github.com/legacysurvey/legacypipe/blob/large-gals-only/py/legacypipe/runbrick.py#L1668
    # Note that the factor of 0.262 has nothing to do with the DECam pixel scale!
    tycho['RADIUS'] = np.minimum(1800., 150. * 2.5**((11. - tycho['MAG_BT']) / 4) ) * 0.262 / 3600

    #import matplotlib.pyplot as plt
    #oldrad = (0.0802*(tycho['MAG_BT'])**2 - 1.860*tycho['MAG_BT'] + 11.625) / 60 # [degree]
    #plt.scatter(tycho['MAG_BT'], oldrad*60, s=1) ; plt.scatter(tycho['MAG_BT'], tycho['RADIUS']*60, s=1) ; plt.show()
    #pdb.set_trace()
    
    return tycho

def read_hyperleda(verbose=False, version=None):
    """Read the Hyperleda catalog.
    
    """
    if version is None:
        version = parent_version()
        
    if version == 'v1.0':
        hyperfile = 'hyperleda-d25min10-18may13.fits'
    elif version == 'v2.0':
        hyperfile = 'hyperleda-d25min10-18nov14.fits'
    else:
        print('Unknown version!')
        raise ValueError
    
    hyperledafile = os.path.join(sample_dir(), version, hyperfile)
    allwisefile = hyperledafile.replace('.fits', '-allwise.fits')

    leda = Table(fitsio.read(hyperledafile, ext=1, upper=True))
    #leda.add_column(Column(name='GROUPID', dtype='i8', length=len(leda)))
    if verbose:
        print('Read {} objects from {}'.format(len(leda), hyperledafile), flush=True)

    allwise = Table(fitsio.read(allwisefile, ext=1, upper=True))
    if verbose:
        print('Read {} objects from {}'.format(len(allwise), allwisefile), flush=True)

    # Merge the tables
    allwise.rename_column('RA', 'WISE_RA')
    allwise.rename_column('DEC', 'WISE_DEC')
    
    leda = hstack( (leda, allwise) )
    leda.add_column(Column(name='IN_ALLWISE', data=np.zeros(len(leda)).astype(bool)))

    haswise = np.where(allwise['CNTR'] != -1)[0]
    #nowise = np.where(allwise['CNTR'] == 0)[0]
    #print('unWISE match: {}/{} ({:.2f}%) galaxies.'.format(len(haswise), len(leda)))
    
    #print('EXT_FLG summary:')
    #for flg in sorted(set(leda['EXT_FLG'][haswise])):
    #    nn = np.sum(flg == leda['EXT_FLG'][haswise])
    #    print('  {}: {}/{} ({:.2f}%)'.format(flg, nn, len(haswise), 100*nn/len(haswise)))
    #print('Need to think this through a bit more; look at:')
    #print('  http://wise2.ipac.caltech.edu/docs/release/allsky/expsup/sec4_4c.html#xsc')
    #leda['INWISE'] = (np.array(['NULL' not in dd for dd in allwise['DESIGNATION']]) * 
    #                  np.isfinite(allwise['W1SIGM']) * np.isfinite(allwise['W2SIGM']) )
    leda['IN_ALLWISE'][haswise] = True
    
    print('  Identified {}/{} ({:.2f}%) objects with AllWISE photometry.'.format(
        np.sum(leda['IN_ALLWISE']), len(leda), 100*np.sum(leda['IN_ALLWISE'])/len(leda) ))
    
    return leda

def read_multiband(galaxy, galaxydir, band=('g', 'r', 'z'), refband='r',
                   pixscale=0.262, galex_pixscale=1.5, unwise_pixscale=2.75,
                   maskfactor=2.0):
    """Read the multi-band images, construct the residual image, and then create a
    masked array from the corresponding inverse variances image.  Finally,
    convert to surface brightness by dividing by the pixel area.

    """
    from scipy.stats import sigmaclip
    from scipy.ndimage.morphology import binary_dilation

    # Dictionary mapping between filter and filename coded up in coadds.py,
    # galex.py, and unwise.py (see the LSLGA product, too).
    filt2imfile = {
        'g':   ['custom-image', 'custom-model-nocentral', 'invvar'],
        'r':   ['custom-image', 'custom-model-nocentral', 'invvar'],
        'z':   ['custom-image', 'custom-model-nocentral', 'invvar'],
        'FUV': ['image', 'model-nocentral'],
        'NUV': ['image', 'model-nocentral'],
        'W1':  ['image', 'model-nocentral'],
        'W2':  ['image', 'model-nocentral'],
        'W3':  ['image', 'model-nocentral'],
        'W4':  ['image', 'model-nocentral']}
        
    filt2pixscale =  {
        'g':   pixscale,
        'r':   pixscale,
        'z':   pixscale,
        'FUV': galex_pixscale,
        'NUV': galex_pixscale,
        'W1':  unwise_pixscale,
        'W2':  unwise_pixscale,
        'W3':  unwise_pixscale,
        'W4':  unwise_pixscale}

    found_data = True
    for filt in band:
        for ii, imtype in enumerate(filt2imfile[filt]):
            for suffix in ('.fz', ''):
                imfile = os.path.join(galaxydir, '{}-{}-{}.fits{}'.format(galaxy, imtype, filt, suffix))
                if os.path.isfile(imfile):
                    filt2imfile[filt][ii] = imfile
                    break
            if not os.path.isfile(imfile):
                print('File {} not found.'.format(imfile))
                found_data = False

    #tractorfile = os.path.join(galaxydir, '{}-tractor.fits'.format(galaxy))
    #if os.path.isfile(tractorfile):
    #    cat = Table(fitsio.read(tractorfile, upper=True))
    #    print('Read {} sources from {}'.format(len(cat), tractorfile))
    #else:
    #    print('Missing Tractor catalog {}'.format(tractorfile))
    #    found_data = False

    data = dict()
    if not found_data:
        return data

    for filt in band:
        image = fitsio.read(filt2imfile[filt][0])
        model = fitsio.read(filt2imfile[filt][1])

        if len(filt2imfile[filt]) == 3:
            invvar = fitsio.read(filt2imfile[filt][2])

            # Mask pixels with ivar<=0. Also build an object mask from the model
            # image, to handle systematic residuals.
            mask = (invvar <= 0) # True-->bad, False-->good
            
            #if np.sum(mask) > 0:
            #    invvar[mask] = 1e-3
            #snr = model * np.sqrt(invvar)
            #mask = np.logical_or( mask, (snr > 1) )

            #sig1 = 1.0 / np.sqrt(np.median(invvar))
            #mask = np.logical_or( mask, (image - model) > (3 * sig1) )

        else:
            mask = np.zeros_like(image).astype(bool)

        # Can give a divide-by-zero error for, e.g., GALEX imaging
        #with np.errstate(divide='ignore', invalid='ignore'):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with np.errstate(all='ignore'):
                model_clipped, _, _ = sigmaclip(model, low=4.0, high=4.0)

        #print(filt, 1-len(model_clipped)/image.size)
        #if filt == 'W1':
        #    pdb.set_trace()
            
        if len(model_clipped) > 0:
            mask = np.logical_or( mask, model > 3 * np.std(model_clipped) )
            #model_clipped = model
        
        mask = binary_dilation(mask, iterations=1) # True-->bad

        thispixscale = filt2pixscale[filt]
        data[filt] = (image - model) / thispixscale**2 # [nanomaggies/arcsec**2]
        
        #data['{}_mask'.format(filt)] = mask # True->bad
        data['{}_masked'.format(filt)] = ma.masked_array(data[filt], mask)
        ma.set_fill_value(data['{}_masked'.format(filt)], 0)

    data['band'] = band
    data['refband'] = refband
    data['pixscale'] = pixscale

    if 'NUV' in band:
        data['galex_pixscale'] = galex_pixscale
    if 'W1' in band:
        data['unwise_pixscale'] = unwise_pixscale

    return data

def write_ellipsefit(galaxy, galaxydir, ellipsefit, verbose=False, noellipsefit=True):
    """Pickle a dictionary of photutils.isophote.isophote.IsophoteList objects (see,
    e.g., ellipse.fit_multiband).

    """
    if noellipsefit:
        suffix = '-fixed'
    else:
        suffix = ''
        
    ellipsefitfile = os.path.join(galaxydir, '{}-ellipsefit{}.p'.format(galaxy, suffix))
    if verbose:
        print('Writing {}'.format(ellipsefitfile))
    with open(ellipsefitfile, 'wb') as ell:
        pickle.dump(ellipsefit, ell)

def read_ellipsefit(galaxy, galaxydir, verbose=True, noellipsefit=True):
    """Read the output of write_ellipsefit.

    """
    if noellipsefit:
        suffix = '-fixed'
    else:
        suffix = ''

    ellipsefitfile = os.path.join(galaxydir, '{}-ellipsefit{}.p'.format(galaxy, suffix))
    try:
        with open(ellipsefitfile, 'rb') as ell:
            ellipsefit = pickle.load(ell)
    except:
        #raise IOError
        if verbose:
            print('File {} not found!'.format(ellipsefitfile))
        ellipsefit = dict()

    return ellipsefit

