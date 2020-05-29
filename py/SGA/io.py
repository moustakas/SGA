"""
SGA.io
======

Code to read and write the various SGA files.

"""
import os, warnings
import pickle, pdb
import numpy as np
import numpy.ma as ma
from glob import glob

import fitsio
from astropy.table import Table, Column, hstack
from astropy.io import fits

def SGA_dir():
    if 'SGA_DIR' not in os.environ:
        print('Required ${SGA_DIR environment variable not set.')
        raise EnvironmentError
    return os.path.abspath(os.getenv('SGA_DIR'))

def analysis_dir():
    adir = os.path.join(SGA_dir(), 'analysis')
    if not os.path.isdir(adir):
        os.makedirs(adir, exist_ok=True)
    return adir

def sample_dir(version=None):
    sdir = os.path.join(SGA_dir(), 'sample')
    if not os.path.isdir(sdir):
        os.makedirs(sdir, exist_ok=True)
    if version:
        sdir = os.path.join(SGA_dir(), 'sample', version)
        if not os.path.isdir(sdir):
            os.makedirs(sdir, exist_ok=True)
    return sdir

def paper1_dir(figures=True):
    pdir = os.path.join(SGA_dir(), 'science', 'paper1')
    if not os.path.ipdir(pdir):
        os.makedirs(pdir, exist_ok=True)
    if figures:
        pdir = os.path.join(pdir, 'figures')
        if not os.path.ipdir(pdir):
            os.makedirs(pdir, exist_ok=True)
    return pdir

def html_dir():
    #if 'NERSC_HOST' in os.environ:
    #    htmldir = '/global/project/projectdirs/cosmo/www/temp/ioannis/SGA'
    #else:
    #    htmldir = os.path.join(SGA_dir(), 'html')

    htmldir = os.path.join(SGA_dir(), 'html')
    if not os.path.isdir(htmldir):
        os.makedirs(htmldir, exist_ok=True)
    return htmldir

def parent_version(version=None):
    """Version of the parent catalog.

    These are the archived versions. For DR9 we reset the counter to start at v3.0!

    #version = 'v1.0' # 18may13
    #version = 'v2.0' # 18nov14
    #version = 'v3.0' # 19sep26
    #version = 'v4.0' # 19dec23
    #version = 'v5.0' # 20jan30 (dr9e)
    #version = 'v6.0' # 20feb25 (DR9-SV)
    version = 'v7.0'  # 20apr18 (DR9)

    """
    if version is None:
        #version = 'v1.0' # 18may13
        #version = 'v2.0' # DR8 (18nov14)
        version = 'v3.0' # DR9
    return version

def get_parentfile(version=None, kd=False):

    if kd:
        suffix = 'kd.fits'
    else:
        suffix = 'fits'
        
    parentfile = os.path.join(sample_dir(version=version), 'SGA-{}.{}'.format(version, suffix))

    return parentfile

def read_parent(columns=None, verbose=False, first=None, last=None,
                version=None, chaos=False):
    """Read the SGA parent catalog.

    """
    if version is None:
        version = parent_version()
    
    parentfile = get_parentfile(version=version)

    if first and last:
        if first > last:
            print('Index first cannot be greater than index last, {} > {}'.format(first, last))
            raise ValueError()
    ext = 1
    info = fitsio.FITS(parentfile)
    nrows = info[ext].get_nrows()

    rows = None
    
    # Read the CHAOS sample.
    if chaos:
        allgals = info[1].read(columns='GALAXY')
        rows = np.hstack( [np.where(np.isin(allgals, chaosgal.encode('utf-8')))[0]
                           for chaosgal in ('NGC0628', 'NGC5194', 'NGC5457', 'NGC3184')] )
        rows = np.sort(rows)
        nrows = len(rows)

        nrows = info[1].get_nrows()

    if first is None:
        first = 0
    if last is None:
        last = nrows
        if rows is None:
            rows = np.arange(first, last)
        else:
            rows = rows[np.arange(first, last)]
    else:
        if last >= nrows:
            print('Index last cannot be greater than the number of rows, {} >= {}'.format(last, nrows))
            raise ValueError()
        if rows is None:
            rows = np.arange(first, last+1)
        else:
            rows = rows[np.arange(first, last+1)]

    parent = Table(info[ext].read(rows=rows, upper=True, columns=columns))
    if verbose:
        if len(rows) == 1:
            print('Read galaxy index {} from {}'.format(first, parentfile))
        else:
            print('Read galaxy indices {} through {} (N={}) from {}'.format(
                first, last, len(parent), parentfile))

    ## Temporary hack to add the data release number, PSF size, and distance.
    #if chaos:
    #    parent.add_column(Column(name='DR', dtype='S3', length=len(parent)))
    #    gal2dr = {'NGC0628': 'DR7', 'NGC5194': 'DR6', 'NGC5457': 'DR6', 'NGC3184': 'DR6'}
    #    for ii, gal in enumerate(np.atleast_1d(parent['GALAXY'])):
    #        if gal in gal2dr.keys():
    #            parent['DR'][ii] = gal2dr[gal]
        
    return parent

def read_desi_tiles(verbose=False):
    """Read the latest DESI tile file.
    
    """
    tilefile = os.path.join(sample_dir(), 'catalogs', 'desi-tiles.fits')
    tiles = Table(fitsio.read(tilefile, ext=1, upper=True))
    tiles = tiles[tiles['IN_DESI'] > 0]
    
    if verbose:
        print('Read {} DESI tiles from {}'.format(len(tiles), tilefile))
    
    return tiles

def read_tycho(magcut=99, verbose=False):
    """Read the Tycho 2 catalog.
    
    """
    tycho2 = os.path.join(sample_dir(), 'catalogs', 'tycho2.kd.fits')
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

def read_hyperleda(verbose=False, allwise=False, version=None):
    """Read the Hyperleda catalog.

    These are the archived versions. For DR9 we reset the counter to start at v3.0!

    if version == 'v1.0':
        hyperfile = 'hyperleda-d25min10-18may13.fits'
    elif version == 'v2.0':
        hyperfile = 'hyperleda-d25min10-18nov14.fits'
    elif version == 'v3.0':
        hyperfile = 'hyperleda-d25min10-18nov14.fits'
    elif version == 'v4.0':
        hyperfile = 'hyperleda-d25min10-18nov14.fits'
    elif version == 'v5.0':
        hyperfile = 'hyperleda-d25min10-18nov14.fits'
    elif version == 'v6.0':
        hyperfile = 'hyperleda-d25min10-18nov14.fits'
    elif version == 'v7.0':
        hyperfile = 'hyperleda-d25min10-18nov14.fits'
    else:
        print('Unknown version!')
        raise ValueError
    
    """
    if version is None:
        version = parent_version()
        
    if version == 'v1.0':
        hyperfile = 'hyperleda-d25min10-18may13.fits'
        ref = 'LEDA-20180513'
    elif version == 'v2.0':
        hyperfile = 'hyperleda-d25min10-18nov14.fits'
        ref = 'LEDA-20181114'
    elif version == 'v3.0':
        hyperfile = 'hyperleda-d25min10-18nov14.fits'
        ref = 'LEDA-20181114'
    else:
        print('Unknown version!')
        raise ValueError

    hyperledafile = os.path.join(sample_dir(), 'hyperleda', hyperfile)
    allwisefile = hyperledafile.replace('.fits', '-allwise.fits')

    leda = Table(fitsio.read(hyperledafile, ext=1, upper=True))
    #leda.add_column(Column(name='GROUPID', dtype='i8', length=len(leda)))
    if verbose:
        print('Read {} objects from {}'.format(len(leda), hyperledafile), flush=True)

    if allwise:
        wise = Table(fitsio.read(allwisefile, ext=1, upper=True))
        if verbose:
            print('Read {} objects from {}'.format(len(wise), allwisefile), flush=True)

        # Merge the tables
        wise.rename_column('RA', 'WISE_RA')
        wise.rename_column('DEC', 'WISE_DEC')

        leda = hstack( (leda, wise) )
        leda.add_column(Column(name='IN_WISE', data=np.zeros(len(leda)).astype(bool)))

        haswise = np.where(wise['CNTR'] != -1)[0]
        #nowise = np.where(wise['CNTR'] == 0)[0]
        #print('unWISE match: {}/{} ({:.2f}%) galaxies.'.format(len(haswise), len(leda)))

        #print('EXT_FLG summary:')
        #for flg in sorted(set(leda['EXT_FLG'][haswise])):
        #    nn = np.sum(flg == leda['EXT_FLG'][haswise])
        #    print('  {}: {}/{} ({:.2f}%)'.format(flg, nn, len(haswise), 100*nn/len(haswise)))
        #print('Need to think this through a bit more; look at:')
        #print('  http://wise2.ipac.caltech.edu/docs/release/allsky/expsup/sec4_4c.html#xsc')
        #leda['INWISE'] = (np.array(['NULL' not in dd for dd in wise['DESIGNATION']]) * 
        #                  np.isfinite(wise['W1SIGM']) * np.isfinite(wise['W2SIGM']) )
        leda['IN_ALLWISE'][haswise] = True

        print('  Identified {}/{} ({:.2f}%) objects with AllWISE photometry.'.format(
            np.sum(leda['IN_ALLWISE']), len(leda), 100*np.sum(leda['IN_ALLWISE'])/len(leda) ))

    # Assign a unique ID and also fix infinite PA and B/A.
    leda.add_column(Column(name='SGA_ID', length=len(leda), dtype='i8'), index=0)
    leda['SGA_ID'] = np.arange(len(leda))
    leda['BYHAND'] = np.zeros(len(leda), bool)
    leda['REF'] = ref
    
    fix = np.isnan(leda['PA'])
    if np.sum(fix) > 0:
        leda['PA'][fix] = 0.0
    fix = np.isnan(leda['BA'])
    if np.sum(fix) > 0:
        leda['BA'][fix] = 1.0
    fix = np.isnan(leda['Z'])
    if np.sum(fix) > 0:
        leda['Z'][fix] = -99.0

    return leda

def read_localgroup_dwarfs():
    """Read the sample generated by bin/SGA-localgroup-dwarfs.

    """
    dwarfsfile = os.path.join(sample_dir(), 'catalogs', 'SGA-dwarfs.fits')
    dwarfs = Table(fitsio.read(dwarfsfile, upper=True))
    print('Read {} Local Group dwarfs from {}'.format(len(dwarfs), dwarfsfile))

    return dwarfs
