"""
LSLGA.io
========

Code to read and write the various LSLGA files.

"""
import os
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

    if dr is not None:
        if ccds:
            parentfile = os.path.join(sample_dir(), 'LSLGA-{}-{}-ccds.{}'.format(
                parent_version(), dr, suffix))
        else:
            parentfile = os.path.join(sample_dir(), 'LSLGA-{}-{}.{}'.format(
                parent_version(), dr, suffix))
    else:
        parentfile = os.path.join(sample_dir(), 'LSLGA-{}.{}'.format(parent_version(), suffix))

    if d25min is not None:
        parentfile = parentfile.replace('.fits', '-d25min{:.2f}.fits'.format(d25min))
    if d25max is not None:
        parentfile = parentfile.replace('.fits', '-d25max{:.2f}.fits'.format(d25max))
        
    return parentfile

def read_parent(columns=None, dr=None, kd=False, ccds=False, d25min=None,
                d25max=None, verbose=False):
    """Read the LSLGA parent catalog.

    """
    parentfile = get_parentfile(dr=dr, kd=kd, ccds=ccds, d25min=d25min, d25max=d25max)
    if kd:
        from astrometry.libkd.spherematch import tree_open
        parent = tree_open(parentfile, 'largegals')
        if verbose:
            print('Read {} galaxies from KD catalog {}'.format(parent.n, parentfile))
    else:
        parent = Table(fitsio.read(parentfile, columns=columns, upper=True))
        if verbose:
            print('Read {} galaxies from {}'.format(len(parent), parentfile))

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

def read_tycho(magcut=12, verbose=False):
    """Read the Tycho 2 catalog.
    
    """
    tycho2 = os.path.join(sample_dir(), 'tycho2.kd.fits')
    tycho = Table(fitsio.read(tycho2, ext=1, upper=True))
    tycho = tycho[np.logical_and(tycho['ISGALAXY'] == 0, tycho['MAG_BT'] <= magcut)]
    if verbose:
        print('Read {} Tycho-2 stars with B<{:.1f}.'.format(len(tycho), magcut), flush=True)
    
    # Radius of influence; see eq. 9 of https://arxiv.org/pdf/1203.6594.pdf
    tycho['RADIUS'] = (0.0802*(tycho['MAG_BT'])**2 - 1.860*tycho['MAG_BT'] + 11.625) / 60 # [degree]
    
    return tycho

def read_hyperleda(verbose=False):
    """Read the Hyperleda catalog.
    
    """
    version = parent_version()
    if version == 'v1.0':
        hyperfile = 'hyperleda-d25min10-18may13.fits'
    elif version == 'v2.0':
        hyperfile = 'hyperleda-d25min10-18nov14.fits'
    else:
        print('Unknown version!')
        raise ValueError
    
    hyperledafile = os.path.join(sample_dir(), hyperfile)
    allwisefile = hyperledafile.replace('.fits', '-allwise.fits')

    leda = Table(fitsio.read(hyperledafile, ext=1, upper=True))
    leda.add_column(Column(name='GROUPID', dtype='i8', length=len(leda)))
    if verbose:
        print('Read {} objects from {}'.format(len(leda), hyperledafile), flush=True)

    allwise = Table(fitsio.read(allwisefile, ext=1, upper=True))
    if verbose:
        print('Read {} objects from {}'.format(len(allwise), allwisefile), flush=True)

    # Merge the tables
    allwise.rename_column('RA', 'WISE_RA')
    allwise.rename_column('DEC', 'WISE_DEC')
    
    leda = hstack( (leda, allwise) )
    print('Need to think this through a bit more; look at:')
    print('  http://wise2.ipac.caltech.edu/docs/release/allsky/expsup/sec4_4c.html#xsc')
    pdb.set_trace()
    leda['INWISE'] = (np.array(['NULL' not in dd for dd in allwise['DESIGNATION']]) * 
                      np.isfinite(allwise['W1SIGM']) * np.isfinite(allwise['W2SIGM']) )
    
    #print('  Identified {} objects with WISE photometry.'.format(np.sum(leda['inwise'])))
    
    return leda
