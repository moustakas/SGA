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

def parent_version():
    """Version of the parent catalog."""
    version = 'v1.0'
    return version

def get_parentfile(dr=None, kd=False):

    if kd:
        suffix = 'kd.fits'
    else:
        suffix = 'fits'

    if dr is not None:
        parentfile = os.path.join(sample_dir(), 'LSLGA-{}-{}.{}'.format(parent_version(), dr, suffix))
    else:
        parentfile = os.path.join(sample_dir(), 'LSLGA-{}.{}'.format(parent_version(), suffix))

    return parentfile

def read_parent(columns=None, dr=None, kd=False, verbose=False):
    """Read the LSLGA parent catalog.

    """
    parentfile = get_parentfile(dr=dr, kd=kd)
    if kd:
        from astrometry.libkd.spherematch import tree_open
        parent = tree_open(parentfile, 'largegals')
        
    else:
        parent = Table(fitsio.read(parentfile, ext=extname, columns=columns, lower=True))
        
    if verbose:
        print('Read {} objects from {} [{}]'.format(len(parent), parentfile))

    return parent

def read_sample(first=None, last=None, dr='dr6-dr7', sfhgrid=1,
                isedfit_lsphot=False, isedfit_sdssphot=False,
                isedfit_lhphot=False, satellites=False,
                kcorr=False, verbose=False):
    """Read the sample.

    """
    if satellites:
        prefix = 'satellites'
    else:
        prefix = 'centrals'

    if isedfit_lsphot:
        samplefile = os.path.join(sample_dir(), '{}-sfhgrid{:02d}-lsphot-{}.fits'.format(prefix, sfhgrid, dr))
    elif isedfit_sdssphot:
        samplefile = os.path.join(sample_dir(), '{}-sfhgrid{:02d}-sdssphot-dr14.fits'.format(prefix, sfhgrid))
    elif isedfit_lhphot:
        samplefile = os.path.join(sample_dir(), '{}-sfhgrid{:02d}-lhphot.fits'.format(prefix, sfhgrid))
    else:
        samplefile = os.path.join(sample_dir(), 'LSLGA-{}-{}.fits'.format(prefix, dr))
        
    if not os.path.isfile(samplefile):
        print('File {} not found.'.format(samplefile))
        return None

    if first and last:
        if first > last:
            print('Index first cannot be greater than index last, {} > {}'.format(first, last))
            raise ValueError()

    info = fitsio.FITS(samplefile)
    nrows = info[1].get_nrows()

    if first is None:
        first = 0
    if last is None:
        last = nrows
    if first == last:
        last = last + 1

    rows = np.arange(first, last)

    if kcorr:
        ext = 2
    else:
        ext = 1

    sample = Table(info[1].read(rows=rows, ext=ext))
    if verbose:
        if len(rows) == 1:
            print('Read galaxy index {} from {}'.format(first, samplefile))
        else:
            print('Read galaxy indices {} through {} (N={}) from {}'.format(
                first, last-1, len(sample), samplefile))
            
    return sample

def read_tycho(magcut=12, verbose=True):
    """Read the Tycho 2 catalog.
    
    """
    tycho2 = os.path.join(sample_dir(), 'tycho2.kd.fits')
    tycho = astropy.table.Table(fitsio.read(tycho2, ext=1, lower=True))
    tycho = tycho[np.logical_and(tycho['isgalaxy'] == 0, tycho['mag_bt'] <= magcut)]
    if verbose:
        print('Read {} Tycho-2 stars with B<{:.1f}.'.format(len(tycho), magcut), flush=True)
    
    # Radius of influence; see eq. 9 of https://arxiv.org/pdf/1203.6594.pdf
    tycho['radius'] = (0.0802*(tycho['mag_bt'])**2 - 1.860*tycho['mag_bt'] + 11.625) / 60 # [degree]
    
    return tycho    

def read_hyperleda(verbose=True):
    """Read the Hyperleda catalog.
    
    """
    hyperledafile = os.path.join(sample_dir(), 'hyperleda-d25min10-18may13.fits')
    allwisefile = hyperledafile.replace('.fits', '-allwise.fits')

    leda = Table(fitsio.read(hyperledafile, ext=1))
    leda.add_column(Column(name='groupid', dtype='i8', length=len(leda)))
    if verbose:
        print('Read {} objects from {}'.format(len(leda), hyperledafile), flush=True)

    allwise = Table(fitsio.read(allwisefile, ext=1, lower=True))
    if verbose:
        print('Read {} objects from {}'.format(len(allwise), allwisefile), flush=True)

    # Merge the tables
    allwise.rename_column('ra', 'wise_ra')
    allwise.rename_column('dec', 'wise_dec')
    
    leda = hstack( (leda, allwise) )
    leda['inwise'] = (np.array(['NULL' not in dd for dd in allwise['designation']]) * 
                      np.isfinite(allwise['w1sigm']) * np.isfinite(allwise['w2sigm']) )
    
    #print('  Identified {} objects with WISE photometry.'.format(np.sum(leda['inwise'])))
    
    return leda
