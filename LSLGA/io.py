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
from astropy.table import Table
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

def read_parent(extname='LSPHOT', upenn=True, isedfit=False, columns=None, verbose=False):
    """Read the various parent catalogs.

    """
    suffix = ''
    if isedfit:
        suffix = '-isedfit'
    elif upenn:
        suffix = '-upenn'

    lsdir = LSLGA_dir()
    catfile = os.path.join(lsdir, 'LSLGA-parent{}.fits'.format(suffix))
    
    cat = Table(fitsio.read(catfile, ext=extname, columns=columns, lower=True))
    if verbose:
        print('Read {} objects from {} [{}]'.format(len(cat), catfile, extname))

    return cat

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
