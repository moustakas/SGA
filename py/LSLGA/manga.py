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

def html_dir():
    #if 'NERSC_HOST' in os.environ:
    #    htmldir = '/global/project/projectdirs/cosmo/www/temp/ioannis/LSLGA'
    #else:
    #    htmldir = os.path.join(LSLGA_dir(), 'html')

    htmldir = os.path.join(LSLGA_dir(), 'html')
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

def read_sample(columns=None, dr=None, ccds=False, verbose=False,
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
