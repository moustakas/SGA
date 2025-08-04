"""
SGA.SGA
=======

Code to build and analyze the SGA sample.

"""
import os, time, pdb
import fitsio
import numpy as np
import numpy.ma as ma
from astropy.table import Table, vstack

from SGA.logger import log

print('!!!!!!!!!!! Update REFCAT')
REFCAT = 'LG'
#REFCAT = 'L4'
RACOLUMN = 'GROUP_RA'   # 'RA'
DECCOLUMN = 'GROUP_DEC' # 'DEC'
DIAMCOLUMN = 'GROUP_DIAMETER' # 'DIAM'
ZCOLUMN = 'Z'
REFIDCOLUMN = 'SGAID'

FITBITS = dict(
    ignore = 2**0,    # no special behavior (e.g., resolved dwarf galaxy)
    forcegaia = 2**1, # only fit Gaia point sources (and any SGA galaxies), e.g., LMC
    forcepsf = 2**2,  # force PSF for source detection and photometry within the SGA mask
)
SAMPLEBITS = dict(
    LVD = 2**0,       # LVD / local dwarfs
)

#SGAMASKBITS = dict(
#    brightstar = 2**0, # BRIGHT, MEDIUM, or CLUSTER MASKBITS
#    gaiastar = 2**1,   # Gaia (type=PSF) stars
#    galaxy = 2**2,     # galaxy (extended) sources
#    reference = 2**3,  # SGA (reference) sources
#                       # bit 4 - intentionally blank
#                       # bit 5 - intentionally blank
#    optband1 = 2**6,   # optical band 1 (e.g., g)
#    optband2 = 2**7,   # optical band 2 (e.g., r)
#    optband3 = 2**8,   # optical band 3 (e.g., i)
#    optband4 = 2**9,   # optical band 4 (e.g., z)
#                       # bit 10 - intentionally blank
#                       # bit 11 - intentionally blank
#    irband1 = 2**12,   # IR band 1 (e.g., W1)
#    irband2 = 2**13,   # IR band 2 (e.g., W2)
#    irband3 = 2**14,   # IR band 3 (e.g., W3)
#    irband4 = 2**15,   # IR band 4 (e.g., W4)
#
#    uvband1 = 2**16,   # UV band 1 (e.g., FUV)
#    uvband2 = 2**17,   # UV band 2 (e.g., NUV)
#)

OPTMASKBITS = dict(
    brightstar = 2**0, # BRIGHT, MEDIUM, or CLUSTER MASKBITS
    gaiastar = 2**1,   # Gaia (type=PSF) stars
    galaxy = 2**2,     # galaxy (extended, non-reference) sources
    reference = 2**3,  # SGA (reference) sources
    g = 2**4,          #
    r = 2**5,          #
    i = 2**6,          #
    z = 2**7,          #
)

GALEXMASKBITS = dict(
    brightstar = 2**0, # BRIGHT, MEDIUM, or CLUSTER MASKBITS
    gaiastar = 2**1,   # Gaia (type=PSF) stars
    galaxy = 2**2,     # galaxy (extended, non-reference) sources
    reference = 2**3,  # SGA (reference) sources
    fuv = 2**4,        #
    nuv = 2**5,        #
)

UNWISEMASKBITS = dict(
    brightstar = 2**0, # BRIGHT, MEDIUM, or CLUSTER MASKBITS
    gaiastar = 2**1,   # Gaia (type=PSF) stars
    galaxy = 2**2,     # galaxy (extended, non-reference) sources
    reference = 2**3,  # SGA (reference) sources
    w1 = 2**4,         #
    w2 = 2**5,         #
    w3 = 2**6,         #
    w4 = 2**7,         #
)


VEGA2AB = {'W1': 2.699, 'W2': 3.339, 'W3': 5.174, 'W4': 6.620}

SBTHRESH = [22, 22.5, 23, 23.5, 24, 24.5, 25, 25.5, 26] # surface brightness thresholds
APERTURES = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 3.0] # multiples of MAJORAXIS


def SGA_version():
    version = 'v1.0'
    return version


def sga_dir():
    if 'SGA_DIR' not in os.environ:
        msg = 'Required ${SGA_DIR} environment variable not set.'
        log.critical(msg)
        raise EnvironmentError(msg)
    ldir = os.path.abspath(os.getenv('SGA_DIR'))
    if not os.path.isdir(ldir):
        os.makedirs(ldir, exist_ok=True)
    return ldir


def sga_data_dir():
    if 'SGA_DATA_DIR' not in os.environ:
        msg = 'Required ${SGA_DATA_DIR} environment variable not set.'
        log.critical(msg)
        raise EnvironmentError(msg)
    ldir = os.path.abspath(os.getenv('SGA_DATA_DIR'))
    if not os.path.isdir(ldir):
        os.makedirs(ldir, exist_ok=True)
    return ldir


def sga_html_dir():
    if 'SGA_HTML_DIR' not in os.environ:
        msg = 'Required ${SGA_HTML_DIR} environment variable not set.'
        log.critical(msg)
        raise EnvironmentError(msg)
    ldir = os.path.abspath(os.getenv('SGA_HTML_DIR'))
    if not os.path.isdir(ldir):
        os.makedirs(ldir, exist_ok=True)
    return ldir


def get_raslice(ra):
    if np.isscalar(ra):
        return f'{int(ra):03d}'
    else:
        return np.array([f'{int(onera):03d}' for onera in ra])


def sga2025_name(ra, dec, unixsafe=False):
    # simple wrapper on radec_to_name with precision=3
    from SGA.io import radec_to_name
    return radec_to_name(ra, dec, prefix='SGA2025', precision=3,
                         unixsafe=unixsafe)


def get_galaxy_galaxydir(sample, region='dr11-south', datadir=None,
                         htmldir=None, html=False):
    """Retrieve the galaxy name and the (nested) directory.

    """
    if datadir is None:
        datadir = sga_data_dir()
    if htmldir is None:
        htmldir = sga_html_dir()
    dataregiondir = os.path.join(datadir, region)
    htmlregiondir = os.path.join(htmldir, region)

    if sample is not None:
        # Handle groups.
        if 'GROUP_NAME' in sample.colnames:
            galcolumn = 'GROUP_NAME'
            racolumn = 'GROUP_RA'
        else:
            galcolumn = 'SGANAME'
            racolumn = 'RA'

        objs = np.atleast_1d(sample[galcolumn])
        ras = np.atleast_1d(sample[racolumn])

    objdirs, htmlobjdirs = [], []
    for obj, ra in zip(objs, ras):
        objdirs.append(os.path.join(dataregiondir, get_raslice(ra), obj))
        if html:
            htmlobjdirs.append(os.path.join(htmlregiondir, get_raslice(ra), obj))
    objdirs = np.array(objdirs)
    if html:
        htmlobjdirs = np.array(htmlobjdirs)

    if objdirs.size == 1:
        objs = objs.item()
        objdirs = objdirs.item()
        if html:
            htmlobjdirs = htmlobjdirs.item()

    if html:
        return objs, objdirs, htmlobjdirs
    else:
        return objs, objdirs


def missing_files(sample=None, bricks=None, region='dr11-south',
                  coadds=False, ellipse=False, htmlplots=False, htmlindex=False,
                  build_catalog=False, clobber=False, clobber_overwrite=None,
                  verbose=False, datadir=None, htmldir=None, size=1, mp=1):
    """Figure out which files are missing and still need to be processed.

    """
    from glob import glob
    import multiprocessing
    import astropy
    from SGA.mpi import weighted_partition
    from SGA.io import _missing_files_one

    if sample is None and bricks is None:
        msg = 'Must provide either sample or bricks.'
        raise IOError(msg)

    if sample is not None:
        if type(sample) is astropy.table.row.Row:
            msg = 'sample must be a Table not a Row'
            raise ValueError(msg)
        indices = np.arange(len(sample))
    elif bricks is not None:
        if type(bricks) is astropy.table.row.Row:
            msg = 'bricks must be a Table not a Row'
            raise ValueError(msg)
        indices = np.arange(len(bricks))

    dependson, dependsondir = None, None
    if htmlplots is False and htmlindex is False:
        if verbose:
            t0 = time.time()
            log.debug('Getting galaxy names and directories...')
        galaxy, galaxydir = get_galaxy_galaxydir(sample, region=region,
                                                 datadir=datadir,
                                                 htmldir=htmldir)
        if verbose:
            log.debug(f'...took {time.time() - t0:.3f} sec')

    if coadds:
        suffix = 'coadds'
        filesuffix = '-coadds.isdone'
    elif ellipse:
        suffix = 'ellipse'
        filesuffix = '-ellipse.isdone'
        dependson = '-coadds.isdone'
    elif build_catalog:
        suffix = 'build-catalog'
        filesuffix = '-SGA.isdone'
        dependson = '-ellipse.isdone'
    elif htmlplots:
        suffix = 'html'
        filesuffix = '-montage.png'
        dependson = '-image.jpg'
        galaxy, dependsondir, galaxydir = get_galaxy_galaxydir(
            sample, datadir=datadir, htmldir=htmldir, region=region,
            html=True)
    elif htmlindex:
        suffix = 'htmlindex'
        filesuffix = '-montage.png'
        galaxy, _, galaxydir = get_galaxy_galaxydir(
            sample, datadir=datadir, htmldir=htmldir,
            region=region, html=True)
    else:
        msg = 'Need at least one keyword argument.'
        log.critical(msg)
        raise ValueError(msg)

    # Make clobber=False for build_catalog and htmlindex because we're not
    # making the files here, we're just looking for them. The argument
    # args.clobber gets used downstream.
    if htmlindex or build_catalog:
        clobber = False

    if clobber_overwrite is not None:
        clobber = clobber_overwrite

    missargs = []
    for igal, (gal, gdir) in enumerate(zip(np.atleast_1d(galaxy),
                                           np.atleast_1d(galaxydir))):
        checkfile = os.path.join(gdir, f'{gal}{filesuffix}')
        #print(checkfile)
        if dependson:
            if dependsondir:
                missargs.append([checkfile, os.path.join(np.atleast_1d(dependsondir)[igal],
                                                         f'{gal}{dependson}'), clobber])
            else:
                missargs.append([checkfile, os.path.join(
                    gdir, f'{gal}{dependson}'), clobber])
        else:
            missargs.append([checkfile, None, clobber])

    if verbose:
        t0 = time.time()
        log.debug('Finding missing files...')
    if mp > 1:
        with multiprocessing.Pool(mp) as P:
            todo = np.array(P.map(_missing_files_one, missargs))
    else:
        todo = np.array([_missing_files_one(_missargs)
                         for _missargs in missargs])

    if verbose:
        log.debug(f'...took {(time.time() - t0)/60.:.3f} min')

    itodo = np.where(todo == 'todo')[0]
    idone = np.where(todo == 'done')[0]
    ifail = np.where(todo == 'fail')[0]

    if len(ifail) > 0:
        fail_indices = [indices[ifail]]
    else:
        fail_indices = [np.array([])]

    if len(idone) > 0:
        done_indices = [indices[idone]]
    else:
        done_indices = [np.array([])]

    if len(itodo) > 0:
        todo_indices = np.array_split(indices[itodo], size)
        #_todo_indices = indices[itodo]
        #if sample is not None:
        #    weight = np.atleast_1d(sample[DIAMCOLUMN])[_todo_indices]
        #    todo_indices = weighted_partition(weight, size)
        #else:
        #    # unweighted
        #    todo_indices = np.array_split(_todo_indices, size)
    else:
        todo_indices = [np.array([])]

    return suffix, todo_indices, done_indices, fail_indices


def read_sample(first=None, last=None, galaxylist=None, verbose=False, columns=None,
                lvd=False, final_sample=False, region='dr11-south', d25min=0., d25max=100.0):
    """Read/generate the parent SGA catalog.

    d25min,d25max in arcmin

    """
    import fitsio
    from SGA.coadds import REGIONBITS
    from SGA.parent import parent_version

    if first and last:
        if first > last:
            msg = f'Index first cannot be greater than index last, {first} > {last}'
            log.critical(msg)
            raise ValueError(msg)

    ext = 1

    if final_sample:
        version = SGA_version()
        samplefile = os.path.join(sga_dir(), '2025', f'SGA2025-ellipse-{version}.fits')
    else:
        version = parent_version()
        if lvd:
            samplefile = os.path.join(sga_dir(), 'parent', f'SGA2025-parent-lvd-{version}.fits')
        else:
            samplefile = os.path.join(sga_dir(), 'parent', f'SGA2025-parent-{version}.fits')

    if not os.path.isfile(samplefile):
        msg = f'Sample file {samplefile} not found.'
        log.critical(msg)
        raise IOError(msg)

    if final_sample:
        cols = ['GROUP_DIAMETER', 'GROUP_PRIMARY', 'SGA_ID', 'PREBURNED']
        info = fitsio.read(samplefile, columns=cols)

        rows = np.where(
            (info['GROUP_DIAMETER'] > d25min) *
            (info['GROUP_DIAMETER'] < d25max) *
            info['GROUP_PRIMARY'] *
            info['PREBURNED'] *
            (info['SGA_ID'] > -1))[0]
    else:
        cols = ['GROUP_DIAMETER', 'GROUP_PRIMARY']
        #cols = ['GROUP_NAME', 'GROUP_RA', 'GROUP_DEC', 'GROUP_DIAMETER', 'GROUP_MULT',
        #        'GROUP_PRIMARY', 'GROUP_ID', 'SGAID', 'RA', 'DEC', 'BRICKNAME']
        info = fitsio.read(samplefile, columns=cols)
        rows = np.where(
            (info['GROUP_DIAMETER'] > d25min) *
            (info['GROUP_DIAMETER'] < d25max) *
            info['GROUP_PRIMARY'])[0]

    nallrows = len(info)
    nrows = len(rows)

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
            msg = f'Index last cannot be greater than the number of rows, {last} >= {nrows}'
            log.critical(msg)
            raise ValueError(msg)
        if rows is None:
            rows = np.arange(first, last+1)
        else:
            rows = rows[np.arange(first, last+1)]
            if len(rows) == 1:
                log.info(f'Selecting index {first} (N=1)')
            else:
                log.info(f'Selecting indices {first} through {last} (N={len(rows):,d})')

    fullsample = Table(fitsio.read(samplefile, upper=True))
    #fullsample.add_column(np.arange(nallrows), name='INDEX', index=0)
    sample = fullsample[rows]

    #sample = Table(info[ext].read(rows=rows, upper=True, columns=columns))
    log.info(f'Read {len(sample):,d}/{len(fullsample):,d} GROUP_PRIMARY objects from {samplefile}')

    # select objects in this region
    I = sample['REGION'] & REGIONBITS[region] != 0
    log.info(f'Selecting {np.sum(I):,d}/{len(sample):,d} objects in ' + \
             f'region={region}')
    sample = sample[I]

    if galaxylist is not None:
        log.debug('Selecting specific galaxies.')
        I = np.isin(sample['GROUP_NAME'], galaxylist)
        if np.count_nonzero(I) == 0:
            log.warning('No matching galaxies using column GROUP_NAME!')
            I = np.isin(sample['SGANAME'], galaxylist)
            if np.count_nonzero(I) == 0:
                log.warning('No matching galaxies using column SGANAME!')
                I = np.isin(sample['OBJNAME'], galaxylist)
                if np.count_nonzero(I) == 0:
                    log.warning('No matching galaxies using column OBJNAME!')
                    return Table(), Table()
            return sample[I], fullsample
    else:
        return sample, fullsample


def _build_catalog_one(args):
    """Wrapper function for the multiprocessing."""
    return build_catalog_one(*args)


def build_catalog_one(galaxy, galaxydir, fullsample, REMCOLS,
                      refcat='R1', verbose=False):
    """Gather the ellipse-fitting results for a single group."""
    import fitsio
    from SGA.io import read_ellipsefit

    tractor, parent, ellipse = [], [], []

    tractorfile = os.path.join(galaxydir, f'{galaxy}-custom-tractor.fits')
    if not os.path.isfile(tractorfile):
        log.warning(f'Missing Tractor catalog {tractorfile}')
        return None, None, None #tractor, parent, ellipse
        #return tractor, parent, ellipse

    for igal, onegal in enumerate(fullsample):
        #print(f'Working on {onegal["GALAXY"]}')
        refid = onegal[REFIDCOLUMN]

        ellipsefile = os.path.join(galaxydir, f'{galaxy}-custom-ellipse-{refid}.fits')
        if not os.path.isfile(ellipsefile):
            log.warning(f'Missing ellipse file {ellipsefile}')
            return None, None, None #tractor, parent, ellipse

        _ellipse = read_ellipsefit(galaxy, galaxydir, galaxy_id=str(refid), asTable=True,
                                  filesuffix='custom', verbose=True)
        # fix the data model
        #_ellipse = _datarelease_table(_ellipse)
        for col in REMCOLS:
            #print(f'Removing {col}')
            _ellipse.remove_column(col)
        _ellipse['ELLIPSEBIT'] = np.zeros(1, dtype=np.int32) # we don't want -1 here

        _tractor = Table(fitsio.read(tractorfile, upper=True))
        match = np.where((_tractor['REF_CAT'] == refcat) * (_tractor['REF_ID'] == refid))[0]
        if len(match) != 1:
            raise ValueError('Problem here!')

        ellipse.append(_ellipse)
        tractor.append(_tractor[match])
        parent.append(onegal)

    tractor = vstack(tractor, join_type='exact', metadata_conflicts='silent')
    parent = vstack(parent, join_type='exact', metadata_conflicts='silent')
    ellipse = vstack(ellipse, join_type='exact', metadata_conflicts='silent')

    return tractor, parent, ellipse


def build_catalog(sample, fullsample, bands, galex=True, unwise=True,
                  mp=1, refcat='R1', verbose=False, clobber=False):
    import time
    import multiprocessing
    from astropy.io import fits

    from SGA.ellipse import FAILCOLS

    version = SGA_version()

    outfile = os.path.join(sga_dir(), f'SGA2025-{version}-legacyphot.fits')
    if os.path.isfile(outfile) and not clobber:
        log.warning(f'Use --clobber to overwrite existing catalog {outfile}')
        return

    galaxy, galaxydir = get_galaxy_galaxydir(sample)

    # figure out which ndim>1 columns to drop
    optbands = bands.copy()
    if galex:
        bands += ['fuv', 'nuv']
    if unwise:
        bands += ['w1', 'w2', 'w3', 'w4']
    REMCOLS = ['BANDS', 'REFPIXSCALE', 'SUCCESS', 'FITGEOMETRY', 'LARGESHIFT',
               'MAXSMA', 'MAJORAXIS', 'EPS_MOMENT', 'INTEGRMODE',
               'INPUT_ELLIPSE', 'SCLIP', 'NCLIP',
               'REFBAND', 'REFBAND_WIDTH', 'REFBAND_HEIGHT']
    for band in optbands:
        for col in ['PSFSIZE', 'PSFDEPTH']:
            REMCOLS += [f'{col}_{band.upper()}']
    for band in bands:
        for col in FAILCOLS:
            REMCOLS += [f'{col.upper()}_{band.upper()}']
        for col in ['SMA', 'FLUX', 'FLUX_IVAR']:
            REMCOLS += [f'COG_{col}_{band.upper()}']
    #print(REMCOLS)

    # build the mp list
    buildargs = []
    for gal, gdir, onegal in zip(galaxy, galaxydir, sample):
        _fullsample = fullsample[fullsample['GROUP_ID'] == onegal['GROUP_ID']]
        buildargs.append((gal, gdir, _fullsample, REMCOLS, refcat, verbose))

    t0 = time.time()
    if mp > 1:
        with multiprocessing.Pool(mp) as P:
            results = P.map(_build_catalog_one, buildargs)
    else:
        results = [build_catalog_one(*_buildargs)
                   for _buildargs in buildargs]

    results = list(zip(*results))
    tractor1 = list(filter(None, results[0]))
    parent1 = list(filter(None, results[1]))
    ellipse1 = list(filter(None, results[2]))

    #for col in ellipse1[0].colnames:
    #    if ellipse1[0][col].ndim > 1:
    #        print(col)

    log.info('Doing an outer join on Tractor because some columns are missing from some catalogs:')
    log.info("  ['mw_transmission_nuv' 'mw_transmission_fuv' 'ngood_g' 'ngood_r' 'ngood_z']")
    tractor = vstack(tractor1, metadata_conflicts='silent')

    # exact join
    parent = vstack(parent1, join_type='exact', metadata_conflicts='silent')
    ellipse = vstack(ellipse1, join_type='exact', metadata_conflicts='silent')
    log.info(f'Merging {len(tractor):,d} galaxies took {(time.time()-t0)/60.0:.2f} min.')

    if len(tractor) == 0:
        log.warning('Something went wrong and no galaxies were fitted.')
        return
    assert(len(tractor) == len(parent))
    assert(np.all(tractor['REF_ID'] == parent[REFIDCOLUMN]))

    # write out
    hdu_primary = fits.PrimaryHDU()
    hdu_parent = fits.convenience.table_to_hdu(parent)
    hdu_parent.header['EXTNAME'] = 'PARENT'

    hdu_ellipse = fits.convenience.table_to_hdu(ellipse)
    hdu_ellipse.header['EXTNAME'] = 'ELLIPSE'

    hdu_tractor = fits.convenience.table_to_hdu(tractor)
    hdu_tractor.header['EXTNAME'] = 'TRACTOR'

    hx = fits.HDUList([hdu_primary, hdu_parent, hdu_ellipse, hdu_tractor])
    hx.writeto(outfile, overwrite=True, checksum=True)

    log.info(f'Wrote {len(parent):,d} galaxies to {outfile}')


def _get_psfsize_and_depth(data, tractor, bands, pixscale, incenter=False):
    """Support function for read_multiband. Compute the average PSF
    size (in arcsec) and depth (in 5-sigma AB mags) in each bandpass
    based on the Tractor catalog.

    """
    # Optionally choose sources in the center of the field.
    H = np.max(tractor.bx) - np.min(tractor.bx)
    W = np.max(tractor.by) - np.min(tractor.by)
    if incenter:
        dH = 0.1 * H
        these = np.where((tractor.bx >= int(H / 2 - dH)) * (tractor.bx <= int(H / 2 + dH)) *
                         (tractor.by >= int(H / 2 - dH)) * (tractor.by <= int(H / 2 + dH)))[0]
    else:
        these = np.arange(len(tractor))

    # Get the average PSF size and depth in each bandpass.
    for filt in bands:
        psfsizecol = f'psfsize_{filt}'
        psfdepthcol = f'psfdepth_{filt}'
        if psfsizecol in tractor.columns():
            good = np.where(tractor.get(psfsizecol)[these] > 0)[0]
            if len(good) == 0:
                log.warning(f'  No good measurements of the PSF size in band {filt}!')
                data[f'psfsigma_{filt}'] = np.float32(0.0)
                data[f'psfsize_{filt}'] = np.float32(0.0)
            else:
                # Get the PSF size and image depth.
                psfsize = tractor.get(psfsizecol)[these][good]   # [FWHM, arcsec]
                psfsigma = psfsize / np.sqrt(8 * np.log(2)) / pixscale # [sigma, pixels]

                data[f'psfsigma_{filt}'] = np.median(psfsigma).astype('f4')
                data[f'psfsize_{filt}'] = np.median(psfsize).astype('f4')

        if psfsizecol in tractor.columns():
            good = np.where(tractor.get(psfdepthcol)[these] > 0)[0]
            if len(good) == 0:
                log.warning(f'  No good measurements of the PSF depth in band {filt}!')
                data[f'psfdepth_{filt}'] = np.float32(0.0)
            else:
                psfdepth = tractor.get(psfdepthcol)[these][good] # [AB mag, 5-sigma]
                data[f'psfdepth_{filt}'] = (22.5-2.5*np.log10(1./np.sqrt(np.median(psfdepth)))).astype('f4')

        # clean up
        tractor.delete_column(psfsizecol)
        tractor.delete_column(psfdepthcol)

    return data


def _read_image_data(data, filt2imfile, verbose=False):
    """Helper function for the project-specific read_multiband method.

    Read the multi-band images and inverse variance images and pack them into a
    dictionary. Also create an initial pixel-level mask and handle images with
    different pixel scales (e.g., GALEX and WISE images).

    """
    from astropy.stats import sigma_clipped_stats
    from scipy.ndimage.morphology import binary_dilation
    from scipy.ndimage.filters import gaussian_filter
    from skimage.transform import resize

    from tractor.psf import PixelizedPSF
    from tractor.tractortime import TAITime
    from astrometry.util.util import Tan
    from legacypipe.survey import LegacySurveyWcs, ConstantFitsWcs

    refband = data['refband']
    galex_refband = data['galex_refband']
    unwise_refband = data['unwise_refband']

    fit_bands = data['fit_bands']
    fit_optical_bands = data['fit_optical_bands']
    unwise_bands = data['unwise_bands']

    # Read the per-filter images and generate an optical and UV/IR
    # mask.
    for filt in fit_bands:
        # Read the data and initialize the mask with the inverse
        # variance image.
        if verbose:
            log.info(f'Reading {filt2imfile[filt]["image"]}')
            log.info(f'Reading {filt2imfile[filt]["model"]}')
            log.info(f'Reading {filt2imfile[filt]["invvar"]}')
        hdr = fitsio.read_header(filt2imfile[filt]['image'], ext=1)
        image = fitsio.read(filt2imfile[filt]['image'])
        invvar = fitsio.read(filt2imfile[filt]['invvar'])
        model = fitsio.read(filt2imfile[filt]['model'])

        if np.any(invvar < 0):
            log.warning(f'Found {np.sum(invvar<0):,d} negative pixels in the ' + \
                        f'{filt}-band inverse variance image!')

        sz = image.shape
        if filt == refband or filt == galex_refband or filt == unwise_refband:
            if filt == refband:
                data['width'] = sz[1]
                data['height'] = sz[0]
                #data['header'] = hdr

            wcs = Tan(hdr)
            if 'MJD_MEAN' in hdr:
                mjd_tai = hdr['MJD_MEAN'] # [TAI]
                wcs = LegacySurveyWcs(wcs, TAITime(None, mjd=mjd_tai))
            else:
                wcs = ConstantFitsWcs(wcs)

            if filt == refband:
                data['opt_wcs'] = wcs
            elif filt == galex_refband:
                data['galex_wcs'] = wcs
            elif filt == unwise_refband:
                data['unwise_wcs'] = wcs

        # convert WISE images from Vega nanomaggies to AB nanomaggies
        # https://www.legacysurvey.org/dr9/description/#photometry
        if filt in unwise_bands:
            image *= 10.**(-0.4 * VEGA2AB[filt])
            invvar /= (10.**(-0.4 * VEGA2AB[filt]))**2.
            model *= 10.**(-0.4 * VEGA2AB[filt])

        if verbose:
            log.info(f'Reading {filt2imfile[filt]["psf"]}')
        psfimg = fitsio.read(filt2imfile[filt]['psf'])
        psfimg /= psfimg.sum()
        data[f'{filt}_psf'] = PixelizedPSF(psfimg)

        # Generate a basic per-band mask, including allmask for the
        # optical bands and wisemask for the unwise bands. In the
        # optical bands, also mask XX% of the border.
        mask = invvar <= 0 # True-->bad

        if filt in fit_optical_bands:
            mask = np.logical_or(mask, data[f'allmask_{filt}'])
            del data[f'allmask_{filt}']

        # add wisemask for W1/W2, if present, but we have to resize
        if data['wisemask'] is not None and filt in unwise_bands[:2]:
            _wisemask = resize(data['wisemask'], mask.shape, mode='edge',
                               anti_aliasing=False) > 0
            mask = np.logical_or(mask, _wisemask)

        mask = binary_dilation(mask, iterations=2)

        #if filt in fit_optical_bands:
        #    edge = int(0.02*sz[0])
        #    mask[:edge, :] = True
        #    mask[:, :edge] = True
        #    mask[:, sz[0]-edge:] = True
        #    mask[sz[0]-edge:, :] = True

        # set invvar of masked pixels to zero and then get the robust
        # sigma from the masked residual image
        image[mask] = 0.
        invvar[mask] = 0.
        data[filt] = image # [nanomaggies]
        data[f'{filt}_invvar'] = invvar # [1/nanomaggies**2]
        data[f'{filt}_mask'] = mask
        #print(filt, np.sum(mask))

        if filt in fit_optical_bands:
            resid = gaussian_filter(image - model, 2.)
            _, _, sig = sigma_clipped_stats(resid[~mask], sigma=2.5)
            data[f'{filt}_sigma'] = sig

    if 'wisemask' in data:
        del data['wisemask']

    ## resize brightstarmask to get the UV/IR brightstarmask
    #if unwise_refband in fit_bands or galex_refband in fit_bands:
    #    uvir_brightstarmask = resize(data['brightstarmask'], data[refband].shape,
    #                                 mode='edge', anti_aliasing=False) > 0
    #    data['uvir_brightstarmask'] = uvir_brightstarmask

    return data


def _tractor2mge(sample, tractor, pixscale, xgrid, ygrid,
                 mindiam=10., factor=2.5):
    """Convert a Tractor catalog entry to an MGE object.

    mindiam - minimum major-axis length in arcsec

    factor - approximate multiplicative factor between half-light
             diameter from Tractor and D(25); see
             https://github.com/moustakas/SGA/blob/main/science/SGA2025/SGA2020-diameters.ipynb

    """
    from SGA.geometry import ellipse_mask, get_tractor_ellipse

    class MGEgalaxy(object):
        pass

    # if the geometry is fixed, fix it!
    if sample['FIXGEOMETRY']:
        diam = sample['DIAM_INIT'] # [arcsec]
        pa = sample['PA_INIT']
        ba = sample['BA_INIT']
        bx = sample['BX_INIT']
        by = sample['BY_INIT']
    else:
        bx = sample['BX']
        by = sample['BY']
        # r50-->D(50)-->D(26)
        diam = sample['SHAPE_R'] * 2. * factor # [arcsec]
        _, ba, pa = get_tractor_ellipse(
            diam, sample['SHAPE_E1'],
            sample['SHAPE_E2'])

    if diam < mindiam:
        diam = mindiam # minimum size [arcsec]
    majoraxis = diam / pixscale # [pixels]
    minoraxis = ba * majoraxis

    mge = MGEgalaxy()
    mge.xmed = by # NB - by!
    mge.ymed = bx
    mge.xpeak = by
    mge.ypeak = bx
    mge.eps = 1. - ba
    mge.pa = pa
    mge.theta = (270. - pa) % 180.
    mge.majoraxis = majoraxis

    # object pixels are True
    objmask = ellipse_mask(mge.xmed, mge.ymed,
                           mge.majoraxis, minoraxis,
                           np.radians(mge.theta-90.),
                           xgrid, ygrid)

    return mge, objmask


def _build_multiband_mask(data, tractor, maxshift=0., niter=1, qaplot=True):
    """Wrapper to mask out all sources except the galaxy we want to ellipse-fit.

    """
    from skimage.transform import resize
    from SGA.geometry import in_ellipse_mask
    from SGA.find_galaxy import find_galaxy
    from SGA.dust import SFDMap, mwdust_transmission

    def make_sourcemask(srcs, wcs, band, psf, sigma, nsigma=2.5):
        """Build a model image and threshold mask from a table of
        Tractor sources; also optionally subtract that model from an
        input image.

        """
        from scipy.ndimage.morphology import binary_dilation
        from SGA.coadds import srcs2image

        model = srcs2image(srcs, wcs, band=band.lower(), pixelized_psf=psf)
        mask = model > nsigma*sigma # True=significant flux
        mask = binary_dilation(mask*1, iterations=2) > 0

        return mask, model


    def get_geometry(pixscale, pixfactor=1., table=None, tractor=None, mge=None):
        """Extract elliptical geometry from either an astropy Table
        (sample), a tractor catalog, or an mge object.

        """
        if table is not None:
            bx, by = table['BX_INIT'], table['BY_INIT']
            semia = table['DIAM_INIT'] / 2. / pixscale # [pixels]
            semib = semia * table['BA_INIT']           # [pixels]
            pa = table['PA_INIT']
        elif tractor is not None:
            from SGA.geometry import get_tractor_ellipse
            (bx, by) = tractor.bx, tractor.by
            semia = tractor.shape_r / pixscale # [pixels]
            _, ba, pa = get_tractor_ellipse(semia, tractor.shape_e1, tractor.shape_e2)
            semib = semia * ba
        elif mge is not None:
            bx = mge.ymed # NB - ymed!
            by = mge.xmed # NB - xmed!
            semia = mge.majoraxis # [pixels]
            semib = semia * (1. - mge.eps)
            pa = mge.pa

        bx *= pixfactor
        by *= pixfactor

        return (bx, by, semia, semib, pa)


    fit_bands = data['fit_bands']
    fit_optical_bands = data['fit_optical_bands']
    galex_bands = data['galex_bands']
    unwise_bands = data['unwise_bands']
    optindx = np.where(np.isin(fit_bands, fit_optical_bands))[0]

    refband = data['refband']
    galex_refband = data['galex_refband']
    unwise_refband = data['unwise_refband']

    filt2pixscale = data['filt2pixscale']
    refpixscale = data['refpixscale']

    sz = data[refband].shape
    width = sz[0]
    assert(width == sz[1])
    xgrid, ygrid = np.meshgrid(np.arange(width),
                               np.arange(width),
                               indexing='xy')
    ygrid_flip = width - ygrid

    galex_sz = data[galex_refband].shape
    unwise_sz = data[unwise_refband].shape

    opt_wcs = data['opt_wcs']
    galex_wcs = data['galex_wcs']
    unwise_wcs = data['unwise_wcs']

    sample = data['sample']
    samplesrcs = data['samplesrcs']
    nsample = len(sample)

    # 20-sigma
    minlevel = max([data[f'psfdepth_{filt}'] for filt in fit_optical_bands])
    minlevel = 20.*10.**(-0.4*(minlevel)) # AB mag-->nanomaggies

    Ipsf = ((tractor.type == 'PSF') * (tractor.type != 'DUP') *
            (tractor.ref_cat == 'GE')) # (tractor.ref_cat != REFCAT)
    Igal = ((tractor.type != 'PSF') * (tractor.type != 'DUP') *
            (tractor.ref_cat != REFCAT))
    psfsrcs = tractor[Ipsf]
    allgalsrcs = tractor[Igal]

    # Initialize the *original* images arrays.
    opt_images = np.zeros((len(fit_optical_bands), *sz), 'f4')
    opt_mask_perband = np.stack([data[f'{filt}_mask'] for filt in fit_optical_bands])
    opt_weight = np.stack([data[f'{filt}_invvar'] for filt in fit_optical_bands])

    # Bright-star mask.
    opt_brightstarmask = data['brightstarmask']

    # Subtract Gaia stars from all optical images and generate the
    # threshold gaiamask.
    opt_gaiamask = np.zeros(sz, bool)
    for iband, filt in enumerate(fit_optical_bands):
        if len(psfsrcs) > 0:
            msk, model = make_sourcemask(
                psfsrcs, opt_wcs, filt, data[f'{filt}_psf'],
                data[f'{filt}_sigma'])
            opt_images[iband, :, :] = data[filt] - model
            opt_gaiamask = np.logical_or(opt_gaiamask, msk)
        else:
            opt_images[iband, :, :] = data[filt]

    geo_initial = np.zeros((nsample, 5)) # [bx,by,semia,semib,pa]
    geo_final = np.zeros_like(geo_initial)

    opt_maskbits = np.zeros((nsample, *sz), np.int32)
    opt_images_final = np.zeros((nsample, len(fit_optical_bands), *sz), 'f4')
    opt_models = np.zeros((nsample, len(fit_optical_bands), *sz), 'f4')

    for iobj, (obj, objsrc) in enumerate(zip(sample, samplesrcs)):
        log.info('Determining the geometry for galaxy ' + \
                 f'{iobj+1}/{nsample}.')

        # Find all reference sources (not dropped by Tractor) except
        # the one we're working on.
        refsrcs = samplesrcs[:iobj] + samplesrcs[iobj+1:]
        refsamples = sample[np.delete(np.arange(nsample), iobj)]

        # Build the reference mask: for each *other* SGA
        # source(s), if any, subtract the model from the optical
        # images.
        opt_refmask = np.zeros(sz, bool)
        opt_images_obj = opt_images.copy() # reset the data
        for refsrc, refsample in zip(refsrcs, refsamples):
            (bx, by, semia, semib, pa) = \
                get_geometry(refpixscale, table=refsample)
            opt_refmask1 = in_ellipse_mask(bx, width-by, semia,
                                           semib, pa, xgrid, ygrid_flip)
            opt_refmask = np.logical_or(opt_refmask, opt_refmask1)

            for iband, filt in enumerate(fit_optical_bands):
                _, model = make_sourcemask(
                    refsrc, opt_wcs, filt, data[f'{filt}_psf'],
                    data[f'{filt}_sigma'])
                opt_images_obj[iband, :, :] = opt_images_obj[iband, :, :] - model
                opt_models[iobj, iband, :, :] += model

        print('Set the blended bit!')

        # Initial geometry and elliptical mask.
        geo_init  = get_geometry(refpixscale, table=obj)
        geo_initial[iobj, :] = geo_init
        (bx, by, semia, semib, pa) = geo_init

        # Next, iteratively update the source geometry.
        for iiter in range(niter):
            log.info(f'  Iteration {iiter+1}/{niter}')

            print(iiter, bx, by, semia, semib, pa)
            inellipse = in_ellipse_mask(bx, width-by, semia, semib, pa,
                                        xgrid, ygrid_flip)

            # Build a galaxy mask (logical_or over all optical
            # bandpasses) from all extended sources outside the
            # (current) elliptical mask (but do not subtract the
            # models).
            opt_galmask = np.zeros(sz, bool)
            if len(allgalsrcs) > 0:
                I = in_ellipse_mask(bx, width-by, semia, semib, pa,
                                    allgalsrcs.bx, width-allgalsrcs.by)
                if np.sum(~I) > 0:
                    galsrcs = allgalsrcs[~I]
                    for iband, filt in enumerate(fit_optical_bands):
                        msk, _ = make_sourcemask(
                            galsrcs, opt_wcs, filt, data[f'{filt}_psf'],
                            data[f'{filt}_sigma'])
                        opt_galmask = np.logical_or(opt_galmask, msk)
                else:
                    galsrcs = None
            else:
                galsrcs = None

            # Combine opt_brightstarmask, opt_gaiamask, opt_refmask,
            # and opt_galmask for this objects with the per-band
            # optical masks.
            brightstarmask = opt_brightstarmask.copy()
            refmask = opt_refmask.copy()
            galmask = opt_galmask.copy()

            # Zero out bright-star and reference pixels within the
            # ellipse mask of the current object. NB: no need to zero
            # out galmask pixels because sources are, by construction,
            # outside of the object ellipse.
            brightstarmask[inellipse] = False
            refmask[inellipse] = False
            galmask[inellipse] = False
            objmask = np.logical_or.reduce((brightstarmask, refmask, opt_gaiamask, galmask))

            #for label, msk in zip(['bright-star', 'reference', 'gaia', 'galaxy'],
            #                      [brightstarmask, refmask, opt_gaiamask, opt_galmask]):
            #    print('  ', label, np.sum(msk)/width**2)

            opt_masks_obj = np.zeros((len(fit_optical_bands), *sz), bool)
            for iband, filt in enumerate(fit_optical_bands):
                opt_masks_obj[iband, :, :] = np.logical_or(objmask, opt_mask_perband[iband, :, :])
                print(iobj, iiter, np.sum(opt_masks_obj))

            # Update the geometry from the masked, coadded optical image.
            wimg = np.sum(opt_weight * np.logical_not(opt_masks_obj) * opt_images_obj, axis=0)

            # Use XX% of the object flux as the level in find_galaxy.
            level = 0.1 * abs(obj['FLUX'])
            if level < minlevel:
                level = minlevel

            #import matplotlib.pyplot as plt
            #plt.clf()
            mge = find_galaxy(wimg, nblob=1, binning=1)#, plot=True)
            #plt.savefig('ioannis/tmp/junk.png')

            print('Check for a large centroid shift!')

            geo_iter = get_geometry(refpixscale, mge=mge)
            (bx, by, semia, semib, pa) = geo_iter
            #geo_obj_iter.append(geo_iter)

        geo_final[iobj, :] = geo_iter # last iteration

        # pack up the images and combine the individual masks into a
        # bitmask
        opt_images_final[iobj, :, :, :] = opt_images_obj

        opt_maskbits_obj = np.zeros(sz, np.int32)
        opt_maskbits_obj[brightstarmask] += OPTMASKBITS['brightstar']
        opt_maskbits_obj[refmask] += OPTMASKBITS['reference']
        opt_maskbits_obj[opt_gaiamask] += OPTMASKBITS['gaiastar']
        opt_maskbits_obj[opt_galmask] += OPTMASKBITS['galaxy']
        for iband, filt in enumerate(fit_optical_bands):
            opt_maskbits_obj[opt_mask_perband[iband, :, :]] += OPTMASKBITS[filt]
            opt_maskbits[iobj, :, :] = opt_maskbits_obj

    #galex_images = np.zeros((len(galex_bands), *galex_sz), 'f4')
    #unwise_images = np.zeros((len(unwise_bands), *unwise_sz), 'f4')
    #
    #galex_weight = np.stack([data[f'{filt}_invvar'] for filt in galex_bands])
    #unwise_weight = np.stack([data[f'{filt}_invvar'] for filt in unwise_bands])
    #
    #galex_starmask = resize(opt_starmask, galex_sz, mode='edge',
    #                        anti_aliasing=False) > 0
    #unwise_starmask = resize(opt_starmask, unwise_sz, mode='edge',
    #                         anti_aliasing=False) > 0
    #
    #for bands, wcs, imgs in zip([galex_bands, unwise_bands],
    #                            [galex_wcs, unwise_wcs],
    #                            [galex_images, unwise_images]):
    #    for iband, filt in enumerate(bands):
    #        if len(psfsrcs) > 0:
    #            # subtract stars but do not update starmask
    #            _, img = make_sourcemask(
    #                psfsrcs, wcs, filt, data[f'{filt}_psf'],
    #                1., image=data[filt])
    #            #print(filt, np.sum(img > 0))
    #            imgs[iband, :, :] = img
    #        else:
    #            imgs[iband, :, :] = data[filt]


    # build a QA figure
    if qaplot:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        from matplotlib.patches import Patch
        from astropy.visualization import simple_norm
        from SGA.qa import overplot_ellipse

        qafile = os.path.join('/global/cfs/cdirs/desi/users/ioannis/tmp',
                              f'qa-ellipsemask-{data["galaxy"]}.png')

        alpha = 0.6
        orange = (0.9, 0.6, 0.0, alpha)   # golden-orange
        blue   = (0.0, 0.45, 0.7, alpha)  # muted blue
        purple = (0.8, 0.6, 0.7, alpha)   # soft violet
        magenta = (0.85, 0.2, 0.5, alpha) # vibrant rose

        ncol = 3
        nrow = 1 + nsample
        inches_per_panel = 3.
        fig, ax = plt.subplots(nrow, ncol,
                               figsize=(inches_per_panel*ncol,
                                        inches_per_panel*nrow),
                               gridspec_kw={'wspace': 0.02, 'hspace': 0.02},
                               constrained_layout=True)

        #cmap = plt.cm.cividis
        #cmap = plt.cm.twilight
        cmap = plt.cm.magma
        #cmap = plt.cm.hot
        #cmap = plt.cm.inferno
        cmap.set_bad('white')

        setcolors1 = plt.rcParams['axes.prop_cycle'].by_key()['color']
        setcolors2 = [
            '#e41a1c',  # strong red
            '#377eb8',  # saturated blue
            '#4daf4a',  # green
            '#984ea3',  # purple
            '#ff7f00',  # orange
            '#ffff33',  # bright yellow (only on dark parts)
            '#a65628',  # brown
            '#f781bf',  # pink
        ]

        # coadded optical, IR, and UV images and initial geometry
        imgbands = [fit_optical_bands, unwise_bands, galex_bands]
        labels = [''.join(fit_optical_bands), 'unWISE', 'GALEX']
        for iax, (xx, bands, ref, label) in enumerate(zip(
                ax[0, :], imgbands, [refband, unwise_refband, galex_refband], labels)):
            wimgs = np.stack([data[filt] for filt in bands])
            wivars = np.stack([data[f'{filt}_invvar'] for filt in bands])
            wimg = np.sum(wivars * wimgs, axis=0)

            norm = simple_norm(wimg, stretch='asinh', percent=99.5, asinh_a=0.1)
            xx.imshow(wimg, origin='lower', cmap=cmap, interpolation='none', norm=norm)
            xx.set_xlim(0, wimg.shape[0])
            xx.set_ylim(0, wimg.shape[1])

            # initial ellipse geometry
            pixscale = filt2pixscale[ref]
            pixfactor = filt2pixscale[refband] / pixscale
            for iobj, obj in enumerate(sample):
                (bx, by, semia, semib, pa) = geo_initial[iobj, :]
                overplot_ellipse(2*semia*pixfactor*pixscale, semib/semia, pa,
                                 bx*pixfactor, by*pixfactor, pixscale=pixscale,
                                 ax=xx, color=setcolors1[iobj], linestyle='-',
                                 linewidth=2, draw_majorminor_axes=True,
                                 jpeg=False, label=obj["SGAID"])

            xx.text(0.03, 0.97, label, transform=xx.transAxes,
                    ha='left', va='top', color='white',
                    linespacing=1.5, fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='k', alpha=0.5))

            if iax == 0:
                xx.legend(loc='lower left', fontsize=7, ncol=2,
                          fancybox=True, framealpha=0.5)
            del wimgs, wivars, wimg

        # one row per object
        for iobj, obj in enumerate(sample):
            # unpack the mask bitmask
            brightstarmask = opt_maskbits[iobj, :, :] & OPTMASKBITS['brightstar'] != 0
            refmask = opt_maskbits[iobj, :, :] & OPTMASKBITS['reference'] != 0
            gaiamask = opt_maskbits[iobj, :, :] & OPTMASKBITS['gaiastar'] != 0
            galmask = opt_maskbits[iobj, :, :] & OPTMASKBITS['galaxy'] != 0
            objmask = np.logical_or.reduce((brightstarmask, refmask, gaiamask, galmask))

            opt_masks_obj = np.zeros((len(fit_optical_bands), *sz), bool)
            for iband, filt in enumerate(fit_optical_bands):
                opt_masks_obj[iband, :, :] = np.logical_or(objmask, opt_mask_perband[iband, :, :])
                print(iobj, np.sum(opt_masks_obj))

            wimg = np.sum(opt_weight * np.logical_not(opt_masks_obj) * opt_images_final[iobj, :, :], axis=0)
            wimg[wimg == 0.] = np.nan
            #(bx, by, _, _, _) = geo_final[iobj, :]
            norm = simple_norm(wimg, stretch='asinh', percent=99.5, asinh_a=0.1)
            ax[1+iobj, 0].imshow(wimg, cmap=cmap, origin='lower', interpolation='none', norm=norm)
            #ax[1+iobj, 0].imshow(np.log(wimg.clip(wimg[int(bx), int(by)]/1e4)),
            #                     cmap=cmap, origin='lower', interpolation='none')
            #ax[1+iobj, 0].imshow(mge.mask, origin='lower', cmap='binary',
            #                     interpolation='none', alpha=0.3)

            wmodel = np.sum(opt_models[iobj, :, :, :], axis=0)
            norm = simple_norm(wmodel, stretch='asinh', percent=99.5, asinh_a=0.1)
            ax[1+iobj, 1].imshow(wmodel, cmap=cmap, origin='lower', interpolation='none', norm=norm)

            # masks
            leg = []
            for msk, col, label in zip([brightstarmask, gaiamask, galmask, refmask],
                                       [orange, blue, purple, magenta],
                                       ['Bright Stars', 'Gaia Stars', 'Galaxies', 'Other SGA']):
                rgba = np.zeros((*msk.shape, 4))
                rgba[msk] = col
                ax[1+iobj, 2].imshow(rgba, origin='lower')
                leg.append(Patch(facecolor=col, edgecolor='none', alpha=0.6, label=label))
            if iobj == 0:
                ax[1+iobj, 2].legend(handles=leg, loc='lower right', fontsize=7)

            for col in range(3):
                # initial geometry
                (bx, by, semia, semib, pa) = geo_initial[iobj, :]
                overplot_ellipse(2*semia*refpixscale, semib/semia, pa, bx, by,
                                 pixscale=refpixscale, ax=ax[1+iobj, col], color=setcolors2[0],
                                 linestyle='-', linewidth=2, draw_majorminor_axes=True,
                                 jpeg=False, label='Initial')

                # final geometry
                (bx, by, semia, semib, pa) = geo_final[iobj, :]
                overplot_ellipse(2*semia*refpixscale, semib/semia, pa, bx, by,
                                 pixscale=refpixscale, ax=ax[1+iobj, col],
                                 color=setcolors2[1], linestyle='--', linewidth=2,
                                 draw_majorminor_axes=True, jpeg=False, label='Final')
                ax[1+iobj, col].set_xlim(0, width)
                ax[1+iobj, col].set_ylim(0, width)
                ax[1+iobj, col].margins(0)

            ax[1+iobj, 0].text(0.03, 0.97, obj["SGAID"], transform=ax[1+iobj, 0].transAxes,
                               ha='left', va='top', color='white',
                               linespacing=1.5, fontsize=10,
                               bbox=dict(boxstyle='round', facecolor='k', alpha=0.5))

            if iobj == 0:
                ax[1+iobj, 1].text(0.03, 0.97, f'{"".join(fit_optical_bands)} model(s)',
                                   transform=ax[1+iobj, 1].transAxes, ha='left', va='top',
                                   color='white', linespacing=1.5, fontsize=10,
                                   bbox=dict(boxstyle='round', facecolor='k', alpha=0.5))
                ax[1+iobj, 2].text(0.03, 0.97, f'{"".join(fit_optical_bands)} masks',
                                   transform=ax[1+iobj, 2].transAxes, ha='left', va='top',
                                   color='white', linespacing=1.5, fontsize=10,
                                   bbox=dict(boxstyle='round', facecolor='k', alpha=0.5))

            ax[1+iobj, 0].legend(loc='lower left', fontsize=7, fancybox=True,
                                 framealpha=0.5)

        #for xx in ax[1+iobj, :].ravel():
        #    xx.set_xlim(0, wimg.shape[0])
        #    xx.set_ylim(0, wimg.shape[1])

        for xx in ax.ravel():
            xx.margins(0)
            xx.set_xticks([])
            xx.set_yticks([])

        fig.suptitle(data['galaxy'].replace('_', ' ').replace(' GROUP', ' Group'))
        fig.savefig(qafile)
        log.info(f'Wrote {qafile}')

        pdb.set_trace()


#            #xx = np.random.uniform(0, dims[0], 1200)
#            #yy = np.random.uniform(0, dims[0], 1200)
#            #I = in_ellipse_mask(
#            #    obj['BX_INIT'], obj['BY_INIT'], obj['DIAM_INIT']/2./refpixscale,
#            #    obj['BA_INIT']*obj['DIAM_INIT']/2./refpixscale,
#            #    180.-obj['PA_INIT'], xx, yy)
#            #ax[0, 0].scatter(xx[I], yy[I], s=5, color='green')
#            #ax[0, 0].scatter(xx[~I], yy[~I], s=5, color='purple')
#
#            nearcentral = np.array([centralmask[int(by), int(bx)]
#                                    for by, bx in zip(tractor.by, tractor.bx)])
#
#            # Find PSFs; if we "forced PSF" (inside the SGA source,
#            # e.g., a resolved dwarf like Phoenix) then we only want
#            # to subtract sources *outside* the elliptical mask.
#            if obj['FORCEPSF']:
#                I = Ipsf * np.logical_not(nearcentral)
#            else:
#                I = Ipsf
#            if np.any(I):
#                psfsrcs = tractor[I]
#
#            # Find extended sources *outside* the elliptical
#            # mask. (There's no way to know at this point which
#            # sources inside the elliptical mask are resolved
#            # structure vs an independent object---FIXME!)
#            I = Igal * np.logical_not(nearcentral)
#            if np.any(I):
#                galsrcs = tractor[I]
#
#            # If one or more *other* reference sources are inside the
#            # mask, set the BLENDED bit. Note, we always reset the bit
#            # to False because the mask can change, and we use
#            # REFOBJ[BX,BY] in case a source is dropped by Tractor (in
#            # which case BX,BY are set to BX_INIT,BY_INIT).
#            sample['BLENDED'][iobj] = False # reset to False!
#            if refobj:
#                inmask = [centralmask[int(by), int(bx)]
#                          for by, bx in zip(refobj['BY'], refobj['BX'])]
#                if np.any(inmask):
#                    sample['BLENDED'][iobj] = True
#
#            # Iterating on each optical image, subtract and
#            # threshold-mask the stars and threshold-mask extended
#            # sources. Also subtract and elliptical-mask other
#            # reference sources.
#            for iband, filt in enumerate(fit_optical_bands):
#                opt_images[iband, :, :] = data[filt] # initialize
#                psf = data[f'{filt}_psf']
#                sigma = data[f'{filt}_sigma']
#
#                # Subtract PSFs and threshold mask.
#                if psfsrcs:
#                    msk, img = make_sourcemask(psfsrcs, opt_wcs, filt, psf, sigma,
#                                               image=opt_images[iband, :, :])
#                    opt_psfmasks[iband, :, :] = msk
#                    opt_images[iband, :, :] = img
#
#                # Threshold-mask (but do not subtract) extended
#                # sources outside the mask.
#                if galsrcs:
#                    msk, _ = make_sourcemask(galsrcs, opt_wcs, filt, psf, sigma)
#                    opt_galmasks[iband, :, :] = msk
#
#                # Subtract reference sources (whether in or out of the
#                # mask). Do not threshold-mask but do apply its
#                # elliptical geometric mask.
#                if refsrcs:
#                    _, img = make_sourcemask(refsrcs, opt_wcs, filt, psf, sigma,
#                                             image=opt_images[iband, :, :])
#                    opt_images[iband, :, :] = img
#                if refobj:
#                    for refobj1 in refobj:
#                        print('Need to use mge here!!')
#                        refmask = in_ellipse_mask(
#                            refobj1['BX_INIT'], refobj1['BY_INIT'], refobj1['DIAM_INIT']/2.,
#                            refobj1['BA_INIT']*refobj1['DIAM_INIT']/2.,
#                            refobj1['PA_INIT'], xgrid, ygrid)
#                        opt_galmasks[iband, :, :] = np.logical_or(opt_galmasks[iband, :, :], refmask)
#
#            # Determine the geometry from the ivar-weighted, coadded,
#            # optical image.
#            wmasks = np.logical_not(np.logical_or.reduce((opt_masks, opt_psfmasks, opt_galmasks))) # True-->good
#            wimg = np.sum(wmasks * opt_weight * opt_images, axis=0)
#            wmask = np.sum(wmasks, axis=0) > 0
#            #wivar = np.sum(opt_weight, axis=0)
#
#            mge = find_galaxy(wimg, nblob=1, binning=1, quiet=False)#, plot=True)
#            centralmask = in_ellipse_mask(mge.xmed, mge.ymed,
#                                       mge.majoraxis,
#                                       mge.majoraxis * (1. - mge.eps),
#                                       mge.pa, xgrid, ygrid)
#
#        if qaplot:
#            # plot various classes of sources on the first image
#            if psfsrcs:
#                for xx in ax[0, :]:
#                    pass
#                    #xx.scatter(psfsrcs.bx, psfsrcs.by, marker='s',
#                    #           s=5, color='red')
#            if galsrcs:
#                for xx in ax[0, :]:
#                    pass
#                    #xx.scatter(galsrcs.bx, galsrcs.by, marker='o',
#                    #           s=5, color='cyan')
#
#            # now the data
#            norm = simple_norm(wimg, stretch='asinh', percent=99.5, asinh_a=0.1)
#            wimg[~wmask] = np.nan
#            ax[1+iobj, 0].imshow(wimg, origin='lower', cmap=cmap,
#                                 interpolation='none', norm=norm)
#            ax[1+iobj, 1].imshow(~opt_psfmasks[0, :, :]*1, cmap='gray', origin='lower')
#            ax[1+iobj, 2].imshow(~opt_galmasks[0, :, :]*1, cmap='gray', origin='lower')
#
#            for icol in range(3):
#                # redraw the initial geometry for this object/row
#                overplot_ellipse(obj['DIAM_INIT'], obj['BA_INIT'], obj['PA_INIT'],
#                                 obj['BX_INIT'], obj['BY_INIT'],
#                                 pixscale=refpixscale, ax=ax[1+iobj, icol], color='cyan',
#                                 linestyle='-', draw_majorminor_axes=True, jpeg=False)
#
#                # for the factor of XX, see
#                # https://github.com/moustakas/SGA/blob/main/science/SGA2025/SGA2020-diameters.ipynb
#                overplot_ellipse(mge.majoraxis*refpixscale*2.5, (1.-mge.eps), mge.pa,
#                                 mge.ymed, mge.xmed, pixscale=refpixscale,
#                                 ax=ax[1+iobj, icol], color='red', linestyle='-',
#                                 draw_majorminor_axes=True, jpeg=False)
#
#            for xx in ax[1+iobj, :]:
#                xx.set_xlim(0, wimg.shape[0])
#                xx.set_ylim(0, wimg.shape[1])
#
#    for xx in ax.ravel():
#        xx.margins(0)
#        #xx.set_xticks([])
#        #xx.set_yticks([])
#
#    fig.suptitle(data['galaxy'].replace('_', ' ').replace(' GROUP', ' Group'))
#    fig.savefig(qafile)
#    log.info(f'Wrote {qafile}')
#
#    pdb.set_trace()


#        largeshift = False
#        var = np.zeros_like(invvar)
#        ok = invvar > 0
#        var[ok] = 1. / invvar[ok]
#        data[f'{filt}_var_'] = var # [nanomaggies**2]
#
#        # Did the galaxy position move? If so, revert back to the Tractor geometry.
#        if np.abs(mgegalaxy.xmed-mge.xmed) > maxshift or np.abs(mgegalaxy.ymed-mge.ymed) > maxshift:
#            log.warning(f'Large centroid shift! (x,y) = ({mgegalaxy.xmed:.3f},{mgegalaxy.ymed:.3f})-->' + \
    #                        f'({mge.xmed:.3f},{mge.ymed:.3f})')
#            largeshift = True
#            mgegalaxy = copy(mge)
#
#        radec_med = data[f'{refband.lower()}_wcs'].pixelToPosition(
#            mgegalaxy.ymed+1, mgegalaxy.xmed+1).vals
#        #radec_peak = data[f'{refband.lower()}_wcs'].pixelToPosition(
#        #    mgegalaxy.ypeak+1, mgegalaxy.xpeak+1).vals
#        mge = {
#            'largeshift': largeshift,
#            'ra': tractor.ra[central], 'dec': tractor.dec[central],
#            'bx': tractor.bx[central], 'by': tractor.by[central],
#            'ra_moment': radec_med[0], 'dec_moment': radec_med[1],
#            }
#
#        # add the dust
#        photsys = 'S'
#        ebv = SFDMap().ebv(radec_med[0], radec_med[1])
#        mge['ebv'] = np.float32(ebv)
#        for band in data['bands']:
#            print('TEMPORARILY SKIPPING MW_TRANSMISSION!!')
#            #mge[f'mw_transmission_{band.lower()}'] = mwdust_transmission(
#            #    ebv, band, photsys, match_legacy_surveys=True).astype('f4')
#            mge[f'mw_transmission_{band.lower()}'] = np.float32(1.)
#
#        for key in ['eps', 'majoraxis', 'pa', 'theta', 'xmed', 'ymed', 'xpeak', 'ypeak']:
#            mge[key] = np.float32(getattr(mgegalaxy, key))
#            if key == 'pa': # put into range [0-180]
#                mge[key] = mge[key] % np.float32(180)
#        data['mge'].append(mge)
#
#        # [2] Create the satellite mask in all the bandpasses. Use
#        # srcs here, which has had the satellites nearest to the
#        # central galaxy trimmed out.
#        log.info('Building the satellite mask.')
#        satmask = np.zeros(data[refband].shape, bool)
#
#        for filt in fit_bands:
#            # do not let GALEX and WISE contribute to the satellite mask
#            if data[filt].shape != satmask.shape:
#                continue
#
#            cenflux = getattr(tractor, f'flux_{filt}')[central]
#            satflux = getattr(srcs, f'flux_{filt}')
#            if cenflux <= 0.0:
#                log.warning('Central galaxy flux is negative! Proceed with caution...')
#
#            satindx = np.where(np.logical_or(
#                (srcs.type != 'PSF') * (srcs.shape_r > r50mask) *
#                (satflux > 0.0) * ((satflux / cenflux) > threshmask),
#                srcs.ref_cat == ref_cat))[0]
#            #satindx = np.where(srcs.ref_cat == ref_cat)[0]
#            #if np.isin(central, satindx):
#            #    satindx = satindx[np.logical_not(np.isin(satindx, central))]
#            if len(satindx) == 0:
#                #raise ValueError('All satellites have been dropped!')
#                log.warning(f'Warning! All satellites have been dropped from band {filt}!')
#            else:
#                satsrcs = srcs.copy()
#                satsrcs.cut(satindx)
#                satimg = srcs2image(satsrcs, data[f'{filt}_wcs'],
#                                    band=filt,
#                                    pixelized_psf=data[f'{filt}_psf'])
#                thissatmask = satimg > sigmamask*data[f'{filt}_sigma']
#                #if filt == 'FUV':
#                #    plt.clf() ; plt.imshow(thissatmask, origin='lower') ; plt.savefig('junk-{}.png'.format(filt))
#                #    #plt.clf() ; plt.imshow(data[filt], origin='lower') ; plt.savefig('junk-{}.png'.format(filt))
#                if satmask.shape != satimg.shape:
#                    thissatmask = resize(thissatmask*1.0, satmask.shape, mode='reflect') > 0
#
#                satmask = np.logical_or(satmask, thissatmask)
#                #if True:
#                #    import matplotlib.pyplot as plt
#                ##    plt.clf() ; plt.imshow(np.log10(satimg), origin='lower') ; plt.savefig('debug.png')
#                #    plt.clf() ; plt.imshow(satmask, origin='lower') ; plt.savefig('desi-users/ioannis/tmp/debug.png')
#                ###    #plt.clf() ; plt.imshow(satmask, origin='lower') ; plt.savefig('/mnt/legacyhalos-data/debug.png')
#            #print(filt, np.sum(satmask), np.sum(thissatmask))
#        #plt.clf() ; plt.imshow(satmask, origin='lower') ; plt.savefig('junk-satmask.png')
#
#        # [3] Build the final image (in each filter) for
#        # ellipse-fitting. First, subtract out the PSF sources. Then
#        # update the mask (but ignore the residual mask). Finally
#        # convert to surface brightness.  for filt in ['W1']:
#        for filt in fit_bands:
#            thismask = ma.getmask(data[filt])
#            if satmask.shape != thismask.shape:
#                _satmask = (resize(satmask*1.0, thismask.shape, mode='reflect') > 0) == 1.0
#                _centralmask = (resize(centralmask*1.0, thismask.shape, mode='reflect') > 0) == 1.0
#                mask = np.logical_or(thismask, _satmask)
#                mask[_centralmask] = False
#            else:
#                mask = np.logical_or(thismask, satmask)
#                mask[centralmask] = False
#            #if filt == 'W1':
#            #    plt.imshow(_satmask, origin='lower') ; plt.savefig('junk-satmask-{}.png'.format(filt))
#            #    plt.imshow(mask, origin='lower') ; plt.savefig('junk-mask-{}.png'.format(filt))
#
#            varkey = f'{filt}_var'
#            imagekey = f'{filt}_masked'
#            psfimgkey = f'{filt}_psfimg'
#            thispixscale = filt2pixscale[filt]
#            if imagekey not in data.keys():
#                data[imagekey], data[varkey], data[psfimgkey] = [], [], []
#
#            img = ma.getdata(data[filt]).copy()
#
#            # Get the PSF sources but ignore W3 and W4 (??)
#            psfindx = np.where((getattr(tractor, f'flux_{filt}') / cenflux > threshmask) *
#                               (tractor.type == 'PSF'))[0]
#            if len(psfindx) > 0 and filt.upper() != 'W3' and filt.upper() != 'W4':
#                psfsrcs = tractor.copy()
#                psfsrcs.cut(psfindx)
#            else:
#                psfsrcs = None
#
#            if psfsrcs:
#                psfimg = srcs2image(psfsrcs, data[f'{filt}_wcs'],
#                                    band=filt,
#                                    pixelized_psf=data[f'{filt}_psf'])
#                if False:#True:
#                    #import fitsio ; fitsio.write('junk-psf-{}.fits'.format(filt), data['{}_psf'.format(filt)].img, clobber=True)
#                    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
#                    im = ax1.imshow(np.log10(img), origin='lower') ; fig.colorbar(im, ax=ax1)
#                    im = ax2.imshow(np.log10(psfimg), origin='lower') ; fig.colorbar(im, ax=ax2)
#                    im = ax3.imshow(np.log10(data[f'{filt}_psf'].img), origin='lower') ; fig.colorbar(im, ax=ax3)
#                    im = ax4.imshow(img-psfimg, origin='lower') ; fig.colorbar(im, ax=ax4)
#                    plt.savefig(f'ioannis/tmp/qa-psf-{filt}.png')
#                    if filt == 'r':
#                        pass
#                img -= psfimg
#            else:
#                psfimg = np.zeros((2, 2), 'f4')
#
#            data[psfimgkey].append(psfimg)
#
#            img = ma.masked_array((img / thispixscale**2).astype('f4'), mask) # [nanomaggies/arcsec**2]
#            var = data[f'{filt}_var_'] / thispixscale**4 # [nanomaggies**2/arcsec**4]
#
#            # Fill with zeros, for fun--
#            ma.set_fill_value(img, fill_value)
#            #if filt == 'r':# or filt == 'r':
#            #    plt.clf() ; plt.imshow(img, origin='lower') ; plt.savefig('desi-users/ioannis/tmp/junk-img-{}.png'.format(filt))
#            #    plt.clf() ; plt.imshow(mask, origin='lower') ; plt.savefig('desi-users/ioannis/tmp/junk-mask-{}.png'.format(filt))
#            ##    plt.clf() ; plt.imshow(thismask, origin='lower') ; plt.savefig('desi-users/ioannis/tmp/junk-thismask-{}.png'.format(filt))
#
#            data[imagekey].append(img)
#            data[varkey].append(var)
#
#        #test = data['r_masked'][0]
#        #plt.clf() ; plt.imshow(np.log(test.clip(test[mgegalaxy.xpeak, mgegalaxy.ypeak]/1e4)), origin='lower') ; plt.savefig('/mnt/legacyhalos-data/debug.png')

    # Cleanup?
    for filt in fit_bands:
        del data[filt]
        del data[f'{filt}_var_']

    return data


def read_multiband(galaxy, galaxydir, sort_by_flux=False, bands=['g', 'r', 'i', 'z'],
                   pixscale=0.262, galex_pixscale=1.5, unwise_pixscale=2.75,
                   galaxy_id=None, galex=False, unwise=False, verbose=False):
    """Read the multi-band images (converted to surface brightness) and create a
    masked array suitable for ellipse-fitting.

    """
    import fitsio
    from astropy.table import Table
    import astropy.units as u
    from astrometry.util.fits import fits_table
    from astrometry.util.util import Tan
    from legacypipe.bits import MASKBITS

    optical_bands = bands.copy()
    galex_bands, unwise_bands = None, None

    # Dictionary mapping between optical filter and filename coded up in
    # coadds.py, galex.py, and unwise.py, which depends on the project.
    data = {}
    data['galaxy'] = galaxy
    data['galaxydir'] = galaxydir

    refband, galex_refband, unwise_refband = None, None, None

    filt2imfile, filt2pixscale = {}, {}
    for band in bands:
        filt2imfile.update({band: {'image': 'image',
                                   'model': 'model',
                                   'invvar': 'invvar',
                                   'psf': 'psf',}})
        filt2pixscale.update({band: pixscale})
    filt2imfile.update({'tractor': 'tractor',
                        'sample': 'sample',
                        'maskbits': 'maskbits',})

    if galex:
        galex_bands = ['FUV', 'NUV']
        bands = bands + galex_bands
        galex_refband = galex_bands[0]
        for band in galex_bands:
            filt2imfile.update({band: {'image': 'image',
                                       'model': 'model',
                                       'invvar': 'invvar',
                                       'psf': 'psf'}})
            filt2pixscale.update({band: galex_pixscale})

    if unwise:
        unwise_bands = ['W1', 'W2', 'W3', 'W4']
        bands = bands + unwise_bands
        unwise_refband = unwise_bands[0]
        for band in unwise_bands:
            filt2imfile.update({band: {'image': 'image',
                                       'model': 'model',
                                       'invvar': 'invvar',
                                       'psf': 'psf'}})
            filt2pixscale.update({band: unwise_pixscale})

    data.update({'filt2pixscale': filt2pixscale})

    # Need to differentiate between missing one or more data products,
    # which indicates something went wrong with the previous (coadds)
    # stage vs missing all the data in a given bandpass, which is OK.
    fit_bands, fit_optical_bands = [], []
    for filt in bands:
        datacount = 0
        for ii, imtype in enumerate(filt2imfile[filt].keys()):
            imfile = os.path.join(galaxydir, f'{galaxy}-{filt2imfile[filt][imtype]}-{filt}.fits.fz')
            if os.path.isfile(imfile):
                filt2imfile[filt][imtype] = imfile
                datacount += 1
            else:
                log.warning(f'Missing {imfile}')

        if datacount > 0:
            if datacount == len(filt2imfile[filt].keys()):
                fit_bands.append(filt)
                # refband can be the first optical band
                if filt in optical_bands:
                    fit_optical_bands.append(filt)
                    if refband is None:
                        refband = filt
            else:
                msg = f'Missing one or more {filt}-band data products!'
                log.critical(msg)
                data['missingdata'] = True
                return data

    log.info(f'Found complete data in bands: {",".join(fit_bands)}')

    # Pack some preliminary info into the output dictionary.
    data['bands'] = bands
    data['optical_bands'] = optical_bands
    data['galex_bands'] = galex_bands
    data['unwise_bands'] = unwise_bands
    data['fit_bands'] = fit_bands
    data['fit_optical_bands'] = fit_optical_bands

    data['refband'] = refband
    data['galex_refband'] = galex_refband
    data['unwise_refband'] = unwise_refband
    data['refpixscale'] = pixscale

    # We ~have~ to read the tractor catalog using fits_table because we will
    # turn these catalog entries into Tractor sources later.
    tractorfile = os.path.join(galaxydir, f'{galaxy}-{filt2imfile["tractor"]}.fits')

    cols = ['ra', 'dec', 'bx', 'by', 'type', 'ref_cat', 'ref_id',
            'sersic', 'shape_r', 'shape_e1', 'shape_e2']
    cols += [f'flux_{filt}' for filt in fit_optical_bands]
    cols += [f'flux_ivar_{filt}' for filt in fit_optical_bands]
    cols += [f'nobs_{filt}' for filt in fit_optical_bands]
    cols += [f'mw_transmission_{filt}' for filt in fit_optical_bands]
    cols += [f'psfdepth_{filt}' for filt in fit_optical_bands]
    cols += [f'psfsize_{filt}' for filt in fit_optical_bands]
    if galex:
        cols += [f'flux_{filt.lower()}' for filt in galex_bands]
        cols += [f'flux_ivar_{filt.lower()}' for filt in galex_bands]
    if unwise:
        cols += [f'flux_{filt.lower()}' for filt in unwise_bands]
        cols += [f'flux_ivar_{filt.lower()}' for filt in unwise_bands]

    tractor = fits_table(tractorfile, columns=cols)
    log.info(f'Read {len(tractor):,d} sources from {tractorfile}')

    # Read the sample catalog from custom_coadds and find each source
    # in the Tractor catalog.
    samplefile = os.path.join(galaxydir, f'{galaxy}-{filt2imfile["sample"]}.fits')
    cols = ['SGAID', 'RA', 'DEC', 'DIAM', 'PA', 'BA', 'FITBIT']
    sample = Table(fitsio.read(samplefile, columns=cols))
    log.info(f'Read {len(sample)} source(s) from {samplefile}')
    for col in ['RA', 'DEC', 'DIAM', 'PA', 'BA']:
        sample.rename_column(col, f'{col}_INIT')

    sample['DIAM_INIT'] *= 60. # [arcsec]

    # populate (BX,BY)_INIT by quickly building the WCS
    wcs = Tan(filt2imfile[refband]['image'], 1)
    (_, x0, y0) = wcs.radec2pixelxy(sample['RA_INIT'].value, sample['DEC_INIT'].value)
    sample['BX_INIT'] = (x0 - 1.).astype('f4') # NB the -1!
    sample['BY_INIT'] = (y0 - 1.).astype('f4')

    #sample['ROW'] = np.zeros(len(sample), int) - 1
    #sample['RA'] = np.zeros(len(sample), 'f8')
    #sample['DEC'] = np.zeros(len(sample), 'f8')
    #sample['BX'] = np.zeros(len(sample), 'f4')
    #sample['BY'] = np.zeros(len(sample), 'f4')
    #sample['TYPE'] = np.zeros(len(sample), 'U3')
    #sample['SERSIC'] = np.zeros(len(sample), 'f4')
    #sample['SHAPE_R'] = np.zeros(len(sample), 'f4') # [arcsec]
    #sample['SHAPE_E1'] = np.zeros(len(sample), 'f4')
    #sample['SHAPE_E2'] = np.zeros(len(sample), 'f4')
    #sample['PSF'] = np.zeros(len(sample), bool)

    sample['FLUX'] = np.zeros(len(sample), 'f4') # brightest band
    sample['DROPPED'] = np.zeros(len(sample), bool)
    sample['LARGESHIFT'] = np.zeros(len(sample), bool)
    sample['BLENDED'] = np.zeros(len(sample), bool)

    # special fitting bit(s) -- FIXME!
    sample['FIXGEOMETRY'] = sample['FITBIT'] & FITBITS['ignore'] != 0
    sample['FORCEPSF'] = sample['FITBIT'] & FITBITS['forcepsf'] != 0

    copycols = ['RA', 'DEC', 'BX', 'BY', 'TYPE', 'SERSIC',
                'SHAPE_R', 'SHAPE_E1', 'SHAPE_E2']

    samplesrcs = []
    for iobj, refid in enumerate(sample[REFIDCOLUMN].value):
        I = np.where((tractor.ref_cat == REFCAT) * (tractor.ref_id == refid))[0]
        if len(I) == 0:
            log.warning(f'ref_id={refid} dropped by Tractor')
            sample['DROPPED'][iobj] = True
            samplesrcs.append(None)
            ## important! set (BX,BY) to (BX_INIT,BY_INIT) because it
            ## gets used downstream
            #sample['BX'][iobj] = sample['BX_INIT'][iobj]
            #sample['BY'][iobj] = sample['BY_INIT'][iobj]
        else:
            samplesrcs.append(tractor[I])
            #sample['ROW'][iobj] = I
            #for col, attr in zip(copycols, np.char.lower(copycols)):
            #    sample[col][iobj] = getattr(tractor[I[0]], attr)
            #if sample['TYPE'][iobj] in ['PSF', 'DUP']:
            if tractor[I[0]].type in ['PSF', 'DUP']:
                log.warning(f'ref_id={refid} fit by Tractor as PSF (or DUP)')
                #sample['PSF'][iobj] = True
            sample['FLUX'][iobj] = max([getattr(tractor[I[0]], f'flux_{filt}')
                                        for filt in fit_optical_bands])

    # Sort by initial diameter or optical brightness (in any band).
    if sort_by_flux:
        log.info('Sorting by optical flux:')
        srt = np.argsort(sample['FLUX'])[::-1]
    else:
        log.info('Sorting by initial diameter:')
        srt = np.argsort(sample['DIAM_INIT'])[::-1]
    sample = sample[srt]
    samplesrcs = [samplesrcs[I] for I in srt]
    for obj in sample:
        log.info(f'  ref_id={obj[REFIDCOLUMN]}: D(25)={obj["DIAM_INIT"]/60.:.3f} arcmin, ' + \
                 f'max optical flux={obj["FLUX"]:.2f} nanomaggies')
    data['sample'] = sample
    data['samplesrcs'] = samplesrcs

    # add the PSF depth and size
    data = _get_psfsize_and_depth(data, tractor, fit_optical_bands,
                                  pixscale, incenter=False)

    # Read the maskbits image and build the starmask.
    maskbitsfile = os.path.join(galaxydir, f'{galaxy}-{filt2imfile["maskbits"]}.fits.fz')
    if verbose:
        log.info(f'Reading {maskbitsfile}')
    F = fitsio.FITS(maskbitsfile)
    maskbits = F['MASKBITS'].read()

    brightstarmask = ( (maskbits & MASKBITS['BRIGHT'] != 0) |
                       (maskbits & MASKBITS['MEDIUM'] != 0) |
                       (maskbits & MASKBITS['CLUSTER'] != 0) )
    data['brightstarmask'] = brightstarmask

    # missing bands have ALLMASK_[GRIZ] == 0
    for filt in fit_optical_bands:
        data[f'allmask_{filt}'] = maskbits & MASKBITS[f'ALLMASK_{filt.upper()}'] != 0

    if unwise and 'WISEM1' in F:
        w1mask = F['WISEM1'].read() # same size as maskbits
        w2mask = F['WISEM2'].read()
        wisemask = np.logical_or(w1mask, w2mask) > 0
    else:
        wisemask = None
    data['wisemask'] = wisemask

    # Read the basic imaging data and masks and build the multiband
    # mask.
    data = _read_image_data(data, filt2imfile, verbose=verbose)
    data = _build_multiband_mask(data, tractor)

    return data
