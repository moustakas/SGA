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

REFCAT = 'L4'
RACOLUMN = 'GROUP_RA'   # 'RA'
DECCOLUMN = 'GROUP_DEC' # 'DEC'
DIAMCOLUMN = 'GROUP_DIAMETER' # 'DIAM'
REFIDCOLUMN = 'SGAID'

FITBITS = dict(
    ignore = 2**0,    # no special behavior (e.g., resolved dwarf galaxy)
    forcegaia = 2**1, # only fit Gaia point sources (and any SGA galaxies), e.g., LMC
    forcepsf = 2**2,  # force PSF for source detection and photometry within the SGA mask
)

SAMPLEBITS = dict(
    LVD = 2**0,       # LVD / local dwarfs
)

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
    FUV = 2**4,        #
    NUV = 2**5,        #
)

UNWISEMASKBITS = dict(
    brightstar = 2**0, # BRIGHT, MEDIUM, or CLUSTER MASKBITS
    gaiastar = 2**1,   # Gaia (type=PSF) stars
    galaxy = 2**2,     # galaxy (extended, non-reference) sources
    reference = 2**3,  # SGA (reference) sources
    W1 = 2**4,         #
    W2 = 2**5,         #
    W3 = 2**6,         #
    W4 = 2**7,         #
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
            #log.warning('No matching galaxies using column GROUP_NAME!')
            I = np.isin(sample['SGANAME'], galaxylist)
            if np.count_nonzero(I) == 0:
                #log.warning('No matching galaxies using column SGANAME!')
                I = np.isin(sample['OBJNAME'], galaxylist)
                if np.count_nonzero(I) == 0:
                    log.warning('No matching galaxies found in sample.')
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

    print('######## Be sure to remove ref_cat==G3 and type==DUP sources from the ellipse catalog passed to legacypipe!')

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
               'OPT_REFBAND', 'REFBAND_WIDTH', 'REFBAND_HEIGHT']
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


def _get_psfsize_and_depth(sample, tractor, bands, pixscale, incenter=False):
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
        psfsigmacol = f'psfsigma_{filt.lower()}'
        psfsizecol = f'psfsize_{filt.lower()}'
        psfdepthcol = f'psfdepth_{filt.lower()}'

        if psfsizecol in tractor.columns():
            good = np.where(tractor.get(psfsizecol)[these] > 0)[0]
            if len(good) == 0:
                log.warning(f'  No good measurements of the PSF size in band {filt}!')
                #data[psfsigmacol] = np.float32(0.0)
                #data[psfsizecol] = np.float32(0.0)
            else:
                # Get the PSF size and image depth.
                psfsize = tractor.get(psfsizecol)[these][good]   # [FWHM, arcsec]
                psfsigma = psfsize / np.sqrt(8 * np.log(2)) / pixscale # [sigma, pixels]

                #data[psfsigmacol] = np.median(psfsigma).astype('f4')
                #data[psfsizecol] = np.median(psfsize).astype('f4')
                sample[psfsizecol.upper()] = np.median(psfsize).astype('f4')
            tractor.delete_column(psfsizecol)

        if psfdepthcol in tractor.columns():
            good = np.where(tractor.get(psfdepthcol)[these] > 0)[0]
            if len(good) == 0:
                log.warning(f'  No good measurements of the PSF depth in band {filt}!')
                #data[psfdepthcol] = np.float32(0.0)
            else:
                psfdepth = tractor.get(psfdepthcol)[these][good] # [AB mag, 5-sigma]
                #data[psfdepthcol] = (22.5-2.5*np.log10(1./np.sqrt(np.median(psfdepth)))).astype('f4')
                sample[psfdepthcol.upper()] = (22.5-2.5*np.log10(1./np.sqrt(np.median(psfdepth)))).astype('f4')

            tractor.delete_column(psfdepthcol)

    return sample


def read_image_data(data, filt2imfile, verbose=False):
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

    all_bands = data['all_bands']
    opt_bands = data['opt_bands']
    unwise_bands = data['unwise_bands']

    opt_refband = data['opt_refband']
    galex_refband = data['galex_refband']
    unwise_refband = data['unwise_refband']

    # Read the per-filter images and generate an optical and UV/IR
    # mask.
    for filt in all_bands:
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
        assert(sz[0] == sz[1])
        if filt == opt_refband or filt == galex_refband or filt == unwise_refband:
            if filt == opt_refband:
                data['width'] = sz[0]

            wcs = Tan(hdr)
            if 'MJD_MEAN' in hdr:
                mjd_tai = hdr['MJD_MEAN'] # [TAI]
                wcs = LegacySurveyWcs(wcs, TAITime(None, mjd=mjd_tai))
            else:
                wcs = ConstantFitsWcs(wcs)

            if filt == opt_refband:
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

        if filt in opt_bands:
            mask = np.logical_or(mask, data[f'allmask_{filt}'])
            del data[f'allmask_{filt}']

        # add wisemask for W1/W2, if present, but we have to resize
        if data['wisemask'] is not None and filt in unwise_bands[:2]:
            _wisemask = resize(data['wisemask'], mask.shape, mode='edge',
                               anti_aliasing=False) > 0
            mask = np.logical_or(mask, _wisemask)

        mask = binary_dilation(mask, iterations=2)

        #if filt in opt_bands:
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

        if filt in opt_bands:
            #import matplotlib.pyplot as plt
            #plt.clf() ; plt.imshow(np.log10(image-model),origin='lower') ; plt.savefig('ioannis/tmp/junk2.png') ; plt.close()
            #resid = gaussian_filter(image - model, 2.)
            #_, _, sig = sigma_clipped_stats(resid[~mask], sigma=2.5)
            _, _, sig = sigma_clipped_stats(image-model, sigma=2.5)
            data[f'{filt}_sigma'] = sig

    if 'wisemask' in data:
        del data['wisemask']

    return data


def unpack_maskbits(maskbits, bands=['g', 'r', 'i', 'z'],
                    BITS=OPTMASKBITS, allmasks=False):
    """Unpack the maskbits bitmask, which has shape [nobj, width,
    width], to include the per-band data with resulting shape
    [nobj,nband,width,width].

    The result is a *boolean* mask and, optionally, all the individual
    masks (brightstarmask, etc.)

    """
    nband = len(bands)
    nobj, width, _ = maskbits.shape
    masks_perband = np.zeros((nobj, nband, width, width), bool) # True=masked

    if allmasks:
        brightstarmasks = np.zeros_like(maskbits, bool)
        refmasks = np.zeros_like(maskbits, bool)
        gaiamasks = np.zeros_like(maskbits, bool)
        galmasks = np.zeros_like(maskbits, bool)

    for iobj in range(nobj):
        brightstarmask = maskbits[iobj, :, :] & BITS['brightstar'] != 0
        refmask = maskbits[iobj, :, :] & BITS['reference'] != 0
        gaiamask = maskbits[iobj, :, :] & BITS['gaiastar'] != 0
        galmask = maskbits[iobj, :, :] & BITS['galaxy'] != 0

        if allmasks:
            brightstarmasks[iobj, :, :] = brightstarmask
            refmasks[iobj, :, :] = refmask
            gaiamasks[iobj, :, :] = gaiamask
            galmasks[iobj, :, :] = galmask

        objmask = np.logical_or.reduce((brightstarmask, refmask, gaiamask, galmask))

        for iband, filt in enumerate(bands):
            masks_perband[iobj, iband, :, :] = np.logical_or(
                objmask, maskbits[iobj, :, :] & BITS[filt] != 0)

    if allmasks:
        return masks_perband, brightstarmasks, refmasks, gaiamasks, galmasks
    else:
        return masks_perband


def _update_masks(brightstarmask, gaiamask, refmask, galmask, mask_perband,
                  bands, sz, MASKDICT=None, build_maskbits=False,
                  do_resize=False, verbose=False):
    """Update the masks.

    """
    from skimage.transform import resize

    # optionally pack into a bitmask
    if build_maskbits:
        if do_resize:
            #import matplotlib.pyplot as plt
            #msk = galmask
            #fig, (ax1, ax2) = plt.subplots(1, 2)
            #ax1.imshow(msk, origin='lower')
            #ax2.imshow(resize(msk, sz, mode='edge', anti_aliasing=False) > 0, origin='lower')
            #plt.savefig('ioannis/tmp/junk.png')

            brightstarmask = resize(brightstarmask, sz, mode='edge', anti_aliasing=False) > 0
            refmask = resize(refmask, sz, mode='edge', anti_aliasing=False) > 0
            galmask = resize(galmask, sz, mode='edge', anti_aliasing=False) > 0
            gaiamask = resize(gaiamask, sz, mode='edge', anti_aliasing=False) > 0

        maskbits = np.zeros(sz, np.int32)
        maskbits[brightstarmask] += MASKDICT['brightstar']
        maskbits[refmask] += MASKDICT['reference']
        maskbits[galmask] += MASKDICT['galaxy']
        maskbits[gaiamask] += MASKDICT['gaiastar']

        for iband, filt in enumerate(bands):
            maskbits[mask_perband[iband, :, :]] += MASKDICT[filt]
        return maskbits
    else:
        objmask = np.logical_or.reduce((brightstarmask, refmask, galmask, gaiamask))
        if verbose:
            for label, msk in zip(['bright-star', 'reference', 'gaia', 'galaxy', 'total'],
                                  [brightstarmask, refmask, galmask, gaiamask, objmask]):
                print('  ', label, np.sum(msk)/sz[0]**2)

        masks = np.zeros((len(bands), *sz), bool)
        for iband, filt in enumerate(bands):
            masks[iband, :, :] = np.logical_or(objmask, mask_perband[iband, :, :])
        return masks


def qa_multiband_mask(data, geo_initial, geo_final):
    """Diagnostic QA for the output of build_multiband_mask.

    """
    import matplotlib.pyplot as plt
    from matplotlib.cm import get_cmap
    import matplotlib.gridspec as gridspec
    from matplotlib.patches import Patch

    from SGA.util import var2ivar
    from SGA.qa import overplot_ellipse, get_norm

    qafile = os.path.join('/global/cfs/cdirs/desi/users/ioannis/tmp',
                          f'qa-ellipsemask-{data["galaxy"]}.png')

    alpha = 0.6
    orange = (0.9, 0.6, 0.0, alpha)   # golden-orange
    blue   = (0.0, 0.45, 0.7, alpha)  # muted blue
    purple = (0.8, 0.6, 0.7, alpha)   # soft violet
    magenta = (0.85, 0.2, 0.5, alpha) # vibrant rose

    opt_bands = data['opt_bands']
    opt_images = data['opt_images']
    opt_maskbits = data['opt_maskbits']
    opt_models = data['opt_models']
    opt_weight = var2ivar(data['opt_variance'])
    opt_pixscale = data['opt_pixscale']

    sample = data['sample']
    nsample = len(sample)

    ncol = 3
    nrow = 1 + nsample
    inches_per_panel = 3.
    fig, ax = plt.subplots(nrow, ncol,
                           figsize=(inches_per_panel*ncol,
                                    inches_per_panel*nrow),
                           gridspec_kw={'wspace': 0.01, 'hspace': 0.01},
                           constrained_layout=True)

    #cmap = plt.cm.viridis
    cmap = plt.cm.cividis
    #cmap = plt.cm.twilight
    #cmap = plt.cm.magma
    #cmap = plt.cm.Greys_r
    #cmap = plt.cm.inferno
    cmap.set_bad('white')

    cmap1 = get_cmap('tab20') # or tab20b or tab20c
    colors1 = [cmap1(i) for i in range(20)]

    #cmap2 = get_cmap('Set1')
    cmap2 = get_cmap('Dark2')
    colors2 = [cmap2(i) for i in range(5)]

    #setcolors1 = [
    #    '#4daf4a',  # green
    #    '#e41a1c',  # strong red
    #    '#ff7f00',  # orange
    #    '#00ffff',  # cyan
    #    '#984ea3',  # purple
    #    '#377eb8',  # saturated blue
    #    '#a65628',  # brown
    #    '#f781bf',  # pink
    #    '#ffff33',  # bright yellow (only on dark parts)
    #]
    #setcolors1 = plt.rcParams['axes.prop_cycle'].by_key()['color']

    width = data['width']
    sz = (width, width)

    # coadded optical, IR, and UV images and initial geometry
    imgbands = [opt_bands, data['unwise_bands'], data['galex_bands']]
    labels = [''.join(opt_bands), 'unWISE', 'GALEX']
    for iax, (xx, bands, label, pixscale, refband) in enumerate(zip(
            ax[0, :], imgbands, labels,
            [data['opt_pixscale'], data['unwise_pixscale'], data['galex_pixscale']],
            [data['opt_refband'], data['unwise_refband'], data['galex_refband']])):
        wimgs = np.stack([data[filt] for filt in bands])
        wivars = np.stack([data[f'{filt}_invvar'] for filt in bands])
        wimg = np.sum(wivars * wimgs, axis=0)

        xx.imshow(wimg, origin='lower', cmap=cmap, interpolation='none',
                  norm=get_norm(wimg), alpha=1.)
        xx.set_xlim(0, wimg.shape[0]-1)
        xx.set_ylim(0, wimg.shape[1]-1)
        xx.margins(0)

        # initial ellipse geometry
        pixfactor = data['opt_pixscale'] / pixscale
        for iobj, obj in enumerate(sample):
            (bx, by, diam, ba, pa) = geo_initial[iobj, :]
            overplot_ellipse(diam*pixfactor*pixscale, ba, pa,
                             bx*pixfactor, by*pixfactor, pixscale=pixscale,
                             ax=xx, color=colors1[iobj], linestyle='-',
                             linewidth=2, draw_majorminor_axes=True,
                             jpeg=False, label=obj[REFIDCOLUMN])

        xx.text(0.03, 0.97, label, transform=xx.transAxes,
                ha='left', va='top', color='white',
                linespacing=1.5, fontsize=10,
                bbox=dict(boxstyle='round', facecolor='k', alpha=0.5))

        if iax == 0:
            xx.legend(loc='lower left', fontsize=7, ncol=2,
                      fancybox=True, framealpha=0.5)
        del wimgs, wivars, wimg

    # unpack the maskbits bitmask
    opt_masks, brightstarmasks, refmasks, gaiamasks, galmasks = \
        unpack_maskbits(opt_maskbits, bands=opt_bands, # [nobj,nband,width,width]
                        BITS=OPTMASKBITS, allmasks=True)

    # one row per object
    for iobj, obj in enumerate(sample):
        opt_masks_obj = opt_masks[iobj, :, :, :]
        brightstarmask = brightstarmasks[iobj, :, :]
        gaiamask = gaiamasks[iobj, :, :]
        galmask = galmasks[iobj, :, :]
        refmask = refmasks[iobj, :, :]

        wimg = np.sum(opt_weight * np.logical_not(opt_masks_obj) * opt_images[iobj, :, :], axis=0)
        wimg[wimg == 0.] = np.nan

        ax[1+iobj, 0].imshow(wimg, cmap=cmap, origin='lower', interpolation='none',
                             norm=get_norm(wimg))
        #ax[1+iobj, 0].imshow(np.log(wimg.clip(wimg[int(bx), int(by)]/1e4)),
        #                     cmap=cmap, origin='lower', interpolation='none')
        #ax[1+iobj, 0].imshow(mge.mask, origin='lower', cmap='binary',
        #                     interpolation='none', alpha=0.3)

        wmodel = np.sum(opt_models[iobj, :, :, :], axis=0)
        ax[1+iobj, 1].imshow(wmodel, cmap=cmap, origin='lower', interpolation='none',
                             norm=get_norm(wmodel))

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
            [bx, by, diam, ba, pa] = geo_initial[iobj, :]
            overplot_ellipse(diam*opt_pixscale, ba, pa, bx, by,
                             pixscale=opt_pixscale, ax=ax[1+iobj, col],
                             color=colors2[0], linestyle='-',
                             linewidth=2, draw_majorminor_axes=True,
                             jpeg=False, label='Initial')

            # final geometry
            [bx, by, diam, ba, pa] = geo_final[iobj, :]
            overplot_ellipse(diam*opt_pixscale, ba, pa, bx, by,
                             pixscale=opt_pixscale, ax=ax[1+iobj, col],
                             color=colors2[1], linestyle='--', linewidth=2,
                             draw_majorminor_axes=True, jpeg=False, label='Final')
            ax[1+iobj, col].set_xlim(0, width-1)
            ax[1+iobj, col].set_ylim(0, width-1)
            ax[1+iobj, col].margins(0)

        ax[1+iobj, 0].text(0.03, 0.97, obj[REFIDCOLUMN], transform=ax[1+iobj, 0].transAxes,
                           ha='left', va='top', color='white',
                           linespacing=1.5, fontsize=10,
                           bbox=dict(boxstyle='round', facecolor='k', alpha=0.5))

        if iobj == 0:
            ax[1+iobj, 1].text(0.03, 0.97, f'{"".join(opt_bands)} models',
                               transform=ax[1+iobj, 1].transAxes, ha='left', va='top',
                               color='white', linespacing=1.5, fontsize=10,
                               bbox=dict(boxstyle='round', facecolor='k', alpha=0.5))
            ax[1+iobj, 2].text(0.03, 0.97, f'{"".join(opt_bands)} masks',
                               transform=ax[1+iobj, 2].transAxes, ha='left', va='top',
                               color='white', linespacing=1.5, fontsize=10,
                               bbox=dict(boxstyle='round', facecolor='k', alpha=0.5))

        ax[1+iobj, 0].legend(loc='lower left', fontsize=7, fancybox=True,
                             framealpha=0.5)

    for xx in ax.ravel():
        xx.margins(0)
        xx.set_xticks([])
        xx.set_yticks([])

    #fig.suptitle(data['galaxy'].replace('_', ' ').replace(' GROUP', ' Group'))
    fig.savefig(qafile)
    log.info(f'Wrote {qafile}')


def build_multiband_mask(data, tractor, maxshift_arcsec=3.5, niter=2,
                         use_mge=False, qaplot=True):#False):
    """Wrapper to mask out all sources except the galaxy we want to
    ellipse-fit.

    """
    from SGA.geometry import in_ellipse_mask
    from SGA.find_galaxy import find_galaxy
    from SGA.dust import SFDMap, mwdust_transmission
    from SGA.util import ivar2var


    def make_sourcemask(srcs, wcs, band, psf, sigma=None, nsigma=1.5):
        """Build a model image and threshold mask from a table of
        Tractor sources; also optionally subtract that model from an
        input image.

        """
        from scipy.ndimage.morphology import binary_dilation
        from SGA.coadds import srcs2image

        model = srcs2image(srcs, wcs, band=band.lower(), pixelized_psf=psf)
        if sigma:
            mask = model > nsigma*sigma # True=significant flux
            mask = binary_dilation(mask*1, iterations=2) > 0
        else:
            mask = np.zeros(model.shape, bool)

        return mask, model


    def find_galaxy_in_cutout(img, bx, by, diam, ba, pa, fraction=0.5,
                              factor=1., skysigma=None, use_mge=False):
        """Run find_galaxy in a cutout centered on the object of
        interest.

        """
        W = img.shape[1]

        x1 = int(bx-factor*diam)
        x2 = int(bx+factor*diam)
        y1 = int(by-factor*diam)
        y2 = int(by+factor*diam)

        if x1 < 0:
            x1 = 0
        if x2 > W:
            x2 = W
        if y1 < 0:
            y1 = 0
        if y2 > W:
            y2 = W
        cutout = img[y1:y2, x1:x2]

        debug = False#True
        if use_mge:
            if debug:
                mge = find_galaxy(cutout, nblob=1, binning=3, fraction=fraction, quiet=False, plot=True)
                import matplotlib.pyplot as plt
                plt.clf()
                mge = find_galaxy(cutout, nblob=1, binning=3, fraction=0.75, quiet=False, plot=True)
                plt.savefig('ioannis/tmp/junk.png')
                plt.close()
            else:
                mge = find_galaxy(cutout, nblob=1, binning=1, fraction=fraction,
                                  quiet=True)
                mge.xmed += x1
                mge.ymed += y1
                mge.xpeak += x1
                mge.ypeak += y1
            return mge
        else:
            #from photutils.morphology import gini
            from SGA.geometry import EllipseProperties

            mask = cutout != 0.
            if skysigma is not None:
                mask = np.logical_and(mask, cutout > 5.*skysigma)

            P = EllipseProperties()
            perc = 0.95
            method = 'percentile'
            #method = 'rms'
            P.fit(cutout, mask=mask, method=method, percentile=perc, smooth_sigma=0.)

            if debug:
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots()
                P.plot(image=np.log10(cutout), ax=ax)
                fig.savefig('ioannis/tmp/junk.png')
                plt.close()
                #pdb.set_trace()

            P.x0 += x1
            P.y0 += y1

            return P


    def get_geometry(pixscale, pixfactor=1., table=None, tractor=None, mge=None,
                     props=None):
        """Extract elliptical geometry from either an astropy Table
        (sample), a tractor catalog, or an mge object.

        """
        if table is not None:
            bx, by = table['BX_INIT'], table['BY_INIT']
            diam = table['DIAM_INIT'] / pixscale # [pixels]
            ba = table['BA_INIT']                # [pixels]
            pa = table['PA_INIT']
        elif tractor is not None:
            from SGA.geometry import get_tractor_ellipse
            (bx, by) = tractor.bx, tractor.by
            diam = 2. * tractor.shape_r / pixscale # [pixels]
            _, ba, pa = get_tractor_ellipse(diam, tractor.shape_e1, tractor.shape_e2)
        elif props is not None:
            bx = props.x0
            by = props.y0
            diam = 2. * props.a # [pixels]
            ba = props.ba
            pa = props.pa
        elif mge is not None:
            bx = mge.ymed # NB bx-->ymed!
            by = mge.xmed
            semia = 1.2 * mge.majoraxis # [pixels]
            ba = (1. - mge.eps)
            pa = mge.pa

        bx *= pixfactor
        by *= pixfactor

        return np.array([bx, by, diam, ba, pa])


    def update_galmask(allgalsrcs, bx, by, diam, ba, pa, opt_models=None,
                       mask_allgals=False):
        """Update the galaxy mask based on the current in-ellipse array.

        """
        opt_galmask = np.zeros(sz, bool)
        if len(allgalsrcs) > 0:
            # optionally subtract all galaxies, irrespective of their location
            if mask_allgals:
                I = np.zeros(len(allgalsrcs), bool)
            else:
                I = in_ellipse_mask(bx, width-by, diam/2., ba*diam/2., pa,
                                    allgalsrcs.bx, width-allgalsrcs.by)
            if np.sum(~I) > 0:
                galsrcs = allgalsrcs[~I]
                for iband, filt in enumerate(opt_bands):
                    msk, model = make_sourcemask(
                        galsrcs, opt_wcs, filt, data[f'{filt}_psf'],
                        data[f'{filt}_sigma'])
                    opt_galmask = np.logical_or(opt_galmask, msk)
                    if opt_models is not None:
                        opt_models[iband, :, :] += model
            else:
                galsrcs = None
        else:
            galsrcs = None

        return galsrcs, opt_galmask, opt_models


    all_bands = data['all_bands']
    opt_bands = data['opt_bands']
    opt_refband = data['opt_refband']
    opt_pixscale = data['opt_pixscale']
    opt_wcs = data['opt_wcs']

    sz = data[opt_refband].shape
    width = sz[0]
    assert(width == sz[1])
    xgrid, ygrid = np.meshgrid(np.arange(width),
                               np.arange(width),
                               indexing='xy')
    ygrid_flip = width - ygrid

    sample = data['sample']
    samplesrcs = data['samplesrcs']
    nsample = len(sample)

    #pdb.set_trace()
    #skysig = [sample[f'PSFDEPTH_{filt.upper()}'] for filt in opt_bands]

    Ipsf = ((tractor.type == 'PSF') * (tractor.type != 'DUP') *
            (tractor.ref_cat != REFCAT) * (tractor.ref_cat != 'LG') *
            np.logical_or(tractor.ref_cat == 'GE', tractor.ref_cat == 'G3'))
    Igal = ((tractor.type != 'PSF') * (tractor.type != 'DUP') *
            (tractor.ref_cat != REFCAT) * (tractor.ref_cat != 'LG'))
    psfsrcs = tractor[Ipsf]
    allgalsrcs = tractor[Igal]

    # Initialize the *original* images arrays.
    opt_images = np.zeros((len(opt_bands), *sz), 'f4')
    opt_mask_perband = np.stack([data[f'{filt}_mask'] for filt in opt_bands])
    opt_weight = np.stack([data[f'{filt}_invvar'] for filt in opt_bands])

    opt_maskbits = np.zeros((nsample, *sz), np.int32)
    opt_images_final = np.zeros((nsample, len(opt_bands), *sz), 'f4')
    opt_models = np.zeros((nsample, len(opt_bands), *sz), 'f4')

    # Bright-star mask.
    opt_brightstarmask = data['brightstarmask']

    # Subtract Gaia stars from all optical images and generate the
    # threshold gaiamask.
    opt_gaiamask = np.zeros(sz, bool)
    for iband, filt in enumerate(opt_bands):
        if len(psfsrcs) > 0:
            msk, model = make_sourcemask(
                psfsrcs, opt_wcs, filt, data[f'{filt}_psf'],
                data[f'{filt}_sigma'])
            opt_models[:, iband, :, :] += model[np.newaxis, :, :]
            opt_images[iband, :, :] = data[filt] - model
            opt_gaiamask = np.logical_or(opt_gaiamask, msk)
        else:
            opt_images[iband, :, :] = data[filt]

    geo_initial = np.zeros((nsample, 5)) # [bx,by,diam,ba,pa]
    geo_final = np.zeros_like(geo_initial)

    for iobj, (obj, objsrc) in enumerate(zip(sample, samplesrcs)):
        log.info('Determining the geometry for galaxy ' + \
                 f'{iobj+1}/{nsample}.')

        # If the CLUSTER bit is set, mask all extended sources,
        # whether or not they're inside the elliptical mask.
        if obj['CLUSTER']:
            log.info('CLUSTER flag set; masking all extended sources.')
            mask_allgals = True
        else:
            mask_allgals = False

        # Find all reference sources (not dropped by Tractor) except
        # the one we're working on.
        refindx = np.delete(np.arange(nsample), iobj)
        refsrcs = samplesrcs[:iobj] + samplesrcs[iobj+1:]
        refsamples = sample[refindx]

        # For each *other* SGA source(s), subtract the model from the
        # optical images.
        opt_refmask = np.zeros(sz, bool)
        opt_images_obj = opt_images.copy() # reset the data
        for indx, refsrc, refsample in zip(refindx, refsrcs, refsamples):
            # for *previously* completed objects, use the final, not
            # initial geometry
            if (iobj > 0):
                [bx, by, diam, ba, pa] = geo_final[indx, :]
            else:
                [bx, by, diam, ba, pa] = \
                    get_geometry(opt_pixscale, table=refsample)
            opt_refmask1 = in_ellipse_mask(bx, width-by, diam/2., ba*diam/2.,
                                           pa, xgrid, ygrid_flip)
            opt_refmask = np.logical_or(opt_refmask, opt_refmask1)

            #import matplotlib.pyplot as plt
            #plt.clf()
            #plt.imshow(opt_refmask, origin='lower')
            #plt.savefig('ioannis/tmp/junk2.png')

            for iband, filt in enumerate(opt_bands):
                _, model = make_sourcemask(
                    refsrc, opt_wcs, filt, data[f'{filt}_psf'],
                    sigma=None)
                opt_images_obj[iband, :, :] = opt_images_obj[iband, :, :] - model
                opt_models[iobj, iband, :, :] += model

        # Initial geometry and elliptical mask.
        geo_init  = get_geometry(opt_pixscale, table=obj)
        geo_initial[iobj, :] = geo_init
        [bx, by, diam, ba, pa] = geo_init

        # Next, iteratively update the source geometry unless
        # FIXGEO has been set.
        if obj['FIXGEO']:
            niter_actual = 1
        else:
            niter_actual = niter

        for iiter in range(niter_actual):
            log.info(f'Iteration {iiter+1}/{niter_actual}')
            #print(iobj, iiter, bx, by, diam, ba, pa)

            # initialize (or update) the in-ellipse mask
            inellipse = in_ellipse_mask(bx, width-by, diam/2., ba*diam/2.,
                                        pa, xgrid, ygrid_flip)

            # Zero out bright-star and reference pixels within the
            # current ellipse mask of the current object...
            iter_brightstarmask = np.copy(opt_brightstarmask)
            iter_refmask = np.copy(opt_refmask)
            iter_brightstarmask[inellipse] = False
            iter_refmask[inellipse] = False

            # Build a galaxy mask from all extended sources outside
            # the (current) elliptical mask (but do not subtract the
            # models). By default, galaxy pixels inside the elliptical
            # mask, unless we're masking *all* galaxies (e.g., in
            # cluster fields).
            galsrcs, opt_galmask, _ = update_galmask(
                allgalsrcs, bx, by, diam, ba, pa,
                opt_models=None, mask_allgals=mask_allgals)
            if not mask_allgals:
                opt_galmask[inellipse] = False
                #import matplotlib.pyplot as plt
                #plt.clf()
                #plt.imshow(opt_galmask, origin='lower')
                #plt.savefig('ioannis/tmp/junk2.png')

            # Hack! If there are other reference sources, double the
            # opt_refmask inellipse veto mask so that the derived
            # geometry can grow, if necessary.
            if iobj > 0:
                inellipse2 = in_ellipse_mask(bx, width-by, 1.5*diam, ba*1.5*diam,
                                             pa, xgrid, ygrid_flip)
                iter_refmask[inellipse2] = False
            #import matplotlib.pyplot as plt
            #plt.clf()
            #plt.imshow(iter_refmask, origin='lower')
            #plt.savefig('ioannis/tmp/junk2.png')

            # Combine opt_brightstarmask, opt_gaiamask, opt_refmask,
            # and opt_galmask with the per-band optical masks.
            opt_masks_obj = _update_masks(iter_brightstarmask, opt_gaiamask,
                                          iter_refmask, opt_galmask,
                                          opt_mask_perband, opt_bands,
                                          sz, verbose=False)

            # Optionally update the geometry from the masked, coadded
            # optical image.
            if obj['FIXGEO']:
                log.info('FIXGEO flag set; not updating geometry.')
                geo_iter = geo_init
            else:
                wimg = np.sum(opt_weight * np.logical_not(opt_masks_obj) * opt_images_obj, axis=0)
                if use_mge:
                    mge = find_galaxy_in_cutout(wimg, bx, by, diam, ba, pa)
                    geo_iter = get_geometry(opt_pixscale, mge=mge)
                else:
                    skysig = min([obj[f'PSFDEPTH_{filt.upper()}'] for filt in opt_bands])
                    skysig = 1e9*10.**(-0.4*(skysig)) # AB mag-->nanomaggies
                    props = find_galaxy_in_cutout(wimg, bx, by, diam, ba, pa,
                                                  skysigma=skysig)
                    geo_iter = get_geometry(opt_pixscale, props=props)

            dshift_arcsec = opt_pixscale * np.hypot(geo_init[0]-geo_iter[0], geo_init[1]-geo_iter[1])
            if dshift_arcsec > maxshift_arcsec:
                log.warning(f'Large shift for iobj={iobj} ({obj[REFIDCOLUMN]}): delta=' + \
                            f'{dshift_arcsec:.3f}>{maxshift_arcsec:.3f} arcsec')
                sample['LARGESHIFT'][iobj] = True
                # revert to the Tractor position
                if objsrc is not None:
                    geo_iter[0] = objsrc.bx
                    geo_iter[1] = objsrc.by
                else:
                    geo_iter[0] = geo_init[0]
                    geo_iter[1] = geo_init[1]

            # update the geometry for the next iteration
            [bx, by, diam, ba, pa] = geo_iter
            #print(iobj, iiter, bx, by, diam, ba, pa)

        # Set the blended bit and (final) dshift.
        if len(refsamples) > 0:
            Iclose = in_ellipse_mask(bx, width-by, diam/2., ba*diam/2., pa,
                                     refsamples['BX_INIT'],
                                     width-refsamples['BY_INIT'])
            if np.any(Iclose):
                sample['BLENDED'][iobj] = True

        sample['DSHIFT'][iobj] = opt_pixscale * np.hypot(
            geo_init[0]-geo_iter[0], geo_init[1]-geo_iter[1])

        # final images and geometry
        opt_images_final[iobj, :, :, :] = opt_images_obj
        geo_final[iobj, :] = geo_iter # last iteration

        # final masks
        inellipse = in_ellipse_mask(bx, width-by, diam/2., ba*diam/2.,
                                    pa, xgrid, ygrid_flip)
        final_brightstarmask = np.copy(opt_brightstarmask)
        final_refmask = np.copy(opt_refmask)
        final_brightstarmask[inellipse] = False
        final_refmask[inellipse] = False

        _, opt_galmask, opt_models_obj = update_galmask(
            allgalsrcs, bx, by, diam, ba, pa,
            opt_models=opt_models[iobj, :, :, :],
            mask_allgals=mask_allgals)
        if not mask_allgals:
            opt_galmask[inellipse] = False

        if iobj > 0:
            inellipse2 = in_ellipse_mask(bx, width-by, 1.5*diam, ba*1.5*diam,
                                         pa, xgrid, ygrid_flip)
            final_refmask[inellipse2] = False

        opt_maskbits_obj = _update_masks(final_brightstarmask, opt_gaiamask, final_refmask,
                                         opt_galmask, opt_mask_perband, opt_bands,
                                         sz, build_maskbits=True, MASKDICT=OPTMASKBITS)
        opt_models[iobj, :, :, :] = opt_models_obj
        opt_maskbits[iobj, :, :] = opt_maskbits_obj

        #import matplotlib.pyplot as plt
        #plt.clf()
        #plt.imshow(np.log10(opt_images_final[iobj, 0, :, :]*(opt_maskbits[iobj, :, :]==0)), origin='lower')
        #plt.savefig('ioannis/tmp/junk.png')
        #plt.close()

    # Update the data dictionary.
    data['opt_images'] = opt_images_final # [nanomaggies]
    data['opt_maskbits'] = opt_maskbits
    data['opt_models'] = opt_models
    data['opt_variance'] = ivar2var(opt_weight) # [nanomaggies**2]

    # Next, process the GALEX and unwise images:
    # --subtract Gaia stars
    # --subtract reference sources
    # --apply opt_maskbits
    for prefix, MASKDICT in zip(['unwise', 'galex'],
                                [UNWISEMASKBITS, GALEXMASKBITS]):
        bands = data[f'{prefix}_bands']
        refband = data[f'{prefix}_refband']
        wcs = data[f'{prefix}_wcs']
        sz = data[refband].shape

        images = np.zeros((len(bands), *sz), 'f4')
        images_final = np.zeros((nsample, len(bands), *sz), 'f4')

        # Subtract Gaia stars.
        for iband, filt in enumerate(bands):
            if len(psfsrcs) > 0:
                _, model = make_sourcemask(
                    psfsrcs, wcs, filt, data[f'{filt}_psf'],
                    sigma=None)
                images[iband, :, :] = data[filt] - model
            else:
                images[iband, :, :] = data[filt]

        # Subtract other reference sources.
        for iobj, (obj, objsrc) in enumerate(zip(sample, samplesrcs)):
            refsrcs = samplesrcs[:iobj] + samplesrcs[iobj+1:]
            refsamples = sample[np.delete(np.arange(nsample), iobj)]
            if len(refsamples) > 0:
                for refsrc, refsample in zip(refsrcs, refsamples):
                    for iband, filt in enumerate(bands):
                        _, model = make_sourcemask(
                            refsrc, wcs, filt, data[f'{filt}_psf'],
                            sigma=None)
                        images_final[iobj, iband, :, :] = images[iband, :, :] - model
            else:
                images_final[iobj, :, :, :] = images

        # Populate maskbits.
        maskbits = np.zeros((nsample, *sz), np.int32)

        mask_perband = np.stack([data[f'{filt}_mask'] for filt in bands])
        for iobj, obj in enumerate(sample):
            brightstarmask = opt_maskbits[iobj, :, :] & OPTMASKBITS['brightstar'] != 0
            refmask = opt_maskbits[iobj, :, :] & OPTMASKBITS['reference'] != 0
            gaiamask = opt_maskbits[iobj, :, :] & OPTMASKBITS['gaiastar'] != 0
            galmask = opt_maskbits[iobj, :, :] & OPTMASKBITS['galaxy'] != 0

            maskbits[iobj, :, :] = _update_masks(
                brightstarmask, gaiamask, refmask, galmask, mask_perband,
                bands, sz, build_maskbits=True, MASKDICT=MASKDICT,
                do_resize=True)

            #import matplotlib.pyplot as plt
            #plt.clf()
            #plt.imshow(maskbits[iobj, :, :], origin='lower')
            #plt.savefig('ioannis/tmp/junk.png')
            #plt.close()

        data[f'{prefix}_images'] = images_final # [nanomaggies]
        data[f'{prefix}_maskbits'] = maskbits
        ivar = np.stack([data[f'{filt}_invvar'] for filt in bands])
        data[f'{prefix}_variance'] = ivar2var(ivar) # [nanomaggies**2]


    # convert to surface brightness
    for prefix in ['opt', 'unwise', 'galex']:
        pixscale = data[f'{prefix}_pixscale']
        data[f'{prefix}_images'] /= pixscale**2   # [nanomaggies/arcsec**2]
        data[f'{prefix}_variance'] /= pixscale**4 # [nanomaggies**2/arcsec**4]

    # final geometry
    ra, dec = opt_wcs.wcs.pixelxy2radec((geo_final[:, 0]+1.), (geo_final[:, 1]+1.))
    for icol, col in enumerate(['BX_MOMENT', 'BY_MOMENT', 'DIAM_MOMENT', 'BA_MOMENT', 'PA_MOMENT']):
        sample[col] = geo_final[:, icol].astype('f4')

    sample['RA_MOMENT'] = ra
    sample['DEC_MOMENT'] = dec
    print('NEED TO ADD MW_TRANSMISSION!')

    pdb.set_trace()
    data['sample'] = sample # updated

    # optionally build a QA figure
    if qaplot:
        qa_multiband_mask(data, geo_initial, geo_final)

    # clean-up
    del data['samplesrcs']
    del data['brightstarmask']

    for prefix in ['opt', 'unwise', 'galex']:
        del data[f'{prefix}_wcs']

    for filt in all_bands:
        del data[filt]
        for col in ['psf', 'invvar', 'mask']:
            del data[f'{filt}_{col}']

    for filt in opt_bands:
        for col in ['sigma']:
            del data[f'{filt}_{col}']

    return data


def read_multiband(galaxy, galaxydir, sort_by_flux=True, bands=['g', 'r', 'i', 'z'],
                   pixscale=0.262, galex_pixscale=1.5, unwise_pixscale=2.75,
                   galex=False, unwise=False, verbose=False):
    """Read the multi-band images (converted to surface brightness) in
    preparation for ellipse-fitting.

    """
    import fitsio
    from astropy.table import Table
    from astrometry.util.fits import fits_table
    from astrometry.util.util import Tan
    from legacypipe.bits import MASKBITS

    # Dictionary mapping between optical filter and filename coded up in
    # coadds.py, galex.py, and unwise.py, which depends on the project.
    data = {}
    data['galaxy'] = galaxy
    data['galaxydir'] = galaxydir

    all_opt_bands = bands
    all_bands = np.copy(bands)

    #opt_pixscale = pixscale

    galex_bands, unwise_bands = None, None
    galex_refband, unwise_refband = None, None

    filt2imfile = {}
    for band in bands:
        filt2imfile.update({band: {'image': 'image',
                                   'model': 'model',
                                   'invvar': 'invvar',
                                   'psf': 'psf',}})
    filt2imfile.update({'tractor': 'tractor',
                        'sample': 'sample',
                        'maskbits': 'maskbits',})

    if unwise:
        unwise_bands = ['W1', 'W2', 'W3', 'W4']
        all_bands = np.append(all_bands, unwise_bands)
        unwise_refband = unwise_bands[0]
        for band in unwise_bands:
            filt2imfile.update({band: {'image': 'image',
                                       'model': 'model',
                                       'invvar': 'invvar',
                                       'psf': 'psf'}})

    if galex:
        galex_bands = ['FUV', 'NUV']
        all_bands = np.append(all_bands, galex_bands)
        galex_refband = galex_bands[1]
        for band in galex_bands:
            filt2imfile.update({band: {'image': 'image',
                                       'model': 'model',
                                       'invvar': 'invvar',
                                       'psf': 'psf'}})

    # Need to differentiate between missing one or more data products,
    # which indicates something went wrong with the previous (coadds)
    # stage vs missing all the data in a given bandpass, which is OK.
    opt_bands, opt_refband = [], None
    for filt in all_bands:
        datacount = 0
        for ii, imtype in enumerate(filt2imfile[filt].keys()):
            imfile = os.path.join(galaxydir, f'{galaxy}-{filt2imfile[filt][imtype]}-{filt}.fits.fz')
            if os.path.isfile(imfile):
                filt2imfile[filt][imtype] = imfile
                datacount += 1
            #else:
            #    log.warning(f'Missing {imfile}')

        if datacount == 0:
            pass
        else:
            if datacount == len(filt2imfile[filt].keys()):
                # opt_refband can be the first optical band
                if filt in all_opt_bands:
                    opt_bands.append(filt)
                    if opt_refband is None:
                        opt_refband = filt
            else:
                msg = f'Missing one or more {filt}-band data products!'
                log.critical(msg)
                return {}

    # update all_bands
    all_bands = opt_bands
    if unwise:
        all_bands = np.hstack((all_bands, unwise_bands))
    if galex:
        all_bands = np.hstack((all_bands, galex_bands))
    log.info(f'Found complete data in bands: {",".join(all_bands)}')

    # Pack some preliminary info into the output dictionary.
    data['all_bands'] = all_bands
    data['opt_bands'] = opt_bands
    data['galex_bands'] = galex_bands
    data['unwise_bands'] = unwise_bands

    data['opt_refband'] = opt_refband
    data['galex_refband'] = galex_refband
    data['unwise_refband'] = unwise_refband

    data['opt_pixscale'] = pixscale
    data['galex_pixscale'] = galex_pixscale
    data['unwise_pixscale'] = unwise_pixscale

    # We ~have~ to read the tractor catalog using fits_table because we will
    # turn these catalog entries into Tractor sources later.
    tractorfile = os.path.join(galaxydir, f'{galaxy}-{filt2imfile["tractor"]}.fits')

    cols = ['ra', 'dec', 'bx', 'by', 'type', 'ref_cat', 'ref_id',
            'sersic', 'shape_r', 'shape_e1', 'shape_e2']
    cols += [f'flux_{filt}' for filt in opt_bands]
    cols += [f'flux_ivar_{filt}' for filt in opt_bands]
    cols += [f'nobs_{filt}' for filt in opt_bands]
    cols += [f'mw_transmission_{filt}' for filt in opt_bands]
    cols += [f'psfdepth_{filt}' for filt in opt_bands]
    cols += [f'psfsize_{filt}' for filt in opt_bands]
    if galex:
        cols += [f'flux_{filt.lower()}' for filt in galex_bands]
        cols += [f'flux_ivar_{filt.lower()}' for filt in galex_bands]
        # after https://github.com/legacysurvey/legacypipe/issues/751 is addressed
        #cols += [f'psfdepth_{filt}' for filt in galex_bands]
    if unwise:
        cols += [f'flux_{filt.lower()}' for filt in unwise_bands]
        cols += [f'flux_ivar_{filt.lower()}' for filt in unwise_bands]
        cols += [f'psfdepth_{filt}' for filt in unwise_bands]

    tractor = fits_table(tractorfile, columns=cols)
    log.info(f'Read {len(tractor):,d} sources from {tractorfile}')

    # Read the sample catalog from custom_coadds and find each source
    # in the Tractor catalog.
    samplefile = os.path.join(galaxydir, f'{galaxy}-{filt2imfile["sample"]}.fits')
    cols = ['SGAID', 'SGANAME', 'OBJNAME', 'RA', 'DEC', 'DIAM', 'PA', 'BA', 'FITBIT']
    sample = Table(fitsio.read(samplefile, columns=cols))
    log.info(f'Read {len(sample)} source(s) from {samplefile}')
    for col in ['RA', 'DEC', 'DIAM', 'PA', 'BA']:
        sample.rename_column(col, f'{col}_INIT')

    sample['DIAM_INIT'] *= 60. # [arcsec]

    # populate (BX,BY)_INIT by quickly building the WCS
    wcs = Tan(filt2imfile[opt_refband]['image'], 1)
    (_, x0, y0) = wcs.radec2pixelxy(sample['RA_INIT'].value, sample['DEC_INIT'].value)
    sample['BX_INIT'] = (x0 - 1.).astype('f4') # NB the -1!
    sample['BY_INIT'] = (y0 - 1.).astype('f4')

    for filt in opt_bands:
        sample[f'PSFSIZE_{filt.upper()}'] = np.zeros(len(sample), 'f4')
    for filt in bands:
        sample[f'PSFDEPTH_{filt.upper()}'] = np.zeros(len(sample), 'f4')

    print('FIXME - special fitting bit(s)')
    sample['FIXGEO'] = sample['FITBIT'] & FITBITS['ignore'] != 0
    sample['FORCEPSF'] = sample['FITBIT'] & FITBITS['forcepsf'] != 0
    sample['CLUSTER'] = np.zeros(len(sample), bool)

    print('########### HACK! Setting cluster bit for III Zw 040 NOTES02!')
    if np.any(np.isin(sample['OBJNAME'], 'III Zw 040 NOTES02')):
        sample['CLUSTER'] = True

    sample['FLUX'] = np.zeros(len(sample), 'f4') # brightest band
    sample['DROPPED'] = np.zeros(len(sample), bool)
    sample['LARGESHIFT'] = np.zeros(len(sample), bool)
    sample['BLENDED'] = np.zeros(len(sample), bool)
    sample['DSHIFT'] = np.zeros(len(sample), 'f4')

    # moment geometry
    sample['RA_MOMENT'] = np.zeros(len(sample), 'f8')
    sample['DEC_MOMENT'] = np.zeros(len(sample), 'f8')
    sample['BX_MOMENT'] = np.zeros(len(sample), 'f4')
    sample['BY_MOMENT'] = np.zeros(len(sample), 'f4')
    sample['DIAM_MOMENT'] = np.zeros(len(sample), 'f4')
    sample['BA_MOMENT'] = np.zeros(len(sample), 'f4')
    sample['PA_MOMENT'] = np.zeros(len(sample), 'f4')

    samplesrcs = []
    for iobj, refid in enumerate(sample[REFIDCOLUMN].value):
        I = np.where((tractor.ref_cat == REFCAT) * (tractor.ref_id == refid))[0]
        if len(I) == 0:
            log.warning(f'ref_id={refid} dropped by Tractor')
            sample['DROPPED'][iobj] = True
            samplesrcs.append(None)
        else:
            samplesrcs.append(tractor[I])
            if tractor[I[0]].type in ['PSF', 'DUP']:
                log.warning(f'ref_id={refid} fit by Tractor as PSF (or DUP)')
                #sample['PSF'][iobj] = True
            sample['FLUX'][iobj] = max([getattr(tractor[I[0]], f'flux_{filt}')
                                        for filt in opt_bands])

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
    _get_psfsize_and_depth(sample, tractor, bands,
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
    for filt in opt_bands:
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
    data = read_image_data(data, filt2imfile, verbose=verbose)
    data = build_multiband_mask(data, tractor, qaplot=True)

    return data
