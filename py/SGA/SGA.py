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
        bands += ['FUV', 'NUV']
    if unwise:
        bands += ['W1', 'W2', 'W3', 'W4']
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


def _get_psfsize_and_depth(tractor, bands, pixscale, incenter=False):
    """Support function for read_multiband. Compute the average PSF size (in arcsec)
    and depth (in 5-sigma AB mags) in each bandpass based on the Tractor
    catalog.

    """
    out = {}

    # Optionally choose sources in the center of the field.
    H = np.max(tractor.bx) - np.min(tractor.bx)
    W = np.max(tractor.by) - np.min(tractor.by)
    if incenter:
        dH = 0.1 * H
        these = np.where((tractor.bx >= int(H / 2 - dH)) * (tractor.bx <= int(H / 2 + dH)) *
                         (tractor.by >= int(H / 2 - dH)) * (tractor.by <= int(H / 2 + dH)))[0]
    else:
        #these = np.where(tractor.get(psfdepthcol) > 0)[0]
        these = np.arange(len(tractor))

    # Get the average PSF size and depth in each bandpass.
    for filt in bands:
        psfsizecol = f'psfsize_{filt.lower()}'
        psfdepthcol = f'psfdepth_{filt.lower()}'
        if psfsizecol in tractor.columns():
            good = np.where(tractor.get(psfsizecol)[these] > 0)[0]
            if len(good) == 0:
                log.warning(f'  No good measurements of the PSF size in band {filt}!')
                out[f'psfsigma_{filt.lower()}'] = np.float32(0.0)
                out[f'psfsize_{filt.lower()}'] = np.float32(0.0)
            else:
                # Get the PSF size and image depth.
                psfsize = tractor.get(psfsizecol)[these][good]   # [FWHM, arcsec]
                psfsigma = psfsize / np.sqrt(8 * np.log(2)) / pixscale # [sigma, pixels]

                out[f'psfsigma_{filt.lower()}'] = np.median(psfsigma).astype('f4')
                out[f'psfsize_{filt.lower()}'] = np.median(psfsize).astype('f4')

        if psfsizecol in tractor.columns():
            good = np.where(tractor.get(psfdepthcol)[these] > 0)[0]
            if len(good) == 0:
                log.warning(f'  No good measurements of the PSF depth in band {filt}!')
                out[f'psfdepth_{filt.lower()}'] = np.float32(0.0)
            else:
                psfdepth = tractor.get(psfdepthcol)[these][good] # [AB mag, 5-sigma]
                out[f'psfdepth_{filt.lower()}'] = (22.5-2.5*np.log10(1/np.sqrt(np.median(psfdepth)))).astype('f4')

    return out


def _read_image_data(data, filt2imfile, starmask=None, allmask=None,
                     fill_value=0.0, filt2pixscale=None, verbose=False):
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

    fit_bands = data['fit_bands']
    fit_optical_bands = data['fit_optical_bands']

    vega2ab = {'W1': 2.699, 'W2': 3.339, 'W3': 5.174, 'W4': 6.620}

    # Loop on each filter and return the masked data.
    residual_mask = None
    for filt in fit_bands:
        # Read the data and initialize the mask with the inverse
        # variance image.
        if verbose:
            log.info(f'Reading {filt2imfile[filt]["image"]}')
            log.info(f'Reading {filt2imfile[filt]["model"]}')
        image = fitsio.read(filt2imfile[filt]['image'])
        hdr = fitsio.read_header(filt2imfile[filt]['image'], ext=1)
        model = fitsio.read(filt2imfile[filt]['model'])

        # add the header to the data dictionary
        data[f'{filt.lower()}_header'] = hdr

        # Initialize the mask based on the inverse variance
        if 'invvar' in filt2imfile[filt].keys():
            if verbose:
                log.info(f'Reading {filt2imfile[filt]["invvar"]}')
            invvar = fitsio.read(filt2imfile[filt]['invvar'])
            mask = invvar <= 0 # True-->bad, False-->good
        else:
            invvar = None
            mask = np.zeros_like(image).astype(bool)

        # convert WISE images from Vega nanomaggies to AB nanomaggies
        # https://www.legacysurvey.org/dr9/description/#photometry
        if filt.lower() in ['w1', 'w2', 'w3', 'w4']:
            image *= 10**(-0.4*vega2ab[filt])
            model *= 10**(-0.4*vega2ab[filt])
            if invvar is not None:
                invvar /= (10**(-0.4*vega2ab[filt]))**2

        sz = image.shape

        # GALEX, unWISE need to be resized. Never resize allmask, if present.
        if starmask is not None:
            if starmask.shape == sz:
                doresize = False
            else:
                doresize = True

        if filt in fit_optical_bands:
            HH, WW = sz
            data['width'] = WW
            data['height'] = HH

        # Retrieve the PSF and WCS.
        if verbose:
            log.info(f'Reading {filt2imfile[filt]["psf"]}')
        psfimg = fitsio.read(filt2imfile[filt]['psf'])
        psfimg /= psfimg.sum()
        data[f'{filt.lower()}_psf'] = PixelizedPSF(psfimg)

        wcs = Tan(filt2imfile[filt]['image'], 1)
        if 'MJD_MEAN' in hdr:
            mjd_tai = hdr['MJD_MEAN'] # [TAI]
            wcs = LegacySurveyWcs(wcs, TAITime(None, mjd=mjd_tai))
        else:
            wcs = ConstantFitsWcs(wcs)
        data[f'{filt.lower()}_wcs'] = wcs

        # Add in the starmask, resizing if necessary for this image/pixel
        # scale. Never resize allmask (it's only for the optical).
        if starmask is not None:
            if doresize:
                _starmask = resize(starmask, mask.shape, mode='edge',
                                   anti_aliasing=False) > 0
                mask = np.logical_or(mask, _starmask)
            else:
                mask = np.logical_or(mask, starmask)
                if allmask is not None:
                    mask = np.logical_or(mask, allmask)

        # Flag significant residual pixels after subtracting *all* the models
        # (we will restore the pixels of the galaxies of interest later). Only
        # consider the optical (grz) bands here.
        resid = gaussian_filter(image - model, 2.0)
        _, _, sig = sigma_clipped_stats(resid, sigma=3.0)
        data[f'{filt.lower()}_sigma'] = sig
        if residual_mask is None:
            residual_mask = np.abs(resid) > 5*sig
        else:
            _residual_mask = np.abs(resid) > 5*sig
            # In grz, use a cumulative residual mask. In UV/IR use an
            # individual-band mask.
            if doresize:
                pass
                #residual_mask = resize(_residual_mask, residual_mask.shape, mode='reflect')
            else:
                residual_mask = np.logical_or(residual_mask, _residual_mask)

        ## Dilate the mask, mask out a 10% border, and pack into a dictionary.
        mask = binary_dilation(mask, iterations=2)
        edge = int(0.02*sz[0])
        mask[:edge, :] = True
        mask[:, :edge] = True
        mask[:, sz[0]-edge:] = True
        mask[sz[0]-edge:, :] = True
        data[filt] = ma.masked_array(image, mask) # [nanomaggies]
        ma.set_fill_value(data[filt], fill_value)

        #if filt == 'W1':
        #    import matplotlib.pyplot as plt
        #    plt.clf() ; plt.imshow(mask, origin='lower') ; plt.savefig('desi-users/ioannis/tmp/junk-mask-{}.png'.format(filt))
        if invvar is not None:
            var = np.zeros_like(invvar)
            ok = invvar > 0
            var[ok] = 1 / invvar[ok]
            data[f'{filt.lower()}_var_'] = var # [nanomaggies**2]
            if np.any(invvar < 0):
                log.warning(f'Found {np.sum(invvar<0):,d} negative pixels in the ' + \
                            f'{filt}-band inverse variance map!')

    data['residual_mask'] = residual_mask
    if starmask is not None:
        data['starmask'] = starmask
    if allmask is not None:
        data['almask'] = allmask

    return data


def _build_multiband_mask(data, tractor, filt2pixscale, fill_value=0.0,
                          threshmask=0.01, r50mask=0.05, maxshift=0.0,
                          ref_cat='LG', sigmamask=3.0, neighborfactor=1.0,
                          verbose=False):
    """Wrapper to mask out all sources except the galaxy we want to ellipse-fit.

    r50mask - mask satellites whose r50 radius (arcsec) is > r50mask

    threshmask - mask satellites whose flux ratio is > threshmmask relative to
    the central galaxy.

    """
    import numpy.ma as ma
    from copy import copy
    from skimage.transform import resize
    from SGA.find_galaxy import find_galaxy
    from SGA.geometry import ellipse_mask
    from SGA.coadds import srcs2image
    from SGA.dust import SFDMap, mwdust_transmission

    import matplotlib.pyplot as plt
    from astropy.visualization import simple_norm

    refband = data['refband']
    fit_bands = data['fit_bands']

    #nbox = 5
    #box = np.arange(nbox)-nbox // 2
    #box = np.meshgrid(np.arange(nbox), np.arange(nbox))[0]-nbox//2

    xobj, yobj = np.ogrid[0:data['height'], 0:data['width']]
    dims = data[refband].shape
    assert(dims[0] == dims[1])

    # If the row-index of the central galaxy is not provided, use the source
    # nearest to the center of the field.
    if 'galaxy_indx' in data.keys():
        galaxy_indx = np.atleast_1d(data['galaxy_indx'])
    else:
        galaxy_indx = np.array([np.argmin((tractor.bx - data['height']/2)**2 +
                                          (tractor.by - data['width']/2)**2)])
        data['galaxy_indx'] = np.atleast_1d(galaxy_indx)
        data['galaxy_id'] = ''

    def tractor2mge(indx, factor=1.0):
        # Convert a Tractor catalog entry to an MGE object.
        class MGEgalaxy(object):
            pass

        if tractor.type[indx] == 'PSF' or tractor.shape_r[indx] < 5.:
            pa = tractor.pa_init[indx]
            ba = tractor.ba_init[indx]
            # take away the extra factor of 2 we put in in read_sample()
            r50 = tractor.diam_init[indx] * 60. / 2. / 2. # [arcsec]
            if r50 < 5:
                r50 = 5.0 # minimum size, arcsec
            majoraxis = factor * r50 / filt2pixscale[refband] # [pixels]
        else:
            ee = np.hypot(tractor.shape_e1[indx], tractor.shape_e2[indx])
            ba = (1. - ee) / (1. + ee)
            phi = -np.rad2deg(np.arctan2(tractor.shape_e2[indx], tractor.shape_e1[indx]) / 2.)
            pa = (180. - phi) % 180.
            #majoraxis = factor * tractor.shape_r[indx] / filt2pixscale[refband] # [pixels]

            # can be zero (or very small) if fit as a PSF or REX
            if tractor.shape_r[indx] > 1:
                majoraxis = factor * tractor.shape_r[indx] / filt2pixscale[refband] # [pixels]
            else:
                majoraxis = factor * tractor.diam_init[indx] * 60. / 2. / 2. / filt2pixscale[refband] # [pixels]

        mgegalaxy = MGEgalaxy()

        mgegalaxy.xmed = tractor.by[indx]
        mgegalaxy.ymed = tractor.bx[indx]
        mgegalaxy.xpeak = tractor.by[indx]
        mgegalaxy.ypeak = tractor.bx[indx]
        mgegalaxy.eps = 1. - ba
        mgegalaxy.pa = pa
        mgegalaxy.theta = (270. - pa) % 180.
        mgegalaxy.majoraxis = majoraxis

        objmask = ellipse_mask(mgegalaxy.xmed, mgegalaxy.ymed, # object pixels are True
                               mgegalaxy.majoraxis,
                               mgegalaxy.majoraxis * (1.-mgegalaxy.eps),
                               np.radians(mgegalaxy.theta-90.), xobj, yobj)

        return mgegalaxy, objmask

    # Now, loop through each 'galaxy_indx' from bright to faint.
    data['mge'] = []
    for ii, central in enumerate(galaxy_indx):
        log.info(f'Determing the geometry for galaxy {ii+1}/{len(galaxy_indx)}.')

        #if tractor.ref_cat[galaxy_indx] == 'R1' and tractor.ref_id[galaxy_indx] == 8587006103:
        #    neighborfactor = 1.0

        # [1] Determine the non-parametric geometry of the galaxy of interest
        # in the reference band. First, subtract all models except the galaxy
        # and galaxies "near" it. Also restore the original pixels of the
        # central in case there was a poor deblend.
        largeshift = False

        mge, centralmask = tractor2mge(central, factor=1.0)
        plt.clf() ; plt.imshow(centralmask, origin='lower') ; plt.savefig('ioannis/tmp/junk-mask.png')

        iclose = np.where([centralmask[int(by), int(bx)]
                           for by, bx in zip(tractor.by, tractor.bx)])[0]

        srcs = tractor.copy()
        srcs.cut(np.delete(np.arange(len(tractor)), iclose))
        model = srcs2image(srcs, data[f'{refband.lower()}_wcs'],
                           band=refband.lower(),
                           pixelized_psf=data[f'{refband.lower()}_psf'])

        img = data[refband].data - model
        img[centralmask] = data[refband].data[centralmask]

        # the "residual mask" is initialized in legacyhalos.io._read_image_data
        # and it includes pixels which are significant residuals (data minus
        # model), pixels with invvar==0, and pixels belonging to maskbits
        # BRIGHT, MEDIUM, CLUSTER, or ALLMASK_[GRZ]

        mask = np.logical_or(ma.getmask(data[refband]), data['residual_mask'])
        #mask = np.logical_or(data[refband].mask, data['residual_mask'])
        mask[centralmask] = False

        img = ma.masked_array(img, mask)
        ma.set_fill_value(img, fill_value)

        #mgegalaxy = find_galaxy(img, nblob=1, binning=1, quiet=False)#, plot=True) ; plt.savefig('desi-users/ioannis/tmp/debug.png')
        mgegalaxy = find_galaxy(img, nblob=1, binning=1, quiet=False, plot=True) ; plt.savefig('ioannis/tmp/junk-mge.png')

        # Did the galaxy position move? If so, revert back to the Tractor geometry.
        if np.abs(mgegalaxy.xmed-mge.xmed) > maxshift or np.abs(mgegalaxy.ymed-mge.ymed) > maxshift:
            log.warning(f'Large centroid shift! (x,y) = ({mgegalaxy.xmed:.3f},{mgegalaxy.ymed:.3f})-->' + \
                        f'({mge.xmed:.3f},{mge.ymed:.3f})')
            largeshift = True
            mgegalaxy = copy(mge)

        radec_med = data[f'{refband.lower()}_wcs'].pixelToPosition(
            mgegalaxy.ymed+1, mgegalaxy.xmed+1).vals
        #radec_peak = data[f'{refband.lower()}_wcs'].pixelToPosition(
        #    mgegalaxy.ypeak+1, mgegalaxy.xpeak+1).vals
        mge = {
            'largeshift': largeshift,
            'ra': tractor.ra[central], 'dec': tractor.dec[central],
            'bx': tractor.bx[central], 'by': tractor.by[central],
            'ra_moment': radec_med[0], 'dec_moment': radec_med[1],
            }

        # add the dust
        photsys = 'S'
        ebv = SFDMap().ebv(radec_med[0], radec_med[1])
        mge['ebv'] = np.float32(ebv)
        for band in data['bands']:
            print('TEMPORARILY SKIPPING MW_TRANSMISSION!!')
            #mge[f'mw_transmission_{band.lower()}'] = mwdust_transmission(
            #    ebv, band, photsys, match_legacy_surveys=True).astype('f4')
            mge[f'mw_transmission_{band.lower()}'] = np.float32(1.)

        for key in ['eps', 'majoraxis', 'pa', 'theta', 'xmed', 'ymed', 'xpeak', 'ypeak']:
            mge[key] = np.float32(getattr(mgegalaxy, key))
            if key == 'pa': # put into range [0-180]
                mge[key] = mge[key] % np.float32(180)
        data['mge'].append(mge)

        # [2] Create the satellite mask in all the bandpasses. Use
        # srcs here, which has had the satellites nearest to the
        # central galaxy trimmed out.
        log.info('Building the satellite mask.')
        satmask = np.zeros(data[refband].shape, bool)

        for filt in fit_bands:
            # do not let GALEX and WISE contribute to the satellite mask
            if data[filt].shape != satmask.shape:
                continue

            cenflux = getattr(tractor, f'flux_{filt.lower()}')[central]
            satflux = getattr(srcs, f'flux_{filt.lower()}')
            if cenflux <= 0.0:
                log.warning('Central galaxy flux is negative! Proceed with caution...')

            satindx = np.where(np.logical_or(
                (srcs.type != 'PSF') * (srcs.shape_r > r50mask) *
                (satflux > 0.0) * ((satflux / cenflux) > threshmask),
                srcs.ref_cat == ref_cat))[0]
            #satindx = np.where(srcs.ref_cat == ref_cat)[0]
            #if np.isin(central, satindx):
            #    satindx = satindx[np.logical_not(np.isin(satindx, central))]
            if len(satindx) == 0:
                #raise ValueError('All satellites have been dropped!')
                log.warning(f'Warning! All satellites have been dropped from band {filt}!')
            else:
                satsrcs = srcs.copy()
                satsrcs.cut(satindx)
                satimg = srcs2image(satsrcs, data[f'{filt.lower()}_wcs'],
                                    band=filt.lower(),
                                    pixelized_psf=data[f'{filt.lower()}_psf'])
                thissatmask = satimg > sigmamask*data[f'{filt.lower()}_sigma']
                #if filt == 'FUV':
                #    plt.clf() ; plt.imshow(thissatmask, origin='lower') ; plt.savefig('junk-{}.png'.format(filt.lower()))
                #    #plt.clf() ; plt.imshow(data[filt], origin='lower') ; plt.savefig('junk-{}.png'.format(filt.lower()))
                if satmask.shape != satimg.shape:
                    thissatmask = resize(thissatmask*1.0, satmask.shape, mode='reflect') > 0

                satmask = np.logical_or(satmask, thissatmask)
                #if True:
                #    import matplotlib.pyplot as plt
                ##    plt.clf() ; plt.imshow(np.log10(satimg), origin='lower') ; plt.savefig('debug.png')
                #    plt.clf() ; plt.imshow(satmask, origin='lower') ; plt.savefig('desi-users/ioannis/tmp/debug.png')
                ###    #plt.clf() ; plt.imshow(satmask, origin='lower') ; plt.savefig('/mnt/legacyhalos-data/debug.png')
            #print(filt, np.sum(satmask), np.sum(thissatmask))
        #plt.clf() ; plt.imshow(satmask, origin='lower') ; plt.savefig('junk-satmask.png')

        # [3] Build the final image (in each filter) for
        # ellipse-fitting. First, subtract out the PSF sources. Then
        # update the mask (but ignore the residual mask). Finally
        # convert to surface brightness.  for filt in ['W1']:
        for filt in fit_bands:
            thismask = ma.getmask(data[filt])
            if satmask.shape != thismask.shape:
                _satmask = (resize(satmask*1.0, thismask.shape, mode='reflect') > 0) == 1.0
                _centralmask = (resize(centralmask*1.0, thismask.shape, mode='reflect') > 0) == 1.0
                mask = np.logical_or(thismask, _satmask)
                mask[_centralmask] = False
            else:
                mask = np.logical_or(thismask, satmask)
                mask[centralmask] = False
            #if filt == 'W1':
            #    plt.imshow(_satmask, origin='lower') ; plt.savefig('junk-satmask-{}.png'.format(filt))
            #    plt.imshow(mask, origin='lower') ; plt.savefig('junk-mask-{}.png'.format(filt))

            varkey = f'{filt.lower()}_var'
            imagekey = f'{filt.lower()}_masked'
            psfimgkey = f'{filt.lower()}_psfimg'
            thispixscale = filt2pixscale[filt]
            if imagekey not in data.keys():
                data[imagekey], data[varkey], data[psfimgkey] = [], [], []

            img = ma.getdata(data[filt]).copy()

            # Get the PSF sources but ignore W3 and W4 (??)
            psfindx = np.where((getattr(tractor, f'flux_{filt.lower()}') / cenflux > threshmask) *
                               (tractor.type == 'PSF'))[0]
            if len(psfindx) > 0 and filt.upper() != 'W3' and filt.upper() != 'W4':
                psfsrcs = tractor.copy()
                psfsrcs.cut(psfindx)
            else:
                psfsrcs = None

            if psfsrcs:
                psfimg = srcs2image(psfsrcs, data[f'{filt.lower()}_wcs'],
                                    band=filt.lower(),
                                    pixelized_psf=data[f'{filt.lower()}_psf'])
                if False:#True:
                    #import fitsio ; fitsio.write('junk-psf-{}.fits'.format(filt.lower()), data['{}_psf'.format(filt.lower())].img, clobber=True)
                    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
                    im = ax1.imshow(np.log10(img), origin='lower') ; fig.colorbar(im, ax=ax1)
                    im = ax2.imshow(np.log10(psfimg), origin='lower') ; fig.colorbar(im, ax=ax2)
                    im = ax3.imshow(np.log10(data[f'{filt.lower()}_psf'].img), origin='lower') ; fig.colorbar(im, ax=ax3)
                    im = ax4.imshow(img-psfimg, origin='lower') ; fig.colorbar(im, ax=ax4)
                    plt.savefig(f'ioannis/tmp/qa-psf-{filt.lower()}.png')
                    if filt == 'r':
                        pass
                img -= psfimg
            else:
                psfimg = np.zeros((2, 2), 'f4')

            data[psfimgkey].append(psfimg)

            img = ma.masked_array((img / thispixscale**2).astype('f4'), mask) # [nanomaggies/arcsec**2]
            var = data[f'{filt.lower()}_var_'] / thispixscale**4 # [nanomaggies**2/arcsec**4]

            # Fill with zeros, for fun--
            ma.set_fill_value(img, fill_value)
            #if filt == 'r':# or filt == 'r':
            #    plt.clf() ; plt.imshow(img, origin='lower') ; plt.savefig('desi-users/ioannis/tmp/junk-img-{}.png'.format(filt.lower()))
            #    plt.clf() ; plt.imshow(mask, origin='lower') ; plt.savefig('desi-users/ioannis/tmp/junk-mask-{}.png'.format(filt.lower()))
            ##    plt.clf() ; plt.imshow(thismask, origin='lower') ; plt.savefig('desi-users/ioannis/tmp/junk-thismask-{}.png'.format(filt.lower()))

            data[imagekey].append(img)
            data[varkey].append(var)

        #test = data['r_masked'][0]
        #plt.clf() ; plt.imshow(np.log(test.clip(test[mgegalaxy.xpeak, mgegalaxy.ypeak]/1e4)), origin='lower') ; plt.savefig('/mnt/legacyhalos-data/debug.png')

    # Cleanup?
    for filt in fit_bands:
        del data[filt]
        del data[f'{filt.lower()}_var_']

    return data


def read_multiband(galaxy, galaxydir, bands=['g', 'r', 'i', 'z'],
                   pixscale=0.262, galex_pixscale=1.5, unwise_pixscale=2.75,
                   galaxy_id=None, galex=False, unwise=False, fill_value=0.0,
                   verbose=False):
    """Read the multi-band images (converted to surface brightness) and create a
    masked array suitable for ellipse-fitting.

    """
    import fitsio
    from astropy.table import Table
    import astropy.units as u
    from astrometry.util.fits import fits_table
    from legacypipe.bits import MASKBITS

    optical_bands = bands.copy()

    # Dictionary mapping between optical filter and filename coded up in
    # coadds.py, galex.py, and unwise.py, which depends on the project.
    data = {}
    data['galaxy'] = galaxy
    data['galaxydir'] = galaxydir

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
        for band in galex_bands:
            filt2imfile.update({band: {'image': 'image',
                                       'model': 'model',
                                       'invvar': 'invvar',
                                       'psf': 'psf'}})
            filt2pixscale.update({band: galex_pixscale})

    if unwise:
        unwise_bands = ['W1', 'W2', 'W3', 'W4']
        bands = bands + unwise_bands
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
    refband = None
    fit_bands, fit_optical_bands = [], []
    for filt in bands:
        datacount = 0
        for ii, imtype in enumerate(filt2imfile[filt].keys()):
            imfile = os.path.join(galaxydir, f'{galaxy}-{filt2imfile[filt][imtype]}-{filt}.fits.fz')
            if os.path.isfile(imfile):
                filt2imfile[filt][imtype] = imfile
                datacount += 1
            else:
                if verbose:
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
    data['failed'] = False # be optimistic!
    data['refband'] = refband
    data['bands'] = bands
    data['optical_bands'] = optical_bands
    data['fit_bands'] = fit_bands
    data['fit_optical_bands'] = fit_optical_bands
    #data['refband'] = refband
    data['refpixscale'] = np.float32(pixscale)

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
        cols += [f'flux_{filt}' for filt in ['fuv', 'nuv']]
        cols += [f'flux_ivar_{filt}' for filt in ['fuv', 'nuv']]
    if unwise:
        cols += [f'flux_{filt}' for filt in ['w1', 'w2', 'w3', 'w4']]
        cols += [f'flux_ivar_{filt}' for filt in ['w1', 'w2', 'w3', 'w4']]

    tractor = fits_table(tractorfile, columns=cols)
    log.info(f'Read {len(tractor):,d} sources from {tractorfile}')

    # read the sample catalog from custom_coadds
    samplefile = os.path.join(galaxydir, f'{galaxy}-{filt2imfile["sample"]}.fits')
    sample = Table(fitsio.read(samplefile))
    log.info(f'Read {len(sample)} source(s) from {samplefile}')

    # Find the reference source(s) in the Tractor catalog and sort by
    # optical brightness (in any band).
    print('!!!!!!!!!!! Need to add L4 REFCAT condition')
    data['sga_dropped'] = False
    data['sga_psf'] = False
    data['tractor_row'] = {}

    tractor_rows, fluxes = [], []
    for refid in sample[REFIDCOLUMN].value:
        tractor_row = np.where((tractor.ref_id == refid))[0]
        #tractor_row = np.where((tractor.ref_cat == REFCAT) * (tractor.ref_id == refid))[0]
        if len(tractor_row) == 0:
            log.warning(f'ref_id={refid} dropped by Tractor')
            data['sga_dropped'] = True
            tractor_row = -1
            fluxes.append(-99.)
        else:
            tractor_row = tractor_row[0]
            if (tractor.type[tractor_row] == 'PSF' or
                tractor.type[tractor_row] == 'DUP'):
                log.warning(f'ref_id={refid} fit by Tractor as PSF (or DUP)')
                data['sga_psf'] = True
            fluxes.append(max([getattr(tractor[tractor_row], f'flux_{filt.lower()}')
                               for filt in fit_optical_bands]))
        tractor_rows.append(tractor_row)
    fluxes = np.array(fluxes)
    tractor_rows = np.array(tractor_rows)

    log.info('Sorting by flux:')
    srt = np.argsort(fluxes)[::-1]
    for refid, tractor_row, flux in zip(sample[REFIDCOLUMN].value[srt],
                                        tractor_rows[srt], fluxes[srt]):
        log.info(f'  ref_id={refid} (row={tractor_row}): max optical flux={flux:.2f} nanomaggies')
        data['tractor_row'].update({refid: tractor_row})

    ## initial geometry
    #tractor.diam = np.zeros(len(tractor), dtype='f4')
    #tractor.pa = np.zeros(len(tractor), dtype='f4')
    #tractor.ba = np.zeros(len(tractor), dtype='f4')
    #if 'DIAM' in sample.colnames and 'PA' in sample.colnames and 'BA' in sample.colnames:
    #    tractor.diam[galaxy_indx] = sample['DIAM']
    #    tractor.pa[galaxy_indx] = sample['PA']
    #    tractor.ba[galaxy_indx] = sample['BA']

    # add the PSF depth and size
    data.update(_get_psfsize_and_depth(tractor, bands, pixscale, incenter=False))

    # Read the maskbits image and build the starmask.
    maskbitsfile = os.path.join(galaxydir, f'{galaxy}-{filt2imfile["maskbits"]}.fits.fz')
    if verbose:
        log.info(f'Reading {maskbitsfile}')
    maskbits = fitsio.read(maskbitsfile)
    # initialize the mask using the maskbits image
    starmask = ( (maskbits & MASKBITS['BRIGHT'] != 0) |
                 (maskbits & MASKBITS['MEDIUM'] != 0) |
                 (maskbits & MASKBITS['CLUSTER'] != 0) |
                 (maskbits & MASKBITS['ALLMASK_G'] != 0) |
                 (maskbits & MASKBITS['ALLMASK_R'] != 0) |
                 (maskbits & MASKBITS['ALLMASK_I'] != 0) |
                 (maskbits & MASKBITS['ALLMASK_Z'] != 0) )

    # Read the basic imaging data and masks and build the multiband
    # mask.
    data = _read_image_data(data, filt2imfile, starmask=starmask,
                            filt2pixscale=filt2pixscale,
                            fill_value=fill_value, verbose=verbose)

    data = _build_multiband_mask(data, tractor, filt2pixscale,
                                 fill_value=fill_value, verbose=verbose)

    #import matplotlib.pyplot as plt
    #plt.clf() ; plt.imshow(np.log10(data['g_masked'][0]), origin='lower') ; plt.savefig('junk1.png')
    ##plt.clf() ; plt.imshow(np.log10(data['r_masked'][1]), origin='lower') ; plt.savefig('junk2.png')
    ##plt.clf() ; plt.imshow(np.log10(data['r_masked'][2]), origin='lower') ; plt.savefig('junk3.png')
    #pdb.set_trace()

    ## Gather some additional info that we want propagated to the output ellipse
    ## catalogs.
    #allgalaxyinfo = []
    #for igal, (galaxy_id, galaxy_indx) in enumerate(zip(data['galaxy_id'], data['galaxy_indx'])):
    #    samp = sample[sample[REFIDCOLUMN] == galaxy_id]
    #    galaxyinfo = {REFIDCOLUMN: (str(galaxy_id), None)}
    #    allgalaxyinfo.append(galaxyinfo)

    return data
