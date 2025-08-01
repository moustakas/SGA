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
    uv_refband = data['uv_refband']
    ir_refband = data['ir_refband']
    fit_bands = data['fit_bands']
    fit_optical_bands = data['fit_optical_bands']

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
        if filt == refband or filt == uv_refband or filt == ir_refband:
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
            elif filt == uv_refband:
                data['uv_wcs'] = wcs
            elif filt == ir_refband:
                data['ir_wcs'] = wcs

        # convert WISE images from Vega nanomaggies to AB nanomaggies
        # https://www.legacysurvey.org/dr9/description/#photometry
        if filt.lower() in ['w1', 'w2', 'w3', 'w4']:
            image *= 10.**(-0.4 * VEGA2AB[filt])
            invvar /= (10.**(-0.4 * VEGA2AB[filt]))**2.
            model *= 10.**(-0.4 * VEGA2AB[filt])

        if verbose:
            log.info(f'Reading {filt2imfile[filt]["psf"]}')
        psfimg = fitsio.read(filt2imfile[filt]['psf'])
        psfimg /= psfimg.sum()
        data[f'{filt.lower()}_psf'] = PixelizedPSF(psfimg)

        # Generate a basic per-band mask, including allmask for the
        # optical bands and wisemask for the unwise bands. In the
        # optical bands, also mask XX% of the border.
        mask = invvar <= 0 # True-->bad

        if filt in fit_optical_bands:
            mask = np.logical_or(mask, data[f'allmask_{filt.lower()}'])
            del data[f'allmask_{filt.lower()}']

        # resize the wisemask, if present, but only for W1 and W2
        if data['wisemask'] is not None and filt.lower() in ['w1', 'w2']:
            _wisemask = resize(data['wisemask'], mask.shape, mode='edge',
                               anti_aliasing=False) > 0
            mask = np.logical_or(mask, _wisemask)

        # dilate
        mask = binary_dilation(mask, iterations=2)

        if filt in fit_optical_bands:
            edge = int(0.02*sz[0])
            mask[:edge, :] = True
            mask[:, :edge] = True
            mask[:, sz[0]-edge:] = True
            mask[sz[0]-edge:, :] = True

        # set invvar of masked pixels to zero and then get the robust
        # sigma from the masked residual image
        image[mask] = 0.
        invvar[mask] = 0.
        data[filt.lower()] = image # [nanomaggies]
        data[f'{filt.lower()}_invvar'] = invvar # [1/nanomaggies**2]
        data[f'{filt.lower()}_mask'] = mask
        print(filt, np.sum(mask))

        if filt in fit_optical_bands:
            resid = gaussian_filter(image - model, 2.)
            _, _, sig = sigma_clipped_stats(resid[~mask], sigma=2.5)
            data[f'{filt.lower()}_sigma'] = sig


    if 'wisemask' in data:
        del data['wisemask']

    # resize starmask to get the UV/IR starmask
    if ir_refband in fit_bands or uv_refband in fit_bands:
        uvir_starmask = resize(data['starmask'], data[refband].shape,
                               mode='edge', anti_aliasing=False) > 0
        data['uvir_starmask'] = uvir_starmask

    return data


def _tractor2mge(sample, tractor, filt2pixscale, band,
                 xgrid, ygrid, mindiam=10., factor=2.5):
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
    majoraxis = diam / filt2pixscale[band] # [pixels]
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


def _build_multiband_mask(data, tractor, maxshift=0.0, sigmamask=3.,
                          neighborfactor=1., verbose=False):
    """Wrapper to mask out all sources except the galaxy we want to ellipse-fit.

    r50mask - mask satellites whose r50 radius (arcsec) is > r50mask

    threshmask - mask satellites whose flux ratio is > threshmmask relative to
    the central galaxy.

    """
    from SGA.geometry import ellipse_mask
    from SGA.find_galaxy import find_galaxy
    from SGA.dust import SFDMap, mwdust_transmission


    def make_sourcemask(srcs, wcs, band, psf, sigma, nsigma=2.5, image=None):
        """Build a model image and threshold mask from a table of
        Tractor sources; also optionally subtract that model from an
        input image.

        """
        from SGA.coadds import srcs2image
        model = srcs2image(srcs, wcs, band=band.lower(), pixelized_psf=psf)
        mask = model > nsigma*sigma # True=significant flux
        if image is not None:
            image -= model
        return mask, image


    refband = data['refband']
    fit_bands = data['fit_bands']
    fit_optical_bands = data['fit_optical_bands']
    filt2pixscale = data['filt2pixscale']

    dims = data[refband].shape
    xgrid, ygrid = np.ogrid[0:dims[0], 0:dims[1]]
    assert(dims[0] == dims[1])

    # Generate "clean" optical images.
    opt_images = np.zeros((len(fit_optical_bands), dims[0], dims[1]), 'f4')
    opt_weight = np.zeros_like(opt_images)
    opt_masks = np.zeros(opt_images.shape, bool)

    for iband, filt in enumerate(fit_optical_bands):
        opt_weight[iband, :, :] = data[f'{filt.lower()}_invvar']
        opt_masks[iband, :, :] = data[f'{filt.lower()}_mask']

    # Loop through each reference source (already sorted from bright
    # to faint).
    sample = data['sample']
    nsample = len(sample)
    niter = 2

    opt_wcs = data['opt_wcs']

    Ipsf = ((tractor.type == 'PSF') * (tractor.type != 'DUP') *
            (tractor.ref_cat != REFCAT))
    Igal = ((tractor.type != 'PSF') * (tractor.type != 'DUP') *
            (tractor.ref_cat != REFCAT))

    # For each SGA source,
    #   --Build a mask from the initial or Tractor geometry.
    #
    #   --Subtract all PSFs, being careful about reference sources
    #     classified as PSF and systems where we used "force PSF".

    #   --Subtract all extended sources outside that mask.

    #   --If another "central" is inside the mask, subtract it and
    #      set the "blended" flag.

    #   --Iterate to convergence.

    qaplot = True

    if qaplot:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        from astropy.visualization import simple_norm

        qafile = os.path.join('/global/cfs/cdirs/desi/users/ioannis/tmp',
                              f'qa-ellipsemask-{data["galaxy"]}.png')

        ncol = 3
        nrow = 1 + nsample
        inches_per_panel = 3.
        fig, ax = plt.subplots(nrow, ncol,
                               figsize=(inches_per_panel*ncol,
                                        inches_per_panel*nrow),
                               gridspec_kw={'wspace': 0.02, 'hspace': 0.02},
                               constrained_layout=True)

        # coadded optical, UV, and IR images
        imgbands = [fit_optical_bands, ['FUV', 'NUV'], ['W1', 'W2', 'W3', 'W4']]
        for xx, bands in zip(ax[0, :], imgbands):
            wimgs = np.stack([data[filt.lower()] for filt in bands])
            wivars = np.stack([data[f'{filt.lower()}_invvar'] for filt in bands])
            wimg = np.sum(wivars * wimgs, axis=0)
            del wimgs, wivars

            norm = simple_norm(wimg, stretch='asinh', percent=99.5, asinh_a=0.1)
            xx.imshow(wimg, origin='lower', cmap='inferno', interpolation='none',
                      norm=norm)
            xx.set_xlim(0, wimg.shape[0])
            xx.set_ylim(0, wimg.shape[1])

    data['mge'] = []
    for iobj, obj in enumerate(sample):
        log.info('Determining the geometry for galaxy ' + \
                 f'{iobj+1}/{nsample}.')

        # Was this source dropped by Tractor (handled in
        # read_multiband)?
        if obj['DROPPED']:
            print('FIXME')
            src_central = None
        else:
            src_central = tractor[obj['ROW']]

        # Find all sources (not dropped by Tractor) except the one
        # we're working on.
        refsrcs, refobj = None, None
        I = (sample['ROW'] != -1) * (sample['ROW'] != obj['ROW'])
        if np.any(I):
            refsrcs = tractor[sample['ROW'][I]]

        I = (sample['ROW'] != obj['ROW'])
        if np.any(I):
            refobj = sample[I]

        # Build the initial mask (note refband).
        mge, centralmask = _tractor2mge(obj, src_central, filt2pixscale,
                                        refband, xgrid, ygrid, factor=1.0)

        # Iteratively determine the object geometry.
        for iiter in range(niter):
            log.info(f'  Iteration {iiter+1}/{niter}')

            opt_galmasks = np.zeros_like(opt_masks)
            opt_psfmasks = np.zeros_like(opt_masks)

            nearcentral = np.array([centralmask[int(by), int(bx)]
                                    for by, bx in zip(tractor.by, tractor.bx)])

            # Find PSFs; if we "forced PSF" (inside the SGA source,
            # e.g., a resolved dwarf like Phoenix) then we only want
            # to subtract sources *outside* the elliptical mask.
            psfsrcs = None
            if obj['FORCEPSF']:
                I = Ipsf * np.logical_not(nearcentral)
            else:
                I = Ipsf
            if np.any(I):
                psfsrcs = tractor[I]

            # Find extended sources *outside* the elliptical
            # mask. (There's no way to know at this point which
            # sources inside the elliptical mask are resolved
            # structure vs an independent object---FIXME!)
            galsrcs = None
            I = Igal * np.logical_not(nearcentral)
            if np.any(I):
                galsrcs = tractor[I]

            # If one or more *other* reference sources are inside the
            # mask, set the BLENDED bit. Note, we always reset the bit
            # to False because the mask can change, and we use
            # REFOBJ[BX,BY] in case a source is dropped by Tractor (in
            # which case BX,BY are set to BX_INIT,BY_INIT).
            sample['BLENDED'][iobj] = False # reset to False!
            if refobj:
                inmask = [centralmask[int(by), int(bx)]
                          for by, bx in zip(refobj['BY'], refobj['BX'])]
                if np.any(inmask):
                    sample['BLENDED'][iobj] = True

            #plt.clf()
            #plt.imshow(centralmask, origin='lower')
            #plt.scatter(psfsrcs.bx, psfsrcs.by, color='white')
            #plt.scatter(galsrcs.bx, galsrcs.by, s=5, color='red')
            #if refsrcs:
            #    plt.scatter(refsrcs.bx, refsrcs.by, color='blue', s=30)
            #plt.savefig('ioannis/tmp/junk.png')

            # Iterate on each optical image and subtract (and
            # threshold-mask) PSFs and threshold-mask extended
            # sources. Also optionally subtract other reference
            # sources (in the elliptical mask).
            for iband, filt in enumerate(fit_optical_bands):
                opt_images[iband, :, :] = data[filt]
                psf = data[f'{filt.lower()}_psf']
                sigma = data[f'{filt.lower()}_sigma']

                # Subtract PSFs and threshold mask.
                if psfsrcs:
                    msk, img = make_sourcemask(psfsrcs, opt_wcs, filt, psf, sigma,
                                               image=opt_images[iband, :, :])
                    opt_psfmasks[iband, :, :] = msk
                    opt_images[iband, :, :] = img

                # Threshold-mask (but do not subtract) extended
                # sources outside the mask.
                if galsrcs:
                    msk, _ = make_sourcemask(galsrcs, opt_wcs, filt, psf, sigma)
                    opt_galmasks[iband, :, :] = msk

                # Subtract reference sources (whether within or out of
                # the mask), but do not threshold mask.
                if refsrcs:
                    _, img = make_sourcemask(refsrcs, opt_wcs, filt, psf, sigma,
                                             image=opt_images[iband, :, :])
                    opt_images[iband, :, :] = img

            # Determine the geometry from the ivar-weighted, coadded,
            # optical image.
            wmasks = np.logical_not(np.logical_or.reduce((opt_masks, opt_psfmasks, opt_galmasks))) # True-->good
            wimg = np.sum(wmasks * opt_weight * opt_images, axis=0)
            #wivar = np.sum(opt_weight, axis=0)

            mge = find_galaxy(wimg, nblob=1, binning=1, quiet=False)#, plot=True)
            centralmask = ellipse_mask(mge.xmed, mge.ymed,
                                       mge.majoraxis,
                                       mge.majoraxis * (1. - mge.eps),
                                       np.radians(mge.theta-90.),
                                       xgrid, ygrid)

            #plt.clf()
            #mge = find_galaxy(wimg, nblob=1, binning=1, quiet=False, plot=True)
            #print(mge.ymed, mge.xmed)
            #plt.savefig('ioannis/tmp/junk-mge.png')

        if qaplot:
            # plot various classes of sources on the first image
            if psfsrcs:
                for xx in ax[0, :]:
                    xx.scatter(psfsrcs.bx, psfsrcs.by, marker='s',
                               s=5, color='red')
            if galsrcs:
                for xx in ax[0, :]:
                    xx.scatter(galsrcs.bx, galsrcs.by, marker='o',
                               s=5, color='cyan')

            norm = simple_norm(wimg, stretch='asinh', percent=99.5, asinh_a=0.1)
            ax[1+iobj, 0].imshow(wimg, origin='lower', cmap='inferno',
                               interpolation='none', norm=norm)
            ax[1+iobj, 1].imshow(opt_psfmasks[0, :, :], origin='lower')
            ax[1+iobj, 2].imshow(opt_galmasks[0, :, :], origin='lower')
            for xx in ax[1+iobj, :]:
                xx.set_xlim(0, wimg.shape[0])
                xx.set_ylim(0, wimg.shape[1])

    for xx in ax.ravel():
        xx.margins(0)
        xx.set_xticks([])
        xx.set_yticks([])

    fig.savefig(qafile)
    log.info(f'Wrote {qafile}')

    pdb.set_trace()


#        largeshift = False
#        var = np.zeros_like(invvar)
#        ok = invvar > 0
#        var[ok] = 1. / invvar[ok]
#        data[f'{filt.lower()}_var_'] = var # [nanomaggies**2]
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
#            cenflux = getattr(tractor, f'flux_{filt.lower()}')[central]
#            satflux = getattr(srcs, f'flux_{filt.lower()}')
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
#                satimg = srcs2image(satsrcs, data[f'{filt.lower()}_wcs'],
#                                    band=filt.lower(),
#                                    pixelized_psf=data[f'{filt.lower()}_psf'])
#                thissatmask = satimg > sigmamask*data[f'{filt.lower()}_sigma']
#                #if filt == 'FUV':
#                #    plt.clf() ; plt.imshow(thissatmask, origin='lower') ; plt.savefig('junk-{}.png'.format(filt.lower()))
#                #    #plt.clf() ; plt.imshow(data[filt], origin='lower') ; plt.savefig('junk-{}.png'.format(filt.lower()))
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
#            varkey = f'{filt.lower()}_var'
#            imagekey = f'{filt.lower()}_masked'
#            psfimgkey = f'{filt.lower()}_psfimg'
#            thispixscale = filt2pixscale[filt]
#            if imagekey not in data.keys():
#                data[imagekey], data[varkey], data[psfimgkey] = [], [], []
#
#            img = ma.getdata(data[filt]).copy()
#
#            # Get the PSF sources but ignore W3 and W4 (??)
#            psfindx = np.where((getattr(tractor, f'flux_{filt.lower()}') / cenflux > threshmask) *
#                               (tractor.type == 'PSF'))[0]
#            if len(psfindx) > 0 and filt.upper() != 'W3' and filt.upper() != 'W4':
#                psfsrcs = tractor.copy()
#                psfsrcs.cut(psfindx)
#            else:
#                psfsrcs = None
#
#            if psfsrcs:
#                psfimg = srcs2image(psfsrcs, data[f'{filt.lower()}_wcs'],
#                                    band=filt.lower(),
#                                    pixelized_psf=data[f'{filt.lower()}_psf'])
#                if False:#True:
#                    #import fitsio ; fitsio.write('junk-psf-{}.fits'.format(filt.lower()), data['{}_psf'.format(filt.lower())].img, clobber=True)
#                    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
#                    im = ax1.imshow(np.log10(img), origin='lower') ; fig.colorbar(im, ax=ax1)
#                    im = ax2.imshow(np.log10(psfimg), origin='lower') ; fig.colorbar(im, ax=ax2)
#                    im = ax3.imshow(np.log10(data[f'{filt.lower()}_psf'].img), origin='lower') ; fig.colorbar(im, ax=ax3)
#                    im = ax4.imshow(img-psfimg, origin='lower') ; fig.colorbar(im, ax=ax4)
#                    plt.savefig(f'ioannis/tmp/qa-psf-{filt.lower()}.png')
#                    if filt == 'r':
#                        pass
#                img -= psfimg
#            else:
#                psfimg = np.zeros((2, 2), 'f4')
#
#            data[psfimgkey].append(psfimg)
#
#            img = ma.masked_array((img / thispixscale**2).astype('f4'), mask) # [nanomaggies/arcsec**2]
#            var = data[f'{filt.lower()}_var_'] / thispixscale**4 # [nanomaggies**2/arcsec**4]
#
#            # Fill with zeros, for fun--
#            ma.set_fill_value(img, fill_value)
#            #if filt == 'r':# or filt == 'r':
#            #    plt.clf() ; plt.imshow(img, origin='lower') ; plt.savefig('desi-users/ioannis/tmp/junk-img-{}.png'.format(filt.lower()))
#            #    plt.clf() ; plt.imshow(mask, origin='lower') ; plt.savefig('desi-users/ioannis/tmp/junk-mask-{}.png'.format(filt.lower()))
#            ##    plt.clf() ; plt.imshow(thismask, origin='lower') ; plt.savefig('desi-users/ioannis/tmp/junk-thismask-{}.png'.format(filt.lower()))
#
#            data[imagekey].append(img)
#            data[varkey].append(var)
#
#        #test = data['r_masked'][0]
#        #plt.clf() ; plt.imshow(np.log(test.clip(test[mgegalaxy.xpeak, mgegalaxy.ypeak]/1e4)), origin='lower') ; plt.savefig('/mnt/legacyhalos-data/debug.png')

    # Cleanup?
    for filt in fit_bands:
        del data[filt]
        del data[f'{filt.lower()}_var_']

    return data


def read_multiband(galaxy, galaxydir, bands=['g', 'r', 'i', 'z'],
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
    galex_bands = ['FUV', 'NUV']
    unwise_bands = ['W1', 'W2', 'W3', 'W4']

    # Dictionary mapping between optical filter and filename coded up in
    # coadds.py, galex.py, and unwise.py, which depends on the project.
    data = {}
    data['galaxy'] = galaxy
    data['galaxydir'] = galaxydir

    refband, uv_refband, ir_refband = None, None, None

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
        bands = bands + galex_bands
        uv_refband = galex_bands[0]
        for band in galex_bands:
            filt2imfile.update({band: {'image': 'image',
                                       'model': 'model',
                                       'invvar': 'invvar',
                                       'psf': 'psf'}})
            filt2pixscale.update({band: galex_pixscale})

    if unwise:
        bands = bands + unwise_bands
        ir_refband = unwise_bands[0]
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
    data['refband'] = refband
    data['uv_refband'] = uv_refband
    data['ir_refband'] = ir_refband
    data['bands'] = bands
    data['optical_bands'] = optical_bands
    data['fit_bands'] = fit_bands
    data['fit_optical_bands'] = fit_optical_bands
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
        cols += [f'flux_{filt}' for filt in np.char.lower(galex_bands)]
        cols += [f'flux_ivar_{filt}' for filt in np.char.lower(galex_bands)]
    if unwise:
        cols += [f'flux_{filt}' for filt in np.char.lower(unwise_bands)]
        cols += [f'flux_ivar_{filt}' for filt in np.char.lower(unwise_bands)]

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

    sample['ROW'] = np.zeros(len(sample), int) - 1
    sample['RA'] = np.zeros(len(sample), 'f8')
    sample['DEC'] = np.zeros(len(sample), 'f8')
    sample['BX'] = np.zeros(len(sample), 'f4')
    sample['BY'] = np.zeros(len(sample), 'f4')
    sample['TYPE'] = np.zeros(len(sample), 'U3')
    sample['SERSIC'] = np.zeros(len(sample), 'f4')
    sample['SHAPE_R'] = np.zeros(len(sample), 'f4') # [arcsec]
    sample['SHAPE_E1'] = np.zeros(len(sample), 'f4')
    sample['SHAPE_E2'] = np.zeros(len(sample), 'f4')
    sample['FLUX'] = np.zeros(len(sample), 'f4') # brightest band
    sample['PSF'] = np.zeros(len(sample), bool)
    sample['DROPPED'] = np.zeros(len(sample), bool)
    sample['LARGESHIFT'] = np.zeros(len(sample), bool)
    sample['BLENDED'] = np.zeros(len(sample), bool)

    # special fitting bit(s) -- FIXME!
    sample['FIXGEOMETRY'] = sample['FITBIT'] & FITBITS['ignore'] != 0
    sample['FORCEPSF'] = sample['FITBIT'] & FITBITS['forcepsf'] != 0

    copycols = ['RA', 'DEC', 'BX', 'BY', 'TYPE', 'SERSIC',
                'SHAPE_R', 'SHAPE_E1', 'SHAPE_E2']

    for iobj, refid in enumerate(sample[REFIDCOLUMN].value):
        I = np.where((tractor.ref_cat == REFCAT) * (tractor.ref_id == refid))[0]
        if len(I) == 0:
            log.warning(f'ref_id={refid} dropped by Tractor')
            sample['DROPPED'][iobj] = True
            # important! set (BX,BY) to (BX_INIT,BY_INIT) because it
            # gets used downstream
            sample['BX'][iobj] = sample['BX_INIT'][iobj]
            sample['BY'][iobj] = sample['BY_INIT'][iobj]
        else:
            sample['ROW'][iobj] = I
            for col, attr in zip(copycols, np.char.lower(copycols)):
                sample[col][iobj] = getattr(tractor[I[0]], attr)
            if sample['TYPE'][iobj] in ['PSF', 'DUP']:
                log.warning(f'ref_id={refid} fit by Tractor as PSF (or DUP)')
                sample['PSF'][iobj] = True
            sample['FLUX'][iobj] = max([getattr(tractor[I[0]], f'flux_{filt.lower()}')
                                        for filt in fit_optical_bands])

    # Sort by optical brightness (in any band).
    log.info('Sorting by optical flux:')
    srt = np.argsort(sample['FLUX'])[::-1]
    sample = sample[srt]
    for obj in sample:
        log.info(f'  ref_id={obj[REFIDCOLUMN]} (row={obj["ROW"]}): ' + \
                 f'max optical flux={obj["FLUX"]:.2f} nanomaggies')
    data['sample'] = sample

    # add the PSF depth and size
    data.update(_get_psfsize_and_depth(tractor, bands, pixscale, incenter=False))

    # Read the maskbits image and build the starmask.
    maskbitsfile = os.path.join(galaxydir, f'{galaxy}-{filt2imfile["maskbits"]}.fits.fz')
    if verbose:
        log.info(f'Reading {maskbitsfile}')
    F = fitsio.FITS(maskbitsfile)
    maskbits = F['MASKBITS'].read()

    starmask = ( (maskbits & MASKBITS['BRIGHT'] != 0) |
                 (maskbits & MASKBITS['MEDIUM'] != 0) |
                 (maskbits & MASKBITS['CLUSTER'] != 0) )
    data['starmask'] = starmask

    # missing bands have ALLMASK_[GRIZ] == 0
    for filt in fit_optical_bands:
        data[f'allmask_{filt.lower()}'] = maskbits & MASKBITS[f'ALLMASK_{filt.upper()}'] != 0

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
    data = _build_multiband_mask(data, tractor, verbose=verbose)

    return data
