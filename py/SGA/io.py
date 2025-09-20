"""
SGA.io
======

General I/O functions.

"""
import os, time, pdb
import fitsio
import numpy as np
from astropy.table import Table, vstack, join

from SGA.logger import log


def set_legacysurvey_dir(region='dr9-north', rank=None):
    if not 'LEGACY_SURVEY_BASEDIR' in os.environ:
        msg = 'Mandatory LEGACY_SURVEY_BASEDIR environment variable not set!'
        log.critical(msg)
        raise EnvironmentError(msg)
    if False:
        log.warning('Temporarily using dr11-early-v2 directory for dr11-south!!')
        dirs = {'dr9-north': 'dr9', 'dr9-south': 'dr9', 'dr10-south': 'dr10', 'dr11-south': 'dr11-early-v2'}
    else:
        dirs = {'dr9-north': 'dr9', 'dr9-south': 'dr9', 'dr10-south': 'dr10', 'dr11-south': 'dr11'}
    legacy_survey_dir = os.path.join(os.getenv('LEGACY_SURVEY_BASEDIR'), dirs[region])
    if rank is not None:
        pre = f'Rank {rank}: '
    else:
        pre = ''
    log.info(f'{pre}Setting LEGACY_SURVEY_DIR={legacy_survey_dir}')
    os.environ['LEGACY_SURVEY_DIR'] = legacy_survey_dir


def get_raslice(ra):
    if np.isscalar(ra):
        return f'{int(ra):03d}'
    else:
        return np.array([f'{int(onera):03d}' for onera in ra])


def radec_to_groupname(ra, dec, prefix=''):
    # 36-arcsec precision (0.01 degrees)
    group_name = []
    for ra1, dec1 in zip(np.atleast_1d(ra), np.atleast_1d(dec)):
        group_name.append('{}{:05d}{}{:04d}'.format(
            prefix, int(100*ra1), 'm' if dec1 < 0 else 'p',
            int(100*np.abs(dec1))))
    group_name = np.array(group_name)
    return group_name


def radec_to_name(target_ra, target_dec, prefix='SGA2025',
                  precision=4, unixsafe=False):
    """Convert the right ascension and declination of an object into a
    disk-friendly "name", for reference in publications.  Length of
    `target_ra` and `target_dec` must be the same if providing an
    array or list.

    Parameters
    ----------
    target_ra: array of :class:`~numpy.float64`
        Right ascension in degrees of target object(s). Can be float, double,
        or array/list of floats or doubles.
    target_dec: array of :class:`~numpy.float64`
        Declination in degrees of target object(s). Can be float, double,
        or array/list of floats or doubles.
    precision: :class:`int`
        Number of decimal places in final naming convention.

    Returns
    -------
    array of :class:`str`
        Names referring to the input target RA and DEC's. Array is the
        same length as the input arrays.

    Raises
    ------
    ValueError
        If any input values are out of bounds.

    Notes
    -----
    Written by A. Kremin (LBNL) for DESI. Taken entirely from
    desiutil.names.radec_to_desiname.

    """
    # Convert to numpy array in case inputs are scalars or lists
    target_ra, target_dec = np.atleast_1d(target_ra), np.atleast_1d(target_dec)

    base_tests = [('NaN values', np.isnan),
                  ('Infinite values', np.isinf),]
    inputs = {'target_ra': {'data': target_ra,
                            'tests': base_tests + [('RA not in range [0, 360)', lambda x: (x < 0) | (x >= 360))]},
              'target_dec': {'data': target_dec,
                             'tests': base_tests + [('Dec not in range [-90, 90]', lambda x: (x < -90) | (x > 90))]}}
    for coord in inputs:
        for message, check in inputs[coord]['tests']:
            if check(inputs[coord]['data']).any():
                raise ValueError(f"{message} detected in {coord}!")

    # Truncate decimals to the given precision
    ratrunc = np.trunc((10.**precision) * target_ra).astype(int).astype(str)
    dectrunc = np.trunc((10.**precision) * target_dec).astype(int).astype(str)

    # Loop over input values and create the name as DESINAME as: DESI JXXX.XXXX+/-YY.YYYY
    # Here J refers to J2000, which isn't strictly correct but is the closest
    #   IAU compliant term
    names = []
    for ra, dec in zip(ratrunc, dectrunc):
        zra = ra.zfill(7)
        name = f'{prefix} J' + zra[:-precision] + '.' + zra[-precision:]
        # Positive numbers need an explicit "+" while negative numbers
        #   already have a "-".
        # zfill works properly with '-' but counts it in number of characters
        #   so need one more
        if dec.startswith('-'):
            zdec = dec.zfill(7)
            name += zdec[:-precision] + '.' + zdec[-precision:]
        else:
            zdec = dec.zfill(6)
            name += '+' + zdec[:-precision] + '.' + zdec[-precision:]
        names.append(name)

    names = np.array(names)

    # convert spaces to underscores
    if unixsafe:
        names = np.char.replace(names, ' ', '_')

    #if len(names) == 1:
    #    return names[0]
    #else:
    #    return names
    return names


def backup_filename(filename):
    """rename filename to next available filename.N

    Args:
        filename (str): full path to filename

    Returns:
        New filename.N, or filename if original file didn't already exist

    if filename=='/dev/null' or filename doesn't exist, just return filename

    """
    if filename == '/dev/null' or not os.path.exists(filename):
        return filename

    n = 0
    while True:
        altfile = f'{filename}.{n}'
        if os.path.exists(altfile):
            n += 1
        else:
            break

    os.rename(filename, altfile)

    return altfile


def read_survey_bricks(survey, brickname=None, custom=False):
    """Read the sample of bricks corresponding to the given the run.

    Currently, we read the full-sky set of bricks, but this should really be
    reduced down to the set of bricks with data.

    """
    def _toTable(_bricks):
        # convert to an astropy Table
        _bricks = _bricks.to_dict()
        bricks = Table()
        for key in _bricks.keys():
            bricks[key.upper()] = _bricks[key]
        return bricks

    if custom:
        from SGA.brick import custom_brickname
        # define a set of custom bricks (for testing purposes)
        bricks = Table()
        # https://www.legacysurvey.org/viewer-desi?ra=15.8232&dec=-4.6630&layer=ls-dr9&zoom=15&sga
        bricks['RA'] = [15.8232] # in '0159m047'
        bricks['DEC'] = [-4.6630]
        bricks['WIDTH'] = [600]
        bricks['BRICKNAME'] = [f'custom-{custom_brickname(ra, dec)}' for ra, dec in zip(bricks['RA'], bricks['DEC'])]
    else:
        if brickname is not None:
            bricks = survey.get_bricks_by_name(brickname)
        else:
            bricks = survey.get_bricks()
        bricks = _toTable(bricks)

    return bricks


def _missing_files_one(args):
    """Wrapper for the multiprocessing."""
    return missing_files_one(*args)


def missing_files_one(checkfile, dependsfile, overwrite):
    """Simple support script for missing_files."""

    from pathlib import Path
    if Path(checkfile).exists() and overwrite is False:
        # Is the stage that this stage depends on done, too?
        #log.warning(checkfile, dependsfile, overwrite)
        if dependsfile is None:
            return 'done'
        else:
            if Path(dependsfile).exists():
                return 'done'
            else:
                return 'todo'
    else:
        #log.warning(f'missing_files_one {checkfile}')
        # Did this object fail?
        # fragile!
        if checkfile[-6:] == 'isdone':
            failfile = checkfile[:-6]+'isfail'
            if Path(failfile).exists():
                if overwrite is False:
                    return 'fail'
                else:
                    os.remove(failfile)
                    return 'todo'
            else:
                return 'todo'
        else:
            if dependsfile is not None:
                if os.path.isfile(dependsfile):
                    return 'todo'
                else:
                    log.warning(f'Missing depends file {dependsfile}')
                    return 'fail'
            else:
                return 'todo'

        return 'todo'


def read_fits_catalog(catfile, ext=1, columns=None, rows=None):
    """Simple wrapper to read an input catalog.

    """
    if not os.path.isfile(catfile):
        log.warning(f'Catalog {catfile} not found')
        return

    try:
        cat = Table(fitsio.read(catfile, ext=ext, rows=rows, columns=columns))
        log.info(f'Read {len(cat):,d} galaxies from {catfile}')
        return cat
    except:
        msg = f'Problem reading {catfile}'
        log.critical(msg)
        raise IOError(msg)


def _read_image_data(data, filt2imfile, verbose=False):
    """Helper function for the project-specific read_multiband method.

    Read the multi-band images and inverse variance images and pack them into a
    dictionary. Also create an initial pixel-level mask and handle images with
    different pixel scales (e.g., GALEX and WISE images).

    """
    from scipy.ndimage.morphology import binary_dilation
    from skimage.transform import resize
    from astropy.stats import sigma_clipped_stats
    from photutils.segmentation import detect_threshold, detect_sources

    from tractor.psf import PixelizedPSF
    from tractor.tractortime import TAITime
    from astrometry.util.util import Tan
    from legacypipe.survey import LegacySurveyWcs, ConstantFitsWcs


    VEGA2AB = {'W1': 2.699, 'W2': 3.339, 'W3': 5.174, 'W4': 6.620}

    all_data_bands = data['all_data_bands']
    opt_bands = data['opt_bands']
    unwise_bands = data['unwise_bands']

    opt_refband = data['opt_refband']
    galex_refband = data['galex_refband']
    unwise_refband = data['unwise_refband']

    # Read the per-filter images and generate an optical and UV/IR
    # mask.
    for filt in all_data_bands:
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
                data['opt_hdr'] = hdr
                data['opt_wcs'] = wcs
            elif filt == galex_refband:
                data['galex_hdr'] = hdr
                data['galex_wcs'] = wcs
            elif filt == unwise_refband:
                data['unwise_hdr'] = hdr
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
        # optical bands and wisemask for the unwise bands.
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

        # Robustly estimate the sky-sigma; we do our own source
        # detection and segmentation here because the Tractor model
        # can sometimes be quite poor, e.g., UGC 05688.
        if filt in opt_bands:
            threshold = detect_threshold(image, nsigma=3., background=0.)
            segment_img = detect_sources(image, threshold, npixels=10)
            if segment_img is not None:
                msk = segment_img.make_source_mask()
                msk *= ~mask # exclude "bad" pixels
            else:
                msk = ~mask
            mn, med, skysigma = sigma_clipped_stats(image, sigma=2.5, mask=msk)

            #import matplotlib.pyplot as plt
            #plt.clf() ; plt.imshow(msk, origin='lower') ; plt.savefig('ioannis/tmp/junk2.png') ; plt.close()
            #plt.clf() ; plt.imshow(np.log10(image-model),origin='lower') ; plt.savefig('ioannis/tmp/junk2.png') ; plt.close()

            #from scipy.ndimage.filters import gaussian_filter
            #resid = gaussian_filter(image - model, 2.)
            #_, _, sig = sigma_clipped_stats(resid[~mask], sigma=2.5)
            #_, _, skysigma = sigma_clipped_stats(image - model, sigma=1.5)
            data[f'{filt}_skysigma'] = skysigma

        # set invvar of masked pixels to zero.
        log.debug('Setting invvar of masked pixels to zero.')
        invvar[mask] = 0.

        data[filt] = image # [nanomaggies]
        data[f'{filt}_invvar'] = invvar # [1/nanomaggies**2]
        data[f'{filt}_mask'] = mask

    if 'wisemask' in data:
        del data['wisemask']

    return data


def table_to_fitsio(tbl):
    """
    Convert an Astropy Table/QTable into (data, names, units) for fitsio.write().
    Units are written verbatim to TUNITn (non-standard allowed).

    """
    names = list(tbl.colnames)
    data  = [tbl[name].value if hasattr(tbl[name], 'value') else np.asarray(tbl[name])
             for name in names]
    units = []
    for name in names:
        u = getattr(tbl[name], 'unit', None)
        units.append(str(u) if u is not None else '')
    return data, names, units


def make_header(src_hdr, keys, *, extname=None, bunit=None, extra=None):
    """
    Build a FITSHDR copying specific keys from an existing fitsio FITSHDR.

    Parameters
    ----------
    src_hdr : fitsio.FITSHDR or None
        Source header to copy from (can be None).
    keys : iterable of str
        Keyword names to copy (if present) with their values and comments.
    extname : str, optional
        Set/override EXTNAME.
    bunit : str, optional
        Set/override BUNIT (free-form text ok in fitsio).
    extra : dict, optional
        Extra cards to add. Values can be scalar, or (value, comment) tuples.

    Returns
    -------
    hdr : fitsio.FITSHDR

    """
    hdr = fitsio.FITSHDR()
    if src_hdr is not None:
        for k in keys:
            if k in src_hdr:
                val = src_hdr[k]
                com = src_hdr.get_comment(k) if hasattr(src_hdr, 'get_comment') else ''
                hdr.add_record({'name': k, 'value': val, 'comment': com})
    if extname is not None:
        hdr['EXTNAME'] = extname
    if bunit is not None:
        hdr['BUNIT'] = bunit
    if extra:
        for k, v in extra.items():
            if isinstance(v, tuple) and len(v) == 2:
                val, com = v
                hdr.add_record({'name': k, 'value': val, 'comment': com})
            else:
                hdr[k] = v

    return hdr


def write_ellipsefit(data, sample, datasets, results, sbprofiles,
                     verbose=False):
    """Write out the ellipse and sbprofiles catalogs.

    """
    from glob import glob
    from astropy.io import fits


    REFIDCOLUMN = data['REFIDCOLUMN']

    # optical header cards to copy from the original image
    opt_cards = ['LEGPIPEV', 'LSDIR', ]




    for idata, dataset in enumerate(datasets):
        if dataset == 'opt':
            suffix = ''.join(data['all_opt_bands']) # always griz in north & south
        else:
            suffix = dataset

        hdr = data[f'{dataset}_hdr']

        # remove old files
        ellipsefiles = glob(os.path.join(data["galaxydir"], f'*-ellipse-{suffix}.fits'))
        if len(ellipsefiles) > 0:
            for ellipsefile in ellipsefiles:
                os.remove(ellipsefile)

        for iobj, (obj, results_obj) in enumerate(zip(sample, results[idata])):

            sganame = obj['SGANAME'].replace(' ', '_') # unix safe
            ellipsefile = os.path.join(data["galaxydir"], f'{sganame}-ellipse-{suffix}.fits')

            # add the sample catalog to the optical ellipse file
            if dataset == 'opt':
                results_obj = join(obj, results_obj)

            sbprofiles_obj = sbprofiles[idata][iobj]
            models = data[f'{dataset}_models'][iobj, :, :, :]
            maskbits = data[f'{dataset}_maskbits'][iobj, :, :]

            # add to header:
            # --WCS!
            # --bands
            # --pixscale(s)
            # --integrmode
            # --sclip
            # --nclip
            # --width,height

            pdb.set_trace()

            tmpfile = ellipsefile + '.tmp'
            with fitsio.FITS(tmpfile, mode='rw', clobber=True) as F:
                # models, maskbits are numpy arrays
                # orig_models_hdr / orig_mask_hdr are fitsio.FITSHDR you inherited
                models_hdr  = make_header(orig_models_hdr,
                                          keys=['FILTER','INSTRUME','EXPTIME','DATE-OBS','BUNIT'],
                                          extname='MODELS')  # add bunit=... if you want to override
                maskbits_hdr = make_header(orig_mask_hdr,
                                           keys=['FILTER','INSTRUME','DATE-OBS'], extname='MASKBITS')
                F.write(models, header=models_hdr)          # IMAGE HDU 0
                F.write(maskbits, header=maskbits_hdr)      # IMAGE HDU 1
                # results_obj → ELLIPSE table
                data, names, units = table_to_fitsio(results_obj)
                F.write(data, names=names, units=units, extname='ELLIPSE')
                # sbprofiles_obj → SBPROFILES table
                data2, names2, units2 = table_to_fitsio(sbprofiles_obj)
                F.write(data2, names=names2, units=units2, extname='SBPROFILES')
                # optional: add checksums
                for h in F:
                    h.write_checksum()
            os.rename(tmpfile, ellipsefile)



            hdu0 = fits.PrimaryHDU(models)
            hdu1 = fits.ImageHDU(maskbits)
            hdu2 = fits.convenience.table_to_hdu(results_obj)
            hdu3 = fits.convenience.table_to_hdu(sbprofiles_obj)

            hdu0.header['EXTNAME'] = 'MODELS'
            hdu1.header['EXTNAME'] = 'MASKBITS'
            hdu2.header['EXTNAME'] = 'ELLIPSE'
            hdu3.header['EXTNAME'] = 'SBPROFILES'

            hx = fits.HDUList([hdu0, hdu1, hdu2, hdu3])
            tmpfile = ellipsefile+'.tmp'
            hx.writeto(tmpfile, overwrite=True, checksum=True)
            os.rename(tmpfile, ellipsefile)
            log.info(f'Wrote {ellipsefile}')

    return 1
