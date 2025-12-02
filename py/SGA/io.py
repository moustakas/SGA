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
        if rank == 0:
            log.info(f'{pre}Setting LEGACY_SURVEY_DIR={legacy_survey_dir}')
    else:
        log.info(f'Setting LEGACY_SURVEY_DIR={legacy_survey_dir}')
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


def _read_image_data(data, filt2imfile, read_jpg=False, verbose=False):
    """Helper function for the project-specific read_multiband method.

    Read the multi-band images and inverse variance images and pack them into a
    dictionary. Also create an initial pixel-level mask and handle images with
    different pixel scales (e.g., GALEX and WISE images).

    """
    from matplotlib.image import imread
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

            if read_jpg:
                if filt == opt_refband:
                    prefix = 'opt'
                    suffix = ''
                elif filt == galex_refband:
                    prefix = 'galex'
                    suffix = '-FUVNUV'
                elif filt == unwise_refband:
                    prefix = 'unwise'
                    suffix = '-W1W2'

                for imtype in ['image', 'model', 'resid']:
                    jpgfile = os.path.join(data['galaxydir'], f"{data['galaxy']}-{imtype}{suffix}.jpg")
                    if os.path.isfile(jpgfile):
                        jpg = imread(jpgfile)
                        data[f'{prefix}_jpg_{imtype}'] = jpg

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

        # Mask a XX% border.
        edge = int(0.02*sz[0])
        mask[:edge, :] = True
        mask[:, :edge] = True
        mask[:, sz[0]-edge:] = True
        mask[sz[0]-edge:, :] = True

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
        #log.debug('Setting invvar of masked pixels to zero.')
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
        if u is None:
            units.append('')
        else:
            u = str(u)
            if 'nmgy' in u:
                u = u.replace('nmgy', 'nanomaggy')
            units.append(u)
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
                     MASKBITS, verbose=False):
    """Write out the ellipse and sbprofiles catalogs.

    """
    from glob import glob
    from astropy.io import fits


    REFIDCOLUMN = data['REFIDCOLUMN']

    hdr_cards = [#'LEGPIPEV', 'LSDIR',
                 'RA', 'DEC', 'CTYPE1', 'CTYPE2', 'CRVAL1', 'CRVAL2',
                 'CRPIX1', 'CRPIX2', 'CD1_1', 'CD1_2', 'CD2_1',
                 'CD2_2', 'EQUINOX', ]
    phot_cards = ['MAGZERO', 'BUNIT', ]

    maskbits_comments = {
        'brightstar': 'BRIGHT|MEDIUM|CLUSTER',
        'gaiastar': 'Gaia star(s)',
        'galaxy': 'extended source(s)',
        'reference': 'SGA source(s)',
    }
    for band in data['all_bands']:
        maskbits_comments.update({band: f'invvar=0 in {band}-band image'})

    for idata, dataset in enumerate(datasets):
        if dataset == 'opt':
            suffix = ''.join(data['all_opt_bands']) # always griz in north & south
        else:
            suffix = dataset

        pixscale = data[f'{dataset}_pixscale']
        models_extra = {
            'PIXSCALE': (pixscale, 'pixel scale (arcsec/pixel)'),
            'PHOTSYS': ('AB', 'photometric system')}

        # the headers are inconsistent
        orighdr = data[f'{dataset}_hdr']
        if 'BUNIT' not in orighdr:
            orighdr['BUNIT'] = data['opt_hdr']['BUNIT']

        bands = data[f'{dataset}_bands']
        nbands = len(bands)

        if 'BANDS' in models_extra.keys():
            models_extra['BANDS'] = (bands, 'bands')
            models_extra['NBANDS'] = (nbands, 'number of bands')
        else:
            models_extra.update({
                'BANDS': (','.join(bands), 'bands'),
                'NBANDS': (nbands, 'number of bands')})

        # add MASKBITS definitions
        maskbits_extra = {'PIXSCALE': (pixscale, 'pixel scale (arcsec/pixel)')}
        bits = MASKBITS[idata]
        for ibit, bit in enumerate(bits):
            #power = int(np.log2(bits[bit]))
            maskbits_extra[f'MBIT_{ibit}'] = (
                bit, f'maskbits bit {ibit} ' + \
                f'({str(hex(bits[bit]))}): {maskbits_comments[bit]}')

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
            # --bands

            tmpfile = ellipsefile + '.tmp'
            with fitsio.FITS(tmpfile, mode='rw', clobber=True) as F:
                # models, maskbits are numpy arrays
                models_hdr = make_header(orighdr, keys=hdr_cards+phot_cards,
                                         extra=models_extra)
                maskbits_hdr = make_header(orighdr, keys=hdr_cards,
                                           extra=maskbits_extra)
                F.write(models, header=models_hdr, extname='MODELS') # [nanomaggies]
                F.write(maskbits, header=maskbits_hdr, extname='MASKBITS')

                # results_obj → ELLIPSE table
                vals, names, units = table_to_fitsio(results_obj)
                F.write(vals, names=names, units=units, extname='ELLIPSE')

                # sbprofiles_obj → SBPROFILES table
                vals, names, units = table_to_fitsio(sbprofiles_obj)
                F.write(vals, names=names, units=units, extname='SBPROFILES')
                # add checksums
                for h in F:
                    h.write_checksum()

            os.rename(tmpfile, ellipsefile)
            log.info(f'Wrote {ellipsefile}')

    return 1


def empty_tractor(cat=None):
    if cat is not None:
        for col in cat.colnames:
            if cat[col].dtype == bool:
                cat[col] = True # False # [brick_primary]
            else:
                cat[col] *= 0
        return cat
    else:
        # fragile!
        # fitsio.read('/global/cfs/cdirs/cosmo/work/legacysurvey/dr11a/tractor/175/tractor-1758m007.fits').dtype.descr
        # fitsio.read('/pscratch/sd/i/ioannis/SGA2025-v0.10/dr11-south/068/06860m2556/SGA2025_06860m2556-tractor.fits').dtype.descr
        COLS = [
            ('ls_id_dr11', '>i8'),
            ('release', '>i2'),
            ('brickid', '>i4'),
            ('brickname', '<U8'),
            ('objid', '>i4'),
            ('brick_primary', '|b1'),
            ('maskbits', '>i4'),
            ('fitbits', '>i2'),
            ('type', '<U3'),
            ('ra', '>f8'),
            ('dec', '>f8'),
            ('ra_ivar', '>f4'),
            ('dec_ivar', '>f4'),
            ('bx', '>f4'),
            ('by', '>f4'),
            ('dchisq', '>f4', (5,)),
            ('ebv', '>f4'),
            ('mjd_min', '>f8'),
            ('mjd_max', '>f8'),
            ('nearest_neighbor', '>f4'),
            ('ref_cat', '<U2'),
            ('ref_id', '>i8'),
            ('pmra', '>f4'),
            ('pmdec', '>f4'),
            ('parallax', '>f4'),
            ('pmra_ivar', '>f4'),
            ('pmdec_ivar', '>f4'),
            ('parallax_ivar', '>f4'),
            ('ref_epoch', '>f4'),
            ('gaia_phot_g_mean_mag', '>f4'),
            ('gaia_phot_g_mean_flux_over_error', '>f4'),
            ('gaia_phot_g_n_obs', '>i4'),
            ('gaia_phot_bp_mean_mag', '>f4'),
            ('gaia_phot_bp_mean_flux_over_error', '>f4'),
            ('gaia_phot_bp_n_obs', '>i4'),
            ('gaia_phot_rp_mean_mag', '>f4'),
            ('gaia_phot_rp_mean_flux_over_error', '>f4'),
            ('gaia_phot_rp_n_obs', '>i4'),
            ('gaia_phot_variable_flag', '|b1'),
            ('gaia_astrometric_excess_noise', '>f4'),
            ('gaia_astrometric_excess_noise_sig', '>f4'),
            ('gaia_astrometric_n_obs_al', '>i2'),
            ('gaia_astrometric_n_good_obs_al', '>i2'),
            ('gaia_astrometric_weight_al', '>f4'),
            ('gaia_duplicated_source', '|b1'),
            ('gaia_a_g_val', '>f4'),
            ('gaia_e_bp_min_rp_val', '>f4'),
            ('gaia_phot_bp_rp_excess_factor', '>f4'),
            ('gaia_astrometric_sigma5d_max', '>f4'),
            ('gaia_astrometric_params_solved', '|u1'),
            ('flux_g', '>f4'),
            ('flux_r', '>f4'),
            ('flux_i', '>f4'),
            ('flux_z', '>f4'),
            ('flux_w1', '>f4'),
            ('flux_w2', '>f4'),
            ('flux_w3', '>f4'),
            ('flux_w4', '>f4'),
            ('flux_nuv', '>f4'),
            ('flux_fuv', '>f4'),
            ('flux_ivar_g', '>f4'),
            ('flux_ivar_r', '>f4'),
            ('flux_ivar_i', '>f4'),
            ('flux_ivar_z', '>f4'),
            ('flux_ivar_w1', '>f4'),
            ('flux_ivar_w2', '>f4'),
            ('flux_ivar_w3', '>f4'),
            ('flux_ivar_w4', '>f4'),
            ('flux_ivar_nuv', '>f4'),
            ('flux_ivar_fuv', '>f4'),
            ('fiberflux_g', '>f4'),
            ('fiberflux_r', '>f4'),
            ('fiberflux_i', '>f4'),
            ('fiberflux_z', '>f4'),
            ('fibertotflux_g', '>f4'),
            ('fibertotflux_r', '>f4'),
            ('fibertotflux_i', '>f4'),
            ('fibertotflux_z', '>f4'),
            ('apflux_g', '>f4', (8,)),
            ('apflux_r', '>f4', (8,)),
            ('apflux_i', '>f4', (8,)),
            ('apflux_z', '>f4', (8,)),
            ('apflux_resid_g', '>f4', (8,)),
            ('apflux_resid_r', '>f4', (8,)),
            ('apflux_resid_i', '>f4', (8,)),
            ('apflux_resid_z', '>f4', (8,)),
            ('apflux_blobresid_g', '>f4', (8,)),
            ('apflux_blobresid_r', '>f4', (8,)),
            ('apflux_blobresid_i', '>f4', (8,)),
            ('apflux_blobresid_z', '>f4', (8,)),
            ('apflux_ivar_g', '>f4', (8,)),
            ('apflux_ivar_r', '>f4', (8,)),
            ('apflux_ivar_i', '>f4', (8,)),
            ('apflux_ivar_z', '>f4', (8,)),
            ('apflux_masked_g', '>f4', (8,)),
            ('apflux_masked_r', '>f4', (8,)),
            ('apflux_masked_i', '>f4', (8,)),
            ('apflux_masked_z', '>f4', (8,)),
            ('apflux_w1', '>f4', (5,)),
            ('apflux_w2', '>f4', (5,)),
            ('apflux_w3', '>f4', (5,)),
            ('apflux_w4', '>f4', (5,)),
            ('apflux_resid_w1', '>f4', (5,)),
            ('apflux_resid_w2', '>f4', (5,)),
            ('apflux_resid_w3', '>f4', (5,)),
            ('apflux_resid_w4', '>f4', (5,)),
            ('apflux_ivar_w1', '>f4', (5,)),
            ('apflux_ivar_w2', '>f4', (5,)),
            ('apflux_ivar_w3', '>f4', (5,)),
            ('apflux_ivar_w4', '>f4', (5,)),
            ('apflux_nuv', '>f4', (5,)),
            ('apflux_fuv', '>f4', (5,)),
            ('apflux_resid_nuv', '>f4', (5,)),
            ('apflux_resid_fuv', '>f4', (5,)),
            ('apflux_ivar_nuv', '>f4', (5,)),
            ('apflux_ivar_fuv', '>f4', (5,)),
            ('mw_transmission_g', '>f4'),
            ('mw_transmission_r', '>f4'),
            ('mw_transmission_i', '>f4'),
            ('mw_transmission_z', '>f4'),
            ('mw_transmission_w1', '>f4'),
            ('mw_transmission_w2', '>f4'),
            ('mw_transmission_w3', '>f4'),
            ('mw_transmission_w4', '>f4'),
            ('mw_transmission_nuv', '>f4'),
            ('mw_transmission_fuv', '>f4'),
            ('nobs_g', '>i2'),
            ('nobs_r', '>i2'),
            ('nobs_i', '>i2'),
            ('nobs_z', '>i2'),
            ('nobs_w1', '>i2'),
            ('nobs_w2', '>i2'),
            ('nobs_w3', '>i2'),
            ('nobs_w4', '>i2'),
            ('nobs_nuv', '>i2'),
            ('nobs_fuv', '>i2'),
            ('rchisq_g', '>f4'),
            ('rchisq_r', '>f4'),
            ('rchisq_i', '>f4'),
            ('rchisq_z', '>f4'),
            ('rchisq_w1', '>f4'),
            ('rchisq_w2', '>f4'),
            ('rchisq_w3', '>f4'),
            ('rchisq_w4', '>f4'),
            ('rchisq_nuv', '>f4'),
            ('rchisq_fuv', '>f4'),
            ('fracflux_g', '>f4'),
            ('fracflux_r', '>f4'),
            ('fracflux_i', '>f4'),
            ('fracflux_z', '>f4'),
            ('fracflux_w1', '>f4'),
            ('fracflux_w2', '>f4'),
            ('fracflux_w3', '>f4'),
            ('fracflux_w4', '>f4'),
            ('fracflux_nuv', '>f4'),
            ('fracflux_fuv', '>f4'),
            ('fracmasked_g', '>f4'),
            ('fracmasked_r', '>f4'),
            ('fracmasked_i', '>f4'),
            ('fracmasked_z', '>f4'),
            ('fracin_g', '>f4'),
            ('fracin_r', '>f4'),
            ('fracin_i', '>f4'),
            ('fracin_z', '>f4'),
            ('ngood_g', '>i2'),
            ('ngood_r', '>i2'),
            ('ngood_i', '>i2'),
            ('ngood_z', '>i2'),
            ('anymask_g', '>i2'),
            ('anymask_r', '>i2'),
            ('anymask_i', '>i2'),
            ('anymask_z', '>i2'),
            ('allmask_g', '>i2'),
            ('allmask_r', '>i2'),
            ('allmask_i', '>i2'),
            ('allmask_z', '>i2'),
            ('wisemask_w1', '|u1'),
            ('wisemask_w2', '|u1'),
            ('psfsize_g', '>f4'),
            ('psfsize_r', '>f4'),
            ('psfsize_i', '>f4'),
            ('psfsize_z', '>f4'),
            ('psfdepth_g', '>f4'),
            ('psfdepth_r', '>f4'),
            ('psfdepth_i', '>f4'),
            ('psfdepth_z', '>f4'),
            ('galdepth_g', '>f4'),
            ('galdepth_r', '>f4'),
            ('galdepth_i', '>f4'),
            ('galdepth_z', '>f4'),
            ('nea_g', '>f4'),
            ('nea_r', '>f4'),
            ('nea_i', '>f4'),
            ('nea_z', '>f4'),
            ('blob_nea_g', '>f4'),
            ('blob_nea_r', '>f4'),
            ('blob_nea_i', '>f4'),
            ('blob_nea_z', '>f4'),
            ('psfdepth_w1', '>f4'),
            ('psfdepth_w2', '>f4'),
            ('psfdepth_w3', '>f4'),
            ('psfdepth_w4', '>f4'),
            ('psfdepth_nuv', '>f4'),
            ('psfdepth_fuv', '>f4'),
            ('wise_coadd_id', '<U8'),
            ('wise_x', '>f4'),
            ('wise_y', '>f4'),
            #('lc_flux_w1', '>f4', (19,)),
            #('lc_flux_w2', '>f4', (19,)),
            #('lc_flux_ivar_w1', '>f4', (19,)),
            #('lc_flux_ivar_w2', '>f4', (19,)),
            #('lc_nobs_w1', '>i2', (19,)),
            #('lc_nobs_w2', '>i2', (19,)),
            #('lc_fracflux_w1', '>f4', (19,)),
            #('lc_fracflux_w2', '>f4', (19,)),
            #('lc_rchisq_w1', '>f4', (19,)),
            #('lc_rchisq_w2', '>f4', (19,)),
            #('lc_mjd_w1', '>f8', (19,)),
            #('lc_mjd_w2', '>f8', (19,)),
            #('lc_epoch_index_w1', '>i2', (19,)),
            #('lc_epoch_index_w2', '>i2', (19,)),
            ('sersic', '>f4'),
            ('sersic_ivar', '>f4'),
            ('shape_r', '>f4'),
            ('shape_r_ivar', '>f4'),
            ('shape_e1', '>f4'),
            ('shape_e1_ivar', '>f4'),
            ('shape_e2', '>f4'),
            ('shape_e2_ivar', '>f4'),
        ]
        tractor = Table()
        for col in COLS:
            colname = col[0]
            if len(col) == 2:
                dtype, shape = col[1], (1,)
            else:
                dtype, shape = col[1], (1,) + col[2]
            tractor[colname] = np.zeros(shape=shape, dtype=dtype)
        return tractor
