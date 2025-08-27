"""
SGA.io
======

General I/O functions.

"""
import os, time, pdb
import fitsio
import numpy as np
from astropy.table import Table

from SGA.logger import log


def set_legacysurvey_dir(region='dr9-north'):
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
    log.info(f'Setting LEGACY_SURVEY_DIR={legacy_survey_dir}')
    os.environ['LEGACY_SURVEY_DIR'] = legacy_survey_dir


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


def deprecated_write_ellipsefit(data, ellipsefit, bands=['g', 'r', 'i', 'z'], sbthresh=None,
                     apertures=None, add_datamodel_cols=None, verbose=False):
    """Write out a FITS file based on the output of
    ellipse.ellipsefit_multiband..

    ellipsefit - input dictionary

    """
    from astropy.io import fits
    from astropy.table import Table

    ellipsefitfile = get_ellipsefit_filename(galaxy, galaxydir, filesuffix=filesuffix, galaxy_id=galaxy_id)

    if sbthresh is None:
        from SGA.ellipse import REF_SBTHRESH as sbthresh
    if apertures is None:
        from SGA.ellipse import REF_APERTURES as apertures

    # Turn the ellipsefit dictionary into a FITS table, starting with the
    # galaxyinfo dictionary (if provided).
    out = Table()
    if galaxyinfo:
        for key in galaxyinfo.keys():
            data = galaxyinfo[key][0]
            if np.isscalar(data):
                data = np.atleast_1d(data)
            else:
                data = np.atleast_2d(data)
            unit = galaxyinfo[key][1] # add units
            col = Column(name=key, data=data, dtype=data.dtype, unit=unit)
            #if type(unit) is str:
            #else:
            #    #data *= unit
            #    #data = u.Quantity(value=data, unit=unit, dtype=data.dtype)
            #    col = Column(name=key, data=data, dtype=data.dtype)
            out.add_column(col)

    # First, unpack the nested dictionaries.
    datadict = {}
    for key in ellipsefit.keys():
        #if type(ellipsefit[key]) is dict: # obsolete
        #    for key2 in ellipsefit[key].keys():
        #        datadict['{}_{}'.format(key, key2)] = ellipsefit[key][key2]
        #else:
        #    datadict[key] = ellipsefit[key]
        datadict[key] = ellipsefit[key]
    del ellipsefit

    # Add to the data table
    datakeys = datadict.keys()
    for key, unit in _get_ellipse_datamodel(sbthresh, apertures, bands=bands, add_datamodel_cols=add_datamodel_cols,
                                            copy_mw_transmission=copy_mw_transmission):
        if key not in datakeys:
            raise ValueError('Data model change -- no column {} for galaxy {}!'.format(key, galaxy))
        data = datadict[key]
        if np.isscalar(data):# or len(np.array(data)) > 1:
            data = np.atleast_1d(data)
        #elif len(data) == 0:
        #    data = np.atleast_1d(data)
        else:
            data = np.atleast_2d(data)
        #if type(unit) is not str:
        #    data = u.Quantity(value=data, unit=unit, dtype=data.dtype)
        #col = Column(name=key, data=data)
        col = Column(name=key, data=data, dtype=data.dtype, unit=unit)
        #if 'z_cog' in key:
        #    print(key)
        #    pdb.set_trace()
        out.add_column(col)

    if np.logical_not(np.all(np.isin([*datakeys], out.colnames))):
        raise ValueError('Data model change -- non-documented columns have been added to ellipsefit dictionary!')

    # uppercase!
    for col in out.colnames:
        out.rename_column(col, col.upper())

    hdr = legacyhalos_header()

    #for col in out.colnames:
    #    print(col, out[col])

    hdu = fits.convenience.table_to_hdu(out)
    hdu.header['EXTNAME'] = 'ELLIPSE'
    hdu.header.update(hdr)
    hdu.add_checksum()

    hdu0 = fits.PrimaryHDU()
    hdu0.header['EXTNAME'] = 'PRIMARY'
    hx = fits.HDUList([hdu0, hdu])

    if verbose:
        print('Writing {}'.format(ellipsefitfile))
    tmpfile = ellipsefitfile+'.tmp'
    hx.writeto(tmpfile, overwrite=True, checksum=True)
    os.rename(tmpfile, ellipsefitfile)
    #hx.writeto(ellipsefitfile, overwrite=True, checksum=True)

    #out.write(ellipsefitfile, overwrite=True)
    #fitsio.write(ellipsefitfile, out.as_array(), extname='ELLIPSE', header=hdr, clobber=True)


def write_ellipsefit(data, datasets, results, sbprofiles, verbose=False):
    # add to header:
    #  --bands
    #  --pixscale(s)
    #  --integrmode
    #  --sclip
    #  --nclip
    #  --width,height

    # output data model:
    #  --use all_bands but do not write to table
    #  --psfdepth, etc.
    #  --maxsma

    for idata, dataset in enumerate(datasets):
        if dataset == 'opt':
            suffix = ''.join(data['all_opt_bands']) # always griz in north & south
        else:
            suffix = dataset

        for iobj, obj in enumerate(sample):
            ellipsefile = os.path.join(data["galaxydir"], f'{data["galaxy"]}-ellipse-{obj[REFIDCOLUMN]}-{suffix}.fits')

            results_obj = results[idata][iobj]
            sbprofiles_obj = sbprofiles[idata][iobj]
            images = data[f'{dataset}_images'][iobj, :, :, :]
            models = data[f'{dataset}_models'][iobj, :, :, :]
            maskbits = data[f'{dataset}_maskbits'][iobj, :, :]

            fitsio.write(ellipsefile, images, clobber=True, extname='IMAGES')
            fitsio.write(ellipsefile, models, extname='MODELS')
            fitsio.write(ellipsefile, maskbits, extname='MASKBITS')
            fitsio.write(ellipsefile, results_obj.as_array(), extname='ELLIPSE')
            fitsio.write(ellipsefile, sbprofiles_obj.as_array(), extname='SBPROFILES')
            log.info(f'Wrote {ellipsefile}')

    return 1
