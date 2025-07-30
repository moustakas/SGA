"""
SGA.io
======

Code to read and write the various SGA files.

"""
import os, time, pdb
import fitsio
import numpy as np
import numpy.ma as ma
from astropy.table import Table, vstack

from SGA.logger import log


RACOLUMN = 'GROUP_RA'   # 'RA'
DECCOLUMN = 'GROUP_DEC' # 'DEC'
DIAMCOLUMN = 'GROUP_DIAMETER' # 'DIAM'
ZCOLUMN = 'Z'
REFIDCOLUMN = 'SGAID'


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


def custom_brickname(ra, dec, more_decimals=False):
    if more_decimals:
        brickname = '{:08d}{}{:07d}'.format(
            int(100000*ra), 'm' if dec < 0 else 'p',
            int(100000*np.abs(dec)))
    else:
        brickname = '{:06d}{}{:05d}'.format(
            int(1000*ra), 'm' if dec < 0 else 'p',
            int(1000*np.abs(dec)))
    return brickname


def get_raslice(ra):
    if np.isscalar(ra):
        return f'{int(ra):03d}'
    else:
        return np.array([f'{int(onera):03d}' for onera in ra])


def sga2025_name(ra, dec, unixsafe=False):
    # simple wrapper on radec_to_name with precision=3
    return radec_to_name(ra, dec, prefix='SGA2025', precision=3,
                         unixsafe=unixsafe)


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


def get_galaxy_galaxydir(sample=None, bricks=None, region='dr11-south',
                         datadir=None, htmldir=None, html=False):
    """Retrieve the galaxy name and the (nested) directory.

    """
    if sample is None and bricks is None:
        msg = 'Must provide either sample or bricks.'
        raise IOError(msg)

    if datadir is None:
        datadir = os.path.join(sga_data_dir(), region)
    if htmldir is None:
        htmldir = os.path.join(sga_html_dir(), region)

    if bricks is not None:
        objs = np.atleast_1d(bricks['BRICKNAME'])
        ras = np.atleast_1d(bricks['RA'])
        datadir = os.path.join(datadir, 'detection')
        htmldir = os.path.join(htmldir, 'detection')
    elif sample is not None:
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
        objdirs.append(os.path.join(datadir, get_raslice(ra), obj))
        if html:
            htmlobjdirs.append(os.path.join(htmldir, get_raslice(ra), obj))
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


def SGA_version():
    version = 'v1.0'
    return version


def parent_version(vicuts=False, nocuts=False, archive=False):
    if nocuts:
        version = 'v1.0'
        # many more objects added
        #version = 'v1.1'
    elif vicuts:
        version = 'v1.0'
        #version = 'v1.1'
    elif archive:
        version = 'v1.0'
        #version = 'v1.1'
    else:
        version = 'v1.0'
    return version


def parent_datamodel(nobj):
    """Initialize the data model for the parent-nocuts sample.

    """
    parent = Table()
    parent['OBJNAME'] = np.zeros(nobj, '<U30')
    parent['OBJNAME_NED'] = np.zeros(nobj, '<U30')
    parent['OBJNAME_HYPERLEDA'] = np.zeros(nobj, '<U30')
    parent['OBJNAME_NEDLVS'] = np.zeros(nobj, '<U30')
    parent['OBJNAME_SGA2020'] = np.zeros(nobj, '<U30')
    parent['OBJNAME_LVD'] = np.zeros(nobj, '<U30')
    parent['OBJTYPE'] = np.zeros(nobj, '<U6')
    parent['MORPH'] = np.zeros(nobj, '<U20')
    parent['BASIC_MORPH'] = np.zeros(nobj, '<U40')

    parent['RA'] = np.zeros(nobj, 'f8') -99.
    parent['DEC'] = np.zeros(nobj, 'f8') -99.
    parent['RA_NED'] = np.zeros(nobj, 'f8') -99.
    parent['DEC_NED'] = np.zeros(nobj, 'f8') -99.
    parent['RA_HYPERLEDA'] = np.zeros(nobj, 'f8') -99.
    parent['DEC_HYPERLEDA'] = np.zeros(nobj, 'f8') -99.
    parent['RA_NEDLVS'] = np.zeros(nobj, 'f8') -99.
    parent['DEC_NEDLVS'] = np.zeros(nobj, 'f8') -99.
    parent['RA_SGA2020'] = np.zeros(nobj, 'f8') -99.
    parent['DEC_SGA2020'] = np.zeros(nobj, 'f8') -99.
    parent['RA_LVD'] = np.zeros(nobj, 'f8') -99.
    parent['DEC_LVD'] = np.zeros(nobj, 'f8') -99.

    parent['Z'] = np.zeros(nobj, 'f8') -99.
    parent['Z_NED'] = np.zeros(nobj, 'f8') -99.
    parent['Z_HYPERLEDA'] = np.zeros(nobj, 'f8') -99.
    parent['Z_NEDLVS'] = np.zeros(nobj, 'f8') -99.

    parent['PGC'] = np.zeros(nobj, '<i8') -99
    parent['ESSENTIAL_NOTE'] = np.zeros(nobj, '<U80')

    parent['MAG_LIT'] = np.zeros(nobj, 'f4') -99.
    parent['MAG_LIT_REF'] = np.zeros(nobj, '<U9')
    parent['BAND_LIT'] = np.zeros(nobj, '<U1')
    parent['DIAM_LIT'] = np.zeros(nobj, 'f4') -99.
    parent['DIAM_LIT_REF'] = np.zeros(nobj, '<U9')
    parent['BA_LIT'] = np.zeros(nobj, 'f4') -99.
    parent['BA_LIT_REF'] = np.zeros(nobj, '<U9')
    parent['PA_LIT'] = np.zeros(nobj, 'f4') -99.
    parent['PA_LIT_REF'] = np.zeros(nobj, '<U9')

    parent['MAG_HYPERLEDA'] = np.zeros(nobj, 'f4') -99.
    parent['BAND_HYPERLEDA'] = np.zeros(nobj, '<U1')
    parent['DIAM_HYPERLEDA'] = np.zeros(nobj, 'f4') -99.
    parent['BA_HYPERLEDA'] = np.zeros(nobj, 'f4') -99.
    parent['PA_HYPERLEDA'] = np.zeros(nobj, 'f4') -99.

    parent['MAG_SGA2020'] = np.zeros(nobj, 'f4') -99.
    parent['BAND_SGA2020'] = np.zeros(nobj, '<U1')
    parent['DIAM_SGA2020'] = np.zeros(nobj, 'f4') -99.
    parent['BA_SGA2020'] = np.zeros(nobj, 'f4') -99.
    parent['PA_SGA2020'] = np.zeros(nobj, 'f4') -99.

    parent['ROW_HYPERLEDA'] = np.zeros(nobj, '<i8') -99
    parent['ROW_NEDLVS'] = np.zeros(nobj, '<i8') -99
    parent['ROW_SGA2020'] = np.zeros(nobj, '<i8') -99
    parent['ROW_LVD'] = np.zeros(nobj, '<i8') -99
    parent['ROW_CUSTOM'] = np.zeros(nobj, '<i8') -99

    return parent


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
        from SGA.coadds import custom_brickname
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


def read_zooniverse_sample(cat, fullcat=None, catfile=None, region='dr9-north',
                           outdir='.', project='project1'):
    """Read the zooniverse VI sample.

    """
    from SGA.util import match
    from SGA.ellipse import choose_geometry

    if project == 'project1':
        # basically the wisesize sample

        # 87. < RA < 300.
        # -10. < DEC < 85.
        # 0.002 < z < 0.025
        # W3 or NUV SNR > 20.   (for this, I divided  'Lum_W3'/'Lum_W3_unc' and 'Lum_NUV'/'Lum_NUV_unc', respectively)
        # diameter > 15. arcsec OR -99., as we are including objects which do not have size measurements in your nedgeometry catalog
        # Lastly, we removed VFS galaxies, since we already have access to those postage stamps

        def get_snr(flux, ferr):
            snr = np.zeros(len(flux))
            J = np.isfinite(flux) * np.isfinite(ferr) * (ferr > 0.)
            snr[J] = flux[J] / ferr[J]
            return snr

        nedlvs = read_nedlvs()

        I = cat['FILTERS'] == 'grz'
        print(f'In {region} grz footprint: {np.sum(I):,d}')
        cat = cat[I]
        nobj = len(cat)

        cat = cat[cat['ROW_NEDLVS'] != -99]
        indx_cat, indx_nedlvs = match(cat['ROW_NEDLVS'], nedlvs['ROW'])
        cat = cat[indx_cat]
        nedlvs = nedlvs[indx_nedlvs]
        print(f'In NED-LVS: {len(cat):,d}/{nobj:,d}')

        I = (cat['RA'] > 87.) * (cat['RA'] < 300.) * (cat['DEC'] > -10.) * (cat['DEC'] < 85.)
        print(f'In 87<RA<300, -10<Dec<85: {np.sum(I):,d}/{len(cat):,d}')
        cat = cat[I]
        nedlvs = nedlvs[I]

        I = (nedlvs['Z'] > 0.002) * (nedlvs['Z'] < 0.025)
        print(f'In 0.002<z<0.025 range: {np.sum(I):,d}/{len(cat):,d}')
        cat = cat[I]
        nedlvs = nedlvs[I]

        mindiam = 30. # [arcsec] # 15.
        diam, _, _, _ = choose_geometry(cat, mindiam=0.)

        I = (diam > mindiam)
        print(f'Diameter (>{mindiam:.0f} arcsec) cut: {np.sum(I):,d}/{len(cat):,d}')
        cat = cat[I]
        nedlvs = nedlvs[I]

        snrmin = 3. # 20.
        snr_W3 = get_snr(nedlvs['LUM_W3'], nedlvs['LUM_W3_UNC'])
        snr_NUV = get_snr(nedlvs['LUM_NUV'], nedlvs['LUM_NUV_UNC'])

        I = np.logical_or(snr_W3 > snrmin, snr_NUV > snrmin)
        print(f'S/N(W3)>{snrmin:.0f}, S/N(NUV)>{snrmin:.0f} cuts: {np.sum(I):,d}/{len(cat):,d}')
        cat = cat[I]
        nedlvs = nedlvs[I]

        if fullcat is not None:
            diam, _, _, _ = choose_geometry(fullcat, mindiam=0.)
            I = diam > 15.
            print(f'Trimmed fullcat to {np.sum(I):,d}/{len(fullcat):,d} objects with diam>15 arcsec')
            fullcat = fullcat[I]

        # optionally write out

        if not os.path.isdir(outdir):
            os.makedirs(outdir, exist_ok=True)
        outfile = os.path.join(outdir, f'zooniverse-{project}-{region}.fits')

        if catfile and not os.path.isfile(outfile):
            rows = np.where(np.isin(fitsio.read(catfile, columns='ROW_PARENT'), cat['ROW_PARENT'].value))[0]
            allcat = read_fits_catalog(catfile, rows=rows)
            indx_cat, indx_allcat = match(cat['ROW_PARENT'], allcat['ROW_PARENT'])

            allcat = allcat[indx_allcat]
            cat = cat[indx_cat]
            nedlvs = nedlvs[indx_cat]

            assert(np.all(allcat['ROW_PARENT'] == cat['ROW_PARENT']))
            allcat.write(outfile, clobber=True)
            print(f'Wrote {len(allcat):,d} objects to {outfile}')

            #outfile = os.path.join(outdir, f'wiseize-nedlvs-{region}.fits')
            #nedlvs.write(outfile, clobber=True)
            #print(f'Wrote {len(nedlvs):,d} objects to {outfile}')

        return cat, fullcat


def read_sample(first=None, last=None, galaxylist=None, verbose=False, columns=None,
                final_sample=False, region='dr11-south', d25min=0., d25max=100.0):
    """Read/generate the parent SGA catalog.

    d25min,d25max in arcmin

    """
    import fitsio
    from SGA.coadds import REGIONBITS

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
