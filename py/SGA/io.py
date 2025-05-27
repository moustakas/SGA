"""
SGA.io
======

Code to read and write the various SGA files.

"""
import os, sys, time, pdb
import fitsio
import numpy as np
from astropy.table import Table, vstack

from SGA.log import get_logger#, DEBUG
log = get_logger()


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
        raise EnvironmentError('Mandatory LEGACY_SURVEY_BASEDIR environment variable not set!')
    print('WARNING: Using dr11-early temporarly directory for dr11-south!!')
    dirs = {'dr9-north': 'dr9', 'dr9-south': 'dr9', 'dr10-south': 'dr10', 'dr11-south': 'dr11-early'}
    legacy_survey_dir = os.path.join(os.getenv('LEGACY_SURVEY_BASEDIR'), dirs[region])
    print(f'Setting LEGACY_SURVEY_DIR={legacy_survey_dir}')
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


def radec_to_name(target_ra, target_dec, prefix='SGA2025', unixsafe=False):
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

    # Number of decimal places in final naming convention
    precision = 4

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

    return names


def get_galaxy_galaxydir(sample=None, bricks=None, datadir=None,
                         htmldir=None, html=False):
    """Retrieve the galaxy name and the (nested) directory.

    """
    if sample is None and bricks is None:
        msg = 'Must provide either sample or bricks.'
        raise IOError(msg)

    if datadir is None:
        datadir = sga_data_dir()
    if htmldir is None:
        htmldir = sga_html_dir()

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
            galcolumn = 'GALAXY'
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


def backup_filename(filename):
    """rename filename to next available filename.N

    Args:
        filename (str): full path to filename

    Returns:
        New filename.N, or filename if original file didn't already exist
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


def altnames_hyperleda(cat):
    """Retrieve the alternate name for the HyperLeda catalog.

    """
    altname = []
    for altnames in cat['ALTNAMES'].value:
        altnames = np.array(altnames.split(','))
        # prefer the first 2MASS name
        M = ['2MASS' in name or '2MASX' in name for name in altnames]
        if np.any(M):
            altname.append(altnames[M][0])
        else:
            #C = [':' in name  for name in altnames]
            #if np.any(C):
            #    print(altnames)
            altname.append(altnames[0])

    return np.array(altname)


def nedfriendly_hyperleda(old, reverse=False):
    """Rename some of the HyperLeda names to be NED-friendly.

    E.g., NED does not resolve HyperLeda's name 3C218A, but this is also
    PGC 26269, which NED *does* resolve--
      http://atlas.obs-hp.fr/hyperleda/ledacat.cgi?o=PGC%2026269
      https://ned.ipac.caltech.edu/byname?objname=PGC+26269

    """
    if reverse:
        return np.char.replace(old, 'SDSS J', 'SDSSJ') # ???
    else:
        return np.char.replace(old, 'SDSSJ', 'SDSS J') # ???

    #new = old.copy()
    #I = pgc <= 73197
    #if np.any(I):
    #    new[I] = [f'PGC{num:06d}' for num in pgc[I].astype(int)]
    #return new


def version_hyperleda_noobjtype():
    return 'meandata_noobjtype_1739457147'


def read_hyperleda_noobjtype(rank=0, rows=None):
    """Read the HyperLeda_noobjtype catalog.

    """
    version = version_hyperleda_noobjtype()
    hyperfile = os.path.join(sga_dir(), 'parent', 'external', f'HyperLeda_{version}.fits')

    if not os.path.isfile(hyperfile):
        txtfile = hyperfile.replace('.fits', '.txt')

        with open(txtfile, 'r') as F:
            nrows = len(F.readlines())

        if version == 'meandata_noobjtype_1739457147':
            header_start = 21
            data_start = 23
            data_offset = 5 # 4846383-20
            delimiter = '|'
        else:
            raise ValueError(f'Unknown version {version}')

        hyper = Table.read(txtfile, format='ascii.csv', data_start=data_start,
                           data_end=nrows-data_offset, header_start=header_start,
                           delimiter=delimiter)

        hyper.rename_column('hl_names(pgc)', 'ALTNAMES')
        [hyper.rename_column(col, col.upper()) for col in hyper.colnames]

        hyper['ROW'] = np.arange(len(hyper))

        nhyper = len(hyper)
        print(f'Read {nhyper:,d} objects from {hyperfile}')
        assert(nhyper == len(np.unique(hyper['PGC'])))

        hyper.rename_columns(['AL2000', 'DE2000'], ['RA', 'DEC'])
        hyper['RA'] *= 15. # [decimal degrees]

        # three objects have nan coordinates
        hyper = hyper[~hyper['RA'].mask]

        # get just the first three alternate names
        altnames = []
        for iobj in range(len(hyper)):
            objname = hyper['OBJNAME'][iobj]
            names = np.array(hyper['ALTNAMES'][iobj].split(','))
            # remove the primary name
            names = names[~np.isin(names, objname)]
            names = ','.join(names[:3]) # top 3
            altnames.append(names)
        hyper['ALTNAMES'] = altnames

        # re-sort by PGC number
        hyper = hyper[np.argsort(hyper['PGC'])]

        print(f'Writing {len(hyper):,d} objects to {hyperfile}')
        hyper.write(hyperfile, overwrite=True)

    hyper = Table(fitsio.read(hyperfile, rows=rows))
    print(f'Read {len(hyper):,d} objects from {hyperfile}')
    #print(f'Rank {rank:03d}: Read {len(hyper):,d} objects from {hyperfile}')

    return hyper


def version_hyperleda_multiples():
    return 'meandata_multiples_1727885775'


def read_hyperleda_multiples(rank=0, rows=None):
    """Read the HyperLeda_multiples catalog.

    """
    version = version_hyperleda_multiples()
    hyperfile = os.path.join(sga_dir(), 'parent', 'external', f'HyperLeda_{version}.fits')

    if not os.path.isfile(hyperfile):
        txtfile = hyperfile.replace('.fits', '.txt')

        with open(txtfile, 'r') as F:
            nrows = len(F.readlines())

        if version == 'meandata_multiples_1727885775':
            header_start = 21
            data_start = 23
            data_offset = 5 # 4846383-20
            delimiter = '|'
        else:
            raise ValueError(f'Unknown version {version}')

        hyper = Table.read(txtfile, format='ascii.csv', data_start=data_start,
                           data_end=nrows-data_offset, header_start=header_start,
                           delimiter=delimiter)

        hyper.rename_column('hl_names(pgc)', 'ALTNAMES')
        [hyper.rename_column(col, col.upper()) for col in hyper.colnames]

        hyper['ROW'] = np.arange(len(hyper))

        nhyper = len(hyper)
        print(f'Read {nhyper:,d} objects from {hyperfile}')
        assert(nhyper == len(np.unique(hyper['PGC'])))

        hyper.rename_columns(['AL2000', 'DE2000'], ['RA', 'DEC'])
        hyper['RA'] *= 15. # [decimal degrees]

        # get just the first three alternate names
        altnames = []
        for iobj in range(len(hyper)):
            objname = hyper['OBJNAME'][iobj]
            names = np.array(hyper['ALTNAMES'][iobj].split(','))
            # remove the primary name
            names = names[~np.isin(names, objname)]
            names = ','.join(names[:3]) # top 3
            altnames.append(names)
        hyper['ALTNAMES'] = altnames

        # re-sort by PGC number
        hyper = hyper[np.argsort(hyper['PGC'])]

        print(f'Writing {len(hyper):,d} objects to {hyperfile}')
        hyper.write(hyperfile, overwrite=True)

    hyper = Table(fitsio.read(hyperfile, rows=rows))
    print(f'Read {len(hyper):,d} objects from {hyperfile}')
    #print(f'Rank {rank:03d}: Read {len(hyper):,d} objects from {hyperfile}')

    return hyper


def version_hyperleda_galaxies():
    #return 'meandata_1718379336'
    #return 'meandata_1720804662'
    return 'meandata_galaxies_1725482144'


def read_hyperleda_galaxies(rank=0, rows=None):
    """Read the HyperLeda 'G' and 'g' catalog.

    Feedback for Dmitry:

    * 87 duplicated entries (see below)

    """
    version = version_hyperleda_galaxies()
    hyperfile = os.path.join(sga_dir(), 'parent', 'external', f'HyperLeda_{version}.fits')

    if not os.path.isfile(hyperfile):
        txtfile = hyperfile.replace('.fits', '.txt')

        with open(txtfile, 'r') as F:
            nrows = len(F.readlines())

        if version == 'meandata_galaxies_1718379336':
            header_start = 20
            data_start = 22
            data_offset = 5
            delimiter = ','
        elif version == 'meandata_galaxies_1720804662':
            header_start = 22
            data_start = 24
            data_offset = 7 # 4846383-20
            delimiter = '|'
        elif version == 'meandata_galaxies_1725482144':
            header_start = 21
            data_start = 23
            data_offset = 5 # 4846383-20
            delimiter = '|'

        hyper = Table.read(txtfile, format='ascii.csv', data_start=data_start,
                           data_end=nrows-data_offset, header_start=header_start,
                           delimiter=delimiter)

        if version == 'meandata_galaxies_1720804662' or version == 'meandata_galaxies_1725482144':
            hyper.rename_column('hl_names(pgc)', 'ALTNAMES')
        #hyper.remove_column('f_astrom')

        [hyper.rename_column(col, col.upper()) for col in hyper.colnames]

        hyper['ROW'] = np.arange(len(hyper))

        nhyper = len(hyper)
        print(f'Read {nhyper:,d} objects from {hyperfile}')
        assert(nhyper == len(np.unique(hyper['PGC'])))

        # There are 87 duplicated object names. The entries are either
        # identical or they differ mildly in their redshift, B-band magnitude,
        # or position angle. We do not attempt to merge or average the data; we
        # simply choose the first entry.
        objs, uindx, count = np.unique(hyper['OBJNAME'], return_counts=True, return_index=True)
        #dups = objs[count > 1]
        #for dup in dups:
        #    I = np.where(hyper['OBJNAME'] == dup)[0]
        #    print(hyper[I])
        #    print()

        hyper = hyper[uindx]
        print(f'Trimming to {len(hyper):,d}/{nhyper:,d} unique objects.')

        hyper.rename_columns(['AL2000', 'DE2000'], ['RA', 'DEC'])
        hyper['RA'] *= 15. # [decimal degrees]

        ## objects with all three of diameter, Bt-mag, and redshift
        #J = np.logical_and.reduce((hyper['LOGD25'].mask, hyper['BT'].mask, hyper['v'].mask))
        #
        ## objects with *none* of diameter, Bt-mag, or redshift, which are most likely to be stars or spurious
        #I = np.logical_or.reduce((~hyper['LOGD25'].mask, ~hyper['BT'].mask, ~hyper['v'].mask))

        # get just the first three alternate names
        altnames = []
        for iobj in range(len(hyper)):
            objname = hyper['OBJNAME'][iobj]
            names = np.array(hyper['ALTNAMES'][iobj].split(','))
            # remove the primary name
            names = names[~np.isin(names, objname)]
            names = ','.join(names[:3]) # top 3
            altnames.append(names)
        hyper['ALTNAMES'] = altnames

        # re-sort by PGC number
        hyper = hyper[np.argsort(hyper['PGC'])]

        print(f'Writing {len(hyper):,d} objects to {hyperfile}')
        hyper.write(hyperfile, overwrite=True)

    hyper = Table(fitsio.read(hyperfile, rows=rows))
    print(f'Read {len(hyper):,d} objects from {hyperfile}')
    #print(f'Rank {rank:03d}: Read {len(hyper):,d} objects from {hyperfile}')

    return hyper


def version_hyperleda():
    return 'meandata_v1.0'


def read_hyperleda(rank=0, rows=None):
    """Read the complete (galaxies + multiples) HyperLeda catalog.

    """
    version = version_hyperleda()
    hyperfile = os.path.join(sga_dir(), 'parent', 'external', f'HyperLeda_{version}.fits')

    if not os.path.isfile(hyperfile):
        gals = read_hyperleda_galaxies(rank=rank, rows=rows)

        # The 'mult' table has objects with OBJTYPE==? (where
        # multiple={M,M2,M3}); remove them here since we get them in
        # the 'noobjtype' table, too.
        mult = read_hyperleda_multiples(rank=rank, rows=rows)
        mult = mult[mult['OBJTYPE'] != '?']

        noobj = read_hyperleda_noobjtype(rank=rank, rows=rows)
        #gals['ROW_GALAXIES'] = np.zeros(len(gals), np.int64) - 99
        #mult['ROW_MULTIPLES'] = np.zeros(len(mult), np.int64) - 99
        gals.rename_column('ROW', 'ROW_GALAXIES')
        mult.rename_column('ROW', 'ROW_MULTIPLES')
        noobj.rename_column('ROW', 'ROW_NOOBJTYPE')
        hyper = vstack((gals, mult, noobj))
        hyper['ROW_GALAXIES'].fill_value = -99
        hyper['ROW_MULTIPLES'].fill_value = -99
        hyper['ROW_NOOBJTYPE'].fill_value = -99
        hyper = hyper.filled()

        # sort by PGC number and reset ROW
        hyper = hyper[np.argsort(hyper['PGC'])]
        hyper['ROW'] = np.arange(len(hyper))

        print(f'Writing {len(hyper):,d} objects to {hyperfile}')
        hyper.write(hyperfile, overwrite=True)

    hyper = Table(fitsio.read(hyperfile, rows=rows))
    print(f'Read {len(hyper):,d} objects from {hyperfile}')
    #print(f'Rank {rank:03d}: Read {len(hyper):,d} objects from {hyperfile}')

    return hyper


def nedfriendly_lvd(old):
    """Rename some of the LVD names to be NED-friendly.

    E.g., Canes Venatici I is CVn I dSph

    from SGA.io import read_lvd ; lvd = read_lvd() ; lvd = lvd[np.argsort(np.char.lower(lvd['OBJNAME']))]
    ned = Table(fitsio.read('NEDbyname-LVD_v1.0.2.fits'))

    for gal in lvd['OBJNAME'].value:
        nedgal = ned[gal == ned['OBJNAME']]['OBJNAME_NED']
        if len(nedgal) > 0:
            _ = print(f"        '{gal}': '{nedgal[0]}',")
        else:
            _ = print(f"        '{gal}': '',")

    """
    ned = {
        '[KKH2011]S11': '[KKH2011] S11',
        '[TT2009] 25': '[TT2009] 25',
        '[TT2009] 30': '[TT2009] 30',
        'A0952+69': 'AO 0952+69',
        'AGC 112521': 'AGC 112521',
        'AGC 238890': 'AGC 238890',
        'AGC 239141': 'AGC 239141',
        'AGC 749241': 'AGC 749241',
        'AGC749235': 'AGC 749235',
        'AM 1320-230': 'AM 1320-230',
        'Andromeda I': 'Andromeda I',
        'Andromeda II': 'Andromeda II',
        'Andromeda III': 'Andromeda III',
        'Andromeda IV': 'Andromeda IV',
        'Andromeda IX': 'Andromeda IX',
        'Andromeda V': 'Andromeda V',
        'Andromeda VI': 'Andromeda VI',
        'Andromeda VII': 'Andromeda VII',
        'Andromeda X': 'Andromeda X',
        'Andromeda XI': 'Andromeda XI',
        'Andromeda XII': 'Andromeda XII',
        'Andromeda XIII': 'Andromeda XIII',
        'Andromeda XIV': 'Andromeda XIV',
        'Andromeda XIX': 'Andromeda XIX',
        'Andromeda XV': 'Andromeda XV',
        'Andromeda XVI': 'Andromeda XVI',
        'Andromeda XVII': 'Andromeda XVII',
        'Andromeda XVIII': 'Andromeda XVIII',
        'Andromeda XX': 'Andromeda XX',
        'Andromeda XXI': 'Andromeda XXI',
        'Andromeda XXII': 'Andromeda XXII',
        'Andromeda XXIII': 'Andromeda XXIII',
        'Andromeda XXIV': 'Andromeda XXIV',
        'Andromeda XXIX': 'Andromeda XXIX',
        'Andromeda XXV': 'Andromeda XXV',
        'Andromeda XXVI': 'Andromeda XXVI',
        'Andromeda XXVII': 'Andromeda XXVII',
        'Andromeda XXVIII': 'Andromeda XXVIII',
        #'Andromeda XXXV': '', not in NED
        'Antlia': 'Antlia Dwarf Spheroidal',
        'Antlia B': 'Antlia B',
        'Antlia II': 'Antlia II Dwarf',
        'Aquarius': 'Aquarius dIrr',
        'Aquarius II': 'Aquarius II',
        #'Aquarius III': '', not in NED
        'Bedin 1': 'Bedin I',
        'BK3N': 'BK 03N',
        'BK5N': 'BK 05N',
        'BK6N': 'BK 06N',
        'Bootes I': 'Bootes Satellite',
        'Bootes II': 'Bootes II',
        'Bootes III': 'Bootes III',
        'Bootes IV': 'Bootes IV Dwarf',
        'Bootes V': 'Bootes V Dwarf',
        'BTS 116': 'WISEA J121857.24+283311.1',
        'BTS151': 'BTS 151',
        'Camelopardalis A': 'Cam A',
        'Camelopardalis B': 'Camelopardalis B',
        'Canes Venatici I': 'CVn I dSph',
        'Canes Venatici II': 'CVn II dSph',
        'Carina': 'Carina dSph',
        'Carina II': 'Carina II Dwarf',
        'Carina III': 'Carina III Dwarf',
        'Cassiopea dIrr 1': 'Cas 1',
        'Cassiopeia II': 'Andromeda XXX', # NED incorrectly matches to Cas II=[MIM2013] 009 with incorrect coordinates
        'Cassiopeia III': 'Cas III',
        'CenA-MM-Dw1': 'CenA-Dw-133013-415321',
        'CenA-MM-Dw2': 'CenA-Dw-132956-415220',
        'CenA-MM-Dw3': 'Centaurus A:[CSS2016] MM-Dw03', # HyperLeda matches to WISEA J133020.71-421130.6
        'CenA-MM-Dw4': 'CenA-Dw-132302-414705',
        'CenA-MM-Dw5': 'CenA-Dw-131952-415938',
        'CenA-MM-Dw6': 'CenA-Dw-132557-410538',
        'CenA-MM-Dw7': 'CenA-Dw-132628-433318',
        'Centaurus I': 'Centaurus I Dwarf',
        'Centaurus N': 'Cen N',
        'Cepheus1': 'Cepheus 1',
        'Cetus': 'Cetus Dwarf Spheroidal', # NED incorrectly matches Cetus to Cetus II!
        'Cetus II': 'Cetus II Dwarf',
        'CGCG 189-050': 'CGCG 189-050',
        'CGCG 217-018': 'CGCG 217-018',
        'Clump I': 'PGC1 0028630 NED040',
        'Clump III': 'PGC1 0028630 NED039',
        'Columba I': 'Columba I Dwarf',
        'Coma Berenices': 'Coma Berenices Dwarf',
        'Corvus A': 'Corvus A',
        'Crater II': 'Crater II Dwarf',
        'd0926+70': 'MESSIER 081:[CKT2009] d0926+70',
        'd0934+70': 'MESSIER 081:[CKT2009] d0934+70',
        'd0939+71': 'MESSIER 081:[CKT2009] d0939+71',
        'd0944+69': 'MESSIER 081:[CKT2009] d0944+69',
        'd0944+71': 'MESSIER 081:[CKT2009] d0944+71',
        'd0955+70': 'MESSIER 081:[CKT2009] d0955+70',
        'd0958+66': 'MESSIER 081:[CKT2009] d0958+66',
        'd0959+68': 'MESSIER 081:[CKT2009] d0959+68',
        'd1006+67': 'MESSIER 081:[CKT2009] d1006+67',
        'd1014+68': 'MESSIER 081:[CKT2009] d1014+68',
        'd1015+69': 'MESSIER 081:[CKT2009] d1015+69',
        'd1028+70': 'MESSIER 081:[CKT2009] d1028+70',
        'd1041+70': 'MESSIER 081:[CKT2009] d1041+70',
        'DDO 113': 'DDO 113',
        'DDO 125': 'DDO 125',
        'DDO 126': 'DDO 126',
        'DDO 127': 'DDO 127',
        'DDO 133': 'DDO 133',
        'DDO 147': 'DDO 147',
        'DDO 153': 'DDO 153',
        'DDO 154': 'DDO 154',
        'DDO 161': 'DDO 161',
        'DDO 165': 'DDO 165',
        'DDO 167': 'DDO 167',
        'DDO 168': 'DDO 168',
        'DDO 169': 'DDO 169',
        'DDO 169NW': 'DDO 169NW',
        'DDO 181': 'DDO 181',
        'DDO 183': 'DDO 183',
        'DDO 190': 'DDO 190',
        'DDO 226': 'DDO 226',
        'DDO 44': 'DDO 044',
        'DDO 53': 'DDO 053',
        'DDO 6': 'DDO 006',
        'DDO 78': 'DDO 78',
        'DDO 82': 'DDO 82',
        'DDO 99': 'DDO 99',
        'Donatiello III': 'Donatiello III',
        'Donatiello IV': 'Donatiello IV',
        'Draco': 'Draco Dwarf',
        'Draco II': 'Draco II',
        'dw0036m2828': '[CGB2022] dw 0036-2828',
        'dw0132+1422': '[CGB2022] dw 0132+1422',
        'dw0133p1543': '[CGB2022] dw 0133+1543',
        'dw0134+1438': '[CGB2022] dw 0134+1438',
        'dw0134p1544': '[CGB2022] dw 0134+1544',
        'dw0136p1628': '[CGB2022] dw 0136+1628',
        'dw0137p1607': '[CGB2022] dw 0137+1607',
        'dw0138+1458': '[CGB2022] dw 0138+1458',
        'dw0139p1433': '[CGB2022] dw 0139+1433',
        'dw0140p1556': '[CGB2022] dw 0140+1556',
        'dw0235p3850': '[CGB2022] dw 0235+3850',
        'dw0237p3836': '[CGB2022] dw 0237+3836',
        'dw0237p3855': '[CGB2022] dw 0237+3855',
        'dw0239p3902': '[CGB2022] dw 0239+3902',
        'dw0239p3903': '[CGB2022] dw 0239+3903',
        'dw0239p3926': '[CGB2022] dw 0239+3926',
        'dw0240p3854': '[CGB2022] dw 0240+3854',
        'dw0240p3922': '[CGB2022] dw 0240+3922',
        'dw0241p3904': '[CGB2022] dw 0241+3904',
        'dw0242p3838': '[CGB2022] dw 0242+3838',
        'dw0506m3739': '[CGB2022] dw 0506-3739',
        'dw0507m3629': '[CGB2022] dw 0507-3629',
        'dw0507m3739': '[CGB2022] dw 0507-3739',
        'dw0507m3744': '[CGB2022] dw 0507-3744',
        'dw0507m3800': '[CGB2022] dw 0507-3800',
        'dw0508m3617': '[CGB2022] dw 0508-3617',
        'dw0508m3808': '[CGB2022] dw 0508-3808',
        'dw0929+2213': '[CGB2022] dw 0929+2213',
        'dw0932+1952': '[CGB2022] dw 0932+1952',
        'dw0936+2135': '[CGB2022] dw 0936+2135',
        'dw1000-0741': '[CGB2022] dw 1000-0741',
        'dw1000m0821': '[CGB2022] dw 1000-0821',
        'dw1000m0831': '[CGB2022] dw 1000-0831',
        'dw1002m0642': '[CGB2022] dw 1002-0642',
        'dw1002m0818': '[CGB2022] dw 1002-0818',
        'dw1004m0657': '[CGB2022] dw 1004-0657',
        'dw1004m0737': '[CGB2022] dw 1004-0737',
        'dw1006m0730': '[CGB2022] dw 1006-0730',
        'dw1006m0730-n2': '[KMA2022] dw 1006-0730-n2',
        'dw1006m0732': '[CGB2022] dw 1006-0732',
        'dw1007m0715': '[CGB2022] dw 1007-0715',
        'dw1007m0830': '[CGB2022] dw 1007-0830',
        'dw1007m0835': '[CGB2022] dw 1007-0835',
        'dw1042p1359': '[CGB2022] dw 1042+1359',
        'dw1043p1410': '[CGB2022] dw 1043+1410',
        'dw1043p1415': '[CGB2022] dw 1043+1415',
        'dw1044p1351': '[CGB2022] dw 1044+1351',
        'dw1044p1351b': 'ELVES dw J1044p1351b',
        'dw1044p1356': '[CGB2022] dw 1044+1356',
        'dw1044p1359': '[CGB2022] dw 1044+1359',
        'dw1046+1244': '[CGB2022] dw 1046+1244',
        'dw1047p1153': '[CGB2022] dw 1047+1153',
        'dw1047p1258': '[CGB2022] dw 1047+1258',
        'dw1048p1154': '[CGB2022] dw 1048+1154',
        'dw1048p1259': '[CGB2022] dw 1048+1259',
        'dw1048p1303': '[CGB2022] dw 1048+1303',
        'dw1048p1407': '[CGB2022] dw 1048+1407',
        'dw1049p1233': '[CGB2022] dw 1049+1233',
        'dw1049p1247': '[CGB2022] dw 1049+1247',
        'dw1050p1213': '[CGB2022] dw 1050+1213',
        'dw1050p1236': '[CGB2022] dw 1050+1236',
        'dw1051p1406': '[CGB2022] dw 1051+1406',
        'dw1052p1102': '[CGB2022] dw 1052+1102',
        'dw1104+0004': '[CGB2022] dw 1104+0004',
        'dw1104+0005': '[CGB2022] dw 1104+0005',
        'dw1106-0052': '[CGB2022] dw 1106-0052',
        'dw1109-0016': '[CGB2022] dw 1109-0016',
        'dw1110+0037': '[CGB2022] dw 1110+0037',
        'dw1111+0049': '[CGB2022] dw 1111+0049',
        'dw1114p1238': '[CGB2022] dw 1114+1238',
        'dw1118p1233': '[CGB2022] dw 1118+1233',
        'dw1118p1348': '[CGB2022] dw 1118+1348',
        'dw1119p1157': '[CGB2022] dw 1119+1157',
        'dw1119p1404': '[CGB2022] dw 1119+1404',
        'dw1119p1417': '[CGB2022] dw 1119+1417',
        'dw1119p1419': '[CGB2022] dw 1119+1419',
        'dw1120p1332': '[CGB2022] dw 1120+1332',
        'dw1120p1337': '[CGB2022] dw 1120+1337',
        'dw1121p1326': '[CGB2022] dw 1121+1326',
        'dw1122p1258': '[CGB2022] dw 1122+1258',
        'dw1123p1342': '[CGB2022] dw 1123+1342',
        'dw1124p1240': '[CGB2022] dw 1124+1240',
        'dw1220+4649': '[CGB2022] dw 1220+4649',
        'dw1227p0136': '[CGB2022] dw 1227+0136',
        'dw1231p0140': '[CGB2022] dw 1231+0140',
        'dw1232p0015': '[CGB2022] dw 1232+0015',
        'dw1234p2531': '[CGB2022] dw 1234+2531',
        'dw1236+3336': '[CGB2022] dw 1236+3336',
        'dw1236m0025': '[CGB2022] dw 1236-0025',
        'dw1237p2602': '[CGB2022] dw 1237+2602',
        'dw1238m0035': '[CGB2022] dw 1238-0035',
        'dw1238m0105': '[CGB2022] dw 1238-0105',
        'dw1238p0028': '[CGB2022] dw 1238+0028',
        'dw1239+3230': '[CGB2022] dw 1239+3230',
        'dw1239+3251': '[CGB2022] dw 1239+3251',
        'dw1239-1159': '[CGB2022] dw 1239-1159',
        'dw1240-1118': '[CGB2022] dw 1240-1118',
        'Dw1245+6158': 'SMDG J1245495+615810', # 'SDSS J124551.28+615816.9' ??
        'dw1300+1843': '[CGB2022] dw 1300+1843', # 'WISEA J130030.82+184304.5'
        'dw1310+4153': '[CGB2022] dw 1310+4153',
        'dw1315+4123': '[CGB2022] dw 1315+4123',
        'dw1321+4226': '[CGB2022] dw 1321+4226',
        'dw1322-39': 'Cen A:[MJB2017] dw 1322-39',
        'dw1323-40a': '[CGG2021] dw 1323-40', # note: "40" not "40a"
        'dw1323-40b': '[MJB2017] dw 1323-40b',
        'dw1328+4703': '[CGB2022] dw 1328+4703',
        'dw1329-45': 'Cen A:[MJP2016] dw 1329-45',
        'dw1330+4731': '[CGB2022] dw 1330+4731',
        'dw1335-29': '[MJB2015] dw 1335-29',
        'dw1336-44': 'Cen A:[MJP2016] dw 1336-44',
        'dw1340-30': '[MJB2015] dw 1340-30',
        'dw1341-29': '[MPR2024] dw 1341-29',
        'dw1341-43': 'Cen A:[MJP2016] dw 1341-43',
        'dw1342-43': 'Cen A:[MJP2016] dw 1342-43',
        'dw1343+58': 'WISEA J134307.12+581340.2', # ??
        'Eridanus II': 'Eridanus II Dwarf',
        'Eridanus IV': 'Eridanus IV Dwarf',
        'ESO 006-001': 'ESO 006- G 001',
        'ESO 059-001': 'ESO 059- G 001',
        'ESO 115-021': 'ESO 115- G 021',
        'ESO 121-020': 'ESO 121- G 020',
        'ESO 137-018': 'ESO 137- G 018',
        'ESO 154-023': 'ESO 154- G 023',
        'ESO 199-007': 'ESO 199- G 007',
        'ESO 215-009': 'ESO 215- G 009',
        'ESO 222-010': 'ESO 222- G 010',
        'ESO 245-005': 'ESO 245- G 005',
        'ESO 269-037': 'ESO 269- G 037',
        'ESO 269-058': 'ESO 269- G 058',
        'ESO 269-066': 'ESO 269- G?066',
        'ESO 272-025': 'ESO 272- G 025',
        'ESO 273-014': 'ESO 273- G 014',
        'ESO 274-001': 'ESO 274- G 001',
        'ESO 290-028': 'ESO 290- G 028',
        'ESO 294-G010': 'ESO 294- G 010',
        'ESO 300-016': 'ESO 300- G 016',
        'ESO 301-007': 'ESO 301- G 007',
        'ESO 301-010': 'ESO 301- G 010',
        'ESO 320-014': 'ESO 320- G 014',
        'ESO 321-014': 'ESO 321- G 014',
        'ESO 324-024': 'ESO 324- G 024',
        'ESO 325-011': 'ESO 325- G?011',
        'ESO 346-007': 'ESO 346- G 007',
        'ESO 349-031': 'ESO 349- G 031',
        'ESO 379-007': 'ESO 379- G 007',
        'ESO 379-024': 'ESO 379- G 024',
        'ESO 381-018': 'ESO 381- G 018',
        'ESO 381-020': 'ESO 381- G 020',
        'ESO 383-087': 'ESO 383- G 087',
        'ESO 384-016': 'ESO 384- G 016',
        'ESO 410-G005': 'ESO 410- G 005',
        'ESO 443-009': 'ESO 443- G 009',
        'ESO 444-084': 'ESO 444- G 084',
        'ESO 472-015': 'ESO 472- G 015',
        'ESO 540-032': 'ESO 540- G 032',
        'ESO104-044': 'ESO104- G 044',
        'ESO219-010': 'ESO219- G 010',
        'F8D1': 'F08D1',
        'Fluffy': '[OSG2024] Fluffy',
        'FM1': 'F06D1',
        'Fornax': 'Fornax Dwarf Spheroidal',
        'FS04': '[CGB2022] dw 1042+1220', # 'CGCG 065-086'
        'GALFA Dw3': 'GALFA-Dw3', # 'GALFA J044.7+13.6+528',
        'GALFA Dw4': 'GALFA-Dw4', # 'PGC1 5072715 NED001',
        'GARLAND': 'The Garland',
        'GHOSTS I': 'GHOSTS 1',
        'GR 8': 'GR 8',
        'Grapes': '[KKM2018a] Grapes',
        'Grus I': 'Grus I Dwarf',
        'Grus II': 'Grus II Dwarf',
        'Hercules': 'Hercules dSph',
        'HIDEEP J1337-3320': 'HIDEEP J1337-33',
        'HIPASS J1131-31': 'HIPASS J1131-31', # 'WISEA J113134.59-314013.2',
        'HIPASS J1133-32': 'HIPASS J1133-32',
        'HIPASS J1247-77': 'HIPASS J1247-77', # 'PGC1 0039573 NED002',
        'HIPASS J1337-39': 'HIPASS J1337-39', # 'GALEXASC J133725.28-395345.4',
        'HIPASS J1348-37': 'HIPASS J1348-37',
        'HIPASS J1351-47': 'HIPASS J1351-47',
        'HIZSS-021': 'HIZSS 021',
        'Holm IV': 'Holmberg IV',
        'Holmberg I': 'Holmberg I',
        'Holmberg II': 'Holmberg II',
        'Holmberg IX': 'Holmberg IX',
        'Horologium I': 'Horologium I Dwarf',
        'Horologium II': 'Horologium II Dwarf',
        'HS 117': '[HS98] 117',
        'HSC-10': '[CGB2022] dw 1242+3158',
        'HSC-9': '[CGB2022] dw 1240+3216',
        'Hydra II': 'Hydra II Dwarf',
        'Hydrus I': 'Hydrus I Dwarf',
        'IC 10': 'IC 10',
        'IC 1613': 'IC 1613',
        'IC 2574': 'IC 2574',
        'IC 2684': 'IC 2684',
        'IC 2782': 'IC 2782',
        'IC 2787': 'IC 2787',
        'IC 2791': 'IC 2791',
        'IC 3104': 'IC 3104',
        'IC 3687': 'IC 3687',
        'IC 3840': 'IC 3840',
        'IC 4182': 'IC 4182',
        'IC 4247': 'IC 4247',
        'IC 4316': 'IC 4316',
        'IC 4662': 'IC 4662',
        'IC 5052': 'IC 5052',
        'IC 5152': 'IC 5152',
        'IC1959': 'IC 1959',
        'IC239': 'IC 239',
        'IC3571': 'IC 3571',
        'IKN': 'IKN', # 'PGC1 0028630 NED014',
        'JKB129': 'JKB 129',
        'JKB142': 'JKB 142', # 'SDSS J014548.23+162240.6',
        'JKB83': 'JKB 3', # 'SDSS J095549.64+691957.4',
        'KDG 171': 'KDG 171',
        'KDG 2': 'KDG 2', # 'ESO 540- G 030',
        'KDG 215': 'KDG 215', # 'LSBC D575-05',
        'KDG 52': 'KDG 52', # 'MESSIER 081 DWARF A',
        'KDG 56': 'KDG 56',
        'KDG 61': 'KDG 61',
        'KDG 63': 'KDG 63', # 'UGC 05428',
        'KDG 64': 'KDG 64', # 'UGC 05442',
        'KDG 73': 'KDG 73', # 'GALEXASC J105256.68+693258.1',
        'KDG 74': 'KDG 74', # 'WISEA J110222.39+701551.0',
        'KDG010': 'KDG 10',
        'KK 109': '[KK98] 109',
        'KK 132': '[KK98] 132',
        'KK 135': '[KK98] 135',
        'KK 144': '[KK98] 144',
        'KK 153': '[KK98] 153',
        'KK 16': '[KK98] 16',
        'KK 160': '[KK98] 160',
        'KK 165': '[KK98] 165',
        'KK 166': '[KK98] 166',
        'KK 17': '[KK98] 17',
        'KK 176': '[KK98] 176',
        'KK 177': '[KK98] 177', # IC 4107; HyperLeda matches to WISEA J130241.76+215952.2
        'KK 180': '[KK98] 180',
        'KK 182': '[KK98] 182',
        'KK 189': '[KK98] 189',
        'KK 191': '[KK98] 191',
        'KK 193': '[KK98] 193',
        'KK 195': '[KK98] 195',
        'KK 196': '[KK98] 196',
        'KK 197': '[KK98] 197',
        'KK 200': '[KK98] 200',
        'KK 203': '[KK98] 203',
        'KK 208': '[KK98] 208',
        'KK 211': '[KK98] 211',
        'KK 213': '[KK98] 213',
        'KK 217': '[KK98] 217',
        'KK 218': '[KK98] 218',
        'KK 221': '[KK98] 221',
        'KK 242': '[KK98] 242',
        'KK 258': '[KK98] 258',
        'KK 27': '[KK98] 27',
        'KK 35': '[KK98] 35',
        'KK 69': '[KK98] 69',
        'KK 70': '[KK98] 70',
        'KK 77': '[KK98] 77',
        'KK 93': '[KK98] 93',
        'KK94': '[KK98] 94',
        'KK96': '[KK98] 96',
        'KKH 11': 'KKH 11',
        'KKH 12': 'KKH 12',
        'KKH 18': 'KKH 18', # 'PGC1 2807110 NED001',
        'KKH 22': 'KKH 22', # 'PGC1 0013826 NED008',
        'KKH 30': 'KKH 30', # 'WISEA J051743.29+813724.2',
        'KKH 34': 'KKH 34',
        'KKH 37': 'KKH 37', # 'WISEA J064745.67+800725.1',
        'KKH 5': 'KKH 5',
        'KKH 57': 'KKH 57', # 'SDSS J100015.40+631101.3',
        'KKH 6': 'KKH 6', # 'WISEA J013451.63+520530.8',
        'KKH 78': 'KKH 78', # 'WISEA J121745.51+332035.4',
        'KKH 86': 'KKH 86', # GALEXASC J135433.63+041438.4',
        'KKH 98': 'KKH 98', # GALEXMSC J234534.33+384303.7',
        'KKR 25': 'KKR 25', # PGC1 2801026 NED001',
        'KKR 3': 'KKR 3', # 'PGC1 0166185 NED001',
        'KKs 3': '[KK2000] 3',
        'KKs 51': '[KK2000] 51',
        'KKs 53': '[KK2000] 53',
        'KKs 54': '[KK2000] 54',
        'KKs 55': '[KK2000] 55',
        'KKs 57': '[KK2000] 57',
        'KKs 58': '[KK2000] 58',
        'KKs 59': '[KK2000] 59',
        'KKSG 17': '[KKS2000] 17',
        'KKSG 18': '[KKS2000] 18',
        'KKSG 20': '[KKS2000] 20',
        'KKSG 22': '[KKS2000] 22',
        'KKSG 29': '[KKS2000] 29',
        'KKSG 31': '[KKS2000] 31',
        'KKSG 32': '[KKS2000] 32',
        'KKSG 33': '[KKS2000] 33',
        'KKSG 37': '[KKS2000] 37',
        'KV19-212': '[OSG2024] KV19-212',
        'KV19-271': '[OSG2024] KV19-271',
        'KV19-329': '[OSG2024] KV19-329',
        'KV19-442': '[OSG2024] KV19-442',
        'Lacerta I': 'Lacerta I', # 'Andromeda XXXI',
        'LeG05': 'LeG 05',
        'LeG09': 'LeG 09',
        'LeG13': 'LeG 13',
        'LeG14': 'LeG 14',
        'LeG16': 'LeG 16',
        'LeG17': 'LeG 17',
        'LeG19': 'LeG 19',
        'LeG21': 'LeG 21',
        'LeG26': 'LeG 26',
        'Leo A': 'Leo A',
        'Leo I': 'Leo I',
        'Leo I 09': 'Leo dw A', #'NGC 3368:[CVD2018] DF6', # HyperLeda matches to SDSS J104653.19+124441.4
        'Leo II': 'Leo II',
        'Leo IV': 'Leo IV',
        #'Leo K': '', # not in NED
        #'Leo M': '',  # not in NED
        'Leo Minor I': 'Leo Minor I Dwarf',
        'Leo P': 'Leo P',
        'Leo T': 'Leo T',
        'Leo V': 'Leo V',
        #'Leo VI': '', not in NED
        'LGS 3': 'Pisces I',
        'LMC': 'Large Magellanic Cloud',
        'LSBC D565-09': 'LSBC D565-09',
        'LSBC D640-11': 'LSBC D640-11',
        'LV J0055-2310': 'LV J0055-2310', # 'GALEXASC J005501.01-231008.9',
        'LV J0616-5745': 'LV J0616-5745',
        'LV J1149+1715': 'LV J1149+1715',
        'LV J1157+5638': 'LV J1157+5638', # 'SDSS J115754.18+563816.6',
        'LV J1157+5638 sat': 'LV J1157+5638 sat', # 'SDSS J115753.02+563649.0',
        'LV J1158+1535': 'LV J1158+1535',
        'LV J1218+4655': 'LV J1218+4655',
        'LV J1228+4358': 'LV J1228+4358', # 'PGC1 0040904 NED002',
        'LV J1235-1104': 'LV J1235-1104', # 'WISEA J123539.40-110401.7',
        'LV J1241+3251': 'LV J1241+3251',
        'LV J1243+4127': 'LV J1243+4127', # 'SDSS J124354.70+412724.9',
        'LV J1313+1003': 'LV J1313+1003', # 'WISEA J131347.31+100310.9',
        'M 32': 'MESSIER 032',
        'M101 Dw9': 'Messier 101:[BSC2017] Dw 09',
        'M101 DwA': 'Messier 101:[BSC2017] Dw A',
        'M101-DF1': 'MESSIER 101:[MVA2014] DF_1',
        'M101-DF2': 'MESSIER 101:[MVA2014] DF_2',
        'M101-DF3': 'MESSIER 101:[MVA2014] DF_3',
        'M96-DF1': 'ELVES dw J1048p1158',
        'M96-DF2': 'ELVES dw J1047p1202',
        'M96-DF7': 'ELVES dw J1047p1248',
        'MADCASH-1': 'MADCASH J074238+652501-dw',
        #'MADCASH-2': '', not in NED?? MADCASH J121007+352635-dw maybe??
        'MAPS 1231+42': '[KMK2013] MAPS J1231+42',
        'MAPS 1249+44': '[KMK2013] MAPS 1249+44',
        'MCG -01-26-009': 'MCG -01-26-009',
        'MCG+06-27-017': 'MCG +06-27-017',
        'MCG+09-20-131': 'MCG +09-20-131',
        'MCG-04-31-038': 'MCG-04-31-038', # 'AM 1306-265',
        'N1291-DW10': 'NGC 1291:[OSG2024] DW10',
        'N1291-DW12': 'NGC 1291:[OSG2024] DW12',
        'N1291-DW13': 'NGC 1291:[OSG2024] DW13',
        'N1291-DW2': 'NGC 1291:[OSG2024] DW2',
        'N1291-DW3': 'NGC 1291:[OSG2024] DW3',
        'N1291-DW4': 'NGC 1291:[OSG2024] DW4',
        'N1291-DW5': 'NGC 1291:[OSG2024] DW5',
        'N1291-DW6': 'NGC 1291:[OSG2024] DW6',
        'N1291-DW8': 'NGC 1291:[OSG2024] DW8',
        'N1291-DW9': 'NGC 1291:[OSG2024] DW9',
        'NGC 1042': 'NGC 1042',
        'NGC 1313': 'NGC 1313',
        'NGC 147': 'NGC 147',
        'NGC 1560': 'NGC 1560',
        'NGC 1569': 'NGC 1569',
        'NGC 1592': 'NGC 1592',
        'NGC 1705': 'NGC 1705',
        'NGC 1792': 'NGC 1792',
        'NGC 1800': 'NGC 1800',
        'NGC 1827': 'NGC 1827',
        'NGC 185': 'NGC 185',
        'NGC 205': 'NGC 205',
        'NGC 2188': 'NGC 2188',
        'NGC 2366': 'NGC 2366',
        'NGC 24': 'NGC 24',
        'NGC 247': 'NGC 247',
        'NGC 2683': 'NGC 2683',
        'NGC 2683-dw1': '[CGB2022] dw 0853+3318',
        'NGC 2903-HI-1': 'NGC 2903-HI-1',
        'NGC 2915': 'NGC 2915',
        'NGC 300': 'NGC 300',
        'NGC 3077': 'NGC 3077',
        'NGC 3109': 'NGC 3109',
        'NGC 3351': 'NGC 3351',
        'NGC 3377': 'NGC 3377',
        'NGC 3384': 'NGC 3384',
        'NGC 3412': 'NGC 3412',
        'NGC 3521-dwTBG': 'NGC 3521:[KRZ2020] dwTBG',
        'NGC 3521-sat': '[CGB2022] dw 1105+0006', # 'NGC 3521sat' ??
        'NGC 3593': 'NGC 3593',
        'NGC 3623': 'NGC 3623',
        'NGC 3628': 'NGC 3628',
        'NGC 3738': 'NGC 3738',
        'NGC 3741': 'NGC 3741',
        'NGC 404': 'NGC 404',
        'NGC 4068': 'NGC 4068',
        'NGC 4151': 'NGC 4151',
        'NGC 4163': 'NGC 4163',
        'NGC 4190': 'NGC 4190',
        'NGC 4214': 'NGC 4214',
        'NGC 4236': 'NGC 4236',
        'NGC 4242': 'NGC 4242',
        'NGC 4244': 'NGC 4244',
        'NGC 4248': 'NGC 4248',
        'NGC 4395': 'NGC 4395',
        'NGC 4424': 'NGC 4424',
        'NGC 4449': 'NGC 4449',
        'NGC 45': 'NGC 45',
        'NGC 4517': 'NGC 4517',
        'NGC 4562': 'NGC 4562',
        'NGC 4592': 'NGC 4592',
        'NGC 4594-DGSAT-2': 'ELVES dw J1239m1120',
        'NGC 4594-DGSAT-3': 'ELVES dw J1239m1113',
        'NGC 4594-DW1': 'ELVES dw J1239m1143',
        'NGC 4627': 'NGC 4627',
        'NGC 4631-dw1': 'NGC 4631:[KKM2018a] dw1',
        'NGC 4631-dw2': 'NGC 4631:[KKM2018a] dw2',
        'NGC 4631-dw3': 'NGC 4631:[KKM2018a] dw3',
        'NGC 5011C': 'NGC 5011C',
        'NGC 5023': 'NGC 5023',
        'NGC 5055-dwTBG1': 'ELVES dw J1312p4158',
        'NGC 5068': 'NGC 5068',
        'NGC 5102': 'NGC 5102',
        'NGC 5194': 'MESSIER 051a',
        'NGC 5195': 'MESSIER 051b',
        'NGC 5204': 'NGC 5204',
        'NGC 5206': 'NGC 5206',
        'NGC 5237': 'NGC 5237',
        'NGC 5238': 'NGC 5238',
        'NGC 5253': 'NGC 5253',
        'NGC 5264': 'NGC 5264',
        'NGC 5408': 'NGC 5408',
        'NGC 5474': 'NGC 5474',
        'NGC 5477': 'NGC 5477',
        'NGC 55': 'NGC 0055',
        'NGC 5585': 'NGC 5585',
        'NGC 59': 'NGC 0059',
        'NGC 625': 'NGC 0625',
        'NGC 628-dwA': '[CGB2022] dw 0137+1537',
        'NGC 628-dwB': 'dw0137+1537',
        'NGC 6503': 'NGC 6503',
        'NGC 6744 dwTBGa': 'NGC 6744:[KRZ2020] dwTBGa',
        'NGC 6789': 'NGC 6789',
        'NGC 6822': 'NGC 6822',
        'NGC 7793': 'NGC 7793',
        'NGC 784': 'NGC 784',
        'NGC 891': 'NGC 0891',
        'NGC1311': 'NGC 1311',
        'NGC2976': 'NGC 2976',
        'NGC4765': 'NGC 4765',
        'Pavo': 'Pavo Dwarf',
        'Pegasus dIrr': 'Pegasus Dwarf',
        'Pegasus III': 'Pegasus III Dwarf',
        'Pegasus IV': 'Pegasus IV Dwarf',
        'Pegasus V': 'Pegasus V Dwarf', # ='Andromeda XXXIV'
        #'Pegasus VII': '', not in NED
        'Pegasus W': 'Pegasus W',
        'Perseus I': 'Perseus I', # 'Andromeda XXXIII'
        'PGC 100170': 'PGC 100170',
        'PGC 1059300': 'PGC 1059300',
        'PGC 138836': 'PGC 138836',
        'PGC 166192': 'PGC 166192',
        'PGC 166193': 'PGC 166193',
        'PGC 170257': 'PGC 170257', # '2MASX J13292099-2110452',
        'PGC 20125': 'PGC 20125',
        'PGC 2601822': 'PGC 2601822',
        'PGC 3272767': 'PGC 3272767',
        'PGC 34671': 'PGC 34671', # 'MCG +09-19-043',
        'PGC 39646': 'PGC 39646',
        'PGC 42730': 'PGC 42730',
        'PGC 4560429': 'PGC 4560429',
        'PGC 4561602': 'PGC 4561602',
        'PGC 51659': 'PGC 51659', # 'UKS 1424-460',
        'PGC 704814': 'PGC 704814',
        'PGC 725719': 'PGC 725719',
        'PGC1242': 'PGC 1242',
        'Phoenix': 'Phoenix Dwarf',
        'Phoenix II': 'Phoenix II',
        'Pictor I': 'Pictor I',
        'Pictor II': 'Pictor II',
        'Pisces A': 'Pisces A',
        'Pisces B': 'Pisces B',
        'Pisces II': 'Pisces II',
        'Pisces VII': 'Pisces VII',
        'Reticulum II': 'Reticulum II',
        'Reticulum III': 'Reticulum III',
        'Sagittarius': 'Sagittarius Dwarf Spheroidal',
        'Sagittarius dIrr': 'Sagittarius Dwarf Irregular',
        'SBS 1224+533': 'SBS 1224+533',
        'Scl-MM-Dw1': 'Scl-MM-Dw1',
        'Scl-MM-Dw2': 'Scl-MM-Dw2',
        'Scl-MM-Dw3': 'Scl-MM-Dw3',
        'Scl-MM-Dw4': 'Scl-MM-Dw4',
        'Scl-MM-Dw5': 'Scl-MM-Dw5',
        'Sculptor': 'Sculptor Dwarf Elliptical',
        'Sculptor-dE1': '[OMS2023] 6000343969257', # GALEXASC J002351.38-244215.7
        'Segue 1': 'Segue 1',
        'Segue 2': 'Segue 2',
        'Sextans': 'Sextans dSph',
        'Sextans A': 'Sextans A',
        'Sextans B': 'Sextans B',
        'Sextans II': 'Sextans II',
        'SMC': 'Small Magellanic Cloud',
        'SUCD1': 'SUCD 1',
        'Triangulum II': 'Triangulum II', # 'Laevens 2',
        'Tucana': 'Tucana Dwarf',
        'Tucana B': 'Tucana B', # 'SMDG J2247005-582429',
        'Tucana II': 'Tucana II Dwarf',
        'Tucana III': 'Tucana III Dwarf',
        'Tucana IV': 'Tucana IV Dwarf',
        'Tucana V': 'Tucana V Dwarf',
        'UGC 1104': 'UGC 1104',
        'UGC 11411': 'UGC 11411',
        'UGC 1171': 'UGC 1171',
        'UGC 1176': 'UGC 1176',
        'UGC 1281': 'UGC 1281',
        'UGC 12894': 'UGC 12894',
        'UGC 1703': 'UGC 1703',
        'UGC 1807': 'UGC 1807',
        'UGC 2157': 'UGC 2157',
        'UGC 2165': 'UGC 2165',
        'UGC 2684': 'UGC 2684',
        'UGC 2716': 'UGC 2716',
        'UGC 2773': 'UGC 2773',
        'UGC 288': 'UGC 288',
        'UGC 2905': 'UGC 2905',
        'UGC 3600': 'UGC 3600',
        'UGC 3698': 'UGC 3698',
        'UGC 4483': 'UGC 4483',
        'UGC 4879': 'UGC 4879',
        'UGC 5086': 'UGC 5086',
        'UGC 5497': 'UGC 5497',
        'UGC 5812': 'UGC 5812',
        'UGC 5944': 'UGC 5944',
        'UGC 64': 'UGC 64',
        'UGC 6451': 'UGC 6451',
        'UGC 6456': 'UGC 6456',
        'UGC 6541': 'UGC 6541',
        'UGC 6757': 'UGC 6757',
        'UGC 685': 'UGC 685',
        'UGC 7242': 'UGC 7242',
        'UGC 7298': 'UGC 07298',
        'UGC 7356': 'UGC 7356',
        'UGC 7490': 'UGC 7490',
        'UGC 7596': 'UGC 7596',
        'UGC 7605': 'UGC 7605',
        'UGC 7636': 'UGC 7636',
        'UGC 7929': 'UGC 7929',
        'UGC 8215': 'UGC 8215',
        'UGC 8245': 'UGC 8245',
        'UGC 8313': 'UGC 8313',
        'UGC 8508': 'UGC 8508',
        'UGC 8638': 'UGC 8638',
        'UGC 8833': 'UGC 8833',
        'UGC 8882': 'UGC 8882',
        'UGC 9128': 'UGC 9128',
        'UGC 9405': 'UGC 9405',
        'UGCA 105': 'UGCA 105',
        'UGCA 281': 'UGCA 281',
        'UGCA 287': 'UGCA 287',
        'UGCA 292': 'UGCA 292',
        'UGCA 319': 'UGCA 319',
        'UGCA 337': 'UGCA 337',
        'UGCA 365': 'UGCA 365',
        'UGCA 442': 'UGCA 442',
        'UGCA 86': 'UGCA 86',
        'UGCA 92': 'UGCA 92',
        'UKS 2323-326': 'UKS 2323-326',
        'Ursa Major I': 'UMa Dwarf',
        'Ursa Major II': 'Ursa Major II Dwarf',
        'Ursa Minor': 'UMi Dwarf',
        'Virgo I': 'Virgo I',
        'Virgo II': 'Virgo II Dwarf',
        'Willman 1': 'Willman 1',
        'WLM': 'WLM',
        }

    #lvd = read_lvd()
    #with open('junk.txt', 'w') as F:
    #    for key in ned.keys():
    #        F.write(f"        '{key}': {lvd[key == lvd["OBJNAME"]]["PGC"][0]},'\n")
    #return

    new = old.copy()

    maxlen = 0
    for obj in ned.keys():
        maxlen = np.max((maxlen, len(ned[obj])))
    new = new.astype(f'<U{maxlen}')

    #new['OBJNAME2'] = np.zeros(len(old), f'<U{maxlen}')
    #new['OBJNAME'] = new['OBJNAME2']
    #new.remove_column('OBJNAME2')

    I = np.where(np.isin(old, list(ned.keys())))[0]
    if len(I) > 0:
        for ii in I:
            #print(f'Replacing {new[ii]} --> {ned[old[ii]]}')
            new[ii] = ned[old[ii]]

    return new


def version_lvd():
    #ver = 'dwarf-all-b685634'
    #ver = 'v1.0.2'
    ver = 'v1.0.5'
    return ver


def read_lvd(rank=0, rows=None, overwrite=False):
    """Read the Local Volume Database (LVD) dwarf-galaxy catalog.

    """
    version = version_lvd()

    # combine the dwarf-all and dwarf-local-field-distant files
    lvdfile = os.path.join(sga_dir(), 'parent', 'external', f'LVD_{version}.fits')
    if not os.path.isfile(lvdfile) or overwrite:
        allfile = os.path.join(sga_dir(), 'parent', 'external', f'LVD_dwarf_all_{version}.csv')
        lvd = Table.read(allfile)

        if version == 'v1.0.4':
            lvd['ra'][lvd['name'] == 'MADCASH-1'] = 115.6641667

        if version == 'v1.0.5':
            lvd['ra'][lvd['name'] == 'AGC 198606'] = 142.519635
            lvd['ra'][lvd['name'] == 'NGC 1042'] = 40.0999125
            lvd['ra'][lvd['name'] == 'NGC 4151'] = 182.6360025
            lvd['ra'][lvd['name'] == 'NGC 4424'] = 186.79875
            lvd['dec'][lvd['name'] == 'NGC 4424'] = 9.4205
            lvd['ra'][lvd['name'] == 'PGC 100170'] = 44.2158075
            lvd['ra'][lvd['name'] == 'PGC 166192'] = 307.6358955
            lvd['ra'][lvd['name'] == 'PGC 166193'] = 307.8832995

            I = lvd['name'] == 'KK 166'
            lvd['rhalf'][I] = 11.97
            lvd['ellipticity'][I] = 0.12
            lvd['position_angle'][I] = 67.09
            lvd['ref_structure'][I] = 'Zaritsky2023ApJS..267...27Z'

        # drop unconfirmed systems
        print(f'Dropping {np.sum(lvd["confirmed_real"]==0):,d}/{len(lvd):,d} unconfirmed dwarfs.')
        lvd = lvd[lvd['confirmed_real'] == 1]
        lvd.remove_columns(['key', 'confirmed_real'])

        lvd.write(lvdfile, overwrite=True)


    F = fitsio.FITS(lvdfile)
    row = np.arange(F[1].get_nrows())
    if rows is not None:
        row = row[rows]

    lvd = Table(F[1].read(rows=rows))
    lvd['ROW'] = row
    print(f'Read {len(lvd):,d} objects from {lvdfile}')
    #print(f'Rank {rank:03d}: Read {len(lvd):,d} objects from {lvdfile}')

    [lvd.rename_column(col, col.upper()) for col in lvd.colnames]

    lvd.rename_column('NAME', 'OBJNAME')
    lvd = lvd[np.argsort(lvd['OBJNAME'])]

    # add PGC numbers
    # 0 = not in HyperLeda; -1 not checked yet
    lvd['PGC'] = np.zeros(len(lvd), int) - 1

    # http://atlas.obs-hp.fr/hyperleda/fG.cgi?c=o&n=a000&s=hc1719669736 - north
    # http://atlas.obs-hp.fr/hyperleda/fG.cgi?c=o&n=a000&s=hc1719674256 - south

    pgc = {
        '[KKH2011]S11': -1,
        '[TT2009] 25': 5061799,
        '[TT2009] 30': 5072545,
        'A0952+69': 0,
        'AGC 112521': 5057055,
        'AGC 238890': 1726175,
        'AGC 239141': 5808786,
        'AGC 749241': 5059213,
        'AGC749235': 5059199,
        'AM 1320-230': 3097728, # = PGC3097728
        'Andromeda I': 2666,
        'Andromeda II': 4601,
        'Andromeda III': 2121,
        'Andromeda IV': 2544,
        'Andromeda IX': 4689222,
        'Andromeda V': 3097824,
        'Andromeda VI': 2807158,
        'Andromeda VII': 2807155,
        'Andromeda X': 5056921,
        'Andromeda XI': 5056923,
        'Andromeda XII': 5056924,
        'Andromeda XIII': 5056925,
        'Andromeda XIV': 5056922,
        'Andromeda XIX': 5056919,
        'Andromeda XV': 5056926,
        'Andromeda XVI': 5056927,
        'Andromeda XVII': 4608690,
        'Andromeda XVIII': 5056918,
        'Andromeda XX': 5056920,
        'Andromeda XXI': 5057231,
        'Andromeda XXII': 5057232,
        'Andromeda XXIII': 5057226,
        'Andromeda XXIV': 5057227,
        'Andromeda XXIX': 5060430,
        'Andromeda XXV': 5057228,
        'Andromeda XXVI': 5057229,
        'Andromeda XXVII': 5057230,
        'Andromeda XXVIII': 5060429,
        'Andromeda XXXV': 0,
        'Antlia': 29194,
        'Antlia B': 5098252,
        'Antlia II': 6775392,
        'Aquarius': 65367,
        'Aquarius II': 5953206,
        'Aquarius III': 0,
        'Bedin 1': 0,
        'BK3N': 28529,
        'BK5N': 29231,
        'BK6N': 31286,
        'Bootes I': 4713553,
        'Bootes II': 4713552,
        'Bootes III': 4713562,
        'Bootes IV': 0,
        'Bootes V': 0,
        'BTS 116': 1839154,
        'BTS151': 2832120,
        'Camelopardalis A': 166082,
        'Camelopardalis B': 166084,
        'Canes Venatici I': 4689223,
        'Canes Venatici II': 4713558,
        'Carina': 19441,
        'Carina II': 0,
        'Carina III': 0,
        'Cassiopea dIrr 1': 100169,
        'Cassiopeia II': 5065056,
        'Cassiopeia III': 5065678,
        'CenA-MM-Dw1': 5072213,
        'CenA-MM-Dw2': 5072214,
        'CenA-MM-Dw3': 5509262,
        'CenA-MM-Dw4': 5509263,
        'CenA-MM-Dw5': 5509264,
        'CenA-MM-Dw6': 5509265,
        'CenA-MM-Dw7': 5509266,
        'Centaurus I': 0,
        'Centaurus N': 4689187,
        'Cepheus1': 3097690,
        'Cetus': 3097691,
        'Cetus II': 6740632,
        'CGCG 189-050': 46257,
        'CGCG 217-018': 45889,
        'Clump I': 5057029,
        'Clump III': 5057030,
        'Columba I': 6740626,
        'Coma Berenices': 0,
        'Corvus A': 0,
        'Crater II': 5742923,
        'd0926+70': 5056943,
        'd0934+70': 5056931,
        'd0939+71': 5056932,
        'd0944+69': 5056933,
        'd0944+71': 5056944,
        'd0955+70': 5056934,
        'd0958+66': 28826,
        'd0959+68': 5056936,
        'd1006+67': 5056937,
        'd1014+68': 5056938,
        'd1015+69': 5056947,
        'd1028+70': 5056941,
        'd1041+70': 5056942,
        'DDO 113': 39145,
        'DDO 125': 40904,
        'DDO 126': 40791,
        'DDO 127': 41020,
        'DDO 133': 41636,
        'DDO 147': 43129,
        'DDO 153': 43851,
        'DDO 154': 43869,
        'DDO 161': 45084,
        'DDO 165': 45372,
        'DDO 167': 45939,
        'DDO 168': 46039,
        'DDO 169': 46127,
        'DDO 169NW': 5057032,
        'DDO 181': 48332,
        'DDO 183': 49158,
        'DDO 190': 51472,
        'DDO 226': 2578,
        'DDO 44': 21302,
        'DDO 53': 24050,
        'DDO 6': 2902,
        'DDO 78': 30664,
        'DDO 82': 30997,
        'DDO 99': 37050,
        'Donatiello III': 0,
        'Donatiello IV': 0,
        'Draco': 60095,
        'Draco II': 0,
        'dw0036m2828': 0,
        'dw0132+1422': -1,
        'dw0133p1543': -1,
        'dw0134+1438': -1,
        'dw0134p1544': -1,
        'dw0136p1628': -1,
        'dw0137p1607': -1,
        'dw0138+1458': -1,
        'dw0139p1433': -1,
        'dw0140p1556': -1,
        'dw0235p3850': -1,
        'dw0237p3836': -1,
        'dw0237p3855': -1,
        'dw0239p3902': -1,
        'dw0239p3903': -1,
        'dw0239p3926': -1,
        'dw0240p3854': -1,
        'dw0240p3922': -1,
        'dw0241p3904': -1,
        'dw0242p3838': -1,
        'dw0506m3739': -1,
        'dw0507m3629': -1,
        'dw0507m3739': -1,
        'dw0507m3744': -1,
        'dw0507m3800': -1,
        'dw0508m3617': -1,
        'dw0508m3808': -1,
        'dw0929+2213': -1,
        'dw0932+1952': -1,
        'dw0936+2135': -1,
        'dw1000-0741': -1,
        'dw1000m0821': -1,
        'dw1000m0831': -1,
        'dw1002m0642': -1,
        'dw1002m0818': -1,
        'dw1004m0657': -1,
        'dw1004m0737': -1,
        'dw1006m0730': -1,
        'dw1006m0730-n2': -1,
        'dw1006m0732': -1,
        'dw1007m0715': -1,
        'dw1007m0830': -1,
        'dw1007m0835': -1,
        'dw1042p1359': -1,
        'dw1043p1410': -1,
        'dw1043p1415': -1,
        'dw1044p1351': -1,
        'dw1044p1351b': -1,
        'dw1044p1356': -1,
        'dw1044p1359': -1,
        'dw1046+1244': 0,
        'dw1047p1153': -1,
        'dw1047p1258': -1,
        'dw1048p1154': -1,
        'dw1048p1259': -1,
        'dw1048p1303': -1,
        'dw1048p1407': -1,
        'dw1049p1233': -1,
        'dw1049p1247': -1,
        'dw1050p1213': -1,
        'dw1050p1236': -1,
        'dw1051p1406': -1,
        'dw1052p1102': -1,
        'dw1104+0004': -1,
        'dw1104+0005': -1,
        'dw1106-0052': -1,
        'dw1109-0016': -1,
        'dw1110+0037': -1,
        'dw1111+0049': -1,
        'dw1114p1238': -1,
        'dw1118p1233': -1,
        'dw1118p1348': -1,
        'dw1119p1157': -1,
        'dw1119p1404': -1,
        'dw1119p1417': -1,
        'dw1119p1419': -1,
        'dw1120p1332': -1,
        'dw1120p1337': -1,
        'dw1121p1326': -1,
        'dw1122p1258': -1,
        'dw1123p1342': -1,
        'dw1124p1240': -1,
        'dw1220+4649': -1,
        'dw1227p0136': -1,
        'dw1231p0140': -1,
        'dw1232p0015': -1,
        'dw1234p2531': -1,
        'dw1236+3336': -1,
        'dw1236m0025': -1,
        'dw1237p2602': -1,
        'dw1238m0035': -1,
        'dw1238m0105': -1,
        'dw1238p0028': -1,
        'dw1239+3230': -1,
        'dw1239+3251': -1,
        'dw1239-1159': -1,
        'dw1240-1118': 42428,
        'Dw1245+6158': -1,
        'dw1300+1843': 0,
        'dw1310+4153': -1,
        'dw1315+4123': -1,
        'dw1321+4226': -1,
        'dw1322-39': 5912201, # = [MJB2016]DW1322-39
        'dw1323-40a': 0,
        'dw1323-40b': 0,
        'dw1328+4703': 0,
        'dw1329-45': 5912206, # = [MJB2016]DW1329-45
        'dw1330+4731': 0,
        'dw1335-29': 5477876,
        'dw1336-44': 5912210, # = [MJB2016]DW1336-44
        'dw1340-30': 0,
        'dw1341-29': 0,
        'dw1341-43': 5912213, # = [MJB2016]DW1341-43
        'dw1342-43': 5912214, # = [MJB2016]DW1342-43
        'dw1343+58': -1,
        'Eridanus II': 5074553,
        'Eridanus IV': 0,
        'ESO 006-001': 23344,
        'ESO 059-001': 21199,
        'ESO 115-021': 9962,
        'ESO 121-020': 18731,
        'ESO 137-018': 57888,
        'ESO 154-023': 11139,
        'ESO 199-007': 11211,
        'ESO 215-009': 490287,
        'ESO 222-010': 52125,
        'ESO 245-005': 6430,
        'ESO 269-037': 45104,
        'ESO 269-058': 45717,
        'ESO 269-066': 45916,
        'ESO 272-025': 52591,
        'ESO 273-014': 53500,
        'ESO 274-001': 54392,
        'ESO 290-028': 70089,
        'ESO 294-G010': 1641,
        'ESO 300-016': 11842,
        'ESO 301-007': 12586,
        'ESO 301-010': 12676,
        'ESO 320-014': 36014,
        'ESO 321-014': 39032,
        'ESO 324-024': 47171,
        'ESO 325-011': 48738,
        'ESO 346-007': 69923,
        'ESO 349-031': 621,
        'ESO 379-007': 37369,
        'ESO 379-024': 38252,
        'ESO 381-018': 42936,
        'ESO 381-020': 43048,
        'ESO 383-087': 49050,
        'ESO 384-016': 49615,
        'ESO 410-G005': 1038,
        'ESO 443-009': 43978,
        'ESO 444-084': 48111,
        'ESO 472-015': 513,
        'ESO 540-032': 2933,
        'ESO104-044': 62869,
        'ESO219-010': 44110,
        'F8D1': 3097827,
        'Fluffy': 0,
        'FM1': 3097828,
        'Fornax': 10074,
        'FS04': 31877,
        'GALFA Dw3': 5072714,
        'GALFA Dw4': 5072715,
        'GARLAND': 29167,
        'GHOSTS I': 5067066,
        'GR 8': 44491,
        'Grapes': 0,
        'Grus I': 5074558,
        'Grus II': 6740630,
        'Hercules': 4713560,
        'HIDEEP J1337-3320': 677373,
        'HIPASS J1131-31': 5060432,
        'HIPASS J1133-32': 683190,
        'HIPASS J1247-77': 3994669,
        'HIPASS J1337-39': 592761,
        'HIPASS J1348-37': 4614882,
        'HIPASS J1351-47': 3097113,
        'HIZSS-021': 0,
        'Holm IV': 49448,
        'Holmberg I': 27605,
        'Holmberg II': 23324,
        'Holmberg IX': 28757,
        'Horologium I': 5074554,
        'Horologium II': 5092747,
        'HS 117': 4689216,
        'HSC-10': 6726342,
        'HSC-9': 6726341,
        'Hydra II': 5074546,
        'Hydrus I': 0,
        'IC 10': 1305,
        'IC 1613': 3844,
        'IC 2574': 30819,
        'IC 2684': 34438,
        'IC 2782': 34934,
        'IC 2787': 34969,
        'IC 2791': 3542933,
        'IC 3104': 39573,
        'IC 3687': 42656,
        'IC 3840': 0,
        'IC 4182': 45314,
        'IC 4247': 47073,
        'IC 4316': 48368,
        'IC 4662': 60849,
        'IC 5052': 65603,
        'IC 5152': 67908,
        'IC1959': 13163,
        'IC239': 9899,
        'IC3571': 0,
        'IKN': 4689195,
        'JKB129': 4668290,
        'JKB142': 5808737,
        'JKB83': 6657020,
        'KDG 171': 42294,
        'KDG 2': 2881,
        'KDG 215': 44055,
        'KDG 52': 23521,
        'KDG 56': 26972,
        'KDG 61': 28731,
        'KDG 63': 29257,
        'KDG 64': 29388,
        'KDG 73': 32667,
        'KDG 74': 33305,
        'KDG010': 6354,
        'KK 109': 166115,
        'KK 132': 166127,
        'KK 135': 166130,
        'KK 144': 166137,
        'KK 153': 41920,
        'KK 16': 166064,
        'KK 160': 166142,
        'KK 165': 166145,
        'KK 166': 166146,
        'KK 17': 166065,
        'KK 176': 44681,
        'KK 177': 87149,
        'KK 180': 86645,
        'KK 182': 166152,
        'KK 189': 166158,
        'KK 191': 166159,
        'KK 193': 166161,
        'KK 195': 166163,
        'KK 196': 46663,
        'KK 197': 46680,
        'KK 200': 46885,
        'KK 203': 166167,
        'KK 208': 166170,
        'KK 211': 48515,
        'KK 213': 166172,
        'KK 217': 166175,
        'KK 218': 166176,
        'KK 221': 166179,
        'KK 242': 4689184,
        'KK 258': 69468,
        'KK 27': 166073,
        'KK 35': 166077,
        'KK 69': 166095,
        'KK 70': 166096,
        'KK 77': 166101,
        'KK 93': 83333,
        'KK94': 83339,
        'KK96': 166107,
        'KKH 11': 168300,
        'KKH 12': 2807107,
        'KKH 18': 2807110,
        'KKH 22': 2807114,
        'KKH 30': 95591,
        'KKH 34': 95594,
        'KKH 37': 95597,
        'KKH 5': 2807102,
        'KKH 57': 2807133,
        'KKH 6': 2807103,
        'KKH 78': 2807147,
        'KKH 86': 2807150,
        'KKH 98': 2807157,
        'KKR 25': 2801026,
        'KKR 3': 166185,
        'KKs 3': 9140,
        'KKs 51': 2815819,
        'KKs 53': 2815820,
        'KKs 54': 2815821,
        'KKs 55': 2815822,
        'KKs 57': 2815823,
        'KKs 58': 2815824,
        'KKs 59': 48937,
        'KKSG 17': 29033,
        'KKSG 18': 29300,
        'KKSG 20': 135770,
        'KKSG 22': 3097701,
        'KKSG 29': 42120,
        'KKSG 31': 3097709,
        'KKSG 32': 3097710,
        'KKSG 33': 3097711,
        'KKSG 37': 3097714,
        'KV19-212': 0,
        'KV19-271': 0,
        'KV19-329': 0,
        'KV19-442': 0,
        'Lacerta I': 5065677,
        'LeG05': 83286,
        'LeG09': 83321,
        'LeG13': 83326,
        'LeG14': 83329,
        'LeG16': 4689213,
        'LeG17': 83336,
        'LeG19': 83338,
        'LeG21': 4689200,
        'LeG26': 4689214,
        'Leo A': 28868,
        'Leo I': 29488,
        'Leo I 09': 4689210,
        'Leo II': 34176,
        'Leo IV': 4713561,
        'Leo K': 0,
        'Leo M': 0,
        'Leo Minor I': 0,
        'Leo P': 5065058,
        'Leo T': 4713564,
        'Leo V': 4713563,
        'Leo VI': 0,
        'LGS 3': 3792,
        'LMC': 17223,
        'LSBC D565-09': -1,
        'LSBC D640-11': 83364,
        'LV J0055-2310': 6740710,
        'LV J0616-5745': 385975,
        'LV J1149+1715': 1528400,
        'LV J1157+5638': 2543081,
        'LV J1157+5638 sat': 6740587,
        'LV J1158+1535': 1488625,
        'LV J1218+4655': 4320422,
        'LV J1228+4358': 5057024,
        'LV J1235-1104': 970397,
        'LV J1241+3251': 100707,
        'LV J1243+4127': 5056993,
        'LV J1313+1003': 4573336,
        'M 32': 2555,
        'M101 Dw9': 6740596,
        'M101 DwA': 5067392,
        'M101-DF1': 5067385,
        'M101-DF2': 5067386,
        'M101-DF3': 5067387,
        'M96-DF1': -1,
        'M96-DF2': -1,
        'M96-DF7': -1,
        'MADCASH-1': 0,
        'MADCASH-2': 0,
        'MAPS 1231+42': 0,
        'MAPS 1249+44': 0,
        'MCG -01-26-009': 29038,
        'MCG+06-27-017': 38685,
        'MCG+09-20-131': 39228,
        'MCG-04-31-038': 45628,
        'N1291-DW10': -1,
        'N1291-DW12': -1,
        'N1291-DW13': -1,
        'N1291-DW2': -1,
        'N1291-DW3': -1,
        'N1291-DW4': -1,
        'N1291-DW5': -1,
        'N1291-DW6': -1,
        'N1291-DW8': -1,
        'N1291-DW9': -1,
        'NGC 1042': 10122,
        'NGC 1313': 12286,
        'NGC 147': 2004,
        'NGC 1560': 15488,
        'NGC 1569': 15345,
        'NGC 1592': 15292,
        'NGC 1705': 16282,
        'NGC 1792': 16709,
        'NGC 1800': 16744,
        'NGC 1827': 16849,
        'NGC 185': 2329,
        'NGC 205': 2429,
        'NGC 2188': 18536,
        'NGC 2366': 21102,
        'NGC 24': 701,
        'NGC 247': 2758,
        'NGC 2683': 24930,
        'NGC 2683-dw1': -1,
        'NGC 2903-HI-1': -1,
        'NGC 2915': 26761,
        'NGC 300': 3238,
        'NGC 3077': 29146,
        'NGC 3109': 29128,
        'NGC 3351': 32007,
        'NGC 3377': 32249,
        'NGC 3384': 32292,
        'NGC 3412': 32508,
        'NGC 3521-dwTBG': -1,
        'NGC 3521-sat': -1,
        'NGC 3593': 34257,
        'NGC 3623': 34612,
        'NGC 3628': 34697,
        'NGC 3738': 35856,
        'NGC 3741': 35878,
        'NGC 404': 4126,
        'NGC 4068': 38148,
        'NGC 4151': 38739,
        'NGC 4163': 38881,
        'NGC 4190': 39023,
        'NGC 4214': 39225,
        'NGC 4236': 39346,
        'NGC 4242': 39423,
        'NGC 4244': 39422,
        'NGC 4248': 39461,
        'NGC 4395': 40596,
        'NGC 4424': 40809,
        'NGC 4449': 40973,
        'NGC 45': 930,
        'NGC 4517': 41618,
        'NGC 4562': 41955,
        'NGC 4592': 42336,
        'NGC 4594-DGSAT-2': 5473059,
        'NGC 4594-DGSAT-3': 5473061,
        'NGC 4594-DW1': 0,
        'NGC 4627': 42620,
        'NGC 4631-dw1': -1,
        'NGC 4631-dw2': -1,
        'NGC 4631-dw3': -1,
        'NGC 5011C': 45917,
        'NGC 5023': 45849,
        'NGC 5055-dwTBG1': -1,
        'NGC 5068': 46400,
        'NGC 5102': 46674,
        'NGC 5194': 47404,
        'NGC 5195': 47413,
        'NGC 5204': 47368,
        'NGC 5206': 47762,
        'NGC 5237': 48139,
        'NGC 5238': 47853,
        'NGC 5253': 48334,
        'NGC 5264': 48467,
        'NGC 5408': 50073,
        'NGC 5474': 50216,
        'NGC 5477': 50262,
        'NGC 55': 1014,
        'NGC 5585': 51210,
        'NGC 59': 1034,
        'NGC 625': 5896,
        'NGC 628-dwA': -1,
        'NGC 628-dwB': -1,
        'NGC 6503': 60921,
        'NGC 6744 dwTBGa': -1,
        'NGC 6789': 63000,
        'NGC 6822': 63616,
        'NGC 7793': 73049,
        'NGC 784': 7671,
        'NGC 891': 9031,
        'NGC1311': 12460,
        'NGC2976': 28120,
        'NGC4765': 43775,
        'Pavo': 0,
        'Pegasus dIrr': 71538,
        'Pegasus III': 5074547,
        'Pegasus IV': 0,
        'Pegasus V': 0,
        'Pegasus VII': 0,
        'Pegasus W': 0,
        'Perseus I': 5067061,
        'PGC 100170': 100170,
        'PGC 1059300': 1059300,
        'PGC 138836': 138836,
        'PGC 166192': 166192,
        'PGC 166193': 166193,
        'PGC 170257': 170257,
        'PGC 20125': 20125,
        'PGC 2601822': 2601822,
        'PGC 3272767': 3272767,
        'PGC 34671': 34671,
        'PGC 39646': 39646,
        'PGC 42730': 42730,
        'PGC 4560429': 4560429,
        'PGC 4561602': 4561602,
        'PGC 51659': 51659,
        'PGC 704814': 704814,
        'PGC 725719': 725719,
        'PGC1242': 1242,
        'Phoenix': 6830,
        'Phoenix II': 5074556,
        'Pictor I': 5074555, # =PICTORIS1
        'Pictor II': 6657033,
        'Pisces A': 5072710,
        'Pisces B': 5072711,
        'Pisces II': 5056949,
        'Pisces VII': 0,
        'Reticulum II': 5074552,
        'Reticulum III': 6740628,
        'Sagittarius': 4689212,
        'Sagittarius dIrr': 63287,
        'SBS 1224+533': 40750,
        'Scl-MM-Dw1': 5067869,
        'Scl-MM-Dw2': 5478807,
        'Scl-MM-Dw3': 0,
        'Scl-MM-Dw4': 0,
        'Scl-MM-Dw5': 0,
        'Sculptor': 3589,
        'Sculptor-dE1': 3097727,
        'Segue 1': 4713559,
        'Segue 2': 4713565,
        'Sextans': 88608, # = PGC088608
        'Sextans A': 29653,
        'Sextans B': 28913,
        'Sextans II': 0,
        'SMC': 3085,
        'SUCD1': 3793583,
        'Triangulum II': 5074545,
        'Tucana': 69519,
        'Tucana B': 0, # = SMDG J2247005-582429
        'Tucana II': 5074560,
        'Tucana III': 6657034,
        'Tucana IV': 6740629,
        'Tucana V': 6740631,
        'UGC 1104': 5761,
        'UGC 11411': 62814,
        'UGC 1171': 6150,
        'UGC 1176': 6174,
        'UGC 1281': 6699,
        'UGC 12894': 35,
        'UGC 1703': 8484,
        'UGC 1807': 8947,
        'UGC 2157': 10124,
        'UGC 2165': 10180,
        'UGC 2684': 12514,
        'UGC 2716': 12719,
        'UGC 2773': 13115,
        'UGC 288': 1777,
        'UGC 2905': 14149,
        'UGC 3600': 19871,
        'UGC 3698': 20264,
        'UGC 4483': 24213,
        'UGC 4879': 26142,
        'UGC 5086': 27115,
        'UGC 5497': 29735,
        'UGC 5812': 31801,
        'UGC 5944': 32471,
        'UGC 64': 591,
        'UGC 6451': 35264,
        'UGC 6456': 35286,
        'UGC 6541': 35684,
        'UGC 6757': 36758,
        'UGC 685': 3974,
        'UGC 7242': 39058,
        'UGC 7298': 39316,
        'UGC 7356': 39615,
        'UGC 7490': 40367,
        'UGC 7596': 41036,
        'UGC 7605': 41048,
        'UGC 7636': 41258,
        'UGC 7929': 42985,
        'UGC 8215': 45506,
        'UGC 8245': 45546,
        'UGC 8313': 45992,
        'UGC 8508': 47495,
        'UGC 8638': 48280,
        'UGC 8833': 49452,
        'UGC 8882': 49636,
        'UGC 9128': 50961,
        'UGC 9405': 52142,
        'UGCA 105': 16957,
        'UGCA 281': 40665,
        'UGCA 287': 41743,
        'UGCA 292': 42275,
        'UGCA 319': 44982,
        'UGCA 337': 45897,
        'UGCA 365': 48029,
        'UGCA 442': 72228,
        'UGCA 86': 14241,
        'UGCA 92': 15439,
        'UKS 2323-326': 71431,
        'Ursa Major I': 4713554,
        'Ursa Major II': 4713555,
        'Ursa Minor': 54074,
        'Virgo I': 6657032,
        'Virgo II': 0,
        'Willman 1': 4713556,
        'WLM': 143,
        }


    for key in pgc.keys():
        I = np.where(lvd['OBJNAME'] == key)[0]
        if len(I) == 0:
            #raise ValueError(f'Error in galaxy name {key}')
            continue
        lvd['PGC'][I] = pgc[key]

    lvd['GALAXY'] = lvd['OBJNAME']

    return lvd


def version_custom_external():
    ver = 'v1.0'
    return ver


def read_custom_external(rank=0, rows=None, overwrite=False):
    """Read the custom external catalog.

    """
    version = version_custom_external()

    customfile = os.path.join(sga_dir(), 'parent', 'external', f'custom-external_{version}.fits')
    if not os.path.isfile(customfile) or overwrite:
        from importlib import resources
        csvfile = str(resources.files('SGA').joinpath(f'data/SGA2025/custom-external_{version}.csv'))
        data = Table.read(csvfile, format='csv', comment='#')
        data['mag_band'] = data['mag_band'].astype('<U1')
        data['objname'].fill_value = ''
        data['mag_band'].fill_value = ''
        data['objname_ned'].fill_value = ''
        data = data.filled()
        for col in ['diam', 'ba', 'pa', 'mag']:
            data[col] = data[col].astype('f4')
        [data.rename_column(col, col.upper()) for col in data.colnames]
        data['PA'] = data['PA'].value % 180 # put in range [0-->180]

        data.write(customfile, overwrite=True)

    F = fitsio.FITS(customfile)
    row = np.arange(F[1].get_nrows())
    if rows is not None:
        row = row[rows]

    data = Table(F[1].read(rows=rows))
    data['ROW'] = row
    print(f'Read {len(data):,d} objects from {customfile}')

    data = data[np.argsort(data['OBJNAME'])]
    data['GALAXY'] = data['OBJNAME']

    return data


def version_nedlvs():
    return '20210922_v2'


def read_nedlvs(rank=0, rows=None):
    """Read the NED-LVS catalog.

    """
    nedlvsfile = os.path.join(sga_dir(), 'parent', 'external', f'NEDLVS_{version_nedlvs()}.fits')
    nedlvs = Table(fitsio.read(nedlvsfile, rows=rows))
    nedlvs['ROW'] = np.arange(len(nedlvs))
    print(f'Read {len(nedlvs):,d} objects from {nedlvsfile}')
    #print(f'Rank {rank:03d}: Read {len(nedlvs):,d} objects from {nedlvsfile}')

    [nedlvs.rename_column(col, col.upper()) for col in nedlvs.colnames]
    nedlvs['GALAXY'] = nedlvs['OBJNAME']

    return nedlvs


def read_sga2020(rank=0, rows=None):
    """Read the SGA-2020 catalog.

    """
    sga2020file = os.path.join(sga_dir(), 'parent', 'external', 'SGA-2020.fits')
    sga2020 = Table(fitsio.read(sga2020file, ext='ELLIPSE', rows=rows))
    sga2020['ROW'] = np.arange(len(sga2020))
    print(f'Read {len(sga2020):,d} objects from {sga2020file}')
    #print(f'Rank {rank:03d}: Read {len(sga2020):,d} objects from {sga2020file}')

    return sga2020


def read_wxsc(rank=0):
    """Read the WXSC catalog.

    """
    wxscfile = os.path.join(sga_dir(), 'parent', 'external', 'WXSC_Riso_W1mag_24Jun024.tbl') # 'WXSC_Riso_1arcmin_10Jun2024.tbl')
    wxsc = Table.read(wxscfile, format='ascii.ipac')
    wxsc['ROW'] = np.arange(len(wxsc))
    print(f'Rank {rank:03d}: Read {len(wxsc):,d} objects from {wxscfile}')

    # toss out duplicates
    radec = np.array([f'{ra}-{dec}' for ra, dec in zip(wxsc['ra'].astype(str), wxsc['dec'].astype(str))])
    u, c = np.unique(radec, return_counts=True)

    _, uindx = np.unique(radec, return_index=True)

    print(f'Rank {rank:03d}: Trimming to {len(uindx):,d}/{len(wxsc):,d} unique objects based on (ra,dec)')
    wxsc = wxsc[uindx]

    _, uindx = np.unique(wxsc['NED_name'], return_index=True)
    print(f'Rank {rank:03d}: WARNING: Trimming to {len(uindx):,d}/{len(wxsc):,d} unique objects based on NED name')
    wxsc = wxsc[uindx]

    #u, c = np.unique(wxsc['NED_name'], return_counts=True)
    #print(np.unique(c))
    #dup = vstack([wxsc[wxsc['NED_name'] == gal] for gal in u[c>1]])
    #return dup

    [wxsc.rename_column(col, col.upper()) for col in wxsc.colnames]
    wxsc['GALAXY'] = wxsc['WXSCNAME']

    return wxsc


#@contextmanager
#def stdouterr_redirected(to=None, comm=None, overwrite=False):
#    """
#    Redirect stdout and stderr to a file.
#
#    The general technique is based on:
#
#    http://stackoverflow.com/questions/5081657
#    http://eli.thegreenplace.net/2015/redirecting-all-kinds-of-stdout-in-python/
#
#    One difference here is that each process in the communicator
#    redirects to a different temporary file, and the upon exit
#    from the context the rank zero process concatenates these
#    in order to the file result.
#
#    Args:
#        to (str): The output file name.
#        comm (mpi4py.MPI.Comm): The optional MPI communicator.
#        overwrite (bool): if True overwrite file, otherwise backup to to.N first
#    """
#    nproc = 1
#    rank = 0
#    if comm is not None:
#        nproc = comm.size
#        rank = comm.rank
#
#    # The currently active POSIX file descriptors
#    fd_out = sys.stdout.fileno()
#    fd_err = sys.stderr.fileno()
#
#    # The DESI loggers.
#    #desi_loggers = desiutil.log._desiutil_log_root
#
#    def _redirect(out_to, err_to):
#
#        # Flush the C-level buffers
#        if c_stdout is not None:
#            libc.fflush(c_stdout)
#        if c_stderr is not None:
#            libc.fflush(c_stderr)
#
#        # This closes the python file handles, and marks the POSIX
#        # file descriptors for garbage collection- UNLESS those
#        # are the special file descriptors for stderr/stdout.
#        sys.stdout.close()
#        print('HI!!!!!!!!!!!')
#        sys.stderr.close()
#
#        # Close fd_out/fd_err if they are open, and copy the
#        # input file descriptors to these.
#        os.dup2(out_to, fd_out)
#        os.dup2(err_to, fd_err)
#
#        # Create a new sys.stdout / sys.stderr that points to the
#        # redirected POSIX file descriptors.  In Python 3, these
#        # are actually higher level IO objects.
#        if sys.version_info[0] < 3:
#            sys.stdout = os.fdopen(fd_out, "wb")
#            sys.stderr = os.fdopen(fd_err, "wb")
#        else:
#            # Python 3 case
#            sys.stdout = io.TextIOWrapper(os.fdopen(fd_out, 'wb'))
#            sys.stderr = io.TextIOWrapper(os.fdopen(fd_err, 'wb'))
#
#        # update DESI logging to use new stdout
#        for name, logger in desi_loggers.items():
#            hformat = None
#            while len(logger.handlers) > 0:
#                h = logger.handlers[0]
#                if hformat is None:
#                    hformat = h.formatter._fmt
#                logger.removeHandler(h)
#            # Add the current stdout.
#            ch = logging.StreamHandler(sys.stdout)
#            formatter = logging.Formatter(hformat, datefmt='%Y-%m-%dT%H:%M:%S')
#            ch.setFormatter(formatter)
#            logger.addHandler(ch)
#
#    # redirect both stdout and stderr to the same file
#
#    if to is None:
#        to = "/dev/null"
#
#    if rank == 0:
#        log = get_logger()
#        log.info("Begin log redirection to {} at {}".format(to, time.asctime()))
#        #print("Begin log redirection to {} at {}".format(to, time.asctime()))
#        if not overwrite:
#            backup_filename(to)
#
#    #- all ranks wait for logfile backup
#    if comm is not None:
#        comm.barrier()
#
#    # Save the original file descriptors so we can restore them later
#    saved_fd_out = os.dup(fd_out)
#    saved_fd_err = os.dup(fd_err)
#
#    try:
#        pto = to
#        if to != "/dev/null":
#            pto = "{}_{}".format(to, rank)
#
#        # open python file, which creates low-level POSIX file
#        # descriptor.
#        file = open(pto, "w")
#
#        # redirect stdout/stderr to this new file descriptor.
#        _redirect(out_to=file.fileno(), err_to=file.fileno())
#
#        yield # allow code to be run with the redirected output
#
#        # close python file handle, which will mark POSIX file
#        # descriptor for garbage collection.  That is fine since
#        # we are about to overwrite those in the finally clause.
#        file.close()
#    finally:
#        print('HI!!!!!!!!!!!')
#        import pdb ; pdb.set_trace()
#        # flush python handles for good measure
#        sys.stdout.flush()
#        sys.stderr.flush()
#
#        # restore old stdout and stderr
#        _redirect(out_to=saved_fd_out, err_to=saved_fd_err)
#
#        if nproc > 1:
#            comm.barrier()
#
#        # concatenate per-process files
#        if rank == 0 and to != "/dev/null":
#            with open(to, "w") as outfile:
#                for p in range(nproc):
#                    outfile.write("================ Start of Process {} ================\n".format(p))
#                    fname = "{}_{}".format(to, p)
#                    with open(fname) as infile:
#                        outfile.write(infile.read())
#                    outfile.write("================= End of Process {} =================\n\n".format(p))
#                    os.remove(fname)
#
#        if nproc > 1:
#            comm.barrier()
#
#        if rank == 0:
#            log = get_logger()
#            log.info("End log redirection to {} at {}".format(to, time.asctime()))
#            #print("End log redirection to {} at {}".format(to, time.asctime()))
#
#        # flush python handles for good measure
#        sys.stdout.flush()
#        sys.stderr.flush()
#
#    return


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


def missing_files(sample=None, bricks=None, detection_coadds=False, candidate_cutouts=False,
                  coadds=False, ellipse=False, htmlplots=False, htmlindex=False,
                  build_SGA=False, overwrite=False, verbose=False, htmldir='.',
                  size=1, mp=1):
    """Figure out which files are missing and still need to be processed.

    """
    from glob import glob
    import multiprocessing
    import astropy
    from SGA.util import weighted_partition

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
    if detection_coadds or candidate_cutouts:
        galaxy, galaxydir = get_galaxy_galaxydir(bricks=bricks)
    else:
        if htmlplots is False and htmlindex is False:
            if verbose:
                t0 = time.time()
                log.info('Getting galaxy names and directories...', end='')
            galaxy, galaxydir = get_galaxy_galaxydir(sample)
            if verbose:
                log.info(f'...took {time.time() - t0:.3f} sec')

    if detection_coadds:
        suffix = 'detection-coadds'
        filesuffix = '-detection-coadds.isdone'
    elif candidate_cutouts:
        suffix = 'candidate-cutouts'
        filesuffix = '-candidate-cutouts.isdone'
        dependson = '-detection-coadds.isdone'
    elif coadds:
        suffix = 'coadds'
        filesuffix = '-largegalaxy-coadds.isdone'
    elif ellipse:
        suffix = 'ellipse'
        filesuffix = '-largegalaxy-ellipse.isdone'
        dependson = '-largegalaxy-coadds.isdone'
    elif build_SGA:
        suffix = 'build-SGA'
        filesuffix = '-largegalaxy-SGA.isdone'
        dependson = '-largegalaxy-ellipse.isdone'
    elif htmlplots:
        suffix = 'html'
        filesuffix = '-largegalaxy-grz-montage.png'
        dependson = '-largegalaxy-image-grz.jpg'
        galaxy, dependsondir, galaxydir = get_galaxy_galaxydir(sample, htmldir=htmldir, html=True)
    elif htmlindex:
        suffix = 'htmlindex'
        filesuffix = '-largegalaxy-grz-montage.png'
        galaxy, _, galaxydir = get_galaxy_galaxydir(sample, htmldir=htmldir, html=True)
    else:
        raise ValueError('Need at least one keyword argument.')

    # Make overwrite=False for build_SGA and htmlindex because we're not making
    # the files here, we're just looking for them. The argument overwrite gets
    # used downstream.
    if htmlindex:
        overwrite = False

    missargs = []
    for igal, (gal, gdir) in enumerate(zip(np.atleast_1d(galaxy), np.atleast_1d(galaxydir))):
        checkfile = os.path.join(gdir, f'{gal}{filesuffix}')
        if dependson:
            if dependsondir:
                missargs.append([checkfile, os.path.join(np.atleast_1d(dependsondir)[igal], f'{gal}{dependson}'), overwrite])
            else:
                missargs.append([checkfile, os.path.join(gdir, f'{gal}{dependson}'), overwrite])
        else:
            missargs.append([checkfile, None, overwrite])

    if verbose:
        t0 = time.time()
        log.info('Finding missing files...', end='')
    if mp > 1:
        with multiprocessing.Pool(mp) as P:
            todo = np.array(P.map(_missing_files_one, missargs))
    else:
        todo = np.array([_missing_files_one(_missargs) for _missargs in missargs])

    if verbose:
        log.info(f'...took {(time.time() - t0)/60.:.3f} min')

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
        _todo_indices = indices[itodo]

        if sample is not None:
            weight = np.atleast_1d(sample[DIAMCOLUMN])[_todo_indices]
            todo_indices = weighted_partition(weight, size)
        else:
            # unweighted
            todo_indices = np.array_split(_todo_indices, size)
    else:
        todo_indices = [np.array([])]

    return suffix, todo_indices, done_indices, fail_indices


def read_fits_catalog(catfile, ext=1, columns=None, rows=None):
    """Simple wrapper to read an input catalog.

    """
    if not os.path.isfile(catfile):
        print(f'Catalog {catfile} not found')
        return

    try:
        cat = Table(fitsio.read(catfile, ext=ext, rows=rows, columns=columns))
        print(f'Read {len(cat):,d} galaxies from {catfile}')
        return cat
    except:
        msg = f'Problem reading {catfile}'
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
            allcat.write(outfile, overwrite=True)
            print(f'Wrote {len(allcat):,d} objects to {outfile}')

            #outfile = os.path.join(outdir, f'wiseize-nedlvs-{region}.fits')
            #nedlvs.write(outfile, overwrite=True)
            #print(f'Wrote {len(nedlvs):,d} objects to {outfile}')

        return cat, fullcat


#def get_parentfile(version=None, kd=False):
#
#    if kd:
#        suffix = 'kd.fits'
#    else:
#        suffix = 'fits'
#        
#    parentfile = os.path.join(sample_dir(version=version), 'SGA-parent-{}.{}'.format(version, suffix))
#
#    return parentfile
#
#def read_parent(columns=None, verbose=False, first=None, last=None,
#                version=None, chaos=False):
#    """Read the SGA parent catalog.
#
#    """
#    if version is None:
#        version = parent_version()
#    
#    parentfile = get_parentfile(version=version)
#
#    if first and last:
#        if first > last:
#            print('Index first cannot be greater than index last, {} > {}'.format(first, last))
#            raise ValueError()
#    ext = 1
#    info = fitsio.FITS(parentfile)
#    nrows = info[ext].get_nrows()
#
#    rows = None
#    
#    # Read the CHAOS sample.
#    if chaos:
#        allgals = info[1].read(columns='GALAXY')
#        rows = np.hstack( [np.where(np.isin(allgals, chaosgal.encode('utf-8')))[0]
#                           for chaosgal in ('NGC0628', 'NGC5194', 'NGC5457', 'NGC3184')] )
#        rows = np.sort(rows)
#        nrows = len(rows)
#
#        nrows = info[1].get_nrows()
#
#    if first is None:
#        first = 0
#    if last is None:
#        last = nrows
#        if rows is None:
#            rows = np.arange(first, last)
#        else:
#            rows = rows[np.arange(first, last)]
#    else:
#        if last >= nrows:
#            print('Index last cannot be greater than the number of rows, {} >= {}'.format(last, nrows))
#            raise ValueError()
#        if rows is None:
#            rows = np.arange(first, last+1)
#        else:
#            rows = rows[np.arange(first, last+1)]
#
#    parent = Table(info[ext].read(rows=rows, upper=True, columns=columns))
#    if verbose:
#        if len(rows) == 1:
#            print('Read galaxy index {} from {}'.format(first, parentfile))
#        else:
#            print('Read galaxy indices {} through {} (N={}) from {}'.format(
#                first, last, len(parent), parentfile))
#
#    ## Temporary hack to add the data release number, PSF size, and distance.
#    #if chaos:
#    #    parent.add_column(Column(name='DR', dtype='S3', length=len(parent)))
#    #    gal2dr = {'NGC0628': 'DR7', 'NGC5194': 'DR6', 'NGC5457': 'DR6', 'NGC3184': 'DR6'}
#    #    for ii, gal in enumerate(np.atleast_1d(parent['GALAXY'])):
#    #        if gal in gal2dr.keys():
#    #            parent['DR'][ii] = gal2dr[gal]
#        
#    return parent
