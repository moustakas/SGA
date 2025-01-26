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


def get_raslice(ra):
    if np.isscalar(ra):
        return f'{int(ra):03d}'
    else:
        return [f'{int(onera):03d}' for onera in ra]


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


def parent_version(vicuts=False, nocuts=False):
    if nocuts:
        version = 'v1.0'
    elif vicuts:
        version = 'v1.0'
    else:
        version = 'v1.0'
    return version


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
        mult = read_hyperleda_multiples(rank=rank, rows=rows)
        #gals['ROW_GALAXIES'] = np.zeros(len(gals), np.int64) - 99
        #mult['ROW_MULTIPLES'] = np.zeros(len(mult), np.int64) - 99
        gals.rename_column('ROW', 'ROW_GALAXIES')
        mult.rename_column('ROW', 'ROW_MULTIPLES')
        hyper = vstack((gals, mult))
        hyper['ROW_GALAXIES'].fill_value = -99
        hyper['ROW_MULTIPLES'].fill_value = -99
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

    """
    ned = {
        'Antlia': 'Antlia Dwarf Spheroidal',
        'Aquarius': 'Aquarius dIrr',
        'Canes Venatici I': 'CVn I dSph',
        'Canes Venatici II': 'CVn II dSph',
        'CenA-MM-Dw1': 'CenA-Dw-133013-415321',
        'CenA-MM-Dw2': 'CenA-Dw-132956-415220',
        'CenA-MM-Dw3': 'Centaurus A:[CSS2016] MM-Dw03',
        'CenA-MM-Dw4': 'CenA-Dw-132302-414705',
        'CenA-MM-Dw5': 'CenA-Dw-131952-415938',
        'CenA-MM-Dw6': 'CenA-Dw-132557-410538',
        'CenA-MM-Dw7': 'CenA-Dw-132628-433318',
        'Cetus': 'Cetus Dwarf Spheroidal', # NED matches Cetus to Cetus II!
        'Clump I': 'PGC1 0028630 NED040',
        'Clump III': 'PGC1 0028630 NED039',
        'Donatiello III': '[TDW2021] 300548367', # ???
        'Draco': 'Draco Dwarf',
        'FM1': 'F06D1',
        'Fornax': 'Fornax Dwarf Spheroidal',
        'Hercules': 'Hercules dSph',
        'HIDEEP J1337-3320': 'GALEXASC J133700.38-332144.3',
        'HS 117': 'PGC1 0028630 NED027',
        'KK 17': 'WISEA J020010.18+284958.9',
        'KK 27': 'AM 0319-662',
        'KK 35': 'PGC1 0013826 NED006',
        'KK 77': 'F12D1',
        'KK 109': 'GALEXASC J114711.81+434018.1',
        'KK 160': 'SDSS J124357.79+433940.3',
        'KK 177': 'IC 4107',
        'KK 180': 'LSBC D575-08',
        'KK 182': 'PGC1 0166152 NED001',
        'KK 189': 'Centaurus A-dE1',
        'KK 195': 'GALEXASC J132108.25-313149.7',
        'KK 196': 'AM 1318-444',
        'KK 200': 'AM 1321-304',
        'KK 203': 'AM 1324-450',
        'KK 208': 'PGC1 0048082 NED002',
        'KK 211': 'AM 1339-445',
        'KK 217': 'AM 1343-452',
        'KK 218': 'Centaurus A-dE4',
        'KK 221': 'PGC1 0046957 NED025',
        'KK 242': 'SMDG J1752485+700816',
        'KK 258': 'ESO 468- G 020',
        'KKs 3': 'SGC 0224.3-7345',
        'KKs 51': 'WISEA J124421.29-425620.9',
        'KKs 53': 'PGC1 0046957 NED015',
        'KKs 54': 'Centaurus A-dE2',
        'KKs 55': 'PGC1 0046957 NED024',
        'KKs 57': 'PGC1 0046957 NED023',
        'KKs 58': 'Centaurus A-dE3',
        'KKSG 29': 'PGC1 0042407 NED014',
        'KKSG 31': 'PGC1 0042407 NED011',
        'KKSG 32': 'PGC1 0042407 NED015',
        'KKSG 33': 'PGC1 0042407 NED016',
        'KKSG 37': 'NSA 142339',
        'Leo I 09': 'NGC 3368:[CVD2018] DF6',
        'Leo VI': 'RCS 04020600360',
        'M101-DF1': 'PGC1 0050063 NED008',
        'M101-DF2': 'PGC1 0050063 NED007',
        'M101-DF3': 'PGC1 0050063 NED009',
        'M101 Dw9': 'Messier 101:[BSC2017] Dw 09',
        'M101 DwA': 'Messier 101:[BSC2017] Dw A',
        'NGC 4594-DGSAT-2': 'ELVES dw J1239m1120',
        'NGC 4594-DGSAT-3': 'ELVES dw J1239m1113',
        'NGC 4594-DW1': 'ELVES dw J1239m1143',
        'PGC 1059300': 'GALEXASC J130000.25-042135.8',
        'PGC 704814': 'WISEA J235840.73-312802.8',
        'PGC 725719': 'WISEA J132257.15-294417.7',
        'Pegasus IV': 'Pegasus IV Dwarf',
        'Pegasus V': 'Pegasus V Dwarf', # ='Andromeda XXXIV'
        'Pegasus III': 'Pegasus III Dwarf',
        'Pegasus dIrr': 'Pegasus Dwarf',
        'Perseus I': 'Andromeda XXXIII', # incorrectly resolves to 'PGC1 5067061 NED001'
        'Phoenix': 'Phoenix Dwarf',
        'Sagittarius': 'Sagittarius Dwarf Spheroidal',
        'Sagittarius dIrr': 'Sagittarius Dwarf Irregular',
        'Sculptor': 'Sculptor Dwarf Elliptical',
        'Sextans': 'Sextans dSph',
        'Tucana': 'Tucana Dwarf',
        'd0926+70': 'GALEXMSC J092627.92+703027.0',
        'd0939+71': 'PGC1 0028630 NED038',
        'd0955+70': 'PGC1 0028630 NED029',
        'd1028+70': 'WISEA J102839.75+701401.6',
        'd1041+70': 'PGC1 0028630 NED037',
        'dw1335-29': '[MJB2015] dw J1335-29',
        'd0934+70': 'PGC1 0028630 NED026',
        'd0944+69': 'PGC1 0028630 NED036',
        'd0944+71': 'GALEXMSC J094435.06+712857.6',
        'd0958+66': 'KUG 0945+670', # could also be GALEXASC J095848.78+665057.9??
        'd0959+68': 'PGC1 0028630 NED031',
        'd1006+67': 'PGC1 0028630 NED030',
        'd1014+68': 'WISEA J101456.37+684529.2',
        'd1015+69': 'PGC1 0028630 NED035',
        'dw1323-40a': '[CGG2021] dw J1323-40',
        'dw1329-45': 'Cen A:[MJP2016] dw1329-45',
        'dw1336-44': 'Cen A:[MJP2016] dw1336-44',
        'dw1340-30': '[KKM2018a] dw J1340-30',
        'dw1341-43': 'Cen A:[MJP2016] dw1341-43',
        'dw1342-43': 'Cen A:[MJP2016] dw1342-43',
        'dw1322-39': 'Cen A:[MJP2016] dw1322-39',
        'dw0036m2828': '[CGB2022] dw J0036-2828',
        # not in NED
        #'A0952+69': '',
        #'Aquarius III': '',
        #'Bedin 1': '',
        #'Cassiopea dIrr 1': '',
        #'Corvus A': '',
        #'Donatiello IV': '',
        #'Eridanus IV': '',
        #'Fluffy': '',
        #'KKs 59': '',
        #'KV19-212': '',
        #'KV19-271': '',
        #'KV19-329': '',
        #'KV19-442': '',
        #'Leo K': '',
        #'Leo M': '',
        #'Leo Minor I': '',
        #'MADCASH-1': '',
        #'MADCASH-2': '',
        #'MAPS 1231+42':
        #'MAPS 1249+44':
        #'Pavo': '',
        #'Sextans II': '',
        #'dw1323-40b': '[CGG2021] dw J1323-40b', # NED incorrectly cross-identifies this with [CGG2021] dw J1323-40
    }

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
    ver = 'v1.0.2'
    return ver


def read_lvd(rank=0, rows=None):
    """Read the Local Volume Database (LVD) dwarf-galaxy catalog.

    """
    version = version_lvd()

    # combine the dwarf-all and dwarf-local-field-distant files
    lvdfile = os.path.join(sga_dir(), 'parent', 'external', f'LVD_{version}.fits')
    if not os.path.isfile(lvdfile):
        from astropy.table import vstack
        allfile = os.path.join(sga_dir(), 'parent', 'external', f'LVD_dwarf_all_{version}.csv')
        lvd = Table.read(allfile)
        ## typos
        #lvd['name'][lvd['name'] == 'KKS53'] = 'KKS 53'
        #for obj in ['KKs 3', 'KKs 51', 'KKs 53', 'KKs 54', 'KKs 55', 'KKs 57', 'KKs 58', 'KKs 59']:
        #    I = np.where(lvd['name'] == obj)[0]
        #    lvd['name'][I] = obj.upper()

        # drop unconfirmed systems
        print(f'Dropping {np.sum(lvd["confirmed_real"]==0):,d}/{len(lvd):,d} unconfirmed dwarfs.')
        #print(f'Rank {rank:03d}: Dropping {np.sum(lvd["confirmed_real"]==0):,d}/{len(lvd):,d} unconfirmed dwarfs.')
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

    # PGC1198787 = KKR53 ???
    # PGC1330929 = KKR33 ??
    # PGC200320 = KKR13 ??
    # PGC2801041 = KKR43 ??
    # PGC2801056 = KKR63
    # PGC2801063 = KKR73
    # PGC57522 = KKR23
    pgc = {
		'Aquarius III': 0,
		'A0952+69': 0,
		'AGC 239141': 5808786,
		'BK3N': 28529,
		'BK6N': 31286,
		'BTS 116': 0,
		'Camelopardalis A': 0,
		'Camelopardalis B': 0,
		'Cassiopea dIrr 1': 0,
		'Fluffy': 0,
		'KV19-212': 0,
		'KV19-271': 0,
		'KV19-329': 0,
		'KV19-442': 0,
		'Centaurus N': 4689187,
		'CGCG 189-050': 46257,
		'DDO 53': 24050,
		'DDO 78': 30664,
		'DDO 82': 30997,
		'DDO126': 40791,
		'DDO127': 41020,
		'DDO 133': 41636,
		'DDO 154': 43869,
		'DDO 165': 45372,
		'DDO 167': 45939,
		'DDO 168': 46039,
		'DDO 169': 46127,
		'DDO 169NW': 5057032,
		'DDO 181': 48332,
		'DDO 183': 49158,
		'DDO 226': 2578,
		'ESO 059-001': 21199,
		'ESO 115-021': 9962,
		'ESO 222-010': 52125,
		'ESO 245-005': 6430,
		'ESO 269-037': 45104,
		'ESO 269-058': 45717,
		'ESO272-025': 52591,
		'ESO 321-014': 39032,
		'ESO 324-024': 47171,
		'ESO 349-031': 621,
		'GARLAND': 29167,
		'GHOSTS I': 5067066,
		'HIPASS J1247-77': 3994669,
		'Holmberg I': 27605,
		'Holmberg II': 23324,
		'Holmberg IX': 28757,
		'IC 2574': 30819,
		'IC 3687': 42656,
		'IC 4182': 45314,
		'JKB83': 6657020,
		'JKB129': 4668290,
		'JKB142': 5808737,
		'KDG 52': 23521,
		'KDG 63': 29257,
		'KDG 73': 32667,
		'KDG 74': 33305,
		'KDG 215': 44055,
		'KK 17': 166065,
		'KK 27': 166073,
		'KK 35': 166077,
		'KK 77': 2815838,
		'KK 109': 166115,
		'KK 160': 166142,
		'KK 166': 166146,
		'KK 177': 87149,
		'KK 180': 86645,
		'KK 217': 166175,
		'KK 242': 4689184,
		'KKH 6': 2807103,
		'KKH 18': 2807110,
		'KKH 30': 95591,
		'KKH 37': 95597,
		'KKH 57': 2807133,
		'KKH 78': 2807147,
		'LV J1243+4127': 0,
		'LV J1313+1003': 0,
		'HS 117': 0,
		'Clump I': 0,
		'Clump III': 0,
		'MAPS 1231+42': 0,
		'MAPS 1249+44': 0,
		'MCG +06-27-017': 38685,
		'MCG +09-20-131': 39228,
		'NGC 59': 1034,
		'NGC 625': 5896,
		'NGC 891': 9031,
		'TT2009 25': 0,
		'TT2009 30': 0,
		'NGC 1313': 12286,
		'NGC 1569': 15345,
		'NGC 2366': 21102,
		'NGC 2915': 26761,
		'NGC2976': 28120,
		'NGC 3077': 29146,
		'NGC 3741': 35878,
		'NGC 4068': 38148,
		'NGC 4236': 39346,
		'NGC 4244': 39422,
		'NGC 4395': 40596,
		'NGC 4594-DGSAT-2': 0,
		'NGC 4594-DGSAT-3': 0,
		'NGC 4594-DW1': 0,
		'dw1240-1118': 0,
		'KKSG 29': 42120,
		'KKSG 31': 3097709,
		'KKSG 32': 3097710,
		'KKSG 33': 3097711,
		'KKSG 37': 3097714,
		'LV J1235-1104': 0,
		'SUCD1': 3793583,
		'dw1300+1843': 0,
		'NGC 5102': 46674,
		'NGC 5194': 47404,
		'dw1328+4703': 0,
		'dw1330+4731': 0,
		'NGC 5195': 47413,
		'NGC 5204': 47368,
		'NGC 5206': 47762,
		'NGC 5237': 48139,
		'NGC 5238': 47853,
		'NGC 5253': 48334,
		'NGC 6503': 60921,
		'NGC 6789': 63000,
		'NGC 7793': 73049,
		'PGC 34671': 34671,
		'PGC42730': 42730,
		'PGC 51659': 51659,
		'PGC 170257': 170257,
		'PGC 704814': 704814,
		'PGC 725719': 725719,
		'PGC 1059300': 1059300,
		'UGC00064': 591,
		'UGC 288': 1777,
		'UGC 685': 3974,
		'UGC 1703': 8484,
		'UGC 1807': 8947,
		'UGC 2684': 12514,
		'UGC 2716': 12719,
		'UGC 3600': 19871,
		'UGC 3698': 20264,
		'UGC 4483': 24213,
		'UGC 6451': 35264,
		'UGC 6456': 35286,
		'UGC 6541': 35684,
		'UGC 6757': 36758,
		'UGC 7298': 39316,
		'UGC 7490': 40367,
		'UGC 7596': 41036,
		'UGC 7605': 41048,
		'UGC 7929': 42985,
		'UGC 8215': 45506,
		'UGC 8245': 45546,
		'UGC 8638': 48280,
		'UGC 8833': 49452,
		'UGC 11411': 62814,
		'UGC 12894': 35,
		'UGCA 92': 15439,
		'UGCA 105': 16957,
		'UGCA 287': 41743,
		'UGCA 292': 42275,
		'UGCA 442': 72228,
        # older objects here
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
        'Antlia': 29194,
        'Antlia B': 5098252,
        'Antlia II': 6775392,
        'Aquarius': 65367,
        'Aquarius II': 5953206,
        'BK5N': 29231,
        'Bedin 1': 0,
        'Bootes I': 4713553,
        'Bootes II': 4713552,
        'Bootes III': 4713562,
        'Bootes IV': 0,
        'Bootes V': 0,
        'Canes Venatici I': 4689223,
        'Canes Venatici II': 4713558,
        'Carina': 19441,
        'Carina II': 0,
        'Carina III': 0,
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
        'Cetus': 3097691,
        'Cetus II': 6740632,
        'Columba I': 6740626,
        'Coma Berenices': 0,
        'Corvus A': 0,
        'Crater II': 5742923,
        'DDO 113': 39145,
        'DDO 125': 40904,
        'DDO 147': 43129,
        'DDO 190': 51472,
        'DDO 44': 21302,
        'DDO 6': 2902,
        'DDO 99': 37050,
        'Donatiello III': 0,
        'Donatiello IV': 0,
        'Draco': 60095,
        'Draco II': 0,
        'ESO 006-001': 23344,
        'ESO 269-066': 45916,
        'ESO 274-001': 54392,
        'ESO 294-G010': 1641,
        'ESO 325-011': 48738,
        'ESO 381-018': 42936,
        'ESO 381-020': 43048,
        'ESO 383-087': 49050,
        'ESO 384-016': 49615,
        'ESO 410-G005': 1038,
        'ESO 443-009': 43978,
        'ESO 444-084': 48111,
        'ESO 540-032': 2933,
        'Eridanus II': 5074553,
        'Eridanus IV': 0,
        'F8D1': 3097827,
        'FM1': 0,
        'Fornax': 10074,
        'GALFA Dw3': 5072714,
        'GALFA Dw4': 5072715,
        'GR 8': 44491,
        'Grus I': 5074558,
        'Grus II': 6740630,
        'HIDEEP J1337-3320': 677373,
        'HIPASS J1131-31': 5060432,
        'HIPASS J1337-39': 592761,
        'HIPASS J1348-37': 4614882,
        'Hercules': 4713560,
        'Holm IV': 49448,
        'Horologium I': 5074554,
        'Horologium II': 5092747,
        'Hydra II': 5074546,
        'Hydrus I': 0,
        'IC 10': 1305,
        'IC 1613': 3844,
        'IC 3104': 39573,
        'IC 4247': 47073,
        'IC 4316': 48368,
        'IC 4662': 60849,
        'IC 5152': 67908,
        'IKN': 4689195,
        'KDG 2': 2881,
        'KDG 61': 28731,
        'KDG 64': 29388,
        'KK 182': 166152,
        'KK 189': 166158,
        'KK 195': 166163,
        'KK 196': 46663,
        'KK 197': 100039,
        'KK 200': 46885,
        'KK 203': 166167,
        'KK 208': 166170,
        'KK 211': 48515,
        'KK 213': 100040,
        'KK 218': 166176,
        'KK 221': 166179,
        'KK 258': 69468,
        'KKH 22': 2807114,
        'KKH 86': 2807150,
        'KKH 98': 2807157,
        'KKR 25': 2801026,
        'KKR 3': 166185,
        'KKs 3': 9140,
        'KKs 53': 2815820,
        'KKs 51': 2815819,
        'KKs 54': 2815821,
        'KKs 55': 2815822,
        'KKs 57': 2815823,
        'KKs 58': 2815824,
		'KKs 59': 0, # incorrectly matches to PGC135780
        'LGS 3': 3792,
        'LMC': 17223,
        'LV J0055-2310': 6740710,
        'LV J1157+5638': 2543081,
        'LV J1157+5638 sat': 6740587,
        'LV J1228+4358': 5057024,
        'Lacerta I': 5065677,
        'Leo A': 28868,  # =Leo 3
        'Leo I': 29488,
        'Leo I 09': 4689210,
        'Leo II': 34176, # = Leo B
        'Leo IV': 4713561,
        'Leo K': 0,
        'Leo M': 0,
        'Leo Minor I': 0,
        'Leo P': 5065058,
        'Leo T': 4713564,
        'Leo V': 4713563,
        'Leo VI': 0,
        'M 32': 2555,
        'M101 Dw9': 0,
        'M101 DwA': 5067392,
        'M101-DF1': 5067385,
        'M101-DF2': 5067386,
        'M101-DF3': 5067387,
        'MADCASH-1': 0,
        'MADCASH-2': 0,
        'MCG-04-31-038': 45628,
        'NGC 147': 2004,
        'NGC 1560': 15488,
        'NGC 185': 2329,
        'NGC 205': 2429,
        'NGC 247': 2758,
        'NGC 300': 3238,
        'NGC 3109': 29128,
        'NGC 404': 4126,
        'NGC 4163': 38881,
        'NGC 4190': 39023,
        'NGC 4214': 39225,
        'NGC 4449': 40973,
        'NGC 5011C': 45917,
        'NGC 5264': 48467,
        'NGC 5474': 50216,
        'NGC 5477': 50262,
        'NGC 55': 1014,
        'NGC 5585': 51210,
        'NGC 6822': 63616,
        'Pavo': 0,
        'Pegasus III': 5074547,
        'Pegasus IV': 0,
        'Pegasus V': 0,
        'Pegasus W': 0, # in NED
        'Pegasus dIrr': 71538,
        'Perseus I': 5067061,
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
        'SMC': 3085,
        'Sagittarius': 4689212,
        'Sagittarius dIrr': 63287,
        'Scl-MM-Dw1': 5067869,
        'Scl-MM-Dw2': 5478807,
        'Scl-MM-Dw3': 0,
        'Scl-MM-Dw4': 0,
        'Scl-MM-Dw5': 0,
        'Sculptor': 3589,
        'Sculptor-dE1': 0,
        'Segue 1': 4713559,
        'Segue 2': 4713565,
        'Sextans': 88608, # = PGC088608
        'Sextans A': 29653,
        'Sextans B': 28913,
        'Sextans II': 0,
        'Triangulum II': 5074545,
        'Tucana': 69519,
        'Tucana B': 0, # = SMDG J2247005-582429
        'Tucana II': 5074560,
        'Tucana III': 6657034,
        'Tucana IV': 6740629,
        'Tucana V': 6740631,
        'UGC 4879': 26142,
        'UGC 5497': 29735,
        'UGC 8508': 47495,
        'UGC 8882': 49636,
        'UGC 9128': 50961,
        'UGC 9405': 52142,
        'UGCA 365': 48029,
        'UGCA 86': 14241,
        'UKS 2323-326': 71431,
        'Ursa Major I': 4713554,
        'Ursa Major II': 4713555,
        'Ursa Major III': 0, # discovered in 2024
        'Ursa Minor': 54074,
        'Virgo I': 6657032,
        'Virgo II': 0,
        'WLM': 143,
        'Willman 1': 4713556,
        'd0926+70': 5056943,
        'd0939+71': 5056932,
        'd0955+70': 5056934,
        'd1028+70': 5056941,
        'd1041+70': 5056942,
        'dw1335-29': 5477876, # = [MJB2015]DW1335-29
        'd0934+70': 5056931,
        'd0944+69': 5056933,
        'd0944+71': 5056944, # GALEXMSC J094435.06+712857.6??
        'd0958+66': 28826, # KUG 0945+670??
        'd0959+68': 5056936,
        'd1006+67': 5056937,
        'd1014+68': 5056938,
        'd1015+69': 5056947,
        'dw0036m2828': 0,
        'dw1046+1244': 0,
        'dw1322-39': 5912201, # = [MJB2016]DW1322-39
        'dw1323-40a': 0,
        'dw1323-40b': 0,
        'dw1329-45': 5912206, # = [MJB2016]DW1329-45
        'dw1336-44': 5912210, # = [MJB2016]DW1336-44
        'dw1340-30': 0,
        'dw1341-43': 5912213, # = [MJB2016]DW1341-43
        'dw1342-43': 5912214, # = [MJB2016]DW1342-43
        #'Cetus III': 6726344,
        #'Virgo III': 0,
        #'NGC 55-dw1': 0,
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
        csvfile = os.path.join(sga_dir(), 'parent', 'external', f'custom-external_{version}.csv')
        data = Table.read(csvfile, format='csv', comment='#')
        data['mag_band'] = data['mag_band'].astype('<U1')
        data['mag_band'].fill_value = ''
        data['objname_ned'].fill_value = ''
        data = data.filled()
        for col in ['diam', 'ba', 'pa', 'mag']:
            data[col] = data[col].astype('f4')
        [data.rename_column(col, col.upper()) for col in data.colnames]
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


def custom_brickname(ra, dec):
    brickname = '{:08d}{}{:07d}'.format(
        int(100000*ra), 'm' if dec < 0 else 'p',
        int(100000*np.abs(dec)))
    #brickname = '{:06d}{}{:05d}'.format(
    #    int(1000*ra), 'm' if dec < 0 else 'p',
    #    int(1000*np.abs(dec)))
    return brickname

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
