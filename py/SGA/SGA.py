"""
SGA.SGA
=======

Code to build and analyze the SGA sample.

"""
import os, time, pdb
import fitsio
import numpy as np
import numpy.ma as ma
from astropy.table import Table, vstack, hstack, join

from SGA.ellipse import MAXSHIFT_ARCSEC
from SGA.logger import log


REFCAT = 'L4'
RACOLUMN = 'GROUP_RA'   # 'RA'
DECCOLUMN = 'GROUP_DEC' # 'DEC'
DIAMCOLUMN = 'GROUP_DIAMETER' # 'DIAM'
REFIDCOLUMN = 'SGAID'

SAMPLE = dict(
    LVD = 2**0,      # Local Volume Database dwarfs
    MCLOUDS = 2**1,  # in the Magellanic Clouds
    GCLPNE = 2**2,   # in a globular cluster or PNe mask (implies --no-force-gaia)
    NEARSTAR = 2**3, # STARFDIST < 1.2
    INSTAR = 2**4,   # STARFDIST < 0.5
)

OPTMASKBITS = dict(
    brightstar = 2**0, # BRIGHT,MEDIUM,CLUSTER legacypipe MASKBITS
    gaiastar = 2**1,   # Gaia (type=PSF) stars
    galaxy = 2**2,     # galaxy (extended, non-reference) sources
    reference = 2**3,  # SGA (reference) sources
    g = 2**4,          # g-band
    r = 2**5,          # r-band
    i = 2**6,          # i-band
    z = 2**7,          # z-band
)

GALEXMASKBITS = dict(
    brightstar = 2**0,
    gaiastar = 2**1,
    galaxy = 2**2,
    reference = 2**3,
    FUV = 2**4,
    NUV = 2**5,
)

UNWISEMASKBITS = dict(
    brightstar = 2**0,
    gaiastar = 2**1,
    galaxy = 2**2,
    reference = 2**3,
    W1 = 2**4,
    W2 = 2**5,
    W3 = 2**6,
    W4 = 2**7,
)

SBTHRESH = [22., 23., 24., 25., 26.] # surface brightness thresholds
APERTURES = [0.5, 1., 1.25, 1.5, 2.] # multiples of SMA_MOMENT


def SGA_version(vicuts=False, nocuts=False, archive=False, parent=False):
    # nocuts, vicuts, and archive *have* to share a version (too
    # confusing otherwise!)
    version_work = 'v0.10'

    if nocuts:
        version = version_work
    elif vicuts:
        version = version_work
    elif archive:
        version = version_work
    elif parent:
        version = 'v0.10'
    else:
        # merged SGA catalog
        version = 'v0.10'
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


def sga2025_name(ra, dec, group_name=False, unixsafe=False):
    # simple wrapper on radec_to_name with precision=3
    from SGA.io import radec_to_name, radec_to_groupname
    if group_name:
        # 36-arcsec precision (0.01 degrees)
        return radec_to_groupname(ra, dec, prefix='SGA2025_')
    else:
        # 3.6-arcsec precision (0.001 degrees)
        return radec_to_name(ra, dec, prefix='SGA2025',
                             precision=3, unixsafe=unixsafe)


def get_galaxy_galaxydir(sample, region='dr11-south', group=True,
                         datadir=None, htmldir=None, html=False):
    """Retrieve the galaxy name and the (nested) directory.

    """
    from SGA.io import get_raslice

    if datadir is None:
        datadir = sga_data_dir()
    if htmldir is None:
        htmldir = sga_html_dir()
    dataregiondir = os.path.join(datadir, region)
    htmlregiondir = os.path.join(htmldir, region)

    sample = Table(sample) # can't be a Row

    objdirs, htmlobjdirs = [], []
    if group:# and 'SGAGROUP' in sample.colnames:
        racolumn = 'GROUP_RA'
        galcolumn = 'SGAGROUP'
        groupcolumn = 'GROUP_NAME'

        ras = np.atleast_1d(sample[racolumn].value)
        objs = np.atleast_1d(sample[galcolumn].value)
        grps = np.atleast_1d(sample[groupcolumn].value)
    else:
        racolumn = 'RA'
        #galcolumn = 'OBJNAME'
        #galcolumn = 'SGANAME'

        ras = np.atleast_1d(sample[racolumn].value)
        objs = sga2025_name(sample['RA'].value, sample['DEC'].value, unixsafe=True)
        grps = objs

    for ra, grp in zip(ras, grps):
        objdirs.append(os.path.join(dataregiondir, get_raslice(ra), grp))
        if html:
            htmlobjdirs.append(os.path.join(htmlregiondir, get_raslice(ra), grp))

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
                  no_groups=False, verbose=False, datadir=None, htmldir=None,
                  size=1, mp=1):
    """Figure out which files are missing and still need to be processed.

    """
    from glob import glob
    import multiprocessing
    import astropy
    from SGA.mpi import distribute_work
    #from SGA.mpi import weighted_partition
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

    if no_groups:
        group = False
        DIAMCOL = 'DIAM'
    else:
        group = True
        DIAMCOL = DIAMCOLUMN

    dependson, dependsondir = None, None
    if htmlplots is False and htmlindex is False:
        if verbose:
            t0 = time.time()
            log.debug('Getting galaxy names and directories...')
        galaxy, galaxydir = get_galaxy_galaxydir(sample, region=region,
                                                 group=group,
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
            group=group, html=True)
    elif htmlindex:
        suffix = 'htmlindex'
        filesuffix = '-montage.png'
        galaxy, _, galaxydir = get_galaxy_galaxydir(
            sample, datadir=datadir, htmldir=htmldir,
            region=region, group=group, html=True)
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
        todo_indices, loads = distribute_work(sample[DIAMCOL].value, itodo=itodo,
                                              size=size, p=2.0, verbose=True)
    else:
        todo_indices = [np.array([])]

    return suffix, todo_indices, done_indices, fail_indices


def read_sample(first=None, last=None, galaxylist=None, verbose=False, columns=None,
                no_groups=False, lvd=False, final_sample=False, test_bricks=False,
                region='dr11-south', mindiam=0., maxdiam=200., maxmult=None):
    """Read/generate the parent SGA catalog.

    mindiam,maxdiam in arcmin
    maxmult - maximum number of group members (ignored if --no-groups is set)

    """
    import fitsio

    if lvd:
        no_groups = True # NB

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
        version = SGA_version(parent=True)
        samplefile = os.path.join(sga_dir(), 'sample', f'SGA2025-parent-{version}.fits')

    if not os.path.isfile(samplefile):
        msg = f'Sample file {samplefile} not found.'
        log.critical(msg)
        raise IOError(msg)

    if final_sample:
        cols = ['GROUP_DIAMETER', 'GROUP_PRIMARY', 'SGA_ID', 'PREBURNED']
        info = fitsio.read(samplefile, columns=cols)

        #rows = np.where(
        #    (info['GROUP_DIAMETER'] > mindiam) *
        #    (info['GROUP_DIAMETER'] < maxdiam) *
        #    info['GROUP_PRIMARY'] *
        #    info['PREBURNED'] *
        #    (info['SGA_ID'] > -1))[0]
    else:
        #cols = ['GROUP_NAME', 'GROUP_RA', 'GROUP_DEC', 'GROUP_DIAMETER', 'GROUP_MULT',
        #        'GROUP_PRIMARY', 'GROUP_ID', 'SGAID', 'RA', 'DEC', 'BRICKNAME']
        if no_groups:
            cols = ['DIAM']
        else:
            cols = ['GROUP_DIAMETER', 'GROUP_PRIMARY']
            if maxmult is not None:
                cols += ['GROUP_MULT']
        info = fitsio.read(samplefile, columns=cols)
        if no_groups:
            rows = np.where(
                (info['DIAM'] > mindiam) *
                (info['DIAM'] <= maxdiam))[0]
        else:
            I = ((info['GROUP_DIAMETER'] > mindiam) *
                 (info['GROUP_DIAMETER'] <= maxdiam) *
                 info['GROUP_PRIMARY'])
            if maxmult is not None:
                I *= info['GROUP_MULT'] <= maxmult
            rows = np.where(I)[0]

    nallrows = len(info)
    nrows = len(rows)

    fullsample = Table(fitsio.read(samplefile, upper=True))
    #fullsample.add_column(np.arange(nallrows), name='INDEX', index=0)
    sample = fullsample[rows]

    #sample = Table(info[ext].read(rows=rows, upper=True, columns=columns))
    log.info(f'Read {len(sample):,d}/{len(fullsample):,d} GROUP_PRIMARY objects from {samplefile}')
    if len(sample) == 0:
        return sample, fullsample

    if region is not None:
        # select objects in this region
        from SGA.coadds import REGIONBITS
        I = sample['REGION'] & REGIONBITS[region] != 0
        #J = fullsample['REGION'] & REGIONBITS[region] != 0
        log.info(f'Selecting {np.sum(I):,d}/{len(sample):,d} objects in ' + \
                 f'region={region}')

        sample = sample[I]
        if no_groups:
            fullsample = sample
        else:
            fullsample = fullsample[np.isin(fullsample['GROUP_ID'], sample['GROUP_ID'])]
        if len(sample) == 0:
            return sample, fullsample

    # select objects in the set of test bricks
    if test_bricks:
        from SGA.brick import brickname as get_brickname
        testbricksfile = os.path.join(sga_dir(), 'sample', 'dr11a-testbricks.csv')
        #testbricksfile = os.path.join(sga_dir(), 'sample', 'dr11-testbricks.csv')
        testbricks = Table.read(testbricksfile, format='csv')['brickname'].value
        log.info(f'Read {len(testbricks)} test bricks from {testbricksfile}')
        allbricks = get_brickname(sample['GROUP_RA'].value, sample['GROUP_DEC'].value)
        sample = sample[np.isin(allbricks, testbricks)]
        if no_groups:
            fullsample = sample
        else:
            fullsample = fullsample[np.isin(fullsample['GROUP_ID'], sample['GROUP_ID'])]
        if len(sample) == 0:
            return sample, fullsample

    # select the LVD sample; remember that --lvd always implies --no-groups
    if lvd:
        sample = sample[sample['SAMPLE'] & SAMPLE['LVD'] != 0]
        fullsample = sample
        if len(sample) == 0:
            return sample, fullsample

    if False:#True:
        from SGA.ellipse import ELLIPSEMODE
        I = sample['ELLIPSEMODE'] & ELLIPSEMODE['RESOLVED'] == 0
        log.warning(f'Temporarily removing {np.sum(~I):,d} LVD-RESOLVED sources!')
        sample = sample[I]
        fullsample = fullsample[np.isin(fullsample['GROUP_ID'], sample['GROUP_ID'])]

    if galaxylist is not None:
        galaxylist = np.array(galaxylist.split(','))
        log.debug('Selecting specific galaxies.')
        I = np.isin(sample['GROUP_NAME'], galaxylist)
        if np.count_nonzero(I) == 0:
            #log.warning('No matching galaxies using column GROUP_NAME!')
            #I = np.isin(sample['SGANAME'], galaxylist)
            I = np.isin(sample['SGAID'], galaxylist)
            if np.count_nonzero(I) == 0:
                #log.warning('No matching galaxies using column SGANAME!')
                I = np.isin(sample['OBJNAME'], galaxylist)
                if np.count_nonzero(I) == 0:
                    log.warning('No matching galaxies found in sample; try a different region?')
                    sample, fullsample = Table(), Table()
        sample = sample[I]

    # select a subset of objects
    if first is not None or last is not None:
        nsample = len(sample)
        if first is None:
            first = 0
        if last is None:
            last = nsample
        if last > nsample:
            log.warning('Index last is greater than the number of ' + \
                        f'objects in sample, {last} >= {nsample}')
            last = nsample
        I = np.arange(first, last)
        if nsample == 1:
            log.info(f'Selecting index {first} (N=1)')
        else:
            log.info(f'Selecting indices {first} through {last} (N={nsample:,d})')
        sample = sample[I]
        if no_groups:
            fullsample = sample
        else:
            fullsample = fullsample[np.isin(fullsample['GROUP_ID'], sample['GROUP_ID'])]

    return sample, fullsample


def SGA_diameter(ellipse, radius_arcsec=False):
    """
    radius_arcsec - do not convert to diameter in arcmin

    """
    radius = np.zeros(len(ellipse))
    ref = np.zeros(len(ellipse), '<U6')

    # r-band R(26)
    I =  (radius == 0.) * (ellipse['R26_R'] > 0.) * (ellipse['R26_ERR_R'] > 0.)
    if np.any(I):
        radius[I] = ellipse['R26_R'][I].value
        ref[I] = 'R26_R'

    # r-band R(25)
    I =  (radius == 0.) * (ellipse['R25_R'] > 0.) * (ellipse['R25_ERR_R'] > 0.)
    if np.any(I):
        radius[I] = ellipse['R25_R'][I].value * 1.28 # median factor
        ref[I] = 'R25_R'

    # r-band R(24)
    I =  (radius == 0.) * (ellipse['R24_R'] > 0.) * (ellipse['R24_ERR_R'] > 0.)
    if np.any(I):
        radius[I] = ellipse['R24_R'][I].value * 1.70 # median factor
        ref[I] = 'R24_R'

    # sma_moment
    I =  (radius == 0.) * (ellipse['SMA_MOMENT'] > 0.)
    if np.any(I):
        radius[I] = ellipse['SMA_MOMENT'][I].value * 1.19 # median factor
        ref[I] = 'MOMENT'

    if radius_arcsec:
        return radius, ref
    else:
        # convert radius-->diameter and arcsec-->arcmin
        diam = radius * 2. / 60. # [arcmin]
        return np.float32(diam), ref


def SGA_geometry(ellipse):

    ba = ellipse['BA_MOMENT'].value
    pa = ellipse['PA_MOMENT'].value
    diam, diam_ref = SGA_diameter(ellipse)
    return diam, ba, pa, diam_ref


def SGA_datamodel(ellipse, bands, all_bands):
    import astropy.units as u
    from astropy.table import Column, MaskedColumn

    nobj = len(ellipse)

    ubands = np.char.upper(bands)
    uall_bands = np.char.upper(all_bands)

    print('Need to add DIAM_INIT_REF and BANDS.')

    dmcols = [
        # from original sample
        ('SGAID', np.int64, None),
        ('SGAGROUP', 'U18', None),
        ('REGION', np.int16, None),
        ('OBJNAME', 'U30', None),
        ('PGC', np.int64, None),
        ('SAMPLE', np.int32, None),
        ('ELLIPSEMODE', np.int32, None),
        ('FITMODE', np.int32, None),
        ('BX_INIT', np.float32, u.pixel),
        ('BY_INIT', np.float32, u.pixel),
        ('RA_INIT', np.float64, u.degree),
        ('DEC_INIT', np.float64, u.degree),
        ('SMA_INIT', np.float32, u.arcsec),
        ('DIAM_INIT', np.float32, u.arcmin),
        ('BA_INIT', np.float32, None),
        ('PA_INIT', np.float32, u.degree),
        ('MAG_INIT', np.float32, u.mag),
        ('DIAM_REF_INIT', 'U9', None),
        #('BAND', 'U1', None),
        ('EBV', np.float32, u.mag),
        ('GROUP_ID', np.int32, None),
        ('GROUP_NAME', 'U10', None),
        ('GROUP_MULT', np.int16, None),
        ('GROUP_PRIMARY', bool, None),
        ('GROUP_RA', np.float64, u.degree),
        ('GROUP_DEC', np.float64, u.degree),
        ('GROUP_DIAMETER', np.float32, u.arcmin),
        ('PSFSIZE_G', np.float32, u.arcsec),
        ('PSFSIZE_R', np.float32, u.arcsec),
        ('PSFSIZE_I', np.float32, u.arcsec),
        ('PSFSIZE_Z', np.float32, u.arcsec),
        ('PSFDEPTH_G', np.float32, u.mag),
        ('PSFDEPTH_R', np.float32, u.mag),
        ('PSFDEPTH_I', np.float32, u.mag),
        ('PSFDEPTH_Z', np.float32, u.mag),
        ('BANDS', 'U4', None),
        ('SGANAME', 'U25', None),
        ('RA', np.float64, u.degree),
        ('DEC', np.float64, u.degree),
        ('BX', np.float32, u.pixel),
        ('BY', np.float32, u.pixel),
        ('SMA_MOMENT', np.float32, u.arcsec),
        ('BA_MOMENT', np.float32, None),
        ('PA_MOMENT', np.float32, u.degree),
        ('RA_TRACTOR', np.float64, u.degree),
        ('DEC_TRACTOR', np.float64, u.degree),
        ('ELLIPSEBIT', np.int32, None),
    ]
    for filt in ubands:
        dmcols += [(f'MW_TRANSMISSION_{filt}', np.float32, None)]
    for filt in uall_bands:
        dmcols += [(f'GINI_{filt}', np.float32, None)]
    for param, unit, dtype in zip(
            ['COG_MTOT', 'COG_DMAG', 'COG_LNALPHA1', 'COG_LNALPHA2', 'COG_CHI2', 'COG_NDOF', 'SMA50'],
            [u.mag, u.mag, None, None, None, None, u.arcsec],
            ['f4', 'f4', 'f4', 'f4', 'f4', np.int32, 'f4']):
        for filt in uall_bands:
            dmcols += [(f'{param}_{filt}', dtype, unit)]
        if not ('CHI2' in param or 'NDOF' in param):
            for filt in uall_bands:
                dmcols += [(f'{param}_ERR_{filt}', dtype, unit)]

    # flux within apertures that are multiples of sma_moment
    for iap in range(len(APERTURES)):
        dmcols += [(f'SMA_AP{iap:02}', np.float32, u.arcsec)]
    for iap in range(len(APERTURES)):
        for filt in uall_bands:
            dmcols += [(f'FLUX_AP{iap:02}_{filt}', np.float32, u.nanomaggy)]
        for filt in uall_bands:
            dmcols += [(f'FLUX_ERR_AP{iap:02}_{filt}', np.float32, u.nanomaggy)]
        for filt in uall_bands:
            dmcols += [(f'FMASKED_AP{iap:02}_{filt}', np.float32, None)]

    # optical isophotal radii
    for thresh in SBTHRESH:
        for filt in ubands:
            dmcols += [(f'R{thresh:.0f}_{filt}', np.float32, u.arcsec)]
        for filt in ubands:
            dmcols += [(f'R{thresh:.0f}_ERR_{filt}', np.float32, u.arcsec)]

    # final diameters
    dmcols += [
        ('D26', np.float32, u.arcmin),
        ('BA', np.float32, None),
        ('PA', np.float32, u.degree),
        ('D26_REF', '<U5', None),
    ]

    out = Table()
    for col in dmcols:
        out.add_column(Column(name=col[0], data=np.zeros(nobj, dtype=col[1]), unit=col[2]))

    # copy over the data
    check = []
    for col in out.colnames:
        if col in ellipse.colnames:
            val = ellipse[col]
            if not (isinstance(val, str) or 'U' in str(val.dtype)):
                if type(val) is MaskedColumn:
                    I = val.mask
                else:
                    I = np.logical_or(np.isnan(val.value), np.logical_not(np.isfinite(val.value)))
                if np.any(I):
                    log.warning(f'Zeroing out {np.sum(I):,d} masked (or NaN) {col} values.')
                    I = np.where(I)[0]
                    #pdb.set_trace()
                    check.append(I)
                    val[I] = 0
            out[col] = val
    if len(check) > 0:
        check = np.unique(np.hstack(check))
        print(','.join(ellipse['GROUP_NAME'][check].value))
        #pdb.set_trace()

    return out


def _empty_tractor(cat):
    for col in cat.colnames:
        if cat[col].dtype == bool:
            cat[col] = True # False # [brick_primary]
        else:
            cat[col] *= 0
    return cat


def _build_catalog_one(args):
    """Wrapper function for the multiprocessing."""
    return build_catalog_one(*args)


def build_catalog_one(igrp, grp, gdir, refid_array, datasets, opt_bands):
    """Gather the ellipse-fitting results for a single group.

    No tractor or ellipse catalog for this object:
    dr11-south/203/20337p3381

    """
    import fitsio
    from os import getpid
    from glob import glob
    from legacypipe.bits import MASKBITS
    from SGA.ellipse import ELLIPSEBIT

    if not os.path.isdir(gdir):
        return Table(), Table()

    # gather the ellipse catalogs
    ellipsefiles = glob(os.path.join(gdir, f'*-ellipse-{opt_bands}.fits'))
    if len(ellipsefiles) == 0:
        log.warning(f'All ellipse files missing for {gdir}/{grp}')
        return Table(), Table()

    if igrp % 500 == 0:
        log.info(f'Process {getpid()}: working on group {igrp:,d}')

    nsample = len(refid_array)

    if len(ellipsefiles) > nsample:
        msg = f'Found vestigial ellipse files in {gdir}; please remove!'
        log.critical(msg)
        raise IOError(msg)


    ellipse = []
    for ellipsefile in ellipsefiles:
        # fragile!!
        sganame = os.path.basename(ellipsefile).split('-')[:-2]
        if len(sganame) == 1:
            sganame = sganame[0]
        else:
            sganame = '-'.join(sganame)

        # loop on datasets and join
        for idata, dataset in enumerate(datasets):
            ellipsefile_dataset = os.path.join(gdir, f'{sganame}-ellipse-{dataset}.fits')
            try:
                ellipse_dataset = Table(fitsio.read(ellipsefile_dataset, ext='ELLIPSE'))
            except:
                msg = f'Problem reading {ellipsefile_dataset}!'
                log.critical(msg)
                break

            if idata == 0:
                ellipse1 = ellipse_dataset
            else:
                ellipse1 = join(ellipse1, ellipse_dataset)

        ellipse.append(ellipse1)

    if len(ellipse) > 0:
        ellipse = vstack(ellipse)

    assert(np.all(np.isin(ellipse[REFIDCOLUMN], refid_array)))

    # Read the Tractor catalog for all the SGA sources as well as for
    # all sources within the SGA ellipse (using MASKBITS).
    tractorfile = os.path.join(gdir, f'{grp}-tractor.fits')
    refs = fitsio.read(tractorfile, columns=['brick_primary', 'ra', 'dec', 'type', 'fitbits',
                                             'ref_cat', 'ref_id', 'maskbits'])
    I = refs['brick_primary'] * (refs['ref_cat'] != 'G3') * (refs['type'] != 'DUP')
    # if np.sum(I)==0, this is a problem...
    if np.sum(I) == 0:
        msg = f'All sources dropped in {tractorfile}'
        log.critical(msg)
        raise ValueError(msg)
    else:
        J = I * np.logical_or((refs['maskbits'] & MASKBITS['GALAXY'] != 0),
                              refs['ref_cat'] == REFCAT)
        # J can be empty if the initial geometry is so pathological
        # that the SGA source doesn't get MASKBITS set (or that the
        # SGA source is dropped *and* there are no sources within the
        # initial elliptical geometry).
        if np.sum(J) == 0:
            I = np.where(I)[0]
        else:
            I = np.where(J)[0]
    tractor = Table(fitsio.read(tractorfile, rows=I))

    # Remove SGA Tractor that don't "belong" to this group.
    rem = np.where(np.logical_not(np.isin(tractor['ref_id'], ellipse[REFIDCOLUMN])) *
                   (tractor['ref_cat'] == REFCAT))[0]
    if len(rem) > 0:
        tractor.remove_rows(rem)

    # Tractor catalog of the SGA source(s)
    tractor_sga = []
    for ellipse1 in ellipse:
        I = np.where((refs['ref_id'] == ellipse1[REFIDCOLUMN]) *
                     (refs['ref_cat'] == REFCAT))[0]
        # If there's no match, confirm that the NOTRACTOR bit was set,
        # read a blank catalog, and then move on. Should never happen!
        # dr11-south/195/19533p2848/SDSS J130120.01+282848.5
        if len(I) == 0:
            assert(ellipse1['ELLIPSEBIT'] & ELLIPSEBIT['NOTRACTOR'] != 0)
            tractor_sga1 = _empty_tractor(Table(fitsio.read(tractorfile, rows=[0])))
            tractor_sga1['ref_cat'] = REFCAT
            tractor_sga1['ref_id'] = ellipse1[REFIDCOLUMN]
            tractor_sga.append(tractor_sga1)

    if len(tractor_sga) > 0:
        tractor_sga = vstack(tractor_sga)
        tractor = vstack((tractor, tractor_sga))

    return ellipse, tractor


def build_catalog(sample, fullsample, comm=None, bands=['g', 'r', 'i', 'z'],
                  region='dr11-south', galex=True, unwise=True, mp=1,
                  no_groups=False, datadir=None, verbose=False,
                  clobber=False):
    """Build the final catalog.

    FIXME - combine the north and south

    """
    import time
    from glob import glob
    import multiprocessing
    from astropy.io import fits
    from SGA.ellipse import ELLIPSEBIT, FITMODE
    from SGA.coadds import REGIONBITS
    from SGA.util import match, get_dt


    if comm:
        rank, size = comm.rank, comm.size
    else:
        rank, size = 0, 1

    print('If the GCPNe samplebit is set, do not pass forward Tractor sources (other than the SGA source).')
    print('E.g., ESO 050- G 010 is on the edge of NGC104 and we want the sources to match the DR11 maskbits')

    all_bands = np.copy(bands)
    opt_bands = ''.join(bands)
    datasets = [opt_bands]
    if unwise:
        datasets += ['unwise']
        all_bands = np.append(all_bands, ['W1', 'W2', 'W3', 'W4'])
    if galex:
        datasets += ['galex']
        all_bands = np.append(all_bands, ['FUV', 'NUV'])


    version = SGA_version()
    outfile = os.path.join(sga_dir(), 'sample', f'SGA2025-{version}-{region}.fits')
    outfile_ellipse = os.path.join(sga_dir(), 'sample', f'SGA2025-ellipse-{version}-{region}.fits')
    kdoutfile_ellipse = os.path.join(sga_dir(), 'sample', f'SGA2025-ellipse-{version}-{region}.kd.fits')
    if os.path.isfile(outfile) and not clobber:
        if rank == 0:
            log.warning(f'Use --clobber to overwrite existing catalog {outfile}')
        return

    #I = sample['REGION'] & REGIONBITS[region] != 0
    #J = fullsample['REGION'] & REGIONBITS[region] != 0
    #sample_region = sample[I]
    #fullsample_region = fullsample[J]

    # testing
    #sample = sample[sample['GROUP_MULT'] > 1]
    #sample = sample[:700]
    #log.info(f'Trimmed to {len(sample):,d} groups in region={region}')

    print('HACK!!!')
    from SGA.brick import brickname as get_brickname
    bricks = get_brickname(sample['GROUP_RA'].value, sample['GROUP_DEC'].value)
    I = np.isin(bricks, ['1943p265'])
    sample = sample[I]
    pdb.set_trace()

    group, groupdir = get_galaxy_galaxydir(
        sample, region=region,
        group=not no_groups, datadir=datadir)
    group = np.atleast_1d(group)
    groupdir = np.atleast_1d(groupdir)
    ngrp = len(group)

    # divide into chunks and assign to different ranks (if running with MPI)
    if rank == 0:
        t0 = time.time()
        nperchunk = 2**12 # 14 # 2**8

        # clean up previous runs
        chunkfiles = glob(os.path.join(sga_dir(), 'sample', f'SGA2025-{version}-{region}-chunk*.fits'))
        for chunkfile in chunkfiles:
            os.remove(chunkfile)

        nchunk = int(np.ceil(ngrp / nperchunk))
        chunkindx = np.array_split(np.arange(ngrp), nchunk)
        nperchunk = int(np.mean([len(indx) for indx in chunkindx]))

        log.info(f'Dividing the sample into {nchunk:,d} chunk(s) with ' + \
                 f'{nperchunk:,d} objects per chunk.')

        #nchunkperrank = int(np.ceil(nperchunk / size))
        #log.info(f'Distributing chunks to {size} MPI ranks with approximately {nchunkperrank} chunks per rank.')
        #chunkfiles = [os.path.join(sga_dir(), 'sample', f'SGA2025-{version}-{region}-chunk{ichunk}.fits')
        #              for ichunk in range(nchunk)]

    chunkfiles = comm.bcast(chunkfiles, root=0)

    if comm:
        comm.barrier()

    chunkfiles = []
    for ichunk, indx in enumerate(chunkindx):
        chunkfile = os.path.join(sga_dir(), 'sample', f'SGA2025-{version}-{region}-chunk{ichunk}.fits')
        log.info(f'Dividing chunk {ichunk+1}/{nchunk} among {mp} cores ({len(indx)//mp:,d} objects per core).')
        log.info(f'  Writing to: {chunkfile}')
        chunkfiles.append(chunkfile)

        mpargs = []
        for iobj, grp, gdir in zip(indx, group[indx], groupdir[indx]):
            refids = fullsample[REFIDCOLUMN][fullsample['GROUP_ID'] == sample['GROUP_ID'][iobj]].value
            mpargs.append((iobj, grp, gdir, refids, datasets, opt_bands))

        if mp > 1:
            with multiprocessing.Pool(mp) as P:
                out = P.map(_build_catalog_one, mpargs)
        else:
            out = [build_catalog_one(*mparg) for mparg in mpargs]

        out = list(zip(*out))
        ellipse = vstack(out[0])
        tractor = vstack(out[1])
        if len(ellipse) > 0:
            fitsio.write(chunkfile, ellipse.as_array(), extname='ELLIPSE', clobber=True)
            fitsio.write(chunkfile, tractor.as_array(), extname='TRACTOR')

    # gather the results
    ellipse, tractor = [], []
    for chunkfile in chunkfiles:
        if os.path.isfile(chunkfile):
            ellipse.append(Table(fitsio.read(chunkfile, 'ELLIPSE')))
            tractor.append(Table(fitsio.read(chunkfile, 'TRACTOR')))
            os.remove(chunkfile)
    ellipse = vstack(ellipse)
    tractor = vstack(tractor)
    nobj = len(ellipse)

    dt, unit = get_dt(t0)
    log.info(f'Gathered ellipse measurements for {nobj:,d} unique objects and ' + \
             f'{len(tractor):,d} Tractor sources in {dt:.3f} {unit}.')

    try:
        assert(np.all(np.isin(ellipse[REFIDCOLUMN], tractor['ref_id'])))
    except:
        pdb.set_trace()
    #tractor[np.isin(tractor['ref_id'], ellipse[REFIDCOLUMN])]

    #np.sum(~np.isfinite(ellipse['COG_LNALPHA1_ERR_R']))
    # re-organize the ellipse table to match the datamodel and assign units
    outellipse = SGA_datamodel(ellipse, bands, all_bands)

    # final geometry
    diam, ba, pa, diam_ref = SGA_geometry(outellipse)
    for col, val in zip(['D26', 'BA', 'PA', 'D26_REF'],
                        [diam, ba, pa, diam_ref]):
        outellipse[col] = val

    #I = (outellipse['R26_G'] > 0.) * (outellipse['R26_R'] > 0.) ; np.median(outellipse['R26_R'][I]/outellipse['R26_G'][I])
    outellipse[outellipse['D26'] == 0.]['SGAGROUP', 'OBJNAME', 'R24_R', 'R25_R', 'R26_R', 'D26']
    assert(np.all(outellipse['D26'] > 0.))

    # separate out (and sort) the tractor catalog of the SGA sources
    I = np.where(tractor['ref_cat'] == REFCAT)[0]
    m1, m2 = match(outellipse[REFIDCOLUMN], tractor['ref_id'][I])
    outellipse = outellipse[m1]
    tractor_sga = tractor[I[m2]]

    tractor_nosga = tractor[np.delete(np.arange(len(tractor)), I)]

    # Write out ellipsefile with the ELLIPSE and TRACTOR HDUs.
    hdu_primary = fits.PrimaryHDU()
    hdu_ellipse = fits.convenience.table_to_hdu(outellipse)
    hdu_tractor_sga = fits.convenience.table_to_hdu(tractor_sga)
    hdu_ellipse.header['EXTNAME'] = 'ELLIPSE'
    hdu_tractor_sga.header['EXTNAME'] = 'TRACTOR'
    hx = fits.HDUList([hdu_primary, hdu_ellipse, hdu_tractor_sga])
    hx.writeto(outfile, overwrite=True, checksum=True)
    log.info(f'Wrote {len(outellipse):,d} objects to {outfile}')

    # Write out ellipsefile_ellipse by combining the ellipse and
    # tractor catalogs.
    ellipse_cols = ['RA', 'DEC', 'SGAID', 'MAG_INIT', 'PA', 'BA', 'D26', 'FITMODE']
    tractor_cols = ['type', 'sersic', 'shape_r', 'shape_e1', 'shape_e2', ] + \
        [f'flux_{filt}' for filt in bands]

    out_sga = outellipse[ellipse_cols]
    out_sga.rename_columns(['SGAID', 'D26', 'MAG_INIT'], ['REF_ID', 'DIAM', 'MAG'])
    [out_sga.rename_column(col, col.lower()) for col in out_sga.colnames]

    out_nosga = Table()
    for col in out_sga.colnames:
        out_nosga[col] = np.zeros(len(tractor_nosga), dtype=out_sga[col].dtype)
    out_nosga = hstack((out_nosga, tractor_nosga[tractor_cols]))
    out_nosga['ra'] = tractor_nosga['ra']
    out_nosga['dec'] = tractor_nosga['dec']
    out_nosga['ref_id'] = -1

    out_sga = hstack((out_sga, tractor_sga[tractor_cols]))
    out = vstack((out_sga, out_nosga))

    out['fitmode'] += FITMODE['FREEZE']

    hdu_primary = fits.PrimaryHDU()
    hdu_out = fits.convenience.table_to_hdu(out)
    hdu_out.header['EXTNAME'] = 'SGA2025'
    hdu_out.header['VER'] = REFCAT
    hx = fits.HDUList([hdu_primary, hdu_out])
    hx.writeto(outfile_ellipse, overwrite=True, checksum=True)
    log.info(f'Wrote {len(out):,d} objects to {outfile_ellipse}')

    # KD version
    cmd1 = f'startree -i {outfile_ellipse} -o {kdoutfile_ellipse} -T -P -k -n stars'
    cmd2 = f'modhead {kdoutfile_ellipse} VER {REFCAT}-ellipse'
    _ = os.system(cmd1)
    _ = os.system(cmd2)
    log.info(f'Wrote {len(out):,d} objects to {kdoutfile_ellipse}')

    print('NB: When combining north-south catalogs, need to look at OBJNAME; SGANAME may not be the same!')


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
                log.warning(f'No good measurements of the PSF size in band {filt}!')
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
                log.warning(f'No good measurements of the PSF depth in band {filt}!')
                #data[psfdepthcol] = np.float32(0.0)
            else:
                psfdepth = tractor.get(psfdepthcol)[these][good] # [AB mag, 5-sigma]
                #data[psfdepthcol] = (22.5-2.5*np.log10(1./np.sqrt(np.median(psfdepth)))).astype('f4')
                sample[psfdepthcol.upper()] = (22.5-2.5*np.log10(1./np.sqrt(np.median(psfdepth)))).astype('f4')

            tractor.delete_column(psfdepthcol)

    return sample


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


def qa_multiband_mask(data, sample, htmlgalaxydir):
    """Diagnostic QA for the output of build_multiband_mask.

    """
    import matplotlib.pyplot as plt
    from matplotlib.cm import get_cmap
    import matplotlib.gridspec as gridspec
    from matplotlib.patches import Patch

    from SGA.util import var2ivar
    from SGA.sky import map_bxby
    from SGA.qa import overplot_ellipse, get_norm, matched_norm


    if not os.path.isdir(htmlgalaxydir):
        os.makedirs(htmlgalaxydir, exist_ok=True)
    qafile = os.path.join(htmlgalaxydir, f'qa-ellipsemask-{data["galaxy"]}.png')

    alpha = 0.6
    orange = (0.9, 0.6, 0.0, alpha)   # golden-orange
    blue   = (0.0, 0.45, 0.7, alpha)  # muted blue
    purple = (0.8, 0.6, 0.7, alpha)   # soft violet
    magenta = (0.85, 0.2, 0.5, alpha) # vibrant rose

    nsample = len(sample)

    opt_bands = data['opt_bands']
    opt_images = data['opt_images']
    opt_maskbits = data['opt_maskbits']
    opt_models = data['opt_models']
    opt_invvar = data['opt_invvar']
    #opt_invvar = var2ivar(data['opt_sigma'], sigma=True)
    opt_pixscale = data['opt_pixscale']

    opt_wcs = data['opt_wcs']
    unwise_wcs = data['unwise_wcs']
    galex_wcs = data['galex_wcs']

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

    GEOINITCOLS = ['BX_INIT', 'BY_INIT', 'SMA_INIT', 'BA_INIT', 'PA_INIT']
    GEOFINALCOLS = ['BX', 'BY', 'SMA_MOMENT', 'BA_MOMENT', 'PA_MOMENT']

    # coadded optical, IR, and UV images and initial geometry
    imgbands = [opt_bands, data['unwise_bands'], data['galex_bands']]
    labels = [''.join(opt_bands), 'unWISE', 'GALEX']
    for iax, (xx, bands, label, wcs, pixscale, refband) in enumerate(zip(
            ax[0, :], imgbands, labels,
            [opt_wcs, unwise_wcs, galex_wcs],
            [data['opt_pixscale'], data['unwise_pixscale'], data['galex_pixscale']],
            [data['opt_refband'], data['unwise_refband'], data['galex_refband']])):
        wimgs = np.stack([data[filt] for filt in bands])
        wivars = np.stack([data[f'{filt}_invvar'] for filt in bands])
        wimg = np.sum(wivars * wimgs, axis=0)
        wnorm = np.sum(wivars, axis=0)
        wimg[wnorm > 0.] /= wnorm[wnorm > 0.]

        try:
            norm = get_norm(wimg)
        except:
            norm = None
        xx.imshow(wimg, origin='lower', cmap=cmap, interpolation='none',
                  norm=norm, alpha=1.)
        xx.set_xlim(0, wimg.shape[0]-1)
        xx.set_ylim(0, wimg.shape[1]-1)
        xx.margins(0)

        # initial ellipse geometry
        #pixfactor = data['opt_pixscale'] / pixscale
        for iobj, obj in enumerate(sample):
            [bx, by, sma, ba, pa] = list(obj[GEOINITCOLS].values())
            bx, by = map_bxby(bx, by, from_wcs=opt_wcs, to_wcs=wcs)
            overplot_ellipse(2*sma, ba, pa, bx, by, pixscale=pixscale, ax=xx,
                             color=colors1[iobj], linestyle='-', linewidth=2,
                             draw_majorminor_axes=True, jpeg=False,
                             label=obj[REFIDCOLUMN])

        xx.text(0.03, 0.97, label, transform=xx.transAxes,
                ha='left', va='top', color='white',
                linespacing=1.5, fontsize=8,
                bbox=dict(boxstyle='round', facecolor='k', alpha=0.5))

        if iax == 0:
            xx.legend(loc='lower left', fontsize=8, ncol=2,
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

        wimg = np.sum(opt_invvar * np.logical_not(opt_masks_obj) * opt_images[iobj, :, :], axis=0)
        wnorm = np.sum(opt_invvar * np.logical_not(opt_masks_obj), axis=0)
        wimg[wnorm > 0.] /= wnorm[wnorm > 0.]
        wimg[wimg == 0.] = np.nan

        wmodel = np.sum(opt_invvar * opt_models[iobj, :, :, :], axis=0)
        wnorm = np.sum(opt_invvar, axis=0)
        wmodel[wnorm > 0.] /= wnorm[wnorm > 0.] / pixscale**2 # [nanomaggies/arcsec**2]

        try:
            #norm = matched_norm(wimg, wmodel)
            norm = get_norm(wimg)
        except:
            norm = None
        ax[1+iobj, 0].imshow(wimg, cmap=cmap, origin='lower', interpolation='none',
                             norm=norm)
        norm = get_norm(wmodel)
        ax[1+iobj, 1].imshow(wmodel, cmap=cmap, origin='lower', interpolation='none',
                             norm=norm)
        #ax[1+iobj, 1].scatter(allgalsrcs.bx, allgalsrcs.by, color='red', marker='s')
        #pdb.set_trace()
        #fig, xx = plt.subplots(1, 2, sharex=True, sharey=True)
        #xx[0].imshow(wimg, origin='lower', norm=norm)
        #wnorm = get_norm(wmodel)
        #wnorm.vmin = norm.vmin
        #wnorm.vmax = norm.vmax
        #xx[1].imshow(wmodel, origin='lower', norm=wnorm)
        #fig.savefig('ioannis/tmp/junk.png')

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
            ax[1+iobj, 2].legend(handles=leg, loc='lower right', fontsize=8)

        for col in range(3):
            # initial geometry
            [bx, by, sma, ba, pa] = list(obj[GEOINITCOLS].values())
            overplot_ellipse(2*sma, ba, pa, bx, by, pixscale=opt_pixscale,
                             ax=ax[1+iobj, col], color=colors2[0], linestyle='-',
                             linewidth=2, draw_majorminor_axes=True,
                             jpeg=False, label='Initial')

            # final geometry
            [bx, by, sma, ba, pa] = list(obj[GEOFINALCOLS].values())
            overplot_ellipse(2*sma, ba, pa, bx, by, pixscale=opt_pixscale,
                             ax=ax[1+iobj, col], color=colors2[1], linestyle='--',
                             linewidth=2, draw_majorminor_axes=True,
                             jpeg=False, label='Final')
            ax[1+iobj, col].set_xlim(0, width-1)
            ax[1+iobj, col].set_ylim(0, width-1)
            ax[1+iobj, col].margins(0)

        ax[1+iobj, 0].text(0.03, 0.97, f'{obj["OBJNAME"]} ({obj[REFIDCOLUMN]})',
                           transform=ax[1+iobj, 0].transAxes,
                           ha='left', va='top', color='white',
                           linespacing=1.5, fontsize=8,
                           bbox=dict(boxstyle='round', facecolor='k', alpha=0.5))

        if iobj == 0:
            ax[1+iobj, 1].text(0.03, 0.97, f'{"".join(opt_bands)} models',
                               transform=ax[1+iobj, 1].transAxes, ha='left', va='top',
                               color='white', linespacing=1.5, fontsize=8,
                               bbox=dict(boxstyle='round', facecolor='k', alpha=0.5))
            ax[1+iobj, 2].text(0.03, 0.97, f'{"".join(opt_bands)} masks',
                               transform=ax[1+iobj, 2].transAxes, ha='left', va='top',
                               color='white', linespacing=1.5, fontsize=8,
                               bbox=dict(boxstyle='round', facecolor='k', alpha=0.5))

        ax[1+iobj, 0].legend(loc='lower left', fontsize=8, fancybox=True,
                             framealpha=0.5)

    for xx in ax.ravel():
        xx.margins(0)
        xx.set_xticks([])
        xx.set_yticks([])

    fig.suptitle(data['galaxy'].replace('_', ' ').replace(' GROUP', ' Group'))
    fig.savefig(qafile)
    plt.close()
    log.info(f'Wrote {qafile}')


def build_multiband_mask(data, tractor, sample, samplesrcs, niter_geometry=2,
                         input_geo_initial=None, qaplot=False,
                         maxshift_arcsec=MAXSHIFT_ARCSEC, cleanup=True,
                         htmlgalaxydir=None):
    """Wrapper to mask out all sources except the galaxy we want to
    ellipse-fit.

    """
    from astrometry.util.starutil_numpy import arcsec_between
    from SGA.geometry import in_ellipse_mask
    from SGA.util import ivar2var, mwdust_transmission
    from SGA.ellipse import ELLIPSEBIT, ELLIPSEMODE


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


    def find_galaxy_in_cutout(img, bx, by, sma, ba, pa, fraction=0.5,
                              factor=2., wmask=None):
        """Measure the light-weighted center and elliptical geometry
        of the object of interest.

        """
        from SGA.geometry import EllipseProperties

        W = img.shape[1]

        x1 = int(bx-factor*sma)
        x2 = int(bx+factor*sma)
        y1 = int(by-factor*sma)
        y2 = int(by+factor*sma)

        if x1 < 0:
            x1 = 0
        if x2 > W:
            x2 = W
        if y1 < 0:
            y1 = 0
        if y2 > W:
            y2 = W

        cutout = img[y1:y2, x1:x2]
        if wmask is not None:
            cutout_mask = wmask[y1:y2, x1:x2]
        else:
            cutout_mask = np.ones(cutout.shape, bool)

        P = EllipseProperties()
        perc = 0.95 # 0.975
        method = 'percentile'
        P.fit(cutout, mask=cutout_mask, method=method, percentile=perc, smooth_sigma=0.)

        if False:
            import matplotlib.pyplot as plt
            fig, (ax1, ax2) = plt.subplots(1, 2)
            P.plot(image=np.log10(cutout), ax=ax1)
            ax2.imshow(cutout_mask, origin='lower')
            fig.savefig('ioannis/tmp/junk.png')
            plt.close()

        if P.a <= 0.:
            log.warning('Reverting to input geometry; moment-derived ' + \
                        'semi-major axis is zero!')
            P.bx = bx
            P.by = by
            P.a = sma # [pixels]
            P.ba = ba
            P.pa = pa
        else:
            P.x0 += x1
            P.y0 += y1

        return P


    def get_geometry(pixscale, pixfactor=1., table=None, tractor=None,
                     props=None):
        """Extract elliptical geometry from either an astropy Table
        (sample), a tractor catalog, or an ellipse_properties object.

        """
        if table is not None:
            bx, by = table['BX_INIT'], table['BY_INIT']
            sma = table['SMA_INIT'] / pixscale # [pixels]
            ba = table['BA_INIT']
            pa = table['PA_INIT']
        elif tractor is not None:
            from SGA.geometry import get_tractor_ellipse
            (bx, by) = tractor.bx, tractor.by
            sma = tractor.shape_r / pixscale # [pixels]
            _, ba, pa = get_tractor_ellipse(sma, tractor.shape_e1, tractor.shape_e2)
        elif props is not None:
            bx = props.x0
            by = props.y0
            sma = props.a # semimajor [pixels]
            ba = props.ba
            pa = props.pa

        bx *= pixfactor
        by *= pixfactor

        return np.array([bx, by, sma, ba, pa])


    def update_galmask(allgalsrcs, bx, by, sma, ba, pa, opt_skysigmas=None,
                       opt_models=None, mask_allgals=False):
        """Update the galaxy mask based on the current in-ellipse array.

        """
        sigma = None
        opt_galmask = np.zeros(sz, bool)
        if len(allgalsrcs) > 0:
            # optionally subtract all galaxies, irrespective of their location
            if mask_allgals:
                I = np.zeros(len(allgalsrcs), bool)
            else:
                I = in_ellipse_mask(bx, width-by, sma, sma*ba, pa,
                                    allgalsrcs.bx, width-allgalsrcs.by)
            if np.sum(~I) > 0:
                galsrcs = allgalsrcs[~I]
                for iband, filt in enumerate(opt_bands):
                    if opt_skysigmas is not None:
                        sigma = opt_skysigmas[filt]
                    msk, model = make_sourcemask(
                        galsrcs, opt_wcs, filt, data[f'{filt}_psf'],
                        sigma=sigma)
                    opt_galmask = np.logical_or(opt_galmask, msk)
                    if opt_models is not None:
                        opt_models[iband, :, :] += model
            else:
                galsrcs = None
        else:
            galsrcs = None

        return galsrcs, opt_galmask, opt_models


    nsample = len(sample)

    all_data_bands = data['all_data_bands']
    opt_bands = data['opt_bands']
    opt_refband = data['opt_refband']
    opt_pixscale = data['opt_pixscale']
    opt_wcs = data['opt_wcs']
    REFIDCOLUMN = data['REFIDCOLUMN']

    sz = data[opt_refband].shape
    width = sz[0]
    assert(width == sz[1])
    xgrid, ygrid = np.meshgrid(np.arange(width),
                               np.arange(width),
                               indexing='xy')
    ygrid_flip = width - ygrid

    opt_skysigmas = {}
    for filt in opt_bands:
        #opt_skysigmas[filt] = 1e9*10.**(-0.4*sample[f'PSFDEPTH_{filt.upper()}'][0])
        opt_skysigmas[filt] = data[f'{filt}_skysigma']

    Ipsf = ((tractor.type == 'PSF') * (tractor.type != 'DUP') *
            (tractor.ref_cat != REFCAT) * (tractor.ref_cat != 'LG') *
            np.logical_or(tractor.ref_cat == 'GE', tractor.ref_cat == 'G3'))
    Igal = ((tractor.type != 'PSF') * (tractor.type != 'DUP') *
            (tractor.ref_cat != REFCAT) * (tractor.ref_cat != 'LG'))
    psfsrcs = tractor[Ipsf]
    allgalsrcs = tractor[Igal]

    #import matplotlib.pyplot as plt
    #from SGA.coadds import srcs2image
    #from SGA.qa import get_norm
    #ncols = 4
    #nrows = int(np.ceil(len(allgalsrcs) / ncols))
    #norm = get_norm(data['r'])
    #fig, ax = plt.subplots(nrows, ncols, sharex=True, sharey=True)
    #for xx, src in zip(ax.flat, allgalsrcs):
    #    model = srcs2image(src, opt_wcs, band='r', pixelized_psf=data['r_psf'])
    #    xx.imshow(np.log10(model), origin='lower', cmap=plt.cm.cividis)#, norm=norm)
    #for xx in ax.flat:
    #    xx.axis('off')
    #fig.tight_layout()
    #fig.savefig('ioannis/tmp/junk.png')

    # Initialize the *original* images arrays.
    opt_images = np.zeros((len(opt_bands), *sz), 'f4')
    opt_mask_perband = np.stack([data[f'{filt}_mask'] for filt in opt_bands])
    opt_invvar = np.stack([data[f'{filt}_invvar'] for filt in opt_bands])

    opt_maskbits = np.zeros((nsample, *sz), np.int32)
    opt_images_final = np.zeros((nsample, len(opt_bands), *sz), 'f4')
    opt_models = np.zeros((nsample, len(opt_bands), *sz), 'f4')

    # Bright-star mask.
    opt_brightstarmask = data['brightstarmask']
    #opt_brightstarmask[:, :] = False

    # Subtract Gaia stars from all optical images and generate the
    # threshold gaiamask (which will be used unless the LESSMASKING
    # bit is set).
    opt_gaiamask = np.zeros(sz, bool)
    for iband, filt in enumerate(opt_bands):
        if len(psfsrcs) > 0:
            msk, model = make_sourcemask(
                psfsrcs, opt_wcs, filt, data[f'{filt}_psf'],
                data[f'{filt}_skysigma'])
            opt_models[:, iband, :, :] += model[np.newaxis, :, :]
            opt_images[iband, :, :] = data[filt] - model
            opt_gaiamask = np.logical_or(opt_gaiamask, msk)
        else:
            opt_images[iband, :, :] = data[filt]

    geo_initial = np.zeros((nsample, 5)) # [bx,by,sma,ba,pa]
    geo_final = np.zeros_like(geo_initial)

    for iobj, (obj, objsrc) in enumerate(zip(sample, samplesrcs)):
        log.info('Determining the geometry for galaxy ' + \
                 f'{iobj+1}/{nsample}.')

        # If the LESSMASKING bit is set, do not use the Gaia threshold
        # mask.
        opt_gaiamask_obj = np.copy(opt_gaiamask)
        if obj['ELLIPSEMODE'] & ELLIPSEMODE['LESSMASKING'] != 0:# or True:
            log.info('LESSMASKING bit set; no Gaia threshold-masking.')
            opt_gaiamask_obj[:, :] = False

        # If the MOREMASKING bit is set, mask all extended sources,
        # whether or not they're inside the elliptical mask.
        if obj['ELLIPSEMODE'] & ELLIPSEMODE['MOREMASKING'] != 0:# or True:
            log.info('MOREMASKING bit set; masking all extended sources.')
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
            if indx < iobj:
                [bx, by, sma, ba, pa] = geo_final[indx, :]
            else:
                [bx, by, sma, ba, pa] = \
                    get_geometry(opt_pixscale, table=refsample)
            opt_refmask1 = in_ellipse_mask(bx, width-by, sma, ba*sma,
                                           pa, xgrid, ygrid_flip)
            opt_refmask = np.logical_or(opt_refmask, opt_refmask1)

            for iband, filt in enumerate(opt_bands):
                _, model = make_sourcemask(
                    refsrc, opt_wcs, filt, data[f'{filt}_psf'],
                    sigma=None)
                opt_images_obj[iband, :, :] = opt_images_obj[iband, :, :] - model
                opt_models[iobj, iband, :, :] += model

        # Initial geometry and elliptical mask.
        if input_geo_initial is not None:
            geo_init = input_geo_initial[iobj, :]
        else:
            geo_init = get_geometry(opt_pixscale, table=obj)
        geo_initial[iobj, :] = geo_init
        [bx, by, sma, ba, pa] = geo_init
        print(iobj, bx, by, sma, ba, pa)

        #print('HACK!')
        #obj['ELLIPSEMODE'] += 2**0

        # Next, iteratively update the source geometry unless
        # FIXGEO has been set.
        if obj['ELLIPSEMODE'] & ELLIPSEMODE['FIXGEO'] != 0:
            niter_actual = 1
        else:
            niter_actual = niter_geometry

        for iiter in range(niter_actual):
            log.debug(f'Iteration {iiter+1}/{niter_actual}')
            #print(iobj, iiter, bx, by, sma, ba, pa)

            # initialize (or update) the in-ellipse mask
            inellipse = in_ellipse_mask(bx, width-by, sma, ba*sma,
                                        pa, xgrid, ygrid_flip)

            # Zero out bright-star and reference pixels within the
            # current ellipse mask of the current object...
            iter_brightstarmask = np.copy(opt_brightstarmask)
            iter_refmask = np.copy(opt_refmask)
            iter_brightstarmask[inellipse] = False
            iter_refmask[inellipse] = False

            # Expand the brightstarmask veto if the NEARSTAR or GCLPNE
            # bits are set (factor of 2).
            if (obj['SAMPLE'] & SAMPLE['NEARSTAR'] != 0 or \
                obj['SAMPLE'] & SAMPLE['GCLPNE'] != 0):
                inellipse2 = in_ellipse_mask(bx, width-by, 2.*sma, 2.*sma*ba,
                                             pa, xgrid, ygrid_flip)
                iter_brightstarmask[inellipse2] = False

            #iter_gaiamask = np.copy(opt_gaiamask_obj)
            #iter_gaiamask[inellipse] = False
            #import matplotlib.pyplot as plt
            #plt.clf()
            #plt.imshow(opt_models[0, 0, :, :], origin='lower')
            #plt.savefig('ioannis/tmp/junk2.png')

            # Build a galaxy mask from all extended sources outside
            # the (current) elliptical mask (but do not subtract the
            # models). By default, galaxy pixels inside the elliptical
            # mask, unless we're masking *all* galaxies (e.g., in
            # cluster fields).
            galsrcs, opt_galmask, _ = update_galmask(
                allgalsrcs, bx, by, sma, ba, pa,
                opt_skysigmas=opt_skysigmas, opt_models=None,
                mask_allgals=mask_allgals)
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
                inellipse2 = in_ellipse_mask(bx, width-by, 4.*sma, 4.*sma*ba,
                                             pa, xgrid, ygrid_flip)
                iter_refmask[inellipse2] = False
            #import matplotlib.pyplot as plt
            #plt.clf()
            #plt.imshow(iter_refmask, origin='lower')
            #plt.savefig('ioannis/tmp/junk2.png')

            # Combine opt_brightstarmask, opt_gaiamask, opt_refmask,
            # and opt_galmask with the per-band optical masks.
            #opt_galmask[:] = False
            opt_masks_obj = _update_masks(iter_brightstarmask, opt_gaiamask_obj,
                                          iter_refmask, opt_galmask,
                                          opt_mask_perband, opt_bands,
                                          sz, verbose=False)

            # Optionally update the geometry from the masked, coadded
            # optical image.
            if obj['ELLIPSEMODE'] & ELLIPSEMODE['FIXGEO'] != 0:
                log.info('FIXGEO bit set; fixing the elliptical geometry.')
                geo_iter = geo_init
            else:
                # generate a detection image and pixel mask for use with find_galaxy_in_cutout
                wimg = np.sum(opt_invvar * np.logical_not(opt_masks_obj) * opt_images_obj, axis=0)
                wnorm = np.sum(opt_invvar * np.logical_not(opt_masks_obj), axis=0)
                wimg[wnorm > 0.] /= wnorm[wnorm > 0.]

                wmasks = np.zeros_like(opt_images_obj, bool)
                for iband, filt in enumerate(opt_bands):
                    wmasks[iband, :, :] = (~opt_masks_obj[iband, :, :]) * (opt_images_obj[iband, :, :] > opt_skysigmas[filt])
                # True=any pixel is >5*skynoise and positive in the
                # coadded image (otherwise the ellipse_moments matrix
                # can become ill-defined).
                wmask = np.any(wmasks, axis=0) * (wimg > 0.)

                props = find_galaxy_in_cutout(wimg, bx, by, sma, ba, pa, wmask=wmask)
                geo_iter = get_geometry(opt_pixscale, props=props)

            ra_iter, dec_iter = opt_wcs.wcs.pixelxy2radec(geo_iter[0]+1., geo_iter[1]+1.)
            dshift_arcsec = arcsec_between(obj['RA_INIT'], obj['DEC_INIT'], ra_iter, dec_iter)
            if dshift_arcsec > maxshift_arcsec:
                log.warning(f'Large shift for iobj={iobj} ({obj[REFIDCOLUMN]}): delta=' + \
                            f'{dshift_arcsec:.3f}>{maxshift_arcsec:.3f} arcsec')
                # revert to the Tractor position or the initial position
                if objsrc is not None:
                    geo_iter[0] = objsrc.bx
                    geo_iter[1] = objsrc.by
                else:
                    geo_iter[0] = geo_init[0]
                    geo_iter[1] = geo_init[1]

            # update the geometry for the next iteration
            [bx, by, sma, ba, pa] = geo_iter
            print(iobj, iiter, bx, by, sma, ba, pa)

        # set the largeshift bits
        ra_final, dec_iter = opt_wcs.wcs.pixelxy2radec(geo_iter[0]+1., geo_iter[1]+1.)
        dshift_arcsec = arcsec_between(obj['RA_INIT'], obj['DEC_INIT'], ra_iter, dec_iter)
        if dshift_arcsec > maxshift_arcsec:
            sample['ELLIPSEBIT'][iobj] += ELLIPSEBIT['LARGESHIFT']
        #sample['DSHIFT'][iobj] = dshift_arcsec

        # Was there a large shift between the Tractor and final position?
        if objsrc is not None:
            dshift_tractor_arcsec = arcsec_between(objsrc.ra, objsrc.dec, ra_iter, dec_iter)
            if dshift_tractor_arcsec > maxshift_arcsec:
                sample['ELLIPSEBIT'][iobj] += ELLIPSEBIT['LARGESHIFT_TRACTOR']
            #sample['DSHIFT_TRACTOR'][iobj] = dshift_tractor_arcsec

        # final images and geometry
        opt_images_final[iobj, :, :, :] = opt_images_obj
        geo_final[iobj, :] = geo_iter # last iteration

        # final masks
        inellipse = in_ellipse_mask(bx, width-by, sma, sma*ba,
                                    pa, xgrid, ygrid_flip)
        final_brightstarmask = np.copy(opt_brightstarmask)
        final_refmask = np.copy(opt_refmask)
        final_brightstarmask[inellipse] = False
        final_refmask[inellipse] = False

        if (sample['SAMPLE'][iobj] & SAMPLE['NEARSTAR'] != 0 or \
            sample['SAMPLE'][iobj] & SAMPLE['GCLPNE'] != 0):
            inellipse2 = in_ellipse_mask(bx, width-by, 2.*sma, 2.*sma*ba,
                                         pa, xgrid, ygrid_flip)
            final_brightstarmask[inellipse2] = False

        _, opt_galmask, opt_models_obj = update_galmask(
            allgalsrcs, bx, by, sma, ba, pa,
            opt_models=opt_models[iobj, :, :, :],
            opt_skysigmas=opt_skysigmas,
            mask_allgals=mask_allgals)
        if not mask_allgals:
            opt_galmask[inellipse] = False
        #opt_galmask[:] = False

        #import matplotlib.pyplot as plt
        #plt.clf()
        #plt.imshow(opt_gaiamask_obj, origin='lower')
        ##plt.imshow(final_refmask, origin='lower')
        #plt.savefig('ioannis/tmp/junk2.png')

        opt_maskbits_obj = _update_masks(
            final_brightstarmask, opt_gaiamask_obj,
            final_refmask, opt_galmask, opt_mask_perband,
            opt_bands, sz, build_maskbits=True,
            MASKDICT=OPTMASKBITS)
        opt_models[iobj, :, :, :] = opt_models_obj
        opt_maskbits[iobj, :, :] = opt_maskbits_obj

        #import matplotlib.pyplot as plt
        #plt.clf()
        #plt.imshow(np.log10(opt_images_final[iobj, 0, :, :]*(opt_maskbits[iobj, :, :]==0)), origin='lower')
        #plt.savefig('ioannis/tmp/junk.png')
        #plt.close()

    # Loop through all objects again to set the blended bit.
    for iobj, obj in enumerate(sample):
        refindx = np.delete(np.arange(nsample), iobj)
        for indx in refindx:
            [bx, by, _, _, _] = geo_final[iobj, :]
            [refbx, refby, refsma, refba, refpa] = geo_final[indx, :]
            Iclose = in_ellipse_mask(refbx, width-refby, refsma,
                                     refsma*refba, refpa, bx, width-by)
            if Iclose:
                sample['ELLIPSEBIT'][iobj] += ELLIPSEBIT['BLENDED']

    # Update the data dictionary.
    data['opt_images'] = opt_images_final # [nanomaggies]
    data['opt_maskbits'] = opt_maskbits
    data['opt_models'] = opt_models # [nanomaggies]
    data['opt_invvar'] = opt_invvar
    sig, _ = ivar2var(opt_invvar, sigma=True) # [nanomaggies]
    data['opt_sigma'] = sig

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
        models = np.zeros((nsample, len(bands), *sz), 'f4')

        # Subtract Gaia stars.
        for iband, filt in enumerate(bands):
            if len(psfsrcs) > 0:
                _, model = make_sourcemask(
                    psfsrcs, wcs, filt, data[f'{filt}_psf'],
                    sigma=None)
                models[:, iband, :, :] += model[np.newaxis, :, :]
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
                        models[iobj, iband, :, :] += model
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
        data[f'{prefix}_models'] = models
        ivar = np.stack([data[f'{filt}_invvar'] for filt in bands])
        sig, _ = ivar2var(ivar, sigma=True, allmasked_ok=True) # [nanomaggies]
        data[f'{prefix}_sigma'] = sig


    # convert to surface brightness
    for prefix in ['opt', 'unwise', 'galex']:
        pixscale = data[f'{prefix}_pixscale']
        data[f'{prefix}_images'] /= pixscale**2 # [nanomaggies/arcsec**2]
        data[f'{prefix}_sigma'] /= pixscale**2  # [nanomaggies/arcsec**2]

    # final geometry
    ra, dec = opt_wcs.wcs.pixelxy2radec((geo_final[:, 0]+1.), (geo_final[:, 1]+1.))
    for icol, col in enumerate(['BX', 'BY', 'SMA_MOMENT', 'BA_MOMENT', 'PA_MOMENT']):
        sample[col] = geo_final[:, icol].astype('f4')
    sample['SMA_MOMENT'] *= opt_pixscale # [pixels-->arcsec]

    sample['RA'] = ra
    sample['DEC'] = dec
    sample['SGANAME'] = sga2025_name(ra, dec)
    #sample['RA_MOMENT'] = ra
    #sample['DEC_MOMENT'] = dec
    for filt in data['all_opt_bands']: # NB: all optical bands
        sample[f'MW_TRANSMISSION_{filt.upper()}'] = mwdust_transmission(
            sample['EBV'], band=filt, run=data['run'])

    # optionally build a QA figure
    if qaplot:
        #sample['BY_INIT'] += 30.
        qa_multiband_mask(data, sample, htmlgalaxydir=htmlgalaxydir)

    # clean-up
    if cleanup:
        del data['brightstarmask']
        for filt in all_data_bands:
            del data[filt]
            #for col in ['psf', 'invvar', 'mask']:
            for col in ['psf', 'mask']:
                del data[f'{filt}_{col}']
        for filt in opt_bands:
            for col in ['skysigma']:
                del data[f'{filt}_{col}']

    return data, sample


def read_multiband(galaxy, galaxydir, REFIDCOLUMN, bands=['g', 'r', 'i', 'z'],
                   sort_by_flux=True, run='south', niter_geometry=2,
                   pixscale=0.262, galex_pixscale=1.5, unwise_pixscale=2.75,
                   galex=True, unwise=True, verbose=False, qaplot=False,
                   cleanup=False, htmlgalaxydir=None):
    """Read the multi-band images (converted to surface brightness) in
    preparation for ellipse-fitting.

    """
    import fitsio
    from astropy.table import Table
    from astrometry.util.fits import fits_table
    from astrometry.util.util import Tan
    from legacypipe.bits import MASKBITS

    from SGA.io import _read_image_data
    from SGA.ellipse import ELLIPSEBIT

    # Dictionary mapping between optical filter and filename coded up in
    # coadds.py, galex.py, and unwise.py, which depends on the project.
    data = {}
    data['galaxy'] = galaxy
    data['galaxydir'] = galaxydir
    data['run'] = run
    data['REFIDCOLUMN'] = REFIDCOLUMN
    data['all_opt_bands'] = bands

    all_opt_bands = bands # initialize
    all_bands = np.copy(bands)

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
                return {}, 0

    all_data_bands = opt_bands
    if unwise:
        all_data_bands = np.hstack((all_data_bands, unwise_bands))
    if galex:
        all_data_bands = np.hstack((all_data_bands, galex_bands))
    ## update all_bands
    #all_bands = opt_bands
    #if unwise:
    #    all_bands = np.hstack((all_bands, unwise_bands))
    #if galex:
    #    all_bands = np.hstack((all_bands, galex_bands))
    log.info(f'Found complete data in bands: {",".join(all_data_bands)}')

    # Pack some preliminary info into the output dictionary.
    data['all_bands'] = all_bands
    data['all_data_bands'] = all_data_bands
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
    #cols += [f'mw_transmission_{filt}' for filt in all_opt_bands]
    cols += [f'psfdepth_{filt}' for filt in all_opt_bands] # NB: all optical bands
    cols += [f'psfsize_{filt}' for filt in all_opt_bands]
    if galex:
        cols += [f'flux_{filt.lower()}' for filt in galex_bands]
        cols += [f'flux_ivar_{filt.lower()}' for filt in galex_bands]
        cols += [f'psfdepth_{filt.lower()}' for filt in galex_bands]
    if unwise:
        cols += [f'flux_{filt.lower()}' for filt in unwise_bands]
        cols += [f'flux_ivar_{filt.lower()}' for filt in unwise_bands]
        cols += [f'psfdepth_{filt.lower()}' for filt in unwise_bands]

    prim = fitsio.read(tractorfile, columns='brick_primary')
    tractor = fits_table(tractorfile, rows=np.where(prim)[0], columns=cols)
    log.info(f'Read {len(tractor):,d} brick_primary sources from {tractorfile}')

    # Read the sample catalog from custom_coadds and find each source
    # in the Tractor catalog.
    samplefile = os.path.join(galaxydir, f'{galaxy}-{filt2imfile["sample"]}.fits')
    sample = Table(fitsio.read(samplefile))#, columns=cols))
    log.info(f'Read {len(sample)} source(s) from {samplefile}')
    for col in ['RA', 'DEC', 'DIAM', 'PA', 'BA', 'MAG']:
        sample.rename_column(col, f'{col}_INIT')
    sample.rename_column('DIAM_REF', 'DIAM_INIT_REF')
    sample.add_column(sample['DIAM_INIT']*60./2., name='SMA_INIT', # [radius, arcsec]
                      index=np.where(np.array(sample.colnames) == 'DIAM_INIT')[0][0])

    #print('HACK!!!!!!!!')
    #sample['SMA_INIT'] = 179.8639
    #sample['DIAM_INIT'] = 179.8639*2./60.

    # populate (BX,BY)_INIT by quickly building the WCS
    wcs = Tan(filt2imfile[opt_refband]['image'], 1)
    (_, x0, y0) = wcs.radec2pixelxy(sample['RA_INIT'].value, sample['DEC_INIT'].value)

    sample.add_column((x0 - 1.).astype('f4'), name='BX_INIT',
                      index=np.where(np.array(sample.colnames) == 'FITMODE')[0][0]+1)  # NB the -1!
    sample.add_column((y0 - 1.).astype('f4'), name='BY_INIT',
                      index=np.where(np.array(sample.colnames) == 'BX_INIT')[0][0]+1)  # NB the -1!
    #sample['BY_INIT'] = (y0 - 1.).astype('f4')

    sample['FLUX'] = np.zeros(len(sample), 'f4') # brightest band

    # optical bands
    sample['BANDS'] = np.zeros(len(sample), f'<U{len(bands)}')
    sample['BANDS'] = ''.join(data['opt_bands'])

    # moment geometry
    sample['SGANAME'] = np.zeros(len(sample), '<U25')
    sample['RA'] = np.zeros(len(sample), 'f8')
    sample['DEC'] = np.zeros(len(sample), 'f8')
    sample['BX'] = np.zeros(len(sample), 'f4')
    sample['BY'] = np.zeros(len(sample), 'f4')
    sample['SMA_MOMENT'] = np.zeros(len(sample), 'f4') # [arcsec]
    sample['BA_MOMENT'] = np.zeros(len(sample), 'f4')
    sample['PA_MOMENT'] = np.zeros(len(sample), 'f4')
    sample['RA_TRACTOR'] = np.zeros(len(sample), 'f8')
    sample['DEC_TRACTOR'] = np.zeros(len(sample), 'f8')

    # initialize the ELLIPSEBIT bitmask
    sample['ELLIPSEBIT'] = np.zeros(len(sample), np.int32)

    samplesrcs = []
    for iobj, refid in enumerate(sample[REFIDCOLUMN].value):
        I = np.where(np.logical_or(tractor.ref_cat == REFCAT, tractor.ref_cat == 'LG') *
                     (tractor.ref_id == refid))[0]
        if len(I) == 0:
            log.warning(f'ref_id={refid} dropped by Tractor')
            sample['ELLIPSEBIT'][iobj] += ELLIPSEBIT['NOTRACTOR']
            samplesrcs.append(None)
        else:
            samplesrcs.append(tractor[I])
            if tractor[I[0]].type in ['PSF', 'DUP']:
                log.warning(f'ref_id={refid} fit by Tractor as PSF (or DUP)')
                #sample['PSF'][iobj] = True
            sample['FLUX'][iobj] = max([getattr(tractor[I[0]], f'flux_{filt}')
                                        for filt in opt_bands])
            sample['RA_TRACTOR'][iobj] = tractor[I[0]].ra
            sample['DEC_TRACTOR'][iobj] = tractor[I[0]].dec

    # Sort by initial diameter or optical brightness (in any band).
    if sort_by_flux:
        log.info('Sorting by optical flux:')
        srt = np.argsort(sample['FLUX'])[::-1]
    else:
        log.info('Sorting by initial diameter:')
        srt = np.argsort(sample['SMA_INIT'])[::-1]

    sample = sample[srt]
    samplesrcs = [samplesrcs[I] for I in srt]
    for obj in sample:
        log.info(f'  ref_id={obj[REFIDCOLUMN]}: D(25)={obj["DIAM_INIT"]:.3f} arcmin, ' + \
                 f'max optical flux={obj["FLUX"]:.2f} nanomaggies')
    sample.remove_column('FLUX')

    # PSF size and depth
    for filt in all_opt_bands:
        sample[f'PSFSIZE_{filt.upper()}'] = np.zeros(len(sample), 'f4')
    for filt in all_bands:
        sample[f'PSFDEPTH_{filt.upper()}'] = np.zeros(len(sample), 'f4')

    # add the PSF depth and size
    _get_psfsize_and_depth(sample, tractor, all_data_bands,
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
    # masks.
    #print('HACK!!')
    #niter = 1

    data = _read_image_data(data, filt2imfile, verbose=verbose)
    data, sample = build_multiband_mask(data, tractor, sample, samplesrcs,
                                        qaplot=qaplot, cleanup=cleanup,
                                        niter_geometry=niter_geometry,
                                        htmlgalaxydir=htmlgalaxydir)

    return data, tractor, sample, samplesrcs, 1


def get_radius_mosaic(diam, multiplicity=1, mindiam=0.5,
                      pixscale=0.262, get_barlen=False):
    """Get the mosaic radius.

    diam, mindiam in arcmin

    """
    if diam < mindiam:
        diam = mindiam # arcmin

    radius_mosaic_arcsec = 60. * diam / 2. # [arcsec]
    if multiplicity == 1:
        if diam > 10.:
            radius_mosaic_arcsec *= 1.
        elif diam > 3. and diam <= 10:
            radius_mosaic_arcsec *= 1.1
        elif diam > 1. and diam <= 3.:
            radius_mosaic_arcsec *= 1.3
        else:
            radius_mosaic_arcsec *= 1.5

    if get_barlen:
        if radius_mosaic_arcsec > 6. * 60.: # [>6] arcmin
            barlabel = '2 arcmin'
            barlen = np.ceil(120. / pixscale).astype(int) # [pixels]
        elif (radius_mosaic_arcsec > 3. * 60.) & (radius_mosaic_arcsec < 6. * 60.): # [3-6] arcmin
            barlabel = '1 arcmin'
            barlen = np.ceil(60. / pixscale).astype(int) # [pixels]
        else:
            barlabel = '30 arcsec'
            barlen = np.ceil(30. / pixscale).astype(int) # [pixels]
        return radius_mosaic_arcsec, barlen, barlabel
    else:
        return radius_mosaic_arcsec
