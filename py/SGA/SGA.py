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
    GCLPNE = 2**2,   # in a globular cluster or PNe mask
    NEARSTAR = 2**3, # STARFDIST < 1.2
    INSTAR = 2**4,   # STARFDIST < 0.5
    OVERLAP = 2**5,  # initial ellipse overlaps another (SGA) ellipse
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
        ## first major run
        #version = 'v0.10'

        ## no duplicate groups; cleanup of REGION bits; some dropped
        ## sources via VI.
        #version = 'v0.11'

        ## re-initialize diameters with v0.11 ellipse results; drop
        ## sources with no Tractor; VI update of galaxy properties
        #version = 'v0.12'

        # remove D(26)<0.5 sources (and groups where /all/ members
        # have D(26)<0.5) based on v0.11 fitting results; keep
        # diameters at their initial values
        #version = 'v0.20'

        # tons of VI results
        #version = 'v0.21'

        # more VI; D<0.5 arcmin systems in the test region removed; SGA2020 galaxies added
        #version = 'v0.22'

        # major refactor of build_parent
        #version = 'v0.30'

        # significant trimming of small galaxies; numerous new ELLIPSEBIT
        #version = 'v0.40'

        # tons of additional sample cleanup
        #version = 'v0.50'

        # more cleanup
        version = 'v0.60'

        # more cleanup
        version = 'v0.70'
    else:
        # parent-refcat, parent-ellipse, and final SGA2025
        #version = 'v0.10' # parent_version = v0.10
        #version = 'v0.11' # parent_version = v0.10 --> v0.11
        #version = 'v0.12' # parent_version = v0.11 --> v0.12
        #version = 'v0.20' # parent_version = v0.12 --> v0.20
        #version = 'v0.21' # parent_version = v0.20 --> v0.21
        #version = 'v0.22'  # parent_version = v0.21 --> v0.22
        #version = 'v0.30'  # parent_version = v0.22 --> v0.30
        #version = 'v0.40'  # parent_version = v0.30 --> v0.40
        #version = 'v0.50'  # parent_version = v0.40 --> v0.50
        #version = 'v0.60'  # parent_version = v0.50 --> v0.60
        version = 'v0.70'  # parent_version = v0.50 --> v0.60
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
                  clobber=False, clobber_overwrite=None,
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
    elif htmlplots:
        suffix = 'html'
        filesuffix = '-html.isdone'
        dependson = None # '-image.jpg'
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

    # Make clobber=False for htmlindex because we're not making the
    # files here, we're just looking for them. The argument
    # args.clobber gets used downstream.
    if htmlindex:
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
    iwait = np.where(todo == 'wait')[0]

    if len(ifail) > 0:
        fail_indices = [indices[ifail]]
    else:
        fail_indices = [np.array([], int)]

    if len(iwait) > 0:
        wait_indices = [indices[iwait]]
    else:
        wait_indices = [np.array([], int)]

    if len(idone) > 0:
        done_indices = [indices[idone]]
    else:
        done_indices = [np.array([], int)]

    if len(itodo) > 0:
        todo_indices, loads = distribute_work(sample[DIAMCOL].value, itodo=itodo,
                                              size=size, p=2.0, verbose=True)
    else:
        todo_indices = []

    return suffix, todo_indices, done_indices, fail_indices, wait_indices


def read_sample(first=None, last=None, galaxylist=None, verbose=False, columns=None,
                no_groups=False, lvd=False, wisesize=False, final_sample=False,
                version=None, tractor=False, test_bricks=False, region='dr11-south',
                mindiam=0., maxdiam=1e3, minmult=None, maxmult=None, beta=True):
    """Read/generate the parent SGA catalog.

    mindiam,maxdiam in arcmin
    maxmult - maximum number of group members (ignored if --no-groups is set)

    """
    import fitsio

    if first and last:
        if first > last:
            msg = f'Index first cannot be greater than index last, {first} > {last}'
            log.critical(msg)
            raise ValueError(msg)

    if final_sample:
        if tractor:
            ext = 'TRACTOR'
        else:
            ext = 'ELLIPSE'
        if version is None:
            version = SGA_version()
        if beta:
            samplefile = os.path.join(sga_dir(), 'sample', f'SGA2025-beta-{version}-{region}.fits')
        else:
            samplefile = os.path.join(sga_dir(), 'sample', f'SGA2025-{version}-{region}.fits')
    else:
        ext = 'PARENT'
        if version is None:
            version = SGA_version(parent=True)
        if beta:
            samplefile = os.path.join(sga_dir(), 'sample', f'SGA2025-beta-parent-{version}.fits')
        else:
            samplefile = os.path.join(sga_dir(), 'sample', f'SGA2025-parent-{version}.fits')

    if not os.path.isfile(samplefile):
        msg = f'Sample file {samplefile} not found.'
        log.critical(msg)
        raise IOError(msg)

    if final_sample:
        if no_groups:
            cols = ['D26']
        else:
            #cols = ['D26', 'GROUP_PRIMARY']
            cols = ['GROUP_DIAMETER', 'GROUP_PRIMARY']
            if maxmult or minmult:
                cols += ['GROUP_MULT']
        info = fitsio.read(samplefile, ext=ext, columns=cols)
        if no_groups:
            rows = np.where(
                (info['DIAM'] >= mindiam) *
                (info['DIAM'] < maxdiam))[0]
        else:
            I = ((info['GROUP_DIAMETER'] >= mindiam) *
                 (info['GROUP_DIAMETER'] < maxdiam) *
                 info['GROUP_PRIMARY'])
            if minmult:
                I *= info['GROUP_MULT'] >= minmult
            if maxmult:
                I *= info['GROUP_MULT'] <= maxmult
            rows = np.where(I)[0]
    else:
        #cols = ['GROUP_NAME', 'GROUP_RA', 'GROUP_DEC', 'GROUP_DIAMETER', 'GROUP_MULT',
        #        'GROUP_PRIMARY', 'GROUP_ID', 'SGAID', 'RA', 'DEC', 'BRICKNAME']
        if no_groups:
            cols = ['DIAM']
        else:
            cols = ['GROUP_DIAMETER', 'GROUP_PRIMARY']
            if maxmult or minmult:
                cols += ['GROUP_MULT']
        info = fitsio.read(samplefile, ext=ext, columns=cols)
        if no_groups:
            rows = np.where(
                (info['DIAM'] >= mindiam) *
                (info['DIAM'] < maxdiam))[0]
        else:
            I = ((info['GROUP_DIAMETER'] >= mindiam) *
                 (info['GROUP_DIAMETER'] < maxdiam) *
                 info['GROUP_PRIMARY'])
            if minmult:
                I *= info['GROUP_MULT'] >= minmult
            if maxmult:
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

    if False:#True:
        log.info('Refitting!')
        refitfile = os.path.join(sga_dir(), 'sample', 'SGA2025-v0.70-refit.fits')
        refit = Table(fitsio.read(refitfile))
        refit_groups = fullsample['GROUP_NAME'][np.isin(fullsample['OBJNAME'], refit['OBJNAME'])]
        fullsample = fullsample[np.isin(fullsample['GROUP_NAME'], refit_groups)]
        sample = fullsample[fullsample['GROUP_PRIMARY']]


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

    # select the LVD sample
    if lvd:
        from SGA.ellipse import ELLIPSEMODE
        #is_LVD = (fullsample['SAMPLE'] & SAMPLE['LVD'] != 0) & (fullsample['ELLIPSEMODE'] & ELLIPSEMODE['FORCEPSF'] != 0)
        #is_LVD = (fullsample['SAMPLE'] & SAMPLE['LVD'] != 0) & (fullsample['ELLIPSEMODE'] & ELLIPSEMODE['FIXGEO'] == 0) & (fullsample['ELLIPSEMODE'] & ELLIPSEMODE['RESOLVED'] == 0)
        #is_LVD = (fullsample['SAMPLE'] & SAMPLE['LVD'] != 0) & (fullsample['ELLIPSEMODE'] & ELLIPSEMODE['FIXGEO'] != 0) & (fullsample['ELLIPSEMODE'] & ELLIPSEMODE['RESOLVED'] == 0)
        #is_LVD = (fullsample['SAMPLE'] & SAMPLE['LVD'] != 0) & (fullsample['ELLIPSEMODE'] & ELLIPSEMODE['RESOLVED'] != 0)
        is_LVD = fullsample['SAMPLE'] & SAMPLE['LVD'] != 0
        LVD_group_names = np.unique(fullsample['GROUP_NAME'][is_LVD])
        I = np.isin(fullsample['GROUP_NAME'], LVD_group_names)
        fullsample = fullsample[I]
        sample = fullsample[fullsample['GROUP_PRIMARY']]
        if len(sample) == 0:
            return sample, fullsample

    if False:#True:
        print('HACK!!')
        from SGA.ellipse import ELLIPSEMODE, ELLIPSEBIT
        I = fullsample['ELLIPSEBIT'] & ELLIPSEBIT['NOTRACTOR'] != 0
        #I = (fullsample['ELLIPSEBIT'] & ELLIPSEBIT['FAILGEO'] != 0) & (fullsample['SAMPLE'] & SAMPLE['LVD'] == 0)
        J = np.isin(fullsample['GROUP_NAME'], np.unique(fullsample['GROUP_NAME'][I]))
        fullsample = fullsample[J]
        sample = fullsample[fullsample['GROUP_PRIMARY']]

    #if True:
    #    redo = np.unique(Table.read('/global/u2/i/ioannis/rerun.txt', format='ascii')['col1'].value)
    #    fullsample = fullsample[np.isin(fullsample['OBJNAME'], redo)]
    #    sample = fullsample[fullsample['GROUP_PRIMARY']]

    if wisesize:
        from SGA.util import match

        ofullsample = fullsample.copy()

        nobj = len(sample)
        nfullobj = len(fullsample)
        I = ((fullsample['SAMPLE'] == 0) * (fullsample['DIAM'] > 0.75) * (fullsample['DIAM'] < 5) *
             (fullsample['GROUP_DIAMETER'] < 5) * (fullsample['GROUP_RA'] > 87.) * (fullsample['GROUP_RA'] < 300.) *
             (fullsample['GROUP_DEC'] > -10.) * (fullsample['GROUP_DEC'] < 85.))
        fullsample = fullsample[I]

        version_archive = SGA_version(archive=True)
        parentdir = os.path.join(sga_dir(), 'parent')
        parentfile = os.path.join(parentdir, f'SGA2025-parent-archive-{region}-{version_archive}.fits')
        parent_rows = fitsio.read(parentfile, columns='ROW_PARENT')
        rows = np.where(np.isin(parent_rows, fullsample['SGAID'].value))[0]
        parent = Table(fitsio.read(parentfile, rows=rows))
        indx_fullsample, indx_parent = match(fullsample['SGAID'], parent['ROW_PARENT'])
        fullsample = fullsample[indx_fullsample]
        parent = parent[indx_parent]

        I = (parent['Z'] > 0.002) * (parent['Z'] < 0.025)
        fullsample = fullsample[I]

        # build primary member sample and then we need to restore all
        # group members otherwise we run into problems in
        # build_catalog.
        sample = sample[np.isin(sample['GROUP_ID'], fullsample['GROUP_ID'])]
        fullsample = ofullsample[np.isin(ofullsample['GROUP_ID'], sample['GROUP_ID'])]
        log.info(f'Selecting {len(fullsample):,d}/{nfullobj:,d} ({len(sample):,d}/{nobj:,d}) wisesize groups (objects)')


    if False:
        from SGA.ellipse import ELLIPSEMODE
        ## remove
        #I = sample['ELLIPSEMODE'] & ELLIPSEMODE['RESOLVED'] == 0
        #log.warning(f'Temporarily removing {np.sum(~I):,d} LVD-RESOLVED sources!')
        # keep
        I = sample['ELLIPSEMODE'] & ELLIPSEMODE['RESOLVED'] != 0
        log.warning(f'Temporarily restricting to {np.sum(I):,d} LVD-RESOLVED sources!')
        sample = sample[I]

        fullsample = fullsample[np.isin(fullsample['GROUP_ID'], sample['GROUP_ID'])]

    if False:#True:
        from SGA.ellipse import ELLIPSEMODE
        I = sample['ELLIPSEMODE'] & ELLIPSEMODE['FIXGEO'] != 0
        log.warning(f'Temporarily restricting to {np.sum(I):,d} sources with FIXGEO!')
        sample = sample[I]
        fullsample = fullsample[np.isin(fullsample['GROUP_ID'], sample['GROUP_ID'])]

    ##############################
    # v0.10 had 30 duplicate groups (29 in dr11-south; 1 in
    # dr9-north); remove the duplicates here.
    if version == 'v0.10':
        dupgroups = [
            '01298m3385','01747p1908','02097m0808','03152m3114','06833m1343','07544m1821','12066m0929','12991m0980',
            '13264p1019','14230p3936','15224p0781','17788m3116','18407p1841','19454p2754','20114m2844','20405m0103',
            '20722m1948','20748m1818','21013p1208','21208m2877','21882p1216','22721p0784','24739m1655','26426m0091',
            '33922p1322','34799m1279','35482p2642','35555m4025','35846m1262','35893p1399']
        drop = ['DUKST 351-033','2MASX J01095516+1904570','WISEA J012355.12-080514.3','WISEA J020605.35-310900.1',
                'WISEA J043320.16-132621.2','WISEA J050147.86-181307.2','WISEA J080239.07-091758.5','WISEA J083939.57-094802.8',
                'WISEA J085033.88+101123.9','WISEA J092912.50+392211.3','WISEA J100859.72+074842.9','WISEA J115133.31-311009.8',
                'WISEA J121619.21+182509.4','WISEA J125809.70+273257.8','WISEA J132435.80-282653.3','UGC 08584 NED01',
                'WISEA J134855.15-192913.8','WISEA J134955.46-181051.6','WISEA J140032.74+120521.4','WISEA J140821.57-284644.5',
                'SDSS J143518.67+120938.6','WISEA J150852.22+075029.9','WISEA J162933.60-163306.0','WISEA J173704.33-005508.8',
                'WISEA J223653.00+131314.7','WISEA J231159.69-124755.9','WISEA J233916.97+262513.0','WISEA J234212.09-401532.7',
                'WISEA J235352.66-123739.4','WISEA J235543.52+140000.7']
        if False:
            I = np.isin(sample['OBJNAME'], drop)
            fullsample = fullsample[np.isin(fullsample['OBJNAME'], drop)]
            log.warning(f'version v0.10---keeping {np.sum(I)} objects in duplicate groups!')
        else:
            I = ~np.isin(sample['OBJNAME'], drop)
            fullsample = fullsample[~np.isin(fullsample['OBJNAME'], drop)]
            log.warning(f'version v0.10---dropping {np.sum(~I)} objects in duplicate groups!')
        sample = sample[I]
    ##############################

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

    #if version == 'v0.40':
    #    print('HACK!!!')
    #    redo = Table.read('/global/u2/i/ioannis/redo-galdir.txt', format='csv')['C'].value
    #    base = np.array([os.path.basename(path) for path in redo])
    #    sample = sample[np.isin(sample['GROUP_NAME'], base)]
    #    fullsample = fullsample[np.isin(fullsample['GROUP_NAME'], base)]

    #if version == 'v0.50' and False:
    #    print('HACK!!!')
    #    redo = Table.read('/global/u2/i/ioannis/redo-objname.txt', format='csv')['C'].value
    #    fullsample = fullsample[np.isin(fullsample['OBJNAME'], redo)]
    #    sample = fullsample[fullsample['GROUP_PRIMARY']]

    return sample, fullsample


def SGA_diameter(ellipse, region, radius_arcsec=False, censor_all_zband=False,
                 verbose=False):
    """Compute D26 diameter from ellipse measurements.

    Parameters
    ----------
    ellipse : astropy.table.Table
        Table with isophotal radii (R{TH}_{BAND} columns), ELLIPSEMODE, and SMA_MOMENT.
    region : str
        Survey region ('dr9-north', 'dr9-south', 'dr9-south-ngc5128', etc.).
        Required to handle region-specific data quality issues.
    radius_arcsec : bool, optional
        If True, return radius in arcsec instead of diameter in arcmin.
    censor_all_zband : bool, optional
        If True and region is 'dr9-north', censor all z-band profiles. If False
        (default), only censor z-band for rows that have valid isophotal radii
        in other bands (g, r, i), preserving z-band as fallback when it's the
        only available measurement.
    verbose : bool, optional
        If True, print per-object diagnostic output showing available radii
        and the resulting D26.

    Returns
    -------
    d26 : ndarray
        D26 diameter in arcmin (or R26 radius in arcsec if radius_arcsec=True).
    d26_err : ndarray
        1-sigma uncertainty.
    d26_ref : ndarray
        Channel that contributed highest weight ('r26', 'g25', 'mom', 'fix', etc.).
    d26_weight : ndarray
        Weight of the highest-contributing channel.

    """
    from SGA.ellipse import ELLIPSEMODE, ELLIPSEBIT
    from SGA.calibrate import infer_best_r26

    for col in ['ELLIPSEMODE', 'SMA_MOMENT']:
        if col not in ellipse.colnames:
            msg = f'Missing mandatory column {col}'
            log.critical(msg)
            raise ValueError(msg)

    # Work on a copy to avoid modifying the input table
    ellipse = ellipse.copy()

    # Censor the 26 mag/arcsec2 isophotes if the FAILGEO bit is set
    # (near a bright star).
    if 'ELLIPSEBIT' in ellipse.colnames:
        I = ellipse['ELLIPSEBIT'] & (ELLIPSEBIT['FAILGEO'] | ELLIPSEBIT['NORADWEIGHT']) != 0
        if np.any(I):
            r26_cols = [col for col in ellipse.colnames if col.startswith('R26_')]
            for col in r26_cols:
                ellipse[col][I] = np.nan

    # Censor unreliable z-band profiles in dr9-north
    if region == 'dr9-north':
        z_cols = [col for col in ellipse.colnames if col.startswith('R2') and '_Z' in col]

        if censor_all_zband:
            for col in z_cols:
                ellipse[col] = np.nan
        else:
            # Only censor z-band for rows that have other valid isophotal radii
            gri_cols = [col for col in ellipse.colnames
                        if col.startswith('R2') and ('_G' in col or '_R' in col or '_I' in col)
                        and '_ERR' not in col]
            if gri_cols:
                has_gri = np.zeros(len(ellipse), dtype=bool)
                for col in gri_cols:
                    has_gri |= np.isfinite(ellipse[col]) & (ellipse[col] > 0)
                for col in z_cols:
                    ellipse[col] = np.where(has_gri, np.nan, ellipse[col])

    d26, d26_err, d26_ref, d26_weight = infer_best_r26(ellipse)

    I = np.isin(d26_ref, 'moment')
    if np.any(I):
        d26_ref[I] = 'mom'
    d26_ref = d26_ref.astype('<U3')

    # Fixed geometry
    I = (ellipse['ELLIPSEMODE'] & ELLIPSEMODE['FIXGEO']) != 0
    if np.any(I):
        d26[I] = ellipse['SMA_MOMENT'][I] * 2. / 60.
        d26_err[I] = 0.
        d26_ref[I] = 'fix'
        d26_weight[I] = 1.

    # If the SKIPTRACTOR bit is set and SMA_MOMENT is the only radius
    # measurement, revert to the initial geometry (otherwise
    # SMA_MOMENT will be used and D26 will be significantly larger
    # than its initial size.)
    if 'ELLIPSEBIT' in ellipse.colnames:
        I = (ellipse['ELLIPSEBIT'] & ELLIPSEBIT['SKIPTRACTOR'] != 0) & (d26_ref == 'mom')
        if np.any(I):
            d26[I] = ellipse['SMA_MOMENT'][I] * 2. / 60. # [arcmin]
            d26_err[I] = 0.
            d26_ref[I] = 'ini'
            d26_weight[I] = 1.

    if verbose:
        for i in range(len(ellipse)):
            for thresholds in [[24, 23], [26, 25]]:
                parts = []
                for th in thresholds:
                    for band in ['R', 'I', 'Z', 'G']:
                        rcol = f'R{th}_{band}'
                        ecol = f'R{th}_ERR_{band}'
                        if rcol in ellipse.colnames:
                            r = ellipse[rcol][i]
                            if np.isfinite(r) and r > 0:
                                e = ellipse[ecol][i] if ecol in ellipse.colnames else 0
                                if np.isfinite(e) and e > 0:
                                    parts.append(f"{band.lower()}({th})={r:.1f}±{e:.2f}")
                                else:
                                    parts.append(f"{band.lower()}({th})={r:.1f}")
                if parts:
                    log.info(" ".join(parts) + " arcsec")
            log.info(f"D(26)={d26[i]:.3f}±{d26_err[i]:.3f} arcmin [ref={d26_ref[i]}]")

    if radius_arcsec:
        r26 = d26 / 2. * 60.
        r26_err = d26_err / 2. * 60.
        return r26, r26_err, d26_ref, d26_weight
    else:
        return d26, d26_err, d26_ref, d26_weight


def SGA_geometry(ellipse, region, radius_arcsec=False):
    """Extract galaxy geometry (size and shape) from ellipse measurements.

    Parameters
    ----------
    ellipse : astropy.table.Table
        Table with isophotal radii, BA_MOMENT, PA_MOMENT, ELLIPSEMODE, and SMA_MOMENT.
    region : str
        Survey region ('dr9-north', 'dr9-south', etc.). Passed to SGA_diameter
        to handle region-specific data quality issues.
    radius_arcsec : bool, optional
        If True, return radius in arcsec instead of diameter in arcmin.

    Returns
    -------
    diam : ndarray
        D26 diameter in arcmin (or R26 radius in arcsec if radius_arcsec=True).
    ba : ndarray
        Axis ratio (b/a) from moment analysis.
    pa : ndarray
        Position angle in degrees (astronomical convention) from moment analysis.
    diam_err : ndarray
        1-sigma uncertainty on diameter.
    diam_ref : ndarray
        Channel that contributed highest weight ('r26', 'g25', 'mom', 'fix', etc.).
    diam_weight : ndarray
        Weight of the highest-contributing channel.

    """
    ba = ellipse['BA_MOMENT'].value
    pa = ellipse['PA_MOMENT'].value
    diam, diam_err, diam_ref, diam_weight = SGA_diameter(
        ellipse, region, radius_arcsec=radius_arcsec)
    return diam, ba, pa, diam_err, diam_ref, diam_weight


def SGA_datamodel(ellipse, bands, all_bands, copy=True):
    import astropy.units as u
    from astropy.table import Column, MaskedColumn

    nobj = len(ellipse)

    ubands = np.char.upper(bands)
    uall_bands = np.char.upper(all_bands)

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
        ('OPTFLUX', np.float32, u.nanomaggy),
        ('SGANAME', 'U25', None),
        ('RA', np.float64, u.degree),
        ('DEC', np.float64, u.degree),
        ('BX', np.float32, u.pixel),
        ('BY', np.float32, u.pixel),
        ('SMA_MASK', np.float32, u.arcsec),
        ('SMA_MOMENT', np.float32, u.arcsec),
        ('BA_MOMENT', np.float32, None),
        ('PA_MOMENT', np.float32, u.degree),
        ('RA_TRACTOR', np.float64, u.degree),
        ('DEC_TRACTOR', np.float64, u.degree),
        ('ELLIPSEBIT', np.int32, None),
    ]
    for filt in uall_bands:
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
        ('D26_ERR', np.float32, u.arcmin),
        ('D26_REF', '<U3', None),
        ('BA', np.float32, None),
        ('PA', np.float32, u.degree),
    ]

    out = Table()
    for col in dmcols:
        out.add_column(Column(name=col[0], data=np.zeros(nobj, dtype=col[1]), unit=col[2]))

    # copy over the data
    if copy:
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
                        check.append(I)
                        val[I] = 0
                out[col] = val
        if len(check) > 0:
            check = np.unique(np.hstack(check))
            print(','.join(ellipse['GROUP_NAME'][check].value))

    return out


def _create_mock_ellipse_from_sample(grpsample):
    """Create mock ellipse catalog from parent sample when processing failed.

    Sets SKIPTRACTOR bit, FIXGEO fitmode, and populates geometry from initial values.

    """
    from SGA.ellipse import ELLIPSEBIT, FITMODE

    n = len(grpsample)

    # Validate GROUP_ID consistency within grpsample
    if 'GROUP_ID' in grpsample.colnames:
        unique_gids = np.unique(grpsample['GROUP_ID'])
        if len(unique_gids) > 1:
            raise ValueError(f'Inconsistent GROUP_ID in grpsample: {unique_gids}')

    ellipse = Table()

    # Copy identifying columns
    for col in [REFIDCOLUMN, 'SGAGROUP', 'OBJNAME', 'GROUP_ID', 'GROUP_NAME', 'GROUP_MULT',
                'GROUP_PRIMARY', 'GROUP_RA', 'GROUP_DEC', 'GROUP_DIAMETER',
                'REGION', 'PGC', 'SAMPLE']:
        if col in grpsample.colnames:
            ellipse[col] = grpsample[col]

    # Geometry from initial values
    ellipse['RA'] = grpsample['RA']
    ellipse['DEC'] = grpsample['DEC']
    ellipse['SMA_MOMENT'] = grpsample['DIAM'] * 60. / 2.  # arcmin -> arcsec radius
    ellipse['BA_MOMENT'] = grpsample['BA'] if 'BA' in grpsample.colnames else np.ones(n, dtype=np.float32)
    ellipse['PA_MOMENT'] = grpsample['PA'] if 'PA' in grpsample.colnames else np.zeros(n, dtype=np.float32)

    # Copy initial geometry columns
    ellipse['RA_INIT'] = grpsample['RA']
    ellipse['DEC_INIT'] = grpsample['DEC']
    ellipse['DIAM_INIT'] = grpsample['DIAM']
    ellipse['BA_INIT'] = grpsample['BA'] if 'BA' in grpsample.colnames else np.ones(n, dtype=np.float32)
    ellipse['PA_INIT'] = grpsample['PA'] if 'PA' in grpsample.colnames else np.zeros(n, dtype=np.float32)
    ellipse['MAG_INIT'] = grpsample['MAG'] if 'MAG' in grpsample.colnames else np.zeros(n, dtype=np.float32)
    ellipse['DIAM_REF_INIT'] = grpsample['DIAM_REF'] if 'DIAM_REF' in grpsample.colnames else np.full(n, '', dtype='U9')

    ellipse['ELLIPSEBIT'] = np.full(n, ELLIPSEBIT['SKIPTRACTOR'], dtype=np.int32)
    ellipse['ELLIPSEMODE'] = np.zeros(n, dtype=np.int32)
    ellipse['FITMODE'] = np.full(n, FITMODE['FIXGEO'], dtype=np.int32)

    return ellipse


def _create_mock_tractor_sga(refids):
    """Create mock Tractor catalog entries for SKIPTRACTOR/NOTRACTOR sources."""
    from SGA.io import empty_tractor

    refids = np.atleast_1d(refids)

    if len(refids) == 0:
        return Table()

    tractor_list = []
    for refid in refids:
        t = empty_tractor()
        t['ref_cat'] = REFCAT
        t['ref_id'] = refid
        tractor_list.append(t)

    return vstack(tractor_list)


def _build_tractor_sga_entries(tractor, ellipse):
    """Build mock Tractor entries for SGA sources not in Tractor catalog."""
    from SGA.ellipse import ELLIPSEBIT, FITMODE

    tractor_sga_list = []
    notractor_indices = []

    for i in range(len(ellipse)):
        if len(tractor) > 0:
            match = (
                (tractor['ref_id'] == ellipse[REFIDCOLUMN][i]) &
                (tractor['ref_cat'] == REFCAT)
            )
            n_match = np.sum(match)

            if n_match > 1:
                raise IOError('Multiple SGA sources in Tractor catalog!')

            if n_match == 1:
                continue

        # No match - verify NOTRACTOR bit
        assert ellipse['ELLIPSEBIT'][i] & ELLIPSEBIT['NOTRACTOR'] != 0
        notractor_indices.append(i)

    # Set FIXGEO for NOTRACTOR sources
    if notractor_indices:
        ellipse['FITMODE'][notractor_indices] |= FITMODE['FIXGEO']

    # Create mock tractor entries and return as Table (not list)
    if notractor_indices:
        tractor_sga = _create_mock_tractor_sga(ellipse[REFIDCOLUMN][notractor_indices])
    else:
        tractor_sga = Table()

    return tractor_sga


def _read_ellipse_catalogs(gdir, datasets, opt_bands, grpsample):
    """Read and join ellipse catalogs across datasets.

    Returns None if no ellipse files found or on read error.

    """
    import fitsio
    from glob import glob

    ellipsefiles = glob(os.path.join(gdir, f'*-ellipse-{opt_bands}.fits'))
    if len(ellipsefiles) == 0:
        return None

    ellipse_list = []
    for ellipsefile in ellipsefiles:
        # Extract SGA name from filename (fragile)
        sganame_parts = os.path.basename(ellipsefile).split('-')[:-2]
        sganame = sganame_parts[0] if len(sganame_parts) == 1 else '-'.join(sganame_parts)

        # Join across datasets
        ellipse1 = None
        for idata, dataset in enumerate(datasets):
            ellipsefile_dataset = os.path.join(gdir, f'{sganame}-ellipse-{dataset}.fits')
            try:
                ellipse_dataset = Table(fitsio.read(ellipsefile_dataset, ext='ELLIPSE'))
            except Exception:
                log.critical(f'Problem reading {ellipsefile_dataset}!')
                return None

            if idata == 0:
                ellipse1 = ellipse_dataset
            else:
                ellipse1 = join(ellipse1, ellipse_dataset)

        ellipse_list.append(ellipse1)

    ellipse = vstack(ellipse_list) if ellipse_list else None

    # Validate GROUP_ID matches grpsample
    if False and ellipse is not None and 'GROUP_ID' in ellipse.colnames and 'GROUP_ID' in grpsample.colnames:
        for i, row in enumerate(ellipse):
            sgaid = row[REFIDCOLUMN]
            match_idx = np.where(grpsample[REFIDCOLUMN] == sgaid)[0]
            if len(match_idx) > 0:
                expected_gid = grpsample['GROUP_ID'][match_idx[0]]
                actual_gid = row['GROUP_ID']
                if expected_gid != actual_gid:
                    print(f'Ellipse file may be out of date: {gdir}')
                    pdb.set_trace()
                    raise ValueError(f'GROUP_ID mismatch for SGAID {sgaid}: '
                                     f'ellipse has {actual_gid}, grpsample has {expected_gid}. '
                                     f'Ellipse file may be out of date: {gdir}')

    return ellipse


def _read_tractor_catalog(gdir, grp, ellipse, refid_array, region):
    """Read Tractor catalog and extract SGA source entries.

    Returns
    -------
    tractor : Table
        All relevant Tractor sources (SGA + sources within ellipses)
    tractor_sga : Table
        Mock entries for SGA sources without Tractor matches

    """
    import fitsio
    from SGA.io import empty_tractor
    from SGA.ellipse import ELLIPSEBIT, ELLIPSEMODE, FITMODE
    from SGA.sky import in_ellipse_mask_sky

    tractorfile = os.path.join(gdir, f'{grp}-tractor.fits')

    # --- Case 1: No Tractor file ---
    if not os.path.isfile(tractorfile):
        tractor, tractor_sga = _handle_missing_tractor_file(ellipse)
        return tractor, tractor_sga

    # --- Case 2: Tractor file exists ---
    refs = fitsio.read(tractorfile, columns=[
        'brick_primary', 'ra', 'dec', 'type', 'fitbits', 'ref_cat', 'ref_id'])

    # Filter to valid sources
    valid = refs['brick_primary'] & (refs['type'] != 'DUP')
    if np.sum(valid) == 0:
        log.warning(f'No sources in {tractorfile}')
        return Table(), Table()

    # Find sources inside SGA ellipses
    isin = _sources_in_ellipses(refs, ellipse, region)

    # Select: valid AND (inside ellipse OR is SGA source)
    keep = valid & (isin | (refs['ref_cat'] == REFCAT))

    if np.sum(keep) == 0:
        tractor = Table()
    else:
        tractor = Table(fitsio.read(tractorfile, rows=np.where(keep)[0]))

        # check the data model
        dm = empty_tractor()
        missing_in_tractor = set(dm.colnames) - set(tractor.colnames)
        extra_in_tractor = set(tractor.colnames) - set(dm.colnames)
        if missing_in_tractor or extra_in_tractor:
            raise ValueError(f'Tractor schema mismatch in {tractorfile}: '
                             f'missing={missing_in_tractor}, extra={extra_in_tractor}')

        # Remove SGA sources that don't belong to this group
        foreign_sga = ((tractor['ref_cat'] == REFCAT) &
            ~np.isin(tractor['ref_id'], ellipse[REFIDCOLUMN]))
        if np.any(foreign_sga):
            tractor.remove_rows(np.where(foreign_sga)[0])

    # Build tractor_sga entries for NOTRACTOR sources (also sets FIXGEO on ellipse)
    tractor_sga = _build_tractor_sga_entries(tractor, ellipse)

    return tractor, tractor_sga


def _handle_missing_tractor_file(ellipse):
    """Handle cases where Tractor file doesn't exist."""
    from SGA.ellipse import ELLIPSEBIT, ELLIPSEMODE, FITMODE

    # RESOLVED sources don't have Tractor catalogs (FIXGEO already set in parent)
    if len(ellipse) == 1 and (ellipse['ELLIPSEMODE'][0] & ELLIPSEMODE['RESOLVED'] != 0):
        tractor_sga = _create_mock_tractor_sga(ellipse[REFIDCOLUMN])
        return Table(), tractor_sga

    # SKIPTRACTOR means no Tractor for whole mosaic
    if np.any(ellipse['ELLIPSEBIT'] & ELLIPSEBIT['SKIPTRACTOR'] != 0):
        skiptractor_mask = (ellipse['ELLIPSEBIT'] & ELLIPSEBIT['SKIPTRACTOR']) != 0
        ellipse['FITMODE'][skiptractor_mask] |= FITMODE['FIXGEO']
        tractor_sga = _create_mock_tractor_sga(ellipse[REFIDCOLUMN])
        return Table(), tractor_sga

    raise ValueError('Unexpected case: no Tractor file but no RESOLVED/SKIPTRACTOR flag')


def _sources_in_ellipses(refs, ellipse, region):
    """Return boolean mask of refs that fall inside any SGA ellipse."""
    from SGA.sky import in_ellipse_mask_sky

    isin = np.zeros(len(refs), bool)
    rad, ba, pa, _, _, _ = SGA_geometry(ellipse, region, radius_arcsec=True)

    for iobj in range(len(ellipse)):
        isin |= in_ellipse_mask_sky(
            ellipse['RA'][iobj], ellipse['DEC'][iobj],
            rad[iobj] / 3600., rad[iobj] * ba[iobj] / 3600., pa[iobj],
            np.asarray(refs['ra']), np.asarray(refs['dec']))

    return isin


def build_catalog_one(datadir, region, datasets, opt_bands, grpsample, no_groups):
    """Gather ellipse-fitting results for a single group."""
    import fitsio
    from glob import glob

    from SGA.io import empty_tractor
    from SGA.ellipse import ELLIPSEBIT, ELLIPSEMODE
    from SGA.sky import in_ellipse_mask_sky

    # --- Locate group directory ---
    grp, gdir = get_galaxy_galaxydir(
        grpsample[0], region=region,
        group=not no_groups, datadir=datadir)

    # entire directory missing (not yet started)
    if not os.path.isdir(gdir):
        #for obj in grpsample:
        #    log.warning(f'Missing directory {gdir} {obj["OBJNAME"]} d={obj[DIAMCOLUMN]:.3f} arcmin')
        ellipse = _create_mock_ellipse_from_sample(grpsample)
        tractor = _create_mock_tractor_sga(ellipse[REFIDCOLUMN])
        return ellipse, tractor

    # --- Read ellipse catalogs ---
    ellipse = _read_ellipse_catalogs(gdir, datasets, opt_bands, grpsample)
    if ellipse is None:
        #for obj in grpsample:
        #    log.warning(f'Missing ellipse files {gdir} {obj["OBJNAME"]} d={obj[DIAMCOLUMN]:.3f} arcmin')
        ellipse = _create_mock_ellipse_from_sample(grpsample)
        tractor = _create_mock_tractor_sga(ellipse[REFIDCOLUMN])
        return ellipse, tractor

    # --- Validate ellipse catalogs match input sample ---
    refid_array = grpsample['SGAID'].value
    if not np.all(np.isin(ellipse[REFIDCOLUMN], refid_array)):
        for obj in grpsample:
            log.warning(f'Mismatch ref_id {gdir} {obj["OBJNAME"]} d={obj[DIAMCOLUMN]:.3f} arcmin')
        ellipse = _create_mock_ellipse_from_sample(grpsample)
        tractor = _create_mock_tractor_sga(ellipse[REFIDCOLUMN])
        return ellipse, tractor

    # --- Read Tractor catalog ---
    tractor, tractor_sga = _read_tractor_catalog(
        gdir, grp, ellipse, refid_array, region)

    # Append mock SGA entries to tractor
    if len(tractor_sga) > 0:
        tractor = vstack((tractor, tractor_sga)) if len(tractor) > 0 else tractor_sga

    #if np.any(np.isin(grpsample['GROUP_NAME'], ['02327m8659'])):
    #    pdb.set_trace()

    return ellipse, tractor


def build_catalog(sample, fullsample, comm=None, bands=['g', 'r', 'i', 'z'],
                  region='dr11-south', test_bricks=False, galex=True, unwise=True,
                  wisesize=False, no_groups=False, datadir=None, verbose=False,
                  clobber=False):
    """Build the final catalog.

    FIXME - combine the north and south

    NB: When combining north-south catalogs, need to look at OBJNAME;
    SGANAME may not be the same!

    """
    import time
    from glob import glob
    import multiprocessing
    from astropy.io import fits
    from SGA.io import get_raslice
    from SGA.ellipse import ELLIPSEBIT, FITMODE
    from SGA.coadds import REGIONBITS
    from SGA.util import match, get_dt


    def write_kdfile(outfile, kdoutfile):
        # KD version
        cmd1 = f'startree -i {outfile} -o {kdoutfile} -T -P -k -n stars'
        cmd2 = f'modhead {kdoutfile} VER {REFCAT}-ellipse'
        _ = os.system(cmd1)
        _ = os.system(cmd2)


    if comm:
        rank, size = comm.rank, comm.size
    else:
        rank, size = 0, 1

    # Initialize the variables we will broadcast if using MPI.
    if comm:
        outfile = None
        datasets = None
        opt_bands = None
        raslices_todo = None

    t0 = time.time()

    if rank == 0:
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
        #version = 'v0.10b'
        if test_bricks:
            version = 'testbricks-v0.60'
            outprefix = 'SGA2025'
            outfile = f'{outprefix}-{version}.fits'
            kdoutfile = f'{outprefix}-{version}.fits'
            outfile_ellipse = f'{outprefix}-ellipse-{version}.fits'
            kdoutfile_ellipse = f'{outprefix}-ellipse-{version}.kd.fits'
        else:
            if wisesize:
                outprefix = 'SGA2025-wisesize'
            else:
                outprefix = 'SGA2025'
                if False:#True:
                    print('TESTING!!!')
                    version = 'test'
                    outprefix = 'SGA2025-test'
            outfile = f'{outprefix}-beta-{version}-{region}.fits'
            kdoutfile = f'{outprefix}-beta-{version}-{region}.fits'
            outfile_ellipse = f'{outprefix}-ellipse-{version}-{region}.fits'
            kdoutfile_ellipse = f'{outprefix}-ellipse-{version}-{region}.kd.fits'

        outfile = os.path.join(sga_dir(), 'sample', outfile)
        kdoutfile = os.path.join(sga_dir(), 'sample', kdoutfile)
        outfile_ellipse = os.path.join(sga_dir(), 'sample', outfile_ellipse)
        kdoutfile_ellipse = os.path.join(sga_dir(), 'sample', kdoutfile_ellipse)

    if comm:
        outfile = comm.bcast(outfile, root=0)

    if os.path.isfile(outfile) and not clobber:
        if rank == 0:
            log.warning(f'Use --clobber to overwrite existing catalog {outfile}')
        return


    # outer loop is on RA slice
    if rank == 0:
        allraslices = get_raslice(sample['GROUP_RA'].value)
        uraslices = sorted(set(allraslices))

        raslices_todo = []
        for raslice in uraslices:
            slicefile = os.path.join(datadir, region, f'{outprefix}-{raslice}.fits')
            if os.path.isfile(slicefile):# and not clobber:
                log.warning(f'Skipping existing catalog {slicefile}')
                continue
            raslices_todo.append(raslice)
        raslices_todo = np.array(raslices_todo)

        #raslices_todo = ['097']
        #raslices_todo = raslices_todo[131:]

    if comm:
        datasets = comm.bcast(datasets, root=0)
        opt_bands = comm.bcast(opt_bands, root=0)
        raslices_todo = comm.bcast(raslices_todo, root=0)
        #allraslices = comm.bcast(allraslices, root=0)

    # outer loop on RA slices
    for islice, raslice in enumerate(raslices_todo):
        #log.info(f'Rank {rank:03}: working on RA slice {raslice} ({islice+1:03}/{len(raslices_todo):03}) ')
        if rank == 0:
            #log.info(f'Working on RA slice {raslice} ({islice+1:03}/{len(raslices_todo):03}) ')
            I = np.where(raslice == allraslices)[0]

        if comm is not None and size > 1:
            # rank 0 sends work to the other ranks...
            if rank == 0:
                indx_byrank = np.array_split(I, size-1)
                for onerank, indx in zip(np.arange(size-1)+1, indx_byrank):
                    #log.info(f'Rank {rank:03} distributing {len(indx):,d} objects to rank {onerank:03}')
                    comm.send(indx, dest=onerank, tag=1)
            else:
                # ...and the other ranks receive the work.
                indx = comm.recv(source=0, tag=1)
                #log.info(f'Rank {rank:03} received {len(indx):,d} objects from rank 0')
        else:
            indx = I

        if comm is not None and size > 1:
            # The other ranks do work and send their results...
            if rank > 0:
                if len(indx) == 0:
                    ellipse = Table()
                    tractor = Table()
                else:
                    #log.info(f'Rank {rank:03} RA slice {raslice}: working on {len(indx):,d} groups')
                    out = []
                    for count, grpindx in enumerate(indx):
                        #if count % 100 == 0:
                        #    log.info(f'Rank {rank:03} RA slice {raslice}: working on group {count+1:,d}/{len(indx):,d}')
                        #refids = fullsample[REFIDCOLUMN][fullsample['GROUP_ID'] == sample['GROUP_ID'][igrp]].value
                        grpsample = fullsample[fullsample['GROUP_ID'] == sample['GROUP_ID'][grpindx]]
                        out1 = build_catalog_one(datadir, region, datasets, opt_bands, grpsample, no_groups)
                        #t1 = Table()
                        #t1['RA'] = [180.]
                        #t1['DEC'] = [30.]
                        #out1 = [t1, t1]
                        out.append(out1)
                    out = list(zip(*out))
                    ellipse = vstack(out[0])
                    tractor = vstack(out[1])

                #log.info(f'Rank {rank:03} RA slice {raslice}: sending ellipse (Tractor) table ' + \
                #         f'with {len(ellipse):,d} ({len(tractor):,d}) rows to rank 000')
                comm.send(ellipse, dest=0, tag=2)
                comm.send(tractor, dest=0, tag=3)
            else:
                # ...to rank 0.
                allellipse, alltractor = [], []
                for onerank in np.arange(size-1)+1:
                    ellipse = comm.recv(source=onerank, tag=2)
                    tractor = comm.recv(source=onerank, tag=3)
                    #log.info(f'Rank {rank:03} RA slice {raslice}: received ellipse (Tractor) catalogs ' + \
                    #         f'with {len(ellipse):,d} ({len(tractor):,d}) objects from rank {onerank:03}')
                    allellipse.append(ellipse)
                    alltractor.append(tractor)
                allellipse = vstack(allellipse)
                alltractor = vstack(alltractor)
        else:
            #log.info(f'Rank {rank:03} RA slice {raslice}: working on {len(indx):,d} groups')
            out = []
            for count, grpindx in enumerate(indx):
                #if count % 100 == 0:
                #    log.info(f'Rank {rank:03} RA slice {raslice}: working on group {count+1:,d}/{len(indx):,d}')
                #refids = fullsample[REFIDCOLUMN][fullsample['GROUP_ID'] == sample['GROUP_ID'][igrp]].value
                grpsample = fullsample[fullsample['GROUP_ID'] == sample['GROUP_ID'][grpindx]]
                out1 = build_catalog_one(datadir, region, datasets, opt_bands, grpsample, no_groups)
                out.append(out1)
            out = list(zip(*out))
            allellipse = vstack(out[0])
            alltractor = vstack(out[1])

        if rank == 0:
            slicefile = os.path.join(datadir, region, f'{outprefix}-{raslice}.fits')
            if len(allellipse) > 0:
                #log.info(f'Writing {len(allellipse):,d} ({len(alltractor):,d}) groups (Tractor sources) to {slicefile}')
                fitsio.write(slicefile, allellipse.as_array(), extname='ELLIPSE', clobber=True)
                fitsio.write(slicefile, alltractor.as_array(), extname='TRACTOR')

    if rank == 0:
        dt, unit = get_dt(t0)
        log.info(f'Rank {rank:03} all done in {dt:.3f} {unit}')

    if comm:
        comm.barrier()

    # Now loop back through and gather up all the results (on rank 0).
    if rank == 0:
        t0 = time.time()
        t1 = time.time()
        #log.info(f'Rank {rank:03} gathering catalogs from {len(raslices_todo)} RA slices.')

        ellipse, tractor = [], []
        for islice, raslice in enumerate(uraslices):
            slicefile = os.path.join(datadir, region, f'{outprefix}-{raslice}.fits')
            if not os.path.isfile(slicefile):
                log.info(f'Skipping missing file {slicefile}')
            else:
                ellipse.append(Table(fitsio.read(slicefile, 'ELLIPSE')))
                tractor.append(Table(fitsio.read(slicefile, 'TRACTOR')))
                #os.remove(slicefile)

        if len(ellipse) == 0:
            log.warning('No ellipse catalogs to stack; returning')
            return

        ellipse = vstack(ellipse)
        tractor = vstack(tractor)
        nobj = len(ellipse)

        dt, unit = get_dt(t1)
        log.info(f'Gathered ellipse measurements for {nobj:,d} unique objects and ' + \
                 f'{len(tractor):,d} Tractor sources from {len(uraslices)} RA ' + \
                 f'slices took {dt:.3f} {unit}.')

        I = np.isin(ellipse[REFIDCOLUMN], tractor['ref_id'])
        if not np.all(I):
            log.warning('ref_id mismatch between ellipse and tractor!')
        #tractor[np.isin(tractor['ref_id'], ellipse[REFIDCOLUMN])]

        # re-organize the ellipse table to match the datamodel and assign units
        outellipse = SGA_datamodel(ellipse, bands, all_bands)

        # final geometry
        diam, ba, pa, diam_err, diam_ref, _ = SGA_geometry(outellipse, region)
        for col, val in zip(['D26', 'BA', 'PA', 'D26_ERR', 'D26_REF'],
                            [diam, ba, pa, diam_err, diam_ref]):
            outellipse[col] = val

        I = np.logical_or(outellipse['D26'] <= 0., np.isnan(outellipse['D26']))
        if np.any(I):
            log.warning(f'Negative or infinite diameters for {np.sum(I):,d} objects!')
            #outellipse[outellipse['D26'] == 0.]['SGAGROUP', 'OBJNAME', 'R24_R', 'R25_R', 'R26_R', 'D26']

        #I_skip = (outellipse['ELLIPSEBIT'] & ELLIPSEBIT['SKIPTRACTOR']) != 0
        #skip_refids = outellipse[REFIDCOLUMN][I_skip]
        #in_tractor = np.isin(skip_refids, tractor['ref_id'])
        #log.info(f'SKIPTRACTOR entries in outellipse before match: {np.sum(I_skip)}')
        #log.info(f'SKIPTRACTOR ref_ids in tractor: {np.sum(in_tractor)}/{len(skip_refids)}')

        # separate out (and sort) the tractor catalog of the SGA sources
        I = np.where(tractor['ref_cat'] == REFCAT)[0]
        m1, m2 = match(outellipse[REFIDCOLUMN], tractor['ref_id'][I])
        outellipse = outellipse[m1]
        tractor_sga = tractor[I[m2]]
        tractor_nosga = tractor[np.delete(np.arange(len(tractor)), I)]

        assert(len(outellipse) == len(np.unique(outellipse['SGAID'])))
        assert(len(outellipse) == len(np.unique(outellipse['OBJNAME'])))

        log.warning('Need to ensure GROUP_ID is unique!!!')

        # Write out outfile with the ELLIPSE and TRACTOR HDUs.
        hdu_primary = fits.PrimaryHDU()
        hdu_ellipse = fits.convenience.table_to_hdu(outellipse)
        hdu_tractor_sga = fits.convenience.table_to_hdu(tractor_sga)
        hdu_ellipse.header['EXTNAME'] = 'ELLIPSE'
        hdu_tractor_sga.header['EXTNAME'] = 'TRACTOR'
        hx = fits.HDUList([hdu_primary, hdu_ellipse, hdu_tractor_sga])
        hx.writeto(outfile, overwrite=True, checksum=True)
        log.info(f'Wrote {len(outellipse):,d} objects to {outfile}')

        #write_kdfile(outfile, kdoutfile)
        #log.info(f'Wrote {len(outellipse):,d} objects to {kdoutfile}')

        # Write out outfile_ellipse by combining the ellipse and
        # tractor catalogs.
        ellipse_cols = ['RA', 'DEC', 'SGAID', 'MAG_INIT', 'PA', 'BA', 'D26', 'FITMODE']
        tractor_cols = ['ref_cat', 'type', 'sersic', 'shape_r', 'shape_e1', 'shape_e2', ] + \
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
        out_nosga['ref_id'] = tractor_nosga['ref_id']

        out_sga = hstack((out_sga, tractor_sga[tractor_cols]))
        out = vstack((out_sga, out_nosga))

        # FITMODE=FIXGEO is set in build_catalog_one for
        # SKIPTRACTOR/NOTRACTOR/RESOLVED, so set FITMODE=FREEZE for
        # the remaining sources.
        I = out['fitmode'] == 0
        log.info(f'Setting fitmode=FREEZE for {np.sum(I):,d}/{len(out):,d} objects.')
        out['fitmode'][I] |= FITMODE['FREEZE']

        I = out['fitmode'] & (FITMODE['FIXGEO'] | FITMODE['RESOLVED']) == (FITMODE['RESOLVED'] | FITMODE['FIXGEO'])
        log.info(f'Found {np.sum(I):,d}/{len(out):,d} objects with fitmode=FIXGEO & RESOLVED')

        I = (out['fitmode'] & FITMODE['FIXGEO'] != 0) & (out['fitmode'] & FITMODE['RESOLVED'] == 0)
        log.info(f'Found {np.sum(I):,d}/{len(out):,d} objects with fitmode=FIXGEO & not RESOLVED')

        hdu_primary = fits.PrimaryHDU()
        hdu_out = fits.convenience.table_to_hdu(out)
        hdu_out.header['EXTNAME'] = 'SGA2025'
        hdu_out.header['VER'] = REFCAT
        hx = fits.HDUList([hdu_primary, hdu_out])
        hx.writeto(outfile_ellipse, overwrite=True, checksum=True)
        log.info(f'Wrote {len(out):,d} objects to {outfile_ellipse}')

        write_kdfile(outfile_ellipse, kdoutfile_ellipse)
        log.info(f'Wrote {len(out):,d} objects to {kdoutfile_ellipse}')

        dt, unit = get_dt(t0)
        log.info(f'Rank {rank:03} all done in {dt:.3f} {unit}')



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
                log.info(f'No good measurements of the PSF size in band {filt}')
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
                log.info(f'No good measurements of the PSF depth in band {filt}.')
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


def _update_masks(brightstarmask, gaiamask, refmask, galmask,
                  mask_perband, bands, sz, MASKDICT=None, build_maskbits=False,
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
        maskbits[brightstarmask] |= MASKDICT['brightstar']
        maskbits[refmask] |= MASKDICT['reference']
        maskbits[galmask] |= MASKDICT['galaxy']
        maskbits[gaiamask] |= MASKDICT['gaiastar']

        for iband, filt in enumerate(bands):
            maskbits[mask_perband[iband, :, :]] |= MASKDICT[filt]
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


def _compute_major_minor_masks(flux_sga, fracflux_sga, allgalsrcs, galsrcs_optflux,
                               FMAJOR, objsrc, use_tractor_position_obj,
                               arcsec_between):
    """Classify Tractor galaxies into major / minor companions.

    Returns (major_mask, minor_mask), both boolean arrays of length len(allgalsrcs).

    """
    if len(allgalsrcs) == 0 or flux_sga <= 0.0:
        major_mask = np.zeros(len(allgalsrcs), bool)
        minor_mask = np.ones(len(allgalsrcs), bool)
        return major_mask, minor_mask

    R_flux = galsrcs_optflux / flux_sga
    major_mask = (R_flux >= FMAJOR)
    minor_mask = ~major_mask

    # Optionally ignore all sources that may be a shred of the SGA
    # itself. We originally included the condition "and
    # use_tractor_position_obj" but then MOMENTPOS was resulting in
    # many more "major" sources. The logic is not ideal because we
    # don't check how far the moment and Tractor positions are, but
    # proceed for now.
    if (objsrc is not None) and fracflux_sga > 0.2:
        sep_arcsec = arcsec_between(objsrc.ra, objsrc.dec,
                                    allgalsrcs.ra, allgalsrcs.dec)
        major_mask &= (sep_arcsec > objsrc.shape_r)

    return major_mask, minor_mask


def _log_object_modes(log, iobj, obj, use_radial_weight, use_tractor_geometry_obj,
                      ELLIPSEMODE, ELLIPSEBIT, stage="initial"):
    """
    stage = "initial" or "final".

    - initial: mimic your existing logging of ELLIPSEMODE + initial ELLIPSEBIT,
      and check that same-named bits in ELLIPSEMODE → ELLIPSEBIT are propagated.
    - final: mimic your "after all bits" logging of final ELLIPSEBIT.

    """
    mode_bits = obj["ELLIPSEMODE"]
    bit_bits  = obj["ELLIPSEBIT"]

    msg = []
    if stage == "initial":
        # --- ELLIPSEMODE bits ---
        for mode, bit in ELLIPSEMODE.items():
            if mode == "NORADWEIGHT":
                if ((mode_bits & bit) != 0) or not use_radial_weight:
                    #log.info(f"  {mode} = True ")
                    #msg.append(f"{mode} = True ")
                    if mode in ELLIPSEBIT and (bit_bits & ELLIPSEBIT[mode]) == 0:
                        log.warning(f"{mode}: ELLIPSEMODE set/used but ELLIPSEBIT[{mode}] is not set.")
            elif mode == "TRACTORGEO":
                if (mode_bits & bit) != 0:
                    #log.info(f"  {mode} = True ")
                    #msg.append(f"{mode} = True ")
                    if mode in ELLIPSEBIT and (bit_bits & ELLIPSEBIT[mode]) == 0:
                        log.warning(f"{mode}: ELLIPSEMODE set but ELLIPSEBIT[{mode}] is not set.")
                elif use_tractor_geometry_obj:
                    #log.info(f"  {mode} [satellite] = True ")
                    #msg.append(f"{mode} [satellite] = True ")
                    if mode in ELLIPSEBIT and (bit_bits & ELLIPSEBIT[mode]) == 0:
                        log.warning(f"{mode}: satellite runtime flag true but ELLIPSEBIT[{mode}] is not set.")
            else:
                if (mode_bits & bit) != 0:
                    #log.info(f"  {mode} = True ")
                    msg.append(f"{mode} = True ")
                    if mode in ELLIPSEBIT and (bit_bits & ELLIPSEBIT[mode]) == 0:
                        log.warning(f"{mode}: ELLIPSEMODE set but ELLIPSEBIT[{mode}] is not set.")

        # --- Initial ELLIPSEBIT bits ---
        for mode, bit in ELLIPSEBIT.items():
            if (bit_bits & bit) != 0:
                #log.info(f"  {mode} [initial] = True ")
                msg.append(f"{mode} [initial] = True ")

    elif stage == "final":
        # --- Final ELLIPSEBIT bits ---
        for mode, bit in ELLIPSEBIT.items():
            if (bit_bits & bit) != 0:
                #log.info(f"    {mode} = True ")
                msg.append(f"{mode} [final] = True ")

    return msg


def _get_radial_weight_and_tractor_geometry(sample, samplesrcs,
    opt_pixscale, use_tractor_position, use_radial_weight,
    use_radial_weight_for_overlaps, SATELLITE_FRAC, get_geometry,
    ellipses_overlap):
    """
    Decide per-object:
      - use_radial_weight_obj[i]: whether to use radial weighting in moments
      - use_tractor_geometry_obj[i]: whether to force Tractor geometry for satellites

    Global knobs:
      - use_radial_weight: default radial-weighting preference (True/False)
      - use_radial_weight_for_overlaps: if False, *any* overlap disables radial weighting

    """
    nsample = len(sample)

    # Precompute approximate geometry for overlap classification.
    geo_overlap = np.zeros((nsample, 5), "f4")  # [bx, by, sma, ba, pa] in pixels
    for iobj, (obj, objsrc) in enumerate(zip(sample, samplesrcs)):
        geo_overlap[iobj, :] = get_geometry(
            opt_pixscale, table=obj, ref_tractor=objsrc,
            use_sma_mask=False, use_tractor_position=use_tractor_position)

    overlap_obj = np.zeros(nsample, bool)
    satellite_obj = np.zeros(nsample, bool)

    # Baseline defaults:
    # - radial weighting follows global default
    # - Tractor-geometry override is off unless we decide "satellite"
    use_radial_weight_obj = np.full(nsample, bool(use_radial_weight), dtype=bool)
    use_tractor_geometry_obj = np.zeros(nsample, bool)

    if nsample == 1:
        return use_radial_weight_obj, use_tractor_geometry_obj, satellite_obj, overlap_obj

    for iobj in range(nsample):
        bx_i, by_i, sma_i, ba_i, pa_i = geo_overlap[iobj, :]

        if sma_i <= 0:
            # Degenerate geometry -> be conservative
            use_radial_weight_obj[iobj] = False
            use_tractor_geometry_obj[iobj] = False
            continue

        overlapping_indices = []
        for jobj in range(nsample):
            if jobj == iobj:
                continue

            bx_j, by_j, sma_j, ba_j, pa_j = geo_overlap[jobj, :]
            if sma_j <= 0:
                continue

            if ellipses_overlap(bx_i, by_i, sma_i, ba_i, pa_i,
                                bx_j, by_j, sma_j, ba_j, pa_j):
                overlapping_indices.append(jobj)

        if not overlapping_indices:
            # Isolated: keep baseline defaults
            continue

        overlap_obj[iobj] = True

        max_sma_neighbor = max(geo_overlap[jobj, 2] for jobj in overlapping_indices)
        is_satellite = (sma_i < SATELLITE_FRAC * max_sma_neighbor)

        if is_satellite:
            satellite_obj[iobj] = True

        if not use_radial_weight_for_overlaps:
            # Any overlap => force OFF radial weighting (regardless of global default)
            use_radial_weight_obj[iobj] = False
        else:
            # Overlaps are allowed to follow global default,
            # but satellites always forced OFF.
            if is_satellite:
                use_radial_weight_obj[iobj] = False

        # Tractor geometry only for satellites
        use_tractor_geometry_obj[iobj] = bool(is_satellite)

    return use_radial_weight_obj, use_tractor_geometry_obj, satellite_obj, overlap_obj


def build_multiband_mask(data, tractor, sample, samplesrcs, niter_geometry=2,
                         FMAJOR_geo=0.01, FMAJOR_final=None, ref_factor=2.0,
                         moment_method='rms', maxshift_arcsec=MAXSHIFT_ARCSEC,
                         radial_power=0.7, SATELLITE_FRAC=0.3, mask_minor_galaxies=False,
                         input_geo_initial=None, qaplot=False, mask_nearby=None,
                         use_tractor_position=True, use_radial_weight=True, fixgeo=False,
                         use_radial_weight_for_overlaps=True, use_tractor_geometry=True,
                         cleanup=True, htmlgalaxydir=None):
    """Wrapper to mask out all sources except the galaxy we want to
    ellipse-fit.

    FMAJOR - major if >= XX% of SGA source flux
    moment_method - 'rms' or 'percentile'

    SATELLITE_FRAC - If an SGA source is smaller than SATELLITE_FRAC
    of an overlapping neighbour, treat it as a "satellite" and
    *disable* radial weighting for its moments.

    """
    from astrometry.util.starutil_numpy import arcsec_between
    from SGA.geometry import in_ellipse_mask, ellipses_overlap
    from SGA.util import ivar2var
    from SGA.ellipse import ELLIPSEBIT, ELLIPSEMODE


    def make_sourcemask(srcs, wcs, band, psf, sigma=None, stars=False):
        """Build a model image and threshold mask from a table of
        Tractor sources; also optionally subtract that model from an
        input image.

        """
        from legacypipe.bits import MASKBITS
        from scipy.ndimage.morphology import binary_dilation
        from SGA.coadds import srcs2image

        if stars:
            nsigma = 1.0
        else:
            nsigma = 1.5

        model = srcs2image(srcs, wcs, band=band.lower(), pixelized_psf=psf)
        if sigma:
            mask = model > nsigma*sigma # True=significant flux
            mask = binary_dilation(mask*1, iterations=2) > 0
            #if stars:
            #    B = [(src.maskbits & MASKBITS['BRIGHT'] != 0) | (src.maskbits & MASKBITS['MEDIUM'] != 0) for src in srcs]
        else:
            mask = np.zeros(model.shape, bool)

        return mask, model


    def find_galaxy_in_cutout(img, bx, by, sma, ba, pa, fraction=0.5,
                              factor=1.5, radial_power=0.7, moment_method='rms',
                              wmask=None, use_tractor_position=False,
                              use_radial_weight=True, input_ba_pa=None,
                              debug=False):
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

        if use_tractor_position:
            x0y0 = (bx-x1, by-y1)
        else:
            x0y0 = None

        P = EllipseProperties()
        P.fit(cutout, mask=cutout_mask, method=moment_method,
              percentile=0.95, x0y0=x0y0, smooth_sigma=1.,
              radial_power=radial_power,
              input_ba_pa=input_ba_pa,
              use_radial_weight=use_radial_weight)
        #print(use_radial_weight, use_tractor_position, input_ba_pa, bx, by, P.ba, P.pa, sma)

        if debug:
            import matplotlib.pyplot as plt
            from SGA.qa import overplot_ellipse
            fig, (ax1, ax2) = plt.subplots(1, 2)
            P.plot(image=np.log10(cutout), ax=ax1)
            ax2.imshow(cutout_mask, origin='lower')
            overplot_ellipse(2*(sma*0.262), ba, pa, bx-x1,
                             by-y1, pixscale=0.262,
                             ax=ax1, color='blue')
            fig.savefig('ioannis/tmp/junk.png')
            plt.close()
            pdb.set_trace()

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
                     moment_method='rms', ref_tractor=None,
                     use_sma_mask=False, use_tractor_position=False,
                     props=None):
        """Extract elliptical geometry from either an astropy Table
        (sample), a tractor catalog, or an ellipse_properties object.

        Returns np.array([bx, by, sma, ba, pa]) in *pixel* units.

        """
        def _table_geometry(table):
            # Prefer fully-updated geometry if present
            if table['SMA_MOMENT'] > 0.:
                bx, by = table['BX'], table['BY']
                if use_sma_mask and table['SMA_MASK'] > 0.:
                    sma_arcsec = table['SMA_MASK']
                else:
                    sma_arcsec = table['SMA_MOMENT']
                ba = table['BA_MOMENT']
                pa = table['PA_MOMENT']
            else:
                if use_tractor_position and ref_tractor is not None:
                    bx, by = ref_tractor.bx[0], ref_tractor.by[0]
                else:
                    bx, by = table['BX_INIT'], table['BY_INIT']

                if use_sma_mask:
                    if table['SMA_MASK'] > 0.:
                        sma_arcsec = table['SMA_MASK']
                    elif table['SMA_MOMENT'] > 0.:
                        sma_arcsec = table['SMA_MOMENT']
                    else:
                        sma_arcsec = table['SMA_INIT']
                else:
                    if table['SMA_MOMENT'] > 0.:
                        sma_arcsec = table['SMA_MOMENT']
                    else:
                        sma_arcsec = table['SMA_INIT']

                ba = table['BA_INIT']
                pa = table['PA_INIT']

            sma = sma_arcsec / pixscale  # [pixels]
            return bx, by, sma, ba, pa


        def _tractor_geometry(tractor):
            from SGA.geometry import get_tractor_ellipse
            (bx, by) = tractor.bx, tractor.by
            sma = tractor.shape_r / pixscale # [pixels]
            _, ba, pa = get_tractor_ellipse(sma, tractor.shape_e1, tractor.shape_e2)
            return bx[0], by[0], sma[0], ba[0], pa[0]


        def _props_geometry(props):
            if use_tractor_position and ref_tractor is not None:
                bx, by = ref_tractor.bx[0], ref_tractor.by[0]
            else:
                bx = props.x0
                by = props.y0
            if moment_method == 'rms':
                #sma = props.a # semimajor [pixels]
                sma = 1.75 * props.a # semimajor [pixels]
            else:
                sma = props.a # semimajor [pixels]
            ba = props.ba
            pa = props.pa
            return bx, by, sma, ba, pa


        if table is not None:
            bx, by, sma, ba, pa = _table_geometry(table)
        elif tractor is not None:
            bx, by, sma, ba, pa = _tractor_geometry(tractor)
        elif props is not None:
            bx, by, sma, ba, pa = _props_geometry(props)

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


    def _mask_edges(mask, frac=0.05):
        # Mask a XX% border.
        sz = mask.shape
        edge = int(frac*sz[0])
        mask[:edge, :] = True
        mask[:, :edge] = True
        mask[:, sz[0]-edge:] = True
        mask[sz[0]-edge:, :] = True


    #print('Testing!!')
    #sample['ELLIPSEMODE'] &= ~ELLIPSEMODE['TRACTORGEO']
    #sample['ELLIPSEMODE'] |= ELLIPSEMODE['FIXGEO']
    #sample['ELLIPSEMODE'] |= ELLIPSEMODE['MOMENTPOS']
    #sample['ELLIPSEMODE'] &= ~ELLIPSEMODE['FIXGEO']

    if FMAJOR_final is None:
        FMAJOR_final = FMAJOR_geo

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
        opt_skysigmas[filt] = data[f'{filt}_skysigma']

    if tractor is not None:
        Ipsf = ((tractor.type == 'PSF') * (tractor.type != 'DUP') *
                (tractor.ref_cat != REFCAT) * (tractor.ref_cat != 'LG') *
                (tractor.ref_cat == 'G3'))
        psfsrcs = tractor[Ipsf]

        # Flux-based classification of extended Tractor galaxies.
        Igal = ((tractor.type != 'PSF') * (tractor.type != 'DUP') *
                #(tractor.shape_r > 0.1) *
                (tractor.ref_cat != REFCAT) * (tractor.ref_cat != 'LG'))
        allgalsrcs = tractor[Igal]
    else:
        psfsrcs = []
        allgalsrcs = []

    galsrcs_optflux = np.zeros(len(allgalsrcs), 'f4')
    galsrcs_optsb = np.zeros(len(allgalsrcs), 'f4')
    for j, src in enumerate(allgalsrcs):
        galsrcs_optflux[j] = max([getattr(src, f'flux_{filt}') for filt in opt_bands])
        r50 = getattr(src, 'shape_r')
        if (galsrcs_optflux[j] > 0.) & (r50 > 1e-2):
            galsrcs_optsb[j] = 22.5 - 2.5 * np.log10(galsrcs_optflux[j] / r50)

    # Initialize the *original* images arrays.
    opt_images = np.zeros((len(opt_bands), *sz), 'f4')
    opt_mask_perband = np.stack([data[f'{filt}_mask'] for filt in opt_bands])
    opt_invvar = np.stack([data[f'{filt}_invvar'] for filt in opt_bands])

    opt_maskbits = np.zeros((nsample, *sz), np.int32)
    opt_images_final = np.zeros((nsample, len(opt_bands), *sz), 'f4')
    opt_models = np.zeros((nsample, len(opt_bands), *sz), 'f4')

    # arrays to hold per-object state for final mask building
    opt_refmask_all = np.zeros((nsample, *sz), bool)
    opt_gaiamask_obj_all = np.zeros((nsample, *sz), bool)
    mask_allgals_arr = np.zeros(nsample, bool)

    # shifts relative to initial / Tractor centers
    dshift_arcsec_arr = np.zeros(nsample, 'f4')
    dshift_tractor_arcsec_arr = np.zeros(nsample, 'f4')

    # store table-based initial geometry for later reversion
    geo_initial = np.zeros((nsample, 5))       # [bx,by,sma,ba,pa] (starting geo)
    geo_init_ref_all = np.zeros((nsample, 5))  # table-based geometry fallback
    geo_final = np.zeros_like(geo_initial)

    # Bright-star mask.
    opt_brightstarmask = data['brightstarmask']
    opt_brightstarmask_core = data['brightstarmask_core']

    # Nearby-galaxy mask.
    opt_nearbymask = np.zeros(sz, bool)
    if mask_nearby:
        for mask_one in mask_nearby:
            ba, pa = mask_one['BA'], mask_one['PA']
            sma_mask_nearby = mask_one['DIAM'] * 60. / 2. / opt_pixscale
            (_, bxm, bym) = opt_wcs.wcs.radec2pixelxy(mask_one['RA'], mask_one['DEC'])
            I = in_ellipse_mask(bxm-1., width-(bym-1.), sma_mask_nearby,
                                sma_mask_nearby*ba, pa, xgrid, ygrid_flip)
            opt_nearbymask[I] = True

    # Subtract Gaia stars from all optical images and generate the
    # threshold gaiamask (which will be used unless the LESSMASKING
    # bit is set).
    opt_gaiamask = np.zeros(sz, bool)
    for iband, filt in enumerate(opt_bands):
        if len(psfsrcs) > 0:
            msk, model = make_sourcemask(
                psfsrcs, opt_wcs, filt, data[f'{filt}_psf'],
                data[f'{filt}_skysigma'], stars=True)
            opt_models[:, iband, :, :] += model[np.newaxis, :, :]
            opt_images[iband, :, :] = data[filt] - model
            opt_gaiamask = np.logical_or(opt_gaiamask, msk)
        else:
            opt_images[iband, :, :] = data[filt]


    # Decide on radial weighting and use of Tractor geometry
    use_radial_weight_obj, use_tractor_geometry_obj, satellite_obj, overlap_obj = \
        _get_radial_weight_and_tractor_geometry(
            sample=sample, samplesrcs=samplesrcs, opt_pixscale=opt_pixscale,
            use_tractor_position=use_tractor_position, use_radial_weight=use_radial_weight,
            use_radial_weight_for_overlaps=use_radial_weight_for_overlaps,
            SATELLITE_FRAC=SATELLITE_FRAC, get_geometry=get_geometry,
            ellipses_overlap=ellipses_overlap)

    # Pre-determine which objects will use Tractor or moment geometry.
    use_tractor_position_obj = np.full(nsample, use_tractor_position, dtype=bool)
    for iobj in range(nsample):
        if fixgeo or (sample['ELLIPSEMODE'][iobj] & (ELLIPSEMODE['MOMENTPOS'] | ELLIPSEMODE['FIXGEO']) != 0):
            use_tractor_position_obj[iobj] = False

    # Clear bits which may change between the initial and final
    # geometry. Note: SATELLITE and OVERLAP ELLIPSEBIT bits will be
    # set with the final geometry, below, but we clear them here
    # conservatively.
    sample['ELLIPSEBIT'] &= ~(ELLIPSEBIT['BLENDED'] | ELLIPSEBIT['MAJORGAL'] |
                              ELLIPSEBIT['OVERLAP'] | ELLIPSEBIT['SATELLITE'])
    for iobj in range(nsample):
        # If TRACTORGEO is set, force Tractor-based geometry for this object.
        if (sample['ELLIPSEMODE'][iobj] & ELLIPSEMODE['TRACTORGEO']) != 0:
            use_tractor_geometry_obj[iobj] = True
            # Geometry is coming from Tractor, so radial weighting is irrelevant.
            use_radial_weight_obj[iobj] = False

        # If NORADWEIGHT is set, suppress radial weighting, regardless
        # of overlap logic.
        if (sample['ELLIPSEMODE'][iobj] & ELLIPSEMODE['NORADWEIGHT']) != 0:
            use_radial_weight_obj[iobj] = False

        if not use_radial_weight_obj[iobj]:
            sample['ELLIPSEBIT'][iobj] |= ELLIPSEBIT['NORADWEIGHT']
        if use_tractor_geometry_obj[iobj]:
            sample['ELLIPSEBIT'][iobj] |= ELLIPSEBIT['TRACTORGEO']

        # Set MOMENTPOS bit: using moment positions (not Tractor, not fixed)
        if (not use_tractor_position_obj[iobj] and
            (sample['ELLIPSEMODE'][iobj] & ELLIPSEMODE['FIXGEO'] == 0) and
            not fixgeo):
            sample['ELLIPSEBIT'][iobj] |= ELLIPSEBIT['MOMENTPOS']

        if fixgeo or (sample['ELLIPSEMODE'][iobj] & ELLIPSEMODE['FIXGEO'] != 0):
            sample['ELLIPSEBIT'][iobj] |= ELLIPSEBIT['FIXGEO']

        if sample['ELLIPSEMODE'][iobj] & ELLIPSEMODE['LESSMASKING'] != 0:
            sample['ELLIPSEBIT'][iobj] |= ELLIPSEBIT['LESSMASKING']

        if sample['ELLIPSEMODE'][iobj] & ELLIPSEMODE['MOREMASKING'] != 0:
            sample['ELLIPSEBIT'][iobj] |= ELLIPSEBIT['MOREMASKING']


    # Minimum semi-major axis used for masks (not for stored geometry).
    SMA_MASK_MIN_ARCSEC = 10. # [arcsec]
    SMA_MASK_MIN_PIX = SMA_MASK_MIN_ARCSEC / opt_pixscale

    # are we allowed to change the geometry in this call?
    update_geometry = input_geo_initial is None

    # iterate to get the geometry
    for iobj, (obj, objsrc) in enumerate(zip(sample, samplesrcs)):
        log.info('Determining the geometry for galaxy ' +
                 f'{iobj+1}/{nsample}.')

        msg = _log_object_modes(log, iobj, obj, use_radial_weight_obj[iobj],
                                use_tractor_geometry_obj[iobj], ELLIPSEMODE,
                                ELLIPSEBIT, stage="initial")
        for msg1 in msg:
            log.info(f'  {msg1}')

        # If the LESSMASKING bit is set, do not use the Gaia threshold
        # mask.
        opt_gaiamask_obj = np.copy(opt_gaiamask)
        if obj['ELLIPSEMODE'] & ELLIPSEMODE['LESSMASKING'] != 0:
            #log.info('LESSMASKING bit set; no Gaia threshold-masking.')
            opt_gaiamask_obj[:, :] = False

        ## Possibly deprected - If the MOREMASKING bit is set, mask
        ## all extended sources, whether or not they're inside the
        ## elliptical mask.
        #if obj['ELLIPSEMODE'] & ELLIPSEMODE['MOREMASKING'] != 0:
        #    #log.info('MOREMASKING bit set; masking all extended sources.')
        #    mask_allgals_arr[iobj] = True

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
            # For *previously* completed objects, use the final, not
            # initial geometry.
            if indx < iobj:
                [bxr, byr, smar_moment, bar, par] = geo_final[indx, :]
                # Use SMA_MASK for masking, not moment SMA
                smar = max(max(smar_moment, SMA_MASK_MIN_PIX), sample['SMA_MASK'][indx] / opt_pixscale)
            else:
                # always use get_geometry with use_sma_mask=True for
                # future refs; don't use input_geo_initial for
                # reference objects.
                if use_tractor_geometry and use_tractor_geometry_obj[indx]:
                    if refsrc is None:
                        bxr, byr, smar, bar, par = get_geometry(
                            opt_pixscale, table=refsample,
                            use_sma_mask=True)
                    else:
                        bxr, byr, smar, bar, par = get_geometry(
                            opt_pixscale, tractor=refsrc)
                else:
                    bxr, byr, smar, bar, par = get_geometry(
                        opt_pixscale, table=refsample, ref_tractor=refsrc,
                        use_tractor_position=use_tractor_position_obj[indx],
                        use_sma_mask=True)

            opt_refmask1 = in_ellipse_mask(bxr, width-byr, smar*ref_factor,
                                           bar*smar*ref_factor,
                                           par, xgrid, ygrid_flip)
            opt_refmask = np.logical_or(opt_refmask, opt_refmask1)

            for iband, filt in enumerate(opt_bands):
                _, model = make_sourcemask(
                    refsrc, opt_wcs, filt, data[f'{filt}_psf'],
                    sigma=None)
                opt_images_obj[iband, :, :] = opt_images_obj[iband, :, :] - model
                opt_models[iobj, iband, :, :] += model

        # Initial geometry and elliptical mask.
        if input_geo_initial is not None:
            # Fallback: table-based geometry (same as first pass)
            geo_init_ref = get_geometry(opt_pixscale, table=obj,
                                        use_sma_mask=False)
            # Starting geometry for this pass (e.g. R26 based)
            geo_init = input_geo_initial[iobj, :].copy()
        else:
            # initial geometry used as fallback
            geo_init_ref = get_geometry(opt_pixscale, table=obj,
                                        use_sma_mask=False)
            geo_init = get_geometry(
                opt_pixscale, table=obj, ref_tractor=objsrc,
                use_tractor_position=use_tractor_position_obj[iobj],
                use_sma_mask=False)


        geo_initial[iobj, :] = geo_init
        geo_init_ref_all[iobj, :] = geo_init_ref

        [bx, by, sma, ba, pa] = geo_init

        # Next, iteratively update the source geometry unless
        # FIXGEO or TRACTORGEO have been set.
        if update_geometry:
            if fixgeo or (obj['ELLIPSEMODE'] & (ELLIPSEMODE['FIXGEO'] | ELLIPSEMODE['TRACTORGEO']) != 0):
                niter_actual = 1
            else:
                niter_actual = niter_geometry
        else:
            niter_actual = 1

        dshift_arcsec = 0.0
        dshift_tractor_arcsec = 0.0

        sma_floor_pix = obj['SMA_MASK'] / opt_pixscale

        for iiter in range(niter_actual):
            #log.info(f'  Iteration {iiter+1}/{niter_actual}:')
            bx_init, by_init, sma_init, ba_init, pa_init = \
                np.copy(bx), np.copy(by), np.copy(sma), np.copy(ba), np.copy(pa)

            # Use a minimum radius for masking but also make sure mask
            # doesn't shrink below our initial estimate of R26 in the
            # second pass.
            sma_mask = max(max(sma, SMA_MASK_MIN_PIX), sma_floor_pix)

            # initialize (or update) the in-ellipse mask
            inellipse = in_ellipse_mask(bx, width-by, sma_mask, ba*sma_mask,
                                        pa, xgrid, ygrid_flip)

            # Zero out bright-star and reference pixels within the
            # current ellipse mask of the current object...
            iter_brightstarmask = np.copy(opt_brightstarmask)
            iter_refmask = np.copy(opt_refmask)
            iter_brightstarmask[inellipse] = False
            iter_refmask[inellipse] = False

            # Expand the brightstarmask veto if the NEARSTAR or GCLPNE
            # bits are set (factor of XX), with a floor for very small
            # galaxies.
            if (obj['SAMPLE'] & (SAMPLE['INSTAR'] | SAMPLE['NEARSTAR'] | SAMPLE['GCLPNE'])) != 0:
                if 1.5*sma_mask*opt_pixscale < 10.: # [arcsec]
                    sma_veto = 10. / opt_pixscale
                else:
                    sma_veto = 1.5 * sma_mask
                inellipse2 = in_ellipse_mask(bx, width-by, sma_veto, sma_veto*ba,
                                             pa, xgrid, ygrid_flip)
                iter_brightstarmask[inellipse2] = False
            else:
                inellipse2 = inellipse # used below

            # never veto the "core" brightstarmask except for FIXGEO
            # or TRACTORGEO
            if not fixgeo and (obj['ELLIPSEMODE'] & (ELLIPSEMODE['FIXGEO'] | ELLIPSEMODE['TRACTORGEO']) == 0):
                iter_brightstarmask |= opt_brightstarmask_core

            # if more than XX% of the pixels are masked by the core of
            # the star, fall back to the initial geometry and fail
            #plt.imshow(iter_brightstarmask, origin='lower')
            #import matplotlib.pyplot as plt
            #plt.clf()
            #plt.imshow(inellipse2, origin='lower')
            #plt.savefig('ioannis/tmp/junk.png')
            denom = iter_brightstarmask[inellipse2].size
            if denom > 0: # should always be true...
                frac = np.sum(iter_brightstarmask[inellipse2]) / denom
                if frac > 0.3:
                    log.warning(f'Nearly fully masked by bright-star core (F={100.*frac:.1f}%>30%); reverting to initial geometry.')
                    sample['ELLIPSEBIT'][iobj] |= ELLIPSEBIT['FAILGEO']
                    geo_iter = geo_init.copy()
                    [bx, by, sma, ba, pa] = geo_iter
                    break

            # mask edges aggressively to not bias our 'moment'
            # geometry *after* setting FAILGEO
            _mask_edges(iter_brightstarmask)

            # Build a galaxy mask from extended sources, split into
            # "major" and "minor" based on flux ratio relative to the
            # SGA source.
            opt_galmask = np.zeros(sz, bool)

            if mask_allgals_arr[iobj]:
                _, opt_galmask, _ = update_galmask(
                    allgalsrcs, bx, by, sma_mask, ba, pa,
                    opt_skysigmas=opt_skysigmas, opt_models=None,
                    mask_allgals=True)
            else:
                flux_sga = sample['OPTFLUX'][iobj]
                fracflux_sga = sample['FRACFLUX'][iobj]
                major_mask, minor_mask = _compute_major_minor_masks(
                    flux_sga, fracflux_sga, allgalsrcs, galsrcs_optflux,
                    FMAJOR_geo, objsrc, use_tractor_position_obj[iobj],
                    arcsec_between)

                # Major companions: mask their flux everywhere (inside and out).
                if np.any(major_mask):
                    _, galmask_major, _ = update_galmask(
                        allgalsrcs[major_mask], bx, by,
                        sma_mask, ba, pa, opt_skysigmas=opt_skysigmas,
                        opt_models=None, mask_allgals=True)
                    opt_galmask = np.logical_or(opt_galmask, galmask_major)

                # Minor companions: use the original "outside-ellipse only" logic.
                if np.any(minor_mask) and mask_minor_galaxies:
                    _, galmask_minor, _ = update_galmask(
                        allgalsrcs[minor_mask], bx, by,
                        sma_mask, ba, pa, opt_skysigmas=opt_skysigmas,
                        opt_models=None, mask_allgals=False)
                    opt_galmask = np.logical_or(opt_galmask, galmask_minor)

                # Optionally do not mask within the current SGA ellipse itself.
                opt_galmask[inellipse] = False

            # apply the mask_nearby mask
            opt_galmask = np.logical_or(opt_galmask, opt_nearbymask)

            # Combine opt_brightstarmask, opt_gaiamask, opt_refmask,
            # and opt_galmask with the per-band optical masks.
            opt_masks_obj = _update_masks(iter_brightstarmask, opt_gaiamask_obj, iter_refmask,
                                          opt_galmask, opt_mask_perband, opt_bands,
                                          sz, verbose=False)

            # Generate a detection image and pixel mask for use with
            # find_galaxy_in_cutout even if we're not updating the
            # geometry.
            wimg = np.sum(opt_invvar * np.logical_not(opt_masks_obj) * opt_images_obj, axis=0)
            wnorm = np.sum(opt_invvar * np.logical_not(opt_masks_obj), axis=0)
            wimg[wnorm > 0.] /= wnorm[wnorm > 0.]

            wmasks = np.zeros_like(opt_images_obj, bool)
            for iband, filt in enumerate(opt_bands):
                wmasks[iband, :, :] = ((~opt_masks_obj[iband, :, :]) * \
                                       (opt_images_obj[iband, :, :] > opt_skysigmas[filt]))
            # True=any pixel is >5*skynoise and positive in the
            # coadded image.
            wmask = np.any(wmasks, axis=0) * (wimg > 0.)

            if fixgeo or (obj['ELLIPSEMODE'] & ELLIPSEMODE['FIXGEO'] != 0):
                log.info('FIXGEO bit set; fixing the elliptical geometry.')
                geo_iter = geo_init
            elif (obj['ELLIPSEMODE'] & ELLIPSEMODE['TRACTORGEO']) != 0 and objsrc is not None:
                log.info('TRACTORGEO bit set; fixing the elliptical geometry.')
                geo_iter = get_geometry(opt_pixscale, tractor=objsrc)
            elif not update_geometry:
                # Recompute sma_moment with the updated mask even when
                # not updating the geometry; fix bx, by, ba, and pa to
                # their previously determined values. NB: set
                # use_tractor_position=True as a trick to fix (bx,by);
                # this isn't necessarily the Tractor position).
                props = find_galaxy_in_cutout(
                    wimg, bx, by, sma_mask, ba, pa, wmask=wmask,
                    moment_method=moment_method, input_ba_pa=(ba, pa),
                    radial_power=radial_power,
                    use_radial_weight=use_radial_weight_obj[iobj],
                    use_tractor_position=True)
                _, _, sma_new, _, _ = get_geometry(
                    opt_pixscale, props=props, ref_tractor=objsrc,
                    moment_method=moment_method,
                    use_tractor_position=use_tractor_position_obj[iobj])
                geo_iter = geo_init
                geo_iter[2] = sma_new
            else:
                # Optionally use Tractor for small overlapping satellites.
                if use_tractor_geometry and use_tractor_geometry_obj[iobj]:
                    if objsrc is None or objsrc.type == 'PSF':
                        input_ba_pa = None
                    else:
                        _, _, _, ba_tr, pa_tr = get_geometry(opt_pixscale, tractor=objsrc)
                        input_ba_pa = (ba_tr, pa_tr)
                else:
                    input_ba_pa = None

                props = find_galaxy_in_cutout(
                    wimg, bx, by, sma_mask, ba, pa, wmask=wmask,
                    moment_method=moment_method, input_ba_pa=input_ba_pa,
                    radial_power=radial_power,
                    use_radial_weight=use_radial_weight_obj[iobj],
                    use_tractor_position=use_tractor_position_obj[iobj])

                geo_iter = get_geometry(
                    opt_pixscale, props=props, ref_tractor=objsrc,
                    moment_method=moment_method,
                    use_tractor_position=use_tractor_position_obj[iobj])

            if update_geometry:
                ra_iter, dec_iter = opt_wcs.wcs.pixelxy2radec(geo_iter[0] + 1., geo_iter[1] + 1.)

                dshift_arcsec = arcsec_between(obj['RA_INIT'], obj['DEC_INIT'], ra_iter, dec_iter)
                if objsrc is not None:
                    dshift_tractor_arcsec = arcsec_between(objsrc.ra, objsrc.dec, ra_iter, dec_iter)

                if dshift_arcsec > maxshift_arcsec:
                    log.warning(f'Large shift for iobj={iobj} ({obj[REFIDCOLUMN]}): delta=' +
                                f'{dshift_arcsec:.3f}>{maxshift_arcsec:.3f} arcsec')

            # update the geometry for the next iteration
            [bx, by, sma, ba, pa] = geo_iter

            log.info(f'  Iteration {iiter+1}/{niter_actual}: (bx,by)=({bx_init:.1f},{by_init:.1f})-->({bx:.1f},{by:.1f}) ' + \
                     f'b/a={ba_init:.2f}-->{ba:.2f} PA={pa_init:.1f}-->{pa:.1f} degree ' + \
                     f'sma={sma_init*opt_pixscale:.2f}-->{sma*opt_pixscale:.2f} arcsec ' + \
                     f'[sma_mask={sma_mask*opt_pixscale:.2f} arcsec]')

            # Validate geometry; revert to initial if invalid
            geometry_failed = False
            if sma <= 0.:
                log.warning(f'Semi-major axis for {obj["OBJNAME"]} is zero or negative; reverting to initial geometry.')
                geometry_failed = True
            elif (ba < 1e-2) or (ba > 1.):
                log.warning(f'Ellipticity b/a={ba:.3f} is unphysical for {obj["OBJNAME"]}; reverting to initial geometry.')
                geometry_failed = True
            elif (pa < 0.) or (pa > 180.):
                log.warning(f'Position angle PA={pa:.1f} is out of bounds for {obj["OBJNAME"]}; reverting to initial geometry.')
                geometry_failed = True

            if geometry_failed:
                sample['ELLIPSEBIT'][iobj] |= ELLIPSEBIT['FAILGEO']
                geo_iter = geo_init.copy()
                [bx, by, sma, ba, pa] = geo_iter
                break

        # store shifts
        dshift_arcsec_arr[iobj] = dshift_arcsec
        dshift_tractor_arcsec_arr[iobj] = dshift_tractor_arcsec

        # store images, geometry, and masks needed later
        opt_images_final[iobj, :, :, :] = opt_images_obj
        geo_final[iobj, :] = geo_iter  # last iteration
        opt_refmask_all[iobj, :, :] = opt_refmask
        opt_gaiamask_obj_all[iobj, :, :] = opt_gaiamask_obj

        # Store SMA_MASK for this object so subsequent objects can use
        # it for building their reference masks.
        [bx_final, by_final, sma_final, ba_final, pa_final] = geo_iter
        sma_floor_pix = obj['SMA_MASK'] / opt_pixscale
        sma_mask_final = max(max(sma_final, SMA_MASK_MIN_PIX), sma_floor_pix)
        sample['SMA_MASK'][iobj] = sma_mask_final * opt_pixscale  # [arcsec]

    if update_geometry:
        # enforce minimum separation between centers
        ra_final, dec_final = opt_wcs.wcs.pixelxy2radec(
            (geo_final[:, 0] + 1.), (geo_final[:, 1] + 1.))

        for iobj in range(nsample):
            for j in range(iobj+1, nsample):
                sep = arcsec_between(
                    ra_final[iobj], dec_final[iobj],
                    ra_final[j], dec_final[j])
                if sep < maxshift_arcsec:
                    log.warning(f'Objects {iobj} and {j} converged to nearly the same center '
                                f'({sep:.2f} < {maxshift_arcsec} arcsec); reverting both to '
                                'input centers.')
                    # revert both centers to table-based geometry
                    geo_final[iobj, 0] = geo_init_ref_all[iobj, 0]
                    geo_final[iobj, 1] = geo_init_ref_all[iobj, 1]
                    geo_final[j, 0] = geo_init_ref_all[j, 0]
                    geo_final[j, 1] = geo_init_ref_all[j, 1]

                    ra_final[iobj], dec_final[iobj] = opt_wcs.wcs.pixelxy2radec(
                        geo_final[iobj, 0] + 1., geo_final[iobj, 1] + 1.)
                    ra_final[j], dec_final[j] = opt_wcs.wcs.pixelxy2radec(
                        geo_final[j, 0] + 1., geo_final[j, 1] + 1.)

        # set LARGESHIFT bits
        for iobj, objsrc in enumerate(samplesrcs):
            if dshift_arcsec_arr[iobj] > maxshift_arcsec:
                sample['ELLIPSEBIT'][iobj] |= ELLIPSEBIT['LARGESHIFT']
            if objsrc is not None:
                if dshift_tractor_arcsec_arr[iobj] > maxshift_arcsec:
                    sample['ELLIPSEBIT'][iobj] |= ELLIPSEBIT['LARGESHIFT_TRACTOR']


        # Rebuild reference masks using *final* geometry for all SGA sources.
        opt_refmask_all[:, :, :] = False # clear everything from the first pass

        for iobj in range(nsample):
            bx_i, by_i, sma_i, ba_i, pa_i = geo_final[iobj, :]

            # Mask all *other* SGA sources using their final ellipses
            refmask_i = np.zeros(sz, bool)
            for j in range(nsample):
                if j == iobj:
                    continue

                bx_j, by_j, sma_j_moment, ba_j, pa_j = geo_final[j, :]
                # Use SMA_MASK for masking, not moment SMA
                sma_j = max(max(sma_j_moment, SMA_MASK_MIN_PIX), sample['SMA_MASK'][j] / opt_pixscale)
                if sma_j <= 0:
                    continue

                refmask_j = in_ellipse_mask(bx_j, width - by_j, sma_j * ref_factor,
                    ba_j*sma_j*ref_factor, pa_j, xgrid, ygrid_flip)
                refmask_i |= refmask_j

            opt_refmask_all[iobj, :, :] = refmask_i

    # final optical masks, quantities, and bits
    ra, dec = opt_wcs.wcs.pixelxy2radec((geo_final[:, 0]+1.), (geo_final[:, 1]+1.))
    sample['RA'] = ra
    sample['DEC'] = dec
    sample['SGANAME'] = sga2025_name(ra, dec)
    if nsample > 1:
        if len(sample['SGANAME']) != len(np.unique(sample['SGANAME'])):
            msg = f'Duplicate SGA names {sample['SGANAME'][0]}'
            log.critical(msg)
            raise ValueError(msg)

    # Bits that must be cleared because they depend on final geometry.
    sample['ELLIPSEBIT'] &= ~(ELLIPSEBIT['NORADWEIGHT'] | ELLIPSEBIT['TRACTORGEO'] |
                              ELLIPSEBIT['OVERLAP'] | ELLIPSEBIT['SATELLITE'] |
                              ELLIPSEBIT['BLENDED'] | ELLIPSEBIT['MAJORGAL'])

    log.info('Final geometry:')
    for iobj, (obj, objsrc) in enumerate(zip(sample, samplesrcs)):

        [bx, by, sma, ba, pa] = geo_final[iobj, :]

        sma_floor_pix = obj['SMA_MASK'] / opt_pixscale
        sma_mask = max(max(sma, SMA_MASK_MIN_PIX), sma_floor_pix)

        log.info(f'  Galaxy {iobj+1}/{nsample}: (bx,by)=({bx:.1f},{by:.1f}) b/a={ba:.2f} PA={pa:.1f} degree ' + \
                 f'sma={sma*opt_pixscale:.2f} arcsec [sma_mask={sma_mask*opt_pixscale:.2f} arcsec]')

        sample['BX'][iobj] = bx
        sample['BY'][iobj] = by
        sample['SMA_MOMENT'][iobj] = sma * opt_pixscale    # [arcsec]
        sample['SMA_MASK'][iobj] = sma_mask * opt_pixscale # [arcsec]
        sample['BA_MOMENT'][iobj] = ba
        sample['PA_MOMENT'][iobj] = pa

        inellipse = in_ellipse_mask(bx, width-by, sma_mask, sma_mask*ba,
                                    pa, xgrid, ygrid_flip)
        final_brightstarmask = np.copy(opt_brightstarmask)
        final_refmask = np.copy(opt_refmask_all[iobj])
        final_brightstarmask[inellipse] = False
        final_refmask[inellipse] = False

        if (sample['SAMPLE'][iobj] & (SAMPLE['INSTAR'] | SAMPLE['NEARSTAR'] | SAMPLE['GCLPNE'])) != 0:
            if 1.5*sma_mask*opt_pixscale < 10.: # [arcsec]
                sma_veto = 10. / opt_pixscale
            else:
                sma_veto = 1.5 * sma_mask
            inellipse2 = in_ellipse_mask(bx, width-by, sma_veto, sma_veto*ba,
                                         pa, xgrid, ygrid_flip)
            final_brightstarmask[inellipse2] = False

        # never veto the "core" brightstarmask except for FIXGEO
        # or TRACTORGEO
        if not fixgeo and (sample['ELLIPSEMODE'][iobj] & (ELLIPSEMODE['FIXGEO'] | ELLIPSEMODE['TRACTORGEO']) == 0):
            final_brightstarmask |= opt_brightstarmask_core

        # Build the final galaxy mask
        opt_galmask = np.zeros(sz, bool)
        opt_models_obj = opt_models[iobj, :, :, :]
        if mask_allgals_arr[iobj]:
            _, opt_galmask, opt_models_obj = update_galmask(
                allgalsrcs, bx, by, sma_mask, ba, pa,
                opt_models=opt_models_obj,
                opt_skysigmas=opt_skysigmas,
                mask_allgals=True)
        else:
            flux_sga = sample['OPTFLUX'][iobj]
            fracflux_sga = sample['FRACFLUX'][iobj]
            major_mask, minor_mask = _compute_major_minor_masks(
                flux_sga, fracflux_sga, allgalsrcs, galsrcs_optflux,
                FMAJOR_final, objsrc, use_tractor_position_obj[iobj],
                arcsec_between)

            if np.any(major_mask):
                _, galmask_major, opt_models_obj = update_galmask(
                    allgalsrcs[major_mask], bx, by,
                    sma_mask, ba, pa, opt_models=opt_models_obj,
                    opt_skysigmas=opt_skysigmas,
                    mask_allgals=True)
                opt_galmask = np.logical_or(opt_galmask, galmask_major)

            #import matplotlib.pyplot as plt
            #plt.clf()
            #plt.imshow(opt_galmask, origin='lower')
            #plt.savefig('ioannis/tmp/junk.png')

            #import matplotlib.pyplot as plt
            #plt.clf()
            #plt.scatter(objsrc.bx, objsrc.by, s=50, marker='x', label=f'SGA {iobj}')
            #plt.scatter(allgalsrcs.bx, allgalsrcs.by, s=10, label='All galaxies')
            #plt.scatter(allgalsrcs.bx[major_mask], allgalsrcs.by[major_mask], s=25, color='red', label='Major galaxies')
            #plt.scatter(allgalsrcs.bx[minor_mask], allgalsrcs.by[minor_mask], s=25, color='blue', label='Minor galaxies')
            #plt.legend()
            #plt.savefig('ioannis/tmp/junk.png')
            if np.any(minor_mask) and mask_minor_galaxies:
                _, galmask_minor, opt_models_obj = update_galmask(
                    allgalsrcs[minor_mask], bx, by,
                    sma_mask, ba, pa, opt_models=opt_models_obj,
                    opt_skysigmas=opt_skysigmas,
                    mask_allgals=False)
                opt_galmask = np.logical_or(opt_galmask, galmask_minor)

            # Optionally do not mask within the current SGA ellipse itself.
            #import matplotlib.pyplot as plt
            #plt.clf()
            #plt.imshow(opt_models_obj[1, :, :], origin='lower')
            #plt.savefig('ioannis/tmp/junk.png')
            opt_galmask[inellipse] = False

        # apply the mask_nearby mask
        opt_galmask = np.logical_or(opt_galmask, opt_nearbymask)

        opt_maskbits_obj = _update_masks(
            final_brightstarmask, opt_gaiamask_obj_all[iobj],
            final_refmask, opt_galmask, opt_mask_perband,
            opt_bands, sz, build_maskbits=True,
            MASKDICT=OPTMASKBITS)
        opt_models[iobj, :, :, :] = opt_models_obj
        opt_maskbits[iobj, :, :] = opt_maskbits_obj

        # Set the ELLIPSEBIT bits.
        if not use_radial_weight_obj[iobj]:
            sample['ELLIPSEBIT'][iobj] |= ELLIPSEBIT['NORADWEIGHT']

        if use_tractor_geometry and use_tractor_geometry_obj[iobj]:
            sample['ELLIPSEBIT'][iobj] |= ELLIPSEBIT['TRACTORGEO']

        # Overlap bit. Note! use sma_mask because on the second pass
        # it is very close to the (final) R(26) value.
        overlapping_indices = []
        for jobj in range(nsample):
            if jobj == iobj:
                continue
            bx_j, by_j, sma_j, ba_j, pa_j = geo_final[jobj, :]
            sma_j = max(max(sma_j, SMA_MASK_MIN_PIX), sample['SMA_MASK'][jobj] / opt_pixscale)
            if sma_j <= 0:
                continue
            if ellipses_overlap(bx, by, sma_mask, ba, pa,
                                bx_j, by_j, sma_j, ba_j, pa_j):
                overlapping_indices.append(jobj)

        #print(iobj, overlapping_indices, bx, by, sma, ba, pa,
        #      bx_j, by_j, sma_j, ba_j, pa_j)
        #import matplotlib.pyplot as plt
        #plt.clf()
        #plt.imshow(in_ellipse_mask(bx, width-by, sma, sma*ba, pa, xgrid, ygrid_flip), origin='lower')
        #plt.savefig('ioannis/tmp/junk.png')

        # Overlap bit -- any part of this galaxy's ellipse overlaps
        # any part of any other galaxy.
        if len(overlapping_indices) > 0:
            sample['ELLIPSEBIT'][iobj] |= ELLIPSEBIT['OVERLAP']

        # Satellite bit -- use sma_moment here, not sma_mask
        if len(overlapping_indices) > 0:
            max_sma_neighbor = max(geo_final[jobj, 2] for jobj in overlapping_indices)
            if sma < SATELLITE_FRAC * max_sma_neighbor:
                sample['ELLIPSEBIT'][iobj] |= ELLIPSEBIT['SATELLITE']

        # Blended bit -- center of this galaxy is inside the ellipse
        # of another galaxy.
        refindx = np.delete(np.arange(nsample), iobj)
        for indx in refindx:
            [refbx, refby, refsma, refba, refpa] = geo_final[indx, :]
            refsma = max(max(refsma, SMA_MASK_MIN_PIX), sample['SMA_MASK'][indx] / opt_pixscale)
            center_inside = in_ellipse_mask(refbx, width-refby, refsma, refsma*refba,
                                            refpa, bx, width-by)
            if center_inside:
                sample['ELLIPSEBIT'][iobj] |= ELLIPSEBIT['BLENDED']
                break

        # Was the Tractor position used?
        flux_sga = sample['OPTFLUX'][iobj]
        if flux_sga > 0 and len(allgalsrcs) > 0:
            fracflux_sga = sample['FRACFLUX'][iobj]
            major_mask, _ = _compute_major_minor_masks(
                flux_sga, fracflux_sga, allgalsrcs, galsrcs_optflux,
                FMAJOR_final, objsrc, use_tractor_position_obj[iobj],
                arcsec_between)

            if np.any(major_mask):
                # Are any major companions’ centers inside this ellipse?
                inside = in_ellipse_mask(bx, width-by, sma_mask, sma_mask*ba,
                                         pa, allgalsrcs[major_mask].bx,
                                         width - allgalsrcs[major_mask].by)
                if np.any(inside):
                    sample['ELLIPSEBIT'][iobj] |= ELLIPSEBIT['MAJORGAL']

        msg = _log_object_modes(log, iobj, sample[iobj], use_radial_weight_obj[iobj],
                                use_tractor_geometry_obj[iobj], ELLIPSEMODE, ELLIPSEBIT,
                                stage="final")
        for msg1 in msg:
            log.info(f'    {msg1}')


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
                brightstarmask, gaiamask, refmask, galmask,
                mask_perband, bands, sz, build_maskbits=True, MASKDICT=MASKDICT,
                do_resize=True)

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

    # optionally build a QA figure
    if qaplot:
        qa_multiband_mask(data, sample, htmlgalaxydir=htmlgalaxydir)

    # clean-up
    if cleanup:
        del data['brightstarmask']
        del data['brightstarmask_core']
        for filt in all_data_bands:
            del data[filt]
            for col in ['psf', 'mask']:
                del data[f'{filt}_{col}']
        for filt in opt_bands:
            for col in ['skysigma']:
                del data[f'{filt}_{col}']

    return data, sample


def read_multiband(galaxy, galaxydir, REFIDCOLUMN, bands=['g', 'r', 'i', 'z'],
                   sort_by_flux=True, run='south', pixscale=0.262,
                   galex_pixscale=1.5, unwise_pixscale=2.75,
                   galex=True, unwise=True, verbose=False, read_jpg=False,
                   skip_ellipse=False, skip_tractor=False):
    """Read the multi-band images (converted to surface brightness) in
    preparation for ellipse-fitting.

    """
    import fitsio
    from matplotlib.image import imread
    from astropy.table import Table
    from astrometry.util.fits import fits_table
    from astrometry.util.util import Tan
    from legacypipe.bits import MASKBITS

    from SGA.util import mwdust_transmission
    from SGA.io import _read_image_data
    from SGA.ellipse import ELLIPSEBIT

    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None


    def _read_sample(opt_refband, tractor):
        # Read the sample catalog from custom_coadds and find each source
        # in the Tractor catalog.
        samplefile = os.path.join(galaxydir, f'{galaxy}-{filt2imfile["sample"]}.fits')
        sample = Table(fitsio.read(samplefile))#, columns=cols))
        log.info(f'Read {len(sample)} source(s) from {samplefile}')
        for col in ['RA', 'DEC', 'DIAM', 'PA', 'BA', 'MAG']:
            sample.rename_column(col, f'{col}_INIT')
        #print('###########################')
        #print('Hack!')
        #sample['DIAM_INIT'] = 1.1
        #sample['PA_INIT'] = 0.0
        #sample['BA_INIT'] = 1.0
        sample.rename_column('DIAM_REF', 'DIAM_INIT_REF')
        sample.add_column(sample['DIAM_INIT']*60./2., name='SMA_INIT', # [radius, arcsec]
                          index=np.where(np.array(sample.colnames) == 'DIAM_INIT')[0][0])

        # populate (BX,BY)_INIT by quickly building the WCS
        wcs = Tan(filt2imfile[opt_refband]['image'], 1)
        (_, x0, y0) = wcs.radec2pixelxy(sample['RA_INIT'].value, sample['DEC_INIT'].value)

        sample.add_column((x0 - 1.).astype('f4'), name='BX_INIT',
                          index=np.where(np.array(sample.colnames) == 'FITMODE')[0][0]+1)  # NB the -1!
        sample.add_column((y0 - 1.).astype('f4'), name='BY_INIT',
                          index=np.where(np.array(sample.colnames) == 'BX_INIT')[0][0]+1)  # NB the -1!
        #sample['BY_INIT'] = (y0 - 1.).astype('f4')

        sample['OPTFLUX'] = np.zeros(len(sample), 'f4') # brightest band
        sample['FRACFLUX'] = np.zeros(len(sample), 'f4') # brightest band

        # optical bands
        sample['BANDS'] = np.zeros(len(sample), f'<U{len(bands)}')
        sample['BANDS'] = ''.join(data['opt_bands'])

        # moment geometry
        sample['SGANAME'] = np.zeros(len(sample), '<U25')
        sample['RA'] = np.zeros(len(sample), 'f8')
        sample['DEC'] = np.zeros(len(sample), 'f8')
        sample['BX'] = np.zeros(len(sample), 'f4')
        sample['BY'] = np.zeros(len(sample), 'f4')
        sample['SMA_MASK'] = np.zeros(len(sample), 'f4') # [arcsec]
        sample['SMA_MOMENT'] = np.zeros(len(sample), 'f4') # [arcsec]
        sample['BA_MOMENT'] = np.zeros(len(sample), 'f4')
        sample['PA_MOMENT'] = np.zeros(len(sample), 'f4')
        sample['RA_TRACTOR'] = np.zeros(len(sample), 'f8')
        sample['DEC_TRACTOR'] = np.zeros(len(sample), 'f8')

        # initialize the ELLIPSEBIT bitmask
        sample['ELLIPSEBIT'] = np.zeros(len(sample), np.int32)

        if tractor is None:
            samplesrcs = [None] * len(sample)
            sample['ELLIPSEBIT'] |= ELLIPSEBIT['SKIPTRACTOR']
        else:
            samplesrcs = []
            for iobj, refid in enumerate(sample[REFIDCOLUMN].value):
                I = np.where(np.logical_or(tractor.ref_cat == REFCAT, tractor.ref_cat == 'LG') *
                             (tractor.ref_id == refid))[0]
                if len(I) == 0:
                    log.warning(f'ref_id={refid} dropped by Tractor')
                    sample['ELLIPSEBIT'][iobj] |= ELLIPSEBIT['NOTRACTOR']
                    samplesrcs.append(None)
                else:
                    samplesrcs.append(tractor[I])
                    if tractor[I[0]].type in ['PSF', 'DUP']:
                        log.warning(f'ref_id={refid} fit by Tractor as PSF (or DUP)')
                        sample['ELLIPSEBIT'][iobj] |= ELLIPSEBIT['TRACTORPSF']
                    sample['OPTFLUX'][iobj] = max([getattr(tractor[I[0]], f'flux_{filt}')
                                                   for filt in opt_bands])
                    sample['FRACFLUX'][iobj] = max([getattr(tractor[I[0]], f'fracflux_{filt}')
                                                   for filt in opt_bands])
                    sample['RA_TRACTOR'][iobj] = tractor[I[0]].ra
                    sample['DEC_TRACTOR'][iobj] = tractor[I[0]].dec

        # Sort by initial diameter or optical brightness (in any band).
        if sort_by_flux:
            log.info('Sorting by optical flux:')
            srt = np.argsort(sample['OPTFLUX'])[::-1]
        else:
            log.info('Sorting by initial diameter:')
            srt = np.argsort(sample['SMA_INIT'])[::-1]

        sample = sample[srt]
        samplesrcs = [samplesrcs[I] for I in srt]
        for obj in sample:
            log.info(f'  ref_id={obj[REFIDCOLUMN]}: D(25)={obj["DIAM_INIT"]:.3f} arcmin, ' + \
                     f'max optical flux={obj["OPTFLUX"]:.2f} nanomaggies')

        # PSF size and depth
        for filt in all_opt_bands:
            sample[f'PSFSIZE_{filt.upper()}'] = np.zeros(len(sample), 'f4')
        for filt in all_bands:
            sample[f'PSFDEPTH_{filt.upper()}'] = np.zeros(len(sample), 'f4')

        # add the PSF depth and size
        if tractor is not None:
            _get_psfsize_and_depth(sample, tractor, all_data_bands,
                                   pixscale, incenter=False)

        return sample, samplesrcs, tractor


    err = 1 # assume success!

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
        # only images (no models, PSFs, etc.) if skip_ellipse=True
        if skip_ellipse:
            filt2imfile.update({band: {'image': 'image'}})
        elif skip_tractor:
            filt2imfile.update({band: {'image': 'image',
                                       'invvar': 'invvar',}})
        else:
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
            if skip_ellipse:
                filt2imfile.update({band: {'image': 'image'}})
            elif skip_tractor:
                filt2imfile.update({band: {'image': 'image',
                                           'invvar': 'invvar',}})
            else:
                filt2imfile.update({band: {'image': 'image',
                                           'model': 'model',
                                           'invvar': 'invvar',
                                           'psf': 'psf'}})

    if galex:
        galex_bands = ['FUV', 'NUV']
        all_bands = np.append(all_bands, galex_bands)
        galex_refband = galex_bands[1]
        for band in galex_bands:
            if skip_ellipse:
                filt2imfile.update({band: {'image': 'image'}})
            elif skip_tractor:
                filt2imfile.update({band: {'image': 'image',
                                           'invvar': 'invvar',}})
            else:
                filt2imfile.update({band: {'image': 'image',
                                           'model': 'model',
                                           'invvar': 'invvar',
                                           'psf': 'psf'}})

    # OK to miss files (e.g., -model, -psf) for some classes of
    # objects (e.g., RESOLVED) that we do not ellipse-fit.
    if skip_ellipse:
        missing_ok = True
    elif skip_tractor:
        missing_ok = True
    else:
        missing_ok = False


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
            else:
                # try without the .fz extension (e.g., RESOLVED mosaics)
                imfile = os.path.join(galaxydir, f'{galaxy}-{filt2imfile[filt][imtype]}-{filt}.fits')
                if os.path.isfile(imfile):
                    filt2imfile[filt][imtype] = imfile
                    datacount += 1
                else:
                    log.debug(f'Missing {imfile}')

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
                return {}, None, None, None, 0

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


    # Intercept objects (e.g., RESOLVED) that we do not ellipse-fit.
    if skip_ellipse:
        tractor = None
    else:
        if skip_tractor:
            tractor = None
            maskbitsfile = os.path.join(galaxydir, f'{galaxy}-{filt2imfile["maskbits"]}.fits')
        else:
            maskbitsfile = os.path.join(galaxydir, f'{galaxy}-{filt2imfile["maskbits"]}.fits.fz')

            # We ~have~ to read the tractor catalog using fits_table because we will
            # turn these catalog entries into Tractor sources later.
            tractorfile = os.path.join(galaxydir, f'{galaxy}-{filt2imfile["tractor"]}.fits')

            cols = ['ra', 'dec', 'bx', 'by', 'type', 'ref_cat', 'ref_id',
                    'sersic', 'shape_r', 'shape_e1', 'shape_e2', 'maskbits']
            cols += [f'flux_{filt}' for filt in opt_bands]
            cols += [f'flux_ivar_{filt}' for filt in opt_bands]
            cols += [f'nobs_{filt}' for filt in opt_bands]
            cols += [f'fracin_{filt}' for filt in opt_bands]
            cols += [f'fracflux_{filt}' for filt in opt_bands]
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

            # make sure there are sources
            with fitsio.FITS(tractorfile) as F:
                if F['CATALOG'].get_nrows() == 0:
                    log.warning('No sources in brick!')
                    return {}, None, None, None, 1

            prim = fitsio.read(tractorfile, columns='brick_primary')
            tractor = fits_table(tractorfile, rows=np.where(prim)[0], columns=cols)
            if len(tractor) == 0:
                log.warning('No brick_primary sources in brick!')
                return {}, None, None, None, 1
            else:
                log.info(f'Read {len(tractor):,d} brick_primary sources from {tractorfile}')


        # Read the maskbits image and build the starmask.
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


    sample, samplesrcs, tractor = _read_sample(opt_refband, tractor=tractor)

    if skip_ellipse:
        # Populate columns which would otherwise be added in
        # build_multiband_mask. NB: RA_TRACTOR,DEC_TRACTOR stay zero!
        sample['BX'] = sample['BX_INIT']
        sample['BY'] = sample['BY_INIT']
        sample['RA'] = sample['RA_INIT']
        sample['DEC'] = sample['DEC_INIT']
        sample['SGANAME'] = sga2025_name(sample['RA'], sample['DEC'])
        sample['SMA_MASK'] = sample['SMA_INIT'] # [arcsec]
        sample['SMA_MOMENT'] = sample['SMA_INIT'] # [arcsec]
        sample['BA_MOMENT'] = sample['BA_INIT']
        sample['PA_MOMENT'] = sample['PA_INIT']
        sample['ELLIPSEBIT'] |= ELLIPSEBIT['SKIPTRACTOR']

        # FIXME - duplicate code from io._read_image_data
        from tractor.tractortime import TAITime
        from astrometry.util.util import Tan
        from legacypipe.survey import LegacySurveyWcs, ConstantFitsWcs

        for filt, these_bands, this_pixscale, dataset in zip(
                [opt_refband, galex_refband, unwise_refband],
                [opt_bands, galex_bands, unwise_bands],
                [pixscale, galex_pixscale, unwise_pixscale],
                ['opt', 'galex', 'unwise']):

            hdr = fitsio.read_header(filt2imfile[filt]['image'], ext=1)
            wcs = Tan(hdr)
            if 'MJD_MEAN' in hdr:
                mjd_tai = hdr['MJD_MEAN'] # [TAI]
                wcs = LegacySurveyWcs(wcs, TAITime(None, mjd=mjd_tai))
            else:
                wcs = ConstantFitsWcs(wcs)
            opt_sz = (int(wcs.wcs.imageh), int(wcs.wcs.imagew))
            sz = (int(wcs.wcs.imageh), int(wcs.wcs.imagew))
            data[f'{dataset}_wcs'] = wcs
            data[f'{dataset}_hdr'] = hdr

            if filt == opt_refband:
                data['opt_hdr'] = hdr
                data['opt_wcs'] = wcs
            elif filt == galex_refband:
                data['galex_hdr'] = hdr
                data['galex_wcs'] = wcs
            elif filt == unwise_refband:
                data['unwise_hdr'] = hdr
                data['unwise_wcs'] = wcs

            # empty models and maskbits images
            data[f'{dataset}_models'] = np.zeros((len(sample), len(these_bands), *sz), 'f4')
            data[f'{dataset}_maskbits'] = np.zeros((len(sample), *sz), np.int32)

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
                    else:
                        data[f'{prefix}_jpg_{imtype}'] = np.zeros_like(data['opt_jpg_image'])

    else:
        # Read the basic imaging data and masks and build the multiband
        # masks.
        data = _read_image_data(data, filt2imfile, read_jpg=read_jpg,
                                skip_tractor=skip_tractor,
                                verbose=verbose)


    if not skip_ellipse:
        # build a brightstarmask "core" mask
        from legacypipe.runs import get_survey
        from legacypipe.reference import get_reference_sources, get_reference_map

        survey = get_survey(run)
        refstars, _ = get_reference_sources(survey, data['opt_wcs'].wcs,
                                            bands=data['opt_bands'],
                                            tycho_stars=True, gaia_stars=True,
                                            large_galaxies=False, star_clusters=False)
        refstars.radius /= 2. # shrink!
        refmap = get_reference_map(data['opt_wcs'].wcs, refstars)
        data['brightstarmask_core'] = refmap > 0

    # add MW dust extinction
    for filt in data['all_bands']: # NB: all bands
        sample[f'MW_TRANSMISSION_{filt.upper()}'] = mwdust_transmission(
            sample['EBV'], band=filt, run=data['run'])

    return data, tractor, sample, samplesrcs, err


def _get_radius_mosaic(diam, multiplicity=1, mindiam=0.5,
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


def get_radius_mosaic(diam_arcmin,
                      multiplicity=1,
                      q_primary=None,      # BA of the primary; if None, no ellipticity inflation
                      mindiam_arcmin=0.5,  # floor on input diameter (arcmin)
                      pixscale=0.262,      # arcsec/pixel
                      # single-object inflation f_single(d) = 1 + A / [1 + (d/d0)^beta]
                      single_A=0.6,
                      single_d0=2.5,
                      single_beta=1.7,
                      # group inflation f_mult(m) = min(1 + slope*(m-1), cap)
                      mult_slope=0.06,
                      mult_cap=1.30,
                      # ellipticity inflation f_q = 1 + eta*(1 - q)
                      eta=0.3,
                      get_barlen=False):
    """Compute a mosaic radius (arcsec) from a group diameter
    (arcmin), using smooth, size-aware inflation and optional BA-based
    padding. The final radius is rounded up to an integer number of
    pixels.

    Parameters
    ----------
    diam_arcmin : float
        Group diameter in arcmin (for singles, this is the object diam).
    multiplicity : int, default 1
        Group multiplicity; >1 triggers the group-inflation rule.
    q_primary : float or None, default None
        Axis ratio b/a of the primary. If None, skip ellipticity inflation.
    mindiam_arcmin : float, default 0.5
        Minimum diameter allowed (arcmin).
    pixscale : float, default 0.262
        Pixel scale in arcsec/pixel.
    single_A, single_d0, single_beta : floats
        Parameters of the single-object inflation curve.
    mult_slope : float, default 0.06
        Per-companion inflation for groups.
    mult_cap : float, default 1.30
        Maximum inflation for groups (set to a large value to effectively disable).
    eta : float, default 0.3
        Strength of BA inflation; f_q = 1 + eta*(1 - q).

    Returns
    -------
    float
        Mosaic radius in arcsec, rounded up to a whole number of pixels.

    """
    from math import ceil

    # diameter floor (arcmin) and base radius (arcsec)
    d = float(diam_arcmin)
    if d < mindiam_arcmin:
        d = mindiam_arcmin
    r0 = 30.0 * d  # arcsec (half-diameter)

    # inflation for singles vs groups
    if multiplicity <= 1:
        f_single = 1.0 + single_A / (1.0 + (d / single_d0) ** single_beta)
        f = f_single
    else:
        f_mult = 1.0 + mult_slope * (multiplicity - 1)
        if f_mult > mult_cap:
            f_mult = mult_cap
        f = f_mult

    # optional BA padding (thin primaries → larger radius)
    if q_primary is not None:
        q = float(q_primary)
        if q < 0.0:
            q = 0.0
        if q > 1.0:
            q = 1.0
        f_q = 1.0 + eta * (1.0 - q)
        f *= f_q

    # apply inflation
    r_arcsec = r0 * f

    # enforce 30″ minimum and round up to whole pixels (no max cap)
    if r_arcsec < 30.0:
        r_arcsec = 30.0
    npix = ceil(r_arcsec / pixscale)
    r_arcsec = npix * pixscale

    if get_barlen:
        if r_arcsec > 6. * 60.: # [>6] arcmin
            barlabel = '2 arcmin'
            barlen = np.ceil(120. / pixscale).astype(int) # [pixels]
        elif (r_arcsec > 3. * 60.) & (r_arcsec < 6. * 60.): # [3-6] arcmin
            barlabel = '1 arcmin'
            barlen = np.ceil(60. / pixscale).astype(int) # [pixels]
        else:
            barlabel = '30 arcsec'
            barlen = np.ceil(30. / pixscale).astype(int) # [pixels]
        return r_arcsec, barlen, barlabel
    else:
        return r_arcsec
