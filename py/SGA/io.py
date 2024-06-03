"""
SGA.io
======

Code to read and write the various SGA files.

"""
import os, sys, time, pdb
import fitsio
import numpy as np
from astropy.table import Table


# C file descriptors for stderr and stdout, used in redirection
# context manager.
import ctypes
from contextlib import contextmanager

libc = ctypes.CDLL(None)
c_stdout = None
c_stderr = None
try:
    # Linux systems
    c_stdout = ctypes.c_void_p.in_dll(libc, 'stdout')
    c_stderr = ctypes.c_void_p.in_dll(libc, 'stderr')
except:
    try:
        # Darwin
        c_stdout = ctypes.c_void_p.in_dll(libc, '__stdoutp')
        c_stderr = ctypes.c_void_p.in_dll(libc, '__stdoutp')
    except:
        # Neither!
        pass

    
def sga_dir():
    if 'SGA_DIR' not in os.environ:
        print('Required ${SGA_DIR} environment variable not set.')
        raise EnvironmentError
    ldir = os.path.abspath(os.getenv('SGA_DIR'))
    if not os.path.isdir(ldir):
        os.makedirs(ldir, exist_ok=True)
    return ldir


def sga_data_dir():
    if 'SGA_DATA_DIR' not in os.environ:
        print('Required ${SGA_DATA_DIR} environment variable not set.')
        raise EnvironmentError
    ldir = os.path.abspath(os.getenv('SGA_DATA_DIR'))
    if not os.path.isdir(ldir):
        os.makedirs(ldir, exist_ok=True)
    return ldir


def sga_html_dir():
    if 'SGA_HTML_DIR' not in os.environ:
        print('Required ${SGA_HTML_DIR} environment variable not set.')
        raise EnvironmentError
    ldir = os.path.abspath(os.getenv('SGA_HTML_DIR'))
    if not os.path.isdir(ldir):
        os.makedirs(ldir, exist_ok=True)
    return ldir


def get_raslice(ra):
    return f'{int(ra):03d}'


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
        objs = bricks['BRICKNAME']
        ras = bricks['RA']
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
    
        objs = sample[galcolumn]
        ras = sample[racolumn]

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


def weighted_partition(weights, n):
    '''
    Partition `weights` into `n` groups with approximately same sum(weights)

    Args:
        weights: array-like weights
        n: number of groups

    Returns list of lists of indices of weights for each group

    Notes:
        compared to `dist_discrete_all`, this function allows non-contiguous
        items to be grouped together which allows better balancing.

    '''
    #- sumweights will track the sum of the weights that have been assigned
    #- to each group so far
    sumweights = np.zeros(n, dtype=float)

    #- Initialize list of lists of indices for each group
    groups = list()
    for i in range(n):
        groups.append(list())

    #- Assign items from highest weight to lowest weight, always assigning
    #- to whichever group currently has the fewest weights
    weights = np.asarray(weights)
    for i in np.argsort(-weights):
        j = np.argmin(sumweights)
        groups[j].append(i)
        sumweights[j] += weights[i]

    assert len(groups) == n

    return groups


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


@contextmanager
def stdouterr_redirected(to=None, comm=None, overwrite=False):
    """
    Redirect stdout and stderr to a file.

    The general technique is based on:

    http://stackoverflow.com/questions/5081657
    http://eli.thegreenplace.net/2015/redirecting-all-kinds-of-stdout-in-python/

    One difference here is that each process in the communicator
    redirects to a different temporary file, and the upon exit
    from the context the rank zero process concatenates these
    in order to the file result.

    Args:
        to (str): The output file name.
        comm (mpi4py.MPI.Comm): The optional MPI communicator.
        overwrite (bool): if True overwrite file, otherwise backup to to.N first
    """
    nproc = 1
    rank = 0
    if comm is not None:
        nproc = comm.size
        rank = comm.rank

    # The currently active POSIX file descriptors
    fd_out = sys.stdout.fileno()
    fd_err = sys.stderr.fileno()

    # The DESI loggers.
    #desi_loggers = desiutil.log._desiutil_log_root

    def _redirect(out_to, err_to):

        # Flush the C-level buffers
        if c_stdout is not None:
            libc.fflush(c_stdout)
        if c_stderr is not None:
            libc.fflush(c_stderr)

        # This closes the python file handles, and marks the POSIX
        # file descriptors for garbage collection- UNLESS those
        # are the special file descriptors for stderr/stdout.
        sys.stdout.close()
        sys.stderr.close()

        # Close fd_out/fd_err if they are open, and copy the
        # input file descriptors to these.
        os.dup2(out_to, fd_out)
        os.dup2(err_to, fd_err)

        # Create a new sys.stdout / sys.stderr that points to the
        # redirected POSIX file descriptors.  In Python 3, these
        # are actually higher level IO objects.
        if sys.version_info[0] < 3:
            sys.stdout = os.fdopen(fd_out, "wb")
            sys.stderr = os.fdopen(fd_err, "wb")
        else:
            # Python 3 case
            sys.stdout = io.TextIOWrapper(os.fdopen(fd_out, 'wb'))
            sys.stderr = io.TextIOWrapper(os.fdopen(fd_err, 'wb'))

        # update DESI logging to use new stdout
        for name, logger in desi_loggers.items():
            hformat = None
            while len(logger.handlers) > 0:
                h = logger.handlers[0]
                if hformat is None:
                    hformat = h.formatter._fmt
                logger.removeHandler(h)
            # Add the current stdout.
            ch = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(hformat, datefmt='%Y-%m-%dT%H:%M:%S')
            ch.setFormatter(formatter)
            logger.addHandler(ch)

    # redirect both stdout and stderr to the same file

    if to is None:
        to = "/dev/null"

    if rank == 0:
        #log = get_logger()
        #log.info("Begin log redirection to {} at {}".format(to, time.asctime()))
        print("Begin log redirection to {} at {}".format(to, time.asctime()))
        if not overwrite:
            backup_filename(to)

    #- all ranks wait for logfile backup
    if comm is not None:
        comm.barrier()

    # Save the original file descriptors so we can restore them later
    saved_fd_out = os.dup(fd_out)
    saved_fd_err = os.dup(fd_err)

    try:
        pto = to
        if to != "/dev/null":
            pto = "{}_{}".format(to, rank)

        # open python file, which creates low-level POSIX file
        # descriptor.
        file = open(pto, "w")

        # redirect stdout/stderr to this new file descriptor.
        _redirect(out_to=file.fileno(), err_to=file.fileno())

        yield # allow code to be run with the redirected output

        # close python file handle, which will mark POSIX file
        # descriptor for garbage collection.  That is fine since
        # we are about to overwrite those in the finally clause.
        file.close()

    finally:
        # flush python handles for good measure
        sys.stdout.flush()
        sys.stderr.flush()

        # restore old stdout and stderr
        _redirect(out_to=saved_fd_out, err_to=saved_fd_err)

        if nproc > 1:
            comm.barrier()

        # concatenate per-process files
        if rank == 0 and to != "/dev/null":
            with open(to, "w") as outfile:
                for p in range(nproc):
                    outfile.write("================ Start of Process {} ================\n".format(p))
                    fname = "{}_{}".format(to, p)
                    with open(fname) as infile:
                        outfile.write(infile.read())
                    outfile.write("================= End of Process {} =================\n\n".format(p))
                    os.remove(fname)

        if nproc > 1:
            comm.barrier()

        if rank == 0:
            #log = get_logger()
            #log.info("End log redirection to {} at {}".format(to, time.asctime()))
            print("End log redirection to {} at {}".format(to, time.asctime()))

        # flush python handles for good measure
        sys.stdout.flush()
        sys.stderr.flush()

    return


def _missing_files_one(args):
    """Wrapper for the multiprocessing."""
    return missing_files_one(*args)

def missing_files_one(checkfile, dependsfile, overwrite):
    """Simple support script for missing_files."""
    
    from pathlib import Path
    if Path(checkfile).exists() and overwrite is False:
        # Is the stage that this stage depends on done, too?
        #print(checkfile, dependsfile, overwrite)
        if dependsfile is None:
            return 'done'
        else:
            if Path(dependsfile).exists():
                return 'done'
            else:
                return 'todo'
    else:
        #print(f'missing_files_one {checkfile}')
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
                    print(f'Missing depends file {dependsfile}')
                    return 'fail'
            else:
                return 'todo'

        return 'todo'

    
def missing_files(sample=None, bricks=None, detection_coadds=False, coadds=False,
                  ellipse=False, htmlplots=False, htmlindex=False, build_SGA=False, 
                  overwrite=False, verbose=False, htmldir='.', size=1, mp=1):
    """Figure out which files are missing and still need to be processed.

    """
    from glob import glob
    import multiprocessing
    import astropy

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

    dependson = None
    if detection_coadds:
        galaxy, galaxydir = get_galaxy_galaxydir(bricks=bricks)
    else:
        if htmlplots is False and htmlindex is False:
            if verbose:
                t0 = time.time()
                print('Getting galaxy names and directories...', end='')
            galaxy, galaxydir = get_galaxy_galaxydir(sample)
            if verbose:
                print(f'...took {time.time() - t0:.3f} sec')

    if detection_coadds:
        suffix = 'detection-coadds'
        filesuffix = '-detection-coadds.isdone'
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
            missargs.append([checkfile, os.path.join(np.atleast_1d(dependsondir)[igal], f'{gal}{dependson}'), overwrite])
        else:
            missargs.append([checkfile, None, overwrite])

    if verbose:
        t0 = time.time()
        print('Finding missing files...', end='')
    if mp > 1:
        with multiprocessing.Pool(mp) as P:
            todo = np.array(P.map(_missing_files_one, missargs))
    else:
        todo = np.array([_missing_files_one(_missargs) for _missargs in missargs])
        
    if verbose:
        print(f'...took {(time.time() - t0)/60.:.3f} min')

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


#def custom_brickname(ra, dec):
#    brickname = '{:08d}{}{:07d}'.format(
#        int(100000*ra), 'm' if dec < 0 else 'p',
#        int(100000*np.abs(dec)))
#    #brickname = '{:06d}{}{:05d}'.format(
#    #    int(1000*ra), 'm' if dec < 0 else 'p',
#    #    int(1000*np.abs(dec)))
#    return brickname
#
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
#
#def read_desi_tiles(verbose=False):
#    """Read the latest DESI tile file.
#    
#    """
#    tilefile = os.path.join(sample_dir(), 'catalogs', 'desi-tiles.fits')
#    tiles = Table(fitsio.read(tilefile, ext=1, upper=True))
#    tiles = tiles[tiles['IN_DESI'] > 0]
#    
#    if verbose:
#        print('Read {} DESI tiles from {}'.format(len(tiles), tilefile))
#    
#    return tiles
#
#def read_tycho(magcut=99, verbose=False):
#    """Read the Tycho 2 catalog.
#    
#    """
#    tycho2 = os.path.join(sample_dir(), 'catalogs', 'tycho2.kd.fits')
#    tycho = Table(fitsio.read(tycho2, ext=1, upper=True))
#    tycho = tycho[np.logical_and(tycho['ISGALAXY'] == 0, tycho['MAG_BT'] <= magcut)]
#    if verbose:
#        print('Read {} Tycho-2 stars with B<{:.1f}.'.format(len(tycho), magcut), flush=True)
#    
#    # Radius of influence; see eq. 9 of https://arxiv.org/pdf/1203.6594.pdf
#    #tycho['RADIUS'] = (0.0802*(tycho['MAG_BT'])**2 - 1.860*tycho['MAG_BT'] + 11.625) / 60 # [degree]
#
#    # From https://github.com/legacysurvey/legacypipe/blob/large-gals-only/py/legacypipe/runbrick.py#L1668
#    # Note that the factor of 0.262 has nothing to do with the DECam pixel scale!
#    tycho['RADIUS'] = np.minimum(1800., 150. * 2.5**((11. - tycho['MAG_BT']) / 4) ) * 0.262 / 3600
#
#    #import matplotlib.pyplot as plt
#    #oldrad = (0.0802*(tycho['MAG_BT'])**2 - 1.860*tycho['MAG_BT'] + 11.625) / 60 # [degree]
#    #plt.scatter(tycho['MAG_BT'], oldrad*60, s=1) ; plt.scatter(tycho['MAG_BT'], tycho['RADIUS']*60, s=1) ; plt.show()
#    #pdb.set_trace()
#    
#    return tycho
#
#def read_hyperleda(verbose=False, allwise=False, version=None):
#    """Read the Hyperleda catalog.
#
#    These are the archived versions. For DR9 we reset the counter to start at v3.0!
#
#    if version == 'v1.0':
#        hyperfile = 'hyperleda-d25min10-18may13.fits'
#    elif version == 'v2.0':
#        hyperfile = 'hyperleda-d25min10-18nov14.fits'
#    elif version == 'v3.0':
#        hyperfile = 'hyperleda-d25min10-18nov14.fits'
#    elif version == 'v4.0':
#        hyperfile = 'hyperleda-d25min10-18nov14.fits'
#    elif version == 'v5.0':
#        hyperfile = 'hyperleda-d25min10-18nov14.fits'
#    elif version == 'v6.0':
#        hyperfile = 'hyperleda-d25min10-18nov14.fits'
#    elif version == 'v7.0':
#        hyperfile = 'hyperleda-d25min10-18nov14.fits'
#    else:
#        print('Unknown version!')
#        raise ValueError
#    
#    """
#    if version is None:
#        version = parent_version()
#        
#    if version == 'v1.0':
#        hyperfile = 'hyperleda-d25min10-18may13.fits'
#        ref = 'LEDA-20180513'
#    elif version == 'v2.0':
#        hyperfile = 'hyperleda-d25min10-18nov14.fits'
#        ref = 'LEDA-20181114'
#    elif version == 'v3.0':
#        hyperfile = 'hyperleda-d25min10-18nov14.fits'
#        ref = 'LEDA-20181114'
#    else:
#        print('Unknown version!')
#        raise ValueError
#
#    hyperledafile = os.path.join(sample_dir(), 'hyperleda', hyperfile)
#    allwisefile = hyperledafile.replace('.fits', '-allwise.fits')
#
#    leda = Table(fitsio.read(hyperledafile, ext=1, upper=True))
#    #leda.add_column(Column(name='GROUPID', dtype='i8', length=len(leda)))
#    if verbose:
#        print('Read {} objects from {}'.format(len(leda), hyperledafile), flush=True)
#
#    if allwise:
#        wise = Table(fitsio.read(allwisefile, ext=1, upper=True))
#        if verbose:
#            print('Read {} objects from {}'.format(len(wise), allwisefile), flush=True)
#
#        # Merge the tables
#        wise.rename_column('RA', 'WISE_RA')
#        wise.rename_column('DEC', 'WISE_DEC')
#
#        leda = hstack( (leda, wise) )
#        leda.add_column(Column(name='IN_WISE', data=np.zeros(len(leda)).astype(bool)))
#
#        haswise = np.where(wise['CNTR'] != -1)[0]
#        #nowise = np.where(wise['CNTR'] == 0)[0]
#        #print('unWISE match: {}/{} ({:.2f}%) galaxies.'.format(len(haswise), len(leda)))
#
#        #print('EXT_FLG summary:')
#        #for flg in sorted(set(leda['EXT_FLG'][haswise])):
#        #    nn = np.sum(flg == leda['EXT_FLG'][haswise])
#        #    print('  {}: {}/{} ({:.2f}%)'.format(flg, nn, len(haswise), 100*nn/len(haswise)))
#        #print('Need to think this through a bit more; look at:')
#        #print('  http://wise2.ipac.caltech.edu/docs/release/allsky/expsup/sec4_4c.html#xsc')
#        #leda['INWISE'] = (np.array(['NULL' not in dd for dd in wise['DESIGNATION']]) * 
#        #                  np.isfinite(wise['W1SIGM']) * np.isfinite(wise['W2SIGM']) )
#        leda['IN_ALLWISE'][haswise] = True
#
#        print('  Identified {}/{} ({:.2f}%) objects with AllWISE photometry.'.format(
#            np.sum(leda['IN_ALLWISE']), len(leda), 100*np.sum(leda['IN_ALLWISE'])/len(leda) ))
#
#    # Assign a unique ID and also fix infinite PA and B/A.
#    leda.add_column(Column(name='SGA_ID', length=len(leda), dtype='i8'), index=0)
#    leda['SGA_ID'] = np.arange(len(leda))
#    leda['BYHAND'] = np.zeros(len(leda), bool)
#    leda['REF'] = ref
#    
#    fix = np.isnan(leda['PA'])
#    if np.sum(fix) > 0:
#        leda['PA'][fix] = 0.0
#    fix = np.isnan(leda['BA'])
#    if np.sum(fix) > 0:
#        leda['BA'][fix] = 1.0
#    fix = np.isnan(leda['Z'])
#    if np.sum(fix) > 0:
#        leda['Z'][fix] = -99.0
#
#    return leda
#
#def read_localgroup_dwarfs():
#    """Read the sample generated by bin/SGA-localgroup-dwarfs.
#
#    """
#    dwarfsfile = os.path.join(sample_dir(), 'catalogs', 'SGA-dwarfs.fits')
#    dwarfs = Table(fitsio.read(dwarfsfile, upper=True))
#    print('Read {} Local Group dwarfs from {}'.format(len(dwarfs), dwarfsfile))
#
#    return dwarfs
#
##def in_footprint(parent, verbose=False):
##    """Find all galaxies in the DESI footprint.
##
##    """
##    import time
##    import healpy as hp
##    import legacyhalos.misc
##    
##    tiles = read_desi_tiles(verbose=verbose)
##    indesi = SGA.misc.is_point_in_desi(tiles, parent['RA'], parent['DEC']).astype(bool)
##
##    t0 = time.time()
##
##    return parent
#
#def in_footprint(parent, nside=2048, dr='dr9'):
#    """Find all galaxies in the DESI footprint.
#
#    """
#    import time
#    import healpy as hp
#    import legacyhalos.misc
#    
#    #tiles = SGA.io.read_desi_tiles(verbose=verbose)
#    #indesi = SGA.misc.is_point_in_desi(tiles, parent['RA'], parent['DEC']).astype(bool)
#
#    parentpix = legacyhalos.misc.radec2pix(nside, parent['RA'], parent['DEC'])
#    #parentpix = np.hstack((parentpix, hp.pixelfunc.get_all_neighbours(nside, parentpix, nest=True).flatten()))
#
#    drdir = os.path.join(sample_dir(), dr)
#
#    bands = ('g', 'r', 'z')
#    camera = ('90prime', 'mosaic', 'decam')
#
#    indesi = dict()
#    for cam in camera:
#        for band in bands:
#            indesi.update({'{}_{}'.format(cam, band): np.zeros(len(parent), dtype=bool)})
#
#    #indesi = np.zeros(len(parent), dtype=bool)
#    t0 = time.time()
#    for cam, radius in zip(camera, (0.44, 0.21, 0.17)):
#        if False:
#            from astrometry.libkd.spherematch import trees_match, tree_open
#            kdccds = tree_open(os.path.join(drdir, 'survey-ccds-{}-{}.kd.fits'.format(cam, dr)))
#            I, J, dd = trees_match(kdparent, kdccds, np.radians(radius))#, nearest=True)
#        else:
#            ccdsfile = os.path.join(drdir, 'survey-ccds-{}-{}.kd.fits'.format(cam, dr))
#            ccds = fitsio.read(ccdsfile)
#            ccds = ccds[ccds['ccd_cuts'] == 0]
#            print('Read {} CCDs from {}'.format(len(ccds), ccdsfile))
#
#            for band in bands:
#                ww = ccds['filter'] == band
#                if np.sum(ww) > 0:
#                    # add the neighboring healpixels to protect against edge effects
#                    ccdpix = legacyhalos.misc.radec2pix(nside, ccds['ra'][ww], ccds['dec'][ww])
#                    ccdpix = np.hstack((ccdpix, hp.pixelfunc.get_all_neighbours(nside, ccdpix, nest=True).flatten()))
#                    if np.sum(ccdpix == -1) > 0: # remove the "no neighbors" healpixel, if it exists
#                        ccdpix = np.delete(ccdpix, np.where(ccdpix == -1)[0])
#                    I = np.isin(parentpix, ccdpix)
#                    indesi['{}_{}'.format(cam, band)][I] = True
#                else:
#                    I = [False]
#                #print('Found {} galaxies in {} {} footprint in {:.1f} sec'.format(np.sum(I), cam, time.time() - t0))
#                print('  Found {} galaxies in {} {} footprint.'.format(np.sum(I), cam, band))
#    print('Total time to find galaxies in footprint = {:.1f} sec'.format(time.time() - t0))
#    
#    parent['IN_FOOTPRINT_NORTH'] = indesi['90prime_g'] | indesi['90prime_r'] | indesi['mosaic_z']
#    parent['IN_FOOTPRINT_NORTH_GRZ'] = indesi['90prime_g'] & indesi['90prime_r'] & indesi['mosaic_z']
#
#    parent['IN_FOOTPRINT_SOUTH'] = indesi['decam_g'] | indesi['decam_r'] | indesi['decam_z']
#    parent['IN_FOOTPRINT_SOUTH_GRZ'] = indesi['decam_g'] & indesi['decam_r'] & indesi['decam_z']
#    
#    parent['IN_FOOTPRINT'] = parent['IN_FOOTPRINT_NORTH'] | parent['IN_FOOTPRINT_SOUTH']
#    parent['IN_FOOTPRINT_GRZ'] = parent['IN_FOOTPRINT_NORTH_GRZ'] | parent['IN_FOOTPRINT_SOUTH_GRZ']
#
#    #plt.scatter(parent['RA'], parent['DEC'], s=1)
#    #plt.scatter(parent['RA'][indesi], parent['DEC'][indesi], s=1)
#    #plt.xlim(360, 0)
#    #plt.show()
#
#    #bb = parent[parent['IN_FOOTPRINT_NORTH_GRZ'] & parent['IN_FOOTPRINT_SOUTH_GRZ']]
#    #plt.scatter(bb['RA'], bb['DEC'], s=1)
#    #plt.xlim(300, 90) ; plt.ylim(30, 36)
#    #plt.axhline(y=32.375, color='k')
#    #plt.xlabel('RA') ; plt.ylabel('Dec')
#    #plt.show()
#    
#    print('  Identified {}/{} ({:.2f}%) galaxies inside and {}/{} ({:.2f}%) galaxies outside the DESI footprint.'.format(
#        np.sum(parent['IN_FOOTPRINT']), len(parent), 100*np.sum(parent['IN_FOOTPRINT'])/len(parent), np.sum(~parent['IN_FOOTPRINT']),
#        len(parent), 100*np.sum(~parent['IN_FOOTPRINT'])/len(parent)))
#
#    return parent

