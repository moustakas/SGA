"""
LSLGA.mpi
=========

Code to deal with the MPI portion of the pipeline.

"""
import os, time, pdb
import numpy as np
from contextlib import redirect_stdout, redirect_stderr

import LSLGA.io

def _start(galaxy, log=None, seed=None):
    if seed:
        print('Random seed = {}'.format(seed), flush=True)        
    print('Started working on galaxy {} at {}'.format(
        galaxy, time.asctime()), flush=True, file=log)

def _done(galaxy, err, t0, log=None):
    if err == 0:
        print('ERROR: galaxy {}; please check the logfile.'.format(galaxy), flush=True, file=log)
    print('Finished galaxy {} in {:.3f} minutes.'.format(
          galaxy, (time.time() - t0)/60), flush=True, file=log)

def call_pipeline_coadds(onegal, galaxy, radius_mosaic, survey, kdccds_north,
                         kdccds_south, pixscale=0.262, nproc=1, force=False,
                         debug=False, logfile=None, apodize=False, unwise=True,
                         no_large_galaxies=False, no_gaia=False, no_tycho=False,
                         cleanup=True):
    """Wrapper script to build the pipeline coadds.

    radius_mosaic in arcsec

    """
    t0 = time.time()
    if debug:
        _start(galaxy)
        run = LSLGA.io.get_run(onegal, radius_mosaic, pixscale, kdccds_north, kdccds_south)
        err = LSLGA.coadds.pipeline_coadds(onegal, galaxy=galaxy, radius_mosaic=radius_mosaic,
                                           survey=survey, pixscale=pixscale, run=run,
                                           nproc=nproc, force=force, cleanup=cleanup,
                                           apodize=apodize, unwise=unwise, no_large_galaxies=no_large_galaxies,
                                           no_gaia=no_gaia, no_tycho=no_tycho)
        _done(galaxy, err, t0)
    else:
        with open(logfile, 'a') as log:
            with redirect_stdout(log), redirect_stderr(log):
                _start(galaxy, log=log)
                run = LSLGA.io.get_run(onegal, radius_mosaic, pixscale, kdccds_north, kdccds_south, log=log)
                err = LSLGA.coadds.pipeline_coadds(onegal, galaxy=galaxy, radius_mosaic=radius_mosaic,
                                                   survey=survey, pixscale=pixscale, run=run,
                                                   nproc=nproc, force=force, log=log, cleanup=cleanup,
                                                   apodize=apodize, unwise=unwise,
                                                   no_large_galaxies=no_large_galaxies,
                                                   no_gaia=no_gaia, no_tycho=no_tycho)
                _done(galaxy, err, t0, log=log)

def call_custom_coadds(onegal, galaxy, radius_mosaic, survey, pixscale=0.262,
                        nproc=1, debug=False, logfile=None, radius_mask=None,
                        sdss=False, sdss_pixscale=0.396, write_ccddata=False,
                        doforced_phot=True, apodize=False):
    """Wrapper script to build the pipeline coadds."""
    t0 = time.time()
    if debug:
        _start(galaxy)
        if sdss:
            err = LSLGA.sdss.custom_coadds(onegal, galaxy=galaxy, radius_mosaic=radius_mosaic,
                                           survey=survey, radius_mask=radius_mask, pixscale=sdss_pixscale,
                                           nproc=nproc)
        else:
            err = LSLGA.coadds.custom_coadds(onegal, galaxy=galaxy, radius_mosaic=radius_mosaic,
                                             survey=survey, radius_mask=radius_mask, pixscale=pixscale,
                                             nproc=nproc, write_ccddata=write_ccddata,
                                             doforced_phot=doforced_phot, apodize=apodize)
        _done(galaxy, err, t0)
    else:
        with open(logfile, 'a') as log:
            with redirect_stdout(log), redirect_stderr(log):
                _start(galaxy, log=log)
                if sdss:
                    err = LSLGA.sdss.custom_coadds(onegal, galaxy=galaxy, radius_mosaic=radius_mosaic,
                                                   survey=survey, radius_mask=radius_mask, pixscale=sdss_pixscale,
                                                   nproc=nproc, log=log)
                else:
                    err = LSLGA.coadds.custom_coadds(onegal, galaxy=galaxy, radius_mosaic=radius_mosaic,
                                                     survey=survey, radius_mask=radius_mask, pixscale=pixscale,
                                                     nproc=nproc, log=log, write_ccddata=write_ccddata,
                                                     doforced_phot=doforced_phot, apodize=apodize)
                _done(galaxy, err, t0, log=log)
                
def call_ellipse(onegal, galaxy, galaxydir, pixscale=0.262, nproc=1, verbose=False,
                 debug=False, logfile=None, input_ellipse=False, zcolumn=None,
                 sdss=False, sdss_pixscale=0.396, custom_tractor=True):
    """Wrapper script to do ellipse-fitting.

    """
    import LSLGA.ellipse

    if zcolumn is None:
        zcolumn = 'Z_LAMBDA'

    t0 = time.time()
    if debug:
        _start(galaxy)
        err = LSLGA.ellipse.LSLGA_ellipse(onegal, galaxy=galaxy, galaxydir=galaxydir,
                                          pixscale=pixscale, nproc=nproc,
                                          zcolumn=zcolumn, input_ellipse=input_ellipse,
                                          verbose=verbose, debug=debug,
                                          sdss=sdss, sdss_pixscale=sdss_pixscale,
                                          pipeline=True, unwise=False,
                                          custom_tractor=custom_tractor)
        _done(galaxy, err, t0)
    else:
        with open(logfile, 'a') as log:
            with redirect_stdout(log), redirect_stderr(log):
                _start(galaxy, log=log)
                err = LSLGA.ellipse.LSLGA_ellipse(onegal, galaxy=galaxy, galaxydir=galaxydir,
                                                  pixscale=pixscale, nproc=nproc,
                                                  zcolumn=zcolumn, input_ellipse=input_ellipse,
                                                  verbose=verbose, debug=debug,
                                                  sdss=sdss, sdss_pixscale=sdss_pixscale,
                                                  pipeline=True, unwise=False,
                                                  custom_tractor=custom_tractor)
                _done(galaxy, err, t0, log=log)

def call_sersic(onegal, galaxy, galaxydir, seed, verbose, debug, logfile):
    """Wrapper script to do Sersic-fitting.

    """
    import LSLGA.sersic

    t0 = time.time()
    if debug:
        _start(galaxy, seed=seed)
        err = LSLGA.sersic.LSLGA_sersic(onegal, galaxy=galaxy, galaxydir=galaxydir,
                                        debug=debug, verbose=verbose, seed=seed)
        _done(galaxy, err, t0)
    else:
        with open(logfile, 'a') as log:
            with redirect_stdout(log), redirect_stderr(log):
                _start(galaxy, log=log, seed=seed)
                err = LSLGA.sersic.LSLGA_sersic(onegal, galaxy=galaxy, galaxydir=galaxydir,
                                                debug=debug, verbose=verbose, seed=seed)
                _done(galaxy, err, t0, log=log)

def call_sky(onegal, galaxy, galaxydir, survey, seed, nproc, pixscale,
              verbose, debug, logfile):
    """Wrapper script to do Sersic-fitting.

    """
    import LSLGA.sky

    t0 = time.time()
    if debug:
        _start(galaxy, seed=seed)
        err = LSLGA.sky.LSLGA_sky(onegal, survey=survey, galaxy=galaxy, galaxydir=galaxydir,
                                  nproc=nproc, pixscale=pixscale, seed=seed,
                                  debug=debug, verbose=verbose, force=force)
        _done(galaxy, err, t0)
    else:
        with open(logfile, 'a') as log:
            with redirect_stdout(log), redirect_stderr(log):
                _start(galaxy, log=log, seed=seed)
                err = LSLGA.sky.LSLGA_sky(onegal, survey=survey, galaxy=galaxy, galaxydir=galaxydir,
                                          nproc=nproc, pixscale=pixscale, seed=seed,
                                          debug=debug, verbose=verbose, force=force)
                _done(galaxy, err, t0, log=log)
                
def call_htmlplots(onegal, galaxy, survey, pixscale, nproc, debug, clobber,
                    verbose, ccdqa, logfile, htmldir, zcolumn):
    """Wrapper script to build the pipeline coadds."""
    t0 = time.time()

    if debug:
        _start(galaxy)
        err = LSLGA.html.make_plots(onegal, datadir=None, htmldir=htmldir,
                                    pixscale=pixscale, survey=survey, clobber=clobber,
                                    verbose=verbose, nproc=nproc, zcolumn=zcolumn, 
                                    ccdqa=ccdqa, maketrends=False)
        _done(galaxy, err, t0)
    else:
        with open(logfile, 'a') as log:
            with redirect_stdout(log), redirect_stderr(log):
                _start(galaxy, log=log)
                err = LSLGA.html.make_plots(onegal, datadir=None, htmldir=htmldir,
                                            pixscale=pixscale, survey=survey, clobber=clobber,
                                            verbose=verbose, nproc=nproc, zcolumn=zcolumn, 
                                            ccdqa=ccdqa, maketrends=False)
                _done(galaxy, err, t0, log=log)

def mpi_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--nproc', default=1, type=int, help='number of multiprocessing processes per MPI rank.')
    parser.add_argument('--mpi', action='store_true', help='Use MPI parallelism')

    parser.add_argument('--sdss', action='store_true', help='Analyze the SDSS galaxies.')

    parser.add_argument('--first', type=int, help='Index of first object to process.')
    parser.add_argument('--last', type=int, help='Index of last object to process.')
    parser.add_argument('--seed', type=int, default=1, help='Random seed (used with --sky and --sersic).')

    parser.add_argument('--coadds', action='store_true', help='Build the pipeline coadds.')
    parser.add_argument('--custom-coadds', action='store_true', help='Build the custom coadds.')
    parser.add_argument('--ellipse', action='store_true', help='Do the ellipse fitting.')
    parser.add_argument('--sersic', action='store_true', help='Perform Sersic fitting.')
    parser.add_argument('--integrate', action='store_true', help='Integrate the surface brightness profiles.')
    parser.add_argument('--sky', action='store_true', help='Estimate the sky variance.')
    parser.add_argument('--htmlplots', action='store_true', help='Build the HTML output.')
    parser.add_argument('--htmlindex', action='store_true', help='Build HTML index.html page.')

    parser.add_argument('--htmldir', type=str, help='Output directory for HTML files.')
    
    parser.add_argument('--pixscale', default=0.262, type=float, help='pixel scale (arcsec/pix).')
    parser.add_argument('--sdss-pixscale', default=0.396, type=float, help='SDSS pixel scale (arcsec/pix).')
    
    parser.add_argument('--ccdqa', action='store_true', help='Build the CCD-level diagnostics.')
    parser.add_argument('--force', action='store_true', help='Use with --coadds; ignore previous pickle files.')
    parser.add_argument('--count', action='store_true', help='Count how many objects are left to analyze and then return.')
    parser.add_argument('--nomakeplots', action='store_true', help='Do not remake the QA plots for the HTML pages.')

    parser.add_argument('--debug', action='store_true', help='Log to STDOUT and build debugging plots.')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output.')
    parser.add_argument('--clobber', action='store_true', help='Overwrite existing files.')                                
    args = parser.parse_args()

    return args
