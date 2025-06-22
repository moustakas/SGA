"""
SGA.mpi
=======

Code to deal with the MPI portion of the pipeline.

"""
import os, time, subprocess, pdb
import numpy as np
from contextlib import redirect_stdout, redirect_stderr

import SGA.io
import SGA.html


def mpi_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--mp', default=1, type=int, help='number of multiprocessing processes per MPI rank.')

    parser.add_argument('--first', type=int, help='Index of first object to process.')
    parser.add_argument('--last', type=int, help='Index of last object to process.')
    parser.add_argument('--galaxylist', type=str, nargs='*', default=None, help='List of galaxy names to process.')

    parser.add_argument('--d25min', default=0.0, type=float, help='Minimum diameter (arcmin).')
    parser.add_argument('--d25max', default=100.0, type=float, help='Maximum diameter (arcmin).')

    parser.add_argument('--coadds', action='store_true', help='Build the large-galaxy coadds.')
    parser.add_argument('--customsky', action='store_true', help='Build the largest large-galaxy coadds with custom sky-subtraction.')
    parser.add_argument('--just-coadds', action='store_true', help='Just build the coadds and return (using --early-coadds in runbrick.py.')

    parser.add_argument('--ellipse', action='store_true', help='Do the ellipse fitting.')
    parser.add_argument('--htmlplots', action='store_true', help='Build the pipeline figures.')
    parser.add_argument('--htmlindex', action='store_true', help='Build HTML index.html page.')

    parser.add_argument('--htmlhome', default='index.html', type=str, help='Home page file name (use in tandem with --htmlindex).')
    parser.add_argument('--html-raslices', action='store_true',
                        help='Organize HTML pages by RA slice (use in tandem with --htmlindex).')
    parser.add_argument('--htmldir', type=str, help='Output directory for HTML files.')
    
    parser.add_argument('--pixscale', default=0.262, type=float, help='pixel scale (arcsec/pix).')
    parser.add_argument('--region', default='dr11-south', choices=['dr9-north', 'dr11-south'], type=str, help='Region analyze')

    parser.add_argument('--no-unwise', action='store_false', dest='unwise', help='Do not build unWISE coadds or do forced unWISE photometry.')
    parser.add_argument('--no-galex', action='store_false', dest='galex', help='Do not build GALEX coadds or do forced GALEX photometry.')
    parser.add_argument('--no-cleanup', action='store_false', dest='cleanup', help='Do not clean up legacypipe files after coadds.')

    parser.add_argument('--ubercal-sky', action='store_true', help='Build the largest large-galaxy coadds with custom (ubercal) sky-subtraction.')
    parser.add_argument('--force', action='store_true', help='Use with --coadds; ignore previous pickle files.')
    parser.add_argument('--count', action='store_true', help='Count how many objects are left to analyze and then return.')
    parser.add_argument('--debug', action='store_true', help='Log to STDOUT and build debugging plots.')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output.')
    parser.add_argument('--clobber', action='store_true', help='Overwrite existing files.')                                
    parser.add_argument('--mpi', action='store_true', help='Use MPI parallelism')

    parser.add_argument('--build-refcat', action='store_true', help='Build the legacypipe reference catalog.')
    parser.add_argument('--build-catalog', action='store_true', help='Build the final catalog.')
    args = parser.parse_args()

    return args


def _start(galaxy, log=None, seed=None):
    if seed:
        print('Random seed = {}'.format(seed), flush=True)        
    print('Started working on galaxy {} at {}'.format(
        galaxy, time.asctime()), flush=True, file=log)


def _done(galaxy, galaxydir, err, t0, stage, filesuffix=None, log=None):
    if filesuffix is None:
        suffix = ''
    else:
        suffix = '-{}'.format(filesuffix)
    if err == 0:
        print('ERROR: galaxy {}; please check the logfile.'.format(galaxy), flush=True, file=log)
        donefile = os.path.join(galaxydir, '{}{}-{}.isfail'.format(galaxy, suffix, stage))
    else:
        donefile = os.path.join(galaxydir, '{}{}-{}.isdone'.format(galaxy, suffix, stage))
        
    cmd = 'touch {}'.format(donefile)
    subprocess.call(cmd.split())
        
    print('Finished galaxy {} in {:.3f} minutes.'.format(
          galaxy, (time.time() - t0)/60), flush=True, file=log)

    
def call_ellipse(galaxy, galaxydir, data, galaxyinfo=None,
                 pixscale=0.262, mp=1, bands=['g', 'r', 'z'], refband='r',
                 delta_logsma=5, maxsma=None, logsma=True,
                 copy_mw_transmission=False, 
                 verbose=False, debug=False, write_donefile=True,
                 logfile=None, input_ellipse=None, sbthresh=None,
                 apertures=None, clobber=False):
    """Wrapper script to do ellipse-fitting.

    """
    import SGA.ellipse

    # Do not force zcolumn here; it's not always wanted or needed in ellipse.
    #if zcolumn is None:
    #    zcolumn = 'Z_LAMBDA'

    t0 = time.time()
    if debug:
        _start(galaxy)
        err = SGA.ellipse.SGA_ellipse(
            galaxy, galaxydir, data, galaxyinfo=galaxyinfo,
            bands=bands, refband=refband,
            pixscale=pixscale, mp=mp,
            sbthresh=sbthresh, apertures=apertures, input_ellipse=input_ellipse,
            delta_logsma=delta_logsma, maxsma=maxsma, logsma=logsma,
            copy_mw_transmission=copy_mw_transmission,
            verbose=verbose, debug=debug, clobber=clobber)
        if write_donefile:
            _done(galaxy, galaxydir, err, t0, 'ellipse', data['filesuffix'])
    else:
        with open(logfile, 'a') as log:
            with redirect_stdout(log), redirect_stderr(log):
                _start(galaxy, log=log)
                err = SGA.ellipse.SGA_ellipse(
                    galaxy, galaxydir, data, galaxyinfo=galaxyinfo,
                    bands=bands, refband=refband,
                    pixscale=pixscale, mp=mp,
                    sbthresh=sbthresh, apertures=apertures, input_ellipse=input_ellipse,
                    delta_logsma=delta_logsma, maxsma=maxsma, logsma=logsma,
                    copy_mw_transmission=copy_mw_transmission,
                    verbose=verbose, clobber=clobber)
                if write_donefile:
                    _done(galaxy, galaxydir, err, t0, 'ellipse', data['filesuffix'], log=log)

    return err


def call_sky(onegal, galaxy, galaxydir, survey, seed, mp, pixscale,
              verbose, debug, logfile):
    """Wrapper script to do Sersic-fitting.

    """
    import legacyhalos.sky

    t0 = time.time()
    if debug:
        _start(galaxy, seed=seed)
        err = legacyhalos.sky.legacyhalos_sky(onegal, survey=survey, galaxy=galaxy, galaxydir=galaxydir,
                                              mp=mp, pixscale=pixscale, seed=seed,
                                              debug=debug, verbose=verbose, force=force)
        _done(galaxy, err, t0)
    else:
        with open(logfile, 'a') as log:
            with redirect_stdout(log), redirect_stderr(log):
                _start(galaxy, log=log, seed=seed)
                err = legacyhalos.sky.legacyhalos_sky(onegal, survey=survey, galaxy=galaxy, galaxydir=galaxydir,
                                                      mp=mp, pixscale=pixscale, seed=seed,
                                                      debug=debug, verbose=verbose, force=force)
                _done(galaxy, err, t0, log=log)
                
def call_htmlplots(onegal, galaxy, survey, pixscale=0.262, mp=1, 
                   verbose=False, debug=False, clobber=False, ccdqa=False,
                   logfile=None, zcolumn='Z', galaxy_id=None,
                   bands=['g', 'r', 'z'], SBTHRESH=None,
                   datadir=None, htmldir=None, cosmo=None,
                   linear=False, plot_colors=True,
                   galex=False, unwise=False, just_coadds=False, write_donefile=True,
                   barlen=None, barlabel=None, radius_mosaic_arcsec=None,
                   get_galaxy_galaxydir=None, read_multiband=None,
                   qa_multiwavelength_sed=None):
    """Wrapper script to build the pipeline coadds."""
    t0 = time.time()

    if debug:
        _start(galaxy)
        err = SGA.html.make_plots(
            onegal, datadir=datadir, htmldir=htmldir, survey=survey, 
            pixscale=pixscale, zcolumn=zcolumn, galaxy_id=galaxy_id,
            mp=mp, barlen=barlen, barlabel=barlabel,
            radius_mosaic_arcsec=radius_mosaic_arcsec,
            bands=bands, SBTHRESH=SBTHRESH,
            linear=linear, plot_colors=plot_colors,
            maketrends=False, ccdqa=ccdqa,
            clobber=clobber, verbose=verbose, 
            cosmo=cosmo, galex=galex, unwise=unwise, just_coadds=just_coadds,
            get_galaxy_galaxydir=get_galaxy_galaxydir,
            read_multiband=read_multiband,
            qa_multiwavelength_sed=qa_multiwavelength_sed)
        if write_donefile:
            _done(galaxy, survey.output_dir, err, t0, 'html')
    else:
        with open(logfile, 'a') as log:
            with redirect_stdout(log), redirect_stderr(log):
                _start(galaxy, log=log)
                err = SGA.html.make_plots(
                    onegal, datadir=datadir, htmldir=htmldir, survey=survey, 
                    pixscale=pixscale, zcolumn=zcolumn, galaxy_id=galaxy_id,
                    mp=mp, barlen=barlen, barlabel=barlabel,
                    radius_mosaic_arcsec=radius_mosaic_arcsec,
                    bands=bands, SBTHRESH=SBTHRESH,
                    linear=linear, plot_colors=plot_colors,
                    maketrends=False, ccdqa=ccdqa,
                    clobber=clobber, verbose=verbose,
                    cosmo=cosmo, galex=galex, unwise=unwise, just_coadds=just_coadds,
                    get_galaxy_galaxydir=get_galaxy_galaxydir,
                    read_multiband=read_multiband,
                    qa_multiwavelength_sed=qa_multiwavelength_sed)
                if write_donefile:
                    _done(galaxy, survey.output_dir, err, t0, 'html')


def call_custom_coadds(onegal, galaxy, survey, run, radius_mosaic, mp=1,
                       pixscale=0.262, racolumn='RA', deccolumn='DEC', nsigma=None,
                       custom=True,
                       bands=['g', 'r', 'z'], 
                       apodize=False, unwise=True, galex=False, force=False, plots=False,
                       verbose=False, cleanup=True, write_all_pickles=False,
                       #no_subsky=False,
                       subsky_radii=None,
                       #ubercal_sky=False,
                       write_wise_psf=False,
                       just_coadds=False, require_grz=True, 
                       no_gaia=False, no_tycho=False,
                       debug=False, logfile=None):
    """Wrapper script to build custom coadds.

    radius_mosaic in arcsec

    """
    import SGA.coadds
    
    t0 = time.time()
    if debug:
        _start(galaxy)
        err, filesuffix = SGA.coadds.custom_coadds(
            onegal, galaxy=galaxy, survey=survey, 
            radius_mosaic=radius_mosaic, mp=mp, 
            pixscale=pixscale, racolumn=racolumn, deccolumn=deccolumn,
            nsigma=nsigma, custom=custom,
            bands=bands,
            run=run, apodize=apodize, unwise=unwise, galex=galex, force=force, plots=plots,
            verbose=verbose, cleanup=cleanup, write_all_pickles=write_all_pickles,
            write_wise_psf=write_wise_psf,
            #no_subsky=no_subsky,
            subsky_radii=subsky_radii, #ubercal_sky=ubercal_sky,
            just_coadds=just_coadds,
            require_grz=require_grz, no_gaia=no_gaia, no_tycho=no_tycho)
        _done(galaxy, survey.output_dir, err, t0, 'coadds', filesuffix)
    else:
        with open(logfile, 'a') as log:
            with redirect_stdout(log), redirect_stderr(log):
                _start(galaxy, log=log)
                err, filesuffix = SGA.coadds.custom_coadds(
                    onegal, galaxy=galaxy, survey=survey, 
                    radius_mosaic=radius_mosaic, mp=mp, 
                    pixscale=pixscale, racolumn=racolumn, deccolumn=deccolumn,
                    nsigma=nsigma, custom=custom,
                    bands=bands,
                    run=run, apodize=apodize, unwise=unwise, galex=galex, force=force, plots=plots,
                    verbose=verbose, cleanup=cleanup, write_all_pickles=write_all_pickles,
                    write_wise_psf=write_wise_psf,
                    #no_subsky=no_subsky,
                    subsky_radii=subsky_radii, #ubercal_sky=ubercal_sky,
                    just_coadds=just_coadds,
                    require_grz=require_grz, no_gaia=no_gaia, no_tycho=no_tycho,
                    log=log)
                _done(galaxy, survey.output_dir, err, t0, 'coadds', filesuffix, log=log)
