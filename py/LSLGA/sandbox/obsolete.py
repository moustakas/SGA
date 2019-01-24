def mgefit_multiband(galaxy, galaxydir, data, debug=False, nowrite=False,
                     noellipsefit=True, verbose=False):
    """MGE-fit the multiband data.

    See http://www-astro.physics.ox.ac.uk/~mxc/software/#mge

    """
    from mge.find_galaxy import find_galaxy
    from mge.sectors_photometry import sectors_photometry
    from mge.mge_fit_sectors import mge_fit_sectors as fit_sectors
    #from mge.mge_print_contours import mge_print_contours as print_contours

    band, refband, pixscale = data['band'], data['refband'], data['pixscale']

    # Get the geometry of the galaxy in the reference band.
    if verbose:
        print('Finding the galaxy in the reference {}-band image.'.format(refband))

    mgegalaxy = find_galaxy(data[refband], nblob=1, binning=3,
                            plot=debug, quiet=not verbose)
    if debug:
        #plt.show()
        pass
    
    #galaxy.xmed -= 1
    #galaxy.ymed -= 1
    #galaxy.xpeak -= 1
    #galaxy.ypeak -= 1
    
    mgefit = dict()
    for key in ('eps', 'majoraxis', 'pa', 'theta',
                'xmed', 'ymed', 'xpeak', 'ypeak'):
        mgefit[key] = getattr(mgegalaxy, key)

    if not noellipsefit:
        t0 = time.time()
        for filt in band:
            if verbose:
                print('Running MGE on the {}-band image.'.format(filt))

            mgephot = sectors_photometry(data[filt], mgegalaxy.eps, mgegalaxy.theta, mgegalaxy.xmed,
                                         mgegalaxy.ymed, n_sectors=11, minlevel=0, plot=debug,
                                         mask=data['{}_mask'.format(filt)])
            if debug:
                #plt.show()
                pass

            mgefit[filt] = fit_sectors(mgephot.radius, mgephot.angle, mgephot.counts,
                                       mgegalaxy.eps, ngauss=None, negative=False,
                                       sigmaPSF=0, normPSF=1, scale=pixscale,
                                       quiet=not debug, outer_slope=4, bulge_disk=False,
                                       plot=debug)
            if debug:
                pass
                #plt.show()

            #_ = print_contours(data[refband], mgegalaxy.pa, mgegalaxy.xpeak, mgegalaxy.ypeak, pp.sol, 
            #                   binning=2, normpsf=1, magrange=6, mask=None, 
            #                   scale=pixscale, sigmapsf=0)

        if verbose:
            print('Time = {:.3f} sec'.format( (time.time() - t0) / 1))

    if not nowrite:
        LSLGA.io.write_mgefit(galaxy, galaxydir, mgefit, band=refband, verbose=verbose)

    return mgefit
    
