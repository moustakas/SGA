"""
SGA.photo
=========

Utilities for performing simple photometry.

"""
import pdb
import numpy as np


def photo_datamodel(out, ra, dec, diam, ba, pa, bands=['g', 'r', 'i', 'z']):
    out.add_column(radec_to_name(ra, dec, unixsafe=False)[0], name='SGANAME', index=0)
    for col in ['RA', 'DEC']:
        out[f'{col}_PHOT'] = [-99.]
    out['IN_GAIA'] = [False]
    out['NODATA'] = [False]
    out['CENTERMASKED'] = [False]
    out['MGE_FAIL'] = [False]
    out['SEP'] = [np.float32(-99.)]
    out['DIAM_INIT'] = diam[0].astype('f4') # [arcsec]
    out['BA_INIT'] = ba[0].astype('f4')
    out['PA_INIT'] = pa[0].astype('f4') # CCW from y-axis
    for col in ['DIAM', 'BA', 'PA']:
        out[f'{col}_PHOT'] = [np.float32(-99.)]
    for col in ['INIT', 'PHOT']:
        for band in bands:
            out[f'FLUX_{col}_{band.upper()}'] = [np.float32(-99.)]
        for band in bands:
            out[f'FLUX_{col}_ERR_{band.upper()}'] = [np.float32(-99.)]
        for band in bands:
            out[f'GINI_{col}_{band.upper()}'] = [np.float32(-99.)]
        for band in bands:
            out[f'FRACMASK_{col}_{band.upper()}'] = [np.float32(-99.)]
    return out


def _get_photo_filename(args):
    return get_photo_filename(*args)


def get_photo_filename(obj, objname, cutoutdir, photodir, gather_photo=False,
                       overwrite=False, verbose=False):
    raslice = get_raslice(obj['RA'])

    fitsfile = os.path.join(cutoutdir, get_raslice(obj['RA']), f'{objname}.fits')
    jpgfile = os.path.join(cutoutdir, get_raslice(obj['RA']), f'{objname}.jpeg')
    photfile = os.path.join(photodir, raslice, f'{objname}-phot.fits')
    qafile = os.path.join(photodir, raslice, f'{objname}-phot.png')
    nobj = 1

    if gather_photo:
        nobj = len(glob(photfile))
        return fitsfile, jpgfile, photfile, qafile, nobj

    if not os.path.isfile(fitsfile):
        nobj = 0
        log.warning(f'Missing input FITS file {fitsfile}')
    else:
        if overwrite is False:
            if os.path.isfile(photfile) and os.path.isfile(qafile):
                nobj = 0
                if verbose:
                    log.info(f'Skipping existing photometry file {photfile}')

    return fitsfile, jpgfile, photfile, qafile, nobj


def qaplot_photo_one(qafile, jpgfile, out, ra, dec, pixscale, width,
                     diam, ba, pa, xyinit, wimg, wmask, wcs, xyphot=None,
                     xypeak=None, render_jpeg=True):

    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    from SGA.qa import draw_ellipse

    barlen = 15. / pixscale # [pixels]
    barlabel = '15 arcsec'

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, width)
    ax.set_ylim(0, width)
    if render_jpeg:
        jpg = mpimg.imread(jpgfile)
        jpgmask = np.flipud(~wmask)[:, :, np.newaxis]
        #jpgmask = np.flipud(~gaia_mask)[:, :, np.newaxis]
        #jpgmask = np.flipud(apmask_phot)[:, :, np.newaxis]
        im = ax.imshow(jpg * jpgmask, origin='lower', interpolation='nearest')
        ax.invert_yaxis() # JPEG is flipped relative to FITS
    else:
        #ax.imshow(img * apmask_phot, origin='lower', cmap=cmap)
        ax.imshow(np.log(wimg.clip(wimg[xpeak, ypeak]/1e4)) * ~wmask,
                  origin='lower', cmap='inferno', interpolation='nearest')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.margins(0)

    # these two sets of drawings should be identical
    #ap_init.plot(color='red', ls='-', lw=2, ax=ax)
    #for ap_phot in aps_phot:
    #    ap_phot.plot(color='black', ls='-', lw=2, ax=ax)
    draw_ellipse(major_axis_arcsec=diam[0], ba=ba[0], pa=pa[0], x0=xyinit[0],
                 y0=xyinit[1], height_pixels=width, ax=ax, pixscale=pixscale,
                 color='yellow', linestyle='--', draw_majorminor_axes=True,
                 jpeg=render_jpeg)
    if xyphot is not None:
        xphot, yphot = wcs.wcs_world2pix(out['RA_PHOT'], out['DEC_PHOT'], 1)
        draw_ellipse(major_axis_arcsec=out[f'DIAM_PHOT'], ba=out['BA_PHOT'],
                     pa=out['PA_PHOT'], x0=xphot, y0=yphot, height_pixels=width,
                     pixscale=pixscale, color='cyan', linestyle='-', linewidth=2,
                     ax=ax, draw_majorminor_axes=True, jpeg=render_jpeg)

    txt = '\n'.join([out['SGANAME'][0], out['OBJNAME'][0], f'{ra:.7f}, {dec:.6f}'])
    ax.text(0.03, 0.93, txt, transform=ax.transAxes, ha='left', va='center',
            color='white', bbox=dict(boxstyle='round', facecolor='k', alpha=0.5),
            linespacing=1.5, fontsize=10)

    # add the scale bar
    xpos, ypos = 0.07, 0.07
    dx = barlen / wimg.shape[0]
    ax.plot([xpos, xpos+dx], [ypos, ypos], transform=ax.transAxes,
            color='white', lw=2)
    ax.text(xpos + dx/2., ypos+0.02, barlabel, transform=ax.transAxes,
            ha='center', va='center', color='white')

    fig.tight_layout()
    fig.savefig(qafile, bbox_inches=0)#, dpi=200)
    plt.close()
    #log.info(f'Wrote {qafile}')


def _photo_one(args):
    return photo_one(*args)


def photo_one(fitsfile, jpgfile, photfile, qafile, obj, survey,
              bands=['g', 'r', 'i', 'z'], box_arcsec=5.,
              verbose=False, qaplot=True):
    """Perform photometry on a single cutout.

    """
    from astropy.io import fits
    from astropy.wcs import WCS
    from astropy.wcs.utils import proj_plane_pixel_scales as get_pixscale
    from photutils.aperture import EllipticalAperture, CircularAperture
    from photutils.morphology import gini

    from legacypipe.survey import wcs_for_brick, BrickDuck
    from legacypipe.reference import get_reference_sources, get_reference_map
    from SGA.find_galaxy import find_galaxy
    from SGA.io import custom_brickname


    def get_error(ivar):
        error = ivar.copy()
        I = error > 0.
        error[I] = 1. / np.sqrt(error[I])
        return error

    def write_photfile(out, photfile):
        # with MPI, astropy's .write in 7.0.1 hits this bug--
        # https://github.com/astropy/astropy/issues/15350
        #out.write(photfile, overwrite=True)
        fitsio.write(photfile+'.tmp', out.as_array(), clobber=True)
        os.rename(photfile+'.tmp', photfile)
        log.info(f'Wrote {photfile}')


    ra, dec = obj['RA'], obj['DEC']
    objname = radec_to_name(ra, dec, unixsafe=True)[0]

    diam, ba, pa, ref = choose_geometry(Table(obj), mindiam=15.)
    out = photo_datamodel(Table(obj['OBJNAME', 'STARFDIST', 'STARMAG', 'RA', 'DEC']),
                          ra, dec, diam, ba, pa, bands=bands)

    # read the data
    with fits.open(fitsfile) as H:
        if len(H) != 2:
            msg = f'{fitsfile} ({obj["OBJNAME"]}) is missing an inverse variance extension!'
            log.critical(msg)
            raise ValueError(msg)
        hdr = H[0].header
        imgs = H[0].data
        ivars = H[1].data

    # any data?
    if np.all(imgs == 0.):
        log.warning(f'No data for object {out["OBJNAME"][0]} = {out["SGANAME"][0]}')
        out['NODATA'] = True

    hdr['NAXIS'] = 2
    hdr.pop('NAXIS3')

    wcs = WCS(hdr, naxis=2)
    pixscale = get_pixscale(wcs)[0] * 3600. # [arcsec/pixel]
    minsb = 10.**(-0.4*(30.-22.5)) #/ pixscale**2.

    nband, height, width = imgs.shape

    # build the Gaia/Tycho mask
    gaia_mask = np.zeros((height, width), bool) # True=Gaia star(s)
    gaia_mask_faint = np.zeros_like(gaia_mask)

    brickname = f'custom-{custom_brickname(ra, dec)}'
    brick = BrickDuck(ra, dec, brickname)

    targetwcs = wcs_for_brick(brick, W=float(width), H=float(height), pixscale=pixscale)

    refstars = get_reference_sources(survey, targetwcs, pixscale, bands=bands,
                                     tycho_stars=True, gaia_stars=True,
                                     large_galaxies=False, star_clusters=False)
    refstars = refstars[0]
    if len(refstars) > 0:
        # remove Gaia stars within 5 arcsec of the initial coordinates
        m1, m2, _ = match_radec(refstars.ra, refstars.dec, ra, dec, 5./3600., nearest=True)
        if len(m1) > 0:
            out['IN_GAIA'] = True
            refstars.cut(np.delete(np.arange(len(refstars)), m1))

        # refmap contains just MEDIUM and BRIGHT stars
        refmap = get_reference_map(targetwcs, refstars)
        gaia_mask = np.logical_or(gaia_mask, refmap > 0) # True=Gaia star(s)

        # add fainter stars to the mask
        for star in refstars[refstars.in_bounds]:
            if star.radius_pix <= 0.:
                radius_pix = int(5. / pixscale)
            else:
                radius_pix = star.radius_pix
            ap = CircularAperture((star.ibx, star.iby), radius_pix) # note! (ibx,iby) not (iby,ibx)
            #gaia_mask = np.logical_or(gaia_mask, ap.to_mask().to_image((height, width)) != 0.) # object mask=True
            gaia_mask_faint = np.logical_or(gaia_mask_faint, ap.to_mask().to_image((height, width)) != 0.) # object mask=True

    # initial ellipse geometry and aperture
    xyinit = wcs.wcs_world2pix(ra, dec, 1)
    if np.any(np.isnan(xyinit)):
        msg = f'WCS problem analyzing {photfile}'
        log.critical(msg)
        raise ValueError(msg)
    a_init = diam[0] / 2. / pixscale # [pixels]
    b_init = a_init * ba[0]
    theta_init = np.radians(pa[0] - 90.) # CCW from x-axis

    ap_init = EllipticalAperture(xyinit, a=a_init, b=b_init, theta=theta_init)
    apmask_init = ap_init.to_mask().to_image((height, width)) != 0. # object mask=True

    # 5x5 arcsec box centered on initial position
    box_init = EllipticalAperture(xyinit, a=box_arcsec/2./pixscale,
                                  b=box_arcsec/2./pixscale, theta=0.)
    boxmask_init = box_init.to_mask().to_image((height, width)) != 0. # True=object mask

    # generate the ivar-weighted mean image; flag pixels that are
    # masked in all bandpasses
    wivar = np.sum(ivars, axis=0)
    wimg = np.sum(ivars * imgs, axis=0)
    I = wivar > 0.
    wimg[I] /= wivar[I]
    #wmask = np.sum(ivars <= 0., axis=0) == nband
    wmask = np.logical_or(np.sum(ivars <= 0., axis=0) == nband, gaia_mask, gaia_mask_faint)

    # if the center is fully masked, first try dropping the faint-Gaia
    # mask; if still fully masked, write out and move on
    if np.all(wimg[boxmask_init] == 0.) or np.all(wmask[boxmask_init]):
        wmask = np.logical_or(np.sum(ivars <= 0., axis=0) == nband, gaia_mask)
        if np.all(wimg[boxmask_init] == 0.) or np.all(wmask[boxmask_init]):
            wmask = np.sum(ivars <= 0., axis=0) == nband
            if np.all(wimg[boxmask_init] == 0.) or np.all(wmask[boxmask_init]):
                log.warning(f'Fully masked {box_arcsec:.1f}x{box_arcsec:.1f} arcsec ' + \
                            f'center {out["OBJNAME"][0]} = {out["SGANAME"][0]}')
                out['CENTERMASKED'] = True

                write_photfile(out, photfile)
                qaplot_photo_one(qafile, jpgfile, out, ra, dec, pixscale, width,
                                 diam, ba, pa, xyinit, wimg, wmask, wcs)
                return

    # compute the mean geometry

    ## photutils version which does not perform as well as find_galaxy
    #from photutils.morphology import data_properties
    ##from photutils.segmentation import SourceCatalog, SegmentationImage
    ##src = SourceCatalog(img, SegmentationImage(np.ones_like(img, int)), error=error, mask=mask)[0]
    #src = data_properties(img, mask=wmask)
    #xyphot = (src.xcentroid, src.ycentroid)
    #ba_phot = src.ellipticity.value
    #a_phot = src.semimajor_sigma.value # * 1.5
    #b_phot = a_phot * ba_phot
    #pa_phot = (360. - src.orientation.value) % 180 + 90. # CCW from y-axis
    #theta_phot = np.radians(pa_phot - 90.) # CCW from x-axis

    mge_fail = False
    try:
        mge = find_galaxy(wimg * ~wmask, binning=5, level=minsb, quiet=True)
    except:
        mge_fail = True
        #mge = find_galaxy(wimg * ~wmask, binning=5, level=minsb, quiet=False, plot=True)
        #import matplotlib.pyplot as plt
        #plt.clf() ; plt.imshow(wimg * ~wmask, origin='lower') ; plt.savefig('ioannis/tmp/junk2.png')

    # In rare cases, find_galaxy will return invalid parameters, e.g.,
    # CGMW 4-1190. Capture those here and return.
    for param in ('xpeak', 'ypeak', 'xmed', 'ymed', 'majoraxis', 'eps', 'pa', 'theta'):
        if np.isnan(getattr(mge, param)):
            log.warning(f'Problem determing the geometry of {out["OBJNAME"][0]} = {out["SGANAME"][0]}')
            mge_fail = True
            break

    if mge_fail:
        out['MGE_FAIL'] = True
        write_photfile(out, photfile)
        qaplot_photo_one(qafile, jpgfile, out, ra, dec, pixscale, width,
                         diam, ba, pa, xyinit, wimg, wmask, wcs)
        return


    xypeak = (mge.xpeak, mge.ypeak) # not swapped coordinates!
    xyphot = (mge.ymed, mge.xmed)   # swapped coordinates!
    a_phot = mge.majoraxis # * 1.5 # multiplicative factor? hack?? [pixels]
    ba_phot = 1. - mge.eps
    b_phot = a_phot * ba_phot
    pa_phot = mge.pa # CCW from y-axis
    theta_phot = np.radians((360. - mge.theta) % 180.) # convert from CW from x-axis to CCW from x-axis

    out['DIAM_PHOT'] = a_phot * pixscale # [arcsec]
    out['BA_PHOT'] = ba_phot
    out['PA_PHOT'] = pa_phot

    # aperture photometry in photometric ellipse
    ap_phot = EllipticalAperture(xyphot, a=a_phot, b=b_phot, theta=theta_phot)
    apmask_phot = ap_phot.to_mask().to_image((height, width)) != 0.

    ra_phot, dec_phot = wcs.all_pix2world(xyphot[0], xyphot[1], 1)
    out['RA_PHOT'] = ra_phot
    out['DEC_PHOT'] = dec_phot

    # separation between initial and final coordinates
    out['SEP'] = arcsec_between(ra, dec, ra_phot, dec_phot)

    # next loop on each bandpass
    for iband, band in enumerate(bands):
        img = imgs[iband, :, :]

        ivar = ivars[iband, :, :]
        mask = np.logical_or(ivar <= 0., wmask) # True=masked
        #mask = np.logical_or(ivar <= 0., gaia_mask) # True=masked
        error = get_error(ivar)

        # aperture photometry, fraction of masked pixels, and Gini
        # coefficient in the initial ellipse geometry
        flux_init, ferr_init = ap_init.do_photometry(img, error=error, mask=mask)
        fracmask_init = np.sum(mask[apmask_init]) / mask[apmask_init].size
        gini_init = gini(img * apmask_init, mask=mask)

        out[f'FLUX_INIT_{band.upper()}'] = flux_init[0]
        out[f'FLUX_INIT_ERR_{band.upper()}'] = ferr_init[0]
        out[f'FRACMASK_INIT_{band.upper()}'] = fracmask_init
        if np.isfinite(gini_init):
            out[f'GINI_INIT_{band.upper()}'] = gini_init

        # aperture photometry, fraction of masked pixels, and Gini
        # coefficient in the derived geometry
        flux_phot, ferr_phot = ap_phot.do_photometry(img, error=error, mask=mask)
        fracmask_phot = np.sum(mask[apmask_phot]) / mask[apmask_phot].size
        gini_phot = gini(img * apmask_phot, mask=mask)

        #out[f'FLUX_PHOT_{band.upper()}'] = 22.5 - 2.5 * np.log10(flux_phot[0])
        #out[f'FLUX_PHOT_ERR_{band.upper()}'] = ferr_phot[0] / flux_phot[0] / np.log(10.)
        out[f'FLUX_PHOT_{band.upper()}'] = flux_phot[0]
        out[f'FLUX_PHOT_ERR_{band.upper()}'] = ferr_phot[0]
        out[f'FRACMASK_PHOT_{band.upper()}'] = fracmask_phot
        if np.isfinite(gini_phot):
            out[f'GINI_PHOT_{band.upper()}'] = gini_phot

    #print(out.pprint(max_width=-1))
    #print(out[out.colnames[-8:]])
    write_photfile(out, photfile)

    # build QA
    if qaplot:
        qaplot_photo_one(qafile, jpgfile, out, ra, dec, pixscale, width,
                         diam, ba, pa, xyinit, wimg, wmask, wcs, xyphot=xyphot,
                         xypeak=xypeak)

def _read_one_photfile(args):
    return read_one_photfile(*args)


def read_one_photfile(photfile):
    #tt = Table(fitsio.read(photfile))
    #tt.rename_column('EXCEPTION', 'MGE_FAIL')
    #fitsio.write(photfile, tt.as_array(), clobber=True)
    #print(f'Wrote {photfile}')
    #return Table()
    return Table(fitsio.read(photfile))


def gather_photo(cat, mp=1, region='dr9-north', cutoutdir='.', photodir='.',
                 photo_version='v1.0'):

    catdir = os.path.join(sga_dir(), 'parent', 'photo')
    if not os.path.isdir(catdir):
        os.makedirs(catdir, exist_ok=True)

    catfile = os.path.join(catdir, f'parent-photo-{region}-{photo_version}.fits')
    if os.path.isfile(catfile):
        log.warning(f'Existing photo catalog {catfile} must be removed by-hand.')
        return

    _, _, photfiles, _, groups = plan(cat, size=1, photodir=photodir,
                                      mp=mp, gather_photo=True)
    indx = groups[0]

    # single-rank only but read in parallel
    mpargs = [(photfile, ) for photfile in photfiles[indx]]
    if mp > 1:
        with multiprocessing.Pool(mp) as P:
            out = P.map(_read_one_photfile, mpargs)
    else:
        out = [read_one_photfile(*mparg) for mparg in mpargs]

    if len(out) > 0:
        out = vstack(out)
        out.write(catfile, overwrite=True)
        log.info(f'Wrote photometry for {len(out):,d} objects to {catfile}')


def do_photo(cat, comm=None, mp=1, bands=['g', 'r', 'i', 'z'],
             region='dr9-north', cutoutdir='.', photodir='.',
             photo_version='v1.0', overwrite=False, verbose=False):

    """Wrapper to carry out simple photometry on all objects in the
    input catalog.

    'dr11-south'
        # photo-version=v1.0
        I = (primaries['ROW_LVD'] == -99) * (primaries['STARFDIST'] > 1.) * (diam > 0.) * (diam < 10.) # N=4,434

        # photo-version=v1.1
        I = (primaries['ROW_LVD'] == -99) * (primaries['STARFDIST'] > 1.) * (diam >= 10.) * (diam < 12.) # N=434,092

    'dr9-north'
        # photo-version=v1.0
        #I = (primaries['ROW_LVD'] == -99) * (primaries['STARFDIST'] > 1.) * (diam > 0.) * (diam < 11.) # N=94,229

        # photo-version=v1.1
        I = (primaries['ROW_LVD'] == -99) * (primaries['STARFDIST'] > 1.) * (diam >= 11.) * (diam < 15.) # N=212,676    

    """
    if comm is None:
        rank, size = 0, 1
    else:
        rank, size = comm.rank, comm.size

    if rank == 0:
        t0 = time.time()
        fitsfiles, jpgfiles, photfiles, qafiles, groups = plan(
            cat, size=size, cutoutdir=cutoutdir, photodir=photodir,
            overwrite=overwrite, mp=mp, verbose=verbose, photo=True)
        log.info(f'Planning took {time.time() - t0:.2f} sec')
        #groups = np.array_split(range(len(cat)), size) # unweighted distribution
    else:
        fitsfiles, jpgfiles, photfiles, qafiles, groups = [], [], [], [], []

    if comm:
        fitsfiles = comm.bcast(fitsfiles, root=0)
        jpgfiles = comm.bcast(jpgfiles, root=0)
        photfiles = comm.bcast(photfiles, root=0)
        qafiles = comm.bcast(qafiles, root=0)
        groups = comm.bcast(groups, root=0)
    sys.stdout.flush()

    # all done
    if len(photfiles) == 0 or len(np.hstack(photfiles)) == 0:
        return

    assert(len(groups) == size)

    log.info(f'Rank {rank} started at {time.asctime()}')
    sys.stdout.flush()

    indx = groups[rank]
    if len(indx) == 0:
        return

    if rank == 0:
        from legacypipe.runs import get_survey
        from SGA.coadds import RUNS
        from SGA.io import set_legacysurvey_dir

        set_legacysurvey_dir(region)
        survey = get_survey(RUNS[region])
    else:
        survey = None

    if comm:
        survey = comm.bcast(survey, root=0)

    mpargs = [(fitsfiles[indx[iobj]], jpgfiles[indx[iobj]], photfiles[indx[iobj]],
               qafiles[indx[iobj]], cat[indx[iobj]], survey, bands) for iobj in range(len(indx))]
    if mp > 1:
        with multiprocessing.Pool(mp) as P:
            P.map(_photo_one, mpargs)
    else:
        [photo_one(*mparg) for mparg in mpargs]

    sys.stdout.flush()

    #if comm is not None:
    #    comm.barrier()

    if rank == 0:
        log.info(f'All done at {time.asctime()}')
